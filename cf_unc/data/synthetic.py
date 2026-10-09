"""StochasticIDM synthetic car-following data (docs/design.md, section 3.2).

    a_t = clip( IDM(s_t, dv_t, v_t; theta_driver) + sigma(s_t) eps_t, A_MIN, A_MAX ),   eps_t ~ N(0, 1) iid

``sigma(s) = noise_sigma`` ("const") or ``noise_sigma (1 + noise_kappa exp(-s / noise_s_scale))`` ("gap").
The noise is white per step on the acceleration [m/s^2]; the diffusion coefficient of the equivalent
continuous-time (Laval-type) model is ``sigma sqrt(DT)``. Drivers get lognormal parameter multipliers
(unit median, CV ``heterogeneity_cv``). Leaders track a regime-switching speed target (``LeaderConfig``).
Integration uses the scheme of design.md, section 1 (shared with ``cf_unc.physics.idm``). All events of
a data set are simulated in parallel: numpy over events, python loop over time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from cf_unc.constants import A_MAX, A_MIN, DT
from cf_unc.data.schema import Event, EventSet
from cf_unc.physics.idm import (
    PARAM_NAMES,
    IDMParams,
    equilibrium_gap,
    idm_accel,
    leader_positions,
    speed_step,
)

NOISE_KINDS = ("const", "gap")
GENERATOR_NAME = "stochastic_idm"


def _plain(obj: Any) -> Any:
    """Mappings / sequences (also OmegaConf containers) -> plain dicts / lists, recursively."""
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return [_plain(v) for v in obj]
    return obj


def _check_keys(cls: type, d: Mapping[str, Any]) -> None:
    unknown = sorted(set(d) - {f.name for f in fields(cls)})
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {unknown}")


def _default_idm() -> IDMParams:
    return IDMParams(v0=30.0, T=1.5, s0=2.0, a_max=1.2, b=1.8)


@dataclass
class LeaderConfig:
    """Regime-switching leader.

    Speed-target levels are drawn from a mixture of uniform ranges [m/s] (congested / medium / free,
    component weights ``level_weights``); holding times are ``hold_min_s + Exp(hold_mean_s - hold_min_s)``
    [s] (mean ``hold_mean_s``). The leader starts at its first level and tracks the target with
    ``a_L = clip(gain (v_target - v_L), accel_min, accel_max) + eta``, where ``eta`` is an
    Ornstein-Uhlenbeck acceleration noise (stationary std ``ou_sigma`` [m/s^2], correlation time
    ``ou_tau_s`` [s]); ``v_L >= v_min`` [m/s] so that the standstill clip never distorts the noise law.
    """

    level_ranges: tuple[tuple[float, float], ...] = ((2.0, 8.0), (8.0, 20.0), (20.0, 30.0))
    level_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    hold_mean_s: float = 12.0
    hold_min_s: float = 4.0
    gain: float = 0.5
    accel_min: float = -3.0
    accel_max: float = 2.0
    ou_sigma: float = 0.3
    ou_tau_s: float = 2.0
    v_min: float = 2.0

    def __post_init__(self) -> None:
        self.level_ranges = tuple((float(lo), float(hi)) for lo, hi in self.level_ranges)
        self.level_weights = tuple(float(w) for w in self.level_weights)
        for name in ("hold_mean_s", "hold_min_s", "gain", "accel_min", "accel_max", "ou_sigma", "ou_tau_s", "v_min"):
            setattr(self, name, float(getattr(self, name)))
        if not self.level_ranges or len(self.level_weights) != len(self.level_ranges):
            raise ValueError("level_ranges and level_weights must be non-empty and of equal length")
        if any(not 0.0 <= lo < hi for lo, hi in self.level_ranges):
            raise ValueError(f"invalid level ranges {self.level_ranges}")
        if any(w < 0.0 for w in self.level_weights) or sum(self.level_weights) <= 0.0:
            raise ValueError(f"invalid level weights {self.level_weights}")
        if not 0.0 < self.hold_min_s <= self.hold_mean_s:
            raise ValueError("need 0 < hold_min_s <= hold_mean_s")
        if self.gain <= 0.0 or self.ou_tau_s <= 0.0 or self.ou_sigma < 0.0 or self.v_min < 0.0:
            raise ValueError("need gain > 0, ou_tau_s > 0, ou_sigma >= 0, v_min >= 0")
        if not self.accel_min < 0.0 < self.accel_max:
            raise ValueError("need accel_min < 0 < accel_max")

    def to_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["level_ranges"] = [list(r) for r in self.level_ranges]
        out["level_weights"] = list(self.level_weights)
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "LeaderConfig":
        """Missing keys keep their defaults; unknown keys are an error."""
        d = _plain(d or {})
        _check_keys(cls, d)
        return cls(**d)


@dataclass
class SyntheticConfig:
    """StochasticIDM data set. ``idm`` holds the population (median) parameters.

    Initial state: ``v(0) = v_L(0) + N(0, init_speed_std^2)`` clipped at 0, ``s(0) = s_eq(v(0)) U(init_gap_factor)``
    clipped to ``init_gap_range`` [m]. Events with ``min(s) < min_gap`` [m] are redrawn (at most
    ``max_regen_rounds`` times).
    """

    n_drivers: int = 200
    events_per_driver: int = 4
    duration_s: float = 40.0
    idm: IDMParams = field(default_factory=_default_idm)
    heterogeneity_cv: float = 0.0
    noise_kind: str = "const"
    noise_sigma: float = 0.4
    noise_kappa: float = 1.5
    noise_s_scale: float = 15.0
    min_gap: float = 0.5
    seed: int = 0
    dataset: str = "synthetic"
    site: str = "sim"
    leader: LeaderConfig = field(default_factory=LeaderConfig)
    init_speed_std: float = 1.5
    init_gap_factor: tuple[float, float] = (0.6, 2.5)
    init_gap_range: tuple[float, float] = (3.0, 150.0)
    max_regen_rounds: int = 100

    def __post_init__(self) -> None:
        if isinstance(self.idm, Mapping):  # nested dicts: missing keys keep their defaults
            self.idm = IDMParams.from_dict({**_default_idm().to_dict(), **_plain(self.idm)})
        if isinstance(self.leader, Mapping):
            self.leader = LeaderConfig.from_dict(self.leader)
        for name in ("n_drivers", "events_per_driver", "seed", "max_regen_rounds"):
            setattr(self, name, int(getattr(self, name)))
        for name in (
            "duration_s",
            "heterogeneity_cv",
            "noise_sigma",
            "noise_kappa",
            "noise_s_scale",
            "min_gap",
            "init_speed_std",
        ):
            setattr(self, name, float(getattr(self, name)))
        self.init_gap_factor = tuple(float(x) for x in self.init_gap_factor)
        self.init_gap_range = tuple(float(x) for x in self.init_gap_range)
        self.dataset, self.site = str(self.dataset), str(self.site)
        if not isinstance(self.idm, IDMParams) or not all(isinstance(getattr(self.idm, n), float) for n in PARAM_NAMES):
            raise ValueError("idm must hold scalar population parameters")
        if self.noise_kind not in NOISE_KINDS:
            raise ValueError(f"noise_kind must be one of {NOISE_KINDS}, got {self.noise_kind!r}")
        if self.n_drivers < 1 or self.events_per_driver < 1 or self.duration_s < DT:
            raise ValueError("need n_drivers >= 1, events_per_driver >= 1, duration_s >= DT")
        if min(self.heterogeneity_cv, self.noise_sigma, self.noise_kappa, self.init_speed_std) < 0.0:
            raise ValueError("heterogeneity_cv, noise_sigma, noise_kappa and init_speed_std must be >= 0")
        if self.noise_s_scale <= 0.0 or self.max_regen_rounds < 0:
            raise ValueError("need noise_s_scale > 0 and max_regen_rounds >= 0")
        if len(self.init_gap_factor) != 2 or not 0.0 < self.init_gap_factor[0] <= self.init_gap_factor[1]:
            raise ValueError(f"invalid init_gap_factor {self.init_gap_factor}")
        if len(self.init_gap_range) != 2 or not self.min_gap < self.init_gap_range[0] <= self.init_gap_range[1]:
            raise ValueError(f"invalid init_gap_range {self.init_gap_range} (must exceed min_gap)")

    @property
    def n_samples(self) -> int:
        """Samples per event (t = 0, DT, ..., duration_s)."""
        return int(round(self.duration_s / DT)) + 1

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict (tuples as lists), the inverse of ``from_dict``."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (IDMParams, LeaderConfig)):
                value = value.to_dict()
            elif isinstance(value, tuple):
                value = list(value)
            out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "SyntheticConfig":
        """Build from a (possibly nested, possibly OmegaConf) mapping; missing keys keep their defaults."""
        d = _plain(d or {})
        _check_keys(cls, d)
        return cls(**d)


def noise_std(s: Any, cfg: SyntheticConfig) -> np.ndarray:
    """Std sigma(s) [m/s^2] of the acceleration noise at gap ``s`` [m]."""
    s = np.asarray(s, dtype=np.float64)
    if cfg.noise_kind == "const":
        return np.full(s.shape, cfg.noise_sigma)
    return cfg.noise_sigma * (1.0 + cfg.noise_kappa * np.exp(-s / cfg.noise_s_scale))


def driver_parameters(cfg: SyntheticConfig, rngs: Sequence[np.random.Generator]) -> np.ndarray:
    """(D, 5) per-driver [v0, T, s0, a_max, b]: population values times lognormal multipliers.

    The multipliers have unit median and coefficient of variation ``heterogeneity_cv``; the normal draws
    are made even for cv = 0 so that the remaining random streams do not depend on the cv.
    """
    z = np.stack([g.standard_normal(len(PARAM_NAMES)) for g in rngs])
    sd = math.sqrt(math.log1p(cfg.heterogeneity_cv**2))
    return cfg.idm.as_vector()[None, :] * np.exp(sd * z)


def _draw_randoms(g: np.random.Generator, m: int, cfg: SyntheticConfig) -> dict[str, np.ndarray]:
    """All random inputs of ``m`` events of one driver (fixed draw order)."""
    lead, n = cfg.leader, cfg.n_samples
    n_regimes = int(math.ceil(cfg.duration_s / lead.hold_min_s)) + 1
    weights = np.asarray(lead.level_weights) / sum(lead.level_weights)
    lo, hi = np.asarray(lead.level_ranges).T
    comp = g.choice(len(weights), size=(m, n_regimes), p=weights)
    levels = lo[comp] + g.random((m, n_regimes)) * (hi - lo)[comp]
    holds = lead.hold_min_s + g.exponential(lead.hold_mean_s - lead.hold_min_s, size=(m, n_regimes))
    return {
        "levels": levels,  # (m, K) speed targets [m/s]
        "holds": holds,  # (m, K) holding times [s]
        "v_noise": g.standard_normal(m),  # initial follower speed offset (standardised)
        "gap_u": g.random(m),  # initial gap factor (uniform)
        "ou_xi": g.standard_normal((m, n)),  # innovations of the leader OU noise
        "eps": g.standard_normal((m, n)),  # follower acceleration noise
    }


def _simulate(r: dict[str, np.ndarray], theta: np.ndarray, cfg: SyntheticConfig) -> dict[str, np.ndarray]:
    """Simulate m events in parallel from their random inputs ``r`` and parameters ``theta`` (m, 5).

    Returns (m, n) float64 arrays s, dv, v, a, v_lead, x_lead, x_follower, a_det, sigma.
    """
    lead, n = cfg.leader, cfg.n_samples
    m = theta.shape[0]
    # leader: piecewise-constant target, first-order tracking plus OU acceleration noise, v >= v_min
    t_grid = np.arange(n) * DT
    ends = np.cumsum(r["holds"], axis=1)
    seg = np.zeros((m, n), dtype=np.intp)
    for k in range(ends.shape[1] - 1):
        seg += t_grid[None, :] >= ends[:, k : k + 1]
    target = np.ascontiguousarray(np.take_along_axis(r["levels"], seg, axis=1).T)  # (n, m)
    xi = np.ascontiguousarray(r["ou_xi"].T)
    phi = math.exp(-DT / lead.ou_tau_s)
    innov = lead.ou_sigma * math.sqrt(1.0 - phi * phi)
    vl = np.empty((n, m))
    vl[0] = np.maximum(r["levels"][:, 0], lead.v_min)
    eta = lead.ou_sigma * xi[0]
    for t in range(n - 1):
        a_lead = np.clip(lead.gain * (target[t] - vl[t]), lead.accel_min, lead.accel_max) + eta
        vl[t + 1] = np.maximum(vl[t] + a_lead * DT, lead.v_min)
        eta = phi * eta + innov * xi[t + 1]

    # follower initial state
    p = IDMParams.from_vector(theta, delta=cfg.idm.delta)  # fields (m,)
    v_init = np.maximum(vl[0] + cfg.init_speed_std * r["v_noise"], 0.0)
    f_lo, f_hi = cfg.init_gap_factor
    s_init = np.clip(equilibrium_gap(v_init, p) * (f_lo + (f_hi - f_lo) * r["gap_u"]), *cfg.init_gap_range)
    xl = np.ascontiguousarray(leader_positions(s_init, vl.T, DT).T)  # (n, m), xl[0] = s_init

    # follower: same arithmetic as physics.idm.simulate_follower plus the noise term
    eps = np.ascontiguousarray(r["eps"].T)
    v, x, a, a_det, sig = (np.empty((n, m)) for _ in range(5))
    v[0], x[0] = v_init, 0.0
    for t in range(n):
        s_t = xl[t] - x[t]
        raw = idm_accel(s_t, v[t] - vl[t], v[t], p, clip=False)
        a_det[t] = np.clip(raw, A_MIN, A_MAX)
        sig[t] = noise_std(s_t, cfg)
        a_cmd = np.clip(raw + sig[t] * eps[t], A_MIN, A_MAX)
        a[t], v_next = speed_step(v[t], a_cmd, DT)
        if t + 1 < n:
            v[t + 1] = v_next
            x[t + 1] = x[t] + v_next * DT
    out = {"s": xl - x, "dv": v - vl, "v": v, "a": a, "v_lead": vl, "x_lead": xl, "x_follower": x}
    out.update(a_det=a_det, sigma=sig)
    return {key: value.T for key, value in out.items()}


def generate(cfg: SyntheticConfig | Mapping[str, Any] | None = None) -> tuple[EventSet, dict[str, dict[str, Any]]]:
    """Generate a StochasticIDM data set.

    Returns the EventSet (``follower_id = drv<k>``, ``leader_id = lead<k>_<j>``,
    ``event_id = <dataset>_<k>_<j>``) and ``truth[event_id] = {"a_det": (n,) noise-free IDM acceleration
    (clipped to [A_MIN, A_MAX]) [m/s^2], "sigma_true": (n,) noise std [m/s^2], "idm": driver parameters}``.
    Reproducible from ``cfg.seed``: one child generator per driver (``SeedSequence.spawn``).
    """
    if not isinstance(cfg, SyntheticConfig):
        cfg = SyntheticConfig.from_dict(cfg)
    n, n_drv, per = cfg.n_samples, cfg.n_drivers, cfg.events_per_driver
    n_ev = n_drv * per
    rngs = [np.random.default_rng(child) for child in np.random.SeedSequence(cfg.seed).spawn(n_drv)]
    theta_drv = driver_parameters(cfg, rngs)  # (D, 5)
    theta = np.repeat(theta_drv, per, axis=0)  # (E, 5)
    driver_of = np.repeat(np.arange(n_drv), per)
    parts = [_draw_randoms(g, per, cfg) for g in rngs]
    draws = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}

    event_keys = ("s", "dv", "v", "a", "v_lead", "x_lead", "x_follower")
    keys = (*event_keys, "a_det", "sigma")
    res = {key: np.empty((n_ev, n)) for key in keys}
    regen = np.zeros(n_ev, dtype=np.int64)
    pending = np.arange(n_ev)
    for round_ in range(cfg.max_regen_rounds + 1):
        sim = _simulate({key: value[pending] for key, value in draws.items()}, theta[pending], cfg)
        ok = np.all(np.isfinite(sim["s"]), axis=1) & (np.min(sim["s"], axis=1) >= cfg.min_gap)
        for key in keys:
            res[key][pending[ok]] = sim[key][ok]
        regen[pending[ok]] = round_
        pending = pending[~ok]
        if pending.size == 0:
            break
        for d in np.unique(driver_of[pending]):  # fresh inputs from the driver's own stream
            rows = pending[driver_of[pending] == d]
            fresh = _draw_randoms(rngs[d], rows.size, cfg)
            for key in draws:
                draws[key][rows] = fresh[key]
    else:
        raise RuntimeError(f"{pending.size} events still violate min_gap after {cfg.max_regen_rounds} redraws")

    t = np.arange(n) * DT
    idm_dicts = [IDMParams.from_vector(theta_drv[k], delta=cfg.idm.delta).to_dict() for k in range(n_drv)]
    events, truth = [], {}
    for e in range(n_ev):
        k, j = divmod(e, per)
        event_id = f"{cfg.dataset}_{k:05d}_{j:02d}"
        meta = {"generator": GENERATOR_NAME, "driver": k, "episode": j, "regen_rounds": int(regen[e])}
        events.append(
            Event(
                event_id=event_id,
                dataset=cfg.dataset,
                site=cfg.site,
                follower_id=f"drv{k:05d}",
                leader_id=f"lead{k:05d}_{j}",
                t=t.copy(),
                **{key: res[key][e].copy() for key in event_keys},
                meta={**meta, "idm": dict(idm_dicts[k])},
            )
        )
        truth[event_id] = {
            "a_det": res["a_det"][e].copy(),
            "sigma_true": res["sigma"][e].copy(),
            "idm": dict(idm_dicts[k]),
        }
    return EventSet(events), truth
