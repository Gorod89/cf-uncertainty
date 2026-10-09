"""Platoon simulation of experiment E3 (docs/design.md, section 13.2): a cascade of closed-loop rollouts.

Time line of a profile with ``n`` leader samples ``u[0 .. n-1]`` (10 Hz), rollout steps ``k = 0 .. K``::

    history (3 s)       every vehicle at speed u[0], gap s_init, dv = 0           (the model input before step 0)
    burn-in  k <= B     leader at u[0]                                             B = burn_in_s / DT = 300
    profile  k = B + j  leader at u[j], j = 0 .. n - 1
    tail     k > B+n-1  leader at u[n-1]                                           T = tail_s / DT = 400 steps
    K >= B + n - 1 + T  (a batch of profiles shares one K; the steps after a profile's own end change nothing
                         before it - rollouts are causal - except for the 5 s look-ahead of plan samplers
                         (DiffusionCF), which near the end see the continued leader instead of a padded one)

Initial state of every follower: speed u[0], gap ``s_init`` (the caller passes the equilibrium gap of the
closed-loop IDM, capped). Leader rear bumper ``x_L[0] = 0``, ``x_L[k+1] = x_L[k] + v_L[k+1] DT`` (scheme of section
1); vehicle i starts at ``-i s_init``. Vehicle 1 is rolled out behind the leader, vehicle i >= 2 behind the simulated
paths of vehicle i - 1 of the same rollout (``rollout_events(..., leader_override=(E, R, K + 1), keep_paths=True)``).
The E profiles of a call are one random-stream block per vehicle (one model call per step for all of them; a
model call costs milliseconds even for a few trajectories): vehicle i draws from its own stream seeded with
``seeds[i - 1]``, so draws of different vehicles are independent and a ``per_trajectory`` sampler state is drawn
per vehicle and rollout; the draws of a profile depend on the seeds and on the profiles simulated with it. Net gaps
are used throughout, so vehicle lengths do not enter (the front bumper of vehicle i - 1 is the leader position of
vehicle i).

Statistics per profile e, vehicle i and rollout r (speeds of the rollout grid):

    response window  k = B .. B + n_e - 1 + T   amplitude max(v) - min(v), sd = std(v) (ddof = 0)   design.md 13.3
    profile window   k = B .. B + n_e - 1       the same over the window of the measured data (sensitivity)
    min_gap          smallest gap over k = 1 .. K_e = B + n_e - 1 + T (burn-in included)
    first_collision  first k in 1 .. K_e with gap <= 0, -1 without collision
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
import torch

from cf_unc.constants import DT, HIST_LEN
from cf_unc.data.schema import Event
from cf_unc.models.base import CFModel
from cf_unc.physics.idm import IDMParams, equilibrium_gap
from cf_unc.rollout.engine import RolloutConfig, rollout_events

__all__ = [
    "PlatoonConfig",
    "PlatoonResult",
    "initial_gap",
    "leader_timeline",
    "simulate_platoon",
    "steady_event",
    "time_line_steps",
]


@dataclass(frozen=True)
class PlatoonConfig:
    """Settings of the platoon simulation (times in s, gaps in m)."""

    n_followers: int = 20
    n_rollouts: int = 100
    n_rollouts_sampled: int = 32  # plan samplers (DiffusionCF)
    history_s: float = HIST_LEN * DT
    burn_in_s: float = 30.0
    tail_s: float = 40.0
    max_batch: int = 100_000
    seed: int = 0
    gap_cap: float = 120.0
    v_max: float | None = None  # optional cap of the follower speed [m/s] (sensitivity; None = design.md 13)

    def __post_init__(self) -> None:
        if abs(float(self.history_s) - HIST_LEN * DT) > 1e-9:
            raise ValueError(f"history_s must equal HIST_LEN * DT = {HIST_LEN * DT} s (the model input), got {self.history_s}")
        if int(self.n_followers) < 1 or int(self.n_rollouts) < 1 or int(self.n_rollouts_sampled) < 1:
            raise ValueError("n_followers, n_rollouts and n_rollouts_sampled must be >= 1")
        if self.burn_in_s < 0 or self.tail_s < 0 or not self.gap_cap > 0:
            raise ValueError("burn_in_s and tail_s must be >= 0, gap_cap > 0")
        if self.v_max is not None and not float(self.v_max) > 0:
            raise ValueError(f"v_max must be None or positive, got {self.v_max!r}")

    @property
    def burn_in(self) -> int:
        """B, burn-in steps."""
        return int(round(self.burn_in_s / DT))

    @property
    def tail(self) -> int:
        """T, tail steps."""
        return int(round(self.tail_s / DT))


def time_line_steps(n_profile: int, cfg: PlatoonConfig) -> int:
    """K_e = B + n - 1 + T, the last step of the response window of a profile of n samples."""
    return cfg.burn_in + int(n_profile) - 1 + cfg.tail


def initial_gap(speed: float | np.ndarray, idm: IDMParams, cap: float) -> np.ndarray:
    """min(IDM equilibrium gap at ``speed`` [m/s], ``cap``) [m] (the equilibrium gap is infinite at or above v0)."""
    return np.minimum(np.asarray(equilibrium_gap(np.asarray(speed, dtype=np.float64), idm), dtype=np.float64), cap)


def leader_timeline(speed: np.ndarray, burn_in: int, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Leader speed [m/s] and rear-bumper position [m] at steps 0 .. K (``n_steps`` = K), each (K + 1,).

    ``v[k] = u[0]`` for k <= B, ``u[k - B]`` on the profile, ``u[n - 1]`` afterwards; ``x[0] = 0``,
    ``x[k + 1] = x[k] + v[k + 1] DT``.
    """
    u = np.asarray(speed, dtype=np.float64)
    if u.ndim != 1 or u.size < 1 or not np.all(np.isfinite(u)):
        raise ValueError("a leader profile must be a non-empty finite 1-D array")
    if n_steps < burn_in + u.size - 1:
        raise ValueError(f"K = {n_steps} steps cannot hold a burn-in of {burn_in} steps and {u.size} profile samples")
    k = np.arange(n_steps + 1)
    v = u[np.clip(k - burn_in, 0, u.size - 1)]
    inc = v * DT
    inc[0] = 0.0
    return v, np.cumsum(inc)


def steady_event(event_id: str, speed: float, gap: float, x_front: float, length: int) -> Event:
    """Event of ``length`` samples in steady following at ``speed`` [m/s] and ``gap`` [m]; the follower front bumper
    is at ``x_front`` [m] at the last history sample ``t0 = HIST_LEN - 1`` (only the history and t0 are used)."""
    j = np.arange(length, dtype=np.float64)
    x_f = x_front + speed * DT * (j - (HIST_LEN - 1))
    v = np.full(length, float(speed))
    return Event(
        event_id=event_id, dataset="platoon", site="platoon", follower_id=f"{event_id}/follower",
        leader_id=f"{event_id}/leader", t=j * DT, s=np.full(length, float(gap)), dv=np.zeros(length), v=v,
        a=np.zeros(length), v_lead=v.copy(), x_lead=x_f + gap, x_follower=x_f,
    )  # fmt: skip


@dataclass
class PlatoonResult:
    """Output of :func:`simulate_platoon` for E profiles, V followers and R rollouts (module docstring).

    amplitude, sd                    (E, V, R) response window [m/s]
    amplitude_profile, sd_profile    (E, V, R) profile window [m/s]
    min_gap                          (E, V, R) [m];  first_collision (E, V, R) int64 (-1: none)
    leader_amplitude, leader_sd      (E,) response window;  leader_sd_profile (E,) profile window
    mean_speed                       (E, V, K + 1) float32 mean speed over the rollouts [m/s] (None if not kept)
    paths                            {"s", "v", "x"}: (E, V, R, K + 1) gap, speed, front-bumper position (keep_paths)
    n_profile                        (E,) profile samples;  n_steps K;  burn_in B;  tail T
    """

    amplitude: np.ndarray
    sd: np.ndarray
    amplitude_profile: np.ndarray
    sd_profile: np.ndarray
    min_gap: np.ndarray
    first_collision: np.ndarray
    leader_amplitude: np.ndarray
    leader_sd: np.ndarray
    leader_sd_profile: np.ndarray
    n_profile: np.ndarray
    n_steps: int
    burn_in: int
    tail: int
    s_init: np.ndarray
    mean_speed: np.ndarray | None = None
    paths: dict[str, np.ndarray] | None = None
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n_vehicles(self) -> int:
        return int(self.amplitude.shape[1])


def _window_stats(v: np.ndarray, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude and standard deviation (ddof = 0) over steps ``start .. stop - 1`` of paths (..., K + 1)."""
    w = v[..., start:stop]
    return w.max(axis=-1) - w.min(axis=-1), w.std(axis=-1)


def simulate_platoon(
    model: CFModel,
    profiles: Sequence[np.ndarray],
    s_init: Sequence[float] | np.ndarray,
    seeds: np.ndarray,
    cfg: PlatoonConfig = PlatoonConfig(),
    *,
    n_rollouts: int | None = None,
    hold_steps: int = 1,
    sampler_mode: str = "per_step",
    n_steps: int | None = None,
    device: str | torch.device | None = None,
    keep_mean_speed: bool = True,
    keep_paths: bool = False,
    ids: Sequence[str] | None = None,
) -> PlatoonResult:
    """Cascade of ``cfg.n_followers`` followers behind every leader profile (module docstring).

    profiles      E leader speed profiles (n_e,) [m/s], 10 Hz
    s_init        (E,) initial gap of every follower [m]
    seeds         (V,) non-negative int stream seeds (< 2**63), one per vehicle (shared by the E profiles)
    n_rollouts    R (default ``cfg.n_rollouts``);  hold_steps / sampler_mode: see ``RolloutConfig``
    n_steps       K (default: the smallest K that holds every profile); must be >= B + n_e - 1 + T for all e
    keep_paths    also return every path (memory: 3 x E x V x R x (K + 1) float64; for tests and diagnostics)
    """
    t_start = time.perf_counter()
    n_e, n_v = len(profiles), int(cfg.n_followers)
    n_r = int(cfg.n_rollouts if n_rollouts is None else n_rollouts)
    if n_e == 0:
        raise ValueError("no profiles to simulate")
    lengths = np.array([np.asarray(p).size for p in profiles], dtype=np.int64)
    k_e = np.array([time_line_steps(n, cfg) for n in lengths], dtype=np.int64)
    n_k = int(k_e.max()) if n_steps is None else int(n_steps)
    if n_k < int(k_e.max()):
        raise ValueError(f"n_steps = {n_k} is shorter than the longest time line ({int(k_e.max())} steps)")
    gaps = np.asarray(s_init, dtype=np.float64).reshape(n_e)
    if not np.all(np.isfinite(gaps) & (gaps > 0)):
        raise ValueError("initial gaps must be finite and positive")
    seeds = [int(x) for x in np.asarray(seeds, dtype=object).ravel()]
    if len(seeds) != n_v or any(x < 0 or x >= 2**63 for x in seeds):
        raise ValueError(f"need {n_v} stream seeds in [0, 2**63), got {len(seeds)}")
    names = [f"profile{e}" for e in range(n_e)] if ids is None else [str(i) for i in ids]
    burn, tail = cfg.burn_in, cfg.tail

    lead = [leader_timeline(p, burn, n_k) for p in profiles]
    lead_v = np.stack([v for v, _ in lead])  # (E, K + 1)
    lead_x = np.stack([x for _, x in lead])
    u0 = lead_v[:, 0]
    if n_e * n_r > int(cfg.max_batch):
        raise ValueError(f"{n_e} profiles x {n_r} rollouts exceed max_batch = {cfg.max_batch}: simulate fewer profiles at once")
    base = RolloutConfig(
        n_rollouts=n_r, horizon_s=n_k * DT, horizons_s=(n_k * DT,), hold_steps=int(hold_steps), sampler_mode=sampler_mode,
        max_batch=int(cfg.max_batch), stream_events=n_e, seed=0, v_max=None if cfg.v_max is None else float(cfg.v_max),
    )  # fmt: skip
    if base.n_steps != n_k:
        raise RuntimeError(f"time line of {n_k} steps is not representable ({base.n_steps})")

    shape = (n_e, n_v, n_r)
    out = {k: np.empty(shape) for k in ("amplitude", "sd", "amplitude_profile", "sd_profile", "min_gap")}
    first = np.empty(shape, dtype=np.int64)
    mean_speed = np.empty((n_e, n_v, n_k + 1), dtype=np.float32) if keep_mean_speed else None
    paths = {k: np.empty((n_e, n_v, n_r, n_k + 1)) for k in ("s", "v", "x")} if keep_paths else None
    override_x, override_v = lead_x[:, None, :], lead_v[:, None, :]  # vehicle 1: the leader, per profile
    seconds = []
    for i in range(1, n_v + 1):
        started = time.perf_counter()
        events = [
            steady_event(f"{names[e]}/veh{i}", float(u0[e]), float(gaps[e]), float(override_x[e, 0, 0] - gaps[e]), HIST_LEN + n_k)
            for e in range(n_e)
        ]  # fmt: skip
        res = rollout_events(
            model, events, cfg=replace(base, seed=seeds[i - 1]), device=device, keep_paths=True,
            leader_override={"x_lead": override_x, "v_lead": override_v},
        )  # fmt: skip
        v_path, s_path = res.v_path, res.s_path  # (E, R, K + 1)
        for e in range(n_e):
            a, b = burn, burn + int(lengths[e])
            out["amplitude"][e, i - 1], out["sd"][e, i - 1] = _window_stats(v_path[e], a, int(k_e[e]) + 1)
            out["amplitude_profile"][e, i - 1], out["sd_profile"][e, i - 1] = _window_stats(v_path[e], a, b)
            out["min_gap"][e, i - 1] = s_path[e, :, 1 : int(k_e[e]) + 1].min(axis=-1)
            fc = res.first_collision[e]
            first[e, i - 1] = np.where((fc >= 1) & (fc <= k_e[e]), fc, -1)
        if mean_speed is not None:
            mean_speed[:, i - 1] = v_path.mean(axis=1)
        if paths is not None:
            paths["s"][:, i - 1], paths["v"][:, i - 1], paths["x"][:, i - 1] = s_path, v_path, res.x_path
        override_x, override_v = res.x_path, v_path  # vehicle i + 1 follows vehicle i of the same rollout
        seconds.append(time.perf_counter() - started)
        del res, s_path

    lead_amp = np.array([np.ptp(lead_v[e, burn : k_e[e] + 1]) for e in range(n_e)])
    lead_sd = np.array([lead_v[e, burn : k_e[e] + 1].std() for e in range(n_e)])
    lead_sd_profile = np.array([lead_v[e, burn : burn + lengths[e]].std() for e in range(n_e)])
    info = {
        "seconds": time.perf_counter() - t_start,
        "seconds_per_vehicle": seconds,
        "n_profiles": n_e,
        "n_rollouts": n_r,
        "n_steps": n_k,
        "trajectories_per_call": n_e * n_r,
        "device": str(model.device),
    }
    return PlatoonResult(
        amplitude=out["amplitude"], sd=out["sd"], amplitude_profile=out["amplitude_profile"], sd_profile=out["sd_profile"],
        min_gap=out["min_gap"], first_collision=first, leader_amplitude=lead_amp, leader_sd=lead_sd,
        leader_sd_profile=lead_sd_profile, n_profile=lengths, n_steps=n_k, burn_in=burn, tail=tail, s_init=gaps,
        mean_speed=mean_speed, paths=paths, info=info,
    )  # fmt: skip
