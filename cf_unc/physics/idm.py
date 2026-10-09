"""Intelligent Driver Model: acceleration law, closed-loop simulation, calibration (docs/design.md, section 4).

    s*(v, dv) = s0 + max(0, v T + v dv / (2 sqrt(a_max b)))
    a_IDM     = a_max [1 - (v / v0)^delta - (s* / max(s, S_EPS))^2]        clipped to [A_MIN, A_MAX]

with ``dv = v - v_lead`` (approach rate, positive when closing in). Units: gap s [m], speeds [m/s],
accelerations [m/s^2], time [s]. Closed-loop simulations use the semi-implicit Euler integrator of
design.md, section 1 (``speed_step`` / ``leader_positions``), shared with the synthetic generator.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence, Union

import numpy as np
from scipy.optimize import differential_evolution, minimize

from cf_unc.constants import A_MAX, A_MIN, DT, S_EPS
from cf_unc.data.splits import subsample_events

if TYPE_CHECKING:
    import torch

    from cf_unc.data.schema import Event

ParamValue = Union[float, np.ndarray, "torch.Tensor"]
AccelFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

PARAM_NAMES = ("v0", "T", "s0", "a_max", "b")  # order of IDMParams.as_vector()
DEFAULT_DELTA = 4.0
IDM_BOUNDS: dict[str, tuple[float, float]] = {
    "v0": (10.0, 45.0),
    "T": (0.3, 3.5),
    "s0": (0.5, 8.0),
    "a_max": (0.3, 4.0),
    "b": (0.3, 5.0),
}
OBJECTIVES = ("spacing", "spacing_nrmse", "accel")


# --------------------------------------------------------------------------- helpers
def _is_torch(x: Any) -> bool:
    """True for torch tensors (checked without importing torch)."""
    return type(x).__module__.partition(".")[0] == "torch"


def _normalise(value: Any) -> ParamValue:
    """Scalars (python, numpy, 0-d) -> float; tensors untouched; other array-likes -> float ndarray."""
    if _is_torch(value):
        return value
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return arr if arr.dtype.kind == "f" else arr.astype(np.float64)


def _to_numpy(value: ParamValue) -> float | np.ndarray:
    return value.detach().cpu().numpy() if _is_torch(value) else value


def _plain(value: ParamValue) -> float | list:
    if _is_torch(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return float(value)


def _sqrt(x: Any) -> Any:
    if isinstance(x, float):
        return math.sqrt(x)
    return x.sqrt() if _is_torch(x) else np.sqrt(x)


def _free_term(ratio: Any, delta: Any) -> Any:
    """(v / v0)^delta, with an exact multiplication path for the usual delta = 4."""
    if isinstance(delta, float) and delta == 4.0:
        r2 = ratio * ratio
        return r2 * r2
    return ratio**delta


# --------------------------------------------------------------------------- parameters
@dataclass
class IDMParams:
    """IDM parameters: v0 [m/s], T [s], s0 [m], a_max [m/s^2], b [m/s^2], delta [-].

    A field is a float or an array / tensor broadcastable to the state (per-driver parameters).
    Scalars are stored as python floats and must be finite and positive.
    """

    v0: ParamValue
    T: ParamValue
    s0: ParamValue
    a_max: ParamValue
    b: ParamValue
    delta: ParamValue = DEFAULT_DELTA

    def __post_init__(self) -> None:
        for f in fields(self):
            value = _normalise(getattr(self, f.name))
            if isinstance(value, float) and not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"IDM parameter {f.name} must be finite and positive, got {value}")
            setattr(self, f.name, value)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict (array / tensor fields become lists)."""
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "IDMParams":
        """Inverse of ``to_dict``; ``delta`` is optional, unknown keys are an error."""
        d = dict(d)
        unknown = sorted(set(d) - {f.name for f in fields(cls)})
        missing = [name for name in PARAM_NAMES if name not in d]
        if unknown or missing:
            raise ValueError(f"IDMParams.from_dict: unknown keys {unknown}, missing keys {missing}")
        return cls(**d)

    def as_vector(self) -> np.ndarray:
        """Free parameters [v0, T, s0, a_max, b], float64 of shape (5,) (or (..., 5) for array fields)."""
        values = [np.asarray(_to_numpy(getattr(self, name)), dtype=np.float64) for name in PARAM_NAMES]
        return np.stack(np.broadcast_arrays(*values), axis=-1)

    @classmethod
    def from_vector(cls, vec: Any, delta: float = DEFAULT_DELTA) -> "IDMParams":
        """Inverse of ``as_vector``: ``vec`` (5,) or (..., 5), numpy or torch (tensors stay differentiable)."""
        if not _is_torch(vec):
            vec = np.asarray(vec, dtype=np.float64)
        if vec.ndim < 1 or vec.shape[-1] != len(PARAM_NAMES):
            raise ValueError(f"expected shape (..., {len(PARAM_NAMES)}), got {tuple(vec.shape)}")
        return cls(*(vec[..., i] for i in range(len(PARAM_NAMES))), delta=delta)


# --------------------------------------------------------------------------- IDM law
def idm_accel(s: Any, dv: Any, v: Any, params: IDMParams, *, clip: bool = True) -> Any:
    """IDM acceleration [m/s^2] at gap ``s`` [m], approach rate ``dv = v - v_lead`` [m/s], speed ``v`` [m/s].

    Inputs broadcast against each other and against the parameter fields. If any of ``s, dv, v`` is a
    torch tensor, the computation runs in torch with the dtype / device of that tensor and is
    differentiable (w.r.t. states and tensor-valued parameters); otherwise numpy is used. ``v`` is
    floored at 0 (no effect on valid states). The output is clipped to [A_MIN, A_MAX] unless
    ``clip=False``.
    """
    if _is_torch(s) or _is_torch(dv) or _is_torch(v):
        return _idm_accel_torch(s, dv, v, params, clip)
    return _idm_accel_numpy(s, dv, v, params, clip)


def _idm_accel_numpy(s: Any, dv: Any, v: Any, p: IDMParams, clip: bool) -> np.ndarray:
    s, dv = np.asarray(s), np.asarray(dv)
    v = np.maximum(np.asarray(v), 0.0)
    v0, T, s0, a_max, b, delta = (_to_numpy(getattr(p, f.name)) for f in fields(p))
    s_star = s0 + np.maximum(v * T + v * dv / (2.0 * _sqrt(a_max * b)), 0.0)
    ratio = s_star / np.maximum(s, S_EPS)
    acc = a_max * (1.0 - _free_term(v / v0, delta) - ratio * ratio)
    return np.clip(acc, A_MIN, A_MAX) if clip else acc


def _torch_value(value: Any, dtype: "torch.dtype", device: "torch.device") -> Any:
    """Python floats pass through; tensors are cast (differentiably), array-likes converted, to dtype / device."""
    import torch

    if isinstance(value, float):
        return value
    if _is_torch(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)


def _idm_accel_torch(s: Any, dv: Any, v: Any, p: IDMParams, clip: bool) -> "torch.Tensor":
    import torch

    ref = next(x for x in (s, dv, v) if _is_torch(x))
    dtype = ref.dtype if ref.is_floating_point() else torch.get_default_dtype()
    device = ref.device
    s, dv, v = (_torch_value(x if _is_torch(x) else np.asarray(x, dtype=np.float64), dtype, device) for x in (s, dv, v))
    v0, T, s0, a_max, b, delta = (_torch_value(getattr(p, f.name), dtype, device) for f in fields(p))
    v = torch.clamp(v, min=0.0)
    s_star = s0 + torch.clamp(v * T + v * dv / (2.0 * _sqrt(a_max * b)), min=0.0)
    ratio = s_star / torch.clamp(s, min=S_EPS)
    acc = a_max * (1.0 - _free_term(v / v0, delta) - ratio * ratio)
    return torch.clamp(acc, A_MIN, A_MAX) if clip else acc


def equilibrium_gap(v: Any, params: IDMParams) -> Any:
    """Steady-state gap [m] at speed ``v`` [m/s] (dv = 0): (s0 + v T) / sqrt(1 - (v / v0)^delta), inf for v >= v0.

    numpy (float for scalar input) or torch, depending on the type of ``v``.
    """
    if _is_torch(v):
        import torch

        dtype = v.dtype if v.is_floating_point() else torch.get_default_dtype()
        v = torch.clamp(v.to(dtype), min=0.0)
        v0, T, s0, delta = (_torch_value(getattr(params, n), dtype, v.device) for n in ("v0", "T", "s0", "delta"))
        denom = 1.0 - _free_term(v / v0, delta)
        ok = denom > 0.0
        gap = (s0 + v * T) / torch.sqrt(torch.where(ok, denom, torch.ones_like(denom)))
        return torch.where(ok, gap, torch.full_like(gap, math.inf))
    v = np.maximum(np.asarray(v, dtype=np.float64), 0.0)
    v0, T, s0, delta = (_to_numpy(getattr(params, n)) for n in ("v0", "T", "s0", "delta"))
    denom = 1.0 - _free_term(v / v0, delta)
    ok = denom > 0.0
    gap = np.where(ok, (s0 + v * T) / np.sqrt(np.where(ok, denom, 1.0)), np.inf)
    return float(gap) if gap.ndim == 0 else gap


# --------------------------------------------------------------------------- integrator
def speed_step(v: np.ndarray, a: np.ndarray, dt: float = DT) -> tuple[np.ndarray, np.ndarray]:
    """Speed update of the integrator for an acceleration already clipped to [A_MIN, A_MAX].

    Returns ``(a_eff, v_next)`` with ``v_next = max(v + a dt, 0)``. Where the standstill clip is active
    the effective acceleration ``-v / dt`` is returned, so ``v_next == v + a_eff dt`` up to rounding.
    """
    v_next = v + a * dt
    stop = v_next < 0.0
    if np.any(stop):
        a = np.where(stop, -v / dt, a)
        v_next = np.where(stop, 0.0, v_next)
    return a, v_next


def leader_positions(x0: Any, v_lead: np.ndarray, dt: float = DT) -> np.ndarray:
    """Positions [m] of the semi-implicit Euler scheme along the last axis of ``v_lead`` [m/s].

    ``x[..., 0] = x0`` and ``x[..., t+1] = x[..., t] + v_lead[..., t+1] dt`` (sequential cumulative sum).
    """
    inc = np.asarray(v_lead, dtype=np.float64) * dt
    inc[..., 0] = x0
    return np.cumsum(inc, axis=-1)


def simulate_follower(
    accel_fn: AccelFn,
    s_init: Any,
    v_init: Any,
    v_lead: np.ndarray,
    x_lead: np.ndarray | None = None,
    n_valid: Any = None,
    dt: float = DT,
) -> dict[str, Any]:
    """Deterministic closed-loop follower simulation, vectorised over a batch of B events.

    accel_fn(s, dv, v) -> a     (B,) float64 arrays in, (B,) acceleration [m/s^2] out (clipped here)
    s_init, v_init              (B,) initial gap [m] and speed [m/s]
    v_lead                      (B, T) leader speed [m/s], padded after ``n_valid``; a (T,) array is one event
    x_lead                      (B, T) leader rear-bumper position [m]; if None it is integrated from
                                ``v_lead`` with ``x_lead[0] = s_init`` (see ``leader_positions``)
    n_valid                     (B,) number of valid samples per row (default T)

    The follower front bumper starts at ``x_lead[:, 0] - s_init`` (0 when ``x_lead`` is None) and is
    integrated with the scheme of design.md, section 1. Returns (B, T) float64 arrays ``s, dv, v, a, x,
    x_lead`` (``a`` is the effective acceleration: ``v[t+1] = v[t] + a[t] dt``) and ``collision_index``
    (B,) int64: first valid t with ``s <= 0``, -1 if none. Gaps may become negative after a collision.
    Entries at ``t >= n_valid`` are NaN (the padded leader is extrapolated at constant speed internally).
    """
    v_lead = np.asarray(v_lead, dtype=np.float64)
    single = v_lead.ndim == 1
    if single:
        v_lead = v_lead[None, :]
        x_lead = None if x_lead is None else np.asarray(x_lead, dtype=np.float64)[None, :]
    if v_lead.ndim != 2:
        raise ValueError(f"v_lead must have shape (B, T), got {v_lead.shape}")
    n_batch, n_steps = v_lead.shape
    s_init = np.broadcast_to(np.asarray(s_init, dtype=np.float64), (n_batch,))
    v_init = np.broadcast_to(np.asarray(v_init, dtype=np.float64), (n_batch,))
    if n_valid is None:
        n_valid = np.full(n_batch, n_steps, dtype=np.int64)
    n_valid = np.broadcast_to(np.asarray(n_valid, dtype=np.int64), (n_batch,))
    if np.any(n_valid < 1) or np.any(n_valid > n_steps):
        raise ValueError("n_valid must lie in [1, T]")

    steps = np.arange(n_steps)
    valid = steps[None, :] < n_valid[:, None]
    padded = not bool(valid.all())
    rows, last = np.arange(n_batch), n_valid - 1
    if padded:  # constant-speed extrapolation keeps the padded part finite
        vl_last = v_lead[rows, last][:, None]
        v_lead = np.where(valid, v_lead, vl_last)
    if x_lead is None:
        x_lead = leader_positions(s_init, v_lead, dt)
    else:
        x_lead = np.asarray(x_lead, dtype=np.float64)
        if x_lead.shape != v_lead.shape:
            raise ValueError(f"x_lead has shape {x_lead.shape}, expected {v_lead.shape}")
        if padded:
            ahead = (steps[None, :] - last[:, None]) * dt
            x_lead = np.where(valid, x_lead, x_lead[rows, last][:, None] + ahead * vl_last)

    # time-major storage: every step works on contiguous (B,) slices
    xl, vl = np.ascontiguousarray(x_lead.T), np.ascontiguousarray(v_lead.T)
    s, v, a, x = (np.empty((n_steps, n_batch)) for _ in range(4))
    x[0] = xl[0] - s_init
    v[0] = v_init
    for t in range(n_steps):
        s[t] = xl[t] - x[t]
        a_cmd = np.clip(np.asarray(accel_fn(s[t], v[t] - vl[t], v[t]), dtype=np.float64), A_MIN, A_MAX)
        a[t], v_next = speed_step(v[t], a_cmd, dt)
        if t + 1 < n_steps:
            v[t + 1] = v_next
            x[t + 1] = x[t] + v_next * dt

    out = {
        "s": np.ascontiguousarray(s.T),
        "dv": np.ascontiguousarray((v - vl).T),
        "v": np.ascontiguousarray(v.T),
        "a": np.ascontiguousarray(a.T),
        "x": np.ascontiguousarray(x.T),
        "x_lead": np.array(x_lead, dtype=np.float64),
    }
    hit = (out["s"] <= 0.0) & valid
    out["collision_index"] = np.where(hit.any(axis=1), hit.argmax(axis=1), -1).astype(np.int64)
    if padded:
        for key in ("s", "dv", "v", "a", "x", "x_lead"):
            out[key][~valid] = np.nan
    if single:
        out = {key: (int(value[0]) if key == "collision_index" else value[0]) for key, value in out.items()}
    return out


# --------------------------------------------------------------------------- calibration
@dataclass
class _Padded:
    """Events padded to a common length T; leader positions relative to the follower start (x_f(0) = 0)."""

    s: np.ndarray  # (B, T) observed gap [m]
    dv: np.ndarray  # (B, T) [m/s]
    v: np.ndarray  # (B, T) [m/s]
    a: np.ndarray  # (B, T) [m/s^2]
    v_lead: np.ndarray  # (B, T) [m/s], edge-padded
    x_lead: np.ndarray  # (B, T) [m], x_lead - x_lead[0] + s[0], constant-speed extrapolation when padded
    n_valid: np.ndarray  # (B,)
    valid: np.ndarray  # (B, T) bool

    @property
    def s_init(self) -> np.ndarray:
        return self.s[:, 0]

    @property
    def v_init(self) -> np.ndarray:
        return self.v[:, 0]


def _pad_events(events: Sequence["Event"], dt: float = DT) -> _Padded:
    if not events:
        raise ValueError("no events")
    n_valid = np.array([len(ev) for ev in events], dtype=np.int64)
    n_batch, n_steps = len(events), int(n_valid.max())
    arr = {name: np.zeros((n_batch, n_steps)) for name in ("s", "dv", "v", "a", "v_lead", "x_lead")}
    for i, ev in enumerate(events):
        m = len(ev)
        for name in ("s", "dv", "v", "a", "v_lead"):
            arr[name][i, :m] = getattr(ev, name)
        arr["x_lead"][i, :m] = (ev.x_lead - ev.x_lead[0]) + ev.s[0]
        if m < n_steps:
            arr["v_lead"][i, m:] = ev.v_lead[-1]
            arr["x_lead"][i, m:] = arr["x_lead"][i, m - 1] + ev.v_lead[-1] * dt * np.arange(1, n_steps - m + 1)
    valid = np.arange(n_steps)[None, :] < n_valid[:, None]
    return _Padded(**arr, n_valid=n_valid, valid=valid)


class _SpacingObjective:
    """Pooled closed-loop spacing RMSE for a batch of candidate parameter vectors theta (P, 5) -> (P,).

    All candidates and events are simulated together ((P, B) arrays, python loop over time only). The
    simulated gap is clipped at 0 in the error and ``penalty`` [m] times the fraction of valid samples
    with s <= 0 is added (small steps in theta give small steps in the penalty). With ``normalise`` the
    value is divided by the RMS of the observed gaps (NRMSE, same minimiser).
    """

    def __init__(self, batch: _Padded, dt: float, penalty: float, normalise: bool, delta: float) -> None:
        self.xl = np.ascontiguousarray(batch.x_lead.T)  # (T, B)
        self.vl = np.ascontiguousarray(batch.v_lead.T)
        self.s_obs = np.ascontiguousarray(batch.s.T)
        self.w = np.ascontiguousarray(batch.valid.T, dtype=np.float64)
        self.v_init = batch.v_init.copy()
        self.n = float(self.w.sum())
        self.scale = math.sqrt(float(np.sum(self.w * self.s_obs**2)) / self.n) if normalise else 1.0
        self.dt, self.penalty, self.delta = dt, penalty, delta

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        p = IDMParams.from_vector(theta[:, None, :], delta=self.delta)  # fields (P, 1)
        n_cand = theta.shape[0]
        x = np.zeros((n_cand, self.xl.shape[1]))
        v = np.repeat(self.v_init[None, :], n_cand, axis=0)
        s = self.xl[0] - x
        sse, hits = np.zeros(n_cand), np.zeros(n_cand)
        for t in range(self.xl.shape[0] - 1):
            a = idm_accel(s, v - self.vl[t], v, p)
            a, v = speed_step(v, a, self.dt)
            x = x + v * self.dt
            s = self.xl[t + 1] - x
            err = np.maximum(s, 0.0) - self.s_obs[t + 1]
            sse += (err * err) @ self.w[t + 1]
            hits += (s <= 0.0) @ self.w[t + 1]
        return (np.sqrt(sse / self.n) + self.penalty * hits / self.n) / self.scale


class _AccelObjective:
    """One-step acceleration MSE mean((a - a_IDM)^2) over all valid samples, theta (P, 5) -> (P,).

    Candidates are evaluated one by one with scalar parameters: the (N,) temporaries stay in cache,
    which is faster than (P, N) broadcasting for the sample counts of interest.
    """

    def __init__(self, batch: _Padded, delta: float) -> None:
        m = batch.valid
        self.s, self.dv, self.v, self.a = batch.s[m], batch.dv[m], batch.v[m], batch.a[m]
        self.delta = delta

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        out = np.empty(theta.shape[0])
        for i, row in enumerate(theta):
            r = self.a - idm_accel(self.s, self.dv, self.v, IDMParams.from_vector(row, delta=self.delta))
            out[i] = (r @ r) / r.size
        return out


def _resolve_bounds(bounds: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | None) -> np.ndarray:
    """(5, 2) array of [low, high] in PARAM_NAMES order; a mapping overrides the defaults per parameter."""
    table = dict(IDM_BOUNDS)
    if isinstance(bounds, Mapping):
        unknown = sorted(set(bounds) - set(PARAM_NAMES))
        if unknown:
            raise ValueError(f"unknown parameters in bounds: {unknown}")
        table.update({k: tuple(v) for k, v in bounds.items()})
    elif bounds is not None:
        if len(bounds) != len(PARAM_NAMES):
            raise ValueError(f"bounds must have {len(PARAM_NAMES)} (low, high) pairs")
        table = {name: tuple(b) for name, b in zip(PARAM_NAMES, bounds)}
    out = np.array([table[name] for name in PARAM_NAMES], dtype=np.float64)
    if out.shape != (len(PARAM_NAMES), 2) or np.any(out[:, 0] <= 0.0) or np.any(out[:, 1] <= out[:, 0]):
        raise ValueError(f"invalid bounds {out.tolist()}")
    return out


def evaluate_idm(events: Iterable["Event"], params: IDMParams, dt: float = DT) -> dict[str, Any]:
    """Fit statistics of ``params`` on ``events``.

    Closed loop: every event is simulated from its true initial state with its leader trajectory given;
    ``spacing_rmse`` [m] pools the (unclipped) gap errors of all samples, ``spacing_nrmse`` divides it by
    the RMS observed gap. One step: residuals ``a - a_IDM`` at the observed states (``accel_rmse``,
    ``accel_resid_std``, ``accel_resid_mean`` [m/s^2]). ``n_collisions`` counts events whose simulation
    reaches ``s <= 0``, ``collision_samples`` the valid samples with ``s <= 0``.
    """
    return _evaluate(_pad_events(list(events), dt), params, dt)


def _evaluate(batch: _Padded, params: IDMParams, dt: float) -> dict[str, Any]:
    sim = simulate_follower(
        lambda s, dv, v: idm_accel(s, dv, v, params),
        batch.s_init,
        batch.v_init,
        batch.v_lead,
        x_lead=batch.x_lead,
        n_valid=batch.n_valid,
        dt=dt,
    )
    m = batch.valid
    gap_err = sim["s"][m] - batch.s[m]
    resid = (batch.a - idm_accel(batch.s, batch.dv, batch.v, params))[m]
    rmse = float(np.sqrt(np.mean(gap_err**2)))
    return {
        "spacing_rmse": rmse,
        "spacing_nrmse": rmse / float(np.sqrt(np.mean(batch.s[m] ** 2))),
        "accel_rmse": float(np.sqrt(np.mean(resid**2))),
        "accel_resid_std": float(np.std(resid)),
        "accel_resid_mean": float(np.mean(resid)),
        "n_events": int(batch.s.shape[0]),
        "n_samples": int(m.sum()),
        "n_collisions": int(np.sum(sim["collision_index"] >= 0)),
        "collision_samples": int(np.sum((sim["s"] <= 0.0) & m)),
    }


def calibrate_idm(
    events: Iterable["Event"],
    objective: str = "spacing",
    max_events: int | None = 300,
    seed: int = 0,
    maxiter: int = 100,
    popsize: int = 15,
    bounds: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | None = None,
    *,
    tol: float = 0.01,
    polish: bool = True,
    polish_maxfev: int = 250,
    collision_penalty: float = 10.0,
    delta: float = DEFAULT_DELTA,
    dt: float = DT,
) -> tuple[IDMParams, dict[str, Any]]:
    """One population IDM parameter set for ``events`` (design.md, section 4), ``delta`` fixed.

    objective   "spacing": pooled closed-loop spacing RMSE [m] (every event simulated from its true
                initial state, leader given; collisions penalised, see ``_SpacingObjective``);
                "spacing_nrmse": the same divided by the RMS observed gap; "accel": one-step
                acceleration MSE [m^2/s^4].
    max_events  a deterministic random subset (``seed``) of at most this many events is used (None: all).

    Global search: ``scipy.optimize.differential_evolution`` (seeded, deferred updating, no built-in
    polish); the whole population of a generation is evaluated in one batched simulation through the
    map-like ``workers`` hook. Local polish: bounded Nelder-Mead in bound-normalised coordinates
    (``polish_maxfev`` evaluations). Returns the parameters and a JSON-serialisable report with the
    objective ``value`` and the ``evaluate_idm`` statistics of the used events.
    """
    started = time.perf_counter()
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    evs = list(events)
    if not evs:
        raise ValueError("no events to calibrate on")
    if max_events is not None and len(evs) > max_events:
        evs = list(subsample_events(evs, max_events, seed))
    batch = _pad_events(evs, dt)
    box = _resolve_bounds(bounds)
    lo, span = box[:, 0], box[:, 1] - box[:, 0]

    if objective == "accel":
        batched: Callable[[np.ndarray], np.ndarray] = _AccelObjective(batch, delta)
    else:
        batched = _SpacingObjective(batch, dt, collision_penalty, objective == "spacing_nrmse", delta)

    def scalar(theta: np.ndarray) -> float:
        return float(batched(np.asarray(theta, dtype=np.float64)[None, :])[0])

    def population_map(func: Callable, population: Iterable[np.ndarray]) -> np.ndarray:
        # DE hands the whole generation (deferred updating) to this map: evaluate it in one batch
        pop = population if isinstance(population, np.ndarray) else list(population)
        return batched(np.asarray(pop, dtype=np.float64))

    de = differential_evolution(
        scalar,
        bounds=[tuple(b) for b in box],
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        polish=False,
        updating="deferred",
        vectorized=False,
        workers=population_map,
    )
    best_x, best_f = np.asarray(de.x, dtype=np.float64), float(de.fun)
    polish_info = None
    if polish and polish_maxfev > 0:

        def unit_objective(u: np.ndarray) -> float:
            return scalar(lo + np.clip(u, 0.0, 1.0) * span)

        u0 = np.clip((best_x - lo) / span, 0.0, 1.0)
        step = np.where(u0 + 0.05 <= 1.0, 0.05, -0.05)
        simplex = np.vstack([u0, u0 + np.diag(step)])
        res = minimize(
            unit_objective,
            u0,
            method="Nelder-Mead",
            bounds=[(0.0, 1.0)] * len(PARAM_NAMES),
            options={"initial_simplex": simplex, "maxfev": polish_maxfev, "xatol": 1e-4, "fatol": 1e-9},
        )
        polish_info = {"method": "Nelder-Mead", "value": float(res.fun), "nfev": int(res.nfev)}
        if res.fun < best_f:
            best_x, best_f = lo + np.clip(res.x, 0.0, 1.0) * span, float(res.fun)

    params = IDMParams.from_vector(best_x, delta=delta)
    stats = _evaluate(batch, params, dt)
    report = {
        "objective": objective,
        "value": best_f,
        **stats,
        "seconds": time.perf_counter() - started,
        "params": params.to_dict(),
        "bounds": {name: [float(b[0]), float(b[1])] for name, b in zip(PARAM_NAMES, box)},
        "de": {"value": float(de.fun), "nit": int(de.nit), "nfev": int(de.nfev), "message": str(de.message)},
        "polish": polish_info,
        "settings": {
            "max_events": max_events,
            "seed": seed,
            "maxiter": maxiter,
            "popsize": popsize,
            "tol": tol,
            "polish_maxfev": polish_maxfev,
            "collision_penalty": collision_penalty,
        },
    }
    return params, report
