"""Closed-loop stochastic rollouts of car-following models (docs/design.md, section 11.2).

For event ``e`` with start index ``t0`` the model sees the true history ``(s, dv, v)[t0 - H + 1 : t0 + 1]``, the
leader follows its recorded trajectory ``x_lead[t0:]``, ``v_lead[t0:]`` and the follower starts at
``x_follower[t0]``, ``v[t0]``. With ``K = horizon_s / DT`` steps::

    step k = 0 .. K-1
      if k % hold == 0:  a_cmd = model.sample_step(hist, state, generator)     one draw per trajectory
      a      = clip(a_cmd, A_MIN, A_MAX)            (v is floored at 0 and, only if v_max is set, capped at v_max)
      v'     = max(v + a DT, 0);   x' = x + v' DT
      s'     = x_lead[t0 + k + 1] - x';   dv' = v' - v_lead[t0 + k + 1]
      hist   = (hist[1:], (max(s', s_floor), dv', v'))        the floor acts on the model input only
      first collision step = k + 1 if s' <= 0 and none before

Collisions are flagged, never removed: a trajectory continues with a negative gap.

Implementation
--------------
* Trajectories are the flattened pairs (event, rollout), event-major (row ``e * R + r``). They run in batches
  of whole events with at most ``max_batch`` trajectories on the device of the model.
* The kinematic state (positions, speeds, gaps) is float64; the model receives float32 histories (see
  "Deviations" of the M3 report: a float32 integrator loses ~1e-4 m over 10 s at highway speeds).
* Histories live in one preallocated float32 track buffer ``(B, H + K, 3)`` per batch: step ``k`` writes
  column ``H + k`` and the model reads the view ``buffer[:, k : k + H]`` (no copy, no reallocation).
* Random streams: consecutive blocks of ``stream_events`` events share one ``torch.Generator`` on the
  device, seeded from ``(seed, index of the first event of the block)``; the model is called once per block
  and step and every block has its own sampler state. A batch holds whole blocks, so the results depend on
  ``(seed, event order, stream_events)`` but not on ``max_batch``.
* Models whose sampler needs the leader (``init_sampler(..., v_lead=...)``, DiffusionCF) get the leader
  speeds of the rollout window, ``v_lead[:, k]`` = leader speed at step k (k = 0 .. K); the keyword is only
  passed to samplers that accept it. Plan samplers (``needs_leader_future``) follow their plan step by step
  and therefore need ``hold_steps = 1``.
"""

from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from cf_unc.constants import A_MAX, A_MIN, DT, HIST_LEN, N_FEATURES, S_EPS
from cf_unc.data.schema import Event, EventSet
from cf_unc.models.base import SAMPLER_MODES, CFModel, resolve_device

__all__ = ["RolloutConfig", "RolloutResult", "rollout_events", "stream_seed", "accepts_keyword"]

_F64 = torch.float64
_STEP_TOL = 1e-9  # horizons must be integer multiples of DT up to this relative tolerance


# ----------------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class RolloutConfig:
    """Settings of :func:`rollout_events`.

    n_rollouts       R, rollouts per event
    horizon_s        simulated time [s], K = horizon_s / DT steps
    horizons_s       evaluation horizons [s] (multiples of DT inside (0, horizon_s])
    hold_steps       a model draw is kept for this many steps (``mean_1s`` target: 10)
    sampler_mode     ``per_step`` | ``per_trajectory`` (design.md, section 7)
    max_batch        trajectories per batch (memory bound; results do not depend on it)
    stream_events    events per random stream (the unit of randomness, see the module docstring)
    seed             base seed of the random streams
    s_floor          floor of the gap in the model input [m]
    v_max            optional cap of the follower speed [m/s] (None: no cap, design.md section 7); a sensitivity of E3
    """

    n_rollouts: int = 200
    horizon_s: float = 10.0
    horizons_s: tuple[float, ...] = (1.0, 3.0, 5.0, 10.0)
    hold_steps: int = 1
    sampler_mode: str = "per_step"
    max_batch: int = 100_000
    stream_events: int = 100
    seed: int = 0
    s_floor: float = S_EPS
    v_max: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizons_s", tuple(float(h) for h in self.horizons_s))
        if int(self.n_rollouts) < 1 or int(self.hold_steps) < 1 or int(self.max_batch) < 1 or int(self.stream_events) < 1:
            raise ValueError("n_rollouts, hold_steps, max_batch and stream_events must be >= 1")
        if self.sampler_mode not in SAMPLER_MODES:
            raise ValueError(f"sampler_mode must be one of {SAMPLER_MODES}, got {self.sampler_mode!r}")
        if not (math.isfinite(self.s_floor) and self.s_floor > 0.0):
            raise ValueError("s_floor must be positive")
        if self.v_max is not None and not (math.isfinite(float(self.v_max)) and float(self.v_max) > 0.0):
            raise ValueError(f"v_max must be None or positive, got {self.v_max!r}")
        _steps(self.horizon_s, "horizon_s")
        if not self.horizons_s:
            raise ValueError("need at least one evaluation horizon")
        steps = [_steps(h, "horizons_s") for h in self.horizons_s]
        if any(k < 1 or k > self.n_steps for k in steps) or len(set(steps)) != len(steps):
            raise ValueError(f"horizons_s must be distinct multiples of DT in (0, horizon_s], got {self.horizons_s}")

    @property
    def n_steps(self) -> int:
        """K, simulated steps."""
        return _steps(self.horizon_s, "horizon_s")

    @property
    def horizon_steps(self) -> np.ndarray:
        """(n_h,) int64 step index of every evaluation horizon."""
        return np.array([_steps(h, "horizons_s") for h in self.horizons_s], dtype=np.int64)


def _steps(seconds: float, name: str) -> int:
    k = float(seconds) / DT
    if not (math.isfinite(k) and k >= 1.0 - _STEP_TOL and abs(k - round(k)) <= _STEP_TOL * max(1.0, k)):
        raise ValueError(f"{name}: {seconds} s is not a positive multiple of DT = {DT} s")
    return int(round(k))


def stream_seed(seed: int, first_event: int) -> int:
    """63-bit torch seed of the random stream that starts at event ``first_event`` (independent streams)."""
    state = np.random.SeedSequence([abs(int(seed)), abs(int(first_event))]).generate_state(1, np.uint64)[0]
    return int(state >> np.uint64(1))


def accepts_keyword(fn: Any, name: str) -> bool:
    """Whether the callable ``fn`` has a parameter ``name`` (or ``**kwargs``)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


# ------------------------------------------------------------------------------------------ result
@dataclass
class RolloutResult:
    """Output of :func:`rollout_events` for E events, R rollouts and n_h evaluation horizons.

    s, v               (E, R, n_h) float64 gap [m] and follower speed [m/s] at the horizons
    first_collision    (E, R) int64 step index (1 .. K) of the first gap <= 0, -1 without collision
    s_min              (E, R) float64 smallest simulated gap over steps 1 .. K [m]
    truth_s, truth_v   (E, n_h) float64 recorded gap and speed at ``t0 + horizon``
    t0                 (E,) int64 start index (last history sample)
    horizons_s         (n_h,) float64, horizon_steps (n_h,) int64
    s_path, v_path, x_path   (E, R, K + 1) float64 full gap, speed and follower front-bumper position
                             (event frame) at steps 0 .. K, only with ``keep_paths=True``
    info               timing, batches, device and peak GPU memory
    """

    s: np.ndarray
    v: np.ndarray
    first_collision: np.ndarray
    s_min: np.ndarray
    truth_s: np.ndarray
    truth_v: np.ndarray
    t0: np.ndarray
    horizons_s: np.ndarray
    horizon_steps: np.ndarray
    n_rollouts: int
    hold_steps: int
    sampler_mode: str
    seed: int
    event_ids: list[str] = field(default_factory=list)
    s_path: np.ndarray | None = None
    v_path: np.ndarray | None = None
    x_path: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        return int(self.s.shape[0])

    def collided(self) -> np.ndarray:
        """(E, n_h) share of rollouts whose first collision is at or before each horizon."""
        fc = self.first_collision[:, :, None]
        hit = (fc >= 0) & (fc <= self.horizon_steps[None, None, :])
        return hit.mean(axis=1)


# ------------------------------------------------------------------------------------------ inputs
def _event_list(events: EventSet | Sequence[Event]) -> list[Event]:
    evs = list(events)
    if not evs:
        raise ValueError("no events to roll out")
    return evs


def _starts(starts: int | Sequence[int] | np.ndarray | None, n: int) -> np.ndarray:
    if starts is None:
        return np.full(n, HIST_LEN - 1, dtype=np.int64)
    t0 = np.asarray(starts, dtype=np.int64)
    if t0.ndim == 0:
        return np.full(n, int(t0), dtype=np.int64)
    if t0.shape != (n,):
        raise ValueError(f"starts must be a scalar or have shape ({n},), got {t0.shape}")
    return t0


def _check_lengths(evs: list[Event], t0: np.ndarray, n_steps: int) -> None:
    lengths = np.array([len(ev) for ev in evs], dtype=np.int64)
    early = np.flatnonzero(t0 < HIST_LEN - 1)
    if early.size:
        raise ValueError(f"start index below HIST_LEN - 1 = {HIST_LEN - 1} for {early.size} events (first: event "
                         f"{evs[early[0]].event_id!r}, t0 = {t0[early[0]]})")  # fmt: skip
    short = np.flatnonzero(t0 + n_steps >= lengths)
    if short.size:
        e = int(short[0])
        raise ValueError(
            f"{short.size} events are too short for t0 + horizon: event {evs[e].event_id!r} has {lengths[e]} samples, "
            f"t0 = {t0[e]} and {n_steps} steps need {t0[e] + n_steps + 1} (horizon {n_steps * DT:g} s)"
        )


def _leader_override(
    override: Mapping[str, np.ndarray] | None, n_events: int, n_rollouts: int, n_steps: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Validated override arrays ``(x_lead, v_lead)``, each (E, 1 | R, K + 1) float64."""
    if override is None:
        return None
    out = []
    for key in ("x_lead", "v_lead"):
        if key not in override:
            raise KeyError(f"leader_override needs {key!r}")
        arr = np.array(override[key], dtype=np.float64)  # a copy: broadcast views and read-only arrays are fine
        if arr.ndim == 2:
            arr = arr[:, None, :]
        if arr.ndim != 3 or arr.shape[0] != n_events or arr.shape[1] not in (1, n_rollouts) or arr.shape[2] != n_steps + 1:
            raise ValueError(
                f"leader_override[{key!r}] must have shape (E, K + 1) or (E, R, K + 1) = ({n_events}, [{n_rollouts},] "
                f"{n_steps + 1}), got {np.shape(override[key])}"
            )
        out.append(arr)
    if out[0].shape[1] != out[1].shape[1]:
        raise ValueError("x_lead and v_lead of the leader override must both be per event or both per rollout")
    return out[0], out[1]


def _blocks(n_events: int, stream_events: int, n_rollouts: int, max_batch: int) -> list[list[tuple[int, int]]]:
    """Batches of whole random-stream blocks: [[(first_event, last_event_excl), ...], ...]."""
    block = min(int(stream_events), n_events)
    if block * n_rollouts > int(max_batch):
        raise ValueError(
            f"a random stream block of {block} events x {n_rollouts} rollouts exceeds max_batch = {max_batch}; "
            "lower stream_events or raise max_batch"
        )
    per_batch = max(1, int(max_batch) // (block * n_rollouts))
    starts = list(range(0, n_events, block))
    spans = [(a, min(a + block, n_events)) for a in starts]
    return [spans[i : i + per_batch] for i in range(0, len(spans), per_batch)]


# -------------------------------------------------------------------------------------------- engine
def rollout_events(
    model: CFModel,
    events: EventSet | Sequence[Event],
    starts: int | Sequence[int] | np.ndarray | None = None,
    cfg: RolloutConfig | None = None,
    device: str | torch.device | None = None,
    leader_override: Mapping[str, np.ndarray] | None = None,
    keep_paths: bool = False,
) -> RolloutResult:
    """Roll ``model`` out ``cfg.n_rollouts`` times from every event (module docstring).

    events           E events (``EventSet`` or list); rolled in this order (the order enters the random streams)
    starts           start index t0 per event (scalar or (E,)), default ``HIST_LEN - 1``; ``t0 + K`` must lie
                     inside the event, else ValueError
    device           default: the model's device; another device moves the model there (``model.to``)
    leader_override  ``{"x_lead", "v_lead"}`` arrays (E, K + 1) or (E, R, K + 1): leader rear-bumper position
                     [m] (event frame) and speed [m/s] at t0 .. t0 + K replacing the recorded leader (the history
                     stays the recorded one); used by platoon rollouts
    keep_paths       also return the full gap, speed and position paths (E, R, K + 1)
    """
    cfg = cfg or RolloutConfig()
    evs = _event_list(events)
    n_ev, n_r, n_k, hist = len(evs), int(cfg.n_rollouts), cfg.n_steps, HIST_LEN
    t0 = _starts(starts, n_ev)
    _check_lengths(evs, t0, n_k)
    hold = int(cfg.hold_steps)
    if getattr(model, "needs_leader_future", False) and hold != 1:
        raise ValueError(f"{type(model).__name__} follows a sampled plan step by step: it needs hold_steps = 1, got {hold}")
    dev = model.device if device is None else resolve_device(device)
    if dev != model.device:
        model.to(dev)
    override = _leader_override(leader_override, n_ev, n_r, n_k)
    h_steps = cfg.horizon_steps
    n_h = h_steps.size
    pass_v_lead = accepts_keyword(model.init_sampler, "v_lead")

    # per-event inputs (float64): history, leader over the window, initial follower state, truth
    window = np.arange(n_k + 1)
    hist_np = np.stack([np.stack([ev.s, ev.dv, ev.v], axis=-1)[t - hist + 1 : t + 1] for ev, t in zip(evs, t0)])
    if override is None:
        lead_x = np.stack([ev.x_lead[t + window] for ev, t in zip(evs, t0)])[:, None, :]
        lead_v = np.stack([ev.v_lead[t + window] for ev, t in zip(evs, t0)])[:, None, :]
    else:
        lead_x, lead_v = override
    x_init = np.array([ev.x_follower[t] for ev, t in zip(evs, t0)])
    v_init = np.array([ev.v[t] for ev, t in zip(evs, t0)])
    truth_s = np.stack([ev.s[t + h_steps] for ev, t in zip(evs, t0)])
    truth_v = np.stack([ev.v[t + h_steps] for ev, t in zip(evs, t0)])

    out_s = np.empty((n_ev, n_r, n_h))
    out_v = np.empty((n_ev, n_r, n_h))
    out_first = np.empty((n_ev, n_r), dtype=np.int64)
    out_smin = np.empty((n_ev, n_r))
    paths = {k: np.empty((n_ev, n_r, n_k + 1)) for k in ("s", "v", "x")} if keep_paths else None

    started = time.perf_counter()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    batches = _blocks(n_ev, cfg.stream_events, n_r, cfg.max_batch)
    horizon_col = {int(k): j for j, k in enumerate(h_steps)}
    with torch.no_grad():
        for spans in batches:
            e0, e1 = spans[0][0], spans[-1][1]
            n_b, b = e1 - e0, (e1 - e0) * n_r
            # track buffer of model inputs: columns 0..H-1 true history, column H+k the state after step k
            track = torch.empty((b, hist + n_k, N_FEATURES), dtype=torch.float32, device=dev)
            h0 = torch.as_tensor(hist_np[e0:e1], dtype=_F64, device=dev)
            h0[..., 0].clamp_(min=cfg.s_floor)
            track[:, :hist] = h0.repeat_interleave(n_r, dim=0)
            lx = torch.as_tensor(lead_x[e0:e1], dtype=_F64, device=dev)  # (n_b, 1 | R, K + 1)
            lv = torch.as_tensor(lead_v[e0:e1], dtype=_F64, device=dev)
            x = torch.as_tensor(x_init[e0:e1], dtype=_F64, device=dev).repeat_interleave(n_r)
            v = torch.as_tensor(v_init[e0:e1], dtype=_F64, device=dev).repeat_interleave(n_r)
            first = torch.full((b,), -1, dtype=torch.int64, device=dev)
            s_min = torch.full((b,), math.inf, dtype=_F64, device=dev)
            s_h = torch.empty((b, n_h), dtype=_F64, device=dev)
            v_h = torch.empty((b, n_h), dtype=_F64, device=dev)
            a_cmd = torch.zeros(b, dtype=_F64, device=dev)
            if paths is not None:
                p_s, p_v, p_x = (torch.empty((b, n_k + 1), dtype=_F64, device=dev) for _ in range(3))
                p_s[:, 0] = lx[:, :, 0].expand(n_b, n_r).reshape(b) - x  # = s[t0] for the recorded leader
                p_v[:, 0], p_x[:, 0] = v, x

            # one generator and sampler state per random-stream block
            streams = []
            for a, z in spans:
                gen = torch.Generator(device=dev).manual_seed(stream_seed(cfg.seed, a))
                rows = slice((a - e0) * n_r, (z - e0) * n_r)
                kwargs: dict[str, Any] = {}
                if pass_v_lead:
                    kwargs["v_lead"] = lv[a - e0 : z - e0].expand(z - a, n_r, n_k + 1).reshape((z - a) * n_r, n_k + 1).float()
                state = model.init_sampler((z - a) * n_r, gen, cfg.sampler_mode, **kwargs)
                streams.append((rows, gen, state))

            for k in range(n_k):
                if k % hold == 0:
                    for rows, gen, state in streams:
                        a_cmd[rows] = model.sample_step(track[rows, k : k + hist], state, gen).to(_F64)
                acc = a_cmd.clamp(A_MIN, A_MAX)
                v = (v + acc * DT).clamp_(min=0.0, max=cfg.v_max)
                x = x + v * DT
                s = lx[:, :, k + 1].expand(n_b, n_r).reshape(b) - x
                dv = v - lv[:, :, k + 1].expand(n_b, n_r).reshape(b)
                track[:, hist + k] = torch.stack([s.clamp(min=cfg.s_floor), dv, v], dim=1)
                first = torch.where((s <= 0.0) & (first < 0), k + 1, first)
                s_min = torch.minimum(s_min, s)
                j = horizon_col.get(k + 1)
                if j is not None:
                    s_h[:, j], v_h[:, j] = s, v
                if paths is not None:
                    p_s[:, k + 1], p_v[:, k + 1], p_x[:, k + 1] = s, v, x

            shape = (n_b, n_r)
            out_s[e0:e1] = s_h.reshape(*shape, n_h).cpu().numpy()
            out_v[e0:e1] = v_h.reshape(*shape, n_h).cpu().numpy()
            out_first[e0:e1] = first.reshape(shape).cpu().numpy()
            out_smin[e0:e1] = s_min.reshape(shape).cpu().numpy()
            if paths is not None:
                for key, tensor in (("s", p_s), ("v", p_v), ("x", p_x)):
                    paths[key][e0:e1] = tensor.reshape(*shape, n_k + 1).cpu().numpy()
            del track, streams

    info: dict[str, Any] = {
        "seconds": time.perf_counter() - started,
        "device": str(dev),
        "n_batches": len(batches),
        "n_streams": sum(len(s) for s in batches),
        "trajectories": n_ev * n_r,
        "steps": n_k,
    }
    if dev.type == "cuda":
        info["peak_memory_mb"] = torch.cuda.max_memory_allocated(dev) / 2**20
    return RolloutResult(
        s=out_s,
        v=out_v,
        first_collision=out_first,
        s_min=out_smin,
        truth_s=truth_s,
        truth_v=truth_v,
        t0=t0,
        horizons_s=np.asarray(cfg.horizons_s, dtype=np.float64),
        horizon_steps=h_steps,
        n_rollouts=n_r,
        hold_steps=hold,
        sampler_mode=cfg.sampler_mode,
        seed=int(cfg.seed),
        event_ids=[ev.event_id for ev in evs],
        s_path=None if paths is None else paths["s"],
        v_path=None if paths is None else paths["v"],
        x_path=None if paths is None else paths["x"],
        info=info,
    )
