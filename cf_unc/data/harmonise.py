"""Harmonised signals (docs/design.md, section 12): one signal processing for every data set.

The data sets were recorded and smoothed differently (docs/m2_report.md, section 3.1); models trained on highD
learn the smoothness of its signals. With ``signal: harmonised`` every event is brought to one processing at load
time, before windows are cut (``cf_unc.data.pipeline.load_events``); the stored events stay untouched.

Per event (arrays (n,), SI units, ``DT = 0.1`` s, sample index t)::

    v_h       = max(SG(v), 0),   v_lead' = SG(v_lead)     Savitzky-Golay, 11 samples (1.1 s), order 2, mode "interp"
    a_h       = central differences of v_h                 (cf_unc.data.pipeline.acceleration_from_speed)
    I[0] = 0, I[t] = I[t-1] + (v_lead'[t] - v_h[t]) DT     relative displacement, scheme of design.md section 1
    least squares over the event:  s[t] - I[t] ~ s0 + c (t DT)   -> gap offset s0 [m], leader speed offset c [m/s]
    v_lead_h  = max(v_lead' + c, 0)                        c is attributed to the leader's speed
    dv_h      = v_h - v_lead_h
    s_h[t]    = s0 + sum_{k=1..t} (v_lead_h[k] - v_h[k]) DT  ( = s0 + c t DT + I[t] when no speed is floored)
    x_follower_h[t] = x_follower[0] + sum_{k=1..t} v_h[k] DT,   x_lead_h = x_follower_h + s_h

so that ``s_h[t] - s_h[t-1] = -dv_h[t] DT`` (the kinematic identity of the integrator) and the event passes
``Event.validate``. The coefficient of ``I`` is fixed to 1: only ``s0`` and ``c`` are fitted.

Speed floor: the filter of a speed that stops at 0 can dip a few cm/s below 0; such samples of ``v_h`` and of
``v_lead_h`` are floored at 0, like the integrator of section 1 does. The positions and the gap are integrated from
the floored speeds, so the kinematic identity and ``validate`` hold exactly; the gap then deviates from
``s0 + c t DT + I`` by the floored amount (counted as ``speed_floored`` per event and in the report).

An event is dropped when ``min(s_h) <= min_gap`` (0.5 m), ``|c| > max_abs_c`` (0.3 m/s) or ``gap_rms = rms(s - s_h)
> max_gap_rms`` (2 m); a non-finite input value drops it too (cannot happen for validated events). An event with
fewer samples than the filter window is dropped unprocessed with the reason ``short_event`` (design.md 15.1: the
filter needs the whole window; with the window 11 of section 12 this drops no event of the imported data sets, which
all have >= 150 samples, so the harmonised artefacts of section 12 are unchanged). ``dropped`` counts every dropped
event once, under its first violated reason in the order of ``DROP_REASONS``; ``violations`` counts every violated
reason (a short event violates only ``short_event``). Kept events get the meta ``signal = "harmonised"``,
``gap_offset`` (s0), ``leader_speed_offset`` (c), ``gap_rms``, ``speed_floored`` and ``a_source_original`` (the
stored acceleration source); ``a_source`` becomes ``"savgol"``.

Events of equal length are processed together as (k, n) arrays (numpy, no loop over samples).

Signal variant ``matched`` (design.md 15.1, experiment E5): the harmonised processing with one window per data set.
The reference data set (``MatchConfig.reference``, highD) keeps ``HarmoniseConfig.window``; every other data set gets
the window of ``MatchConfig.grid`` chosen by ``choose_window`` from its roughness (rms of the second difference of the
harmonised follower speed) relative to the reference. ``scripts/match_roughness.py`` computes the windows once and
writes the matching file; the experiment runners read it (``read_matching``, ``matched_windows``) and refuse to run
(``MatchingError``) when it is missing or was computed with other settings. The signal key of a data set of a matched
experiment is the harmonised key with that data set's window (``cf_unc.experiments.e1.signal_key``), so the reference
data set has exactly the key of the harmonised variant.
"""

from __future__ import annotations

import copy
import functools
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cf_unc.constants import DT
from cf_unc.data.pipeline import central_differences, savgol_speed
from cf_unc.data.schema import Event, EventSet

__all__ = [
    "DEFAULT_SIGNAL",
    "DEFAULT_TARGET_FROM",
    "DROP_REASONS",
    "MATCHING_KIND",
    "MATCH_FLAGS",
    "SIGNALS",
    "TARGET_FROM",
    "HarmoniseConfig",
    "MatchConfig",
    "MatchingError",
    "TargetSpeedEvent",
    "causal_events",
    "causal_savgol",
    "causal_savgol_coeffs",
    "check_signal",
    "check_target_from",
    "choose_window",
    "harmonise_event",
    "harmonise_events",
    "matched_windows",
    "matching_kind",
    "read_matching",
    "with_recorded_targets",
]

# raw | harmonised (section 12) | matched (15.1) | causal and matched_causal (review control R6, design.md 18.5)
SIGNALS = ("raw", "harmonised", "matched", "causal", "matched_causal")
DEFAULT_SIGNAL = "raw"
MATCHED_SIGNALS = {"matched": "harmonised", "matched_causal": "causal"}  # matched variant -> per-data-set processing
# where the speed-based targets (mean_1s, fwd_diff) come from: the processed speed (signal) or the stored one (recorded)
TARGET_FROM = ("signal", "recorded")
DEFAULT_TARGET_FROM = "signal"
DROP_REASONS = ("non_finite", "min_gap", "max_abs_c", "max_gap_rms", "short_event")
_SHORT = DROP_REASONS.index("short_event")
QUANTILES = (0.05, 0.50, 0.95)
_BATCH_SAMPLES = 1 << 21  # samples per vectorised batch (bounds the memory of the (k, n) work arrays)


def check_signal(signal: str) -> str:
    """``signal`` if it is one of ``SIGNALS``, else ValueError."""
    if signal not in SIGNALS:
        raise ValueError(f"signal must be one of {SIGNALS}, got {signal!r}")
    return signal


def check_target_from(value: str | None) -> str:
    """``value`` (None: ``DEFAULT_TARGET_FROM``) if it is one of ``TARGET_FROM``, else ValueError."""
    value = DEFAULT_TARGET_FROM if value is None else str(value)
    if value not in TARGET_FROM:
        raise ValueError(f"target_from must be one of {TARGET_FROM}, got {value!r}")
    return value


@dataclass(frozen=True)
class HarmoniseConfig:
    """Settings of the harmonisation (design.md section 12; experiment config section ``harmonise``)."""

    window: int = 11  # samples of the Savitzky-Golay filter (1.1 s), odd
    order: int = 2  # polynomial order of the filter
    min_gap: float = 0.5  # m, an event is dropped when min(s_h) <= min_gap
    max_abs_c: float = 0.3  # m/s, ... when |c| > max_abs_c
    max_gap_rms: float = 2.0  # m, ... when rms(s - s_h) > max_gap_rms

    def __post_init__(self) -> None:
        if self.window % 2 != 1 or self.window <= self.order or self.order < 0:
            raise ValueError(f"harmonise: need an odd window longer than the order, got {self.window}, {self.order}")

    @classmethod
    def from_config(cls, cfg: "Mapping[str, Any] | HarmoniseConfig | None" = None) -> "HarmoniseConfig":
        """From a mapping (unknown keys are an error), an instance or None (defaults of design.md section 12)."""
        if cfg is None:
            return cls()
        if isinstance(cfg, HarmoniseConfig):
            return cfg
        names = {f.name for f in fields(cls)}
        values = {str(k): cfg[k] for k in cfg}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"unknown harmonise settings {unknown}; known: {sorted(names)}")
        return cls(**{k: int(v) if k in ("window", "order") else float(v) for k, v in values.items()})

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (the form stored in windows specs, checkpoint logs and reports)."""
        return asdict(self)


# ------------------------------------------------------------------------------------------- core
def _shift_cumsum(step: np.ndarray) -> np.ndarray:
    """(k, n) running sums ``out[:, t] = sum_{j=1..t} step[:, j]`` (``out[:, 0] = 0``)."""
    out = np.zeros_like(step)
    np.cumsum(step[:, 1:], axis=1, out=out[:, 1:])
    return out


def _harmonise_arrays(
    s: np.ndarray, dv: np.ndarray, v: np.ndarray, v_lead: np.ndarray, x0: np.ndarray, cfg: HarmoniseConfig
) -> dict[str, np.ndarray]:
    """Harmonisation of k events of equal length n (module docstring).

    s, dv, v, v_lead: (k, n) stored gap [m], approach rate [m/s], follower and leader speed [m/s]; x0: (k,) stored
    follower position at t = 0 [m]. Returns the harmonised (k, n) arrays ``s``, ``dv``, ``v``, ``a``, ``v_lead``,
    ``x_lead``, ``x_follower`` and per event (k,): ``gap_offset``, ``leader_speed_offset``, ``gap_rms``,
    ``min_gap``, ``speed_floored`` (samples floored at 0), ``kin_ss_before`` / ``kin_ss_after`` (sums of squares of
    the kinematic residual of the stored and harmonised gap), ``finite``.
    """
    k, n = s.shape
    v_sg = savgol_speed(v, cfg.window, cfg.order)
    v_h = np.maximum(v_sg, 0.0)
    vl_f = savgol_speed(v_lead, cfg.window, cfg.order)
    rel = np.zeros((k, n))
    rel[:, 1:] = (vl_f[:, 1:] - v_h[:, 1:]) * DT
    disp = _shift_cumsum(rel)  # I[t]
    tau = np.arange(n, dtype=np.float64) * DT
    y = s - disp
    if n >= 2:
        centred = tau - tau.mean()
        c = (y @ centred) / float(centred @ centred)  # least squares slope (sum of centred tau is 0)
    else:
        c = np.zeros(k)
    s0 = y.mean(axis=1) - c * tau.mean()
    vl_shift = vl_f + c[:, None]
    vl_h = np.maximum(vl_shift, 0.0)
    dv_h = v_h - vl_h
    step = np.zeros((k, n))
    step[:, 1:] = -dv_h[:, 1:] * DT
    s_h = s0[:, None] + _shift_cumsum(step)
    move = np.zeros((k, n))
    move[:, 1:] = v_h[:, 1:] * DT
    x_f = x0[:, None] + _shift_cumsum(move)
    resid = s - s_h
    r_before = np.diff(s, axis=1) / DT + dv[:, 1:]
    r_after = np.diff(s_h, axis=1) / DT + dv_h[:, 1:]
    finite = np.isfinite(s).all(axis=1) & np.isfinite(dv).all(axis=1) & np.isfinite(v).all(axis=1)
    finite &= np.isfinite(v_lead).all(axis=1) & np.isfinite(x0)
    return {
        "s": s_h,
        "dv": dv_h,
        "v": v_h,
        "a": central_differences(v_h, DT),
        "v_lead": vl_h,
        "x_lead": x_f + s_h,
        "x_follower": x_f,
        "gap_offset": s0,
        "leader_speed_offset": c,
        "gap_rms": np.sqrt(np.mean(resid * resid, axis=1)),
        "min_gap": s_h.min(axis=1) if n else np.full(k, np.nan),
        "speed_floored": (v_sg < 0.0).sum(axis=1) + (vl_shift < 0.0).sum(axis=1),
        "kin_ss_before": np.sum(r_before * r_before, axis=1),
        "kin_ss_after": np.sum(r_after * r_after, axis=1),
        "finite": finite,
    }


def _violations(out: Mapping[str, np.ndarray], cfg: HarmoniseConfig) -> np.ndarray:
    """(k, len(DROP_REASONS)) bool: the drop rules violated by every processed event (non-finite events: only that
    rule; ``short_event`` is never set here: short events are not processed)."""
    finite = np.asarray(out["finite"], dtype=bool)
    with np.errstate(invalid="ignore"):
        rules = np.stack(
            [
                ~finite,
                finite & ~(out["min_gap"] > cfg.min_gap),
                finite & ~(np.abs(out["leader_speed_offset"]) <= cfg.max_abs_c),
                finite & ~(out["gap_rms"] <= cfg.max_gap_rms),
                np.zeros_like(finite),
            ],
            axis=1,
        )
    return rules


PER_EVENT = ("gap_offset", "leader_speed_offset", "gap_rms", "min_gap", "speed_floored", "kin_ss_before", "kin_ss_after")
ARRAYS = ("s", "dv", "v", "a", "v_lead", "x_lead", "x_follower")


def _harmonise(evs: list[Event], cfg: HarmoniseConfig) -> tuple[list[Event | None], dict[str, np.ndarray]]:
    """Harmonised events (None where dropped, input order) and per-event arrays (m,) of ``PER_EVENT`` (NaN for short
    events, which are not processed), ``n_diff``, ``violations`` (m, len(DROP_REASONS)) and ``kept``."""
    m = len(evs)
    lengths = np.array([len(ev) for ev in evs], dtype=np.int64)
    stats = {name: np.full(m, np.nan) for name in PER_EVENT}
    violations = np.zeros((m, len(DROP_REASONS)), dtype=bool)
    out_events: list[Event | None] = [None] * m
    for n in np.unique(lengths):
        same = np.flatnonzero(lengths == n)
        if int(n) < cfg.window:  # shorter than the filter window: dropped unprocessed (design.md 15.1)
            violations[same, _SHORT] = True
            continue
        size = max(1, _BATCH_SAMPLES // max(int(n), 1))
        for start in range(0, same.size, size):
            idx = same[start : start + size]
            batch = [evs[i] for i in idx]
            out = _harmonise_arrays(
                np.stack([ev.s for ev in batch]),
                np.stack([ev.dv for ev in batch]),
                np.stack([ev.v for ev in batch]),
                np.stack([ev.v_lead for ev in batch]),
                np.array([ev.x_follower[0] if len(ev) else np.nan for ev in batch], dtype=np.float64),
                cfg,
            )
            for name in PER_EVENT:
                stats[name][idx] = out[name]
            rules = _violations(out, cfg)
            violations[idx] = rules
            for j, (i, ev) in enumerate(zip(idx, batch)):
                if rules[j].any():
                    continue
                meta = dict(ev.meta)
                meta["a_source_original"] = meta.get("a_source_original", meta.get("a_source"))
                meta.update(
                    a_source="savgol",
                    signal="harmonised",
                    gap_offset=float(out["gap_offset"][j]),
                    leader_speed_offset=float(out["leader_speed_offset"][j]),
                    gap_rms=float(out["gap_rms"][j]),
                    speed_floored=int(out["speed_floored"][j]),
                )
                out_events[i] = Event(
                    event_id=ev.event_id, dataset=ev.dataset, site=ev.site, follower_id=ev.follower_id,
                    leader_id=ev.leader_id, t=ev.t, meta=meta, **{name: out[name][j] for name in ARRAYS},
                )  # fmt: skip
    stats["n_diff"] = np.maximum(lengths - 1, 0).astype(np.float64)
    stats["violations"] = violations
    stats["kept"] = ~violations.any(axis=1)
    return out_events, stats


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"q{round(100 * q):02d}": None for q in QUANTILES}
    return {f"q{round(100 * q):02d}": float(x) for q, x in zip(QUANTILES, np.quantile(v, QUANTILES))}


def _rms(ss: np.ndarray, count: np.ndarray, rows: np.ndarray) -> float | None:
    """rms from per-event sums of squares ``ss`` and counts over the events ``rows`` with a finite sum."""
    rows = np.asarray(rows, dtype=bool) & np.isfinite(ss)
    total = float(np.sum(count[rows]))
    return float(np.sqrt(np.sum(ss[rows]) / total)) if total > 0 else None


def _report(stats: Mapping[str, np.ndarray], cfg: HarmoniseConfig, seconds: float) -> dict[str, Any]:
    """Report of one data set (``harmonise_events``)."""
    kept, rules = np.asarray(stats["kept"]), np.asarray(stats["violations"])
    m = int(kept.size)
    first = np.where(rules.any(axis=1), rules.argmax(axis=1), -1)
    allrows = np.ones(m, dtype=bool)
    floored = np.asarray(stats["speed_floored"])
    return {
        "n_in": m,
        "n_out": int(kept.sum()),
        "n_dropped": int(m - kept.sum()),
        "share_dropped": float((m - kept.sum()) / m) if m else 0.0,
        "dropped": {r: int(np.sum(first == i)) for i, r in enumerate(DROP_REASONS)},
        "violations": {r: int(rules[:, i].sum()) for i, r in enumerate(DROP_REASONS)},
        "gap_rms_m": _quantiles(stats["gap_rms"]),
        "abs_c_mps": _quantiles(np.abs(np.asarray(stats["leader_speed_offset"]))),
        "gap_offset_m": _quantiles(stats["gap_offset"]),
        "kinematic_rms_before": _rms(stats["kin_ss_before"], stats["n_diff"], allrows),
        "kinematic_rms_before_kept": _rms(stats["kin_ss_before"], stats["n_diff"], kept),
        "kinematic_rms_after": _rms(stats["kin_ss_after"], stats["n_diff"], kept),
        "speed_floored": {"samples": int(np.nansum(floored[kept])), "events": int(np.sum(floored[kept] > 0))},
        "settings": cfg.to_dict(),
        "seconds": round(float(seconds), 3),
    }


# -------------------------------------------------------------------------------------------- API
def harmonise_event(ev: Event, cfg: "Mapping[str, Any] | HarmoniseConfig | None" = None) -> tuple[Event | None, dict]:
    """Harmonised copy of one event (module docstring) or None when a drop rule is violated, and its info.

    info: ``event_id``, ``kept``, ``dropped`` (first violated reason or None), ``violations`` (list),
    ``gap_offset`` s0 [m], ``leader_speed_offset`` c [m/s], ``gap_rms`` [m], ``min_gap`` = min(s_h) [m],
    ``speed_floored`` (samples), ``kinematic_rms_before`` / ``_after`` = rms of ``ds/dt + dv`` [m/s] (the numbers are
    NaN, ``speed_floored`` 0, for an event shorter than the window: dropped as ``short_event`` without processing).
    """
    conf = HarmoniseConfig.from_config(cfg)
    events, stats = _harmonise([ev], conf)
    rules = [r for r, bad in zip(DROP_REASONS, stats["violations"][0]) if bad]
    n_diff = max(float(stats["n_diff"][0]), 1.0)
    floored = float(stats["speed_floored"][0])  # NaN for a short event (not processed)
    info = {
        "event_id": ev.event_id,
        "kept": bool(stats["kept"][0]),
        "dropped": rules[0] if rules else None,
        "violations": rules,
        **{name: float(stats[name][0]) for name in ("gap_offset", "leader_speed_offset", "gap_rms", "min_gap")},
        "speed_floored": int(floored) if np.isfinite(floored) else 0,
        "kinematic_rms_before": float(np.sqrt(stats["kin_ss_before"][0] / n_diff)),
        "kinematic_rms_after": float(np.sqrt(stats["kin_ss_after"][0] / n_diff)),
    }
    return events[0], info


def harmonise_events(
    events: Iterable[Event], cfg: "Mapping[str, Any] | HarmoniseConfig | None" = None
) -> tuple[EventSet, dict[str, Any]]:
    """Harmonised EventSet (kept events in their input order; the inputs are not modified) and the report.

    Report (JSON-like): ``n_in``, ``n_out``, ``n_dropped``, ``share_dropped``, ``dropped`` / ``violations`` by
    reason (module docstring), quantiles (``q05``, ``q50``, ``q95``) over all input events of ``gap_rms_m``,
    ``abs_c_mps`` = |c| and ``gap_offset_m``, the kinematic check ``rms(ds/dt + dv)`` [m/s] of the stored gaps
    (``kinematic_rms_before``: all events, ``_before_kept``: kept events) and of the harmonised gaps
    (``kinematic_rms_after``, zero up to rounding), ``speed_floored`` (samples and events among the kept),
    ``settings`` and ``seconds``.
    """
    conf = HarmoniseConfig.from_config(cfg)
    started = time.perf_counter()
    evs = list(events)
    out, stats = _harmonise(evs, conf)
    kept = EventSet(ev for ev in out if ev is not None)
    return kept, _report(stats, conf, time.perf_counter() - started)


# ------------------------------------------------------------------------------- causal signals (review control R6)
@functools.lru_cache(maxsize=1024)
def _causal_coeffs(length: int, order: int) -> tuple[float, ...]:
    m, p = int(length), min(int(order), int(length) - 1)
    tau = np.arange(-(m - 1), 1, dtype=np.float64)  # sample positions relative to the last one
    return tuple(np.linalg.pinv(np.vander(tau, p + 1, increasing=True))[0])  # fitted value at tau = 0


def causal_savgol_coeffs(length: int, order: int) -> np.ndarray:
    """(length,) weights of the causal Savitzky-Golay filter: the least-squares polynomial of degree
    ``min(order, length - 1)`` fitted to the trailing ``length`` samples and evaluated at the LAST one,
    ``y[t] = sum_j w[j] x[t - length + 1 + j]`` (the weights sum to 1 and reproduce polynomials up to the order)."""
    if int(length) < 1 or int(order) < 0:
        raise ValueError(f"need length >= 1 and order >= 0, got {length}, {order}")
    return np.array(_causal_coeffs(int(length), int(order)))


def causal_savgol(x: np.ndarray, window: int, order: int = 2) -> np.ndarray:
    """Causal (one-sided) Savitzky-Golay filter along the last axis of ``x`` (n,) or (k, n), float64.

    ``y[t]`` = the least-squares polynomial of degree ``order`` over ``x[t - window + 1 .. t]`` evaluated at ``t``
    (``causal_savgol_coeffs``): it uses past and present samples only. The first ``window - 1`` samples of an event
    use the samples available, ``x[0 .. t]`` (degree ``min(order, t)``; ``y[0] = x[0]``), so nothing is dropped and the
    output stays causal. Unlike the centred filter it has no look-ahead, at the price of a larger noise gain.
    """
    x = np.asarray(x, dtype=np.float64)
    window, n = int(window), x.shape[-1]
    if window < 1 or int(order) < 0:
        raise ValueError(f"need window >= 1 and order >= 0, got {window}, {order}")
    out = np.empty_like(x)
    if n >= window:
        out[..., window - 1 :] = np.lib.stride_tricks.sliding_window_view(x, window, axis=-1) @ causal_savgol_coeffs(window, order)
    for t in range(min(window - 1, n)):
        out[..., t] = x[..., : t + 1] @ causal_savgol_coeffs(t + 1, order)
    return out


@dataclass
class TargetSpeedEvent(Event):
    """An Event whose speed-based targets (``mean_1s``, ``fwd_diff``) and their ``y_prev`` come from ``v_target``
    instead of ``v`` (``cf_unc.data.windows.target_speed``): the inputs are processed signals, the target the
    recorded speed (``train.target_from: recorded``, review control R6). ``v_target`` is a transient array: it is not
    one of the parquet fields and is never written; everything else is the Event."""

    v_target: np.ndarray | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.v_target is not None:
            self.v_target = np.ascontiguousarray(self.v_target, dtype=np.float64)
            if self.v_target.shape != self.v.shape:
                raise ValueError(f"{self.event_id}: v_target has shape {self.v_target.shape}, v {self.v.shape}")


def with_recorded_targets(events: Iterable[Event], recorded: Iterable[Event]) -> EventSet:
    """``TargetSpeedEvent`` copies of ``events`` whose ``v_target`` is the speed of the event with the same id in
    ``recorded`` (the stored events before any signal processing; every id must be there with the same length).
    The arrays are shared, not copied."""
    speed = {ev.event_id: ev.v for ev in recorded}
    out = []
    for ev in events:
        if ev.event_id not in speed:
            raise KeyError(f"event {ev.event_id!r} has no recorded counterpart")
        fields_ = {f.name: getattr(ev, f.name) for f in fields(Event)}
        out.append(TargetSpeedEvent(**fields_, v_target=speed[ev.event_id]))
    return EventSet(out)


def _causal_arrays(s: np.ndarray, v: np.ndarray, v_lead: np.ndarray, x0: np.ndarray, cfg: HarmoniseConfig) -> dict[str, np.ndarray]:
    """Causal processing of k events of equal length n ((k, n) stored gap, follower and leader speed; x0 (k,)):
    every input signal smoothed with ``causal_savgol(window, order)``, speeds floored at 0, the gap floored at
    ``cfg.min_gap`` (the endpoint of a long trailing quadratic can overshoot a recorded gap that stays above it),
    ``dv = v - v_lead``, ``a`` = central differences of the smoothed speed (``a[t-1]``, the regime input, uses samples
    <= t), positions integrated from the smoothed speed so that ``s = x_lead - x_follower`` holds exactly."""
    k, n = s.shape
    v_c = np.maximum(causal_savgol(v, cfg.window, cfg.order), 0.0)
    vl_c = np.maximum(causal_savgol(v_lead, cfg.window, cfg.order), 0.0)
    s_raw = causal_savgol(s, cfg.window, cfg.order)
    s_c = np.maximum(s_raw, cfg.min_gap)
    move = np.zeros((k, n))
    move[:, 1:] = v_c[:, 1:] * DT
    x_f = x0[:, None] + _shift_cumsum(move)
    resid = s - s_c
    finite = np.isfinite(s).all(axis=1) & np.isfinite(v).all(axis=1) & np.isfinite(v_lead).all(axis=1) & np.isfinite(x0)
    return {
        "s": s_c, "dv": v_c - vl_c, "v": v_c, "a": central_differences(v_c, DT), "v_lead": vl_c, "x_lead": x_f + s_c,
        "x_follower": x_f, "gap_rms": np.sqrt(np.mean(resid * resid, axis=1)),
        "min_gap_recorded": s.min(axis=1) if n else np.full(k, np.nan), "gap_floored": (s_raw < cfg.min_gap).sum(axis=1),
        "finite": finite,
    }  # fmt: skip


def causal_events(
    events: Iterable[Event], cfg: "Mapping[str, Any] | HarmoniseConfig | None" = None
) -> tuple[EventSet, dict[str, Any]]:
    """Causally processed EventSet (signal ``causal``, review control R6, design.md 18.5) and its report.

    Per event: ``s``, ``v``, ``v_lead`` smoothed with ``causal_savgol(cfg.window, cfg.order)`` (the gap with the same
    filter: the kinematic gap reconstruction of section 12 fits an offset and a slope over the whole event and is not
    causal), ``dv = v - v_lead``, ``a`` = central differences of the smoothed speed, positions integrated from it.
    Drop rules: ``non_finite``, ``min_gap`` (``min(s) <= cfg.min_gap`` of the RECORDED gap: data cleaning that does
    not depend on the filter, so every window keeps the same events; the smoothed gap is floored at ``cfg.min_gap``
    where the endpoint of the trailing quadratic overshoots, counted as ``gap_floored``) and ``short_event`` (fewer
    samples than the window; the rule of 15.1). ``max_abs_c`` and ``max_gap_rms`` belong to the offset fit of the
    harmonisation and are not applied (counted as 0). Kept events get the meta ``signal = "causal"``, ``causal_window``,
    ``gap_rms`` (rms of the stored minus the smoothed gap), ``gap_floored`` and ``a_source = "causal_savgol"``. The
    report has the keys of ``harmonise_events`` (quantities of the offset fit are None) plus ``filter = "causal"`` and
    ``gap_floored`` (samples and events among the kept).
    """
    conf = HarmoniseConfig.from_config(cfg)
    started = time.perf_counter()
    evs = list(events)
    m = len(evs)
    lengths = np.array([len(ev) for ev in evs], dtype=np.int64)
    violations = np.zeros((m, len(DROP_REASONS)), dtype=bool)
    gap_rms = np.full(m, np.nan)
    floored = np.zeros(m, dtype=np.int64)
    out: list[Event | None] = [None] * m
    for n in np.unique(lengths):
        same = np.flatnonzero(lengths == n)
        if int(n) < conf.window:
            violations[same, _SHORT] = True
            continue
        size = max(1, _BATCH_SAMPLES // max(int(n), 1))
        for start in range(0, same.size, size):
            idx = same[start : start + size]
            batch = [evs[i] for i in idx]
            arr = _causal_arrays(np.stack([ev.s for ev in batch]), np.stack([ev.v for ev in batch]),
                                 np.stack([ev.v_lead for ev in batch]),
                                 np.array([ev.x_follower[0] if len(ev) else np.nan for ev in batch]), conf)  # fmt: skip
            gap_rms[idx] = arr["gap_rms"]
            floored[idx] = arr["gap_floored"]
            finite = arr["finite"]
            with np.errstate(invalid="ignore"):
                violations[idx, DROP_REASONS.index("non_finite")] = ~finite
                violations[idx, DROP_REASONS.index("min_gap")] = finite & ~(arr["min_gap_recorded"] > conf.min_gap)
            for j, (i, ev) in enumerate(zip(idx, batch)):
                if violations[i].any():
                    continue
                meta = dict(ev.meta)
                meta["a_source_original"] = meta.get("a_source_original", meta.get("a_source"))
                meta.update(a_source="causal_savgol", signal="causal", causal_window=int(conf.window),
                            gap_rms=float(arr["gap_rms"][j]), gap_floored=int(arr["gap_floored"][j]))  # fmt: skip
                out[i] = Event(
                    event_id=ev.event_id, dataset=ev.dataset, site=ev.site, follower_id=ev.follower_id,
                    leader_id=ev.leader_id, t=ev.t, meta=meta, **{name: arr[name][j] for name in ARRAYS},
                )  # fmt: skip
    kept = ~violations.any(axis=1)
    first = np.where(violations.any(axis=1), violations.argmax(axis=1), -1)
    report = {
        "filter": "causal",
        "n_in": m,
        "n_out": int(kept.sum()),
        "n_dropped": int(m - kept.sum()),
        "share_dropped": float((m - kept.sum()) / m) if m else 0.0,
        "dropped": {r: int(np.sum(first == i)) for i, r in enumerate(DROP_REASONS)},
        "violations": {r: int(violations[:, i].sum()) for i, r in enumerate(DROP_REASONS)},
        "gap_rms_m": _quantiles(gap_rms),
        "abs_c_mps": {f"q{round(100 * q):02d}": None for q in QUANTILES},
        "gap_offset_m": {f"q{round(100 * q):02d}": None for q in QUANTILES},
        "gap_floored": {"samples": int(floored[kept].sum()), "events": int(np.sum(floored[kept] > 0)),
                        "share_samples": float(floored[kept].sum() / max(int(lengths[kept].sum()), 1))},
        "settings": conf.to_dict(),
        "seconds": round(float(time.perf_counter() - started), 3),
    }
    return EventSet(ev for ev in out if ev is not None), report


# ------------------------------------------------------------------------------------ matched signals (15.1)
# flags of a matched window (choose_window); the last one marks a reference whose short-event share is not below the
# limit (reported by scripts/match_roughness.py, the reference keeps its window)
MATCH_FLAGS = ("not_reached", "capped", "cap_unreachable", "short_share_above_limit")


class MatchingError(RuntimeError):
    """The matching file of the signal variant ``matched`` is missing, incomplete or was computed with other settings
    than the experiment config asks (the runners refuse to run; design.md 15.1)."""


@dataclass(frozen=True)
class MatchConfig:
    """Settings of the roughness matching (design.md 15.1; experiment config section ``match``)."""

    reference: str = "highd"  # data set that keeps HarmoniseConfig.window (the training data of the reused models)
    grid: tuple[int, ...] = (11, 15, 21, 25, 31, 41, 51, 61, 81, 101)  # odd Savitzky-Golay windows [samples]
    tolerance: float = 0.1  # a window matches when its roughness <= (1 + tolerance) x the reference roughness
    file: str | None = None  # matching file: written by scripts/match_roughness.py, read by the runners
    max_short_share: float = 0.05  # a window that drops this share of the events or more as short_event is capped

    def __post_init__(self) -> None:
        grid = tuple(int(w) for w in self.grid)
        if not grid or any(w % 2 != 1 or w < 3 for w in grid) or list(grid) != sorted(set(grid)):
            raise ValueError(f"match: the grid must be increasing odd windows >= 3 samples, got {list(self.grid)}")
        if not float(self.tolerance) >= 0.0:
            raise ValueError(f"match: the tolerance must be >= 0, got {self.tolerance}")
        if not 0.0 < float(self.max_short_share) <= 1.0:
            raise ValueError(f"match: max_short_share must be in (0, 1], got {self.max_short_share}")
        object.__setattr__(self, "grid", grid)

    @classmethod
    def from_config(cls, cfg: "Mapping[str, Any] | MatchConfig | None" = None) -> "MatchConfig":
        """From a mapping (unknown keys are an error), an instance or None (defaults of design.md 15.1)."""
        if cfg is None:
            return cls()
        if isinstance(cfg, MatchConfig):
            return cfg
        names = {f.name for f in fields(cls)}
        values = {str(k): cfg[k] for k in cfg}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"unknown match settings {unknown}; known: {sorted(names)}")
        out: dict[str, Any] = {}
        for k, v in values.items():
            if k == "grid":
                out[k] = tuple(int(w) for w in v)
            elif k in ("tolerance", "max_short_share"):
                out[k] = float(v)
            else:
                out[k] = None if v is None else str(v)
        return cls(**out)

    def settings(self) -> dict[str, Any]:
        """The settings a matching file must have been computed with (all but ``file``), plain JSON types."""
        return {"reference": self.reference, "grid": list(self.grid), "tolerance": float(self.tolerance),
                "max_short_share": float(self.max_short_share)}  # fmt: skip


def choose_window(
    roughness: Mapping[int, float], short_share: Mapping[int, float], target: float, grid: Sequence[int],
    max_short_share: float = 0.05,
) -> dict[str, Any]:  # fmt: skip
    """Window of a matched data set (design.md 15.1) from its roughness at every grid window.

    roughness[w]: roughness of the events kept with window w (NaN never matches); short_share[w]: share of the
    events of the data set that are shorter than w; target: ``(1 + tolerance)`` x the reference roughness. Rule: the
    smallest grid window with ``roughness <= target`` (``window_rule``; none: the largest window, flag
    ``not_reached``); when the share of short events at that window is not below ``max_short_share``, the window is
    capped at the largest smaller grid window whose share is below it (``cap``, flag ``capped``; none: the smallest
    grid window, flag ``cap_unreachable``). Returns ``window``, ``window_rule``, ``cap`` (None without a cap) and
    ``flags``.
    """
    windows = sorted(int(w) for w in grid)
    reached = [w for w in windows if np.isfinite(float(roughness.get(w, np.nan))) and float(roughness[w]) <= float(target)]
    flags: list[str] = [] if reached else ["not_reached"]
    rule = reached[0] if reached else windows[-1]
    window, cap = rule, None
    if not float(short_share.get(rule, 0.0)) < float(max_short_share):
        below = [w for w in windows if w < rule and float(short_share.get(w, 0.0)) < float(max_short_share)]
        cap = window = below[-1] if below else windows[0]
        flags.append("capped" if below else "cap_unreachable")
    return {"window": int(window), "window_rule": int(rule), "cap": None if cap is None else int(cap), "flags": flags}


_MATCHING_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


def read_matching(path: str | Path) -> dict[str, Any]:
    """Content of a matching file (a copy; parsed once per path, modification time and size). MatchingError when the
    file does not exist."""
    path = Path(path)
    try:
        stat = path.stat()
    except FileNotFoundError:
        raise MatchingError(
            f"signal matched: the matching file {path} does not exist; compute it first with "
            "scripts/match_roughness.py --config <experiment config> (design.md 15.1)"
        ) from None
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    if key not in _MATCHING_CACHE:
        _MATCHING_CACHE[key] = json.loads(path.read_text(encoding="utf-8"))
    return copy.deepcopy(_MATCHING_CACHE[key])


# how a matching file of each matched variant was computed: the filter and the part of every data set it used
# (files written before the causal variant hold neither key: they are centred, all events)
MATCHING_KIND: dict[str, dict[str, str]] = {"matched": {"filter": "centred", "part": "all"},
                                            "matched_causal": {"filter": "causal", "part": "val"}}  # fmt: skip


def matching_kind(content: Mapping[str, Any]) -> dict[str, str]:
    """``filter`` and ``part`` of a matching file's content (defaults of the files of section 15: centred, all)."""
    return {"filter": str(content.get("filter", "centred")), "part": str(content.get("part", "all"))}


def matched_windows(
    path: str | Path,
    harmonise: "Mapping[str, Any] | HarmoniseConfig | None" = None,
    match: "Mapping[str, Any] | MatchConfig | None" = None,
    variant: str = "matched",
) -> dict[str, int]:
    """Window [samples] per data set of the matching file ``path``, checked against the experiment's settings: the
    harmonisation settings, ``reference``, ``grid``, ``tolerance`` and ``max_short_share`` must be those the file
    was computed with, the reference must keep ``harmonise.window`` and every other window must be a grid window;
    the file must be of the kind of ``variant`` (``MATCHING_KIND``: ``matched`` a centred matching over all events,
    ``matched_causal`` a causal matching over the validation folds). MatchingError otherwise (and when the file is
    missing)."""
    content = read_matching(path)
    if variant not in MATCHING_KIND:
        raise ValueError(f"variant must be one of {sorted(MATCHING_KIND)}, got {variant!r}")
    if matching_kind(content) != MATCHING_KIND[variant]:
        raise MatchingError(f"{path} is a matching of the kind {matching_kind(content)}, the signal {variant} needs "
                            f"{MATCHING_KIND[variant]}; compute it with scripts/match_roughness.py for that signal")  # fmt: skip
    conf, mconf = HarmoniseConfig.from_config(harmonise), MatchConfig.from_config(match)
    expected = {"harmonise": conf.to_dict(), **mconf.settings()}
    stored = {k: content.get(k) for k in expected}
    if json.dumps(stored, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise MatchingError(
            f"{path} was computed with {stored}, the config asks {expected}; compute it again with "
            "scripts/match_roughness.py"
        )
    windows = content.get("windows")
    if not isinstance(windows, Mapping) or not windows:
        raise MatchingError(f"{path} holds no windows")
    out = {str(k): int(v) for k, v in windows.items()}
    if out.get(mconf.reference) != conf.window:
        raise MatchingError(f"{path}: the reference {mconf.reference!r} must keep the window {conf.window}, "
                            f"the file gives {out.get(mconf.reference)}")  # fmt: skip
    odd = {k: w for k, w in out.items() if k != mconf.reference and w not in mconf.grid}
    if odd:
        raise MatchingError(f"{path}: windows outside the grid {list(mconf.grid)}: {odd}")
    return out
