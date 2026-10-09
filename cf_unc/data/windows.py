"""Sliding windows over events (see docs/design.md, sections 3.3 and 11.1)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Iterable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from cf_unc.constants import DT, HIST_LEN, MEAN_TARGET_STEPS, N_FEATURES

if TYPE_CHECKING:
    from cf_unc.data.schema import Event

TARGETS = ("a", "fwd_diff", "mean_1s")
DEFAULT_TARGET = "a"
# samples a target spans (its "own length": y_prev is the target this many samples earlier) and the samples after t
# it needs (a[t] is stored at t; the forward difference needs v[t+1], mean_1s needs v[t+10])
TARGET_STEPS: dict[str, int] = {"a": 1, "fwd_diff": 1, "mean_1s": MEAN_TARGET_STEPS}
TARGET_LOOKAHEAD: dict[str, int] = {"a": 0, "fwd_diff": 1, "mean_1s": MEAN_TARGET_STEPS}


def check_target(target: str) -> str:
    """``target`` if it is one of ``TARGETS``, else ValueError."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    return target


def target_values(a: np.ndarray | None, v: np.ndarray | None, t: np.ndarray, target: str) -> np.ndarray:
    """Target at sample indices ``t`` (float64, m/s^2) of one event, or of concatenated events with global indices.

    ``"a"``: ``a[t]``; ``"fwd_diff"``: ``(v[t+1] - v[t]) / DT``; ``"mean_1s"``: ``(v[t+K] - v[t]) / (K DT)`` with
    ``K = MEAN_TARGET_STEPS``, the mean acceleration over the next second. ``a`` (m/s^2) is only needed for ``"a"``,
    ``v`` (m/s) only for the others. Needs ``t + TARGET_LOOKAHEAD[target]`` inside the arrays (the caller keeps
    windows from straddling two events).
    """
    t = np.asarray(t, dtype=np.int64)
    if check_target(target) == "a":
        return np.asarray(a, dtype=np.float64)[t]
    v = np.asarray(v, dtype=np.float64)
    if target == "fwd_diff":
        return (v[t + 1] - v[t]) / DT
    k = MEAN_TARGET_STEPS
    return (v[t + k] - v[t]) / (k * DT)


@dataclass
class WindowSet:
    """Supervised samples cut from an EventSet.

    X               (N, H, 3) float32  history of (s, dv, v), indices t-H+1 .. t
    y               (N,)      float32  target: a[t] (default), (v[t+1]-v[t])/DT or (v[t+10]-v[t])/(10 DT), m/s^2
    driver          (N,)      int64    code of (dataset, follower_id)
    event_index     (N,)      int64    index of the source event in the EventSet
    t_index         (N,)      int64    index t of the last history sample in the event
    driver_names    list[str]          decoding table of ``driver``
    y_seq           (N, L)    float32  a[t : t+L]            (only if seq_len > 0)
    v_lead_future   (N, L)    float32  v_lead[t+1 : t+L+1]   (only if seq_len > 0)
    """

    X: np.ndarray
    y: np.ndarray
    driver: np.ndarray
    event_index: np.ndarray
    t_index: np.ndarray
    driver_names: list[str]
    y_seq: np.ndarray | None = None
    v_lead_future: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def hist_len(self) -> int:
        return int(self.X.shape[1])

    def subset(self, idx: np.ndarray) -> "WindowSet":
        """Rows ``idx`` (integer or boolean index, repetitions allowed)."""
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "driver_names" or value is None:
                out[f.name] = value
            else:
                out[f.name] = value[idx]
        return WindowSet(**out)


def driver_key(event: "Event") -> str:
    """Name of the driver ``(dataset, follower_id)`` of an event: ``"<dataset>/<follower_id>"``."""
    return f"{event.dataset}/{event.follower_id}"


def count_windows(
    lengths: "np.ndarray | Iterable[int]", hist_len: int = HIST_LEN, stride: int = 1, seq_len: int = 0, target: str = "a"
) -> np.ndarray:
    """(n_events,) int64 number of windows ``make_windows`` cuts from events of the given lengths (same rule, no
    window is built): window ends ``t = hist_len-1, hist_len-1+stride, ...`` with ``t + max(seq_len,
    TARGET_LOOKAHEAD[target]) < n``."""
    if hist_len < 1 or stride < 1 or seq_len < 0:
        raise ValueError("need hist_len >= 1, stride >= 1, seq_len >= 0")
    lengths = np.asarray(list(lengths) if not isinstance(lengths, np.ndarray) else lengths, dtype=np.int64)
    t_last = lengths - 1 - max(seq_len, TARGET_LOOKAHEAD[check_target(target)])
    first = hist_len - 1
    return np.where(t_last >= first, (t_last - first) // stride + 1, 0).astype(np.int64)


def concat_windows(parts: "Iterable[WindowSet]", n_events: "Iterable[int]") -> WindowSet:
    """One WindowSet of several, as if the events of all parts were one EventSet in the order of ``parts``.

    ``n_events[i]``: number of events of the EventSet part ``i`` was cut from; ``event_index`` is shifted by the
    events of the parts before it. Driver codes are re-coded into the sorted union of the parts' ``driver_names``
    (the codes ``make_windows`` gives the union of the events). ``y_seq`` / ``v_lead_future`` are kept when every
    part has them (else None).
    """
    parts, n_events = list(parts), [int(n) for n in n_events]
    if len(parts) != len(n_events) or not parts:
        raise ValueError("concat_windows needs one event count per part and at least one part")
    names = sorted({name for p in parts for name in p.driver_names})
    code = {name: i for i, name in enumerate(names)}
    offsets = np.concatenate([[0], np.cumsum(n_events)[:-1]]).astype(np.int64)
    hist = {p.hist_len for p in parts if len(p)} or {parts[0].hist_len}
    if len(hist) > 1:
        raise ValueError(f"concat_windows: parts with different history lengths {sorted(hist)}")

    def recode(p: WindowSet) -> np.ndarray:
        table = np.array([code[name] for name in p.driver_names], dtype=np.int64)
        return table[p.driver] if len(p) else np.empty(0, dtype=np.int64)

    def optional(name: str) -> np.ndarray | None:
        values = [getattr(p, name) for p in parts]
        return None if any(v is None for v in values) else np.concatenate(values)

    return WindowSet(
        X=np.concatenate([p.X for p in parts]),
        y=np.concatenate([p.y for p in parts]),
        driver=np.concatenate([recode(p) for p in parts]),
        event_index=np.concatenate([p.event_index.astype(np.int64) + off for p, off in zip(parts, offsets)]),
        t_index=np.concatenate([p.t_index.astype(np.int64) for p in parts]),
        driver_names=names,
        y_seq=optional("y_seq"),
        v_lead_future=optional("v_lead_future"),
    )


def make_windows(
    events: Iterable["Event"],
    hist_len: int = HIST_LEN,
    stride: int = 1,
    seq_len: int = 0,
    target: str = "a",
) -> WindowSet:
    """Cut every event into supervised windows (design.md, sections 3.3 and 11.1).

    For each event of length n the window ends are ``t = hist_len-1, hist_len-1+stride, ...`` up to the
    last t with ``t + max(seq_len, TARGET_LOOKAHEAD[target]) < n`` (``t+1 < n`` for ``"fwd_diff"``,
    ``t+10 < n`` for ``"mean_1s"``, ``t+seq_len < n`` for ``seq_len > 0``).
    ``X[i] = (s, dv, v)[t-hist_len+1 : t+1]``; ``y[i]`` = ``target_values`` at t: ``a[t]``, ``(v[t+1]-v[t]) / DT``
    or ``(v[t+10]-v[t]) / (10 DT)``; with ``seq_len = L > 0`` also ``y_seq[i] = a[t : t+L]`` and
    ``v_lead_future[i] = v_lead[t+1 : t+L+1]`` (whatever the target). Driver codes index the sorted
    ``driver_names`` of all events (``driver_key``).
    """
    if hist_len < 1 or stride < 1 or seq_len < 0:
        raise ValueError("need hist_len >= 1, stride >= 1, seq_len >= 0")
    check_target(target)
    evs = list(events)
    driver_names = sorted({driver_key(ev) for ev in evs})
    code_of = {name: i for i, name in enumerate(driver_names)}
    ev_driver = np.array([code_of[driver_key(ev)] for ev in evs], dtype=np.int64)
    lengths = np.array([len(ev) for ev in evs], dtype=np.int64)

    # window ends per event, then flattened: event_index, t_index and global row g in the concatenation
    lookahead = max(seq_len, TARGET_LOOKAHEAD[target])
    first = hist_len - 1
    t_last = lengths - 1 - lookahead
    counts = np.where(t_last >= first, (t_last - first) // stride + 1, 0)
    n_win = int(counts.sum())
    event_index = np.repeat(np.arange(len(evs), dtype=np.int64), counts)
    k = np.arange(n_win, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    t_index = first + k * stride
    g = (np.cumsum(lengths) - lengths)[event_index] + t_index

    if n_win == 0:
        empty = np.empty((0, seq_len), dtype=np.float32) if seq_len > 0 else None
        return WindowSet(
            X=np.empty((0, hist_len, N_FEATURES), dtype=np.float32),
            y=np.empty(0, dtype=np.float32),
            driver=np.empty(0, dtype=np.int64),
            event_index=event_index,
            t_index=t_index,
            driver_names=driver_names,
            y_seq=empty,
            v_lead_future=None if empty is None else empty.copy(),
        )

    def cat(name: str) -> np.ndarray:
        return np.concatenate([getattr(ev, name) for ev in evs])

    starts = g - (hist_len - 1)
    X = np.stack(
        [sliding_window_view(cat(name).astype(np.float32), hist_len)[starts] for name in ("s", "dv", "v")],
        axis=-1,
    )
    y = cat("a")[g] if target == "a" else target_values(None, cat("v"), g, target)
    y_seq = v_lead_future = None
    if seq_len > 0:
        y_seq = np.ascontiguousarray(sliding_window_view(cat("a").astype(np.float32), seq_len)[g])
        v_lead_future = np.ascontiguousarray(sliding_window_view(cat("v_lead").astype(np.float32), seq_len)[g + 1])
    return WindowSet(
        X=X,
        y=y.astype(np.float32),
        driver=ev_driver[event_index],
        event_index=event_index,
        t_index=t_index,
        driver_names=driver_names,
        y_seq=y_seq,
        v_lead_future=v_lead_future,
    )
