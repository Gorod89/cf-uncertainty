"""Glue: events -> driver-disjoint splits -> windows (+ ground truth aligned with the windows).

A pooled training domain (design.md 16.1, experiment E6) is cut by ``build_pooled_bundle``: the windows of every data
set, each capped by ``cap_windows`` (uniform subsample with a fixed seed after the stride), concatenated
(``cf_unc.data.windows.concat_windows``).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import savgol_filter

from cf_unc.constants import DT, HIST_LEN
from cf_unc.data.schema import EventSet
from cf_unc.data.splits import split_by_driver, split_by_folds
from cf_unc.data.synthetic import SyntheticConfig, generate
from cf_unc.data.windows import WindowSet, concat_windows, make_windows

SPLITS = ("train", "val", "test")
POOL_SPLITS = ("train", "val")  # parts of a pooled training bundle (the test folds are evaluated per data set)


@dataclass
class DataBundle:
    """One data set, split by driver and cut into windows.

    events[split]   EventSet
    windows[split]  WindowSet
    truth[split]    dict of (N,) arrays aligned with the windows (``a_det``, ``sigma_true``); synthetic data only
    """

    name: str
    events: dict[str, EventSet]
    windows: dict[str, WindowSet]
    truth: dict[str, dict[str, np.ndarray]] | None = None
    meta: dict[str, Any] | None = None


def align_truth(events: EventSet, windows: WindowSet, truth: Mapping[str, Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Per-window ground truth: value at ``t_index`` of the source event, for every array-valued key."""
    if len(windows) == 0:
        return {}
    ids = [ev.event_id for ev in events]
    keys = [k for k, v in truth[ids[0]].items() if isinstance(v, np.ndarray)]
    out = {k: np.empty(len(windows), dtype=np.float64) for k in keys}
    order = np.argsort(windows.event_index, kind="stable")
    bounds = np.flatnonzero(np.diff(windows.event_index[order])) + 1
    for rows in np.split(order, bounds):
        record = truth[ids[int(windows.event_index[rows[0]])]]
        for k in keys:
            out[k][rows] = record[k][windows.t_index[rows]]
    return out


def savgol_speed(v: np.ndarray, window: int = 11, order: int = 2) -> np.ndarray:
    """Savitzky-Golay filtered speed (mode ``"interp"``) along the last axis [m/s], float64, shape of ``v``.

    ``v``: (n,) or (k, n) speeds of k events of equal length. Events shorter than ``window`` samples use the
    longest odd window that fits; no filtering (a copy) when that window is not longer than ``order``
    (cf-stability, ``data/processing.py``).
    """
    v = np.asarray(v, dtype=np.float64)
    n = v.shape[-1]
    if n < window:
        window = n if n % 2 else n - 1
    return savgol_filter(v, window, order, mode="interp", axis=-1) if window > order else v.copy()


def central_differences(x: np.ndarray, dt: float = DT) -> np.ndarray:
    """Time derivative along the last axis (``x`` (n,) or (k, n), step ``dt`` [s]): central differences inside,
    second-order one-sided differences at both ends (``np.gradient``); n = 2: the one difference; n < 2: zeros."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    if n >= 3:
        return np.gradient(x, dt, axis=-1, edge_order=2)
    if n == 2:
        return np.repeat((x[..., 1:] - x[..., :1]) / dt, 2, axis=-1)
    return np.zeros_like(x)


def acceleration_from_speed(v: np.ndarray, dt: float = DT, window: int = 11, order: int = 2) -> np.ndarray:
    """Central differences of the Savitzky-Golay filtered speed, as in cf-stability (``data/processing.py``).

    ``v``: (n,) [m/s] (or (k, n), events of equal length); returns m/s^2 of the same shape.
    """
    return central_differences(savgol_speed(v, window, order), dt)


def unify_acceleration(events: EventSet) -> tuple[EventSet, int]:
    """One acceleration definition for every data set (design.md, section 10.1).

    Events whose ``meta["a_source"]`` is not ``"savgol"`` get ``a`` recomputed from the stored speed;
    the original source is kept in ``meta["a_source_original"]``. Returns the set and the number of
    changed events. Events are modified in place.
    """
    changed = 0
    for ev in events:
        if ev.meta.get("a_source", "savgol") != "savgol":
            ev.a = np.ascontiguousarray(acceleration_from_speed(ev.v), dtype=np.float64)
            ev.meta["a_source_original"] = ev.meta["a_source"]
            ev.meta["a_source"] = "savgol"
            changed += 1
    return events, changed


def build_bundle(
    events: EventSet,
    truth: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    name: str,
    hist_len: int = HIST_LEN,
    stride: int = 1,
    seq_len: int = 0,
    target: str = "a",
    split_seed: int = 0,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    fold_file: str | Path | None = None,
    fold: int = 0,
    meta: dict[str, Any] | None = None,
) -> DataBundle:
    """Split by driver (seeded 70/15/15, or the folds of ``fold_file``) and cut every part into windows.

    ``target`` goes to ``make_windows`` (``"a"``, ``"fwd_diff"``, ``"mean_1s"``); the aligned ground truth
    (``a_det``, ``sigma_true`` at the last history sample) describes the one-step law whatever the target.
    """
    if fold_file is not None:
        parts = split_by_folds(events, fold_file, fold)
    else:
        parts = split_by_driver(events, fractions=fractions, seed=split_seed)
    windows = {k: make_windows(parts[k], hist_len=hist_len, stride=stride, seq_len=seq_len, target=target) for k in SPLITS}
    aligned = None if truth is None else {k: align_truth(parts[k], windows[k], truth) for k in SPLITS}
    return DataBundle(name=name, events=parts, windows=windows, truth=aligned, meta=meta)


def subsample_seed(seed: int, name: str, part: str) -> int:
    """Seed of the window subsample of data set ``name``, part ``part`` (``train`` / ``val``): ``seed`` mixed with the
    two names, so that the subsample of a data set does not depend on the other data sets of a pool (a LODO pool
    keeps the subsamples of the full pool)."""
    text = f"{name}\x1f{part}".encode("utf-8")
    return int(np.random.SeedSequence([abs(int(seed)), zlib.crc32(text)]).generate_state(1)[0])


def cap_windows(windows: WindowSet, cap: int | None, seed: int) -> WindowSet:
    """At most ``cap`` windows: a uniform draw without replacement from ``numpy.random.default_rng(seed)``, rows in
    their original order (events and window ends stay sorted); all windows (the same object) when ``cap`` is None or
    not exceeded."""
    if cap is None or len(windows) <= int(cap):
        return windows
    if int(cap) < 0:
        raise ValueError(f"cap must be >= 0, got {cap}")
    rows = np.sort(np.random.default_rng(int(seed)).choice(len(windows), size=int(cap), replace=False))
    return windows.subset(rows)


def build_pooled_bundle(
    parts: Mapping[str, Mapping[str, EventSet]],
    *,
    hist_len: int = HIST_LEN,
    stride: int = 1,
    seq_len: int = 0,
    target: str = "a",
    max_windows: int | None = None,
    seed: int = 0,
    splits: Sequence[str] = POOL_SPLITS,
    name: str | None = None,
) -> DataBundle:
    """Pooled training bundle of several data sets (design.md 16.1).

    parts: data set -> split -> EventSet (the folds of each data set, e.g. ``split_by_folds``), in the order of the
    pool. Per data set and split the windows (``make_windows`` with ``stride``, ``seq_len``, ``target``) are capped at
    ``max_windows`` (``cap_windows`` with ``subsample_seed(seed, data set, split)``), then concatenated in the order
    of ``parts`` (``concat_windows``: the events of a split are the concatenation of the data sets' events,
    ``event_index`` refers to it, driver codes to the union of the drivers). Only ``splits`` are cut (default train
    and val). ``meta["pool"]``: per data set and split ``events``, ``windows`` (after the stride) and ``kept``
    (after the cap); ``meta`` also holds ``max_windows`` and ``seed``.
    """
    if not parts:
        raise ValueError("build_pooled_bundle needs at least one data set")
    events: dict[str, EventSet] = {}
    windows: dict[str, WindowSet] = {}
    counts: dict[str, dict[str, dict[str, int]]] = {str(n): {} for n in parts}
    for split in splits:
        pieces, n_events, evs = [], [], []
        for set_name, by_split in parts.items():
            split_events = list(by_split[split])
            full = make_windows(split_events, hist_len=hist_len, stride=stride, seq_len=seq_len, target=target)
            kept = cap_windows(full, max_windows, subsample_seed(seed, str(set_name), split))
            counts[str(set_name)][split] = {"events": len(split_events), "windows": len(full), "kept": len(kept)}
            del full
            pieces.append(kept)
            n_events.append(len(split_events))
            evs += split_events
        events[split] = EventSet(evs)
        windows[split] = concat_windows(pieces, n_events)
    meta = {"pool": counts, "max_windows": None if max_windows is None else int(max_windows), "seed": int(seed)}
    return DataBundle(name=name or "+".join(str(n) for n in parts), events=events, windows=windows, meta=meta)


def synthetic_bundle(cfg: SyntheticConfig | Mapping[str, Any] | None = None, **kwargs: Any) -> DataBundle:
    """Generate StochasticIDM data and cut it (``kwargs`` go to ``build_bundle``)."""
    if not isinstance(cfg, SyntheticConfig):
        cfg = SyntheticConfig.from_dict(cfg)
    events, truth = generate(cfg)
    kwargs.setdefault("name", cfg.dataset)
    return build_bundle(events, truth, meta={"generator": cfg.to_dict()}, **kwargs)


def load_events(
    directory: str | Path,
    unify: bool = True,
    signal: str = "raw",
    harmonise: Any = None,
    return_report: bool = False,
) -> EventSet | tuple[EventSet, dict[str, Any] | None]:
    """EventSet of a parquet directory, with the common acceleration definition when ``unify``.

    ``signal``: ``"raw"`` (the stored signals) or ``"harmonised"`` (one signal processing for every data set,
    ``cf_unc.data.harmonise.harmonise_events`` with the settings ``harmonise``, a mapping or
    ``HarmoniseConfig``; dropped events are left out). The variant ``"matched"`` (design.md 15.1) is the harmonised
    processing with the data set's matched window: the caller resolves it (``cf_unc.experiments.e1.signal_key``)
    and passes ``"harmonised"`` with that window; ``"matched"`` itself is refused here. The stored events are never
    modified. With ``return_report`` the result is ``(events, report)``, report None for raw signals.
    """
    from cf_unc.data.harmonise import check_signal, harmonise_events  # harmonise imports this module

    if check_signal(str(signal)) == "matched":
        raise ValueError("signal 'matched' needs the window of the data set: pass signal='harmonised' with the matched "
                         "window (cf_unc.experiments.e1.signal_key resolves it from the matching file)")  # fmt: skip
    events = EventSet.from_parquet(Path(directory))
    if unify:
        unify_acceleration(events)
    report = None
    if str(signal) == "harmonised":
        events, report = harmonise_events(events, harmonise)
    return (events, report) if return_report else events


def load_bundle(
    directory: str | Path, *, name: str | None = None, unify: bool = True, signal: str = "raw", harmonise: Any = None,
    **kwargs: Any,
) -> DataBundle:  # fmt: skip
    """Read an EventSet directory (parquet layout shared with cf-stability), optionally harmonised, and cut it."""
    directory = Path(directory)
    events = load_events(directory, unify=unify, signal=signal, harmonise=harmonise)
    meta = {"source": str(directory), **({"signal": signal} if signal != "raw" else {})}
    return build_bundle(events, None, name=name or directory.name, meta=meta, **kwargs)  # type: ignore[arg-type]
