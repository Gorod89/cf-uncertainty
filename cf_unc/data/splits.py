"""Driver-disjoint splits, driver bootstrap and event subsampling (docs/design.md, section 3.4).

A driver is the pair ``(dataset, follower_id)`` of an event.
"""

from __future__ import annotations

import json
from pathlib import Path

from typing import Iterable, Sequence

import numpy as np

from cf_unc.data.schema import Event, EventSet

SPLIT_NAMES = ("train", "val", "test")


def split_counts(n: int, fractions: Sequence[float] = (0.7, 0.15, 0.15)) -> np.ndarray:
    """Number of drivers per split: largest-remainder rounding of ``fractions * n`` (normalised).

    Every split with a positive fraction gets at least one driver when ``n`` allows it (the donor is
    the currently largest split).
    """
    f = np.asarray(fractions, dtype=np.float64)
    if f.shape != (len(SPLIT_NAMES),) or np.any(f < 0.0) or f.sum() <= 0.0:
        raise ValueError(f"fractions must be {len(SPLIT_NAMES)} non-negative numbers with a positive sum")
    raw = f / f.sum() * n
    counts = np.floor(raw).astype(np.int64)
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[: n - int(counts.sum())]] += 1
    if n >= int(np.count_nonzero(f)):
        for i in np.flatnonzero((f > 0.0) & (counts == 0)):
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[i] += 1
    return counts


def split_by_driver(
    events: Iterable[Event], fractions: Sequence[float] = (0.7, 0.15, 0.15), seed: int = 0
) -> dict[str, EventSet]:
    """Driver-disjoint train / val / test EventSets.

    The sorted drivers are shuffled with ``numpy.random.default_rng(seed)`` and dealt to the splits in
    the order train, val, test according to ``split_counts``; events keep their original order.
    """
    evs = list(events)
    drivers = sorted({(ev.dataset, ev.follower_id) for ev in evs})
    perm = np.random.default_rng(seed).permutation(len(drivers))
    counts = split_counts(len(drivers), fractions)
    split_of: dict[tuple[str, str], str] = {}
    bounds = np.concatenate([[0], np.cumsum(counts)])
    for name, lo, hi in zip(SPLIT_NAMES, bounds[:-1], bounds[1:]):
        for i in perm[lo:hi]:
            split_of[drivers[i]] = name
    return {name: EventSet(ev for ev in evs if split_of[(ev.dataset, ev.follower_id)] == name) for name in SPLIT_NAMES}


def bootstrap_drivers(driver_codes: np.ndarray, rng: np.random.Generator | int | None) -> np.ndarray:
    """Window indices (int64) of a with-replacement resample of drivers.

    As many drivers are drawn as there are distinct codes in ``driver_codes`` (N,); every drawn driver
    contributes all of its windows (ascending), repeated as often as it is drawn, in draw order.
    """
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    codes = np.asarray(driver_codes).ravel()
    if codes.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(codes, kind="stable")
    _, start, count = np.unique(codes[order], return_index=True, return_counts=True)
    drawn = rng.integers(0, start.size, size=start.size)
    c = count[drawn]
    shift = np.repeat(start[drawn] - (np.cumsum(c) - c), c)
    return order[np.arange(int(c.sum())) + shift].astype(np.int64)


def subsample_events(events: Iterable[Event], n: int, seed: int = 0) -> EventSet:
    """Deterministic random subset of ``min(n, len(events))`` events (original order kept)."""
    if n < 0:
        raise ValueError("n must be >= 0")
    evs = list(events)
    if n >= len(evs):
        return EventSet(evs)
    idx = np.sort(np.random.default_rng(seed).choice(len(evs), size=n, replace=False))
    return EventSet(evs[i] for i in idx)


def split_by_folds(events: Iterable[Event], fold_file: str | Path, fold: int = 0) -> dict[str, EventSet]:
    """Train / val / test of a cf-stability fold file (``data/splits/<dataset>_driver.json``).

    Rule of cf-stability: fold ``k`` = test, fold ``(k + 1) % n_folds`` = validation, the rest =
    training. Every event must be listed in the file.
    """
    spec = json.loads(Path(fold_file).read_text(encoding="utf-8"))
    n_folds, folds = int(spec["n_folds"]), spec["folds"]
    if n_folds < 3:
        raise ValueError(f"{fold_file}: {n_folds} folds cannot give train, val and test")
    test, val = int(fold) % n_folds, (int(fold) + 1) % n_folds
    parts: dict[str, list[Event]] = {"train": [], "val": [], "test": []}
    for ev in events:
        if ev.event_id not in folds:
            raise KeyError(f"event {ev.event_id!r} is not listed in {fold_file}")
        k = int(folds[ev.event_id])
        parts["test" if k == test else "val" if k == val else "train"].append(ev)
    return {name: EventSet(evs) for name, evs in parts.items()}
