"""Driver-disjoint splits, driver bootstrap, event subsampling (docs/design.md, section 3.4)."""

from __future__ import annotations

import numpy as np
import pytest

from cf_unc.data.schema import EventSet
from cf_unc.data.splits import SPLIT_NAMES, bootstrap_drivers, split_by_driver, split_counts, subsample_events
from cf_unc.data.synthetic import SyntheticConfig, generate
from cf_unc.data.windows import make_windows


def small_events(n_drivers: int, per: int = 3, dataset: str = "synthetic", seed: int = 0) -> EventSet:
    cfg = SyntheticConfig(n_drivers=n_drivers, events_per_driver=per, duration_s=4.0, dataset=dataset, seed=seed)
    es, _ = generate(cfg)
    return es


def drivers_of(es: EventSet) -> set[tuple[str, str]]:
    return {(ev.dataset, ev.follower_id) for ev in es}


def test_splits_are_driver_disjoint_and_complete():
    # two data sets that reuse the same follower ids: drivers are (dataset, follower_id)
    es = EventSet([*small_events(25, dataset="a"), *small_events(15, dataset="b", seed=1)])
    parts = split_by_driver(es, seed=3)
    assert tuple(parts) == SPLIT_NAMES
    sets = {name: drivers_of(part) for name, part in parts.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert set().union(*sets.values()) == drivers_of(es)
    assert sorted(ev.event_id for part in parts.values() for ev in part) == sorted(ev.event_id for ev in es)
    # all events of a driver land in the same split, the original order is kept
    for part in parts.values():
        ids = [ev.event_id for ev in part]
        assert ids == [ev.event_id for ev in es if ev.event_id in set(ids)]
        assert len(part) == 3 * len(drivers_of(part))


def test_fractions_are_respected():
    parts = split_by_driver(small_events(200, per=1), seed=0)
    assert [len(drivers_of(parts[name])) for name in SPLIT_NAMES] == [140, 30, 30]
    parts = split_by_driver(small_events(41, per=1), fractions=(0.8, 0.1, 0.1), seed=0)
    counts = [len(drivers_of(parts[name])) for name in SPLIT_NAMES]
    assert sum(counts) == 41 and abs(counts[0] - 32.8) <= 1 and abs(counts[1] - 4.1) <= 1 and abs(counts[2] - 4.1) <= 1


@pytest.mark.parametrize("n", [3, 4, 5, 7])
def test_every_split_non_empty(n):
    parts = split_by_driver(small_events(n, per=2), seed=n)
    assert all(len(part) > 0 for part in parts.values())


def test_split_counts_edge_cases():
    assert split_counts(0).tolist() == [0, 0, 0]
    assert split_counts(1).tolist() == [1, 0, 0]
    assert split_counts(3).tolist() == [1, 1, 1]
    assert split_counts(10, (1.0, 0.0, 1.0)).tolist() == [5, 0, 5]
    assert split_counts(20, (7, 1.5, 1.5)).tolist() == [14, 3, 3]  # un-normalised fractions
    with pytest.raises(ValueError):
        split_counts(10, (0.5, 0.5))
    with pytest.raises(ValueError):
        split_counts(10, (1.0, -0.5, 0.5))


def test_splits_are_deterministic():
    es = small_events(30)
    a, b = split_by_driver(es, seed=7), split_by_driver(es, seed=7)
    for name in SPLIT_NAMES:
        assert [ev.event_id for ev in a[name]] == [ev.event_id for ev in b[name]]
    # independent of the event order in the input
    c = split_by_driver(EventSet(reversed(list(es))), seed=7)
    assert all(drivers_of(a[name]) == drivers_of(c[name]) for name in SPLIT_NAMES)
    other = split_by_driver(es, seed=8)
    assert any(drivers_of(a[name]) != drivers_of(other[name]) for name in SPLIT_NAMES)


def test_bootstrap_drivers_properties():
    codes = np.array([3, 0, 3, 1, 1, 1, 7, 3, 0, 7, 7, 7, 7, 1])  # 4 drivers, windows interleaved
    size = {c: int(np.sum(codes == c)) for c in np.unique(codes)}
    for seed in range(20):
        idx = bootstrap_drivers(codes, np.random.default_rng(seed))
        assert idx.dtype == np.int64 and idx.min() >= 0 and idx.max() < codes.size
        drawn = codes[idx]
        # every drawn driver contributes all of its windows, as often as it was drawn
        for c in np.unique(drawn):
            reps = np.sum(drawn == c) / size[c]
            assert reps == int(reps) and reps >= 1
            np.testing.assert_array_equal(np.bincount(idx[drawn == c], minlength=codes.size)[codes == c], int(reps))
        assert sum(np.sum(drawn == c) / size[c] for c in np.unique(drawn)) == len(size)  # as many draws as drivers
        # blocks are contiguous and ascending within a driver
        blocks = np.split(idx, np.flatnonzero(np.diff(drawn) != 0) + 1)
        for block in blocks:
            for rep in np.split(block, len(block) // size[int(codes[block[0]])]):
                np.testing.assert_array_equal(rep, np.flatnonzero(codes == codes[rep[0]]))
    np.testing.assert_array_equal(bootstrap_drivers(codes, 5), bootstrap_drivers(codes, np.random.default_rng(5)))
    assert bootstrap_drivers(np.array([], dtype=np.int64), 0).size == 0


def test_bootstrap_drivers_on_windows():
    es = small_events(60, per=2)
    w = make_windows(es, hist_len=10)
    per_driver = np.bincount(w.driver)
    n_drivers = int(np.count_nonzero(per_driver))
    assert n_drivers == 60
    distinct = []
    for seed in range(30):
        idx = bootstrap_drivers(w.driver, seed)
        reps = np.bincount(w.driver[idx], minlength=per_driver.size) / per_driver
        np.testing.assert_array_equal(reps, np.round(reps))  # whole drivers only
        assert reps.sum() == n_drivers  # n drivers drawn with replacement
        distinct.append(int(np.count_nonzero(reps)))
    # expected share of distinct drivers in a bootstrap resample is 1 - (1 - 1/n)^n ~ 0.63
    assert 0.55 < np.mean(distinct) / n_drivers < 0.72


def test_subsample_events():
    es = small_events(20)
    sub = subsample_events(es, 15, seed=2)
    assert len(sub) == 15
    ids = [ev.event_id for ev in sub]
    assert ids == [ev.event_id for ev in es if ev.event_id in set(ids)]  # original order
    assert ids == [ev.event_id for ev in subsample_events(es, 15, seed=2)]
    assert ids != [ev.event_id for ev in subsample_events(es, 15, seed=3)]
    assert len(subsample_events(es, 1000, seed=0)) == len(es)
    assert len(subsample_events(es, 0, seed=0)) == 0
    with pytest.raises(ValueError):
        subsample_events(es, -1)
