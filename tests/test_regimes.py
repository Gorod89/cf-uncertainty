"""Tests of cf_unc.maps.regimes: definitions, priority, partition, NaN handling, lagged acceleration."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.signal import lfilter

from cf_unc.maps.analysis import load_config
from cf_unc.maps.regimes import PRIORITY, REGIMES, RegimeThresholds, label_regimes, regime_masks, regime_shares

V0 = 30.0  # free speed; 0.8 v0 = 24 m/s
TH = RegimeThresholds()
IDX = {name: i for i, name in enumerate(REGIMES)}


def _naive_label(s: float, dv: float, v: float, a: float, v0: float, th: RegimeThresholds, priority=PRIORITY) -> int:
    """Row-by-row reference implementation of design.md 10.5 (Python comparisons with NaN are False)."""

    def holds(name: str) -> bool:
        if name == "congested":
            return s < th.s_congested and v < th.v_congested
        if name == "braking":
            return a < th.a_brake
        if name == "approaching":
            return dv > th.dv_eps
        if name == "free":
            return s > th.s_free or (abs(dv) < th.dv_eps and v > th.v_free_frac * v0)
        if name == "steady":
            return abs(dv) < th.dv_eps and th.s_lo < s < th.s_free
        raise KeyError(name)

    for name in priority:
        if holds(name):
            return IDX[name]
    return IDX["other"]


def test_defaults_match_config() -> None:
    section = load_config()["regimes"]
    assert RegimeThresholds.from_config(section) == TH
    assert RegimeThresholds.from_config({"regimes": section}) == TH
    assert tuple(section["priority"]) == PRIORITY and section["reference"] == "steady"
    assert REGIMES == ("congested", "braking", "approaching", "free", "steady", "other")


@pytest.mark.parametrize(
    "state, expected",
    [
        ((10.0, 0.0, 5.0, 0.0), {"congested"}),  # s = 10 is not > 10: not steady
        ((14.9, 0.0, 7.9, 0.0), {"congested", "steady"}),
        ((15.0, 0.0, 5.0, 0.0), {"steady"}),  # s < 15 is strict
        ((12.0, 1.0, 8.0, 0.0), {"approaching"}),  # v < 8 is strict
        ((30.0, 0.0, 15.0, -0.6), {"braking", "steady"}),
        ((30.0, 0.0, 15.0, -0.5), {"steady"}),  # a_prev < -0.5 is strict
        ((30.0, 0.6, 15.0, 0.0), {"approaching"}),
        ((30.0, 0.5, 15.0, 0.0), {"other"}),  # neither |dv| < 0.5 nor dv > 0.5
        ((60.1, -2.0, 15.0, 0.0), {"free"}),  # large gap
        ((60.0, -2.0, 15.0, 0.0), {"other"}),  # s > 60 is strict, dv not calm
        ((40.0, 0.2, 25.0, 0.0), {"free", "steady"}),  # calm at more than 0.8 v0
        ((40.0, 0.2, 24.0, 0.0), {"steady"}),  # v > 0.8 v0 is strict
        ((80.0, 1.0, 30.0, -1.0), {"free", "approaching", "braking"}),
        ((5.0, -1.0, 10.0, 0.0), {"other"}),  # opening gap at a small gap, not congested (v >= 8)
        ((30.0, 0.0, 15.0, math.nan), {"steady"}),  # NaN a_prev is never braking
    ],
)
def test_each_definition_on_hand_made_states(state: tuple[float, ...], expected: set[str]) -> None:
    s, dv, v, a = (np.array([x]) for x in state)
    masks = regime_masks(s, dv, v, a, V0, TH)
    assert {name for name, m in masks.items() if m[0]} == expected
    label = label_regimes(s, dv, v, a, V0, TH)[0]
    assert label == _naive_label(*state, V0, TH)


def test_priority_order() -> None:
    #                  congested+braking+appr.  braking+approaching  approaching+free   free+steady
    s = np.array([10.0, 30.0, 80.0, 40.0])
    dv = np.array([1.0, 1.0, 1.0, 0.2])
    v = np.array([5.0, 15.0, 30.0, 25.0])
    a = np.array([-2.0, -2.0, 0.0, 0.0])
    labels = label_regimes(s, dv, v, a, V0)
    assert [REGIMES[i] for i in labels] == ["congested", "braking", "approaching", "free"]
    custom = label_regimes(s, dv, v, a, V0, priority=("steady", "free", "approaching", "braking", "congested"))
    assert [REGIMES[i] for i in custom] == ["approaching", "approaching", "free", "steady"]
    only = label_regimes(s, dv, v, a, V0, priority=("steady",))  # regimes left out are never assigned
    assert [REGIMES[i] for i in only] == ["other", "other", "other", "steady"]
    for bad in (("other",), ("steady", "steady"), ("cruising",)):
        with pytest.raises(ValueError):
            label_regimes(s, dv, v, a, V0, priority=bad)
    with pytest.raises(ValueError):
        regime_masks(s, dv, v, a, None)  # type: ignore[arg-type]


def _random_states(rng: np.random.Generator, n: int) -> tuple[np.ndarray, ...]:
    """Random states with about 15 % of the values exactly on a threshold, NaN accelerations included."""

    def mix(values: np.ndarray, special: list[float]) -> np.ndarray:
        pick = rng.random(n) < 0.15
        values[pick] = rng.choice(special, pick.sum())
        return values

    s = mix(rng.uniform(0.0, 100.0, n), [10.0, 15.0, 60.0])
    dv = mix(rng.normal(0.0, 1.5, n), [-0.5, 0.5, 0.0])
    v = mix(rng.uniform(0.0, 35.0, n), [8.0, 24.0])
    a = mix(rng.normal(0.0, 1.0, n), [-0.5, math.nan])
    return s, dv, v, a


def test_partition_covers_every_row_exactly_once() -> None:
    rng = np.random.default_rng(7)
    s, dv, v, a = _random_states(rng, 10_000)
    labels = label_regimes(s, dv, v, a, V0)
    masks = regime_masks(s, dv, v, a, V0)
    assert labels.dtype == np.int64 and labels.min() >= 0 and labels.max() < len(REGIMES)
    one_hot = np.zeros((len(REGIMES), labels.size), dtype=bool)
    one_hot[labels, np.arange(labels.size)] = True
    assert np.all(one_hot.sum(axis=0) == 1)  # every row in exactly one regime
    for i, name in enumerate(REGIMES):
        rows = labels == i
        assert np.all(masks[name][rows])  # the label's own mask holds
        for earlier in PRIORITY[: PRIORITY.index(name)] if name in PRIORITY else PRIORITY:
            assert not np.any(masks[earlier][rows])  # no regime of higher priority holds
    assert np.array_equal(labels == IDX["other"], masks["other"])
    shares = regime_shares(labels)
    assert shares.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(shares, np.bincount(labels, minlength=6) / labels.size)
    assert np.all(regime_shares(np.array([], dtype=np.int64)) == 0)


def test_nan_a_prev_never_braking() -> None:
    n = 50
    s, dv, v = np.full(n, 30.0), np.zeros(n), np.full(n, 15.0)
    a = np.full(n, math.nan)
    assert not regime_masks(s, dv, v, a, V0)["braking"].any()
    assert np.all(label_regimes(s, dv, v, a, V0) == IDX["steady"])
    s_nan = s.copy()
    s_nan[0] = math.nan  # a NaN state satisfies no definition
    assert label_regimes(s_nan, dv, v, a, V0)[0] == IDX["other"]


def test_vectorised_matches_rowwise_reference() -> None:
    rng = np.random.default_rng(11)
    s, dv, v, a = _random_states(rng, 10_000)
    th = RegimeThresholds(s_congested=12.0, v_congested=6.0, s_free=70.0, s_lo=8.0, dv_eps=0.4, a_brake=-0.8)
    for thresholds, priority in ((TH, PRIORITY), (th, ("free", "steady", "braking", "congested", "approaching"))):
        fast = label_regimes(s, dv, v, a, V0, thresholds, priority)
        slow = np.array([_naive_label(*row, V0, thresholds, priority) for row in zip(s, dv, v, a)])
        np.testing.assert_array_equal(fast, slow)
    # float32 inputs (as stored in the windows table) give the same labels as their float64 values
    s32, dv32, v32, a32 = (x.astype(np.float32) for x in (s, dv, v, a))
    fast32 = label_regimes(s32, dv32, v32, a32, V0)
    slow32 = np.array([_naive_label(*map(float, row), V0, TH) for row in zip(s32, dv32, v32, a32)])
    np.testing.assert_array_equal(fast32, slow32)


def test_lagged_acceleration_instead_of_the_target() -> None:
    """Labelling with the target a[t] instead of a[t-1] changes the labels and truncates the target."""
    rng = np.random.default_rng(3)
    n = 20_000
    acc = lfilter([1.0], [1.0, -0.6], rng.normal(0.0, 0.8, n + 1))  # AR(1) acceleration, stationary sd 1
    a_prev, y = acc[:-1], acc[1:]
    s, dv, v = np.full(n, 30.0), np.zeros(n), np.full(n, 15.0)  # steady unless braking
    lab_prev = label_regimes(s, dv, v, a_prev, V0)
    lab_target = label_regimes(s, dv, v, y, V0)
    assert (lab_prev != lab_target).mean() > 0.1  # 0.26 with this seed (theory: 2 P(a_prev < -0.5, y >= -0.5))
    braking = IDX["braking"]
    # conditioning on a label computed from the target truncates it: every target is below -0.5 ...
    assert y[lab_target == braking].max() < -0.5
    # ... while the label known before the outcome leaves the target law intact (0.42 here, theory 0.42)
    assert (y[lab_prev == braking] >= -0.5).mean() > 0.3
    # the truncation also shrinks the residual variance of the regime (ratio 0.36 here)
    assert y[lab_target == braking].var() < 0.7 * y[lab_prev == braking].var()
