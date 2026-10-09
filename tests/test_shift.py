"""Tests of cf_unc.maps.shift: event bootstrap on sufficient statistics, shift ratios, bin matching,
concentration, across-seed decomposition and the H3.1 verdict / summary.

Data and bootstrap draws use fixed seeds (deterministic); statistical tolerances state their margin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cf_unc.dist import MixtureDist, decompose
from cf_unc.maps import shift as S
from cf_unc.maps.analysis import load_config
from cf_unc.maps.grid import StateGrid

NAN = np.nan


def _events(
    rng: np.random.Generator, n_events: int, level: float = 1.0, cv: float = 0.5, size: tuple[int, int] = (20, 60)
) -> tuple[np.ndarray, np.ndarray]:
    """Positive values with within-event correlation: level x event effect (mean 1, CV cv) x window noise (CV 0.5)."""
    sizes = rng.integers(size[0], size[1], n_events)
    ev = np.repeat(np.arange(n_events), sizes)
    shape = 1.0 / cv**2
    effect = rng.gamma(shape, 1.0 / shape, n_events)
    return level * effect[ev] * rng.gamma(4.0, 0.25, ev.size), ev


def _width(res: dict) -> float:
    return float(res["hi"] - res["lo"])


# ------------------------------------------------------------------------------- machinery
def test_event_sums_encoding_and_resampled_sums() -> None:
    x = np.array([1.0, 2.0, NAN, 4.0, 5.0])
    codes = np.array([0, 0, 1, 1, 2])
    sums, counts = S.event_sums(x, codes, 4)
    np.testing.assert_array_equal(sums, [3.0, 4.0, 5.0, 0.0])
    np.testing.assert_array_equal(counts, [2, 1, 1, 0])
    sums2, counts2 = S.event_sums(np.vstack([x, 2 * x]), codes, 4)
    np.testing.assert_array_equal(sums2, [[3, 4, 5, 0], [6, 8, 10, 0]])
    np.testing.assert_array_equal(counts2, [[2, 1, 1, 0], [2, 1, 1, 0]])
    with pytest.raises(ValueError):
        S.event_sums(x, codes, 2)
    enc, units = S.encode_units(np.array([7, 3, 7, 12], dtype=np.int32))
    np.testing.assert_array_equal(enc, [1, 0, 1, 2])
    np.testing.assert_array_equal(units, [3, 7, 12])
    enc_s, units_s = S.encode_units(np.array(["b", "a", "b"]))
    np.testing.assert_array_equal(enc_s, [1, 0, 1])
    assert list(units_s) == ["a", "b"]
    # every replicate draws n_units units; several blocks
    blocks = list(S.multiplicities(50, 30, seed=1, block=200))
    assert len(blocks) == 8 and all(np.all(c.sum(axis=1) == 50) for _, c in blocks)
    rng = np.random.default_rng(0)
    stats = rng.random((50, 3))
    dense = S.resample_sums(stats, 30, seed=2, block=200)
    manual = np.vstack([c @ stats for _, c in S.multiplicities(50, 30, seed=2, block=200)])
    np.testing.assert_allclose(dense, manual, rtol=1e-12)
    np.testing.assert_allclose(S.resample_sums(sparse.csr_array(stats), 30, seed=2, block=200), dense, rtol=1e-12)
    # two-sided percentile p-value with +1 smoothing
    assert S.bootstrap_pvalue(np.full(99, 1.0)) == pytest.approx(0.02)
    assert S.bootstrap_pvalue(np.linspace(-1, 1, 101)) == pytest.approx(1.0)
    assert np.isnan(S.bootstrap_pvalue(np.full(5, NAN)))


# ---------------------------------------------------------------------------- shift ratio
def test_shift_ratio_recovers_known_ratio() -> None:
    rng = np.random.default_rng(1)
    x_in, ev_in = _events(rng, 400)
    x_d, ev_d = _events(rng, 300, level=3.0)
    res = S.shift_ratio(x_d, ev_d, x_in, ev_in, n_boot=1000, seed=5)
    assert res["estimate"] == pytest.approx(x_d.mean() / x_in.mean(), rel=1e-12)
    # the bootstrap se is ~4 % of 3 (event CV 0.5 over 300-400 events); 3 se is the margin
    assert abs(res["estimate"] - 3.0) < 3 * res["se"]
    assert res["lo"] <= 3.0 <= res["hi"]
    assert 0.03 < res["se"] / res["estimate"] < 0.06
    assert (res["n_events_shift"], res["n_events_in"], res["n_seeds"]) == (300, 400, 1)
    assert res["replicates"].shape == (1000,) and np.isnan(res["seed_sd"])


def test_interval_width_matches_rowwise_event_bootstrap() -> None:
    """The sufficient-statistic bootstrap equals resampling whole events row by row (in distribution)."""
    rng = np.random.default_rng(21)
    x_in, ev_in = _events(rng, 150)
    x_d, ev_d = _events(rng, 120, level=2.0)
    xs_in = np.vstack([x_in, x_in * rng.gamma(20.0, 0.05, x_in.size)])  # two seeds
    xs_d = np.vstack([x_d, x_d * rng.gamma(20.0, 0.05, x_d.size)])
    n_boot = 2000
    res = S.shift_ratio(xs_d, ev_d, xs_in, ev_in, n_boot=n_boot, seed=3)
    rows_d = np.split(np.arange(ev_d.size), np.cumsum(np.bincount(ev_d))[:-1])
    rows_in = np.split(np.arange(ev_in.size), np.cumsum(np.bincount(ev_in))[:-1])
    brng = np.random.default_rng(99)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx_d = np.concatenate([rows_d[e] for e in brng.integers(0, 120, 120)])
        idx_in = np.concatenate([rows_in[e] for e in brng.integers(0, 150, 150)])
        reps[b] = xs_d[:, idx_d].mean(axis=1).mean() / xs_in[:, idx_in].mean(axis=1).mean()
    lo, hi = np.quantile(reps, [0.025, 0.975])
    # Monte Carlo sd of each width ~2.5 % at 2000 replicates, of their ratio ~3.5 %: 15 % is > 4 sd
    assert abs(_width(res) / (hi - lo) - 1.0) < 0.15
    assert abs(np.median(res["replicates"]) / np.median(reps) - 1.0) < 0.02


def test_common_random_numbers_across_seeds() -> None:
    rng = np.random.default_rng(5)
    x_in, ev_in = _events(rng, 200)
    x_d, ev_d = _events(rng, 150, level=2.0)
    single = S.shift_ratio(x_d, ev_d, x_in, ev_in, n_boot=500, seed=8)
    # identical seeds: the same resampled events for every seed, so every replicate equals the single seed's
    same = S.shift_ratio(np.vstack([x_d] * 3), ev_d, np.vstack([x_in] * 3), ev_in, n_boot=500, seed=8)
    np.testing.assert_allclose(same["replicates"], single["replicates"], rtol=1e-12)
    assert same["seed_sd"] == 0.0 and same["n_seeds"] == 3
    # seeds with independent event-level noise (log-sd 0.6, mean 1): averaging the seeds removes most of it
    def noisy(x: np.ndarray, ev: np.ndarray) -> np.ndarray:
        return np.vstack([x * np.exp(0.6 * rng.standard_normal(ev.max() + 1)[ev] - 0.18) for _ in range(5)])

    xs_d, xs_in = noisy(x_d, ev_d), noisy(x_in, ev_in)
    avg = S.shift_ratio(xs_d, ev_d, xs_in, ev_in, n_boot=1000, seed=8)
    one = S.shift_ratio(xs_d[:1], ev_d, xs_in[:1], ev_in, n_boot=1000, seed=8)
    # width ratio 0.59 here (rough expectation sqrt((0.25 + 0.43 / 5) / (0.25 + 0.43)) = 0.7); MC error ~5 %
    assert _width(avg) < 0.85 * _width(one)
    assert avg["estimate"] == pytest.approx(xs_d.mean(axis=1).mean() / xs_in.mean(axis=1).mean(), rel=1e-12)
    np.testing.assert_allclose(avg["per_seed"], xs_d.mean(axis=1) / xs_in.mean(axis=1), rtol=1e-12)
    assert avg["seed_sd"] == pytest.approx(np.std(avg["per_seed"], ddof=1), rel=1e-12)


def test_shift_ratio_nan_rows_and_seeds() -> None:
    rng = np.random.default_rng(9)
    x_in, ev_in = _events(rng, 100)
    x_d, ev_d = _events(rng, 80, level=2.0)
    xs_d = np.vstack([x_d, 1.1 * x_d])
    xs_in = np.vstack([x_in, 1.1 * x_in])
    xs_d[:, ev_d < 10] = NAN  # 10 events without any evaluated window: left out of the resampling
    xs_d[1, ::3] = NAN  # the second seed misses a third of the rows
    res = S.shift_ratio(xs_d, ev_d, xs_in, ev_in, n_boot=200, seed=1)
    means_d = np.array([np.nanmean(r) for r in xs_d])
    assert res["estimate"] == pytest.approx(means_d.mean() / xs_in.mean(axis=1).mean(), rel=1e-12)
    assert res["n_events_shift"] == 70 and res["n_events_in"] == 100
    assert np.isfinite(res["replicates"]).all()


def test_collapsing_seeds_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seeds that miss the same windows are resampled as their seed mean: identical replicates, K times cheaper."""
    rng = np.random.default_rng(31)
    x_in, ev_in = _events(rng, 80)
    x_d, ev_d = _events(rng, 60, level=2.0)
    xs_in = np.vstack([x_in * rng.gamma(10.0, 0.1, x_in.size) for _ in range(4)])
    xs_d = np.vstack([x_d * rng.gamma(10.0, 0.1, x_d.size) for _ in range(4)])
    xs_d[:, ::5] = NAN  # the same missing windows in every seed
    bins_in, bins_d = rng.integers(0, 40, x_in.size), rng.integers(0, 40, x_d.size)
    mask = rng.random(x_d.size) < 0.3
    assert S._collapse(xs_d).shape == (1, x_d.size) and S._collapse(xs_d[:, :10]).shape == (1, 10)
    ragged = xs_d.copy()
    ragged[0, 1] = NAN
    assert S._collapse(ragged).shape == ragged.shape  # different missing windows: no collapse

    def run() -> list[np.ndarray]:
        return [
            S.shift_ratio(xs_d, ev_d, xs_in, ev_in, n_boot=300, seed=4)["replicates"],
            S.matched_shift_ratio(xs_d, bins_d, ev_d, xs_in, bins_in, ev_in, 480, 5, n_boot=300, seed=4)["replicates"],
            S.concentration(xs_d, mask, ev_d, n_boot=300, seed=4)["replicates"],
            S.bootstrap_seed_means({"x": xs_d}, ev_d, n_boot=300, seed=4)["x"]["replicates"],
        ]

    fast = run()
    monkeypatch.setattr(S, "_collapse", lambda x: x)
    slow = run()
    for a, b in zip(fast, slow):
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, rtol=1e-10)


# --------------------------------------------------------------------------- matched ratio
def test_matched_ratio_removes_a_pure_state_distribution_shift() -> None:
    rng = np.random.default_rng(13)
    grid = StateGrid()
    f = 1.0 + (np.arange(grid.n_cells) % 7)  # a quantity that depends on the bin only
    ev_in = np.repeat(np.arange(300), 30)
    bins_in = rng.integers(0, 100, ev_in.size)
    high = np.flatnonzero(f[:100] >= 5)
    ev_d = np.repeat(np.arange(200), 30)
    bins_d = np.where(rng.random(ev_d.size) < 0.9, rng.choice(high, ev_d.size), rng.integers(0, 100, ev_d.size))
    x_in, x_d = f[bins_in], f[bins_d]
    plain = S.shift_ratio(x_d, ev_d, x_in, ev_in, n_boot=300, seed=1)
    matched = S.matched_shift_ratio(x_d, bins_d, ev_d, x_in, bins_in, ev_in, grid, min_count=30, n_boot=300, seed=1)
    assert plain["estimate"] > 1.3 and plain["lo"] > 1.2  # expected 5.8 / 4.0 = 1.45
    assert matched["estimate"] == pytest.approx(1.0, abs=1e-12)
    assert matched["lo"] == pytest.approx(1.0, abs=1e-12) and matched["hi"] == pytest.approx(1.0, abs=1e-12)
    cnt_in = np.bincount(bins_in, minlength=grid.n_cells)[: grid.n_bins]
    cnt_d = np.bincount(bins_d, minlength=grid.n_cells)[: grid.n_bins]
    both = (cnt_in >= 30) & (cnt_d >= 30)
    assert matched["n_bins_matched"] == both.sum() == high.size
    assert matched["mass_in"] == pytest.approx(cnt_in[both].sum() / ev_in.size, rel=1e-12)
    assert matched["mass_shift"] == pytest.approx(cnt_d[both].sum() / ev_d.size, rel=1e-12)
    # with event-level and window noise the matched ratio stays near 1 while the plain one does not
    noise_in = rng.gamma(10.0, 0.1, 300)[ev_in] * rng.gamma(4.0, 0.25, ev_in.size)
    noise_d = rng.gamma(10.0, 0.1, 200)[ev_d] * rng.gamma(4.0, 0.25, ev_d.size)
    noisy = S.matched_shift_ratio(
        x_d * noise_d, bins_d, ev_d, x_in * noise_in, bins_in, ev_in, grid, min_count=30, n_boot=500, seed=2
    )
    # se ~3 % (event CV 0.32 over 200-300 events): 0.1 is > 3 se
    assert abs(noisy["estimate"] - 1.0) < 0.1 and noisy["lo"] < 1.0 < noisy["hi"]
    assert S.shift_ratio(x_d * noise_d, ev_d, x_in * noise_in, ev_in, n_boot=500, seed=2)["lo"] > 1.2
    # outside cell and sparse bins are never matched
    none = S.matched_shift_ratio(x_d, np.full(ev_d.size, 480), ev_d, x_in, bins_in, ev_in, grid, 30, 50, 0)
    assert none["n_bins_matched"] == 0 and np.isnan(none["estimate"]) and none["mass_shift"] == 0.0
    with pytest.raises(ValueError):
        S.matched_shift_ratio(x_d, bins_d + 1000, ev_d, x_in, bins_in, ev_in, grid)


def test_batched_quantities_equal_single_calls() -> None:
    """Several quantities share the draws (and count blocks when they miss the same windows) without changing
    any number: batched results equal one call per quantity."""
    rng = np.random.default_rng(41)
    x_in, ev_in = _events(rng, 90)
    x_d, ev_d = _events(rng, 70, level=1.5)
    bins_in, bins_d = rng.integers(0, 30, x_in.size), rng.integers(0, 30, x_d.size)
    a_in, a_d = np.vstack([x_in, 1.2 * x_in]), np.vstack([x_d, 0.9 * x_d])
    b_in, b_d = 0.5 * a_in, 0.7 * a_d  # same missing windows as a: shared counts
    c_in, c_d = a_in.copy(), a_d.copy()
    c_d[:, 3::11] = NAN  # scattered missing windows: own counts, same units
    xs_d, xs_in = {"a": a_d, "b": b_d, "c": c_d}, {"a": a_in, "b": b_in, "c": c_in}
    batched = S.matched_shift_ratios(xs_d, bins_d, ev_d, xs_in, bins_in, ev_in, 480, 10, n_boot=200, seed=7)
    plain = S.shift_ratios(xs_d, ev_d, xs_in, ev_in, n_boot=200, seed=7)
    for name in xs_d:
        single = S.matched_shift_ratio(xs_d[name], bins_d, ev_d, xs_in[name], bins_in, ev_in, 480, 10, 200, 7)
        np.testing.assert_allclose(batched[name]["replicates"], single["replicates"], rtol=1e-12)
        assert batched[name]["estimate"] == pytest.approx(single["estimate"], rel=1e-12)
        one = S.shift_ratio(xs_d[name], ev_d, xs_in[name], ev_in, n_boot=200, seed=7)
        np.testing.assert_allclose(plain[name]["replicates"], one["replicates"], rtol=1e-12)
    assert batched["b"]["estimate"] == pytest.approx(batched["a"]["estimate"] * 0.7 / 0.5, rel=1e-12)


# --------------------------------------------------------------------------- concentration
def test_concentration_and_share_mass() -> None:
    rng = np.random.default_rng(17)
    ev = np.repeat(np.arange(100), 20)
    mask = rng.random(ev.size) < 0.25
    x = np.where(mask, 2.0, 1.0)
    res = S.concentration(x, mask, ev, n_boot=200, seed=0)
    assert res["estimate"] == pytest.approx(2.0, rel=1e-12)
    assert res["lo"] == pytest.approx(2.0, rel=1e-12) and res["hi"] == pytest.approx(2.0, rel=1e-12)
    m, n = int(mask.sum()), mask.size
    assert res["share_windows"] == pytest.approx(m / n, rel=1e-12)
    assert res["share_mass"] == pytest.approx(2 * m / (2 * m + (n - m)), rel=1e-12)
    assert (res["mean_mask"], res["mean_rest"], res["n_events"]) == (2.0, 1.0, 100)
    # three seeds: per-seed ratios 2, 3, 4; the estimate is the ratio of the seed-averaged means
    xs = np.vstack([np.where(mask, 2.0 + k, 1.0) for k in range(3)])
    res3 = S.concentration(xs, mask, ev, n_boot=200, seed=0)
    np.testing.assert_allclose(res3["per_seed"], [2.0, 3.0, 4.0])
    assert res3["estimate"] == pytest.approx(3.0) and res3["seed_sd"] == pytest.approx(1.0)
    # windows without a value are left out of every share, events without any value out of the resampling
    xn = x.copy()
    xn[ev < 10] = NAN
    xn[::7] = NAN
    resn = S.concentration(xn, mask, ev, n_boot=200, seed=0)
    keep = ~np.isnan(xn)
    assert resn["share_windows"] == pytest.approx(mask[keep].mean(), rel=1e-12)
    assert resn["share_mass"] == pytest.approx(xn[keep & mask].sum() / xn[keep].sum(), rel=1e-12)
    assert resn["n_events"] == 90
    # the batched version gives the same numbers and shares the draws between masks
    batch = S.concentrations({"x": x}, {"a": mask, "b": ~mask}, ev, n_boot=200, seed=0)
    assert batch[("x", "a")]["estimate"] == res["estimate"]
    assert batch[("x", "b")]["estimate"] == pytest.approx(0.5)
    # a real concentration with event noise: the interval covers the true ratio 2
    noise = rng.gamma(10.0, 0.1, 100)[ev] * rng.gamma(4.0, 0.25, ev.size)
    real = S.concentration(x * noise, mask, ev, n_boot=1000, seed=4)
    assert real["lo"] < 2.0 < real["hi"] and real["lo"] > 1.5


# ------------------------------------------------------------------ across-seed decomposition
def test_seed_decomposition_identity_and_nan() -> None:
    rng = np.random.default_rng(23)
    k, n = 4, 500
    means = rng.normal(0.0, 1.0, (k, n))
    variances = rng.uniform(0.1, 1.0, (k, n))
    alea, epi = S.seed_decomposition(means, variances)
    a_ref, e_ref, _ = decompose(means, variances)
    np.testing.assert_allclose(alea, a_ref, rtol=1e-12)
    np.testing.assert_allclose(epi, e_ref, rtol=1e-10)
    mixture = MixtureDist(np.full((n, k), 1.0 / k), means.T, np.sqrt(variances).T)
    np.testing.assert_allclose(alea + epi, mixture.var(), rtol=1e-10)  # law of total variance, ddof = 0
    m2, v2 = means.copy(), variances.copy()
    m2[0, :10] = NAN  # windows 0-9: three valid seeds
    v2[1:, 10:20] = NAN  # windows 10-19: one valid seed -> undefined
    a2, e2 = S.seed_decomposition(m2, v2)
    a3, e3, _ = decompose(means[1:, :10], variances[1:, :10])
    np.testing.assert_allclose(a2[:10], a3, rtol=1e-12)
    np.testing.assert_allclose(e2[:10], e3, rtol=1e-10)
    assert np.isnan(a2[10:20]).all() and np.isnan(e2[10:20]).all()
    np.testing.assert_allclose(e2[20:], e_ref[20:], rtol=1e-10)
    a1, e1 = S.seed_decomposition(means[:1], variances[:1])
    assert np.isnan(a1).all() and np.isnan(e1).all()


# ------------------------------------------------------------------------------ H3.1 verdict
def _r(estimate: float, lo: float, hi: float) -> dict[str, float]:
    return {"estimate": estimate, "lo": lo, "hi": hi}


@pytest.mark.parametrize(
    "epi, alea, ci, point",
    [
        ((4.0, 3.2, 5.0), (1.1, 0.9, 1.25), "supported", "supported"),
        ((4.0, 3.0, 5.0), (1.0, 0.71, 1.29), "supported", "supported"),  # lower bound exactly 3 (>=)
        ((3.5, 2.5, 4.5), (1.0, 0.9, 1.1), "inconclusive", "supported"),  # point supports, interval does not
        ((4.0, 3.2, 5.0), (1.0, 0.8, 1.3), "inconclusive", "supported"),  # band (0.7, 1.3) is open
        ((4.0, 3.2, 5.0), (1.0, 0.65, 1.2), "inconclusive", "supported"),  # aleatoric interval leaves the band
        ((1.2, 1.0, 1.4), (1.0, 0.9, 1.1), "refuted", "refuted"),  # upper bound of R_epi < 1.5
        ((1.3, 1.1, 1.5), (1.0, 0.9, 1.1), "inconclusive", "refuted"),  # upper bound 1.5 is not < 1.5
        ((4.0, 3.2, 5.0), (2.5, 2.1, 3.0), "refuted", "refuted"),  # aleatoric entirely above 2
        ((4.0, 3.2, 5.0), (0.4, 0.3, 0.49), "refuted", "refuted"),  # aleatoric entirely below 0.5
        ((4.0, 3.2, 5.0), (2.2, 1.9, 2.6), "inconclusive", "refuted"),  # not entirely above 2
        ((2.0, 1.6, 2.5), (1.0, 0.9, 1.1), "inconclusive", "inconclusive"),
        ((NAN, NAN, NAN), (1.0, 0.9, 1.1), "inconclusive", "inconclusive"),  # no epistemic ratio
        ((NAN, NAN, NAN), (3.0, 2.5, 3.5), "refuted", "refuted"),  # the aleatoric rule alone refutes
    ],
)
def test_h31_verdict_branches(epi: tuple, alea: tuple, ci: str, point: str) -> None:
    out = S.h31_verdict(_r(*epi), _r(*alea), load_config())
    assert out == {"verdict_ci": ci, "verdict_point": point}


def test_h31_thresholds_from_config() -> None:
    assert S.H31Thresholds.from_config(load_config()) == S.H31Thresholds()
    assert S.H31Thresholds.from_config(None) == S.H31Thresholds()
    strict = {"h31": {"epi_support": 5.0, "alea_band": [0.9, 1.1]}}
    assert S.h31_verdict(_r(4.0, 3.2, 5.0), _r(1.0, 0.95, 1.05), strict)["verdict_ci"] == "inconclusive"
    assert S.h31_verdict(_r(6.0, 5.2, 7.0), _r(1.0, 0.95, 1.05), strict)["verdict_ci"] == "supported"


def test_h31_summary_rules() -> None:
    def row(model: str, dataset: str, ci: str, point: str, **kw: str) -> dict[str, str]:
        base = {"model": model, "kind": "model", "dataset": dataset, "subset": "all", "pair": "shift"}
        return {**base, "verdict_ci": ci, "verdict_point": point, **kw}

    rows = [
        row("A", "d1", "supported", "supported"),
        row("A", "d2", "supported", "supported"),
        row("A", "d3", "inconclusive", "supported"),
        row("A", "d3", "refuted", "refuted", subset="HV"),  # subsets do not count
        row("A", "p", "refuted", "refuted", pair="placebo"),  # placebo pairs are listed, not counted
        row("B", "d1", "supported", "supported"),
        row("B", "d2", "inconclusive", "inconclusive"),
        row("C", "d1", "supported", "supported"),
        row("C", "d2", "supported", "supported"),
        row("C", "d3", "refuted", "inconclusive"),
        row("A", "d1", "refuted", "inconclusive", kind="seed_ensemble"),
    ]
    out = S.h31_summary(pd.DataFrame(rows)).set_index(["model", "kind"])
    assert list(out.reset_index().columns) == S.H31_SUMMARY_COLUMNS
    a = out.loc[("A", "model")]
    assert (a["n_pairs"], a["n_supported"], a["n_refuted"], a["n_inconclusive"]) == (3, 2, 0, 1)
    assert a["overall"] == "supported" and a["n_supported_point"] == 3 and a["overall_point"] == "supported"
    assert a["n_placebo"] == 1 and a["placebo_verdicts"] == "p:refuted"
    b = out.loc[("B", "model")]
    assert b["overall"] == "inconclusive" and b["overall_point"] == "inconclusive"  # one supported pair < 2
    c = out.loc[("C", "model")]
    assert c["overall"] == "refuted" and c["overall_point"] == "supported"  # any refuted pair refutes
    e = out.loc[("A", "seed_ensemble")]
    assert (e["n_pairs"], e["overall"], e["overall_point"]) == (1, "refuted", "inconclusive")
    strict = S.h31_summary(rows, min_pairs=3).set_index(["model", "kind"])  # a list of dicts works too
    assert strict.loc[("A", "model"), "overall"] == "inconclusive"
    assert strict.loc[("A", "model"), "overall_point"] == "supported"
    bare = S.h31_summary([{"model": "M", "verdict_ci": "supported", "verdict_point": "supported"}] * 2)
    assert bare.loc[0, "kind"] == "model" and bare.loc[0, "overall"] == "supported" and bare.loc[0, "n_placebo"] == 0
    assert list(S.h31_summary([]).columns) == S.H31_SUMMARY_COLUMNS


def test_h31_summary_does_not_count_cross_checks_and_placebos():
    import pandas as pd

    from cf_unc.maps.shift import h31_summary

    rows = pd.DataFrame(
        {
            "model": ["m"] * 5,
            "kind": ["model"] * 5,
            "subset": ["all"] * 5,
            "dataset": ["ngsim", "waymo", "ngsim_fn", "highd_val", "openacc"],
            "pair": ["shift", "shift", "cross_check", "placebo", "shift"],
            "verdict_ci": ["supported", "supported", "supported", "refuted", "inconclusive"],
            "verdict_point": ["supported", "supported", "refuted", "refuted", "supported"],
        }
    )
    out = h31_summary(rows, min_pairs=2).iloc[0]
    assert out["n_pairs"] == 3 and out["n_supported"] == 2 and out["n_refuted"] == 0 and out["overall"] == "supported"
    assert out["n_supported_point"] == 3 and out["overall_point"] == "supported"  # the refuted cross-check is ignored
    assert out["n_placebo"] == 1 and out["placebo_verdicts"] == "highd_val:refuted"
    assert out["n_cross_check"] == 1 and out["cross_check_verdicts"] == "ngsim_fn:supported"
    # a second extraction alone must not make the two pairs the hypothesis asks for
    two = h31_summary(rows[rows["dataset"].isin(["ngsim", "ngsim_fn"])], min_pairs=2).iloc[0]
    assert two["n_pairs"] == 1 and two["overall"] == "inconclusive"


# ------------------------------------------------------------------------------ H3.1e verdict
def _conc(estimate: float, lo: float, hi: float) -> dict[str, float]:
    """A rare_concentration row (columns of the analysis table)."""
    return {"ratio_epistemic": estimate, "ratio_epistemic_lo": lo, "ratio_epistemic_hi": hi}


def _pair(dataset: str, estimate: float, lo: float, hi: float, pair: str = "shift", **kw: str) -> dict:
    """An h31_verdicts row (identifying keys and the epistemic ratio)."""
    return {"model": "m", "kind": "model", "dataset": dataset, "subset": "all", "pair": pair, "r_epi": estimate,
            "r_epi_lo": lo, "r_epi_hi": hi, "r_alea": 9.0, **kw}  # fmt: skip


@pytest.mark.parametrize(
    "conc, ci, point",
    [
        ((4.0, 1.5, 9.0), "supported", "supported"),
        ((4.0, 1.0, 9.0), "inconclusive", "supported"),  # lower bound must exceed 1 (strict)
        ((0.9, 0.5, 1.4), "inconclusive", "refuted"),  # interval covers 1, the point lies below
        ((0.5, 0.3, 0.8), "refuted", "refuted"),  # less epistemic variance in rare states
        ((0.95, 0.9, 1.0), "inconclusive", "refuted"),  # upper bound exactly 1 is not < 1
        ((NAN, NAN, NAN), "inconclusive", "inconclusive"),  # unknown concentration
    ],
)
def test_h31e_concentration_branches(conc: tuple, ci: str, point: str) -> None:
    (row,) = S.h31e_verdict(_conc(*conc), [_pair("d", 4.0, 3.5, 5.0)], load_config())
    assert (row["verdict_conc_ci"], row["verdict_conc_point"]) == (ci, point)
    assert (row["conc"], row["conc_lo"], row["conc_hi"]) == pytest.approx(conc, nan_ok=True) and row["rule"] == "rare_mass"


@pytest.mark.parametrize(
    "epi, ci, point",
    [
        ((4.0, 3.2, 5.0), "supported", "supported"),
        ((4.0, 3.0, 5.0), "supported", "supported"),  # lower bound exactly 3 (>=)
        ((3.5, 2.5, 4.5), "inconclusive", "supported"),
        ((1.2, 1.0, 1.4), "refuted", "refuted"),  # upper bound < 1.5
        ((1.3, 1.1, 1.5), "inconclusive", "refuted"),  # upper bound 1.5 is not < 1.5
        ((2.0, 1.6, 2.5), "inconclusive", "inconclusive"),
        ((NAN, NAN, NAN), "inconclusive", "inconclusive"),
    ],
)
def test_h31e_growth_branches(epi: tuple, ci: str, point: str) -> None:
    (row,) = S.h31e_verdict(_conc(4.0, 2.0, 8.0), [_pair("d", *epi)])
    assert (row["verdict_ci"], row["verdict_point"]) == (ci, point)
    assert row["dataset"] == "d" and row["r_alea"] == 9.0  # identifying keys and other columns are copied
    (alt,) = S.h31e_verdict({"estimate": 4.0, "lo": 2.0, "hi": 8.0}, [{"estimate": epi[0], "lo": epi[1], "hi": epi[2]}])
    assert (alt["verdict_ci"], alt["verdict_point"], alt["verdict_conc_ci"]) == (ci, point, "supported")  # plain keys


def test_h31e_thresholds_from_config() -> None:
    assert S.H31eThresholds.from_config(load_config()) == S.H31eThresholds() and S.H31eThresholds.from_config(None) == S.H31eThresholds()
    th = S.H31eThresholds.from_config({"h31": {"epi_support": 5.0, "min_pairs": 3, "alea_band": [0.7, 1.3]},
                                       "h31e": {"conc_support": 2.0, "epi_support": 4.0}})  # fmt: skip
    assert (th.epi_support, th.min_pairs, th.conc_support, th.epi_refute, th.rule) == (4.0, 3, 2.0, 1.5, "rare_mass")
    assert S.H31eThresholds.from_config({"conc_refute": 0.8}).conc_refute == 0.8  # the h31e section alone
    (row,) = S.h31e_verdict(_conc(4.0, 1.5, 9.0), [_pair("d", 4.0, 3.5, 5.0)], {"h31e": {"conc_support": 2.0}})
    assert row["verdict_conc_ci"] == "inconclusive"


def test_h31e_summary_rules() -> None:
    def rows(model: str, conc: tuple, pairs: list[tuple], kind: str = "model") -> list[dict]:
        return [{**r, "model": model, "kind": kind} for r in S.h31e_verdict(_conc(*conc), [_pair(*p) for p in pairs])]

    good = (4.0, 1.5, 9.0)
    table = pd.DataFrame(
        rows("A", good, [("d1", 5, 4, 6), ("d2", 5, 3.1, 7), ("d3", 2, 1.6, 2.5), ("p", 1.0, 0.9, 1.1, "placebo"),
                         ("x", 1.2, 1.0, 1.4, "cross_check"), ("d1", 1.0, 0.9, 1.1, "shift", )])  # fmt: skip
        + rows("B", good, [("d1", 5, 4, 6), ("d2", 2.5, 2.0, 3.5)])  # one supported pair < min_pairs
        + rows("C", (0.5, 0.3, 0.8), [("d1", 5, 4, 6), ("d2", 5, 4, 6)])  # concentration refuted
        + rows("D", (2.0, 0.9, 3.0), [("d1", 5, 4, 6), ("d2", 5, 4, 6)])  # concentration inconclusive
        + rows("E", good, [("d1", 5, 4, 6), ("d2", 5, 4, 6), ("d3", 1.2, 1.0, 1.4)])  # a refuted pair refutes
        + rows("A", good, [("d1", 5, 4, 6), ("d2", 5, 4, 6)], kind="seed_ensemble")
    )
    table.loc[5, "subset"] = "HV"  # a subset row (refuted) does not count
    out = S.h31e_summary(table).set_index(["model", "kind"])
    assert list(out.reset_index().columns) == S.H31E_SUMMARY_COLUMNS
    a = out.loc[("A", "model")]
    assert (a["n_pairs"], a["n_supported"], a["n_refuted"], a["n_inconclusive"], a["overall"]) == (3, 2, 0, 1, "supported")
    assert a["n_supported_point"] == 2 and a["overall_point"] == "supported" and bool(a["main"])
    assert a["placebo_verdicts"] == "p:refuted" and a["cross_check_verdicts"] == "x:refuted" and a["n_cross_check"] == 1
    assert (a["conc"], a["conc_lo"], a["conc_hi"], a["verdict_conc_ci"], a["rule"]) == (4.0, 1.5, 9.0, "supported", "rare_mass")
    assert out.loc[("B", "model"), "overall"] == "inconclusive" and out.loc[("B", "model"), "overall_point"] == "inconclusive"
    assert out.loc[("C", "model"), "overall"] == "refuted" and out.loc[("C", "model"), "n_supported"] == 2
    d = out.loc[("D", "model")]
    assert (d["verdict_conc_ci"], d["overall"], d["verdict_conc_point"], d["overall_point"]) == (
        "inconclusive", "inconclusive", "supported", "supported")  # fmt: skip
    assert out.loc[("E", "model"), "overall"] == "refuted" and out.loc[("E", "model"), "n_refuted"] == 1
    se = out.loc[("A", "seed_ensemble")]
    assert se["overall"] == "supported" and not bool(se["main"])
    strict = S.h31e_summary(table, min_pairs=3).set_index(["model", "kind"])
    assert strict.loc[("A", "model"), "overall"] == "inconclusive" and strict.loc[("E", "model"), "overall"] == "refuted"
    bare = S.h31e_summary([{"model": "M", "verdict_ci": "supported", "verdict_point": "supported", "rule": "rare_mass",
                            "verdict_conc_ci": "supported", "verdict_conc_point": "supported"}] * 2)  # fmt: skip
    assert bare.loc[0, "kind"] == "model" and bare.loc[0, "overall"] == "supported" and bare.loc[0, "n_pairs"] == 2
    assert list(S.h31e_summary([]).columns) == S.H31E_SUMMARY_COLUMNS
