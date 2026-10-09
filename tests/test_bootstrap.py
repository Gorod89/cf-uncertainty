"""Tests of cf_unc.eval.bootstrap: group (driver) bootstrap, ratios, paired test, Holm.

Random inputs and bootstrap draws use fixed seeds (deterministic); statistical tolerances
state their margin.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special, stats

from cf_unc.eval import bootstrap as B

Z975 = float(special.ndtri(0.975))


def _grouped(
    rng: np.random.Generator,
    sizes: np.ndarray,
    group_sd: float,
    noise_sd: float,
    mean: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """values = mean + group effect + noise, rows of group g repeated sizes[g] times."""
    groups = np.repeat(np.arange(len(sizes)), sizes)
    effect = rng.normal(0.0, group_sd, len(sizes))
    return mean + effect[groups] + rng.normal(0.0, noise_sd, groups.size), groups


# --------------------------------------------------------------------------------- by group
def test_group_bootstrap_ci_matches_analytic_se() -> None:
    rng = np.random.default_rng(0)
    n_groups, size = 400, 10
    values, groups = _grouped(rng, np.full(n_groups, size), 1.0, 1.0)
    res = B.bootstrap_by_group(values, groups, n_boot=2000, seed=1)
    assert res["estimate"] == pytest.approx(values.mean(), rel=1e-12)
    assert res["n_groups"] == n_groups and res["n"] == values.size and res["replicates"].shape == (2000,)
    se = values.reshape(n_groups, size).mean(axis=1).std(ddof=1) / np.sqrt(n_groups)  # se of the mean of group means
    half_width = (res["hi"] - res["lo"]) / (2 * Z975)
    # Monte Carlo error of 2.5 / 97.5 % percentiles at n_boot = 2000 is ~3 % of the half width and
    # the bootstrap se differs from the formula by sqrt((G - 1) / G) only: 15 % is a > 4 sd margin
    assert abs(half_width / se - 1.0) < 0.15
    assert abs(res["se"] / se - 1.0) < 0.10  # MC error of the se ~1.6 %
    assert res["lo"] < res["estimate"] < res["hi"]


def test_group_bootstrap_wider_than_naive_under_within_group_correlation() -> None:
    rng = np.random.default_rng(2)
    values, groups = _grouped(rng, np.full(60, 50), 1.0, 1.0)  # intra-class correlation 0.5
    grouped = B.bootstrap_by_group(values, groups, n_boot=1000, seed=3)
    naive = B.bootstrap_by_group(values, np.arange(values.size), n_boot=1000, seed=3)  # rows i.i.d.
    ratio = (grouped["hi"] - grouped["lo"]) / (naive["hi"] - naive["lo"])
    # design effect: expected ratio sqrt((1 + 1 / 50) * 50 / 2) ~ 5 (+-15 % from 60 groups)
    assert ratio > 2, ratio


def test_group_bootstrap_ci_coverage() -> None:
    rng = np.random.default_rng(4)
    true_mean, n_rep, hits = 1.5, 200, 0
    for rep in range(n_rep):
        values, groups = _grouped(rng, rng.integers(5, 30, 60), 1.0, 1.0, mean=true_mean)
        res = B.bootstrap_by_group(values, groups, n_boot=300, seed=rep)
        hits += res["lo"] <= true_mean <= res["hi"]
    coverage = hits / n_rep
    # nominal 95 %; the percentile bootstrap with 60 groups under-covers slightly (~94 %); the
    # binomial sd over 200 repetitions is ~1.7 %, so [0.90, 0.99] is a > 2 sd band on both sides
    assert 0.90 <= coverage <= 0.99, coverage


def test_bootstrap_by_group_nan_labels_and_reproducibility() -> None:
    rng = np.random.default_rng(12)
    values, groups = _grouped(rng, np.full(30, 10), 1.0, 1.0)
    with_nan = values.copy()
    with_nan[::7] = np.nan
    keep = ~np.isnan(with_nan)
    res = B.bootstrap_by_group(with_nan, groups, n_boot=200, seed=4)
    ref = B.bootstrap_by_group(values[keep], groups[keep], n_boot=200, seed=4)
    np.testing.assert_array_equal(res["replicates"], ref["replicates"])
    assert res["estimate"] == pytest.approx(np.nanmean(with_nan), rel=1e-12) and res["n"] == keep.sum()
    all_nan = values.copy()
    all_nan[groups == 3] = np.nan
    assert B.bootstrap_by_group(all_nan, groups, n_boot=50)["n_groups"] == 29  # empty group dropped
    labels = np.array([f"driver_{g}" for g in groups])
    a = B.bootstrap_by_group(values, labels, n_boot=100, seed=7)
    np.testing.assert_array_equal(a["replicates"], B.bootstrap_by_group(values, labels, n_boot=100, seed=7)["replicates"])
    assert not np.array_equal(a["replicates"], B.bootstrap_by_group(values, labels, n_boot=100, seed=8)["replicates"])
    with pytest.raises(ValueError):
        B.bootstrap_by_group(values, groups[:-1])
    with pytest.raises(ValueError):
        B.bootstrap_by_group(values, groups, ci=1.0)


def test_bootstrap_many_groups_is_processed_in_blocks() -> None:
    rng = np.random.default_rng(13)
    n_groups = 6000  # 2^22 / 6000 = 699 replicates per block -> 3 blocks for n_boot = 1500
    values = rng.normal(size=2 * n_groups)
    groups = np.repeat(np.arange(n_groups), 2)
    res = B.bootstrap_by_group(values, groups, n_boot=1500, seed=0)
    assert res["replicates"].shape == (1500,) and np.all(np.isfinite(res["replicates"]))
    se = values.reshape(-1, 2).mean(axis=1).std(ddof=1) / np.sqrt(n_groups)
    assert abs(res["se"] / se - 1.0) < 0.1  # MC error of the se at n_boot = 1500 ~1.8 %


# ----------------------------------------------------------------------------------- ratios
def _ratio_data(rng: np.random.Generator, n_groups: int = 80, size: int = 20) -> tuple[np.ndarray, ...]:
    """Positive den (mean 1) and num with E[num] / E[den] = 3 and driver-level ratio noise."""
    groups = np.repeat(np.arange(n_groups), size)
    den = rng.gamma(4.0, 0.25, n_groups)[groups] * rng.gamma(4.0, 0.25, groups.size)
    num = 3.0 * den * rng.gamma(20.0, 0.05, n_groups)[groups] * rng.gamma(50.0, 0.02, groups.size)
    return num, den, groups


def _two_sample_data(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    num, g_num = _grouped(rng, np.full(50, 30), 0.5, 1.0, mean=3.0)  # e.g. shifted data set
    den, g_den = _grouped(rng, np.full(80, 20), 0.2, 0.5, mean=1.0)  # e.g. in-domain data set
    return num, g_num, den, g_den


def test_ratio_bootstrap_recovers_known_ratio() -> None:
    num, den, groups = _ratio_data(np.random.default_rng(8))
    res = B.bootstrap_ratio_by_group(num, den, groups, n_boot=1000, seed=6)
    assert res["estimate"] == pytest.approx(num.mean() / den.mean(), rel=1e-12)
    assert res["n_groups"] == 80 and res["replicates"].shape == (1000,)
    # this data set: estimate 2.968, CI [2.811, 3.139], se ~0.085 (z = -0.4)
    assert res["lo"] < 3.0 < res["hi"]
    assert abs(res["estimate"] - 3.0) < 0.3
    # coverage over 100 data sets (observed 0.96 for this seed); binomial sd ~2.2 % around the
    # nominal 95 %, so >= 0.85 is a > 4 sd margin
    rng = np.random.default_rng(31)
    hits = 0
    for rep in range(100):
        num, den, groups = _ratio_data(rng)
        r = B.bootstrap_ratio_by_group(num, den, groups, n_boot=300, seed=rep)
        hits += r["lo"] < 3.0 < r["hi"]
    assert hits >= 85, hits
    num_nan, den_nan = num.copy(), den.copy()
    num_nan[::5] = np.nan
    den_nan[1::9] = np.nan
    res_nan = B.bootstrap_ratio_by_group(num_nan, den_nan, groups, n_boot=200, seed=6)
    assert res_nan["estimate"] == pytest.approx(np.nanmean(num_nan) / np.nanmean(den_nan), rel=1e-12)
    assert np.all(np.isfinite(res_nan["replicates"]))


def test_two_sample_ratio_bootstrap() -> None:
    num, g_num, den, g_den = _two_sample_data(np.random.default_rng(7))
    res = B.bootstrap_two_sample_ratio(num, g_num, den, g_den, n_boot=1000, seed=8)
    assert res["estimate"] == pytest.approx(num.mean() / den.mean(), rel=1e-12)
    assert res["n_groups_num"] == 50 and res["n_groups_den"] == 80
    # this data set: estimate 2.857, CI [2.651, 3.067] (se ~0.10 = 3 * sqrt(0.025^2 + 0.024^2))
    assert res["lo"] < 3.0 < res["hi"]
    # both data sets contribute: wider than resampling the numerator groups alone
    num_only = B.bootstrap_by_group(num / den.mean(), g_num, n_boot=1000, seed=8)
    assert res["hi"] - res["lo"] > num_only["hi"] - num_only["lo"]
    # coverage over 100 data sets (observed 0.92 for this seed): >= 0.85 is a ~3 sd margin
    rng = np.random.default_rng(31)
    hits = 0
    for rep in range(100):
        r = B.bootstrap_two_sample_ratio(*_two_sample_data(rng), n_boot=300, seed=rep)
        hits += r["lo"] < 3.0 < r["hi"]
    assert hits >= 85, hits


# ------------------------------------------------------------------------------------ paired
def test_paired_bootstrap_identical_inputs() -> None:
    rng = np.random.default_rng(9)
    a, groups = _grouped(rng, np.full(40, 10), 1.0, 1.0, mean=2.0)
    res = B.paired_bootstrap_test(a, a.copy(), groups, n_boot=500, seed=1)
    assert res["diff"] == 0.0 and res["pvalue"] == 1.0
    assert res["lo"] == 0.0 and res["hi"] == 0.0 and res["rel_diff"] == 0.0


def test_paired_bootstrap_detects_difference_and_sign() -> None:
    rng = np.random.default_rng(10)
    b, groups = _grouped(rng, np.full(40, 25), 1.0, 1.0, mean=2.0)
    a = b + 0.3 + rng.normal(0.0, 0.5, b.size)  # a is larger by 0.3 on average
    res = B.paired_bootstrap_test(a, b, groups, n_boot=1000, seed=2)
    assert res["diff"] == pytest.approx((a - b).mean(), rel=1e-12) and res["diff"] > 0
    # se of the mean difference ~0.016, i.e. 0.3 is ~19 se: every replicate is > 0 and the
    # p-value is the smallest attainable one, 2 / (B + 1)
    assert res["lo"] > 0 and res["pvalue"] < 0.01
    assert res["pvalue"] == pytest.approx(2 / 1001)
    assert res["rel_diff"] == pytest.approx(res["diff"] / b.mean(), rel=1e-12)
    rev = B.paired_bootstrap_test(b, a, groups, n_boot=1000, seed=2)
    assert rev["diff"] == pytest.approx(-res["diff"], rel=1e-12) and rev["hi"] < 0 and rev["pvalue"] < 0.01


def test_paired_bootstrap_null_calibration() -> None:
    rng = np.random.default_rng(14)
    n_rep, rejections = 400, 0
    for rep in range(n_rep):
        b, groups = _grouped(rng, rng.integers(5, 40, 40), 1.0, 1.0)
        a = b + 0.3 * rng.normal(0.0, 1.0, 40)[groups] + rng.normal(0.0, 0.5, b.size)  # no difference in mean
        rejections += B.paired_bootstrap_test(a, b, groups, n_boot=200, seed=rep)["pvalue"] <= 0.05
    # nominal 5 %; the percentile bootstrap with 40 groups gives ~6 % (0.0525-0.0725 over three
    # seeds, 0.0725 for this one). Binomial sd over 400 repetitions ~1.2 %: [0.02, 0.12] is > 3 sd.
    assert 0.02 <= rejections / n_rep <= 0.12, rejections / n_rep


# -------------------------------------------------------------------------------------- Holm
def test_holm_hand_computed_and_properties() -> None:
    adjusted, reject = B.holm([0.01, 0.04, 0.03, 0.005])
    # sorted: 4 * 0.005 = 0.02, 3 * 0.01 = 0.03, 2 * 0.03 = 0.06, 1 * 0.04 = 0.04 -> running max
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02], rtol=1e-12)
    np.testing.assert_array_equal(reject, [True, False, False, True])
    p = np.array([0.2, 0.001, 0.04, 0.5, 0.01, 0.03])
    adjusted, _ = B.holm(p)
    assert np.all(np.diff(adjusted[np.argsort(p)]) >= 0)  # monotone in the raw p-values
    assert np.all(adjusted >= p) and np.all(adjusted <= 1.0)
    perm = np.random.default_rng(0).permutation(p.size)
    np.testing.assert_allclose(B.holm(p[perm])[0], adjusted[perm])  # order preserved
    np.testing.assert_allclose(B.holm([0.6, 0.9])[0], [1.0, 1.0])  # capped at 1
    adj_nan, rej_nan = B.holm([0.01, np.nan, 0.02])
    np.testing.assert_allclose(adj_nan, [0.02, np.nan, 0.02])
    np.testing.assert_array_equal(rej_nan, [True, False, True])


# ----------------------------------------------------------------------------------- generic
def test_bootstrap_generic_non_mean_statistics() -> None:
    rng = np.random.default_rng(11)
    values, groups = _grouped(rng, np.full(50, 20), 0.5, 1.0)
    pit = special.ndtr(values / np.sqrt(1.25))
    med = B.bootstrap_generic({"v": values}, groups, lambda d: np.median(d["v"]), n_boot=300, seed=1)
    assert med["estimate"] == pytest.approx(np.median(values)) and med["lo"] <= med["estimate"] <= med["hi"]
    assert med["n_groups"] == 50
    ks = B.bootstrap_generic({"pit": pit}, groups, lambda d: stats.kstest(d["pit"], "uniform").statistic, n_boot=200, seed=2)
    assert ks["replicates"].shape == (200,) and 0.0 <= ks["lo"] <= ks["hi"] <= 1.0
    qs = B.bootstrap_generic({"v": values}, groups, lambda d: np.quantile(d["v"], [0.1, 0.9]), n_boot=200, seed=3)
    assert qs["replicates"].shape == (200, 2) and qs["lo"].shape == (2,) and np.all(qs["lo"] <= qs["hi"])
    # same seed -> same drawn multiplicities: the generic mean equals the vectorised one
    fast = B.bootstrap_by_group(values, groups, n_boot=300, seed=5)
    slow = B.bootstrap_generic({"v": values}, groups, lambda d: d["v"].mean(), n_boot=300, seed=5)
    np.testing.assert_allclose(slow["replicates"], fast["replicates"], rtol=1e-12)
    via_stat = B.bootstrap_by_group(values, groups, n_boot=300, seed=1, stat=np.median)
    np.testing.assert_array_equal(via_stat["replicates"], med["replicates"])
