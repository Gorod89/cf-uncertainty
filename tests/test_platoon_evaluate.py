"""Evaluation of experiment E3 (cf_unc.platoon.evaluate, docs/design.md 13.3)."""

from __future__ import annotations

import numpy as np
import pytest

from cf_unc.eval.metrics import crps_samples
from cf_unc.platoon.evaluate import (
    boot_seed,
    coverage_by_method,
    fair_crps,
    growth_curve,
    growth_exponent,
    h32_verdict,
    loo_static_correction,
    mean_with_ci,
    profile_mean,
    profile_order,
    relative_reduction,
    sample_quantiles,
    second_difference_share,
)
from cf_unc.rollout.calibration import conformal_quantile, interval_score


def test_fair_crps_matches_crps_samples() -> None:
    rng = np.random.default_rng(0)
    x = rng.gamma(3.0, 2.0, size=(6, 300))
    y = rng.gamma(3.0, 2.0, size=6)
    np.testing.assert_allclose(fair_crps(x, y), crps_samples(x, y, fair=True))
    y[2] = np.nan  # an index without a measured follower
    out = fair_crps(x, y)
    assert np.isnan(out[2]) and np.isfinite(np.delete(out, 2)).all()
    np.testing.assert_allclose(np.delete(out, 2), crps_samples(np.delete(x, 2, axis=0), np.delete(y, 2), fair=True))
    # ratio CRPS = CRPS / A0 (positive scaling)
    np.testing.assert_allclose(fair_crps(x / 4.0, y / 4.0)[[0, 1]], out[[0, 1]] / 4.0)
    # a degenerate law at the truth scores 0; one sample falls back to the plain CRPS
    assert fair_crps(np.full((1, 5), 2.0), [2.0])[0] == pytest.approx(0.0)
    assert fair_crps(np.array([[1.0]]), [3.0])[0] == pytest.approx(2.0)


def test_sample_quantiles_and_profile_mean() -> None:
    x = np.arange(1.0, 20.0)[None, :]  # S = 19: weibull positions put q05 / q50 / q95 on samples 1, 10, 19
    np.testing.assert_allclose(sample_quantiles(x), [[1.0, 10.0, 19.0]])
    v = np.array([[1.0, 2.0, np.nan, 4.0, 5.0], [np.nan, np.nan, np.nan, np.nan, np.nan], [3.0, 1.0, 2.0, 6.0, 9.0]])
    np.testing.assert_allclose(profile_mean(v), [3.0, np.nan, 4.2], equal_nan=True)
    np.testing.assert_allclose(profile_mean(v, range(1, 5)), [7.0 / 3, np.nan, 3.0], equal_nan=True)
    np.testing.assert_allclose(profile_mean(v, [5, 9]), [5.0, np.nan, 9.0], equal_nan=True)


def test_h32_verdict_branches() -> None:
    assert h32_verdict(0.30, 0.21, 0.40) == ("supported", "supported")
    assert h32_verdict(0.30, 0.20, 0.40) == ("supported", "supported")  # the bound itself counts
    assert h32_verdict(0.25, 0.10, 0.40) == ("inconclusive", "supported")
    assert h32_verdict(0.02, -0.10, 0.04) == ("refuted", "refuted")
    assert h32_verdict(0.10, -0.10, 0.06) == ("inconclusive", "inconclusive")
    assert h32_verdict(-0.50, -0.90, -0.20) == ("refuted", "refuted")
    assert h32_verdict(np.nan, np.nan, np.nan) == ("inconclusive", "inconclusive")
    assert h32_verdict(0.3, 0.1, 0.5, support_min=0.05) == ("supported", "supported")


def test_relative_reduction_paired_bootstrap() -> None:
    rng = np.random.default_rng(1)
    ref = rng.uniform(1.0, 3.0, 40)
    ids = [f"p{k}" for k in range(40)]
    exact = relative_reduction(0.7 * ref, ref, ids, n_boot=300, seed=3)
    assert exact["estimate"] == pytest.approx(0.3) and exact["lo"] == pytest.approx(0.3) and exact["hi"] == pytest.approx(0.3)
    assert exact["n_profiles"] == 40 and exact["crps_reference"] == pytest.approx(ref.mean())
    noisy = 0.7 * ref * rng.uniform(0.8, 1.2, 40)
    res = relative_reduction(noisy, ref, ids, n_boot=500, seed=3)
    assert res["lo"] < res["estimate"] < res["hi"] and res["estimate"] == pytest.approx(1 - noisy.mean() / ref.mean())
    assert res == relative_reduction(noisy, ref, ids, n_boot=500, seed=3)  # reproducible from the seed
    with_gap = noisy.copy()
    with_gap[:5] = np.nan  # profiles without the model are left out of both means
    part = relative_reduction(with_gap, ref, ids, n_boot=200, seed=3)
    assert part["n_profiles"] == 35 and part["crps_reference"] == pytest.approx(ref[5:].mean())
    assert np.isnan(relative_reduction([np.nan], [1.0], ["a"])["estimate"])
    est = mean_with_ci(ref, ids, 300, 4, 0.95)
    assert est["lo"] < est["estimate"] < est["hi"] and est["estimate"] == pytest.approx(ref.mean()) and est["n"] == 40


def test_growth_exponent_on_synthetic_concave_data() -> None:
    i = np.arange(1, 21)
    beta, log_sd1 = growth_exponent(2.0 * i**0.5)
    assert beta == pytest.approx(0.5) and log_sd1 == pytest.approx(np.log(2.0))
    assert second_difference_share(2.0 * i**0.5) == 1.0 and second_difference_share(i**1.5) == 0.0
    assert growth_exponent(i**1.5)[0] == pytest.approx(1.5)
    assert np.isnan(growth_exponent([1.0, np.nan, -1.0])[0])
    rng = np.random.default_rng(2)
    curves = 2.0 * i**0.5 * rng.lognormal(0.0, 0.15, size=(30, 20))  # 30 profiles, multiplicative noise
    res = growth_curve(curves, n_boot=400, seed=5)
    assert res["beta_lo"] < 0.5 < res["beta_hi"] and res["beta_hi"] - res["beta_lo"] < 0.2
    assert res["beta"] < 1.0 and (res["lo"] <= res["curve"]).all() and (res["curve"] <= res["hi"]).all()
    assert (res["n_profiles"] == 30).all()
    # missing indices (shorter platoons) and the minimum number of profiles per index
    ragged = curves.copy()
    ragged[5:, 4:] = np.nan
    res = growth_curve(ragged, n_boot=100, seed=5, min_profiles=6)
    assert res["n_profiles"].tolist() == [30] * 4 + [5] * 16 and np.isnan(res["curve"][4:]).all()
    assert np.isfinite(res["beta"])  # fitted over the four usable indices
    single = growth_curve(curves[:1], n_boot=10, seed=0)
    assert np.isnan(single["beta_lo"]) and np.isfinite(single["beta"])


def test_coverage_methods() -> None:
    rng = np.random.default_rng(3)
    p = 40
    mid = rng.uniform(5.0, 10.0, p)
    lo, hi = mid - 1.0, mid + 1.0
    y = mid + rng.normal(0.0, 1.0, p)  # the raw interval covers about 68 %
    y[3] = np.nan
    order = list(rng.permutation(p))
    res = coverage_by_method(y, lo, mid, hi, order, alpha=0.1, gamma=0.01, n_min=5)
    score = interval_score(y, lo, mid, hi)
    ok = np.isfinite(y)
    np.testing.assert_array_equal(res["raw"]["covered"][ok], (score[ok] <= 0).astype(float))
    np.testing.assert_allclose(res["raw"]["width"][ok], 2.0)
    assert np.isnan(res["raw"]["covered"][3]) and np.isnan(res["static_loo"]["covered"][3])
    q = loo_static_correction(np.where(ok, score, np.nan), 0.1)
    assert q[0] == pytest.approx(conformal_quantile(np.delete(score[ok], 0), 0.1)) and np.isnan(q[3])
    np.testing.assert_allclose(res["static_loo"]["q"][ok], q[ok])
    np.testing.assert_allclose(res["static_loo"]["width"][ok], 2.0 * (1.0 + q[ok]))
    assert np.nanmean(res["static_loo"]["covered"]) > np.nanmean(res["raw"]["covered"])
    first = [i for i in order if ok[i]][:5]
    assert (res["aci_online"]["q"][first] == 0.0).all()  # fewer than n_min scores seen: raw interval
    few = coverage_by_method(y[:4], lo[:4], mid[:4], hi[:4], [0, 1, 2, 3])
    assert np.isinf(few["static_loo"]["q"][0]) and few["static_loo"]["covered"][0] == 1.0  # 2 other scores: no quantile


def test_seeds_and_orders() -> None:
    assert boot_seed(0, "crps", "extended", "all") == boot_seed(0, "crps", "extended", "all")
    assert boot_seed(0, "crps", "extended", "all") != boot_seed(0, "crps", "headline", "all") != boot_seed(1, "crps", "headline", "all")
    ids = [f"p{k:02d}" for k in range(1, 11)]
    order = profile_order(ids, 7)
    assert sorted(order) == ids and order == profile_order(ids, 7) and order != ids


def test_growth_exponent_fit_excludes_ragged_indices() -> None:
    """``fit_min_profiles``: a few long platoons with a lower level must not bend the fitted exponent."""
    i = np.arange(1, 9, dtype=np.float64)
    full = np.tile(2.0 * i[:4] ** 0.3, (20, 1))  # 20 profiles with 4 followers, beta = 0.3
    curves = np.full((23, 8), np.nan)
    curves[:20, :4] = full
    curves[20:, :] = 0.5 * i**0.3  # 3 long platoons at a quarter of the level
    pooled = growth_curve(curves, n_boot=50, seed=1, min_profiles=3)
    balanced = growth_curve(curves, n_boot=50, seed=1, min_profiles=3, fit_min_profiles=12)
    assert pooled["beta"] < 0.0  # composition effect: the mean curve drops after index 4
    assert balanced["beta"] == pytest.approx(0.3, abs=0.02)
    assert balanced["fitted"].tolist() == [True] * 4 + [False] * 4
    assert np.isfinite(balanced["curve"][:8]).all()  # the curve is still drawn to index 8
    assert balanced["beta_lo"] <= balanced["beta"] <= balanced["beta_hi"]
