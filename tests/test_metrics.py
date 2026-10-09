"""Tests of cf_unc.eval.metrics: closed forms vs Monte Carlo, limits, calibration diagnostics.

All random inputs use fixed seeds, so every statistical assertion is deterministic; the
comments state the Monte Carlo error each tolerance is compared with (safety margin).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special, stats

from cf_unc.dist import GaussianDist, MixtureDist, SampleDist
from cf_unc.eval import metrics as M

N_SETS, N_DRAWS = 50, 20_000
Z95 = float(special.ndtri(0.95))


def _mixture_params(rng: np.random.Generator, n: int, k: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = rng.dirichlet(np.ones(k), n)
    mu = rng.uniform(-3.0, 3.0, (n, k))
    sd = np.exp(rng.uniform(np.log(0.2), np.log(2.0), (n, k)))
    return w, mu, sd


def _mixture_draws(rng: np.random.Generator, w: np.ndarray, mu: np.ndarray, sd: np.ndarray, n: int) -> np.ndarray:
    """(rows, n) mixture draws, independent of the implementation under test."""
    comp = (rng.random((w.shape[0], n))[:, :, None] > np.cumsum(w, axis=1)[:, None, :]).sum(axis=2)
    comp = np.minimum(comp, w.shape[1] - 1)
    return np.take_along_axis(mu, comp, 1) + np.take_along_axis(sd, comp, 1) * rng.standard_normal(comp.shape)


def _grimit_loop(w: np.ndarray, mu: np.ndarray, sd: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Plain double loop over components of the Grimit et al. (2006) formula (reference)."""

    def a_fn(m: np.ndarray, s2: np.ndarray) -> np.ndarray:
        s = np.sqrt(s2)
        return 2 * s * stats.norm.pdf(m / s) + m * (2 * stats.norm.cdf(m / s) - 1)

    out = np.zeros(len(y))
    for i in range(w.shape[1]):
        out += w[:, i] * a_fn(y - mu[:, i], sd[:, i] ** 2)
        for j in range(w.shape[1]):
            out -= 0.5 * w[:, i] * w[:, j] * a_fn(mu[:, i] - mu[:, j], sd[:, i] ** 2 + sd[:, j] ** 2)
    return out


# ------------------------------------------------------------------------ CRPS: closed form vs MC
def test_crps_gaussian_closed_form_vs_samples() -> None:
    rng = np.random.default_rng(101)
    mu = rng.uniform(-2.0, 2.0, N_SETS)
    sigma = np.exp(rng.uniform(np.log(0.1), np.log(3.0), N_SETS))
    y = mu + sigma * rng.normal(0.0, 1.5, N_SETS)
    x = mu[:, None] + sigma[:, None] * rng.standard_normal((N_SETS, N_DRAWS))
    rel = np.abs(M.crps_samples(x, y) / M.crps_gaussian(mu, sigma, y) - 1.0)
    # Monte Carlo relative sd of the sample CRPS at S = 20 000 is ~1.3 % at y = mu (influence
    # function sd 0.45 sigma vs CRPS 0.23 sigma) and smaller in the tails: expected mean |rel|
    # ~0.6 % (sd of that mean ~0.05 %) and max over 50 sets ~2 %; observed 0.61 % / 1.8 %.
    # Limits 1 % / 5 % leave > 7 sd for the mean and > 2.5x for the max.
    assert rel.mean() < 0.01, rel.mean()
    assert rel.max() < 0.05, rel.max()


def test_crps_mixture_closed_form_vs_samples() -> None:
    rng = np.random.default_rng(102)
    w, mu, sd = _mixture_params(rng, N_SETS)
    y = _mixture_draws(rng, w, mu, sd, 1)[:, 0] + rng.normal(0.0, 1.0, N_SETS)
    x = _mixture_draws(rng, w, mu, sd, N_DRAWS)
    exact = M.crps_mixture(w, mu, sd, y)
    np.testing.assert_allclose(exact, _grimit_loop(w, mu, sd, y), rtol=1e-12)
    rel = np.abs(M.crps_samples(x, y) / exact - 1.0)
    # same Monte Carlo error budget as for the Gaussian; observed mean 0.59 %, max 1.9 %
    assert rel.mean() < 0.01, rel.mean()
    assert rel.max() < 0.05, rel.max()


def test_crps_samples_matches_pairwise_formula_and_fair_gap() -> None:
    rng = np.random.default_rng(103)
    s = 40
    x = rng.normal(size=(30, s))
    y = rng.normal(size=30)
    e_xy = np.abs(x - y[:, None]).mean(axis=1)
    pair_sum = np.abs(x[:, :, None] - x[:, None, :]).sum(axis=(1, 2))
    np.testing.assert_allclose(M.crps_samples(x, y), e_xy - pair_sum / (2 * s * s), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(M.crps_samples(x, y, fair=True), e_xy - pair_sum / (2 * s * (s - 1)), rtol=1e-12, atol=1e-14)
    # gap = sum|x_i - x_j| / (2 S^2 (S - 1)) ~ E|X - X'| / (2 S) = 1 / (sqrt(pi) S) for N(0, 1): O(1/S)
    for s in (100, 1000, 10_000):
        x = rng.normal(size=(20, s))
        y = rng.normal(size=20)
        gap = M.crps_samples(x, y) - M.crps_samples(x, y, fair=True)
        assert np.all(gap > 0)
        # mean over 20 rows of the U-statistic; its relative sd is < 3 % already at S = 100
        assert abs(gap.mean() * s * np.sqrt(np.pi) - 1.0) < 0.1


# ----------------------------------------------------------------------- CRPS: exact identities
def test_crps_mixture_single_component_equals_gaussian() -> None:
    rng = np.random.default_rng(104)
    n = 2000
    mu = rng.normal(0.0, 3.0, n)
    sigma = np.exp(rng.uniform(np.log(1e-3), np.log(10.0), n))
    y = mu + sigma * rng.uniform(-40.0, 40.0, n)  # |y - mu| / sigma up to 40
    ref = M.crps_gaussian(mu, sigma, y)
    assert np.all(np.isfinite(ref))
    got = M.crps_mixture(np.ones((n, 1)), mu[:, None], sigma[:, None], y)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_crps_mixture_of_identical_components_equals_gaussian() -> None:
    rng = np.random.default_rng(105)
    n, k = 500, 5
    mu = rng.normal(size=n)
    sigma = rng.uniform(0.1, 3.0, n)
    y = mu + sigma * rng.uniform(-40.0, 40.0, n)
    w = rng.dirichlet(np.ones(k), n) * 7.0  # unnormalised on purpose
    got = M.crps_mixture(w, np.repeat(mu[:, None], k, 1), np.repeat(sigma[:, None], k, 1), y)
    np.testing.assert_allclose(got, M.crps_gaussian(mu, sigma, y), rtol=1e-12, atol=1e-12)


def test_crps_gaussian_known_values() -> None:
    # CRPS(N(0, 1), 0) = 2 phi(0) - 1 / sqrt(pi) = (sqrt(2) - 1) / sqrt(pi)
    assert M.crps_gaussian(0.0, 1.0, 0.0) == pytest.approx((np.sqrt(2) - 1) / np.sqrt(np.pi), rel=1e-14)
    # scale equivariance and translation invariance
    assert M.crps_gaussian(3.0, 2.5, 3.0 + 2.5 * 0.7) == pytest.approx(2.5 * M.crps_gaussian(0.0, 1.0, 0.7), rel=1e-14)


def test_crps_point_mass_limit() -> None:
    rng = np.random.default_rng(106)
    mu = rng.normal(size=100)
    y = rng.normal(size=100)
    np.testing.assert_allclose(M.crps_gaussian(mu, 1e-12, y), np.abs(y - mu), rtol=0, atol=1e-11)
    np.testing.assert_allclose(M.crps_gaussian(mu, 0.0, y), np.abs(y - mu), rtol=0, atol=1e-15)
    assert M.crps_gaussian(1.0, 0.0, 1.0) == 0.0
    # mixture of point masses: E|X - y| - 1/2 E|X - X'| with discrete X
    w = rng.dirichlet(np.ones(3), 100)
    m = rng.normal(size=(100, 3))
    ref = (w * np.abs(y[:, None] - m)).sum(1) - 0.5 * np.einsum("ni,nj,nij->n", w, w, np.abs(m[:, :, None] - m[:, None, :]))
    np.testing.assert_allclose(M.crps_mixture(w, m, np.zeros_like(m), y), ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(M.crps_mixture(w, m, np.full_like(m, 1e-12), y), ref, rtol=0, atol=1e-11)


@pytest.mark.parametrize("n, k", [(1, 4), (6, 1), (5, 3)])
def test_crps_and_nll_mixture_leave_inputs_untouched(n: int, k: int) -> None:
    rng = np.random.default_rng(117)
    w, mu, sd = rng.uniform(0.5, 2.0, (n, k)), rng.normal(size=(n, k)), rng.uniform(0.5, 1.5, (n, k))
    copies = (w.copy(), mu.copy(), sd.copy())
    y = rng.normal(size=n)
    wn = w / w.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(M.crps_mixture(w, mu, sd, y, chunk=1), _grimit_loop(wn, mu, sd, y), rtol=1e-12)
    M.nll_mixture(w, mu, sd, y)
    for arr, ref in zip((w, mu, sd), copies):
        np.testing.assert_array_equal(arr, ref)
    # read-only broadcast inputs (one weight row for all samples) are accepted
    shared = np.broadcast_to(w[:1], (n, k))
    np.testing.assert_allclose(M.crps_mixture(shared, mu, sd, y), M.crps_mixture(np.array(shared), mu, sd, y), rtol=1e-15)


def test_crps_mixture_chunking_and_threads_do_not_change_result() -> None:
    rng = np.random.default_rng(107)
    w, mu, sd = _mixture_params(rng, 257, k=9)
    y = rng.normal(size=257)
    ref = M.crps_mixture(w, mu, sd, y)
    np.testing.assert_allclose(M.crps_mixture(w, mu, sd, y, chunk=5, n_jobs=3), ref, rtol=1e-14, atol=0)
    np.testing.assert_allclose(M.crps_mixture(w, mu, sd, y, chunk=1000, n_jobs=1), ref, rtol=1e-14, atol=0)
    np.testing.assert_allclose(ref, _grimit_loop(w / w.sum(1, keepdims=True), mu, sd, y), rtol=1e-12)


def test_crps_quantile_function_dense_grid_matches_gaussian() -> None:
    rng = np.random.default_rng(108)
    levels = np.arange(1, 100) / 100.0
    mu, sigma = rng.normal(size=40), rng.uniform(0.5, 2.0, 40)
    y = mu + sigma * rng.normal(size=40)
    values = mu[:, None] + sigma[:, None] * special.ndtri(levels)
    exact = M.crps_gaussian(mu, sigma, y)
    closed = M.crps_quantile_function(levels, values, y)
    open_ = M.crps_quantile_function(levels, values, y, endpoints=False)
    # trapezoid error on a 0.01 grid is ~0.1-0.3 % (the tails below 0.01 / above 0.99 are the rest)
    assert np.max(np.abs(closed / exact - 1)) < 0.01
    assert np.all(open_ < closed)
    # pinball identity at a single row with hand-made numbers: tau = (0.25, 0.5, 0.75), q = (-1, 0, 1), y = 0
    rho = np.array([0.25, 0.0, 0.25])  # rho_tau(0 - q)
    expected = 2 * np.trapezoid(np.r_[0.0, rho, 0.0], np.r_[0.0, 0.25, 0.5, 0.75, 1.0])
    assert M.crps_quantile_function([0.25, 0.5, 0.75], [[-1.0, 0.0, 1.0]], [0.0])[0] == pytest.approx(expected)


# ----------------------------------------------------------------------------------------- NLL
def test_nll_mixture_single_component_equals_gaussian() -> None:
    rng = np.random.default_rng(109)
    mu, sigma = rng.normal(size=300), rng.uniform(0.01, 5.0, 300)
    y = mu + sigma * rng.uniform(-40, 40, 300)
    ref = M.nll_gaussian(mu, sigma, y)
    np.testing.assert_allclose(M.nll_mixture(np.ones((300, 1)), mu[:, None], sigma[:, None], y), ref, rtol=1e-12)
    np.testing.assert_allclose(ref, -stats.norm.logpdf(y, mu, sigma), rtol=1e-12)
    # analytic value at y = mu
    np.testing.assert_allclose(M.nll_gaussian(mu, sigma, mu), 0.5 * np.log(2 * np.pi) + np.log(sigma), rtol=1e-14)


def test_nll_mixture_log_space() -> None:
    rng = np.random.default_rng(110)
    w, mu, sd = _mixture_params(rng, 200)
    y = rng.normal(size=200)
    ref = -np.log((w * stats.norm.pdf(y[:, None], mu, sd)).sum(1))
    np.testing.assert_allclose(M.nll_mixture(w, mu, sd, y), ref, rtol=1e-12)
    # 40 sigma away from every component: the density underflows in linear space, not here
    far = mu.max(1) + 40 * sd.max(1)
    nll_far = M.nll_mixture(w, mu, sd, far)
    assert np.all(np.isfinite(nll_far)) and np.all(nll_far > 700)
    # (N, G) grid of targets
    grid = np.stack([y, y + 1.0], axis=1)
    np.testing.assert_allclose(M.nll_mixture(w, mu, sd, grid)[:, 1], M.nll_mixture(w, mu, sd, y + 1.0), rtol=1e-14)


# ------------------------------------------------------------------------- quantiles, intervals
def test_pinball_loss_values() -> None:
    assert M.pinball_loss(1.0, 0.0, 0.9) == pytest.approx(0.9)
    assert M.pinball_loss(-1.0, 0.0, 0.9) == pytest.approx(0.1)
    out = M.pinball_loss(np.array([0.0, 2.0]), np.array([[1.0, -1.0, 0.0], [1.0, 3.0, 2.0]]), np.array([0.1, 0.5, 0.9]))
    np.testing.assert_allclose(out, [[0.9, 0.5, 0.0], [0.1, 0.5, 0.0]])


def test_picp_of_exact_gaussian_intervals() -> None:
    rng = np.random.default_rng(111)
    n = 200_000
    mu, sigma = rng.normal(size=n), rng.uniform(0.2, 3.0, n)
    y = mu + sigma * rng.standard_normal(n)
    lo, hi = mu - Z95 * sigma, mu + Z95 * sigma
    # binomial sd of the coverage is sqrt(0.09 / 2e5) = 6.7e-4: +-0.005 is a 7.5 sd margin
    assert abs(M.picp(lo, hi, y) - 0.9) < 0.005
    assert M.mean_width(lo, hi) == pytest.approx(2 * Z95 * sigma.mean())
    assert np.isnan(M.picp([np.nan], [1.0], [0.0]))


def test_interval_score_minimised_by_true_quantiles() -> None:
    rng = np.random.default_rng(112)
    n = 200_000
    mu, sigma = rng.normal(size=n), rng.uniform(0.2, 3.0, n)
    y = mu + sigma * rng.standard_normal(n)
    best = M.interval_score(mu - Z95 * sigma, mu + Z95 * sigma, y, 0.9).mean()
    # expected excess over the optimum is >= ~0.05 * mean(sigma) for these perturbations while the
    # Monte Carlo sd of the mean difference is ~0.003 * mean(sigma): margin > 15 sd
    for scale in (0.8, 0.9, 1.1, 1.25):
        assert M.interval_score(mu - scale * Z95 * sigma, mu + scale * Z95 * sigma, y, 0.9).mean() > best
    for shift in (-0.25, 0.25):
        lo, hi = mu + (shift - Z95) * sigma, mu + (shift + Z95) * sigma
        assert M.interval_score(lo, hi, y, 0.9).mean() > best
    # hand values: alpha = 0.1 -> penalty factor 2 / alpha = 20
    np.testing.assert_allclose(M.interval_score([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, -0.1, 1.2], 0.9), [1.0, 3.0, 5.0])


# --------------------------------------------------------------------------------- calibration
def test_pit_ks_calibrated_vs_overconfident() -> None:
    rng = np.random.default_rng(113)
    n = 5000
    mu, sigma = rng.normal(size=n), rng.uniform(0.2, 3.0, n)
    y = mu + sigma * rng.standard_normal(n)
    _, p_cal = M.ks_uniform(special.ndtr((y - mu) / sigma))
    stat_over, p_over = M.ks_uniform(special.ndtr((y - mu) / (0.5 * sigma)))
    # calibrated: p ~ U(0, 1) under H0 (fails with prob. 1 %); the fixed seed gives p = 0.074.
    # sigma halved: population D ~ 0.17 at n = 5000 gives p ~ 1e-134, far below 1e-6
    assert p_cal > 0.01
    assert p_over < 1e-6 and stat_over > 0.1


def test_calibration_error_and_curve() -> None:
    rng = np.random.default_rng(114)
    n = 20_000
    z = rng.standard_normal(n)
    pit_cal = special.ndtr(z)
    pit_over = special.ndtr(2 * z)  # sigma halved
    pit_under = special.ndtr(0.5 * z)  # sigma doubled
    ce_cal = M.calibration_error(pit_cal)
    # sd of each empirical CDF value <= sqrt(0.25 / 2e4) = 0.0035: expected error ~0.003 (observed
    # 0.0017) against the limit 0.01
    assert ce_cal < 0.01
    # population values ~0.10 (over) and ~0.11 (under), i.e. > 25 sd above the calibrated one
    assert M.calibration_error(pit_over) > 0.05
    assert M.calibration_error(pit_under) > 0.05
    curve = M.calibration_curve(pit_cal)
    np.testing.assert_allclose(curve["nominal"], np.linspace(0.05, 0.95, 19))
    assert np.max(np.abs(curve["empirical_central"] - curve["nominal"])) < 0.02
    over = M.calibration_curve(pit_over)
    assert np.all(over["empirical_central"] < over["nominal"])  # over-confident: too few inside
    # exact counting on a tiny example
    small = M.calibration_curve(np.array([0.1, 0.5, 0.5, 0.95, np.nan]), levels=np.array([0.5, 0.8]))
    np.testing.assert_allclose(small["empirical_cdf"], [0.75, 0.75])
    np.testing.assert_allclose(small["empirical_central"], [0.5, 0.75])  # [0.25, 0.75] and [0.1, 0.9], closed


def test_pit_histogram() -> None:
    rng = np.random.default_rng(115)
    pit = rng.random(100_000)
    counts, edges, density = M.pit_histogram(np.r_[pit, np.nan], bins=20)
    assert counts.sum() == 100_000 and edges.shape == (21,) and edges[0] == 0.0 and edges[-1] == 1.0
    # binomial sd per bin ~ sqrt(5000 * 0.95) / 5000 = 1.4 %: 7 sd margin
    assert np.max(np.abs(density - 1.0)) < 0.1


# ---------------------------------------------------------------------------------- summarize
@pytest.mark.parametrize("kind", ["gaussian", "mixture", "samples"])
def test_summarize_per_sample_arrays(kind: str) -> None:
    rng = np.random.default_rng(116)
    n = 50
    mu, sigma = rng.normal(size=n), rng.uniform(0.5, 1.5, n)
    if kind == "gaussian":
        dist = GaussianDist(mu, sigma, aleatoric_var=sigma**2, epistemic_var=np.zeros(n))
    elif kind == "mixture":
        dist = MixtureDist(np.full((n, 2), 0.5), np.stack([mu - 0.5, mu + 0.5], 1), np.stack([sigma, sigma], 1))
    else:
        dist = SampleDist(mu[:, None] + sigma[:, None] * rng.standard_normal((n, 400)))
    y = mu + sigma * rng.standard_normal(n)
    y[3] = np.nan
    out = M.summarize(dist, y, coverage=0.8, rng=1)
    for key in ("nll", "crps", "covered", "width", "pit", "sq_err", "var"):
        assert out[key].shape == (n,), key
        assert out[key].dtype == np.float64
    ok = ~np.isnan(y)
    lo, hi = dist.interval(0.8)
    np.testing.assert_array_equal(out["covered"][ok], ((y >= lo) & (y <= hi))[ok].astype(float))
    np.testing.assert_allclose(out["width"], hi - lo)
    y_filled = np.where(ok, y, 0.0)
    np.testing.assert_allclose(out["crps"][ok], dist.crps(y_filled)[ok])
    np.testing.assert_allclose(out["nll"][ok], -dist.log_prob(y_filled)[ok])
    np.testing.assert_allclose(out["sq_err"][ok], ((y_filled - dist.mean()) ** 2)[ok])
    assert np.isnan(out["covered"][3]) and np.isnan(out["crps"][3]) and np.isnan(out["pit"][3])
    assert ("epistemic_var" in out) == (kind == "gaussian")
