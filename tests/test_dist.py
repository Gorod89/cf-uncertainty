"""Tests of cf_unc.dist: predictive laws, variance decomposition, row indexing and concat.

Random inputs use fixed seeds (deterministic); statistical tolerances state their margin.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pytest
from scipy import special, stats

from cf_unc.dist import GaussianDist, MixtureDist, PredDist, QuantileDist, SampleDist, decompose
from cf_unc.eval import metrics as M

LEVELS = np.array([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
Q_LEVELS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])


def make_gaussian(rng: np.random.Generator, n: int) -> GaussianDist:
    return GaussianDist(rng.normal(0.0, 2.0, n), rng.uniform(0.3, 2.0, n))


def make_mixture(rng: np.random.Generator, n: int) -> MixtureDist:
    return MixtureDist(rng.dirichlet(np.ones(3), n), rng.normal(0.0, 2.0, (n, 3)), rng.uniform(0.3, 1.5, (n, 3)))


def make_quantile(rng: np.random.Generator, n: int) -> QuantileDist:
    knots = stats.gamma.ppf(Q_LEVELS, 2.0)  # right-skewed quantiles
    return QuantileDist(Q_LEVELS, rng.normal(0.0, 1.0, (n, 1)) + rng.uniform(0.5, 2.0, (n, 1)) * knots)


def make_samples(rng: np.random.Generator, n: int, s: int = 2000) -> SampleDist:
    return SampleDist(rng.gamma(2.0, 1.0, (n, s)) * rng.uniform(0.5, 2.0, (n, 1)) - 2.0)


FACTORIES: dict[str, Callable[[np.random.Generator, int], PredDist]] = {
    "gaussian": make_gaussian,
    "mixture": make_mixture,
    "quantile": make_quantile,
    "samples": make_samples,
}
CONTINUOUS = ("gaussian", "mixture", "quantile")


def _params(d: PredDist) -> tuple[np.ndarray, ...]:
    if isinstance(d, GaussianDist):
        return d.mu, d.sigma
    if isinstance(d, MixtureDist):
        return d.weights, d.mus, d.sigmas
    if isinstance(d, QuantileDist):
        return d.values, d.n_crossings
    assert isinstance(d, SampleDist)
    return (d.samples,)


# ----------------------------------------------------------------------- generic law properties
@pytest.mark.parametrize("kind", FACTORIES)
def test_cdf_of_quantile_returns_levels(kind: str) -> None:
    d = FACTORIES[kind](np.random.default_rng(1), 6)
    q = d.quantile(LEVELS)
    assert q.shape == (6, LEVELS.size)
    # continuous laws: exact inverse (mixture: bisection to floating-point resolution);
    # ECDF of linearly interpolated sample quantiles is within 1 / S of the level
    tol = 1e-12 if kind in CONTINUOUS else 1.0 / d.n_samples + 1e-12  # type: ignore[attr-defined]
    np.testing.assert_allclose(d.cdf(q), np.broadcast_to(LEVELS, q.shape), rtol=0, atol=tol)
    np.testing.assert_allclose(d.quantile(np.tile(LEVELS, (6, 1))), q, rtol=1e-13, atol=1e-13)  # per-row levels
    lo, hi = d.interval(0.9)
    np.testing.assert_allclose(np.stack([lo, hi], 1), d.quantile([0.05, 0.95]), rtol=1e-13, atol=1e-13)
    with pytest.raises(ValueError):
        d.interval(1.0)


@pytest.mark.parametrize("kind", FACTORIES)
def test_quantile_is_monotone(kind: str) -> None:
    d = FACTORIES[kind](np.random.default_rng(2), 5)
    dq = np.diff(d.quantile(np.linspace(1e-4, 1 - 1e-4, 501)), axis=1)
    assert np.all(dq >= 0)
    if kind in CONTINUOUS:
        assert np.all(dq > 0)


@pytest.mark.parametrize("kind", FACTORIES)
def test_interval_covers_90_percent_of_own_samples(kind: str) -> None:
    rng = np.random.default_rng(3)
    d = FACTORIES[kind](rng, 4)
    lo, hi = d.interval(0.9)
    x = d.sample(100_000, rng)
    coverage = ((x >= lo[:, None]) & (x <= hi[:, None])).mean()
    # 400 000 draws: binomial sd 4.7e-4, limit 0.005 = 10 sd. SampleDist: resampled coverage is
    # the share of the S = 2000 stored samples inside the interval, within ~2 / S of 0.9.
    assert abs(coverage - 0.9) < 0.005, coverage


@pytest.mark.parametrize("kind", FACTORIES)
def test_mean_and_var_match_moments_of_own_samples(kind: str) -> None:
    rng = np.random.default_rng(4)
    d = FACTORIES[kind](rng, 3)
    n = 400_000
    x = d.sample(n, rng)
    assert x.shape == (3, n)
    mean, var = d.mean(), d.var()
    dev = x - x.mean(axis=1, keepdims=True)
    se_mean = np.sqrt(var / n)
    se_var = np.sqrt(((dev**4).mean(axis=1) - x.var(axis=1) ** 2) / n)
    # limits of 6 standard errors (12 comparisons: no false alarm in practice). SampleDist:
    # resampling reproduces the ddof = 0 variance, var() uses ddof = 1: bias var / S < 0.2 se.
    assert np.all(np.abs(x.mean(axis=1) - mean) < 6 * se_mean)
    assert np.all(np.abs(x.var(axis=1) - var) < 6 * se_var)


@pytest.mark.parametrize("kind", FACTORIES)
def test_log_prob_integrates_to_one(kind: str) -> None:
    rng = np.random.default_rng(5)
    n = 3
    if kind == "samples":
        d = make_samples(rng, n, s=200)
        h = d.bandwidth()
        breaks = np.stack([d.samples.min(axis=1) - 12 * h, d.samples.max(axis=1) + 12 * h], axis=1)
        points = 20_000
    else:
        d = FACTORIES[kind](rng, n)
        ends = d.quantile([1e-12, 1 - 1e-12])
        knots = d.values if isinstance(d, QuantileDist) else np.empty((n, 0))
        breaks = np.concatenate([ends[:, :1], knots, ends[:, 1:]], axis=1)
        points = 20_000 if kind == "quantile" else 200_000
    # midpoint rule on every piece between breaks (exact for the piecewise-constant density of
    # QuantileDist, O(h^2) elsewhere); mass outside the truncation points is <= 2e-12
    width = np.diff(breaks, axis=1)
    frac = (np.arange(points) + 0.5) / points
    grid = breaks[:, :-1, None] + width[:, :, None] * frac
    dens = np.exp(d.log_prob(grid.reshape(n, -1))).reshape(grid.shape)
    step = (width / points)[:, :, None]
    mass = (dens * step).sum(axis=(1, 2))
    np.testing.assert_allclose(mass, 1.0, rtol=0, atol=1e-6)
    if kind != "samples":  # density moments agree with the closed-form mean / var
        m1 = (dens * grid * step).sum(axis=(1, 2))
        m2 = (dens * (grid - m1[:, None, None]) ** 2 * step).sum(axis=(1, 2))
        np.testing.assert_allclose(m1, d.mean(), rtol=0, atol=1e-6)
        np.testing.assert_allclose(m2, d.var(), rtol=1e-6)


@pytest.mark.parametrize("kind", FACTORIES)
def test_extreme_targets_are_finite_and_warning_free(kind: str) -> None:
    d = FACTORIES[kind](np.random.default_rng(8), 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no overflow / underflow / invalid warnings either
        for sign in (-1.0, 1.0):
            y = d.mean() + sign * 40.0 * d.std()  # 40 standard deviations away
            assert np.all(np.isfinite(d.log_prob(y)))
            cdf = d.cdf(y)
            assert np.all((cdf >= 0) & (cdf <= 1))
            crps = d.crps(y)
            assert np.all(np.isfinite(crps)) and np.all(crps > 0)
            assert np.all(np.isfinite(d.pit(y, np.random.default_rng(0))))
        assert np.all(np.isfinite(d.quantile([1e-12, 1 - 1e-12])))


@pytest.mark.parametrize("kind", FACTORIES)
def test_getitem_and_concat_roundtrip(kind: str) -> None:
    rng = np.random.default_rng(6)
    d = FACTORIES[kind](rng, 7)
    d.aleatoric_var = rng.uniform(0.1, 1.0, 7)
    d.epistemic_var = rng.uniform(0.0, 0.1, 7)
    parts = [d[:2], d[2:6], d[6]]
    assert [len(p) for p in parts] == [2, 4, 1]
    for back in (type(d).concat(parts), PredDist.concat(parts)):
        assert type(back) is type(d)
        for got, ref in zip(_params(back), _params(d)):
            np.testing.assert_array_equal(got, ref)
        np.testing.assert_array_equal(back.aleatoric_var, d.aleatoric_var)
        np.testing.assert_array_equal(back.epistemic_var, d.epistemic_var)
        np.testing.assert_array_equal(back.mean(), d.mean())
    idx = np.array([5, 0, 5])
    sub = d[idx]
    for got, ref in zip(_params(sub), _params(d)):
        np.testing.assert_array_equal(got, ref[idx])
    np.testing.assert_array_equal(sub.epistemic_var, d.epistemic_var[idx])
    y = rng.normal(size=7)
    np.testing.assert_allclose(sub.cdf(y[idx]), d.cdf(y)[idx], rtol=1e-14)
    np.testing.assert_allclose(sub.crps(y[idx]), d.crps(y)[idx], rtol=1e-12)
    mask = np.arange(7) % 2 == 0
    np.testing.assert_allclose(d[mask].var(), d.var()[mask], rtol=1e-12)
    np.testing.assert_array_equal(d[mask].aleatoric_var, d.aleatoric_var[mask])
    last = d[-1]
    assert len(last) == 1 and last.mean()[0] == pytest.approx(d.mean()[-1], rel=1e-12)
    bare = FACTORIES[kind](rng, 2)
    bare.aleatoric_var = bare.epistemic_var = None
    with pytest.raises(ValueError):
        type(d).concat([d[:2], bare])


def test_concat_type_errors() -> None:
    rng = np.random.default_rng(7)
    with pytest.raises(TypeError):
        PredDist.concat([make_gaussian(rng, 2), make_mixture(rng, 2)])
    with pytest.raises(TypeError):
        GaussianDist.concat([make_mixture(rng, 2)])
    with pytest.raises(ValueError):
        PredDist.concat([])
    with pytest.raises(ValueError):
        QuantileDist.concat([make_quantile(rng, 2), QuantileDist([0.1, 0.5, 0.9], [[0.0, 1.0, 2.0]])])
    with pytest.raises(ValueError):
        SampleDist.concat([make_samples(rng, 2, s=10), make_samples(rng, 2, s=11)])


# ---------------------------------------------------------------------------- decomposition
def test_decomposition_identity_gaussian_members() -> None:
    rng = np.random.default_rng(10)
    n_members, n = 5, 1000
    means = rng.normal(0.0, 2.0, (n_members, n))
    variances = rng.uniform(0.05, 3.0, (n_members, n))
    aleatoric, epistemic, total = decompose(means, variances)
    np.testing.assert_allclose(aleatoric + epistemic, total, rtol=0, atol=1e-10)
    np.testing.assert_allclose(aleatoric, variances.mean(axis=0), rtol=1e-14)
    np.testing.assert_allclose(epistemic, means.var(axis=0, ddof=0), rtol=1e-14)
    pooled = MixtureDist.from_members(None, means, np.sqrt(variances))  # K = 1 members as (M, N)
    assert pooled.n_components == n_members and len(pooled) == n
    np.testing.assert_allclose(pooled.var(), total, rtol=0, atol=1e-10)
    np.testing.assert_allclose(pooled.mean(), means.mean(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(pooled.aleatoric_var, aleatoric, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pooled.epistemic_var, epistemic, rtol=0, atol=1e-12)
    same = MixtureDist.from_members(np.ones((n_members, n, 1)), means[..., None], np.sqrt(variances)[..., None])
    np.testing.assert_allclose(same.weights, pooled.weights, rtol=1e-15)
    sub = pooled[:5]
    x = sub.sample(200_000, rng)
    # relative se of a sample variance at n = 2e5 is sqrt((kurtosis - 1) / n) < 0.5 %: 2 % = 4 se
    np.testing.assert_allclose(x.var(axis=1), sub.var(), rtol=0.02)


def test_decomposition_identity_mixture_members() -> None:
    rng = np.random.default_rng(11)
    n_members, n, k = 5, 1000, 3
    w = rng.dirichlet(np.ones(k), (n_members, n))
    mu = rng.normal(0.0, 2.0, (n_members, n, k))
    sd = rng.uniform(0.2, 1.5, (n_members, n, k))
    members = [MixtureDist(w[m], mu[m], sd[m]) for m in range(n_members)]
    means = np.stack([mb.mean() for mb in members])
    variances = np.stack([mb.var() for mb in members])
    aleatoric, epistemic, total = decompose(means, variances)
    np.testing.assert_allclose(aleatoric + epistemic, total, rtol=0, atol=1e-10)
    pooled = MixtureDist.from_members(w, mu, sd)
    assert pooled.n_components == n_members * k
    np.testing.assert_allclose(pooled.weights.sum(axis=1), 1.0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(pooled.var(), total, rtol=0, atol=1e-10)
    np.testing.assert_allclose(pooled.aleatoric_var, aleatoric, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pooled.epistemic_var, epistemic, rtol=0, atol=1e-12)
    y = rng.normal(size=n)  # the pooled law is the uniform mixture of the members
    np.testing.assert_allclose(pooled.cdf(y), np.mean([mb.cdf(y) for mb in members], axis=0), rtol=0, atol=1e-14)
    sub = pooled[:5]
    x = sub.sample(200_000, rng)
    np.testing.assert_allclose(x.var(axis=1), sub.var(), rtol=0.02)  # same 4 se margin as above


def test_to_gaussian_keeps_moments_and_extras() -> None:
    rng = np.random.default_rng(12)
    pooled = MixtureDist.from_members(None, rng.normal(size=(4, 50)), rng.uniform(0.5, 1.0, (4, 50)))
    g = pooled.to_gaussian()
    assert isinstance(g, GaussianDist)
    np.testing.assert_allclose(g.mean(), pooled.mean(), rtol=1e-14)
    np.testing.assert_allclose(g.var(), pooled.var(), rtol=1e-13)
    np.testing.assert_array_equal(g.aleatoric_var, pooled.aleatoric_var)
    np.testing.assert_array_equal(g.epistemic_var, pooled.epistemic_var)


def test_decompose_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        decompose(np.zeros((3, 4)), np.zeros((3, 5)))
    with pytest.raises(ValueError):
        decompose(np.zeros(4), np.zeros(4))


# ---------------------------------------------------------------------------------- Gaussian
def test_gaussian_matches_scipy() -> None:
    rng = np.random.default_rng(13)
    d = make_gaussian(rng, 10)
    y = rng.normal(0.0, 3.0, 10)
    np.testing.assert_allclose(d.cdf(y), stats.norm.cdf(y, d.mu, d.sigma), rtol=1e-13)
    np.testing.assert_allclose(d.log_prob(y), stats.norm.logpdf(y, d.mu, d.sigma), rtol=1e-13)
    np.testing.assert_allclose(d.quantile(LEVELS), stats.norm.ppf(LEVELS, d.mu[:, None], d.sigma[:, None]), rtol=1e-13)
    np.testing.assert_allclose(d.crps(y), M.crps_gaussian(d.mu, d.sigma, y), rtol=1e-15)
    far = d.mu + 40 * d.sigma
    assert np.all(np.isfinite(d.log_prob(far))) and np.all(d.cdf(far) == 1.0)
    grid = np.stack([y, y + 1.0], axis=1)
    assert d.cdf(grid).shape == (10, 2)
    np.testing.assert_allclose(d.log_prob(grid)[:, 1], d.log_prob(y + 1.0), rtol=1e-15)
    np.testing.assert_array_equal(d.pit(y), d.cdf(y))
    const = GaussianDist(np.zeros(3), 0.5)  # scalar sigma is broadcast
    np.testing.assert_array_equal(const.var(), np.full(3, 0.25))


# ----------------------------------------------------------------------------------- mixture
def test_mixture_matches_direct_formulas() -> None:
    rng = np.random.default_rng(14)
    w, mu, sd = rng.dirichlet(np.ones(4), 20) * 3.0, rng.normal(size=(20, 4)), rng.uniform(0.2, 2.0, (20, 4))
    d = MixtureDist(w, mu, sd)  # weights renormalised
    wn = w / w.sum(axis=1, keepdims=True)
    y = rng.normal(size=20)
    np.testing.assert_allclose(d.weights, wn, rtol=1e-15)
    np.testing.assert_allclose(d.cdf(y), (wn * stats.norm.cdf(y[:, None], mu, sd)).sum(1), rtol=1e-13)
    np.testing.assert_allclose(d.log_prob(y), np.log((wn * stats.norm.pdf(y[:, None], mu, sd)).sum(1)), rtol=1e-12)
    mean = (wn * mu).sum(1)
    np.testing.assert_allclose(d.mean(), mean, rtol=1e-13)
    np.testing.assert_allclose(d.var(), (wn * (sd**2 + mu**2)).sum(1) - mean**2, rtol=1e-10)
    np.testing.assert_allclose(d.crps(y), M.crps_mixture(w, mu, sd, y), rtol=1e-13)  # weights normalised twice
    with pytest.raises(ValueError):
        MixtureDist(-w, mu, sd)


def test_mixture_quantile_extreme_levels() -> None:
    d = make_mixture(np.random.default_rng(15), 4)
    q = d.quantile([0.0, 1e-30, 1e-10, 0.5, 1 - 1e-10, 1.0])
    assert np.all(np.isneginf(q[:, 0])) and np.all(np.isposinf(q[:, -1]))
    cdf = d.cdf(q[:, 1:5])
    np.testing.assert_allclose(cdf[:, 0], 1e-30, rtol=1e-9)
    np.testing.assert_allclose(cdf[:, 1], 1e-10, rtol=1e-9)
    np.testing.assert_allclose(cdf[:, 2], 0.5, rtol=1e-14)
    np.testing.assert_allclose(1.0 - cdf[:, 3], 1e-10, rtol=1e-5)  # 1 - F limited by the 1e-16 resolution of F


def test_mixture_concat_pads_with_zero_weight_components() -> None:
    rng = np.random.default_rng(16)
    a = make_mixture(rng, 3)
    b = MixtureDist(np.ones((2, 1)), rng.normal(size=(2, 1)), rng.uniform(0.5, 1.0, (2, 1)))
    c = MixtureDist.concat([a, b])
    assert c.n_components == 3 and len(c) == 5
    y = rng.normal(size=5)
    for method in ("mean", "var"):
        np.testing.assert_allclose(getattr(c, method)(), np.r_[getattr(a, method)(), getattr(b, method)()], rtol=1e-13)
    np.testing.assert_allclose(c.cdf(y), np.r_[a.cdf(y[:3]), b.cdf(y[3:])], rtol=1e-14)
    np.testing.assert_allclose(c.log_prob(y), np.r_[a.log_prob(y[:3]), b.log_prob(y[3:])], rtol=1e-14)
    np.testing.assert_allclose(c.crps(y), np.r_[a.crps(y[:3]), b.crps(y[3:])], rtol=1e-13)
    np.testing.assert_allclose(c.quantile(LEVELS), np.r_[a.quantile(LEVELS), b.quantile(LEVELS)], rtol=1e-13)
    x = c.sample(1000, rng)  # zero-weight padding components are never drawn
    assert np.all(np.abs(x[3:] - b.mus) < 8 * b.sigmas)


# ---------------------------------------------------------------------------------- quantile
def test_quantile_dist_sorts_and_counts_crossings() -> None:
    values = np.array([[0.0, 1.0, 0.5, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 2.0, 3.0]])
    original = values.copy()
    d = QuantileDist([0.1, 0.3, 0.5, 0.7, 0.9], values)
    np.testing.assert_array_equal(d.n_crossings, [1, 4, 0])
    np.testing.assert_array_equal(d.values, np.sort(values, axis=1))
    np.testing.assert_array_equal(values, original)  # input untouched
    np.testing.assert_array_equal(d[[2, 0]].n_crossings, [0, 1])
    np.testing.assert_array_equal(QuantileDist.concat([d[:1], d[1:]]).n_crossings, [1, 4, 0])
    with pytest.raises(ValueError):
        QuantileDist([0.1, 0.5], [[0.0, 1.0]])  # Q < 3
    with pytest.raises(ValueError):
        QuantileDist([0.0, 0.5, 0.9], [[0.0, 1.0, 2.0]])  # level outside (0, 1)
    with pytest.raises(ValueError):
        QuantileDist([0.1, 0.5, 0.5], [[0.0, 1.0, 2.0]])  # not strictly increasing


def test_quantile_dist_minimal_knot_spacing() -> None:
    d = QuantileDist([0.1, 0.5, 0.9], [[1.0, 1.0, 1.0], [0.0, 5.0, 5.0], [0.0, 0.0, 1e-9], [-2.0, 0.0, 2.0]])
    np.testing.assert_allclose(d.values[0], [1.0, 1.0 + 1e-6, 1.0 + 2e-6], rtol=0, atol=1e-15)
    np.testing.assert_allclose(d.values[1], [0.0, 5.0, 5.0 + 5e-6], rtol=0, atol=1e-14)  # delta = 1e-6 * range
    np.testing.assert_allclose(d.values[2], [0.0, 1e-6, 2e-6], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(d.values[3], [-2.0, 0.0, 2.0])  # rows with enough spacing are untouched
    np.testing.assert_array_equal(d.n_crossings, 0)
    assert np.all(np.diff(d.values, axis=1) > 0)
    for y in (d.values[:, 1], d.values[:, 0] - 1.0, d.values[:, 2] + 1.0):
        assert np.all(np.isfinite(d.log_prob(y)))
    assert np.all(np.isfinite(d.var())) and np.all(np.isfinite(d.crps(d.values[:, 1])))


def test_quantile_dist_density_continuous_at_outer_knots() -> None:
    d = make_quantile(np.random.default_rng(17), 5)
    tau, q = d.levels, d.values
    eps = 1e-9 * (q[:, -1] - q[:, 0])
    for knot in (q[:, 0], q[:, -1]):
        np.testing.assert_allclose(d.log_prob(knot - eps), d.log_prob(knot + eps), rtol=0, atol=1e-6)
        np.testing.assert_allclose(d.cdf(knot - eps), d.cdf(knot + eps), rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.exp(d.log_prob(q[:, 0] - eps)), (tau[1] - tau[0]) / (q[:, 1] - q[:, 0]), rtol=1e-6)
    np.testing.assert_allclose(np.exp(d.log_prob(q[:, -1] + eps)), (tau[-1] - tau[-2]) / (q[:, -1] - q[:, -2]), rtol=1e-6)
    np.testing.assert_allclose(d.cdf(q), np.broadcast_to(tau, q.shape), rtol=0, atol=1e-15)
    # exponential tails with the stated scales
    beta_l = tau[0] * (q[:, 1] - q[:, 0]) / (tau[1] - tau[0])
    np.testing.assert_allclose(d.cdf(q[:, 0] - 2 * beta_l), tau[0] * np.exp(-2.0), rtol=1e-13)
    beta_u = (1 - tau[-1]) * (q[:, -1] - q[:, -2]) / (tau[-1] - tau[-2])
    np.testing.assert_allclose(1.0 - d.cdf(q[:, -1] + 3 * beta_u), (1 - tau[-1]) * np.exp(-3.0), rtol=1e-10)
    assert np.all(np.isneginf(d.quantile([0.0])[:, 0])) and np.all(np.isposinf(d.quantile([1.0])[:, 0]))


def test_quantile_dist_dense_gaussian_grid_matches_gaussian() -> None:
    rng = np.random.default_rng(18)
    levels = np.arange(1, 100) / 100.0
    g = GaussianDist(rng.normal(0.0, 1.0, 8), rng.uniform(0.3, 2.0, 8))
    d = QuantileDist(levels, g.quantile(levels))
    assert np.all(d.n_crossings == 0)
    # deterministic approximation errors: mean ~0 (symmetric knots), var -0.8 %, CRPS < 0.3 %
    assert np.all(np.abs(d.mean() - g.mu) < 0.02 * g.sigma)
    assert np.all(np.abs(d.var() / g.var() - 1.0) < 0.02)
    for z in (-4.0, -1.5, 0.0, 0.3, 1.0, 2.5):
        y = g.mu + z * g.sigma
        assert np.all(np.abs(d.crps(y) / g.crps(y) - 1.0) < 0.02), z


def test_quantile_dist_crps_is_exact() -> None:
    rng = np.random.default_rng(19)
    d = make_quantile(rng, 6)
    q = d.values
    y = np.array([q[0, 0], q[1, -1], q[2, 2] + 0.1, q[3, 0] - 5.0, q[4, -1] + 7.0, 0.5 * (q[5, 1] + q[5, 2])])
    exact = d.crps(y)
    # int (F(z) - 1[z >= y])^2 dz by the midpoint rule on pieces split at y and at every knot
    ends = d.quantile([1e-14, 1 - 1e-14])
    lo, hi = np.minimum(ends[:, 0], y - 1.0), np.maximum(ends[:, 1], y + 1.0)
    breaks = np.sort(np.concatenate([lo[:, None], q, y[:, None], hi[:, None]], axis=1), axis=1)
    width = np.diff(breaks, axis=1)
    frac = (np.arange(4000) + 0.5) / 4000
    z = breaks[:, :-1, None] + width[:, :, None] * frac
    integrand = (d.cdf(z.reshape(6, -1)).reshape(z.shape) - (z >= y[:, None, None])) ** 2
    numeric = (integrand * (width / 4000)[:, :, None]).sum(axis=(1, 2))
    np.testing.assert_allclose(exact, numeric, rtol=1e-7)
    # the pinball-integral cross-check on the 5 knots is a coarse approximation of the same score
    assert np.all(np.abs(M.crps_quantile_function(d.levels, d.values, y) / exact - 1.0) < 0.25)


def test_quantile_and_sample_dist_default_aleatoric_var() -> None:
    rng = np.random.default_rng(20)
    for d in (make_quantile(rng, 4), make_samples(rng, 4, s=50)):
        np.testing.assert_allclose(d.aleatoric_var, d.var(), rtol=1e-15)
        assert d.epistemic_var is None
    q = make_quantile(rng, 4)
    given = QuantileDist(q.levels, q.values, aleatoric_var=np.ones(4), epistemic_var=0.0)
    np.testing.assert_array_equal(given.aleatoric_var, np.ones(4))
    np.testing.assert_array_equal(given.epistemic_var, np.zeros(4))
    assert GaussianDist([0.0], [1.0]).aleatoric_var is None


# ----------------------------------------------------------------------------------- samples
def test_sample_dist_randomised_pit_is_uniform() -> None:
    rng = np.random.default_rng(21)
    n, s = 5000, 64
    pit = SampleDist(rng.gamma(2.0, 1.0, (n, s))).pit(rng.gamma(2.0, 1.0, n), rng)
    # exactly U(0, 1) under the null; KS fails with prob. 1 % for a random seed, the fixed seed passes
    assert M.ks_uniform(pit)[1] > 0.01
    # discrete law with many ties: still uniform thanks to the randomisation of the tie rank
    xd = rng.poisson(3.0, (n, s)).astype(float)
    yd = rng.poisson(3.0, n).astype(float)
    assert M.ks_uniform(SampleDist(xd).pit(yd, rng))[1] > 0.01
    assert M.ks_uniform(SampleDist(xd).pit(yd))[1] < 1e-6  # the mid-rank PIT (U = 0.5) is not


def test_sample_dist_pit_cdf_hand_example() -> None:
    d = SampleDist([[0.0, 1.0, 1.0, 2.0]])
    assert d.pit([1.0])[0] == pytest.approx((1 + 0.5 * 3) / 5)  # one below, two ties
    assert d.pit([5.0])[0] == pytest.approx(4.5 / 5)
    assert d.pit([-1.0])[0] == pytest.approx(0.5 / 5)
    assert d.cdf(1.0)[0] == 0.75 and d.cdf(0.99)[0] == 0.25
    u = d.pit([1.0], np.random.default_rng(0))[0]
    assert 1 / 5 <= u <= 4 / 5
    assert np.isnan(d.pit([np.nan])[0]) and np.isnan(d.cdf(np.nan)[0])


def test_sample_dist_quantile_matches_numpy() -> None:
    rng = np.random.default_rng(22)
    d = make_samples(rng, 5, s=101)
    np.testing.assert_array_equal(d.quantile(LEVELS), np.quantile(d.samples, LEVELS, axis=1, method="linear").T)
    t = rng.random((5, 7))
    ref = np.stack([np.quantile(d.samples[i], t[i]) for i in range(5)])
    np.testing.assert_allclose(d.quantile(t), ref, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(d.var(), d.samples.var(axis=1, ddof=1), rtol=1e-15)


def test_sample_dist_kde_and_bandwidth() -> None:
    rng = np.random.default_rng(23)
    d = make_samples(rng, 3, s=300)
    h = d.bandwidth()
    y = rng.normal(size=3)
    lp = d.log_prob(y)
    for i in range(3):
        kde = stats.gaussian_kde(d.samples[i], bw_method=h[i] / d.samples[i].std(ddof=1))
        assert lp[i] == pytest.approx(float(kde.logpdf(y[i])[0]), rel=1e-10)
    x = d.samples[0]
    iqr = np.subtract(*np.quantile(x, [0.75, 0.25]))
    assert h[0] == pytest.approx(0.9 * min(x.std(ddof=1), iqr / 1.34) * 300 ** (-0.2), rel=1e-14)
    tied = SampleDist(np.r_[np.zeros(90), np.linspace(1.0, 2.0, 10)][None, :])  # IQR = 0 -> std is used
    assert tied.bandwidth()[0] == pytest.approx(0.9 * tied.samples.std(ddof=1) * 100 ** (-0.2))
    assert SampleDist(np.zeros((1, 10))).bandwidth()[0] == 1e-3  # floor
    assert np.isfinite(d.log_prob(d.samples.max(axis=1) + 100.0)).all()  # log-sum-exp, no underflow


def test_sample_dist_crps_fair_option() -> None:
    rng = np.random.default_rng(24)
    d = make_samples(rng, 6, s=50)
    y = rng.normal(size=6)
    np.testing.assert_allclose(d.crps(y), M.crps_samples(d.samples, y), rtol=1e-15)
    np.testing.assert_allclose(d.crps(y, fair=True), M.crps_samples(d.samples, y, fair=True), rtol=1e-15)
    assert np.all(d.crps(y, fair=True) < d.crps(y))
