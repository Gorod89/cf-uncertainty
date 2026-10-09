"""Scoring rules and calibration diagnostics (docs/design.md, section 6).

numpy/scipy only, float64, vectorised over the first axis ``N`` (sample index).
Per-sample scores are returned as ``(N,)`` arrays so that they can be averaged
with a driver bootstrap (:mod:`cf_unc.eval.bootstrap`); ``picp``,
``mean_width``, ``ks_uniform`` and ``calibration_error`` aggregate.

Units: targets, locations and scales in m/s^2; CRPS and interval widths in
m/s^2; NLL in nats.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy import special, stats

if TYPE_CHECKING:
    from cf_unc.dist import PredDist

__all__ = [
    "DEFAULT_LEVELS",
    "crps_gaussian",
    "crps_mixture",
    "crps_samples",
    "crps_quantile_function",
    "nll_gaussian",
    "nll_mixture",
    "pinball_loss",
    "picp",
    "mean_width",
    "interval_score",
    "pit_histogram",
    "ks_uniform",
    "calibration_curve",
    "calibration_error",
    "summarize",
]

DEFAULT_LEVELS = np.linspace(0.05, 0.95, 19)  # nominal levels of calibration curves (read-only use)

_LOG_2PI = float(np.log(2.0 * np.pi))
_INV_SQRT_PI = float(1.0 / np.sqrt(np.pi))
_SQRT2 = float(np.sqrt(2.0))
_TINY = float(np.finfo(np.float64).tiny)
_BLOCK = 1 << 17  # elements of one temporary (components x rows) in the chunked kernels (1 MiB)
_PARALLEL_MIN_WORK = 1 << 23  # kernel evaluations below which no worker threads are started
_MAX_THREADS = 8


# --------------------------------------------------------------------------- helpers
def _f64(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _rows(y: ArrayLike, n: int) -> np.ndarray:
    """``y`` broadcast to ``(n,)`` float64 (read-only view if broadcast)."""
    return np.broadcast_to(_f64(y), (n,))


def _mixture_arrays(
    weights: ArrayLike, mus: ArrayLike, sigmas: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validated ``(N, K)`` mixture parameters (broadcast views, no copies) and row weight sums (N,)."""
    w, mu, sd = np.broadcast_arrays(_f64(weights), _f64(mus), _f64(sigmas))
    if w.ndim != 2:
        raise ValueError(f"mixture parameters must have shape (N, K), got {w.shape}")
    if w.size and w.min() < 0:  # reductions, no (N, K) boolean temporaries
        raise ValueError("mixture weights must be non-negative")
    if sd.size and sd.min() < 0:
        raise ValueError("mixture sigmas must be non-negative")
    total = w.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("every row of the mixture weights needs a positive sum")
    return w, mu, sd, total


def _mixture_params(weights: ArrayLike, mus: ArrayLike, sigmas: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validated ``(N, K)`` mixture parameters with weights renormalised to sum 1 per row."""
    w, mu, sd, total = _mixture_arrays(weights, mus, sigmas)
    return w / total[:, None], mu, sd


def _row_chunks(n: int, per_row: int, block: int = _BLOCK, chunk: int | None = None) -> list[tuple[int, int]]:
    """Row ranges ``[a, b)`` with about ``block`` elements each (``chunk`` rows if given)."""
    step = int(chunk) if chunk is not None else max(1, block // max(1, per_row))
    if step < 1:
        raise ValueError("chunk must be a positive number of rows")
    return [(a, min(n, a + step)) for a in range(0, n, step)]


def _run_chunks(
    fn: Callable[[int, int], None], bounds: list[tuple[int, int]], work: float, n_jobs: int | None = None
) -> None:
    """Call ``fn(a, b)`` for every row range, on worker threads for large workloads.

    ``fn`` must write into disjoint slices of a preallocated output, so the result does not
    depend on the number of threads (numpy / scipy ufuncs release the GIL).
    ``n_jobs=None`` picks up to 8 threads when ``work`` (kernel evaluations) is large.
    """
    if n_jobs is None:
        workers = min(_MAX_THREADS, os.cpu_count() or 1) if work >= _PARALLEL_MIN_WORK else 1
    else:
        workers = max(1, int(n_jobs))
    workers = min(workers, len(bounds))
    if workers <= 1:
        for a, b in bounds:
            fn(a, b)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(lambda ab: fn(*ab), bounds):
            pass


def _expected_abs(m: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``E|m + s Z|`` for ``Z ~ N(0, 1)``, ``s = t / sqrt(2)``, elementwise (``t > 0``).

    This is ``A(m, s^2) = 2 s phi(m / s) + m (2 Phi(m / s) - 1)`` of Grimit et al. (2006),
    evaluated as ``(t / sqrt(pi)) exp(-u^2) + m erf(u)`` with ``u = m / t``; finite for any
    ``|m| / s`` (the exponential underflows to 0 and ``erf`` saturates at +-1).
    """
    with np.errstate(over="ignore", under="ignore"):
        u = np.divide(m, t, out=np.empty(np.broadcast_shapes(np.shape(m), np.shape(t))))  # array even if 0-d
        e = np.square(u, out=np.empty_like(u))
        np.negative(e, out=e)
        np.exp(e, out=e)
        e *= t
        e *= _INV_SQRT_PI
        special.erf(u, out=u)
        u *= m
        e += u
    return e


def _mixture_logpdf(w: np.ndarray, mu: np.ndarray, sd: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Log density of a Gaussian mixture, log-sum-exp over the last axis (no validation).

    ``w``, ``mu``, ``sd`` broadcast against ``y[..., None]``; weights must be normalised.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = (y[..., None] - mu) / sd
        terms = np.log(w) - np.log(sd) - 0.5 * z * z
    return special.logsumexp(terms, axis=-1) - 0.5 * _LOG_2PI


# ----------------------------------------------------------------------------- CRPS
def crps_gaussian(mu: ArrayLike, sigma: ArrayLike, y: ArrayLike) -> np.ndarray:
    """CRPS of ``N(mu, sigma^2)`` at ``y``, closed form; broadcastable shapes, m/s^2.

    ``sigma * [z (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi)]`` with ``z = (y - mu) / sigma``;
    ``sigma = 0`` gives the point-mass limit ``|y - mu|``.
    """
    mu_a, sigma_a, y_a = np.broadcast_arrays(_f64(mu), _f64(sigma), _f64(y))
    if np.any(sigma_a < 0):
        raise ValueError("sigma must be non-negative")
    t = np.maximum(_SQRT2 * sigma_a, _TINY)
    out = _expected_abs(y_a - mu_a, t) - _INV_SQRT_PI * sigma_a
    if np.any(sigma_a == 0):
        out = np.where(sigma_a == 0, np.abs(y_a - mu_a), out)
    return out


def crps_mixture(
    weights: ArrayLike,
    mus: ArrayLike,
    sigmas: ArrayLike,
    y: ArrayLike,
    chunk: int | None = None,
    n_jobs: int | None = None,
) -> np.ndarray:
    """CRPS of a Gaussian mixture at ``y``, closed form of Grimit et al. (2006).

    ``CRPS = sum_i w_i A(y - mu_i, s_i^2) - 1/2 sum_ij w_i w_j A(mu_i - mu_j, s_i^2 + s_j^2)``.

    weights, mus, sigmas: (N, K) (weights renormalised per row); y: (N,) -> (N,).
    The O(K^2) pair term uses the symmetry ``A(-m, .) = A(m, .)`` (K (K - 1) / 2 pairs) and is
    evaluated over row chunks of ``chunk`` rows (default: ~2^17 / K rows, i.e. about 1 MiB per
    temporary). No (N, K) copy of the inputs is made, so the extra memory is a few MiB per
    thread for any N and K (K = 250 included). ``n_jobs``: worker threads over chunks
    (None = automatic, up to 8 for large inputs); the result does not depend on it.
    Components with sigma = 0 are point masses (A(m, 0) = |m|).
    """
    w, mu, sd, total = _mixture_arrays(weights, mus, sigmas)
    n, k = w.shape
    y_a = _rows(y, n)
    out = np.empty(n)

    def run(a: int, b: int) -> None:
        with np.errstate(over="ignore", under="ignore"):
            # (K, C) component-major blocks with contiguous rows; wt is a fresh buffer (never a
            # view of the caller's weights), mt / st may be views and are only read
            wt = np.divide(w[a:b].T, total[a:b], out=np.empty((k, b - a)))
            mt = np.ascontiguousarray(mu[a:b].T)
            st = np.ascontiguousarray(sd[a:b].T)
            # E_F |X - y| = sum_k w_k A(y - mu_k, s_k^2)
            t = np.maximum(_SQRT2 * st, _TINY)
            e_xy = np.einsum("kc,kc->c", wt, _expected_abs(y_a[a:b] - mt, t))
            # 1/2 E_F |X - X'|: diagonal i == j (A(0, 2 s^2) = 2 s / sqrt(pi)) plus pairs i < j with
            # A(m, s_i^2 + s_j^2) = ts exp(-u^2) + sqrt(pi) ms erf(u), ts = t / sqrt(pi),
            # ms = m / sqrt(pi), u = ms / ts (same as _expected_abs, fewer passes).
            half_xx = _INV_SQRT_PI * np.einsum("kc,kc->c", wt * wt, st)
            v2 = st * st
            v2 *= 2.0 / np.pi
            np.maximum(v2, _TINY, out=v2)  # ts >= sqrt(2 tiny) > 0 even for point masses
            ms = mt * _INV_SQRT_PI
            acc_exp = np.zeros(b - a)
            acc_erf = np.zeros(b - a)
            for d in range(1, k):  # pairs (i, i + d): contiguous (K - d, C) blocks
                ts = v2[d:] + v2[:-d]
                np.sqrt(ts, out=ts)
                m = ms[d:] - ms[:-d]
                u = m / ts
                e = np.square(u)
                np.negative(e, out=e)
                np.exp(e, out=e)
                special.erf(u, out=u)
                ww = wt[d:] * wt[:-d]
                acc_exp += np.einsum("kc,kc,kc->c", ww, ts, e)
                acc_erf += np.einsum("kc,kc,kc->c", ww, m, u)
            out[a:b] = e_xy - (half_xx + acc_exp + np.sqrt(np.pi) * acc_erf)

    bounds = _row_chunks(n, k, chunk=chunk)
    _run_chunks(run, bounds, work=n * k * (k + 1) / 2, n_jobs=n_jobs)
    return out


def crps_samples(samples: ArrayLike, y: ArrayLike, fair: bool = False) -> np.ndarray:
    """Sample-based CRPS, O(S log S) per row via sorting (no S x S matrix).

    samples: (N, S) (a single (S,) row is accepted), y: (N,) -> (N,).
    ``fair=False``: CRPS of the empirical law, ``mean|x - y| - sum_ij |x_i - x_j| / (2 S^2)``;
    ``fair=True``: unbiased for the law the samples come from, denominator ``2 S (S - 1)``.
    Uses ``sum_ij |x_i - x_j| = 2 sum_i (2 i - S - 1) x_(i)`` (1-based order statistics).
    Samples must be finite.
    """
    x = _f64(samples)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"samples must have shape (N, S), got {x.shape}")
    n, s = x.shape
    if s < (2 if fair else 1):
        raise ValueError("fair CRPS needs at least 2 samples per row")
    y_a = _rows(y, n)
    coef = 2.0 * np.arange(1, s + 1) - (s + 1.0)
    denom = s * (s - 1.0) if fair else float(s) * s
    out = np.empty(n)
    for a, b in _row_chunks(n, s, block=1 << 20):
        xs = np.sort(x[a:b], axis=1)
        center = xs.mean(axis=1, keepdims=True)  # both terms are shift invariant
        xs -= center
        e_xy = np.abs(xs - (y_a[a:b, None] - center)).mean(axis=1)
        out[a:b] = e_xy - (xs @ coef) / denom
    return out


def crps_quantile_function(levels: ArrayLike, values: ArrayLike, y: ArrayLike, endpoints: bool = True) -> np.ndarray:
    """CRPS in pinball-integral form ``2 * int_0^1 rho_tau(y - Q(tau)) dtau`` (trapezoid rule).

    levels: (Q,) increasing in [0, 1]; values: (N, Q) quantile predictions (used as given, not
    sorted); y: (N,) -> (N,). With ``endpoints=True`` the grid is closed with the integrand's limit
    0 at tau = 0 and tau = 1 (valid for any law with a finite mean) so that the rule covers
    (0, 1); ``endpoints=False`` integrates over [tau_1, tau_Q] only. Accurate for dense grids,
    a coarse approximation otherwise (exact CRPS of the interpolated law: ``QuantileDist.crps``).
    """
    tau = _f64(levels).ravel()
    v = _f64(values)
    if v.ndim == 1:
        v = v[None, :]
    if v.ndim != 2 or v.shape[1] != tau.shape[0]:
        raise ValueError(f"values must have shape (N, {tau.shape[0]}), got {v.shape}")
    if np.any(np.diff(tau) <= 0) or tau[0] < 0 or tau[-1] > 1:
        raise ValueError("levels must be strictly increasing in [0, 1]")
    n = v.shape[0]
    u = _rows(y, n)[:, None] - v
    rho = np.maximum(tau * u, (tau - 1.0) * u)
    if endpoints:
        if tau[0] > 0:
            tau = np.concatenate([[0.0], tau])
            rho = np.concatenate([np.zeros((n, 1)), rho], axis=1)
        if tau[-1] < 1:
            tau = np.concatenate([tau, [1.0]])
            rho = np.concatenate([rho, np.zeros((n, 1))], axis=1)
    return 2.0 * np.trapezoid(rho, tau, axis=1)


# ------------------------------------------------------------------------------ NLL
def nll_gaussian(mu: ArrayLike, sigma: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Negative log density of ``N(mu, sigma^2)`` at ``y`` (nats); broadcastable shapes."""
    mu_a, sigma_a, y_a = np.broadcast_arrays(_f64(mu), _f64(sigma), _f64(y))
    z = (y_a - mu_a) / sigma_a
    return 0.5 * z * z + np.log(sigma_a) + 0.5 * _LOG_2PI


def nll_mixture(weights: ArrayLike, mus: ArrayLike, sigmas: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Negative log density of a Gaussian mixture (log-sum-exp over components, nats).

    weights, mus, sigmas: (N, K) (weights renormalised); y: (N,) -> (N,) or (N, G) -> (N, G).
    Finite for standardised residuals of any size (no density underflow); evaluated in row
    chunks, so temporaries stay small for large N * K.
    """
    w, mu, sd, total = _mixture_arrays(weights, mus, sigmas)
    n, k = w.shape
    y_a = _f64(y)
    squeeze = y_a.ndim < 2
    y_a = _rows(y_a, n)[:, None] if squeeze else np.broadcast_to(y_a, (n, y_a.shape[1]))
    out = np.empty(y_a.shape)
    for a, b in _row_chunks(n, k * y_a.shape[1], block=1 << 20):
        wn = (w[a:b] / total[a:b, None])[:, None, :]
        out[a:b] = -_mixture_logpdf(wn, mu[a:b, None, :], sd[a:b, None, :], y_a[a:b])
    return out[:, 0] if squeeze else out


# ------------------------------------------------------------------------ quantiles
def pinball_loss(y: ArrayLike, q: ArrayLike, level: ArrayLike) -> np.ndarray:
    """Pinball loss ``rho_tau(y - q) = max(tau (y - q), (tau - 1)(y - q))``, elementwise.

    y: (N,); q: (N,) or (N, Q); level: scalar or (Q,). Returns the broadcast shape.
    """
    y_a, q_a, tau = _f64(y), _f64(q), _f64(level)
    if q_a.ndim == 2 and y_a.ndim == 1:
        y_a = y_a[:, None]
    u = y_a - q_a
    return np.maximum(tau * u, (tau - 1.0) * u)


def picp(lo: ArrayLike, hi: ArrayLike, y: ArrayLike) -> float:
    """Prediction interval coverage probability: share of ``lo <= y <= hi`` (NaN rows ignored)."""
    lo_a, hi_a, y_a = np.broadcast_arrays(_f64(lo), _f64(hi), _f64(y))
    ok = ~(np.isnan(lo_a) | np.isnan(hi_a) | np.isnan(y_a))
    if not ok.any():
        return float("nan")
    return float(np.mean((y_a[ok] >= lo_a[ok]) & (y_a[ok] <= hi_a[ok])))


def mean_width(lo: ArrayLike, hi: ArrayLike) -> float:
    """Mean interval width ``hi - lo`` (m/s^2, NaN rows ignored)."""
    width = _f64(hi) - _f64(lo)
    return float(np.nanmean(width)) if np.any(~np.isnan(width)) else float("nan")


def interval_score(lo: ArrayLike, hi: ArrayLike, y: ArrayLike, coverage: float) -> np.ndarray:
    """Interval score of Gneiting & Raftery (2007) for a central interval, per sample (m/s^2).

    ``(hi - lo) + (2 / alpha) (lo - y) 1[y < lo] + (2 / alpha) (y - hi) 1[y > hi]``,
    ``alpha = 1 - coverage``; minimised in expectation by the true alpha/2, 1-alpha/2 quantiles.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie in (0, 1)")
    alpha = 1.0 - coverage
    lo_a, hi_a, y_a = np.broadcast_arrays(_f64(lo), _f64(hi), _f64(y))
    return (hi_a - lo_a) + (2.0 / alpha) * (np.maximum(lo_a - y_a, 0.0) + np.maximum(y_a - hi_a, 0.0))


# ---------------------------------------------------------------------- calibration
def _clean_pit(pit: ArrayLike) -> np.ndarray:
    p = _f64(pit).ravel()
    return p[~np.isnan(p)]


def pit_histogram(pit: ArrayLike, bins: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Histogram of PIT values on [0, 1]: ``(counts (bins,), edges (bins + 1,), density (bins,))``.

    ``density`` is normalised so that a uniform PIT gives 1 in every bin; NaNs are ignored.
    """
    p = _clean_pit(pit)
    counts, edges = np.histogram(p, bins=bins, range=(0.0, 1.0))
    total = counts.sum()
    if total == 0:
        density = np.full(counts.shape, np.nan)
    else:
        density = counts / (total * np.diff(edges))
    return counts, edges, density


def ks_uniform(pit: ArrayLike) -> tuple[float, float]:
    """Kolmogorov-Smirnov test of PIT values against U(0, 1): ``(statistic, pvalue)``."""
    res = stats.kstest(_clean_pit(pit), "uniform")
    return float(res.statistic), float(res.pvalue)


def calibration_curve(pit: ArrayLike, levels: ArrayLike = DEFAULT_LEVELS) -> dict[str, np.ndarray]:
    """Empirical vs nominal coverage from PIT values (NaNs ignored).

    Returns ``nominal`` (L,), ``empirical_cdf`` = share of PIT <= level (one-sided CDF levels)
    and ``empirical_central`` = share of PIT in ``[(1 - c)/2, (1 + c)/2]`` for nominal central
    coverage ``c = level``.
    """
    p = np.sort(_clean_pit(pit))
    lv = _f64(levels).ravel().copy()
    n = p.shape[0]
    if n == 0:
        nan = np.full(lv.shape, np.nan)
        return {"nominal": lv, "empirical_cdf": nan, "empirical_central": nan.copy()}
    emp_cdf = np.searchsorted(p, lv, side="right") / n
    inside = np.searchsorted(p, 0.5 * (1.0 + lv), side="right") - np.searchsorted(p, 0.5 * (1.0 - lv), side="left")
    return {"nominal": lv, "empirical_cdf": emp_cdf, "empirical_central": inside / n}


def calibration_error(pit: ArrayLike, levels: ArrayLike = DEFAULT_LEVELS) -> float:
    """Mean absolute deviation of ``empirical_cdf`` from the nominal levels (0 = calibrated)."""
    curve = calibration_curve(pit, levels)
    return float(np.mean(np.abs(curve["empirical_cdf"] - curve["nominal"])))


# ------------------------------------------------------------------------ per sample
def summarize(
    dist: "PredDist",
    y: ArrayLike,
    coverage: float = 0.9,
    rng: np.random.Generator | int | None = None,
) -> dict[str, np.ndarray]:
    """Per-sample scores of a predictive law at the observed targets ``y`` (N,).

    Returns (N,) float64 arrays ``nll`` (nats), ``crps`` (m/s^2), ``covered`` (1.0 if y lies in
    the central ``coverage`` interval, else 0.0), ``width`` (m/s^2), ``pit`` (randomised for
    sample laws when ``rng`` is given), plus ``sq_err`` = (y - mean)^2, ``var`` = predictive
    variance and, when the law carries them, ``aleatoric_var`` / ``epistemic_var``.
    Rows with NaN ``y`` give NaN everywhere (``covered`` included).
    """
    y_a = np.array(_rows(y, len(dist)), dtype=np.float64)
    missing = np.isnan(y_a)
    lo, hi = dist.interval(coverage)
    covered = ((y_a >= lo) & (y_a <= hi)).astype(np.float64)
    covered[missing] = np.nan
    out = {
        "nll": -dist.log_prob(y_a),
        "crps": dist.crps(y_a),
        "covered": covered,
        "width": hi - lo,
        "pit": dist.pit(y_a, rng=rng),
        "sq_err": (y_a - dist.mean()) ** 2,
        "var": dist.var(),
    }
    for name in ("aleatoric_var", "epistemic_var"):
        value = getattr(dist, name, None)
        if value is not None:
            out[name] = np.array(value, dtype=np.float64)
    return out
