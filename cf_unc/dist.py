"""Predictive distributions of the follower acceleration (docs/design.md, section 5).

numpy/scipy only (no torch), float64. The first axis of every array is the sample
index ``N``; values are accelerations in m/s^2. Every law can carry the variance
decomposition of the model that produced it, ``aleatoric_var`` / ``epistemic_var``
(N,) in (m/s^2)^2, which row indexing and ``concat`` carry along.

Methods taking targets ``y`` accept a scalar or an (N,) array and return (N,);
``cdf`` and ``log_prob`` also accept an (N, G) grid and return (N, G).
``quantile`` takes levels (Q,) shared by all rows, or (N, Q) per row, and returns (N, Q).
"""

from __future__ import annotations

import abc
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import special

from cf_unc.eval import metrics
from cf_unc.eval.metrics import _TINY, _mixture_logpdf, _mixture_params, _row_chunks, _run_chunks

__all__ = ["PredDist", "GaussianDist", "MixtureDist", "QuantileDist", "SampleDist", "decompose"]

RngLike = np.random.Generator | int | np.random.SeedSequence

_BLOCK = 1 << 20  # elements of one temporary array in chunked evaluations (8 MiB)
MIN_SPACING_REL = 1e-6  # minimal knot spacing of QuantileDist relative to max(1, row range)
_MIN_SPACING_REL = MIN_SPACING_REL  # backward-compatible alias


# --------------------------------------------------------------------------- helpers
def _f64(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _generator(rng: RngLike | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _row_selector(idx: Any) -> Any:
    """numpy row selector that never drops the sample axis (an int selects a length-1 law)."""
    if isinstance(idx, (int, np.integer)):
        return np.array([idx], dtype=np.intp)
    if isinstance(idx, slice):
        return idx
    arr = np.asarray(idx)
    if arr.dtype == bool:
        if arr.ndim != 1:
            raise IndexError("boolean row mask must be 1-D")
        return arr
    if arr.size == 0:
        return np.zeros(0, dtype=np.intp)
    if not np.issubdtype(arr.dtype, np.integer):
        raise IndexError(f"rows must be selected by int, slice, int array or bool mask, got {arr.dtype}")
    if arr.ndim > 1:
        raise IndexError("row index must be 0-D or 1-D")
    return arr.reshape(-1)


def _optional_rows(x: ArrayLike | None, n: int, name: str) -> np.ndarray | None:
    if x is None:
        return None
    arr = _f64(x)
    try:
        arr = np.broadcast_to(arr, (n,))
    except ValueError as exc:
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}") from exc
    return np.array(arr, dtype=np.float64)


def _concat_optional(arrays: Sequence[np.ndarray | None], name: str) -> np.ndarray | None:
    present = [a is not None for a in arrays]
    if all(present):
        return np.concatenate(arrays)  # type: ignore[arg-type]
    if not any(present):
        return None
    raise ValueError(f"cannot concat: {name} is set in some parts but not in others")


def _as_y(y: ArrayLike, n: int) -> tuple[np.ndarray, bool]:
    """Targets as an (N, G) array and whether the result must be squeezed back to (N,)."""
    arr = _f64(y)
    if arr.ndim == 2:
        return np.broadcast_to(arr, (n, arr.shape[1])), False
    if arr.ndim > 2:
        raise ValueError(f"y must be a scalar, (N,) or (N, G), got {arr.shape}")
    return np.broadcast_to(arr, (n,))[:, None], True


def _as_levels(levels: ArrayLike, n: int) -> np.ndarray:
    """Probability levels as an (N, Q) array (shared (Q,) levels are broadcast)."""
    arr = _f64(levels)
    if arr.ndim <= 1:
        arr = np.atleast_1d(arr)
        arr = np.broadcast_to(arr, (n, arr.shape[0]))
    elif arr.ndim == 2:
        arr = np.broadcast_to(arr, (n, arr.shape[1]))
    else:
        raise ValueError(f"levels must be (Q,) or (N, Q), got {arr.shape}")
    if not np.all((arr >= 0.0) & (arr <= 1.0)):
        raise ValueError("levels must lie in [0, 1]")
    return arr


def _searchsorted_rows(a: np.ndarray, v: np.ndarray, side: str = "right") -> np.ndarray:
    """Row-wise ``np.searchsorted``: ``a`` (N, K) sorted along axis 1, ``v`` (N, G) -> (N, G) int.

    Vectorised binary search (ceil(log2(K + 1)) passes); NaN queries give 0.
    """
    k = a.shape[1]
    lo = np.zeros(v.shape, dtype=np.intp)
    hi = np.full(v.shape, k, dtype=np.intp)
    for _ in range(int(k).bit_length()):
        active = lo < hi
        mid = (lo + hi) >> 1
        pivot = np.take_along_axis(a, np.minimum(mid, k - 1), axis=1)
        right = (pivot <= v) if side == "right" else (pivot < v)
        right &= active
        lo = np.where(right, mid + 1, lo)
        hi = np.where(active & ~right, mid, hi)
    return lo


def _squeeze(out: np.ndarray, squeeze: bool) -> np.ndarray:
    return out[:, 0] if squeeze else out


def decompose(member_means: ArrayLike, member_vars: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Law of total variance over M members: ``(aleatoric, epistemic, total)``, each (N,).

    member_means, member_vars: (M, N). ``aleatoric = mean_m Var_m``, ``epistemic = var_m E_m``
    with population variance (ddof = 0), so ``total`` equals the variance of the uniform
    mixture of the members exactly.
    """
    means = _f64(member_means)
    variances = _f64(member_vars)
    if means.shape != variances.shape or means.ndim != 2:
        raise ValueError(f"member_means and member_vars must both be (M, N), got {means.shape}, {variances.shape}")
    aleatoric = variances.mean(axis=0)
    epistemic = means.var(axis=0, ddof=0)
    return aleatoric, epistemic, aleatoric + epistemic


# ------------------------------------------------------------------------------ base
class PredDist(abc.ABC):
    """Predictive law of N scalar targets (one law per row); float64, m/s^2."""

    aleatoric_var: np.ndarray | None  # (N,)
    epistemic_var: np.ndarray | None  # (N,)

    def _init_extras(self, aleatoric_var: ArrayLike | None, epistemic_var: ArrayLike | None) -> None:
        n = len(self)
        self.aleatoric_var = _optional_rows(aleatoric_var, n, "aleatoric_var")
        self.epistemic_var = _optional_rows(epistemic_var, n, "epistemic_var")

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of rows N."""

    @abc.abstractmethod
    def mean(self) -> np.ndarray:
        """(N,) mean."""

    @abc.abstractmethod
    def var(self) -> np.ndarray:
        """(N,) total variance of the represented law."""

    @abc.abstractmethod
    def cdf(self, y: ArrayLike) -> np.ndarray:
        """P(X <= y): y (N,) -> (N,) or (N, G) -> (N, G)."""

    @abc.abstractmethod
    def log_prob(self, y: ArrayLike) -> np.ndarray:
        """Log density at y: (N,) -> (N,) or (N, G) -> (N, G)."""

    @abc.abstractmethod
    def quantile(self, levels: ArrayLike) -> np.ndarray:
        """Quantiles at levels (Q,) or (N, Q) in [0, 1] -> (N, Q)."""

    @abc.abstractmethod
    def sample(self, n: int, rng: RngLike) -> np.ndarray:
        """(N, n) independent draws (rng: Generator or seed)."""

    @abc.abstractmethod
    def crps(self, y: ArrayLike) -> np.ndarray:
        """(N,) continuous ranked probability score at y (m/s^2)."""

    @abc.abstractmethod
    def _take(self, sel: Any) -> "PredDist":
        """New law with the rows ``sel`` of the parameters (extras are set by ``__getitem__``)."""

    @classmethod
    @abc.abstractmethod
    def _concat_params(cls, parts: Sequence["PredDist"]) -> "PredDist":
        """New law stacking the parameters of ``parts`` (all of type ``cls``)."""

    def std(self) -> np.ndarray:
        """(N,) standard deviation."""
        return np.sqrt(self.var())

    def interval(self, coverage: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        """Central interval ``(quantile((1 - c) / 2), quantile((1 + c) / 2))``, each (N,)."""
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must lie in (0, 1)")
        q = self.quantile([0.5 * (1.0 - coverage), 0.5 * (1.0 + coverage)])
        return q[:, 0], q[:, 1]

    def pit(self, y: ArrayLike, rng: RngLike | None = None) -> np.ndarray:
        """(N,) probability integral transform ``F(y)`` (``rng`` only used by sample laws)."""
        return self.cdf(np.broadcast_to(_f64(y), (len(self),)))

    def __getitem__(self, idx: Any) -> "PredDist":
        sel = _row_selector(idx)
        out = self._take(sel)
        out.aleatoric_var = None if self.aleatoric_var is None else self.aleatoric_var[sel]
        out.epistemic_var = None if self.epistemic_var is None else self.epistemic_var[sel]
        return out

    @classmethod
    def concat(cls, parts: Sequence["PredDist"]) -> "PredDist":
        """Stack laws of the same class along N (all parts must agree on the extras' presence)."""
        parts = list(parts)
        if not parts:
            raise ValueError("concat needs at least one part")
        kind = type(parts[0])
        if any(type(p) is not kind for p in parts):
            raise TypeError("concat needs parts of one class")
        if not issubclass(kind, cls):
            raise TypeError(f"cannot concat {kind.__name__} parts with {cls.__name__}.concat")
        out = kind._concat_params(parts)
        out.aleatoric_var = _concat_optional([p.aleatoric_var for p in parts], "aleatoric_var")
        out.epistemic_var = _concat_optional([p.epistemic_var for p in parts], "epistemic_var")
        return out

    def __repr__(self) -> str:
        extras = [name for name in ("aleatoric_var", "epistemic_var") if getattr(self, name, None) is not None]
        return f"{type(self).__name__}(n={len(self)}{''.join(', ' + e for e in extras)})"


# -------------------------------------------------------------------------- Gaussian
class GaussianDist(PredDist):
    """``N(mu, sigma^2)`` per row; mu, sigma (N,) (sigma >= 0, 0 = point-mass limit)."""

    def __init__(
        self,
        mu: ArrayLike,
        sigma: ArrayLike,
        aleatoric_var: ArrayLike | None = None,
        epistemic_var: ArrayLike | None = None,
    ) -> None:
        mu_a, sigma_a = np.broadcast_arrays(np.atleast_1d(_f64(mu)), np.atleast_1d(_f64(sigma)))
        if mu_a.ndim != 1:
            raise ValueError(f"mu and sigma must be (N,), got {mu_a.shape}")
        if np.any(sigma_a < 0):
            raise ValueError("sigma must be non-negative")
        self.mu = np.array(mu_a, dtype=np.float64)
        self.sigma = np.array(sigma_a, dtype=np.float64)
        self._init_extras(aleatoric_var, epistemic_var)

    @classmethod
    def _raw(cls, mu: np.ndarray, sigma: np.ndarray) -> "GaussianDist":
        obj = cls.__new__(cls)
        obj.mu, obj.sigma = mu, sigma
        obj.aleatoric_var = obj.epistemic_var = None
        return obj

    def __len__(self) -> int:
        return int(self.mu.shape[0])

    def mean(self) -> np.ndarray:
        return self.mu.copy()

    def var(self) -> np.ndarray:
        return self.sigma**2

    def _z(self, y2: np.ndarray) -> np.ndarray:
        mu, sd = self.mu[:, None], self.sigma[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (y2 - mu) / sd
        if np.any(self.sigma == 0):
            z = np.where(sd > 0, z, np.where(y2 >= mu, np.inf, -np.inf))
        return z

    def cdf(self, y: ArrayLike) -> np.ndarray:
        y2, squeeze = _as_y(y, len(self))
        return _squeeze(special.ndtr(self._z(y2)), squeeze)

    def log_prob(self, y: ArrayLike) -> np.ndarray:
        y2, squeeze = _as_y(y, len(self))
        with np.errstate(divide="ignore", invalid="ignore"):
            out = -metrics.nll_gaussian(self.mu[:, None], self.sigma[:, None], y2)
        return _squeeze(out, squeeze)

    def quantile(self, levels: ArrayLike) -> np.ndarray:
        t = _as_levels(levels, len(self))
        with np.errstate(invalid="ignore"):
            q = self.mu[:, None] + self.sigma[:, None] * special.ndtri(t)
        if np.any(self.sigma == 0):
            q = np.where(self.sigma[:, None] > 0, q, self.mu[:, None])
        return q

    def sample(self, n: int, rng: RngLike) -> np.ndarray:
        g = _generator(rng)
        return self.mu[:, None] + self.sigma[:, None] * g.standard_normal((len(self), int(n)))

    def crps(self, y: ArrayLike) -> np.ndarray:
        return metrics.crps_gaussian(self.mu, self.sigma, np.broadcast_to(_f64(y), (len(self),)))

    def _take(self, sel: Any) -> "GaussianDist":
        return GaussianDist._raw(self.mu[sel], self.sigma[sel])

    @classmethod
    def _concat_params(cls, parts: Sequence[PredDist]) -> "GaussianDist":
        return cls._raw(np.concatenate([p.mu for p in parts]), np.concatenate([p.sigma for p in parts]))  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- mixture
class MixtureDist(PredDist):
    """Gaussian mixture per row: weights, mus, sigmas (N, K); weights renormalised to sum 1.

    Closed-form NLL (log-sum-exp) and CRPS (Grimit et al. 2006); quantiles by vectorised
    bisection on the CDF. Components with zero weight are allowed (used to pad ``concat``).
    """

    bisect_iters: int = 64  # >= 60: brackets shrink by 2^-64, i.e. to floating-point resolution

    def __init__(
        self,
        weights: ArrayLike,
        mus: ArrayLike,
        sigmas: ArrayLike,
        aleatoric_var: ArrayLike | None = None,
        epistemic_var: ArrayLike | None = None,
    ) -> None:
        w, mu, sd = _mixture_params(weights, mus, sigmas)
        self.weights = np.ascontiguousarray(w)
        self.mus = np.ascontiguousarray(mu)  # no copy of large contiguous float64 inputs
        self.sigmas = np.ascontiguousarray(sd)
        self._init_extras(aleatoric_var, epistemic_var)

    @classmethod
    def _raw(cls, weights: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> "MixtureDist":
        obj = cls.__new__(cls)
        obj.weights, obj.mus, obj.sigmas = weights, mus, sigmas
        obj.aleatoric_var = obj.epistemic_var = None
        return obj

    @classmethod
    def from_members(
        cls,
        member_weights: ArrayLike | None,
        member_mus: ArrayLike,
        member_sigmas: ArrayLike,
    ) -> "MixtureDist":
        """Pooled (uniform) mixture of M member mixtures with the variance decomposition attached.

        member_weights, member_mus, member_sigmas: (M, N, K); members with K = 1 may be given as
        (M, N) arrays and ``member_weights=None`` means equal weights within each member.
        Result: M * K components with weights ``w_mk / M``, ``aleatoric_var = mean_m Var_m`` and
        ``epistemic_var = var_m E_m`` (ddof = 0), so ``var() == aleatoric_var + epistemic_var``.
        """
        mu = _f64(member_mus)
        sd = _f64(member_sigmas)
        mu = mu[:, :, None] if mu.ndim == 2 else mu
        sd = sd[:, :, None] if sd.ndim == 2 else sd
        if member_weights is None:
            w = np.ones(np.broadcast_shapes(mu.shape, sd.shape))
        else:
            w = _f64(member_weights)
            w = w[:, :, None] if w.ndim == 2 else w
        w, mu, sd = np.broadcast_arrays(w, mu, sd)
        if w.ndim != 3:
            raise ValueError(f"member parameters must be (M, N, K) or (M, N), got {w.shape}")
        if (w.size and w.min() < 0) or (sd.size and sd.min() < 0):
            raise ValueError("member weights and sigmas must be non-negative")
        total = w.sum(axis=2, keepdims=True)
        if np.any(total <= 0):
            raise ValueError("every member needs a positive total weight in every row")
        m, n, k = w.shape
        member_mean = np.empty((m, n))
        member_var = np.empty((m, n))
        for a, b in _row_chunks(n, m * k, block=_BLOCK):  # member moments, bounded temporaries
            wn = w[:, a:b] / total[:, a:b]
            member_mean[:, a:b] = np.einsum("mnk,mnk->mn", wn, mu[:, a:b])
            dev = mu[:, a:b] - member_mean[:, a:b, None]
            dev *= dev
            dev += sd[:, a:b] ** 2
            member_var[:, a:b] = np.einsum("mnk,mnk->mn", wn, dev)
        aleatoric, epistemic, _ = decompose(member_mean, member_var)

        def pooled(a: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(np.moveaxis(a, 0, 1).reshape(n, m * k))

        weights = pooled(w / total)
        weights /= m
        return cls(weights, pooled(mu), pooled(sd), aleatoric_var=aleatoric, epistemic_var=epistemic)

    @property
    def n_components(self) -> int:
        return int(self.weights.shape[1])

    def __len__(self) -> int:
        return int(self.weights.shape[0])

    def mean(self) -> np.ndarray:
        return np.einsum("nk,nk->n", self.weights, self.mus)

    def var(self) -> np.ndarray:
        """Centred form ``sum_k w_k (sigma_k^2 + (mu_k - mean)^2)`` (no cancellation)."""
        n, k = self.weights.shape
        mean = self.mean()
        out = np.empty(n)
        for a, b in _row_chunks(n, k, block=_BLOCK):
            dev = self.mus[a:b] - mean[a:b, None]
            dev *= dev
            dev += self.sigmas[a:b] ** 2
            out[a:b] = np.einsum("nk,nk->n", self.weights[a:b], dev)
        return out

    def to_gaussian(self) -> GaussianDist:
        """Moment-matched Gaussian (same mean and variance), extras kept."""
        return GaussianDist(self.mean(), np.sqrt(self.var()), self.aleatoric_var, self.epistemic_var)

    def cdf(self, y: ArrayLike) -> np.ndarray:
        y2, squeeze = _as_y(y, len(self))
        n, g = y2.shape
        k = self.n_components
        out = np.empty((n, g))

        def run(a: int, b: int) -> None:
            with np.errstate(over="ignore", invalid="ignore"):
                inv_sd = 1.0 / np.maximum(self.sigmas[a:b], _TINY)
                z = (y2[a:b, :, None] - self.mus[a:b, None, :]) * inv_sd[:, None, :]
                out[a:b] = np.einsum("cgk,ck->cg", special.ndtr(z), self.weights[a:b])

        _run_chunks(run, _row_chunks(n, g * k, block=_BLOCK >> 2), work=n * g * k)
        return _squeeze(out, squeeze)

    def log_prob(self, y: ArrayLike) -> np.ndarray:
        y2, squeeze = _as_y(y, len(self))
        n, g = y2.shape
        out = np.empty((n, g))
        for a, b in _row_chunks(n, g * self.n_components, block=_BLOCK):
            out[a:b] = _mixture_logpdf(
                self.weights[a:b, None, :], self.mus[a:b, None, :], self.sigmas[a:b, None, :], y2[a:b]
            )
        return _squeeze(out, squeeze)

    def quantile(self, levels: ArrayLike) -> np.ndarray:
        """Quantiles by vectorised bisection on the CDF (``bisect_iters`` >= 60 halvings).

        Initial bracket per level: ``[min_k(mu_k + sigma_k z), max_k(mu_k + sigma_k z)]`` with
        ``z = Phi^-1(level)`` over components with positive weight. It always contains the
        mixture quantile (F <= level at the left end, F >= level at the right end) and lies
        inside the contract's ``[min(mu - 12 sigma), max(mu + 12 sigma)]`` for levels in
        [Phi(-12), Phi(12)]. The loop stops early only once every bracket has collapsed to
        adjacent floating-point numbers, where further halvings are no-ops.
        """
        n, k = self.weights.shape
        t = _as_levels(levels, n)
        nq = t.shape[1]
        out = np.empty((n, nq))
        z_level = special.ndtri(t)  # +-inf at levels 0 / 1 -> quantile +-inf

        def run(a: int, b: int) -> None:
            with np.errstate(over="ignore", invalid="ignore"):
                w = self.weights[a:b]
                mu = self.mus[a:b, None, :]
                sd = np.maximum(self.sigmas[a:b, None, :], _TINY)
                inv_sd = 1.0 / sd
                level = t[a:b]
                comp_q = mu + sd * z_level[a:b, :, None]  # (C, Q, K) component quantiles
                pos = w[:, None, :] > 0
                lo = np.where(pos, comp_q, np.inf).min(axis=2)
                hi = np.where(pos, comp_q, -np.inf).max(axis=2)
                del comp_q, sd
                for _ in range(self.bisect_iters):
                    mid = 0.5 * (lo + hi)
                    if np.all((mid == lo) | (mid == hi)):
                        break
                    cdf_mid = np.einsum("cqk,ck->cq", special.ndtr((mid[:, :, None] - mu) * inv_sd), w)
                    below = cdf_mid < level
                    lo = np.where(below, mid, lo)
                    hi = np.where(below, hi, mid)
                out[a:b] = 0.5 * (lo + hi)

        _run_chunks(run, _row_chunks(n, nq * k, block=_BLOCK >> 2), work=n * nq * k * 40)
        return out

    def sample(self, n: int, rng: RngLike) -> np.ndarray:
        """(N, n) draws: component by inverse CDF of the weights, then a Gaussian draw.

        Generated in row chunks (bounded temporaries); reproducible for a given rng state.
        """
        g = _generator(rng)
        rows, k = self.weights.shape
        n = int(n)
        out = np.empty((rows, n))
        for a, b in _row_chunks(rows, n + k, block=_BLOCK):
            cum = np.cumsum(self.weights[a:b], axis=1)
            cum /= cum[:, -1:]  # last entry exactly 1, so u < 1 never runs past the last component
            comp = np.minimum(_searchsorted_rows(cum, g.random((b - a, n)), side="right"), k - 1)
            out[a:b] = g.standard_normal((b - a, n))
            out[a:b] *= np.take_along_axis(self.sigmas[a:b], comp, axis=1)
            out[a:b] += np.take_along_axis(self.mus[a:b], comp, axis=1)
        return out

    def crps(self, y: ArrayLike) -> np.ndarray:
        return metrics.crps_mixture(self.weights, self.mus, self.sigmas, np.broadcast_to(_f64(y), (len(self),)))

    def _take(self, sel: Any) -> "MixtureDist":
        return MixtureDist._raw(self.weights[sel], self.mus[sel], self.sigmas[sel])

    @classmethod
    def _concat_params(cls, parts: Sequence[PredDist]) -> "MixtureDist":
        """Stack rows; parts with fewer components are padded with zero-weight components."""
        k = max(p.n_components for p in parts)  # type: ignore[attr-defined]

        def pad(a: np.ndarray, fill: float) -> np.ndarray:
            if a.shape[1] == k:
                return a
            return np.concatenate([a, np.full((a.shape[0], k - a.shape[1]), fill)], axis=1)

        return cls._raw(
            np.concatenate([pad(p.weights, 0.0) for p in parts]),  # type: ignore[attr-defined]
            np.concatenate([pad(p.mus, 0.0) for p in parts]),  # type: ignore[attr-defined]
            np.concatenate([pad(p.sigmas, 1.0) for p in parts]),  # type: ignore[attr-defined]
        )


# -------------------------------------------------------------------------- quantile
class QuantileDist(PredDist):
    """Law given by quantiles ``values`` (N, Q) at ``levels`` (Q,), Q >= 3, levels in (0, 1).

    Piecewise-linear quantile function between the knots (piecewise-constant density) and
    exponential tails whose density is continuous at the outer knots:

    * lower: ``F(y) = tau_1 exp((y - q_1) / beta_L)``, ``beta_L = tau_1 (q_2 - q_1) / (tau_2 - tau_1)``
    * upper: ``1 - F(y) = (1 - tau_Q) exp(-(y - q_Q) / beta_U)``,
      ``beta_U = (1 - tau_Q) (q_Q - q_{Q-1}) / (tau_Q - tau_{Q-1})``.

    On construction the values are sorted along Q (``n_crossings`` (N,) counts the adjacent
    pairs that were out of order) and knots closer than ``1e-6 * max(1, row range)`` are pushed
    apart, so densities stay finite. ``mean``, ``var``, ``cdf``, ``quantile``, ``log_prob`` and
    ``crps`` are exact closed forms of this law. ``aleatoric_var`` defaults to ``var()``.
    """

    def __init__(
        self,
        levels: ArrayLike,
        values: ArrayLike,
        aleatoric_var: ArrayLike | None = None,
        epistemic_var: ArrayLike | None = None,
    ) -> None:
        tau = _f64(levels)
        if tau.ndim != 1 or tau.shape[0] < 3:
            raise ValueError(f"levels must be (Q,) with Q >= 3, got {tau.shape}")
        if not (tau[0] > 0.0 and tau[-1] < 1.0 and np.all(np.diff(tau) > 0)):
            raise ValueError("levels must be strictly increasing inside (0, 1)")
        q = _f64(values)
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2 or q.shape[1] != tau.shape[0]:
            raise ValueError(f"values must have shape (N, {tau.shape[0]}), got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError("quantile values must be finite")
        self.levels = tau.copy()
        self.n_crossings = np.count_nonzero(np.diff(q, axis=1) < 0, axis=1)
        self.values = self._enforce_spacing(np.sort(q, axis=1))
        self._finalize()
        self._init_extras(self.var() if aleatoric_var is None else aleatoric_var, epistemic_var)

    @staticmethod
    def _enforce_spacing(q: np.ndarray) -> np.ndarray:
        """Push sorted knots apart to a spacing >= 1e-6 * max(1, row range) (in place)."""
        delta = _MIN_SPACING_REL * np.maximum(1.0, q[:, -1] - q[:, 0])
        bad = np.any(np.diff(q, axis=1) < delta[:, None], axis=1)
        if np.any(bad):
            shift = delta[bad, None] * np.arange(q.shape[1])
            q[bad] = np.maximum.accumulate(q[bad] - shift, axis=1) + shift
        return q

    def _finalize(self) -> None:
        tau, q = self.levels, self.values
        self.beta_lower = tau[0] * (q[:, 1] - q[:, 0]) / (tau[1] - tau[0])
        self.beta_upper = (1.0 - tau[-1]) * (q[:, -1] - q[:, -2]) / (tau[-1] - tau[-2])
        self._density = np.diff(tau)[None, :] / np.diff(q, axis=1)  # (N, Q - 1), segment densities

    @classmethod
    def _raw(cls, levels: np.ndarray, values: np.ndarray, n_crossings: np.ndarray) -> "QuantileDist":
        obj = cls.__new__(cls)
        obj.levels, obj.values, obj.n_crossings = levels, values, n_crossings
        obj._finalize()
        obj.aleatoric_var = obj.epistemic_var = None
        return obj

    def __len__(self) -> int:
        return int(self.values.shape[0])

    # moments ---------------------------------------------------------------
    def _raw_moments(self, shift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact E[X - shift] and E[(X - shift)^2], each (N,)."""
        tau = self.levels
        q = self.values - shift[:, None]
        bl, bu = self.beta_lower, self.beta_upper
        t1, p = tau[0], 1.0 - tau[-1]
        h = np.diff(tau)
        a, b = q[:, :-1], q[:, 1:]
        lower_q, upper_q = q[:, 0] - bl, q[:, -1] + bu
        m1 = t1 * lower_q + (0.5 * h * (a + b)).sum(axis=1) + p * upper_q
        m2 = (
            t1 * (lower_q**2 + bl**2)
            + (h * (a * a + a * b + b * b)).sum(axis=1) / 3.0
            + p * (upper_q**2 + bu**2)
        )
        return m1, m2

    def mean(self) -> np.ndarray:
        center = self.values[:, self.values.shape[1] // 2]
        m1, _ = self._raw_moments(center)
        return center + m1

    def var(self) -> np.ndarray:
        m1, m2 = self._raw_moments(self.mean())
        return np.maximum(m2 - m1 * m1, 0.0)

    # law -------------------------------------------------------------------
    def cdf(self, y: ArrayLike) -> np.ndarray:
        y2, squeeze = _as_y(y, len(self))
        tau, q = self.levels, self.values
        nq = tau.shape[0]
        j = _searchsorted_rows(q, y2, side="right")  # knots <= y: 0 lower tail, Q upper tail
        s = np.clip(j - 1, 0, nq - 2)
        q0 = np.take_along_axis(q, s, axis=1)
        q1 = np.take_along_axis(q, s + 1, axis=1)
        inside = tau[s] + (tau[s + 1] - tau[s]) * (y2 - q0) / (q1 - q0)
        lower = tau[0] * np.exp(np.minimum(y2 - q[:, :1], 0.0) / self.beta_lower[:, None])
        upper = 1.0 - (1.0 - tau[-1]) * np.exp(-np.maximum(y2 - q[:, -1:], 0.0) / self.beta_upper[:, None])
        out = np.where(j == 0, lower, np.where(j == nq, upper, inside))
        return _squeeze(out, squeeze)

    def log_prob(self, y: ArrayLike) -> np.ndarray:
        y2, squeeze = _as_y(y, len(self))
        tau, q = self.levels, self.values
        nq = tau.shape[0]
        bl, bu = self.beta_lower[:, None], self.beta_upper[:, None]
        j = _searchsorted_rows(q, y2, side="right")
        s = np.clip(j - 1, 0, nq - 2)
        inside = np.log(np.take_along_axis(self._density, s, axis=1))
        lower = np.log(tau[0] / bl) + (y2 - q[:, :1]) / bl
        upper = np.log((1.0 - tau[-1]) / bu) - (y2 - q[:, -1:]) / bu
        out = np.where(j == 0, lower, np.where(j == nq, upper, inside))
        return _squeeze(out, squeeze)

    def quantile(self, levels: ArrayLike) -> np.ndarray:
        """Exact inverse of ``cdf`` for any levels (0 -> -inf, 1 -> +inf)."""
        t = _as_levels(levels, len(self))
        tau, q = self.levels, self.values
        nq = tau.shape[0]
        s = np.clip(np.searchsorted(tau, t, side="right") - 1, 0, nq - 2)
        q0 = np.take_along_axis(q, s, axis=1)
        q1 = np.take_along_axis(q, s + 1, axis=1)
        inside = q0 + (q1 - q0) * (t - tau[s]) / (tau[s + 1] - tau[s])
        with np.errstate(divide="ignore"):
            lower = q[:, :1] + self.beta_lower[:, None] * np.log(np.minimum(t, tau[0]) / tau[0])
            upper = q[:, -1:] - self.beta_upper[:, None] * np.log(np.minimum(1.0 - t, 1.0 - tau[-1]) / (1.0 - tau[-1]))
        return np.where(t < tau[0], lower, np.where(t > tau[-1], upper, inside))

    def sample(self, n: int, rng: RngLike) -> np.ndarray:
        """(N, n) draws by inverse CDF (row chunks, bounded temporaries)."""
        g = _generator(rng)
        rows, n = len(self), int(n)
        out = np.empty((rows, n))
        for a, b in _row_chunks(rows, n, block=_BLOCK):
            u = g.random((b - a, n))
            np.maximum(u, _TINY, out=u)  # rng.random is in [0, 1); keep draws finite
            out[a:b] = self._take(slice(a, b)).quantile(u)
        return out

    def crps(self, y: ArrayLike) -> np.ndarray:
        """Exact CRPS of the interpolated law (closed-form piecewise integration).

        ``CRPS = E|X - y| - 1/2 E|X - X'|`` with ``E|X - y| = int_0^1 |Q(tau) - y| dtau`` and
        ``1/2 E|X - X'| = int_0^1 (2 tau - 1) Q(tau) dtau``, both integrated exactly over the
        linear segments and the two exponential tails; identical to
        ``int (F(z) - 1[z >= y])^2 dz``.
        """
        n = len(self)
        tau = self.levels
        center = self.values[:, self.values.shape[1] // 2]  # both terms are shift invariant
        q = self.values - center[:, None]
        yc = np.broadcast_to(_f64(y), (n,)) - center
        bl, bu = self.beta_lower, self.beta_upper
        t1, p = tau[0], 1.0 - tau[-1]
        h = np.diff(tau)
        a, b = q[:, :-1], q[:, 1:]
        # 1/2 E|X - X'| (Simpson's rule is exact for the quadratic integrand on each segment)
        t_mid = 0.5 * (tau[:-1] + tau[1:])
        spread_mid = (h / 6.0 * ((2 * tau[:-1] - 1) * a + 2.0 * (2 * t_mid - 1) * (a + b) + (2 * tau[1:] - 1) * b)).sum(axis=1)
        spread = t1 * (q[:, 0] * (t1 - 1.0) + bl * (1.0 - 0.5 * t1)) + spread_mid + p * (q[:, -1] * (1.0 - p) + bu * (1.0 - 0.5 * p))
        # E|X - y| on the segments: mean of |linear function| on [0, 1]
        da, db = a - yc[:, None], b - yc[:, None]
        same_sign = da * db >= 0
        seg = np.where(same_sign, 0.5 * np.abs(da + db), (da * da + db * db) / (2.0 * (db - da)))
        abs_mid = (h * seg).sum(axis=1)
        # tails: below q_1 (d = y - q_1) and above q_Q (e = y - q_Q)
        d = yc - q[:, 0]
        abs_lower = t1 * np.where(d >= 0, d + bl, 2.0 * bl * np.exp(np.minimum(d, 0.0) / bl) - bl - d)
        e = yc - q[:, -1]
        abs_upper = p * np.where(e <= 0, bu - e, 2.0 * bu * np.exp(-np.maximum(e, 0.0) / bu) - bu + e)
        return abs_lower + abs_mid + abs_upper - spread

    def _take(self, sel: Any) -> "QuantileDist":
        return QuantileDist._raw(self.levels, self.values[sel], self.n_crossings[sel])

    @classmethod
    def _concat_params(cls, parts: Sequence[PredDist]) -> "QuantileDist":
        levels = parts[0].levels  # type: ignore[attr-defined]
        if any(not np.array_equal(p.levels, levels) for p in parts):  # type: ignore[attr-defined]
            raise ValueError("cannot concat QuantileDist parts with different levels")
        return cls._raw(
            levels,
            np.concatenate([p.values for p in parts]),  # type: ignore[attr-defined]
            np.concatenate([p.n_crossings for p in parts]),  # type: ignore[attr-defined]
        )


# ---------------------------------------------------------------------------- sample
class SampleDist(PredDist):
    """Empirical law of S samples per row, samples (N, S), S >= 2.

    ``cdf`` is the ECDF (share <= y), ``quantile`` = ``np.quantile(..., method="linear")``,
    ``var`` uses ddof = 1, ``crps`` is sample based. ``log_prob`` is **approximate**: a Gaussian
    KDE with Silverman's bandwidth per row. ``aleatoric_var`` defaults to ``var()``.
    """

    def __init__(
        self,
        samples: ArrayLike,
        aleatoric_var: ArrayLike | None = None,
        epistemic_var: ArrayLike | None = None,
    ) -> None:
        x = _f64(samples)
        if x.ndim != 2 or x.shape[1] < 2:
            raise ValueError(f"samples must have shape (N, S) with S >= 2, got {x.shape}")
        self.samples = x
        self._sorted: np.ndarray | None = None
        self._init_extras(self.var() if aleatoric_var is None else aleatoric_var, epistemic_var)

    @classmethod
    def _raw(cls, samples: np.ndarray, sorted_samples: np.ndarray | None) -> "SampleDist":
        obj = cls.__new__(cls)
        obj.samples, obj._sorted = samples, sorted_samples
        obj.aleatoric_var = obj.epistemic_var = None
        return obj

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[1])

    @property
    def sorted_samples(self) -> np.ndarray:
        """(N, S) samples sorted along S (cached)."""
        if self._sorted is None:
            self._sorted = np.sort(self.samples, axis=1)
        return self._sorted

    def mean(self) -> np.ndarray:
        return self.samples.mean(axis=1)

    def var(self) -> np.ndarray:
        return self.samples.var(axis=1, ddof=1)

    def bandwidth(self) -> np.ndarray:
        """(N,) Silverman bandwidth ``0.9 min(std, IQR / 1.34) S^(-1/5)``, floored at 1e-3.

        If the IQR is 0 (many tied samples) the standard deviation is used instead.
        """
        x = self.samples
        std = x.std(axis=1, ddof=1)
        q25, q75 = np.quantile(x, [0.25, 0.75], axis=1)
        iqr = (q75 - q25) / 1.34
        spread = np.where(iqr > 0, np.minimum(std, iqr), std)
        return np.maximum(0.9 * spread * self.n_samples ** (-0.2), 1e-3)

    def cdf(self, y: ArrayLike) -> np.ndarray:
        y2, squeeze = _as_y(y, len(self))
        out = _searchsorted_rows(self.sorted_samples, y2, side="right") / self.n_samples
        out = np.where(np.isnan(y2), np.nan, out)
        return _squeeze(out, squeeze)

    def log_prob(self, y: ArrayLike) -> np.ndarray:
        """Approximate log density: Gaussian KDE, Silverman bandwidth per row (log-sum-exp)."""
        y2, squeeze = _as_y(y, len(self))
        n, g = y2.shape
        s = self.n_samples
        h = self.bandwidth()
        out = np.empty((n, g))
        for a, b in _row_chunks(n, g * s, block=_BLOCK):
            z = (y2[a:b, :, None] - self.samples[a:b, None, :]) / h[a:b, None, None]
            out[a:b] = special.logsumexp(-0.5 * z * z, axis=2) - np.log(s * h[a:b])[:, None] - 0.5 * np.log(2 * np.pi)
        return _squeeze(out, squeeze)

    def quantile(self, levels: ArrayLike) -> np.ndarray:
        lv = _f64(levels)
        if lv.ndim <= 1:
            lv = np.atleast_1d(lv)
            if not np.all((lv >= 0.0) & (lv <= 1.0)):
                raise ValueError("levels must lie in [0, 1]")
            return np.quantile(self.samples, lv, axis=1, method="linear").T.reshape(len(self), lv.shape[0])
        t = _as_levels(lv, len(self))  # per-row levels: same linear rule on the sorted samples
        xs = self.sorted_samples
        s = self.n_samples
        pos = (s - 1) * t
        i = np.clip(np.floor(pos).astype(np.intp), 0, s - 2)
        frac = pos - i
        lo = np.take_along_axis(xs, i, axis=1)
        hi = np.take_along_axis(xs, i + 1, axis=1)
        diff = hi - lo
        return np.where(frac >= 0.5, hi - diff * (1.0 - frac), lo + diff * frac)

    def sample(self, n: int, rng: RngLike) -> np.ndarray:
        """(N, n) draws with replacement from the stored samples."""
        idx = _generator(rng).integers(0, self.n_samples, size=(len(self), int(n)))
        return np.take_along_axis(self.samples, idx, axis=1)

    def crps(self, y: ArrayLike, fair: bool = False) -> np.ndarray:
        """Sample CRPS (``fair=True``: unbiased for the sampled law, ``2 S (S - 1)`` denominator)."""
        return metrics.crps_samples(self.samples, np.broadcast_to(_f64(y), (len(self),)), fair=fair)

    def pit(self, y: ArrayLike, rng: RngLike | None = None) -> np.ndarray:
        """Randomised PIT ``(#{x < y} + U (#{x == y} + 1)) / (S + 1)``; U = 0.5 if ``rng`` is None.

        Exactly U(0, 1) when y and the samples are i.i.d. draws of one law (ties included).
        """
        n, s = len(self), self.n_samples
        y2 = np.broadcast_to(_f64(y), (n,))[:, None]
        below = _searchsorted_rows(self.sorted_samples, y2, side="left")[:, 0]
        equal = _searchsorted_rows(self.sorted_samples, y2, side="right")[:, 0] - below
        u = np.full(n, 0.5) if rng is None else _generator(rng).random(n)
        out = (below + u * (equal + 1)) / (s + 1.0)
        return np.where(np.isnan(y2[:, 0]), np.nan, out)

    def _take(self, sel: Any) -> "SampleDist":
        return SampleDist._raw(self.samples[sel], None if self._sorted is None else self._sorted[sel])

    @classmethod
    def _concat_params(cls, parts: Sequence[PredDist]) -> "SampleDist":
        s = parts[0].n_samples  # type: ignore[attr-defined]
        if any(p.n_samples != s for p in parts):  # type: ignore[attr-defined]
            raise ValueError("cannot concat SampleDist parts with different numbers of samples")
        return cls._raw(np.concatenate([p.samples for p in parts]), None)  # type: ignore[attr-defined]
