"""Evaluation of experiment E3 (docs/design.md, section 13.3), hypothesis H3.2. numpy / pandas only.

Conventions (choices the contract leaves open are marked *):

* Model law of a (profile, vehicle index): the amplitudes (or speed standard deviations) of every rollout of every
  seed of that profile, pooled (``S = R x n_seeds`` samples). Empirical value: the measured follower of the same index.
* CRPS: fair sample CRPS (``crps_samples(..., fair=True)``) of the pooled law at the empirical value [m/s]; the ratio
  CRPS is the CRPS of ``A_i / A_0`` at the empirical ratio (= the amplitude CRPS divided by the leader amplitude).
* Aggregation*: per profile the mean over its measured indices (``all``; ``1-4``: indices 1..4 only), then the mean
  over profiles; every profile weighs the same. Intervals: percentile bootstrap over profiles (``n_boot`` replicates;
  the bootstrap seed depends on the keys of the row - set, subset, index, window - only, so every model and variant is
  resampled with the same profiles).
* H3.2: relative reduction ``1 - mean_p CRPS_model / mean_p CRPS_reference`` over the profiles both models have,
  bootstrap of the ratio of means with the same resampled profiles for both (paired); ``supported`` when the lower
  bound >= ``support_min`` (0.20), ``refuted`` when the upper bound < ``refute_max`` (0.05), else ``inconclusive``; the
  same rule on the point estimate is reported as ``verdict_point``.
* Intervals of the amplitude: ``[q05, q95]`` of the pooled law with ``(S + 1)`` plotting positions (numpy ``weibull``,
  as the rollout tables of E2), median ``q50``; score and corrected interval of design.md 11.4
  (``cf_unc.rollout.calibration``). Static correction*: leave one profile out, per index (the conformal quantile of the
  scores of the other profiles at that index). ACI*: per index, online over the profiles of the extended set in one
  seeded random order (the same for every model and variant), ``alpha``, ``gamma``, ``n_min`` of the config.
* Growth: ``sd_i`` = mean over rollouts, seeds and profiles; exponent* ``beta`` of ``log sd_i = log sd_1 + beta log i``
  by ordinary least squares over the indices with a finite positive mean (free intercept); interval: bootstrap over
  profiles (the mean curve and the fit are recomputed for every replicate); concavity: the share of negative second
  differences ``sd_{i+1} - 2 sd_i + sd_{i-1}`` of the mean curve. Empirical curve: mean over the profiles that measured
  the index; fitted over the indices measured in at least ``min_profiles`` (3) profiles*.
"""

from __future__ import annotations

import math
import zlib
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike

from cf_unc.eval.bootstrap import bootstrap_by_group, bootstrap_ratio_by_group
from cf_unc.eval.metrics import crps_samples
from cf_unc.rollout.calibration import aci_run, conformal_quantile, corrected_interval, interval_score

__all__ = [
    "VERDICTS",
    "boot_seed",
    "coverage_by_method",
    "fair_crps",
    "growth_curve",
    "growth_exponent",
    "h32_verdict",
    "loo_static_correction",
    "mean_with_ci",
    "profile_mean",
    "profile_order",
    "relative_reduction",
    "sample_quantiles",
    "second_difference_share",
]

VERDICTS = ("supported", "refuted", "inconclusive")


def boot_seed(base: int, *keys: object) -> int:
    """Bootstrap seed of a table row: depends on the base seed and the keys only (not on the model)."""
    text = "\x1f".join(str(k) for k in keys).encode("utf-8")
    return int(np.random.SeedSequence([int(base) & 0xFFFFFFFF, zlib.crc32(text)]).generate_state(1)[0])


# ------------------------------------------------------------------------------------------- scores
def fair_crps(samples: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Fair sample CRPS (N,) of rows ``samples`` (N, S) at ``y`` (N,); NaN where ``y`` is NaN (index not measured)."""
    x = np.asarray(samples, dtype=np.float64)
    x = x[None, :] if x.ndim == 1 else x
    yy = np.broadcast_to(np.asarray(y, dtype=np.float64), (x.shape[0],))
    out = np.full(x.shape[0], np.nan)
    ok = np.isfinite(yy)
    if ok.any():
        out[ok] = crps_samples(x[ok], yy[ok], fair=x.shape[1] > 1)
    return out


def sample_quantiles(samples: ArrayLike, levels: Sequence[float] = (0.05, 0.5, 0.95)) -> np.ndarray:
    """(N, Q) quantiles of rows (N, S) with (S + 1) plotting positions (numpy ``method="weibull"``)."""
    x = np.asarray(samples, dtype=np.float64)
    x = x[None, :] if x.ndim == 1 else x
    return np.quantile(x, list(levels), axis=1, method="weibull").T


def profile_mean(values: ArrayLike, indices: Sequence[int] | None = None) -> np.ndarray:
    """(P,) mean of ``values`` (P, V) over the finite entries of the given 1-based indices (all when None); NaN for a
    profile without any."""
    v = np.asarray(values, dtype=np.float64)
    if indices is not None:
        cols = [int(i) - 1 for i in indices if 1 <= int(i) <= v.shape[1]]
        v = v[:, cols]
    ok = np.isfinite(v)
    n = ok.sum(axis=1)
    total = np.where(ok, v, 0.0).sum(axis=1)
    return np.where(n > 0, total / np.maximum(n, 1), np.nan)


# --------------------------------------------------------------------------------------------- H3.2
def h32_verdict(estimate: float, lo: float, hi: float, support_min: float = 0.20, refute_max: float = 0.05) -> tuple[str, str]:
    """(verdict of the interval, verdict of the point estimate): ``supported`` when the lower bound (estimate) is
    >= ``support_min``, ``refuted`` when the upper bound (estimate) is < ``refute_max``, else ``inconclusive`` (also
    when a number is NaN)."""

    def rule(low: float, high: float) -> str:
        if math.isfinite(low) and low >= support_min - 1e-12:
            return "supported"
        if math.isfinite(high) and high < refute_max:
            return "refuted"
        return "inconclusive"

    return rule(float(lo), float(hi)), rule(float(estimate), float(estimate))


def relative_reduction(
    model: ArrayLike, reference: ArrayLike, groups: ArrayLike, n_boot: int = 1000, seed: int = 0, ci: float = 0.95
) -> dict[str, Any]:
    """``1 - mean(model) / mean(reference)`` over the profiles where both are finite, paired bootstrap over profiles.

    model, reference: (P,) per-profile scores (lower is better); groups: (P,) profile ids. Returns ``estimate``,
    ``lo``, ``hi`` (percentile interval of the reduction), ``crps_model``, ``crps_reference``, ``n_profiles``.
    """
    m, r = np.asarray(model, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    g = np.asarray(groups)
    ok = np.isfinite(m) & np.isfinite(r)
    if ok.sum() == 0:
        return {"estimate": np.nan, "lo": np.nan, "hi": np.nan, "crps_model": np.nan, "crps_reference": np.nan,
                "n_profiles": 0}  # fmt: skip
    res = bootstrap_ratio_by_group(m[ok], r[ok], g[ok], n_boot=n_boot, seed=seed, ci=ci)
    return {
        "estimate": 1.0 - float(res["estimate"]),
        "lo": 1.0 - float(res["hi"]),
        "hi": 1.0 - float(res["lo"]),
        "crps_model": float(res["num_mean"]),
        "crps_reference": float(res["den_mean"]),
        "n_profiles": int(ok.sum()),
    }


def mean_with_ci(values: ArrayLike, groups: ArrayLike, n_boot: int, seed: int, ci: float) -> dict[str, Any]:
    """Mean of per-profile values (P,) with a percentile bootstrap interval over profiles (NaN entries dropped)."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return {"estimate": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    res = bootstrap_by_group(v[ok], np.asarray(groups)[ok], n_boot=n_boot, seed=seed, ci=ci)
    return {"estimate": float(res["estimate"]), "lo": float(res["lo"]), "hi": float(res["hi"]), "n": int(ok.sum())}


# ---------------------------------------------------------------------------------------- intervals
def loo_static_correction(scores: ArrayLike, alpha: float) -> np.ndarray:
    """(P,) leave-one-out static corrections: the conformal quantile of the other finite scores (NaN rows: NaN)."""
    s = np.asarray(scores, dtype=np.float64)
    out = np.full(s.shape, np.nan)
    for p in np.flatnonzero(np.isfinite(s)):
        others = np.delete(s, p)
        out[p] = conformal_quantile(others[np.isfinite(others)], alpha)
    return out


def profile_order(ids: Sequence[str], seed: int) -> list[str]:
    """The seeded random order of the profiles used by ACI (a permutation of ``ids``)."""
    perm = np.random.default_rng(int(seed)).permutation(len(ids))
    return [str(ids[i]) for i in perm]


def coverage_by_method(
    y: ArrayLike, lo: ArrayLike, mid: ArrayLike, hi: ArrayLike, order: Sequence[int], alpha: float = 0.1,
    gamma: float = 0.01, n_min: int = 5,
) -> dict[str, dict[str, np.ndarray]]:  # fmt: skip
    """Covered indicators and widths (P,) of the raw, leave-one-out static and ACI-online corrected intervals.

    y (P,) empirical values (NaN: not measured, the row gets NaN), lo / mid / hi (P,) q05 / q50 / q95 of the model
    law, order: positions of the profiles in ACI order (a permutation of the rows). Returns
    ``{method: {"covered", "width", "q"}}`` for ``raw``, ``static_loo`` and ``aci_online``.
    """
    y, lo, mid, hi = (np.asarray(a, dtype=np.float64) for a in (y, lo, mid, hi))
    ok = np.isfinite(y)
    score = np.where(ok, interval_score(np.where(ok, y, 0.0), lo, mid, hi), np.nan)

    def pack(q: np.ndarray) -> dict[str, np.ndarray]:
        a, b = corrected_interval(lo, mid, hi, np.where(np.isfinite(q), q, 0.0))
        width = np.where(np.isinf(q), np.inf, np.maximum(b - a, 0.0))
        return {"covered": np.where(ok, (score <= q).astype(np.float64), np.nan), "width": np.where(ok, width, np.nan),
                "q": np.where(ok, q, np.nan)}  # fmt: skip

    out = {"raw": pack(np.zeros_like(score))}
    out["static_loo"] = pack(loo_static_correction(score, alpha))
    seq = [int(i) for i in order if ok[int(i)]]
    q_aci = np.full(score.shape, np.nan)
    if seq:
        run = aci_run(score[seq], alpha, gamma, n_min)
        q_aci[seq] = run["q"]
    out["aci_online"] = pack(q_aci)
    return out


# -------------------------------------------------------------------------------------------- growth
def growth_exponent(curve: ArrayLike, index: ArrayLike | None = None) -> tuple[float, float]:
    """(beta, log sd_1) of ``log sd_i = c + beta log i`` by least squares over the finite positive entries of
    ``curve`` (V,) at ``index`` (default 1..V); NaN with fewer than two usable points."""
    c = np.asarray(curve, dtype=np.float64)
    i = np.arange(1, c.size + 1, dtype=np.float64) if index is None else np.asarray(index, dtype=np.float64)
    ok = np.isfinite(c) & (c > 0) & np.isfinite(i) & (i > 0)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    x, z = np.log(i[ok]), np.log(c[ok])
    xc = x - x.mean()
    beta = float((xc * (z - z.mean())).sum() / (xc * xc).sum())
    return beta, float(z.mean() - beta * x.mean())


def second_difference_share(curve: ArrayLike) -> float:
    """Share of negative second differences ``c[i+1] - 2 c[i] + c[i-1]`` among the finite ones (NaN if none)."""
    c = np.asarray(curve, dtype=np.float64)
    d2 = c[2:] - 2.0 * c[1:-1] + c[:-2]
    d2 = d2[np.isfinite(d2)]
    return float(np.mean(d2 < 0.0)) if d2.size else float("nan")


def growth_curve(
    curves: ArrayLike, n_boot: int = 1000, seed: int = 0, ci: float = 0.95, min_profiles: int = 1,
    fit_min_profiles: int | None = None,
) -> dict[str, Any]:
    """Mean growth curve of per-profile curves (P, V) (NaN = index not available) with bootstrap over profiles.

    Returns ``curve`` (V,) (NaN where fewer than ``min_profiles`` profiles have the index), ``lo`` / ``hi`` (V,),
    ``n_profiles`` (V,), ``beta`` / ``beta_lo`` / ``beta_hi``, ``log_sd1`` and ``share_neg_d2`` of the mean curve,
    ``fitted`` (V,) bool: the indices that entered the exponent fit. The exponent is fitted over the indices held by
    at least ``fit_min_profiles`` profiles (default ``min_profiles``): a ragged set of measured platoons would
    otherwise mix the level of the few long platoons into the slope (composition effect).
    """
    c = np.asarray(curves, dtype=np.float64)
    n_p, n_v = c.shape
    ok = np.isfinite(c)
    count = ok.sum(axis=0)
    usable = count >= max(1, int(min_profiles))
    fitted = usable & (count >= max(1, int(min_profiles if fit_min_profiles is None else fit_min_profiles)))
    mean = np.where(usable, np.where(ok, c, 0.0).sum(axis=0) / np.maximum(count, 1), np.nan)
    beta, log_sd1 = growth_exponent(np.where(fitted, mean, np.nan))
    alpha = 1.0 - ci
    out: dict[str, Any] = {"curve": mean, "lo": np.full(n_v, np.nan), "hi": np.full(n_v, np.nan), "n_profiles": count,
                           "beta": beta, "beta_lo": np.nan, "beta_hi": np.nan, "log_sd1": log_sd1,
                           "share_neg_d2": second_difference_share(mean), "fitted": fitted}  # fmt: skip
    if n_p < 2:
        return out
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, n_p, n_p), minlength=n_p) for _ in range(int(n_boot))]).astype(np.float64)
    num = w @ np.where(ok, c, 0.0)  # (B, V)
    den = w @ ok.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = np.where(den > 0, num / den, np.nan)
    reps[:, ~usable] = np.nan
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    cols = np.flatnonzero(usable & np.isfinite(reps).any(axis=0))
    if cols.size:
        out["lo"][cols], out["hi"][cols] = np.nanpercentile(reps[:, cols], [lo_q, hi_q], axis=0)
    betas = np.array([growth_exponent(np.where(fitted, r, np.nan))[0] for r in reps])
    betas = betas[np.isfinite(betas)]
    if betas.size:
        out["beta_lo"], out["beta_hi"] = (float(x) for x in np.percentile(betas, [lo_q, hi_q]))
    return out
