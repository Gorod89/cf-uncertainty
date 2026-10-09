"""Driver (group) bootstrap, paired comparisons and Holm correction (docs/design.md, section 6).

Windows of one driver are strongly dependent, so every confidence interval here
resamples **groups** (drivers) with replacement and keeps all rows of a drawn
group (repeated as often as the group is drawn). Mean-type statistics are
vectorised through per-group sums and counts and a (replicates x groups) matrix
of draw multiplicities, processed in blocks of at most ~4M entries.

All CIs are percentile intervals; ``seed`` may be an int or a ``numpy`` Generator.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Mapping

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "bootstrap_by_group",
    "bootstrap_ratio_by_group",
    "bootstrap_two_sample_ratio",
    "bootstrap_generic",
    "paired_bootstrap_test",
    "holm",
]

SeedLike = int | np.random.Generator | np.random.SeedSequence | None

_COUNT_BLOCK = 1 << 22  # entries of one (replicates x groups) block of draw multiplicities


# --------------------------------------------------------------------------- helpers
def _f64(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def _check(n_boot: int, ci: float) -> None:
    if int(n_boot) < 2:
        raise ValueError("n_boot must be at least 2")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must lie in (0, 1)")


def _codes(groups: ArrayLike, n: int) -> tuple[np.ndarray, int]:
    """Integer group codes (n,) in 0..G-1 and the number of groups G."""
    g = np.asarray(groups).ravel()
    if g.shape[0] != n:
        raise ValueError(f"groups must have {n} entries, got {g.shape[0]}")
    if n == 0:
        return np.zeros(0, dtype=np.intp), 0
    _, codes = np.unique(g, return_inverse=True)
    codes = codes.ravel().astype(np.intp)
    return codes, int(codes.max()) + 1


def _count_blocks(rng: np.random.Generator, n_groups: int, n_boot: int) -> Iterator[np.ndarray]:
    """Yield (b, G) blocks: how often each of G groups is drawn in each bootstrap replicate.

    Each replicate draws G groups uniformly with replacement (rows sum to G).
    """
    step = max(1, _COUNT_BLOCK // max(1, n_groups))
    for start in range(0, n_boot, step):
        b = min(step, n_boot - start)
        draws = rng.integers(0, n_groups, size=(b, n_groups))
        draws += (np.arange(b) * n_groups)[:, None]
        yield np.bincount(draws.ravel(), minlength=b * n_groups).reshape(b, n_groups)


def _group_sums(values: np.ndarray, codes: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-group sums and counts of the non-NaN values."""
    ok = ~np.isnan(values)
    sums = np.bincount(codes[ok], weights=values[ok], minlength=n_groups)
    counts = np.bincount(codes[ok], minlength=n_groups).astype(np.float64)
    return sums, counts


def _replicate_means(
    rng: np.random.Generator, sums: list[np.ndarray], counts: list[np.ndarray], n_boot: int
) -> np.ndarray:
    """(len(sums), n_boot) bootstrap means ``sum_g c_g S_g / sum_g c_g N_g``, same draws for all."""
    n_groups = sums[0].shape[0]
    out = np.empty((len(sums), n_boot))
    if n_groups == 0:
        out.fill(np.nan)
        return out
    start = 0
    for block in _count_blocks(rng, n_groups, n_boot):
        c = block.astype(np.float64)
        b = c.shape[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            for i, (s, m) in enumerate(zip(sums, counts)):
                out[i, start : start + b] = (c @ s) / (c @ m)
        start += b
    return out


def _summary(replicates: np.ndarray, ci: float) -> tuple[Any, Any, Any]:
    """Percentile CI bounds and bootstrap standard error along axis 0.

    Undefined replicates (NaN or +-inf, e.g. from a zero resampled denominator) are ignored;
    bounds / se are NaN where fewer than 1 / 2 defined replicates remain.
    """
    alpha = 1.0 - ci
    shape = replicates.shape[1:]
    flat = replicates.reshape(replicates.shape[0], -1)
    flat = np.where(np.isfinite(flat), flat, np.nan)
    n_ok = np.count_nonzero(~np.isnan(flat), axis=0)
    lo, hi, se = (np.full(flat.shape[1], np.nan) for _ in range(3))
    ok = n_ok >= 1
    if np.any(ok):
        q = np.nanquantile(flat[:, ok], [0.5 * alpha, 1.0 - 0.5 * alpha], axis=0)
        lo[ok], hi[ok] = q[0], q[1]
    ok2 = n_ok >= 2
    if np.any(ok2):
        se[ok2] = np.nanstd(flat[:, ok2], axis=0, ddof=1)
    return _scalar(lo.reshape(shape)), _scalar(hi.reshape(shape)), _scalar(se.reshape(shape))


def _scalar(x: Any) -> Any:
    arr = np.asarray(x, dtype=np.float64)
    return float(arr) if arr.ndim == 0 else arr


def _nan_mean(values: np.ndarray) -> float:
    ok = ~np.isnan(values)
    return float(values[ok].mean()) if ok.any() else float("nan")


def _is_mean(stat: Callable[..., Any] | None) -> bool:
    return stat is None or stat is np.mean or stat is np.nanmean


# ------------------------------------------------------------------------ bootstraps
def bootstrap_by_group(
    values: ArrayLike,
    groups: ArrayLike,
    n_boot: int = 1000,
    seed: SeedLike = 0,
    ci: float = 0.95,
    *,
    stat: Callable[[np.ndarray], Any] | None = None,
) -> dict[str, Any]:
    """Group bootstrap of the mean of ``values`` (n,) with group labels ``groups`` (n,).

    Groups are resampled with replacement; NaN values are dropped (per group), groups without
    any valid value are removed. Returns ``estimate`` (mean of all valid values), percentile
    ``lo`` / ``hi``, ``se`` (std of the replicates), ``replicates`` (n_boot,), ``n_groups``, ``n``.
    ``stat``: optional statistic of the pooled resampled values (default: mean, vectorised);
    anything other than ``np.mean`` / ``np.nanmean`` goes through :func:`bootstrap_generic`.
    """
    _check(n_boot, ci)
    v = _f64(values)
    g = np.asarray(groups).ravel()
    if g.shape[0] != v.shape[0]:
        raise ValueError("values and groups must have the same length")
    ok = ~np.isnan(v)
    v, g = v[ok], g[ok]
    if not _is_mean(stat):
        res = bootstrap_generic({"values": v}, g, lambda d: stat(d["values"]), n_boot=n_boot, seed=seed, ci=ci)
        res["n"] = int(v.shape[0])
        return res
    codes, n_groups = _codes(g, v.shape[0])
    sums, counts = _group_sums(v, codes, n_groups)
    reps = _replicate_means(np.random.default_rng(seed), [sums], [counts], int(n_boot))[0]
    lo, hi, se = _summary(reps, ci)
    return {
        "estimate": _nan_mean(v),
        "lo": lo,
        "hi": hi,
        "se": se,
        "replicates": reps,
        "n_groups": n_groups,
        "n": int(v.shape[0]),
    }


def bootstrap_ratio_by_group(
    num: ArrayLike,
    den: ArrayLike,
    groups: ArrayLike,
    n_boot: int = 1000,
    seed: SeedLike = 0,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Group bootstrap of ``mean(num) / mean(den)`` over the same resampled groups.

    num, den, groups: (n,). NaNs are dropped separately in ``num`` and ``den`` (each mean uses
    its own valid rows); groups with no valid row in either are removed. Returns ``estimate``,
    ``lo``, ``hi``, ``se``, ``replicates``, ``n_groups``, ``num_mean``, ``den_mean``.
    """
    _check(n_boot, ci)
    x, z = _f64(num), _f64(den)
    if x.shape != z.shape:
        raise ValueError("num and den must have the same length")
    g = np.asarray(groups).ravel()
    keep = ~(np.isnan(x) & np.isnan(z))
    x, z, g = x[keep], z[keep], g[keep]
    codes, n_groups = _codes(g, x.shape[0])
    s_num, c_num = _group_sums(x, codes, n_groups)
    s_den, c_den = _group_sums(z, codes, n_groups)
    reps_num, reps_den = _replicate_means(np.random.default_rng(seed), [s_num, s_den], [c_num, c_den], int(n_boot))
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = reps_num / reps_den
        num_mean, den_mean = _nan_mean(x), _nan_mean(z)
        estimate = num_mean / den_mean if den_mean != 0 else float("nan")
    lo, hi, se = _summary(reps, ci)
    return {
        "estimate": float(estimate),
        "lo": lo,
        "hi": hi,
        "se": se,
        "replicates": reps,
        "n_groups": n_groups,
        "num_mean": num_mean,
        "den_mean": den_mean,
    }


def bootstrap_two_sample_ratio(
    num_values: ArrayLike,
    num_groups: ArrayLike,
    den_values: ArrayLike,
    den_groups: ArrayLike,
    n_boot: int = 1000,
    seed: SeedLike = 0,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Ratio of the means of two independent data sets, groups resampled independently in each.

    E.g. mean epistemic variance on a shifted data set over the in-domain one. NaNs are
    dropped. Returns ``estimate``, percentile ``lo`` / ``hi``, ``se``, ``replicates``,
    ``num_mean``, ``den_mean``, ``n_groups_num``, ``n_groups_den``.
    """
    _check(n_boot, ci)
    rng = np.random.default_rng(seed)
    reps, means, n_groups = [], [], []
    for values, groups in ((num_values, num_groups), (den_values, den_groups)):
        v = _f64(values)
        g = np.asarray(groups).ravel()
        if g.shape[0] != v.shape[0]:
            raise ValueError("values and groups must have the same length")
        ok = ~np.isnan(v)
        v, g = v[ok], g[ok]
        codes, ng = _codes(g, v.shape[0])
        s, c = _group_sums(v, codes, ng)
        reps.append(_replicate_means(rng, [s], [c], int(n_boot))[0])
        means.append(_nan_mean(v))
        n_groups.append(ng)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = reps[0] / reps[1]
        estimate = means[0] / means[1] if means[1] != 0 else float("nan")
    lo, hi, se = _summary(ratio, ci)
    return {
        "estimate": float(estimate),
        "lo": lo,
        "hi": hi,
        "se": se,
        "replicates": ratio,
        "num_mean": means[0],
        "den_mean": means[1],
        "n_groups_num": n_groups[0],
        "n_groups_den": n_groups[1],
    }


def bootstrap_generic(
    arrays: Mapping[str, ArrayLike],
    groups: ArrayLike,
    stat_fn: Callable[[dict[str, np.ndarray]], Any],
    n_boot: int = 1000,
    seed: SeedLike = 0,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Group bootstrap of an arbitrary statistic (KS statistic, quantiles, ...).

    arrays: name -> array whose first axis has the n rows of ``groups`` (n,). ``stat_fn`` gets a
    dict with the rows of the drawn groups (a group drawn twice appears twice; row order is not
    meaningful) and returns a scalar or a fixed-shape array. No NaN handling: ``stat_fn`` sees the
    raw rows. Returns ``estimate`` = ``stat_fn(arrays)``, percentile ``lo`` / ``hi``, ``se``,
    ``replicates`` (n_boot, ...) and ``n_groups``.
    """
    _check(n_boot, ci)
    data = {k: np.asarray(v) for k, v in arrays.items()}
    g = np.asarray(groups).ravel()
    n = g.shape[0]
    for k, v in data.items():
        if v.shape[0] != n:
            raise ValueError(f"array {k!r} has {v.shape[0]} rows, groups has {n}")
    codes, n_groups = _codes(g, n)
    estimate = np.asarray(stat_fn(data), dtype=np.float64)
    reps = np.empty((int(n_boot),) + estimate.shape)
    rows = np.arange(n)
    i = 0
    for block in _count_blocks(np.random.default_rng(seed), max(n_groups, 1), int(n_boot)):
        for counts in block:
            idx = np.repeat(rows, counts[codes]) if n_groups else rows
            reps[i] = stat_fn({k: v[idx] for k, v in data.items()})
            i += 1
    lo, hi, se = _summary(reps, ci)
    return {
        "estimate": _scalar(estimate),
        "lo": lo,
        "hi": hi,
        "se": se,
        "replicates": reps,
        "n_groups": n_groups,
    }


def paired_bootstrap_test(
    values_a: ArrayLike,
    values_b: ArrayLike,
    groups: ArrayLike,
    n_boot: int = 1000,
    seed: SeedLike = 0,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Paired comparison of per-sample scores a and b (n,) on the same rows, by group bootstrap.

    ``diff = mean(a - b)`` (positive when a is larger), CI of the group-bootstrapped mean
    difference, two-sided percentile p-value with +1 smoothing
    ``min(1, 2 min((#{d* <= 0} + 1) / (B + 1), (#{d* >= 0} + 1) / (B + 1)))`` and
    ``rel_diff = diff / mean(b)``. Rows where a or b is NaN are dropped.
    Returns ``diff``, ``lo``, ``hi``, ``pvalue``, ``rel_diff``, ``se``, ``replicates``,
    ``n_groups``, ``mean_a``, ``mean_b``.
    """
    _check(n_boot, ci)
    a, b = _f64(values_a), _f64(values_b)
    g = np.asarray(groups).ravel()
    if not (a.shape == b.shape == g.shape):
        raise ValueError("values_a, values_b and groups must have the same length")
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b, g = a[ok], b[ok], g[ok]
    d = a - b
    codes, n_groups = _codes(g, d.shape[0])
    sums, counts = _group_sums(d, codes, n_groups)
    reps = _replicate_means(np.random.default_rng(seed), [sums], [counts], int(n_boot))[0]
    lo, hi, se = _summary(reps, ci)
    n_b = reps.shape[0]
    p_low = (np.count_nonzero(reps <= 0) + 1) / (n_b + 1)
    p_high = (np.count_nonzero(reps >= 0) + 1) / (n_b + 1)
    pvalue = float(min(1.0, 2.0 * min(p_low, p_high))) if n_groups else float("nan")
    diff = _nan_mean(d)
    mean_b = _nan_mean(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = diff / mean_b if mean_b != 0 else float("nan")
    return {
        "diff": diff,
        "lo": lo,
        "hi": hi,
        "pvalue": pvalue,
        "rel_diff": float(rel),
        "se": se,
        "replicates": reps,
        "n_groups": n_groups,
        "mean_a": _nan_mean(a),
        "mean_b": mean_b,
    }


def holm(pvalues: ArrayLike, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down correction: ``(adjusted p-values, reject flags)`` in the input order.

    Adjusted p-values are ``max_{j <= i} min(1, (m - j + 1) p_(j))`` (monotone in the raw
    p-values, capped at 1); a hypothesis is rejected when its adjusted p-value is <= alpha.
    NaN p-values are left out (adjusted NaN, not rejected, not counted in m).
    """
    p = _f64(pvalues)
    adjusted = np.full(p.shape, np.nan)
    reject = np.zeros(p.shape, dtype=bool)
    valid = np.flatnonzero(~np.isnan(p))
    m = valid.shape[0]
    if m == 0:
        return adjusted, reject
    order = valid[np.argsort(p[valid], kind="stable")]
    stepped = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    adjusted[order] = stepped
    reject[order] = stepped <= alpha
    return adjusted, reject
