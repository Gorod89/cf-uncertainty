"""Dataset shift, concentration in rare states and the H3.1 verdicts (docs/design.md, sections 10.4 and 12: H3.1 as
written in the plan, and H3.1e, its epistemic part restated at the M3 review).

Event bootstrap on sufficient statistics
----------------------------------------
Every statistic of this module is a ratio of means of per-window values ``x``. Windows are grouped
into resampling units (events by default, drivers as a sensitivity analysis). The sums and counts
of the non-NaN values are formed once per unit (and per seed, bin, regime or mask group where
needed); a bootstrap replicate draws the units with replacement, i.e. a vector of multiplicities
``c`` with ``sum(c) = n_units``, and every resampled mean is ``(c @ S) / (c @ N)``. 1000 replicates
cost a few ``(block x units) @ (units x columns)`` products (blocks of at most 2^22 multiplicities),
independent of the number of windows.

* Units without any non-NaN value (e.g. events of which DiffusionCF evaluated no window) are left
  out before resampling, so that the resampled population is the one the estimate is formed from.
* ``x`` arrays are ``(n_seeds, n)``: one row per training seed, the same windows in every row (a
  1-D array is one seed). A replicate uses the SAME resampled units for every seed (common random
  numbers) and forms the statistic from the seed-averaged means, exactly like the estimate. Seeds
  whose full-data mean is undefined (no valid value) are left out of the average. When all seeds
  miss the same windows (the usual case), the replicates are computed from the seed-averaged values,
  which gives exactly the same numbers at 1 / n_seeds of the cost (:func:`_collapse`).
* An integer seed makes the draws reproducible: the same seed and the same units give the same
  multiplicities in every call, so separate calls share their draws (common random numbers).
* Intervals are percentile intervals over the finite replicates, ``se`` is their standard deviation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import sparse

from cf_unc.maps.io import PRED_COLUMNS

if TYPE_CHECKING:
    from cf_unc.maps.grid import StateGrid

__all__ = [
    "QUANTITIES",
    "KINDS",
    "VERDICTS",
    "SeedStack",
    "own_decomposition",
    "seed_decomposition",
    "quantity_arrays",
    "has_epistemic",
    "encode_units",
    "event_sums",
    "multiplicities",
    "resample_sums",
    "bootstrap_pvalue",
    "bootstrap_seed_means",
    "shift_ratio",
    "shift_ratios",
    "matched_shift_ratio",
    "matched_shift_ratios",
    "concentration",
    "concentrations",
    "H31Thresholds",
    "H31_SUMMARY_COLUMNS",
    "h31_verdict",
    "h31_summary",
    "H31eThresholds",
    "H31E_VERDICT_COLUMNS",
    "H31E_SUMMARY_COLUMNS",
    "h31e_verdict",
    "h31e_summary",
]

SeedLike = int | np.random.SeedSequence | np.random.Generator | None

QUANTITIES = ("epistemic", "aleatoric", "total", "nll", "mse")  # per-window quantities of the maps
KINDS = ("model", "seed_ensemble")  # own decomposition of the model | across-seed decomposition
VERDICTS = ("supported", "refuted", "inconclusive")

_BLOCK = 1 << 22  # multiplicities per block of replicates (32 MiB as int64)


# ----------------------------------------------------------------------------------- helpers
def _as_2d(x: ArrayLike) -> np.ndarray:
    """(n,) -> (1, n); (n_seeds, n) as is; float64."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"expected (n,) or (n_seeds, n), got shape {arr.shape}")
    return arr


def _any_valid(arrays: Iterable[np.ndarray]) -> np.ndarray:
    """(n,) rows with at least one non-NaN value in any seed of any array."""
    out: np.ndarray | None = None
    for arr in arrays:
        ok = (~np.isnan(arr)).any(axis=0)
        out = ok if out is None else (out | ok)
    if out is None:
        raise ValueError("no arrays given")
    return out


def _row_nanmean(a: np.ndarray) -> np.ndarray:
    """Mean over the last axis of the finite entries (NaN where none), without warnings."""
    arr = np.asarray(a, dtype=np.float64)
    fin = np.isfinite(arr)
    cnt = fin.sum(axis=-1)
    tot = np.where(fin, arr, 0.0).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = tot / cnt
    return np.where(cnt > 0, out, np.nan)


def _sd(values: np.ndarray) -> float:
    """Sample standard deviation (ddof = 1) of the finite values; NaN with fewer than two."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.std(ddof=1)) if v.size >= 2 else float("nan")


def _interval(replicates: np.ndarray, ci: float) -> tuple[float, float, float]:
    """Percentile interval and standard deviation of the finite replicates (NaN if none)."""
    r = np.asarray(replicates, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan"), float("nan"), float("nan")
    alpha = 1.0 - ci
    lo, hi = np.quantile(r, [0.5 * alpha, 1.0 - 0.5 * alpha])
    se = float(r.std(ddof=1)) if r.size >= 2 else float("nan")
    return float(lo), float(hi), se


def _check(n_boot: int, ci: float) -> None:
    if int(n_boot) < 2:
        raise ValueError("n_boot must be at least 2")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must lie in (0, 1)")


def _spawn(seed: SeedLike, n: int) -> list[np.random.SeedSequence]:
    """``n`` independent child seed sequences, deterministic for an integer ``seed``."""
    if isinstance(seed, np.random.Generator):
        return [np.random.SeedSequence(int(v)) for v in seed.integers(0, 2**63 - 1, size=n)]
    parent = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    return list(parent.spawn(n))


def _ratio(num: float, den: float) -> float:
    if not (np.isfinite(num) and np.isfinite(den)) or den == 0:
        return float("nan")
    return float(num / den)


# ------------------------------------------------------------------------ prediction stacks
@dataclass
class SeedStack:
    """Prediction columns of several seeds on the same n windows.

    ``columns[name]`` is ``(n_seeds, n)`` for every name of ``cf_unc.maps.io.PRED_COLUMNS`` (mean, var,
    aleatoric_var, epistemic_var, nll, crps, covered, width, pit; m/s^2, (m/s^2)^2, nats).
    ``covered`` is NaN on rows without a prediction (the tables store 0 there).
    """

    seeds: tuple[int, ...]
    columns: dict[str, np.ndarray]

    @classmethod
    def from_frames(
        cls, frames: Sequence[pd.DataFrame], seeds: Sequence[int] | None = None, dtype: Any = np.float32
    ) -> "SeedStack":
        """Stack prediction tables (one per seed, same row order); missing columns become NaN."""
        frames = list(frames)
        if not frames:
            raise ValueError("need at least one prediction table")
        n = len(frames[0])
        if any(len(f) != n for f in frames):
            raise ValueError("prediction tables of different lengths")
        cols: dict[str, np.ndarray] = {}
        for name in PRED_COLUMNS:
            arr = np.full((len(frames), n), np.nan, dtype=dtype)
            for k, frame in enumerate(frames):
                if name in frame.columns:
                    arr[k] = frame[name].to_numpy(dtype=dtype, na_value=np.nan)
            cols[name] = arr
        valid = np.isfinite(cols["mean"]) & np.isfinite(cols["var"])
        cols["covered"] = np.where(valid, cols["covered"], np.nan).astype(dtype)
        chosen = tuple(int(s) for s in (range(len(frames)) if seeds is None else seeds))
        if len(chosen) != len(frames):
            raise ValueError("one seed per prediction table")
        return cls(chosen, cols)

    @classmethod
    def coerce(cls, preds: "SeedStack | Sequence[pd.DataFrame]") -> "SeedStack":
        return preds if isinstance(preds, SeedStack) else cls.from_frames(list(preds))

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)

    def __len__(self) -> int:
        return int(self.columns["mean"].shape[1])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]

    @property
    def valid(self) -> np.ndarray:
        """(n_seeds, n) rows with a prediction (finite mean and variance)."""
        return np.isfinite(self.columns["mean"]) & np.isfinite(self.columns["var"])

    def take(self, rows: np.ndarray | slice | None) -> "SeedStack":
        """Windows ``rows`` (index or boolean mask) of every seed; ``None`` returns ``self``."""
        if rows is None:
            return self
        return SeedStack(self.seeds, {k: v[:, rows] for k, v in self.columns.items()})

    def select(self, positions: Sequence[int]) -> "SeedStack":
        """Seeds at ``positions`` (indices into ``self.seeds``)."""
        pos = list(positions)
        return SeedStack(tuple(self.seeds[p] for p in pos), {k: v[pos] for k, v in self.columns.items()})


def own_decomposition(preds: "SeedStack | Sequence[pd.DataFrame]") -> dict[str, np.ndarray]:
    """The model's own decomposition, ``aleatoric``, ``epistemic``, ``total``: (n_seeds, n) float64.

    ``total`` = predictive variance; ``epistemic`` = the model's epistemic output (NaN when it has none);
    ``aleatoric`` = the model's aleatoric output, or, for a model that reports none (DiffusionCF,
    QuantilePIDL), its whole variance minus a finite epistemic part (design.md, section 7).
    """
    stack = SeedStack.coerce(preds)
    total = stack["var"].astype(np.float64)
    epistemic = stack["epistemic_var"].astype(np.float64)
    aleatoric = stack["aleatoric_var"].astype(np.float64)
    if not np.isfinite(aleatoric).any():
        aleatoric = np.where(np.isfinite(epistemic), total - epistemic, total)
    return {"aleatoric": aleatoric, "epistemic": epistemic, "total": total}


def seed_decomposition(means: ArrayLike, variances: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Across-seed decomposition ``(aleatoric_seed, epistemic_seed)``, each (n,) float64.

    means, variances: (n_seeds, n) predictive means and variances of the seeds. Per window over the seeds
    with a finite mean and variance: ``aleatoric_seed = mean_k var_k``, ``epistemic_seed = var_k mean_k``
    (population variance, ddof = 0), so their sum is the variance of the equal-weight mixture of the
    seeds' laws. NaN where fewer than two seeds are valid (a seed decomposition needs two seeds).
    """
    m, v = _as_2d(means), _as_2d(variances)
    if m.shape != v.shape:
        raise ValueError(f"means and variances must have the same shape, got {m.shape}, {v.shape}")
    ok = np.isfinite(m) & np.isfinite(v)
    k = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        aleatoric = np.where(ok, v, 0.0).sum(axis=0) / k
        center = np.where(ok, m, 0.0).sum(axis=0) / k
        epistemic = np.where(ok, (m - center) ** 2, 0.0).sum(axis=0) / k
    few = k < 2
    aleatoric[few] = np.nan
    epistemic[few] = np.nan
    return aleatoric, epistemic


def quantity_arrays(
    preds: "SeedStack | Sequence[pd.DataFrame]", y: ArrayLike, kind: str = "model"
) -> dict[str, np.ndarray]:
    """Per-window quantities of ``QUANTITIES`` for one kind of decomposition, each (n_rows, n) float64.

    ``kind="model"``: the model's own decomposition (:func:`own_decomposition`), one row per seed.
    ``kind="seed_ensemble"``: ``epistemic`` = across-seed variance of the means and ``total`` =
    aleatoric_seed + epistemic_seed (one row, :func:`seed_decomposition`); ``aleatoric`` = the seeds'
    predictive variances (one row per seed, restricted to windows with at least two valid seeds, so that
    their seed average is aleatoric_seed). ``nll`` and ``mse = (y - mean)^2`` are the per-seed scores in
    both kinds (the seed ensemble is not scored as a law of its own).
    """
    stack = SeedStack.coerce(preds)
    y64 = np.asarray(y, dtype=np.float64).ravel()
    if y64.shape[0] != len(stack):
        raise ValueError("y and the predictions must have the same number of rows")
    mean = stack["mean"].astype(np.float64)
    out: dict[str, np.ndarray]
    if kind == "model":
        out = own_decomposition(stack)
    elif kind == "seed_ensemble":
        var = stack["var"].astype(np.float64)
        aleatoric_seed, epistemic_seed = seed_decomposition(mean, var)
        two = np.isfinite(aleatoric_seed)
        out = {
            "aleatoric": np.where(two[None, :], var, np.nan),
            "epistemic": epistemic_seed[None, :],
            "total": (aleatoric_seed + epistemic_seed)[None, :],
        }
    else:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    out["nll"] = stack["nll"].astype(np.float64)
    out["mse"] = (y64[None, :] - mean) ** 2
    return {q: out[q] for q in QUANTITIES}


def has_epistemic(epistemic: ArrayLike) -> bool:
    """Whether a model reports a usable epistemic part: some finite, positive value.

    False for models without an epistemic output (NaN) and for the constant-noise baselines (0).
    """
    e = np.asarray(epistemic, dtype=np.float64)
    return bool(np.any(np.isfinite(e) & (e > 0)))


# --------------------------------------------------------------------------- unit statistics
def encode_units(labels: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """``(codes (n,) intp in 0..U-1, units (U,))``: dense codes of the unit labels in sorted order.

    Non-negative integer labels with a moderate range are encoded in O(n) (no sort).
    """
    lab = np.asarray(labels).ravel()
    if lab.size == 0:
        return np.zeros(0, dtype=np.intp), lab[:0]
    if lab.dtype.kind in "iu":
        lo, hi = int(lab.min()), int(lab.max())
        if hi - lo < 4 * lab.size + 1024:
            shifted = (lab.astype(np.int64) - lo).astype(np.intp)
            present = np.bincount(shifted, minlength=hi - lo + 1) > 0
            remap = np.cumsum(present) - 1
            return remap[shifted].astype(np.intp), (np.flatnonzero(present) + lo).astype(lab.dtype)
    units, codes = np.unique(lab, return_inverse=True)
    return codes.ravel().astype(np.intp), units


def _unit_group_sums(
    x: np.ndarray, codes: np.ndarray, n_units: int, groups: np.ndarray | None = None, n_groups: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit sums and counts of the non-NaN values of x (n_seeds, n), per seed and group.

    Returns two (n_units, n_seeds * n_groups) float64 arrays; column ``k * n_groups + g``.
    """
    n_seeds = x.shape[0]
    width = n_seeds * n_groups
    size = n_units * width
    base = codes.astype(np.int64) * width
    if groups is not None:
        base = base + np.asarray(groups, dtype=np.int64)
    sums = np.zeros(size)
    counts = np.zeros(size)
    for k in range(n_seeds):
        ok = ~np.isnan(x[k])
        idx = base[ok] + k * n_groups
        sums += np.bincount(idx, weights=x[k][ok], minlength=size)
        counts += np.bincount(idx, minlength=size)
    return sums.reshape(n_units, width), counts.reshape(n_units, width)


def event_sums(values: ArrayLike, event_codes: ArrayLike, n_events: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-event sums and counts of the non-NaN values.

    values: (n,) or (n_seeds, n); event_codes: (n,) integers in 0..n_events-1.
    Returns ``(sums, counts)`` float64 of shape (n_events,) or (n_seeds, n_events).
    """
    arr = np.asarray(values, dtype=np.float64)
    x = _as_2d(arr)
    codes = np.asarray(event_codes, dtype=np.int64).ravel()
    if codes.shape[0] != x.shape[1]:
        raise ValueError("values and event_codes must have the same number of windows")
    if codes.size and (codes.min() < 0 or codes.max() >= n_events):
        raise ValueError("event codes must lie in 0..n_events-1")
    sums, counts = _unit_group_sums(x, codes, int(n_events))
    if arr.ndim == 1:
        return sums[:, 0], counts[:, 0]
    return sums.T.copy(), counts.T.copy()


def multiplicities(
    n_units: int, n_boot: int, seed: SeedLike = 0, block: int = _BLOCK
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(start, counts)``: counts (b, n_units) int64, how often every unit is drawn in replicates
    ``start .. start + b - 1``; each replicate draws ``n_units`` units uniformly with replacement."""
    rng = np.random.default_rng(seed)
    step = max(1, int(block) // max(1, n_units))
    for start in range(0, int(n_boot), step):
        b = min(step, int(n_boot) - start)
        draws = rng.integers(0, n_units, size=(b, n_units))
        draws += (np.arange(b, dtype=np.int64) * n_units)[:, None]
        yield start, np.bincount(draws.ravel(), minlength=b * n_units).reshape(b, n_units)


def resample_sums(unit_stats: Any, n_boot: int = 1000, seed: SeedLike = 0, block: int = _BLOCK) -> np.ndarray:
    """Bootstrap replicates of column sums over units: ``out[b] = c_b @ unit_stats``, (n_boot, m).

    unit_stats: (n_units, m) dense array or scipy sparse matrix of per-unit sums / counts; ``c_b`` are
    the multiplicities of :func:`multiplicities` (same ``seed`` and ``n_units`` -> same draws).
    """
    is_sparse = sparse.issparse(unit_stats)
    if is_sparse:
        mat_t = sparse.csr_array(unit_stats.T)
        n_units, m = unit_stats.shape
    else:
        mat = np.asarray(unit_stats, dtype=np.float64)
        if mat.ndim == 1:
            mat = mat[:, None]
        n_units, m = mat.shape
    out = np.full((int(n_boot), m), np.nan)
    if n_units == 0:
        return out
    for start, counts in multiplicities(n_units, n_boot, seed, block):
        c = counts.astype(np.float64)
        stop = start + c.shape[0]
        out[start:stop] = np.asarray(mat_t @ c.T).T if is_sparse else c @ mat
    return out


def bootstrap_pvalue(replicates: ArrayLike, null: float = 0.0) -> float:
    """Two-sided percentile bootstrap p-value of ``statistic = null`` with +1 smoothing.

    ``min(1, 2 min((#{r <= null} + 1) / (B + 1), (#{r >= null} + 1) / (B + 1)))`` over the B non-NaN
    replicates (the convention of ``cf_unc.eval.bootstrap.paired_bootstrap_test``). It is below alpha
    exactly when the percentile interval of level 1 - alpha excludes ``null`` (up to the smoothing).
    """
    r = np.asarray(replicates, dtype=np.float64).ravel()
    r = r[~np.isnan(r)]
    if r.size == 0:
        return float("nan")
    b = r.size
    low = (np.count_nonzero(r <= null) + 1) / (b + 1)
    high = (np.count_nonzero(r >= null) + 1) / (b + 1)
    return float(min(1.0, 2.0 * min(low, high)))


# ------------------------------------------------------------------------ seed-averaged means
def _collapse(x: np.ndarray) -> np.ndarray:
    """(K, n) -> (1, n) seed means when every seed misses the same windows, else ``x`` unchanged.

    Replicates only need seed averages of resampled means ``mean_k S_k / N_k``. With a common set of valid
    windows every ``N_k`` equals N, so ``mean_k (S_k / N) = (mean_k S_k) / N``: exactly the resampled mean of
    the seed-averaged values. This divides the bootstrap work by the number of seeds.
    """
    if x.shape[0] < 2:
        return x
    missing = np.isnan(x)
    if not (missing == missing[0]).all():
        return x
    return x.mean(axis=0, keepdims=True)


def _seed_group_means(x: np.ndarray, groups: np.ndarray | None = None, n_groups: int = 1) -> np.ndarray:
    """Full-data means of the non-NaN values per seed (row of x) and group, (K, n_groups); NaN when empty."""
    k = x.shape[0]
    g = None if groups is None else np.asarray(groups, dtype=np.int64)
    sums = np.zeros((k, n_groups))
    counts = np.zeros((k, n_groups))
    for i in range(k):
        ok = ~np.isnan(x[i])
        if g is None:
            sums[i, 0], counts[i, 0] = x[i][ok].sum(), ok.sum()
        else:
            sums[i] = np.bincount(g[ok], weights=x[i][ok], minlength=n_groups)
            counts[i] = np.bincount(g[ok], minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / counts


def _replicate_means(
    arrays: Sequence[np.ndarray],
    events: np.ndarray,
    n_boot: int,
    seed: SeedLike,
    groups: np.ndarray | None = None,
    n_groups: int = 1,
    keep: np.ndarray | None = None,
) -> tuple[list[np.ndarray], int]:
    """Replicate means per seed and group, (n_boot, K_eff, n_groups), of several (K, n) arrays, same draws.

    The units are those with a non-NaN value in any array (or the rows ``keep``). An array whose seeds all
    miss the same windows is collapsed to its seed mean (:func:`_collapse`, K_eff = 1), else K_eff = K.
    """
    if keep is None:
        keep = _any_valid(arrays)
    codes, units = encode_units(np.asarray(events).ravel()[keep])
    n_units = len(units)
    g = None if groups is None else np.asarray(groups, dtype=np.int64)[keep]
    blocks: list[np.ndarray] = []
    widths: list[int] = []
    for x in arrays:
        xe = _collapse(x)
        s, c = _unit_group_sums(xe[:, keep], codes, n_units, g, n_groups)
        blocks += [s, c]
        widths.append(xe.shape[0])
    reps = resample_sums(np.hstack(blocks), n_boot, seed) if blocks else np.empty((int(n_boot), 0))
    out: list[np.ndarray] = []
    col = 0
    for k in widths:
        w = k * n_groups
        if k == 0:
            out.append(np.full((int(n_boot), 0, n_groups), np.nan))
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append((reps[:, col : col + w] / reps[:, col + w : col + 2 * w]).reshape(-1, k, n_groups))
        col += 2 * w
    return out, n_units


def _avg_seeds(rep: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Seed average (n_boot,) of replicate means ``rep`` (n_boot, K_eff): over the seeds ``ok`` (K,), or the
    collapsed row (K_eff = 1 < K), which already is the average over the seeds."""
    if rep.shape[1] == 1 and ok.shape[0] > 1:
        return rep[:, 0] if ok.any() else np.full(rep.shape[0], np.nan)
    return _row_nanmean(rep[:, ok])


def bootstrap_seed_means(
    xs: Mapping[str, ArrayLike], events: ArrayLike, n_boot: int = 1000, seed: SeedLike = 0, ci: float = 0.95
) -> dict[str, dict[str, Any]]:
    """Event bootstrap of seed-averaged means of several quantities with the same draws.

    xs: name -> (n_seeds, n) or (n,); events: (n,) unit labels. Per name: ``estimate`` (mean over seeds of
    the per-seed means), ``lo``, ``hi``, ``se``, ``per_seed`` (n_seeds,), ``seed_sd``, ``n_events``,
    ``n_seeds`` (seeds with a defined mean), ``replicates`` (n_boot,).
    """
    _check(n_boot, ci)
    arrays = {k: _as_2d(v) for k, v in xs.items()}
    reps, n_units = _replicate_means(list(arrays.values()), np.asarray(events), int(n_boot), seed)
    out: dict[str, dict[str, Any]] = {}
    for (name, x), rep in zip(arrays.items(), reps):
        full = _seed_group_means(x)[:, 0]
        ok = np.isfinite(full)
        rep_avg = _avg_seeds(rep[:, :, 0], ok)
        lo, hi, se = _interval(rep_avg, ci)
        out[name] = {
            "estimate": float(full[ok].mean()) if ok.any() else float("nan"),
            "lo": lo,
            "hi": hi,
            "se": se,
            "per_seed": full,
            "seed_sd": _sd(full),
            "n_events": n_units,
            "n_seeds": int(ok.sum()),
            "replicates": rep_avg,
        }
    return out


# ------------------------------------------------------------------------------ shift ratios
def shift_ratios(
    xs_shift: Mapping[str, ArrayLike],
    events_shift: ArrayLike,
    xs_in: Mapping[str, ArrayLike],
    events_in: ArrayLike,
    n_boot: int = 1000,
    seed: SeedLike = 0,
    ci: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """:func:`shift_ratio` for several quantities with the same draws (dict name -> result)."""
    _check(n_boot, ci)
    d = {k: _as_2d(v) for k, v in xs_shift.items()}
    i = {k: _as_2d(v) for k, v in xs_in.items()}
    if set(d) != set(i):
        raise ValueError("xs_shift and xs_in must have the same quantities")
    for k in d:
        if d[k].shape[0] != i[k].shape[0]:
            raise ValueError(f"{k}: different numbers of seeds in the two data sets")
    seed_d, seed_i = _spawn(seed, 2)
    names = list(d)
    full_d = {k: _seed_group_means(d[k])[:, 0] for k in names}
    full_i = {k: _seed_group_means(i[k])[:, 0] for k in names}
    oks = {k: np.isfinite(full_d[k]) & np.isfinite(full_i[k]) for k in names}  # seeds defined in both sets
    ev_d, ev_i = np.asarray(events_shift), np.asarray(events_in)
    keep_d, keep_i = _any_valid(d.values()), _any_valid(i.values())  # unit populations (all seeds)
    rep_d, n_d = _replicate_means([d[k][oks[k]] for k in names], ev_d, int(n_boot), seed_d, keep=keep_d)
    rep_i, n_i = _replicate_means([i[k][oks[k]] for k in names], ev_i, int(n_boot), seed_i, keep=keep_i)
    out: dict[str, dict[str, Any]] = {}
    for j, name in enumerate(names):
        with np.errstate(invalid="ignore", divide="ignore"):
            per_seed = full_d[name] / full_i[name]
        ok = oks[name]
        mean_d = float(full_d[name][ok].mean()) if ok.any() else float("nan")
        mean_i = float(full_i[name][ok].mean()) if ok.any() else float("nan")
        every = np.ones(int(ok.sum()), dtype=bool)  # the arrays were restricted to the seeds ok
        with np.errstate(invalid="ignore", divide="ignore"):
            reps = _avg_seeds(rep_d[j][:, :, 0], every) / _avg_seeds(rep_i[j][:, :, 0], every)
        lo, hi, se = _interval(reps, ci)
        out[name] = {
            "estimate": _ratio(mean_d, mean_i),
            "lo": lo,
            "hi": hi,
            "se": se,
            "per_seed": per_seed,
            "seed_sd": _sd(per_seed[ok]),
            "n_events_shift": n_d,
            "n_events_in": n_i,
            "n_seeds": int(ok.sum()),
            "mean_shift": mean_d,
            "mean_in": mean_i,
            "replicates": reps,
        }
    return out


def shift_ratio(
    x_shift: ArrayLike,
    events_shift: ArrayLike,
    x_in: ArrayLike,
    events_in: ArrayLike,
    n_boot: int = 1000,
    seed: SeedLike = 0,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Shift ratio ``R = mean_D(x) / mean_I(x)`` with a two-sample event bootstrap (design.md 10.4).

    x_shift: (n_seeds, n_D) values on the shifted data set D, x_in: (n_seeds, n_I) on the in-domain set I
    (one row per seed, same seeds in both; (n,) = one seed); events_*: unit labels per window. The
    estimate is the ratio of the seed-averaged means; D and I are resampled independently, every
    replicate uses the same resampled units for all seeds. Returns ``estimate``, ``lo``, ``hi``, ``se``,
    ``per_seed`` (n_seeds,) per-seed ratios, ``seed_sd`` (their sd, ddof = 1), ``n_events_shift``,
    ``n_events_in``, ``n_seeds``, ``mean_shift``, ``mean_in`` and ``replicates`` (n_boot,).
    """
    return shift_ratios({"x": x_shift}, events_shift, {"x": x_in}, events_in, n_boot, seed, ci)["x"]


def _weighted(means: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted average over the last axis of the finite bin means, weights renormalised over them."""
    fin = np.isfinite(means)
    num = np.where(fin, means * weights, 0.0).sum(axis=-1)
    den = np.where(fin, weights, 0.0).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den > 0, out, np.nan)


def _binned_full(x: np.ndarray, bins: np.ndarray, pos: np.ndarray, n_matched: int) -> np.ndarray:
    """Full-data per-seed means in the matched bins, (K, C); ``pos[bin]`` = matched index or -1."""
    p = pos[bins]
    inside = p >= 0
    return _seed_group_means(x[:, inside], p[inside], n_matched)


def _binned_replicates(
    xs: Sequence[np.ndarray], bins: np.ndarray, events: np.ndarray, keep: np.ndarray, pos: np.ndarray,
    n_matched: int, n_boot: int, seed: SeedLike,
) -> tuple[list[np.ndarray], int]:
    """Replicate means in the matched bins, one (n_boot, K_eff, C) array per input, all from the same draws.

    Seeds are collapsed as in :func:`_collapse`. Rows ``keep`` define the resampled units (all units of the
    data set with a valid value); only rows in matched bins (``pos[bin] >= 0``) contribute. Per-(unit, seed,
    bin) sums are sparse; inputs that miss the same windows share one block of counts.
    """
    codes, units = encode_units(events[keep])
    n_units = len(units)
    p = pos[bins[keep]]
    inside = p >= 0
    mats: list[Any] = []
    layout: list[tuple[int, int, int]] = []  # (K_eff, first column of the sums, first column of the counts)
    count_blocks: list[tuple[np.ndarray, int]] = []
    col = 0
    for x in xs:
        xe = _collapse(x)[:, keep]
        k_eff = xe.shape[0]
        if k_eff == 0:
            layout.append((0, -1, -1))
            continue
        valid = inside[None, :] & ~np.isnan(xe)
        kk, jj = np.nonzero(valid)
        shape = (n_units, k_eff * n_matched)
        coords = (codes[jj], kk * n_matched + p[jj])
        c_col = next((c for v, c in count_blocks if v.shape == valid.shape and np.array_equal(v, valid)), -1)
        if c_col < 0:
            mats.append(sparse.coo_array((np.ones(kk.size), coords), shape=shape))
            c_col = col
            count_blocks.append((valid, c_col))
            col += shape[1]
        mats.append(sparse.coo_array((xe[kk, jj], coords), shape=shape))
        layout.append((k_eff, col, c_col))
        col += shape[1]
    out: list[np.ndarray] = []
    reps = resample_sums(sparse.hstack(mats, format="csr"), n_boot, seed) if mats else None
    for k_eff, s_col, c_col in layout:
        if k_eff == 0 or reps is None:
            out.append(np.full((int(n_boot), 0, n_matched), np.nan))
            continue
        w = k_eff * n_matched
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append((reps[:, s_col : s_col + w] / reps[:, c_col : c_col + w]).reshape(-1, k_eff, n_matched))
    return out, n_units


def matched_shift_ratios(
    xs_shift: Mapping[str, ArrayLike],
    bins_shift: ArrayLike,
    events_shift: ArrayLike,
    xs_in: Mapping[str, ArrayLike],
    bins_in: ArrayLike,
    events_in: ArrayLike,
    grid: "StateGrid | int",
    min_count: int = 30,
    n_boot: int = 1000,
    seed: SeedLike = 0,
    ci: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """:func:`matched_shift_ratio` for several quantities (same matched bins and draws)."""
    _check(n_boot, ci)
    n_bins = int(grid) if isinstance(grid, (int, np.integer)) else int(grid.n_bins)
    d = {k: _as_2d(v) for k, v in xs_shift.items()}
    i = {k: _as_2d(v) for k, v in xs_in.items()}
    if set(d) != set(i):
        raise ValueError("xs_shift and xs_in must have the same quantities")
    b_d = np.asarray(bins_shift, dtype=np.int64).ravel()
    b_i = np.asarray(bins_in, dtype=np.int64).ravel()
    for b, x in ((b_d, d), (b_i, i)):
        if b.shape[0] != next(iter(x.values())).shape[1]:
            raise ValueError("bins and values must have the same number of windows")
        if b.size and (b.min() < 0 or b.max() > n_bins):
            raise ValueError(f"bin indices must lie in 0..{n_bins} (outside cell {n_bins})")
    ev_d, ev_i = np.asarray(events_shift).ravel(), np.asarray(events_in).ravel()
    keep_d, keep_i = _any_valid(d.values()), _any_valid(i.values())
    cnt_d = np.bincount(b_d[keep_d], minlength=n_bins + 1)
    cnt_i = np.bincount(b_i[keep_i], minlength=n_bins + 1)
    matched = np.flatnonzero((cnt_d[:n_bins] >= min_count) & (cnt_i[:n_bins] >= min_count))
    n_matched = int(matched.size)
    mass_in = _ratio(float(cnt_i[matched].sum()), float(cnt_i.sum()))
    mass_shift = _ratio(float(cnt_d[matched].sum()), float(cnt_d.sum()))
    base = {"n_bins_matched": n_matched, "mass_in": mass_in, "mass_shift": mass_shift}
    nan = float("nan")
    if n_matched == 0:
        empty = {
            "estimate": nan, "lo": nan, "hi": nan, "se": nan, "seed_sd": nan, "n_seeds": 0,
            "n_events_shift": len(encode_units(ev_d[keep_d])[1]), "n_events_in": len(encode_units(ev_i[keep_i])[1]),
        }  # fmt: skip
        return {name: {**empty, **base, "per_seed": np.full(d[name].shape[0], nan), "replicates": np.full(n_boot, nan)}
                for name in d}  # fmt: skip
    weights = cnt_i[matched] / cnt_i[matched].sum()
    pos = np.full(n_bins + 1, -1, dtype=np.int64)
    pos[matched] = np.arange(n_matched)
    seed_d, seed_i = _spawn(seed, 2)
    out: dict[str, dict[str, Any]] = {}
    names = list(d)
    m_d = {k: _weighted(_binned_full(d[k], b_d, pos, n_matched), weights) for k in names}  # (K,) per seed
    m_i = {k: _weighted(_binned_full(i[k], b_i, pos, n_matched), weights) for k in names}
    oks = {k: np.isfinite(m_d[k]) & np.isfinite(m_i[k]) for k in names}  # seeds defined in both sets
    reps_d, n_d = _binned_replicates(
        [d[k][oks[k]] for k in names], b_d, ev_d, keep_d, pos, n_matched, int(n_boot), seed_d
    )
    reps_i, n_i = _binned_replicates(
        [i[k][oks[k]] for k in names], b_i, ev_i, keep_i, pos, n_matched, int(n_boot), seed_i
    )
    for j, name in enumerate(names):
        md, mi, ok = m_d[name], m_i[name], oks[name]
        with np.errstate(invalid="ignore", divide="ignore"):
            per_seed = md / mi
        every = np.ones(int(ok.sum()), dtype=bool)  # the replicates were formed from the seeds ok only
        with np.errstate(invalid="ignore", divide="ignore"):
            reps = _avg_seeds(_weighted(reps_d[j], weights), every) / _avg_seeds(_weighted(reps_i[j], weights), every)
        lo, hi, se = _interval(reps, ci)
        out[name] = {
            "estimate": _ratio(float(md[ok].mean()), float(mi[ok].mean())) if ok.any() else nan,
            "lo": lo,
            "hi": hi,
            "se": se,
            "per_seed": per_seed,
            "seed_sd": _sd(per_seed[ok]),
            "n_seeds": int(ok.sum()),
            "n_events_shift": n_d,
            "n_events_in": n_i,
            "replicates": reps,
            **base,
        }
    return out


def matched_shift_ratio(
    x_shift: ArrayLike,
    bins_shift: ArrayLike,
    events_shift: ArrayLike,
    x_in: ArrayLike,
    bins_in: ArrayLike,
    events_in: ArrayLike,
    grid: "StateGrid | int",
    min_count: int = 30,
    n_boot: int = 1000,
    seed: SeedLike = 0,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Bin-matched (covariate-shift adjusted) shift ratio of design.md 10.4.

    Matched bins: in-grid bins (the outside cell is excluded) with at least ``min_count`` windows with a
    valid value in BOTH data sets. With the in-domain bin frequencies ``w_c`` renormalised over the matched
    bins, ``M_X = sum_c w_c mean_X(x | bin c)`` per seed and ``R = mean_k M_D,k / mean_k M_I,k``.
    Matched bins and weights are fixed from the full data; the bootstrap resamples the units of each data
    set (independently, the same units for every seed) and recomputes the bin means inside the matched
    bins. A matched bin that happens to be empty in a replicate is left out of that replicate's weighted
    average (weights renormalised over the non-empty bins). x_*: (n_seeds, n) or (n,); bins_*: flat cell
    indices of :meth:`StateGrid.bin_index`; grid: the grid or its number of in-grid bins.
    Returns the keys of :func:`shift_ratio` plus ``n_bins_matched``, ``mass_in`` and ``mass_shift`` (share
    of each data set's valid windows that lies in the matched bins).
    """
    return matched_shift_ratios(
        {"x": x_shift}, bins_shift, events_shift, {"x": x_in}, bins_in, events_in, grid, min_count, n_boot, seed, ci
    )["x"]


# ----------------------------------------------------------------------------- concentration
def concentrations(
    xs: Mapping[str, ArrayLike],
    masks: Mapping[str, ArrayLike],
    events: ArrayLike,
    n_boot: int = 1000,
    seed: SeedLike = 0,
    ci: float = 0.95,
) -> dict[tuple[str, str], dict[str, Any]]:
    """:func:`concentration` for every (quantity, mask) pair with the same draws; keys ``(name, rule)``."""
    _check(n_boot, ci)
    arrays = {k: _as_2d(v) for k, v in xs.items()}
    ev = np.asarray(events).ravel()
    flags = {k: np.asarray(m, dtype=bool).ravel() for k, m in masks.items()}
    keep = _any_valid(arrays.values())
    codes, units = encode_units(ev[keep])
    n_units = len(units)
    blocks: list[np.ndarray] = []
    layout: list[tuple[str, str, int]] = []
    for name, x in arrays.items():
        xe = _collapse(x)[:, keep]
        for rule, mask in flags.items():
            s, c = _unit_group_sums(xe, codes, n_units, mask[keep].astype(np.int64), 2)
            blocks += [s, c]
            layout.append((name, rule, xe.shape[0]))
    reps = resample_sums(np.hstack(blocks), n_boot, seed)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    col = 0
    for name, rule, k_eff in layout:
        width = 2 * k_eff
        with np.errstate(invalid="ignore", divide="ignore"):
            rep = (reps[:, col : col + width] / reps[:, col + width : col + 2 * width]).reshape(-1, k_eff, 2)
        col += 2 * width
        x = arrays[name]
        mask = flags[rule]
        full = _seed_group_means(x, mask.astype(np.int64), 2)  # [:, 1] inside the mask, [:, 0] outside
        ok = np.isfinite(full[:, 0]) & np.isfinite(full[:, 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            per_seed = full[:, 1] / full[:, 0]
            reps_ratio = _avg_seeds(rep[:, :, 1], ok) / _avg_seeds(rep[:, :, 0], ok)
        lo, hi, se = _interval(reps_ratio, ci)
        # mass-share concentration (share of the mass in the mask / share of the windows in it) = mean(x | mask) /
        # mean(x): the same draws, the overall mean from the resampled sums and counts of both groups
        sums = reps[:, col - 2 * width : col - width].reshape(-1, k_eff, 2)
        counts = reps[:, col - width : col].reshape(-1, k_eff, 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            rep_all = sums.sum(axis=-1) / counts.sum(axis=-1)
            reps_mass = _avg_seeds(rep[:, :, 1], ok) / _avg_seeds(rep_all, ok)
        mass_lo, mass_hi, _ = _interval(reps_mass, ci)
        all_means = _seed_group_means(x)[:, 0]
        valid = (~np.isnan(x)).any(axis=0)
        xbar = _row_nanmean(x.T)  # seed mean per window
        total = float(np.sum(xbar[valid]))
        mean_in = float(full[ok, 1].mean()) if ok.any() else float("nan")
        mean_out = float(full[ok, 0].mean()) if ok.any() else float("nan")
        mean_all = float(all_means[ok].mean()) if ok.any() else float("nan")
        out[(name, rule)] = {
            "estimate": _ratio(mean_in, mean_out),
            "lo": lo,
            "hi": hi,
            "se": se,
            "per_seed": per_seed,
            "seed_sd": _sd(per_seed[ok]),
            "share_windows": float(mask[valid].mean()) if valid.any() else float("nan"),
            "share_mass": _ratio(float(np.sum(xbar[valid & mask])), total),
            "mass_ratio": _ratio(mean_in, mean_all),
            "mass_ratio_lo": mass_lo,
            "mass_ratio_hi": mass_hi,
            "mean_mask": mean_in,
            "mean_rest": mean_out,
            "n_events": n_units,
            "n_seeds": int(ok.sum()),
            "replicates": reps_ratio,
        }
    return out


def concentration(
    x: ArrayLike,
    mask: ArrayLike,
    events: ArrayLike,
    n_boot: int = 1000,
    seed: SeedLike = 0,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Concentration ``mean(x | mask) / mean(x | not mask)`` with a one-sample event bootstrap.

    x: (n_seeds, n) or (n,); mask: (n,) bool (e.g. rare bins, ``s < 15``); events: (n,) unit labels.
    Both conditional means come from the same resampled units (and seeds). Returns ``estimate``, ``lo``,
    ``hi``, ``se``, ``per_seed``, ``seed_sd``, ``share_windows`` = share of the valid windows in the mask,
    ``share_mass`` = ``sum(x[mask]) / sum(x)`` of the seed-averaged x, ``mass_ratio`` (with ``mass_ratio_lo`` /
    ``mass_ratio_hi``) = share of the mass / share of the windows = ``mean(x | mask) / mean(x)`` (the definition of
    the Methods; > 1 exactly when ``estimate`` > 1), ``mean_mask``, ``mean_rest``, ``n_events``, ``n_seeds``,
    ``replicates``.
    """
    return concentrations({"x": x}, {"mask": mask}, events, n_boot, seed, ci)[("x", "mask")]


# ------------------------------------------------------------------------------ H3.1 verdict
@dataclass(frozen=True)
class H31Thresholds:
    """Thresholds of hypothesis H3.1 (``h31`` section of ``configs/maps/default.yaml``)."""

    epi_support: float = 3.0
    epi_refute: float = 1.5
    alea_band: tuple[float, float] = (0.7, 1.3)
    alea_refute: tuple[float, float] = (0.5, 2.0)
    min_pairs: int = 2

    @classmethod
    def from_config(cls, cfg: "Mapping[str, Any] | H31Thresholds | None" = None) -> "H31Thresholds":
        """From the ``h31`` section, a whole maps config, an instance or None (defaults)."""
        if cfg is None:
            return cls()
        if isinstance(cfg, H31Thresholds):
            return cfg
        section = cfg.get("h31", cfg) if isinstance(cfg, Mapping) else cfg
        names = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in dict(section).items():
            if key not in names:
                continue
            if key in ("alea_band", "alea_refute"):
                kwargs[key] = (float(value[0]), float(value[1]))
            elif key == "min_pairs":
                kwargs[key] = int(value)
            else:
                kwargs[key] = float(value)
        return cls(**kwargs)


def _verdict(e_lo: float, e_hi: float, a_lo: float, a_hi: float, th: H31Thresholds) -> str:
    supported = e_lo >= th.epi_support and th.alea_band[0] < a_lo and a_hi < th.alea_band[1]
    refuted = e_hi < th.epi_refute or a_lo > th.alea_refute[1] or a_hi < th.alea_refute[0]
    if supported:
        return "supported"
    return "refuted" if refuted else "inconclusive"


def h31_verdict(
    r_epi: Mapping[str, float], r_alea: Mapping[str, float], cfg: "Mapping[str, Any] | H31Thresholds | None" = None
) -> dict[str, str]:
    """Verdict of H3.1 for one model and pair of data sets (design.md 10.4).

    r_epi, r_alea: shift ratios of the epistemic and aleatoric variance with ``estimate``, ``lo``, ``hi``.
    ``supported``: lower bound of R_epi >= epi_support (3) and the interval of R_alea inside the open band
    (0.7, 1.3); ``refuted``: upper bound of R_epi < epi_refute (1.5), or the interval of R_alea entirely
    above 2 (lower bound > 2) or entirely below 0.5 (upper bound < 0.5); ``inconclusive`` otherwise (also
    when a needed number is NaN). The two conditions are mutually exclusive. ``verdict_point`` applies
    the same rules with both bounds set to the point estimate.
    """
    th = H31Thresholds.from_config(cfg)

    def num(r: Mapping[str, float], key: str) -> float:
        value = r.get(key, float("nan"))
        return float("nan") if value is None else float(value)

    e, a = num(r_epi, "estimate"), num(r_alea, "estimate")
    return {
        "verdict_ci": _verdict(num(r_epi, "lo"), num(r_epi, "hi"), num(r_alea, "lo"), num(r_alea, "hi"), th),
        "verdict_point": _verdict(e, e, a, a, th),
    }


def _overall(n_supported: int, n_refuted: int, min_pairs: int) -> str:
    if n_refuted > 0:
        return "refuted"
    return "supported" if n_supported >= min_pairs else "inconclusive"


H31_SUMMARY_COLUMNS = [
    "model", "kind", "n_pairs", "n_supported", "n_refuted", "n_inconclusive", "overall",
    "n_supported_point", "n_refuted_point", "n_inconclusive_point", "overall_point", "n_placebo", "placebo_verdicts",
    "n_cross_check", "cross_check_verdicts",
]  # fmt: skip


def h31_summary(rows: "pd.DataFrame | Iterable[Mapping[str, Any]]", min_pairs: int = 2) -> pd.DataFrame:
    """Per model (and ``kind``): numbers of supported / refuted / inconclusive pairs and the overall statement.

    rows: verdict rows with ``model``, ``verdict_ci``, ``verdict_point`` and optionally ``kind``, ``subset``
    (only ``"all"`` rows count: subsets are secondary analyses), ``pair`` and ``dataset``. Pairs with
    ``pair == "placebo"`` (another split of the in-domain source, expected R close to 1) do not count;
    they are listed in ``n_placebo`` / ``placebo_verdicts`` (``"dataset:verdict_ci"``) next to the
    verdict. Pairs with ``pair == "cross_check"`` (a second extraction of a data set that is already
    counted, e.g. the FollowNet events of NGSIM I-80) do not count either and are listed in
    ``n_cross_check`` / ``cross_check_verdicts``. Rule, applied separately to the CI-based and to the point verdicts of the counted pairs:

    * ``refuted`` if at least one pair is refuted (a shift pair that contradicts the hypothesis);
    * otherwise ``supported`` if at least ``min_pairs`` pairs are supported;
    * otherwise ``inconclusive`` (including fewer than ``min_pairs`` pairs in total).
    """
    frame = pd.DataFrame(list(rows)) if not isinstance(rows, pd.DataFrame) else rows.copy()
    if frame.empty:
        return pd.DataFrame(columns=H31_SUMMARY_COLUMNS)
    if "subset" in frame.columns:
        frame = frame[frame["subset"].astype(str) == "all"]
    if "kind" not in frame.columns:
        frame = frame.assign(kind="model")
    pair = frame["pair"].astype(str) if "pair" in frame.columns else pd.Series("shift", index=frame.index)
    placebo, cross = pair == "placebo", pair == "cross_check"
    out = []
    for (model, kind), grp in frame.groupby(["model", "kind"], sort=False):
        counted = grp[~(placebo | cross).loc[grp.index]]
        controls, repeats = grp[placebo.loc[grp.index]], grp[cross.loc[grp.index]]
        row: dict[str, Any] = {"model": model, "kind": kind, "n_pairs": int(len(counted))}
        for suffix, column in (("", "verdict_ci"), ("_point", "verdict_point")):
            counts = counted[column].astype(str).value_counts()
            n_sup, n_ref = int(counts.get("supported", 0)), int(counts.get("refuted", 0))
            row[f"n_supported{suffix}"] = n_sup
            row[f"n_refuted{suffix}"] = n_ref
            row[f"n_inconclusive{suffix}"] = int(counts.get("inconclusive", 0))
            row[f"overall{suffix}"] = _overall(n_sup, n_ref, int(min_pairs))
        names = controls["dataset"].astype(str) if "dataset" in controls.columns else ["?"] * len(controls)
        row["n_placebo"] = int(len(controls))
        row["placebo_verdicts"] = ";".join(f"{n}:{v}" for n, v in zip(names, controls["verdict_ci"].astype(str)))
        names = repeats["dataset"].astype(str) if "dataset" in repeats.columns else ["?"] * len(repeats)
        row["n_cross_check"] = int(len(repeats))
        row["cross_check_verdicts"] = ";".join(f"{n}:{v}" for n, v in zip(names, repeats["verdict_ci"].astype(str)))
        out.append(row)
    return pd.DataFrame(out, columns=H31_SUMMARY_COLUMNS)


# ----------------------------------------------------------------------------- H3.1e verdict
@dataclass(frozen=True)
class H31eThresholds:
    """Thresholds of the restated hypothesis H3.1e, epistemic part only (design.md, section 12).

    rule: concentration rule of the in-domain test set (``rare_mass``: the least frequent bins holding at most 10 %
    of the training windows); the concentration is supported when the lower bound of the ratio inside / outside the
    rule is > ``conc_support`` (1) and refuted when its upper bound is < ``conc_refute`` (1: significantly LESS
    epistemic variance in rare states). Growth under shift as in H3.1: a pair supports when the lower bound of
    ``R_epi`` is >= ``epi_support`` (3) and refutes when its upper bound is < ``epi_refute`` (1.5); ``min_pairs``
    (2) supported pairs are needed. The growth thresholds and ``min_pairs`` come from the ``h31`` section of the
    maps config unless the ``h31e`` section sets them.
    """

    rule: str = "rare_mass"
    conc_support: float = 1.0
    conc_refute: float = 1.0
    epi_support: float = 3.0
    epi_refute: float = 1.5
    min_pairs: int = 2

    @classmethod
    def from_config(cls, cfg: "Mapping[str, Any] | H31eThresholds | None" = None) -> "H31eThresholds":
        """From a maps config (section ``h31``, then ``h31e`` over it), the ``h31e`` section alone, an instance or
        None (defaults)."""
        if cfg is None:
            return cls()
        if isinstance(cfg, H31eThresholds):
            return cfg
        values: dict[str, Any] = {}
        if "h31" in cfg or "h31e" in cfg:
            for section in ("h31", "h31e"):
                values.update(dict(cfg.get(section) or {}))
        else:
            values.update(dict(cfg))
        names = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in values.items():
            if key in names:
                kwargs[key] = str(value) if key == "rule" else int(value) if key == "min_pairs" else float(value)
        return cls(**kwargs)


H31E_VERDICT_COLUMNS = [
    "model", "kind", "dataset", "subset", "pair", "rule", "conc", "conc_lo", "conc_hi", "verdict_conc_ci",
    "verdict_conc_point", "r_epi", "r_epi_lo", "r_epi_hi", "verdict_ci", "verdict_point", "n_seeds", "unit",
]  # fmt: skip
H31E_SUMMARY_COLUMNS = [
    "model", "kind", "main", "rule", "conc", "conc_lo", "conc_hi", "verdict_conc_ci", "verdict_conc_point",
    "n_pairs", "n_supported", "n_refuted", "n_inconclusive", "overall", "n_supported_point", "n_refuted_point",
    "n_inconclusive_point", "overall_point", "n_placebo", "placebo_verdicts", "n_cross_check", "cross_check_verdicts",
]  # fmt: skip


def _triple(row: Mapping[str, Any] | None, prefix: str) -> tuple[float, float, float]:
    """(estimate, lo, hi) of ``row``: keys ``<prefix>``, ``<prefix>_lo``, ``<prefix>_hi``, else ``estimate``, ``lo``,
    ``hi``; NaN for a missing row or value."""
    nan = float("nan")
    if row is None:
        return nan, nan, nan
    keys = (prefix, f"{prefix}_lo", f"{prefix}_hi") if prefix in row else ("estimate", "lo", "hi")

    def num(key: str) -> float:
        value = row.get(key)
        try:
            return nan if value is None else float(value)
        except (TypeError, ValueError):
            return nan

    est, lo, hi = (num(k) for k in keys)
    return est, lo, hi


def _rule_verdict(lo: float, hi: float, support: Callable[[float], bool], refute: Callable[[float], bool]) -> str:
    """``supported`` if ``support(lo)``, ``refuted`` if ``refute(hi)``, else ``inconclusive`` (NaN bounds never
    decide)."""
    if np.isfinite(lo) and support(lo):
        return "supported"
    if np.isfinite(hi) and refute(hi):
        return "refuted"
    return "inconclusive"


def h31e_verdict(
    concentration_row: Mapping[str, Any] | None,
    r_epi_rows: Iterable[Mapping[str, Any]],
    cfg: "Mapping[str, Any] | H31eThresholds | None" = None,
) -> list[dict[str, Any]]:
    """Verdict rows of H3.1e for one model and kind of decomposition, one per pair of data sets (design.md 12).

    concentration_row: concentration of the epistemic variance in the in-domain test set for the rule of the
    thresholds (``ratio_epistemic``, ``_lo``, ``_hi`` of a ``rare_concentration`` row, or ``estimate``, ``lo``,
    ``hi``); None or NaN: unknown. r_epi_rows: per pair the epistemic shift ratio (``r_epi``, ``r_epi_lo``,
    ``r_epi_hi`` of an ``h31_verdicts`` row, or ``estimate``, ``lo``, ``hi``) with identifying keys (``model``,
    ``kind``, ``dataset``, ``subset``, ``pair``, ...), which are copied. Adds ``rule``, ``conc``, ``conc_lo``,
    ``conc_hi``, ``verdict_conc_ci`` / ``verdict_conc_point`` (supported: lower bound > conc_support; refuted:
    upper bound < conc_refute), ``r_epi``, ``r_epi_lo``, ``r_epi_hi`` and the growth verdict of the pair,
    ``verdict_ci`` / ``verdict_point`` (supported: lower bound >= epi_support; refuted: upper bound < epi_refute).
    Point verdicts apply the rules with both bounds set to the estimate.
    """
    th = H31eThresholds.from_config(cfg)
    c_est, c_lo, c_hi = _triple(concentration_row, "ratio_epistemic")

    def conc(lo: float, hi: float) -> str:
        return _rule_verdict(lo, hi, lambda x: x > th.conc_support, lambda x: x < th.conc_refute)

    def growth(lo: float, hi: float) -> str:
        return _rule_verdict(lo, hi, lambda x: x >= th.epi_support, lambda x: x < th.epi_refute)

    out = []
    for row in r_epi_rows:
        e_est, e_lo, e_hi = _triple(row, "r_epi")
        out.append(
            {
                **dict(row),
                "rule": th.rule,
                "conc": c_est,
                "conc_lo": c_lo,
                "conc_hi": c_hi,
                "verdict_conc_ci": conc(c_lo, c_hi),
                "verdict_conc_point": conc(c_est, c_est),
                "r_epi": e_est,
                "r_epi_lo": e_lo,
                "r_epi_hi": e_hi,
                "verdict_ci": growth(e_lo, e_hi),
                "verdict_point": growth(e_est, e_est),
            }
        )
    return out


def _overall_e(conc: str, n_supported: int, n_refuted: int, min_pairs: int) -> str:
    if conc == "refuted" or n_refuted > 0:
        return "refuted"
    return "supported" if conc == "supported" and n_supported >= min_pairs else "inconclusive"


def h31e_summary(rows: "pd.DataFrame | Iterable[Mapping[str, Any]]", min_pairs: int = 2) -> pd.DataFrame:
    """Per model and ``kind``: the H3.1e statement from the verdict rows of :func:`h31e_verdict`.

    Pairs are counted as in :func:`h31_summary`: whole data sets only (``subset == "all"``); placebo and
    cross-check pairs are listed (``placebo_verdicts``, ``cross_check_verdicts``: ``"dataset:verdict_ci"`` of the
    growth) but not counted. Rule, separately for the CI-based and the point verdicts:

    * ``refuted`` if the concentration is refuted or at least one counted pair is refuted;
    * otherwise ``supported`` if the concentration is supported and at least ``min_pairs`` pairs are supported;
    * otherwise ``inconclusive`` (also with an unknown concentration or fewer than ``min_pairs`` pairs).

    ``main`` is True for ``kind == "model"`` (own epistemic output of the model: the main verdict); seed-ensemble
    rows are reported separately.
    """
    frame = pd.DataFrame(list(rows)) if not isinstance(rows, pd.DataFrame) else rows.copy()
    if frame.empty:
        return pd.DataFrame(columns=H31E_SUMMARY_COLUMNS)
    if "subset" in frame.columns:
        frame = frame[frame["subset"].astype(str) == "all"]
    if "kind" not in frame.columns:
        frame = frame.assign(kind="model")
    pair = frame["pair"].astype(str) if "pair" in frame.columns else pd.Series("shift", index=frame.index)
    placebo, cross = pair == "placebo", pair == "cross_check"
    out = []
    for (model, kind), grp in frame.groupby(["model", "kind"], sort=False):
        counted = grp[~(placebo | cross).loc[grp.index]]
        controls, repeats = grp[placebo.loc[grp.index]], grp[cross.loc[grp.index]]
        first = grp.iloc[0]
        row: dict[str, Any] = {"model": model, "kind": kind, "main": kind == "model", "rule": first.get("rule")}
        for key in ("conc", "conc_lo", "conc_hi", "verdict_conc_ci", "verdict_conc_point"):
            row[key] = first.get(key)
        row["n_pairs"] = int(len(counted))
        for suffix, column, conc in (("", "verdict_ci", "verdict_conc_ci"), ("_point", "verdict_point", "verdict_conc_point")):
            counts = counted[column].astype(str).value_counts()
            n_sup, n_ref = int(counts.get("supported", 0)), int(counts.get("refuted", 0))
            row[f"n_supported{suffix}"] = n_sup
            row[f"n_refuted{suffix}"] = n_ref
            row[f"n_inconclusive{suffix}"] = int(counts.get("inconclusive", 0))
            row[f"overall{suffix}"] = _overall_e(str(first.get(conc)), n_sup, n_ref, int(min_pairs))
        names = controls["dataset"].astype(str) if "dataset" in controls.columns else ["?"] * len(controls)
        row["n_placebo"] = int(len(controls))
        row["placebo_verdicts"] = ";".join(f"{n}:{v}" for n, v in zip(names, controls["verdict_ci"].astype(str)))
        names = repeats["dataset"].astype(str) if "dataset" in repeats.columns else ["?"] * len(repeats)
        row["n_cross_check"] = int(len(repeats))
        row["cross_check_verdicts"] = ";".join(f"{n}:{v}" for n, v in zip(names, repeats["verdict_ci"].astype(str)))
        out.append(row)
    return pd.DataFrame(out, columns=H31E_SUMMARY_COLUMNS)
