"""State grid, bin assignment, rare-state flags and uncertainty maps (docs/design.md, section 10.3).

Grid: gap ``s`` (m) x relative speed ``dv = v - v_lead`` (m/s) with right-open bins
``[lo + k step, lo + (k + 1) step)``; the bin of a window is the bin of its last history sample.
Cells are indexed flat, ``i_s * n_dv + i_dv`` for the ``n_bins = n_s * n_dv`` in-grid bins; the extra
cell ``n_bins`` ("outside") collects windows outside the grid or with a NaN state, so no window is
dropped. Arrays over cells have ``n_bins + 1`` entries, the outside cell last.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from cf_unc.maps.shift import QUANTITIES, SeedStack, quantity_arrays

__all__ = [
    "QUANTITIES",
    "StateGrid",
    "bin_counts",
    "frequencies",
    "rare_flags",
    "bin_means",
    "UncertaintyMap",
    "build_map",
    "map_from_arrays",
]

_EDGE_DECIMALS = 9  # edges are snapped to 9 decimals, so that e.g. 0 + 3 * 0.1 is the double nearest to 0.3


@dataclass(frozen=True)
class StateGrid:
    """Rectangular (s, dv) grid; the defaults are those of ``configs/maps/default.yaml`` (24 x 20 bins)."""

    s_lo: float = 0.0
    s_hi: float = 120.0
    s_step: float = 5.0
    dv_lo: float = -5.0
    dv_hi: float = 5.0
    dv_step: float = 0.5

    def __post_init__(self) -> None:
        for axis in ("s", "dv"):
            lo, hi, step = (float(getattr(self, f"{axis}_{part}")) for part in ("lo", "hi", "step"))
            if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(step) and step > 0 and hi > lo):
                raise ValueError(f"{axis}: need finite lo < hi and step > 0, got {lo}, {hi}, {step}")
            n = (hi - lo) / step
            if abs(n - round(n)) > 1e-6:
                raise ValueError(f"{axis}: the range {hi - lo} is not a multiple of the step {step}")

    @classmethod
    def from_config(cls, mapping: Mapping[str, Any]) -> "StateGrid":
        """From the ``grid`` section ``{s: {lo, hi, step}, dv: {...}}``, a whole maps config or flat keys."""
        m = dict(mapping)
        if isinstance(m.get("grid"), Mapping):
            m = dict(m["grid"])
        if isinstance(m.get("s"), Mapping) and isinstance(m.get("dv"), Mapping):
            s, dv = m["s"], m["dv"]
            return cls(
                float(s["lo"]), float(s["hi"]), float(s["step"]), float(dv["lo"]), float(dv["hi"]), float(dv["step"])
            )
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in m.items() if k in names})

    @property
    def n_s(self) -> int:
        return int(round((self.s_hi - self.s_lo) / self.s_step))

    @property
    def n_dv(self) -> int:
        return int(round((self.dv_hi - self.dv_lo) / self.dv_step))

    @property
    def n_bins(self) -> int:
        """In-grid bins; the outside cell has index ``n_bins``."""
        return self.n_s * self.n_dv

    @property
    def n_cells(self) -> int:
        return self.n_bins + 1

    @property
    def s_edges(self) -> np.ndarray:
        """(n_s + 1,) float64, m."""
        return _edges(self.s_lo, self.s_step, self.n_s)

    @property
    def dv_edges(self) -> np.ndarray:
        """(n_dv + 1,) float64, m/s."""
        return _edges(self.dv_lo, self.dv_step, self.n_dv)

    def bin_index(self, s: ArrayLike, dv: ArrayLike) -> np.ndarray:
        """Flat cell index (n,) int64 of states (s, dv); ``n_bins`` outside the grid or for NaN.

        Comparisons are made in the precision of the input: a float32 value equal to the float32 rounding
        of an edge lies on that edge (bin to its right), e.g. ``np.float32(0.7)`` on a 0.1 grid.
        """
        i_s = _axis_index(s, self.s_edges)
        i_dv = _axis_index(dv, self.dv_edges)
        if i_s.shape != i_dv.shape:
            raise ValueError("s and dv must have the same shape")
        inside = (i_s >= 0) & (i_dv >= 0)
        return np.where(inside, i_s * self.n_dv + i_dv, self.n_bins).astype(np.int64)

    def cell_bounds(self) -> dict[str, np.ndarray]:
        """Per cell (n_bins + 1,): ``i_s``, ``i_dv`` (-1 for the outside cell) and ``s_lo``, ``s_hi``,
        ``dv_lo``, ``dv_hi`` (NaN for the outside cell)."""
        i_s = np.repeat(np.arange(self.n_s), self.n_dv)
        i_dv = np.tile(np.arange(self.n_dv), self.n_s)
        se, de = self.s_edges, self.dv_edges
        nan = np.array([np.nan])
        return {
            "i_s": np.append(i_s, -1),
            "i_dv": np.append(i_dv, -1),
            "s_lo": np.concatenate([se[i_s], nan]),
            "s_hi": np.concatenate([se[i_s + 1], nan]),
            "dv_lo": np.concatenate([de[i_dv], nan]),
            "dv_hi": np.concatenate([de[i_dv + 1], nan]),
        }

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def _edges(lo: float, step: float, n: int) -> np.ndarray:
    return np.round(lo + step * np.arange(n + 1, dtype=np.float64), _EDGE_DECIMALS)


def _axis_index(x: ArrayLike, edges: np.ndarray) -> np.ndarray:
    """Index k of the right-open bin ``[edges[k], edges[k + 1])`` of x (int64), -1 outside or for NaN."""
    arr = np.asarray(x)
    if arr.dtype.kind == "f" and arr.dtype.itemsize < 8:
        e = edges.astype(arr.dtype)  # compare in the input precision
    else:
        arr = arr.astype(np.float64)
        e = edges
    k = np.searchsorted(e, arr, side="right").astype(np.int64) - 1
    ok = (k >= 0) & (k < len(edges) - 1) & ~np.isnan(arr)
    return np.where(ok, k, -1)


def bin_counts(bins: ArrayLike, grid: StateGrid) -> np.ndarray:
    """Windows per cell, (n_bins + 1,) int64 (outside cell last); sums to the number of windows."""
    b = np.asarray(bins, dtype=np.int64).ravel()
    if b.size and (b.min() < 0 or b.max() > grid.n_bins):
        raise ValueError(f"bin indices must lie in 0..{grid.n_bins}")
    return np.bincount(b, minlength=grid.n_cells).astype(np.int64)


def frequencies(counts: ArrayLike) -> np.ndarray:
    """Counts divided by their total (float64, NaN everywhere when the total is 0)."""
    c = np.asarray(counts, dtype=np.float64)
    total = c.sum()
    return c / total if total > 0 else np.full(c.shape, np.nan)


def rare_flags(
    train_counts: ArrayLike, freq_threshold: float = 0.01, mass_threshold: float = 0.10
) -> dict[str, np.ndarray]:
    """Rare cells relative to the TRAINING density (design.md 10.3); boolean (n_bins + 1,) arrays.

    train_counts: (n_bins + 1,) training windows per cell, the outside cell last. Frequencies are relative
    to all training windows (outside cell included).

    * ``rare_freq``: training frequency < ``freq_threshold`` (definition of the plan);
    * ``rare_mass``: the least frequent cells that together hold at most ``mass_threshold`` of the training
      windows. Cells are accumulated in the order: outside cell first, then the in-grid bins by increasing
      count, ties by increasing flat index (stable sort); a cell is rare while the cumulative count
      (itself included) is <= ``mass_threshold * total``.

    In both rules the outside cell and every cell without training windows are rare. Also returns
    ``freq_train`` (n_bins + 1,).
    """
    c = np.asarray(train_counts, dtype=np.float64).ravel()
    if c.size < 2:
        raise ValueError("train_counts needs the in-grid bins and the outside cell")
    if np.any(c < 0) or not np.all(np.isfinite(c)):
        raise ValueError("train_counts must be finite and non-negative")
    outside = c.size - 1
    total = c.sum()
    if total <= 0:
        ones = np.ones(c.size, dtype=bool)
        return {"rare_freq": ones, "rare_mass": ones.copy(), "freq_train": np.full(c.size, np.nan)}
    freq = c / total
    rare_freq = freq < freq_threshold
    order = np.concatenate([[outside], np.argsort(c[:outside], kind="stable")])
    cumulative = np.cumsum(c[order])
    rare_mass = np.zeros(c.size, dtype=bool)
    rare_mass[order] = cumulative <= mass_threshold * total + 1e-9 * total
    for flags in (rare_freq, rare_mass):
        flags[outside] = True
        flags[c == 0] = True
    return {"rare_freq": rare_freq, "rare_mass": rare_mass, "freq_train": freq}


def bin_means(values: ArrayLike, bins: ArrayLike, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware per-cell means: ``(means, counts)``.

    values: (n,) or (n_rows, n) (e.g. one row per seed); bins: (n,) cell indices in 0..n_cells-1.
    Returns float64 means and int64 counts of the non-NaN values, (n_cells,) or (n_rows, n_cells);
    cells without a value give NaN.
    """
    v = np.asarray(values, dtype=np.float64)
    squeeze = v.ndim == 1
    v2 = v[None, :] if squeeze else v
    b = np.asarray(bins, dtype=np.int64).ravel()
    if v2.ndim != 2 or v2.shape[1] != b.shape[0]:
        raise ValueError(f"values must be (n,) or (n_rows, n) with n = {b.shape[0]}, got {v.shape}")
    if b.size and (b.min() < 0 or b.max() >= n_cells):
        raise ValueError(f"bin indices must lie in 0..{n_cells - 1}")
    rows = v2.shape[0]
    idx = (np.arange(rows, dtype=np.int64)[:, None] * n_cells + b[None, :]).ravel()
    flat = v2.ravel()
    ok = ~np.isnan(flat)
    sums = np.bincount(idx[ok], weights=flat[ok], minlength=rows * n_cells).reshape(rows, n_cells)
    counts = np.bincount(idx[ok], minlength=rows * n_cells).reshape(rows, n_cells).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    means[counts == 0] = np.nan
    return (means[0], counts[0]) if squeeze else (means, counts)


def _seed_stats(cell_means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sample std (ddof = 1) over axis 0 of the finite entries (NaN where too few)."""
    fin = np.isfinite(cell_means)
    n = fin.sum(axis=0)
    x = np.where(fin, cell_means, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = x.sum(axis=0) / n
        dev = np.where(fin, (cell_means - mean) ** 2, 0.0).sum(axis=0)
        std = np.sqrt(dev / (n - 1))
    return np.where(n > 0, mean, np.nan), np.where(n > 1, std, np.nan)


@dataclass
class UncertaintyMap:
    """Per-cell statistics of one model / kind / data set on a :class:`StateGrid` (arrays (n_bins + 1,)).

    count       evaluated windows (a prediction in at least one seed) per cell, int64
    freq_eval   count / total evaluated windows
    freq_train  frequency in the highD training windows
    rare_freq, rare_mass   rare flags of :func:`rare_flags` (training density)
    mean[q]     seed mean of the per-seed cell means of quantity q (``QUANTITIES``)
    std[q]      across-seed standard deviation (ddof = 1) of the per-seed cell means (NaN with one seed,
                and for the single-row quantities of the seed ensemble)
    """

    grid: StateGrid
    count: np.ndarray
    freq_eval: np.ndarray
    freq_train: np.ndarray
    rare_freq: np.ndarray
    rare_mass: np.ndarray
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]
    n_seeds: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def quantities(self) -> tuple[str, ...]:
        return tuple(self.mean)

    def to_frame(self) -> pd.DataFrame:
        """Long format, one row per cell (in-grid bins in flat order, the outside cell last with NaN edges)."""
        bounds = self.grid.cell_bounds()
        data: dict[str, Any] = {"cell": np.arange(self.grid.n_cells), **bounds}
        data.update(
            count=self.count,
            freq_eval=self.freq_eval,
            freq_train=self.freq_train,
            rare_freq=self.rare_freq,
            rare_mass=self.rare_mass,
        )
        for q in self.quantities:
            data[q] = self.mean[q]
            data[f"{q}_sd"] = self.std[q]
        data["n_seeds"] = np.full(self.grid.n_cells, int(self.n_seeds))
        return pd.DataFrame(data)

    def save(self, path: str | Path) -> Path:
        """Write a compressed ``.npz`` (no pickled objects)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        q = list(self.quantities)
        np.savez_compressed(
            path,
            grid=np.array([getattr(self.grid, f.name) for f in fields(StateGrid)], dtype=np.float64),
            count=self.count,
            freq_eval=self.freq_eval,
            freq_train=self.freq_train,
            rare_freq=self.rare_freq,
            rare_mass=self.rare_mass,
            quantities=np.array(q, dtype=str),
            mean=np.stack([self.mean[k] for k in q]) if q else np.empty((0, self.grid.n_cells)),
            std=np.stack([self.std[k] for k in q]) if q else np.empty((0, self.grid.n_cells)),
            n_seeds=np.int64(self.n_seeds),
            meta=np.array(json.dumps(self.meta, default=str)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "UncertaintyMap":
        with np.load(Path(path), allow_pickle=False) as z:
            grid = StateGrid(*(float(v) for v in z["grid"]))
            names = [str(s) for s in z["quantities"]]
            return cls(
                grid=grid,
                count=z["count"].copy(),
                freq_eval=z["freq_eval"].copy(),
                freq_train=z["freq_train"].copy(),
                rare_freq=z["rare_freq"].copy(),
                rare_mass=z["rare_mass"].copy(),
                mean={k: z["mean"][i].copy() for i, k in enumerate(names)},
                std={k: z["std"][i].copy() for i, k in enumerate(names)},
                n_seeds=int(z["n_seeds"]),
                meta=json.loads(str(z["meta"])),
            )

    def masked(self, quantity: str, min_count: int = 30) -> np.ndarray:
        """(n_s, n_dv) seed-mean map of ``quantity`` with NaN where fewer than ``min_count`` windows."""
        values = np.where(self.count[: self.grid.n_bins] >= min_count, self.mean[quantity][: self.grid.n_bins], np.nan)
        return values.reshape(self.grid.n_s, self.grid.n_dv)


def map_from_arrays(
    bins: ArrayLike,
    quantities: Mapping[str, ArrayLike],
    evaluated: ArrayLike,
    grid: StateGrid,
    train_counts: ArrayLike,
    *,
    rare: Mapping[str, np.ndarray] | None = None,
    freq_threshold: float = 0.01,
    mass_threshold: float = 0.10,
    n_seeds: int | None = None,
    meta: Mapping[str, Any] | None = None,
) -> UncertaintyMap:
    """Map from per-window arrays: bins (n,), quantities name -> (n_rows, n) or (n,), evaluated (n,) bool.

    ``rare``: precomputed :func:`rare_flags` of ``train_counts`` (computed from the thresholds otherwise).
    """
    b = np.asarray(bins, dtype=np.int64).ravel()
    train = np.asarray(train_counts, dtype=np.float64).ravel()
    if train.shape[0] != grid.n_cells:
        raise ValueError(f"train_counts must have {grid.n_cells} entries (outside cell last), got {train.shape[0]}")
    flags = dict(rare) if rare is not None else rare_flags(train, freq_threshold, mass_threshold)
    ev = np.asarray(evaluated, dtype=bool).ravel()
    count = bin_counts(b[ev], grid)
    mean: dict[str, np.ndarray] = {}
    std: dict[str, np.ndarray] = {}
    rows = 0
    for name, values in quantities.items():
        v = np.asarray(values, dtype=np.float64)
        per_row, _ = bin_means(v if v.ndim == 2 else v[None, :], b, grid.n_cells)
        mean[name], std[name] = _seed_stats(per_row)
        rows = max(rows, per_row.shape[0])
    return UncertaintyMap(
        grid=grid,
        count=count,
        freq_eval=frequencies(count),
        freq_train=frequencies(train),
        rare_freq=np.asarray(flags["rare_freq"], dtype=bool),
        rare_mass=np.asarray(flags["rare_mass"], dtype=bool),
        mean=mean,
        std=std,
        n_seeds=int(rows if n_seeds is None else n_seeds),
        meta=dict(meta or {}),
    )


def build_map(
    windows_frame: pd.DataFrame,
    pred_frames: "Sequence[pd.DataFrame] | SeedStack",
    grid: StateGrid,
    train_counts: ArrayLike,
    *,
    kind: str = "model",
    freq_threshold: float = 0.01,
    mass_threshold: float = 0.10,
    meta: Mapping[str, Any] | None = None,
) -> UncertaintyMap:
    """Uncertainty map of one data set: windows table (``s``, ``dv``, ``y``) and one prediction table per seed.

    Quantities (``QUANTITIES``) per kind as in :func:`cf_unc.maps.shift.quantity_arrays`: ``"model"`` uses
    the model's own decomposition, ``"seed_ensemble"`` the across-seed decomposition. Per cell: the mean
    of every quantity per seed (NaN-aware), then the mean and std over the seeds.
    """
    stack = SeedStack.coerce(pred_frames)
    if len(stack) != len(windows_frame):
        raise ValueError("windows and prediction tables have different numbers of rows")
    bins = grid.bin_index(np.asarray(windows_frame["s"]), np.asarray(windows_frame["dv"]))
    q = quantity_arrays(stack, np.asarray(windows_frame["y"], dtype=np.float64), kind)
    info = {"kind": kind, **dict(meta or {})}
    return map_from_arrays(
        bins, q, stack.valid.any(axis=0), grid, train_counts,
        freq_threshold=freq_threshold, mass_threshold=mass_threshold, n_seeds=stack.n_seeds, meta=info,
    )  # fmt: skip
