"""Analysis of experiment E1: uncertainty maps, dataset shift (H3.1), rare states and regimes (H3.3).

:func:`analyze_e1` reads what :func:`cf_unc.maps.io.available` lists under ``root`` (windows tables and one
prediction table per model, seed and data set), ``root/train_density.npz`` and ``root/idm.json``, and
writes (docs/design.md, section 10)::

    tables/one_step_metrics.csv    model x data set x subset x seed (+ seed mean with event-bootstrap CIs)
    tables/shift_ratios.csv        R = mean_D(x) / mean_I(x), plain and bin-matched, per kind and quantity
    tables/h31_verdicts.csv        H3.1 verdict per model, kind and pair of data sets
    tables/h31_summary.csv         per model and kind, whole shifted data sets only
    tables/h31e_verdicts.csv       H3.1e (epistemic part only, design.md section 12): in-domain concentration and
                                   growth verdict per model, kind and pair (from rare_concentration and h31_verdicts)
    tables/h31e_summary.csv        per model and kind (main verdict: kind "model", own epistemic output)
    tables/rare_concentration.csv  mean(x | rule) / mean(x | not rule): rare_freq, rare_mass, small_gap, large_dv; next to
                                   it the mass-share concentration conc_mass_* = share of the mass / share of the
                                   windows = mean(x | rule) / mean(x) (both > 1 together)
    tables/regime_stats.csv        statistics per regime
    tables/regime_ordering.csv     ratio of every regime to the reference regime (Holm per model, data set, subset)
    tables/h33_nll.csv             NLL of hetero_mdn minus lstm_const_sigma (Holm over the whole table); other pairs
                                   with ``h33_pairs`` (e.g. lstm_hetero_frozen vs lstm_const_sigma, review control R13)
    tables/maps_long.csv           every map in long format
    tables/analysis_info.json      settings, seeds per model and data set, run time
    maps/<model>__<kind>/<dataset>.npz, <dataset>__<subset>.npz

Conventions
* ``kind = "model"``: the model's own decomposition; ``kind = "seed_ensemble"``: across-seed decomposition
  (variance of the seeds' means, ddof = 0, and the seed mean of the model variance), produced for every
  model with at least two seeds. The two are always separate rows. H3.1 verdicts need a non-degenerate
  epistemic part: none for kind "model" of models without one (HeteroMDN, QuantilePIDL, DiffusionCF,
  LSTMConstSigma, IDMWhiteNoise) and none when the in-domain epistemic mean is 0 (identical seeds).
* Subsets (column ``subset`` of the windows table) are analysed in addition to the whole data set
  (``subset = "all"``) when a data set has more than one.
* Every data set other than the in-domain one is treated as a shifted set and compared with the whole
  in-domain set (ratios, matched ratios, verdicts); no data set name is hard-coded. ``pair`` is
  ``"placebo"`` for another split of the in-domain source (source = ``spec.data`` of the windows meta,
  e.g. ``highd_val``: its ratios should come out near 1) and ``"shift"`` otherwise; ``placebo=[...]``
  overrides the automatic choice. Placebo pairs are reported next to the H3.1 summary but not counted
  as evidence (a placebo that "refutes" would otherwise refute H3.1 for every model).
* Pooled training domain (design.md 16.1): the in-domain set may be a pool (``spec.members`` of its meta, e.g.
  ``pool_test``, the union of the test folds of the training sets); its sources are the members' data sets. Its
  members are parts of the in-domain reference and get no pair (they are analysed as data sets); a data set whose
  source is a training data set (``train_data``: argument, else ``pool.data`` of ``train_density.npz``, else the
  in-domain sources) but not an in-domain source gets no pair either, nor does a cross-check that repeats a training
  data set (``cross_check_of`` of the config: cross-check -> the data set it repeats, e.g. ``ngsim_fn`` -> ``ngsim``).
  The shift pairs are therefore the data sets that are not in the training domain.
* Seeds: every table uses the seeds whose prediction table exists for that data set (``n_seeds``);
  ratios between two data sets use the seeds present for both.
* Bootstrap units: events (default) or drivers (``bootstrap.unit`` or ``unit``). Bootstrap seeds are
  derived from ``bootstrap.seed`` and the data set / subset / statistic (not the model), so every model is
  resampled with the same events (common random numbers across models).
* Tests: a count-based bootstrap p-value (+1 smoothing) cannot fall below 2 / (B + 1), while Holm over m
  hypotheses needs p < alpha / m. P-values that enter a Holm family of m hypotheses are therefore computed
  with ``B_test = max(n_boot, ceil(20 m / alpha))`` replicates (floor 10 times below alpha / m):
  regime_ordering (m = 10 per model, data set and subset) and h33_nll (m = all rows of the table).
* Every statistic is NaN-aware: rows a model did not evaluate (DiffusionCF) are NaN in its tables.
"""

from __future__ import annotations

import copy
import json
import re
import time
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from cf_unc.eval.bootstrap import holm
from cf_unc.eval.metrics import calibration_error
from cf_unc.maps import io
from cf_unc.maps.grid import StateGrid, map_from_arrays, rare_flags
from cf_unc.maps.regime_stats import NLL_COLUMNS, ORDERING_COLUMNS, REGIME_TABLE_COLUMNS, nll_comparison
from cf_unc.maps.regime_stats import regime_ordering, regime_table
from cf_unc.maps.regimes import PRIORITY, REGIMES, RegimeThresholds, label_regimes
from cf_unc.maps.shift import (
    H31_SUMMARY_COLUMNS,
    H31E_SUMMARY_COLUMNS,
    H31E_VERDICT_COLUMNS,
    H31eThresholds,
    H31Thresholds,
    SeedStack,
    bootstrap_seed_means,
    concentrations,
    encode_units,
    h31_summary,
    h31_verdict,
    h31e_summary,
    h31e_verdict,
    has_epistemic,
    matched_shift_ratios,
    own_decomposition,
    quantity_arrays,
    shift_ratios,
)

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_CONFIG_PATH",
    "H33_MODELS",
    "RULES",
    "TABLE_COLUMNS",
    "analyze_e1",
    "format_report",
    "h31e_tables",
    "holm_replicates",
    "load_config",
    "load_train_density",
    "pair_roles",
    "resolve_v0",
    "stored_train_data",
]

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "maps" / "default.yaml"
DEFAULT_CONFIG: dict[str, Any] = {  # mirror of configs/maps/default.yaml (used when the file is absent)
    "grid": {"s": {"lo": 0.0, "hi": 120.0, "step": 5.0}, "dv": {"lo": -5.0, "hi": 5.0, "step": 0.5}},
    "rare": {"freq_threshold": 0.01, "mass_threshold": 0.10, "min_count_display": 30},
    "regions": {"small_gap": {"s_max": 15.0}, "large_dv": {"abs_dv_min": 3.0}},
    "regimes": {
        "s_congested": 15.0, "v_congested": 8.0, "s_free": 60.0, "s_lo": 10.0, "dv_eps": 0.5, "a_brake": -0.5,
        "v_free_frac": 0.8, "v0": None, "priority": list(PRIORITY), "reference": "steady",
    },  # fmt: skip
    "bootstrap": {"n_boot": 1000, "ci": 0.95, "seed": 0, "unit": "event"},
    "h31": {"epi_support": 3.0, "epi_refute": 1.5, "alea_band": [0.7, 1.3], "alea_refute": [0.5, 2.0], "min_pairs": 2},
    "h31e": {"rule": "rare_mass", "conc_support": 1.0, "conc_refute": 1.0},
    "matched": {"min_count": 30},
}
H33_MODELS = ("hetero_mdn", "lstm_const_sigma")  # heteroscedastic model a, homoscedastic model b
ALPHA = 0.05  # level of the Holm corrections
RULES = ("rare_freq", "rare_mass", "small_gap", "large_dv")
SHIFT_QUANTITIES = ("epistemic", "aleatoric", "total")
_METRICS = ("nll", "crps", "picp90", "width90", "rmse", "var", "aleatoric", "epistemic", "pit_ks", "calibration_error")

TABLE_COLUMNS: dict[str, list[str]] = {
    "one_step_metrics": [
        "model", "dataset", "subset", "seed", "n_seeds", "n_windows", "n_events", *_METRICS,
        "nll_lo", "nll_hi", "crps_lo", "crps_hi", "picp90_lo", "picp90_hi", "unit", "n_followers",
    ],
    "shift_ratios": [
        "model", "kind", "dataset", "subset", "pair", "quantity", "estimate", "lo", "hi", "se", "seed_sd",
        "matched_estimate", "matched_lo", "matched_hi", "n_bins_matched", "mass_in", "mass_shift",
        "unit", "n_events", "n_events_in", "n_seeds", "mean_shift", "mean_in",
    ],
    "h31_verdicts": [
        "model", "kind", "dataset", "subset", "pair", "r_epi", "r_epi_lo", "r_epi_hi", "r_alea", "r_alea_lo",
        "r_alea_hi", "verdict_ci", "verdict_point", "r_epi_matched", "r_epi_matched_lo", "r_epi_matched_hi",
        "r_alea_matched", "r_alea_matched_lo", "r_alea_matched_hi", "verdict_ci_matched", "verdict_point_matched",
        "n_seeds", "unit",
    ],
    "h31_summary": list(H31_SUMMARY_COLUMNS),
    "h31e_verdicts": list(H31E_VERDICT_COLUMNS),
    "h31e_summary": list(H31E_SUMMARY_COLUMNS),
    "rare_concentration": [
        "model", "kind", "dataset", "subset", "rule", "share_windows", "ratio_epistemic", "ratio_epistemic_lo",
        "ratio_epistemic_hi", "ratio_aleatoric", "ratio_aleatoric_lo", "ratio_aleatoric_hi", "share_mass_epistemic",
        "share_mass_aleatoric", "n_events", "n_seeds", "unit", "conc_mass_epistemic", "conc_mass_epistemic_lo",
        "conc_mass_epistemic_hi", "conc_mass_aleatoric", "conc_mass_aleatoric_lo", "conc_mass_aleatoric_hi",
    ],
    "regime_stats": ["model", "dataset", "subset", *REGIME_TABLE_COLUMNS, "unit"],
    "regime_ordering": ["model", "dataset", "subset", *ORDERING_COLUMNS, "unit"],
    "h33_nll": ["dataset", "subset", "model_a", "model_b", *NLL_COLUMNS, "p_holm", "reject", "unit"],
}  # fmt: skip
MAP_KEYS = ["model", "kind", "dataset", "subset"]


# ------------------------------------------------------------------------------ configuration
def _merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
    return base


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml  # PyYAML ships with hydra-core / omegaconf

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return dict(data or {})


def load_config(cfg: "str | Path | Mapping[str, Any] | None" = None) -> dict[str, Any]:
    """Maps config: ``DEFAULT_CONFIG`` <- ``configs/maps/default.yaml`` (or the file ``cfg``) <- mapping ``cfg``.

    A mapping may be partial (e.g. ``{"bootstrap": {"n_boot": 200}}``); nested sections are merged.
    """
    merged = copy.deepcopy(DEFAULT_CONFIG)
    if isinstance(cfg, (str, Path)):
        return _merge(merged, _read_yaml(Path(cfg)))
    if DEFAULT_CONFIG_PATH.exists():
        _merge(merged, _read_yaml(DEFAULT_CONFIG_PATH))
    if cfg is not None:
        _merge(merged, cfg)
    return merged


def load_train_density(root: str | Path, grid: StateGrid) -> np.ndarray:
    """Training windows per cell (n_bins + 1,) float64 from ``root/train_density.npz`` (outside cell last).

    The stored edges must be those of ``grid``; ``n_total`` (when present) must equal the sum.
    """
    path = Path(root) / "train_density.npz"
    with np.load(path, allow_pickle=False) as z:
        counts = np.asarray(z["counts"], dtype=np.float64)
        n_outside = float(z["n_outside"])
        s_edges = np.asarray(z["s_edges"], dtype=np.float64) if "s_edges" in z.files else None
        dv_edges = np.asarray(z["dv_edges"], dtype=np.float64) if "dv_edges" in z.files else None
        n_total = float(z["n_total"]) if "n_total" in z.files else None
    if counts.shape != (grid.n_s, grid.n_dv):
        raise ValueError(f"{path}: counts have shape {counts.shape}, the map grid is {(grid.n_s, grid.n_dv)}")
    for name, stored, own in (("s", s_edges, grid.s_edges), ("dv", dv_edges, grid.dv_edges)):
        if stored is not None and (stored.shape != own.shape or not np.allclose(stored, own, rtol=0, atol=1e-6)):
            raise ValueError(f"{path}: {name} edges differ from the map grid ({stored} vs {own})")
    train = np.concatenate([counts.ravel(), [n_outside]])
    if n_total is not None and abs(train.sum() - n_total) > 0.5:
        message = f"{path}: counts sum to {train.sum():.0f}, n_total is {n_total:.0f}"
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return train


def resolve_v0(root: str | Path, cfg: Mapping[str, Any]) -> float | None:
    """IDM free speed for the regimes: ``regimes.v0`` of the config, else ``params.v0`` of ``root/idm.json``."""
    value = cfg.get("regimes", {}).get("v0")
    if value is not None:
        return float(value)
    path = Path(root) / "idm.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        params = data.get("params", data)
        if "v0" in params:
            return float(params["v0"])
    warnings.warn(f"no v0 (config regimes.v0 is null and {path} has none): regime tables are skipped",
                  RuntimeWarning, stacklevel=2)  # fmt: skip
    return None


def _seed(base: int, *keys: object) -> int:
    """Deterministic bootstrap seed of a statistic (independent streams for different keys)."""
    text = "\x1f".join(str(k) for k in keys).encode("utf-8")
    return int(np.random.SeedSequence([int(base) & 0xFFFFFFFF, zlib.crc32(text)]).generate_state(1)[0])


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def holm_replicates(n_boot: int, n_tests: int, alpha: float = ALPHA, factor: float = 20.0) -> int:
    """Bootstrap replicates for p-values of a Holm family of ``n_tests`` hypotheses at level ``alpha``.

    ``max(n_boot, ceil(factor * n_tests / alpha))``: the smallest count-based p-value 2 / (B + 1) then lies
    ``factor / 2`` times below the smallest Holm threshold ``alpha / n_tests``.
    """
    return max(int(n_boot), int(np.ceil(factor * max(int(n_tests), 1) / alpha)))


# --------------------------------------------------------------------------------- windows
@dataclass
class _View:
    """Arrays of one data set or subset: y (n,) float64, s, dv (n,), bins (n,), labels (n,) | None, units (n,) (the
    bootstrap units), events and drivers (n,) (the event and follower codes, whatever the unit; None if absent)."""

    y: np.ndarray
    s: np.ndarray
    dv: np.ndarray
    bins: np.ndarray
    labels: np.ndarray | None
    units: np.ndarray
    events: np.ndarray | None = None
    drivers: np.ndarray | None = None


@dataclass
class _Windows:
    name: str
    source: str
    n: int
    full: _View
    views: list[tuple[str, np.ndarray | None]]
    signal: str = "raw"  # signal variant of the windows (spec.signal of the meta; design.md section 12)
    sources: tuple[str, ...] = ()  # data sets of the windows: (source,), or the members' data of a pool
    members: tuple[str, ...] = ()  # member data sets of a pool (design.md 16.1), else empty

    def view(self, rows: np.ndarray | None) -> _View:
        if rows is None:
            return self.full
        f = self.full
        return _View(
            f.y[rows], f.s[rows], f.dv[rows], f.bins[rows], None if f.labels is None else f.labels[rows], f.units[rows],
            None if f.events is None else f.events[rows], None if f.drivers is None else f.drivers[rows],
        )


def _load_windows(
    root: Path, name: str, grid: StateGrid, th: RegimeThresholds, v0: float | None, priority: Sequence[str], unit: str
) -> _Windows:
    frame, meta = io.load_windows(root, name)
    n = len(frame)
    needed = ["s", "dv", "y", unit]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError(f"windows/{name}: missing columns {missing}")
    s, dv = frame["s"].to_numpy(), frame["dv"].to_numpy()
    labels = None
    if v0 is not None and "v" in frame.columns and "a_prev" in frame.columns:
        labels = label_regimes(s, dv, frame["v"].to_numpy(), frame["a_prev"].to_numpy(), v0, th, priority)
    full = _View(
        y=frame["y"].to_numpy(dtype=np.float64),
        s=np.asarray(s),
        dv=np.asarray(dv),
        bins=grid.bin_index(s, dv),
        labels=labels,
        units=frame[unit].to_numpy(),
        events=frame["event"].to_numpy() if "event" in frame.columns else None,
        drivers=frame["driver"].to_numpy() if "driver" in frame.columns else None,
    )
    views: list[tuple[str, np.ndarray | None]] = [("all", None)]
    if "subset" in frame.columns and n:
        cat = pd.Categorical(frame["subset"])
        codes = np.asarray(cat.codes, dtype=np.int64)
        present = np.bincount(codes[codes >= 0], minlength=len(cat.categories)) > 0
        names = [str(c) for c in cat.categories]
        if present.sum() > 1:
            for i, sub in enumerate(names):
                if present[i]:
                    views.append(("unlabelled" if sub == "all" else sub, np.flatnonzero(codes == i)))
    spec = meta.get("spec") if isinstance(meta, Mapping) else None
    data = spec.get("data", name) if isinstance(spec, Mapping) else name
    sources = tuple(str(x) for x in data) if isinstance(data, (list, tuple)) else (str(data),)
    members = tuple(str(m) for m in spec.get("members") or ()) if isinstance(spec, Mapping) else ()
    source = ",".join(sources)
    signal = str(spec.get("signal", "raw")) if isinstance(spec, Mapping) else "raw"
    return _Windows(name=name, source=source, n=n, full=full, views=views, signal=signal, sources=sources,
                    members=members)  # fmt: skip


def stored_train_data(root: str | Path) -> list[str] | None:
    """Training data sets of a pooled root (``pool.data`` of ``root/train_density.npz``, design.md 16.1); None when
    the file has no pool (one training data set) or does not exist."""
    path = Path(root) / "train_density.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        pool = json.loads(str(z["pool"])) if "pool" in z.files else None
    return None if not pool else [str(x) for x in pool.get("data") or []]


def pair_roles(
    windows: Mapping[str, "_Windows"], in_domain: str, train_data: Sequence[str] | None = None,
    placebo: Sequence[str] | None = None, cross_check: Sequence[str] = (), cross_check_of: Mapping[str, str] | None = None,
) -> dict[str, str | None]:  # fmt: skip
    """Role of every data set against the in-domain set: None (the in-domain set, a member of an in-domain pool, a
    data set of a training set that is not an in-domain source, or a cross-check of a training set), ``"placebo"``
    (another split of an in-domain source; ``placebo`` overrides the automatic choice), ``"cross_check"`` or
    ``"shift"`` (module docstring). ``windows`` must hold the in-domain set (else every other data set is a shift
    pair, as without a reference)."""
    ref = windows.get(in_domain)
    in_sources = set(ref.sources) if ref is not None else set()
    parts = set(ref.members) if ref is not None else set()
    trained = set(train_data) if train_data is not None else set(in_sources)
    if placebo is None:
        placebos = {d for d, w in windows.items() if d != in_domain and d not in parts and set(w.sources) & in_sources}
    else:
        placebos = {str(d) for d in placebo}
    repeats = {str(k): str(v) for k, v in (cross_check_of or {}).items()}
    crosses = {str(d) for d in cross_check} - placebos
    excluded = set(parts)
    excluded |= {d for d, w in windows.items() if d not in placebos and set(w.sources) & trained and not set(w.sources) & in_sources}
    trained_names = trained | {t for t, w in windows.items() if set(w.sources) & trained}  # data sets and their tables
    excluded |= {d for d in crosses if repeats.get(d) in trained_names}
    roles: dict[str, str | None] = {}
    for d in windows:
        if d == in_domain or d in excluded - placebos:
            roles[d] = None
        else:
            roles[d] = "placebo" if d in placebos else "cross_check" if d in crosses else "shift"
    return roles


def _load_stack(root: Path, model: str, seeds: Sequence[int], dataset: str, n: int) -> SeedStack:
    frames = []
    for seed in seeds:
        frame = io.load_preds(root, model, seed, dataset)
        if len(frame) != n:
            raise ValueError(f"preds/{model}/seed{seed}/{dataset}: {len(frame)} rows, the windows table has {n}")
        frames.append(frame)
    return SeedStack.from_frames(frames, seeds)


# ------------------------------------------------------------------------------ table rows
def _mean_of(values: np.ndarray, rows: np.ndarray) -> float:
    v = np.asarray(values[rows], dtype=np.float64)
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else float("nan")


def _n_units(units: np.ndarray) -> int:
    return int(len(encode_units(units)[1]))


def _ks_statistic(pit: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance of PIT values from U(0, 1): the statistic of ``metrics.ks_uniform``.

    Its p-value is not needed (it is 0 for any real model at 1e5 windows) and costs ~0.1 s per call there.
    """
    p = np.sort(np.asarray(pit, dtype=np.float64))
    n = p.size
    if n == 0:
        return float("nan")
    i = np.arange(1.0, n + 1.0)
    return float(max(np.max(i / n - p), np.max(p - (i - 1.0) / n)))


def _one_step_rows(
    keys: dict[str, Any], sv: SeedStack, wv: _View, n_boot: int, seed: int, ci: float
) -> list[dict[str, Any]]:
    """Per-seed one-step metrics on the evaluated rows and their seed mean with event-bootstrap CIs."""
    valid = sv.valid
    own = own_decomposition(sv)
    sq = (wv.y[None, :] - sv["mean"].astype(np.float64)) ** 2
    columns = {
        "nll": sv["nll"], "crps": sv["crps"], "picp90": sv["covered"], "width90": sv["width"],
        "var": own["total"], "aleatoric": own["aleatoric"], "epistemic": own["epistemic"],
    }  # fmt: skip
    rows = []
    for k, seed_k in enumerate(sv.seeds):
        v = valid[k]
        row: dict[str, Any] = {**keys, "seed": str(seed_k), "n_seeds": 1, "n_windows": int(v.sum())}
        row["n_events"] = _n_units((wv.events if wv.events is not None else wv.units)[v])
        row["n_followers"] = _n_units(wv.drivers[v]) if wv.drivers is not None else float("nan")
        for name, arr in columns.items():
            row[name] = _mean_of(arr[k], v)
        row["rmse"] = float(np.sqrt(_mean_of(sq[k], v)))
        pit = np.asarray(sv["pit"][k][v], dtype=np.float64)
        pit = pit[np.isfinite(pit)]
        row["pit_ks"] = _ks_statistic(pit)
        row["calibration_error"] = calibration_error(pit) if pit.size else float("nan")
        rows.append(row)
    any_valid = valid.any(axis=0)
    mean_row: dict[str, Any] = {**keys, "seed": "mean", "n_seeds": sv.n_seeds, "n_windows": int(any_valid.sum())}
    mean_row["n_events"] = _n_units((wv.events if wv.events is not None else wv.units)[any_valid])
    mean_row["n_followers"] = _n_units(wv.drivers[any_valid]) if wv.drivers is not None else float("nan")
    for name in _METRICS:
        values = np.array([r[name] for r in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        mean_row[name] = float(values.mean()) if values.size else float("nan")
    boot = bootstrap_seed_means(
        {"nll": sv["nll"], "crps": sv["crps"], "picp90": sv["covered"]}, wv.units, n_boot, seed, ci
    )
    for name, res in boot.items():
        mean_row[f"{name}_lo"], mean_row[f"{name}_hi"] = res["lo"], res["hi"]
    rows.append(mean_row)
    return rows


def _concentration_rows(
    keys: dict[str, Any], q: Mapping[str, np.ndarray], wv: _View, rare: Mapping[str, np.ndarray],
    regions: Mapping[str, Any], with_epistemic: bool, n_boot: int, seed: int, ci: float,
) -> list[dict[str, Any]]:  # fmt: skip
    s64 = np.asarray(wv.s, dtype=np.float64)
    dv64 = np.asarray(wv.dv, dtype=np.float64)
    masks = {
        "rare_freq": rare["rare_freq"][wv.bins],
        "rare_mass": rare["rare_mass"][wv.bins],
        "small_gap": s64 < float(regions["small_gap"]["s_max"]),
        "large_dv": np.abs(dv64) > float(regions["large_dv"]["abs_dv_min"]),
    }
    xs = {"aleatoric": q["aleatoric"]}
    if with_epistemic:
        xs["epistemic"] = q["epistemic"]
    res = concentrations(xs, masks, wv.units, n_boot, seed, ci)
    nan = float("nan")
    rows = []
    for rule in RULES:
        alea = res[("aleatoric", rule)]
        epi = res.get(("epistemic", rule), {})
        rows.append(
            {
                **keys,
                "rule": rule,
                "share_windows": alea["share_windows"],
                "ratio_epistemic": epi.get("estimate", nan),
                "ratio_epistemic_lo": epi.get("lo", nan),
                "ratio_epistemic_hi": epi.get("hi", nan),
                "ratio_aleatoric": alea["estimate"],
                "ratio_aleatoric_lo": alea["lo"],
                "ratio_aleatoric_hi": alea["hi"],
                "share_mass_epistemic": epi.get("share_mass", nan),
                "share_mass_aleatoric": alea["share_mass"],
                "n_events": alea["n_events"],
                "n_seeds": q["aleatoric"].shape[0],
                "conc_mass_epistemic": epi.get("mass_ratio", nan),
                "conc_mass_epistemic_lo": epi.get("mass_ratio_lo", nan),
                "conc_mass_epistemic_hi": epi.get("mass_ratio_hi", nan),
                "conc_mass_aleatoric": alea["mass_ratio"],
                "conc_mass_aleatoric_lo": alea["mass_ratio_lo"],
                "conc_mass_aleatoric_hi": alea["mass_ratio_hi"],
            }
        )
    return rows


@dataclass
class _Shift:
    """In-domain reference of the shift ratios of one model."""

    stack: SeedStack
    windows: _Windows
    cache: dict[tuple[str, tuple[int, ...]], dict[str, np.ndarray]]

    def quantities(self, kind: str, seeds: tuple[int, ...]) -> dict[str, np.ndarray]:
        key = (kind, seeds)
        if key not in self.cache:
            pos = [self.stack.seeds.index(s) for s in seeds]
            self.cache[key] = quantity_arrays(self.stack.select(pos), self.windows.full.y, kind)
        return self.cache[key]


def _shift_rows(
    keys: dict[str, Any], kind: str, sv: SeedStack, q_view: Mapping[str, np.ndarray], wv: _View, ref: _Shift,
    with_epistemic: bool, grid: StateGrid, min_count: int, h31: H31Thresholds, n_boot: int, seed: int, ci: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:  # fmt: skip
    common = tuple(s for s in sv.seeds if s in ref.stack.seeds)
    if len(common) < (2 if kind == "seed_ensemble" else 1):
        return [], []
    pos = [sv.seeds.index(s) for s in common]
    q_d = q_view if pos == list(range(sv.n_seeds)) else quantity_arrays(sv.select(pos), wv.y, kind)
    q_i = ref.quantities(kind, common)
    names = [n for n in SHIFT_QUANTITIES if n != "epistemic" or with_epistemic]
    xs_d = {n: q_d[n] for n in names}
    xs_i = {n: q_i[n] for n in names}
    ref_view = ref.windows.full
    plain = shift_ratios(xs_d, wv.units, xs_i, ref_view.units, n_boot, seed, ci)
    matched = matched_shift_ratios(
        xs_d, wv.bins, wv.units, xs_i, ref_view.bins, ref_view.units, grid, min_count, n_boot, seed, ci
    )
    rows = []
    for name in names:
        p, m = plain[name], matched[name]
        rows.append(
            {
                **keys,
                "kind": kind,
                "quantity": name,
                "estimate": p["estimate"],
                "lo": p["lo"],
                "hi": p["hi"],
                "se": p["se"],
                "seed_sd": p["seed_sd"],
                "matched_estimate": m["estimate"],
                "matched_lo": m["lo"],
                "matched_hi": m["hi"],
                "n_bins_matched": m["n_bins_matched"],
                "mass_in": m["mass_in"],
                "mass_shift": m["mass_shift"],
                "n_events": p["n_events_shift"],
                "n_events_in": p["n_events_in"],
                "n_seeds": len(common),
                "mean_shift": p["mean_shift"],
                "mean_in": p["mean_in"],
            }
        )
    verdicts = []
    epi_in = plain["epistemic"]["mean_in"] if "epistemic" in names else float("nan")
    if np.isfinite(epi_in) and epi_in > 0:  # no verdict on a degenerate epistemic part (identical seeds: 0 / 0)
        v_plain = h31_verdict(plain["epistemic"], plain["aleatoric"], h31)
        v_matched = h31_verdict(matched["epistemic"], matched["aleatoric"], h31)
        row: dict[str, Any] = {**keys, "kind": kind}
        for tag, res in (("", plain), ("_matched", matched)):
            for short, name in (("epi", "epistemic"), ("alea", "aleatoric")):
                row[f"r_{short}{tag}"] = res[name]["estimate"]
                row[f"r_{short}{tag}_lo"] = res[name]["lo"]
                row[f"r_{short}{tag}_hi"] = res[name]["hi"]
        row.update(verdict_ci=v_plain["verdict_ci"], verdict_point=v_plain["verdict_point"])
        row.update(verdict_ci_matched=v_matched["verdict_ci"], verdict_point_matched=v_matched["verdict_point"])
        row["n_seeds"] = len(common)
        verdicts.append(row)
    return rows, verdicts


def h31e_tables(
    rare_concentration: pd.DataFrame,
    h31_verdicts: pd.DataFrame,
    cfg: "Mapping[str, Any] | H31eThresholds | None" = None,
    in_domain: str = "highd",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(h31e_verdicts, h31e_summary)`` of restated hypothesis H3.1e from two analysis tables (design.md, 12).

    Per model and kind of every ``h31_verdicts`` row (verdict rows exist only for a non-degenerate epistemic part:
    kind ``"model"`` of models with an own epistemic output, kind ``"seed_ensemble"`` with an in-domain across-seed
    variance > 0): the concentration is the ``rare_concentration`` row of the in-domain data set (subset ``"all"``)
    and the rule of ``H31eThresholds`` (``rare_mass``), the growth the epistemic shift ratio of every pair
    (:func:`cf_unc.maps.shift.h31e_verdict`); the summary counts as :func:`cf_unc.maps.shift.h31e_summary`. cfg:
    maps config (sections ``h31`` and ``h31e``) or thresholds. Works on the CSV tables of an analysed root too.
    """
    th = H31eThresholds.from_config(cfg)
    rows: list[dict[str, Any]] = []
    if len(h31_verdicts) and len(rare_concentration):
        rc = rare_concentration
        in_ref = (rc["dataset"].astype(str) == in_domain) & (rc["subset"].astype(str) == "all")
        conc = rc[in_ref & (rc["rule"].astype(str) == th.rule)]
        for (model, kind), grp in h31_verdicts.groupby(["model", "kind"], sort=False):
            match = conc[(conc["model"].astype(str) == str(model)) & (conc["kind"].astype(str) == str(kind))]
            rows += h31e_verdict(match.iloc[0].to_dict() if len(match) else None, grp.to_dict("records"), th)
    verdicts = _frame(rows, "h31e_verdicts")
    return verdicts, h31e_summary(verdicts, min_pairs=th.min_pairs)


def _frame(rows: list[dict[str, Any]] | list[pd.DataFrame], name: str) -> pd.DataFrame:
    columns = TABLE_COLUMNS[name]
    if rows and isinstance(rows[0], pd.DataFrame):
        frame = pd.concat(rows, ignore_index=True)
    else:
        frame = pd.DataFrame(rows)
    for col in columns:
        if col not in frame.columns:
            frame[col] = pd.Series(dtype=object) if frame.empty else np.nan
    return frame[columns]


# ---------------------------------------------------------------------------------- driver
def analyze_e1(
    root: str | Path = "artifacts/e1",
    cfg: "str | Path | Mapping[str, Any] | None" = None,
    in_domain: str = "highd",
    models: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    *,
    unit: str | None = None,
    placebo: Sequence[str] | None = None,
    cross_check: Sequence[str] | None = None,
    train_data: Sequence[str] | None = None,
    write: bool = True,
    verbose: bool = False,
    h33_pairs: Sequence[Sequence[str]] | None = None,
    tables_dir: str = "tables",
) -> dict[str, pd.DataFrame]:
    """Analyse the E1 prediction tables under ``root`` and write the tables and maps (module docstring).

    cfg: maps config (path, mapping merged over ``configs/maps/default.yaml``, or None); in_domain: data set
    the shifted sets are compared with (always loaded as the reference, even when ``datasets`` omits it);
    models / datasets: subsets to analyse (default: everything available); unit: ``"event"`` or
    ``"driver"`` (default: ``bootstrap.unit``); placebo: data sets labelled ``pair = "placebo"`` (default:
    those whose source equals the in-domain source; ``[]``: none); cross_check: data sets labelled
    ``pair = "cross_check"`` (second extraction of a counted data set; default: ``cross_check`` of the
    config), reported but not counted for H3.1; train_data: training data sets of the root (default: ``pool.data``
    of ``train_density.npz`` for a pooled root, else the in-domain sources), whose data sets get no shift pair
    (``pair_roles``). h33_pairs: the (heteroscedastic, homoscedastic) model pairs of the H3.3 NLL table (default
    ``[H33_MODELS]``; every pair's rows in one table with one Holm family, the bootstrap seeds of a data set and subset
    shared by the pairs). tables_dir: directory below ``root`` for the tables (the maps are written only with the
    default ``tables``: they do not depend on the bootstrap unit). Returns the tables by name (``TABLE_COLUMNS`` plus
    ``maps_long``).
    """
    started = time.perf_counter()
    root = Path(root)
    conf = load_config(cfg)
    boot = conf["bootstrap"]
    n_boot, ci, base_seed = int(boot["n_boot"]), float(boot["ci"]), int(boot["seed"])
    unit = str(unit or boot.get("unit", "event"))
    if unit not in ("event", "driver"):
        raise ValueError(f"unit must be 'event' or 'driver', got {unit!r}")
    grid = StateGrid.from_config(conf["grid"])
    th = RegimeThresholds.from_config(conf["regimes"])
    priority = [str(p) for p in conf["regimes"].get("priority", PRIORITY)]
    reference = str(conf["regimes"].get("reference", "steady"))
    h31 = H31Thresholds.from_config(conf["h31"])
    min_count = int(conf["matched"]["min_count"])
    n_boot_order = holm_replicates(n_boot, 2 * (len(REGIMES) - 1))  # family: var and mse x regimes vs reference
    say: Callable[[str], None] = (lambda msg: print(msg, flush=True)) if verbose else (lambda msg: None)

    avail = io.available(root)
    if avail.empty:
        raise FileNotFoundError(f"no prediction tables under {root / 'preds'}")
    model_names = sorted(avail["model"].unique())
    if models is not None:
        absent = [m for m in models if m not in model_names]
        if absent:
            warnings.warn(f"models without prediction tables: {absent}", RuntimeWarning, stacklevel=2)
        model_names = [m for m in models if m in model_names]
    ds_avail = sorted(avail["dataset"].unique())
    report = list(ds_avail)
    if datasets is not None:
        absent = [d for d in datasets if d not in ds_avail]
        if absent:
            warnings.warn(f"data sets without prediction tables: {absent}", RuntimeWarning, stacklevel=2)
        report = [d for d in datasets if d in ds_avail]
    has_ref = in_domain in ds_avail
    if not has_ref:
        warnings.warn(f"in-domain data set {in_domain!r} has no prediction tables: no shift ratios", RuntimeWarning,
                      stacklevel=2)  # fmt: skip
    order = ([in_domain] if has_ref else []) + [d for d in report if d != in_domain]

    train_counts = load_train_density(root, grid)
    rare = rare_flags(train_counts, float(conf["rare"]["freq_threshold"]), float(conf["rare"]["mass_threshold"]))
    v0 = resolve_v0(root, conf)
    windows = {d: _load_windows(root, d, grid, th, v0, priority, unit) for d in order}
    trained = [str(d) for d in train_data] if train_data is not None else stored_train_data(root)
    roles = pair_roles(
        windows, in_domain, trained, placebo, conf.get("cross_check", []) if cross_check is None else cross_check,
        conf.get("cross_check_of") or {},
    )  # fmt: skip
    placebos = {d for d, r in roles.items() if r == "placebo"}
    crosses = {d for d, r in roles.items() if r == "cross_check"}
    no_pair = sorted(d for d, r in roles.items() if r is None and d != in_domain)
    say(f"analyze_e1: {len(model_names)} models, data sets {order} (placebo {sorted(placebos)}, cross-check {sorted(crosses)}"
        f"{f', no pair (training domain) {no_pair}' if no_pair else ''}), unit {unit}, n_boot {n_boot}")  # fmt: skip

    rows: dict[str, list[Any]] = {name: [] for name in [*TABLE_COLUMNS, "maps_long"]}
    seeds_info: dict[str, dict[str, list[int]]] = {}
    pairs33 = [tuple(str(m) for m in pair) for pair in (h33_pairs or [H33_MODELS])]
    if any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs33):
        raise ValueError(f"h33_pairs must hold pairs of two different models, got {pairs33}")
    h33: dict[str, dict[str, tuple[np.ndarray, int]]] = {m: {} for pair in pairs33 for m in pair}
    maps_dir = root / "maps"

    for model in model_names:
        sel = avail[avail["model"] == model]
        seeds_by_ds = {d: sorted(int(s) for s in sel.loc[sel["dataset"] == d, "seed"]) for d in order}
        seeds_info[model] = {d: s for d, s in seeds_by_ds.items() if s}
        ref: _Shift | None = None
        if has_ref and seeds_by_ds.get(in_domain):
            in_stack = _load_stack(root, model, seeds_by_ds[in_domain], in_domain, windows[in_domain].n)
            ref = _Shift(in_stack, windows[in_domain], {})
        with_epi: bool | None = None
        for d in order:
            seeds = seeds_by_ds.get(d, [])
            if not seeds or d not in report:
                continue
            t0 = time.perf_counter()
            win = windows[d]
            stack = ref.stack if (d == in_domain and ref is not None) else _load_stack(root, model, seeds, d, win.n)
            if with_epi is None:
                with_epi = has_epistemic(stack["epistemic_var"])
            kinds = ["model"] + (["seed_ensemble"] if stack.n_seeds >= 2 else [])
            pair = roles.get(d)
            for subset, sub_rows in win.views:
                sv, wv = stack.take(sub_rows), win.view(sub_rows)
                keys = {"model": model, "dataset": d, "subset": subset}
                tag = (d, subset)
                rows["one_step_metrics"] += [
                    {**r, "unit": unit}
                    for r in _one_step_rows(keys, sv, wv, n_boot, _seed(base_seed, "metrics", *tag), ci)
                ]
                for kind in kinds:
                    q = quantity_arrays(sv, wv.y, kind)
                    epi = kind == "seed_ensemble" or bool(with_epi)
                    info = {**keys, "kind": kind}
                    umap = map_from_arrays(
                        wv.bins, q, sv.valid.any(axis=0), grid, train_counts, rare=rare, n_seeds=sv.n_seeds, meta=info
                    )
                    if write and tables_dir == "tables":
                        stem = d if subset == "all" else f"{d}__{_safe(subset)}"
                        umap.save(maps_dir / f"{_safe(model)}__{kind}" / f"{stem}.npz")
                    long = umap.to_frame()
                    for i, key in enumerate(MAP_KEYS):
                        long.insert(i, key, info[key])
                    rows["maps_long"].append(long)
                    rows["rare_concentration"] += [
                        {**r, "kind": kind, "unit": unit}
                        for r in _concentration_rows(
                            keys, q, wv, rare, conf["regions"], epi, n_boot, _seed(base_seed, "rare", *tag), ci
                        )
                    ]
                    if pair is not None and ref is not None:
                        shift, verdict = _shift_rows(
                            {**keys, "pair": pair}, kind, sv, q, wv, ref, epi, grid, min_count, h31, n_boot,
                            _seed(base_seed, "shift", *tag), ci,
                        )  # fmt: skip
                        rows["shift_ratios"] += [{**r, "unit": unit} for r in shift]
                        rows["h31_verdicts"] += [{**r, "unit": unit} for r in verdict]
                if wv.labels is not None:
                    table = regime_table(
                        {"y": wv.y, "units": wv.units}, sv, wv.labels, n_boot, _seed(base_seed, "regimes", *tag), ci,
                        unit="units",
                    )  # fmt: skip
                    for i, key in enumerate(("model", "dataset", "subset")):
                        table.insert(i, key, keys[key])
                    rows["regime_stats"].append(table.assign(unit=unit))
                    sq = (wv.y[None, :] - sv["mean"].astype(np.float64)) ** 2
                    ordering = regime_ordering(
                        {"var": sv["var"], "mse": sq}, wv.labels, wv.units, reference, n_boot_order,
                        _seed(base_seed, "ordering", *tag), ci, alpha=ALPHA,
                    )  # fmt: skip
                    for i, key in enumerate(("model", "dataset", "subset")):
                        ordering.insert(i, key, keys[key])
                    rows["regime_ordering"].append(ordering.assign(unit=unit))
            if model in h33:
                nll = stack["nll"].astype(np.float64)
                ok = ~np.isnan(nll)
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean_nll = np.where(ok, nll, 0.0).sum(axis=0) / ok.sum(axis=0)
                h33[model][d] = (np.where(ok.any(axis=0), mean_nll, np.nan), stack.n_seeds)
            say(f"  {model} / {d}: {stack.n_seeds} seeds, {win.n:,} windows, {time.perf_counter() - t0:.1f} s")
            del stack

    h33_rows, n_boot_h33 = _h33_rows(h33, windows, [d for d in order if d in report], n_boot, base_seed, ci, unit,
                                     pairs33)  # fmt: skip
    derived = ("h31_summary", "h31e_verdicts", "h31e_summary", "h33_nll")
    tables = {name: _frame(rows[name], name) for name in TABLE_COLUMNS if name not in derived}
    tables["h33_nll"] = h33_rows
    tables["h31_summary"] = h31_summary(tables["h31_verdicts"], min_pairs=h31.min_pairs)
    tables["h31e_verdicts"], tables["h31e_summary"] = h31e_tables(
        tables["rare_concentration"], tables["h31_verdicts"], conf, in_domain
    )
    maps_long = pd.concat(rows["maps_long"], ignore_index=True) if rows["maps_long"] else pd.DataFrame(columns=MAP_KEYS)
    tables["maps_long"] = maps_long
    elapsed = time.perf_counter() - started
    if write:
        out_dir = root / tables_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_csv(out_dir / f"{name}.csv", index=False)
        info = {
            "root": str(root),
            "in_domain": in_domain,
            "datasets": order,
            "sources": {d: w.source for d, w in windows.items()},
            "signals": {d: w.signal for d, w in windows.items()},
            "placebo": sorted(placebos),
            "cross_check": sorted(crosses),
            "shift": sorted(d for d, r in roles.items() if r == "shift"),
            "no_pair": no_pair,
            "train_data": trained,
            "in_domain_members": list(windows[in_domain].members) if has_ref else [],
            "models": model_names,
            "seeds": seeds_info,
            "unit": unit,
            "v0": v0,
            "n_boot": n_boot,
            "n_boot_tests": {"regime_ordering": n_boot_order, "h33_nll": n_boot_h33},
            **({} if pairs33 == [H33_MODELS] else {"h33_pairs": [list(p) for p in pairs33]}),
            "config": conf,
            "seconds": round(elapsed, 2),
        }
        (out_dir / "analysis_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    say(f"analyze_e1: done in {elapsed:.1f} s")
    return tables


def _h33_rows(
    h33: Mapping[str, Mapping[str, tuple[np.ndarray, int]]], windows: Mapping[str, _Windows], datasets: Sequence[str],
    n_boot: int, base_seed: int, ci: float, unit: str, pairs: Sequence[Sequence[str]] = (H33_MODELS,),
) -> tuple[pd.DataFrame, int]:  # fmt: skip
    """NLL of the heteroscedastic against the homoscedastic model of every pair (default ``H33_MODELS``) per data
    set, subset and regime; Holm over all rows of all pairs. Returns the table and the number of bootstrap replicates
    used (:func:`holm_replicates` of the family)."""
    common = {}
    for name_a, name_b in pairs:
        common[(name_a, name_b)] = [d for d in datasets if d in h33[name_a] and d in h33[name_b]]
        if not common[(name_a, name_b)]:
            absent = [m for m in (name_a, name_b) if not h33[m]]
            warnings.warn(f"H3.3 NLL comparison skipped: no data set with predictions of both {(name_a, name_b)}"
                          + (f" (missing: {absent})" if absent else ""), RuntimeWarning, stacklevel=3)  # fmt: skip
    if not any(common.values()):
        return _frame([], "h33_nll"), int(n_boot)
    family = sum(
        len(windows[d].views) * (1 + (len(REGIMES) if windows[d].full.labels is not None else 0))
        for ds in common.values() for d in ds
    )
    n_boot_h33 = holm_replicates(n_boot, family)
    frames = []
    for (name_a, name_b), ds in common.items():
        for d in ds:
            (nll_a, k_a), (nll_b, k_b) = h33[name_a][d], h33[name_b][d]
            win = windows[d]
            for subset, sub_rows in win.views:
                wv = win.view(sub_rows)
                a = nll_a if sub_rows is None else nll_a[sub_rows]
                b = nll_b if sub_rows is None else nll_b[sub_rows]
                res = nll_comparison(a, b, wv.units, wv.labels, n_boot_h33, _seed(base_seed, "h33", d, subset), ci)
                res = res.assign(n_seeds_a=k_a, n_seeds_b=k_b, unit=unit)
                keys = (("dataset", d), ("subset", subset), ("model_a", name_a), ("model_b", name_b))
                for i, (key, value) in enumerate(keys):
                    res.insert(i, key, value)
                frames.append(res)
    frame = pd.concat(frames, ignore_index=True)
    adjusted, reject = holm(frame["pvalue"].to_numpy(dtype=np.float64), alpha=ALPHA)
    frame["p_holm"] = adjusted
    frame["reject"] = reject
    return frame[TABLE_COLUMNS["h33_nll"]], n_boot_h33


# ---------------------------------------------------------------------------------- report
def _fmt(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return "-" if not np.isfinite(value) else f"{value:.4g}"
    return str(value)


def format_report(tables: Mapping[str, pd.DataFrame]) -> str:
    """Plain-text H3.1 summary (with the verdicts per pair) and the H3.3 NLL table."""
    parts = ["H3.1 summary (whole data sets; CI-based and point verdicts; placebo pairs listed, not counted)"]
    summary = tables.get("h31_summary")
    parts.append(summary.to_string(index=False) if summary is not None and len(summary) else "  (no verdicts)")
    verdicts = tables.get("h31_verdicts")
    if verdicts is not None and len(verdicts):
        cols = ["model", "kind", "dataset", "subset", "pair", "r_epi", "r_epi_lo", "r_epi_hi", "r_alea", "r_alea_lo",
                "r_alea_hi", "verdict_ci", "verdict_point"]  # fmt: skip
        parts += ["", "H3.1 verdicts per pair", verdicts[cols].to_string(index=False, float_format=_fmt)]
    summary_e = tables.get("h31e_summary")
    if summary_e is not None and len(summary_e):
        cols = ["model", "kind", "main", "conc", "conc_lo", "conc_hi", "verdict_conc_ci", "n_pairs", "n_supported",
                "n_refuted", "n_inconclusive", "overall", "overall_point", "placebo_verdicts", "cross_check_verdicts"]  # fmt: skip
        parts += ["", "H3.1e summary (epistemic part only: concentration in rare_mass states on the in-domain test set, "
                  "growth under shift; main verdict: kind model)", summary_e[cols].to_string(index=False, float_format=_fmt)]  # fmt: skip
    h33 = tables.get("h33_nll")
    cols = ["dataset", "subset", "group", "n_windows", "nll_a", "nll_b", "diff", "diff_lo", "diff_hi", "p_holm",
            "ci_overlap"]  # fmt: skip
    pairs = list(dict.fromkeys(zip(h33["model_a"], h33["model_b"]))) if h33 is not None and len(h33) else [H33_MODELS]
    for name_a, name_b in pairs:
        title = f"H3.3 NLL, {name_a} minus {name_b} (negative: heteroscedastic better; Holm over the table)"
        parts += ["", title]
        if h33 is not None and len(h33):
            sel = h33[(h33["model_a"] == name_a) & (h33["model_b"] == name_b)]
            parts.append(sel[cols].to_string(index=False, float_format=_fmt))
        else:
            parts.append("  (not available)")
    return "\n".join(parts)
