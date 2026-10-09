"""Tables, figures and index of the final report (milestone M5, docs/design.md section 14.2).

    python scripts/make_report.py                                   # -> artifacts/report
    python scripts/make_report.py --out artifacts/report --roots e3=artifacts/e3 e3_vcap=artifacts/e3_vcap

Reads only the tables and figures that the analysis scripts wrote into the experiment roots (``artifacts/e1``,
``e1_mean1s``, ``e1h``, ``e1h_mean1s``, ``e2``, ``e2h``, ``e3``, ``e3_vcap``, ``e3_step``; the roots of experiment E5
with roughness-matched signals, ``e1m``, ``e1m_mean1s`` and ``e2m`` (design.md 15.1), only once they exist with the
tables of their analysis: they add the variant ``matched`` to T1, T2, T4 and F2-F4, and nothing changes while they are
absent; the same holds for the roots of experiment E6 with a pooled training domain (design.md 16.1), ``e6_pool``,
``e6_lodo_ngsim``, ``e6_lodo_openacc``, ``e6_lodo_ngsim_all`` (T1, T2, F2, F3; in-domain reference ``pool_test``) and ``e2_pool``,
``e2_lodo_ngsim``, ``e2_lodo_openacc``, ``e2_lodo_ngsim_all`` (T4, F4), which add the variants ``pool``,
``lodo_ngsim``, ``lodo_openacc``, ``lodo_ngsim_all``); the E3-pooled variants ``pool`` and ``lodo_openacc`` of the
main E3 root (design.md 17.1: the platoons with the checkpoints of ``e6_pool`` / ``e6_lodo_openacc``) join T5, T6 and
F5-F7 once the E3 analysis has written them (labels of ``cf_unc.report.figures_e3``);
it runs no experiment, no training and no bootstrap and needs no GPU. Writes into ``--out``:

* ``tables/<name>.csv`` (the numbers) and ``tables/<name>.tex`` (booktabs; ``longtable`` above 35 rows; 2-3
  significant digits; model labels of ``MODEL_LABEL`` / ``MODEL_LABEL_E3``): T1 one-step metrics, T2 H3.1 / H3.1e,
  T3 H3.3, T4 H3.4, T5 H3.2 per driver mode, T6 platoon diagnostics;
* ``figures/``: copies of the root figures named ``f<k>_<figure>__<root>.png`` (F1 maps and the regime-variance
  figures of F3 as one directory per root);
* ``index.md``: every table and figure with a one-line caption and the artifact it was built from.

Conventions: rows with a ``subset`` column use ``all``; one-step metrics are mean +- sd over the training seeds;
H3.2 uses the reference ``idm_white_noise``, the window ``response`` (profile + tail), the extended set of the main
run and the headline set of the two sensitivities (design.md 13.5); a driver mode with fewer than ``h32.min_profiles``
profiles (``<root>/config.yaml``) is listed as "not judged" (design.md 14.1). The script is idempotent: it rewrites
every output and removes outputs of earlier runs that it no longer produces (names ``t<k>_*`` and ``f<k>_*`` only).
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from cf_unc.report import figures_e3
from cf_unc.report.figures import MODEL_LABEL
from cf_unc.report.figures_e3 import MODEL_LABEL_E3

ROOT = Path(__file__).resolve().parents[1]
ROOT_KEYS = ("e1", "e1_mean1s", "e1h", "e1h_mean1s", "e2", "e2h", "e3", "e3_vcap", "e3_step")
MATCHED_KEYS = ("e1m", "e1m_mean1s", "e2m")  # E5 (design.md 15.1): optional roots, used once <root>/tables exists
POOLED_KEYS = ("e6_pool", "e6_lodo_ngsim", "e6_lodo_openacc", "e6_lodo_ngsim_all",
               "e2_pool", "e2_lodo_ngsim", "e2_lodo_openacc", "e2_lodo_ngsim_all")  # E6 (16.1, 16.4)
OPTIONAL_KEYS = (*MATCHED_KEYS, *POOLED_KEYS)  # used once <root>/tables exists
E1_ROOTS = {"e1": ("raw", "a"), "e1_mean1s": ("raw", "mean_1s"), "e1h": ("harmonised", "a"), "e1h_mean1s": ("harmonised", "mean_1s")}
E1_MATCHED = {"e1m": ("matched", "a"), "e1m_mean1s": ("matched", "mean_1s")}  # in T1, T2, F2, F3 (not in T3, F1)
E1_POOLED = {"e6_pool": ("pool", "a"), "e6_lodo_ngsim": ("lodo_ngsim", "a"), "e6_lodo_openacc": ("lodo_openacc", "a"),
             "e6_lodo_ngsim_all": ("lodo_ngsim_all", "a")}
E2_ROOTS = {"e2": "raw", "e2h": "harmonised"}
E2_MATCHED = {"e2m": "matched"}  # in T4, F4
E2_POOLED = {"e2_pool": "pool", "e2_lodo_ngsim": "lodo_ngsim", "e2_lodo_openacc": "lodo_openacc",
             "e2_lodo_ngsim_all": "lodo_ngsim_all"}  # in T4, F4
E3_SETS = {"e3": "extended", "e3_vcap": "headline", "e3_step": "headline"}  # profile set reported per E3 root
E3_RUN_LABEL = {"e3": "main run", "e3_vcap": "speed cap", "e3_step": "ensemble, per-step draws"}
ROLE = {
    "e1": "E1 one-step, target a, recorded signals", "e1_mean1s": "E1 one-step, target mean_1s, recorded signals",
    "e1h": "E1 one-step, target a, harmonised signals", "e1h_mean1s": "E1 one-step, target mean_1s, harmonised signals",
    "e2": "E2 rollouts, recorded signals", "e2h": "E2 rollouts, harmonised signals",
    "e3": "E3 platoons, main run (both signal variants)", "e3_vcap": "E3 sensitivity: follower speed cap",
    "e3_step": "E3 sensitivity: deep ensemble with per-step draws",
    "e1m": "E5: E1 one-step, target a, roughness-matched signals",
    "e1m_mean1s": "E5: E1 one-step, target mean_1s, roughness-matched signals",
    "e2m": "E5: E2 rollouts, roughness-matched signals",
    "e6_pool": "E6: E1 one-step, target a, matched signals, trained on the pooled domain (five data sets)",
    "e6_lodo_ngsim": "E6: E1 one-step, target a, matched signals, pooled domain without NGSIM I-80 (held out)",
    "e6_lodo_openacc": "E6: E1 one-step, target a, matched signals, pooled domain without OpenACC (held out)",
    "e6_lodo_ngsim_all": "E6: E1 one-step, target a, matched signals, pooled domain without both NGSIM sets (held out)",
    "e2_pool": "E6: E2 rollouts of e6_pool, target a", "e2_lodo_ngsim": "E6: E2 rollouts of e6_lodo_ngsim, target a",
    "e2_lodo_openacc": "E6: E2 rollouts of e6_lodo_openacc, target a",
    "e2_lodo_ngsim_all": "E6: E2 rollouts of e6_lodo_ngsim_all, target a",
}  # fmt: skip
DATASETS = ("pool_test", "highd", "highd_val", "ngsim", "ngsim_us101", "ngsim_fn", "waymo", "openacc")
DATASET_LABEL = {"pool_test": "pooled test folds", "highd": "highD", "highd_val": "highD validation", "ngsim": "NGSIM I-80",
                 "ngsim_us101": "NGSIM US-101", "ngsim_fn": "NGSIM I-80 (FollowNet)", "waymo": "Waymo", "openacc": "OpenACC"}  # fmt: skip
SHIFT_ORDER = ("ngsim", "ngsim_us101", "waymo", "openacc", "highd_val", "ngsim_fn")  # counted pairs, placebo, cross-check
ROLLOUT_DATASETS = ("highd", "ngsim", "ngsim_us101", "waymo", "openacc")  # E2 evaluation sets (highd_val calibrates)
VARIANT_LABEL = {"raw": "recorded signals", "harmonised": "harmonised signals", "matched": "roughness-matched signals",
                 "pool": "matched signals, pooled training domain", "lodo_ngsim": "matched signals, pooled domain without NGSIM I-80",
                 "lodo_openacc": "matched signals, pooled domain without OpenACC",
                 "lodo_ngsim_all": "matched signals, pooled domain without both NGSIM sets"}  # fmt: skip
E3_VARIANT_LABEL = {**VARIANT_LABEL, **{v: figures_e3.VARIANT_LABEL[v] for v in figures_e3.ONE_STEP_VARIANTS}}  # 17.1
TARGET_LABEL = {"a": "target a (one-step acceleration)", "mean_1s": "target mean_1s (1 s mean acceleration)"}
KIND_LABEL = {"model": "own", "seed_ensemble": "across seeds"}
REGIMES = ("congested", "braking", "approaching", "free", "steady", "other")
MODES = ("ACC", "Human", "all", "mixed")
HORIZONS = (1.0, 3.0, 5.0, 10.0)
E2_METHODS = (("raw", "raw"), ("static", "static"), ("aci_frozen", "ACI frozen"), ("aci_online", "ACI online"))
COLLISION_INDICES = (1, 2, 5, 10, 20)
MODEL_ORDER = list(MODEL_LABEL_E3)  # MODEL_LABEL order, then the E3-only labels
REFERENCE = "idm_white_noise"
LONG_ROWS = 35  # more body rows than this: longtable instead of a floating tabular
DASH = "–"


# ------------------------------------------------------------------------------------------------ model
@dataclass
class Table:
    """One report table: ``data`` -> ``<name>.csv`` (numbers), ``view`` (formatted cells, headers = column names)
    -> ``<name>.tex``. The first ``group_depth`` view columns form a group key: repeated values are blanked and a
    rule separates the groups."""

    name: str
    group: str
    caption: str
    sources: list[Path]
    data: pd.DataFrame
    view: pd.DataFrame
    group_depth: int = 0


@dataclass
class Figure:
    """A copied figure (``name`` ends in .png) or a directory of figures (``members``: file name -> caption)."""

    group: str
    name: str
    source: Path
    caption: str
    members: dict[str, str] = field(default_factory=dict)


@dataclass
class Context:
    """Roots by key, plus what was missing and what looked inconsistent in the sources."""

    roots: dict[str, Path]
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def file(self, key: str, *parts: str) -> Path | None:
        root = self.roots.get(key)
        path = root.joinpath(*parts) if root is not None else None
        if path is None or not path.exists():
            self.missing.append(f"{key}: {'/'.join(parts)}" + ("" if root is None else f" (under {display(root)})"))
            return None
        return path

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def check_finite(self, frame: pd.DataFrame, columns: Sequence[str], source: Path, what: str) -> None:
        """Note the columns of the rows a table uses that hold NaN or infinite values."""
        for c in columns:
            if c not in frame.columns:
                self.note(f"{display(source)}: column {c!r} missing ({what})")
                continue
            values = pd.to_numeric(frame[c], errors="coerce").to_numpy(np.float64)
            bad = int((~np.isfinite(values)).sum())
            if bad:
                self.note(f"{display(source)}: {bad} non-finite value(s) in {c!r} of the rows used for {what}")


def e1_roots(ctx: "Context", matched: bool = True) -> dict[str, tuple[str, str]]:
    """E1 roots of a table or figure: the four main roots and, with ``matched``, the E5 and E6 roots that exist."""
    extra = {**E1_MATCHED, **E1_POOLED} if matched else {}
    return {**E1_ROOTS, **{k: v for k, v in extra.items() if k in ctx.roots}}


def e2_roots(ctx: "Context") -> dict[str, str]:
    """E2 roots of T4 and F4: the two main roots and the E5 and E6 roots that exist."""
    return {**E2_ROOTS, **{k: v for k, v in {**E2_MATCHED, **E2_POOLED}.items() if k in ctx.roots}}


def in_domain_label(ctx: "Context", key: str) -> str:
    """The in-domain reference of an E1 root's shift ratios (``in_domain`` of its ``tables/analysis_info.json``)."""
    path = ctx.roots.get(key, Path("-")) / "tables" / "analysis_info.json"
    name = "highd"
    if path.exists():
        import json

        name = str(json.loads(path.read_text(encoding="utf-8")).get("in_domain") or "highd")
    return "highD test" if name == "highd" else DATASET_LABEL.get(name, name)


def display(path: Path) -> str:
    """A path relative to the repository root when it lies inside it (posix separators), else absolute."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def _resolve(path: str | Path) -> Path:
    """A relative path is taken from the working directory if it exists there, else from the repository root."""
    p = Path(path)
    return p if p.is_absolute() or p.exists() else ROOT / p


# -------------------------------------------------------------------------------------------- numbers
def _decimals(x: float, sig: int) -> int:
    """Decimals that show ``x`` with ``sig`` significant digits (negative: rounded to tens, hundreds, ...)."""
    if x is None or not np.isfinite(x) or x == 0:
        return sig - 1
    return sig - 1 - int(math.floor(math.log10(abs(x))))


def _sci(x: float) -> bool:
    """Values shown in exponent notation (fixed decimals would need more than 4 leading zeros or 8 digits)."""
    return x != 0 and (abs(x) < 1e-4 or abs(x) >= 1e7)


def _fixed(x: Any, d: int, sci: bool = True) -> str:
    """``x`` with ``d`` decimals (``d`` < 0: rounded to tens, hundreds, ...); exponent notation for extreme values
    when ``sci``; no negative zero."""
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return DASH
    x = float(x)
    if not np.isfinite(x):
        return "∞" if x > 0 else "-∞"
    if sci and _sci(x):
        return f"{x:.2e}"
    s = f"{round(x, d):.0f}" if d < 0 else f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def num(x: Any, sig: int = 3) -> str:
    """``x`` with ``sig`` significant digits."""
    return _fixed(x, _decimals(float(x), sig)) if x is not None and np.isfinite(float(x)) else _fixed(x, 0)


def _companions(est: float, values: Sequence[Any], d: int) -> list[str]:
    """Bounds or spread of an estimate, in the estimate's notation (fixed decimals ``d`` unless it is exponential)."""
    return [_fixed(v, d, sci=_sci(est)) for v in values]


def ci(est: Any, lo: Any, hi: Any, sig: int = 3, max_dec: int | None = None) -> str:
    """'estimate [lo, hi]': the estimate with ``sig`` significant digits (at most ``max_dec`` decimals), the bounds
    with the same decimals."""
    if est is None or not np.isfinite(float(est)):
        return DASH
    d = _decimals(float(est), sig) if max_dec is None else min(max_dec, _decimals(float(est), sig))
    if lo is None or hi is None or not (np.isfinite(float(lo)) and np.isfinite(float(hi))):
        return _fixed(est, d)
    a, b = _companions(float(est), (lo, hi), d)
    return f"{_fixed(est, d)} [{a}, {b}]"


def ci2(est: Any, lo: Any, hi: Any) -> str:
    """Interval of a reduction or an exponent: three significant digits, at most two decimals."""
    return ci(est, lo, hi, sig=3, max_dec=2)


def pm(mean: Any, sd: Any, sig: int = 3) -> str:
    """'mean ± sd' with the decimals of the mean at ``sig`` significant digits."""
    if mean is None or not np.isfinite(float(mean)):
        return DASH
    d = _decimals(float(mean), sig)
    if sd is None or not np.isfinite(float(sd)):
        return _fixed(mean, d)
    return f"{_fixed(mean, d)} ± {_companions(float(mean), (sd,), d)[0]}"


def prob(x: Any, d: int = 2) -> str:
    """A share or coverage with ``d`` decimals."""
    return _fixed(x, d)


def _truthy(x: Any) -> bool:
    return str(x).strip().lower() in ("true", "1", "1.0", "yes")


def _label(model: str, table: Mapping[str, str], ctx: Context | None = None) -> str:
    if model not in table and ctx is not None:
        ctx.note(f"model {model!r} has no label in MODEL_LABEL / MODEL_LABEL_E3 (raw name used)")
    return table.get(model, model)


def _order(values: Iterable[Any], order: Sequence[Any]) -> list[Any]:
    present = list(dict.fromkeys(values))
    return [v for v in order if v in present] + sorted((v for v in present if v not in order), key=str)


def _sort(frame: pd.DataFrame, **orders: Sequence[Any]) -> pd.DataFrame:
    """Rows sorted by the given columns in the given value orders (unknown values last, alphabetically)."""
    if frame.empty:
        return frame.reset_index(drop=True)
    keys = []
    for column, order in orders.items():
        rank = {v: i for i, v in enumerate(order)}
        frame = frame.assign(**{f"_{column}": [rank.get(v, len(rank)) for v in frame[column]]})
        keys += [f"_{column}", column]
    return frame.sort_values(keys, kind="stable").drop(columns=[f"_{c}" for c in orders]).reset_index(drop=True)


def _read(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def _all_subset(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["subset"].astype(str) == "all"] if "subset" in frame.columns else frame


# ------------------------------------------------------------------------------------------- T1
T1_METRICS = (("nll", "NLL, nats"), ("crps", "CRPS, m/s²"), ("picp90", "PICP 90 %"), ("width90", "width 90 %, m/s²"),
              ("rmse", "RMSE, m/s²"))  # fmt: skip


def t1_tables(ctx: Context) -> list[Table]:
    """T1: one-step metrics per data set and model, mean ± sd over the training seeds (one table per E1 root, the E5
    roots included when they exist)."""
    out = []
    for key, (variant, target) in e1_roots(ctx).items():
        path = ctx.file(key, "tables", "one_step_metrics.csv")
        if path is None:
            continue
        df = _all_subset(_read(path))
        df = df[df["seed"].astype(str) != "mean"]
        ctx.check_finite(df, [m for m, _ in T1_METRICS], path, "T1")
        rows = []
        for (dataset, model), g in df.groupby(["dataset", "model"], sort=False):
            row: dict[str, Any] = {"variant": variant, "target": target, "dataset": dataset,
                                   "dataset_label": DATASET_LABEL.get(dataset, dataset), "model": model,
                                   "model_label": _label(model, MODEL_LABEL, ctx), "n_seeds": int(g["seed"].nunique()),
                                   "n_windows": int(g["n_windows"].iloc[0]), "n_events": int(g["n_events"].iloc[0])}  # fmt: skip
            for m, _ in T1_METRICS:
                v = pd.to_numeric(g[m], errors="coerce").to_numpy(np.float64)
                row[f"{m}_mean"] = float(np.mean(v))
                row[f"{m}_sd"] = float(np.std(v, ddof=1)) if v.size > 1 else np.nan
            rows.append(row)
        data = _sort(pd.DataFrame(rows), dataset=DATASETS, model=MODEL_ORDER)
        seeds = sorted(set(data["n_seeds"]))
        if len(seeds) > 1:
            ctx.note(f"{display(path)}: the number of seeds differs between models / data sets ({seeds})")
        view = pd.DataFrame({"Data set": data["dataset_label"], "Model": data["model_label"],
                             **{h: [pm(a, b) for a, b in zip(data[f"{m}_mean"], data[f"{m}_sd"])] for m, h in T1_METRICS}})  # fmt: skip
        n = "/".join(str(s) for s in seeds)
        out.append(Table(
            f"t1_one_step__{variant}__{target}", "T1",
            f"One-step metrics, {TARGET_LABEL[target]}, {VARIANT_LABEL[variant]}: NLL, CRPS, coverage (PICP) and width of "
            f"the 90 % interval, RMSE of the predictive mean per data set and model; mean ± sd over {n} training seeds.",
            [path], data, view, group_depth=1,
        ))  # fmt: skip
    return out


# ------------------------------------------------------------------------------------------- T2
def t2_tables(ctx: Context) -> list[Table]:
    """T2: H3.1 per model, decomposition and shifted set (ratios of means and of geometric means, verdicts; H3.1e
    growth verdict of the pair) per E1 root, and one summary of H3.1 and H3.1e over all roots (the E5 roots included
    when they exist)."""
    out, summary, sources = [], [], []
    for key, (variant, target) in e1_roots(ctx).items():
        pv = ctx.file(key, "tables", "h31_verdicts.csv")
        pe = ctx.file(key, "tables", "h31e_verdicts.csv")
        pg = ctx.file(key, "tables", "extra_shift_geometric.csv")
        if pv is None:
            continue
        ver = _all_subset(_read(pv))
        e = _all_subset(_read(pe)) if pe is not None else pd.DataFrame(columns=["model", "kind", "dataset", "verdict_ci", "verdict_point"])
        geo = _read(pg) if pg is not None else pd.DataFrame(columns=["model", "dataset", "quantity", "ratio_geometric", "geo_lo",
                                                                     "geo_hi", "ratio_means"])  # fmt: skip
        ctx.check_finite(ver, ["r_epi", "r_epi_lo", "r_epi_hi", "r_alea", "r_alea_lo", "r_alea_hi"], pv, "T2")
        rows = []
        for r in ver.itertuples(index=False):
            # geometric twins (analyze_e1_extra.py): own decomposition -> the model's epistemic / aleatoric output;
            # across seeds -> no geometric epistemic ratio, aleatoric = seed mean of the predictive variance ("var")
            q_epi, q_alea = ("epistemic_var", "aleatoric_var") if r.kind == "model" else (None, "var")
            g = {}
            for name, q in (("epi", q_epi), ("alea", q_alea)):
                hit = geo[(geo["model"] == r.model) & (geo["dataset"] == r.dataset) & (geo["quantity"] == q)] if q else geo.iloc[0:0]
                g[name] = hit.iloc[0] if len(hit) else None
            if g["alea"] is not None and not np.isclose(float(g["alea"]["ratio_means"]), float(r.r_alea), rtol=1e-6):
                ctx.note(f"{display(pg)}: ratio of means of {q_alea!r} differs from r_alea of h31_verdicts.csv "
                         f"({r.model}, {r.kind}, {r.dataset})")  # fmt: skip
            ee = e[(e["model"] == r.model) & (e["kind"] == r.kind) & (e["dataset"] == r.dataset)]
            rows.append({
                "variant": variant, "target": target, "model": r.model, "model_label": _label(r.model, MODEL_LABEL, ctx),
                "kind": r.kind, "dataset": r.dataset, "dataset_label": DATASET_LABEL.get(r.dataset, r.dataset), "pair": r.pair,
                "r_epi": r.r_epi, "r_epi_lo": r.r_epi_lo, "r_epi_hi": r.r_epi_hi,
                "r_alea": r.r_alea, "r_alea_lo": r.r_alea_lo, "r_alea_hi": r.r_alea_hi,
                **{f"geo_{n}{s}": (float(g[n][c]) if g[n] is not None else np.nan)
                   for n in ("epi", "alea") for s, c in (("", "ratio_geometric"), ("_lo", "geo_lo"), ("_hi", "geo_hi"))},
                "verdict_ci": r.verdict_ci, "verdict_point": r.verdict_point,
                "h31e_growth_ci": ee.iloc[0]["verdict_ci"] if len(ee) else "", "h31e_growth_point": ee.iloc[0]["verdict_point"] if len(ee) else "",
            })  # fmt: skip
        data = _sort(pd.DataFrame(rows), model=MODEL_ORDER, kind=tuple(KIND_LABEL), dataset=SHIFT_ORDER)
        view = pd.DataFrame({
            "Model": data["model_label"], "Decomposition": [KIND_LABEL.get(k, k) for k in data["kind"]],
            "Data set": [lab + ("" if p == "shift" else f", {p.replace('_', '-')}") for lab, p in zip(data["dataset_label"], data["pair"])],
            "R epistemic": [ci(a, b, c) for a, b, c in zip(data["r_epi"], data["r_epi_lo"], data["r_epi_hi"])],
            "R aleatoric": [ci(a, b, c) for a, b, c in zip(data["r_alea"], data["r_alea_lo"], data["r_alea_hi"])],
            "geometric R epistemic": [ci(a, b, c) for a, b, c in zip(data["geo_epi"], data["geo_epi_lo"], data["geo_epi_hi"])],
            "geometric R aleatoric": [ci(a, b, c) for a, b, c in zip(data["geo_alea"], data["geo_alea_lo"], data["geo_alea_hi"])],
            "H3.1 (interval / point)": [f"{a} / {b}" for a, b in zip(data["verdict_ci"], data["verdict_point"])],
            "H3.1e growth": [a or DASH for a in data["h31e_growth_ci"]],
        })  # fmt: skip
        out.append(Table(
            f"t2_h31__{variant}__{target}", "T2",
            f"H3.1 under dataset shift, {TARGET_LABEL[target]}, {VARIANT_LABEL[variant]}: variance ratios shifted / "
            f"{in_domain_label(ctx, key)} (ratio of means, the basis of the verdict, and ratio of geometric means) with "
            "95 % event-bootstrap intervals, H3.1 verdict and H3.1e growth verdict per model, decomposition and data set.",
            [p for p in (pv, pg, pe) if p is not None], data, view, group_depth=2,
        ))  # fmt: skip
        ps, pes = ctx.file(key, "tables", "h31_summary.csv"), ctx.file(key, "tables", "h31e_summary.csv")
        if ps is None or pes is None:
            continue
        sources += [ps, pes]
        s, es = _read(ps), _read(pes)
        for r in s.to_dict("records"):
            er = es[(es["model"] == r["model"]) & (es["kind"] == r["kind"])]
            er = er.iloc[0].to_dict() if len(er) else {}
            summary.append({
                "variant": variant, "target": target, "model": r["model"], "model_label": _label(r["model"], MODEL_LABEL, ctx),
                "kind": r["kind"], "h31_overall": r["overall"], "h31_overall_point": r["overall_point"], "h31_n_pairs": r["n_pairs"],
                "h31_n_supported": r["n_supported"], "h31_n_refuted": r["n_refuted"], "h31_n_inconclusive": r["n_inconclusive"],
                "h31_placebo": r.get("placebo_verdicts", ""), "h31_cross_check": r.get("cross_check_verdicts", ""),
                "h31e_main": _truthy(er.get("main", False)), "h31e_rule": er.get("rule", ""), "h31e_conc": er.get("conc", np.nan),
                "h31e_conc_lo": er.get("conc_lo", np.nan), "h31e_conc_hi": er.get("conc_hi", np.nan),
                "h31e_conc_verdict": er.get("verdict_conc_ci", ""), "h31e_n_supported": er.get("n_supported", np.nan),
                "h31e_n_refuted": er.get("n_refuted", np.nan), "h31e_n_inconclusive": er.get("n_inconclusive", np.nan),
                "h31e_overall": er.get("overall", ""), "h31e_overall_point": er.get("overall_point", ""),
            })  # fmt: skip
    if summary:
        data = _sort(pd.DataFrame(summary), variant=tuple(VARIANT_LABEL), target=tuple(TARGET_LABEL), model=MODEL_ORDER,
                     kind=tuple(KIND_LABEL))  # fmt: skip

        def counts(r: pd.Series, p: str) -> str:
            v = [r[f"{p}n_supported"], r[f"{p}n_refuted"], r[f"{p}n_inconclusive"]]
            return DASH if any(pd.isna(x) for x in v) else "/".join(str(int(x)) for x in v)

        view = pd.DataFrame({
            "Signals": [VARIANT_LABEL[v] for v in data["variant"]], "Target": data["target"], "Model": data["model_label"],
            "Decomposition": [KIND_LABEL.get(k, k) + ("*" if m else "") for k, m in zip(data["kind"], data["h31e_main"])],
            "H3.1 (interval / point)": [f"{a} / {b}" for a, b in zip(data["h31_overall"], data["h31_overall_point"])],
            "H3.1 pairs s/r/i": [counts(r, "h31_") for _, r in data.iterrows()],
            "placebo": [str(p).split(":", 1)[-1] if str(p) not in ("", "nan") else DASH for p in data["h31_placebo"]],
            "H3.1e concentration": [ci(a, b, c) for a, b, c in zip(data["h31e_conc"], data["h31e_conc_lo"], data["h31e_conc_hi"])],
            "H3.1e pairs s/r/i": [counts(r, "h31e_") for _, r in data.iterrows()],
            "H3.1e (interval / point)": [f"{a} / {b}" if a else DASH for a, b in zip(data["h31e_overall"], data["h31e_overall_point"])],
        })  # fmt: skip
        out.insert(0, Table(
            "t2_h31_summary", "T2",
            "H3.1 and H3.1e per signal variant, target, model and decomposition: overall verdicts (interval / point "
            "rule), supported/refuted/inconclusive counts over the four counted shift pairs, placebo verdict, "
            "concentration of the epistemic variance in the rare states of the in-domain test set (rare_mass; highD "
            "test, the pooled test folds for a pooled training domain) with its 95 % interval; * = main H3.1e verdict "
            "(the model's own decomposition).",
            sources, data, view, group_depth=2,
        ))  # fmt: skip
    return out


# ------------------------------------------------------------------------------------------- T3
def t3_tables(ctx: Context) -> list[Table]:
    """T3: H3.3 NLL of HeteroMDN against LSTMConstSigma by regime (Holm), and HeteroMDN's predicted variance and
    squared residual by regime relative to steady following, per E1 root."""
    out = []
    for key, (variant, target) in E1_ROOTS.items():
        what = f"{TARGET_LABEL[target]}, {VARIANT_LABEL[variant]}"
        path = ctx.file(key, "tables", "h33_nll.csv")
        if path is not None:
            df = _all_subset(_read(path))
            ctx.check_finite(df, ["nll_a", "nll_b", "diff", "diff_lo", "diff_hi", "p_holm"], path, "T3")
            data = pd.DataFrame({
                "variant": variant, "target": target, "dataset": df["dataset"],
                "dataset_label": [DATASET_LABEL.get(d, d) for d in df["dataset"]], "regime": df["group"],
                "n_windows": df["n_windows"], "n_events": df["n_events"], "model_a": df["model_a"], "model_b": df["model_b"],
                **{c: df[c] for c in ("nll_a", "nll_a_lo", "nll_a_hi", "nll_b", "nll_b_lo", "nll_b_hi", "diff", "diff_lo",
                                      "diff_hi", "p_holm")},
                "reject": [_truthy(x) for x in df["reject"]],
                "intervals_separate": [not _truthy(x) if str(x) != "nan" else np.nan for x in df["ci_overlap"]],
            })  # fmt: skip
            data = _sort(data, dataset=DATASETS, regime=("all", *REGIMES))
            a = _label(str(data["model_a"].iloc[0]), MODEL_LABEL, ctx) if len(data) else "model a"
            b = _label(str(data["model_b"].iloc[0]), MODEL_LABEL, ctx) if len(data) else "model b"
            view = pd.DataFrame({
                "Data set": data["dataset_label"], "Regime": ["all windows" if g == "all" else g for g in data["regime"]],
                "windows": [f"{int(n):,}" for n in data["n_windows"]],
                f"NLL {a}": [ci(x, y, z) for x, y, z in zip(data["nll_a"], data["nll_a_lo"], data["nll_a_hi"])],
                f"NLL {b}": [ci(x, y, z) for x, y, z in zip(data["nll_b"], data["nll_b_lo"], data["nll_b_hi"])],
                "difference": [ci(x, y, z) for x, y, z in zip(data["diff"], data["diff_lo"], data["diff_hi"])],
                "p (Holm)": [prob(p, 3) for p in data["p_holm"]],
                "intervals separate": [DASH if pd.isna(s) else ("yes" if s else "no") for s in data["intervals_separate"]],
            })  # fmt: skip
            out.append(Table(
                f"t3_h33_nll__{variant}__{target}", "T3",
                f"H3.3, {what}: NLL (nats) of {a} and {b} by driving regime with 95 % event-bootstrap intervals, their "
                "paired difference (negative: the heteroscedastic model is better), Holm-adjusted bootstrap p-value and "
                "whether the two intervals are separate (the literal criterion of the plan).",
                [path], data, view, group_depth=1,
            ))  # fmt: skip
        path = ctx.file(key, "tables", "regime_ordering.csv")
        if path is None:
            continue
        df = _all_subset(_read(path))
        df = df[df["model"] == "hetero_mdn"]
        ctx.check_finite(df, ["estimate", "lo", "hi"], path, "T3 (HeteroMDN)")
        rows = []
        for (dataset, quantity), g in df.groupby(["dataset", "quantity"], sort=False):
            row: dict[str, Any] = {"variant": variant, "target": target, "model": "hetero_mdn", "dataset": dataset,
                                   "dataset_label": DATASET_LABEL.get(dataset, dataset), "quantity": quantity,
                                   "reference": str(g["reference"].iloc[0])}  # fmt: skip
            for regime in [x for x in REGIMES if x != "steady"]:
                hit = g[g["regime"] == regime]
                h = hit.iloc[0] if len(hit) else None
                for c in ("estimate", "lo", "hi", "p_holm"):
                    row[f"{c}_{regime}"] = float(h[c]) if h is not None else np.nan
                row[f"reject_{regime}"] = _truthy(h["reject"]) if h is not None else False
            rows.append(row)
        data = _sort(pd.DataFrame(rows), dataset=DATASETS, quantity=("var", "mse"))
        view = pd.DataFrame({
            "Data set": data["dataset_label"],
            "Quantity": [{"var": "predicted variance", "mse": "squared residual"}.get(q, q) for q in data["quantity"]],
            **{regime: [ci(r[f"estimate_{regime}"], r[f"lo_{regime}"], r[f"hi_{regime}"]) + ("*" if r[f"reject_{regime}"] else "")
                        for _, r in data.iterrows()] for regime in REGIMES if regime != "steady"},
        })  # fmt: skip
        out.append(Table(
            f"t3_regime_variance__{variant}__{target}", "T3",
            f"H3.3 variance ordering, {what}: HeteroMDN's mean predicted variance and mean squared residual per regime "
            "relative to steady following, 95 % event-bootstrap intervals; * = Holm-adjusted p < 0.05 for ratio = 1.",
            [path], data, view, group_depth=1,
        ))  # fmt: skip
    return out


# ------------------------------------------------------------------------------------------- T4
def t4_tables(ctx: Context) -> list[Table]:
    """T4: H3.4 coverage of the 90 % gap interval by horizon and correction, one-step conformal coverage, verdict
    counts (E2 roots, the E5 root included when it exists)."""
    out, counts, count_sources = [], [], []
    for key, variant in e2_roots(ctx).items():
        pc = ctx.file(key, "tables", "coverage_by_horizon.csv")
        pv = ctx.file(key, "tables", "h34_verdicts.csv")
        po = ctx.file(key, "tables", "onestep_conformal.csv")
        cov = _read(pc) if pc is not None else None
        ver = _read(pv) if pv is not None else None
        one = _read(po) if po is not None else None
        targets = _order(pd.concat([f["target"] for f in (cov, ver, one) if f is not None]).astype(str), tuple(TARGET_LABEL)) \
            if any(f is not None for f in (cov, ver, one)) else []  # fmt: skip
        for target in targets:
            what = f"{TARGET_LABEL[target]}, {VARIANT_LABEL[variant]}"
            if cov is not None:
                c = cov[(cov["target"] == target) & cov["dataset"].isin(ROLLOUT_DATASETS)]
                ctx.check_finite(c[c["variant"].isin([m for m, _ in E2_METHODS])], ["picp"], pc, "T4")
                rows = []
                for (dataset, model), g in c.groupby(["dataset", "model"], sort=False):
                    row: dict[str, Any] = {"variant": variant, "target": target, "dataset": dataset,
                                           "dataset_label": DATASET_LABEL.get(dataset, dataset), "model": model,
                                           "model_label": _label(model, MODEL_LABEL, ctx)}  # fmt: skip
                    for m, _ in E2_METHODS:
                        gm = g[g["variant"] == m].set_index("horizon_s")
                        for h in HORIZONS:
                            row[f"picp_{m}_{h:g}s"] = float(gm.loc[h, "picp"]) if h in gm.index else np.nan
                    for m in ("raw", "aci_online"):
                        gm = g[g["variant"] == m].set_index("horizon_s")
                        for h in HORIZONS:
                            row[f"width_median_{m}_{h:g}s"] = float(gm.loc[h, "width_median"]) if h in gm.index else np.nan
                    raw = g[g["variant"] == "raw"].set_index("horizon_s")
                    row["collided_10s"] = float(raw.loc[10.0, "collided"]) if 10.0 in raw.index else np.nan
                    if ver is not None:
                        hv = ver[(ver["target"] == target) & (ver["dataset"] == dataset) & (ver["model"] == model)
                                 & (ver["corrected"] == "aci_online")]  # fmt: skip
                        hv = hv.iloc[0] if len(hv) else None
                        row["h34_verdict"] = hv["verdict"] if hv is not None else ""
                        row["h34_verdict_ci"] = hv["verdict_ci"] if hv is not None else ""
                    rows.append(row)
                data = _sort(pd.DataFrame(rows), dataset=ROLLOUT_DATASETS, model=MODEL_ORDER)
                view = pd.DataFrame({
                    "Data set": data["dataset_label"], "Model": data["model_label"],
                    **{f"{lab}, 1/3/5/10 s": [" ".join(prob(r[f"picp_{m}_{h:g}s"]) for h in HORIZONS) for _, r in data.iterrows()]
                       for m, lab in E2_METHODS},
                    "median width at 10 s, raw → ACI, m": [f"{num(a)} → {num(b)}" for a, b in
                                                            zip(data["width_median_raw_10s"], data["width_median_aci_online_10s"])],
                    "H3.4 (point / interval)": [f"{a} / {b}" if a else DASH for a, b in
                                                zip(data.get("h34_verdict", [""] * len(data)), data.get("h34_verdict_ci", [""] * len(data)))],
                })  # fmt: skip
                out.append(Table(
                    f"t4_coverage__{variant}__{target}", "T4",
                    f"H3.4 in closed-loop rollouts, {what}: coverage of the 90 % gap interval at 1, 3, 5 and 10 s, raw and "
                    "with the static, ACI frozen and ACI online corrections (calibrated on highD validation), median "
                    "width at 10 s, and the H3.4 verdict with ACI online.",
                    [p for p in (pc, pv) if p is not None], data, view, group_depth=1,
                ))  # fmt: skip
            if one is not None:
                o = one[one["target"] == target]
                ctx.check_finite(o, ["picp_raw", "picp_conformal"], po, "T4 (one-step)")
                data = o[["model", "dataset", "score", "picp_raw", "picp_raw_lo", "picp_raw_hi", "picp_conformal",
                          "picp_conformal_lo", "picp_conformal_hi", "width_raw", "width_conformal", "q", "n_windows"]].copy()  # fmt: skip
                data.insert(0, "target", target)
                data.insert(0, "variant", variant)
                data.insert(3, "model_label", [_label(m, MODEL_LABEL, ctx) for m in data["model"]])
                data = _sort(data, model=MODEL_ORDER, dataset=DATASETS, score=("normalised", "absolute"))
                norm = data[data["score"] == "normalised"]
                datasets = _order(norm["dataset"], DATASETS)
                vrows = []
                for model in _order(norm["model"], MODEL_ORDER):
                    cells = {"Model": _label(model, MODEL_LABEL)}
                    for d in datasets:
                        hit = norm[(norm["model"] == model) & (norm["dataset"] == d)]
                        cells[DATASET_LABEL.get(d, d)] = (f"{prob(hit.iloc[0]['picp_raw'], 3)} → {prob(hit.iloc[0]['picp_conformal'], 3)}"
                                                          if len(hit) else DASH)  # fmt: skip
                    vrows.append(cells)
                out.append(Table(
                    f"t4_onestep_conformal__{variant}__{target}", "T4",
                    f"H3.4 (i), {what}: one-step coverage of the 90 % interval, raw → split conformal with the normalised "
                    "score |y - mu| / sigma calibrated on highD validation (the absolute score is in the CSV).",
                    [po], data, pd.DataFrame(vrows),
                ))  # fmt: skip
            if ver is not None:
                count_sources.append(pv)
                for corrected in _order(ver["corrected"], ("static", "aci_frozen", "aci_online")):
                    v = ver[(ver["target"] == target) & (ver["corrected"] == corrected)]
                    pt, iv = v["verdict"].value_counts(), v["verdict_ci"].value_counts()
                    counts.append({"variant": variant, "target": target, "corrected": corrected, "n_cases": len(v),
                                   **{f"{k}_point": int(pt.get(k, 0)) for k in ("supported", "refuted", "inconclusive")},
                                   **{f"{k}_interval": int(iv.get(k, 0)) for k in ("supported", "refuted", "inconclusive")}})  # fmt: skip
    if counts:
        data = pd.DataFrame(counts)
        label = dict(E2_METHODS)
        view = pd.DataFrame({
            "Signals": [VARIANT_LABEL[v] for v in data["variant"]], "Target": data["target"],
            "Correction": [label.get(c, c) for c in data["corrected"]], "cases": data["n_cases"].astype(str),
            "point rule s/r/i": [f"{a}/{b}/{c}" for a, b, c in zip(data["supported_point"], data["refuted_point"], data["inconclusive_point"])],
            "interval rule s/r/i": [f"{a}/{b}/{c}" for a, b, c in
                                    zip(data["supported_interval"], data["refuted_interval"], data["inconclusive_interval"])],
        })  # fmt: skip
        out.append(Table(
            "t4_h34_verdict_counts", "T4",
            "H3.4 verdict counts over the (model, data set) cases per signal variant, target and correction: "
            "supported/refuted/inconclusive by point estimates and by interval bounds.",
            sorted(set(count_sources), key=str), data, view, group_depth=2,
        ))  # fmt: skip
    return out


# ------------------------------------------------------------------------------------------- E3
def e3_settings(ctx: Context, key: str) -> dict[str, Any]:
    """reference, min_profiles and v_max of an E3 root (``config.yaml``; design defaults when absent)."""
    out = {"reference": REFERENCE, "min_profiles": 5, "v_max": None}
    path = ctx.roots.get(key, Path("-")) / "config.yaml"
    if path.exists():
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        h = cfg.get("h32") or {}
        out["reference"] = str(h.get("reference") or REFERENCE)
        out["min_profiles"] = int(h.get("min_profiles") or 5)
        out["v_max"] = (cfg.get("platoon") or {}).get("v_max")
    return out


def run_label(ctx: Context, key: str, variant: str | None = None) -> str:
    base = E3_RUN_LABEL.get(key, key)
    v_max = e3_settings(ctx, key)["v_max"]
    if v_max is not None:
        base += f" {float(v_max):g} m/s"
    return base if variant is None else f"{base}, {E3_VARIANT_LABEL.get(variant, variant)}"


def _e3_model_order(models: Iterable[str], reference: str) -> list[str]:
    order = _order(models, MODEL_ORDER)
    return [reference, *[m for m in order if m != reference]] if reference in order else order


def h32_rows(ctx: Context, key: str, variant: str, set_name: str) -> tuple[list[dict[str, Any]], list[Path]]:
    """T5 rows of one E3 root, signal variant and profile set: per driver mode and model the amplitude CRPS (index
    all, window response) and the H3.2 reduction against the reference with its verdict."""
    pc, ph = ctx.file(key, "tables", "amplitude_crps.csv"), ctx.file(key, "tables", "h32.csv")
    if pc is None or ph is None:
        return [], []
    s = e3_settings(ctx, key)
    crps = _read(pc, dtype={"index": str})
    h32 = _read(ph)
    crps = crps[(crps["variant"] == variant) & (crps["set"] == set_name) & (crps["window"] == "response") & (crps["index"] == "all")]
    h32 = h32[(h32["variant"] == variant) & (h32["set"] == set_name) & (h32["window"] == "response")
              & (h32["quantity"] == "amplitude") & (h32["reference"] == s["reference"])]  # fmt: skip
    if crps.empty:
        ctx.missing.append(f"{key}: no amplitude_crps rows for variant {variant!r}, set {set_name!r}")
    ctx.check_finite(crps, ["crps", "crps_lo", "crps_hi"], pc, "T5")
    rows = []
    for mode in _order(crps["mode"], MODES):
        c = crps[crps["mode"] == mode]
        n = int(c["n_profiles"].max())
        for model in _e3_model_order(c["model"], s["reference"]):
            cr = c[c["model"] == model].iloc[0]
            hr = h32[(h32["mode"] == mode) & (h32["model"] == model)]
            hr = hr.iloc[0] if len(hr) else None
            if hr is not None and not np.isclose(float(hr["crps_model"]), float(cr["crps"]), rtol=1e-9, atol=1e-12):
                ctx.note(f"{display(ph)}: crps_model differs from amplitude_crps.csv ({variant}, {set_name}, {mode}, {model})")
            judged = n >= s["min_profiles"]
            rows.append({
                "run": key, "run_label": run_label(ctx, key), "variant": variant, "set": set_name, "mode": mode, "n_profiles": n,
                "model": model, "model_label": _label(model, figures_e3.model_labels(variant), ctx), "reference": s["reference"],
                "crps": cr["crps"], "crps_lo": cr["crps_lo"], "crps_hi": cr["crps_hi"],
                "reduction": hr["reduction"] if hr is not None else np.nan, "reduction_lo": hr["lo"] if hr is not None else np.nan,
                "reduction_hi": hr["hi"] if hr is not None else np.nan,
                "verdict": "reference" if model == s["reference"] else (hr["verdict"] if hr is not None else ""),
                "verdict_point": "" if hr is None else hr["verdict_point"], "judged": bool(judged and model != s["reference"]),
                "min_profiles": s["min_profiles"],
            })  # fmt: skip
    return rows, [pc, ph]


def _t5_view(data: pd.DataFrame, with_run: bool) -> pd.DataFrame:
    verdict = [v if v == "reference" or j else "not judged" for v, j in zip(data["verdict"], data["judged"])]
    cols: dict[str, Any] = {}
    if with_run:
        cols["Run"] = data["run_label"]
    cols.update({
        "Mode (profiles)": [f"{m} ({n})" for m, n in zip(data["mode"], data["n_profiles"])], "Model": data["model_label"],
        "CRPS, m/s": [ci(a, b, c) for a, b, c in zip(data["crps"], data["crps_lo"], data["crps_hi"])],
        "reduction": [DASH if v == "reference" else ci2(a, b, c)
                      for v, a, b, c in zip(data["verdict"], data["reduction"], data["reduction_lo"], data["reduction_hi"])],
        "verdict": verdict,
    })  # fmt: skip
    return pd.DataFrame(cols)


def t5_tables(ctx: Context) -> list[Table]:
    """T5: H3.2 per driver mode (main run, both signal variants, extended set; the sensitivities with the main run
    on the headline set)."""
    out = []
    if "e3" in ctx.roots:
        pc = ctx.file("e3", "tables", "amplitude_crps.csv")
        variants = _order(_read(pc)["variant"].astype(str), tuple(VARIANT_LABEL)) if pc is not None else []
        for variant in variants:
            rows, sources = h32_rows(ctx, "e3", variant, E3_SETS["e3"])
            if not rows:
                continue
            data = pd.DataFrame(rows)
            n_all = int(data.loc[data["mode"] == "all", "n_profiles"].max())
            out.append(Table(
                f"t5_h32__e3__{variant}", "T5",
                f"H3.2 per driver mode, {E3_VARIANT_LABEL.get(variant, variant)}, {E3_SETS['e3']} set ({n_all} profiles): "
                "amplitude CRPS (m/s, window profile + tail, mean over the measured vehicles and the profiles) with 95 % "
                "bootstrap interval, reduction 1 - CRPS/CRPS(IDM + white noise) paired by profile, verdict (supported: "
                "lower bound >= 0.20; refuted: upper bound < 0.05; fewer than "
                f"{int(data['min_profiles'].iloc[0])} profiles: not judged). The ACC and Human rows are the headline "
                "verdicts, 'all' the pooled secondary line.",
                sources, data, _t5_view(data, with_run=False), group_depth=1,
            ))  # fmt: skip
    rows, sources = [], []
    for key in ("e3", "e3_vcap", "e3_step"):
        if key not in ctx.roots:
            continue
        r, s = h32_rows(ctx, key, "raw", "headline")
        rows += r
        sources += s
    if rows:
        data = pd.DataFrame(rows)
        out.append(Table(
            "t5_h32__sensitivities", "T5",
            "H3.2 sensitivities of design.md 13.5 on the headline set (recorded signals): the main run, the follower "
            "speed cap, and the deep ensemble with per-step draws; columns as in the main T5 tables.",
            sources, data, _t5_view(data, with_run=True), group_depth=2,
        ))  # fmt: skip
    return out


def _e3_blocks(ctx: Context) -> list[tuple[str, str, str]]:
    """(root, variant, set) of the platoon diagnostics: the main run per signal variant, the sensitivities."""
    blocks = []
    for key, set_name in E3_SETS.items():
        path = ctx.roots.get(key, Path("-")) / "tables" / "collisions.csv"
        if key not in ctx.roots or not path.exists():
            continue
        for variant in _order(_read(path, dtype={"index": str})["variant"].astype(str), tuple(VARIANT_LABEL)):
            blocks.append((key, variant, set_name))
    return blocks


def _run_cell(ctx: Context, key: str, variant: str, n: int) -> str:
    return f"{run_label(ctx, key, variant)} ({n} profiles)"


def t6_tables(ctx: Context) -> list[Table]:
    """T6: platoon diagnostics: collisions by vehicle index, amplitude interval coverage, growth exponents."""
    out = []
    blocks = _e3_blocks(ctx)
    coll, cover, growth, src = [], [], [], {"collisions": [], "coverage": [], "growth": []}
    for key, variant, set_name in blocks:
        n_prof = _profiles_of(ctx, key, variant, set_name)
        run = _run_cell(ctx, key, variant, n_prof)
        path = ctx.file(key, "tables", "collisions.csv")
        if path is not None:
            src["collisions"].append(path)
            df = _read(path, dtype={"index": str})
            df = df[(df["variant"] == variant) & (df["set"] == set_name)]
            ctx.check_finite(df, ["collided", "collided_burn_in"], path, "T6 collisions")
            for model in _e3_model_order(df["model"], REFERENCE):
                g = df[df["model"] == model].set_index("index")
                row: dict[str, Any] = {"run": key, "run_label": run, "variant": variant, "set": set_name, "model": model,
                                       "model_label": _label(model, figures_e3.model_labels(variant), ctx)}  # fmt: skip
                indices = sorted((int(i) for i in g.index if str(i).isdigit()))
                for i in indices:
                    row[f"collided_{i}"] = float(g.loc[str(i), "collided"])
                for c in ("collided", "collided_burn_in", "profiles_with_collision"):
                    row[f"{c}_any"] = float(g.loc["any", c]) if "any" in g.index else np.nan
                row["collided_burn_in_1"] = float(g.loc["1", "collided_burn_in"]) if "1" in g.index else np.nan
                row["n_rollouts"] = int(g.loc["any", "n_rollouts"]) if "any" in g.index else np.nan
                coll.append(row)
        path = ctx.file(key, "tables", "amplitude_coverage.csv")
        if path is not None:
            src["coverage"].append(path)
            df = _read(path, dtype={"index": str})
            df = df[(df["variant"] == variant) & (df["set"] == set_name) & (df["index"] == "all")]
            ctx.check_finite(df, ["coverage", "coverage_lo", "coverage_hi", "width_median"], path, "T6 coverage")
            for model in _e3_model_order(df["model"], REFERENCE):
                g = df[df["model"] == model].set_index("method")
                row = {"run": key, "run_label": run, "variant": variant, "set": set_name, "model": model,
                       "model_label": _label(model, figures_e3.model_labels(variant), ctx)}  # fmt: skip
                for m in ("raw", "static_loo", "aci_online"):
                    for c in ("coverage", "coverage_lo", "coverage_hi", "width_median"):
                        row[f"{c}_{m}"] = float(g.loc[m, c]) if m in g.index else np.nan
                row["nominal"] = float(g["nominal"].iloc[0]) if len(g) else np.nan
                row["n_profiles"] = int(g["n_profiles"].max()) if len(g) else 0
                cover.append(row)
        path = ctx.file(key, "tables", "growth.csv")
        if path is not None:
            src["growth"].append(path)
            df = _read(path)
            df = df[df["set"] == set_name]
            models = df[df["variant"] == variant]
            emp = df[(df["model"] == "empirical") & (df["window"] == "profile")]
            for model in [*_e3_model_order(models["model"], REFERENCE), *(["empirical"] if len(emp) else [])]:
                g = emp if model == "empirical" else models[models["model"] == model]
                row = {"run": key, "run_label": run, "variant": variant, "set": set_name, "model": model,
                       "model_label": "measured platoons" if model == "empirical" else
                                      _label(model, figures_e3.model_labels(variant), ctx)}  # fmt: skip
                for window in ("profile", "response"):
                    w = g[g["window"] == window].sort_values("index")
                    curve = w[w["sd"].notna()]
                    fitted = curve[curve["fitted"].astype(str).str.lower() == "true"] if "fitted" in curve else curve
                    fitted = fitted if len(fitted) else curve
                    first = w.iloc[0] if len(w) else None
                    row.update({
                        f"beta_{window}": first["beta"] if first is not None else np.nan,
                        f"beta_lo_{window}": first["beta_lo"] if first is not None else np.nan,
                        f"beta_hi_{window}": first["beta_hi"] if first is not None else np.nan,
                        f"share_neg_d2_{window}": first["share_neg_d2"] if first is not None else np.nan,
                        f"fit_first_{window}": int(fitted["index"].iloc[0]) if len(fitted) else np.nan,
                        f"fit_last_{window}": int(fitted["index"].iloc[-1]) if len(fitted) else np.nan,
                        f"sd_fit_first_{window}": fitted["sd"].iloc[0] if len(fitted) else np.nan,
                        f"sd_fit_last_{window}": fitted["sd"].iloc[-1] if len(fitted) else np.nan,
                        f"index_last_{window}": int(curve["index"].iloc[-1]) if len(curve) else np.nan,  # last measured
                        f"sd_last_{window}": curve["sd"].iloc[-1] if len(curve) else np.nan,
                    })  # fmt: skip
                    if len(w) and w["sd"].isna().any():
                        ctx.note(f"{display(path)}: sd is NaN for {model} ({set_name}, window {window}) at indices "
                                 f"{w.loc[w['sd'].isna(), 'index'].astype(int).tolist()} (too few profiles measured them)")
                growth.append(row)
    if coll:
        data = pd.DataFrame(coll)
        view = pd.DataFrame({
            "Run": data["run_label"], "Model": data["model_label"],
            **{f"i = {i}": [prob(x) for x in data.get(f"collided_{i}", pd.Series([np.nan] * len(data)))] for i in COLLISION_INDICES},
            "any pair": [prob(x, 3) for x in data["collided_any"]], "burn-in, i = 1": [prob(x, 3) for x in data["collided_burn_in_1"]],
        })  # fmt: skip
        out.append(Table(
            "t6_collisions", "T6",
            "Platoon collisions: share of rollouts in which vehicle i collided (whole time line), in which any pair of the "
            "platoon collided, and in which the first follower collided during the constant-speed burn-in; main run on "
            "the extended set, sensitivities on the headline set (every index 1-20 in the CSV).",
            src["collisions"], data, view, group_depth=1,
        ))  # fmt: skip
    if cover:
        data = pd.DataFrame(cover)
        view = pd.DataFrame({
            "Run": data["run_label"], "Model": data["model_label"],
            **{lab: [f"{prob(r[f'coverage_{m}'])} [{prob(r[f'coverage_lo_{m}'])}, {prob(r[f'coverage_hi_{m}'])}]" for _, r in data.iterrows()]
               for m, lab in (("raw", "raw"), ("static_loo", "static (LOO)"), ("aci_online", "ACI online"))},
            "median width raw / static / ACI, m/s": [" / ".join(num(r[f"width_median_{m}"]) for m in ("raw", "static_loo", "aci_online"))
                                                     for _, r in data.iterrows()],
        })  # fmt: skip
        out.append(Table(
            "t6_amplitude_coverage", "T6",
            "Coverage of the 90 % amplitude interval over all measured vehicles (95 % bootstrap interval over profiles): "
            "raw (S + 1) sample quantiles, leave-one-profile-out static correction, ACI online over the profiles, and "
            "the median interval width.",
            src["coverage"], data, view, group_depth=1,
        ))  # fmt: skip
    if growth:
        data = pd.DataFrame(growth)
        for c in [c for c in data.columns if c.startswith(("fit_first_", "fit_last_", "index_last_"))]:
            data[c] = data[c].astype("Int64")  # vehicle indices; NaN where a window is absent (measured platoons)

        def fit_range(r: pd.Series) -> str:
            a, b = r["fit_first_profile"], r["fit_last_profile"]
            return DASH if pd.isna(a) else f"{int(a)}-{int(b)}"

        view = pd.DataFrame({
            "Run": data["run_label"], "Model": data["model_label"],
            "beta, profile window": [ci2(a, b, c) for a, b, c in zip(data["beta_profile"], data["beta_lo_profile"], data["beta_hi_profile"])],
            "fitted indices": [fit_range(r) for _, r in data.iterrows()],
            "beta, profile + tail": [ci2(a, b, c) for a, b, c in zip(data["beta_response"], data["beta_lo_response"], data["beta_hi_response"])],
            "sd, first → last fitted index, m/s": [DASH if pd.isna(r["sd_fit_first_profile"]) else
                                                   f"{num(r['sd_fit_first_profile'])} → {num(r['sd_fit_last_profile'])}"
                                                   for _, r in data.iterrows()],
            "d2 < 0": [prob(x) for x in data["share_neg_d2_profile"]],
        })  # fmt: skip
        out.append(Table(
            "t6_growth", "T6",
            "Growth of the speed standard deviation along the platoon, sd_i = sd_1 i^beta (least squares in logs, 95 % "
            "bootstrap interval over profiles), over the window of the measured data and over profile + tail; the "
            "measured platoons are fitted over the indices that at least half of the profiles measured; mean sd at the "
            "first and the last fitted index and share of negative second differences (d2 < 0), profile window.",
            src["growth"], data, view, group_depth=1,
        ))  # fmt: skip
    return out


def _profiles_of(ctx: Context, key: str, variant: str, set_name: str) -> int:
    """Number of profiles of a set (amplitude_crps.csv, mode all, index all)."""
    path = ctx.roots.get(key, Path("-")) / "tables" / "amplitude_crps.csv"
    if not path.exists():
        return 0
    c = _read(path, dtype={"index": str})
    c = c[(c["variant"] == variant) & (c["set"] == set_name) & (c["mode"] == "all") & (c["index"] == "all")]
    return int(c["n_profiles"].max()) if len(c) else 0


# ------------------------------------------------------------------------------------------- figures
def _map_caption(name: str, what: str) -> str:
    model, _, kind = Path(name).stem.partition("__")
    tag = "own decomposition" if kind == "model" else "across-seed decomposition"
    return f"{_label(model, MODEL_LABEL)} ({tag}): epistemic and aleatoric variance over (dv, s) per data set; {what}."


def _regime_caption(name: str, what: str) -> str:
    model = Path(name).stem.partition("__")[2]
    return f"{_label(model, MODEL_LABEL)}: predicted variance and squared residual per regime relative to steady following; {what}."


def _variant_rank(path: Path) -> tuple[int, str]:
    """Sort key of an E3 figure ``<figure>__<variant>.png``: recorded signals first, then harmonised, then the rest."""
    variants = list(VARIANT_LABEL)
    v = path.stem.partition("__")[2]
    return (variants.index(v) if v in variants else len(variants), path.name)


def expected_f2(root: Path) -> set[str]:
    """F2 figures an E1 root must have: the shift figures when its ``h31_verdicts.csv`` has a shift pair, the
    controls when it has a placebo or cross-check pair (every figure when the table is absent or has no ``pair``
    column). A pooled root of E6 has no shift pair (design.md 16.1), so its missing shift figures are expected."""
    everything = {"shift_ratios", "shift_ratios_matched", "shift_ratios_controls"}
    path = root / "tables" / "h31_verdicts.csv"
    frame = _read(path) if path.exists() else pd.DataFrame()
    if "pair" not in frame.columns:
        return everything
    pairs = set(frame["pair"].dropna().astype(str))
    out = {"shift_ratios", "shift_ratios_matched"} if "shift" in pairs else set()
    return out | ({"shift_ratios_controls"} if pairs & {"placebo", "cross_check"} else set())


def collect_figures(ctx: Context) -> list[Figure]:
    """The figures F1-F7 in the roots, with their report names and captions (the E5 and E6 roots, when they exist, in
    F2-F4 only)."""
    figs: list[Figure] = []
    for key, (variant, target) in e1_roots(ctx).items():
        root = ctx.roots.get(key)
        if root is None:
            continue
        what = f"{TARGET_LABEL[target]}, {VARIANT_LABEL[variant]}"
        src = root / "figures" / "maps"
        if key in E1_MATCHED or key in E1_POOLED:
            pass  # F1 holds the maps of the main roots only
        elif src.is_dir() and any(src.glob("*.png")):
            figs.append(Figure("F1", f"f1_maps__{key}", src, f"Uncertainty maps per model and data set; {what}",
                               {p.name: _map_caption(p.name, what) for p in sorted(src.glob("*.png"))}))  # fmt: skip
        else:
            ctx.missing.append(f"{key}: figures/maps")
        for stem, text in (("shift_ratios", "Variance ratios shifted / in-domain (epistemic, aleatoric) per model and shifted "
                                            "set with 95 % event-bootstrap intervals"),
                           ("shift_ratios_matched", "Variance ratios shifted / in-domain at matched (s, dv) states"),
                           ("shift_ratios_controls", "Variance ratios of the controls: placebo split (highD validation) and "
                                                     "second NGSIM I-80 extraction (FollowNet)")):  # fmt: skip
            p = root / "figures" / f"{stem}.png"
            if p.exists():
                figs.append(Figure("F2", f"f2_{stem}__{key}.png", p, f"{text}; {what}."))
            elif stem in expected_f2(root):
                ctx.missing.append(f"{key}: figures/{stem}.png")
        p = root / "figures" / "nll_by_regime.png"
        if p.exists():
            figs.append(Figure("F3", f"f3_nll_by_regime__{key}.png", p,
                               f"NLL of HeteroMDN and LSTM (constant sigma) by driving regime per data set; {what}."))  # fmt: skip
        else:
            ctx.missing.append(f"{key}: figures/nll_by_regime.png")
        src = root / "figures" / "regimes"
        if src.is_dir() and any(src.glob("*.png")):
            figs.append(Figure("F3", f"f3_regime_variance__{key}", src,
                               f"Predicted variance and squared residual by regime relative to steady following, per model; {what}",
                               {p.name: _regime_caption(p.name, what) for p in sorted(src.glob("*.png"))}))  # fmt: skip
    for key, variant in e2_roots(ctx).items():
        root = ctx.roots.get(key)
        if root is None:
            continue
        paths = sorted((root / "figures").glob("coverage_by_horizon__*.png")) + sorted((root / "figures").glob("width_by_horizon__*.png"))
        if not paths:
            ctx.missing.append(f"{key}: figures/coverage_by_horizon__*.png")
        for p in paths:
            kind, _, rest = p.stem.partition("__")
            target, _, dataset = rest.partition("__")
            where = DATASET_LABEL.get(dataset or "highd", dataset)
            text = ("Coverage of the 90 % gap interval by horizon, raw rollouts and ACI online, per model"
                    if kind == "coverage_by_horizon" else "Width of the 90 % gap interval by horizon, raw and ACI online, per model")
            figs.append(Figure("F4", f"f4_{p.stem}__{key}.png", p,
                               f"{text}; {where}{' test' if where == 'highD' else ''}, {TARGET_LABEL.get(target, target)}, "
                               f"{VARIANT_LABEL[variant]}."))  # fmt: skip
    for key, set_name in E3_SETS.items():
        root = ctx.roots.get(key)
        if root is None:
            continue
        for prefix, group, text in (
            ("crps_bars", "F5", "Platoon amplitude CRPS per driver mode (ACC, human-driven, all profiles), one bar per model "
                                "with its 95 % bootstrap interval, IDM + white noise as the reference line"),
            ("growth", "F6", "Speed standard deviation by vehicle index for the models and the measured platoons, with the "
                             "growth exponents"),
            ("crps_by_index", "F7", "Amplitude CRPS and amplitude-ratio CRPS by vehicle index with 95 % bootstrap intervals"),
        ):  # fmt: skip
            paths = sorted((root / "figures").glob(f"{prefix}__*.png"), key=_variant_rank)
            if not paths:
                ctx.missing.append(f"{key}: figures/{prefix}__*.png")
            for p in paths:
                variant = p.stem.partition("__")[2]
                figs.append(Figure(group, f"{group.lower()}_{p.stem}__{key}.png", p,
                                   f"{text}; {run_label(ctx, key, variant)}, {set_name} set."))  # fmt: skip
    return sorted(figs, key=lambda f: int(f.group[1:]))


# ------------------------------------------------------------------------------------------- LaTeX
_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"$<$", ">": r"$>$", "|": r"$|$", "±": r"$\pm$",
        "→": r"$\rightarrow$", "≥": r"$\geq$", "≤": r"$\leq$", "–": "--", "—": "---", "×": r"$\times$", "β": r"$\beta$",
        "²": r"$^2$", "∞": r"$\infty$", "·": r"$\cdot$", "−": "-"}  # fmt: skip
_NUMERIC = re.compile(r"^[\s\d.,+\-–±\[\]/→×∞*e()=%]*$")


def tex(text: Any) -> str:
    """LaTeX of a plain-text cell: special characters escaped, the symbols of the tables as math, 1.2e-05 as
    1.2 x 10^-5, a sign in front of a number as a math minus (a hyphen between digits stays a range)."""
    text = str(text).replace(">=", "≥").replace("<=", "≤")
    out = "".join(_TEX.get(ch, ch) for ch in text)
    out = re.sub(r"(\d)e([+-]?)0*(\d+)", lambda m: f"{m.group(1)}$\\times10^{{{'-' if m.group(2) == '-' else ''}{m.group(3)}}}$", out)
    return re.sub(r"(?:(?<=[\s\[(/])|^)-(?=[\d.])", "$-$", out)


def _align(column: pd.Series) -> str:
    cells = [str(c) for c in column if str(c) not in ("", DASH)]
    return "r" if cells and all(_NUMERIC.match(c) for c in cells) else "l"


def latex(table: Table) -> str:
    """booktabs LaTeX of ``table.view``: a floating ``table`` with ``tabular`` up to LONG_ROWS body rows, else a
    ``longtable`` with the header repeated on every page."""
    view = table.view.astype(str)
    headers = [tex(h) for h in view.columns]
    spec = "".join(_align(view[c]) for c in view.columns)
    rows = view.to_numpy().tolist()
    body: list[str] = []
    prev: list[str] | None = None
    k = table.group_depth
    for r in rows:
        cells = list(r)
        if prev is not None and k and cells[:k] != prev[:k]:
            body.append(r"\midrule")
        shown = [("" if k and prev is not None and j < k and cells[: j + 1] == prev[: j + 1] else c) for j, c in enumerate(cells)]
        line = " & ".join(tex(c) for c in shown)
        body.append(("{}" if line.startswith("[") else "") + line + r" \\")
        prev = cells
    header = " & ".join(headers) + r" \\"
    label = "tab:" + table.name.replace("_", "-")
    lines = [f"% {table.group}: {table.caption}",
             "% Sources: " + ", ".join(display(p) for p in table.sources),
             r"% Generated by scripts/make_report.py; needs \usepackage{booktabs}" + (r" and \usepackage{longtable}" if len(rows) > LONG_ROWS else "") + "."]  # fmt: skip
    size = r"\footnotesize" if len(headers) >= 7 else r"\small"
    if len(rows) > LONG_ROWS:
        lines += [r"\begingroup", size, rf"\begin{{longtable}}{{{spec}}}", rf"\caption{{{tex(table.caption)}}}\label{{{label}}} \\",
                  r"\toprule", header, r"\midrule", r"\endfirsthead", r"\toprule", header, r"\midrule", r"\endhead",
                  r"\bottomrule", r"\endlastfoot", *body, r"\end{longtable}", r"\endgroup"]  # fmt: skip
    else:
        lines += [r"\begin{table}[htbp]", r"\centering", size, rf"\caption{{{tex(table.caption)}}}", rf"\label{{{label}}}",
                  rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header, r"\midrule", *body, r"\bottomrule", r"\end{tabular}",
                  r"\end{table}"]  # fmt: skip
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------- output
GROUP_TITLE = {
    "T1": "One-step metrics", "T2": "H3.1 and H3.1e: dataset shift and rare states", "T3": "H3.3: regimes",
    "T4": "H3.4: coverage in closed-loop rollouts", "T5": "H3.2 per driver mode", "T6": "Platoon diagnostics",
    "F1": "Uncertainty maps", "F2": "Variance ratios under shift", "F3": "NLL and variance by regime",
    "F4": "Coverage by horizon before and after ACI", "F5": "Platoon amplitude CRPS per driver mode",
    "F6": "Growth along the platoon", "F7": "CRPS by vehicle index",
}  # fmt: skip
_MANAGED_TABLE = re.compile(r"^t\d_.*\.(csv|tex)$")
_MANAGED_FIGURE = re.compile(r"^f\d_")


def build_tables(ctx: Context) -> list[Table]:
    tables: list[Table] = []
    for builder in (t1_tables, t2_tables, t3_tables, t4_tables, t5_tables, t6_tables):
        tables += builder(ctx)
    return tables


def write_tables(tables: Sequence[Table], directory: Path) -> set[str]:
    directory.mkdir(parents=True, exist_ok=True)
    written = set()
    for t in tables:
        t.data.to_csv(directory / f"{t.name}.csv", index=False)
        (directory / f"{t.name}.tex").write_text(latex(t), encoding="utf-8")
        written |= {f"{t.name}.csv", f"{t.name}.tex"}
    return written


def copy_figures(figures: Sequence[Figure], directory: Path) -> set[str]:
    directory.mkdir(parents=True, exist_ok=True)
    written = set()
    for f in figures:
        target = directory / f.name
        if f.members:
            target.mkdir(parents=True, exist_ok=True)
            for name in f.members:
                shutil.copy2(f.source / name, target / name)
            for stale in (p for p in target.iterdir() if p.name not in f.members):
                shutil.rmtree(stale) if stale.is_dir() else stale.unlink()
        else:
            shutil.copy2(f.source, target)
        written.add(f.name)
    return written


def prune(directory: Path, keep: set[str], pattern: re.Pattern[str]) -> list[str]:
    """Remove the outputs of earlier runs (names matching ``pattern``) that this run did not write."""
    removed = []
    for p in sorted(directory.iterdir()) if directory.is_dir() else []:
        if p.name in keep or not pattern.match(p.name):
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()
        removed.append(p.name)
    return removed


def _role(ctx: Context, key: str) -> str:
    """Content of a root in index.md (``ROLE``); the main E3 root names the signal variants beyond the recorded and
    the harmonised ones when its analysis has them (E3-pooled, design.md 17.1)."""
    role = ROLE.get(key, "")
    path = ctx.roots.get(key, Path("-")) / "tables" / "amplitude_crps.csv"
    if key == "e3" and path.exists():
        extra = [v for v in _order(_read(path)["variant"].astype(str), tuple(VARIANT_LABEL)) if v not in ("raw", "harmonised")]
        if extra:
            role = f"E3 platoons, main run (recorded and harmonised signals; further variants: {', '.join(extra)})"
    return role


def index_markdown(ctx: Context, tables: Sequence[Table], figures: Sequence[Figure]) -> str:
    lines = ["# Report tables and figures (milestone M5)", "",
             "Built by `scripts/make_report.py` (docs/design.md, section 14.2) from the tables and figures of the "
             "experiment roots below; no experiment, training or bootstrap is run. Tables: `tables/<name>.csv` (numbers) "
             "and `tables/<name>.tex` (booktabs; `longtable` above "
             f"{LONG_ROWS} rows). Figures: copies named `f<k>_<figure>__<root>.png`; the maps of F1 and the "
             "regime-variance figures of F3 are directories per root.", "",
             "## Sources", "", "| root | path | content |", "|---|---|---|"]  # fmt: skip
    for key in (*ROOT_KEYS, *OPTIONAL_KEYS):
        if key in ctx.roots:
            found = "" if ctx.roots[key].exists() else " (missing)"
            lines.append(f"| {key} | `{display(ctx.roots[key])}`{found} | {_role(ctx, key)} |")
    lines += ["", "## Tables"]
    for group in dict.fromkeys(t.group for t in tables):
        lines += ["", f"### {group} {GROUP_TITLE.get(group, '')}", ""]
        for t in (t for t in tables if t.group == group):
            src = ", ".join(f"`{display(p)}`" for p in dict.fromkeys(t.sources))
            lines.append(f"* `tables/{t.name}.csv`, `tables/{t.name}.tex` ({len(t.view)} rows): {t.caption} Source: {src}.")
    lines += ["", "## Figures"]
    for group in dict.fromkeys(f.group for f in figures):
        lines += ["", f"### {group} {GROUP_TITLE.get(group, '')}", ""]
        for f in (f for f in figures if f.group == group):
            if f.members:
                lines.append(f"* `figures/{f.name}/` ({len(f.members)} figures): {f.caption}. Source: `{display(f.source)}/`.")
                lines += [f"  * `{name}`: {caption}" for name, caption in f.members.items()]
            else:
                lines.append(f"* `figures/{f.name}`: {f.caption} Source: `{display(f.source)}`.")
    if ctx.missing:
        lines += ["", "## Not found", "", *[f"* {m}" for m in dict.fromkeys(ctx.missing)]]
    if ctx.notes:
        lines += ["", "## Notes on the sources", "", *[f"* {n}" for n in ctx.notes]]
    return "\n".join(lines) + "\n"


def make_report(out: str | Path = "artifacts/report", roots: Mapping[str, str | Path] | None = None) -> dict[str, Any]:
    """Every table, figure and the index into ``out`` (module docstring); ``roots`` overrides the default
    ``artifacts/<key>`` per key. The E5 and E6 roots (``OPTIONAL_KEYS``) are used only once they exist with the
    tables of their analysis (``<root>/tables``; a root that holds only the matching file or the runner's output is
    skipped silently, like an absent one: the outputs are then those without it). Returns the tables, the figures,
    what was not found, the notes on the sources and the removed outputs of an earlier run."""
    out = _resolve(out)
    keys = (*ROOT_KEYS, *OPTIONAL_KEYS)
    paths = {k: _resolve(v) for k, v in {**{k: f"artifacts/{k}" for k in keys}, **dict(roots or {})}.items()}
    ctx = Context({k: p for k, p in paths.items() if k not in OPTIONAL_KEYS or (p / "tables").is_dir()})
    tables = build_tables(ctx)
    figures = collect_figures(ctx)
    written = write_tables(tables, out / "tables")
    copied = copy_figures(figures, out / "figures")
    removed = prune(out / "tables", written, _MANAGED_TABLE) + prune(out / "figures", copied, _MANAGED_FIGURE)
    (out / "index.md").write_text(index_markdown(ctx, tables, figures), encoding="utf-8")
    return {"out": out, "tables": tables, "figures": figures, "missing": list(dict.fromkeys(ctx.missing)),
            "notes": ctx.notes, "removed": removed}  # fmt: skip


def listing(result: Mapping[str, Any]) -> str:
    tables, figures = result["tables"], result["figures"]
    n_files = sum(len(f.members) if f.members else 1 for f in figures)
    lines = [f"{display(result['out'])}: {len(tables)} tables (csv + tex), {n_files} figures "
             f"({sum(1 for f in figures if f.members)} directories), index.md"]  # fmt: skip
    for group in dict.fromkeys(t.group for t in tables):
        names = [f"{t.name} ({len(t.view)})" for t in tables if t.group == group]
        lines.append(f"  {group}  " + ", ".join(names))
    for group in dict.fromkeys(f.group for f in figures):
        names = [f"{f.name}/ ({len(f.members)})" if f.members else f.name for f in figures if f.group == group]
        lines.append(f"  {group}  " + ", ".join(names))
    if result["missing"]:
        lines.append("  not found: " + "; ".join(result["missing"]))
    if result["notes"]:
        lines.append(f"  {len(result['notes'])} note(s) on the sources in index.md")
    if result["removed"]:
        lines.append("  removed outputs of an earlier run: " + ", ".join(result["removed"]))
    return "\n".join(lines)


def _parse_roots(items: Sequence[str]) -> dict[str, str]:
    out = {}
    for item in items:
        key, sep, path = item.partition("=")
        if not sep or key not in (*ROOT_KEYS, *OPTIONAL_KEYS):
            raise SystemExit(f"--roots expects key=path with key in {', '.join((*ROOT_KEYS, *OPTIONAL_KEYS))}; got {item!r}")
        out[key] = path
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="artifacts/report", help="output directory (default: artifacts/report)")
    parser.add_argument("--roots", nargs="*", default=[], metavar="KEY=PATH",
                        help=f"override experiment roots; keys: {', '.join((*ROOT_KEYS, *OPTIONAL_KEYS))} (E5 and E6 "
                             "roots: used once their analysis tables exist)")  # fmt: skip
    args = parser.parse_args(argv)
    print(listing(make_report(args.out, _parse_roots(args.roots))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
