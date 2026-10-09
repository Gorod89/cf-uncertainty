"""Compact markdown tables of experiment E1 (from the CSV tables of analyze_e1.py / analyze_e1_extra.py).

    python scripts/summarize_e1.py --root artifacts/e1 > artifacts/e1/tables/summary.md
    python scripts/summarize_e1.py --root artifacts/e6_pool > artifacts/e6_pool/tables/summary.md   # E6 (pooled domain)

A root trained on a pooled domain (design.md 16.1) states its training data sets and the in-domain reference of the
analysis (``pool_test``) in the header; the pool is the first column of the tables.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from cf_unc.maps.analysis import h31e_tables
from cf_unc.maps.analysis import load_config as load_maps_config
from cf_unc.maps.io import root_target

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("idm_white_noise", "idm_white_noise_mle", "lstm_const_sigma", "hetero_mdn", "hetero_mdn_mix3", "deep_ensemble",
          "mc_dropout", "gp_residual_idm", "quantile_pidl", "diffusion_cf")  # fmt: skip
DATASETS = ("pool_test", "highd", "highd_val", "ngsim", "ngsim_us101", "ngsim_fn", "waymo", "openacc")
REGIMES = ("congested", "braking", "approaching", "free", "steady", "other")
TARGET_TEXT = {"a": "one-step acceleration a[t]", "mean_1s": "mean acceleration over the next 1 s, (v[t+10] - v[t]) / (10 DT)"}


def num(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.{digits}g}"


def ci(est: float, lo: float, hi: float, digits: int = 3) -> str:
    if not (isinstance(est, float) and math.isfinite(est)):
        return "-"
    return f"{num(est, digits)} [{num(lo, digits)}, {num(hi, digits)}]"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def ordered(values, order) -> list[str]:
    present = set(values)
    return [v for v in order if v in present] + sorted(present - set(order))


def root_signal(root: Path) -> str:
    """Signal variant of the windows tables of an experiment root (``spec.signal`` of the first meta; absent: raw;
    ``matched`` when a meta records a matched window, design.md 15.1)."""
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "windows").glob("*_meta.json"))]
    if any("match" in meta for meta in metas):
        return "matched"
    for meta in metas:
        return str((meta.get("spec") or {}).get("signal", "raw"))
    return "raw"


def training_note(root: Path) -> str | None:
    """Header line of a root trained on a pooled domain (``pool`` of ``train_density.npz``, design.md 16.1): its data
    sets, caps and the in-domain reference of the last analysis; None for one training data set."""
    import numpy as np

    path = root / "train_density.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        pool = json.loads(str(z["pool"])) if "pool" in z.files else None
    if not pool:
        return None
    info_path = root / "tables" / "analysis_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}

    def cap(v: object) -> str:
        return "all" if v is None else f"{int(v):,}"  # type: ignore[call-overload]

    shift = info.get("shift")
    pairs = "" if shift is None else (f"; shift pairs: {', '.join(shift)}" if shift else "; no shift pair")
    return (f"Training: pooled domain of the training folds of {', '.join(pool.get('data') or [])} (design.md 16.1; at most "
            f"{cap(pool.get('max_windows_per_dataset'))} one-step / {cap(pool.get('max_sequences_per_dataset'))} sequence "
            f"windows per data set). In-domain reference: `{info.get('in_domain', '-')}`{pairs}.")  # fmt: skip


def h31e_tables_of(t: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """H3.1e verdicts and summary of a tables directory: the CSVs of analyze_e1.py, or, for a root analysed before
    H3.1e existed, derived from ``rare_concentration.csv`` and ``h31_verdicts.csv`` with the analysis config."""
    if (t / "h31e_verdicts.csv").exists() and (t / "h31e_summary.csv").exists():
        return pd.read_csv(t / "h31e_verdicts.csv"), pd.read_csv(t / "h31e_summary.csv"), "h31e_verdicts.csv"
    info_path = t / "analysis_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    stored = info.get("config") or {}
    conf = load_maps_config({k: v for k, v in stored.items() if k in ("h31", "h31e")})
    verdicts, summary = h31e_tables(pd.read_csv(t / "rare_concentration.csv"), pd.read_csv(t / "h31_verdicts.csv"), conf,
                                    str(info.get("in_domain", "highd")))  # fmt: skip
    return verdicts, summary, "derived from rare_concentration.csv and h31_verdicts.csv"


def h31e_section(t: Path) -> list[str]:
    """Markdown of hypothesis H3.1e (design.md, section 12): summary per model and kind, verdicts per pair."""
    verdicts, summary, source = h31e_tables_of(t)
    out = ["### H3.1e: epistemic part only (design.md, section 12)", "",
           "Concentration: epistemic variance inside / outside the lowest 10 % of the training mass (rare_mass) on the "
           "in-domain test set, lower bound > 1. Growth: lower bound of R_epi >= 3 on at least two counted pairs; refuted "
           f"when an upper bound is < 1.5 (or the concentration interval lies below 1). Main verdict: decomposition `model` "
           f"(own epistemic output). Source: {source}.", ""]  # fmt: skip
    if summary.empty:
        return out + ["(no verdict rows)", ""]
    summary = summary.assign(_m=pd.Categorical(summary.model, ordered(summary.model, MODELS), ordered=True))
    summary = summary.sort_values(["_m", "kind"])
    main = summary[summary["main"].astype(str) == "True"]
    if len(main):
        out += ["Main verdict: " + ", ".join(f"{r.model} {r.overall} (point: {r.overall_point})" for r in main.itertuples()), ""]
    rows = [[str(r.model), str(r.kind), ci(float(r.conc), float(r.conc_lo), float(r.conc_hi)), str(r.verdict_conc_ci),
             str(int(r.n_pairs)), str(int(r.n_supported)), str(int(r.n_refuted)), str(int(r.n_inconclusive)), str(r.overall),
             str(r.overall_point), "" if pd.isna(r.placebo_verdicts) else str(r.placebo_verdicts),
             "" if pd.isna(r.cross_check_verdicts) else str(r.cross_check_verdicts)] for r in summary.itertuples()]  # fmt: skip
    out += [table(["model", "decomposition", "concentration (rare_mass)", "concentration verdict", "pairs", "supported",
                   "refuted", "inconclusive", "overall (CI)", "overall (point)", "placebo", "cross-check"], rows), ""]  # fmt: skip
    ver = verdicts[verdicts.subset.astype(str) == "all"]
    rows = []
    for m in ordered(ver.model, MODELS):
        for k in ("model", "seed_ensemble"):
            for d in ordered(ver.dataset, DATASETS):
                r = ver[(ver.model == m) & (ver.kind == k) & (ver.dataset == d)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append([m, k, d, str(r.pair), ci(float(r.r_epi), float(r.r_epi_lo), float(r.r_epi_hi)), str(r.verdict_ci),
                             str(r.verdict_point)])  # fmt: skip
    out += [table(["model", "decomposition", "data set", "pair", "R epistemic", "growth (CI)", "growth (point)"], rows), ""]
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="artifacts/e1")
    args = parser.parse_args(argv)
    root = Path(args.root) if Path(args.root).is_absolute() else ROOT / args.root
    t = root / "tables"
    target = root_target(root)
    signal = root_signal(root)
    note = {"raw": ".", "harmonised": " (one signal processing for every data set, design.md section 12).",
            "matched": " (harmonised with one window per data set, matched in roughness to highD, design.md section 15)."}  # fmt: skip
    out: list[str] = [f"Target: `{target}`, {TARGET_TEXT.get(target, target)}. Signals: `{signal}`" + note.get(signal, "."), ""]
    pooled = training_note(root)
    if pooled:
        out += [pooled, ""]

    one = pd.read_csv(t / "one_step_metrics.csv")
    one = one[(one.subset == "all") & (one.seed.astype(str) == "mean")]
    models, datasets = ordered(one.model, MODELS), ordered(one.dataset, DATASETS)
    for metric, title, digits in (("nll", "NLL, nats", 3), ("picp90", "PICP of the 90 % interval", 3), ("crps", "CRPS, m/s^2", 3),
                                  ("rmse", "RMSE of the predictive mean, m/s^2", 3)):  # fmt: skip
        piv = one.pivot(index="model", columns="dataset", values=metric)
        rows = [[m, *[num(float(piv.loc[m, d]), digits) if d in piv.columns else "-" for d in datasets]] for m in models]
        out += [f"### {title} (mean over seeds)", "", table(["model", *datasets], rows), ""]
    seeds = one.set_index("model")["n_seeds"].groupby(level=0).max()
    out += ["Seeds per model: " + ", ".join(f"{m} {int(seeds[m])}" for m in models), ""]

    ver = pd.read_csv(t / "h31_verdicts.csv")
    ver = ver[ver.subset == "all"]
    rows = []
    for m in ordered(ver.model, MODELS):
        for k in ("model", "seed_ensemble"):
            for d in ordered(ver.dataset, DATASETS):
                r = ver[(ver.model == m) & (ver.kind == k) & (ver.dataset == d)]
                if r.empty or not math.isfinite(float(r.iloc[0].r_epi)):
                    continue
                r = r.iloc[0]
                rows.append([m, k, d, str(r.pair), ci(r.r_epi, r.r_epi_lo, r.r_epi_hi), ci(r.r_alea, r.r_alea_lo, r.r_alea_hi),
                             ci(r.r_epi_matched, r.r_epi_matched_lo, r.r_epi_matched_hi),
                             ci(r.r_alea_matched, r.r_alea_matched_lo, r.r_alea_matched_hi), str(r.verdict_ci), str(r.verdict_point)])  # fmt: skip
    out += ["### H3.1: variance ratios shifted / in-domain", "",
            table(["model", "decomposition", "data set", "pair", "R epistemic", "R aleatoric", "R epistemic, matched (s, dv)",
                   "R aleatoric, matched", "verdict (CI)", "verdict (point)"], rows), ""]  # fmt: skip
    summ = pd.read_csv(t / "h31_summary.csv")
    rows = [[str(r.model), str(r.kind), str(int(r.n_pairs)), str(int(r.n_supported)), str(int(r.n_refuted)), str(int(r.n_inconclusive)),
             str(r.overall), str(r.overall_point), str(r.placebo_verdicts), str(getattr(r, "cross_check_verdicts", ""))]
            for r in summ.itertuples()]  # fmt: skip
    out += ["### H3.1: summary per model", "",
            table(["model", "decomposition", "pairs", "supported", "refuted", "inconclusive", "overall (CI)", "overall (point)",
                   "placebo", "cross-check"], rows), ""]  # fmt: skip
    out += h31e_section(t)

    rare = pd.read_csv(t / "rare_concentration.csv")
    rare = rare[(rare.subset == "all") & rare.ratio_epistemic.notna()]
    rows = []
    for m in ordered(rare.model, MODELS):
        for k in ("model", "seed_ensemble"):
            for d in ordered(rare.dataset, DATASETS):
                block = rare[(rare.model == m) & (rare.kind == k) & (rare.dataset == d)].set_index("rule")
                if block.empty:
                    continue
                cells = []
                for rule in ("rare_freq", "rare_mass", "small_gap", "large_dv"):
                    if rule in block.index:
                        r = block.loc[rule]
                        cells += [f"{num(float(r.share_windows), 2)} / {num(float(r.share_mass_epistemic), 2)}",
                                  ci(float(r.ratio_epistemic), float(r.ratio_epistemic_lo), float(r.ratio_epistemic_hi))]  # fmt: skip
                    else:
                        cells += ["-", "-"]
                rows.append([m, k, d, *cells])
    header = ["model", "decomposition", "data set"]
    for rule in ("freq < 1 %", "lowest 10 % mass", "s < 15 m", "|dv| > 3 m/s"):
        header += [f"{rule}: share of windows / of epistemic", f"{rule}: ratio"]
    out += ["### Concentration of the epistemic variance (inside the region / outside)", "", table(header, rows), ""]

    h33 = pd.read_csv(t / "h33_nll.csv")
    h33 = h33[h33.subset == "all"]
    rows = []
    for d in ordered(h33.dataset, DATASETS):
        for g in ("all", *REGIMES):
            r = h33[(h33.dataset == d) & (h33.group == g)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([d, g, str(int(r.n_windows)), ci(r.nll_a, r.nll_a_lo, r.nll_a_hi), ci(r.nll_b, r.nll_b_lo, r.nll_b_hi),
                         ci(r["diff"], r.diff_lo, r.diff_hi), num(float(r.p_holm), 2), "no" if bool(r.ci_overlap) else "yes"])  # fmt: skip
    out += ["### H3.3: NLL of HeteroMDN (a) against LSTMConstSigma (b)", "",
            table(["data set", "regime", "windows", "NLL a", "NLL b", "a - b", "p (Holm)", "intervals separate"], rows), ""]  # fmt: skip

    order = pd.read_csv(t / "regime_ordering.csv")
    order = order[order.subset == "all"]
    for m in [x for x in ("hetero_mdn", "deep_ensemble", "gp_residual_idm") if x in set(order.model)]:
        rows = []
        for d in ordered(order.dataset, DATASETS):
            for q, name in (("var", "predicted variance"), ("mse", "squared residual")):
                block = order[(order.model == m) & (order.dataset == d) & (order.quantity == q)].set_index("regime")
                if block.empty:
                    continue
                cells = []
                for g in [x for x in REGIMES if x != "steady"]:
                    if g in block.index:
                        r = block.loc[g]
                        star = "*" if bool(r.reject) else ""
                        cells.append(ci(float(r.estimate), float(r.lo), float(r.hi)) + star)
                    else:
                        cells.append("-")
                rows.append([d, name, *cells])
        out += [f"### H3.3: variance relative to steady following, {m} (* = Holm-adjusted p < 0.05)", "",
                table(["data set", "quantity", *[x for x in REGIMES if x != "steady"]], rows), ""]  # fmt: skip

    stats = pd.read_csv(t / "regime_stats.csv")
    stats = stats[(stats.subset == "all") & (stats.model == "hetero_mdn")]
    rows = []
    for d in ordered(stats.dataset, DATASETS):
        block = stats[stats.dataset == d].set_index("regime")
        rows.append([d, *[num(float(block.loc[g, "share"]), 2) if g in block.index else "-" for g in REGIMES]])
    out += ["### Share of windows per regime", "", table(["data set", *REGIMES], rows), ""]

    for name, title in (("extra_signal", "Signal: one-step predictability of the acceleration without any model"),
                        ("extra_rare3d_occupancy", "Windows in states that are rare in (s, dv, v)"),
                        ("extra_calibration", "Calibration of the predicted variance"),
                        ("extra_rare3d", "Epistemic variance and rarity in (s, dv, v)"),
                        ("extra_shift_geometric", "Shift ratios: geometric means against means")):  # fmt: skip
        path = t / f"{name}.csv"
        if not path.exists() or path.stat().st_size <= 2:  # an empty table (e.g. no model with an epistemic output)
            continue
        frame = pd.read_csv(path)
        if "model" in frame.columns:
            frame["model"] = pd.Categorical(frame["model"], ordered(frame.model, MODELS), ordered=True)
            frame["dataset"] = pd.Categorical(frame["dataset"], ordered(frame.dataset, DATASETS), ordered=True)
            frame = frame.sort_values(["model", "dataset"])
        rows = [[num(v, 3) if isinstance(v, float) else str(v) for v in r] for r in frame.itertuples(index=False)]
        out += [f"### {title}", "", table(list(frame.columns), rows), ""]

    print("\n".join(out))


if __name__ == "__main__":
    main()
