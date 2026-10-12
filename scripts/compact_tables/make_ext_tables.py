#!/usr/bin/env python3
"""Compact tables c20-c23 of manuscript v14 from the extension artefacts (extension/docs/PLAN.md), IEEE layout.

    CF_UNC_EXT_ARTIFACTS=<dir> PYTHONUTF8=1 python scripts/compact_tables/make_ext_tables.py [--only c20 c21 ...] [--out DIR]

Public copy of the generator of the extension tables of the manuscript (c20-c24). <dir> holds the extension artefacts in
the layout <root>/tables/ (the published results/extension/ has it: CF_UNC_EXT_ARTIFACTS=results/extension).

c20_cost       extension/artifacts/ext_cost/tables/cost.csv (X2)
c21_follownet  extension/artifacts/ext_followcheck/tables/summary.csv (X5), with the published FollowNet rows
c22_preview    extension/artifacts/ext_e3_preview/tables/preview_sweep.csv (X1; written by extension/scripts/preview_sweep.py)
c23_ngsim      extension/artifacts/ext_e1_ngsim/tables/{one_step_metrics,shift_ratios,h31_summary,h31e_summary}.csv and
               extension/artifacts/ext_e3_ngsim/tables/{h32,collisions}.csv (X6)
A table whose inputs are missing keeps its placeholder file. Every file holds a complete float (table* for the wide
ones), inserted with \\input{tables/<name>} at paragraph level, with the sources in its header comment.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
V14 = ROOT / "results"  # the public copy writes to results/compact_tables by default
# the extension artefacts: extension/artifacts/<root>/tables of the development layout, or the directory named by
# CF_UNC_EXT_ARTIFACTS (a public checkout: results/extension, which holds the same <root>/tables files)
EXT = Path(os.environ.get("CF_UNC_EXT_ARTIFACTS", str(ROOT / "extension/artifacts")))
OUT = ROOT / "results" / "compact_tables"  # --out changes it

LABEL = {
    "idm_white_noise": "IDM + white noise", "idm_white_noise_mle": "IDM + white noise (MLE fit)",
    "idm_ar_noise": "IDM + AR(1) noise", "lstm_const_sigma": r"LSTM, constant $\sigma$", "hetero_mdn": "HeteroMDN",
    "hetero_mdn_mix3": "HeteroMDN, 3 components", "deep_ensemble": "Deep ensemble", "mc_dropout": "MC dropout",
    "gp_residual_idm": "GP residual on IDM", "quantile_pidl": "Quantile PIDL", "diffusion_cf": "Diffusion",
    "diffusion_cf_hist": "Diffusion, history only", "lstm_hetero_frozen": "LSTM, variance head at fixed mean",
}
ORDER = list(LABEL)
SETS = {"highd": "highD", "ngsim": "NGSIM I-80", "ngsim_val": "NGSIM I-80 val.", "ngsim_us101": "NGSIM US-101",
        "waymo": "Waymo", "openacc": "OpenACC", "ngsim_fn": "NGSIM I-80 (FollowNet)"}
PUBLISHED = {"highd": {"GHR": (23.76, 0.0), "IDM": (27.50, 0.0), "NN": (24.06, 39.87), "LSTM": (22.90, 3.72), "DDPG": (38.52, 0.0)},
             "ngsim_fn": {"GHR": (62.18, 0.0), "IDM": (54.97, 0.0), "NN": (29.00, 3.46), "LSTM": (23.86, 17.30), "DDPG": (46.93, 0.0)}}


def num(x, d=2, dash="--"):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return dash
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def sig(x, s=3):
    """Significant digits, like the compact tables (large numbers without decimals)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    ax = abs(x)
    d = 0 if ax >= 100 else (1 if ax >= 10 else 2)
    return num(x, d)


def write(name: str, lines: list[str], caption_start: str, sources: list[str]) -> None:
    head = [f"% tab:{name}: {caption_start}", "% Sources: " + ", ".join(sources),
            "% Written by scripts/compact_tables/make_ext_tables.py; do not edit by hand. Insert with",
            f"% \\input{{tables/{name}}} at paragraph level (this file holds the complete float)."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(head + lines) + "\n", encoding="utf-8")
    print("wrote", name)


def c20_cost() -> None:
    src = EXT / "ext_cost/tables/cost.csv"
    if not src.exists():
        print("c20: missing", src); return
    df = pd.read_csv(src).set_index("label")
    rows = []
    for label in ORDER:
        if label not in df.index:
            continue
        r = df.loc[label]
        n_par = int(r.n_params) if pd.notna(r.n_params) else None
        if label.startswith("idm_"):  # the physical baselines: six IDM parameters plus the fitted noise parameters
            n_par = 6 + (2 if label == "idm_ar_noise" else 1)
        train_s = r.train_s_per_seed
        gpu, cpu = r.get("infer_ms_per_1000_cuda", np.nan), r.get("infer_ms_per_1000_cpu", np.nan)
        cells = [LABEL[label], f"{n_par:,}".replace(",", "\\,") if n_par is not None else "--",
                 "$<1$" if pd.notna(train_s) and train_s < 1 else num(train_s, 0),
                 num(gpu, 2 if gpu < 0.1 else 1), num(cpu, 2 if cpu < 0.1 else (1 if cpu < 10 else 0)),
                 num(r.seconds_per_unit_median, 2), num(r.us_per_vehicle_rollout_s, 1)]
        rows.append(" & ".join(cells) + r" \\")
    cap = ("Computational cost of the laws (extension X2): parameters, training wall time per seed on highD (458\\,000 windows, "
           "mean over five seeds), one-step inference time of the predictive law per 1000 highD test windows on the GPU and on the CPU "
           "(median of repeated calls; the diffusion law samples 32 sequences of 50 DDIM steps per window and is timed on fewer rows), "
           "and the platoon simulation time per unit (one profile, 20 followers, 100 rollouts, 32 for the diffusion law; median over the "
           "platoon cells run with the current engine: the pooled cells and the two highD cells of the smoothing matrix) and per simulated "
           "vehicle-rollout-second (microseconds); one workstation with an NVIDIA RTX 6000 "
           "Ada GPU. The laws without platoon units are those outside the platoon model set.")
    lines = [r"\begin{table*}[t]", rf"\caption{{{cap}\label{{tab:c20_cost}}}}", r"\centering", r"\footnotesize",
             r"\begin{tabularx}{\textwidth}{lCCCCCC}", r"\toprule",
             r"\textbf{Law} & \textbf{Parameters} & \textbf{Training [s / seed]} & \textbf{Inference GPU [ms / 1000 windows]} & "
             r"\textbf{Inference CPU [ms / 1000 windows]} & \textbf{Platoon [s / unit]} & \textbf{Platoon [$\mu$s / vehicle-rollout-s]} \\",
             r"\midrule", *rows, r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    write("c20_cost", lines, "Computational cost of the laws (extension X2)", ["extension/artifacts/ext_cost/tables/cost.csv"])


def c21_follownet() -> None:
    src = EXT / "ext_followcheck/tables/summary.csv"
    if not src.exists():
        print("c21: missing", src); return
    df = pd.read_csv(src)
    if df.n_events.max() < 1000:
        print("c21: the summary is a smoke test (", int(df.n_events.max()), "events); skipped"); return
    rows = []
    for label in ORDER:
        sub = df[df.label == label]
        if sub.empty:
            continue
        cells = [LABEL[label]]
        for s in ("highd", "ngsim_fn"):
            r = sub[sub.set == s]
            if r.empty:
                cells += ["--", "--"]
            else:
                r = r.iloc[0]
                fixed = abs(r.mse_s_max - r.mse_s_min) < 1e-9  # seed-independent laws: one value
                cells += [sig(r.mse_s) if fixed else f"{sig(r.mse_s)} ({sig(r.mse_s_min)}--{sig(r.mse_s_max)})",
                          num(r.collision_permille, 1) if fixed else f"{num(r.collision_permille, 1)} ({num(r.collision_min, 1)}--{num(r.collision_max, 1)})"]
        rows.append(" & ".join(cells) + r" \\")
    pub = []
    for m in ("GHR", "IDM", "NN", "LSTM", "DDPG"):
        h, n = PUBLISHED["highd"][m], PUBLISHED["ngsim_fn"][m]
        pub.append(f"{m} (published) & {num(h[0])} & {num(h[1])} & {num(n[0])} & {num(n[1])} \\\\")
    n_h = int(df[df.set == "highd"].n_events.max()); n_n = int(df[df.set == "ngsim_fn"].n_events.max())
    cap = ("Implementation check under the FollowNet protocol (extension X5): closed-loop simulation of every event behind its recorded "
           "leader (semi-implicit Euler at 10~Hz, acceleration clipped to $[-8, 4]$~m/s$^2$, 3~s observed, collision = gap $\\leq 0$) with the "
           "point prediction of each law (mean; median of the quantile law; mean of the 32 sampled 5~s plans of the diffusion laws, re-planned "
           "every 5~s, with the recorded leader future for Diffusion and without it for the history-only twin); MSE of the spacing per event "
           "averaged over events [m$^2$] and collision rate [per mille], mean over the five training seeds (min--max), on the highD test fold "
           f"({n_h} FollowNet highD events, driver-disjoint from training) and on the FollowNet NGSIM I-80 events ({n_n}, a shifted set). "
           "The published rows are FollowNet's Tables 2 and 3 (its own split and 15~s events): an anchor for the order of magnitude, "
           "not a matched comparison.")
    lines = [r"\begin{table*}[t]", rf"\caption{{{cap}\label{{tab:c21_follownet}}}}", r"\centering", r"\footnotesize",
             r"\begin{tabularx}{\textwidth}{lCCCC}", r"\toprule",
             r" & \multicolumn{2}{c}{\textbf{highD test fold}} & \multicolumn{2}{c}{\textbf{NGSIM I-80 (FollowNet)}} \\",
             r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
             r"\textbf{Law} & \textbf{MSE spacing} & \textbf{Collisions, per mille} & \textbf{MSE spacing} & \textbf{Collisions, per mille} \\",
             r"\midrule", *rows, r"\midrule", *pub, r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    write("c21_follownet", lines, "Implementation check under the FollowNet protocol (extension X5)",
          ["extension/artifacts/ext_followcheck/tables/summary.csv", "extension/artifacts/ext_followcheck/tables/follownet_check.csv"])


def c22_preview() -> None:
    src = EXT / "ext_e3_preview/tables/preview_sweep.csv"
    if not src.exists():
        print("c22: missing", src); return
    df = pd.read_csv(src)
    sub = df[(df.set == "extended") & (df.window == "response")]
    bases = [("raw", "highD"), ("pool", "pooled"), ("lodo_openacc", "OpenACC held out")]
    cells_order = [("horizon", 0.0, "$h = 0$ (constant leader)"), ("horizon", 0.5, "$h = 0.5$~s"), ("horizon", 1.0, "$h = 1$~s"),
                   ("horizon", 2.0, "$h = 2$~s"), ("horizon", 3.0, "$h = 3$~s"), ("horizon", 5.0, "$h = 5$~s (full plan)"),
                   ("noise", 0.25, "noise 0.25~m/s"), ("noise", 0.5, "noise 0.5~m/s"), ("noise", 1.0, "noise 1.0~m/s"),
                   ("delay", 0.5, "delay 0.5~s"), ("delay", 1.0, "delay 1.0~s"),
                   ("trained", 1.0, "trained at 1~s"), ("trained", 2.0, "trained at 2~s")]
    rows = []
    short = {"supported": "supp.", "refuted": "ref.", "inconclusive": "inc."}
    for kind, value, text in cells_order:
        for mode in ("ACC", "Human"):
            cells = [text if mode == "ACC" else "", mode]
            found = False
            for base, _ in bases:
                r = sub[(sub.base == base) & (sub.kind == kind) & (np.isclose(sub.value, value)) & (sub["mode"] == mode)]
                if r.empty:
                    cells += ["--", "--"]
                else:
                    r = r.iloc[0]; found = True
                    cells.append(f"{num(r.reduction)} [{num(r.lo_cluster)}, {num(r.hi_cluster)}] {short.get(r.verdict_cluster, r.verdict_cluster)}")
                    cells.append(num(r.collided_mean, 2) if mode == "ACC" else "")  # the collision share is over all 60 profiles
            if found:
                rows.append(" & ".join(cells) + r" \\")
    cap = ("Preview sweep (extension X1): reduction of the amplitude CRPS of the diffusion law against the IDM + white noise reference of its "
           "cell, extended set of leader profiles, response window, per driver mode (47 ACC and 12 human-driven platoons), with the 95\\% "
           "file-cluster interval and its verdict (supp.: lower bound $\\geq 0.20$; ref.: upper bound $< 0.05$; inc.: otherwise), and the mean "
           "share of collided rollouts over the 20 followers and all 60 profiles (coll., on the ACC line of each row); by training regime (highD, recorded signals, leader window 11; pooled and "
           "OpenACC held out, matched signals, leader window 61). Rows: the preview horizon $h$ handed at every re-plan on the stored "
           "checkpoints (the last seen value held beyond $h$), i.i.d.\\ noise and staleness at the full horizon, and the laws trained at a "
           "budget of 1 and 2~s on the pooled roots.")
    lines = [r"\begin{table*}[t]", rf"\caption{{{cap}\label{{tab:c22_preview}}}}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabularx}{\textwidth}{llCcCcCc}", r"\toprule",
             r" & & \multicolumn{2}{c}{\textbf{highD}} & \multicolumn{2}{c}{\textbf{pooled}} & \multicolumn{2}{c}{\textbf{OpenACC held out}} \\",
             r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
             r"\textbf{Preview} & \textbf{Mode} & \textbf{reduction [file interval]} & \textbf{coll.} & \textbf{reduction [file interval]} & \textbf{coll.} & \textbf{reduction [file interval]} & \textbf{coll.} \\",
             r"\midrule", *rows, r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    write("c22_preview", lines, "Preview sweep (extension X1)", ["extension/artifacts/ext_e3_preview/tables/preview_sweep.csv"])


def c23_ngsim() -> None:
    e1 = EXT / "ext_e1_ngsim/tables"
    e3 = EXT / "ext_e3_ngsim/tables"
    if not (e1 / "one_step_metrics.csv").exists():
        print("c23: missing", e1 / "one_step_metrics.csv"); return
    om = pd.read_csv(e1 / "one_step_metrics.csv")
    om = om[(om.subset == "all") & (om.n_seeds == om.n_seeds.max())]
    sets = [s for s in ("ngsim", "highd", "ngsim_us101", "waymo", "openacc") if s in set(om.dataset)]
    def summary(name: str) -> pd.DataFrame | None:
        """One row per model of an H3.1 / H3.1e summary table: the row of the model's own epistemic part (kind
        ``model``) when it has one, else its first row (the across-seed proxy)."""
        if not (e1 / name).exists():
            return None
        t = pd.read_csv(e1 / name)
        if "kind" in t.columns:
            t = pd.concat([t[t.kind == "model"], t[t.kind != "model"]]).drop_duplicates("model", keep="first")
        return t.set_index("model")

    h31, h31e = summary("h31_summary.csv"), summary("h31e_summary.csv")
    sr = pd.read_csv(e1 / "shift_ratios.csv") if (e1 / "shift_ratios.csv").exists() else None
    h32 = pd.read_csv(e3 / "h32.csv") if (e3 / "h32.csv").exists() else None
    col = pd.read_csv(e3 / "collisions.csv") if (e3 / "collisions.csv").exists() else None
    short = {"supported": "supp.", "refuted": "ref.", "inconclusive": "inc."}
    rows = []
    for label in ORDER:
        sub = om[om.model == label]
        if sub.empty:
            continue
        cells = [LABEL[label]]
        for s in sets:
            r = sub[sub.dataset == s]
            cells.append("--" if r.empty else f"{sig(r.nll.iloc[0])} / {num(r.picp90.iloc[0])}")
        # epistemic ratio range over the counted pairs (highd, us101, waymo, openacc)
        if sr is not None:
            base = sr[(sr.model == label) & (sr.pair == "shift") & (sr.subset == "all")]
            if "kind" in base.columns:
                base = base[base.kind == "model"]
            e = base[base.quantity == "epistemic"]
            a = base[base.quantity == "aleatoric"]
            cells.append("--" if e.empty else f"{sig(e.estimate.min())}--{sig(e.estimate.max())}")
            if a.empty:
                cells.append("--")
            elif abs(a.estimate.max() - a.estimate.min()) < 1e-6 and abs(a.estimate.max() - 1.0) < 1e-6:
                cells.append("1 (const.)")
            else:
                cells.append(f"{sig(a.estimate.min())}--{sig(a.estimate.max())}")
        else:
            cells += ["--", "--"]
        v1 = short.get(h31.loc[label, "overall"], "--") if h31 is not None and label in h31.index else "--"
        v2 = short.get(h31e.loc[label, "overall"], "--") if h31e is not None and label in h31e.index else "--"
        cells.append(f"{v1} / {v2}")
        rows.append(" & ".join(cells) + r" \\")
    set_heads = " & ".join(rf"\textbf{{{SETS.get(s, s)}}}" for s in sets)
    cap = ("Second training domain (extension X6), one-step level: every law trained on the NGSIM I-80 reconstruction (recorded signals, "
           "target $a$, five seeds, the IDM calibrated on its training folds). One-step NLL [nats] / PICP of the 90\\% interval on the NGSIM "
           "I-80 test fold (in-domain) and on the shifted sets; range over the four counted shift pairs (highD, US-101, Waymo, OpenACC) of the "
           "ratio of the mean epistemic (laws with an epistemic part of their own) and aleatoric variance, shifted set over the in-domain test "
           "fold; and the overall H3.1 / H3.1e verdicts (the across-seed proxy for the laws without an epistemic part).")
    lines = [r"\begin{table*}[t]", rf"\caption{{{cap}\label{{tab:c23_ngsim}}}}", r"\centering", r"\footnotesize",
             rf"\begin{{tabularx}}{{\textwidth}}{{l{'C' * len(sets)}CCc}}", r"\toprule",
             rf" & \multicolumn{{{len(sets)}}}{{c}}{{\textbf{{One-step NLL / PICP}}}} & \multicolumn{{2}}{{c}}{{\textbf{{Shift ratios}}}} & "
             rf"\textbf{{H3.1 / H3.1e}} \\",
             rf"\cmidrule(lr){{2-{1 + len(sets)}}} \cmidrule(lr){{{2 + len(sets)}-{3 + len(sets)}}}",
             rf"\textbf{{Law}} & {set_heads} & \textbf{{epi.}} & \textbf{{alea.}} & \\",
             r"\midrule", *rows, r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    write("c23_ngsim", lines, "Second training domain (extension X6), one-step level",
          [f"extension/artifacts/ext_e1_ngsim/tables/{f}" for f in ("one_step_metrics.csv", "shift_ratios.csv", "h31_summary.csv", "h31e_summary.csv")])
    # the platoon table of the second domain
    if h32 is None:
        print("c24: missing", e3 / "h32.csv"); return
    gr = pd.read_csv(e3 / "growth.csv") if (e3 / "growth.csv").exists() else None
    prow = []
    for label in ORDER:
        base_rows = h32[(h32.variant == "raw") & (h32.model == label) & (h32.quantity == "amplitude") & (h32.set == "extended")
                        & (h32.window == "response") & (h32.reference == "idm_white_noise")]
        if base_rows.empty:
            continue
        cells = [LABEL[label]]
        for mode in ("ACC", "Human"):
            r = base_rows[base_rows["mode"] == mode]
            cells.append("--" if r.empty else f"{sig(r.reduction.iloc[0])} [{sig(r.lo_cluster.iloc[0])}, {sig(r.hi_cluster.iloc[0])}] {short.get(r.verdict_cluster.iloc[0], '')}")
        r = base_rows[base_rows["mode"] == "ACC"]
        cells.append("--" if r.empty else num(r.crps_model.iloc[0]))
        for idx in ("1", "20", "any"):
            c = col[(col.variant == "raw") & (col.model == label) & (col.set == "extended") & (col["index"].astype(str) == idx)] if col is not None else pd.DataFrame()
            cells.append("--" if c.empty else num(c.collided.iloc[0], 2))
        g = gr[(gr.variant == "raw") & (gr.model == label) & (gr.set == "extended") & (gr.window == "profile") & (gr["index"] == 1)] if gr is not None else pd.DataFrame()
        cells.append("--" if g.empty else f"{num(g.beta.iloc[0])} [{num(g.beta_lo_cluster.iloc[0])}, {num(g.beta_hi_cluster.iloc[0])}]")
        prow.append(" & ".join(cells) + r" \\")
    ref = h32[(h32.variant == "raw") & (h32.model == "diffusion_cf") & (h32.quantity == "amplitude") & (h32.set == "extended") & (h32.window == "response") & (h32["mode"] == "ACC") & (h32.reference == "idm_white_noise")]
    ref_crps = num(ref.crps_reference.iloc[0]) if not ref.empty else "--"
    cap2 = ("Second training domain (extension X6), platoon level: the platoon experiment with the NGSIM-trained checkpoints, leader window 11 as "
            "recorded (the window-61 cells change no verdict and are in Supplementary File S3). Reduction of the amplitude CRPS against the "
            f"NGSIM-calibrated IDM + white noise of the cell (ACC reference {ref_crps}~m/s) on the ACC and the human-driven platoons with the "
            "file-cluster interval and verdict; the ACC amplitude CRPS [m/s]; the share of rollouts in which the first follower, the twentieth "
            "follower and any follower collided; and the growth exponent $\\beta$ of the speed standard deviation along the platoon over the 20 "
            "simulated vehicles, profile window, with its file-cluster interval (measured platoons: 0.12 over all profiles, 0.085 on the common support).")
    lines2 = [r"\begin{table*}[t]", rf"\caption{{{cap2}\label{{tab:c24_ngsim_platoon}}}}", r"\centering", r"\footnotesize",
              r"\begin{tabularx}{\textwidth}{lCCcccc c}", r"\toprule",
              r"\textbf{Law} & \textbf{ACC reduction [file interval]} & \textbf{Human reduction [file interval]} & \textbf{ACC CRPS} & "
              r"\textbf{coll. 1} & \textbf{coll. 20} & \textbf{coll. any} & \textbf{growth $\beta$} \\",
              r"\midrule", *prow, r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    write("c24_ngsim_platoon", lines2, "Second training domain (extension X6), platoon level",
          [f"extension/artifacts/ext_e3_ngsim/tables/{f}" for f in ("h32.csv", "collisions.csv", "growth.csv")])


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=None, help="directory of the table files (default: the tables/ of this paper version)")
    args = ap.parse_args()
    if args.out is not None:
        OUT = args.out
    for name, fn in (("c20", c20_cost), ("c21", c21_follownet), ("c22", c22_preview), ("c23", c23_ngsim)):
        if args.only is None or name in args.only:
            fn()


if __name__ == "__main__":
    main()
