#!/usr/bin/env python3
"""Compact tables c20-c31 of manuscripts v14 and v15 from the extension artefacts (extension/docs/PLAN.md), IEEE layout.

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
# the experiment artefacts of the article (c25): artifacts/<root>/tables of the development layout, or the directory
# named by CF_UNC_ARTIFACTS (a public checkout: results/controls, which holds the same <root>/tables files)
ART = Path(os.environ.get("CF_UNC_ARTIFACTS", str(ROOT / "artifacts")))
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


def src(p: Path) -> str:
    """A source path for the header comment: relative to the repository when it lies inside it, else as given."""
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


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
                   ("trained", 1.0, "trained at 1~s"), ("trained", 2.0, "trained at 2~s"),
                   # extension S4: the factorial cells and the 2 s budget-trained law under noise or delay
                   ("combo_h2_n05", 2.0, "$h = 2$~s + noise 0.5~m/s"), ("combo_h2_d05", 2.0, "$h = 2$~s + delay 0.5~s"),
                   ("combo_n05_d05", 2.0, "noise 0.5~m/s + delay 0.5~s"), ("combo_h2_n05_d05", 3.0, "$h = 2$~s + noise + delay"),
                   ("trained_noise", 2.0, "trained at 2~s, noise 0.5~m/s"), ("trained_delay", 2.0, "trained at 2~s, delay 0.5~s")]
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
           "checkpoints (the last seen value held beyond $h$), i.i.d.\\ noise and staleness at the full horizon, the laws trained at a "
           "budget of 1 and 2~s on the pooled roots, and (extension S4) the pairwise and triple combinations of horizon 2~s, noise "
           "0.5~m/s and delay 0.5~s and the 2~s budget-trained law under noise or delay.")
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


def c25_scorecard() -> None:
    """Admissibility scorecard (extension S1): one row per law, one column per check, every cell a verdict or a value
    of a stored table (no new number)."""
    short = {"supported": "supp.", "refuted": "ref.", "inconclusive": "inc."}
    paths = {
        "one_step": ART / "e1/tables/one_step_metrics.csv", "h31": ART / "e1/tables/h31_verdicts.csv",
        "h31e": ART / "e1/tables/h31e_verdicts.csv", "h34": ART / "e2/tables/h34_verdicts.csv",
        "collisions": ART / "e3/tables/collisions.csv", "h32": ART / "e3/tables/h32.csv",
        "preview": EXT / "ext_e3_preview/tables/preview_sweep.csv", "cost": EXT / "ext_cost/tables/cost.csv",
    }  # fmt: skip
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print("c25: missing", *missing); return
    t = {k: pd.read_csv(p) for k, p in paths.items()}
    om = t["one_step"]
    om = om[(om.dataset == "highd") & (om.subset == "all") & (om.seed.astype(str) == "mean")].set_index("model")
    h31 = t["h31"]
    h31 = h31[(h31.kind == "model") & (h31.pair == "shift") & (h31.subset == "all")]
    h31e = t["h31e"]
    h31e = h31e[(h31e.kind == "model") & (h31e.pair == "shift") & (h31e.rule == "rare_mass")]
    if "subset" in h31e.columns:
        h31e = h31e[h31e.subset == "all"]
    h34 = t["h34"]
    h34 = h34[(h34.target == "a") & (h34.corrected == "aci_online")]
    col = t["collisions"]
    col = col[(col.variant == "pool") & (col.set == "extended") & (col["index"].astype(str) != "any")]
    col = col.assign(index=pd.to_numeric(col["index"], errors="coerce")).dropna(subset=["index"])
    h32 = t["h32"]
    h32 = h32[(h32.variant == "pool") & (h32.set == "extended") & (h32["mode"] == "ACC") & (h32.window == "response") &
              (h32.quantity == "amplitude") & (h32.reference == "idm_white_noise")].set_index("model")  # fmt: skip
    pv = t["preview"]
    pv = pv[(pv.base == "pool") & (pv.kind == "horizon") & (pv.set == "extended") & (pv["mode"] == "ACC") & (pv.window == "response")]
    cost = t["cost"].set_index("label")
    shifted = ("ngsim", "ngsim_us101", "waymo", "openacc")

    def count(frame: pd.DataFrame, model: str, column: str) -> str:
        sub = frame[(frame.model == model) & (frame.dataset.isin(shifted))]
        if sub.empty:
            return "--"
        v = sub.set_index("dataset")[column].reindex(shifted)
        return f"{int((v == 'supported').sum())} / {int((v == 'refuted').sum())} / {int((v == 'inconclusive').sum())}"

    laws = ["idm_white_noise", "idm_ar_noise", "lstm_const_sigma", "hetero_mdn", "deep_ensemble", "mc_dropout",
            "gp_residual_idm", "quantile_pidl", "diffusion_cf", "diffusion_cf_hist"]  # fmt: skip
    rows = []
    for m in laws:
        cells = [LABEL.get(m, m)]
        cells.append(f"{num(om.loc[m, 'picp90'], 3)} / {num(om.loc[m, 'nll'], 2)}" if m in om.index else "--")
        cells.append(count(h31, m, "verdict_ci"))
        cells.append(count(h31e, m, "verdict_ci"))
        hd = h34[(h34.model == m) & (h34.dataset == "highd")]
        cells.append((short.get(hd.iloc[0].verdict_ci, "--") if not hd.empty else "--") + " / " + count(h34, m, "verdict_ci"))
        c = col[col.model == m]
        cells.append(num(1.0 - float(c.collided.mean()), 2) if not c.empty else "--")
        if m in h32.index:
            r = h32.loc[m]
            cells.append(f"{num(r.reduction)} [{num(r.lo_cluster)}, {num(r.hi_cluster)}] {short.get(r.verdict_cluster, r.verdict_cluster)}")
        else:
            cells.append("--")
        if m == "diffusion_cf" and not pv.empty:
            ok = pv[pv.verdict_cluster == "supported"].sort_values("value")
            cells.append(f"{ok.iloc[0].value:g}~s" if not ok.empty else "none")
        elif m in ("idm_white_noise", "idm_ar_noise", "lstm_const_sigma", "hetero_mdn", "deep_ensemble", "mc_dropout",
                   "gp_residual_idm", "quantile_pidl", "diffusion_cf_hist"):
            cells.append("0 (history)")
        else:
            cells.append("--")
        cells.append(num(cost.loc[m, "us_per_vehicle_rollout_s"], 1) if m in cost.index else "--")
        rows.append(" & ".join(cells) + r" \\")
    cap = ("Admissibility scorecard (extension S1): every cell is a verdict or a value of a stored table. In domain: PICP of the 90\\% "
           "one-step interval and NLL (nats) on the highD test fold (seed mean). H3.1 and H3.1e: the number of shifted sets (NGSIM I-80, "
           "US-101, Waymo, OpenACC) on which the shift rule is supported / refuted / inconclusive (interval verdict). H3.4: the interval verdict "
           "with the ACI-online correction on highD, then supported / refuted / inconclusive over the four shifted sets; its part (iii) holds by "
           "construction (Section~\\ref{sec:res_rollouts}), and every inconclusive cell comes from the event-bootstrap lower bound of the "
           "corrected coverage straddling 0.87, so the column does not rank the laws. Platoon (pooled training, OpenACC seen, leader window 61; "
           "the exploratory follow-up, not a test of H3.2 as prespecified): the share of rollouts in which a given follower does not collide, "
           "averaged over the 20 followers and all 60 profiles (not the share of collision-free platoons; Supplementary Table~S2.3: 19.8\\% for the "
           "diffusion law, 44.3\\% for the GP residual), and the H3.2 reduction of the amplitude CRPS against the IDM + white noise reference on the 47 ACC platoons "
           "(response window, 95\\% file-cluster interval, verdict). Preview: the leader information the law needs at every re-plan "
           "(history only, or the smallest preview horizon at which the H3.2 verdict is supported, extension X1). Cost: platoon "
           "simulation time per vehicle-rollout-second on the GPU (Section~\\ref{sec:res_cost}). The scorecard summarises; it changes no verdict.")
    lines = [r"\begin{table*}[t]", rf"\caption{{{cap}\label{{tab:c25_scorecard}}}}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabularx}{\textwidth}{lccccccCcc}", r"\toprule",
             r"\textbf{Law} & \textbf{PICP / NLL} & \textbf{H3.1} & \textbf{H3.1e} & \textbf{H3.4} & \textbf{Coll.-free} & "
             r"\textbf{H3.2 pooled, ACC} & \textbf{Preview} & \textbf{Cost} \\",
             r" & \textbf{in domain} & \textbf{s / r / i} & \textbf{s / r / i} & \textbf{highD / s / r / i} & \textbf{share} & "
             r"\textbf{reduction [file interval]} & \textbf{need} & \textbf{[$\mu$s]} \\",
             r"\midrule", *rows, r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    write("c25_scorecard", lines, "Admissibility scorecard (extension S1)", [src(p) for p in paths.values()])


def c26_reweight() -> None:
    """State-distribution part of the shift (extension S2): NLL in domain, raw and reweighted, and the share of the gap
    explained by the state distribution, for the laws of the article on the four shifted sets."""
    src = EXT / "ext_reweight/tables/state_reweight.csv"
    bins = EXT / "ext_reweight/tables/bins.json"
    if not src.exists() or not bins.exists():
        print("c26: missing", src); return
    import json as _json
    df = pd.read_csv(src)
    info = _json.loads(bins.read_text(encoding="utf-8"))["sets"]
    laws = ["idm_white_noise", "lstm_const_sigma", "hetero_mdn", "deep_ensemble", "mc_dropout", "gp_residual_idm", "quantile_pidl", "diffusion_cf"]
    sets = [("ngsim", "NGSIM I-80"), ("ngsim_us101", "US-101"), ("waymo", "Waymo"), ("openacc", "OpenACC")]
    rows = []
    for m in laws:
        sub = df[df.model == m].set_index("dataset")
        if sub.empty:
            continue
        cells = [LABEL.get(m, m), num(float(sub.nll_in.iloc[0]), 2)]
        for s, _ in sets:
            if s in sub.index:
                r = sub.loc[s]
                cells.append(f"{sig(r.nll_raw)} / {sig(r.nll_rw)}")
            else:
                cells.append("--")
        rows.append(" & ".join(cells) + r" \\")
    ess = ", ".join(f"{t} {info[s]['ess'] / info[s]['n_windows'] * 100:.1f}\\%" for s, t in sets if s in info)
    cells = df[df.model.isin(laws) & df.dataset.isin([s for s, _ in sets])]
    higher, lower = int((cells.nll_rw > cells.nll_raw).sum()), cells[cells.nll_rw < cells.nll_raw]
    exceptions = "; ".join(f"{LABEL.get(r.model, r.model)} on {dict(sets)[r.dataset]}" for r in lower.itertuples())
    cap = ("State-distribution part of the dataset shift (extension S2): one-step NLL (nats per window, seed means) on the highD test "
           "fold (in domain) and on every shifted set, raw and reweighted to the highD occupancy of the state space by importance "
           "weights over $6 \\times 6 \\times 6$ quantile bins of $(s, \\Delta v, v)$ (raw / reweighted). A reweighted NLL below the raw one "
           f"would attribute the gap to the states visited; it is higher in {higher} of {len(cells)} cells (the exceptions: {exceptions}): "
           "the gap is conditional on the state, not a matter of which states are visited. Effective sample size of the reweighted "
           f"sets: {ess} of their windows.")
    lines = [r"\begin{table}[t]", rf"\caption{{{cap}\label{{tab:c26_reweight}}}}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabularx}{\columnwidth}{lcCCCC}", r"\toprule",
             r"\textbf{Law} & \textbf{highD} & \textbf{NGSIM I-80} & \textbf{US-101} & \textbf{Waymo} & \textbf{OpenACC} \\",
             r"\midrule", *rows, r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    write("c26_reweight", lines, "State-distribution part of the shift (extension S2)",
          ["extension/artifacts/ext_reweight/tables/state_reweight.csv", "extension/artifacts/ext_reweight/tables/bins.json"])


def c27_coverage() -> None:
    """Conditional coverage (extension S5): per law and set, the number of regimes (at least 30 events) outside 0.85--0.95
    at 10 s for the static, CQR and ACI-online intervals, and the ACI coverage of the braking regime."""
    val = EXT / "ext_coverage/tables/conditional_validity.csv"
    reg = EXT / "ext_coverage/tables/coverage_by_regime.csv"
    if not val.exists() or not reg.exists():
        print("c27: missing", val); return
    v = pd.read_csv(val)
    r = pd.read_csv(reg)
    r = r[np.isclose(r.horizon_s, 10.0) & (r.variant == "aci_online") & (r.regime == "braking")].set_index(["model", "dataset"])
    laws = ["idm_white_noise", "idm_ar_noise", "lstm_const_sigma", "hetero_mdn", "deep_ensemble", "mc_dropout", "gp_residual_idm", "quantile_pidl", "diffusion_cf"]
    sets = [("highd", "highD"), ("ngsim", "NGSIM I-80"), ("ngsim_us101", "US-101"), ("waymo", "Waymo"), ("openacc", "OpenACC")]
    rows = []
    for m in laws:
        cells = [LABEL.get(m, m)]
        for s, _ in sets:
            sub = v[(v.model == m) & (v.dataset == s)].set_index("variant")
            if sub.empty:
                cells.append("--"); continue
            n = sub.loc["static", "n_regimes"] if "static" in sub.index else 0
            parts = [f"{int(sub.loc[k, 'n_failing'])}" if k in sub.index else "--" for k in ("static", "cqr", "aci_online")]
            br = r.loc[(m, s), "picp"] if (m, s) in r.index and r.loc[(m, s), "n_events"] >= 30 else float("nan")
            cells.append(f"{'/'.join(parts)} of {int(n)}; {num(br, 2)}")
        rows.append(" & ".join(cells) + r" \\")
    cap = ("Conditional coverage of the 90\\% gap intervals at 10~s (extension S5): per law and set, the number of regimes at the "
           "issue instant (congested, braking, approaching, free, steady, other; those with at least 30 events) whose coverage lies "
           "outside $0.90 \\pm 0.05$ for the static, the CQR and the ACI-online intervals (static/CQR/ACI of the regimes counted), "
           "followed by the ACI-online coverage of the braking regime. ACI restores the marginal coverage on every set; the braking "
           "regime stays below 0.85 on the shifted sets for most laws.")
    lines = [r"\begin{table*}[t]", rf"\caption{{{cap}\label{{tab:c27_coverage}}}}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabularx}{\textwidth}{lCCCCC}", r"\toprule",
             r"\textbf{Law} & \textbf{highD} & \textbf{NGSIM I-80} & \textbf{US-101} & \textbf{Waymo} & \textbf{OpenACC} \\",
             r"\midrule", *rows, r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    write("c27_coverage", lines, "Conditional coverage (extension S5)",
          ["extension/artifacts/ext_coverage/tables/conditional_validity.csv", "extension/artifacts/ext_coverage/tables/coverage_by_regime.csv"])


def c28_string() -> None:
    """String stability (extension S9): geometric mean of the consecutive amplitude ratios over the 20 followers with the
    file-cluster interval and verdict, the head-to-tail ratio, and the ratio on the recorded indices next to the recorded one."""
    src = EXT / "ext_string/tables/string_summary.csv"
    if not src.exists():
        print("c28: missing", src); return
    df = pd.read_csv(src)
    sub = df[(df.root == "e3") & (df["mode"] == "ACC")]
    cells_order = [("raw", "highD"), ("harmonised", "harmonised"), ("pool", "pooled"), ("lodo_openacc", "OpenACC held out")]
    laws = ["idm_white_noise", "idm_ar_noise", "lstm_const_sigma", "deep_ensemble", "gp_residual_idm", "quantile_pidl", "diffusion_cf", "diffusion_cf_hist"]
    short = {"string-stable": "stable", "amplifying": "ampl.", "inconclusive": "inc."}
    rows = []
    rec = sub.recorded_geo_ratio.dropna()
    for m in laws:
        cells = [LABEL.get(m, m)]
        for variant, _ in cells_order:
            r = sub[(sub.variant == variant) & (sub.model == m)]
            if r.empty:
                cells += ["--", "--"]; continue
            r = r.iloc[0]
            cells.append(f"{num(r.geo_ratio, 3)} [{num(r.lo, 3)}, {num(r.hi, 3)}] {short.get(r.verdict, r.verdict)}; {num(r.head_to_tail, 1)}")
            cells.append(num(r.simulated_geo_ratio_recorded_indices, 2))
        rows.append(" & ".join(cells) + r" \\")
    rec_text = f"{rec.iloc[0]:.2f}" if len(rec) else "--"
    cap = ("String stability of the simulated ACC platoons (extension S9, 47 ACC profiles, extended set): geometric mean over the "
           "20 followers of the consecutive amplitude ratios $A_{i+1} / A_i$ of the median sampled amplitude, with its 95\\% "
           "file-cluster interval and verdict (stable: interval below 1; ampl.: above 1; inc.: otherwise), followed by the "
           "head-to-tail ratio $A_{20} / A_{\\mathrm{leader}}$; and the same geometric ratio on the recorded indices (rec.), to be read "
           f"against the recorded platoons, whose ratio on those indices is {rec_text}. By training regime.")
    lines = [r"\begin{table*}[t]", rf"\caption{{{cap}\label{{tab:c28_string}}}}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabularx}{\textwidth}{lCcCcCcCc}", r"\toprule",
             r" & \multicolumn{2}{c}{\textbf{highD}} & \multicolumn{2}{c}{\textbf{harmonised}} & \multicolumn{2}{c}{\textbf{pooled}} & \multicolumn{2}{c}{\textbf{OpenACC held out}} \\",
             r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
             r"\textbf{Law} & \textbf{ratio [interval]; head-to-tail} & \textbf{rec.} & \textbf{ratio [interval]; head-to-tail} & \textbf{rec.} & "
             r"\textbf{ratio [interval]; head-to-tail} & \textbf{rec.} & \textbf{ratio [interval]; head-to-tail} & \textbf{rec.} \\",
             r"\midrule", *rows, r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    write("c28_string", lines, "String stability (extension S9)", ["extension/artifacts/ext_string/tables/string_summary.csv"])


def c29_replan() -> None:
    """Sampler settings and plan sharing (extensions S3 and S8): per pooled regime and cell, the H3.2 reduction on the
    ACC and the human-driven profiles, the collision share, the time per unit and its ratio to the stored setting."""
    src = EXT / "ext_e3_replan/tables/replan_tradeoff.csv"
    if not src.exists():
        print("c29: missing", src); return
    df = pd.read_csv(src)
    short = {"supported": "supp.", "refuted": "ref.", "inconclusive": "inc."}
    rows = []
    for base, text in (("pool", "pooled"), ("lodo_openacc", "OpenACC held out")):
        part = df[df.base == base]
        first = True
        for r in part.itertuples():
            acc = (f"{num(r.reduction_response_ACC)} [{num(r.lo_response_ACC)}, {num(r.hi_response_ACC)}] {short.get(r.verdict_response_ACC, '--')}"
                   if np.isfinite(getattr(r, "reduction_response_ACC", float("nan"))) else "--")
            hum = (f"{num(r.reduction_response_Human)} [{num(r.lo_response_Human)}, {num(r.hi_response_Human)}] {short.get(r.verdict_response_Human, '--')}"
                   if np.isfinite(getattr(r, "reduction_response_Human", float("nan"))) else "--")
            rows.append(" & ".join([text if first else "", f"{r.replan_steps / 10:g}", str(int(r.sampling_steps)), "yes" if r.share else "no",
                                    acc, hum, num(r.collided_mean, 2), num(r.seconds_per_unit, 1), num(r.cost_ratio, 2)]) + r" \\")
            first = False
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()
    cap = ("Sampler settings and plan sharing of the diffusion law (extensions S3 and S8): per pooled regime, the re-planning interval "
           "(s), the number of DDIM sampling steps and whether the followers receive their predecessor's sampled plan instead of its "
           "realised future; the reduction of the amplitude CRPS against the IDM + white noise reference (extended set, response "
           "window, 95\\% file-cluster interval, verdict) on the ACC and on the human-driven profiles, the mean share of collided "
           "rollouts over the 20 followers and all 60 profiles, the simulation time per unit (one profile, 20 followers, 32 "
           "rollouts) and its ratio to the stored setting (5~s, 50 steps, no sharing), which repeats the stored cell. The cells without "
           "sharing ran on a GPU shared with another workload and are comparable with each other; the plan-sharing cells ran later "
           "under a lighter load, so their times are not comparable with the other rows (sharing adds no sampler work).")
    lines = [r"\begin{table*}[t]", rf"\caption{{{cap}\label{{tab:c29_replan}}}}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabularx}{\textwidth}{lcccCCccc}", r"\toprule",
             r"\textbf{Regime} & \textbf{Re-plan [s]} & \textbf{Steps} & \textbf{Shared} & \textbf{ACC reduction [file interval]} & "
             r"\textbf{Human reduction [file interval]} & \textbf{Coll.} & \textbf{s / unit} & \textbf{Cost ratio} \\",
             r"\midrule", *rows, r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    write("c29_replan", lines, "Sampler settings and plan sharing (extensions S3 and S8)", ["extension/artifacts/ext_e3_replan/tables/replan_tradeoff.csv"])


def c30_moderef() -> None:
    """H3.2 against the references calibrated per driver mode (extension S6): per law, mode and cell, the reduction against
    the mode-matched reference next to the stored verdict."""
    src = EXT / "ext_e3_idmref/tables/h32_mode_reference.csv"
    if not src.exists():
        print("c30: missing", src); return
    df = pd.read_csv(src)
    df = df[df.window == "response"]
    short = {"supported": "supp.", "refuted": "ref.", "inconclusive": "inc."}
    cells_order = [("raw", "highD"), ("harmonised", "harmonised"), ("pool", "pooled"), ("lodo_openacc", "OpenACC held out")]
    laws = ["idm_ar_noise", "lstm_const_sigma", "deep_ensemble", "gp_residual_idm", "quantile_pidl", "diffusion_cf", "diffusion_cf_hist"]
    rows = []
    for m in laws:
        for mode in ("ACC", "Human"):
            cells = [LABEL.get(m, m) if mode == "ACC" else "", mode]
            found = False
            for variant, _ in cells_order:
                r = df[(df.model == m) & (df["mode"] == mode) & (df.variant == variant)]
                if r.empty:
                    cells.append("--"); continue
                r = r.iloc[0]; found = True
                cells.append(f"{num(r.reduction_matched)} [{num(r.lo_matched)}, {num(r.hi_matched)}] {short.get(r.verdict_matched, '--')} "
                             f"({short.get(r.verdict_stored, '--')})")
            if found:
                rows.append(" & ".join(cells) + r" \\")
    n_rob = int(df.reference_robust.sum())
    cap = ("H3.2 against the references calibrated per driver mode (extension S6): reduction of the amplitude CRPS of every stored law "
           "against the IDM + white noise reference calibrated on the OpenACC events of the profiles' driver mode (ACC profiles: the "
           "ACC reference; human-driven profiles: the human reference), extended set, response window, 95\\% file-cluster interval and "
           "verdict, with the stored verdict against the highD reference in parentheses; by training regime. Reference-robust (the two "
           f"verdicts agree) in {n_rob} of {len(df)} law--mode--cell combinations.")
    lines = [r"\begin{table*}[t]", rf"\caption{{{cap}\label{{tab:c30_moderef}}}}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabularx}{\textwidth}{llCCCC}", r"\toprule",
             r"\textbf{Law} & \textbf{Mode} & \textbf{highD} & \textbf{harmonised} & \textbf{pooled} & \textbf{OpenACC held out} \\",
             r"\midrule", *rows, r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    write("c30_moderef", lines, "H3.2 against the mode-calibrated references (extension S6)", ["extension/artifacts/ext_e3_idmref/tables/h32_mode_reference.csv"])


def c31_closedloop() -> None:
    """Closed-loop fine-tuned networks versus their one-step twins (extension S7): one-step NLL in domain and on OpenACC, the
    H3.4 verdict (highD regime), the platoon H3.2 reduction on the ACC profiles and the collision share, twin -> fine-tuned."""
    d = EXT / "ext_cl_summary/tables"
    paths = {k: d / f"{k}.csv" for k in ("one_step", "e2", "platoon")}
    if not all(p.exists() for p in paths.values()):
        print("c31: missing", d); return
    frames = {}
    for k, p in paths.items():
        try:
            frames[k] = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            frames[k] = pd.DataFrame()
    if any(f.empty or "which" in f.columns and not (f.which == "cl").any() for f in frames.values()):
        print("c31: the fine-tuned rows are not there yet", d); return
    one, e2, pl = frames["one_step"], frames["e2"], frames["platoon"]
    short = {"supported": "supp.", "refuted": "ref.", "inconclusive": "inc."}
    pairs = [("hetero_mdn", "hetero_mdn_cl"), ("lstm_const_sigma", "lstm_const_sigma_cl")]
    regimes = [("highd", "raw", "highD"), ("pool", "pool", "pooled"), ("lodo_openacc", "lodo_openacc", "OpenACC held out")]
    rows = []
    for twin, cl in pairs:
        first = True
        for regime, cell, text in regimes:
            cells = [LABEL.get(twin, twin) if first else "", text]
            first = False
            for ds in ("highd", "openacc"):
                r = one[(one.regime == regime) & (one.twin == twin) & (one.dataset == ds)]
                cells.append(f"{sig(r.nll_twin.iloc[0])} $\\to$ {sig(r.nll_cl.iloc[0])}" if not r.empty else "--")
            if regime == "highd":
                a = e2[(e2.which == "twin") & (e2.model == twin) & (e2.dataset == "highd")]
                b = e2[(e2.which == "cl") & (e2.model == cl) & (e2.dataset == "highd")]
                cells.append(f"{short.get(a.verdict_ci.iloc[0], '--') if not a.empty else '--'} $\\to$ {short.get(b.verdict_ci.iloc[0], '--') if not b.empty else '--'}")
            else:
                cells.append("--")
            a = pl[(pl.which == "twin") & (pl.model == twin) & (pl.variant == cell) & (pl["mode"] == "ACC")]
            b = pl[(pl.which == "cl") & (pl.model == cl) & (pl.variant == cell) & (pl["mode"] == "ACC")]
            fmt = lambda r: f"{num(r.reduction.iloc[0])} [{num(r.lo_cluster.iloc[0])}, {num(r.hi_cluster.iloc[0])}] {short.get(r.verdict_cluster.iloc[0], '--')}" if not r.empty else "--"
            cells.append(f"{fmt(a)} $\\to$ {fmt(b)}")
            cells.append(f"{num(a.collided_mean.iloc[0], 2) if not a.empty else '--'} $\\to$ {num(b.collided_mean.iloc[0], 2) if not b.empty else '--'}")
            rows.append(" & ".join(cells) + r" \\")
    cap = ("One-step networks fine-tuned in closed loop (extension S7), twin $\\to$ fine-tuned, per training regime: one-step NLL "
           "(nats, seed means) on the highD test fold and on OpenACC; the H3.4 verdict of the ACI-corrected rollout intervals on highD "
           "(highD regime); the platoon H3.2 reduction against the IDM + white noise reference on the 47 ACC profiles (extended set, "
           "response window, 95\\% file-cluster interval, verdict) and the mean share of collided rollouts over the 20 followers and "
           "all 60 profiles in the cell of the regime (highD, pooled, OpenACC held out).")
    lines = [r"\begin{table*}[t]", rf"\caption{{{cap}\label{{tab:c31_closedloop}}}}", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabularx}{\textwidth}{llccCCc}", r"\toprule",
             r"\textbf{Law} & \textbf{Regime} & \textbf{NLL highD} & \textbf{NLL OpenACC} & \textbf{H3.4 highD} & "
             r"\textbf{H3.2 ACC reduction [file interval]} & \textbf{Coll.} \\",
             r"\midrule", *rows, r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    write("c31_closedloop", lines, "Closed-loop fine-tuning (extension S7)", [src(p) for p in paths.values()])


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=None, help="directory of the table files (default: the tables/ of this paper version)")
    args = ap.parse_args()
    if args.out is not None:
        OUT = args.out
    for name, fn in (("c20", c20_cost), ("c21", c21_follownet), ("c22", c22_preview), ("c23", c23_ngsim), ("c25", c25_scorecard), ("c26", c26_reweight), ("c27", c27_coverage), ("c28", c28_string), ("c29", c29_replan), ("c30", c30_moderef), ("c31", c31_closedloop)):
        if args.only is None or name in args.only:
            fn()


if __name__ == "__main__":
    main()
