#!/usr/bin/env python3
"""Table c19 of version v8 (review V7-R06): the joint calibration table of the rollout corrections on one label
stream, NGSIM I-80, target a, recorded signals, horizon 10 s: coverage, median finite width of the gap interval,
share of infinite intervals, cold-start coverage (first 50 / 100 events of the stream) and the mean label delay,
for the static correction, the prequential static correction and ACI online (random, chronological, delayed).

    PYTHONUTF8=1 python scripts/compact_tables/make_c19.py [--out DIR]   # writes results/compact_tables/c19_aci_joint.tex (or DIR/c19_aci_joint.tex)

Source: artifacts/e2/tables/coverage_by_horizon.csv (scripts/analyze_e2.py).
"""
from __future__ import annotations

import pathlib

import pandas as pd

import os

ROOT = pathlib.Path(__file__).resolve().parents[2]
ART = pathlib.Path(os.environ.get("CF_UNC_ARTIFACTS", str(ROOT / "artifacts")))
SRC = ART / "e2/tables/coverage_by_horizon.csv"
OUT = ROOT / "results/compact_tables/c19_aci_joint.tex"

MODELS = [("idm_white_noise", "IDM + white noise"), ("idm_ar_noise", "IDM + AR(1) noise"), ("lstm_const_sigma", "LSTM, constant sigma"),
          ("hetero_mdn", "HeteroMDN"), ("deep_ensemble", "Deep ensemble"), ("mc_dropout", "MC dropout"),
          ("gp_residual_idm", "GP residual on IDM"), ("quantile_pidl", "Quantile PIDL"), ("diffusion_cf", "Diffusion")]
METHODS = [("static", "static"), ("static_prequential", "static preq."), ("aci_online", "ACI"),
           ("aci_online_chrono", "ACI chrono."), ("aci_online_delayed", "ACI delayed")]
DATASET, TARGET, HORIZON = "ngsim", "a", 10.0


def fmt(x: float, d: int = 2) -> str:
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=None, help="directory of the table file (default: next to the other compact tables)")
    args = ap.parse_args()
    global OUT
    if args.out is not None:
        OUT = args.out / OUT.name
        OUT.parent.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(SRC)
    d = d[(d.dataset == DATASET) & (d.target == TARGET) & (d.horizon_s == HORIZON)]
    rows = []
    for model, label in MODELS:
        m = d[d.model == model]
        if m.empty:
            continue
        cells = [label]
        for method, _ in METHODS:
            r = m[m.variant == method]
            if r.empty:
                cells += ["--", "--", "--", "--"]
                continue
            r = r.iloc[0]
            inf = r["share_infinite"] * 100
            cold = "--" if pd.isna(r["picp_first50"]) else f"{r['picp_first50']:.2f}"
            cells += [fmt(r["picp"]), f"{r['width_finite_median']:.0f}", f"{inf:.1f}" if inf > 0 else "0", cold]
        rows.append(" & ".join(cells) + r" \\")
    delay = d[d.variant == "aci_online_delayed"]["delay_events_mean"].dropna()
    delay_txt = f"{delay.iloc[0]:.1f}" if len(delay) else "--"
    head = " & ".join([r"\textbf{Model}"] + [f"\\multicolumn{{4}}{{c}}{{\\textbf{{{lab}}}}}" for _, lab in METHODS]) + r" \\"
    sub = " & ".join([""] + [r"cov. & width & inf. & cold" for _ in METHODS]) + r" \\"
    cmid = " ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(METHODS)))
    caption = (
        "Calibration of the rollout corrections on one label stream, NGSIM I-80, recorded signals, target $a$, horizon 10~s: "
        "coverage of the 90\\% gap interval (cov.), median width of the finite intervals (m), share of infinite intervals "
        "(inf., \\%) and the coverage over the first 50 events of the stream (cold start), for the static correction of highD "
        "validation, the static correction recalibrated prequentially on the same labels as ACI, and ACI online in a random "
        "order, in chronological order and with the labels delayed by the horizon (mean delay " + delay_txt + " events). "
        "Infinite intervals count as covering and are excluded from the width; no interval score is reported because it is "
        "infinite for those events. Source: \\texttt{artifacts/e2/tables/coverage\\_by\\_horizon.csv}.")
    body = "\n".join([
        "% tab:c19_aci_joint: " + caption.replace("\n", " "),
        "% Sources: artifacts/e2/tables/coverage_by_horizon.csv",
        "% Written by scripts/compact_tables/make_c19.py; do not edit by hand.",
        r"\begin{table}[H]",
        r"\caption{" + caption + r"\label{tab:c19_aci_joint}}",
        r"\begin{adjustwidth}{-\extralength}{0cm}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabularx}{\fulllength}{l" + "CCCC" * len(METHODS) + "}",
        r"\toprule",
        head, cmid, sub,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{adjustwidth}",
        r"\end{table}",
        "",
    ])
    OUT.write_text(body, encoding="utf-8")
    print("wrote", OUT, "rows", len(rows))


if __name__ == "__main__":
    main()
