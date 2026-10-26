#!/usr/bin/env python3
"""Figure of the platoon amplitude CRPS per model and driver mode (recorded signals, highD training), drawn two
columns wide for the journal layout: 8 pt fonts, no title inside the picture, vector PDF plus a 600 dpi PNG.
The values are those of ``<root>/tables/amplitude_crps.csv`` (variant ``raw``, extended set, response window, index
``all``: the mean over the measured vehicles of every profile); nothing is recomputed.

    PYTHONUTF8=1 python extension/scripts/platoon_figure.py --out extension/paper/v19/figures/f5_crps_bars__raw__e3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {  # the full names are in the caption of the figure
    "lstm_const_sigma": "LSTM (const. $\\sigma$)",
    "deep_ensemble": "Deep ensemble",
    "gp_residual_idm": "GP residual",
    "quantile_pidl": "Quantile PIDL",
    "diffusion_cf": "Diffusion",
    "diffusion_cf_hist": "Diffusion, hist.",
    "idm_ar_noise": "IDM + AR(1)",
}
REFERENCE = "idm_white_noise"
MODES = (("ACC", "ACC platoons"), ("Human", "Human-driven platoons"), ("all", "All profiles"))
BLUE = "#3b6fb6"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", default="artifacts/e3/tables/amplitude_crps.csv")
    ap.add_argument("--variant", default="raw")
    ap.add_argument("--out", required=True, help="output path without extension")
    a = ap.parse_args()
    t = pd.read_csv(a.table, dtype={"index": str})
    t = t[(t["variant"] == a.variant) & (t["set"] == "extended") & (t["window"] == "response") & (t["index"] == "all")]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 7.5, "pdf.fonttype": 42, "mathtext.fontset": "stix"})
    fig, axes = plt.subplots(1, len(MODES), figsize=(7.16, 2.9), sharey=True)
    for ax, (mode, title) in zip(axes, MODES):
        rows = t[t["mode"] == mode].drop_duplicates("model").set_index("model")
        ref = rows.loc[REFERENCE]
        models = [m for m in LABELS if m in rows.index]
        y = rows.loc[models, "crps"].to_numpy(float)
        lo = rows.loc[models, "crps_lo"].to_numpy(float)
        hi = rows.loc[models, "crps_hi"].to_numpy(float)
        x = np.arange(len(models), dtype=float)
        ax.axhspan(float(ref["crps_lo"]), float(ref["crps_hi"]), color=BLUE, alpha=0.15, lw=0, zorder=1)
        ax.axhline(float(ref["crps"]), color=BLUE, lw=1.4, zorder=5)
        ax.bar(x, y, width=0.7, color="#dcdcdc", edgecolor="#444444", lw=0.6, zorder=3)
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor="#222222", elinewidth=0.8, capsize=2, zorder=4)
        for xi, yi, hi_i in zip(x, y, hi):
            ax.annotate(f"{yi:.3g}", (xi, hi_i), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.5, zorder=6)
        ax.annotate(f"reference {float(ref['crps']):.2f} m/s", (x[-1] + 0.45, float(ref["crps"])), xytext=(0, 2),
                    textcoords="offset points", ha="right", va="bottom", fontsize=6.5, color=BLUE, zorder=6)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in models], rotation=35, ha="right", rotation_mode="anchor")
        ax.set_yscale("log")
        ax.set_ylim(0.8, 200)
        ax.set_title(f"{title} ({int(ref['n_profiles'])} profiles)")
        ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Amplitude CRPS, m/s")
    fig.tight_layout(w_pad=0.8)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=600)
    print("wrote", out.with_suffix(".pdf"), "and .png")


if __name__ == "__main__":
    main()
