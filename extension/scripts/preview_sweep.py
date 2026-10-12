#!/usr/bin/env python3
"""Collect the preview sweep (extension X1) from the E3 analysis of its root into one table and one figure.

Reads <root>/tables/h32.csv, collisions.csv and growth.csv written by scripts/analyze_e3.py; the cells are the
variants <base>_<cell> with base in {raw, pool, lodo_openacc} (raw alone = the highD training at the full preview)
and cell h0 / h05 / h1 / h2 / h3 / h5 (preview horizon, s), n025 / n05 / n1 (noise, m/s, at the full horizon), d05 / d1
(delay, s, at the full horizon), t1 / t2 (stage B: the law trained at a budget of 1 / 2 s). Writes
<root>/tables/preview_sweep.csv (every mode, set and window), preview_sweep.md (the reading of PLAN.md X1: the
smallest horizon whose verdict is supported, per base and mode) and figures/preview_sweep.png.

    PYTHONUTF8=1 ./.venv/Scripts/python.exe extension/scripts/preview_sweep.py [--root extension/artifacts/ext_e3_preview]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASES = ("lodo_openacc", "pool", "raw")  # longest first: the variant name is matched by prefix
CELLS = {"h0": ("horizon", 0.0), "h05": ("horizon", 0.5), "h1": ("horizon", 1.0), "h2": ("horizon", 2.0),
         "h3": ("horizon", 3.0), "h5": ("horizon", 5.0), "n025": ("noise", 0.25), "n05": ("noise", 0.5),
         "n1": ("noise", 1.0), "d05": ("delay", 0.5), "d1": ("delay", 1.0), "t1": ("trained", 1.0), "t2": ("trained", 2.0)}
MODEL, REFERENCE = "diffusion_cf", "idm_white_noise"


def parse_variant(name: str) -> tuple[str, str, float] | None:
    if name == "raw":
        return "raw", "horizon", 5.0
    for base in BASES:
        m = re.fullmatch(re.escape(base) + r"_([a-z]+\d+)", name)
        if m and m.group(1) in CELLS:
            return (base, *CELLS[m.group(1)])
    return None


def collect(root: Path) -> pd.DataFrame:
    h32 = pd.read_csv(root / "tables/h32.csv")
    col = pd.read_csv(root / "tables/collisions.csv")
    h32 = h32[(h32.model == MODEL) & (h32.reference == REFERENCE) & (h32.quantity == "amplitude")].copy()
    parsed = h32["variant"].map(parse_variant)
    h32 = h32[parsed.notna()].copy()
    h32[["base", "kind", "value"]] = pd.DataFrame(parsed[parsed.notna()].tolist(), index=h32.index)
    # collisions.csv holds one row per follower index 1..20 and an "any" row (rollouts in which any follower collided):
    # collided_mean = mean over the 20 followers, collided_last = follower 20, collided_any = the "any" row
    col = col.copy()
    col["index"] = col["index"].astype(str)
    veh = col[col["index"] != "any"].copy()
    veh["index"] = veh["index"].astype(int)
    c = veh.groupby(["variant", "model", "set"], as_index=False).agg(collided_mean=("collided", "mean"))
    last = veh[veh["index"] == veh["index"].max()][["variant", "model", "set", "collided"]].rename(columns={"collided": "collided_last"})
    any_row = col[col["index"] == "any"][["variant", "model", "set", "collided"]].rename(columns={"collided": "collided_any"})
    c = c.merge(last, on=["variant", "model", "set"], how="left").merge(any_row, on=["variant", "model", "set"], how="left")
    cm = c[c.model == MODEL].drop(columns="model")
    cr = c[c.model == REFERENCE].drop(columns="model").rename(columns={"collided_mean": "collided_ref_mean",
                                                                      "collided_last": "collided_ref_last",
                                                                      "collided_any": "collided_ref_any"})  # fmt: skip
    out = h32.merge(cm, on=["variant", "set"], how="left").merge(cr, on=["variant", "set"], how="left")
    keep = ["base", "kind", "value", "variant", "set", "mode", "window", "reduction", "lo", "hi", "verdict", "lo_cluster",
            "hi_cluster", "verdict_cluster", "n_profiles", "n_files", "crps_model", "crps_reference", "collided_mean",
            "collided_last", "collided_any", "collided_ref_mean", "collided_ref_last", "collided_ref_any"]  # fmt: skip
    out = out[[k for k in keep if k in out.columns]].sort_values(["base", "kind", "value", "set", "mode", "window"])
    return out.reset_index(drop=True)


def reading(df: pd.DataFrame) -> str:
    lines = ["# Preview sweep: reading (PLAN.md X1)", "",
             "Set `extended`, window `response`, amplitude score of diffusion_cf against idm_white_noise, verdict of the "
             "file-cluster interval (support >= 0.20, refute <= 0.05).", "",
             "| Base | Mode | Smallest supported horizon, stage A [s] | Stage B supported budgets [s] | Noise cells supported [m/s] | Delay cells supported [s] |",
             "|---|---|---|---|---|---|"]
    sub = df[(df.set == "extended") & (df.window == "response")]
    for base in ("raw", "pool", "lodo_openacc"):
        for mode in ("all", "ACC", "Human"):
            s = sub[(sub.base == base) & (sub["mode"] == mode)]
            if s.empty:
                continue
            hz = s[(s.kind == "horizon") & (s.verdict_cluster == "supported")]["value"]
            tr = s[(s.kind == "trained") & (s.verdict_cluster == "supported")]["value"]
            nz = s[(s.kind == "noise") & (s.verdict_cluster == "supported")]["value"]
            dl = s[(s.kind == "delay") & (s.verdict_cluster == "supported")]["value"]
            fmt = lambda v: ", ".join(f"{x:g}" for x in sorted(v)) if len(v) else "none"  # noqa: E731
            lines.append(f"| {base} | {mode} | {hz.min():g} | {fmt(tr)} | {fmt(nz)} | {fmt(dl)} |" if len(hz)
                         else f"| {base} | {mode} | none | {fmt(tr)} | {fmt(nz)} | {fmt(dl)} |")
    lines += ["", "Every cell (reduction, cluster interval, verdict, collisions): `preview_sweep.csv`."]
    return "\n".join(lines) + "\n"


def figure(df: pd.DataFrame, path: Path) -> None:
    sub = df[(df.set == "extended") & (df.window == "response")]
    bases = [b for b in ("raw", "pool", "lodo_openacc") if (sub.base == b).any()]
    modes = [m for m in ("all", "ACC", "Human") if (sub["mode"] == m).any()]
    fig, axes = plt.subplots(len(modes), len(bases), figsize=(4.2 * len(bases), 3.0 * len(modes)), squeeze=False, sharey=True)
    for i, mode in enumerate(modes):
        for j, base in enumerate(bases):
            ax = axes[i, j]
            s = sub[(sub.base == base) & (sub["mode"] == mode)]
            hz = s[s.kind == "horizon"].sort_values("value")
            if not hz.empty:
                ax.errorbar(hz["value"], hz["reduction"], yerr=[hz["reduction"] - hz["lo_cluster"], hz["hi_cluster"] - hz["reduction"]],
                            fmt="o-", color="C0", capsize=3, label="preview horizon (stage A)")  # fmt: skip
            tr = s[s.kind == "trained"].sort_values("value")
            if not tr.empty:
                ax.errorbar(tr["value"], tr["reduction"], yerr=[tr["reduction"] - tr["lo_cluster"], tr["hi_cluster"] - tr["reduction"]],
                            fmt="s", mfc="none", color="C3", capsize=3, label="trained at the budget (stage B)")  # fmt: skip
            for kind, marker, color, dx in (("noise", "^", "C2", 0.15), ("delay", "v", "C1", -0.15)):
                k = s[s.kind == kind].sort_values("value")
                for n, (_, r) in enumerate(k.iterrows()):
                    x = 5.0 + dx * (n + 1)
                    ax.errorbar([x], [r["reduction"]], yerr=[[r["reduction"] - r["lo_cluster"]], [r["hi_cluster"] - r["reduction"]]],
                                fmt=marker, color=color, capsize=2, label=f"{kind} at 5 s" if n == 0 else None)  # fmt: skip
                    ax.annotate(f"{r['value']:g}", (x, r["reduction"]), textcoords="offset points", xytext=(3, 3), fontsize=7)
            ax.axhline(0.20, color="grey", lw=0.8, ls="--")
            ax.axhline(0.05, color="grey", lw=0.8, ls=":")
            ax.axhline(0.0, color="black", lw=0.6)
            ax.set_title(f"{base}, {mode}", fontsize=10)
            ax.set_xlabel("preview horizon [s]")
            if j == 0:
                ax.set_ylabel("amplitude reduction vs IDM")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "extension/artifacts/ext_e3_preview"))
    args = ap.parse_args()
    root = Path(args.root)
    df = collect(root)
    df.to_csv(root / "tables/preview_sweep.csv", index=False)
    (root / "tables/preview_sweep.md").write_text(reading(df), encoding="utf-8")
    figure(df, root / "figures/preview_sweep.png")
    print("cells:", df.variant.nunique(), "rows:", len(df))
    print(reading(df))
    sub = df[(df.set == "extended") & (df.window == "response") & (df["mode"] == "ACC")]
    print(sub.sort_values(["base", "kind", "value"])[["base", "kind", "value", "reduction", "lo", "hi", "lo_cluster", "hi_cluster",
                                                      "verdict", "verdict_cluster", "collided_mean", "collided_last", "collided_any"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
