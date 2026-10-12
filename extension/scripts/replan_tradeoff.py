#!/usr/bin/env python3
"""Cost and quality of the diffusion sampler settings and of plan sharing (extension S3 and S8).

From the root ext_e3_replan (tables/h32.csv, tables/collisions.csv, logs/timings.jsonl): per cell the H3.2 reduction
of the diffusion law against the IDM + white noise reference (extended set, ACC and human-driven profiles, response
and profile windows, file-cluster interval and verdict), the mean share of collided rollouts over the 20 followers
and all 60 profiles, and the simulation time per unit (one profile, 20 followers, 32 rollouts; median over the
batches of the timing log) with its ratio to the stored setting of the same base (re-planning every 5 s, 50 DDIM
steps). Cells: <base>_r<steps> (re-planning interval), <base>_s<steps> (DDIM sampling steps), <base>_r<steps>_share
(plan sharing, S8); raw = the highD training at the stored setting. Reading rule (PLAN.md S3): a setting is
admissible when its verdict is the stored one; the cheapest admissible setting per base is reported.

    extension/artifacts/ext_e3_replan/tables/replan_tradeoff.csv
    extension/artifacts/ext_e3_replan/tables/replan_tradeoff.md
    extension/artifacts/ext_e3_replan/figures/replan_tradeoff.png

    PYTHONUTF8=1 python extension/scripts/replan_tradeoff.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "extension/artifacts/ext_e3_replan"
MODEL, REFERENCE = "diffusion_cf", "idm_white_noise"
BASES = ("lodo_openacc", "pool", "raw")


def parse_variant(name: str) -> dict | None:
    if name == "raw":
        return {"base": "raw", "replan_steps": 50, "sampling_steps": 50, "share": False}
    for base in BASES:
        m = re.fullmatch(re.escape(base) + r"_(r|s)(\d+)(_share)?", name)
        if m:
            kind, value, share = m.group(1), int(m.group(2)), m.group(3) is not None
            return {"base": base, "replan_steps": value if kind == "r" else 50, "sampling_steps": value if kind == "s" else 50,
                    "share": share}  # fmt: skip
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(RUN))
    a = ap.parse_args()
    root = Path(a.root)
    h32 = pd.read_csv(root / "tables/h32.csv")
    col = pd.read_csv(root / "tables/collisions.csv")
    tim = pd.DataFrame([json.loads(line) for line in (root / "logs/timings.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()])
    h32 = h32[(h32.model == MODEL) & (h32.reference == REFERENCE) & (h32.quantity == "amplitude") & (h32["set"] == "extended")]
    col = col.copy()
    col["index"] = col["index"].astype(str)
    veh = col[(col["index"] != "any") & (col["set"] == "extended")].copy()
    collided = veh.groupby(["variant", "model"])["collided"].mean()
    seconds = tim[tim.model == MODEL].groupby("variant")["seconds_per_unit"].median()
    rows = []
    for variant in sorted(set(h32["variant"])):
        p = parse_variant(variant)
        if p is None:
            continue
        row = {"variant": variant, **p, "seconds_per_unit": float(seconds.get(variant, np.nan)),
               "collided_mean": float(collided.get((variant, MODEL), np.nan)),
               "collided_ref_mean": float(collided.get((variant, REFERENCE), np.nan))}  # fmt: skip
        for window in ("response", "profile"):
            for mode in ("ACC", "Human", "all"):
                r = h32[(h32.variant == variant) & (h32.window == window) & (h32["mode"] == mode)]
                if r.empty:
                    continue
                r = r.iloc[0]
                key = f"{window}_{mode}"
                row.update({f"reduction_{key}": r.reduction, f"lo_{key}": r.lo_cluster, f"hi_{key}": r.hi_cluster,
                            f"verdict_{key}": r.verdict_cluster, f"n_profiles_{key}": r.n_profiles})  # fmt: skip
        rows.append(row)
    table = pd.DataFrame(rows)
    stored = {b: table[(table.base == b) & (table.replan_steps == 50) & (table.sampling_steps == 50) & (~table.share)] for b in BASES}
    table["cost_ratio"] = [r.seconds_per_unit / float(stored[r.base].seconds_per_unit.iloc[0]) if not stored[r.base].empty else np.nan
                           for r in table.itertuples()]  # fmt: skip
    table["admissible_ACC"] = [bool(not stored[r.base].empty and r.verdict_response_ACC == stored[r.base].verdict_response_ACC.iloc[0])
                               for r in table.itertuples()]  # fmt: skip
    table = table.sort_values(["base", "share", "replan_steps", "sampling_steps"]).reset_index(drop=True)
    (root / "tables").mkdir(exist_ok=True)
    table.to_csv(root / "tables/replan_tradeoff.csv", index=False)

    lines = ["# Sampler settings and plan sharing: cost and quality (extension S3 and S8)", "",
             "Per cell: re-planning interval and DDIM sampling steps of the diffusion sampler, plan sharing, the H3.2 reduction "
             "against the IDM + white noise reference (extended set, response window, 95 % file-cluster interval, verdict) on "
             "the ACC and the human-driven profiles, the mean share of collided rollouts over the 20 followers and all 60 "
             "profiles, the simulation time per unit (s) and its ratio to the stored setting of the base.", "",
             "| base | re-plan [s] | steps | shared plans | ACC reduction [interval] verdict | human reduction [interval] verdict | coll. | s / unit | cost ratio | admissible (ACC) |",
             "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in table.itertuples():
        acc = f"{r.reduction_response_ACC:.2f} [{r.lo_response_ACC:.2f}, {r.hi_response_ACC:.2f}] {r.verdict_response_ACC}" if hasattr(r, "reduction_response_ACC") and np.isfinite(r.reduction_response_ACC) else "--"
        hum = f"{r.reduction_response_Human:.2f} [{r.lo_response_Human:.2f}, {r.hi_response_Human:.2f}] {r.verdict_response_Human}" if hasattr(r, "reduction_response_Human") and np.isfinite(r.reduction_response_Human) else "--"
        lines.append(f"| {r.base} | {r.replan_steps / 10:g} | {r.sampling_steps} | {'yes' if r.share else 'no'} | {acc} | {hum} | "
                     f"{r.collided_mean:.2f} | {r.seconds_per_unit:.1f} | {r.cost_ratio:.2f} | {'yes' if r.admissible_ACC else 'no'} |")  # fmt: skip
    lines.append("")
    for base in ("pool", "lodo_openacc"):
        ok = table[(table.base == base) & table.admissible_ACC & (~table.share)].sort_values("seconds_per_unit")
        if not ok.empty:
            r = ok.iloc[0]
            lines.append(f"Cheapest admissible setting of {base}: re-planning every {r.replan_steps / 10:g} s with {r.sampling_steps} "
                         f"steps ({r.seconds_per_unit:.1f} s per unit, cost ratio {r.cost_ratio:.2f}).")  # fmt: skip
    (root / "tables/replan_tradeoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for base, marker in (("pool", "o"), ("lodo_openacc", "s")):
            part = table[(table.base == base) & (~table.share)]
            ax.errorbar(part.cost_ratio, part.reduction_response_ACC,
                        yerr=[part.reduction_response_ACC - part.lo_response_ACC, part.hi_response_ACC - part.reduction_response_ACC],
                        fmt=marker, capsize=2, label=f"{base}")  # fmt: skip
            for r in part.itertuples():
                ax.annotate(f"r{r.replan_steps / 10:g}s/{r.sampling_steps}", (r.cost_ratio, r.reduction_response_ACC), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")  # fmt: skip
            shared = table[(table.base == base) & table.share]
            if not shared.empty:
                ax.plot(shared.cost_ratio, shared.reduction_response_ACC, marker, mfc="none", ms=9, label=f"{base}, shared plans")
        ax.axhline(0.2, color="grey", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("cost ratio to the stored setting (5 s re-planning, 50 steps)")
        ax.set_ylabel("H3.2 reduction, ACC profiles")
        ax.legend(fontsize=8)
        fig.tight_layout()
        (root / "figures").mkdir(exist_ok=True)
        fig.savefig(root / "figures/replan_tradeoff.png", dpi=150)
    except Exception as exc:  # noqa: BLE001
        print(f"figure skipped: {exc}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
