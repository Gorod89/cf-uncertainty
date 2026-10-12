#!/usr/bin/env python3
"""String stability of the simulated platoons (extension S9, extension/docs/PLAN.md, section S).

From the stored per-profile tables (profile_index.csv of artifacts/e3 and of the extension platoon roots): the
amplitude ratio A(i+1) / A(i) of the median sampled amplitude by vehicle index (the median over the rollouts, robust
to the runaway amplitudes of collided rollouts), the first-follower ratio A(1) / A(leader), the head-to-tail ratio
A(20) / A(leader), and for the recorded followers the same ratios of the recorded amplitudes on the recorded indices.
Per law, cell and driver mode: the geometric mean of the 19 consecutive ratios (which telescopes to (A(20) / A(1))
to the power 1/19), with a bootstrap interval over the recording files (the cluster of E3), the share of indices
whose mean log ratio is above 0, and the recorded and simulated geometric ratios on the recorded indices.

    extension/artifacts/ext_string/tables/string_summary.csv    one row per (root, variant, model, mode)
    extension/artifacts/ext_string/tables/string_by_index.csv   mean ratio by index with intervals
    extension/artifacts/ext_string/tables/string_recorded.csv   recorded versus simulated ratios on the recorded indices
    extension/artifacts/ext_string/figures/string_stability.png the ratio by index of the four cells of the article
    extension/artifacts/ext_string/tables/README.md             the reading (PLAN.md S9)

    PYTHONUTF8=1 python extension/scripts/string_stability.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cf_unc.platoon.evaluate import mean_with_ci
from cf_unc.platoon.profiles import load_profiles

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "extension/artifacts"
OUT = EXT / "ext_string"
ROOTS = {"e3": ROOT / "artifacts/e3", "preview": EXT / "ext_e3_preview", "ngsim": EXT / "ext_e3_ngsim",
         "replan": EXT / "ext_e3_replan"}  # fmt: skip
VARIANTS = {"preview": ("raw", "raw_h0", "raw_h2", "pool_h0", "pool_h1", "pool_h2", "pool_h5", "pool_t2",
                        "lodo_openacc_h0", "lodo_openacc_h1", "lodo_openacc_h2", "lodo_openacc_h5", "lodo_openacc_t2")}  # fmt: skip
N_BOOT, SEED, CI = 1000, 0, 0.95
FIGURE_CELLS = (("e3", "harmonised"), ("e3", "raw"), ("e3", "pool"), ("e3", "lodo_openacc"))


def geo(values: np.ndarray, groups: np.ndarray, seed: int) -> dict[str, float]:
    """exp of the cluster-bootstrap mean of log ratios (NaN rows are left out by mean_with_ci)."""
    r = mean_with_ci(values, groups, N_BOOT, seed, CI)
    return {"estimate": float(np.exp(r["estimate"])), "lo": float(np.exp(r["lo"])), "hi": float(np.exp(r["hi"])), "n": int(r["n"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    profiles, _ = load_profiles(ROOT / "configs/platoon/profiles.yaml")
    mode_of = {p.id: p.mode for p in profiles}
    cluster_of = {p.id: f"{p.campaign}/{p.file}" for p in profiles}
    summary, by_index, recorded = [], [], []
    curves: dict[tuple[str, str, str], pd.DataFrame] = {}
    for root_name, root in ROOTS.items():
        path = root / "tables" / "profile_index.csv"
        if not path.exists():
            print(f"{root_name}: no profile_index.csv yet, skipped")
            continue
        pi = pd.read_csv(path)
        wanted = VARIANTS.get(root_name)
        for (variant, model), pm in pi.groupby(["variant", "model"], sort=False):
            if wanted is not None and variant not in wanted:
                continue
            amp = pm.pivot(index="profile", columns="index", values="q50").sort_index()
            emp = pm.pivot(index="profile", columns="index", values="emp_amplitude").reindex(amp.index)
            lead = pm.groupby("profile")["leader_amplitude"].first().reindex(amp.index).to_numpy(np.float64)
            ids = amp.index.to_numpy()
            n_v = int(amp.columns.max())
            A = amp.to_numpy(np.float64)  # (P, V)
            E = emp.reindex(columns=range(1, n_v + 1)).to_numpy(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_ratio = np.log(A[:, 1:] / A[:, :-1])  # (P, V - 1): index i -> ratio A(i+1) / A(i)
                log_first = np.log(A[:, 0] / lead)
                log_tail = np.log(A[:, -1] / lead)
                log_emp = np.log(E[:, 1:] / E[:, :-1])
                log_emp_first = np.log(E[:, 0] / lead)
            groups = np.array([cluster_of.get(i, i) for i in ids])
            modes = np.array([mode_of.get(i, "unknown") for i in ids])
            for mode in ("all", "ACC", "Human"):
                sel = np.ones(len(ids), bool) if mode == "all" else modes == mode
                if sel.sum() < 2:
                    continue
                g = groups[sel]
                per_profile = np.nanmean(log_ratio[sel], axis=1)  # geometric mean of the consecutive ratios
                geo_all = geo(per_profile, g, SEED)
                first, tail = geo(log_first[sel], g, SEED + 1), geo(log_tail[sel], g, SEED + 2)
                index_means = np.nanmean(log_ratio[sel], axis=0)
                rec_mask = np.isfinite(log_emp[sel])
                rec_ratio = geo(np.where(rec_mask, log_emp[sel], np.nan).mean(axis=1, where=rec_mask) if rec_mask.any() else np.full(sel.sum(), np.nan), g, SEED + 3) \
                    if rec_mask.any() else {"estimate": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
                sim_on_rec = geo(np.where(rec_mask, log_ratio[sel], np.nan).mean(axis=1, where=rec_mask), g, SEED + 4) \
                    if rec_mask.any() else {"estimate": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
                summary.append({
                    "root": root_name, "variant": variant, "model": model, "mode": mode, "n_profiles": int(sel.sum()),
                    "n_files": int(np.unique(g).size), "geo_ratio": geo_all["estimate"], "lo": geo_all["lo"], "hi": geo_all["hi"],
                    "verdict": "string-stable" if geo_all["hi"] < 1.0 else ("amplifying" if geo_all["lo"] > 1.0 else "inconclusive"),
                    "share_indices_above_1": float(np.mean(index_means > 0.0)), "first_vs_leader": first["estimate"],
                    "first_lo": first["lo"], "first_hi": first["hi"], "head_to_tail": tail["estimate"], "tail_lo": tail["lo"],
                    "tail_hi": tail["hi"], "recorded_geo_ratio": rec_ratio["estimate"], "recorded_lo": rec_ratio["lo"],
                    "recorded_hi": rec_ratio["hi"], "simulated_geo_ratio_recorded_indices": sim_on_rec["estimate"],
                    "simulated_lo_recorded_indices": sim_on_rec["lo"], "simulated_hi_recorded_indices": sim_on_rec["hi"],
                    "n_profiles_recorded": int(rec_mask.any(axis=1).sum()),
                })  # fmt: skip
                for j in range(log_ratio.shape[1]):
                    r = geo(log_ratio[sel, j], g, SEED + 10 + j)
                    by_index.append({"root": root_name, "variant": variant, "model": model, "mode": mode, "index": j + 1,
                                     "ratio": r["estimate"], "lo": r["lo"], "hi": r["hi"], "n_profiles": r["n"]})  # fmt: skip
                    if rec_mask[:, j].any():
                        e = geo(log_emp[sel, j], g, SEED + 100 + j)
                        s = geo(np.where(rec_mask[:, j], log_ratio[sel, j], np.nan), g, SEED + 100 + j)
                        recorded.append({"root": root_name, "variant": variant, "model": model, "mode": mode, "index": j + 1,
                                         "recorded_ratio": e["estimate"], "recorded_lo": e["lo"], "recorded_hi": e["hi"],
                                         "simulated_ratio": s["estimate"], "simulated_lo": s["lo"], "simulated_hi": s["hi"],
                                         "n_profiles": e["n"]})  # fmt: skip
                if mode == "all" and (root_name, variant) in FIGURE_CELLS:
                    curves[(root_name, variant, model)] = pd.DataFrame({"index": np.arange(1, log_ratio.shape[1] + 1),
                                                                        "ratio": np.exp(index_means)})  # fmt: skip
            if (root_name, variant) in FIGURE_CELLS and "recorded" not in {k[2] for k in curves if k[:2] == (root_name, variant)}:
                rec = np.nanmean(log_emp, axis=0)
                curves[(root_name, variant, "recorded")] = pd.DataFrame({"index": np.arange(1, len(rec) + 1), "ratio": np.exp(rec)})
    tables = {"string_summary": pd.DataFrame(summary), "string_by_index": pd.DataFrame(by_index),
              "string_recorded": pd.DataFrame(recorded)}  # fmt: skip
    for name, frame in tables.items():
        frame.to_csv(out / "tables" / f"{name}.csv", index=False)

    lines = ["# String stability of the simulated platoons (extension S9)", "",
             "Geometric mean of the consecutive amplitude ratios A(i+1) / A(i) of the median sampled amplitude over the "
             "20 followers (cluster bootstrap over recording files, 95 %), the first-follower and head-to-tail ratios to the "
             "leader, and the recorded ratios on the recorded indices. Reading rule (PLAN.md S9): string-stable in the mean "
             "when the interval lies below 1, amplifying when it lies above 1.", ""]  # fmt: skip
    s = tables["string_summary"]
    for (root_name, variant), part in s[s["mode"] != "Human"].groupby(["root", "variant"], sort=False):
        lines += [f"## {root_name} / {variant}", "", "| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |",
                  "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for r in part.itertuples():
            rec = "--" if not np.isfinite(r.recorded_geo_ratio) else f"{r.recorded_geo_ratio:.2f} [{r.recorded_lo:.2f}, {r.recorded_hi:.2f}]"
            sim = "--" if not np.isfinite(r.simulated_geo_ratio_recorded_indices) else f"{r.simulated_geo_ratio_recorded_indices:.2f}"
            lines.append(f"| {r.model} | {r.mode} | {r.n_profiles} | {r.geo_ratio:.3f} [{r.lo:.3f}, {r.hi:.3f}] | {r.verdict} | "
                         f"{r.share_indices_above_1:.2f} | {r.first_vs_leader:.2f} | {r.head_to_tail:.2f} | {rec} | {sim} |")  # fmt: skip
        lines.append("")
    (out / "tables" / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, sharey=True)
        for ax, (root_name, variant) in zip(axes.ravel(), FIGURE_CELLS):
            for (rn, var, model), frame in curves.items():
                if (rn, var) != (root_name, variant):
                    continue
                if model == "recorded":
                    ax.plot(frame["index"], frame["ratio"], "ko", ms=5, label="recorded")
                else:
                    ax.plot(frame["index"], frame["ratio"], lw=1.3, label=model)
            ax.axhline(1.0, color="grey", lw=0.8, ls="--")
            ax.set_title(f"{variant}", fontsize=10)
            ax.set_yscale("log")
        axes[1, 0].set_xlabel("vehicle index i (ratio A(i+1) / A(i))")
        axes[1, 1].set_xlabel("vehicle index i (ratio A(i+1) / A(i))")
        axes[0, 0].set_ylabel("amplitude ratio")
        axes[1, 0].set_ylabel("amplitude ratio")
        axes[0, 1].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out / "figures" / "string_stability.png", dpi=150)
    except Exception as exc:  # noqa: BLE001
        print(f"figure skipped: {exc}")
    print("\n".join(lines[:40]))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
