"""Compact markdown tables of experiment E2 (from the CSV tables of analyze_e2.py).

    python scripts/summarize_e2.py --root artifacts/e2 > artifacts/e2/tables/summary.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("idm_white_noise", "idm_white_noise_mle", "lstm_const_sigma", "hetero_mdn", "hetero_mdn_mix3", "deep_ensemble",
          "deep_ensemble__traj", "mc_dropout", "mc_dropout__traj", "gp_residual_idm", "gp_residual_idm__step", "quantile_pidl",
          "diffusion_cf")  # fmt: skip
DATASETS = ("highd", "ngsim", "ngsim_us101", "waymo", "openacc")
VARIANTS = ("raw", "static", "aci_frozen", "aci_online")
TARGETS = ("a", "mean_1s")


def num(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.{digits}f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def ordered(values, order) -> list[str]:
    present = set(values)
    return [v for v in order if v in present] + sorted(present - set(order))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="artifacts/e2")
    args = parser.parse_args()
    root = Path(args.root) if Path(args.root).is_absolute() else ROOT / args.root
    t = root / "tables"
    out: list[str] = []

    cov = pd.read_csv(t / "coverage_by_horizon.csv")
    horizons = sorted(cov.horizon_s.unique())
    for target in ordered(cov.target, TARGETS):
        for d in ordered(cov[cov.target == target].dataset, DATASETS):
            sub = cov[(cov.target == target) & (cov.dataset == d)]
            if d == "highd_val":
                continue
            rows = []
            for m in ordered(sub.model, MODELS):
                block = sub[sub.model == m]
                cells = [m]
                for v in VARIANTS:
                    vb = block[block.variant == v].set_index("horizon_s")
                    cells.append(" ".join(num(float(vb.loc[h, "picp"])) if h in vb.index else "-" for h in horizons))
                for v in ("raw", "aci_online"):
                    vb = block[block.variant == v].set_index("horizon_s")
                    cells.append(" ".join(num(float(vb.loc[h, "width_median"]), 2) if h in vb.index else "-" for h in horizons))
                raw = block[block.variant == "raw"].set_index("horizon_s")
                cells.append(num(float(raw.loc[horizons[-1], "collided"]), 3) if horizons[-1] in raw.index else "-")
                cells.append(num(float(raw.loc[horizons[-1], "crps"]), 2) if horizons[-1] in raw.index else "-")
                rows.append(cells)
            hz = "/".join(f"{h:g}" for h in horizons)
            out += [f"### Target `{target}`, {d}: coverage of the 90 % gap interval at {hz} s", "",
                    table(["model", "PICP raw", "PICP static", "PICP ACI frozen", "PICP ACI online", "median width raw, m",
                           "median width ACI online, m", f"collided by {horizons[-1]:g} s", f"CRPS at {horizons[-1]:g} s, m"], rows), ""]  # fmt: skip

    one = pd.read_csv(t / "onestep_conformal.csv")
    for target in ordered(one.target, TARGETS):
        for score in ("absolute", "normalised"):
            sub = one[(one.target == target) & (one.score == score)]
            datasets = ordered(sub.dataset, DATASETS)
            rows = []
            for m in ordered(sub.model, MODELS):
                block = sub[sub.model == m].set_index("dataset")
                cells = [m]
                for d in datasets:
                    if d in block.index:
                        r = block.loc[d]
                        cells.append(f"{num(float(r.picp_raw), 3)} -> {num(float(r.picp_conformal), 3)}")
                    else:
                        cells.append("-")
                rows.append(cells)
            out += [f"### Target `{target}`: one-step coverage, raw -> split conformal ({score} score, calibrated on highD validation)", "",
                    table(["model", *datasets], rows), ""]  # fmt: skip

    ver = pd.read_csv(t / "h34_verdicts.csv")
    ver = ver[ver.corrected == "aci_online"]
    for target in ordered(ver.target, TARGETS):
        sub = ver[ver.target == target]
        datasets = ordered(sub.dataset, DATASETS)
        rows = []
        for m in ordered(sub.model, MODELS):
            block = sub[sub.model == m].set_index("dataset")
            cells = [m]
            for d in datasets:
                if d in block.index:
                    r = block.loc[d]
                    cells.append(f"{num(float(r.onestep_picp), 3)} / {num(float(r.raw_picp_min))} @ {float(r.raw_min_horizon):g} s / "
                                 f"{num(float(r.corrected_picp_min))}: {r.verdict} ({r.verdict_ci})")  # fmt: skip
                else:
                    cells.append("-")
            rows.append(cells)
        out += [f"### Target `{target}`: H3.4 (one-step conformal on highD / lowest raw rollout coverage / lowest coverage after ACI "
                "online: verdict by point estimates (by intervals))", "", table(["model", *datasets], rows), ""]  # fmt: skip

    col = pd.read_csv(t / "collisions.csv")
    for target in ordered(col.target, TARGETS):
        sub = col[col.target == target]
        datasets = ordered(sub.dataset, ("highd", "highd_val", *DATASETS[1:]))
        piv = sub.pivot_table(index="model", columns="dataset", values="collided_10s")
        rows = [[m, *[num(float(piv.loc[m, d]), 3) if d in piv.columns and m in piv.index else "-" for d in datasets]]
                for m in ordered(sub.model, MODELS)]  # fmt: skip
        out += [f"### Target `{target}`: share of rollouts with a collision within 10 s", "", table(["model", *datasets], rows), ""]

    gam = pd.read_csv(t / "aci_gamma_sensitivity.csv")
    gam = gam[(gam.variant == "aci_online") & gam.dataset.isin(["highd", "ngsim"])]
    for target in ordered(gam.target, TARGETS):
        sub = gam[gam.target == target]
        rows = []
        for d in ordered(sub.dataset, DATASETS):
            for g in sorted(sub.gamma.unique()):
                block = sub[(sub.dataset == d) & (sub.gamma == g)]
                by_h = block.groupby("horizon_s")
                lo, hi = by_h.picp.min(), by_h.picp.max()
                width = by_h.width_median.median()
                rows.append([d, f"{g:g}", *[f"{num(float(lo[h]), 3)}-{num(float(hi[h]), 3)}" for h in horizons],
                             *[num(float(width[h]), 2) for h in horizons]])  # fmt: skip
        out += [f"### Target `{target}`: ACI online, sensitivity to gamma (range of the coverage over the models; median over the "
                "models of the median width, m)", "",
                table(["data set", "gamma", *[f"PICP {h:g} s" for h in horizons], *[f"width {h:g} s" for h in horizons]], rows), ""]  # fmt: skip

    print("\n".join(out))


if __name__ == "__main__":
    main()
