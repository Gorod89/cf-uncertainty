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

    out += review_sections(cov, horizons, t)
    print("\n".join(out))


BUDGET = (("static", "static"), ("static_prequential", "static prequential"), ("aci_frozen", "ACI frozen"),
          ("aci_online", "ACI online"), ("aci_online_chrono", "ACI chronological"),
          ("aci_online_delayed", "ACI chronological, delayed"), ("static_prequential_delayed", "static prequential, delayed"))


def review_sections(cov: pd.DataFrame, horizons: list[float], t: Path) -> list[str]:
    """Tables of the review fixes (docs/review_fixes_v3.md): ACI at the ends of the level range (R1), information
    budget of the online corrections (R8), follower-cluster intervals (R4); empty for tables without their columns."""
    out: list[str] = []
    if "share_infinite" not in cov.columns:
        return out
    last = horizons[-1]
    for target in ordered(cov.target, TARGETS):
        for d in ordered(cov[cov.target == target].dataset, DATASETS):
            sub = cov[(cov.target == target) & (cov.dataset == d)]
            if d == "highd_val" or not set(sub.variant) & {"aci_online"}:
                continue
            rows = []
            for m in ordered(sub.model, MODELS):
                block = sub[sub.model == m]
                cells = [m]
                for v in ("aci_frozen", "aci_online"):
                    vb = block[block.variant == v].set_index("horizon_s")
                    cells.append(" ".join(num(100 * float(vb.loc[h, "share_infinite"]), 1) if h in vb.index else "-"
                                          for h in horizons))  # fmt: skip
                vb = block[block.variant == "aci_online"].set_index("horizon_s")
                cells.append(" ".join(num(float(vb.loc[h, "width_finite_median"]), 2) if h in vb.index else "-" for h in horizons))
                cells.append(" ".join(num(float(vb.loc[h, "width_median"]), 2) if h in vb.index else "-" for h in horizons))
                rows.append(cells)
            hz = "/".join(f"{h:g}" for h in horizons)
            out += [f"### Target `{target}`, {d}: ACI at the ends of the level range (R1) at {hz} s", "",
                    "Share of the intervals that are the whole line (level >= 1, Gibbs and Candes), per cent; median "
                    "width of the finite intervals and of all intervals (inf when more than half are infinite), m.", "",
                    table(["model", "infinite % ACI frozen", "infinite % ACI online", "median finite width ACI online, m",
                           "median width ACI online, m"], rows), ""]  # fmt: skip
            if not set(sub.variant) & {"aci_online_delayed"}:
                continue
            rows = []
            for m in ordered(sub.model, MODELS):
                block = sub[sub.model == m]
                cells = [m]
                for v, _ in BUDGET:
                    vb = block[block.variant == v].set_index("horizon_s")
                    cells.append(" ".join(num(float(vb.loc[h, "picp"])) if h in vb.index else "-" for h in horizons))
                vb = block[block.variant == "aci_online_delayed"].set_index("horizon_s")
                cells.append(f"{num(float(vb.loc[last, 'picp_first50']))} / {num(float(vb.loc[last, 'picp_first100']))}"
                             if last in vb.index else "-")  # fmt: skip
                cells.append(num(float(vb.loc[last, "delay_events_mean"]), 1) if last in vb.index else "-")
                rows.append(cells)
            out += [f"### Target `{target}`, {d}: information budget of the corrections (R8), PICP at {hz} s", "",
                    "static / ACI frozen: no label of the evaluated set; static prequential: the static correction "
                    "recalibrated on every label seen so far (the labels of ACI online, no learning rate); ACI online: "
                    "random order, label after each event; chronological: events in time order; delayed: a label used "
                    f"only a horizon after its interval was issued. First 50 / 100: coverage of delayed ACI over the first "
                    f"events of the stream at {last:g} s; delay: mean events issued before a label arrives at {last:g} s.", "",
                    table(["model", *[lab for _, lab in BUDGET], f"first 50 / 100, {last:g} s", f"delay, events"], rows), ""]  # fmt: skip
    streams = t / "stream_info.csv"
    if streams.exists():
        s = pd.read_csv(streams)
        rows = [[str(r.target), str(r.dataset), str(r.clock), str(int(r.n_events)), num(float(r.n_followers), 0),
                 num(float(r.span_s), 0), num(float(r.events_per_s), 3)] for r in s.itertuples()]  # fmt: skip
        out += ["### Clocks of the evaluated streams (R8)", "",
                table(["target", "data set", "clock", "events", "followers", "recorded span, s", "events per s"], rows), ""]
    if "picp_lo_follower" in cov.columns:
        rows = []
        for (target, d), grp in cov[cov.dataset != "highd_val"].groupby(["target", "dataset"], sort=False):
            for v in ("raw", "aci_online"):
                g = grp[grp.variant == v]
                ratio = (g.picp_hi_follower - g.picp_lo_follower) / (g.picp_hi - g.picp_lo)
                rows.append([str(target), str(d), v, num(float(g.n_followers.max()), 0), num(float(g.n_events.max()), 0),
                             num(float(ratio.median()), 2), num(float(ratio.max()), 2)])  # fmt: skip
        out += ["### Follower-cluster bootstrap of the coverage intervals (R4)", "",
                "Width of the 95 % interval when whole followers are resampled, relative to the event bootstrap "
                "(median and largest over models and horizons).", "",
                table(["target", "data set", "variant", "followers", "events", "width ratio, median", "width ratio, max"], rows), ""]
    return out


if __name__ == "__main__":
    main()
