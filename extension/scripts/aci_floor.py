#!/usr/bin/env python3
"""Starting level and guaranteed coverage floor of every online-ACI test stream of an E2 root.

The test stream of the variant ``aci_online`` continues the state reached on the calibration events
(``coverage_table`` in cf_unc/rollout/calibration.py: ``aci_run(score[perm], ..., state=run["state"])``), so its
starting level alpha_1 is the final level of the calibration run, not the target alpha. For the unclipped recursion
an interval issued at alpha_t <= 0 is the whole line and covers, so alpha_t > -gamma (1 - alpha) throughout, and
summing alpha_{t+1} = alpha_t + gamma (alpha - err_t) over the T events of the test stream bounds its error rate:

    coverage >= 1 - alpha - (alpha_1 + gamma (1 - alpha)) / (gamma T)      for any sequence of scores.

This script recomputes alpha_1, T and the floor of every (target, model, seed, horizon, data set) stream of a root
from the stored rollout tables, together with the achieved coverage, and writes <out>/tables/aci_floor.csv and
<out>/tables/aci_floor_summary.md. Nothing is re-simulated.

    PYTHONUTF8=1 python extension/scripts/aci_floor.py --root artifacts/e2 --out extension/artifacts/ext_aci_floor
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cf_unc.rollout import tables  # noqa: E402
from cf_unc.rollout.calibration import HALF_WIDTH_FLOOR, _horizon_arrays, aci_run, interval_score  # noqa: E402

COLUMNS = ("root", "target", "model", "seed", "horizon_s", "dataset", "n_cal", "alpha_1", "T", "floor", "coverage",
           "alpha_min", "alpha_end", "share_infinite")


def stream_rows(root: Path, target: str, calibration: str, alpha: float, gamma: float, n_min: int,
                log=print) -> list[dict]:  # noqa: ANN001
    listing = tables.available(root, target)
    if listing.empty:
        return []
    events = {d: tables.load_events_table(root, target, d) for d in listing["dataset"].unique()}
    if calibration not in events:
        log(f"  [{target}] no calibration set {calibration!r}: skipped")
        return []
    cal_events = events[calibration]
    cal_codes = cal_events["event"].to_numpy(np.int64)
    cal_perm = np.argsort(cal_events["order"].to_numpy(np.int64), kind="stable")
    perms = {d: (ev["event"].to_numpy(np.int64), np.argsort(ev["order"].to_numpy(np.int64), kind="stable"))
             for d, ev in events.items()}
    rows: list[dict] = []
    for model in listing["model"].unique():
        sub = listing[listing["model"] == model]
        cal_rows = sub[sub["dataset"] == calibration]
        if cal_rows.empty:
            continue
        eval_sets = [d for d in sub["dataset"].unique() if d != calibration]
        for r in cal_rows.itertuples():
            seed = int(r.seed)
            ctab = pd.read_parquet(r.path)
            etabs = {}
            for d in eval_sets:
                erow = sub[(sub["dataset"] == d) & (sub["seed"].astype(int) == seed)]
                if not erow.empty:
                    etabs[d] = pd.read_parquet(erow.iloc[0].path)
            for h in np.unique(ctab["horizon_s"].to_numpy(np.float64)):
                c = _horizon_arrays(ctab, h, cal_codes)
                cal_score = interval_score(c["truth_s"], c["s_q05"], c["s_q50"], c["s_q95"], HALF_WIDTH_FLOOR)
                run = aci_run(cal_score[cal_perm], alpha, gamma, n_min)
                a1 = float(run["state"].alpha_t)
                n_cal = int(np.sum(~np.isnan(cal_score)))
                for d, etab in etabs.items():
                    codes, perm = perms[d]
                    e = _horizon_arrays(etab, h, codes)
                    score = interval_score(e["truth_s"], e["s_q05"], e["s_q50"], e["s_q95"], HALF_WIDTH_FLOOR)
                    res = aci_run(score[perm], alpha, gamma, n_min, state=run["state"])
                    cov = res["covered"]
                    t = int(np.sum(~np.isnan(cov)))
                    floor = 1.0 - alpha - (a1 + gamma * (1.0 - alpha)) / (gamma * t) if t else float("nan")
                    rows.append({"root": root.name, "target": target, "model": model, "seed": seed,
                                 "horizon_s": float(h), "dataset": d, "n_cal": n_cal, "alpha_1": a1, "T": t,
                                 "floor": floor, "coverage": float(np.nanmean(cov)),
                                 "alpha_min": float(np.nanmin(res["alpha_t"])),
                                 "alpha_end": float(res["state"].alpha_t),
                                 "share_infinite": float(np.mean(res["infinite"]))})
        log(f"  [{target}] {model}: seeds {sorted(int(s) for s in cal_rows['seed'])}, sets {eval_sets}")
    return rows


def summary(frame: pd.DataFrame, alpha: float, gamma: float) -> str:
    lines = [f"# Online ACI: starting level and guaranteed floor of the test streams (alpha {alpha}, gamma {gamma})", "",
             "Floor = 1 - alpha - (alpha_1 + gamma (1 - alpha)) / (gamma T) per stream; alpha_1 = level reached on the",
             "calibration events; T = events of the test stream with a score. The achieved coverage is the stream's own.", "",
             "| root | target | streams | alpha_1 min | alpha_1 max | T | floor min | floor max | coverage min | coverage max | streams with floor < 0.87 | alpha_t min |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (root, target), g in frame.groupby(["root", "target"], sort=True):
        t_vals = sorted(g["T"].unique())
        t_txt = str(t_vals[0]) if len(t_vals) == 1 else f"{t_vals[0]}-{t_vals[-1]}"
        lines.append(f"| {root} | {target} | {len(g)} | {g.alpha_1.min():.4f} | {g.alpha_1.max():.4f} | {t_txt} | "
                     f"{g.floor.min():.4f} | {g.floor.max():.4f} | {g.coverage.min():.4f} | {g.coverage.max():.4f} | "
                     f"{int((g.floor < 0.87).sum())} | {g.alpha_min.min():.4f} |")
    worst = frame.sort_values("floor").head(5)
    lines += ["", "Five lowest floors:", "", "| root | target | model | seed | horizon | set | alpha_1 | T | floor | coverage |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in worst.itertuples():
        lines.append(f"| {r.root} | {r.target} | {r.model} | {r.seed} | {r.horizon_s:g} | {r.dataset} | {r.alpha_1:.4f} | {r.T} | {r.floor:.4f} | {r.coverage:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", nargs="+", required=True, help="E2 roots (artifacts/e2, artifacts/e2m, ...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibration", default="highd_val")
    a = ap.parse_args()
    rows: list[dict] = []
    alpha = gamma = None
    for root_s in a.root:
        root = Path(root_s)
        cfg = yaml.safe_load(io.open(root / "config.yaml", encoding="utf-8")) or {}
        cal = cfg.get("calibration", {}) or {}
        alpha, gamma, n_min = float(cal.get("alpha", 0.1)), float(cal.get("gamma", 0.01)), int(cal.get("n_min", 20))
        targets = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "rollouts").is_dir())
        print(f"{root}: targets {targets}, alpha {alpha}, gamma {gamma}, n_min {n_min}")
        for target in targets:
            rows += stream_rows(root, target, a.calibration, alpha, gamma, n_min)
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    out = Path(a.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "aci_floor.csv", index=False, float_format="%.6g")
    text = summary(frame, alpha, gamma)
    (out / "aci_floor_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
