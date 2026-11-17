#!/usr/bin/env python3
"""Do the inducing inputs stored in the GP-residual checkpoints coincide with training rows?

The sparse variational GP of ``gp_residual_idm`` keeps 300 inducing inputs in the standardised state space
(s, dv, v). They start at k-means++ centres refined by Lloyd iterations (a centre of a one-point cluster IS a
training row at that moment) and are then trained with the other parameters (``learn_inducing: true``). A released
checkpoint must not carry rows of a data set whose terms forbid redistribution, so this script rebuilds the training
windows of every root exactly as the training run did and measures, for every checkpoint, the distance from each
stored inducing input to its nearest training row (standardised units), next to the spacing of the training rows
themselves. Only aggregates are written: no state of the data leaves the machine.

    PYTHONUTF8=1 python extension/scripts/gp_inducing_check.py --out extension/artifacts/ext_gp_inducing \
        --configs configs/experiment/e1.yaml configs/experiment/e6_pool.yaml ...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cf_unc.experiments.e1 import Data, load_config  # noqa: E402

COLUMNS = ("root", "train_data", "seed", "n_train_rows", "n_distinct_rows", "n_inducing", "coincide_1e-9", "within_1e-6",
           "within_1e-3", "dist_min", "dist_median", "dist_max", "rows_nn_median", "rows_nn_p95", "inside_data_box")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows: list[dict] = []
    for c in a.configs:
        started = time.perf_counter()
        cfg = load_config(c)
        root = ROOT / str(cfg.root)
        cks = sorted(root.glob("checkpoints/gp_residual_idm/seed*/model.pt"))
        if not cks:
            print(f"{c}: no gp_residual_idm checkpoint under {root}")
            continue
        data = Data(cfg, log=lambda m: None)
        train = data.bundle(False).windows["train"]
        x = np.asarray(train.X[:, -1, :], dtype=np.float64)
        names = "+".join(data.train_names)
        for ck in cks:
            d = torch.load(ck, map_location="cpu", weights_only=False)
            post, sc = d["state"]["posterior"], d["state"]["scalers"]
            z = post["Z"].double().numpy()
            xm, xs = sc["x_mean"].double().numpy(), sc["x_std"].double().numpy()
            xn = np.unique((x - xm) / xs, axis=0)
            tree = cKDTree(xn)
            dist, _ = tree.query(z, k=1)
            rng = np.random.default_rng(0)
            sub = xn[rng.choice(len(xn), size=min(20_000, len(xn)), replace=False)]
            nn = tree.query(sub, k=2)[0][:, 1]
            inside = np.all((z >= xn.min(0)) & (z <= xn.max(0)), axis=1)
            rows.append({
                "root": root.name, "train_data": names, "seed": int(ck.parent.name.replace("seed", "")),
                "n_train_rows": int(len(x)), "n_distinct_rows": int(len(xn)), "n_inducing": int(len(z)),
                "coincide_1e-9": int((dist < 1e-9).sum()), "within_1e-6": int((dist < 1e-6).sum()),
                "within_1e-3": int((dist < 1e-3).sum()), "dist_min": float(dist.min()),
                "dist_median": float(np.median(dist)), "dist_max": float(dist.max()),
                "rows_nn_median": float(np.median(nn)), "rows_nn_p95": float(np.quantile(nn, 0.95)),
                "inside_data_box": int(inside.sum()),
            })
        print(f"{root.name}: {len(cks)} checkpoints, {len(x):,} training rows ({names}), {time.perf_counter() - started:.0f} s", flush=True)
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    out = Path(a.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "gp_inducing_check.csv", index=False, float_format="%.6g")
    lines = ["# Inducing inputs of the GP-residual checkpoints against the training rows", "",
             "Distance (standardised units of s, dv, v) from each of the stored inducing inputs to its nearest training row;",
             "`rows_nn_*`: the same distance between a training row and its nearest other training row (20 000 sampled rows).", "",
             "| root | training data | checkpoints | inducing inputs | coincide (< 1e-9) | within 1e-6 | within 1e-3 | smallest distance | median distance | rows: median nearest-neighbour distance |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for root, g in frame.groupby("root", sort=True):
        lines.append(f"| {root} | {g.train_data.iloc[0]} | {len(g)} | {int(g.n_inducing.sum())} | {int(g['coincide_1e-9'].sum())} | "
                     f"{int(g['within_1e-6'].sum())} | {int(g['within_1e-3'].sum())} | {g.dist_min.min():.3g} | {g.dist_median.median():.3g} | "
                     f"{g.rows_nn_median.median():.3g} |")
    lines += ["", f"Total: {len(frame)} checkpoints, {int(frame.n_inducing.sum())} inducing inputs, "
              f"{int(frame['coincide_1e-9'].sum())} coincide with a training row, {int(frame['within_1e-6'].sum())} within 1e-6, "
              f"{int(frame['within_1e-3'].sum())} within 1e-3; smallest distance {frame.dist_min.min():.3g}."]
    (out / "gp_inducing_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
