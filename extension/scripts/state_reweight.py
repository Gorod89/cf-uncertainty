#!/usr/bin/env python3
"""State-distribution part of the dataset shift (extension S2, extension/docs/PLAN.md, section S).

The one-step tables of E1 (artifacts/e1/preds/<model>/seed<k>/<set>.parquet, row-aligned with
artifacts/e1/windows/<set>.parquet) are re-read with importance weights that bring the occupancy of the state space
(s, dv, v) of every shifted set to that of the in-domain set: 6 x 6 x 6 quantile bins built on the in-domain highD
test fold (the training windows are not stored as a table; the test fold is driver-disjoint from the training folds and
has the same occupancy), weight p_in(b) / p_shift(b) for a window in bin b, zero (dropped, mass reported) for the
bins that the in-domain set never visits. For every model, seed and shifted set the NLL, CRPS, PICP90 and the mean
variances are computed raw and reweighted; the NLL gap to the in-domain set is split into the state-distribution
part (raw minus reweighted) and the conditional part (reweighted minus in-domain). Seeds are averaged after.

    extension/artifacts/ext_reweight/tables/state_reweight.csv        model x shifted set, seed means
    extension/artifacts/ext_reweight/tables/state_reweight_seed.csv   the per-seed rows
    extension/artifacts/ext_reweight/tables/occupancy.csv             mass of every bin per set
    extension/artifacts/ext_reweight/tables/bins.json                 the bin edges and the in-domain source
    extension/artifacts/ext_reweight/tables/README.md                 the reading (PLAN.md S2)

    PYTHONUTF8=1 python extension/scripts/state_reweight.py [--e1 artifacts/e1] [--bins 6]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "extension/artifacts/ext_reweight/tables"
IN_DOMAIN = "highd"
SETS = ("highd_val", "ngsim", "ngsim_us101", "ngsim_fn", "waymo", "openacc")
FEATURES = ("s", "dv", "v")
METRICS = ("nll", "crps", "covered", "var", "aleatoric_var", "epistemic_var")
SHARE_EXPLAINED = 0.75  # reading rule: conditional part below 25 % of the NLL gap


def quantile_edges(x: np.ndarray, n: int) -> np.ndarray:
    e = np.quantile(x, np.linspace(0.0, 1.0, n + 1))
    e[0], e[-1] = -np.inf, np.inf
    return np.unique(e)


def bin_index(frame: pd.DataFrame, edges: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    idx = np.zeros(len(frame), dtype=np.int64)
    total = 1
    for f in FEATURES:
        e = edges[f]
        k = len(e) - 1
        b = np.clip(np.searchsorted(e, frame[f].to_numpy(np.float64), side="right") - 1, 0, k - 1)
        idx = idx * k + b
        total *= k
    return idx, total


def ratio(a: float, b: float) -> float:
    """a / b, NaN when the reference is zero or not finite (laws without an epistemic part)."""
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0.0 else float("nan")


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(x) & (w > 0)
    return float(np.sum(w[ok] * x[ok]) / np.sum(w[ok])) if ok.any() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--e1", default="artifacts/e1")
    ap.add_argument("--bins", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    e1, out = ROOT / a.e1, Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["event_index", "driver", *FEATURES]
    win = {name: pd.read_parquet(e1 / "windows" / f"{name}.parquet", columns=cols) for name in (IN_DOMAIN, *SETS)
           if (e1 / "windows" / f"{name}.parquet").exists()}  # fmt: skip
    edges = {f: quantile_edges(win[IN_DOMAIN][f].to_numpy(np.float64), a.bins) for f in FEATURES}
    idx_in, n_bins = bin_index(win[IN_DOMAIN], edges)
    p_in = np.bincount(idx_in, minlength=n_bins) / len(idx_in)
    weights: dict[str, np.ndarray] = {}
    occupancy = [{"set": IN_DOMAIN, "bin": b, "mass": float(p_in[b])} for b in range(n_bins)]
    info: dict[str, dict[str, float]] = {}
    for name, frame in win.items():
        if name == IN_DOMAIN:
            continue
        idx, _ = bin_index(frame, edges)
        p_sh = np.bincount(idx, minlength=n_bins) / len(idx)
        occupancy += [{"set": name, "bin": b, "mass": float(p_sh[b])} for b in range(n_bins)]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(p_in[idx] > 0, p_in[idx] / p_sh[idx], 0.0)
        kept = w > 0
        ess = float(w.sum() ** 2 / np.sum(w**2)) if kept.any() else 0.0
        weights[name] = w
        info[name] = {
            "n_windows": int(len(frame)), "mass_dropped": float(1.0 - kept.mean()), "ess": ess,
            "ess_ratio": ess / max(int(kept.sum()), 1), "bins_occupied": int((p_sh > 0).sum()),
            "bins_common": int(((p_sh > 0) & (p_in > 0)).sum()), "overlap_mass": float(np.minimum(p_in, p_sh).sum()),
        }  # fmt: skip
    pd.DataFrame(occupancy).to_csv(out / "occupancy.csv", index=False)
    (out / "bins.json").write_text(json.dumps({
        "in_domain": IN_DOMAIN, "source": str(e1.relative_to(ROOT)), "bins_per_feature": a.bins, "n_bins": n_bins,
        "edges": {f: [None if not np.isfinite(v) else float(v) for v in edges[f]] for f in FEATURES}, "sets": info,
    }, indent=1), encoding="utf-8")  # fmt: skip

    rows = []
    for model_dir in sorted((e1 / "preds").iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for seed_dir in sorted(model_dir.glob("seed*")):
            seed = int(seed_dir.name[4:])
            base = pd.read_parquet(seed_dir / f"{IN_DOMAIN}.parquet", columns=list(METRICS))
            if len(base) != len(win[IN_DOMAIN]):
                raise ValueError(f"{seed_dir}/{IN_DOMAIN}: {len(base)} rows, windows {len(win[IN_DOMAIN])}")
            ones = np.ones(len(base))
            ref = {m: weighted_mean(base[m].to_numpy(np.float64), ones) for m in METRICS}
            for name, w in weights.items():
                path = seed_dir / f"{name}.parquet"
                if not path.exists():
                    continue
                pred = pd.read_parquet(path, columns=list(METRICS))
                if len(pred) != len(w):
                    raise ValueError(f"{path}: {len(pred)} rows, windows {len(w)}")
                raw = {m: weighted_mean(pred[m].to_numpy(np.float64), np.ones(len(pred))) for m in METRICS}
                rw = {m: weighted_mean(pred[m].to_numpy(np.float64), w) for m in METRICS}
                gap = raw["nll"] - ref["nll"]
                part_state = raw["nll"] - rw["nll"]
                row = {"model": model, "seed": seed, "dataset": name, **info[name]}
                for m in METRICS:
                    row[f"{m}_in"], row[f"{m}_raw"], row[f"{m}_rw"] = ref[m], raw[m], rw[m]
                row.update({
                    "nll_gap": gap, "part_state": part_state, "part_cond": rw["nll"] - ref["nll"],
                    "share_state": part_state / gap if gap >= 0.05 else float("nan"),  # rated only for a positive gap
                    "r_alea_raw": ratio(raw["aleatoric_var"], ref["aleatoric_var"]), "r_alea_rw": ratio(rw["aleatoric_var"], ref["aleatoric_var"]),
                    "r_epi_raw": ratio(raw["epistemic_var"], ref["epistemic_var"]), "r_epi_rw": ratio(rw["epistemic_var"], ref["epistemic_var"]),
                    "r_var_raw": ratio(raw["var"], ref["var"]), "r_var_rw": ratio(rw["var"], ref["var"]),
                })  # fmt: skip
                rows.append(row)
    seed_table = pd.DataFrame(rows)
    seed_table.to_csv(out / "state_reweight_seed.csv", index=False)
    value_cols = [c for c in seed_table.columns if c not in ("model", "seed", "dataset")]
    mean = seed_table.groupby(["model", "dataset"], as_index=False)[value_cols].mean()
    mean["n_seeds"] = seed_table.groupby(["model", "dataset"]).size().to_numpy()
    sd = seed_table.groupby(["model", "dataset"])[["nll_raw", "nll_rw", "share_state"]].std().reset_index()
    mean = mean.merge(sd.rename(columns={"nll_raw": "nll_raw_sd", "nll_rw": "nll_rw_sd", "share_state": "share_state_sd"}),
                      on=["model", "dataset"])  # fmt: skip
    mean["explained_by_state"] = mean["share_state"] >= SHARE_EXPLAINED
    order = [IN_DOMAIN, *SETS]
    mean["dataset"] = pd.Categorical(mean["dataset"], [s for s in order if s in set(mean["dataset"])])
    mean = mean.sort_values(["model", "dataset"]).reset_index(drop=True)
    mean.to_csv(out / "state_reweight.csv", index=False)

    lines = ["# State-distribution part of the shift (extension S2)", "",
             f"Importance weights over {a.bins} x {a.bins} x {a.bins} quantile bins of (s, dv, v) built on the in-domain "
             f"set `{IN_DOMAIN}` ({len(win[IN_DOMAIN]):,} windows). Per shifted set: windows, mass in bins the in-domain "
             "set never visits (dropped), effective sample size of the kept windows, overlap mass of the two occupancies.", "",
             "| set | windows | mass dropped | ESS | ESS / kept | overlap mass |", "|---|---|---|---|---|---|"]  # fmt: skip
    for name, i in info.items():
        lines.append(f"| {name} | {i['n_windows']:,} | {i['mass_dropped']:.3f} | {i['ess']:,.0f} | {i['ess_ratio']:.3f} | {i['overlap_mass']:.3f} |")
    lines += ["", "NLL per window (nats; seed means): in domain, raw on the shifted set, reweighted to the in-domain occupancy; "
              "the gap split into its state-distribution part (raw - reweighted) and its conditional part (reweighted - in-domain); "
              f"`share_state` = state part / gap. Reading rule (PLAN.md S2): a set is explained by the state distribution when "
              f"share_state >= {SHARE_EXPLAINED:.2f}.", "",
              "| model | set | NLL in | NLL raw | NLL rw | state part | conditional part | share state | r_alea raw -> rw | r_epi raw -> rw |",
              "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in mean.itertuples():
        epi = "--" if not np.isfinite(r.r_epi_raw) else f"{r.r_epi_raw:.2f} -> {r.r_epi_rw:.2f}"
        share = "--" if not np.isfinite(r.share_state) else f"{r.share_state:.2f}"
        lines.append(f"| {r.model} | {r.dataset} | {r.nll_in:.2f} | {r.nll_raw:.2f} | {r.nll_rw:.2f} | {r.part_state:+.2f} | "
                     f"{r.part_cond:+.2f} | {share} | {r.r_alea_raw:.2f} -> {r.r_alea_rw:.2f} | {epi} |")  # fmt: skip
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"... {len(mean)} model x set rows written to {out}")


if __name__ == "__main__":
    main()
