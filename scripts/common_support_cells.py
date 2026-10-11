#!/usr/bin/env python3
"""Common-window analysis of the three cells of the past-only control (design.md 18.9, third pre-submission review).

The two centred cells (e1m_mean1s_abv: target from the processed speed; e1m_rt: target from the recorded speed)
share their evaluated windows exactly; the causal cell (e1m_causal) applies the same fold-exclusion rule but keeps
1-7 % more events, because the causal processing does not apply the gap-reconstruction constraints of the centred
one. This script aligns the three roots by the stable key (event_id, t_index), restricts every cell to the windows
the three have in common, and recomputes NLL, CRPS, coverage and width of the 90 % interval per model (mean over the
windows of each seed, then over the seeds, as scripts/analyze_e1.py does), together with the paired difference
causal - centred (recorded target) per window with a 95 % bootstrap interval over events and over followers.

    PYTHONUTF8=1 ./.venv/Scripts/python.exe scripts/common_support_cells.py            # artifacts/e1m_causal/tables/common_support.csv
    PYTHONUTF8=1 ./.venv/Scripts/python.exe scripts/common_support_cells.py --n-boot 200 --datasets ngsim openacc

Nothing is trained or predicted: only the stored per-window predictions (artifacts/<root>/preds/<model>/seed*/<set>.parquet)
and window tables (artifacts/<root>/windows/<set>.parquet, <set>_meta.json) are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOTS = {"e1m_mean1s_abv": "centred, processed target", "e1m_rt": "centred, recorded target", "e1m_causal": "causal, recorded target"}
MODELS = ["lstm_const_sigma", "hetero_mdn", "deep_ensemble", "mc_dropout", "gp_residual_idm"]
DATASETS = ["highd", "highd_val", "ngsim", "ngsim_us101", "ngsim_fn", "waymo", "openacc"]
METRICS = {"nll": "nll", "crps": "crps", "picp90": "covered", "width90": "width"}


def keyed_windows(repo: Path, root: str, dataset: str) -> pd.DataFrame:
    """The window table of a root with the stable key (event_id, t_index) and the follower name."""
    directory = repo / "artifacts" / root / "windows"
    meta = json.loads((directory / f"{dataset}_meta.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(directory / f"{dataset}.parquet", columns=["event", "t_index", "driver", "y"])
    frame["event_id"] = frame["event"].map(dict(enumerate(meta["event_ids"])))
    names = meta.get("driver_names")
    frame["follower"] = frame["driver"].map(dict(enumerate(names))) if names else frame["driver"]
    if frame["event_id"].isna().any():
        raise SystemExit(f"{root}/{dataset}: event ids not resolved")
    frame["row"] = np.arange(len(frame))
    return frame.set_index(["event_id", "t_index"], verify_integrity=True)


def seeds_of(repo: Path, root: str, model: str) -> list[Path]:
    return sorted((repo / "artifacts" / root / "preds" / model).glob("seed*"))


def bootstrap_mean(values: np.ndarray, groups: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    """95 % percentile interval of the mean of `values` under a cluster bootstrap over `groups`."""
    codes, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(codes))
    counts = np.bincount(inverse, minlength=len(codes)).astype(np.float64)
    draws = rng.integers(0, len(codes), size=(n_boot, len(codes)))
    means = (sums[draws].sum(axis=1)) / (counts[draws].sum(axis=1))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="default: artifacts/e1m_causal/tables/common_support.csv")
    args = ap.parse_args()
    repo = args.repo
    out = args.out or repo / "artifacts/e1m_causal/tables/common_support.csv"
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    support: dict[str, dict] = {}
    for dataset in args.datasets:
        frames = {root: keyed_windows(repo, root, dataset) for root in ROOTS}
        common = frames["e1m_mean1s_abv"].index
        for root in ("e1m_rt", "e1m_causal"):
            common = common.intersection(frames[root].index)
        counts = {root: {"events": int(f.index.get_level_values(0).nunique()), "windows": int(len(f))} for root, f in frames.items()}
        support[dataset] = {"counts": counts, "windows_common": int(len(common)),
                            "events_common": int(len(set(common.get_level_values(0)))),
                            "centred_pair_equal": bool(frames["e1m_mean1s_abv"].index.equals(frames["e1m_rt"].index)),
                            "recorded_target_max_difference_on_common": float(
                                (frames["e1m_rt"].loc[common, "y"] - frames["e1m_causal"].loc[common, "y"]).abs().max())}
        pos = {root: f.loc[common, "row"].to_numpy() for root, f in frames.items()}  # row positions of the common windows
        events = frames["e1m_rt"].loc[common].index.get_level_values(0).to_numpy()
        followers = frames["e1m_rt"].loc[common, "follower"].to_numpy()
        per_window: dict[tuple[str, str], np.ndarray] = {}
        for model in MODELS:
            for root in ROOTS:
                seeds = seeds_of(repo, root, model)
                if not seeds:
                    continue
                acc = {m: [] for m in METRICS}
                nll_windows = np.zeros(len(common))
                for s in seeds:
                    pred = pd.read_parquet(s / f"{dataset}.parquet", columns=list(METRICS.values()))
                    if len(pred) != counts[root]["windows"]:
                        raise SystemExit(f"{root}/{model}/{s.name}/{dataset}: {len(pred)} predictions for {counts[root]['windows']} windows")
                    sub = pred.iloc[pos[root]]
                    for m, col in METRICS.items():
                        acc[m].append(float(sub[col].astype(np.float64).mean()))
                    nll_windows += sub["nll"].to_numpy(np.float64) / len(seeds)
                per_window[(model, root)] = nll_windows
                row = {"dataset": dataset, "model": model, "root": root, "cell": ROOTS[root], "n_seeds": len(seeds),
                       "n_windows_common": int(len(common)), "n_events_common": support[dataset]["events_common"],
                       "n_windows_root": counts[root]["windows"], "n_events_root": counts[root]["events"]}
                row.update({m: float(np.mean(v)) for m, v in acc.items()})
                rows.append(row)
            if (model, "e1m_rt") in per_window and (model, "e1m_causal") in per_window:
                diff = per_window[(model, "e1m_causal")] - per_window[(model, "e1m_rt")]
                lo_e, hi_e = bootstrap_mean(diff, events, args.n_boot, rng)
                lo_f, hi_f = bootstrap_mean(diff, followers, args.n_boot, rng)
                for r in rows:
                    if r["dataset"] == dataset and r["model"] == model and r["root"] == "e1m_causal":
                        r.update({"nll_diff_vs_rt": float(diff.mean()), "nll_diff_lo_event": lo_e, "nll_diff_hi_event": hi_e,
                                  "nll_diff_lo_follower": lo_f, "nll_diff_hi_follower": hi_f})
        print(f"{dataset}: common windows {len(common):,} of {counts['e1m_causal']['windows']:,} causal "
              f"({counts['e1m_causal']['events'] - support[dataset]['events_common']} causal-only events)")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    (out.with_name("common_support_windows.json")).write_text(json.dumps(support, indent=2), encoding="utf-8")
    print("wrote", out, len(rows), "rows")


if __name__ == "__main__":
    main()
