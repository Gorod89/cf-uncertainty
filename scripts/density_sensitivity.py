"""Density sensitivity of the DiffusionCF scores (review control R14, docs/design.md 18.6).

The NLL of a sampled law (``cf_unc.dist.SampleDist``) is a Gaussian KDE over the samples with Silverman's bandwidth
(floored at 1e-3); the Gaussian models have a closed-form NLL. This script re-evaluates stored DiffusionCF checkpoints of
an E1 root on the rows that E1 evaluated (``eval.max_rows_sampled`` sampled windows per data set, the rows of the stored
prediction tables) with ``S`` = 16 / 32 / 64 first-step samples and bandwidth factors 0.5 / 1 / 2 x Silverman, and
writes per cell the mean NLL, the sample CRPS (plain as in the E1 tables, and fair), the PICP and mean width of the
central 90 % interval, next to the values of the stored E1 table on the same rows.

    python scripts/density_sensitivity.py --root artifacts/e1 --models diffusion_cf --datasets highd ngsim --dry-run
    python scripts/density_sensitivity.py --root artifacts/e1 --models diffusion_cf --datasets highd ngsim

One draw of ``max(--samples)`` sequences per window (the seeded streams of ``predict_seed``, as ``predict_dist``) is cut
into nested subsets (the first 16, 32, 64 samples of every row; the samples of a row are exchangeable), so the cells
differ by the number of samples, not by another random draw; the stored 32-sample tables come from another draw of the
same law (``stored_*`` columns). CRPS and the interval do not depend on the bandwidth. Seed 0 by default. Output:
``<root>/tables/density_sensitivity.csv`` (``--out``). ``--dry-run``: checks the checkpoints, the windows and the stored
tables and states the rows and the sampling work, without sampling (no GPU) and without writing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from cf_unc.dist import SampleDist  # noqa: E402
from cf_unc.eval.metrics import crps_samples  # noqa: E402
from cf_unc.experiments.e1 import Data, _same_spec, checkpoint_dir, configure_torch, load_config, seed_everything  # noqa: E402
from cf_unc.maps import io as tables  # noqa: E402
from cf_unc.models import load_model, resolve_device  # noqa: E402
from cf_unc.train.run import ROOT, _path  # noqa: E402

COLUMNS = ["dataset", "model", "seed", "n_samples", "bandwidth_factor", "n_rows", "nll", "nll_median", "crps", "crps_fair",
           "picp90", "width90", "stored_nll", "stored_crps", "stored_picp90", "stored_width90", "sampling_seconds"]  # fmt: skip


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="artifacts/e1", help="E1 root with the checkpoints, windows and preds")
    parser.add_argument("--config", default="configs/experiment/e1.yaml", help="the experiment config of the root")
    parser.add_argument("--models", nargs="+", default=["diffusion_cf"], help="sampled models (labels of the root)")
    parser.add_argument("--datasets", nargs="+", default=["highd", "ngsim"], help="evaluated data sets of the root")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="model seeds (default: 0)")
    parser.add_argument("--samples", nargs="+", type=int, default=[16, 32, 64], help="numbers of samples S")
    parser.add_argument("--bandwidths", nargs="+", type=float, default=[0.5, 1.0, 2.0], help="factors of Silverman's bandwidth")
    parser.add_argument("--max-rows", type=int, default=None, help="rows per data set (default: eval.max_rows_sampled)")
    parser.add_argument("--coverage", type=float, default=0.9, help="central interval")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--out", default=None, help="output csv (default: <root>/tables/density_sensitivity.csv)")
    parser.add_argument("--dry-run", action="store_true", help="check inputs and state the work; no sampling, no output")
    return parser.parse_args(argv)


def evaluated_rows(root: Path, model: str, seed: int, dataset: str) -> np.ndarray:
    """Rows of the stored prediction table with a finite mean (the rows E1 evaluated, ``sampled_rows``)."""
    frame = pd.read_parquet(tables.preds_path(root, model, seed, dataset), columns=["mean"])
    return np.flatnonzero(np.isfinite(frame["mean"].to_numpy(np.float64)))


def scores(samples: np.ndarray, y: np.ndarray, factor: float, coverage: float) -> dict[str, float]:
    """Means over the rows of the KDE NLL (bandwidth ``factor`` x Silverman), sample CRPS (plain and fair), PICP and
    width of the central interval of a (N, S) sample law at the targets ``y`` (N,)."""
    dist = SampleDist(samples)
    nll = -dist.log_prob(y, bandwidth_factor=factor)
    lo, hi = dist.interval(coverage)
    return {
        "nll": float(np.mean(nll)), "nll_median": float(np.median(nll)), "crps": float(np.mean(dist.crps(y))),
        "crps_fair": float(np.mean(crps_samples(samples, y, fair=True))), "picp90": float(np.mean((y >= lo) & (y <= hi))),
        "width90": float(np.mean(hi - lo)),
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    root = _path(args.root)
    cfg.root = str(root)
    configure_torch(bool(cfg.get("tf32", False)))
    data = Data(cfg, print)
    target = str((cfg.get("train") or {}).get("target") or "a")
    s_max = max(int(s) for s in args.samples)
    if min(int(s) for s in args.samples) < 2:
        raise SystemExit("every number of samples must be >= 2")
    plan, problems = [], []
    for dataset in args.datasets:
        meta_path = tables.windows_path(root, dataset).with_name(f"{dataset}_meta.json")
        if not meta_path.exists():
            problems.append(f"{dataset}: no windows table in {root}")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not _same_spec(meta.get("spec"), data.eval_spec(dataset)):
            problems.append(f"{dataset}: the windows of {root} were written with {meta.get('spec')}, {args.config} "
                            f"gives {data.eval_spec(dataset)}")  # fmt: skip
            continue
        ev = data.eval_set(dataset)  # loads the events: the windows must be those of the stored table
        if meta["n_windows"] != len(ev.windows) or meta["event_ids"] != [e.event_id for e in ev.events]:
            problems.append(f"{dataset}: the windows cut from the events differ from the stored table of {root}")
            continue
        for model in args.models:
            for seed in args.seeds:
                ckpt = checkpoint_dir(root, model, seed) / "model.pt"
                preds = tables.preds_path(root, model, seed, dataset)
                if not ckpt.exists() or not preds.exists():
                    problems.append(f"{model}/seed{seed}/{dataset}: {'checkpoint' if not ckpt.exists() else 'table'} missing")
                    continue
                rows = evaluated_rows(root, model, seed, dataset)
                if args.max_rows is not None:
                    rows = rows[: int(args.max_rows)]
                plan.append({"dataset": dataset, "model": model, "seed": seed, "rows": rows})
    work = sum(p["rows"].size for p in plan) * s_max
    print(f"density sensitivity: root {root}, target {target}; {len(plan)} unit(s): "
          + ", ".join(f"{p['model']}/seed{p['seed']}/{p['dataset']} {p['rows'].size:,} rows" for p in plan)
          + f"; {work:,} sampled sequences (S = {s_max}); cells S {args.samples} x bandwidth {args.bandwidths}")  # fmt: skip
    for problem in problems:
        print(f"  problem: {problem}")
    if args.dry_run:
        return 1 if problems else 0
    if problems:
        raise SystemExit("inputs missing or inconsistent (see above); nothing sampled")
    device = resolve_device(args.device)
    out_rows = []
    for unit in plan:
        dataset, model_name, seed, rows = unit["dataset"], unit["model"], int(unit["seed"]), unit["rows"]
        ev = data.eval_set(dataset)
        model = load_model(checkpoint_dir(root, model_name, seed) / "model.pt", device=device)
        if not getattr(model, "plan_sampler", False):
            raise SystemExit(f"{model_name} is not a sampled model (plan sampler)")
        future = ev.leader_future(rows) if model.needs_leader_future else None
        seed_everything(seed)
        started = time.perf_counter()
        seq = model.sample_sequences(ev.windows.X[rows], future, n_samples=s_max, batch_size=8192)
        seconds = time.perf_counter() - started
        steps = 1 if target == "a" else 10
        draws = seq[:, :, 0].astype(np.float64) if steps == 1 else seq[:, :, :steps].astype(np.float64).mean(axis=2)
        y = ev.windows.y[rows].astype(np.float64)
        stored = pd.read_parquet(tables.preds_path(root, model_name, seed, dataset)).iloc[rows]
        ref = {"stored_nll": float(stored["nll"].mean()), "stored_crps": float(stored["crps"].mean()),
               "stored_picp90": float(stored["covered"].mean()), "stored_width90": float(stored["width"].mean())}  # fmt: skip
        print(f"[{model_name} seed{seed}] {dataset}: {rows.size:,} rows x {s_max} samples in {seconds:.1f} s "
              f"(stored S = 32: NLL {ref['stored_nll']:.4f}, CRPS {ref['stored_crps']:.4f}, PICP {ref['stored_picp90']:.3f})", flush=True)  # fmt: skip
        for s in sorted(int(x) for x in args.samples):
            for factor in args.bandwidths:
                cell = scores(draws[:, :s], y, float(factor), float(args.coverage))
                out_rows.append({"dataset": dataset, "model": model_name, "seed": seed, "n_samples": s,
                                 "bandwidth_factor": float(factor), "n_rows": int(rows.size), **cell, **ref,
                                 "sampling_seconds": seconds})  # fmt: skip
        del model, seq
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = pd.DataFrame(out_rows, columns=COLUMNS)
    out = _path(args.out) if args.out else root / "tables" / "density_sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(frame.drop(columns=["model", "seed", "sampling_seconds"]).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"written to {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
