#!/usr/bin/env python3
"""Cost table of the laws (extension X2, extension/docs/PLAN.md).

Three sources, written to extension/artifacts/ext_cost/tables/:
* cost_training.csv: training wall time per seed from the stored training logs of the roots (artifacts/e1 = recorded
  highD signals, artifacts/e1_ctrl = the review controls, artifacts/e6_pool and e6_lodo_openacc = the pooled domain)
  and the parameter count of the seed-0 checkpoint;
* cost_inference.csv: one-step inference time of ``predict_dist`` on highD test windows (the rows that E1 evaluates
  for the plan samplers: windows with a full 5 s future), median of repeated calls, per device, in ms per 1000
  windows; the plan samplers (DiffusionCF: 32 samples x 50 DDIM steps per window) are timed on fewer rows and scaled;
* cost_platoon.csv: the platoon simulation cost from artifacts/e3/logs/timings.jsonl (the cells of the current
  engine), seconds per unit (60-profile batches: one profile x 20 followers x R rollouts) and ms per simulated
  vehicle-rollout-second.
cost.csv merges them per model; README.md holds the table and the machine.

    PYTHONUTF8=1 ./.venv/Scripts/python.exe extension/scripts/cost_table.py [--no-benchmark] [--rows 8192] [--repeats 5]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cf_unc.constants import DT  # noqa: E402
from cf_unc.experiments.e1 import Data, load_config, sampled_rows  # noqa: E402
from cf_unc.models import load_model  # noqa: E402
from cf_unc.models.base import CFModel  # noqa: E402

OUT = ROOT / "extension/artifacts/ext_cost"
TRAIN_ROOTS = {"e1": "artifacts/e1", "e1_ctrl": "artifacts/e1_ctrl", "e6_pool": "artifacts/e6_pool",
               "e6_lodo_openacc": "artifacts/e6_lodo_openacc"}
BENCH_ROOT = "artifacts/e1"  # checkpoints timed (seed 0)
PLATOON_VARIANTS = ("pool", "lodo_openacc", "pool_w11", "lodo_openacc_w11", "raw_w61", "raw_a")  # current engine
PLAN_ROWS = {"cuda": 1024, "cpu": 128}  # rows timed for the plan samplers (scaled to 1000 windows)


def n_params(model: CFModel) -> int:
    """Parameters of every torch module reachable from the model (members of an ensemble included); a model without
    modules (the IDM baselines) counts the scalars of its state."""
    seen: set[int] = set()
    total = 0

    def visit(obj: Any) -> None:
        nonlocal total
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            total += sum(p.numel() for p in obj.parameters())
        elif isinstance(obj, CFModel):
            for value in vars(obj).values():
                visit(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                visit(value)
        elif isinstance(obj, dict):
            for value in obj.values():
                visit(value)

    visit(model)
    if total == 0:
        def scalars(obj: Any) -> int:
            if isinstance(obj, (bool, str)) or obj is None:
                return 0
            if isinstance(obj, (int, float, np.number)):
                return 1
            if isinstance(obj, (np.ndarray, torch.Tensor)):
                return int(np.prod(obj.shape)) if obj.shape else 1
            if isinstance(obj, dict):
                return sum(scalars(v) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                return sum(scalars(v) for v in obj)
            if dataclasses.is_dataclass(obj):  # IDMParams: the six parameters of the physical law
                return sum(scalars(getattr(obj, f.name)) for f in dataclasses.fields(obj))
            return 0
        total = scalars(model.get_state())
        idm = getattr(model, "idm", None)
        if idm is not None and dataclasses.is_dataclass(idm) and "idm" not in (model.get_state() or {}):
            total += scalars(idm)
    return int(total)


def training_table() -> pd.DataFrame:
    rows = []
    for key, root in TRAIN_ROOTS.items():
        for log_path in sorted((ROOT / root / "checkpoints").glob("*/seed*/train_log.json")):
            log = json.loads(log_path.read_text(encoding="utf-8"))
            rows.append({"root": key, "label": log_path.parents[1].name, "model": log.get("registry_name", log.get("model")),
                         "seed": int(log.get("seed", log_path.parent.name.replace("seed", ""))),
                         "seconds": float(log.get("seconds", float("nan"))), "n_train": log.get("n_train"),
                         "device": log.get("device")})  # fmt: skip
    df = pd.DataFrame(rows)
    agg = df.groupby(["root", "label", "model"], as_index=False).agg(
        seeds=("seed", "nunique"), seconds_mean=("seconds", "mean"), seconds_min=("seconds", "min"),
        seconds_max=("seconds", "max"), n_train=("n_train", "max"), device=("device", "first"))  # fmt: skip
    params = {}
    for (root, label), _ in agg.groupby(["root", "label"]):
        path = ROOT / TRAIN_ROOTS[root] / "checkpoints" / label / "seed0" / "model.pt"
        if path.exists():
            try:
                params[(root, label)] = n_params(load_model(path, device="cpu"))
            except Exception as exc:  # noqa: BLE001 - reported in the table
                params[(root, label)] = f"error: {type(exc).__name__}"
    agg["n_params"] = [params.get((r, l)) for r, l in zip(agg["root"], agg["label"])]
    return agg


def platoon_table() -> pd.DataFrame:
    rows = []
    for line in (ROOT / "artifacts/e3/logs/timings.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("variant") not in PLATOON_VARIANTS or r.get("seconds_per_unit") is None:
            continue
        sim_s = float(r["n_vehicles"]) * float(r["n_rollouts"]) * float(r["n_steps"]) * DT
        rows.append({"variant": r["variant"], "label": r["model"], "seed": r.get("seed"), "n_rollouts": r["n_rollouts"],
                     "n_steps": r["n_steps"], "n_vehicles": r["n_vehicles"], "seconds_per_unit": float(r["seconds_per_unit"]),
                     "us_per_vehicle_rollout_s": 1e6 * float(r["seconds_per_unit"]) / sim_s,
                     "peak_memory_mb": r.get("peak_memory_mb"), "device": r.get("device")})  # fmt: skip
    df = pd.DataFrame(rows)
    return df.groupby("label", as_index=False).agg(
        cells=("variant", "nunique"), units=("seed", "size"), n_rollouts=("n_rollouts", "first"),
        seconds_per_unit_median=("seconds_per_unit", "median"), seconds_per_unit_min=("seconds_per_unit", "min"),
        seconds_per_unit_max=("seconds_per_unit", "max"), us_per_vehicle_rollout_s=("us_per_vehicle_rollout_s", "median"),
        peak_memory_mb=("peak_memory_mb", "max"), device=("device", "first"))  # fmt: skip


def benchmark(rows: int, repeats: int, devices: list[str]) -> pd.DataFrame:
    cfg = load_config(ROOT / "configs/experiment/e1.yaml")
    data = Data(cfg)
    es = data.eval_set("highd")
    idx = sampled_rows(es.windows.event_index, es.windows.t_index, es.lengths, rows, seed=0)
    x_all = es.windows.X[idx]
    fut_all = es.leader_future(idx)
    out = []
    labels = sorted(p.name for p in (ROOT / BENCH_ROOT / "checkpoints").iterdir() if (p / "seed0" / "model.pt").exists())
    for label in labels:
        for device in devices:
            model = load_model(ROOT / BENCH_ROOT / "checkpoints" / label / "seed0" / "model.pt", device=device)
            plan = bool(getattr(model, "plan_sampler", False))
            n = min(len(idx), PLAN_ROWS[device]) if plan else len(idx)
            x, fut = x_all[:n], fut_all[:n]
            kwargs: dict[str, Any] = {"v_lead_future": fut} if getattr(model, "needs_leader_future", False) else {}
            times = []
            for r in range(repeats + 1):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model.predict_dist(x, batch_size=8192, **kwargs)
                if device == "cuda":
                    torch.cuda.synchronize()
                if r > 0:  # the first call warms up (cuDNN autotuning, allocator)
                    times.append(time.perf_counter() - t0)
            med = float(np.median(times))
            out.append({"label": label, "model": type(model).__name__, "device": device, "rows": n, "repeats": repeats,
                        "seconds_median": med, "ms_per_1000_windows": 1000.0 * med / n * 1000.0,
                        "plan_sampler": plan, "n_params": n_params(model)})  # fmt: skip
            print(f"{label:22s} {device:5s} rows {n:5d}  {1000.0 * med / n * 1000.0:10.2f} ms / 1000 windows", flush=True)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-benchmark", action="store_true")
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--cpu-only", action="store_true")
    args = ap.parse_args()
    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    train = training_table()
    train.to_csv(tables / "cost_training.csv", index=False)
    plat = platoon_table()
    plat.to_csv(tables / "cost_platoon.csv", index=False)
    machine = {"cpu": platform.processor(), "torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "threads": torch.get_num_threads()}  # fmt: skip
    inf = None
    if not args.no_benchmark:
        devices = ["cpu"] if args.cpu_only or not torch.cuda.is_available() else ["cuda", "cpu"]
        inf = benchmark(args.rows, args.repeats, devices)
        inf.to_csv(tables / "cost_inference.csv", index=False)
    elif (tables / "cost_inference.csv").exists():
        inf = pd.read_csv(tables / "cost_inference.csv")  # the stored benchmark of an earlier run
    e1 = train[train.root == "e1"].set_index("label")
    merged = pd.DataFrame({"label": e1.index, "n_params": e1["n_params"].values, "train_s_per_seed": e1["seconds_mean"].values,
                           "train_s_min": e1["seconds_min"].values, "train_s_max": e1["seconds_max"].values})  # fmt: skip
    if inf is not None:
        for device in inf.device.unique():
            col = inf[inf.device == device].set_index("label")["ms_per_1000_windows"]
            merged[f"infer_ms_per_1000_{device}"] = merged["label"].map(col)
    merged = merged.merge(plat[["label", "seconds_per_unit_median", "us_per_vehicle_rollout_s", "n_rollouts"]], on="label", how="left")
    merged.to_csv(tables / "cost.csv", index=False)
    head = "| " + " | ".join(merged.columns) + " |\n|" + "---|" * len(merged.columns) + "\n"
    body = "".join("| " + " | ".join("" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in row) + " |\n"
                   for row in merged.itertuples(index=False))
    (OUT / "README.md").write_text(
        "# Cost of the laws (extension X2)\n\nMachine: " + json.dumps(machine) + "\n\n" + head + body
        + "\nTraining: artifacts/e1 (highD, recorded signals), mean over seeds; inference: highD test windows with a full "
        "5 s future (seed-0 checkpoints), median of repeated calls, plan samplers timed on fewer rows and scaled; platoon: "
        "artifacts/e3 timings of the cells " + ", ".join(PLATOON_VARIANTS) + " (one unit = one profile x 20 followers x "
        "R rollouts over the profile's time line; us per simulated vehicle-rollout-second).\n", encoding="utf-8")
    print("wrote", tables)


if __name__ == "__main__":
    main()
