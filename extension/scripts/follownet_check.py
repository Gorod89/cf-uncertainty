#!/usr/bin/env python3
"""FollowNet-protocol check of the laws (extension X5, item 1, extension/docs/PLAN.md).

Every test event is simulated in closed loop behind its recorded leader with the FollowNet rules as fixed in
FollowNet-2 (semi-implicit Euler at 10 Hz, acceleration clipped to [-8, 4] m/s^2, the first 3 s observed, a collision
is s <= 0 at any simulated sample), deterministically: the point prediction of the law at every step (the predictive
mean; the median of the quantile law; for the diffusion laws the mean of the 32 sampled 5 s plans, re-planned every
5 s, with the recorded leader future for diffusion_cf and without it for diffusion_cf_hist). Scores as in FollowNet
(Chen et al. 2023, Tables 2 and 3): the MSE of the spacing over the simulated samples of an event, averaged over the
events, and the collision rate in per mille; the RMSE of the speed is added. Sets: the highD test fold of the roots
(the FollowNet highD events, driver-disjoint from the training folds) and the FollowNet NGSIM I-80 events (every
event, never trained on). Checkpoints: artifacts/e1 (every model, seeds 0-4) and artifacts/e1_ctrl (the controls).

Writes extension/artifacts/ext_followcheck/tables/follownet_check.csv (per set, model and seed), summary.csv (mean
and range over the seeds) and README.md with the published FollowNet rows and the FollowNet-2 reference rows next to
ours (different splits: stated there).

    PYTHONUTF8=1 ./.venv/Scripts/python.exe extension/scripts/follownet_check.py [--device cuda] [--seeds 0 1 2 3 4]
        [--models ...] [--max-events N] [--sets highd ngsim_fn]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cf_unc.constants import A_MAX, A_MIN, DT, HIST_LEN, S_EPS, SEQ_LEN  # noqa: E402
from cf_unc.dist import QuantileDist  # noqa: E402
from cf_unc.experiments.e1 import Data, load_config  # noqa: E402
from cf_unc.models import load_model  # noqa: E402

OUT = ROOT / "extension/artifacts/ext_followcheck"
ROOTS = {"e1": "artifacts/e1", "e1_ctrl": "artifacts/e1_ctrl"}
CONTROLS = ("diffusion_cf_hist", "lstm_hetero_frozen")  # read from e1_ctrl; the other labels from e1
SETS = {"highd": ("highd", "test"), "ngsim_fn": ("ngsim_fn", "all")}
WARMUP = HIST_LEN  # observed samples; the first simulated sample has index WARMUP (FollowNet-2: index warmup + 1 with warmup = 29)

PUBLISHED = {  # FollowNet, Chen et al. 2023, Table 2 (MSE of spacing, m^2) and Table 3 (collision rate, per mille)
    "highd": {"GHR": (23.76, 0.0), "IDM": (27.50, 0.0), "NN": (24.06, 39.87), "LSTM": (22.90, 3.72), "DDPG": (38.52, 0.0)},
    "ngsim_fn": {"GHR": (62.18, 0.0), "IDM": (54.97, 0.0), "NN": (29.00, 3.46), "LSTM": (23.86, 17.30), "DDPG": (46.93, 0.0)},
}
FOLLOWNET2 = {  # FollowNet-2 (plan P4, results/m2/REPORT.md), FollowNet highD events, reconstructed, split driver-0, 2503 test events:
    # closed-loop RMSE of the spacing [m] and collision share
    "highd": {"idm": (4.005, 0.000), "lstm": (3.482, 0.002), "lstm_loop": (3.532, 0.003), "seq2seq": (3.093, 0.002),
              "diffusion": (3.361, 0.006), "pidl": (8.892, 0.129), "ddpg": (11.097, 0.000)},
}


def point(model: Any, hist: np.ndarray, future: np.ndarray | None) -> np.ndarray:
    """Point prediction [m/s^2] of the law for histories (N, H, 3): mean, or the median of a quantile law."""
    kwargs = {"v_lead_future": future} if getattr(model, "needs_leader_future", False) else {}
    dist = model.predict_dist(hist, batch_size=8192, **kwargs)
    if isinstance(dist, QuantileDist):
        return np.asarray(dist.quantile([0.5]))[:, 0].astype(np.float64)
    return np.asarray(dist.mean()).astype(np.float64)


def _padded(events: list, attr: str, length: int) -> np.ndarray:
    """(E, length) float64 array of ``attr`` of every event, padded with the event's last value."""
    out = np.empty((len(events), length), dtype=np.float64)
    for i, ev in enumerate(events):
        a = np.asarray(getattr(ev, attr), dtype=np.float64)
        out[i, : a.size] = a
        out[i, a.size :] = a[-1]
    return out


def simulate(model: Any, events: list, device: str, batch_events: int = 4096) -> pd.DataFrame:
    """One row per event: n simulated samples, SSE of the spacing and of the speed, collision flag.

    Events of different lengths are simulated together, padded to the longest one (the padded samples repeat the
    last recorded value and are never scored); an event is advanced while ``k + 1 < len(event)``.
    """
    plan = bool(getattr(model, "plan_sampler", False))
    events = [ev for ev in events if len(ev) >= WARMUP + 2]
    rows: list[dict[str, Any]] = []
    for start in range(0, len(events), batch_events):
        evs = events[start : start + batch_events]
        e = len(evs)
        lengths = np.array([len(ev) for ev in evs], dtype=np.int64)
        n = int(lengths.max())
        s_true, v_true = _padded(evs, "s", n), _padded(evs, "v", n)
        v_lead, x_lead = _padded(evs, "v_lead", n), _padded(evs, "x_lead", n)
        s, v = s_true.copy(), v_true.copy()
        x = x_lead - s  # follower position in the event frame
        collided = np.zeros(e, dtype=bool)
        plan_a = np.zeros((e, SEQ_LEN))
        last = lengths - 1  # last valid sample of every event
        for k in range(WARMUP - 1, n - 1):
            active = k + 1 <= last
            if not active.any():
                break
            hist = np.stack([np.maximum(s[:, k - HIST_LEN + 1 : k + 1], S_EPS), v[:, k - HIST_LEN + 1 : k + 1] - v_lead[:, k - HIST_LEN + 1 : k + 1],
                             v[:, k - HIST_LEN + 1 : k + 1]], axis=-1).astype(np.float32)  # fmt: skip
            if plan:
                j = (k - (WARMUP - 1)) % SEQ_LEN
                if j == 0:
                    cols = np.minimum(np.arange(k + 1, k + 1 + SEQ_LEN)[None, :], last[:, None])  # (E, L), held at the event's end
                    fut = np.take_along_axis(v_lead, cols, axis=1).astype(np.float32)
                    seq = model.sample_sequences(hist, fut)  # (E, S, L), the model's seeded stream
                    plan_a = seq.mean(axis=1).astype(np.float64)
                a = plan_a[:, j]
            else:
                a = point(model, hist, None)
            a = np.clip(a, A_MIN, A_MAX)
            v_next = np.maximum(v[:, k] + a * DT, 0.0)
            x_next = x[:, k] + v_next * DT
            v[active, k + 1] = v_next[active]
            x[active, k + 1] = x_next[active]
            s[active, k + 1] = x_lead[active, k + 1] - x_next[active]
            collided[active] |= s[active, k + 1] <= 0.0
        for r, ev in enumerate(evs):
            sim = slice(WARMUP, int(lengths[r]))
            err_s = s[r, sim] - s_true[r, sim]
            err_v = v[r, sim] - v_true[r, sim]
            rows.append({"event_id": ev.event_id, "n": int(lengths[r]) - WARMUP, "sse_s": float((err_s ** 2).sum()),
                         "sse_v": float((err_v ** 2).sum()), "collided": bool(collided[r])})  # fmt: skip
    return pd.DataFrame(rows)


def scores(table: pd.DataFrame) -> dict[str, float]:
    mse_event = table["sse_s"] / table["n"]
    return {"mse_s": float(mse_event.mean()), "mse_s_median": float(mse_event.median()),
            "rmse_s_pooled": float(np.sqrt(table["sse_s"].sum() / table["n"].sum())),
            "rmse_v_pooled": float(np.sqrt(table["sse_v"].sum() / table["n"].sum())),
            "collision_permille": 1000.0 * float(table["collided"].mean()), "n_collisions": int(table["collided"].sum()),
            "n_events": int(len(table))}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args()
    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs/experiment/e1.yaml")
    data = Data(cfg)
    labels = args.models or sorted(p.name for p in (ROOT / ROOTS["e1"] / "checkpoints").iterdir()) + list(CONTROLS)
    seed_independent = set(cfg.get("seed_independent") or [])
    out = []
    for set_name in args.sets:
        name, part = SETS[set_name]
        events = list(data.split(name, part))
        if args.max_events:
            events = events[: args.max_events]
        print(f"{set_name}: {len(events)} events", flush=True)
        for label in labels:
            root = ROOTS["e1_ctrl"] if label in CONTROLS else ROOTS["e1"]
            for seed in ([args.seeds[0]] if label in seed_independent else args.seeds):
                path = ROOT / root / "checkpoints" / label / f"seed{seed}" / "model.pt"
                if not path.exists():
                    print(f"  {label} seed{seed}: no checkpoint", flush=True)
                    continue
                started = time.perf_counter()
                model = load_model(path, device=args.device)
                table = simulate(model, events, args.device)
                sc = scores(table)
                out.append({"set": set_name, "label": label, "seed": seed, **sc, "seconds": time.perf_counter() - started})
                print(f"  {label:22s} seed{seed}  MSE_s {sc['mse_s']:8.2f}  RMSE_s {sc['rmse_s_pooled']:6.2f}  "
                      f"collisions {sc['collision_permille']:7.2f} permille ({sc['n_collisions']})  {out[-1]['seconds']:.0f} s", flush=True)
                del model
                pd.DataFrame(out).to_csv(tables / "follownet_check.csv", index=False)
    df = pd.DataFrame(out)
    summary = df.groupby(["set", "label"], as_index=False).agg(
        seeds=("seed", "nunique"), mse_s=("mse_s", "mean"), mse_s_min=("mse_s", "min"), mse_s_max=("mse_s", "max"),
        rmse_s_pooled=("rmse_s_pooled", "mean"), rmse_v_pooled=("rmse_v_pooled", "mean"),
        collision_permille=("collision_permille", "mean"), collision_min=("collision_permille", "min"),
        collision_max=("collision_permille", "max"), n_events=("n_events", "first"))  # fmt: skip
    summary.to_csv(tables / "summary.csv", index=False)
    lines = ["# FollowNet-protocol check (extension X5, item 1)", "",
             "Closed loop behind the recorded leader (semi-implicit Euler, 10 Hz, acceleration clipped to [-8, 4] m/s^2, "
             "3 s observed, collision = s <= 0), point predictions of the laws (mean; median of the quantile law; mean of the "
             "32 sampled plans of the diffusion laws, re-planned every 5 s). MSE of the spacing per event averaged over "
             "events [m^2], collision rate [per mille]; mean over the seeds 0-4 (min-max).", ""]
    for set_name in args.sets:
        s = summary[summary.set == set_name]
        lines += [f"## {set_name} ({int(s.n_events.iloc[0]) if len(s) else 0} events)", "",
                  "| Law | MSE spacing | min-max | RMSE spacing (pooled) | collisions, per mille | min-max |", "|---|---|---|---|---|---|"]
        for _, r in s.iterrows():
            lines.append(f"| {r.label} | {r.mse_s:.2f} | {r.mse_s_min:.2f}-{r.mse_s_max:.2f} | {r.rmse_s_pooled:.2f} | "
                         f"{r.collision_permille:.2f} | {r.collision_min:.2f}-{r.collision_max:.2f} |")
        lines += ["", "Published FollowNet rows (Chen et al. 2023, Tables 2 and 3; the benchmark's own test split, 15 s events):", "",
                  "| Model | MSE spacing | collisions, per mille |", "|---|---|---|"]
        lines += [f"| {m} | {mse:.2f} | {col:.2f} |" for m, (mse, col) in PUBLISHED[set_name].items()]
        if set_name in FOLLOWNET2:
            lines += ["", "FollowNet-2 reference implementations (plan P4, FollowNet highD events, reconstructed signals, split "
                      "driver-0, 2503 test events): closed-loop RMSE of the spacing [m] and collision share:", "",
                      "| Model | RMSE spacing | MSE (RMSE^2) | collision share |", "|---|---|---|---|"]
            lines += [f"| {m} | {r:.3f} | {r * r:.2f} | {c:.3f} |" for m, (r, c) in FOLLOWNET2[set_name].items()]
        lines.append("")
    lines += ["Protocol differences: the published rows use FollowNet's own train/validation/test split and its unified "
              "15 s events; ours use the driver-disjoint fold 0 of the roots (highD: test fold; NGSIM I-80: every FollowNet "
              "event, a shifted set for laws trained on highD); FollowNet-2 uses its reconstructed signals and its own split. "
              "The published MSE is the mean over events of the per-event MSE; the pooled RMSE weights every sample equally.", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "settings.json").write_text(json.dumps({"device": args.device, "seeds": args.seeds, "sets": args.sets,
                                                   "max_events": args.max_events, "warmup": WARMUP,
                                                   "clip": [A_MIN, A_MAX]}, indent=1), encoding="utf-8")  # fmt: skip
    print("wrote", OUT)


if __name__ == "__main__":
    main()
