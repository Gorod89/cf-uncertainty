"""Prediction tables of experiment E1 (docs/design.md, section 10.2).

The experiment runner writes them, the analysis reads them; the row order of
``windows/<dataset>.parquet`` is shared by every ``preds/<model>/seed<k>/<dataset>.parquet``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cf_unc.data.schema import EventSet
from cf_unc.data.windows import DEFAULT_TARGET, TARGET_STEPS, WindowSet, check_target, target_speed, target_values

WINDOW_INT = ("event_index", "t_index", "event", "driver", "fold")
WINDOW_FLOAT = ("s", "dv", "v", "v_lead", "a_prev", "y")
# columns added after E1 (target option): tables written before lack them, every reader must tolerate that
WINDOW_EXTRA = ("y_prev",)
WINDOW_COLUMNS = (*WINDOW_INT, "subset", *WINDOW_FLOAT, *WINDOW_EXTRA)  # column order of windows_table
PRED_FLOAT = ("mean", "var", "aleatoric_var", "epistemic_var", "nll", "crps", "width", "pit")
PRED_COLUMNS = (*PRED_FLOAT[:6], "covered", *PRED_FLOAT[6:])
SUBSET_KEYS = ("driver_mode", "follower_type")  # event meta fields that define analysis subsets


def event_subset(meta: Mapping[str, Any]) -> str:
    """Analysis subset of an event: OpenACC driver mode, Waymo follower type, else ``"all"``."""
    for key in SUBSET_KEYS:
        if key in meta and meta[key] is not None:
            return str(meta[key])
    return "all"


def windows_table(
    events: EventSet, windows: WindowSet, folds: Mapping[str, int] | None = None, target: str = DEFAULT_TARGET
) -> tuple[pd.DataFrame, dict]:
    """State columns of the evaluated windows and the decoding tables (event ids, drivers, subsets).

    ``folds`` maps event ids to the fold of a cf-stability split file (column ``fold``, -1 when unknown).
    ``target`` names the target of ``windows.y`` (column ``y``); column ``y_prev`` is the same target one step of
    its own length earlier (``TARGET_STEPS``), the persistence predictor: ``a[t-1]`` for ``"a"``,
    ``(v[t] - v[t-10]) / (10 DT)`` for ``"mean_1s"``; NaN when the event has too few earlier samples (cannot
    happen with a 3 s history). ``a_prev = a[t-1]`` whatever the target.
    """
    steps = TARGET_STEPS[check_target(target)]
    evs = list(events)
    n = len(windows)
    ev_idx = windows.event_index.astype(np.int64)
    t_idx = windows.t_index.astype(np.int64)
    a_prev = np.full(n, np.nan, dtype=np.float32)
    y_prev = np.full(n, np.nan, dtype=np.float32)
    v_lead = np.empty(n, dtype=np.float32)
    order = np.argsort(ev_idx, kind="stable")
    bounds = np.flatnonzero(np.diff(ev_idx[order])) + 1
    for rows in np.split(order, bounds) if n else []:
        ev = evs[int(ev_idx[rows[0]])]
        t = t_idx[rows]
        v_lead[rows] = ev.v_lead[t]
        ok = t >= 1
        a_prev[rows[ok]] = ev.a[t[ok] - 1]
        ok = t >= steps
        y_prev[rows[ok]] = target_values(ev.a, target_speed(ev), t[ok] - steps, target)  # v, or a recorded v_target
    fold_of_event = np.array([-1 if folds is None else int(folds.get(ev.event_id, -1)) for ev in evs], dtype=np.int64)
    subsets = sorted({event_subset(ev.meta) for ev in evs})
    subset_of_event = np.array([subsets.index(event_subset(ev.meta)) for ev in evs], dtype=np.int64)
    frame = pd.DataFrame(
        {
            "event_index": ev_idx.astype(np.int32),
            "t_index": t_idx.astype(np.int32),
            "event": ev_idx.astype(np.int32),
            "driver": windows.driver.astype(np.int32),
            "fold": fold_of_event[ev_idx].astype(np.int8) if n else np.empty(0, dtype=np.int8),
            "subset": pd.Categorical.from_codes(subset_of_event[ev_idx] if n else [], categories=subsets),
            "s": windows.X[:, -1, 0],
            "dv": windows.X[:, -1, 1],
            "v": windows.X[:, -1, 2],
            "v_lead": v_lead,
            "a_prev": a_prev,
            "y": windows.y.astype(np.float32),
            "y_prev": y_prev,
        }
    )
    meta = {
        "n_windows": int(n),
        "n_events": len(evs),
        "event_ids": [ev.event_id for ev in evs],
        "driver_names": list(windows.driver_names),
        "subsets": subsets,
        "hist_len": windows.hist_len,
        "target": target,
    }
    return frame, meta


def windows_target(meta: Mapping[str, Any]) -> str:
    """Target of a stored windows table from its meta (tables written before the target option: ``"a"``)."""
    spec = meta.get("spec")
    spec_target = spec.get("target") if isinstance(spec, Mapping) else None
    return str(meta.get("target") or spec_target or DEFAULT_TARGET)


def root_target(root: str | Path) -> str:
    """Target of the windows tables of an experiment root (the first ``windows/*_meta.json``; none: ``"a"``)."""
    for path in sorted((Path(root) / "windows").glob("*_meta.json")):
        return windows_target(json.loads(path.read_text(encoding="utf-8")))
    return DEFAULT_TARGET


def pred_table(dist: Any, scores: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Prediction columns from a predictive law and ``cf_unc.eval.metrics.summarize`` scores."""
    n = len(dist)
    columns: dict[str, np.ndarray] = {
        "mean": np.asarray(dist.mean()),
        "var": np.asarray(scores["var"]),
        "aleatoric_var": _optional(scores.get("aleatoric_var"), n),
        "epistemic_var": _optional(scores.get("epistemic_var"), n),
        "nll": np.asarray(scores["nll"]),
        "crps": np.asarray(scores["crps"]),
        "covered": np.nan_to_num(np.asarray(scores["covered"]), nan=0.0).astype(np.uint8),
        "width": np.asarray(scores["width"]),
        "pit": np.asarray(scores["pit"]),
    }
    frame = pd.DataFrame({k: (v if k == "covered" else v.astype(np.float32)) for k, v in columns.items()})
    return frame[list(PRED_COLUMNS)]


def _optional(values: np.ndarray | None, n: int) -> np.ndarray:
    return np.full(n, np.nan) if values is None else np.asarray(values, dtype=np.float64)


# ------------------------------------------------------------------------------ paths
def windows_path(root: str | Path, dataset: str) -> Path:
    return Path(root) / "windows" / f"{dataset}.parquet"


def preds_path(root: str | Path, model: str, seed: int, dataset: str) -> Path:
    return Path(root) / "preds" / model / f"seed{int(seed)}" / f"{dataset}.parquet"


def save_windows(root: str | Path, dataset: str, frame: pd.DataFrame, meta: Mapping[str, Any]) -> Path:
    path = windows_path(root, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    path.with_name(f"{dataset}_meta.json").write_text(json.dumps(dict(meta), indent=1), encoding="utf-8")
    return path


def load_windows(root: str | Path, dataset: str) -> tuple[pd.DataFrame, dict]:
    path = windows_path(root, dataset)
    meta = json.loads(path.with_name(f"{dataset}_meta.json").read_text(encoding="utf-8"))
    return pd.read_parquet(path), meta


def save_preds(root: str | Path, model: str, seed: int, dataset: str, frame: pd.DataFrame) -> Path:
    missing = [c for c in PRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"prediction table lacks columns {missing}")
    path = preds_path(root, model, seed, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame[list(PRED_COLUMNS)].to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)  # a table is either complete or absent (the runner resumes on absence)
    return path


def load_preds(root: str | Path, model: str, seed: int, dataset: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(preds_path(root, model, seed, dataset), columns=columns)


def available(root: str | Path) -> pd.DataFrame:
    """One row per stored prediction table: model, seed, dataset, path."""
    rows = []
    for path in sorted((Path(root) / "preds").glob("*/seed*/*.parquet")):
        rows.append(
            {"model": path.parts[-3], "seed": int(path.parts[-2][4:]), "dataset": path.stem, "path": str(path)}
        )
    return pd.DataFrame(rows, columns=["model", "seed", "dataset", "path"])
