"""Rollout tables of experiment E2 (docs/design.md, section 11.3).

::

    <root>/<target>/events/<dataset>.parquet                     one row per rolled event (EVENT_COLUMNS)
    <root>/<target>/rollouts/<model>/seed<k>/<dataset>.parquet   one row per (event, horizon) (ROLLOUT_COLUMNS),
                                                                  events in the order of the events table

Quantiles of the rollout gaps use the (S + 1) plotting positions (numpy ``method="weibull"``), the columns
``s_lin05`` / ``s_lin95`` the linear interpolation of numpy's default (comparison only). Gap and speed
statistics are float64 (the tables are small), ``horizon_s`` and ``collided`` float32, counts int32.
Writers replace files atomically (a table is complete or absent; runners resume on absence).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cf_unc.eval.metrics import crps_samples
from cf_unc.rollout.engine import RolloutResult

__all__ = [
    "EVENT_COLUMNS",
    "QUANTILE_LEVELS",
    "ROLLOUT_COLUMNS",
    "available",
    "events_path",
    "load_events_table",
    "load_rollout",
    "rollout_path",
    "save_events_table",
    "save_rollout",
    "summarize_rollout",
]

QUANTILE_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
QUANTILE_COLUMNS = ("s_q05", "s_q25", "s_q50", "s_q75", "s_q95")
EVENT_COLUMNS = ("event", "event_id", "driver", "driver_name", "subset", "fold", "t0", "order")
ROLLOUT_COLUMNS = (
    "event", "horizon_s", "truth_s", "truth_v", *QUANTILE_COLUMNS, "s_lin05", "s_lin95",
    "s_mean", "s_std", "s_min", "s_max", "v_mean", "v_std", "rank", "n_rollouts", "crps_s", "collided",
)  # fmt: skip


def summarize_rollout(result: RolloutResult, events_meta: pd.DataFrame) -> pd.DataFrame:
    """Rollout table (ROLLOUT_COLUMNS): one row per event and horizon, events in the order of ``result``.

    events_meta: the events table rows of the rolled events, same order (column ``event`` gives the codes).
    ``s_std`` / ``v_std`` are sample standard deviations over the R rollouts (ddof = 1); ``rank`` = number of
    rollouts with gap <= truth; ``crps_s`` = fair sample CRPS of the gap [m]; ``collided`` = share of rollouts
    whose first collision is at or before the horizon.
    """
    e, r, n_h = result.s.shape
    if len(events_meta) != e:
        raise ValueError(f"events_meta has {len(events_meta)} rows, the rollout {e} events")
    s = np.moveaxis(result.s, 2, 1).reshape(e * n_h, r)  # rows (event, horizon), columns rollouts
    v = np.moveaxis(result.v, 2, 1).reshape(e * n_h, r)
    truth_s = result.truth_s.reshape(e * n_h)
    q_w = np.quantile(s, QUANTILE_LEVELS, axis=1, method="weibull")  # (Q, rows)
    q_lin = np.quantile(s, (0.05, 0.95), axis=1, method="linear")
    ddof = 1 if r > 1 else 0
    columns: dict[str, Any] = {
        "event": np.repeat(events_meta["event"].to_numpy(np.int64), n_h).astype(np.int32),
        "horizon_s": np.tile(result.horizons_s, e).astype(np.float32),
        "truth_s": truth_s,
        "truth_v": result.truth_v.reshape(e * n_h),
    }
    columns.update({name: q_w[i] for i, name in enumerate(QUANTILE_COLUMNS)})
    columns.update(
        s_lin05=q_lin[0],
        s_lin95=q_lin[1],
        s_mean=s.mean(axis=1),
        s_std=s.std(axis=1, ddof=ddof),
        s_min=s.min(axis=1),
        s_max=s.max(axis=1),
        v_mean=v.mean(axis=1),
        v_std=v.std(axis=1, ddof=ddof),
        rank=(s <= truth_s[:, None]).sum(axis=1).astype(np.int32),
        n_rollouts=np.full(e * n_h, r, dtype=np.int32),
        crps_s=crps_samples(s, truth_s, fair=r > 1),
        collided=result.collided().reshape(e * n_h).astype(np.float32),
    )
    return pd.DataFrame(columns)[list(ROLLOUT_COLUMNS)]


# ------------------------------------------------------------------------------------------ paths
def events_path(root: str | Path, target: str, dataset: str) -> Path:
    return Path(root) / target / "events" / f"{dataset}.parquet"


def rollout_path(root: str | Path, target: str, model: str, seed: int, dataset: str) -> Path:
    return Path(root) / target / "rollouts" / model / f"seed{int(seed)}" / f"{dataset}.parquet"


def _write_atomic(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)
    return path


def _check(frame: pd.DataFrame, columns: tuple[str, ...], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{what} lacks columns {missing}")


def save_events_table(root: str | Path, target: str, dataset: str, frame: pd.DataFrame) -> Path:
    _check(frame, EVENT_COLUMNS, "events table")
    return _write_atomic(frame[list(EVENT_COLUMNS)], events_path(root, target, dataset))


def load_events_table(root: str | Path, target: str, dataset: str) -> pd.DataFrame:
    return pd.read_parquet(events_path(root, target, dataset))


def save_rollout(root: str | Path, target: str, model: str, seed: int, dataset: str, frame: pd.DataFrame) -> Path:
    _check(frame, ROLLOUT_COLUMNS, "rollout table")
    return _write_atomic(frame[list(ROLLOUT_COLUMNS)], rollout_path(root, target, model, seed, dataset))


def load_rollout(root: str | Path, target: str, model: str, seed: int, dataset: str) -> pd.DataFrame:
    return pd.read_parquet(rollout_path(root, target, model, seed, dataset))


def available(root: str | Path, target: str) -> pd.DataFrame:
    """One row per stored rollout table of ``target``: model, seed, dataset, path."""
    rows = [
        {"model": p.parts[-3], "seed": int(p.parts[-2][4:]), "dataset": p.stem, "path": str(p)}
        for p in sorted((Path(root) / target / "rollouts").glob("*/seed*/*.parquet"))
    ]
    return pd.DataFrame(rows, columns=["model", "seed", "dataset", "path"])
