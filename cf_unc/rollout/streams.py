"""Chronological streams of the rolled events: order and issue times for the information-budget analysis of ACI
(review item R8; ``cf_unc.rollout.calibration.coverage_table``, variants ``aci_online_chrono`` / ``*_delayed``).

The interval of an E2 event is issued when its rollout starts, ``t0`` samples (column ``t0`` of the events table,
``HIST_LEN - 1``) after the start of the event, and its label (the gap at horizon h) exists h seconds later. The
clock of a data set comes from the ``meta`` of its events (``data/events/<set>/events.parquet``):

* ``absolute``: one clock for the whole data set: NGSIM I-80 and US-101 (``meta.t0`` = seconds since frame 0 of the
  first of the three recording periods, ``period`` given; cf-stability's reconstruction), so labels and issue times
  are comparable across every event.
* ``per_run``: one clock per recording (OpenACC: ``meta.t0`` = time within the run file, runs ordered by date,
  campaign and run name in natural order; Waymo: ``meta.t0`` within a 20 s segment, segments ordered by
  ``segment_id``). The runs are put one after the other on a common clock with a gap of :data:`RUN_GAP_S`, so every
  label of a run is available before the next run starts (separate recordings).
* ``untimed``: no comparable time (the FollowNet highD events start at ``t0 = 0``): the events are ordered by their
  index in the source file (``meta.source_index``, the order of extraction) and the delay of a label is counted in
  events (``cf_unc.rollout.calibration.delayed_availability``); ``t_issue`` is NaN.

The order of an untimed set and the run order of a per-run set are proxies of the recording order, not measured
times; only the ``absolute`` clock is a true time line.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cf_unc.constants import DT

__all__ = ["CLOCKS", "RUN_GAP_S", "event_meta", "natural_key", "stream_order"]

CLOCKS = ("absolute", "per_run", "untimed")
RUN_GAP_S = 1.0e5  # s between two runs on the common clock of a per-run data set (longer than any horizon)


def natural_key(text: str) -> tuple[Any, ...]:
    """Sort key that orders the numbers inside a name numerically (``platoon3`` before ``platoon10``)."""
    parts = re.split(r"(\d+)", str(text))
    return tuple(int(p) if p.isdigit() else p for p in parts)


def event_meta(events_dir: str | Path, event_ids: Any = None) -> dict[str, dict[str, Any]]:
    """event_id -> meta dict of the events of an event directory (``events.parquet``; only ``event_ids`` if given)."""
    frame = pd.read_parquet(Path(events_dir) / "events.parquet", columns=["event_id", "meta_json"])
    if event_ids is not None:
        wanted = set(str(e) for e in event_ids)
        frame = frame[frame["event_id"].astype(str).isin(wanted)]
    return {str(e): json.loads(m) for e, m in zip(frame["event_id"], frame["meta_json"])}


def _clock(meta: Mapping[str, Mapping[str, Any]], ids: list[str]) -> tuple[str, np.ndarray, list[Any]]:
    """(clock kind, time within the run [s] (n,), run key per event) of the events ``ids``."""
    rows = [meta.get(i) for i in ids]
    if any(r is None for r in rows) or not rows:
        return "untimed", np.full(len(ids), np.nan), [None] * len(ids)
    t = np.array([float(r.get("t0", np.nan)) for r in rows], dtype=np.float64)
    if all("period" in r for r in rows) and np.all(np.isfinite(t)):
        return "absolute", t, [None] * len(ids)
    if all("run" in r for r in rows) and np.all(np.isfinite(t)):
        keys = [(str(r.get("date") or ""), str(r.get("campaign") or ""), natural_key(r["run"])) for r in rows]
        return "per_run", t, keys
    if all("segment_id" in r for r in rows) and np.all(np.isfinite(t)):
        return "per_run", t, [(int(r["segment_id"]),) for r in rows]
    return "untimed", np.full(len(ids), np.nan), [None] * len(ids)


def stream_order(events: pd.DataFrame, meta: Mapping[str, Mapping[str, Any]] | None) -> tuple[pd.DataFrame, str]:
    """The events table with ``chrono`` (rank in chronological order, 0..n-1) and ``t_issue`` (issue time [s] on
    the common clock; NaN for an untimed set), and the kind of clock (module docstring).

    events: an E2 events table (``event``, ``event_id``, ``t0``); meta: event_id -> meta of the data set's events
    (None or incomplete: untimed, ordered by ``source_index`` when every event has one, else by ``event``).
    """
    out = events.copy()
    ids = [str(e) for e in out["event_id"]]
    kind, t, runs = _clock(meta or {}, ids)
    start = out["t0"].to_numpy(np.float64) * DT if "t0" in out.columns else np.zeros(len(out))
    position = out["event"].to_numpy(np.int64)
    if kind == "untimed":
        rows = [(meta or {}).get(i) for i in ids]
        if rows and all(r is not None and "source_index" in r for r in rows):
            key = np.array([int(r["source_index"]) for r in rows], dtype=np.int64)  # type: ignore[index]
        else:
            key = position
        order = np.lexsort((position, key))
        t_issue = np.full(len(out), np.nan)
    else:
        t_abs = t.copy()
        if kind == "per_run":
            offset = 0.0
            for run in sorted(set(runs)):
                sel = np.array([r == run for r in runs])
                t_abs[sel] = offset + (t[sel] - t[sel].min())
                offset = float(t_abs[sel].max()) + RUN_GAP_S
        t_issue = t_abs + start
        order = np.lexsort((position, t_issue))
    chrono = np.empty(len(out), dtype=np.int64)
    chrono[order] = np.arange(len(out))
    out["chrono"] = chrono
    out["t_issue"] = t_issue
    return out, kind
