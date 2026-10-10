"""Experiment E2 (docs/design.md, section 11): closed-loop stochastic rollouts of the E1 models, per target.

``run_e2(cfg)`` (config ``configs/experiment/e2.yaml``) writes under ``cfg.root``::

    config.yaml                                          resolved config of the last call (read by the analysis)
    <target>/spec.json                                   settings that define the target's tables (checked on resume)
    <target>/events/<dataset>.parquet                    the rolled events (cf_unc.rollout.tables.EVENT_COLUMNS)
    <target>/rollouts/<label>/spec.json                  sampler settings of a model label (checked on resume)
    <target>/rollouts/<label>/seed<k>/<dataset>.parquet  one row per (event, horizon)
    logs/run.log, logs/errors.log, logs/timings.jsonl, status.json

Events: per data set a seeded random subsample of ``n_events`` events (seed from ``rollout.event_seed`` and the
data set name) among the events long enough for ``t0 + horizon`` with ``t0 = HIST_LEN - 1``; the events table
lists them in the order of the data set (column ``event`` = position in the data set split, the code of the E1
windows tables), ``order`` = rank in the random draw (the order of the ACI runs). The same events serve every
model, seed and target. Rollouts: ``n_rollouts`` per event (``n_rollouts_sampled`` for plan samplers, DiffusionCF),
sampler mode ``sampler_mode[label]`` (default ``sampler_mode.default``), hold ``targets[t].hold_steps`` (1 for plan
samplers, which follow their plan step by step), random streams seeded from ``(rollout.seed, model seed, data set)``.
Seed-independent checkpoints (IDM baselines) are rolled once per seed anyway (the rollouts are random).

Resumable at the level of (target, label, seed, data set) tables: existing tables are skipped, never rewritten. A
target whose checkpoint directory does not exist is skipped with a warning; a failing unit is logged with its
traceback to ``logs/errors.log`` and the run continues; ``run_e2`` returns the final status (``failures``).

Signal variant (design.md, section 12): with the experiment-level ``signal: harmonised`` (and ``harmonise``
settings) the events are harmonised at load time (``cf_unc.experiments.e1.Data``); the variant is part of the
target spec (stored only when not raw) and every checkpoint must have been trained on the signal key of the training
data set ``train.data`` (its ``train_log.json``; a log without ``signal`` counts as raw; without ``train.data``: the
experiment-level key), so raw and harmonised artefacts are never mixed. Signal ``matched`` (design.md 15.1): every
data set is harmonised with its window of the matching file ``match.file`` (``cf_unc.experiments.e1.signal_key``);
the target spec stores the experiment-level key with every window, the checkpoints are checked against the key of
the training data set (the reference keeps the harmonised window, so harmonised checkpoints pass), and the runner
refuses to run (``MatchingError``, nothing is created) without a valid matching file or without ``train.data``.
A config may inherit another one with ``base: <file>`` (as in E1; an inherited entry of ``datasets``, ``targets``,
``e1_roots`` or ``variants`` set to null is removed, e.g. ``targets: {mean_1s: null}`` runs target ``a`` only).

Pooled training domain (design.md 16.1, experiment E6): ``train.data`` lists the training data sets of the
checkpoints; every checkpoint must have been trained on the ordered list of their signal keys
(``cf_unc.experiments.e1.training_signal_key``), which never equals the key of one data set. A rolled data set of a
training set is drawn from its test fold (or, as a calibration set, its validation fold) only: ``split: all`` or
``train`` would roll training events and is refused before anything is created; the held-out set and the other data
sets are rolled as configured. The target spec also stores the training data sets and their signal key
(``train_data``, ``train_signal``), so the tables of a root rolled with checkpoints of one training data set are never
resumed by a pooled config (or the other way round).

``dry_run_e2(cfg)`` (``scripts/run_e2.py --dry-run``, added with the review controls, design.md 18): what a run would do,
loading no events and writing nothing: per target whether its checkpoint directory exists, the target spec, the events
tables and the label specs compared with the stored ones, per label and seed whether the checkpoint exists and was
trained on the signal key of ``train.data``, and the tables done and missing.
"""

from __future__ import annotations

import gc
import json
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from cf_unc.constants import DT, HIST_LEN
from cf_unc.data.harmonise import MatchingError
from cf_unc.data.schema import Event, EventSet
from cf_unc.data.windows import driver_key
from cf_unc.experiments.e1 import (
    Data,
    RunLog,
    _display,
    _jsonable,
    _load_file,
    _write_json_atomic,
    check_matching,
    configure_torch,
    describe_signal,
    is_pooled,
    matching_path,
    pilot_path,
    seed_everything,
    signal_key,
    train_datasets,
    training_signal_key,
)
from cf_unc.maps.io import event_subset
from cf_unc.models import CFModel, load_model, resolve_device
from cf_unc.rollout import tables
from cf_unc.rollout.engine import RolloutConfig, rollout_events
from cf_unc.train.run import ROOT, _path, _plain

__all__ = [
    "DEFAULT_CONFIG",
    "PILOT",
    "E2Runner",
    "dry_run_e2",
    "events_table",
    "load_config",
    "run_e2",
    "select_events",
    "unit_seed",
]

DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "e2.yaml"
PILOT: dict[str, Any] = {"root": "artifacts/e2_pilot", "seeds": [0], "n_events": 100, "targets": ["a"]}
T0 = HIST_LEN - 1


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _crc(text: str) -> int:
    return zlib.crc32(str(text).encode("utf-8"))


# ------------------------------------------------------------------------------------------ config
def load_config(path: str | Path = DEFAULT_CONFIG, overrides: Sequence[str] = (), pilot: bool = False) -> DictConfig:
    """Experiment config: the YAML file (merged over its ``base`` file, if any), then ``PILOT`` (root
    ``<root>_pilot``, seed 0, 100 events per data set, target ``a``), then dotted ``key=value`` overrides."""
    cfg = _load_file(_path(path))
    if pilot:
        cfg.root = pilot_path(str(cfg.root)) if cfg.get("root") is not None else PILOT["root"]
        cfg.seeds = list(PILOT["seeds"])
        for name in cfg.datasets:
            cfg.datasets[name].n_events = int(PILOT["n_events"])
        cfg.targets = OmegaConf.create({k: v for k, v in cfg.targets.items() if k in PILOT["targets"]})
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([str(o) for o in overrides]))
    return cfg  # type: ignore[return-value]


def unit_seed(base: int, model_seed: int, dataset: str) -> int:
    """Rollout seed of (model seed, data set): the same for every model label and target (common random numbers
    between models that draw alike, e.g. the two IDM baselines)."""
    state = np.random.SeedSequence([abs(int(base)), abs(int(model_seed)), _crc(dataset)]).generate_state(1, np.uint64)[0]
    return int(state >> np.uint64(1))


# ------------------------------------------------------------------------------------------ events
def select_events(lengths: np.ndarray, n_events: int | None, seed: int, n_steps: int, t0: int = T0) -> np.ndarray:
    """Positions of the selected events in draw order: a seeded permutation of the events with ``t0 + n_steps <
    length``, cut at ``n_events`` (None: all)."""
    eligible = np.flatnonzero(t0 + int(n_steps) < np.asarray(lengths, dtype=np.int64))
    drawn = eligible[np.random.default_rng(int(seed)).permutation(eligible.size)]
    return drawn if n_events is None else drawn[: int(n_events)]


def events_table(split: EventSet, drawn: np.ndarray, folds: Mapping[str, int] | None, t0: int = T0) -> pd.DataFrame:
    """Events table (``tables.EVENT_COLUMNS``) of the events ``drawn`` (positions in ``split``, draw order), sorted by
    position. ``driver`` codes index the sorted driver names of the whole split (the codes of ``make_windows``)."""
    evs = list(split)
    names = sorted({driver_key(ev) for ev in evs})
    code = {n: i for i, n in enumerate(names)}
    pos = np.sort(np.asarray(drawn, dtype=np.int64))
    rank = np.empty(len(drawn), dtype=np.int64)
    rank[np.argsort(np.asarray(drawn, dtype=np.int64), kind="stable")] = np.arange(len(drawn))
    picked = [evs[i] for i in pos]
    return pd.DataFrame(
        {
            "event": pos.astype(np.int32),
            "event_id": [ev.event_id for ev in picked],
            "driver": np.array([code[driver_key(ev)] for ev in picked], dtype=np.int32),
            "driver_name": [driver_key(ev) for ev in picked],
            "subset": [event_subset(ev.meta) for ev in picked],
            "fold": np.array([-1 if folds is None else int(folds.get(ev.event_id, -1)) for ev in picked], dtype=np.int8),
            "t0": np.full(len(picked), t0, dtype=np.int32),
            "order": rank.astype(np.int32),
        }
    )


# ------------------------------------------------------------------------------------------ runner
class E2Runner:
    """State of one call of ``run_e2`` (module docstring); ``dry=True``: nothing is created or written (dry runs)."""

    def __init__(self, cfg: DictConfig, echo: bool = True, dry: bool = False) -> None:
        self.cfg = cfg
        self.root = _path(cfg.root)
        self.train_names = train_datasets(cfg)  # training data sets of the checkpoints (a list for a pooled domain)
        self.pooled = is_pooled(cfg)
        self.train_data = None if not self.train_names else ", ".join(self.train_names)  # for messages
        if str(cfg.get("signal") or "") in ("matched", "matched_causal") and self.train_data is None:
            raise MatchingError("signal matched: E2 needs train.data, the training data set of the checkpoints, whose "
                                "signal key the checkpoints are checked against (design.md 15.1)")  # fmt: skip
        # signal matched: the windows of the matching file; refuses (MatchingError) before anything is created
        self.windows = check_matching(cfg)
        self.check_pooled_splits()  # a pooled domain never rolls training events (ValueError, nothing created)
        self.dry = bool(dry)
        if not self.dry:
            self.root.mkdir(parents=True, exist_ok=True)
        self.log = RunLog(self.root, echo, write=not self.dry)
        self.device = resolve_device(str(cfg.get("device", "auto")))
        self.data = Data(cfg, self.log.line)
        self.variants = {str(k): dict(_plain(v)) for k, v in (cfg.get("variants") or {}).items()}
        self.labels = [str(m) for m in cfg.models] + [k for k in self.variants if k not in {str(m) for m in cfg.models}]
        self.seeds = [int(s) for s in cfg.seeds]
        self.datasets = [str(d) for d in cfg.datasets]
        self.targets = {str(k): dict(_plain(v)) for k, v in cfg.targets.items()}
        self.active: list[str] = []  # targets whose checkpoint directory exists
        self.skipped: dict[str, str] = {}
        self.events: dict[tuple[str, str], pd.DataFrame] = {}
        self.written: list[str] = []
        self.current: str | None = None
        self.started, self.started_at = time.perf_counter(), _now()
        r = cfg.rollout
        self.horizon_s, self.horizons_s = float(r.horizon_s), tuple(float(h) for h in r.horizons_s)
        self.n_steps = RolloutConfig(horizon_s=self.horizon_s, horizons_s=self.horizons_s).n_steps

    # ------------------------------------------------------------------ bookkeeping
    def guard(self, unit: str, fn: Callable[..., Any], *args: Any) -> Any:
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 - one failing unit must not stop the run
            self.log.error(unit, exc)
            return None

    def status(self, finished: bool = False) -> dict[str, Any]:
        per_target = {}
        for t in self.active:
            done = sum(
                tables.rollout_path(self.root, t, m, s, d).exists() for m in self.labels for s in self.seeds for d in self.datasets
            )
            per_target[t] = {"done": done, "total": len(self.labels) * len(self.seeds) * len(self.datasets)}
        return {
            "root": str(self.root),
            "signal": str(self.cfg.get("signal") or "raw"),
            "device": str(self.device),
            "started": self.started_at,
            "updated": _now(),
            "elapsed_s": round(time.perf_counter() - self.started, 1),
            "finished": finished,
            "current": None if finished else self.current,
            "done": sum(v["done"] for v in per_target.values()),
            "total": sum(v["total"] for v in per_target.values()),
            "targets": per_target,
            "skipped_targets": dict(self.skipped),
            "written_now": len(self.written),
            "failures": list(self.log.failures),
        }

    def write_status(self, finished: bool = False) -> dict[str, Any]:
        status = self.status(finished)
        _write_json_atomic(self.root / "status.json", status)
        return status

    def cleanup(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def timing(self, record: Mapping[str, Any]) -> None:
        with open(self.root / "logs" / "timings.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(_jsonable(dict(record))) + "\n")

    # ------------------------------------------------------------------ settings
    def check_pooled_splits(self) -> None:
        """Pooled training domain (design.md 16.1): every rolled data set of a training set uses its test fold (or
        its validation fold, a calibration set); ValueError for ``split: all`` or ``train``."""
        if not self.pooled:
            return
        for name, d in (self.cfg.get("datasets") or {}).items():
            part = str(d.get("split", "all"))
            if str(d.data) in self.train_names and part not in ("test", "val"):
                raise ValueError(f"datasets.{name}: {d.data} is a training data set of the pooled domain "
                                 f"{self.train_names}; its events are drawn from its test fold only (split test, or "
                                 f"val for a calibration set), not from split {part!r}")  # fmt: skip

    def label_spec(self, label: str) -> dict[str, Any]:
        """Base model (checkpoint directory), sampler mode and optional rollout count of a model label."""
        modes = self.cfg.get("sampler_mode") or {}
        extra = self.variants.get(label, {})
        base = str(extra.get("model", label))
        mode = str(extra.get("sampler_mode", modes.get(label, modes.get(base, modes.get("default", "per_step")))))
        return {"model": base, "sampler_mode": mode, "n_rollouts": extra.get("n_rollouts")}

    def target_spec(self, target: str) -> dict[str, Any]:
        """Settings that define the tables of ``target`` (``signal``: only when the experiment's signal is not raw)."""
        r = self.cfg.rollout
        sig = self.signal_key()
        return _jsonable({
            "target": target, "hold_steps": int(self.targets[target].get("hold_steps", 1)), "t0": T0, "dt": DT,
            "horizon_s": self.horizon_s, "horizons_s": list(self.horizons_s), "seed": int(r.seed),
            "event_seed": int(r.event_seed), "stream_events": int(r.get("stream_events", 100)),
            **({} if sig is None else {"signal": sig}),
            # a pooled domain (design.md 16.1): its data sets and training signal key (absent for one data set)
            **({"train_data": self.train_names, "train_signal": self.train_signal_key()} if self.pooled else {}),
        })  # fmt: skip

    def signal_key(self) -> dict[str, Any] | None:
        """``signal_key`` of the experiment-level ``signal`` (None: raw, also when the key is absent; signal
        ``matched``: the key with the windows of every data set)."""
        return signal_key(self.cfg)

    def train_signal_key(self) -> dict[str, Any] | list[Any] | None:
        """``signal_key`` of the training data set ``train.data`` (the key a checkpoint must have been trained on;
        without ``train.data`` the experiment-level key). Signal ``matched``: the harmonised key with the reference
        window (design.md 15.1). A pooled domain (design.md 16.1): the ordered list of the per-set keys
        (``training_signal_key``)."""
        return training_signal_key(self.cfg) if self.train_names else self.signal_key()

    def check_signal(self, checkpoint: Path) -> None:
        """The checkpoint (``model.pt``) must have been trained on the signal key of the training data set
        (``train_signal_key``; ``signal`` of the ``train_log.json`` next to it, absent: raw), not on the
        experiment-level key. ValueError otherwise; a missing log is a warning."""
        log_path = checkpoint.parent / "train_log.json"
        if not log_path.exists():
            self.log.line(f"warning: {log_path} is missing, the signal variant of the checkpoint cannot be checked")
            return
        stored = json.loads(log_path.read_text(encoding="utf-8")).get("signal")
        expected = self.train_signal_key()
        if stored != expected:
            what = "pooled training domain" if self.pooled else "training data set"
            where = "" if self.train_data is None else f" for the {what} {self.train_data}"
            raise ValueError(
                f"{checkpoint} was trained on {describe_signal(stored)} signals, the config asks "
                f"{describe_signal(expected)}{where}; use the checkpoints of the same signal variant"
            )

    def compare_spec(self, path: Path, spec: Mapping[str, Any]) -> str:
        """``check_spec`` without writing: ``new`` (no stored spec), ``equal`` or ``conflict``."""
        if not path.exists():
            return "new"
        return "equal" if json.loads(path.read_text(encoding="utf-8")) == dict(spec) else "conflict"

    def check_spec(self, path: Path, spec: Mapping[str, Any]) -> bool:
        """Write ``spec`` to ``path`` or verify that the stored one is equal (ValueError otherwise)."""
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored != spec:
                raise ValueError(f"{path} was written with {stored}, the config asks for {dict(spec)}; use another root")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(path, dict(spec))
        return True

    def checkpoint(self, target: str, label: str, seed: int) -> Path:
        """``<checkpoints>/<model>/seed<k>/model.pt``; ``checkpoint_overrides`` of the target (keyed by label or model)
        replace the checkpoint directory, e.g. DiffusionCF, which is not retrained for ``mean_1s`` (design.md 11.1)."""
        spec, model = self.targets[target], self.label_spec(label)["model"]
        overrides = spec.get("checkpoint_overrides") or {}
        directory = overrides.get(label, overrides.get(model, spec["checkpoints"]))
        return _path(directory) / model / f"seed{int(seed)}" / "model.pt"

    # ------------------------------------------------------------------ events
    def split_events(self, dataset: str) -> tuple[EventSet, dict[str, Any]]:
        d = self.cfg.datasets[dataset]
        name, part = str(d.data), str(d.get("split", "all"))
        return self.data.split(name, part), {"data": name, "split": part, "n_events": d.get("n_events")}

    def ensure_events(self, target: str, dataset: str) -> pd.DataFrame:
        """The events table of (target, data set): written on first use, else checked against the selection."""
        key = (target, dataset)
        if key in self.events:
            return self.events[key]
        split, spec = self.split_events(dataset)
        lengths = np.array([len(ev) for ev in split], dtype=np.int64)
        n_events = None if spec["n_events"] is None else int(spec["n_events"])
        seed = int(np.random.SeedSequence([int(self.cfg.rollout.event_seed), _crc(dataset)]).generate_state(1)[0])
        drawn = select_events(lengths, n_events, seed, self.n_steps)
        frame = events_table(split, drawn, self.data.folds(spec["data"]))
        path = tables.events_path(self.root, target, dataset)
        if path.exists():
            stored = tables.load_events_table(self.root, target, dataset)
            same = stored["event_id"].astype(str).tolist() == frame["event_id"].tolist()
            same = same and stored["order"].tolist() == frame["order"].tolist() and stored["t0"].tolist() == frame["t0"].tolist()
            if not same:
                raise ValueError(f"{path} holds other events than the config selects; use another root")
            frame = stored
        else:
            tables.save_events_table(self.root, target, dataset, frame)
            n_long = int(np.count_nonzero(lengths > T0 + self.n_steps))
            self.log.line(
                f"[{target}] events {dataset}: {len(frame)} of {len(split):,} events ({spec['data']}, split {spec['split']}, "
                f"{n_long:,} long enough)"
            )
        self.events[key] = frame
        return frame

    def event_objects(self, dataset: str, frame: pd.DataFrame) -> list[Event]:
        split, _ = self.split_events(dataset)
        evs = [split[int(i)] for i in frame["event"]]
        if [ev.event_id for ev in evs] != frame["event_id"].astype(str).tolist():
            raise RuntimeError(f"events of {dataset} differ from the stored events table; use another root")
        return evs

    # ------------------------------------------------------------------ rollouts
    def rollout_config(self, target: str, label: str, model: CFModel, seed: int, dataset: str) -> RolloutConfig:
        r, spec = self.cfg.rollout, self.label_spec(label)
        sampled = bool(getattr(model, "plan_sampler", False))  # plan samplers: n_rollouts_sampled, hold 1
        n_r = spec["n_rollouts"] or (int(r.n_rollouts_sampled) if sampled else int(r.n_rollouts))
        return RolloutConfig(
            n_rollouts=int(n_r),
            horizon_s=self.horizon_s,
            horizons_s=self.horizons_s,
            hold_steps=1 if sampled else int(self.targets[target].get("hold_steps", 1)),
            sampler_mode=spec["sampler_mode"],
            max_batch=int(r.max_batch),
            stream_events=int(r.get("stream_events", 100)),
            seed=unit_seed(int(r.seed), seed, dataset),
        )

    def roll(self, target: str, label: str, seed: int, model: CFModel, dataset: str) -> None:
        """Write ``rollouts/<label>/seed<k>/<dataset>.parquet`` of ``target``."""
        frame = self.ensure_events(target, dataset)
        evs = self.event_objects(dataset, frame)
        rcfg = self.rollout_config(target, label, model, seed, dataset)
        seed_everything(seed)
        started = time.perf_counter()
        result = rollout_events(model, evs, starts=frame["t0"].to_numpy(np.int64), cfg=rcfg, device=self.device)
        table = tables.summarize_rollout(result, frame)
        tables.save_rollout(self.root, target, label, seed, dataset, table)
        seconds = time.perf_counter() - started
        self.written.append(f"{target}/{label}/seed{seed}/{dataset}")
        info = result.info
        self.timing({
            "time": _now(), "target": target, "model": label, "seed": seed, "dataset": dataset, "seconds": seconds,
            "rollout_seconds": info["seconds"], "n_events": len(evs), "n_rollouts": rcfg.n_rollouts,
            "trajectories": info["trajectories"], "n_batches": info["n_batches"], "hold_steps": rcfg.hold_steps,
            "sampler_mode": rcfg.sampler_mode, "peak_memory_mb": info.get("peak_memory_mb"), "device": info["device"],
        })  # fmt: skip
        last = table[table["horizon_s"] == table["horizon_s"].max()]
        inside = ((last["s_q05"] <= last["truth_s"]) & (last["truth_s"] <= last["s_q95"])).mean()
        mem = info.get("peak_memory_mb")
        self.log.line(
            f"[{target} {label} seed{seed}] {dataset}: {len(evs)} events x {rcfg.n_rollouts} rollouts in {seconds:.1f} s"
            f"{'' if mem is None else f', peak {mem:.0f} MB'} (raw PICP at {self.horizons_s[-1]:g} s {inside:.3f}, "
            f"collided {last['collided'].mean():.3f})"
        )

    def run_unit(self, target: str, label: str, seed: int) -> None:
        """Missing tables of one (target, label, seed); the checkpoint is loaded once."""
        unit = f"{target}/{label}/seed{seed}"
        missing = [d for d in self.datasets if not tables.rollout_path(self.root, target, label, seed, d).exists()]
        if not missing:
            return
        spec = self.label_spec(label)
        self.check_spec(self.root / target / "rollouts" / label / "spec.json", _jsonable({**spec, "label": label}))
        path = self.checkpoint(target, label, seed)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint {path} does not exist")
        self.check_signal(path)
        self.current = unit
        model = load_model(path, device=self.device)
        for dataset in missing:
            self.current = f"{unit}/{dataset}"
            try:
                self.roll(target, label, seed, model, dataset)
            except Exception as exc:  # noqa: BLE001 - the other data sets of this unit still run
                self.log.error(f"{unit}/{dataset}", exc)
            self.write_status()
        del model

    # ------------------------------------------------------------------ run
    def run(self) -> dict[str, Any]:
        OmegaConf.save(self.cfg, self.root / "config.yaml", resolve=True)
        for target, spec in self.targets.items():
            ckpt = _path(spec["checkpoints"])
            if ckpt.is_dir():
                self.active.append(target)
            else:
                self.skipped[target] = f"checkpoint directory {ckpt} does not exist"
                self.log.line(f"warning: target {target} skipped: checkpoint directory {ckpt} does not exist")
        status = self.status()
        self.log.line(
            f"E2 run: root {self.root}, signal {self.cfg.get('signal') or 'raw'}, device {self.device}; targets "
            f"{self.active} x {len(self.labels)} models x "
            f"{len(self.seeds)} seeds x {len(self.datasets)} data sets; tables {status['done']}/{status['total']} done"
        )
        if self.windows:
            self.log.line(f"matched windows ({_display(matching_path(self.cfg))}): "
                          + ", ".join(f"{k} {v}" for k, v in self.windows.items()))  # fmt: skip
        self.write_status()
        finished = False
        try:
            for target in list(self.active):
                spec_path = self.root / target / "spec.json"
                if self.guard(f"{target}/spec", self.check_spec, spec_path, self.target_spec(target)) is None:
                    continue  # settings differ from the stored ones: nothing of this target runs
                for dataset in self.datasets:
                    self.guard(f"{target}/events/{dataset}", self.ensure_events, target, dataset)
                for label in self.labels:
                    for seed in self.seeds:
                        self.guard(f"{target}/{label}/seed{seed}", self.run_unit, target, label, seed)
                        self.cleanup()
                        self.write_status()
            finished = True
        finally:
            status = self.write_status(finished=finished)
        self.log.line(
            f"{'finished' if finished else 'stopped'} after {status['elapsed_s']:.0f} s: tables {status['done']}/"
            f"{status['total']} ({len(self.written)} new), {len(self.log.failures)} failure(s), skipped targets "
            f"{sorted(self.skipped) or 'none'}"
        )
        for failure in self.log.failures:
            self.log.line(f"  failed: {failure['unit']}: {failure['error']}")
        return status


    # ------------------------------------------------------------------ dry run
    def dry_run(self) -> dict[str, Any]:
        """What ``run`` would do (module docstring), loading no events and writing nothing.

        Returns ``root``, ``signal``, ``targets`` (per configured target: ``checkpoints``, ``active``, ``spec`` = new /
        equal / conflict against ``<target>/spec.json``, ``events`` = data set -> stored / new, and per label its base
        ``model``, ``sampler_mode``, ``spec`` against the stored label spec, ``missing_checkpoints`` (seeds), ``done``
        and ``missing`` tables), ``conflicts`` (stored settings that differ from the config, checkpoints trained on
        another signal key: a run would refuse these units), ``done``, ``missing`` and ``nothing_to_do``.
        """
        conflicts: list[dict[str, str]] = []
        targets: dict[str, Any] = {}
        for target, tspec in self.targets.items():
            ckpt = _path(tspec["checkpoints"])
            entry: dict[str, Any] = {"checkpoints": str(tspec["checkpoints"]), "active": ckpt.is_dir(),
                                     "hold_steps": int(tspec.get("hold_steps", 1)), "labels": {}}  # fmt: skip
            if not entry["active"]:  # a run skips the target; its tables are listed as missing all the same
                entry["skipped"] = f"checkpoint directory {ckpt} does not exist (a run skips this target)"
            entry["spec"] = self.compare_spec(self.root / target / "spec.json", self.target_spec(target))
            if entry["spec"] == "conflict":
                conflicts.append({"unit": f"{target}/spec", "error": "stored spec.json differs from the config"})
            entry["events"] = {d: "stored" if tables.events_path(self.root, target, d).exists() else "new"
                               for d in self.datasets}  # fmt: skip
            for label in self.labels:
                spec = self.label_spec(label)
                row: dict[str, Any] = {"model": spec["model"], "sampler_mode": spec["sampler_mode"],
                                       "spec": self.compare_spec(self.root / target / "rollouts" / label / "spec.json",
                                                                 _jsonable({**spec, "label": label})),
                                       "missing_checkpoints": [], "done": 0, "missing": 0}  # fmt: skip
                if row["spec"] == "conflict":
                    conflicts.append({"unit": f"{target}/{label}", "error": "stored label spec.json differs from the config"})
                for seed in self.seeds:
                    path = self.checkpoint(target, label, seed)
                    if not path.exists():
                        row["missing_checkpoints"].append(seed)
                    else:
                        try:
                            self.check_signal(path)
                        except ValueError as exc:
                            conflicts.append({"unit": f"{target}/{label}/seed{seed}", "error": str(exc)})
                    for d in self.datasets:
                        done = tables.rollout_path(self.root, target, label, seed, d).exists()
                        row["done" if done else "missing"] += 1
                entry["labels"][label] = row
            entry["done"] = sum(r["done"] for r in entry["labels"].values())
            entry["missing"] = sum(r["missing"] for r in entry["labels"].values())
            targets[target] = entry
        done = sum(e.get("done", 0) for e in targets.values())
        missing = sum(e.get("missing", 0) for e in targets.values())
        out = {"root": str(self.root), "signal": str(self.cfg.get("signal") or "raw"), "matched_windows": self.windows,
               "datasets": list(self.datasets), "seeds": list(self.seeds), "targets": targets, "conflicts": conflicts,
               "done": done, "missing": missing, "nothing_to_do": missing == 0 and not conflicts}  # fmt: skip
        self.log.line(f"E2 dry run: root {self.root}, signal {out['signal']}; tables {done}/{done + missing} done, "
                      f"{len(conflicts)} conflict(s){'; nothing to do' if out['nothing_to_do'] else ''}")  # fmt: skip
        for target, e in targets.items():
            skipped = f" - SKIPPED: {e['skipped']}" if not e["active"] else ""
            self.log.line(f"  target {target} ({e['checkpoints']}, hold {e['hold_steps']}): spec {e['spec']}, "
                          f"tables {e['done']}/{e['done'] + e['missing']}{skipped}")  # fmt: skip
            for label, r in e["labels"].items():
                gaps = f", checkpoints missing for seeds {r['missing_checkpoints']}" if r["missing_checkpoints"] else ""
                self.log.line(f"    {label} ({r['model']}, {r['sampler_mode']}): {r['done']} done, {r['missing']} "
                              f"missing, spec {r['spec']}{gaps}")  # fmt: skip
        for c in conflicts:
            self.log.line(f"  conflict: {c['unit']}: {c['error']}")
        return out


def run_e2(cfg: DictConfig | Mapping[str, Any], echo: bool = True) -> dict[str, Any]:
    """Run or resume experiment E2 (module docstring); returns the final status (also in ``status.json``)."""
    cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(dict(cfg))
    configure_torch(bool(cfg.get("tf32", False)))
    return E2Runner(cfg, echo=echo).run()


def dry_run_e2(cfg: DictConfig | Mapping[str, Any], echo: bool = True) -> dict[str, Any]:
    """``E2Runner.dry_run`` of the config: what a run would do and which stored settings conflict; writes nothing."""
    cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(dict(cfg))
    return E2Runner(cfg, echo=echo, dry=True).dry_run()
