"""Experiment E1 (docs/design.md, section 10): every model trained on highD (seeds 0-4), evaluated on the highD
test and validation folds, NGSIM I-80, Waymo and OpenACC.

``run_e1(cfg)`` (config ``configs/experiment/e1.yaml``) writes under ``cfg.root`` (section 10.2)::

    windows/<dataset>.parquet, <dataset>_meta.json   evaluated windows in a fixed row order (cf_unc.maps.io)
    train_density.npz                                highD training windows binned on the map grid
    idm.json                                         named population IDM calibrations of the highD training events
    checkpoints/<model>/seed<k>/model.pt, train_log.json, metrics.json
    preds/<model>/seed<k>/<dataset>.parquet          one row per row of windows/<dataset>.parquet
    logs/run.log, logs/errors.log, status.json

IDM calibrations are named (config ``idm: {closed_loop: {...}, mle: {...}}``); a model config with
``needs_idm: true`` selects one with ``idm: <name>`` (default ``closed_loop``). Only the calibrations that a
selected model needs are computed. Resumable at the level of (model, seed) checkpoints and (model, seed,
dataset) tables: existing files are loaded or skipped, never rewritten. A failing unit is logged with its
traceback to ``logs/errors.log`` and the run continues; ``run_e1`` returns the final status, whose ``failures``
list every failed unit.

Target (design.md, section 11.1): ``train.target`` (``"a"`` by default, or ``"mean_1s"``) defines the training
windows of the one-step models and the evaluated windows (column ``y`` of the windows tables); it is part of the
settings checked on resume (windows spec, training density, checkpoint logs; files written before the option
existed count as target ``"a"``). Sequence models (DiffusionCF) are trained on 5 s acceleration sequences for
every target; models whose ``predict_dist`` takes a ``target`` keyword get the experiment's target.
``reuse_checkpoints: {<model>: <root>/checkpoints}`` copies the checkpoint of (model, seed) from another
experiment root instead of training it (checked like an own checkpoint, noted in ``train_log.json``).
An experiment config may inherit another one with ``base: <file>`` (``load_config``).

Signal variant (design.md, section 12): the experiment-level key ``signal: raw | harmonised`` (it overrides every
data set; otherwise the data config's ``signal``, default ``raw``) and the settings ``harmonise`` (thresholds of
``cf_unc.data.harmonise.HarmoniseConfig``). Harmonised events are produced at load time; the variant is part of
every setting checked on resume (windows specs, training density, IDM calibration keys, checkpoint logs, also of
reused checkpoints), stored only when it is not ``raw`` so that everything written before the option stays valid
and no raw artefact can be reused by a harmonised run (or the other way round). ``Runner.dry_run`` (``scripts/
run_e1.py --dry-run``) compares every stored setting with the config without loading data or writing any file, and
states for every checkpoint to copy from ``reuse_checkpoints`` whether the source passes the checks.

Signal variant ``matched`` (design.md 15.1, experiment E5; experiment level only): the harmonised processing with one
Savitzky-Golay window per data set, read from the matching file ``match.file`` of ``scripts/match_roughness.py``
(``match: {reference, grid, tolerance, file}``). The signal key of data set ``name`` is ``{"signal": "harmonised",
"harmonise": {... "window": w_name}}``: the reference data set (``match.reference``, the training data) has the key
of the harmonised variant, so the checkpoints of a harmonised root are reused legitimately (checked against the key
of the training data set), while the tables of the other data sets are new. The runners refuse to run
(``MatchingError``, nothing is created) when the matching file is missing, was computed with other settings or lacks
a data set of the experiment.

Pooled training domain (design.md 16.1, experiment E6): ``train.data`` may be a list of data config names. The
training windows are the union of the training folds of the listed data sets, each capped at
``train.max_windows_per_dataset`` one-step windows (``train.max_sequences_per_dataset`` sequence windows for
DiffusionCF; uniform subsample after the stride, seed ``train.subsample_seed`` mixed with the data set name, so a LODO
pool keeps the subsamples of the full pool; ``cf_unc.data.pipeline.build_pooled_bundle``); the validation windows are
the union of the validation folds, capped the same way; the IDM calibrations and the training density use the pooled
training events / windows. The signal key of a pooled training is the ordered list of the per-set keys
(``training_signal_key``) and ``pool_key`` (data sets, strides, caps, seed) is checked with it (checkpoint logs,
reused checkpoints, training density; the IDM keys hold the lists), so nothing of a root trained on one data set is
accepted by a pooled config or the other way round. An evaluated data set ``{pool: [<data set>, ...]}`` is the union
of the listed tables (the test folds of every training set): its windows table is their concatenation (event and
driver codes re-coded over the union, event ids, states and folds as in the member tables; ``spec`` = the member
specs) and its prediction tables are the concatenation of the members' tables (no prediction of its own). With a
pooled domain an evaluated data set of a training set must use its test or validation fold (``split: all`` or
``train`` would evaluate training events: refused before anything is created). The dry run of a pooled config loads
the events of the training sets (nothing is written) to state the window counts after the caps.

Frozen IDM prior (``train.idm_from: <idm.json of another root>``, decision for E6): the runner fits no IDM calibration
of its own; every named calibration a model needs is copied from that file (parameters, report, the source's key) into
``<root>/idm.json`` with its provenance (``reused_from``: source path, sha256 of the file, note) and ``train_eval`` =
``evaluate_idm`` of the copied parameters on this root's training events. The key of a copied calibration is
``{"idm_from": <path>, "name", "source_key", "params"}``: it is stored in ``idm.json`` and in the ``idm`` entry of every
checkpoint log, so a root, a checkpoint or a reuse source made with another ``idm_from`` (another file, other
parameters) or with a fitted calibration is refused. The regimes of the analyses take ``v0`` from ``<root>/idm.json``,
hence from the copied calibration. Models still fit their own noise level (e.g. the white-noise sigma of the IDM
baselines) on the training windows.

An entry of ``eval.datasets``, ``datasets``, ``targets``, ``e1_roots``, ``variants`` or ``reuse_checkpoints`` that is
set to null is removed (``load_config``): a config inheriting another one with ``base:`` drops an entry this way, and
``reuse_checkpoints: null`` drops every inherited reuse.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import random
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from cf_unc.constants import HIST_LEN, SEQ_LEN
from cf_unc.data.harmonise import (
    DEFAULT_SIGNAL,
    HarmoniseConfig,
    MatchConfig,
    MatchingError,
    check_signal,
    matched_windows,
    read_matching,
)
from cf_unc.data.pipeline import POOL_SPLITS, DataBundle, build_bundle, build_pooled_bundle, load_events
from cf_unc.data.schema import EventSet
from cf_unc.data.splits import split_by_folds
from cf_unc.data.windows import DEFAULT_TARGET, WindowSet, check_target, concat_windows, count_windows, make_windows
from cf_unc.eval.metrics import summarize
from cf_unc.maps import io as tables
from cf_unc.models import CFModel, build_model, load_model, resolve_device
from cf_unc.physics.idm import IDMParams, calibrate_idm, evaluate_idm
from cf_unc.train.run import ROOT, _path, _plain, model_kwargs, write_json

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_IDM",
    "PILOT",
    "EvalSet",
    "ModelSpec",
    "Runner",
    "accepts_keyword",
    "bin_counts",
    "check_matching",
    "checkpoint_dir",
    "config_target",
    "configure_torch",
    "data_config",
    "data_names",
    "data_signal",
    "dry_run_e1",
    "grid_edges",
    "idm_from_path",
    "is_pooled",
    "leader_future",
    "load_config",
    "match_config",
    "matching_path",
    "matching_windows",
    "model_config",
    "model_spec",
    "pilot_path",
    "pool_key",
    "pool_members",
    "predict_table",
    "quick_metrics",
    "run_e1",
    "sampled_rows",
    "seed_everything",
    "signal_key",
    "train_datasets",
    "training_signal_key",
]

DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "e1.yaml"
DEFAULT_IDM = "closed_loop"  # calibration of a model config with needs_idm and no ``idm`` key
Calibration = Callable[[str], tuple[IDMParams, dict[str, Any]]]  # name -> (parameters, key of the settings)
# settings of a windows spec that were added later, with the value that files written before them imply
SPEC_DEFAULTS: dict[str, Any] = {"target": DEFAULT_TARGET}
# reduced experiment for timing and end-to-end checks (scripts/run_e1.py --pilot), merged over the config file;
# load_config replaces the root by "<root of the config>_pilot" (the value below is that of e1.yaml)
PILOT: dict[str, Any] = {
    "root": "artifacts/e1_pilot",
    "seeds": [0],
    "train": {"overrides": {"max_epochs": 3}},  # every model with a TrainConfig (ensemble members, DiffusionCF too)
    "model_overrides": {"gp_residual_idm": {"max_epochs": 3}},  # the GP has its own optimiser settings
    "eval": {"max_events": 400, "max_rows_sampled": 2000},
}
COVERED = "covered"
# mappings of an experiment config whose entries are removed when set to null (a config inheriting another one with
# `base:` drops an inherited data set, target, E1 root, variant or reuse source this way)
REMOVABLE = ("eval.datasets", "datasets", "targets", "e1_roots", "variants", "reuse_checkpoints")


# ------------------------------------------------------------------------------ helpers
def _drop_removed(cfg: DictConfig) -> DictConfig:
    """Remove the entries set to null of the mappings ``REMOVABLE`` (in place; returns ``cfg``)."""
    for dotted in REMOVABLE:
        node = OmegaConf.select(cfg, dotted, default=None)
        if isinstance(node, DictConfig):
            for key in [k for k in node.keys() if node.get(k) is None]:
                node.pop(key)
    return cfg


def _load_file(path: Path, chain: tuple[Path, ...] = ()) -> DictConfig:
    """One experiment config file merged over its ``base`` file (resolved recursively, cycles rejected); entries of
    ``REMOVABLE`` set to null are removed after the merge."""
    path = path.resolve()
    if path in chain:
        raise ValueError(f"experiment configs inherit from each other in a cycle: {[str(p) for p in (*chain, path)]}")
    cfg = OmegaConf.load(path)
    base = cfg.pop("base", None) if isinstance(cfg, DictConfig) else None
    if base is None:
        return _drop_removed(cfg) if isinstance(cfg, DictConfig) else cfg  # type: ignore[return-value]
    base_path = Path(str(base))
    parent = _load_file(base_path if base_path.is_absolute() else path.parent / base_path, (*chain, path))
    return _drop_removed(OmegaConf.merge(parent, cfg))  # type: ignore[arg-type]


def pilot_path(path: str) -> str:
    """Pilot counterpart of an experiment root (``artifacts/e1`` -> ``artifacts/e1_pilot``) or of its checkpoints
    directory (``artifacts/e1/checkpoints`` -> ``artifacts/e1_pilot/checkpoints``)."""
    p = PurePosixPath(str(path).replace("\\", "/"))
    if p.name == "checkpoints":
        return str(p.parent.with_name(p.parent.name + "_pilot") / p.name)
    return str(p.with_name(p.name + "_pilot"))


def load_config(path: str | Path = DEFAULT_CONFIG, overrides: Sequence[str] = (), pilot: bool = False) -> DictConfig:
    """Experiment config: the YAML file, then ``PILOT`` (if ``pilot``), then dotted ``key=value`` overrides.

    ``base: <file>`` (relative to the file's directory) inherits another experiment config: the file is merged
    over it (mappings deep-merged, lists and values replaced). With ``pilot`` the root becomes ``<root>_pilot``
    and every ``reuse_checkpoints`` directory ``<root>/checkpoints`` the pilot's ``<root>_pilot/checkpoints``
    (checkpoints trained with the pilot settings, which the kwargs check requires).
    """
    cfg = _load_file(_path(path))
    if pilot:
        root = cfg.get("root")
        cfg = OmegaConf.merge(cfg, PILOT)  # type: ignore[assignment]
        if root is not None:
            cfg.root = pilot_path(str(root))
        reuse = cfg.get("reuse_checkpoints") or {}
        if reuse:
            cfg.reuse_checkpoints = {str(k): pilot_path(str(v)) for k, v in reuse.items()}
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([str(o) for o in overrides]))  # type: ignore[assignment]
    return cfg


def config_target(cfg: DictConfig | Mapping[str, Any]) -> str:
    """Target of an experiment config: ``train.target`` (default ``"a"``), checked against ``TARGETS``."""
    train = cfg.get("train") or {}
    return check_target(str(train.get("target") or DEFAULT_TARGET))


def is_pooled(cfg: DictConfig | Mapping[str, Any]) -> bool:
    """Whether ``train.data`` is a list of data sets (a pooled training domain, design.md 16.1; also a list of one)."""
    return isinstance((cfg.get("train") or {}).get("data"), (ListConfig, list, tuple))


def train_datasets(cfg: DictConfig | Mapping[str, Any]) -> list[str]:
    """Training data sets (data config names) of an experiment config: ``train.data`` as a list (one name: a list of
    one; absent: empty). ValueError for an empty pool or a data set listed twice."""
    data = (cfg.get("train") or {}).get("data")
    if data is None:
        return []
    names = [str(x) for x in data] if is_pooled(cfg) else [str(data)]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"train.data must list distinct data sets, got {names}")
    return names


def _cap(value: Any) -> int | None:
    return None if value is None else int(value)


def pool_key(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any] | None:
    """What defines the windows of a pooled training domain besides the signals (design.md 16.1): the data sets in
    their order, the strides, the caps per data set (``train.max_windows_per_dataset`` for one-step windows,
    ``train.max_sequences_per_dataset`` for sequence windows; null: no cap) and ``train.subsample_seed`` (default 0).
    None for one training data set. Checked on resume with the signal key (checkpoint logs, training density)."""
    if not is_pooled(cfg):
        return None
    train = cfg.get("train") or {}
    return _jsonable({
        "data": train_datasets(cfg), "stride": int(train.get("stride", 1)), "seq_stride": int(train.get("seq_stride", 1)),
        "max_windows_per_dataset": _cap(train.get("max_windows_per_dataset")),
        "max_sequences_per_dataset": _cap(train.get("max_sequences_per_dataset")),
        "subsample_seed": int(train.get("subsample_seed", 0) or 0),
    })  # fmt: skip


def idm_from_path(cfg: DictConfig | Mapping[str, Any]) -> Path | None:
    """The IDM file of ``train.idm_from`` (an ``idm.json`` of another root; relative paths: repository root) whose
    calibrations the runner copies instead of fitting its own, or None when the key is not set."""
    value = (cfg.get("train") or {}).get("idm_from")
    return None if value is None or str(value) == "" else _path(str(value))


def calibration_entries(content: Mapping[str, Any]) -> dict[str, Any]:
    """The named calibrations of the content of an ``idm.json`` (``calibrations``; an old file of a single
    calibration: its top-level entry is ``closed_loop``)."""
    entries = content.get("calibrations")
    if entries is None and "params" in content:
        entries = {DEFAULT_IDM: {k: content.get(k) for k in ("params", "report", "key")}}
    return dict(entries or {})


def pool_members(cfg: DictConfig | Mapping[str, Any], dataset: str) -> list[str] | None:
    """Member data sets (keys of ``eval.datasets``) of the evaluated data set ``dataset`` when it is a pool
    (``{pool: [...]}``, the union of their tables, design.md 16.1), else None."""
    entry = ((cfg.get("eval") or {}).get("datasets") or {}).get(dataset)
    members = None if entry is None else entry.get("pool")
    return None if members is None else [str(m) for m in members]


def data_config(cfg: DictConfig | Mapping[str, Any], name: str) -> DictConfig:
    """Data config ``name``: inline (``data_configs``) or ``config_paths.data/<name>.yaml``."""
    inline = cfg.get("data_configs") or {}
    if name in inline:
        return inline[name]
    return OmegaConf.load(_path(cfg["config_paths"]["data"]) / f"{name}.yaml")  # type: ignore[return-value]


def data_signal(cfg: DictConfig | Mapping[str, Any], name: str) -> str:
    """Signal variant of data set ``name``: the experiment-level ``signal`` (overrides every data set), else the
    data config's ``signal``, default ``"raw"`` (design.md, section 12). ``matched`` (design.md 15.1) is an
    experiment-level variant: a data config that asks for it is an error."""
    value = cfg.get("signal")
    if value is None:
        value = data_config(cfg, name).get("signal")
        if value is not None and str(value) == "matched":
            raise ValueError(f"data config {name!r}: signal 'matched' is an experiment-level variant (it needs the "
                             "experiment's match settings); set `signal: matched` in the experiment config")  # fmt: skip
    return check_signal(str(DEFAULT_SIGNAL if value is None else value))


def match_config(cfg: DictConfig | Mapping[str, Any]) -> MatchConfig:
    """``MatchConfig`` of the config's ``match`` section (design.md 15.1; defaults when absent)."""
    return MatchConfig.from_config(cfg.get("match"))


def matching_path(cfg: DictConfig | Mapping[str, Any]) -> Path:
    """The matching file ``match.file`` (relative paths: repository root); MatchingError when it is not configured."""
    file = match_config(cfg).file
    if not file:
        raise MatchingError("signal matched needs match.file, the matching file of scripts/match_roughness.py")
    return _path(file)


def matching_windows(cfg: DictConfig | Mapping[str, Any]) -> dict[str, int]:
    """Matched window per data set of the config's matching file, checked against its ``harmonise`` and ``match``
    settings (``cf_unc.data.harmonise.matched_windows``; MatchingError when missing or computed otherwise)."""
    return matched_windows(matching_path(cfg), cfg.get("harmonise"), cfg.get("match"))


def signal_key(cfg: DictConfig | Mapping[str, Any], name: str | None = None) -> dict[str, Any] | None:
    """What defines the signal variant of the events of data set ``name`` (None: of the experiment-level ``signal``
    alone): None for raw signals, else ``{"signal": "harmonised", "harmonise": <settings>}`` (``HarmoniseConfig``
    from the config's ``harmonise``). Signal ``matched`` (design.md 15.1): for a data set the harmonised key with its
    matched window (``matching_windows``; the reference keeps ``harmonise.window``, so its key is the harmonised
    key); at the experiment level ``{"signal": "matched", "harmonise": <settings>, "match": {<MatchConfig
    settings>, "windows": {data set: window}}}``. MatchingError when the matching file is missing, stale or lacks
    the data set.

    Stored with every setting checked on resume (windows specs, training density, IDM keys, checkpoint logs) only
    when it is not None, so that the files written before the option count as raw.
    """
    signal = data_signal(cfg, name) if name is not None else check_signal(str(cfg.get("signal") or DEFAULT_SIGNAL))
    if signal == DEFAULT_SIGNAL:
        return None
    settings = HarmoniseConfig.from_config(cfg.get("harmonise")).to_dict()
    if signal == "harmonised":
        return _jsonable({"signal": signal, "harmonise": settings})
    windows = matching_windows(cfg)
    if name is None:
        return _jsonable({"signal": signal, "harmonise": settings, "match": {**match_config(cfg).settings(), "windows": windows}})
    if name not in windows:
        raise MatchingError(f"data set {name!r} has no matched window in {matching_path(cfg)} (it holds "
                            f"{sorted(windows)}); compute the file with a config that uses the data set")  # fmt: skip
    return _jsonable({"signal": "harmonised", "harmonise": {**settings, "window": int(windows[name])}})


def training_signal_key(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any] | list[Any] | None:
    """Signal key of the training data, the key every checkpoint is checked against: ``signal_key`` of ``train.data``
    for one data set; for a pooled training domain (design.md 16.1) the ordered list of the per-set keys (a list also
    for raw signals, ``[None, ...]``, so it never equals the key of one data set); None without ``train.data``."""
    names = train_datasets(cfg)
    if not names:
        return None
    if is_pooled(cfg):
        return _jsonable([signal_key(cfg, n) for n in names])
    return signal_key(cfg, names[0])


def describe_signal(key: Mapping[str, Any] | Sequence[Any] | None) -> str:
    """Readable form of a ``signal_key`` (None: ``"raw"``) or of the list of a pooled training domain."""
    if key is None:
        return DEFAULT_SIGNAL
    if isinstance(key, (list, tuple)):
        return "pooled [" + "; ".join(describe_signal(k) for k in key) + "]"
    settings = ", ".join(f"{k} {v:g}" if isinstance(v, (int, float)) else f"{k} {v}" for k, v in (key.get("harmonise") or {}).items())
    text = f"{key.get('signal')} ({settings})" if settings else str(key.get("signal"))
    windows = (key.get("match") or {}).get("windows")
    return text + (" with the windows " + ", ".join(f"{k} {v}" for k, v in windows.items()) if windows else "")


def describe_pool(key: Mapping[str, Any] | None) -> str:
    """Readable form of a ``pool_key`` (None: ``"one data set"``)."""
    if key is None:
        return "one data set"

    def cap(v: Any) -> str:
        return "all" if v is None else f"{int(v):,}"

    return (f"pool of {', '.join(key['data'])} (training folds; at most {cap(key.get('max_windows_per_dataset'))} "
            f"one-step / {cap(key.get('max_sequences_per_dataset'))} sequence windows per data set after the strides "
            f"{key.get('stride')} / {key.get('seq_stride')}, subsample seed {key.get('subsample_seed')})")  # fmt: skip


def data_names(cfg: DictConfig | Mapping[str, Any]) -> list[str]:
    """Data sets (data config names) of an experiment config: the training data (``train.data``, every data set of a
    pooled domain), then the data of the evaluated data sets (``eval.datasets``, E1; a pool has no data of its own)
    and of the rolled data sets (``datasets``, E2), each once."""
    names = train_datasets(cfg)
    for section in ((cfg.get("eval") or {}).get("datasets") or {}, cfg.get("datasets") or {}):
        names += [str(d["data"]) for d in section.values() if d.get("pool") is None]
    return list(dict.fromkeys(names))


def check_pool_config(cfg: DictConfig | Mapping[str, Any]) -> None:
    """Consistency of the evaluated data sets with the training domain (design.md 16.1); ValueError otherwise.

    With a pooled training domain no evaluated data set of a training set may hold training events (its split must be
    ``test`` or ``val``). A pool entry (``{pool: [...]}``) must list evaluated data sets (no pools) that are the test
    folds of the training data sets, every training data set once.
    """
    names = train_datasets(cfg)
    datasets = (cfg.get("eval") or {}).get("datasets") or {}
    for key, entry in datasets.items():
        members = pool_members(cfg, str(key))
        if members is None:
            part = str(entry.get("split", "all"))
            if is_pooled(cfg) and str(entry["data"]) in names and part not in ("test", "val"):
                raise ValueError(f"eval.datasets.{key}: {entry['data']} is a training data set of the pooled domain, its "
                                 f"split {part!r} holds training events; evaluate its test (or validation) fold")  # fmt: skip
            continue
        unknown = [m for m in members if m not in datasets]
        if unknown:
            raise ValueError(f"eval.datasets.{key}: pool members {unknown} are not evaluated data sets")
        data = []
        for m in members:
            entry_m = datasets[m]
            if pool_members(cfg, m) is not None or str(entry_m.get("split", "all")) != "test":
                raise ValueError(f"eval.datasets.{key}: pool member {m!r} must be the test fold of a training data set")
            data.append(str(entry_m["data"]))
        if sorted(data) != sorted(names) or len(set(data)) != len(data):
            raise ValueError(f"eval.datasets.{key}: the pool must hold the test fold of every training data set "
                             f"{names} once, it lists the data {data}")  # fmt: skip


def check_matching(cfg: DictConfig | Mapping[str, Any], names: Sequence[str] | None = None) -> dict[str, int] | None:
    """Matched windows of the data sets ``names`` (default: ``data_names``) when the experiment's signal is
    ``matched`` (None otherwise). MatchingError when the matching file is missing, was computed with other settings,
    lacks one of the data sets, or lists a data set from another event directory than its data config; the runners
    call it before anything else and refuse to run."""
    if str(cfg.get("signal") or DEFAULT_SIGNAL) != "matched":
        return None
    names = data_names(cfg) if names is None else list(dict.fromkeys(str(n) for n in names))
    windows = {n: int(signal_key(cfg, n)["harmonise"]["window"]) for n in names}  # type: ignore[index]
    stored = read_matching(matching_path(cfg)).get("datasets") or {}
    for n in names:
        source = (stored.get(n) or {}).get("source")
        here = data_config(cfg, n).get("path")
        if source is not None and here is not None and source != _display(_path(str(here))):
            raise MatchingError(f"{matching_path(cfg)}: data set {n!r} was matched on {source}, its data config reads "
                                f"{_display(_path(str(here)))}; compute the file again")  # fmt: skip
    return windows


def accepts_keyword(fn: Callable[..., Any], name: str) -> bool:
    """Whether ``fn`` has an explicit parameter ``name`` (``**kwargs`` does not count)."""
    return name in inspect.signature(fn).parameters


def _same_spec(stored: Any, spec: Mapping[str, Any]) -> bool:
    """A stored windows spec equals the configured one (settings added later take ``SPEC_DEFAULTS`` when absent)."""
    if not isinstance(stored, Mapping):
        return False
    return {**SPEC_DEFAULTS, **stored} == {**SPEC_DEFAULTS, **spec}


def _display(path: Path) -> str:
    """``path`` relative to the repository root when inside it (POSIX separators), else as given."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _seed_list(seeds: Sequence[int]) -> str:
    """``0-4`` for a contiguous run of three or more seeds, else ``0, 2, 5``."""
    s = sorted(int(x) for x in seeds)
    if len(s) > 2 and s == list(range(s[0], s[-1] + 1)):
        return f"{s[0]}-{s[-1]}"
    return ", ".join(str(x) for x in s)


def _jsonable(value: Any) -> Any:
    """Plain JSON form (the form in which settings are stored and compared on resume)."""
    return json.loads(json.dumps(_plain(value), sort_keys=True))


def _deep_merge(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write_json(tmp, payload)
    tmp.replace(path)


def seed_everything(seed: int) -> None:
    """Reset the global RNGs (python, numpy, torch on every device): no unit depends on the one before it."""
    random.seed(int(seed))
    np.random.seed(int(seed) % 2**32)
    torch.manual_seed(int(seed))


def configure_torch(tf32: bool = False) -> None:
    """Float32 precision and deterministic kernels for the CUDA work of the run (no effect on CPU results).

    ``tf32=False``: IEEE float32 in cuDNN convolutions / RNNs and cuBLAS matmuls (torch enables TF32 for cuDNN
    by default), so GPU predictions agree with CPU predictions of the same checkpoint. The legacy flags are set
    first and the per-operator flags of torch >= 2.9 after them, which keeps both states consistent (reading
    ``torch.backends.cudnn.allow_tf32`` raises otherwise). The cuBLAS workspace setting only takes effect if
    CUDA has not been initialised yet in this process.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cudnn = torch.backends.cudnn
    cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    if hasattr(cudnn, "conv") and hasattr(cudnn.conv, "fp32_precision"):
        precision = "tf32" if tf32 else "ieee"
        cudnn.conv.fp32_precision = precision
        cudnn.rnn.fp32_precision = precision
        torch.backends.cuda.matmul.fp32_precision = precision
    cudnn.deterministic = True
    cudnn.benchmark = False


def checkpoint_dir(root: str | Path, label: str, seed: int) -> Path:
    return Path(root) / "checkpoints" / label / f"seed{int(seed)}"


# --------------------------------------------------------------------------- map grid
def grid_edges(grid: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Bin edges of ``s`` and ``dv`` from a map grid config ``{s: {lo, hi, step}, dv: {...}}`` (float64)."""

    def edges(axis: Mapping[str, Any]) -> np.ndarray:
        lo, hi, step = float(axis["lo"]), float(axis["hi"]), float(axis["step"])
        return lo + step * np.arange(int(round((hi - lo) / step)) + 1)

    return edges(grid["s"]), edges(grid["dv"])


def bin_counts(s: np.ndarray, dv: np.ndarray, s_edges: np.ndarray, dv_edges: np.ndarray) -> tuple[np.ndarray, int]:
    """Counts (n_s, n_dv) int64 of states on right-open bins ``[e_i, e_{i+1})`` and the number outside.

    A state on the upper edge, outside the grid or NaN counts as outside.
    """
    i = np.searchsorted(s_edges, np.asarray(s, dtype=np.float64), side="right") - 1
    j = np.searchsorted(dv_edges, np.asarray(dv, dtype=np.float64), side="right") - 1
    n_s, n_dv = len(s_edges) - 1, len(dv_edges) - 1
    inside = (i >= 0) & (i < n_s) & (j >= 0) & (j < n_dv)
    counts = np.bincount(i[inside] * n_dv + j[inside], minlength=n_s * n_dv).reshape(n_s, n_dv)
    return counts.astype(np.int64), int(np.count_nonzero(~inside))


# ------------------------------------------------------------------------ data layer
@dataclass
class EvalSet:
    """One evaluated data set: its events (split, capped) and their windows in the row order of the table."""

    name: str
    spec: dict[str, Any]
    events: EventSet
    windows: WindowSet

    def __post_init__(self) -> None:
        self.lengths = np.array([len(ev) for ev in self.events], dtype=np.int64)
        self._v_lead: np.ndarray | None = None

    def leader_future(self, rows: np.ndarray, horizon: int = SEQ_LEN) -> np.ndarray:
        """(len(rows), horizon) float32 leader speeds ``v_lead[t+1 : t+horizon+1]`` [m/s] of the windows ``rows``."""
        if self._v_lead is None:
            parts = [ev.v_lead for ev in self.events]
            self._v_lead = np.concatenate(parts).astype(np.float32) if parts else np.empty(0, np.float32)
        return leader_future(self._v_lead, self.lengths, self.windows.event_index[rows], self.windows.t_index[rows], horizon)


def leader_future(
    v_lead: np.ndarray, lengths: np.ndarray, event_index: np.ndarray, t_index: np.ndarray, horizon: int = SEQ_LEN
) -> np.ndarray:
    """(N, horizon) float32 ``v_lead[t+1 : t+horizon+1]`` of windows (event, t), cut like ``make_windows`` does.

    v_lead: concatenated leader speeds of all events [m/s]; lengths (n_events,) samples per event.
    """
    event_index = np.asarray(event_index, dtype=np.int64)
    t_index = np.asarray(t_index, dtype=np.int64)
    if np.any(t_index + horizon >= lengths[event_index]):
        raise ValueError("some windows have no complete leader future inside their event")
    start = (np.cumsum(lengths) - lengths)[event_index] + t_index + 1
    return np.asarray(v_lead).astype(np.float32, copy=False)[start[:, None] + np.arange(horizon)]


def sampled_rows(
    event_index: np.ndarray,
    t_index: np.ndarray,
    lengths: np.ndarray,
    max_rows: int | None,
    seed: int = 0,
    horizon: int = SEQ_LEN,
) -> np.ndarray:
    """Sorted rows evaluated by a sampling model (DiffusionCF): at most ``max_rows`` windows with a full future.

    Window (event e, last history sample t) qualifies when ``t + horizon < lengths[e]``: the targets
    ``a[t : t+horizon]`` and leader speeds ``v_lead[t+1 : t+horizon+1]`` lie inside the event. The subset is a
    draw without replacement from ``np.random.default_rng(seed)``; it depends only on the windows, ``max_rows``
    and ``seed``, so every model seed is evaluated on the same rows.
    """
    event_index = np.asarray(event_index, dtype=np.int64)
    t_index = np.asarray(t_index, dtype=np.int64)
    ok = np.flatnonzero(t_index + int(horizon) < np.asarray(lengths, dtype=np.int64)[event_index])
    if max_rows is None or ok.size <= int(max_rows):
        return ok
    return np.sort(np.random.default_rng(int(seed)).choice(ok, size=int(max_rows), replace=False))


class Data:
    """Event sets (each directory loaded once), fold splits, highD training bundles and evaluation windows."""

    def __init__(self, cfg: DictConfig, log: Callable[[str], None] | None = None) -> None:
        self.cfg = cfg
        self.log = log or (lambda message: None)
        self._events: dict[tuple[str, bool, str], EventSet] = {}
        self._reports: dict[tuple[str, bool, str], dict[str, Any]] = {}
        self._parts: dict[str, dict[str, EventSet]] = {}
        self._bundles: dict[bool, DataBundle] = {}
        self._eval: dict[str, EvalSet] = {}

    def config(self, name: str) -> DictConfig:
        """Data config ``name``: inline (``data_configs``) or ``config_paths.data/<name>.yaml``."""
        return data_config(self.cfg, name)

    def signal(self, name: str) -> str:
        """Signal variant of data set ``name`` (``data_signal``)."""
        return data_signal(self.cfg, name)

    def signal_key(self, name: str) -> dict[str, Any] | None:
        """``signal_key`` of data set ``name`` (None for raw signals)."""
        return signal_key(self.cfg, name)

    def _key(self, name: str) -> tuple[str, bool, str]:
        dcfg = self.config(name)
        if str(dcfg.get("kind", "parquet")) != "parquet":
            raise ValueError(f"data set {name!r}: E1 needs a parquet EventSet, got kind {dcfg.kind!r}")
        unify = bool(dcfg.get("unify_acceleration", True))
        return str(_path(dcfg.path).resolve()), unify, json.dumps(self.signal_key(name), sort_keys=True)

    def events(self, name: str) -> EventSet:
        """All events of data set ``name`` with the common acceleration definition and its signal variant (harmonised:
        dropped events left out; loaded once per directory and variant)."""
        key = self._key(name)
        if key not in self._events:
            dcfg, sig = self.config(name), self.signal_key(name)
            started = time.perf_counter()
            if sig is None:
                self._events[key] = load_events(_path(dcfg.path), unify=key[1])
                self.log(f"data {name}: {len(self._events[key]):,} events loaded in {time.perf_counter() - started:.1f} s")
            else:
                events, report = load_events(
                    _path(dcfg.path), unify=key[1], signal=sig["signal"], harmonise=sig["harmonise"], return_report=True
                )  # fmt: skip
                rep = dict(report or {})
                self._events[key], self._reports[key] = events, rep
                dropped = ", ".join(f"{k} {v}" for k, v in rep.get("dropped", {}).items() if v)
                gap, c = rep.get("gap_rms_m", {}).get("q50"), rep.get("abs_c_mps", {}).get("q50")
                self.log(
                    f"data {name}: {len(events):,} of {rep.get('n_in', 0):,} events kept after harmonisation (window "
                    f"{sig['harmonise']['window']}{', matched' if self.signal(name) == 'matched' else ''}; dropped: "
                    f"{dropped or 'none'}; gap rms median {'-' if gap is None else f'{gap:.3f}'} m, |c| median "
                    f"{'-' if c is None else f'{c:.4f}'} m/s) in {time.perf_counter() - started:.1f} s"
                )
        return self._events[key]

    def harmonisation(self, name: str) -> dict[str, Any] | None:
        """Harmonisation report of data set ``name`` (``harmonise_events``); None for raw signals or before its
        events were loaded."""
        return self._reports.get(self._key(name))

    def match_info(self, name: str) -> dict[str, Any] | None:
        """Provenance of the matched window of data set ``name`` (signal ``matched``, design.md 15.1): the matching
        file, the window and the file's entry of the data set (role, roughness, flags, ...); None otherwise."""
        if self.signal(name) != "matched":
            return None
        path = matching_path(self.cfg)
        content = read_matching(path)
        entry = (content.get("datasets") or {}).get(name) or {}
        keep = ("role", "window_rule", "cap", "flags", "roughness", "roughness_harmonised", "ratio", "short_event_share",
                "share_dropped")  # fmt: skip
        return _jsonable({"file": _display(path), "reference": content.get("reference"),
                          "window": int(self.signal_key(name)["harmonise"]["window"]),  # type: ignore[index]
                          **{k: content[k] for k in ("reference_roughness", "target_roughness") if k in content},
                          **{k: entry[k] for k in keep if k in entry}})  # fmt: skip

    def fold_file(self, name: str) -> Path | None:
        split = self.config(name).get("split")
        return _path(split.file) if split is not None and split.get("file") else None

    def fold(self, name: str) -> int | None:
        return None if self.fold_file(name) is None else int(self.config(name).split.get("fold", 0))

    def folds(self, name: str) -> dict[str, int] | None:
        """Event id -> fold of the data set's cf-stability split file (None without one)."""
        file = self.fold_file(name)
        if file is None:
            return None
        return {str(k): int(v) for k, v in json.loads(file.read_text(encoding="utf-8"))["folds"].items()}

    def split(self, name: str, part: str) -> EventSet:
        """Events of ``part`` (all | train | val | test; fold rule of cf-stability)."""
        events = self.events(name)
        if part == "all":
            return events
        if name not in self._parts:
            file = self.fold_file(name)
            if file is None:
                raise ValueError(f"data set {name!r} has no split file: only split 'all' is available")
            self._parts[name] = split_by_folds(events, file, int(self.fold(name)))  # type: ignore[arg-type]
        return self._parts[name][part]

    @property
    def target(self) -> str:
        """Target of the experiment (``train.target``, default ``"a"``): training and evaluated windows."""
        return config_target(self.cfg)

    @property
    def train_names(self) -> list[str]:
        """Training data sets (``train_datasets``: one, or the data sets of a pooled domain)."""
        return train_datasets(self.cfg)

    @property
    def pooled(self) -> bool:
        return is_pooled(self.cfg)

    def train_events(self) -> EventSet:
        """The training events: the training fold of ``train.data``, or the union of the training folds of a pooled
        domain in the order of ``train.data`` (IDM calibrations, design.md 16.1)."""
        if not self.pooled:
            return self.split(self.train_names[0], "train")
        return EventSet([ev for name in self.train_names for ev in self.split(name, "train")])

    def _window_settings(self, sequences: bool) -> dict[str, Any]:
        """Stride, sequence length, target and (pooled domain) cap of the one-step or the sequence training windows."""
        train = self.cfg.train
        cap = train.get("max_sequences_per_dataset" if sequences else "max_windows_per_dataset")
        return {
            "stride": int(train.seq_stride if sequences else train.stride),
            "seq_len": SEQ_LEN if sequences else 0,
            "target": DEFAULT_TARGET if sequences else self.target,
            "cap": _cap(cap) if self.pooled else None,
        }

    def bundle(self, sequences: bool) -> DataBundle:
        """Training bundle (train / val / test windows); ``sequences`` adds 5 s targets and leader futures.

        One-step windows carry the experiment's target; sequence windows keep ``y = a[t]`` whatever the target
        (DiffusionCF learns the acceleration sequences ``y_seq``, so its checkpoints serve every target). A pooled
        training domain (design.md 16.1): ``build_pooled_bundle`` of the training and validation folds of every data
        set of ``train.data``, each capped at ``train.max_windows_per_dataset`` (sequences:
        ``train.max_sequences_per_dataset``) with the seed ``train.subsample_seed`` (no test windows: the test folds
        are evaluated per data set); the counts per data set are logged and kept in ``meta["pool"]``.
        """
        if sequences not in self._bundles:
            train = self.cfg.train
            w = self._window_settings(sequences)
            if self.pooled:
                parts = {n: {p: self.split(n, p) for p in POOL_SPLITS} for n in self.train_names}
                started = time.perf_counter()
                bundle = build_pooled_bundle(
                    parts, hist_len=HIST_LEN, stride=w["stride"], seq_len=w["seq_len"], target=w["target"],
                    max_windows=w["cap"], seed=int(train.get("subsample_seed", 0) or 0), name="pool",
                )  # fmt: skip
                kind = "sequence" if sequences else "one-step"
                counts = "; ".join(
                    f"{n} " + ", ".join(f"{p} {c[p]['windows']:,} -> {c[p]['kept']:,}" for p in POOL_SPLITS)
                    for n, c in bundle.meta["pool"].items()
                )  # fmt: skip
                self.log(f"pooled {kind} windows (stride {w['stride']}, cap {w['cap'] if w['cap'] is not None else 'none'} "
                         f"per data set): {counts}; total train {len(bundle.windows['train']):,}, val "
                         f"{len(bundle.windows['val']):,} ({time.perf_counter() - started:.1f} s)")  # fmt: skip
                self._bundles[sequences] = bundle
            else:
                name = self.train_names[0]
                kwargs: dict[str, Any] = dict(hist_len=HIST_LEN, stride=w["stride"], seq_len=w["seq_len"], target=w["target"])
                if self.fold_file(name) is not None:
                    kwargs.update(fold_file=self.fold_file(name), fold=self.fold(name))
                self._bundles[sequences] = build_bundle(self.events(name), None, name=name, **kwargs)
        return self._bundles[sequences]

    def training_plan(self) -> dict[str, Any]:
        """Window counts of a pooled training domain without cutting a window (``count_windows`` on the event lengths;
        loads the events of the training data sets): per kind (``one_step``, ``sequences``), data set and split
        (train, val) ``events``, ``windows`` after the stride and ``kept`` after the cap, plus the totals."""
        out: dict[str, Any] = {"data": self.train_names, "signal": describe_signal(training_signal_key(self.cfg)),
                               "pool": pool_key(self.cfg)}  # fmt: skip
        for kind, sequences in (("one_step", False), ("sequences", True)):
            w = self._window_settings(sequences)
            per: dict[str, Any] = {}
            for name in self.train_names:
                per[name] = {}
                for part in POOL_SPLITS:
                    events = self.split(name, part)
                    total = int(count_windows([len(ev) for ev in events], HIST_LEN, w["stride"], w["seq_len"], w["target"]).sum())
                    kept = total if w["cap"] is None else min(total, w["cap"])
                    per[name][part] = {"events": len(events), "windows": total, "kept": kept}
            totals = {p: sum(v[p]["kept"] for v in per.values()) for p in POOL_SPLITS}
            out[kind] = {"stride": w["stride"], "cap": w["cap"], "datasets": per, "total": totals}
        return out

    def pool_spec(self, dataset: str, members: Sequence[str]) -> dict[str, Any]:
        """Windows spec of a pool (design.md 16.1): ``members`` (in order), ``pool`` = the spec of every member,
        ``data`` = their data sets, ``split`` (common split of the members), ``hist_len``; ``target`` when not
        ``"a"``; ``signal`` when every member has the same (not raw) signal variant."""
        datasets = self.cfg.eval.datasets
        specs = {}
        for m in members:
            if m not in datasets or pool_members(self.cfg, m) is not None:
                raise ValueError(f"pool {dataset!r}: member {m!r} must be an evaluated data set that is not a pool")
            specs[m] = self.eval_spec(m)
        splits = sorted({str(s["split"]) for s in specs.values()})
        spec: dict[str, Any] = {"members": list(members), "pool": specs, "data": [specs[m]["data"] for m in members],
                                "split": splits[0] if len(splits) == 1 else splits, "hist_len": HIST_LEN}  # fmt: skip
        if self.target != DEFAULT_TARGET:
            spec["target"] = self.target
        signals = {s.get("signal") for s in specs.values()}
        if len(signals) == 1 and None not in signals:
            spec["signal"] = signals.pop()
        return _jsonable(spec)

    def eval_spec(self, dataset: str) -> dict[str, Any]:
        """Settings that define the windows of ``dataset`` (stored in its meta and compared on resume).

        ``target`` is included only when it is not the default ``"a"``, ``signal`` and ``harmonise`` only for
        harmonised signals, so that the specs stored before these options existed (target ``"a"``, raw signals)
        stay valid. A pool (``{pool: [...]}``): ``pool_spec``.
        """
        members = pool_members(self.cfg, dataset)
        if members is not None:
            return self.pool_spec(dataset, members)
        d = self.cfg.eval.datasets[dataset]
        name, part = str(d.data), str(d.get("split", "all"))
        max_events = d.get("max_events", self.cfg.eval.get("max_events"))
        spec: dict[str, Any] = {
            "data": name,
            "split": part,
            "fold": None if part == "all" else self.fold(name),
            "stride": int(d.get("stride", 1)),
            "hist_len": HIST_LEN,
            "max_events": None if max_events is None else int(max_events),
            "unify_acceleration": bool(self.config(name).get("unify_acceleration", True)),
        }
        if self.target != DEFAULT_TARGET:
            spec["target"] = self.target
        sig = self.signal_key(name)
        if sig is not None:
            spec.update(sig)
        return _jsonable(spec)

    def pool_folds(self, dataset: str) -> dict[str, int]:
        """Event id -> fold of the member events of a pool (each member's split file; ValueError when an event id of
        two members has two folds)."""
        out: dict[str, int] = {}
        for m in self.eval_spec(dataset)["members"]:
            folds = self.folds(str(self.cfg.eval.datasets[m].data)) or {}
            for ev in self.eval_set(m).events:
                fold = int(folds.get(ev.event_id, -1))
                if out.setdefault(ev.event_id, fold) != fold:
                    raise ValueError(f"pool {dataset!r}: event {ev.event_id!r} has two folds")
        return out

    def eval_set(self, dataset: str) -> EvalSet:
        """Events and windows of an evaluated data set (a pool: the members' events and windows concatenated in the
        order of its members, ``concat_windows``)."""
        if dataset not in self._eval:
            spec = self.eval_spec(dataset)
            if "members" in spec:
                sets = [self.eval_set(m) for m in spec["members"]]
                events = EventSet([ev for s in sets for ev in s.events])
                windows = concat_windows([s.windows for s in sets], [len(s.events) for s in sets])
                self._eval[dataset] = EvalSet(dataset, spec, events, windows)
                return self._eval[dataset]
            events = self.split(spec["data"], spec["split"])
            if spec["max_events"] is not None:
                events = EventSet(list(events)[: spec["max_events"]])
            target = str(spec.get("target", DEFAULT_TARGET))
            windows = make_windows(events, hist_len=HIST_LEN, stride=spec["stride"], target=target)
            self._eval[dataset] = EvalSet(dataset, spec, events, windows)
        return self._eval[dataset]


# ---------------------------------------------------------------------------- models
@dataclass
class ModelSpec:
    """One model configuration of the experiment.

    label: model config name and artifact directory (``hetero_mdn_mix3``); name: registry name (``hetero_mdn``);
    kwargs: constructor arguments (plain JSON); sequences: trained on 5 s sequence windows (DiffusionCF);
    seed_independent: trained once, the checkpoint is copied to every seed; idm: ``{"name", "key"}`` of the IDM
    calibration in the kwargs (None without IDM), checked on resume together with the kwargs; target: target of
    the training windows (``train.target``), checked on resume; None for sequence models, whose checkpoints
    serve every target; signal: ``training_signal_key`` of the training data (None: raw signals; a list for a
    pooled training domain), checked on resume for every model; pool: ``pool_key`` of a pooled training domain
    (None: one training data set), checked with the signal.
    """

    label: str
    name: str
    kwargs: dict[str, Any]
    sequences: bool = False
    seed_independent: bool = False
    idm: dict[str, Any] | None = None
    target: str | None = DEFAULT_TARGET
    signal: dict[str, Any] | list[Any] | None = None
    pool: dict[str, Any] | None = None


def model_config(cfg: DictConfig, label: str) -> DictConfig:
    """Model config ``label``: inline (``model_configs``) or ``config_paths.model/<label>.yaml``."""
    inline = cfg.get("model_configs") or {}
    if label in inline:
        return inline[label]
    return OmegaConf.load(_path(cfg.config_paths.model) / f"{label}.yaml")  # type: ignore[return-value]


def model_spec(cfg: DictConfig, label: str, calibration: Calibration | None = None) -> ModelSpec:
    """Constructor kwargs exactly as ``cf_unc.train.run.model_kwargs`` builds them (model config, train config,
    IDM), then ``train.overrides`` (models with a ``train`` argument) and ``model_overrides[label]`` (deep merge).

    A model config with ``needs_idm`` gets the IDM calibration named by its ``idm`` key (default ``closed_loop``)
    from ``calibration(name) -> (params, key)``.
    """
    mcfg = model_config(cfg, label)
    idm, idm_ref = None, None
    if mcfg.get("needs_idm", False):
        name = str(mcfg.get("idm", DEFAULT_IDM))
        if calibration is None:
            raise ValueError(f"model {label!r} needs the IDM calibration {name!r}")
        idm, key = calibration(name)
        idm_ref = {"name": name, "key": key}
    composed = OmegaConf.create({"model": mcfg, "train": OmegaConf.load(_path(cfg.config_paths.train))})
    kwargs = model_kwargs(composed, idm)  # type: ignore[arg-type]
    overrides = _plain(cfg.train.get("overrides")) or {}
    if overrides and "train" in kwargs:
        kwargs["train"] = {**kwargs["train"], **overrides}
    kwargs = _deep_merge(kwargs, _plain((cfg.get("model_overrides") or {}).get(label)) or {})
    sequences = bool(mcfg.get("needs_sequences", False))
    return ModelSpec(
        label=label,
        name=str(mcfg.name),
        kwargs=_jsonable(kwargs),
        sequences=sequences,
        seed_independent=label in [str(m) for m in (cfg.get("seed_independent") or [])],
        idm=_jsonable(idm_ref),
        target=None if sequences else config_target(cfg),
        signal=training_signal_key(cfg),
        pool=pool_key(cfg),
    )


def _fit_summary(log: Mapping[str, Any]) -> str:
    """``epochs`` and ``val NLL`` of a training log (whatever the model family), for progress lines."""
    epochs = log.get("epochs", (log.get("fit") or {}).get("epochs"))
    if epochs is None and "members" in log:
        epochs = "+".join(str(m.get("fit", {}).get("epochs", "?")) for m in log["members"])
    val = log.get("val_nll")
    return f"epochs {epochs if epochs is not None else '-'}, val NLL {'-' if val is None else f'{val:.4f}'}"


# ------------------------------------------------------------------------ prediction
def predict_table(
    model: CFModel,
    ev: EvalSet,
    rows: np.ndarray,
    *,
    seed: int,
    chunk_rows: int = 50_000,
    batch_size: int = 8192,
    coverage: float = 0.9,
    target: str | None = None,
) -> pd.DataFrame:
    """Prediction table of ``ev`` (columns ``cf_unc.maps.io.PRED_COLUMNS``, one row per window).

    The windows ``rows`` are predicted in chunks of ``chunk_rows`` (``predict_dist`` -> ``summarize`` ->
    ``pred_table``; one generator seeded with ``seed`` drives the randomised PIT of sample laws); every other row
    is NaN with ``covered = 0``. ``target`` (the target of ``ev.windows.y``) is passed to ``predict_dist`` when it
    has a ``target`` parameter (DiffusionCF); every other model predicts the target it was trained for.
    """
    n = len(ev.windows)
    out = {c: np.zeros(n, np.uint8) if c == COVERED else np.full(n, np.nan, np.float32) for c in tables.PRED_COLUMNS}
    rng = np.random.default_rng(int(seed))
    extra = {"target": target} if target is not None and accepts_keyword(model.predict_dist, "target") else {}
    for a in range(0, rows.size, int(chunk_rows)):
        idx = rows[a : a + int(chunk_rows)]
        future = ev.leader_future(idx) if model.needs_leader_future else None
        dist = model.predict_dist(ev.windows.X[idx], v_lead_future=future, batch_size=int(batch_size), **extra)
        part = tables.pred_table(dist, summarize(dist, ev.windows.y[idx], coverage=float(coverage), rng=rng))
        for c in tables.PRED_COLUMNS:
            out[c][idx] = part[c].to_numpy()
    return pd.DataFrame(out)[list(tables.PRED_COLUMNS)]


def quick_metrics(pred: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    """Quick look of one table: means over the evaluated rows (finite ``mean``), ``picp`` = mean of ``covered``.

    ``nonfinite`` counts non-finite entries of the columns that must be finite on evaluated rows.
    """
    mean = pred["mean"].to_numpy(np.float64)
    ok = np.isfinite(mean)

    def avg(col: str) -> float | None:
        values = pred[col].to_numpy(np.float64)[ok]
        return None if values.size == 0 or np.all(np.isnan(values)) else float(np.mean(values))

    out: dict[str, Any] = {"n_rows": int(len(pred)), "n_evaluated": int(ok.sum())}
    out.update({c: avg(c) for c in ("nll", "crps", "var", "aleatoric_var", "epistemic_var", "width")})
    out["picp"] = avg(COVERED)
    resid = np.asarray(y, dtype=np.float64)[ok] - mean[ok]
    out["rmse"] = float(np.sqrt(np.mean(resid**2))) if resid.size else None
    out["nonfinite"] = {c: int(np.sum(~np.isfinite(pred[c].to_numpy(np.float64)[ok]))) for c in ("var", "nll", "crps", "width", "pit")}
    return out


# ---------------------------------------------------------------------------- runner
class _NotStored(Exception):
    """A stored setting the dry run needs is absent (e.g. an IDM calibration not computed yet)."""


class RunLog:
    """Timestamped progress lines (stdout and ``logs/<name>``) and failure tracebacks (``logs/errors.log``).

    ``write=False``: stdout only, nothing is created or written (dry runs).
    """

    def __init__(self, root: Path, echo: bool = True, name: str = "run.log", write: bool = True) -> None:
        self.dir = root / "logs"
        self.write = write
        if write:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self.name = name
        self.failures: list[dict[str, str]] = []

    def line(self, message: str) -> None:
        text = f"{_now()}  {message}"
        if self.echo:
            print(text, flush=True)
        if self.write:
            with open(self.dir / self.name, "a", encoding="utf-8") as fh:
                fh.write(text + "\n")

    def error(self, unit: str, exc: BaseException) -> None:
        if self.write:
            with open(self.dir / "errors.log", "a", encoding="utf-8") as fh:
                fh.write(f"{_now()}  {unit}\n{''.join(traceback.format_exception(exc))}\n")
        self.failures.append({"unit": unit, "error": f"{type(exc).__name__}: {exc}", "time": _now()})
        self.line(f"FAILED {unit}: {type(exc).__name__}: {exc}")


class Runner:
    """State of one call of ``run_e1`` (see the module docstring); also the toolbox of scripts/select_w_phys.py."""

    def __init__(self, cfg: DictConfig, echo: bool = True, log_name: str = "run.log", dry: bool = False) -> None:
        self.cfg = cfg
        self.root = _path(cfg.root)
        # signal matched: the windows of the matching file; refuses (MatchingError) before anything is created
        self.windows = check_matching(cfg)
        self.train_names = train_datasets(cfg)
        if not self.train_names:
            raise ValueError("the experiment config needs train.data (a data set or a list of data sets)")
        self.pooled = is_pooled(cfg)
        check_pool_config(cfg)  # a pooled domain never evaluates training events (ValueError, nothing created)
        if not dry:  # a dry run creates nothing (dry_run)
            self.root.mkdir(parents=True, exist_ok=True)
        self.log = RunLog(self.root, echo, log_name, write=not dry)
        self.device = resolve_device(str(cfg.get("device", "auto")))
        self.target = config_target(cfg)
        self.signal = data_signal(cfg, self.train_names[0])  # signal variant of the training data
        self.calibrations: dict[str, tuple[IDMParams, dict[str, Any]]] = {}
        self.data = Data(cfg, self.log.line)
        self.models = [str(m) for m in cfg.models]
        self.seeds = [int(s) for s in cfg.seeds]
        # a pool (union of member tables) after every other data set: its tables are assembled from the members'
        names = [str(d) for d in cfg.eval.datasets]
        self.pools = {d: m for d in names if (m := pool_members(cfg, d)) is not None}
        self.datasets = [d for d in names if d not in self.pools] + [d for d in names if d in self.pools]
        self.ready: list[str] = []  # data sets whose windows table is in place
        self.checked: set[str] = set()  # data sets whose in-memory windows were compared with the stored table
        self.trained: list[str] = []
        self.written: list[str] = []
        self.current: str | None = None
        self.started, self.started_at = time.perf_counter(), _now()

    # ------------------------------------------------------------------ bookkeeping
    def guard(self, unit: str, fn: Callable[..., Any], *args: Any) -> Any:
        """``fn(*args)``; a failure is logged (traceback in errors.log) and gives None."""
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 - one failing unit must not stop the run
            self.log.error(unit, exc)
            return None

    def status(self, finished: bool = False) -> dict[str, Any]:
        pairs = [(m, s) for m in self.models for s in self.seeds]
        n_ckpt = sum((checkpoint_dir(self.root, m, s) / "model.pt").exists() for m, s in pairs)
        n_tab = sum(tables.preds_path(self.root, m, s, d).exists() for m, s in pairs for d in self.datasets)
        return {
            "root": str(self.root),
            "target": self.target,
            "signal": self.signal,
            **({"train_data": list(self.train_names)} if self.pooled else {}),
            "device": str(self.device),
            "started": self.started_at,
            "updated": _now(),
            "elapsed_s": round(time.perf_counter() - self.started, 1),
            "finished": finished,
            "current": None if finished else self.current,
            "done": n_ckpt + n_tab,
            "total": len(pairs) * (1 + len(self.datasets)),
            "checkpoints": {"done": n_ckpt, "total": len(pairs), "trained_now": len(self.trained)},
            "tables": {"done": n_tab, "total": len(pairs) * len(self.datasets), "written_now": len(self.written)},
            "trained": list(self.trained),
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

    # --------------------------------------------------------------------- data
    def stored_windows(self, dataset: str) -> bool:
        """Whether ``windows/<dataset>.parquet`` and its meta exist with the configured settings (target and signal
        included); ValueError when they were written with other settings. Reads the meta only."""
        spec = self.data.eval_spec(dataset)
        path = tables.windows_path(self.root, dataset)
        meta_path = path.with_name(f"{dataset}_meta.json")
        if not (path.exists() and meta_path.exists()):
            return False
        stored = json.loads(meta_path.read_text(encoding="utf-8")).get("spec")
        if not _same_spec(stored, spec):
            raise ValueError(f"windows/{dataset} was written with {stored}, the config asks for {spec}; use another root")
        return True

    def ensure_windows(self, dataset: str) -> None:
        """Write ``windows/<dataset>.parquet`` and its meta unless present with the same settings (target and signal
        included); harmonised tables also store the harmonisation report of their data set in the meta. A pool
        (design.md 16.1): the concatenation of its members' windows (``Data.eval_set``), with the folds of their split
        files; its meta adds ``members`` (name, data, split, stride and the row and event ranges of every member in the
        pool table), and the per-member harmonisation reports and matched windows."""
        if not self.stored_windows(dataset):
            spec = self.data.eval_spec(dataset)
            ev = self.data.eval_set(dataset)
            members = spec.get("members")
            folds = self.data.pool_folds(dataset) if members is not None else self.data.folds(spec["data"])
            frame, meta = tables.windows_table(ev.events, ev.windows, folds, target=self.target)
            meta.update(
                dataset=dataset,
                spec=spec,
                stride=spec["stride"] if members is None else {m: spec["pool"][m]["stride"] for m in members},
                counts={
                    "windows": int(len(frame)),
                    "events_with_windows": int(frame["event"].nunique()),
                    "drivers": int(frame["driver"].nunique()),
                    "subset": {str(k): int(v) for k, v in frame["subset"].value_counts(sort=False).items()},
                    "fold": {str(k): int(v) for k, v in frame["fold"].value_counts().sort_index().items()},
                },
            )
            if members is None:
                if "signal" in spec:
                    meta.update(signal=spec["signal"], harmonisation=self.data.harmonisation(spec["data"]))
                match = self.data.match_info(spec["data"])
                if match is not None:  # signal matched: provenance of the window (not part of the spec)
                    meta["match"] = match
                where = f"{spec['data']}, split {spec['split']}, stride {spec['stride']}"
            else:
                rows = events = 0
                info = []
                for m in members:
                    part, data = self.data.eval_set(m), spec["pool"][m]["data"]
                    info.append({"name": m, "data": data, "split": spec["pool"][m]["split"], "stride": spec["pool"][m]["stride"],
                                 "rows": [rows, rows + len(part.windows)], "events": [events, events + len(part.events)]})  # fmt: skip
                    rows, events = rows + len(part.windows), events + len(part.events)
                meta["members"] = info
                if "signal" in spec:
                    meta.update(signal=spec["signal"],
                                harmonisation={m["name"]: self.data.harmonisation(m["data"]) for m in info})  # fmt: skip
                matches = {m["name"]: self.data.match_info(m["data"]) for m in info}
                if any(v is not None for v in matches.values()):
                    meta["match"] = matches
                where = "pool of " + ", ".join(f"{m['name']} {m['rows'][1] - m['rows'][0]:,}" for m in info)
            tables.save_windows(self.root, dataset, frame, meta)
            self.log.line(f"windows {dataset}: {len(frame):,} rows from {len(ev.events):,} events ({where})")
        self.ready.append(dataset)

    def check_windows(self, dataset: str, ev: EvalSet) -> None:
        """The in-memory windows are those of the stored table (same count and events)."""
        if dataset in self.checked:
            return
        meta_path = tables.windows_path(self.root, dataset).with_name(f"{dataset}_meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["n_windows"] != len(ev.windows) or meta["event_ids"] != [e.event_id for e in ev.events]:
            raise RuntimeError(f"windows of {dataset} differ from the stored table; use another root")
        self.checked.add(dataset)

    def density_stride(self) -> int:
        """Stride of the training density: ``density_stride`` (default 1) for one training data set; for a pooled
        domain the training windows themselves (``train.stride``, after the caps)."""
        return int(self.cfg.train.stride) if self.pooled else int(self.cfg.get("density_stride", 1))

    def stored_density(self) -> bool:
        """Whether ``train_density.npz`` exists with the configured stride, target, signal and (pooled domain) pool;
        ValueError when it was computed with other settings (a file without ``target`` counts as ``"a"``, without
        ``signal`` as raw, without ``pool`` as one training data set)."""
        path = self.root / "train_density.npz"
        if not path.exists():
            return False
        stride = self.density_stride()
        sig, pool = training_signal_key(self.cfg), pool_key(self.cfg)
        with np.load(path) as stored:
            stored_stride = int(stored["stride"])
            stored_target = str(stored["target"]) if "target" in stored.files else DEFAULT_TARGET
            stored_signal = json.loads(str(stored["signal"])) if "signal" in stored.files else None
            stored_pool = json.loads(str(stored["pool"])) if "pool" in stored.files else None
        if stored_stride != stride or stored_target != self.target:
            raise ValueError(
                f"{path} was computed with stride {stored_stride} and target {stored_target!r}, the config asks "
                f"{stride} and {self.target!r}"
            )
        if stored_signal != sig:
            raise ValueError(
                f"{path} was computed on {describe_signal(stored_signal)} signals, the config asks "
                f"{describe_signal(sig)}; use another root"
            )
        if stored_pool != pool:
            raise ValueError(f"{path} was computed on the training windows of {describe_pool(stored_pool)}, the config "
                             f"asks {describe_pool(pool)}; use another root")  # fmt: skip
        return True

    def ensure_density(self) -> None:
        """``train_density.npz``: highD training windows (``density_stride``, cut for the experiment's target, of its
        signal variant) on the grid of the maps config; ``target`` and (when not raw) ``signal`` are stored. A pooled
        domain (design.md 16.1): the pooled one-step training windows the models are trained on (stride
        ``train.stride``, capped per data set); ``signal`` (the list key) and ``pool`` (``pool_key``) are stored."""
        if self.stored_density():
            return
        path = self.root / "train_density.npz"
        stride = self.density_stride()
        sig, pool = training_signal_key(self.cfg), pool_key(self.cfg)
        s_edges, dv_edges = grid_edges(OmegaConf.load(_path(self.cfg.config_paths.maps)).grid)  # type: ignore[arg-type]
        counts, outside, total = np.zeros((len(s_edges) - 1, len(dv_edges) - 1), np.int64), 0, 0
        if self.pooled:
            last = self.data.bundle(False).windows["train"].X[:, -1, :]
            counts, outside = bin_counts(last[:, 0], last[:, 1], s_edges, dv_edges)
            total = last.shape[0]
        else:
            events = list(self.data.split(self.train_names[0], "train"))
            for a in range(0, len(events), 1000):  # event chunks bound the memory of the windows
                chunk = events[a : a + 1000]
                last = make_windows(chunk, hist_len=HIST_LEN, stride=stride, target=self.target).X[:, -1, :]
                part, part_outside = bin_counts(last[:, 0], last[:, 1], s_edges, dv_edges)
                counts, outside, total = counts + part, outside + part_outside, total + last.shape[0]
        tmp = self.root / "train_density.tmp.npz"
        np.savez(
            tmp,
            s_edges=s_edges,
            dv_edges=dv_edges,
            counts=counts,
            n_outside=np.int64(outside),
            n_total=np.int64(total),
            stride=np.int64(stride),
            target=np.str_(self.target),
            **({} if sig is None else {"signal": np.str_(json.dumps(sig, sort_keys=True))}),
            **({} if pool is None else {"pool": np.str_(json.dumps(pool, sort_keys=True))}),
        )
        tmp.replace(path)
        what = "training windows" if pool is None else "pooled training windows (after the caps)"
        self.log.line(
            f"train density: {total:,} {what} (stride {stride}, target {self.target}, signal "
            f"{self.signal}), {outside / max(total, 1):.1%} outside the grid"
        )

    def calibration(self, name: str) -> tuple[IDMParams, dict[str, Any]]:
        """Named IDM calibration ``cfg.idm[name]`` of the training events and its key, cached in ``idm.json``.

        ``idm.json`` holds every computed calibration under ``calibrations/<name>`` (``params``, ``report`` of
        ``calibrate_idm``, ``train_eval`` = ``evaluate_idm`` on all training events, ``key`` = data, fold and
        settings, plus the ``signal_key`` of the training data when it is not raw); ``closed_loop`` is also kept at
        the top level (``params``, ``report``, ``key``) next to the cf-stability ``reference``. A stored calibration
        whose key differs from the configured one is an error. A pooled domain (design.md 16.1) is calibrated on the
        union of the training folds (sampled as for one data set); its key holds the lists of data sets, folds and
        signal keys.
        """
        if name in self.calibrations:
            return self.calibrations[name]
        found = self.stored_calibration(name)
        if found is not None:
            self.calibrations[name] = found
            return found
        if idm_from_path(self.cfg) is not None:  # a frozen prior: copied, not fitted
            return self.copy_calibration(name)
        key = self.calibration_key(name)
        options = dict(_plain((self.cfg.get("idm") or {})[name]))
        path = self.root / "idm.json"
        stored, entries = self._stored_calibrations()
        events = self.data.train_events()  # a pooled domain: the union of the training folds (design.md 16.1)
        objective = str(options.pop("objective", "spacing"))
        max_events = options.pop("max_events", 300)
        seed = int(options.pop("seed", 0))
        self.log.line(f"idm {name}: calibrating ({objective}, at most {max_events} of {len(events):,} training events)")
        params, report = calibrate_idm(
            events, objective=objective, max_events=None if max_events is None else int(max_events), seed=seed,
            **options,
        )  # fmt: skip
        train_eval = evaluate_idm(events, params)
        entries[name] = {"params": params.to_dict(), "report": report, "train_eval": train_eval, "key": key}
        payload = {**stored, "calibrations": entries, "reference": self.idm_reference()}
        if name == DEFAULT_IDM:
            payload.update(params=params.to_dict(), report=report, key=key)
        _write_json_atomic(path, payload)
        values = ", ".join(f"{k}={v:.3g}" for k, v in params.to_dict().items())
        self.log.line(
            f"idm {name}: {values} ({report['seconds']:.0f} s); all {train_eval['n_events']:,} training events: "
            f"one-step RMSE {train_eval['accel_rmse']:.3f} m/s^2, closed-loop spacing RMSE "
            f"{train_eval['spacing_rmse']:.2f} m, {train_eval['n_collisions']} collisions"
        )
        self.calibrations[name] = (params, key)
        return params, key

    def source_calibration(self, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """The named calibration of the ``train.idm_from`` file (read only) and its provenance (``path``, ``sha256``
        of the file, ``note``). ValueError when the file does not exist or holds no such calibration."""
        path = idm_from_path(self.cfg)
        if path is None:
            raise ValueError("train.idm_from is not set")
        if not path.exists():
            raise ValueError(f"train.idm_from: {path} does not exist")
        raw = path.read_bytes()
        entries = calibration_entries(json.loads(raw.decode("utf-8")))
        if name not in entries or not (entries[name] or {}).get("params"):
            raise ValueError(f"train.idm_from: {_display(path)} holds no IDM calibration {name!r} (it holds {sorted(entries)})")
        provenance = {"path": _display(path), "sha256": hashlib.sha256(raw).hexdigest(),
                      "note": "frozen IDM prior: the calibration of another root, copied (train.idm_from), not fitted here"}  # fmt: skip
        return _jsonable(entries[name]), provenance

    def source_names(self) -> list[str]:
        """Names of the calibrations of the ``train.idm_from`` file (``closed_loop`` first; [] without the key or the
        file): a run copies all of them, so the regimes always get the reference ``v0``."""
        path = idm_from_path(self.cfg)
        if path is None or not path.exists():
            return []
        names = sorted(calibration_entries(json.loads(path.read_text(encoding="utf-8"))))
        return sorted(names, key=lambda n: n != DEFAULT_IDM)

    def reused_calibration(self, name: str) -> tuple[IDMParams, dict[str, Any]]:
        """Parameters and key of the calibration ``name`` that ``copy_calibration`` would copy (nothing is written)."""
        entry, _ = self.source_calibration(name)
        return IDMParams.from_dict(entry["params"]), self.calibration_key(name)

    def copy_calibration(self, name: str) -> tuple[IDMParams, dict[str, Any]]:
        """Copy the calibration ``name`` of ``train.idm_from`` into ``idm.json`` (module docstring): the source's
        ``params``, ``report`` and key (as ``source_key`` of the new key), ``reused_from`` (path, sha256 of the file,
        note, the source's ``train_eval``), ``train_eval`` = ``evaluate_idm`` on this root's training events."""
        entry, provenance = self.source_calibration(name)
        params, key = IDMParams.from_dict(entry["params"]), self.calibration_key(name)
        stored, entries = self._stored_calibrations()
        events = self.data.train_events()
        train_eval = evaluate_idm(events, params)
        report = entry.get("report") or {}
        entries[name] = {"params": params.to_dict(), "report": report, "train_eval": train_eval, "key": key,
                         "reused_from": {**provenance, "train_eval_source": entry.get("train_eval")}}  # fmt: skip
        payload = {**stored, "calibrations": entries, "reference": self.idm_reference()}
        if name == DEFAULT_IDM:
            payload.update(params=params.to_dict(), report=report, key=key)
        _write_json_atomic(self.root / "idm.json", payload)
        values = ", ".join(f"{k}={v:.3g}" for k, v in params.to_dict().items())
        self.log.line(
            f"idm {name}: reused from {provenance['path']} (sha256 {provenance['sha256'][:12]}), not fitted: {values}; "
            f"all {train_eval['n_events']:,} training events: one-step RMSE {train_eval['accel_rmse']:.3f} m/s^2, "
            f"closed-loop spacing RMSE {train_eval['spacing_rmse']:.2f} m, {train_eval['n_collisions']} collisions"
        )
        self.calibrations[name] = (params, key)
        return params, key

    def calibration_key(self, name: str) -> dict[str, Any]:
        """Key of the named IDM calibration: training data, fold, the options of ``cfg.idm[name]`` and, for harmonised
        training data, its ``signal_key`` (KeyError when ``name`` is not configured); a pooled domain: the lists of
        its data sets, their folds and ``training_signal_key``. With ``train.idm_from``: ``{"idm_from": <path>,
        "name", "source_key": <the key of the calibration in that file>, "params"}`` (ValueError when the file or the
        calibration is missing), never the key of a fitted calibration."""
        if idm_from_path(self.cfg) is not None:
            entry, provenance = self.source_calibration(name)
            return _jsonable({"idm_from": provenance["path"], "name": name, "source_key": entry.get("key"),
                              "params": entry["params"]})  # fmt: skip
        named = self.cfg.get("idm") or {}
        if name not in named:
            raise KeyError(f"IDM calibration {name!r} is not configured (idm: {list(named)})")
        if self.pooled:  # the lists of a pooled domain: never the key of a calibration on one data set
            names = self.train_names
            return _jsonable({"data": names, "fold": [self.data.fold(n) for n in names], **dict(_plain(named[name])),
                              "signal": training_signal_key(self.cfg)})  # fmt: skip
        data = self.train_names[0]
        sig = self.data.signal_key(data)
        return _jsonable({"data": data, "fold": self.data.fold(data), **dict(_plain(named[name])),
                          **({} if sig is None else {"signal": sig})})  # fmt: skip

    def _stored_calibrations(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Content of ``idm.json`` ({} when absent) and its calibrations by name (old single-calibration files: the
        top-level entry is ``closed_loop``)."""
        path = self.root / "idm.json"
        stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return stored, calibration_entries(stored)

    def stored_calibration(self, name: str) -> tuple[IDMParams, dict[str, Any]] | None:
        """The named calibration of ``idm.json`` with the configured key (read only); None when it is not stored,
        ValueError when it is stored with another key."""
        key = self.calibration_key(name)
        _, entries = self._stored_calibrations()
        if name not in entries:
            return None
        if entries[name].get("key") != key:
            raise ValueError(
                f"IDM calibration {name!r} in {self.root / 'idm.json'} has the key {entries[name].get('key')}, "
                f"the config asks {key}; use another root"
            )
        return IDMParams.from_dict(entries[name]["params"]), key

    def idm_reference(self) -> dict[str, Any] | None:
        """``params`` of the cf-stability calibration file ``idm_reference`` (None if absent)."""
        ref_path = self.cfg.get("idm_reference")
        if ref_path and _path(ref_path).exists():
            return json.loads(_path(ref_path).read_text(encoding="utf-8")).get("params")
        return None

    def spec(self, label: str) -> ModelSpec:
        """``model_spec`` of ``label`` with its IDM calibration (computed on first use)."""
        return model_spec(self.cfg, label, self.calibration)

    # ------------------------------------------------------------------- models
    def check_kwargs(self, directory: Path, spec: ModelSpec) -> None:
        """A stored checkpoint must have the signal variant of the training data (a log without ``signal`` counts as
        raw; a pooled domain: the list of the per-set keys) and its pooled domain (``pool``; a log without it counts
        as one training data set), the configured constructor kwargs, IDM calibration (name and key) and, unless it
        is a sequence model, training target (a log without ``target`` counts as ``"a"``)."""
        log_path = directory / "train_log.json"
        if not log_path.exists():
            self.log.line(f"warning: {log_path} is missing, the kwargs of the checkpoint cannot be checked")
            return
        stored = json.loads(log_path.read_text(encoding="utf-8"))
        if stored.get("signal") != spec.signal:
            raise ValueError(
                f"{directory / 'model.pt'} was trained on {describe_signal(stored.get('signal'))} signals, the config "
                f"asks {describe_signal(spec.signal)}; use another root"
            )
        if stored.get("pool") != spec.pool:
            raise ValueError(
                f"{directory / 'model.pt'} was trained on the training windows of {describe_pool(stored.get('pool'))}, "
                f"the config asks {describe_pool(spec.pool)}; use another root"
            )
        if stored.get("kwargs") != spec.kwargs or stored.get("idm") != spec.idm:
            raise ValueError(
                f"{directory / 'model.pt'} was trained with other kwargs or another IDM calibration than configured; "
                "use another root"
            )
        stored_target = stored.get("target", DEFAULT_TARGET)
        if spec.target is not None and stored_target != spec.target:
            raise ValueError(
                f"{directory / 'model.pt'} was trained for the target {stored_target!r}, the config asks "
                f"{spec.target!r}; use another root"
            )

    def reuse_dir(self, label: str, seed: int) -> Path | None:
        """Checkpoint directory of (label, seed) in another experiment root (``reuse_checkpoints``), else None."""
        reuse = self.cfg.get("reuse_checkpoints") or {}
        if label not in reuse:
            return None
        return _path(str(reuse[label])) / label / f"seed{int(seed)}"

    def reuse_checkpoint(self, spec: ModelSpec, seed: int, source: Path, directory: Path) -> CFModel:
        """Copy the checkpoint of (model, seed) from another experiment root instead of training it.

        The source needs ``model.pt`` and ``train_log.json`` with the configured kwargs, IDM calibration and target
        (``check_kwargs``); the copied ``train_log.json`` records ``reused_from``. Returns the copy loaded.
        """
        for name in ("model.pt", "train_log.json"):
            if not (source / name).exists():
                raise FileNotFoundError(f"reuse_checkpoints: {source / name} does not exist")
        self.check_kwargs(source, spec)
        info = json.loads((source / "train_log.json").read_text(encoding="utf-8"))
        info.update(
            reused_from=_display(source),
            reused_at=_now(),
            reuse_note="checkpoint of another experiment root (config reuse_checkpoints), not trained in this root",
        )
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "train_log.json", info)  # before model.pt: a checkpoint always has its log
        shutil.copyfile(source / "model.pt", directory / "model.pt.tmp")
        (directory / "model.pt.tmp").replace(directory / "model.pt")
        self.trained.append(f"{spec.label}/seed{seed} (reused)")
        self.log.line(f"[{spec.label} seed{seed}] checkpoint reused from {_display(source)} (not trained)")
        return load_model(directory / "model.pt", device=self.device)

    def make_checkpoint(self, spec: ModelSpec, seed: int, directory: Path) -> CFModel:
        """Reuse (``reuse_checkpoints``), copy (seed independent) or train the checkpoint of (model, seed); returns
        it loaded on the device."""
        tag = f"[{spec.label} seed{seed}]"
        source = self.reuse_dir(spec.label, seed)
        if source is not None:
            return self.reuse_checkpoint(spec, seed, source, directory)
        directory.mkdir(parents=True, exist_ok=True)
        if spec.seed_independent:
            sources = sorted(p.parent for p in (self.root / "checkpoints" / spec.label).glob("seed*/model.pt"))
            if sources:
                self.check_kwargs(sources[0], spec)
                info = json.loads((sources[0] / "train_log.json").read_text(encoding="utf-8"))
                write_json(directory / "train_log.json", {**info, "copied_from": sources[0].name})
                shutil.copyfile(sources[0] / "model.pt", directory / "model.pt.tmp")
                (directory / "model.pt.tmp").replace(directory / "model.pt")
                self.trained.append(f"{spec.label}/seed{seed} (copy)")
                self.log.line(f"{tag} seed independent: checkpoint copied from {sources[0].name}")
                return load_model(directory / "model.pt", device=self.device)
        bundle = self.data.bundle(spec.sequences)
        train, val = bundle.windows["train"], bundle.windows["val"]
        self.log.line(f"{tag} training {spec.name} on {len(train):,} windows (val {len(val):,}), device {self.device}")
        seed_everything(seed)
        model = build_model(spec.name, **spec.kwargs).to(self.device)
        started = time.perf_counter()
        fit_log = model.fit(train, val, seed=seed, verbose=bool(self.cfg.get("verbose", False)))
        seconds = time.perf_counter() - started
        info = {
            "model": spec.label,
            "registry_name": spec.name,
            "seed": seed,
            "device": str(self.device),
            "kwargs": spec.kwargs,
            "idm": spec.idm,
            "target": spec.target,
            **({} if spec.signal is None else {"signal": spec.signal}),  # absent: raw signals (logs written before)
            **({} if spec.pool is None else {"pool": spec.pool, "pool_windows": (bundle.meta or {}).get("pool")}),
            "n_train": len(train),
            "n_val": len(val),
            "seconds": seconds,
            "finished": _now(),
            "log": fit_log,
        }
        write_json(directory / "train_log.json", info)  # before model.pt: a checkpoint always has its log
        model.save(directory / "model.pt.tmp")
        (directory / "model.pt.tmp").replace(directory / "model.pt")
        self.trained.append(f"{spec.label}/seed{seed}")
        self.log.line(f"{tag} trained in {seconds:.1f} s ({_fit_summary(fit_log)})")
        del model
        self.cleanup()
        # predictions always come from the stored checkpoint, so that a regenerated table is identical
        return load_model(directory / "model.pt", device=self.device)

    def predict(self, model: CFModel, spec: ModelSpec, seed: int, dataset: str) -> dict[str, Any]:
        """Write ``preds/<model>/seed<k>/<dataset>.parquet``; returns its timing."""
        ev = self.data.eval_set(dataset)
        self.check_windows(dataset, ev)
        ecfg = self.cfg.eval
        rows = np.arange(len(ev.windows))
        if model.needs_leader_future:
            max_rows = ecfg.get("max_rows_sampled")
            rows = sampled_rows(
                ev.windows.event_index, ev.windows.t_index, ev.lengths,
                None if max_rows is None else int(max_rows), int(ecfg.get("sample_seed", 0)),
            )  # fmt: skip
        seed_everything(seed)
        started = time.perf_counter()
        frame = predict_table(
            model, ev, rows, seed=seed, chunk_rows=int(ecfg.chunk_rows), batch_size=int(ecfg.get("batch_size", 8192)),
            coverage=float(ecfg.coverage), target=self.target,
        )  # fmt: skip
        seconds = time.perf_counter() - started
        tables.save_preds(self.root, spec.label, seed, dataset, frame)
        self.written.append(f"{spec.label}/seed{seed}/{dataset}")
        quick = quick_metrics(frame, ev.windows.y)
        nll = "-" if quick["nll"] is None else f"{quick['nll']:.4f}"
        picp = "-" if quick["picp"] is None else f"{quick['picp']:.3f}"
        self.log.line(
            f"[{spec.label} seed{seed}] {dataset}: {rows.size:,} of {len(ev.windows):,} rows in {seconds:.1f} s "
            f"(NLL {nll}, PICP {picp})"
        )
        return {"predict_seconds": seconds, "n_predicted": int(rows.size), "device": str(self.device)}

    def assemble_pool(self, spec: ModelSpec, seed: int, dataset: str) -> dict[str, Any]:
        """Write the prediction table of the pool ``dataset`` (design.md 16.1): the members' tables of (model, seed)
        concatenated in the order of the pool, the row order of its windows table. Every member table must exist with
        the row count of its windows table (FileNotFoundError / RuntimeError otherwise: the unit is retried on resume
        once the member is written). Returns its timing."""
        started = time.perf_counter()
        self.check_windows(dataset, self.data.eval_set(dataset))
        frames = []
        for m in self.pools[dataset]:
            path = tables.preds_path(self.root, spec.label, seed, m)
            if not path.exists():
                raise FileNotFoundError(f"pool {dataset}: the table of its member {m} is missing ({path})")
            self.check_windows(m, self.data.eval_set(m))
            frame = pd.read_parquet(path)
            if len(frame) != len(self.data.eval_set(m).windows):
                raise RuntimeError(f"pool {dataset}: preds of {m} have {len(frame)} rows, its windows table "
                                   f"{len(self.data.eval_set(m).windows)}")  # fmt: skip
            frames.append(frame)
        pool = pd.concat(frames, ignore_index=True)[list(tables.PRED_COLUMNS)]
        ev = self.data.eval_set(dataset)
        if len(pool) != len(ev.windows):
            raise RuntimeError(f"pool {dataset}: {len(pool)} rows from its members, its windows table {len(ev.windows)}")
        tables.save_preds(self.root, spec.label, seed, dataset, pool)
        seconds = time.perf_counter() - started
        self.written.append(f"{spec.label}/seed{seed}/{dataset}")
        quick = quick_metrics(pool, ev.windows.y)
        nll = "-" if quick["nll"] is None else f"{quick['nll']:.4f}"
        picp = "-" if quick["picp"] is None else f"{quick['picp']:.3f}"
        self.log.line(f"[{spec.label} seed{seed}] {dataset}: {len(pool):,} rows assembled from "
                      f"{', '.join(self.pools[dataset])} in {seconds:.1f} s (NLL {nll}, PICP {picp})")  # fmt: skip
        return {"assembled_from": list(self.pools[dataset]), "assemble_seconds": seconds,
                "n_predicted": int(quick["n_evaluated"])}  # fmt: skip

    def write_metrics(self, spec: ModelSpec, seed: int, directory: Path, timings: Mapping[str, Any]) -> None:
        """``metrics.json``: quick-look means of every stored table of (model, seed) and their timings."""
        path = directory / "metrics.json"
        old = json.loads(path.read_text(encoding="utf-8")).get("datasets", {}) if path.exists() else {}
        out: dict[str, Any] = {"model": spec.label, "registry_name": spec.name, "seed": seed, "datasets": {}}
        stored = tables.available(self.root)
        for row in stored[(stored["model"] == spec.label) & (stored["seed"] == seed)].itertuples():
            dataset = str(row.dataset)
            y = pd.read_parquet(tables.windows_path(self.root, dataset), columns=["y"])["y"].to_numpy()
            entry = quick_metrics(pd.read_parquet(row.path), y)
            timing = timings.get(dataset) or old.get(dataset, {}).get("timing")
            if timing:
                entry["timing"] = timing
            out["datasets"][dataset] = entry
        directory.mkdir(parents=True, exist_ok=True)
        write_json(path, out)

    def run_unit(self, spec: ModelSpec, seed: int) -> None:
        """Checkpoint (load / train) and missing tables of one (model, seed)."""
        unit = f"{spec.label}/seed{seed}"
        directory = checkpoint_dir(self.root, spec.label, seed)
        model: CFModel | None = None
        self.current = unit
        if (directory / "model.pt").exists():
            self.check_kwargs(directory, spec)
        else:
            model = self.make_checkpoint(spec, seed, directory)
            self.write_status()
        timings: dict[str, Any] = {}
        for dataset in self.ready:
            if tables.preds_path(self.root, spec.label, seed, dataset).exists():
                continue
            self.current = f"{unit}/{dataset}"
            try:
                if dataset in self.pools:  # the union of the members' tables, no prediction of its own
                    timings[dataset] = self.assemble_pool(spec, seed, dataset)
                    continue
                if model is None:
                    model = load_model(directory / "model.pt", device=self.device)
                timings[dataset] = self.predict(model, spec, seed, dataset)
            except Exception as exc:  # noqa: BLE001 - the other data sets of this model still run
                self.log.error(f"{unit}/{dataset}", exc)
            finally:
                self.write_status()
        if timings or not (directory / "metrics.json").exists():
            self.write_metrics(spec, seed, directory, timings)

    # ---------------------------------------------------------------------- run
    def pool_line(self) -> str:
        """Log line of a pooled training domain (design.md 16.1): data sets, caps, pool tables."""
        pools = "; ".join(f"{d} = union of {', '.join(m)}" for d, m in self.pools.items()) or "none"
        return f"training domain: {describe_pool(pool_key(self.cfg))}; pool tables: {pools}"

    def matched_line(self) -> str | None:
        """Log line of the matched windows (signal matched), else None."""
        if not self.windows:
            return None
        return f"matched windows ({_display(matching_path(self.cfg))}): " + ", ".join(f"{k} {v}" for k, v in self.windows.items())

    def run(self) -> dict[str, Any]:
        status = self.status()
        self.log.line(
            f"E1 run: root {self.root}, target {self.target}, signal {self.signal}, device {self.device}; {len(self.models)} models x "
            f"{len(self.seeds)} seeds x {len(self.datasets)} data sets; "
            f"checkpoints {status['checkpoints']['done']}/{status['checkpoints']['total']}, "
            f"tables {status['tables']['done']}/{status['tables']['total']} already done"
        )
        if self.matched_line():
            self.log.line(str(self.matched_line()))
        if self.pooled:
            self.log.line(self.pool_line())
        self.write_status()
        finished = False
        try:
            for dataset in self.datasets:
                self.guard(f"windows/{dataset}", self.ensure_windows, dataset)
            self.guard("train_density", self.ensure_density)
            for name in self.source_names():  # train.idm_from: every calibration of the file is copied, none fitted
                self.guard(f"idm/{name}", self.calibration, name)
            # every model config first: the IDM calibrations the selected models need are computed here
            specs = {label: self.guard(f"{label}/config", self.spec, label) for label in self.models}
            for label, spec in specs.items():
                for seed in self.seeds if spec is not None else []:
                    self.guard(f"{label}/seed{seed}", self.run_unit, spec, seed)
                    self.cleanup()
                    self.write_status()
            finished = True
        finally:
            status = self.write_status(finished=finished)
        self.log.line(
            f"{'finished' if finished else 'stopped'} after {status['elapsed_s']:.0f} s: checkpoints "
            f"{status['checkpoints']['done']}/{status['checkpoints']['total']} ({len(self.trained)} new), tables "
            f"{status['tables']['done']}/{status['tables']['total']} ({len(self.written)} new), "
            f"{len(self.log.failures)} failure(s)"
        )
        for failure in self.log.failures:
            self.log.line(f"  failed: {failure['unit']}: {failure['error']}")
        return status

    # ------------------------------------------------------------------ dry run
    def reuse_spec(self, label: str, source: Path) -> ModelSpec:
        """``model_spec`` of ``label`` for the dry-run check of a reuse source while an IDM calibration that the model
        needs is not stored in this root yet: that calibration is taken from the source checkpoint's log when it was
        computed with the configured key (a run computes it again with the same settings on the same training
        events, deterministically), else ValueError. Stored calibrations are used as stored."""
        log = json.loads((source / "train_log.json").read_text(encoding="utf-8"))

        def calibration(name: str) -> tuple[IDMParams, dict[str, Any]]:
            found = self.stored_calibration(name)
            if found is not None:
                return found
            if idm_from_path(self.cfg) is not None:  # the calibration a run copies (train.idm_from)
                return self.reused_calibration(name)
            key, ref = self.calibration_key(name), log.get("idm") or {}
            if ref.get("name") != name or ref.get("key") != key or "idm" not in (log.get("kwargs") or {}):
                raise ValueError(f"{source / 'model.pt'} was calibrated with {ref or 'no IDM'}, the config asks the IDM "
                                 f"calibration {name!r} with the key {key}; use another source")  # fmt: skip
            return IDMParams.from_dict(log["kwargs"]["idm"]), key

        return model_spec(self.cfg, label, calibration)

    def dry_run(self) -> dict[str, Any]:
        """What ``run`` would do, without loading events, training, predicting or writing anything (no log file, no
        status, no directory): every stored setting is compared with the config as ``run`` compares it (windows
        specs, training density, IDM calibration keys, checkpoint logs with kwargs, IDM, target and signal), and every
        missing checkpoint with a ``reuse_checkpoints`` source is checked against that source as ``run`` checks it
        (signal of the training data set, kwargs, IDM calibration, target); an IDM calibration that this root does not
        store yet is taken from the source when its key is the configured one (``reuse_spec``).

        Returns ``root``, ``target``, ``signal``, ``matched_windows`` (signal matched, else None), per kind
        (``windows``, ``train_density``, ``idm``, ``checkpoints``, ``tables``) the lists ``done`` and ``missing``
        (checkpoints also ``unchecked``: their IDM calibration is not stored yet, and ``reuse``: missing here, the
        source passes the checks, a run copies it; ``missing`` checkpoints are trained), ``reuse`` (one decision per
        model and source: ``model``, ``source``, ``seeds``, ``idm``), ``conflicts`` (unit and message: settings that
        differ or a reuse source that fails the checks, a run would refuse these units) and ``nothing_to_do``.

        Also ``evaluated`` (per evaluated data set its data, split and stride, a pool its members) and ``keys`` (the
        signal key of the training data, ``describe_signal``, and ``pool_key``). A pooled training domain (design.md
        16.1): ``training`` = ``Data.training_plan`` (the window counts per data set after the caps), which loads the
        events of the training data sets (still nothing is written).
        """
        out: dict[str, Any] = {"root": str(self.root), "target": self.target, "signal": self.signal,
                               "matched_windows": self.windows}  # fmt: skip
        out["evaluated"] = {
            d: ({"pool": list(self.pools[d])} if d in self.pools else
                {k: self.cfg.eval.datasets[d].get(k, v) for k, v in (("data", None), ("split", "all"), ("stride", 1))})
            for d in self.datasets
        }  # fmt: skip
        out["keys"] = {"signal": describe_signal(training_signal_key(self.cfg)), "pool": pool_key(self.cfg)}
        if self.pooled:
            out["training"] = self.data.training_plan()
        conflicts: list[dict[str, str]] = []
        kinds: dict[str, dict[str, list[str]]] = {
            k: {"done": [], "missing": []} for k in ("windows", "train_density", "idm", "checkpoints", "tables")
        }
        kinds["checkpoints"]["unchecked"] = []
        kinds["checkpoints"]["reuse"] = []
        kinds["idm"]["reuse"] = []  # calibrations a run copies from train.idm_from (never fitted)
        idm_source = idm_from_path(self.cfg)
        out["idm_from"] = None if idm_source is None else _display(idm_source)
        decisions: list[dict[str, Any]] = []

        def check(kind: str, unit: str, fn: Callable[[], bool]) -> bool | None:
            try:
                ok = bool(fn())
            except (ValueError, KeyError) as exc:
                conflicts.append({"unit": unit, "error": f"{type(exc).__name__}: {exc}"})
                return None
            kinds[kind]["done" if ok else "missing"].append(unit)
            return ok

        for dataset in self.datasets:
            check("windows", f"windows/{dataset}", lambda d=dataset: self.stored_windows(d))
        check("train_density", "train_density", self.stored_density)
        needed: dict[str, None] = dict.fromkeys(self.source_names())  # train.idm_from: all of them are copied
        for label in self.models:
            mcfg = model_config(self.cfg, label)
            if mcfg.get("needs_idm", False):
                needed[str(mcfg.get("idm", DEFAULT_IDM))] = None
        for name in needed:
            stored = check("idm", f"idm/{name}", lambda n=name: self.stored_calibration(n) is not None)
            if stored is False and idm_source is not None:  # copied by the run, not fitted
                kinds["idm"]["missing"].remove(f"idm/{name}")
                try:
                    self.source_calibration(name)
                except ValueError as exc:
                    conflicts.append({"unit": f"idm/{name}", "error": f"{type(exc).__name__}: {exc}"})
                else:
                    kinds["idm"]["reuse"].append(f"idm/{name}")

        def stored_only(name: str) -> tuple[IDMParams, dict[str, Any]]:
            found = self.stored_calibration(name)
            if found is None and idm_source is not None:
                return self.reused_calibration(name)  # what the run copies: the specs can be checked
            if found is None:
                raise _NotStored(name)
            return found

        for label in self.models:
            try:
                spec: ModelSpec | None = model_spec(self.cfg, label, stored_only)
            except _NotStored:  # its IDM calibration is not stored yet: the kwargs cannot be compared
                spec = None
            except (ValueError, KeyError) as exc:
                conflicts.append({"unit": f"{label}/config", "error": f"{type(exc).__name__}: {exc}"})
                continue
            accepted: dict[str, list[int]] = {}  # checkpoints root of the source -> seeds whose reuse passes the checks
            for seed in self.seeds:
                unit = f"{label}/seed{seed}"
                directory = checkpoint_dir(self.root, label, seed)
                source = self.reuse_dir(label, seed)
                if (directory / "model.pt").exists():
                    if spec is None:
                        kinds["checkpoints"]["unchecked"].append(unit)
                    else:
                        check("checkpoints", unit, lambda d=directory, s=spec: (self.check_kwargs(d, s), True)[1])
                elif source is None:
                    kinds["checkpoints"]["missing"].append(unit)  # trained by the run
                else:
                    try:
                        for name in ("model.pt", "train_log.json"):
                            if not (source / name).exists():
                                raise FileNotFoundError(f"reuse_checkpoints: {source / name} does not exist")
                        self.check_kwargs(source, spec if spec is not None else self.reuse_spec(label, source))
                    except (ValueError, KeyError, FileNotFoundError) as exc:
                        conflicts.append({"unit": f"{unit} (reuse)", "error": f"{type(exc).__name__}: {exc}"})
                    else:
                        kinds["checkpoints"]["reuse"].append(unit)
                        accepted.setdefault(_display(source.parent.parent), []).append(seed)
                for dataset in self.datasets:
                    done = tables.preds_path(self.root, label, seed, dataset).exists()
                    kinds["tables"]["done" if done else "missing"].append(f"{unit}/{dataset}")
            mcfg = model_config(self.cfg, label)
            idm = None
            if mcfg.get("needs_idm", False):
                name = str(mcfg.get("idm", DEFAULT_IDM))
                idm = (f"{name}: stored in this root" if spec is not None else
                       f"{name}: not stored in this root yet, checked with the source's calibration of the same key "
                       "(the run computes it again)")  # fmt: skip
            decisions += [{"model": label, "source": src, "seeds": seeds, "idm": idm} for src, seeds in accepted.items()]
        out.update(kinds)
        out["reuse"] = decisions
        out["conflicts"] = conflicts
        out["nothing_to_do"] = not conflicts and not any(v["missing"] for v in kinds.values()) and not kinds[
            "checkpoints"]["unchecked"] and not kinds["checkpoints"]["reuse"] and not kinds["idm"]["reuse"]  # fmt: skip

        def count(kind: str, v: dict[str, list[str]]) -> str:
            text = f"{kind} {len(v['done'])}/{sum(len(x) for x in v.values())}"
            if kind == "idm" and idm_source is not None:
                text += f" (to copy {len(v['reuse'])}, to fit {len(v['missing'])})"
            if kind == "checkpoints":
                text += f" (to reuse {len(v['reuse'])}, to train {len(v['missing'])}" + (
                    f", unchecked {len(v['unchecked'])})" if v["unchecked"] else ")")
            return text

        counts = ", ".join(count(k, v) for k, v in kinds.items())
        self.log.line(
            f"E1 dry run: root {self.root}, target {self.target}, signal {self.signal}; done: {counts}; "
            f"{len(conflicts)} conflict(s){'; nothing to do' if out['nothing_to_do'] else ''}"
        )
        if self.matched_line():
            self.log.line(f"  {self.matched_line()}")
        self.log.line("  evaluated: " + ", ".join(
            f"{d} (pool of {', '.join(v['pool'])})" if "pool" in v else f"{d} ({v['data']}, {v['split']}, stride {v['stride']})"
            for d, v in out["evaluated"].items()))  # fmt: skip
        self.log.line(f"  training signal key: {out['keys']['signal']}")
        if idm_source is not None:
            names = [n.split("/", 1)[1] for n in (*kinds["idm"]["done"], *kinds["idm"]["reuse"])]
            sha = ""
            if idm_source.is_file():
                sha = f", sha256 {hashlib.sha256(idm_source.read_bytes()).hexdigest()[:12]}"
            self.log.line(f"  IDM: reused from {out['idm_from']} ({', '.join(names) or 'no calibration needed'}{sha}; "
                          f"copied {len(kinds['idm']['done'])}, to copy {len(kinds['idm']['reuse'])}); "
                          f"{len(kinds['idm']['missing'])} calibrations to fit")  # fmt: skip
        if self.pooled:
            self.log.line(f"  {self.pool_line()}")
            for kind in ("one_step", "sequences"):
                plan = out["training"][kind]
                per = "; ".join(f"{n} " + ", ".join(f"{p} {c[p]['windows']:,} -> {c[p]['kept']:,}" for p in POOL_SPLITS)
                                for n, c in plan["datasets"].items())  # fmt: skip
                self.log.line(f"  {kind.replace('_', '-')} windows after the caps (stride {plan['stride']}, cap "
                              f"{plan['cap'] if plan['cap'] is not None else 'none'}): {per}; total train "
                              f"{plan['total']['train']:,}, val {plan['total']['val']:,}")  # fmt: skip
        for d in decisions:
            idm_note = "" if d["idm"] is None else f"; IDM {d['idm']}"
            self.log.line(f"  reuse {d['model']} seeds {_seed_list(d['seeds'])} from {d['source']}: checks passed "
                          f"(signal, kwargs, IDM, target){idm_note}")  # fmt: skip
        for c in conflicts:
            self.log.line(f"  conflict: {c['unit']}: {c['error']}")
        return out


def run_e1(cfg: DictConfig | Mapping[str, Any], echo: bool = True) -> dict[str, Any]:
    """Run or resume experiment E1 (module docstring); returns the final status (also in ``status.json``)."""
    cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(dict(cfg))
    configure_torch(bool(cfg.get("tf32", False)))
    return Runner(cfg, echo=echo).run()


def dry_run_e1(cfg: DictConfig | Mapping[str, Any], echo: bool = True) -> dict[str, Any]:
    """``Runner.dry_run`` of the config: what a run would do and which stored settings conflict; writes nothing."""
    cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(dict(cfg))
    return Runner(cfg, echo=echo, dry=True).dry_run()
