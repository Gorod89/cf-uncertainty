"""Experiment E3 (docs/design.md, section 13): platoon-level distributional evaluation of the E1 models.

``run_e3(cfg)`` (config ``configs/experiment/e3.yaml``) writes under ``cfg.root``::

    config.yaml                                              resolved config of the last call (read by the analysis)
    spec.json                                                settings that define the units (checked on resume)
    profiles__<variant>.yaml                                 re-smoothed profiles of a variant with ``profile_window``
    units/<variant>/<model>/spec.json                        checkpoint, sampler and hold of a model (checked on resume)
    units/<variant>/<model>/seed<k>/<profile>.parquet        one row per (vehicle 1..V, rollout)  (UNIT_COLUMNS)
    units/<variant>/<model>/seed<k>/<profile>.npz            mean speed path per vehicle over the response window
    logs/run.log, logs/errors.log, logs/timings.jsonl, status.json

A *unit* is (signal variant, model, seed, profile). Signal variants: ``raw`` (``cfg.checkpoints``) and every entry of
``cfg.variants``: target -> checkpoint directory, and two optional keys (E3-pooled, design.md 17.1):

* ``models``: per-label overrides of the model spec (``target``, ``hold_steps``, ``sampler_mode``, ``model``) for this
  variant only, e.g. ``{quantile_pidl: {target: a, hold_steps: 1}}`` for checkpoint roots without the target
  ``mean_1s``; every label must be a key of ``cfg.models``.
* ``profile_window``: the leader speed of every profile is smoothed again from the raw OpenACC file of the profile
  (``source`` of the profiles file; same file, start sample and length, the selection is not repeated) with this
  Savitzky-Golay window and the order of the selection (``cf_unc.platoon.profiles.resmooth_profiles``). The window is
  a number of samples or ``{matching: <matching file>, data: <data set>}`` (the window of that data set in the file,
  e.g. 61 for OpenACC in ``artifacts/e1m/matching.json``). The re-smoothed profiles are written with their provenance
  to ``profiles__<variant>.yaml`` on the first run and recomputed and compared on every later one (a difference
  refuses the variant); the analysis reads the leader of a variant from that file. The measured follower statistics
  stay those of the profiles file.

* ``only_models``: the labels of ``cfg.models`` that run in this variant (absent: all of them), e.g. the review control
  ``raw_a`` (design.md 18.2), whose other models would repeat the ``raw`` units bit for bit.

Model specs (``cfg.models``, design.md 13.2 and 18): ``model``, ``target``, ``sampler_mode``, ``hold_steps`` and two
optional keys (review controls, design.md 18):

* ``checkpoints``: a checkpoints directory of this label that replaces the variant's directory of its target (e.g.
  ``diffusion_cf_hist`` of ``artifacts/e1_ctrl/checkpoints`` in the ``raw`` variant); part of the unit spec as the
  variant's directory always was, so the specs of the models without it are unchanged. A variant's ``models``
  override may set it too.
* ``variants``: the variants this label runs in (absent: every variant), e.g. a control model whose checkpoints exist
  for some variants only. Not part of the unit spec.

A (variant, target) whose checkpoint directory does not exist is skipped with a warning (the targets of a variant are
those of its model specs after the overrides). Profiles come from ``cfg.profiles``; ``cfg.run_profiles`` selects
``headline`` (the first ``sets.headline`` profiles), ``extended`` (all, or the first ``sets.extended``) or a list of
profile ids.

Simulation (``cf_unc.platoon.simulate``): ``platoon.n_followers`` vehicles behind every leader profile,
``platoon.n_rollouts`` rollouts (``n_rollouts_sampled`` for plan samplers), sampler mode and hold of the model spec
of the variant; initial gap = equilibrium gap of the closed-loop IDM of ``cfg.idm`` (one IDM for every variant,
design.md 13.4) at the first leader speed of the variant's profile, capped at ``platoon.gap_cap``. The profiles of the
run set are cut, in file order, into batches of at most ``platoon.batch_profiles`` profiles (and at most
``max_batch / n_rollouts``), simulated together on a common time line of ``K = B + n_max - 1 + T`` steps (``n_max`` =
the longest profile of the file). A batch has one random stream per vehicle (one model call per step for the whole
batch), seeded from ``(platoon.seed, model seed, ids of the batch, vehicle)``: the same for every model and variant
(common random numbers). A batch with a missing unit is simulated whole and only its missing units are written, so a
unit depends on (config, run set, batch size) only and a resumed or regenerated unit is identical to the original;
a call with another run set (e.g. ``headline`` only) therefore belongs in another root.

Resumable at the level of units (a unit is complete when its parquet table exists; the npz is written first); a
failing (variant, model, seed) is logged with its traceback to ``logs/errors.log`` and the run continues;
``run_e3`` returns the final status (``failures``). ``dry_run_e3(cfg)``: what a run would do (units done and missing
per variant and model, stored specs and profile files compared with the config), writing nothing.
"""

from __future__ import annotations

import gc
import json
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from cf_unc.constants import DT, HIST_LEN
from cf_unc.data.harmonise import read_matching
from cf_unc.experiments.e1 import RunLog, _jsonable, _write_json_atomic, configure_torch, seed_everything
from cf_unc.models import CFModel, load_model, resolve_device
from cf_unc.physics.idm import IDMParams
from cf_unc.platoon.profiles import Profile, load_profiles, profiles_digest, resmooth_profiles, write_profiles
from cf_unc.platoon.simulate import PlatoonConfig, PlatoonResult, initial_gap, simulate_platoon, time_line_steps
from cf_unc.train.run import ROOT, _path, _plain

__all__ = [
    "DEFAULT_CONFIG",
    "MODEL_KEYS",
    "PILOT",
    "UNIT_COLUMNS",
    "E3Runner",
    "Variant",
    "available_units",
    "dry_run_e3",
    "load_config",
    "load_unit",
    "parse_variants",
    "profile_window",
    "run_e3",
    "select_profile_ids",
    "stream_seeds",
    "unit_path",
    "variant_profiles_path",
]

DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "e3.yaml"
PILOT: dict[str, Any] = {"root": "artifacts/e3_pilot", "seeds": [0], "run_profiles": "headline", "n_rollouts": 20}
RAW = "raw"
MODEL_KEYS = ("model", "target", "sampler_mode", "hold_steps", "checkpoints")  # keys of a model spec and of a variant's override
SELECT_KEYS = ("variants",)  # keys of a model spec that select where it runs (not part of the unit spec)
VARIANT_KEYS = ("models", "profile_window", "only_models")  # keys of a variant that are not targets
UNIT_COLUMNS = (
    "profile", "seed", "vehicle", "rollout", "amplitude", "sd", "amplitude_profile", "sd_profile", "min_gap",
    "first_collision", "leader_amplitude", "leader_sd", "leader_sd_profile", "emp_amplitude", "emp_sd", "s_init",
)  # fmt: skip


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _crc(text: str) -> int:
    return zlib.crc32(str(text).encode("utf-8"))


# ------------------------------------------------------------------------------------------ config
def load_config(path: str | Path = DEFAULT_CONFIG, overrides: Sequence[str] = (), pilot: bool = False) -> DictConfig:
    """Experiment config: the YAML file, then ``PILOT`` (root ``artifacts/e3_pilot``, seed 0, headline profiles, every
    model, 20 rollouts), then dotted ``key=value`` overrides."""
    cfg = OmegaConf.load(_path(path))
    if pilot:
        cfg.root = PILOT["root"]
        cfg.seeds = list(PILOT["seeds"])
        cfg.run_profiles = PILOT["run_profiles"]
        cfg.platoon.n_rollouts = int(PILOT["n_rollouts"])
        cfg.platoon.n_rollouts_sampled = int(PILOT["n_rollouts"])
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([str(o) for o in overrides]))
    return cfg  # type: ignore[return-value]


def select_profile_ids(profiles: Sequence[Profile], which: Any, sets: Mapping[str, Any] | None = None) -> list[str]:
    """Profile ids in file order: ``headline`` / ``extended`` (first ``sets[name]`` profiles, None = all) or an
    explicit list of ids (unknown ids are an error)."""
    ids = [p.id for p in profiles]
    sets = dict(sets or {"headline": 10, "extended": None})
    if isinstance(which, str) and which in sets:
        n = sets[which]
        return ids if n is None else ids[: int(n)]
    wanted = [str(which)] if isinstance(which, str) else [str(w) for w in which]
    unknown = sorted(set(wanted) - set(ids))
    if unknown:
        raise ValueError(f"unknown profiles {unknown}; the profiles file has {ids[0]} .. {ids[-1]}")
    return [i for i in ids if i in set(wanted)]


def stream_seeds(base: int, model_seed: int, profile_ids: Sequence[str], n_vehicles: int) -> list[int]:
    """63-bit stream seeds of vehicles 1..V of a batch of profiles (the same for every model and variant)."""
    batch = _crc(",".join(str(i) for i in profile_ids))
    out = []
    for i in range(1, int(n_vehicles) + 1):
        state = np.random.SeedSequence([abs(int(base)), abs(int(model_seed)), batch, i]).generate_state(1, np.uint64)
        out.append(int(state[0] >> np.uint64(1)))
    return out


# ------------------------------------------------------------------------------------------ variants
@dataclass
class Variant:
    """One signal variant (module docstring): checkpoint directory per target, per-label overrides of the model
    specs, and the Savitzky-Golay window of the leader re-smoothing (None: the profiles file as it is)."""

    name: str
    checkpoints: dict[str, str]
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    profile_window: int | None = None
    window_source: dict[str, str] | None = None  # {"matching", "data"} when the window was read from a matching file
    only_models: list[str] | None = None  # labels that run in this variant (None: every label)


def profile_window(value: Any) -> tuple[int, dict[str, str] | None]:
    """(window, source) of a variant's ``profile_window``: a number of samples, or ``{matching: <file>, data: <name>}``
    = the window of that data set in the matching file of E5 (``windows``). ValueError for anything else."""
    source = None
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - {"matching", "data"})
        if unknown or not value.get("matching") or not value.get("data"):
            raise ValueError(f"profile_window: expected {{matching: <file>, data: <name>}}, got {dict(value)}")
        windows = read_matching(_path(str(value["matching"]))).get("windows") or {}
        if str(value["data"]) not in windows:
            raise ValueError(f"profile_window: {value['matching']} has no window for {value['data']!r} ({sorted(windows)})")
        source = {"matching": str(value["matching"]), "data": str(value["data"])}
        value = windows[str(value["data"])]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 3 or int(value) % 2 == 0:
        raise ValueError(f"profile_window must be an odd number of samples >= 3, got {value!r}")
    return int(value), source


def check_models(cfg: DictConfig | Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The model specs of ``cfg.models``; ValueError for a key outside ``MODEL_KEYS`` + ``SELECT_KEYS`` or a
    ``variants`` entry that is not a list of names."""
    models = {str(k): dict(v or {}) for k, v in (_plain(cfg.get("models")) or {}).items()}
    for label, spec in models.items():
        bad = sorted(set(spec) - set(MODEL_KEYS) - set(SELECT_KEYS))
        if bad:
            raise ValueError(f"models.{label}: unknown keys {bad}; known: {[*MODEL_KEYS, *SELECT_KEYS]}")
        where = spec.get("variants")
        if where is not None and (isinstance(where, str) or not all(isinstance(v, str) for v in where)):
            raise ValueError(f"models.{label}.variants must be a list of variant names, got {where!r}")
    return models


def parse_variants(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Variant]:
    """The signal variants of an E3 config: ``raw`` (``checkpoints``) first, then ``variants`` in config order. A
    variant's ``models`` may only override ``MODEL_KEYS`` of labels of ``cfg.models``, its ``only_models`` may only
    list such labels (ValueError otherwise)."""
    labels = set(check_models(cfg))
    out = {RAW: Variant(RAW, {str(k): str(v) for k, v in (_plain(cfg.get("checkpoints")) or {}).items() if v is not None})}
    for name, spec in (_plain(cfg.get("variants")) or {}).items():
        spec = dict(spec or {})
        if str(name) == RAW:
            raise ValueError(f"variant name {RAW!r} is reserved for cfg.checkpoints")
        overrides = {str(k): dict(v or {}) for k, v in (spec.pop("models", None) or {}).items()}
        for label, override in overrides.items():
            if label not in labels:
                raise ValueError(f"variant {name}: models overrides {label!r}, which is not a key of models {sorted(labels)}")
            bad = sorted(set(override) - set(MODEL_KEYS))
            if bad:
                raise ValueError(f"variant {name}: models.{label} may override {list(MODEL_KEYS)} only, got {bad}")
        window = spec.pop("profile_window", None)
        w, source = (None, None) if window is None else profile_window(window)
        only = spec.pop("only_models", None)
        if only is not None:
            only = [str(m) for m in ([only] if isinstance(only, str) else only)]
            unknown = sorted(set(only) - labels)
            if unknown:
                raise ValueError(f"variant {name}: only_models lists {unknown}, which are not keys of models {sorted(labels)}")
        dirs = {str(k): str(v) for k, v in spec.items() if v is not None}
        out[str(name)] = Variant(str(name), dirs, overrides, w, source, only)
    return out


def variant_profiles_path(root: str | Path, variant: str) -> Path:
    """The re-smoothed profiles of a variant with ``profile_window`` (written by the runner, read by the analysis)."""
    return Path(root) / f"profiles__{variant}.yaml"


# ------------------------------------------------------------------------------------------ units
def unit_path(root: str | Path, variant: str, model: str, seed: int, profile: str, suffix: str = ".parquet") -> Path:
    return Path(root) / "units" / variant / model / f"seed{int(seed)}" / f"{profile}{suffix}"


def available_units(root: str | Path) -> pd.DataFrame:
    """One row per stored unit table: variant, model, seed, profile, path."""
    rows = [
        {"variant": p.parts[-4], "model": p.parts[-3], "seed": int(p.parts[-2][4:]), "profile": p.stem, "path": str(p)}
        for p in sorted((Path(root) / "units").glob("*/*/seed*/*.parquet"))
    ]
    return pd.DataFrame(rows, columns=["variant", "model", "seed", "profile", "path"])


def load_unit(root: str | Path, variant: str, model: str, seed: int, profile: str) -> pd.DataFrame:
    return pd.read_parquet(unit_path(root, variant, model, seed, profile))


def unit_table(result: PlatoonResult, e: int, profile: Profile, seed: int) -> pd.DataFrame:
    """UNIT_COLUMNS of profile ``e`` of a platoon result: one row per (vehicle, rollout), vehicle-major."""
    n_v, n_r = result.amplitude.shape[1:]
    vehicle = np.repeat(np.arange(1, n_v + 1), n_r)
    emp_amp = np.full(n_v, np.nan)
    emp_sd = np.full(n_v, np.nan)
    m = min(n_v, int(profile.n_followers))
    emp_amp[:m], emp_sd[:m] = profile.follower_amplitude[:m], profile.follower_sd[:m]
    f32 = np.float32
    frame = pd.DataFrame({
        "profile": pd.Categorical([profile.id] * (n_v * n_r)),
        "seed": np.full(n_v * n_r, seed, dtype=np.int16),
        "vehicle": vehicle.astype(np.int16),
        "rollout": np.tile(np.arange(n_r), n_v).astype(np.int16),
        "amplitude": result.amplitude[e].reshape(-1).astype(f32),
        "sd": result.sd[e].reshape(-1).astype(f32),
        "amplitude_profile": result.amplitude_profile[e].reshape(-1).astype(f32),
        "sd_profile": result.sd_profile[e].reshape(-1).astype(f32),
        "min_gap": result.min_gap[e].reshape(-1).astype(f32),
        "first_collision": result.first_collision[e].reshape(-1).astype(np.int32),
        "leader_amplitude": np.full(n_v * n_r, result.leader_amplitude[e], dtype=f32),
        "leader_sd": np.full(n_v * n_r, result.leader_sd[e], dtype=f32),
        "leader_sd_profile": np.full(n_v * n_r, result.leader_sd_profile[e], dtype=f32),
        "emp_amplitude": np.repeat(emp_amp, n_r).astype(f32),
        "emp_sd": np.repeat(emp_sd, n_r).astype(f32),
        "s_init": np.full(n_v * n_r, result.s_init[e], dtype=f32),
    })  # fmt: skip
    return frame[list(UNIT_COLUMNS)]


def _write_unit(root: Path, variant: str, model: str, seed: int, profile: Profile, frame: pd.DataFrame,
                paths: dict[str, np.ndarray]) -> Path:  # fmt: skip
    """npz first, then the parquet table (atomic replace each): a present table means a complete unit."""
    target = unit_path(root, variant, model, seed, profile.id)
    target.parent.mkdir(parents=True, exist_ok=True)
    npz = target.with_suffix(".npz")
    tmp = npz.with_name(npz.stem + ".tmp.npz")
    np.savez_compressed(tmp, **paths)
    tmp.replace(npz)
    tmp = target.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(target)
    return target


# ------------------------------------------------------------------------------------------ runner
class E3Runner:
    """State of one call of ``run_e3`` (module docstring); ``dry=True``: nothing is created or written (dry runs)."""

    def __init__(self, cfg: DictConfig, echo: bool = True, dry: bool = False) -> None:
        self.cfg = cfg
        self.dry = bool(dry)
        self.root = _path(cfg.root)
        self.models = check_models(cfg)
        self.variants = parse_variants(cfg)  # a config error refuses before anything is created
        if not self.dry:
            self.root.mkdir(parents=True, exist_ok=True)
        self.log = RunLog(self.root, echo, write=not self.dry)
        self.device = resolve_device(str(cfg.get("device", "auto")))
        self.profiles_file = _path(cfg.profiles)
        self.profiles, self.profiles_meta = load_profiles(self.profiles_file)
        if not self.profiles:
            raise ValueError(f"no profiles in {cfg.profiles}")
        self.by_id = {p.id: p for p in self.profiles}
        rules = dict(self.profiles_meta.get("rules") or {})
        self.sg_window, self.sg_order = int(rules.get("sg_window", 11)), int(rules.get("sg_order", 2))
        self.steady_s = float(rules.get("steady_s", 5.0))
        sets = _plain(cfg.get("sets")) or {"headline": 10, "extended": None}
        self.run_ids = select_profile_ids(self.profiles, _plain(cfg.get("run_profiles", "extended")), sets)
        self.seeds = [int(s) for s in cfg.seeds]
        p = dict(_plain(cfg.platoon))
        self.batch_profiles = int(p.pop("batch_profiles", 60))
        self.pcfg = PlatoonConfig(**p)
        self.n_steps = time_line_steps(max(q.n_samples for q in self.profiles), self.pcfg)
        idm_file = json.loads(_path(cfg.idm).read_text(encoding="utf-8"))
        self.idm_params = dict(idm_file["calibrations"]["closed_loop"]["params"])
        self.idm = IDMParams.from_dict(self.idm_params)
        self.s_init = {q.id: float(initial_gap(q.initial_speed, self.idm, self.pcfg.gap_cap)) for q in self.profiles}
        # variants with profile_window: their profiles (id -> Profile), initial gaps and provenance (prepare_profiles)
        self.variant_profiles: dict[str, dict[str, Profile]] = {}
        self.variant_s_init: dict[str, dict[str, float]] = {}
        self.variant_info: dict[str, dict[str, Any]] = {}
        self._runs: dict[str, Any] = {}  # parsed raw OpenACC files, shared by the variants of one call
        self.active: dict[str, list[str]] = {}  # variant -> targets whose checkpoint directory exists
        self.skipped: dict[str, str] = {}
        self.written: list[str] = []
        self.current: str | None = None
        self.started, self.started_at = time.perf_counter(), _now()

    # ------------------------------------------------------------------ bookkeeping
    def guard(self, unit: str, fn: Callable[..., Any], *args: Any) -> Any:
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 - one failing unit must not stop the run
            self.log.error(unit, exc)
            return None

    def spec_of(self, variant: str, label: str) -> dict[str, Any]:
        """The model spec of ``label`` in ``variant``: ``cfg.models[label]`` with the variant's overrides."""
        return {**self.models[label], **self.variants[variant].models.get(label, {})}

    def target_of(self, variant: str, label: str) -> str:
        return str(self.spec_of(variant, label).get("target", "a"))

    def runs_in(self, variant: str, label: str) -> bool:
        """Whether ``label`` runs in ``variant``: listed by the model's ``variants`` (absent: every variant) and by the
        variant's ``only_models`` (absent: every label)."""
        where = self.models[label].get("variants")
        only = self.variants[variant].only_models
        return (where is None or variant in where) and (only is None or label in only)

    def labels(self, variant: str) -> list[str]:
        """Model labels that run in ``variant`` (``runs_in``) and whose target directory exists in it."""
        return [m for m in self.models
                if self.runs_in(variant, m) and self.target_of(variant, m) in self.active.get(variant, [])]

    def status(self, finished: bool = False) -> dict[str, Any]:
        per_variant = {}
        for v in self.active:
            labels = self.labels(v)
            done = sum(unit_path(self.root, v, m, s, p).exists() for m in labels for s in self.seeds for p in self.run_ids)
            per_variant[v] = {"done": done, "total": len(labels) * len(self.seeds) * len(self.run_ids)}
        return {
            "root": str(self.root),
            "device": str(self.device),
            "started": self.started_at,
            "updated": _now(),
            "elapsed_s": round(time.perf_counter() - self.started, 1),
            "finished": finished,
            "current": None if finished else self.current,
            "done": sum(v["done"] for v in per_variant.values()),
            "total": sum(v["total"] for v in per_variant.values()),
            "variants": per_variant,
            "skipped": dict(self.skipped),
            "written_now": len(self.written),
            "failures": list(self.log.failures),
        }

    def write_status(self, finished: bool = False) -> dict[str, Any]:
        status = self.status(finished)
        _write_json_atomic(self.root / "status.json", status)
        return status

    def timing(self, record: Mapping[str, Any]) -> None:
        with open(self.root / "logs" / "timings.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(_jsonable(dict(record))) + "\n")

    def cleanup(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ settings
    def run_spec(self) -> dict[str, Any]:
        """Settings shared by every unit of the root (compared on resume)."""
        return _jsonable({
            "platoon": {**{k: getattr(self.pcfg, k) for k in ("n_followers", "n_rollouts", "n_rollouts_sampled", "burn_in_s",
                                                             "tail_s", "seed", "gap_cap", "max_batch", "v_max")},
                        "batch_profiles": self.batch_profiles},  # the batches define the random streams
            "n_steps": self.n_steps, "dt": DT, "hist_len": HIST_LEN, "profiles_digest": profiles_digest(self.profiles),
            "idm": self.idm_params,
        })  # fmt: skip

    def model_spec(self, variant: str, label: str) -> dict[str, Any]:
        """Settings of the units of (variant, label), stored in ``units/<variant>/<label>/spec.json``; a variant with
        ``profile_window`` adds the window, the order and the digest of its re-smoothed profiles."""
        spec = self.spec_of(variant, label)
        target = str(spec.get("target", "a"))
        out = {
            "label": label, "model": str(spec.get("model", label)), "target": target,
            "checkpoints": self.checkpoint_dir(variant, label),
            "sampler_mode": str(spec.get("sampler_mode", "per_step")), "hold_steps": int(spec.get("hold_steps", 1)),
        }  # fmt: skip
        if self.variants[variant].profile_window is not None:
            self.prepare_profiles(variant)
            out.update({"profile_window": self.variants[variant].profile_window, "profile_order": self.sg_order,
                        "profiles_digest": self.variant_info[variant]["digest"]})  # fmt: skip
        return _jsonable(out)

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

    def compare_spec(self, path: Path, spec: Mapping[str, Any]) -> str:
        """``check_spec`` without writing: ``new`` (no stored spec), ``equal`` or ``conflict``."""
        if not path.exists():
            return "new"
        return "equal" if json.loads(path.read_text(encoding="utf-8")) == dict(spec) else "conflict"

    def checkpoint_dir(self, variant: str, label: str) -> str:
        """Checkpoints directory of ``label`` in ``variant``: the model's ``checkpoints`` (after the variant's
        overrides), else the variant's directory of the label's target."""
        spec = self.spec_of(variant, label)
        own = spec.get("checkpoints")
        return str(own) if own is not None else self.variants[variant].checkpoints[str(spec.get("target", "a"))]

    def checkpoint(self, variant: str, label: str, seed: int) -> Path:
        spec = self.model_spec(variant, label)
        return _path(spec["checkpoints"]) / spec["model"] / f"seed{int(seed)}" / "model.pt"

    # ------------------------------------------------------------------ profiles of a variant
    def profiles_of(self, variant: str) -> dict[str, Profile]:
        """id -> profile simulated in ``variant`` (the re-smoothed ones of a variant with ``profile_window``)."""
        if self.variants[variant].profile_window is None:
            return self.by_id
        return self.prepare_profiles(variant)

    def s_init_of(self, variant: str) -> dict[str, float]:
        """id -> initial gap in ``variant``: the IDM of ``cfg.idm`` at the variant's first leader speed."""
        if self.variants[variant].profile_window is None:
            return self.s_init
        self.prepare_profiles(variant)
        return self.variant_s_init[variant]

    def prepare_profiles(self, variant: str) -> dict[str, Profile]:
        """Re-smoothed profiles of a variant with ``profile_window`` (module docstring): computed from the raw files,
        written to ``profiles__<variant>.yaml`` when absent (not in a dry run), compared with the stored file
        otherwise (ValueError when the settings or the content differ)."""
        if variant in self.variant_profiles:
            return self.variant_profiles[variant]
        v = self.variants[variant]
        source = self.profiles_meta.get("source")
        if source is None:
            raise ValueError(f"variant {variant}: profile_window needs the raw OpenACC directory (key 'source') in "
                             f"{self.profiles_file}")  # fmt: skip
        raw = _path(str(source))
        profiles, info = resmooth_profiles(self.profiles, raw, int(v.profile_window), self.sg_order,
                                           base_window=self.sg_window, steady_s=self.steady_s, runs=self._runs)  # fmt: skip
        s_init = {q.id: float(initial_gap(q.initial_speed, self.idm, self.pcfg.gap_cap)) for q in profiles}
        meta = {
            "description": f"Leader profiles of E3 signal variant {variant}: the leader speed smoothed again from the raw "
                           "OpenACC files (docs/design.md 17.1), written by cf_unc.experiments.e3",
            "variant": variant, "profiles": str(self.cfg.profiles), "source": str(source),
            "source_digest": profiles_digest(self.profiles), "window_source": v.window_source,
            **{k: info[k] for k in ("window", "order", "base_window")},
            "n_profiles": len(profiles), "digest": info["digest"], "check": info["check"], "change": info["change"],
            "leader_speed": info["leader_speed"],
            "initial_gap": {"idm_file": str(self.cfg.idm), "calibration": "closed_loop",
                            "change": _change(s_init, self.s_init)},
            "files": info["files"],
        }  # fmt: skip
        path = variant_profiles_path(self.root, variant)
        settings = ("window", "order", "base_window", "source_digest", "digest")
        if path.exists():
            stored = load_profiles(path)[1]
            asked = {k: meta[k] for k in settings}
            found = {k: stored.get(k) for k in settings}
            if found != asked:
                raise ValueError(f"{path} was written with {found}, the config and the raw files give {asked}; use another root")
            state = "stored"
        elif self.dry:
            state = "new"
        else:
            write_profiles(path, profiles, meta)
            state = "written"
            median = info["change"]["leader_amplitude"].get("median", 0.0)
            self.log.line(f"[{variant}] {len(profiles)} profiles smoothed again with {v.profile_window} samples -> "
                          f"{path.name} (digest {meta['digest']}; leader amplitude change, median {median:+.3f} m/s)")  # fmt: skip
        self.variant_profiles[variant] = {q.id: q for q in profiles}
        self.variant_s_init[variant] = s_init
        self.variant_info[variant] = {"path": path, "state": state, "digest": meta["digest"], "meta": meta}
        return self.variant_profiles[variant]

    # ------------------------------------------------------------------ simulation
    def run_group(self, variant: str, label: str, seed: int) -> None:
        """Every missing unit of (variant, label, seed); the checkpoint is loaded once."""
        group = f"{variant}/{label}/seed{seed}"
        if all(unit_path(self.root, variant, label, seed, p).exists() for p in self.run_ids):
            return
        spec = self.model_spec(variant, label)
        self.check_spec(self.root / "units" / variant / label / "spec.json", spec)
        path = self.checkpoint(variant, label, seed)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint {path} does not exist")
        self.current = group
        model = load_model(path, device=self.device)
        sampled = bool(getattr(model, "plan_sampler", False))  # plan samplers: n_rollouts_sampled, hold 1
        n_r = int(self.pcfg.n_rollouts_sampled if sampled else self.pcfg.n_rollouts)
        hold = 1 if sampled else int(spec["hold_steps"])
        size = max(1, min(self.batch_profiles, int(self.pcfg.max_batch) // n_r))
        by_id = self.profiles_of(variant)
        for start in range(0, len(self.run_ids), size):
            batch = [by_id[i] for i in self.run_ids[start : start + size]]
            if all(unit_path(self.root, variant, label, seed, p.id).exists() for p in batch):
                continue
            self.current = f"{group} ({batch[0].id}..{batch[-1].id})"
            self.simulate_batch(variant, label, seed, model, batch, n_r, hold, spec["sampler_mode"])
            self.write_status()
        del model

    def simulate_batch(self, variant: str, label: str, seed: int, model: CFModel, batch: list[Profile], n_r: int,
                       hold: int, mode: str) -> None:  # fmt: skip
        n_v = self.pcfg.n_followers
        seeds = stream_seeds(self.pcfg.seed, seed, [p.id for p in batch], n_v)
        seed_everything(seed)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        s_init = self.s_init_of(variant)
        result = simulate_platoon(
            model, [p.leader_speed for p in batch], [s_init[p.id] for p in batch], seeds, self.pcfg,
            n_rollouts=n_r, hold_steps=hold, sampler_mode=mode, n_steps=self.n_steps, device=self.device,
            ids=[p.id for p in batch],
        )  # fmt: skip
        seconds = time.perf_counter() - started
        burn = result.burn_in
        for e, profile in enumerate(batch):
            if unit_path(self.root, variant, label, seed, profile.id).exists():
                continue  # simulated again only because another unit of its batch was missing
            k_end = burn + profile.n_samples - 1 + result.tail
            paths = {
                "mean_speed": result.mean_speed[e, :, burn : k_end + 1].astype(np.float16),  # (V, n + T) from the profile start
                "leader_speed": np.asarray(_leader_window(profile, result.tail), dtype=np.float32),
                "dt": np.float32(DT),
            }  # fmt: skip
            _write_unit(self.root, variant, label, seed, profile, unit_table(result, e, profile, seed), paths)
            self.written.append(f"{variant}/{label}/seed{seed}/{profile.id}")
        mem = torch.cuda.max_memory_allocated(self.device) / 2**20 if self.device.type == "cuda" else None
        collided = float(np.mean(result.first_collision >= 0))
        self.timing({
            "time": _now(), "variant": variant, "model": label, "seed": seed, "profiles": [p.id for p in batch],
            "n_profiles": len(batch), "seconds": seconds, "seconds_per_unit": seconds / len(batch), "n_rollouts": n_r,
            "hold_steps": hold, "sampler_mode": mode, "n_steps": result.n_steps, "n_vehicles": n_v,
            "seconds_per_vehicle": result.info["seconds_per_vehicle"], "peak_memory_mb": mem, "device": str(self.device),
        })  # fmt: skip
        amp1 = float(np.mean(result.amplitude[:, 0]))
        self.log.line(
            f"[{variant} {label} seed{seed}] {len(batch)} profiles ({batch[0].id}..{batch[-1].id}) x {n_v} vehicles x "
            f"{n_r} rollouts, {result.n_steps} steps in {seconds:.1f} s ({seconds / len(batch):.1f} s per unit"
            f"{'' if mem is None else f', peak {mem:.0f} MB'}); mean amplitude of vehicle 1 {amp1:.2f} m/s, "
            f"collided {collided:.3f}"
        )

    # ------------------------------------------------------------------ run
    def activate(self) -> None:
        """``active``: per requested variant the targets of its model specs whose checkpoint directory exists; the
        others are ``skipped`` with a warning."""
        for variant, v in self.variants.items():
            if not v.checkpoints:  # not requested (run_e3.py --variants without this one)
                continue
            present = []
            for t in sorted({self.target_of(variant, m) for m in self.models if self.runs_in(variant, m)}):
                directory = v.checkpoints.get(t)
                if directory is not None and _path(directory).is_dir():
                    present.append(t)
                else:
                    self.skipped[f"{variant}/{t}"] = f"checkpoint directory {directory} does not exist"
                    self.log.line(f"warning: variant {variant}, target {t} skipped: checkpoint directory {directory} does not exist")
            if present:
                self.active[variant] = present

    def run(self) -> dict[str, Any]:
        OmegaConf.save(self.cfg, self.root / "config.yaml", resolve=True)
        self.activate()
        status = self.status()
        self.log.line(
            f"E3 run: root {self.root}, device {self.device}; variants {list(self.active)}; {len(self.models)} models x "
            f"{len(self.seeds)} seeds x {len(self.run_ids)} profiles; K = {self.n_steps} steps; units "
            f"{status['done']}/{status['total']} done"
        )
        self.write_status()
        finished = False
        try:
            if self.guard("spec", self.check_spec, self.root / "spec.json", self.run_spec()) is not None:
                for variant in self.active:
                    if self.variants[variant].profile_window is not None and \
                            self.guard(f"{variant}/profiles", self.prepare_profiles, variant) is None:  # fmt: skip
                        continue  # the variant's profiles cannot be derived or differ from the stored ones
                    for label in self.labels(variant):
                        for seed in self.seeds:
                            self.guard(f"{variant}/{label}/seed{seed}", self.run_group, variant, label, seed)
                            self.cleanup()
                            self.write_status()
            finished = True
        finally:
            status = self.write_status(finished=finished)
        self.log.line(
            f"{'finished' if finished else 'stopped'} after {status['elapsed_s']:.0f} s: units {status['done']}/"
            f"{status['total']} ({len(self.written)} new), {len(self.log.failures)} failure(s), skipped "
            f"{sorted(self.skipped) or 'none'}"
        )
        for failure in self.log.failures:
            self.log.line(f"  failed: {failure['unit']}: {failure['error']}")
        return status

    def dry_run(self) -> dict[str, Any]:
        """What ``run`` would do, writing nothing (no directory, log, status or spec): per requested variant the
        targets, the profile file (re-smoothed profiles computed from the raw files and compared with the stored file;
        their change against the profiles file) and per model the spec (``new`` / ``equal`` / ``conflict`` against the
        stored ``spec.json``), the units done and missing over the run set and the seeds without a checkpoint. A
        conflict (stored settings that differ from the config) is listed in ``conflicts``: a run would refuse it."""
        conflicts: list[dict[str, str]] = []
        state = self.compare_spec(self.root / "spec.json", self.run_spec())
        if state == "conflict":
            conflicts.append({"unit": "spec", "error": f"{self.root / 'spec.json'} differs from the config"})
        self.activate()
        variants: dict[str, Any] = {}
        for variant in self.active:
            v = self.variants[variant]
            entry: dict[str, Any] = {"targets": list(self.active[variant]), "profile_window": v.profile_window,
                                     "window_source": v.window_source, "models": {}}  # fmt: skip
            ok = True
            if v.profile_window is not None:
                try:
                    self.prepare_profiles(variant)
                    info = self.variant_info[variant]
                    meta = info["meta"]
                    entry["profiles"] = {"file": str(info["path"]), "state": info["state"], "digest": info["digest"],
                                         "check": meta["check"], "change": meta["change"],
                                         "initial_gap_change": meta["initial_gap"]["change"]}  # fmt: skip
                except Exception as exc:  # noqa: BLE001 - reported as a conflict, the dry run goes on
                    ok = False
                    entry["profiles"] = {"error": f"{type(exc).__name__}: {exc}"}
                    conflicts.append({"unit": f"{variant}/profiles", "error": entry["profiles"]["error"]})
            for label in self.labels(variant):
                spec = self.spec_of(variant, label)
                done = sum(unit_path(self.root, variant, label, s, p).exists() for s in self.seeds for p in self.run_ids)
                row: dict[str, Any] = {
                    "target": self.target_of(variant, label), "hold_steps": int(spec.get("hold_steps", 1)),
                    "sampler_mode": str(spec.get("sampler_mode", "per_step")),
                    "checkpoints": self.checkpoint_dir(variant, label), "done": done,
                    "missing": len(self.seeds) * len(self.run_ids) - done,
                }  # fmt: skip
                if ok:
                    row["spec"] = self.compare_spec(self.root / "units" / variant / label / "spec.json",
                                                    self.model_spec(variant, label))  # fmt: skip
                    if row["spec"] == "conflict":
                        conflicts.append({"unit": f"{variant}/{label}", "error": "stored spec.json differs from the config"})
                    row["missing_checkpoints"] = [s for s in self.seeds if not self.checkpoint(variant, label, s).exists()]
                entry["models"][label] = row
            entry["done"] = sum(r["done"] for r in entry["models"].values())
            entry["missing"] = sum(r["missing"] for r in entry["models"].values())
            variants[variant] = entry
        missing = sum(e["missing"] for e in variants.values())
        return {
            "root": str(self.root), "spec": state, "profiles": len(self.profiles), "run_profiles": len(self.run_ids),
            "seeds": list(self.seeds), "variants": variants, "skipped": dict(self.skipped), "conflicts": conflicts,
            "done": sum(e["done"] for e in variants.values()), "missing": missing, "nothing_to_do": missing == 0,
        }  # fmt: skip


def _leader_window(profile: Profile, tail: int) -> np.ndarray:
    """Leader speed over the response window: the profile and ``tail`` samples at its last speed."""
    u = np.asarray(profile.leader_speed, dtype=np.float64)
    return np.concatenate([u, np.full(tail, u[-1])])


def _change(new: Mapping[str, float], base: Mapping[str, float]) -> dict[str, float]:
    """min / median / max of ``new - base`` over the common keys."""
    d = np.array([new[k] - base[k] for k in new if k in base], dtype=np.float64)
    return {"min": float(d.min()), "median": float(np.median(d)), "max": float(d.max())} if d.size else {}


def run_e3(cfg: DictConfig | Mapping[str, Any], echo: bool = True) -> dict[str, Any]:
    """Run or resume experiment E3 (module docstring); returns the final status (also in ``status.json``)."""
    cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(dict(cfg))
    configure_torch(bool(cfg.get("tf32", False)))
    return E3Runner(cfg, echo=echo).run()


def dry_run_e3(cfg: DictConfig | Mapping[str, Any], echo: bool = True) -> dict[str, Any]:
    """``E3Runner.dry_run`` of the config: units done and missing, stored settings compared; writes nothing."""
    cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(dict(cfg))
    return E3Runner(cfg, echo=echo, dry=True).dry_run()
