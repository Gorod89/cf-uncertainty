"""Training and one-step evaluation of a single model; shared by scripts/train.py and the experiment scripts."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from omegaconf import DictConfig, OmegaConf

from cf_unc.constants import SEQ_LEN
from cf_unc.data.pipeline import DataBundle, load_bundle, synthetic_bundle
from cf_unc.data.windows import WindowSet
from cf_unc.eval.bootstrap import bootstrap_by_group
from cf_unc.eval.metrics import calibration_error, ks_uniform, summarize
from cf_unc.models import CFModel, build_model, constructor_accepts, resolve_device
from cf_unc.physics.idm import IDMParams, calibrate_idm

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAPPED = ("nll", "crps", "covered", "width", "sq_err", "var", "aleatoric_var", "epistemic_var")


def _plain(cfg: Any) -> Any:
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


# ------------------------------------------------------------------------------ data
def prepare_bundle(cfg: DictConfig, sequences: bool = False) -> DataBundle:
    """Windows of the configured data set; ``sequences`` adds 5 s targets and the leader's future speed."""
    kwargs = dict(
        hist_len=int(cfg.hist_len),
        stride=int(cfg.seq_stride if sequences else cfg.stride),
        seq_len=SEQ_LEN if sequences else 0,
        split_seed=int(cfg.get("split_seed", 0)),
    )
    data = cfg.data
    if data.kind == "synthetic":
        return synthetic_bundle(_plain(data.generator), name=str(data.name), **kwargs)
    if data.kind == "parquet":
        split = data.get("split", None)
        if split is not None and split.get("file", None):
            kwargs.update(fold_file=_path(split.file), fold=int(split.get("fold", 0)))
        return load_bundle(_path(data.path), name=str(data.name), unify=bool(data.get("unify_acceleration", True)), **kwargs)
    raise ValueError(f"unknown data kind {data.kind!r}")


def resolve_idm(cfg: DictConfig, bundle: DataBundle) -> tuple[IDMParams, dict]:
    """Calibrated population IDM of the training events (cached as JSON next to the other artifacts)."""
    cache = _path(cfg.idm.cache)
    key = {"data": _plain(cfg.data), "objective": str(cfg.idm.objective), "max_events": int(cfg.idm.max_events)}
    key["split_seed"] = int(cfg.get("split_seed", 0))
    if cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if stored.get("key") == json.loads(json.dumps(key)):
            return IDMParams.from_dict(stored["params"]), stored["report"]
    params, report = calibrate_idm(
        bundle.events["train"], objective=str(cfg.idm.objective), max_events=int(cfg.idm.max_events), seed=0
    )
    write_json(cache, {"key": key, "params": params.to_dict(), "report": report})
    return params, report


# ------------------------------------------------------------------------ evaluation
def one_step_report(
    model: CFModel,
    windows: WindowSet,
    truth: Mapping[str, np.ndarray] | None = None,
    *,
    coverage: float = 0.9,
    n_boot: int = 1000,
    seed: int = 0,
    max_rows: int | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """One-step scores on ``windows`` with driver-bootstrap confidence intervals.

    Returns the report (JSON-like) and the per-window score arrays (with ``driver`` and ``row``).
    """
    n = len(windows)
    rows = np.arange(n)
    if max_rows is not None and n > max_rows:
        rows = np.sort(np.random.default_rng(seed).choice(n, size=max_rows, replace=False))
    future = windows.v_lead_future[rows] if model.needs_leader_future and windows.v_lead_future is not None else None
    started = time.perf_counter()
    dist = model.predict_dist(windows.X[rows], v_lead_future=future)
    seconds = time.perf_counter() - started
    y = windows.y[rows].astype(np.float64)
    groups = windows.driver[rows]
    scores = summarize(dist, y, coverage=coverage, rng=seed)

    report: dict[str, Any] = {"n": int(rows.size), "n_drivers": int(np.unique(groups).size), "predict_seconds": seconds}
    for name in BOOTSTRAPPED:
        if name in scores:
            b = bootstrap_by_group(scores[name], groups, n_boot=n_boot, seed=seed)
            report[name] = {"estimate": float(b["estimate"]), "lo": float(b["lo"]), "hi": float(b["hi"])}
    report["rmse"] = math.sqrt(report["sq_err"]["estimate"])
    ks_stat, ks_p = ks_uniform(scores["pit"])
    report["pit_ks"] = {"statistic": float(ks_stat), "pvalue": float(ks_p)}
    report["calibration_error"] = float(calibration_error(scores["pit"]))
    crossings = getattr(dist, "n_crossings", None)
    if crossings is not None:
        report["crossing_share"] = float(np.mean(np.asarray(crossings) > 0))
    if truth:
        a_det = np.asarray(truth["a_det"])[rows]
        report["rmse_vs_truth"] = float(np.sqrt(np.mean((dist.mean() - a_det) ** 2)))
        report["aleatoric_true"] = float(np.mean(np.asarray(truth["sigma_true"])[rows] ** 2))
        scores["a_det"] = a_det
    scores["driver"] = groups
    scores["row"] = rows
    return report, scores


# -------------------------------------------------------------------------- training
def model_kwargs(cfg: DictConfig, idm: IDMParams | None) -> dict[str, Any]:
    kwargs = dict(_plain(cfg.model.get("kwargs", {})) or {})
    overrides = dict(_plain(cfg.model.get("train", {})) or {})
    if constructor_accepts(str(cfg.model.name), "train"):
        kwargs["train"] = {**dict(_plain(cfg.train) or {}), **overrides}
    elif overrides:
        raise ValueError(f"model {cfg.model.name!r} has no `train` argument but its config sets train overrides")
    if cfg.model.get("needs_idm", False):
        if idm is None:
            raise ValueError("this model needs calibrated IDM parameters")
        kwargs["idm"] = idm.to_dict()
    return kwargs


def run_training(cfg: DictConfig, bundles: dict[str, DataBundle] | None = None, save: bool = True) -> dict:
    """Train ``cfg.model`` on ``cfg.data``, evaluate one-step metrics on val and test, store the artifacts.

    ``bundles`` may carry prepared data (keys ``step`` / ``seq``) to avoid regenerating it for every model.
    """
    started = time.perf_counter()
    sequences = bool(cfg.model.get("needs_sequences", False))
    which = "seq" if sequences else "step"
    bundles = bundles if bundles is not None else {}
    if which not in bundles:
        bundles[which] = prepare_bundle(cfg, sequences=sequences)
    bundle = bundles[which]

    idm, idm_report = (None, None)
    if cfg.model.get("needs_idm", False):
        idm, idm_report = resolve_idm(cfg, bundle)

    device = resolve_device(str(cfg.device))
    model = build_model(str(cfg.model.name), **model_kwargs(cfg, idm)).to(device)
    log = model.fit(bundle.windows["train"], bundle.windows["val"], seed=int(cfg.seed), verbose=bool(cfg.get("verbose", False)))

    max_rows = cfg.get("eval_max_rows_sampled", 2000) if model.needs_leader_future else cfg.get("eval_max_rows", None)
    result: dict[str, Any] = {
        "model": str(cfg.model.name),
        "config_name": str(cfg.get("model_label", cfg.model.name)),
        "data": str(cfg.data.name),
        "seed": int(cfg.seed),
        "device": str(device),
        "n_train": len(bundle.windows["train"]),
        "idm": None if idm is None else {"params": idm.to_dict(), "report": idm_report},
    }
    for split in ("val", "test"):
        truth = None if bundle.truth is None else bundle.truth[split]
        report, _ = one_step_report(
            model,
            bundle.windows[split],
            truth,
            coverage=float(cfg.coverage),
            n_boot=int(cfg.n_boot),
            seed=int(cfg.seed),
            max_rows=None if max_rows is None else int(max_rows),
        )
        result[split] = report
    result["fit_seconds"] = float(log.get("seconds", math.nan))
    result["total_seconds"] = time.perf_counter() - started

    if save:
        out_dir = _path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save(out_dir / "model.pt")
        write_json(out_dir / "metrics.json", result)
        write_json(out_dir / "train_log.json", log)
        (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
        result["out_dir"] = str(out_dir)
    return result


def format_row(result: Mapping[str, Any], split: str = "test") -> dict[str, Any]:
    """Flat summary of one run for tables."""
    rep = result[split]

    def est(name: str) -> float:
        return rep[name]["estimate"] if name in rep else math.nan

    return {
        "model": result.get("config_name", result["model"]),
        "data": result["data"],
        "seed": result["seed"],
        "nll": est("nll"),
        "nll_lo": rep["nll"]["lo"],
        "nll_hi": rep["nll"]["hi"],
        "crps": est("crps"),
        "picp90": est("covered"),
        "width90": est("width"),
        "rmse": rep["rmse"],
        "rmse_vs_truth": rep.get("rmse_vs_truth", math.nan),
        "aleatoric": est("aleatoric_var"),
        "epistemic": est("epistemic_var"),
        "aleatoric_true": rep.get("aleatoric_true", math.nan),
        "pit_ks": rep["pit_ks"]["statistic"],
        "calib_err": rep["calibration_error"],
        "fit_s": result["fit_seconds"],
    }
