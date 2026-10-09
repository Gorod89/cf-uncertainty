"""Select ``w_phys`` of QuantilePIDL on the highD validation fold (experiment E1 training data, seed 0).

    python scripts/select_w_phys.py --root artifacts/e1
    python scripts/select_w_phys.py --config configs/experiment/e1_mean1s.yaml --grid 0 0.0001 0.001 0.01 0.1 0.3
    python scripts/select_w_phys.py --config configs/experiment/e1h.yaml --grid 0 0.0001 0.001 0.01 0.1 0.3   # harmonised
    python scripts/select_w_phys.py --root artifacts/e1 --max-epochs 20           # capped training budget
    python scripts/select_w_phys.py --grid 0 0.03 0.3 --set eval.batch_size=16384  # dotted config overrides

Every grid value trains ``quantile_pidl`` as the experiment of ``--config`` would (training windows and target,
signal variant, stride, IDM calibration and training config, seed 0; data, kwargs and IDM come from the runner's
``Runner``) and is scored on the highD validation windows of that target: mean pinball loss over the quantile levels, PICP
and mean width of the 5-95 % interval, RMSE of the median, share of windows with crossing quantiles, training
seconds. The IDM calibrations of every model of the experiment are computed first (``<root>/idm.json``, reused
by the full run).

Rule (``select_value``): the smallest validation pinball loss among ``w_phys > 0``; among the values within 1 % of
it the LARGEST ``w_phys`` (physics is kept when it costs nothing); ``w_phys = 0`` is the no-physics reference and
never selected. Results: ``<root>/selection/w_phys.json`` (root of the config or ``--root``), rewritten after
every grid value (values already scored with the same settings, target and signal included, are reused on a
rerun). The experiment config is not edited: the script prints the line for ``model_overrides``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from cf_unc.data.windows import DEFAULT_TARGET, WindowSet  # noqa: E402
from cf_unc.eval.metrics import pinball_loss  # noqa: E402
from cf_unc.experiments.e1 import DEFAULT_CONFIG, Runner, configure_torch, load_config, seed_everything  # noqa: E402
from cf_unc.models import CFModel, build_model  # noqa: E402
from cf_unc.train.run import write_json  # noqa: E402

LABEL = "quantile_pidl"
DEFAULT_GRID = (0.0, 0.01, 0.03, 0.1, 0.3)
TOLERANCE = 0.01
SEED = 0
RULE = (
    "smallest validation pinball loss among w_phys > 0; among the values within 1 % of it the largest w_phys "
    "(physics is kept when it costs nothing); w_phys = 0 is the no-physics reference and never selected"
)
COLUMNS = ("pinball", "picp", "width", "rmse_median", "crossing_share")


def select_value(results: Sequence[Mapping[str, Any]], tolerance: float = TOLERANCE) -> float:
    """Selected ``w_phys`` of result rows ``{"w_phys", "pinball", ...}`` (rule of the module docstring).

    Rows with ``w_phys <= 0`` or a non-finite pinball loss are ignored; ValueError if none is left.
    """
    rows = [r for r in results if float(r["w_phys"]) > 0.0 and math.isfinite(float(r["pinball"]))]
    if not rows:
        raise ValueError("no grid value with w_phys > 0 and a finite validation pinball loss")
    best = min(float(r["pinball"]) for r in rows)
    return max(float(r["w_phys"]) for r in rows if float(r["pinball"]) <= best * (1.0 + tolerance))


def score(model: CFModel, windows: WindowSet, batch_size: int = 8192) -> dict[str, float]:
    """Scores of a fitted QuantilePIDL on ``windows`` [m/s^2]: mean pinball loss over its levels (knots of the
    predictive law), PICP and mean width of the 5-95 % interval, RMSE of the median, share of rows whose raw
    quantiles cross."""
    dist = model.predict_dist(windows.X, batch_size=batch_size)
    y = windows.y.astype(np.float64)
    levels = np.asarray(model.levels, dtype=np.float64)  # type: ignore[attr-defined]
    lo, med, hi = dist.quantile([0.05, 0.5, 0.95]).T
    return {
        "pinball": float(np.mean(pinball_loss(y, dist.quantile(levels), levels))),
        "picp": float(np.mean((y >= lo) & (y <= hi))),
        "width": float(np.mean(hi - lo)),
        "rmse_median": float(np.sqrt(np.mean((y - med) ** 2))),
        "crossing_share": float(np.mean(np.asarray(dist.n_crossings) > 0)),  # type: ignore[attr-defined]
    }


def table(results: Sequence[Mapping[str, Any]], selected: float | None) -> str:
    """Plain-text table of the grid results (relative pinball against the best ``w_phys > 0``)."""
    finite = [r["pinball"] for r in results if r["w_phys"] > 0 and math.isfinite(r["pinball"])]
    best = min(finite) if finite else math.nan
    lines = [f"{'w_phys':>7} {'pinball':>9} {'vs best':>8} {'PICP90':>7} {'width':>7} {'RMSE med':>9} "
             f"{'crossing':>9} {'train s':>8} {'epochs':>7}"]  # fmt: skip
    for r in results:
        note = "  <- selected" if selected is not None and r["w_phys"] == selected else ""
        note = "  (reference, no physics)" if r["w_phys"] == 0 else note
        lines.append(
            f"{r['w_phys']:>7g} {r['pinball']:>9.5f} {100 * (r['pinball'] / best - 1):>7.2f}% {r['picp']:>7.3f} "
            f"{r['width']:>7.3f} {r['rmse_median']:>9.4f} {r['crossing_share']:>9.4f} {r['train_seconds']:>8.0f} "
            f"{r['epochs']:>7d}{note}"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="experiment config (YAML)")
    parser.add_argument("--root", help="experiment root (overrides the config); results in <root>/selection")
    parser.add_argument("--grid", nargs="+", type=float, default=list(DEFAULT_GRID), help="w_phys values")
    parser.add_argument("--max-epochs", type=int, help=f"cap of {LABEL}'s max_epochs (model_overrides)")
    parser.add_argument("--device", help="auto | cpu | cuda")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help="dotted overrides of the config")
    parser.add_argument("--verbose", action="store_true", help="per-epoch lines of the training loop")
    parser.add_argument("--quiet", action="store_true", help="no progress lines on stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    if args.root:
        cfg.root = args.root
    if args.device:
        cfg.device = args.device
    if args.max_epochs is not None:
        cap = {LABEL: {"train": {"max_epochs": int(args.max_epochs)}}}
        cfg.model_overrides = OmegaConf.merge(cfg.get("model_overrides") or {}, cap)
    configure_torch(bool(cfg.get("tf32", False)))
    runner = Runner(cfg, echo=not args.quiet, log_name="select_w_phys.log")
    for label in runner.models:  # the IDM calibrations of every experiment model (idm.json for the full run)
        if label != LABEL:
            runner.guard(f"{label}/config", runner.spec, label)
    spec = runner.spec(LABEL)
    base = {k: v for k, v in spec.kwargs.items() if k != "w_phys"}
    bundle = runner.data.bundle(spec.sequences)
    train, val = bundle.windows["train"], bundle.windows["val"]
    batch_size = int(cfg.eval.get("batch_size", 8192))

    path = runner.root / "selection" / "w_phys.json"
    settings = json.loads(json.dumps({"model": LABEL, "seed": SEED, "kwargs": base, "idm": spec.idm,
                                      "target": spec.target, "n_train": len(train), "n_val": len(val),
                                      **({} if spec.signal is None else {"signal": spec.signal})}))  # fmt: skip
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    stored_settings = stored.get("settings")
    if isinstance(stored_settings, dict):  # files written before the target option: target "a"
        stored_settings = {"target": DEFAULT_TARGET, **stored_settings}
    done = {float(r["w_phys"]): r for r in stored.get("results", [])} if stored_settings == settings else {}

    def write(results: list[dict], selected: float | None) -> None:
        reference = next((r for r in results if r["w_phys"] == 0), None)
        write_json(path, {"settings": settings, "grid": list(args.grid), "rule": RULE, "tolerance": TOLERANCE,
                          "results": results, "selected": selected, "reference": reference,
                          "device": str(runner.device), "updated": time.strftime("%Y-%m-%d %H:%M:%S")})  # fmt: skip

    results: list[dict] = []
    for w_phys in (float(w) for w in args.grid):
        if w_phys in done:
            results.append(done[w_phys])
            runner.log.line(f"w_phys {w_phys:g}: stored result reused")
            continue
        runner.log.line(f"w_phys {w_phys:g}: training {LABEL} on {len(train):,} windows (seed {SEED}), device {runner.device}")
        seed_everything(SEED)
        model = build_model(spec.name, **{**base, "w_phys": w_phys}).to(runner.device)
        started = time.perf_counter()
        log = model.fit(train, val, seed=SEED, verbose=args.verbose)
        seconds = time.perf_counter() - started
        row = {"w_phys": w_phys, **score(model, val, batch_size), "train_seconds": seconds, "epochs": int(log["epochs"]),
               "best_epoch": int(log["fit"]["best_epoch"]), "val_nll": log.get("val_nll")}  # fmt: skip
        del model
        runner.cleanup()
        results.append(row)
        write(results, None)
        runner.log.line(
            f"w_phys {w_phys:g}: pinball {row['pinball']:.5f}, PICP {row['picp']:.3f}, width {row['width']:.3f}, "
            f"median RMSE {row['rmse_median']:.4f}, crossings {row['crossing_share']:.4f}, {seconds:.0f} s, "
            f"{row['epochs']} epochs"
        )
    selected = select_value(results)
    write(results, selected)
    overrides = OmegaConf.to_container((cfg.get("model_overrides") or {}).get(LABEL) or OmegaConf.create({}))
    line = json.dumps({**overrides, "w_phys": selected}).replace('"', "")
    print(f"\n{LABEL} on the highD validation windows ({len(val):,}, target {spec.target}, signal {runner.signal}), "
          f"seed {SEED}; rule: {RULE}\n")
    print(table(results, selected))
    print(f"\nselected w_phys = {selected:g}   (written to {path})")
    print(f"{args.config}, model_overrides:\n  {LABEL}: {line}")
    print(f"or on the command line: --set model_overrides.{LABEL}.w_phys={selected:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
