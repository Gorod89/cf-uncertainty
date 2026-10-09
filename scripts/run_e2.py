"""Experiment E2: closed-loop stochastic rollouts of the E1 models (docs/design.md, section 11), resumable.

    python scripts/run_e2.py                                   # full experiment -> artifacts/e2
    python scripts/run_e2.py --pilot                           # artifacts/e2_pilot: seed 0, 100 events per data set, target a
    python scripts/run_e2.py --targets a --models hetero_mdn deep_ensemble --seeds 0 1 --datasets highd ngsim
    python scripts/run_e2.py --set rollout.max_batch=50000 --root artifacts/e2_test
    python scripts/run_e2.py --config configs/experiment/e2h.yaml  # harmonised signals -> artifacts/e2h (checkpoints of e1h)
    python scripts/run_e2.py --config configs/experiment/e2m.yaml  # E5, matched signals -> artifacts/e2m (checkpoints of e1h)

Precedence: config file < --pilot < --set < explicit flags. Exit code 1 if any unit failed (see logs/errors.log), 2
when the run is refused because the matching file of signal ``matched`` is missing or stale (design.md 15.1).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

from omegaconf import OmegaConf  # noqa: E402

from cf_unc.data.harmonise import MatchingError  # noqa: E402
from cf_unc.experiments.e2 import DEFAULT_CONFIG, load_config, run_e2  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="experiment config (YAML)")
    parser.add_argument("--root", help="output directory (overrides the config)")
    parser.add_argument("--targets", nargs="+", help="targets (keys of targets; default: all)")
    parser.add_argument("--models", nargs="+", help="model labels, variants included (default: all of the config)")
    parser.add_argument("--seeds", nargs="+", type=int, help="seeds (default: those of the config)")
    parser.add_argument("--datasets", nargs="+", help="data sets (keys of datasets)")
    parser.add_argument("--device", help="auto | cpu | cuda")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help="dotted overrides of the config")
    parser.add_argument("--pilot", action="store_true", help="reduced experiment (cf_unc.experiments.e2.PILOT)")
    parser.add_argument("--quiet", action="store_true", help="no progress lines on stdout (logs/run.log only)")
    return parser.parse_args(argv)


def _select(cfg_section, names: list[str], what: str):  # noqa: ANN001, ANN202
    unknown = sorted(set(names) - set(cfg_section))
    if unknown:
        raise SystemExit(f"unknown {what} {unknown}; configured: {list(cfg_section)}")
    return OmegaConf.create({k: cfg_section[k] for k in names})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.set, pilot=args.pilot)
    if args.root:
        cfg.root = args.root
    if args.targets:
        cfg.targets = _select(cfg.targets, args.targets, "targets")
    if args.models:
        variants = set(cfg.get("variants") or {})
        cfg.models = [m for m in args.models if m not in variants]
        cfg.variants = OmegaConf.create({k: v for k, v in (cfg.get("variants") or {}).items() if k in args.models})
    if args.seeds:
        cfg.seeds = list(args.seeds)
    if args.device:
        cfg.device = args.device
    if args.datasets:
        cfg.datasets = _select(cfg.datasets, args.datasets, "data sets")
    try:
        status = run_e2(cfg, echo=not args.quiet)
    except MatchingError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(status, indent=2))
    return 1 if status["failures"] or not status["finished"] else 0


if __name__ == "__main__":
    sys.exit(main())
