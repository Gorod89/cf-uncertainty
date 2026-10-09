"""Experiment E1: train every model on highD (seeds 0-4) and write the prediction tables of design.md, section 10.2.

    python scripts/run_e1.py                                   # full experiment -> artifacts/e1 (resumable)
    python scripts/run_e1.py --pilot                           # reduced run -> artifacts/e1_pilot (timing, checks)
    python scripts/run_e1.py --config configs/experiment/e1_mean1s.yaml [--pilot]   # target mean_1s -> artifacts/e1_mean1s[_pilot]
    python scripts/run_e1.py --models hetero_mdn deep_ensemble --seeds 0 1 --datasets highd ngsim
    python scripts/run_e1.py --set eval.chunk_rows=20000 train.overrides.max_epochs=20 --root artifacts/e1_test
    python scripts/run_e1.py --config configs/experiment/e1h.yaml [--pilot]   # harmonised signals -> artifacts/e1h[_pilot]
    python scripts/run_e1.py --dry-run [--config ...]            # what a run would do; loads no data, writes nothing
    python scripts/match_roughness.py --config configs/experiment/e1m.yaml    # E5: matching file first, then
    python scripts/run_e1.py --config configs/experiment/e1m.yaml --dry-run   # matched signals -> artifacts/e1m
    python scripts/run_e1.py --config configs/experiment/e6_pool.yaml --dry-run   # E6: pooled training domain; the dry
                                                               # run loads its events to count the windows after the caps

Precedence: config file < --pilot < --set < explicit flags. Exit code 1 if any unit failed (see logs/errors.log);
with --dry-run, if a stored setting conflicts with the config (or a reuse source fails its checks). Exit code 2 when
the run is refused because the matching file of signal ``matched`` is missing or stale (design.md 15.1).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

from omegaconf import OmegaConf  # noqa: E402

from cf_unc.data.harmonise import MatchingError  # noqa: E402
from cf_unc.experiments.e1 import DEFAULT_CONFIG, dry_run_e1, load_config, pool_members, run_e1  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="experiment config (YAML)")
    parser.add_argument("--root", help="output directory (overrides the config)")
    parser.add_argument("--models", nargs="+", help="model config names (default: all of the config)")
    parser.add_argument("--seeds", nargs="+", type=int, help="seeds (default: those of the config)")
    parser.add_argument("--datasets", nargs="+", help="evaluation data sets (keys of eval.datasets; a pool brings its members)")
    parser.add_argument("--device", help="auto | cpu | cuda")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help="dotted overrides of the config")
    parser.add_argument("--pilot", action="store_true", help="reduced experiment (cf_unc.experiments.e1.PILOT)")
    parser.add_argument("--quiet", action="store_true", help="no progress lines on stdout (logs/run.log only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="compare the stored settings with the config and count what is done; writes nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.set, pilot=args.pilot)
    if args.root:
        cfg.root = args.root
    if args.models:
        cfg.models = list(args.models)
    if args.seeds:
        cfg.seeds = list(args.seeds)
    if args.device:
        cfg.device = args.device
    if args.datasets:
        unknown = sorted(set(args.datasets) - set(cfg.eval.datasets))
        if unknown:
            raise SystemExit(f"unknown data sets {unknown}; configured: {list(cfg.eval.datasets)}")
        selected = list(args.datasets)
        for name in args.datasets:  # a pool (design.md 16.1) is assembled from its members: they come along
            selected += [m for m in (pool_members(cfg, name) or []) if m not in selected]
        cfg.eval.datasets = OmegaConf.create({k: cfg.eval.datasets[k] for k in selected})
    try:
        if args.dry_run:
            plan = dry_run_e1(cfg, echo=not args.quiet)
            kinds = ("windows", "train_density", "idm", "checkpoints", "tables")
            brief = {k: ({n: len(x) for n, x in v.items()} if k in kinds else v) for k, v in plan.items()}
            brief["reuse"] = [f"{d['model']} seeds {d['seeds']} <- {d['source']}" for d in plan["reuse"]]
            print(json.dumps(brief, indent=2))
            return 1 if plan["conflicts"] else 0
        status = run_e1(cfg, echo=not args.quiet)
    except MatchingError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(status, indent=2))
    return 1 if status["failures"] or not status["finished"] else 0


if __name__ == "__main__":
    sys.exit(main())
