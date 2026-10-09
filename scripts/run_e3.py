"""Experiment E3: platoon-level distributional evaluation (docs/design.md, section 13), resumable.

    python scripts/run_e3.py                                    # full experiment -> artifacts/e3
    python scripts/run_e3.py --pilot                            # artifacts/e3_pilot: seed 0, headline profiles, 20 rollouts
    python scripts/run_e3.py --variants raw --models idm_white_noise diffusion_cf --seeds 0 1 --profiles headline --root artifacts/e3_try
    python scripts/run_e3.py --profiles p01 p07 --set platoon.batch_profiles=10 --root artifacts/e3_test
    python scripts/run_e3.py --dry-run                          # units done / missing per variant, stored specs; writes nothing
    bash scripts/run_e3_retry.sh artifacts/e3/logs/run_pooled.out 8 --variants pool lodo_openacc --quiet   # E3-pooled (17.1)

A run set other than the one of the root's other units (e.g. ``--profiles headline`` in a root of the extended set)
belongs in another root (``--root``): the random streams are seeded per batch of the run set, so its units would
differ from the units of the full run set (cf_unc.experiments.e3).

Precedence: config file < --pilot < --set < explicit flags. Exit code 1 if any unit failed (see logs/errors.log); with
--dry-run, if a stored setting conflicts with the config.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

from omegaconf import OmegaConf  # noqa: E402

from cf_unc.experiments.e3 import DEFAULT_CONFIG, RAW, dry_run_e3, load_config, run_e3  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="experiment config (YAML)")
    parser.add_argument("--root", help="output directory (overrides the config)")
    parser.add_argument("--variants", nargs="+", help=f"signal variants ({RAW} and the keys of variants; default: all)")
    parser.add_argument("--models", nargs="+", help="model labels (keys of models; default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, help="seeds (default: those of the config)")
    parser.add_argument("--profiles", nargs="+", help="headline | extended | profile ids (default: run_profiles)")
    parser.add_argument("--device", help="auto | cpu | cuda")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help="dotted overrides of the config")
    parser.add_argument("--pilot", action="store_true", help="reduced experiment (cf_unc.experiments.e3.PILOT)")
    parser.add_argument("--quiet", action="store_true", help="no progress lines on stdout (logs/run.log only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="units done and missing per variant and model, stored settings compared; writes nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.set, pilot=args.pilot)
    if args.root:
        cfg.root = args.root
    if args.variants:
        known = {RAW, *(cfg.get("variants") or {})}
        unknown = sorted(set(args.variants) - known)
        if unknown:
            raise SystemExit(f"unknown variants {unknown}; configured: {sorted(known)}")
        if RAW not in args.variants:
            cfg.checkpoints = OmegaConf.create({})
        cfg.variants = OmegaConf.create({k: v for k, v in (cfg.get("variants") or {}).items() if k in args.variants})
    if args.models:
        unknown = sorted(set(args.models) - set(cfg.models))
        if unknown:
            raise SystemExit(f"unknown models {unknown}; configured: {list(cfg.models)}")
        cfg.models = OmegaConf.create({k: cfg.models[k] for k in args.models})
        for name, spec in (cfg.get("variants") or {}).items():  # per-variant overrides of the models left out go too
            if spec is not None and spec.get("models") is not None:
                spec.models = OmegaConf.create({k: v for k, v in spec.models.items() if k in args.models})
    if args.seeds:
        cfg.seeds = list(args.seeds)
    if args.profiles:
        cfg.run_profiles = args.profiles[0] if len(args.profiles) == 1 and args.profiles[0] in (cfg.get("sets") or {}) \
            else list(args.profiles)  # fmt: skip
    if args.device:
        cfg.device = args.device
    if args.dry_run:
        plan = dry_run_e3(cfg, echo=not args.quiet)
        print(json.dumps(plan, indent=2))
        return 1 if plan["conflicts"] else 0
    status = run_e3(cfg, echo=not args.quiet)
    print(json.dumps(status, indent=2))
    return 1 if status["failures"] or not status["finished"] else 0


if __name__ == "__main__":
    sys.exit(main())
