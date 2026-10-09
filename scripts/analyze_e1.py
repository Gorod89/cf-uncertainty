"""Analysis of experiment E1 (docs/design.md, section 10): uncertainty maps, dataset shift (H3.1), regimes (H3.3).

Reads the prediction tables written by ``scripts/run_e1.py`` and writes ``<root>/tables/*.csv`` and
``<root>/maps/``; prints the H3.1 summary and the H3.3 table at the end.

    python scripts/analyze_e1.py                                       # artifacts/e1, configs/maps/default.yaml
    python scripts/analyze_e1.py --unit driver                         # driver bootstrap (sensitivity analysis)
    python scripts/analyze_e1.py --root artifacts/e1_pilot --models hetero_mdn lstm_const_sigma --n-boot 200
    python scripts/analyze_e1.py --root artifacts/e6_lodo_ngsim --config configs/experiment/e6_lodo_ngsim.yaml \
        --in-domain pool_test                                          # E6: pooled training domain (design.md 16.1)

``--config`` takes a maps config or an experiment config (recognised by its keys ``base``, ``root``, ``train``,
``eval`` or ``config_paths``): an experiment config gives its maps config (``config_paths.maps``) and its training
data sets (``train.data``). The training data sets decide the pairs of the shift analysis: a data set of the training
domain gets no shift pair (``cf_unc.maps.analysis.pair_roles``); without an experiment config or ``--train-data`` they
are read from the root (``train_density.npz`` of a pooled root), else they are the in-domain set's sources.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cf_unc.maps.analysis import _read_yaml, analyze_e1, format_report, load_config

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_KEYS = ("base", "root", "train", "eval", "config_paths")  # keys of an experiment config, absent in a maps config


def _resolve(path: str) -> Path:
    """A relative path is taken from the working directory, else from the repository root."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return ROOT / p


def read_config(path: Path) -> tuple[dict[str, Any], list[str] | None]:
    """Maps config and training data sets of ``--config``: a maps config as it is (no training data sets); an
    experiment config (``cf_unc.experiments.e1.load_config``, ``base`` resolved) gives the maps config of its
    ``config_paths.maps`` and the data sets of ``train.data``."""
    raw = _read_yaml(path)
    if not any(k in raw for k in EXPERIMENT_KEYS):
        return load_config(path), None
    from cf_unc.experiments.e1 import load_config as load_experiment, train_datasets

    exp = load_experiment(path)
    maps = (exp.get("config_paths") or {}).get("maps") or "configs/maps/default.yaml"
    return load_config(_resolve(str(maps))), train_datasets(exp) or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="artifacts/e1", help="experiment directory (windows/, preds/, ...)")
    parser.add_argument("--config", default="configs/maps/default.yaml",
                        help="maps config (thresholds, bootstrap) or experiment config (its maps config and train.data)")  # fmt: skip
    parser.add_argument("--models", nargs="+", default=None, help="models to analyse (default: all available)")
    parser.add_argument("--datasets", nargs="+", default=None, help="data sets to analyse (default: all available)")
    parser.add_argument("--unit", choices=("event", "driver"), default=None, help="bootstrap unit (default: config)")
    parser.add_argument("--in-domain", default="highd", help="reference data set of the shift ratios (E6: pool_test)")
    parser.add_argument(
        "--placebo", nargs="*", default=None,
        help="data sets whose pairs are placebos, not counted for H3.1 (default: same source as the in-domain "
        "set, e.g. highd_val; the flag without names: none)",
    )  # fmt: skip
    parser.add_argument(
        "--cross-check", nargs="*", default=None,
        help="data sets that repeat a counted data set (reported, not counted for H3.1; default: config)",
    )  # fmt: skip
    parser.add_argument(
        "--train-data", nargs="+", default=None,
        help="training data sets of the root, which get no shift pair (default: train.data of an experiment --config, "
        "else the pool of train_density.npz, else the in-domain sources)",
    )  # fmt: skip
    parser.add_argument(
        "--v0", type=float, default=None,
        help="free speed [m/s] of the regime 'free' (default: regimes.v0 of the maps config, else params.v0 of "
        "<root>/idm.json, which for an E6 root is the IDM calibrated on the pooled training events; pass the highD "
        "value of artifacts/e1m/idm.json for regimes defined as in E1-E5)",
    )  # fmt: skip
    parser.add_argument("--n-boot", type=int, default=None, help="bootstrap replicates (default: config)")
    parser.add_argument("--quiet", action="store_true", help="no progress lines")
    args = parser.parse_args(argv)

    cfg, train_data = read_config(_resolve(args.config))
    if args.train_data is not None:
        train_data = list(args.train_data)
    if args.n_boot is not None:
        cfg["bootstrap"]["n_boot"] = int(args.n_boot)
    if args.v0 is not None:
        cfg["regimes"]["v0"] = float(args.v0)
    tables = analyze_e1(
        _resolve(args.root),
        cfg,
        in_domain=args.in_domain,
        models=args.models,
        datasets=args.datasets,
        unit=args.unit,
        placebo=args.placebo,
        cross_check=args.cross_check,
        train_data=train_data,
        verbose=not args.quiet,
    )
    print()
    print(format_report(tables))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
