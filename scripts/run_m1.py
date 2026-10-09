"""Milestone M1 benchmark: train every model family on synthetic StochasticIDM data and tabulate one-step metrics.

    python scripts/run_m1.py                          # all models, data=synthetic, seed 0
    python scripts/run_m1.py --data synthetic_hetero --models hetero_mdn deep_ensemble --seeds 0 1
    python scripts/run_m1.py --override stride=3 train.max_epochs=20
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

from hydra import compose, initialize_config_dir  # noqa: E402

from cf_unc.train.run import ROOT, format_row, run_training, write_json  # noqa: E402

MODELS = (
    "idm_white_noise",
    "lstm_const_sigma",
    "hetero_mdn",
    "hetero_mdn_mix3",
    "deep_ensemble",
    "mc_dropout",
    "gp_residual_idm",
    "quantile_pidl",
    "diffusion_cf",
)
COLUMNS = (
    "model",
    "seed",
    "nll",
    "crps",
    "picp90",
    "width90",
    "rmse_vs_truth",
    "aleatoric",
    "epistemic",
    "aleatoric_true",
    "calib_err",
    "fit_s",
)


def fmt(value: object) -> str:
    if isinstance(value, float):
        return "-" if math.isnan(value) else f"{value:.4g}"
    return str(value)


def markdown(rows: list[dict]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(fmt(row[c]) for c in COLUMNS) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="synthetic")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--override", nargs="*", default=[], help="extra hydra overrides, e.g. stride=3 device=cpu")
    parser.add_argument("--out", default="artifacts/m1")
    args = parser.parse_args()

    out_dir = ROOT / args.out / args.data
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    bundles: dict = {}
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        for seed in args.seeds:
            for model in args.models:
                # the compose API has no hydra runtime, so the label is given explicitly
                overrides = [f"model={model}", f"model_label={model}", f"data={args.data}", f"seed={seed}", *args.override]
                cfg = compose(config_name="config", overrides=overrides)
                result = run_training(cfg, bundles=bundles)
                row = format_row(result)
                rows.append(row)
                print(f"[{model} seed={seed}] " + "  ".join(f"{c}={fmt(row[c])}" for c in COLUMNS[2:]), flush=True)
                write_json(out_dir / "summary.json", rows)

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    table = markdown(rows)
    Path(out_dir / "summary.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\nwritten: {out_dir}")


if __name__ == "__main__":
    main()
