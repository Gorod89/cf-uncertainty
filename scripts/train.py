"""Train one model and report one-step metrics.

    python scripts/train.py model=deep_ensemble data=synthetic seed=0
    python scripts/train.py -m model=hetero_mdn,lstm_const_sigma seed=0,1,2,3,4
"""

from __future__ import annotations

import json
import warnings

import hydra
from omegaconf import DictConfig

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

from cf_unc.train.run import format_row, run_training  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    result = run_training(cfg)
    print(json.dumps(format_row(result), indent=2))
    print(f"artifacts: {result['out_dir']}")


if __name__ == "__main__":
    main()
