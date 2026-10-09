"""Figures of experiment E1 from the analysis tables.

    python scripts/analyze_e1.py --root artifacts/e1
    python scripts/make_figures_e1.py --root artifacts/e1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cf_unc.report.figures import make_e1_figures

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="artifacts/e1")
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root) if Path(args.root).is_absolute() else ROOT / args.root
    for path in make_e1_figures(root, models=args.models):
        print(path)


if __name__ == "__main__":
    main()
