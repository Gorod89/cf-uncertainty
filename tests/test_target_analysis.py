"""Analysis scripts with the target option and the data set ngsim_us101 (signal table, data set order)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cf_unc.report.figures import DATASET_LABEL

REPO = Path(__file__).resolve().parents[1]
ORDER = ("highd", "highd_val", "ngsim", "ngsim_us101", "ngsim_fn", "waymo", "openacc")


def _load_script(name: str) -> Any:
    """Module of ``scripts/<name>.py`` (the scripts directory is not a package)."""
    spec = importlib.util.spec_from_file_location(f"{name}_script", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _windows(with_y_prev: bool) -> pd.DataFrame:
    y = np.array([1.0, 2.0, -1.0, 0.5])
    frame = pd.DataFrame({"event": [0, 0, 1, 1], "s": [10.0, 20.0, 70.0, 5.0], "dv": [0.0, 4.0, -1.0, 0.2],
                          "v": [5.0, 12.0, 30.0, 3.0], "a_prev": [0.0, 0.0, 0.0, 0.0], "y": y})  # fmt: skip
    if with_y_prev:
        frame["y_prev"] = y - np.array([0.1, -0.1, 0.1, np.nan])  # NaN: no earlier target (ignored)
    return frame


def test_signal_uses_the_persistence_of_the_evaluated_target() -> None:
    extra = _load_script("analyze_e1_extra")
    row = extra.signal_row("ngsim_us101", _windows(True), "mean_1s")
    assert (row["target"], row["persistence"]) == ("mean_1s", "y_prev")
    assert row["rmse_persistence"] == pytest.approx(0.1)
    assert row["rmse_zero"] == pytest.approx(np.sqrt(np.mean(np.array([1.0, 4.0, 1.0, 0.25]))))
    assert row["std_a"] == pytest.approx(np.std([1.0, 2.0, -1.0, 0.5])) and row["n_events"] == 2
    old = extra.signal_row("ngsim", _windows(False), "a")  # table written before y_prev existed: a_prev = a[t-1]
    assert (old["target"], old["persistence"]) == ("a", "a_prev")
    assert old["rmse_persistence"] == pytest.approx(np.sqrt(np.mean(np.array([1.0, 4.0, 1.0, 0.25]))))
    assert list(row)[:3] == ["dataset", "target", "persistence"]


def test_ngsim_us101_in_every_data_set_order() -> None:
    order = ("pool_test", *ORDER)  # the pool of the test folds of a pooled training domain first (E6, design.md 16.1)
    assert _load_script("analyze_e1_extra").DATASETS == order
    assert _load_script("summarize_e1").DATASETS == order
    assert [d for d in DATASET_LABEL] == list(order)
