"""Rollout tables (cf_unc.rollout.tables, docs/design.md, section 11.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cf_unc.eval.metrics import crps_samples
from cf_unc.rollout import tables
from cf_unc.rollout.engine import RolloutResult


def make_result(s: np.ndarray, truth_s: np.ndarray, first: np.ndarray, horizons_s=(1.0, 3.0)) -> RolloutResult:
    """RolloutResult from gaps s (E, R, n_h), truths (E, n_h) and first collision steps (E, R)."""
    e, r, n_h = s.shape
    return RolloutResult(
        s=s, v=s * 0.5 + 10.0, first_collision=first, s_min=s.min(axis=2), truth_s=truth_s, truth_v=truth_s * 0.5 + 10.0,
        t0=np.full(e, 29), horizons_s=np.asarray(horizons_s, dtype=float),
        horizon_steps=np.round(np.asarray(horizons_s) / 0.1).astype(np.int64), n_rollouts=r, hold_steps=1,
        sampler_mode="per_step", seed=0,
    )  # fmt: skip


def events_meta(codes: list[int]) -> pd.DataFrame:
    n = len(codes)
    return pd.DataFrame({
        "event": np.asarray(codes, dtype=np.int32), "event_id": [f"ev{c}" for c in codes],
        "driver": np.arange(n, dtype=np.int32), "driver_name": [f"d/{c}" for c in codes], "subset": ["all"] * n,
        "fold": np.zeros(n, dtype=np.int8), "t0": np.full(n, 29, dtype=np.int32), "order": np.arange(n, dtype=np.int32)[::-1],
    })  # fmt: skip


def test_quantile_conventions_rank_and_moments() -> None:
    rng = np.random.default_rng(0)
    sample = rng.permutation(np.arange(1.0, 20.0))  # 1 .. 19 in random order, S = 19
    s = np.stack([sample, sample + 100.0], axis=-1)[None]  # (1, 19, 2): second horizon shifted by 100
    res = make_result(s, truth_s=np.array([[10.0, 150.0]]), first=np.full((1, 19), -1))
    frame = tables.summarize_rollout(res, events_meta([7]))
    assert list(frame.columns) == list(tables.ROLLOUT_COLUMNS) and len(frame) == 2
    row = frame.iloc[0]
    # (S + 1) plotting positions: level p sits at order statistic p (S + 1) = 20 p
    assert [row.s_q05, row.s_q25, row.s_q50, row.s_q75, row.s_q95] == [1.0, 5.0, 10.0, 15.0, 19.0]
    # linear interpolation: order statistic 1 + p (S - 1)
    assert row.s_lin05 == pytest.approx(1.9) and row.s_lin95 == pytest.approx(18.1)
    assert row["rank"] == 10 and frame.iloc[1]["rank"] == 19  # rollouts with gap <= truth
    assert row.s_mean == 10.0 and row.s_std == pytest.approx(np.std(np.arange(1.0, 20.0), ddof=1))
    assert (row.s_min, row.s_max) == (1.0, 19.0) and frame.iloc[1].s_q50 == 110.0
    assert row.v_mean == pytest.approx(15.0) and row.n_rollouts == 19
    assert frame["event"].tolist() == [7, 7] and frame["horizon_s"].tolist() == [1.0, 3.0]


def test_crps_and_collided_share() -> None:
    rng = np.random.default_rng(1)
    e, r = 3, 50
    s = rng.normal(20.0, 3.0, size=(e, r, 2))
    truth = rng.normal(20.0, 3.0, size=(e, 2))
    first = np.full((e, r), -1)
    first[0, :5] = [5, 10, 11, 30, 31]  # horizons at steps 10 and 30
    first[2, :] = 1
    frame = tables.summarize_rollout(make_result(s, truth, first), events_meta([4, 0, 9]))
    expected = crps_samples(np.moveaxis(s, 2, 1).reshape(-1, r), truth.reshape(-1), fair=True)
    np.testing.assert_allclose(frame["crps_s"], expected, rtol=1e-12)
    np.testing.assert_allclose(frame["collided"], [2 / r, 4 / r, 0.0, 0.0, 1.0, 1.0], rtol=1e-6)
    assert frame["event"].tolist() == [4, 4, 0, 0, 9, 9]  # events in the order of the events table
    np.testing.assert_allclose(frame["truth_s"], truth.reshape(-1))


def test_round_trip_and_available(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    s = rng.normal(30.0, 2.0, size=(2, 8, 2))
    frame = tables.summarize_rollout(make_result(s, rng.normal(30.0, 2.0, (2, 2)), np.full((2, 8), -1)), events_meta([3, 1]))
    path = tables.save_rollout(tmp_path, "a", "hetero_mdn", 2, "highd", frame)
    assert path == tables.rollout_path(tmp_path, "a", "hetero_mdn", 2, "highd") and not path.with_suffix(".tmp").exists()
    back = tables.load_rollout(tmp_path, "a", "hetero_mdn", 2, "highd")
    pd.testing.assert_frame_equal(back, frame, check_exact=True)
    assert back["event"].dtype == np.int32 and back["rank"].dtype == np.int32 and back["horizon_s"].dtype == np.float32
    assert back["collided"].dtype == np.float32 and back["s_q50"].dtype == np.float64
    ev = events_meta([3, 1])
    tables.save_events_table(tmp_path, "a", "highd", ev)
    pd.testing.assert_frame_equal(tables.load_events_table(tmp_path, "a", "highd"), ev[list(tables.EVENT_COLUMNS)],
                                  check_dtype=False)  # fmt: skip
    tables.save_rollout(tmp_path, "a", "gp_residual_idm", 0, "ngsim", frame)
    listing = tables.available(tmp_path, "a")
    assert listing[["model", "seed", "dataset"]].values.tolist() == [["gp_residual_idm", 0, "ngsim"], ["hetero_mdn", 2, "highd"]]
    assert tables.available(tmp_path, "mean_1s").empty
    with pytest.raises(ValueError, match="lacks columns"):
        tables.save_rollout(tmp_path, "a", "x", 0, "y", frame.drop(columns=["rank"]))
    with pytest.raises(ValueError, match="rows"):
        tables.summarize_rollout(make_result(s, np.zeros((2, 2)), np.full((2, 8), -1)), events_meta([1]))
