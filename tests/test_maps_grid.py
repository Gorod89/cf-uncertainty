"""Tests of cf_unc.maps.grid: bin assignment, counts, rare flags, NaN-aware bin means, uncertainty maps.

Everything here is exact arithmetic on constructed inputs (no statistical tolerances).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cf_unc.maps.analysis import load_config
from cf_unc.maps.grid import (
    QUANTITIES,
    StateGrid,
    UncertaintyMap,
    bin_counts,
    bin_means,
    build_map,
    frequencies,
    rare_flags,
)

NAN = np.nan


def _pred(mean, var, alea=None, epi=None, nll=None) -> pd.DataFrame:
    """Prediction table with the columns of cf_unc.maps.io.PRED_COLUMNS (float32 as written by the runner)."""
    n = len(mean)
    f32 = lambda v: np.asarray(v, dtype=np.float32)  # noqa: E731
    return pd.DataFrame(
        {
            "mean": f32(mean),
            "var": f32(var),
            "aleatoric_var": f32(var if alea is None else alea),
            "epistemic_var": f32(np.full(n, NAN) if epi is None else epi),
            "nll": f32(np.zeros(n) if nll is None else nll),
            "crps": f32(np.zeros(n)),
            "covered": np.ones(n, dtype=np.uint8),
            "width": f32(np.ones(n)),
            "pit": f32(np.full(n, 0.5)),
        }
    )


# ------------------------------------------------------------------------------------ grid
def test_grid_defaults_match_config() -> None:
    grid = StateGrid.from_config(load_config())  # configs/maps/default.yaml
    assert grid == StateGrid()
    assert (grid.n_s, grid.n_dv, grid.n_bins, grid.n_cells) == (24, 20, 480, 481)
    np.testing.assert_array_equal(grid.s_edges, 5.0 * np.arange(25))
    np.testing.assert_array_equal(grid.dv_edges, -5.0 + 0.5 * np.arange(21))
    flat = StateGrid.from_config({"s_lo": 0, "s_hi": 120, "s_step": 5, "dv_lo": -5, "dv_hi": 5, "dv_step": 0.5})
    assert flat == grid
    with pytest.raises(ValueError):
        StateGrid(0.0, 10.0, 3.0)  # range not a multiple of the step


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_bin_index_edges_outside_and_nan(dtype: type) -> None:
    grid = StateGrid()
    s = np.array([0.0, 4.999, 5.0, 60.0, 119.99, 120.0, -1e-3, 1e3, NAN, 10.0, 10.0, 10.0], dtype=dtype)
    dv = np.array([-5.0, -4.5, 0.0, -0.5, 4.99, 0.0, 0.0, 0.0, 0.0, 5.0, -5.001, NAN], dtype=dtype)
    expected = [
        0 * 20 + 0,  # both lower edges
        0 * 20 + 1,  # dv on the edge -4.5 -> second dv bin
        1 * 20 + 10,  # s = 5 on an edge -> second s bin; dv = 0 -> bin [0, 0.5)
        12 * 20 + 9,  # s = 60 -> bin 12; dv = -0.5 -> bin [-0.5, 0)
        23 * 20 + 19,  # last bin
        480,  # s = 120: upper edge is outside (right-open)
        480,  # below the grid
        480,  # far above
        480,  # NaN gap
        480,  # dv = 5: upper edge
        480,  # dv below the grid
        480,  # NaN dv
    ]
    bins = grid.bin_index(s, dv)
    assert bins.dtype == np.int64
    np.testing.assert_array_equal(bins, expected)


def test_bin_index_float32_on_decimal_edges() -> None:
    """Edges that are not binary fractions: values equal to (the float32 rounding of) an edge go right."""
    grid = StateGrid(0.0, 1.2, 0.1, -0.3, 0.3, 0.1)
    k = np.arange(grid.n_s)
    decimal = np.array([round(0.1 * i, 10) for i in k])  # 0.0, 0.1, ..., 1.1 as decimal literals
    zeros = np.zeros_like(decimal)
    np.testing.assert_array_equal(grid.bin_index(decimal, zeros) // grid.n_dv, k)  # float64: 0.3 != 3 * 0.1
    s32 = decimal.astype(np.float32)
    np.testing.assert_array_equal(grid.bin_index(s32, zeros.astype(np.float32)) // grid.n_dv, k)
    # the naive floor((x - lo) / step) misplaces some of these values, which is why it is not used
    assert not np.array_equal(np.floor(s32.astype(np.float64) / 0.1).astype(int), k)
    assert not np.array_equal(np.floor(decimal / 0.1).astype(int), k)
    # the next float32 below an edge stays in the lower bin
    below = np.nextafter(s32[1:], np.float32(-np.inf))
    np.testing.assert_array_equal(grid.bin_index(below, np.zeros_like(below)) // grid.n_dv, k[1:] - 1)
    # dv axis with negative decimals, float32
    dv32 = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2], dtype=np.float32)
    np.testing.assert_array_equal(grid.bin_index(np.full(6, 0.05, np.float32), dv32) % grid.n_dv, np.arange(6))
    assert grid.bin_index(np.float32([1.2]), np.float32([0.0]))[0] == grid.n_bins  # upper edge in float32


def test_bin_counts_sum_to_n_and_match_histogram() -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    s = rng.uniform(-10.0, 140.0, n)
    dv = rng.normal(0.0, 3.0, n)
    s[::97] = NAN
    grid = StateGrid()
    bins = grid.bin_index(s, dv)
    counts = bin_counts(bins, grid)
    assert counts.shape == (481,) and counts.dtype == np.int64 and counts.sum() == n
    outside = np.isnan(s) | (s < 0) | (s >= 120) | (dv < -5) | (dv >= 5)
    assert counts[-1] == outside.sum()
    hist, _, _ = np.histogram2d(s[~outside], dv[~outside], bins=[grid.s_edges, grid.dv_edges])  # no value on an edge
    np.testing.assert_array_equal(counts[:-1], hist.ravel())
    freq = frequencies(counts)
    assert freq.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.isnan(frequencies(np.zeros(5))))
    with pytest.raises(ValueError):
        bin_counts(np.array([0, 481]), grid)


# ------------------------------------------------------------------------------ rare flags
SMALL = StateGrid(0.0, 20.0, 5.0, -1.0, 1.0, 1.0)  # 4 x 2 = 8 bins + outside cell (index 8)


def _set(flags: np.ndarray) -> set[int]:
    return set(np.flatnonzero(flags).tolist())


def test_rare_flags_hand_made_density() -> None:
    counts = np.array([40, 0, 6, 6, 200, 12, 3, 25, 8], dtype=float)  # 8 bins + outside; total 300
    assert SMALL.n_cells == counts.size
    flags = rare_flags(counts, freq_threshold=0.05, mass_threshold=0.10)
    # frequency rule: count < 15
    assert _set(flags["rare_freq"]) == {1, 2, 3, 5, 6, 8}
    # mass rule, budget 30 windows: outside (8) -> 8; bin 1 (0) -> 8; bin 6 (3) -> 11; bin 2 (6) -> 17;
    # bin 3 (6) -> 23; bin 5 (12) -> 35 > 30 stops
    assert _set(flags["rare_mass"]) == {1, 2, 3, 6, 8}
    np.testing.assert_allclose(flags["freq_train"], counts / 300.0)
    # tie between bins 2 and 3 (6 windows each) at the budget 18: the lower flat index is taken first
    assert _set(rare_flags(counts, 0.05, 0.06)["rare_mass"]) == {1, 2, 6, 8}
    # budget exactly reached (cumulative 23 <= 23) counts as inside
    assert _set(rare_flags(counts, 0.05, 23 / 300)["rare_mass"]) == {1, 2, 3, 6, 8}


def test_rare_flags_outside_cell_and_empty_bins() -> None:
    counts = np.array([40, 0, 6, 6, 168, 12, 3, 25, 40], dtype=float)  # outside holds 13.3 % > 10 %
    flags = rare_flags(counts, freq_threshold=0.05, mass_threshold=0.10)
    assert _set(flags["rare_mass"]) == {1, 8}  # only the outside cell and the empty bin
    assert _set(rare_flags(counts, freq_threshold=0.0)["rare_freq"]) == {1, 8}  # rare in both rules regardless
    assert rare_flags(np.zeros(9))["rare_mass"].all()


# ------------------------------------------------------------------------------- bin means
def test_bin_means_nan_aware() -> None:
    values = np.array([1.0, 2.0, NAN, 4.0, 5.0, NAN])
    bins = np.array([0, 0, 1, 1, 2, 3])
    means, counts = bin_means(values, bins, 5)
    np.testing.assert_array_equal(counts, [2, 1, 1, 0, 0])
    np.testing.assert_array_equal(means, [1.5, 4.0, 5.0, NAN, NAN])
    means2, counts2 = bin_means(np.vstack([values, 10 * values]), bins, 5)
    np.testing.assert_array_equal(means2, [[1.5, 4.0, 5.0, NAN, NAN], [15.0, 40.0, 50.0, NAN, NAN]])
    np.testing.assert_array_equal(counts2, [counts, counts])
    with pytest.raises(ValueError):
        bin_means(values, bins, 3)


# ------------------------------------------------------------------------------------ maps
def _map_fixture() -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, np.ndarray]]:
    """Windows in cell A (s 12.5, dv 0.25), cell B (s 62.5, dv -2.25) and outside; three seeds."""
    rng = np.random.default_rng(1)
    n_a, n_b, n_o = 40, 10, 5
    s = np.r_[np.full(n_a, 12.5), np.full(n_b, 62.5), np.full(n_o, 130.0)]
    dv = np.r_[np.full(n_a, 0.25), np.full(n_b, -2.25), np.zeros(n_o)]
    n = s.size
    y = rng.normal(0.0, 1.0, n)
    frame = pd.DataFrame({"s": s.astype(np.float32), "dv": dv.astype(np.float32), "y": y.astype(np.float32)})
    arrays: dict[str, np.ndarray] = {k: np.empty((3, n)) for k in ("mean", "alea", "epi", "nll")}
    preds = []
    for k in range(3):
        arrays["mean"][k] = rng.normal(0.0, 0.5, n)
        arrays["alea"][k] = rng.uniform(0.2, 0.4, n)
        arrays["epi"][k] = rng.uniform(0.0, 0.1, n) * (k + 1)
        arrays["nll"][k] = rng.normal(1.0, 0.2, n)
        if k == 2:  # seed 2 did not evaluate the first 8 windows of cell A
            for key in arrays:
                arrays[key][k, :8] = NAN
        var = arrays["alea"][k] + arrays["epi"][k]
        preds.append(_pred(arrays["mean"][k], var, arrays["alea"][k], arrays["epi"][k], arrays["nll"][k]))
    for key in arrays:  # the values as stored (float32)
        arrays[key] = arrays[key].astype(np.float32).astype(np.float64)
    arrays["var"] = np.vstack([p["var"].to_numpy(np.float64) for p in preds])
    arrays["y"] = frame["y"].to_numpy(np.float64)
    return frame, preds, arrays


def test_build_map_seed_mean_and_std() -> None:
    frame, preds, a = _map_fixture()
    grid = StateGrid()
    train = np.arange(481, dtype=float)
    m = build_map(frame, preds, grid, train, kind="model", meta={"model": "toy"})
    cell_a, cell_b = 2 * 20 + 10, 12 * 20 + 5
    assert m.n_seeds == 3 and m.meta == {"kind": "model", "model": "toy"}
    assert (m.count[cell_a], m.count[cell_b], m.count[480]) == (40, 10, 5)  # valid in at least one seed
    assert m.count.sum() == 55
    np.testing.assert_allclose(m.freq_eval, m.count / 55)
    np.testing.assert_allclose(m.freq_train, train / train.sum())
    np.testing.assert_array_equal(m.rare_freq, rare_flags(train)["rare_freq"])
    rows_a = slice(0, 40)
    for q, x in (("epistemic", a["epi"]), ("aleatoric", a["alea"]), ("total", a["var"]), ("nll", a["nll"]),
                 ("mse", (a["y"][None, :] - a["mean"]) ** 2)):  # fmt: skip
        per_seed = np.array([np.nanmean(x[k, rows_a]) for k in range(3)])
        assert m.mean[q][cell_a] == pytest.approx(per_seed.mean(), rel=1e-12), q
        assert m.std[q][cell_a] == pytest.approx(per_seed.std(ddof=1), rel=1e-10), q
    assert np.isnan(m.mean["epistemic"][0])  # cell without windows
    # across-seed decomposition: one row for epistemic / total, so no across-seed std
    ens = build_map(frame, preds, grid, train, kind="seed_ensemble")
    epi = np.var(a["mean"][:, :40], axis=0)  # ddof = 0 over the seeds; the first 8 windows have two seeds
    epi[:8] = np.var(a["mean"][:2, :8], axis=0)
    assert ens.mean["epistemic"][cell_a] == pytest.approx(epi.mean(), rel=1e-10)
    assert np.isnan(ens.std["epistemic"][cell_a])
    alea_seed = np.array([np.nanmean(a["var"][k, :40]) for k in range(3)]).mean()
    assert ens.mean["aleatoric"][cell_a] == pytest.approx(alea_seed, rel=1e-10)
    assert np.isfinite(ens.std["aleatoric"][cell_a])  # per-seed variances: the across-seed std exists
    assert ens.mean["total"][cell_b] == pytest.approx(
        np.mean(np.var(a["mean"][:, 40:50], axis=0) + a["var"][:, 40:50].mean(axis=0)), rel=1e-10
    )


def test_map_round_trip_and_frame(tmp_path) -> None:
    frame, preds, _ = _map_fixture()
    grid = StateGrid()
    m = build_map(frame, preds, grid, np.ones(481), meta={"model": "toy", "dataset": "d"})
    path = m.save(tmp_path / "maps" / "toy.npz")
    back = UncertaintyMap.load(path)
    assert back.grid == grid and back.n_seeds == m.n_seeds and back.meta == m.meta
    for name in ("count", "freq_eval", "freq_train", "rare_freq", "rare_mass"):
        np.testing.assert_array_equal(getattr(back, name), getattr(m, name))
    assert back.quantities == QUANTITIES
    for q in QUANTITIES:
        np.testing.assert_array_equal(back.mean[q], m.mean[q])
        np.testing.assert_array_equal(back.std[q], m.std[q])
    tab = back.to_frame()
    assert len(tab) == grid.n_bins + 1
    assert list(tab["cell"]) == list(range(481))
    last = tab.iloc[-1]
    assert last["i_s"] == -1 and np.isnan(last["s_lo"]) and np.isnan(last["dv_hi"]) and last["count"] == 5
    row = tab.iloc[2 * 20 + 10]
    assert (row["s_lo"], row["s_hi"], row["dv_lo"], row["dv_hi"]) == (10.0, 15.0, 0.0, 0.5)
    assert row["epistemic"] == m.mean["epistemic"][2 * 20 + 10]
    assert {f"{q}_sd" for q in QUANTITIES} <= set(tab.columns)
    masked = back.masked("epistemic", min_count=30)
    assert masked.shape == (24, 20)
    assert np.isfinite(masked[2, 10]) and np.isnan(masked[12, 5])  # 40 windows vs 10 windows
    assert np.isfinite(back.masked("epistemic", min_count=10)[12, 5])
    assert np.isfinite(masked).sum() == 1
