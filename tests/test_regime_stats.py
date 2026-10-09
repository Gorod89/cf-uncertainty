"""Tests of cf_unc.maps.regime_stats: statistics per regime, ordering against the reference regime, NLL comparison.

Data and bootstrap draws use fixed seeds (deterministic); statistical tolerances state their margin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cf_unc.eval.bootstrap import holm
from cf_unc.maps.regime_stats import NLL_COLUMNS, ORDERING_COLUMNS, REGIME_TABLE_COLUMNS
from cf_unc.maps.regime_stats import nll_comparison, regime_ordering, regime_table
from cf_unc.maps.regimes import REGIMES

NAN = np.nan
IDX = {name: i for i, name in enumerate(REGIMES)}
LEVELS = {"steady": 1.0, "free": 1.0, "braking": 2.0, "approaching": 1.5}  # regime means relative to steady


def _regime_data(
    rng: np.random.Generator, n_events: int = 300, per: int = 40, n_seeds: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positive per-window values with known regime means: level x event effect (CV 0.32) x window noise (CV 0.5)
    x seed noise (CV 0.14). Returns x (n_seeds, n), labels (n,), events (n,); no congested / other windows."""
    ev = np.repeat(np.arange(n_events), per)
    names = list(LEVELS)
    lab = np.array([IDX[name] for name in names])[rng.integers(0, len(names), ev.size)]
    level = np.zeros(len(REGIMES))
    for name, value in LEVELS.items():
        level[IDX[name]] = value
    base = level[lab] * rng.gamma(10.0, 0.1, n_events)[ev] * rng.gamma(4.0, 0.25, ev.size)
    x = np.vstack([base * rng.gamma(50.0, 0.02, ev.size) for _ in range(n_seeds)])
    return x, lab, ev


def _seed_mean(x: np.ndarray, rows: np.ndarray) -> float:
    return float(np.mean([np.nanmean(x[k, rows]) for k in range(x.shape[0])]))


def test_regime_ordering_known_ratios_and_holm() -> None:
    rng = np.random.default_rng(0)
    x, lab, ev = _regime_data(rng)
    sq_resid = rng.normal(0.0, np.sqrt(x[0])) ** 2  # squared residuals with the same regime variances
    out = regime_ordering({"var": x, "mse": sq_resid}, lab, ev, "steady", n_boot=1000, seed=1)
    assert list(out.columns) == ORDERING_COLUMNS
    assert len(out) == 2 * (len(REGIMES) - 1) and set(out["reference"]) == {"steady"}
    var = out[out["quantity"] == "var"].set_index("regime")
    steady = lab == IDX["steady"]
    for name in ("free", "braking", "approaching"):
        expected = _seed_mean(x, lab == IDX[name]) / _seed_mean(x, steady)  # ratio of the seed-averaged means
        assert var.loc[name, "estimate"] == pytest.approx(expected, rel=1e-12)
    # true ratio 2: the interval covers it (se 1.5 %: the event effects are shared by the regimes of an event
    # and cancel, because both regimes come from the same resampled events), excludes 1, p at its floor 2/1001
    brk = var.loc["braking"]
    assert brk["lo"] < 2.0 < brk["hi"] and brk["lo"] > 1.6
    assert brk["pvalue"] < 0.01 and brk["reject"]
    # true ratio 1: the interval covers 1 and the p-value is large (uniform under the null; 0.37 with this seed)
    free = var.loc["free"]
    assert free["lo"] < 1.0 < free["hi"] and free["pvalue"] > 0.2 and not free["reject"]
    # regimes without windows give NaN and stay out of the Holm family
    assert np.isnan(var.loc["congested", "estimate"]) and np.isnan(var.loc["congested", "pvalue"])
    assert var.loc["congested", "n_windows"] == 0 and not var.loc["congested", "reject"]
    # Holm over every row of the call (both quantities)
    adjusted, reject = holm(out["pvalue"].to_numpy(dtype=float))
    np.testing.assert_allclose(out["p_holm"].to_numpy(dtype=float), adjusted, equal_nan=True)
    np.testing.assert_array_equal(out["reject"].to_numpy(dtype=bool), reject)
    mse = out[out["quantity"] == "mse"].set_index("regime")
    assert mse.loc["braking", "lo"] > 1.3 and mse.loc["braking", "p_holm"] < 0.05  # noisier: chi-square(1) residuals
    assert mse.loc["free", "lo"] < 1.0 < mse.loc["free", "hi"]
    # a 1-D array is one seed; the reference must be a regime name
    one = regime_ordering({"var": x[0]}, lab, ev, n_boot=200, seed=1)
    assert set(one["n_seeds"].dropna()) <= {0, 1}
    with pytest.raises(ValueError):
        regime_ordering({"var": x}, lab, ev, reference="cruising", n_boot=10)


def test_collapsed_seeds_give_the_same_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    import cf_unc.maps.regime_stats as RS

    rng = np.random.default_rng(6)
    x, lab, ev = _regime_data(rng, n_events=80, per=20)
    x[:, ::7] = NAN  # the same missing windows for every seed: collapsed to the seed mean
    fast = regime_ordering({"var": x}, lab, ev, n_boot=300, seed=2)
    monkeypatch.setattr(RS, "_collapse", lambda a: a)
    slow = regime_ordering({"var": x}, lab, ev, n_boot=300, seed=2)
    for col in ("estimate", "lo", "hi", "se", "pvalue"):
        np.testing.assert_allclose(fast[col].to_numpy(float), slow[col].to_numpy(float), rtol=1e-10, equal_nan=True)


def _pred_frames(y: np.ndarray, var: np.ndarray, means: np.ndarray) -> list[pd.DataFrame]:
    """Prediction tables (float32, covered uint8) of Gaussian predictions; rows with NaN mean are missing."""
    frames = []
    for k in range(var.shape[0]):
        sd = np.sqrt(var[k])
        z = (y - means[k]) / sd
        covered = np.nan_to_num(np.abs(z) <= 1.6448536269514722, nan=0.0).astype(np.uint8)
        covered[np.isnan(means[k])] = 0  # the runner stores 0 for rows it did not evaluate
        frames.append(
            pd.DataFrame(
                {
                    "mean": means[k].astype(np.float32),
                    "var": np.where(np.isnan(means[k]), NAN, var[k]).astype(np.float32),
                    "aleatoric_var": (0.9 * var[k]).astype(np.float32),
                    "epistemic_var": (0.1 * var[k]).astype(np.float32),
                    "nll": (0.5 * z**2 + 0.5 * np.log(2 * np.pi * var[k])).astype(np.float32),
                    "crps": np.abs(z).astype(np.float32),
                    "covered": covered,
                    "width": (2 * 1.6448536269514722 * sd).astype(np.float32),
                    "pit": np.full(y.size, 0.5, np.float32),
                }
            )
        )
    return frames


def test_regime_table_matches_direct_computation() -> None:
    rng = np.random.default_rng(2)
    x, lab, ev = _regime_data(rng, n_events=120, per=30)
    n = ev.size
    y = rng.normal(0.0, 1.0, n)
    means = rng.normal(0.0, 0.3, (3, n))
    means[2, :300] = NAN  # seed 2 did not evaluate the first 300 windows
    frames = _pred_frames(y, x, means)
    windows = pd.DataFrame(
        {"y": y.astype(np.float32), "event": ev.astype(np.int32), "driver": (ev // 3).astype(np.int32)}
    )
    tab = regime_table(windows, frames, lab, n_boot=500, seed=0)
    assert list(tab.columns) == REGIME_TABLE_COLUMNS
    tab = tab.set_index("regime")
    assert list(tab.index) == ["all", *REGIMES]
    np.testing.assert_allclose(tab.loc[list(REGIMES), "share"], np.bincount(lab, minlength=6) / n)
    assert tab.loc["all", "share"] == 1.0 and tab.loc["all", "n_windows"] == n and tab.loc["all", "n_events"] == 120
    stored = {name: np.vstack([f[name].to_numpy(np.float64) for f in frames]) for name in frames[0].columns}
    ok = np.isfinite(stored["mean"]) & np.isfinite(stored["var"])
    covered = np.where(ok, stored["covered"], NAN)  # rows without a prediction do not count as misses
    y64 = windows["y"].to_numpy(np.float64)
    for name in ("braking", "steady", "all"):
        rows = np.ones(n, bool) if name == "all" else lab == IDX[name]
        assert tab.loc[name, "var"] == pytest.approx(_seed_mean(stored["var"], rows), rel=1e-10)
        assert tab.loc[name, "aleatoric"] == pytest.approx(_seed_mean(stored["aleatoric_var"], rows), rel=1e-10)
        assert tab.loc[name, "epistemic"] == pytest.approx(_seed_mean(stored["epistemic_var"], rows), rel=1e-10)
        assert tab.loc[name, "nll"] == pytest.approx(_seed_mean(stored["nll"], rows), rel=1e-10)
        assert tab.loc[name, "crps"] == pytest.approx(_seed_mean(stored["crps"], rows), rel=1e-10)
        assert tab.loc[name, "picp"] == pytest.approx(_seed_mean(covered, rows), rel=1e-10)
        assert tab.loc[name, "mse"] == pytest.approx(_seed_mean((y64 - stored["mean"]) ** 2, rows), rel=1e-10)
        m = stored["mean"][:, rows]
        epi_seed = np.where(np.isnan(m[2]), np.nanvar(m[:2], axis=0), np.var(m, axis=0))  # ddof = 0, NaN-aware
        assert tab.loc[name, "epistemic_seed"] == pytest.approx(epi_seed.mean(), rel=1e-9)
        for q in ("var", "mse", "nll"):
            assert tab.loc[name, f"{q}_lo"] < tab.loc[name, q] < tab.loc[name, f"{q}_hi"]
    assert tab.loc["congested", "n_windows"] == 0 and tab.loc["congested", "share"] == 0.0
    assert np.isnan(tab.loc["congested", "var"]) and np.isnan(tab.loc["congested", "var_lo"])
    assert (tab["n_seeds"] == 3).all()
    by_driver = regime_table(windows, frames, lab, n_boot=200, seed=0, unit="driver").set_index("regime")
    assert by_driver.loc["all", "n_events"] == 40 and by_driver.loc["all", "var"] == tab.loc["all", "var"]
    single = regime_table(windows, frames[:1], lab, n_boot=200, seed=0).set_index("regime")
    assert single["epistemic_seed"].isna().all() and (single["n_seeds"] == 1).all()


def test_nll_comparison_sign_convention_and_overlap() -> None:
    rng = np.random.default_rng(4)
    ev = np.repeat(np.arange(200), 25)
    n = ev.size
    lab = rng.integers(0, len(REGIMES), n)
    event_level = rng.normal(0.0, 0.5, 200)[ev]  # large between-event spread of the NLL
    nll_b = 1.0 + event_level + rng.normal(0.0, 0.3, (2, n))  # model b: two seeds
    nll_a = nll_b.mean(axis=0) - 0.05 + rng.normal(0.0, 0.05, (3, n))  # model a: 0.05 nats better, three seeds
    out = nll_comparison(nll_a, nll_b, ev, lab, n_boot=1000, seed=0)
    assert list(out.columns) == NLL_COLUMNS
    out = out.set_index("group")
    assert list(out.index) == ["all", *REGIMES]
    row = out.loc["all"]
    assert row["diff"] == pytest.approx(nll_a.mean(axis=0).mean() - nll_b.mean(axis=0).mean(), rel=1e-12)
    assert row["nll_a"] == pytest.approx(nll_a.mean(), rel=1e-12) and row["nll_b"] == pytest.approx(nll_b.mean())
    # a - b < 0: a is better; the paired interval excludes 0 ...
    assert row["diff"] < 0 and row["diff_hi"] < 0 and row["pvalue"] < 0.01
    # ... while the separate intervals overlap (half widths ~0.07 from the between-event spread > 0.05)
    assert bool(row["ci_overlap"])
    assert (row["n_windows"], row["n_events"], row["n_seeds_a"], row["n_seeds_b"]) == (n, 200, 3, 2)
    for name in ("braking", "free"):
        rows = lab == IDX[name]
        assert out.loc[name, "diff"] == pytest.approx(nll_a[:, rows].mean() - nll_b[:, rows].mean(), rel=1e-10)
        assert out.loc[name, "diff_hi"] < 0
    swapped = nll_comparison(nll_b, nll_a, ev, lab, n_boot=1000, seed=0).set_index("group")
    assert swapped.loc["all", "diff"] == pytest.approx(-row["diff"], rel=1e-12)
    assert swapped.loc["all", "diff_lo"] == pytest.approx(-row["diff_hi"], rel=1e-9)  # same draws
    far = nll_comparison(nll_b.mean(axis=0) - 1.0, nll_b, ev, None, n_boot=500, seed=0)
    assert list(far["group"]) == ["all"]
    assert far.loc[0, "ci_overlap"] is not None and not bool(far.loc[0, "ci_overlap"])  # 1 nats apart
    # windows where either model has no value are dropped (paired by window); seeds are averaged NaN-aware
    a_nan = nll_a.copy()
    a_nan[:, :100] = NAN
    a_nan[0, 100:200] = NAN
    res = nll_comparison(a_nan, nll_b, ev, None, n_boot=200, seed=0).loc[0]
    keep = np.arange(n) >= 100
    abar = np.nanmean(a_nan[:, keep], axis=0)
    assert res["n_windows"] == n - 100 and res["n_events"] == 196
    assert res["nll_a"] == pytest.approx(abar.mean(), rel=1e-12)
    assert res["nll_b"] == pytest.approx(nll_b[:, keep].mean(), rel=1e-12)
