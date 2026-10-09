"""Conformal calibration, ACI and the H3.4 verdict (cf_unc.rollout.calibration, docs/design.md, section 11.4)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from cf_unc.rollout import calibration as cal


# ------------------------------------------------------------------------------------ split conformal
def test_conformal_quantile_hand_computed() -> None:
    s = [3.0, 1.0, 2.0, 5.0, 4.0]
    assert cal.conformal_quantile(s, 0.1) == math.inf  # rank ceil(6 * 0.9) = 6 > 5
    assert cal.conformal_quantile(s, 0.2) == 5.0  # rank ceil(4.8) = 5
    assert cal.conformal_quantile(s, 0.5) == 3.0  # rank ceil(3.0) = 3 exactly (no rounding up to 4)
    assert cal.conformal_quantile(np.arange(1.0, 20.0), 0.1) == 18.0  # n = 19: rank 20 * 0.9 = 18
    assert cal.conformal_quantile(np.arange(1.0, 100.0), 0.1) == 90.0  # n = 99: rank 90
    assert cal.conformal_quantile(np.arange(1.0, 101.0), 0.1) == 91.0  # n = 100: rank ceil(90.9) = 91
    assert cal.conformal_quantile([np.nan, 3.0, 1.0, np.nan, 2.0, 5.0, 4.0], 0.2) == 5.0  # NaN ignored
    assert cal.conformal_quantile([], 0.1) == math.inf and cal.static_correction([2.0, 1.0], 0.5) == 2.0
    with pytest.raises(ValueError):
        cal.conformal_quantile(s, 1.0)


def test_split_conformal_coverage_on_exchangeable_data() -> None:
    rng = np.random.default_rng(0)

    def draw(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = rng.normal(0.0, 1.0, n)
        sd = rng.uniform(0.2, 2.0, n)  # heteroscedastic truth
        return mu, (0.5 * sd) ** 2, mu + sd * rng.standard_normal(n)  # the model is over-confident by 2x

    n_cal, n_eval = 2000, 20_000
    m_c, v_c, y_c = draw(n_cal)
    m_e, v_e, y_e = draw(n_eval)
    raw = np.mean(np.abs(y_e - m_e) <= norm.ppf(0.95) * np.sqrt(v_e))
    assert raw < 0.65  # 90 % interval of the over-confident law
    # coverage given the calibration set is Beta distributed (sd sqrt(0.09 / n_cal)); plus binomial noise
    margin = 4 * math.sqrt(0.09 * (1 / n_cal + 1 / n_eval))
    for score in cal.SCORES:
        out = cal.split_conformal_onestep(m_c, v_c, y_c, m_e, v_e, y_e, alpha=0.1, score=score)
        assert abs(np.mean(out["covered"]) - 0.90) < margin, score
        assert out["n_cal"] == n_cal
        np.testing.assert_array_equal(out["covered"], (y_e >= out["lo"]) & (y_e <= out["hi"]))
        half = out["q"] if score == "absolute" else out["q"] * np.sqrt(v_e)
        np.testing.assert_allclose(out["width"], 2 * half)
    # NaN rows: ignored in the calibration set, NaN in the evaluated set
    m_c[:10] = np.nan
    v_e[:5] = np.nan
    out = cal.split_conformal_onestep(m_c, v_c, y_c, m_e, v_e, y_e, alpha=0.1, score="normalised")
    assert out["n_cal"] == n_cal - 10 and np.isnan(out["covered"][:5]).all() and np.isnan(out["width"][:5]).all()
    assert np.isfinite(out["covered"][5:]).all()


def test_onestep_scores_zero_sigma() -> None:
    s = cal.onestep_scores([0.0, 0.0, 1.0], [0.0, 0.0, 4.0], [0.0, 1.0, 3.0], "normalised")
    np.testing.assert_array_equal(s, [0.0, np.inf, 1.0])
    with pytest.raises(ValueError):
        cal.onestep_scores([0.0], [1.0], [0.0], "relative")


# --------------------------------------------------------------------------------- rollout intervals
def test_score_and_corrected_interval_identity() -> None:
    rng = np.random.default_rng(1)
    n = 200_000
    mid = rng.normal(30.0, 10.0, n)
    d_lo = rng.exponential(2.0, n) * rng.choice([1.0, 1e-4], n, p=[0.9, 0.1])  # some below the 1 mm floor
    d_hi = rng.exponential(5.0, n)  # asymmetric
    lo, hi = mid - d_lo, mid + d_hi
    y = mid + rng.normal(0.0, 6.0, n)
    q = rng.uniform(-0.95, 3.0, n)
    s = cal.interval_score(y, lo, mid, hi)
    a, b = cal.corrected_interval(lo, mid, hi, q)
    np.testing.assert_array_equal((a <= y) & (y <= b), s <= q)  # y inside iff S <= Q
    np.testing.assert_array_equal((lo <= y) & (y <= hi), s <= 0.0)  # raw interval: Q = 0
    assert cal.interval_score(10.0, 10.0, 12.0, 20.0) == 0.0 and cal.interval_score(20.0, 10.0, 12.0, 20.0) == 0.0
    assert cal.interval_score(8.0, 10.0, 12.0, 20.0) == pytest.approx(1.0)  # one lower half-width below
    assert cal.interval_score(24.0, 10.0, 12.0, 20.0) == pytest.approx(0.5)  # half an upper half-width above
    assert cal.interval_score(12.0, 10.0, 12.0, 20.0) == pytest.approx(-1.0)  # the median
    assert cal.interval_score(10.0005, 10.0, 10.0, 10.0) == pytest.approx(0.5)  # half-widths floored at 1 mm
    assert np.isnan(cal.interval_score(np.nan, 1.0, 2.0, 3.0))


# ------------------------------------------------------------------------------------------------ ACI
def test_aci_quantile_levels() -> None:
    s = [1.0, 2.0, 3.0, 4.0]
    assert cal.aci_quantile(s, 0.5, n_min=1) == 2.0 and cal.aci_quantile(s, 0.51, n_min=1) == 3.0
    assert cal.aci_quantile(s, 1.7, n_min=1) == 4.0  # level clipped to 1: the largest score, finite
    assert cal.aci_quantile(s, -0.2, n_min=1) == 1.0  # level clipped to 0: the smallest score
    assert cal.aci_quantile(s, 0.9, n_min=5) == 0.0  # fewer than n_min scores: no correction


def test_aci_warm_up_and_update() -> None:
    scores = np.linspace(-1.0, 1.0, 30)
    out = cal.aci_run(scores, alpha=0.1, gamma=0.05, n_min=20)
    assert (out["q"][:20] == 0.0).all() and out["q"][20] != 0.0
    np.testing.assert_array_equal(out["covered"][:20], (scores[:20] <= 0.0).astype(float))
    err = 1.0 - out["covered"]
    np.testing.assert_allclose(out["alpha_t"][1:], 0.1 + 0.05 * np.cumsum(0.1 - err)[:-1])  # the recursion
    assert out["state"].n_events == 30 and np.array_equal(out["state"].history, np.sort(scores))
    nan = cal.aci_run(np.array([np.nan, 0.5]), 0.1, 0.05, n_min=1, history=[0.0, 1.0])
    assert np.isnan(nan["covered"][0]) and nan["alpha_t"][1] == 0.1 and nan["state"].n_events == 1


def drifting_scores(n: int = 4000, seed: int = 7) -> np.ndarray:
    """Signed scores whose scale grows linearly from 1 to 5 over n events."""
    rng = np.random.default_rng(seed)
    return (1.0 + 4.0 * np.arange(n) / (n - 1)) * rng.standard_normal(n)


def test_aci_coverage_on_drifting_series() -> None:
    """Scale x5 over 4000 events: static and frozen corrections lose coverage, online ACI keeps it.

    Over 30 seeds (see the M3 report): static 0.60-0.67 and frozen 0.61-0.69 on the last 1000 events, online
    0.898-0.901 over the evaluation events and >= 0.882 in every window of 500.
    """
    s = drifting_scores()
    n_cal = 500
    q = cal.static_correction(s[:n_cal], 0.1)
    assert np.mean(s[-1000:] <= q) < 0.75
    run = cal.aci_run(s[:n_cal], alpha=0.1, gamma=0.01)
    online = cal.aci_run(s[n_cal:], alpha=0.1, gamma=0.01, state=run["state"])
    covered = online["covered"]
    assert abs(covered.mean() - 0.90) <= 0.02
    local = np.convolve(covered, np.ones(500) / 500, mode="valid")
    assert local.min() > 0.84
    frozen = cal.aci_run(s[n_cal:], alpha=0.1, gamma=0.01, state=run["state"], frozen=True)
    assert np.unique(frozen["q"]).size == 1 and frozen["state"].n_events == run["state"].n_events
    assert np.mean(frozen["covered"][-1000:]) < 0.75
    assert online["q"][-1] > 3 * frozen["q"][0]  # the online correction followed the scale


def test_aci_continuation_equals_one_run() -> None:
    s = drifting_scores(n=1200, seed=3)
    whole = cal.aci_run(s, alpha=0.1, gamma=0.02, n_min=20)
    first = cal.aci_run(s[:700], alpha=0.1, gamma=0.02, n_min=20)
    saved = json.loads(json.dumps(first["state"].to_dict()))  # a state survives a JSON round trip
    second = cal.aci_run(s[700:], alpha=0.1, gamma=0.02, n_min=20, state=cal.ACIState.from_dict(saved))
    for key in ("q", "alpha_t", "covered"):
        np.testing.assert_array_equal(np.concatenate([first[key], second[key]]), whole[key])
    assert second["state"].to_dict() == whole["state"].to_dict()
    assert first["state"].n_events == 700  # the input state is not modified by the continuation
    with pytest.raises(ValueError):
        cal.aci_run(s, alpha=0.1, gamma=0.05, state=first["state"])  # other parameters than the saved run
    with pytest.raises(ValueError):
        cal.aci_run(s, alpha=0.1, gamma=0.02, history=[1.0], state=first["state"])


# ------------------------------------------------------------------------------------ coverage tables
def rollout_tables(n_events: int, seeds: list[int], spread: float, rng: np.random.Generator) -> dict[int, pd.DataFrame]:
    """Rollout tables whose truth is N(30, (spread * 2)^2) while the rollouts say N(30, 2^2) (horizons 1 and 5 s)."""
    out = {}
    z = norm.ppf([0.05, 0.25, 0.5, 0.75, 0.95])
    for seed in seeds:
        rows = []
        for h, sd in ((1.0, 2.0), (5.0, 4.0)):
            truth = 30.0 + spread * sd * rng.standard_normal(n_events)
            q = 30.0 + sd * z
            rows.append(pd.DataFrame({
                "event": np.arange(n_events, dtype=np.int32), "horizon_s": np.float32(h), "truth_s": truth,
                **{c: np.full(n_events, v) for c, v in zip(("s_q05", "s_q25", "s_q50", "s_q75", "s_q95"), q)},
                "s_lin05": np.full(n_events, q[0] * 1.001), "s_lin95": np.full(n_events, q[-1]),
                "crps_s": np.full(n_events, sd), "collided": np.full(n_events, 0.01, dtype=np.float32),
            }))  # fmt: skip
        out[seed] = pd.concat(rows).sort_values(["event", "horizon_s"], kind="stable").reset_index(drop=True)
    return out


def events(n: int, seed: int) -> pd.DataFrame:
    return pd.DataFrame({"event": np.arange(n), "order": np.random.default_rng(seed).permutation(n)})


def test_coverage_table_raw_static_aci() -> None:
    rng = np.random.default_rng(5)
    n = 600
    cal_t, ev_t = rollout_tables(n, [0, 1, 2], 2.0, rng), rollout_tables(n, [0, 1, 2, 3], 2.0, rng)
    table = cal.coverage_table(ev_t, events(n, 1), cal_t, events(n, 2), alpha=0.1, gammas=(0.01, 0.05), n_boot=300)
    assert list(table.columns) == list(cal.COVERAGE_COLUMNS)
    assert set(table["variant"]) == set(cal.VARIANTS) and len(table) == 2 * (3 + 2 * 2)
    assert (table["n_seeds"] == 3).all() and (table["n_events"] == n).all()  # seeds present in both sets
    raw = table[table.variant == "raw"]
    expected_raw = 2 * norm.cdf(norm.ppf(0.95) / 2.0) - 1  # truth twice as wide as the rollouts: 0.59
    assert (np.abs(raw["picp"] - expected_raw) < 0.05).all()
    margin = 4 * math.sqrt(0.09 / n) + 0.02
    raw_width = raw.set_index("horizon_s")["width_mean"]
    for variant in ("static", "aci_frozen", "aci_online"):
        sel = table[table.variant == variant]
        assert (np.abs(sel["picp"] - 0.90) < margin).all(), variant
        assert (sel["width_mean"].to_numpy() > 1.5 * raw_width.loc[sel["horizon_s"]].to_numpy()).all()  # widened
    assert ((table.picp_lo <= table.picp) & (table.picp <= table.picp_hi)).all()
    assert raw["crps"].notna().all() and table[table.variant != "raw"]["crps"].isna().all()
    assert np.allclose(table["collided"], 0.01)
    assert table[table.variant.str.startswith("aci")]["gamma"].notna().all() and raw["gamma"].isna().all()
    # without calibration tables: raw variants only; the same events are resampled in every call
    alone = cal.coverage_table(ev_t, events(n, 1), n_boot=300)
    assert set(alone["variant"]) == {"raw", "raw_linear"} and (alone["n_seeds"] == 4).all()
    again = cal.coverage_table(ev_t, events(n, 1), n_boot=300)
    pd.testing.assert_frame_equal(alone, again)


# --------------------------------------------------------------------------------------------- H3.4
def test_h34_verdict_branches() -> None:
    good_raw = [0.88, 0.80, 0.70, 0.60]
    good_corr = [0.90, 0.89, 0.88, 0.87]
    v = cal.h34_verdict(0.905, good_raw, good_corr)
    assert v["verdict"] == "supported" and v["part_i"] and v["part_ii"] and v["part_iii"] and v["verdict_ci"] is None
    assert cal.h34_verdict(0.95, good_raw, good_corr)["verdict"] == "inconclusive"  # (i) fails
    assert cal.h34_verdict(0.87, good_raw, good_corr)["part_i"]  # the band is closed
    not_low = [0.86, 0.85, 0.84, 0.83]  # below 0.87 somewhere (not refuted) but never below 0.80
    out = cal.h34_verdict(0.9, not_low, good_corr)
    assert out["verdict"] == "inconclusive" and not out["part_ii"] and not out["refuted_raw"]
    out = cal.h34_verdict(0.9, good_raw, [0.9, 0.86, 0.9, 0.9])  # (iii) fails
    assert out["verdict"] == "inconclusive" and not out["part_iii"]
    out = cal.h34_verdict(0.9, [0.87, 0.9, 0.95, 0.88], good_corr)  # raw >= 0.87 everywhere
    assert out["verdict"] == "refuted" and out["refuted_raw"]
    out = cal.h34_verdict(0.9, [np.nan, 0.6], [0.9, np.nan])  # NaN: (ii) from the finite values, (iii) fails
    assert out["part_ii"] and not out["part_iii"] and out["verdict"] == "inconclusive"
    assert cal.h34_verdict(np.nan, good_raw, good_corr)["verdict"] == "inconclusive"
    # interval-based verdict
    raw_ci = (np.array(good_raw) - 0.03, np.array(good_raw) + 0.03)
    corr_ci = (np.array(good_corr) + 0.005, np.array(good_corr) + 0.05)
    ok = cal.h34_verdict(0.9, good_raw, good_corr, onestep_ci=(0.88, 0.92), raw_ci=raw_ci, corrected_ci=corr_ci)
    assert ok["verdict_ci"] == "supported" and ok["part_i_ci"] and ok["part_ii_ci"] and ok["part_iii_ci"]
    wide = cal.h34_verdict(0.9, good_raw, good_corr, onestep_ci=(0.86, 0.94), raw_ci=raw_ci, corrected_ci=corr_ci)
    assert wide["verdict"] == "supported" and wide["verdict_ci"] == "inconclusive" and not wide["part_i_ci"]
    corr_low = (np.array(good_corr) - 0.03, np.array(good_corr) + 0.03)
    assert cal.h34_verdict(0.9, good_raw, good_corr, onestep_ci=(0.88, 0.92), raw_ci=raw_ci,
                           corrected_ci=corr_low)["verdict_ci"] == "inconclusive"  # fmt: skip
    high = np.array([0.95, 0.94, 0.93, 0.92])
    ref = cal.h34_verdict(0.9, high, good_corr, onestep_ci=(0.88, 0.92), raw_ci=(high - 0.02, high + 0.02),
                          corrected_ci=corr_ci)  # fmt: skip
    assert ref["verdict"] == "refuted" and ref["verdict_ci"] == "refuted"
    edge = cal.h34_verdict(0.9, high, good_corr, onestep_ci=(0.88, 0.92), raw_ci=(high - 0.06, high + 0.02),
                           corrected_ci=corr_ci)  # fmt: skip
    assert edge["verdict"] == "refuted" and edge["verdict_ci"] == "inconclusive"  # not significant
