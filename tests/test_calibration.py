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
    # Gibbs and Candes: a level >= 1 is the whole line, a level <= 0 the empty set (R1)
    assert cal.aci_quantile(s, 1.0, n_min=1) == math.inf and cal.aci_quantile(s, 1.7, n_min=1) == math.inf
    assert cal.aci_quantile(s, 0.0, n_min=1) == -math.inf and cal.aci_quantile(s, -0.2, n_min=1) == -math.inf
    assert cal.aci_quantile(s, 0.9, n_min=5) == 0.0  # fewer than n_min scores: no correction inside (0, 1)
    assert cal.aci_quantile(s, 1.2, n_min=5) == math.inf and cal.aci_quantile([], -0.1) == -math.inf  # ends first
    # the heuristic finite variant of the submitted tables: the level clipped to [0, 1], always finite
    assert cal.aci_quantile(s, 1.7, n_min=1, finite_quantile=True) == 4.0  # level 1: the largest score
    assert cal.aci_quantile(s, -0.2, n_min=1, finite_quantile=True) == 1.0  # level 0: the smallest score
    assert cal.aci_quantile(s, 1.7, n_min=5, finite_quantile=True) == 0.0  # warm-up first


def test_aci_warm_up_and_update() -> None:
    scores = np.linspace(-1.0, 1.0, 30)
    for finite in (True, False):
        out = cal.aci_run(scores, alpha=0.1, gamma=0.01, n_min=20, finite_quantile=finite)
        assert (out["q"][:20] == 0.0).all() and out["q"][20] != 0.0  # alpha_t stays in (0, 1) during the warm-up
        np.testing.assert_array_equal(out["covered"][:20], (scores[:20] <= 0.0).astype(float))
        err = 1.0 - out["covered"]
        np.testing.assert_allclose(out["alpha_t"][1:], 0.1 + 0.01 * np.cumsum(0.1 - err)[:-1])  # the recursion
        assert out["state"].n_events == 30 and np.array_equal(out["state"].history, np.sort(scores))
        assert out["state"].finite_quantile is finite
    # a large step drives alpha_t below 0 within the warm-up: the whole line takes precedence over the warm-up
    out = cal.aci_run(scores, alpha=0.1, gamma=0.05, n_min=20)
    below = out["alpha_t"] <= 0
    assert below[:20].any() and np.isposinf(out["q"][below]).all() and (out["covered"][below] == 1).all()
    assert (out["infinite"] == np.isposinf(out["q"])).all() and not out["empty"].any()
    old = cal.aci_run(scores, alpha=0.1, gamma=0.05, n_min=20, finite_quantile=True)  # the old rule: raw interval
    assert (old["q"][:20] == 0.0).all() and np.isfinite(old["q"]).all()
    nan = cal.aci_run(np.array([np.nan, 0.5]), 0.1, 0.05, n_min=1, history=[0.0, 1.0])
    assert np.isnan(nan["covered"][0]) and nan["alpha_t"][1] == 0.1 and nan["state"].n_events == 1


def test_aci_reviewer_counterexample() -> None:
    """Scores 1, ..., 1000 after a history [0] (alpha 0.1, gamma 0.01, n_min 1; review item R1): the finite variant
    never covers and alpha_t drifts to -8.9; the corrected ACI covers 0.9 of the events with alpha_t in [-gamma, 1]."""
    scores = np.arange(1.0, 1001.0)
    old = cal.aci_run(scores, alpha=0.1, gamma=0.01, n_min=1, history=[0.0], finite_quantile=True)
    assert old["covered"].mean() == 0.0 and old["state"].alpha_t == pytest.approx(-8.9)
    new = cal.aci_run(scores, alpha=0.1, gamma=0.01, n_min=1, history=[0.0])
    assert new["covered"].mean() == pytest.approx(0.89)  # the first 12 events miss, then one miss in ten
    bound = (max(0.1, 0.9) + 0.01) / (scores.size * 0.01)
    assert abs((1.0 - new["covered"]).mean() - 0.1) <= bound
    alphas = np.append(new["alpha_t"], new["state"].alpha_t)
    assert alphas.min() >= -0.01 - 1e-12 and alphas.max() <= 1.01 + 1e-12
    # every covered event of this monotone stream is covered by the whole line (alpha_t <= 0): the honest answer
    assert (new["infinite"] == (new["covered"] == 1)).all() and new["infinite"].mean() == pytest.approx(0.89)


def _streams(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return {
        "iid": rng.standard_normal(n),
        "drift": (1.0 + 4.0 * t / (n - 1)) * rng.standard_normal(n),
        "shift": np.where(t < n // 2, rng.standard_normal(n), 5.0 + 3.0 * rng.standard_normal(n)),
        "monotone": np.cumsum(rng.exponential(1.0, n)),
        "alternating": np.where(t % 7 == 0, 100.0 + t, rng.uniform(-1.0, 0.0, n)),
    }


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_aci_long_run_miscoverage_bound(seed: int) -> None:
    """|mean err - alpha| <= (max(alpha_1, 1 - alpha_1) + gamma) / (T gamma) and alpha_t in [-gamma, 1 + gamma] on
    random, drifting, shifted, monotone and adversarial streams, with and without warm-up and history."""
    n = 1500
    for name, scores in _streams(n, seed).items():
        for alpha in (0.1, 0.3):
            for gamma in (0.005, 0.02, 0.1):
                for n_min, history in ((0, None), (20, None), (5, [0.0, 0.5, 1.0])):
                    out = cal.aci_run(scores, alpha, gamma, n_min=n_min, history=history)
                    err = 1.0 - out["covered"]
                    bound = (max(alpha, 1.0 - alpha) + gamma) / (n * gamma)
                    assert abs(err.mean() - alpha) <= bound + 1e-12, (name, alpha, gamma, n_min)
                    alphas = np.append(out["alpha_t"], out["state"].alpha_t)
                    assert alphas.min() >= -gamma - 1e-12 and alphas.max() <= 1.0 + gamma + 1e-12, (name, gamma)
                    # the identity behind the bound: the final level is alpha_1 + gamma * sum(alpha - err)
                    assert out["state"].alpha_t == pytest.approx(alpha + gamma * np.sum(alpha - err), abs=1e-9)


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
    assert not online["infinite"].any() and not online["empty"].any()  # a gentle drift never reaches the ends


def test_aci_continuation_equals_one_run() -> None:
    s = drifting_scores(n=1200, seed=3)
    for finite in (False, True):
        whole = cal.aci_run(s, alpha=0.1, gamma=0.02, n_min=20, finite_quantile=finite)
        first = cal.aci_run(s[:700], alpha=0.1, gamma=0.02, n_min=20, finite_quantile=finite)
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
        with pytest.raises(ValueError):
            cal.aci_run(s, alpha=0.1, gamma=0.02, state=first["state"], finite_quantile=not finite)
    legacy = {k: v for k, v in first["state"].to_dict().items() if k != "finite_quantile"}  # a state of the old code
    assert cal.ACIState.from_dict(legacy).finite_quantile is False


# ------------------------------------------------------------------------------- prequential streams (R8)
def test_stream_run_immediate_feedback_equals_aci_run() -> None:
    s = drifting_scores(n=900, seed=5)
    s[[10, 400]] = np.nan
    start = cal.aci_run(s[:300], alpha=0.1, gamma=0.01)["state"]
    ref = cal.aci_run(s[300:], alpha=0.1, gamma=0.01, state=start)
    out = cal.stream_run(s[300:], 0.1, method="aci", gamma=0.01, n_min=20, state=start)
    for key in ("q", "alpha_t", "covered", "infinite", "empty"):
        np.testing.assert_array_equal(out[key], ref[key])
    assert out["state"].to_dict() == ref["state"].to_dict()
    ok = ~np.isnan(s[300:])
    assert (out["delay_events"][ok][:-1] == 0).all() and np.isnan(out["delay_events"][~ok]).all()
    # the same stream with explicit immediate times
    pos = np.arange(600, dtype=float)
    again = cal.stream_run(s[300:], 0.1, method="aci", gamma=0.01, n_min=20, state=start, issue_at=pos,
                           available_at=pos + 1.0)  # fmt: skip
    np.testing.assert_array_equal(again["q"], ref["q"])


def test_stream_run_delayed_labels() -> None:
    s = np.array([5.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    hist = np.linspace(-1.0, 1.0, 21)
    # every label arrives two events later: event i is issued with the labels of the events before i - 2
    issue, avail = cal.delayed_availability(6, 2.0, None, events_per_s=1.0)
    np.testing.assert_array_equal(avail - issue, 3.0)
    out = cal.stream_run(s, 0.1, method="aci", gamma=0.05, n_min=5, history=hist, issue_at=issue, available_at=avail)
    np.testing.assert_array_equal(out["alpha_t"][:3], [0.1, 0.1, 0.1])  # no label used yet
    assert out["alpha_t"][3] == pytest.approx(0.1 + 0.05 * (0.1 - 1.0))  # the miss of event 0 arrives before event 3
    np.testing.assert_array_equal(out["delay_events"][:3], [2.0, 2.0, 2.0])
    assert np.isnan(out["delay_events"][3:]).all()  # those labels arrive after the end of the stream
    assert out["state"].n_events == 6  # ... but still update the final state
    # time-based availability: labels a horizon after the issue time, ties broken by issue order
    t = np.array([0.0, 0.5, 1.0, 1.0, 4.0, 9.0])
    issue_t, avail_t = cal.delayed_availability(6, 3.0, t)
    np.testing.assert_array_equal(avail_t, t + 3.0)
    timed = cal.stream_run(s, 0.1, method="aci", gamma=0.05, n_min=5, history=hist, issue_at=issue_t,
                           available_at=avail_t)  # fmt: skip
    np.testing.assert_array_equal(timed["delay_events"][:4], [3.0, 2.0, 1.0, 0.0])  # all four used before t = 4
    with pytest.raises(ValueError):
        cal.delayed_availability(3, 1.0, [2.0, 1.0, 3.0])
    with pytest.raises(ValueError):
        cal.stream_run(s, 0.1, issue_at=issue)  # availability missing


def test_static_prequential_uses_the_scores_seen() -> None:
    rng = np.random.default_rng(4)
    cal_scores, s = rng.standard_normal(200), 2.0 + rng.standard_normal(300)
    out = cal.stream_run(s, 0.1, method="static", history=cal_scores)
    for i in (0, 1, 57, 299):
        assert out["q"][i] == cal.conformal_quantile(np.concatenate([cal_scores, s[:i]]), 0.1)
    assert np.isnan(out["alpha_t"]).all() and "state" not in out and out["history"].size == 500
    assert out["q"][-1] > out["q"][0]  # the correction followed the shifted scores
    # with delayed labels it sees fewer of them
    issue, avail = cal.delayed_availability(300, 10.0, None, events_per_s=1.0)
    late = cal.stream_run(s, 0.1, method="static", history=cal_scores, issue_at=issue, available_at=avail)
    assert late["q"][5] == out["q"][0] and late["q"][20] == cal.conformal_quantile(np.concatenate([cal_scores, s[:10]]), 0.1)


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


def test_coverage_table_infinite_intervals_and_streams() -> None:
    """A strong shift between calibration and evaluation drives ACI to alpha_t <= 0: some intervals are the whole
    line (counted as covered, infinite width, reported in share_infinite and left out of width_finite_*); the finite
    variant never produces them; the stream variants carry their delays and first-event coverages."""
    rng = np.random.default_rng(9)
    n = 400
    cal_t, ev_t = rollout_tables(n, [0, 1], 1.0, rng), rollout_tables(n, [0, 1], 6.0, rng)
    for frame in ev_t.values():  # 8 events per follower share their truth: clustered coverage
        for h in frame["horizon_s"].unique():
            rows = frame.index[frame["horizon_s"] == h]
            truth = frame.loc[rows, "truth_s"].to_numpy()
            frame.loc[rows, "truth_s"] = truth[(np.arange(n) // 8) * 8]
    ev = events(n, 1)
    ev["chrono"] = np.arange(n)[::-1]  # chronological order: the reverse of the table order
    ev["t_issue"] = np.where(ev["chrono"] < n // 2, 0.5 * ev["chrono"], np.nan)  # one NaN: the stream is untimed
    ev["driver"] = np.arange(n) // 8  # 8 events per follower
    kw = dict(alpha=0.1, gammas=(0.01, 0.05), n_boot=200)
    new = cal.coverage_table(ev_t, ev, cal_t, events(n, 2), stream_gamma=0.05, delay_events_per_s=2.0, **kw)
    old = cal.coverage_table(ev_t, ev, cal_t, events(n, 2), finite_quantile=True, **kw)
    assert list(new.columns) == list(cal.COVERAGE_COLUMNS)
    assert set(new["variant"]) == set(cal.VARIANTS) | set(cal.STREAM_VARIANTS) and set(old["variant"]) == set(cal.VARIANTS)
    online = new[(new["variant"] == "aci_online") & (new["gamma"] == 0.05)]
    assert (online["share_infinite"] > 0).all() and np.isinf(online["width_mean"]).all()
    assert np.isfinite(online["width_finite_mean"]).all() and (online["width_finite_mean"] > 0).all()
    assert (old["share_infinite"] == 0).all() and (old["share_empty"] == 0).all() and np.isfinite(old["width_mean"]).all()
    for frame in (new, old):  # raw and static do not depend on the ACI variant
        fixed = frame[frame["variant"].isin(["raw", "raw_linear", "static"])].reset_index(drop=True)
        ref = new[new["variant"].isin(["raw", "raw_linear", "static"])].reset_index(drop=True)
        np.testing.assert_allclose(fixed["picp"], ref["picp"])
    streams = new[new["variant"].isin(cal.STREAM_VARIANTS)]
    assert streams["picp_first50"].notna().all() and streams["picp_first100"].notna().all()
    delayed = new[new["variant"].str.endswith("_delayed")]
    for h, grp in delayed.groupby("horizon_s"):  # untimed: ceil(2 h) events in flight (fewer at the end)
        assert (grp["delay_events_mean"] <= math.ceil(2.0 * h) + 1e-9).all() and (grp["delay_events_mean"] > 0).all()
    assert (new.loc[new["variant"].isin(["aci_online", "aci_online_chrono", "static_prequential"]), "delay_events_mean"] == 0).all()
    assert new.loc[new["variant"].isin(["raw", "static", "aci_frozen"]), "picp_first50"].isna().all()
    assert (new["n_followers"] == n // 8).all()
    widen = (new["picp_hi_follower"] - new["picp_lo_follower"]) / (new["picp_hi"] - new["picp_lo"])
    assert np.nanmedian(widen) > 1.0  # whole followers are resampled: wider intervals
    assert ((new["picp_lo"] <= new["picp"]) & (new["picp"] <= new["picp_hi"])).all()


# --------------------------------------------------------------------------------------------- H3.4
def test_h34_decision_table() -> None:
    """The decision table of R7: supported iff (i), (ii), (iii) hold; refuted when any part fails; inconclusive only
    for a straddling bound (interval rule) or a missing number."""
    good_raw = [0.88, 0.80, 0.70, 0.60]
    good_corr = [0.90, 0.89, 0.88, 0.87]
    v = cal.h34_verdict(0.905, good_raw, good_corr)
    assert v["verdict"] == "supported" and v["part_i"] and v["part_ii"] and v["part_iii"] and v["verdict_ci"] is None
    assert cal.h34_verdict(0.87, good_raw, good_corr)["part_i"]  # the band is closed
    out = cal.h34_verdict(0.95, good_raw, good_corr)  # (i) fails: one-step calibration does not hold in-domain
    assert out["verdict"] == "refuted" and out["part_i"] is False
    out = cal.h34_verdict(0.9, [0.86, 0.85, 0.84, 0.83], good_corr)  # (ii) fails: the raw intervals already cover
    assert out["verdict"] == "refuted" and out["part_ii"] is False
    out = cal.h34_verdict(0.9, [0.5] * 4, [0.5] * 4)  # (iii) fails: the reviewer's case, "inconclusive" before
    assert out["verdict"] == "refuted" and out["part_iii"] is False and out["part_ii"]
    out = cal.h34_verdict(0.9, [np.nan, 0.6], [0.9, np.nan])  # a missing horizon of (iii), the others hold
    assert out["part_ii"] and out["part_iii"] is None and out["verdict"] == "inconclusive"
    assert cal.h34_verdict(0.9, [np.nan, 0.6], [0.5, np.nan])["verdict"] == "refuted"  # a failing horizon decides
    assert cal.h34_verdict(np.nan, good_raw, good_corr)["verdict"] == "inconclusive"
    assert cal.h34_verdict(0.9, good_raw, good_corr, refute_min=0.87)["verdict"] == "supported"  # old key ignored
    # interval rule
    raw_ci = (np.array(good_raw) - 0.03, np.array(good_raw) + 0.03)
    corr_ci = (np.array(good_corr) + 0.005, np.array(good_corr) + 0.05)
    ok = cal.h34_verdict(0.9, good_raw, good_corr, onestep_ci=(0.88, 0.92), raw_ci=raw_ci, corrected_ci=corr_ci)
    assert ok["verdict_ci"] == "supported" and (ok["part_i_ci"], ok["part_ii_ci"], ok["part_iii_ci"]) == ("holds",) * 3
    wide = cal.h34_verdict(0.9, good_raw, good_corr, onestep_ci=(0.86, 0.94), raw_ci=raw_ci, corrected_ci=corr_ci)
    assert wide["verdict"] == "supported" and wide["verdict_ci"] == "inconclusive" and wide["part_i_ci"] == "straddles"
    far = cal.h34_verdict(0.95, good_raw, good_corr, onestep_ci=(0.94, 0.96), raw_ci=raw_ci, corrected_ci=corr_ci)
    assert far["part_i_ci"] == "fails" and far["verdict_ci"] == "refuted"
    corr_low = (np.array(good_corr) - 0.03, np.array(good_corr) + 0.03)  # straddles 0.87, never entirely below
    low = cal.h34_verdict(0.9, good_raw, good_corr, onestep_ci=(0.88, 0.92), raw_ci=raw_ci, corrected_ci=corr_low)
    assert low["part_iii_ci"] == "straddles" and low["verdict_ci"] == "inconclusive"
    corr_fail = (np.array([0.80, 0.85, 0.86, 0.80]), np.array([0.84, 0.95, 0.95, 0.86]))
    fail = cal.h34_verdict(0.9, good_raw, [0.82, 0.9, 0.9, 0.83], onestep_ci=(0.88, 0.92), raw_ci=raw_ci,
                           corrected_ci=corr_fail)  # fmt: skip
    assert fail["verdict"] == "refuted" and fail["part_iii_ci"] == "fails" and fail["verdict_ci"] == "refuted"
    high = np.array([0.95, 0.94, 0.93, 0.92])
    ref = cal.h34_verdict(0.9, high, good_corr, onestep_ci=(0.88, 0.92), raw_ci=(high - 0.02, high + 0.02),
                          corrected_ci=corr_ci)  # fmt: skip
    assert ref["verdict"] == "refuted" and ref["part_ii_ci"] == "fails" and ref["verdict_ci"] == "refuted"
    edge = cal.h34_verdict(0.9, high, good_corr, onestep_ci=(0.88, 0.92), raw_ci=(high - 0.16, high + 0.02),
                           corrected_ci=corr_ci)  # fmt: skip
    assert edge["verdict"] == "refuted" and edge["part_ii_ci"] == "straddles" and edge["verdict_ci"] == "inconclusive"
    gap = cal.h34_verdict(0.9, good_raw, good_corr, onestep_ci=(np.nan, np.nan), raw_ci=raw_ci, corrected_ci=corr_ci)
    assert gap["part_i_ci"] == "missing" and gap["verdict_ci"] == "inconclusive"
    assert set(cal.H34_STATES) >= {ok["part_i_ci"], wide["part_i_ci"], far["part_i_ci"], gap["part_i_ci"]}
