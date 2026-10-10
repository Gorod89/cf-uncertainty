"""Review fixes of version v3 (docs/review_fixes_v3.md): event streams of the ACI information budget (R8), the
file-cluster bootstrap and the common-support growth of the platoon analysis (R4, R11), the mass-share concentration
(consistency item a)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cf_unc.eval.bootstrap import bootstrap_ratio_by_group
from cf_unc.maps.shift import concentration
from cf_unc.platoon.evaluate import growth_curve, growth_exponent, growth_exponents, relative_reduction
from cf_unc.rollout.streams import RUN_GAP_S, natural_key, stream_order


# ----------------------------------------------------------------------------------------------- streams (R8)
def _events(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"event": np.arange(len(ids)), "event_id": ids, "t0": np.full(len(ids), 29)})


def test_stream_order_absolute_clock() -> None:
    """NGSIM: meta.t0 is one clock for the three periods; the interval is issued t0 = 29 samples after the start."""
    meta = {"a": {"period": 0, "t0": 50.0}, "b": {"period": 1, "t0": 3700.0}, "c": {"period": 0, "t0": 10.0},
            "d": {"period": 2, "t0": 3700.0}}  # fmt: skip
    out, kind = stream_order(_events(["a", "b", "c", "d"]), meta)
    assert kind == "absolute"
    np.testing.assert_allclose(out["t_issue"], [52.9, 3702.9, 12.9, 3702.9])
    assert out["chrono"].tolist() == [1, 2, 0, 3]  # ties by the event code


def test_stream_order_runs_and_untimed() -> None:
    meta = {"x1": {"run": "platoon10", "date": "2019-07-04", "campaign": "A", "t0": 5.0},
            "x2": {"run": "platoon3", "date": "2019-07-04", "campaign": "A", "t0": 400.0},
            "x3": {"run": "platoon3", "date": "2019-07-04", "campaign": "A", "t0": 100.0},
            "x4": {"run": "part1", "date": "2019-07-03", "campaign": "A", "t0": 7.0}}  # fmt: skip
    out, kind = stream_order(_events(["x1", "x2", "x3", "x4"]), meta)
    assert kind == "per_run" and out["chrono"].tolist() == [3, 2, 1, 0]  # date, then platoon3 before platoon10
    t = out.set_index("event_id")["t_issue"]
    assert t["x2"] - t["x3"] == pytest.approx(300.0)  # one clock within a run
    assert t["x1"] - t["x2"] >= RUN_GAP_S and t["x3"] - t["x4"] >= RUN_GAP_S  # runs one after the other
    assert natural_key("platoon3") < natural_key("platoon10")
    fn = {"h1": {"t0": 0.0, "source_index": 7}, "h2": {"t0": 0.0, "source_index": 2}}
    out, kind = stream_order(_events(["h1", "h2"]), fn)
    assert kind == "untimed" and out["chrono"].tolist() == [1, 0] and out["t_issue"].isna().all()
    out, kind = stream_order(_events(["h1", "zz"]), fn)  # an event without meta: untimed, ordered by the code
    assert kind == "untimed" and out["chrono"].tolist() == [0, 1]
    out, kind = stream_order(_events(["h1", "h2"]), None)
    assert kind == "untimed"


# ---------------------------------------------------------------------------------------- clusters (R4, R11)
def test_file_cluster_bootstrap_of_a_reduction() -> None:
    """The paired cluster bootstrap keeps the profile-weighted estimate and resamples files with all their
    profiles: the draws of the reviewer's check (rng.integers(0, n_files, (B, n_files))) give the same interval."""
    rng = np.random.default_rng(3)
    files = np.repeat([f"f{i}" for i in range(8)], [1, 2, 3, 4, 5, 6, 7, 8])
    effect = rng.normal(0.0, 0.3, 8)[np.unique(files, return_inverse=True)[1]]  # a file effect: dependent profiles
    ref = rng.uniform(2.0, 4.0, files.size)
    model = ref * (0.7 + effect)
    by_profile = relative_reduction(model, ref, np.arange(files.size), n_boot=2000, seed=1)
    by_file = relative_reduction(model, ref, files, n_boot=10000, seed=20261009)
    assert by_file["estimate"] == pytest.approx(by_profile["estimate"]) == pytest.approx(1 - model.mean() / ref.mean())
    assert by_file["hi"] - by_file["lo"] > by_profile["hi"] - by_profile["lo"]
    unique, codes = np.unique(files, return_inverse=True)
    sm, sr = np.bincount(codes, weights=model), np.bincount(codes, weights=ref)
    draw = np.random.default_rng(20261009).integers(0, unique.size, size=(10000, unique.size))
    lo, hi = np.quantile(1 - sm[draw].sum(axis=1) / sr[draw].sum(axis=1), [0.025, 0.975])
    assert (by_file["lo"], by_file["hi"]) == (pytest.approx(lo, abs=1e-12), pytest.approx(hi, abs=1e-12))
    again = bootstrap_ratio_by_group(model, ref, files, n_boot=10000, seed=20261009)
    assert again["estimate"] == pytest.approx(model.mean() / ref.mean())


def test_growth_curve_by_files_and_vectorised_exponents() -> None:
    rng = np.random.default_rng(0)
    i = np.arange(1, 21, dtype=float)
    files = np.repeat(["a", "b", "c", "d", "e", "f"], 5)
    level = rng.lognormal(0.0, 0.5, 6)[np.unique(files, return_inverse=True)[1]]
    curves = level[:, None] * i[None, :] ** 0.2 * rng.lognormal(0.0, 0.05, (files.size, 20))
    by_profile = growth_curve(curves, n_boot=500, seed=2)
    by_file = growth_curve(curves, n_boot=2000, seed=2, groups=files)
    assert by_file["beta"] == by_profile["beta"] and np.allclose(by_file["curve"], by_profile["curve"])
    assert by_file["beta_lo"] <= by_profile["beta"] <= by_file["beta_hi"]
    reps = rng.lognormal(0.0, 0.3, (50, 20))
    reps[3, 4] = np.nan
    reps[7, :19] = np.nan  # a single usable point: NaN
    expected = np.array([growth_exponent(r)[0] for r in reps])
    np.testing.assert_allclose(growth_exponents(reps), expected, equal_nan=True)
    assert np.isnan(growth_exponents(reps)[7])
    assert np.isnan(growth_curve(curves[:5], n_boot=50, seed=0, groups=files[:5])["beta_lo"])  # one file: no interval


# ------------------------------------------------------------------------------- concentration (consistency)
def test_mass_share_concentration() -> None:
    """share of the mass / share of the windows = mean(x | mask) / mean(x); > 1 exactly when the ratio of the
    conditional means is > 1 (88 % of the mass on 34 % of the windows: 2.6, while the ratio of means is ~14)."""
    rng = np.random.default_rng(1)
    n = 6000
    mask = rng.uniform(size=n) < 0.34
    x = np.where(mask, rng.exponential(14.0, n), rng.exponential(1.0, n))
    x = np.stack([x, x * rng.lognormal(0.0, 0.1, n)])  # two seeds
    events = np.arange(n) // 6
    res = concentration(x, mask, events, n_boot=400, seed=5)
    assert res["mass_ratio"] == pytest.approx(res["share_mass"] / res["share_windows"], rel=1e-9)
    assert res["mass_ratio_lo"] <= res["mass_ratio"] <= res["mass_ratio_hi"]
    r, p = res["estimate"], res["share_windows"]
    assert res["mass_ratio"] == pytest.approx(r / (p * r + 1.0 - p), rel=1e-9)  # the identity between the two
    assert res["mass_ratio"] < res["estimate"] and (res["mass_ratio"] > 1) == (res["estimate"] > 1)


# ------------------------------------------------------------------------------------------ report tables
def _script(name: str) -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_review_v3", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # dataclasses of the script need their module registered
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_report_tables_of_the_review_fixes(tmp_path: Path) -> None:
    """t5_h32_cluster__e3 and t6_collision_diagnostics come from the columns of analyze_e3; without the file
    clusters (an old analysis) the table is left out."""
    report, e3 = _script("make_report"), _script("analyze_e3")
    t = tmp_path / "e3" / "tables"
    t.mkdir(parents=True)
    rows = []
    for mode, n, files in (("ACC", 47, 23), ("Human", 12, 3), ("all", 60, 26)):
        rows.append({"variant": "raw", "set": "extended", "mode": mode, "window": "response", "quantity": "amplitude",
                     "reference": "idm_white_noise", "model": "diffusion_cf", "reduction": 0.34, "lo": 0.09, "hi": 0.5,
                     "verdict": "inconclusive", "verdict_point": "supported", "crps_model": 2.3, "crps_reference": 3.5,
                     "n_profiles": n, "lo_cluster": -0.26, "hi_cluster": 0.57, "verdict_cluster": "inconclusive",
                     "n_files": files, "reduction_seed_min": 0.23, "reduction_seed_max": 0.45, "n_seeds": 5})
    pd.DataFrame(rows).reindex(columns=e3.COLUMNS["h32"]).to_csv(t / "h32.csv", index=False)
    diag = [{"variant": "raw", "model": m, "set": "extended", "mode": "all", "n_profiles": 60, "n_rollouts": 9600,
             "collision_free_share": f, "ttfc_median_s": 35.0, "ttfc_p10_s": 18.0, "ttfc_p90_s": 89.0,
             "collided_in_burn_in_share": 0.37, "collided_v1": 0.08, "collided_v5": 0.14, "collided_v10": 0.27,
             "collided_v20": 0.54, "crps_1_4": 2.06, "crps_1_4_same_profiles": 2.06, "crps_1_4_survivors_biased": 1.86,
             "kept_share_1_4": 0.85, "n_profiles_survivors": 60} for m, f in (("diffusion_cf", 0.1), ("idm_white_noise", 1.0))]
    pd.DataFrame(diag).reindex(columns=e3.COLUMNS["collision_diagnostics"]).to_csv(t / "collision_diagnostics.csv", index=False)
    ctx = report.Context({"e3": tmp_path / "e3"})
    (h32,) = report.h32_cluster_tables(ctx)
    assert h32.name == "t5_h32_cluster__e3" and len(h32.data) == 3 and set(h32.data["n_files"]) == {23, 3, 26}
    assert h32.view.iloc[0]["verdict profiles / files"] == "inconclusive / inconclusive"
    (col,) = report.collision_diag_tables(ctx)
    assert col.name == "t6_collision_diagnostics" and len(col.view) == 2
    old = pd.DataFrame(rows).reindex(columns=e3.COLUMNS["h32"]).assign(lo_cluster=np.nan)
    old.to_csv(t / "h32.csv", index=False)
    assert report.h32_cluster_tables(ctx) == []  # an analysis without the file clusters
