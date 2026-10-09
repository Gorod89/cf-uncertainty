"""Report builder (scripts/make_report.py, design.md 14.2) on a tiny fake set of experiment roots (the optional roots
of E5 and E6 included), and the new E3 figure of CRPS bars per driver mode (cf_unc.report.figures_e3.plot_crps_bars)."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

from cf_unc.report import figures_e3

REPO = Path(__file__).resolve().parents[1]
REGIMES = ("all", "congested", "braking", "approaching", "free", "steady", "other")
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64  # copied, never decoded
E3_MODELS = {"e3": ("idm_white_noise", "lstm_const_sigma", "diffusion_cf"), "e3_vcap": ("idm_white_noise", "lstm_const_sigma"),
             "e3_step": ("idm_white_noise", "deep_ensemble__step")}  # fmt: skip
PROFILES = {"extended": {"all": 10, "ACC": 6, "Human": 3, "mixed": 1}, "headline": {"all": 5, "ACC": 3, "Human": 2}}
EXPECTED_TABLES = (
    [f"t1_one_step__{v}__{t}" for v in ("raw", "harmonised") for t in ("a", "mean_1s")]
    + ["t2_h31_summary"] + [f"t2_h31__{v}__{t}" for v in ("raw", "harmonised") for t in ("a", "mean_1s")]
    + [f"t3_{k}__{v}__{t}" for k in ("h33_nll", "regime_variance") for v in ("raw", "harmonised") for t in ("a", "mean_1s")]
    + [f"t4_{k}__{v}__{t}" for k in ("coverage", "onestep_conformal") for v in ("raw", "harmonised") for t in ("a", "mean_1s")]
    + ["t4_h34_verdict_counts", "t5_h32__e3__raw", "t5_h32__e3__harmonised", "t5_h32__sensitivities", "t6_collisions",
       "t6_amplitude_coverage", "t6_growth"]
)  # fmt: skip


def _load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"{name}_script", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # dataclasses of the script resolve their annotations through sys.modules
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def report() -> Any:
    return _load_script("make_report")


@pytest.fixture(scope="module")
def e3_columns() -> dict[str, list[str]]:
    return _load_script("analyze_e3").COLUMNS  # the schema the E3 analysis writes


# ---------------------------------------------------------------------------------------- fake roots
def _csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if columns is not None:  # the full schema of the real table, unused columns empty
        frame = frame.reindex(columns=columns)
    frame.to_csv(path, index=False)


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG)


def fake_e1(root: Path) -> None:
    t = root / "tables"
    models, datasets = ("idm_white_noise", "lstm_const_sigma", "hetero_mdn", "deep_ensemble"), ("highd", "highd_val", "ngsim", "ngsim_fn")
    _csv(t / "one_step_metrics.csv", [
        {"model": m, "dataset": d, "subset": s, "seed": seed, "n_seeds": 3 if seed == "mean" else 1, "n_windows": 100,
         "n_events": 10, "nll": 1.0 + k, "crps": 0.1 * (1 + k), "picp90": 0.9, "width90": 0.5, "rmse": 0.2,
         "nll_lo": np.nan, "nll_hi": np.nan}
        for m in models for d in datasets for s in ("all", "ACC") for k, seed in enumerate(("0", "1", "2"))
    ] + [{"model": m, "dataset": d, "subset": "all", "seed": "mean", "n_seeds": 3, "n_windows": 100, "n_events": 10, "nll": 2.0,
          "crps": 0.2, "picp90": 0.9, "width90": 0.5, "rmse": 0.2} for m in models for d in datasets])  # fmt: skip
    pairs = {"highd_val": "placebo", "ngsim": "shift", "ngsim_fn": "cross_check"}
    keys = [("deep_ensemble", "model"), ("deep_ensemble", "seed_ensemble"), ("lstm_const_sigma", "seed_ensemble")]
    ver, geo, h31e = [], [], []
    for (m, kind) in keys:
        for d, pair in pairs.items():
            r_alea = 1.0 if m == "lstm_const_sigma" else 4.0
            for s in ("all", "ACC"):
                ver.append({"model": m, "kind": kind, "dataset": d, "subset": s, "pair": pair, "r_epi": 50.0, "r_epi_lo": 30.0,
                            "r_epi_hi": 80.0, "r_alea": r_alea, "r_alea_lo": 0.8 * r_alea, "r_alea_hi": 1.2 * r_alea,
                            "verdict_ci": "refuted", "verdict_point": "refuted"})  # fmt: skip
                h31e.append({"model": m, "kind": kind, "dataset": d, "subset": s, "pair": pair, "verdict_ci": "supported",
                             "verdict_point": "supported"})  # fmt: skip
            for q in (("epistemic_var", "aleatoric_var") if kind == "model" else ("var",)):
                geo.append({"model": m, "dataset": d, "quantity": q, "ratio_geometric": 9.0, "geo_lo": 8.0, "geo_hi": 10.0,
                            "ratio_means": 50.0 if q == "epistemic_var" else r_alea})  # fmt: skip
    _csv(t / "h31_verdicts.csv", ver)
    _csv(t / "h31e_verdicts.csv", h31e)
    _csv(t / "extra_shift_geometric.csv", geo)
    _csv(t / "h31_summary.csv", [{"model": m, "kind": k, "n_pairs": 1, "n_supported": 0, "n_refuted": 1, "n_inconclusive": 0,
                                  "overall": "refuted", "overall_point": "refuted", "placebo_verdicts": "highd_val:inconclusive",
                                  "cross_check_verdicts": "ngsim_fn:refuted"} for m, k in keys])  # fmt: skip
    _csv(t / "h31e_summary.csv", [{"model": m, "kind": k, "main": k == "model", "rule": "rare_mass", "conc": 20.0, "conc_lo": 5.0,
                                   "conc_hi": 50.0, "verdict_conc_ci": "supported", "n_supported": 1, "n_refuted": 0,
                                   "n_inconclusive": 0, "overall": "supported", "overall_point": "supported"} for m, k in keys])  # fmt: skip
    _csv(t / "h33_nll.csv", [
        {"dataset": d, "subset": s, "model_a": "hetero_mdn", "model_b": "lstm_const_sigma", "group": g, "n_windows": 1500,
         "n_events": 20, "nll_a": -2.9, "nll_a_lo": -3.0, "nll_a_hi": -2.8, "nll_b": -2.7, "nll_b_lo": -2.8, "nll_b_hi": -2.6,
         "diff": -0.2, "diff_lo": -0.3, "diff_hi": -0.1, "p_holm": 0.005, "reject": True, "ci_overlap": g != "all"}
        for d in datasets for s in ("all", "ACC") for g in REGIMES
    ])  # fmt: skip
    _csv(t / "regime_ordering.csv", [
        {"model": m, "dataset": d, "subset": "all", "quantity": q, "regime": g, "reference": "steady", "estimate": 2.0 + k,
         "lo": 1.5 + k, "hi": 2.5 + k, "p_holm": 0.01, "reject": k % 2 == 0}
        for m in ("hetero_mdn", "deep_ensemble") for d in datasets for q in ("var", "mse")
        for k, g in enumerate(r for r in REGIMES[1:] if r != "steady")
    ])  # fmt: skip
    f = root / "figures"
    for m in models:
        _png(f / "maps" / f"{m}__model.png")
        _png(f / "regimes" / f"variance__{m}.png")
    for name in ("shift_ratios", "shift_ratios_matched", "shift_ratios_controls", "nll_by_regime"):
        _png(f / f"{name}.png")


def fake_e2(root: Path, targets: tuple[str, ...] = ("a", "mean_1s")) -> None:
    t = root / "tables"
    models, datasets = ("idm_white_noise", "lstm_const_sigma"), ("highd", "highd_val", "ngsim")
    _csv(t / "coverage_by_horizon.csv", [
        {"model": m, "target": tg, "dataset": d, "horizon_s": h, "variant": v, "picp": 0.5 + 0.1 * k, "picp_lo": 0.4,
         "picp_hi": 0.6, "width_mean": h, "width_median": h, "crps": 1.0, "collided": 0.0, "n_events": 50, "n_seeds": 5}
        for m in models for tg in targets for d in datasets for h in (1.0, 3.0, 5.0, 10.0)
        for k, v in enumerate(("raw", "raw_linear", "static", "aci_frozen", "aci_online"))
    ])  # fmt: skip
    _csv(t / "h34_verdicts.csv", [
        {"model": m, "target": tg, "dataset": d, "corrected": c, "verdict": "supported", "verdict_ci": "inconclusive"}
        for m in models for tg in targets for d in ("highd", "ngsim") for c in ("static", "aci_frozen", "aci_online")
    ])  # fmt: skip
    _csv(t / "onestep_conformal.csv", [
        {"model": m, "target": tg, "dataset": d, "score": s, "picp_raw": 0.8, "picp_raw_lo": 0.79, "picp_raw_hi": 0.81,
         "picp_conformal": 0.9, "picp_conformal_lo": 0.89, "picp_conformal_hi": 0.91, "width_raw": 1.0, "width_conformal": 1.2,
         "q": 1.1, "n_windows": 100, "n_events": 10, "n_seeds": 5}
        for m in models for tg in targets for d in ("highd", "ngsim") for s in ("absolute", "normalised")
    ])  # fmt: skip
    for name in ("coverage_by_horizon__a", "coverage_by_horizon__a__ngsim", "width_by_horizon__a"):
        _png(root / "figures" / f"{name}.png")


def fake_e3(root: Path, key: str, columns: dict[str, list[str]], extra: dict[str, tuple[str, ...]] | None = None) -> None:
    """Tables, config and figures of an E3 root; ``extra``: further signal variants (name -> models) after the
    default ones (E3-pooled, design.md 17.1), whose rows leave those of the default variants unchanged."""
    t = root / "tables"
    variants = ("raw", "harmonised") if key == "e3" else ("raw",)
    by_variant = {**{v: E3_MODELS[key] for v in variants}, **dict(extra or {})}
    sets = ("headline", "extended") if key == "e3" else ("headline",)
    crps, h32, coll, cov, growth = [], [], [], [], []
    for v, models in by_variant.items():
        value = {m: 3.0 * (1 + 4 * k) for k, m in enumerate(models)}  # reference 3, then 15, 27: a span > 10 for the figure
        for s in sets:
            for mode, n in PROFILES[s].items():
                for w in ("response", "profile"):
                    for m in models:
                        for index in ("1", "all", "1-4"):
                            crps.append({"variant": v, "model": m, "set": s, "mode": mode, "window": w, "index": index,
                                         "crps": value[m], "crps_lo": 0.8 * value[m], "crps_hi": 1.2 * value[m],
                                         "crps_ratio": 0.5, "crps_ratio_lo": 0.4, "crps_ratio_hi": 0.6, "n_profiles": n})  # fmt: skip
                        if m != "idm_white_noise":
                            for q in ("amplitude", "ratio"):
                                red = 1 - value[m] / value["idm_white_noise"]
                                h32.append({"variant": v, "set": s, "mode": mode, "window": w, "quantity": q,
                                            "reference": "idm_white_noise", "model": m, "reduction": red, "lo": red - 1,
                                            "hi": red + 1, "verdict": "inconclusive" if n < 5 else "refuted",
                                            "verdict_point": "refuted", "crps_model": value[m],
                                            "crps_reference": value["idm_white_noise"], "n_profiles": n})  # fmt: skip
            for m in models:
                for index in [*map(str, range(1, 21)), "any"]:
                    coll.append({"variant": v, "model": m, "set": s, "index": index, "collided": 0.1, "collided_burn_in": 0.01,
                                 "profiles_with_collision": 0.5, "min_gap_mean": 5.0, "n_rollouts": 100})  # fmt: skip
                for index in ("1", "all"):
                    for meth in ("raw", "static_loo", "aci_online"):
                        cov.append({"variant": v, "model": m, "set": s, "index": index, "method": meth, "coverage": 0.85,
                                    "coverage_lo": 0.8, "coverage_hi": 0.9, "nominal": 0.9, "width_median": 2.0, "q_mean": 0.1,
                                    "n_profiles": PROFILES[s]["all"]})  # fmt: skip
                for w in ("response", "profile"):
                    for i in range(1, 21):
                        growth.append({"variant": v, "model": m, "set": s, "window": w, "index": i, "sd": 3.0 * i**0.1,
                                       "sd_lo": 2.0, "sd_hi": 4.0, "n_profiles": 5, "beta": 0.1, "beta_lo": 0.05, "beta_hi": 0.15,
                                       "log_sd1": 1.1, "share_neg_d2": 0.5, "fitted": True})  # fmt: skip
    for s in sets:
        for i in range(1, 7):  # measured platoons: fitted over 1-4, index 6 measured by too few profiles (NaN)
            growth.append({"variant": "data", "model": "empirical", "set": s, "window": "profile", "index": i,
                           "sd": np.nan if i == 6 else 3.5 + 0.1 * i, "sd_lo": 3.0, "sd_hi": 4.0, "n_profiles": 2 if i >= 5 else 5,
                           "beta": 0.12, "beta_lo": 0.08, "beta_hi": 0.16, "log_sd1": 1.2, "share_neg_d2": 0.3, "fitted": i <= 4})  # fmt: skip
    _csv(t / "amplitude_crps.csv", crps, columns["amplitude_crps"])
    _csv(t / "h32.csv", h32, columns["h32"])
    _csv(t / "collisions.csv", coll, columns["collisions"])
    _csv(t / "amplitude_coverage.csv", cov, columns["amplitude_coverage"])
    _csv(t / "growth.csv", growth, columns["growth"])
    cfg = {"h32": {"reference": "idm_white_noise", "min_profiles": 5}, "platoon": {"v_max": 35.9 if key == "e3_vcap" else None}}
    (root / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    for v in by_variant:
        for name in ("crps_bars", "growth", "crps_by_index"):
            _png(root / "figures" / f"{name}__{v}.png")


MATCHED = ("e1m", "e1m_mean1s", "e2m")  # E5 roots (design.md 15.1): optional, absent unless a test creates them
POOLED = ("e6_pool", "e6_lodo_ngsim", "e6_lodo_openacc", "e6_lodo_ngsim_all",
          "e2_pool", "e2_lodo_ngsim", "e2_lodo_openacc", "e2_lodo_ngsim_all")  # E6 (16.1, 16.4)


@pytest.fixture()
def roots(tmp_path: Path, e3_columns: dict[str, list[str]]) -> dict[str, Path]:
    out = {key: tmp_path / "artifacts" / key for key in ("e1", "e1_mean1s", "e1h", "e1h_mean1s", "e2", "e2h", "e3", "e3_vcap", "e3_step")}
    for key, root in out.items():
        if key.startswith("e1"):
            fake_e1(root)
        elif key.startswith("e2"):
            fake_e2(root)
        else:
            fake_e3(root, key, e3_columns)
    # the E5 and E6 roots are given (so that the real artifacts/e1m etc. are never read) but do not exist
    return {**out, **{key: tmp_path / "artifacts" / key for key in (*MATCHED, *POOLED)}}


def add_matched_roots(roots: dict[str, Path]) -> None:
    """Fake E5 roots: E1 tables and figures for e1m / e1m_mean1s, E2 tables and figures for e2m."""
    for key in MATCHED:
        (fake_e1 if key.startswith("e1") else fake_e2)(roots[key])


def add_pooled_roots(roots: dict[str, Path]) -> None:
    """Fake E6 roots (design.md 16.1): E1 tables and figures analysed against pool_test, target a; the pooled root has
    no shift pair (only the placebo, no shift figures); E2 tables and figures of target a."""
    for key in POOLED:
        if key.startswith("e2"):
            fake_e2(roots[key], targets=("a",))
            continue
        fake_e1(roots[key])
        (roots[key] / "tables" / "analysis_info.json").write_text('{"in_domain": "pool_test"}', encoding="utf-8")
        if key == "e6_pool":
            ver = pd.read_csv(roots[key] / "tables" / "h31_verdicts.csv")
            ver[ver["pair"] == "placebo"].to_csv(roots[key] / "tables" / "h31_verdicts.csv", index=False)
            for name in ("shift_ratios", "shift_ratios_matched"):
                (roots[key] / "figures" / f"{name}.png").unlink()


# ------------------------------------------------------------------------------------------ checks
def check_latex(text: str) -> None:
    """Structure of a generated .tex file: environments nested and closed, braces balanced, math closed on every
    line, booktabs rules, the same number of cells in every row as in the column specification, no unescaped
    special character outside math."""
    lines = [line for line in text.splitlines() if not line.startswith("%")]
    src = "\n".join(lines)
    stack: list[str] = []
    for m in re.finditer(r"\\(begin|end)\{(\w+)\}", src):
        if m.group(1) == "begin":
            stack.append(m.group(2))
        else:
            assert stack and stack.pop() == m.group(2), m.group(0)
    assert not stack
    depth = 0
    for i, ch in enumerate(src):
        if ch in "{}" and src[i - 1 : i] != "\\":
            depth += 1 if ch == "{" else -1
            assert depth >= 0
    assert depth == 0
    assert all(rule in src for rule in (r"\toprule", r"\midrule", r"\bottomrule"))
    spec = re.search(r"\\begin\{(?:tabular|longtable)\}\{([lrc]+)\}", src)
    assert spec is not None
    n_cols = len(spec.group(1))
    rows = [line for line in lines if line.endswith(r"\\") and not line.startswith(r"\caption")]
    assert len(rows) >= 2  # header and at least one body row
    for line in lines:
        assert len(re.findall(r"(?<!\\)\$", line)) % 2 == 0, line
        if line in rows:
            assert len(re.findall(r"(?<!\\)&", line)) == n_cols - 1, line
        bare = re.sub(r"\$[^$]*\$", "", line)
        assert not re.search(r"(?<!\\)[_%#^&]", bare.replace(" & ", " ")), line


def _files(directory: Path) -> dict[str, bytes]:
    return {str(p.relative_to(directory)): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


# ------------------------------------------------------------------------------------------- tests
def test_number_formats(report: Any) -> None:
    assert report.ci(2.98, 2.41, 3.57) == "2.98 [2.41, 3.57]"
    assert report.ci(805.7, 470.3, 1845.9) == "806 [470, 1846]"
    assert report.ci2(-18.36, -24.50, -13.59) == "-18.4 [-24.5, -13.6]"
    assert report.ci2(-0.0569, -0.338, 0.164) == "-0.06 [-0.34, 0.16]"
    assert report.ci2(0.336, 0.0851, 0.5008) == "0.34 [0.09, 0.50]"
    assert report.pm(-2.7829, 0.0312) == "-2.78 ± 0.03"
    assert report.pm(0.452, 6.2e-17) == "0.452 ± 0.000"  # floating-point noise of a deterministic model
    assert report.pm(467.0, 123.4) == "467 ± 123" and report.pm(2449.9, 120.0) == "2450 ± 120"
    assert report.num(1.5e-5) == "1.50e-05" and report.num(np.nan) == report.DASH and report.ci(np.nan, 0, 1) == report.DASH
    assert report.prob(-0.0001) == "0.00"  # no negative zero
    assert report.tex("a_b 50 % & x #1") == r"a\_b 50 \% \& x \#1"
    assert report.tex("-0.5 [-1.2, 2] 1-4 per-step") == "$-$0.5 [$-$1.2, 2] 1-4 per-step"
    assert report.tex(">= 0.20, < 0.05, ±, →") == r"$\geq$ 0.20, $<$ 0.05, $\pm$, $\rightarrow$"
    assert report.tex("1.50e-05") == r"1.50$\times10^{-5}$"


def test_make_report_builds_every_table_figure_and_index(report: Any, roots: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "report"
    result = report.make_report(out, roots)
    assert not result["missing"], result["missing"]
    names = [t.name for t in result["tables"]]
    assert sorted(names) == sorted(EXPECTED_TABLES) and len(set(names)) == len(names)
    index = (out / "index.md").read_text(encoding="utf-8")
    for name in EXPECTED_TABLES:
        assert (out / "tables" / f"{name}.csv").exists() and (out / "tables" / f"{name}.tex").exists(), name
        assert f"`tables/{name}.csv`" in index, name
        check_latex((out / "tables" / f"{name}.tex").read_text(encoding="utf-8"))
        assert len(pd.read_csv(out / "tables" / f"{name}.csv")) > 0, name
    # figures: stable names carrying the root, F1 maps and F3 regime figures as one directory per root
    expected = ({f"f1_maps__{k}" for k in ("e1", "e1_mean1s", "e1h", "e1h_mean1s")}
                | {f"f3_regime_variance__{k}" for k in ("e1", "e1_mean1s", "e1h", "e1h_mean1s")}
                | {f"f2_{s}__{k}.png" for s in ("shift_ratios", "shift_ratios_matched", "shift_ratios_controls")
                   for k in ("e1", "e1_mean1s", "e1h", "e1h_mean1s")}
                | {f"f3_nll_by_regime__{k}.png" for k in ("e1", "e1_mean1s", "e1h", "e1h_mean1s")}
                | {f"f4_{s}__{k}.png" for s in ("coverage_by_horizon__a", "coverage_by_horizon__a__ngsim", "width_by_horizon__a")
                   for k in ("e2", "e2h")}
                | {f"f{n}_{s}__{v}__e3.png" for n, s in ((5, "crps_bars"), (6, "growth"), (7, "crps_by_index")) for v in ("raw", "harmonised")}
                | {f"f{n}_{s}__raw__{k}.png" for n, s in ((5, "crps_bars"), (6, "growth"), (7, "crps_by_index")) for k in ("e3_vcap", "e3_step")})  # fmt: skip
    assert {p.name for p in (out / "figures").iterdir()} == expected
    assert sorted(p.name for p in (out / "figures" / "f1_maps__e1").iterdir()) == sorted(p.name for p in (roots["e1"] / "figures" / "maps").iterdir())
    for name in expected:
        assert f"`figures/{name}" in index, name
    assert "`deep_ensemble__model.png`:" in index and "## Not found" not in index
    for group in ("T1", "T2", "T3", "T4", "T5", "T6", "F1", "F2", "F3", "F4", "F5", "F6", "F7"):
        assert f"### {group} " in index, group
    # the measured platoons' index 6 is NaN in growth.csv: reported as a note, not an error
    assert any("growth.csv" in n and "[6]" in n for n in result["notes"])


def test_report_contents(report: Any, roots: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "report"
    report.make_report(out, roots)
    t1 = pd.read_csv(out / "tables" / "t1_one_step__raw__a.csv")
    row = t1[(t1["model"] == "deep_ensemble") & (t1["dataset"] == "ngsim")].iloc[0]
    assert row["n_seeds"] == 3 and row["nll_mean"] == pytest.approx(2.0) and row["nll_sd"] == pytest.approx(1.0)  # seeds 1, 2, 3
    assert set(t1["dataset_label"]) == {"highD", "highD validation", "NGSIM I-80", "NGSIM I-80 (FollowNet)"}
    assert "2.00 $\\pm$ 1.00" in (out / "tables" / "t1_one_step__raw__a.tex").read_text(encoding="utf-8")
    assert list(t1["model"].drop_duplicates()) == ["idm_white_noise", "lstm_const_sigma", "hetero_mdn", "deep_ensemble"]
    t2 = pd.read_csv(out / "tables" / "t2_h31__raw__a.csv")
    assert len(t2) == 9 and t2.loc[t2["kind"] == "seed_ensemble", "geo_epi"].isna().all()  # subset "all" only
    assert (t2.loc[t2["kind"] == "model", "geo_epi"] == 9.0).all()
    # T5: per driver mode; Human (3 profiles) and mixed (1) are below min_profiles 5: listed, not judged
    t5 = pd.read_csv(out / "tables" / "t5_h32__e3__raw.csv")
    assert list(t5["mode"].drop_duplicates()) == ["ACC", "Human", "all", "mixed"] and (t5["set"] == "extended").all()
    assert t5.groupby("mode")["model"].first().eq("idm_white_noise").all()  # the reference leads every mode
    judged = t5.set_index(["mode", "model"])["judged"]
    assert judged[("ACC", "diffusion_cf")] and not judged[("Human", "diffusion_cf")] and not judged[("mixed", "lstm_const_sigma")]
    tex5 = (out / "tables" / "t5_h32__e3__raw.tex").read_text(encoding="utf-8")
    assert "Human (3)" in tex5 and "not judged" in tex5 and "LSTM, constant sigma" in tex5 and "reference" in tex5
    sens = pd.read_csv(out / "tables" / "t5_h32__sensitivities.csv")
    assert set(sens["run"]) == {"e3", "e3_vcap", "e3_step"} and (sens["set"] == "headline").all()
    assert set(sens["run_label"]) == {"main run", "speed cap 35.9 m/s", "ensemble, per-step draws"}
    assert "Deep ensemble (per-step draws)" in set(sens["model_label"])
    growth = pd.read_csv(out / "tables" / "t6_growth.csv")
    emp = growth[growth["model"] == "empirical"]
    assert len(emp) == 4 and (emp["fit_last_profile"] == 4).all() and (emp["index_last_profile"] == 5).all()
    assert emp["sd_fit_last_profile"].iloc[0] == pytest.approx(3.9)
    coll = pd.read_csv(out / "tables" / "t6_collisions.csv")
    assert {"collided_1", "collided_20", "collided_any", "collided_burn_in_1"} <= set(coll.columns) and len(coll) == 3 * 2 + 2 + 2
    counts = pd.read_csv(out / "tables" / "t4_h34_verdict_counts.csv")
    assert len(counts) == 2 * 2 * 3 and (counts["n_cases"] == 4).all() and (counts["supported_point"] == 4).all()
    cov = pd.read_csv(out / "tables" / "t4_coverage__raw__a.csv")
    assert set(cov["dataset"]) == {"highd", "ngsim"}  # the calibration set highd_val is not an evaluation set
    assert cov["picp_aci_online_10s"].iloc[0] == pytest.approx(0.9)


def test_idempotent_and_prunes_only_its_own_outputs(report: Any, roots: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "report"
    report.make_report(out, roots)
    first = _files(out)
    (out / "tables" / "t9_stale.csv").write_bytes(b"x\n")
    (out / "figures" / "f9_stale__e1.png").write_bytes(PNG)
    (out / "figures" / "f1_maps__e1" / "stale.png").write_bytes(PNG)
    (out / "tables" / "notes.txt").write_bytes(b"keep\n")  # not an output of the script: kept
    result = report.make_report(out, roots)
    assert set(result["removed"]) == {"t9_stale.csv", "f9_stale__e1.png"}
    second = _files(out)
    assert second.pop(str(Path("tables") / "notes.txt")) == b"keep\n"
    assert second == first


def test_missing_root_and_cli(report: Any, roots: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "report"
    args = ["--out", str(out), "--roots", *[f"{k}={v}" for k, v in roots.items() if k != "e2h"], f"e2h={tmp_path / 'absent'}"]
    assert report.main(args) == 0
    printed = capsys.readouterr().out
    assert "tables (csv + tex)" in printed and "T5  t5_h32__e3__raw" in printed and "not found: e2h" in printed
    assert not (out / "tables" / "t4_coverage__harmonised__a.csv").exists() and (out / "tables" / "t4_coverage__raw__a.csv").exists()
    index = (out / "index.md").read_text(encoding="utf-8")
    assert "## Not found" in index and "e2h: tables/coverage_by_horizon.csv" in index
    with pytest.raises(SystemExit):
        report.main(["--roots", "e9=somewhere"])


def test_matched_roots_join_t1_t2_t4_and_f2_f4(report: Any, roots: dict[str, Path], tmp_path: Path) -> None:
    roots["e1m"].mkdir(parents=True)
    (roots["e1m"] / "matching.json").write_text("{}", encoding="utf-8")  # an E5 root before its analysis
    base = tmp_path / "base"
    assert not report.make_report(base, roots)["missing"]  # E5 roots absent or not analysed: skipped silently
    index = (base / "index.md").read_text(encoding="utf-8")
    assert all(key not in index for key in MATCHED) and "matched signals" not in index
    add_matched_roots(roots)
    out = tmp_path / "matched"
    result = report.make_report(out, roots)
    assert not result["missing"], result["missing"]
    extra = ([f"t1_one_step__matched__{t}" for t in ("a", "mean_1s")] + [f"t2_h31__matched__{t}" for t in ("a", "mean_1s")]
             + [f"t4_{k}__matched__{t}" for k in ("coverage", "onestep_conformal") for t in ("a", "mean_1s")])  # fmt: skip
    assert sorted(t.name for t in result["tables"]) == sorted([*EXPECTED_TABLES, *extra])  # no T3 of the E5 roots
    for name in extra:
        check_latex((out / "tables" / f"{name}.tex").read_text(encoding="utf-8"))
    t1 = pd.read_csv(out / "tables" / "t1_one_step__matched__a.csv")
    assert set(t1["variant"]) == {"matched"} and len(t1) == 16
    assert "roughness-matched signals" in (out / "tables" / "t1_one_step__matched__a.tex").read_text(encoding="utf-8")
    assert list(pd.read_csv(out / "tables" / "t2_h31_summary.csv")["variant"].drop_duplicates()) == ["raw", "harmonised", "matched"]
    assert list(pd.read_csv(out / "tables" / "t4_h34_verdict_counts.csv")["variant"].drop_duplicates()) == ["raw", "harmonised", "matched"]
    added = {p.name for p in (out / "figures").iterdir()} - {p.name for p in (base / "figures").iterdir()}
    assert added == ({f"f2_{s}__{k}.png" for s in ("shift_ratios", "shift_ratios_matched", "shift_ratios_controls")
                      for k in ("e1m", "e1m_mean1s")}
                     | {f"f3_nll_by_regime__{k}.png" for k in ("e1m", "e1m_mean1s")}
                     | {f"f3_regime_variance__{k}" for k in ("e1m", "e1m_mean1s")}
                     | {f"f4_{s}__e2m.png" for s in ("coverage_by_horizon__a", "coverage_by_horizon__a__ngsim", "width_by_horizon__a")})  # fmt: skip
    index = (out / "index.md").read_text(encoding="utf-8")
    assert all(f"| {key} |" in index for key in MATCHED) and "f1_maps__e1m" not in index
    # the outputs of the main roots are unchanged: only the summaries over all roots and the index gain rows
    shared = {"t2_h31_summary.csv", "t2_h31_summary.tex", "t4_h34_verdict_counts.csv", "t4_h34_verdict_counts.tex", "index.md"}
    before, after = _files(base), _files(out)
    assert {k: v for k, v in before.items() if Path(k).name not in shared} == {k: after[k] for k in before if Path(k).name not in shared}
    assert report._parse_roots([f"e1m={roots['e1m']}"]) == {"e1m": str(roots["e1m"])}


# ----------------------------------------------------------------------------- figure F5 (figures_e3)
def _crps_rows(values: dict[str, float], modes: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame([
        {"variant": "raw", "model": m, "set": "extended", "mode": mode, "window": w, "index": index, "crps": v * (1 + 0.1 * k),
         "crps_lo": 0.8 * v, "crps_hi": 1.3 * v, "n_profiles": n}
        for k, (mode, n) in enumerate(modes.items()) for m, v in values.items() for w in ("response", "profile") for index in ("1", "all")
    ])  # fmt: skip


def test_crps_bars_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []
    save = figures_e3._save

    def spy(fig: plt.Figure, path: Path) -> Path:
        axes = [ax for ax in fig.axes if ax.get_title(loc="left")]  # panel titles are left-aligned (house style)
        seen.append({"titles": [ax.get_title(loc="left") for ax in axes], "scale": axes[0].get_yscale(),
                     "bars": [len(ax.containers[0]) for ax in axes], "lines": [len(ax.lines) for ax in axes]})  # fmt: skip
        return save(fig, path)

    monkeypatch.setattr(figures_e3, "_save", spy)
    modes = {"ACC": 47, "Human": 12, "all": 60, "mixed": 1}
    wide = _crps_rows({"idm_white_noise": 3.0, "lstm_const_sigma": 58.0, "diffusion_cf": 2.2}, modes)
    path = figures_e3.plot_crps_bars(wide, "raw", tmp_path / "crps_bars__raw.png")
    assert path is not None and path.exists() and path.stat().st_size > 5000
    first = seen[-1]
    assert first["titles"] == ["ACC platoons", "Human-driven platoons", "All profiles"]  # mixed is not drawn
    assert first["scale"] == "log" and first["bars"] == [2, 2, 2] and all(n >= 1 for n in first["lines"])  # reference line
    narrow = _crps_rows({"idm_white_noise": 3.0, "lstm_const_sigma": 4.0}, {"ACC": 6, "all": 10})
    assert figures_e3.plot_crps_bars(narrow, "raw", tmp_path / "narrow.png") is not None
    assert seen[-1]["scale"] == "linear" and seen[-1]["bars"] == [1, 1] and len(seen[-1]["titles"]) == 2
    only_reference = _crps_rows({"idm_white_noise": 3.0}, {"all": 10})
    assert figures_e3.plot_crps_bars(only_reference, "raw", tmp_path / "none.png") is None
    assert figures_e3.plot_crps_bars(wide, "harmonised", tmp_path / "none.png") is None
    assert figures_e3.MODEL_LABEL_E3["deep_ensemble__step"] == "Deep ensemble (per-step draws)"


def test_pooled_roots_join_t1_t2_t4_and_f2_f4(report: Any, roots: dict[str, Path], tmp_path: Path) -> None:
    base = tmp_path / "base"
    report.make_report(base, roots)
    add_pooled_roots(roots)
    out = tmp_path / "pooled"
    result = report.make_report(out, roots)
    assert not result["missing"], result["missing"]  # the pooled root has no shift pair: no shift figure expected
    variants = ("pool", "lodo_ngsim", "lodo_openacc", "lodo_ngsim_all")
    extra = ([f"t1_one_step__{v}__a" for v in variants] + [f"t2_h31__{v}__a" for v in variants]
             + [f"t4_{k}__{v}__a" for k in ("coverage", "onestep_conformal") for v in variants])  # fmt: skip
    assert sorted(t.name for t in result["tables"]) == sorted([*EXPECTED_TABLES, *extra])  # target a only, no T3
    for name in extra:
        check_latex((out / "tables" / f"{name}.tex").read_text(encoding="utf-8"))
    t2 = (out / "tables" / "t2_h31__pool__a.tex").read_text(encoding="utf-8")
    assert "pooled test folds" in t2 and "matched signals, pooled training domain" in t2
    assert set(pd.read_csv(out / "tables" / "t2_h31__pool__a.csv")["pair"]) == {"placebo"}
    summary = pd.read_csv(out / "tables" / "t2_h31_summary.csv")
    assert list(summary["variant"].drop_duplicates()) == ["raw", "harmonised", *variants]
    counts = pd.read_csv(out / "tables" / "t4_h34_verdict_counts.csv")
    assert list(counts["variant"].drop_duplicates()) == ["raw", "harmonised", *variants]
    assert set(counts.loc[counts["variant"].isin(variants), "target"]) == {"a"}
    added = {p.name for p in (out / "figures").iterdir()} - {p.name for p in (base / "figures").iterdir()}
    e1 = ("e6_pool", "e6_lodo_ngsim", "e6_lodo_openacc", "e6_lodo_ngsim_all")
    assert added == ({"f2_shift_ratios_controls__e6_pool.png"}
                     | {f"f2_{s}__{k}.png" for s in ("shift_ratios", "shift_ratios_matched", "shift_ratios_controls")
                        for k in e1[1:]}
                     | {f"f3_nll_by_regime__{k}.png" for k in e1} | {f"f3_regime_variance__{k}" for k in e1}
                     | {f"f4_{s}__{k}.png" for s in ("coverage_by_horizon__a", "coverage_by_horizon__a__ngsim", "width_by_horizon__a")
                        for k in ("e2_pool", "e2_lodo_ngsim", "e2_lodo_openacc", "e2_lodo_ngsim_all")})  # fmt: skip
    index = (out / "index.md").read_text(encoding="utf-8")
    assert all(f"| {key} |" in index for key in POOLED) and "f1_maps__e6_pool" not in index
    assert report._parse_roots([f"e6_pool={roots['e6_pool']}"]) == {"e6_pool": str(roots["e6_pool"])}
    # a missing shift figure of a root that has a shift pair is still reported
    (roots["e6_lodo_ngsim"] / "figures" / "shift_ratios.png").unlink()
    assert report.make_report(tmp_path / "again", roots)["missing"] == ["e6_lodo_ngsim: figures/shift_ratios.png"]


def test_e3_pooled_variants_join_t5_t6_and_f5_f7(report: Any, roots: dict[str, Path], tmp_path: Path,
                                                 e3_columns: dict[str, list[str]]) -> None:  # fmt: skip
    """E3-pooled (design.md 17.1): the variants pool and lodo_openacc of the main E3 root get their own T5 tables, rows
    in T6 and figures F5-F7 with the labels of figures_e3; the outputs of the other variants do not change."""
    base = tmp_path / "base"
    report.make_report(base, roots)
    models = ("idm_white_noise", "lstm_const_sigma", "quantile_pidl", "diffusion_cf")
    fake_e3(roots["e3"], "e3", e3_columns, extra={"pool": models, "lodo_openacc": models})
    out = tmp_path / "pooled"
    result = report.make_report(out, roots)
    assert not result["missing"], result["missing"]
    assert sorted(t.name for t in result["tables"]) == sorted([*EXPECTED_TABLES, "t5_h32__e3__pool", "t5_h32__e3__lodo_openacc"])
    for name in ("t5_h32__e3__pool", "t5_h32__e3__lodo_openacc", "t6_collisions", "t6_amplitude_coverage", "t6_growth"):
        check_latex((out / "tables" / f"{name}.tex").read_text(encoding="utf-8"))
    t5 = pd.read_csv(out / "tables" / "t5_h32__e3__lodo_openacc.csv")
    assert set(t5["variant"]) == {"lodo_openacc"} and (t5["set"] == "extended").all()
    assert list(t5["mode"].drop_duplicates()) == ["ACC", "Human", "all", "mixed"]
    assert dict(zip(t5["model"], t5["model_label"]))["quantile_pidl"] == "Quantile PIDL (one-step target)"
    assert figures_e3.model_labels("raw")["quantile_pidl"] == "Quantile PIDL (1 s target)"  # the target of 13.2
    tex = (out / "tables" / "t5_h32__e3__pool.tex").read_text(encoding="utf-8")
    assert "pooled domain, OpenACC seen" in tex and "Quantile PIDL (one-step target)" in tex
    coll = pd.read_csv(out / "tables" / "t6_collisions.csv")
    main = coll[coll["run"] == "e3"]
    assert list(main["variant"].drop_duplicates()) == ["raw", "harmonised", "pool", "lodo_openacc"]
    assert set(main.loc[main["variant"] == "lodo_openacc", "run_label"]) == {"main run, pooled domain, OpenACC held out (10 profiles)"}
    assert set(main.loc[main["variant"] == "raw", "run_label"]) == {"main run, recorded signals (10 profiles)"}
    growth = pd.read_csv(out / "tables" / "t6_growth.csv")
    assert {"Quantile PIDL (one-step target)", "measured platoons"} <= set(growth.loc[growth["variant"] == "pool", "model_label"])
    added = {p.name for p in (out / "figures").iterdir()} - {p.name for p in (base / "figures").iterdir()}
    assert added == {f"f{n}_{s}__{v}__e3.png" for n, s in ((5, "crps_bars"), (6, "growth"), (7, "crps_by_index"))
                     for v in ("pool", "lodo_openacc")}  # fmt: skip
    index = (out / "index.md").read_text(encoding="utf-8")
    assert "further variants: pool, lodo_openacc" in index and "pooled domain, OpenACC held out, extended set." in index
    assert "E3 platoons, main run (both signal variants)" in (base / "index.md").read_text(encoding="utf-8")
    # every other output is byte-identical: the T5 tables of raw / harmonised, T1-T4, the sensitivities, the figures
    before, after = _files(base), _files(out)
    changed = {k for k in before if before[k] != after.get(k)}
    assert changed == {str(Path("tables") / f"{n}.{x}") for n in ("t6_collisions", "t6_amplitude_coverage", "t6_growth")
                       for x in ("csv", "tex")} | {"index.md"}  # fmt: skip
