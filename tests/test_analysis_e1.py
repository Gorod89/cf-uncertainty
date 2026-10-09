"""End-to-end test of cf_unc.maps.analysis.analyze_e1 on a small synthetic experiment.

The experiment is written with cf_unc.maps.io into a temporary directory (windows tables, prediction tables of
3 models x 3 seeds, train_density.npz, idm.json):

* data sets: ``highd`` (in-domain), ``highd_val`` (another split of the highd source: placebo), ``ngsim``
  (planted shift: epistemic x4, aleatoric x1.1) and ``openacc`` (subsets ACC / Human; planted non-shift of
  the epistemic variance);
* models: ``deep_ensemble`` (own decomposition), ``hetero_mdn`` (no epistemic output, 25 % of the rows not
  evaluated, across-seed spread of the means x2 on ngsim), ``lstm_const_sigma`` (constant variance,
  epistemic 0) and ``idm_white_noise`` (seed independent: the same table for every seed, as the runner
  copies its checkpoint).

Every number checked against a direct computation is recomputed here from the stored tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from cf_unc.eval.bootstrap import holm
from cf_unc.eval.metrics import calibration_error, crps_gaussian, ks_uniform
from cf_unc.maps import io
from cf_unc.maps.analysis import TABLE_COLUMNS, analyze_e1, format_report, holm_replicates
from cf_unc.maps.grid import StateGrid, UncertaintyMap, rare_flags
from cf_unc.maps.regimes import REGIMES, label_regimes

MODELS = ("deep_ensemble", "hetero_mdn", "lstm_const_sigma", "idm_white_noise")
SEEDS = (0, 1, 2)
V0 = 30.0
Z90 = 1.6448536269514722
# name: (source, events, windows per event, subsets, gap range m, epistemic level, aleatoric factor, seed spread)
DATASETS = {
    "highd": ("highd", 160, 50, ("all",), (4.0, 110.0), 0.010, 1.0, 0.05),
    "highd_val": ("highd", 80, 50, ("all",), (4.0, 110.0), 0.010, 1.0, 0.05),
    "ngsim": ("ngsim", 120, 50, ("all",), (4.0, 110.0), 0.040, 1.1, 0.10),
    "openacc": ("openacc", 120, 50, ("ACC", "Human"), (4.0, 60.0), 0.010, 1.0, 0.05),
}


def _table(y: np.ndarray, mean: np.ndarray, var: np.ndarray, alea: np.ndarray, epi: np.ndarray) -> pd.DataFrame:
    """Prediction table of Gaussian laws N(mean, var) at the targets y (columns of io.PRED_COLUMNS)."""
    sd = np.sqrt(var)
    z = (y - mean) / sd
    columns = {
        "mean": mean,
        "var": var,
        "aleatoric_var": alea,
        "epistemic_var": epi,
        "nll": 0.5 * z**2 + np.log(sd) + 0.5 * np.log(2 * np.pi),
        "crps": crps_gaussian(mean, sd, y),
        "width": np.full(y.size, 2 * Z90) * sd,
        "pit": ndtr(z),
    }
    frame = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in columns.items()})
    frame["covered"] = (np.abs(z) <= Z90).astype(np.uint8)
    return frame[list(io.PRED_COLUMNS)]


def _write_experiment(root: Path) -> None:
    rng = np.random.default_rng(2026)
    grid = StateGrid()
    for name, (source, n_events, per, subsets, s_range, epi_level, alea_factor, spread) in DATASETS.items():
        n = n_events * per
        event = np.repeat(np.arange(n_events), per)
        s = rng.uniform(*s_range, n)
        dv = rng.normal(0.0, 1.5, n)
        v = rng.uniform(2.0, 32.0, n)
        a_prev = rng.normal(0.0, 0.8, n)
        mu = -0.3 * dv
        alea_true = 0.3 * alea_factor * (1.0 + np.exp(-s / 15.0))  # heteroscedastic in the gap
        y = mu + np.sqrt(alea_true) * rng.standard_normal(n)
        frame = pd.DataFrame(
            {
                "event_index": event.astype(np.int32),
                "t_index": np.tile(np.arange(29, 29 + per), n_events).astype(np.int32),
                "event": event.astype(np.int32),
                "driver": (event // 2).astype(np.int32),
                "fold": np.zeros(n, dtype=np.int8),
                "subset": pd.Categorical.from_codes(rng.integers(0, len(subsets), n_events)[event], categories=subsets),
                "s": s.astype(np.float32),
                "dv": dv.astype(np.float32),
                "v": v.astype(np.float32),
                "v_lead": (v - dv).astype(np.float32),
                "a_prev": a_prev.astype(np.float32),
                "y": y.astype(np.float32),
            }
        )
        meta = {
            "n_windows": n,
            "n_events": n_events,
            "event_ids": [f"{name}-{i}" for i in range(n_events)],
            "driver_names": [f"{name}/{i}" for i in range(n_events // 2)],
            "subsets": list(subsets),
            "hist_len": 30,
            "dataset": name,
            "spec": {"data": source, "split": "test", "stride": 1},
        }
        io.save_windows(root, name, frame, meta)
        y_stored = frame["y"].to_numpy(np.float64)
        epi_event = rng.gamma(20.0, 0.05, n_events)[event]  # event factor of the epistemic variance (CV 0.22)
        skipped = rng.random(n) < 0.25  # rows hetero_mdn does not evaluate (the same for every seed)
        idm = _table(y_stored, mu - 0.1, np.full(n, 0.5), np.full(n, 0.5), np.zeros(n))  # one fit for all seeds
        for model in MODELS:
            for seed in SEEDS:
                noise = rng.gamma(400.0, 1.0 / 400.0, n)  # 5 % noise of the variance
                if model == "idm_white_noise":
                    io.save_preds(root, model, seed, name, idm)
                    continue
                if model == "deep_ensemble":
                    mean = mu + rng.normal(0.0, 0.05, n)
                    alea, epi = alea_true * noise, epi_level * epi_event * (1.0 + 0.1 * seed)
                elif model == "hetero_mdn":
                    mean = mu + rng.normal(0.0, spread, n)
                    alea, epi = alea_true * noise, np.full(n, np.nan)
                else:
                    mean = mu + rng.normal(0.0, 0.05, n)
                    alea, epi = np.full(n, 0.45), np.zeros(n)
                table = _table(y_stored, mean, alea + np.nan_to_num(epi), alea, epi)
                if model == "hetero_mdn":
                    table.loc[skipped, list(io.PRED_FLOAT)] = np.nan
                    table.loc[skipped, "covered"] = 0  # as the runner stores rows it did not evaluate
                io.save_preds(root, model, seed, name, table)
    s_train, dv_train = rng.uniform(4.0, 110.0, 30_000), rng.normal(0.0, 1.5, 30_000)
    counts = np.bincount(grid.bin_index(s_train, dv_train), minlength=grid.n_cells)
    np.savez(
        root / "train_density.npz",
        s_edges=grid.s_edges,
        dv_edges=grid.dv_edges,
        counts=counts[: grid.n_bins].reshape(grid.n_s, grid.n_dv),
        n_outside=np.int64(counts[-1]),
        n_total=np.int64(counts.sum()),
        stride=np.int64(1),
    )
    (root / "idm.json").write_text(json.dumps({"key": {}, "params": {"v0": V0, "T": 1.5}}), encoding="utf-8")


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, pd.DataFrame]]:
    root = tmp_path_factory.mktemp("e1")
    _write_experiment(root)
    tables = analyze_e1(root, {"bootstrap": {"n_boot": 1000}})
    return root, tables


def _stored(root: Path, model: str, dataset: str, column: str) -> np.ndarray:
    return np.vstack([io.load_preds(root, model, s, dataset)[column].to_numpy(np.float64) for s in SEEDS])


def _one(frame: pd.DataFrame, **keys: str) -> pd.Series:
    mask = np.ones(len(frame), dtype=bool)
    for key, value in keys.items():
        mask &= frame[key].astype(str).to_numpy() == value
    assert mask.sum() == 1, (keys, int(mask.sum()))
    return frame[mask].iloc[0]


def test_every_output_exists_with_documented_columns(experiment) -> None:
    root, tables = experiment
    for name, columns in TABLE_COLUMNS.items():
        written = pd.read_csv(root / "tables" / f"{name}.csv")
        assert list(written.columns) == columns, name
        assert list(tables[name].columns) == columns and len(written) == len(tables[name]) > 0, name
    maps_long = pd.read_csv(root / "tables" / "maps_long.csv")
    groups = maps_long.groupby(["model", "kind", "dataset", "subset"]).size()
    assert (groups == 481).all()
    # every model has both kinds; every data set has "all", openacc also its two subsets
    assert len(groups) == len(MODELS) * 2 * (4 + 2)
    for model in MODELS:
        for kind in ("model", "seed_ensemble"):
            for stem in ("highd", "highd_val", "ngsim", "openacc", "openacc__ACC", "openacc__Human"):
                assert (root / "maps" / f"{model}__{kind}" / f"{stem}.npz").exists()
    info = json.loads((root / "tables" / "analysis_info.json").read_text(encoding="utf-8"))
    assert info["placebo"] == ["highd_val"] and info["v0"] == V0 and info["seeds"]["hetero_mdn"]["ngsim"] == [0, 1, 2]
    # p-values of Holm families get enough replicates: 20 m / alpha (m = 10 and 6 views x 7 groups = 42)
    assert info["n_boot_tests"] == {"regime_ordering": 4000, "h33_nll": 16800}
    assert holm_replicates(1000, 3) == 1200 and holm_replicates(5000, 3) == 5000
    report = format_report(tables)
    assert "H3.1 summary" in report and "H3.3" in report


def test_one_step_metrics_match_direct_computation(experiment) -> None:
    root, tables = experiment
    metrics = tables["one_step_metrics"]
    windows, _ = io.load_windows(root, "ngsim")
    y = windows["y"].to_numpy(np.float64)
    nll, mean = _stored(root, "hetero_mdn", "ngsim", "nll"), _stored(root, "hetero_mdn", "ngsim", "mean")
    covered, var = _stored(root, "hetero_mdn", "ngsim", "covered"), _stored(root, "hetero_mdn", "ngsim", "var")
    ok = np.isfinite(mean[1])
    row = _one(metrics, model="hetero_mdn", dataset="ngsim", subset="all", seed="1")
    assert row["n_windows"] == ok.sum() < len(y)  # not-evaluated rows are left out
    assert row["nll"] == pytest.approx(nll[1, ok].mean(), rel=1e-9)
    assert row["picp90"] == pytest.approx(covered[1, ok].mean(), rel=1e-9)  # the stored 0 of skipped rows ignored
    assert row["rmse"] == pytest.approx(np.sqrt(np.mean((y[ok] - mean[1, ok]) ** 2)), rel=1e-9)
    assert row["aleatoric"] == pytest.approx(var[1, ok].mean(), rel=1e-9) and np.isnan(row["epistemic"])
    pit = _stored(root, "hetero_mdn", "ngsim", "pit")[1, ok]
    assert row["pit_ks"] == pytest.approx(ks_uniform(pit)[0], rel=1e-12)  # the statistic of the shared metric
    assert row["calibration_error"] == pytest.approx(calibration_error(pit), rel=1e-12)
    seeds = metrics[(metrics["model"] == "hetero_mdn") & (metrics["dataset"] == "ngsim") & (metrics["subset"] == "all")]
    mean_row = _one(seeds, seed="mean")
    assert mean_row["nll"] == pytest.approx(seeds[seeds["seed"] != "mean"]["nll"].mean(), rel=1e-12)
    for q in ("nll", "crps", "picp90"):
        assert mean_row[f"{q}_lo"] < mean_row[q] < mean_row[f"{q}_hi"]
    assert 0.85 < mean_row["picp90"] < 0.95  # calibrated Gaussian laws
    assert mean_row["n_seeds"] == 3 and mean_row["unit"] == "event"
    acc = _one(metrics, model="deep_ensemble", dataset="openacc", subset="ACC", seed="0")
    in_acc = io.load_windows(root, "openacc")[0]["subset"].astype(str).to_numpy() == "ACC"
    assert 0 < acc["n_windows"] == in_acc.sum() < in_acc.size


def test_shift_ratios_match_direct_computation(experiment) -> None:
    root, tables = experiment
    shift = tables["shift_ratios"]
    epi_d = _stored(root, "deep_ensemble", "ngsim", "epistemic_var")
    epi_i = _stored(root, "deep_ensemble", "highd", "epistemic_var")
    row = _one(shift, model="deep_ensemble", kind="model", dataset="ngsim", subset="all", quantity="epistemic")
    assert row["estimate"] == pytest.approx(epi_d.mean(axis=1).mean() / epi_i.mean(axis=1).mean(), rel=1e-9)
    assert row["pair"] == "shift" and row["n_events"] == 120 and row["n_events_in"] == 160 and row["n_seeds"] == 3
    per_seed = epi_d.mean(axis=1) / epi_i.mean(axis=1)
    assert row["seed_sd"] == pytest.approx(per_seed.std(ddof=1), rel=1e-9)
    assert row["lo"] < row["estimate"] < row["hi"] and row["n_bins_matched"] > 50
    assert 0.0 < row["mass_in"] <= 1.0 and 0.0 < row["mass_shift"] <= 1.0
    # across-seed decomposition of hetero_mdn: variance of the seeds' means (ddof = 0) over the evaluated rows
    m_d, m_i = _stored(root, "hetero_mdn", "ngsim", "mean"), _stored(root, "hetero_mdn", "highd", "mean")
    expected = np.nanmean(np.var(m_d, axis=0)) / np.nanmean(np.var(m_i, axis=0))
    ens = _one(shift, model="hetero_mdn", kind="seed_ensemble", dataset="ngsim", subset="all", quantity="epistemic")
    assert ens["estimate"] == pytest.approx(expected, rel=1e-9) and 3.5 < ens["estimate"] < 4.5  # planted x4
    # models without an own epistemic part have no epistemic rows of kind "model"
    for model in ("hetero_mdn", "lstm_const_sigma"):
        own = shift[(shift["model"] == model) & (shift["kind"] == "model")]
        assert set(own["quantity"]) == {"aleatoric", "total"}
    assert set(shift["pair"]) == {"shift", "placebo"}
    assert set(shift.loc[shift["pair"] == "placebo", "dataset"]) == {"highd_val"}
    placebo = _one(shift, model="deep_ensemble", kind="model", dataset="highd_val", subset="all", quantity="epistemic")
    assert placebo["lo"] < 1.0 < placebo["hi"]  # the placebo ratio comes out near 1
    subsets = set(shift.loc[shift["dataset"] == "openacc", "subset"])
    assert subsets == {"all", "ACC", "Human"}


def test_planted_shift_supported_and_non_shift_refuted(experiment) -> None:
    _, tables = experiment
    verdicts = tables["h31_verdicts"]
    shifted = _one(verdicts, model="deep_ensemble", kind="model", dataset="ngsim", subset="all")
    assert shifted["r_epi_lo"] >= 3.0 and 0.7 < shifted["r_alea_lo"] and shifted["r_alea_hi"] < 1.3
    assert shifted["verdict_ci"] == "supported" and shifted["verdict_point"] == "supported"
    same = _one(verdicts, model="deep_ensemble", kind="model", dataset="openacc", subset="all")
    assert same["r_epi_hi"] < 1.5 and same["verdict_ci"] == "refuted"
    ens = _one(verdicts, model="hetero_mdn", kind="seed_ensemble", dataset="ngsim", subset="all")
    assert ens["verdict_ci"] == "supported"  # planted x4 of the across-seed variance
    assert not ((verdicts["model"] == "hetero_mdn") & (verdicts["kind"] == "model")).any()  # no own epistemic part
    # identical seeds (idm_white_noise): the seed epistemic variance is 0, its ratio 0 / 0 is NaN and gets no verdict
    assert not (verdicts["model"] == "idm_white_noise").any()
    idm = tables["shift_ratios"]
    idm = idm[(idm["model"] == "idm_white_noise") & (idm["kind"] == "seed_ensemble") & (idm["quantity"] == "epistemic")]
    assert len(idm) > 0 and idm["estimate"].isna().all() and (idm["mean_in"] == 0).all()
    summary = tables["h31_summary"].set_index(["model", "kind"])
    de = summary.loc[("deep_ensemble", "model")]
    assert (de["n_pairs"], de["n_supported"], de["n_refuted"], de["overall"]) == (2, 1, 1, "refuted")
    assert de["n_placebo"] == 1 and de["placebo_verdicts"] == "highd_val:refuted"


def test_maps_rare_concentration_and_regimes_match_direct_computation(experiment) -> None:
    root, tables = experiment
    grid = StateGrid()
    windows, _ = io.load_windows(root, "highd")
    bins = grid.bin_index(windows["s"].to_numpy(), windows["dv"].to_numpy())
    epi = _stored(root, "deep_ensemble", "highd", "epistemic_var")
    cell = int(np.argmax(np.bincount(bins, minlength=grid.n_cells)[: grid.n_bins]))
    umap = UncertaintyMap.load(root / "maps" / "deep_ensemble__model" / "highd.npz")
    per_seed = [epi[k, bins == cell].mean() for k in range(3)]
    assert umap.mean["epistemic"][cell] == pytest.approx(np.mean(per_seed), rel=1e-9)
    assert umap.std["epistemic"][cell] == pytest.approx(np.std(per_seed, ddof=1), rel=1e-9)
    assert umap.count[cell] == (bins == cell).sum() and umap.count.sum() == len(windows)
    long = tables["maps_long"]
    row = _one(long[long["cell"] == cell], model="deep_ensemble", kind="model", dataset="highd", subset="all")
    assert row["epistemic"] == pytest.approx(umap.mean["epistemic"][cell], rel=1e-12)
    hetero = UncertaintyMap.load(root / "maps" / "hetero_mdn__model" / "highd.npz")
    assert hetero.count.sum() == np.isfinite(_stored(root, "hetero_mdn", "highd", "mean")[0]).sum()
    # rare-state shares on the training density of train_density.npz
    with np.load(root / "train_density.npz") as z:
        train = np.r_[z["counts"].ravel(), z["n_outside"]]
    flags = rare_flags(train)
    conc = tables["rare_concentration"]
    rule = _one(conc, model="deep_ensemble", kind="model", dataset="highd", subset="all", rule="rare_mass")
    assert rule["share_windows"] == pytest.approx(flags["rare_mass"][bins].mean(), rel=1e-12)
    epi_bar = epi.mean(axis=0)
    in_rare = flags["rare_mass"][bins]
    assert rule["share_mass_epistemic"] == pytest.approx(epi_bar[in_rare].sum() / epi_bar.sum(), rel=1e-9)
    gap = _one(conc, model="lstm_const_sigma", kind="model", dataset="highd", subset="all", rule="small_gap")
    assert np.isnan(gap["ratio_epistemic"]) and gap["ratio_aleatoric"] == pytest.approx(1.0, rel=1e-6)  # constant
    assert set(conc["rule"]) == {"rare_freq", "rare_mass", "small_gap", "large_dv"}
    # regimes of the stored states with v0 of idm.json
    labels = label_regimes(windows["s"], windows["dv"], windows["v"], windows["a_prev"], V0)
    stats = tables["regime_stats"]
    for name in ("braking", "steady", "approaching"):
        r = _one(stats, model="deep_ensemble", dataset="highd", subset="all", regime=name)
        rows = labels == REGIMES.index(name)
        assert r["share"] == pytest.approx(rows.mean(), rel=1e-12)
        assert r["epistemic"] == pytest.approx(np.mean([epi[k, rows].mean() for k in range(3)]), rel=1e-9)
    # the heteroscedastic truth: the squared residual of small-gap regimes exceeds that of steady driving
    order = tables["regime_ordering"]
    for (model, dataset, subset), grp in order.groupby(["model", "dataset", "subset"]):
        adjusted, _ = holm(grp["pvalue"].to_numpy(dtype=float))
        np.testing.assert_allclose(grp["p_holm"].to_numpy(dtype=float), adjusted, equal_nan=True)
    assert set(order["quantity"]) == {"var", "mse"} and set(order["reference"]) == {"steady"}


def test_h33_nll_comparison(experiment) -> None:
    root, tables = experiment
    h33 = tables["h33_nll"]
    assert set(h33["dataset"]) == set(DATASETS)
    assert set(h33["group"]) == {"all", *REGIMES}
    adjusted, reject = holm(h33["pvalue"].to_numpy(dtype=float))  # Holm over the whole table
    np.testing.assert_allclose(h33["p_holm"].to_numpy(dtype=float), adjusted, equal_nan=True)
    row = _one(h33, dataset="highd", subset="all", group="all")
    nll_a = _stored(root, "hetero_mdn", "highd", "nll")
    nll_b = _stored(root, "lstm_const_sigma", "highd", "nll")
    paired = np.isfinite(nll_a).any(axis=0)
    diff = np.nanmean(nll_a[:, paired], axis=0) - nll_b[:, paired].mean(axis=0)
    assert row["diff"] == pytest.approx(diff.mean(), rel=1e-9) and row["n_windows"] == paired.sum()
    assert row["diff"] < 0 and row["diff_hi"] < 0 and row["reject"]  # the heteroscedastic model wins
    models_and_seeds = (row["model_a"], row["model_b"], row["n_seeds_a"], row["n_seeds_b"])
    assert models_and_seeds == ("hetero_mdn", "lstm_const_sigma", 3, 3)
    assert isinstance(row["ci_overlap"], (bool, np.bool_))


def test_filters_units_and_missing_models(experiment) -> None:
    root, _ = experiment
    with pytest.warns(RuntimeWarning, match="H3.3"):
        only = analyze_e1(root, {"bootstrap": {"n_boot": 50}}, models=["deep_ensemble"], datasets=["ngsim"],
                          unit="driver", write=False)  # fmt: skip
    assert only["h33_nll"].empty and list(only["h33_nll"].columns) == TABLE_COLUMNS["h33_nll"]
    assert set(only["one_step_metrics"]["dataset"]) == {"ngsim"}  # highd is loaded as the reference only
    row = _one(only["shift_ratios"], kind="model", dataset="ngsim", subset="all", quantity="epistemic")
    assert row["unit"] == "driver" and row["n_events"] == 60 and row["n_events_in"] == 80  # two events per driver
    with pytest.warns(RuntimeWarning, match="H3.3"):
        none = analyze_e1(root, {"bootstrap": {"n_boot": 50}}, models=["deep_ensemble"], datasets=["highd_val"],
                          placebo=[], write=False)  # fmt: skip
    assert set(none["h31_verdicts"]["pair"]) == {"shift"}  # placebo=[]: every other data set counts as a pair
    assert none["h31_summary"].set_index("kind").loc["model", "n_pairs"] == 1


def test_h31e_tables_follow_concentration_and_shift_ratios(experiment) -> None:
    from cf_unc.maps.analysis import h31e_tables
    from cf_unc.maps.shift import H31E_SUMMARY_COLUMNS, H31E_VERDICT_COLUMNS

    root, tables = experiment
    ver, summ = tables["h31e_verdicts"], tables["h31e_summary"]
    assert list(ver.columns) == H31E_VERDICT_COLUMNS and list(summ.columns) == H31E_SUMMARY_COLUMNS
    h31 = tables["h31_verdicts"]
    assert len(ver) == len(h31)  # one H3.1e row per H3.1 verdict row (same pairs, subsets, kinds)
    assert not ((ver["model"] == "hetero_mdn") & (ver["kind"] == "model")).any()  # no own epistemic output
    rare = tables["rare_concentration"]
    for (model, kind), grp in ver.groupby(["model", "kind"]):
        conc = _one(rare, model=model, kind=kind, dataset="highd", subset="all", rule="rare_mass")
        assert (grp["conc"] == conc["ratio_epistemic"]).all() and (grp["conc_lo"] == conc["ratio_epistemic_lo"]).all()
        assert (grp["conc_hi"] == conc["ratio_epistemic_hi"]).all()
    row = _one(ver, model="deep_ensemble", kind="model", dataset="ngsim", subset="all")
    ref = _one(h31, model="deep_ensemble", kind="model", dataset="ngsim", subset="all")
    assert (row["r_epi"], row["r_epi_lo"], row["r_epi_hi"]) == (ref["r_epi"], ref["r_epi_lo"], ref["r_epi_hi"])
    assert row["verdict_ci"] == "supported" and row["pair"] == "shift"  # planted x4 growth
    same = _one(ver, model="deep_ensemble", kind="model", dataset="openacc", subset="all")
    assert same["verdict_ci"] == "refuted"  # planted non-shift
    de = summ.set_index(["model", "kind"]).loc[("deep_ensemble", "model")]
    assert bool(de["main"]) and de["n_pairs"] == 2 and de["n_refuted"] == 1 and de["overall"] == "refuted"
    assert de["placebo_verdicts"].startswith("highd_val:")
    assert not summ.loc[summ["kind"] == "seed_ensemble", "main"].astype(bool).any()
    csv_ver, csv_summ = h31e_tables(pd.read_csv(root / "tables" / "rare_concentration.csv"),
                                    pd.read_csv(root / "tables" / "h31_verdicts.csv"), load_config_maps())  # fmt: skip
    pd.testing.assert_frame_equal(csv_ver.reset_index(drop=True), pd.read_csv(root / "tables" / "h31e_verdicts.csv"),
                                  check_dtype=False)  # fmt: skip
    written = pd.read_csv(root / "tables" / "h31e_summary.csv").fillna("")  # empty lists are read back as NaN
    pd.testing.assert_frame_equal(csv_summ.fillna("").astype(str), written.astype(str))
    assert "H3.1e summary" in format_report(tables)


def load_config_maps():  # noqa: ANN201 - the maps config of the analysis (sections h31 and h31e)
    from cf_unc.maps.analysis import load_config

    return load_config()
