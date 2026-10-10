"""Experiment E2 runner (cf_unc.experiments.e2) and analysis (scripts/analyze_e2.py) on tiny synthetic data; CPU."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from cf_unc.data.pipeline import load_events
from cf_unc.data.splits import split_by_folds
from cf_unc.data.synthetic import generate
from cf_unc.experiments.e1 import load_config as load_e1_config
from cf_unc.experiments.e1 import run_e1
from cf_unc.experiments.e2 import load_config, run_e2, select_events
from cf_unc.models.base import MODEL_REGISTRY, CFModel, register
from cf_unc.rollout import tables
from cf_unc.rollout.calibration import STREAM_VARIANTS, h34_verdict

REPO = Path(__file__).resolve().parents[1]
N_FOLDS = 5
MODELS = ["idm_white_noise", "lstm_const_sigma"]
VARIANT = {"idm_traj": {"model": "idm_white_noise", "sampler_mode": "per_trajectory"}}
LABELS = [*MODELS, "idm_traj"]
SEEDS = [0, 1]
DATASETS = {
    "highd_val": {"data": "hd", "split": "val", "n_events": 14},
    "highd": {"data": "hd", "split": "test", "n_events": 14},
    "ngsim": {"data": "ng", "split": "all", "n_events": 10},
}
QUICK = {"maxiter": 3, "popsize": 5, "polish": False}  # tiny differential-evolution budget of the E1 calibration


def _load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"{name}_script", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _write_set(root: Path, name: str, seed: int, n_drivers: int, per: int, duration: float) -> dict:
    events, _ = generate(dict(n_drivers=n_drivers, events_per_driver=per, duration_s=duration, seed=seed, dataset=name))
    events.to_parquet(root / name)
    folds = {ev.event_id: int(ev.follower_id[3:]) % N_FOLDS for ev in events}  # driver-disjoint folds
    split = root / f"{name}_driver.json"
    split.write_text(json.dumps({"dataset": name, "n_folds": N_FOLDS, "folds": folds}), encoding="utf-8")
    return {"name": name, "kind": "parquet", "path": str(root / name), "unify_acceleration": True,
            "split": {"file": str(split), "fold": 0}}  # fmt: skip


@pytest.fixture(scope="module")
def e1_root(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    """Event sets and a tiny E1 run: checkpoints of two models (seeds 0, 1) and their prediction tables."""
    base = tmp_path_factory.mktemp("e2_data")
    data = {"hd": _write_set(base, "hd", 1, 40, 2, 8.0), "ng": _write_set(base, "ng", 2, 6, 2, 9.0)}
    cfg = load_e1_config()
    cfg.root = str(base / "e1")
    cfg.device = "cpu"
    cfg.models = list(MODELS)
    cfg.seeds = list(SEEDS)
    cfg.train = {"data": "hd", "stride": 2, "seq_stride": 5, "overrides": {"max_epochs": 1, "batch_size": 256}}
    cfg.eval.datasets = OmegaConf.create({"highd": {"data": "hd", "split": "test", "stride": 3},
                                          "highd_val": {"data": "hd", "split": "val", "stride": 3},
                                          "ngsim": {"data": "ng", "split": "all", "stride": 3}})  # fmt: skip
    cfg.data_configs = data
    cfg.idm = {"closed_loop": {"objective": "spacing", "max_events": 10, "seed": 0, **QUICK}}
    cfg.idm_reference = None
    status = run_e1(cfg, echo=False)
    assert status["failures"] == []
    return base / "e1", data


def make_config(root: Path, e1: Path, data: dict, models=MODELS, seeds=SEEDS, variants=VARIANT) -> DictConfig:
    cfg = load_config()
    cfg.root = str(root)
    cfg.device = "cpu"
    cfg.targets = OmegaConf.create({"a": {"checkpoints": str(e1 / "checkpoints"), "hold_steps": 1},
                                    "mean_1s": {"checkpoints": str(root.parent / "missing"), "hold_steps": 10}})  # fmt: skip
    cfg.models = list(models)
    cfg.seeds = list(seeds)
    cfg.variants = OmegaConf.create(dict(variants))
    cfg.datasets = OmegaConf.create(DATASETS)
    cfg.rollout = OmegaConf.create({"n_rollouts": 12, "n_rollouts_sampled": 4, "horizon_s": 2.0, "horizons_s": [1, 2],
                                    "max_batch": 100, "stream_events": 3, "event_seed": 0, "seed": 0})  # fmt: skip
    cfg.calibration.n_min = 4
    cfg.bootstrap.n_boot = 200
    cfg.data_configs = data
    return cfg


@pytest.fixture(scope="module")
def first_run(tmp_path_factory: pytest.TempPathFactory, e1_root: tuple[Path, dict]) -> tuple[DictConfig, dict]:
    e1, data = e1_root
    cfg = make_config(tmp_path_factory.mktemp("e2") / "run", e1, data)
    return cfg, run_e2(cfg, echo=False)


def _files(root: Path) -> dict[str, int]:
    skip = {"status.json", "run.log", "errors.log", "timings.jsonl", "config.yaml"}
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file() and p.name not in skip}


# ------------------------------------------------------------------------------------------ tests
def test_pilot_config() -> None:
    cfg = load_config(pilot=True)
    assert cfg.root == "artifacts/e2_pilot" and list(cfg.seeds) == [0] and list(cfg.targets) == ["a"]
    assert {int(d.n_events) for d in cfg.datasets.values()} == {100} and len(cfg.models) == 11  # + idm_ar_noise (R10)
    assert load_config(pilot=True, overrides=["rollout.max_batch=5000"]).rollout.max_batch == 5000


def test_select_events() -> None:
    lengths = np.array([200, 100, 200, 200, 130, 129, 200])  # t0 + K = 129 must be a valid index: >= 130 samples
    drawn = select_events(lengths, 3, seed=0, n_steps=100)
    assert len(drawn) == 3 and len(set(drawn)) == 3 and not set(drawn) & {1, 5}
    np.testing.assert_array_equal(drawn, select_events(lengths, 3, seed=0, n_steps=100))
    assert sorted(select_events(lengths, None, seed=1, n_steps=100)) == [0, 2, 3, 4, 6]


def test_first_run_complete(first_run: tuple[DictConfig, dict]) -> None:
    cfg, status = first_run
    root = Path(cfg.root)
    assert status["finished"] and status["failures"] == []
    n = len(LABELS) * len(SEEDS) * len(DATASETS)
    assert status["targets"] == {"a": {"done": n, "total": n}} and status["written_now"] == n
    assert "mean_1s" in status["skipped_targets"]  # its checkpoint directory does not exist: skipped, no failure
    log = (root / "logs" / "run.log").read_text(encoding="utf-8")
    assert "warning: target mean_1s skipped" in log and not (root / "logs" / "errors.log").exists()
    assert not (root / "mean_1s").exists() and (root / "config.yaml").exists()
    timings = [json.loads(line) for line in (root / "logs" / "timings.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(timings) == n and {t["model"] for t in timings} == set(LABELS)
    assert {t["sampler_mode"] for t in timings if t["model"] == "idm_traj"} == {"per_trajectory"}
    spec = json.loads((root / "a" / "rollouts" / "idm_traj" / "spec.json").read_text(encoding="utf-8"))
    assert spec["model"] == "idm_white_noise" and spec["sampler_mode"] == "per_trajectory"


def test_tables_columns_and_events(first_run: tuple[DictConfig, dict], e1_root: tuple[Path, dict]) -> None:
    cfg, _ = first_run
    root = Path(cfg.root)
    _, data = e1_root
    for dataset, spec in DATASETS.items():
        ev = tables.load_events_table(root, "a", dataset)
        assert list(ev.columns) == list(tables.EVENT_COLUMNS) and len(ev) == spec["n_events"]
        assert np.all(np.diff(ev["event"]) > 0) and sorted(ev["order"]) == list(range(len(ev)))
        assert (ev["t0"] == 29).all()
        events = load_events(data[spec["data"]]["path"])
        if spec["split"] != "all":
            events = split_by_folds(events, data[spec["data"]]["split"]["file"], 0)[spec["split"]]
            assert (ev["fold"] == (0 if spec["split"] == "test" else 1)).all()
        assert ev["event_id"].tolist() == [events[int(i)].event_id for i in ev["event"]]  # codes = split positions
        codes = None
        for label in LABELS:
            for seed in SEEDS:
                frame = tables.load_rollout(root, "a", label, seed, dataset)
                assert list(frame.columns) == list(tables.ROLLOUT_COLUMNS) and len(frame) == 2 * len(ev)
                assert frame["event"].dtype == np.int32 and frame["rank"].dtype == np.int32
                assert frame["horizon_s"].dtype == np.float32 and frame["collided"].dtype == np.float32
                assert (frame["n_rollouts"] == 12).all() and sorted(frame["horizon_s"].unique()) == [1.0, 2.0]
                assert np.isfinite(frame.drop(columns=["event"]).to_numpy(np.float64)).all()
                assert (frame["s_q05"] <= frame["s_q50"]).all() and (frame["s_q50"] <= frame["s_q95"]).all()
                per_h = frame[frame["horizon_s"] == 1.0]["event"].to_numpy()
                np.testing.assert_array_equal(per_h, ev["event"])  # events in the order of the events table
                codes = per_h if codes is None else codes
                np.testing.assert_array_equal(per_h, codes)  # the same events for every model and seed
        truth = tables.load_rollout(root, "a", "idm_white_noise", 0, dataset)["truth_s"].to_numpy()
        np.testing.assert_array_equal(truth, tables.load_rollout(root, "a", "lstm_const_sigma", 1, dataset)["truth_s"])
    # different seeds give different rollouts, the IDM variant shares the noise of its base model (common streams)
    a0 = tables.load_rollout(root, "a", "idm_white_noise", 0, "highd")
    a1 = tables.load_rollout(root, "a", "idm_white_noise", 1, "highd")
    assert not np.array_equal(a0["s_q50"], a1["s_q50"])
    np.testing.assert_array_equal(a0["s_q50"], tables.load_rollout(root, "a", "idm_traj", 0, "highd")["s_q50"])


def test_resume_and_regenerate(first_run: tuple[DictConfig, dict]) -> None:
    cfg, _ = first_run
    root = Path(cfg.root)
    before = _files(root)
    status = run_e2(cfg, echo=False)
    assert status["finished"] and status["failures"] == [] and status["written_now"] == 0
    assert _files(root) == before  # nothing was rewritten
    target = tables.rollout_path(root, "a", "lstm_const_sigma", 1, "ngsim")
    original = pd.read_parquet(target)
    target.unlink()
    status = run_e2(cfg, echo=False)
    assert status["written_now"] == 1
    after = _files(root)
    assert {k for k in after if after[k] != before.get(k)} == {str(target.relative_to(root))}
    pd.testing.assert_frame_equal(pd.read_parquet(target), original, check_exact=True)
    # other rollout settings than the stored ones are refused for the whole target
    changed = OmegaConf.merge(cfg, {"rollout": {"seed": 5}})
    status = run_e2(changed, echo=False)
    assert [f["unit"] for f in status["failures"]] == ["a/spec"] and _files(root) == after


def test_only_missing_target(tmp_path: Path, e1_root: tuple[Path, dict]) -> None:
    e1, data = e1_root
    cfg = make_config(tmp_path / "run", e1, data)
    cfg.targets = OmegaConf.create({"mean_1s": {"checkpoints": str(tmp_path / "nowhere"), "hold_steps": 10}})
    status = run_e2(cfg, echo=False)
    assert status["finished"] and status["failures"] == [] and status["total"] == 0
    assert list(status["skipped_targets"]) == ["mean_1s"]


def test_checkpoint_overrides_and_hold(tmp_path: Path, e1_root: tuple[Path, dict]) -> None:
    e1, data = e1_root
    empty = tmp_path / "mean1s_checkpoints"
    empty.mkdir()
    cfg = make_config(tmp_path / "run", e1, data, models=["lstm_const_sigma"], seeds=[0], variants={})
    cfg.targets = OmegaConf.create({
        "a": {"checkpoints": str(e1 / "checkpoints"), "hold_steps": 1},
        "mean_1s": {"checkpoints": str(empty), "hold_steps": 10,
                    "checkpoint_overrides": {"lstm_const_sigma": str(e1 / "checkpoints")}},
    })  # fmt: skip
    status = run_e2(cfg, echo=False)
    assert status["failures"] == [] and status["targets"]["mean_1s"] == {"done": len(DATASETS), "total": len(DATASETS)}
    root = tmp_path / "run"
    assert json.loads((root / "mean_1s" / "spec.json").read_text(encoding="utf-8"))["hold_steps"] == 10
    one = tables.load_rollout(root, "a", "lstm_const_sigma", 0, "highd")
    held = tables.load_rollout(root, "mean_1s", "lstm_const_sigma", 0, "highd")
    np.testing.assert_array_equal(one["truth_s"], held["truth_s"])  # same events
    assert not np.allclose(one["s_q95"] - one["s_q05"], held["s_q95"] - held["s_q05"])  # draws kept for 10 steps
    holds = {(t["target"], t["hold_steps"]) for t in map(json.loads, (root / "logs" / "timings.jsonl").read_text(encoding="utf-8").splitlines())}
    assert holds == {("a", 1), ("mean_1s", 10)}


class _Boom(CFModel):
    """A model whose sampler always fails."""

    def fit(self, train, val=None, *, seed=0, verbose=False):  # noqa: ANN001, ANN201
        return {}

    def predict_dist(self, state_history, *, v_lead_future=None, batch_size=8192):  # noqa: ANN001, ANN201
        raise NotImplementedError

    def sample_step(self, hist, state, generator):  # noqa: ANN001, ANN201
        raise RuntimeError("boom: injected failure")

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict) -> None:
        pass


@pytest.fixture
def boom_checkpoint(e1_root: tuple[Path, dict]):  # noqa: ANN201
    register("e2_test_boom")(_Boom)
    e1, _ = e1_root
    path = e1 / "checkpoints" / "boom" / "seed0" / "model.pt"
    _Boom().save(path)
    yield "boom"
    MODEL_REGISTRY.pop("e2_test_boom", None)
    path.unlink()


def test_failing_model_is_logged(tmp_path: Path, e1_root: tuple[Path, dict], boom_checkpoint: str) -> None:
    e1, data = e1_root
    cfg = make_config(tmp_path / "run", e1, data, models=["boom", "idm_white_noise"], seeds=[0], variants={})
    config_file = tmp_path / "e2_test.yaml"
    OmegaConf.save(cfg, config_file)
    assert _load_script("run_e2").main(["--config", str(config_file), "--quiet"]) == 1
    root = tmp_path / "run"
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["finished"] and sorted(f["unit"] for f in status["failures"]) == [f"a/boom/seed0/{d}" for d in sorted(DATASETS)]
    errors = (root / "logs" / "errors.log").read_text(encoding="utf-8")
    assert "Traceback" in errors and "RuntimeError: boom: injected failure" in errors
    for dataset in DATASETS:  # the model after the failing one ran completely
        assert tables.rollout_path(root, "a", "idm_white_noise", 0, dataset).exists()
        assert not tables.rollout_path(root, "a", "boom", 0, dataset).exists()
    # a missing checkpoint of a configured model is a failure of that unit, the rest runs
    cfg2 = make_config(tmp_path / "run2", e1, data, models=["hetero_mdn", "idm_white_noise"], seeds=[0], variants={})
    status = run_e2(cfg2, echo=False)
    assert [f["unit"] for f in status["failures"]] == ["a/hetero_mdn/seed0"]
    assert "does not exist" in status["failures"][0]["error"]
    assert status["targets"]["a"]["done"] == len(DATASETS)


def test_analyze_e2_end_to_end(first_run: tuple[DictConfig, dict], e1_root: tuple[Path, dict]) -> None:
    cfg, _ = first_run
    root = Path(cfg.root)
    e1, _ = e1_root
    analysis = _load_script("analyze_e2")
    out = analysis.analyze_e2(root, {"a": e1, "mean_1s": root / "none"}, verbose=False)
    for name in ("coverage_by_horizon", "aci_gamma_sensitivity", "onestep_conformal", "collisions", "h34_verdicts",
                 "stream_info"):
        frame = pd.read_csv(root / "tables" / f"{name}.csv")
        assert len(frame) > 0, name
        if name in analysis.COLUMNS:
            assert list(frame.columns) == analysis.COLUMNS[name], name
    assert (root / "tables" / "analysis_info.json").exists()
    cov = out["coverage_by_horizon"]
    assert set(cov["model"]) == set(LABELS) and set(cov["dataset"]) == set(DATASETS)
    assert set(cov[cov["dataset"] == "highd"]["variant"]) == {"raw", "raw_linear", "static", "aci_frozen", "aci_online",
                                                               *STREAM_VARIANTS}  # fmt: skip
    assert set(cov[cov["dataset"] == "highd_val"]["variant"]) == {"raw", "raw_linear"}  # the calibration set itself
    assert ((cov["picp"] >= 0) & (cov["picp"] <= 1)).all() and (cov["n_seeds"] == 2).all()
    assert (cov[cov["variant"].str.startswith("aci")]["gamma"] == 0.01).all()
    online = cov[cov["variant"].isin(["aci_online", "aci_online_chrono", "static_prequential"])]
    assert (online["delay_events_mean"] == 0).all() and online["picp_first50"].notna().all()
    assert cov.loc[cov["variant"] == "raw", "picp_first50"].isna().all()
    delayed = cov[cov["variant"] == "aci_online_delayed"]  # synthetic events: untimed, 1 event per second of horizon
    assert (delayed["delay_events_mean"] > 0).all()
    assert ((cov["share_infinite"] >= 0) & (cov["share_infinite"] <= 1)).all() and (cov["share_empty"] >= 0).all()
    assert (cov["picp_lo_follower"] <= cov["picp"] + 1e-12).all() and cov["n_followers"].notna().all()
    streams = out["stream_info"]
    assert set(streams["dataset"]) == {"highd", "ngsim"} and set(streams["clock"]) == {"untimed"}
    sens = out["aci_gamma_sensitivity"]
    assert sorted(sens["gamma"].unique()) == [0.005, 0.01, 0.02, 0.05]
    assert set(sens["variant"]) == {"aci_frozen", "aci_online"}
    one = out["onestep_conformal"]
    assert set(one["model"]) == set(MODELS) and set(one["dataset"]) == {"highd", "ngsim"} and set(one["score"]) == {"absolute", "normalised"}
    assert ((one["picp_conformal_lo"] <= one["picp_conformal"]) & (one["picp_conformal"] <= one["picp_conformal_hi"])).all()
    assert one["picp_conformal_lo_follower"].notna().all() and (one["n_followers"] > 0).all()
    col = out["collisions"]
    assert {"collided_1s", "collided_2s", "events_with_collision"} <= set(col.columns) and len(col) == len(LABELS) * len(DATASETS)
    ver = out["h34_verdicts"]
    assert set(ver["verdict"]) <= {"supported", "refuted", "inconclusive"} and set(ver["dataset"]) == {"highd", "ngsim"}
    assert set(ver["corrected"]) == {"static", "aci_frozen", "aci_online"} and ver["onestep_picp"].notna().all()
    assert set(ver["onestep_source"]) == {"highd"}
    # the table rows follow the decision table of h34_verdict (R7)
    for row in ver.itertuples():
        raw = cov[(cov["model"] == row.model) & (cov["target"] == row.target) & (cov["dataset"] == row.dataset)
                  & (cov["variant"] == "raw")].sort_values("horizon_s")  # fmt: skip
        corr = cov[(cov["model"] == row.model) & (cov["target"] == row.target) & (cov["dataset"] == row.dataset)
                   & (cov["variant"] == row.corrected)].sort_values("horizon_s")  # fmt: skip
        again = h34_verdict(row.onestep_picp, raw["picp"], corr["picp"], onestep_ci=(row.onestep_lo, row.onestep_hi),
                            raw_ci=(raw["picp_lo"], raw["picp_hi"]), corrected_ci=(corr["picp_lo"], corr["picp_hi"]))  # fmt: skip
        assert (again["verdict"], again["verdict_ci"]) == (row.verdict, row.verdict_ci)
    figures = {p.name for p in (root / "figures").glob("*.png")}
    assert figures == {"coverage_by_horizon__a.png", "coverage_by_horizon__a__ngsim.png", "width_by_horizon__a.png"}
    assert all(p.stat().st_size > 5000 for p in (root / "figures").glob("*.png"))
    text = analysis.format_summary(out)
    assert "H3.4 verdicts" in text and "lstm_const_sigma" in text
    # the heuristic finite ACI of the submitted tables is still available, in its own directory
    finite = analysis.analyze_e2(root, {"a": e1, "mean_1s": root / "none"}, verbose=False, figures=False,
                                 settings={"calibration": {"finite_quantile": True, "streams": False}},
                                 tables_dir="tables_finite")  # fmt: skip
    assert (root / "tables_finite" / "coverage_by_horizon.csv").exists()
    f_cov = finite["coverage_by_horizon"]
    assert set(f_cov["variant"]) == {"raw", "raw_linear", "static", "aci_frozen", "aci_online"}
    assert (f_cov["share_infinite"] == 0).all() and (f_cov["share_empty"] == 0).all()  # every correction finite
    same = ["model", "target", "dataset", "horizon_s", "variant"]
    merged = cov.merge(f_cov, on=same, suffixes=("", "_finite"))
    fixed = merged[merged["variant"].isin(["raw", "raw_linear", "static"])]
    assert len(fixed) and np.allclose(fixed["picp"], fixed["picp_finite"])  # only ACI changes


# ------------------------------------------------------------------------------ signal variant
def test_harmonised_config_refuses_raw_checkpoints(tmp_path: Path, e1_root: tuple[Path, dict]) -> None:
    from cf_unc.data.harmonise import HarmoniseConfig

    e1, data = e1_root
    root = tmp_path / "harm"
    cfg = make_config(root, e1, data, models=["lstm_const_sigma"], seeds=[0], variants={})
    cfg.signal = "harmonised"
    status = run_e2(cfg, echo=False)
    assert status["signal"] == "harmonised" and status["finished"]
    failed = {f["unit"]: f["error"] for f in status["failures"]}
    assert set(failed) == {"a/lstm_const_sigma/seed0"}  # refused before any rollout
    assert "trained on raw signals, the config asks harmonised" in failed["a/lstm_const_sigma/seed0"]
    spec = json.loads((root / "a" / "spec.json").read_text(encoding="utf-8"))
    assert spec["signal"] == {"signal": "harmonised", "harmonise": HarmoniseConfig().to_dict()}
    assert not list((root / "a" / "rollouts").rglob("*.parquet"))
    raw = make_config(root, e1, data, models=["lstm_const_sigma"], seeds=[0], variants={})
    status = run_e2(raw, echo=False)  # the same root with raw signals: the stored target spec refuses it
    assert [f["unit"] for f in status["failures"]] == ["a/spec"] and status["signal"] == "raw"


def test_e2h_config_differs_from_e2_only_where_intended() -> None:
    from cf_unc.data.harmonise import HarmoniseConfig

    exp = REPO / "configs" / "experiment"
    e2 = OmegaConf.to_container(load_config(exp / "e2.yaml"))
    e2h = OmegaConf.to_container(load_config(exp / "e2h.yaml"))
    assert e2h["root"] == "artifacts/e2h" and e2h["signal"] == "harmonised" and e2h["harmonise"] == HarmoniseConfig().to_dict()
    assert e2h["targets"] == {
        "a": {"checkpoints": "artifacts/e1h/checkpoints", "hold_steps": 1},
        "mean_1s": {"checkpoints": "artifacts/e1h_mean1s/checkpoints", "hold_steps": 10,
                    "checkpoint_overrides": {"diffusion_cf": "artifacts/e1h/checkpoints"}},
    }  # fmt: skip
    assert e2h["e1_roots"] == {"a": "artifacts/e1h", "mean_1s": "artifacts/e1h_mean1s"}
    for cfg in (e2, e2h):
        for key in ("root", "signal", "harmonise", "targets", "e1_roots"):
            cfg.pop(key, None)
    assert e2 == e2h
    pilot = load_config(exp / "e2h.yaml", pilot=True)
    assert pilot.root == "artifacts/e2h_pilot" and list(pilot.targets) == ["a"] and list(pilot.seeds) == [0]


def test_analyze_e2_takes_e1_roots_from_the_stored_config(first_run: tuple[DictConfig, dict], e1_root: tuple[Path, dict],
                                                           tmp_path: Path) -> None:  # fmt: skip
    import shutil

    import yaml

    cfg, _ = first_run
    e1, _ = e1_root
    root = tmp_path / "copy"
    shutil.copytree(Path(cfg.root) / "a", root / "a")
    stored = yaml.safe_load((Path(cfg.root) / "config.yaml").read_text(encoding="utf-8"))
    stored["e1_roots"] = {"a": str(e1), "mean_1s": str(tmp_path / "none")}
    (root / "config.yaml").write_text(yaml.safe_dump(stored), encoding="utf-8")
    analysis = _load_script("analyze_e2")
    out = analysis.analyze_e2(root, None, figures=False, verbose=False)
    info = json.loads((root / "tables" / "analysis_info.json").read_text(encoding="utf-8"))
    assert info["e1_roots"] == stored["e1_roots"] and len(out["onestep_conformal"]) > 0
