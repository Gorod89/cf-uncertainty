"""Experiment runner with the target mean_1s (design.md, section 11.1), reuse_checkpoints and config inheritance.

Tiny synthetic event sets, CPU only, a few seconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from cf_unc.constants import DT, MEAN_TARGET_STEPS
from cf_unc.data.pipeline import load_events
from cf_unc.data.schema import EventSet
from cf_unc.data.splits import split_by_folds
from cf_unc.data.synthetic import generate
from cf_unc.data.windows import make_windows
from cf_unc.experiments.e1 import (
    SPEC_DEFAULTS,
    Data,
    _same_spec,
    accepts_keyword,
    checkpoint_dir,
    config_target,
    load_config,
    model_spec,
    pilot_path,
    predict_table,
    run_e1,
    sampled_rows,
)
from cf_unc.maps import io as tables
from cf_unc.models import load_model
from cf_unc.physics.idm import IDMParams, idm_accel

REPO = Path(__file__).resolve().parents[1]
N_FOLDS = 5
DATASETS: dict[str, dict[str, Any]] = {
    "highd": {"data": "hd", "split": "test", "stride": 2},
    "ngsim": {"data": "ng", "split": "all", "stride": 3},
}
QUICK = {"maxiter": 3, "popsize": 5, "polish": False}  # tiny differential-evolution budget
TINY_DIFFUSION = {  # inline model config: a DiffusionCF that trains in about a second
    "name": "diffusion_cf",
    "kwargs": {"n_samples": 8, "n_sampling_steps": 10, "t_train": 100, "channels": 16, "n_blocks": 2, "hidden": 16},
    "train": {},
    "needs_idm": False,
    "needs_sequences": True,
}


def _write_set(root: Path, name: str, seed: int, n_drivers: int, per: int, duration: float) -> dict:
    events, _ = generate(dict(n_drivers=n_drivers, events_per_driver=per, duration_s=duration, seed=seed, dataset=name))
    events.to_parquet(root / name)
    folds = {ev.event_id: int(ev.follower_id[3:]) % N_FOLDS for ev in events}  # driver-disjoint folds
    split = root / f"{name}_driver.json"
    split.write_text(json.dumps({"dataset": name, "n_folds": N_FOLDS, "folds": folds}), encoding="utf-8")
    return {"name": name, "kind": "parquet", "path": str(root / name), "unify_acceleration": True,
            "split": {"file": str(split), "fold": 0}}  # fmt: skip


@pytest.fixture(scope="module")
def data_configs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    root = tmp_path_factory.mktemp("target_events")
    return {
        "hd": _write_set(root, "hd", seed=1, n_drivers=25, per=2, duration=12.0),
        "ng": _write_set(root, "ng", seed=2, n_drivers=8, per=2, duration=15.0),
    }


def make_config(
    root: Path, data_configs: dict, models: list[str], target: str = "mean_1s", datasets: dict = DATASETS, **extra: Any
) -> DictConfig:
    cfg = load_config()
    cfg.root = str(root)
    cfg.device = "cpu"
    cfg.models = list(models)
    cfg.seeds = [0]
    cfg.train = {"data": "hd", "target": target, "stride": 2, "seq_stride": 5,
                 "overrides": {"max_epochs": 1, "batch_size": 256}}  # fmt: skip
    cfg.eval.datasets = OmegaConf.create(datasets)  # replaces the data sets of e1.yaml
    cfg.eval.chunk_rows = 150
    cfg.data_configs = data_configs
    cfg.model_configs = {"dcf": TINY_DIFFUSION}
    cfg.idm = {"closed_loop": {"objective": "spacing", "max_events": 10, "seed": 0, **QUICK}}
    cfg.idm_reference = None
    for key, value in extra.items():
        cfg[key] = value
    return cfg


def _files(root: Path) -> dict[str, int]:
    """mtime (ns) of every output file except the logs and the status."""
    skip = {"status.json", "run.log", "errors.log"}
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file() and p.name not in skip}


def _eval_events(data_configs: dict, spec: dict) -> EventSet:
    events = load_events(data_configs[spec["data"]]["path"], unify=True)
    if spec["split"] != "all":
        events = split_by_folds(events, data_configs[spec["data"]]["split"]["file"], 0)[spec["split"]]
    return events


@pytest.fixture(scope="module")
def mean_run(tmp_path_factory: pytest.TempPathFactory, data_configs: dict) -> tuple[DictConfig, dict]:
    cfg = make_config(tmp_path_factory.mktemp("mean_1s") / "run", data_configs, ["idm_white_noise", "lstm_const_sigma"])
    return cfg, run_e1(cfg, echo=False)


# ------------------------------------------------------------------------------ target mean_1s
def test_windows_tables_hold_the_mean_1s_target(mean_run: tuple[DictConfig, dict], data_configs: dict) -> None:
    cfg, status = mean_run
    root = Path(cfg.root)
    assert status["finished"] and status["failures"] == [] and status["target"] == "mean_1s"
    assert status["tables"]["done"] == status["tables"]["total"] == 2 * len(DATASETS)
    for dataset, spec in DATASETS.items():
        frame, meta = tables.load_windows(root, dataset)
        assert list(frame.columns) == list(tables.WINDOW_COLUMNS)
        assert meta["target"] == meta["spec"]["target"] == "mean_1s" and tables.windows_target(meta) == "mean_1s"
        events = _eval_events(data_configs, spec)
        expected = make_windows(events, stride=spec["stride"], target="mean_1s")
        one_step = make_windows(events, stride=spec["stride"])
        assert len(frame) == len(expected) == meta["n_windows"] < len(one_step)  # the next second must lie inside
        evs = [events[int(e)] for e in frame["event"]]
        t = frame["t_index"].to_numpy()
        assert all(k + MEAN_TARGET_STEPS < len(ev) for ev, k in zip(evs, t))
        k = MEAN_TARGET_STEPS
        y = np.array([(ev.v[j + k] - ev.v[j]) / (k * DT) for ev, j in zip(evs, t)], dtype=np.float32)
        y_prev = np.array([(ev.v[j] - ev.v[j - k]) / (k * DT) for ev, j in zip(evs, t)], dtype=np.float32)
        a_prev = np.array([ev.a[j - 1] for ev, j in zip(evs, t)], dtype=np.float32)
        np.testing.assert_allclose(frame["y"], y, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(frame["y_prev"], y_prev, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(frame["a_prev"], a_prev)  # regimes keep a[t-1] whatever the target
        np.testing.assert_array_equal(frame["y"], expected.y)
        for model in ("idm_white_noise", "lstm_const_sigma"):
            pred = tables.load_preds(root, model, 0, dataset)
            assert len(pred) == len(frame) and np.isfinite(pred[["mean", "var", "nll"]].to_numpy()).all()
        metrics = json.loads((checkpoint_dir(root, "lstm_const_sigma", 0) / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["datasets"][dataset]["n_rows"] == len(frame)


def test_models_are_trained_on_the_mean_1s_target(mean_run: tuple[DictConfig, dict]) -> None:
    cfg, _ = mean_run
    root = Path(cfg.root)
    train = Data(cfg).bundle(sequences=False).windows["train"]
    one_step = Data(OmegaConf.merge(cfg, {"train": {"target": "a"}})).bundle(sequences=False).windows["train"]
    assert len(train) < len(one_step)
    params = IDMParams.from_dict(json.loads((root / "idm.json").read_text(encoding="utf-8"))["params"])
    last = train.X[:, -1, :].astype(np.float64)
    sigma = np.sqrt(np.mean((train.y - idm_accel(last[:, 0], last[:, 1], last[:, 2], params)) ** 2))
    frame, _ = tables.load_windows(root, "highd")
    pred = tables.load_preds(root, "idm_white_noise", 0, "highd")
    s, dv, v = (frame[c].to_numpy(np.float64) for c in ("s", "dv", "v"))
    # the instantaneous IDM acceleration is the mean of both targets, the noise level is fitted per target
    np.testing.assert_array_equal(pred["mean"], idm_accel(s, dv, v, params).astype(np.float32))
    np.testing.assert_allclose(np.sqrt(pred["var"].to_numpy(np.float64)), sigma, rtol=1e-5)
    for model in ("idm_white_noise", "lstm_const_sigma"):
        log = json.loads((checkpoint_dir(root, model, 0) / "train_log.json").read_text(encoding="utf-8"))
        assert log["target"] == "mean_1s" and log["n_train"] == len(train)
    with np.load(root / "train_density.npz") as z:
        assert str(z["target"]) == "mean_1s" and int(z["stride"]) == 1
        hd = load_events(cfg.data_configs.hd.path)
        density = make_windows(split_by_folds(hd, cfg.data_configs.hd.split.file, 0)["train"], target="mean_1s")
        assert int(z["n_total"]) == len(density)


def test_resume_and_changed_target_is_a_mismatch(mean_run: tuple[DictConfig, dict]) -> None:
    cfg, _ = mean_run
    root = Path(cfg.root)
    before = _files(root)
    status = run_e1(cfg, echo=False)
    assert status["finished"] and status["failures"] == [] and _files(root) == before
    assert status["tables"]["written_now"] == 0 and status["checkpoints"]["trained_now"] == 0

    other = OmegaConf.merge(cfg, {"train": {"target": "a"}})  # same root, other target: refused, nothing rewritten
    status = run_e1(other, echo=False)
    failed = {f["unit"]: f["error"] for f in status["failures"]}
    assert set(failed) == {"windows/highd", "windows/ngsim", "train_density", "idm_white_noise/seed0",
                           "lstm_const_sigma/seed0"}  # fmt: skip
    assert "'target': 'mean_1s'" in failed["windows/highd"] and "the config asks 1 and 'a'" in failed["train_density"]
    assert "trained for the target 'mean_1s', the config asks 'a'" in failed["lstm_const_sigma/seed0"]
    assert status["tables"]["written_now"] == 0 and _files(root) == before


def test_spec_comparison_and_config_target(data_configs: dict, tmp_path: Path) -> None:
    old = {"data": "highd", "split": "test", "stride": 2}  # spec stored before the target option
    assert SPEC_DEFAULTS == {"target": "a"}
    assert _same_spec(old, dict(old)) and _same_spec(old, {**old, "target": "a"})
    assert not _same_spec(old, {**old, "target": "mean_1s"}) and not _same_spec(None, old)
    assert _same_spec({**old, "target": "mean_1s"}, {**old, "target": "mean_1s"})
    cfg = make_config(tmp_path, data_configs, ["dcf", "lstm_const_sigma"], target="mean_1s")
    assert config_target(cfg) == "mean_1s" and config_target(load_config()) == "a"
    assert Data(cfg).eval_spec("highd")["target"] == "mean_1s"
    assert "target" not in Data(load_config()).eval_spec("highd")  # specs of the default target are unchanged
    assert model_spec(cfg, "lstm_const_sigma").target == "mean_1s"
    assert model_spec(cfg, "dcf").target is None  # sequence model: the checkpoint serves every target
    seq = Data(cfg).bundle(sequences=True).windows["test"]
    np.testing.assert_array_equal(seq.y, seq.y_seq[:, 0])  # sequence windows keep y = a[t]
    with pytest.raises(ValueError, match="target"):
        config_target(OmegaConf.create({"train": {"target": "jerk"}}))


# ------------------------------------------------------------------------------ reuse_checkpoints
@pytest.fixture(scope="module")
def source_run(tmp_path_factory: pytest.TempPathFactory, data_configs: dict) -> DictConfig:
    """Experiment root with target a: a tiny DiffusionCF and an LSTM (the checkpoints to reuse)."""
    cfg = make_config(tmp_path_factory.mktemp("source") / "e1", data_configs, ["dcf", "lstm_const_sigma"], target="a",
                      datasets={"highd": DATASETS["highd"]})  # fmt: skip
    status = run_e1(cfg, echo=False)
    assert status["failures"] == [] and status["checkpoints"]["trained_now"] == 2 and status["tables"]["done"] == 2
    return cfg


def test_reuse_checkpoint_of_diffusion_cf(source_run: DictConfig, data_configs: dict, tmp_path: Path) -> None:
    source = Path(source_run.root) / "checkpoints"
    cfg = make_config(tmp_path / "mean", data_configs, ["dcf"], reuse_checkpoints={"dcf": str(source)})
    status = run_e1(cfg, echo=False)
    root = Path(cfg.root)
    assert status["failures"] == [] and status["trained"] == ["dcf/seed0 (reused)"]
    copy, original = checkpoint_dir(root, "dcf", 0), checkpoint_dir(source_run.root, "dcf", 0)
    assert (copy / "model.pt").read_bytes() == (original / "model.pt").read_bytes()
    log = json.loads((copy / "train_log.json").read_text(encoding="utf-8"))
    stored = json.loads((original / "train_log.json").read_text(encoding="utf-8"))
    assert Path(log["reused_from"]).resolve() == original.resolve() and "reuse_note" in log
    assert log["kwargs"] == stored["kwargs"] and log["target"] is None and log["seconds"] == stored["seconds"]
    assert "checkpoint reused from" in (root / "logs" / "run.log").read_text(encoding="utf-8")
    # the law of mean_1s: mean of the first 10 steps of the sampled sequences of the reused checkpoint
    for dataset, spec in DATASETS.items():
        frame, _ = tables.load_windows(root, dataset)
        pred = tables.load_preds(root, "dcf", 0, dataset)
        ev = Data(cfg).eval_set(dataset)
        rows = sampled_rows(ev.windows.event_index, ev.windows.t_index, ev.lengths, 10_000, 0)
        assert rows.size > 0 and np.isfinite(pred["mean"].to_numpy()[rows]).all()
        assert np.isnan(np.delete(pred["mean"].to_numpy(), rows)).all()
        model = load_model(copy / "model.pt")
        chunks = [rows[a : a + int(cfg.eval.chunk_rows)] for a in range(0, rows.size, int(cfg.eval.chunk_rows))]
        seq = np.concatenate([model.sample_sequences(ev.windows.X[c], ev.leader_future(c)) for c in chunks])  # noise per call
        mean_1s = seq[:, :, :MEAN_TARGET_STEPS].astype(np.float64).mean(axis=2).mean(axis=1)
        np.testing.assert_allclose(pred["mean"].to_numpy()[rows], mean_1s, rtol=1e-5, atol=1e-6)
        assert not np.allclose(pred["mean"].to_numpy()[rows], seq[:, :, 0].mean(axis=1), atol=1e-3)
        np.testing.assert_array_equal(frame["y"], ev.windows.y)
    before = _files(root)
    status = run_e1(cfg, echo=False)  # resume: the copy is checked like an own checkpoint, nothing is rewritten
    assert status["failures"] == [] and status["trained"] == [] and _files(root) == before


def test_reuse_refuses_other_target_and_missing_source(source_run: DictConfig, data_configs: dict, tmp_path: Path) -> None:
    source = Path(source_run.root) / "checkpoints"
    cfg = make_config(tmp_path / "lstm", data_configs, ["lstm_const_sigma"], reuse_checkpoints={"lstm_const_sigma": str(source)})
    status = run_e1(cfg, echo=False)  # a one-step model trained for target a is not silently reused for mean_1s
    assert [f["unit"] for f in status["failures"]] == ["lstm_const_sigma/seed0"]
    assert "trained for the target 'a', the config asks 'mean_1s'" in status["failures"][0]["error"]
    assert not (checkpoint_dir(cfg.root, "lstm_const_sigma", 0) / "model.pt").exists()
    assert not tables.preds_path(cfg.root, "lstm_const_sigma", 0, "highd").exists()

    missing = make_config(tmp_path / "missing", data_configs, ["dcf"], reuse_checkpoints={"dcf": str(tmp_path / "none" / "checkpoints")})
    status = run_e1(missing, echo=False)  # no source: the unit fails, it is not trained instead
    assert [f["unit"] for f in status["failures"]] == ["dcf/seed0"] and "FileNotFoundError" in status["failures"][0]["error"]
    assert status["trained"] == []

    kwargs = make_config(tmp_path / "kwargs", data_configs, ["dcf"], reuse_checkpoints={"dcf": str(source)})
    kwargs.model_configs.dcf.kwargs.n_samples = 16  # other constructor arguments than the source checkpoint
    status = run_e1(kwargs, echo=False)
    assert [f["unit"] for f in status["failures"]] == ["dcf/seed0"] and "other kwargs" in status["failures"][0]["error"]


# ------------------------------------------------------------------------ predict_table, keywords
class _Called(Exception):
    """Raised by the stand-in predict_dist functions once their keywords are recorded."""


def test_predict_table_passes_the_target_only_when_accepted(source_run: DictConfig) -> None:
    calls: list[dict] = []

    def with_target(state_history, *, v_lead_future=None, batch_size=8192, target="a"):  # noqa: ANN001, ANN202
        calls.append({"target": target})
        raise _Called

    def without(state_history, *, v_lead_future=None, batch_size=8192, **kwargs):  # noqa: ANN001, ANN202
        calls.append(dict(kwargs))
        raise _Called

    assert accepts_keyword(with_target, "target") and not accepts_keyword(without, "target")  # **kwargs do not count
    ev = Data(source_run).eval_set("highd")
    for fn in (with_target, without):
        model = type("StandIn", (), {"needs_leader_future": False, "predict_dist": staticmethod(fn)})()
        with pytest.raises(_Called):
            predict_table(model, ev, np.arange(5), seed=0, target="mean_1s")  # type: ignore[arg-type]
        with pytest.raises(_Called):
            predict_table(model, ev, np.arange(5), seed=0)  # type: ignore[arg-type]
    assert calls == [{"target": "mean_1s"}, {"target": "a"}, {}, {}]


# ------------------------------------------------------------------------------ config inheritance
def test_base_inheritance_and_cycles(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "parent.yaml").write_text(
        "root: artifacts/p\nseeds: [0, 1, 2]\ntrain: {data: hd, target: a, stride: 2}\n"
        "model_overrides: {q: {w: 1.0, keep: true}}\n", encoding="utf-8",
    )  # fmt: skip
    (tmp_path / "sub" / "child.yaml").write_text(
        "base: ../parent.yaml\nroot: artifacts/c\nseeds: [4]\ntrain: {target: mean_1s}\nmodel_overrides: {q: {w: 2.0}}\n",
        encoding="utf-8",
    )  # fmt: skip
    cfg = load_config(tmp_path / "sub" / "child.yaml")
    assert "base" not in cfg and cfg.root == "artifacts/c" and list(cfg.seeds) == [4]  # lists are replaced
    assert OmegaConf.to_container(cfg.train) == {"data": "hd", "target": "mean_1s", "stride": 2}
    assert OmegaConf.to_container(cfg.model_overrides) == {"q": {"w": 2.0, "keep": True}}  # mappings are merged
    assert load_config(tmp_path / "sub" / "child.yaml", overrides=["seeds=[1]"]).seeds == [1]
    (tmp_path / "a.yaml").write_text("base: b.yaml\nroot: x\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("base: a.yaml\nroot: y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "a.yaml")


def test_e1_mean1s_config_differs_from_e1_only_where_intended() -> None:
    e1 = OmegaConf.to_container(load_config(REPO / "configs" / "experiment" / "e1.yaml"))
    mean = OmegaConf.to_container(load_config(REPO / "configs" / "experiment" / "e1_mean1s.yaml"))
    assert mean["root"] == "artifacts/e1_mean1s" and mean["train"]["target"] == "mean_1s" and e1["train"]["target"] == "a"
    assert mean["reuse_checkpoints"] == {"diffusion_cf": "artifacts/e1/checkpoints"} and e1["reuse_checkpoints"] == {}
    assert set(mean["model_overrides"]) == {"quantile_pidl"} and set(mean["model_overrides"]["quantile_pidl"]) == {"w_phys"}
    for cfg in (e1, mean):
        cfg.pop("root"), cfg["train"].pop("target"), cfg.pop("reuse_checkpoints"), cfg.pop("model_overrides")
    assert mean == e1
    assert list(e1["eval"]["datasets"]) == ["highd", "highd_val", "ngsim", "ngsim_us101", "ngsim_fn", "waymo", "openacc"]
    assert e1["eval"]["datasets"]["ngsim_us101"] == {"data": "ngsim_us101", "split": "all", "stride": 8}


def test_pilot_roots() -> None:
    assert pilot_path("artifacts/e1") == "artifacts/e1_pilot"
    assert pilot_path("artifacts/e1/checkpoints") == "artifacts/e1_pilot/checkpoints"
    assert pilot_path("artifacts\\e1_mean1s") == "artifacts/e1_mean1s_pilot"
    assert load_config(pilot=True).root == "artifacts/e1_pilot"  # e1.yaml: as before the option
    pilot = load_config(REPO / "configs" / "experiment" / "e1_mean1s.yaml", pilot=True)
    assert pilot.root == "artifacts/e1_mean1s_pilot" and list(pilot.seeds) == [0] and pilot.eval.max_events == 400
    assert OmegaConf.to_container(pilot.reuse_checkpoints) == {"diffusion_cf": "artifacts/e1_pilot/checkpoints"}
    assert load_config(REPO / "configs" / "experiment" / "e1_mean1s.yaml", ["root=elsewhere"], pilot=True).root == "elsewhere"
