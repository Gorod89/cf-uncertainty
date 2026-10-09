"""Experiment E1 runner (cf_unc.experiments.e1) on three tiny synthetic event sets; CPU only, a few seconds."""

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
from cf_unc.data.schema import EventSet
from cf_unc.data.splits import split_by_folds
from cf_unc.data.synthetic import generate
from cf_unc.data.windows import make_windows
from cf_unc.experiments.e1 import bin_counts, checkpoint_dir, grid_edges, load_config, run_e1, sampled_rows
from cf_unc.maps import io as tables
from cf_unc.models import load_model
from cf_unc.models.base import MODEL_REGISTRY, CFModel, register
from cf_unc.physics.idm import IDMParams, idm_accel

REPO = Path(__file__).resolve().parents[1]
N_FOLDS = 5
MODELS = ["idm_white_noise", "lstm_const_sigma", "hetero_mdn"]
SEEDS = [0, 1]
DATASETS: dict[str, dict[str, Any]] = {
    "highd": {"data": "hd", "split": "test", "stride": 2},
    "highd_val": {"data": "hd", "split": "val", "stride": 2},
    "ngsim": {"data": "ng", "split": "all", "stride": 3},
    "openacc": {"data": "oa", "split": "all", "stride": 5, "max_events": 10},
}
WINDOW_DTYPES = {c: np.int32 for c in ("event_index", "t_index", "event", "driver")} | {"fold": np.int8}
WINDOW_DTYPES |= {c: np.float32 for c in (*tables.WINDOW_FLOAT, *tables.WINDOW_EXTRA)}
QUICK = {"maxiter": 3, "popsize": 5, "polish": False}  # tiny differential-evolution budget
CALIBRATIONS = {
    "closed_loop": {"objective": "spacing", "max_events": 10, "seed": 0, **QUICK},
    "mle": {"objective": "accel", "max_events": 20, "seed": 0, **QUICK},
}


def _load_script(name: str) -> Any:
    """Module of ``scripts/<name>.py`` (the scripts directory is not a package)."""
    spec = importlib.util.spec_from_file_location(f"{name}_script", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ------------------------------------------------------------------------------ data and config
def _write_set(root: Path, name: str, seed: int, n_drivers: int, per: int, duration: float, edit=None) -> dict:
    events, _ = generate(dict(n_drivers=n_drivers, events_per_driver=per, duration_s=duration, seed=seed, dataset=name))
    for i, ev in enumerate(events):
        if edit is not None:
            edit(i, ev)
    events.to_parquet(root / name)
    folds = {ev.event_id: int(ev.follower_id[3:]) % N_FOLDS for ev in events}  # driver-disjoint folds
    split = root / f"{name}_driver.json"
    split.write_text(json.dumps({"dataset": name, "n_folds": N_FOLDS, "folds": folds}), encoding="utf-8")
    return {"name": name, "kind": "parquet", "path": str(root / name), "unify_acceleration": True,
            "split": {"file": str(split), "fold": 0}}  # fmt: skip


def _driver_mode(i: int, ev) -> None:
    ev.meta["driver_mode"] = "ACC" if i % 2 == 0 else "Human"


def _dataset_accel(i: int, ev) -> None:
    ev.meta["a_source"] = "dataset"  # the runner must recompute a from the speed (common definition)


@pytest.fixture(scope="module")
def data_configs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    root = tmp_path_factory.mktemp("e1_events")
    return {
        "hd": _write_set(root, "hd", seed=1, n_drivers=25, per=2, duration=12.0),
        "ng": _write_set(root, "ng", seed=2, n_drivers=8, per=2, duration=15.0, edit=_dataset_accel),
        "oa": _write_set(root, "oa", seed=3, n_drivers=6, per=3, duration=10.0, edit=_driver_mode),
    }


def make_config(root: Path, data_configs: dict, models=MODELS, seeds=SEEDS, datasets=DATASETS) -> DictConfig:
    cfg = load_config()
    cfg.root = str(root)
    cfg.device = "cpu"
    cfg.models = list(models)
    cfg.seeds = list(seeds)
    cfg.train = {"data": "hd", "stride": 2, "seq_stride": 5, "overrides": {"max_epochs": 1, "batch_size": 256}}
    cfg.eval.datasets = OmegaConf.create(datasets)
    cfg.eval.chunk_rows = 150  # several chunks per table
    cfg.data_configs = data_configs
    cfg.idm = CALIBRATIONS
    cfg.idm_reference = None
    return cfg


@pytest.fixture(scope="module")
def first_run(tmp_path_factory: pytest.TempPathFactory, data_configs: dict) -> tuple[DictConfig, dict]:
    cfg = make_config(tmp_path_factory.mktemp("e1") / "run", data_configs)
    return cfg, run_e1(cfg, echo=False)


def _events(data_configs: dict, name: str) -> EventSet:
    return load_events(data_configs[name]["path"], unify=True)


def _eval_events(data_configs: dict, spec: dict) -> EventSet:
    events = _events(data_configs, spec["data"])
    if spec["split"] != "all":
        events = split_by_folds(events, data_configs[spec["data"]]["split"]["file"], 0)[spec["split"]]
    return EventSet(list(events)[: spec.get("max_events")])


def _files(root: Path) -> dict[str, int]:
    """mtime (ns) of every output file except the logs and the status."""
    skip = {"status.json", "run.log", "errors.log"}
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file() and p.name not in skip}


# ------------------------------------------------------------------------------------- tests
def test_first_run_complete(first_run: tuple[DictConfig, dict]) -> None:
    cfg, status = first_run
    root = Path(cfg.root)
    assert status["finished"] and status["failures"] == []
    n = len(MODELS) * len(SEEDS)
    assert status["checkpoints"] == {"done": n, "total": n, "trained_now": n}
    assert status["tables"]["done"] == status["tables"]["total"] == n * len(DATASETS)
    assert json.loads((root / "status.json").read_text(encoding="utf-8"))["finished"]
    assert "E1 run" in (root / "logs" / "run.log").read_text(encoding="utf-8")
    assert not (root / "logs" / "errors.log").exists()
    for model in MODELS:
        for seed in SEEDS:
            ckpt = checkpoint_dir(root, model, seed)
            assert {"model.pt", "train_log.json", "metrics.json"} <= {p.name for p in ckpt.iterdir()}
            metrics = json.loads((ckpt / "metrics.json").read_text(encoding="utf-8"))
            assert set(metrics["datasets"]) == set(DATASETS)
            assert all(np.isfinite(m["nll"]) and 0.0 <= m["picp"] <= 1.0 for m in metrics["datasets"].values())
    # the seed-independent IDM baseline is trained once and copied
    ckpts = [checkpoint_dir(root, "idm_white_noise", s) for s in SEEDS]
    assert (ckpts[0] / "model.pt").read_bytes() == (ckpts[1] / "model.pt").read_bytes()
    assert json.loads((ckpts[1] / "train_log.json").read_text(encoding="utf-8"))["copied_from"] == "seed0"
    idm = json.loads((root / "idm.json").read_text(encoding="utf-8"))
    assert {"params", "report", "reference", "key"} <= set(idm) and idm["reference"] is None
    assert set(idm["calibrations"]) == {"closed_loop"}  # no selected model needs the mle calibration


def test_tables_columns_rows_and_values(first_run: tuple[DictConfig, dict], data_configs: dict) -> None:
    cfg, _ = first_run
    root = Path(cfg.root)
    params = IDMParams.from_dict(json.loads((root / "idm.json").read_text(encoding="utf-8"))["params"])
    for dataset, spec in DATASETS.items():
        frame, meta = tables.load_windows(root, dataset)
        assert list(frame.columns) == list(tables.WINDOW_COLUMNS)
        assert list(tables.WINDOW_COLUMNS) == [*tables.WINDOW_INT, "subset", *tables.WINDOW_FLOAT, "y_prev"]
        assert {c: frame[c].dtype.type for c in WINDOW_DTYPES} == WINDOW_DTYPES
        assert meta["target"] == "a" and "target" not in meta["spec"]  # the default target keeps the old spec
        np.testing.assert_array_equal(frame["y_prev"], frame["a_prev"])  # persistence of a[t] is a[t-1]
        assert isinstance(frame["subset"].dtype, pd.CategoricalDtype) or frame["subset"].dtype == object
        assert meta["stride"] == spec["stride"] and meta["counts"]["windows"] == len(frame) == meta["n_windows"]
        # every column agrees with the (unified) source events, row by row
        events = _eval_events(data_configs, spec)
        assert meta["event_ids"] == [ev.event_id for ev in events]
        windows = make_windows(events, stride=spec["stride"])
        np.testing.assert_array_equal(frame["event_index"], windows.event_index)
        np.testing.assert_array_equal(frame["t_index"], windows.t_index)
        evs = [events[i] for i in frame["event"]]
        t = frame["t_index"].to_numpy()
        for col, attr, shift in (("s", "s", 0), ("dv", "dv", 0), ("v", "v", 0), ("v_lead", "v_lead", 0),
                                 ("y", "a", 0), ("a_prev", "a", 1)):  # fmt: skip
            expected = np.array([getattr(ev, attr)[k - shift] for ev, k in zip(evs, t)], dtype=np.float32)
            np.testing.assert_array_equal(frame[col].to_numpy(), expected, err_msg=f"{dataset}.{col}")
        for model in MODELS:
            for seed in SEEDS:
                pred = tables.load_preds(root, model, seed, dataset)
                assert list(pred.columns) == list(tables.PRED_COLUMNS) and len(pred) == len(frame)
                assert pred["covered"].dtype == np.uint8
                assert all(pred[c].dtype == np.float32 for c in tables.PRED_FLOAT)
                assert np.isfinite(pred[["mean", "var", "nll", "crps", "width", "pit"]].to_numpy()).all()
        # row order: the IDM baseline predicts the IDM acceleration of the state in the same row
        s, dv, v = (frame[c].to_numpy(np.float64) for c in ("s", "dv", "v"))
        expected_mean = idm_accel(s, dv, v, params).astype(np.float32)
        np.testing.assert_array_equal(tables.load_preds(root, "idm_white_noise", 0, dataset)["mean"], expected_mean)
    # row order of a network: the checkpoint's predictions on windows rebuilt from the events
    spec = DATASETS["ngsim"]
    events = _eval_events(data_configs, spec)
    windows = make_windows(events, stride=spec["stride"])
    model = load_model(checkpoint_dir(root, "lstm_const_sigma", 1) / "model.pt")
    stored = tables.load_preds(root, "lstm_const_sigma", 1, "ngsim")
    np.testing.assert_allclose(stored["mean"], model.predict_dist(windows.X).mean(), rtol=1e-5, atol=1e-6)


def test_splits_folds_subsets(first_run: tuple[DictConfig, dict], data_configs: dict) -> None:
    cfg, _ = first_run
    root = Path(cfg.root)
    folds = json.loads(Path(data_configs["hd"]["split"]["file"]).read_text(encoding="utf-8"))["folds"]
    train_ids = {e for e, k in folds.items() if k not in (0, 1)}
    for dataset, fold in (("highd", 0), ("highd_val", 1)):
        frame, meta = tables.load_windows(root, dataset)
        assert (frame["fold"] == fold).all()
        ids = {meta["event_ids"][i] for i in frame["event"].unique()}
        assert ids and not ids & train_ids
        assert all(folds[e] == fold for e in ids)
    ng, meta = tables.load_windows(root, "ngsim")
    ng_folds = json.loads(Path(data_configs["ng"]["split"]["file"]).read_text(encoding="utf-8"))["folds"]
    np.testing.assert_array_equal(ng["fold"], [ng_folds[meta["event_ids"][i]] for i in ng["event"]])
    # the stored ("dataset") acceleration of this set was replaced by the common definition (y is checked
    # against the unified events in test_tables_columns_rows_and_values)
    raw = load_events(data_configs["ng"]["path"], unify=False)
    raw_y = [raw[meta["event_ids"][e]].a[t] for e, t in zip(ng["event"], ng["t_index"])]
    assert not np.allclose(ng["y"], np.asarray(raw_y, dtype=np.float32))
    oa, meta = tables.load_windows(root, "openacc")
    assert meta["subsets"] == ["ACC", "Human"] and set(oa["subset"].astype(str)) == {"ACC", "Human"}
    assert meta["n_events"] == 10  # max_events cap: the first 10 events


def test_training_density(first_run: tuple[DictConfig, dict], data_configs: dict) -> None:
    cfg, _ = first_run
    with np.load(Path(cfg.root) / "train_density.npz") as z:
        density = {k: z[k] for k in z.files}
    train = split_by_folds(_events(data_configs, "hd"), data_configs["hd"]["split"]["file"], 0)["train"]
    windows = make_windows(train, stride=1)
    assert int(density["stride"]) == 1 and int(density["n_total"]) == len(windows)
    assert int(density["counts"].sum()) + int(density["n_outside"]) == len(windows)
    grid = OmegaConf.load(REPO / "configs" / "maps" / "default.yaml").grid
    s_edges, dv_edges = grid_edges(grid)
    np.testing.assert_array_equal(density["s_edges"], s_edges)
    np.testing.assert_array_equal(density["dv_edges"], dv_edges)
    assert density["counts"].shape == (24, 20)
    counts, outside = bin_counts(windows.X[:, -1, 0], windows.X[:, -1, 1], s_edges, dv_edges)
    np.testing.assert_array_equal(density["counts"], counts)
    assert outside == int(density["n_outside"])


def test_bin_counts_edges() -> None:
    s_edges, dv_edges = np.array([0.0, 5.0, 10.0]), np.array([-1.0, 0.0, 1.0])
    s = np.array([0.0, 4.999, 5.0, 9.99, 10.0, -0.1, 3.0, np.nan])
    dv = np.array([-1.0, 0.5, 0.0, 0.999, 0.0, 0.0, 1.0, 0.0])
    counts, outside = bin_counts(s, dv, s_edges, dv_edges)
    np.testing.assert_array_equal(counts, [[1, 1], [0, 2]])  # right-open bins
    assert outside == 4  # s on the upper edge, s below the grid, dv on the upper edge, NaN


def test_resume_and_regenerate(first_run: tuple[DictConfig, dict]) -> None:
    cfg, _ = first_run
    root = Path(cfg.root)
    before = _files(root)
    status = run_e1(cfg, echo=False)
    assert status["finished"] and status["failures"] == []
    assert status["tables"]["written_now"] == 0 and status["checkpoints"]["trained_now"] == 0
    assert status["done"] == status["total"]
    assert _files(root) == before  # nothing was rewritten
    assert "tables 24/24 (0 new)" in (root / "logs" / "run.log").read_text(encoding="utf-8")

    target = tables.preds_path(root, "lstm_const_sigma", 1, "ngsim")
    original = pd.read_parquet(target)
    target.unlink()
    status = run_e1(cfg, echo=False)
    assert status["tables"]["written_now"] == 1 and status["checkpoints"]["trained_now"] == 0
    after = _files(root)
    changed = {k for k in after if after[k] != before.get(k)}
    rel = str(target.relative_to(root))
    assert changed == {rel, str(Path("checkpoints/lstm_const_sigma/seed1/metrics.json"))}
    pd.testing.assert_frame_equal(pd.read_parquet(target), original, check_exact=True)


def test_sampled_rows_selection() -> None:
    lengths = np.array([60, 81, 120, 200])
    event_index = np.repeat(np.arange(4), [31, 52, 91, 171])
    t_index = np.concatenate([np.arange(29, n) for n in lengths])
    full = t_index + 50 < lengths[event_index]
    rows = sampled_rows(event_index, t_index, lengths, max_rows=40, seed=0)
    assert rows.size == 40 and np.all(np.diff(rows) > 0) and full[rows].all()
    np.testing.assert_array_equal(rows, sampled_rows(event_index, t_index, lengths, max_rows=40, seed=0))
    assert not np.array_equal(rows, sampled_rows(event_index, t_index, lengths, max_rows=40, seed=1))
    everything = sampled_rows(event_index, t_index, lengths, max_rows=10_000, seed=0)
    np.testing.assert_array_equal(everything, np.flatnonzero(full))
    np.testing.assert_array_equal(sampled_rows(event_index, t_index, lengths, None), np.flatnonzero(full))
    assert not full[event_index == 0].any() and full[(event_index == 1) & (t_index == 30)].all()


class _Boom(CFModel):
    """A model whose training always fails."""

    def fit(self, train, val=None, *, seed=0, verbose=False):  # noqa: ANN001, ANN201
        raise RuntimeError("boom: injected failure")

    def predict_dist(self, state_history, *, v_lead_future=None, batch_size=8192):  # noqa: ANN001, ANN201
        raise NotImplementedError

    def sample_step(self, hist, state, generator):  # noqa: ANN001, ANN201
        raise NotImplementedError

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict) -> None:
        pass


@pytest.fixture
def boom_model():
    register("e1_test_boom")(_Boom)
    yield "e1_test_boom"
    MODEL_REGISTRY.pop("e1_test_boom", None)


def test_failing_model_is_logged(tmp_path: Path, data_configs: dict, boom_model: str) -> None:
    datasets = {k: DATASETS[k] for k in ("highd", "openacc")}
    cfg = make_config(tmp_path / "run", data_configs, models=["boom", "idm_white_noise"], seeds=[0], datasets=datasets)
    cfg.model_configs = {"boom": {"name": boom_model, "kwargs": {}, "train": {}, "needs_idm": False}}
    config_file = tmp_path / "e1_test.yaml"
    OmegaConf.save(cfg, config_file)
    assert _load_script("run_e1").main(["--config", str(config_file), "--quiet"]) == 1

    root = tmp_path / "run"
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["finished"] and [f["unit"] for f in status["failures"]] == ["boom/seed0"]
    errors = (root / "logs" / "errors.log").read_text(encoding="utf-8")
    assert "boom/seed0" in errors and "Traceback" in errors and "RuntimeError: boom: injected failure" in errors
    assert "FAILED boom/seed0" in (root / "logs" / "run.log").read_text(encoding="utf-8")
    for dataset in datasets:  # the model after the failing one ran completely
        assert tables.preds_path(root, "idm_white_noise", 0, dataset).exists()
        assert not tables.preds_path(root, "boom", 0, dataset).exists()
    assert not (checkpoint_dir(root, "boom", 0) / "model.pt").exists()


def test_named_idm_calibrations(tmp_path: Path, data_configs: dict) -> None:
    datasets = {"highd": DATASETS["highd"]}
    labels = {"idm_white_noise": "closed_loop", "idm_white_noise_mle": "mle"}
    cfg = make_config(tmp_path / "both", data_configs, models=list(labels), seeds=[0], datasets=datasets)
    root = Path(cfg.root)
    assert run_e1(cfg, echo=False)["failures"] == []
    idm = json.loads((root / "idm.json").read_text(encoding="utf-8"))
    entries = idm["calibrations"]
    assert set(entries) == {"closed_loop", "mle"}
    assert all(idm[k] == entries["closed_loop"][k] for k in ("params", "report", "key"))  # top level: closed loop
    assert entries["closed_loop"]["report"]["objective"] == "spacing" and entries["mle"]["report"]["objective"] == "accel"
    assert entries["closed_loop"]["params"] != entries["mle"]["params"]
    assert all({"accel_rmse", "spacing_rmse"} <= set(e["train_eval"]) for e in entries.values())
    frame, _ = tables.load_windows(root, "highd")
    s, dv, v = (frame[c].to_numpy(np.float64) for c in ("s", "dv", "v"))
    for label, name in labels.items():  # each model uses its own calibration
        params = IDMParams.from_dict(entries[name]["params"])
        expected = idm_accel(s, dv, v, params).astype(np.float32)
        np.testing.assert_array_equal(tables.load_preds(root, label, 0, "highd")["mean"], expected)
        log = json.loads((checkpoint_dir(root, label, 0) / "train_log.json").read_text(encoding="utf-8"))
        assert log["idm"] == {"name": name, "key": entries[name]["key"]}

    # only the calibrations that a selected model needs are computed
    only = make_config(tmp_path / "mle_only", data_configs, models=["idm_white_noise_mle", "lstm_const_sigma"],
                       seeds=[0], datasets=datasets)  # fmt: skip
    assert run_e1(only, echo=False)["failures"] == []
    stored = json.loads((tmp_path / "mle_only" / "idm.json").read_text(encoding="utf-8"))
    assert set(stored["calibrations"]) == {"mle"} and "params" not in stored

    # resume with a changed calibration key: refused, the other model runs, nothing is rewritten
    before = _files(root)
    cfg.idm.mle.max_events = 15
    status = run_e1(cfg, echo=False)
    assert [f["unit"] for f in status["failures"]] == ["idm_white_noise_mle/config"]
    assert "has the key" in status["failures"][0]["error"]
    assert _files(root) == before
    # the checkpoint itself carries name and key of its calibration: a recalibration with the new key is refused
    (root / "idm.json").unlink()
    status = run_e1(cfg, echo=False)
    assert [f["unit"] for f in status["failures"]] == ["idm_white_noise_mle/seed0"]
    assert "IDM calibration" in status["failures"][0]["error"]


def test_select_w_phys_rule() -> None:
    select = _load_script("select_w_phys").select_value

    def rows(*pairs: tuple[float, float]) -> list[dict]:
        return [{"w_phys": w, "pinball": p} for w, p in pairs]

    # best among w_phys > 0: 0.03; 0.01 and 0.1 lie within 1 % of it, 0.3 does not -> the largest, 0.1
    assert select(rows((0, 0.0900), (0.01, 0.1010), (0.03, 0.1005), (0.1, 0.1012), (0.3, 0.1100))) == 0.1
    # w_phys = 0 is never selected, even when it is the best by far
    assert select(rows((0, 0.05), (0.01, 0.10), (0.3, 0.20))) == 0.01
    # the band is relative to the best value: +0.9 % is inside, +1.2 % is outside
    assert select(rows((0.01, 0.1), (0.1, 0.1009), (0.3, 0.1012))) == 0.1
    # ties go to the largest value; non-finite results are ignored
    assert select(rows((0.01, 0.1), (0.03, 0.1), (0.1, float("nan")))) == 0.03
    assert select(rows((0.3, 0.2), (0.1, 0.2), (0.03, 0.3))) == 0.3  # the order of the rows does not matter
    with pytest.raises(ValueError):
        select(rows((0, 0.1)))
