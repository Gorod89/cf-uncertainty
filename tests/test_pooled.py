"""Pooled training domain (docs/design.md, section 16.1, experiment E6): the capped pooled bundle with its fixed
subsampling, the pool table ``pool_test`` and its windows spec, the list signal key (resume and reuse refused across
roots), the test-fold restriction of the E1 and E2 runners, the analyses with ``--in-domain pool_test`` and the E6
configs. Synthetic events (harmonised signals), CPU only, a few seconds per runner call."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from cf_unc.data.harmonise import HarmoniseConfig
from cf_unc.data.pipeline import build_pooled_bundle, cap_windows, subsample_seed
from cf_unc.data.schema import EventSet
from cf_unc.data.splits import split_by_folds
from cf_unc.data.synthetic import generate
from cf_unc.data.windows import concat_windows, count_windows, make_windows
from cf_unc.experiments import e2 as e2_mod
from cf_unc.experiments.e1 import (
    Data,
    checkpoint_dir,
    dry_run_e1,
    load_config,
    pool_key,
    run_e1,
    signal_key,
    train_datasets,
    training_signal_key,
)
from cf_unc.experiments.e1 import _display
from cf_unc.maps import io as tables
from cf_unc.maps.analysis import _Windows, pair_roles, resolve_v0
from cf_unc.maps.analysis import load_config as load_maps_config
from cf_unc.physics.idm import IDMParams, idm_accel

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "configs" / "experiment"
N_FOLDS = 5
KEY = {"signal": "harmonised", "harmonise": HarmoniseConfig().to_dict()}
QUICK = {"maxiter": 3, "popsize": 5, "polish": False}  # tiny differential-evolution budget
MODELS = ("idm_white_noise", "lstm_const_sigma")
POOL = ["highd", "ngsim"]
CAP, SEQ_CAP = 300, 50  # both bite on the training and validation folds of the two pooled sets
EVAL: dict[str, dict[str, Any]] = {
    "highd": {"data": "highd", "split": "test", "stride": 2},
    "highd_val": {"data": "highd", "split": "val", "stride": 2},
    "ngsim": {"data": "ngsim", "split": "test", "stride": 3},
    "ngsim_fn": {"data": "ngsim_fn", "split": "all", "stride": 3},
    "openacc": {"data": "openacc", "split": "all", "stride": 5},
    "pool_test": {"pool": POOL},
}


def _load_script(name: str) -> Any:
    """Module of ``scripts/<name>.py`` (the scripts directory is not a package)."""
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(f"{name}_script", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ------------------------------------------------------------------------------------ data and configs
def _write_set(root: Path, name: str, seed: int, n_drivers: int, per: int, duration: float, mode: bool = False) -> dict:
    events, _ = generate(dict(n_drivers=n_drivers, events_per_driver=per, duration_s=duration, seed=seed, dataset=name))
    for i, ev in enumerate(events):
        if mode:
            ev.meta["driver_mode"] = "ACC" if i % 2 == 0 else "Human"
    events.to_parquet(root / name)
    folds = {ev.event_id: int(ev.follower_id[3:]) % N_FOLDS for ev in events}  # driver-disjoint folds
    split = root / f"{name}_driver.json"
    split.write_text(json.dumps({"dataset": name, "n_folds": N_FOLDS, "folds": folds}), encoding="utf-8")
    return {"name": name, "kind": "parquet", "path": str(root / name), "unify_acceleration": True,
            "split": {"file": str(split), "fold": 0}}  # fmt: skip


@pytest.fixture(scope="module")
def data_configs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    root = tmp_path_factory.mktemp("pooled_events")
    return {
        "highd": _write_set(root, "highd", seed=1, n_drivers=25, per=2, duration=12.0),
        "ngsim": _write_set(root, "ngsim", seed=2, n_drivers=12, per=2, duration=15.0),
        "openacc": _write_set(root, "openacc", seed=3, n_drivers=10, per=2, duration=10.0, mode=True),
        "ngsim_fn": _write_set(root, "ngsim_fn", seed=4, n_drivers=6, per=2, duration=12.0),
    }


def make_config(root: Path, data_configs: dict, train_data: Any = tuple(POOL), datasets: dict | None = None,
                models: tuple[str, ...] = MODELS, **train: Any) -> DictConfig:  # fmt: skip
    cfg = load_config()
    cfg.root = str(root)
    cfg.device = "cpu"
    cfg.models = list(models)
    cfg.seeds = [0]
    data = list(train_data) if isinstance(train_data, (list, tuple)) else train_data
    cfg.train = {"data": data, "stride": 2, "seq_stride": 5, "overrides": {"max_epochs": 1, "batch_size": 256},
                 "max_windows_per_dataset": CAP, "max_sequences_per_dataset": SEQ_CAP, "subsample_seed": 0, **train}  # fmt: skip
    if datasets is None:
        datasets = EVAL if isinstance(train_data, (list, tuple)) else {k: v for k, v in EVAL.items() if k != "pool_test"}
    cfg.eval.datasets = OmegaConf.create(datasets)
    cfg.eval.chunk_rows = 150
    cfg.data_configs = data_configs
    cfg.idm = {"closed_loop": {"objective": "spacing", "max_events": 10, "seed": 0, **QUICK}}
    cfg.idm_reference = None
    cfg.signal = "harmonised"
    return cfg


def _files(root: Path) -> dict[str, int]:
    skip = {"status.json", "run.log", "errors.log", "timings.jsonl", "config.yaml"}
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file() and p.name not in skip}


@pytest.fixture(scope="module")
def pooled_run(tmp_path_factory: pytest.TempPathFactory, data_configs: dict) -> DictConfig:
    """A pooled E1 root (training folds of highd and ngsim, both capped): two models, seeds 0 and 1 (the across-seed
    decomposition gives the shift analysis verdict rows)."""
    cfg = make_config(tmp_path_factory.mktemp("e6") / "e6_pool", data_configs)
    cfg.seeds = [0, 1]
    status = run_e1(cfg, echo=False)
    assert status["failures"] == [] and status["finished"] and status["train_data"] == POOL
    return cfg


@pytest.fixture(scope="module")
def single_run(tmp_path_factory: pytest.TempPathFactory, data_configs: dict) -> DictConfig:
    """A root trained on highd alone (the stand-in of artifacts/e1m), the same signals."""
    cfg = make_config(tmp_path_factory.mktemp("e1s") / "e1_single", data_configs, train_data="highd",
                      models=("lstm_const_sigma",))  # fmt: skip
    assert run_e1(cfg, echo=False)["failures"] == []
    return cfg


# ------------------------------------------------------------------------------------ windows and bundle
def test_count_and_concat_windows() -> None:
    events, _ = generate(dict(n_drivers=4, events_per_driver=2, duration_s=9.0, seed=5, dataset="a"))
    lengths = [len(ev) for ev in events]
    for stride, seq_len, target in ((1, 0, "a"), (2, 0, "mean_1s"), (5, 50, "a"), (3, 0, "fwd_diff")):
        counts = count_windows(lengths, stride=stride, seq_len=seq_len, target=target)
        w = make_windows(events, stride=stride, seq_len=seq_len, target=target)
        np.testing.assert_array_equal(counts, np.bincount(w.event_index, minlength=len(events)))
    other, _ = generate(dict(n_drivers=3, events_per_driver=1, duration_s=8.0, seed=6, dataset="b"))
    wa, wb = make_windows(events, stride=2), make_windows(other, stride=2)
    both = concat_windows([wa, wb], [len(events), len(other)])
    union = make_windows(EventSet([*events, *other]), stride=2)  # the windows of the concatenated events
    for name in ("X", "y", "driver", "event_index", "t_index"):
        np.testing.assert_array_equal(getattr(both, name), getattr(union, name), err_msg=name)
    assert both.driver_names == union.driver_names and both.y_seq is None


def test_cap_windows_is_a_fixed_uniform_subsample() -> None:
    events, _ = generate(dict(n_drivers=10, events_per_driver=2, duration_s=12.0, seed=7, dataset="c"))
    w = make_windows(events, stride=1)
    capped = cap_windows(w, 100, seed=3)
    assert len(capped) == 100 and cap_windows(w, None, 3) is w and cap_windows(w, len(w), 3) is w
    key = {(int(e), int(t)): i for i, (e, t) in enumerate(zip(w.event_index, w.t_index))}
    rows = np.array([key[(int(e), int(t))] for e, t in zip(capped.event_index, capped.t_index)])
    assert np.all(np.diff(rows) > 0)  # a subset of distinct windows in their original order
    np.testing.assert_array_equal(capped.X, w.X[rows])
    np.testing.assert_array_equal(capped.y, cap_windows(w, 100, seed=3).y)  # fixed seed: the same windows
    assert not np.array_equal(capped.y, cap_windows(w, 100, seed=4).y)
    # uniform: every window is drawn with probability cap / n (inclusion frequencies over 400 seeds)
    hits = np.zeros(len(w))
    for seed in range(400):
        c = cap_windows(w, 60, seed=seed)
        hits[[key[(int(e), int(t))] for e, t in zip(c.event_index, c.t_index)]] += 1
    p_hat, p = hits / 400, 60 / len(w)
    assert hits.sum() == 400 * 60 and np.max(np.abs(p_hat - p)) < 6 * np.sqrt(p * (1 - p) / 400)
    assert subsample_seed(0, "highd", "train") != subsample_seed(0, "highd", "val") != subsample_seed(0, "ngsim", "train")
    assert subsample_seed(0, "highd", "train") == subsample_seed(0, "highd", "train") != subsample_seed(1, "highd", "train")


def test_build_pooled_bundle_caps_and_fixed_subsampling(data_configs: dict) -> None:
    parts = {}
    for name in ("highd", "ngsim", "openacc"):
        events = EventSet.from_parquet(Path(data_configs[name]["path"]))
        parts[name] = split_by_folds(events, data_configs[name]["split"]["file"], 0)
    bundle = build_pooled_bundle(parts, stride=2, max_windows=CAP, seed=0)
    assert set(bundle.windows) == {"train", "val"}
    pieces = []
    for name, by_split in parts.items():
        for split in ("train", "val"):
            full = make_windows(by_split[split], stride=2)
            info = bundle.meta["pool"][name][split]
            assert info == {"events": len(by_split[split]), "windows": len(full), "kept": min(len(full), CAP)}
        pieces.append(cap_windows(make_windows(by_split["train"], stride=2), CAP, subsample_seed(0, name, "train")))
    assert bundle.meta["pool"]["highd"]["train"]["windows"] > CAP > bundle.meta["pool"]["openacc"]["val"]["windows"]  # cap bites / not
    expected = concat_windows(pieces, [len(parts[n]["train"]) for n in parts])
    for name in ("X", "y", "driver", "event_index", "t_index"):
        np.testing.assert_array_equal(getattr(bundle.windows["train"], name), getattr(expected, name), err_msg=name)
    events = bundle.events["train"]
    assert [ev.event_id for ev in events] == [ev.event_id for n in parts for ev in parts[n]["train"]]
    w = bundle.windows["train"]  # event_index and driver codes refer to the pooled events
    assert all(w.driver_names[d] == f"{events[int(e)].dataset}/{events[int(e)].follower_id}" for d, e in zip(w.driver, w.event_index))
    # leaving a data set out keeps the subsamples of the others (the seed is mixed with the data set name)
    lodo = build_pooled_bundle({k: parts[k] for k in ("highd", "openacc")}, stride=2, max_windows=CAP, seed=0)
    n_hd, n_ng = (bundle.meta["pool"][n]["train"]["kept"] for n in ("highd", "ngsim"))
    np.testing.assert_array_equal(lodo.windows["train"].X[:n_hd], bundle.windows["train"].X[:n_hd])
    np.testing.assert_array_equal(lodo.windows["train"].X[n_hd:], bundle.windows["train"].X[n_hd + n_ng :])  # openacc
    # sequence windows with their own cap
    seq = build_pooled_bundle(parts, stride=5, seq_len=50, max_windows=SEQ_CAP, seed=0)
    assert all(c["train"]["kept"] <= SEQ_CAP for c in seq.meta["pool"].values())
    assert seq.windows["train"].y_seq.shape == (len(seq.windows["train"]), 50)


# ------------------------------------------------------------------------------------ E1 runner
def test_pooled_run_trains_on_the_capped_pool(pooled_run: DictConfig, data_configs: dict) -> None:
    root = Path(pooled_run.root)
    key = training_signal_key(pooled_run)
    assert key == [KEY, KEY] and key == [signal_key(pooled_run, n) for n in POOL]
    assert train_datasets(pooled_run) == POOL and pool_key(pooled_run)["data"] == POOL
    data = Data(pooled_run)
    bundle = data.bundle(False)
    for name in POOL:
        for split in ("train", "val"):
            c = bundle.meta["pool"][name][split]
            assert c["kept"] == min(c["windows"], CAP) and c["windows"] > CAP  # the cap bites on every part
    for model in MODELS:
        log = json.loads((checkpoint_dir(root, model, 0) / "train_log.json").read_text(encoding="utf-8"))
        assert log["signal"] == key and log["pool"] == pool_key(pooled_run)
        assert log["n_train"] == len(bundle.windows["train"]) == 2 * CAP and log["n_val"] == 2 * CAP
        assert log["pool_windows"] == bundle.meta["pool"]
    idm = json.loads((root / "idm.json").read_text(encoding="utf-8"))
    assert idm["key"]["data"] == POOL and idm["key"]["fold"] == [0, 0] and idm["key"]["signal"] == key
    assert idm["calibrations"]["closed_loop"]["train_eval"]["n_events"] == len(data.train_events())
    assert len(data.train_events()) == sum(len(data.split(n, "train")) for n in POOL)
    with np.load(root / "train_density.npz") as z:  # the density of the pooled one-step training windows
        assert int(z["n_total"]) == 2 * CAP and int(z["stride"]) == 2
        assert json.loads(str(z["signal"])) == key and json.loads(str(z["pool"])) == pool_key(pooled_run)


def test_pool_test_table_is_the_union_of_the_test_folds(pooled_run: DictConfig) -> None:
    root = Path(pooled_run.root)
    frame, meta = tables.load_windows(root, "pool_test")
    parts = [tables.load_windows(root, m) for m in POOL]
    spec = meta["spec"]
    assert spec["members"] == POOL and spec["data"] == POOL and spec["split"] == "test" and spec["signal"] == "harmonised"
    assert spec["pool"] == {m: p[1]["spec"] for m, p in zip(POOL, parts)}  # its own spec: the member specs
    assert meta["n_windows"] == sum(len(p[0]) for p in parts) == len(frame)
    assert meta["event_ids"] == [e for p in parts for e in p[1]["event_ids"]]
    assert [m["rows"] for m in meta["members"]] == [[0, len(parts[0][0])], [len(parts[0][0]), len(frame)]]
    assert set(meta["harmonisation"]) == set(POOL) and meta["stride"] == {"highd": 2, "ngsim": 3}
    offset = 0
    for (part, pmeta), info in zip(parts, meta["members"]):
        rows = frame.iloc[info["rows"][0] : info["rows"][1]].reset_index(drop=True)
        for col in ("t_index", "s", "dv", "v", "v_lead", "a_prev", "y", "y_prev", "fold"):
            np.testing.assert_array_equal(rows[col].to_numpy(), part[col].to_numpy(), err_msg=col)
        np.testing.assert_array_equal(rows["event"].to_numpy(), part["event"].to_numpy() + offset)  # re-coded events
        np.testing.assert_array_equal(rows["event_index"].to_numpy(), part["event_index"].to_numpy() + offset)
        assert [meta["event_ids"][e] for e in rows["event"]] == [pmeta["event_ids"][e] for e in part["event"]]
        assert [meta["driver_names"][d] for d in rows["driver"]] == [pmeta["driver_names"][d] for d in part["driver"]]
        assert rows["subset"].astype(str).tolist() == part["subset"].astype(str).tolist()
        offset += pmeta["n_events"]
    assert (frame["fold"] == 0).all()
    for model in MODELS:  # predictions: the members' tables concatenated, no prediction of its own
        pool = tables.load_preds(root, model, 0, "pool_test")
        joined = pd.concat([tables.load_preds(root, model, 0, m) for m in POOL], ignore_index=True)
        pd.testing.assert_frame_equal(pool, joined)
        metrics = json.loads((checkpoint_dir(root, model, 0) / "metrics.json").read_text(encoding="utf-8"))
        assert "pool_test" in metrics["datasets"] and "assembled_from" in metrics["datasets"]["pool_test"]["timing"]
    assert "rows assembled from highd, ngsim" in (root / "logs" / "run.log").read_text(encoding="utf-8")


def test_resume_reassembles_a_missing_pool_table(pooled_run: DictConfig) -> None:
    root = Path(pooled_run.root)
    before = _files(root)
    status = run_e1(pooled_run, echo=False)
    assert status["failures"] == [] and status["tables"]["written_now"] == 0 and _files(root) == before
    target = tables.preds_path(root, "lstm_const_sigma", 0, "pool_test")
    original = pd.read_parquet(target)
    target.unlink()
    status = run_e1(pooled_run, echo=False)
    assert status["failures"] == [] and status["tables"]["written_now"] == 1 and status["checkpoints"]["trained_now"] == 0
    pd.testing.assert_frame_equal(pd.read_parquet(target), original, check_exact=True)


def test_list_signal_key_refuses_resume_and_reuse_across_roots(pooled_run: DictConfig, single_run: DictConfig,
                                                               data_configs: dict, tmp_path: Path) -> None:  # fmt: skip
    pooled, single = Path(pooled_run.root), Path(single_run.root)
    # a config of one training data set on the pooled root: density, IDM and checkpoints refused, nothing rewritten
    before = _files(pooled)
    one = make_config(pooled, data_configs, train_data="highd")
    status = run_e1(one, echo=False)
    assert sorted(f["unit"] for f in status["failures"]) == ["idm_white_noise/config", "lstm_const_sigma/seed0", "train_density"]
    errors = {f["unit"]: f["error"] for f in status["failures"]}
    assert "pooled [" in errors["lstm_const_sigma/seed0"] and "has the key" in errors["idm_white_noise/config"]
    assert {k: v for k, v in _files(pooled).items() if "metrics.json" not in k} == {
        k: v for k, v in before.items() if "metrics.json" not in k}  # fmt: skip
    plan = dry_run_e1(one, echo=False)
    assert {c["unit"] for c in plan["conflicts"]} == {"train_density", "idm/closed_loop", "idm_white_noise/config",
                                                      "lstm_const_sigma/seed0"}  # fmt: skip
    # the pooled config on the root of one training data set: refused the other way round
    plan = dry_run_e1(make_config(single, data_configs, models=("lstm_const_sigma",)), echo=False)
    assert {c["unit"] for c in plan["conflicts"]} == {"train_density", "lstm_const_sigma/seed0"}
    assert "harmonised (" in plan["conflicts"][-1]["error"] and "pooled [" in plan["conflicts"][-1]["error"]
    # reuse in either direction fails the checks
    reuse = make_config(tmp_path / "e6_reuse", data_configs, models=("lstm_const_sigma",),
                        reuse_checkpoints=None)  # fmt: skip
    reuse.reuse_checkpoints = {"lstm_const_sigma": str(single / "checkpoints")}
    plan = dry_run_e1(reuse, echo=False)
    assert [c["unit"] for c in plan["conflicts"]] == ["lstm_const_sigma/seed0 (reuse)"] and plan["reuse"] == []
    back = make_config(tmp_path / "e1_reuse", data_configs, train_data="highd", models=("lstm_const_sigma",))
    back.reuse_checkpoints = {"lstm_const_sigma": str(pooled / "checkpoints")}
    plan = dry_run_e1(back, echo=False)
    assert [c["unit"] for c in plan["conflicts"]] == ["lstm_const_sigma/seed0 (reuse)"]
    assert not (tmp_path / "e6_reuse").exists() and not (tmp_path / "e1_reuse").exists()
    # the same pool with another cap: the pool key refuses checkpoints and density
    capped = make_config(pooled, data_configs, max_windows_per_dataset=CAP + 1)
    plan = dry_run_e1(capped, echo=False)
    units = {c["unit"]: c["error"] for c in plan["conflicts"]}
    assert set(units) == {"train_density", "idm_white_noise/seed0", "lstm_const_sigma/seed0"}
    assert "at most 301 one-step" in units["lstm_const_sigma/seed0"]


def test_pooled_configs_never_evaluate_training_events(data_configs: dict, tmp_path: Path) -> None:
    bad = {**EVAL, "ngsim": {"data": "ngsim", "split": "all", "stride": 3}}
    for fn in (run_e1, dry_run_e1):
        with pytest.raises(ValueError, match="holds training events"):
            fn(make_config(tmp_path / "bad", data_configs, datasets=bad), echo=False)
    partial = {**EVAL, "pool_test": {"pool": ["highd"]}}
    with pytest.raises(ValueError, match="every training data set"):
        run_e1(make_config(tmp_path / "bad", data_configs, datasets=partial), echo=False)
    wrong = {**EVAL, "pool_test": {"pool": ["highd", "highd_val"]}}
    with pytest.raises(ValueError, match="test fold"):
        run_e1(make_config(tmp_path / "bad", data_configs, datasets=wrong), echo=False)
    with pytest.raises(ValueError, match="distinct"):
        run_e1(make_config(tmp_path / "bad", data_configs, train_data=("highd", "highd")), echo=False)
    assert not (tmp_path / "bad").exists()


def test_dry_run_states_the_windows_after_the_caps(pooled_run: DictConfig, data_configs: dict, tmp_path: Path,
                                                   capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    cfg = make_config(tmp_path / "fresh", data_configs)
    plan = dry_run_e1(cfg, echo=True)
    assert not (tmp_path / "fresh").exists() and plan["reuse"] == [] and plan["conflicts"] == []
    assert len(plan["checkpoints"]["missing"]) == 2 and len(plan["tables"]["missing"]) == 2 * len(EVAL)
    bundle = Data(pooled_run).bundle(False)
    one = plan["training"]["one_step"]
    for name in POOL:
        for split in ("train", "val"):
            assert one["datasets"][name][split] == bundle.meta["pool"][name][split]
    assert one["total"] == {"train": 2 * CAP, "val": 2 * CAP} and one["cap"] == CAP and one["stride"] == 2
    seq = Data(pooled_run).bundle(True)
    assert plan["training"]["sequences"]["datasets"]["ngsim"]["train"] == seq.meta["pool"]["ngsim"]["train"]
    assert plan["evaluated"]["pool_test"] == {"pool": POOL} and plan["keys"]["pool"] == pool_key(cfg)
    printed = capsys.readouterr().out
    assert "one-step windows after the caps" in printed and f"highd train {bundle.meta['pool']['highd']['train']['windows']:,} -> 300" in printed
    # scripts/run_e1.py --datasets pool_test brings the members along
    config_file = tmp_path / "e6.yaml"
    OmegaConf.save(cfg, config_file)
    assert _load_script("run_e1").main(["--config", str(config_file), "--dry-run", "--datasets", "pool_test", "--quiet"]) == 0
    brief = json.loads(capsys.readouterr().out)
    assert set(brief["evaluated"]) == {"pool_test", "highd", "ngsim"} and brief["tables"] == {"done": 0, "missing": 6}
    assert brief["training"]["one_step"]["total"]["train"] == 2 * CAP and brief["reuse"] == []


def _write_idm(path: Path, v0: float) -> Path:
    """An idm.json as the runner writes it (calibrations closed_loop and mle, the closed loop at the top level)."""
    params = {"closed_loop": {"v0": v0, "T": 1.2, "s0": 2.5, "a_max": 0.9, "b": 1.6, "delta": 4.0},
              "mle": {"v0": v0 - 3.0, "T": 0.8, "s0": 2.0, "a_max": 0.5, "b": 2.5, "delta": 4.0}}  # fmt: skip
    entries = {name: {"params": p, "report": {"objective": "spacing" if name == "closed_loop" else "accel", "seconds": 1.0},
                      "train_eval": {"accel_rmse": 1.0, "n_events": 7}, "key": {"data": "highd", "fold": 0, "signal": KEY,
                                                                                 "objective": name}}
               for name, p in params.items()}  # fmt: skip
    content = {"calibrations": entries, **{k: entries["closed_loop"][k] for k in ("params", "report", "key")}, "reference": None}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_idm_from_copies_a_frozen_prior(data_configs: dict, tmp_path: Path) -> None:
    source = _write_idm(tmp_path / "e1m" / "idm.json", v0=31.0)
    src = json.loads(source.read_text(encoding="utf-8"))
    models = ("idm_white_noise", "idm_white_noise_mle")
    cfg = make_config(tmp_path / "e6_idm", data_configs, models=models, idm_from=str(source))
    plan = dry_run_e1(cfg, echo=False)
    assert plan["conflicts"] == [] and plan["idm_from"] == _display(source)
    assert plan["idm"] == {"done": [], "missing": [], "reuse": ["idm/closed_loop", "idm/mle"]}  # nothing to fit
    assert plan["checkpoints"]["missing"] == ["idm_white_noise/seed0", "idm_white_noise_mle/seed0"] and not (tmp_path / "e6_idm").exists()
    status = run_e1(cfg, echo=False)
    root = Path(cfg.root)
    assert status["failures"] == []
    log = (root / "logs" / "run.log").read_text(encoding="utf-8")
    assert "calibrating" not in log and log.count("reused from") == 2
    stored = json.loads((root / "idm.json").read_text(encoding="utf-8"))
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    n_train = len(Data(cfg).train_events())
    for name in ("closed_loop", "mle"):
        entry, origin = stored["calibrations"][name], src["calibrations"][name]
        assert entry["params"] == origin["params"] and entry["report"] == origin["report"]
        assert entry["key"] == {"idm_from": _display(source), "name": name, "source_key": origin["key"], "params": origin["params"]}
        assert entry["reused_from"]["path"] == _display(source) and entry["reused_from"]["sha256"] == sha
        assert "frozen" in entry["reused_from"]["note"] and entry["reused_from"]["train_eval_source"] == origin["train_eval"]
        assert entry["train_eval"]["n_events"] == n_train  # the copied prior evaluated on the pooled training events
    assert stored["params"] == src["params"] and stored["key"] == stored["calibrations"]["closed_loop"]["key"]
    assert resolve_v0(root, load_maps_config()) == 31.0  # the regimes of the analyses use the copied v0
    frame, _ = tables.load_windows(root, "highd")
    s_, dv, v = (frame[c].to_numpy(np.float64) for c in ("s", "dv", "v"))
    for label, name in (("idm_white_noise", "closed_loop"), ("idm_white_noise_mle", "mle")):
        expected = idm_accel(s_, dv, v, IDMParams.from_dict(src["calibrations"][name]["params"])).astype(np.float32)
        np.testing.assert_array_equal(tables.load_preds(root, label, 0, "highd")["mean"], expected)
        train_log = json.loads((checkpoint_dir(root, label, 0) / "train_log.json").read_text(encoding="utf-8"))
        assert train_log["idm"] == {"name": name, "key": stored["calibrations"][name]["key"]}
    # resume with the same idm_from: nothing is rewritten
    before = _files(root)
    assert run_e1(cfg, echo=False)["failures"] == [] and _files(root) == before
    # another idm_from (other parameters, or the same file at another path) and a fitted calibration: refused
    other = _write_idm(tmp_path / "other" / "idm.json", v0=33.0)
    copy = tmp_path / "copy" / "idm.json"
    copy.parent.mkdir()
    copy.write_bytes(source.read_bytes())
    refused = {"idm/closed_loop", "idm/mle", "idm_white_noise/config", "idm_white_noise_mle/config"}
    for path in (other, copy):
        plan = dry_run_e1(make_config(root, data_configs, models=models, idm_from=str(path)), echo=False)
        assert {c["unit"] for c in plan["conflicts"]} == refused and "has the key" in plan["conflicts"][0]["error"]
    status = run_e1(make_config(root, data_configs, models=models, idm_from=str(other)), echo=False)
    assert {f["unit"] for f in status["failures"]} == refused and _files(root) == before
    plan = dry_run_e1(make_config(root, data_configs, models=("idm_white_noise",)), echo=False)  # no idm_from
    assert {c["unit"] for c in plan["conflicts"]} == {"idm/closed_loop", "idm_white_noise/config"}
    missing = dry_run_e1(make_config(tmp_path / "e6_none", data_configs, models=models,
                                     idm_from=str(tmp_path / "none" / "idm.json")), echo=False)  # fmt: skip
    assert {c["unit"] for c in missing["conflicts"]} >= {"idm/closed_loop", "idm_white_noise/config"}
    assert "does not exist" in missing["conflicts"][0]["error"]
    # a checkpoint of this root is no reuse source for a root with another idm_from
    reuse = make_config(tmp_path / "e6_other", data_configs, models=("idm_white_noise",), idm_from=str(other))
    reuse.reuse_checkpoints = {"idm_white_noise": str(root / "checkpoints")}
    plan = dry_run_e1(reuse, echo=False)
    assert [c["unit"] for c in plan["conflicts"]] == ["idm_white_noise/seed0 (reuse)"] and plan["idm"]["reuse"] == ["idm/closed_loop", "idm/mle"]


# ------------------------------------------------------------------------------------ E2 runner
def _e2_config(root: Path, e1: DictConfig, data_configs: dict, train_data: Any = tuple(POOL), **datasets: Any) -> DictConfig:
    cfg = e2_mod.load_config()
    cfg.root = str(root)
    cfg.device = "cpu"
    cfg.signal = "harmonised"
    cfg.train = {"data": list(train_data) if isinstance(train_data, (list, tuple)) else train_data}
    cfg.targets = OmegaConf.create({"a": {"checkpoints": str(Path(e1.root) / "checkpoints"), "hold_steps": 1}})
    cfg.models = ["lstm_const_sigma"]
    cfg.seeds = [0]
    cfg.variants = OmegaConf.create({})
    rolled = {"highd_val": {"data": "highd", "split": "val", "n_events": 4},
              "highd": {"data": "highd", "split": "test", "n_events": 4},
              "ngsim": {"data": "ngsim", "split": "test", "n_events": 4},
              "openacc": {"data": "openacc", "split": "all", "n_events": 4}}  # fmt: skip
    cfg.datasets = OmegaConf.create({**rolled, **datasets})
    cfg.rollout = OmegaConf.create({"n_rollouts": 6, "n_rollouts_sampled": 4, "horizon_s": 2.0, "horizons_s": [1, 2],
                                    "max_batch": 100, "stream_events": 3, "event_seed": 0, "seed": 0})  # fmt: skip
    cfg.calibration.n_min = 2
    cfg.e1_roots = {"a": str(e1.root)}
    cfg.data_configs = data_configs
    return cfg


def test_e2_draws_training_sets_from_their_test_folds(pooled_run: DictConfig, data_configs: dict, tmp_path: Path) -> None:
    cfg = _e2_config(tmp_path / "e2_pool", pooled_run, data_configs)
    status = e2_mod.run_e2(cfg, echo=False)
    root = Path(cfg.root)
    assert status["failures"] == [] and status["targets"]["a"] == {"done": 4, "total": 4}
    folds = {name: json.loads(Path(c["split"]["file"]).read_text(encoding="utf-8"))["folds"] for name, c in data_configs.items()}
    for dataset, fold in (("highd", 0), ("ngsim", 0), ("highd_val", 1)):
        events = e2_mod.tables.load_events_table(root, "a", dataset)
        assert len(events) == 4 and (events["fold"] == fold).all()
        assert all(folds[dataset.split("_")[0]][e] == fold for e in events["event_id"].astype(str))
    held_out = e2_mod.tables.load_events_table(root, "a", "openacc")
    assert set(held_out["fold"]) - {0} and len(held_out) == 4  # the held-out set: all its events, any fold
    spec = json.loads((root / "a" / "spec.json").read_text(encoding="utf-8"))
    assert spec["train_data"] == POOL and spec["train_signal"] == [KEY, KEY]
    runner = e2_mod.E2Runner(cfg, echo=False)
    assert runner.train_signal_key() == training_signal_key(pooled_run)
    runner.check_signal(runner.checkpoint("a", "lstm_const_sigma", 0))
    # a training set rolled on all its events: refused before anything is created
    for split in ("all", "train"):
        with pytest.raises(ValueError, match="test fold only"):
            e2_mod.run_e2(_e2_config(tmp_path / "bad", pooled_run, data_configs, ngsim={"data": "ngsim", "split": split,
                                                                                         "n_events": 4}), echo=False)  # fmt: skip
    assert not (tmp_path / "bad").exists()
    # one training data set: the pooled checkpoints are refused (check_signal compares with the list key)
    one = e2_mod.run_e2(_e2_config(tmp_path / "e2_one", pooled_run, data_configs, train_data="highd"), echo=False)
    assert [f["unit"] for f in one["failures"]] == ["a/lstm_const_sigma/seed0"]
    assert "for the training data set highd" in one["failures"][0]["error"]
    # analyze_e2: the one-step conformal analysis reads the E1 root of config.yaml (the pooled root, pool_test included)
    analyze = _load_script("analyze_e2")
    out = analyze.analyze_e2(root, settings={"bootstrap": {"n_boot": 50}}, figures=False, verbose=False)
    one_step = out["onestep_conformal"]
    assert {"highd", "ngsim", "openacc", "pool_test"} <= set(one_step["dataset"]) and "highd_val" not in set(one_step["dataset"])
    assert set(out["h34_verdicts"]["dataset"]) == {"highd", "ngsim", "openacc"}
    info = json.loads((root / "tables" / "analysis_info.json").read_text(encoding="utf-8"))
    assert info["e1_roots"] == {"a": str(pooled_run.root)} and info["targets"] == ["a"]


# ------------------------------------------------------------------------------------ analyses
def _w(name: str, sources: tuple[str, ...], members: tuple[str, ...] = ()) -> _Windows:
    return _Windows(name=name, source=",".join(sources), n=0, full=None, views=[], sources=sources, members=members)  # type: ignore[arg-type]


def test_pair_roles() -> None:
    of = {"ngsim_fn": "ngsim"}
    legacy = {"highd": _w("highd", ("highd",)), "highd_val": _w("highd_val", ("highd",)), "ngsim": _w("ngsim", ("ngsim",)),
              "ngsim_fn": _w("ngsim_fn", ("ngsim_fn",)), "openacc": _w("openacc", ("openacc",))}  # fmt: skip
    expected = {"highd": None, "highd_val": "placebo", "ngsim": "shift", "ngsim_fn": "cross_check", "openacc": "shift"}
    assert pair_roles(legacy, "highd", None, None, ["ngsim_fn"], of) == expected  # as before E6
    assert pair_roles(legacy, "highd", ["highd"], None, ["ngsim_fn"], of) == expected
    assert pair_roles(legacy, "highd", None, [], ["ngsim_fn"], of)["highd_val"] == "shift"  # explicit: no placebo
    pool = {**legacy, "ngsim_us101": _w("ngsim_us101", ("ngsim_us101",))}
    pool["pool_test"] = _w("pool_test", ("highd", "ngsim_us101"), ("highd", "ngsim_us101"))
    lodo = pair_roles(pool, "pool_test", ["highd", "ngsim_us101"], None, ["ngsim_fn"], of)  # NGSIM I-80 held out
    assert lodo == {**expected, "highd": None, "ngsim_us101": None, "pool_test": None}
    pool["pool_test"] = _w("pool_test", ("highd", "ngsim", "ngsim_us101"), ("highd", "ngsim", "ngsim_us101"))
    full = pair_roles(pool, "pool_test", ["highd", "ngsim", "ngsim_us101"], None, ["ngsim_fn"], of)
    assert full == {"highd": None, "highd_val": "placebo", "ngsim": None, "ngsim_fn": None, "openacc": "shift",
                    "ngsim_us101": None, "pool_test": None}  # fmt: skip
    # the training data default to the in-domain sources; another training set than the in-domain one: no pair
    assert pair_roles(pool, "highd", ["highd", "ngsim", "ngsim_us101"], None, ["ngsim_fn"], of)["ngsim"] is None


def test_analyses_run_on_a_pooled_root(pooled_run: DictConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = Path(pooled_run.root)
    config_file = tmp_path / "e6_pool.yaml"
    OmegaConf.save(pooled_run, config_file)
    args = ["--root", str(root), "--config", str(config_file), "--in-domain", "pool_test", "--n-boot", "40", "--quiet"]
    assert _load_script("analyze_e1").main(args) == 0
    t = root / "tables"
    info = json.loads((t / "analysis_info.json").read_text(encoding="utf-8"))
    assert info["in_domain"] == "pool_test" and info["train_data"] == POOL and info["in_domain_members"] == POOL
    assert info["placebo"] == ["highd_val"] and info["shift"] == ["openacc"] and info["cross_check"] == []
    assert info["no_pair"] == ["highd", "ngsim", "ngsim_fn"]  # parts of the pool; ngsim_fn repeats a training set
    assert info["config"]["grid"]["s"]["step"] == 5.0 and "root" not in info["config"]  # the maps config of the experiment
    ver = pd.read_csv(t / "h31_verdicts.csv")
    assert set(ver["dataset"]) == {"highd_val", "openacc"} and set(ver["pair"]) == {"placebo", "shift"}
    assert set(ver["kind"]) == {"seed_ensemble"}  # two seeds; neither model has an epistemic output of its own
    shift = pd.read_csv(t / "shift_ratios.csv")
    assert set(shift["dataset"]) == {"highd_val", "openacc"}
    one = pd.read_csv(t / "one_step_metrics.csv")
    assert {"pool_test", "highd", "ngsim", "openacc", "ngsim_fn", "highd_val"} <= set(one["dataset"])
    row = one[(one["model"] == "lstm_const_sigma") & (one["dataset"] == "pool_test") & (one["subset"] == "all") & (one["seed"].astype(str) == "0")]
    joined = pd.concat([tables.load_preds(root, "lstm_const_sigma", 0, m) for m in POOL], ignore_index=True)
    assert float(row["nll"].iloc[0]) == pytest.approx(float(joined["nll"].astype(np.float64).mean()), rel=1e-9)
    rare = pd.read_csv(t / "rare_concentration.csv")
    assert "pool_test" in set(rare["dataset"])
    h31e = pd.read_csv(t / "h31e_verdicts.csv")
    assert set(h31e["dataset"]) <= {"highd_val", "openacc"}
    assert info["v0"] == pytest.approx(json.loads((root / "idm.json").read_text(encoding="utf-8"))["params"]["v0"])
    # the same root analysed without the experiment config: the training data come from train_density.npz; --v0
    # replaces the free speed of the pooled IDM calibration (regimes defined as in E1-E5)
    args = ["--root", str(root), "--in-domain", "pool_test", "--n-boot", "40", "--quiet", "--v0", "29.5"]
    assert _load_script("analyze_e1").main(args) == 0
    again = json.loads((t / "analysis_info.json").read_text(encoding="utf-8"))
    assert again["no_pair"] == ["highd", "ngsim", "ngsim_fn"] and again["v0"] == 29.5
    # the supplementary analysis, the summary and the figures
    _load_script("analyze_e1_extra").main(["--root", str(root), "--config", str(config_file), "--n-boot", "40"])
    extra = json.loads((t / "extra_info.json").read_text(encoding="utf-8"))
    assert extra["in_domain"] == "pool_test" and extra["n_train_windows"] == 2 * CAP  # pooled windows, stride 1, capped
    assert "pool_test" in set(pd.read_csv(t / "extra_signal.csv")["dataset"])
    geo = pd.read_csv(t / "extra_shift_geometric.csv")
    assert "pool_test" not in set(geo["dataset"]) and "openacc" in set(geo["dataset"])
    capsys.readouterr()
    _load_script("summarize_e1").main(["--root", str(root)])
    summary = capsys.readouterr().out
    assert "pooled domain of the training folds of highd, ngsim" in summary and "`pool_test`; shift pairs: openacc" in summary
    assert "| model | pool_test | highd |" in summary
    _load_script("make_figures_e1").main(["--root", str(root)])
    figures = root / "figures"
    for name in ("shift_ratios", "shift_ratios_matched", "shift_ratios_controls", "nll_by_regime"):
        assert (figures / f"{name}.png").exists(), name
    assert (figures / "maps" / "lstm_const_sigma__seed_ensemble.png").exists()


# ------------------------------------------------------------------------------------ configs
def test_e6_configs() -> None:
    names = {"e6_pool": ["highd", "ngsim", "ngsim_us101", "waymo", "openacc"],
             "e6_lodo_ngsim": ["highd", "ngsim_us101", "waymo", "openacc"],
             "e6_lodo_openacc": ["highd", "ngsim", "ngsim_us101", "waymo"]}  # fmt: skip
    trimmed = {}
    for name, pool in names.items():
        assert OmegaConf.load(EXP / f"{name}.yaml").base == "e1m.yaml"
        cfg = load_config(EXP / f"{name}.yaml")
        assert cfg.root == f"artifacts/{name}" and train_datasets(cfg) == pool and cfg.signal == "matched"
        assert cfg.match.file == "artifacts/e1m/matching.json" and cfg.harmonise.window == 11
        assert pool_key(cfg) == {"data": pool, "stride": 2, "seq_stride": 5, "max_windows_per_dataset": 150000,
                                 "max_sequences_per_dataset": 40000, "subsample_seed": 0}  # fmt: skip
        assert not cfg.get("reuse_checkpoints") and cfg.model_overrides.quantile_pidl.w_phys == 0.001
        assert cfg.train.idm_from == "artifacts/e1m/idm.json"  # the frozen IDM prior of e1m, no calibration of its own
        assert list(cfg.models) == list(load_config(EXP / "e1m.yaml").models) and list(cfg.seeds) == [0, 1, 2, 3, 4]
        assert cfg.train.target == "a"
        ds = cfg.eval.datasets
        assert list(ds.pool_test.pool) == pool
        for d in ("highd", "ngsim", "ngsim_us101", "waymo", "openacc"):
            assert ds[d].data == d and ds[d].split == ("test" if d in pool else "all")
            assert ds[d].stride == load_config(EXP / "e1m.yaml").eval.datasets[d].stride  # the strides of E1
        assert ds.ngsim_fn.split == "all" and ds.highd_val.split == "val"
        assert load_config(EXP / f"{name}.yaml", pilot=True).root == f"artifacts/{name}_pilot"
        c = OmegaConf.to_container(cfg)
        c.pop("root")
        c["train"].pop("data")
        for d in ("ngsim", "openacc", "pool_test"):
            c["eval"]["datasets"].pop(d)
        trimmed[name] = c
    assert trimmed["e6_pool"] == trimmed["e6_lodo_ngsim"] == trimmed["e6_lodo_openacc"]  # only the domain differs
    e1m = OmegaConf.to_container(load_config(EXP / "e1m.yaml"))
    assert e1m["reuse_checkpoints"] and e1m["train"]["data"] == "highd"  # e1m.yaml unchanged by the inheritance
    rolled = {"e2_pool": ("e6_pool", {"highd_val": "val", "highd": "test", "ngsim": "test", "ngsim_us101": "test",
                                      "waymo": "test", "openacc": "test"}),
              "e2_lodo_ngsim": ("e6_lodo_ngsim", {"highd_val": "val", "highd": "test", "ngsim": "all"}),
              "e2_lodo_openacc": ("e6_lodo_openacc", {"highd_val": "val", "highd": "test", "openacc": "all"})}  # fmt: skip
    e2m = OmegaConf.to_container(e2_mod.load_config(EXP / "e2m.yaml"))
    for name, (e1_name, splits) in rolled.items():
        assert OmegaConf.load(EXP / f"{name}.yaml").base == "e2m.yaml"
        cfg = e2_mod.load_config(EXP / f"{name}.yaml")
        assert cfg.root == f"artifacts/{name}" and list(cfg.train.data) == names[e1_name]
        assert OmegaConf.to_container(cfg.targets) == {"a": {"checkpoints": f"artifacts/{e1_name}/checkpoints", "hold_steps": 1}}
        assert {k: v.split for k, v in cfg.datasets.items()} == splits and {v.n_events for v in cfg.datasets.values()} == {500}
        assert OmegaConf.to_container(cfg.e1_roots) == {"a": f"artifacts/{e1_name}"}
        c = OmegaConf.to_container(cfg)
        for key in ("root", "train", "targets", "datasets", "e1_roots"):
            c.pop(key)
            e2m_trim = {k: v for k, v in e2m.items() if k not in ("root", "train", "targets", "datasets", "e1_roots")}
        assert c == e2m_trim  # everything else is inherited from e2m.yaml
        pilot = e2_mod.load_config(EXP / f"{name}.yaml", pilot=True)
        assert list(pilot.targets) == ["a"] and pilot.root == f"artifacts/{name}_pilot"
