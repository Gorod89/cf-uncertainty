"""Roughness-matched signals (docs/design.md, section 15.1, experiment E5): the short-event rule, the window choice of
scripts/match_roughness.py, the per-data-set signal keys, the reuse of harmonised checkpoints by the E1 and E2 runners,
their refusal without a matching file, and the E5 configs. Synthetic events, CPU only, a few seconds."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from cf_unc.constants import DT
from cf_unc.data.harmonise import (
    DROP_REASONS,
    HarmoniseConfig,
    MatchConfig,
    MatchingError,
    choose_window,
    harmonise_event,
    harmonise_events,
    matched_windows,
)
from cf_unc.data.pipeline import load_events
from cf_unc.data.schema import Event, EventSet
from cf_unc.data.synthetic import generate
from cf_unc.experiments import e2 as e2_mod
from cf_unc.experiments.e1 import (
    Data,
    checkpoint_dir,
    data_signal,
    dry_run_e1,
    load_config,
    model_spec,
    run_e1,
    signal_key,
)
from cf_unc.maps import io as tables

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "configs" / "experiment"
GRID = [11, 15, 21, 25, 31, 41, 51, 61, 81, 101]
SETTINGS = HarmoniseConfig().to_dict()
KEY = {"signal": "harmonised", "harmonise": SETTINGS}  # the key of the harmonised variant (section 12)
N_FOLDS = 5
QUICK = {"maxiter": 3, "popsize": 5, "polish": False}  # tiny differential-evolution budget
DATASETS: dict[str, dict[str, Any]] = {
    "highd": {"data": "hd", "split": "test", "stride": 2},
    "ngsim": {"data": "ng", "split": "all", "stride": 3},
}
MODELS = ("idm_white_noise", "lstm_const_sigma")


def _load_script(name: str) -> Any:
    """Module of ``scripts/<name>.py`` (the scripts directory is not a package)."""
    sys.path.insert(0, str(REPO / "scripts"))  # match_roughness imports harmonise_report as a sibling
    spec = importlib.util.spec_from_file_location(f"{name}_script", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def matcher() -> Any:
    return _load_script("match_roughness")


# ------------------------------------------------------------------------------------ events
def _event(event_id: str, n: int = 300, sigma: float = 0.0, seed: int = 0) -> Event:
    """Smooth speeds (sine, period 40 s) with white noise of sd ``sigma`` [m/s] on the recorded follower speed; the gap
    is kinematically consistent with the recorded speeds (as in the reconstructions)."""
    t = np.arange(n) * DT
    v = 15.0 + 2.0 * np.sin(2 * np.pi * t / 40.0) + sigma * np.random.default_rng(seed).standard_normal(n)
    v_lead = 15.0 + 2.0 * np.sin(2 * np.pi * t / 40.0 + 0.6)
    s = 20.0 + DT * np.concatenate([[0.0], np.cumsum((v_lead - v)[1:])])
    x_f = 100.0 + DT * np.concatenate([[0.0], np.cumsum(v[1:])])
    return Event(
        event_id=event_id, dataset="syn", site="s", follower_id=event_id, leader_id=f"L{event_id}", t=t, s=s,
        dv=v - v_lead, v=v, a=np.gradient(v, DT), v_lead=v_lead, x_lead=x_f + s, x_follower=x_f, meta={"a_source": "savgol"},
    )  # fmt: skip


def _set(prefix: str, sigma: float, k: int = 12, n: int = 300, short: int = 0, short_n: int = 30) -> EventSet:
    """``k`` events of ``n`` samples with speed noise ``sigma``, plus ``short`` events of ``short_n`` samples."""
    events = [_event(f"{prefix}{i}", n, sigma, seed=i) for i in range(k)]
    events += [_event(f"{prefix}short{i}", short_n, sigma, seed=100 + i) for i in range(short)]
    return EventSet(events)


# ------------------------------------------------------------------------------------ short events
def test_short_events_are_dropped_unprocessed() -> None:
    events = [_event("long", 300, 0.1), _event("edge", 11, 0.1), _event("short", 10, 0.1), _event("tiny", 3)]
    kept, report = harmonise_events(events)  # window 11
    assert [ev.event_id for ev in kept] == ["long", "edge"]  # 11 samples fit the window, 10 do not
    assert DROP_REASONS[-1] == "short_event" and list(report["dropped"]) == list(DROP_REASONS)
    assert report["dropped"]["short_event"] == 2 == report["violations"]["short_event"] and report["n_dropped"] == 2
    assert sum(report["violations"].values()) == 2  # a short event violates nothing else (it is not processed)
    out, info = harmonise_event(events[2])
    assert out is None and info["dropped"] == "short_event" and info["violations"] == ["short_event"]
    assert np.isnan(info["gap_rms"]) and info["speed_floored"] == 0
    wide, report = harmonise_events(events, {"window": 41})
    assert [ev.event_id for ev in wide] == ["long"] and report["dropped"]["short_event"] == 3
    assert report["share_dropped"] == pytest.approx(3 / 4)


def test_choose_window_rule_cap_and_flags() -> None:
    rough = {11: 9.0, 15: 6.0, 21: 3.0, 25: 1.08, 31: 0.9, 41: 0.8}
    none = {w: 0.0 for w in rough}
    grid = sorted(rough)
    assert choose_window(rough, none, 1.1, grid) == {"window": 25, "window_rule": 25, "cap": None, "flags": []}
    assert choose_window(rough, none, 1.0, grid)["window"] == 31  # the smallest window within the target
    assert choose_window(rough, none, 20.0, grid)["window"] == 11  # smoother than the reference: the smallest window
    assert choose_window(rough, none, 0.5, grid) == {"window": 41, "window_rule": 41, "cap": None, "flags": ["not_reached"]}
    assert choose_window({**rough, 25: np.nan}, none, 1.1, grid)["window"] == 31  # NaN (nothing kept) never matches
    short = {11: 0.0, 15: 0.0, 21: 0.04, 25: 0.07, 31: 0.07, 41: 0.2}  # >= 5 % of the events shorter than 25
    assert choose_window(rough, short, 1.1, grid) == {"window": 21, "window_rule": 25, "cap": 21, "flags": ["capped"]}
    assert choose_window(rough, short, 0.5, grid) == {"window": 21, "window_rule": 41, "cap": 21,
                                                      "flags": ["not_reached", "capped"]}  # fmt: skip
    assert choose_window(rough, {w: 0.5 for w in grid}, 1.1, grid)["flags"] == ["cap_unreachable"]
    assert choose_window(rough, {**none, 25: 0.0499}, 1.1, grid)["window"] == 25  # below the 5 % mark: no cap


def test_window_choice_on_a_rough_and_a_smooth_set(matcher: Any) -> None:
    # speed noise of sd 0.01 m/s (reference), 0.05 m/s (rough: about 5x the reference roughness with window 11, 0.9x
    # with 51), 0.3 m/s (too rough for the grid: 2.5x with window 101) and none (smoother than the reference)
    conf, match = HarmoniseConfig(), MatchConfig(reference="ref", grid=tuple(GRID), tolerance=0.1)
    sets = {"ref": _set("r", 0.01), "rough": _set("x", 0.05), "very_rough": _set("y", 0.3), "smooth": _set("s", 0.0)}
    stats = {name: matcher.window_stats(events, conf, GRID) for name, events in sets.items()}
    content = matcher.match_windows(stats, conf, match)
    reference = stats["ref"][11]["roughness"]["v"]
    assert content["reference_roughness"] == reference and content["target_roughness"] == pytest.approx(1.1 * reference)
    assert content["windows"]["ref"] == 11 and content["datasets"]["ref"]["role"] == "reference"
    ratio = {w: stats["rough"][w]["roughness"]["v"] / reference for w in GRID}
    assert ratio[11] > 3 and all(ratio[a] > ratio[b] for a, b in zip(GRID, GRID[1:]))  # smoothing lowers the roughness
    chosen = content["windows"]["rough"]
    assert chosen > 11 and ratio[chosen] <= 1.1 and all(ratio[w] > 1.1 for w in GRID if w < chosen)
    entry = content["datasets"]["rough"]
    assert entry["flags"] == [] and entry["ratio"] == pytest.approx(ratio[chosen]) and entry["roughness_harmonised"] > 3 * reference
    assert set(entry["per_window"]) == {str(w) for w in GRID} and entry["short_event_share"] == 0.0
    very = content["datasets"]["very_rough"]  # no grid window reaches the target: the largest, flagged
    assert very["window"] == 101 and very["flags"] == ["not_reached"] and very["ratio"] > 1.1
    assert content["windows"]["smooth"] == 11  # already smoother than the reference: the smallest grid window
    # the roughness is the v entry of harmonise_report.roughness of the events kept with the window
    kept, _ = harmonise_events(sets["rough"], {"window": chosen})
    assert entry["roughness"] == pytest.approx(matcher.roughness(kept)["v"], rel=1e-12)


def test_short_event_share_caps_the_window(matcher: Any) -> None:
    conf, match = HarmoniseConfig(), MatchConfig(reference="ref", grid=tuple(GRID), tolerance=0.1)
    rough = _set("x", 0.05, k=18, short=2, short_n=30)  # 10 % of the events have 30 samples
    stats = {"ref": matcher.window_stats(_set("r", 0.01), conf, GRID), "rough": matcher.window_stats(rough, conf, GRID)}
    per = stats["rough"]
    assert [per[w]["short_event_share"] for w in GRID] == [0.0] * 4 + [0.1] * 6  # events shorter than 31, 41, ...
    assert per[31]["dropped"]["short_event"] == 2 and per[25]["dropped"]["short_event"] == 0
    assert per[41]["n_out"] <= 18 and per[25]["n_out"] <= 20
    entry = matcher.match_windows(stats, conf, match)["datasets"]["rough"]
    assert entry["window_rule"] >= 31  # far too rough below: the rule wants a window that drops the short events
    assert entry["window"] == entry["cap"] == 25 and entry["flags"] == ["capped"] and entry["short_event_share"] == 0.0


# ------------------------------------------------------------------------------------ keys
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
    root = tmp_path_factory.mktemp("matched_events")
    return {
        "hd": _write_set(root, "hd", seed=1, n_drivers=25, per=2, duration=12.0),
        "ng": _write_set(root, "ng", seed=2, n_drivers=8, per=2, duration=15.0),
    }


def write_matching(path: Path, windows: dict[str, int], reference: str = "hd", **settings: Any) -> Path:
    """A matching file with the settings the runners check (as scripts/match_roughness.py writes them)."""
    match = MatchConfig(reference=reference, grid=tuple(GRID), tolerance=0.1)
    content = {"version": 1, "harmonise": HarmoniseConfig.from_config(settings or None).to_dict(), **match.settings(),
               "windows": windows, "datasets": {}}  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def make_config(root: Path, data_configs: dict, signal: str | None, file: Path | None = None, models=MODELS,
                **extra: Any) -> DictConfig:  # fmt: skip
    cfg = load_config()
    cfg.root = str(root)
    cfg.device = "cpu"
    cfg.models = list(models)
    cfg.seeds = [0]
    cfg.train = {"data": "hd", "stride": 2, "seq_stride": 5, "overrides": {"max_epochs": 1, "batch_size": 256}}
    cfg.eval.datasets = OmegaConf.create(DATASETS)
    cfg.eval.chunk_rows = 150
    cfg.data_configs = data_configs
    cfg.idm = {"closed_loop": {"objective": "spacing", "max_events": 10, "seed": 0, **QUICK}}
    cfg.idm_reference = None
    if signal is not None:
        cfg.signal = signal
    if signal == "matched":
        cfg.match = {"reference": "hd", "grid": GRID, "tolerance": 0.1, "file": None if file is None else str(file)}
    for key, value in extra.items():
        cfg[key] = value
    return cfg


def test_per_data_set_signal_keys(data_configs: dict, tmp_path: Path) -> None:
    file = write_matching(tmp_path / "matching.json", {"hd": 11, "ng": 21})
    matched = make_config(tmp_path / "m", data_configs, "matched", file)
    harmonised = make_config(tmp_path / "h", data_configs, "harmonised")
    assert data_signal(matched, "hd") == data_signal(matched, "ng") == "matched"
    assert signal_key(matched, "hd") == signal_key(harmonised, "hd") == KEY  # the reference: the harmonised key
    assert signal_key(matched, "ng") == {"signal": "harmonised", "harmonise": {**SETTINGS, "window": 21}} != KEY
    exp = signal_key(matched)
    assert exp["signal"] == "matched" and exp["harmonise"] == SETTINGS
    assert exp["match"] == {"reference": "hd", "grid": GRID, "tolerance": 0.1, "max_short_share": 0.05, "windows": {"hd": 11, "ng": 21}}
    assert model_spec(matched, "lstm_const_sigma").signal == KEY  # checkpoints: the key of the training data set
    a, b = Data(matched), Data(harmonised)
    assert a.eval_spec("highd") == b.eval_spec("highd") and a.eval_spec("ngsim") != b.eval_spec("ngsim")
    assert a.eval_spec("ngsim")["harmonise"]["window"] == 21
    ng = a.events("ng")  # harmonised with the window 21: exactly the harmonised variant with that window
    expected, _ = harmonise_events(load_events(data_configs["ng"]["path"]), {"window": 21})
    assert [e.event_id for e in ng] == [e.event_id for e in expected]
    np.testing.assert_array_equal(np.concatenate([e.v for e in ng]), np.concatenate([e.v for e in expected]))
    assert a.match_info("hd")["window"] == 11 and a.match_info("ng")["window"] == 21 and b.match_info("ng") is None
    # a stale, incomplete or inconsistent file is refused
    with pytest.raises(MatchingError, match="was computed with"):
        signal_key(OmegaConf.merge(matched, {"match": {"tolerance": 0.2}}), "ng")
    with pytest.raises(MatchingError, match="no matched window"):
        signal_key(matched, "oa")
    with pytest.raises(MatchingError, match="must keep the window"):
        matched_windows(write_matching(tmp_path / "bad.json", {"hd": 15, "ng": 21}), None, matched.match)
    with pytest.raises(MatchingError, match="outside the grid"):
        matched_windows(write_matching(tmp_path / "odd.json", {"hd": 11, "ng": 23}), None, matched.match)
    with pytest.raises(ValueError, match="experiment-level"):  # matched needs the experiment's match settings
        data_signal(make_config(tmp_path / "x", {**data_configs, "hd": {**data_configs["hd"], "signal": "matched"}}, None), "hd")
    with pytest.raises(ValueError, match="matched"):
        load_events(data_configs["ng"]["path"], signal="matched")


# ------------------------------------------------------------------------------------ runners
def _files(root: Path) -> dict[str, int]:
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def harmonised_root(tmp_path_factory: pytest.TempPathFactory, data_configs: dict) -> DictConfig:
    """A harmonised E1 root (the stand-in of artifacts/e1h): checkpoints and tables of two models."""
    cfg = make_config(tmp_path_factory.mktemp("e1h") / "e1h", data_configs, "harmonised")
    status = run_e1(cfg, echo=False)
    assert status["failures"] == [] and status["checkpoints"]["trained_now"] == 2
    return cfg


def test_matched_run_reuses_harmonised_checkpoints(harmonised_root: DictConfig, data_configs: dict, tmp_path: Path) -> None:
    source = Path(harmonised_root.root)
    file = write_matching(tmp_path / "matching.json", {"hd": 11, "ng": 21})
    cfg = make_config(tmp_path / "e1m", data_configs, "matched", file,
                      reuse_checkpoints={m: str(source / "checkpoints") for m in MODELS})  # fmt: skip
    plan = dry_run_e1(cfg, echo=False)  # the reuse decisions, before anything exists in the root
    assert plan["conflicts"] == [] and plan["signal"] == "matched" and plan["matched_windows"] == {"hd": 11, "ng": 21}
    assert plan["checkpoints"]["reuse"] == ["idm_white_noise/seed0", "lstm_const_sigma/seed0"] and plan["checkpoints"]["missing"] == []
    assert [d["model"] for d in plan["reuse"]] == list(MODELS) and "same key" in plan["reuse"][0]["idm"]
    assert plan["reuse"][1]["idm"] is None and len(plan["tables"]["missing"]) == 4 and not (tmp_path / "e1m").exists()
    status = run_e1(cfg, echo=False)
    root = Path(cfg.root)
    assert status["failures"] == [] and status["signal"] == "matched"
    assert status["trained"] == ["idm_white_noise/seed0 (reused)", "lstm_const_sigma/seed0 (reused)"]  # nothing trained
    for model in MODELS:
        log = json.loads((checkpoint_dir(root, model, 0) / "train_log.json").read_text(encoding="utf-8"))
        assert log["signal"] == KEY and "reused_from" in log
    _, meta = tables.load_windows(root, "highd")
    _, ref_meta = tables.load_windows(source, "highd")
    assert meta["spec"] == ref_meta["spec"] and meta["event_ids"] == ref_meta["event_ids"]  # in-domain: the same windows
    assert meta["match"]["window"] == 11 and meta["match"]["file"].endswith("matching.json")
    _, ng_meta = tables.load_windows(root, "ngsim")
    assert ng_meta["spec"]["harmonise"]["window"] == 21 and ng_meta["harmonisation"]["settings"]["window"] == 21
    for model in MODELS:  # the reused checkpoints predict the in-domain tables of the harmonised root exactly
        pd.testing.assert_frame_equal(tables.load_preds(root, model, 0, "highd"), tables.load_preds(source, model, 0, "highd"))
        assert not tables.load_preds(root, model, 0, "ngsim").equals(tables.load_preds(source, model, 0, "ngsim"))
    with np.load(root / "train_density.npz") as z:
        assert json.loads(str(z["signal"])) == KEY
    plan = dry_run_e1(cfg, echo=False)
    assert plan["nothing_to_do"] and plan["reuse"] == []
    # a harmonised config with another window for the training data is refused by the same checks
    other = make_config(tmp_path / "w15", data_configs, "harmonised", harmonise={**SETTINGS, "window": 15},
                        reuse_checkpoints={"lstm_const_sigma": str(source / "checkpoints")}, models=("lstm_const_sigma",))  # fmt: skip
    plan = dry_run_e1(other, echo=False)
    assert [c["unit"] for c in plan["conflicts"]] == ["lstm_const_sigma/seed0 (reuse)"] and "window 11" in plan["conflicts"][0]["error"]


def test_runners_refuse_without_matching_file(data_configs: dict, tmp_path: Path, harmonised_root: DictConfig) -> None:
    missing = make_config(tmp_path / "e1m", data_configs, "matched", tmp_path / "none" / "matching.json")
    for fn in (run_e1, dry_run_e1):
        with pytest.raises(MatchingError, match="does not exist"):
            fn(missing, echo=False)
    with pytest.raises(MatchingError, match="match.file"):
        run_e1(make_config(tmp_path / "e1m", data_configs, "matched", None), echo=False)
    assert not (tmp_path / "e1m").exists()  # nothing was created
    config_file = tmp_path / "e1m.yaml"
    OmegaConf.save(missing, config_file)
    assert _load_script("run_e1").main(["--config", str(config_file), "--quiet"]) == 2
    assert _load_script("run_e1").main(["--config", str(config_file), "--dry-run", "--quiet"]) == 2
    incomplete = make_config(tmp_path / "e1m", data_configs, "matched", write_matching(tmp_path / "hd_only.json", {"hd": 11}))
    with pytest.raises(MatchingError, match="'ng' has no matched window"):
        run_e1(incomplete, echo=False)
    e2 = _e2_config(tmp_path / "e2m", harmonised_root, data_configs, tmp_path / "none" / "matching.json")
    with pytest.raises(MatchingError, match="does not exist"):
        e2_mod.run_e2(e2, echo=False)
    e2.train = None
    with pytest.raises(MatchingError, match="train.data"):
        e2_mod.run_e2(e2, echo=False)
    assert not (tmp_path / "e2m").exists() and not (tmp_path / "e1m").exists()


def _e2_config(root: Path, e1: DictConfig, data_configs: dict, file: Path) -> DictConfig:
    cfg = e2_mod.load_config()
    cfg.root = str(root)
    cfg.device = "cpu"
    cfg.signal = "matched"
    cfg.match = {"reference": "hd", "grid": GRID, "tolerance": 0.1, "file": str(file)}
    cfg.train = {"data": "hd"}
    ckpt = str(Path(e1.root) / "checkpoints")
    cfg.targets = OmegaConf.create({"a": {"checkpoints": ckpt, "hold_steps": 1},
                                    "mean_1s": {"checkpoints": str(root.parent / "missing"), "hold_steps": 10}})  # fmt: skip
    cfg.models = ["lstm_const_sigma"]
    cfg.seeds = [0]
    cfg.variants = OmegaConf.create({})
    cfg.datasets = OmegaConf.create({"highd": {"data": "hd", "split": "test", "n_events": 4},
                                     "ngsim": {"data": "ng", "split": "all", "n_events": 4}})  # fmt: skip
    cfg.rollout = OmegaConf.create({"n_rollouts": 6, "n_rollouts_sampled": 4, "horizon_s": 2.0, "horizons_s": [1, 2],
                                    "max_batch": 100, "stream_events": 3, "event_seed": 0, "seed": 0})  # fmt: skip
    cfg.data_configs = data_configs
    return cfg


def test_e2_checks_the_key_of_the_training_data_set(harmonised_root: DictConfig, data_configs: dict, tmp_path: Path) -> None:
    file = write_matching(tmp_path / "matching.json", {"hd": 11, "ng": 21})
    cfg = _e2_config(tmp_path / "e2m", harmonised_root, data_configs, file)
    status = e2_mod.run_e2(cfg, echo=False)
    root = Path(cfg.root)
    assert status["failures"] == [] and status["signal"] == "matched" and status["targets"]["a"] == {"done": 2, "total": 2}
    spec = json.loads((root / "a" / "spec.json").read_text(encoding="utf-8"))
    assert spec["signal"]["signal"] == "matched" and spec["signal"]["match"]["windows"] == {"hd": 11, "ng": 21}
    runner = e2_mod.E2Runner(cfg, echo=False)
    path = runner.checkpoint("a", "lstm_const_sigma", 0)
    stored = json.loads((path.parent / "train_log.json").read_text(encoding="utf-8"))["signal"]
    assert runner.train_signal_key() == stored == KEY != runner.signal_key()  # the experiment-level key would refuse it
    runner.check_signal(path)
    raw = e2_mod.E2Runner(OmegaConf.merge(cfg, {"signal": None, "root": str(tmp_path / "raw")}), echo=False)
    with pytest.raises(ValueError, match="for the training data set hd"):
        raw.check_signal(path)
    before = _files(root)
    other = write_matching(tmp_path / "other.json", {"hd": 11, "ng": 25})
    status = e2_mod.run_e2(OmegaConf.merge(cfg, {"match": {"file": str(other)}}), echo=False)  # other windows: refused
    assert [f["unit"] for f in status["failures"]] == ["a/spec"]
    assert {k: v for k, v in _files(root).items() if "logs" not in k and k not in ("status.json", "config.yaml")} == {
        k: v for k, v in before.items() if "logs" not in k and k not in ("status.json", "config.yaml")}  # fmt: skip


# ------------------------------------------------------------------------------------ script
def test_match_roughness_script_writes_a_file_the_runner_accepts(data_configs: dict, tmp_path: Path, matcher: Any,
                                                                 capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    file = tmp_path / "e1m" / "matching.json"
    cfg = make_config(tmp_path / "e1m", data_configs, "matched", file)
    config_file = tmp_path / "e1m.yaml"
    OmegaConf.save(cfg, config_file)
    assert matcher.main(["--config", str(config_file)]) == 0
    printed = capsys.readouterr().out
    assert "| data set | role |" in printed and "persistence RMSE harm. -> matched" in printed
    content = json.loads(file.read_text(encoding="utf-8"))
    assert content["reference"] == "hd" and content["windows"]["hd"] == 11 and set(content["windows"]) == {"hd", "ng"}
    assert content["harmonise"] == SETTINGS and content["grid"] == GRID and content["tolerance"] == 0.1
    for name, entry in content["datasets"].items():
        assert set(entry["per_window"]) == {str(w) for w in GRID} and entry["source"] == str(Path(data_configs[name]["path"]))
        assert {"short_event", "min_gap"} <= set(entry["dropped"]) and entry["role"] == ("reference" if name == "hd" else "matched")
    assert content["datasets"]["hd"]["flags"] == [] and content["datasets"]["hd"]["window"] == 11
    signal = pd.DataFrame(content["predictability"])
    assert set(signal["signal"]) == {"harmonised", "matched"} and set(signal["target"]) == {"a", "mean_1s"}
    ref = signal[signal["dataset"] == "highd"].set_index(["target", "signal"])["rmse_persistence"]
    assert ref[("a", "harmonised")] == ref[("a", "matched")]  # the reference keeps its window
    assert Data(cfg).signal_key("ng")["harmonise"]["window"] == content["windows"]["ng"]  # read by the runners
    # the same config again: the same windows, the file may be replaced; other windows need --force
    assert matcher.main(["--config", str(config_file), "--no-predictability"]) == 0
    content["windows"]["ng"] = 101 if content["windows"]["ng"] != 101 else 81
    file.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(SystemExit, match="force"):
        matcher.main(["--config", str(config_file), "--no-predictability"])
    assert matcher.main(["--config", str(config_file), "--no-predictability", "--force"]) == 0
    with pytest.raises(SystemExit, match="signal: matched"):
        harmonised = tmp_path / "h.yaml"
        OmegaConf.save(make_config(tmp_path / "h", data_configs, "harmonised"), harmonised)
        matcher.main(["--config", str(harmonised)])


# ------------------------------------------------------------------------------------ configs
def test_e5_configs_differ_from_the_harmonised_ones_only_where_intended() -> None:
    models = [str(m) for m in load_config(EXP / "e1h.yaml").models]
    match = {"reference": "highd", "grid": GRID, "tolerance": 0.1, "file": "artifacts/e1m/matching.json"}
    for name, base, sources in (("e1m", "e1h", {m: "artifacts/e1h/checkpoints" for m in models}),
                                ("e1m_mean1s", "e1h_mean1s", {**{m: "artifacts/e1h_mean1s/checkpoints" for m in models},
                                                              "diffusion_cf": "artifacts/e1h/checkpoints"})):  # fmt: skip
        child = OmegaConf.to_container(load_config(EXP / f"{name}.yaml"))
        parent = OmegaConf.to_container(load_config(EXP / f"{base}.yaml"))
        assert child["root"] == f"artifacts/{name}" and child["signal"] == "matched" and child["match"] == match
        assert child["harmonise"] == SETTINGS and child["reuse_checkpoints"] == sources  # every model, none trained
        for cfg in (child, parent):
            for key in ("root", "signal", "match", "reuse_checkpoints"):
                cfg.pop(key, None)
        assert child == parent
        pilot = load_config(EXP / f"{name}.yaml", pilot=True)
        assert pilot.root == f"artifacts/{name}_pilot" and pilot.match.file == match["file"]  # one matching for E5
    e2m = OmegaConf.to_container(e2_mod.load_config(EXP / "e2m.yaml"))
    e2h = OmegaConf.to_container(e2_mod.load_config(EXP / "e2h.yaml"))
    assert e2m["root"] == "artifacts/e2m" and e2m["signal"] == "matched" and e2m["match"] == match
    assert e2m["train"] == {"data": "highd"} == e2h["train"] and e2m["targets"] == e2h["targets"]  # the e1h checkpoints
    assert e2m["e1_roots"] == {"a": "artifacts/e1m", "mean_1s": "artifacts/e1m_mean1s"}
    for cfg in (e2m, e2h):
        for key in ("root", "signal", "match", "e1_roots"):
            cfg.pop(key, None)
    assert e2m == e2h
