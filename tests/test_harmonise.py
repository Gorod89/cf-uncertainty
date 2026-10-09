"""Harmonised signals (docs/design.md, section 12): the transform, the loader, the resume keys of the E1 runner and
the experiment configs. Synthetic events, CPU only, a few seconds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from cf_unc.constants import DT
from cf_unc.data.harmonise import DROP_REASONS, HarmoniseConfig, check_signal, harmonise_event, harmonise_events
from cf_unc.data.pipeline import acceleration_from_speed, central_differences, load_events, savgol_speed
from cf_unc.data.schema import Event, EventSet
from cf_unc.data.splits import split_by_folds
from cf_unc.data.synthetic import generate
from cf_unc.data.windows import make_windows
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
N_FOLDS = 5
QUICK = {"maxiter": 3, "popsize": 5, "polish": False}  # tiny differential-evolution budget
DATASETS: dict[str, dict[str, Any]] = {
    "highd": {"data": "hd", "split": "test", "stride": 2},
    "ngsim": {"data": "ng", "split": "all", "stride": 3},
}
SETTINGS = HarmoniseConfig().to_dict()
KEY = {"signal": "harmonised", "harmonise": SETTINGS}


# ------------------------------------------------------------------------------------ events
def _displacement(v: np.ndarray, v_lead: np.ndarray) -> np.ndarray:
    """I[t] = sum_{k=1..t} (v_lead[k] - v[k]) DT (scheme of design.md section 1)."""
    return DT * np.concatenate([[0.0], np.cumsum((v_lead - v)[1:])])


def _smooth_event(
    event_id: str = "e0", n: int = 300, s0: float = 20.0, c: float = 0.0, gap_noise: float = 0.0, seed: int = 0,
    period: float = 40.0, amp: float = 2.0, x0: float = 100.0,
) -> Event:  # fmt: skip
    """Smooth event whose true gap follows the integrator exactly; the leader speed is recorded with the offset -c
    and the gap with white noise of sd ``gap_noise`` [m]."""
    t = np.arange(n) * DT
    v = 15.0 + amp * np.sin(2 * np.pi * t / period)
    v_lead_true = 15.0 + amp * np.sin(2 * np.pi * t / period + 0.6)
    s = s0 + _displacement(v, v_lead_true) + gap_noise * np.random.default_rng(seed).standard_normal(n)
    v_lead = v_lead_true - c
    x_f = x0 + DT * np.concatenate([[0.0], np.cumsum(v[1:])])
    return Event(
        event_id=event_id, dataset="syn", site="s", follower_id=event_id, leader_id=f"L{event_id}", t=t, s=s,
        dv=v - v_lead, v=v, a=np.gradient(v, DT), v_lead=v_lead, x_lead=x_f + s, x_follower=x_f,
        meta={"a_source": "dataset"},
    )  # fmt: skip


def _stop_event() -> Event:
    """Follower and leader brake hard to a standstill, wait 10 s and drive off: the filtered speed dips below 0."""
    t = np.arange(400) * DT

    def profile(shift: float) -> np.ndarray:
        tt = t - shift
        v = np.where(tt < 8.0, 10.0, np.maximum(10.0 - 5.0 * (tt - 8.0), 0.0))
        return np.where(tt > 20.0, np.minimum(1.5 * (tt - 20.0), 10.0), v)

    v, v_lead = profile(1.5), profile(0.0)
    s = 20.0 + _displacement(v, v_lead)  # the follower closes in by 15 m while the leader waits
    x_f = DT * np.concatenate([[0.0], np.cumsum(v[1:])])
    return Event(
        event_id="stop", dataset="syn", site="s", follower_id="f", leader_id="l", t=t, s=s, dv=v - v_lead, v=v,
        a=np.gradient(v, DT), v_lead=v_lead, x_lead=x_f + s, x_follower=x_f, meta={"a_source": "savgol"},
    )  # fmt: skip


# ------------------------------------------------------------------------------------ transform
def test_known_offsets_are_recovered() -> None:
    ev = _smooth_event(s0=20.0, c=0.2)
    out, info = harmonise_event(ev)
    assert out is not None and info["kept"] and info["dropped"] is None and info["violations"] == []
    assert info["gap_offset"] == pytest.approx(20.0, abs=1e-3)
    assert info["leader_speed_offset"] == pytest.approx(0.2, abs=1e-4)
    np.testing.assert_allclose(out.v_lead, ev.v_lead + 0.2, atol=1e-3)  # the offset is given back to the leader
    np.testing.assert_allclose(out.s, ev.s, atol=1e-3)  # the stored gap was consistent with the true speeds
    assert info["gap_rms"] < 1e-3 and info["kinematic_rms_before"] > 0.15 > 1e-9 > info["kinematic_rms_after"]
    meta = out.meta
    assert meta["signal"] == "harmonised" and meta["gap_offset"] == info["gap_offset"]
    assert meta["leader_speed_offset"] == info["leader_speed_offset"] and meta["gap_rms"] == info["gap_rms"]
    assert meta["a_source_original"] == "dataset" and meta["a_source"] == "savgol" and meta["speed_floored"] == 0
    assert ev.meta == {"a_source": "dataset"}  # the input event is not modified
    noisy = _smooth_event(s0=20.0, c=-0.15, gap_noise=0.3, seed=1)
    _, info = harmonise_event(noisy)
    assert info["leader_speed_offset"] == pytest.approx(-0.15, abs=0.01)
    assert info["gap_offset"] == pytest.approx(20.0, abs=0.15) and 0.25 < info["gap_rms"] < 0.35


@pytest.mark.parametrize("name", ["smooth", "offset", "noisy", "stop", "generated"])
def test_kinematic_identity_holds_exactly_and_the_event_validates(name: str) -> None:
    generated = generate(dict(n_drivers=2, events_per_driver=1, duration_s=30.0, seed=5))[0][0]  # noisy speeds
    ev = {
        "smooth": _smooth_event(),
        "offset": _smooth_event(c=0.25, x0=2.0e5),  # large positions: the identity must not suffer from rounding
        "noisy": _smooth_event(c=0.1, gap_noise=0.5, seed=3),
        "stop": _stop_event(),
        "generated": generated,
    }[name]
    out, info = harmonise_event(ev)
    assert out is not None
    out.validate()
    assert np.max(np.abs(np.diff(out.s) / DT + out.dv[1:])) < 1e-9  # s_h[t] - s_h[t-1] = -dv_h[t] DT
    assert info["kinematic_rms_after"] < 1e-9
    np.testing.assert_array_equal(out.dv, out.v - out.v_lead)
    assert np.max(np.abs(out.s - (out.x_lead - out.x_follower))) < 1e-9
    np.testing.assert_allclose(np.diff(out.x_follower), out.v[1:] * DT, rtol=0, atol=1e-8)  # positions: scheme of
    np.testing.assert_allclose(np.diff(out.x_lead), out.v_lead[1:] * DT, rtol=0, atol=1e-8)  # design.md section 1
    assert out.x_follower[0] == ev.x_follower[0] and np.array_equal(out.t, ev.t)
    np.testing.assert_allclose(out.a, central_differences(out.v), rtol=0, atol=1e-12)
    assert out.v.min() >= 0.0 and out.v_lead.min() >= 0.0
    if name == "stop":  # the filter undershoots at the standstill: floored, still consistent
        assert info["speed_floored"] > 0 and out.meta["speed_floored"] == info["speed_floored"]
    else:  # the acceleration is the definition used so far (central differences of the filtered speed)
        np.testing.assert_allclose(out.a, acceleration_from_speed(ev.v), rtol=0, atol=1e-9)


def test_speeds_of_a_smooth_event_hardly_change() -> None:
    ev = _smooth_event(c=0.0)
    out, info = harmonise_event(ev)
    assert out is not None
    assert np.max(np.abs(out.v - ev.v)) < 1e-3 and np.max(np.abs(out.v_lead - ev.v_lead)) < 1e-3
    assert np.max(np.abs(out.s - ev.s)) < 1e-3 and abs(info["leader_speed_offset"]) < 1e-5 and info["gap_rms"] < 1e-3


def test_dropped_events_are_counted_by_reason() -> None:
    probe = _smooth_event()
    low = 0.4 - float(np.min(probe.s - 20.0))  # s0 with min(s) = 0.4 m
    nan = _smooth_event("nan")
    nan.s[10] = np.nan
    events = [
        _smooth_event("ok"),
        _smooth_event("gap", s0=low),
        _smooth_event("c", c=0.5),
        _smooth_event("rms", s0=40.0, gap_noise=3.0, seed=2),
        _smooth_event("two", s0=40.0, c=0.5, gap_noise=3.0, seed=4),
        nan,
        _smooth_event("ok2", n=200, c=-0.29),
    ]
    kept, report = harmonise_events(events)
    assert [ev.event_id for ev in kept] == ["ok", "ok2"]
    assert report["n_in"] == 7 and report["n_out"] == 2 and report["n_dropped"] == 5
    assert report["share_dropped"] == pytest.approx(5 / 7)
    assert report["dropped"] == {"non_finite": 1, "min_gap": 1, "max_abs_c": 2, "max_gap_rms": 1, "short_event": 0}  # first reason
    assert report["violations"] == {"non_finite": 1, "min_gap": 1, "max_abs_c": 2, "max_gap_rms": 2, "short_event": 0}  # every reason
    assert sum(report["dropped"].values()) == report["n_dropped"] and list(report["dropped"]) == list(DROP_REASONS)
    for key in ("gap_rms_m", "abs_c_mps", "gap_offset_m"):
        assert set(report[key]) == {"q05", "q50", "q95"}
    assert report["abs_c_mps"]["q95"] > 0.4 and report["gap_rms_m"]["q95"] > 2.0
    assert report["kinematic_rms_after"] < 1e-9 < report["kinematic_rms_before_kept"] <= report["kinematic_rms_before"]
    assert report["settings"] == SETTINGS and report["speed_floored"] == {"samples": 0, "events": 0}
    infos = {ev.event_id: harmonise_event(ev)[1] for ev in events}
    assert {k: v["dropped"] for k, v in infos.items()} == {
        "ok": None, "gap": "min_gap", "c": "max_abs_c", "rms": "max_gap_rms", "two": "max_abs_c", "nan": "non_finite",
        "ok2": None,
    }  # fmt: skip
    assert infos["two"]["violations"] == ["max_abs_c", "max_gap_rms"] and infos["nan"]["violations"] == ["non_finite"]
    assert harmonise_event(events[1])[0] is None
    relaxed, report = harmonise_events(events, {"max_abs_c": 0.6, "max_gap_rms": 5.0, "min_gap": 0.3})
    assert [ev.event_id for ev in relaxed] == ["ok", "gap", "c", "rms", "two", "ok2"] and report["dropped"]["non_finite"] == 1


def test_batches_keep_the_order_and_equal_single_events() -> None:
    events = [_smooth_event(f"e{i}", n=n, c=0.05 * i, gap_noise=0.2, seed=i) for i, n in enumerate((300, 200, 300, 250))]
    kept, _ = harmonise_events(events)
    assert [ev.event_id for ev in kept] == ["e0", "e1", "e2", "e3"]
    for ev, out in zip(events, kept):
        single, _ = harmonise_event(ev)
        for name in ("s", "dv", "v", "a", "v_lead", "x_lead", "x_follower"):
            np.testing.assert_allclose(getattr(out, name), getattr(single, name), rtol=0, atol=1e-10)


def test_settings() -> None:
    assert HarmoniseConfig.from_config(None) == HarmoniseConfig() and SETTINGS == {
        "window": 11, "order": 2, "min_gap": 0.5, "max_abs_c": 0.3, "max_gap_rms": 2.0}  # fmt: skip
    assert HarmoniseConfig.from_config(OmegaConf.create({"max_abs_c": 0.2})).max_abs_c == 0.2
    with pytest.raises(ValueError, match="unknown harmonise"):
        HarmoniseConfig.from_config({"max_c": 0.2})
    with pytest.raises(ValueError, match="odd window"):
        HarmoniseConfig(window=10)
    assert check_signal("raw") == "raw" and check_signal("matched") == "matched"  # matched: design.md 15.1
    with pytest.raises(ValueError, match="signal"):
        check_signal("smooth")


def test_speed_helpers_keep_the_acceleration_definition() -> None:
    rng = np.random.default_rng(0)

    def reference(v: np.ndarray) -> np.ndarray:  # acceleration_from_speed before it was split into two helpers
        from scipy.signal import savgol_filter

        n, window = v.shape[0], 11
        if n < window:
            window = n if n % 2 else n - 1
        smooth = savgol_filter(v, window, 2, mode="interp") if window > 2 else v
        if n >= 3:
            return np.gradient(smooth, DT, edge_order=2)
        return np.full(n, (smooth[-1] - smooth[0]) / DT) if n == 2 else np.zeros(n)

    for n in (0, 1, 2, 3, 4, 7, 11, 12, 150):
        v = 10.0 + rng.standard_normal(n)
        np.testing.assert_array_equal(acceleration_from_speed(v), reference(v))
    batch = 10.0 + rng.standard_normal((4, 60))
    np.testing.assert_allclose(savgol_speed(batch), np.stack([savgol_speed(r) for r in batch]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(acceleration_from_speed(batch), np.stack([reference(r) for r in batch]), rtol=0, atol=1e-12)


# ------------------------------------------------------------------------------------ loader
def test_loader_applies_the_transform_only_when_asked(tmp_path: Path) -> None:
    stored = EventSet([_smooth_event(f"e{i}", c=0.1 * i, gap_noise=0.1, seed=i) for i in range(3)])
    stored.to_parquet(tmp_path / "set")
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in (tmp_path / "set").iterdir()}
    raw = load_events(tmp_path / "set")
    for a, b in zip(raw, stored):
        assert np.array_equal(a.s, b.s) and np.array_equal(a.v, b.v) and np.array_equal(a.v_lead, b.v_lead)
        assert "signal" not in a.meta and a.meta["a_source_original"] == "dataset"  # unified, not harmonised
    harm = load_events(tmp_path / "set", signal="harmonised")
    assert [ev.meta["signal"] for ev in harm] == ["harmonised"] * 3
    assert [ev.meta["a_source_original"] for ev in harm] == ["dataset"] * 3
    expected, _ = harmonise_events(load_events(tmp_path / "set"))
    for a, b in zip(harm, expected):
        np.testing.assert_array_equal(a.s, b.s)
    strict, report = load_events(tmp_path / "set", signal="harmonised", harmonise={"max_abs_c": 0.15}, return_report=True)
    assert [ev.event_id for ev in strict] == ["e0", "e1"] and report["dropped"]["max_abs_c"] == 1  # type: ignore[index]
    again, none = load_events(tmp_path / "set", return_report=True)
    assert none is None and all(np.array_equal(a.s, b.s) for a, b in zip(again, stored))  # type: ignore[arg-type]
    assert before == {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in (tmp_path / "set").iterdir()}
    with pytest.raises(ValueError, match="signal"):
        load_events(tmp_path / "set", signal="smooth")


# ------------------------------------------------------------------------------------ runner
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
    root = tmp_path_factory.mktemp("signal_events")
    return {
        "hd": _write_set(root, "hd", seed=1, n_drivers=25, per=2, duration=12.0),
        "ng": _write_set(root, "ng", seed=2, n_drivers=8, per=2, duration=15.0),
    }


def make_config(root: Path, data_configs: dict, models: tuple[str, ...] = ("idm_white_noise", "lstm_const_sigma"),
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
    for key, value in extra.items():
        cfg[key] = value
    return cfg


def _files(root: Path, skip: tuple[str, ...] = ("status.json", "run.log", "errors.log")) -> dict[str, int]:
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file() and p.name not in skip}


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory, data_configs: dict) -> tuple[DictConfig, DictConfig]:
    """A raw and a harmonised run of the same experiment (two roots)."""
    base = tmp_path_factory.mktemp("signal_runs")
    raw = make_config(base / "raw", data_configs)
    harm = make_config(base / "harm", data_configs, signal="harmonised")
    for cfg in (raw, harm):
        status = run_e1(cfg, echo=False)
        assert status["finished"] and status["failures"] == [] and status["tables"]["done"] == 2 * len(DATASETS)
    return raw, harm


def test_harmonised_run_stores_the_signal_variant_in_every_resume_key(runs: tuple[DictConfig, DictConfig],
                                                                      data_configs: dict) -> None:  # fmt: skip
    raw, harm = runs
    r_root, h_root = Path(raw.root), Path(harm.root)
    assert json.loads((h_root / "status.json").read_text(encoding="utf-8"))["signal"] == "harmonised"
    hd, report = harmonise_events(load_events(data_configs["hd"]["path"]))
    for dataset, spec in DATASETS.items():
        frame, meta = tables.load_windows(h_root, dataset)
        _, raw_meta = tables.load_windows(r_root, dataset)
        assert meta["spec"]["signal"] == "harmonised" and meta["spec"]["harmonise"] == SETTINGS and meta["signal"] == "harmonised"
        assert "signal" not in raw_meta["spec"] and "signal" not in raw_meta and "harmonisation" not in raw_meta
        events = hd if spec["data"] == "hd" else harmonise_events(load_events(data_configs["ng"]["path"]))[0]
        assert meta["harmonisation"]["n_in"] == meta["harmonisation"]["n_out"] + meta["harmonisation"]["n_dropped"]
        if spec["split"] != "all":
            events = split_by_folds(events, data_configs[spec["data"]]["split"]["file"], 0)[spec["split"]]
        windows = make_windows(events, stride=spec["stride"])
        assert meta["event_ids"] == [ev.event_id for ev in events] and len(frame) == len(windows)
        np.testing.assert_array_equal(frame["y"], windows.y)  # harmonised target and states
        np.testing.assert_array_equal(frame["s"], windows.X[:, -1, 0])
        np.testing.assert_array_equal(frame["v"], windows.X[:, -1, 2])
        assert all(tables.preds_path(h_root, m, 0, dataset).exists() for m in harm.models)
    stored = json.loads((h_root / "windows" / "highd_meta.json").read_text(encoding="utf-8"))["harmonisation"]
    assert stored["settings"] == SETTINGS and stored.pop("seconds") >= 0.0
    assert stored == {k: v for k, v in report.items() if k != "seconds"}  # the report of the training data set
    with np.load(h_root / "train_density.npz") as z, np.load(r_root / "train_density.npz") as zr:
        assert json.loads(str(z["signal"])) == KEY and "signal" not in zr.files
        train = split_by_folds(hd, data_configs["hd"]["split"]["file"], 0)["train"]
        assert int(z["n_total"]) == len(make_windows(train))
    idm, idm_raw = (json.loads((r / "idm.json").read_text(encoding="utf-8")) for r in (h_root, r_root))
    assert idm["calibrations"]["closed_loop"]["key"]["signal"] == KEY and "signal" not in idm_raw["key"]
    for model in harm.models:
        log = json.loads((checkpoint_dir(h_root, model, 0) / "train_log.json").read_text(encoding="utf-8"))
        raw_log = json.loads((checkpoint_dir(r_root, model, 0) / "train_log.json").read_text(encoding="utf-8"))
        assert log["signal"] == KEY and "signal" not in raw_log
    before = _files(h_root)
    status = run_e1(harm, echo=False)  # resume: nothing is rewritten
    assert status["failures"] == [] and status["tables"]["written_now"] == 0 and _files(h_root) == before


def test_raw_artefacts_are_never_reused_by_a_harmonised_config(runs: tuple[DictConfig, DictConfig], tmp_path: Path) -> None:
    raw, harm = runs
    r_root, h_root = Path(raw.root), Path(harm.root)
    before = _files(r_root)
    mixed = OmegaConf.merge(harm, {"root": str(r_root)})  # harmonised config, raw root: every unit is refused
    status = run_e1(mixed, echo=False)
    failed = {f["unit"]: f["error"] for f in status["failures"]}
    assert set(failed) == {"windows/highd", "windows/ngsim", "train_density", "idm_white_noise/config", "lstm_const_sigma/seed0"}
    assert "computed on raw signals" in failed["train_density"] and "'signal': 'harmonised'" in failed["windows/highd"]
    assert "has the key" in failed["idm_white_noise/config"]
    assert "was trained on raw signals, the config asks harmonised" in failed["lstm_const_sigma/seed0"]
    assert status["tables"]["written_now"] == 0 and _files(r_root) == before
    h_before = _files(h_root)
    status = run_e1(OmegaConf.merge(raw, {"root": str(h_root)}), echo=False)  # and the other way round
    assert len(status["failures"]) == 5 and _files(h_root) == h_before
    other = OmegaConf.merge(harm, {"harmonise": {"max_abs_c": 0.25}})  # other settings: another variant
    assert len(run_e1(other, echo=False)["failures"]) == 5 and _files(h_root) == h_before
    reuse = make_config(tmp_path / "reuse", raw.data_configs, models=("lstm_const_sigma",), signal="harmonised",
                        reuse_checkpoints={"lstm_const_sigma": str(r_root / "checkpoints")})  # fmt: skip
    status = run_e1(reuse, echo=False)
    assert [f["unit"] for f in status["failures"]] == ["lstm_const_sigma/seed0"]
    assert "trained on raw signals" in status["failures"][0]["error"] and status["trained"] == []
    assert not (checkpoint_dir(tmp_path / "reuse", "lstm_const_sigma", 0) / "model.pt").exists()


def test_dry_run_reads_only(runs: tuple[DictConfig, DictConfig], tmp_path: Path) -> None:
    raw, harm = runs
    r_root = Path(raw.root)
    every = _files(r_root, skip=())  # logs and status included: a dry run writes nothing at all
    plan = dry_run_e1(raw, echo=False)
    assert plan["nothing_to_do"] and plan["conflicts"] == [] and plan["signal"] == "raw"
    assert len(plan["tables"]["done"]) == 4 and len(plan["checkpoints"]["done"]) == 2 and plan["idm"]["done"] == ["idm/closed_loop"]
    plan = dry_run_e1(OmegaConf.merge(harm, {"root": str(r_root)}), echo=False)
    assert not plan["nothing_to_do"] and plan["signal"] == "harmonised"
    assert {c["unit"] for c in plan["conflicts"]} == {"windows/highd", "windows/ngsim", "train_density", "idm/closed_loop",
                                                      "idm_white_noise/config", "lstm_const_sigma/seed0"}  # fmt: skip
    assert _files(r_root, skip=()) == every
    fresh = OmegaConf.merge(harm, {"root": str(tmp_path / "fresh")})
    plan = dry_run_e1(fresh, echo=False)
    assert plan["conflicts"] == [] and not (tmp_path / "fresh").exists()
    assert plan["windows"]["missing"] == ["windows/highd", "windows/ngsim"] and plan["idm"]["missing"] == ["idm/closed_loop"]
    assert plan["checkpoints"]["missing"] == ["idm_white_noise/seed0", "lstm_const_sigma/seed0"] and len(plan["tables"]["missing"]) == 4
    assert dry_run_e1(harm, echo=False)["nothing_to_do"]


def test_signal_resolution(data_configs: dict) -> None:
    cfg = make_config(Path("unused"), data_configs)
    assert data_signal(cfg, "hd") == "raw" and signal_key(cfg, "hd") is None and signal_key(cfg) is None
    assert "signal" not in Data(cfg).eval_spec("highd") and model_spec(cfg, "lstm_const_sigma").signal is None
    per_set = make_config(Path("unused"), {**data_configs, "hd": {**data_configs["hd"], "signal": "harmonised"}})
    assert data_signal(per_set, "hd") == "harmonised" and data_signal(per_set, "ng") == "raw"  # data config key
    assert Data(per_set).eval_spec("highd")["signal"] == "harmonised" and "signal" not in Data(per_set).eval_spec("ngsim")
    assert model_spec(per_set, "lstm_const_sigma").signal == KEY
    over = OmegaConf.merge(per_set, {"signal": "raw"})  # the experiment-level key overrides every data set
    assert data_signal(over, "hd") == "raw" and signal_key(over, "hd") is None
    harm = make_config(Path("unused"), data_configs, signal="harmonised", harmonise={"max_gap_rms": 1.5})
    assert signal_key(harm, "ng") == {"signal": "harmonised", "harmonise": {**SETTINGS, "max_gap_rms": 1.5}} == signal_key(harm)
    with pytest.raises(ValueError, match="signal"):
        data_signal(make_config(Path("unused"), data_configs, signal="smooth"), "hd")


def _load_script(name: str) -> Any:
    """Module of ``scripts/<name>.py`` (the scripts directory is not a package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"{name}_script", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_analysis_scripts_check_the_signal_variant(tmp_path: Path) -> None:
    extra = _load_script("analyze_e1_extra")
    raw, harm = {"spec": {"data": "highd"}}, {"spec": {"data": "highd", "signal": "harmonised"}}
    e1, e1h = str(REPO / "configs" / "experiment" / "e1.yaml"), str(REPO / "configs" / "experiment" / "e1h.yaml")
    assert extra.check_signal_variant(tmp_path, {"highd": raw, "ngsim": raw}, e1) == "raw"
    assert extra.check_signal_variant(tmp_path, {"highd": harm}, e1h) == "harmonised"
    with pytest.raises(SystemExit, match="e1h.yaml"):  # rare3d would use raw training windows for a harmonised root
        extra.check_signal_variant(tmp_path, {"highd": harm}, e1)
    with pytest.raises(SystemExit, match="different signal variants"):
        extra.check_signal_variant(tmp_path, {"highd": harm, "ngsim": raw}, e1h)
    summarize = _load_script("summarize_e1")
    (tmp_path / "windows").mkdir()
    assert summarize.root_signal(tmp_path) == "raw"
    (tmp_path / "windows" / "highd_meta.json").write_text(json.dumps(harm), encoding="utf-8")
    assert summarize.root_signal(tmp_path) == "harmonised"


# ------------------------------------------------------------------------------------ configs
def test_harmonised_configs_differ_only_where_intended() -> None:
    exp = REPO / "configs" / "experiment"
    for name, base in (("e1h", "e1"), ("e1h_mean1s", "e1_mean1s")):
        child = OmegaConf.to_container(load_config(exp / f"{name}.yaml"))
        parent = OmegaConf.to_container(load_config(exp / f"{base}.yaml"))
        assert child["root"] == f"artifacts/{name}" and child["signal"] == "harmonised" and child["harmonise"] == SETTINGS
        assert set(child["model_overrides"]) == {"quantile_pidl"} and set(child["model_overrides"]["quantile_pidl"]) == {"w_phys"}
        assert load_config(exp / f"{name}.yaml", pilot=True).root == f"artifacts/{name}_pilot"
        for cfg in (child, parent):
            for key in ("root", "signal", "harmonise", "model_overrides", "reuse_checkpoints"):
                cfg.pop(key, None)
        assert child == parent
    mean = load_config(exp / "e1h_mean1s.yaml")
    assert OmegaConf.to_container(mean.reuse_checkpoints) == {"diffusion_cf": "artifacts/e1h/checkpoints"}
    pilot = load_config(exp / "e1h_mean1s.yaml", pilot=True)
    assert OmegaConf.to_container(pilot.reuse_checkpoints) == {"diffusion_cf": "artifacts/e1h_pilot/checkpoints"}
    assert signal_key(load_config(exp / "e1h.yaml"), "highd") == signal_key(mean, "highd") == KEY
