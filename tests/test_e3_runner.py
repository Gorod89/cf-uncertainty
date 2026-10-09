"""Experiment E3 runner (cf_unc.experiments.e3) and analysis (scripts/analyze_e3.py) on tiny checkpoints; CPU."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from cf_unc.constants import DT
from cf_unc.experiments.e3 import UNIT_COLUMNS, available_units, load_config, run_e3, select_profile_ids, unit_path
from cf_unc.models.base import MODEL_REGISTRY, CFModel, register
from cf_unc.models.baselines import IDMWhiteNoise
from cf_unc.physics.idm import IDMParams, equilibrium_gap
from cf_unc.platoon.evaluate import VERDICTS
from cf_unc.platoon.profiles import Profile, load_profiles, write_profiles

REPO = Path(__file__).resolve().parents[1]
IDM = {"v0": 30.0, "T": 1.5, "s0": 2.0, "a_max": 1.2, "b": 1.8, "delta": 4.0}
SEEDS = [0, 1]
MODELS = {
    "idm_white_noise": {"target": "a", "sampler_mode": "per_step", "hold_steps": 1},
    "idm_copy": {"model": "idm_white_noise", "target": "a", "sampler_mode": "per_step", "hold_steps": 1},
    "noisy": {"target": "a", "sampler_mode": "per_step", "hold_steps": 1},
    "held": {"target": "mean_1s", "sampler_mode": "per_step", "hold_steps": 10},
}
N_V, N_R = 3, 5


def _load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"{name}_script", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _profile(k: int, n: int, n_followers: int, campaign: str, mode: str, base: float) -> Profile:
    j = np.arange(n)
    u = base - 1.5 * (1 - np.cos(2 * np.pi * np.clip(j - 50, 0, 100) / 100))  # a 3 m/s dip from sample 50 to 150
    amp = 3.0 + 0.5 * np.arange(1, n_followers + 1)
    return Profile(
        id=f"p{k:02d}", file=f"run{k}.csv", campaign=campaign, mode=mode, start_s=10.0 * k, start_index=100 * k,
        length_s=n * DT, n_followers=n_followers, leader_amplitude=float(np.ptp(u)), leader_sd=float(u.std()),
        initial_speed=float(u[0]), final_speed=float(u[-1]), leader_speed=u, follower_amplitude=amp,
        follower_sd=amp / 3.0, headline=k <= 2,
    )  # fmt: skip


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("e3_setup")
    for seed in SEEDS:
        IDMWhiteNoise(idm=IDM, sigma=0.3).save(base / "ckpt_a" / "idm_white_noise" / f"seed{seed}" / "model.pt")
        IDMWhiteNoise(idm=IDM, sigma=0.8).save(base / "ckpt_a" / "noisy" / f"seed{seed}" / "model.pt")
        IDMWhiteNoise(idm=IDM, sigma=0.2).save(base / "ckpt_m" / "held" / f"seed{seed}" / "model.pt")
    (base / "idm.json").write_text(json.dumps({"calibrations": {"closed_loop": {"params": IDM}}}), encoding="utf-8")
    profiles = [_profile(1, 300, 3, "AstaZero", "ACC", 12.0), _profile(2, 320, 2, "ZalaZone", "Human", 9.0),
                _profile(3, 350, 4, "Vicolungo", "ACC", 15.0)]  # fmt: skip
    write_profiles(base / "profiles.yaml", profiles, {"description": "test profiles"})
    return {"base": base, "profiles": base / "profiles.yaml"}


def make_config(root: Path, setup: dict[str, Path], models: dict | None = None, seeds: list[int] = SEEDS) -> DictConfig:
    base = setup["base"]
    cfg = load_config()
    cfg.root = str(root)
    cfg.device = "cpu"
    cfg.checkpoints = OmegaConf.create({"a": str(base / "ckpt_a"), "mean_1s": str(base / "ckpt_m")})
    cfg.variants = OmegaConf.create({"harmonised": {"a": str(base / "missing_a"), "mean_1s": str(base / "missing_m")}})
    cfg.models = OmegaConf.create(dict(models or MODELS))
    cfg.seeds = list(seeds)
    cfg.platoon = OmegaConf.create({"n_followers": N_V, "n_rollouts": N_R, "n_rollouts_sampled": 3, "history_s": 3.0,
                                    "burn_in_s": 2.0, "tail_s": 3.0, "max_batch": 1000, "seed": 0, "gap_cap": 120.0,
                                    "batch_profiles": 2})  # fmt: skip
    cfg.profiles = str(setup["profiles"])
    cfg.idm = str(base / "idm.json")
    cfg.sets = OmegaConf.create({"headline": 2, "extended": None})
    cfg.bootstrap.n_boot = 100
    cfg.calibration.n_min = 1
    return cfg


@pytest.fixture(scope="module")
def first_run(tmp_path_factory: pytest.TempPathFactory, setup: dict[str, Path]) -> tuple[DictConfig, dict]:
    cfg = make_config(tmp_path_factory.mktemp("e3") / "run", setup)
    return cfg, run_e3(cfg, echo=False)


def _files(root: Path) -> dict[str, int]:
    skip = {"status.json", "run.log", "errors.log", "timings.jsonl", "config.yaml"}
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file() and p.name not in skip}


# ------------------------------------------------------------------------------------------ tests
def test_pilot_config_and_profile_sets(setup: dict[str, Path]) -> None:
    cfg = load_config(pilot=True)
    assert cfg.root == "artifacts/e3_pilot" and list(cfg.seeds) == [0] and cfg.run_profiles == "headline"
    assert cfg.platoon.n_rollouts == 20 and cfg.platoon.n_rollouts_sampled == 20 and len(cfg.models) == 6
    assert load_config(pilot=True, overrides=["platoon.max_batch=5000"]).platoon.max_batch == 5000
    profiles, _ = load_profiles(setup["profiles"])
    assert select_profile_ids(profiles, "headline", {"headline": 2, "extended": None}) == ["p01", "p02"]
    assert select_profile_ids(profiles, "extended", {"headline": 2, "extended": None}) == ["p01", "p02", "p03"]
    assert select_profile_ids(profiles, ["p03", "p01"]) == ["p01", "p03"]  # file order
    with pytest.raises(ValueError, match="unknown profiles"):
        select_profile_ids(profiles, ["p09"])


def test_first_run_complete(first_run: tuple[DictConfig, dict]) -> None:
    cfg, status = first_run
    root = Path(cfg.root)
    n = len(MODELS) * len(SEEDS) * 3
    assert status["finished"] and status["failures"] == [] and status["variants"] == {"raw": {"done": n, "total": n}}
    assert status["written_now"] == n and set(status["skipped"]) == {"harmonised/a", "harmonised/mean_1s"}
    assert not (root / "logs" / "errors.log").exists() and (root / "config.yaml").exists()
    log = (root / "logs" / "run.log").read_text(encoding="utf-8")
    assert "warning: variant harmonised, target a skipped" in log
    timings = [json.loads(line) for line in (root / "logs" / "timings.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(timings) == len(MODELS) * len(SEEDS) * 2  # batches of 2 profiles: (p01, p02), (p03)
    assert {tuple(t["profiles"]) for t in timings} == {("p01", "p02"), ("p03",)}
    assert {t["hold_steps"] for t in timings if t["model"] == "held"} == {10}
    spec = json.loads((root / "spec.json").read_text(encoding="utf-8"))
    assert spec["n_steps"] == 20 + 350 - 1 + 30 and spec["platoon"]["n_followers"] == N_V
    held = json.loads((root / "units" / "raw" / "held" / "spec.json").read_text(encoding="utf-8"))
    assert held["target"] == "mean_1s" and held["hold_steps"] == 10 and held["checkpoints"].endswith("ckpt_m")
    assert len(available_units(root)) == n


def test_unit_tables(first_run: tuple[DictConfig, dict], setup: dict[str, Path]) -> None:
    cfg, _ = first_run
    root = Path(cfg.root)
    profiles = {p.id: p for p in load_profiles(setup["profiles"])[0]}
    for pid, p in profiles.items():
        t = pd.read_parquet(unit_path(root, "raw", "noisy", 1, pid))
        assert list(t.columns) == list(UNIT_COLUMNS) and len(t) == N_V * N_R
        assert t["vehicle"].tolist() == np.repeat(np.arange(1, N_V + 1), N_R).tolist()
        assert t["rollout"].tolist() == np.tile(np.arange(N_R), N_V).tolist() and (t["seed"] == 1).all()
        assert t["amplitude"].dtype == np.float32 and t["first_collision"].dtype == np.int32
        assert (t["profile"].astype(str) == pid).all()
        m = min(N_V, p.n_followers)
        emp = t.groupby("vehicle")["emp_amplitude"].first().to_numpy()
        np.testing.assert_allclose(emp[:m], p.follower_amplitude[:m], rtol=1e-6)
        assert np.isnan(emp[m:]).all()
        np.testing.assert_allclose(t["leader_amplitude"], p.leader_amplitude, rtol=1e-5)
        gap = min(float(equilibrium_gap(p.initial_speed, IDMParams.from_dict(IDM))), 120.0)
        np.testing.assert_allclose(t["s_init"], gap, rtol=1e-6)
        assert (t["amplitude"] >= t["amplitude_profile"] - 1e-6).all() and (t["sd"] > 0).all()
        paths = np.load(unit_path(root, "raw", "noisy", 1, pid, ".npz"))
        assert paths["mean_speed"].shape == (N_V, p.n_samples + 30) and paths["mean_speed"].dtype == np.float16
        np.testing.assert_allclose(paths["leader_speed"][: p.n_samples], p.leader_speed, rtol=1e-6)
    # common random numbers: the same checkpoint under two labels gives identical tables; seeds differ
    a = pd.read_parquet(unit_path(root, "raw", "idm_white_noise", 0, "p01"))
    pd.testing.assert_frame_equal(a, pd.read_parquet(unit_path(root, "raw", "idm_copy", 0, "p01")), check_exact=True)
    assert not np.array_equal(a["amplitude"], pd.read_parquet(unit_path(root, "raw", "idm_white_noise", 1, "p01"))["amplitude"])
    assert not np.array_equal(a["amplitude"], pd.read_parquet(unit_path(root, "raw", "held", 0, "p01"))["amplitude"])


def test_resume_and_regenerate(first_run: tuple[DictConfig, dict]) -> None:
    cfg, _ = first_run
    root = Path(cfg.root)
    before = _files(root)
    status = run_e3(cfg, echo=False)
    assert status["finished"] and status["failures"] == [] and status["written_now"] == 0
    assert _files(root) == before  # nothing was rewritten
    target = unit_path(root, "raw", "noisy", 1, "p02")
    original = pd.read_parquet(target)
    original_paths = dict(np.load(target.with_suffix(".npz")))
    target.unlink()
    target.with_suffix(".npz").unlink()
    status = run_e3(cfg, echo=False)
    assert status["written_now"] == 1
    after = _files(root)
    changed = {k for k in after if after[k] != before.get(k)}
    assert changed == {str(target.relative_to(root)), str(target.with_suffix(".npz").relative_to(root))}
    pd.testing.assert_frame_equal(pd.read_parquet(target), original, check_exact=True)  # the batch is simulated again
    for key, value in np.load(target.with_suffix(".npz")).items():
        np.testing.assert_array_equal(value, original_paths[key])
    changed_cfg = OmegaConf.merge(cfg, {"platoon": {"seed": 5}})  # other settings than the stored ones: refused
    status = run_e3(changed_cfg, echo=False)
    assert [f["unit"] for f in status["failures"]] == ["spec"] and _files(root) == after


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
def boom_checkpoint(setup: dict[str, Path]):  # noqa: ANN201
    register("e3_test_boom")(_Boom)
    path = setup["base"] / "ckpt_a" / "boom" / "seed0" / "model.pt"
    _Boom().save(path)
    yield "boom"
    MODEL_REGISTRY.pop("e3_test_boom", None)
    path.unlink()


def test_failing_model_and_cli(tmp_path: Path, setup: dict[str, Path], boom_checkpoint: str) -> None:
    models = {"boom": {"target": "a"}, "idm_white_noise": MODELS["idm_white_noise"], "held": MODELS["held"]}
    cfg = make_config(tmp_path / "run", setup, models=models, seeds=[0])
    config_file = tmp_path / "e3_test.yaml"
    OmegaConf.save(cfg, config_file)
    run = _load_script("run_e3")
    code = run.main(["--config", str(config_file), "--quiet", "--models", "boom", "idm_white_noise", "--profiles", "headline",
                     "--variants", "raw"])  # fmt: skip
    assert code == 1
    root = tmp_path / "run"
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["finished"] and [f["unit"] for f in status["failures"]] == ["raw/boom/seed0"]
    assert status["variants"]["raw"] == {"done": 2, "total": 4} and not status["skipped"]  # only raw, headline profiles
    errors = (root / "logs" / "errors.log").read_text(encoding="utf-8")
    assert "Traceback" in errors and "RuntimeError: boom: injected failure" in errors
    assert [unit_path(root, "raw", "idm_white_noise", 0, p).exists() for p in ("p01", "p02", "p03")] == [True, True, False]
    assert not unit_path(root, "raw", "held", 0, "p01").exists()  # not selected with --models
    # a missing checkpoint of a configured model is a failure of that (variant, model, seed), the rest runs
    cfg2 = make_config(tmp_path / "run2", setup, models={"lstm_const_sigma": {"target": "a"}, **MODELS}, seeds=[0])
    cfg2.run_profiles = "headline"
    status = run_e3(cfg2, echo=False)
    assert [f["unit"] for f in status["failures"]] == ["raw/lstm_const_sigma/seed0"] and "does not exist" in status["failures"][0]["error"]
    assert status["done"] == len(MODELS) * 2


def test_analyze_e3_end_to_end(first_run: tuple[DictConfig, dict]) -> None:
    cfg, _ = first_run
    root = Path(cfg.root)
    analysis = _load_script("analyze_e3")
    out = analysis.analyze_e3(root, verbose=False)
    for name, columns in analysis.COLUMNS.items():
        frame = pd.read_csv(root / "tables" / f"{name}.csv")
        assert list(frame.columns) == columns and len(frame) > 0, name
    pi = out["profile_index"]
    assert len(pi) == len(MODELS) * 3 * N_V and (pi["n_seeds"] == 2).all() and (pi["n_samples"] == 2 * N_R).all()
    crps = out["amplitude_crps"]
    assert set(crps["set"]) == {"headline", "extended"} and set(crps["window"]) == {"response", "profile"}
    assert {"all", "1-4", "1", "2", "3"} <= set(crps["index"].astype(str)) and set(crps["mode"]) == {"all", "ACC", "Human"}
    ext = crps[(crps["set"] == "extended") & (crps["mode"] == "all") & (crps["window"] == "response")]
    assert (ext.loc[ext["index"].astype(str) == "all", "n_profiles"] == 3).all()
    assert (ext.loc[ext["index"].astype(str) == "3", "n_profiles"] == 2).all()  # p02 measured two followers only
    assert ((ext["crps_lo"] <= ext["crps"]) & (ext["crps"] <= ext["crps_hi"])).all()
    h32 = out["h32"]
    assert set(h32["verdict"]) <= set(VERDICTS) and set(h32["reference"]) == {"idm_white_noise"}  # no lstm in this run
    copy = h32[(h32["model"] == "idm_copy") & (h32["set"] == "extended") & (h32["mode"] == "all")]
    assert np.allclose(copy["reduction"], 0.0) and np.allclose(copy["lo"], 0.0)  # identical draws: no difference
    cov = out["amplitude_coverage"]
    assert set(cov["method"]) == {"raw", "static_loo", "aci_online"} and ((cov["coverage"] >= 0) & (cov["coverage"] <= 1)).all()
    growth = out["growth"]
    assert "empirical" in set(growth["model"]) and set(growth.loc[growth["model"] != "empirical", "index"]) == {1, 2, 3}
    col = out["collisions"]
    assert set(col["index"].astype(str)) == {"1", "2", "3", "any"} and ((col["collided"] >= 0) & (col["collided"] <= 1)).all()
    figures = {p.name for p in (root / "figures").glob("*.png")}
    assert figures == {"crps_by_index__raw.png", "growth__raw.png", "crps_bars__raw.png"}
    assert all(p.stat().st_size > 5000 for p in (root / "figures").glob("*.png"))
    text = analysis.format_summary(out)
    assert "H3.2 vs idm_white_noise" in text and "noisy" in text and "empirical" in text
    # per-mode H3.2 block (design.md 14.1): ACC 2 profiles, Human 1, all 3; a mode needs min_profiles for a verdict
    assert "H3.2 by driver mode: variant raw, set extended" in text

    def mode_rows(summary_text: str) -> list[list[str]]:  # split lines of the per-mode block, reference rows dropped
        rows = [line.split() for line in summary_text.splitlines() if line.split()[:1] in (["ACC"], ["Human"], ["all"])]
        return [r for r in rows if r[2] != "idm_white_noise"]

    rows = mode_rows(text)  # default min_profiles 5: no mode of this run is judged
    assert {r[0] for r in rows} == {"ACC", "Human", "all"} and {r[1] for r in rows if r[0] == "ACC"} == {"2"}
    assert all(r[-2:] == ["not", "judged"] for r in rows)
    rows = mode_rows(analysis.format_summary(out, min_profiles=2))
    assert all((r[-2:] == ["not", "judged"]) == (r[0] == "Human") for r in rows) and len(rows) == 3 * (len(MODELS) - 1)
    summary = (root / "tables" / "summary.md").read_text(encoding="utf-8")
    assert summary.startswith("# E3 summary") and "### H3.2 by driver mode: variant raw, set extended" in summary
    assert "| ACC | 2 | idm_white_noise |" in summary and "| Human | 1 |" in summary
