"""E3-pooled (docs/design.md 17.1): per-variant model specs, the leader re-smoothed from the raw OpenACC files, strict
specs and profile files on resume, the dry run, and the E3 analysis with a third signal variant; tiny CPU checkpoints
and synthetic OpenACC files."""

from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from cf_unc.constants import DT
from cf_unc.experiments.e3 import (
    E3Runner,
    dry_run_e3,
    load_config,
    parse_variants,
    profile_window,
    run_e3,
    unit_path,
    variant_profiles_path,
)
from cf_unc.models.baselines import IDMWhiteNoise
from cf_unc.physics.idm import IDMParams, equilibrium_gap
from cf_unc.platoon.openacc import load_runs
from cf_unc.platoon.profiles import (
    SelectionRules,
    harmonise_speed,
    load_profiles,
    profiles_digest,
    resmooth_profiles,
    select_profiles,
    write_profiles,
)

REPO = Path(__file__).resolve().parents[1]
IDM = {"v0": 30.0, "T": 1.5, "s0": 2.0, "a_max": 1.2, "b": 1.8, "delta": 4.0}
RULES = SelectionRules(max_length_s=40.0, preferred_length_s=35.0, max_profiles=4, n_headline=2)
WINDOW = 31  # the re-smoothing window of the test variant (the real one: 61, artifacts/e1m/matching.json)
MODELS = {
    "idm_white_noise": {"target": "a", "sampler_mode": "per_step", "hold_steps": 1},
    "noisy": {"target": "a", "sampler_mode": "per_step", "hold_steps": 1},
    "held": {"target": "mean_1s", "sampler_mode": "per_step", "hold_steps": 10},  # QuantilePIDL's place in 13.2
}
N_V, N_R = 3, 4


def _load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"{name}_script", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ------------------------------------------------------------------------------------- synthetic data
def platoon_speeds(depths: list[float], n_followers: int = 4, seed: int = 0, base: float = 12.0) -> np.ndarray:
    """(V, T) speeds: a noisy leader with 10 s cosine dips between steady stretches, followers delayed by 1 s each."""
    rng = np.random.default_rng(seed)
    j = np.arange(100)
    parts = [np.full(500, base)]
    for d in depths:
        parts += [base - d / 2 * (1 - np.cos(2 * np.pi * j / 100)), np.full(600, base)]
    u = np.concatenate(parts)
    n = u.size
    out = [u + rng.normal(0.0, 0.08, n)]
    for i in range(1, n_followers + 1):
        out.append(np.concatenate([np.full(10 * i, u[0]), u[: n - 10 * i]]) + rng.normal(0.0, 0.08, n))
    return np.stack(out)


def write_openacc(path: Path, speed: np.ndarray, acc: int, gap: float = 20.0) -> None:
    """An OpenACC csv file (header lines, then Time, Speed<i>, IVS<i>) at 10 Hz."""
    n_v, n = speed.shape
    head = ["Date,8,10,2019,", "Vehicle_order," + ",".join(f"V{i}" for i in range(1, n_v + 1)) + ",",
            f"Number_of_vehicles,{n_v},", f"ACC,{acc},", "Distance_setting,S,"]  # fmt: skip
    rows = [",".join(["Time"] + [c for i in range(1, n_v + 1) for c in (f"Speed{i}", f"IVS{i}")])]
    for k in range(n):
        cells = [f"{k * DT:.1f}"]
        for i in range(n_v):
            cells += [f"{speed[i, k]:.6f}", "" if i == n_v - 1 else f"{gap:.3f}"]
        rows.append(",".join(cells))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(head + rows) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Raw OpenACC files (an ACC run in AstaZero, a human-driven one in ZalaZone), the profiles selected from them, a
    matching file, the IDM file and checkpoints: raw (target a and mean_1s) and a pooled root (target a only)."""
    base = tmp_path_factory.mktemp("e3_pooled")
    raw = base / "raw"
    write_openacc(raw / "AstaZero" / "acc_run.csv", platoon_speeds([4.0, 3.0], seed=1), acc=1)
    write_openacc(raw / "ZalaZone" / "human_run.csv", platoon_speeds([5.0, 3.5], seed=2, base=10.0), acc=0)
    profiles, counts = select_profiles(load_runs(raw, ("AstaZero", "ZalaZone")), RULES)
    rules = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(RULES).items()}
    write_profiles(base / "profiles.yaml", profiles, {"description": "test", "source": str(raw), "rules": rules,
                                                     "digest": profiles_digest(profiles), "counts": counts})  # fmt: skip
    (base / "matching.json").write_text(json.dumps({"windows": {"highd": 11, "openacc": WINDOW}}), encoding="utf-8")
    (base / "idm.json").write_text(json.dumps({"calibrations": {"closed_loop": {"params": IDM}}}), encoding="utf-8")
    for root, labels in (("ckpt_a", ("idm_white_noise", "noisy")), ("ckpt_m", ("held",)),
                         ("pool_a", ("idm_white_noise", "noisy", "held"))):  # fmt: skip
        for label in labels:
            sigma = {"idm_white_noise": 0.3, "noisy": 0.8, "held": 0.2}[label]
            IDMWhiteNoise(idm=IDM, sigma=sigma).save(base / root / label / "seed0" / "model.pt")
    return {"base": base, "raw": raw, "profiles": base / "profiles.yaml"}


def make_config(root: Path, world: dict[str, Any], window: Any = None, override: dict | None = None) -> DictConfig:
    base = world["base"]
    cfg = load_config()
    cfg.root = str(root)
    cfg.device = "cpu"
    cfg.checkpoints = OmegaConf.create({"a": str(base / "ckpt_a"), "mean_1s": str(base / "ckpt_m")})
    window = {"matching": str(base / "matching.json"), "data": "openacc"} if window is None else window
    cfg.variants = OmegaConf.create({"pool": {"a": str(base / "pool_a"), "profile_window": window,
                                              "models": override or {"held": {"target": "a", "hold_steps": 1}}}})  # fmt: skip
    cfg.models = OmegaConf.create(dict(MODELS))
    cfg.seeds = [0]
    cfg.platoon = OmegaConf.create({"n_followers": N_V, "n_rollouts": N_R, "n_rollouts_sampled": 3, "history_s": 3.0,
                                    "burn_in_s": 2.0, "tail_s": 3.0, "max_batch": 1000, "seed": 0, "gap_cap": 120.0,
                                    "batch_profiles": 2})  # fmt: skip
    cfg.profiles = str(world["profiles"])
    cfg.idm = str(base / "idm.json")
    cfg.sets = OmegaConf.create({"headline": 2, "extended": None})
    cfg.bootstrap.n_boot = 50
    cfg.calibration.n_min = 1
    return cfg


@pytest.fixture(scope="module")
def first(tmp_path_factory: pytest.TempPathFactory, world: dict[str, Any]) -> tuple[DictConfig, dict]:
    cfg = make_config(tmp_path_factory.mktemp("e3_pooled_run") / "run", world)
    return cfg, run_e3(cfg, echo=False)


def _mtimes(root: Path) -> dict[str, int]:
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


def _copy(cfg: DictConfig, target: Path) -> DictConfig:
    shutil.copytree(cfg.root, target)
    out = OmegaConf.create(OmegaConf.to_container(cfg))
    out.root = str(target)
    return out


# ------------------------------------------------------------------------------------------ re-smoothing
def test_resmooth_with_the_selection_window_reproduces_the_profiles(world: dict[str, Any]) -> None:
    profiles, meta = load_profiles(world["profiles"])
    assert [p.mode for p in profiles] == ["ACC", "Human", "ACC", "Human"] and meta["source"] == str(world["raw"])
    same, info = resmooth_profiles(profiles, world["raw"], 11, 2)
    assert info["digest"] == profiles_digest(profiles) == meta["digest"]  # same file, start, length and window
    assert info["check"]["max_abs_deviation"] <= 5.1e-5 and set(info["files"]) == {"AstaZero/acc_run.csv", "ZalaZone/human_run.csv"}
    for a, b in zip(same, profiles):
        assert a.to_dict() == b.to_dict()


def test_resmooth_other_window_changes_only_the_leader(world: dict[str, Any]) -> None:
    profiles, _ = load_profiles(world["profiles"])
    runs: dict[str, Any] = {}
    new, info = resmooth_profiles(profiles, world["raw"], WINDOW, 2, runs=runs)
    assert len(runs) == 2  # every raw file parsed once
    assert info["window"] == WINDOW and info["order"] == 2 and info["base_window"] == 11
    assert info["digest"] == profiles_digest(new) != profiles_digest(profiles)
    sha = __import__("hashlib").sha256((world["raw"] / "AstaZero" / "acc_run.csv").read_bytes()).hexdigest()[:16]
    assert info["files"]["AstaZero/acc_run.csv"]["sha256"] == sha
    for a, b in zip(new, profiles):
        keep = ("id", "file", "campaign", "mode", "start_s", "start_index", "length_s", "n_followers", "group",
                "rank_in_group", "round", "headline", "vehicles")  # fmt: skip
        assert {k: getattr(a, k) for k in keep} == {k: getattr(b, k) for k in keep}
        np.testing.assert_array_equal(a.follower_amplitude, b.follower_amplitude)  # measured: window of the selection
        np.testing.assert_array_equal(a.follower_sd, b.follower_sd)
        assert a.n_samples == b.n_samples and 0 < np.max(np.abs(a.leader_speed - b.leader_speed)) < 0.5
        assert np.sqrt(np.mean(np.diff(a.leader_speed, 2) ** 2)) < 0.5 * np.sqrt(np.mean(np.diff(b.leader_speed, 2) ** 2))
        assert a.initial_speed == a.leader_speed[0] and a.final_speed == a.leader_speed[-1]
        assert a.leader_amplitude == pytest.approx(np.ptp(a.leader_speed), abs=1e-4) and a.leader_amplitude < b.leader_amplitude
        run = load_runs(world["raw"], (a.campaign,))[0]
        u = harmonise_speed(run.speed[0], WINDOW, 2)[a.start_index : a.start_index + a.n_samples]
        np.testing.assert_allclose(a.leader_speed, u, atol=5.1e-5)  # the raw file, same samples, the variant's window
    assert info["change"]["leader_amplitude"]["max"] < 0


def test_resmooth_is_idempotent_and_checks_the_raw_files(world: dict[str, Any], tmp_path: Path) -> None:
    profiles, _ = load_profiles(world["profiles"])
    a, info_a = resmooth_profiles(profiles, world["raw"], WINDOW)
    b, info_b = resmooth_profiles(profiles, world["raw"], WINDOW)
    assert info_a == info_b and [p.to_dict() for p in a] == [p.to_dict() for p in b]
    first = write_profiles(tmp_path / "a.yaml", a, info_a).read_bytes()
    assert write_profiles(tmp_path / "b.yaml", b, info_b).read_bytes() == first
    loaded, meta = load_profiles(tmp_path / "a.yaml")  # the stored rounding: loading gives the same profiles
    assert [p.to_dict() for p in loaded] == [p.to_dict() for p in a] and meta["digest"] == profiles_digest(loaded)
    for bad in (10, 1, 0):
        with pytest.raises(ValueError, match="odd"):
            resmooth_profiles(profiles, world["raw"], bad)
    changed = tmp_path / "raw"
    shutil.copytree(world["raw"], changed)
    path = changed / "AstaZero" / "acc_run.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    k = 6 + profiles[0].start_index + 100  # a data row inside the first profile (5 header lines, the table header)
    cells = lines[k].split(",")
    cells[1] = f"{float(cells[1]) + 2.0:.6f}"  # the leader's speed changed
    lines[k] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not reproduce"):
        resmooth_profiles(profiles, changed, WINDOW)
    path.unlink()
    with pytest.raises(ValueError, match="not found"):
        resmooth_profiles(profiles, changed, WINDOW)


# --------------------------------------------------------------------------------------------- runner
def test_variant_config(world: dict[str, Any], tmp_path: Path) -> None:
    cfg = make_config(tmp_path / "x", world)
    v = parse_variants(cfg)
    assert list(v) == ["raw", "pool"] and v["raw"].models == {} and v["raw"].profile_window is None
    assert v["pool"].checkpoints == {"a": str(world["base"] / "pool_a")} and v["pool"].profile_window == WINDOW
    assert v["pool"].window_source == {"matching": str(world["base"] / "matching.json"), "data": "openacc"}
    assert v["pool"].models == {"held": {"target": "a", "hold_steps": 1}}
    assert profile_window(61) == (61, None)
    for bad in (60, True, "61", 1):
        with pytest.raises(ValueError):
            profile_window(bad)
    with pytest.raises(ValueError, match="no window"):
        profile_window({"matching": str(world["base"] / "matching.json"), "data": "waymo"})
    for override, message in (({"lstm": {"target": "a"}}, "not a key of models"), ({"held": {"w_phys": 1}}, "may override")):
        with pytest.raises(ValueError, match=message):
            E3Runner(make_config(tmp_path / "bad", world, override=override), echo=False)
    assert not (tmp_path / "bad").exists()  # a config error creates nothing


@pytest.mark.skipif(not Path("artifacts/e1m/matching.json").exists(), reason="the real E3 config resolves its profile window from the E5 matching file (an experiment artefact)")
def test_real_variant_config() -> None:
    """The shipped config: QuantilePIDL on target a with hold 1 in both E3-pooled variants, 13.2 elsewhere."""
    real = parse_variants(load_config())
    # the variants of 13 and 17.1 first; the review controls of design.md 18.2 after them (tests/test_review_controls.py)
    assert list(real) == ["raw", "harmonised", "pool", "lodo_openacc", "raw_w61", "raw_a", "pool_w11", "lodo_openacc_w11"]
    for name in ("pool", "lodo_openacc"):
        assert real[name].models == {"quantile_pidl": {"target": "a", "hold_steps": 1}} and set(real[name].checkpoints) == {"a"}
        assert real[name].window_source == {"matching": "artifacts/e1m/matching.json", "data": "openacc"}
    assert real["harmonised"].models == {} and real["harmonised"].profile_window is None


def test_pooled_variant_runs_on_target_a_with_its_leader(first: tuple[DictConfig, dict], world: dict[str, Any]) -> None:
    cfg, status = first
    root = Path(cfg.root)
    n = len(MODELS) * 4
    assert status["finished"] and status["failures"] == [] and status["skipped"] == {}  # no pool/mean_1s to skip
    assert status["variants"] == {"raw": {"done": n, "total": n}, "pool": {"done": n, "total": n}}
    timings = [json.loads(line) for line in (root / "logs" / "timings.jsonl").read_text(encoding="utf-8").splitlines()]
    hold = {(t["variant"], t["model"]): t["hold_steps"] for t in timings}
    assert hold[("raw", "held")] == 10 and hold[("pool", "held")] == 1
    spec = json.loads((root / "units" / "pool" / "held" / "spec.json").read_text(encoding="utf-8"))
    stored, meta = load_profiles(variant_profiles_path(root, "pool"))
    assert spec == {"label": "held", "model": "held", "target": "a", "checkpoints": str(world["base"] / "pool_a"),
                    "sampler_mode": "per_step", "hold_steps": 1, "profile_window": WINDOW, "profile_order": 2,
                    "profiles_digest": meta["digest"]}  # fmt: skip
    raw_spec = json.loads((root / "units" / "raw" / "held" / "spec.json").read_text(encoding="utf-8"))
    assert set(raw_spec) == {"label", "model", "target", "checkpoints", "sampler_mode", "hold_steps"}  # as before 17.1
    assert raw_spec["target"] == "mean_1s" and raw_spec["hold_steps"] == 10
    base, _ = load_profiles(world["profiles"])
    assert meta["variant"] == "pool" and meta["window"] == WINDOW and meta["source_digest"] == profiles_digest(base)
    assert meta["window_source"]["data"] == "openacc" and meta["digest"] == profiles_digest(stored) != profiles_digest(base)
    idm = IDMParams.from_dict(IDM)
    for p, q in zip(base, stored):
        pool = pd.read_parquet(unit_path(root, "pool", "noisy", 0, p.id))
        raw = pd.read_parquet(unit_path(root, "raw", "noisy", 0, p.id))
        np.testing.assert_allclose(pool["leader_amplitude"], q.leader_amplitude, atol=2e-4)
        np.testing.assert_allclose(raw["leader_amplitude"], p.leader_amplitude, atol=2e-4)
        np.testing.assert_allclose(pool["s_init"], min(float(equilibrium_gap(q.initial_speed, idm)), 120.0), rtol=1e-6)
        np.testing.assert_allclose(raw["s_init"], min(float(equilibrium_gap(p.initial_speed, idm)), 120.0), rtol=1e-6)
        np.testing.assert_array_equal(pool["emp_amplitude"].to_numpy(), raw["emp_amplitude"].to_numpy())  # measured
        paths = np.load(unit_path(root, "pool", "noisy", 0, p.id, ".npz"))
        np.testing.assert_allclose(paths["leader_speed"][: q.n_samples], q.leader_speed, rtol=1e-6)
        assert not np.array_equal(pool["amplitude"], raw["amplitude"])  # another leader (and checkpoint root)


def test_resume_is_strict(first: tuple[DictConfig, dict], world: dict[str, Any], tmp_path: Path) -> None:
    cfg = _copy(first[0], tmp_path / "copy")
    root = Path(cfg.root)
    before = _mtimes(root)
    status = run_e3(cfg, echo=False)
    assert status["finished"] and status["failures"] == [] and status["written_now"] == 0
    skip = {"status.json", "run.log", "config.yaml"}
    after = _mtimes(root)
    assert {k: v for k, v in after.items() if Path(k).name not in skip} == {k: v for k, v in before.items()
                                                                           if Path(k).name not in skip}  # fmt: skip
    target = unit_path(root, "pool", "held", 0, "p02")
    original = pd.read_parquet(target)
    target.unlink()
    target.with_suffix(".npz").unlink()
    # another window: the stored profile file refuses the variant, nothing of it is written; raw is untouched
    status = run_e3(make_config(root, world, window=21), echo=False)
    assert [f["unit"] for f in status["failures"]] == ["pool/profiles"] and "was written with" in status["failures"][0]["error"]
    assert not target.exists()
    # another hold of the override: the unit spec refuses the group
    status = run_e3(make_config(root, world, override={"held": {"target": "a", "hold_steps": 2}}), echo=False)
    assert [f["unit"] for f in status["failures"]] == ["pool/held/seed0"] and "spec.json was written with" in status["failures"][0]["error"]
    assert not target.exists()
    # the settings of the first run: the unit is regenerated identically
    status = run_e3(cfg, echo=False)
    assert status["failures"] == [] and status["written_now"] == 1
    pd.testing.assert_frame_equal(pd.read_parquet(target), original, check_exact=True)
    changed = {k for k, v in _mtimes(root).items() if before.get(k) != v and Path(k).name not in skip | {"errors.log", "timings.jsonl"}}
    assert changed == {str(target.relative_to(root)), str(target.with_suffix(".npz").relative_to(root))}


def test_dry_run_writes_nothing(first: tuple[DictConfig, dict], world: dict[str, Any], tmp_path: Path) -> None:
    cfg, _ = first
    root = Path(cfg.root)
    before = _mtimes(root)
    plan = dry_run_e3(cfg, echo=False)
    assert _mtimes(root) == before  # no log, no status, no spec
    assert plan["conflicts"] == [] and plan["nothing_to_do"] and plan["spec"] == "equal" and plan["done"] == 2 * len(MODELS) * 4
    pool = plan["variants"]["pool"]
    assert pool["targets"] == ["a"] and pool["profile_window"] == WINDOW and pool["profiles"]["state"] == "stored"
    held = pool["models"]["held"]
    assert {k: held[k] for k in ("target", "hold_steps", "spec", "done", "missing", "missing_checkpoints")} == {
        "target": "a", "hold_steps": 1, "spec": "equal", "done": 4, "missing": 0, "missing_checkpoints": []}
    assert plan["variants"]["raw"]["models"]["held"]["target"] == "mean_1s"
    conflict = dry_run_e3(make_config(root, world, window=21), echo=False)
    assert [c["unit"] for c in conflict["conflicts"]] == ["pool/profiles"] and _mtimes(root) == before
    fresh = dry_run_e3(make_config(tmp_path / "fresh", world), echo=False)
    assert not (tmp_path / "fresh").exists() and fresh["missing"] == 2 * len(MODELS) * 4 and fresh["conflicts"] == []
    assert fresh["variants"]["pool"]["profiles"]["state"] == "new" and fresh["variants"]["pool"]["models"]["held"]["spec"] == "new"
    # CLI: --dry-run, and --models drops the overrides of the models left out
    config_file = tmp_path / "e3_test.yaml"
    OmegaConf.save(cfg, config_file)
    run = _load_script("run_e3")
    assert run.main(["--config", str(config_file), "--dry-run", "--quiet", "--variants", "pool", "--models", "idm_white_noise"]) == 0
    assert _mtimes(root) == before


def test_analysis_with_a_third_variant(first: tuple[DictConfig, dict], world: dict[str, Any], tmp_path: Path) -> None:
    cfg = _copy(first[0], tmp_path / "copy")
    root = Path(cfg.root)
    analysis = _load_script("analyze_e3")
    out = analysis.analyze_e3(root, verbose=False)
    base = {p.id: p for p in load_profiles(world["profiles"])[0]}
    pool = {p.id: p for p in load_profiles(variant_profiles_path(root, "pool"))[0]}
    pi = out["profile_index"]
    assert set(pi["variant"]) == {"raw", "pool"} and len(pi[pi["variant"] == "pool"]) == len(MODELS) * 4 * N_V
    for variant, profiles in (("raw", base), ("pool", pool)):
        sel = pi[pi["variant"] == variant]
        np.testing.assert_allclose(sel["leader_amplitude"], [profiles[i].leader_amplitude for i in sel["profile"]])
        ok = sel["crps_response"].notna()
        np.testing.assert_allclose(sel.loc[ok, "crps_ratio_response"], sel.loc[ok, "crps_response"] / sel.loc[ok, "leader_amplitude"])
    assert min(abs(base[i].leader_amplitude - pool[i].leader_amplitude) for i in base) > 1e-2
    for name in ("amplitude_crps", "h32", "amplitude_coverage", "growth", "collisions"):
        assert set(out[name]["variant"]) >= {"raw", "pool"}, name
    h32 = out["h32"]
    assert set(h32.loc[h32["variant"] == "pool", "mode"]) == {"all", "ACC", "Human"}
    figures = {p.name for p in (root / "figures").glob("*.png")}
    assert figures == {f"{f}__{v}.png" for f in ("crps_by_index", "growth", "crps_bars") for v in ("raw", "pool")}
    summary = (root / "tables" / "summary.md").read_text(encoding="utf-8")
    assert summary.index("## Variant raw (signals as recorded)") < summary.index("## Variant pool (pooled domain, OpenACC seen)")
    assert "### H3.2 by driver mode: variant pool, set extended" in summary
    # without its profile file the units of the variant do not match the leader of the profiles file: refused
    variant_profiles_path(root, "pool").unlink()
    with pytest.raises(ValueError, match="simulated with another leader"):
        analysis.analyze_e3(root, verbose=False, figures=False)
