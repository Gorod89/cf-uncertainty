"""Review controls of the v4 revision (docs/design.md, section 18): DiffusionCF without the leader's future (R2), the E3
2 x 2 matrix and the per-model selection of the E3 runner (R3), the AR(1)-IDM baseline (R10), the frozen-mean
heteroscedastic ablation (R13), causal signals with recorded targets (R6), the KDE sensitivity of sampled laws (R14),
the E2 dry run and the H3.3 pair option of the E1 analysis. Tiny synthetic data, CPU only, a few seconds each."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from cf_unc.constants import A_MAX, A_MIN, DT, HIST_LEN, MEAN_TARGET_STEPS
from cf_unc.data.harmonise import (
    DROP_REASONS,
    HarmoniseConfig,
    MatchConfig,
    MatchingError,
    TargetSpeedEvent,
    causal_events,
    causal_savgol,
    causal_savgol_coeffs,
    matched_windows,
    with_recorded_targets,
)
from cf_unc.data.pipeline import load_events
from cf_unc.data.schema import Event, EventSet
from cf_unc.data.synthetic import generate
from cf_unc.data.windows import make_windows, target_speed
from cf_unc.dist import SampleDist
from cf_unc.experiments import e1 as e1_mod
from cf_unc.experiments import e2 as e2_mod
from cf_unc.experiments.e3 import E3Runner, dry_run_e3, parse_variants, run_e3, unit_path
from cf_unc.maps.io import windows_table
from cf_unc.models import build_model, load_model
from cf_unc.models.baselines import IDMARNoise, IDMWhiteNoise, LSTMConstSigma, fit_ar1, previous_residual
from cf_unc.models.diffusion_cf import DiffusionCF, DiffusionCFHist
from cf_unc.models.lstm_hetero_frozen import LSTMHeteroFrozen
from cf_unc.physics.idm import IDMParams, equilibrium_gap, idm_accel
from cf_unc.rollout.engine import RolloutConfig, rollout_events

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "configs" / "experiment"
MODEL_CFG = REPO / "configs" / "model"
IDM = IDMParams(v0=30.0, T=1.5, s0=2.0, a_max=1.2, b=1.8)
TINY_DIFF = dict(n_samples=8, n_sampling_steps=10, t_train=100, channels=16, n_blocks=2, hidden=16)
N_FOLDS = 5


def _load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"{name}_ctrl_script", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _load_test_module(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"{name}_helpers", REPO / "tests" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# =============================================================================================== configs
def test_model_configs_of_the_controls() -> None:
    hist = yaml.safe_load((MODEL_CFG / "diffusion_cf_hist.yaml").read_text(encoding="utf-8"))
    base = yaml.safe_load((MODEL_CFG / "diffusion_cf.yaml").read_text(encoding="utf-8"))
    assert hist["name"] == "diffusion_cf_hist" and base["name"] == "diffusion_cf"
    # the same sampler budget (32 samples, 50 DDIM steps) and training budget as diffusion_cf
    assert {k: v for k, v in hist.items() if k != "name"} == {k: v for k, v in base.items() if k != "name"}
    assert hist["kwargs"]["n_samples"] == 32 and hist["kwargs"]["n_sampling_steps"] == 50
    ar = yaml.safe_load((MODEL_CFG / "idm_ar_noise.yaml").read_text(encoding="utf-8"))
    assert ar["name"] == "idm_ar_noise" and ar["needs_idm"] and ar.get("idm", "closed_loop") == "closed_loop"
    assert ar["kwargs"] == {"rho": None, "sigma": None}
    frozen = yaml.safe_load((MODEL_CFG / "lstm_hetero_frozen.yaml").read_text(encoding="utf-8"))
    assert frozen["kwargs"]["mean_from"] == "artifacts/e1/checkpoints/lstm_const_sigma" and frozen["train"] == {}
    assert frozen["kwargs"]["head_hidden"] == 64  # the head of HeteroMDN


def test_experiment_configs_of_the_controls() -> None:
    e1 = e1_mod.load_config(EXP / "e1.yaml")
    assert "idm_ar_noise" in list(e1.models) and "idm_ar_noise" in list(e1.seed_independent)
    assert "idm_ar_noise" in list(e2_mod.load_config(EXP / "e2.yaml").models)
    ctrl = e1_mod.load_config(EXP / "e1_ctrl.yaml")
    assert OmegaConf.load(EXP / "e1_ctrl.yaml").base == "e1.yaml" and ctrl.root == "artifacts/e1_ctrl"
    assert list(ctrl.models) == ["diffusion_cf_hist", "lstm_hetero_frozen", "lstm_const_sigma", "hetero_mdn"]
    assert e1_mod.config_target(ctrl) == "a" and e1_mod.signal_key(ctrl, "highd") is None and list(ctrl.seeds) == [0, 1, 2, 3, 4]
    assert OmegaConf.to_container(ctrl.reuse_checkpoints) == {"lstm_const_sigma": "artifacts/e1/checkpoints",
                                                              "hetero_mdn": "artifacts/e1/checkpoints"}  # fmt: skip
    assert ctrl.eval.datasets == e1.eval.datasets and ctrl.train.stride == e1.train.stride  # row-aligned with artifacts/e1
    e2c = e2_mod.load_config(EXP / "e2_ctrl.yaml")
    assert e2c.root == "artifacts/e2_ctrl" and OmegaConf.to_container(e2c.targets) == {
        "a": {"checkpoints": "artifacts/e1_ctrl/checkpoints", "hold_steps": 1}}
    assert list(e2c.datasets) == ["highd_val", "highd", "ngsim", "openacc"] and not e2c.get("variants")
    assert list(e2c.models) == ["diffusion_cf_hist", "lstm_hetero_frozen"]
    assert OmegaConf.to_container(e2c.e1_roots) == {"a": "artifacts/e1_ctrl"}
    causal = e1_mod.load_config(EXP / "e1m_causal.yaml")
    assert causal.root == "artifacts/e1m_causal" and causal.signal == "matched_causal"
    assert causal.match.file == "artifacts/e1m_causal/matching.json" and causal.train.target == "mean_1s"
    assert causal.train.target_from == "recorded" and not causal.get("reuse_checkpoints")
    assert list(causal.models) == ["lstm_const_sigma", "hetero_mdn", "deep_ensemble", "mc_dropout", "gp_residual_idm"]
    splits = {k: str(v.get("split", "all")) for k, v in causal.eval.datasets.items()}
    assert splits == {"highd": "test", "highd_val": "val", "ngsim": "all_but_val", "ngsim_us101": "all_but_val",
                      "ngsim_fn": "all_but_val", "waymo": "all_but_val", "openacc": "all_but_val"}  # fmt: skip


@pytest.mark.skipif(not (REPO / "artifacts/e1m/matching.json").exists(), reason="the E3 config reads the E5 matching file")
def test_e3_config_of_the_controls() -> None:
    from cf_unc.experiments.e3 import load_config as load_e3

    cfg = load_e3()
    v = parse_variants(cfg)
    names = list(v)
    assert v["raw_w61"].checkpoints == {"a": "artifacts/e1/checkpoints"} and v["raw_w61"].profile_window == 61
    assert v["raw_a"].checkpoints == {"a": "artifacts/e1/checkpoints"} and v["raw_a"].profile_window is None
    assert v["raw_a"].only_models == ["idm_white_noise", "quantile_pidl"]
    assert v["pool_w11"].checkpoints == {"a": "artifacts/e6_pool/checkpoints"} and v["pool_w11"].profile_window is None
    assert v["lodo_openacc_w11"].checkpoints == {"a": "artifacts/e6_lodo_openacc/checkpoints"}
    for name in ("raw_w61", "raw_a", "pool_w11", "lodo_openacc_w11"):  # one target rule
        assert v[name].models == {"quantile_pidl": {"target": "a", "hold_steps": 1}}
    # the pooled variants of 17.1 add the pooled checkpoints of the history-only Diffusion (design.md 18.8)
    for name, hist in (("pool", "artifacts/e6_pool_ctrl/checkpoints"), ("lodo_openacc", "artifacts/e6_lodo_openacc_ctrl/checkpoints")):
        assert v[name].models == {"quantile_pidl": {"target": "a", "hold_steps": 1}, "diffusion_cf_hist": {"checkpoints": hist}}
    models = OmegaConf.to_container(cfg.models)
    assert models["diffusion_cf_hist"]["checkpoints"] == "artifacts/e1_ctrl/checkpoints"
    assert models["diffusion_cf_hist"]["variants"] == ["raw", "pool", "lodo_openacc"]  # 18.1 and 18.8
    assert "harmonised" not in models["idm_ar_noise"]["variants"]
    for spec in models.values():  # every variant a model is restricted to exists
        assert set(spec.get("variants") or []) <= set(names)
    # the units of the models of 13.2 keep their specs: no review key on them
    for label in ("idm_white_noise", "lstm_const_sigma", "deep_ensemble", "gp_residual_idm", "quantile_pidl", "diffusion_cf"):
        assert set(models[label]) == {"target", "sampler_mode", "hold_steps"}


# =============================================================================================== R2
@pytest.fixture(scope="module")
def tiny_hist(synth_small_seq) -> DiffusionCFHist:
    model = DiffusionCFHist(**TINY_DIFF, train={"max_epochs": 2, "batch_size": 256})
    model.fit(synth_small_seq.windows["train"], synth_small_seq.windows["val"], seed=0)
    return model


def test_diffusion_hist_has_no_leader_input(synth_small_seq, tiny_hist: DiffusionCFHist) -> None:
    assert not DiffusionCFHist.needs_leader_future and DiffusionCFHist.plan_sampler and DiffusionCF.plan_sampler
    assert tiny_hist.net.inp.in_channels == 1 and not tiny_hist.net.leader
    assert DiffusionCF(**TINY_DIFF).arch.get("leader", True) and "leader" not in DiffusionCF(**TINY_DIFF).arch
    test = synth_small_seq.windows["test"]
    x, future = test.X[:12], test.v_lead_future[:12]
    a = tiny_hist.predict_dist(x).samples
    b = tiny_hist.predict_dist(x, v_lead_future=future).samples
    c = tiny_hist.predict_dist(x, v_lead_future=future + 5.0).samples
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)  # the leader's future is not an input
    assert tiny_hist.sample_sequences(x, n_samples=3).shape == (12, 3, tiny_hist.seq_len)


def test_diffusion_hist_trains_without_leader_futures(synth_small_seq) -> None:
    train = synth_small_seq.windows["train"].subset(np.arange(300))
    train.v_lead_future = None
    model = DiffusionCFHist(**TINY_DIFF, train={"max_epochs": 1, "batch_size": 128})
    log = model.fit(train, None, seed=0)
    assert log["epochs"] == 1 and float(model.net.vl_scale) == 1.0
    with pytest.raises(ValueError, match="v_lead_future"):
        DiffusionCF(**TINY_DIFF, train={"max_epochs": 1}).fit(train, None, seed=0)


def test_diffusion_hist_round_trip_and_rollout(tmp_path: Path, synth_small_seq, tiny_hist: DiffusionCFHist) -> None:
    path = tiny_hist.save(tmp_path / "hist.pt")
    loaded = load_model(path)
    assert isinstance(loaded, DiffusionCFHist) and loaded.net.inp.in_channels == 1
    x = synth_small_seq.windows["test"].X[:8]
    np.testing.assert_array_equal(loaded.predict_dist(x).samples, tiny_hist.predict_dist(x).samples)
    events, _ = generate(dict(n_drivers=2, events_per_driver=1, duration_s=8.0, seed=4))
    cfg = RolloutConfig(n_rollouts=3, horizon_s=1.0, horizons_s=(1.0,), stream_events=1)
    res = rollout_events(loaded, list(events), cfg=cfg)
    assert np.isfinite(res.s).all()
    state = loaded.init_sampler(2, torch.Generator().manual_seed(0), "per_step", v_lead=np.ones((2, 20)))
    assert state["v_lead"] is None  # ignored by the history-only model
    with pytest.raises(ValueError, match="hold_steps = 1"):
        rollout_events(loaded, list(events), cfg=RolloutConfig(n_rollouts=2, horizon_s=1.0, horizons_s=(1.0,), hold_steps=10))


# =============================================================================================== R10
def ar_pairs(rho: float, sigma: float, n_events: int, length: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(r_t, r_{t-1}) pairs of stationary AR(1) series within events."""
    rng = np.random.default_rng(seed)
    e = np.empty((n_events, length))
    e[:, 0] = rng.normal(0.0, sigma / np.sqrt(1.0 - rho**2), n_events)
    for t in range(1, length):
        e[:, t] = rho * e[:, t - 1] + rng.normal(0.0, sigma, n_events)
    return e[:, 1:].ravel(), e[:, :-1].ravel()


@pytest.mark.parametrize("rho", [0.0, 0.6, 0.95])
def test_fit_ar1_recovers_rho_and_sigma(rho: float) -> None:
    r, p = ar_pairs(rho, 0.3, n_events=200, length=300, seed=1)
    stats = fit_ar1(r, p)
    assert stats["rho"] == pytest.approx(rho, abs=0.01)
    assert stats["sigma_eta"] == pytest.approx(0.3, rel=0.02)
    assert stats["sigma_stationary"] == pytest.approx(0.3 / np.sqrt(1 - rho**2), rel=0.03)
    assert stats["conditional_rms"] == pytest.approx(stats["sigma_eta"], rel=0.02)  # previous residual observed exactly
    assert fit_ar1(r, p, rho=0.0)["sigma_eta"] == pytest.approx(np.sqrt(np.mean(r**2)), rel=1e-12)
    with pytest.raises(ValueError):
        fit_ar1(r, p, rho=1.0)


def ar_events(rho: float, sigma: float, n_events: int, seconds: float, seed: int, lag: int = 1, offset: int = 0) -> EventSet:
    """Followers driven by ``a_t = IDM(x_t) + e_t`` (AR(1) noise; a draw ``IDM + e`` is held for ``lag`` steps from the
    samples ``t = offset (mod lag)``) behind a leader with slow speed changes; ``a`` is the applied acceleration, so a
    window that ends at a draw shows the previous draw's residual exactly."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds / DT)) + 1
    out = []
    for i in range(n_events):
        v_l = 15.0 + 3.0 * np.sin(np.arange(n) * DT * 2 * np.pi / rng.uniform(20, 40) + rng.uniform(0, 6))
        x_l = 50.0 + np.concatenate([[0.0], np.cumsum(v_l[1:] * DT)])
        v = np.empty(n)
        x = np.empty(n)
        a = np.empty(n)
        v[0], x[0] = v_l[0], x_l[0] - float(equilibrium_gap(v_l[0], IDM))
        e = 0.0
        for t in range(n):
            if t == 0 or (t - offset) % lag == 0:
                e = rho * e + rng.normal(0.0, sigma)
                cmd = float(idm_accel(x_l[t] - x[t], v[t] - v_l[t], v[t], IDM)) + e
            a[t] = float(np.clip(cmd, A_MIN, A_MAX))
            if t + 1 < n:
                v[t + 1] = max(v[t] + a[t] * DT, 0.0)
                x[t + 1] = x[t] + v[t + 1] * DT
        a[-1] = a[-2]
        out.append(Event(event_id=f"ar{i}", dataset="ar", site="sim", follower_id=f"f{i}", leader_id=f"l{i}",
                         t=np.arange(n) * DT, s=x_l - x, dv=v - v_l, v=v, a=a, v_lead=v_l, x_lead=x_l, x_follower=x))  # fmt: skip
    return EventSet(out)


def test_idm_ar_noise_recovers_the_noise_of_its_own_law() -> None:
    events = ar_events(rho=0.8, sigma=0.15, n_events=40, seconds=60.0, seed=3)
    windows = make_windows(events, stride=2)
    model = IDMARNoise(idm=IDM.to_dict())
    log = model.fit(windows, windows)
    assert log["rho"] == pytest.approx(0.8, abs=0.03) and log["sigma_eta"] == pytest.approx(0.15, rel=0.05)
    assert log["lag"] == 1 and log["conditional_rms"] == pytest.approx(log["sigma_eta"], rel=0.05)
    # the previous residual of the window is the AR state of the generating process
    x = windows.X.astype(np.float64)
    resid = windows.y - idm_accel(x[:, -1, 0], x[:, -1, 1], x[:, -1, 2], IDM)
    assert np.corrcoef(resid, previous_residual(x, IDM, 1))[0, 1] == pytest.approx(0.8, abs=0.03)
    # target mean_1s: lag 10, the previous target of the window is the target ten samples earlier
    # (windows at the draws: t = 29, 39, ... with draws held from t = 9 (mod 10); then y_t - IDM(x_t) is the AR state
    # of the draw and the window's previous residual is the state of the draw before)
    held = ar_events(rho=0.6, sigma=0.1, n_events=60, seconds=60.0, seed=4, lag=MEAN_TARGET_STEPS, offset=HIST_LEN - 1)
    w10 = make_windows(held, stride=MEAN_TARGET_STEPS, target="mean_1s")
    m10 = IDMARNoise(idm=IDM.to_dict(), target="mean_1s")
    log10 = m10.fit(w10, None)
    assert m10.lag == 10 and log10["rho"] == pytest.approx(0.6, abs=0.05) and log10["sigma_eta"] == pytest.approx(0.1, rel=0.1)
    frame, _ = windows_table(held, w10, None, target="mean_1s")  # the previous target of the window is y_prev
    x10 = w10.X.astype(np.float64)
    np.testing.assert_allclose(previous_residual(x10, IDM, 10) + idm_accel(x10[:, -11, 0], x10[:, -11, 1], x10[:, -11, 2], IDM),
                               frame["y_prev"].to_numpy(np.float64), atol=1e-4)


def test_idm_ar_noise_at_rho_zero_is_idm_white_noise() -> None:
    events = ar_events(rho=0.0, sigma=0.2, n_events=10, seconds=30.0, seed=5)
    windows = make_windows(events, stride=3)
    ar = IDMARNoise(idm=IDM.to_dict(), rho=0.0, sigma=0.37)
    wn = IDMWhiteNoise(idm=IDM.to_dict(), sigma=0.37)
    a, b = ar.predict_dist(windows.X), wn.predict_dist(windows.X)
    np.testing.assert_allclose(a.mean(), b.mean(), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(a.var(), b.var())
    np.testing.assert_allclose(a.log_prob(windows.y), b.log_prob(windows.y), atol=1e-12)
    assert a.aleatoric_var is not None and np.allclose(a.aleatoric_var, 0.37**2) and np.all(a.epistemic_var == 0)
    fit_ar, fit_wn = IDMARNoise(idm=IDM.to_dict(), rho=0.0), IDMWhiteNoise(idm=IDM.to_dict())
    fit_ar.fit(windows)
    fit_wn.fit(windows)
    assert fit_ar.sigma == pytest.approx(fit_wn.sigma, rel=1e-12)  # sigma_eta = the RMS residual at rho = 0


def test_idm_ar_noise_rollouts_have_the_ar_autocorrelation() -> None:
    rho, sigma = 0.9, 0.08
    model = IDMARNoise(idm=IDM.to_dict(), rho=rho, sigma=sigma)
    n = 900  # 90 s at a constant leader speed, the follower at the IDM equilibrium
    v_l = np.full(n, 20.0)
    x_l = 100.0 + np.arange(n) * DT * 20.0
    gap = float(equilibrium_gap(20.0, IDM))
    ev = Event(event_id="cruise", dataset="t", site="t", follower_id="f", leader_id="l", t=np.arange(n) * DT,
               s=np.full(n, gap), dv=np.zeros(n), v=v_l.copy(), a=np.zeros(n), v_lead=v_l, x_lead=x_l, x_follower=x_l - gap)  # fmt: skip
    cfg = RolloutConfig(n_rollouts=200, horizon_s=80.0, horizons_s=(80.0,), stream_events=1)
    res = rollout_events(model, [ev], cfg=cfg, keep_paths=True)
    v, s = res.v_path[0], res.s_path[0]  # (R, K + 1)
    acc = np.diff(v, axis=1) / DT  # applied accelerations (no clipping, no standstill at these speeds)
    dv = v - 20.0
    e = acc - idm_accel(s[:, :-1], dv[:, :-1], v[:, :-1], IDM)  # the AR noise of every step
    e = e[:, 100:]  # the process starts at the residual of the steady history (0): skip 10 s
    lag1 = np.sum(e[:, 1:] * e[:, :-1]) / np.sqrt(np.sum(e[:, 1:] ** 2) * np.sum(e[:, :-1] ** 2))
    assert lag1 == pytest.approx(rho, abs=0.02)
    assert e.std() == pytest.approx(sigma / np.sqrt(1 - rho**2), rel=0.08)
    eta = e[:, 1:] - rho * e[:, :-1]
    assert eta.std() == pytest.approx(sigma, rel=0.03) and abs(np.corrcoef(eta[:, 1:].ravel(), eta[:, :-1].ravel())[0, 1]) < 0.03
    # both sampler modes are the same sampler; the state starts from the history and is carried
    hist = torch.as_tensor(np.stack([ev.s, ev.dv, ev.v], axis=1)[:HIST_LEN][None].repeat(4, axis=0), dtype=torch.float32)
    draws = []
    for mode in ("per_step", "per_trajectory"):
        gen = torch.Generator().manual_seed(7)
        state = model.init_sampler(4, gen, mode)
        draws.append(torch.stack([model.sample_step(hist, state, gen) for _ in range(3)]))
        assert state["e"].shape == (4,)
    torch.testing.assert_close(draws[0], draws[1], rtol=0, atol=0)


def test_idm_ar_noise_round_trip_and_target_injection(tmp_path: Path) -> None:
    events = ar_events(rho=0.5, sigma=0.1, n_events=8, seconds=20.0, seed=6)
    windows = make_windows(events, stride=2)
    model = IDMARNoise(idm=IDM.to_dict())
    model.fit(windows)
    loaded = load_model(model.save(tmp_path / "ar.pt"))
    assert isinstance(loaded, IDMARNoise) and loaded.rho == model.rho and loaded.sigma == model.sigma
    np.testing.assert_array_equal(loaded.predict_dist(windows.X).mean(), model.predict_dist(windows.X).mean())
    with pytest.raises(ValueError):
        IDMARNoise(idm=IDM.to_dict(), target="b")
    cfg = e1_mod.load_config()
    calibration = lambda name: (IDM, {"name": name})  # noqa: E731
    assert e1_mod.model_spec(cfg, "idm_ar_noise", calibration).kwargs["target"] == "a"
    cfg.train.target = "mean_1s"
    assert e1_mod.model_spec(cfg, "idm_ar_noise", calibration).kwargs["target"] == "mean_1s"
    for label in ("idm_white_noise", "lstm_const_sigma", "quantile_pidl", "diffusion_cf"):  # the original kwargs unchanged
        assert "target" not in e1_mod.model_spec(cfg, label, calibration).kwargs


# =============================================================================================== R13
@pytest.fixture(scope="module")
def gap_noise():  # noqa: ANN201
    """Synthetic events with strongly gap-dependent noise, sigma(s) = 0.4 (1 + 3 exp(-s / 15)), stride 2."""
    from cf_unc.data.pipeline import synthetic_bundle

    cfg = dict(n_drivers=40, events_per_driver=2, duration_s=20.0, noise_kind="gap", noise_sigma=0.4, noise_kappa=3.0,
               noise_s_scale=15.0, seed=22)  # fmt: skip
    return synthetic_bundle(cfg, name="gap", stride=2)


@pytest.fixture(scope="module")
def frozen_setup(tmp_path_factory: pytest.TempPathFactory, gap_noise) -> dict[str, Any]:
    """A small LSTMConstSigma stored as <dir>/seed0 (its mean is frozen) and the frozen-mean model fitted on top."""
    base = tmp_path_factory.mktemp("frozen")
    train, val = gap_noise.windows["train"], gap_noise.windows["val"]
    const = LSTMConstSigma(hidden=32, head_hidden=32, train={"max_epochs": 3, "batch_size": 256})
    const.fit(train, val, seed=0)
    const.save(base / "lc" / "seed0" / "model.pt")
    (base / "lc" / "seed0" / "train_log.json").write_text(json.dumps({"target": "a", "kwargs": const.init_kwargs}), encoding="utf-8")
    frozen = LSTMHeteroFrozen(mean_from=str(base / "lc"), head_hidden=32, train={"max_epochs": 5, "batch_size": 256})
    log = frozen.fit(train, val, seed=0)
    return {"base": base, "const": const, "frozen": frozen, "log": log, "test": gap_noise.windows["test"]}


def test_frozen_mean_is_the_constant_sigma_mean(frozen_setup: dict[str, Any]) -> None:
    const, frozen, test = frozen_setup["const"], frozen_setup["frozen"], frozen_setup["test"]
    a, b = frozen.predict_dist(test.X), const.predict_dist(test.X)
    np.testing.assert_allclose(a.mean(), b.mean(), rtol=0, atol=1e-6)  # one mean
    sd = np.sqrt(a.var())
    assert sd.std() > 0.01  # a state-dependent sigma ...
    small, large = test.X[:, -1, 0] < 15.0, test.X[:, -1, 0] > 40.0
    assert sd[small].mean() > sd[large].mean()  # ... that follows the gap-dependent noise of the data
    assert np.all(a.epistemic_var == 0) and np.allclose(a.aleatoric_var, a.var())
    assert -np.mean(a.log_prob(test.y)) < -np.mean(b.log_prob(test.y))  # better NLL than the constant sigma
    log = frozen_setup["log"]
    assert log["mean_from"]["path"].endswith("lc/seed0/model.pt") and len(log["mean_from"]["sha256"]) == 64
    assert {p.data_ptr() for p in frozen.mean_model.net.parameters()} and not any(
        p.requires_grad for p in frozen.mean_model.net.parameters())


def test_frozen_mean_round_trip_and_checks(tmp_path: Path, frozen_setup: dict[str, Any], gap_noise) -> None:
    frozen, test = frozen_setup["frozen"], frozen_setup["test"]
    loaded = load_model(frozen.save(tmp_path / "f.pt"))
    assert isinstance(loaded, LSTMHeteroFrozen)
    np.testing.assert_array_equal(loaded.predict_dist(test.X[:300]).mean(), frozen.predict_dist(test.X[:300]).mean())
    np.testing.assert_array_equal(loaded.predict_dist(test.X[:300]).var(), frozen.predict_dist(test.X[:300]).var())
    gen = torch.Generator().manual_seed(0)
    draw = loaded.sample_step(torch.as_tensor(test.X[:5]), loaded.init_sampler(5, gen), gen)
    assert draw.shape == (5,) and torch.isfinite(draw).all()
    train = gap_noise.windows["train"].subset(np.arange(500))
    with pytest.raises(ValueError, match="target"):
        LSTMHeteroFrozen(mean_from=str(frozen_setup["base"] / "lc"), target="mean_1s").fit(train, None, seed=0)
    with pytest.raises(FileNotFoundError):
        LSTMHeteroFrozen(mean_from=str(frozen_setup["base"] / "lc")).fit(train, None, seed=3)  # no seed-3 source


# =============================================================================================== R3 (E3 runner)
@pytest.fixture(scope="module")
def e3_world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    helpers = _load_test_module("test_e3_pooled")
    from dataclasses import asdict

    from cf_unc.platoon.openacc import load_runs
    from cf_unc.platoon.profiles import profiles_digest, select_profiles, write_profiles

    base = tmp_path_factory.mktemp("e3_ctrl")
    raw = base / "raw"
    helpers.write_openacc(raw / "AstaZero" / "acc_run.csv", helpers.platoon_speeds([4.0, 3.0], seed=1), acc=1)
    profiles, counts = select_profiles(load_runs(raw, ("AstaZero",)), helpers.RULES)
    rules = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(helpers.RULES).items()}
    write_profiles(base / "profiles.yaml", profiles, {"description": "test", "source": str(raw), "rules": rules,
                                                     "digest": profiles_digest(profiles), "counts": counts})  # fmt: skip
    (base / "idm.json").write_text(json.dumps({"calibrations": {"closed_loop": {"params": helpers.IDM}}}), encoding="utf-8")
    for root, labels in (("ckpt_a", ("idm_white_noise", "idm_ar_noise")), ("pool_a", ("idm_white_noise", "idm_ar_noise")),
                         ("ctrl", ("control",))):  # fmt: skip
        for label in labels:
            if label == "idm_ar_noise":
                model = IDMARNoise(idm=helpers.IDM, rho=0.9, sigma=0.1)
            else:
                model = IDMWhiteNoise(idm=helpers.IDM, sigma=0.3 if label == "idm_white_noise" else 0.6)
            model.save(base / root / label / "seed0" / "model.pt")
    return {"base": base, "profiles": base / "profiles.yaml"}


def e3_config(root: Path, world: dict[str, Any]) -> DictConfig:
    from cf_unc.experiments.e3 import load_config as load_e3

    base = world["base"]
    cfg = load_e3()
    cfg.root = str(root)
    cfg.device = "cpu"
    cfg.checkpoints = OmegaConf.create({"a": str(base / "ckpt_a")})
    cfg.variants = OmegaConf.create({
        "pool_w11": {"a": str(base / "pool_a")},
        "solo": {"a": str(base / "ckpt_a"), "only_models": ["idm_white_noise"]},
    })  # fmt: skip
    cfg.models = OmegaConf.create({
        "idm_white_noise": {"target": "a", "sampler_mode": "per_step", "hold_steps": 1},
        "idm_ar_noise": {"target": "a", "sampler_mode": "per_step", "hold_steps": 1, "variants": ["raw", "pool_w11", "solo"]},
        "control": {"target": "a", "sampler_mode": "per_step", "hold_steps": 1, "checkpoints": str(base / "ctrl"),
                    "variants": ["raw"]},
    })  # fmt: skip
    cfg.seeds = [0]
    cfg.platoon = OmegaConf.create({"n_followers": 2, "n_rollouts": 3, "n_rollouts_sampled": 2, "history_s": 3.0,
                                    "burn_in_s": 2.0, "tail_s": 2.0, "max_batch": 1000, "seed": 0, "gap_cap": 120.0,
                                    "batch_profiles": 2})  # fmt: skip
    cfg.profiles = str(world["profiles"])
    cfg.idm = str(base / "idm.json")
    cfg.sets = OmegaConf.create({"headline": 1, "extended": None})
    return cfg


def test_e3_model_selection_and_checkpoint_override(tmp_path: Path, e3_world: dict[str, Any]) -> None:
    cfg = e3_config(tmp_path / "run", e3_world)
    plan = dry_run_e3(cfg, echo=False)
    assert not (tmp_path / "run").exists() and plan["conflicts"] == []
    n_p = plan["run_profiles"]
    models = {v: sorted(e["models"]) for v, e in plan["variants"].items()}
    assert models == {"raw": ["control", "idm_ar_noise", "idm_white_noise"], "pool_w11": ["idm_ar_noise", "idm_white_noise"],
                      "solo": ["idm_white_noise"]}  # fmt: skip
    assert plan["variants"]["raw"]["models"]["control"]["checkpoints"] == str(e3_world["base"] / "ctrl")
    assert plan["missing"] == 6 * n_p
    status = run_e3(cfg, echo=False)
    assert status["finished"] and status["failures"] == [] and status["done"] == 6 * n_p
    root = Path(cfg.root)
    spec = json.loads((root / "units" / "raw" / "control" / "spec.json").read_text(encoding="utf-8"))
    assert spec["checkpoints"] == str(e3_world["base"] / "ctrl") and "variants" not in spec
    plain = json.loads((root / "units" / "raw" / "idm_white_noise" / "spec.json").read_text(encoding="utf-8"))
    assert set(plain) == {"label", "model", "target", "checkpoints", "sampler_mode", "hold_steps"}  # as before 18
    assert not (root / "units" / "pool_w11" / "control").exists() and not (root / "units" / "solo" / "idm_ar_noise").exists()
    p = plan["variants"]["raw"]["models"]["idm_white_noise"]
    assert p["done"] == 0 and dry_run_e3(cfg, echo=False)["nothing_to_do"]
    # identical inputs give identical units: solo/idm_white_noise repeats raw/idm_white_noise bit for bit
    first = load_profile_ids(root)[0]
    pd.testing.assert_frame_equal(pd.read_parquet(unit_path(root, "solo", "idm_white_noise", 0, first)),
                                  pd.read_parquet(unit_path(root, "raw", "idm_white_noise", 0, first)))  # fmt: skip
    # bad selections are refused before anything is created
    bad = e3_config(tmp_path / "bad", e3_world)
    bad.variants.solo.only_models = ["nope"]
    with pytest.raises(ValueError, match="only_models"):
        E3Runner(bad, echo=False)
    bad = e3_config(tmp_path / "bad", e3_world)
    bad.models.control.colour = "red"
    with pytest.raises(ValueError, match="unknown keys"):
        E3Runner(bad, echo=False)
    assert not (tmp_path / "bad").exists()
    # the CLI: --models drops the left-out labels from only_models too
    config_file = tmp_path / "e3_ctrl.yaml"
    OmegaConf.save(cfg, config_file)
    assert _load_script("run_e3").main(["--config", str(config_file), "--dry-run", "--quiet", "--models", "idm_ar_noise"]) == 0


def load_profile_ids(root: Path) -> list[str]:
    return sorted(p.stem for p in (root / "units" / "raw" / "idm_white_noise" / "seed0").glob("*.parquet"))


# =============================================================================================== E2 dry run
def test_e2_dry_run_writes_nothing(tmp_path: Path) -> None:
    events, _ = generate(dict(n_drivers=6, events_per_driver=2, duration_s=20.0, seed=3, dataset="d"))
    events.to_parquet(tmp_path / "ev")
    data = {"d": {"name": "d", "kind": "parquet", "path": str(tmp_path / "ev"), "unify_acceleration": True}}
    ckpt = tmp_path / "ckpt"
    IDMARNoise(idm=IDM.to_dict(), rho=0.5, sigma=0.2).save(ckpt / "idm_ar_noise" / "seed0" / "model.pt")
    cfg = e2_mod.load_config(EXP / "e2_ctrl.yaml")
    cfg.root = str(tmp_path / "e2")
    cfg.targets = OmegaConf.create({"a": {"checkpoints": str(ckpt), "hold_steps": 1}})
    cfg.models = ["idm_ar_noise", "diffusion_cf_hist"]
    cfg.seeds = [0, 1]
    cfg.datasets = OmegaConf.create({"x": {"data": "d", "split": "all", "n_events": 5}})
    cfg.data_configs = data
    cfg.train = {}
    plan = e2_mod.dry_run_e2(cfg, echo=False)
    assert not (tmp_path / "e2").exists()
    a = plan["targets"]["a"]
    assert a["active"] and a["spec"] == "new" and a["events"] == {"x": "new"} and plan["missing"] == 4 and plan["done"] == 0
    assert a["labels"]["idm_ar_noise"]["missing_checkpoints"] == [1]
    assert a["labels"]["diffusion_cf_hist"]["missing_checkpoints"] == [0, 1]
    cfg.targets.a.checkpoints = str(tmp_path / "none")
    plan = e2_mod.dry_run_e2(cfg, echo=False)
    assert not plan["targets"]["a"]["active"] and "a run skips this target" in plan["targets"]["a"]["skipped"]
    assert not (tmp_path / "e2").exists()


# =============================================================================================== R6
def test_causal_savgol_is_causal_and_exact_on_quadratics() -> None:
    from scipy.signal import savgol_coeffs

    for w in (5, 11, 31):
        np.testing.assert_allclose(causal_savgol_coeffs(w, 2), savgol_coeffs(w, 2, pos=w - 1, use="dot"), atol=1e-12)
    t = np.arange(150) * DT
    quad = 1.0 + 0.5 * t - 0.2 * t**2
    np.testing.assert_allclose(causal_savgol(quad, 21, 2), quad, atol=1e-9)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 150))
    y = x.copy()
    y[:, 80:] += 50.0  # the future changes ...
    np.testing.assert_array_equal(causal_savgol(x, 21, 2)[:, :80], causal_savgol(y, 21, 2)[:, :80])  # ... not the past
    assert causal_savgol(x, 21, 2)[0, 0] == x[0, 0]


def test_causal_events_and_recorded_targets() -> None:
    events, _ = generate(dict(n_drivers=4, events_per_driver=2, duration_s=30.0, seed=8))
    rng = np.random.default_rng(1)
    for ev in events:  # rough speeds, a recorded gap that stays above the drop threshold
        ev.v = np.maximum(ev.v + rng.normal(0, 0.2, len(ev)), 0.0)
        ev.dv = ev.v - ev.v_lead
        ev.x_follower = ev.x_lead - ev.s
    kept, report = causal_events(events, {"window": 31})
    assert report["filter"] == "causal" and report["n_out"] == len(events) and set(report["dropped"]) == set(DROP_REASONS)
    for raw, ev in zip(events, kept):
        ev.validate()
        np.testing.assert_allclose(ev.v, np.maximum(causal_savgol(raw.v, 31, 2), 0.0))
        assert ev.s.min() >= 0.5 and ev.meta["signal"] == "causal" and ev.meta["causal_window"] == 31
    recorded = with_recorded_targets(kept, events)
    assert all(isinstance(ev, TargetSpeedEvent) for ev in recorded)
    w = make_windows(recorded, stride=7, target="mean_1s")
    raw_w = make_windows(events, stride=7, target="mean_1s")
    proc_w = make_windows(kept, stride=7, target="mean_1s")
    np.testing.assert_array_equal(w.y, raw_w.y)  # the target of the recorded speed ...
    np.testing.assert_array_equal(w.X, proc_w.X)  # ... with the processed inputs
    assert not np.allclose(w.y, proc_w.y)
    frame, _ = windows_table(recorded, w, None, target="mean_1s")
    ev0 = events[0]
    t = int(w.t_index[0])
    assert frame["y_prev"].iloc[0] == pytest.approx((ev0.v[t] - ev0.v[t - 10]) / (10 * DT), abs=1e-6)
    assert target_speed(recorded[0]) is recorded[0].v_target and target_speed(events[0]) is events[0].v


def test_causal_signal_keys_and_matching_kind(tmp_path: Path) -> None:
    events, _ = generate(dict(n_drivers=10, events_per_driver=2, duration_s=12.0, seed=1, dataset="hd"))
    events.to_parquet(tmp_path / "hd")
    folds = {ev.event_id: int(ev.follower_id[3:]) % N_FOLDS for ev in events}
    (tmp_path / "hd_driver.json").write_text(json.dumps({"n_folds": N_FOLDS, "folds": folds}), encoding="utf-8")
    data = {"hd": {"name": "hd", "kind": "parquet", "path": str(tmp_path / "hd"), "unify_acceleration": True,
                   "split": {"file": str(tmp_path / "hd_driver.json"), "fold": 0}}}  # fmt: skip
    match = MatchConfig(reference="hd", grid=(11, 21), tolerance=0.1)
    content = {"harmonise": HarmoniseConfig().to_dict(), **match.settings(), "windows": {"hd": 11}, "datasets": {}}
    centred, causal = tmp_path / "centred.json", tmp_path / "causal.json"
    centred.write_text(json.dumps(content), encoding="utf-8")
    causal.write_text(json.dumps({**content, "filter": "causal", "part": "val"}), encoding="utf-8")
    assert matched_windows(centred, None, match) == {"hd": 11}
    assert matched_windows(causal, None, match, variant="matched_causal") == {"hd": 11}
    with pytest.raises(MatchingError, match="kind"):
        matched_windows(centred, None, match, variant="matched_causal")
    with pytest.raises(MatchingError, match="kind"):
        matched_windows(causal, None, match)
    cfg = e1_mod.load_config()
    cfg.data_configs = data
    cfg.signal = "matched_causal"
    cfg.match = {"reference": "hd", "grid": [11, 21], "tolerance": 0.1, "file": str(causal)}
    cfg.train = {"data": "hd", "target": "mean_1s", "target_from": "recorded"}
    key = e1_mod.signal_key(cfg, "hd")
    assert key == {"signal": "causal", "harmonise": {**HarmoniseConfig().to_dict(), "window": 11}, "target_from": "recorded"}
    harmonised = OmegaConf.merge(cfg, {"signal": "harmonised", "train": {"target_from": "signal"}})
    assert e1_mod.signal_key(harmonised, "hd") == {"signal": "harmonised", "harmonise": HarmoniseConfig().to_dict()}  # unchanged
    assert e1_mod.training_signal_key(cfg) != e1_mod.training_signal_key(harmonised)  # no reuse across the two
    with pytest.raises(ValueError, match="speed-based target"):
        e1_mod.signal_key(OmegaConf.merge(cfg, {"train": {"target": "a"}}), "hd")
    with pytest.raises(ValueError, match="processed signal"):
        e1_mod.signal_key(OmegaConf.merge(cfg, {"signal": "raw"}), "hd")
    d = e1_mod.Data(cfg)
    loaded = d.events("hd")
    assert all(isinstance(ev, TargetSpeedEvent) for ev in loaded) and loaded[0].meta["signal"] == "causal"
    rest = d.split("hd", "all_but_val")
    val = {ev.event_id for ev in d.split("hd", "val")}
    assert len(rest) + len(val) == len(loaded) and not val & {ev.event_id for ev in rest}
    assert [ev.event_id for ev in rest] == [ev.event_id for ev in loaded if ev.event_id not in val]  # data set order
    with pytest.raises(ValueError, match="raw signal"):
        load_events(tmp_path / "hd", target_from="recorded")


def test_causal_matching_uses_the_validation_folds_only(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    sets = {}
    for name, noise, seed in (("hd", 0.02, 1), ("rough", 0.3, 2)):
        events, _ = generate(dict(n_drivers=20, events_per_driver=1, duration_s=20.0, seed=seed, dataset=name))
        folds = {ev.event_id: int(ev.follower_id[3:]) % N_FOLDS for ev in events}
        for ev in events:  # validation fold (1): rough as given; every other fold: far rougher (must not matter)
            scale = noise if folds[ev.event_id] == 1 else 5 * noise
            ev.v = np.maximum(ev.v + rng.normal(0, scale, len(ev)), 0.0)
            ev.dv = ev.v - ev.v_lead
            ev.x_follower = ev.x_lead - ev.s
        events.to_parquet(tmp_path / name)
        (tmp_path / f"{name}_driver.json").write_text(json.dumps({"n_folds": N_FOLDS, "folds": folds}), encoding="utf-8")
        sets[name] = {"name": name, "kind": "parquet", "path": str(tmp_path / name), "unify_acceleration": True,
                      "split": {"file": str(tmp_path / f"{name}_driver.json"), "fold": 0}}  # fmt: skip
    cfg = e1_mod.load_config(EXP / "e1m_causal.yaml")
    cfg.root = str(tmp_path / "root")
    cfg.data_configs = sets
    cfg.config_paths.data = str(tmp_path)
    cfg.train.data = "hd"
    cfg.match.reference = "hd"
    cfg.match.file = str(tmp_path / "root" / "matching.json")
    cfg.eval.datasets = OmegaConf.create({"hd": {"data": "hd", "split": "test"}, "rough": {"data": "rough", "split": "all_but_val"}})
    config_file = tmp_path / "causal.yaml"
    OmegaConf.save(cfg, config_file)
    matcher = _load_script("match_roughness")
    assert matcher.main(["--config", str(config_file), "--causal"]) == 0
    content = json.loads(Path(cfg.match.file).read_text(encoding="utf-8"))
    assert content["filter"] == "causal" and content["part"] == "val" and content["windows"]["hd"] == 11
    assert content["datasets"]["rough"]["fold"] == 1 and content["datasets"]["rough"]["n_events"] == 4  # 20 drivers, 5 folds
    # the roughness of the matched set is computed on its validation events alone
    d = e1_mod.Data(OmegaConf.merge(cfg, {"signal": "raw", "train": {"target_from": "signal"}}))
    w = content["windows"]["rough"]
    kept, _ = causal_events(d.split("rough", "val"), {"window": w})
    assert content["datasets"]["rough"]["roughness"] == pytest.approx(matcher.roughness(kept)["v"], rel=1e-12)
    with pytest.raises(SystemExit):
        matcher.main(["--config", str(config_file)])  # a causal config needs --causal
    assert e1_mod.check_matching(cfg) == {"hd": 11, "rough": w}


# =============================================================================================== R14, H3.3 option
def test_bandwidth_factor_of_the_kde() -> None:
    rng = np.random.default_rng(0)
    dist = SampleDist(rng.normal(size=(50, 32)))
    y = rng.normal(size=50)
    np.testing.assert_array_equal(dist.log_prob(y, bandwidth_factor=1.0), dist.log_prob(y))  # default bit-identical
    np.testing.assert_allclose(dist.bandwidth(2.0), 2.0 * dist.bandwidth(), rtol=1e-12)
    assert not np.allclose(dist.log_prob(y, bandwidth_factor=0.5), dist.log_prob(y))
    assert np.all(SampleDist(np.zeros((3, 8))).bandwidth(0.5) == 1e-3)  # the floor stays


def test_density_sensitivity_end_to_end(tmp_path: Path) -> None:
    helpers = _load_test_module("test_e1_runner")
    root = tmp_path / "events"
    root.mkdir()
    data = {"hd": helpers._write_set(root, "hd", seed=1, n_drivers=20, per=2, duration=14.0)}
    cfg = e1_mod.load_config()
    cfg.root = str(tmp_path / "e1")
    cfg.device = "cpu"
    cfg.models = ["diffusion_cf"]
    cfg.seeds = [0]
    cfg.train = {"data": "hd", "stride": 2, "seq_stride": 5, "overrides": {"max_epochs": 1, "batch_size": 128}}
    cfg.model_overrides = {"diffusion_cf": {k: v for k, v in TINY_DIFF.items() if k != "n_samples"}}
    cfg.eval.datasets = OmegaConf.create({"highd": {"data": "hd", "split": "test", "stride": 2}})
    cfg.eval.max_rows_sampled = 40
    cfg.data_configs = data
    cfg.idm = {}
    cfg.idm_reference = None
    status = e1_mod.run_e1(cfg, echo=False)
    assert status["failures"] == [] and status["tables"]["done"] == 1
    config_file = tmp_path / "e1_test.yaml"
    OmegaConf.save(cfg, config_file)
    script = _load_script("density_sensitivity")
    args = ["--root", str(tmp_path / "e1"), "--config", str(config_file), "--models", "diffusion_cf", "--datasets", "highd",
            "--samples", "4", "8", "--bandwidths", "0.5", "1", "2", "--device", "cpu"]  # fmt: skip
    assert script.main([*args, "--dry-run"]) == 0 and not (tmp_path / "e1" / "tables").exists()
    assert script.main(args) == 0
    out = pd.read_csv(tmp_path / "e1" / "tables" / "density_sensitivity.csv")
    assert len(out) == 6 and set(out["n_samples"]) == {4, 8} and set(out["bandwidth_factor"]) == {0.5, 1.0, 2.0}
    assert (out["n_rows"] == 40).all() and np.isfinite(out[["nll", "crps", "crps_fair", "picp90", "width90", "stored_nll"]]).all().all()
    for _, grp in out.groupby("n_samples"):  # CRPS and the interval do not depend on the bandwidth
        assert grp["crps"].nunique() == 1 and grp["picp90"].nunique() == 1
    assert script.main([*args[:-2], "--device", "cpu", "--datasets", "nope", "--dry-run"]) == 1


def test_h33_pairs_option() -> None:
    from cf_unc.maps.analysis import H33_MODELS, format_report

    def rows(a: str, b: str) -> pd.DataFrame:
        return pd.DataFrame({"dataset": ["highd"], "subset": ["all"], "group": ["all"], "n_windows": [10], "nll_a": [1.0],
                             "nll_b": [2.0], "diff": [-1.0], "diff_lo": [-1.5], "diff_hi": [-0.5], "p_holm": [0.01],
                             "ci_overlap": [False], "model_a": [a], "model_b": [b]})  # fmt: skip

    single = format_report({"h33_nll": rows(*H33_MODELS)})
    assert single.count("H3.3 NLL, hetero_mdn minus lstm_const_sigma") == 1  # the default title
    both = format_report({"h33_nll": pd.concat([rows(*H33_MODELS), rows("lstm_hetero_frozen", "lstm_const_sigma")])})
    assert "H3.3 NLL, lstm_hetero_frozen minus lstm_const_sigma" in both and both.count("H3.3 NLL,") == 2
    assert format_report({}).endswith("(not available)")
    script = _load_script("analyze_e1")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--h33-pair", nargs=2, action="append", default=None)
    assert parser.parse_args(["--h33-pair", "a", "b", "--h33-pair", "c", "d"]).h33_pair == [["a", "b"], ["c", "d"]]
    assert "--h33-pair" in Path(script.__file__).read_text(encoding="utf-8")
