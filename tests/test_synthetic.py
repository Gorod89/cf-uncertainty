"""StochasticIDM generator (docs/design.md, section 3.2)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from cf_unc.constants import A_MAX, A_MIN, DT
from cf_unc.data.schema import EventSet
from cf_unc.data.synthetic import LeaderConfig, SyntheticConfig, driver_parameters, generate, noise_std
from cf_unc.physics.idm import PARAM_NAMES, IDMParams, idm_accel

CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "data" / "synthetic.yaml"
SMALL = dict(n_drivers=12, events_per_driver=3, duration_s=20.0)


@pytest.fixture(scope="module")
def default_data():
    return generate(SyntheticConfig())


@pytest.fixture(scope="module")
def gap_data():
    return generate(SyntheticConfig(noise_kind="gap", seed=7))


def pooled(es: EventSet, name: str) -> np.ndarray:
    return np.concatenate([getattr(ev, name) for ev in es])


def clean_residuals(es: EventSet, truth: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """a - a_det, sigma_true and s on samples where neither the speed clip nor the acceleration clip acted."""
    a, s = pooled(es, "a"), pooled(es, "s")
    a_det = np.concatenate([truth[ev.event_id]["a_det"] for ev in es])
    sigma = np.concatenate([truth[ev.event_id]["sigma_true"] for ev in es])
    speed_clip = np.concatenate([ev.v + ev.a * DT <= 1e-9 for ev in es])
    accel_clip = np.isclose(a, A_MIN) | np.isclose(a, A_MAX) | np.isclose(a_det, A_MIN) | np.isclose(a_det, A_MAX)
    keep = ~(speed_clip | accel_clip)
    return (a - a_det)[keep], sigma[keep], s[keep]


def test_default_data_set(default_data):
    es, truth = default_data
    cfg = SyntheticConfig()
    es.validate()
    assert len(es) == 800 and set(truth) == {ev.event_id for ev in es}
    ev = es[5]  # driver 1, episode 1
    assert (ev.event_id, ev.follower_id, ev.leader_id) == ("synthetic_00001_01", "drv00001", "lead00001_1")
    assert (ev.dataset, ev.site) == ("synthetic", "sim")
    assert len(ev) == 401 and ev.duration == pytest.approx(40.0) and ev.t[0] == 0.0
    t = truth[ev.event_id]
    assert t["a_det"].shape == (401,) and t["sigma_true"].shape == (401,)
    assert t["idm"] == cfg.idm.to_dict() and ev.meta["idm"] == cfg.idm.to_dict()
    np.testing.assert_array_equal(t["sigma_true"], cfg.noise_sigma)
    # a_det is the noise-free (clipped) IDM acceleration at the recorded state
    np.testing.assert_allclose(t["a_det"], idm_accel(ev.s, ev.dv, ev.v, cfg.idm), rtol=0, atol=1e-12)


def test_kinematic_identities(default_data):
    es, _ = default_data
    lead = LeaderConfig()
    for ev in es:
        np.testing.assert_allclose(ev.v[1:], np.maximum(ev.v[:-1] + ev.a[:-1] * DT, 0.0), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(ev.s, ev.x_lead - ev.x_follower)
        np.testing.assert_array_equal(ev.dv, ev.v - ev.v_lead)
        np.testing.assert_allclose(np.diff(ev.x_follower), ev.v[1:] * DT, rtol=0, atol=1e-9)
        np.testing.assert_allclose(np.diff(ev.x_lead), ev.v_lead[1:] * DT, rtol=0, atol=1e-9)
        assert ev.x_follower[0] == 0.0 and ev.x_lead[0] == ev.s[0]
    assert pooled(es, "v").min() >= 0.0 and pooled(es, "v_lead").min() >= lead.v_min
    a = pooled(es, "a")
    assert a.min() >= A_MIN and a.max() <= A_MAX
    assert pooled(es, "s").min() >= SyntheticConfig().min_gap


def test_reproducible():
    cfg = SyntheticConfig(**SMALL, heterogeneity_cv=0.1, noise_kind="gap", seed=4)
    es1, truth1 = generate(cfg)
    es2, truth2 = generate(SyntheticConfig.from_dict(cfg.to_dict()))
    for e1, e2 in zip(es1, es2):
        assert e1.event_id == e2.event_id and e1.meta == e2.meta
        for name in ("s", "dv", "v", "a", "v_lead", "x_lead", "x_follower"):
            np.testing.assert_array_equal(getattr(e1, name), getattr(e2, name))
        for key in ("a_det", "sigma_true"):
            np.testing.assert_array_equal(truth1[e1.event_id][key], truth2[e2.event_id][key])
    es3, _ = generate(SyntheticConfig(**SMALL, heterogeneity_cv=0.1, noise_kind="gap", seed=5))
    assert not np.array_equal(pooled(es1, "a"), pooled(es3, "a"))
    assert not np.array_equal(pooled(es1, "v_lead"), pooled(es3, "v_lead"))


def test_residual_std_matches_noise_sigma(default_data):
    es, truth = default_data
    resid, sigma, _ = clean_residuals(es, truth)
    assert resid.size >= 50_000
    # sampling error of the std is ~0.13 % here, so 3 % leaves a margin of > 20 standard errors
    assert resid.std() == pytest.approx(0.4, rel=0.03)
    assert abs(resid.mean()) < 0.01
    np.testing.assert_array_equal(sigma, 0.4)


def test_noise_free_data_is_deterministic_idm():
    es, truth = generate(SyntheticConfig(**SMALL, noise_sigma=0.0))
    for ev in es:
        np.testing.assert_array_equal(ev.a, truth[ev.event_id]["a_det"])


def test_gap_dependent_noise(gap_data):
    es, truth = gap_data
    cfg = SyntheticConfig(noise_kind="gap")
    resid, sigma, s = clean_residuals(es, truth)
    np.testing.assert_allclose(sigma, noise_std(s, cfg), rtol=1e-12)
    near, far = s < 10.0, s > 40.0
    assert near.sum() > 5_000 and far.sum() > 5_000
    std_near, std_far = resid[near].std(), resid[far].std()
    assert std_near > 1.5 * std_far
    # each matches the RMS of the true sigma in its region
    assert std_near == pytest.approx(math.sqrt(np.mean(sigma[near] ** 2)), rel=0.05)
    assert std_far == pytest.approx(math.sqrt(np.mean(sigma[far] ** 2)), rel=0.05)


def test_regime_coverage(default_data):
    es, _ = default_data
    s, v, dv = pooled(es, "s"), pooled(es, "v"), pooled(es, "dv")
    assert np.mean((s < 15.0) & (v < 8.0)) >= 0.05  # congested, close following
    assert np.mean(v > 20.0) >= 0.15  # free flow
    assert np.mean(dv > 1.0) >= 0.03  # closing in
    assert np.mean(dv < -1.0) >= 0.03  # falling back
    v_lead = pooled(es, "v_lead")
    for lo, hi in LeaderConfig().level_ranges:  # every leader regime is visited
        assert np.mean((v_lead >= lo) & (v_lead < hi)) >= 0.15


def test_driver_heterogeneity():
    _, truth0 = generate(SyntheticConfig(**SMALL, heterogeneity_cv=0.0))
    assert all(t["idm"] == SyntheticConfig().idm.to_dict() for t in truth0.values())
    es, truth = generate(SyntheticConfig(**SMALL, heterogeneity_cv=0.2))
    by_driver: dict[str, list[dict]] = {}
    for ev in es:
        by_driver.setdefault(ev.follower_id, []).append(truth[ev.event_id]["idm"])
    assert len(by_driver) == SMALL["n_drivers"]
    for params in by_driver.values():  # one parameter set per driver
        assert all(p == params[0] for p in params)
    firsts = np.array([[p[0][name] for name in PARAM_NAMES] for p in by_driver.values()])
    assert np.all(np.ptp(firsts, axis=0) > 0.0)  # every parameter varies across drivers
    # the multipliers are lognormal with unit median and the requested CV
    cfg = SyntheticConfig(n_drivers=4000, heterogeneity_cv=0.2)
    rngs = [np.random.default_rng(c) for c in np.random.SeedSequence(0).spawn(cfg.n_drivers)]
    mult = driver_parameters(cfg, rngs) / cfg.idm.as_vector()
    assert np.all(mult > 0.0)
    np.testing.assert_allclose(np.median(mult, axis=0), 1.0, atol=0.02)
    np.testing.assert_allclose(mult.std(axis=0) / mult.mean(axis=0), 0.2, rtol=0.1)


def test_min_gap_regeneration():
    cfg = SyntheticConfig(
        n_drivers=30,
        events_per_driver=2,
        duration_s=20.0,
        noise_sigma=1.0,
        min_gap=2.9,
        init_gap_factor=(0.6, 0.8),
        leader={"v_min": 0.5, "level_ranges": [[0.5, 3.0]], "level_weights": [1.0]},
    )
    es, truth = generate(cfg)
    es.validate()
    rounds = np.array([ev.meta["regen_rounds"] for ev in es])
    assert np.any(rounds > 0)
    assert min(ev.s.min() for ev in es) >= cfg.min_gap
    es2, _ = generate(cfg)
    np.testing.assert_array_equal(pooled(es, "s"), pooled(es2, "s"))
    with pytest.raises(RuntimeError, match="min_gap"):
        harsh = dict(min_gap=2.9, leader=cfg.leader, init_gap_factor=(0.6, 0.8), noise_sigma=1.0)
        generate(SyntheticConfig(**SMALL, **harsh, max_regen_rounds=0))


def test_config_dicts_and_yaml():
    cfg = SyntheticConfig()
    d = cfg.to_dict()
    assert d["idm"] == {"v0": 30.0, "T": 1.5, "s0": 2.0, "a_max": 1.2, "b": 1.8, "delta": 4.0}
    assert d["leader"]["level_ranges"] == [[2.0, 8.0], [8.0, 20.0], [20.0, 30.0]]
    assert SyntheticConfig.from_dict(d) == cfg
    assert SyntheticConfig.from_dict(None) == cfg
    partial = SyntheticConfig.from_dict({"noise_kind": "gap", "idm": {"T": 1.2}, "leader": {"gain": 0.8}})
    assert partial.idm == IDMParams(v0=30.0, T=1.2, s0=2.0, a_max=1.2, b=1.8)
    assert partial.leader.gain == 0.8 and partial.leader.hold_mean_s == 12.0
    with pytest.raises(ValueError, match="unknown"):
        SyntheticConfig.from_dict({"n_driver": 3})
    with pytest.raises(ValueError, match="unknown"):
        SyntheticConfig.from_dict({"leader": {"speed": 3}})
    with pytest.raises(ValueError, match="noise_kind"):
        SyntheticConfig(noise_kind="white")

    raw = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    assert raw["name"] == "synthetic" and raw["kind"] == "synthetic"
    assert SyntheticConfig.from_dict(raw["generator"]) == cfg
    assert raw["generator"] == d

    from omegaconf import OmegaConf

    assert SyntheticConfig.from_dict(OmegaConf.create(raw)["generator"]) == cfg


def test_parquet_round_trip_of_generated_events(tmp_path):
    es, _ = generate(SyntheticConfig(**SMALL))
    es.to_parquet(tmp_path / "synthetic", manifest={"generator": SyntheticConfig(**SMALL).to_dict()})
    back = EventSet.from_parquet(tmp_path / "synthetic")
    back.validate()
    assert [ev.meta for ev in back] == [ev.meta for ev in es]
    np.testing.assert_array_equal(pooled(back, "a"), pooled(es, "a"))
