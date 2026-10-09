"""IDM law, closed-loop simulation and calibration (docs/design.md, section 4)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from cf_unc.constants import A_MAX, A_MIN, DT, S_EPS
from cf_unc.data.synthetic import SyntheticConfig, generate
from cf_unc.physics.idm import (
    IDM_BOUNDS,
    PARAM_NAMES,
    IDMParams,
    calibrate_idm,
    equilibrium_gap,
    evaluate_idm,
    idm_accel,
    leader_positions,
    simulate_follower,
)

TRUE = IDMParams(v0=30.0, T=1.5, s0=2.0, a_max=1.2, b=1.8)
WRONG = IDMParams(v0=20.0, T=2.5, s0=4.0, a_max=0.6, b=3.0)
REPORT_KEYS = (
    "objective", "value", "spacing_rmse", "accel_rmse", "accel_resid_std", "n_events", "n_collisions", "seconds"
)


def random_states(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = np.concatenate([rng.uniform(0.01, 0.5, n // 10), rng.uniform(0.5, 150.0, n - n // 10)])
    return s, rng.uniform(-8.0, 8.0, n), rng.uniform(0.0, 40.0, n)


# ----------------------------------------------------------------------------- parameters
def test_params_dict_and_vector_round_trip():
    d = TRUE.to_dict()
    assert d == {"v0": 30.0, "T": 1.5, "s0": 2.0, "a_max": 1.2, "b": 1.8, "delta": 4.0}
    assert json.loads(json.dumps(d)) == d
    assert IDMParams.from_dict(d) == TRUE
    vec = TRUE.as_vector()
    assert vec.shape == (5,) and vec.dtype == np.float64
    np.testing.assert_array_equal(vec, [30.0, 1.5, 2.0, 1.2, 1.8])
    back = IDMParams.from_vector(vec, delta=4.0)
    assert back == TRUE and all(type(getattr(back, f)) is float for f in (*PARAM_NAMES, "delta"))
    batch = IDMParams.from_vector(np.stack([vec, 2 * vec]))
    assert batch.T.shape == (2,)
    np.testing.assert_array_equal(batch.as_vector(), np.stack([vec, 2 * vec]))
    assert IDMParams.from_dict({k: v for k, v in d.items() if k != "delta"}).delta == 4.0


def test_params_validation():
    with pytest.raises(ValueError, match="unknown"):
        IDMParams.from_dict({**TRUE.to_dict(), "a": 1.0})
    with pytest.raises(ValueError, match="missing"):
        IDMParams.from_dict({"v0": 30.0, "T": 1.5})
    with pytest.raises(ValueError, match="positive"):
        IDMParams(v0=30.0, T=-1.0, s0=2.0, a_max=1.2, b=1.8)
    with pytest.raises(ValueError):
        IDMParams.from_vector([1.0, 2.0, 3.0])
    assert set(IDM_BOUNDS) == set(PARAM_NAMES)


# ----------------------------------------------------------------------------- IDM law
def test_equilibrium_gap_gives_zero_acceleration():
    v = np.linspace(0.0, 29.5, 60)
    s_eq = equilibrium_gap(v, TRUE)
    np.testing.assert_allclose(idm_accel(s_eq, np.zeros_like(v), v, TRUE), 0.0, atol=1e-9)
    assert equilibrium_gap(0.0, TRUE) == pytest.approx(TRUE.s0)
    assert equilibrium_gap(10.0, TRUE) == pytest.approx((2.0 + 15.0) / math.sqrt(1.0 - (10.0 / 30.0) ** 4))
    assert np.all(np.isinf(equilibrium_gap(np.array([30.0, 35.0]), TRUE)))
    s_eq_t = equilibrium_gap(torch.tensor(v), TRUE)
    np.testing.assert_allclose(s_eq_t.numpy(), s_eq, rtol=1e-12)
    # heterogeneous drivers: every driver is in equilibrium at its own gap
    per = IDMParams(v0=np.array([25.0, 30.0, 35.0]), T=np.array([1.0, 1.5, 2.0]), s0=2.0, a_max=1.0, b=1.5)
    np.testing.assert_allclose(idm_accel(equilibrium_gap(15.0, per), 0.0, 15.0, per), 0.0, atol=1e-9)


def test_numpy_and_torch_agree():
    s, dv, v = random_states()
    ref = idm_accel(s, dv, v, TRUE)
    assert isinstance(ref, np.ndarray) and ref.dtype == np.float64
    out64 = idm_accel(torch.from_numpy(s), torch.from_numpy(dv), torch.from_numpy(v), TRUE)
    assert out64.dtype == torch.float64
    np.testing.assert_allclose(out64.numpy(), ref, rtol=0, atol=1e-12)
    t32 = [torch.from_numpy(x.astype(np.float32)) for x in (s, dv, v)]
    out32 = idm_accel(*t32, TRUE)
    assert out32.dtype == torch.float32 and out32.device.type == "cpu"
    np.testing.assert_allclose(out32.numpy().astype(np.float64), ref, rtol=0, atol=1e-5)
    # mixed inputs (tensor + numpy / python scalars) and tensor-valued parameters
    mixed = idm_accel(t32[0], dv.astype(np.float32), 10.0, IDMParams.from_vector(torch.tensor(TRUE.as_vector())))
    assert mixed.dtype == torch.float32
    np.testing.assert_allclose(mixed.numpy(), idm_accel(s, dv, 10.0, TRUE), rtol=0, atol=1e-5)
    # numpy float32 stays float32 (python-float parameters do not upcast)
    assert idm_accel(s.astype(np.float32), dv.astype(np.float32), v.astype(np.float32), TRUE).dtype == np.float32


def test_torch_gradient():
    s = torch.tensor([20.0, 35.0, 60.0], dtype=torch.float64, requires_grad=True)
    dv = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([10.0, 15.0, 20.0], dtype=torch.float64, requires_grad=True)
    theta = torch.tensor(TRUE.as_vector(), requires_grad=True)
    out = idm_accel(s, dv, v, IDMParams.from_vector(theta))
    assert out.requires_grad
    out.sum().backward()
    for x in (s, dv, v, theta):
        assert x.grad is not None and torch.all(torch.isfinite(x.grad))
    assert torch.all(s.grad > 0) and torch.all(dv.grad < 0)

    def fn(s_, dv_, v_, th_):
        return idm_accel(s_, dv_, v_, IDMParams.from_vector(th_))

    inputs = tuple(x.detach().clone().requires_grad_(True) for x in (s, dv, v, theta))
    assert torch.autograd.gradcheck(fn, inputs)


def test_clipping():
    near = idm_accel(np.array([0.05, 1.0]), np.array([5.0, 10.0]), np.array([25.0, 30.0]), TRUE)
    np.testing.assert_array_equal(near, [A_MIN, A_MIN])
    assert np.all(idm_accel(np.array([0.05, 1.0]), 5.0, 25.0, TRUE, clip=False) < A_MIN)
    strong = IDMParams(v0=30.0, T=1.5, s0=2.0, a_max=6.0, b=1.8)
    assert idm_accel(np.array([500.0]), 0.0, 0.0, strong)[0] == A_MAX
    both = idm_accel(torch.tensor([500.0, 0.05]), torch.tensor([0.0, 5.0]), torch.tensor([0.0, 25.0]), strong)
    assert both.tolist() == [A_MAX, A_MIN]
    s, dv, v = random_states(seed=1)
    out = idm_accel(s, dv, v, TRUE)
    assert out.min() >= A_MIN and out.max() <= A_MAX
    # the gap floor: every s below S_EPS behaves like S_EPS
    np.testing.assert_array_equal(idm_accel(np.array([0.0, S_EPS / 2]), 0.0, 1.0, TRUE, clip=False),
                                  idm_accel(np.array([S_EPS, S_EPS]), 0.0, 1.0, TRUE, clip=False))


def test_monotonicity():
    s = np.linspace(0.0, 200.0, 801)[:, None]
    dv = np.linspace(-10.0, 10.0, 81)[None, :]
    for v in (0.0, 5.0, 15.0, 28.0, 36.0):
        a = idm_accel(s, dv, v, TRUE)
        assert np.all(np.diff(a, axis=0) >= -1e-12), "not non-decreasing in s"
        assert np.all(np.diff(a, axis=1) <= 1e-12), "not non-increasing in dv"
    # strictly monotone away from the clips and the max(0, .) kink
    a = idm_accel(np.linspace(25.0, 80.0, 50), 1.0, 15.0, TRUE)
    assert A_MIN < a.min() and np.all(np.diff(a) > 0)
    a = idm_accel(30.0, np.linspace(0.0, 5.0, 50), 15.0, TRUE)
    assert np.all(np.diff(a) < 0)


def test_per_driver_parameters_broadcast():
    theta = np.array([[30.0, 1.5, 2.0, 1.2, 1.8], [25.0, 1.0, 3.0, 0.8, 2.5], [40.0, 2.2, 1.0, 2.0, 1.0]])
    per = IDMParams.from_vector(theta[:, None, :])  # fields (3, 1)
    s, dv, v = (x[:100] for x in random_states(seed=2))
    out = idm_accel(s[None, :], dv[None, :], v[None, :], per)
    assert out.shape == (3, 100)
    for i in range(3):
        np.testing.assert_array_equal(out[i], idm_accel(s, dv, v, IDMParams.from_vector(theta[i])))
    out_t = idm_accel(torch.from_numpy(s)[None, :], torch.from_numpy(dv)[None, :], torch.from_numpy(v)[None, :], per)
    np.testing.assert_allclose(out_t.numpy(), out, rtol=0, atol=1e-12)


# ----------------------------------------------------------------------------- simulation
@pytest.fixture(scope="module")
def noise_free():
    cfg = SyntheticConfig(n_drivers=20, events_per_driver=3, noise_sigma=0.0, seed=11)
    es, truth = generate(cfg)
    return cfg, es, truth


def test_simulate_follower_reproduces_noise_free_event(noise_free):
    cfg, es, _ = noise_free
    evs = list(es)[:12]
    names = ("s", "dv", "v", "a", "v_lead", "x_lead", "x_follower")
    stack = {name: np.stack([getattr(ev, name) for ev in evs]) for name in names}

    def accel(s, dv, v):
        return idm_accel(s, dv, v, cfg.idm)

    sim = simulate_follower(accel, stack["s"][:, 0], stack["v"][:, 0], stack["v_lead"])
    for key, name in (("s", "s"), ("dv", "dv"), ("v", "v"), ("a", "a"), ("x", "x_follower"), ("x_lead", "x_lead")):
        np.testing.assert_allclose(sim[key], stack[name], rtol=0, atol=1e-6, err_msg=key)
    np.testing.assert_array_equal(sim["collision_index"], -1)

    # given (shifted) leader positions: the follower starts at x_lead[0] - s_init
    x_lead = stack["x_lead"] + 1000.0
    shifted = simulate_follower(accel, stack["s"][:, 0], stack["v"][:, 0], stack["v_lead"], x_lead=x_lead)
    np.testing.assert_allclose(shifted["s"], stack["s"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(shifted["x"], stack["x_follower"] + 1000.0, rtol=0, atol=1e-6)

    # a single (T,) event
    one = simulate_follower(accel, evs[0].s[0], evs[0].v[0], evs[0].v_lead)
    np.testing.assert_allclose(one["s"], evs[0].s, rtol=0, atol=1e-6)
    assert one["collision_index"] == -1


def test_simulate_follower_padding(noise_free):
    cfg, es, _ = noise_free
    evs = list(es)[:3]
    n_valid = np.array([401, 250, 90])
    v_lead = np.full((3, 401), np.nan)
    for i, (ev, n) in enumerate(zip(evs, n_valid)):
        v_lead[i, :n] = ev.v_lead[:n]

    def accel(s, dv, v):
        return idm_accel(s, dv, v, cfg.idm)

    s0, v0 = np.array([ev.s[0] for ev in evs]), np.array([ev.v[0] for ev in evs])
    sim = simulate_follower(accel, s0, v0, v_lead, n_valid=n_valid)
    for i, (ev, n) in enumerate(zip(evs, n_valid)):
        np.testing.assert_allclose(sim["s"][i, :n], ev.s[:n], rtol=0, atol=1e-9)
        np.testing.assert_allclose(sim["a"][i, :n], ev.a[:n], rtol=0, atol=1e-9)
        assert np.all(np.isnan(sim["s"][i, n:])) and np.all(np.isnan(sim["v"][i, n:]))
    with pytest.raises(ValueError):
        simulate_follower(accel, s0, v0, v_lead, n_valid=np.array([401, 0, 5]))


def _reference_collision(s_init: float, v_init: float, v_lead: np.ndarray, accel: float, n_valid: int) -> int:
    """Scalar re-implementation of the integrator (design.md, section 1)."""
    x_lead, x, v = s_init, 0.0, v_init
    for t in range(n_valid):
        if t > 0:
            x_lead += v_lead[t] * DT
        if x_lead - x <= 0.0:
            return t
        v = max(v + min(max(accel, A_MIN), A_MAX) * DT, 0.0)
        x += v * DT
    return -1


def test_collision_index():
    n = 80
    t = np.arange(n) * DT
    braking = np.where(t < 1.0, 20.0, np.maximum(20.0 - 6.0 * (t - 1.0), 0.0))  # hard stop from 20 m/s
    v_lead = np.stack([braking, np.full(n, 30.0), braking])
    s_init, v_init = np.array([15.0, 50.0, 15.0]), np.array([20.0, 20.0, 20.0])
    n_valid = np.array([n, n, 12])

    def constant_push(s, dv, v):
        return np.full_like(v, 1.0)

    sim = simulate_follower(constant_push, s_init, v_init, v_lead, n_valid=n_valid)
    expected = [_reference_collision(s_init[i], v_init[i], v_lead[i], 1.0, int(n_valid[i])) for i in range(3)]
    assert expected[0] > 0 and expected[1] == -1 and expected[2] == -1
    np.testing.assert_array_equal(sim["collision_index"], expected)
    k = expected[0]
    assert sim["s"][0, k] <= 0.0 < sim["s"][0, k - 1]
    # no special handling after the collision: the gap keeps shrinking below zero
    assert np.all(np.diff(sim["s"][0, k:]) < 0.0) and sim["s"][0, -1] < -10.0
    np.testing.assert_allclose(sim["a"][0], 1.0)
    np.testing.assert_allclose(sim["v"][0, 1:], sim["v"][0, :-1] + DT * sim["a"][0, :-1], atol=1e-12)


def test_leader_positions_scheme():
    rng = np.random.default_rng(3)
    v = rng.uniform(0.0, 30.0, (4, 50))
    x = leader_positions(np.array([1.0, 2.0, 3.0, 4.0]), v)
    np.testing.assert_array_equal(x[:, 0], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(np.diff(x, axis=1), v[:, 1:] * DT, rtol=1e-12, atol=1e-12)


def test_speed_clip_records_effective_acceleration():
    sim = simulate_follower(lambda s, dv, v: np.full_like(v, -8.0), 50.0, 1.0, np.full(20, 10.0))
    np.testing.assert_allclose(sim["v"][1:], np.maximum(sim["v"][:-1] + sim["a"][:-1] * DT, 0.0), atol=1e-12)
    assert sim["a"][1] == pytest.approx(-sim["v"][1] / DT) and np.all(sim["v"][2:] == 0.0)


# ----------------------------------------------------------------------------- calibration
def test_calibrate_accel_recovers_noise_free_parameters(noise_free):
    cfg, es, _ = noise_free
    params, report = calibrate_idm(es, objective="accel", seed=0)
    for key in REPORT_KEYS:
        assert key in report
    json.dumps(report)
    true = cfg.idm
    for name in ("T", "s0", "a_max", "b"):
        assert getattr(params, name) == pytest.approx(getattr(true, name), rel=0.05), name
    assert params.v0 == pytest.approx(true.v0, rel=0.10)
    assert params.delta == 4.0
    assert report["objective"] == "accel" and report["n_events"] == len(es)
    assert report["value"] == pytest.approx(report["accel_rmse"] ** 2, rel=1e-9)
    assert report["accel_rmse"] < 0.02 and report["n_collisions"] == 0 and report["seconds"] > 0


def test_calibrate_spacing_on_noisy_data():
    cfg = SyntheticConfig(n_drivers=40, events_per_driver=2, noise_sigma=0.4, seed=5)
    es, _ = generate(cfg)
    params, report = calibrate_idm(es, objective="spacing", seed=1, polish_maxfev=150)
    wrong = evaluate_idm(es, WRONG)
    truth = evaluate_idm(es, cfg.idm)
    assert report["spacing_rmse"] < 0.5 * wrong["spacing_rmse"]
    assert report["value"] <= 1.01 * truth["spacing_rmse"]  # at least as good as the generating parameters
    assert params.T == pytest.approx(cfg.idm.T, rel=0.10)
    if report["n_collisions"] == 0:  # batched objective == simulate_follower-based statistics
        assert report["value"] == pytest.approx(report["spacing_rmse"], rel=1e-9)
    lo_hi = np.array([IDM_BOUNDS[name] for name in PARAM_NAMES])
    assert np.all(params.as_vector() >= lo_hi[:, 0]) and np.all(params.as_vector() <= lo_hi[:, 1])


def test_calibrate_options(noise_free):
    _, es, _ = noise_free
    params, report = calibrate_idm(
        es, objective="spacing_nrmse", max_events=10, maxiter=3, popsize=5, bounds={"T": (2.0, 3.0)}, polish=False
    )
    assert report["n_events"] == 10 and report["polish"] is None
    assert 2.0 <= params.T <= 3.0 and report["bounds"]["T"] == [2.0, 3.0]
    if report["n_collisions"] == 0:
        assert report["value"] == pytest.approx(report["spacing_nrmse"], rel=1e-9)
    with pytest.raises(ValueError, match="objective"):
        calibrate_idm(es, objective="speed")
    with pytest.raises(ValueError, match="bounds"):
        calibrate_idm(es, bounds={"T": (3.0, 2.0)})


@pytest.mark.slow
def test_calibrate_spacing_time_budget():
    """300 events of 40 s with the default settings must finish well within ~60 s."""
    cfg = SyntheticConfig(n_drivers=150, events_per_driver=2, noise_sigma=0.4, seed=21)
    es, _ = generate(cfg)
    params, report = calibrate_idm(es, objective="spacing")
    assert report["n_events"] == 300 and report["seconds"] < 60.0
    assert params.T == pytest.approx(cfg.idm.T, rel=0.05)
    assert params.s0 == pytest.approx(cfg.idm.s0, rel=0.20)
    assert report["n_collisions"] == 0
