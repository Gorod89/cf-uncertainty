"""Platoon cascade (cf_unc.platoon.simulate, docs/design.md 13.2) with toy models; CPU, seconds."""

from __future__ import annotations

from typing import Any

from dataclasses import replace

import numpy as np
import pytest
import torch

from cf_unc.constants import A_MAX, A_MIN, DT, HIST_LEN, S_EPS
from cf_unc.models.base import CFModel
from cf_unc.physics.idm import IDMParams, equilibrium_gap, idm_accel, simulate_follower
from cf_unc.platoon.simulate import (
    PlatoonConfig,
    initial_gap,
    leader_timeline,
    simulate_platoon,
    steady_event,
    time_line_steps,
)
from cf_unc.rollout.engine import RolloutConfig, rollout_events

IDM = IDMParams(v0=30.0, T=1.5, s0=2.0, a_max=1.2, b=1.8)
SMALL = PlatoonConfig(n_followers=4, n_rollouts=3, burn_in_s=2.0, tail_s=3.0)


class _Toy(CFModel):
    """Stubs of the training / persistence API."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fitted = True

    def fit(self, train, val=None, *, seed=0, verbose=False):  # noqa: ANN001, ANN201
        return {}

    def predict_dist(self, state_history, *, v_lead_future=None, batch_size=8192):  # noqa: ANN001, ANN201
        raise NotImplementedError

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict) -> None:
        pass


class IDMStep(_Toy):
    """IDM acceleration of the last history sample (float64 arithmetic) plus optional white noise."""

    def __init__(self, params: IDMParams, sigma: float = 0.0) -> None:
        super().__init__(sigma=sigma)
        self.params, self.sigma = params, float(sigma)

    def sample_step(self, hist, state, generator):  # noqa: ANN001, ANN201
        last = hist[:, -1].to(torch.float64)
        a = idm_accel(last[:, 0], last[:, 1], last[:, 2], self.params)
        if self.sigma > 0.0:
            a = a + self.sigma * torch.randn(a.shape, generator=generator, dtype=a.dtype, device=a.device)
        return a


class FrozenOffset(IDMStep):
    """IDM plus a constant offset per trajectory drawn at init (``per_trajectory``), redrawn per step otherwise."""

    def init_sampler(self, n_traj, generator, mode="per_step"):  # noqa: ANN001, ANN201
        super().init_sampler(n_traj, generator, mode)
        return 0.5 * torch.randn(n_traj, generator=generator, dtype=torch.float64, device=generator.device)

    def sample_step(self, hist, state, generator):  # noqa: ANN001, ANN201
        return super().sample_step(hist, state, generator) + state


class ExactFollower(_Toy):
    """``a = -dv / DT``: the next speed is the leader speed of the current step (a one-step delay)."""

    def sample_step(self, hist, state, generator):  # noqa: ANN001, ANN201
        return -hist[:, -1, 1].to(torch.float64) / DT


class ConstSpeed(_Toy):
    def sample_step(self, hist, state, generator):  # noqa: ANN001, ANN201
        return torch.zeros(hist.shape[0], dtype=torch.float64, device=hist.device)


def dip(n: int = 120, base: float = 12.0, depth: float = 3.0, at: int = 20, width: int = 40) -> np.ndarray:
    """Leader profile [m/s]: ``base`` with a cosine dip of ``depth`` over ``width`` samples from ``at`` on."""
    u = np.full(n, base)
    j = np.arange(width)
    u[at : at + width] = base - depth / 2 * (1 - np.cos(2 * np.pi * j / width))
    return u


# --------------------------------------------------------------------------------------------- basics
def test_config_time_line_and_initial_gap() -> None:
    cfg = PlatoonConfig()
    assert (cfg.burn_in, cfg.tail) == (300, 400) and time_line_steps(1200, cfg) == 300 + 1199 + 400
    with pytest.raises(ValueError):
        PlatoonConfig(history_s=2.0)
    u = np.array([5.0, 6.0, 7.0])
    v, x = leader_timeline(u, burn_in=4, n_steps=9)
    np.testing.assert_allclose(v, [5, 5, 5, 5, 5, 6, 7, 7, 7, 7])
    np.testing.assert_allclose(x, np.concatenate([[0.0], np.cumsum(v[1:] * DT)]))  # x[k + 1] = x[k] + v[k + 1] DT
    with pytest.raises(ValueError):
        leader_timeline(u, burn_in=8, n_steps=9)
    np.testing.assert_allclose(initial_gap([10.0, 20.0], IDM, 120.0), equilibrium_gap(np.array([10.0, 20.0]), IDM))
    np.testing.assert_allclose(initial_gap([29.99, 31.0], IDM, 120.0), [120.0, 120.0])  # capped; infinite above v0
    ev = steady_event("e", 12.0, 15.0, -15.0, HIST_LEN + 5)
    ev.validate()
    assert ev.x_follower[HIST_LEN - 1] == -15.0 and np.all(ev.s == 15.0) and np.all(ev.dv == 0.0)


def test_input_validation() -> None:
    model, u = IDMStep(IDM), [dip()]
    with pytest.raises(ValueError, match="stream seeds"):
        simulate_platoon(model, u, [20.0], [1, 2], SMALL)
    with pytest.raises(ValueError, match="shorter"):
        simulate_platoon(model, u, [20.0], [1, 2, 3, 4], SMALL, n_steps=50)
    with pytest.raises(ValueError, match="positive"):
        simulate_platoon(model, u, [0.0], [1, 2, 3, 4], SMALL)
    with pytest.raises(ValueError, match="max_batch"):
        simulate_platoon(model, u * 2, [20.0, 20.0], [1, 2, 3, 4], PlatoonConfig(n_followers=4, n_rollouts=3, max_batch=5))


# ------------------------------------------------------------------------------------ cascade vs joint
def joint_simulation(u: np.ndarray, gap: float, n_v: int, cfg: PlatoonConfig, n_steps: int) -> dict[str, np.ndarray]:
    """All vehicles stepped together (numpy), model inputs rounded to float32 like the engine's history buffer."""
    v_l, x_l = leader_timeline(u, cfg.burn_in, n_steps)
    x = -gap * np.arange(1, n_v + 1, dtype=np.float64)
    v = np.full(n_v, u[0])
    inp = np.array([[gap, 0.0, u[0]]] * n_v, dtype=np.float32)
    out = {k: np.empty((n_v, n_steps + 1)) for k in ("s", "v", "x")}
    out["s"][:, 0], out["v"][:, 0], out["x"][:, 0] = gap, v, x
    for k in range(n_steps):
        last = inp.astype(np.float64)
        a = np.clip(idm_accel(last[:, 0], last[:, 1], last[:, 2], IDM), A_MIN, A_MAX)
        v = np.maximum(v + a * DT, 0.0)
        x = x + v * DT
        lead_x = np.concatenate([[x_l[k + 1]], x[:-1]])
        lead_v = np.concatenate([[v_l[k + 1]], v[:-1]])
        s = lead_x - x
        inp = np.stack([np.maximum(s, S_EPS), v - lead_v, v], axis=1).astype(np.float32)
        out["s"][:, k + 1], out["v"][:, k + 1], out["x"][:, k + 1] = s, v, x
    return out


def test_deterministic_cascade_reproduces_the_joint_simulation() -> None:
    profiles = [dip(depth=3.0), dip(n=90, base=8.0, depth=4.0, at=10)]
    gaps = [float(initial_gap(p[0], IDM, 120.0)) for p in profiles]
    res = simulate_platoon(IDMStep(IDM), profiles, gaps, [11, 12, 13, 14], SMALL, keep_paths=True)
    assert res.n_steps == time_line_steps(120, SMALL) and res.paths["v"].shape == (2, 4, 3, res.n_steps + 1)
    for e, (u, gap) in enumerate(zip(profiles, gaps)):
        ref = joint_simulation(u, gap, 4, SMALL, res.n_steps)
        for key in ("s", "v", "x"):
            for r in range(3):
                np.testing.assert_allclose(res.paths[key][e, :, r], ref[key], rtol=0, atol=1e-9)
        # sequential: vehicle by vehicle with the deterministic simulator of physics/idm.py (float64 inputs, no floor)
        lead_v, lead_x = leader_timeline(u, SMALL.burn_in, res.n_steps)
        for i in range(4):
            sim = simulate_follower(lambda s, dv, v: idm_accel(s, dv, v, IDM), gap, u[0], lead_v, x_lead=lead_x)
            np.testing.assert_allclose(res.paths["s"][e, i, 0], sim["s"], rtol=0, atol=1e-4)  # float32 model inputs
            np.testing.assert_allclose(res.paths["v"][e, i, 0], sim["v"], rtol=0, atol=1e-5)
            lead_v, lead_x = sim["v"], sim["x"]
        k_e = time_line_steps(u.size, SMALL)
        window = ref["v"][:, SMALL.burn_in : k_e + 1]
        np.testing.assert_allclose(res.amplitude[e, :, 0], np.ptp(window, axis=1), atol=1e-9)
        np.testing.assert_allclose(res.sd[e, :, 0], window.std(axis=1), atol=1e-9)
        np.testing.assert_allclose(res.min_gap[e, :, 0], ref["s"][:, 1 : k_e + 1].min(axis=1), atol=1e-9)
    assert (res.first_collision == -1).all()


def test_same_rollout_coupling_and_independent_vehicles() -> None:
    cfg = PlatoonConfig(n_followers=3, n_rollouts=64, burn_in_s=2.0, tail_s=3.0)
    u = dip()
    res = simulate_platoon(FrozenOffset(IDM), [u, u], [20.0, 20.0], [5, 6, 7], cfg, sampler_mode="per_trajectory",
                           keep_paths=True)  # fmt: skip
    s, x, v = res.paths["s"], res.paths["x"], res.paths["v"]
    v_l, x_l = leader_timeline(u, cfg.burn_in, res.n_steps)
    np.testing.assert_allclose(s[:, 0], x_l[None, None] - x[:, 0], atol=1e-9)  # vehicle 1 behind the leader profile
    for i in range(1, 3):
        np.testing.assert_allclose(s[:, i], x[:, i - 1] - x[:, i], atol=1e-9)  # vehicle i + 1 behind vehicle i, same r
    # leader override (E, R, K + 1): every rollout of vehicle 2 has its own leader; offsets per vehicle are independent
    assert np.ptp(v[0, 0, :, -1]) > 0.5
    offset = (v[0, :, :, 1] - v[0, :, :, 0]) / DT  # first step: identical inputs, IDM(steady) + the frozen offset
    assert np.ptp(offset[0]) > 1.0
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert abs(np.corrcoef(offset[a], offset[b])[0, 1]) < 0.4  # 64 rollouts: sd of a null correlation 0.125
    # the two identical profiles in one batch draw different numbers (one stream for the batch)
    assert not np.allclose(v[0, 0], v[1, 0])


def test_override_shapes_of_the_engine() -> None:
    ev = steady_event("e", 10.0, 15.0, -15.0, HIST_LEN + 20)
    cfg = RolloutConfig(n_rollouts=2, horizon_s=2.0, horizons_s=(2.0,))
    lead_v = np.full((1, 2, 21), 10.0)
    lead_x = np.cumsum(np.concatenate([np.zeros((1, 2, 1)), lead_v[:, :, 1:] * DT], axis=2), axis=2)
    out = rollout_events(ConstSpeed(), [ev], cfg=cfg, leader_override={"x_lead": lead_x, "v_lead": lead_v}, keep_paths=True)
    np.testing.assert_allclose(out.s_path[0], 15.0, atol=1e-9)
    with pytest.raises(ValueError, match="leader_override"):
        rollout_events(ConstSpeed(), [ev], cfg=cfg, leader_override={"x_lead": lead_x[:, :, :-1], "v_lead": lead_v[:, :, :-1]})


# ------------------------------------------------------------------------------------------ statistics
def test_white_noise_speed_std_grows_with_the_index() -> None:
    cfg = PlatoonConfig(n_followers=6, n_rollouts=200, burn_in_s=5.0, tail_s=5.0)
    res = simulate_platoon(IDMStep(IDM, sigma=0.3), [np.full(200, 15.0)], [float(initial_gap(15.0, IDM, 120.0))],
                           [1, 2, 3, 4, 5, 6], cfg)  # fmt: skip
    sd = res.sd[0].mean(axis=1)
    # noise of the predecessors adds to the own noise: increasing, concave (string-stable IDM); the first two
    # increments are 4.8 and 8 standard errors in a reference run with other seeds
    assert sd[1] > sd[0] and sd[2] > sd[1] and sd[-1] > 1.25 * sd[0]
    assert np.diff(sd)[:2].mean() > np.diff(sd)[-3:].mean()
    assert res.leader_amplitude[0] == 0.0 and (res.amplitude > 0).all()


def test_amplitude_and_sd_of_an_exact_follower() -> None:
    """Vehicle i repeats the leader with a delay of i steps: amplitudes equal the leader's; with equal first and last
    speeds the response-window sd equals the leader's too; profile-window sd over the shifted samples."""
    cfg = PlatoonConfig(n_followers=4, n_rollouts=2, burn_in_s=2.0, tail_s=3.0)
    u = dip(n=100, base=12.0, depth=3.0, at=20, width=40)
    res = simulate_platoon(ExactFollower(), [u], [20.0], [1, 2, 3, 4], cfg, keep_paths=True)
    b, k_e = cfg.burn_in, time_line_steps(u.size, cfg)
    v_l, _ = leader_timeline(u, b, res.n_steps)
    for i in range(1, 5):
        delayed = v_l[np.arange(res.n_steps + 1) - i].copy()
        delayed[:i] = u[0]
        np.testing.assert_allclose(res.paths["v"][0, i - 1, 0], delayed, atol=1e-4)
        assert res.amplitude[0, i - 1] == pytest.approx([3.0, 3.0], abs=1e-4)
        assert res.sd[0, i - 1] == pytest.approx([v_l[b : k_e + 1].std()] * 2, abs=1e-5)
        assert res.amplitude_profile[0, i - 1] == pytest.approx([3.0, 3.0], abs=1e-4)
        assert res.sd_profile[0, i - 1] == pytest.approx([delayed[b : b + u.size].std()] * 2, abs=1e-5)
    assert res.leader_amplitude[0] == pytest.approx(3.0) and res.leader_sd[0] == pytest.approx(v_l[b : k_e + 1].std())
    assert res.leader_sd_profile[0] == pytest.approx(u.std())
    np.testing.assert_allclose(res.mean_speed[0], res.paths["v"][0].mean(axis=1), atol=1e-5)


def test_collision_step_and_min_gap_by_hand() -> None:
    """Constant-speed followers behind a braking leader: vehicle 1 runs into it, the others keep their gap."""
    cfg = PlatoonConfig(n_followers=3, n_rollouts=2, burn_in_s=1.0, tail_s=2.0)
    brake = np.concatenate([np.full(10, 10.0), np.maximum(10.0 - 0.2 * np.arange(1, 41), 2.0), np.full(10, 2.0)])
    long_profile = np.full(140, 10.0)  # makes the batch's time line longer than the braking profile's own
    gap = 3.0
    res = simulate_platoon(ConstSpeed(), [brake, long_profile], [gap, gap], [1, 2, 3], cfg)
    v_l, x_l = leader_timeline(brake, cfg.burn_in, res.n_steps)
    k = np.arange(res.n_steps + 1)
    s1 = x_l - (-gap + 10.0 * DT * k)  # follower 1 at 10 m/s from x = -gap
    k_e = time_line_steps(brake.size, cfg)
    first = int(np.argmax(s1[1:] <= 0.0)) + 1
    assert 1 <= first <= k_e and s1[first] <= 0.0 < s1[first - 1]
    assert (res.first_collision[0, 0] == first).all()
    assert res.min_gap[0, 0] == pytest.approx([s1[1 : k_e + 1].min()] * 2, abs=1e-9)
    assert (res.first_collision[0, 1:] == -1).all() and res.min_gap[0, 1:] == pytest.approx(np.full((2, 2), gap), abs=1e-9)
    assert (res.first_collision[1] == -1).all()  # the leader of profile 2 never brakes
    # a collision after the profile's own time line (the batch simulates further) is not counted
    late = np.concatenate([np.full(70, 10.0), np.full(10, 9.0)])  # follower 1 closes in at 1 m/s from the end on
    res = simulate_platoon(ConstSpeed(), [late, long_profile], [5.0, 5.0], [1, 2, 3], cfg)
    v_l, x_l = leader_timeline(late, cfg.burn_in, res.n_steps)
    s1 = x_l - (-5.0 + 10.0 * DT * k)
    assert (s1[1 : time_line_steps(late.size, cfg) + 1] > 0).all() and (s1[res.n_steps] <= 0)
    assert (res.first_collision[0, 0] == -1).all()


def test_speed_cap_setting() -> None:
    """``PlatoonConfig.v_max`` is optional, validated and passed to the engine (e3_vcap.yaml sensitivity)."""
    with pytest.raises(ValueError, match="v_max"):
        PlatoonConfig(v_max=-1.0)
    assert PlatoonConfig().v_max is None
    model, u = IDMStep(IDM), [dip()]
    capped = simulate_platoon(model, u, [20.0], [1, 2, 3, 4], replace(SMALL, v_max=1.0), keep_paths=True)
    assert float(capped.paths["v"][..., 1:].max()) <= 1.0 + 1e-9  # step 0 is the initial state
