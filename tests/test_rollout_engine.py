"""Closed-loop rollout engine (cf_unc.rollout.engine, docs/design.md, section 11.2)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterator

import numpy as np
import pytest
import torch

from cf_unc.constants import A_MAX, A_MIN, DT, HIST_LEN, S_EPS
from cf_unc.data.pipeline import DataBundle
from cf_unc.data.schema import Event
from cf_unc.data.synthetic import generate
from cf_unc.experiments.e1 import configure_torch, load_config, model_spec, seed_everything
from cf_unc.models import SAMPLER_MODES, build_model
from cf_unc.models.base import CFModel
from cf_unc.physics.idm import IDMParams, idm_accel, simulate_follower
from cf_unc.rollout.engine import RolloutConfig, rollout_events, stream_seed

T0 = HIST_LEN - 1


# ------------------------------------------------------------------------------------------ helpers
def make_event(x_lead: np.ndarray, v_lead: np.ndarray, x_f: np.ndarray, v_f: np.ndarray, eid: str = "ev") -> Event:
    """Event from leader / follower positions and speeds (arrays of equal length, 10 Hz)."""
    n = len(v_f)
    return Event(
        event_id=eid, dataset="toy", site="toy", follower_id=f"f_{eid}", leader_id=f"l_{eid}", t=np.arange(n) * DT,
        s=x_lead - x_f, dv=v_f - v_lead, v=v_f, a=np.gradient(v_f, DT), v_lead=v_lead, x_lead=x_lead, x_follower=x_f,
    )  # fmt: skip


def cruise_event(speed: float = 20.0, gap: float = 30.0, n: int = 140, eid: str = "cruise") -> Event:
    """Leader and follower at the same constant speed [m/s] and gap [m]."""
    v = np.full(n, speed)
    x_lead = 100.0 + gap + speed * DT * np.arange(n)
    return make_event(x_lead, v, x_lead - gap, v.copy(), eid)


def braking_leader_event(v0: float = 10.0, b: float = 2.0, gap: float = 20.0, n: int = 140) -> Event:
    """Leader at v0 until T0, then braking at b [m/s^2] to a stop (semi-implicit Euler); follower recorded at gap."""
    v = np.full(n, v0)
    j = np.arange(n)
    v[T0:] = np.maximum(v0 - (j[T0:] - T0) * b * DT, 0.0)
    x = np.empty(n)
    x[0] = 50.0
    for t in range(n - 1):
        x[t + 1] = x[t] + v[t + 1] * DT
    return make_event(x, v, x - gap, v.copy(), "brake")


class _Toy(CFModel):
    """Test models: stubs for the abstract training / persistence API."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fitted = True
        self.calls = 0
        self.min_gap_seen = np.inf

    def fit(self, train, val=None, *, seed=0, verbose=False):  # noqa: ANN001, ANN201
        return {}

    def predict_dist(self, state_history, *, v_lead_future=None, batch_size=8192):  # noqa: ANN001, ANN201
        raise NotImplementedError

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict) -> None:
        pass

    def observe(self, hist: torch.Tensor) -> None:
        self.calls += 1
        self.min_gap_seen = min(self.min_gap_seen, float(hist[:, :, 0].min()))


class ConstAccel(_Toy):
    """Always the same command a [m/s^2] (records the calls and the smallest input gap)."""

    def __init__(self, a: float) -> None:
        super().__init__(a=a)
        self.a = float(a)

    def sample_step(self, hist, state, generator):  # noqa: ANN001, ANN201
        self.observe(hist)
        return torch.full((hist.shape[0],), self.a, dtype=torch.float32, device=hist.device)


class WhiteNoise(_Toy):
    """a = mean + sigma * eps, eps ~ N(0, 1) iid per call and trajectory."""

    def __init__(self, sigma: float, mean: float = 0.0) -> None:
        super().__init__(sigma=sigma, mean=mean)
        self.sigma, self.mean = float(sigma), float(mean)

    def sample_step(self, hist, state, generator):  # noqa: ANN001, ANN201
        self.observe(hist)
        eps = torch.randn(hist.shape[0], generator=generator, device=hist.device)
        return self.mean + self.sigma * eps


class IDMStep(_Toy):
    """IDM acceleration of the last history sample (float64 arithmetic) plus optional white noise."""

    def __init__(self, params: IDMParams, sigma: float = 0.0) -> None:
        super().__init__(sigma=sigma)
        self.params, self.sigma = params, float(sigma)

    def sample_step(self, hist, state, generator):  # noqa: ANN001, ANN201
        self.observe(hist)
        last = hist[:, -1].to(torch.float64)
        a = idm_accel(last[:, 0], last[:, 1], last[:, 2], self.params)
        if self.sigma > 0.0:
            a = a + self.sigma * torch.randn(a.shape, generator=generator, dtype=a.dtype, device=a.device)
        return a


class TwoMemberEnsemble(_Toy):
    """Members with means +mu and -mu and aleatoric sd ``sd``; per_trajectory freezes the member."""

    def __init__(self, mu: float = 1.0, sd: float = 0.1) -> None:
        super().__init__(mu=mu, sd=sd)
        self.mu, self.sd = float(mu), float(sd)

    def init_sampler(self, n_traj, generator, mode="per_step"):  # noqa: ANN001, ANN201
        super().init_sampler(n_traj, generator, mode)
        return None if mode == "per_step" else torch.randint(2, (n_traj,), generator=generator, device=generator.device)

    def sample_step(self, hist, state, generator):  # noqa: ANN001, ANN201
        b = hist.shape[0]
        member = torch.randint(2, (b,), generator=generator, device=hist.device) if state is None else state
        eps = torch.randn(b, generator=generator, device=hist.device)
        return (2.0 * member.float() - 1.0) * self.mu + self.sd * eps


# -------------------------------------------------------------------------------------------- tests
def test_config_validation() -> None:
    cfg = RolloutConfig()
    assert cfg.n_steps == 100 and cfg.horizon_steps.tolist() == [10, 30, 50, 100]
    for bad in ({"horizon_s": 1.05}, {"horizons_s": (1.0, 11.0)}, {"horizons_s": (0.05,)}, {"sampler_mode": "x"},
                {"hold_steps": 0}, {"horizons_s": (1.0, 1.0)}, {"s_floor": 0.0}):  # fmt: skip
        with pytest.raises(ValueError):
            RolloutConfig(**bad)
    assert stream_seed(0, 5) == stream_seed(0, 5) != stream_seed(0, 6) != stream_seed(1, 5)


def test_deterministic_idm_reproduces_simulate_follower(true_idm: IDMParams) -> None:
    events, _ = generate(dict(n_drivers=3, events_per_driver=2, duration_s=16.0, seed=5))
    cfg = RolloutConfig(n_rollouts=3, horizon_s=12.0, horizons_s=(1.0, 6.0, 12.0), stream_events=2)
    res = rollout_events(IDMStep(true_idm), events, cfg=cfg, keep_paths=True)
    k = cfg.n_steps
    for e, ev in enumerate(events):
        xl, vl = ev.x_lead[T0 : T0 + k + 1], ev.v_lead[T0 : T0 + k + 1]
        sim = simulate_follower(lambda s, dv, v: idm_accel(s, dv, v, true_idm), xl[0] - ev.x_follower[T0], ev.v[T0], vl, x_lead=xl)
        for r in range(cfg.n_rollouts):
            np.testing.assert_allclose(res.s_path[e, r], sim["s"], rtol=0, atol=1e-4)  # 1e-4 m (observed below 1e-6)
            np.testing.assert_allclose(res.v_path[e, r], sim["v"], rtol=0, atol=1e-5)
            np.testing.assert_allclose(res.s[e, r], sim["s"][cfg.horizon_steps], rtol=0, atol=1e-4)
        np.testing.assert_allclose(res.truth_s[e], ev.s[T0 + cfg.horizon_steps])
        np.testing.assert_allclose(res.truth_v[e], ev.v[T0 + cfg.horizon_steps])
    assert (res.first_collision == -1).all()
    np.testing.assert_allclose(res.s_min, res.s_path[:, :, 1:].min(axis=2))
    assert res.t0.tolist() == [T0] * len(events) and res.event_ids == [ev.event_id for ev in events]


def test_hold_steps_keep_the_command() -> None:
    model = WhiteNoise(sigma=0.5)
    cfg = RolloutConfig(n_rollouts=16, horizon_s=5.0, horizons_s=(5.0,), hold_steps=10)
    res = rollout_events(model, [cruise_event()], cfg=cfg, keep_paths=True)
    assert model.calls == 5  # steps 0, 10, 20, 30, 40
    acc = np.diff(res.v_path[0], axis=1) / DT  # (R, K) effective acceleration, no clip or standstill here
    blocks = acc.reshape(16, 5, 10)
    np.testing.assert_allclose(blocks, blocks[:, :, :1].repeat(10, axis=2), rtol=0, atol=1e-9)
    assert (np.abs(np.diff(blocks[:, :, 0], axis=1)) > 1e-6).all()  # a new draw every 10th step
    one = WhiteNoise(sigma=0.5)
    rollout_events(one, [cruise_event()], cfg=RolloutConfig(n_rollouts=16, horizon_s=5.0, horizons_s=(5.0,)))
    assert one.calls == 50


def test_acceleration_clip_and_speed_floor() -> None:
    ev = cruise_event(speed=10.0, gap=500.0)
    cfg = RolloutConfig(n_rollouts=2, horizon_s=4.0, horizons_s=(4.0,))
    fast = rollout_events(ConstAccel(100.0), [ev], cfg=cfg, keep_paths=True)
    np.testing.assert_allclose(np.diff(fast.v_path[0], axis=1), A_MAX * DT, atol=1e-12)  # clipped at A_MAX
    slow = rollout_events(ConstAccel(-100.0), [ev], cfg=cfg, keep_paths=True)
    v = slow.v_path[0, 0]
    expected = np.maximum(10.0 + A_MIN * DT * np.arange(v.size), 0.0)  # clipped at A_MIN, floored at 0
    np.testing.assert_allclose(v, expected, atol=1e-12)
    x = slow.x_path[0, 0]
    stop = int(np.argmax(v == 0.0))
    assert stop > 0 and (v[stop:] == 0.0).all() and np.ptp(x[stop:]) == 0.0  # standing still
    np.testing.assert_allclose(np.diff(x), v[1:] * DT, atol=1e-12)  # x' = x + v' DT


def test_rollout_collision_detection() -> None:
    """Leader brakes to a stop, the model always accelerates: first collision at the analytic step."""
    v0, b, gap, a = 10.0, 2.0, 20.0, 1.0
    ev = braking_leader_event(v0=v0, b=b, gap=gap)
    cfg = RolloutConfig(n_rollouts=5, horizon_s=10.0, horizons_s=(1.0, 3.0, 5.0, 10.0))
    model = ConstAccel(a)
    res = rollout_events(model, [ev], cfg=cfg, keep_paths=True)
    # closed forms of the semi-implicit Euler scheme from t0 (step k = 0 .. K)
    k = np.arange(cfg.n_steps + 1)
    x_f = ev.x_follower[T0] + DT * (k * v0 + a * DT * k * (k + 1) / 2)
    j = np.arange(1, cfg.n_steps + 1)
    x_l = ev.x_lead[T0] + DT * np.concatenate([[0.0], np.cumsum(np.maximum(v0 - j * b * DT, 0.0))])
    s_exact = x_l - x_f
    k_star = int(np.argmax(s_exact[1:] <= 0.0)) + 1
    assert 30 < k_star < 100 and s_exact[k_star] <= 0 < s_exact[k_star - 1]
    np.testing.assert_allclose(res.s_path[0], np.broadcast_to(s_exact, res.s_path[0].shape), atol=1e-9)
    assert (res.first_collision == k_star).all()
    collided = res.collided()[0]
    np.testing.assert_array_equal(collided, (cfg.horizon_steps >= k_star).astype(float))
    assert collided[-1] == 1.0
    assert (res.s_path[0, :, k_star:] < 0.0).all()  # the trajectory continues, the gap stays negative
    assert res.s_min.max() < 0.0
    assert model.min_gap_seen >= np.float32(S_EPS)  # the model never receives a gap below the floor
    assert model.min_gap_seen == pytest.approx(S_EPS, abs=1e-7)  # ... and the floor was active

    safe = ConstAccel(-8.0)  # full braking: stops within 6.6 m, the leader needs 25 m
    res_safe = rollout_events(safe, [ev], cfg=cfg)
    assert (res_safe.first_collision == -1).all() and (res_safe.collided() == 0.0).all() and res_safe.s_min.min() > 0


def test_white_noise_gap_variance_closed_form() -> None:
    """a = sigma eps with the leader at constant speed: Var s_k = DT^4 sigma^2 k (k + 1) (2k + 1) / 6."""
    sigma, n_r = 0.5, 20_000
    cfg = RolloutConfig(n_rollouts=n_r, horizon_s=10.0, horizons_s=(1.0, 3.0, 5.0, 10.0), seed=3)
    res = rollout_events(WhiteNoise(sigma), [cruise_event(speed=20.0, gap=60.0)], cfg=cfg)
    k = cfg.horizon_steps.astype(np.float64)
    var_s = DT**4 * sigma**2 * k * (k + 1) * (2 * k + 1) / 6
    var_v = DT**2 * sigma**2 * k
    # relative sd of a sample variance of 20 000 normal values: sqrt(2 / n) = 1 %; tolerance 5 % = 5 sd
    np.testing.assert_allclose(res.s[0].var(axis=0, ddof=1), var_s, rtol=0.05)
    np.testing.assert_allclose(res.v[0].var(axis=0, ddof=1), var_v, rtol=0.05)
    assert (np.abs(res.s[0].mean(axis=0) - 60.0) < 4 * np.sqrt(var_s / n_r)).all()  # unbiased (4 sd)


def test_per_trajectory_ensemble_is_bimodal() -> None:
    ev = cruise_event(speed=20.0, gap=100.0)
    cfg = dict(n_rollouts=2000, horizon_s=10.0, horizons_s=(10.0,), seed=1)
    # frozen member: gaps 100 -/+ DT^2 mu K (K + 1) / 2 = 100 -/+ 50.5 m; per step: sd about 5.8 m around 100
    traj = rollout_events(TwoMemberEnsemble(), [ev], cfg=RolloutConfig(sampler_mode="per_trajectory", **cfg)).s[0, :, 0]
    step = rollout_events(TwoMemberEnsemble(), [ev], cfg=RolloutConfig(sampler_mode="per_step", **cfg)).s[0, :, 0]
    near_traj, near_step = np.abs(traj - 100.0) < 35.0, np.abs(step - 100.0) < 35.0  # 35 m = 6 sd of per-step
    assert near_traj.sum() == 0 and near_step.all()
    low = np.mean(traj < 100.0)
    assert abs(low - 0.5) < 5 * np.sqrt(0.25 / 2000)  # two modes of equal weight (5 binomial sd)
    for mode in (traj[traj < 100.0], traj[traj > 100.0]):
        assert np.ptp(mode) < 10.0  # tight clusters (aleatoric sd 0.1 m/s^2 -> gap sd 0.6 m)


def _stochastic_events() -> list[Event]:
    events, _ = generate(dict(n_drivers=3, events_per_driver=2, duration_s=14.0, seed=7))
    return list(events)


def test_results_do_not_depend_on_max_batch(true_idm: IDMParams) -> None:
    events = _stochastic_events()
    base = dict(n_rollouts=8, horizon_s=10.0, stream_events=2, seed=11)
    runs = [rollout_events(IDMStep(true_idm, sigma=0.3), events, cfg=RolloutConfig(max_batch=m, **base))
            for m in (16, 32, 48, 1000)]  # fmt: skip
    assert [r.info["n_batches"] for r in runs] == [3, 2, 1, 1]
    for other in runs[1:]:
        for name in ("s", "v", "first_collision", "s_min"):
            np.testing.assert_array_equal(getattr(other, name), getattr(runs[0], name))
    with pytest.raises(ValueError, match="exceeds max_batch"):
        rollout_events(IDMStep(true_idm, sigma=0.3), events, cfg=RolloutConfig(max_batch=15, **base))


def test_reproducible_from_seed_and_event_order(true_idm: IDMParams) -> None:
    events = _stochastic_events()
    cfg = RolloutConfig(n_rollouts=6, horizon_s=5.0, horizons_s=(5.0,), stream_events=1, seed=4)
    a = rollout_events(IDMStep(true_idm, sigma=0.3), events, cfg=cfg)
    b = rollout_events(IDMStep(true_idm, sigma=0.3), events, cfg=cfg)
    np.testing.assert_array_equal(a.s, b.s)
    c = rollout_events(IDMStep(true_idm, sigma=0.3), events, cfg=replace(cfg, seed=5))
    assert not np.array_equal(a.s, c.s)
    # with one stream per event, an event's rollouts depend on its position in the list, not on the others
    d = rollout_events(IDMStep(true_idm, sigma=0.3), events[:2], cfg=cfg)
    np.testing.assert_array_equal(d.s, a.s[:2])


def test_start_index_and_short_events(true_idm: IDMParams) -> None:
    ev = cruise_event(n=140)
    cfg = RolloutConfig(n_rollouts=2, horizon_s=10.0)
    assert rollout_events(IDMStep(true_idm), [ev], starts=39, cfg=cfg).t0.tolist() == [39]
    with pytest.raises(ValueError, match="too short"):
        rollout_events(IDMStep(true_idm), [ev], starts=40, cfg=cfg)  # needs 40 + 100 + 1 = 141 samples
    with pytest.raises(ValueError, match="below HIST_LEN"):
        rollout_events(IDMStep(true_idm), [ev], starts=10, cfg=cfg)


def test_leader_override_per_rollout(true_idm: IDMParams) -> None:
    ev = cruise_event(speed=15.0, gap=25.0)
    cfg = RolloutConfig(n_rollouts=3, horizon_s=5.0, horizons_s=(5.0,))
    k = cfg.n_steps
    xl, vl = ev.x_lead[T0 : T0 + k + 1], ev.v_lead[T0 : T0 + k + 1]
    same = rollout_events(IDMStep(true_idm), [ev], cfg=cfg, leader_override={"x_lead": xl[None], "v_lead": vl[None]})
    plain = rollout_events(IDMStep(true_idm), [ev], cfg=cfg)
    np.testing.assert_array_equal(same.s, plain.s)
    shift = np.array([0.0, 5.0, 10.0])[None, :, None]  # rollout r sees its leader shifted by 5 r metres
    res = rollout_events(IDMStep(true_idm), [ev], cfg=cfg, keep_paths=True,
                         leader_override={"x_lead": xl[None, None] + shift, "v_lead": np.broadcast_to(vl, (1, 3, k + 1))})  # fmt: skip
    assert res.s[0, 0, 0] < res.s[0, 1, 0] < res.s[0, 2, 0]
    np.testing.assert_allclose(res.s_path[0, :, 0], 25.0 + shift[0, :, 0])


class _LeaderProbe(ConstAccel):
    """Accepts ``v_lead`` in init_sampler like DiffusionCF and keeps it."""

    def init_sampler(self, n_traj, generator, mode="per_step", v_lead=None):  # noqa: ANN001, ANN201
        super().init_sampler(n_traj, generator, mode)
        return {"v_lead": v_lead}


def test_v_lead_is_offered_to_samplers_that_take_it() -> None:
    ev = braking_leader_event()
    cfg = RolloutConfig(n_rollouts=2, horizon_s=3.0, horizons_s=(3.0,))
    captured: dict[str, Any] = {}
    probe = _LeaderProbe(0.0)
    original = probe.sample_step

    def spy(hist, state, generator):  # noqa: ANN001, ANN202
        captured.setdefault("v_lead", state["v_lead"])
        return original(hist, state, generator)

    probe.sample_step = spy  # type: ignore[method-assign]
    rollout_events(probe, [ev], cfg=cfg)
    expected = np.broadcast_to(ev.v_lead[T0 : T0 + 31], (2, 31))
    np.testing.assert_allclose(captured["v_lead"].numpy(), expected, rtol=1e-6)
    rollout_events(ConstAccel(0.0), [ev], cfg=cfg)  # a sampler without the keyword works too


# --------------------------------------------------------------------------------- the real models
LABELS = [
    "idm_white_noise", "idm_white_noise_mle", "lstm_const_sigma", "hetero_mdn", "hetero_mdn_mix3", "deep_ensemble",
    "mc_dropout", "gp_residual_idm", "quantile_pidl", "diffusion_cf",
    "idm_ar_noise", "diffusion_cf_hist",  # review controls (design.md 18); lstm_hetero_frozen needs a stored mean
]  # fmt: skip


def _tiny_model(label: str, idm: IDMParams, small: DataBundle, small_seq: DataBundle, device: str) -> CFModel:
    """The model configuration of the experiments with a minimal training budget on a few hundred windows."""
    cfg = load_config()
    cfg.train.overrides = {"max_epochs": 1, "batch_size": 128}
    cfg.model_overrides = {"gp_residual_idm": {"max_epochs": 1, "batch_size": 256, "n_inducing": 16},
                           "diffusion_cf": {"n_sampling_steps": 5}, "diffusion_cf_hist": {"n_sampling_steps": 5}}  # fmt: skip
    spec = model_spec(cfg, label, lambda name: (idm, {"name": name}))
    data = small_seq if spec.sequences else small
    rows = np.arange(min(400, len(data.windows["train"])))
    seed_everything(0)
    model = build_model(spec.name, **spec.kwargs).to(device)
    model.fit(data.windows["train"].subset(rows), data.windows["val"].subset(rows[:100]), seed=0)
    return model


def _check_rollout(model: CFModel, events: list[Event], device: str) -> None:
    for mode in SAMPLER_MODES:
        cfg = RolloutConfig(n_rollouts=4, horizon_s=1.0, horizons_s=(0.5, 1.0), sampler_mode=mode, stream_events=2)
        res = rollout_events(model, events, cfg=cfg, device=device)
        assert res.s.shape == (len(events), 4, 2) and res.first_collision.shape == (len(events), 4)
        assert np.isfinite(res.s).all() and np.isfinite(res.v).all() and (res.v >= 0).all()
        assert res.info["device"].startswith(device)


@pytest.fixture(scope="module")
def roll_events() -> list[Event]:
    events, _ = generate(dict(n_drivers=2, events_per_driver=2, duration_s=6.0, seed=9))
    return list(events)[:3]


@pytest.mark.parametrize("label", LABELS)
def test_every_model_rolls_out_on_cpu(label: str, synth_small: DataBundle, synth_small_seq: DataBundle,
                                      true_idm: IDMParams, roll_events: list[Event]) -> None:  # fmt: skip
    model = _tiny_model(label, true_idm, synth_small, synth_small_seq, "cpu")
    _check_rollout(model, roll_events, "cpu")


@pytest.fixture(scope="module")
def ieee_float32() -> Iterator[None]:
    """IEEE float32 on the GPU for the CUDA test (as in the experiment runners); flags restored afterwards."""
    cudnn, matmul = torch.backends.cudnn, torch.backends.cuda.matmul
    legacy = (cudnn.allow_tf32, matmul.allow_tf32)
    precision = (cudnn.conv.fp32_precision, cudnn.rnn.fp32_precision, matmul.fp32_precision)
    flags = (cudnn.deterministic, cudnn.benchmark)
    configure_torch(tf32=False)
    yield
    cudnn.allow_tf32, matmul.allow_tf32 = legacy
    cudnn.conv.fp32_precision, cudnn.rnn.fp32_precision, matmul.fp32_precision = precision
    cudnn.deterministic, cudnn.benchmark = flags


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("label", LABELS)
def test_every_model_rolls_out_on_cuda(label: str, synth_small: DataBundle, synth_small_seq: DataBundle,
                                       true_idm: IDMParams, roll_events: list[Event], ieee_float32: None) -> None:  # fmt: skip
    model = _tiny_model(label, true_idm, synth_small, synth_small_seq, "cuda")
    _check_rollout(model, roll_events, "cuda")
    cfg = RolloutConfig(n_rollouts=4, horizon_s=1.0, horizons_s=(1.0,), seed=2)
    first = rollout_events(model, roll_events, cfg=cfg)
    np.testing.assert_array_equal(first.s, rollout_events(model, roll_events, cfg=cfg).s)  # reproducible on the GPU


# ------------------------------------------------------------------------------------------ speed cap
def test_speed_cap_v_max() -> None:
    """``v_max`` caps the follower speed (E3 sensitivity); without it the speed grows freely (design.md, section 7)."""
    cfg = RolloutConfig(n_rollouts=3, horizon_s=5.0, horizons_s=(5.0,))
    free = rollout_events(ConstAccel(A_MAX), [cruise_event(speed=20.0)], cfg=cfg, keep_paths=True)
    capped = rollout_events(ConstAccel(A_MAX), [cruise_event(speed=20.0)], cfg=replace(cfg, v_max=25.0), keep_paths=True)
    assert float(free.v_path.max()) == pytest.approx(20.0 + A_MAX * 5.0)
    assert float(capped.v_path.max()) == pytest.approx(25.0)
    assert float(capped.v_path[..., -1].min()) == pytest.approx(25.0)
    with pytest.raises(ValueError, match="v_max"):
        RolloutConfig(v_max=0.0)
    with pytest.raises(ValueError, match="v_max"):
        RolloutConfig(v_max=float("nan"))
