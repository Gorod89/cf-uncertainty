"""DiffusionCF (docs/design.md, section 7): schedule, forward process, denoiser, DDIM sampler, rollout sampler,
persistence (fast, synth_small_seq) and the E4 sanity checks of the trained model (slow, synth_e4_seq)."""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cf_unc.constants import A_MAX, A_MIN, DT, MEAN_TARGET_STEPS
from cf_unc.dist import SampleDist
from cf_unc.models import MODEL_REGISTRY, build_model, load_model
from cf_unc.models.diffusion_cf import (
    DEFAULT_TRAIN,
    Denoiser,
    DiffusionCF,
    cosine_alpha_bar,
    ddim_sample,
    q_sample,
    sampling_steps,
)
from cf_unc.train.trainer import TrainConfig

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "model" / "diffusion_cf.yaml"
TINY = dict(n_samples=8, n_sampling_steps=10, t_train=100, channels=16, n_blocks=2, hidden=16)
TINY_TRAIN = {"max_epochs": 2, "batch_size": 256}


@pytest.fixture(scope="module")
def tiny(synth_small_seq) -> tuple[DiffusionCF, dict]:
    """Small network trained for two epochs on synth_small_seq (about 1 s)."""
    model = DiffusionCF(**TINY, train=TINY_TRAIN)
    log = model.fit(synth_small_seq.windows["train"], synth_small_seq.windows["val"], seed=0)
    return model, log


def clone(model: DiffusionCF, **overrides) -> DiffusionCF:
    """Same weights, different constructor arguments."""
    out = DiffusionCF(**{**model.init_kwargs, **overrides})
    out.set_state(model.get_state())
    return out


def point_mass_oracle(x_star: torch.Tensor, alpha_bar: np.ndarray, trace: list | None = None):
    """True noise of the forward process when the data law is a point mass at ``x_star``."""

    def eps_fn(x: torch.Tensor, k: int) -> torch.Tensor:
        if trace is not None:
            trace.append((k, x.clone()))
        ab = float(alpha_bar[k])
        return (x - math.sqrt(ab) * x_star) / math.sqrt(1.0 - ab)

    return eps_fn


# ------------------------------------------------------------------ registry and config
def test_registered_and_config_matches_defaults():
    assert MODEL_REGISTRY["diffusion_cf"] is DiffusionCF
    assert DiffusionCF.needs_leader_future and not DiffusionCF.has_epistemic
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["name"] == "diffusion_cf" and cfg["needs_idm"] is False and cfg["needs_sequences"] is True
    defaults = {k: p.default for k, p in inspect.signature(DiffusionCF).parameters.items() if k != "train"}
    assert cfg["kwargs"] == defaults
    assert set(cfg["train"]) <= set(TrainConfig.__dataclass_fields__) and cfg["train"] == DEFAULT_TRAIN
    model = build_model("diffusion_cf", **cfg["kwargs"], train=cfg["train"])
    assert model.train_cfg.ema_decay == 0.999 and model.init_kwargs["train"] == cfg["train"]
    json.dumps(model.init_kwargs)


def test_constructor_validation():
    with pytest.raises(ValueError, match="replan_every"):
        DiffusionCF(replan_every=51)
    with pytest.raises(ValueError, match="replan_every"):
        DiffusionCF(replan_every=0)
    with pytest.raises(ValueError, match="n_sampling_steps"):
        DiffusionCF(n_sampling_steps=1)
    with pytest.raises(ValueError, match="unknown train"):
        DiffusionCF(train={"learning_rate": 1e-3})
    with pytest.raises(ValueError):
        DiffusionCF(n_samples=1)


# ---------------------------------------------------------------------- diffusion maths
def test_cosine_schedule():
    t_train, s = 1000, 0.008
    ab = cosine_alpha_bar(t_train)
    assert ab.shape == (t_train,) and ab.dtype == np.float64
    assert np.all(np.diff(ab) < 0.0) and np.all((ab > 0.0) & (ab < 1.0))
    # first value 1 - 4.1e-5; last value 2.4e-9: f(T) = 0 makes the last beta 1, clipped to 0.999
    assert ab[0] > 0.99 and ab[-1] < 1e-2
    assert ab[0] == pytest.approx(1.0 - 4.13e-5, abs=1e-7) and ab[-1] == pytest.approx(2.43e-9, rel=1e-2)
    f = np.cos((np.arange(t_train + 1) / t_train + s) / (1.0 + s) * np.pi / 2.0) ** 2
    np.testing.assert_allclose(ab[:-1], f[1:-1] / f[0], rtol=1e-9)  # closed form wherever beta is not clipped


def test_sampling_steps():
    steps = sampling_steps(1000, 50)
    assert steps.shape == (50,) and len(np.unique(steps)) == 50
    assert np.all(np.diff(steps) < 0) and steps[0] == 999 and steps[-1] == 0
    np.testing.assert_array_equal(sampling_steps(100, 100), np.arange(99, -1, -1))
    np.testing.assert_array_equal(sampling_steps(1000, 2), [999, 0])
    for n, t in ((1, 1000), (11, 10)):
        with pytest.raises(ValueError):
            sampling_steps(t, n)
    assert np.array_equal(DiffusionCF().steps, steps)


def test_q_sample_moments():
    ab_np = cosine_alpha_bar(1000)
    ab = torch.as_tensor(ab_np, dtype=torch.float32)
    x0 = torch.linspace(-2.0, 2.0, 50)
    n = 20_000
    gen = torch.Generator().manual_seed(0)
    for k in (0, 100, 500, 900, 999):
        eps = torch.randn((n, 50), generator=gen)
        x_k = q_sample(x0.expand(n, -1), torch.full((n,), k), eps, ab).double()
        mean, var = x_k.mean(0).numpy(), x_k.var(0).numpy()
        m_true, v_true = math.sqrt(ab_np[k]) * x0.double().numpy(), 1.0 - ab_np[k]
        np.testing.assert_allclose(mean, m_true, atol=5.0 * math.sqrt(v_true / n) + 1e-6)  # 5 standard errors
        np.testing.assert_allclose(var, v_true, rtol=5.0 * math.sqrt(2.0 / n), atol=1e-9)


# ------------------------------------------------------------------------------ denoiser
def test_denoiser_shape_zero_init_and_gradients():
    torch.manual_seed(0)
    net = Denoiser(seq_len=50, channels=16, n_blocks=4, kernel=5, hidden=8)
    b = 12
    x_k, k = torch.randn(b, 50), torch.randint(0, 1000, (b,))
    hist, vl = torch.randn(b, 30, 3), torch.randn(b, 50)
    out = net(x_k, k, hist, vl)
    assert out.shape == (b, 50) and torch.all(out == 0.0)  # zero-initialised 1x1 output convolution
    assert net.denoise(x_k, k[:1], net.context(hist), vl).shape == (b, 50)  # one step shared by the batch
    params = dict(net.named_parameters())
    loss = torch.nn.functional.mse_loss(out, torch.randn(b, 50))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values())
    assert params["out.weight"].grad.abs().sum() > 0.0
    # the zero output layer blocks the first backward pass; after one optimiser step every parameter gets a gradient
    torch.optim.SGD(net.parameters(), lr=0.1).step()
    net.zero_grad()
    torch.nn.functional.mse_loss(net(x_k, k, hist, vl), torch.randn(b, 50)).backward()
    dead = [name for name, p in params.items() if p.grad is None or not torch.any(p.grad != 0.0)]
    assert not dead, f"no gradient reaches {dead}"


# ------------------------------------------------------------------------------- sampler
@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_oracle_point_mass_is_recovered(eta):
    """Oracle eps of a point-mass data law: the sampler returns the point, and every intermediate x_k has the
    forward marginal q(x_k | x*) = N(sqrt(ab_k) x*, 1 - ab_k): the normalised residual keeps the initial
    noise exactly (eta = 0) or stays standard normal with the DDPM step-to-step correlation (eta = 1)."""
    ab, steps = cosine_alpha_bar(1000), sampling_steps(1000, 50)
    n, length = 4000, 50
    gen = torch.Generator().manual_seed(1)
    x_star = (torch.linspace(A_MIN + 1.0, A_MAX - 1.0, length) / 3.0).expand(n, -1)
    trace: list = []
    x_init = torch.randn((n, length), generator=gen)
    out = ddim_sample(
        point_mass_oracle(x_star, ab, trace), x_init, ab, steps, eta,
        lambda: torch.randn((n, length), generator=gen), x0_range=(-3.0, 3.0),
    )  # fmt: skip
    torch.testing.assert_close(out, x_star, atol=1e-4, rtol=0.0)
    assert [k for k, _ in trace] == list(steps)
    resid = [(x.double() - math.sqrt(ab[k]) * x_star.double()) / math.sqrt(1.0 - ab[k]) for k, x in trace]
    for i, r in enumerate(resid):
        if eta == 0.0:
            torch.testing.assert_close(r, x_init.double(), atol=2e-3, rtol=0.0)
        else:
            assert abs(float(r.mean())) < 0.01 and abs(float(r.var()) - 1.0) < 0.02  # 200k values: 5 std errors
            if i > 0:  # corr(r_prev, r_k) = sqrt(1 - sigma^2 / (1 - ab_prev)), DDIM eq. 16 with eta = 1
                ab_t, ab_p = ab[steps[i - 1]], ab[steps[i]]
                sigma2 = (1.0 - ab_p) / (1.0 - ab_t) * (1.0 - ab_t / ab_p)
                rho = float(torch.corrcoef(torch.stack([r.flatten(), resid[i - 1].flatten()]))[0, 1])
                assert rho == pytest.approx(math.sqrt(1.0 - sigma2 / (1.0 - ab_p)), abs=0.01)


def test_eta0_is_deterministic_given_initial_noise():
    torch.manual_seed(0)
    net = Denoiser(seq_len=50, channels=16, n_blocks=2, kernel=5, hidden=8).eval()
    torch.nn.init.normal_(net.out.weight, std=0.1)  # a non-trivial (untrained) noise predictor
    hist, vl = torch.randn(64, 30, 3), torch.randn(64, 50)
    ctx = net.context(hist)
    ab, steps = cosine_alpha_bar(1000), sampling_steps(1000, 20)

    @torch.no_grad()
    def eps_fn(x: torch.Tensor, k: int) -> torch.Tensor:
        return net.denoise(x, torch.full((1,), k), ctx, vl)

    def no_noise() -> torch.Tensor:
        raise AssertionError("eta = 0 must not draw noise")

    x_init = torch.randn(64, 50, generator=torch.Generator().manual_seed(3))
    a = ddim_sample(eps_fn, x_init.clone(), ab, steps, 0.0, no_noise, (-5.0, 5.0))
    b = ddim_sample(eps_fn, x_init.clone(), ab, steps, 0.0, None, (-5.0, 5.0))
    assert torch.equal(a, b) and torch.isfinite(a).all()
    c, d = (
        ddim_sample(eps_fn, x_init.clone(), ab, steps, 1.0, lambda g=g: torch.randn(64, 50, generator=g))
        for g in (torch.Generator().manual_seed(1), torch.Generator().manual_seed(2))
    )
    assert not torch.allclose(c, d)  # eta = 1 draws fresh noise at every step


# ----------------------------------------------------------------------------- the model
def test_fit_log_is_json_serialisable(tiny):
    model, log = tiny
    assert model.fitted and log["n_train"] == 1050 and log["epochs"] == 2
    assert {"seconds", "val_loss", "val_nll", "history", "config", "n_params"} <= set(log)
    assert log["val_loss"] == log["best_val"] and np.isfinite(log["val_nll"])
    assert log["config"]["ema_decay"] == 0.999
    json.loads(json.dumps(log))


def test_fit_is_deterministic(synth_small_seq, tiny):
    model, _ = tiny
    again = DiffusionCF(**TINY, train=TINY_TRAIN)
    again.fit(synth_small_seq.windows["train"], synth_small_seq.windows["val"], seed=0)
    for key, value in model.get_state()["net"].items():
        assert torch.equal(value, again.get_state()["net"][key]), key


def test_fit_requires_sequences(synth_small_seq):
    train = synth_small_seq.windows["train"]
    model = DiffusionCF(**TINY, train=TINY_TRAIN)
    for missing in ("y_seq", "v_lead_future"):
        with pytest.raises(ValueError, match="y_seq and v_lead_future"):
            model.fit(dataclasses.replace(train, **{missing: None}))
    with pytest.raises(ValueError, match="seq_len"):
        DiffusionCF(**TINY, seq_len=40).fit(train)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_dist(train.X[:3])


def test_predict_dist_reproducible_and_batch_independent(synth_small_seq, tiny):
    model, _ = tiny
    w = synth_small_seq.windows["test"]
    d1 = model.predict_dist(w.X, v_lead_future=w.v_lead_future)
    assert isinstance(d1, SampleDist) and d1.samples.shape == (len(w), 8)
    assert d1.epistemic_var is None and np.array_equal(d1.aleatoric_var, d1.var())
    assert np.all((d1.samples >= A_MIN) & (d1.samples <= A_MAX))
    for batch_size in (8192, 5, 37):
        d2 = model.predict_dist(w.X, v_lead_future=w.v_lead_future, batch_size=batch_size)
        np.testing.assert_array_equal(d2.samples, d1.samples)
    part = model.predict_dist(w.X[:50], v_lead_future=w.v_lead_future[:50])  # rows keep their noise streams
    np.testing.assert_array_equal(part.samples, d1.samples[:50])
    other = clone(model, predict_seed=1).predict_dist(w.X, v_lead_future=w.v_lead_future)
    assert not np.allclose(other.samples, d1.samples)


def test_sample_sequences_shapes_and_constant_leader(synth_small_seq, tiny):
    model, _ = tiny
    w = synth_small_seq.windows["test"]
    x, vlf = w.X[:20], w.v_lead_future[:20]
    seq = model.sample_sequences(x, vlf, n_samples=3)
    assert seq.shape == (20, 3, 50) and seq.dtype == np.float32
    assert model.sample_sequences(x[0], vlf[0]).shape == (1, 8, 50)
    first = model.predict_dist(x, v_lead_future=vlf).samples
    np.testing.assert_array_equal(model.sample_sequences(x, vlf)[:, :, 0], first)
    const = np.repeat((x[:, -1, 2] - x[:, -1, 1])[:, None], 50, axis=1)  # v_lead = v_t - dv_t
    np.testing.assert_array_equal(model.sample_sequences(x, None), model.sample_sequences(x, const))
    np.testing.assert_array_equal(model.predict_dist(x).samples, model.sample_sequences(x, const)[:, :, 0])
    gen_a, gen_b = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    same = [model.sample_sequences(x, vlf, generator=g) for g in (gen_a, gen_b)]
    np.testing.assert_array_equal(*same)
    with pytest.raises(ValueError, match="v_lead_future"):
        model.sample_sequences(x, vlf[:, :10])


def test_default_target_is_bit_identical_to_the_first_step_law(synth_small_seq, tiny):
    """``target="a"`` (default) is the law of the code before the option: SampleDist of the first sampled step."""
    model, _ = tiny
    w = synth_small_seq.windows["test"]
    seq = model.sample_sequences(w.X, w.v_lead_future)
    before = SampleDist(seq[:, :, 0].astype(np.float64), epistemic_var=None)  # predict_dist before the target option
    y = w.y.astype(np.float64)
    for dist in (model.predict_dist(w.X, v_lead_future=w.v_lead_future),
                 model.predict_dist(w.X, v_lead_future=w.v_lead_future, target="a")):  # fmt: skip
        assert dist.samples.dtype == np.float64 and dist.epistemic_var is None
        np.testing.assert_array_equal(dist.samples, before.samples)
        np.testing.assert_array_equal(dist.aleatoric_var, before.aleatoric_var)
        np.testing.assert_array_equal(dist.log_prob(y), before.log_prob(y))
        np.testing.assert_array_equal(dist.crps(y), before.crps(y))


def test_mean_1s_law_is_the_mean_of_the_first_ten_sampled_steps(synth_small_seq, tiny):
    model, _ = tiny
    w = synth_small_seq.windows["test"]
    seq = model.sample_sequences(w.X, w.v_lead_future)  # same noise streams (predict_seed) as predict_dist
    dist = model.predict_dist(w.X, v_lead_future=w.v_lead_future, target="mean_1s")
    assert isinstance(dist, SampleDist) and dist.samples.shape == (len(w), 8) and dist.epistemic_var is None
    np.testing.assert_array_equal(dist.samples, seq[:, :, :MEAN_TARGET_STEPS].astype(np.float64).mean(axis=2))
    np.testing.assert_array_equal(dist.aleatoric_var, dist.var())
    assert not np.allclose(dist.samples, seq[:, :, 0])
    for batch_size in (5, 37):  # same noise per row whatever the chunking; float32 arithmetic may differ by an ulp
        again = model.predict_dist(w.X, v_lead_future=w.v_lead_future, batch_size=batch_size, target="mean_1s")
        np.testing.assert_allclose(again.samples, dist.samples, rtol=0.0, atol=1e-5)
    const = model.predict_dist(w.X[:10], target="mean_1s")  # constant leader, as sample_sequences(v_lead_future=None)
    np.testing.assert_array_equal(const.samples, model.sample_sequences(w.X[:10])[:, :, :10].astype(np.float64).mean(axis=2))
    with pytest.raises(ValueError, match="targets"):
        model.predict_dist(w.X[:2], v_lead_future=w.v_lead_future[:2], target="fwd_diff")
    short = clone(model, seq_len=5, replan_every=5)  # the network does not depend on the sequence length
    with pytest.raises(ValueError, match="needs 10 sampled steps"):
        short.predict_dist(w.X[:2], v_lead_future=w.v_lead_future[:2, :5], target="mean_1s")
    assert short.predict_dist(w.X[:2], v_lead_future=w.v_lead_future[:2, :5]).samples.shape == (2, 8)


def test_save_load_reproduces_predictions(tmp_path, synth_small_seq, tiny):
    model, _ = tiny
    w = synth_small_seq.windows["test"]
    path = model.save(tmp_path / "diffusion.pt")
    loaded = load_model(path)
    assert isinstance(loaded, DiffusionCF) and loaded.fitted and loaded.init_kwargs == model.init_kwargs
    np.testing.assert_array_equal(
        loaded.predict_dist(w.X, v_lead_future=w.v_lead_future).samples,
        model.predict_dist(w.X, v_lead_future=w.v_lead_future).samples,
    )
    assert all(torch.is_tensor(v) for v in model.get_state()["net"].values())


# -------------------------------------------------------------------------- rollout sampler
def recording(model: DiffusionCF) -> list:
    """Record the (hist, v_lead_future) of every sampling call of ``model``."""
    calls: list = []
    original = model._sample

    def wrapper(hist, vlf, n_samples, generator, batch_size):
        calls.append((hist.clone(), vlf.clone()))
        assert n_samples == 1
        return original(hist, vlf, n_samples, generator, batch_size)

    model._sample = wrapper
    return calls


def test_rollout_replans_at_multiples_of_replan_every(synth_small_seq, tiny):
    model = clone(tiny[0], replan_every=5)
    calls = recording(model)
    hist = torch.as_tensor(synth_small_seq.windows["test"].X[:6])
    v_lead = torch.linspace(5.0, 15.0, 30).expand(6, -1)
    gen = torch.Generator().manual_seed(0)
    state = model.init_sampler(6, gen, v_lead=v_lead)
    plans = []
    for k in range(12):
        a = model.sample_step(hist, state, gen)
        assert a.shape == (6,) and a.dtype == torch.float32 and torch.isfinite(a).all()
        plans.append(state["plan"])
        assert len(calls) == k // 5 + 1  # re-plans exactly at k = 0, 5, 10
        assert torch.equal(a, state["plan"][:, k % 5])
    assert state["k"] == 12
    assert all(plans[k] is plans[0] for k in range(5)) and plans[5] is not plans[4] and plans[10] is not plans[9]
    assert not torch.equal(plans[5], plans[0])


def test_rollout_pads_leader_speed_and_constant_leader(synth_small_seq, tiny):
    model = clone(tiny[0], replan_every=5)
    calls = recording(model)
    hist = torch.as_tensor(synth_small_seq.windows["test"].X[:4])
    v_lead = torch.arange(8, dtype=torch.float32).expand(4, -1) + torch.arange(4.0)[:, None]  # (4, T_total = 8)
    gen = torch.Generator().manual_seed(0)
    state = model.init_sampler(4, gen, mode="per_trajectory", v_lead=v_lead)
    for _ in range(6):
        model.sample_step(hist, state, gen)
    for (_, vlf), k in zip(calls, (0, 5)):
        cols = np.minimum(np.arange(k + 1, k + 51), 7)  # v_lead[:, k+1 : k+51], last value repeated
        np.testing.assert_array_equal(vlf.numpy(), v_lead.numpy()[:, cols])
    calls.clear()
    state = model.init_sampler(4, gen, v_lead=None)
    a = model.sample_step(hist, state, gen)
    assert a.shape == (4,) and torch.isfinite(a).all()
    const = (hist[:, -1, 2] - hist[:, -1, 1])[:, None].expand(-1, 50)  # v_lead = v_t - dv_t
    np.testing.assert_array_equal(calls[0][1].numpy(), const.numpy())
    with pytest.raises(ValueError, match="mode"):
        model.init_sampler(4, gen, mode="bogus")
    with pytest.raises(ValueError, match="v_lead"):
        model.init_sampler(4, gen, v_lead=v_lead[:3])


# --------------------------------------------------------------------- E4 sanity (slow)
# The generator's noise is white, sigma = 0.4 m/s^2 per step; the IDM feedback makes the true 5 s speed change
# much less variable than white noise would (std 0.136 instead of 0.28 m/s). "true law" values below come from
# simulating the true StochasticIDM forward from the same 400 windows (32 samples, 20 repetitions).
EVAL_WINDOWS = 400  # seeded subsample of the synth_e4_seq test windows
NOISE_VAR = 0.4**2  # generator noise variance per step [(m/s^2)^2]


@pytest.fixture(scope="module")
def e4(synth_e4_seq) -> dict:
    """Default DiffusionCF trained on synth_e4_seq; 32 sampled sequences for EVAL_WINDOWS test windows."""
    train, val, test = (synth_e4_seq.windows[k] for k in ("train", "val", "test"))
    model = DiffusionCF(train={"max_seconds": 220.0})  # guard only: the 12 default epochs normally end earlier
    log = model.fit(train, val, seed=0)
    idx = np.sort(np.random.default_rng(0).choice(len(test), EVAL_WINDOWS, replace=False))
    started = time.perf_counter()
    seq = model.sample_sequences(test.X[idx], test.v_lead_future[idx]).astype(np.float64)  # (400, 32, 50)
    rate = seq.shape[0] * seq.shape[1] * len(model.steps) / (time.perf_counter() - started)
    print(
        f"\nDiffusionCF E4: {log['n_params']} parameters, {log['epochs']} epochs in {log['seconds']:.0f} s "
        f"(fit incl. val_nll {log['total_seconds']:.0f} s), val loss {log['val_loss']:.5f}, "
        f"val_nll {log['val_nll']:.3f}, sampling {rate:.0f} sequences x steps / s"
    )
    return dict(
        model=model, log=log, seq=seq, x=test.X[idx], y_seq=test.y_seq[idx].astype(np.float64),
        a_det=synth_e4_seq.truth["test"]["a_det"][idx],
    )  # fmt: skip


@pytest.mark.slow
def test_e4_training_log(e4):
    log, hist = e4["log"], e4["log"]["history"]
    assert [h["ema"] for h in hist] == [False] + [True] * (len(hist) - 1)  # one EMA warm-up epoch
    assert log["val_loss"] == log["best_val"] <= min(h["val"] for h in hist[1:]) + DEFAULT_TRAIN["min_delta"]
    assert log["val_loss"] < hist[1]["val"] and log["val_loss"] < 0.315  # measured 0.3109
    assert np.isfinite(log["val_nll"]) and log["val_nll"] < 0.75  # measured 0.555; true law N(a_det, 0.16): 0.50
    json.dumps(log)


@pytest.mark.slow
def test_e4_first_step_mean_and_variance(e4):
    first = e4["seq"][:, :, 0]
    rmse = float(np.sqrt(np.mean((first.mean(axis=1) - e4["a_det"]) ** 2)))
    var = float(first.var(axis=1, ddof=1).mean())
    print(f"\nfirst step: RMSE(sample mean, a_det) {rmse:.4f} (MC floor 0.071), mean sample variance {var:.4f}")
    assert rmse < 0.15  # measured 0.085; true law with 32 samples 0.071
    assert abs(var / NOISE_VAR - 1.0) < 0.25  # measured 0.154 (-3.7 %), also within 15 %; true law 0.159


@pytest.mark.slow
def test_e4_generated_noise_is_white(e4):
    dev = e4["seq"] - e4["seq"].mean(axis=1, keepdims=True)  # (N, S, L) minus the per-window sample mean
    rho1 = float((dev[:, :, 1:] * dev[:, :, :-1]).mean() / (dev * dev).mean())
    per_step = dev.var(axis=1, ddof=1).mean(axis=0)  # (L,)
    ratio = float(per_step[-10:].mean() / per_step[:10].mean())
    print(f"\nwhiteness: lag-1 autocorrelation {rho1:.4f}, variance ratio last / first ten steps {ratio:.3f}")
    assert abs(rho1) < 0.15  # measured -0.017; true law -0.027
    assert 0.6 < ratio < 1.6  # measured 1.01; true law 1.02


@pytest.mark.slow
def test_e4_horizon_calibration(e4):
    dv_samples = DT * e4["seq"].sum(axis=2)  # (N, S) speed change over 5 s [m/s]
    dv_true = DT * e4["y_seq"].sum(axis=1)
    lo, hi = np.quantile(dv_samples, [0.05, 0.95], axis=1)
    picp = float(np.mean((dv_true >= lo) & (dv_true <= hi)))
    rmse = float(np.sqrt(np.mean((dv_samples.mean(axis=1) - dv_true) ** 2)))  # = RMSE of the speed after 5 s
    spread = float(dv_samples.std(axis=1, ddof=1).mean())
    print(f"\n5 s speed change: PICP(5-95 %) {picp:.4f}, RMSE {rmse:.3f} m/s, mean sample std {spread:.3f} m/s")
    assert 0.80 <= picp <= 0.97  # measured 0.93; true law with 32 samples 0.86
    assert rmse < 0.3  # measured 0.159 m/s; true law 0.142, zero-change forecast 3.7
    assert spread < 0.28  # measured 0.193; true law 0.136, white noise without IDM feedback 0.28


@pytest.mark.slow
def test_e4_leader_conditioning_matters(e4):
    model, sub = e4["model"], slice(0, 200)
    dv_true = DT * e4["y_seq"][sub].sum(axis=1)
    rmse_true = float(np.sqrt(np.mean((DT * e4["seq"][sub, :8].sum(axis=2).mean(axis=1) - dv_true) ** 2)))
    const = model.sample_sequences(e4["x"][sub], None, n_samples=8).astype(np.float64)  # constant-speed leader
    rmse_const = float(np.sqrt(np.mean((DT * const.sum(axis=2).mean(axis=1) - dv_true) ** 2)))
    print(f"\n5 s speed change RMSE (200 windows, 8 samples): true leader {rmse_true:.3f}, const. {rmse_const:.3f} m/s")
    assert rmse_const > 3.0 * rmse_true  # measured 0.145 vs 2.41 m/s; true law 0.13 vs 2.42
