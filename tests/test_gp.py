"""GPResidualIDM (docs/design.md, section 7): export vs GPyTorch, sampler, persistence, E4 sanity."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from cf_unc.dist import GaussianDist
from cf_unc.eval.metrics import picp
from cf_unc.models import CFModel, build_model
from cf_unc.models.gp_residual_idm import GPResidualIDM, kmeans_inducing
from cf_unc.physics.idm import IDMParams, calibrate_idm, idm_accel

N_SAMPLER = 40_000
WRONG_IDM = IDMParams(v0=30.0, T=1.0, s0=2.0, a_max=0.8, b=1.8)  # true: T = 1.5, a_max = 1.2
NOISE_VAR = 0.16  # E4 data: sigma = 0.4 m/s^2


def constant_histories(states: np.ndarray, hist_len: int = 30) -> np.ndarray:
    """(K, 3) states -> (K, hist_len, 3) float32 histories that repeat each state (only the last sample matters)."""
    return np.repeat(np.asarray(states, dtype=np.float32)[:, None, :], hist_len, axis=1)


def frozen_share(model: GPResidualIDM, hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Share of the predictive variance carried by the frozen inducing values, and Var[f] (standardised)."""
    post = model.posterior
    a = post.interp(model.standardize(model.last_states(hist)))
    _, var_f = post.moments(a)
    frozen = (post.L_S.mT @ a).square().sum(dim=0)
    return (frozen / (var_f + post.noise)).numpy(), var_f.numpy()


@pytest.fixture(scope="module")
def tiny(synth_small, true_idm) -> GPResidualIDM:
    model = GPResidualIDM(idm=true_idm.to_dict(), n_inducing=32, max_epochs=3, batch_size=512)
    model.fit_log = model.fit(synth_small.windows["train"], synth_small.windows["val"], seed=0)
    return model


# ----------------------------------------------------------------------------------------- fast tests
def test_config_mirrors_constructor_defaults(true_idm):
    path = Path(__file__).resolve().parents[1] / "configs" / "model" / "gp_residual_idm.yaml"
    cfg = OmegaConf.to_container(OmegaConf.create(path.read_text(encoding="utf-8")))
    assert cfg["name"] == "gp_residual_idm" and cfg["needs_idm"] is True and cfg["train"] == {}
    params = inspect.signature(GPResidualIDM.__init__).parameters
    assert cfg["kwargs"] == {k: p.default for k, p in params.items() if k not in ("self", "idm")}
    model = build_model(cfg["name"], idm=true_idm.to_dict(), **cfg["kwargs"])
    assert model.init_kwargs["idm"] == true_idm.to_dict()


def test_fit_log_and_registry(tiny, true_idm):
    log = tiny.fit_log
    json.dumps(log)
    for key in ("val_nll", "n_train", "seconds", "epochs", "n_inducing", "hyper", "history"):
        assert key in log
    assert log["n_inducing"] == 32 and 1 <= log["epochs"] <= 3 and math.isfinite(log["val_nll"])
    assert tiny.fitted and not GPResidualIDM.uses_history and GPResidualIDM.has_epistemic
    assert isinstance(build_model("gp_residual_idm", idm=true_idm.to_dict()), GPResidualIDM)
    with pytest.raises(ValueError):
        GPResidualIDM(idm=true_idm.to_dict(), inducing_init="grid")
    with pytest.raises(ValueError, match="sampler mode"):
        tiny.init_sampler(3, torch.Generator().manual_seed(0), mode="per_member")


def test_export_equals_gpytorch(tiny):
    """Mean / latent variance / predictive variance of the plain tensors == GPyTorch model(x), likelihood(model(x))."""
    gen = torch.Generator().manual_seed(3)
    xn = torch.cat(
        [
            1.5 * torch.randn((500, 3), generator=gen, dtype=torch.float64),  # mostly inside the data
            6.0 * torch.randn((100, 3), generator=gen, dtype=torch.float64),  # mostly far outside
            tiny.posterior.Z[:10],  # at inducing inputs
        ]
    )
    with torch.no_grad():
        latent = tiny.gp_model(xn)
        observed = tiny.gp_likelihood(latent)
        mean_gp, var_gp, var_obs_gp = latent.mean, latent.variance, observed.variance
    mean, var = tiny.posterior.latent(xn)
    scale = float(mean_gp.abs().max())
    torch.testing.assert_close(mean, mean_gp, rtol=1e-6, atol=1e-6 * scale)
    torch.testing.assert_close(var, var_gp, rtol=1e-6, atol=0.0)
    torch.testing.assert_close(var + tiny.posterior.noise, var_obs_gp, rtol=1e-6, atol=0.0)
    # the latent variance is not trivially the prior: it varies over the inputs
    assert float(var.max() - var.min()) > 0.1 * float(var.max())


def test_predict_dist_contract(tiny, synth_small):
    test = synth_small.windows["test"]
    dist = tiny.predict_dist(test.X)
    assert isinstance(dist, GaussianDist) and len(dist) == len(test)
    for arr in (dist.mu, dist.sigma, dist.aleatoric_var, dist.epistemic_var):
        assert arr.dtype == np.float64 and arr.shape == (len(test),) and np.all(np.isfinite(arr))
    assert np.all(dist.sigma > 0) and np.all(dist.epistemic_var > 0) and np.all(dist.aleatoric_var > 0)
    assert np.all(dist.aleatoric_var == dist.aleatoric_var[0])  # homoscedastic GP likelihood
    np.testing.assert_allclose(dist.aleatoric_var + dist.epistemic_var, dist.sigma**2, rtol=1e-10)
    # mean = IDM + residual mean + GP correction; the correction is a smooth function of the last sample
    last = test.X[:, -1, :].astype(np.float64)
    a_idm = idm_accel(last[:, 0], last[:, 1], last[:, 2], tiny.idm)
    assert np.all(np.abs(dist.mu - a_idm) < 1.0)


def test_inputs_batch_size_and_single_history(tiny, synth_small):
    x = synth_small.windows["test"].X
    ref = tiny.predict_dist(x)
    small_batches = tiny.predict_dist(torch.as_tensor(x), batch_size=37)
    np.testing.assert_allclose(small_batches.mu, ref.mu, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(small_batches.sigma, ref.sigma, rtol=1e-12, atol=0.0)
    as_f64 = tiny.predict_dist(x.astype(np.float64))  # rounded to float32 like histories: identical
    np.testing.assert_array_equal(as_f64.mu, ref.mu)
    single = tiny.predict_dist(x[5])
    assert len(single) == 1
    np.testing.assert_allclose(single.mu, ref.mu[5:6], rtol=1e-12)
    np.testing.assert_allclose(single.epistemic_var, ref.epistemic_var[5:6], rtol=1e-10)
    # only the last sample of the history is used
    shuffled = x.copy()
    shuffled[:, :-1] = x[::-1, :-1]
    np.testing.assert_array_equal(tiny.predict_dist(shuffled).mu, ref.mu)


def test_save_load_exact(tiny, synth_small, tmp_path):
    x = synth_small.windows["test"].X
    path = tiny.save(tmp_path / "gp.pt")
    loaded = CFModel.load(path)
    assert isinstance(loaded, GPResidualIDM) and loaded.gp_model is None
    a, b = tiny.predict_dist(x), loaded.predict_dist(x)
    for name in ("mu", "sigma", "aleatoric_var", "epistemic_var"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    hist = torch.as_tensor(x[:500])
    for mode in ("per_step", "per_trajectory"):
        draws = []
        for model in (tiny, loaded):
            gen = torch.Generator().manual_seed(7)
            state = model.init_sampler(500, gen, mode)
            draws.append(model.sample_step(hist, state, gen))
        torch.testing.assert_close(draws[0], draws[1], rtol=0.0, atol=0.0)


def test_fit_is_deterministic_and_random_init(synth_small, true_idm):
    train, val = synth_small.windows["train"], synth_small.windows["val"]
    kw = dict(idm=true_idm.to_dict(), n_inducing=16, max_epochs=1, batch_size=512, max_train=1500)
    first, second = GPResidualIDM(**kw), GPResidualIDM(**kw)
    log = first.fit(train, val, seed=4)
    second.fit(train, val, seed=4)
    assert log["n_train"] == 1500
    for name, value in first.posterior.state().items():
        torch.testing.assert_close(value, second.posterior.state()[name], rtol=0.0, atol=0.0)
    rand = GPResidualIDM(**{**kw, "inducing_init": "random", "learn_inducing": False})
    rand.fit(train, val, seed=4)
    z = rand.posterior.Z.numpy()
    assert z.shape == (16, 3) and np.unique(z, axis=0).shape[0] == 16
    assert not isinstance(rand.gp_model.variational_strategy.inducing_points, torch.nn.Parameter)
    x_train = rand.standardize(rand.last_states(train.X)).numpy()  # random init: Z is a subset of the inputs
    assert all(np.any(np.all(x_train == row, axis=1)) for row in z)


def test_kmeans_inducing(synth_small):
    last = synth_small.windows["train"].X[:, -1, :].astype(np.float64)
    x = (last - last.mean(axis=0)) / last.std(axis=0)
    for k in (32, 300):
        z = kmeans_inducing(x, k, np.random.default_rng(0))
        assert z.shape == (k, 3) and np.unique(z, axis=0).shape[0] == k
        assert np.all(z >= x.min(axis=0)) and np.all(z <= x.max(axis=0))
    np.testing.assert_array_equal(kmeans_inducing(x, 32, np.random.default_rng(1)), kmeans_inducing(x, 32, np.random.default_rng(1)))
    # Lloyd iterations lower the k-means objective of the seeding
    seeds = kmeans_inducing(x, 32, np.random.default_rng(2), n_lloyd=0)
    refined = kmeans_inducing(x, 32, np.random.default_rng(2), n_lloyd=10)

    def inertia(c: np.ndarray) -> float:
        return float(np.min(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1).sum())

    assert inertia(refined) < inertia(seeds)
    with pytest.raises(ValueError):
        kmeans_inducing(x[:5], 6, np.random.default_rng(0))


def test_sampler_moments_match_marginal_law(tiny, synth_small):
    """Both modes, one fixed history x 40 000 trajectories: mean and variance of the marginal predictive law.

    Tolerances are 5 standard errors (Gaussian draws): |mean - mu| < 5 sigma / sqrt(n),
    |var / sigma^2 - 1| < 5 sqrt(2 / n) = 0.035.
    """
    x0 = synth_small.windows["test"].X[:1]
    law = tiny.predict_dist(x0)
    mu, sd = float(law.mu[0]), float(law.sigma[0])
    hist = torch.as_tensor(np.repeat(x0, N_SAMPLER, axis=0))
    for mode, seed in (("per_trajectory", 11), ("per_step", 12)):
        gen = torch.Generator().manual_seed(seed)
        state = tiny.init_sampler(N_SAMPLER, gen, mode)
        draws = tiny.sample_step(hist, state, gen)
        assert draws.shape == (N_SAMPLER,) and draws.dtype == torch.float32
        a = draws.double().numpy()
        assert abs(a.mean() - mu) < 5.0 * sd / math.sqrt(N_SAMPLER), mode
        assert abs(a.var() / sd**2 - 1.0) < 5.0 * math.sqrt(2.0 / N_SAMPLER), mode
    # reproducible from the generator alone
    runs = []
    for _ in range(2):
        gen = torch.Generator().manual_seed(5)
        runs.append(tiny.sample_step(hist[:1000], tiny.init_sampler(1000, gen, "per_trajectory"), gen))
    torch.testing.assert_close(runs[0], runs[1], rtol=0.0, atol=0.0)


def test_per_trajectory_freezes_the_latent_draw(tiny, synth_small):
    """Two steps with the same history and state: corr = |L_S^T A|^2 / (Var[f] + noise) (frozen share).

    The check input is out of distribution: inducing inputs pushed outwards by 0.5 standardised units, kept
    only if outside the training range; the one with the largest frozen share is used.
    """
    post, sc = tiny.posterior, tiny.scalers
    last = synth_small.windows["train"].X[:, -1, :]
    z = post.Z
    outward = z + 0.5 * z / z.norm(dim=1, keepdim=True)
    states = (outward * sc["x_std"] + sc["x_mean"]).numpy()
    outside = np.any((states < last.min(axis=0)) | (states > last.max(axis=0)), axis=1)
    states = states[outside & (states[:, 0] > 0.0) & (states[:, 2] >= 0.0)]
    assert len(states) > 0
    hists = constant_histories(states)
    share, _ = frozen_share(tiny, hists)
    best = int(np.argmax(share))
    hist_row = hists[best]
    law = tiny.predict_dist(hist_row)
    prior = float(sc["r_std"]) ** 2 * float(post.outputscale)
    assert law.epistemic_var[0] > 0.5 * prior  # Var[f] is large here (measured 0.084 vs prior 0.100) ...
    assert share[best] > 0.2  # ... and mostly carried by the frozen inducing values (measured 0.40)

    hist = torch.as_tensor(np.repeat(hist_row[None], N_SAMPLER, axis=0))
    gen = torch.Generator().manual_seed(21)
    state = tiny.init_sampler(N_SAMPLER, gen, "per_trajectory")
    a1, a2 = (tiny.sample_step(hist, state, gen).double().numpy() for _ in range(2))
    corr = np.corrcoef(a1, a2)[0, 1]
    assert abs(corr - share[best]) < 0.03  # measured 0.408 vs 0.404; SE of corr <= 1 / sqrt(n) = 0.005
    gen = torch.Generator().manual_seed(22)
    b1, b2 = (tiny.sample_step(hist, None, gen).double().numpy() for _ in range(2))
    assert abs(np.corrcoef(b1, b2)[0, 1]) < 0.03  # per_step: independent draws (measured -0.002)
    with pytest.raises(ValueError, match="sampler state"):
        tiny.sample_step(hist[:10], state, gen)


# ----------------------------------------------------------------------------------------- slow: E4
@pytest.fixture(scope="module")
def e4_cal_idm(synth_e4) -> IDMParams:
    params, _ = calibrate_idm(synth_e4.events["train"], objective="accel")
    return params


@pytest.fixture(scope="module")
def e4_gp(synth_e4, e4_cal_idm) -> GPResidualIDM:
    model = GPResidualIDM(idm=e4_cal_idm.to_dict())
    model.fit_log = model.fit(synth_e4.windows["train"], synth_e4.windows["val"], seed=0)
    return model


@pytest.fixture(scope="module")
def e4_test_dist(e4_gp, synth_e4) -> GaussianDist:
    return e4_gp.predict_dist(synth_e4.windows["test"].X)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


@pytest.mark.slow
def test_e4_aleatoric_and_mean(e4_gp, e4_test_dist, synth_e4):
    alea = float(e4_test_dist.aleatoric_var.mean())
    assert abs(alea / NOISE_VAR - 1.0) < 0.15  # measured 0.158
    assert rmse(e4_test_dist.mu, synth_e4.truth["test"]["a_det"]) < 0.1  # measured 0.016
    assert e4_gp.fit_log["seconds"] < 120.0  # measured ~10 s (4 threads)


@pytest.mark.slow
def test_e4_pit_is_roughly_uniform(e4_test_dist, synth_e4):
    y = synth_e4.windows["test"].y
    lo, hi = e4_test_dist.interval(0.9)
    assert abs(picp(lo, hi, y) - 0.9) < 0.03  # measured 0.898
    pit = e4_test_dist.pit(y)
    hist, _ = np.histogram(pit, bins=10, range=(0.0, 1.0))
    assert np.all(np.abs(hist / len(y) - 0.1) < 0.02)  # measured max deviation 0.007 (KS p = 0.08)


@pytest.mark.slow
def test_e4_residual_corrects_a_wrong_idm(synth_e4):
    model = GPResidualIDM(idm=WRONG_IDM.to_dict())
    model.fit(synth_e4.windows["train"], synth_e4.windows["val"], seed=0)
    test = synth_e4.windows["test"]
    a_det = synth_e4.truth["test"]["a_det"]
    dist = model.predict_dist(test.X)
    last = test.X[:, -1, :].astype(np.float64)
    rmse_idm = rmse(idm_accel(last[:, 0], last[:, 1], last[:, 2], WRONG_IDM), a_det)  # measured 0.397
    rmse_gp = rmse(dist.mu, a_det)  # measured 0.043
    assert 3.0 * rmse_gp < rmse_idm
    assert abs(float(dist.aleatoric_var.mean()) / NOISE_VAR - 1.0) < 0.15  # measured 0.163


@pytest.mark.slow
def test_e4_epistemic_shrinks_with_data(synth_e4, e4_cal_idm, e4_test_dist):
    train = synth_e4.windows["train"]
    drivers = np.unique(train.driver)
    chosen = np.random.default_rng(0).choice(drivers, size=round(0.1 * drivers.size), replace=False)
    small = train.subset(np.isin(train.driver, chosen))  # 12 of 119 drivers, 3600 windows
    model = GPResidualIDM(idm=e4_cal_idm.to_dict())
    model.fit(small, synth_e4.windows["val"], seed=0)
    small_epi = float(model.predict_dist(synth_e4.windows["test"].X).epistemic_var.mean())
    full_epi = float(e4_test_dist.epistemic_var.mean())
    # measured: 3.97e-2 vs 1.52e-3 (ratio 26; 25.8 when the small set is trained with batch_size=256)
    assert small_epi > 5.0 * full_epi


@pytest.mark.slow
def test_e4_epistemic_grows_off_support(e4_gp, e4_test_dist):
    far = constant_histories(np.array([[250.0, 15.0, 38.0], [250.0, -15.0, 38.0], [200.0, 0.0, 45.0]]))
    dist = e4_gp.predict_dist(far)
    mean_test = float(e4_test_dist.epistemic_var.mean())
    assert np.all(dist.epistemic_var > 5.0 * mean_test)  # measured 0.0836 vs 1.5e-3 (55x)
    prior = float(e4_gp.scalers["r_std"]) ** 2 * float(e4_gp.posterior.outputscale)
    np.testing.assert_allclose(dist.epistemic_var, prior, rtol=1e-3)  # K_zx -> 0: Var[f] -> k_xx (+ jitter 1e-6)
