"""HeteroMDN, DeepEnsemble, MCDropout and the homoscedastic baselines: fast contract tests.

Every configuration is trained once per module on ``synth_small`` with a tiny budget; the statistical
checks use fixed seeds and tolerances with the stated margins (docs/design.md, section 8).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from cf_unc.data.splits import bootstrap_drivers
from cf_unc.dist import GaussianDist, MixtureDist, PredDist
from cf_unc.models import MODEL_REGISTRY, SAMPLER_MODES, build_model, load_model
from cf_unc.models.baselines import IDMWhiteNoise, LSTMConstSigma
from cf_unc.models.deep_ensemble import DeepEnsemble
from cf_unc.models.hetero_mdn import HeteroMDN
from cf_unc.models.mc_dropout import MCDropout
from cf_unc.physics.idm import idm_accel
from cf_unc.train.trainer import TrainConfig

ROOT = Path(__file__).resolve().parents[1]
FAST = {"max_epochs": 3}
N_DRAWS = 20_000  # draws of the sampler moment checks
Z = 5.0  # tolerance of the statistical checks in standard errors (two-sided p ~ 6e-7 per check)

CONFIGS: dict[str, tuple[str, dict]] = {
    "hetero_k1": ("hetero_mdn", {"n_components": 1, "train": FAST}),
    "hetero_k3": ("hetero_mdn", {"n_components": 3, "train": FAST}),
    "ensemble": ("deep_ensemble", {"n_members": 3, "train": FAST}),
    "mc_dropout": ("mc_dropout", {"n_passes": 10, "train": FAST}),
    "lstm_const": ("lstm_const_sigma", {"train": FAST}),
    "idm_noise": ("idm_white_noise", {}),  # the true IDM parameters are injected from the fixture
}
# design.md, section 7: predictive law, aleatoric_var present, epistemic_var present
EXPECTED: dict[str, tuple[type[PredDist], bool, bool]] = {
    "hetero_k1": (GaussianDist, True, False),
    "hetero_k3": (MixtureDist, True, False),
    "ensemble": (MixtureDist, True, True),
    "mc_dropout": (MixtureDist, True, True),
    "lstm_const": (GaussianDist, True, True),
    "idm_noise": (GaussianDist, True, True),
}
MDN_BASED = ("hetero_k1", "hetero_k3", "ensemble", "mc_dropout")


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def fitted(synth_small, true_idm) -> dict[str, tuple]:
    """(model, fit log) of every configuration, trained once on synth_small (seed 0)."""
    w = synth_small.windows
    out = {}
    for key, (name, kwargs) in CONFIGS.items():
        if name == "idm_white_noise":
            kwargs = {**kwargs, "idm": true_idm.to_dict()}
        model = build_model(name, **kwargs)
        out[key] = (model, model.fit(w["train"], w["val"], seed=0))
    return out


@pytest.fixture(params=list(CONFIGS))
def case(request, fitted) -> tuple:
    return (request.param, *fitted[request.param])


@pytest.fixture(scope="module")
def test_x(synth_small) -> np.ndarray:
    return synth_small.windows["test"].X


def params_of(dist: PredDist) -> list[np.ndarray]:
    """Parameter arrays and extras of a law, for exact comparisons."""
    names = ("mu", "sigma") if isinstance(dist, GaussianDist) else ("weights", "mus", "sigmas")
    arrays = [getattr(dist, n) for n in names]
    return arrays + [a for a in (dist.aleatoric_var, dist.epistemic_var) if a is not None]


def moments(dist: PredDist, row: int) -> tuple[float, float, float]:
    """Mean, variance and fourth central moment of row ``row`` of a Gaussian or mixture law."""
    if isinstance(dist, GaussianDist):
        var = float(dist.sigma[row] ** 2)
        return float(dist.mu[row]), var, 3.0 * var**2
    w, mu, sd = dist.weights[row], dist.mus[row], dist.sigmas[row]
    mean = float(w @ mu)
    dev2, sd2 = (mu - mean) ** 2, sd**2
    return mean, float(w @ (dev2 + sd2)), float(w @ (dev2**2 + 6.0 * dev2 * sd2 + 3.0 * sd2**2))


def many_pass_copy(model: MCDropout, n_passes: int = 4000) -> MCDropout:
    """The same dropout network pooled over many passes (approximates the per-step dropout law)."""
    copy = MCDropout(**{**model.init_kwargs, "n_passes": n_passes})
    copy.set_state(model.get_state())
    return copy


# ------------------------------------------------------------------ fit and predictive law
def test_fit_log_and_predictive_law(case, synth_small, test_x):
    key, model, log = case
    w = synth_small.windows
    json.dumps(log)  # JSON-serialisable
    assert log["n_train"] == len(w["train"]) and log["seconds"] > 0.0 and math.isfinite(log["val_nll"])
    if key in ("hetero_k1", "hetero_k3", "mc_dropout", "lstm_const"):
        assert {"epochs", "best_epoch", "best_val", "history", "config"} <= set(log["fit"])  # fit_module log
    elif key == "ensemble":
        assert len(log["members"]) == 3 and all("fit" in m for m in log["members"])

    dist = model.predict_dist(test_x)
    cls, has_aleatoric, has_epistemic = EXPECTED[key]
    assert type(dist) is cls and len(dist) == len(test_x)
    mean, var = dist.mean(), dist.var()
    assert mean.dtype == np.float64 and np.all(np.isfinite(mean)) and np.all(np.isfinite(var)) and np.all(var > 0)
    assert (dist.aleatoric_var is not None) == has_aleatoric
    assert (dist.epistemic_var is not None) == has_epistemic
    if key in ("hetero_k1", "hetero_k3"):  # its own variance
        np.testing.assert_allclose(dist.aleatoric_var, var, rtol=1e-12)
    elif key in ("ensemble", "mc_dropout"):
        np.testing.assert_allclose(dist.aleatoric_var + dist.epistemic_var, var, rtol=1e-9)
    else:  # constant sigma, no epistemic part
        np.testing.assert_array_equal(dist.epistemic_var, 0.0)
        np.testing.assert_allclose(dist.aleatoric_var, model.sigma**2, rtol=1e-12)
        np.testing.assert_allclose(var, model.sigma**2, rtol=1e-12)
    # val_nll: mean NLL of the final predictive law on val, physical units
    val_dist = model.predict_dist(w["val"].X)
    assert log["val_nll"] == pytest.approx(float(-np.mean(val_dist.log_prob(w["val"].y))), rel=1e-12)


@pytest.mark.parametrize("key", ["hetero_k1", "hetero_k3", "mc_dropout"])
def test_early_stopping_uses_the_exact_nll(fitted, synth_small, key):
    """The restored weights score the best validation loss: physical NLL = normalised NLL + log(a_std)."""
    model, log = fitted[key]
    mdn = model.mdn if key == "mc_dropout" else model
    physical = log["val_nll_no_dropout"] if key == "mc_dropout" else log["val_nll"]
    y_std = float(mdn.normalizer.y_std)
    assert physical == pytest.approx(log["fit"]["best_val"] + math.log(y_std), abs=2e-5)  # float32 loss


def test_beta_nll_weighting_trains_but_early_stopping_uses_plain_nll(synth_small):
    w = synth_small.windows
    model = HeteroMDN(beta_nll=0.5, train=FAST)
    log = model.fit(w["train"], w["val"], seed=0)
    y_std = float(model.normalizer.y_std)
    assert all(math.isfinite(h["train"]) for h in log["fit"]["history"])
    assert log["val_nll"] == pytest.approx(log["fit"]["best_val"] + math.log(y_std), abs=2e-5)
    plain = HeteroMDN(train=FAST)
    plain.fit(w["train"], w["val"], seed=0)
    assert not np.array_equal(model.predict_dist(w["test"].X).mean(), plain.predict_dist(w["test"].X).mean())


# ------------------------------------------------------------------------------ inputs
def test_input_forms_and_batch_size(case, test_x):
    key, model, _ = case
    x = test_x[:50]
    ref = model.predict_dist(x)
    single = model.predict_dist(x[3])  # one (H, 3) history
    assert len(single) == 1
    for a, b in zip(params_of(single), params_of(ref[3])):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)
    from_torch = model.predict_dist(torch.as_tensor(x))
    for a, b in zip(params_of(from_torch), params_of(ref)):
        np.testing.assert_array_equal(a, b)
    # float32 GEMM blocking depends on the number of rows: differences of ~3e-8, far below the tolerance
    for batch_size in (1, 7, 10_000):
        other = model.predict_dist(x, batch_size=batch_size)
        for a, b in zip(params_of(other), params_of(ref)):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)
    with pytest.raises(ValueError):
        model.predict_dist(x[:, :, :2])


def test_mc_dropout_passes_are_shared_seeded_and_order_free(fitted, test_x):
    model = fitted["mc_dropout"][0]
    x = test_x[:40]
    m1, m2 = model.pass_masks()
    assert m1.shape == (10, 1, 64) and m2.shape == (10, 1, 64)
    w, mu, sd = model.predict_passes(x)
    assert w.shape == mu.shape == sd.shape == (10, 40, 1)
    # pass j is ONE network: evaluating it row by row (one shared mask pair) gives the same parameters
    for j in (0, 7):
        per_row = model.mdn.predict_params_np(x, batch_size=1, masks=(m1[j], m2[j]))
        np.testing.assert_allclose(per_row[1], mu[j], rtol=1e-5, atol=1e-7)
    # passes differ from each other and from the network without dropout
    assert np.ptp(mu[:, :, 0], axis=0).min() > 0
    # row order does not matter
    perm = np.random.default_rng(0).permutation(len(x))
    np.testing.assert_allclose(model.predict_dist(x[perm]).mean(), model.predict_dist(x).mean()[perm], rtol=1e-5, atol=1e-7)
    # the law depends on predict_seed only
    same = many_pass_copy(model, n_passes=10)
    np.testing.assert_array_equal(same.predict_dist(x).mean(), model.predict_dist(x).mean())
    other = MCDropout(**{**model.init_kwargs, "predict_seed": 1})
    other.set_state(model.get_state())
    assert not np.array_equal(other.predict_dist(x).mean(), model.predict_dist(x).mean())


# ----------------------------------------------------------------------- persistence
def test_save_load_roundtrip(case, tmp_path, test_x):
    key, model, _ = case
    path = model.save(tmp_path / f"{key}.pt")
    loaded = load_model(path)
    assert type(loaded) is type(model) and loaded.name == model.name and MODEL_REGISTRY[loaded.name] is type(model)
    assert loaded.init_kwargs == model.init_kwargs and loaded.fitted and loaded.to("cpu") is loaded
    a, b = model.predict_dist(test_x), loaded.predict_dist(test_x)
    assert type(a) is type(b)
    for x, y in zip(params_of(a), params_of(b)):
        np.testing.assert_array_equal(x, y)
    hist = torch.as_tensor(test_x[:16])
    draws = []
    for m in (model, loaded):
        g = torch.Generator().manual_seed(5)
        draws.append(m.sample_step(hist, m.init_sampler(16, g, "per_trajectory"), g))
    assert torch.equal(draws[0], draws[1])


def test_unfitted_models_refuse_to_predict(true_idm, test_x):
    for model in (HeteroMDN(), DeepEnsemble(n_members=2), MCDropout(), LSTMConstSigma(), IDMWhiteNoise(true_idm.to_dict())):
        with pytest.raises(RuntimeError):
            model.predict_dist(test_x[:2])


def test_invalid_arguments(true_idm):
    for bad in (
        lambda: HeteroMDN(n_components=0),
        lambda: HeteroMDN(n_components=3, beta_nll=0.5),
        lambda: HeteroMDN(beta_nll=-1.0),
        lambda: HeteroMDN(train={"max_epoch": 3}),  # typo of max_epochs
        lambda: DeepEnsemble(n_members=0),
        lambda: DeepEnsemble(member={"train": {"lr_decay": 0.1}}),
        lambda: MCDropout(p=0.0),
        lambda: MCDropout(n_passes=0),
        lambda: IDMWhiteNoise(true_idm.to_dict(), sigma=-0.1),
    ):
        with pytest.raises(ValueError):
            bad()


# --------------------------------------------------------------------- decomposition
@pytest.mark.parametrize("key", ["ensemble", "mc_dropout"])
def test_decomposition_identity(fitted, test_x, key):
    model = fitted[key][0]
    dist = model.predict_dist(test_x)
    np.testing.assert_allclose(dist.aleatoric_var + dist.epistemic_var, dist.var(), rtol=1e-9)
    assert np.all(dist.epistemic_var >= 0.0) and dist.epistemic_var.mean() > 0.0
    # table of design.md: aleatoric = mean member variance, epistemic = variance of member means (K = 1)
    _, mus, sigmas = model.predict_members(test_x) if key == "ensemble" else model.predict_passes(test_x)
    np.testing.assert_allclose(dist.aleatoric_var, np.mean(sigmas[..., 0] ** 2, axis=0), rtol=1e-9)
    np.testing.assert_allclose(dist.epistemic_var, np.var(mus[..., 0], axis=0), rtol=1e-9, atol=1e-15)
    # ... and the total is the variance of the represented law: 200k draws, relative SE ~0.3 %, tolerance 3 %
    rows = np.linspace(0, len(test_x) - 1, 5).astype(int)
    draws = dist[rows].sample(200_000, np.random.default_rng(0))
    np.testing.assert_allclose(draws.var(axis=1), dist.var()[rows], rtol=0.03)


# --------------------------------------------------------------------------- sampler
def test_sampler_shapes_modes_and_generator(case, test_x):
    key, model, _ = case
    hist = torch.as_tensor(test_x[:12])
    for mode in SAMPLER_MODES:
        g = torch.Generator().manual_seed(3)
        rng_before = torch.get_rng_state()
        state = model.init_sampler(12, g, mode)
        draws = model.sample_step(hist, state, g)
        assert torch.equal(torch.get_rng_state(), rng_before)  # only the given generator is used
        assert draws.shape == (12,) and draws.dtype == torch.float32 and torch.all(torch.isfinite(draws))
        g2 = torch.Generator().manual_seed(3)
        assert torch.equal(model.sample_step(hist, model.init_sampler(12, g2, mode), g2), draws)
    with pytest.raises(ValueError):
        model.init_sampler(12, torch.Generator(), "per_rollout")


def test_sampler_per_step_matches_predict_dist(case, test_x):
    """Mean and variance of 20 000 draws for one history repeated B times, within 5 standard errors."""
    key, model, _ = case
    rows = [0, int(np.argmax(model.predict_dist(test_x).var()))]
    law_model = many_pass_copy(model) if key == "mc_dropout" else model  # fresh masks = many-pass limit
    law = law_model.predict_dist(test_x[rows])
    for i, row in enumerate(rows):
        mean, var, mu4 = moments(law, i)
        hist = torch.as_tensor(test_x[row : row + 1]).expand(N_DRAWS, -1, -1)
        g = torch.Generator().manual_seed(11 + i)
        draws = model.sample_step(hist, model.init_sampler(N_DRAWS, g, "per_step"), g).double().numpy()
        extra = law.epistemic_var[i] / 4000 if key == "mc_dropout" else 0.0  # MC error of the reference law
        se_mean = math.sqrt(var / N_DRAWS + extra)
        se_var = math.sqrt((mu4 - var**2) / N_DRAWS)
        assert abs(draws.mean() - mean) < Z * se_mean, (key, row, draws.mean(), mean, se_mean)
        assert abs(draws.var() - var) < Z * se_var + 1e-3 * var, (key, row, draws.var(), var, se_var)


def test_ensemble_per_trajectory_draws_follow_one_member(fitted, test_x):
    """Member 0 is shifted by +2 m/s^2 so that member means and the pooled mean are far apart."""
    base = fitted["ensemble"][0]
    model = DeepEnsemble(**base.init_kwargs)
    model.set_state(base.get_state())
    model.members[0].normalizer.y_mean += 2.0
    x = test_x[:1]
    _, member_mu, member_sd = model.predict_members(x)
    member_mu, member_sd = member_mu[:, 0, 0], member_sd[:, 0, 0]
    pooled = float(model.predict_dist(x).mean()[0])
    n_traj, n_steps = 30, 400
    g = torch.Generator().manual_seed(4)
    state = model.init_sampler(n_traj, g, "per_trajectory")
    assert state.shape == (n_traj,) and state.dtype == torch.int64 and set(state.tolist()) == {0, 1, 2}
    frozen = state.clone()
    hist = torch.as_tensor(x).expand(n_traj, -1, -1)
    # exact: every trajectory gets the parameters of its own member
    _, mu, sd = model.step_params(hist, state, g)
    np.testing.assert_allclose(mu[:, 0].numpy(), member_mu[state.numpy()], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sd[:, 0].numpy(), member_sd[state.numpy()], rtol=1e-5, atol=1e-6)
    draws = torch.stack([model.sample_step(hist, state, g) for _ in range(n_steps)]).double().numpy()
    assert torch.equal(state, frozen)
    traj_mean = draws.mean(axis=0)
    se = member_sd[state.numpy()] / math.sqrt(n_steps)  # ~0.04 m/s^2
    assert np.all(np.abs(traj_mean - member_mu[state.numpy()]) < Z * se)
    assert np.all(np.abs(traj_mean - pooled) > Z * se)  # member means are >= 0.6 m/s^2 from the pooled mean
    # per step: every trajectory mixes the members, so all time averages sit at the pooled mean
    per_step = torch.stack([model.sample_step(hist, None, g) for _ in range(n_steps)]).double().numpy()
    se_pooled = math.sqrt(float(model.predict_dist(x).var()[0]) / n_steps)
    assert np.all(np.abs(per_step.mean(axis=0) - pooled) < Z * se_pooled)


def test_mc_dropout_per_trajectory_reuses_masks(fitted, test_x):
    model = fitted["mc_dropout"][0]
    n_traj, n_steps = 30, 400
    g = torch.Generator().manual_seed(6)
    state = model.init_sampler(n_traj, g, "per_trajectory")
    m1, m2 = state
    assert m1.shape == (n_traj, 64) and m2.shape == (n_traj, 64)
    scale = 1.0 / (1.0 - 0.1)
    assert torch.all((m1 == 0) | torch.isclose(m1, torch.tensor(scale)))
    frozen = (m1.clone(), m2.clone())
    hist = torch.as_tensor(test_x[:1]).expand(n_traj, -1, -1)
    first = model.step_params(hist, state, g)
    second = model.step_params(hist, state, g)
    for a, b in zip(first, second):  # the same masks at every step
        assert torch.equal(a, b)
    assert torch.unique(first[1][:, 0]).numel() > n_traj // 2  # trajectories have different masks
    fresh = [model.step_params(hist, None, g)[1] for _ in range(2)]
    assert not torch.equal(fresh[0], fresh[1])  # per step: new masks every step
    draws = torch.stack([model.sample_step(hist, state, g) for _ in range(n_steps)]).double().numpy()
    assert torch.equal(m1, frozen[0]) and torch.equal(m2, frozen[1])
    mu, sd = first[1][:, 0].double().numpy(), first[2][:, 0].double().numpy()
    assert np.all(np.abs(draws.mean(axis=0) - mu) < Z * sd / math.sqrt(n_steps))


# ------------------------------------------------------------------------- ensemble
def test_ensemble_members_differ_and_see_different_drivers(fitted, synth_small, test_x):
    model, log = fitted["ensemble"]
    train = synth_small.windows["train"]
    _, mus, _ = model.predict_members(test_x)
    assert mus.shape == (3, len(test_x), 1)
    for a in range(3):
        for b in range(a + 1, 3):
            assert np.max(np.abs(mus[a] - mus[b])) > 1e-3
    assert log["member_seeds"] == [0, 1, 2]
    sets = []
    for m in range(3):
        idx = bootstrap_drivers(train.driver, np.random.default_rng(m))  # seed * 1000 + m with seed 0
        sets.append(frozenset(np.unique(train.driver[idx]).tolist()))
        assert log["n_drivers"][m] == len(sets[m]) < np.unique(train.driver).size
        assert log["members"][m]["n_train"] == len(idx)
    assert len(set(sets)) == 3
    y_means = [float(member.normalizer.y_mean) for member in model.members]  # normaliser of its own resample
    assert len(set(y_means)) == 3


def test_ensemble_without_bootstrap_members_differ_only_by_seed(synth_small, test_x):
    w = synth_small.windows
    member = {"hidden": 16, "head_hidden": 16}
    model = DeepEnsemble(n_members=2, bootstrap=False, member=member, train=FAST)
    log = model.fit(w["train"], w["val"], seed=3)
    assert log["member_seeds"] == [3000, 3001]
    assert log["n_drivers"] == [np.unique(w["train"].driver).size] * 2
    assert [m["n_train"] for m in log["members"]] == [len(w["train"])] * 2
    reference = HeteroMDN(**member, train=FAST)
    reference.fit(w["train"], w["val"], seed=3001)
    np.testing.assert_array_equal(model.members[1].predict_dist(test_x).mean(), reference.predict_dist(test_x).mean())
    _, mus, _ = model.predict_members(test_x)
    assert np.max(np.abs(mus[0] - mus[1])) > 1e-3


def test_member_train_overrides_take_precedence():
    model = DeepEnsemble(n_members=2, member={"hidden": 8, "train": {"lr": 5e-3}}, train={"max_epochs": 2, "lr": 1e-3})
    cfg = model.members[0].train_cfg
    assert (cfg.max_epochs, cfg.lr) == (2, 5e-3) and model.members[0].net_spec["hidden"] == 8


# ------------------------------------------------------------------------ baselines
def test_idm_white_noise_mean_and_sigma(fitted, synth_small, true_idm):
    model, log = fitted["idm_noise"]
    w = synth_small.windows
    test = w["test"]
    last = test.X[:, -1].astype(np.float64)
    expected = idm_accel(last[:, 0], last[:, 1], last[:, 2], true_idm)
    np.testing.assert_array_equal(model.predict_dist(test.X).mean(), expected)
    # only the last sample matters (uses_history = False)
    shuffled = test.X.copy()
    shuffled[:, :-1] = test.X[::-1, :-1]
    np.testing.assert_array_equal(model.predict_dist(shuffled).mean(), expected)
    train_last = w["train"].X[:, -1].astype(np.float64)
    residual = w["train"].y - idm_accel(train_last[:, 0], train_last[:, 1], train_last[:, 2], true_idm)
    assert model.sigma == pytest.approx(float(np.sqrt(np.mean(residual**2))), rel=1e-12)
    assert log["sigma_fitted"] is True and log["sigma"] == model.sigma
    assert abs(model.sigma / 0.4 - 1.0) < 0.05  # generator noise 0.4; measured 0.391 (SE ~0.005)
    fixed = IDMWhiteNoise(true_idm.to_dict(), sigma=0.3)
    assert fixed.fit(w["train"], w["val"])["sigma_fitted"] is False and fixed.sigma == 0.3
    # the torch sampler draws around the same mean
    g = torch.Generator().manual_seed(0)
    hist = torch.as_tensor(test.X[:1]).expand(N_DRAWS, -1, -1)
    draws = model.sample_step(hist, None, g).double().numpy()
    assert abs(draws.mean() - expected[0]) < Z * model.sigma / math.sqrt(N_DRAWS)


def test_lstm_const_sigma_is_the_rms_residual(fitted, synth_small):
    model, log = fitted["lstm_const"]
    w = synth_small.windows
    val_rms = float(np.sqrt(np.mean((w["val"].y - model.predict_mean(w["val"].X)) ** 2)))
    assert log["sigma_from"] == "val" and model.sigma == pytest.approx(val_rms, rel=1e-12)
    np.testing.assert_array_equal(model.predict_dist(w["val"].X).mean(), model.predict_mean(w["val"].X))
    no_val = LSTMConstSigma(hidden=16, head_hidden=16, train={"max_epochs": 1})
    log2 = no_val.fit(w["train"], None, seed=0)
    train_rms = float(np.sqrt(np.mean((w["train"].y - no_val.predict_mean(w["train"].X)) ** 2)))
    assert log2["sigma_from"] == "train" and log2["val_nll"] is None and no_val.sigma == pytest.approx(train_rms)


# -------------------------------------------------------------------------- configs
CONFIG_FILES = {
    "hetero_mdn": ("hetero_mdn", HeteroMDN),
    "hetero_mdn_mix3": ("hetero_mdn", HeteroMDN),
    "deep_ensemble": ("deep_ensemble", DeepEnsemble),
    "mc_dropout": ("mc_dropout", MCDropout),
    "lstm_const_sigma": ("lstm_const_sigma", LSTMConstSigma),
    "idm_white_noise": ("idm_white_noise", IDMWhiteNoise),
}


@pytest.mark.parametrize("stem", list(CONFIG_FILES))
def test_model_config_files_build(stem, true_idm):
    """What the training script does: build_model(name, **kwargs, train=default merged with overrides)."""
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / "model" / f"{stem}.yaml"), resolve=True)
    assert set(cfg) == {"name", "kwargs", "train", "needs_idm"}
    name, cls = CONFIG_FILES[stem]
    assert cfg["name"] == name and cfg["needs_idm"] == (name == "idm_white_noise")
    default = OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / "train" / "default.yaml"))
    assert set(default) == set(TrainConfig.__dataclass_fields__)
    kwargs = dict(cfg["kwargs"])
    if cfg["needs_idm"]:
        kwargs["idm"] = true_idm.to_dict()
    model = build_model(cfg["name"], **kwargs, train={**default, **cfg["train"]})
    assert type(model) is cls
    if stem == "hetero_mdn_mix3":
        assert model.n_components == 3
