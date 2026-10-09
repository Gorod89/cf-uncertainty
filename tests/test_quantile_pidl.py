"""QuantilePIDL (docs/design.md, section 7): loss terms, collocation windows, sampler, persistence, E4 sanity."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from cf_unc.constants import DT, HIST_LEN
from cf_unc.dist import QuantileDist
from cf_unc.eval import metrics
from cf_unc.models import CFModel, build_model
from cf_unc.models.quantile_pidl import (
    DEFAULT_COLLOC_BOX,
    DEFAULT_LEVELS,
    SYNTH_MIN_GAP,
    QuantilePIDL,
    noncrossing_penalty,
    pinball_loss,
    quantile_icdf,
    synthetic_windows,
)
from cf_unc.physics.idm import IDMParams, calibrate_idm, idm_accel

N_DRAWS = 200_000
NOISE_SD = 0.4  # E4 data


# ----------------------------------------------------------------------------------------- loss terms
def test_pinball_loss_hand_computed():
    y = torch.tensor([1.0, -0.5])
    q = torch.tensor([[0.5, 1.0, 2.0], [0.0, -1.0, -0.5]])
    levels = torch.tensor([0.1, 0.5, 0.9])
    # residuals u = y - q: row 1 (0.5, 0, -1) -> (0.05, 0, 0.1); row 2 (-0.5, 0.5, 0) -> (0.45, 0.25, 0)
    assert float(pinball_loss(y, q, levels)) == pytest.approx(0.85 / 6.0, rel=1e-6)
    ref = metrics.pinball_loss(y.numpy(), q.numpy(), levels.numpy()).mean()
    assert float(pinball_loss(y, q, levels)) == pytest.approx(ref, rel=1e-6)


def test_noncrossing_penalty():
    ordered = torch.tensor([[-1.0, 0.0, 0.5, 2.0], [0.0, 0.1, 0.2, 0.3]])
    assert float(noncrossing_penalty(ordered)) == 0.0
    crossing = torch.tensor([[0.0, 1.0, 0.5, 2.0], [0.0, 0.1, 0.2, 0.3]])
    assert float(noncrossing_penalty(crossing)) == pytest.approx(0.5**2 / 2.0)  # mean over rows
    assert float(noncrossing_penalty(ordered, margin=0.15)) == pytest.approx(3 * 0.05**2 / 2.0, rel=1e-4)  # gaps 0.1


def test_synthetic_windows():
    box = DEFAULT_COLLOC_BOX
    x = synthetic_windows(5000, torch.Generator().manual_seed(3))
    assert x.shape == (5000, HIST_LEN, 3) and x.dtype == torch.float32
    last = x[:, -1, :].double()
    for j, key in enumerate(("s", "dv", "v")):
        assert torch.all(last[:, j] >= box[key][0]) and torch.all(last[:, j] <= box[key][1])
        assert float(last[:, j].min()) < box[key][0] + 0.05 * (box[key][1] - box[key][0])  # fills the box
    assert torch.all(x[:, :, 1] == x[:, -1:, 1]) and torch.all(x[:, :, 2] == x[:, -1:, 2])  # constant dv, v
    s, dv = x[:, :, 0].double(), x[:, :, 1].double()
    unclipped = (s[:, :-1] > SYNTH_MIN_GAP) & (s[:, 1:] > SYNTH_MIN_GAP)
    step = s[:, 1:] - s[:, :-1]
    torch.testing.assert_close(step[unclipped], (-DT * dv[:, 1:])[unclipped], rtol=0.0, atol=1e-4)
    assert float(s.min()) == pytest.approx(SYNTH_MIN_GAP) and bool((~unclipped).any())  # the clip is active
    again = synthetic_windows(5000, torch.Generator().manual_seed(3))
    torch.testing.assert_close(again, x, rtol=0.0, atol=0.0)
    other = synthetic_windows(5000, torch.Generator().manual_seed(4))
    assert not torch.equal(other, x)
    narrow = synthetic_windows(100, torch.Generator().manual_seed(0), {"v": (10.0, 11.0)}, hist_len=4)
    assert narrow.shape == (100, 4, 3) and float(narrow[:, :, 2].min()) >= 10.0
    with pytest.raises(ValueError):
        synthetic_windows(10, torch.Generator(), {"s": (0.0, 10.0)})


def test_quantile_icdf_is_the_quantile_dist_law():
    """Torch inverse cdf == QuantileDist.quantile (crossing knots, tied knots, both tails)."""
    rng = np.random.default_rng(0)
    levels = np.array(DEFAULT_LEVELS)
    values = np.sort(rng.normal(size=(400, 5)), axis=1) * rng.uniform(0.1, 2.0, size=(400, 1))
    values[:50] = values[:50, ::-1]  # crossing rows
    values[50:60, 2] = values[50:60, 1]  # tied knots (pushed apart by the law)
    u = rng.uniform(size=400)
    u[:20] = rng.uniform(0.0, 1e-3, size=20)
    u[20:40] = 1.0 - rng.uniform(0.0, 1e-3, size=20)
    ours = quantile_icdf(torch.as_tensor(levels), torch.as_tensor(values), torch.as_tensor(u)).numpy()
    law = QuantileDist(levels, values)
    np.testing.assert_allclose(ours, law.quantile(u[:, None])[:, 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(law.cdf(ours), u, rtol=1e-9, atol=1e-12)


def test_config_mirrors_constructor_defaults(true_idm):
    root = Path(__file__).resolve().parents[1] / "configs"
    cfg = OmegaConf.to_container(OmegaConf.create((root / "model" / "quantile_pidl.yaml").read_text(encoding="utf-8")))
    assert cfg["name"] == "quantile_pidl" and cfg["needs_idm"] is True and cfg["train"] == {}
    params = inspect.signature(QuantilePIDL.__init__).parameters
    defaults = {k: p.default for k, p in params.items() if k not in ("self", "idm", "train")}
    defaults["levels"] = list(defaults["levels"])
    defaults["colloc_box"] = {k: list(v) for k, v in DEFAULT_COLLOC_BOX.items()}  # None -> the default box
    assert cfg["kwargs"] == defaults
    train = OmegaConf.to_container(OmegaConf.create((root / "train" / "default.yaml").read_text(encoding="utf-8")))
    model = build_model(cfg["name"], idm=true_idm.to_dict(), train=train, **cfg["kwargs"])
    assert model.train_cfg.max_epochs == train["max_epochs"]
    assert model.init_kwargs["colloc_box"] == QuantilePIDL(idm=true_idm.to_dict()).init_kwargs["colloc_box"]


# ----------------------------------------------------------------------------------------- fitted model
@pytest.fixture(scope="module")
def tiny(synth_small, true_idm) -> QuantilePIDL:
    model = QuantilePIDL(idm=true_idm.to_dict(), train={"max_epochs": 2, "batch_size": 512})
    model.fit_log = model.fit(synth_small.windows["train"], synth_small.windows["val"], seed=0)
    return model


def test_fit_log_and_contract(tiny, synth_small, true_idm):
    log = tiny.fit_log
    json.dumps(log)
    for key in ("val_nll", "n_train", "seconds", "epochs", "terms", "val_crossing_share", "val_pinball"):
        assert key in log
    assert set(log["terms"]) == {"pinball", "physics", "noncrossing", "total"}
    assert log["epochs"] == 2 and log["n_train"] == len(synth_small.windows["train"])
    assert 0.0 <= log["val_crossing_share"] <= 1.0 and math.isfinite(log["val_nll"])
    assert isinstance(build_model("quantile_pidl", idm=true_idm.to_dict()), QuantilePIDL)
    assert tiny.median_index == 2 and not QuantilePIDL.has_epistemic
    test = synth_small.windows["test"]
    dist = tiny.predict_dist(test.X)
    assert isinstance(dist, QuantileDist) and len(dist) == len(test)
    np.testing.assert_array_equal(dist.levels, DEFAULT_LEVELS)
    assert dist.values.dtype == np.float64 and np.all(np.isfinite(dist.values))
    assert dist.epistemic_var is None and np.all(dist.aleatoric_var > 0)
    # float32 LSTM: results depend on the batch composition at the 1e-8 level
    single = tiny.predict_dist(test.X[3])
    np.testing.assert_allclose(single.values, dist.values[3:4], rtol=0.0, atol=1e-5)
    chunked = tiny.predict_dist(torch.as_tensor(test.X), batch_size=37)
    np.testing.assert_allclose(chunked.values, dist.values, rtol=0.0, atol=1e-5)
    with pytest.raises(ValueError):
        QuantilePIDL(idm=true_idm.to_dict(), train={"max_epoch": 3})
    with pytest.raises(ValueError):
        QuantilePIDL(idm=true_idm.to_dict(), levels=(0.5, 0.25, 0.75))


def test_fit_is_deterministic(synth_small, true_idm):
    train = synth_small.windows["train"].subset(np.arange(1500))
    kw = dict(idm=true_idm.to_dict(), hidden=16, head_hidden=16, train={"max_epochs": 1, "batch_size": 256})
    logs, values = [], []
    for _ in range(2):
        model = QuantilePIDL(**kw)
        logs.append(model.fit(train, None, seed=3))
        values.append(model.quantile_values(synth_small.windows["test"].X[:50]))
    np.testing.assert_array_equal(values[0], values[1])
    assert logs[0]["terms"] == logs[1]["terms"] and logs[0]["val_nll"] is None


def test_save_load_exact(tiny, synth_small, tmp_path):
    x = synth_small.windows["test"].X
    loaded = CFModel.load(tiny.save(tmp_path / "qpidl.pt"))
    assert isinstance(loaded, QuantilePIDL)
    np.testing.assert_array_equal(loaded.predict_dist(x).values, tiny.predict_dist(x).values)
    hist = torch.as_tensor(x[:300])
    draws = [m.sample_step(hist, None, torch.Generator().manual_seed(9)) for m in (tiny, loaded)]
    torch.testing.assert_close(draws[0], draws[1], rtol=0.0, atol=0.0)


def test_sampler_matches_the_law(tiny, synth_small):
    """200 000 draws for one history: ECDF at 9 probe points within 0.01 of ``dist.cdf`` (SE <= 0.0011);
    mean within 2 % of the law's sd (SE = 0.22 %), variance within 2 % (SE ~0.4 % with the exponential tails)."""
    x0 = synth_small.windows["test"].X[7:8]
    law = tiny.predict_dist(x0)
    hist = torch.as_tensor(np.repeat(x0, N_DRAWS, axis=0))
    gen = torch.Generator().manual_seed(1)
    state = tiny.init_sampler(N_DRAWS, gen, "per_step")
    assert state is None and tiny.init_sampler(N_DRAWS, gen, "per_trajectory") is None
    draws = tiny.sample_step(hist, state, gen)
    assert draws.shape == (N_DRAWS,) and draws.dtype == torch.float32
    a = draws.double().numpy()
    probes = law.quantile([0.01, 0.05, 0.15, 0.25, 0.5, 0.75, 0.85, 0.95, 0.99])[0]
    ecdf = np.searchsorted(np.sort(a), probes, side="right") / N_DRAWS
    np.testing.assert_allclose(ecdf, law.cdf(probes[None, :])[0], atol=0.01)
    sd = float(law.std()[0])
    assert abs(a.mean() - float(law.mean()[0])) < 0.02 * sd
    assert abs(a.var() / float(law.var()[0]) - 1.0) < 0.02
    # both modes are the same law and use the generator identically
    same = [tiny.sample_step(hist[:1000], None, torch.Generator().manual_seed(4)) for _ in range(2)]
    torch.testing.assert_close(same[0], same[1], rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="sampler mode"):
        tiny.init_sampler(3, gen, "per_member")


# ----------------------------------------------------------------------------------------- slow: E4
# Budget: the defaults (lr 1e-3, n_colloc = batch) early-stop after ~31 epochs (~130 s) with PICP 0.906, width
# +3.4 %, law variance +7.6 %; lr 3e-3 with 256 synthetic windows per batch gets there in ~17 epochs (~60 s).
E4_KWARGS = {"n_colloc": 256, "train": {"max_epochs": 20, "lr": 3e-3}}
WRONG_T = 1.0  # true T = 1.5


@pytest.fixture(scope="module")
def e4_model(synth_e4) -> QuantilePIDL:
    idm, _ = calibrate_idm(synth_e4.events["train"], objective="accel")
    model = QuantilePIDL(idm=idm.to_dict(), **E4_KWARGS)
    model.fit_log = model.fit(synth_e4.windows["train"], synth_e4.windows["val"], seed=0)
    return model


@pytest.fixture(scope="module")
def e4_dist(e4_model, synth_e4) -> QuantileDist:
    return e4_model.predict_dist(synth_e4.windows["test"].X)


@pytest.mark.slow
def test_e4_no_crossings(e4_model, e4_dist):
    assert np.mean(e4_dist.n_crossings > 0) < 0.01  # measured 0.0
    assert e4_model.fit_log["val_crossing_share"] < 0.01  # measured 0.0


@pytest.mark.slow
def test_e4_calibration(e4_model, e4_dist, synth_e4):
    y = synth_e4.windows["test"].y
    lo, hi = e4_dist.interval(0.9)
    assert abs(metrics.picp(lo, hi, y) - 0.90) < 0.03  # measured 0.905
    q = e4_dist.quantile(e4_model.levels)
    shares = np.mean(y[:, None] <= q, axis=0)
    np.testing.assert_allclose(shares, e4_model.levels, atol=0.03)  # measured max deviation 0.015
    assert e4_model.fit_log["seconds"] < 300.0  # measured ~60 s (4 threads)


@pytest.mark.slow
def test_e4_median_width_and_aleatoric(e4_model, e4_dist, synth_e4):
    median = e4_dist.quantile([0.5])[:, 0]
    rmse = float(np.sqrt(np.mean((median - synth_e4.truth["test"]["a_det"]) ** 2)))
    assert rmse < 0.15  # measured 0.059
    lo, hi = e4_dist.interval(0.9)
    target = 2.0 * 1.645 * NOISE_SD
    assert abs(metrics.mean_width(lo, hi) / target - 1.0) < 0.10  # measured 1.350 vs 1.316 (+2.6 %)
    # variance of the interpolated law; exact N(0, 0.16) knots would give 0.1612 (interpolation +0.7 %)
    assert abs(float(e4_dist.var().mean()) / NOISE_SD**2 - 1.0) < 0.15  # measured 0.169 (+5.4 %)


def _small_subset(synth_e4, share: float = 0.05):
    train = synth_e4.windows["train"]
    drivers = np.unique(train.driver)
    chosen = np.random.default_rng(1).choice(drivers, size=round(share * drivers.size), replace=False)
    return train.subset(np.isin(train.driver, chosen))


def _median(model: QuantilePIDL, x: np.ndarray | torch.Tensor) -> np.ndarray:
    return model.predict_dist(x).quantile([0.5])[:, 0]


@pytest.mark.slow
def test_e4_physics_term_is_active(synth_e4, true_idm):
    """Wrong IDM (T = 1.0) and 5 % of the drivers: the physics term pulls the median towards that IDM.

    The three models minimise their own loss for 25 epochs (no validation set, so neither early stopping nor
    the restored weights depend on the pinball term alone).
    """
    wrong = IDMParams(**{**true_idm.to_dict(), "T": WRONG_T})
    small = _small_subset(synth_e4)  # 6 drivers, 1800 windows
    models = {}
    for w_phys in (0.0, 0.3, 10.0):
        model = QuantilePIDL(idm=wrong.to_dict(), w_phys=w_phys, train={"batch_size": 256, "max_epochs": 25})
        model.fit(small, None, seed=0)
        models[w_phys] = model

    test = synth_e4.windows["test"]
    last = test.X[:, -1, :].astype(np.float64)
    a_wrong = idm_accel(last[:, 0], last[:, 1], last[:, 2], wrong)  # RMSE vs a_det: 0.484
    rmse_test = {w: float(np.sqrt(np.mean((_median(m, test.X) - a_wrong) ** 2))) for w, m in models.items()}
    assert rmse_test[10.0] < rmse_test[0.0]  # measured 0.303 vs 0.513 (w_phys = 0.3: 0.470)

    # off-data collocation states: synthetic windows whose last state is > 1 training sd from every training state
    colloc = synthetic_windows(4000, torch.Generator().manual_seed(123))
    x_train = small.X[:, -1, :].astype(np.float64)
    std = x_train.std(axis=0)
    xc = colloc[:, -1, :].double().numpy()
    nearest, _ = cKDTree(x_train / std).query(xc / std)
    off = colloc[torch.as_tensor(nearest > 1.0)]
    assert len(off) > 1000
    a_idm_off = idm_accel(xc[nearest > 1.0, 0], xc[nearest > 1.0, 1], xc[nearest > 1.0, 2], wrong)
    rmse_off = {w: float(np.sqrt(np.mean((_median(models[w], off) - a_idm_off) ** 2))) for w in (0.0, 0.3)}
    assert rmse_off[0.3] < rmse_off[0.0]  # measured 0.30 vs 2.95 (1758 of 4000 states are off-data)
