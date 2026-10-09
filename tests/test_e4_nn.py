"""E4 sanity of the paper plan for the NN family (slow): noise recovery, epistemic variance versus data
size, heteroscedasticity (machinery for H3.3) and rare states (machinery for H3.1).

synth_e4: identical drivers, constant noise sigma = 0.4 (variance 0.16), ``truth[split]["a_det"]`` is the
noise-free mean. synth_e4_gap: sigma(s) = 0.4 (1 + 1.5 exp(-s / 15)). Every model is trained once per
module (seed 0) and reused. The values quoted next to the thresholds were measured with seed 0 on
4 CPU threads (whole file: about 6 minutes).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cf_unc.data.windows import WindowSet
from cf_unc.models.base import CFModel
from cf_unc.models.baselines import LSTMConstSigma
from cf_unc.models.deep_ensemble import DeepEnsemble
from cf_unc.models.hetero_mdn import HeteroMDN
from cf_unc.models.mc_dropout import MCDropout

pytestmark = pytest.mark.slow

NOISE_VAR = 0.16  # generator noise sigma = 0.4 m/s^2
# Full data: 35.7k training windows, ~1.5 s per epoch and network; the cap binds. Learning curve of
# HeteroMDN without early stopping: aleatoric variance 0.158-0.167 and RMSE vs a_det 0.066-0.056 at
# epochs 14-20, 0.162 / 0.043 at epoch 40.
FULL = {"max_epochs": 15}
# 10 % of the training drivers (12 drivers, 3.6k windows, 4 steps per epoch): early stopping (patience 6)
# ends the ensemble members after 53-71 epochs, before the cap.
SMALL = {"max_epochs": 80}
S_EDGES = np.arange(0.0, 120.0 + 1e-9, 5.0)  # m, 24 bins
DV_EDGES = np.arange(-5.0, 5.0 + 1e-9, 0.5)  # m/s, 20 bins
RARE_FREQ = 0.01  # a bin is rare when it holds < 1 % of the training windows

MC_SHRINK_REASON = (
    "scientific finding, not a bug: MC dropout epistemic variance barely shrinks with 10x more data; measured "
    "mean epistemic_var 5.64e-3 (10 % of the drivers, 3.6k windows) vs 4.96e-3 (all drivers, 35.7k windows), "
    "ratio 1.14 (deep ensemble: 3.15)"
)
RARE_REASON = (
    "not a bug: with the 15-epoch budget of this test the rare / other ratio of a 5-member ensemble is dominated "
    "by optimisation noise (measured 0.95 with 4 threads, 1.4 with 12 threads; driver-bootstrap 95 % interval "
    "0.86-1.04). Converged ensembles support H3.1 on synthetic data: 2.9 after 41-44 epochs on the same data and "
    "3.2 [2.7, 3.7] in the M1 benchmark (docs/m1_report.md). Note that 75 % of the test windows lie in bins "
    "with < 1 % of the training windows, so the plan's definition of a rare bin is very wide"
)


def train_model(model: CFModel, train: WindowSet, val: WindowSet) -> tuple[CFModel, dict]:
    return model, model.fit(train, val, seed=0)


def ten_percent_of_drivers(train: WindowSet, seed: int = 0) -> WindowSet:
    """Windows of a random 10 % of the training drivers."""
    drivers = np.unique(train.driver)
    chosen = np.random.default_rng(seed).choice(drivers, size=max(2, round(0.1 * drivers.size)), replace=False)
    return train.subset(np.flatnonzero(np.isin(train.driver, chosen)))


def state_bins(x: np.ndarray) -> np.ndarray:
    """(s, dv) bin of the last history sample on the S_EDGES x DV_EDGES grid, -1 outside the grid."""
    s, dv = x[:, -1, 0].astype(np.float64), x[:, -1, 1].astype(np.float64)
    i = np.searchsorted(S_EDGES, s, side="right") - 1
    j = np.searchsorted(DV_EDGES, dv, side="right") - 1
    inside = (i >= 0) & (i < S_EDGES.size - 1) & (j >= 0) & (j < DV_EDGES.size - 1)
    return np.where(inside, i * (DV_EDGES.size - 1) + j, -1)


def rare_state_mask(train_x: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Windows of ``x`` in bins holding < RARE_FREQ of the training windows, or outside the grid."""
    n_bins = (S_EDGES.size - 1) * (DV_EDGES.size - 1)
    train_bins = state_bins(train_x)
    freq = np.bincount(train_bins[train_bins >= 0], minlength=n_bins) / train_x.shape[0]
    bins = state_bins(x)
    return (bins < 0) | (freq[np.maximum(bins, 0)] < RARE_FREQ)


def summary(dist, windows: WindowSet, a_det: np.ndarray) -> dict[str, float]:
    return {
        "nll": float(-np.mean(dist.log_prob(windows.y))),
        "aleatoric": float(np.mean(dist.aleatoric_var)),
        "rmse": math.sqrt(float(np.mean((dist.mean() - a_det) ** 2))),
    }


# --------------------------------------------------------------------------- models (trained once)
@pytest.fixture(scope="module")
def hetero_e4(synth_e4):
    w = synth_e4.windows
    return train_model(HeteroMDN(train=FULL), w["train"], w["val"])


@pytest.fixture(scope="module")
def ensemble_e4(synth_e4):
    w = synth_e4.windows
    return train_model(DeepEnsemble(n_members=5, train=FULL), w["train"], w["val"])


@pytest.fixture(scope="module")
def mc_dropout_e4(synth_e4):
    w = synth_e4.windows
    return train_model(MCDropout(train=FULL), w["train"], w["val"])


@pytest.fixture(scope="module")
def ensemble_e4_small(synth_e4):
    w = synth_e4.windows
    return train_model(DeepEnsemble(n_members=5, train=SMALL), ten_percent_of_drivers(w["train"]), w["val"])


@pytest.fixture(scope="module")
def mc_dropout_e4_small(synth_e4):
    w = synth_e4.windows
    return train_model(MCDropout(train=SMALL), ten_percent_of_drivers(w["train"]), w["val"])


@pytest.fixture(scope="module")
def hetero_gap(synth_e4_gap):
    w = synth_e4_gap.windows
    return train_model(HeteroMDN(train=FULL), w["train"], w["val"])


@pytest.fixture(scope="module")
def const_sigma_gap(synth_e4_gap):
    w = synth_e4_gap.windows
    return train_model(LSTMConstSigma(train=FULL), w["train"], w["val"])


# --------------------------------------------------------------------------- 1. aleatoric
@pytest.mark.parametrize("fixture", ["hetero_e4", "ensemble_e4", "mc_dropout_e4"])
def test_aleatoric_variance_recovers_the_noise_level(fixture, request, synth_e4):
    model, log = request.getfixturevalue(fixture)
    test = synth_e4.windows["test"]
    stats = summary(model.predict_dist(test.X), test, synth_e4.truth["test"]["a_det"])
    print(f"\n{fixture}: {stats}, fit {log['seconds']:.0f} s")
    # measured: HeteroMDN 0.158 (-1 %), ensemble 0.164 (+3 %), MC dropout 0.179 (+12 %: training with
    # dropout inflates sigma; 0.175 after 30 epochs without early stopping)
    assert abs(stats["aleatoric"] / NOISE_VAR - 1.0) < 0.15
    assert stats["rmse"] < 0.15  # measured: HeteroMDN 0.066, ensemble 0.050, MC dropout 0.071


# --------------------------------------------------------------------------- 2. epistemic vs data
@pytest.mark.parametrize(
    "small_fixture, full_fixture",
    [
        ("ensemble_e4_small", "ensemble_e4"),
        pytest.param("mc_dropout_e4_small", "mc_dropout_e4", marks=pytest.mark.xfail(strict=False, reason=MC_SHRINK_REASON)),
    ],
)
def test_epistemic_variance_shrinks_with_data(small_fixture, full_fixture, request, synth_e4):
    test_x = synth_e4.windows["test"].X
    small_model, small_log = request.getfixturevalue(small_fixture)
    full_model, _ = request.getfixturevalue(full_fixture)
    small = float(np.mean(small_model.predict_dist(test_x).epistemic_var))
    full = float(np.mean(full_model.predict_dist(test_x).epistemic_var))
    print(f"\n{small_fixture}: epistemic {small:.3e} ({small_log['n_train']} windows); {full_fixture}: {full:.3e}; "
          f"ratio {small / full:.2f}")
    # measured: ensemble 2.90e-3 / 9.18e-4 = 3.15 (4.4 against a 40-epoch full-data ensemble); MC dropout 1.14
    assert small / full >= 1.5


# --------------------------------------------------------------------------- 3. heteroscedasticity
def test_heteroscedastic_variance_follows_the_gap(hetero_gap, const_sigma_gap, synth_e4_gap):
    test = synth_e4_gap.windows["test"]
    s = test.X[:, -1, 0]
    true_var = synth_e4_gap.truth["test"]["sigma_true"] ** 2
    dist = hetero_gap[0].predict_dist(test.X)
    for name, group in (("s < 10 m", s < 10.0), ("s > 40 m", s > 40.0)):
        predicted, true = float(dist.var()[group].mean()), float(true_var[group].mean())
        print(f"\n{name}: {group.sum()} windows, predicted variance {predicted:.4f}, true {true:.4f}")
        # measured: s < 10 m 0.546 vs 0.573 (-5 %), s > 40 m 0.175 vs 0.170 (+3 %)
        assert abs(predicted / true - 1.0) < 0.25
    nll_hetero = float(-np.mean(dist.log_prob(test.y)))
    nll_const = float(-np.mean(const_sigma_gap[0].predict_dist(test.X).log_prob(test.y)))
    print(f"test NLL HeteroMDN {nll_hetero:.4f} (fit {hetero_gap[1]['seconds']:.0f} s), "
          f"LSTMConstSigma {nll_const:.4f} (fit {const_sigma_gap[1]['seconds']:.0f} s)")
    assert nll_const - nll_hetero > 0.03  # measured: 0.8166 - 0.7566 = 0.060 nats


# --------------------------------------------------------------------------- 4. rare states
@pytest.fixture(scope="module")
def rare_states(ensemble_e4, synth_e4) -> dict:
    w = synth_e4.windows
    rare = rare_state_mask(w["train"].X, w["test"].X)
    epistemic = ensemble_e4[0].predict_dist(w["test"].X).epistemic_var
    ratio = float(epistemic[rare].mean() / epistemic[~rare].mean())
    print(f"\nrare windows {rare.sum()} / {rare.size}; epistemic rare {epistemic[rare].mean():.3e}, "
          f"other {epistemic[~rare].mean():.3e}, ratio {ratio:.2f}")
    return {"rare": rare, "epistemic": epistemic, "ratio": ratio}


def test_rare_state_bins():
    """Grid edges, out-of-grid states and the 1 % frequency threshold on hand-made last samples."""

    def windows(states: list[tuple[float, float]]) -> np.ndarray:
        x = np.zeros((len(states), 30, 3), dtype=np.float32)
        x[:, -1, :2] = states
        return x

    probe = windows([(2.5, 0.25), (119.9, 4.9), (50.0, -5.0), (120.0, 0.0), (50.0, 5.0), (-1.0, 0.0)])
    np.testing.assert_array_equal(state_bins(probe), [10, 23 * 20 + 19, 10 * 20, -1, -1, -1])
    # 200 training windows: 197 in bin A, 2 (= 1 %) in bin B, 1 (0.5 %) in bin C
    train = windows([(60.0, 0.1)] * 197 + [(2.5, 0.25)] * 2 + [(7.5, 0.25)])
    test = windows([(60.0, 0.1), (2.5, 0.25), (7.5, 0.25), (12.5, 0.25), (130.0, 0.0)])
    np.testing.assert_array_equal(rare_state_mask(train, test), [False, False, True, True, True])


def test_rare_state_groups(rare_states, synth_e4):
    rare, epistemic = rare_states["rare"], rare_states["epistemic"]
    outside = state_bins(synth_e4.windows["test"].X) < 0
    assert np.all(rare[outside])  # measured: 1037 of 7500 test windows outside the grid
    assert 0 < rare.sum() < rare.size  # measured: 5631 rare
    assert np.all(epistemic > 0.0) and 0.5 < rare_states["ratio"] < 2.0  # measured: 0.95


@pytest.mark.xfail(strict=False, reason=RARE_REASON)
def test_rare_states_have_larger_epistemic_variance(rare_states):
    assert rare_states["ratio"] > 1.2
