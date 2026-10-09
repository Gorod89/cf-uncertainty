"""Shared fixtures. Data sets are generated once per session and must be treated as read-only."""

from __future__ import annotations

import os
import warnings

import pytest
import torch

from cf_unc.data.pipeline import DataBundle, synthetic_bundle
from cf_unc.physics.idm import IDMParams

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
torch.set_num_threads(min(4, os.cpu_count() or 1))

TRUE_IDM = IDMParams(v0=30.0, T=1.5, s0=2.0, a_max=1.2, b=1.8)
NOISE_SIGMA = 0.4


def pytest_configure(config: pytest.Config) -> None:
    os.environ.setdefault("PYTHONUTF8", "1")


@pytest.fixture(scope="session")
def true_idm() -> IDMParams:
    return TRUE_IDM


@pytest.fixture(scope="session")
def synth_small() -> DataBundle:
    """Tiny data for smoke tests: 30 drivers x 2 events x 20 s, stride 2 (about 3.5k training windows)."""
    cfg = dict(n_drivers=30, events_per_driver=2, duration_s=20.0, noise_sigma=NOISE_SIGMA, seed=11)
    return synthetic_bundle(cfg, name="synth_small", stride=2)


@pytest.fixture(scope="session")
def synth_small_seq() -> DataBundle:
    """Same events as ``synth_small`` cut for sequence models (seq_len 50, stride 5)."""
    cfg = dict(n_drivers=30, events_per_driver=2, duration_s=20.0, noise_sigma=NOISE_SIGMA, seed=11)
    return synthetic_bundle(cfg, name="synth_small_seq", stride=5, seq_len=50)


@pytest.fixture(scope="session")
def synth_e4() -> DataBundle:
    """E4 sanity data: identical drivers, constant noise sigma = 0.4; 170 drivers x 4 events x 40 s, stride 5."""
    cfg = dict(n_drivers=170, events_per_driver=4, duration_s=40.0, noise_sigma=NOISE_SIGMA, seed=21)
    return synthetic_bundle(cfg, name="synth_e4", stride=5)


@pytest.fixture(scope="session")
def synth_e4_gap() -> DataBundle:
    """Heteroscedastic variant: sigma(s) = 0.4 (1 + 1.5 exp(-s / 15))."""
    cfg = dict(
        n_drivers=170,
        events_per_driver=4,
        duration_s=40.0,
        noise_kind="gap",
        noise_sigma=NOISE_SIGMA,
        noise_kappa=1.5,
        noise_s_scale=15.0,
        seed=22,
    )
    return synthetic_bundle(cfg, name="synth_e4_gap", stride=5)


@pytest.fixture(scope="session")
def synth_e4_seq() -> DataBundle:
    """E4 events cut for sequence models (seq_len 50, stride 5)."""
    cfg = dict(n_drivers=170, events_per_driver=4, duration_s=40.0, noise_sigma=NOISE_SIGMA, seed=21)
    return synthetic_bundle(cfg, name="synth_e4_seq", stride=5, seq_len=50)
