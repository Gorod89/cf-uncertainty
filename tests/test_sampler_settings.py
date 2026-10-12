"""Inference-time sampler settings of DiffusionCF (extension S3): ``configure_sampler`` and the E3 model-spec key
``sampler``."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from cf_unc.constants import HIST_LEN
from cf_unc.data.windows import WindowSet
from cf_unc.experiments.e3 import check_models, parse_variants, sampler_settings
from cf_unc.models.diffusion_cf import DiffusionCF

SEQ = 10
TRAIN = dict(max_epochs=2, batch_size=16, lr=1e-3, ema_decay=0.0, patience=5, lr_patience=2, min_delta=1e-5)


def _windows(n: int, seed: int) -> WindowSet:
    rng = np.random.default_rng(seed)
    x = np.stack([rng.uniform(10, 40, (n, HIST_LEN)), rng.normal(0, 1.5, (n, HIST_LEN)), rng.uniform(5, 30, (n, HIST_LEN))], -1)
    y_seq = rng.normal(0, 0.5, (n, SEQ))
    fut = x[:, -1:, 2] - x[:, -1:, 1] + np.cumsum(rng.normal(0, 0.3, (n, SEQ)), axis=1)
    return WindowSet(
        X=x.astype(np.float32), y=y_seq[:, 0].astype(np.float32), driver=np.zeros(n, np.int64), driver_names=["d"],
        event_index=np.zeros(n, np.int64), t_index=np.arange(n), y_seq=y_seq.astype(np.float32),
        v_lead_future=fut.astype(np.float32),
    )  # fmt: skip


@pytest.fixture(scope="module")
def fitted() -> DiffusionCF:
    model = DiffusionCF(seq_len=SEQ, n_samples=4, n_sampling_steps=4, t_train=20, channels=8, n_blocks=1, hidden=8,
                        replan_every=SEQ, train=TRAIN)  # fmt: skip
    model.fit(_windows(48, 4), None, seed=0)
    return model


def _rollout(model: DiffusionCF, steps: int, seed: int = 0) -> np.ndarray:
    gen = torch.Generator().manual_seed(seed)
    hist = torch.as_tensor(_windows(3, 1).X[:, :, :])
    v_lead = torch.full((3, steps + SEQ + 1), 20.0)
    state = model.init_sampler(3, gen, "per_step", v_lead=v_lead)
    return np.stack([model.sample_step(hist, state, gen).cpu().numpy() for _ in range(steps)], axis=1)


def test_configure_sampler_validates(fitted: DiffusionCF):
    for bad in ({"replan_every": 0}, {"replan_every": SEQ + 1}, {"replan_every": True}, {"replan_every": 2.5}):
        with pytest.raises(ValueError, match="replan_every"):
            fitted.configure_sampler(**bad)
    for bad in ({"n_sampling_steps": 1}, {"n_sampling_steps": 21}, {"n_sampling_steps": True}):
        with pytest.raises(ValueError):
            fitted.configure_sampler(**bad)
    assert fitted.configure_sampler() == {"replan_every": SEQ, "n_sampling_steps": 4}


def test_configure_sampler_changes_the_rollout(fitted: DiffusionCF):
    fitted.configure_sampler(replan_every=SEQ, n_sampling_steps=4)
    base = _rollout(fitted, 2 * SEQ)
    assert np.array_equal(base, _rollout(fitted, 2 * SEQ))  # reproducible
    fitted.configure_sampler(replan_every=SEQ // 2)
    assert fitted.replan_every == SEQ // 2
    short = _rollout(fitted, 2 * SEQ)
    assert np.array_equal(short[:, : SEQ // 2], base[:, : SEQ // 2])  # the first plan is the same until the re-plan
    assert not np.array_equal(short, base)
    fitted.configure_sampler(replan_every=SEQ, n_sampling_steps=2)
    assert fitted.steps.size == 2 and not np.array_equal(_rollout(fitted, SEQ), base[:, :SEQ])
    fitted.configure_sampler(n_sampling_steps=4)
    assert np.array_equal(_rollout(fitted, 2 * SEQ), base)  # restored


def test_sampler_settings_and_e3_key():
    assert sampler_settings({"replan_every": 10, "n_sampling_steps": 25}) == {"replan_every": 10, "n_sampling_steps": 25}
    for bad in ({}, {"steps": 3}, {"replan_every": 0}, {"replan_every": True}, {"n_sampling_steps": "25"}, 5):
        with pytest.raises(ValueError):
            sampler_settings(bad)
    cfg = OmegaConf.create({
        "checkpoints": {"a": "ckpt"},
        "variants": {"pool_r10": {"a": "ckpt", "models": {"diffusion_cf": {"sampler": {"replan_every": 10}}}}},
        "models": {"diffusion_cf": {"target": "a"}, "idm_white_noise": {"target": "a"}},
    })  # fmt: skip
    assert "sampler" not in check_models(cfg)["diffusion_cf"]
    variants = parse_variants(cfg)
    assert variants["pool_r10"].models == {"diffusion_cf": {"sampler": {"replan_every": 10}}}
    cfg.models.diffusion_cf.sampler = {"n_sampling_steps": 25}
    assert check_models(cfg)["diffusion_cf"]["sampler"] == {"n_sampling_steps": 25}
