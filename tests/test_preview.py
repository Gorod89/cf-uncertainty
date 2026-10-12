"""Preview budget of the leader's future (extension X1): PreviewSpec, the DiffusionCF budget and rollout preview, the
engine hand-over and the E3 variant key. CPU, tiny models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from cf_unc.constants import DT, HIST_LEN
from cf_unc.data.windows import WindowSet
from cf_unc.experiments.e3 import parse_variants
from cf_unc.models.diffusion_cf import DiffusionCF, DiffusionCFHist, hold_beyond, hold_beyond_np
from cf_unc.preview import PreviewSpec
from cf_unc.rollout.engine import RolloutConfig

SEQ = 10
TRAIN = dict(max_epochs=2, batch_size=16, lr=1e-3, ema_decay=0.0, patience=5, lr_patience=2, min_delta=1e-5)


def _windows(n: int, seed: int) -> WindowSet:
    rng = np.random.default_rng(seed)
    x = np.stack([rng.uniform(10, 40, (n, HIST_LEN)), rng.normal(0, 1.5, (n, HIST_LEN)), rng.uniform(5, 30, (n, HIST_LEN))], -1)
    y = rng.normal(0, 0.5, n)
    y_seq = rng.normal(0, 0.5, (n, SEQ))
    fut = x[:, -1:, 2] - x[:, -1:, 1] + np.cumsum(rng.normal(0, 0.3, (n, SEQ)), axis=1)
    return WindowSet(
        X=x.astype(np.float32), y=y.astype(np.float32), driver=np.zeros(n, np.int64), driver_names=["d"],
        event_index=np.zeros(n, np.int64), t_index=np.arange(n), y_seq=y_seq.astype(np.float32),
        v_lead_future=fut.astype(np.float32),
    )  # fmt: skip


def _tiny(**kw) -> DiffusionCF:
    return DiffusionCF(seq_len=SEQ, n_samples=4, n_sampling_steps=4, t_train=20, channels=8, n_blocks=1, hidden=8,
                       replan_every=SEQ, train=TRAIN, **kw)  # fmt: skip


@pytest.fixture(scope="module")
def fitted() -> DiffusionCF:
    model = _tiny()
    model.fit(_windows(64, 0), _windows(16, 1), seed=0)
    return model


# ------------------------------------------------------------------------------------------ PreviewSpec
def test_spec_validation_and_steps():
    spec = PreviewSpec(horizon_s=1.0, noise_sigma=0.5, delay_s=0.5)
    assert spec.horizon_steps == round(1.0 / DT) and spec.delay_steps == round(0.5 / DT) and not spec.identity
    assert PreviewSpec().identity and PreviewSpec().horizon_steps is None
    assert PreviewSpec.from_any(None) is None and PreviewSpec.from_any(spec) is spec
    assert PreviewSpec.from_any({"horizon_s": 2.0}) == PreviewSpec(horizon_s=2.0)
    assert PreviewSpec(horizon_s=0.0).horizon_steps == 0
    for bad in ({"horizon": 1.0}, 3, {"noise_sigma": -1.0}, {"delay_s": -0.1}, {"horizon_s": 0.33}):
        with pytest.raises(ValueError):
            PreviewSpec.from_any(bad)
    assert json.loads(json.dumps(spec.to_dict())) == {"horizon_s": 1.0, "noise_sigma": 0.5, "delay_s": 0.5}


def test_rollout_config_converts_mapping():
    cfg = RolloutConfig(preview={"horizon_s": 1.0})
    assert isinstance(cfg.preview, PreviewSpec) and cfg.preview.horizon_s == 1.0
    assert RolloutConfig().preview is None


# --------------------------------------------------------------------------------------------- budget
def test_hold_beyond():
    v = torch.arange(12.0).reshape(2, 6)
    held = hold_beyond(v, 2)
    assert torch.equal(held[:, :2], v[:, :2]) and torch.equal(held[:, 2:], v[:, 1:2].expand(-1, 4))
    assert hold_beyond(v, None) is v and hold_beyond(v, 6) is v
    assert np.array_equal(hold_beyond_np(v.numpy(), 2), held.numpy())


def test_preview_steps_kwarg_round_trip(tmp_path: Path):
    model = _tiny(preview_steps=3)
    model.fit(_windows(48, 2), None, seed=0)
    path = model.save(tmp_path / "m.pt")
    loaded = DiffusionCF.load(path)
    assert loaded.preview_steps == 3 and loaded.init_kwargs["preview_steps"] == 3
    hist = _windows(5, 3)
    full = loaded.sample_sequences(hist.X, hist.v_lead_future)
    held = loaded.sample_sequences(hist.X, hold_beyond_np(hist.v_lead_future, 3))
    assert np.array_equal(full, held)  # the budget is applied inside the model
    with pytest.raises(ValueError, match="preview_steps"):
        _tiny(preview_steps=0)
    with pytest.raises(ValueError, match="preview_steps"):
        _tiny(preview_steps=SEQ + 1)
    with pytest.raises(ValueError, match="leader input"):
        DiffusionCFHist(seq_len=SEQ, n_samples=4, n_sampling_steps=4, t_train=20, channels=8, n_blocks=1, hidden=8,
                        replan_every=SEQ, train=TRAIN, preview_steps=2)  # fmt: skip


# -------------------------------------------------------------------------------------- rollout preview
def _roll(model: DiffusionCF, preview, seed: int = 0, n: int = 6, steps: int = 2 * SEQ) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    hist = torch.as_tensor(_windows(n, 7).X)
    v_lead = torch.as_tensor(20.0 + np.cumsum(rng.normal(0, 0.2, (n, steps + 1)), axis=1), dtype=torch.float32)
    gen = torch.Generator().manual_seed(123)
    state = model.init_sampler(n, gen, "per_step", v_lead=v_lead, preview=preview)
    return torch.stack([model.sample_step(hist, state, gen) for _ in range(steps)], dim=1)


def test_identity_spec_equals_no_spec(fitted: DiffusionCF):
    plain = _roll(fitted, None)
    assert torch.equal(plain, _roll(fitted, PreviewSpec()))
    assert torch.equal(plain, _roll(fitted, {"horizon_s": SEQ * DT, "noise_sigma": 0.0, "delay_s": 0.0}))


def test_horizon_zero_equals_constant_leader(fitted: DiffusionCF):
    n, steps = 6, SEQ
    rng = np.random.default_rng(1)
    hist = torch.as_tensor(_windows(n, 7).X)
    v_lead = torch.as_tensor(20.0 + np.cumsum(rng.normal(0, 0.2, (n, steps + 1)), axis=1), dtype=torch.float32)
    v_lead[:, 0] = hist[:, -1, 2] - hist[:, -1, 1]  # the leader speed at step 0 is the one of the history
    gen_a, gen_b = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    held = fitted.sample_step(hist, fitted.init_sampler(n, gen_a, "per_step", v_lead=v_lead, preview={"horizon_s": 0.0}), gen_a)
    const = fitted.sample_step(hist, fitted.init_sampler(n, gen_b, "per_step", v_lead=None), gen_b)
    assert torch.allclose(held, const, atol=1e-5)


def test_truncation_noise_and_delay_change_the_preview(fitted: DiffusionCF):
    plain = _roll(fitted, None)
    short = _roll(fitted, {"horizon_s": 2 * DT})
    noisy = _roll(fitted, {"noise_sigma": 1.0})
    stale = _roll(fitted, {"delay_s": 3 * DT})
    assert not torch.equal(plain, short) and not torch.equal(plain, noisy) and not torch.equal(plain, stale)
    assert torch.equal(noisy, _roll(fitted, {"noise_sigma": 1.0}))  # seeded by the rollout stream
    # a horizon as long as the plan is the identity; a delay beyond the window start clamps every column at step 0
    assert torch.equal(plain, _roll(fitted, {"horizon_s": SEQ * DT}))
    assert torch.equal(_roll(fitted, {"delay_s": 2 * SEQ * DT}, steps=SEQ), _roll(fitted, {"delay_s": 3 * SEQ * DT}, steps=SEQ))


def test_hist_model_ignores_preview():
    model = DiffusionCFHist(seq_len=SEQ, n_samples=4, n_sampling_steps=4, t_train=20, channels=8, n_blocks=1, hidden=8,
                            replan_every=SEQ, train=TRAIN)  # fmt: skip
    model.fit(_windows(48, 4), None, seed=0)
    state = model.init_sampler(3, torch.Generator().manual_seed(0), "per_step", v_lead=torch.ones(3, 5),
                               preview={"noise_sigma": 2.0})  # fmt: skip
    assert state["v_lead"] is None and state["preview"] is None


# --------------------------------------------------------------------------------------------- E3 key
def test_e3_variant_preview_key():
    cfg = OmegaConf.create({
        "checkpoints": {"a": "ckpt"},
        "variants": {"pool_h1": {"a": "ckpt", "preview": {"horizon_s": 1.0}, "only_models": ["diffusion_cf"]}},
        "models": {"diffusion_cf": {"target": "a"}, "idm_white_noise": {"target": "a"}},
    })  # fmt: skip
    variants = parse_variants(cfg)
    assert variants["pool_h1"].preview == {"horizon_s": 1.0, "noise_sigma": 0.0, "delay_s": 0.0}
    assert variants["raw"].preview is None
    cfg.variants.pool_h1.preview = {"horizon": 1.0}
    with pytest.raises(ValueError, match="preview"):
        parse_variants(cfg)
