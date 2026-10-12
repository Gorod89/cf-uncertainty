"""Plan sharing between followers (extension S8): the recorded plans of the engine, the plan window of the diffusion
sampler, ``simulate_platoon(share_plans=True)`` and the E3 variant key."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from cf_unc.constants import DT, HIST_LEN
from cf_unc.data.windows import WindowSet
from cf_unc.experiments.e3 import parse_variants
from cf_unc.models.diffusion_cf import DiffusionCF, DiffusionCFHist, planned_speeds
from cf_unc.platoon.simulate import PlatoonConfig, simulate_platoon
from cf_unc.preview import PreviewSpec

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


def _fit(cls=DiffusionCF, replan: int = SEQ) -> DiffusionCF:
    model = cls(seq_len=SEQ, n_samples=4, n_sampling_steps=4, t_train=20, channels=8, n_blocks=1, hidden=8,
                replan_every=replan, train=TRAIN)  # fmt: skip
    model.fit(_windows(48, 4), None, seed=0)
    return model


@pytest.fixture(scope="module")
def model() -> DiffusionCF:
    return _fit()


def _profiles(n_e: int = 2, n: int = 40) -> list[np.ndarray]:
    t = np.arange(n) * DT
    return [20.0 + 2.0 * np.sin(2 * np.pi * t / 3.0 + e) for e in range(n_e)]


CFG = PlatoonConfig(n_followers=3, n_rollouts=3, burn_in_s=0.5, tail_s=0.5, max_batch=1000)


def test_planned_speeds_integrate_like_the_engine():
    v0 = torch.tensor([10.0, 0.05])
    plan = torch.tensor([[1.0, -1.0, 0.5], [-1.0, -1.0, 2.0]])
    v = planned_speeds(v0, plan)
    assert torch.allclose(v[0], torch.tensor([10.1, 10.0, 10.05]))
    assert torch.allclose(v[1], torch.tensor([0.0, 0.0, 0.2]))  # standstill floor


def test_plan_window_reads_the_right_plan(model: DiffusionCF):
    gen = torch.Generator().manual_seed(0)
    plans = torch.arange(3 * SEQ, dtype=torch.float32).reshape(1, 3, SEQ) + 100.0  # plan j holds 100 + 10 j + c
    every = 5
    w = model._plan_window(plans, 0, None, gen, every)
    assert torch.equal(w[0], plans[0, 0])
    w = model._plan_window(plans, 7, None, gen, every)  # plan 1 (issued at 5), offset 2, held beyond its end
    expect = torch.cat([plans[0, 1, 2:], plans[0, 1, -1:].expand(2)])
    assert torch.equal(w[0], expect)
    w = model._plan_window(plans, 7, PreviewSpec(delay_s=0.5), gen, every)  # delay 5 steps: plan 0, offset 7
    expect = torch.cat([plans[0, 0, 7:], plans[0, 0, -1:].expand(7)])
    assert torch.equal(w[0], expect)
    w = model._plan_window(plans, 7, PreviewSpec(horizon_s=0.3), gen, every)  # 3 columns, then held
    expect = torch.cat([plans[0, 1, 2:5], plans[0, 1, 4:5].expand(7)])
    assert torch.equal(w[0], expect)
    w = model._plan_window(plans, 40, None, gen, every)  # beyond the last plan: its last value
    assert torch.all(w[0] == plans[0, 2, -1])


def test_init_sampler_validates_the_plans(model: DiffusionCF):
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="v_lead_plan"):
        model.init_sampler(2, gen, "per_step", v_lead_plan=torch.zeros(2, 3, SEQ + 1), plan_every=5)
    with pytest.raises(ValueError, match="plan_every"):
        model.init_sampler(2, gen, "per_step", v_lead_plan=torch.zeros(2, 3, SEQ), plan_every=0)
    state = model.init_sampler(2, gen, "per_step", v_lead_plan=torch.zeros(2, 3, SEQ), plan_every=5, record_plans=True)
    assert state["v_lead_plan"].shape == (2, 3, SEQ) and state["plan_every"] == 5 and state["plans"] == []
    hist = _fit(DiffusionCFHist)
    state = hist.init_sampler(2, gen, "per_step", v_lead_plan=torch.zeros(2, 3, SEQ), plan_every=5)
    assert state["v_lead_plan"] is None


def test_engine_records_the_plans_and_shares_them(model: DiffusionCF):
    profiles = _profiles()
    base = simulate_platoon(model, profiles, [15.0, 15.0], [1, 2, 3], CFG, keep_paths=True)
    shared = simulate_platoon(model, profiles, [15.0, 15.0], [1, 2, 3], CFG, keep_paths=True, share_plans=True)
    # vehicle 1 sees the recorded leader in both runs: identical
    assert np.array_equal(base.paths["v"][:, 0], shared.paths["v"][:, 0])
    # the plans are recorded with the right shape and start at the realised speeds (no standstill here)
    n_k = base.n_steps
    assert shared.info["share_plans"] is True
    # vehicle 2 sees vehicle 1's plans: with re-planning every SEQ steps a plan is the realised future up to float32
    # rounding, so the two runs stay close; with a shorter re-planning interval they differ
    diff = np.abs(base.paths["v"][:, 1] - shared.paths["v"][:, 1]).max()
    assert np.isfinite(diff)
    short = _fit(replan=SEQ // 2)
    a = simulate_platoon(short, profiles, [15.0, 15.0], [1, 2, 3], CFG, keep_paths=True)
    b = simulate_platoon(short, profiles, [15.0, 15.0], [1, 2, 3], CFG, keep_paths=True, share_plans=True)
    assert np.array_equal(a.paths["v"][:, 0], b.paths["v"][:, 0])
    assert not np.array_equal(a.paths["v"][:, 1:], b.paths["v"][:, 1:])
    assert n_k >= SEQ


def test_recorded_plans_match_the_realised_path():
    from cf_unc.rollout.engine import RolloutConfig, rollout_events
    from cf_unc.platoon.simulate import steady_event

    model = _fit(replan=SEQ // 2)
    ev = [steady_event("e", 20.0, 15.0, 100.0, HIST_LEN + 2 * SEQ)]
    cfg = RolloutConfig(n_rollouts=2, horizon_s=2 * SEQ * DT, horizons_s=(2 * SEQ * DT,), stream_events=1, seed=3)
    res = rollout_events(model, ev, cfg=cfg, keep_paths=True, record_plans=True)
    assert res.plans is not None and res.plans.shape == (1, 2, 4, SEQ) and res.plan_every == SEQ // 2
    # plan j is followed for SEQ // 2 steps from re-plan step j P: the realised speeds equal its first columns
    for j in range(4):
        k0 = j * (SEQ // 2)
        realised = res.v_path[0, :, k0 + 1 : k0 + 1 + SEQ // 2]
        assert np.allclose(res.plans[0, :, j, : SEQ // 2], realised, atol=1e-4)
    hist = _fit(DiffusionCFHist, replan=SEQ // 2)
    res = rollout_events(hist, ev, cfg=cfg, keep_paths=True, record_plans=True)
    assert res.plans is not None  # the history-only law also records its plans (it ignores shared ones)


def test_e3_variant_share_plans_key():
    cfg = OmegaConf.create({
        "checkpoints": {"a": "ckpt"},
        "variants": {"pool_r10_share": {"a": "ckpt", "share_plans": True, "models": {"diffusion_cf": {"sampler": {"replan_every": 10}}}}},
        "models": {"diffusion_cf": {"target": "a"}, "idm_white_noise": {"target": "a"}},
    })  # fmt: skip
    variants = parse_variants(cfg)
    assert variants["pool_r10_share"].share_plans is True and variants["raw"].share_plans is False
    cfg.variants.pool_r10_share.share_plans = "yes"
    with pytest.raises(ValueError, match="share_plans"):
        parse_variants(cfg)
