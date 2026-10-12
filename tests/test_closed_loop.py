"""Closed-loop fine-tuning of the one-step networks (extension S7, cf_unc.models.closed_loop)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cf_unc.constants import A_MAX, DT, HIST_LEN
from cf_unc.data.windows import WindowSet
from cf_unc.models import build_model, load_model
from cf_unc.models.closed_loop import rollout_loss

SEQ = 10
TWIN_TRAIN = dict(max_epochs=2, batch_size=32, lr=1e-3, patience=3, ema_decay=0.0, grad_clip=1.0)
CL_TRAIN = dict(max_epochs=2, batch_size=32, lr=1e-3, patience=2, ema_decay=0.0)


def _windows(n: int, seed: int, seq: bool = True) -> WindowSet:
    rng = np.random.default_rng(seed)
    x = np.stack([rng.uniform(10, 40, (n, HIST_LEN)), rng.normal(0, 1.5, (n, HIST_LEN)), rng.uniform(5, 30, (n, HIST_LEN))], -1)
    y_seq = rng.normal(0, 0.5, (n, SEQ))
    fut = x[:, -1:, 2] - x[:, -1:, 1] + np.cumsum(rng.normal(0, 0.3, (n, SEQ)), axis=1)
    return WindowSet(
        X=x.astype(np.float32), y=y_seq[:, 0].astype(np.float32), driver=np.zeros(n, np.int64), driver_names=["d"],
        event_index=np.zeros(n, np.int64), t_index=np.arange(n),
        y_seq=y_seq.astype(np.float32) if seq else None, v_lead_future=fut.astype(np.float32) if seq else None,
    )  # fmt: skip


@pytest.fixture(scope="module")
def twins(tmp_path_factory):
    """Stored one-step twins (seed 0) of both families in <dir>/<name>/seed0/model.pt."""
    root = tmp_path_factory.mktemp("twins")
    for name, kw in (("hetero_mdn", dict(hidden=8, head_hidden=8)), ("lstm_const_sigma", dict(hidden=8, head_hidden=8))):
        model = build_model(name, train=TWIN_TRAIN, **kw)
        model.fit(_windows(64, 1, seq=False), _windows(16, 2, seq=False), seed=0)
        model.save(root / name / "seed0" / "model.pt")
    return root


def test_rollout_loss_zero_for_the_recorded_accelerations():
    ws = _windows(5, 3)
    hist, vl, a_obs = (torch.as_tensor(v) for v in (ws.X, ws.v_lead_future, ws.y_seq))
    plans = [a_obs[:, h] for h in range(SEQ)]
    calls = iter(plans)
    loss = rollout_loss(lambda h: next(calls), hist, vl, a_obs, torch.tensor(5.0), torch.tensor(2.0), SEQ)
    assert float(loss) == 0.0
    loss = rollout_loss(lambda h: torch.full((5,), A_MAX), hist, vl, a_obs, torch.tensor(5.0), torch.tensor(2.0), SEQ)
    assert float(loss) > 0.0
    with pytest.raises(ValueError, match="horizon"):
        rollout_loss(lambda h: a_obs[:, 0], hist, vl, a_obs, torch.tensor(1.0), torch.tensor(1.0), SEQ + 1)


def test_rollout_loss_integrates_like_the_engine():
    """A constant acceleration rollout reproduces v + a dt and s + (v_lead - v) dt step by step."""
    ws = _windows(3, 4)
    hist = torch.as_tensor(ws.X)
    vl = torch.as_tensor(ws.v_lead_future)
    a_obs = torch.zeros(3, SEQ)
    seen = []
    loss = rollout_loss(lambda h: (seen.append(h[:, -1].clone()) or torch.full((3,), 0.5)), hist, vl, a_obs,
                        torch.tensor(1.0), torch.tensor(1.0), 3)  # fmt: skip
    v0, s0 = hist[:, -1, 2], hist[:, -1, 0]
    v1 = v0 + 0.5 * DT
    s1 = s0 + (vl[:, 0] - v1) * DT
    assert torch.allclose(seen[1][:, 2], v1) and torch.allclose(seen[1][:, 0], s1) and torch.allclose(seen[1][:, 1], v1 - vl[:, 0])
    assert float(loss) > 0.0


@pytest.mark.parametrize("name,twin", [("hetero_mdn_cl", "hetero_mdn"), ("lstm_const_sigma_cl", "lstm_const_sigma")])
def test_fine_tuning_round_trip(twins, tmp_path, name, twin):
    model = build_model(name, init_from=str(twins / twin), horizon_steps=SEQ, weight=1.0, hidden=8, head_hidden=8, train=CL_TRAIN)
    with pytest.raises(ValueError, match="y_seq"):
        model.fit(_windows(32, 5, seq=False), None, seed=0)
    with pytest.raises(FileNotFoundError):
        model.fit(_windows(32, 5), None, seed=3)  # no seed3 twin
    log = model.fit(_windows(64, 5), _windows(16, 6), seed=0)
    assert log["twin"] == twin and log["horizon_steps"] == SEQ and log["fit"]["epochs"] >= 1
    dist = model.predict_dist(_windows(7, 8).X)
    assert np.isfinite(dist.log_prob(np.zeros(7))).all()
    model.save(tmp_path / "model.pt")
    back = load_model(tmp_path / "model.pt")
    assert type(back).__name__ == type(model).__name__ and back.init_kwargs["init_from"] == str(twins / twin)
    assert np.allclose(back.predict_dist(_windows(7, 8).X).log_prob(np.zeros(7)), dist.log_prob(np.zeros(7)))
    gen = torch.Generator().manual_seed(0)
    state = back.init_sampler(4, gen, "per_step")
    draw = back.sample_step(torch.as_tensor(_windows(4, 9).X), state, gen)
    assert draw.shape == (4,)


def test_architecture_mismatch_is_refused(twins):
    model = build_model("hetero_mdn_cl", init_from=str(twins / "hetero_mdn"), horizon_steps=SEQ, hidden=16, head_hidden=8, train=CL_TRAIN)
    with pytest.raises(ValueError, match="architecture"):
        model.fit(_windows(32, 5), None, seed=0)
    wrong = build_model("lstm_const_sigma_cl", init_from=str(twins / "hetero_mdn"), horizon_steps=SEQ, hidden=8, head_hidden=8, train=CL_TRAIN)
    with pytest.raises(ValueError, match="checkpoint"):
        wrong.fit(_windows(32, 5), None, seed=0)
