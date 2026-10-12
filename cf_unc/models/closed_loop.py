"""One-step networks fine-tuned in closed loop (extension S7, extension/docs/PLAN.md, section S).

``hetero_mdn_cl`` and ``lstm_const_sigma_cl`` are the stored one-step networks (``init_from``: the checkpoint directory
of the one-step twin, ``<init_from>/seed<k>/model.pt``, the same architecture) fine-tuned with the twin's one-step
loss plus ``weight`` times a closed-loop term: the squared gap and speed errors of a deterministic rollout of
``horizon_steps`` steps with the predicted mean acceleration behind the recorded leader (``v_lead_future``), in units
of the standard deviations of the gap and of the speed over the training histories, against the recorded follower
integrated from its recorded accelerations (``y_seq``) with the same scheme. The integration is that of the rollout
engine (:mod:`cf_unc.rollout.engine`): acceleration clipped to [A_MIN, A_MAX], semi-implicit Euler at DT with the
standstill floor, the gap floored at S_EPS in the model input. The predictive law, the sampler, the evaluation and
the persistence are those of the twin (the fine-tuned model is a one-step law: it needs no leader future at
inference); training needs sequence windows (``needs_sequences`` in the model config: ``y_seq`` and
``v_lead_future``). Early stopping uses the combined loss on the validation windows. Both are one-step laws in E2
and E3 (per-step sampler, hold 1).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from cf_unc.constants import A_MAX, A_MIN, DT, S_EPS, SEQ_LEN
from cf_unc.data.windows import WindowSet
from cf_unc.models.base import register
from cf_unc.models.baselines import LSTMConstSigma, _mse_loss, _rms
from cf_unc.models.hetero_mdn import HeteroMDN, finish_fit_log
from cf_unc.train.trainer import fit_module

Batch = dict[str, torch.Tensor]
MeanFn = Callable[[torch.Tensor], torch.Tensor]  # physical histories (B, H, 3) -> mean acceleration (B,) [m/s^2]


def rollout_loss(mean_fn: MeanFn, hist: torch.Tensor, v_lead: torch.Tensor, a_obs: torch.Tensor, s_scale: torch.Tensor,
                 v_scale: torch.Tensor, horizon: int) -> torch.Tensor:  # fmt: skip
    """Mean over ``horizon`` steps of the squared gap and speed errors (each in units of its scale) between the
    rollout of ``mean_fn`` and the recorded follower, both integrated as the rollout engine from the last history
    row of ``hist`` (B, H, 3) [s, dv, v] behind the leader speeds ``v_lead`` (B, >= horizon) at t+1 .. t+horizon;
    ``a_obs`` (B, >= horizon) the recorded accelerations a[t] .. a[t+horizon-1]. Gradients flow through ``mean_fn``."""
    if horizon < 1 or v_lead.shape[1] < horizon or a_obs.shape[1] < horizon:
        raise ValueError(f"need 1 <= horizon <= the stored future, got {horizon}, {v_lead.shape[1]}, {a_obs.shape[1]}")
    v, s = hist[:, -1, 2], hist[:, -1, 0]
    v_obs, s_obs = v, s
    err = torch.zeros((), dtype=hist.dtype, device=hist.device)
    for h in range(horizon):
        a = mean_fn(hist).clamp(A_MIN, A_MAX)
        v = (v + a * DT).clamp(min=0.0)
        s = s + (v_lead[:, h] - v) * DT
        v_obs = (v_obs + a_obs[:, h].clamp(A_MIN, A_MAX) * DT).clamp(min=0.0)
        s_obs = s_obs + (v_lead[:, h] - v_obs) * DT
        err = err + torch.mean(((s - s_obs) / s_scale) ** 2 + ((v - v_obs) / v_scale) ** 2)
        row = torch.stack([s.clamp(min=S_EPS), v - v_lead[:, h], v], dim=-1)
        hist = torch.cat([hist[:, 1:], row.unsqueeze(1)], dim=1)
    return err / horizon


class _ClosedLoopMixin:
    """Fine-tuning of a stored one-step twin with the combined loss; ``mean_of`` and ``one_step_loss`` come from the
    concrete class."""

    init_from: str
    horizon_steps: int
    weight: float
    twin_name: str

    def _check_sequences(self, name: str, ws: WindowSet | None) -> None:
        if ws is None:
            return
        if ws.y_seq is None or ws.v_lead_future is None:
            raise ValueError(f"{type(self).__name__} needs {name} windows with y_seq and v_lead_future (seq_len > 0)")
        if ws.y_seq.shape[1] < self.horizon_steps:
            raise ValueError(f"{name} windows hold {ws.y_seq.shape[1]} future steps, horizon_steps is {self.horizon_steps}")

    def _load_twin(self, seed: int):  # noqa: ANN202
        from cf_unc.models import load_model

        path = Path(self.init_from) / f"seed{int(seed)}" / "model.pt"
        if not path.exists():
            raise FileNotFoundError(f"init_from checkpoint {path} does not exist")
        twin = load_model(path, device=self.device)  # type: ignore[attr-defined]
        if twin.name != self.twin_name:
            raise ValueError(f"{path} holds a {twin.name!r} checkpoint, {type(self).__name__} fine-tunes {self.twin_name!r}")
        if twin.net_spec != self.net_spec:  # type: ignore[attr-defined]
            raise ValueError(f"architecture of {path} {twin.net_spec} differs from the model config {self.net_spec}")  # type: ignore[attr-defined]
        return twin

    def _arrays_cl(self, ws: WindowSet) -> Batch:
        arrays = self._arrays(ws)  # type: ignore[attr-defined]  # normalised x and y of the twin
        arrays["y_seq"] = torch.as_tensor(ws.y_seq, dtype=torch.float32, device=self.device)  # type: ignore[attr-defined]
        arrays["vl"] = torch.as_tensor(ws.v_lead_future, dtype=torch.float32, device=self.device)  # type: ignore[attr-defined]
        return arrays

    def _combined_loss(self) -> Callable[[nn.Module, Batch], torch.Tensor]:
        normalizer = self.normalizer  # type: ignore[attr-defined]
        s_scale, v_scale = normalizer.x_std[0], normalizer.x_std[2]
        one_step = self.one_step_loss()
        weight, horizon = float(self.weight), int(self.horizon_steps)

        def loss(module: nn.Module, batch: Batch) -> torch.Tensor:
            hist = batch["x"] * normalizer.x_std + normalizer.x_mean  # physical histories
            closed = rollout_loss(lambda h: self.mean_of(module, h), hist, batch["vl"], batch["y_seq"], s_scale, v_scale, horizon)
            return one_step(module, batch) + weight * closed

        return loss

    def fit_closed_loop(self, train: WindowSet, val: WindowSet | None, *, seed: int, verbose: bool) -> dict:
        self._check_sequences("train", train)
        self._check_sequences("val", val)
        if len(train) == 0:
            raise ValueError("empty training set")
        twin = self._load_twin(seed)
        self.normalizer = twin.normalizer.to(self.device)  # type: ignore[attr-defined]
        self.net = twin.net.to(self.device)  # type: ignore[attr-defined]
        torch.manual_seed(int(seed))
        val_arrays = self._arrays_cl(val) if val is not None and len(val) > 0 else None
        loss = self._combined_loss()
        log = fit_module(self.net, loss, self._arrays_cl(train), val_arrays, self.train_cfg, seed=int(seed),  # type: ignore[attr-defined]
                         device=self.device, verbose=verbose, val_loss_fn=loss)  # type: ignore[attr-defined]
        self.fitted = True
        return {"fit": log, "init_from": str(self.init_from), "twin": self.twin_name, "horizon_steps": int(self.horizon_steps),
                "weight": float(self.weight)}  # fmt: skip


@register("hetero_mdn_cl")
class HeteroMDNClosedLoop(_ClosedLoopMixin, HeteroMDN):
    """HeteroMDN fine-tuned in closed loop (module docstring): the twin's NLL plus ``weight`` times the rollout loss of
    the mixture mean over ``horizon_steps`` steps."""

    twin_name = "hetero_mdn"

    def __init__(
        self,
        init_from: str,
        horizon_steps: int = SEQ_LEN,
        weight: float = 1.0,
        n_components: int = 1,
        hidden: int = 64,
        head_hidden: int = 64,
        dropout: float = 0.0,
        log_sigma_min: float = -7.0,
        log_sigma_max: float = 3.0,
        beta_nll: float = 0.0,
        train: dict | None = None,
    ) -> None:
        super().__init__(n_components=n_components, hidden=hidden, head_hidden=head_hidden, dropout=dropout,
                         log_sigma_min=log_sigma_min, log_sigma_max=log_sigma_max, beta_nll=beta_nll, train=train)  # fmt: skip
        if int(horizon_steps) < 1 or float(weight) < 0.0:
            raise ValueError("need horizon_steps >= 1 and weight >= 0")
        self.init_from, self.horizon_steps, self.weight = str(init_from), int(horizon_steps), float(weight)
        self.init_kwargs.update(init_from=self.init_from, horizon_steps=self.horizon_steps, weight=self.weight)

    def mean_of(self, module: nn.Module, hist: torch.Tensor) -> torch.Tensor:
        """Mixture mean acceleration (B,) [m/s^2] of ``module`` for physical histories (B, H, 3), with gradients."""
        log_w, mu_n, _ = self.split_raw(module(self.normalizer.norm_x(hist)))
        return self.normalizer.denorm_mu((log_w.exp() * mu_n).sum(dim=-1))

    def one_step_loss(self) -> Callable[[nn.Module, Batch], torch.Tensor]:
        return self.loss_fn(self.beta_nll)

    def fit(self, train: WindowSet, val: WindowSet | None = None, *, seed: int = 0, verbose: bool = False) -> dict:
        started = time.perf_counter()
        return finish_fit_log(self, self.fit_closed_loop(train, val, seed=seed, verbose=verbose), train, val, started)


@register("lstm_const_sigma_cl")
class LSTMConstSigmaClosedLoop(_ClosedLoopMixin, LSTMConstSigma):
    """LSTMConstSigma fine-tuned in closed loop (module docstring): the twin's MSE plus ``weight`` times the rollout
    loss of the predicted mean; the constant sigma is refitted on the validation residuals as for the twin."""

    twin_name = "lstm_const_sigma"

    def __init__(self, init_from: str, horizon_steps: int = SEQ_LEN, weight: float = 1.0, hidden: int = 64,
                 head_hidden: int = 64, train: dict | None = None) -> None:  # fmt: skip
        super().__init__(hidden=hidden, head_hidden=head_hidden, train=train)
        if int(horizon_steps) < 1 or float(weight) < 0.0:
            raise ValueError("need horizon_steps >= 1 and weight >= 0")
        self.init_from, self.horizon_steps, self.weight = str(init_from), int(horizon_steps), float(weight)
        self.init_kwargs.update(init_from=self.init_from, horizon_steps=self.horizon_steps, weight=self.weight)

    def mean_of(self, module: nn.Module, hist: torch.Tensor) -> torch.Tensor:
        return self.normalizer.denorm_mu(module(self.normalizer.norm_x(hist)))[:, 0]

    def one_step_loss(self) -> Callable[[nn.Module, Batch], torch.Tensor]:
        return _mse_loss

    def fit(self, train: WindowSet, val: WindowSet | None = None, *, seed: int = 0, verbose: bool = False) -> dict:
        started = time.perf_counter()
        log = self.fit_closed_loop(train, val, seed=seed, verbose=verbose)
        ref, ref_name = (val, "val") if val is not None and len(val) > 0 else (train, "train")
        self.sigma = _rms(ref.y - self.predict_mean(ref.X))
        log.update(sigma=self.sigma, sigma_from=ref_name)
        return finish_fit_log(self, log, train, val, started)


__all__ = ["HeteroMDNClosedLoop", "LSTMConstSigmaClosedLoop", "rollout_loss"]
