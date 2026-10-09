"""Homoscedastic baselines (docs/design.md, section 7): LSTM mean with constant sigma, IDM plus white noise.

Both return ``GaussianDist(mu, sigma)`` with ``aleatoric_var = sigma^2`` and ``epistemic_var = 0``.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from cf_unc.data.windows import WindowSet
from cf_unc.dist import GaussianDist
from cf_unc.models.base import CFModel, as_history_tensor, register
from cf_unc.models.hetero_mdn import SeqNetModel, finish_fit_log, history_rows, train_config
from cf_unc.physics.idm import IDMParams, idm_accel
from cf_unc.train.trainer import Batch

__all__ = ["IDMWhiteNoise", "LSTMConstSigma"]


def _rms(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(residual, dtype=np.float64)))))


def _mse_loss(module: nn.Module, batch: Batch) -> torch.Tensor:
    return (module(batch["x"])[:, 0] - batch["y"]).square().mean()


def _constant_gaussian(mu: np.ndarray, sigma: float | None) -> GaussianDist:
    if sigma is None:
        raise RuntimeError("model is not fitted (sigma unknown)")
    return GaussianDist(mu, sigma, aleatoric_var=sigma**2, epistemic_var=0.0)


@register("lstm_const_sigma")
class LSTMConstSigma(SeqNetModel):
    """LSTM mean (MSE loss in normalised space) with a constant sigma [m/s^2].

    sigma = RMS residual on the validation windows (on the training windows without validation data).
    """

    def __init__(self, hidden: int = 64, head_hidden: int = 64, train: dict | None = None) -> None:
        super().__init__(hidden=hidden, head_hidden=head_hidden, train=train)
        self.net_spec = dict(out_dim=1, hidden=int(hidden), head_hidden=int(head_hidden))
        self.sigma: float | None = None

    def mean_tensor(self, hist: torch.Tensor) -> torch.Tensor:
        """Predicted mean acceleration (B, 1) [m/s^2] for physical histories (B, H, 3)."""
        return self.normalizer.denorm_mu(self.raw_outputs(hist))

    def predict_mean(self, state_history: np.ndarray | torch.Tensor, batch_size: int = 8192) -> np.ndarray:
        """(N,) float64 predicted mean [m/s^2]."""
        return self.evaluate(state_history, lambda h: (self.mean_tensor(h),), batch_size)[0][:, 0]

    def fit(self, train: WindowSet, val: WindowSet | None = None, *, seed: int = 0, verbose: bool = False) -> dict:
        started = time.perf_counter()
        fit_log = self.fit_net(train, val, _mse_loss, seed=seed, verbose=verbose)
        ref, ref_name = (val, "val") if val is not None and len(val) > 0 else (train, "train")
        self.sigma = _rms(ref.y - self.predict_mean(ref.X))
        return finish_fit_log(self, {"fit": fit_log, "sigma": self.sigma, "sigma_from": ref_name}, train, val, started)

    def predict_dist(
        self,
        state_history: np.ndarray | torch.Tensor,
        *,
        v_lead_future: np.ndarray | torch.Tensor | None = None,
        batch_size: int = 8192,
    ) -> GaussianDist:
        return _constant_gaussian(self.predict_mean(state_history, batch_size), self.sigma)

    @torch.no_grad()
    def sample_step(self, hist: torch.Tensor, state: Any, generator: torch.Generator) -> torch.Tensor:
        if self.sigma is None:
            raise RuntimeError("LSTMConstSigma is not fitted")
        self.require_net().eval()
        mu = self.mean_tensor(as_history_tensor(hist, self.device))[:, 0]
        return mu + self.sigma * torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)

    def get_state(self) -> dict:
        return {**super().get_state(), "sigma": self.sigma}

    def set_state(self, state: dict) -> None:
        super().set_state(state)
        self.sigma = float(state["sigma"])


@register("idm_white_noise")
class IDMWhiteNoise(CFModel):
    """IDM acceleration of the LAST history sample plus white Gaussian noise of constant sigma [m/s^2].

    idm: ``IDMParams.to_dict()``. sigma: fixed value, or None to fit the RMS of ``y - a_IDM`` on train.
    train: accepted like every model's TrainConfig overrides (validated, unused: nothing is optimised).
    """

    uses_history = False

    def __init__(self, idm: dict, sigma: float | None = None, train: dict | None = None) -> None:
        super().__init__(idm=idm, sigma=sigma, train=train)
        train_config(train)
        if sigma is not None and not (math.isfinite(sigma) and sigma > 0.0):
            raise ValueError("sigma must be finite and positive (or None to fit it)")
        self.params = IDMParams.from_dict(idm)
        self.sigma: float | None = None if sigma is None else float(sigma)
        self.fitted = sigma is not None

    def predict_mean(self, state_history: np.ndarray | torch.Tensor) -> np.ndarray:
        """(N,) float64 IDM acceleration [m/s^2] at the last sample of every history."""
        x = history_rows(state_history)
        last = x[:, -1].detach().cpu().double().numpy() if torch.is_tensor(x) else np.asarray(x[:, -1], dtype=np.float64)
        return np.asarray(idm_accel(last[:, 0], last[:, 1], last[:, 2], self.params), dtype=np.float64)

    def fit(self, train: WindowSet, val: WindowSet | None = None, *, seed: int = 0, verbose: bool = False) -> dict:
        started = time.perf_counter()
        fitted_here = self.init_kwargs["sigma"] is None
        if fitted_here:
            self.sigma = _rms(train.y - self.predict_mean(train.X))
        return finish_fit_log(self, {"sigma": self.sigma, "sigma_fitted": fitted_here}, train, val, started)

    def predict_dist(
        self,
        state_history: np.ndarray | torch.Tensor,
        *,
        v_lead_future: np.ndarray | torch.Tensor | None = None,
        batch_size: int = 8192,
    ) -> GaussianDist:
        return _constant_gaussian(self.predict_mean(state_history), self.sigma)

    @torch.no_grad()
    def sample_step(self, hist: torch.Tensor, state: Any, generator: torch.Generator) -> torch.Tensor:
        if self.sigma is None:
            raise RuntimeError("IDMWhiteNoise is not fitted")
        last = as_history_tensor(hist, self.device)[:, -1]
        mu = idm_accel(last[:, 0], last[:, 1], last[:, 2], self.params)
        return mu + self.sigma * torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)

    def get_state(self) -> dict:
        return {"sigma": self.sigma}

    def set_state(self, state: dict) -> None:
        self.sigma = None if state["sigma"] is None else float(state["sigma"])
        self.fitted = self.sigma is not None
