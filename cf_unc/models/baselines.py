"""Homoscedastic baselines (docs/design.md, section 7): LSTM mean with constant sigma, IDM plus white noise, and
IDM plus AR(1) noise (review control R10, docs/design.md 18.3).

All return ``GaussianDist(mu, sigma)`` with ``aleatoric_var = sigma^2`` and ``epistemic_var = 0``.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from cf_unc.constants import DT
from cf_unc.data.windows import TARGET_STEPS, WindowSet, check_target
from cf_unc.dist import GaussianDist
from cf_unc.models.base import CFModel, as_history_tensor, register
from cf_unc.models.hetero_mdn import SeqNetModel, finish_fit_log, history_rows, train_config
from cf_unc.physics.idm import IDMParams, idm_accel
from cf_unc.train.trainer import Batch

__all__ = ["IDMARNoise", "IDMWhiteNoise", "LSTMConstSigma", "fit_ar1", "previous_residual"]


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


# ------------------------------------------------------------------------------------------- AR(1) noise
def previous_residual(state_history: np.ndarray | torch.Tensor, params: IDMParams, lag: int) -> np.ndarray | torch.Tensor:
    """Residual of the previous target that the window shows: ``(v_t - v_{t-L}) / (L DT) - IDM(x_{t-L})``, (N,).

    ``L = lag`` (the target's own length: 1 for ``a``, 10 for ``mean_1s``); ``v`` is the speed column of the history
    (N, H, 3), ``x_{t-L}`` the state ``L`` samples before the last one (needs ``H > L``). Numpy in, float64 numpy out;
    a torch tensor in, a float64 tensor on its device out. In a rollout whose draws are held for ``L`` steps this is
    exactly the previous draw's realised residual (``v_t = v_{t-L} + a DT L`` before clipping); for the data it is the
    mean acceleration of the recorded window over ``[t-L, t)`` (for ``mean_1s`` exactly the target ``L`` samples
    earlier, for ``a`` the backward difference of the speed).
    """
    x = history_rows(state_history)
    lag = int(lag)
    if x.shape[1] <= lag:
        raise ValueError(f"the history ({x.shape[1]} samples) must be longer than the lag {lag}")
    if torch.is_tensor(x):
        x = x.to(torch.float64)
    else:
        x = np.asarray(x, dtype=np.float64)
    prev = x[:, -1 - lag]
    observed = (x[:, -1, 2] - prev[:, 2]) / (lag * DT)
    return observed - idm_accel(prev[:, 0], prev[:, 1], prev[:, 2], params)


def fit_ar1(resid: np.ndarray, resid_prev: np.ndarray, rho: float | None = None) -> dict[str, float]:
    """AR(1) parameters of a zero-mean residual series from pairs ``(r_t, r_{t-L})`` of the same event (float64).

    ``rho`` (unless given) = the lag-L autocorrelation ``sum(r_t r_{t-L}) / sqrt(sum(r_t^2) sum(r_{t-L}^2))``
    (uncentred, as the noise of the model has mean zero; |rho| <= 1 by Cauchy-Schwarz); the stationary variance is
    the mean square of the residual (the white-noise sigma^2 of IDMWhiteNoise on the same windows) and the innovation
    variance ``sigma_eta^2 = (1 - rho^2) mean(r^2)`` (floored at 1e-12), so that the AR(1) process keeps the marginal
    noise level of the white-noise baseline and differs from it only by the temporal correlation. Also returns the
    one-step root mean square of ``r_t - rho r_{t-L}`` (``conditional_rms``: equals ``sigma_eta`` when the observed
    previous residual has the second moment of the residual) and the number of pairs. Non-finite pairs are dropped.
    """
    r = np.asarray(resid, dtype=np.float64).ravel()
    p = np.asarray(resid_prev, dtype=np.float64).ravel()
    if r.shape != p.shape:
        raise ValueError(f"resid and resid_prev must have the same length, got {r.shape} and {p.shape}")
    ok = np.isfinite(r) & np.isfinite(p)
    r, p = r[ok], p[ok]
    if r.size < 2:
        raise ValueError("fit_ar1 needs at least two finite pairs")
    ms, ms_prev = float(np.mean(r * r)), float(np.mean(p * p))
    if rho is None:
        denom = math.sqrt(ms * ms_prev)
        rho = float(np.mean(r * p) / denom) if denom > 0.0 else 0.0
    rho = float(rho)
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must lie in (-1, 1), got {rho}")
    var_eta = max((1.0 - rho * rho) * ms, 1e-12)
    return {
        "rho": rho,
        "sigma_eta": math.sqrt(var_eta),
        "sigma_stationary": math.sqrt(ms),
        "rms_prev": math.sqrt(ms_prev),
        "conditional_rms": float(np.sqrt(np.mean((r - rho * p) ** 2))),
        "mean_resid": float(np.mean(r)),
        "n_pairs": int(r.size),
    }


@register("idm_ar_noise")
class IDMARNoise(CFModel):
    """Calibrated IDM plus AR(1) acceleration noise (review control R10, docs/design.md 18.3).

    Generative law: ``a_t = IDM(x_t) + e_t``, ``e_t = rho e_{t-L} + eta_t``, ``eta_t ~ N(0, sigma_eta^2)`` i.i.d.,
    with ``L`` the target's own length (``target`` ``a``: 1 step, ``mean_1s``: 10 steps; the E1 runner passes the
    experiment's target to every constructor with a ``target`` parameter). ``rho`` and ``sigma_eta`` come from the
    one-step residuals of the training windows (``fit_ar1``: lag-L autocorrelation of ``r = y - IDM(x_t)`` against
    the previous residual that the window shows, ``previous_residual``; ``sigma_eta^2 = (1 - rho^2) mean(r^2)``), so
    the stationary noise level equals the white-noise sigma of ``idm_white_noise`` on the same windows.

    One-step predictive law given the history: ``N(IDM(x_t) + rho r_prev, sigma_eta^2)`` with ``r_prev`` =
    ``previous_residual`` of the window; ``aleatoric_var`` is this CONDITIONAL variance ``sigma_eta^2`` (the variance
    of the represented law, so that aleatoric + epistemic = total; the stationary variance ``sigma_eta^2 / (1 -
    rho^2)`` is in the fit log as ``sigma_stationary``), ``epistemic_var`` 0. At ``rho = 0`` the law is exactly that of
    ``IDMWhiteNoise`` with the same sigma.

    Closed-loop sampler: the AR state ``e`` (B,) is carried per trajectory in the sampler state; the first draw starts
    from ``previous_residual`` of the true history, every draw advances it once (``e <- rho e + eta``, so with
    ``hold_steps = L`` it is the AR(1) of the held draws). There is no epistemic part: ``per_step`` and
    ``per_trajectory`` are the same sampler (the same random stream, ``eta`` drawn at every call).

    rho / sigma: fixed values, or None to fit them (``sigma`` is the innovation sd ``sigma_eta``). train: accepted
    like every model's TrainConfig overrides (validated, unused: nothing is optimised).
    """

    uses_history = True  # the last L + 1 samples (current state and the previous target)

    def __init__(
        self,
        idm: dict,
        target: str = "a",
        rho: float | None = None,
        sigma: float | None = None,
        train: dict | None = None,
    ) -> None:
        super().__init__(idm=idm, target=target, rho=rho, sigma=sigma, train=train)
        train_config(train)
        self.lag = int(TARGET_STEPS[check_target(str(target))])
        if rho is not None and not (math.isfinite(rho) and -1.0 < rho < 1.0):
            raise ValueError("rho must lie in (-1, 1) (or be None to fit it)")
        if sigma is not None and not (math.isfinite(sigma) and sigma > 0.0):
            raise ValueError("sigma must be finite and positive (or None to fit it)")
        self.params = IDMParams.from_dict(idm)
        self.rho: float | None = None if rho is None else float(rho)
        self.sigma: float | None = None if sigma is None else float(sigma)  # innovation sd sigma_eta
        self.fitted = rho is not None and sigma is not None

    def _require(self) -> tuple[float, float]:
        if self.rho is None or self.sigma is None:
            raise RuntimeError("IDMARNoise is not fitted")
        return self.rho, self.sigma

    def predict_mean(self, state_history: np.ndarray | torch.Tensor) -> np.ndarray:
        """(N,) float64 conditional mean ``IDM(x_t) + rho r_prev`` [m/s^2]."""
        rho, _ = self._require()
        x = history_rows(state_history)
        x = x.detach().cpu().double().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float64)
        last = x[:, -1]
        mu = np.asarray(idm_accel(last[:, 0], last[:, 1], last[:, 2], self.params), dtype=np.float64)
        return mu + rho * previous_residual(x, self.params, self.lag)

    def fit(self, train: WindowSet, val: WindowSet | None = None, *, seed: int = 0, verbose: bool = False) -> dict:
        started = time.perf_counter()
        x = np.asarray(train.X, dtype=np.float64)
        last = x[:, -1]
        resid = np.asarray(train.y, dtype=np.float64) - idm_accel(last[:, 0], last[:, 1], last[:, 2], self.params)
        stats = fit_ar1(resid, previous_residual(x, self.params, self.lag), self.init_kwargs["rho"])
        self.rho = stats["rho"]
        fixed_sigma = self.init_kwargs["sigma"]
        self.sigma = stats["sigma_eta"] if fixed_sigma is None else float(fixed_sigma)
        if fixed_sigma is not None:
            stats["sigma_stationary"] = self.sigma / math.sqrt(1.0 - self.rho * self.rho)
        log = {**stats, "sigma_eta": self.sigma, "lag": self.lag, "target": self.init_kwargs["target"],
               "rho_fitted": self.init_kwargs["rho"] is None, "sigma_fitted": fixed_sigma is None,
               "aleatoric": "conditional variance sigma_eta^2 of the one-step law (stationary: sigma_stationary^2)"}  # fmt: skip
        return finish_fit_log(self, log, train, val, started)

    def predict_dist(
        self,
        state_history: np.ndarray | torch.Tensor,
        *,
        v_lead_future: np.ndarray | torch.Tensor | None = None,
        batch_size: int = 8192,
    ) -> GaussianDist:
        _, sigma = self._require()
        return _constant_gaussian(self.predict_mean(state_history), sigma)

    def init_sampler(self, n_traj: int, generator: torch.Generator, mode: str = "per_step") -> dict:
        """AR state ``{"e": None}``: set from the history at the first draw (``previous_residual``), then carried."""
        super().init_sampler(n_traj, generator, mode)  # validates the mode (both modes are the same sampler)
        return {"e": None}

    @torch.no_grad()
    def sample_step(self, hist: torch.Tensor, state: dict | None, generator: torch.Generator) -> torch.Tensor:
        """(B,) ``IDM(x_t) + e_t`` with ``e_t = rho e_prev + sigma_eta eta_t`` [m/s^2], float64 (not clipped)."""
        rho, sigma = self._require()
        h = as_history_tensor(hist, self.device)
        last = h[:, -1].to(torch.float64)
        mu = idm_accel(last[:, 0], last[:, 1], last[:, 2], self.params)
        e_prev = None if state is None else state.get("e")
        if e_prev is None:
            e_prev = previous_residual(h, self.params, self.lag)
        eta = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=torch.float32).to(torch.float64)
        e = rho * e_prev + sigma * eta
        if state is not None:
            state["e"] = e
        return mu + e

    def get_state(self) -> dict:
        return {"rho": self.rho, "sigma": self.sigma}

    def set_state(self, state: dict) -> None:
        self.rho = None if state["rho"] is None else float(state["rho"])
        self.sigma = None if state["sigma"] is None else float(state["sigma"])
        self.fitted = self.rho is not None and self.sigma is not None
