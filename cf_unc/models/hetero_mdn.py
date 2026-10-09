"""Heteroscedastic LSTM mixture density network (docs/design.md, section 7).

``SeqNetModel`` holds what every model built on one ``SeqRegressor`` shares (normaliser, training
call, chunked evaluation, persistence); ``HeteroMDN`` adds the Gaussian / mixture output head.
``DeepEnsemble`` and ``MCDropout`` are built from ``HeteroMDN`` objects, ``LSTMConstSigma`` from
``SeqNetModel``. The helpers ``train_config``, ``finish_fit_log`` and ``draw_mixture`` are shared
by all model files of this family.

Shapes: histories (N, H, 3) of ``(s, dv, v)`` in physical units; mixture parameters (N, K) with
weights summing to 1, means and standard deviations in m/s^2.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn

from cf_unc.data.windows import WindowSet
from cf_unc.dist import GaussianDist, MixtureDist, PredDist
from cf_unc.models.base import CFModel, as_history_tensor, register
from cf_unc.models.nets import Normalizer, SeqRegressor
from cf_unc.train.trainer import Batch, TrainConfig, fit_module

__all__ = [
    "HeteroMDN",
    "SeqNetModel",
    "draw_mixture",
    "finish_fit_log",
    "history_rows",
    "train_config",
]

LOG_2PI = math.log(2.0 * math.pi)
Masks = tuple[torch.Tensor, torch.Tensor]  # dropout masks after the LSTM state / after the hidden layer
Params = tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # weights, mus, sigmas, each (..., B, K)


# --------------------------------------------------------------------------- helpers
def train_config(train: Mapping[str, Any] | None) -> TrainConfig:
    """``TrainConfig`` with the overrides ``train``; unknown keys are an error (typos)."""
    overrides = dict(train or {})
    unknown = sorted(set(overrides) - set(TrainConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown TrainConfig fields {unknown}")
    return TrainConfig(**overrides)


def history_rows(state_history: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """(N, H, 3) view of a batch of histories or of one (H, 3) history (numpy or torch, no copy)."""
    x = state_history if torch.is_tensor(state_history) else np.asarray(state_history)
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"state history must have shape (N, H, 3) or (H, 3), got {tuple(x.shape)}")
    return x


def finish_fit_log(model: CFModel, log: dict, train: WindowSet, val: WindowSet | None, started: float) -> dict:
    """Mark ``model`` fitted and complete its training log.

    Adds ``val_nll`` (mean negative log-likelihood of ``val.y`` under the model's final predictive
    law, physical units, nats; None without validation data), ``n_train`` and ``seconds`` (wall time
    since ``started``, a ``time.perf_counter()`` value).
    """
    model.fitted = True
    val_nll = None
    if val is not None and len(val) > 0:
        val_nll = float(-np.mean(model.predict_dist(val.X).log_prob(val.y)))
    log.update(val_nll=val_nll, n_train=int(len(train)), seconds=time.perf_counter() - started)
    return log


def draw_mixture(weights: torch.Tensor, mus: torch.Tensor, sigmas: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """One draw per row of Gaussian mixtures with (B, K) parameters -> (B,); uses only ``generator``.

    The component is chosen by inverse CDF of the weights (skipped for K = 1), then a Gaussian is drawn.
    """
    if mus.shape[-1] == 1:
        mu, sd = mus[:, 0], sigmas[:, 0]
    else:
        u = torch.rand((mus.shape[0], 1), generator=generator, device=mus.device, dtype=weights.dtype)
        cum = weights.cumsum(dim=1)
        comp = (cum < u * cum[:, -1:]).sum(dim=1, keepdim=True).clamp_(max=mus.shape[1] - 1)
        mu, sd = mus.gather(1, comp)[:, 0], sigmas.gather(1, comp)[:, 0]
    eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
    return mu + sd * eps


def _mixture_nll(log_w: torch.Tensor, mu: torch.Tensor, log_sd: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-sample NLL (B,) of y (B,) under Gaussian mixtures with (B, K) log-weights, means, log-sigmas."""
    z = (y[:, None] - mu) * torch.exp(-log_sd)
    return 0.5 * LOG_2PI - torch.logsumexp(log_w - log_sd - 0.5 * z * z, dim=-1)


def _state_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


# --------------------------------------------------------------------------- shared base
class SeqNetModel(CFModel):
    """CFModel around one ``SeqRegressor`` that works on normalised histories and targets.

    Subclasses set ``self.net_spec`` (``SeqRegressor`` keyword arguments) in their constructor.
    """

    net_spec: dict[str, Any]

    def __init__(self, **init_kwargs: Any) -> None:
        super().__init__(**init_kwargs)
        self.train_cfg = train_config(init_kwargs.get("train"))
        self.normalizer = Normalizer()
        self.net: SeqRegressor | None = None

    # ------------------------------------------------------------------ training
    def _arrays(self, windows: WindowSet) -> Batch:
        """Normalised training arrays: ``x`` (N, H, 3), ``y`` (N,), float32 on the model device."""
        with torch.no_grad():
            x = torch.as_tensor(windows.X, dtype=torch.float32, device=self.device)
            y = torch.as_tensor(windows.y, dtype=torch.float32, device=self.device)
            return {"x": self.normalizer.norm_x(x), "y": self.normalizer.norm_y(y)}

    def fit_net(
        self,
        train: WindowSet,
        val: WindowSet | None,
        loss_fn: Callable[[nn.Module, Batch], torch.Tensor],
        *,
        seed: int,
        verbose: bool,
        val_loss_fn: Callable[[nn.Module, Batch], torch.Tensor] | None = None,
    ) -> dict:
        """Seed torch, fit the normaliser on ``train``, build the network and run ``fit_module``."""
        if len(train) == 0:
            raise ValueError("empty training set")
        torch.manual_seed(int(seed))
        self.normalizer = Normalizer().fit(train.X, train.y).to(self.device)
        self.net = SeqRegressor(**self.net_spec).to(self.device)
        val_arrays = self._arrays(val) if val is not None and len(val) > 0 else None
        log = fit_module(
            self.net,
            loss_fn,
            self._arrays(train),
            val_arrays,
            self.train_cfg,
            seed=int(seed),
            device=self.device,
            verbose=verbose,
            val_loss_fn=val_loss_fn,
        )
        self.fitted = True
        return log

    # ---------------------------------------------------------------- evaluation
    def require_net(self) -> SeqRegressor:
        if self.net is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        return self.net

    def raw_outputs(self, hist: torch.Tensor, masks: Masks | None = None) -> torch.Tensor:
        """Raw network outputs (..., B, out_dim) for physical histories (B, H, 3).

        ``masks`` are dropout masks broadcast against the hidden states: (B, width) per row,
        (1, width) shared by all rows, or (P, 1, width) for P networks sharing one LSTM pass.
        """
        return self.require_net()(self.normalizer.norm_x(hist), masks)

    @torch.no_grad()
    def evaluate(
        self,
        state_history: np.ndarray | torch.Tensor,
        fn: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
        batch_size: int = 8192,
        width: int = 1,
    ) -> tuple[np.ndarray, ...]:
        """Apply ``fn`` to row chunks of the histories in eval mode; float64 numpy results.

        ``fn`` maps a physical (b, H, 3) float32 tensor to tensors with the rows on axis -2. Chunks
        hold ``batch_size // width`` rows (``width`` = networks evaluated per row), which bounds memory.
        """
        self.require_net().eval()
        x = history_rows(state_history)
        step = max(1, int(batch_size) // max(1, int(width)))
        starts = range(0, x.shape[0], step) if x.shape[0] > 0 else [0]
        parts = [fn(as_history_tensor(x[a : a + step], self.device)) for a in starts]
        return tuple(torch.cat([p[i] for p in parts], dim=-2).double().cpu().numpy() for i in range(len(parts[0])))

    # --------------------------------------------------------------- persistence
    def get_state(self) -> dict:
        return {"net": _state_to_cpu(self.require_net()), "normalizer": _state_to_cpu(self.normalizer)}

    def set_state(self, state: dict) -> None:
        normalizer = Normalizer()
        normalizer.load_state_dict(state["normalizer"])
        with torch.random.fork_rng(devices=[]):  # the throw-away initial weights leave the global RNG alone
            net = SeqRegressor(**self.net_spec)
        net.load_state_dict(state["net"])
        self.normalizer = normalizer.to(self.device)
        self.net = net.to(self.device).eval()
        self.fitted = True


# --------------------------------------------------------------------------- HeteroMDN
@register("hetero_mdn")
class HeteroMDN(SeqNetModel):
    """LSTM + MLP head -> Gaussian (K = 1) or K-component Gaussian mixture, trained by exact NLL.

    Raw outputs: ``(mu, log_sigma)`` for K = 1, ``(logits (K), mu (K), log_sigma (K))`` for K > 1, in
    normalised target space with ``log_sigma`` clamped to ``[log_sigma_min, log_sigma_max]``.
    ``beta_nll > 0`` (K = 1 only) weights the per-sample NLL by ``sigma^(2 beta)`` with a stopped
    gradient (Seitzer et al. 2022); early stopping always uses the plain NLL.
    """

    def __init__(
        self,
        n_components: int = 1,
        hidden: int = 64,
        head_hidden: int = 64,
        dropout: float = 0.0,
        log_sigma_min: float = -7.0,
        log_sigma_max: float = 3.0,
        beta_nll: float = 0.0,
        train: dict | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            hidden=hidden,
            head_hidden=head_hidden,
            dropout=dropout,
            log_sigma_min=log_sigma_min,
            log_sigma_max=log_sigma_max,
            beta_nll=beta_nll,
            train=train,
        )
        if int(n_components) < 1:
            raise ValueError("n_components must be >= 1")
        if not log_sigma_min < log_sigma_max:
            raise ValueError("need log_sigma_min < log_sigma_max")
        if beta_nll < 0.0 or (beta_nll > 0.0 and int(n_components) != 1):
            raise ValueError("beta_nll must be >= 0 and is only defined for n_components = 1")
        self.n_components = int(n_components)
        self.log_sigma_min = float(log_sigma_min)
        self.log_sigma_max = float(log_sigma_max)
        self.beta_nll = float(beta_nll)
        out_dim = 2 if self.n_components == 1 else 3 * self.n_components
        self.net_spec = dict(out_dim=out_dim, hidden=int(hidden), head_hidden=int(head_hidden), dropout=float(dropout))

    # ------------------------------------------------------------------ decoding
    def split_raw(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Raw outputs (..., out_dim) -> log-weights, means, clamped log-sigmas (..., K), normalised space."""
        k = self.n_components
        if k == 1:
            mu, log_sd = raw[..., 0:1], raw[..., 1:2]
            log_w = torch.zeros_like(mu)
        else:
            log_w = torch.log_softmax(raw[..., :k], dim=-1)
            mu, log_sd = raw[..., k : 2 * k], raw[..., 2 * k :]
        return log_w, mu, log_sd.clamp(self.log_sigma_min, self.log_sigma_max)

    def predict_params(self, hist_tensor: torch.Tensor, masks: Masks | None = None) -> Params:
        """Mixture weights, means, sigmas (B, K) in physical units for histories (B, H, 3) (torch).

        The single place where raw outputs become physical parameters. With (P, 1, width) masks
        (see ``raw_outputs``) the results are (P, B, K), one set per dropout network.
        """
        log_w, mu_n, log_sd_n = self.split_raw(self.raw_outputs(hist_tensor, masks))
        return log_w.exp(), self.normalizer.denorm_mu(mu_n), self.normalizer.denorm_sigma(log_sd_n.exp())

    def predict_params_np(
        self, state_history: np.ndarray | torch.Tensor, batch_size: int = 8192, masks: Masks | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Chunked ``predict_params`` without gradients: float64 (N, K), or (P, N, K) with (P, 1, width) masks."""
        width = int(masks[0].shape[0]) if masks is not None and masks[0].ndim == 3 else 1
        return self.evaluate(state_history, lambda h: self.predict_params(h, masks), batch_size, width)  # type: ignore[return-value]

    # ------------------------------------------------------------------ training
    def loss_fn(self, beta: float) -> Callable[[nn.Module, Batch], torch.Tensor]:
        """Mean (beta-weighted) NLL of a normalised batch ``{"x", "y"}`` under the module's outputs."""

        def loss(module: nn.Module, batch: Batch) -> torch.Tensor:
            log_w, mu, log_sd = self.split_raw(module(batch["x"]))
            nll = _mixture_nll(log_w, mu, log_sd, batch["y"])
            if beta > 0.0:
                nll = nll * torch.exp(2.0 * beta * log_sd[:, 0]).detach()
            return nll.mean()

        return loss

    def fit(self, train: WindowSet, val: WindowSet | None = None, *, seed: int = 0, verbose: bool = False) -> dict:
        started = time.perf_counter()
        fit_log = self.fit_net(
            train, val, self.loss_fn(self.beta_nll), seed=seed, verbose=verbose, val_loss_fn=self.loss_fn(0.0)
        )
        return finish_fit_log(self, {"fit": fit_log}, train, val, started)

    # ---------------------------------------------------------------- prediction
    def predict_dist(
        self,
        state_history: np.ndarray | torch.Tensor,
        *,
        v_lead_future: np.ndarray | torch.Tensor | None = None,
        batch_size: int = 8192,
    ) -> PredDist:
        """``GaussianDist`` (K = 1) or ``MixtureDist``; ``aleatoric_var`` = its variance, no epistemic part."""
        w, mu, sd = self.predict_params_np(state_history, batch_size)
        if self.n_components == 1:
            return GaussianDist(mu[:, 0], sd[:, 0], aleatoric_var=sd[:, 0] ** 2)
        dist = MixtureDist(w, mu, sd)
        dist.aleatoric_var = dist.var()
        return dist

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def step_params(self, hist: torch.Tensor, state: Any, generator: torch.Generator) -> Params:
        """Law of this step's draw for every trajectory, (B, K) physical (nothing is frozen)."""
        self.require_net().eval()
        return self.predict_params(hist)

    @torch.no_grad()
    def sample_step(self, hist: torch.Tensor, state: Any, generator: torch.Generator) -> torch.Tensor:
        return draw_mixture(*self.step_params(as_history_tensor(hist, self.device), state, generator), generator)
