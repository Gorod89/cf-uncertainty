"""Frozen-mean heteroscedastic ablation (review control R13, docs/design.md 18.4).

``LSTMHeteroFrozen`` (registry name ``lstm_hetero_frozen``) takes the trained ``LSTMConstSigma`` of the same seed
(``<mean_from>/seed<k>/model.pt``), freezes it (LSTM, mean head and normaliser), and trains only a variance head
``log sigma(x)`` by Gaussian NLL on top of the frozen LSTM features (the last LSTM state, the representation the mean
head reads). The one-step law is ``N(mu_frozen(x), sigma(x)^2)``: its mean is exactly the mean of
``lstm_const_sigma`` of that seed, so the H3.3 comparison ``lstm_hetero_frozen`` against ``lstm_const_sigma`` changes
the variance only (constant sigma -> learned sigma(x)), at one mean and one backbone.

Choice of the variance input: the frozen LSTM features (not a second network on the state history), so that the two
models of the comparison share the backbone as well as the mean (the reviewer's "common backbone"); the variance head
has the shape of the head of ``HeteroMDN`` (Linear(hidden, head_hidden), SiLU, Linear(head_hidden, 1)), ``log sigma``
clamped to ``[log_sigma_min, log_sigma_max]`` in normalised target space as there, and the same training budget
(the TrainConfig of ``HeteroMDN``: configs/train/default.yaml, max 40 epochs, batch 1024, early stopping on the
validation NLL). The features of the training and validation windows are computed once (the backbone is frozen), which
only saves time.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cf_unc.data.windows import WindowSet, check_target
from cf_unc.dist import GaussianDist
from cf_unc.models.base import CFModel, as_history_tensor, register
from cf_unc.models.baselines import LSTMConstSigma
from cf_unc.models.hetero_mdn import finish_fit_log, history_rows, train_config
from cf_unc.train.trainer import Batch, fit_module

__all__ = ["LSTMHeteroFrozen"]

LOG_2PI = math.log(2.0 * math.pi)
FEATURE_CHUNK = 65_536  # windows per chunk of the frozen forward pass


def _state_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


@register("lstm_hetero_frozen")
class LSTMHeteroFrozen(CFModel):
    """Frozen ``LSTMConstSigma`` mean plus a learned ``sigma(x)`` on its LSTM features (module docstring).

    mean_from: checkpoints directory of ``lstm_const_sigma`` (``<mean_from>/seed<k>/model.pt`` is used for seed k;
    relative paths: repository root). target: the experiment's target (passed by the E1 runner); the source
    checkpoint must have been trained for it (its ``train_log.json``). train: TrainConfig overrides of the variance
    head (default: the budget of HeteroMDN).
    """

    def __init__(
        self,
        mean_from: str,
        target: str = "a",
        head_hidden: int = 64,
        log_sigma_min: float = -7.0,
        log_sigma_max: float = 3.0,
        train: dict | None = None,
    ) -> None:
        super().__init__(mean_from=str(mean_from), target=str(target), head_hidden=int(head_hidden),
                         log_sigma_min=float(log_sigma_min), log_sigma_max=float(log_sigma_max),
                         train=None if train is None else dict(train))  # fmt: skip
        check_target(str(target))
        if not log_sigma_min < log_sigma_max:
            raise ValueError("need log_sigma_min < log_sigma_max")
        if int(head_hidden) < 1:
            raise ValueError("head_hidden must be >= 1")
        self.train_cfg = train_config(train)
        self.mean_model: LSTMConstSigma | None = None
        self.head: nn.Sequential | None = None
        self.source: dict[str, Any] | None = None

    # ------------------------------------------------------------------ helpers
    def to(self, device: str | torch.device) -> "LSTMHeteroFrozen":
        super().to(device)
        if self.mean_model is not None:
            self.mean_model.to(self.device)
        return self

    def _build_head(self, hidden: int) -> nn.Sequential:
        h = int(self.init_kwargs["head_hidden"])
        return nn.Sequential(nn.Linear(hidden, h), nn.SiLU(), nn.Linear(h, 1))

    def _require(self) -> tuple[LSTMConstSigma, nn.Sequential]:
        if self.mean_model is None or self.head is None:
            raise RuntimeError("LSTMHeteroFrozen is not fitted")
        return self.mean_model, self.head

    def _frozen(self, hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalised mean (B,) and LSTM features (B, hidden) of physical histories (B, H, 3), frozen network."""
        mean = self.mean_model
        assert mean is not None
        net = mean.require_net()
        h = net.encode(mean.normalizer.norm_x(hist))
        mu_n = net.fc2(net.drop2(F.silu(net.fc1(net.drop1(h)))))[:, 0]  # = net(x)[:, 0] (SeqRegressor.forward)
        return mu_n, h

    def _log_sigma(self, features: torch.Tensor) -> torch.Tensor:
        """(B,) clamped log sigma in normalised target space."""
        lo, hi = float(self.init_kwargs["log_sigma_min"]), float(self.init_kwargs["log_sigma_max"])
        return self.head(features)[:, 0].clamp(lo, hi)  # type: ignore[misc]

    @torch.no_grad()
    def _arrays(self, windows: WindowSet) -> Batch:
        """Frozen features ``f`` (N, hidden), normalised mean ``mu`` (N,) and target ``y`` (N,) on the device."""
        mean = self.mean_model
        assert mean is not None
        parts_f, parts_mu = [], []
        for a in range(0, len(windows), FEATURE_CHUNK):
            hist = as_history_tensor(windows.X[a : a + FEATURE_CHUNK], self.device)
            mu_n, h = self._frozen(hist)
            parts_f.append(h)
            parts_mu.append(mu_n)
        y = torch.as_tensor(windows.y, dtype=torch.float32, device=self.device)
        return {"f": torch.cat(parts_f), "mu": torch.cat(parts_mu), "y": mean.normalizer.norm_y(y)}

    def _nll(self, module: nn.Module, batch: Batch) -> torch.Tensor:
        lo, hi = float(self.init_kwargs["log_sigma_min"]), float(self.init_kwargs["log_sigma_max"])
        log_sd = module(batch["f"])[:, 0].clamp(lo, hi)
        z = (batch["y"] - batch["mu"]) * torch.exp(-log_sd)
        return (0.5 * LOG_2PI + log_sd + 0.5 * z * z).mean()

    def _load_source(self, seed: int) -> tuple[LSTMConstSigma, dict[str, Any]]:
        """The frozen mean model of ``seed`` and its provenance; ValueError for another model or target."""
        from cf_unc.train.run import _path  # the repository-root resolution of the runners

        directory = _path(str(self.init_kwargs["mean_from"])) / f"seed{int(seed)}"
        path = directory / "model.pt"
        if not path.exists():
            raise FileNotFoundError(f"lstm_hetero_frozen: the mean checkpoint {path} does not exist (mean_from, seed {seed})")
        model = CFModel.load(path, device=self.device)
        if not isinstance(model, LSTMConstSigma):
            raise ValueError(f"lstm_hetero_frozen: {path} holds a {type(model).__name__}, expected LSTMConstSigma")
        log_path = directory / "train_log.json"
        info: dict[str, Any] = {"path": str(self.init_kwargs["mean_from"]).replace("\\", "/") + f"/seed{int(seed)}/model.pt",
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "seed": int(seed),
                                "sigma_const": float(model.sigma) if model.sigma is not None else None}  # fmt: skip
        if log_path.exists():
            log = json.loads(log_path.read_text(encoding="utf-8"))
            stored_target = log.get("target") or "a"
            if stored_target != self.init_kwargs["target"]:
                raise ValueError(f"lstm_hetero_frozen: {path} was trained for the target {stored_target!r}, this model "
                                 f"is fitted for {self.init_kwargs['target']!r}")  # fmt: skip
            info.update(target=stored_target, signal=log.get("signal"), kwargs=log.get("kwargs"))
        return model, info

    # ------------------------------------------------------------------ training
    def fit(self, train: WindowSet, val: WindowSet | None = None, *, seed: int = 0, verbose: bool = False) -> dict:
        """Load the mean of ``seed`` (frozen), train the variance head by NLL; JSON-able log."""
        started = time.perf_counter()
        if len(train) == 0:
            raise ValueError("empty training set")
        mean, self.source = self._load_source(seed)
        mean.require_net().eval().requires_grad_(False)
        mean.normalizer.requires_grad_(False)
        self.mean_model = mean
        train_arrays = self._arrays(train)
        val_arrays = self._arrays(val) if val is not None and len(val) > 0 else None
        torch.manual_seed(int(seed))
        self.head = head = self._build_head(int(train_arrays["f"].shape[1])).to(self.device)
        fit_log = fit_module(head, self._nll, train_arrays, val_arrays, self.train_cfg, seed=int(seed),
                             device=self.device, verbose=verbose)  # fmt: skip
        head.eval()
        del train_arrays, val_arrays
        log: dict[str, Any] = {"fit": fit_log, "mean_from": self.source}
        if val is not None and len(val) > 0:
            dist = self.predict_dist(val.X)
            sd = np.sqrt(dist.var())
            log["sigma_val"] = {"mean": float(sd.mean()), "q05": float(np.quantile(sd, 0.05)),
                                "q95": float(np.quantile(sd, 0.95)), "const_sigma": self.source.get("sigma_const")}  # fmt: skip
        return finish_fit_log(self, log, train, val, started)

    # ---------------------------------------------------------------- prediction
    @torch.no_grad()
    def predict_params(self, state_history: np.ndarray | torch.Tensor, batch_size: int = 8192) -> tuple[np.ndarray, np.ndarray]:
        """(N,) float64 mean and sigma [m/s^2] of the one-step law."""
        mean, _ = self._require()
        x = history_rows(state_history)
        mus, sds = [], []
        step = max(1, int(batch_size))
        for a in range(0, x.shape[0], step):
            hist = as_history_tensor(x[a : a + step], self.device)
            mu_n, h = self._frozen(hist)
            mus.append(mean.normalizer.denorm_mu(mu_n).double().cpu())
            sds.append(mean.normalizer.denorm_sigma(self._log_sigma(h).exp()).double().cpu())
        if not mus:
            return np.empty(0), np.empty(0)
        return torch.cat(mus).numpy(), torch.cat(sds).numpy()

    def predict_dist(
        self,
        state_history: np.ndarray | torch.Tensor,
        *,
        v_lead_future: np.ndarray | torch.Tensor | None = None,
        batch_size: int = 8192,
    ) -> GaussianDist:
        """``GaussianDist(mu_frozen, sigma(x))``; ``aleatoric_var = sigma^2``, ``epistemic_var = 0``."""
        mu, sd = self.predict_params(state_history, batch_size)
        return GaussianDist(mu, sd, aleatoric_var=sd**2, epistemic_var=0.0)

    @torch.no_grad()
    def sample_step(self, hist: torch.Tensor, state: Any, generator: torch.Generator) -> torch.Tensor:
        mean, _ = self._require()
        mu_n, h = self._frozen(as_history_tensor(hist, self.device))
        mu = mean.normalizer.denorm_mu(mu_n)
        sd = mean.normalizer.denorm_sigma(self._log_sigma(h).exp())
        return mu + sd * torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)

    # --------------------------------------------------------------- persistence
    def get_state(self) -> dict:
        mean, head = self._require()
        return {"mean_kwargs": dict(mean.init_kwargs), "mean": mean.get_state(), "head": _state_to_cpu(head),
                "source": dict(self.source or {})}  # fmt: skip

    def set_state(self, state: dict) -> None:
        mean = LSTMConstSigma(**state["mean_kwargs"])
        mean.set_state(state["mean"])
        mean.fitted = True
        mean.require_net().requires_grad_(False)
        self.mean_model = mean.to(self.device)
        with torch.random.fork_rng(devices=[]):  # the throw-away initial weights leave the global RNG alone
            head = self._build_head(int(mean.net_spec["hidden"]))
        head.load_state_dict(state["head"])
        self.head = head.to(self.device).eval()
        self.source = dict(state.get("source") or {})
        self.fitted = True
