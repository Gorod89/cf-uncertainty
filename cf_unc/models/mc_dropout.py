"""MC dropout: a HeteroMDN with dropout kept active at inference (docs/design.md, section 7).

Dropout acts after the LSTM state and after the hidden head layer (``SeqRegressor``); the LSTM
encoding is therefore shared by all dropout networks and computed once per history.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from cf_unc.data.windows import WindowSet
from cf_unc.dist import MixtureDist
from cf_unc.models.base import CFModel, as_history_tensor, register, resolve_device
from cf_unc.models.hetero_mdn import HeteroMDN, Masks, Params, draw_mixture, finish_fit_log

__all__ = ["MCDropout"]


@register("mc_dropout")
class MCDropout(CFModel):
    """HeteroMDN with dropout probability ``p``, trained normally; epistemic part from dropout masks.

    ``predict_dist``: pooled mixture of ``n_passes`` dropout networks; pass ``j`` uses ONE pair of
    masks shared by all inputs. The mask pairs come from ``predict_seed`` alone, so the law does not
    depend on the batch size or on the order of the rows.
    """

    has_epistemic = True

    def __init__(
        self,
        p: float = 0.1,
        n_passes: int = 50,
        n_components: int = 1,
        hidden: int = 64,
        head_hidden: int = 64,
        log_sigma_min: float = -7.0,
        log_sigma_max: float = 3.0,
        predict_seed: int = 0,
        train: dict | None = None,
    ) -> None:
        super().__init__(
            p=p,
            n_passes=n_passes,
            n_components=n_components,
            hidden=hidden,
            head_hidden=head_hidden,
            log_sigma_min=log_sigma_min,
            log_sigma_max=log_sigma_max,
            predict_seed=predict_seed,
            train=train,
        )
        if not 0.0 < p < 1.0:
            raise ValueError("MC dropout needs 0 < p < 1")
        if int(n_passes) < 1:
            raise ValueError("n_passes must be >= 1")
        self.n_passes = int(n_passes)
        self.predict_seed = int(predict_seed)
        self.mdn = HeteroMDN(
            n_components=n_components,
            hidden=hidden,
            head_hidden=head_hidden,
            dropout=p,
            log_sigma_min=log_sigma_min,
            log_sigma_max=log_sigma_max,
            train=train,
        )
        self._pass_masks: Masks | None = None

    def to(self, device: str | torch.device) -> "MCDropout":
        self.device = resolve_device(device)
        self.mdn.to(self.device)
        self._pass_masks = None
        return self

    # ------------------------------------------------------------------ training
    def fit(self, train: WindowSet, val: WindowSet | None = None, *, seed: int = 0, verbose: bool = False) -> dict:
        """Train the dropout network; ``val_nll_no_dropout`` is the NLL of the deterministic network."""
        started = time.perf_counter()
        mdn_log = self.mdn.fit(train, val, seed=seed, verbose=verbose)
        self._pass_masks = None
        log = {"fit": mdn_log["fit"], "val_nll_no_dropout": mdn_log["val_nll"]}
        return finish_fit_log(self, log, train, val, started)

    # ---------------------------------------------------------------- prediction
    def pass_masks(self) -> Masks:
        """Masks of the ``n_passes`` networks of ``predict_dist``: (P, 1, hidden), (P, 1, head_hidden).

        Drawn on the CPU from ``predict_seed`` (identical on every device), scaled by 1 / (1 - p).
        """
        if self._pass_masks is None:
            net = self.mdn.require_net()
            gen = torch.Generator(device="cpu").manual_seed(self.predict_seed)
            cpu = torch.device("cpu")
            m1 = net.drop1.sample_mask(self.n_passes, net.hidden, gen, cpu)
            m2 = net.drop2.sample_mask(self.n_passes, net.head_hidden, gen, cpu)
            self._pass_masks = (m1[:, None].to(self.device), m2[:, None].to(self.device))
        return self._pass_masks

    def predict_passes(
        self, state_history: np.ndarray | torch.Tensor, batch_size: int = 8192
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Weights, means, sigmas of every dropout network, each float64 (P, N, K), physical units.

        One LSTM pass per history; ``batch_size`` bounds the rows times networks per chunk.
        """
        return self.mdn.predict_params_np(state_history, batch_size, masks=self.pass_masks())

    def predict_dist(
        self,
        state_history: np.ndarray | torch.Tensor,
        *,
        v_lead_future: np.ndarray | torch.Tensor | None = None,
        batch_size: int = 8192,
    ) -> MixtureDist:
        """Pooled mixture of the ``n_passes`` dropout networks with the decomposition attached."""
        return MixtureDist.from_members(*self.predict_passes(state_history, batch_size))

    # ------------------------------------------------------------------ sampling
    def init_sampler(self, n_traj: int, generator: torch.Generator, mode: str = "per_step") -> Masks | None:
        """None for ``"per_step"``; for ``"per_trajectory"`` one mask pair per trajectory, (n_traj, width)."""
        super().init_sampler(n_traj, generator, mode)
        if mode == "per_step":
            return None
        return self.mdn.require_net().sample_masks(int(n_traj), generator)

    @torch.no_grad()
    def step_params(self, hist: torch.Tensor, state: Masks | None, generator: torch.Generator) -> Params:
        """Parameters (B, K) of the dropout network of every trajectory at this step.

        ``state`` None (per step): independent fresh masks per trajectory, else the frozen masks.
        """
        net = self.mdn.require_net().eval()
        if state is None:
            masks = net.sample_masks(hist.shape[0], generator)
        else:
            masks = state
            if masks[0].shape[0] != hist.shape[0]:
                raise ValueError(f"sampler state holds masks of {masks[0].shape[0]} trajectories, got {hist.shape[0]}")
        return self.mdn.predict_params(hist, masks)

    @torch.no_grad()
    def sample_step(self, hist: torch.Tensor, state: Any, generator: torch.Generator) -> torch.Tensor:
        return draw_mixture(*self.step_params(as_history_tensor(hist, self.device), state, generator), generator)

    # --------------------------------------------------------------- persistence
    def get_state(self) -> dict:
        return {"mdn": self.mdn.get_state()}

    def set_state(self, state: dict) -> None:
        self.mdn.set_state(state["mdn"])
        self._pass_masks = None
        self.fitted = True
