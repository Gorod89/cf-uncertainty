"""Deep ensemble of HeteroMDN members trained on driver-bootstrap resamples (docs/design.md, section 7).

Predictive law: the pooled (uniform) mixture of the members with the law-of-total-variance
decomposition attached (``aleatoric_var`` = mean member variance, ``epistemic_var`` = variance of
the member means).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from cf_unc.data.splits import bootstrap_drivers
from cf_unc.data.windows import WindowSet
from cf_unc.dist import MixtureDist
from cf_unc.models.base import CFModel, as_history_tensor, register, resolve_device
from cf_unc.models.hetero_mdn import HeteroMDN, Params, draw_mixture, finish_fit_log, history_rows

__all__ = ["DeepEnsemble"]


@register("deep_ensemble")
class DeepEnsemble(CFModel):
    """``n_members`` HeteroMDN networks; member ``m`` uses seed ``seed * 1000 + m``.

    member: HeteroMDN keyword arguments. train: TrainConfig overrides applied to every member; keys
    of ``member["train"]`` take precedence over them. With ``bootstrap`` each member is trained on a
    with-replacement resample of the training drivers (``bootstrap_drivers`` with
    ``np.random.default_rng(seed * 1000 + m)``); all members share the validation windows.
    """

    has_epistemic = True

    def __init__(
        self, n_members: int = 5, bootstrap: bool = True, member: dict | None = None, train: dict | None = None
    ) -> None:
        super().__init__(n_members=n_members, bootstrap=bootstrap, member=member, train=train)
        if int(n_members) < 1:
            raise ValueError("n_members must be >= 1")
        member_kwargs = dict(member or {})
        member_kwargs["train"] = {**(train or {}), **(member_kwargs.get("train") or {})}
        self.bootstrap = bool(bootstrap)
        self.members = [HeteroMDN(**member_kwargs) for _ in range(int(n_members))]

    @property
    def n_members(self) -> int:
        return len(self.members)

    def to(self, device: str | torch.device) -> "DeepEnsemble":
        self.device = resolve_device(device)
        for member in self.members:
            member.to(self.device)
        return self

    # ------------------------------------------------------------------ training
    def fit(self, train: WindowSet, val: WindowSet | None = None, *, seed: int = 0, verbose: bool = False) -> dict:
        """Train the members one after the other; the log holds every member log and data size."""
        started = time.perf_counter()
        member_logs, seeds, n_drivers = [], [], []
        for m, member in enumerate(self.members):
            member_seed = int(seed) * 1000 + m
            data = train
            if self.bootstrap:
                data = train.subset(bootstrap_drivers(train.driver, np.random.default_rng(member_seed)))
            if verbose:
                print(f"member {m}: seed {member_seed}, {len(data)} windows")
            member_logs.append(member.fit(data, val, seed=member_seed, verbose=verbose))
            seeds.append(member_seed)
            n_drivers.append(int(np.unique(data.driver).size))
        log = {"members": member_logs, "member_seeds": seeds, "n_drivers": n_drivers}
        return finish_fit_log(self, log, train, val, started)

    # ---------------------------------------------------------------- prediction
    def predict_members(
        self, state_history: np.ndarray | torch.Tensor, batch_size: int = 8192
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Member mixture weights, means, sigmas, each float64 (M, N, K), physical units."""
        x = history_rows(state_history)
        params = [member.predict_params_np(x, batch_size) for member in self.members]
        return tuple(np.stack([p[i] for p in params]) for i in range(3))  # type: ignore[return-value]

    def predict_dist(
        self,
        state_history: np.ndarray | torch.Tensor,
        *,
        v_lead_future: np.ndarray | torch.Tensor | None = None,
        batch_size: int = 8192,
    ) -> MixtureDist:
        """Pooled mixture of the members (M * K components) with the decomposition attached."""
        return MixtureDist.from_members(*self.predict_members(state_history, batch_size))

    # ------------------------------------------------------------------ sampling
    def init_sampler(self, n_traj: int, generator: torch.Generator, mode: str = "per_step") -> torch.Tensor | None:
        """None for ``"per_step"``; for ``"per_trajectory"`` the frozen member index of each trajectory (n_traj,)."""
        super().init_sampler(n_traj, generator, mode)
        if mode == "per_step":
            return None
        return torch.randint(self.n_members, (int(n_traj),), generator=generator, device=self.device)

    @torch.no_grad()
    def step_params(self, hist: torch.Tensor, state: torch.Tensor | None, generator: torch.Generator) -> Params:
        """Parameters (B, K) of the member each trajectory draws from at this step.

        ``state`` None (per step): a fresh uniform member index per trajectory. Every member is
        evaluated only on the trajectories assigned to it.
        """
        b = hist.shape[0]
        if state is None:
            idx = torch.randint(self.n_members, (b,), generator=generator, device=hist.device)
        else:
            idx = state
            if idx.shape != (b,):
                raise ValueError(f"sampler state holds {tuple(idx.shape)} member indices for {b} trajectories")
        out: list[torch.Tensor] | None = None
        for m, member in enumerate(self.members):
            rows = torch.nonzero(idx == m).squeeze(1)
            if rows.numel() == 0:
                continue
            params = member.step_params(hist[rows], None, generator)
            if out is None:
                k = params[0].shape[-1]
                out = [torch.empty((b, k), dtype=p.dtype, device=p.device) for p in params]
            for o, p in zip(out, params):
                o[rows] = p
        if out is None:  # b == 0
            return self.members[0].step_params(hist, None, generator)
        return out[0], out[1], out[2]

    @torch.no_grad()
    def sample_step(self, hist: torch.Tensor, state: Any, generator: torch.Generator) -> torch.Tensor:
        return draw_mixture(*self.step_params(as_history_tensor(hist, self.device), state, generator), generator)

    # --------------------------------------------------------------- persistence
    def get_state(self) -> dict:
        return {"members": [member.get_state() for member in self.members]}

    def set_state(self, state: dict) -> None:
        if len(state["members"]) != self.n_members:
            raise ValueError(f"checkpoint holds {len(state['members'])} members, expected {self.n_members}")
        for member, member_state in zip(self.members, state["members"]):
            member.set_state(member_state)
        self.fitted = True
