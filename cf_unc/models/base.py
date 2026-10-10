"""Common interface of predictive car-following models (docs/design.md, section 7)."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import numpy as np
import torch

if TYPE_CHECKING:
    from cf_unc.data.windows import WindowSet
    from cf_unc.dist import PredDist

SAMPLER_MODES = ("per_step", "per_trajectory")

MODEL_REGISTRY: dict[str, type["CFModel"]] = {}


def register(name: str) -> Callable[[type["CFModel"]], type["CFModel"]]:
    """Class decorator: make the model constructible by name (configs, checkpoints)."""

    def wrap(cls: type["CFModel"]) -> type["CFModel"]:
        if name in MODEL_REGISTRY and MODEL_REGISTRY[name] is not cls:
            raise ValueError(f"model name {name!r} is already registered")
        cls.name = name
        MODEL_REGISTRY[name] = cls
        return cls

    return wrap


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class CFModel(abc.ABC):
    """Predictive model of the follower acceleration a[t] given the state history.

    Subclasses pass every constructor argument to ``super().__init__(**kwargs)``;
    the arguments must be JSON-like (numbers, strings, bools, lists, dicts) so
    that ``save`` / ``load`` can rebuild the object.
    """

    name: ClassVar[str] = "base"
    uses_history: ClassVar[bool] = True  # False: only the last sample of the history is used
    needs_leader_future: ClassVar[bool] = False  # True: DiffusionCF (predict_dist / init_sampler take the leader's future)
    # True: a sampler of 5 s plans (DiffusionCF, DiffusionCFHist): E1 evaluates it on the sampled rows
    # (``eval.max_rows_sampled``), E2 / E3 roll it out ``n_rollouts_sampled`` times and it follows its plan step by
    # step (hold 1). Separate from ``needs_leader_future``, which only says whether the leader's future is an input.
    plan_sampler: ClassVar[bool] = False
    has_epistemic: ClassVar[bool] = False  # True: predict_dist fills epistemic_var

    def __init__(self, **init_kwargs: Any) -> None:
        self.init_kwargs: dict[str, Any] = dict(init_kwargs)
        self.device: torch.device = torch.device("cpu")
        self.fitted: bool = False

    # ------------------------------------------------------------------ training
    @abc.abstractmethod
    def fit(self, train: "WindowSet", val: "WindowSet | None" = None, *, seed: int = 0, verbose: bool = False) -> dict:
        """Train on windows; returns a JSON-serialisable training log."""

    # ---------------------------------------------------------------- prediction
    @abc.abstractmethod
    def predict_dist(
        self,
        state_history: np.ndarray | torch.Tensor,
        *,
        v_lead_future: np.ndarray | torch.Tensor | None = None,
        batch_size: int = 8192,
    ) -> "PredDist":
        """Predictive law of a[t] for histories of shape (N, H, 3), physical units."""

    # ------------------------------------------------------------------ sampling
    def init_sampler(self, n_traj: int, generator: torch.Generator, mode: str = "per_step") -> Any:
        """State that is frozen along each of ``n_traj`` trajectories (None if nothing is frozen)."""
        if mode not in SAMPLER_MODES:
            raise ValueError(f"unknown sampler mode {mode!r}")
        return None

    @abc.abstractmethod
    def sample_step(self, hist: torch.Tensor, state: Any, generator: torch.Generator) -> torch.Tensor:
        """One acceleration draw per trajectory.

        hist (B, H, 3) float32 on ``self.device``; returns (B,) in m/s^2, not clipped.
        """

    # --------------------------------------------------------------- persistence
    @abc.abstractmethod
    def get_state(self) -> dict:
        """Tensors and plain values that restore a fitted model given ``init_kwargs``."""

    @abc.abstractmethod
    def set_state(self, state: dict) -> None:
        """Inverse of ``get_state`` (must also build the networks)."""

    def to(self, device: str | torch.device) -> "CFModel":
        self.device = resolve_device(device)
        for value in vars(self).values():
            if isinstance(value, torch.nn.Module):
                value.to(self.device)
        return self

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"name": self.name, "init_kwargs": self.init_kwargs, "state": self.get_state()}
        with open(path, "wb") as fh:
            torch.save(payload, fh)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "CFModel":
        with open(Path(path), "rb") as fh:
            payload = torch.load(fh, map_location="cpu", weights_only=True)
        model_cls = MODEL_REGISTRY[payload["name"]] if cls is CFModel else cls
        model = model_cls(**payload["init_kwargs"])
        model.set_state(payload["state"])
        model.fitted = True
        return model.to(device)


def as_history_tensor(state_history: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """(N, H, 3) float32 tensor on ``device`` (a single (H, 3) history gets a batch axis)."""
    x = torch.as_tensor(np.asarray(state_history) if not torch.is_tensor(state_history) else state_history)
    x = x.to(dtype=torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"state history must have shape (N, H, 3), got {tuple(x.shape)}")
    return x.to(device)
