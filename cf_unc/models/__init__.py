"""Model zoo. Importing the package registers every available model family."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

from cf_unc.models.base import MODEL_REGISTRY, SAMPLER_MODES, CFModel, register, resolve_device

_MODULES = (
    "hetero_mdn",
    "deep_ensemble",
    "mc_dropout",
    "baselines",
    "gp_residual_idm",
    "quantile_pidl",
    "diffusion_cf",
    "lstm_hetero_frozen",
)

for _name in _MODULES:
    try:
        importlib.import_module(f"cf_unc.models.{_name}")
    except ModuleNotFoundError as err:  # a family that is not implemented yet
        if err.name != f"cf_unc.models.{_name}":
            raise


def build_model(name: str, **kwargs: Any) -> CFModel:
    """Construct a registered model by name (keys of ``MODEL_REGISTRY``)."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def constructor_accepts(name: str, argument: str) -> bool:
    """Whether the constructor of the registered model ``name`` has a parameter ``argument``."""
    return argument in inspect.signature(MODEL_REGISTRY[name].__init__).parameters


def load_model(path: str | Path, device: str = "cpu") -> CFModel:
    """Load any checkpoint written by ``CFModel.save``."""
    return CFModel.load(path, device=device)


__all__ = [
    "CFModel",
    "MODEL_REGISTRY",
    "SAMPLER_MODES",
    "build_model",
    "constructor_accepts",
    "load_model",
    "register",
    "resolve_device",
]
