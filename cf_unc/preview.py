"""Information budget of the leader's future handed to a plan sampler (extension item X1, extension/docs/PLAN.md).

A ``PreviewSpec`` describes the preview of the predecessor's speeds that a plan sampler (DiffusionCF) receives at
every re-plan of a rollout: truncated to a horizon, noisy, or stale. It is applied at inference only (stage A of
X1: stored checkpoints, a truncated preview of a model trained on exact ones). Stage B, a model trained at a
budget, is the constructor argument ``preview_steps`` of DiffusionCF. ``None`` everywhere is the stored behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from cf_unc.constants import DT

__all__ = ["PreviewSpec", "preview_steps"]

_TOL = 1e-9


def preview_steps(seconds: float, name: str) -> int:
    """Non-negative number of DT steps of ``seconds`` (ValueError if not a multiple of DT)."""
    k = float(seconds) / DT
    if not (math.isfinite(k) and k >= -_TOL and abs(k - round(k)) <= _TOL * max(1.0, abs(k))):
        raise ValueError(f"{name}: {seconds} s is not a non-negative multiple of DT = {DT} s")
    return int(round(k))


@dataclass(frozen=True)
class PreviewSpec:
    """Preview of the leader's future at a re-plan of rollout step k (leader speeds ``v_lead[:, j]`` at step j).

    horizon_s    preview horizon [s]: the sampler sees the speeds up to ``horizon_s`` ahead of the preview start and
                 the last seen value held constant for the rest of its plan; None: the whole plan (the stored
                 behaviour); 0: the constant-leader assumption (the current speed held)
    noise_sigma  standard deviation [m/s] of i.i.d. Gaussian noise added to every preview sample (0: none); drawn
                 from the rollout's own stream, so a cell with noise 0 draws nothing extra
    delay_s      staleness [s]: the preview handed at step k starts at step ``k + 1 - delay_s / DT`` (0: current)
    """

    horizon_s: float | None = None
    noise_sigma: float = 0.0
    delay_s: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon_s is not None:
            object.__setattr__(self, "horizon_s", float(self.horizon_s))
            preview_steps(self.horizon_s, "horizon_s")
        object.__setattr__(self, "noise_sigma", float(self.noise_sigma))
        object.__setattr__(self, "delay_s", float(self.delay_s))
        if not (math.isfinite(self.noise_sigma) and self.noise_sigma >= 0.0):
            raise ValueError(f"noise_sigma must be a non-negative number of m/s, got {self.noise_sigma!r}")
        preview_steps(self.delay_s, "delay_s")

    @property
    def horizon_steps(self) -> int | None:
        """Preview horizon in steps (None: unlimited)."""
        return None if self.horizon_s is None else preview_steps(self.horizon_s, "horizon_s")

    @property
    def delay_steps(self) -> int:
        """Staleness in steps."""
        return preview_steps(self.delay_s, "delay_s")

    @property
    def identity(self) -> bool:
        """Whether the spec leaves the preview as it is (unlimited, clean, current)."""
        return self.horizon_s is None and self.noise_sigma == 0.0 and self.delay_s == 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"horizon_s": self.horizon_s, "noise_sigma": self.noise_sigma, "delay_s": self.delay_s}

    @classmethod
    def from_any(cls, value: Any) -> PreviewSpec | None:
        """None -> None; a PreviewSpec -> itself; a mapping with keys of the fields -> PreviewSpec (ValueError
        for any other key or value)."""
        if value is None or isinstance(value, PreviewSpec):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(f"preview: expected a mapping with horizon_s / noise_sigma / delay_s, got {value!r}")
        unknown = sorted(set(map(str, value)) - {"horizon_s", "noise_sigma", "delay_s"})
        if unknown:
            raise ValueError(f"preview: unknown keys {unknown}; known: horizon_s, noise_sigma, delay_s")
        return cls(**{str(k): v for k, v in value.items()})
