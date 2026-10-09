"""Driving regimes of hypothesis H3.3 (docs/design.md, section 10.5).

Definitions of cf-identifiability (``cf_ident/diag/regimes.py``) plus congestion, with ``dv = v - v_lead``
(m/s, positive when closing in), ``a_prev = a[t-1]`` (m/s^2) and ``v0`` = free speed of the IDM calibrated
on the highD training events (the same number for every data set)::

    congested    s < 15 and v < 8
    braking      a_prev < -0.5
    approaching  dv > 0.5
    free         s > 60  or  (|dv| < 0.5 and v > 0.8 v0)
    steady       |dv| < 0.5 and 10 < s < 60
    other        everything else

The masks overlap; :func:`label_regimes` turns them into a partition by priority, first match wins
(default ``congested > braking > approaching > free > steady``). ``braking`` uses the PREVIOUS
acceleration: a regime label must be known before the outcome, otherwise conditioning on it truncates
the target ``a[t]`` and biases NLL and residual variance of that regime. Comparisons with NaN are
false, so a NaN ``a_prev`` is never ``braking`` and a window with a NaN state ends in ``other``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["REGIMES", "PRIORITY", "RegimeThresholds", "regime_masks", "label_regimes", "regime_shares"]

REGIMES = ("congested", "braking", "approaching", "free", "steady", "other")
PRIORITY = ("congested", "braking", "approaching", "free", "steady")
OTHER = REGIMES.index("other")


@dataclass(frozen=True)
class RegimeThresholds:
    """Thresholds of the regime definitions (``regimes`` section of ``configs/maps/default.yaml``)."""

    s_congested: float = 15.0  # m
    v_congested: float = 8.0  # m/s
    s_free: float = 60.0  # m
    s_lo: float = 10.0  # m
    dv_eps: float = 0.5  # m/s
    a_brake: float = -0.5  # m/s^2, applied to a[t-1]
    v_free_frac: float = 0.8

    @classmethod
    def from_config(cls, mapping: Mapping[str, Any] | None = None) -> "RegimeThresholds":
        """From the ``regimes`` section or a whole maps config; other keys (v0, priority, ...) are ignored."""
        if mapping is None:
            return cls()
        m = dict(mapping)
        if isinstance(m.get("regimes"), Mapping):
            m = dict(m["regimes"])
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in m.items() if k in names})


def _arrays(*values: ArrayLike) -> list[np.ndarray]:
    arrays = [np.asarray(v, dtype=np.float64) for v in values]
    shape = np.broadcast_shapes(*(a.shape for a in arrays))
    return [np.broadcast_to(a, shape) for a in arrays]


def regime_masks(
    s: ArrayLike, dv: ArrayLike, v: ArrayLike, a_prev: ArrayLike, v0: float, th: RegimeThresholds = RegimeThresholds()
) -> dict[str, np.ndarray]:
    """Overlapping boolean masks (n,) of the regimes, exactly the definitions above.

    s (m), dv (m/s), v (m/s), a_prev (m/s^2): (n,) arrays; v0: IDM free speed (m/s). ``other`` is the
    complement of the union of the five named regimes.
    """
    if v0 is None or not np.isfinite(float(v0)):
        raise ValueError("v0 (free speed of the calibrated IDM) is required")
    s_a, dv_a, v_a, a_a = _arrays(s, dv, v, a_prev)
    calm = np.abs(dv_a) < th.dv_eps
    masks = {
        "congested": (s_a < th.s_congested) & (v_a < th.v_congested),
        "braking": a_a < th.a_brake,
        "approaching": dv_a > th.dv_eps,
        "free": (s_a > th.s_free) | (calm & (v_a > th.v_free_frac * float(v0))),
        "steady": calm & (s_a > th.s_lo) & (s_a < th.s_free),
    }
    union = np.zeros(s_a.shape, dtype=bool)
    for mask in masks.values():
        union |= mask
    masks["other"] = ~union
    return masks


def label_regimes(
    s: ArrayLike,
    dv: ArrayLike,
    v: ArrayLike,
    a_prev: ArrayLike,
    v0: float,
    th: RegimeThresholds = RegimeThresholds(),
    priority: Sequence[str] = PRIORITY,
) -> np.ndarray:
    """Partition labels (n,) int64, indices into ``REGIMES``: the first regime of ``priority`` whose mask
    holds, ``other`` when none does (regimes left out of ``priority`` are never assigned)."""
    names = [str(p) for p in priority]
    unknown = [p for p in names if p not in PRIORITY]
    if unknown or len(set(names)) != len(names):
        raise ValueError(f"priority must be distinct names out of {PRIORITY}, got {names}")
    masks = regime_masks(s, dv, v, a_prev, v0, th)
    labels = np.full(masks["other"].shape, OTHER, dtype=np.int64)
    assigned = np.zeros(labels.shape, dtype=bool)
    for name in names:
        hit = masks[name] & ~assigned
        labels[hit] = REGIMES.index(name)
        assigned |= hit
    return labels


def regime_shares(labels: ArrayLike, n_regimes: int = len(REGIMES)) -> np.ndarray:
    """Share of windows per regime, (n_regimes,) float64 in ``REGIMES`` order (zeros for no windows)."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64).ravel(), minlength=n_regimes)[:n_regimes]
    total = counts.sum()
    return counts / total if total > 0 else np.zeros(n_regimes)
