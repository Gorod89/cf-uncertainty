"""Figures of experiment E2 (docs/design.md, section 11): coverage and width of the rollout intervals by horizon.

Every figure has a table twin under ``<root>/tables`` (``coverage_by_horizon.csv``). Style, colours and the save
helper come from :mod:`cf_unc.report.figures`: colour carries one job, the identity of two series in a fixed order
(raw rollouts = blue, ACI online = orange); text is always ink; one y axis per panel; solid hairline grid.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cf_unc.report.figures import (
    BLUE,
    DATASET_LABEL,
    GRID,
    INK_2,
    MODEL_LABEL,
    ORANGE,
    SURFACE,
    _plain_log_axis,
    _save,
    apply_style,
    label,
)

__all__ = ["DATASET_LABEL_E2", "SERIES", "make_e2_figures", "plot_coverage_by_horizon", "plot_width_by_horizon"]

DATASET_LABEL_E2 = {**DATASET_LABEL, "ngsim_us101": "NGSIM US-101 (reconstructed)"}
TARGET_LABEL = {"a": "one-step acceleration target", "mean_1s": "1 s mean acceleration target"}
SERIES = (("raw", BLUE, "raw rollouts [q05, q95]"), ("aci_online", ORANGE, "ACI online"))  # variant, colour, legend
_DODGE = 0.18  # s, horizontal offset of the two series at a horizon
_TITLE_IN, _SUBTITLE_IN, _HEADER_IN, _LEGEND_IN = 0.08, 0.36, 0.5, 0.42  # inches: title / subtitle tops, bands


def _models(sel: pd.DataFrame, models: Sequence[str] | None) -> list[str]:
    present = list(dict.fromkeys(sel["model"].astype(str)))
    if models is not None:
        return [m for m in models if m in present]
    known = [m for m in MODEL_LABEL if m in present]
    return known + sorted(m for m in present if m not in known)


def _grid(n: int) -> tuple[int, int]:
    n_col = min(5, n)
    return int(math.ceil(n / n_col)), n_col


def _series(ax: plt.Axes, rows: pd.DataFrame, colour: str, offset: float, column: str, whiskers: bool) -> None:
    rows = rows.sort_values("horizon_s")
    x = rows["horizon_s"].to_numpy(float) + offset
    y = rows[column].to_numpy(float)
    if whiskers:  # not clipped: values at 0 or 1 stay fully visible
        ax.vlines(x, rows["picp_lo"].to_numpy(float), rows["picp_hi"].to_numpy(float), color=colour, lw=1.6, zorder=2,
                  clip_on=False)  # fmt: skip
    ax.plot(x, y, color=colour, lw=1.6, zorder=3, solid_capstyle="round", clip_on=False)
    ax.plot(x, y, ls="none", marker="o", ms=5.5, mfc=colour, mec=SURFACE, mew=1.2, zorder=4, clip_on=False)


def _finish(fig: plt.Figure, title: str, subtitle: str, handles: list, ncol: int) -> None:
    """Title and subtitle at fixed distances (inches) from the top, legend in a band at the bottom, tight layout."""
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - _TITLE_IN / h, ha="left", va="top", fontsize=12, fontweight="bold")
    fig.text(0.01, 1 - _SUBTITLE_IN / h, subtitle, fontsize=8, color=INK_2, ha="left", va="top")
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=ncol, fontsize=8.5, handlelength=1.8)
    fig.tight_layout(rect=(0, _LEGEND_IN / h, 1, 1 - _HEADER_IN / h))


def _layout(n: int, height: float) -> tuple[plt.Figure, np.ndarray]:
    n_row, n_col = _grid(n)
    fig, axes = plt.subplots(n_row, n_col, figsize=(2.45 * n_col + 0.5, height * n_row + 1.25), sharex=True,
                             sharey=True, squeeze=False)  # fmt: skip
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return fig, axes


def _label_x(axes: np.ndarray) -> None:
    """x label and tick labels on the lowest visible panel of every column (rows may be incomplete)."""
    n_row, n_col = axes.shape
    for c in range(n_col):
        rows = [r for r in range(n_row) if axes[r, c].get_visible()]
        if rows:
            axes[rows[-1], c].tick_params(labelbottom=True)
            axes[rows[-1], c].set_xlabel("horizon, s")


def _x_axis(ax: plt.Axes, horizons: np.ndarray) -> None:
    ax.set_xticks(horizons)
    ax.set_xticklabels([f"{h:g}" for h in horizons])
    ax.set_xlim(horizons.min() - 0.9, horizons.max() + 0.9)
    ax.grid(axis="x", visible=False)


def plot_coverage_by_horizon(
    coverage: pd.DataFrame,
    target: str,
    dataset: str,
    path: Path,
    models: Sequence[str] | None = None,
    nominal: float = 0.90,
    band: tuple[float, float] = (0.87, 0.93),
) -> Path | None:
    """PICP of the 90 % gap interval against the horizon, one panel per model: raw and ACI online with whiskers."""
    sel = coverage[(coverage["target"] == target) & (coverage["dataset"] == dataset)]
    names = _models(sel, models)
    if not names:
        return None
    apply_style()
    horizons = np.sort(sel["horizon_s"].unique().astype(float))
    fig, axes = _layout(len(names), 2.05)
    for ax, model in zip(axes.ravel(), names):
        ax.axhspan(band[0], band[1], color=GRID, alpha=0.7, lw=0, zorder=0)
        ax.axhline(nominal, color=INK_2, lw=0.9, zorder=1)
        for i, (variant, colour, _) in enumerate(SERIES):
            rows = sel[(sel["model"] == model) & (sel["variant"] == variant)]
            if not rows.empty:
                _series(ax, rows, colour, (2 * i - 1) * _DODGE, "picp", whiskers=True)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        _x_axis(ax, horizons)
        ax.set_title(label(model, MODEL_LABEL), fontsize=9.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("coverage (PICP)")
    _label_x(axes)
    handles = [
        Line2D([], [], color=BLUE, marker="o", ms=5.5, mec=SURFACE, lw=1.6, label=SERIES[0][2]),
        Line2D([], [], color=ORANGE, marker="o", ms=5.5, mec=SURFACE, lw=1.6, label=SERIES[1][2]),
        Line2D([], [], color=INK_2, lw=0.9, label=f"nominal {nominal:.2f}"),
        Patch(facecolor=GRID, alpha=0.7, lw=0, label=f"{band[0]:.2f} to {band[1]:.2f}"),
    ]
    _finish(
        fig, f"Coverage of the 90 % gap interval by horizon: {label(dataset, DATASET_LABEL_E2)}",
        f"Closed-loop rollouts, {TARGET_LABEL.get(target, target)}. Whiskers: 95 % event-bootstrap interval "
        "(seeds averaged per event); ACI calibrated on highD validation, then updated online.",
        handles, ncol=4,
    )  # fmt: skip
    return _save(fig, path)


def plot_width_by_horizon(
    coverage: pd.DataFrame, target: str, dataset: str, path: Path, models: Sequence[str] | None = None
) -> Path | None:
    """Mean width [m] of the raw and of the ACI-corrected 90 % gap interval against the horizon (log scale): the
    mean over the finite intervals (``width_finite_mean``; an ACI interval at a level >= 1 is infinite) when the table
    has it, else ``width_mean``."""
    sel = coverage[(coverage["target"] == target) & (coverage["dataset"] == dataset)]
    column = "width_finite_mean" if "width_finite_mean" in sel.columns else "width_mean"
    sel = sel.assign(width_mean=sel[column])
    sel = sel[np.isfinite(sel["width_mean"].to_numpy(float)) & (sel["width_mean"].to_numpy(float) > 0)]
    names = _models(sel, models)
    if not names:
        return None
    apply_style()
    horizons = np.sort(sel["horizon_s"].unique().astype(float))
    fig, axes = _layout(len(names), 2.05)
    for ax, model in zip(axes.ravel(), names):
        for i, (variant, colour, _) in enumerate(SERIES):
            rows = sel[(sel["model"] == model) & (sel["variant"] == variant)]
            if not rows.empty:
                _series(ax, rows, colour, (2 * i - 1) * _DODGE, "width_mean", whiskers=False)
        ax.set_yscale("log")
        _x_axis(ax, horizons)
        ax.set_title(label(model, MODEL_LABEL), fontsize=9.5)
    lo, hi = sel["width_mean"].min(), sel["width_mean"].max()
    axes[0, 0].set_ylim(lo / 1.6, hi * 1.6)
    _plain_log_axis(axes[0, 0].yaxis)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean width, m")
    _label_x(axes)
    handles = [
        Line2D([], [], color=BLUE, marker="o", ms=5.5, mec=SURFACE, lw=1.6, label=SERIES[0][2]),
        Line2D([], [], color=ORANGE, marker="o", ms=5.5, mec=SURFACE, lw=1.6, label=SERIES[1][2]),
    ]
    _finish(
        fig, f"Width of the 90 % gap interval by horizon: {label(dataset, DATASET_LABEL_E2)}",
        f"Closed-loop rollouts, {TARGET_LABEL.get(target, target)}. Mean over events of the seed-averaged width of "
        "the finite intervals; log scale shared by all panels.",
        handles, ncol=2,
    )  # fmt: skip
    return _save(fig, path)


def make_e2_figures(
    root: str | Path, in_domain: str = "highd", calibration: str = "highd_val", models: Sequence[str] | None = None
) -> list[Path]:
    """Every E2 figure from ``<root>/tables/coverage_by_horizon.csv`` into ``<root>/figures``.

    Per target: ``coverage_by_horizon__<target>.png`` (in-domain test set), ``coverage_by_horizon__<target>__<dataset>.png``
    for every other data set except the calibration set, ``width_by_horizon__<target>.png`` (in-domain).
    """
    root = Path(root)
    coverage = pd.read_csv(root / "tables" / "coverage_by_horizon.csv")
    out: list[Path | None] = []
    for target in list(dict.fromkeys(coverage["target"].astype(str))):
        sel = coverage[coverage["target"] == target]
        figures = root / "figures"
        out.append(plot_coverage_by_horizon(sel, target, in_domain, figures / f"coverage_by_horizon__{target}.png", models))
        for dataset in dict.fromkeys(sel["dataset"].astype(str)):
            if dataset in (in_domain, calibration):
                continue
            path = figures / f"coverage_by_horizon__{target}__{dataset}.png"
            out.append(plot_coverage_by_horizon(sel, target, dataset, path, models))
        out.append(plot_width_by_horizon(sel, target, in_domain, figures / f"width_by_horizon__{target}.png", models))
    return [p for p in out if p is not None]
