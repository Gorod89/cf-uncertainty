"""Static figures of the uncertainty analysis (matplotlib).

Every figure has a table twin under ``<root>/tables`` (the CSV it is drawn from). Colour carries one job
per figure: magnitude on a single-hue ramp (maps), identity of two series in a fixed order (epistemic =
blue, aleatoric = orange; heteroscedastic = blue, homoscedastic = orange). Text is always ink.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator  # noqa: E402

from cf_unc.maps.io import root_target  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_RAMP = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95", "#104281", "#0d366b")  # fmt: skip
ORANGE_RAMP = ("#fde4d8", "#f9cbb6", "#f5b092", "#f1946e", "#ee7d50", "#eb6834", "#d1592a", "#b24a22", "#933c1b",
               "#742e14", "#57220e")  # fmt: skip
CMAP = {
    "epistemic": LinearSegmentedColormap.from_list("cf_blue", BLUE_RAMP),
    "aleatoric": LinearSegmentedColormap.from_list("cf_orange", ORANGE_RAMP),
}
DATASET_LABEL = {
    "pool_test": "Pooled test folds (in-domain)",  # experiment E6 (design.md 16.1)
    "highd": "highD test (in-domain)",
    "highd_val": "highD validation (placebo)",
    "ngsim": "NGSIM I-80 (reconstructed)",
    "ngsim_us101": "NGSIM US-101 (reconstructed)",
    "ngsim_fn": "NGSIM I-80 (FollowNet events)",
    "waymo": "Waymo pairs",
    "openacc": "OpenACC",
}
# what the predictive law is about, per target of the windows tables (titles of the regime figures)
TARGET_LABEL = {"a": "one-step acceleration", "mean_1s": "1 s mean acceleration"}
MODEL_LABEL = {
    "idm_white_noise": "IDM + white noise",
    "idm_white_noise_mle": "IDM + white noise (one-step fit)",
    "lstm_const_sigma": "LSTM, constant sigma",
    "hetero_mdn": "HeteroMDN",
    "hetero_mdn_mix3": "HeteroMDN, 3 components",
    "deep_ensemble": "Deep ensemble",
    "mc_dropout": "MC dropout",
    "gp_residual_idm": "GP residual on IDM",
    "quantile_pidl": "Quantile PIDL",
    "diffusion_cf": "Diffusion",
    # sampler-mode variants of the rollout experiment (configs/experiment/e2.yaml, `variants`)
    "deep_ensemble__traj": "Deep ensemble (kept member)",
    "mc_dropout__traj": "MC dropout (kept masks)",
    "gp_residual_idm__step": "GP residual (per-step draws)",
}
REGIMES = ("congested", "braking", "approaching", "free", "steady", "other")


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.weight": "normal",
            "font.size": 9.5,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.5,
        }
    )


def label(name: str, table: dict[str, str]) -> str:
    return table.get(name, name)


def _plain_log_axis(axis) -> None:
    """Log axis with plain numbers: ticks at 1, 2, 5 x 10^k, decades only when the axis spans more than 2.5 decades."""
    lo, hi = axis.get_view_interval()
    wide = lo > 0 and np.log10(hi / lo) > 2.5
    axis.set_major_locator(LogLocator(base=10, subs=(1.0,) if wide else (1.0, 2.0, 5.0), numticks=12))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _caption(fig: plt.Figure, text: str, below: bool = False) -> None:
    """Figure caption at the bottom right, next to the legend; ``below``: under the legend, left-aligned (narrow
    figures, where legend and caption would overlap on one line)."""
    if below:
        fig.text(0.01, -0.035, text, fontsize=8, color=INK_2, ha="left", va="top")
    else:
        fig.text(0.99, -0.015, text, fontsize=8, color=INK_2, ha="right", va="bottom")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _grid2d(rows: pd.DataFrame, column: str) -> np.ndarray:
    """(n_s, n_dv) array of ``column`` from the in-grid rows of ``maps_long`` (cells ordered by i_s, i_dv)."""
    inside = rows.dropna(subset=["s_lo"]).sort_values(["i_s", "i_dv"])
    n_s, n_dv = int(inside["i_s"].max()) + 1, int(inside["i_dv"].max()) + 1
    return np.array(inside[column].to_numpy(dtype=np.float64)).reshape(n_s, n_dv)


# ------------------------------------------------------------------------------- maps
def plot_maps(
    maps_long: pd.DataFrame,
    model: str,
    kind: str,
    datasets: Sequence[str],
    path: Path,
    min_count: int = 30,
    rare_rule: str = "rare_mass",
) -> Path | None:
    """Epistemic and aleatoric variance over (dv, s): one row per data set, colour scale shared per column."""
    sel = maps_long[(maps_long.model == model) & (maps_long.kind == kind) & (maps_long.subset == "all")]
    datasets = [d for d in datasets if d in set(sel.dataset)]
    if not datasets:
        return None
    apply_style()
    quantities = [q for q in ("epistemic", "aleatoric") if np.nanmax(sel[q].to_numpy(dtype=float), initial=0.0) > 0.0]
    if not quantities:
        return None
    fig, axes = plt.subplots(
        len(datasets), len(quantities), figsize=(3.9 * len(quantities) + 0.6, 2.35 * len(datasets) + 0.7), squeeze=False
    )
    first = sel[sel.dataset == datasets[0]].dropna(subset=["s_lo"])
    extent = [first.dv_lo.min(), first.dv_hi.max(), first.s_lo.min(), first.s_hi.max()]
    for col, quantity in enumerate(quantities):
        values = {}
        for d in datasets:
            rows = sel[sel.dataset == d]
            grid = _grid2d(rows, quantity)
            grid[_grid2d(rows, "count") < min_count] = np.nan
            values[d] = grid
        pooled = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in values.values()])
        if pooled.size == 0:
            continue
        lo, hi = np.quantile(pooled, [0.02, 0.98])
        hi = max(hi, lo * 1.5)
        norm = LogNorm(vmin=lo, vmax=hi)
        for row, d in enumerate(datasets):
            ax = axes[row][col]
            ax.grid(False)
            image = ax.imshow(
                np.clip(values[d], lo, hi), origin="lower", extent=extent, aspect="auto", cmap=CMAP[quantity], norm=norm,
                interpolation="nearest",
            )  # fmt: skip
            rare = _grid2d(sel[sel.dataset == d], rare_rule).astype(float)
            xs = np.linspace(extent[0], extent[1], rare.shape[1] + 1)
            ys = np.linspace(extent[2], extent[3], rare.shape[0] + 1)
            ax.contour(
                0.5 * (xs[1:] + xs[:-1]), 0.5 * (ys[1:] + ys[:-1]), rare, levels=[0.5], colors=[INK_2], linewidths=0.9
            )
            ax.set_title(f"{quantity.capitalize()} variance" if row == 0 else "", pad=6)
            if col == 0:
                ax.set_ylabel(f"{label(d, DATASET_LABEL)}\ngap s, m")
            if row == len(datasets) - 1:
                ax.set_xlabel("approach rate dv = v - v_lead, m/s")
            bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
            bar.outline.set_visible(False)
            bar.ax.tick_params(labelsize=8, colors=MUTED, labelcolor=INK_2, length=0)
            if row == 0:
                bar.ax.set_title("(m/s²)²", fontsize=8, color=INK_2, loc="left", pad=4)
    tag = "" if kind == "model" else " (across-seed decomposition)"
    fig.suptitle(f"{label(model, MODEL_LABEL)}{tag}: uncertainty over the state space", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(
        0.01, -0.005,
        f"Cells with fewer than {min_count} windows are blank. The line encloses the frequent states of the training data "
        f"({'least frequent cells holding 10 % of the training windows lie outside' if rare_rule == 'rare_mass' else 'cells with a training frequency below 1 % lie outside'}).",
        fontsize=8, color=INK_2, ha="left", va="top",
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    return _save(fig, path)


# ----------------------------------------------------------------------- shift ratios
def plot_shift_ratios(
    verdicts: pd.DataFrame,
    models: Sequence[str],
    datasets: Sequence[str],
    path: Path,
    matched: bool = False,
    title: str | None = None,
) -> Path | None:
    """Ratio shifted / in-domain of the mean epistemic and aleatoric variance with bootstrap intervals."""
    sel = verdicts[(verdicts.subset == "all")]
    datasets = [d for d in datasets if d in set(sel.dataset)]
    rows = []
    for m in models:
        own = sel[(sel.model == m) & (sel.kind == "model") & sel.r_epi.notna()]
        kind = "model" if len(own) else "seed_ensemble"
        if len(sel[(sel.model == m) & (sel.kind == kind)]):
            rows.append((m, kind))
    if not rows or not datasets:
        return None
    apply_style()
    suffix = "_matched" if matched else ""
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.3 * len(datasets) + 1.6, 0.42 * len(rows) + 1.9), sharey=True, squeeze=False)
    y = np.arange(len(rows))[::-1]
    for ax, d in zip(axes[0], datasets):
        ax.axvspan(0.7, 1.3, color=GRID, alpha=0.55, lw=0)
        ax.axvline(1.0, color=AXIS, lw=1.0)
        # threshold labels on the outer side of their lines (close on a log axis, they would otherwise run together),
        # on a surface-coloured patch so that the reference line at 1 does not cross the glyphs of narrow panels
        for x, text, ha, factor in ((3.0, "3×", "left", 1.04), (1.5, "1.5×", "right", 1.0 / 1.04)):
            ax.axvline(x, color=MUTED, lw=0.8)
            ax.text(x * factor, len(rows) - 0.42, text, fontsize=7.5, color=INK_2, ha=ha, va="bottom", zorder=4,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.8})  # fmt: skip
        for (m, kind), yy in zip(rows, y):
            r = sel[(sel.model == m) & (sel.kind == kind) & (sel.dataset == d)]
            if r.empty:
                continue
            r = r.iloc[0]
            for name, colour, off in (("epi", BLUE, 0.16), ("alea", ORANGE, -0.16)):
                est, lo, hi = r[f"r_{name}{suffix}"], r[f"r_{name}{suffix}_lo"], r[f"r_{name}{suffix}_hi"]
                if not np.isfinite(est) or est <= 0:
                    continue
                ax.plot([max(lo, 1e-3), hi], [yy + off, yy + off], color=colour, lw=2.0, solid_capstyle="round")
                ax.plot(
                    [est], [yy + off], marker="o", ms=6.5, mfc=colour if kind == "model" else SURFACE, mec=colour, mew=1.6,
                    ls="none", zorder=3,
                )  # fmt: skip
        ax.set_xscale("log")
        _plain_log_axis(ax.xaxis)
        ax.set_title(label(d, DATASET_LABEL))
        ax.set_xlabel("variance ratio, shifted / in-domain")
        ax.set_ylim(-0.6, len(rows) - 0.05)
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(y)
    axes[0][0].set_yticklabels([label(m, MODEL_LABEL) for m, _ in rows])
    handles = [
        Line2D([], [], color=BLUE, marker="o", ms=6.5, lw=2, label="epistemic"),
        Line2D([], [], color=ORANGE, marker="o", ms=6.5, lw=2, label="aleatoric"),
        Line2D([], [], color=INK_2, marker="o", ms=6.5, mfc=SURFACE, mew=1.6, lw=0, label="across-seed decomposition (model has no epistemic output)"),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, -0.02), ncol=3, fontsize=8.5, handlelength=1.6)
    fig.suptitle(
        title or ("Dataset shift: variance ratios at matched (s, dv) states" if matched else "Dataset shift: variance ratios"),
        x=0.01, ha="left", fontsize=12, fontweight="bold",
    )  # fmt: skip
    caption = "Whiskers: 95 % bootstrap interval over events. Grey band: aleatoric change below 30 %."
    _caption(fig, caption, below=len(datasets) < 3)  # the three-entry legend fills a figure of two panels
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return _save(fig, path)


# ---------------------------------------------------------------------------- regimes
def _regime_axis(ax: plt.Axes, regimes: Sequence[str]) -> np.ndarray:
    x = np.arange(len(regimes))
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=30, ha="right")
    ax.grid(axis="x", visible=False)
    return x


def plot_nll_by_regime(
    regime_stats: pd.DataFrame,
    datasets: Sequence[str],
    path: Path,
    models: tuple[str, str] = ("hetero_mdn", "lstm_const_sigma"),
    target: str = "a",
) -> Path | None:
    """NLL of the heteroscedastic and the homoscedastic model per regime, one panel (own scale) per data set."""
    sel = regime_stats[(regime_stats.subset == "all") & regime_stats.model.isin(models)]
    datasets = [d for d in datasets if d in set(sel.dataset)]
    if sel.empty or not datasets:
        return None
    apply_style()
    regimes = ["all", *[r for r in REGIMES if r in set(sel.regime)]]
    n_col = 2 if len(datasets) == 4 else min(3, len(datasets))
    n_row = int(np.ceil(len(datasets) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(3.7 * n_col + 0.5, 2.9 * n_row + 0.9), squeeze=False)
    for ax, d in zip(axes.ravel(), datasets):
        x = _regime_axis(ax, regimes)
        for m, colour, off in ((models[0], BLUE, -0.13), (models[1], ORANGE, 0.13)):
            r = sel[(sel.dataset == d) & (sel.model == m)].set_index("regime").reindex(regimes)
            ax.vlines(x + off, r.nll_lo, r.nll_hi, color=colour, lw=2.0)
            ax.plot(x + off, r.nll, marker="o", ms=6.5, mfc=colour, mec=SURFACE, mew=1.2, ls="none", zorder=3)
        ax.set_title(label(d, DATASET_LABEL))
        ax.set_ylabel("NLL, nats")
    for ax in axes.ravel()[len(datasets):]:
        ax.set_visible(False)
    handles = [
        Line2D([], [], color=BLUE, marker="o", ms=6.5, lw=2, label=label(models[0], MODEL_LABEL)),
        Line2D([], [], color=ORANGE, marker="o", ms=6.5, lw=2, label=label(models[1], MODEL_LABEL)),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, -0.02), ncol=2, fontsize=8.5, handlelength=1.6)
    title = "One-step NLL" if target == "a" else f"NLL of the {label(target, TARGET_LABEL)}"
    fig.suptitle(f"{title} by driving regime (every panel has its own scale)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    _caption(fig, "Whiskers: 95 % bootstrap interval over events; mean over seeds.", below=n_col < 2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return _save(fig, path)


def plot_regime_variance(
    ordering: pd.DataFrame, model: str, datasets: Sequence[str], path: Path, target: str = "a"
) -> Path | None:
    """Predicted variance and squared residual of every regime relative to steady following."""
    sel = ordering[(ordering.subset == "all") & (ordering.model == model)]
    datasets = [d for d in datasets if d in set(sel.dataset)]
    if sel.empty or not datasets:
        return None
    apply_style()
    regimes = [r for r in REGIMES if r in set(sel.regime)]
    n_col = 2 if len(datasets) == 4 else min(3, len(datasets))
    n_row = int(np.ceil(len(datasets) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(3.7 * n_col + 0.5, 2.9 * n_row + 0.9), squeeze=False, sharey=True)
    for ax, d in zip(axes.ravel(), datasets):
        x = _regime_axis(ax, regimes)
        ax.axhline(1.0, color=AXIS, lw=1.0)
        for quantity, colour, off in (("var", BLUE, -0.13), ("mse", ORANGE, 0.13)):
            r = sel[(sel.dataset == d) & (sel.quantity == quantity)].set_index("regime").reindex(regimes)
            ax.vlines(x + off, r.lo, r.hi, color=colour, lw=2.0)
            ax.plot(x + off, r.estimate, marker="o", ms=6.5, mfc=colour, mec=SURFACE, mew=1.2, ls="none", zorder=3)
        ax.set_yscale("log")
        _plain_log_axis(ax.yaxis)
        ax.set_title(label(d, DATASET_LABEL))
    for ax in axes[:, 0]:
        ax.set_ylabel("ratio to steady following")
    for ax in axes.ravel()[len(datasets):]:
        ax.set_visible(False)
    handles = [
        Line2D([], [], color=BLUE, marker="o", ms=6.5, lw=2, label="predicted variance"),
        Line2D([], [], color=ORANGE, marker="o", ms=6.5, lw=2, label="squared residual"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, -0.02), ncol=2, fontsize=8.5, handlelength=1.6)
    what = "acceleration" if target == "a" else label(target, TARGET_LABEL)
    fig.suptitle(f"{label(model, MODEL_LABEL)}: {what} variance by regime", x=0.01, ha="left", fontsize=12, fontweight="bold")
    _caption(fig, "Whiskers: 95 % bootstrap interval over events; mean over seeds.", below=n_col < 2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return _save(fig, path)


# ------------------------------------------------------------------------------ entry
def make_e1_figures(
    root: str | Path,
    models: Iterable[str] | None = None,
    datasets: Sequence[str] = ("pool_test", "highd", "ngsim", "ngsim_us101", "waymo", "openacc"),
    shifted: Sequence[str] = ("ngsim", "ngsim_us101", "waymo", "openacc"),
    target: str | None = None,
) -> list[Path]:
    """Every M2 figure from the tables under ``root/tables`` into ``root/figures``.

    ``target`` names the target in the regime titles (default: that of the windows tables under ``root``). Data sets
    without rows are left out (``pool_test`` exists in the roots of experiment E6 only, design.md 16.1).
    """
    root = Path(root)
    tables = root / "tables"
    target = target or root_target(root)
    maps_long = pd.read_csv(tables / "maps_long.csv")
    verdicts = pd.read_csv(tables / "h31_verdicts.csv")
    stats = pd.read_csv(tables / "regime_stats.csv")
    ordering = pd.read_csv(tables / "regime_ordering.csv")
    names = list(models) if models is not None else [m for m in MODEL_LABEL if m in set(maps_long.model)]
    out: list[Path | None] = []
    for m in names:
        for kind in sorted(set(maps_long[maps_long.model == m].kind)):
            out.append(plot_maps(maps_long, m, kind, datasets, root / "figures" / "maps" / f"{m}__{kind}.png"))
        out.append(plot_regime_variance(ordering, m, datasets, root / "figures" / "regimes" / f"variance__{m}.png", target))
    out.append(plot_shift_ratios(verdicts, names, shifted, root / "figures" / "shift_ratios.png"))
    out.append(plot_shift_ratios(verdicts, names, shifted, root / "figures" / "shift_ratios_matched.png", matched=True))
    out.append(plot_shift_ratios(verdicts, names, [d for d in ("highd_val", "ngsim_fn") if d in set(verdicts.dataset)],
                                 root / "figures" / "shift_ratios_controls.png", title="Controls: placebo split and second NGSIM extraction"))  # fmt: skip
    out.append(plot_nll_by_regime(stats, [*datasets], root / "figures" / "nll_by_regime.png", target=target))
    return [p for p in out if p is not None]
