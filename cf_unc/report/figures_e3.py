"""Figures of experiment E3 (docs/design.md, sections 13 and 14.2): amplitude CRPS by vehicle index and by driver
mode, speed-variability growth by vehicle index.

Every figure has a table twin under ``<root>/tables`` (``amplitude_crps.csv``, ``growth.csv``). Style and helpers come
from :mod:`cf_unc.report.figures` / :mod:`cf_unc.report.figures_e2`. Colour carries one job: the emphasised series
(the reference model in the CRPS figures, the measured platoons in the growth figure) is blue, every other model is
grey; models are told apart by their marker (fixed per model) and the legend, or by their axis label, never by colour
alone. Text is ink.

Signal variants (``VARIANT_LABEL``, ``VARIANT_ORDER``): the recorded and harmonised signals of design.md 13.2 and the
E3-pooled variants of 17.1 (checkpoints of the E6 roots ``e6_pool`` / ``e6_lodo_openacc``), in which every model runs
on its target-``a`` checkpoint: their model labels (``model_labels``) name QuantilePIDL's one-step target.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cf_unc.report.figures import BLUE, GRID, INK_2, MODEL_LABEL, MUTED, SURFACE, _plain_log_axis, _save, apply_style, label
from cf_unc.report.figures_e2 import _finish

__all__ = ["MARKERS", "MODEL_LABEL_E3", "MODEL_LABEL_E3_ONE_STEP", "MODE_LABEL", "ONE_STEP_VARIANTS", "VARIANT_LABEL",
           "VARIANT_ORDER", "make_e3_figures", "model_labels", "plot_crps_bars", "plot_crps_by_index", "plot_growth"]  # fmt: skip

MODEL_LABEL_E3 = {
    **MODEL_LABEL,
    "deep_ensemble": "Deep ensemble (kept member)",
    "quantile_pidl": "Quantile PIDL (1 s target)",
    "gp_residual_idm": "GP residual on IDM",
    "deep_ensemble__step": "Deep ensemble (per-step draws)",  # sensitivity of design.md 13.5 (artifacts/e3_step)
}
VARIANT_LABEL = {"raw": "signals as recorded", "harmonised": "harmonised signals",
                 "pool": "pooled domain, OpenACC seen", "lodo_openacc": "pooled domain, OpenACC held out"}  # 17.1
VARIANT_ORDER = ("raw", "harmonised", "pool", "lodo_openacc")  # order of the summaries; other variants follow
ONE_STEP_VARIANTS = ("pool", "lodo_openacc")  # every model on its target-a checkpoint (design.md 17.1)
MODEL_LABEL_E3_ONE_STEP = {**MODEL_LABEL_E3, "quantile_pidl": "Quantile PIDL (one-step target)"}
MODE_LABEL = {"ACC": "ACC platoons", "Human": "Human-driven platoons", "mixed": "Mixed platoon", "all": "All profiles"}
BAR_FILL = "#d6d5ce"  # grey fill of the model bars (between GRID and AXIS of the house palette)
MARKERS = {"idm_white_noise": "o", "lstm_const_sigma": "s", "deep_ensemble": "^", "gp_residual_idm": "D",
           "quantile_pidl": "v", "diffusion_cf": "P"}  # fmt: skip
_OTHER_MARKERS = ("X", "*", "h", "<", ">", "p")
GREY = MUTED
_DODGE = 0.11  # horizontal offset between models at one index
_LOG_SPAN = 10.0  # a value axis spanning more than this factor is drawn on a log scale


def model_labels(variant: str) -> dict[str, str]:
    """Model labels of an E3 signal variant: ``MODEL_LABEL_E3``, with QuantilePIDL on the one-step target in the
    variants of ``ONE_STEP_VARIANTS``."""
    return MODEL_LABEL_E3_ONE_STEP if str(variant) in ONE_STEP_VARIANTS else MODEL_LABEL_E3


def _value_axis(ax: plt.Axes, values: np.ndarray) -> bool:
    """Log scale (plain tick labels) when the finite positive ``values`` span more than ``_LOG_SPAN``, else linear
    from 0; returns whether the axis is logarithmic."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size and v.max() / v.min() > _LOG_SPAN:
        ax.set_yscale("log")
        ax.set_ylim(v.min() / 1.4, v.max() * 1.4)
        _plain_log_axis(ax.yaxis)
        return True
    ax.set_ylim(bottom=0.0)
    return False


def _models(names: Sequence[str], reference: str | None) -> list[str]:
    """Fixed order: the reference first, then the known models in MODEL_LABEL order, then the rest sorted."""
    present = list(dict.fromkeys(str(n) for n in names))
    known = [m for m in MODEL_LABEL_E3 if m in present]
    order = known + sorted(m for m in present if m not in known)
    if reference in order:
        order.remove(reference)
        order.insert(0, reference)
    return order


def _marker(model: str, order: Sequence[str]) -> str:
    if model in MARKERS:
        return MARKERS[model]
    extra = [m for m in order if m not in MARKERS]
    return _OTHER_MARKERS[extra.index(model) % len(_OTHER_MARKERS)]


def _style(model: str, reference: str | None) -> dict:
    """Colour and fill of a model's marks: the reference is blue and filled, the others grey and hollow."""
    if model == reference:
        return {"color": BLUE, "mfc": BLUE, "zorder": 4}
    return {"color": GREY, "mfc": SURFACE, "zorder": 3}


# ----------------------------------------------------------------------------------------------- CRPS
def plot_crps_by_index(
    crps: pd.DataFrame,
    variant: str,
    path: Path,
    set_name: str = "extended",
    mode: str = "all",
    window: str = "response",
    reference: str = "idm_white_noise",
) -> Path | None:
    """Mean amplitude CRPS (left) and amplitude-ratio CRPS (right) by vehicle index with bootstrap whiskers; the
    group ``all`` = mean over the measured indices of every profile."""
    sel = crps[(crps["variant"] == variant) & (crps["set"] == set_name) & (crps["mode"] == mode) & (crps["window"] == window)]
    sel = sel[sel["index"].astype(str) != "1-4"]
    if sel.empty:
        return None
    apply_style()
    models = _models(sel["model"], reference)
    numeric = sorted({int(i) for i in sel["index"].astype(str) if i.isdigit()})
    cats = [str(i) for i in numeric] + ["all"]
    xpos = {c: float(k) + (0.6 if c == "all" else 0.0) for k, c in enumerate(cats)}
    n_prof = sel[sel["model"] == models[0]].set_index(sel[sel["model"] == models[0]]["index"].astype(str))["n_profiles"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * _DODGE
    for ax, (column, ylabel) in zip(axes, (("crps", "amplitude CRPS, m/s"), ("crps_ratio", "CRPS of A_i / A_0"))):
        for off, model in zip(offsets, models):
            rows = sel[sel["model"] == model].assign(cat=lambda d: d["index"].astype(str)).set_index("cat").reindex(cats)
            x = np.array([xpos[c] for c in cats]) + off
            y = rows[column].to_numpy(float)
            st = _style(model, reference)
            ax.vlines(x, rows[f"{column}_lo"].to_numpy(float), rows[f"{column}_hi"].to_numpy(float), color=st["color"],
                      lw=1.4, zorder=st["zorder"] - 1)  # fmt: skip
            ax.plot(x, y, ls="none", marker=_marker(model, models), ms=6.0, mfc=st["mfc"], mec=st["color"], mew=1.4,
                    zorder=st["zorder"])  # fmt: skip
        ax.set_xticks([xpos[c] for c in cats])
        ax.set_xticklabels([f"{c}\nn={int(n_prof.get(c, 0))}" for c in cats], fontsize=8.5)
        ax.axvline((xpos[cats[-2]] + xpos["all"]) / 2 if len(cats) > 1 else -0.5, color=GRID, lw=1.0, zorder=0)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.6, xpos["all"] + 0.6)
        log = _value_axis(ax, sel[[column, f"{column}_lo", f"{column}_hi"]].to_numpy(float))
        ax.set_xlabel("vehicle index behind the leader (n = profiles that measured it)")
        ax.set_ylabel(ylabel + (", log scale" if log else ""))
    axes[0].set_title("Amplitude max(v) - min(v)", fontsize=10)
    axes[1].set_title("Amplitude relative to the leader's", fontsize=10)
    handles = [
        Line2D([], [], ls="none", marker=_marker(m, models), ms=6.0, mew=1.4, mec=_style(m, reference)["color"],
               mfc=_style(m, reference)["mfc"], label=label(m, model_labels(variant)) + (" (reference)" if m == reference else ""))
        for m in models
    ]  # fmt: skip
    n_total = int(sel["n_profiles"].max())
    _finish(
        fig, f"Amplitude CRPS by vehicle index: {VARIANT_LABEL.get(variant, variant)}",
        f"{set_name.capitalize()} set ({n_total} profiles, followers: {mode}); model amplitude over the profile and a 40 s "
        "tail, lower is better. Whiskers: 95 % bootstrap interval over profiles; 'all' averages the measured indices of "
        "every profile.",
        handles, ncol=min(3, len(handles)),
    )  # fmt: skip
    return _save(fig, path)


def _bar_value(v: float) -> str:
    """Value label of a bar: three significant digits with trailing zeros, never in exponent notation."""
    a = abs(v)
    return f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}" if a >= 1 else f"{v:.3f}"


def plot_crps_bars(
    crps: pd.DataFrame,
    variant: str,
    path: Path,
    set_name: str = "extended",
    window: str = "response",
    reference: str = "idm_white_noise",
    modes: Sequence[str] = ("ACC", "Human", "all"),
) -> Path | None:
    """Amplitude CRPS per driver mode (design.md 14.2, F5): one panel per mode, one bar per model (index ``all``: the
    mean over the measured vehicles of every profile) with its 95 % bootstrap interval as a whisker, the reference
    model as a horizontal line with its interval as a band. One value axis for all panels, logarithmic when the
    values span more than a factor ``_LOG_SPAN``."""
    sel = crps[(crps["variant"] == variant) & (crps["set"] == set_name) & (crps["window"] == window)
               & (crps["index"].astype(str) == "all")]  # fmt: skip
    present = set(sel["mode"].astype(str))
    modes = [m for m in modes if m in present]
    sel = sel[sel["mode"].isin(modes)]
    order = _models(sel["model"], reference)
    bars = [m for m in order if m != reference]
    if sel.empty or not bars:
        return None
    apply_style()
    n = len(bars)
    panel_w = max(2.7, 0.66 * n + 1.0)
    fig, axes = plt.subplots(1, len(modes), figsize=(panel_w * len(modes) + 0.9, 4.8), sharey=True, squeeze=False)
    axes = axes[0]
    values = sel[["crps", "crps_lo", "crps_hi"]].to_numpy(float)
    positive = values[np.isfinite(values) & (values > 0)]
    log = bool(positive.size) and positive.max() / positive.min() > _LOG_SPAN
    bottom = positive.min() / 1.4 if log else 0.0  # bars start at the bottom of the axis (same limit as _value_axis)
    x = np.arange(n, dtype=float)
    margin = 0.1 if n > 2 else 0.4  # a lone bar does not fill its panel
    for ax, mode in zip(axes, modes):
        rows = sel[sel["mode"] == mode].drop_duplicates("model").set_index("model")
        ref = rows.loc[reference] if reference in rows.index else None
        ref_text = ""
        if ref is not None and np.isfinite(ref["crps"]):
            ax.axhspan(ref["crps_lo"], ref["crps_hi"], color=BLUE, alpha=0.13, lw=0, zorder=1)
            ax.axhline(ref["crps"], color=BLUE, lw=1.8, zorder=5)
            ref_text = f", reference {_bar_value(float(ref['crps']))} m/s"
        sub = rows.reindex(bars)
        y, lo, hi = (sub[c].to_numpy(float) for c in ("crps", "crps_lo", "crps_hi"))
        ok = np.isfinite(y)
        ax.bar(x[ok], y[ok] - bottom, bottom=bottom, width=0.62, color=BAR_FILL, edgecolor=MUTED, lw=0.8, zorder=2)
        ax.vlines(x[ok], lo[ok], hi[ok], color=INK_2, lw=1.3, zorder=3)
        for xi, yi, hii in zip(x[ok], y[ok], hi[ok]):
            top = hii if np.isfinite(hii) else yi
            ax.annotate(_bar_value(float(yi)), xy=(xi, top), xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7.5, color=INK_2, zorder=6,
                        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6})  # the reference line passes behind  # fmt: skip
        ax.set_xticks(x)
        ax.set_xticklabels([textwrap.fill(label(m, model_labels(variant)), 13) for m in bars], fontsize=8)
        ax.set_xlim(-0.5 - margin, n - 0.5 + margin)
        ax.grid(axis="x", visible=False)
        n_prof = int(rows["n_profiles"].max()) if len(rows) else 0
        ax.set_title(MODE_LABEL.get(mode, mode), fontsize=10, pad=17)
        ax.text(0.0, 1.015, f"{n_prof} profiles{ref_text}", transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
                color=INK_2)  # fmt: skip
    _value_axis(axes[0], values)
    if not log:  # headroom for the value labels above the whiskers
        top = np.nanmax(np.concatenate([values.ravel(), [0.0]]))
        axes[0].set_ylim(0.0, top * 1.15 if top > 0 else 1.0)
    axes[0].set_ylabel("amplitude CRPS, m/s" + (", log scale" if log else ""))
    handles = [
        Patch(facecolor=BAR_FILL, edgecolor=MUTED, lw=0.8, label="model (bar: mean over profiles)"),
        Line2D([], [], color=INK_2, lw=1.3, label="95 % bootstrap interval over profiles"),
        Line2D([], [], color=BLUE, lw=1.8, label=f"{label(reference, model_labels(variant))} (reference)"),
        Patch(facecolor=BLUE, alpha=0.13, lw=0, label="reference: 95 % interval"),
    ]
    n_total = int(sel["n_profiles"].max())
    where = "the profile and the tail" if window == "response" else "the profile"
    _finish(
        fig, f"Platoon amplitude CRPS by driver mode: {VARIANT_LABEL.get(variant, variant)}",
        f"{set_name.capitalize()} set ({n_total} profiles); CRPS of the model's amplitude law at the measured amplitude "
        f"over {where}, mean over the measured vehicles of a profile, then over profiles; lower is better.",
        handles, ncol=4,
    )  # fmt: skip
    return _save(fig, path)


# --------------------------------------------------------------------------------------------- growth
def plot_growth(
    growth: pd.DataFrame, variant: str, path: Path, set_name: str = "extended", reference: str = "idm_white_noise"
) -> Path | None:
    """Mean speed standard deviation by vehicle index: left over the window of the data with the measured platoons
    (blue), right over the profile and the tail (design definition, models only); the legend gives each model's
    growth exponent beta of sd_i = sd_1 i^beta."""
    sel = growth[(growth["set"] == set_name) & (growth["variant"] == variant)]
    emp = growth[(growth["set"] == set_name) & (growth["model"] == "empirical")]
    lit = growth[(growth["variant"] == "literature") & (growth["window"] == "literature:sd")]  # configs/empirical_growth.yaml
    if sel.empty:
        return None
    apply_style()
    models = _models(sel["model"], reference)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    panels = (("profile", "Over the window of the measured data"), ("response", "Over the profile and the 40 s tail"))
    n_emp = emp[emp["sd"].notna()].set_index("index")["n_profiles"] if not emp.empty else pd.Series(dtype=float)
    for ax, (window, title) in zip(axes, panels):
        for model in models:
            rows = sel[(sel["model"] == model) & (sel["window"] == window)].sort_values("index")
            if rows.empty:
                continue
            ax.plot(rows["index"], rows["sd"], color=GREY, lw=1.2, marker=_marker(model, models), ms=4.5, mfc=SURFACE,
                    mec=GREY, mew=1.1, zorder=3)  # fmt: skip
        if window == "profile" and not emp.empty:
            e = emp[(emp["window"] == "profile") & emp["sd"].notna()].sort_values("index")
            ax.fill_between(e["index"], e["sd_lo"], e["sd_hi"], color=BLUE, alpha=0.15, lw=0, zorder=2)
            ax.plot(e["index"], e["sd"], color=BLUE, lw=2.0, marker="o", ms=6.0, mfc=BLUE, mec=SURFACE, mew=1.2, zorder=5)
        if window == "profile":
            for _, rows in lit.groupby("model", sort=False):
                rows = rows.sort_values("index")
                ax.plot(rows["index"], rows["sd"], color=INK_2, lw=1.2, ls="--", marker="x", ms=5.0, zorder=4)
        ax.set_title(title, fontsize=10)
        ticks = list(range(1, int(sel["index"].max()) + 1, 2 if sel["index"].max() > 10 else 1))
        ax.set_xticks(ticks)
        if window == "profile" and len(n_emp):  # profiles that measured the index under the left panel's ticks
            ax.set_xticklabels([f"{t}\nn={int(n_emp[t])}" if t in n_emp.index else str(t) for t in ticks], fontsize=8.5)
            ax.set_xlabel("vehicle index behind the leader (n = measured profiles)")
        else:
            ax.set_xlabel("vehicle index behind the leader")
        ax.grid(axis="x", visible=False)
    shown = np.concatenate([sel["sd"].to_numpy(float), emp["sd_lo"].to_numpy(float), emp["sd_hi"].to_numpy(float),
                            lit["sd"].to_numpy(float)])  # fmt: skip
    log = _value_axis(axes[0], shown)
    axes[0].set_ylabel("speed standard deviation, m/s" + (", log scale" if log else ""))

    def beta(model: str, window: str = "response") -> str:
        rows = sel[(sel["model"] == model) & (sel["window"] == window)]
        return "" if rows.empty or not np.isfinite(rows["beta"].iloc[0]) else f", beta {rows['beta'].iloc[0]:.2f}"

    handles = [
        Line2D([], [], color=GREY, lw=1.2, marker=_marker(m, models), ms=4.5, mfc=SURFACE, mec=GREY, mew=1.1,
               label=label(m, model_labels(variant)) + beta(m))
        for m in models
    ]  # fmt: skip
    if not emp.empty:
        b = emp["beta"].iloc[0]
        handles.insert(0, Line2D([], [], color=BLUE, lw=2.0, marker="o", ms=6.0, mfc=BLUE, mec=SURFACE,
                                 label="measured platoons" + (f", beta {b:.2f}" if np.isfinite(b) else "")))  # fmt: skip
    for name, rows in lit.groupby("model", sort=False):
        handles.append(Line2D([], [], color=INK_2, lw=1.2, ls="--", marker="x", ms=5.0,
                              label=f"literature: {name}, beta {rows['beta'].iloc[0]:.2f}"))  # fmt: skip
    n_total = int(sel["n_profiles"].max())
    _finish(
        fig, f"Growth of the speed variability along the platoon: {VARIANT_LABEL.get(variant, variant)}",
        f"{set_name.capitalize()} set ({n_total} profiles); mean over rollouts, seeds and profiles. Blue band: 95 % "
        "bootstrap interval over the profiles that measured the index. beta of the models: profile + tail window.",
        handles, ncol=min(4, len(handles)),
    )  # fmt: skip
    return _save(fig, path)


def make_e3_figures(root: str | Path, reference: str = "idm_white_noise", set_name: str = "extended") -> list[Path]:
    """Every E3 figure from ``<root>/tables`` into ``<root>/figures``: ``crps_by_index__<variant>.png``,
    ``growth__<variant>.png`` and ``crps_bars__<variant>.png`` (CRPS per driver mode) for the profile set
    ``set_name`` (falls back to the first set of the tables when it is absent)."""
    root = Path(root)
    crps = pd.read_csv(root / "tables" / "amplitude_crps.csv", dtype={"index": str})
    growth = pd.read_csv(root / "tables" / "growth.csv")
    out: list[Path | None] = []
    for variant in dict.fromkeys(crps["variant"].astype(str)):
        sets = set(crps.loc[crps["variant"] == variant, "set"])
        sn = set_name if set_name in sets else sorted(sets)[0]
        out.append(plot_crps_by_index(crps, variant, root / "figures" / f"crps_by_index__{variant}.png", sn, reference=reference))
        out.append(plot_growth(growth, variant, root / "figures" / f"growth__{variant}.png", sn, reference=reference))
        out.append(plot_crps_bars(crps, variant, root / "figures" / f"crps_bars__{variant}.png", sn, reference=reference))
    return [p for p in out if p is not None]
