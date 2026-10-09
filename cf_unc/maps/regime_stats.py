"""Statistics per driving regime and the heteroscedastic-vs-homoscedastic comparison (H3.3, design.md 10.5).

Intervals and p-values come from the event bootstrap of :mod:`cf_unc.maps.shift`: per-unit sums per
seed and regime, the same resampled units for every seed and every regime of a call (so the regimes of
a ratio come from the same resampled events; an event without windows of a regime simply contributes
nothing to that regime). Estimates are seed averages of the per-seed means.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from cf_unc.eval.bootstrap import holm
from cf_unc.maps.regimes import REGIMES
from cf_unc.maps.shift import (
    SeedLike,
    SeedStack,
    _any_valid,
    _as_2d,
    _avg_seeds,
    _check,
    _collapse,
    _interval,
    _ratio,
    _seed_group_means,
    _unit_group_sums,
    bootstrap_pvalue,
    encode_units,
    own_decomposition,
    resample_sums,
    seed_decomposition,
)

__all__ = [
    "REGIME_TABLE_COLUMNS",
    "ORDERING_COLUMNS",
    "NLL_COLUMNS",
    "regime_table",
    "regime_ordering",
    "nll_comparison",
]

REGIME_TABLE_COLUMNS = [
    "regime", "share", "n_windows", "n_events",
    "var", "var_lo", "var_hi", "aleatoric", "epistemic", "epistemic_seed",
    "mse", "mse_lo", "mse_hi", "nll", "nll_lo", "nll_hi", "crps", "picp", "n_seeds",
]  # fmt: skip
ORDERING_COLUMNS = [
    "quantity", "regime", "reference", "estimate", "lo", "hi", "se", "pvalue", "p_holm", "reject",
    "n_windows", "n_windows_ref", "n_events", "n_seeds",
]  # fmt: skip
NLL_COLUMNS = [
    "group", "n_windows", "n_events", "nll_a", "nll_a_lo", "nll_a_hi", "nll_b", "nll_b_lo", "nll_b_hi",
    "diff", "diff_lo", "diff_hi", "se", "pvalue", "ci_overlap", "n_seeds_a", "n_seeds_b",
]  # fmt: skip
_BOOT_QUANTITIES = ("var", "mse", "nll")  # quantities of regime_table with bootstrap intervals


def _labels(labels: ArrayLike, n: int, n_groups: int) -> np.ndarray:
    lab = np.asarray(labels, dtype=np.int64).ravel()
    if lab.shape[0] != n:
        raise ValueError(f"labels must have {n} entries, got {lab.shape[0]}")
    if lab.size and (lab.min() < 0 or lab.max() >= n_groups):
        raise ValueError(f"labels must lie in 0..{n_groups - 1}")
    return lab


def _group_means(x: np.ndarray, lab: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (seed) means of the non-NaN values per group (K, G) and over all rows (K,)."""
    k = x.shape[0]
    idx = (np.arange(k, dtype=np.int64)[:, None] * n_groups + lab[None, :]).ravel()
    flat = x.ravel()
    ok = ~np.isnan(flat)
    sums = np.bincount(idx[ok], weights=flat[ok], minlength=k * n_groups).reshape(k, n_groups)
    counts = np.bincount(idx[ok], minlength=k * n_groups).reshape(k, n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / counts, sums.sum(axis=1) / counts.sum(axis=1)


def _seed_avg(per_seed: np.ndarray) -> float:
    fin = np.isfinite(per_seed)
    return float(per_seed[fin].mean()) if fin.any() else float("nan")


def regime_table(
    windows_frame: "pd.DataFrame | Mapping[str, ArrayLike]",
    pred_frames: "Sequence[pd.DataFrame] | SeedStack",
    labels: ArrayLike,
    n_boot: int = 1000,
    seed: SeedLike = 0,
    ci: float = 0.95,
    *,
    unit: str = "event",
    regimes: Sequence[str] = REGIMES,
    include_all: bool = True,
) -> pd.DataFrame:
    """Statistics per regime of one model on one data set (design.md 10.5).

    windows_frame: columns ``y`` and ``unit`` (``event`` or ``driver``); pred_frames: one prediction table
    per seed (same rows) or a :class:`SeedStack`; labels: (n,) indices into ``regimes``.
    One row per regime (``"all"`` first when ``include_all``) with ``share`` and ``n_windows`` of the
    evaluated windows (a prediction in at least one seed), ``n_events`` (units with such a window), the
    seed mean of the per-seed means of the predicted total variance ``var``, ``aleatoric``, ``epistemic``
    (own decomposition, NaN when the model has none), ``epistemic_seed`` (across-seed variance of the
    means, ddof = 0; NaN with one seed), mean squared residual ``mse = (y - mean)^2``, ``nll``, ``crps``,
    ``picp`` (share of covered targets), and event-bootstrap intervals for ``var``, ``mse`` and ``nll``.
    """
    _check(n_boot, ci)
    stack = SeedStack.coerce(pred_frames)
    n = len(stack)
    y = np.asarray(windows_frame["y"], dtype=np.float64).ravel()
    units = np.asarray(windows_frame[unit]).ravel()
    if y.shape[0] != n or units.shape[0] != n:
        raise ValueError("windows and prediction tables have different numbers of rows")
    n_reg = len(regimes)
    lab = _labels(labels, n, n_reg)
    own = own_decomposition(stack)
    mean = stack["mean"].astype(np.float64)
    q: dict[str, np.ndarray] = {
        "var": own["total"],
        "aleatoric": own["aleatoric"],
        "epistemic": own["epistemic"],
        "mse": (y[None, :] - mean) ** 2,
        "nll": stack["nll"].astype(np.float64),
        "crps": stack["crps"].astype(np.float64),
        "picp": stack["covered"].astype(np.float64),
    }
    if stack.n_seeds >= 2:
        q["epistemic_seed"] = seed_decomposition(mean, own["total"])[1][None, :]
    else:
        q["epistemic_seed"] = np.full((1, n), np.nan)
    evaluated = stack.valid.any(axis=0)
    per_group = {name: _group_means(x, lab, n_reg) for name, x in q.items()}

    codes, unit_ids = encode_units(units[evaluated])
    n_units = len(unit_ids)
    lab_e = lab[evaluated]
    blocks: list[np.ndarray] = []
    k_eff: dict[str, int] = {}
    for name in _BOOT_QUANTITIES:
        xe = _collapse(q[name])  # seed means when all seeds evaluated the same windows (exact, n_seeds times cheaper)
        s, c = _unit_group_sums(xe[:, evaluated], codes, n_units, lab_e, n_reg)
        blocks += [s, c]
        k_eff[name] = xe.shape[0]
    reps = resample_sums(np.hstack(blocks), n_boot, seed)
    k = stack.n_seeds
    rep_sums = {}
    col = 0
    for name in _BOOT_QUANTITIES:
        width = k_eff[name] * n_reg
        s_rep = reps[:, col : col + width].reshape(-1, k_eff[name], n_reg)
        c_rep = reps[:, col + width : col + 2 * width].reshape(-1, k_eff[name], n_reg)
        rep_sums[name] = (s_rep, c_rep)
        col += 2 * width

    n_windows = np.bincount(lab_e, minlength=n_reg)
    has = np.bincount(codes.astype(np.int64) * n_reg + lab_e, minlength=n_units * n_reg).reshape(n_units, n_reg) > 0
    n_events = has.sum(axis=0)
    total = int(evaluated.sum())

    groups: list[tuple[str, int | None]] = ([("all", None)] if include_all else []) + [
        (name, r) for r, name in enumerate(regimes)
    ]
    rows = []
    for name, r in groups:
        row: dict[str, Any] = {"regime": name}
        if r is None:
            row.update(share=1.0 if total else float("nan"), n_windows=total, n_events=n_units)
        else:
            row.update(
                share=_ratio(float(n_windows[r]), float(total)), n_windows=int(n_windows[r]), n_events=int(n_events[r])
            )
        for qname, (by_group, overall) in per_group.items():
            per_seed = overall if r is None else by_group[:, r]
            row[qname] = _seed_avg(per_seed)
            if qname in rep_sums:
                s_rep, c_rep = rep_sums[qname]
                with np.errstate(invalid="ignore", divide="ignore"):
                    rep = s_rep.sum(axis=2) / c_rep.sum(axis=2) if r is None else s_rep[:, :, r] / c_rep[:, :, r]
                lo, hi, _ = _interval(_avg_seeds(rep, np.isfinite(per_seed)), ci)
                row[f"{qname}_lo"], row[f"{qname}_hi"] = lo, hi
        row["n_seeds"] = k
        rows.append(row)
    return pd.DataFrame(rows, columns=REGIME_TABLE_COLUMNS)


def regime_ordering(
    quantities: Mapping[str, ArrayLike],
    labels: ArrayLike,
    events: ArrayLike,
    reference: str = "steady",
    n_boot: int = 1000,
    seed: SeedLike = 0,
    ci: float = 0.95,
    *,
    regimes: Sequence[str] = REGIMES,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Ratio of every regime to the reference regime, with the event bootstrap (design.md 10.5).

    quantities: name -> (n_seeds, n) or (n,) per-window values (e.g. ``var`` = predicted variance, ``mse`` =
    squared residual); labels: (n,) indices into ``regimes``; events: (n,) unit labels.
    ``estimate = mean_k m_k(regime) / mean_k m_k(reference)`` (seed averages of the per-seed means); the
    replicates take both regimes from the same resampled units. ``pvalue``: two-sided bootstrap p-value of
    "ratio = 1", :func:`cf_unc.maps.shift.bootstrap_pvalue` on the log ratio (+1 smoothing).
    ``p_holm`` / ``reject``: Holm correction over every row of this call (the family passed in).
    """
    _check(n_boot, ci)
    arrays = {k: _as_2d(v) for k, v in quantities.items()}
    names = list(regimes)
    if reference not in names:
        raise ValueError(f"reference {reference!r} is not one of {names}")
    ref = names.index(reference)
    n_reg = len(names)
    n = next(iter(arrays.values())).shape[1]
    lab = _labels(labels, n, n_reg)
    ev = np.asarray(events).ravel()
    keep = _any_valid(arrays.values())
    codes, unit_ids = encode_units(ev[keep])
    n_units = len(unit_ids)
    blocks: list[np.ndarray] = []
    k_eff: list[int] = []
    for x in arrays.values():
        xe = _collapse(x)  # seed means when all seeds cover the same windows (exact)
        s, c = _unit_group_sums(xe[:, keep], codes, n_units, lab[keep], n_reg)
        blocks += [s, c]
        k_eff.append(xe.shape[0])
    reps = resample_sums(np.hstack(blocks), n_boot, seed)
    rows = []
    col = 0
    for i, (qname, x) in enumerate(arrays.items()):
        width = k_eff[i] * n_reg
        full = _seed_group_means(x, lab, n_reg)  # (K, regimes) per-seed means of the full data
        with np.errstate(invalid="ignore", divide="ignore"):
            rep = (reps[:, col : col + width] / reps[:, col + width : col + 2 * width]).reshape(-1, k_eff[i], n_reg)
        col += 2 * width
        valid = (~np.isnan(x)).any(axis=0)
        n_win = np.bincount(lab[valid], minlength=n_reg)
        for r, name in enumerate(names):
            if r == ref:
                continue
            ok = np.isfinite(full[:, r]) & np.isfinite(full[:, ref])
            num = float(full[ok, r].mean()) if ok.any() else float("nan")
            den = float(full[ok, ref].mean()) if ok.any() else float("nan")
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio_reps = _avg_seeds(rep[:, :, r], ok) / _avg_seeds(rep[:, :, ref], ok)
                log_reps = np.log(ratio_reps)
            lo, hi, se = _interval(ratio_reps, ci)
            rows.append(
                {
                    "quantity": qname,
                    "regime": name,
                    "reference": reference,
                    "estimate": _ratio(num, den),
                    "lo": lo,
                    "hi": hi,
                    "se": se,
                    "pvalue": bootstrap_pvalue(log_reps, 0.0) if ok.any() else float("nan"),
                    "n_windows": int(n_win[r]),
                    "n_windows_ref": int(n_win[ref]),
                    "n_events": n_units,
                    "n_seeds": int(ok.sum()),
                }
            )
    frame = pd.DataFrame(rows, columns=ORDERING_COLUMNS)
    if len(frame):
        adjusted, reject = holm(frame["pvalue"].to_numpy(dtype=np.float64), alpha=alpha)
        frame["p_holm"] = adjusted
        frame["reject"] = reject
    return frame


def _nan_seed_mean(x: np.ndarray) -> np.ndarray:
    """Mean over the seeds (axis 0) of the non-NaN values (inf kept), NaN where none."""
    ok = ~np.isnan(x)
    cnt = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(ok, x, 0.0).sum(axis=0) / cnt
    return np.where(cnt > 0, out, np.nan)


def nll_comparison(
    nll_a: ArrayLike,
    nll_b: ArrayLike,
    events: ArrayLike,
    labels: ArrayLike | None = None,
    n_boot: int = 1000,
    seed: SeedLike = 0,
    ci: float = 0.95,
    *,
    regimes: Sequence[str] = REGIMES,
) -> pd.DataFrame:
    """Paired NLL comparison of models a and b (e.g. HeteroMDN against LSTMConstSigma), design.md 10.5.

    nll_a: (n_seeds_a, n), nll_b: (n_seeds_b, n) per-window NLL (nats) on the same windows; the per-window
    NLL is first averaged over each model's seeds (NaN-aware), windows where either average is NaN are
    dropped (pairing by window). One row for ``"all"`` and, with ``labels`` (n,) indices into ``regimes``,
    one per regime: ``nll_a`` / ``nll_b`` with their own event-bootstrap intervals, ``diff = nll_a - nll_b``
    (negative: a has the lower NLL) with interval, ``se``, two-sided bootstrap ``pvalue`` of ``diff = 0``
    (+1 smoothing), and ``ci_overlap`` = whether the two separate intervals overlap (the literal criterion
    of the plan). All quantities of a call share the resampled units. Holm is applied by the caller.
    """
    _check(n_boot, ci)
    a, b = _as_2d(nll_a), _as_2d(nll_b)
    if a.shape[1] != b.shape[1]:
        raise ValueError("nll_a and nll_b must cover the same windows")
    n = a.shape[1]
    abar, bbar = _nan_seed_mean(a), _nan_seed_mean(b)
    paired = ~np.isnan(abar) & ~np.isnan(bbar)
    names = ["all"] + (list(regimes) if labels is not None else [])
    n_reg = len(regimes) if labels is not None else 1
    lab = _labels(labels, n, n_reg) if labels is not None else np.zeros(n, dtype=np.int64)
    ev = np.asarray(events).ravel()
    if ev.shape[0] != n:
        raise ValueError("events must have one entry per window")
    codes, unit_ids = encode_units(ev[paired])
    n_units = len(unit_ids)
    lab_p = lab[paired]
    x = np.vstack([abar[paired], bbar[paired]])
    s, c = _unit_group_sums(x, codes, n_units, lab_p, n_reg)  # columns k * n_reg + r, k = 0 (a), 1 (b)
    reps = resample_sums(np.hstack([s, c]), n_boot, seed)
    s_rep = reps[:, : 2 * n_reg].reshape(-1, 2, n_reg)
    c_rep = reps[:, 2 * n_reg :].reshape(-1, 2, n_reg)
    has = np.bincount(codes.astype(np.int64) * n_reg + lab_p, minlength=n_units * n_reg).reshape(n_units, n_reg) > 0
    rows = []
    for g, name in enumerate(names):
        r = None if g == 0 else g - 1
        in_group = np.ones(lab_p.shape[0], dtype=bool) if r is None else lab_p == r
        with np.errstate(invalid="ignore", divide="ignore"):
            if r is None:
                rep_ab = s_rep.sum(axis=2) / c_rep.sum(axis=2)
            else:
                rep_ab = s_rep[:, :, r] / c_rep[:, :, r]
        diff_reps = rep_ab[:, 0] - rep_ab[:, 1]
        mean_a = float(x[0, in_group].mean()) if in_group.any() else float("nan")
        mean_b = float(x[1, in_group].mean()) if in_group.any() else float("nan")
        a_lo, a_hi, _ = _interval(rep_ab[:, 0], ci)
        b_lo, b_hi, _ = _interval(rep_ab[:, 1], ci)
        d_lo, d_hi, d_se = _interval(diff_reps, ci)
        bounds = np.array([a_lo, a_hi, b_lo, b_hi])
        overlap = bool(a_lo <= b_hi and b_lo <= a_hi) if np.all(np.isfinite(bounds)) else None
        rows.append(
            {
                "group": name,
                "n_windows": int(in_group.sum()),
                "n_events": int(n_units if r is None else has[:, r].sum()),
                "nll_a": mean_a,
                "nll_a_lo": a_lo,
                "nll_a_hi": a_hi,
                "nll_b": mean_b,
                "nll_b_lo": b_lo,
                "nll_b_hi": b_hi,
                "diff": mean_a - mean_b,
                "diff_lo": d_lo,
                "diff_hi": d_hi,
                "se": d_se,
                "pvalue": bootstrap_pvalue(diff_reps, 0.0) if in_group.any() else float("nan"),
                "ci_overlap": overlap,
                "n_seeds_a": int(a.shape[0]),
                "n_seeds_b": int(b.shape[0]),
            }
        )
    return pd.DataFrame(rows, columns=NLL_COLUMNS)
