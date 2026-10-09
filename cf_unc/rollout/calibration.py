"""Coverage, split conformal calibration and adaptive conformal inference (docs/design.md, section 11.4).

numpy / pandas only (no torch). Conventions:

* Split conformal quantile of n calibration scores: the order statistic of rank ``ceil((n + 1)(1 - alpha))``,
  ``+inf`` when the rank exceeds n (too few scores for the level). NaN scores are ignored.
* One-step intervals (E1 prediction tables): ``mu +- q`` for the absolute score ``|y - mu|`` and
  ``mu +- q sigma`` for the normalised score ``|y - mu| / sigma``.
* Score of a rollout interval ``[lo, hi]`` with median ``m`` (half-widths floored at 1 mm)::

      S = max((lo - y) / (m - lo), (y - hi) / (hi - m))          negative inside the interval

  The interval with correction ``Q`` is ``[lo - Q (m - lo), hi + Q (hi - m)]``; it contains ``y`` iff ``S <= Q``
  (an interval with ``Q`` so negative that it would be inverted is empty; its width is reported as 0).
* ACI (Gibbs and Candes 2021): ``alpha_1 = alpha``; ``Q_e`` = empirical quantile of level ``1 - alpha_e`` of the
  scores seen so far (order statistic of rank ``ceil(level n)``, level clipped to [0, 1], so level 0 = the
  smallest and level 1 = the largest score: corrections are always finite; ``Q_e = 0`` while fewer than
  ``n_min`` scores were seen); ``err_e = 1[S_e > Q_e]``; ``alpha_{e+1} = alpha_e + gamma (alpha - err_e)``; the
  score is appended to the history after use. ``frozen``: level and history of a state are kept fixed.
* Coverage tables: every seed is calibrated separately; covered indicators and widths are averaged over the seeds
  per event, the PICP interval is an event bootstrap (percentile, ``cf_unc.eval.bootstrap.bootstrap_by_group``)
  of these event means; the bootstrap seed depends on ``boot_seed`` and the horizon only, so every model and
  variant is resampled with the same events.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from cf_unc.eval.bootstrap import bootstrap_by_group

__all__ = [
    "ACIState",
    "COVERAGE_COLUMNS",
    "HALF_WIDTH_FLOOR",
    "SCORES",
    "VARIANTS",
    "aci_quantile",
    "aci_run",
    "conformal_quantile",
    "corrected_interval",
    "coverage_table",
    "h34_verdict",
    "interval_score",
    "onestep_scores",
    "split_conformal_onestep",
    "static_correction",
]

VARIANTS = ("raw", "raw_linear", "static", "aci_frozen", "aci_online")
SCORES = ("absolute", "normalised")
HALF_WIDTH_FLOOR = 1e-3  # m, floor of the half-widths in the rollout interval score
COVERAGE_COLUMNS = (
    "horizon_s", "variant", "gamma", "picp", "picp_lo", "picp_hi", "width_mean", "width_median", "crps", "collided",
    "q_mean", "n_events", "n_seeds",
)  # fmt: skip
_RANK_TOL = 1e-9  # (n + 1)(1 - alpha) and level * n are exact integers more often than float arithmetic shows


def _f64(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _check_alpha(alpha: float) -> None:
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")


# ------------------------------------------------------------------------------------- split conformal
def conformal_quantile(scores: ArrayLike, alpha: float) -> float:
    """Split conformal quantile: order statistic of rank ``ceil((n + 1)(1 - alpha))`` of the non-NaN scores.

    ``+inf`` when the rank exceeds n (including n = 0).
    """
    _check_alpha(alpha)
    s = _f64(scores).ravel()
    s = s[~np.isnan(s)]
    n = s.size
    rank = math.ceil((n + 1) * (1.0 - float(alpha)) - _RANK_TOL)
    if n == 0 or rank > n:
        return math.inf
    return float(np.partition(s, rank - 1)[rank - 1])


def static_correction(scores_cal: ArrayLike, alpha: float) -> float:
    """Static correction ``Q`` of a horizon: the conformal quantile of the calibration scores."""
    return conformal_quantile(scores_cal, alpha)


def onestep_scores(mean: ArrayLike, var: ArrayLike, y: ArrayLike, score: str) -> np.ndarray:
    """Conformity scores (n,) of one-step predictions: ``|y - mu|`` or ``|y - mu| / sigma``; NaN where any input is.

    A zero sigma gives 0 for an exact prediction and +inf otherwise.
    """
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}, got {score!r}")
    mu, v, yy = np.broadcast_arrays(_f64(mean), _f64(var), _f64(y))
    resid = np.abs(yy - mu)
    if score == "absolute":
        out = resid.copy()
        out[np.isnan(v)] = np.nan
        return out
    sd = np.sqrt(np.maximum(v, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 0, resid / sd, np.where(resid > 0, np.inf, 0.0))
    out[np.isnan(resid) | np.isnan(v)] = np.nan
    return out


def split_conformal_onestep(
    mean_cal: ArrayLike,
    var_cal: ArrayLike,
    y_cal: ArrayLike,
    mean_eval: ArrayLike,
    var_eval: ArrayLike,
    y_eval: ArrayLike,
    alpha: float = 0.1,
    score: str = "normalised",
) -> dict[str, Any]:
    """Split conformal intervals of one-step predictions (means [m/s^2], variances [(m/s^2)^2]).

    Calibration rows with a NaN input are ignored. Returns ``covered`` (n_eval,) float 0 / 1 (NaN on invalid rows),
    ``width`` (n_eval,) [m/s^2] (``2 q`` or ``2 q sigma``; NaN on invalid rows), ``lo`` / ``hi`` (n_eval,),
    ``q`` (conformal quantile of the calibration scores) and ``n_cal`` (valid calibration rows).
    """
    s_cal = onestep_scores(mean_cal, var_cal, y_cal, score)
    q = conformal_quantile(s_cal, alpha)
    mu, v, y = np.broadcast_arrays(_f64(mean_eval), _f64(var_eval), _f64(y_eval))
    s_eval = onestep_scores(mu, v, y, score)
    valid = ~np.isnan(s_eval)
    half = np.full(mu.shape, q) if score == "absolute" else q * np.sqrt(np.maximum(v, 0.0))
    if score == "normalised" and math.isinf(q):
        half = np.full(mu.shape, math.inf)
    covered = np.where(valid, (s_eval <= q).astype(np.float64), np.nan)
    width = np.where(valid, 2.0 * half, np.nan)
    return {
        "covered": covered,
        "width": width,
        "lo": np.where(valid, mu - half, np.nan),
        "hi": np.where(valid, mu + half, np.nan),
        "q": q,
        "n_cal": int(np.count_nonzero(~np.isnan(s_cal))),
    }


# ----------------------------------------------------------------------------------- rollout intervals
def _half_widths(lo: np.ndarray, mid: np.ndarray, hi: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    return np.maximum(mid - lo, floor), np.maximum(hi - mid, floor)


def interval_score(y: ArrayLike, lo: ArrayLike, mid: ArrayLike, hi: ArrayLike, floor: float = HALF_WIDTH_FLOOR) -> np.ndarray:
    """``S = max((lo - y) / (m - lo), (y - hi) / (hi - m))`` with half-widths floored at ``floor`` [m].

    Negative inside the interval, 0 on its bounds, positive outside (in units of the half-width on that side).
    """
    yy, lo_a, mid_a, hi_a = np.broadcast_arrays(_f64(y), _f64(lo), _f64(mid), _f64(hi))
    d_lo, d_hi = _half_widths(lo_a, mid_a, hi_a, floor)
    return np.maximum((lo_a - yy) / d_lo, (yy - hi_a) / d_hi)


def corrected_interval(
    lo: ArrayLike, mid: ArrayLike, hi: ArrayLike, q: ArrayLike, floor: float = HALF_WIDTH_FLOOR
) -> tuple[np.ndarray, np.ndarray]:
    """``[lo - q (m - lo), hi + q (hi - m)]`` (half-widths floored as in :func:`interval_score`); q scalar or array."""
    lo_a, mid_a, hi_a, qq = np.broadcast_arrays(_f64(lo), _f64(mid), _f64(hi), _f64(q))
    d_lo, d_hi = _half_widths(lo_a, mid_a, hi_a, floor)
    return lo_a - qq * d_lo, hi_a + qq * d_hi


def _corrected_width(lo: np.ndarray, mid: np.ndarray, hi: np.ndarray, q: ArrayLike, floor: float) -> np.ndarray:
    a, b = corrected_interval(lo, mid, hi, q, floor)
    with np.errstate(invalid="ignore"):
        return np.maximum(b - a, 0.0)


# ------------------------------------------------------------------------------------------------ ACI
@dataclass
class ACIState:
    """State of an ACI run: target ``alpha``, step ``gamma``, ``n_min``, the level parameter ``alpha_t`` of the next
    event, the sorted ``history`` of the scores seen so far and ``n_events`` processed (non-NaN) events."""

    alpha: float
    gamma: float
    n_min: int
    alpha_t: float
    history: np.ndarray
    n_events: int = 0

    def quantile(self) -> float:
        """Correction ``Q`` for the next event."""
        return aci_quantile(self.history, 1.0 - self.alpha_t, self.n_min)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form (scores as a list)."""
        return {"alpha": self.alpha, "gamma": self.gamma, "n_min": self.n_min, "alpha_t": self.alpha_t,
                "history": [float(v) for v in self.history], "n_events": self.n_events}  # fmt: skip

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ACIState":
        return cls(
            alpha=float(data["alpha"]), gamma=float(data["gamma"]), n_min=int(data["n_min"]),
            alpha_t=float(data["alpha_t"]), history=np.sort(_f64(data["history"])), n_events=int(data["n_events"]),
        )  # fmt: skip


def aci_quantile(sorted_scores: ArrayLike, level: float, n_min: int = 20) -> float:
    """Empirical quantile of level ``level`` (clipped to [0, 1]) of sorted scores: rank ``ceil(level n)`` in [1, n].

    0 while fewer than ``n_min`` scores (or none) are available.
    """
    s = np.asarray(sorted_scores, dtype=np.float64)
    n = s.size
    if n == 0 or n < int(n_min):
        return 0.0
    lev = min(max(float(level), 0.0), 1.0)
    rank = min(max(math.ceil(lev * n - _RANK_TOL), 1), n)
    return float(s[rank - 1])


def aci_run(
    scores: ArrayLike,
    alpha: float,
    gamma: float,
    n_min: int = 20,
    history: ArrayLike | None = None,
    state: ACIState | None = None,
    frozen: bool = False,
) -> dict[str, Any]:
    """Adaptive conformal inference over ``scores`` (n,) in the given order (module docstring).

    history: scores known before the first event (no recursion over them); state: continue a previous run (its
    ``alpha``, ``gamma``, ``n_min`` must match); frozen: keep level and history fixed (no update). NaN scores give
    ``covered = NaN`` and no update. Returns per event ``q`` (correction used), ``alpha_t`` (level parameter used),
    ``covered`` (1.0 iff ``S <= q``) and the final ``state`` (a new object; the input state is not modified).
    """
    _check_alpha(alpha)
    if not gamma >= 0.0:
        raise ValueError("gamma must be >= 0")
    if state is not None and history is not None:
        raise ValueError("give either history or state")
    if state is None:
        init = np.sort(_f64(history if history is not None else []).ravel())
        state = ACIState(float(alpha), float(gamma), int(n_min), float(alpha), init[~np.isnan(init)], 0)
    elif (state.alpha, state.gamma, state.n_min) != (float(alpha), float(gamma), int(n_min)):
        raise ValueError(f"state was run with alpha {state.alpha}, gamma {state.gamma}, n_min {state.n_min}")
    s = _f64(scores).ravel()
    n = s.size
    q, levels, covered = np.empty(n), np.empty(n), np.empty(n)
    hist = [float(v) for v in state.history]
    alpha_t, done = float(state.alpha_t), int(state.n_events)
    fixed_q = aci_quantile(hist, 1.0 - alpha_t, n_min) if frozen else None
    for i in range(n):
        q_i = fixed_q if fixed_q is not None else aci_quantile(hist, 1.0 - alpha_t, n_min)
        q[i], levels[i] = q_i, alpha_t
        if math.isnan(s[i]):
            covered[i] = np.nan
            continue
        err = 1.0 if s[i] > q_i else 0.0
        covered[i] = 1.0 - err
        if not frozen:
            alpha_t += gamma * (alpha - err)
            bisect.insort(hist, float(s[i]))
            done += 1
    final = ACIState(float(alpha), float(gamma), int(n_min), alpha_t, np.asarray(hist, dtype=np.float64), done)
    return {"q": q, "alpha_t": levels, "covered": covered, "state": final}


# -------------------------------------------------------------------------------------- coverage tables
def _horizon_arrays(table: pd.DataFrame, horizon: float, codes: np.ndarray) -> dict[str, np.ndarray]:
    """Columns of one horizon, rows aligned with the event codes ``codes`` of the events table."""
    rows = table[np.isclose(table["horizon_s"].to_numpy(np.float64), horizon)]
    ev = rows["event"].to_numpy(np.int64)
    if ev.size != codes.size or not np.array_equal(np.sort(ev), np.sort(codes)):
        raise ValueError(f"rollout table at {horizon} s holds other events than the events table")
    if not np.array_equal(ev, codes):
        rows = rows.set_index("event").loc[codes].reset_index()
    names = ("truth_s", "s_q05", "s_q50", "s_q95", "s_lin05", "s_lin95", "crps_s", "collided")
    return {c: rows[c].to_numpy(np.float64) for c in names}


def _boot_seed(boot_seed: int, horizon: float) -> int:
    return int(np.random.SeedSequence([abs(int(boot_seed)), int(round(1000 * float(horizon)))]).generate_state(1)[0])


def _mean0(a: np.ndarray) -> np.ndarray:
    """Mean over axis 0 of the non-NaN entries (NaN where there is none), without warnings."""
    ok = ~np.isnan(a)
    count = ok.sum(axis=0)
    total = np.where(ok, a, 0.0).sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _mean(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else float("nan")


def _summary_row(cov: np.ndarray, width: np.ndarray, seed: int, n_boot: int, ci: float) -> dict[str, float]:
    """PICP of the seed-averaged per-event indicators with an event bootstrap interval, width mean / median."""
    per_event, w_event = _mean0(cov), _mean0(width)
    ok = ~np.isnan(per_event)
    boot = bootstrap_by_group(per_event[ok], np.flatnonzero(ok), n_boot=n_boot, seed=seed, ci=ci)
    w = w_event[~np.isnan(w_event)]
    return {
        "picp": float(boot["estimate"]),
        "picp_lo": float(boot["lo"]),
        "picp_hi": float(boot["hi"]),
        "width_mean": float(np.mean(w)) if w.size else float("nan"),
        "width_median": float(np.median(w)) if w.size else float("nan"),
        "n_events": int(ok.sum()),
    }


def coverage_table(
    eval_tables: Mapping[int, pd.DataFrame],
    eval_events: pd.DataFrame,
    cal_tables: Mapping[int, pd.DataFrame] | None = None,
    cal_events: pd.DataFrame | None = None,
    *,
    alpha: float = 0.1,
    gammas: Sequence[float] = (0.01,),
    n_min: int = 20,
    n_boot: int = 1000,
    ci: float = 0.95,
    boot_seed: int = 0,
    floor: float = HALF_WIDTH_FLOOR,
) -> pd.DataFrame:
    """Coverage of the rollout intervals of one model, target and data set per horizon and variant.

    eval_tables / cal_tables: seed -> rollout table (``cf_unc.rollout.tables``) of the evaluated and of the
    calibration data set; eval_events / cal_events: their events tables (``event`` codes in table order, ACI order
    ``order``). Variants: ``raw`` = [s_q05, s_q95]; ``raw_linear`` = [s_lin05, s_lin95]; with calibration tables
    also ``static`` (conformal quantile of the calibration scores of the same seed and horizon), ``aci_frozen`` and
    ``aci_online`` for every gamma (ACI over the calibration events in their ``order``, then frozen or continued
    over the evaluated events in their ``order``). Seeds present in both inputs are used. Returns COVERAGE_COLUMNS:
    ``crps`` (raw rows only) and ``collided`` (every row) are event means of the seed means of ``crps_s`` and
    ``collided``; ``q_mean`` is the seed mean of the static / frozen correction and the mean correction of the
    online run; widths in m (a corrected interval that would be inverted counts as 0).
    """
    seeds = sorted(eval_tables) if cal_tables is None else sorted(set(eval_tables) & set(cal_tables))
    if not seeds:
        return pd.DataFrame(columns=list(COVERAGE_COLUMNS))
    codes = eval_events["event"].to_numpy(np.int64)
    perm = np.argsort(eval_events["order"].to_numpy(np.int64), kind="stable")
    calibrated = cal_tables is not None
    if calibrated:
        if cal_events is None:
            raise ValueError("cal_events is needed with cal_tables")
        cal_codes = cal_events["event"].to_numpy(np.int64)
        cal_perm = np.argsort(cal_events["order"].to_numpy(np.int64), kind="stable")
    horizons = np.unique(eval_tables[seeds[0]]["horizon_s"].to_numpy(np.float64))
    n_s, n_e = len(seeds), codes.size
    out: list[dict[str, Any]] = []
    for h in horizons:
        keys = ["raw", "raw_linear"]
        if calibrated:
            keys.append("static")
            keys += [f"{v}@{g}" for g in gammas for v in ("aci_frozen", "aci_online")]
        cov = {k: np.full((n_s, n_e), np.nan) for k in keys}
        width = {k: np.full((n_s, n_e), np.nan) for k in keys}
        qs: dict[str, list[float]] = {k: [] for k in keys}
        crps = np.full((n_s, n_e), np.nan)
        collided = np.full((n_s, n_e), np.nan)
        for i, seed in enumerate(seeds):
            a = _horizon_arrays(eval_tables[seed], h, codes)
            y, lo, mid, hi = a["truth_s"], a["s_q05"], a["s_q50"], a["s_q95"]
            score = interval_score(y, lo, mid, hi, floor)
            cov["raw"][i] = (score <= 0.0).astype(np.float64)
            width["raw"][i] = hi - lo
            cov["raw_linear"][i] = ((a["s_lin05"] <= y) & (y <= a["s_lin95"])).astype(np.float64)
            width["raw_linear"][i] = a["s_lin95"] - a["s_lin05"]
            crps[i], collided[i] = a["crps_s"], a["collided"]
            if not calibrated:
                continue
            c = _horizon_arrays(cal_tables[seed], h, cal_codes)  # type: ignore[index]
            cal_score = interval_score(c["truth_s"], c["s_q05"], c["s_q50"], c["s_q95"], floor)
            q_static = static_correction(cal_score, alpha)
            cov["static"][i] = (score <= q_static).astype(np.float64)
            width["static"][i] = _corrected_width(lo, mid, hi, q_static, floor)
            qs["static"].append(q_static)
            for g in gammas:
                run = aci_run(cal_score[cal_perm], alpha, g, n_min)
                for variant, frozen in (("aci_frozen", True), ("aci_online", False)):
                    res = aci_run(score[perm], alpha, g, n_min, state=run["state"], frozen=frozen)
                    k = f"{variant}@{g}"
                    q_e = np.empty(n_e)
                    q_e[perm] = res["q"]
                    cov[k][i, perm] = res["covered"]
                    width[k][i] = _corrected_width(lo, mid, hi, q_e, floor)
                    qs[k].append(float(np.mean(q_e)))
        seed_boot = _boot_seed(boot_seed, h)
        common = {
            "horizon_s": float(h),
            "collided": _mean(_mean0(collided)),
            "n_seeds": n_s,
        }
        for k in keys:
            variant, _, g = k.partition("@")
            row = {**common, "variant": variant, "gamma": float(g) if g else float("nan")}
            row.update(_summary_row(cov[k], width[k], seed_boot, n_boot, ci))
            row["crps"] = _mean(_mean0(crps)) if variant == "raw" else float("nan")
            row["q_mean"] = float(np.mean(qs[k])) if qs[k] else float("nan")
            out.append(row)
    return pd.DataFrame(out, columns=list(COVERAGE_COLUMNS))


# --------------------------------------------------------------------------------------------- H3.4
def _all(values: np.ndarray, test) -> bool:  # noqa: ANN001
    v = np.asarray(values, dtype=np.float64)
    return bool(v.size and np.all(np.isfinite(v)) and np.all(test(v)))


def _any(values: np.ndarray, test) -> bool:  # noqa: ANN001
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return bool(v.size and np.any(test(v)))


def h34_verdict(
    onestep_picp: float,
    raw_picp: ArrayLike,
    corrected_picp: ArrayLike,
    *,
    onestep_ci: tuple[float, float] | None = None,
    raw_ci: tuple[ArrayLike, ArrayLike] | None = None,
    corrected_ci: tuple[ArrayLike, ArrayLike] | None = None,
    nominal: float = 0.90,
    onestep_tol: float = 0.03,
    raw_max: float = 0.80,
    corrected_min: float = 0.87,
    refute_min: float = 0.87,
) -> dict[str, Any]:
    """Verdict of H3.4 for one model, target and data set (design.md 11.4).

    Parts: (i) one-step conformal coverage on highD test within ``nominal +- onestep_tol``; (ii) raw rollout
    coverage below ``raw_max`` at some horizon; (iii) corrected coverage >= ``corrected_min`` at every horizon.
    ``supported`` = (i) and (ii) and (iii); ``refuted`` = raw coverage >= ``refute_min`` at every horizon;
    ``inconclusive`` otherwise (also when a needed number is NaN). ``verdict`` uses the point estimates;
    ``verdict_ci`` (only with all three intervals) the bootstrap intervals: (i) interval inside the band, (ii) upper
    bound below ``raw_max`` at some horizon, (iii) lower bound >= ``corrected_min`` at every horizon, refuted when
    the lower bound of the raw coverage is >= ``refute_min`` at every horizon.
    """
    raw, corr = _f64(raw_picp).ravel(), _f64(corrected_picp).ravel()
    one = float(onestep_picp)

    def decide(p1: bool, p2: bool, p3: bool, refuted: bool) -> str:
        if refuted:
            return "refuted"
        return "supported" if (p1 and p2 and p3) else "inconclusive"

    part_i = bool(math.isfinite(one) and abs(one - nominal) <= onestep_tol + 1e-12)
    part_ii = _any(raw, lambda v: v < raw_max)
    part_iii = _all(corr, lambda v: v >= corrected_min)
    refuted = _all(raw, lambda v: v >= refute_min)
    out: dict[str, Any] = {
        "part_i": part_i, "part_ii": part_ii, "part_iii": part_iii, "refuted_raw": refuted,
        "verdict": decide(part_i, part_ii, part_iii, refuted),
        "part_i_ci": None, "part_ii_ci": None, "part_iii_ci": None, "refuted_raw_ci": None, "verdict_ci": None,
    }  # fmt: skip
    if onestep_ci is not None and raw_ci is not None and corrected_ci is not None:
        lo1, hi1 = (float(x) for x in onestep_ci)
        ci_i = bool(math.isfinite(lo1) and math.isfinite(hi1) and lo1 >= nominal - onestep_tol - 1e-12
                    and hi1 <= nominal + onestep_tol + 1e-12)  # fmt: skip
        ci_ii = _any(_f64(raw_ci[1]).ravel(), lambda v: v < raw_max)
        ci_iii = _all(_f64(corrected_ci[0]).ravel(), lambda v: v >= corrected_min)
        ci_ref = _all(_f64(raw_ci[0]).ravel(), lambda v: v >= refute_min)
        out.update(part_i_ci=ci_i, part_ii_ci=ci_ii, part_iii_ci=ci_iii, refuted_raw_ci=ci_ref,
                   verdict_ci=decide(ci_i, ci_ii, ci_iii, ci_ref))  # fmt: skip
    return out
