"""Coverage, split conformal calibration and adaptive conformal inference (docs/design.md, section 11.4).

numpy / pandas only (no torch). Conventions:

* Split conformal quantile of n calibration scores: the order statistic of rank ``ceil((n + 1)(1 - alpha))``,
  ``+inf`` when the rank exceeds n (too few scores for the level). NaN scores are ignored.
* One-step intervals (E1 prediction tables): ``mu +- q`` for the absolute score ``|y - mu|`` and
  ``mu +- q sigma`` for the normalised score ``|y - mu| / sigma``.
* Score of a rollout interval ``[lo, hi]`` with median ``m`` (half-widths floored at 1 mm)::

      S = max((lo - y) / (m - lo), (y - hi) / (hi - m))          negative inside the interval, never below -1

  The interval with correction ``Q`` is ``[lo - Q (m - lo), hi + Q (hi - m)]``; it contains ``y`` iff ``S <= Q``.
  ``Q = +inf`` is the whole line (infinite width, always covered); ``Q < -1`` (``-inf`` included) is the empty set
  (width 0, never covered); ``-1 <= Q < 0`` shrinks the interval towards the median.
* ACI (Gibbs and Candes 2021, section 4.1; review item R1): ``alpha_1 = alpha``; the correction of event e is the
  empirical quantile ``Q_e`` of level ``1 - alpha_e`` of the scores seen so far, with the theoretical behaviour at
  the ends: level ``>= 1`` (``alpha_e <= 0``) gives ``Q_e = +inf`` (the whole line), level ``<= 0`` (``alpha_e >= 1``)
  gives ``Q_e = -inf`` (the empty set); inside (0, 1) ``Q_e = 0`` while fewer than ``n_min`` scores were seen
  (warm-up: the raw interval), else the order statistic of rank ``ceil(level n)``. ``err_e = 1[S_e > Q_e]``;
  ``alpha_{e+1} = alpha_e + gamma (alpha - err_e)`` is never clipped; the score is appended to the history after
  use. Because an interval issued at ``alpha_e <= 0`` always covers and one issued at ``alpha_e >= 1`` never does,
  ``alpha_e`` stays in ``[-gamma, 1 + gamma]`` and over T events ``|mean(err) - alpha| <= (max(alpha_1, 1 - alpha_1)
  + gamma) / (T gamma)`` for ANY sequence of scores (the warm-up does not break this: the bound only uses the
  behaviour at the two ends). ``frozen``: level and history of a state are kept fixed.
* Heuristic finite variant (``finite_quantile=True``; the tables of the submitted manuscript): the level is clipped
  to [0, 1] after the warm-up, so level 0 = the smallest and level 1 = the largest score seen and every correction
  is finite. The long-run guarantee does not hold for it (scores 1, 2, ..., 1000 after a history [0], alpha 0.1,
  gamma 0.01: coverage 0, ``alpha_t`` -> -8.9); it is kept to reproduce the old tables.
* Prequential streams (review item R8, :func:`stream_run`): the evaluated events are issued in a given order at
  times ``issue_at``; the label (score) of event j becomes available at ``available_at[j]`` and is used (ACI update,
  or appended to the history of the static correction) before the first event issued at or after that time.
  Immediate feedback (the label is used before the next event) is exactly :func:`aci_run`. The prequential static
  correction is the split conformal quantile of all scores seen so far (no learning rate), so that ACI and the
  static correction use the same labels.
* Coverage tables: every seed is calibrated separately; covered indicators and widths are averaged over the seeds
  per event, the PICP interval is an event bootstrap (percentile, ``cf_unc.eval.bootstrap.bootstrap_by_group``)
  of these event means, and a second interval resamples followers (the ``driver`` codes of the events table; every
  event of a drawn follower is kept); the bootstrap seed depends on ``boot_seed`` and the horizon only, so every
  model and variant is resampled with the same events.
"""

from __future__ import annotations

import bisect
import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from cf_unc.eval.bootstrap import bootstrap_by_group

__all__ = [
    "ACIState",
    "COVERAGE_COLUMNS",
    "FIRST_COUNTS",
    "H34_STATES",
    "HALF_WIDTH_FLOOR",
    "SCORES",
    "STREAM_VARIANTS",
    "VARIANTS",
    "aci_quantile",
    "aci_run",
    "conformal_quantile",
    "corrected_interval",
    "coverage_table",
    "delayed_availability",
    "h34_verdict",
    "interval_score",
    "onestep_scores",
    "split_conformal_onestep",
    "static_correction",
    "stream_run",
]

VARIANTS = ("raw", "raw_linear", "static", "aci_frozen", "aci_online")
# Prequential variants of the information-budget analysis (R8): chronological order of the evaluated events, labels
# delayed by the horizon, and the static correction recalibrated on the same labels as ACI.
STREAM_VARIANTS = ("aci_online_chrono", "aci_online_delayed", "static_prequential", "static_prequential_delayed")
SCORES = ("absolute", "normalised")
HALF_WIDTH_FLOOR = 1e-3  # m, floor of the half-widths in the rollout interval score
FIRST_COUNTS = (50, 100)  # coverage over the first events of an evaluated stream (the switch of domain)
COVERAGE_COLUMNS = (
    "horizon_s", "variant", "gamma", "picp", "picp_lo", "picp_hi", "width_mean", "width_median", "crps", "collided",
    "q_mean", "n_events", "n_seeds",
    "share_infinite", "share_empty", "width_finite_mean", "width_finite_median", "picp_first50", "picp_first100",
    "delay_events_mean", "picp_lo_follower", "picp_hi_follower", "n_followers",
)  # fmt: skip
H34_STATES = ("holds", "fails", "straddles", "missing")  # state of a part of H3.4 under the interval rule
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


def _conformal_sorted(sorted_scores: Sequence[float], alpha: float) -> float:
    """:func:`conformal_quantile` of a sorted list without NaN (the prequential static correction)."""
    n = len(sorted_scores)
    rank = math.ceil((n + 1) * (1.0 - float(alpha)) - _RANK_TOL)
    return math.inf if n == 0 or rank > n else float(sorted_scores[rank - 1])


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

    Negative inside the interval, 0 on its bounds, positive outside (in units of the half-width on that side);
    never below -1 (its value at the median).
    """
    yy, lo_a, mid_a, hi_a = np.broadcast_arrays(_f64(y), _f64(lo), _f64(mid), _f64(hi))
    d_lo, d_hi = _half_widths(lo_a, mid_a, hi_a, floor)
    return np.maximum((lo_a - yy) / d_lo, (yy - hi_a) / d_hi)


def corrected_interval(
    lo: ArrayLike, mid: ArrayLike, hi: ArrayLike, q: ArrayLike, floor: float = HALF_WIDTH_FLOOR
) -> tuple[np.ndarray, np.ndarray]:
    """``[lo - q (m - lo), hi + q (hi - m)]`` (half-widths floored as in :func:`interval_score`); q scalar or array.

    ``q = +inf`` gives ``(-inf, +inf)``; ``q = -inf`` the inverted (empty) pair ``(+inf, -inf)``.
    """
    lo_a, mid_a, hi_a, qq = np.broadcast_arrays(_f64(lo), _f64(mid), _f64(hi), _f64(q))
    d_lo, d_hi = _half_widths(lo_a, mid_a, hi_a, floor)
    return lo_a - qq * d_lo, hi_a + qq * d_hi


def _corrected_width(lo: np.ndarray, mid: np.ndarray, hi: np.ndarray, q: ArrayLike, floor: float) -> np.ndarray:
    """Width of the corrected interval: +inf for ``q = +inf``, 0 for an empty (inverted) interval."""
    a, b = corrected_interval(lo, mid, hi, q, floor)
    with np.errstate(invalid="ignore"):
        return np.maximum(b - a, 0.0)


# ------------------------------------------------------------------------------------------------ ACI
@dataclass
class ACIState:
    """State of an ACI run: target ``alpha``, step ``gamma``, ``n_min``, the level parameter ``alpha_t`` of the next
    event, the sorted ``history`` of the scores seen so far, ``n_events`` processed (non-NaN) events and
    ``finite_quantile`` (the heuristic finite variant of :func:`aci_quantile`)."""

    alpha: float
    gamma: float
    n_min: int
    alpha_t: float
    history: np.ndarray
    n_events: int = 0
    finite_quantile: bool = False

    def quantile(self) -> float:
        """Correction ``Q`` for the next event."""
        return aci_quantile(self.history, 1.0 - self.alpha_t, self.n_min, self.finite_quantile)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form (scores as a list)."""
        return {"alpha": self.alpha, "gamma": self.gamma, "n_min": self.n_min, "alpha_t": self.alpha_t,
                "history": [float(v) for v in self.history], "n_events": self.n_events,
                "finite_quantile": bool(self.finite_quantile)}  # fmt: skip

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ACIState":
        return cls(
            alpha=float(data["alpha"]), gamma=float(data["gamma"]), n_min=int(data["n_min"]),
            alpha_t=float(data["alpha_t"]), history=np.sort(_f64(data["history"])), n_events=int(data["n_events"]),
            finite_quantile=bool(data.get("finite_quantile", False)),
        )  # fmt: skip


def aci_quantile(sorted_scores: ArrayLike, level: float, n_min: int = 20, finite_quantile: bool = False) -> float:
    """Correction of level ``level`` from the sorted scores seen so far (module docstring).

    Default (Gibbs and Candes): ``+inf`` for a level >= 1, ``-inf`` for a level <= 0; inside (0, 1): 0 while fewer
    than ``n_min`` scores (or none) are available, else the order statistic of rank ``ceil(level n)``.
    ``finite_quantile`` (heuristic variant of the submitted tables): 0 while fewer than ``n_min`` scores (or none),
    else the level clipped to [0, 1] and the rank to [1, n]: always finite.
    """
    s = np.asarray(sorted_scores, dtype=np.float64)
    n = s.size
    lev = float(level)
    if not finite_quantile:
        if lev >= 1.0:
            return math.inf
        if lev <= 0.0:
            return -math.inf
    if n == 0 or n < int(n_min):
        return 0.0
    lev = min(max(lev, 0.0), 1.0)
    rank = min(max(math.ceil(lev * n - _RANK_TOL), 1), n)
    return float(s[rank - 1])


def _start_state(
    alpha: float, gamma: float, n_min: int, history: ArrayLike | None, state: ACIState | None,
    finite_quantile: bool | None,
) -> ACIState:  # fmt: skip
    """The state an ACI run starts from (a new one from ``history``, or ``state`` after checking its settings)."""
    _check_alpha(alpha)
    if not gamma >= 0.0:
        raise ValueError("gamma must be >= 0")
    if state is not None and history is not None:
        raise ValueError("give either history or state")
    if state is None:
        init = np.sort(_f64(history if history is not None else []).ravel())
        return ACIState(float(alpha), float(gamma), int(n_min), float(alpha), init[~np.isnan(init)], 0,
                        bool(finite_quantile))  # fmt: skip
    if (state.alpha, state.gamma, state.n_min) != (float(alpha), float(gamma), int(n_min)):
        raise ValueError(f"state was run with alpha {state.alpha}, gamma {state.gamma}, n_min {state.n_min}")
    if finite_quantile is not None and bool(finite_quantile) != bool(state.finite_quantile):
        raise ValueError(f"state was run with finite_quantile={state.finite_quantile}")
    return state


def aci_run(
    scores: ArrayLike,
    alpha: float,
    gamma: float,
    n_min: int = 20,
    history: ArrayLike | None = None,
    state: ACIState | None = None,
    frozen: bool = False,
    finite_quantile: bool | None = None,
) -> dict[str, Any]:
    """Adaptive conformal inference over ``scores`` (n,) in the given order (module docstring).

    history: scores known before the first event (no recursion over them); state: continue a previous run (its
    ``alpha``, ``gamma``, ``n_min`` and ``finite_quantile`` must match); frozen: keep level and history fixed (no
    update); finite_quantile: the heuristic finite variant (default False; with a state: the state's). NaN scores
    give ``covered = NaN`` and no update. Returns per event ``q`` (correction used; +-inf at the ends), ``alpha_t``
    (level parameter used), ``covered`` (1.0 iff ``S <= q``), ``infinite`` / ``empty`` (bool: ``q = +inf`` /
    ``q = -inf``) and the final ``state`` (a new object; the input state is not modified).
    """
    state = _start_state(alpha, gamma, n_min, history, state, finite_quantile)
    fq = bool(state.finite_quantile)
    s = _f64(scores).ravel()
    n = s.size
    q, levels, covered = np.empty(n), np.empty(n), np.empty(n)
    hist = [float(v) for v in state.history]
    alpha_t, done = float(state.alpha_t), int(state.n_events)
    fixed_q = aci_quantile(hist, 1.0 - alpha_t, n_min, fq) if frozen else None
    for i in range(n):
        q_i = fixed_q if fixed_q is not None else aci_quantile(hist, 1.0 - alpha_t, n_min, fq)
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
    final = ACIState(float(alpha), float(gamma), int(n_min), alpha_t, np.asarray(hist, dtype=np.float64), done, fq)
    return {"q": q, "alpha_t": levels, "covered": covered, "infinite": np.isposinf(q), "empty": np.isneginf(q),
            "state": final}  # fmt: skip


def delayed_availability(
    n: int, horizon_s: float, issue_times: ArrayLike | None = None, events_per_s: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """``(issue_at, available_at)`` (n,) of a stream of n events in issue order whose labels arrive ``horizon_s``
    after the interval is issued.

    With comparable ``issue_times`` [s] (non-decreasing, all finite): ``available_at = issue_times + horizon_s``.
    Without them (None or any NaN) the delay is counted in events: ``k = ceil(horizon_s * events_per_s)`` further
    events are issued before the label arrives (``issue_at = 0, 1, ...``, ``available_at = issue_at + k + 1``).
    """
    t = None if issue_times is None else _f64(issue_times).ravel()
    if t is not None and t.size == n and np.all(np.isfinite(t)):
        if np.any(np.diff(t) < 0):
            raise ValueError("issue_times must be non-decreasing (events in issue order)")
        return t, t + float(horizon_s)
    k = max(0, math.ceil(float(horizon_s) * float(events_per_s) - _RANK_TOL))
    pos = np.arange(n, dtype=np.float64)
    return pos, pos + k + 1.0


def stream_run(
    scores: ArrayLike,
    alpha: float,
    *,
    method: str = "aci",
    gamma: float = 0.01,
    n_min: int = 20,
    issue_at: ArrayLike | None = None,
    available_at: ArrayLike | None = None,
    history: ArrayLike | None = None,
    state: ACIState | None = None,
    finite_quantile: bool | None = None,
) -> dict[str, Any]:
    """Prequential calibration of a stream of events given in issue order, with delayed labels (module docstring).

    method ``aci``: the ACI recursion of :func:`aci_run` (``state`` or ``history`` as there); ``static``: the split
    conformal quantile of every score seen so far (``history`` = the calibration scores, or the history of
    ``state``), no learning rate. issue_at / available_at (n,): issue time of every event (non-decreasing) and the time
    from which its label can be used; both None: immediate feedback (the label is used before the next event), which
    reproduces :func:`aci_run` exactly. Labels are used in the order of ``available_at`` (ties by issue order).
    Returns per event ``q``, ``alpha_t`` (NaN for ``static``), ``covered`` (NaN for a NaN score), ``infinite``,
    ``empty``, ``delay_events`` (events issued after this one before its label was used: 0 for immediate feedback;
    NaN when the stream ended first or the score is NaN) and the final ``history`` (sorted scores) and, for ``aci``,
    ``state``.
    """
    if method not in ("aci", "static"):
        raise ValueError(f"method must be 'aci' or 'static', got {method!r}")
    s = _f64(scores).ravel()
    n = s.size
    if (issue_at is None) != (available_at is None):
        raise ValueError("give both issue_at and available_at, or neither")
    if issue_at is None:
        issue = np.arange(n, dtype=np.float64)
        avail = issue + 1.0
    else:
        issue, avail = _f64(issue_at).ravel(), _f64(available_at).ravel()
    if issue.size != n or avail.size != n:
        raise ValueError("issue_at and available_at need one entry per score")
    if np.any(np.diff(issue) < 0):
        raise ValueError("issue_at must be non-decreasing (events in issue order)")
    if method == "aci":
        st = _start_state(alpha, gamma, n_min, history, state, finite_quantile)
        fq, alpha_t, done = bool(st.finite_quantile), float(st.alpha_t), int(st.n_events)
        hist = [float(v) for v in st.history]
    else:
        _check_alpha(alpha)
        base = _f64(state.history if state is not None else (history if history is not None else [])).ravel()
        hist = sorted(float(v) for v in base[~np.isnan(base)])
        fq, alpha_t, done = False, float("nan"), 0
    q, levels = np.empty(n), np.full(n, np.nan)
    covered, delay = np.full(n, np.nan), np.full(n, np.nan)
    pending: list[tuple[float, int]] = []

    def use(j: int, i: int) -> None:
        nonlocal alpha_t, done
        delay[j] = i - j - 1
        if method == "aci":
            alpha_t += gamma * (alpha - (1.0 - covered[j]))
            done += 1
        bisect.insort(hist, float(s[j]))

    for i in range(n):
        while pending and pending[0][0] <= issue[i]:
            use(heapq.heappop(pending)[1], i)
        if method == "aci":
            q_i = aci_quantile(hist, 1.0 - alpha_t, n_min, fq)
            levels[i] = alpha_t
        else:
            q_i = _conformal_sorted(hist, alpha)
        q[i] = q_i
        if not math.isnan(s[i]):
            covered[i] = 1.0 if s[i] <= q_i else 0.0
            heapq.heappush(pending, (float(avail[i]), i))
    late = delay.copy()
    while pending:  # labels that arrive after the last event still update the final state
        use(heapq.heappop(pending)[1], n)
    out: dict[str, Any] = {"q": q, "alpha_t": levels, "covered": covered, "infinite": np.isposinf(q),
                           "empty": np.isneginf(q), "delay_events": late,
                           "history": np.asarray(hist, dtype=np.float64)}  # fmt: skip
    if method == "aci":
        out["state"] = ACIState(float(alpha), float(gamma), int(n_min), alpha_t, out["history"], done, fq)
    return out


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
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _mean(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else float("nan")


@dataclass
class _Acc:
    """Per (seed, event) arrays of one variant at one horizon, events in the order of the events table."""

    cov: np.ndarray
    width: np.ndarray
    q: np.ndarray
    q_means: list[float] = field(default_factory=list)  # per seed: the correction (static, frozen) or the mean finite one
    issue: np.ndarray | None = None  # rows of the events table in issue order (online and prequential variants)
    delay: np.ndarray | None = None  # (seeds, events) events issued before the label was used

    @classmethod
    def empty(cls, n_s: int, n_e: int) -> "_Acc":
        return cls(np.full((n_s, n_e), np.nan), np.full((n_s, n_e), np.nan), np.full((n_s, n_e), np.nan))

    def put(self, i: int, rows: np.ndarray, res: Mapping[str, Any], lo: np.ndarray, mid: np.ndarray, hi: np.ndarray,
            floor: float, stream: bool) -> None:  # fmt: skip
        """Store the result of a run over the events ``rows`` (issue order) of seed ``i``, in table order."""
        q_e = np.empty(rows.size)
        q_e[rows] = res["q"]
        self.cov[i, rows] = res["covered"]
        self.width[i] = _corrected_width(lo, mid, hi, q_e, floor)
        self.q[i] = q_e
        finite = q_e[np.isfinite(q_e)]
        self.q_means.append(float(np.mean(finite)) if finite.size else float("nan"))
        if stream:
            self.issue = rows
            if self.delay is None:
                self.delay = np.full(self.cov.shape, np.nan)
            d = res["delay_events"] if "delay_events" in res else np.zeros(rows.size)
            self.delay[i, rows] = np.where(np.isnan(res["covered"]), np.nan, d)


def _summary_row(acc: _Acc, seed: int, n_boot: int, ci: float, followers: np.ndarray | None) -> dict[str, float]:
    """PICP of the seed-averaged per-event indicators with an event bootstrap interval (and a follower-cluster
    one), widths over all and over the finite intervals, shares of infinite and empty intervals, coverage of the
    first events of a stream and the mean label delay."""
    per_event, w_event = _mean0(acc.cov), _mean0(acc.width)
    ok = ~np.isnan(per_event)
    boot = bootstrap_by_group(per_event[ok], np.flatnonzero(ok), n_boot=n_boot, seed=seed, ci=ci)
    w = w_event[~np.isnan(w_event)]
    valid = ~np.isnan(acc.cov)
    wf = _mean0(np.where(valid & np.isfinite(acc.width), acc.width, np.nan))
    wf = wf[~np.isnan(wf)]
    nan = float("nan")
    out: dict[str, float] = {
        "picp": float(boot["estimate"]),
        "picp_lo": float(boot["lo"]),
        "picp_hi": float(boot["hi"]),
        "width_mean": float(np.mean(w)) if w.size else nan,
        "width_median": float(np.median(w)) if w.size else nan,
        "n_events": int(ok.sum()),
        "share_infinite": float(np.mean(np.isposinf(acc.q[valid]))) if valid.any() else nan,
        "share_empty": float(np.mean(acc.q[valid] < -1.0)) if valid.any() else nan,
        "width_finite_mean": float(np.mean(wf)) if wf.size else nan,
        "width_finite_median": float(np.median(wf)) if wf.size else nan,
        "delay_events_mean": _mean(acc.delay) if acc.delay is not None else nan,
        "picp_lo_follower": nan,
        "picp_hi_follower": nan,
        "n_followers": nan,
    }
    for k in FIRST_COUNTS:
        out[f"picp_first{k}"] = _mean(per_event[acc.issue[:k]]) if acc.issue is not None else nan
    if followers is not None and ok.any():
        fb = bootstrap_by_group(per_event[ok], followers[ok], n_boot=n_boot, seed=seed, ci=ci)
        out.update(picp_lo_follower=float(fb["lo"]), picp_hi_follower=float(fb["hi"]),
                   n_followers=int(np.unique(followers[ok]).size))  # fmt: skip
    return out


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
    finite_quantile: bool = False,
    stream_gamma: float | None = None,
    delay_events_per_s: float = 1.0,
) -> pd.DataFrame:
    """Coverage of the rollout intervals of one model, target and data set per horizon and variant.

    eval_tables / cal_tables: seed -> rollout table (``cf_unc.rollout.tables``) of the evaluated and of the
    calibration data set; eval_events / cal_events: their events tables (``event`` codes in table order, ACI order
    ``order``; optional ``driver`` = follower codes of the second bootstrap interval, ``chrono`` = rank of the event
    in chronological order and ``t_issue`` = time [s] at which its interval is issued, NaN when the data set has no
    comparable times).

    Variants: ``raw`` = [s_q05, s_q95]; ``raw_linear`` = [s_lin05, s_lin95]; with calibration tables also
    ``static`` (conformal quantile of the calibration scores of the same seed and horizon), ``aci_frozen`` and
    ``aci_online`` for every gamma (ACI over the calibration events in their ``order``, then frozen or continued over
    the evaluated events in their ``order``), and, with ``stream_gamma`` and a ``chrono`` column, the variants of
    STREAM_VARIANTS: ``aci_online_chrono`` (ACI of ``stream_gamma`` continued over the evaluated events in
    chronological order), ``aci_online_delayed`` (chronological, the label of an event used only a horizon later: by
    ``t_issue`` when it is finite, else after ``ceil(horizon * delay_events_per_s)`` further events,
    :func:`delayed_availability`), ``static_prequential`` (the static correction recalibrated on the calibration
    scores and every evaluated score seen so far, in the stream of ``aci_online``) and ``static_prequential_delayed``
    (the same in the stream of ``aci_online_delayed``). Seeds present in both inputs are used; finite_quantile: the
    heuristic finite ACI of the submitted tables (module docstring).

    Returns COVERAGE_COLUMNS: ``crps`` (raw rows only) and ``collided`` (every row) are event means of the seed means
    of ``crps_s`` and ``collided``; ``q_mean`` is the seed mean of the static / frozen correction and of the mean
    finite correction of the online and prequential runs; widths in m: ``width_mean`` / ``width_median`` over all
    intervals (+inf as soon as an infinite interval enters the mean, or dominates the median), ``width_finite_*``
    over the finite ones (per event the mean over the seeds with a finite interval); ``share_infinite`` /
    ``share_empty``: share of the (seed, event) intervals with ``Q = +inf`` / an empty interval (``Q < -1``);
    ``picp_first50`` / ``picp_first100``: coverage over the first events of the evaluated stream of an online or
    prequential variant (NaN for the others); ``delay_events_mean``: mean number of events issued before a label was
    used (0 for immediate feedback; NaN for the other variants); ``picp_lo_follower`` / ``picp_hi_follower``: the
    follower-cluster bootstrap interval, ``n_followers`` (NaN without ``driver``).
    """
    seeds = sorted(eval_tables) if cal_tables is None else sorted(set(eval_tables) & set(cal_tables))
    if not seeds:
        return pd.DataFrame(columns=list(COVERAGE_COLUMNS))
    codes = eval_events["event"].to_numpy(np.int64)
    perm = np.argsort(eval_events["order"].to_numpy(np.int64), kind="stable")
    followers = eval_events["driver"].to_numpy() if "driver" in eval_events.columns else None
    calibrated = cal_tables is not None
    streams = calibrated and stream_gamma is not None and "chrono" in eval_events.columns
    if calibrated:
        if cal_events is None:
            raise ValueError("cal_events is needed with cal_tables")
        cal_codes = cal_events["event"].to_numpy(np.int64)
        cal_perm = np.argsort(cal_events["order"].to_numpy(np.int64), kind="stable")
    main_gammas = [float(g) for g in gammas]
    run_gammas = sorted({*main_gammas, *([float(stream_gamma)] if streams else [])})  # type: ignore[arg-type]
    chrono, t_issue = None, None
    if streams:
        chrono = np.argsort(eval_events["chrono"].to_numpy(np.int64), kind="stable")
        if "t_issue" in eval_events.columns:
            t_issue = eval_events["t_issue"].to_numpy(np.float64)[chrono]
    horizons = np.unique(eval_tables[seeds[0]]["horizon_s"].to_numpy(np.float64))
    n_s, n_e = len(seeds), codes.size
    out: list[dict[str, Any]] = []
    for h in horizons:
        keys = ["raw", "raw_linear"]
        if calibrated:
            keys.append("static")
            keys += [f"{v}@{g}" for g in main_gammas for v in ("aci_frozen", "aci_online")]
        if streams:
            keys += [f"aci_online_chrono@{float(stream_gamma)}", f"aci_online_delayed@{float(stream_gamma)}",  # type: ignore[arg-type]
                     "static_prequential", "static_prequential_delayed"]  # fmt: skip
            issue_d, avail_d = delayed_availability(n_e, float(h), t_issue, delay_events_per_s)
        acc = {k: _Acc.empty(n_s, n_e) for k in keys}
        crps = np.full((n_s, n_e), np.nan)
        collided = np.full((n_s, n_e), np.nan)
        for i, seed in enumerate(seeds):
            a = _horizon_arrays(eval_tables[seed], h, codes)
            y, lo, mid, hi = a["truth_s"], a["s_q05"], a["s_q50"], a["s_q95"]
            score = interval_score(y, lo, mid, hi, floor)
            acc["raw"].cov[i] = (score <= 0.0).astype(np.float64)
            acc["raw"].width[i] = hi - lo
            acc["raw"].q[i] = 0.0
            acc["raw_linear"].cov[i] = ((a["s_lin05"] <= y) & (y <= a["s_lin95"])).astype(np.float64)
            acc["raw_linear"].width[i] = a["s_lin95"] - a["s_lin05"]
            acc["raw_linear"].q[i] = 0.0
            crps[i], collided[i] = a["crps_s"], a["collided"]
            if not calibrated:
                continue
            c = _horizon_arrays(cal_tables[seed], h, cal_codes)  # type: ignore[index]
            cal_score = interval_score(c["truth_s"], c["s_q05"], c["s_q50"], c["s_q95"], floor)
            q_static = static_correction(cal_score, alpha)
            acc["static"].cov[i] = (score <= q_static).astype(np.float64)
            acc["static"].width[i] = _corrected_width(lo, mid, hi, q_static, floor)
            acc["static"].q[i] = q_static
            acc["static"].q_means.append(q_static)
            for g in run_gammas:
                run = aci_run(cal_score[cal_perm], alpha, g, n_min, finite_quantile=finite_quantile)
                if g in main_gammas:
                    for variant, frozen in (("aci_frozen", True), ("aci_online", False)):
                        res = aci_run(score[perm], alpha, g, n_min, state=run["state"], frozen=frozen)
                        acc[f"{variant}@{g}"].put(i, perm, res, lo, mid, hi, floor, stream=not frozen)
                if streams and g == float(stream_gamma):  # type: ignore[arg-type]
                    res = stream_run(score[chrono], alpha, method="aci", gamma=g, n_min=n_min, state=run["state"])
                    acc[f"aci_online_chrono@{g}"].put(i, chrono, res, lo, mid, hi, floor, stream=True)  # type: ignore[arg-type]
                    res = stream_run(score[chrono], alpha, method="aci", gamma=g, n_min=n_min, state=run["state"],
                                     issue_at=issue_d, available_at=avail_d)  # fmt: skip
                    acc[f"aci_online_delayed@{g}"].put(i, chrono, res, lo, mid, hi, floor, stream=True)  # type: ignore[arg-type]
            if streams:
                res = stream_run(score[perm], alpha, method="static", history=cal_score)
                acc["static_prequential"].put(i, perm, res, lo, mid, hi, floor, stream=True)
                res = stream_run(score[chrono], alpha, method="static", history=cal_score, issue_at=issue_d,
                                 available_at=avail_d)  # fmt: skip
                acc["static_prequential_delayed"].put(i, chrono, res, lo, mid, hi, floor, stream=True)  # type: ignore[arg-type]
        seed_boot = _boot_seed(boot_seed, h)
        common = {"horizon_s": float(h), "collided": _mean(_mean0(collided)), "n_seeds": n_s}
        for k in keys:
            variant, _, g = k.partition("@")
            row = {**common, "variant": variant, "gamma": float(g) if g else float("nan")}
            row.update(_summary_row(acc[k], seed_boot, n_boot, ci, followers))
            row["crps"] = _mean(_mean0(crps)) if variant == "raw" else float("nan")
            row["q_mean"] = float(np.mean(acc[k].q_means)) if acc[k].q_means else float("nan")
            out.append(row)
    return pd.DataFrame(out, columns=list(COVERAGE_COLUMNS))


# --------------------------------------------------------------------------------------------- H3.4
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
    refute_min: float | None = None,  # noqa: ARG001 - the refutation rule of the submitted version, ignored
) -> dict[str, Any]:
    """Verdict of H3.4 for one model, target and data set: the decision table of docs/review_fixes_v3.md (R7).

    Parts: (i) one-step split conformal coverage of the in-domain test fold (highD; the same number for every data
    set) within ``nominal +- onestep_tol``; (ii) raw rollout coverage below ``raw_max`` at some horizon of the data
    set; (iii) corrected coverage >= ``corrected_min`` at every horizon of the data set.

    Point rule (``verdict``): ``supported`` when (i), (ii) and (iii) hold; ``refuted`` when any part fails ((ii)
    fails: the premise does not hold, the raw intervals already cover; (iii) fails: the correction does not restore
    the coverage; (i) fails: the one-step calibration does not hold in-domain); ``inconclusive`` only when a number
    is missing (NaN). Interval rule (``verdict_ci``, with all three intervals; None otherwise): every part ``holds``
    (its interval lies on the holding side of the threshold), ``fails`` (on the failing side), ``straddles`` or is
    ``missing``; ``refuted`` when any part fails, ``supported`` when all hold, ``inconclusive`` otherwise (a bound
    straddles a threshold, or a number is missing). (i) holds when the interval lies inside the band and fails when
    it lies outside; (ii) holds when the upper bound is below ``raw_max`` at some horizon and fails when the lower
    bound is >= ``raw_max`` at every horizon; (iii) holds when the lower bound is >= ``corrected_min`` at every
    horizon and fails when the upper bound is below it at some horizon. ``refute_min`` (the rule "raw coverage >=
    0.87 at every horizon" of the submitted version) is accepted for old configs and ignored.
    """
    raw, corr = _f64(raw_picp).ravel(), _f64(corrected_picp).ravel()
    one = float(onestep_picp)
    band_lo, band_hi = nominal - onestep_tol - 1e-12, nominal + onestep_tol + 1e-12

    part_i = bool(band_lo <= one <= band_hi) if math.isfinite(one) else None
    fin_raw = raw[np.isfinite(raw)]
    part_ii = True if np.any(fin_raw < raw_max) else (False if raw.size and fin_raw.size == raw.size else None)
    fin_corr = corr[np.isfinite(corr)]
    part_iii = False if np.any(fin_corr < corrected_min) else (True if corr.size and fin_corr.size == corr.size else None)
    parts = (part_i, part_ii, part_iii)
    verdict = "refuted" if any(p is False for p in parts) else (
        "supported" if all(p is True for p in parts) else "inconclusive")  # fmt: skip
    out: dict[str, Any] = {
        "part_i": part_i, "part_ii": part_ii, "part_iii": part_iii, "verdict": verdict,
        "part_i_ci": None, "part_ii_ci": None, "part_iii_ci": None, "verdict_ci": None,
    }  # fmt: skip
    if onestep_ci is None or raw_ci is None or corrected_ci is None:
        return out
    lo1, hi1 = (float(x) for x in onestep_ci)
    if not (math.isfinite(lo1) and math.isfinite(hi1)):
        ci_i = "missing"
    elif lo1 >= band_lo and hi1 <= band_hi:
        ci_i = "holds"
    elif hi1 < band_lo or lo1 > band_hi:
        ci_i = "fails"
    else:
        ci_i = "straddles"
    r_lo, r_hi = _f64(raw_ci[0]).ravel(), _f64(raw_ci[1]).ravel()
    ok = np.isfinite(r_lo) & np.isfinite(r_hi)
    if np.any(r_hi[ok] < raw_max):
        ci_ii = "holds"
    elif not ok.size or not ok.all():
        ci_ii = "missing"
    else:
        ci_ii = "fails" if np.all(r_lo >= raw_max) else "straddles"
    c_lo, c_hi = _f64(corrected_ci[0]).ravel(), _f64(corrected_ci[1]).ravel()
    ok = np.isfinite(c_lo) & np.isfinite(c_hi)
    if np.any(c_hi[ok] < corrected_min):
        ci_iii = "fails"
    elif not ok.size or not ok.all():
        ci_iii = "missing"
    else:
        ci_iii = "holds" if np.all(c_lo >= corrected_min) else "straddles"
    states = (ci_i, ci_ii, ci_iii)
    verdict_ci = "refuted" if "fails" in states else ("supported" if all(s == "holds" for s in states) else
                                                       "inconclusive")  # fmt: skip
    out.update(part_i_ci=ci_i, part_ii_ci=ci_ii, part_iii_ci=ci_iii, verdict_ci=verdict_ci)
    return out
