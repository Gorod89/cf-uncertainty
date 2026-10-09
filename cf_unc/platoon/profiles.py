"""Leader profiles of the platoon experiment E3 (docs/design.md, section 13.1).

A *profile* is a window of an OpenACC platoon run (:mod:`cf_unc.platoon.openacc`): the leader (vehicle 1 of the file)
and its followers in platoon order. Speeds are harmonised first: Savitzky-Golay filter (11 samples, order 2, mode
``interp``, design.md section 12) on every finite stretch of every vehicle's speed (stretches shorter than the filter
are dropped).

Validity (every sample of the window, vehicles 1..m of the platoon order)::

    speed (harmonised) finite and > min_speed (1 m/s)          every vehicle
    IVS<i-1> finite and inside (0.5, 120) m                     every follower i = 2..m
    m - 1 >= min_followers (3)                                  followers = the longest valid prefix of the order
    min_length_s <= length <= max_length_s (30 s .. 120 s),  min_amplitude <= max(v_0) - min(v_0) <= max_amplitude
    leader speed <= max_speed (optional; scripts/select_profiles.py: the speed at which the equilibrium gap of the
                               closed-loop IDM reaches 120 m, so that the initial state of the simulation exists)

Candidate windows (choices the contract leaves open): the simulated platoon starts in equilibrium at the leader's
first speed, so a window starts where the leader is steady (range of its speed over the first ``steady_s`` = 5 s at
most ``steady_tol`` = 0.5 m/s) and as late as possible: at the last steady start before the leader leaves the steady
band (an *onset*), i.e. the perturbation begins at most ``steady_s`` after the window start. A window also ends with
the leader steady over its last ``steady_s`` (the simulation holds the last speed). From every onset one candidate:
the largest platoon (m) whose longest admissible window reaches ``preferred_length_s`` (60 s), else the longest
admissible window (ties: more followers); lengths on a ``length_step_s`` (1 s) grid.

Selection: the driver mode of a candidate is that of its followers (``ACC`` if every follower sample is ACC,
``Human`` if every one is Human, else ``mixed``); candidates are ranked by leader amplitude (descending; ties by run
name and start) within each (campaign, mode) group and taken round-robin over the groups (campaigns in the order of
:data:`cf_unc.platoon.openacc.CAMPAIGNS`, modes ACC, Human, mixed), skipping candidates that overlap a window already
taken from the same run, until ``max_profiles`` (60) windows are taken. The first ``n_headline`` (10) form the
headline set, all of them the extended set.

Every profile carries what the experiment needs (the experiment never reads the raw files, except for the
re-smoothing below): the harmonised leader speed over the window, the measured follower amplitudes
``max(v_i) - min(v_i)`` and speed standard deviations (population, ddof = 0) over the window, and descriptive fields
(file, campaign, mode, start, length, ...).

Re-smoothing (E3-pooled, design.md 17.1, :func:`resmooth_profiles`): the leader speed of every selected profile is
derived again from its raw file (same file, start sample and length; the selection is not repeated) with another
Savitzky-Golay window of the same :func:`harmonise_speed`. The window of the selection is applied as well and must
reproduce the stored leader speed and follower statistics (the raw data and the windows are the ones of the profiles
file); the measured follower statistics are kept (they stay those of the selection's window).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import savgol_filter

from cf_unc.constants import DT
from cf_unc.platoon.openacc import ACC, CAMPAIGNS, HUMAN, PlatoonRun, read_run

__all__ = [
    "MODES",
    "Candidate",
    "Profile",
    "SelectionRules",
    "candidate_windows",
    "chain_prefix",
    "follower_mode",
    "harmonise_speed",
    "load_profiles",
    "make_profile",
    "profiles_digest",
    "resmooth_profiles",
    "select_candidates",
    "select_profiles",
    "valid_vehicles",
    "write_profiles",
]

MODES = ("ACC", "Human", "mixed")  # profile driver modes, in round-robin order
_DECIMALS = 4  # stored speeds [m/s] and statistics are rounded to 0.1 mm/s


@dataclass(frozen=True)
class SelectionRules:
    """Window rules of design.md 13.1 (speeds in m/s, gaps in m, times in s)."""

    min_length_s: float = 30.0
    max_length_s: float = 120.0
    length_step_s: float = 1.0
    preferred_length_s: float = 60.0
    min_amplitude: float = 2.0
    max_amplitude: float = 15.0
    min_followers: int = 3
    min_speed: float = 1.0
    max_speed: float | None = None
    gap_bounds: tuple[float, float] = (0.5, 120.0)
    steady_s: float = 5.0
    steady_tol: float = 0.5
    max_profiles: int = 60
    n_headline: int = 10
    sg_window: int = 11
    sg_order: int = 2

    def samples(self, seconds: float) -> int:
        return int(round(float(seconds) / DT))


@dataclass
class Candidate:
    """A valid window of a run: samples ``[start, start + n)``; ``n_followers`` vehicles behind the leader."""

    run: int  # index of the run in the list given to candidate_windows
    run_name: str
    campaign: str
    start: int
    n: int
    n_followers: int
    amplitude: float  # leader amplitude [m/s]
    mode: str

    @property
    def stop(self) -> int:
        return self.start + self.n

    def overlaps(self, other: "Candidate") -> bool:
        return self.run == other.run and self.start < other.stop and other.start < self.stop


@dataclass
class Profile:
    """One selected profile (fields of ``configs/platoon/profiles.yaml``).

    leader_speed          (n,) harmonised leader speed [m/s] at 10 Hz over the window
    follower_amplitude    (n_followers,) measured max - min of the harmonised follower speeds [m/s], platoon order
    follower_sd           (n_followers,) their standard deviations over the window [m/s] (ddof = 0)
    initial_deviation     max |v_i - u[0]| of the followers over the first ``steady_s`` of the window [m/s]: how far
                          the measured platoon is from the equilibrium the simulation starts in
    """

    id: str
    file: str
    campaign: str
    mode: str
    start_s: float  # source time of the first sample [s]
    start_index: int  # grid index of the first sample in the run
    length_s: float  # n * DT [s]
    n_followers: int
    leader_amplitude: float
    leader_sd: float
    initial_speed: float
    final_speed: float
    leader_speed: np.ndarray
    follower_amplitude: np.ndarray
    follower_sd: np.ndarray
    group: str = ""
    rank_in_group: int = 0
    round: int = 0
    headline: bool = False
    initial_deviation: float = 0.0
    acc_flag: int | None = None
    distance_setting: str | None = None
    date: str | None = None
    vehicles: list[str] = field(default_factory=list)  # names, leader first

    @property
    def n_samples(self) -> int:
        return int(self.leader_speed.shape[0])

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("leader_speed", "follower_amplitude", "follower_sd"):
            out[key] = [round(float(x), _DECIMALS) for x in np.asarray(getattr(self, key), dtype=np.float64)]
        for key in ("start_s", "length_s", "leader_amplitude", "leader_sd", "initial_speed", "final_speed",
                    "initial_deviation"):
            out[key] = round(float(out[key]), _DECIMALS)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Profile":
        d = dict(data)
        for key in ("leader_speed", "follower_amplitude", "follower_sd"):
            d[key] = np.asarray(d[key], dtype=np.float64)
        return cls(**d)


# ------------------------------------------------------------------------------------------- signals
def harmonise_speed(speed: np.ndarray, window: int = 11, order: int = 2) -> np.ndarray:
    """Savitzky-Golay filtered speeds (..., T): every maximal finite stretch of every row is filtered on its own
    (mode ``interp``); stretches shorter than ``window`` samples become NaN."""
    v = np.atleast_2d(np.asarray(speed, dtype=np.float64))
    out = np.full(v.shape, np.nan)
    for row, (x, y) in enumerate(zip(v, out)):
        ok = np.isfinite(x)
        if not ok.any():
            continue
        edges = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b - a >= window:
                y[a:b] = savgol_filter(x[a:b], window, order, mode="interp")
    return out.reshape(np.shape(speed))


def valid_vehicles(run: PlatoonRun, speed_h: np.ndarray, rules: SelectionRules) -> np.ndarray:
    """(N, T) bool: vehicle i is usable at a sample (harmonised speed finite and > min_speed; for a follower also its
    gap to the predecessor finite and inside ``gap_bounds``)."""
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(speed_h) & (speed_h > rules.min_speed)
        lo, hi = rules.gap_bounds
        gap = run.ivs[:-1]  # IVS<j> (row j - 1) is the gap of vehicle j + 1 (row j): aligned with ok[1:]
        gap_ok = np.isfinite(gap) & (gap > lo) & (gap < hi)
    ok[1:] &= gap_ok
    return ok


def chain_prefix(valid: np.ndarray) -> np.ndarray:
    """(T,) number of consecutive usable vehicles from vehicle 1 (the leader) at every sample."""
    if valid.shape[0] == 0:
        return np.zeros(valid.shape[1], dtype=np.int64)
    broken = ~valid
    first_bad = np.where(broken.any(axis=0), broken.argmax(axis=0), valid.shape[0])
    return first_bad.astype(np.int64)


def follower_mode(mode: np.ndarray) -> str:
    """Driver mode of the followers from their per-sample codes (n_followers, n): ACC, Human or mixed."""
    if mode.size and np.all(mode == ACC):
        return "ACC"
    if mode.size and np.all(mode == HUMAN):
        return "Human"
    return "mixed"


# ----------------------------------------------------------------------------------------- candidates
def _steady_starts(u: np.ndarray, n_steady: int, tol: float) -> np.ndarray:
    """(T,) bool: the leader speed over ``[t, t + n_steady)`` is finite with a range of at most ``tol``."""
    out = np.zeros(u.shape[0], dtype=bool)
    if u.shape[0] < n_steady:
        return out
    view = sliding_window_view(u, n_steady)
    with np.errstate(invalid="ignore"):
        rng = view.max(axis=1) - view.min(axis=1)
    out[: view.shape[0]] = np.isfinite(rng) & (rng <= tol)
    return out


def candidate_windows(runs: Sequence[PlatoonRun], rules: SelectionRules = SelectionRules()) -> list[Candidate]:
    """Candidate windows of every run (module docstring), in run order and start order."""
    out: list[Candidate] = []
    n_min, n_max = rules.samples(rules.min_length_s), rules.samples(rules.max_length_s)
    step = max(1, rules.samples(rules.length_step_s))
    n_steady = rules.samples(rules.steady_s)
    lengths = np.arange(n_min, n_max + 1, step)
    for r, run in enumerate(runs):
        k = run.chain_length()
        if k < rules.min_followers + 1 or len(run) < n_min:
            continue
        speed_h = harmonise_speed(run.speed[:k], rules.sg_window, rules.sg_order)
        prefix = chain_prefix(valid_vehicles(_sub(run, k), speed_h, rules))
        u = speed_h[0]
        steady = _steady_starts(u, n_steady, rules.steady_tol)
        onsets = np.flatnonzero(steady[:-1] & ~steady[1:])
        for t in onsets:
            cand = _window_at(run, r, int(t), u, prefix, steady, lengths, rules)
            if cand is not None:
                out.append(cand)
    return out


def _sub(run: PlatoonRun, k: int) -> PlatoonRun:
    """The first k vehicles of a run (views)."""
    return PlatoonRun(
        path=run.path, campaign=run.campaign, date=run.date, vehicle_order=run.vehicle_order[:k],
        acc_flag=run.acc_flag, distance_setting=run.distance_setting, t=run.t, speed=run.speed[:k],
        ivs=run.ivs[:k], mode=run.mode[:k], has_speed=run.has_speed[:k], has_ivs=run.has_ivs[:k],
        has_driver=run.has_driver[:k],
    )  # fmt: skip


def _window_at(
    run: PlatoonRun, r: int, t: int, u: np.ndarray, prefix: np.ndarray, steady: np.ndarray, lengths: np.ndarray,
    rules: SelectionRules,
) -> Candidate | None:  # fmt: skip
    """The window that starts at the onset ``t`` (module docstring), or None.

    A length n (from ``lengths``) is admissible for a platoon of m vehicles when vehicles 1..m are valid on
    ``[t, t + n)``, the leader amplitude is at most ``max_amplitude`` and the leader is steady over the last
    ``steady_s`` of the window. The largest m whose longest admissible window reaches ``preferred_length_s`` wins
    (more followers); without such an m, the longest admissible window (ties: more followers).
    """
    n_avail = min(int(lengths[-1]), len(run) - t)
    n_steady = rules.samples(rules.steady_s)
    lengths = lengths[lengths <= n_avail]
    if lengths.size == 0:
        return None
    seg_u = u[t : t + n_avail]
    with np.errstate(invalid="ignore"):
        amp = np.fmax.accumulate(seg_u) - np.fmin.accumulate(seg_u)  # amplitude of [t, t + j + 1)
    count = np.minimum.accumulate(prefix[t : t + n_avail])  # vehicles valid on the whole of [t, t + j + 1)
    a, n_vehicles = amp[lengths - 1], count[lengths - 1]
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(a) & (a <= rules.max_amplitude) & steady[t + lengths - n_steady]
        if rules.max_speed is not None:
            ok &= np.fmax.accumulate(seg_u)[lengths - 1] <= rules.max_speed
    best: tuple[int, int] | None = None  # (n, m)
    longest: tuple[int, int] | None = None
    for m in range(int(prefix[t]), rules.min_followers, -1):
        good = lengths[ok & (n_vehicles >= m)]
        if good.size == 0:
            continue
        n = int(good[-1])
        if best is None and n >= rules.samples(rules.preferred_length_s):
            best = (n, m)
        if longest is None or n > longest[0]:
            longest = (n, m)
    pick = best or longest
    if pick is None:
        return None
    n, m = pick
    amplitude = float(amp[n - 1])
    if amplitude < rules.min_amplitude:
        return None
    mode = follower_mode(run.mode[1:m, t : t + n])
    return Candidate(r, run.name, run.campaign, int(t), n, m - 1, amplitude, mode)


# ------------------------------------------------------------------------------------------ selection
def _group_order(groups: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    camp = {c: i for i, c in enumerate(CAMPAIGNS)}
    mode = {m: i for i, m in enumerate(MODES)}
    return sorted(set(groups), key=lambda g: (camp.get(g[0], len(camp)), g[0], mode.get(g[1], len(mode)), g[1]))


def select_candidates(candidates: Sequence[Candidate], max_profiles: int = 60) -> list[tuple[Candidate, int, int]]:
    """Round-robin selection (module docstring): ``(candidate, round, rank in its group)`` in selection order."""
    by_group: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_group[(c.campaign, c.mode)].append(c)
    order = _group_order(by_group)
    for g in order:
        by_group[g].sort(key=lambda c: (-c.amplitude, c.run_name, c.start))
    pointer = {g: 0 for g in order}
    taken: dict[int, list[Candidate]] = defaultdict(list)
    out: list[tuple[Candidate, int, int]] = []
    rnd = 0
    while len(out) < max_profiles:
        rnd += 1
        progressed = False
        for g in order:
            cands, i = by_group[g], pointer[g]
            while i < len(cands) and any(cands[i].overlaps(o) for o in taken[cands[i].run]):
                i += 1
            if i < len(cands):
                taken[cands[i].run].append(cands[i])
                out.append((cands[i], rnd, i + 1))
                progressed = True
                i += 1
            pointer[g] = i
            if len(out) >= max_profiles:
                break
        if not progressed:
            break
    return out


def make_profile(run: PlatoonRun, cand: Candidate, rules: SelectionRules = SelectionRules()) -> Profile:
    """The profile record of a candidate window (harmonised speeds, measured follower statistics)."""
    m = cand.n_followers + 1
    speed_h = harmonise_speed(run.speed[:m], rules.sg_window, rules.sg_order)[:, cand.start : cand.stop]
    u, follow = speed_h[0], speed_h[1:]
    return Profile(
        id="",
        file=run.path.name,
        campaign=run.campaign,
        mode=cand.mode,
        start_s=float(run.t[cand.start]),
        start_index=cand.start,
        length_s=cand.n * DT,
        n_followers=cand.n_followers,
        leader_amplitude=float(u.max() - u.min()),
        leader_sd=float(u.std()),
        initial_speed=float(u[0]),
        final_speed=float(u[-1]),
        leader_speed=u.copy(),
        follower_amplitude=follow.max(axis=1) - follow.min(axis=1),
        follower_sd=follow.std(axis=1),
        initial_deviation=float(np.abs(follow[:, : rules.samples(rules.steady_s)] - u[0]).max()),
        acc_flag=run.acc_flag,
        distance_setting=run.distance_setting,
        date=run.date,
        vehicles=list(run.vehicle_order[:m]),
    )


def select_profiles(runs: Sequence[PlatoonRun], rules: SelectionRules = SelectionRules()) -> tuple[list[Profile], dict[str, Any]]:
    """Candidates, round-robin selection and profile records of ``runs``; returns the profiles in selection order
    (ids ``p01``, ``p02``, ...) and counts (candidates and selected per group, runs without a usable platoon)."""
    candidates = candidate_windows(runs, rules)
    chosen = select_candidates(candidates, rules.max_profiles)
    width = max(2, len(str(len(chosen))))
    profiles = []
    for j, (cand, rnd, rank) in enumerate(chosen):
        p = make_profile(runs[cand.run], cand, rules)
        p.id = f"p{j + 1:0{width}d}"
        p.group, p.round, p.rank_in_group, p.headline = f"{cand.campaign}/{cand.mode}", rnd, rank, j < rules.n_headline
        profiles.append(p)
    counts: dict[str, Any] = {
        "runs": len(runs),
        "runs_without_platoon": sorted(r.name for r in runs if r.chain_length() < rules.min_followers + 1),
        "candidates": _count(f"{c.campaign}/{c.mode}" for c in candidates),
        "selected": _count(p.group for p in profiles),
        "headline": _count(p.group for p in profiles if p.headline),
    }
    return profiles, counts


def _count(keys: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for k in keys:
        out[k] += 1
    return dict(sorted(out.items()))


# ------------------------------------------------------------------------------------------- storage
class _FlowList(list):
    """A list written in YAML flow style (one line)."""


def _represent_flow(dumper: yaml.SafeDumper, data: _FlowList) -> yaml.Node:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(_FlowList, _represent_flow)


def write_profiles(path: str | Path, profiles: Sequence[Profile], meta: Mapping[str, Any]) -> Path:
    """Write the profiles file (deterministic: fixed key order, rounded floats, no time stamp)."""
    path = Path(path)
    records = []
    for p in profiles:
        d = p.to_dict()
        for key in ("leader_speed", "follower_amplitude", "follower_sd", "vehicles"):
            d[key] = _FlowList(d[key])
        records.append(d)
    payload = {**dict(meta), "profiles": records}
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(payload, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=1 << 16)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:  # the same bytes on every platform
        fh.write(text)
    tmp.replace(path)
    return path


def load_profiles(path: str | Path) -> tuple[list[Profile], dict[str, Any]]:
    """Profiles (file order) and the metadata of a profiles file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    records = data.pop("profiles", []) or []
    return [Profile.from_dict(r) for r in records], data


def profiles_digest(profiles: Sequence[Profile]) -> str:
    """SHA-256 (16 hex digits) of the simulation-relevant content of the profiles (ids, speeds, measured values)."""
    h = hashlib.sha256()
    for p in profiles:
        h.update(json.dumps([p.id, p.file, p.start_index, p.n_followers], sort_keys=True).encode("utf-8"))
        for key in ("leader_speed", "follower_amplitude", "follower_sd"):
            h.update(np.round(np.asarray(getattr(p, key), dtype=np.float64), _DECIMALS).tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------------------- re-smoothing
def _stats(x: Sequence[float]) -> dict[str, float]:
    a = np.asarray(x, dtype=np.float64)
    return {"min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())} if a.size else {}


def resmooth_profiles(
    profiles: Sequence[Profile],
    raw_dir: str | Path,
    window: int,
    order: int = 2,
    *,
    base_window: int = 11,
    steady_s: float = 5.0,
    tol: float = 2e-4,
    runs: dict[str, PlatoonRun] | None = None,
) -> tuple[list[Profile], dict[str, Any]]:
    """The profiles with the leader speed smoothed again from the raw OpenACC files (module docstring, design.md 17.1).

    For every profile the run ``raw_dir/<campaign>/<file>`` is read (:func:`cf_unc.platoon.openacc.read_run`; ``runs``
    caches the parsed files by path) and its leader speed is filtered with :func:`harmonise_speed` (``window``,
    ``order``) and cut to the samples ``[start_index, start_index + n)`` of the profile. Checks (ValueError): the file
    holds the window and the vehicles of the profile, and the selection's window ``base_window`` reproduces the stored
    leader speed and follower statistics within ``tol`` (the stored values are rounded to 0.1 mm/s); the re-smoothed
    leader speed is finite. The new profile keeps every field of the stored one except the leader speed and what is
    derived from it (``leader_amplitude``, ``leader_sd``, ``initial_speed``, ``final_speed``, ``initial_deviation``
    against the stored follower speeds of ``base_window``); its values are rounded as in the profiles file, so a
    profile written with :func:`write_profiles` and loaded again is identical.

    Returns the profiles (same order and ids) and the provenance: ``window``, ``order``, ``base_window``, ``files``
    (bytes and SHA-256 of every raw file read), ``check`` (largest deviation of the ``base_window`` re-derivation from
    the stored values), ``change`` (re-smoothed minus stored: leader amplitude, initial speed, largest absolute
    difference of the leader speed; min / median / max over the profiles), ``leader_speed`` (min and max of the
    re-smoothed leaders) and ``digest`` (:func:`profiles_digest` of the new profiles).
    """
    window, order, base_window = int(window), int(order), int(base_window)
    if window < 3 or window % 2 == 0 or window <= order:
        raise ValueError(f"the Savitzky-Golay window must be an odd number of samples > order, got {window}")
    raw = Path(raw_dir)
    cache = {} if runs is None else runs
    n_steady = int(round(float(steady_s) / DT))
    out: list[Profile] = []
    files: dict[str, dict[str, Any]] = {}
    worst, change = 0.0, {"leader_amplitude": [], "initial_speed": [], "max_abs_leader": []}
    for p in profiles:
        path = raw / p.campaign / p.file
        key = str(path.resolve())
        if key not in cache:
            if not path.is_file():
                raise ValueError(f"profile {p.id}: raw file {path} not found")
            cache[key] = read_run(path, p.campaign)
        run = cache[key]
        name = f"{p.campaign}/{p.file}"
        if name not in files:
            data = path.read_bytes()
            files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()[:16]}
        m, a = int(p.n_followers) + 1, int(p.start_index)
        b = a + p.n_samples
        if b > len(run) or m > run.n_vehicles:
            raise ValueError(f"profile {p.id}: {name} has {len(run)} samples and {run.n_vehicles} vehicles, the profile "
                             f"needs samples {a}..{b - 1} of {m} vehicles")  # fmt: skip
        if p.vehicles and list(run.vehicle_order[:m]) != list(p.vehicles):
            raise ValueError(f"profile {p.id}: vehicles {run.vehicle_order[:m]} in {name}, the profile has {p.vehicles}")
        if abs(float(run.t[a]) - float(p.start_s)) > 1e-3:
            raise ValueError(f"profile {p.id}: sample {a} of {name} is at {run.t[a]:.4f} s, the profile starts at {p.start_s} s")
        base = harmonise_speed(run.speed[:m], base_window, order)[:, a:b]
        follow = base[1:]
        dev = max(
            float(np.max(np.abs(base[0] - p.leader_speed))),
            float(np.max(np.abs(np.ptp(follow, axis=1) - p.follower_amplitude))),
            float(np.max(np.abs(follow.std(axis=1) - p.follower_sd))),
        )
        if not np.isfinite(dev) or dev > tol:
            raise ValueError(f"profile {p.id}: the window of {base_window} samples does not reproduce the stored profile "
                             f"from {name} (largest deviation {dev:.2e} > {tol:.0e})")  # fmt: skip
        worst = max(worst, dev)
        u = harmonise_speed(run.speed[0], window, order)[a:b]
        if not np.all(np.isfinite(u)):
            raise ValueError(f"profile {p.id}: the leader speed of {name} smoothed with {window} samples is not finite")
        new = replace(
            p, leader_speed=u, leader_amplitude=float(u.max() - u.min()), leader_sd=float(u.std()),
            initial_speed=float(u[0]), final_speed=float(u[-1]),
            initial_deviation=float(np.abs(follow[:, :n_steady] - u[0]).max()),
        )  # fmt: skip
        new = Profile.from_dict(new.to_dict())  # the rounding of the profiles file
        change["leader_amplitude"].append(new.leader_amplitude - p.leader_amplitude)
        change["initial_speed"].append(new.initial_speed - p.initial_speed)
        change["max_abs_leader"].append(float(np.max(np.abs(new.leader_speed - p.leader_speed))))
        out.append(new)
    speeds = np.concatenate([q.leader_speed for q in out]) if out else np.zeros(0)
    info = {
        "window": window, "order": order, "base_window": base_window,
        "files": dict(sorted(files.items())),
        "check": {"max_abs_deviation": worst, "tolerance": float(tol)},
        "change": {k: _stats(v) for k, v in change.items()},
        "leader_speed": {"min": float(speeds.min()), "max": float(speeds.max())} if speeds.size else {},
        "digest": profiles_digest(out),
    }  # fmt: skip
    return out, info
