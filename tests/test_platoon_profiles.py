"""Window selection of the platoon profiles (cf_unc.platoon.profiles) on synthetic runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.signal import savgol_filter

from cf_unc.constants import DT
from cf_unc.platoon.openacc import ACC, HUMAN, PlatoonRun
from cf_unc.platoon.profiles import (
    Candidate,
    SelectionRules,
    candidate_windows,
    chain_prefix,
    follower_mode,
    harmonise_speed,
    load_profiles,
    profiles_digest,
    select_candidates,
    select_profiles,
    valid_vehicles,
    write_profiles,
)

RULES = SelectionRules()


def leader_with_dips(depths: list[float], steady_s: float = 140.0, lead_s: float = 50.0, base: float = 10.0) -> np.ndarray:
    """Leader speed [m/s]: ``lead_s`` steady, then per depth a 10 s cosine dip followed by ``steady_s`` steady."""
    parts = [np.full(int(lead_s / DT), base)]
    j = np.arange(100)
    for d in depths:
        parts.append(base - d / 2 * (1 - np.cos(2 * np.pi * j / 100)))
        parts.append(np.full(int(steady_s / DT), base))
    return np.concatenate(parts)


def synthetic_run(leader: np.ndarray, n_followers: int = 4, delay: int = 10, gap: float = 20.0, campaign: str = "AstaZero",
                  name: str = "run1", mode: int = ACC) -> PlatoonRun:  # fmt: skip
    """Followers repeat the leader with ``delay`` samples per position; constant gaps; one driver mode."""
    n_v, n = n_followers + 1, leader.size
    speed = np.empty((n_v, n))
    for i in range(n_v):
        speed[i] = np.concatenate([np.full(delay * i, leader[0]), leader[: n - delay * i]])
    ivs = np.full((n_v, n), gap)
    ivs[-1] = np.nan
    return PlatoonRun(
        path=Path(campaign) / f"{name}.csv", campaign=campaign, date=None, vehicle_order=[f"V{i}" for i in range(1, n_v + 1)],
        acc_flag=1 if mode == ACC else 0, distance_setting=None, t=np.arange(n) * DT, speed=speed, ivs=ivs,
        mode=np.full((n_v, n), mode, dtype=np.int8), has_speed=np.ones(n_v, bool),
        has_ivs=np.array([True] * n_followers + [False]), has_driver=np.zeros(n_v, bool),
    )  # fmt: skip


# ------------------------------------------------------------------------------------------- signals
def test_harmonise_speed_per_finite_stretch() -> None:
    rng = np.random.default_rng(0)
    v = 10 + rng.normal(0, 0.3, (2, 200))
    v[0, 50:53] = np.nan  # splits row 0 into stretches of 50 and 147 samples
    v[1, 5:] = np.nan  # row 1: a stretch of 5 samples, shorter than the filter -> dropped
    h = harmonise_speed(v)
    np.testing.assert_allclose(h[0, :50], savgol_filter(v[0, :50], 11, 2, mode="interp"))
    np.testing.assert_allclose(h[0, 53:], savgol_filter(v[0, 53:], 11, 2, mode="interp"))
    assert np.isnan(h[0, 50:53]).all() and np.isnan(h[1]).all()
    assert harmonise_speed(v[0]).shape == (200,)


def test_validity_and_chain_prefix() -> None:
    run = synthetic_run(np.full(400, 10.0), n_followers=3)
    run.speed[2, 100] = np.nan  # vehicle 3 not observed
    run.ivs[0, 200] = 0.4  # gap of vehicle 2 below 0.5 m
    run.ivs[2, 300] = 120.0  # gap of vehicle 4 not below 120 m
    run.speed[0, 350:] = 0.9  # leader below 1 m/s
    ok = valid_vehicles(run, harmonise_speed(run.speed), RULES)
    prefix = chain_prefix(ok)
    assert prefix[0] == 4 and prefix[200] == 1 and prefix[300] == 3 and (prefix[360:] == 0).all()
    assert prefix[100] == 2 and prefix[99] == prefix[101] == 4  # only the missing sample (stretches filtered apart)
    assert follower_mode(np.full((2, 5), ACC)) == "ACC" and follower_mode(np.full((2, 5), HUMAN)) == "Human"
    assert follower_mode(np.array([[ACC, HUMAN]])) == "mixed" and follower_mode(np.array([[ACC, -1]])) == "mixed"


# ----------------------------------------------------------------------------------------- candidates
def test_window_starts_steady_before_the_dip_and_ends_steady() -> None:
    run = synthetic_run(leader_with_dips([4.0, 6.0]))
    cands = candidate_windows([run], RULES)
    assert len(cands) == 2
    u = harmonise_speed(run.speed[0])
    for c, dip_start, depth in zip(cands, (500, 2000), (4.0, 6.0)):
        assert c.n == int(120 / DT) and c.n_followers == 4 and c.mode == "ACC"
        first = u[c.start : c.start + 50]
        assert np.ptp(first) <= 0.5 < np.ptp(u[c.start + 1 : c.start + 51])  # the last steady start: an onset
        assert c.start < dip_start + 20 and c.stop > dip_start + 100  # the whole dip lies inside the window
        assert np.ptp(u[c.stop - 50 : c.stop]) <= 0.5  # steady end
        assert c.amplitude == pytest.approx(np.ptp(u[c.start : c.stop])) and abs(c.amplitude - depth) < 0.1
        assert 2.0 <= c.amplitude <= 15.0


def test_dropout_of_a_far_follower_costs_the_follower_not_the_length() -> None:
    run = synthetic_run(leader_with_dips([4.0]))
    run.speed[4, 800] = np.nan  # vehicle 5 (follower 4) not observed 30 s after the dip starts
    (c,) = candidate_windows([run], RULES)
    assert c.n_followers == 3 and c.n == int(120 / DT)  # a 120 s window with three followers beats a 30 s one with four
    run.speed[2, 900] = np.nan  # also follower 2: no platoon of >= 3 followers reaches 60 s any more
    (c,) = candidate_windows([run], RULES)
    # fallback: the longest admissible window, 3 followers up to sample 900 (43 s on the 1 s grid) beats 4 up to 800
    assert c.n_followers == 3 and c.stop <= 900 and c.n == 430


def test_validity_rules_remove_windows() -> None:
    gap_bad = synthetic_run(leader_with_dips([4.0]))
    gap_bad.ivs[0, 540:] = 125.0  # vehicle 2 too far behind from 4 s into the dip on
    (c,) = candidate_windows([gap_bad], RULES) or [None]
    assert c is None or c.stop <= 540
    assert candidate_windows([synthetic_run(leader_with_dips([9.5]))], RULES) == []  # leader below 1 m/s in the dip
    assert candidate_windows([synthetic_run(leader_with_dips([1.5]))], RULES) == []  # amplitude below 2 m/s
    assert candidate_windows([synthetic_run(leader_with_dips([16.0], base=20.0))], RULES) == []  # above 15 m/s
    assert candidate_windows([synthetic_run(leader_with_dips([4.0]), n_followers=2)], RULES) == []  # < 3 followers
    capped = candidate_windows([synthetic_run(leader_with_dips([4.0], base=10.0))], SelectionRules(max_speed=9.0))
    assert capped == []  # the leader exceeds the speed bound
    assert candidate_windows([synthetic_run(leader_with_dips([4.0], steady_s=12.0))], RULES) == []  # < 30 s after the onset
    short = synthetic_run(leader_with_dips([4.0], steady_s=30.0))  # the run ends 30 s after the dip
    (c,) = candidate_windows([short], RULES)
    assert c.stop <= short.t.size and int(30 / DT) <= c.n < int(60 / DT)


# ----------------------------------------------------------------------------------------- selection
def _cand(run: int, group: tuple[str, str], start: int, amp: float, n: int = 600) -> Candidate:
    return Candidate(run, f"{group[0]}/r{run}", group[0], start, n, 4, amp, group[1])


def test_round_robin_and_non_overlap() -> None:
    a_acc, a_hum, z_acc = ("AstaZero", "ACC"), ("AstaZero", "Human"), ("ZalaZone", "ACC")
    cands = [
        _cand(0, a_acc, 0, 9.0), _cand(0, a_acc, 300, 8.0),  # overlaps the first (same run): skipped
        _cand(0, a_acc, 1000, 7.0), _cand(1, a_acc, 0, 6.0),
        _cand(2, a_hum, 0, 5.0),
        _cand(3, z_acc, 0, 3.0), _cand(3, z_acc, 2000, 4.0), _cand(4, z_acc, 0, 3.5),
    ]  # fmt: skip
    chosen = select_candidates(cands, max_profiles=60)
    got = [(c.campaign, c.mode, c.run, c.start) for c, _, _ in chosen]
    assert got == [
        ("AstaZero", "ACC", 0, 0), ("AstaZero", "Human", 2, 0), ("ZalaZone", "ACC", 3, 2000),  # round 1
        ("AstaZero", "ACC", 0, 1000), ("ZalaZone", "ACC", 4, 0),  # round 2 (the overlapping candidate is skipped)
        ("AstaZero", "ACC", 1, 0), ("ZalaZone", "ACC", 3, 0),  # round 3
    ]
    assert [r for _, r, _ in chosen] == [1, 1, 1, 2, 2, 3, 3]
    assert [c.start for c, _, _ in select_candidates(cands, max_profiles=2)] == [0, 0]
    for i, (c, _, _) in enumerate(chosen):
        assert not any(c.overlaps(o) for o, _, _ in chosen[:i])


def test_select_profiles_headline_and_records() -> None:
    runs = [synthetic_run(leader_with_dips([3.0 + 0.5 * k, 2.5 + 0.5 * k, 4.0]), campaign=camp, name=f"r{k}", mode=mode)
            for k, (camp, mode) in enumerate([("AstaZero", ACC), ("AstaZero", HUMAN), ("Vicolungo", ACC), ("ZalaZone", ACC),
                                              ("AstaZero", ACC), ("ZalaZone", ACC)])]  # fmt: skip
    rules = SelectionRules(max_profiles=12, n_headline=10)
    profiles, counts = select_profiles(runs, rules)
    assert len(profiles) == 12 and [p.id for p in profiles][:3] == ["p01", "p02", "p03"]
    assert [p.headline for p in profiles] == [True] * 10 + [False] * 2
    assert [p.group for p in profiles[:4]] == ["AstaZero/ACC", "AstaZero/Human", "ZalaZone/ACC", "Vicolungo/ACC"]
    assert counts["candidates"]["AstaZero/ACC"] == 6 and sum(counts["selected"].values()) == 12
    p = profiles[0]
    run = next(r for r in runs if r.path.name == p.file and r.campaign == p.campaign)
    u = harmonise_speed(run.speed[: p.n_followers + 1])[:, p.start_index : p.start_index + p.n_samples]
    np.testing.assert_allclose(p.leader_speed, u[0])
    np.testing.assert_allclose(p.follower_amplitude, np.ptp(u[1:], axis=1))
    np.testing.assert_allclose(p.follower_sd, u[1:].std(axis=1))
    assert p.leader_amplitude == pytest.approx(np.ptp(u[0])) and p.initial_speed == pytest.approx(u[0, 0])
    assert p.length_s == pytest.approx(p.n_samples * DT) and p.vehicles == run.vehicle_order[: p.n_followers + 1]
    assert max(p.leader_amplitude for p in profiles if p.group == "AstaZero/ACC") == profiles[0].leader_amplitude


def test_profiles_file_round_trip_and_determinism(tmp_path: Path) -> None:
    runs = [synthetic_run(leader_with_dips([4.0, 5.0]), name="a"), synthetic_run(leader_with_dips([3.0]), name="b")]
    profiles, counts = select_profiles(runs, RULES)
    meta = {"description": "test", "counts": counts, "digest": profiles_digest(profiles)}
    first = write_profiles(tmp_path / "p.yaml", profiles, meta).read_bytes()
    second = write_profiles(tmp_path / "q.yaml", profiles, meta).read_bytes()
    assert first == second and b"\r\n" not in first
    loaded, stored = load_profiles(tmp_path / "p.yaml")
    assert stored["digest"] == meta["digest"] == profiles_digest(loaded)  # stored values are rounded before hashing
    assert [p.id for p in loaded] == [p.id for p in profiles]
    for a, b in zip(loaded, profiles):
        np.testing.assert_allclose(a.leader_speed, b.leader_speed, atol=1e-4)
        np.testing.assert_allclose(a.follower_amplitude, b.follower_amplitude, atol=1e-4)
        assert (a.file, a.start_index, a.n_followers, a.mode, a.headline) == (b.file, b.start_index, b.n_followers, b.mode, b.headline)
