"""Sliding windows (docs/design.md, section 3.3)."""

from __future__ import annotations

import numpy as np
import pytest

from cf_unc.constants import DT, HIST_LEN
from cf_unc.data.schema import Event, EventSet
from cf_unc.data.synthetic import SyntheticConfig, generate
from cf_unc.data.windows import WindowSet, driver_key, make_windows


def hand_event(
    n: int = 8, event_id: str = "e0", dataset: str = "ds", follower_id: str = "f1", offset: float = 0.0
) -> Event:
    """s = 20 + i, dv = 1 - i, v = 10 + i, a = 100 + i, v_lead = 9 + 2 i (i = sample index, plus ``offset``)."""
    i = np.arange(n, dtype=np.float64)
    v, v_lead = 10.0 + i + offset, 9.0 + 2.0 * i + offset
    s = 20.0 + i + offset
    x_follower = i
    return Event(
        event_id=event_id,
        dataset=dataset,
        site="site",
        follower_id=follower_id,
        leader_id="lead",
        t=i * DT,
        s=s,
        dv=v - v_lead,
        v=v,
        a=100.0 + i + offset,
        v_lead=v_lead,
        x_lead=x_follower + s,
        x_follower=x_follower,
    )


def test_alignment_on_hand_made_event():
    ev = hand_event()
    ev.validate()
    w = make_windows([ev], hist_len=3)
    assert len(w) == 6 and w.hist_len == 3
    np.testing.assert_array_equal(w.t_index, [2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(w.event_index, 0)
    # X[0] = (s, dv, v)[0:3], y[0] = a[2]
    np.testing.assert_array_equal(w.X[0], [[20.0, 1.0, 10.0], [21.0, 0.0, 11.0], [22.0, -1.0, 12.0]])
    np.testing.assert_array_equal(w.y, [102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
    np.testing.assert_array_equal(w.X[-1], [[25.0, -4.0, 15.0], [26.0, -5.0, 16.0], [27.0, -6.0, 17.0]])
    assert w.X.dtype == np.float32 and w.y.dtype == np.float32 and w.X.flags.c_contiguous
    for arr in (w.driver, w.event_index, w.t_index):
        assert arr.dtype == np.int64
    assert w.y_seq is None and w.v_lead_future is None
    assert w.driver_names == ["ds/f1"]


def test_forward_difference_target():
    w = make_windows([hand_event()], hist_len=3, target="fwd_diff")
    np.testing.assert_array_equal(w.t_index, [2, 3, 4, 5, 6])  # needs t + 1 < n
    np.testing.assert_allclose(w.y, 1.0 / DT, rtol=1e-6)
    with pytest.raises(ValueError, match="target"):
        make_windows([hand_event()], target="jerk")


def test_sequence_extras():
    w = make_windows([hand_event()], hist_len=3, seq_len=2)
    np.testing.assert_array_equal(w.t_index, [2, 3, 4, 5])  # needs t + L < n
    np.testing.assert_array_equal(w.y_seq, [[102.0, 103.0], [103.0, 104.0], [104.0, 105.0], [105.0, 106.0]])
    # v_lead[t+1 : t+L+1] with v_lead = 9 + 2 i
    np.testing.assert_array_equal(w.v_lead_future, [[15.0, 17.0], [17.0, 19.0], [19.0, 21.0], [21.0, 23.0]])
    np.testing.assert_array_equal(w.y, w.y_seq[:, 0])
    assert w.y_seq.dtype == np.float32 and w.v_lead_future.dtype == np.float32
    w_fd = make_windows([hand_event()], hist_len=3, seq_len=2, target="fwd_diff")
    np.testing.assert_array_equal(w_fd.t_index, [2, 3, 4, 5])


def test_stride():
    w = make_windows([hand_event(n=10)], hist_len=3, stride=3)
    np.testing.assert_array_equal(w.t_index, [2, 5, 8])
    np.testing.assert_array_equal(w.y, [102.0, 105.0, 108.0])
    np.testing.assert_array_equal(w.X[1, :, 0], [23.0, 24.0, 25.0])


def test_several_events_and_driver_codes():
    events = EventSet(
        [
            hand_event(8, "a", "ds1", "f9"),
            hand_event(2, "short", "ds1", "f0"),  # shorter than the history: no window
            hand_event(6, "b", "ds2", "f9", offset=1000.0),  # same follower id, other dataset
            hand_event(5, "c", "ds1", "f9", offset=2000.0),
        ]
    )
    w = make_windows(events, hist_len=3)
    assert w.driver_names == ["ds1/f0", "ds1/f9", "ds2/f9"]
    np.testing.assert_array_equal(np.bincount(w.event_index, minlength=4), [6, 0, 4, 3])
    for i in range(len(w)):
        ev = events[int(w.event_index[i])]
        t = int(w.t_index[i])
        assert w.driver_names[w.driver[i]] == driver_key(ev)
        np.testing.assert_array_equal(w.X[i], np.stack([ev.s, ev.dv, ev.v], axis=1)[t - 2 : t + 1].astype(np.float32))
        assert w.y[i] == np.float32(ev.a[t])
    # windows never straddle two events: the offsets make every value event-specific
    assert np.all(w.X[w.event_index == 2, :, 0] > 1000.0) and np.all(w.X[w.event_index == 3, :, 0] > 2000.0)


def test_matches_brute_force_on_synthetic_events():
    es, _ = generate(SyntheticConfig(n_drivers=3, events_per_driver=2, duration_s=12.0))
    w = make_windows(es, seq_len=20, stride=2)
    assert w.X.shape[1:] == (HIST_LEN, 3) and w.y_seq.shape[1] == 20
    rng = np.random.default_rng(0)
    for i in rng.choice(len(w), size=40, replace=False):
        ev = es[int(w.event_index[i])]
        t = int(w.t_index[i])
        assert (t - HIST_LEN + 1) % 2 == 0 and t + 20 < len(ev)
        states = np.stack([ev.s, ev.dv, ev.v], axis=1).astype(np.float32)
        np.testing.assert_array_equal(w.X[i], states[t - HIST_LEN + 1 : t + 1])
        np.testing.assert_array_equal(w.y_seq[i], ev.a[t : t + 20].astype(np.float32))
        np.testing.assert_array_equal(w.v_lead_future[i], ev.v_lead[t + 1 : t + 21].astype(np.float32))
    # counts per event: t = 29, 31, ..., <= n - 1 - 20
    n = len(es[0])
    assert np.sum(w.event_index == 0) == (n - 1 - 20 - (HIST_LEN - 1)) // 2 + 1


def test_subset_and_empty():
    w = make_windows([hand_event(), hand_event(9, "e1", follower_id="f2")], hist_len=3, seq_len=1)
    idx = np.array([4, 0, 0, 7])
    sub = w.subset(idx)
    assert isinstance(sub, WindowSet) and len(sub) == 4 and sub.driver_names == w.driver_names
    for name in ("X", "y", "driver", "event_index", "t_index", "y_seq", "v_lead_future"):
        np.testing.assert_array_equal(getattr(sub, name), getattr(w, name)[idx])
    mask = w.driver == 1
    assert len(w.subset(mask)) == int(mask.sum()) and np.all(w.subset(mask).driver == 1)
    assert make_windows([hand_event()], hist_len=3).subset(np.array([1])).y_seq is None

    empty = make_windows([hand_event(2)], hist_len=3, seq_len=4)
    assert len(empty) == 0 and empty.X.shape == (0, 3, 3) and empty.y_seq.shape == (0, 4)
    assert len(make_windows([], hist_len=5)) == 0
    with pytest.raises(ValueError):
        make_windows([hand_event()], stride=0)
