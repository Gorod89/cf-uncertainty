"""Target mean_1s (docs/design.md, section 11.1): windows, look-ahead, y_prev of the windows tables, pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cf_unc.constants import DT, MEAN_TARGET_STEPS
from cf_unc.data.pipeline import load_bundle, synthetic_bundle
from cf_unc.data.schema import Event, EventSet
from cf_unc.data.synthetic import SyntheticConfig, generate
from cf_unc.data.windows import TARGET_LOOKAHEAD, TARGET_STEPS, TARGETS, make_windows, target_values
from cf_unc.maps import io as tables

# follower speed of the hand-made event (14 samples): windows with hist_len 3 end at t = 2 and t = 3 (t + 10 < 14)
V_HAND = [10.0, 10.5, 11.2, 11.8, 12.0, 12.1, 12.5, 13.0, 13.2, 13.9, 14.0, 14.6, 15.0, 15.3]


def event_from_speed(v: list[float] | np.ndarray, event_id: str = "e0", follower_id: str = "f1") -> Event:
    """Event with the follower speed ``v``; a = 100 + i (unrelated to v on purpose), s = 20 + i, v_lead = v + 1."""
    v = np.asarray(v, dtype=np.float64)
    i = np.arange(v.size, dtype=np.float64)
    v_lead = v + 1.0
    s = 20.0 + i
    x_follower = np.cumsum(v) * DT
    return Event(
        event_id=event_id, dataset="ds", site="site", follower_id=follower_id, leader_id="lead", t=i * DT, s=s,
        dv=v - v_lead, v=v, a=100.0 + i, v_lead=v_lead, x_lead=x_follower + s, x_follower=x_follower,
    )  # fmt: skip


def synthetic_events(seed: int = 3) -> EventSet:
    return generate(SyntheticConfig(n_drivers=3, events_per_driver=2, duration_s=12.0, seed=seed))[0]


# ----------------------------------------------------------------------------------- windows
def test_constants_and_tables_of_the_targets() -> None:
    assert MEAN_TARGET_STEPS == 10 and "mean_1s" in TARGETS
    assert TARGET_STEPS == {"a": 1, "fwd_diff": 1, "mean_1s": 10}
    assert TARGET_LOOKAHEAD == {"a": 0, "fwd_diff": 1, "mean_1s": 10}
    with pytest.raises(ValueError, match="target"):
        make_windows([event_from_speed(V_HAND)], hist_len=3, target="mean_2s")


def test_mean_1s_hand_computed() -> None:
    ev = event_from_speed(V_HAND)
    w = make_windows([ev], hist_len=3, target="mean_1s")
    np.testing.assert_array_equal(w.t_index, [2, 3])  # needs t + 10 < n = 14
    # (v[12] - v[2]) / (10 DT) = 15.0 - 11.2 and (v[13] - v[3]) / 1 s = 15.3 - 11.8
    np.testing.assert_allclose(w.y, [3.8, 3.5], rtol=1e-6)
    assert w.y.dtype == np.float32
    one_step = make_windows([ev], hist_len=3)  # the history does not depend on the target
    np.testing.assert_array_equal(one_step.t_index, np.arange(2, 14))
    np.testing.assert_array_equal(w.X, one_step.X[:2])
    np.testing.assert_array_equal(target_values(ev.a, ev.v, np.array([2, 3]), "mean_1s"), [15.0 - 11.2, 15.3 - 11.8])
    np.testing.assert_array_equal(target_values(ev.a, None, np.array([4]), "a"), [104.0])
    np.testing.assert_allclose(target_values(None, ev.v, np.array([0]), "fwd_diff"), [5.0])  # 0.5 m/s in 0.1 s
    assert len(make_windows([event_from_speed(V_HAND[:12])], hist_len=3, target="mean_1s")) == 0  # 2 + 10 = n


def test_mean_1s_is_the_mean_of_ten_forward_differences() -> None:
    events = synthetic_events()
    w = make_windows(events, target="mean_1s")
    fd = make_windows(events, target="fwd_diff")
    row = {(int(e), int(t)): i for i, (e, t) in enumerate(zip(fd.event_index, fd.t_index))}
    for i in np.random.default_rng(0).choice(len(w), size=60, replace=False):
        e, t = int(w.event_index[i]), int(w.t_index[i])
        steps = [row[(e, t + k)] for k in range(MEAN_TARGET_STEPS)]  # forward differences at t .. t+9
        assert w.y[i] == pytest.approx(float(np.mean(fd.y[steps].astype(np.float64))), abs=2e-5)
    # every event loses the windows whose next second leaves the event
    for e, ev in enumerate(events):
        t_last = w.t_index[w.event_index == e].max()
        assert t_last == len(ev) - 1 - MEAN_TARGET_STEPS


def test_mean_1s_equals_mean_of_the_acceleration_sequence_on_synthetic_data() -> None:
    """The generator keeps v[t+1] = v[t] + a[t] DT: the target is the mean of a[t : t+10] (DiffusionCF's law)."""
    w = make_windows(synthetic_events(), seq_len=20, stride=2, target="mean_1s")
    np.testing.assert_allclose(w.y, w.y_seq[:, :MEAN_TARGET_STEPS].astype(np.float64).mean(axis=1), atol=2e-5)


def test_lookahead_together_with_seq_len() -> None:
    ev = event_from_speed(10.0 + 0.1 * np.arange(30.0))
    short = make_windows([ev], hist_len=3, seq_len=5, target="mean_1s")  # the target needs more than the sequence
    np.testing.assert_array_equal(short.t_index, np.arange(2, 20))  # t + 10 < 30
    np.testing.assert_array_equal(short.y_seq, np.stack([ev.a[t : t + 5] for t in range(2, 20)]).astype(np.float32))
    np.testing.assert_array_equal(short.v_lead_future, np.stack([ev.v_lead[t + 1 : t + 6] for t in range(2, 20)]).astype(np.float32))
    np.testing.assert_allclose(short.y, 1.0, rtol=1e-5)  # speed ramp of 0.1 m/s per sample = 1 m/s^2
    long = make_windows([ev], hist_len=3, seq_len=15, target="mean_1s")  # the sequence needs more than the target
    np.testing.assert_array_equal(long.t_index, np.arange(2, 15))  # t + 15 < 30
    assert long.y_seq.shape == (13, 15)
    empty = make_windows([event_from_speed(V_HAND)], hist_len=5, seq_len=3, target="mean_1s")
    assert len(empty) == 0 and empty.y_seq.shape == (0, 3)


# ------------------------------------------------------------------------ windows table, y_prev
def test_y_prev_of_the_one_step_target_is_a_prev() -> None:
    events = synthetic_events()
    w = make_windows(events, stride=3)
    frame, meta = tables.windows_table(events, w)
    assert list(frame.columns) == list(tables.WINDOW_COLUMNS) and frame["y_prev"].dtype == np.float32
    assert meta["target"] == "a" and tables.windows_target(meta) == "a"
    np.testing.assert_array_equal(frame["y_prev"], frame["a_prev"])
    expected = [events[int(e)].a[int(t) - 1] for e, t in zip(frame["event_index"], frame["t_index"])]
    np.testing.assert_array_equal(frame["y_prev"], np.asarray(expected, dtype=np.float32))


def test_y_prev_of_mean_1s_by_hand_and_as_the_target_ten_samples_earlier() -> None:
    v = 10.0 + 0.5 * np.arange(25.0) + 0.01 * np.arange(25.0) ** 2
    ev = event_from_speed(v)
    w = make_windows([ev], hist_len=3, target="mean_1s")  # t = 2 .. 14
    frame, meta = tables.windows_table([ev], w, target="mean_1s")
    assert meta["target"] == "mean_1s"
    t = frame["t_index"].to_numpy()
    early = t < MEAN_TARGET_STEPS
    assert np.isnan(frame["y_prev"][early]).all() and early.sum() == 8  # no 10 earlier samples before t = 10
    # (v[t] - v[t-10]) / 1 s with v = 10 + 0.5 i + 0.01 i^2: 5 + 0.01 (20 t - 100) = 4 + 0.2 t
    np.testing.assert_allclose(frame["y_prev"][~early], 4.0 + 0.2 * t[~early], rtol=1e-6)
    np.testing.assert_allclose(frame["y"], 6.0 + 0.2 * t, rtol=1e-6)  # (v[t+10] - v[t]) / 1 s = 6 + 0.2 t
    y_of = dict(zip(t.tolist(), frame["y"]))  # stride 1: y_prev at t is the target of the window at t - 10
    pairs = [(prev, y_of[tt - MEAN_TARGET_STEPS]) for tt, prev in zip(t.tolist(), frame["y_prev"]) if tt - MEAN_TARGET_STEPS in y_of]
    assert len(pairs) == 3 and all(prev == y for prev, y in pairs)  # windows t = 12, 13, 14
    np.testing.assert_array_equal(frame["a_prev"], (100.0 + t - 1).astype(np.float32))  # a_prev stays a[t-1]


def test_y_prev_with_a_three_second_history_is_always_finite() -> None:
    events = synthetic_events()
    w = make_windows(events, stride=2, target="mean_1s")
    frame, _ = tables.windows_table(events, w, target="mean_1s")
    assert np.isfinite(frame["y_prev"]).all()
    expected = [(ev.v[k] - ev.v[k - 10]) / (10 * DT) for ev, k in ((events[int(e)], int(t)) for e, t in zip(frame["event_index"], frame["t_index"]))]
    np.testing.assert_allclose(frame["y_prev"], np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_windows_target_of_old_and_new_tables(tmp_path: Path) -> None:
    assert tables.windows_target({"spec": {"data": "highd"}}) == "a"  # table written before the target option
    assert tables.windows_target({"spec": {"target": "mean_1s"}}) == "mean_1s"
    assert tables.windows_target({"target": "mean_1s", "spec": {}}) == "mean_1s"
    assert tables.root_target(tmp_path) == "a"
    events = synthetic_events()
    frame, meta = tables.windows_table(events, make_windows(events, stride=4, target="mean_1s"), target="mean_1s")
    tables.save_windows(tmp_path, "highd", frame, meta)
    assert tables.root_target(tmp_path) == "mean_1s"
    stored, stored_meta = tables.load_windows(tmp_path, "highd")
    pd.testing.assert_frame_equal(stored, frame)
    assert json.loads(json.dumps(stored_meta)) == stored_meta


# ------------------------------------------------------------------------------------ pipeline
def test_bundles_pass_the_target_through(tmp_path: Path) -> None:
    cfg = dict(n_drivers=12, events_per_driver=2, duration_s=12.0, seed=5)
    bundle = synthetic_bundle(cfg, stride=2, target="mean_1s")
    plain = synthetic_bundle(cfg, stride=2)
    for split in ("train", "val", "test"):
        w, events = bundle.windows[split], bundle.events[split]
        expected = [target_values(None, events[int(e)].v, int(t), "mean_1s") for e, t in zip(w.event_index, w.t_index)]
        np.testing.assert_allclose(w.y, np.asarray(expected, dtype=np.float32), rtol=1e-6)
        assert len(w) < len(plain.windows[split])
        assert len(bundle.truth[split]["a_det"]) == len(w)
    events, _ = generate(SyntheticConfig(**cfg))
    events.to_parquet(tmp_path / "events")
    loaded = load_bundle(tmp_path / "events", stride=2, target="mean_1s", unify=False)
    np.testing.assert_array_equal(loaded.windows["test"].y, bundle.windows["test"].y)
