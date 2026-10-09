"""Event schema: validation rules and parquet round trip (pandas >= 3 string dtypes)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cf_unc.constants import DT
from cf_unc.data.schema import (
    ARRAY_FIELDS,
    EVENTS_FILE,
    ID_FIELDS,
    MANIFEST_FILE,
    SAMPLES_FILE,
    Event,
    EventSet,
    EventValidationError,
)


def make_event(k: int = 0, n: int = 50, **overrides) -> Event:
    """A kinematically consistent event; ``overrides`` replace constructor arguments."""
    t = np.arange(n) * DT
    v = 12.0 + np.sin(0.3 * t + k)
    v_lead = 12.0 + np.cos(0.2 * t + k)
    x_follower = 100.0 * k + np.cumsum(v) * DT
    x_lead = x_follower + 20.0 + np.sin(0.1 * t)
    kwargs = dict(
        event_id=f"ds_{k:03d}",
        dataset="ds",
        site="site/Город",
        follower_id=f"drv{k}",
        leader_id=f"lead{k}",
        t=t,
        s=x_lead - x_follower,
        dv=v - v_lead,
        v=v,
        a=np.gradient(v, DT) if n > 1 else np.zeros(n),
        v_lead=v_lead,
        x_lead=x_lead,
        x_follower=x_follower,
        meta={"k": k, "lane": np.int64(2), "gain": np.float64(0.5), "tags": ["x", "ü"], "arr": np.arange(3)},
    )
    kwargs.update(overrides)
    return Event(**kwargs)


def test_event_arrays_are_float64_and_valid():
    ev = make_event(n=30)
    ev.validate()
    for name in ARRAY_FIELDS:
        arr = getattr(ev, name)
        assert arr.dtype == np.float64 and arr.flags.c_contiguous and arr.shape == (30,)
    assert len(ev) == 30
    assert ev.duration == pytest.approx(29 * DT)


def test_parquet_round_trip(tmp_path):
    es = EventSet([make_event(k, n=20 + 7 * k) for k in range(5)])
    directory = tmp_path / "событие" / "ds"  # non-ASCII path on purpose
    assert es.to_parquet(directory, manifest={"source": "unit-test"}) == directory

    # on-disk layout identical to cf-stability
    samples = pd.read_parquet(directory / SAMPLES_FILE)
    events = pd.read_parquet(directory / EVENTS_FILE)
    assert list(samples.columns) == ["event_id", *ARRAY_FIELDS]
    assert list(events.columns) == ["event_id", *ID_FIELDS, "n_samples", "duration", "meta_json"]
    assert len(samples) == sum(len(ev) for ev in es)
    manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1 and manifest["dt"] == DT and manifest["source"] == "unit-test"
    assert manifest["summary"]["n_events"] == 5

    back = EventSet.from_parquet(directory)
    back.validate()
    assert len(back) == len(es)
    for a, b in zip(es, back):
        for name in ("event_id", *ID_FIELDS):
            assert getattr(b, name) == getattr(a, name)
            assert type(getattr(b, name)) is str
        for name in ARRAY_FIELDS:
            np.testing.assert_array_equal(getattr(b, name), getattr(a, name))
        # numpy scalars / arrays in meta come back as JSON numbers / lists
        assert b.meta == {"k": a.meta["k"], "lane": 2, "gain": 0.5, "tags": ["x", "ü"], "arr": [0, 1, 2]}
    assert back["ds_003"].follower_id == "drv3"
    assert [ev.event_id for ev in back.subset(["ds_004", "ds_001"])] == ["ds_004", "ds_001"]


def test_empty_event_set_round_trip(tmp_path):
    EventSet([]).to_parquet(tmp_path / "empty")
    back = EventSet.from_parquet(tmp_path / "empty")
    assert len(back) == 0
    assert back.summary()["n_events"] == 0


def test_duplicate_event_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        EventSet([make_event(1), make_event(1)])


def _shifted(name: str, index: int, delta: float) -> np.ndarray:
    arr = getattr(make_event(), name).copy()
    arr[index] += delta
    return arr


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dv": _shifted("dv", 7, 0.01)}, "dv != v - v_lead"),
        ({"t": _shifted("t", 10, 0.05)}, "not uniform"),
        ({"t": np.arange(50) * 0.2}, "not uniform"),
        ({"s": _shifted("s", 5, 0.5)}, "s != x_lead - x_follower"),
        ({"v": _shifted("v", 3, -30.0), "dv": _shifted("dv", 3, -30.0)}, "negative speed"),
        ({"a": _shifted("a", 2, np.nan)}, "non-finite"),
    ],
)
def test_validate_catches_violations(overrides, message):
    with pytest.raises(EventValidationError, match=message):
        make_event(**overrides).validate()


@pytest.mark.parametrize("bad_gap", [0.0, -2.0])
def test_validate_catches_non_positive_gap(bad_gap):
    ev = make_event()
    s = ev.s.copy()
    s[12] = bad_gap
    # keep s == x_lead - x_follower so that only the gap rule is violated
    bad = make_event(s=s, x_lead=ev.x_follower + s)
    with pytest.raises(EventValidationError, match="non-positive spacing"):
        bad.validate()


def test_validate_catches_short_event():
    with pytest.raises(EventValidationError, match="fewer than 2 samples"):
        make_event(n=1).validate()
