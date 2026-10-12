"""Glue code: bundles, truth alignment, hydra configs, the training entry point."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from hydra import compose, initialize_config_dir

from cf_unc.constants import DT, HIST_LEN, SEQ_LEN
from cf_unc.data.pipeline import SPLITS, synthetic_bundle
from cf_unc.physics.idm import idm_accel

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_bundle_is_driver_disjoint_and_aligned(synth_small, true_idm):
    drivers = {k: {ev.follower_id for ev in synth_small.events[k]} for k in SPLITS}
    assert not (drivers["train"] & drivers["val"]) and not (drivers["train"] & drivers["test"])
    assert not (drivers["val"] & drivers["test"])
    for split in SPLITS:
        w, truth = synth_small.windows[split], synth_small.truth[split]
        assert w.X.shape[1:] == (HIST_LEN, 3) and len(truth["a_det"]) == len(w)
        last = w.X[:, -1, :].astype(np.float64)
        a_idm = idm_accel(last[:, 0], last[:, 1], last[:, 2], true_idm)
        # float32 storage of the state limits the agreement
        assert np.max(np.abs(a_idm - truth["a_det"])) < 1e-4
        events = synth_small.events[split]
        for i in (0, len(w) // 2, len(w) - 1):
            ev = events[int(w.event_index[i])]
            t = int(w.t_index[i])
            assert w.y[i] == pytest.approx(ev.a[t], abs=1e-6)
            assert w.X[i, -1, 1] == pytest.approx(ev.v[t] - ev.v_lead[t], abs=1e-5)  # dv = v - v_lead
            assert w.X[i, 0, 0] == pytest.approx(ev.s[t - HIST_LEN + 1], abs=1e-4)


def test_sequence_bundle_kinematics(synth_small_seq):
    w = synth_small_seq.windows["train"]
    assert w.y_seq.shape == (len(w), SEQ_LEN) and w.v_lead_future.shape == (len(w), SEQ_LEN)
    assert np.allclose(w.y_seq[:, 0], w.y, atol=1e-6)
    events = synth_small_seq.events["train"]
    i = len(w) // 3
    ev, t = events[int(w.event_index[i])], int(w.t_index[i])
    v_end = ev.v[t] + DT * np.sum(w.y_seq[i].astype(np.float64))
    assert v_end == pytest.approx(ev.v[t + SEQ_LEN], abs=1e-4)
    assert w.v_lead_future[i, -1] == pytest.approx(ev.v_lead[t + SEQ_LEN], abs=1e-5)


def test_bundle_is_reproducible():
    cfg = dict(n_drivers=6, events_per_driver=1, duration_s=10.0, seed=3)
    a = synthetic_bundle(cfg, name="a", stride=3)
    b = synthetic_bundle(cfg, name="b", stride=3)
    for split in SPLITS:
        assert np.array_equal(a.windows[split].X, b.windows[split].X)
        assert np.array_equal(a.truth[split]["a_det"], b.truth[split]["a_det"])


@pytest.mark.parametrize("path", sorted((CONFIGS / "model").glob("*.yaml")), ids=lambda p: p.stem)
def test_model_configs_are_constructible(path, true_idm):
    from cf_unc.models import MODEL_REGISTRY, build_model, constructor_accepts

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(cfg) <= {"name", "kwargs", "train", "needs_idm", "needs_sequences", "idm"}
    assert cfg["name"] in MODEL_REGISTRY
    kwargs = dict(cfg.get("kwargs") or {})
    if cfg.get("needs_idm"):
        kwargs["idm"] = true_idm.to_dict()
    if constructor_accepts(cfg["name"], "train"):
        kwargs["train"] = cfg.get("train") or {}
    else:
        assert not cfg.get("train"), "train overrides for a model without a `train` argument"
    model = build_model(cfg["name"], **kwargs)
    assert model.name == cfg["name"]
    # a plan sampler needs sequence windows; a model trained on them need not be one (the closed-loop fine-tuned
    # networks of extension S7 use the 5 s futures in their loss and stay one-step laws)
    assert model.plan_sampler <= bool(cfg.get("needs_sequences", False))
    assert model.needs_leader_future <= model.plan_sampler  # only a plan sampler may take the leader's future


def test_run_training_smoke(tmp_path):
    pytest.importorskip("cf_unc.models.baselines")
    from cf_unc.train.run import format_row, run_training

    overrides = [
        "model=idm_white_noise",
        "model_label=idm_white_noise",
        "data=synthetic",
        "data.generator.n_drivers=12",
        "data.generator.events_per_driver=1",
        "data.generator.duration_s=15.0",
        "idm.objective=accel",
        "idm.max_events=8",
        # quoted: hydra's override grammar rejects bare non-ASCII characters (the temp path may contain them)
        f"idm.cache='{(tmp_path / 'idm.json').as_posix()}'",
        f"out_dir='{(tmp_path / 'run').as_posix()}'",
        "n_boot=50",
        "stride=2",
    ]
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    result = run_training(cfg)
    row = format_row(result)
    assert np.isfinite(row["nll"]) and 0.0 < row["picp90"] <= 1.0
    # identical drivers, constant noise: the fitted sigma^2 is the generator's variance
    assert row["aleatoric"] == pytest.approx(0.16, rel=0.15)
    assert row["epistemic"] == 0.0
    out = Path(result["out_dir"])
    assert (out / "model.pt").exists() and (out / "config.yaml").exists()
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test"]["nll"]["lo"] <= metrics["test"]["nll"]["estimate"] <= metrics["test"]["nll"]["hi"]
    assert (tmp_path / "idm.json").exists()

    from cf_unc.models import load_model

    model = load_model(out / "model.pt")
    assert model.name == "idm_white_noise" and model.fitted
