#!/usr/bin/env python3
"""Reference IDM calibrated per driver mode (extension S6, extension/docs/PLAN.md, section S).

The reference of H3.2 is the IDM calibrated on the highD training events with a white-noise sigma. This script
calibrates the IDM (closed-loop spacing objective of E1, cf_unc.physics.idm.calibrate_idm, 300 events, seed 0) on the
OpenACC events driven under ACC and, separately, on those driven by humans (the mode is the suffix of the follower id
of the stored events), leaving out the recordings of the 60 platoon profiles (the third token of the event id is the
recording file of the pair); the white-noise sigma of each reference is the RMS residual on the windows of its own
mode (stride 5, as the OpenACC evaluation set of E1). Checkpoints, identical for the five seeds as the seed-independent
checkpoints of E1:

    extension/artifacts/ext_idm_modes/<acc|human>/checkpoints/idm_white_noise/seed{0..4}/{model.pt,train_log.json}

and the summary extension/artifacts/ext_idm_modes/idm_modes.json; the marker extension/artifacts/ext_idm_modes/checkpoints/DONE
releases the platoon units of ext_e3_idmref.yaml in the GPU queue.

    PYTHONUTF8=1 python extension/scripts/calibrate_idm_modes.py [--max-events 300] [--seed 0] [--seeds 5]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from cf_unc.constants import HIST_LEN
from cf_unc.data.pipeline import load_events
from cf_unc.data.windows import make_windows
from cf_unc.models import build_model
from cf_unc.physics.idm import calibrate_idm
from cf_unc.platoon.profiles import load_profiles

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "extension/artifacts/ext_idm_modes"
MODES = {"acc": "ACC", "human": "Human"}
STRIDE = 5  # windows of the OpenACC evaluation set of E1


def mode_of(event) -> str | None:  # noqa: ANN001
    m = re.search(r":(ACC|Human)", event.follower_id)
    return None if m is None else m.group(1)


def recording_of(event) -> str:  # noqa: ANN001
    tokens = event.event_id.split("|")
    return tokens[2] if len(tokens) > 2 else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--max-events", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5, help="number of identical seed copies of every checkpoint")
    a = ap.parse_args()
    events = load_events(ROOT / "data/events/openacc")
    profiles, _ = load_profiles(ROOT / "configs/platoon/profiles.yaml")
    stems = {Path(p.file).stem for p in profiles}
    summary = {
        "source": "data/events/openacc", "objective": "spacing", "max_events": a.max_events, "seed": a.seed,
        "sigma_windows": {"hist_len": HIST_LEN, "stride": STRIDE, "target": "a"},
        "excluded_recordings": sorted(stems), "n_events": len(events), "modes": {},
    }  # fmt: skip
    print(f"{len(events)} OpenACC events; {len(stems)} profile recordings left out of the calibrations")
    for key, mode in MODES.items():
        pool = [e for e in events if mode_of(e) == mode]
        held = [e for e in pool if recording_of(e) not in stems]
        started = time.perf_counter()
        params, report = calibrate_idm(held, objective="spacing", max_events=a.max_events, seed=a.seed)
        windows = make_windows(held, hist_len=HIST_LEN, stride=STRIDE, target="a")
        model = build_model("idm_white_noise", idm=params.to_dict(), sigma=None)
        fit_log = model.fit(windows, None, seed=a.seed)
        seconds = time.perf_counter() - started
        entry = {
            "mode": mode, "n_events_mode": len(pool), "n_events_calibration_pool": len(held),
            "n_windows_sigma": int(len(windows)), "params": params.to_dict(), "sigma": float(model.sigma),
            "report": report, "seconds": seconds,
        }  # fmt: skip
        summary["modes"][key] = entry
        for seed in range(a.seeds):
            directory = OUT / key / "checkpoints" / "idm_white_noise" / f"seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            info = {
                "model": "idm_white_noise", "registry_name": "idm_white_noise", "seed": seed, "seed_independent": True,
                "kwargs": {"idm": params.to_dict(), "sigma": None}, "idm": {"name": f"openacc_{key}", "objective": "spacing"},
                "target": "a", "n_train": int(len(windows)), "n_val": 0, "seconds": seconds, "log": fit_log,
                "calibration": {k: v for k, v in entry.items() if k != "report"},
            }  # fmt: skip
            (directory / "train_log.json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
            model.save(directory / "model.pt")
        p = params.to_dict()
        print(f"{mode}: {len(pool)} events, {len(held)} in the calibration pool, {len(windows)} windows; "
              f"v0 {p['v0']:.2f} T {p['T']:.2f} s0 {p['s0']:.2f} a_max {p['a_max']:.2f} b {p['b']:.2f}; "
              f"sigma {model.sigma:.3f} m/s^2; {seconds:.0f} s")  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "idm_modes.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    (OUT / "checkpoints").mkdir(exist_ok=True)
    (OUT / "checkpoints" / "DONE").write_text("calibrations written\n", encoding="utf-8")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
