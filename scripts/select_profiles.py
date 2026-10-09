"""Select the leader profiles of experiment E3 from the raw OpenACC files (docs/design.md, section 13.1).

    python scripts/select_profiles.py                                   # -> configs/platoon/profiles.yaml
    python scripts/select_profiles.py --out /tmp/profiles.yaml --max-profiles 30
    python scripts/select_profiles.py --no-speed-bound                  # the rules of design.md 13.1 only

Reads the run files of the campaigns AstaZero, ZalaZone and Vicolungo (read-only), harmonises the speeds
(Savitzky-Golay, 11 samples, order 2), cuts candidate windows and selects them round-robin over (campaign, driver mode)
(``cf_unc.platoon.profiles``). Speed bound (on by default): the leader speed must stay at or below the speed at which
the equilibrium gap of the closed-loop IDM (``--idm``, calibration ``closed_loop``) reaches the upper gap bound
(120 m); above it the initial state of the simulation (IDM equilibrium gap, design.md 13.2) does not exist and the
reference model cannot follow the leader. Writes the profiles file (deterministic, the only input of the experiment)
and prints the selected profiles and the counts per campaign and driver mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from cf_unc.physics.idm import IDMParams, equilibrium_gap
from cf_unc.platoon.openacc import CAMPAIGNS, load_runs
from cf_unc.platoon.profiles import MODES, SelectionRules, profiles_digest, select_profiles, write_profiles

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT.parent / "cf-stability" / "data" / "raw" / "openacc"
DEFAULT_OUT = ROOT / "configs" / "platoon" / "profiles.yaml"
DEFAULT_IDM = ROOT / "artifacts" / "e1" / "idm.json"


def _rel(path: Path) -> str:
    """Path relative to the repository root where possible (no user-specific absolute paths in the file)."""
    p = Path(path).resolve()
    for base, prefix in ((ROOT, ""), (ROOT.parent, "../")):
        try:
            return prefix + p.relative_to(base).as_posix()
        except ValueError:
            continue
    return p.as_posix()


def idm_speed_bound(idm_file: Path, gap_max: float) -> tuple[float, dict]:
    """Speed [m/s] at which the equilibrium gap of the closed-loop IDM equals ``gap_max`` [m], and the parameters."""
    params = json.loads(idm_file.read_text(encoding="utf-8"))["calibrations"]["closed_loop"]["params"]
    idm = IDMParams.from_dict(params)
    v = brentq(lambda x: equilibrium_gap(x, idm) - gap_max, 0.0, float(idm.v0) * (1.0 - 1e-9), xtol=1e-10)
    return float(v), params


def format_table(profiles: list) -> str:
    head = (f"{'id':4s} {'set':8s} {'campaign':9s} {'mode':6s} {'file':34s} {'start_s':>10s} {'len_s':>6s} {'nf':>3s} "
            f"{'A0':>6s} {'v_init':>6s} {'v_max':>6s} {'dev0':>5s}  follower amplitudes [m/s]")  # fmt: skip
    lines = [head]
    for p in profiles:
        amp = " ".join(f"{a:.1f}" for a in p.follower_amplitude)
        lines.append(
            f"{p.id:4s} {'headline' if p.headline else 'extended':8s} {p.campaign:9s} {p.mode:6s} {p.file:34s} "
            f"{p.start_s:10.1f} {p.length_s:6.1f} {p.n_followers:3d} {p.leader_amplitude:6.2f} {p.initial_speed:6.2f} "
            f"{float(np.max(p.leader_speed)):6.2f} {p.initial_deviation:5.2f}  {amp}"
        )
    return "\n".join(lines)


def format_counts(profiles: list, counts: dict) -> str:
    lines = ["", "Candidates and selected profiles per campaign and driver mode (headline in brackets):"]
    selected = Counter((p.campaign, p.mode) for p in profiles)
    headline = Counter((p.campaign, p.mode) for p in profiles if p.headline)
    lines.append(f"{'campaign':10s} " + " ".join(f"{m:>18s}" for m in MODES) + f" {'total':>10s}")
    for c in CAMPAIGNS:
        cells = []
        for m in MODES:
            cand = counts["candidates"].get(f"{c}/{m}", 0)
            cells.append(f"{cand:5d} -> {selected[(c, m)]:3d} [{headline[(c, m)]:2d}]")
        lines.append(f"{c:10s} " + " ".join(f"{x:>18s}" for x in cells) + f" {sum(selected[(c, m)] for m in MODES):10d}")
    nf = np.array([p.n_followers for p in profiles])
    amp = np.array([p.leader_amplitude for p in profiles])
    length = np.array([p.length_s for p in profiles])
    if len(profiles):
        lines.append(f"followers per profile: min {nf.min()}, max {nf.max()}, mean {nf.mean():.1f}; leader amplitude "
                     f"min / median / max {amp.min():.2f} / {np.median(amp):.2f} / {amp.max():.2f} m/s; length "
                     f"min / median / max {length.min():.0f} / {np.median(length):.0f} / {length.max():.0f} s")  # fmt: skip
    if len(profiles):
        dev = np.array([p.initial_deviation for p in profiles])
        lines.append(f"initial deviation of the measured followers from the leader's first speed (dev0, first 5 s): median "
                     f"{np.median(dev):.2f} m/s, {np.mean(dev > 1.0):.0%} of the profiles above 1 m/s")  # fmt: skip
    lines.append(f"runs: {counts['runs']}, without a platoon of the leader and 3 followers: "
                 f"{len(counts['runs_without_platoon'])}")  # fmt: skip
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default=str(DEFAULT_RAW), help="raw OpenACC directory (campaign folders)")
    parser.add_argument("--campaigns", nargs="+", default=list(CAMPAIGNS))
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="profiles file to write")
    parser.add_argument("--idm", default=str(DEFAULT_IDM), help="IDM calibration file (closed_loop) of the speed bound")
    parser.add_argument("--no-speed-bound", action="store_true", help="no bound on the leader speed")
    parser.add_argument("--max-profiles", type=int, default=SelectionRules.max_profiles)
    parser.add_argument("--headline", type=int, default=SelectionRules.n_headline)
    args = parser.parse_args(argv)

    rules = SelectionRules(max_profiles=args.max_profiles, n_headline=args.headline)
    idm_params = None
    if not args.no_speed_bound:
        bound, idm_params = idm_speed_bound(Path(args.idm), rules.gap_bounds[1])
        rules = SelectionRules(max_profiles=args.max_profiles, n_headline=args.headline, max_speed=round(bound, 4))
    runs = load_runs(args.raw, args.campaigns)
    profiles, counts = select_profiles(runs, rules)
    meta = {
        "description": "Leader profiles of experiment E3 (docs/design.md 13.1), written by scripts/select_profiles.py",
        "source": _rel(Path(args.raw)),
        "campaigns": list(args.campaigns),
        "rules": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(rules).items()},
        "speed_bound": None if idm_params is None else {
            "idm_file": _rel(Path(args.idm)), "calibration": "closed_loop", "params": idm_params,
            "rule": "leader speed <= speed at which the IDM equilibrium gap equals rules.gap_bounds[1]",
        },  # fmt: skip
        "n_profiles": len(profiles),
        "n_headline": sum(p.headline for p in profiles),
        "digest": profiles_digest(profiles),
        "counts": counts,
    }
    path = write_profiles(args.out, profiles, meta)
    print(format_table(profiles))
    print(format_counts(profiles, counts))
    print(f"\n{len(profiles)} profiles written to {path} (digest {meta['digest']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
