"""Harmonisation report (docs/design.md, section 12): every imported data set brought to one signal processing.

    python scripts/harmonise_report.py                 # -> artifacts/e1h/harmonisation.json and markdown tables
    python scripts/harmonise_report.py --raw-config configs/experiment/e1.yaml --config configs/experiment/e1h.yaml

1. Per data set, all events of its event directory (``cf_unc.data.harmonise.harmonise_events`` with the settings
   of ``--config``): events in / out, dropped by reason, quantiles (5, 50, 95 %) of ``gap_rms`` and ``|c|``, the
   kinematic check ``rms(ds/dt + dv)`` [m/s] before and after, speed samples floored at 0.
2. Per data set, the evaluated windows of E1 (split, stride and event cap of the experiment config) cut for the
   targets ``a`` and ``mean_1s``, raw (``--raw-config``) and harmonised (``--config``): std of the target and RMSE
   of persistence (the "signal" table of scripts/analyze_e1_extra.py, ``signal_row``).
3. Per data set, all events, raw and harmonised: roughness of the model inputs, rms of the second difference
   ``x[t+1] - 2 x[t] + x[t-1]`` of ``v``, ``dv`` [m/s] and ``s`` [m].
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cf_unc.constants import HIST_LEN
from cf_unc.data.schema import EventSet
from cf_unc.data.windows import make_windows
from cf_unc.experiments.e1 import Data, load_config
from cf_unc.maps.io import windows_table
from cf_unc.train.run import ROOT, _path, write_json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_e1_extra import signal_row  # noqa: E402  (scripts/ is not a package)

DATASETS = ("highd", "ngsim", "ngsim_us101", "ngsim_fn", "waymo", "openacc")  # evaluated data sets of E1 (keys)
TARGETS = ("a", "mean_1s")


def signal_rows(data: Data, cfg: Any, datasets: tuple[str, ...], signal: str) -> list[dict[str, Any]]:
    """``signal_row`` of the evaluated windows of every data set and target (split, stride, cap of ``cfg.eval``)."""
    rows = []
    for name in datasets:
        d = cfg.eval.datasets[name]
        events = data.split(str(d.data), str(d.get("split", "all")))
        cap = d.get("max_events", cfg.eval.get("max_events"))
        if cap is not None:
            events = EventSet(list(events)[: int(cap)])
        for target in TARGETS:
            windows = make_windows(events, hist_len=HIST_LEN, stride=int(d.get("stride", 1)), target=target)
            frame, _ = windows_table(events, windows, None, target=target)
            rows.append({"signal": signal, **signal_row(name, frame, target)})
    return rows


def roughness(events: EventSet) -> dict[str, float]:
    """rms of the second difference ``x[t+1] - 2 x[t] + x[t-1]`` over all samples of the events, x = v, dv [m/s], s [m]."""
    out = {}
    for name in ("v", "dv", "s"):
        d2 = np.concatenate([np.diff(getattr(ev, name), 2) for ev in events])
        out[name] = float(np.sqrt(np.mean(d2 * d2)))
    return out


def _q(report: dict[str, Any], key: str, digits: int = 3) -> str:
    q = report[key]
    return " / ".join("-" if q[k] is None else f"{q[k]:.{digits}g}" for k in ("q05", "q50", "q95"))


def markdown(
    reports: dict[str, dict[str, Any]], sources: dict[str, str], signal: pd.DataFrame, rough: dict[str, dict[str, Any]]
) -> str:
    """Three markdown tables: the harmonisation per data set, the predictability of the targets, the input roughness."""
    lines = [
        "| data set | events in | kept | dropped: min gap / abs c / gap rms (share) | gap_rms m, q05 / q50 / q95 | "
        "abs c m/s, q05 / q50 / q95 | rms(ds/dt + dv) m/s, before -> after | speed floored: samples (events) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in reports.items():
        d = r["dropped"]
        dropped = f"{d['min_gap']} / {d['max_abs_c']} / {d['max_gap_rms']}" + (f" + {d['non_finite']} non-finite" if d["non_finite"] else "")
        dropped += f" + {d['short_event']} short" if d.get("short_event") else ""  # shorter than the window (design.md 15.1)
        lines.append(
            f"| {name} ({sources[name]}) | {r['n_in']:,} | {r['n_out']:,} | {dropped} ({100 * r['share_dropped']:.2f} %) | "
            f"{_q(r, 'gap_rms_m')} | {_q(r, 'abs_c_mps')} | {r['kinematic_rms_before']:.3g} -> {r['kinematic_rms_after']:.1e} | "
            f"{r['speed_floored']['samples']:,} ({r['speed_floored']['events']:,}) |"
        )
    lines += ["", "| data set | target | windows raw / harm. | std raw -> harm. | persistence RMSE raw -> harm. | "
              "persistence / std raw -> harm. |", "|---|---|---|---|---|---|"]  # fmt: skip
    piv = signal.set_index(["dataset", "target", "signal"])
    for name in dict.fromkeys(signal["dataset"]):
        for target in TARGETS:
            raw, harm = piv.loc[(name, target, "raw")], piv.loc[(name, target, "harmonised")]
            lines.append(
                f"| {name} | {target} | {int(raw.n_windows):,} / {int(harm.n_windows):,} | {raw.std_a:.3f} -> {harm.std_a:.3f} | "
                f"{raw.rmse_persistence:.4f} -> {harm.rmse_persistence:.4f} | "
                f"{raw.rmse_persistence / raw.std_a:.3f} -> {harm.rmse_persistence / harm.std_a:.3f} |"
            )
    lines += ["", "| data set | rms second difference of v, m/s: raw -> harm. | of dv, m/s | of s, m |", "|---|---|---|---|"]
    for name, r in rough.items():
        cells = [f"{r['raw'][k]:.2e} -> {r['harmonised'][k]:.2e}" for k in ("v", "dv", "s")]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/experiment/e1h.yaml", help="harmonised experiment config")
    parser.add_argument("--raw-config", default="configs/experiment/e1.yaml", help="raw experiment config (same data sets)")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), help="keys of eval.datasets")
    parser.add_argument("--out", default="artifacts/e1h/harmonisation.json", help="JSON output")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    cfg, raw_cfg = load_config(args.config), load_config(args.raw_config)
    harm, raw = Data(cfg), Data(raw_cfg)
    datasets = tuple(args.datasets)
    names = {d: str(cfg.eval.datasets[d].data) for d in datasets}
    reports, sources, rough = {}, {}, {}
    for d in datasets:
        kept = harm.events(names[d])
        report = harm.harmonisation(names[d])
        if report is None:
            raise SystemExit(f"{args.config}: data set {names[d]!r} is not harmonised (signal: {harm.signal(names[d])})")
        reports[d] = report
        sources[d] = Path(str(harm.config(names[d]).path)).name
        rough[d] = {"raw": roughness(raw.events(names[d])), "harmonised": roughness(kept)}
        print(f"{d}: {report['n_out']:,} of {report['n_in']:,} events kept in {report['seconds']:.2f} s", flush=True)
    signal = pd.DataFrame(signal_rows(raw, raw_cfg, datasets, "raw") + signal_rows(harm, cfg, datasets, "harmonised"))
    payload = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": args.config,
        "raw_config": args.raw_config,
        "settings": next(iter(reports.values()))["settings"] if reports else None,
        "sources": sources,
        "harmonisation": reports,
        "signal": signal.to_dict("records"),
        "roughness": rough,
        "seconds": round(time.perf_counter() - started, 1),
    }
    out = _path(args.out)
    write_json(out, payload)
    print()
    print(markdown(reports, sources, signal, rough))
    print(f"\nwritten to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} in {payload['seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
