"""Roughness matching of the signal variant ``matched`` (docs/design.md, section 15.1, experiment E5).

    python scripts/match_roughness.py --config configs/experiment/e1m.yaml     # -> match.file (artifacts/e1m/matching.json)
    python scripts/match_roughness.py --config configs/experiment/e1m.yaml --force   # replace a file with other windows

For every data set of the experiment config (``match.reference``, the training data ``train.data`` and the data of
``eval.datasets`` / ``datasets``) and every window of ``match.grid`` (and ``harmonise.window``): ALL events of the data
set (never a split) harmonised with that window (``cf_unc.data.harmonise.harmonise_events``, the other settings of
``harmonise``; events shorter than the window are dropped as ``short_event``), roughness = the ``v`` entry of
``scripts/harmonise_report.py:roughness`` of the kept events (rms of the second difference of the harmonised follower
speed). The reference keeps ``harmonise.window``; every other data set gets the window of
``cf_unc.data.harmonise.choose_window``: the smallest grid window whose roughness is at most ``(1 + match.tolerance)``
times the reference roughness (none: the largest, flag ``not_reached``), capped at the largest grid window that keeps
the share of short events below 5 % (flag ``capped``). The events are read with the loader of
``scripts/harmonise_report.py`` (``cf_unc.experiments.e1.Data``).

Writes the matching file (settings, reference and target roughness, ``windows`` = the window per data set that the
runners read, and per data set: role, window, flags, roughness, drop shares by reason including ``short_event``, and
all of it per grid window) and prints the tables. For E1 configs it also prints (and stores under ``predictability``)
the predictability of the targets of the evaluated windows (``harmonise_report.signal_rows``: std of the target, RMSE
of persistence) with harmonised and with matched signals, which states how much easier the smoothed targets are.

An existing matching file is replaced only when its settings and the windows of its data sets stay the same (they
define the artefacts of the E5 roots), or with ``--force``.

Causal matching (review control R6, design.md 18.5): a config with ``signal: matched_causal`` (``--causal`` states it
explicitly and refuses any other config), e.g.

    python scripts/match_roughness.py --config configs/experiment/e1m_causal.yaml --causal   # -> artifacts/e1m_causal/matching.json

uses the one-sided filter of ``cf_unc.data.harmonise.causal_events`` instead of the centred one and, for EVERY data set,
only the events of its VALIDATION fold (fold ``k + 1`` of its cf-stability split file with ``split.fold`` = k = 0, i.e.
fold 1; for the shifted data sets, which the models never train on, this is the fold that the causal E1 config does not
evaluate: ``split: all_but_val``). The test folds and the evaluated events never enter the choice of a window. The file
records ``filter: causal`` and ``part: val`` (the runners of ``matched_causal`` refuse a file without them, those of
``matched`` a file with them). The predictability table is not computed in this mode (the causal root's E1 analyses
report the persistence RMSE of its targets).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from cf_unc.data.harmonise import HarmoniseConfig, MatchConfig, causal_events, choose_window, harmonise_events, matching_kind
from cf_unc.experiments.e1 import Data, _display, data_config, data_names, load_config, match_config, matching_path
from cf_unc.train.run import ROOT, _path, write_json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harmonise_report import DATASETS, TARGETS, roughness, signal_rows  # noqa: E402  (scripts/ is not a package)

VERSION = 1
SETTING_KEYS = ("harmonise", "reference", "grid", "tolerance", "max_short_share")


# ------------------------------------------------------------------------------------------- matching
def window_stats(events: Sequence[Any], settings: HarmoniseConfig, windows: Sequence[int],
                 causal: bool = False) -> dict[int, dict[str, Any]]:  # fmt: skip
    """Per window w (samples): the events harmonised with w (``causal``: processed by the one-sided filter of
    ``causal_events``; the other settings unchanged; inputs not modified) ->
    ``roughness`` (``v``, ``dv``, ``s`` of ``harmonise_report.roughness`` over the kept events; NaN when none is kept),
    ``n_in``, ``n_out``, ``share_dropped``, ``dropped`` by reason (``harmonise_events``) and ``short_event_share``
    (share of the events shorter than w)."""
    out: dict[int, dict[str, Any]] = {}
    base = settings.to_dict()
    for w in sorted({int(x) for x in windows}):
        kept, report = (causal_events if causal else harmonise_events)(events, {**base, "window": w})
        n_in = int(report["n_in"])
        out[w] = {
            "roughness": roughness(kept) if len(kept) else {k: float("nan") for k in ("v", "dv", "s")},
            "n_in": n_in,
            "n_out": int(report["n_out"]),
            "share_dropped": float(report["share_dropped"]),
            "short_event_share": float(report["violations"]["short_event"]) / n_in if n_in else 0.0,
            "dropped": {str(k): int(v) for k, v in report["dropped"].items()},
        }
    return out


def match_windows(
    stats: Mapping[str, Mapping[int, Mapping[str, Any]]], settings: HarmoniseConfig, match: MatchConfig,
    sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """The matching of design.md 15.1 from ``window_stats`` per data set (each over ``match.grid`` and
    ``settings.window``): reference roughness = ``v`` of the reference at ``settings.window``, target = ``(1 +
    tolerance)`` x that, per data set the window (``choose_window``; the reference keeps ``settings.window``), its
    flags, roughness and drop shares at the window and at every grid window. Returns the content of the matching file
    (without ``created``, ``config``, ``seconds``)."""
    ref = match.reference
    if ref not in stats:
        raise ValueError(f"the reference data set {ref!r} is not among the matched data sets {sorted(stats)}")
    reference = float(stats[ref][settings.window]["roughness"]["v"])
    target = (1.0 + float(match.tolerance)) * reference
    datasets: dict[str, Any] = {}
    for name, per in stats.items():
        if name == ref:
            choice: dict[str, Any] = {"window": settings.window, "window_rule": settings.window, "cap": None, "flags": []}
            if not per[settings.window]["short_event_share"] < match.max_short_share:  # reported, the window stays
                choice["flags"] = ["short_share_above_limit"]
        else:
            choice = choose_window({w: per[w]["roughness"]["v"] for w in match.grid},
                                   {w: per[w]["short_event_share"] for w in match.grid}, target, match.grid,
                                   match.max_short_share)  # fmt: skip
        at = per[choice["window"]]
        datasets[name] = {
            **({} if sources is None else {"source": sources[name]}),
            "role": "reference" if name == ref else "matched",
            **choice,
            "roughness": float(at["roughness"]["v"]),
            "roughness_harmonised": float(per[settings.window]["roughness"]["v"]),
            "ratio": float(at["roughness"]["v"]) / reference,
            "n_events": int(at["n_in"]),
            "short_event_share": float(at["short_event_share"]),
            "share_dropped": float(at["share_dropped"]),
            "dropped": dict(at["dropped"]),
            "per_window": {str(w): {**v, "ratio": float(v["roughness"]["v"]) / reference} for w, v in sorted(per.items())},
        }
    return {
        "version": VERSION,
        "harmonise": settings.to_dict(),
        **match.settings(),
        "reference_window": int(settings.window),
        "reference_roughness": reference,
        "target_roughness": target,
        "windows": {name: int(entry["window"]) for name, entry in datasets.items()},
        "datasets": datasets,
    }


def matched_datasets(cfg: DictConfig) -> list[str]:
    """The data sets to match: the reference first, then ``data_names`` of the config."""
    return list(dict.fromkeys([match_config(cfg).reference, *data_names(cfg)]))


def check_replace(path: Path, content: Mapping[str, Any]) -> str | None:
    """Why an existing matching file must not be replaced by ``content`` (other settings, a data set lost or another
    window), None when it may be (absent, or the same settings and windows)."""
    if not path.exists():
        return None
    old = json.loads(path.read_text(encoding="utf-8"))
    differ = [k for k in SETTING_KEYS if json.dumps(old.get(k), sort_keys=True) != json.dumps(content.get(k), sort_keys=True)]
    if matching_kind(old) != matching_kind(content):
        differ.append(f"filter / part {matching_kind(old)} -> {matching_kind(content)}")
    old_w, new_w = old.get("windows") or {}, content["windows"]
    lost = sorted(set(old_w) - set(new_w))
    changed = {k: (old_w[k], new_w[k]) for k in old_w if k in new_w and int(old_w[k]) != int(new_w[k])}
    reasons = ([f"other settings {differ}"] if differ else []) + ([f"data sets that this config does not match {lost}"]
                                                                   if lost else [])  # fmt: skip
    reasons += [f"other windows (old, new) {changed}"] if changed else []
    return "; ".join(reasons) or None


# -------------------------------------------------------------------------------------- predictability
def predictability(cfg: DictConfig, windows: Mapping[str, int]) -> pd.DataFrame:
    """``harmonise_report.signal_rows`` of the evaluated data sets of an E1 config (keys of ``eval.datasets`` that
    harmonise_report reports, split / stride / cap of the config), harmonised (``harmonise.window``) and matched (the
    data set's window: exactly its harmonised key). One data set at a time; the reference is computed once."""
    keys = [k for k in cfg.eval.datasets if k in DATASETS] or [str(k) for k in cfg.eval.datasets]
    harmonised = OmegaConf.merge(cfg, {"signal": "harmonised"})
    base = int(HarmoniseConfig.from_config(cfg.get("harmonise")).window)
    rows = []
    for key in keys:
        name = str(cfg.eval.datasets[key].data)
        harm = signal_rows(Data(harmonised), harmonised, (key,), "harmonised")
        if int(windows[name]) == base:
            matched = [{**r, "signal": "matched"} for r in harm]
        else:
            at_window = OmegaConf.merge(cfg, {"signal": "harmonised", "harmonise": {"window": int(windows[name])}})
            matched = [{**r, "signal": "matched"} for r in signal_rows(Data(at_window), at_window, (key,), "matched")]
        rows += [{**r, "sg_window": base} for r in harm] + [{**r, "sg_window": int(windows[name])} for r in matched]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- output
def _sci(x: float) -> str:
    return "-" if x is None or not np.isfinite(float(x)) else f"{float(x):.3g}"


def markdown(content: Mapping[str, Any], signal: pd.DataFrame | None) -> str:
    """The matching per data set, the roughness ratio per grid window and (E1 configs) the predictability."""
    ref, base = content["reference"], int(content["reference_window"])
    lines = [
        f"Reference {ref} (window {base}): roughness of v {content['reference_roughness']:.4g} m/s, target "
        f"<= {content['target_roughness']:.4g} m/s (tolerance {content['tolerance']:g}); short events capped at "
        f"{100 * content['max_short_share']:g} %.", "",
        "| data set | role | events | window (rule, cap) | roughness of v, m/s: window 11 -> matched | x reference: 11 -> "
        "matched | flags | short_event share | dropped share: window 11 -> matched (min gap / abs c / gap rms) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    for name, d in content["datasets"].items():
        per = d["per_window"]
        at11, at = per[str(base)], per[str(d["window"])]
        rule = f"{d['window']} (rule {d['window_rule']}" + (f", cap {d['cap']}" if d["cap"] is not None else "") + ")"
        drops = at["dropped"]
        lines.append(
            f"| {name} | {d['role']} | {d['n_events']:,} | {rule} | {_sci(at11['roughness']['v'])} -> {_sci(d['roughness'])} | "
            f"{at11['ratio']:.2f} -> {d['ratio']:.2f} | {', '.join(d['flags']) or '-'} | {100 * d['short_event_share']:.2f} % | "
            f"{100 * at11['share_dropped']:.2f} % -> {100 * at['share_dropped']:.2f} % ({drops['min_gap']} / {drops['max_abs_c']} / "
            f"{drops['max_gap_rms']}) |"
        )  # fmt: skip
    grid = [int(w) for w in content["grid"]]
    lines += ["", "| roughness of v / reference | " + " | ".join(f"w {w}" for w in grid) + " |", "|---|" + "---|" * len(grid)]
    for name, d in content["datasets"].items():
        cells = [f"{d['per_window'][str(w)]['ratio']:.2f}" + ("*" if w == d["window"] else "") for w in grid]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("(* = chosen window; the reference keeps its window whatever its ratios)")
    if signal is not None and len(signal):
        lines += ["", "| data set | target | SG window harm. / matched | evaluated windows harm. / matched | std harm. -> matched | "
                  "persistence RMSE harm. -> matched | persistence / std harm. -> matched |", "|---|---|---|---|---|---|---|"]  # fmt: skip
        piv = signal.set_index(["dataset", "target", "signal"])
        for name in dict.fromkeys(signal["dataset"]):
            for target in TARGETS:
                h, m = piv.loc[(name, target, "harmonised")], piv.loc[(name, target, "matched")]
                lines.append(
                    f"| {name} | {target} | {int(h.sg_window)} / {int(m.sg_window)} | {int(h.n_windows):,} / {int(m.n_windows):,} | "
                    f"{h.std_a:.3f} -> {m.std_a:.3f} | {h.rmse_persistence:.4f} -> {m.rmse_persistence:.4f} | "
                    f"{h.rmse_persistence / h.std_a:.3f} -> {m.rmse_persistence / m.std_a:.3f} |"
                )  # fmt: skip
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="experiment config with signal: matched (e.g. configs/experiment/e1m.yaml)")
    parser.add_argument("--out", help="matching file (default: match.file of the config)")
    parser.add_argument("--force", action="store_true", help="replace a matching file with other settings or windows")
    parser.add_argument("--no-predictability", action="store_true", help="skip the predictability of the targets")
    parser.add_argument("--causal", action="store_true",
                        help="causal matching on the validation folds (the config must have signal: matched_causal)")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    cfg = load_config(args.config)
    signal_name = str(cfg.get("signal") or "")
    if signal_name not in ("matched", "matched_causal"):
        raise SystemExit(f"{args.config}: signal is {cfg.get('signal')!r}, the matching needs `signal: matched` or "
                         "`signal: matched_causal`")  # fmt: skip
    causal = signal_name == "matched_causal"
    if args.causal != causal:
        raise SystemExit(f"{args.config}: signal {signal_name!r}; --causal goes with `signal: matched_causal` (and only with it)")
    settings, match = HarmoniseConfig.from_config(cfg.get("harmonise")), match_config(cfg)
    out = _path(args.out) if args.out else matching_path(cfg)
    raw_cfg = OmegaConf.merge(cfg, {"signal": "raw", "train": {"target_from": "signal"}})  # the stored signals
    windows = sorted(set(match.grid) | {settings.window})
    stats, sources, parts = {}, {}, {}
    for name in matched_datasets(cfg):  # one data set in memory at a time
        t0 = time.perf_counter()
        data = Data(raw_cfg)
        if causal:  # the validation fold only (never the test fold or the evaluated events of a shifted set)
            if data.fold_file(name) is None:
                raise SystemExit(f"{name}: no split file; the causal matching uses the validation fold of every data set")
            events = data.split(name, "val")
            parts[name] = {"part": "val", "fold": (int(data.fold(name)) + 1) % int(json.loads(
                data.fold_file(name).read_text(encoding="utf-8"))["n_folds"]), "n_events_all": len(data.events(name))}  # fmt: skip
        else:
            events = data.events(name)  # all events, the common acceleration definition (unused by the filter)
        stats[name] = window_stats(events, settings, windows, causal=causal)
        sources[name] = _display(_path(str(data_config(cfg, name).path)))
        what = f"validation-fold events (fold {parts[name]['fold']})" if causal else "events"
        print(f"{name}: {len(events):,} {what}, roughness at {len(windows)} windows in {time.perf_counter() - t0:.1f} s", flush=True)
        del events, data
    content = match_windows(stats, settings, match, sources)
    if causal:
        content.update(filter="causal", part="val")
        for name, info in parts.items():
            content["datasets"][name].update(info)
    reason = check_replace(out, content)
    if reason is not None and not args.force:
        raise SystemExit(f"{out} exists with {reason}; the artefacts of the matched roots depend on it. Use --force to replace it.")
    signal = None
    if causal:
        print("causal matching: the predictability table is not computed (see the E1 analyses of the causal root)")
    elif not args.no_predictability and (cfg.get("eval") or {}).get("datasets"):
        signal = predictability(cfg, content["windows"])
    payload = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": _display(_path(args.config)),
        **content,
        **({} if signal is None else {"predictability": signal.to_dict("records")}),
        "seconds": round(time.perf_counter() - started, 1),
    }
    tmp = out.with_name(out.name + ".tmp")
    write_json(tmp, payload)
    tmp.replace(out)
    print()
    print(markdown(payload, signal))
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwindows: {', '.join(f'{k} {v}' for k, v in payload['windows'].items())}")
    print(f"written to {shown} in {payload['seconds']:.0f} s" + (" (replaced: --force)" if reason else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
