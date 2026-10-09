"""Analysis of experiment E2 (docs/design.md, section 11.4): coverage of the rollout intervals, conformal corrections,
ACI, one-step conformal coverage, collisions and the H3.4 verdicts.

    python scripts/analyze_e2.py                                       # artifacts/e2 and the E1 roots of the config
    python scripts/analyze_e2.py --root artifacts/e2_pilot
    python scripts/analyze_e2.py --root artifacts/e2 --e1-roots a=artifacts/e1 mean_1s=artifacts/e1_mean1s --n-boot 2000

Writes ``<root>/tables/``: coverage_by_horizon.csv, aci_gamma_sensitivity.csv, onestep_conformal.csv, collisions.csv,
h34_verdicts.csv, analysis_info.json; the figures of ``cf_unc.report.figures_e2`` into ``<root>/figures``; prints a
summary. Settings (alpha, gammas, n_min, bootstrap, H3.4 thresholds) come from ``<root>/config.yaml`` (written by the
runner) with the defaults of ``configs/experiment/e2.yaml`` for missing keys.

Statistical conventions: every seed is calibrated separately (static correction, ACI state, one-step conformal
quantile); per event the covered indicators and widths are averaged over the seeds, then events are resampled
(1000 percentile bootstrap replicates; one-step tables: windows grouped by event). Bootstrap seeds depend on the data
set and horizon only, so every model and variant is resampled with the same events. ACI runs over the calibration
events in their random ``order`` (events table), then continues over the evaluated events in their ``order``.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from cf_unc.eval.bootstrap import bootstrap_by_group
from cf_unc.maps import io as e1_io
from cf_unc.report.figures_e2 import make_e2_figures
from cf_unc.rollout import tables
from cf_unc.rollout.calibration import SCORES, coverage_table, h34_verdict, split_conformal_onestep

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS: dict[str, dict[str, Any]] = {  # mirror of configs/experiment/e2.yaml
    "calibration": {"alpha": 0.1, "gamma": 0.01, "gammas": [0.005, 0.01, 0.02, 0.05], "n_min": 20,
                    "onestep_score": "normalised", "corrected": "aci_online"},  # fmt: skip
    "bootstrap": {"n_boot": 1000, "ci": 0.95, "seed": 0},
    "h34": {"nominal": 0.90, "onestep_tol": 0.03, "raw_max": 0.80, "corrected_min": 0.87, "refute_min": 0.87},
}
CORRECTED = ("static", "aci_frozen", "aci_online")
COLUMNS: dict[str, list[str]] = {
    "coverage_by_horizon": [
        "model", "target", "dataset", "horizon_s", "variant", "picp", "picp_lo", "picp_hi", "width_mean", "width_median",
        "crps", "collided", "q_mean", "n_events", "n_seeds", "gamma",
    ],
    "aci_gamma_sensitivity": [
        "model", "target", "dataset", "horizon_s", "variant", "gamma", "picp", "picp_lo", "picp_hi", "width_mean",
        "width_median", "q_mean", "n_events", "n_seeds",
    ],
    "onestep_conformal": [
        "model", "target", "dataset", "score", "picp_raw", "picp_raw_lo", "picp_raw_hi", "picp_conformal",
        "picp_conformal_lo", "picp_conformal_hi", "width_raw", "width_conformal", "q", "n_windows", "n_events", "n_seeds",
    ],
    "h34_verdicts": [
        "model", "target", "dataset", "corrected", "onestep_score", "onestep_picp", "onestep_lo", "onestep_hi",
        "onestep_picp_dataset", "raw_picp_min", "raw_min_horizon", "corrected_picp_min", "corrected_min_horizon",
        "part_i", "part_ii", "part_iii", "refuted_raw", "verdict", "part_i_ci", "part_ii_ci", "part_iii_ci",
        "refuted_raw_ci", "verdict_ci", "n_seeds",
    ],
}  # fmt: skip


def _resolve(path: str | Path) -> Path:
    """A relative path is taken from the working directory if it exists there, else from the repository root."""
    p = Path(path)
    return p if p.is_absolute() or p.exists() else ROOT / p


def _seed(base: int, *keys: object) -> int:
    text = "\x1f".join(str(k) for k in keys).encode("utf-8")
    return int(np.random.SeedSequence([int(base) & 0xFFFFFFFF, zlib.crc32(text)]).generate_state(1)[0])


def load_settings(root: Path, overrides: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """DEFAULTS <- sections calibration / bootstrap / h34 of ``<root>/config.yaml`` <- ``overrides``; plus
    ``base_models`` (variant label -> checkpoint model, from the runner config)."""
    out = {k: dict(v) for k, v in DEFAULTS.items()}
    path = root / "config.yaml"
    stored = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}
    for key in out:
        out[key].update({k: v for k, v in (stored.get(key) or {}).items() if k in out[key]})
    for key, values in (overrides or {}).items():
        out[key].update({k: v for k, v in values.items() if v is not None})
    out["base_models"] = {str(k): str((v or {}).get("model", k)) for k, v in (stored.get("variants") or {}).items()}
    return out


def _mean0(a: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(a)
    return np.where(ok.any(axis=0), np.where(ok, a, 0.0).sum(axis=0) / np.maximum(ok.sum(axis=0), 1), np.nan)


def _frame(rows: list[dict[str, Any]], name: str) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS[name])


# ------------------------------------------------------------------------------------ rollout tables
def rollout_tables(root: Path, target: str, calibration: str, s: Mapping[str, Mapping[str, Any]],
                   log=print) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[dict[str, Any]]]:  # fmt: skip  # noqa: ANN001
    """Coverage rows (main gamma), gamma-sensitivity rows and collision rows of one target."""
    listing = tables.available(root, target)
    if listing.empty:
        return [], [], []
    c, b = s["calibration"], s["bootstrap"]
    gamma = float(c["gamma"])
    gammas = sorted({gamma, *(float(g) for g in c["gammas"])})
    events = {d: tables.load_events_table(root, target, d) for d in listing["dataset"].unique()}
    coverage, sensitivity, collisions = [], [], []
    for model in listing["model"].unique():
        sub = listing[listing["model"] == model]
        loaded = {d: {int(r.seed): pd.read_parquet(r.path) for r in sub[sub["dataset"] == d].itertuples()}
                  for d in sub["dataset"].unique()}  # fmt: skip
        cal = loaded.get(calibration)
        for dataset, eval_tables in loaded.items():
            with_cal = cal is not None and dataset != calibration
            table = coverage_table(
                eval_tables, events[dataset], cal if with_cal else None, events.get(calibration) if with_cal else None,
                alpha=float(c["alpha"]), gammas=gammas, n_min=int(c["n_min"]), n_boot=int(b["n_boot"]), ci=float(b["ci"]),
                boot_seed=_seed(int(b["seed"]), dataset),
            )  # fmt: skip
            table.insert(0, "dataset", dataset)
            table.insert(0, "target", target)
            table.insert(0, "model", model)
            aci = table["variant"].str.startswith("aci")
            main = ~aci | np.isclose(table["gamma"].fillna(-1.0), gamma)
            coverage.append(table[main][COLUMNS["coverage_by_horizon"]])
            sensitivity.append(table[aci][COLUMNS["aci_gamma_sensitivity"]])
            collisions.append(collision_row(model, target, dataset, eval_tables))
        log(f"  [{target}] {model}: {len(loaded)} data sets, seeds {sorted({s for t in loaded.values() for s in t})}")
    return coverage, sensitivity, collisions


def collision_row(model: str, target: str, dataset: str, eval_tables: Mapping[int, pd.DataFrame]) -> dict[str, Any]:
    """Share of rollouts with a collision within every horizon and share of events with at least one colliding
    rollout within the longest horizon (means over seeds)."""
    shares: dict[float, list[float]] = {}
    events_any, n_events = [], 0
    for frame in eval_tables.values():
        horizons = np.sort(frame["horizon_s"].unique().astype(float))
        for h in horizons:
            shares.setdefault(float(h), []).append(float(frame.loc[np.isclose(frame["horizon_s"], h), "collided"].mean()))
        last = frame[np.isclose(frame["horizon_s"], horizons[-1])]
        events_any.append(float((last["collided"] > 0).mean()))
        n_events = int(last["event"].nunique())
    row: dict[str, Any] = {"model": model, "target": target, "dataset": dataset}
    row.update({f"collided_{h:g}s": float(np.mean(v)) for h, v in sorted(shares.items())})
    row.update(events_with_collision=float(np.mean(events_any)), n_events=n_events, n_seeds=len(eval_tables))
    return row


# ------------------------------------------------------------------------------------ one-step conformal
def onestep_conformal(e1_root: Path, target: str, calibration: str, s: Mapping[str, Mapping[str, Any]],
                      log=print) -> list[dict[str, Any]]:  # fmt: skip  # noqa: ANN001
    """Split conformal coverage of the one-step predictions of E1 (calibration: ``calibration`` windows)."""
    listing = e1_io.available(e1_root)
    if listing.empty or calibration not in set(listing["dataset"]):
        log(f"  [{target}] no E1 prediction tables with a {calibration} set under {e1_root}: one-step conformal skipped")
        return []
    c, b = s["calibration"], s["bootstrap"]
    alpha = float(c["alpha"])
    windows: dict[str, pd.DataFrame] = {}

    def window(d: str) -> pd.DataFrame:
        if d not in windows:
            windows[d] = pd.read_parquet(e1_io.windows_path(e1_root, d), columns=["event", "y"])
        return windows[d]

    def preds(model: str, seed: int, d: str) -> pd.DataFrame:
        frame = e1_io.load_preds(e1_root, model, seed, d, columns=["mean", "var", "covered", "width"])
        valid = np.isfinite(frame["mean"].to_numpy(np.float64)) & np.isfinite(frame["var"].to_numpy(np.float64))
        return frame.assign(valid=valid)

    rows: list[dict[str, Any]] = []
    for model in listing["model"].unique():
        sub = listing[listing["model"] == model]
        cal_seeds = set(sub.loc[sub["dataset"] == calibration, "seed"].astype(int))
        cal_preds = {k: preds(model, k, calibration) for k in sorted(cal_seeds)}
        y_cal = window(calibration)["y"].to_numpy(np.float64)
        for dataset in [d for d in sub["dataset"].unique() if d != calibration]:
            seeds = sorted(cal_seeds & set(sub.loc[sub["dataset"] == dataset, "seed"].astype(int)))
            if not seeds:
                continue
            w = window(dataset)
            y, events = w["y"].to_numpy(np.float64), w["event"].to_numpy()
            ev = {k: preds(model, k, dataset) for k in seeds}
            raw_cov = np.stack([np.where(ev[k]["valid"], ev[k]["covered"].to_numpy(np.float64), np.nan) for k in seeds])
            raw_width = np.stack([np.where(ev[k]["valid"], ev[k]["width"].to_numpy(np.float64), np.nan) for k in seeds])
            raw_mean = _mean0(raw_cov)
            ok = ~np.isnan(raw_mean)
            boot_seed = _seed(int(b["seed"]), "onestep", dataset)
            raw_boot = bootstrap_by_group(raw_mean[ok], events[ok], n_boot=int(b["n_boot"]), seed=boot_seed, ci=float(b["ci"]))
            for score in SCORES:
                cov, width, qs = [], [], []
                for k in seeds:
                    cp, ep = cal_preds[k], ev[k]
                    out = split_conformal_onestep(
                        np.where(cp["valid"], cp["mean"], np.nan), cp["var"], y_cal,
                        np.where(ep["valid"], ep["mean"], np.nan), ep["var"], y, alpha=alpha, score=score,
                    )  # fmt: skip
                    cov.append(out["covered"])
                    width.append(out["width"])
                    qs.append(out["q"])
                per_window = _mean0(np.stack(cov))
                good = ~np.isnan(per_window)
                boot = bootstrap_by_group(per_window[good], events[good], n_boot=int(b["n_boot"]), seed=boot_seed,
                                          ci=float(b["ci"]))  # fmt: skip
                rows.append({
                    "model": model, "target": target, "dataset": dataset, "score": score,
                    "picp_raw": raw_boot["estimate"], "picp_raw_lo": raw_boot["lo"], "picp_raw_hi": raw_boot["hi"],
                    "picp_conformal": boot["estimate"], "picp_conformal_lo": boot["lo"], "picp_conformal_hi": boot["hi"],
                    "width_raw": float(np.nanmean(_mean0(raw_width)[ok])), "width_conformal": float(np.mean(_mean0(np.stack(width))[good])),
                    "q": float(np.mean(qs)), "n_windows": int(good.sum()), "n_events": int(np.unique(events[good]).size),
                    "n_seeds": len(seeds),
                })  # fmt: skip
        log(f"  [{target}] one-step conformal {model}: seeds {sorted(cal_seeds)}")
    return rows


# --------------------------------------------------------------------------------------------- H3.4
def h34_rows(coverage: pd.DataFrame, onestep: pd.DataFrame, s: Mapping[str, Mapping[str, Any]], in_domain: str,
             calibration: str) -> list[dict[str, Any]]:  # fmt: skip
    """One verdict row per (model, target, data set, corrected variant); part (i) = one-step conformal coverage of
    the in-domain test set (score of the settings), parts (ii) and (iii) on the row's data set."""
    score = str(s["calibration"]["onestep_score"])
    th = {k: float(v) for k, v in s["h34"].items()}
    bases = s.get("base_models", {})
    rows = []
    for (model, target, dataset), grp in coverage.groupby(["model", "target", "dataset"], sort=False):
        if dataset == calibration:
            continue
        # a variant label shares the one-step predictions of its base model; the stored runner config lists only
        # the variants of the LAST run (a run with --models filters them), hence the fallback on the label
        base = bases.get(model, str(model).split("__")[0])
        one = onestep[(onestep["model"] == base) & (onestep["target"] == target) & (onestep["score"] == score)] \
            if not onestep.empty else onestep  # fmt: skip
        dom = one[one["dataset"] == in_domain] if not one.empty else one
        here = one[one["dataset"] == dataset] if not one.empty else one
        o_est, o_lo, o_hi = ((float(dom.iloc[0][c]) for c in ("picp_conformal", "picp_conformal_lo", "picp_conformal_hi"))
                             if len(dom) else (np.nan, np.nan, np.nan))  # fmt: skip
        raw = grp[grp["variant"] == "raw"].sort_values("horizon_s")
        for corrected in CORRECTED:
            corr = grp[grp["variant"] == corrected].sort_values("horizon_s")
            if corr.empty or raw.empty:
                continue
            v = h34_verdict(
                o_est, raw["picp"], corr["picp"], onestep_ci=(o_lo, o_hi), raw_ci=(raw["picp_lo"], raw["picp_hi"]),
                corrected_ci=(corr["picp_lo"], corr["picp_hi"]), **th,
            )  # fmt: skip
            rows.append({
                "model": model, "target": target, "dataset": dataset, "corrected": corrected, "onestep_score": score,
                "onestep_picp": o_est, "onestep_lo": o_lo, "onestep_hi": o_hi,
                "onestep_picp_dataset": float(here.iloc[0]["picp_conformal"]) if len(here) else np.nan,
                "raw_picp_min": float(raw["picp"].min()), "raw_min_horizon": float(raw.loc[raw["picp"].idxmin(), "horizon_s"]),
                "corrected_picp_min": float(corr["picp"].min()),
                "corrected_min_horizon": float(corr.loc[corr["picp"].idxmin(), "horizon_s"]),
                **{k: v[k] for k in ("part_i", "part_ii", "part_iii", "refuted_raw", "verdict", "part_i_ci", "part_ii_ci",
                                     "part_iii_ci", "refuted_raw_ci", "verdict_ci")},
                "n_seeds": int(corr["n_seeds"].min()),
            })  # fmt: skip
    return rows


# ------------------------------------------------------------------------------------------- entry
def analyze_e2(
    root: str | Path = "artifacts/e2",
    e1_roots: Mapping[str, str | Path] | None = None,
    calibration: str = "highd_val",
    in_domain: str = "highd",
    *,
    settings: Mapping[str, Mapping[str, Any]] | None = None,
    figures: bool = True,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Every E2 table under ``<root>/tables`` (module docstring) and the figures; returns the tables by name.

    e1_roots: target -> E1 root of the one-step conformal analysis (default: ``e1_roots`` of ``<root>/config.yaml``,
    else ``{"a": "artifacts/e1", "mean_1s": "artifacts/e1_mean1s"}``; missing roots are skipped), so that e.g. the
    harmonised experiment (configs/experiment/e2h.yaml) uses the E1 tables of its own signal variant. settings:
    overrides of the calibration / bootstrap / h34 sections.
    """
    started = time.perf_counter()
    root = _resolve(root)
    log = print if verbose else (lambda *a, **k: None)
    if e1_roots is None:
        stored_cfg = root / "config.yaml"
        stored = (yaml.safe_load(stored_cfg.read_text(encoding="utf-8")) or {}) if stored_cfg.exists() else {}
        e1_roots = {str(k): str(v) for k, v in (stored.get("e1_roots") or {}).items()} or None
    e1_roots = {"a": "artifacts/e1", "mean_1s": "artifacts/e1_mean1s"} if e1_roots is None else dict(e1_roots)
    s = load_settings(root, settings)
    targets = sorted(p.name for p in root.iterdir() if (p / "rollouts").is_dir()) if root.is_dir() else []
    if not targets:
        raise FileNotFoundError(f"no rollout tables under {root}")
    log(f"E2 analysis of {root}: targets {targets}")
    cov_parts, sens_parts, collision_rows, onestep_rows = [], [], [], []
    for target in targets:
        cov, sens, col = rollout_tables(root, target, calibration, s, log)
        cov_parts += cov
        sens_parts += sens
        collision_rows += col
        e1_root = e1_roots.get(target)
        if e1_root is not None and _resolve(e1_root).is_dir():
            onestep_rows += onestep_conformal(_resolve(e1_root), target, calibration, s, log)
        else:
            log(f"  [{target}] E1 root {e1_root} not found: one-step conformal skipped")
    out = {
        "coverage_by_horizon": pd.concat(cov_parts, ignore_index=True) if cov_parts else _frame([], "coverage_by_horizon"),
        "aci_gamma_sensitivity": pd.concat(sens_parts, ignore_index=True) if sens_parts else _frame([], "aci_gamma_sensitivity"),
        "onestep_conformal": _frame(onestep_rows, "onestep_conformal"),
        "collisions": pd.DataFrame(collision_rows),
    }  # fmt: skip
    out["h34_verdicts"] = _frame(h34_rows(out["coverage_by_horizon"], out["onestep_conformal"], s, in_domain, calibration),
                                 "h34_verdicts")  # fmt: skip
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in out.items():
        frame.to_csv(table_dir / f"{name}.csv", index=False)
    paths = make_e2_figures(root, in_domain=in_domain, calibration=calibration) if figures and cov_parts else []
    info = {"root": str(root), "targets": targets, "e1_roots": {k: str(v) for k, v in e1_roots.items()},
            "calibration": calibration, "in_domain": in_domain, "settings": s, "figures": [p.name for p in paths],
            "seconds": time.perf_counter() - started}  # fmt: skip
    (table_dir / "analysis_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    log(f"tables written to {table_dir} ({len(paths)} figures) in {info['seconds']:.0f} s")
    return out


def _fmt(x: Any, digits: int = 2) -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{digits}f}"


def format_summary(out: Mapping[str, pd.DataFrame], in_domain: str = "highd", corrected: str = "aci_online") -> str:
    """Plain-text summary: in-domain raw and corrected coverage by horizon, collisions, one-step conformal, H3.4."""
    cov, lines = out["coverage_by_horizon"], []
    for target in cov["target"].unique():
        sel = cov[(cov["target"] == target) & (cov["dataset"] == in_domain)]
        horizons = np.sort(sel["horizon_s"].unique())
        head = " ".join(f"{h:>5g}s" for h in horizons)
        lines.append(f"\nTarget {target}, {in_domain}: PICP of the 90 % gap interval, raw | {corrected}; collided at "
                     f"{horizons[-1]:g} s" if len(horizons) else f"\nTarget {target}: no in-domain rows")  # fmt: skip
        lines.append(f"{'model':22s} {head} | {head} | collided")
        for model in sel["model"].unique():
            m = sel[sel["model"] == model].set_index(["variant", "horizon_s"])["picp"]
            raw = " ".join(f"{_fmt(m.get(('raw', h))):>6s}" for h in horizons)
            corr = " ".join(f"{_fmt(m.get((corrected, h))):>6s}" for h in horizons)
            col = sel[(sel["model"] == model) & (sel["variant"] == "raw") & (sel["horizon_s"] == horizons[-1])]["collided"]
            lines.append(f"{model:22s} {raw} | {corr} | {_fmt(float(col.iloc[0]) if len(col) else None, 3)}")
    one = out["onestep_conformal"]
    if not one.empty:
        sel = one[one["dataset"] == in_domain]
        lines.append(f"\nOne-step conformal coverage on {in_domain} (raw -> absolute / normalised score)")
        for (target, model), grp in sel.groupby(["target", "model"], sort=False):
            g = grp.set_index("score")
            lines.append(f"{target:8s} {model:22s} {_fmt(g['picp_raw'].iloc[0], 3)} -> "
                         f"{_fmt(g.loc['absolute', 'picp_conformal'], 3)} / {_fmt(g.loc['normalised', 'picp_conformal'], 3)}")  # fmt: skip
    ver = out["h34_verdicts"]
    if not ver.empty:
        sel = ver[ver["corrected"] == corrected]
        lines.append(f"\nH3.4 verdicts (corrected = {corrected}): point / interval")
        pivot = sel.assign(v=sel["verdict"] + " / " + sel["verdict_ci"].astype(str)).pivot_table(
            index=["target", "model"], columns="dataset", values="v", aggfunc="first", sort=False)
        lines.append(pivot.to_string())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="artifacts/e2", help="E2 root (rollout tables, config.yaml)")
    parser.add_argument("--e1-roots", nargs="*", default=None, metavar="TARGET=PATH",
                        help="E1 roots of the one-step conformal analysis (default: e1_roots of <root>/config.yaml, "
                             "else a=artifacts/e1 mean_1s=artifacts/e1_mean1s)")
    parser.add_argument("--calibration", default="highd_val", help="calibration data set")
    parser.add_argument("--in-domain", default="highd", help="in-domain test set")
    parser.add_argument("--n-boot", type=int, default=None, help="bootstrap replicates (default: config)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    e1_roots = None if args.e1_roots is None else dict(item.split("=", 1) for item in args.e1_roots)
    settings = {"bootstrap": {"n_boot": args.n_boot}} if args.n_boot else None
    out = analyze_e2(args.root, e1_roots, args.calibration, args.in_domain, settings=settings,
                     figures=not args.no_figures, verbose=not args.quiet)  # fmt: skip
    corrected = str(load_settings(_resolve(args.root))["calibration"]["corrected"])
    print(format_summary(out, args.in_domain, corrected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
