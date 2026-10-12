#!/usr/bin/env python3
"""Conditional coverage of the corrected rollout intervals (extension S5, extension/docs/PLAN.md, section S).

Offline on the stored E2 rollout tables (target a: artifacts/e2/a/rollouts/<model>/seed<k>/<set>.parquet, 500 events
per set, the 90 % gap interval [s_q05, s_q95] with the median s_q50 at every horizon). Every event is labelled with
the E1 regime of its state at the issue instant (the last history sample t0: s, dv, v and the previous acceleration,
cf_unc.maps.regimes.label_regimes with the free speed of the highD IDM calibration of artifacts/e1/idm.json). The
variants: raw; static (the multiplicative score of E2, conformal quantile on highd_val of the same seed and horizon);
cqr (the additive conformity score max(lo - y, y - hi) of conformalised quantile regression, split-conformal on
highd_val); aci_online (the ACI of E2, gamma 0.01, n_min 20, continued from the calibration run over the evaluated
events in their stored order). Per model, set, horizon, variant and regime: the coverage (seed mean of the event
mean) and the number of events; per variant also the interval (Winkler) score of the finite intervals and the share
of infinite intervals. Reading rule (PLAN.md S5): a variant is conditionally valid on a set when every regime with at
least 30 events has a coverage within 0.05 of 0.90 at the 10 s horizon.

    extension/artifacts/ext_coverage/tables/coverage_by_regime.csv
    extension/artifacts/ext_coverage/tables/coverage_variants.csv
    extension/artifacts/ext_coverage/tables/conditional_validity.csv
    extension/artifacts/ext_coverage/tables/regimes.csv              the regime of every evaluated event
    extension/artifacts/ext_coverage/tables/README.md

    PYTHONUTF8=1 python extension/scripts/conditional_coverage.py [--e2 artifacts/e2/a]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cf_unc.data.pipeline import load_events
from cf_unc.maps.regimes import REGIMES, label_regimes
from cf_unc.rollout.calibration import (
    HALF_WIDTH_FLOOR,
    aci_run,
    conformal_quantile,
    corrected_interval,
    interval_score,
    static_correction,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "extension/artifacts/ext_coverage/tables"
CALIBRATION, IN_DOMAIN = "highd_val", "highd"
SETS = ("highd", "ngsim", "ngsim_us101", "waymo", "openacc")
DATA_DIR = {"highd": "follownet_highd", "highd_val": "follownet_highd", "ngsim": "ngsim_i80", "ngsim_us101": "ngsim_us101",
            "waymo": "waymo", "openacc": "openacc"}  # fmt: skip
ALPHA, GAMMA, N_MIN = 0.1, 0.01, 20
NOMINAL, TOL, MIN_EVENTS, VALID_HORIZON = 0.90, 0.05, 30, 10.0
VARIANTS = ("raw", "static", "cqr", "aci_online")


def winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Interval score of Gneiting and Raftery (2007); NaN for an infinite or empty interval."""
    w = (hi - lo) + (2.0 / alpha) * np.maximum(lo - y, 0.0) + (2.0 / alpha) * np.maximum(y - hi, 0.0)
    return np.where(np.isfinite(lo) & np.isfinite(hi) & (hi >= lo), w, np.nan)


def regimes_of(e2: Path, events_cache: dict[str, dict[str, object]], v0: float) -> pd.DataFrame:
    rows = []
    for name in (CALIBRATION, *SETS):
        table = pd.read_parquet(e2 / "events" / f"{name}.parquet")
        key = DATA_DIR[name]
        if key not in events_cache:
            events = load_events(ROOT / "data/events" / key)
            events_cache[key] = {e.event_id: e for e in events}
        by_id = events_cache[key]
        s, dv, v, a_prev = [], [], [], []
        for r in table.itertuples():
            e = by_id[r.event_id]
            t = int(r.t0)
            s.append(e.s[t]), dv.append(e.dv[t]), v.append(e.v[t]), a_prev.append(e.a[t - 1] if t >= 1 else np.nan)
        labels = label_regimes(np.array(s), np.array(dv), np.array(v), np.array(a_prev), v0)
        rows.append(pd.DataFrame({"dataset": name, "event": table["event"].to_numpy(np.int64), "event_id": table["event_id"],
                                  "order": table["order"].to_numpy(np.int64), "s": s, "dv": dv, "v": v, "a_prev": a_prev,
                                  "regime": [REGIMES[i] for i in labels]}))  # fmt: skip
    return pd.concat(rows, ignore_index=True)


def horizon_rows(table: pd.DataFrame, h: float, codes: np.ndarray) -> pd.DataFrame:
    rows = table[np.isclose(table["horizon_s"].to_numpy(np.float64), h)].set_index("event")
    return rows.loc[codes]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--e2", default="artifacts/e2/a")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    e2, out = ROOT / a.e2, Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v0 = float(json.loads((ROOT / "artifacts/e1/idm.json").read_text(encoding="utf-8"))["params"]["v0"])
    regimes = regimes_of(e2, {}, v0)
    regimes.to_csv(out / "regimes.csv", index=False)
    reg = {name: part.set_index("event") for name, part in regimes.groupby("dataset")}
    print("regime shares:")
    print(regimes.groupby(["dataset", "regime"]).size().unstack(fill_value=0).to_string())

    by_regime, by_variant = [], []
    for model_dir in sorted((e2 / "rollouts").iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        seeds = sorted(int(p.name[4:]) for p in model_dir.glob("seed*") if (p / f"{CALIBRATION}.parquet").exists())
        for name in SETS:
            tables = {k: pd.read_parquet(model_dir / f"seed{k}" / f"{name}.parquet") for k in seeds
                      if (model_dir / f"seed{k}" / f"{name}.parquet").exists()}  # fmt: skip
            if not tables:
                continue
            cal = {k: pd.read_parquet(model_dir / f"seed{k}" / f"{CALIBRATION}.parquet") for k in tables}
            ev, cal_ev = reg[name], reg[CALIBRATION]
            codes, cal_codes = ev.index.to_numpy(np.int64), cal_ev.index.to_numpy(np.int64)
            perm = np.argsort(ev["order"].to_numpy(np.int64), kind="stable")
            cal_perm = np.argsort(cal_ev["order"].to_numpy(np.int64), kind="stable")
            labels = ev["regime"].to_numpy()
            horizons = np.unique(next(iter(tables.values()))["horizon_s"].to_numpy(np.float64))
            for h in horizons:
                cov = {v: [] for v in VARIANTS}
                win = {v: [] for v in VARIANTS}
                inf = {v: [] for v in VARIANTS}
                for k, table in tables.items():
                    r = horizon_rows(table, h, codes)
                    c = horizon_rows(cal[k], h, cal_codes)
                    y, lo, mid, hi = (r[col].to_numpy(np.float64) for col in ("truth_s", "s_q05", "s_q50", "s_q95"))
                    yc, loc, midc, hic = (c[col].to_numpy(np.float64) for col in ("truth_s", "s_q05", "s_q50", "s_q95"))
                    score = interval_score(y, lo, mid, hi, HALF_WIDTH_FLOOR)
                    cal_score = interval_score(yc, loc, midc, hic, HALF_WIDTH_FLOOR)
                    # raw
                    cov["raw"].append((score <= 0.0).astype(float))
                    win["raw"].append(winkler(y, lo, hi))
                    inf["raw"].append(np.zeros(len(y)))
                    # static, multiplicative score of E2
                    q = static_correction(cal_score, ALPHA)
                    l2, u2 = corrected_interval(lo, mid, hi, q, HALF_WIDTH_FLOOR)
                    cov["static"].append((score <= q).astype(float))
                    win["static"].append(winkler(y, l2, u2))
                    inf["static"].append(np.full(len(y), float(not np.isfinite(q))))
                    # CQR, additive score
                    s_add = np.maximum(lo - y, y - hi)
                    q_add = conformal_quantile(np.maximum(loc - yc, yc - hic), ALPHA)
                    cov["cqr"].append((s_add <= q_add).astype(float))
                    win["cqr"].append(winkler(y, lo - q_add, hi + q_add))
                    inf["cqr"].append(np.full(len(y), float(not np.isfinite(q_add))))
                    # ACI online, continued from the calibration run
                    run = aci_run(cal_score[cal_perm], ALPHA, GAMMA, N_MIN)
                    res = aci_run(score[perm], ALPHA, GAMMA, N_MIN, state=run["state"], frozen=False)
                    covered = np.empty(len(y))
                    covered[perm] = res["covered"]
                    q_e = np.empty(len(y))
                    q_e[perm] = res["q"]
                    l3, u3 = corrected_interval(lo, mid, hi, q_e, HALF_WIDTH_FLOOR)
                    cov["aci_online"].append(covered)
                    win["aci_online"].append(winkler(y, l3, u3))
                    inf["aci_online"].append((~np.isfinite(q_e)).astype(float))
                for v in VARIANTS:
                    c_arr, w_arr, i_arr = np.array(cov[v]), np.array(win[v]), np.array(inf[v])  # (seeds, events)
                    by_variant.append({
                        "model": model, "dataset": name, "horizon_s": float(h), "variant": v,
                        "picp": float(np.nanmean(c_arr)), "picp_seed_sd": float(np.nanmean(c_arr, axis=1).std(ddof=0)),
                        "winkler": float(np.nanmean(w_arr)) if np.isfinite(w_arr).any() else float("nan"),
                        "share_infinite": float(i_arr.mean()), "n_events": int(len(codes)), "n_seeds": len(tables),
                    })  # fmt: skip
                    for regime in REGIMES:
                        sel = labels == regime
                        if not sel.any():
                            continue
                        by_regime.append({
                            "model": model, "dataset": name, "horizon_s": float(h), "variant": v, "regime": regime,
                            "picp": float(np.nanmean(c_arr[:, sel])), "n_events": int(sel.sum()), "n_seeds": len(tables),
                            "winkler": float(np.nanmean(w_arr[:, sel])) if np.isfinite(w_arr[:, sel]).any() else float("nan"),
                        })  # fmt: skip
        print(f"{model}: {len(seeds)} seeds")
    reg_table, var_table = pd.DataFrame(by_regime), pd.DataFrame(by_variant)
    reg_table.to_csv(out / "coverage_by_regime.csv", index=False)
    var_table.to_csv(out / "coverage_variants.csv", index=False)

    validity = []
    at = reg_table[np.isclose(reg_table["horizon_s"], VALID_HORIZON) & (reg_table["n_events"] >= MIN_EVENTS)]
    for (model, name, v), part in at.groupby(["model", "dataset", "variant"]):
        failing = part[(part["picp"] - NOMINAL).abs() > TOL]
        validity.append({"model": model, "dataset": name, "variant": v, "horizon_s": VALID_HORIZON,
                         "n_regimes": len(part), "n_failing": len(failing), "valid": len(failing) == 0,
                         "failing_regimes": "; ".join(f"{r.regime} {r.picp:.2f} (n={r.n_events})" for r in failing.itertuples()),
                         "picp_min": float(part["picp"].min()), "picp_max": float(part["picp"].max())})  # fmt: skip
    val_table = pd.DataFrame(validity)
    val_table.to_csv(out / "conditional_validity.csv", index=False)

    lines = ["# Conditional coverage of the rollout intervals (extension S5)", "",
             f"Coverage of the 90 % gap intervals by E1 regime at the issue instant (free speed v0 = {v0:.2f} m/s), for the raw, "
             "static (multiplicative score of E2), CQR (additive score) and ACI-online (gamma 0.01) intervals; seed means over the "
             f"stored E2 rollouts (target a). Reading rule (PLAN.md S5): conditionally valid when every regime with at least "
             f"{MIN_EVENTS} events has a coverage within {TOL:.2f} of {NOMINAL:.2f} at {VALID_HORIZON:g} s.", "",
             "## Validity at 10 s (regimes with at least 30 events)", "",
             "| model | set | variant | regimes | failing | min PICP | max PICP | failing regimes |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in val_table.itertuples():
        lines.append(f"| {r.model} | {r.dataset} | {r.variant} | {r.n_regimes} | {r.n_failing} | {r.picp_min:.2f} | {r.picp_max:.2f} | {r.failing_regimes or 'none'} |")
    lines += ["", "## Variants at 10 s: coverage, Winkler score of the finite intervals (m), share of infinite intervals", "",
              "| model | set | variant | PICP | Winkler | infinite |", "|---|---|---|---|---|---|"]  # fmt: skip
    for r in var_table[np.isclose(var_table["horizon_s"], VALID_HORIZON)].itertuples():
        lines.append(f"| {r.model} | {r.dataset} | {r.variant} | {r.picp:.3f} | {r.winkler:.1f} | {r.share_infinite:.3f} |")
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(reg_table)} regime rows, {len(var_table)} variant rows, {len(val_table)} validity rows written to {out}")


if __name__ == "__main__":
    main()
