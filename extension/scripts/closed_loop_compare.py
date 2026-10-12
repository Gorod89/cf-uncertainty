#!/usr/bin/env python3
"""The closed-loop fine-tuned networks next to their one-step twins (extension S7).

Per training regime (highD: artifacts/e1 versus ext_cl_highd; pooled: artifacts/e6_pool versus ext_cl_pool; OpenACC
held out: artifacts/e6_lodo_openacc versus ext_cl_lodo): the one-step NLL, CRPS, PICP90 and RMSE of the twin and of
the fine-tuned law on every evaluation set (seed means). For the highD regime also the H3.4 verdicts and the raw and
ACI-corrected coverage at 10 s of E2 (artifacts/e2 versus ext_e2_cl). For the platoon cells raw, pool and
lodo_openacc: the H3.2 reduction (extended set, response window, ACC and human-driven profiles) and the mean share of
collided rollouts (artifacts/e3 versus ext_e3_cl). Reading rule (PLAN.md S7): an improvement is claimed only when its
sign holds in every regime.

    extension/artifacts/ext_cl_summary/tables/{one_step,e2,platoon}.csv and README.md

    PYTHONUTF8=1 python extension/scripts/closed_loop_compare.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "extension/artifacts"
OUT = EXT / "ext_cl_summary/tables"
PAIRS = {"hetero_mdn": "hetero_mdn_cl", "lstm_const_sigma": "lstm_const_sigma_cl"}
REGIMES = {"highd": (ROOT / "artifacts/e1", EXT / "ext_cl_highd"), "pool": (ROOT / "artifacts/e6_pool", EXT / "ext_cl_pool"),
           "lodo_openacc": (ROOT / "artifacts/e6_lodo_openacc", EXT / "ext_cl_lodo")}  # fmt: skip
CELLS = {"raw": "raw", "pool": "pool", "lodo_openacc": "lodo_openacc"}


def one_step(path: Path) -> pd.DataFrame | None:
    p = path / "tables/one_step_metrics.csv"
    if not p.exists():
        return None
    om = pd.read_csv(p)
    om = om[(om.subset == "all") & (om.seed.astype(str) == "mean")]
    return om.set_index(["model", "dataset"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime, (twin_root, cl_root) in REGIMES.items():
        t, c = one_step(twin_root), one_step(cl_root)
        if t is None or c is None:
            print(f"{regime}: one-step tables missing ({'twin' if t is None else 'fine-tuned'})")
            continue
        for twin, cl in PAIRS.items():
            for (model, dataset) in c.index:
                if model != cl or (twin, dataset) not in t.index:
                    continue
                a, b = t.loc[(twin, dataset)], c.loc[(cl, dataset)]
                rows.append({"regime": regime, "twin": twin, "dataset": dataset,
                             **{f"{m}_twin": float(a[m]) for m in ("nll", "crps", "picp90", "rmse")},
                             **{f"{m}_cl": float(b[m]) for m in ("nll", "crps", "picp90", "rmse")},
                             "nll_delta": float(b["nll"] - a["nll"]), "rmse_delta": float(b["rmse"] - a["rmse"])})  # fmt: skip
    one = pd.DataFrame(rows)
    one.to_csv(OUT / "one_step.csv", index=False)

    e2_rows = []
    for name, root in (("twin", ROOT / "artifacts/e2"), ("cl", EXT / "ext_e2_cl")):
        p = root / "tables/h34_verdicts.csv"
        q = root / "tables/coverage_by_horizon.csv"
        if not p.exists() or not q.exists():
            print(f"E2 tables of {name} missing")
            continue
        h34 = pd.read_csv(p)
        h34 = h34[(h34.target == "a") & (h34.corrected == "aci_online")]
        cov = pd.read_csv(q)
        cov = cov[(cov.target == "a") & np.isclose(cov.horizon_s, 10.0)]
        for r in h34.itertuples():
            if r.model not in PAIRS and r.model not in PAIRS.values():
                continue
            raw = cov[(cov.model == r.model) & (cov.dataset == r.dataset) & (cov.variant == "raw")]
            aci = cov[(cov.model == r.model) & (cov.dataset == r.dataset) & (cov.variant == "aci_online")]
            e2_rows.append({"which": name, "model": r.model, "dataset": r.dataset, "verdict_ci": r.verdict_ci,
                            "picp_raw_10s": float(raw.picp.iloc[0]) if not raw.empty else np.nan,
                            "picp_aci_10s": float(aci.picp.iloc[0]) if not aci.empty else np.nan,
                            "collided_10s": float(raw.collided.iloc[0]) if not raw.empty else np.nan})  # fmt: skip
    e2 = pd.DataFrame(e2_rows)
    e2.to_csv(OUT / "e2.csv", index=False)

    pl_rows = []
    for name, root in (("twin", ROOT / "artifacts/e3"), ("cl", EXT / "ext_e3_cl")):
        p, q = root / "tables/h32.csv", root / "tables/collisions.csv"
        if not p.exists() or not q.exists():
            print(f"platoon tables of {name} missing")
            continue
        h32 = pd.read_csv(p)
        h32 = h32[(h32.reference == "idm_white_noise") & (h32.quantity == "amplitude") & (h32["set"] == "extended") & (h32.window == "response")]
        col = pd.read_csv(q)
        col["index"] = col["index"].astype(str)
        veh = col[(col["index"] != "any") & (col["set"] == "extended")]
        collided = veh.groupby(["variant", "model"])["collided"].mean()
        for r in h32.itertuples():
            if r.variant not in CELLS or (r.model not in PAIRS and r.model not in PAIRS.values()):
                continue
            pl_rows.append({"which": name, "variant": r.variant, "model": r.model, "mode": r.mode, "reduction": r.reduction,
                            "lo_cluster": r.lo_cluster, "hi_cluster": r.hi_cluster, "verdict_cluster": r.verdict_cluster,
                            "collided_mean": float(collided.get((r.variant, r.model), np.nan))})  # fmt: skip
    pl = pd.DataFrame(pl_rows)
    pl.to_csv(OUT / "platoon.csv", index=False)

    lines = ["# Closed-loop fine-tuned networks versus their one-step twins (extension S7)", ""]
    if not one.empty:
        lines += ["## One-step metrics (seed means): twin -> fine-tuned", "", "| regime | twin | set | NLL | CRPS | PICP90 | RMSE |", "|---|---|---|---|---|---|---|"]
        for r in one.itertuples():
            lines.append(f"| {r.regime} | {r.twin} | {r.dataset} | {r.nll_twin:.2f} -> {r.nll_cl:.2f} | {r.crps_twin:.3f} -> {r.crps_cl:.3f} | "
                         f"{r.picp90_twin:.3f} -> {r.picp90_cl:.3f} | {r.rmse_twin:.3f} -> {r.rmse_cl:.3f} |")  # fmt: skip
        lines.append("")
    if not e2.empty:
        lines += ["## E2 (highD training): H3.4 verdict of the ACI-corrected intervals, coverage at 10 s raw / ACI, collided share", "",
                  "| which | model | set | verdict | PICP raw | PICP ACI | collided |", "|---|---|---|---|---|---|---|"]  # fmt: skip
        for r in e2.itertuples():
            lines.append(f"| {r.which} | {r.model} | {r.dataset} | {r.verdict_ci} | {r.picp_raw_10s:.3f} | {r.picp_aci_10s:.3f} | {r.collided_10s:.3f} |")
        lines.append("")
    if not pl.empty:
        lines += ["## Platoons: H3.2 reduction (extended set, response window) and mean collided share", "",
                  "| which | cell | model | mode | reduction [file interval] verdict | collided |", "|---|---|---|---|---|---|"]  # fmt: skip
        for r in pl.itertuples():
            lines.append(f"| {r.which} | {r.variant} | {r.model} | {r.mode} | {r.reduction:.2f} [{r.lo_cluster:.2f}, {r.hi_cluster:.2f}] {r.verdict_cluster} | {r.collided_mean:.2f} |")
        lines.append("")
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
