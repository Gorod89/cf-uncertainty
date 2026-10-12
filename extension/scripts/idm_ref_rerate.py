#!/usr/bin/env python3
"""H3.2 reductions of the stored laws against the references calibrated per driver mode (extension S6).

The platoon units of the two references (extension/scripts/calibrate_idm_modes.py; root ext_e3_idmref, cells raw,
harmonised, pool and lodo_openacc) are joined to the per-profile table of the stored laws of artifacts/e3 and the
amplitude-CRPS analysis of E3 (scripts/analyze_e3.py: crps_tables, the same bootstrap seeds and file clusters) is run
again with ``idm_white_noise_acc`` as the reference and ``idm_white_noise_human`` and the stored ``idm_white_noise`` as
further references. The reading (PLAN.md S6) pairs the ACC profiles with the ACC reference and the human-driven
profiles with the human reference, and calls a verdict reference-robust when it equals the verdict against the highD
reference.

    extension/artifacts/ext_e3_idmref/tables/h32_rerated.csv        the h32 rows of every reference
    extension/artifacts/ext_e3_idmref/tables/amplitude_crps_rerated.csv
    extension/artifacts/ext_e3_idmref/tables/h32_mode_reference.csv the mode-matched reading next to the stored verdict
    extension/artifacts/ext_e3_idmref/tables/README.md

    PYTHONUTF8=1 python extension/scripts/idm_ref_rerate.py
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_e3 import crps_tables, load_settings  # noqa: E402

from cf_unc.experiments.e3 import select_profile_ids  # noqa: E402
from cf_unc.platoon.profiles import load_profiles  # noqa: E402

E3 = ROOT / "artifacts/e3"
REF = ROOT / "extension/artifacts/ext_e3_idmref"
STORED, ACC, HUMAN = "idm_white_noise", "idm_white_noise_acc", "idm_white_noise_human"
MATCH = {"ACC": ACC, "Human": HUMAN}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--e3", default=str(E3))
    ap.add_argument("--ref", default=str(REF))
    a = ap.parse_args()
    e3, ref = Path(a.e3), Path(a.ref)
    laws = pd.read_csv(e3 / "tables/profile_index.csv")
    refs = pd.read_csv(ref / "tables/profile_index.csv")
    variants = sorted(set(laws["variant"]) & set(refs["variant"]))
    if not variants:
        raise SystemExit("no common variant between the stored laws and the references")
    pi = pd.concat([laws[laws["variant"].isin(variants)], refs[refs["variant"].isin(variants)]], ignore_index=True)
    seed_parts = []
    for root in (e3, ref):
        p = root / "tables/profile_index_seed.csv"
        if p.exists():
            sp = pd.read_csv(p)
            seed_parts.append(sp[sp["variant"].isin(variants)])
    seed_pi = pd.concat(seed_parts, ignore_index=True) if seed_parts else None
    s = copy.deepcopy(load_settings(e3))
    s["h32"]["reference"] = ACC
    s["h32"]["references_extra"] = [HUMAN, STORED]
    profiles_list, _ = load_profiles(ROOT / s["profiles"] if not Path(s["profiles"]).is_absolute() else s["profiles"])
    profiles = {p.id: p for p in profiles_list}
    present = set(pi["profile"])
    sets = {name: [i for i in select_profile_ids(profiles_list, name, s["sets"]) if i in present] for name in s["sets"]}
    sets = {k: v for k, v in sets.items() if v}
    print(f"variants {variants}, models {sorted(set(pi['model']))}, sets { {k: len(v) for k, v in sets.items()} }")
    crps, h32 = crps_tables(pi, sets, profiles, s, seed_pi)
    out = ref / "tables"
    out.mkdir(parents=True, exist_ok=True)
    crps.to_csv(out / "amplitude_crps_rerated.csv", index=False)
    h32.to_csv(out / "h32_rerated.csv", index=False)

    sub = h32[(h32["set"] == "extended") & (h32["quantity"] == "amplitude")]
    rows = []
    for (variant, window, mode, model), part in sub.groupby(["variant", "window", "mode", "model"], sort=False):
        if mode not in MATCH or model in (ACC, HUMAN, STORED):
            continue
        matched = part[part["reference"] == MATCH[mode]]
        stored = part[part["reference"] == STORED]
        if matched.empty or stored.empty:
            continue
        m, st = matched.iloc[0], stored.iloc[0]
        rows.append({
            "variant": variant, "window": window, "mode": mode, "model": model, "reference_matched": MATCH[mode],
            "reduction_matched": m.reduction, "lo_matched": m.lo_cluster, "hi_matched": m.hi_cluster, "verdict_matched": m.verdict_cluster,
            "reduction_stored": st.reduction, "lo_stored": st.lo_cluster, "hi_stored": st.hi_cluster, "verdict_stored": st.verdict_cluster,
            "crps_reference_matched": m.crps_reference, "crps_reference_stored": st.crps_reference,
            "reference_robust": m.verdict_cluster == st.verdict_cluster, "n_profiles": m.n_profiles, "n_files": m.n_files,
        })  # fmt: skip
    table = pd.DataFrame(rows)
    table.to_csv(out / "h32_mode_reference.csv", index=False)
    lines = ["# H3.2 against the references calibrated per driver mode (extension S6)", "",
             "Reduction of the amplitude CRPS (extended set, 95 % file-cluster interval, verdict) of every stored law against the "
             "reference of its driver mode (ACC profiles: the IDM calibrated on the OpenACC ACC events; human-driven profiles: "
             "the IDM calibrated on the OpenACC human-driven events) next to the stored verdict against the highD reference. "
             "Reading rule (PLAN.md S6): reference-robust when the two verdicts agree.", ""]  # fmt: skip
    for (variant, window), part in table.groupby(["variant", "window"], sort=False):
        lines += [f"## {variant}, {window} window", "", "| model | mode | matched reference [interval] verdict | stored reference [interval] verdict | robust | CRPS ref. matched / stored |",
                  "|---|---|---|---|---|---|"]  # fmt: skip
        for r in part.itertuples():
            lines.append(f"| {r.model} | {r.mode} | {r.reduction_matched:.2f} [{r.lo_matched:.2f}, {r.hi_matched:.2f}] {r.verdict_matched} | "
                         f"{r.reduction_stored:.2f} [{r.lo_stored:.2f}, {r.hi_stored:.2f}] {r.verdict_stored} | {'yes' if r.reference_robust else 'no'} | "
                         f"{r.crps_reference_matched:.2f} / {r.crps_reference_stored:.2f} |")  # fmt: skip
        lines.append("")
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    resp = table[table["window"] == "response"]
    print(f"{len(table)} rows; robust {int(resp['reference_robust'].sum())} of {len(resp)} (response window)")
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
