#!/usr/bin/env python3
"""Split the released checkpoint archives by the data sources of the experiment roots (preparation of a release).

The three checkpoint archives of the public releases (checkpoints.tar.gz, checkpoints_ext.tar.gz and
checkpoints_s7.tar.gz) mix roots trained on different data sources, whose providers state different terms. This
script repacks them, without extracting anything to disk and without reading a checkpoint, into one archive per group
of sources, each with a notice of the terms of its sources and a SHA-256 manifest:

    checkpoints_highd.tar.gz                 roots trained on highD-derived events only
    checkpoints_ngsim.tar.gz                 the root trained on the NGSIM I-80 events only
    checkpoints_highd_waymo_openacc.tar.gz   the pooled root without NGSIM
    checkpoints_openacc_idm.tar.gz           the two IDM calibrations on openACC events
    checkpoints_mixed_ngsim.tar.gz           roots trained on NGSIM together with highD and Waymo data

The last archive is not published: the providers of its sources ask for different conditions on a published model
(NOTICE.md of the repository). The script only writes local files.

    PYTHONUTF8=1 python extension/scripts/release_split_checkpoints.py \
        --src <directory holding the three released archives> --out extension/artifacts/release_prepared \
        [--waymo-agreement <text file with a copy of the Waymo agreement>]

Inside the new archives the paths start at the repository root (artifacts/<root>/..., extension/artifacts/<root>/...),
so every archive is unpacked there.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

READ = "2026-11-16"  # day on which the providers' terms quoted in the notices were read

# root -> (group, training sets); the training sets are those of the experiment configs (train.data)
POOL = ["highD", "NGSIM I-80", "NGSIM US-101", "Waymo", "openACC"]
LODO_OPENACC = ["highD", "NGSIM I-80", "NGSIM US-101", "Waymo"]
ROOTS: dict[str, tuple[str, list[str]]] = {
    **{r: ("highd", ["highD"]) for r in ("e1", "e1_ctrl", "e1_mean1s", "e1h", "e1h_mean1s", "e1m", "e1m_causal",
                                         "e1m_mean1s", "e1m_mean1s_abv", "e1m_rt", "ext_cl_highd")},
    "ext_e1_ngsim": ("ngsim", ["NGSIM I-80"]),
    "e6_lodo_ngsim_all": ("highd_waymo_openacc", ["highD", "Waymo", "openACC"]),
    "ext_idm_modes": ("openacc_idm", ["openACC"]),
    **{r: ("mixed_ngsim", POOL) for r in ("e6_pool", "e6_pool_ctrl", "ext_pool_prev1", "ext_pool_prev2", "ext_cl_pool")},
    **{r: ("mixed_ngsim", LODO_OPENACC) for r in ("e6_lodo_openacc", "e6_lodo_openacc_ctrl", "ext_lodo_prev1",
                                                  "ext_lodo_prev2", "ext_cl_lodo")},
    "e6_lodo_ngsim": ("mixed_ngsim", ["highD", "NGSIM US-101", "Waymo", "openACC"]),
}
SOURCES = {  # released archive -> prefix of its members in the repository layout
    "checkpoints.tar.gz": "artifacts",
    "checkpoints_ext.tar.gz": "extension/artifacts",
    "checkpoints_s7.tar.gz": "extension/artifacts",
}
OLD_MANIFESTS = {"CHECKPOINTS_MANIFEST.json", "CHECKPOINTS_EXT_MANIFEST.json", "CHECKPOINTS_S7_MANIFEST.json"}

HIGHD = (f"highD (Krajewski et al., 2018; terms and conditions at https://levelxdata.com/highd-dataset/, read {READ}): free for "
         "non-commercial use on application. Under these terms the data set and modified versions may not be distributed; "
         "derivative works may be distributed as far as they are abstract representations of the data set, trained models being "
         "named as an example, that do not allow the data set, or something similar in character, to be recovered; neither the data set nor a derivative work may be "
         "used for commercial purposes; every work that uses the data set references it. The highD events used here are those "
         "released with the FollowNet benchmark (Chen et al., 2023).")
NGSIM = (f"NGSIM (U.S. Department of Transportation, Federal Highway Administration; data.transportation.gov, id 8ect-6jqj, "
         f"doi:10.21949/1504477; read {READ}): the record names the licence \"Creative Commons Attribution | Share Alike 3.0 "
         "Unported\" (https://creativecommons.org/licenses/by-sa/3.0/legalcode). Changes made to the source for the event sets: "
         "reconstruction and smoothing of the trajectories, conversion to SI units, extraction of car-following events.")
WAYMO = (f"Waymo Open Dataset, through the car-following pairs processed by Hu et al. (2021, 2022): provided by Waymo LLC under "
         f"the Waymo Dataset License Agreement for Non-Commercial Use (https://waymo.com/open/terms; the text read on {READ} is "
         "dated March 2025). These checkpoints are released under that agreement: any further use or modification of them is "
         "subject to its terms, the non-commercial restrictions of its Section 4 included (no use in the operation of a vehicle, "
         "in production systems or for primarily commercial purposes), and the agreement applies to every recipient for the whole "
         "of these files.")
OPENACC = (f"openACC (European Commission, Joint Research Centre: Open ACC Database, doi:10.2905/JRC.KMH3D00; Makridis et al., "
           f"2021; read {READ}): the copyright notice delivered with the data licenses any copyright and sui generis right on "
           "the data set under CC BY 4.0 (basis: European Commission reuse notice, Decision 2011/833/EU): reuse is allowed "
           "provided appropriate credit is given and any changes are indicated.")
HOLD = ("The files hold trained parameters, normalisation constants, training logs and metrics; they hold no rows of a data "
        "set (the inducing inputs of the Gaussian-process checkpoints are trained parameters, see NOTICE.md of the repository).")
ATTR_WAYMO = ("Attribution asked by the Waymo agreement: these checkpoints were made using the Waymo Open Dataset, provided by "
              "Waymo LLC under the Waymo Dataset License Agreement for Non-Commercial Use, available at waymo.com/open/terms, "
              "and your access and use of such work are governed by the terms and conditions therein.")
ATTR_NGSIM = ("Attribution asked by the NGSIM record: U.S. Department of Transportation Federal Highway Administration. (2016). "
              "Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data. [Dataset]. Provided by ITS DataHub "
              "through Data.transportation.gov. http://doi.org/10.21949/1504477")
GROUPS = {
    "highd": ("Checkpoints of the roots trained on highD-derived car-following events only.", [HIGHD],
              ["These files are provided for non-commercial research and teaching."]),
    "ngsim": ("Checkpoints of the root trained on car-following events reconstructed from the NGSIM I-80 trajectories only.",
              [NGSIM], [ATTR_NGSIM]),
    "highd_waymo_openacc": ("Checkpoints of the pooled root trained on highD-derived events, the Waymo car-following pairs and "
                            "openACC events (no NGSIM set).", [HIGHD, WAYMO, OPENACC],
                            [ATTR_WAYMO, "These files are provided for non-commercial research and teaching."]),
    "openacc_idm": ("Parameters of the intelligent driver model calibrated on the openACC events driven under ACC and, "
                    "separately, by human drivers.", [OPENACC], []),
    "mixed_ngsim": ("Checkpoints of the roots trained on NGSIM event sets together with highD-derived events and the Waymo "
                    "car-following pairs (some of them with openACC events as well).", [HIGHD, NGSIM, WAYMO, OPENACC],
                    [ATTR_WAYMO, ATTR_NGSIM,
                     "These checkpoints remain subject to the terms of each of their sources. The highD and Waymo terms allow "
                     "non-commercial use only and the Waymo agreement asks to apply to a published model as a whole, while the "
                     "NGSIM record carries a ShareAlike licence; this notice does not state under which single licence the "
                     "files may be used or passed on. Use for non-commercial research and teaching with the attributions above "
                     "is the use that all of these terms allow."]),
}


def notice(group: str, roots: dict[str, list[str]], agreement: bool) -> str:
    head, terms, tail = GROUPS[group]
    lines = [f"NOTICE for checkpoints_{group}.tar.gz", "", head, "",
             "Experiment roots and their training sets (train.data of the experiment configs):"]
    lines += [f"  {r}: {', '.join(s)}" for r, s in sorted(roots.items())]
    lines += ["", "Terms of the data providers, as they state them (this notice reports the terms; it is not legal advice):"]
    lines += [f"  * {t}" for t in terms]
    lines += ["", HOLD]
    lines += [""] + tail if tail else []
    if any("Waymo" in s for s in roots.values()):
        lines += ["", "A copy of the Waymo agreement is in licenses/ of this archive." if agreement else
                  "The Waymo agreement asks that recipients be given a copy of it: read it at https://waymo.com/open/terms before use."]
    lines += ["", "The code of the repository is under the MIT licence; that licence does not extend to these files.", ""]
    return "\n".join(lines)


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes, mtime: int) -> dict:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(data), mtime, 0o644
    tar.addfile(info, io.BytesIO(data))
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="directory that holds the three released archives (searched recursively)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--waymo-agreement", type=Path, default=None, help="text file with a copy of the Waymo agreement (optional)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    found = {name: next(iter(sorted(args.src.rglob(name))), None) for name in SOURCES}
    missing = [n for n, p in found.items() if p is None]
    if missing:
        raise SystemExit(f"missing source archives below {args.src}: {missing}")
    agreement = args.waymo_agreement.read_bytes() if args.waymo_agreement else None
    outs = {g: tarfile.open(args.out / f"checkpoints_{g}.tar.gz", "w:gz", compresslevel=6) for g in GROUPS}
    files: dict[str, dict[str, dict]] = {g: {} for g in GROUPS}
    roots_of: dict[str, dict[str, list[str]]] = {g: {} for g in GROUPS}
    newest, unknown = 0, set()
    for name, path in found.items():
        with tarfile.open(path, "r:gz") as src:
            for m in src:
                if not m.isfile() or m.name in OLD_MANIFESTS:
                    continue
                root = m.name.split("/")[0]
                if root not in ROOTS:
                    unknown.add(root)
                    continue
                group, sets = ROOTS[root]
                data = src.extractfile(m).read()  # bytes only: nothing is unpickled or written outside the new archive
                new_name = f"{SOURCES[name]}/{m.name}"
                files[group][new_name] = {**add_bytes(outs[group], new_name, data, m.mtime), "from": name, "root": root}
                roots_of[group][root] = sets
                newest = max(newest, m.mtime)
    if unknown:
        raise SystemExit(f"roots without a group: {sorted(unknown)}")
    summary = {"read_terms_on": READ, "groups": {}}
    for g, tar in outs.items():
        with_waymo = any("Waymo" in s for s in roots_of[g].values())
        if with_waymo and agreement is not None:
            add_bytes(tar, "licenses/Waymo_Dataset_License_Agreement_for_Non-Commercial_Use.txt", agreement, newest)
        text = notice(g, roots_of[g], with_waymo and agreement is not None)
        add_bytes(tar, f"NOTICE_checkpoints_{g}.txt", text.encode("utf-8"), newest)
        manifest = {"archive": f"checkpoints_{g}.tar.gz", "roots": roots_of[g], "files": files[g]}
        add_bytes(tar, f"MANIFEST_checkpoints_{g}.json", json.dumps(manifest, indent=1, sort_keys=True).encode("utf-8"), newest)
        tar.close()
        (args.out / f"NOTICE_checkpoints_{g}.txt").write_text(text, encoding="utf-8")
        out = args.out / f"checkpoints_{g}.tar.gz"
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        summary["groups"][g] = {"archive": out.name, "bytes": out.stat().st_size, "sha256": digest, "files": len(files[g]),
                                "checkpoints": sum(1 for f in files[g] if f.endswith(".pt")), "roots": roots_of[g]}
        print(f"{out.name}: {len(files[g])} files ({summary['groups'][g]['checkpoints']} .pt), {out.stat().st_size} bytes, "
              f"roots {', '.join(sorted(roots_of[g]))}")
    (args.out / "SPLIT_SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
    if agreement is None:
        print("note: no copy of the Waymo agreement was given (--waymo-agreement); the notices point at its URL instead")


if __name__ == "__main__":
    main()
