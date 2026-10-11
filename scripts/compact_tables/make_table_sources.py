#!/usr/bin/env python3
"""Manifest of the article tables (review V7-R04): for every compact table of paper/v13/tables the report tables or
experiment files it was built from, with their SHA-256, the script that wrote the table and the command that
regenerates the sources. Written to the public checkout as results/TABLE_SOURCES.md.

    PYTHONUTF8=1 python scripts/compact_tables/make_table_sources.py --out results/TABLE_SOURCES.md

The sources are read from the "% Sources:" comment line that make_compact.py / make_c19.py write into every
table file; paths under artifacts/report/ are published as results/report/, the other experiment tables as
results/controls/ (copied by the export script).
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
TABLES = ROOT / "results/compact_tables"  # the compact tables of the article as published

REGENERATE = {  # source directory prefix -> the command that regenerates it (from the repository root)
    "artifacts/report/": "python scripts/make_report.py",
    "artifacts/e1/": "python scripts/analyze_e1.py --root artifacts/e1 --config configs/experiment/e1.yaml (E1 run: scripts/run_e1_retry.sh)",
    "artifacts/e1_ctrl/": "python scripts/analyze_e1.py --root artifacts/e1_ctrl --config configs/experiment/e1_ctrl.yaml --h33-pair lstm_hetero_frozen lstm_const_sigma --h33-pair hetero_mdn lstm_const_sigma [--unit driver --tables-dir tables_driver]",
    "artifacts/e1m_causal/": "scripts/match_roughness.py --causal; python scripts/analyze_e1.py --root artifacts/e1m_causal --config configs/experiment/e1m_causal.yaml",
    "artifacts/e1m_rt/": "python scripts/analyze_e1.py --root artifacts/e1m_rt --config configs/experiment/e1m_rt.yaml",
    "artifacts/e1m_mean1s_abv/": "python scripts/analyze_e1.py --root artifacts/e1m_mean1s_abv --config configs/experiment/e1m_mean1s_abv.yaml",
    "artifacts/e2/": "python scripts/analyze_e2.py --root artifacts/e2 (E2 run: scripts/run_e2_retry.sh)",
    "artifacts/e3/": "python scripts/analyze_e3.py (E3 run: scripts/run_e3_retry.sh, all variants)",
    "artifacts/e3_step/": "python scripts/analyze_e3.py --root artifacts/e3_step",
    "artifacts/e3_vcap/": "python scripts/analyze_e3.py --root artifacts/e3_vcap",
    "artifacts/e6_pool_ctrl/": "python scripts/analyze_e1.py --root artifacts/e6_pool_ctrl --config configs/experiment/e6_pool_ctrl.yaml --in-domain pool_test",
    "artifacts/e6_lodo_openacc_ctrl/": "python scripts/analyze_e1.py --root artifacts/e6_lodo_openacc_ctrl --config configs/experiment/e6_lodo_openacc_ctrl.yaml --in-domain pool_test",
}


TEXT = (".csv", ".json", ".md", ".tex", ".yaml", ".yml", ".txt")


def sha256(path: pathlib.Path) -> str:
    """SHA-256 of the file; text files with their line endings normalised to LF, so that Windows (CRLF) and Linux
    checkouts give the same value."""
    data = path.read_bytes()
    if path.suffix.lower() in TEXT:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


ASSET_ONLY = ("artifacts/e3/tables/profile_index.csv", "artifacts/e3/tables/profile_index_seed.csv")  # 16 and 5 MB: release asset


def published(path: str) -> str:
    if path in ASSET_ONLY:
        return "release asset e3_platoon_tables.tar.gz: " + path[len("artifacts/e3/"):]
    if path.startswith("artifacts/report/"):
        return "results/report/" + path[len("artifacts/report/"):]
    if path.startswith("artifacts/"):
        return "results/controls/" + path[len("artifacts/"):]
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    lines = ["# Sources of the article tables", "",
             "Manuscript version v13 (public release v1.3.1). For every result table of the article: the files it was "
             "built from (SHA-256 of the file as published), the script that wrote the table and the command that "
             "regenerates its sources from the experiment artefacts. Paths under `results/report/` are the report "
             "assembly (`scripts/make_report.py`), paths under `results/controls/` the analysis tables of the experiment "
             "roots copied as they are (`results/controls/<root>/...` corresponds to `artifacts/<root>/...` of the "
             "development layout), and the per-profile platoon tables are in the release asset `e3_platoon_tables.tar.gz`. "
             "The compact tables of the article (`results/compact_tables/`) are assembled from these files by the "
             "published generators `scripts/compact_tables/make_compact.py` (c1-c18) and `make_c19.py` (c19), see "
             "`scripts/compact_tables/README.md` for the artefact layout they read; every number of a compact table is a "
             "cell or a rounding of the listed files. The trained checkpoints are not stored in the Git tree: they are distributed as "
             "the release asset checkpoints.tar.gz (with a SHA-256 manifest), and every run is regenerated by the commands of the "
             "README. The descriptive "
             "tables of the article (data sets, models, control design, protocol status, closest prior work) are written by hand "
             "from the configurations and the design document and are not listed here.", "",
             "| Table | Caption (start) | Sources (published path, SHA-256) | Written by | Regenerate sources |",
             "|---|---|---|---|---|"]
    missing = []
    for tex in sorted(TABLES.glob("c*.tex"), key=lambda p: int(re.match(r"c(\d+)", p.name).group(1))):
        text = tex.read_text(encoding="utf-8")
        label = re.search(r"% (tab:[a-z0-9_]+):", text)
        src = re.search(r"% Sources?: ([^\n]+)", text)
        writer = re.search(r"% Written by ([^;\n]+)", text)
        cap = re.search(r"\\caption\{([^}]{0,90})", text)
        sources = [s.strip() for s in src.group(1).split(",")] if src else []
        cells = []
        for s in sources:
            p = ROOT / s
            if not p.exists() and not published(s).startswith("release asset"):
                p = ROOT / published(s)  # a public checkout: the file under results/
            if p.exists():
                cells.append(f"`{published(s)}` {sha256(p)[:16]}…")
            elif published(s).startswith("release asset"):
                cells.append(f"`{published(s)}`")
            else:
                cells.append(f"`{published(s)}` (missing)")
                missing.append(s)
        regen = sorted({cmd for prefix, cmd in REGENERATE.items() if any(s.startswith(prefix) for s in sources)})
        wrote = writer.group(1).strip() if writer else "by hand"
        wrote = re.sub(r"paper/v\d+/tools/", "scripts/compact_tables/", wrote)  # the published copy of the generator
        lines.append(f"| `{label.group(1) if label else tex.stem}` | {cap.group(1).replace('|', '/') if cap else ''} | "
                     f"{'<br>'.join(cells) or '(hand-written)'} | {wrote} | "
                     f"{'<br>'.join(f'`{c}`' for c in regen)} |")
    lines += ["", "SHA-256 values are computed on the files with line endings normalised to LF (text files), the same on Windows and "
             "Linux checkouts; the first 16 hex digits are listed above.", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", args.out, "missing sources:", missing)


if __name__ == "__main__":
    main()
