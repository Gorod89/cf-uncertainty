"""Copy extracted event sets, driver splits and global IDM fits from the sibling repository cf-stability.

The parquet layout is shared between the repositories (docs/design.md, section 3.1), so nothing is
converted. Provenance (source path, config hash of the extraction, file hashes) is written to
``data/events/<name>/IMPORTED_FROM.json``.

    python scripts/import_cf_stability_events.py                      # default source ../cf-stability
    python scripts/import_cf_stability_events.py --names follownet_highd openacc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAMES = ("follownet_highd", "ngsim_i80", "ngsim_us101", "follownet_ngsim_i80", "waymo", "openacc")
EVENT_FILES = ("samples.parquet", "events.parquet", "manifest.json", "describe.json")


def sha256(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=str(ROOT.parent / "cf-stability"))
    parser.add_argument("--names", nargs="+", default=list(DEFAULT_NAMES))
    parser.add_argument("--force", action="store_true", help="overwrite event sets that were imported before")
    args = parser.parse_args()
    source = Path(args.source).resolve()

    for name in args.names:
        src = source / "data" / "events" / name
        dst = ROOT / "data" / "events" / name
        if not (src / "samples.parquet").exists():
            print(f"{name}: not found in {src}, skipped")
            continue
        if (dst / "IMPORTED_FROM.json").exists() and not args.force:
            print(f"{name}: already imported (use --force to overwrite)")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        record = {"name": name, "source": str(src), "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": {}}
        for file in EVENT_FILES:
            if (src / file).exists():
                shutil.copy2(src / file, dst / file)
                record["files"][file] = {"bytes": (dst / file).stat().st_size, "sha256": sha256(dst / file)}
        manifest = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
        record["extraction_config_hash"] = manifest.get("config_hash")
        record["git_revision"] = manifest.get("git_revision")
        record["summary"] = manifest.get("summary")

        split = source / "data" / "splits" / f"{name}_driver.json"
        if split.exists():
            target = ROOT / "data" / "splits" / split.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(split, target)
            record["files"][f"splits/{split.name}"] = {"bytes": target.stat().st_size, "sha256": sha256(target)}
        idm = source / "data" / "calibration" / name / "idm_global.json"
        if idm.exists():
            target = ROOT / "data" / "calibration" / name / "idm_global.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(idm, target)
            record["files"]["calibration/idm_global.json"] = {"bytes": target.stat().st_size, "sha256": sha256(target)}
        (dst / "IMPORTED_FROM.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        print(f"{name}: {record['summary']['n_events']} events, {record['summary']['n_samples']} samples -> {dst}")


if __name__ == "__main__":
    main()
