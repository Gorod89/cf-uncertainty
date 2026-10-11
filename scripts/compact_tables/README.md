# Compact tables of the article

`make_compact.py` writes the tables c1-c18 of the article and `make_c19.py` the table c19 from the experiment artefacts;
`make_table_sources.py` writes `results/TABLE_SOURCES.md`, the manifest that maps every table to its source files
(SHA-256), the tool that wrote it and the command that regenerates the sources. The published copies of the tables are
in `results/compact_tables/`.

The generators read the artefact layout of the development repository. From a public checkout, build that layout from
the published results and point `CF_UNC_ARTIFACTS` at it:

```bash
mkdir -p /tmp/cf_art && cp -r results/report /tmp/cf_art/report
for r in results/controls/*; do mkdir -p /tmp/cf_art/$(basename $r); cp -r $r/* /tmp/cf_art/$(basename $r)/; done
tar -xzf e3_platoon_tables.tar.gz -C /tmp/cf_art/e3      # the release asset: per-profile platoon tables, config, spec
CF_UNC_ARTIFACTS=/tmp/cf_art PYTHONUTF8=1 python scripts/compact_tables/make_compact.py --out /tmp/cf_build   # tables -> /tmp/cf_build/tables/, figures -> /tmp/cf_build/figures/
CF_UNC_ARTIFACTS=/tmp/cf_art PYTHONUTF8=1 python scripts/compact_tables/make_c19.py --out /tmp/cf_build/tables
PYTHONUTF8=1 python scripts/compact_tables/make_table_sources.py --out results/TABLE_SOURCES.md
```

Compare `/tmp/cf_build/tables/c*.tex` with the published `results/compact_tables/c*.tex`: c1-c8 and c10-c19 regenerate
identically (c19 up to its comment lines); c9 is a hand-assembled table and is not regenerated. `--no-copy` keeps an
existing `figures/` directory of the build directory and fails on a fresh one, so leave it out for a fresh build. The
SHA-256 values of `results/TABLE_SOURCES.md` are computed on the files with line endings normalised to LF (text files),
so they are the same on Windows and Linux checkouts.

`make_compact.py` also reads a few files outside `tables/` (the AR(1) fit logs under `<root>/checkpoints/idm_ar_noise/seed*/train_log.json`,
the matching files `<root>/matching.json`, `e3/config.yaml` and `e3/spec.json`); the ones that are not derived tables are
published under `results/controls/` or in the release asset, as the manifest lists.
