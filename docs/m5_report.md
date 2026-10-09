# Milestone M5: report

Date: 2026-10-07. Contract: `docs/design.md`, section 14 (decision of the M4 review: the H3.2
verdict is stated per driver mode). Deliverables: the final report `docs/report.md` and the
assembled tables and figures in `artifacts/report/` (`index.md` lists every file with its source).

## 1. Commands

```bash
cd cf-uncertainty && export PYTHONUTF8=1
./.venv/Scripts/python.exe scripts/analyze_e3.py --root artifacts/e3 --quiet                   # per-mode H3.2 block, F5 bars
./.venv/Scripts/python.exe scripts/analyze_e3.py --root artifacts/e3_vcap --set headline --quiet
./.venv/Scripts/python.exe scripts/analyze_e3.py --root artifacts/e3_step --set headline --quiet
./.venv/Scripts/python.exe scripts/make_report.py                                             # artifacts/report (about 5 s, idempotent)
./.venv/Scripts/python.exe -m pytest -m "not slow"                                             # 495 passed, 24 deselected
```

## 2. What was built

* `scripts/make_report.py` (+ `tests/test_make_report.py`): reads only the tables and figures of
  the experiment roots (no experiment, no bootstrap) and writes `artifacts/report/tables/*.csv|.tex`
  (32 tables T1-T6, booktabs) and `artifacts/report/figures/` (172 files: F1 maps per root, F2 shift
  ratio bars, F3 NLL and variance by regime, F4 coverage and width by horizon, F5 amplitude CRPS
  bars per driver mode, F6 growth curves, F7 CRPS by vehicle index) and `index.md`.
* `scripts/analyze_e3.py`, `cf_unc/report/figures_e3.py`: the summary gives the H3.2 block per
  driver mode (ACC, Human, all; `mixed` not judged) for every variant; new figure
  `crps_bars__<variant>.png`; `--set` also selects the profile set of the figures; label of the
  kept-member ensemble unified with E2 ("Deep ensemble (kept member)").
* `docs/report.md`: the final report (verdicts, data, models, experiments, results by hypothesis,
  findings, deviations from the plan, limitations, reproduction).
* Tests: 495 fast (6 new), 24 slow / GPU deselected.

## 3. Key numbers (full tables in `docs/report.md`, section 0 and 4.4)

| H3.2 subset | DiffusionCF reduction vs IDM + white noise | verdict | other models |
|---|---|---|---|
| ACC platoons, 47 profiles | 0.34 [0.09, 0.50] (harmonised 0.40 [0.19, 0.53]) | inconclusive | GP residual 0.15 [-0.08, 0.35] inconclusive; QuantilePIDL, deep ensemble, LSTM refuted |
| human-driven platoons, 12 profiles | -0.84 [-1.26, -0.47] | refuted | every learned model refuted (reference CRPS 1.1 m/s) |
| all, 60 profiles (secondary) | 0.25 [0.01, 0.44] (harmonised 0.31 [0.10, 0.46]) | inconclusive | GP residual -0.06 [-0.34, 0.16]; the rest refuted |

H3.1 refuted as written, H3.1e supported, H3.3 supported, H3.4 supported (unchanged from M4).

## 4. Deviations and notes

* No new experiment was run in M5; the three `analyze_e3` reruns reproduce every table
  byte-identically and add the per-mode block and the bar figure.
* The headline subset has 6 ACC profiles, enough to be judged by the >= 5 rule; there the GP residual
  is formally "supported" (0.57 [0.34, 0.74]), which the 47 ACC profiles do not confirm (0.15). The
  report states this as a warning about small subsets; the rule is kept.
* Source-table quirks recorded by the assembly (no effect on the report): the empirical growth
  curve has NaN at indices 9-10 (2 profiles) and an unrepresentative last index (T6 reports the
  fitted range 3.46 -> 4.15 m/s); `static_loo` amplitude intervals have infinite width at
  few-profile indices (index "all" is clean); the `unknown` driver subset of the E1 tables has NaN
  estimates (unused); `h32.csv` writes "inconclusive" both for straddling bounds and for too few
  profiles (the T5 tables add a `judged` column); the copied E1/E2 figures keep their own data-set
  labels.
* `configs/empirical_growth.yaml` stays a template: no literature platoon series were entered, the
  measured growth comes from OpenACC only (report, limitations).
