# Pre-submission review fixes of version v3: analysis side (R1, R4, R7, R8, R9, R11, consistency items)

Date: 2026-10-09. Base: commit `86f9d70` of the development repository plus the working-tree changes listed at the
end. Scope: the analysis items of `paper/submission/review_systems/RESPONSE_PLAN_RU.md` marked v3 (R1, R4, R7, R8, R9,
R11 and the analysis-side rows of the consistency table of `REVIEW_SYSTEMS_RU.md`). Nothing was trained or rolled
out: every number below is recomputed on CPU from the existing prediction, rollout and platoon tables. The text of
the manuscript (`paper/v3/sections`) and `docs/design.md` are not changed here; the decision table of R7 and the
definitions below are written so that they can be copied there.

What was rerun (all on the current code):

* `scripts/analyze_e2.py --root <root> --quiet` and `scripts/summarize_e2.py` for `artifacts/e2`, `e2h`, `e2m`,
  `e2_pool`, `e2_lodo_ngsim`, `e2_lodo_openacc`, `e2_lodo_ngsim_all` (3-8 min each);
* `scripts/analyze_e3.py --root artifacts/e3` (1 min);
* `scripts/analyze_e1.py` for the ten E1 roots with their original arguments (`e1`, `e1_mean1s`; `e1h`, `e1h_mean1s`,
  `e1m`, `e1m_mean1s` with `--config configs/experiment/<root>.yaml`; the four E6 roots with `--config ...
  --in-domain pool_test`), which reproduces every existing column (identical for `e1` and `e1_mean1s`, within 1e-13
  for the roots of harmonised and matched signals; the Monte-Carlo p-values of `regime_ordering` of `e1h`, `e1m` and
  the E6 roots move by at most 0.015, without any change of a Holm rejection or of a verdict) and adds the new
  columns; plus four follower-cluster runs `--unit driver --tables-dir tables_driver` (`e1`, `e1m`,
  `e6_lodo_openacc`, `e6_lodo_ngsim_all`);
* `scripts/make_report.py` (`artifacts/report`) and `paper/v3/tools/make_compact.py` (`paper/v3/tables`).

The tables of the first version stay reproducible: `scripts/analyze_e2.py --root <root> --finite-quantile
--no-streams --tables-dir tables_finite` gives the old coverage, sensitivity, one-step and collision tables
(checked on `e2_lodo_ngsim`: identical to 1e-15), and the H3.4 rule of the first version is documented in R7.

## R1. ACI with the behaviour of Gibbs and Candès at the ends of the level range

**What was wrong.** `aci_quantile` clipped the quantile level `1 - alpha_t` to [0, 1], so a level >= 1 gave the
largest score seen and a level <= 0 the smallest: every correction was finite and an interval issued at `alpha_t <= 0`
could still miss. The bound of Gibbs and Candès (2021, section 4.1) needs exactly the opposite: at `alpha_t <= 0`
the interval is the whole line (it always covers), at `alpha_t >= 1` it is empty (it never covers). The reviewer's
counterexample (scores 1, ..., 1000 after a history [0], alpha 0.1, gamma 0.01, n_min 1) gave coverage 0 and
`alpha_t` -> -8.9.

**What the code does now** (`cf_unc/rollout/calibration.py`):

* `aci_quantile(level)`: `+inf` for a level >= 1, `-inf` for a level <= 0; inside (0, 1) the warm-up (0 while fewer
  than `n_min` scores were seen) and the order statistic of rank `ceil(level n)` as before. The ends take
  precedence over the warm-up. `alpha_t` is never clipped (`alpha_{t+1} = alpha_t + gamma (alpha - err_t)`).
* `aci_run` returns per event `infinite` / `empty` flags next to `q`, `alpha_t`, `covered`; the state carries
  `finite_quantile` (JSON round trip; an old state without the key loads as the corrected variant).
* Rollout intervals: `Q = +inf` is the whole line (infinite width, counted as covered), `Q < -1` (`-inf` included) is
  empty (width 0, never covered; the interval score is never below -1).
* The heuristic finite variant stays available as the explicit option `finite_quantile=True` (`aci_quantile`,
  `aci_run`, `coverage_table`, `coverage_by_method`; settings `calibration.finite_quantile` of E2 and E3;
  `scripts/analyze_e2.py --finite-quantile`), documented as the heuristic of the first-version tables, for which the
  guarantee does not hold.
* Coverage tables (`coverage_by_horizon.csv`, `aci_gamma_sensitivity.csv`) have new columns: `share_infinite` and
  `share_empty` (share of the (seed, event) intervals), `width_finite_mean` / `width_finite_median` (widths of the
  finite intervals; per event the mean over the seeds with a finite interval). `width_mean` / `width_median` stay
  "all intervals" and become +inf as soon as one infinite interval enters the mean (or more than half the median).
  The width figure of E2 plots the finite widths.

**Guarantee that now holds for the code** (for any score sequence, warm-up included, per horizon and stream):
`alpha_t` stays in `[-gamma, 1 + gamma]` and over T events `|mean(err) - alpha| <= (max(alpha_1, 1 - alpha_1) +
gamma) / (T gamma)`. It is a long-run average over the events of a stream, per horizon, not a conditional or a
trajectory guarantee; with labels delayed by d events the state can overshoot by d steps (R8). For the E2 streams
(500 calibration + 500 evaluated events, gamma 0.01, alpha 0.1) the bound is 0.091 over the 1000 events: it is the
measured coverage (0.88-0.94 in every row; 0.85-0.99 on the 185-event Waymo test fold of `e2_pool`), not the bound,
that the tables report.

**Tests** (`tests/test_calibration.py`): the reviewer's counterexample (finite variant: coverage 0, `alpha_t` = -8.9;
corrected: coverage 0.89, `alpha_t` in [-0.009, 0.1], the 89 % covered events are the whole line - the honest answer
for a monotone stream); the long-run bound and the `alpha_t` range on i.i.d., drifting (scale x5), shifted,
monotone and adversarial (rare huge scores) streams for alpha 0.1 / 0.3, gamma 0.005 / 0.02 / 0.1, with and without
warm-up and history (3 seeds x 5 streams x 18 settings); the identity `alpha_T = alpha_1 + gamma sum(alpha - err)`;
the warm-up with both variants; continuation of a state and its JSON round trip; a coverage table in which a strong
shift produces infinite intervals while the finite variant produces none, with raw and static unchanged.

**What changed in the E2 tables** (all seven roots recomputed; ACI corrections only, raw and static are untouched):

| Root | ACI online: PICP before -> after (all rows) | max \|change\| | rows with infinite intervals (of shifted rows) | largest share infinite | ACI frozen: max \|change\| |
|---|---|---:|---:|---:|---:|
| e2 (recorded) | 0.880-0.918 -> 0.880-0.918 | 0.0008 | 151 / 416 | 4.9 % | 0.016 |
| e2h (harmonised) | 0.882-0.922 -> 0.882-0.922 | 0.0004 | 174 / 416 | 5.3 % | 0.012 |
| e2m (matched) | 0.885-0.941 -> 0.885-0.941 | 0.0004 | 2 / 416 | 0.2 % | 0.009 |
| e2_pool | 0.854-0.986 -> 0.854-0.986 | 0.0011 | 2 / 208 | 0.3 % | 0.002 |
| e2_lodo_ngsim | 0.886-0.938 -> 0.886-0.938 | 0.0004 | 0 / 52 | 0 | 0.003 |
| e2_lodo_openacc | 0.890-0.916 -> 0.890-0.916 | 0.0004 | 0 / 52 | 0 | 0.003 |
| e2_lodo_ngsim_all | 0.886-0.936 -> 0.886-0.936 | 0 | 0 / 104 | 0 | 0 |

(in-domain highD rows: no infinite interval in any root; no empty interval anywhere; ACI frozen never ends with
`alpha_t <= 0`, so it has no infinite interval either.) The infinite intervals occur mostly in the first 50-80 events after the
switch from highD validation to a shifted set whose scores are much larger (recorded and harmonised signals; median
position 39-79 of 500 in the cases checked, occasionally later after a run of misses): a run of misses drives
`alpha_t` below 0, the next few intervals are the whole line, and `alpha_t` climbs back by `gamma alpha` per event.
Some also occur in the calibration stream on highD validation, where the raw coverage is low (not evaluated). Before the fix these intervals were "the largest score seen", which covered the same events, so the coverage
does not move. The median widths change by -1.4 % to +9.8 % (an event counts as infinite as soon as one of its five
seeds is); the mean widths of the 151 / 174 affected rows are now infinite, and their finite-interval means
(`width_finite_mean`) are up to 61 % (recorded) and 76 % (harmonised) below the old means, because the old "largest
score" intervals of the adaptation phase were the widest finite ones. In the platoon analysis (E3) the
amplitude ACI over the profiles never reaches a level outside (0, 1): `amplitude_coverage.csv` is unchanged.

H3.4 verdict counts are unchanged by the ACI fix alone (the old rule applied to the new coverage gives the old counts in
every root, target and correction); the changes of the verdict counts come from the decision table of R7 below.

## R7. H3.4: one decision table for the paper, design.md and the code

The Methods said that a correction that fails refutes H3.4, the code (`h34_verdict`) returned "inconclusive" in that
case and refuted only when the raw coverage was >= 0.87 at every horizon; the one-step part was taken from the highD
test fold for every shifted row without saying so. The code now implements the table below
(`cf_unc/rollout/calibration.py::h34_verdict`, used by `scripts/analyze_e2.py`; test
`tests/test_calibration.py::test_h34_decision_table` and the end-to-end check in
`tests/test_e2_runner.py::test_analyze_e2_end_to_end`, which recomputes every row of `h34_verdicts.csv` from the
coverage table). H3.4 is a statement per model, target and evaluated data set; it combines one in-domain proposition
(i) with two propositions about the data set's rollouts (ii, iii).

| Part | Quantity | Data | Holds (point rule) | Interval rule: holds / fails / straddles |
|---|---|---|---|---|
| (i) one-step calibration works in-domain | PICP of the split-conformal one-step 90 % interval (normalised score, calibrated on highD validation) | the highD test fold, the SAME number for every data set's row (column `onestep_source`; the data set's own one-step coverage is reported as `onestep_picp_dataset`, not used) | 0.87 <= PICP <= 0.93 | holds: interval inside [0.87, 0.93]; fails: interval entirely outside; else straddles |
| (ii) premise: raw rollout intervals under-cover | PICP of the raw 90 % gap interval at 1, 3, 5, 10 s | the row's data set | PICP < 0.80 at some horizon | holds: upper bound < 0.80 at some horizon; fails: lower bound >= 0.80 at every horizon; else straddles |
| (iii) the correction restores coverage | PICP of the corrected interval (ACI online for the headline; static and ACI frozen reported too) | the row's data set | PICP >= 0.87 at every horizon | holds: lower bound >= 0.87 at every horizon; fails: upper bound < 0.87 at some horizon; else straddles |

| Verdict | Point rule (`verdict`) | Interval rule (`verdict_ci`, 95 % event bootstrap) |
|---|---|---|
| supported | (i), (ii) and (iii) hold | all three parts hold |
| refuted | any part fails: (iii) fails = the correction does not restore coverage; (ii) fails = the premise does not hold, the raw intervals already cover (>= 0.80 at every horizon); (i) fails = no in-domain one-step calibration | any part fails |
| inconclusive | only when a needed number is missing (never in the tables) | no part fails and at least one bound straddles its threshold |

Change against the first version: "refuted" was "raw coverage >= 0.87 at every horizon" and a failing (i) or
(iii) gave "inconclusive"; now a failing correction refutes, as the Methods say, and the premise threshold of the
refutation is the threshold of (ii) itself (0.80). `h34_verdict` still accepts the key `refute_min` of the stored
configs and ignores it. The point and the interval verdicts are published side by side
(`artifacts/report/tables/t4_h34_verdict_counts.csv`, one row per root, target and correction).

Verdict counts per root (supported / refuted / inconclusive over the (model, data set) cases; "old" = the first
tables, i.e. the old rule with the finite ACI; the old rule applied to the corrected ACI gives exactly the old
counts in every row, so all changes come from the decision table):

| Root | Target | ACI online: point old -> new | ACI online: interval old -> new | static: point old -> new | ACI frozen: point old -> new |
|---|---|---|---|---|---|
| e2 (recorded) | a | 65/0/0 -> 65/0/0 | 32/0/33 -> 33/0/32 | 15/0/50 -> 15/50/0 | 14/0/51 -> 14/51/0 |
| | mean_1s | 65/0/0 -> 65/0/0 | 29/0/36 -> 29/0/36 | 16/0/49 -> 16/49/0 | 16/0/49 -> 16/49/0 |
| e2h (harmonised) | a | 62/2/1 -> 62/3/0 | 32/1/32 -> 32/3/30 | 13/2/50 -> 13/52/0 | 14/2/49 -> 14/51/0 |
| | mean_1s | 63/1/1 -> 63/2/0 | 19/0/46 -> 19/2/44 | 12/1/52 -> 12/53/0 | 12/1/52 -> 12/53/0 |
| e2m (matched) | a | 55/9/1 -> 55/10/0 | 42/7/16 -> 42/10/13 | 20/9/36 -> 20/45/0 | 22/9/34 -> 22/43/0 |
| | mean_1s | 59/4/2 -> 59/6/0 | 40/2/23 -> 40/6/19 | 25/4/36 -> 25/40/0 | 27/4/34 -> 27/38/0 |
| e2_pool | a | 54/9/2 -> 54/11/0 | 38/8/19 -> 38/10/17 | 32/9/24 -> 32/33/0 | 31/9/25 -> 31/34/0 |
| e2_lodo_ngsim | a | 22/4/0 -> 22/4/0 | 16/2/8 -> 16/4/6 | 19/4/3 -> 19/7/0 | 18/4/4 -> 18/8/0 |
| e2_lodo_openacc | a | 22/4/0 -> 22/4/0 | 18/3/5 -> 18/4/4 | 15/4/7 -> 15/11/0 | 20/4/2 -> 20/6/0 |
| e2_lodo_ngsim_all | a | 33/1/5 -> 33/6/0 | 26/0/13 -> 26/5/8 | 24/1/14 -> 24/15/0 | 22/1/16 -> 22/17/0 |

Reading: with ACI online and recorded signals H3.4 is supported in 65/65 cases by the point rule but in 33/65
(target `a`) and 29/65 (`mean_1s`) by the interval rule; every other case is inconclusive because the lower bound of
the corrected coverage at some horizon lies at 0.85-0.87 (part iii straddles; one `mean_1s` case: part ii
straddles), none is refuted. The cases refuted with ACI online (other roots) are rows whose raw intervals already
cover at every horizon (part ii fails: Diffusion with harmonised, matched or pooled signals, MC dropout with frozen
draws per trajectory, IDM + white noise on OpenACC with `mean_1s`) and one row where the correction fails (IDM fitted
by maximum likelihood on Waymo, e2_pool, 0.85). With the static correction and ACI frozen, (iii) fails on most shifted
sets, which the old code called "inconclusive" and which is now "refuted": without target labels the correction does
not restore coverage under shift. The paper should quote the point and the interval counts side by side and say that
part (i) is the in-domain highD number for every row.

## R8. Information budget of the online corrections

Online ACI uses the labels of the evaluated (target) domain one by one, in a seeded random order. Four further variants
are computed for every E2 root at the main gamma (0.01), all starting from the same state calibrated on highD
validation (`cf_unc/rollout/calibration.py::stream_run`, `delayed_availability`; `coverage_table(...,
stream_gamma=...)`):

| Variant | Order of the evaluated events | When a label is used | Correction |
|---|---|---|---|
| `aci_online` (as before) | seeded random | before the next event | ACI |
| `aci_online_chrono` | chronological | before the next event | ACI |
| `aci_online_delayed` | chronological | a horizon after its interval was issued | ACI |
| `static_prequential` | seeded random (as `aci_online`) | before the next event | split-conformal quantile of the highD calibration scores and every evaluated score seen so far: the same labels as ACI, no learning rate |
| `static_prequential_delayed` | chronological | a horizon after issue (as `aci_online_delayed`) | as `static_prequential` |

Clocks (`cf_unc/rollout/streams.py`, from the meta of the events, `stream_info.csv` per root): NGSIM I-80 and US-101
have one absolute clock (`meta.t0` = seconds since frame 0 of the first of the three recording periods, cf-stability's
reconstruction), so issue times (event start + 2.9 s, the start of the rollout) and label times (+ horizon) are real
times; OpenACC has one clock per recording run (`meta.t0` within the file; runs ordered by date, campaign and run name)
and Waymo one per 20 s segment (ordered by segment id), the runs placed one after the other; the FollowNet highD events
carry no comparable time (`t0` = 0 for all), so their order is the source order (`source_index`) and the delay is
counted in events, `ceil(h x 1 event/s)` = 1, 3, 5, 10 events at 1, 3, 5, 10 s (setting
`calibration.delay_events_per_s`; conservative: the timed NGSIM streams carry 0.18 rolled events per second of
recording). Mean number of events issued before a label arrives, at 10 s: NGSIM I-80 1.8-2.0, US-101 1.8-1.9, Waymo
0.5-1.5, OpenACC 0.05-0.25, highD 10 (the 500 rolled events are a subsample of the recordings, so real delays are
short in events). The coverage over the first 50 and the first 100 events of each stream (`picp_first50`,
`picp_first100`) shows the cold start after the switch from highD validation to the evaluated set.

Coverage of the 90 % gap interval at 10 s, target `a`, range over the 13 model labels (full tables per horizon in
`<root>/tables/summary.md`, `artifacts/report/tables/t4_info_budget__<variant>__<target>.csv`, compact table c12):

| Root | Data set | static | ACI frozen | static prequential | ACI online | ACI chronological | ACI delayed | static prequential, delayed | delayed ACI, first 50 / first 100 |
|---|---|---|---|---|---|---|---|---|---|
| e2 (recorded) | highD | 0.88-0.92 | 0.87-0.94 | 0.89-0.92 | 0.89-0.90 | 0.90-0.91 | 0.90-0.91 | 0.89-0.92 | 0.86-0.98 / 0.88-0.93 |
| | NGSIM I-80 | 0.31-0.91 | 0.31-0.90 | 0.65-0.90 | 0.88-0.90 | 0.89-0.91 | 0.89-0.91 | 0.66-0.91 | 0.70-0.92 / 0.79-0.90 |
| | NGSIM US-101 | 0.37-0.93 | 0.36-0.93 | 0.68-0.92 | 0.89-0.90 | 0.89-0.90 | 0.89-0.90 | 0.68-0.92 | 0.69-0.94 / 0.80-0.92 |
| | Waymo | 0.51-0.85 | 0.51-0.85 | 0.72-0.88 | 0.89-0.90 | 0.89-0.90 | 0.89-0.90 | 0.71-0.87 | 0.71-0.88 / 0.82-0.90 |
| | OpenACC | 0.66-0.93 | 0.63-0.92 | 0.77-0.92 | 0.89-0.90 | 0.88-0.91 | 0.88-0.91 | 0.79-0.92 | 0.69-0.92 / 0.81-0.88 |
| e2h (harmonised) | NGSIM I-80 | 0.36-0.90 | 0.35-0.91 | 0.67-0.90 | 0.89-0.90 | 0.89-0.91 | 0.89-0.91 | 0.68-0.90 | 0.69-0.99 / 0.79-0.91 |
| e2m (matched) | NGSIM I-80 | 0.72-0.95 | 0.68-0.95 | 0.81-0.94 | 0.89-0.91 | 0.89-0.91 | 0.89-0.91 | 0.80-0.94 | 0.76-0.94 / 0.82-0.93 |
| | OpenACC | 0.74-0.95 | 0.69-0.95 | 0.82-0.94 | 0.89-0.91 | 0.89-0.91 | 0.89-0.91 | 0.83-0.94 | 0.68-0.92 / 0.83-0.92 |
| e2_pool | NGSIM I-80 | 0.72-0.94 | 0.72-0.93 | 0.81-0.94 | 0.89-0.92 | 0.90-0.92 | 0.90-0.92 | 0.82-0.94 | 0.79-0.96 / 0.84-0.94 |
| | OpenACC | 0.81-0.94 | 0.81-0.94 | 0.85-0.93 | 0.90-0.91 | 0.89-0.92 | 0.89-0.92 | 0.85-0.93 | 0.84-0.93 / 0.84-0.92 |
| e2_lodo_ngsim | NGSIM I-80 (held out) | 0.72-0.95 | 0.73-0.94 | 0.80-0.95 | 0.89-0.91 | 0.90-0.92 | 0.90-0.92 | 0.83-0.95 | 0.77-0.96 / 0.83-0.95 |
| e2_lodo_ngsim_all | NGSIM I-80 (held out) | 0.72-0.91 | 0.72-0.94 | 0.80-0.91 | 0.89-0.90 | 0.89-0.91 | 0.89-0.91 | 0.83-0.92 | 0.77-0.94 / 0.83-0.92 |
| e2_lodo_openacc | OpenACC (held out) | 0.84-0.95 | 0.84-0.95 | 0.87-0.94 | 0.90-0.91 | 0.89-0.92 | 0.89-0.92 | 0.86-0.94 | 0.72-0.94 / 0.82-0.93 |

Reading:

* Order and delay do not explain ACI's coverage: row by row (root, model, target, data set, horizon) the
  chronological and the delayed ACI differ from the random-order ACI by at most 0.022 (median +0.002), and from each
  other by at most 0.002; at 10 s, target `a`, they stay within 0.87-0.92 on every data set of every root (the delays
  of the rolled streams are short).
* The label budget alone does not explain it either: the static correction recalibrated on exactly the labels ACI
  uses reaches 0.60-1.00 under shift over all horizons (up to 0.30 below ACI in the same row; recorded NGSIM I-80 at
  10 s: 0.65-0.90), because its history keeps the 500 highD calibration scores while ACI moves its level: what
  restores coverage is the adaptive level (the learning rate) together with the target-domain labels. A static
  correction on the target labels alone (a different estimator) was not computed.
* The cold start after a domain switch is visible: over the first 50 / 100 events of the delayed stream (10 s) the
  coverage of a shifted set falls to 0.68-0.70 / 0.79-0.84 for the LSTM-based networks (constant-sigma LSTM,
  HeteroMDN) on NGSIM, Waymo and OpenACC with recorded, harmonised and matched signals; it stays near nominal on the
  in-domain stream and is closer to nominal for the pooled-domain models (first 50: 0.72-0.97).
* The coverage of ACI is a long-run average per horizon over a stream with feedback from the target domain. It is not
  the coverage of a whole trajectory and not a safety guarantee; the frozen and static rows are the numbers without
  target-domain labels.

## R4. Dependence of the platoon profiles: paired file-cluster bootstrap

`scripts/analyze_e3.py` now gives every interval of a mean over profiles (`amplitude_crps.csv`), of an H3.2 reduction
(`h32.csv`) and of a growth exponent (`growth.csv`) a second, paired cluster-bootstrap interval over the recording file
of the profiles (`<campaign>/<file>` of `configs/platoon/profiles.yaml`; 60 profiles from 26 files, ACC 47 from 23,
human-driven 12 from 3; all profiles of a drawn file are kept, 10 000 replicates, the seed of the reviewer's check
20261009 for every row; `*_cluster` columns). The estimand stays profile-weighted, so every point estimate is
unchanged (checked: `h32.csv`, `amplitude_crps.csv`, `amplitude_coverage.csv`, `collisions.csv`, `profile_index.csv`
identical to the first-version tables in every old column). `h32.csv` adds `verdict_cluster` (the same thresholds,
supported when the lower bound >= 0.20, refuted when the upper bound < 0.05, on the file interval), `n_files` and the
between-seed variability: `reduction_seed_min` / `reduction_seed_max`, the reduction computed from the rollouts of
each of the five training seeds alone (model seed k against reference seed k; per-seed CRPS in
`profile_index_seed.csv`).

Reproduction of the reviewer's numbers (Diffusion on ACC, extended set, window profile + tail): all four agree to the
fourth decimal (same draws: `rng.integers(0, 23, (10000, 23))` with seed 20261009, linear percentiles):

| Training and signals | reduction | profile interval (first version) | file interval (here) | reviewer | verdict profiles / files |
|---|---:|---:|---:|---:|---|
| highD, recorded | 33.6 % | [8.5, 50.1] | [-26.1, 57.3] | [-26.1, 57.3] | inconclusive / inconclusive |
| highD, harmonised | 39.7 % | [19.3, 53.3] | [-9.3, 59.2] | [-9.3, 59.2] | inconclusive / inconclusive |
| pooled, OpenACC held out | 46.9 % | [38.6, 52.9] | [26.5, 55.4] | [26.5, 55.4] | supported / supported |
| pooled, all five sets | 56.3 % | [45.2, 64.2] | [29.4, 67.4] | [29.4, 67.4] | supported / supported |

The human-driven intervals of the reviewer agree as well (recorded [-100.4, -1.8] %, harmonised [-135.1, 2.0] %,
held out [-58.4, 14.0] %, pooled [-108.4, 14.5] %).

All models and variants (reduction against IDM + white noise; profile interval; file interval; verdict by profiles /
by files; range of the five per-seed reductions):

| Variant | Mode (profiles / files) | Model | reduction [profiles] | [files] | verdict | seeds |
|---|---|---|---|---|---|---|
| recorded | ACC (47 / 23) | Diffusion | 0.34 [0.09, 0.50] | [-0.26, 0.57] | inc. / inc. | 0.23 to 0.45 |
| | | GP residual | 0.15 [-0.08, 0.35] | [-0.40, 0.45] | inc. / inc. | 0.11 to 0.16 |
| | | QuantilePIDL | -0.59 [-1.03, -0.29] | [-1.51, -0.19] | ref. / ref. | -8.39 to -0.06 |
| | | deep ensemble | -5.90 [-9.38, -3.50] | [-11.20, -2.65] | ref. / ref. | -9.42 to -3.82 |
| | | LSTM | -15.76 [-22.20, -10.89] | [-24.77, -8.63] | ref. / ref. | -43.98 to -3.47 |
| | Human (12 / 3) | Diffusion | -0.84 [-1.26, -0.47] | [-1.00, -0.02] | ref. / ref. | -1.31 to -0.60 |
| | | GP residual | -2.75 [-4.19, -1.85] | [-4.44, -0.88] | ref. / ref. | -3.20 to -2.37 |
| | | QuantilePIDL | -7.03 [-8.50, -6.12] | [-7.78, -4.26] | ref. / ref. | -51.95 to -3.71 |
| | | deep ensemble | -23.0 [-29.6, -18.5] | [-31.5, -1.9] | ref. / ref. | -51.1 to -11.5 |
| | | LSTM | -49.7 [-68.9, -37.3] | [-73.0, -7.6] | ref. / ref. | -187.1 to -34.5 |
| | all (60 / 26) | Diffusion | 0.25 [0.01, 0.44] | [-0.31, 0.52] | inc. / inc. | 0.17 to 0.33 |
| | | GP residual | -0.06 [-0.34, 0.16] | [-0.79, 0.34] | inc. / inc. | -0.12 to -0.05 |
| harmonised | ACC | Diffusion | 0.40 [0.19, 0.53] | [-0.09, 0.59] | inc. / inc. | 0.28 to 0.45 |
| | | GP residual | 0.14 [-0.10, 0.33] | [-0.42, 0.43] | inc. / inc. | 0.09 to 0.13 |
| | | QuantilePIDL | -0.37 [-0.81, -0.04] | [-1.51, 0.08] | **ref. / inc.** | -0.76 to -0.44 |
| | | deep ensemble | -13.9 [-20.4, -9.2] | [-22.8, -7.2] | ref. / ref. | -24.7 to -9.5 |
| | | LSTM | -2.69 [-4.25, -1.55] | [-4.81, -1.30] | ref. / ref. | -18.82 to -1.58 |
| | Human | Diffusion | -0.84 [-1.35, -0.52] | [-1.35, 0.02] | ref. / ref. | -1.39 to -0.58 |
| | | GP residual | -2.95 [-4.67, -1.97] | [-5.15, -1.29] | ref. / ref. | -3.17 to -2.81 |
| | all | Diffusion | 0.31 [0.10, 0.47] | [-0.18, 0.54] | inc. / inc. | 0.22 to 0.35 |
| OpenACC held out | ACC | Diffusion | 0.47 [0.39, 0.53] | [0.27, 0.55] | supp. / supp. | 0.37 to 0.52 |
| | | GP residual | 0.19 [-0.03, 0.38] | [-0.23, 0.40] | inc. / inc. | 0.10 to 0.19 |
| | | QuantilePIDL | -0.16 [-0.55, 0.12] | [-0.89, 0.22] | inc. / inc. | -3.15 to -0.03 |
| | | deep ensemble | -9.19 [-12.29, -7.14] | [-13.79, -6.30] | ref. / ref. | -17.07 to -6.31 |
| | | LSTM | -20.5 [-28.0, -15.0] | [-31.7, -12.1] | ref. / ref. | -50.3 to -13.2 |
| | Human | Diffusion | -0.48 [-0.85, -0.16] | [-0.58, 0.14] | **ref. / inc.** | -0.83 to -0.18 |
| | | GP residual | -1.49 [-2.21, -1.03] | [-1.82, -1.01] | ref. / ref. | -1.88 to -1.31 |
| | all | Diffusion | 0.40 [0.30, 0.48] | [0.15, 0.53] | **supp. / inc.** | 0.33 to 0.42 |
| | | QuantilePIDL | -0.42 [-0.89, -0.07] | [-1.32, 0.08] | **ref. / inc.** | -3.90 to -0.41 |
| pooled | ACC | Diffusion | 0.56 [0.45, 0.64] | [0.29, 0.67] | supp. / supp. | 0.54 to 0.59 |
| | | GP residual | 0.21 [0.03, 0.36] | [-0.16, 0.40] | inc. / inc. | 0.11 to 0.22 |
| | | QuantilePIDL | -0.22 [-0.61, 0.06] | [-1.06, 0.14] | inc. / inc. | -3.19 to -0.19 |
| | | deep ensemble | -3.34 [-4.71, -2.20] | [-5.57, -1.63] | ref. / ref. | -8.07 to -0.98 |
| | | LSTM | -6.28 [-7.80, -5.12] | [-8.17, -4.58] | ref. / ref. | -16.32 to -4.04 |
| | Human | Diffusion | -0.90 [-1.36, -0.48] | [-1.08, 0.15] | **ref. / inc.** | -1.11 to -0.72 |
| | | GP residual | -2.04 [-3.46, -1.12] | [-3.51, -0.99] | ref. / ref. | -2.46 to -1.79 |
| | all | Diffusion | 0.46 [0.32, 0.56] | [0.10, 0.64] | **supp. / inc.** | 0.43 to 0.48 |

(The full table, every model in every mode, is `artifacts/report/tables/t5_h32_cluster__e3.csv`; compact table c10.)
Verdict changes by the file interval (bold): six of the 60 (variant, mode, model) cells, always towards
"inconclusive": Diffusion on the pooled "all profiles" lines of both pooled variants (supported -> inconclusive, the
human-driven files pull the lower bound to 0.10-0.15), Diffusion on the human-driven platoons of both pooled variants
and QuantilePIDL in two cells (refuted -> inconclusive). The headline per-mode verdicts of H3.2 for ACC platoons do
not change: Diffusion is supported with both pooled trainings (lower file bounds 0.27 and 0.29) and inconclusive with
highD-only training (lower file bounds -0.26 and -0.09; the profile intervals 0.09 and 0.19 already missed 0.20). For
the human-driven platoons three files cannot support a population statement: Diffusion's file intervals end at -0.02
to +0.15 (crossing 0 in three of the four variants) where its profile intervals ended at -0.16 to -0.52; the other
models stay refuted, but "refuted" there is a statement about these three recordings only. The CRPS intervals widen by a median factor of 1.5 (ACC 1.47, human-driven 1.85), the growth-exponent
intervals by 1.6. Seed variability: the five per-seed reductions of Diffusion on ACC lie within 0.23-0.45 (recorded),
0.37-0.52 (held out) and 0.54-0.59 (pooled), i.e. inside the file intervals; QuantilePIDL and the LSTM-based models
vary by an order of magnitude between seeds (runaway seeds).

### R4 for E1 and E2: events of the same follower

Check: the E1 bootstrap (`cf_unc/maps/shift.py`, `analysis.py`) resamples events (all windows of an event together);
the E2 PICP intervals resample the 500 rolled events (one-step conformal: windows grouped by event). Events of the same
follower were therefore treated as independent. This is immaterial for highD (2 503 events of 2 503 followers in the
test fold) and NGSIM (I-80: 7 651 events of 5 521 followers, 1.4 per follower; US-101 7 353 / 5 866), but not for
OpenACC (4 715 events of 49 vehicle-mode followers, 96 per follower; the 500 rolled E2 events come from 44-46
followers, from 10 in the test fold of `e2_pool`). Options added (the point estimates do not change):

* E1: `scripts/analyze_e1.py --unit driver --tables-dir tables_driver` (the driver unit existed; `--tables-dir`
  keeps the event-unit tables), run for `e1`, `e1m`, `e6_lodo_openacc`, `e6_lodo_ngsim_all`; report table
  `t2_h31_follower_clusters`.
* E2: every coverage row has `picp_lo_follower` / `picp_hi_follower` / `n_followers` (follower codes of the events
  table), every one-step conformal row `picp_*_lo_follower` / `*_hi_follower` (driver codes of the windows); report
  table `t4_follower_clusters`.

Width of the 95 % interval with follower clusters relative to the event bootstrap (median over models, horizons and
targets; largest in parentheses):

| Quantity | Root | NGSIM I-80 | OpenACC |
|---|---|---|---|
| E1 shift ratio R epistemic, log-width ratio | e1 / e1m | 1.04 / 0.99 | 1.98 / 1.76 (max 5.3) |
| E1 shift ratio R aleatoric, log-width ratio | e1 / e1m | 1.02 / 0.98 | 2.69 / 1.92 |
| E1 shift ratios, pooled training, set held out | e6_lodo_ngsim_all / e6_lodo_openacc | R epi 1.48, R alea 1.39 (US-101: 1.94, 1.65) | R epi 2.03, R alea 2.14 |
| E1 one-step PICP interval | e1 / e1m | 1.05 / 1.03 | 7.5 / 6.2 (max 10.9) |
| E1 one-step NLL interval | e1 / e1m | 0.98 / 1.00 | 5.2 / 5.7 (max 8.9) |
| E2 rollout PICP, raw (500 events) | e2 / e2h / e2m | 1.01 / 1.00 / 1.00 (1.08) | 1.59 / 1.53 / 1.39 (2.10) |
| E2 rollout PICP, ACI online | e2 / e2h / e2m | 1.01 / 1.00 / 0.99 (1.09) | 1.49 / 1.36 / 1.16 (2.49) |
| E2 rollout PICP, raw / ACI online | e2_pool (OpenACC test fold: 10 followers) | 1.00 / 1.00 | 2.03 / 2.21 (4.18) |
| E2 rollout PICP, raw / ACI online | e2_lodo_ngsim, e2_lodo_ngsim_all / e2_lodo_openacc | 1.00 / 1.00 | 1.21 / 1.18 (2.67) |
| E2 one-step conformal PICP (all windows) | e2 / e2h / e2m | 1.01 / 1.04 / 1.03 | 7.8 / 8.5 / 6.9 (10.2) |

(E1 rows: medians over models and decompositions of the ratio of the interval widths on the log scale for the shift
ratios, on the linear scale for the one-step metrics; a few aleatoric intervals of the event bootstrap are extremely
narrow, which gives isolated ratios up to 240.) The NGSIM intervals do not change with the highD-only training (1.4
events per follower), OpenACC intervals widen by a factor of 2 (shift ratios), 1.2-2.2 (rollout PICP of 500 events)
and 6-8 (one-step intervals over all windows). In the pooled roots the in-domain reference (`pool_test`, the union of
the test folds) contains the OpenACC test fold (701 events of 11 followers), which widens even the NGSIM ratios of
`e6_lodo_ngsim_all` by a factor of 1.4-1.9.

Verdicts with follower clusters: `e1` and `e1m` change 1 of 44 pair verdicts (GP residual, across-seed decomposition,
OpenACC: supported -> inconclusive), and no overall H3.1 or H3.1e verdict; `e6_lodo_openacc` changes 2 of 11 pairs
(deep ensemble, own; Diffusion, across seeds; both OpenACC: refuted -> inconclusive) and thereby their overall H3.1
verdicts (refuted -> inconclusive; H3.1e unchanged); `e6_lodo_ngsim_all` changes 6 of 22 pairs (refuted ->
inconclusive), which turns the H3.1 and the H3.1e verdicts of the deep ensemble (own decomposition, the main H3.1e
verdict) and of the across-seed decompositions of Diffusion, HeteroMDN and the three-component MDN from refuted to
inconclusive. The statement of the E6 report that "ensembles do not flag an unseen highway site once signals are
matched" (H3.1e refuted on the held-out NGSIM sets) therefore holds for the deep ensemble with the event bootstrap
only: its epistemic ratios stay at 0.85-1.29, but the upper bound of the US-101 pair rises from 1.20 to 1.64 (above
the refutation threshold 1.5); MC dropout and the GP residual (own decompositions) stay refuted. Every point estimate
is unchanged. Report tables: `t2_h31_follower_clusters` (E1, 121 rows), `t4_follower_clusters` (E2).

## R9. Collision diagnostics of the platoons

Collision policy of the engine (`cf_unc/rollout/engine.py`, unchanged): a collision is the first step with gap <= 0;
it is flagged per vehicle and rollout (`first_collision`) and the motion continues with a negative gap (the model
input is floored at `s_floor`); nothing is removed or reset, and the following vehicles keep reacting to the
continued, non-physical motion. Amplitudes, standard deviations and CRPS therefore include post-collision motion
unless restricted as below.

`scripts/analyze_e3.py` writes `collision_diagnostics.csv` (per variant, model, profile set and driver mode) from
the unit tables: share of fully collision-free platoon rollouts (no vehicle of the 20 collides over the whole time
line, burn-in included), time to the first collision of the platoon (first collision of any vehicle, over the
colliding rollouts, seconds from the start of the time line; the first 30 s are the constant-speed burn-in) with its
10 % and 90 % quantiles and the share whose first collision falls in the burn-in, the collision shares of vehicles
1 / 5 / 10 / 20 (whole time line; also in `collisions.csv`), and the amplitude CRPS over the vehicles 1-4 over all
rollouts next to the same score over the rollouts in which neither the scored vehicle nor any vehicle ahead of it
collided (`crps_1_4_survivors_biased`; a (profile, vehicle) needs at least 10 such rollouts; the share of rollouts
kept and the number of profiles left are given, as is the all-rollout CRPS on exactly the profiles with a survivor
value, `crps_1_4_same_profiles`). The survivor score is biased by construction (the hard cases are removed) and is a
diagnostic, not a replacement of the CRPS of the design.

Extended set, all 60 profiles (ACC rows in the CSV and in `t6_collision_diagnostics.csv`):

| Variant | Model | collision-free platoons | first collision, s: median [10 %, 90 %] | first collision in burn-in | vehicle 1 / 5 / 10 / 20 | CRPS 1-4 all -> survivors (kept) |
|---|---|---:|---:|---:|---:|---:|
| recorded | IDM + white noise | 100 % | - | 0 | 0 / 0 / 0 / 0 % | 2.91 -> 2.91 (100 %) |
| | Diffusion | 9.8 % | 35 [18, 89] | 36.8 % | 7.7 / 13.8 / 27.4 / 54.1 % | 2.06 -> 1.86 (85 %) |
| | GP residual | 81.1 % | 62 [47, 122] | 0 | 4.2 / 4.6 / 1.7 / 1.9 % | 3.09 -> 3.07 (88 %) |
| | QuantilePIDL | 0.1 % | 22 [16, 45] | 75.2 % | 48.9 / 83.5 / 90.3 / 92.0 % | 6.12 -> 3.79 (22 %) |
| | deep ensemble | 0 | 10 [8, 16] | 99.8 % | 59.9 / 80.1 / 83.2 / 83.8 % | 24.5 -> 4.04 (14 %) |
| | LSTM | < 0.1 % | 35 [21, 49] | 34.7 % | 51.8 / 66.7 / 77.1 / 81.9 % | 57.7 -> 6.08 (21 %) |
| OpenACC held out | IDM + white noise | 100 % | - | 0 | 0 % | 2.97 -> 2.97 |
| | Diffusion | 33.2 % | 62 [22, 141] | 13.0 % | 2.1 / 4.6 / 10.8 / 22.7 % | 1.72 -> 1.73 (95 %) |
| | GP residual | 72.5 % | 49 [38, 120] | 0 | 13.5 / 10.2 / 10.1 / 5.3 % | 2.73 -> 2.64 (81 %) |
| | QuantilePIDL | 3.4 % | 26 [11, 58] | 57.7 % | 21.1 / 59.6 / 64.5 / 69.0 % | 4.15 -> 2.86 (50 %) |
| | deep ensemble | 0 | 10 [8, 15] | 98.4 % | 51.6 / 73.2 / 74.0 / 74.1 % | 38.5 -> 2.82 (17 %) |
| | LSTM | < 0.1 % | 18 [11, 34] | 84.5 % | 59.3 / 66.5 / 68.9 / 71.2 % | 77.9 -> 3.10 (18 %) |
| pooled | IDM + white noise | 100 % | - | 0 | 0 % | 2.96 -> 2.96 |
| | Diffusion | 19.8 % | 60 [26, 132] | 11.9 % | 1.4 / 4.4 / 15.2 / 32.4 % | 1.53 -> 1.54 (97 %) |
| | GP residual | 44.3 % | 106 [39, 124] | 0 | 9.8 / 19.6 / 19.6 / 4.7 % | 2.78 -> 2.73 (71 %) |
| | QuantilePIDL | 0.1 % | 22 [15, 45] | 75.5 % | 20.8 / 74.6 / 77.8 / 77.9 % | 4.37 -> 3.66 (40 %) |
| | deep ensemble | 0 | 12 [9, 18] | 99.7 % | 48.0 / 75.8 / 79.9 / 80.6 % | 15.6 -> 3.42 (20 %) |
| | LSTM | < 0.1 % | 25 [14, 48] | 65.1 % | 45.3 / 62.3 / 68.8 / 72.8 % | 27.5 -> 2.70 (25 %) |

Reading: no learned model gives collision-free platoons in most rollouts. Diffusion keeps the first followers
physical (85-97 % of the rollouts kept at vehicles 1-4, CRPS unchanged by the restriction in the pooled variants) but
only 10-33 % of its 20-vehicle platoon rollouts are free of collisions, with the first collision typically 5-32 s
after the profile starts (median 35 s recorded, 60-62 s with the pooled trainings, counted from the start of the time
line, i.e. 30 s before the profile); the GP residual has the most collision-free platoons among the learned models
(44-81 %). The deep ensemble, QuantilePIDL and the LSTM often collide already during the 30 s burn-in at constant
leader speed (share of the first collisions in the burn-in: 98-100 %, 58-76 % and 35-85 %): failures of the closed
loop at equilibrium, not of the leader profile. Their survivor CRPS (2.7-6.1 m/s)
looks competitive only because 14-50 % of the rollouts survive: the survivor bias is exactly what the reviewer warns
about and the column name says so. The H3.2 CRPS of the design is not changed.

## R11. Growth exponent on a common support

`growth.csv` keeps the fit of the models over the vehicles 1-20 (`beta`, also as `beta_extrapolated`: beyond the
measured vehicles) and adds, for every model and for the measured platoons, `beta_common`: the exponent of the mean
speed standard deviation on the common support of the measured platoons, i.e. the profiles of the set with at least
four measured followers (54 of 60 profiles, 22 files) and the vehicle indices 1-4, window of the measured data, with
a bootstrap interval over profiles (1000) and one over files (10 000, `beta_common_lo_cluster` /
`beta_common_hi_cluster`); every exponent also has a file-cluster interval (`beta_lo_cluster` / `beta_hi_cluster`).
The figure legend of `growth__<variant>.png` now reads "beta_extrapolated (1-20) ..., common support (1-4) ...".

The reviewer's descriptive values are reproduced exactly (54 profiles, indices 1-4): measured 0.0853, Diffusion
recorded 0.1174, held out 0.0473, pooled 0.0962 (and every other model of the evidence file).

| Variant | Model | beta 1-20 (extrapolated) [profiles] | beta common support [profiles] | [files] |
|---|---|---:|---:|---:|
| measured platoons | - | 0.12 [0.08, 0.16] (indices 1-4, ragged: all profiles that measured them) | 0.085 [0.054, 0.119] | [0.026, 0.156] |
| recorded | Diffusion | 0.19 [0.15, 0.24] | 0.117 [0.090, 0.154] | [0.077, 0.203] |
| | IDM + white noise | -0.01 [-0.06, 0.03] | -0.064 [-0.093, -0.031] | [-0.098, 0.003] |
| | GP residual | -0.19 [-0.26, -0.12] | 0.014 [-0.034, 0.051] | [-0.087, 0.064] |
| | QuantilePIDL | -0.05 [-0.09, 0.00] | -0.116 [-0.203, -0.008] | [-0.252, 0.093] |
| | deep ensemble | -0.05 [-0.09, -0.02] | -0.151 [-0.255, -0.044] | [-0.310, 0.045] |
| | LSTM | -0.22 [-0.31, -0.14] | -0.304 [-0.500, -0.126] | [-0.592, -0.003] |
| harmonised | Diffusion | 0.17 [0.12, 0.22] | 0.117 [0.094, 0.148] | [0.084, 0.188] |
| OpenACC held out | Diffusion | 0.10 [0.06, 0.15] | 0.047 [0.032, 0.068] | [0.025, 0.098] |
| | GP residual | -0.25 [-0.32, -0.18] | 0.001 [-0.036, 0.038] | [-0.039, 0.062] |
| | QuantilePIDL | 0.04 [0.02, 0.07] | 0.087 [0.024, 0.165] | [0.012, 0.169] |
| | deep ensemble | 0.02 [0.01, 0.05] | 0.065 [0.015, 0.128] | [0.002, 0.197] |
| pooled | Diffusion | 0.17 [0.14, 0.21] | 0.096 [0.082, 0.115] | [0.075, 0.142] |
| | GP residual | -0.16 [-0.20, -0.11] | -0.005 [-0.034, 0.026] | [-0.034, 0.029] |
| | QuantilePIDL | 0.07 [0.04, 0.10] | 0.179 [0.140, 0.222] | [0.130, 0.257] |
| | LSTM | -0.10 [-0.18, -0.03] | 0.017 [-0.097, 0.141] | [-0.136, 0.233] |

(beta 1-20: window of the measured data, as in compact table c6; the profile + tail window is the one of the summary
and of the figure legend. All rows: `artifacts/report/tables/t6_growth.csv`.) On the common support the measured
exponent is 0.085 rather than 0.12 (the 0.12 of the paper mixes the composition of the ragged platoons), Diffusion's
exponent is close to it in all four variants (recorded 0.117, harmonised 0.117, held out 0.047, pooled 0.096; each of
its four file intervals contains the measured 0.085), and positive exponents on 1-4 are not unique to Diffusion (QuantilePIDL pooled 0.18, held out 0.09; deep ensemble held out 0.07). The exponents over 1-20 are
extrapolations beyond the measured vehicles and are labelled so; none of this is a string-stability statement.

## Consistency items of the review table (analysis side)

### (a) Concentration of the epistemic variance in rare states: two definitions, both in the tables

`rare_concentration.csv` (every E1 root) keeps the quantity of the code and of the H3.1e verdict and adds the
quantity of the Methods next to it:

* ratio of conditional means (columns `ratio_epistemic`, `ratio_epistemic_lo/hi`; the H3.1e criterion "lower bound
  > 1"): `C_ratio = mean(var_epi | rare) / mean(var_epi | not rare)`;
* mass share over window share (new columns `conc_mass_epistemic`, `conc_mass_epistemic_lo/hi`, and the same for the
  aleatoric variance): `C_mass = (sum_rare var_epi / sum var_epi) / (n_rare / n) = mean(var_epi | rare) / mean(var_epi)`,
  with an interval from the same bootstrap draws (events resampled, seed means as for the ratio).

With `p` = share of the windows in the rare states, `C_mass = C_ratio / (p C_ratio + 1 - p)`; both are > 1 exactly
when the variance is higher inside than outside, so the verdict does not depend on the choice, but the numbers do:
deep ensemble on highD test, rule `rare_freq` (bins below 1 % frequency, 34.3 % of the windows): 88.5 % of the
epistemic mass, `C_mass` = 2.58 [2.14, 2.76] against `C_ratio` = 14.7 [5.4, 27.4]; rule `rare_mass` (the 10 % of the
mass in the rarest bins, 10.2 % of the windows): 71.1 % of the mass, `C_mass` = 6.9 [3.8, 8.5] against `C_ratio` =
21.6 [5.6, 53.7]; GP residual (rare_mass) 9.1 [8.4, 9.8] against 115 [84, 154]; MC dropout 3.06 [2.62, 3.46]
against 3.99 [3.22, 4.77]. The report table `t2_h31_summary` carries both columns (H3.1e concentration = the ratio,
"mass share / window share" = C_mass). Definitions to quote: "C_ratio, the ratio of the mean epistemic variance in
the rare states to its mean in the other states (the verdict uses its lower bound > 1); C_mass, the share of the
epistemic variance mass that falls in the rare states divided by the share of the windows there".

### (b) Dependence structure of the one-step tables: events and followers per data set

`one_step_metrics.csv` (every E1 root) has a new column `n_followers` next to `n_events` (`n_events` now counts
events whatever the bootstrap unit; with the default unit it is unchanged); the report table `t1_one_step__*` and the
compact table c1 (bottom row) carry both. E1, target `a` (windows / events / followers):

| data set | windows | events | followers |
|---|---:|---:|---:|
| highD test (fold 0) | 152 683 | 2 503 | 2 503 |
| highD validation | 152 683 | 2 503 | 2 503 |
| NGSIM I-80 | 438 867 | 7 651 | 5 521 |
| NGSIM US-101 | 425 210 | 7 353 | 5 866 |
| Waymo | 158 668 | 967 | 849 |
| OpenACC | 485 932 | 4 715 | 49 |
| NGSIM I-80 (FollowNet cross-check) | 226 086 | 1 937 | 1 916 |

The windows of an event are always resampled together; events of the same follower were resampled as independent
units, which matters only for OpenACC (96 events per follower on average) and, mildly, NGSIM (1.25-1.4): see R4.

### (c) Test suite of this state

Working tree on commit `86f9d70` (the working tree also holds the uncommitted v4 changes of the control experiments: models `idm_ar_noise`, `diffusion_cf_hist`,
`lstm_hetero_frozen`, configs and tests of R2, R3, R10, R13; the counts include their tests). 602 tests collected:

| Selection | Result | Time |
|---|---|---|
| `pytest -q -m "not slow"` (CPU and GPU tests) | 578 passed, 24 deselected | 2 min 54 s |
| `pytest -q -m slow` (trains small models on CPU) | 22 passed, 2 xfailed (documented scientific findings), 577 deselected | 14 min 41 s |
| `pytest -q -m gpu` (RTX 6000 Ada, 36-43 GB free at the time) | 25 passed, 577 deselected | 50 s |
| `pytest -q -m data` | 0 selected (no test needs the raw data sets now) | - |
| `-m "not slow and not gpu and not data"` (the CI selection) | 553 collected (= 578 - 25 GPU tests) | - |

The slow suite ran at 22:05-22:20 on the same working tree (the later v3 edits touched only analysis,
report and test files outside it). For the paper: "602 tests (553 CPU tests of the fast suite, 25 GPU tests, 24 slow
training tests, of which 2 are expected failures that document findings)".

## Regenerated tables

* `scripts/make_report.py` -> `artifacts/report`: 72 tables (was 58), 288 figures (unchanged). New tables:
  `t2_h31_follower_clusters`, `t4_info_budget__<variant>__<target>` (10), `t4_follower_clusters`,
  `t5_h32_cluster__e3`, `t6_collision_diagnostics`; extended: `t1_one_step__*` (`n_followers`), `t2_h31_summary`
  (mass-share concentration), `t4_coverage__*` (shares of infinite ACI intervals, finite widths),
  `t4_h34_verdict_counts` (new decision table), `t6_growth` (common support, file intervals).
* `paper/v3/tools/make_compact.py` -> `paper/v3/tables` (it resolves its paper directory from its own location; new
  option `--out`, default `paper/v3`, refuses `paper/submission`). New compact tables (labels and captions in
  `paper/v3/generated/LABELS.md`):
  * `tab:c10_h32_cluster` (`tables/c10_h32_cluster.tex`): H3.2 per driver mode (ACC, human-driven) and variant
    (recorded, harmonised, OpenACC held out, pooled) for the five learned models, reduction with the profile and the
    file interval and both verdicts;
  * `tab:c11_collisions` (`tables/c11_collisions.tex`): collision diagnostics (collision-free platoons, time to the
    first collision, burn-in share, vehicles 1 and 20, CRPS 1-4 all rollouts vs survivors and kept share) for the
    recorded, the held-out and the pooled variants;
  * `tab:c12_info_budget` (`tables/c12_info_budget.tex`): static, static prequential, ACI random, chronological and
    delayed, and delayed ACI over the first 100 events, at 10 s, highD test and NGSIM I-80 (recorded, target `a`).
  * changed: `tab:c1_onestep` (bottom row: events / followers per data set), `tab:c4_coverage` (column: share of the ACI
    intervals that are the whole line; now full width), `tab:c6_platoon_diag` (exponent over 1-20 and on the common
    support, two lines per cell). `tables/c9_e3pooled.tex` is hand-written and untouched.
  * Every compact table compiles alone with the official class (Tectonic, offline) without overfull or underfull
    boxes.

## Files changed (v3 analysis)

* `cf_unc/rollout/calibration.py`: ACI at the ends (R1), `finite_quantile`, `stream_run`, `delayed_availability`,
  coverage table with the new variants and columns (R1, R4, R8), H3.4 decision table (R7).
* `cf_unc/rollout/streams.py` (new): clocks and chronological order of the rolled events (R8).
* `scripts/analyze_e2.py`: streams, follower intervals, `--finite-quantile`, `--no-streams`, `--tables-dir`,
  `stream_info.csv`, H3.4 rows of the decision table; `scripts/summarize_e2.py`: sections R1, R8, R4.
* `cf_unc/platoon/evaluate.py`: `growth_curve(..., groups=)`, vectorised `growth_exponents`, ACI ends in
  `coverage_by_method`; `scripts/analyze_e3.py`: file-cluster intervals and verdicts, per-seed reductions
  (`profile_index_seed.csv`), collision diagnostics, survivor CRPS, common-support exponents.
* `cf_unc/maps/shift.py` (mass-share concentration with interval), `cf_unc/maps/analysis.py` (columns
  `n_followers`, `conc_mass_*`; `tables_dir`), `scripts/analyze_e1.py` (`--tables-dir`). These two analysis files are
  also receiving the v4 edits (`h33_pairs`); the edits do not overlap.
* `cf_unc/report/figures_e2.py` (finite widths), `cf_unc/report/figures_e3.py` (legend: beta_extrapolated and common
  support), `scripts/make_report.py` (new and extended tables), `paper/v3/tools/make_compact.py` (`--out`, c1, c4, c6,
  c10-c12).
* Tests: `tests/test_calibration.py` (ACI, streams, decision table), `tests/test_e2_runner.py` (end-to-end check of the
  new columns, of the decision table row by row and of `--finite-quantile`), `tests/test_review_fixes_v3.py` (new:
  streams, file clusters, vectorised exponents, mass-share concentration, report tables).
* Artefacts rewritten: `artifacts/e2*/tables` (+ `summary.md`, figures), `artifacts/e3/tables` (+ figures),
  `artifacts/e1*/tables` and `artifacts/e6_*/tables` (same numbers plus the new columns), `tables_driver/` of `e1`,
  `e1m`, `e6_lodo_openacc`, `e6_lodo_ngsim_all`, `artifacts/report`, `paper/v3/{tables,generated,figures}`.
