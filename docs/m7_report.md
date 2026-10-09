# Follow-up E6: training on the matched signals of a pooled domain

Date: 2026-10-08 (fourth root added the same day, section 3.5). Contract: `docs/design.md`, section 16
(decision of the E5 review: train on the matched signals of a pooled domain; 16.4: both NGSIM sets held
out). Question: is the behavioural remainder left by E5 (1-4 nats of
over-confidence and 9-22 % collisions on NGSIM, the state shift of OpenACC, all for models trained
on highD alone) a lack of behavioural diversity in the training domain? Three roots, target `a`,
5 seeds, the ten model configurations of E1, matched signals (windows of E5), the IDM of
`artifacts/e1m` as a frozen prior (16.3): `artifacts/e6_pool` (highD + NGSIM I-80 + US-101 + Waymo
+ OpenACC training folds), `e6_lodo_ngsim` (without NGSIM I-80), `e6_lodo_openacc` (without
OpenACC) and, after the review of the first three, `e6_lodo_ngsim_all` (without both NGSIM sets, the
true NGSIM domain hold-out); rollouts `artifacts/e2_pool`, `e2_lodo_ngsim`, `e2_lodo_openacc`,
`e2_lodo_ngsim_all`. Reference for every comparison: `artifacts/e1m` / `e2m` (the same signals,
highD-only training).

## 1. Commands

```bash
cd cf-uncertainty && export PYTHONUTF8=1
for R in e6_pool e6_lodo_ngsim e6_lodo_openacc e6_lodo_ngsim_all; do
  ./.venv/Scripts/python.exe scripts/run_e1.py --config configs/experiment/$R.yaml --dry-run       # 50 checkpoints to train, IDM reused from artifacts/e1m/idm.json
  bash scripts/run_e1_retry.sh artifacts/$R/logs/r1.out 8 --config configs/experiment/$R.yaml --models idm_white_noise idm_white_noise_mle lstm_const_sigma hetero_mdn hetero_mdn_mix3 mc_dropout gp_residual_idm quantile_pidl
  bash scripts/run_e1_retry.sh artifacts/$R/logs/r2.out 8 --config configs/experiment/$R.yaml --models deep_ensemble
  bash scripts/run_e1_retry.sh artifacts/$R/logs/r3.out 8 --config configs/experiment/$R.yaml --models diffusion_cf
  ./.venv/Scripts/python.exe scripts/analyze_e1.py --root artifacts/$R --config configs/experiment/$R.yaml --in-domain pool_test --quiet
  ./.venv/Scripts/python.exe scripts/analyze_e1_extra.py --root artifacts/$R --config configs/experiment/$R.yaml
  ./.venv/Scripts/python.exe scripts/summarize_e1.py --root artifacts/$R > artifacts/$R/tables/summary.md
  ./.venv/Scripts/python.exe scripts/make_figures_e1.py --root artifacts/$R
done
for R in e2_pool e2_lodo_ngsim e2_lodo_openacc e2_lodo_ngsim_all; do
  bash scripts/run_e2_retry.sh artifacts/$R/logs/run.out 8 --config configs/experiment/$R.yaml
  ./.venv/Scripts/python.exe scripts/analyze_e2.py --root artifacts/$R --quiet
  ./.venv/Scripts/python.exe scripts/summarize_e2.py --root artifacts/$R > artifacts/$R/tables/summary.md
done
./.venv/Scripts/python.exe scripts/make_report.py                                                  # E6 roots added to T1, T2, T4, F2-F4
./.venv/Scripts/python.exe -m pytest -m "not slow"                                                  # 521 passed, 24 deselected
```

Run times on the shared GPU (three-process split per root): `e6_pool` 3.0 h, `e6_lodo_ngsim`
1.0 h, `e6_lodo_openacc` 1.0 h, `e6_lodo_ngsim_all` 1.1 h, the rollout roots 7-12 min each; 400
prediction tables per E1 root (test folds of the training sets, `pool_test`, the held-out sets and
`ngsim_fn` on all events, `highd_val`), 390 / 195 / 195 / 260 rollout tables; 0 failures.

## 2. What was built

* E1 runner: `train.data` may list several data sets (union of their training folds, each capped at
  150 000 one-step windows and 40 000 sequence windows by a seeded uniform subsample; validation
  the same way), the table `pool_test` (row-wise union of the test-fold tables with its own windows
  spec), the list signal key in every resume and reuse check, `train.idm_from` (the frozen IDM of
  another root, copied with provenance and checked); E2 draws the events of a training set from
  its test fold only; `analyze_e1.py --in-domain pool_test` and `--config <experiment config>`;
  `cross_check_of` in the maps config; `make_report.py` sources for the E6 roots. Tests: 521 fast
  (15 new).

## 3. Key numbers (mean over 5 seeds; `artifacts/e6_*/tables/one_step_metrics.csv`)

### 3.1 Pooled root: calibration on every test fold (design.md 16.2, first rule)

NLL on the test fold minus NLL on `pool_test`, and PICP of the 90 % interval on the fold:

| model | highD | NGSIM I-80 | US-101 | Waymo | OpenACC | `pool_test` NLL |
|---|---|---|---|---|---|---|
| lstm_const_sigma | -0.25 / 0.98 | +0.17 / 0.93 | +0.09 / 0.95 | +0.67 / 0.93 | -0.05 / 0.96 | -2.52 |
| hetero_mdn | -0.24 / 0.96 | +0.24 / 0.95 | +0.11 / 0.96 | +0.37 / 0.95 | -0.07 / 0.96 | -2.62 |
| deep_ensemble | -0.29 / 0.98 | +0.30 / 0.96 | +0.18 / 0.97 | +0.35 / 0.96 | -0.10 / 0.97 | -2.84 |
| mc_dropout | -0.19 / 1.00 | +0.10 / 0.99 | +0.19 / 0.99 | +0.21 / 0.99 | -0.03 / 0.99 | -2.17 |
| gp_residual_idm | -0.13 / 0.95 | +0.07 / 0.91 | +0.10 / 0.89 | +0.17 / 0.89 | 0.00 / 0.91 | 0.42 |
| quantile_pidl | -0.35 / 0.92 | +0.40 / 0.87 | +0.20 / 0.89 | +0.70 / 0.90 | -0.25 / 0.93 | -2.50 |
| diffusion_cf | -0.24 / 0.99 | +0.04 / 0.97 | +0.17 / 0.98 | +0.21 / 0.97 | -0.17 / 0.98 | -2.31 |

Every fold is within 1 nat of `pool_test` (largest +0.70, QuantilePIDL on Waymo). The PICP rule
(0.85-0.95) holds for the GP residual and QuantilePIDL on every fold and fails on the upper side
for the others (0.96-1.00): trained on the pooled domain, the networks become over-dispersed, not
over-confident. No fold is under-covered. In-domain RMSE of the networks stays persistence-like (0.01-0.09 m/s^2).

### 3.2 Leave-one-out roots: transfer to the held-out set (second rule)

Transfer gap = NLL(held-out) - NLL(`pool_test`), with the PICP on the held-out set; next to it the
gap of `e1m` (held-out NLL - highD NLL, highD-only training on the same signals):

| model | NGSIM I-80 held out: E6 gap [PICP] | `e1m` gap [PICP] | OpenACC held out: E6 gap [PICP] | `e1m` gap [PICP] |
|---|---|---|---|---|
| lstm_const_sigma | +0.57 [0.88] | +3.49 [0.61] | +58.8 [0.92] | +297 [0.83] |
| hetero_mdn | +0.45 [0.90] | +2.05 [0.76] | +1.00 [0.94] | +1.63 [0.91] |
| deep_ensemble | +0.45 [0.93] | +1.60 [0.88] | +0.33 [0.97] | +0.85 [0.97] |
| mc_dropout | +0.18 [0.98] | +0.93 [0.98] | +0.17 [1.00] | +0.81 [1.00] |
| gp_residual_idm | +0.32 [0.89] | +0.88 [0.82] | 0.00 [0.93] | +0.24 [0.89] |
| quantile_pidl | +0.70 [0.83] | +2.78 [0.74] | -0.11 [0.96] | +0.64 [0.94] |
| diffusion_cf | +0.23 [0.96] | +1.03 [0.91] | +0.18 [0.95] | +1.01 [0.94] |

NGSIM I-80 held out: every gap is below 1 nat (0.2-0.7 against 0.9-3.5 with highD-only training)
and the PICP is 0.83-0.98 (0.85-0.95 for the LSTM, HeteroMDN, the ensemble and the GP residual;
QuantilePIDL 0.83, MC dropout and DiffusionCF above): by the rule, the remainder of E5 on NGSIM was
the narrowness of the training domain. Caveat: US-101 (the same NGSIM programme and processing)
is in the training set of that root, so the held-out I-80 is a weak domain hold-out; `ngsim_fn`
(FollowNet I-80, never trained on) gives the same gaps (0.1-0.5 nats). OpenACC held out: the
heteroscedastic models reach gaps of 0.0-1.0 nats (HeteroMDN exactly 1.0) with PICP 0.93-1.00,
against 0.2-1.6 with highD-only training; the constant-sigma LSTM stays catastrophic (59 nats,
RMSE 0.25 against 0.02 elsewhere: its mean fails on the ACC followers' tails and the constant
sigma cannot absorb it). The state shift of OpenACC is absorbed by the pooled training for every
model with a learned variance.

### 3.5 Fourth root: both NGSIM sets held out (`e6_lodo_ngsim_all`, training on highD + Waymo + OpenACC)

Transfer gap NLL(held-out) - NLL(`pool_test`) with the PICP on the held-out set, three training
domains side by side:

| model | NGSIM I-80: both NGSIM out | US-101 in training | highD only (`e1m`) | US-101: both out | highD only |
|---|---|---|---|---|---|
| lstm_const_sigma | +0.99 [0.78] | +0.57 [0.88] | +3.49 [0.61] | +0.66 [0.80] | +2.68 [0.65] |
| hetero_mdn | +1.32 [0.75] | +0.45 [0.90] | +2.05 [0.76] | +1.00 [0.74] | +1.54 [0.76] |
| deep_ensemble | +0.96 [0.81] | +0.45 [0.93] | +1.60 [0.88] | +0.72 [0.81] | +1.19 [0.88] |
| mc_dropout | +0.55 [0.90] | +0.18 [0.98] | +0.93 [0.98] | +0.46 [0.92] | +0.81 [0.99] |
| gp_residual_idm | +0.54 [0.86] | +0.32 [0.89] | +0.88 [0.82] | +0.26 [0.85] | +0.55 [0.80] |
| quantile_pidl | +1.88 [0.70] | +0.70 [0.83] | +2.78 [0.74] | +1.58 [0.69] | +1.90 [0.73] |
| diffusion_cf | +0.92 [0.83] | +0.23 [0.96] | +1.03 [0.91] | +0.82 [0.83] | +0.86 [0.93] |

With no NGSIM data in training the gap on NGSIM I-80 is 0.5-1.9 nats (median about 1) with PICP
0.70-0.90, between the two earlier columns: training on highD, Waymo and OpenACC removes a third
to two thirds of the highD-only gap (LSTM 3.5 -> 1.0, HeteroMDN 2.1 -> 1.3, ensemble 1.6 -> 1.0,
QuantilePIDL 2.8 -> 1.9), DiffusionCF and the GP residual gain little (1.0 -> 0.9, 0.9 -> 0.5),
and the rest, 0.5-1.9 nats, is specific to the NGSIM programme (the never-trained FollowNet I-80
set agrees: 0.3-1.5 nats). By the reading of 16.2 this is the "in between" case: part of the
remainder of E5 was the narrowness of the training domain, part is NGSIM itself. The epistemic
ratios on the two unseen NGSIM sets are 0.5-1.6 for every model and the aleatoric ratios 0.6-1.9:
the ensembles do not flag an unseen site of the same kind (highway following, matched signals)
as novel; H3.1 and H3.1e are refuted on these pairs. Rollouts on the held-out sets: 1 s coverage
0.67-0.99, collisions by 10 s 5-13 % for the LSTM-based models and MC dropout (6.8 % LSTM, 12 %
HeteroMDN, 5 % ensemble, 13 % MC dropout), 1.2 % DiffusionCF, 0 for the IDM-backed models; ACI
online 0.89-0.94 with 12-23 m intervals at 10 s; H3.4 33 of 39 supported, 5 inconclusive, 1
premise failure.

### 3.3 H3.1, H3.1e, H3.3 on the pooled domain

* Held-out NGSIM I-80 (root `e6_lodo_ngsim`): the epistemic variance does not grow (ratios 0.4-2.4
  for every model) and the aleatoric ratios are 0.5-3.1; H3.1 and H3.1e are refuted or inconclusive
  for every model on this single pair. Held-out OpenACC: the epistemic ratios stay large (deep
  ensemble 137, MC dropout 5-350, LSTM seed ensemble 196, DiffusionCF 18, GP residual 2-8) with
  aleatoric ratios 1-40; with one counted pair the verdicts are inconclusive by construction
  (two pairs needed). The epistemic signal of a state shift survives pooled training; that of a
  processing shift does not, because there is none left.
* H3.3 on `pool_test`: HeteroMDN against LSTMConstSigma -0.097 [-0.129, -0.073] nats (pooled root;
  -0.122 and -0.144 in the LODO roots), -0.080 on the highD fold, -0.028 [-0.053, -0.007] on the
  NGSIM fold, -0.108 on OpenACC: the regime gain shrinks from 0.22 nats (highD-only) to 0.1 but
  stays significant everywhere.

### 3.4 Rollouts (E2) on the held-out sets and the test folds, target `a`

| held-out set, model | raw PICP 1 / 10 s: `e2m` -> E6 | collided by 10 s: `e2m` -> E6 | ACI width at 10 s, m: `e2m` -> E6 |
|---|---|---|---|
| NGSIM, lstm_const_sigma | 0.55 / 0.23 -> 0.84 / 0.39 | 0.089 -> 0.084 | 27 -> 23 |
| NGSIM, hetero_mdn | 0.51 / 0.19 -> 0.82 / 0.42 | 0.215 -> 0.068 | 40 -> 22 |
| NGSIM, deep_ensemble | 0.82 / 0.40 -> 0.92 / 0.53 | 0.122 -> 0.078 | 22 -> 16 |
| NGSIM, mc_dropout | 0.94 / 0.59 -> 0.94 / 0.65 | 0.179 -> 0.245 | 28 -> 31 |
| NGSIM, quantile_pidl | 0.52 / 0.41 -> 0.71 / 0.32 | 0.015 -> 0.044 | 32 -> 22 |
| NGSIM, diffusion_cf | 0.99 / 0.90 -> 0.99 / 0.89 | 0.015 -> 0.008 | 12 -> 11 |
| OpenACC, lstm_const_sigma | 0.47 / 0.26 -> 0.65 / 0.46 | 0.041 -> 0.082 | 43 -> 39 |
| OpenACC, hetero_mdn | 0.43 / 0.16 -> 0.64 / 0.48 | 0.097 -> 0.072 | 45 -> 35 |
| OpenACC, deep_ensemble | 0.64 / 0.52 -> 0.69 / 0.56 | 0.107 -> 0.092 | 26 -> 24 |
| OpenACC, diffusion_cf | 0.87 / 0.95 -> 0.87 / 0.95 | 0.004 -> 0.000 | 9 -> 10 |

The 1 s coverage on held-out NGSIM rises to the in-domain level or above (0.84-0.99) and the
10 s coverage to 0.32-0.89; collisions fall for HeteroMDN (0.22 -> 0.07) and the ensemble (0.12 ->
0.08) and stay at 8 % for the LSTM and 25 % for MC dropout (its per-step draws of 50 masks are the
widest and the least stable). On held-out OpenACC the picture moves little (collisions 2-9 %).
ACI online keeps 0.88-0.94 at every horizon on every data set and root. H3.4 by the point rule:
54 of 65 (pooled root), 22 of 26 (each LODO root) supported; every "refuted" case is a model
whose raw intervals already cover 0.88-1.00 (DiffusionCF, kept-mask MC dropout), i.e. the
premise of H3.4 failing, not the correction.

## 4. Findings

1. **The behavioural remainder of E5 was largely the narrowness of the training domain.** Trained
   on four data sets with matched signals, the models transfer to the held-out fifth with 0.2-1.0
   nats of extra NLL and 0.83-0.98 coverage (NGSIM I-80 with US-101 in training, and OpenACC),
   against 1-4 nats and 0.6-0.9 with highD-only training. With both NGSIM sets unseen (3.5) the
   gap is 0.5-1.9 nats at 0.70-0.90 coverage: a third to two thirds of the highD-only gap is
   domain narrowness, the rest is the NGSIM programme itself.
2. **The constant-sigma LSTM does not transfer to OpenACC whatever it is trained on** (59 nats held
   out, 297 with highD-only training): a homoscedastic law cannot carry the tails of the ACC
   followers; every heteroscedastic model can.
3. **Pooled training trades over-confidence for over-dispersion**: on its own test folds the pooled
   model covers 0.93-1.00 with the networks and 0.87-0.93 with the IDM-backed models; no fold is
   under-covered. The regime gain of H3.3 shrinks to 0.1 nats and stays significant.
4. **Epistemic growth marks state shift, not a new site of the same kind.** On held-out OpenACC the
   epistemic ratios stay in the tens to hundreds; on held-out NGSIM they are 0.4-2.4 with US-101 in
   training and 0.5-1.6 with both NGSIM sets unseen (H3.1e refuted there): the ensembles do not
   detect an unseen highway site once the signals are matched, although it still costs about
   1 nat of NLL.
5. **Rollout stability is the one thing pooled training does not fix**: 8 % (LSTM) to 25 % (MC
   dropout) of the NGSIM rollouts still collide within 10 s, against 0.1-1 % for DiffusionCF and 0
   for the IDM-backed models; the closed-loop instability of the one-step networks is a property
   of the models, not of the data they saw.

## 5. Deviations from the design

* The pooled closed-loop IDM calibration is degenerate (parameters at the bounds, one-step RMSE
  2.45 m/s^2); the E6 roots reuse the IDM of `artifacts/e1m` as a frozen prior (`train.idm_from`,
  design.md 16.3), so only the learned parts see the pooled domain.
* `highd_val` is evaluated as well (calibration set of the E2 conformal step and placebo pair);
  `pool_test` is a row-wise union; inherited config entries can be removed with null; the E2
  runner refuses non-test splits of a training set.
* The NGSIM I-80 hold-out keeps US-101 in training (same programme and processing): a weak domain
  hold-out, reported as such; the fourth root (16.4) holds both NGSIM sets out and is the true
  domain hold-out; `ngsim_fn` (never trained on) agrees with both.
* GPU: one short period of the pilot went below the 6 GB rule agreed with the sibling
  project (41 GB held by its runs); the full runs waited for >= 6 GB free before starting.

## 6. Open points

1. For the paper: the E5/E6 pair separates the three parts of "dataset shift" in car-following
   (processing roughness: hundreds of nats; training-domain narrowness: 1-4 nats; state shift such
   as ACC following at low speed: epistemic ratios in the hundreds, which pooled training absorbs
   for heteroscedastic models). The M2/M3 verdicts stand for highD-only models on signals as
   recorded; the pooled-domain numbers are the ones to quote for transfer.
2. Done (3.5): with both NGSIM sets unseen the transfer costs about 1 nat; the ensembles do not
   flag the unseen site. Whether a site-level epistemic signal can be recovered (e.g. by feature-
   level density models) is a question for the paper's discussion, not for this pipeline.
3. The closed-loop instability of the one-step networks (8-25 % collisions) is unchanged by the
   training domain; it belongs to M4's platoon finding, not to the shift story.
