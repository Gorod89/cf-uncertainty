# Follow-up E5: roughness-matched signals

Date: 2026-10-07. Contract: `docs/design.md`, section 15 (decision of the M5 review: run the
experiment with a filter that equalises the roughness of the data sets). Question: is the
over-confidence of the history-based models under dataset shift explained by the roughness of
the shifted signals (the NGSIM reconstructions stay 10x rougher than highD after one Savitzky-Golay
window, M4 report 3.6)? No model was retrained: the checkpoints of `artifacts/e1h` /
`e1h_mean1s` (harmonised highD, window 11) were evaluated on every data set smoothed to highD's
roughness, and the rollouts of E2 were repeated. Roots: `artifacts/e1m`, `e1m_mean1s`, `e2m`.

## 1. Commands

```bash
cd cf-uncertainty && export PYTHONUTF8=1
./.venv/Scripts/python.exe scripts/match_roughness.py --config configs/experiment/e1m.yaml   # artifacts/e1m/matching.json (72 s, CPU)
./.venv/Scripts/python.exe scripts/run_e1.py --config configs/experiment/e1m.yaml --dry-run    # 50 checkpoints reused, 0 to train
bash scripts/run_e1_retry.sh artifacts/e1m/logs/run.out 8 --config configs/experiment/e1m.yaml                 # 43 min
bash scripts/run_e1_retry.sh artifacts/e1m_mean1s/logs/run.out 8 --config configs/experiment/e1m_mean1s.yaml   # 44 min, in parallel
for R in e1m e1m_mean1s; do
  ./.venv/Scripts/python.exe scripts/analyze_e1.py --root artifacts/$R --config configs/experiment/$R.yaml --quiet
  ./.venv/Scripts/python.exe scripts/analyze_e1_extra.py --root artifacts/$R --config configs/experiment/$R.yaml
  ./.venv/Scripts/python.exe scripts/summarize_e1.py --root artifacts/$R > artifacts/$R/tables/summary.md
  ./.venv/Scripts/python.exe scripts/make_figures_e1.py --root artifacts/$R
done
bash scripts/run_e2_retry.sh artifacts/e2m/logs/run.out 8 --config configs/experiment/e2m.yaml   # 780 tables, 21 min
./.venv/Scripts/python.exe scripts/analyze_e2.py --root artifacts/e2m --quiet
./.venv/Scripts/python.exe scripts/summarize_e2.py --root artifacts/e2m > artifacts/e2m/tables/summary.md
./.venv/Scripts/python.exe scripts/make_report.py                                                # T1, T2, T4, F2-F4 gain the matched roots
./.venv/Scripts/python.exe -m pytest -m "not slow"                                                # 506 passed, 24 deselected
```

## 2. What was built

* Signal variant `matched` (`cf_unc/data/harmonise.py`, `pipeline.py`, `experiments/e1.py`,
  `experiments/e2.py`): the harmonised processing with a Savitzky-Golay window per data set, chosen
  by `scripts/match_roughness.py` so that the roughness of the smoothed follower speed (rms of its
  second difference) is within 10 % of the reference (harmonised highD, window 11); per-data-set
  signal keys, so the reference key equals the harmonised key and the `e1h` checkpoints are reused
  (the in-domain tables came out bit-identical to `e1h`); events shorter than their window are
  dropped (`short_event`, none here). Configs `e1m.yaml`, `e1m_mean1s.yaml`, `e2m.yaml`; E2 gained
  `train: {data: highd}`. Tests: 506 fast (11 new).

## 3. Key numbers

### 3.1 Windows and task difficulty (`artifacts/e1m/matching.json`)

| data set | window (s) | speed roughness, x highD: window 11 -> chosen | flag | persistence RMSE of `a`, m/s^2 (highD 0.017) | of `mean_1s` (highD 0.157) |
|---|---|---|---|---|---|
| NGSIM I-80 | 101 (10.1) | 10.4 -> 1.5 | not reached | 0.179 -> 0.024 | 1.07 -> 0.20 |
| NGSIM US-101 | 101 | 8.5 -> 1.5 | not reached | 0.150 -> 0.025 | 0.91 -> 0.21 |
| NGSIM I-80 (FollowNet) | 101 | 7.9 -> 1.4 | not reached | 0.135 -> 0.024 | 0.85 -> 0.19 |
| Waymo | 101 | 3.0 -> 1.9 | not reached (flat from window 61) | 0.052 -> 0.026 | 0.40 -> 0.17 |
| OpenACC | 61 (6.1) | 2.6 -> 1.06 | - | 0.038 -> 0.018 | 0.23 -> 0.17 |

The 10 % tolerance is reached only by OpenACC; the NGSIM sets stop at 1.4-1.5x and Waymo at 1.9x
with the largest window of the grid. The persistence RMSE of the matched targets is 1.05-1.5x
highD's, so the one-step task on the matched shifted sets is about as hard as in-domain (the
condition of design.md 15.2). Drop shares: NGSIM I-80 0.3 -> 2.3 % (more `min_gap` violations at
window 101), OpenACC 6.6 %, the others < 1 %.

### 3.2 One-step metrics under shift: recorded -> harmonised -> matched (mean over 5 seeds)

Target `a`, NGSIM I-80 (in-domain highD values in the last column; `e1`, `e1h`, `e1m`):

| model | NLL | PICP of the 90 % interval | RMSE, m/s^2 | NLL highD |
|---|---|---|---|---|
| lstm_const_sigma | 467 -> 397 -> 0.63 | 0.08 -> 0.09 -> 0.61 | 0.47 -> 0.40 -> 0.04 | -2.86 |
| hetero_mdn | 82 -> 86 -> -1.03 | 0.19 -> 0.19 -> 0.76 | 0.48 -> 0.43 -> 0.06 | -3.08 |
| deep_ensemble | 41 -> 38 -> -1.48 | 0.33 -> 0.34 -> 0.88 | 0.47 -> 0.46 -> 0.06 | -3.08 |
| mc_dropout | 4.0 -> 3.4 -> -1.50 | 0.52 -> 0.55 -> 0.98 | 0.47 -> 0.44 -> 0.05 | -2.43 |
| gp_residual_idm | 5.8 -> 5.6 -> 1.10 | 0.54 -> 0.54 -> 0.82 | 1.12 -> 1.11 -> 0.52 | 0.22 |
| quantile_pidl | 102 -> 93 -> -0.13 | 0.16 -> 0.21 -> 0.74 | 0.41 -> 0.57 -> 0.17 | -2.92 |
| diffusion_cf | 16 -> 15 -> -1.45 | 0.42 -> 0.43 -> 0.91 | 0.55 -> 0.49 -> 0.06 | -2.48 |
| idm_white_noise | 1.9 -> 1.9 -> 1.24 | 0.89 -> 0.89 -> 0.98 | 1.42 -> 1.43 -> 0.75 | 1.44 |

US-101 and Waymo behave the same (NLL -1.9 to 2.6 matched, PICP 0.65-0.99); OpenACC improves
for every model but the LSTM, whose NLL stays at 294 (PICP 0.83: a few windows with huge
standardised errors). Behavioural target `mean_1s`, NGSIM I-80, harmonised -> matched: LSTM NLL
383 -> 1.85 (in-domain -1.72), PICP 0.09 -> 0.57; HeteroMDN 80 -> 0.16 (-1.81), 0.17 -> 0.75;
deep ensemble 33 -> -0.45 (-1.84), 0.32 -> 0.82; MC dropout 10 -> -0.70 (-1.81), 0.35 -> 0.88;
QuantilePIDL 151 -> 4.29 (-1.81), 0.16 -> 0.70; DiffusionCF 36 -> -0.30 (-1.70), 0.28 -> 0.79; GP
residual 3.9 -> 0.97 (0.15), 0.60 -> 0.84.

Reading by the rule fixed in design.md 15.2: no model stays more than 5 nats above its in-domain
NLL on the matched NGSIM sets (target `a`: 0.9-3.5 nats above; `mean_1s`: 0.8-3.6, QuantilePIDL
6.1), MC dropout, the GP residual and DiffusionCF are within 1 nat for target `a`, and the PICP
returns to 0.61-0.98 (`mean_1s`: 0.57-0.88). The over-confidence under shift is therefore mostly
a roughness artefact of the source pipelines, with a behavioural remainder of 1-4 nats (the
constant-sigma LSTM and the quantile model keep the largest remainder, the models with an explicit
epistemic term the smallest).

### 3.3 H3.1, H3.1e, H3.3 on the matched signals

* H3.1 as written stays refuted for every model that learns its aleatoric variance, but the
  margins shrink: on NGSIM I-80 the aleatoric ratio of the deep ensemble drops from 8.6 (recorded)
  to 2.8 (matched), of HeteroMDN from 2.9 to 2.7, of MC dropout from 48 to 0.3 (below 1: the
  smoothed NGSIM looks *less* noisy than highD to it); on OpenACC the ratios stay at 85-1070x.
  The epistemic ratios drop from 55-107x to 3-8x on the NGSIM sets (deep ensemble 5.3, DiffusionCF
  6.0, GP residual 1.7-3.4) and stay at 150-750x on OpenACC, where the shift is in the state
  (low speeds, ACC followers), not in the roughness. H3.1e stays supported for every model.
* H3.3 is unchanged (the in-domain tables are the `e1h` ones): gain -0.221 nats for target `a`,
  -0.093 for `mean_1s`.

### 3.4 Rollouts (E2) on the matched signals, target `a` (`e2h` -> `e2m`)

| NGSIM I-80 | raw PICP at 1 / 10 s | collided by 10 s | ACI width at 10 s, m | highD in-domain: raw PICP 1 / 10 s, collided |
|---|---|---|---|---|
| lstm_const_sigma | 0.03 / 0.08 -> 0.55 / 0.23 | 0.228 -> 0.089 | 188 -> 27 | 0.57 / 0.38, 0.006 |
| hetero_mdn | 0.05 / 0.13 -> 0.51 / 0.19 | 0.256 -> 0.215 | 98 -> 40 | 0.51 / 0.27, 0.020 |
| deep_ensemble | 0.13 / 0.29 -> 0.82 / 0.40 | 0.279 -> 0.122 | 65 -> 22 | 0.64 / 0.49, 0.012 |
| mc_dropout | 0.32 / 0.65 -> 0.94 / 0.59 | 0.183 -> 0.179 | 83 -> 28 | 0.89 / 0.65, 0.022 |
| gp_residual_idm | 0.20 / 0.16 -> 0.35 / 0.19 | 0 -> 0 | 22 -> 16 | 0.59 / 0.26, 0.006 |
| quantile_pidl | 0.10 / 0.29 -> 0.52 / 0.41 | 0.014 -> 0.015 | 108 -> 32 | 0.52 / 0.36, 0.004 |
| diffusion_cf | 0.33 / 0.76 -> 0.99 / 0.90 | 0.102 -> 0.015 | 31 -> 12 | 0.96 / 0.89, 0.001 |
| idm_white_noise | 0.49 / 0.42 -> 0.80 / 0.46 | 0 -> 0 | 19 -> 13 | 0.78 / 0.48, 0 |

The 1 s coverage under shift returns to the in-domain level for every model, the ACI intervals at
10 s shrink to the in-domain widths (LSTM 188 -> 27 m against 29 m on highD), and the collisions
of the LSTM-based models fall from 18-28 % to 9-21 % (in-domain 0.6-2 %): the rollout picture of
M3 was dominated by the roughness as well, and a behavioural remainder stays (HeteroMDN and MC
dropout still collide in about a fifth of the NGSIM rollouts; 10 s coverage 0.19-0.59 against
0.27-0.65 in-domain). ACI online keeps 0.89-0.93 everywhere. H3.4 by the point rule: 114 of 130
supported, 13 "refuted" because the raw intervals of DiffusionCF and of the kept-mask MC dropout
already cover 0.88-0.97 on the matched sets (the premise of H3.4 fails for them, the correction
keeps 0.89-0.91), 3 inconclusive.

## 4. Findings

1. **Most of the over-confidence under shift was signal roughness.** Smoothing the shifted sets to
   the training set's roughness (windows of 6-10 s) takes the one-step NLL of the history-based
   models on NGSIM from 40-470 nats to within 1-4 nats of their in-domain values and their 90 %
   coverage from 0.08-0.55 to 0.6-0.98, without retraining; the 1 s rollout coverage and the ACI
   widths return to in-domain values.
2. **A behavioural shift remains, and it is small at the one-step scale**: 1-4 nats (6 for the
   quantile model on the behavioural target), coverage 0.6-0.8 for the constant-sigma LSTM and the
   quantile model, collisions in 9-21 % of the NGSIM rollouts against 0.6-2 % in-domain. The
   models with an explicit epistemic term (deep ensemble, MC dropout, GP residual, DiffusionCF)
   are the closest to calibrated.
3. **OpenACC is a different kind of shift**: its roughness was within 2.6x of highD to begin with,
   the matched window changes its numbers little for the LSTM (NLL 294), and the epistemic and
   aleatoric ratios stay in the hundreds. The ACC followers at low speed are a shift of the
   behaviour and of the state, not of the processing.
4. **The matching is a smoothing of the task as much as of the inputs**: the persistence RMSE of
   the NGSIM target falls 7x (0.18 -> 0.024 m/s^2), to 1.4x highD's. The experiment shows that
   models trained on smooth signals are calibrated on smooth signals; it does not say that the
   smoothed NGSIM is the true driving signal. A paper that uses these models on reconstructed
   trajectories must either match the processing of its training data or report the ACI
   correction, which handles both cases.

## 5. Deviations from the design

* The tolerance of 10 % is reached only by OpenACC; NGSIM and Waymo take the largest grid window
  (101 samples) and are flagged (design.md 15.3). A longer grid would smooth the NGSIM targets
  below highD's persistence RMSE and is not pursued.
* `short_event` applies to every harmonised window (no event is affected); E2 gained
  `train: {data: highd}`; one matching file serves the three roots; the in-domain tables were
  recomputed and equal `e1h`'s bit for bit.
* E3 was not repeated (design.md 15.1).

## 6. Open points

1. The remainder of 1-4 nats and the 9-21 % collisions under shift are the behavioural shift the
   paper can claim; the M2/M3 numbers (hundreds of nats) should be presented as the effect of the
   source pipelines, with the matched numbers as the headline for the models' transfer.
2. Whether to retrain on the matched signals of every data set (a pooled training domain) is a
   new experiment, not a rerun.
