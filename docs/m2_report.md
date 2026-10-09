# Milestone M2 report: uncertainty maps, dataset shift, regimes (experiment E1)

Date: 2026-09-28. Environment: shared `cf-stability/.venv` (Python 3.11.8, torch 2.11.0+cu128,
GPyTorch 1.15.2), GPU RTX 6000 Ada shared with other jobs.
All tables: `artifacts/e1/tables/` (`summary.md` collects them), figures: `artifacts/e1/figures/`.

## 1. Commands

```bash
export PYTHONUTF8=1
.venv/Scripts/python.exe scripts/import_cf_stability_events.py --force          # event sets, fold files, IDM reference
.venv/Scripts/python.exe -m pytest -m "not slow" -p no:cacheprovider            # 334 passed, 45 s
.venv/Scripts/python.exe -m pytest -m gpu -p no:cacheprovider                   # 10 passed, 17 s
.venv/Scripts/python.exe scripts/run_e1.py --pilot                              # 63 units, 10 min
.venv/Scripts/python.exe scripts/select_w_phys.py --root artifacts/e1 --grid 0 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3
bash scripts/run_e1_retry.sh artifacts/e1/logs/run.out 8                        # full E1, resumable, about 2 h on 3 processes
.venv/Scripts/python.exe scripts/analyze_e1.py --root artifacts/e1              # 2 min 49 s
.venv/Scripts/python.exe scripts/analyze_e1_extra.py --root artifacts/e1        # 1 min 18 s
.venv/Scripts/python.exe scripts/summarize_e1.py --root artifacts/e1 > artifacts/e1/tables/summary.md
.venv/Scripts/python.exe scripts/make_figures_e1.py --root artifacts/e1         # 34 figures
```

## 2. What was run

| Item | Value |
|---|---|
| Models | 10 configurations (the 8 of the specification, the 3-component mixture, a second IDM baseline) x seeds 0-4 |
| Training data | FollowNet HighD events, fold 0 of cf-stability: 7 506 training events, 457 866 windows (stride 2) |
| Evaluation | highD test 152 683 windows / 2 503 events; highD validation (placebo) 152 683 / 2 503; NGSIM I-80 reconstructed 438 867 / 7 651; NGSIM I-80 FollowNet (cross-check) 226 086 / 1 937; Waymo pairs 158 668 / 967; OpenACC 485 932 / 4 715 |
| Output | 300 prediction tables (2.3 GB), 50 checkpoints |
| Training | every LSTM model used all 40 epochs (no early stop); GP 12-22 epochs; DiffusionCF 12 epochs |
| IDM | closed-loop: v0 28.9, T 0.615, s0 2.29, a 0.375, b 0.454 (cf-stability: 29.3, 0.668, 1.99, 0.352, 0.500); one-step ML: 26.2, 0.30, 3.66, 0.30, 5.0 (three parameters on a bound) |
| QuantilePIDL | `w_phys = 0.0001` selected on the validation fold (pinball 0.00273; no physics 0.00319; default 0.3 of the plan 0.0631) |

## 3. Key numbers

### 3.1 The one-step task is dominated by the smoothness of the signal

RMSE of a model-free prediction of `a[t]`, m/s^2:

| Data set | std of a | "a = 0" | persistence `a[t-1]` | closed-loop IDM | one-step ML IDM |
|---|---|---|---|---|---|
| highD test | 0.352 | 0.353 | 0.017 | 0.98 | 0.36 |
| NGSIM I-80 | 0.943 | 0.943 | 0.179 | 1.42 | 1.19 |
| NGSIM I-80 (FollowNet) | 0.812 | 0.812 | 0.135 | 1.18 | 0.85 |
| Waymo | 0.657 | 0.657 | 0.052 | 0.91 | 0.70 |
| OpenACC | 0.454 | 0.454 | 0.040 | 1.00 | 0.55 |

No IDM is better than predicting zero; persistence is 20 times better. The LSTM
models reach 0.013-0.027 on highD, i.e. they are persistence-type predictors.

### 3.2 One-step quality (mean over 5 seeds)

| Model | NLL highD | NLL NGSIM | NLL Waymo | NLL OpenACC | PICP highD | PICP NGSIM | PICP Waymo | PICP OpenACC |
|---|---|---|---|---|---|---|---|---|
| idm_white_noise | 1.41 | 1.90 | 1.34 | 1.42 | 0.945 | 0.888 | 0.962 | 0.957 |
| idm_white_noise_mle | 0.39 | 4.91 | 1.65 | 0.99 | 0.930 | 0.623 | 0.731 | 0.841 |
| lstm_const_sigma | -2.78 | 467 | 27.7 | 279 | 0.925 | 0.081 | 0.306 | 0.465 |
| hetero_mdn | -2.93 | 82.5 | 6.39 | 6.76 | 0.908 | 0.187 | 0.502 | 0.644 |
| hetero_mdn_mix3 | -3.01 | 5.72 | 0.26 | -0.08 | 0.909 | 0.206 | 0.510 | 0.619 |
| deep_ensemble | -2.97 | 40.7 | 2.79 | 2.79 | 0.945 | 0.328 | 0.622 | 0.730 |
| mc_dropout | -2.40 | 3.97 | -0.71 | -0.69 | 0.996 | 0.523 | 0.860 | 0.882 |
| gp_residual_idm | 0.22 | 5.75 | 1.52 | 0.66 | 0.928 | 0.542 | 0.715 | 0.865 |
| quantile_pidl | -2.94 | 102 | 17.5 | 10.5 | 0.907 | 0.163 | 0.428 | 0.542 |
| diffusion_cf | -2.43 | 16.3 | 0.11 | 0.42 | 0.988 | 0.417 | 0.743 | 0.766 |

In-domain every model is within 0.05 of the nominal coverage 0.90 except
MCDropout (0.996) and DiffusionCF (0.988), which are over-dispersed. Under shift
every history-based model under-covers (PICP 0.08-0.52 on NGSIM); the median
squared z-score is 2.3-195 against 0.45 for a calibrated law
(`extra_calibration.csv`).

### 3.3 Hypothesis H3.1

**(a) Concentration in rare states, highD test** (share of windows / share of the summed epistemic variance; ratio inside / outside with 95 % interval):

| Model | frequency < 1 % (plan) | lowest 10 % of mass | rare in (s, dv, v), 10 % | `abs(dv) > 3` | `s < 15 m` |
|---|---|---|---|---|---|
| deep_ensemble | 0.34 / 0.88; 14.7 [5.4, 27.4] | 0.10 / 0.71; 21.6 [5.6, 53.7] | 0.10 / 0.86; 54 [20, 105] | 27.5 [8.7, 45.9] | 0.47 [0.25, 1.2] |
| mc_dropout | 0.34 / 0.58; 2.7 [2.4, 3.0] | 0.10 / 0.31; 4.0 [3.2, 4.8] | 0.10 / 0.33; 4.5 [3.8, 5.4] | 8.6 [6.0, 11.7] | 1.9 [1.5, 2.3] |
| gp_residual_idm | 0.34 / 0.98; 84 [60, 116] | 0.10 / 0.93; 115 [84, 154] | 0.10 / 0.94; 136 [101, 177] | 55 [34, 90] | 0.18 [0.12, 0.27] |

**(b) Ratios shifted / in-domain** (event bootstrap, 1000 replicates; geometric mean in brackets is the robust version):

| Model | Quantity | NGSIM | Waymo | OpenACC |
|---|---|---|---|---|
| deep_ensemble | epistemic | 71 [41, 154] (238) | 24 [12, 59] (30) | 806 [470, 1850] (21) |
| deep_ensemble | aleatoric | 8.6 [5.1, 17.3] (9.2) | 13.4 [3.9, 38.6] (5.9) | 1140 [683, 2390] (7.0) |
| mc_dropout | epistemic | 16.6 [15.1, 18.3] (18.6) | 5.4 [4.8, 6.2] (7.6) | 49.5 [43.1, 57.4] (3.6) |
| mc_dropout | aleatoric | 48 [36, 72] (13.3) | 11.4 [7.6, 17.6] (6.3) | 135 [98, 210] (7.1) |
| gp_residual_idm | epistemic | 3.8 [2.9, 5.1] (9.2) | 7.6 [5.3, 11.4] (8.1) | 11.1 [8.4, 15.7] (4.5) |
| gp_residual_idm | aleatoric | 1 by construction | 1 | 1 |

Placebo (highD validation against test): 1.08 [0.40, 3.09], 1.10 [0.96, 1.26],
1.29 [0.83, 2.03] for the epistemic ratios of the three models; geometric
1.03-1.07. The wide intervals of the ratio of means come from a heavy tail: a
few windows hold most of the variance.

**(c) Verdict with the thresholds of the plan**

| Model | Pairs supported / refuted / inconclusive | Overall |
|---|---|---|
| deep_ensemble | 0 / 3 / 0 | refuted (aleatoric changes by more than 2x on every pair) |
| mc_dropout | 0 / 3 / 0 | refuted (same reason) |
| hetero_mdn, hetero_mdn_mix3, quantile_pidl, diffusion_cf (across-seed decomposition) | 0 / 1-3 / 0-2 | refuted |
| gp_residual_idm | 2 / 0 / 1 | supported, but its aleatoric variance cannot change by construction |
| lstm_const_sigma (across-seed decomposition) | 3 / 0 / 0 | supported, same remark |

H3.1 as written is **refuted**: the epistemic part holds everywhere (growth by
3.8x to several hundred times, concentration in rare states and at large
`abs(dv)`, but not at small gaps), the aleatoric part fails for every model
whose aleatoric variance is learned (growth 3-17x by geometric mean, up to
1000x by mean).

**(d) Where the shift comes from.** At matched (s, dv, v) cells the GP ratio
drops to 1.2-1.6 (its epistemic growth is a change of the state distribution),
the ensemble ratio stays at 12-187: for history-based models the shift sits in
the fine structure of the input signal, not in the state.

### 3.4 Hypothesis H3.3 (highD test)

| Regime | Share | NLL HeteroMDN | NLL LSTMConstSigma | Difference | Holm p | Predicted variance / steady | Squared residual / steady |
|---|---|---|---|---|---|---|---|
| all | 1 | -2.93 [-2.96, -2.89] | -2.78 [-2.81, -2.75] | -0.145 [-0.163, -0.130] | 0.005 | - | - |
| congested | 0.024 | -2.07 | -2.27 | +0.197 [0.101, 0.280] | 0.005 | 6.1 [5.1, 7.4] | 7.0 [5.6, 8.6] |
| braking | 0.057 | -2.29 | -1.97 | -0.328 [-0.574, -0.173] | 0.005 | 187 [40, 391] | 63 [17, 131] |
| approaching | 0.21 | -2.90 | -2.80 | -0.103 [-0.125, -0.073] | 0.005 | 19 [1.2, 53.5] | 5.7 [1.3, 12.4] |
| free | 0.23 | -3.09 | -2.92 | -0.172 [-0.182, -0.162] | 0.005 | 0.75 [0.72, 0.79] | 0.87 [0.80, 0.93] |
| steady | 0.25 | -3.02 | -2.88 | -0.143 [-0.158, -0.124] | 0.005 | 1 | 1 |
| other | 0.23 | -2.94 | -2.79 | -0.151 [-0.163, -0.138] | 0.005 | 1.15 [1.10, 1.20] | 1.29 [1.16, 1.41] |

H3.3 is **supported** in-domain: the intervals of the two models do not
overlap; braking, approaching and congestion have 6-190 times the predicted
variance of steady following and free flow 0.75 times, and the squared
residuals show the same picture (5.7-63 and 0.87). One exception: in
congestion the constant-sigma model has the better NLL. Under shift the heteroscedastic model is better on every data set
(difference -21 to -384 nats), both being miscalibrated.

## 4. Findings

1. The one-step target of the specification (Savitzky-Golay acceleration, 10 Hz)
   is predictable from the speed history to 0.013-0.027 m/s^2 on highD. What the
   models learn, and what their aleatoric variance measures, is the smoothness of
   the recorded signal. It differs between the data sets by a factor of 10 in
   standard deviation, so "aleatoric variance is stable across data sets" cannot
   hold for this target whatever the model.
2. Physics does not help the one-step task: the closed-loop IDM has a one-step
   RMSE of 0.98 m/s^2 against 0.35 for predicting zero, and a physics weight of
   0.001 or more degrades the median of QuantilePIDL by an order of magnitude.
3. Ratios of mean variances are fragile (heavy tails, placebo interval 0.4-3.1
   for the ensemble); ratios of geometric means are stable (placebo 1.03-1.07)
   and lead to the same conclusions.
4. OpenACC: 9.4 % of the windows lie in (s, dv, v) cells never seen in training
   (speeds up to 37 m/s against 27 m/s in highD). There the networks extrapolate:
   mean squared error 0.7-3.5 (m/s^2)^2 for HeteroMDN, the ensemble and MCDropout,
   with predicted variances of the same size; 99.8 % of the ensemble's epistemic
   variance on OpenACC sits in states that are rare in three dimensions.
5. MCDropout is over-dispersed in-domain (PICP 0.996, predicted variance 57 times
   the squared error) and therefore looks best calibrated under shift.
6. HeteroMDN has a worse mean than the MSE-trained LSTM (RMSE 0.027 against 0.015),
   the known cost of NLL training.

## 5. Deviations from the specification

| # | Deviation | Reason |
|---|---|---|
| 1 | Training data are FollowNet HighD events, not raw highD | raw highD needs a levelXdata application by the user |
| 2 | Events, fold files and the IDM reference are imported from cf-stability; NGSIM is its own reconstruction of the raw data | the Montanino-Punzo set is not public |
| 3 | Split = fold 0 of cf-stability (60/20/20) instead of 70/15/15 | same splits in all four projects |
| 4 | One acceleration definition for all data sets (recomputed for Waymo and NGSIM from the stored speed) | comparable aleatoric variances |
| 5 | Regime `braking` uses `a[t-1]`, not the target | a label must be known before the outcome |
| 6 | Second rare-bin definition (10 % of mass) and rarity in three dimensions | decision of the M1 review; the (s, dv) grid ignores the speed |
| 7 | Models without an epistemic output get an across-seed decomposition, always labelled | otherwise they have no epistemic map |
| 8 | Extra baseline `idm_white_noise_mle`; `w_phys` selected on validation (0.0001) instead of 0.3 | section 4, finding 2 |
| 9 | DiffusionCF is evaluated on 10 000 windows per data set | sampling budget (32 samples x 50 steps) |
| 10 | Geometric-mean ratios, placebo and cross-check pairs, bin-matched ratios added to the analysis | robustness |
| 11 | Holm families use at least 4000 bootstrap replicates | with 1000 the smallest p-value cannot pass the correction |
| 12 | LSTM models stop at the cap of 40 epochs | budget; early stopping never triggered |

## 6. Open points for the review

1. **Target of the prediction.** Options: (A) keep the specification and report
   finding 1 as a result; (B) bring all data sets to one signal processing (same
   smoothing of speed and gap) and repeat E1; (C) add a behavioural target, the
   mean acceleration over the next 1 s, next to the one-step target. M3 (rollouts)
   depends on the choice: with (A) the rollouts will inherit the over-confidence
   under shift.
2. H3.1: keep the thresholds of the plan (then the verdict is "refuted") or
   restate the hypothesis for the epistemic part only.
3. Raw highD: application to levelXdata, if the FollowNet events are not enough.
4. NGSIM US-101 (7 353 events) is available in cf-stability as a fourth shift pair.
