# Milestone M3 report: closed-loop rollouts and calibration (experiment E2)

Date: 2026-09-28. Environment: shared `cf-stability/.venv` (Python 3.11.8, torch 2.11.0+cu128),
GPU RTX 6000 Ada shared with other jobs. Decisions of the M2 review carried out here: the
behavioural target `mean_1s` and the data set NGSIM US-101.
Tables: `artifacts/e2/tables/` (`summary.md`), `artifacts/e1_mean1s/tables/`, `artifacts/e1/tables/`;
figures: `artifacts/e2/figures/`, `artifacts/e1_mean1s/figures/`.

## 1. Commands

```bash
export PYTHONUTF8=1
.venv/Scripts/python.exe scripts/import_cf_stability_events.py --names ngsim_us101
.venv/Scripts/python.exe -m pytest -m "not slow" -p no:cacheprovider        # 414 passed, 96 s
.venv/Scripts/python.exe -m pytest -m gpu -p no:cacheprovider               # 21 passed
bash scripts/run_e1_retry.sh artifacts/e1/logs/us101.out 8                  # 50 new tables for ngsim_us101, 14 min
.venv/Scripts/python.exe scripts/select_w_phys.py --config configs/experiment/e1_mean1s.yaml --grid 0 0.0001 0.001 0.01 0.1 0.3
bash scripts/run_e1_retry.sh artifacts/e1_mean1s/logs/run.out 8 --config configs/experiment/e1_mean1s.yaml   # 2 h 35 min on 3-4 processes
bash scripts/run_e2_retry.sh artifacts/e2/logs/run.out 8                    # both targets; 2 h for target a, 25 min for mean_1s
.venv/Scripts/python.exe scripts/analyze_e1.py --root artifacts/e1_mean1s   # and analyze_e1_extra.py, summarize_e1.py, make_figures_e1.py
.venv/Scripts/python.exe scripts/analyze_e2.py --root artifacts/e2          # 4 min, tables and 12 figures
.venv/Scripts/python.exe scripts/summarize_e2.py --root artifacts/e2 > artifacts/e2/tables/summary.md
```

No run failed; the retry wrappers were never triggered in M3.

## 2. What was built and run

| Part | Content |
|---|---|
| Rollout engine | batched closed-loop simulation (semi-implicit Euler, clip [-8, 4], speed floor, hold of a draw, collision step, leader given), matches the deterministic simulator to 4e-7 m, results independent of the batch size |
| Rollout tables | per event and horizon: truth, quantiles with (S + 1) plotting positions and with linear interpolation, rank, fair CRPS, collision share |
| Calibration | in-house split conformal (absolute and normalised score), score of a rollout interval, static per-horizon correction, per-horizon ACI (frozen and online), verdict H3.4 |
| Targets | `a` (one step) and `mean_1s` = `(v[t+10] - v[t]) / 1 s`; second training experiment `artifacts/e1_mean1s` (DiffusionCF is not retrained) |
| Data | NGSIM US-101 (7 353 events) added as the fourth shift pair |
| E2 | 2 targets x 13 model labels (10 configurations + 3 sampler-mode variants) x 5 seeds x 6 data sets = 780 rollout tables; 500 events per data set, 200 rollouts per event (DiffusionCF 32), horizon 10 s |
| Tests | 414 fast, 21 GPU; the two tests required by the specification are in: ACI on a drifting series, rollout collision detection |

## 3. Key numbers

### 3.1 H3.4, part (i): one-step split conformal, calibrated on highD validation

Coverage of the 90 % interval, raw -> conformal (normalised score), mean over 5 seeds:

| Model | highD test, target `a` | highD test, `mean_1s` | NGSIM I-80, `a` | NGSIM I-80, `mean_1s` |
|---|---|---|---|---|
| idm_white_noise | 0.945 -> 0.904 | 0.944 -> 0.904 | 0.888 -> 0.778 | 0.926 -> 0.844 |
| lstm_const_sigma | 0.925 -> 0.903 | 0.922 -> 0.903 | 0.081 -> 0.074 | 0.090 -> 0.083 |
| hetero_mdn | 0.908 -> 0.899 | 0.911 -> 0.900 | 0.187 -> 0.182 | 0.172 -> 0.166 |
| deep_ensemble | 0.945 -> 0.897 | 0.929 -> 0.899 | 0.328 -> 0.283 | 0.297 -> 0.276 |
| mc_dropout | 0.996 -> 0.898 | 0.968 -> 0.897 | 0.523 -> 0.299 | 0.355 -> 0.269 |
| gp_residual_idm | 0.928 -> 0.907 | 0.930 -> 0.906 | 0.542 -> 0.500 | 0.608 -> 0.558 |
| quantile_pidl | 0.907 -> 0.898 | 0.897 -> 0.898 | 0.163 -> 0.160 | 0.126 -> 0.126 |
| diffusion_cf | 0.988 -> 0.909 | 0.890 -> 0.903 | 0.417 -> 0.262 | 0.284 -> 0.296 |

In-domain every model is within 0.897-0.910 (tolerance of the plan: 0.87-0.93). Under
shift a correction calibrated on highD does not help: the data are not exchangeable.

### 3.2 H3.4, parts (ii) and (iii): coverage of the 90 % gap interval in rollouts, highD test

PICP at 1 / 3 / 5 / 10 s, mean over 5 seeds and 500 events:

| Model | raw, target `a` | raw, `mean_1s` | ACI online, `a` | median width at 10 s, m: raw -> ACI (`a`) | CRPS at 10 s, m (`a` / `mean_1s`) |
|---|---|---|---|---|---|
| idm_white_noise | 0.78 0.63 0.57 0.47 | 0.94 0.83 0.79 0.66 | 0.90 0.90 0.90 0.89 | 6.3 -> 22.3 | 3.71 / 5.64 |
| idm_white_noise_mle | 0.57 0.41 0.30 0.25 | 0.91 0.83 0.76 0.67 | 0.90 0.90 0.90 0.90 | 4.4 -> 34.4 | 5.10 / 4.42 |
| lstm_const_sigma | 0.22 0.53 0.46 0.42 | 0.55 0.67 0.54 0.48 | 0.91 0.91 0.91 0.90 | 6.5 -> 23.9 | 4.40 / 4.52 |
| hetero_mdn | 0.19 0.46 0.39 0.32 | 0.53 0.67 0.55 0.48 | 0.91 0.91 0.91 0.90 | 5.2 -> 21.8 | 5.13 / 4.45 |
| hetero_mdn_mix3 | 0.20 0.42 0.36 0.30 | 0.51 0.67 0.54 0.48 | 0.91 0.91 0.91 0.90 | 5.6 -> 23.3 | 5.57 / 4.55 |
| deep_ensemble | 0.21 0.57 0.51 0.47 | 0.57 0.71 0.59 0.54 | 0.91 0.91 0.91 0.90 | 5.8 -> 17.3 | 3.80 / 4.10 |
| mc_dropout | 0.47 0.74 0.65 0.56 | 0.66 0.78 0.64 0.55 | 0.90 0.91 0.90 0.89 | 12.1 -> 28.7 | 5.58 / 4.53 |
| gp_residual_idm | 0.54 0.41 0.34 0.27 | 0.92 0.84 0.77 0.69 | 0.91 0.91 0.91 0.90 | 4.1 -> 26.0 | 4.07 / 3.32 |
| quantile_pidl | 0.17 0.49 0.44 0.40 | 0.47 0.64 0.53 0.48 | 0.91 0.91 0.91 0.90 | 5.2 -> 19.0 | 3.85 / 4.55 |
| diffusion_cf | 0.60 0.94 0.93 0.89 | same model | 0.90 0.91 0.90 0.89 | 11.7 -> 11.8 | 2.11 |

Static per-horizon correction (no adaptation) gives 0.88-0.94 in-domain, ACI frozen 0.87-0.96.

### 3.3 The same under shift, NGSIM I-80 (target `a`)

| Model | raw | static | ACI online | median width at 10 s, m: raw -> ACI | collided by 10 s | CRPS at 10 s, m |
|---|---|---|---|---|---|---|
| idm_white_noise | 0.47 0.39 0.39 0.38 | 0.74 0.79 0.85 0.91 | 0.90 0.90 0.90 0.90 | 4.9 -> 17.3 | 0.000 | 3.64 |
| lstm_const_sigma | 0.03 0.04 0.04 0.08 | 0.21 0.11 0.16 0.31 | 0.89 0.89 0.90 0.89 | 3.5 -> 284 | 0.325 | 22.9 |
| hetero_mdn | 0.05 0.06 0.08 0.15 | 0.43 0.21 0.33 0.55 | 0.89 0.89 0.89 0.88 | 6.5 -> 100 | 0.290 | 17.0 |
| deep_ensemble | 0.12 0.14 0.18 0.33 | 0.78 0.40 0.57 0.80 | 0.90 0.89 0.89 0.89 | 13.8 -> 56 | 0.289 | 13.7 |
| mc_dropout | 0.33 0.43 0.53 0.66 | 0.74 0.55 0.76 0.89 | 0.89 0.89 0.89 0.90 | 34 -> 114 | 0.182 | 16.7 |
| gp_residual_idm | 0.19 0.14 0.14 0.17 | 0.51 0.56 0.70 0.82 | 0.89 0.90 0.90 0.90 | 2.3 -> 20.0 | 0.002 | 4.23 |
| quantile_pidl | 0.05 0.07 0.08 0.15 | 0.41 0.19 0.31 0.52 | 0.89 0.89 0.89 0.88 | 5.9 -> 72 | 0.304 | 14.4 |
| diffusion_cf | 0.33 0.45 0.63 0.81 | 0.75 0.41 0.61 0.81 | 0.90 0.89 0.89 0.89 | 22.7 -> 27.6 | 0.121 | 6.27 |

US-101, Waymo and OpenACC show the same picture (`summary.md`).

### 3.4 Verdict H3.4

| Count over 13 model labels x 5 data sets | Target `a` | Target `mean_1s` |
|---|---|---|
| part (i) one-step conformal within 0.90 +- 0.03 | 65 of 65 | 65 of 65 |
| part (ii) raw rollout coverage below 0.80 at some horizon | 65 of 65 (largest minimum 0.70) | 65 of 65 (largest minimum 0.77) |
| part (iii) coverage after ACI online >= 0.87 at every horizon | 65 of 65 (smallest 0.880) | 65 of 65 (smallest 0.882) |
| verdict by point estimates | supported 65 | supported 65 |
| verdict when every part must hold for the bound of its interval | supported 32, inconclusive 33 | supported 29, inconclusive 36 |
| refuted (raw coverage >= 0.87 everywhere) | 0 | 0 |

H3.4 is **supported** for both targets. ACI step: coverage 0.867-0.932 for gamma
0.005, 0.880-0.918 for 0.01 (used), 0.890-0.911 for 0.02, 0.894-0.907 for 0.05; the
width grows with gamma under shift (NGSIM, median over models at 10 s: 62 -> 78 m).

### 3.5 Behavioural target in the one-step analysis (E1 repeated, `artifacts/e1_mean1s`)

| Quantity | Target `a` | Target `mean_1s` |
|---|---|---|
| persistence RMSE on highD / std of the target | 0.017 / 0.35 | 0.157 / 0.35 |
| RMSE of the LSTM models on highD | 0.013-0.027 | 0.041-0.054 |
| NLL of HeteroMDN on highD / NGSIM I-80 | -2.93 / 82.5 | -1.74 / 67.7 |
| PICP of HeteroMDN on NGSIM I-80 | 0.19 | 0.17 |
| H3.1, epistemic ratio of the ensemble (geometric mean): NGSIM, US-101, Waymo, OpenACC | 238, 167, 30, 21 | 241, 225, 26, 20 |
| H3.1, aleatoric ratio of HeteroMDN (geometric mean), same sets | 8.8, 4.6, 5.6, 7.1 | 5.0, 3.5, 3.2, 3.6 |
| H3.1 verdict, four counted pairs | refuted for learned aleatoric variance; supported for the GP (2 of 4 pairs, aleatoric constant by construction) | the same (GP: 3 of 4 pairs) |
| H3.3, NLL HeteroMDN - LSTMConstSigma on highD | -0.145 [-0.163, -0.130], intervals separate | -0.038 [-0.052, -0.019], Holm p 0.005, intervals overlap |
| H3.3, predicted variance relative to steady: congested, braking, approaching, free | 6.1, 187, 19, 0.75 | 3.2, 7.8, 1.4, 0.85 |

## 4. Findings

1. **Where the coverage is lost.** HeteroMDN, target `a`, highD: median error of the
   rollout median 0.03 / 0.22 / 0.89 / 4.4 m at 1 / 3 / 5 / 10 s against a median
   interval width of 0.02 / 0.36 / 1.2 / 5.2 m. The errors are correlated in time, the
   draws are not, so the spread of the rollouts grows too slowly. DiffusionCF draws
   whole 5 s sequences: error 0.03 / 0.17 / 0.61 / 2.2 m, width 0.07 / 1.1 / 3.6 /
   11.9 m, coverage 0.89-0.94 from 3 s on and the smallest CRPS (2.1 m at 10 s).
2. **Freezing the epistemic draw along a rollout** raises the coverage of the
   ensemble from 0.47 to 0.78 at 10 s (target `a`), MC dropout reaches 0.94-0.95, but
   with an interval of 96 m at 10 s; for the GP the mode is irrelevant (its epistemic
   share in-domain is about 1 %). With `mean_1s` the modes differ by 0.02-0.09.
3. **ACI restores the nominal coverage everywhere, the price is the width.**
   In-domain the corrected interval at 10 s is 2.4-7.8 times wider than the raw one
   (17-34 m; DiffusionCF: unchanged, 11.8 m); under shift it becomes useless for the
   one-step LSTM models (NGSIM, 10 s: 56-284 m). The static correction fails under
   shift (coverage 0.11-0.91 on NGSIM).
4. **Physics gives safety and sharpness under shift.** Share of rollouts with a
   collision within 10 s on NGSIM I-80: IDM models 0, GP residual 0.2 %, DiffusionCF
   12 %, LSTM models 18-37 %; in-domain 0-2.3 % (5 % for MC dropout with kept masks). CRPS of the gap at 10 s on NGSIM: GP
   4.2 m, IDM + noise 3.6 m, DiffusionCF 6.3 m, LSTM models 14-23 m.
5. **The behavioural target helps in-domain and in closed loop, not against
   shift.** Raw rollout coverage at 1 s rises from 0.2 to 0.5 for the LSTM models and
   from 0.54 to 0.92 for the GP; the over-confidence under shift stays (NLL 68,
   PICP 0.17 on NGSIM). A target averaged over 1 s is still predicted by the models
   three to four times better than by persistence of the same quantity, i.e. still
   largely from the smoothness of the signal.
6. **H3.3 depends on the target.** The heteroscedastic gain is 0.145 nats for `a`
   and 0.038 for `mean_1s`; the ordering of the variance by regime holds for both.
7. **Finite-sample quantiles.** With linear interpolation instead of (S + 1)
   plotting positions the raw coverage is lower by 0.04-0.07 for DiffusionCF
   (32 rollouts) and by 0.00-0.02 for the models with 200 rollouts; all tables use
   the plotting positions.

## 5. Deviations from the specification

| # | Deviation | Reason |
|---|---|---|
| 1 | Second target `mean_1s`, second training experiment, draws kept for 10 steps in its rollouts | decision of the M2 review |
| 2 | NGSIM US-101 added; four counted shift pairs | decision of the M2 review |
| 3 | In-house split conformal and ACI instead of MAPIE | allowed by the specification; one dependency less in the shared environment |
| 4 | ACI acts on a normalised score of the rollout interval (distance outside the interval in units of its half-widths), not on the quantile level of the rollouts | 200 rollouts cannot widen an interval beyond their own range; the score can |
| 5 | Three corrected variants are reported (static, ACI frozen, ACI online); the verdict uses ACI online | under shift only the online variant can adapt |
| 6 | Quantiles with (S + 1) plotting positions | M1 finding; linear interpolation is kept in the tables for comparison |
| 7 | Sampler-mode variants for the ensemble, MC dropout and the GP | the specification fixes the mode; the variants show how much it matters |
| 8 | One rollout start per event (`t0` = end of the first 3 s), 500 events per data set | independence of the units of the bootstrap and of ACI |
| 9 | DiffusionCF: 32 rollouts per event, re-planning every 5 s, the same checkpoints for both targets | sampling budget of the specification |
| 10 | The model input gap is floored at 0.1 m after a collision; trajectories continue and are counted | models must not receive non-positive gaps |

## 6. Open points for the review

1. M4 (platoons): which target and sampler to carry forward. Proposal: DiffusionCF,
   GP residual and IDM + noise as they are; the ensemble with the member kept per
   rollout; QuantilePIDL with `mean_1s`; all with ACI-corrected intervals reported
   next to the raw ones.
2. Over-confidence under shift is not cured by the target. The remaining option of
   the M2 review, one signal processing for all data sets, is still open.
3. H3.1 stays "refuted as written" for both targets; restating it for the epistemic
   part only is a decision for the paper.
