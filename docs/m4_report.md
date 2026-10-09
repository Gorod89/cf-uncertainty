# Milestone M4: harmonised signals and platoon-level evaluation (experiment E3)

Date: 2026-10-06 (all runs complete: E3 on the recorded and on the harmonised signals, E1 and E2 on
the harmonised signals). Contract: `docs/design.md`, sections 12 (harmonised signals, H3.1e) and 13 (E3,
H3.2). Decisions of the M3 review that shaped this milestone: start M4 with the proposed platoon
model set; restate H3.1 for its epistemic part (H3.1e) and keep the original verdict next to it;
recompute E1 and E2 with one harmonised signal processing for every data set.

## 1. Commands

```bash
cd cf-uncertainty && export PYTHONUTF8=1
# harmonised signals: report, physics weight of QuantilePIDL, pilot, full one-step experiment (both targets)
./.venv/Scripts/python.exe scripts/harmonise_report.py                     # artifacts/e1h/harmonisation.json
./.venv/Scripts/python.exe scripts/select_w_phys.py --config configs/experiment/e1h.yaml --grid 0 0.0001 0.001 0.01 0.1 0.3
./.venv/Scripts/python.exe scripts/select_w_phys.py --config configs/experiment/e1h_mean1s.yaml --grid 0 0.0001 0.001 0.01 0.1 0.3
./.venv/Scripts/python.exe scripts/run_e1.py --config configs/experiment/e1h.yaml --pilot
bash scripts/run_e1_retry.sh artifacts/e1h/logs/r1.out 8 --config configs/experiment/e1h.yaml --models idm_white_noise idm_white_noise_mle lstm_const_sigma hetero_mdn hetero_mdn_mix3 mc_dropout gp_residual_idm quantile_pidl
bash scripts/run_e1_retry.sh artifacts/e1h/logs/r2.out 8 --config configs/experiment/e1h.yaml --models deep_ensemble
bash scripts/run_e1_retry.sh artifacts/e1h/logs/r3.out 8 --config configs/experiment/e1h.yaml --models diffusion_cf
bash scripts/run_e1_retry.sh artifacts/e1h_mean1s/logs/r1.out 8 --config configs/experiment/e1h_mean1s.yaml --models idm_white_noise idm_white_noise_mle lstm_const_sigma hetero_mdn hetero_mdn_mix3 mc_dropout
bash scripts/run_e1_retry.sh artifacts/e1h_mean1s/logs/r2.out 8 --config configs/experiment/e1h_mean1s.yaml --models deep_ensemble
bash scripts/run_e1_retry.sh artifacts/e1h_mean1s/logs/r3.out 8 --config configs/experiment/e1h_mean1s.yaml --models quantile_pidl diffusion_cf
bash scripts/run_e1_retry.sh artifacts/e1h_mean1s/logs/r4.out 8 --config configs/experiment/e1h_mean1s.yaml --models gp_residual_idm
for R in e1h e1h_mean1s; do
  ./.venv/Scripts/python.exe scripts/analyze_e1.py --root artifacts/$R --config configs/experiment/$R.yaml --quiet
  ./.venv/Scripts/python.exe scripts/analyze_e1_extra.py --root artifacts/$R --config configs/experiment/$R.yaml
  ./.venv/Scripts/python.exe scripts/summarize_e1.py --root artifacts/$R > artifacts/$R/tables/summary.md
  ./.venv/Scripts/python.exe scripts/make_figures_e1.py --root artifacts/$R
done
# harmonised rollouts
bash scripts/run_e2_retry.sh artifacts/e2h/logs/run.out 8 --config configs/experiment/e2h.yaml
./.venv/Scripts/python.exe scripts/analyze_e2.py --root artifacts/e2h --quiet
./.venv/Scripts/python.exe scripts/summarize_e2.py --root artifacts/e2h > artifacts/e2h/tables/summary.md
# platoons: profiles, pilot, main run (raw + harmonised checkpoints), sensitivities, analysis
./.venv/Scripts/python.exe scripts/select_profiles.py                      # configs/platoon/profiles.yaml
./.venv/Scripts/python.exe scripts/run_e3.py --pilot                       # artifacts/e3_pilot
bash scripts/run_e3_retry.sh artifacts/e3/logs/run_raw.out 8 --variants raw --quiet
bash scripts/run_e3_retry.sh artifacts/e3/logs/run_h.out 8 --variants harmonised --quiet
bash scripts/run_e3_retry.sh artifacts/e3_vcap/logs/run.out 8 --config configs/experiment/e3_vcap.yaml --quiet
bash scripts/run_e3_retry.sh artifacts/e3_step/logs/run.out 8 --config configs/experiment/e3_step.yaml --quiet
./.venv/Scripts/python.exe scripts/analyze_e3.py --root artifacts/e3 --quiet
./.venv/Scripts/python.exe scripts/analyze_e3.py --root artifacts/e3_vcap --set headline --quiet
./.venv/Scripts/python.exe scripts/analyze_e3.py --root artifacts/e3_step --set headline --quiet
# tests
./.venv/Scripts/python.exe -m pytest -m "not slow"                          # 489 passed, 24 deselected (140 s)
```

## 2. What was built and run

* `cf_unc/data/harmonise.py`, `scripts/harmonise_report.py`: one signal processing for every data
  set (design.md 12): Savitzky-Golay (11, 2) on the speeds, acceleration as the central difference
  of the smoothed speed, kinematically consistent gap with least-squares offsets, drop rules.
  `load_events(..., signal="harmonised")`; E1 and E2 take `signal:` from the experiment config;
  roots `artifacts/e1h`, `artifacts/e1h_mean1s`, `artifacts/e2h`. H3.1e in `analyze_e1_extra.py`.
* `cf_unc/platoon/{openacc,profiles,simulate,evaluate}.py`, `cf_unc/experiments/e3.py`,
  `cf_unc/report/figures_e3.py`, `scripts/{select_profiles,run_e3,analyze_e3}.py`,
  `scripts/run_e3_retry.sh`, `configs/experiment/e3*.yaml`, `configs/platoon/profiles.yaml`,
  `configs/empirical_growth.yaml`: OpenACC profile selection from the raw files of AstaZero,
  ZalaZone and Vicolungo (read in place), cascade of 20 followers behind a measured leader
  (`leader_override` of the rollout engine), amplitude CRPS, intervals, growth exponent, collisions.
* `RolloutConfig.v_max` / `PlatoonConfig.v_max`: optional cap of the follower speed (default none;
  the E3 speed-cap sensitivity). The measured growth exponent is fitted over balanced indices only.
* Tests: 489 fast, all passing (75 new since M3: harmonisation, OpenACC reader, profiles, platoon simulation, evaluation,
  E3 runner, speed cap, growth fit), 24 slow / GPU deselected.
* Runs on the recorded signals (`raw`): E3 main run 6 models x 5 seeds x 60 profiles x 20 followers
  x 100 rollouts (DiffusionCF 32), time line 3 s history + 30 s burn-in + profile (30-120 s) +
  40 s tail = 1899 steps; 1800 units, 0 failures, 4.6 h on the shared GPU (IDM 0.11 h, LSTM 0.96,
  deep ensemble 1.68, GP residual 0.91, QuantilePIDL 0.18, DiffusionCF 0.76). Sensitivities on the
  headline set: 1.8 h (speed cap) and 1.5 h (per-step ensemble). Harmonised branch: `e1h`
  (350 tables, 3.9 h), `e1h_mean1s` (350 tables, 3.9 h in two processes), `e2h` (780 rollout tables,
  2.1 h), E3 harmonised variant (1800 units, 0 failures, 3.9 h). The GPU was shared with the
  two other projects throughout (48 of 49 GB in use; one DiffusionCF
  attempt died of host memory and was resumed by the retry wrapper).

## 3. Key numbers

### 3.1 Harmonisation (`artifacts/e1h/harmonisation.json`)

| data set | events in | dropped | share | gap residual before, m (RMS, kept) |
|---|---|---|---|---|
| highd | 12512 | 2 | 0.02 % | 0.24 |
| ngsim (I-80) | 7651 | 20 | 0.26 % | 0.07 |
| ngsim_us101 | 7353 | 8 | 0.11 % | 0.06 |
| ngsim_fn | 1937 | 3 | 0.15 % | 0.09 |
| waymo | 967 | 3 | 0.31 % | 0.04 |
| openacc | 4715 | 315 | 6.68 % | 0.44 |

OpenACC loses 6.7 % of its events (above the 5 % mark of design.md 12.2; the drops are events
whose recorded gap drifts against the integrated speed difference by more than 0.3 m/s). The
harmonisation removes the gap roughness of highD (RMS of the second difference of `s`: 0.074 ->
0.0032) but leaves the speed roughness of the NGSIM reconstructions where it was (0.019 against
0.0018 for highD, 10x): the smoothing window of 1.1 s does not undo the resampling of NGSIM.

QuantilePIDL physics weight on the harmonised validation fold (rule of `scripts/select_w_phys.py`:
smallest pinball loss among `w_phys > 0`, ties within 1 % to the larger weight): 0.001 for both
targets. Side effect to keep in mind: against 0.0001 the pinball loss is 1 % lower but the RMSE
of the median is 10x worse (0.136 against 0.015 m/s^2 for target `a`), because the physics term
pulls the median towards the IDM; the rule was fixed before the runs and is applied as written.

### 3.2 Platoon profiles (`configs/platoon/profiles.yaml`)

95 OpenACC runs; 43 have no leader plus three followers (the ZalaZone handling runs have no
vehicle 1). Candidate windows -> selected [headline]: AstaZero ACC 209 -> 20 [2], AstaZero human
26 -> 11 [2], ZalaZone ACC 41 -> 20 [2], Vicolungo ACC 20 -> 7 [2], Vicolungo human 5 -> 1 [1],
Vicolungo mixed 1 -> 1 [1]; 60 profiles, 3-10 measured followers, leader amplitude 2.2 / 8.9 /
14.5 m/s (min / median / max), leader speed <= 28.68 m/s (section 5). The headline profiles start
at 13-14 m/s with IDM equilibrium gaps of 7-22 m.

### 3.3 H3.2: amplitude CRPS against `idm_white_noise` (recorded signals, extended set, 60 profiles)

Amplitude of every follower over the profile and the 40 s tail; CRPS of the pooled law of the 500
rollouts (100 x 5 seeds) against the measured amplitude, averaged over the measured indices of a
profile and over profiles; reduction `1 - CRPS_model / CRPS_idm` paired by profile, 95 % bootstrap
over profiles; supported if the lower bound >= 0.20, refuted if the upper bound < 0.05.

| model | CRPS, m/s | reduction [95 %] | verdict | collided (any pair) |
|---|---|---|---|---|
| idm_white_noise (reference) | 2.98 [2.41, 3.57] | - | - | 0.000 |
| diffusion_cf | 2.22 [1.84, 2.62] | 0.25 [0.01, 0.44] | inconclusive | 0.902 |
| gp_residual_idm | 3.15 [2.59, 3.70] | -0.06 [-0.34, 0.16] | inconclusive | 0.189 |
| quantile_pidl (target `mean_1s`) | 6.21 [5.55, 6.97] | -1.09 [-1.69, -0.64] | refuted | 0.999 |
| deep_ensemble (kept member) | 24.5 [18.0, 33.1] | -7.2 [-10.8, -4.9] | refuted | 1.000 |
| lstm_const_sigma | 57.7 [43.8, 73.5] | -18.4 [-24.5, -13.6] | refuted | 1.000 |

Subsets (same window): ACC profiles (47) DiffusionCF 0.34 [0.09, 0.50], GP residual 0.15
[-0.08, 0.35], both inconclusive; human-driven profiles (12) DiffusionCF -0.84 [-1.26, -0.47] and
GP residual -2.75 [-4.19, -1.85], both refuted. Headline set (10 profiles): DiffusionCF 0.14
[-0.59, 0.57], GP residual 0.23 [-0.52, 0.60] (inconclusive), the LSTM-based models refuted.
Against `lstm_const_sigma` as the reference every other model is "supported" (reduction 0.58-0.96),
which only restates that the LSTM runs away. Amplitude ratio `A_i / A_0` instead of the amplitude:
DiffusionCF 0.08 [-0.20, 0.30] (inconclusive), GP residual -0.27 [-0.53, -0.03] (refuted).

### 3.4 Collisions, interval coverage, growth along the platoon (recorded signals, extended set)

Share of rollouts in which vehicle i collided (any time of the 190 s time line) and during the
constant-speed burn-in (vehicle 1):

| model | i = 1 | 2 | 5 | 10 | 20 | burn-in, i = 1 |
|---|---|---|---|---|---|---|
| idm_white_noise | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| gp_residual_idm | 0.04 | 0.07 | 0.05 | 0.02 | 0.02 | 0.000 |
| diffusion_cf | 0.08 | 0.06 | 0.14 | 0.27 | 0.54 | 0.001 |
| quantile_pidl | 0.49 | 0.70 | 0.83 | 0.90 | 0.92 | 0.086 |
| lstm_const_sigma | 0.52 | 0.63 | 0.67 | 0.77 | 0.82 | 0.110 |
| deep_ensemble | 0.60 | 0.70 | 0.80 | 0.83 | 0.84 | 0.270 |

Coverage of the 90 % amplitude interval (all measured indices; raw = (S + 1) sample quantiles,
static = leave-one-profile-out correction, ACI online over the profiles) and its median width:

| model | raw | static | ACI | width raw / static, m/s |
|---|---|---|---|---|
| idm_white_noise | 0.30 | 0.90 | 0.81 | 2.6 / 15.6 |
| gp_residual_idm | 0.20 | 0.90 | 0.84 | 1.9 / 23.7 |
| diffusion_cf | 0.66 | 0.90 | 0.84 | 11.0 / 17.8 |
| quantile_pidl | 0.57 | 0.91 | 0.86 | 71 / 107 |
| deep_ensemble | 0.89 | 0.90 | 0.87 | 400 / 407 |
| lstm_const_sigma | 0.79 | 0.91 | 0.87 | 442 / 530 |

Growth of the speed standard deviation along the platoon, `sd_i = sd_1 i^beta` (profile window,
95 % bootstrap over profiles): measured platoons 0.12 [0.08, 0.16] (fitted over indices 1-4, which
>= 54 of the 60 profiles measured; `sd` 3.5 -> 4.2 m/s), idm_white_noise -0.01 [-0.06, 0.03]
(flat, 2.9 -> 2.7), diffusion_cf 0.19 [0.15, 0.24] (3.6 -> 5.9), gp_residual_idm -0.19
[-0.26, -0.12] (3.4 -> 2.3), the LSTM-based models -0.05 to -0.22 at 10-50 m/s. Only DiffusionCF
reproduces the sign of the measured growth; it overshoots it.

### 3.5 Sensitivities (headline set, 10 profiles, recorded signals; `artifacts/e3_vcap`, `artifacts/e3_step`)

| model | main run CRPS | reduction | speed cap 35.9 m/s CRPS | reduction | collided (cap) |
|---|---|---|---|---|---|
| idm_white_noise | 3.11 | - | 3.11 | - | 0.00 |
| lstm_const_sigma | 33.3 | -9.7 [-18.1, -4.9] refuted | 3.97 | -0.28 [-1.17, 0.10] inconclusive | 0.88 |
| deep_ensemble (kept member) | 14.0 | -3.5 [-7.8, -1.6] refuted | 3.48 | -0.12 [-0.77, 0.22] inconclusive | 0.84 |
| quantile_pidl | 5.39 | -0.73 [-2.20, -0.08] refuted | 4.16 | -0.34 [-1.57, 0.24] inconclusive | 0.94 |
| deep_ensemble, per-step draws | - | - | 6.91 (no cap) | -1.23 [-3.84, 0.11] inconclusive | 0.92 |

A physical speed cap bounds the amplitudes of the runaway rollouts (the LSTM-based models become
"inconclusive" instead of "refuted") but they still collide in 84-94 % of the rollouts and never
beat the IDM reference; the ensemble with per-step draws (member and noise redrawn every step)
halves the CRPS of the kept member on the same profiles (14.0 -> 6.9) and its burn-in collisions
(0.27 -> 0.11), but stays worse than the IDM reference.

### 3.6 Harmonised signals: the whole pipeline repeated (`artifacts/e1h`, `e1h_mean1s`, `e2h`, E3 variant `harmonised`)

Every verdict is unchanged. The pilot of 2026-10-05 10:55 (under-trained models) had suggested a
tenfold smaller over-confidence under shift; the full runs do not confirm it: with the same
Savitzky-Golay window on every data set, the NGSIM reconstructions stay 10x rougher than highD
(3.1) and the one-step picture of M2/M3 repeats almost number for number.

One-step metrics, target `a`, mean over 5 seeds (recorded -> harmonised):

| model | NLL highD | NLL NGSIM I-80 | NLL US-101 | PICP NGSIM I-80 | RMSE highD |
|---|---|---|---|---|---|
| lstm_const_sigma | -2.78 -> -2.86 | 467 -> 397 | 329 -> 269 | 0.08 -> 0.09 | 0.011 -> 0.011 |
| hetero_mdn | -2.93 -> -3.08 | 82 -> 86 | 88 -> 89 | 0.19 -> 0.19 | 0.03 -> 0.03 |
| deep_ensemble | -2.97 -> -3.08 | 41 -> 38 | 46 -> 43 | 0.33 -> 0.34 | 0.02 -> 0.03 |
| mc_dropout | -2.40 -> -2.43 | 4.0 -> 3.4 | 3.9 -> 3.1 | 0.52 -> 0.55 | 0.02 -> 0.04 |
| gp_residual_idm | 0.22 -> 0.22 | 5.8 -> 5.6 | 3.6 -> 3.6 | 0.54 -> 0.54 | 0.30 -> 0.30 |
| quantile_pidl | -2.94 -> -2.92 | 102 -> 93 | 125 -> 102 | 0.16 -> 0.21 | 0.01 -> 0.04 |
| diffusion_cf | -2.43 -> -2.48 | 16 -> 15 | 17 -> 16 | 0.42 -> 0.43 | 0.02 -> 0.02 |
| idm_white_noise | 1.41 -> 1.44 | 1.9 -> 1.9 | 1.6 -> 1.6 | 0.89 -> 0.89 | 0.98 -> 1.02 |

Hypotheses, recorded -> harmonised (target `a` / target `mean_1s`):

* H3.1 as written: refuted for every learned model except the GP residual (supported) and the
  seed ensemble of the LSTM (supported) on both signal variants and both targets; the aleatoric
  variance still grows 4-1000x and the epistemic variance 4-800x from highD to the shifted sets
  (geometric ratios of the same order on both variants, `h31_verdicts.csv`).
* H3.1e (epistemic part, design.md 12.4): supported for every model and kind on both signal
  variants and both targets (recorded signals: epistemic variance up by >= 3x on all 4 counted
  pairs for every model but the GP residual's own posterior, 2 of 4; concentration in the rare
  states of the shifted sets 4-115x with lower bounds >= 3; the cross-check pair `ngsim_fn`
  agrees). The epistemic half of H3.1 holds; what refutes H3.1 as written is the aleatoric half.
* H3.3 (regime-aware law beats the constant-variance LSTM in-domain): NLL gain of HeteroMDN
  -0.145 -> -0.221 nats for target `a` (the gain grows with the rougher harmonised acceleration),
  -0.038 -> -0.093 nats for `mean_1s`; the ordering of the variance by regime is unchanged.
* H3.4 (conformal / ACI corrections restore coverage in rollouts): ACI online 0.89-0.92 at every
  horizon on every data set for both variants; by the point rule 130 of 130 (model, target, data
  set) cases supported on the recorded signals, 125 of 130 on the harmonised ones, where
  DiffusionCF (highD, both targets) and the kept-mask MC dropout (highD, `a`) are "refuted"
  because their *raw* rollout intervals already cover 0.89-0.94: the premise of H3.4 (raw
  under-coverage) does not hold for them and the correction keeps 0.89-0.90. Static and frozen
  corrections fail under shift exactly as in M3 (0.05-0.60 on NGSIM for the LSTM-based models).
* E3 (H3.2): DiffusionCF 0.31 [0.10, 0.46] (inconclusive, recorded 0.25 [0.01, 0.44]), GP residual
  -0.09 [-0.39, 0.14] (inconclusive), QuantilePIDL -0.50 [-0.98, -0.15], LSTM -3.05 [-4.51,
  -1.90], deep ensemble -16.8 [-22.9, -12.0] (refuted); the LSTM-based models swap places between
  the variants (LSTM 57.7 -> 11.6 m/s, ensemble 24.5 -> 51.4 m/s): their runaway is a property
  of the training run, not of the signals. Collisions 0.92 (DiffusionCF), 0.22 (GP residual),
  >= 0.997 (LSTM-based); growth exponents DiffusionCF 0.19, GP residual 0.02, IDM -0.02, LSTM
  +0.14 (its instability grows along the platoon on this variant).

## 4. Findings

1. **H3.2 is not supported on the recorded signals.** The only model whose amplitude CRPS beats the
   IDM reference is DiffusionCF (0.25 reduction, lower bound 0.01 against the 0.20 needed); the
   GP residual on IDM equals the reference; QuantilePIDL, the deep ensemble and the LSTM are far
   worse. On the ACC platoons (47 of 60) DiffusionCF's reduction is 0.34 [0.09, 0.50]; on the
   12 human-driven platoons every learned model is refuted.
2. **The LSTM-based models are not closed-loop stable over a platoon time line.** The headline
   profiles start at 13-14 m/s with equilibrium gaps of 7-22 m, a region highD covers thinly (4 %
   of its windows are below 8 m/s). From there the kept-member ensemble collides within the 30 s
   constant-speed burn-in in 27 % of the rollouts, and beyond the training gaps the networks
   command +4 m/s^2 without limit (speeds above 500 m/s, through the leader). This is the finding
   of M3 (collisions in 18-37 % of the NGSIM rollouts at 10 s) taken to 190 s and 20 vehicles; a
   speed cap makes the numbers finite but not better than the reference.
3. **DiffusionCF is the only learned model that reproduces the measured growth of the speed
   variability along the platoon** (beta 0.19 against 0.12 measured; IDM flat, GP residual
   damping), and it collides in 8 % of the first-follower rollouts rising to 54 % at the 20th
   vehicle: the growth it reproduces is partly the growth of its own instability.
4. **Raw amplitude intervals of the IDM-based models are far too narrow** (coverage 0.20-0.30,
   width 2-3 m/s against the 15-24 m/s a 90 % interval needs): the white noise of IDM and the GP
   residual noise average out over the profile, while the measured amplitudes vary between
   profiles by much more. The conformal correction restores 0.90 by construction (leave-one-out)
   and ACI online reaches 0.81-0.87 over 60 profiles; the corrected widths say that the amplitude
   of a measured follower is predictable to about +-8 m/s only.
5. **One harmonised signal processing changes no verdict** (3.6). A Savitzky-Golay window of
   1.1 s applied to every data set leaves the NGSIM reconstructions 10x rougher than highD, the
   one-step models stay persistence-dominated in-domain (RMSE 0.01 m/s^2) and over-confident by
   two orders of magnitude under shift (NLL 40-400 nats). The over-confidence is therefore not an
   artefact of *different* processing pipelines; it is the models' confidence in the smoothness of
   whatever signal they were trained on. Equalising the roughness would need a stronger filter
   tuned to NGSIM, which is a different experiment (open point 5).

## 5. Deviations from the specification

* Leader speed of a profile <= 28.68 m/s: the closed-loop IDM (`v0 = 28.9 m/s`) has no
  equilibrium gap above `v0` and reaches the cap of 120 m there; most Vicolungo highway windows are
  excluded. The initial gap comes from this IDM for both signal variants.
* Profile windows start at the leader's last steady start and end steady; the largest platoon that
  reaches 60 s is preferred (design.md 13.4).
* One random stream per (batch of 60 profiles, vehicle) instead of one per profile (one model call
  per step for all profiles); the batches are fixed, resumed units are bit-identical.
* DiffusionCF's 5 s plan sees the simulated future of its predecessor, so its cascade is not a joint
  simulation of the platoon.
* The measured growth exponent is fitted over the indices measured by at least half of the profiles
  (a pooled fit over all indices gave -0.75 for the extended set, a composition effect of the few
  long platoons with small amplitudes); the curve is still drawn for every index with >= 3 profiles.
* Sensitivities beyond the design (decision at the pilot, design.md 13.5): speed cap 35.9 m/s and
  per-step ensemble draws, headline set, recorded signals only.
* Harmonisation (design.md 12.5): speeds floored at 0, drop reason `non_finite`, H3.1e also refuted
  when the upper bound of the concentration ratio is < 1; the OpenACC drop share (6.7 %) exceeds the
  5 % mark and is reported, not fixed.
* The physics weight of QuantilePIDL on the harmonised signals is 0.001 by the pre-registered rule
  (0.0001 on the recorded signals), see 3.1.

## 6. Open points for the review

1. H3.2 as written cannot be supported with 60 profiles: the bootstrap interval of DiffusionCF's
   reduction is 0.43 wide. The ACC subset is the natural population for the paper (the human
   platoons are 12 profiles from two campaigns); report the verdict per driver mode?
2. The collision share of DiffusionCF along the platoon (8 % -> 54 %) and the runaway of the
   LSTM-based models argue for a closed-loop stability criterion in the model set of the paper
   (M5): the platoon evaluation is only meaningful for models that stay on the road.
3. The IDM reference's amplitude intervals are too narrow by 6x; the paper should present the
   conformal-corrected intervals as the headline and the raw ones as the diagnostic.
4. The pre-registered rule for the physics weight of QuantilePIDL picked 0.001 on the harmonised
   signals at the price of a 10x worse median (3.1); the rule optimises the pinball loss, which the
   interval quantiles dominate. Keep the rule or add a median-RMSE guard for M5?
5. Harmonisation as specified does not equalise the roughness of the data sets (3.1, finding 5). A
   stronger smoothing of NGSIM (or a roughness-matched filter per data set) would test the
   signal-processing explanation properly; it is a new experiment, not a rerun.
