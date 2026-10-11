# Epistemic and aleatoric uncertainty in learned car-following: state-space maps, dataset shift, and calibration under closed-loop rollouts

Final report of the research pipeline `cf-uncertainty` (plan P3 / G3), milestone M5 with the follow-up
experiments E5 (roughness-matched signals), E6 (pooled-domain training) and E3-pooled (platoons with
the pooled-domain models), 2026-10-08.
Tables T1-T6 and figures F1-F7 referred to below are built by `scripts/make_report.py` into
`artifacts/report/` (`index.md` lists every file with its source); the milestone reports
`docs/m1_report.md` ... `docs/m4_report.md` hold the full commands and the intermediate numbers,
`docs/design.md` the binding conventions. Units are SI, time steps 0.1 s, `dv = v - v_lead`
(positive when closing in), `s` the net gap. Unless stated otherwise, numbers are means over
5 training seeds with 95 % bootstrap intervals over events (one-step and rollout experiments) or
over leader profiles (platoon experiment).

## 0. Verdicts

| Hypothesis | Statement (plan) | Verdict | Decisive numbers |
|---|---|---|---|
| H3.1 | Under dataset shift the epistemic variance grows and concentrates in rare states while the aleatoric variance stays within 2x | **refuted as written** | epistemic variance up 4-800x (geometric 4-240x) and concentrated 3-115x in rare states; aleatoric variance of every model that learns it up 3-17x (geometric), up to 1000x (mean) |
| H3.1e | The epistemic half of H3.1 alone (restated at the M3 review) | **supported** for every model, both targets, both signal variants | epistemic ratio >= 3 on all counted shift pairs, rare-state concentration > 1 with lower bounds >= 3, placebo pair 1.0-1.1 |
| H3.3 | A regime-aware (heteroscedastic) law beats a constant-variance law in-domain | **supported** | HeteroMDN against LSTMConstSigma on highD test: NLL -2.93 vs -2.78, gain -0.145 [-0.163, -0.130] nats; predicted variance 6-190x the steady-following value in congestion, approaching and braking, 0.75x in free flow |
| H3.4 | Raw rollout intervals under-cover; a one-step conformal and a per-horizon adaptive correction restore nominal coverage | **supported** for every model, both targets, all 5 data sets | one-step conformal 0.897-0.910 in-domain; raw 90 % gap interval covers 0.03-0.78 at 10 s; ACI online 0.88-0.92 at every horizon on every data set, at the price of 17-284 m wide intervals under shift |
| H3.2, ACC platoons (47 profiles) | Distributional models beat IDM + white noise on the CRPS of platoon speed amplitudes | **not supported**: DiffusionCF inconclusive, every other learned model refuted | DiffusionCF reduces the CRPS by 0.34 [0.09, 0.50] (harmonised signals 0.40 [0.19, 0.53]) against the support threshold 0.20 on the lower bound; GP residual 0.15 [-0.08, 0.35]; QuantilePIDL, deep ensemble, LSTM worse than the reference |
| H3.2, human-driven platoons (12 profiles) | same | **refuted** for every learned model | the reference reaches 1.1 m/s CRPS; DiffusionCF -0.84 [-1.26, -0.47], GP residual -2.75, QuantilePIDL -7.0 |
| H3.2, pooled (60 profiles, secondary) | same | not supported | DiffusionCF 0.25 [0.01, 0.44] (harmonised 0.31 [0.10, 0.46]), GP residual -0.06 [-0.34, 0.16], the rest refuted |
| E5 (follow-up): is the over-confidence under shift a roughness artefact? | design.md 15.2 | **mostly yes**: partial explanation dominated by roughness | with the shifted sets smoothed to highD's roughness (no retraining) the one-step NLL on NGSIM falls from 40-470 nats to 1-4 nats above in-domain, PICP 0.6-0.98; rollout collisions 18-28 % -> 9-21 % (in-domain 0.6-2 %) |
| E3-pooled (follow-up): do the pooled-domain models change the platoon verdicts? | design.md 17.2 | **H3.2 supported for DiffusionCF on ACC platoons**; the one-step networks' instability is a property of the models | DiffusionCF reduction 0.47 [0.39, 0.53] with OpenACC never seen, 0.56 [0.45, 0.64] pooled; human-driven platoons still refuted; LSTM and ensemble collide at the first follower in 45-59 % whatever the training domain |
| E6 (follow-up): is the remainder the narrowness of the training domain? | design.md 16.2 | **largely yes**: a third to two thirds of it; about 1 nat is site-specific | trained on four data sets (matched signals), the transfer to the held-out fifth costs 0.2-1.0 nats with PICP 0.83-0.98 (NGSIM I-80 with US-101 in training, and OpenACC); with both NGSIM sets unseen 0.5-1.9 nats at PICP 0.70-0.90, and the ensembles' epistemic ratios on the unseen site are 0.5-1.6; the constant-sigma LSTM stays at 59 nats on held-out OpenACC; pooled training over-disperses (0.93-1.00 on its own folds) and does not fix the 5-25 % rollout collisions |

The verdict thresholds are those pre-registered in `docs/design.md` (H3.1: factor 2 / factor 3 on
bootstrap bounds; H3.4: 0.90 +- 0.03, raw < 0.80, corrected >= 0.87; H3.2: lower bound >= 0.20
supports, upper bound < 0.05 refutes, >= 5 profiles per subset). The per-mode statement of H3.2
is the decision of the M4 review (design.md 14.1).

## 1. Data

| Config | Event set | Events | Role |
|---|---|---|---|
| `highd` | FollowNet highD events (stand-in for raw highD, which needs a licence application) | 12 512 | training (folds 2-4), validation (fold 1), in-domain test (fold 0); driver-disjoint folds of cf-stability |
| `ngsim` | NGSIM I-80, cf-stability's own reconstruction from the raw trajectories | 7 651 | shift |
| `ngsim_us101` | NGSIM US-101, same reconstruction (added at the M2 review) | 7 353 | shift |
| `ngsim_fn` | FollowNet NGSIM I-80 | 1 937 | cross-check of `ngsim` |
| `waymo` | Waymo pairs of Hu et al. (2022) | 967 | shift |
| `openacc` | OpenACC (AstaZero, ZalaZone, Vicolungo, Casale) | 4 715 | shift; the raw files also give the platoon profiles of E3 |

Events are car-following episodes at 10 Hz with `(s, dv, v, a, v_lead)`; windows hold 3 s of
history (30 samples). Two one-step targets: `a` = the recorded one-step acceleration
(Savitzky-Golay in the source pipelines) and the behavioural target `mean_1s = (v[t+10] - v[t]) / 1 s`
(added at the M2 review). Two signal variants: the signals as recorded by the source pipelines
(headline) and one harmonised processing for every data set (M4: Savitzky-Golay (11, 2) on the
speeds, acceleration as the central difference of the smoothed speed, kinematically consistent gap;
drops 0.02-0.3 % of the events, 6.7 % of OpenACC). 60 OpenACC leader profiles (30-120 s, 3-10
measured followers, leader amplitude 2.2-14.5 m/s; 47 ACC, 12 human, 1 mixed) drive the platoon
experiment.

## 2. Models

| Model | Predictive law | Aleatoric | Epistemic |
|---|---|---|---|
| HeteroMDN (`hetero_mdn`, `hetero_mdn_mix3`) | LSTM(64) over 3 s -> Gaussian or 3-component mixture | own variance | across-seed decomposition only |
| DeepEnsemble | 5 HeteroMDN members, seeds + driver bootstrap | mean member variance | variance of member means |
| MCDropout | HeteroMDN, dropout 0.1, 50 passes | same | same |
| GPResidualIDM | calibrated IDM mean + sparse variational GP residual (RBF-ARD on (s, dv, v), 300 inducing points) | learned noise (constant) | GP posterior variance |
| QuantilePIDL | pinball quantiles {0.05, 0.25, 0.5, 0.75, 0.95} + IDM physics term + non-crossing penalty; physics weight 0.0001 (recorded) / 0.001 (harmonised) selected on validation | interpolated quantile law | across seeds |
| DiffusionCF | conditional DDPM over 5 s acceleration sequences, 50 DDIM steps, 32 samples | sample variance | across seeds |
| IDMWhiteNoise (reference) | IDM calibrated closed-loop on highD + Gaussian noise fitted on residuals; `idm_white_noise_mle` = one-step maximum-likelihood calibration | sigma^2 | 0 |
| LSTMConstSigma | deterministic LSTM + constant sigma | sigma^2 | across seeds |

All models are trained on highD (5 seeds; 458k windows for the one-step models), evaluated on
every data set; the sanity experiment E4 on the StochasticIDM generator recovers the generator's
noise variance within 15 % for every model and shows the epistemic variance shrinking with data
for every model but MC dropout (M1, `xfail` test).

## 3. Experiments and metrics

* **E1** maps and shift (H3.1, H3.1e, H3.3): prediction tables of 10 model configurations x 5 seeds
  on 7 data sets per target and signal variant (`artifacts/e1`, `e1_mean1s`, `e1h`, `e1h_mean1s`);
  maps on the grid s 0-120 m x dv -5..5 m/s (5 m x 0.5 m/s), rare bins by frequency (< 1 %) and by
  mass (lowest 10 %), regimes congested > braking > approaching > free > steady > other.
* **E2** rollouts and calibration (H3.4): 200 stochastic closed-loop rollouts of 500 events per data
  set, horizons 1/3/5/10 s, (S + 1) plotting-position quantiles, split conformal on the one-step
  residuals, per-horizon static correction, ACI frozen and online (gamma 0.01); 780 rollout
  tables per signal variant (`artifacts/e2`, `e2h`).
* **E3** platoons (H3.2): 20 followers behind each of 60 measured OpenACC leaders, 100 rollouts
  (DiffusionCF 32), 3 s history + 30 s burn-in + profile + 40 s tail, IDM-equilibrium initial
  gaps, cascade through the rollout engine; CRPS of the amplitude `max(v) - min(v)` of every
  measured follower, growth of the speed standard deviation along the platoon, collisions,
  amplitude-interval coverage; 1800 units per signal variant (`artifacts/e3`); sensitivities
  (speed cap 35.9 m/s, per-step ensemble draws) on the headline set (`e3_vcap`, `e3_step`).
* **Metrics**: NLL, CRPS (closed form for Gaussian and mixture, sample based otherwise), PICP and
  width of the 90 % interval, PIT/KS, driver or event bootstrap (1000), paired tests with Holm.
* **Tests**: 489 fast tests and 24 slow / GPU tests (decomposition identity, CRPS closed form against
  samples, ACI on a drifting series, collision detection, cascade against a joint simulation,
  harmonisation invariants, E4 sanity).

## 4. Results by hypothesis

### 4.1 H3.1 and H3.1e: uncertainty maps and dataset shift (F1, F2, T2)

The maps (F1) show the epistemic variance of the deep ensemble and of the GP residual concentrated
at the edge of the visited state space: large `|dv|`, large gaps, and the few congested cells;
the aleatoric maps follow the braking and approaching regions. On highD test the share of the
epistemic variance that falls into the rare bins (frequency < 1 %, 34 % of the windows) is 0.88
for the deep ensemble (ratio inside / outside 14.7 [5.4, 27.4]), 0.98 for the GP residual (84
[60, 116]) and 0.58 for MC dropout (2.7 [2.4, 3.0]); with the mass rule (lowest 10 % of the mass)
21.6, 115 and 4.0; in the three-dimensional (s, dv, v) rarity 54, 136 and 4.5.

Under shift (highD -> NGSIM I-80, US-101, Waymo, OpenACC) the epistemic variance grows for every
model: deep ensemble 24-806x by the ratio of means (geometric 21-238x), MC dropout 5-50x, GP
residual 3.8-11x (geometric 4.5-9.2x), with placebo ratios (highD validation / test) of
1.03-1.10. The aleatoric variance grows as well: 8.6-1140x for the deep ensemble (geometric
5.9-9.2x), 11-135x for MC dropout (geometric 6.3-13x), 3-17x by geometric mean for every model
whose aleatoric variance is learned. H3.1 as written, which requires the aleatoric variance to
stay within a factor 2, is therefore refuted for every learned model; it is "supported" only for
the GP residual and the LSTM baseline, whose aleatoric variance is a constant by construction.
At matched (s, dv, v) cells the GP ratio drops to 1.2-1.6 (its growth is a change of the state
distribution), the ensemble ratio stays at 12-187: for history-based models the shift sits in the
fine structure of the input signal, not in the state.

H3.1e, the epistemic half restated at the M3 review, is supported for every model and
decomposition kind on both targets and both signal variants: epistemic ratio >= 3 on the four
counted pairs (the GP's own posterior on 2 of 4) and concentration in the rare states of the
shifted sets 4-115x with lower bounds >= 3; the cross-check pair (FollowNet NGSIM I-80) agrees.
Everything above is unchanged on the harmonised signals (T2, both variants side by side).

### 4.2 H3.3: regime-conditional variance (F3, T3)

On highD test HeteroMDN beats LSTMConstSigma by 0.145 nats (NLL -2.93 [-2.96, -2.89] against
-2.78 [-2.81, -2.75]; the intervals do not overlap; Holm p = 0.005 in every regime). Its
predicted variance, relative to steady following, is 187 [40, 391] in braking, 19 [1.2, 54] in
approaching, 6.1 [5.1, 7.4] in congestion and 0.75 [0.72, 0.79] in free flow; the squared
residuals of the data give the same ordering (63, 5.7, 7.0, 0.87). The one exception is
congestion, where the constant-sigma law has the better NLL (+0.197 [0.101, 0.280]). With the
behavioural target the gain shrinks to 0.038 nats and the regime intervals overlap; on the
harmonised signals it grows to 0.221 nats (target `a`) and 0.093 nats (`mean_1s`). Under shift
the heteroscedastic model is the better of the two on every data set (by 21-384 nats), both being
miscalibrated.

### 4.3 H3.4: coverage in closed-loop rollouts and its correction (F4, T4)

Part (i): the one-step split conformal correction calibrated on highD validation puts every model
at 0.897-0.910 coverage on highD test (nominal 0.90, tolerance 0.87-0.93). Under shift it does not
help (NGSIM: 0.07-0.78), because the residuals are not exchangeable across data sets.

Part (ii): the raw 90 % interval of the gap in rollouts under-covers at every horizon for every
model: on highD test 0.78 -> 0.47 (IDM + white noise) and 0.22 -> 0.42 (LSTM) from 1 s to 10 s
with target `a`; DiffusionCF is the only model near nominal from 3 s on (0.94 -> 0.89). Under
shift the history-based models cover 0.03-0.15 at 1 s and collide with the leader in 18-33 % of
the NGSIM rollouts by 10 s; the IDM-based models never collide.

Part (iii): the per-horizon adaptive correction (ACI online, gamma 0.01) restores 0.880-0.918 at
every horizon on every data set and for every model (static and frozen corrections reach
0.88-0.96 in-domain and fail under shift, 0.05-0.60). The price is the width: the median 10 s
interval grows from 6 to 22 m in-domain for IDM + white noise and from 3.5 to 284 m under shift
for the LSTM, i.e. the corrected interval says that the LSTM's prediction of the gap at 10 s on
NGSIM is worthless, which is true. Counting 13 model labels x 5 data sets: 65 of 65 supported by
the point rule for either target (32 and 29 when every part must hold on its interval bound, the
rest inconclusive, none refuted). The harmonised replication gives 125 of 130 by the point rule;
the five "refuted" cases (DiffusionCF on highD for both targets, kept-mask MC dropout on highD)
are models whose raw intervals already cover 0.89-0.94, so the premise of H3.4 does not hold for
them and the correction keeps 0.89-0.90. The ACI step changes little: coverage 0.867-0.932 for
gamma 0.005, 0.894-0.907 for 0.05, the width growing with gamma under shift.

### 4.4 H3.2: platoon-level distributional evaluation per driver mode (F5, F7, T5)

Amplitude CRPS of the pooled law of 500 rollouts (100 x 5 seeds) against the measured amplitude
of every follower, averaged over the measured indices of a profile and over profiles; reduction
`1 - CRPS_model / CRPS_reference` paired by profile, 95 % bootstrap over profiles; recorded
signals, extended set, window profile + 40 s tail (harmonised in brackets):

| model | ACC (47 profiles): CRPS, reduction, verdict | Human (12): CRPS, reduction, verdict | all (60): reduction |
|---|---|---|---|
| idm_white_noise (reference) | 3.47 m/s | 1.11 m/s | - |
| diffusion_cf | 2.30; 0.34 [0.09, 0.50] (0.40 [0.19, 0.53]); inconclusive | 2.04; -0.84 [-1.26, -0.47]; refuted | 0.25 [0.01, 0.44] (0.31 [0.10, 0.46]) |
| gp_residual_idm | 2.94; 0.15 [-0.08, 0.35] (0.14 [-0.10, 0.33]); inconclusive | 4.16; -2.75 [-4.19, -1.85]; refuted | -0.06 [-0.34, 0.16] |
| quantile_pidl (target `mean_1s`) | 5.53; -0.59 [-1.03, -0.28]; refuted | 8.90; -7.0 [-8.5, -6.1]; refuted | -1.09 [-1.69, -0.64] |
| deep_ensemble (kept member) | -5.9; refuted | -23; refuted | -7.2 [-10.8, -4.9] |
| lstm_const_sigma | -15.8; refuted | -50; refuted | -18.4 [-24.5, -13.6] |

On the ACC platoons DiffusionCF is the only model that improves on the reference, by a third of
the CRPS, but the lower bound of its reduction (0.09, harmonised 0.19) stays under the support
threshold 0.20: H3.2 is inconclusive for it and refuted for the others. On the human-driven
platoons the IDM reference itself is good (1.1 m/s) and every learned model is refuted. The one
mixed profile is listed and not judged. Small subsets mislead: on the 6 ACC profiles of the headline
set the GP residual is formally "supported" (0.57 [0.34, 0.74]; T5, sensitivities), which the 47 ACC
profiles of the extended set do not confirm (0.15). Against LSTMConstSigma as the reference every other model
"wins" by 0.58-0.96, which only restates that the LSTM runs away.

Why the LSTM-based models fail: the profiles start at 7-29 m/s (median 13.6 m/s) with
IDM-equilibrium gaps of 7-82 m (median 11 m), mostly low speeds and short gaps that highD covers
thinly (4 % of its windows are below 8 m/s). From there the kept-member ensemble collides within the 30 s constant-speed burn-in in
27 % of the rollouts, and beyond the training gaps the networks command +4 m/s^2 without limit
(speeds above 500 m/s, through the leader). The share of rollouts in which the first follower
collides is 0.49-0.60 for QuantilePIDL, LSTM and ensemble, 0.08 for DiffusionCF, 0.04 for the GP
residual, 0 for IDM; by the 20th vehicle 0.82-0.92, 0.54, 0.02 and 0 (T6). With a physical speed
cap of 35.9 m/s (the largest highD speed; sensitivity on the headline set) the LSTM-based models
become "inconclusive" (CRPS 3.5-4.2 against 3.1 m/s) but still collide in 84-94 % of the
rollouts; the ensemble with per-step draws halves the kept member's CRPS (14.0 -> 6.9 m/s) and
its burn-in collisions (0.27 -> 0.11) and remains worse than the reference. The LSTM-based models
swap places between the signal variants (LSTM 58 -> 12 m/s, ensemble 24 -> 51 m/s): their runaway
is a property of the training run, not of the signals.

Growth along the platoon (F6, T6): the measured platoons' speed standard deviation grows with the
vehicle index as `i^0.12` (0.12 [0.08, 0.16], fitted over the indices that at least half of the
profiles measured; 3.5 -> 4.2 m/s from the first to the fourth follower). IDM + white noise is
flat (-0.01 [-0.06, 0.03]), the GP residual damps (-0.19 [-0.26, -0.12] over the profile
window), DiffusionCF grows faster than the data (0.19 [0.15, 0.24], 3.6 -> 5.9 m/s) and is the
only learned model with the right sign; part of what it reproduces is the growth of its own
instability along the platoon (collisions 8 % -> 54 %). The raw 90 % amplitude intervals of the
IDM-based models are far too narrow (coverage 0.20-0.30, median width 2-3 m/s against 15-24 m/s
needed): their noise averages out over a profile while the measured amplitudes vary between
profiles by much more; the leave-one-profile-out conformal correction restores 0.90 by
construction, ACI over the 60 profiles reaches 0.81-0.87.

### 4.5 Replication with one harmonised signal processing (T1, T2, T4, T5)

Every verdict is unchanged. With the same Savitzky-Golay window on every data set, the NGSIM
reconstructions stay 10x rougher than highD (speed roughness 0.019 against 0.0018; the window of
1.1 s does not undo their resampling), and the one-step numbers repeat almost exactly: LSTM NLL
-2.78 -> -2.86 on highD and 467 -> 397 on NGSIM I-80, deep ensemble -2.97 -> -3.08 and 41 -> 38,
PICP on NGSIM 0.08-0.54 on both variants, in-domain RMSE 0.01-0.04 m/s^2 for every history-based
model. The over-confidence under shift is therefore not an artefact of different processing
pipelines; it is the models' confidence in the smoothness of whatever signal they were trained
on.

### 4.6 Follow-up E5: shifted signals smoothed to the training roughness (`artifacts/e1m`, `e1m_mean1s`, `e2m`; `docs/m6_report.md`)

Every shifted data set was smoothed with the Savitzky-Golay window that brings the roughness of
its speed signal closest to harmonised highD (OpenACC 6.1 s, the NGSIM sets and Waymo 10.1 s, the
largest of the grid: NGSIM ends at 1.4-1.5x, Waymo at 1.9x highD's roughness), the `e1h`
checkpoints were evaluated again without retraining and the rollouts repeated. The persistence
RMSE of the matched one-step target is 1.05-1.5x highD's, so the task is about as hard as
in-domain. Result (NGSIM I-80, target `a`, recorded -> matched): LSTM NLL 467 -> 0.63 (in-domain
-2.86), PICP 0.08 -> 0.61; HeteroMDN 82 -> -1.03 (-3.08), 0.19 -> 0.76; deep ensemble 41 -> -1.48
(-3.08), 0.33 -> 0.88; MC dropout 4.0 -> -1.50 (-2.43), 0.52 -> 0.98; QuantilePIDL 102 -> -0.13
(-2.92), 0.16 -> 0.74; DiffusionCF 16 -> -1.45 (-2.48), 0.42 -> 0.91; GP residual 5.8 -> 1.10
(0.22), 0.54 -> 0.82. The behavioural target gives the same picture (LSTM 372 -> 1.85, deep
ensemble 34 -> -0.45, QuantilePIDL 78 -> 4.29). In rollouts the 1 s coverage under shift returns
to the in-domain level, the ACI intervals at 10 s shrink to in-domain widths (LSTM 188 -> 27 m
against 29 m on highD) and the collisions of the LSTM-based models fall from 18-28 % to 9-21 %
(in-domain 0.6-2 %). By the reading fixed in advance (design.md 15.2) the over-confidence under
shift was mostly a roughness artefact of the source pipelines, with a behavioural remainder of
1-4 nats (6 for the quantile model on the behavioural target) and of 9-21 % collisions. OpenACC is
the exception: its roughness was close to highD's, the LSTM's NLL stays at 294 there, and the
epistemic and aleatoric ratios stay in the hundreds; the ACC followers at low speed are a shift
of behaviour and state. H3.1 stays refuted and H3.1e supported on the matched signals, with the
NGSIM ratios shrunk to 2-8x (aleatoric of MC dropout 0.3: smoothed NGSIM looks less noisy than
highD to it).

### 4.7 Follow-up E6: training on the matched signals of a pooled domain (`artifacts/e6_pool`, `e6_lodo_ngsim`, `e6_lodo_openacc`, `e2_pool`, `e2_lodo_*`; `docs/m7_report.md`)

The ten model configurations were trained (5 seeds, target `a`, matched signals, the highD IDM
of E5 as a frozen prior, 150 000 windows per data set) on the training folds of all five data
sets and, in two leave-one-out roots, on four of them with NGSIM I-80 or OpenACC held out. On the
pooled root every test fold is within 1 nat of the pooled in-domain NLL and no fold is
under-covered; the networks over-cover (0.93-1.00), the IDM-backed models sit at 0.87-0.93. Held
out, NGSIM I-80 costs 0.2-0.7 nats of transfer gap with PICP 0.83-0.98 (highD-only training on
the same signals: 0.9-3.5 nats, 0.61-0.98; caveat: US-101 stays in training, and the never-trained
FollowNet I-80 set agrees); held-out OpenACC costs 0.0-1.0 nats with PICP 0.93-1.00 for every
heteroscedastic model, while the constant-sigma LSTM stays at 59 nats (297 with highD-only
training). In rollouts on held-out NGSIM the 1 s coverage rises to 0.84-0.99 and the collisions
of HeteroMDN and the ensemble fall to 7-8 %, but the LSTM (8 %) and MC dropout (25 %) still
collide; ACI online keeps 0.88-0.94 everywhere. The epistemic ratios on held-out OpenACC stay in
the tens to hundreds (a state shift the ensembles still see) and are 0.4-2.4 on held-out NGSIM;
the regime gain of H3.3 shrinks to 0.10 [0.07, 0.13] nats on the pooled test folds and stays
significant. A fourth root with both NGSIM sets unseen (training on highD, Waymo and OpenACC;
`e6_lodo_ngsim_all`) puts the true NGSIM domain hold-out at 0.5-1.9 nats (median about 1) with
PICP 0.70-0.90: a third to two thirds of the highD-only gap was domain narrowness, the rest is
NGSIM-specific; on that unseen site the epistemic ratios are 0.5-1.6 for every model (H3.1e
refuted there): matched signals make an unseen highway site look familiar to the ensembles while
it still costs about 1 nat. Rollouts on the unseen NGSIM sets collide in 5-13 % of the cases for
the one-step networks and 1 % for DiffusionCF.

### 4.8 Follow-up E3-pooled: platoons with the models of the pooled domain (`artifacts/e3`, variants `lodo_openacc` and `pool`; `docs/m8_report.md`)

The platoon experiment was repeated with the checkpoints of the root that holds OpenACC out and of
the pooled root (target `a` for every model, leader profiles smoothed with the matched OpenACC
window). The diffusion model's reduction of the amplitude CRPS on the 47 ACC platoons rises from
0.34 [0.09, 0.50] (highD only) to 0.47 [0.39, 0.53] with OpenACC never seen and 0.56 [0.45, 0.64]
with OpenACC in training (CRPS 2.30 -> 1.88 -> 1.54 m/s against a reference of 3.5 m/s): H3.2 is
supported for it on ACC platoons and on the pooled 60 profiles (0.40 and 0.46), and stays refuted on
the human-driven platoons (-0.48, -0.90). Its first-follower collisions fall from 8 % to 1-2 % and
its growth exponent moves to 0.10-0.17 against the measured 0.12. The GP residual stays at the
reference (0.19-0.21, inconclusive). The LSTM and the kept-member ensemble collide at the first
follower in 45-59 % of the rollouts whatever they were trained on (recorded signals 52-60 %): by
the reading fixed in advance their closed-loop instability is a property of the models; the quantile
network improves to 21 % at the first follower (69-78 % by the twentieth) and moves from refuted to
inconclusive.

## 5. Findings

1. **The one-step target at 10 Hz is a persistence task.** Predicting `a[t-1]` has RMSE
   0.017 m/s^2 on highD against 0.35 for "a = 0"; no IDM is better than zero; the LSTM models reach
   0.01-0.03 and learn the smoothness of the recorded signal. The signal differs 10x in roughness
   between data sets, which makes every history-based model catastrophically over-confident under
   shift (NLL up to 467 nats, PICP 0.08). The behavioural target removes the persistence shortcut
   in-domain (rollout coverage at 1 s 0.2 -> 0.5) but not the over-confidence under shift.
2. **Epistemic uncertainty behaves as the plan expected; aleatoric does not.** Every method's
   epistemic variance grows under shift and concentrates in rare states (H3.1e), but the learned
   aleatoric variance grows with it, by 3-17x (geometric), because the models attribute the
   rougher shifted signal to noise. MC dropout's epistemic variance is the least sensitive to data
   (ratios 3-5x where the ensemble gives 20-240x) and does not shrink with data on synthetic data.
3. **Regime awareness pays in-domain** (0.145 nats, variance 6-190x in braking, approaching and
   congestion) and is irrelevant under shift, where both laws are miscalibrated by tens of nats.
4. **Adaptive conformal correction is the only calibration that survives shift**, and it tells the
   truth about the models: 0.89-0.92 coverage everywhere, with intervals of 17-284 m at 10 s where
   the model's prediction is uninformative. Static corrections calibrated on highD fail under shift
   because the residuals are not exchangeable.
5. **Distributional one-step models do not transfer to the platoon level as the plan assumed; a sequence model trained on a broad domain does.**
   Over 190 s and 20 vehicles the LSTM-based models (including the physics-informed quantile model)
   leave the training region and collide or run away; only the models with a physical backbone
   (IDM + noise, GP residual on IDM) and the sequence model DiffusionCF stay on the road. On ACC
   platoons DiffusionCF reduces the amplitude CRPS by a third (not enough for the threshold) and is
   the only learned model that reproduces the measured growth of the speed variability along the
   platoon (0.19 against 0.12), while colliding in half of its 20-vehicle rollouts; on human-driven
   platoons the IDM reference is hard to beat. Trained on a pooled domain (4.8), the diffusion model
   reduces the amplitude CRPS by 47-56 % on ACC platoons with 1-2 % first-follower collisions, the one
   learned model for which H3.2 is supported, while the one-step networks still collide in about half
   of their rollouts.
6. **One shared processing changes nothing, matching the roughness changes almost everything** (4.5,
   4.6). The over-confidence of hundreds of nats under shift is the effect of the source pipelines'
   roughness; smoothed to the training roughness, the models are 1-4 nats from calibrated and the
   rollouts return to in-domain coverage. The behavioural shift the paper can claim is that
   remainder, plus OpenACC's state shift. Models used on reconstructed trajectories must match the
   processing of their training data or carry the ACI correction, which handles both cases.
7. **Dataset shift in car-following has three separable parts** (E5 and E6, 4.6-4.7): the
   processing roughness of the source pipelines (hundreds of nats, removed by matching the
   filter), the narrowness of a single training domain (1-4 nats, removed by training on several
   data sets: transfer to a held-out set then costs 0.2-1.0 nats, or about 1 nat for a fully
   unseen site, for every model with a learned variance) and the state shift of a new driving
   population (ACC followers at low speed: the
   epistemic ratios stay in the hundreds, pooled training absorbs it for heteroscedastic models,
   never for the constant-sigma LSTM). Pooled training over-disperses rather than over-confides,
   and leaves the closed-loop instability of the one-step networks (8-25 % collisions) untouched.

## 6. Deviations from the plan

* Data: raw highD could not be obtained (licence application), FollowNet highD events are the
  stand-in (gaps <= 120 m, pre-filtered); NGSIM is cf-stability's own reconstruction from the raw
  trajectories plus the FollowNet set as a cross-check; NGSIM US-101 added at the M2 review;
  OpenACC copied from the cf-stability project; events imported from cf-stability, never re-extracted.
* Models: QuantilePIDL's physics weight is selected on validation (0.0001 recorded, 0.001
  harmonised; the plan's default 0.3 was harmful); DiffusionCF uses 50 DDIM steps and 32 samples
  and is not retrained for the behavioural target (its law of `mean_1s` is the mean of the first
  10 sampled steps); a one-step maximum-likelihood IDM calibration (`idm_white_noise_mle`) and a
  persistence baseline were added because the closed-loop IDM is a poor one-step predictor.
* Maps: the mass-based rare-bin rule (lowest 10 %) and a three-dimensional (s, dv, v) rarity were
  added next to the plan's frequency < 1 %; geometric-mean ratios are reported next to ratios of
  means (heavy tails); the regime label `braking` uses `a[t-1]`, not the target.
* Rollouts: (S + 1) plotting-position quantiles (sample quantiles of 200 rollouts under-cover by
  0.01-0.05); two sampler modes (per-step draws, draw kept per trajectory) and model variants
  `deep_ensemble__traj`, `mc_dropout__traj`, `gp_residual_idm__step`; the one-step conformal
  correction is the in-house 30-line implementation, not MAPIE.
* Platoons: leader speed <= 28.68 m/s (the closed-loop IDM has `v0 = 28.9 m/s`, no equilibrium gap
  above it); windows start and end at steady leader speed; one random stream per batch of
  profiles and vehicle; DiffusionCF's 5 s plan sees the simulated future of its predecessor; the
  measured growth exponent is fitted over balanced indices only (a pooled fit gave -0.75, a
  composition effect of the few long platoons); the speed-cap and per-step sensitivities were
  added after the pilot; the per-mode statement of H3.2 is the decision of the M4 review.
* Harmonisation: speeds floored at 0, drop reason `non_finite`; OpenACC loses 6.7 % of its events
  (above the 5 % mark of the design), reported, not fixed; the pilot on harmonised signals
  (under-trained models) suggested a tenfold smaller over-confidence, which the full runs refuted.
* H3.1 restated as H3.1e at the M3 review (the original verdict is kept next to it).
* Environment: one shared Python environment for all research_plans projects (thin venv); the GPU
  was shared and reset several times, so every long run is resumable behind a retry wrapper.

## 7. Limitations

* One training domain (highD stand-in events, 10 Hz, gaps <= 120 m); all shift results are
  relative to it, and the NGSIM reconstructions carry a resampling roughness that no model in the
  set is trained to expect.
* The one-step target is dominated by signal smoothness; the behavioural target and the harmonised
  signals mitigate but do not remove this, so the one-step NLL and PICP under shift measure the
  signal processing as much as the behaviour.
* Platoon ground truth is 60 OpenACC profiles with 3-10 measured followers, 47 of them ACC; the
  bootstrap intervals of the H3.2 reduction are 0.4 wide, the human subset has 12 profiles, and
  the measured growth curve is balanced only up to the fourth follower. The initial state of every
  follower is the IDM equilibrium of the closed-loop calibration (`v0 = 28.9 m/s`), which excludes
  most Vicolungo highway windows.
* DiffusionCF's cascade is not a joint simulation (its plan sees the predecessor's simulated
  future); the deep ensemble's per-trajectory mode keeps one member per rollout, which the per-step
  sensitivity shows to be the worse of its two modes.
* Five training seeds; the across-seed epistemic decomposition of the single-network models is
  only a proxy; MC dropout's epistemic variance does not shrink with data.
* No experimental platoon series from the literature were entered into `configs/empirical_growth.yaml`
  (the file is a template); the measured growth comes from OpenACC only.
* The roughness matching of E5 smooths the task as well as the inputs (windows of 6-10 s, the
  NGSIM target's persistence RMSE falls 7x); it shows that models trained on smooth signals are
  calibrated on smooth signals, not that the smoothed NGSIM is the true driving signal, and the
  tolerance was reached only for OpenACC.
* E6 uses the highD IDM as a frozen prior for every root (the pooled calibration is degenerate),
  caps every data set at 150 000 windows, holds out one data set at a time (the NGSIM I-80
  hold-out keeps US-101 in training) and does not repeat the platoon experiment.

## 8. Reproduction

```bash
cd cf-uncertainty && export PYTHONUTF8=1
./.venv/Scripts/python.exe -m pytest -m "not slow"                       # 521 fast tests
./.venv/Scripts/python.exe scripts/import_cf_stability_events.py         # data from ../cf-stability
# E1 (both targets, both signal variants), E2, E3: docs/m2_report.md, m3_report.md, m4_report.md, section 1 of each
bash scripts/run_e1_retry.sh artifacts/e1/logs/r1.out 8 --config configs/experiment/e1.yaml
bash scripts/run_e2_retry.sh artifacts/e2/logs/run.out 8 --config configs/experiment/e2.yaml
bash scripts/run_e3_retry.sh artifacts/e3/logs/run.out 8
./.venv/Scripts/python.exe scripts/analyze_e1.py --root artifacts/e1 --quiet && ./.venv/Scripts/python.exe scripts/analyze_e1_extra.py --root artifacts/e1
./.venv/Scripts/python.exe scripts/analyze_e2.py --root artifacts/e2 --quiet
./.venv/Scripts/python.exe scripts/analyze_e3.py --root artifacts/e3 --quiet
./.venv/Scripts/python.exe scripts/match_roughness.py --config configs/experiment/e1m.yaml   # E5 windows
bash scripts/run_e1_retry.sh artifacts/e1m/logs/run.out 8 --config configs/experiment/e1m.yaml  # E5, checkpoints reused
bash scripts/run_e1_retry.sh artifacts/e6_pool/logs/r1.out 8 --config configs/experiment/e6_pool.yaml   # E6 (docs/m7_report.md)
./.venv/Scripts/python.exe scripts/make_report.py                        # artifacts/report/{tables,figures,index.md}
```

Artifact roots: `artifacts/e1`, `e1_mean1s`, `e1h`, `e1h_mean1s` (one-step tables, maps,
verdict tables), `artifacts/e2`, `e2h` (rollout tables, coverage, ACI), `artifacts/e3`, `e3_vcap`,
`e3_step` (platoon units, CRPS, growth, collisions), `artifacts/e1m`, `e1m_mean1s`, `e2m` (E5,
roughness-matched signals), `artifacts/e6_pool`, `e6_lodo_ngsim`, `e6_lodo_openacc`,
`e6_lodo_ngsim_all`, `e2_pool`, `e2_lodo_ngsim`, `e2_lodo_openacc`, `e2_lodo_ngsim_all` (E6,
pooled-domain training), `artifacts/e3` variants `lodo_openacc` and `pool` (E3-pooled), `artifacts/report`
(this report's tables and figures). Raw data and event sets are not committed.

## 9. Addendum after the review (2026-10-10)

A pre-submission review of the manuscript (`paper/submission/review_systems/`) led to two further sets of work,
recorded in `docs/review_fixes_v3.md` (analysis) and `docs/review_fixes_v4.md` (control experiments; design
document section 18), and to the manuscript versions v3 and v4. The findings that change the reading of
sections 4-5 above:

1. **The diffusion model's platoon advantage is the leader's future.** Conditioned on the history alone
   (`diffusion_cf_hist`), the model is refuted on ACC platoons (reduction -2.80, file-cluster interval
   [-4.65, -1.94]), collides at the first follower in 44 % of the rollouts and has no collision-free platoon;
   the result of section 4.8 is one of scenario-conditioned generation, not of the architecture. Among
   history-only laws none beat the IDM reference in platoons.
2. **The drop of over-confidence under roughness-matched signals (section 4.6) belongs to offline centred
   reconstruction.** A causal filter with windows chosen on validation folds and the behavioural target from
   the recorded speed leaves the NGSIM NLL at 259 / 70 / 40 / 16 / 5.9 nats (LSTM, HeteroMDN, ensemble,
   MC dropout, GP residual) against 383 / 80 / 33 / 10 / 3.9 harmonised; for past-only processing the
   over-confidence remains.
3. **The pooled-domain platoon gain is mostly the training domain** (2x2 matrix: domain effect +0.23 / +0.14,
   smoothing +0.09 with highD training and 0.00 pooled, interaction -0.09); the quantile network scored on the
   one-step target is inconclusive (0.09) rather than refuted.
4. **Cluster bootstrap over recording files** (60 profiles from 26 files) widens the platoon intervals 1.5x:
   the diffusion model stays supported on ACC platoons for the pooled-domain variants only; six of sixty
   cells move to inconclusive. Follower-cluster intervals widen the OpenACC one-step intervals 6-8x; H3.1e
   for the deep ensemble on the unseen NGSIM sets becomes inconclusive.
5. **ACI with the unclipped recursion** (the finite-quantile variant of M3 broke the long-run bound on
   drifting streams) changes the reported coverages by at most 0.001; infinite intervals occur in the first
   50-80 events after a switch (at most 5.3 % of a row). Chronological order and delayed feedback change the
   coverage by at most 0.022; a static correction with the same label budget reaches 0.60-1.00.
6. **AR(1) acceleration noise** on the IDM raises the in-domain 10 s rollout coverage from 0.47 to 0.76,
   transfers badly to NGSIM and does not improve the platoon amplitudes (reduction 0.06, inconclusive).
7. **A variance head at a fixed mean** gives -0.059 nats, about 41 % of HeteroMDN's -0.145: H3.3 holds at
   equal mean with a smaller effect.
8. **The NLL of the sample-based diffusion law is bandwidth-sensitive** (59-65 / 15-16 / 4.0-4.3 nats on
   NGSIM at 0.5 / 1 / 2 x Silverman) while its CRPS and coverage are stable; the CRPS is the comparable score.
9. Growth exponents on the common support (54 profiles, indices 1-4): measured 0.085 [0.054, 0.119],
   diffusion 0.117 / 0.047 / 0.096; the 1-20 fits are extrapolations.
