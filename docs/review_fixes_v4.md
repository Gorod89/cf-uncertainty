# Pre-submission review controls of version v4: analysis (R2, R3, R6, R10, R13, R14)

Date: 2026-10-10. Scope: the control experiments of the v4 revision (`docs/design.md` section 18;
`paper/submission/review_systems/RESPONSE_PLAN_RU.md` items R2, R3, R6 control, R10, R13, R14). The runs (training,
rollouts, platoons) were done before this note (`artifacts/e1_ctrl/logs/chain_v4.out`, "V4 RUNS DONE", 0 failures);
everything below is computed on CPU from the stored prediction, rollout and platoon tables. The compact tables are
`paper/v4/tables/c13_matrix.tex` ... `c18_density.tex` (labels and captions in `paper/v4/generated/LABELS.md`).

## What was run

Each analysis once, on the current code, in four parallel lanes (wall time 28 min; 105 min of CPU-serial time;
`OPENBLAS_NUM_THREADS=5`). Every run exited with code 0 and `summarize_*` wrote no error.

| Step | Roots (arguments) | Runtime per root |
|---|---|---|
| `scripts/analyze_e1.py --quiet` + `analyze_e1_extra.py --config <experiment config>` + `summarize_e1.py` | `e1_ctrl` (`--config configs/experiment/e1_ctrl.yaml --h33-pair lstm_hetero_frozen lstm_const_sigma --h33-pair hetero_mdn lstm_const_sigma`), `e1m_causal` (`--config configs/experiment/e1m_causal.yaml`) | 164 + 43 s, 135 + 61 s |
| same, the roots that gained `idm_ar_noise`, with the arguments of the v3 analysis | `e1`, `e1_mean1s` (maps config `configs/maps/default.yaml`); `e1h`, `e1h_mean1s`, `e1m`, `e1m_mean1s` (`--config`); `e6_pool`, `e6_lodo_openacc`, `e6_lodo_ngsim`, `e6_lodo_ngsim_all` (`--config ... --in-domain pool_test`) | 312-337 s + 120-133 s (E1-E5 roots), 151-275 s + 126-174 s (E6 roots) |
| `scripts/analyze_e2.py --quiet` + `summarize_e2.py` | `e2_ctrl`, `e2`, `e2h`, `e2m`, `e2_pool`, `e2_lodo_openacc` | 36, 390, 397, 399, 164, 94 s |
| `scripts/analyze_e3.py --root artifacts/e3 --quiet` | all 8 variants, 15 300 units (`run_e3.py --dry-run`: raw 2400, harmonised 1800, pool / lodo_openacc / raw_w61 / pool_w11 / lodo_openacc_w11 2100 each, raw_a 600; 0 missing) | 188 s |
| `scripts/make_report.py` | `artifacts/report`: 76 tables (was 72), 300 figures (was 288) | 15 s |
| `paper/v4/tools/make_compact.py` | `paper/v4/tables` (17 generated tables), `generated/`, `figures/` | 9 s |

The driver-cluster runs of v3 (`tables_driver/` of `e1`, `e1m`, `e6_lodo_openacc`, `e6_lodo_ngsim_all`) were not
repeated: `idm_ar_noise` has no epistemic part and hence no H3.1 verdict row, so the follower-cluster table of the
report (`t2_h31_follower_clusters`, built from the verdict rows) gains nothing from it.

**Reproducibility of the previous numbers.** Every table of every rerun root was compared with a snapshot taken before
the runs, on the rows of the models and variants that existed before: identical, or equal to within 2e-15 (relative;
absolute below 1; float summation order), in every column of every table, with two exceptions:

* `regime_ordering.csv`: the Monte-Carlo p-values of the variance ratios of the constant-sigma LSTM (all ratios are
  exactly 1, so the bootstrap p-values are decided by ties) move by at most 0.015 in 7 of the 10 E1 roots; no Holm
  rejection changes (as in v3).
* `artifacts/e2/tables/h34_verdicts.csv`, one row: QuantilePIDL, target `a`, NGSIM US-101, ACI online. The lower bound
  of the corrected coverage at 10 s is exactly the threshold 0.87 in exact arithmetic; it was stored as
  `0.87` and is now `0.8699999999999999` (one ulp), so part (iii) of the interval rule turns from "holds" to
  "straddles" and the interval verdict from supported to inconclusive. The H3.4 interval count of the 13 old labels
  in `e2`, target `a`, ACI online is therefore 32 / 0 / 33 instead of the 33 / 0 / 32 quoted in v3 (point rule
  unchanged, every other row of every E2 root unchanged). `h34_verdict` uses a 1e-12 tolerance for part (i) only; the
  same tolerance on parts (ii) and (iii) would make this tie deterministic (not changed here: analysis only).

Apart from those, the report tables differ from v3 only by the new rows (and, in `t4_follower_clusters`,
`t4_h34_verdict_counts`, by aggregates over the larger model set); the differing "text" cells found by the comparison
are NaN-versus-NaN artefacts of the CSV round trip.

## R2: Diffusion with and without the leader's future (`tab:c14_information`)

Diffusion conditions its 5 s plan on the 3 s history and the leader's speed over the next 5 s (E1, E2: the recorded
leader; platoons: the simulated predecessor); Diffusion, history only (`diffusion_cf_hist`) is the same DDPM without
the leader channel, i.e. the information of the one-step networks. Recorded signals, target `a`, five seeds.

| Level | Quantity | Diffusion | Diffusion, history only | LSTM | HeteroMDN | Deep ensemble |
|---|---|---:|---:|---:|---:|---:|
| one-step | NLL highD / NGSIM I-80 / OpenACC | -2.43 / 16.3 / 0.42 | -2.12 / 11.5 / -0.01 | -2.78 / 467 / 279 | -2.93 / 82.5 / 6.76 | -2.97 / 40.7 / 2.79 |
| | CRPS (m/s^2) highD / NGSIM / OpenACC | 0.011 / 0.27 / 0.073 | 0.019 / 0.30 / 0.13 | 0.0079 / 0.31 / 0.098 | 0.0077 / 0.30 / 0.21 | 0.0071 / 0.27 / 0.17 |
| | PICP highD / NGSIM / OpenACC | 0.99 / 0.42 / 0.77 | 0.98 / 0.46 / 0.83 | 0.93 / 0.08 / 0.46 | 0.91 / 0.19 / 0.64 | 0.95 / 0.33 / 0.73 |
| rollouts | raw PICP 1 s / 10 s, highD | 0.60 / 0.89 | 0.72 / 0.91 | 0.22 / 0.42 | 0.19 / 0.32 | 0.21 / 0.47 |
| | raw PICP 1 s / 10 s, NGSIM | 0.33 / 0.81 | 0.42 / 0.79 | 0.03 / 0.08 | 0.05 / 0.15 | 0.12 / 0.33 |
| | ACI online 10 s, highD / NGSIM | 0.89 / 0.89 | 0.90 / 0.89 | 0.90 / 0.89 | 0.90 / 0.88 | 0.90 / 0.89 |
| | CRPS of the gap at 10 s (m), highD / NGSIM | 2.11 / 6.27 | 4.13 / 12.2 | 4.40 / 22.9 | 5.13 / 17.0 | 3.80 / 13.7 |
| | collided by 10 s (%), highD / NGSIM | <0.1 / 12.1 | 0.9 / 23.0 | 0.9 / 32.5 | 1.2 / 29.0 | 0.9 / 28.9 |
| platoons, ACC (47 profiles, 23 files) | amplitude CRPS (m/s; reference 3.47) | 2.30 | 13.2 | 58.1 | - | 23.9 |
| | reduction [profiles] [files], verdict by files | 0.34 [0.09, 0.50] [-0.26, 0.57] inc. | -2.80 [-3.74, -2.12] [-4.65, -1.94] ref. | -15.8 ref. | - | -5.90 ref. |
| | vehicle 1 collided (%) | 8.8 | 44.3 | 49.9 | - | 58.3 |

Reading. Without the leader's future the diffusion model loses its advantage where it mattered: in the platoons it is
refuted for the ACC platoons (CRPS 13.2 against 3.47 m/s; human-driven -15.8, refuted), its first follower collides
in 44 % of the rollouts, 99.8 % of the platoons' first collisions fall in the 30 s constant-speed burn-in (median
6.4 s) and no platoon rollout is collision-free; in single-follower rollouts its 10 s gap CRPS doubles (2.1 -> 4.1 m
highD, 6.3 -> 12.2 m NGSIM) and its NGSIM collisions double (12 -> 23 %). The v3 statement "the diffusion model,
which is conditioned on the leader's future, reduces the platoon CRPS by 47-56 %" must therefore be read as a result
about that information: one-step, the history-only DDPM is somewhat worse in-domain (NLL -2.12 against -2.43, CRPS
0.019 against 0.011 m/s^2) and at least as well calibrated under shift (NGSIM PICP 0.46 and NLL 11.5 against 0.42 and
16.3; the one-step networks 0.08-0.33), but the closed-loop and platoon gains of Diffusion come from conditioning on
the leader's (predecessor's) future, not from the generative model class.

## R3: the platoon matrix of training domain x leader smoothing (`tab:c13_matrix`)

All models on their target-`a` checkpoints (QuantilePIDL on `a` with hold 1 everywhere: in the highD x 11 cell the
same-target run `raw_a`). Diffusion, ACC platoons, reduction against IDM + white noise of the cell [profiles]
[files], verdict by files; first-follower collisions in parentheses:

| Training \ leader window | 11 (profiles file) | 61 (re-smoothed, matched OpenACC window) |
|---|---|---|
| highD | 0.34 [0.09, 0.50] [-0.26, 0.57] inc. (8.8 %) | 0.42 [0.22, 0.55] [-0.08, 0.61] inc. (5.7 %) |
| pooled, OpenACC seen | 0.57 [0.46, 0.64] [0.29, 0.68] supp. (2.5 %) | 0.56 [0.45, 0.64] [0.29, 0.67] supp. (1.6 %) |
| pooled, OpenACC held out | 0.47 [0.39, 0.53] [0.28, 0.56] supp. (3.9 %) | 0.47 [0.39, 0.53] [0.27, 0.55] supp. (2.1 %) |

Effects on Diffusion's ACC reduction, paired over the same draws (estimate [profiles] [files]):

| Effect | OpenACC seen | OpenACC held out |
|---|---|---|
| domain (pooled - highD), window 11 | +0.23 [0.12, 0.39] [0.08, 0.57] | +0.14 [0.00, 0.34] [-0.04, 0.56] |
| domain, window 61 | +0.14 [0.05, 0.27] [0.02, 0.39] | +0.05 [-0.05, 0.20] [-0.09, 0.36] |
| smoothing (61 - 11), highD training | +0.09 [0.04, 0.15] [0.03, 0.22] | (the same cell) |
| smoothing, pooled training | -0.00 [-0.02, 0.02] [-0.03, 0.02] | -0.01 [-0.02, 0.01] [-0.03, 0.01] |
| interaction | -0.09 [-0.16, -0.04] [-0.23, -0.03] | -0.09 [-0.16, -0.04] [-0.23, -0.03] |

Other models: GP residual 0.15 (highD) to 0.23 (pooled), inconclusive in every cell; QuantilePIDL on target `a`
0.09 / 0.07 (highD x 11 / 61, inconclusive; its 1 s-target run is -0.59, refuted) and -0.16 to -0.25 pooled
(inconclusive by files); deep ensemble -3.3 to -9.2 and LSTM -6.3 to -20.5, refuted in every cell, with the first
follower colliding in 48-58 % (ensemble) and 45-61 % (LSTM) of the ACC rollouts in every cell; IDM + AR(1)
0.06-0.16, inconclusive, never colliding. Human-driven platoons: every model refuted or (Diffusion, pooled, by files)
inconclusive in every cell, as in v3.

Reading. The support of H3.2 for Diffusion on ACC platoons with pooled training does not depend on the leader
smoothing (window 11: 0.57 and 0.47, file bounds 0.29 and 0.28; smoothing effect with pooled training 0.00 +- 0.03):
it is a training-domain effect (+0.14 to +0.23 with OpenACC in training; +0.05 to +0.14 with OpenACC held out, whose
file intervals include 0), while the re-smoothed leader helps only the highD-trained model (+0.09, which makes it
"supported" by the profile interval but not by the file interval) - the interaction is -0.09. The v3 caveat "these
comparisons change the training domain, the leader smoothing and, for the quantile network, the target and the hold
at once" is resolved: QuantilePIDL's move from refuted to inconclusive was its target and hold, not the domain; the
one-step networks' first-follower collisions stay at 45-61 % in every cell whatever the domain and the smoothing
(above the 0.5 of the reading of design 17.2 in 2 of 6 cells for the LSTM and 4 of 6 for the ensemble, never near
the 0.25 of a training-domain effect).

## R6: causal inputs and a fixed physical target (`tab:c17_causal`)

Target mean_1s. (a) harmonised signals (`e1h_mean1s`), (b) centred matched signals of E5 (`e1m_mean1s`, the same
checkpoints, trained on harmonised highD, target from the processed speed), (c) causal matched signals
(`e1m_causal`: one-sided Savitzky-Golay filter, windows chosen on the validation folds only, shifted sets evaluated on
their other folds, target from the recorded speed, models retrained on causal highD). NGSIM I-80, NLL (PICP):

| Model | (a) harmonised | (b) matched, centred | (c) matched, causal | highD test (a) = (b) / (c) |
|---|---:|---:|---:|---:|
| LSTM, constant sigma | 383 (0.09) | 1.85 (0.57) | 259 (0.09) | -1.72 / -1.62 |
| HeteroMDN | 80.3 (0.17) | 0.16 (0.75) | 70.0 (0.17) | -1.81 / -1.69 |
| Deep ensemble | 32.8 (0.32) | -0.45 (0.82) | 40.4 (0.23) | -1.84 / -1.73 |
| MC dropout | 10.2 (0.35) | -0.70 (0.88) | 16.4 (0.26) | -1.81 / -1.70 |
| GP residual on IDM | 3.88 (0.60) | 0.97 (0.84) | 5.88 (0.58) | 0.15 / 0.15 |

| Data set | highD | NGSIM I-80 | NGSIM US-101 | Waymo | OpenACC |
|---|---:|---:|---:|---:|---:|
| window (roughness ratio to highD), centred E5 | 11 (1.00) | 101 (1.53) | 101 (1.49) | 101 (1.93) | 61 (1.06) |
| window (roughness ratio), causal | 11 (1.00) | 101 (5.80) | 101 (4.45) | 101 (2.48) | 101 (2.09) |
| persistence RMSE of mean_1s (m/s^2): recorded / harmonised / matched centred / causal (recorded target) | 0.16 / 0.16 / 0.16 / 0.16 | 1.08 / 1.07 / 0.20 / 1.08 | 0.92 / 0.91 / 0.21 / 0.92 | 0.40 / 0.40 / 0.17 / 0.40 | 0.24 / 0.23 / 0.17 / 0.24 |

Reading. The causal variant does not reproduce the E5 drop of over-confidence: on NGSIM I-80 the five models stay at
5.9-259 nats with PICP 0.09-0.58, the level of the harmonised signals (US-101, Waymo and OpenACC likewise), against
-0.70 to 1.85 nats and PICP 0.57-0.88 with the centred matching. The causal filter cannot reach the training
roughness within the window grid (ratios 2.1-5.8 against 1.06-1.93), and the target measured on the recorded speed is
as rough as recorded (persistence RMSE 1.08 m/s^2 on NGSIM against 0.20 for the centred-matched target and 0.16 on
highD). The v3 wording "a sensitivity of the benchmark to the processing, not a causal decomposition" is confirmed and
should be sharpened: the removal of the hundreds of nats requires the two-sided smoothing of inputs and target; with
past-only inputs and a fixed physical target the over-confidence under shift persists, so the 1-4 nats "behavioural
remainder" of E5 is a statement about the smoothed target (the control changes inputs and target together and cannot
split them).

## R10: IDM with AR(1) acceleration noise (`tab:c15_ar`)

Fit on the training residuals of the frozen IDM: rho = 0.99886 per step, sigma_eta = 0.0497 m/s^2 (stationary
1.039 m/s^2 = the white-noise sigma; conditional rms of the training windows 0.0497) with highD training; pooled
training rho 0.99881, sigma_eta 0.0422 (stationary 0.864); OpenACC held out rho 0.99872, sigma_eta 0.0433 (0.855).

| Quantity | IDM + white noise | IDM + AR(1) | GP residual | Diffusion |
|---|---:|---:|---:|---:|
| one-step NLL highD / NGSIM I-80 / OpenACC | 1.41 / 1.90 / 1.42 | -1.64 / 4.51 / 2.20 | 0.22 / 5.75 / 0.66 | -2.43 / 16.3 / 0.42 |
| one-step PICP highD / NGSIM / OpenACC | 0.95 / 0.89 / 0.96 | 0.95 / 0.68 / 0.91 | 0.93 / 0.54 / 0.87 | 0.99 / 0.42 / 0.77 |
| rollout raw PICP 1 s / 10 s, highD | 0.78 / 0.47 | 0.72 / 0.76 | 0.54 / 0.27 | 0.60 / 0.89 |
| rollout raw PICP 1 s / 10 s, NGSIM | 0.47 / 0.38 | 0.21 / 0.33 | 0.19 / 0.17 | 0.33 / 0.81 |
| ACI width at 10 s (m), highD / NGSIM | 22.3 / 17.2 | 34.7 / 156 | 26.0 / 20.0 | 11.8 / 27.6 |
| platoons, highD training: CRPS ACC / human (m/s) | 3.47 / 1.11 | 3.25 / 3.73 | 2.94 / 4.16 | 2.30 / 2.04 |
| reduction ACC [profiles] [files] | - | 0.06 [-0.20, 0.24] [-0.52, 0.31] inc. | 0.15 inc. | 0.34 inc. |
| reduction human-driven | - | -2.36 [-2.95, -1.97] [-2.49, -0.87] ref. | -2.75 ref. | -0.84 ref. |
| pooled (OpenACC seen / held out): reduction ACC | - | 0.16 / 0.15 inc. | 0.21 / 0.19 inc. | 0.56 / 0.47 supp. |
| pooled: reduction human-driven | - | -2.26 / -2.37 ref. | -2.04 / -1.49 ref. | -0.90 / -0.48 inc. (files) |
| growth exponent, common support (measured 0.085): highD / pooled / held out | -0.06 / -0.06 / -0.06 | -0.13 / -0.08 / -0.08 | 0.01 / 0.00 / 0.00 | 0.12 / 0.10 / 0.05 |
| collision-free platoons (%), highD / pooled / held out | 100 / 100 / 100 | 100 / 100 / 100 | 81 / 44 / 72 | 10 / 20 / 33 |

Reading. The correlated noise makes the one-step law sharp and calibrated in-domain (NLL -1.64 against +1.41) and
lifts the raw 10 s rollout coverage on highD from 0.47 to 0.76, but it transfers worse (NGSIM one-step PICP 0.68,
raw 1 s rollout coverage 0.21, ACI width 156 m at 10 s, 2 % collisions by 10 s) because its conditional law follows
the rough backward difference of the shifted speed. In the platoons it is not a stronger reference: ACC CRPS 3.25
against 3.47 m/s (reduction 0.06, inconclusive), human-driven 3.73 against 1.11 m/s (refuted), and it damps the
speed variability along the platoon more than white noise (exponent -0.13 against measured 0.085). The H3.2 verdicts
against IDM + white noise therefore stand; as a sensitivity (not the protocol's reference), against IDM + AR(1) the
ACC reduction of Diffusion would be 0.29 [files -0.03, 0.50] (highD), 0.48 [0.31, 0.59] (pooled) and 0.38
[0.18, 0.53] (held out).

## R13: H3.3 with the frozen mean (`tab:c16_frozen`)

`lstm_hetero_frozen` = the constant-sigma LSTM of the same seed with mean and features frozen and a learned sigma(x).
highD test, NLL (nats); paired differences against the constant-sigma model [95 % event bootstrap], Holm over all
data sets, regimes and both pairs:

| Regime (share) | constant sigma | frozen mean, sigma(x) | HeteroMDN | frozen - constant (p Holm) | HeteroMDN - constant (p Holm) | frozen: variance / squared residual ratio to steady |
|---|---:|---:|---:|---:|---:|---:|
| congested (2.4 %) | -2.27 | -2.44 | -2.07 | -0.17 [-0.26, -0.10] (0.005) | +0.20 [0.10, 0.28] (0.005) | 2.50 / 2.61 |
| braking (5.7 %) | -1.97 | -2.33 | -2.29 | -0.36 [-0.62, -0.20] (0.005) | -0.33 [-0.57, -0.17] (0.005) | 9.07 / 3.44 |
| approaching (21.1 %) | -2.80 | -2.82 | -2.90 | -0.03 [-0.05, 0.01] (0.41) | -0.10 [-0.13, -0.07] (0.005) | 1.14 / 1.22 |
| free (22.8 %) | -2.92 | -2.98 | -3.09 | -0.06 [-0.07, -0.05] (0.005) | -0.17 [-0.18, -0.16] (0.005) | 0.86 / 0.89 |
| steady (24.6 %) | -2.88 | -2.91 | -3.02 | -0.03 [-0.05, 0.00] (0.41) | -0.14 [-0.16, -0.12] (0.005) | 1 / 1 |
| other (23.4 %) | -2.79 | -2.82 | -2.94 | -0.03 [-0.04, -0.03] (0.005) | -0.15 [-0.16, -0.14] (0.005) | 1.22 / 1.25 |
| all windows | -2.78 | -2.84 | -2.93 | -0.059 [-0.079, -0.042] (0.005) | -0.145 [-0.163, -0.130] (0.005) | - |

Reading. At an identical mean, a state-dependent sigma(x) alone gains 0.059 nats (41 % of HeteroMDN's 0.145), with
its variance ordered by regime like the squared residuals (all five ratios differ from 1); the remaining 0.086 nats
of H3.3 come from HeteroMDN's own mean and backbone, and the congestion exception of v3 (HeteroMDN +0.20 worse) is
not caused by a state-dependent variance: with the frozen mean, sigma(x) is better there (-0.17). H3.3 holds as "state-dependent
heteroscedasticity" at equal mean, with the smaller gain; under shift sigma(x) cannot repair the mean (NGSIM I-80 NLL
103 against 467 for the constant model and 82 for HeteroMDN).

## R14: density sensitivity of Diffusion's sampled law (`tab:c18_density`)

`artifacts/e1/tables/density_sensitivity.csv` (seed 0, 10 000 windows per data set, S = 16 / 32 / 64 nested
first-step samples, KDE bandwidth 0.5 / 1 / 2 x Silverman):

| | highD test | NGSIM I-80 |
|---|---|---|
| mean NLL over the nine cells | -2.23 to -2.43 (stored -2.40) | 4.0 to 65.4: 59-65 (0.5x), 15.0-16.4 (1x), 4.0-4.25 (2x); stored 14.7 |
| median NLL | -2.52 to -2.31 | 0.46 to 1.16 |
| CRPS plain / fair (m/s^2) | 0.0111-0.0120 / 0.0108 | 0.285-0.290 / 0.283-0.284 |
| PICP of the 90 % interval | 0.97-0.99 | 0.40-0.45 |

Reading. In-domain the KDE NLL is robust (0.2 nats over all cells); under shift the mean KDE NLL is set by the
bandwidth (a factor 15 between 0.5x and 2x, through a few windows: the medians are 0.5-1.2 nats) while CRPS and
PICP do not move: Diffusion's NLL under shift (e.g. 16 nats on NGSIM I-80) is not a robust score and the comparison
of sampled laws under shift should rest on CRPS and coverage, as the v4 text proposes (no number of v3 changes).

## Verdicts of the protocol

No verdict of v3 changes because of the new models or variants (all old rows identical), except the floating-point
tie of the H3.4 interval rule above (one of 65 cases in `e2`, target `a`, ACI online). New verdicts:

* H3.1 / H3.1e: `diffusion_cf_hist` and `lstm_hetero_frozen` (across-seed decompositions) refuted / supported;
  `e1m_causal`: H3.1 supported for the GP residual (own and across seeds) and the constant-sigma LSTM, refuted for the
  deep ensemble, HeteroMDN and MC dropout, H3.1e supported for all five - the same overall verdicts as
  `e1h_mean1s` and `e1m_mean1s`. `idm_ar_noise` has no epistemic part.
* H3.2 per mode (verdict by profiles / by files): ACC - Diffusion supported / supported in the new pooled cells with
  window 11 (0.57, 0.47), supported / inconclusive with highD training and window 61 (0.42); QuantilePIDL on target
  `a` inconclusive / inconclusive with highD training (0.09, 0.07; refuted with its 1 s target) and refuted /
  inconclusive at pooled x 11 (-0.25); IDM + AR(1) inconclusive in every cell; Diffusion, history only refuted.
  Human-driven: every new row refuted (Diffusion pooled x 11: refuted / inconclusive, as pooled x 61). All profiles:
  Diffusion pooled x 11 supported / inconclusive (as pooled x 61), highD x 61 inconclusive.
* H3.3: holds with the frozen mean (-0.059 [-0.079, -0.042] nats; variance ordered by regime).
* H3.4 (ACI online, point rule; interval rule in parentheses): IDM + AR(1), target `a`: 5/5 supported in `e2` and
  `e2_pool` (2 supported, 3 inconclusive), 4/5 in `e2h` and `e2m` (one refuted), 1/2 in `e2_lodo_openacc`; target
  mean_1s: 2/5 supported and 3 refuted in `e2` and `e2h`, 5/5 refuted in `e2m` - premise (ii) fails: its raw
  intervals already cover (highD 0.90-0.94 at every horizon), the first IDM-based model of the set to do so
  in-domain; `diffusion_cf_hist` 3/3 supported (3/3) and `lstm_hetero_frozen` 3/3 supported (1 supported, 2
  inconclusive) in `e2_ctrl`. Counts per root: `artifacts/report/tables/t4_h34_verdict_counts.csv` (70 cases for the
  roots with the AR model).

## Files changed

* `paper/v4/tools/make_compact.py`: tables c13-c18 (read from the experiment roots; the H3.2 cells of c13 are
  recomputed from `profile_index.csv` and checked against `h32.csv` to 1e-9, the one-step and rollout numbers of the
  report roots against the report), IDM + AR(1) in c1, c4-c8, c10-c12 right after IDM + white noise, the history-only
  Diffusion excluded from the platoon tables c5, c6, c10, c11 (it is in c14), v4 paths and refusal of `paper/v3` and
  `paper/submission`.
* `cf_unc/report/figures_e3.py`: labels of the four R3 variants and QuantilePIDL's one-step label in them
  (`ONE_STEP_VARIANTS`, `VARIANT_LABEL`, `VARIANT_ORDER`); labels only.
* Rewritten: `artifacts/*/tables` (+ `summary.md`, figures of E2 and E3), `artifacts/report`, `paper/v4/{tables,
  generated, figures}` (the E2 and E3 report figures copied into `paper/v4/figures` now include IDM + AR(1) and, in the
  recorded-signal platoon figures, the history-only Diffusion; `paper/v3/figures` keeps the v3 versions).
* Build check: every compact table c1-c18 compiles alone with the v4 class (Tectonic 0.16.9 offline, the route of
  `paper/v4/build.sh`: option pdftex dropped, EPS as draft frames): no error, no overfull or underfull box, no
  undefined reference. `pytest -q -m "not slow" tests/test_make_report.py`: 9 passed; with the E3 and figure tests
  touched by the label change (`test_e3_pooled`, `test_e3_runner`, `test_review_fixes_v3`, `test_review_controls`,
  `test_figures`): 49 passed.
