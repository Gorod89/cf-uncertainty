# Plan of the extension (items X1-X6), 2026-10-11

Purpose: the additions that strengthen the study for the journal article (candidate journals IEEE
OJ-ITS / T-ITS / T-IV / TVT / T-SMC:S, Transportmetrica A/B, CACAIE, IEEE Access as the fallback), without touching
the first version `paper/v13`. Every item has a question, a design, a reading rule fixed before the run, a
cost estimate from the timings of the existing roots and a deliverable. Nothing of the existing artefacts is
changed or re-run; every new option defaults to the stored behaviour.

Timing references (RTX 6000 Ada, the timings of the stored roots): training per seed on highD `hetero_mdn` 203 s,
`deep_ensemble` 1195 s, `diffusion_cf` 568 s (112 590 sequences); a full E1 root (11 models x 5 seeds, training +
evaluation on six sets) about 4-5 h; an E3 unit (model x seed, 60 profiles x 20 followers x 100 rollouts) 17-225 s
in the pooled cells (`artifacts/e3/logs/timings.jsonl`), so a full E3 variant of 8 models about 1 h. Disk: 117 GB
free on C: (2026-10-11).

## X1. Preview sweep: how much of the leader's future does the diffusion law need?

Question. The platoon advantage of DiffusionCF is tied to the leader's future (`paper/v13`, Sections 3.6 and 3.9:
history-only cells refuted or inconclusive under both training regimes). The article reports two points only:
the full 5 s preview and none. The journals of the vehicle group (T-IV, TVT) read the preview as V2V information;
a curve over the preview horizon, with noise and delay, turns the yes/no result into a design statement.

Design, stage A (inference only, stored checkpoints). A `preview` spec of an E3 cell, applied to the leader speeds
`v_lead` that the engine hands to `init_sampler` (`cf_unc/rollout/engine.py`, `pass_v_lead`):

* `horizon_s` h: at rollout step k the sampler sees `v_lead[k+1 : k+1+h/DT]` and the value at `k+h/DT` held
  constant for the rest of the 5 s plan (the padding rule the sampler already uses at the end of the window);
  h = 0 is the constant-leader assumption of the `None` path.
* `noise_sigma` [m/s]: i.i.d. Gaussian noise added to every preview sample (its own stream per trajectory, seeded
  by the cell), the measurement / broadcast error of a V2V preview.
* `delay_s` d: the preview is stale by d, i.e. the window handed at step k is `v_lead[k+1-d/DT : ...]`.

Cells (per variant `pool` and `lodo_openacc`, also `raw` for the highD training): h in {0, 0.5, 1, 2, 3, 5} s at
zero noise and delay; sigma in {0.25, 0.5, 1.0} m/s at h = 5; d in {0.5, 1.0} s at h = 5. Model `diffusion_cf` only
(the one-step laws do not read the preview; the reference `idm_white_noise` of each cell is the stored one). Cost:
11 cells x 3 variants x 5 seeds x about 80 s = about 4 h.

Design, stage B (retraining at the budget). `DiffusionCF(preview_steps=...)`: the leader channel of the denoiser
beyond `preview_steps` is replaced by the held value at training time too, so that the law learns to use the
budget it gets (stage A hands a truncated preview to a model trained on exact ones: a distribution shift of the
condition). Horizons 1 s and 2 s on the roots `e6_pool` and `e6_lodo_openacc` (matched signals, target a, 5 seeds;
new roots `extension/artifacts/ext_pool_prev1`, `..._prev2`, `ext_lodo_prev1`, `..._prev2`, each training
`diffusion_cf` only with the new kwarg, the E6 caps and the frozen e1m IDM prior), then the E3 cells `pool` /
`lodo_openacc` with the per-variant checkpoint override of `e3.yaml` (as `diffusion_cf_hist` in 18.8). Cost: 4 roots
x 5 seeds x about 10 min = about 3.5 h training, E1 evaluation about 1 h, E3 about 0.5 h.

Reading rule (fixed before the run). Per driver mode and cell: the H3.2 verdict of the amplitude score against
`idm_white_noise` with the file-cluster interval (design.md 18.7), plus the collision share. Reported quantities:
the smallest horizon at which the verdict is `supported` (stage A and stage B separately); the ratio of the
amplitude score at (sigma, d) to the clean h = 5 cell with its event/follower bootstrap interval. Outcomes: (i) the
advantage holds at h <= 1 s: a short V2V broadcast suffices and the claim of the article is strengthened; (ii) it
needs h >= 3 s: the advantage depends on the whole plan of the predecessor and the article says so; (iii) stage B
differs from stage A: the budget must be trained for, which is itself the design statement. All three are reported.

Deliverables: `extension/artifacts/ext_e3_preview/` (units, tables `preview_sweep.csv`, verdicts), one figure
(amplitude score and collision share against h, with the noise and delay cells as markers), a paragraph of Results
and one of Discussion in v14.

## X2. Cost table

Question. Reviewers of the IEEE journals and of SMPT / EAAI ask for the computational cost of every law; the article
has none. Needed for the practical paragraph (simulator use: a per-step diffusion sampler against a Gaussian head).

Design. `extension/scripts/cost_table.py`:

* training wall time per seed from the stored `train_log.json` of every root (`seconds`, `n_train`, device) for
  `artifacts/e1` (recorded signals) and `artifacts/e6_pool` (pooled domain), mean and range over seeds; parameter
  counts from the checkpoints;
* one-step inference: a benchmark of `predict_dist` on 8 192 highD test windows (the stored checkpoints, seed 0,
  GPU and CPU, 5 repeats, median) in ms per 1 000 windows; for DiffusionCF the 32 samples x 50 DDIM steps of the
  stored setting and the S = 16 / 64 settings of the density sensitivity;
* platoon simulation: seconds per unit from `artifacts/e3/logs/timings.jsonl` (pooled cells), normalised per
  profile x follower x rollout.

Cost: about 20 min of compute, no training. Deliverable: `extension/artifacts/ext_cost/tables/cost.csv`, compact
table c20 of v14 (model, parameters, training s/seed, inference ms/1000 windows GPU and CPU, platoon ms per
vehicle-rollout-second), one sentence in Methods and one in Discussion.

## X3. Framing: the protocol as the first contribution

No new run. The extended manuscript names as contribution 1 the evaluation protocol (prespecified hypotheses with
decision rules; one-step, closed-loop and platoon levels; common-window comparisons; per-horizon ACI with the
feedback budget; the admissibility table of Section 4), with one schematic figure of the protocol (levels, inputs,
decision rules) and a numbered contribution list in the Introduction. The negative findings are then applications
of the protocol. Written in v14 together with X4; the claims audit of v13 is rerun on v14.

## X4. Shortening

Targets per journal family: IEEE OJ-ITS (decision of 2026-10-11): no page limit, a regular paper
recommended not to exceed 10 pages of the journal's two-column template, so about 7-8 k words with 6-8 displayed
tables and 6 figures, the rest in the supplements; IEEE Access: under 20 pages; IEEE Transactions: 14 pages
(about 8 k words); T&F / Wiley: 8-10 k words. v13 has 14.9 k words, 19 tables and 11 figures in the main
text. The controls of Sections 3.6-3.10 move to the supplement with a one-paragraph summary each; the Related-Work
table stays; Results, Discussion and Conclusions lose their repetitions (the v13 check listed them). The supplement
S1 is unchanged (224 pages); the moved sections form a second supplement S2 or an appendix, by the journal's rule.

## X5. Reimplementation check against published numbers

Question. QuantilePIDL (`liu2023qrpidl`, no public code) and DiffusionCF (a minimal DDPM, not `you2025followgen`)
are our own implementations; a reviewer of TR-C / T-ITS asks whether the re-implementations reach the published
performance, else the comparison is against straw men.

Design, in order of feasibility:

1. FollowNet protocol (`chen2023follownet`, Scientific Data; open data and published baseline numbers for IDM, GHR,
   LSTM and others on highD and NGSIM I-80 among five sets): closed-loop simulation of every event given the
   recorded leader, spacing MSE and collision rate. Local data: `data/events/follownet_highd` (25 024 events) and
   `follownet_ngsim_i80` (3 874 events). Our calibrated IDM, LSTMConstSigma (mean), QuantilePIDL (median) and
   DiffusionCF (mean plan) are run under that protocol; the table "ours vs published" with the protocol differences
   listed (splits, horizon, sampling). The anchor is the IDM row: if our IDM lands in the published range, the
   pipeline is consistent; the learned laws are then compared with the published LSTM row.
2. QRPIDL: the evaluation of Liu et al. 2023 reconstructed from the paper (data set, inputs, split, metrics such as
   PICP / MPIW / RMSE); our QuantilePIDL run under it as far as the protocol can be reconstructed, with every
   unreconstructible choice listed. If the protocol cannot be matched, the paper says so and item 1 stands.
3. FollowGen: if its code is public, run it under the E1 protocol on highD and NGSIM I-80 (CRPS, NLL, PICP) as an
   external diffusion reference; otherwise its published numbers are quoted next to ours with the protocol gap.

Cost: item 1 about one day of work (the FollowNet-2 code of plan P4 has the protocol), compute minutes; items 2-3
depend on availability. Deliverable: `extension/artifacts/ext_followcheck/tables/`, a short subsection of Methods
("Implementation checks") and a supplement table.

## X6. Second training domain

Question. Every conclusion of the article is drawn from laws trained on highD (and on the pooled domain). A
reviewer asks whether the aleatoric growth under shift, the H3.1 pattern and the platoon verdicts are properties
of the laws or of highD. The pooled roots answer it in part; a single second domain with the shift reversed
(rough to smooth) answers it directly.

Design. Training domain NGSIM I-80 (own reconstruction, 15 302 events, 3.7 M samples: the largest set after highD
and the roughest signal), recorded signals, target a, all 11 models of `e1.yaml`, 5 seeds; root
`extension/artifacts/ext_e1_ngsim` (`ext_e1_ngsim.yaml`, base `e1.yaml`: `train.data: ngsim`, evaluation on the
NGSIM I-80 test fold (in-domain), the NGSIM validation fold (placebo), highD, US-101, Waymo, OpenACC; IDM
calibrated on the NGSIM training events). Then E3 with these checkpoints: cells `ngsim` (leader window of the
profiles file) and `ngsim_w61` (leader re-smoothed with window 61), the models of `e3.yaml`, 5 seeds. OpenACC as
the alternative domain is not chosen: the pooled cell `pool` already covers "OpenACC seen", and the platoon
profiles are OpenACC leaders. Optional second step, only if the first shows the reversed pattern: the matched
variant (`e1m`-style windows with NGSIM as the reference roughness).
Cost: about 5 h (E1 root) + 2 h (two E3 cells) + analysis.

Reading rule (fixed before the run). The analyses of E1 (H3.1 shift rule, aleatoric / epistemic growth ratios,
H3.3 pairs) with NGSIM I-80 as the in-domain reference and highD as a shifted set; the H3.2 verdicts of the two E3
cells per driver mode. The article's statements are kept if the sign of every reported effect is the same;
every statement whose sign changes is rewritten as domain-dependent. No new threshold.

## Order of work and cost

| Step | Item | Compute | Decision needed |
|---|---|---|---|
| 1 | X2 cost table | 20 min | none |
| 2 | X1 stage A (inference preview sweep) | 4 h | none |
| 3 | X1 stage B (retrained budgets 1 s, 2 s) | 5 h | none |
| 4 | X6 second domain, E1 root + two E3 cells | 7 h | the choice NGSIM I-80 (default) or OpenACC |
| 5 | X5 FollowNet-protocol check (+ QRPIDL / FollowGen if feasible) | minutes | none |
| 6 | X3 + X4 writing: v14 from v13, protocol figure, shortening to the target journal | none | the target journal (sets the length) |

The runs of steps 2-4 go through the retry runners of the repository (shared GPU); every step ends with its
tables, a dated entry in `decisions.md` and a report paragraph before the next one starts.

## S. Strengthening items after release v1.4.0 (decision of 2026-10-12)

Nine additions to the extended manuscript, run in the order S1-S6 and then S7-S9; every item has its reading rule
fixed here before the run. Roots under `extension/artifacts/`, scripts under `extension/scripts/`.

### S1. Admissibility scorecard

One compact table (law x check) assembled from the stored tables: one-step calibration in domain (highD PICP90),
calibration under shift (H3.1 / H3.1e verdicts), closed-loop correction (H3.4 verdict of the ACI-corrected
intervals), collision-free share of the pooled platoon cell, the H3.2 verdict of the pooled cell (ACC profiles,
file-cluster interval), the information budget (smallest supported preview horizon, diffusion only), and the cost
(microseconds per vehicle-rollout-second). Generator `scripts/compact_tables/make_ext_tables.py` (table c25). No
new number is computed: every cell is a verdict or a value of an existing table. Reading rule: the table is a
summary; no verdict is changed by it.

### S2. State-distribution part of the shift

Question: how much of the NLL and variance growth under shift is explained by the different occupancy of the
state space (s, dv, v) alone. Design: the windows of every shifted set are reweighted by importance weights
p_in(b) / p_shift(b) over 6 x 6 x 6 quantile bins of (s, dv, v) built on the in-domain highD test fold (the
training windows are not stored as a table; the test fold is driver-disjoint from the training folds and has the same
occupancy); bins empty in the in-domain set are dropped (their mass is reported); the effective sample size is
reported. Quantities: NLL, CRPS, PICP90, aleatoric and epistemic variance, raw and reweighted, per model (seed
mean, seeds pooled) and shifted set; the decomposition NLL_shift - NLL_in = (NLL_shift - NLL_rw) + (NLL_rw -
NLL_in), the first bracket the state-distribution part, the second the conditional part. Script
`state_reweight.py`, root `ext_reweight`. Reading rule: a shifted set whose conditional part is below 25 % of the
total NLL gap is "explained by the state distribution"; the H3.1 verdicts are not recomputed (the matched estimates
of E1 already restrict to common support).

### S3. Re-planning interval and sampling steps of the diffusion law

Cells of the pooled regimes (pool, lodo_openacc, leader window 61, 60 profiles, 5 seeds, reference
idm_white_noise) with the sampler of the stored checkpoints re-configured at inference: re-planning every 1, 2 or
5 s (replan_every 10, 20, 50 steps, 50 sampling steps) and 10, 25, 50 or 100 DDIM sampling steps (re-planning
every 5 s). The 5 s / 50 steps cell repeats the stored pool / lodo_openacc units. Implementation: the model spec
key `sampler` of an E3 variant (`DiffusionCF.configure_sampler`), recorded in the unit spec. Root `ext_e3_replan`,
config `ext_e3_replan.yaml`. Cost from the timing log (seconds per unit). Reading rule: the H3.2 verdict
(amplitude, file-cluster interval) and the collision share of every cell next to its cost; a setting is
"admissible" when its verdict is the stored one; the cheapest admissible setting is reported.

### S4. Factorial preview cells and the budget-trained laws under noise and delay

Cells added to `ext_e3_preview`: the pairwise and triple combinations of horizon 2 s, noise 0.5 m/s and delay
0.5 s (h2n05, h2d05, n05d05, h2n05d05) for pool and lodo_openacc, and the 2 s budget-trained laws (ext_pool_prev2,
ext_lodo_prev2) under noise 0.5 m/s and under delay 0.5 s. Config `ext_e3_preview_c.yaml` (same root, same run
spec). Reading rule: as X1 (verdict and collision share per cell); an interaction is reported when the combined
cell loses the verdict that each factor alone keeps.

### S5. Conditional coverage of the corrected rollout intervals

Offline on the stored E2 rollout tables (target a, 500 events per set, 5 seeds): coverage of the 90 % gap
intervals by regime at the issue instant (the E1 regimes congested / braking / approaching / free / steady /
other, with the free speed of the highD IDM calibration) and by horizon, for the raw, static (multiplicative score
of E2), additive CQR (conformity score max(lo - y, y - hi), split-conformal on highd_val per horizon) and ACI
online (gamma 0.01) intervals; the interval (Winkler) score of the finite intervals per variant. Script
`conditional_coverage.py`, root `ext_coverage`. Reading rule: a variant is "conditionally valid" on a set when
every regime with at least 30 events has PICP within 0.05 of the nominal 0.90 at the 10 s horizon; otherwise the
regimes that fail are named.

### S6. Reference IDM calibrated per driver mode

The reference of H3.2 is the highD-calibrated IDM with white noise. Two further references: the IDM calibrated
(closed-loop spacing objective of E1, 300 events, seed 0) on the OpenACC events driven under ACC and on those
driven by humans, each with the white-noise sigma fitted on the windows of its own mode (`idm_white_noise_acc`,
`idm_white_noise_human`; script `calibrate_idm_modes.py`, root `ext_idm_modes/checkpoints`). Their platoon units
in the four cells raw, harmonised, pool, lodo_openacc (root `ext_e3_idmref`, config `ext_e3_idmref.yaml`), then
the H3.2 reductions of every stored law of `artifacts/e3` against the mode-matched reference recomputed with the
analysis of E3 (`idm_ref_rerate.py`: the ACC profiles against the ACC reference, the human-driven profiles against
the human reference). Reading rule: a verdict that holds against the highD reference and against the mode-matched
reference is "reference-robust"; a verdict that changes is reported as reference-dependent.

### S7. One-step networks trained in closed loop

`lstm_const_sigma_cl` and `hetero_mdn_cl` (`cf_unc/models/closed_loop.py`): the stored one-step networks
fine-tuned with the one-step loss plus a closed-loop term, the squared gap and speed errors (in units of their
standard deviations) of a 5 s deterministic rollout with the predicted mean acceleration behind the recorded
leader, integrated exactly as the rollout engine (semi-implicit Euler, 0.1 s, clip, standstill floor). Budget: 3
epochs on the sequence windows, weight 1, from the checkpoints of the highD, pool and lodo_openacc roots (5 seeds
each). Then the one-step evaluation (E1 tables), E2 (highD root: H3.4) and the platoon cells raw, pool,
lodo_openacc. Roots `ext_cl_highd`, `ext_cl_pool`, `ext_cl_lodo`, `ext_e2_cl`, `ext_e3_cl`. Reading rule: the
closed-loop law is compared with its one-step twin on the same cells: one-step NLL in domain and under shift, H3.4
verdict, collision share, H3.2 verdict; an improvement is claimed only when the sign holds in every regime.

### S8. Plan sharing between followers

In the platoon cascade a follower's diffusion sampler is handed, at every re-plan, the plan its predecessor sampled
at the same instant (its planned speeds, held beyond the plan) instead of the predecessor's realised future; the
first follower keeps the recorded leader's future. With re-planning every 5 s the shared plan and the realised
future differ only at the standstill floor; with re-planning every 1 or 2 s the predecessor abandons its plan at
the next re-plan and the shared plan is a genuine intention. Implementation: `v_lead_plan` of the rollout engine
and `simulate_platoon(share_plans=True)`, the E3 variant key `share_plans`. Cells pool_r10_share, pool_r20_share,
pool_r50_share and the lodo_openacc twins in `ext_e3_replan`. Reading rule: the shared-plan cell is read against
the oracle cell of the same re-planning interval (S3) and against the 1 s / 2 s horizon cells of X1; the verdict
and the collision share are reported; no new threshold.

### S9. String stability

From the stored per-profile tables (`profile_index.csv` of `artifacts/e3`, `ext_e3_preview`, `ext_e3_ngsim`): the
amplitude ratio A(i+1) / A(i) of the mean sampled amplitudes by vehicle index, the head-to-tail ratio A(20) /
A(leader), and for the recorded followers the same ratios of the recorded amplitudes. Per law and cell: the
geometric mean ratio over the indices 1-19 with a file-cluster bootstrap interval, the share of indices with a ratio
above 1, and the recorded ratio on the recorded indices. Script `string_stability.py`, root `ext_string`. Reading
rule: a law is "string-stable in the mean" on a cell when the interval of its geometric mean ratio lies below 1;
"amplifying" when it lies above 1; the recorded platoons give the empirical reference (no threshold).

### Order and cost

S3 (about 2.5 GPU h) -> S4 (1.5 h) -> S6 units (0.5 h) on the GPU queue (`queue2.sh`); S2, S5, S6 calibration, S9
and S1 offline meanwhile; then S7 (fine-tunings about 2 h, E2 and E3 about 2 h) and S8 (1.5 h) on the queue.
Every step ends with its tables and a dated entry in `decisions.md`.

Actual times (2026-10-12, GPU shared with another workload until about 20:00, which made a diffusion platoon unit
2.7 s instead of 1.05 s): S3 5.0 h, S4 2.0 h, S6 units 0.2 h (queue2 10:53-18:11); S7 fine-tunings 1.1 h, E2 0.3 h,
platoon cells 0.6 h, S8 1.7 h (queue3 18:12-21:56); the offline items minutes each (the S6 calibrations 25 s).
