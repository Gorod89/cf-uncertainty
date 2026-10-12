# cf-uncertainty

[![tests](https://github.com/Gorod89/cf-uncertainty/actions/workflows/tests.yml/badge.svg)](https://github.com/Gorod89/cf-uncertainty/actions/workflows/tests.yml)

Code, configurations, derived tables and figures of the article

> M. Gorodnichev, M. Moseva. *Epistemic and Aleatoric Uncertainty in Learned Car-Following:
> State-Space Maps, Dataset Shift, and Calibration under Closed-Loop Rollouts.* Submitted to
> *Systems* (MDPI), 2026.

The pipeline evaluates eight uncertainty-aware car-following models (heteroscedastic mixture
density networks, deep ensembles, Monte Carlo dropout, a Gaussian-process residual on a calibrated
IDM, a physics-informed quantile network, a conditional diffusion model, and IDM / LSTM baselines
with constant noise) against four pre-registered hypotheses: uncertainty maps over the state space
and variance ratios under dataset shift (H3.1 and its epistemic restatement H3.1e), regime-aware
variance (H3.3), coverage of closed-loop rollouts with split-conformal and adaptive conformal
corrections (H3.4), and the distributional evaluation of 20-vehicle platoons driven by measured
OpenACC leaders (H3.2). Three follow-up experiments decompose the dataset shift: one harmonised
signal processing, roughness-matched signals (E5), training on a pooled domain with
leave-one-domain-out roots (E6), and the platoon evaluation with the pooled-domain models. Eight control
experiments added after two reviews (the diffusion model conditioned on the history only, trained on highD and on
the pooled domains; the training domain crossed with the leader smoothing; past-only processing and the cell with
centred inputs and the target from the recorded speed on the same events; an IDM with AR(1) acceleration noise; a
variance head at a fixed mean; the density sensitivity of the sample-based law) are described in
`docs/design.md` section 18 and in `docs/review_fixes_v3.md` / `docs/review_fixes_v4.md` / `docs/review_fixes_v8.md`; the corrected
adaptive conformal recursion, the cluster bootstraps over recording files and followers, the collision
diagnostics and the common-support growth exponents are in the same notes.

* `docs/design.md`: the binding design (conventions, interfaces, hypotheses, verdict thresholds,
  the decisions taken at each review and every deviation).
* `docs/report.md`: the final report with every verdict and number, with an addendum after the review;
  `docs/m1_report.md` ... `docs/m8_report.md`: the milestone reports (commands, key numbers, deviations);
  `docs/review_fixes_v3.md`, `docs/review_fixes_v4.md`, `docs/review_fixes_v8.md`: the analysis fixes and the control experiments.
* `results/controls/`: the analysis tables of the experiment roots the article tables read, as they are
  (`results/controls/<root>/...` corresponds to `artifacts/<root>/...` of the development layout: one-step metrics,
  H3.3 pairs with the event and the follower unit, matching files, density sensitivity, the platoon and calibration
  tables, the common-window table of the past-only control, the AR(1) fit logs); `results/compact_tables/`: the 19
  tables of the article as published; `results/TABLE_SOURCES.md`: for every table of the article the source files with
  their SHA-256, the generator that wrote the table and the command that regenerates its sources.
* `scripts/compact_tables/`: the generators of the article tables (`make_compact.py` for c1-c18, `make_c19.py` for c19,
  `make_table_sources.py` for the manifest) with a README on the artefact layout they read (`CF_UNC_ARTIFACTS`) and the
  check that a fresh layout built from `results/` and the release asset regenerates c1-c8 and c10-c19 identically;
  `scripts/common_support_cells.py`: the three cells of the past-only control aligned by (event_id, t_index) and scored
  on the windows they share, with paired differences and event / follower bootstrap intervals.
* `results/report/`: the derived tables (`tables/*.csv` and `*.tex`, 85 tables), the figures and `index.md`
  (every table and figure with its caption and source), as produced by `scripts/make_report.py`
  from the experiment artefacts. The manuscript and its supplementary file are not part of this
  repository.

## Conventions

* SI units, 10 Hz, `dv = v - v_lead` (positive when closing in) **everywhere**,
  `s` = net gap, state features in the order `(s, dv, v)`.
* History of 3 s (30 samples, current sample included); targets `a[t]` (one-step acceleration) and
  `mean_1s` (mean acceleration over the next second).
* Semi-implicit Euler at 0.1 s, acceleration clipped to [-8, 4] m/s^2.
* The Event schema and its parquet layout are shared with the sibling repository
  [cf-stability](https://github.com/Gorod89/cf-stability); event directories are interchangeable
  between the two.

## Installation

Python 3.11. A CPU build of PyTorch is enough for the tests and the analyses; the experiments
were run on a CUDA build.

```bash
python3.11 -m venv .venv
. .venv/bin/activate                       # Windows: .venv\Scripts\activate
pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
python -m pytest -m "not slow and not gpu and not data"      # the fast suite (continuous integration): 553 tests
python -m pytest -m slow                                     # tests that train (about 15 min on CPU): 22 passed, 2 expected failures
python -m pytest -m gpu                                      # 25 tests that need CUDA
```

Set `PYTHONUTF8=1` if the repository path contains non-ASCII characters. Every long run is
resumable and the `scripts/run_e*_retry.sh` wrappers restart a run after a GPU fault. `environment.lock.txt`
holds the exact package versions of the development environment (CUDA 12.8); `data/events_manifest.json`
the SHA-256 of every event file the experiments used and how to obtain them. The per-profile platoon tables of E3 (every variant,
`artifacts/e3/tables/profile_index*.csv`) are attached to the GitHub release (v1.3.1, the final snapshot of the article; the same assets as v1.3.0) as `e3_platoon_tables.tar.gz`; the trained checkpoints of every
experiment root (655 PyTorch files with their training logs, 0.2 GB) are attached to the same release as
`checkpoints.tar.gz` with a SHA-256 manifest (`CHECKPOINTS_MANIFEST.json`; unpack into `artifacts/` to reuse them, the
roots e1m, e1m_mean1s and e2* reuse the e1h / e1h_mean1s files) and are regenerated by the commands above.

## Layout

```
cf_unc/constants.py        DT, HIST_LEN, SEQ_LEN, feature order, acceleration clip
cf_unc/data/               schema (Event, EventSet), StochasticIDM generator, windows, driver splits, pipeline,
                           harmonised and roughness-matched signals (harmonise.py)
cf_unc/physics/idm.py      IDM, closed-loop simulation, population calibration (spacing / acceleration objective)
cf_unc/dist.py             predictive laws: Gaussian, mixture, quantile function, samples; variance decomposition
cf_unc/eval/               metrics (NLL, CRPS, PICP, PIT/KS, calibration), driver bootstrap, paired tests, Holm
cf_unc/models/             HeteroMDN, DeepEnsemble, MCDropout, GPResidualIDM, QuantilePIDL, DiffusionCF,
                           IDMWhiteNoise, LSTMConstSigma (+ shared nets, registry, checkpoints)
cf_unc/train/              generic training loop, run_training (train + one-step evaluation + artifacts)
cf_unc/experiments/e1.py   experiment E1: one-step prediction tables for every data set (single or pooled training
                           domain, reusable checkpoints, frozen IDM prior; resumable)
cf_unc/maps/               prediction-table IO, state grid and maps, shift ratios (H3.1, H3.1e), regimes (H3.3)
cf_unc/rollout/            closed-loop rollout engine, rollout tables, conformal calibration and ACI
cf_unc/experiments/e2.py   experiment E2: rollouts of the E1 checkpoints (resumable)
cf_unc/platoon/            OpenACC raw-file reader, profile selection, platoon cascade, amplitude CRPS / growth
cf_unc/experiments/e3.py   experiment E3: platoon rollouts (signal and training-domain variants; resumable units)
cf_unc/report/             figures of E1 (maps, shift ratios, regimes), E2 (coverage by horizon), E3 (CRPS, growth)
configs/                   hydra configs: data/*, model/*, train/*, experiment/* (e1*, e2*, e3*, e6_*), maps, platoon
scripts/train.py           train one model (hydra)
scripts/run_m1.py          M1 benchmark: all models on synthetic data, summary table
scripts/import_cf_stability_events.py   copy event sets, fold files and IDM fits from a cf-stability checkout
scripts/run_e1.py, run_e1_retry.sh, select_w_phys.py, analyze_e1.py, analyze_e1_extra.py, summarize_e1.py,
                           make_figures_e1.py                 experiment E1 and its analysis
scripts/harmonise_report.py, match_roughness.py               harmonised and roughness-matched signals (E5)
scripts/run_e2.py, run_e2_retry.sh, analyze_e2.py, summarize_e2.py   experiment E2 and its analysis
scripts/select_profiles.py, run_e3.py, run_e3_retry.sh, analyze_e3.py   experiment E3 and its analysis
scripts/make_report.py     tables T1-T6 (csv + booktabs tex), figures F1-F7, index.md -> artifacts/report
tests/                     pytest suite (markers: slow, data, gpu)
results/report/            snapshot of artifacts/report for the article
data/splits, data/calibration   driver-disjoint fold files and IDM calibrations (provenance of the sibling project)
cf_unc/preview.py          PreviewSpec: the preview of the predecessor's future handed to the diffusion sampler (horizon, noise, delay)
extension/                 releases v1.4.0 and v1.5.0: configs (extension/configs/experiment), scripts (queues, cost table, preview
                           sweep, FollowNet-protocol check, the nine strengthening items), plan (extension/docs/PLAN.md);
                           results in results/extension/
cf_unc/models/closed_loop.py  the one-step networks fine-tuned in closed loop (hetero_mdn_cl, lstm_const_sigma_cl)
```

## Reproduction

The experiments read event sets from `data/events/<name>` (parquet; see `docs/design.md`, section 2)
which are produced by the sibling repository cf-stability from the public data sets and copied with
`scripts/import_cf_stability_events.py`. Raw data and event sets are not committed. The order of the
runs, with their commands, is in the milestone reports:

```bash
# E1 (both targets, both signal variants), E2, E3: docs/m2_report.md, m3_report.md, m4_report.md, section 1
bash scripts/run_e1_retry.sh artifacts/e1/logs/r1.out 8 --config configs/experiment/e1.yaml
bash scripts/run_e2_retry.sh artifacts/e2/logs/run.out 8 --config configs/experiment/e2.yaml
bash scripts/run_e3_retry.sh artifacts/e3/logs/run.out 8
python scripts/analyze_e1.py --root artifacts/e1 --quiet && python scripts/analyze_e1_extra.py --root artifacts/e1
python scripts/analyze_e2.py --root artifacts/e2 --quiet
python scripts/analyze_e3.py --root artifacts/e3 --quiet
# E5 (docs/m6_report.md): roughness matching, the reused e1h checkpoints, rollouts
python scripts/match_roughness.py --config configs/experiment/e1m.yaml
bash scripts/run_e1_retry.sh artifacts/e1m/logs/run.out 8 --config configs/experiment/e1m.yaml
# E6 (docs/m7_report.md): pooled training domain and leave-one-domain-out roots
bash scripts/run_e1_retry.sh artifacts/e6_pool/logs/r1.out 8 --config configs/experiment/e6_pool.yaml
# E3-pooled (docs/m8_report.md): the platoon variants of the pooled-domain checkpoints
bash scripts/run_e3_retry.sh artifacts/e3/logs/run_pooled.out 8 --variants lodo_openacc pool --quiet
# controls of the second review (docs/review_fixes_v8.md): history-only Diffusion on the pooled roots and its
# platoon units; the cell with centred inputs and the recorded target on the events of the past-only control
bash scripts/run_e1_retry.sh artifacts/e6_pool_ctrl/logs/run.out 8 --config configs/experiment/e6_pool_ctrl.yaml
bash scripts/run_e1_retry.sh artifacts/e6_lodo_openacc_ctrl/logs/run.out 8 --config configs/experiment/e6_lodo_openacc_ctrl.yaml
bash scripts/run_e3_retry.sh artifacts/e3/logs/run_hist_pooled.out 8 --variants pool lodo_openacc --models diffusion_cf_hist --quiet
bash scripts/run_e1_retry.sh artifacts/e1m_rt/logs/run.out 8 --config configs/experiment/e1m_rt.yaml
bash scripts/run_e1_retry.sh artifacts/e1m_mean1s_abv/logs/run.out 8 --config configs/experiment/e1m_mean1s_abv.yaml
# tables and figures of the article (no experiment, no GPU; a few seconds)
python scripts/make_report.py                                   # artifacts/report/{tables,figures,index.md}
```

## Strengthening items (release v1.5.0)

Nine further items after the extension (plan and reading rules in `extension/docs/PLAN.md`, section S); their tables
are published under `results/extension/<root>/tables/`, the compact tables of the manuscript as
`results/compact_tables/c25_scorecard.tex` to `c31_closedloop.tex` (generator `scripts/compact_tables/make_ext_tables.py`,
which reads the article tables from `CF_UNC_ARTIFACTS` and the extension tables from `CF_UNC_EXT_ARTIFACTS`).

* **S1 scorecard** (`c25_scorecard.tex`): one verdict or value of every check per law, from the stored tables.
* **S2 state-distribution part of the shift** (`results/extension/ext_reweight`): the shifted sets reweighted to the
  highD occupancy of (s, dv, v); the NLL gap split into a state-distribution and a conditional part
  (`extension/scripts/state_reweight.py`).
* **S3 sampler settings** (`results/extension/ext_e3_replan`): the diffusion law re-planned every 1, 2 or 5 s and run with
  10, 25, 50 or 100 DDIM steps at inference (the model-spec key `sampler` of an E3 variant, `DiffusionCF.configure_sampler`),
  with the cost of every cell (`extension/scripts/replan_tradeoff.py`).
* **S4 factorial preview cells** (`results/extension/ext_e3_preview`, rows of `preview_sweep.csv`): the combinations of
  horizon 2 s, noise 0.5 m/s and delay 0.5 s, and the 2 s budget-trained law under noise or delay.
* **S5 conditional coverage** (`results/extension/ext_coverage`): coverage of the corrected rollout intervals by regime at
  the issue instant, the additive CQR variant and the interval score (`extension/scripts/conditional_coverage.py`).
* **S6 reference per driver mode** (`results/extension/ext_idm_modes`, `ext_e3_idmref`): the IDM calibrated on the
  OpenACC ACC and human-driven events as references of H3.2 (`extension/scripts/calibrate_idm_modes.py`,
  `idm_ref_rerate.py`).
* **S7 closed-loop fine-tuning** (`results/extension/ext_cl_highd`, `ext_cl_pool`, `ext_cl_lodo`, `ext_e2_cl`,
  `ext_e3_cl`, `ext_cl_summary`): the LSTM and HeteroMDN checkpoints fine-tuned with a 5 s rollout loss
  (`cf_unc/models/closed_loop.py`, `extension/scripts/closed_loop_compare.py`).
* **S8 plan sharing** (cells `*_share` of `results/extension/ext_e3_replan`): followers handed their predecessor's sampled
  plan instead of its realised future (`record_plans` / `v_plan` of the rollout engine, `simulate_platoon(share_plans=True)`,
  the E3 variant key `share_plans`).
* **S9 string stability** (`results/extension/ext_string`): amplitude ratios by vehicle index of the simulated and the
  recorded platoons (`extension/scripts/string_stability.py`).

```bash
# the GPU queues in the order they were run (about 15 GPU hours on a shared RTX 6000 Ada); every step can be run alone
bash extension/scripts/queue2.sh && bash extension/scripts/queue3.sh
# the offline items and collectors
for s in state_reweight conditional_coverage string_stability calibrate_idm_modes idm_ref_rerate replan_tradeoff closed_loop_compare preview_sweep; do
  PYTHONUTF8=1 python extension/scripts/$s.py; done
```

The checkpoints of the fine-tuned laws and of the mode-calibrated references (111 files, 2.6 MB) are attached to release v1.5.0 as
`checkpoints_s7.tar.gz` with a SHA-256 manifest (unpack into `extension/artifacts/`); the per-profile platoon tables of
`ext_e3_replan`, `ext_e3_idmref` and `ext_e3_cl` as `s_platoon_tables.tar.gz`. The assets of v1.3.1 and v1.4.0 are unchanged.

## Extension (release v1.4.0)

Four additions to the study, run after the submitted article (plan and reading rules in `extension/docs/PLAN.md`,
design document section 19); their analysis tables are published under `results/extension/<root>/tables/`, the
compact tables of the extended manuscript as `results/compact_tables/c20_cost.tex` to `c24_ngsim_platoon.tex`.

* **Preview budget of the diffusion law** (`results/extension/ext_e3_preview`): the platoon experiment repeated with
  the preview of the predecessor's future truncated to a horizon of 0, 0.5, 1, 2, 3 or 5 s, perturbed by i.i.d.
  noise of 0.25, 0.5 or 1 m/s, or made stale by 0.5 or 1 s, on the stored checkpoints of the three training regimes
  (`cf_unc/preview.py`, the `preview` key of an E3 variant); and the law trained at a budget of 1 or 2 s on the two
  pooled roots (`DiffusionCF(preview_steps=...)`, roots `ext_pool_prev1`, `ext_pool_prev2`, `ext_lodo_prev1`,
  `ext_lodo_prev2`). Collector: `extension/scripts/preview_sweep.py` (`tables/preview_sweep.csv`, `.md`, the figure).
* **Cost of the laws** (`results/extension/ext_cost`): training time per seed, parameters, one-step inference time
  on the GPU and the CPU, platoon simulation time per unit (`extension/scripts/cost_table.py`).
* **Implementation check under the FollowNet protocol** (`results/extension/ext_followcheck`): closed-loop spacing
  MSE and collision rate of every law on the highD test fold and the FollowNet NGSIM I-80 events, next to the
  published FollowNet rows (`extension/scripts/follownet_check.py`).
* **Second training domain** (`results/extension/ext_e1_ngsim`, `ext_e3_ngsim`): every law trained on the NGSIM
  I-80 reconstruction and evaluated on its test fold, highD, NGSIM US-101, Waymo and OpenACC, and the platoon
  experiment with these checkpoints at both leader windows.

```bash
# the whole extension in the order it was run (about 15 GPU hours on one RTX 6000 Ada); every step can be run alone
bash extension/scripts/queue.sh                                 # log: extension/artifacts/queue.log
# the tables c20-c24 from the published results (no experiment)
CF_UNC_EXT_ARTIFACTS=results/extension PYTHONUTF8=1 python scripts/compact_tables/make_ext_tables.py --out /tmp/cf_build/tables
```

The checkpoints of the five new training roots (75 PyTorch files with their training logs and metrics, 28 MB) are attached to
release v1.4.0 as `checkpoints_ext.tar.gz` with a SHA-256 manifest (`CHECKPOINTS_EXT_MANIFEST.json`; unpack into
`extension/artifacts/`), and the per-profile platoon tables, configs, specs and re-smoothed profiles of the two new
platoon roots as `ext_platoon_tables.tar.gz`. The assets of v1.3.1 (`checkpoints.tar.gz`, `e3_platoon_tables.tar.gz`)
are unchanged and stay on that release.

## Models

| Config | Predictive law | Aleatoric | Epistemic |
|---|---|---|---|
| `hetero_mdn`, `hetero_mdn_mix3` | Gaussian / 3-component mixture, LSTM(64) | own variance | across seeds |
| `deep_ensemble` | pooled mixture of 5 members (seeds + driver bootstrap) | mean member variance | variance of member means |
| `mc_dropout` | pooled mixture of 50 dropout networks (p = 0.1) | same | same |
| `gp_residual_idm` | Gaussian, frozen calibrated IDM + SVGP residual (RBF-ARD, 300 inducing points) | learned noise | latent posterior variance |
| `quantile_pidl` | quantile function at {0.05, 0.25, 0.5, 0.75, 0.95}, pinball + IDM physics + non-crossing | variance of the interpolated law | across seeds |
| `diffusion_cf` | 32 samples of a conditional DDPM over 5 s acceleration sequences | sample variance | across seeds |
| `idm_white_noise` | calibrated IDM + constant Gaussian noise | sigma^2 | 0 |
| `idm_ar_noise` | calibrated IDM + AR(1) acceleration noise fitted on the training residuals (rho about 0.999 per step) | conditional sigma_eta^2 | 0 |
| `diffusion_cf_hist` | the diffusion model conditioned on the 3 s history only (no leader future), trained on highD and on the pooled domains | sample variance | across seeds |
| `lstm_hetero_frozen` | the frozen mean of `lstm_const_sigma` with a state-dependent variance head trained by NLL | own variance | across seeds |
| `lstm_const_sigma` | deterministic LSTM + constant Gaussian noise | sigma^2 | across seeds |

Every model implements `fit`, `predict_dist(state_history)`, a step sampler for closed-loop
rollouts (`init_sampler` / `sample_step`, modes `per_step` and `per_trajectory`) and `save` / `load`.

```python
from cf_unc.data.pipeline import synthetic_bundle
from cf_unc.models import build_model, load_model

bundle = synthetic_bundle({"n_drivers": 100, "noise_sigma": 0.4}, stride=3)
model = build_model("deep_ensemble", n_members=5)
model.fit(bundle.windows["train"], bundle.windows["val"], seed=0)
dist = model.predict_dist(bundle.windows["test"].X)
dist.mean(), dist.aleatoric_var, dist.epistemic_var, dist.interval(0.9), dist.crps(bundle.windows["test"].y)
```

## Data

Synthetic: the built-in StochasticIDM generator (`configs/data/synthetic*.yaml`). Real: event sets
extracted by the sibling repository cf-stability from the public data sets (highD through the
FollowNet benchmark extraction, NGSIM I-80 and US-101 reconstructed from the raw trajectories, the
Waymo car-following pairs of Hu et al. (2022), OpenACC), copied into `data/events/<name>` together
with the fold files (`data/splits`) and the IDM calibrations (`data/calibration`):

| Config | Event set | Role |
|---|---|---|
| `highd` | `follownet_highd` (FollowNet highD events, stand-in for raw highD) | training and in-domain test |
| `ngsim` | `ngsim_i80` (reconstruction of the raw NGSIM I-80) | shift |
| `ngsim_us101` | `ngsim_us101` (reconstruction of the raw NGSIM US-101) | shift |
| `ngsim_fn` | `follownet_ngsim_i80` (FollowNet extraction) | cross-check of `ngsim` |
| `waymo` | `waymo` (pairs of Hu et al. 2022) | shift |
| `openacc` | `openacc`; the raw OpenACC files also provide the platoon leader profiles of E3 | shift |

## Citation and licence

See `CITATION.cff`. The code is released under the MIT licence (`LICENSE`); the data sets keep the
licences of their providers.
