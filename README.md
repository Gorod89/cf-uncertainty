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
leave-one-domain-out roots (E6), and the platoon evaluation with the pooled-domain models.

* `docs/design.md`: the binding design (conventions, interfaces, hypotheses, verdict thresholds,
  the decisions taken at each review and every deviation).
* `docs/report.md`: the final report with every verdict and number; `docs/m1_report.md` ...
  `docs/m8_report.md`: the milestone reports (commands, key numbers, deviations).
* `results/report/`: the derived tables (`tables/*.csv` and `*.tex`), the figures and `index.md`
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
python -m pytest -m "not slow and not gpu and not data"      # the fast suite (continuous integration)
python -m pytest -m slow                                     # tests that train (about 10 min on CPU)
```

Set `PYTHONUTF8=1` if the repository path contains non-ASCII characters. Every long run is
resumable and the `scripts/run_e*_retry.sh` wrappers restart a run after a GPU fault.

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
# tables and figures of the article (no experiment, no GPU; a few seconds)
python scripts/make_report.py                                   # artifacts/report/{tables,figures,index.md}
```

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
