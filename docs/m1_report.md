# Milestone M1 report: models + tests

Date: 2026-09-27. Environment: Windows 10, Python 3.11.8, torch 2.14.0 (CPU build),
GPyTorch 1.15.2, hydra-core 1.3.7, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1.

## 1. Commands

```bash
export PYTHONUTF8=1
.venv/Scripts/python.exe -m pytest -m "not slow" -p no:cacheprovider      # 242 passed, 20.6 s
.venv/Scripts/python.exe -m pytest -m slow -p no:cacheprovider            # 22 passed, 2 xfailed, 7 min 21 s
.venv/Scripts/python.exe scripts/run_m1.py --data synthetic        --override stride=3 eval_max_rows_sampled=1000   # 16 min
.venv/Scripts/python.exe scripts/run_m1.py --data synthetic_hetero --override stride=3 eval_max_rows_sampled=1000   # 12 min
.venv/Scripts/python.exe scripts/train.py model=deep_ensemble data=synthetic seed=0                                # single model
```

Log of the validation run: `artifacts/m1/logs/m1_validation.log`; tables:
`artifacts/m1/<data>/summary.{md,csv,json}`; checkpoints and metrics with
driver-bootstrap CIs: `artifacts/runs/<data>/<model>/seed0/`.

## 2. What was built

| Part | Content | Tests |
|---|---|---|
| Data | Event / EventSet (parquet-compatible with cf-stability), StochasticIDM generator with ground truth, windows, driver-disjoint splits, driver bootstrap | 59 |
| Physics | IDM (numpy and torch), closed-loop simulation with collision index, population calibration (spacing / acceleration objective) | incl. above |
| Predictive laws | Gaussian, mixture, quantile function, samples; NLL, CRPS, quantiles, PIT; variance decomposition | 83 |
| Statistics | driver bootstrap (means, ratios, two-sample ratios, generic), paired bootstrap test, Holm | incl. above |
| Models | HeteroMDN (K = 1, 3), DeepEnsemble, MCDropout, GPResidualIDM, QuantilePIDL, DiffusionCF, IDMWhiteNoise, LSTMConstSigma | 111 |
| Glue | hydra configs, `scripts/train.py`, `scripts/run_m1.py`, one-step evaluation with bootstrap CIs | 13 |

Tests required by section 8 of the specification that belong to M1:
decomposition identity (exact to 7e-15 at the level of the laws, rtol 1e-9 at
the level of the models, within 0.5 % against 200 000 samples) and CRPS
closed form vs sample based (Gaussian: mean deviation 0.61 %, max 1.8 %;
mixture: 0.59 %, max 1.9 %, 20 000 samples). The ACI test and the stochastic
rollout collision test come with their modules in M3 (the deterministic
closed-loop simulator already detects collisions and is tested).

## 3. Key numbers

### 3.1 E4 sanity (identical drivers, white noise sigma = 0.4, variance 0.16; 35.7k training windows)

| Model | Aleatoric (target 0.16 +- 15 %) | RMSE of the mean vs truth | Test NLL |
|---|---|---|---|
| IDMWhiteNoise (true IDM) | 0.161 | 0.000 | 0.501 |
| LSTMConstSigma | 0.162 | 0.061 | 0.512 |
| HeteroMDN K = 1 | 0.158 | 0.066 | 0.515 |
| HeteroMDN K = 3 | 0.164 | 0.064 | 0.515 |
| DeepEnsemble (5) | 0.164 | 0.050 | 0.509 |
| MCDropout (50 passes) | 0.179 (+12 %) | 0.071 | 0.522 |
| GPResidualIDM (calibrated IDM) | 0.158 | 0.016 | 0.502 |
| GPResidualIDM (wrong IDM, T = 1.0, a_max = 0.8) | 0.163 | 0.043 (IDM alone: 0.397) | - |
| QuantilePIDL | 0.169 | 0.059 (median) | pinball 0.1003 (optimum 0.0993) |
| DiffusionCF (32 samples) | 0.154 | 0.085 (Monte-Carlo floor 0.071) | 0.555 (KDE, approximate) |

Every model recovers the generator's noise variance within 15 %.

| Epistemic variance, mean over the same test windows | 10 % of the drivers | all drivers | ratio |
|---|---|---|---|
| DeepEnsemble | 2.90e-3 | 9.18e-4 | 3.2 |
| GPResidualIDM | 3.97e-2 | 1.52e-3 | 26 |
| MCDropout | 5.64e-3 | 4.96e-3 | 1.14 (xfail) |

Further checks: GP epistemic variance far off the support is 55 times its
test mean and equals the prior variance; heteroscedastic generator
(`sigma(s) = 0.4 (1 + 1.5 exp(-s / 15))`): HeteroMDN predicts 0.546 vs 0.573
(s < 10 m) and 0.175 vs 0.170 (s > 40 m), NLL 0.757 against 0.817 of
LSTMConstSigma; QuantilePIDL: PICP 0.905, width 1.350 vs 1.316, no crossings;
DiffusionCF: lag-1 autocorrelation of the generated noise -0.017, 5 s speed
change inside the 5-95 % interval in 93 % of the windows, RMSE of the 5 s
speed change 0.145 m/s with the true leader against 2.41 m/s with a
constant-speed leader.

### 3.2 Benchmark, `synthetic` (200 identical drivers, 69 440 training windows, test: 14 880 windows of 30 drivers)

| Model | NLL | CRPS | PICP 90 % | Width | RMSE vs truth | Aleatoric | Epistemic | Fit, s |
|---|---|---|---|---|---|---|---|---|
| idm_white_noise | 0.5001 | 0.2252 | 0.902 | 1.318 | 0.010 | 0.1605 | 0 | 0 |
| lstm_const_sigma | 0.5036 | 0.2260 | 0.901 | 1.318 | 0.035 | 0.1604 | 0 | 68 |
| hetero_mdn | 0.5032 | 0.2259 | 0.903 | 1.324 | 0.035 | 0.1621 | - | 62 |
| hetero_mdn_mix3 | 0.5035 | 0.2260 | 0.902 | 1.321 | 0.035 | 0.1614 | - | 61 |
| deep_ensemble | 0.5035 | 0.2260 | 0.903 | 1.323 | 0.035 | 0.1616 | 1.7e-4 | 282 |
| mc_dropout | 0.5063 | 0.2263 | 0.914 | 1.371 | 0.040 | 0.1708 | 3.3e-3 | 70 |
| gp_residual_idm | 0.4999 | 0.2251 | 0.903 | 1.321 | 0.010 | 0.1608 | 5.8e-4 | 9 |
| quantile_pidl | 0.5443 | 0.2271 | 0.903 | 1.330 | 0.048 | 0.1646 | - | 94 |
| diffusion_cf (1000 windows) | 0.6369 | 0.2407 | 0.823 | 1.159 | 0.086 | 0.1493 | - | 121 |

Reference: the entropy of N(., 0.16) gives NLL 0.5027 and CRPS 0.2257.

### 3.3 Benchmark, `synthetic_hetero` (driver heterogeneity CV 0.15, gap-dependent noise, mean noise variance 0.279)

| Model | NLL | CRPS | PICP 90 % | Width | RMSE vs truth | Aleatoric | Epistemic | Fit, s |
|---|---|---|---|---|---|---|---|---|
| idm_white_noise | 1.171 | 0.4192 | 0.941 | 2.918 | 0.554 | 0.787 | 0 | 0 |
| lstm_const_sigma | 0.8178 | 0.3057 | 0.913 | 1.855 | 0.141 | 0.318 | 0 | 29 |
| hetero_mdn | 0.7735 | 0.3016 | 0.903 | 1.764 | 0.134 | 0.299 | - | 35 |
| hetero_mdn_mix3 | 0.7805 | 0.3035 | 0.904 | 1.780 | 0.143 | 0.304 | - | 31 |
| deep_ensemble | 0.7743 | 0.3019 | 0.907 | 1.790 | 0.135 | 0.306 | 2.1e-3 | 149 |
| mc_dropout | 0.7799 | 0.3030 | 0.911 | 1.813 | 0.143 | 0.311 | 3.4e-3 | 39 |
| gp_residual_idm | 0.8389 | 0.3121 | 0.914 | 1.916 | 0.177 | 0.336 | 3.1e-3 | 16 |
| quantile_pidl | 0.9641 | 0.3300 | 0.905 | 1.904 | 0.255 | 0.367 | - | 67 |
| diffusion_cf (1000 windows) | 0.8520 | 0.3127 | 0.842 | 1.524 | 0.161 | 0.267 | - | 122 |

## 4. Findings that matter for the next milestones

1. **What "aleatoric" means with heterogeneous drivers.** A population model
   cannot separate the driver's own noise from between-driver differences:
   HeteroMDN reports 0.299 = noise 0.279 + unexplained heterogeneity
   0.134^2 = 0.018. On real data the aleatoric maps therefore contain driver
   heterogeneity; this has to be stated in the paper.
2. **Heteroscedastic vs homoscedastic (machinery of H3.3).** NLL gain of
   HeteroMDN over LSTMConstSigma: 0.044 nats on `synthetic_hetero`, 0.060 on
   the E4 gap-noise data; none on homoscedastic data (0.5032 vs 0.5036), as
   it should be.
3. **Rare states (machinery of H3.1).** On `synthetic` the mean epistemic
   variance in rare bins relative to the other bins is 3.2 [2.7, 3.7]
   (DeepEnsemble), 4.5 [4.0, 4.9] (MCDropout), 4.8 [3.5, 6.3] (GP); the
   aleatoric ratio is 1.00-1.04. Two caveats:
   * the ratio needs converged members: on the E4 data it is 0.95-1.4 after
     15 epochs, 1.4 with the default budget and 2.9 after 41-44 epochs,
     because optimisation noise is spread over all states;
   * the plan's definition (bin frequency < 1 % on the 5 m x 0.5 m/s grid)
     declares 75 % of the windows rare (only 10 of 480 bins exceed 1 %, they
     hold 25 % of the mass; 14.5 % of the mass lies outside the grid). A
     mass-based definition (least frequent bins holding 10 % of the mass)
     gives 3.6 [3.0, 4.2] and is more stable. M2 should report both.
   The rank correlation between density and epistemic variance at window
   level is close to zero (-0.11 to +0.05): the effect sits in the tail
   (large |dv|, large gaps), not in a monotone trend.
4. **MC dropout fails the "epistemic shrinks with data" sanity check**
   (ratio 1.14 against 3.2 for the ensemble and 26 for the GP) and inflates
   the aleatoric variance by 7-12 %. Its epistemic maps must be read with
   this in mind.
5. **Intervals from few samples under-cover by construction.** The nominal
   90 % interval between the 5 % and 95 % sample quantiles (linear
   interpolation) covers 0.851 with 32 samples and 0.892 with 200 samples
   even for a perfect model (simulation, 200 000 repetitions); this explains
   most of the PICP 0.82-0.84 of DiffusionCF above. With (S + 1) plotting
   positions (`numpy.quantile(..., method="weibull")`) the coverage is 0.908
   and 0.902. Rollout intervals in M3 must use them, otherwise part of the
   "coverage loss under rollouts" of H3.4 would be an artefact.
6. **QuantilePIDL with a population IDM** loses accuracy on heterogeneous
   drivers (RMSE 0.255 against 0.134) because the physics term pulls the
   median towards a biased IDM; the weight `w_phys = 0.3` is a default, not
   a tuned value.
7. **DiffusionCF** reproduces the one-step law, but the 5 s speed change is
   over-dispersed (sample std 0.19 against 0.136): the closed-loop feedback
   is only partly learned within the training budget.

## 5. Deviations from the specification

| # | Deviation | Reason |
|---|---|---|
| 1 | Event schema vendored; own StochasticIDM generator, driver splits and IDM calibration | cf-stability / cf-identifiability contained almost no code when M1 started |
| 2 | CPU build of torch | 16 GB of free disk space at the start (56 MB at the end); the code is device agnostic |
| 3 | `predict_dist` returns a law object (mean, variance, quantiles, samples, NLL, CRPS, PIT) instead of a bare `(mu, sigma)` | one interface for Gaussian, mixture, quantile and sample laws |
| 4 | Ensemble and MC dropout return the pooled mixture; `to_gaussian()` gives the moment-matched Gaussian; epistemic variance with ddof = 0 | the decomposition identity is then exact |
| 5 | MC dropout: one mask pair per pass shared by all inputs, generated from `predict_seed` | reproducible, independent of batching, a pass is one sampled network |
| 6 | Two sampler modes: `per_step` (specification) and `per_trajectory` (epistemic draw frozen along a rollout); the GP joint-posterior approximation is its `per_trajectory` mode | needed for H3.4; comparison of both modes becomes possible |
| 7 | QuantilePIDL is a re-implementation (no public code of Liu et al. 2023): physics term on the median, synthetic collocation windows with kinematically consistent histories, output bias initialised at residual quantiles; NLL from a piecewise-linear quantile function with exponential tails (approximate) | - |
| 8 | DiffusionCF: 1000 training steps (cosine), 50-step DDIM-type sampler with eta = 1, one convolution per residual block with dilations 1, 3, 9, 27, EMA warm-up, NLL by KDE (approximate) | training budget on CPU |
| 9 | GP: learned constant mean on top of the frozen IDM, float64, homoscedastic noise as specified (flat aleatoric map) | - |
| 10 | MAPIE is not installed; the in-house split conformal (allowed by the specification) is planned for M3 | disk space, scikit-learn dependency |
| 11 | ACI test and stochastic-rollout collision test are deferred to M3 | they test M3 modules |
| 12 | Two E4 tests are `xfail`: MC dropout epistemic shrinkage (finding 4) and the rare-state ratio of the short-budget test ensemble (finding 3) | findings, not bugs |

## 6. Open points for the review

1. Disk space: 56 MB free on C: at the end of M1. M2 cannot start before
   space is freed (event sets, checkpoints for 8 models x 5 seeds, optional
   CUDA build of torch).
2. Data for M2: raw highD and the reconstructed NGSIM are not available
   locally. FollowNet events (highD 25 Hz, NGSIM, Waymo; no driver ids;
   relative speed with the opposite sign) are in
   `Desktop/FOLLOW-NET/data`. OpenACC needs a download (or reuse of the copy
   in `cf-stability/data/raw`).
3. Definition of rare bins (finding 3) and regime thresholds (the
   cf-identifiability definitions were not available).
