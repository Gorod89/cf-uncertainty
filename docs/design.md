# cf-uncertainty: design contract

Binding for every module. If something here is impossible or wrong, do not
silently deviate: implement the closest sound alternative and list it under
"Deviations" in your final report.

## 1. Conventions

| Item | Value |
|---|---|
| Units | SI (m, m/s, m/s^2, s), float64 in events, float32 in tensors |
| Sampling | 10 Hz, `DT = 0.1` |
| Relative speed | `dv = v - v_lead` (approach rate, **positive when closing in**) everywhere |
| Gap | `s = x_lead - x_follower` (net gap, leader rear bumper minus follower front bumper) |
| Feature order | `(s, dv, v)`, last axis of every state tensor |
| History | `HIST_LEN = 30` samples (3 s), indices `t-29 .. t` (current sample included) |
| Target | `a[t]`: acceleration applied over `[t, t+DT)` |
| Acceleration clip | `A_MIN = -8.0`, `A_MAX = 4.0` |

Integrator (semi-implicit Euler), used by the synthetic generator, IDM
calibration and all rollouts:

```
a_t     = clip(a_t, A_MIN, A_MAX)
v_{t+1} = max(v_t + a_t * DT, 0)
x_{t+1} = x_t + v_{t+1} * DT
s_{t+1} = x_lead_{t+1} - x_{t+1}
dv_{t+1}= v_{t+1} - v_lead_{t+1}
```

Constants live in `cf_unc/constants.py`. Never hard-code them elsewhere.

## 2. Layout and ownership

```
cf_unc/constants.py            contract (written)
cf_unc/data/schema.py          Event, EventSet - vendored from cf-stability, parquet-compatible
cf_unc/data/synthetic.py       StochasticIDM generator
cf_unc/data/windows.py         WindowSet (written) + make_windows
cf_unc/data/splits.py          driver-disjoint splits, driver bootstrap
cf_unc/physics/idm.py          IDMParams, idm_accel, simulate_follower, calibrate_idm
cf_unc/dist.py                 PredDist family + decompose (numpy)
cf_unc/eval/metrics.py         NLL, CRPS, PICP, PIT/KS, calibration curves (numpy)
cf_unc/eval/bootstrap.py       driver bootstrap, paired comparison, Holm
cf_unc/models/base.py          CFModel interface (written)
cf_unc/models/nets.py          Normalizer, LSTMEncoder (written)
cf_unc/train/trainer.py        generic minibatch loop (written)
cf_unc/models/*.py             one file per model family
cf_unc/maps, rollout, platoon, report   milestones M2-M5
```

## 3. Data layer

### 3.1 Event / EventSet
Copy of `cf-stability/cf_stability/data/schema.py` (same fields, same parquet
layout `samples.parquet` / `events.parquet` / `manifest.json`, schema_version 1),
so event directories are interchangeable between the repositories.

### 3.2 StochasticIDM generator (`data/synthetic.py`)
Replaces the not-yet-available cf-identifiability generator.

```
a_t = clip( IDM(s_t, dv_t, v_t; theta_driver) + sigma(s_t) * eps_t , A_MIN, A_MAX ),  eps_t ~ N(0,1) iid
```

* `noise_kind="const"`: `sigma(s) = noise_sigma`.
  `noise_kind="gap"`: `sigma(s) = noise_sigma * (1 + noise_kappa * exp(-s / noise_s_scale))`.
* Noise is white **per step on acceleration** (m/s^2). The equivalent diffusion
  coefficient of a continuous-time Laval-type model is `sigma * sqrt(DT)`.
* Driver heterogeneity: per-driver lognormal multipliers (unit median, CV
  `heterogeneity_cv`) on `v0, T, s0, a_max, b`; `0.0` means identical drivers.
* Leader: regime-switching speed target (levels drawn from a mixture that
  covers congested 2-8, medium 8-20 and free 20-30 m/s; exponential holding
  times, mean 12 s, min 4 s), first-order tracking
  `a_L = clip(k (v_target - v_L), -3, 2) + OU noise`, `v_L >= v_min` with
  `v_min = 2.0` so that standstill truncation does not distort the noise law.
* Initial state: `v(0) = v_L(0) + N(0, 1.5^2)` clipped at 0;
  `s(0) = s_eq(v(0)) * U(0.6, 2.5)` clipped to [3, 150].
* If `v_t + a_t DT < 0` the speed is clipped to 0 and the **recorded** `a_t` is
  the effective one `(v_{t+1}-v_t)/DT`, so `v[t+1] == v[t] + a[t]*DT` always.
* Events with `min(s) < min_gap` (default 0.5 m) are regenerated.
* The generator also returns the ground truth per sample (`a_det` = noise-free
  IDM acceleration and `sigma_true`) in `Event.meta`-independent side arrays:
  `generate(cfg) -> (EventSet, truth: dict[event_id, dict[str, np.ndarray]])`.
* Fully reproducible from `seed` (use `np.random.default_rng`, spawn one child
  generator per driver).

### 3.3 Windows (`data/windows.py`)
`make_windows(events, hist_len=HIST_LEN, stride=1, seq_len=0, target="a")`.

* `X[i] = (s, dv, v)[t-hist_len+1 : t+1]`, `y[i] = a[t]`.
* `target="fwd_diff"` uses `(v[t+1]-v[t])/DT` instead (needs `t+1 < n`).
* `seq_len=L>0` adds `y_seq[i] = a[t : t+L]` and
  `v_lead_future[i] = v_lead[t+1 : t+L+1]`; only windows with `t+L < n` are kept.
* `driver` holds integer codes of `(dataset, follower_id)`; `driver_names`
  the decoding table. `event_index`, `t_index` locate every window.

### 3.4 Splits (`data/splits.py`)
Driver-disjoint 70/15/15 by `(dataset, follower_id)`, seeded permutation.
`bootstrap_drivers(driver_codes, rng)` returns window indices of a
with-replacement resample of drivers (all windows of a drawn driver, repeated
as many times as the driver is drawn).

## 4. Physics (`physics/idm.py`)

```
s*(v, dv) = s0 + max(0, v T + v dv / (2 sqrt(a_max b)))
a_IDM     = a_max [ 1 - (v / v0)^delta - (s* / max(s, S_EPS))^2 ]      S_EPS = 0.1
```

* `idm_accel(s, dv, v, params)` must work on numpy arrays **and** torch tensors
  (dispatch on the input type), output clipped to `[A_MIN, A_MAX]`.
* `calibrate_idm(events, objective="spacing"|"accel", ...)`: one population
  parameter set. `"spacing"`: closed-loop simulation of each event with the
  integrator above from the true initial state, leader trajectory given,
  minimise pooled spacing RMSE (normalised RMSE option), global search with
  `scipy.optimize.differential_evolution` (seeded) and a local polish.
  `"accel"`: one-step acceleration MSE. `delta` fixed at 4. Returns the
  parameters and a report (objective value, residual acceleration std, number
  of collisions in simulation).
* Bounds: `v0 [10, 45]`, `T [0.3, 3.5]`, `s0 [0.5, 8]`, `a_max [0.3, 4]`, `b [0.3, 5]`.

## 5. Predictive distributions (`dist.py`, numpy only)

All arrays are float64, first axis = sample index `N`.

```python
class PredDist(abc.ABC):
    aleatoric_var: np.ndarray | None   # (N,)
    epistemic_var: np.ndarray | None   # (N,)
    def __len__(self) -> int
    def mean(self) -> np.ndarray                   # (N,)
    def var(self) -> np.ndarray                    # (N,) total variance of the represented law
    def cdf(self, y) -> np.ndarray                 # (N,)
    def log_prob(self, y) -> np.ndarray            # (N,)
    def quantile(self, levels) -> np.ndarray       # (N, Q)
    def interval(self, coverage=0.9) -> tuple[np.ndarray, np.ndarray]   # central
    def sample(self, n, rng) -> np.ndarray         # (N, n)
    def crps(self, y) -> np.ndarray                # (N,)
    def pit(self, y, rng=None) -> np.ndarray       # (N,)
    def __getitem__(self, idx) -> "PredDist"       # numpy-style row indexing
    @classmethod
    def concat(cls, parts: list["PredDist"]) -> "PredDist"
```

| Class | Constructor | Notes |
|---|---|---|
| `GaussianDist` | `(mu, sigma, aleatoric_var=None, epistemic_var=None)` | closed forms |
| `MixtureDist` | `(weights, mus, sigmas, ...)` all `(N, K)` | closed-form NLL/CRPS (Grimit et al. 2006), quantile by bisection |
| `QuantileDist` | `(levels (Q,), values (N, Q), ...)` | piecewise-linear quantile function between knots, exponential tails whose density is continuous at the outer knots; values are sorted along Q on construction (and the number of crossings recorded) |
| `SampleDist` | `(samples (N, S), ...)` | ECDF; `log_prob` by Gaussian KDE (Silverman bandwidth per row, flagged approximate); sample CRPS; randomised PIT |

Decomposition (law of total variance, population variance over members, i.e.
`ddof=0`, so that the identity is exact for the uniform mixture of members):

```
aleatoric = mean_m Var_m[a | x]          epistemic = var_m E_m[a | x]
total     = aleatoric + epistemic  ==  Var of the pooled mixture
```

`decompose(member_means (M, N), member_vars (M, N)) -> (aleatoric, epistemic, total)`.

Ensembles return the **pooled mixture** (`MixtureDist` with `M * K` components,
weights `w_mk / M`) with `aleatoric_var` / `epistemic_var` attached;
`MixtureDist.to_gaussian()` gives the moment-matched Gaussian.

## 6. Metrics (`eval/metrics.py`, `eval/bootstrap.py`)

* `crps_gaussian`, `crps_mixture` (closed form), `crps_samples(samples, y, fair=False)`
  (sorting-based O(S log S)), `crps_quantile_function` (2 x integral of the pinball loss over levels).
* `nll_gaussian`, `nll_mixture` (logsumexp).
* `picp(lo, hi, y)`, `mean_width(lo, hi)`, `interval_score` (Gneiting-Raftery).
* `pit_histogram(pit, bins=20)`, `ks_uniform(pit)` (statistic, p-value),
  `calibration_curve(pit, levels)` (empirical vs nominal coverage of central
  intervals and of CDF levels), `calibration_error` (mean absolute deviation).
* `bootstrap_by_group(values, groups, stat=np.mean, n_boot=1000, seed=0, ci=0.95)`:
  resample **drivers** (groups) with replacement; vectorised through per-group
  sums and counts for mean-type statistics; returns point estimate, CI, the
  bootstrap replicates. `bootstrap_ratio_by_group(num, den, groups, ...)` for
  variance ratios (ratio of group-resampled means).
* `paired_bootstrap_test(values_a, values_b, groups, ...)`: difference of means
  over the same resampled drivers, CI and two-sided p-value;
  `holm(pvalues, alpha=0.05)` returns adjusted p-values and reject flags.

## 7. Models

Interface: `cf_unc/models/base.py::CFModel` (read it). Shared parts:
`models/nets.py` (Normalizer, LSTMEncoder), `train/trainer.py` (fit loop).

Every model: `fit(train: WindowSet, val: WindowSet, ...)`, `predict_dist`,
`init_sampler` / `sample_step` (torch, used by rollouts), `save` / `load`
(state dict + constructor kwargs in one `.pt` file, load reproduces
predictions bit-for-bit on the same device).

Normalisation: inputs standardised with training mean/std per feature; NN
outputs live in standardised target space and are mapped back
(`mu = a_mean + a_std * mu_n`, `sigma = a_std * sigma_n`). Everything returned
to the caller is in physical units.

| Model | Predictive law | Aleatoric | Epistemic |
|---|---|---|---|
| `HeteroMDN(n_components=1|3)` | Gaussian / mixture, LSTM(64) + MLP head, NLL loss, `log_sigma` clamped to [-7, 3] (normalised space) | its own variance | none (`None`) |
| `DeepEnsemble(n_members=5)` | pooled mixture of members; member `m` has seed `seed*1000+m` and a bootstrap resample of **drivers** | mean member variance | variance of member means |
| `MCDropout(p=0.1, n_passes=50)` | pooled mixture of stochastic passes; dropout after the LSTM state and after each head layer, active at inference | same | same |
| `GPResidualIDM` | Gaussian; mean `a_IDM + E[f]`, SVGP (GPyTorch), RBF-ARD on standardised `(s, dv, v)` of the **last** history sample, 300 inducing points (k-means++ or random subset init, learned), Gaussian likelihood | learned noise variance | latent posterior variance `Var[f(x)]` |
| `QuantilePIDL` | `QuantileDist` at levels {0.05, 0.25, 0.5, 0.75, 0.95} | variance of the interpolated law | none |
| `DiffusionCF` | `SampleDist` of the first step of 32 sampled 5 s sequences | sample variance | none |
| `IDMWhiteNoise` | Gaussian, mean IDM, constant sigma = RMS residual on train | sigma^2 | 0 |
| `LSTMConstSigma` | Gaussian, mean LSTM (MSE loss), constant sigma = RMS residual on validation | sigma^2 | 0 |

Sampler modes (`init_sampler(n_traj, generator, mode)`):

* `"per_step"` (spec default): acceleration drawn independently each step from
  the full predictive law.
* `"per_trajectory"`: the epistemic draw is frozen along a trajectory
  (ensemble member index / dropout masks / GP inducing values), only the
  aleatoric part is redrawn each step.
* `GPResidualIDM`, `"per_trajectory"` = joint-posterior approximation (the
  default of closed-loop rollouts for the GP): draw the whitened inducing
  values once per trajectory, then at every step
  `f(x) | u ~ N( K_xz K_zz^{-1} u , diag(K_xx - K_xz K_zz^{-1} K_zx) )` plus
  observation noise. `"per_step"` draws from the marginal law independently.

### QuantilePIDL loss
```
L = mean_q pinball_q(y, q_hat)                                   (data)
  + w_phys * mean( (q_hat_0.5(x_c) - a_IDM(x_c))^2 )             (physics, IDM frozen)
  + w_nc   * mean( sum_j relu(q_hat_j - q_hat_{j+1} + margin)^2 ) (non-crossing)
```
Collocation windows `x_c`: half observed windows, half synthetic windows drawn
uniformly in `s [2, 120] x dv [-5, 5] x v [0, 40]` and extended backwards with
constant speeds (`s_{t-k} = s_t + k DT dv`, clipped at 0.5 m), which is
kinematically consistent. No public code of Liu et al. (2023) exists; this is
a re-implementation of the described recipe.

### DiffusionCF
* Data: `y_seq (N, 50)` accelerations, condition = history `(N, 30, 3)` and
  `v_lead_future (N, 50)` given **relative to the current follower speed**.
* Denoiser: LSTM(64) history encoder -> context; sinusoidal step embedding;
  1-D residual conv stack (64 channels, kernel 5) over the 50-step sequence
  with input channels `[x_k, v_lead_rel]`, FiLM conditioning on
  context + step embedding. Epsilon-prediction, MSE loss, EMA of weights.
* Schedule: cosine, `T_train = 1000`; sampling on a 50-step sub-sequence
  (DDIM update with `eta = 1`, configurable), `x0` clipped to the normalised
  acceleration range.
* `predict_dist(hist, v_lead_future=...)` -> first-step samples;
  `sample_sequences(hist, v_lead_future, n_samples)` -> `(N, n_samples, 50)`.
  If `v_lead_future` is `None`, a constant leader speed is assumed.

## 8. Tests (pytest)

Markers: `slow` (anything that trains for more than a few seconds), `data`, `gpu`.
Unit tests must run in seconds; statistical tests use fixed seeds and
tolerances with a stated safety margin (no flaky tests).

## 9. Engineering rules

* Python 3.11, type hints, short docstrings stating shapes and units.
* No new dependencies (disk space is scarce). Allowed: numpy, scipy, pandas,
  pyarrow, torch, gpytorch, hydra-core, matplotlib, tqdm, pytest.
* Interpreter: `.venv/Scripts/python.exe` (CPU build of torch). Run tests with
  `PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest ...` from the repository root.
* The repository path contains non-ASCII characters; always use `pathlib`
  and `encoding="utf-8"`.
* Do not touch files owned by another task, do not edit the sibling
  repositories `cf-stability` / `cf-identifiability`, do not commit.

## 10. Milestone M2: uncertainty maps, dataset shift, regimes (experiment E1)

### 10.1 Data

| Role | Config | Event set (imported from cf-stability) | Split |
|---|---|---|---|
| training / in-domain | `highd` | `data/events/follownet_highd` (FollowNet HighD events, stand-in for raw highD) | cf-stability fold file, fold 0 |
| in-domain, validation fold | `highd_val` | the same event set, fold 1 (placebo for the shift analysis, calibration set of M3) | - |
| shift | `ngsim` | `data/events/follownet_ngsim_i80` | all events evaluated |
| shift | `ngsim_us101` | `data/events/ngsim_us101` (reconstruction of the raw NGSIM US-101 by cf-stability; added at the M2 review) | all events evaluated |
| shift | `waymo` | `data/events/waymo` (Hu et al. 2022 pairs) | all events evaluated |
| shift | `openacc` | `data/events/openacc` | all events evaluated |

* Fold rule of cf-stability (`data/splits/<name>_driver.json`): fold `k` = test,
  fold `(k + 1) % n_folds` = validation, the rest = training.
* One acceleration definition for every data set: if `meta["a_source"]` is
  not `"savgol"`, the target is recomputed from the stored follower speed
  exactly as cf-stability does (`central_diff(savgol(v, 11 samples, order 2))`),
  so that aleatoric variances are comparable across data sets. The dataset's
  own acceleration stays available for a sensitivity check.
* Subsets used in the shift analysis: OpenACC by `meta["driver_mode"]`
  (`ACC`, `Human`, `unknown`), Waymo by `meta["follower_type"]` (`HV`, `AV`).
* Bootstrap units: events (specification of M2); drivers (`follower_id`) as a
  sensitivity analysis where several events share a follower (OpenACC: 49
  followers, Waymo: the AV).

### 10.2 Prediction tables (interface between the experiment runner and the analysis)

```
artifacts/e1/windows/<dataset>.parquet          one row per evaluated window, fixed row order
    event_index int32, t_index int32, event int32 (code of event_id), driver int32 (code of follower_id),
    fold int8 (fold of the cf-stability split file, -1 if unknown),
    subset (string), s, dv, v, v_lead, a_prev (= a[t-1]), y (= a[t])        float32
    y_prev (= the target one step of its own length earlier, the persistence predictor)   float32
        (added with the target option, section 11.1; tables written before lack it, readers must tolerate that)
artifacts/e1/windows/<dataset>_meta.json        event_ids, driver names, subset names, stride, counts, target
artifacts/e1/train_density.npz                  bin counts of the highD TRAINING windows (stride 1) on the map grid
artifacts/e1/preds/<model>/seed<k>/<dataset>.parquet     same row order as the windows table
    mean, var, aleatoric_var, epistemic_var (NaN when the model has none), nll, crps, covered (uint8), width, pit   float32
artifacts/e1/checkpoints/<model>/seed<k>/{model.pt, train_log.json, metrics.json}
```

`a_prev` is NaN for the first sample of an event (cannot happen with a 3 s history).

### 10.3 Maps (`cf_unc/maps/grid.py`)

* Grid: `s` 0..120 m step 5 (24 bins), `dv` -5..5 m/s step 0.5 (20 bins); bin
  of a window = bin of its LAST history sample; windows outside the grid go to
  one extra "outside" cell and are never dropped.
* Per bin, data set, model and seed: count, frequency in the evaluated data
  set, frequency in the highD training windows, mean epistemic, aleatoric and
  total variance, mean NLL, mean squared error. Seeds are averaged per bin
  (mean and standard deviation over seeds are stored).
* Rare flags, both relative to the TRAINING density:
  `rare_freq` = training frequency < 1 % (definition of the plan);
  `rare_mass` = the least frequent bins that together hold at most 10 % of the
  training windows (the outside cell counts as rare in both).
  Bins with fewer than 30 evaluated windows are masked in figures only.

### 10.4 Shift and concentration (`cf_unc/maps/shift.py`), hypothesis H3.1

* Ratio for a shifted data set D against in-domain I (highD test):
  `R = mean_D(x) / mean_I(x)` for `x` = epistemic and aleatoric variance.
  Two-sample bootstrap over events (D and I resampled independently, 1000
  replicates, the same resampled events for every seed); the estimate is the
  ratio of the seed-averaged means; the across-seed standard deviation of the
  per-seed ratios is reported next to the bootstrap interval.
* Bin-matched ratio (covariate-shift adjusted): per-bin means of D and I
  averaged with the in-domain bin frequencies over the bins that hold at least
  30 windows in both; tells state-distribution shift from a change of
  uncertainty at the same state.
* Concentration in rare states (in-domain and per data set):
  `mean(x | rare) / mean(x | not rare)` for both rare definitions, one-sample
  event bootstrap; share of the summed epistemic variance that sits in rare
  windows against their share of windows; the same for the named regions of
  the hypothesis, `s < 15 m` and `|dv| > 3 m/s`.
* Verdict per model and pair of data sets, thresholds from the plan:
  supported: lower bound of `R_epi` >= 3 and the interval of `R_alea` inside
  (0.7, 1.3); refuted: upper bound of `R_epi` < 1.5, or the interval of
  `R_alea` entirely above 2 or below 0.5; otherwise inconclusive. The verdict
  based on point estimates is reported as well. H3.1 needs support on at
  least two pairs.
* Models without an epistemic output (HeteroMDN, QuantilePIDL, DiffusionCF,
  LSTMConstSigma): the across-seed variance (ddof = 0) of the predictive mean
  is reported as "seed epistemic" and the seed-mean of the model variance as
  aleatoric; always labelled as such, never mixed with the model's own
  decomposition.

### 10.5 Regimes (`cf_unc/maps/regimes.py`), hypothesis H3.3

Definitions of cf-identifiability (`cf_ident/diag/regimes.py`) plus congestion:

```
congested    s < 15 and v < 8
braking      a_prev < -0.5
approaching  dv > 0.5
free         s > 60  or  (|dv| < 0.5 and v > 0.8 v0)
steady       |dv| < 0.5 and 10 < s < 60
other        everything else
```

* Partition by priority `congested > braking > approaching > free > steady`,
  first match wins; the overlapping masks are available too.
* `braking` uses the PREVIOUS acceleration `a[t-1]`, not the target `a[t]`:
  a regime label must be known before the outcome, otherwise conditioning on
  it truncates the target and biases NLL and residual variance of that regime.
* `v0` = free speed of the IDM calibrated on the highD training events, the
  same number for every data set.
* Statistics per regime: share of windows, mean predicted total / aleatoric /
  epistemic variance, mean squared residual, NLL, CRPS, PICP.
* Ordering: ratio of the mean predicted variance (and of the mean squared
  residual) of every regime to `steady`, event bootstrap with both regimes
  taken from the same resampled events, two-sided bootstrap p-value of
  "ratio = 1", Holm correction over the family of comparisons.
* Heteroscedastic against homoscedastic: per-window NLL of HeteroMDN minus
  LSTMConstSigma, paired by window and seed, event bootstrap; overall and per
  regime, per data set, Holm over the family. Reported together with the
  literal criterion of the plan (separate intervals of the two models do not
  overlap).

### 10.6 Decisions taken while running M2

* **Two IDM calibrations** (`artifacts/e1/idm.json`, 3000 training events each).
  `closed_loop`: pooled closed-loop spacing RMSE (the calibration of the plan and of
  cf-stability); physics of `gp_residual_idm`, `quantile_pidl` and of the baseline
  `idm_white_noise`; its `v0` defines the regime `free`. `mle`: one-step
  maximum likelihood of the white-noise model (least squares on accelerations);
  used by the additional baseline `idm_white_noise_mle` only, so that the
  baseline is not handicapped in one-step metrics.
* **Physics weight of QuantilePIDL** is selected on the highD validation fold
  (`scripts/select_w_phys.py`): smallest pinball loss among `w_phys > 0`, the largest
  value within 1 % of it. The default 0.3 of section 7 is not used in E1.
* **Data sets.** `ngsim` = reconstruction of the raw NGSIM I-80 by cf-stability;
  `ngsim_fn` = FollowNet events of the same site, a *cross-check*: reported, never
  counted as a second pair for H3.1 (`cross_check` in `configs/maps/default.yaml`).
  `highd_val` is the *placebo*.
* **Robustness of long runs.** The GPU is shared; a driver reset kills every CUDA
  context at once. `scripts/run_e1_retry.sh` restarts the resumable runner until
  nothing fails.
* **Supplementary analysis** (`scripts/analyze_e1_extra.py`): model-free
  predictability of the acceleration per data set (persistence), calibration of the
  predicted variance, rarity and matching of states in three dimensions (s, dv, v).
* **Environment.** One Python environment for all research_plans projects
  (`../cf-stability/.venv`, torch 2.11.0+cu128), attached through a thin `.venv`.

## 11. Milestone M3: closed-loop rollouts and calibration (experiment E2)

Decisions of the M2 review (2026-09-28): a behavioural target is added next to
the one-step target, NGSIM US-101 is added as a fourth shift pair.

### 11.1 Targets

| Name | Definition | Draw kept for | Roots |
|---|---|---|---|
| `a` | `a[t]`, acceleration over `[t, t + DT)` (Savitzky-Golay derivative of the speed) | 1 step | `artifacts/e1`, `artifacts/e2/a` |
| `mean_1s` | `(v[t + 10] - v[t]) / (10 DT)`, mean acceleration over the next second | 10 steps | `artifacts/e1_mean1s`, `artifacts/e2/mean_1s` |

* `make_windows(..., target="mean_1s")` needs `t + 10 < n`. The history, the
  state columns and `a_prev = a[t-1]` do not depend on the target.
* Every model is trained once per target and seed, except DiffusionCF: it
  generates 5 s sequences of one-step accelerations, its law of `mean_1s` is the
  mean of the first 10 steps of every sampled sequence (no retraining).
* IDM-based models use the instantaneous IDM acceleration as the mean of both
  targets; the noise level is fitted per target.
* In a rollout a draw of a `mean_1s` model is an acceleration command that is
  kept for 10 steps (then the mean acceleration of the rollout over a second has
  exactly the predicted law); a draw of an `a` model is kept for one step.
* Implementation (E1 runner): config key `train.target` (default `a`) defines the
  training and the evaluated windows (`y` of the windows tables) and is checked on
  resume (windows spec, training density, `target` of the checkpoint logs; files
  written before the option count as `a`). `y_prev` of the windows tables is the
  target at `t - 1` (`a`) or `(v[t] - v[t-10]) / (10 DT)` (`mean_1s`).
  `DiffusionCF.predict_dist(..., target=...)`; the runner passes the target to every
  `predict_dist` with a `target` parameter. `reuse_checkpoints: {model: <root>/checkpoints}`
  copies a checkpoint from another root instead of training it (kwargs, IDM and,
  for one-step models, target checked). `configs/experiment/e1_mean1s.yaml` inherits
  `e1.yaml` (`base:`); `--pilot` reuses from `<root>_pilot/checkpoints`.

### 11.2 Rollout engine (`cf_unc/rollout/engine.py`)

For event `e` with start index `t0` (default `HIST_LEN - 1`): the history is the
true `(s, dv, v)[t0 - H + 1 : t0 + 1]`, the leader is given by
`x_lead[t0 :]`, `v_lead[t0 :]`, the follower starts at `x_follower[t0]`, `v[t0]`.

```
step k = 0 .. K-1            (K = horizon / DT = 100)
  if k % hold == 0:  a_cmd = model.sample_step(hist, state, generator)     one draw per trajectory
  a      = clip(a_cmd, A_MIN, A_MAX)
  v'     = max(v + a DT, 0);   x' = x + v' DT
  s'     = x_lead[t0 + k + 1] - x';   dv' = v' - v_lead[t0 + k + 1]
  hist   = (hist[1:], (s', dv', v'))
  first collision step = k + 1 if s' <= 0 and none before
```

* Trajectories = events x rollouts, flattened and processed in batches of at most
  `max_batch` trajectories on the device of the model; float32.
* After a collision the trajectory continues (the gap may be negative); the
  model input gap is floored at `S_EPS` so that no model receives a
  non-positive gap. Collisions are reported, never removed.
* Sampler mode per model: `per_step` (specification) for every model except
  `gp_residual_idm`, which uses `per_trajectory` = joint-posterior approximation
  (specification). `per_trajectory` for the ensemble and MC dropout is a
  sensitivity variant. DiffusionCF uses its plan sampler (`v_lead` of the
  rollout, re-planning every 5 s) with 32 rollouts per event instead of 200.
* Reproducible from `(seed, event order)`; the generator lives on the device.

### 11.3 Rollout tables

```
artifacts/e2/<target>/events/<dataset>.parquet       one row per rolled event: event (code), event_id, driver, subset, fold, t0, order
artifacts/e2/<target>/rollouts/<model>/seed<k>/<dataset>.parquet    one row per (event, horizon), events in the order of the events table
    event int32, horizon_s float32, truth_s, truth_v,
    s_q05, s_q25, s_q50, s_q75, s_q95      quantiles of the rollout gaps, (S + 1) plotting positions (numpy method "weibull")
    s_lin05, s_lin95                        the same with linear interpolation (for comparison only)
    s_mean, s_std, s_min, s_max, v_mean, v_std
    rank int32                              number of rollouts with gap <= truth_s
    n_rollouts int32, crps_s                fair sample CRPS of the gap
    collided float32                        share of rollouts with a collision up to this horizon
```

### 11.4 Coverage, conformal calibration, ACI (`cf_unc/rollout/calibration.py`)

* **Raw coverage** per horizon (1, 3, 5, 10 s): PICP and mean width of
  `[s_q05, s_q95]`, CRPS, collision share; event bootstrap.
* **One-step split conformal** (in-house): calibration set = highD validation
  fold, scores `|y - mu|` ("absolute") and `|y - mu| / sigma` ("normalised"),
  quantile of rank `ceil((n + 1)(1 - alpha))`; intervals `mu +- q` and
  `mu +- q sigma`; coverage and width on highD test and on the shifted sets.
  Works on the E1 prediction tables (`mean`, `var`), no model is needed.
* **Score of a rollout interval** at horizon h, with `m = s_q50`, `lo = s_q05`,
  `hi = s_q95` and half-widths floored at 1 mm:
  `S = max((lo - y) / (m - lo), (y - hi) / (hi - m))` (negative inside the
  interval). The interval with correction `Q` is
  `[lo - Q (m - lo), hi + Q (hi - m)]`; it contains `y` iff `S <= Q`.
* **Static correction per horizon**: `Q_h` = conformal quantile of the scores of
  the calibration events (highD validation).
* **ACI per horizon** (Gibbs and Candes 2021): events in a seeded random order,
  `alpha_1 = alpha`, `Q_e` = quantile of level `1 - alpha_e` of the scores seen so
  far (level clipped to [0, 1]; level 1 = largest score seen; fewer than
  `n_min = 20` scores: `Q_e = 0`), `alpha_{e+1} = alpha_e + gamma (alpha - err_e)`,
  `err_e = 1[S_e > Q_e]`. Run over the calibration events, then applied to the
  evaluation events in two variants: `frozen` (final level and calibration scores,
  no update) and `online` (the recursion continues over the evaluation events,
  their scores are appended after use). `gamma` from the config (default 0.01,
  sensitivity 0.005, 0.02, 0.05).
* **H3.4**, per model and target, thresholds of the plan: (i) one-step conformal
  coverage on highD test within 0.90 +- 0.03; (ii) raw rollout coverage of the
  90 % interval below 0.80 at some horizon; (iii) corrected coverage >= 0.87 at
  every horizon. The hypothesis is refuted when the raw coverage is >= 0.87 at
  every horizon. Reported for the in-domain test set and for every shifted set.

## 12. Harmonised signals (decision of the M3 review)

The data sets were recorded and smoothed differently; models trained on highD learn the smoothness
of its signals (docs/m2_report.md, docs/m3_report.md). Every event of every data set is therefore
brought to ONE signal processing before windows are cut. The stored events stay untouched; the
transform is applied at load time when the experiment config says `signal: harmonised`.

```
1. v_h       = SG(v),  v_lead' = SG(v_lead)      Savitzky-Golay, 11 samples (1.1 s), order 2, mode "interp"
2. a_h       = central differences of v_h         (the acceleration definition used so far, unchanged)
3. I[0] = 0, I[t] = I[t-1] + (v_lead'[t] - v_h[t]) DT          relative displacement, scheme of section 1
   least squares over the event:  s[t] ~ s0 + c (t DT) + I[t]   -> s0, c
4. v_lead_h  = v_lead' + c                        the constant c is attributed to the leader's speed
   s_h[t]    = s0 + c (t DT) + I[t]               kinematically consistent gap: s_h[t+1] - s_h[t] = -dv_h[t+1] DT
   dv_h      = v_h - v_lead_h
   x_lead_h, x_follower_h integrated with the scheme of section 1, s_h = x_lead_h - x_follower_h
5. meta: signal = "harmonised", gap_offset s0, leader_speed_offset c, gap_rms = rms(s - s_h)
```

* An event is dropped when `min(s_h) <= 0.5 m`, `|c| > 0.3 m/s` or `gap_rms > 2 m`; the numbers
  dropped per data set and per reason are reported. More than 5 % dropped in any data set is a
  finding to report, not a reason to change the thresholds.
* Roots: `artifacts/e1h` (target `a`), `artifacts/e1h_mean1s`, `artifacts/e2h`; everything else
  (splits, windows, models, seeds, analysis) is identical to E1 / E2, so that the two signal
  variants can be compared row by row. `w_phys` of QuantilePIDL is selected again per target.
* Restated hypothesis H3.1e (epistemic part only, decision of the M3 review): the epistemic
  variance concentrates in rare states (ratio inside / outside the lowest 10 % of mass, lower
  bound of the interval > 1) and grows under shift (lower bound of `R_epi` >= 3 on at least two
  counted pairs; refuted when the upper bound is below 1.5 on any pair). The verdict of H3.1 as
  written in the plan stays in the tables.

### 12.5 Deviations recorded during the implementation (2026-10-05)

* Smoothed speeds are floored at 0; an event whose harmonised signals are not finite is dropped
  with the reason `non_finite` (next to the drop reasons of 12.2).
* H3.1e is also refuted when the upper bound of the concentration ratio is < 1 (the epistemic
  variance would then be *less* concentrated in rare states under shift).
* E2 refuses checkpoints of another signal variant (`signal` of the checkpoint's config must equal
  the experiment's); `scripts/analyze_e2.py` reads the E1 roots of the one-step conformal
  calibration from `config.yaml` of the E2 root; `scripts/run_e1.py --dry-run` prints the plan.
* Findings, not bugs: the harmonisation drops 6.7 % of the OpenACC events (above the 5 % mark of
  12.2; reported in `artifacts/e1h/harmonisation.json`) and does not equalise the roughness of the
  inputs: the NGSIM reconstructions stay about 10x rougher than highD after SG(11, 2).

## 13. Milestone M4: platoon-level distributional evaluation (experiment E3)

### 13.1 Profiles (`cf_unc/platoon/openacc.py`, `configs/platoon/profiles.yaml`)

* Source: raw OpenACC files of the campaigns AstaZero, ZalaZone, Vicolungo (read-only, in
  `../cf-stability/data/raw/openacc`). A *platoon run* is one file: time, and for every vehicle in
  platoon order its speed, the gap to its predecessor and the driver mode.
* A *profile* is a window of a run: 30 s <= length <= 120 s, every vehicle observed without gaps,
  speeds above 1 m/s, leader amplitude `max(v_0) - min(v_0)` between 2 and 15 m/s, at least 3
  followers. Windows of one run do not overlap. Signals are harmonised as in section 12 (speeds
  with the same Savitzky-Golay filter).
* Selection is deterministic: candidate windows are ranked by leader amplitude within each
  (campaign, driver mode) and taken round-robin over these groups. The first 10 are the *headline
  set* (number of the plan), all selected windows (at most 60) are the *extended set*. The
  selection is written to `configs/platoon/profiles.yaml` (file, start, length, campaign, mode,
  number of vehicles), which is the list the experiment reads.
* `configs/empirical_growth.yaml`: optional amplitudes or speed standard deviations by vehicle
  index from platoon experiments (Jiang et al. 2014/2015); empty template, used when filled.

### 13.2 Platoon simulation (`cf_unc/platoon/simulate.py`)

* 20 followers behind the leader profile; 100 stochastic rollouts per profile (DiffusionCF 32).
* Time line: 3 s of steady history + burn-in of 30 s with the leader at the first speed of the
  profile + the profile + a tail of 40 s at the last speed of the profile (propagation).
* Initial state of every follower: speed of the leader, gap = equilibrium gap of the closed-loop
  calibrated IDM at that speed; the burn-in lets every model settle at its own equilibrium.
* Cascade: vehicle i is simulated over the whole time line behind the simulated path of vehicle
  i - 1 of the SAME rollout (`rollout_events(..., leader_override=..., keep_paths=True)`);
  information flows downstream only, so the cascade equals a joint simulation. Draws of different
  vehicles are independent (every vehicle is another driver).
* Models and samplers (decision of the M3 review):

| Label | Checkpoint | Sampler | Hold |
|---|---|---|---|
| `idm_white_noise` | target `a` | per step | 1 |
| `lstm_const_sigma` | target `a` | per step | 1 |
| `deep_ensemble` | target `a` | member kept per rollout and vehicle | 1 |
| `gp_residual_idm` | target `a` | joint-posterior approximation per rollout and vehicle | 1 |
| `quantile_pidl` | target `mean_1s` | per step | 10 |
| `diffusion_cf` | target `a` | plan sampler, re-planning every 5 s | 1 |

  Seeds 0-4. Both signal variants (`artifacts/e1`, `artifacts/e1h`) when their checkpoints exist.

### 13.3 Evaluation (`cf_unc/platoon/evaluate.py`), hypothesis H3.2

* Response window of vehicle i: from the start of the profile to the end of the tail.
  Amplitude `A_i = max(v_i) - min(v_i)`, speed standard deviation `sd_i = std(v_i)` over it.
  Empirical values from the measured followers of the profile (same window without the tail).
* CRPS of the model amplitude law (rollouts x seeds of a profile) at the empirical amplitude, per
  profile and vehicle index (fair sample CRPS); averaged over the indices that exist in the data
  and over profiles; the same for the amplitude ratio `A_i / A_0`.
* H3.2: relative reduction `1 - CRPS_model / CRPS_idm_white_noise` with a bootstrap over profiles
  (1000 replicates, paired by profile); supported when the lower bound is >= 0.20, refuted when
  the upper bound is < 0.05; reported for the headline and the extended set, by driver mode, and
  against `lstm_const_sigma` as well.
* Intervals: coverage of the 90 % amplitude interval by profile and index, raw and with a
  conformal correction (score of section 11.4; leave-one-profile-out static correction and ACI
  online over the profiles of the extended set).
* Growth: `sd_i` against the vehicle index 1..20 (mean over rollouts, seeds and profiles) with
  the exponent of `sd_i = sd_1 i^beta` (least squares in logs; concave growth: `beta < 1`,
  bootstrap over profiles) and the sign of the second differences; the empirical curve for the
  indices that exist.
* Collisions: share of rollouts with a collision of any pair, and by vehicle index.

### 13.4 Deviations recorded during the implementation (2026-10-05)

* Leader speed of a profile <= 28.68 m/s: the closed-loop IDM of `artifacts/e1/idm.json` has
  `v0 = 28.9 m/s`, its equilibrium gap reaches the cap of 120 m there and does not exist above
  `v0`; this drops most Vicolungo highway windows. The initial gap comes from this IDM for both
  signal variants.
* Profile windows start at the leader's last steady start (range of the speed over 5 s
  <= 0.5 m/s), end steady, and the largest platoon that reaches 60 s is preferred. 43 of the 95
  OpenACC runs have no leader plus three followers (the ZalaZone handling runs have no vehicle 1).
* One random stream per (batch of profiles, vehicle), not per profile: a per-profile stream would
  need one model call per profile and step. The batches are fixed (the run set, 60 profiles), so
  a resumed unit is bit-identical; the random numbers are shared across models and variants.
* CRPS: the predictive law of an amplitude is pooled over rollouts x seeds (fair CRPS), averaged
  per profile and then over profiles; the bootstrap and the paired H3.2 ratio are over profiles; a
  verdict needs >= 5 profiles. Intervals use (S + 1) quantiles; the static correction is
  leave-one-profile-out and ACI runs in a seeded profile order, per index.
* Growth exponent by least squares in logs; the measured curve is drawn for indices with >= 3
  profiles, but its exponent is fitted only over the indices measured by at least half of the
  profiles of the set (a pooled fit over the ragged indices mixes the lower level of the few long
  platoons into the slope: beta -0.75 instead of about +0.1 on the extended set); the data window
  (profile without the tail) is a sensitivity.
* DiffusionCF's 5 s plan sees the predecessor's *simulated* future (the cascade hands every
  vehicle the full predecessor trajectory), so for it the cascade is not a joint simulation.

### 13.5 Sensitivities of E3 (decision at the pilot, 2026-10-05)

The pilot (`artifacts/e3_pilot`, headline set, 20 rollouts) showed that the LSTM-based models
(`lstm_const_sigma`, `deep_ensemble` with the kept member, `quantile_pidl`) leave the training
state region in almost every rollout: the headline profiles start at 13-14 m/s with equilibrium
gaps of 7-22 m, and from there the models either close the gap (collision) or command +4 m/s^2
without limit (speeds above 500 m/s, through the leader; the kept-member ensemble collides in
38.5 % of the single-vehicle rollouts within the 30 s constant-speed burn-in, per-step draws in
3.5 %). The headline run stays as specified in 13.2 (no speed cap, the model set of 13.2); this
failure is the result. Two sensitivities on the headline set with the raw checkpoints only:

* `configs/experiment/e3_vcap.yaml` (root `artifacts/e3_vcap`): follower speed capped at
  35.9 m/s, the largest speed of the highD training events (`platoon.v_max`, implemented as an
  optional cap in `RolloutConfig.v_max`, default none; E2 is unchanged), for `idm_white_noise`,
  `lstm_const_sigma`, `deep_ensemble`, `quantile_pidl`. It bounds the amplitudes of the runaway
  rollouts; it does not rescue a model that leaves the leader behind.
* `configs/experiment/e3_step.yaml` (root `artifacts/e3_step`): the deep ensemble with per-step
  draws (`deep_ensemble__step`: member and noise redrawn every step) next to `idm_white_noise`.

## 14. Milestone M5: report (decisions of the M4 review, 2026-10-07)

### 14.1 Decision: the H3.2 verdict is given per driver mode

The headline verdict of H3.2 is stated per driver mode of the followers of a profile: `ACC`
(47 profiles of the extended set) and `Human` (12), with the thresholds of 13.3 unchanged
(supported if the lower bound of the paired bootstrap reduction is >= 0.20, refuted if the upper
bound is < 0.05, otherwise inconclusive; a mode needs >= 5 profiles, so `mixed` (1 profile) is
listed but not judged). The pooled verdict over all 60 profiles stays as a secondary line. Reason:
the two populations differ in what the reference achieves (`idm_white_noise` CRPS 1.1 m/s on the
human-driven platoons against 3.4 m/s on the ACC platoons) and the plan's platoon experiments are
ACC campaigns. Both signal variants are reported; the recorded signals are the headline, the
harmonised ones the replication.

### 14.2 Deliverables

* `scripts/make_report.py` -> `artifacts/report/`: `tables/<name>.csv` and `tables/<name>.tex`
  (booktabs, 2-3 significant digits, labels of `MODEL_LABEL` / `MODEL_LABEL_E3`), `figures/*.png`
  (copied from the experiment roots or generated here), `index.md` (every table and figure with a
  one-line caption and the artifact it was built from). The script only reads the tables of
  `artifacts/e1*`, `e2*`, `e3*`; it runs no experiment and no bootstrap.
* Tables: T1 one-step metrics (NLL, CRPS, PICP and width of the 90 % interval, RMSE) per model
  and data set, mean +- sd over seeds, for both targets and both signal variants; T2 H3.1 (geometric
  epistemic and aleatoric ratios per shifted set, verdict) and H3.1e (concentration ratio with CI,
  verdict); T3 H3.3 (NLL of HeteroMDN against LSTMConstSigma by regime with the Holm-corrected
  test; variance ordering by regime); T4 H3.4 (coverage of the 90 % gap interval at 1/3/5/10 s raw,
  static, ACI frozen, ACI online, per model and data set; one-step conformal coverage; verdict
  counts); T5 H3.2 per driver mode (CRPS, reduction with CI, verdict; all models, both variants,
  the two sensitivities of 13.5 on the headline set); T6 platoon diagnostics (collisions by vehicle
  index, amplitude interval coverage raw / static / ACI, growth exponents of the models and of the
  measured platoons).
* Figures: F1 uncertainty maps (epistemic and aleatoric heat maps per model and data set, from
  `artifacts/e1/figures/maps`), F2 variance ratio bars under shift, F3 NLL by regime, F4 coverage
  by horizon before and after ACI, F5 platoon amplitude CRPS bars per driver mode (new,
  `crps_bars__<variant>.png`: one bar per model and mode, whiskers = 95 % bootstrap interval, the
  reference as a line), F6 growth curves, F7 CRPS by vehicle index.
* `scripts/analyze_e3.py`: the printed summary and `tables/summary.md` give the per-mode block
  (ACC, Human, all) for every variant; `h32.csv` is unchanged.
* `docs/report.md` (written by hand, not by a script): hypotheses, data, models,
  experiments, results by hypothesis with the verdict tables, deviations from the plan,
  limitations, reproduction commands. No new experiment is run in M5.

## 15. Follow-up experiment E5: roughness-matched signals (decision of the M5 review, 2026-10-07)

### 15.1 Question and design

Section 12 showed that one Savitzky-Golay window on every data set leaves the NGSIM
reconstructions 10x rougher than highD and changes no verdict (M4 report, 3.6). E5 asks whether
the over-confidence of the history-based models under shift is explained by that remaining
roughness difference: every shifted data set is smoothed until the roughness of its speed signal
matches the harmonised highD training signal, the models of `artifacts/e1h` / `e1h_mean1s`
(trained on harmonised highD, window 11) are evaluated again without retraining, and the rollouts
of E2 are repeated. No model is retrained; the reference data set keeps its window.

* Signal variant `matched` = the harmonised processing of section 12 with a window per data set:
  the reference (`match.reference`, highD) keeps `harmonise.window` (11); every other data set
  gets the smallest odd window of `match.grid` (11, 15, 21, 25, 31, 41, 51, 61, 81, 101 samples)
  whose roughness is at most `(1 + match.tolerance)` (0.1) times the reference roughness; if no
  window of the grid reaches it, the largest is taken and flagged. Roughness = the `v` entry of
  `scripts/harmonise_report.py:roughness` (rms of the second difference of the smoothed follower
  speed over all events of the data set, after the drop rules of 12.2). The windows are computed
  once by `scripts/match_roughness.py --config <experiment config>` and stored with the roughness
  of every grid value in `<root>/matching.json`; the runners read that file and refuse to run
  without it. The matching uses the data set as a whole (all events), never the test split alone.
* Events shorter than their data set's window are dropped with the reason `short_event` (reported
  next to the drop reasons of 12.2); the share must stay below 5 % per data set, else the window
  is capped at the largest grid value that keeps it below and the cap is reported.
* Signal keys (12.3): for data set `name` the key is `{"signal": "harmonised", "harmonise":
  {... "window": w_name}}`, so the reference data set's key equals the harmonised key, the
  checkpoints of `e1h` / `e1h_mean1s` are reused legitimately (`reuse_checkpoints`, checked against
  the key of the training data set) and the tables of the shifted sets are new. E2 checks the
  checkpoint's signal against the key of the training data set, not against the experiment level.
* Roots: `artifacts/e1m` (target `a`, every model reused from `e1h`), `artifacts/e1m_mean1s`
  (reused from `e1h_mean1s`, DiffusionCF from `e1h`), `artifacts/e2m` (E2 with the `e1h` /
  `e1h_mean1s` checkpoints, all 5 data sets, both targets). E3 is not repeated (its verdicts did not
  move between the recorded and the harmonised signals; the platoon profiles are already smoothed).
* Analyses: the E1 analyses (maps, H3.1, H3.1e, H3.3, extra tables with the persistence baseline)
  on `e1m` and `e1m_mean1s`, the E2 analysis on `e2m`; `scripts/make_report.py` adds the matched
  roots to T1, T2, T4 and F2-F4 when they exist. The report compares recorded -> harmonised ->
  matched for NLL, PICP, RMSE and the persistence RMSE of the target under shift: the persistence
  RMSE states how much easier the smoothed target is, the model NLL at equal task difficulty
  whether the models remain over-confident.

### 15.2 Reading of the outcome (fixed in advance)

* If the NLL of the history-based models on the matched NGSIM sets falls to within 1 nat of their
  in-domain NLL and their PICP returns to 0.80-0.95, the over-confidence under shift was a
  roughness artefact of the source pipelines.
* If the NLL stays above the in-domain value by more than 5 nats or the PICP below 0.6 while the
  persistence RMSE of the matched target is within 2x of highD's, the over-confidence is behavioural
  (a genuine shift of the driving signal), and the paper's finding stands.
* In between: partial explanation, reported with the numbers.
* The behavioural target `mean_1s` is the robust reading (it depends on the smoothing much less
  than the one-step acceleration); the one-step target is reported next to it.

### 15.3 Deviations recorded during the implementation (2026-10-07)

* The tolerance of 15.1 is reached only by OpenACC (window 61, 1.06x the reference roughness);
  NGSIM I-80, US-101, the FollowNet I-80 set and Waymo get the largest grid window (101 samples,
  10.1 s) and the flag `not_reached` (roughness 1.4-1.5x the reference for the NGSIM sets, 1.9x
  for Waymo, which stops improving from window 61 on). The persistence RMSE of the matched targets
  is within 1.05-1.5x of highD's, so the task-difficulty condition of 15.2 holds.
* The `short_event` rule applies to every harmonised window, not only to the matched variant (the
  processing stays a function of the signal key); no event is shorter than 150 samples, so it
  drops nothing here. The extra `min_gap` drops at window 101 raise the NGSIM I-80 drop share from
  0.26 % to 2.3 %; OpenACC stays at 6.6 %.
* E2 gains the key `train: {data: highd}` (the training data set whose signal key the checkpoints
  are checked against); `matched` is allowed at the experiment level only; one matching file
  (`artifacts/e1m/matching.json`) serves `e1m`, `e1m_mean1s` and `e2m`, and
  `scripts/match_roughness.py` refuses to overwrite it with other settings unless forced.
* The in-domain tables (highD test and validation, 100 of 350 per root) are recomputed although
  their spec equals the harmonised one; they must equal the `e1h` tables, which is a consistency
  check of the reuse.

## 16. Follow-up experiment E6: training on the matched signals of a pooled domain (decision of the E5 review, 2026-10-07)

### 16.1 Question and design

E5 (section 15) left a behavioural remainder after the roughness was matched: 1-4 nats of
over-confidence and 9-22 % collisions on NGSIM for models trained on highD alone, and a state
shift on OpenACC (LSTM NLL 294, variance ratios in the hundreds). E6 asks whether that remainder
is a lack of behavioural diversity in the training domain: the same ten model configurations are
trained on the matched signals of several data sets at once.

* Signals: the `matched` variant of section 15 with the windows of `artifacts/e1m/matching.json`
  (highD 11, OpenACC 61, the NGSIM sets and Waymo 101); nothing is re-matched.
* Three roots, target `a` (comparable with `artifacts/e1m`), 5 seeds, the model set of E1,
  QuantilePIDL with `w_phys = 0.001` (the harmonised selection, not re-selected), DiffusionCF
  trained on the pooled sequences:
  - `artifacts/e6_pool`: training folds of highD, NGSIM I-80, NGSIM US-101, Waymo and OpenACC
    together (the driver-disjoint fold files of cf-stability, fold 0: fold 0 test, fold 1
    validation, the rest training, for every set); validation = the union of the validation folds.
  - `artifacts/e6_lodo_ngsim`: the same without NGSIM I-80 (the roughness shift of E5).
  - `artifacts/e6_lodo_openacc`: the same without OpenACC (the state shift of E5).
* Budget: every training set contributes at most `train.max_windows_per_dataset` = 150 000
  one-step windows (uniform subsample with a fixed seed after the stride of 2; `train.
  max_sequences_per_dataset` = 40 000 for the sequence model); the validation folds are capped the
  same way. The IDM calibrations are fitted on the pooled training events (sampled as in E1).
* Evaluation tables per root: every training set on its test fold (table `<name>`), the union of
  those test folds as the table `pool_test` (the in-domain reference of the analyses), the held-out
  set of a LODO root on all its events, and `ngsim_fn` on all events (cross-check of the NGSIM
  LODO). E2 rollouts for every root (`artifacts/e2_pool`, `e2_lodo_ngsim`, `e2_lodo_openacc`):
  the pooled root on the test fold of every data set, a LODO root on its held-out set and on the
  highD test fold; the events of a training set are drawn from its test fold only. E3 is not
  repeated.
* Analyses: the E1 analyses with `--in-domain pool_test` (H3.1 / H3.1e pairs: the held-out set
  of a LODO root; the pooled root has no shift pair), H3.3 on `pool_test` and per data set, the E2
  analysis (H3.4) per root, and the comparison with `artifacts/e1m` / `e2m` (same signals, highD-only
  training): NLL, PICP, RMSE per data set, collisions and ACI widths under rollouts.

### 16.2 Reading of the outcome (fixed in advance)

* Pooled root: "calibrated everywhere" is supported when every test fold has PICP of the 90 %
  interval within 0.85-0.95 and an NLL within 1 nat of the `pool_test` value for the model;
  otherwise the failing data sets are named.
* LODO roots: the transfer gap is NLL(held-out) - NLL(`pool_test`), compared with the gap of
  `e1m` (held-out NLL - highD NLL, 1-4 nats on NGSIM). A gap below 1 nat and a held-out PICP of
  0.85-0.95 mean that the remainder of E5 was the narrowness of the training domain; a gap at or
  above the `e1m` gap means that the held-out set is behaviourally distinct from the other four as
  well; in between: partial.
* Rollouts: collisions and the 1 s / 10 s coverage on the held-out set against `e2m`.
* Cost estimate: one root is about 1.6x the training of E1 (about 5 h of a free GPU with the
  three-process split); the three roots run one after another, then the rollouts.

### 16.3 Deviations recorded during the implementation (2026-10-07)

* The closed-loop IDM calibration on the pooled training events is degenerate (`v0` at the upper
  bound of 45 m/s, `b` at the lower bound of 0.3 m/s^2, one-step RMSE 2.45 m/s^2 against 1.10 for
  highD alone; the spacing objective over 36 min of mixed highD, NGSIM, Waymo and OpenACC
  following has no sensible optimum, and the padding of the events to the longest one makes the
  fit 20x slower). Decision: the E6 roots reuse the IDM calibrations of `artifacts/e1m/idm.json`
  (highD, harmonised window 11; `train.idm_from`) as a frozen physics prior identical to the one
  the E5 models used, so that E6 against `e1m` isolates the effect of the pooled training of the
  learned parts; the white-noise sigma of the IDM baselines is still fitted on the pooled windows.
  The pooled attempt is kept for the record (`artifacts/e6_pool_pilot/idm_pooled_attempt.json`).
* `highd_val` is evaluated as well (the calibration set of the one-step conformal step of the E2
  analysis and the placebo pair); `pool_test` is the row-wise concatenation of the test-fold
  tables (no separate prediction); the E2 runner refuses `split: all` or `train` for a training
  set of a pooled root. Inherited config entries can be removed by setting them to null.
* Caps of 16.1 applied: 150 000 one-step windows per set for training and validation (Waymo is
  under the cap with 46 028 / 18 058), 40 000 sequence windows; pool 646 028 training windows,
  each LODO root 496 028. The pooled training density (rare-state rules) uses the capped windows.
* `configs/maps/default.yaml` gains `cross_check_of: {ngsim_fn: ngsim}` (the cross-check pair is
  used only when `ngsim` is the held-out set of a LODO root); `scripts/analyze_e1.py --config`
  also accepts an experiment config.

### 16.4 Fourth root: both NGSIM sets held out (decision of the E6 review, 2026-10-08)

The NGSIM I-80 hold-out of 16.1 keeps US-101 (the same programme and processing) in training and
is therefore a weak domain hold-out. `artifacts/e6_lodo_ngsim_all` trains on the training folds
of highD, Waymo and OpenACC only (every other setting as in 16.1 and 16.3: matched signals, the
frozen IDM of `e1m`, caps, 5 seeds, target `a`) and evaluates NGSIM I-80 and US-101 on all their
events (two counted shift pairs against `pool_test`, `ngsim_fn` the cross-check of `ngsim`), the
test folds of the training sets and `pool_test`. Rollouts `artifacts/e2_lodo_ngsim_all`: the two
held-out sets and the highD test fold. Reading as in 16.2, with the `e6_lodo_ngsim` gap (US-101 in
training) as the third column: a gap that stays below 1 nat with both NGSIM sets unseen confirms
that the training-domain narrowness, not the NGSIM programme itself, was the remainder of E5.

## 17. Follow-up experiment E3-pooled: platoons with the models of the pooled domain (decision of the paper review, 2026-10-08)

### 17.1 Question and design

E6 left one claim untested at the platoon level: that the closed-loop instability of the one-step
networks is a property of the models, not of the data they saw (M7 report, finding 5, from the
10 s rollouts of E2). E3 is repeated with the checkpoints of two E6 roots as new signal variants of
`configs/experiment/e3.yaml`:

* `pool`: `artifacts/e6_pool/checkpoints` (OpenACC in training: the platoon profiles' population
  was seen);
* `lodo_openacc`: `artifacts/e6_lodo_openacc/checkpoints` (OpenACC held out: the honest transfer).

Every model of 13.2 takes its target-`a` checkpoint (E6 has no behavioural target), so
QuantilePIDL runs on target `a` with `hold_steps` 1 for these variants (deviation from 13.2,
reported); sampler modes as in 13.2; `idm_white_noise` of the same root is the reference of the
variant (its IDM parameters are the frozen `e1m` prior, its noise fitted on the pooled windows).
The leader profiles are the same 60 windows of 13.1 (same files, start samples and lengths), but
for these variants the leader speed is smoothed from the raw OpenACC files with the matched
OpenACC window of E5 (61 samples, `artifacts/e1m/matching.json`) instead of the window of 11,
because the pooled models were trained on roughness-matched signals; the profile selection is
not repeated. Initial gaps from `artifacts/e1/idm.json` as for every variant (13.4). Everything
else (followers, rollouts, time line, cascade, analyses, H3.2 per driver mode, growth,
collisions, intervals) as in sections 13 and 14.1. The `raw` and `harmonised` units of
`artifacts/e3` are untouched; the new units are written under the same root.

### 17.2 Reading of the outcome (fixed in advance)

* If, on the `lodo_openacc` variant, the share of rollouts with a collision of the first follower
  falls below 0.25 for the LSTM-based models (recorded signals: 0.49-0.60) and their H3.2 verdict
  per mode changes from refuted, the closed-loop instability was a training-domain effect.
* If that share stays above 0.5, the instability is a property of the one-step models and the
  claim of the M7 report stands at the platoon level; in between, partial.
* The `pool` variant (OpenACC seen) bounds what pooled training can achieve at best; the
  DiffusionCF and GP-residual verdicts per mode are compared with the recorded-signal verdicts of
  14.1; the growth exponents with the measured 0.12 [0.08, 0.16].

### 17.3 Deviations recorded during the implementation (2026-10-08)

* The initial gap of a windowed variant is the equilibrium gap (13.2) of the `artifacts/e1/idm.json`
  IDM at the variant's own re-smoothed initial speed, not the value of the recorded variant: 57 of
  60 profiles move by less than 0.06 m, three Vicolungo profiles near the IDM's `v0` by 0.1-4.2 m.
* The re-smoothed profiles are stored as `<root>/profiles__<variant>.yaml` with the SHA-256 of
  every raw file and are recomputed and compared at every run start; the window is read from
  `artifacts/e1m/matching.json`. Window 61 against 11 changes the leader amplitude by -0.03 m/s
  (median; range -0.58 to +0.04) and the initial speed by at most 0.07 m/s.
* The pilot of the new variants ran in a separate root (`artifacts/e3_pooled_pilot`, deleted after
  use) because the random streams are seeded per batch of the run set and a headline-only call in
  `artifacts/e3` would leave units that the full run cannot reproduce.
* `scripts/run_e3.py --dry-run` added; the per-variant model override (`models: {quantile_pidl:
  {target: a, hold_steps: 1}}`) is part of the unit spec.

## 18. Review controls of the v4 revision (decision of the pre-submission review, 2026-10-09)

Reviewer report `paper/submission/review_systems/REVIEW_SYSTEMS_RU.md` (experiments, items 1, 2, 3, 9, 10, 11), response
plan `RESPONSE_PLAN_RU.md` (R2, R3, R6 control, R10, R13, R14). Every control adds a model, a variant or a root; no
artefact of sections 10-17 is changed or invalidated (the dry runs of every existing root show their units done, the
new ones missing).

### 18.1 R2: DiffusionCF without the leader's future (`diffusion_cf_hist`)

* `DiffusionCFHist` = the DDPM of section 7 with the leader-future channel of the denoiser REMOVED (the input
  convolution sees the noisy sequence only; `Denoiser(leader=False)`), so the 5 s plan is conditioned on the 3 s history
  alone, the information of the one-step networks. Removal rather than a zeroed channel or a learned null token: no
  unused weights, and the information budget is explicit in the architecture. Same kwargs, sampler (32 samples, 50 DDIM
  steps, eta 1, re-plan every 5 s), training budget (12 epochs, EMA, batch 64) and training windows (seq_stride 5) as
  `diffusion_cf` (`configs/model/diffusion_cf_hist.yaml`, checked equal in tests); 320 fewer weights (216 065).
* A class attribute `plan_sampler` (DiffusionCF and DiffusionCFHist) now carries what `needs_leader_future` carried
  besides the input: sampled-row evaluation in E1 (`eval.max_rows_sampled`), `n_rollouts_sampled` and hold 1 in E2 / E3
  and the hold check of the engine. `needs_leader_future` only says whether the leader's future is an input.
* Roots: `artifacts/e1_ctrl` (`e1_ctrl.yaml`: recorded signals, target a, 5 seeds; windows and sampled rows identical to
  `artifacts/e1`, so the tables are row-aligned with those of `diffusion_cf`), `artifacts/e2_ctrl` (`e2_ctrl.yaml`:
  target a, highd_val (calibration set) + highd + ngsim + openacc; the events, rollout seeds and streams of `artifacts/e2`),
  E3 `raw` variant (model `checkpoints` override, 18.2). Comparison of interest: `diffusion_cf` vs `diffusion_cf_hist` vs
  the one-step networks at equal information. A sequence baseline with access to the leader's future is not added
  (plan R2: a limitation).

### 18.2 R3: the E3 matrix training domain x leader smoothing

* New variants of `e3.yaml`: `raw_w61` (`artifacts/e1` checkpoints, leader re-smoothed with the matched OpenACC window
  61), `raw_a` (`artifacts/e1`, window 11, QuantilePIDL on target a with hold 1: the same-target control, run for
  `idm_white_noise` and `quantile_pidl` only), `pool_w11` / `lodo_openacc_w11` (E6 checkpoints, the profiles file as
  selected, window 11). QuantilePIDL runs on target a with hold 1 in every cell of the matrix (highD x 11: `raw` +
  `raw_a`; highD x 61: `raw_w61`; pooled x 61: `pool`, `lodo_openacc`; pooled x 11: `pool_w11`, `lodo_openacc_w11`).
  Profiles, initial gaps, followers, rollouts, collision policy and the reference of each cell as in 13 / 17.1.
* E3 runner: a model spec may hold `checkpoints` (its own checkpoints directory instead of the variant's directory of
  its target; in the unit spec as the variant's directory always was) and `variants` (the variants it runs in; not in
  the unit spec); a variant may hold `only_models`. The specs of the models of 13.2 are unchanged, so every stored unit
  stays valid. `raw_a`'s other models would repeat the `raw` units bit for bit (same checkpoint, profiles, hold and
  random streams), hence `only_models`.

### 18.3 R10: IDM with AR(1) noise (`idm_ar_noise`)

* `a_t = IDM(x_t) + e_t`, `e_t = rho e_{t-L} + eta_t`, `eta ~ N(0, sigma_eta^2)`; the IDM is the closed-loop calibration
  of the root (the frozen prior of `idm_white_noise`); `L` = the target's own length (1 for `a`, 10 for `mean_1s`; the E1
  runner passes the experiment's target to every constructor with a `target` parameter, which no model of the original
  set has, so their kwargs are unchanged).
* Fit on the training windows: `r_t = y_t - IDM(x_t)` against the previous residual the window shows,
  `r_prev = (v_t - v_{t-L}) / (L DT) - IDM(x_{t-L})` (for `mean_1s` exactly the target L samples earlier; for `a` the
  backward difference of the speed; in a rollout exactly the previous draw's realised residual); `rho` = the uncentred
  lag-L autocorrelation `sum(r r_prev) / sqrt(sum r^2 sum r_prev^2)`, `sigma_eta^2 = (1 - rho^2) mean(r^2)`: the
  stationary noise level equals the white-noise sigma of `idm_white_noise`, the two baselines differ only by the
  temporal correlation. Seed independent (deterministic fit).
* One-step law `N(IDM(x_t) + rho r_prev, sigma_eta^2)`; `aleatoric_var` is this conditional variance (the variance of
  the represented law; the stationary `sigma_eta^2 / (1 - rho^2)` is in the fit log), `epistemic_var` 0; at `rho = 0`
  the law of `idm_white_noise`. Sampler: the AR state is carried per trajectory, started from `r_prev` of the true
  history, advanced once per draw; `per_step` and `per_trajectory` are the same sampler.
* In `e1.yaml` (and every root inheriting its model list; the E5 roots reuse it from `e1h` / `e1h_mean1s`, so those need
  it first), `e2.yaml` and `e3.yaml` (variants `raw`, `raw_w61`, `pool`, `lodo_openacc`, `pool_w11`,
  `lodo_openacc_w11`; its E6 checkpoints come from `run_e1.py --config e6_*.yaml --models idm_ar_noise`).
* Pilot (raw highD, target a): `rho` = 0.9989 per step, `sigma_eta` = 0.050 m/s^2, stationary 1.04 m/s^2 (= the white
  sigma); the conditional rms of the training windows equals `sigma_eta` (0.0497), so the one-step law is calibrated
  in-domain (PICP 0.95, NLL -1.58 against +1.49 for `idm_white_noise`). On NGSIM the backward difference of the rougher
  speed is a poor proxy of the smoothed target residual: PICP 0.62, NLL 5.4.

### 18.4 R13: frozen-mean heteroscedastic ablation (`lstm_hetero_frozen`)

* The mean of `lstm_const_sigma` of the same seed (`mean_from: artifacts/e1/checkpoints/lstm_const_sigma`, seed k from
  `seed<k>/model.pt`, provenance with sha256 in the log; the source's target must be the experiment's) is frozen with its
  normaliser; a variance head of the shape of HeteroMDN's head (Linear(64, 64), SiLU, Linear(64, 1), log sigma clamped
  to [-7, 3] in normalised space) is trained by Gaussian NLL on the frozen LSTM features (the last LSTM state that the
  mean head reads), with HeteroMDN's TrainConfig. Choice: the frozen features, not a second network on the history, so
  that the two models of the comparison share mean and backbone; only constant sigma -> sigma(x) differs.
* `e1_ctrl` holds it with `lstm_const_sigma` and `hetero_mdn` copied from `artifacts/e1`, so the H3.3 table of that root
  can compare all pairs: `scripts/analyze_e1.py --h33-pair lstm_hetero_frozen lstm_const_sigma --h33-pair hetero_mdn
  lstm_const_sigma` (the option is new; without it the table and the report are the H3.3 table of 10.5, unchanged).
* Pilot (seed 0, 3 epochs of the head): highD test NLL -2.895 (const sigma -2.869, HeteroMDN -3.025) at the identical
  mean (RMSE 0.0137); NGSIM 463 (568, 122): sigma(x) widens a little under shift but cannot repair the mean.

### 18.5 R6 control: causal inputs, recorded target (`e1m_causal`)

* Signal `causal`: `s`, `v`, `v_lead` of every event smoothed by the one-sided Savitzky-Golay filter (least-squares
  quadratic over the trailing window evaluated at its last sample; the first samples of an event use the samples
  available), `dv = v - v_lead`, `a` = central differences of the smoothed speed (`a[t-1]`, the regime input, is causal at
  t), positions integrated from the smoothed speed. The gap gets the same filter (the kinematic reconstruction of 12 fits
  an offset over the whole event); the min-gap drop rule acts on the RECORDED gap (data cleaning independent of the
  filter: every window keeps the same events) and the smoothed gap is floored at `min_gap` where the trailing quadratic
  overshoots. `max_abs_c` / `max_gap_rms` do not apply. Signal `matched_causal`: one window per data set from a causal
  matching file; its per-data-set key is the `causal` key with that window.
* `train.target_from: recorded`: the speed-based targets (`mean_1s`, `fwd_diff`) and their `y_prev` are cut from the
  stored speed (`TargetSpeedEvent.v_target`, never written), the inputs are the processed signals; part of every signal
  key (`target_from`), so no root with the processed target is accepted; refused with target `a` or raw signals.
* Matching (`scripts/match_roughness.py --causal`, `artifacts/e1m_causal/matching.json`, `filter: causal`, `part: val`):
  the rule of 15.1 on the VALIDATION fold of every data set only (fold 1 of its split file; the shifted sets are
  evaluated on `split: all_but_val`, every event outside that fold, so the fold that chose a window is never evaluated;
  the test folds never enter). A matched runner refuses a matching file of the other kind.
* Result of the matching: highD 11 (reference, roughness 0.0021 m/s); NGSIM I-80, US-101, the FollowNet I-80 set, Waymo
  and OpenACC all at 101 with `not_reached` (5.8x, 4.5x, 4.6x, 2.5x, 2.1x the reference; centred E5: 1.4-1.9x, OpenACC
  1.06x at 61): the causal filter cannot equalise the roughness within the grid (its endpoint estimate has 2.8x the
  noise gain of the centred one at window 11, 0.58 against 0.21). With these windows no event is dropped; the gap floor
  acts on 0.87 % of the NGSIM I-80 samples (1036 of 7651 events touched), 0.45 % of US-101, <= 0.3 % elsewhere, none
  on highD.
* `e1m_causal.yaml` is a RETRAINING root (the reuse of the `e1h_mean1s` checkpoints is refused: their training key is
  `harmonised`, this one `causal` with recorded targets): `lstm_const_sigma`, `hetero_mdn`, `deep_ensemble`,
  `mc_dropout`, `gp_residual_idm`, 5 seeds, target `mean_1s` from the recorded speed, the frozen IDM prior of
  `e1h_mean1s` (`train.idm_from`).

### 18.6 R14: density sensitivity of the sampled laws

`scripts/density_sensitivity.py` re-evaluates stored DiffusionCF checkpoints on the rows E1 evaluated (10 000 per data
set) with S = 16 / 32 / 64 first-step samples (nested subsets of one 64-sample draw of the `predict_seed` streams) and
KDE bandwidths 0.5 / 1 / 2 x Silverman (`SampleDist.bandwidth(factor)`, `log_prob(y, bandwidth_factor)`; factor 1 is
bit-identical to the tables); NLL, plain and fair sample CRPS, PICP and width of the 90 % interval per cell, next to the
stored 32-sample values, in `<root>/tables/density_sensitivity.csv`. Result (`artifacts/e1`, seed 0, about 3 GPU
minutes): on highD the mean NLL stays within -2.23 .. -2.43 over the nine cells (S 16 / 32 / 64 at 1x: -2.37 / -2.40 /
-2.41; bandwidth 0.5x / 1x / 2x at S = 32: -2.40 / -2.40 / -2.26), the fair CRPS is 0.0108 in every cell (plain 0.0120 ->
0.0111 with S) and the PICP 0.97-0.99; on NGSIM I-80 the mean NLL is set by the bandwidth (about 60 / 15 / 4 nats at
0.5x / 1x / 2x for every S; median 0.5-1.2) while the CRPS (0.283-0.290) and the PICP (0.40-0.45) hardly move: under
shift the KDE NLL of the sampled law is not a robust score, CRPS is.

### 18.7 Tooling and deviations

* `scripts/run_e2.py --dry-run` (`dry_run_e2`): targets, specs, events tables, checkpoints (signal checked) and tables
  done / missing, loading no events and writing nothing; a target whose checkpoint directory is missing is listed with
  its tables as missing and the note that a run skips it.
* `e2_ctrl` keeps `highd_val` next to the requested highd + ngsim + openacc: it is the calibration set of the conformal
  corrections of `scripts/analyze_e2.py`.
* `idm_ar_noise` is in the inherited model lists of every E1 / E2 root: a run of a derived root without its E1
  checkpoint fails for that model only (as for any missing checkpoint); `e1m` / `e1m_mean1s` reuse it from `e1h` /
  `e1h_mean1s` (the E5 rule: nothing trained in the matched roots).

### 18.8 V7-R01: the history-only DiffusionCF on the pooled roots (`e6_pool_ctrl`, `e6_lodo_openacc_ctrl`)

Decision of the second pre-submission review (2026-10-10, V7-R01): the equal-information control of 18.1 exists for the
highD training only, while the positive H3.2 result of 17 belongs to the pooled training; the pooled-domain
history-only cell is therefore trained. `configs/experiment/e6_pool_ctrl.yaml` and `e6_lodo_openacc_ctrl.yaml`
inherit e6_pool.yaml / e6_lodo_openacc.yaml (matched signals, caps, 5 seeds, target a, the frozen e1m IDM prior,
the evaluated sets) and train `diffusion_cf_hist` only; nothing else of the E6 roots is touched. The platoon units
run in the E3 variants `pool` and `lodo_openacc` with a per-variant checkpoint override of the label
(configs/experiment/e3.yaml), so that every other setting of the cell (profiles, leader window 61, initial gaps,
followers, rollouts, collision policy, the reference idm_white_noise of the cell) is the one of 17.1.

Reading fixed before the run: the comparison is diffusion_cf vs diffusion_cf_hist of the same cell, per driver mode,
with the file-cluster interval deciding (18.7). If the history-only model is refuted or inconclusive where the
future-conditioned model is supported, the advantage of the pooled diffusion model is attributed to the information
of the leader future for that training regime as well; if the history-only model is supported too, the advantage is
one of the training domain and the article's statement is corrected accordingly. Both outcomes are reported.

### 18.9 V7-R02: the missing cell of the past-only control, centred inputs with the recorded target (`e1m_rt`)

Decision of the second pre-submission review (2026-10-10, V7-R02): the control of 18.5 changes the filter, the target
source, the checkpoints and the evaluated folds at once, so it compares pipelines and not the filter. The cell that
separates the two factors is added: `configs/experiment/e1m_rt.yaml` trains the five models of e1m_causal on highD
with the centred inputs of e1m_mean1s (harmonised highD, window 11; the matched shifted signals of
artifacts/e1m/matching.json) and the behavioural target from the recorded speed (train.target_from: recorded), and
evaluates them on the events of e1m_causal (the shifted sets without their validation fold).
`configs/experiment/e1m_mean1s_abv.yaml` evaluates the e1h_mean1s checkpoints (centred inputs, target from the
processed speed) on the same windows, nothing trained. The three cells share the fold-exclusion rule; the two centred cells share their windows exactly, while the causal processing keeps 0.4-7 % more events (no gap-reconstruction constraint), so comparisons with it use its own sample or the windows the three share (scripts/common_support_cells.py, third review). The cells are: (centred,
processed target) = e1m_mean1s_abv, (centred, recorded target) = e1m_rt, (causal, recorded target) = e1m_causal.

Reading fixed before the run: e1m_rt against e1m_causal isolates the input filter at a fixed target and fixed
events; e1m_rt against e1m_mean1s_abv isolates the target at fixed inputs and fixed events. The outcome is reported
as two differences of the one-step NLL and PICP on NGSIM I-80 (and the other shifted sets) per model, without a
causal statement about any factor that is not varied (the window selection rule and the training events differ
between the causal and the centred cells and are named as such). The lag of the causal filter is not measured by
any of these cells and is not claimed.
