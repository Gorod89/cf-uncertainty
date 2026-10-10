Target: `a`, one-step acceleration a[t]. Signals: `raw`.

### NLL, nats (mean over seeds)

| model | highd | highd_val | ngsim | ngsim_us101 | ngsim_fn | waymo | openacc |
|---|---|---|---|---|---|---|---|
| lstm_const_sigma | -2.78 | -2.75 | 467 | 329 | 252 | 27.7 | 279 |
| hetero_mdn | -2.93 | -2.91 | 82.5 | 87.7 | 73.9 | 6.39 | 6.76 |
| diffusion_cf_hist | -2.12 | -2.08 | 11.5 | 11.3 | 10.2 | -0.062 | -0.0117 |
| lstm_hetero_frozen | -2.84 | -2.83 | 103 | 108 | 97.6 | 10.1 | 11.2 |

### PICP of the 90 % interval (mean over seeds)

| model | highd | highd_val | ngsim | ngsim_us101 | ngsim_fn | waymo | openacc |
|---|---|---|---|---|---|---|---|
| lstm_const_sigma | 0.925 | 0.923 | 0.0813 | 0.0699 | 0.0728 | 0.306 | 0.465 |
| hetero_mdn | 0.908 | 0.909 | 0.187 | 0.137 | 0.143 | 0.502 | 0.644 |
| diffusion_cf_hist | 0.976 | 0.972 | 0.46 | 0.443 | 0.447 | 0.774 | 0.831 |
| lstm_hetero_frozen | 0.913 | 0.914 | 0.154 | 0.113 | 0.112 | 0.436 | 0.58 |

### CRPS, m/s^2 (mean over seeds)

| model | highd | highd_val | ngsim | ngsim_us101 | ngsim_fn | waymo | openacc |
|---|---|---|---|---|---|---|---|
| lstm_const_sigma | 0.00795 | 0.00805 | 0.308 | 0.282 | 0.257 | 0.0681 | 0.0983 |
| hetero_mdn | 0.00769 | 0.0078 | 0.296 | 0.252 | 0.236 | 0.0679 | 0.21 |
| diffusion_cf_hist | 0.0192 | 0.0204 | 0.296 | 0.267 | 0.249 | 0.0913 | 0.126 |
| lstm_hetero_frozen | 0.00783 | 0.00793 | 0.296 | 0.274 | 0.25 | 0.0648 | 0.0847 |

### RMSE of the predictive mean, m/s^2 (mean over seeds)

| model | highd | highd_val | ngsim | ngsim_us101 | ngsim_fn | waymo | openacc |
|---|---|---|---|---|---|---|---|
| lstm_const_sigma | 0.015 | 0.0154 | 0.471 | 0.397 | 0.348 | 0.121 | 0.356 |
| hetero_mdn | 0.0274 | 0.0247 | 0.476 | 0.374 | 0.333 | 0.148 | 0.978 |
| diffusion_cf_hist | 0.0656 | 0.0707 | 0.565 | 0.474 | 0.43 | 0.203 | 0.428 |
| lstm_hetero_frozen | 0.015 | 0.0154 | 0.471 | 0.397 | 0.348 | 0.121 | 0.356 |

Seeds per model: lstm_const_sigma 5, hetero_mdn 5, diffusion_cf_hist 5, lstm_hetero_frozen 5

### H3.1: variance ratios shifted / in-domain

| model | decomposition | data set | pair | R epistemic | R aleatoric | R epistemic, matched (s, dv) | R aleatoric, matched | verdict (CI) | verdict (point) |
|---|---|---|---|---|---|---|---|---|---|
| lstm_const_sigma | seed_ensemble | highd_val | placebo | 1.06 [0.866, 1.28] | 1 [1, 1] | 1.11 [0.994, 1.25] | 1 [1, 1] | refuted | refuted |
| lstm_const_sigma | seed_ensemble | ngsim | shift | 213 [179, 243] | 1 [1, 1] | 306 [290, 323] | 1 [1, 1] | supported | supported |
| lstm_const_sigma | seed_ensemble | ngsim_us101 | shift | 138 [114, 157] | 1 [1, 1] | 154 [145, 161] | 1 [1, 1] | supported | supported |
| lstm_const_sigma | seed_ensemble | ngsim_fn | cross_check | 107 [90, 123] | 1 [1, 1] | 150 [141, 160] | 1 [1, 1] | supported | supported |
| lstm_const_sigma | seed_ensemble | waymo | shift | 26.2 [20.3, 32.9] | 1 [1, 1] | 14.4 [12.4, 16.5] | 1 [1, 1] | supported | supported |
| lstm_const_sigma | seed_ensemble | openacc | shift | 401 [321, 484] | 1 [1, 1] | 695 [581, 787] | 1 [1, 1] | supported | supported |
| hetero_mdn | seed_ensemble | highd_val | placebo | 0.544 [0.217, 1.6] | 0.627 [0.0988, 3.34] | 0.543 [0.262, 1.24] | 0.262 [0.123, 0.831] | inconclusive | refuted |
| hetero_mdn | seed_ensemble | ngsim | shift | 42.1 [23.5, 111] | 2.92 [1.44, 12.3] | 113 [63.3, 245] | 5.82 [2.74, 19.8] | inconclusive | refuted |
| hetero_mdn | seed_ensemble | ngsim_us101 | shift | 28.6 [15.2, 71.8] | 1.37 [0.637, 5.67] | 53.2 [29.9, 120] | 1.37 [0.669, 4.82] | inconclusive | inconclusive |
| hetero_mdn | seed_ensemble | ngsim_fn | cross_check | 24 [13.7, 68.4] | 0.631 [0.318, 3.24] | 50.3 [28.9, 118] | 0.756 [0.37, 2.72] | inconclusive | inconclusive |
| hetero_mdn | seed_ensemble | waymo | shift | 14 [5.8, 41.6] | 8.5 [1.58, 45] | 37.7 [8.28, 112] | 44.6 [3.24, 206] | inconclusive | refuted |
| hetero_mdn | seed_ensemble | openacc | shift | 564 [316, 1.58e+03] | 702 [353, 3.58e+03] | 1.32e+03 [644, 2.92e+03] | 2.1e+03 [973, 7.02e+03] | refuted | refuted |
| diffusion_cf_hist | seed_ensemble | highd_val | placebo | 1.09 [0.798, 1.51] | 1.13 [0.976, 1.32] | 1.1 [0.875, 1.4] | 1.1 [0.965, 1.24] | inconclusive | refuted |
| diffusion_cf_hist | seed_ensemble | ngsim | shift | 94.1 [77.6, 118] | 8.67 [7.76, 9.76] | 135 [103, 176] | 10.5 [8.89, 12.3] | refuted | refuted |
| diffusion_cf_hist | seed_ensemble | ngsim_us101 | shift | 62 [50, 76] | 7.05 [6.32, 7.91] | 88 [68.8, 111] | 7.92 [6.92, 9.14] | refuted | refuted |
| diffusion_cf_hist | seed_ensemble | ngsim_fn | cross_check | 51.4 [42.3, 65.2] | 6.21 [5.59, 6.91] | 86 [66.9, 112] | 7.52 [6.53, 8.64] | refuted | refuted |
| diffusion_cf_hist | seed_ensemble | waymo | shift | 13.3 [10.6, 16.9] | 3.43 [3.05, 3.85] | 11.9 [8.29, 17.5] | 3 [2.44, 3.73] | refuted | refuted |
| diffusion_cf_hist | seed_ensemble | openacc | shift | 36.1 [28.8, 46.2] | 6.31 [5.56, 7.18] | 75.5 [60.6, 94.7] | 10.6 [9.31, 11.9] | refuted | refuted |
| lstm_hetero_frozen | seed_ensemble | highd_val | placebo | 1.06 [0.866, 1.28] | 1.33 [0.821, 2.23] | 1.11 [0.994, 1.25] | 1.23 [0.745, 2.15] | refuted | refuted |
| lstm_hetero_frozen | seed_ensemble | ngsim | shift | 213 [179, 243] | 593 [449, 737] | 306 [290, 323] | 803 [588, 1.06e+03] | refuted | refuted |
| lstm_hetero_frozen | seed_ensemble | ngsim_us101 | shift | 138 [114, 157] | 180 [134, 225] | 154 [145, 161] | 100 [70.6, 137] | refuted | refuted |
| lstm_hetero_frozen | seed_ensemble | ngsim_fn | cross_check | 107 [90, 123] | 26.1 [14.6, 42.9] | 150 [141, 160] | 13.2 [5.5, 30.3] | refuted | refuted |
| lstm_hetero_frozen | seed_ensemble | waymo | shift | 26.2 [20.3, 32.9] | 34.5 [5.25, 92.3] | 14.4 [12.4, 16.5] | 2.67 [1.74, 3.97] | refuted | refuted |
| lstm_hetero_frozen | seed_ensemble | openacc | shift | 401 [321, 484] | 1.47e+03 [1.1e+03, 1.88e+03] | 695 [581, 787] | 2.33e+03 [1.72e+03, 3e+03] | refuted | refuted |

### H3.1: summary per model

| model | decomposition | pairs | supported | refuted | inconclusive | overall (CI) | overall (point) | placebo | cross-check |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf_hist | seed_ensemble | 4 | 0 | 4 | 0 | refuted | refuted | highd_val:inconclusive | ngsim_fn:refuted |
| hetero_mdn | seed_ensemble | 4 | 0 | 1 | 3 | refuted | refuted | highd_val:inconclusive | ngsim_fn:inconclusive |
| lstm_const_sigma | seed_ensemble | 4 | 4 | 0 | 0 | supported | supported | highd_val:refuted | ngsim_fn:supported |
| lstm_hetero_frozen | seed_ensemble | 4 | 0 | 4 | 0 | refuted | refuted | highd_val:refuted | ngsim_fn:refuted |

### H3.1e: epistemic part only (design.md, section 12)

Concentration: epistemic variance inside / outside the lowest 10 % of the training mass (rare_mass) on the in-domain test set, lower bound > 1. Growth: lower bound of R_epi >= 3 on at least two counted pairs; refuted when an upper bound is < 1.5 (or the concentration interval lies below 1). Main verdict: decomposition `model` (own epistemic output). Source: h31e_verdicts.csv.

| model | decomposition | concentration (rare_mass) | concentration verdict | pairs | supported | refuted | inconclusive | overall (CI) | overall (point) | placebo | cross-check |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lstm_const_sigma | seed_ensemble | 5.02 [3.48, 7.7] | supported | 4 | 4 | 0 | 0 | supported | supported | highd_val:refuted | ngsim_fn:supported |
| hetero_mdn | seed_ensemble | 26.9 [5.49, 76.5] | supported | 4 | 4 | 0 | 0 | supported | supported | highd_val:inconclusive | ngsim_fn:supported |
| diffusion_cf_hist | seed_ensemble | 8.5 [5.89, 12] | supported | 4 | 4 | 0 | 0 | supported | supported | highd_val:inconclusive | ngsim_fn:supported |
| lstm_hetero_frozen | seed_ensemble | 5.02 [3.48, 7.7] | supported | 4 | 4 | 0 | 0 | supported | supported | highd_val:refuted | ngsim_fn:supported |

| model | decomposition | data set | pair | R epistemic | growth (CI) | growth (point) |
|---|---|---|---|---|---|---|
| lstm_const_sigma | seed_ensemble | highd_val | placebo | 1.06 [0.866, 1.28] | refuted | refuted |
| lstm_const_sigma | seed_ensemble | ngsim | shift | 213 [179, 243] | supported | supported |
| lstm_const_sigma | seed_ensemble | ngsim_us101 | shift | 138 [114, 157] | supported | supported |
| lstm_const_sigma | seed_ensemble | ngsim_fn | cross_check | 107 [90, 123] | supported | supported |
| lstm_const_sigma | seed_ensemble | waymo | shift | 26.2 [20.3, 32.9] | supported | supported |
| lstm_const_sigma | seed_ensemble | openacc | shift | 401 [321, 484] | supported | supported |
| hetero_mdn | seed_ensemble | highd_val | placebo | 0.544 [0.217, 1.6] | inconclusive | refuted |
| hetero_mdn | seed_ensemble | ngsim | shift | 42.1 [23.5, 111] | supported | supported |
| hetero_mdn | seed_ensemble | ngsim_us101 | shift | 28.6 [15.2, 71.8] | supported | supported |
| hetero_mdn | seed_ensemble | ngsim_fn | cross_check | 24 [13.7, 68.4] | supported | supported |
| hetero_mdn | seed_ensemble | waymo | shift | 14 [5.8, 41.6] | supported | supported |
| hetero_mdn | seed_ensemble | openacc | shift | 564 [316, 1.58e+03] | supported | supported |
| diffusion_cf_hist | seed_ensemble | highd_val | placebo | 1.09 [0.798, 1.51] | inconclusive | refuted |
| diffusion_cf_hist | seed_ensemble | ngsim | shift | 94.1 [77.6, 118] | supported | supported |
| diffusion_cf_hist | seed_ensemble | ngsim_us101 | shift | 62 [50, 76] | supported | supported |
| diffusion_cf_hist | seed_ensemble | ngsim_fn | cross_check | 51.4 [42.3, 65.2] | supported | supported |
| diffusion_cf_hist | seed_ensemble | waymo | shift | 13.3 [10.6, 16.9] | supported | supported |
| diffusion_cf_hist | seed_ensemble | openacc | shift | 36.1 [28.8, 46.2] | supported | supported |
| lstm_hetero_frozen | seed_ensemble | highd_val | placebo | 1.06 [0.866, 1.28] | refuted | refuted |
| lstm_hetero_frozen | seed_ensemble | ngsim | shift | 213 [179, 243] | supported | supported |
| lstm_hetero_frozen | seed_ensemble | ngsim_us101 | shift | 138 [114, 157] | supported | supported |
| lstm_hetero_frozen | seed_ensemble | ngsim_fn | cross_check | 107 [90, 123] | supported | supported |
| lstm_hetero_frozen | seed_ensemble | waymo | shift | 26.2 [20.3, 32.9] | supported | supported |
| lstm_hetero_frozen | seed_ensemble | openacc | shift | 401 [321, 484] | supported | supported |

### Concentration of the epistemic variance (inside the region / outside)

| model | decomposition | data set | freq < 1 %: share of windows / of epistemic | freq < 1 %: ratio | lowest 10 % mass: share of windows / of epistemic | lowest 10 % mass: ratio | s < 15 m: share of windows / of epistemic | s < 15 m: ratio | |dv| > 3 m/s: share of windows / of epistemic | |dv| > 3 m/s: ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| lstm_const_sigma | seed_ensemble | highd | 0.34 / 0.58 | 2.62 [2.08, 3.5] | 0.1 / 0.36 | 5.02 [3.48, 7.7] | 0.14 / 0.18 | 1.37 [1.06, 1.68] | 0.016 / 0.13 | 9.2 [5.99, 13.9] |
| lstm_const_sigma | seed_ensemble | highd_val | 0.35 / 0.52 | 2.09 [1.71, 2.48] | 0.1 / 0.3 | 3.86 [3.11, 4.75] | 0.15 / 0.21 | 1.47 [1.26, 1.73] | 0.016 / 0.14 | 9.41 [6.36, 13] |
| lstm_const_sigma | seed_ensemble | ngsim | 0.6 / 0.67 | 1.4 [1.37, 1.42] | 0.36 / 0.44 | 1.39 [1.36, 1.43] | 0.74 / 0.67 | 0.718 [0.696, 0.743] | 0.035 / 0.099 | 3.06 [2.9, 3.24] |
| lstm_const_sigma | seed_ensemble | ngsim_us101 | 0.52 / 0.61 | 1.46 [1.43, 1.49] | 0.26 / 0.37 | 1.64 [1.6, 1.69] | 0.55 / 0.56 | 1.04 [1.01, 1.06] | 0.04 / 0.09 | 2.37 [2.2, 2.57] |
| lstm_const_sigma | seed_ensemble | ngsim_fn | 0.52 / 0.58 | 1.29 [1.26, 1.33] | 0.27 / 0.33 | 1.34 [1.29, 1.4] | 0.62 / 0.58 | 0.826 [0.792, 0.862] | 0.033 / 0.065 | 1.99 [1.75, 2.31] |
| lstm_const_sigma | seed_ensemble | waymo | 0.56 / 0.77 | 2.65 [2.06, 3.4] | 0.33 / 0.62 | 3.29 [2.45, 4.43] | 0.53 / 0.53 | 0.991 [0.674, 1.47] | 0.064 / 0.33 | 7.1 [3.93, 11.4] |
| lstm_const_sigma | seed_ensemble | openacc | 0.26 / 0.69 | 6.2 [5.16, 7.5] | 0.11 / 0.33 | 3.86 [3.15, 4.76] | 0.19 / 0.013 | 0.0577 [0.0432, 0.0746] | 0.03 / 0.11 | 4.02 [3.3, 4.8] |
| hetero_mdn | seed_ensemble | highd | 0.34 / 0.92 | 23.1 [7.37, 43.9] | 0.1 / 0.75 | 26.9 [5.49, 76.5] | 0.14 / 0.04 | 0.264 [0.136, 0.782] | 0.016 / 0.33 | 30 [5.94, 54.6] |
| hetero_mdn | seed_ensemble | highd_val | 0.35 / 0.75 | 5.72 [2.46, 12.5] | 0.1 / 0.59 | 12.9 [3.45, 30.9] | 0.15 / 0.095 | 0.594 [0.327, 1.38] | 0.016 / 0.49 | 56.4 [6.86, 151] |
| hetero_mdn | seed_ensemble | ngsim | 0.6 / 0.66 | 1.29 [1.26, 1.33] | 0.36 / 0.43 | 1.31 [1.28, 1.36] | 0.74 / 0.63 | 0.589 [0.565, 0.612] | 0.035 / 0.11 | 3.39 [3.12, 3.73] |
| hetero_mdn | seed_ensemble | ngsim_us101 | 0.52 / 0.6 | 1.41 [1.38, 1.43] | 0.26 / 0.36 | 1.58 [1.55, 1.62] | 0.55 / 0.55 | 0.997 [0.97, 1.02] | 0.04 / 0.092 | 2.4 [2.28, 2.53] |
| hetero_mdn | seed_ensemble | ngsim_fn | 0.52 / 0.57 | 1.21 [1.16, 1.27] | 0.27 / 0.32 | 1.28 [1.21, 1.39] | 0.62 / 0.59 | 0.855 [0.798, 0.901] | 0.033 / 0.07 | 2.17 [1.69, 2.95] |
| hetero_mdn | seed_ensemble | waymo | 0.56 / 0.72 | 2 [1.12, 3.85] | 0.33 / 0.54 | 2.35 [1.59, 4.28] | 0.53 / 0.32 | 0.425 [0.222, 1.2] | 0.064 / 0.23 | 4.46 [2.47, 7.27] |
| hetero_mdn | seed_ensemble | openacc | 0.26 / 0.56 | 3.64 [3.27, 4.11] | 0.11 / 0.25 | 2.66 [2.31, 3.07] | 0.19 / 0.0053 | 0.0225 [0.016, 0.0296] | 0.03 / 0.096 | 3.41 [3.01, 3.88] |
| diffusion_cf_hist | seed_ensemble | highd | 0.36 / 0.73 | 4.79 [3.42, 6.52] | 0.11 / 0.5 | 8.5 [5.89, 12] | 0.13 / 0.2 | 1.75 [1.04, 2.68] | 0.02 / 0.29 | 20.1 [11.3, 31.6] |
| diffusion_cf_hist | seed_ensemble | highd_val | 0.34 / 0.7 | 4.51 [3.13, 6.57] | 0.1 / 0.46 | 7.51 [4.34, 12.2] | 0.14 / 0.21 | 1.65 [1.08, 2.53] | 0.019 / 0.3 | 22.6 [9.2, 43.6] |
| diffusion_cf_hist | seed_ensemble | ngsim | 0.6 / 0.7 | 1.51 [1.32, 1.74] | 0.37 / 0.48 | 1.58 [1.38, 1.8] | 0.76 / 0.67 | 0.654 [0.574, 0.755] | 0.034 / 0.091 | 2.83 [2.22, 3.53] |
| diffusion_cf_hist | seed_ensemble | ngsim_us101 | 0.52 / 0.62 | 1.53 [1.35, 1.73] | 0.27 / 0.37 | 1.64 [1.43, 1.86] | 0.59 / 0.61 | 1.11 [0.971, 1.26] | 0.036 / 0.074 | 2.12 [1.63, 2.62] |
| diffusion_cf_hist | seed_ensemble | ngsim_fn | 0.52 / 0.59 | 1.33 [1.18, 1.5] | 0.27 / 0.33 | 1.36 [1.21, 1.53] | 0.64 / 0.61 | 0.865 [0.765, 0.971] | 0.033 / 0.067 | 2.12 [1.65, 2.68] |
| diffusion_cf_hist | seed_ensemble | waymo | 0.55 / 0.79 | 2.98 [2.39, 3.79] | 0.33 / 0.64 | 3.73 [2.97, 4.71] | 0.53 / 0.57 | 1.15 [0.898, 1.49] | 0.063 / 0.27 | 5.51 [4.08, 7.28] |
| diffusion_cf_hist | seed_ensemble | openacc | 0.26 / 0.56 | 3.53 [3.03, 4.19] | 0.11 / 0.28 | 2.98 [2.48, 3.64] | 0.18 / 0.05 | 0.237 [0.149, 0.355] | 0.031 / 0.12 | 4.4 [3.23, 5.88] |
| lstm_hetero_frozen | seed_ensemble | highd | 0.34 / 0.58 | 2.62 [2.08, 3.5] | 0.1 / 0.36 | 5.02 [3.48, 7.7] | 0.14 / 0.18 | 1.37 [1.06, 1.68] | 0.016 / 0.13 | 9.2 [5.99, 13.9] |
| lstm_hetero_frozen | seed_ensemble | highd_val | 0.35 / 0.52 | 2.09 [1.71, 2.48] | 0.1 / 0.3 | 3.86 [3.11, 4.75] | 0.15 / 0.21 | 1.47 [1.26, 1.73] | 0.016 / 0.14 | 9.41 [6.36, 13] |
| lstm_hetero_frozen | seed_ensemble | ngsim | 0.6 / 0.67 | 1.4 [1.37, 1.42] | 0.36 / 0.44 | 1.39 [1.36, 1.43] | 0.74 / 0.67 | 0.718 [0.696, 0.743] | 0.035 / 0.099 | 3.06 [2.9, 3.24] |
| lstm_hetero_frozen | seed_ensemble | ngsim_us101 | 0.52 / 0.61 | 1.46 [1.43, 1.49] | 0.26 / 0.37 | 1.64 [1.6, 1.69] | 0.55 / 0.56 | 1.04 [1.01, 1.06] | 0.04 / 0.09 | 2.37 [2.2, 2.57] |
| lstm_hetero_frozen | seed_ensemble | ngsim_fn | 0.52 / 0.58 | 1.29 [1.26, 1.33] | 0.27 / 0.33 | 1.34 [1.29, 1.4] | 0.62 / 0.58 | 0.826 [0.792, 0.862] | 0.033 / 0.065 | 1.99 [1.75, 2.31] |
| lstm_hetero_frozen | seed_ensemble | waymo | 0.56 / 0.77 | 2.65 [2.06, 3.4] | 0.33 / 0.62 | 3.29 [2.45, 4.43] | 0.53 / 0.53 | 0.991 [0.674, 1.47] | 0.064 / 0.33 | 7.1 [3.93, 11.4] |
| lstm_hetero_frozen | seed_ensemble | openacc | 0.26 / 0.69 | 6.2 [5.16, 7.5] | 0.11 / 0.33 | 3.86 [3.15, 4.76] | 0.19 / 0.013 | 0.0577 [0.0432, 0.0746] | 0.03 / 0.11 | 4.02 [3.3, 4.8] |

### H3.3: NLL of HeteroMDN (a) against LSTMConstSigma (b)

| data set | regime | windows | NLL a | NLL b | a - b | p (Holm) | intervals separate |
|---|---|---|---|---|---|---|---|
| highd | all | 152683 | -2.84 [-2.87, -2.8] | -2.78 [-2.81, -2.75] | -0.0595 [-0.0789, -0.0415] | 0.0049 | no |
| highd | congested | 3716 | -2.44 [-2.53, -2.35] | -2.27 [-2.42, -2.1] | -0.174 [-0.256, -0.101] | 0.0049 | no |
| highd | braking | 8686 | -2.33 [-2.44, -2.16] | -1.97 [-2.19, -1.66] | -0.363 [-0.62, -0.201] | 0.0049 | no |
| highd | approaching | 32241 | -2.82 [-2.89, -2.72] | -2.8 [-2.85, -2.72] | -0.029 [-0.0524, 0.00643] | 0.41 | no |
| highd | free | 34757 | -2.98 [-3, -2.94] | -2.92 [-2.94, -2.9] | -0.056 [-0.065, -0.0458] | 0.0049 | yes |
| highd | steady | 37510 | -2.91 [-2.96, -2.83] | -2.88 [-2.91, -2.83] | -0.0308 [-0.051, 0.00411] | 0.41 | no |
| highd | other | 35773 | -2.82 [-2.85, -2.8] | -2.79 [-2.81, -2.76] | -0.0348 [-0.0433, -0.0263] | 0.0049 | no |
| highd_val | all | 152683 | -2.83 [-2.86, -2.79] | -2.75 [-2.79, -2.71] | -0.0756 [-0.0901, -0.0628] | 0.0049 | no |
| highd_val | congested | 5021 | -2.49 [-2.56, -2.42] | -2.36 [-2.48, -2.22] | -0.136 [-0.209, -0.0735] | 0.0049 | no |
| highd_val | braking | 9573 | -2.28 [-2.41, -2.07] | -1.89 [-2.11, -1.6] | -0.393 [-0.569, -0.261] | 0.0049 | no |
| highd_val | approaching | 31482 | -2.85 [-2.89, -2.8] | -2.8 [-2.83, -2.75] | -0.0544 [-0.0766, -0.0355] | 0.0049 | no |
| highd_val | free | 33127 | -2.99 [-3.01, -2.97] | -2.93 [-2.95, -2.91] | -0.0635 [-0.0737, -0.0539] | 0.0049 | yes |
| highd_val | steady | 37016 | -2.9 [-2.95, -2.82] | -2.85 [-2.9, -2.76] | -0.0516 [-0.0652, -0.0401] | 0.0049 | no |
| highd_val | other | 36464 | -2.79 [-2.85, -2.7] | -2.75 [-2.81, -2.67] | -0.0376 [-0.0516, -0.0221] | 0.0049 | no |
| ngsim | all | 438867 | 103 [102, 105] | 467 [459, 475] | -364 [-371, -357] | 0.0049 | yes |
| ngsim | congested | 286453 | 75.6 [74.3, 76.9] | 384 [376, 392] | -308 [-316, -301] | 0.0049 | yes |
| ngsim | braking | 32283 | 139 [135, 143] | 970 [939, 1e+03] | -831 [-863, -801] | 0.0049 | yes |
| ngsim | approaching | 39430 | 178 [174, 183] | 676 [656, 697] | -498 [-517, -479] | 0.0049 | yes |
| ngsim | free | 1019 | 216 [181, 255] | 728 [561, 937] | -512 [-717, -351] | 0.0049 | yes |
| ngsim | steady | 32365 | 160 [156, 164] | 392 [379, 405] | -232 [-242, -221] | 0.0049 | yes |
| ngsim | other | 47317 | 142 [139, 145] | 498 [480, 516] | -356 [-373, -339] | 0.0049 | yes |
| ngsim_us101 | all | 425210 | 108 [107, 109] | 329 [325, 334] | -221 [-225, -217] | 0.0049 | yes |
| ngsim_us101 | congested | 144975 | 72.3 [71.2, 73.3] | 305 [298, 312] | -233 [-239, -226] | 0.0049 | yes |
| ngsim_us101 | braking | 54105 | 108 [106, 110] | 496 [485, 508] | -388 [-399, -377] | 0.0049 | yes |
| ngsim_us101 | approaching | 56689 | 151 [148, 153] | 351 [343, 360] | -201 [-208, -194] | 0.0049 | yes |
| ngsim_us101 | free | 1662 | 98 [88, 109] | 304 [265, 351] | -206 [-247, -173] | 0.0049 | yes |
| ngsim_us101 | steady | 67556 | 137 [135, 139] | 263 [258, 268] | -126 [-130, -122] | 0.0049 | yes |
| ngsim_us101 | other | 100223 | 118 [116, 119] | 308 [302, 314] | -190 [-196, -185] | 0.0049 | yes |
| ngsim_fn | all | 226086 | 97.6 [95.9, 99.4] | 252 [247, 257] | -155 [-158, -151] | 0.0049 | yes |
| ngsim_fn | congested | 108018 | 75.5 [73.7, 77.3] | 220 [214, 226] | -144 [-149, -140] | 0.0049 | yes |
| ngsim_fn | braking | 25122 | 101 [98.5, 104] | 370 [359, 381] | -269 [-278, -259] | 0.0049 | yes |
| ngsim_fn | approaching | 28213 | 134 [131, 138] | 294 [284, 303] | -159 [-166, -152] | 0.0049 | yes |
| ngsim_fn | free | 995 | 158 [130, 191] | 380 [302, 495] | -222 [-328, -153] | 0.0049 | yes |
| ngsim_fn | steady | 27013 | 128 [124, 132] | 235 [227, 243] | -107 [-112, -102] | 0.0049 | yes |
| ngsim_fn | other | 36725 | 108 [105, 111] | 244 [237, 252] | -136 [-142, -131] | 0.0049 | yes |
| waymo | all | 158668 | 10.1 [8.73, 11.5] | 27.7 [24.3, 31.3] | -17.6 [-20.2, -15.2] | 0.0049 | yes |
| waymo | congested | 69600 | 8.22 [6.79, 9.84] | 31.3 [26.6, 36.6] | -23.1 [-27.2, -19.5] | 0.0049 | yes |
| waymo | braking | 10888 | 15.1 [11.8, 18.9] | 52.9 [39.7, 68.4] | -37.8 [-51, -26.9] | 0.0049 | yes |
| waymo | approaching | 19826 | 11.3 [8.76, 14.3] | 20 [15.9, 24.7] | -8.67 [-11, -6.61] | 0.0049 | yes |
| waymo | free | 3988 | 12.2 [4.46, 23.3] | 18.5 [8.95, 31.4] | -6.28 [-13.2, -1.71] | 0.013 | no |
| waymo | steady | 22508 | 11.1 [8.39, 14.2] | 16.6 [12.3, 21.9] | -5.51 [-8.05, -3.48] | 0.0049 | no |
| waymo | other | 31858 | 10.6 [8.5, 13.1] | 24.9 [19.5, 31.4] | -14.2 [-19.3, -10.5] | 0.0049 | yes |
| openacc | all | 485932 | 11.2 [9.93, 12.7] | 279 [232, 333] | -268 [-321, -221] | 0.0049 | yes |
| openacc | congested | 42169 | 10.8 [8.62, 13.4] | 30.1 [25.7, 34.9] | -19.3 [-22.2, -16.6] | 0.0049 | yes |
| openacc | braking | 31019 | 21.8 [18.8, 25.4] | 477 [366, 626] | -455 [-602, -346] | 0.0049 | yes |
| openacc | approaching | 67901 | 19 [16.2, 22.1] | 732 [607, 865] | -713 [-844, -590] | 0.0049 | yes |
| openacc | free | 52059 | 21.4 [13.8, 30.5] | 1.1e+03 [848, 1.39e+03] | -1.08e+03 [-1.37e+03, -831] | 0.0049 | yes |
| openacc | steady | 209586 | 5.1 [4.28, 6] | 6.82 [5.61, 8.19] | -1.71 [-2.31, -1.23] | 0.0049 | no |
| openacc | other | 83198 | 10.2 [9.42, 11.1] | 134 [106, 166] | -124 [-156, -95.4] | 0.0049 | yes |

### H3.3: variance relative to steady following, hetero_mdn (* = Holm-adjusted p < 0.05)

| data set | quantity | congested | braking | approaching | free | other |
|---|---|---|---|---|---|---|
| highd | predicted variance | 6.13 [5.1, 7.35]* | 187 [40, 391]* | 19 [1.21, 53.5]* | 0.754 [0.723, 0.786]* | 1.15 [1.1, 1.2]* |
| highd | squared residual | 7.01 [5.61, 8.58]* | 63.3 [17.2, 131]* | 5.73 [1.33, 12.4]* | 0.865 [0.797, 0.934]* | 1.29 [1.16, 1.41]* |
| highd_val | predicted variance | 5.47 [4.81, 6.2]* | 130 [8.32, 323]* | 1.39 [1.09, 1.95]* | 0.909 [0.745, 1.18] | 1.37 [1.13, 1.74]* |
| highd_val | squared residual | 5.24 [3.87, 6.75]* | 37.2 [7.31, 81.7]* | 1.34 [0.974, 1.81] | 0.875 [0.628, 1.22] | 1.64 [1.09, 2.46] |
| ngsim | predicted variance | 1.6 [1.32, 1.97]* | 1.09 [0.847, 1.4] | 2.39 [1.86, 3.05]* | 1.03 [0.398, 1.91] | 1.82 [1.4, 2.38]* |
| ngsim | squared residual | 1.42 [1.37, 1.48]* | 2.21 [2.12, 2.31]* | 1.65 [1.58, 1.73]* | 2 [1.38, 2.81]* | 1.58 [1.5, 1.65]* |
| ngsim_us101 | predicted variance | 10 [8.8, 11.3]* | 3.1 [2.39, 3.91]* | 1.63 [1.3, 2.05]* | 7.37 [1.42, 19.9]* | 5.06 [4.12, 6.19]* |
| ngsim_us101 | squared residual | 1.96 [1.91, 2.02]* | 1.91 [1.86, 1.96]* | 1.34 [1.31, 1.38]* | 1.39 [1.12, 1.78]* | 1.37 [1.33, 1.41]* |
| ngsim_fn | predicted variance | 4.24 [3.87, 4.62]* | 1.43 [1.31, 1.58]* | 0.985 [0.921, 1.05] | 1.85 [0.733, 3.74] | 1.94 [1.67, 2.25]* |
| ngsim_fn | squared residual | 1.34 [1.28, 1.4]* | 1.47 [1.42, 1.52]* | 1.21 [1.16, 1.25]* | 1.6 [1.15, 2.38]* | 1.18 [1.13, 1.22]* |
| waymo | predicted variance | 14.4 [13, 16]* | 125 [3.88, 400]* | 301 [23, 690]* | 448 [14.6, 1.23e+03]* | 4.02 [2.81, 5.52]* |
| waymo | squared residual | 3.34 [2.62, 4.28]* | 6.24 [2.72, 13.5]* | 5.05 [1.56, 11.1]* | 10.1 [1.41, 25.2]* | 1.79 [1.34, 2.45]* |
| openacc | predicted variance | 7.57 [6.55, 8.97]* | 7.18e+03 [6.44e+03, 7.94e+03]* | 1.3e+04 [1.18e+04, 1.42e+04]* | 2.39e+04 [2.18e+04, 2.61e+04]* | 3.46e+03 [2.96e+03, 3.99e+03]* |
| openacc | squared residual | 4.25 [3.58, 5.05]* | 421 [352, 506]* | 596 [502, 707]* | 1.21e+03 [1.02e+03, 1.43e+03]* | 205 [169, 249]* |

### Share of windows per regime

| data set | congested | braking | approaching | free | steady | other |
|---|---|---|---|---|---|---|
| highd | 0.024 | 0.057 | 0.21 | 0.23 | 0.25 | 0.23 |
| highd_val | 0.033 | 0.063 | 0.21 | 0.22 | 0.24 | 0.24 |
| ngsim | 0.65 | 0.074 | 0.09 | 0.0023 | 0.074 | 0.11 |
| ngsim_us101 | 0.34 | 0.13 | 0.13 | 0.0039 | 0.16 | 0.24 |
| ngsim_fn | 0.48 | 0.11 | 0.12 | 0.0044 | 0.12 | 0.16 |
| waymo | 0.44 | 0.069 | 0.12 | 0.025 | 0.14 | 0.2 |
| openacc | 0.087 | 0.064 | 0.14 | 0.11 | 0.43 | 0.17 |

### Signal: one-step predictability of the acceleration without any model

| dataset | target | persistence | n_windows | n_events | std_a | rmse_zero | rmse_persistence | share_v_below_8 | share_s_below_15 | share_absdv_above_3 |
|---|---|---|---|---|---|---|---|---|---|---|
| highd | a | y_prev | 152683 | 2503 | 0.352 | 0.353 | 0.0171 | 0.0401 | 0.137 | 0.0165 |
| highd_val | a | y_prev | 152683 | 2503 | 0.366 | 0.368 | 0.0175 | 0.0477 | 0.15 | 0.0165 |
| ngsim | a | y_prev | 438867 | 7651 | 0.943 | 0.943 | 0.179 | 0.756 | 0.742 | 0.0347 |
| ngsim_us101 | a | y_prev | 425210 | 7353 | 0.887 | 0.889 | 0.15 | 0.411 | 0.554 | 0.0403 |
| ngsim_fn | a | y_prev | 226086 | 1937 | 0.812 | 0.812 | 0.135 | 0.613 | 0.623 | 0.0335 |
| waymo | a | y_prev | 158668 | 967 | 0.657 | 0.657 | 0.0521 | 0.528 | 0.528 | 0.0638 |
| openacc | a | y_prev | 485932 | 4715 | 0.454 | 0.454 | 0.0404 | 0.184 | 0.19 | 0.0301 |

### Windows in states that are rare in (s, dv, v)

| dataset | share_rare3d | share_unseen3d | share_outside3d |
|---|---|---|---|
| highd | 0.0993 | 0.00217 | 0.00168 |
| highd_val | 0.106 | 0.00335 | 0.00172 |
| ngsim | 0.43 | 0.0134 | 0.00371 |
| ngsim_us101 | 0.313 | 0.00865 | 0.00348 |
| ngsim_fn | 0.319 | 0.00948 | 0.00335 |
| waymo | 0.413 | 0.0144 | 0.0152 |
| openacc | 0.283 | 0.0936 | 0.00584 |

### Calibration of the predicted variance

| model | dataset | n_seeds | mse | predicted_var | mse_over_var | median_z2 | share_z_above_3 |
|---|---|---|---|---|---|---|---|
| lstm_const_sigma | highd | 5 | 0.000224 | 0.000238 | 0.942 | 0.286 | 0.0102 |
| lstm_const_sigma | highd_val | 5 | 0.000238 | 0.000238 | 1 | 0.288 | 0.0111 |
| lstm_const_sigma | ngsim | 5 | 0.223 | 0.000238 | 935 | 195 | 0.859 |
| lstm_const_sigma | ngsim_us101 | 5 | 0.158 | 0.000238 | 662 | 205 | 0.876 |
| lstm_const_sigma | ngsim_fn | 5 | 0.121 | 0.000238 | 509 | 182 | 0.87 |
| lstm_const_sigma | waymo | 5 | 0.0146 | 0.000238 | 61.4 | 9.31 | 0.506 |
| lstm_const_sigma | openacc | 5 | 0.131 | 0.000238 | 551 | 3.48 | 0.37 |
| hetero_mdn | highd | 5 | 0.000852 | 0.00262 | 0.326 | 0.368 | 0.00865 |
| hetero_mdn | highd_val | 5 | 0.000624 | 0.00164 | 0.38 | 0.36 | 0.00839 |
| hetero_mdn | ngsim | 5 | 0.227 | 0.00765 | 29.7 | 35.7 | 0.697 |
| hetero_mdn | ngsim_us101 | 5 | 0.14 | 0.00358 | 39.2 | 57.2 | 0.767 |
| hetero_mdn | ngsim_fn | 5 | 0.111 | 0.00165 | 67.3 | 49.4 | 0.754 |
| hetero_mdn | waymo | 5 | 0.0223 | 0.0222 | 1 | 2.68 | 0.28 |
| hetero_mdn | openacc | 5 | 1.09 | 1.84 | 0.593 | 1.14 | 0.195 |
| diffusion_cf_hist | highd | 5 | 0.00437 | 0.00905 | 0.482 | 0.123 | 0.00038 |
| diffusion_cf_hist | highd_val | 5 | 0.00507 | 0.0102 | 0.496 | 0.124 | 0.0003 |
| diffusion_cf_hist | ngsim | 5 | 0.32 | 0.0785 | 4.08 | 2.46 | 0.225 |
| diffusion_cf_hist | ngsim_us101 | 5 | 0.226 | 0.0638 | 3.54 | 2.57 | 0.23 |
| diffusion_cf_hist | ngsim_fn | 5 | 0.185 | 0.0562 | 3.29 | 2.55 | 0.233 |
| diffusion_cf_hist | waymo | 5 | 0.0416 | 0.031 | 1.34 | 0.527 | 0.0286 |
| diffusion_cf_hist | openacc | 5 | 0.185 | 0.0571 | 3.23 | 0.285 | 0.0372 |
| lstm_hetero_frozen | highd | 5 | 0.000224 | 0.000288 | 0.779 | 0.354 | 0.00819 |
| lstm_hetero_frozen | highd_val | 5 | 0.000238 | 0.000384 | 0.62 | 0.349 | 0.00797 |
| lstm_hetero_frozen | ngsim | 5 | 0.223 | 0.171 | 1.3 | 53.8 | 0.748 |
| lstm_hetero_frozen | ngsim_us101 | 5 | 0.158 | 0.0518 | 3.04 | 79.1 | 0.806 |
| lstm_hetero_frozen | ngsim_fn | 5 | 0.121 | 0.00752 | 16.1 | 74.6 | 0.804 |
| lstm_hetero_frozen | waymo | 5 | 0.0146 | 0.00996 | 1.47 | 3.95 | 0.353 |
| lstm_hetero_frozen | openacc | 5 | 0.131 | 0.424 | 0.31 | 1.62 | 0.247 |

### Shift ratios: geometric means against means

| model | dataset | quantity | ratio_geometric | geo_lo | geo_hi | ratio_means | means_lo | means_hi |
|---|---|---|---|---|---|---|---|---|
| lstm_const_sigma | highd_val | aleatoric_var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | highd_val | var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | highd_val | sq_err | 1.01 | 0.981 | 1.05 | 1.06 | 0.961 | 1.18 |
| lstm_const_sigma | ngsim | aleatoric_var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | ngsim | var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | ngsim | sq_err | 677 | 655 | 700 | 993 | 930 | 1.06e+03 |
| lstm_const_sigma | ngsim_us101 | aleatoric_var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | ngsim_us101 | var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | ngsim_us101 | sq_err | 697 | 678 | 716 | 703 | 658 | 748 |
| lstm_const_sigma | ngsim_fn | aleatoric_var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | ngsim_fn | var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | ngsim_fn | sq_err | 610 | 592 | 632 | 540 | 506 | 577 |
| lstm_const_sigma | waymo | aleatoric_var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | waymo | var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | waymo | sq_err | 35.8 | 33.4 | 38.5 | 65.2 | 57.2 | 74.3 |
| lstm_const_sigma | openacc | aleatoric_var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | openacc | var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | openacc | sq_err | 17.4 | 16.3 | 18.5 | 584 | 482 | 707 |
| hetero_mdn | highd_val | aleatoric_var | 1.04 | 1.01 | 1.07 | 0.627 | 0.0901 | 2.75 |
| hetero_mdn | highd_val | var | 1.04 | 1.01 | 1.07 | 0.627 | 0.0901 | 2.75 |
| hetero_mdn | highd_val | sq_err | 1.02 | 0.985 | 1.07 | 0.732 | 0.276 | 1.93 |
| hetero_mdn | ngsim | aleatoric_var | 8.78 | 8.54 | 9.04 | 2.92 | 1.42 | 13.6 |
| hetero_mdn | ngsim | var | 8.78 | 8.54 | 9.04 | 2.92 | 1.42 | 13.6 |
| hetero_mdn | ngsim | sq_err | 849 | 822 | 878 | 267 | 145 | 664 |
| hetero_mdn | ngsim_us101 | aleatoric_var | 4.64 | 4.52 | 4.75 | 1.37 | 0.667 | 6.18 |
| hetero_mdn | ngsim_us101 | var | 4.64 | 4.52 | 4.75 | 1.37 | 0.667 | 6.18 |
| hetero_mdn | ngsim_us101 | sq_err | 699 | 677 | 720 | 165 | 90.3 | 413 |
| hetero_mdn | ngsim_fn | aleatoric_var | 4.89 | 4.75 | 5.05 | 0.631 | 0.307 | 2.97 |
| hetero_mdn | ngsim_fn | var | 4.89 | 4.75 | 5.05 | 0.631 | 0.307 | 2.97 |
| hetero_mdn | ngsim_fn | sq_err | 643 | 622 | 665 | 131 | 71.2 | 326 |
| hetero_mdn | waymo | aleatoric_var | 5.6 | 5.19 | 6.03 | 8.5 | 1.74 | 41.3 |
| hetero_mdn | waymo | var | 5.6 | 5.19 | 6.03 | 8.5 | 1.74 | 41.3 |
| hetero_mdn | waymo | sq_err | 46.8 | 43.1 | 50.5 | 26.2 | 13 | 65.6 |
| hetero_mdn | openacc | aleatoric_var | 7.08 | 6.55 | 7.67 | 702 | 340 | 3.28e+03 |
| hetero_mdn | openacc | var | 7.08 | 6.55 | 7.67 | 702 | 340 | 3.28e+03 |
| hetero_mdn | openacc | sq_err | 31.2 | 28.8 | 34.1 | 1.28e+03 | 703 | 3.25e+03 |
| diffusion_cf_hist | highd_val | aleatoric_var | 1.06 | 0.996 | 1.14 | 1.13 | 0.965 | 1.33 |
| diffusion_cf_hist | highd_val | var | 1.06 | 0.996 | 1.14 | 1.13 | 0.965 | 1.33 |
| diffusion_cf_hist | highd_val | sq_err | 1.07 | 0.969 | 1.19 | 1.16 | 0.838 | 1.59 |
| diffusion_cf_hist | ngsim | aleatoric_var | 12.3 | 11.6 | 12.9 | 8.67 | 7.76 | 9.75 |
| diffusion_cf_hist | ngsim | var | 12.3 | 11.6 | 12.9 | 8.67 | 7.76 | 9.75 |
| diffusion_cf_hist | ngsim | sq_err | 254 | 233 | 276 | 73.4 | 60.6 | 90.3 |
| diffusion_cf_hist | ngsim_us101 | aleatoric_var | 11.2 | 10.6 | 11.9 | 7.05 | 6.35 | 7.95 |
| diffusion_cf_hist | ngsim_us101 | var | 11.2 | 10.6 | 11.9 | 7.05 | 6.35 | 7.95 |
| diffusion_cf_hist | ngsim_us101 | sq_err | 247 | 229 | 267 | 51.7 | 42.9 | 64 |
| diffusion_cf_hist | ngsim_fn | aleatoric_var | 10.2 | 9.63 | 10.8 | 6.21 | 5.59 | 7.05 |
| diffusion_cf_hist | ngsim_fn | var | 10.2 | 9.63 | 10.8 | 6.21 | 5.59 | 7.05 |
| diffusion_cf_hist | ngsim_fn | sq_err | 217 | 200 | 237 | 42.4 | 35.4 | 52.3 |
| diffusion_cf_hist | waymo | aleatoric_var | 6.2 | 5.73 | 6.67 | 3.43 | 3.02 | 3.92 |
| diffusion_cf_hist | waymo | var | 6.2 | 5.73 | 6.67 | 3.43 | 3.02 | 3.92 |
| diffusion_cf_hist | waymo | sq_err | 27.2 | 24.4 | 30.1 | 9.53 | 7.57 | 12.3 |
| diffusion_cf_hist | openacc | aleatoric_var | 4.3 | 4 | 4.61 | 6.31 | 5.63 | 7.21 |
| diffusion_cf_hist | openacc | var | 4.3 | 4 | 4.61 | 6.31 | 5.63 | 7.21 |
| diffusion_cf_hist | openacc | sq_err | 11.3 | 10.1 | 12.6 | 42.3 | 34.2 | 54.2 |
| lstm_hetero_frozen | highd_val | aleatoric_var | 1.02 | 1.01 | 1.05 | 1.33 | 0.834 | 2.18 |
| lstm_hetero_frozen | highd_val | var | 1.02 | 1.01 | 1.05 | 1.33 | 0.834 | 2.18 |
| lstm_hetero_frozen | highd_val | sq_err | 1.01 | 0.981 | 1.05 | 1.06 | 0.961 | 1.18 |
| lstm_hetero_frozen | ngsim | aleatoric_var | 4.77 | 4.68 | 4.86 | 593 | 451 | 738 |
| lstm_hetero_frozen | ngsim | var | 4.77 | 4.68 | 4.86 | 593 | 451 | 738 |
| lstm_hetero_frozen | ngsim | sq_err | 677 | 655 | 700 | 993 | 930 | 1.06e+03 |
| lstm_hetero_frozen | ngsim_us101 | aleatoric_var | 3.29 | 3.23 | 3.34 | 180 | 135 | 226 |
| lstm_hetero_frozen | ngsim_us101 | var | 3.29 | 3.23 | 3.34 | 180 | 135 | 226 |
| lstm_hetero_frozen | ngsim_us101 | sq_err | 697 | 678 | 716 | 703 | 658 | 748 |
| lstm_hetero_frozen | ngsim_fn | aleatoric_var | 3.01 | 2.94 | 3.07 | 26.1 | 14.1 | 43.2 |
| lstm_hetero_frozen | ngsim_fn | var | 3.01 | 2.94 | 3.07 | 26.1 | 14.1 | 43.2 |
| lstm_hetero_frozen | ngsim_fn | sq_err | 610 | 592 | 632 | 540 | 506 | 577 |
| lstm_hetero_frozen | waymo | aleatoric_var | 2.89 | 2.76 | 3.03 | 34.5 | 5.26 | 84.5 |
| lstm_hetero_frozen | waymo | var | 2.89 | 2.76 | 3.03 | 34.5 | 5.26 | 84.5 |
| lstm_hetero_frozen | waymo | sq_err | 35.8 | 33.4 | 38.5 | 65.2 | 57.2 | 74.3 |
| lstm_hetero_frozen | openacc | aleatoric_var | 2.94 | 2.8 | 3.09 | 1.47e+03 | 1.11e+03 | 1.92e+03 |
| lstm_hetero_frozen | openacc | var | 2.94 | 2.8 | 3.09 | 1.47e+03 | 1.11e+03 | 1.92e+03 |
| lstm_hetero_frozen | openacc | sq_err | 17.4 | 16.3 | 18.5 | 584 | 482 | 707 |

