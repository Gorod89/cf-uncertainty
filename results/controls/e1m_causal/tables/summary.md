Target: `mean_1s`, mean acceleration over the next 1 s, (v[t+10] - v[t]) / (10 DT). Signals: `matched` (harmonised with one window per data set, matched in roughness to highD, design.md section 15).

### NLL, nats (mean over seeds)

| model | highd | highd_val | ngsim | ngsim_us101 | ngsim_fn | waymo | openacc |
|---|---|---|---|---|---|---|---|
| lstm_const_sigma | -1.62 | -1.6 | 259 | 188 | 165 | 39.4 | 50.5 |
| hetero_mdn | -1.69 | -1.69 | 70 | 73.3 | 66.2 | 13.2 | 5.27 |
| deep_ensemble | -1.73 | -1.73 | 40.4 | 46.2 | 41.7 | 8.67 | 3.47 |
| mc_dropout | -1.7 | -1.69 | 16.4 | 18.4 | 16.6 | 3.84 | 1.27 |
| gp_residual_idm | 0.15 | 0.178 | 5.88 | 3.62 | 3.24 | 2.24 | 0.698 |

### PICP of the 90 % interval (mean over seeds)

| model | highd | highd_val | ngsim | ngsim_us101 | ngsim_fn | waymo | openacc |
|---|---|---|---|---|---|---|---|
| lstm_const_sigma | 0.92 | 0.918 | 0.0946 | 0.0828 | 0.0902 | 0.229 | 0.455 |
| hetero_mdn | 0.911 | 0.911 | 0.167 | 0.131 | 0.137 | 0.338 | 0.586 |
| deep_ensemble | 0.931 | 0.931 | 0.229 | 0.18 | 0.188 | 0.39 | 0.618 |
| mc_dropout | 0.963 | 0.964 | 0.262 | 0.238 | 0.234 | 0.487 | 0.694 |
| gp_residual_idm | 0.93 | 0.924 | 0.575 | 0.572 | 0.59 | 0.726 | 0.858 |

### CRPS, m/s^2 (mean over seeds)

| model | highd | highd_val | ngsim | ngsim_us101 | ngsim_fn | waymo | openacc |
|---|---|---|---|---|---|---|---|
| lstm_const_sigma | 0.0256 | 0.0258 | 0.739 | 0.687 | 0.652 | 0.281 | 0.205 |
| hetero_mdn | 0.0248 | 0.0253 | 0.706 | 0.662 | 0.632 | 0.269 | 0.251 |
| deep_ensemble | 0.0239 | 0.0244 | 0.673 | 0.64 | 0.609 | 0.259 | 0.264 |
| mc_dropout | 0.0251 | 0.0256 | 0.647 | 0.611 | 0.576 | 0.258 | 0.393 |
| gp_residual_idm | 0.147 | 0.151 | 0.544 | 0.462 | 0.439 | 0.337 | 0.224 |

### RMSE of the predictive mean, m/s^2 (mean over seeds)

| model | highd | highd_val | ngsim | ngsim_us101 | ngsim_fn | waymo | openacc |
|---|---|---|---|---|---|---|---|
| lstm_const_sigma | 0.0481 | 0.049 | 1.12 | 0.955 | 0.896 | 0.446 | 0.498 |
| hetero_mdn | 0.0495 | 0.057 | 1.1 | 0.943 | 0.891 | 0.45 | 0.85 |
| deep_ensemble | 0.047 | 0.0502 | 1.09 | 0.938 | 0.884 | 0.445 | 1 |
| mc_dropout | 0.0476 | 0.052 | 1.14 | 0.971 | 0.889 | 0.502 | 1.73 |
| gp_residual_idm | 0.283 | 0.293 | 1.11 | 0.884 | 0.839 | 0.739 | 0.534 |

Seeds per model: lstm_const_sigma 5, hetero_mdn 5, deep_ensemble 5, mc_dropout 5, gp_residual_idm 5

### H3.1: variance ratios shifted / in-domain

| model | decomposition | data set | pair | R epistemic | R aleatoric | R epistemic, matched (s, dv) | R aleatoric, matched | verdict (CI) | verdict (point) |
|---|---|---|---|---|---|---|---|---|---|
| lstm_const_sigma | seed_ensemble | highd_val | placebo | 1.17 [1, 1.38] | 1 [1, 1] | 1.12 [1.03, 1.23] | 1 [1, 1] | refuted | refuted |
| lstm_const_sigma | seed_ensemble | ngsim | shift | 71 [63.6, 78.3] | 1 [1, 1] | 85.2 [77.1, 93.8] | 1 [1, 1] | supported | supported |
| lstm_const_sigma | seed_ensemble | ngsim_us101 | shift | 44.5 [39.2, 49.2] | 1 [1, 1] | 24.8 [23.1, 26.7] | 1 [1, 1] | supported | supported |
| lstm_const_sigma | seed_ensemble | ngsim_fn | cross_check | 32.6 [28.5, 37.5] | 1 [1, 1] | 23.9 [22.3, 25.6] | 1 [1, 1] | supported | supported |
| lstm_const_sigma | seed_ensemble | waymo | shift | 25.8 [19.9, 32.7] | 1 [1, 1] | 8.2 [6.18, 10.5] | 1 [1, 1] | supported | supported |
| lstm_const_sigma | seed_ensemble | openacc | shift | 159 [131, 191] | 1 [1, 1] | 192 [158, 224] | 1 [1, 1] | supported | supported |
| hetero_mdn | seed_ensemble | highd_val | placebo | 1.75 [0.677, 4.01] | 1.3 [0.8, 2.1] | 1.2 [0.969, 1.54] | 1.02 [0.953, 1.09] | inconclusive | inconclusive |
| hetero_mdn | seed_ensemble | ngsim | shift | 66.8 [40.8, 107] | 15.8 [12.2, 19.5] | 112 [99.3, 127] | 24.1 [19.6, 29.4] | refuted | refuted |
| hetero_mdn | seed_ensemble | ngsim_us101 | shift | 44 [27.1, 69] | 6.7 [5.13, 8.22] | 37.3 [33.9, 40.9] | 3.28 [2.89, 3.74] | refuted | refuted |
| hetero_mdn | seed_ensemble | ngsim_fn | cross_check | 30.6 [19.6, 49.2] | 4.24 [3.28, 5.26] | 31.6 [28.6, 34.7] | 2.98 [2.54, 3.67] | refuted | refuted |
| hetero_mdn | seed_ensemble | waymo | shift | 26.1 [14.6, 45.9] | 6.06 [3.92, 9.22] | 20.8 [10.5, 37.2] | 6.59 [2.97, 12] | refuted | refuted |
| hetero_mdn | seed_ensemble | openacc | shift | 381 [242, 617] | 338 [260, 435] | 801 [618, 926] | 600 [496, 675] | refuted | refuted |
| deep_ensemble | model | highd_val | placebo | 1.46 [0.774, 2.75] | 1.15 [0.865, 1.59] | 1.16 [0.954, 1.47] | 1.01 [0.953, 1.06] | inconclusive | refuted |
| deep_ensemble | model | ngsim | shift | 72.7 [49.7, 104] | 18 [15.2, 20.6] | 118 [103, 133] | 26 [21.6, 31.5] | refuted | refuted |
| deep_ensemble | model | ngsim_us101 | shift | 44.7 [30.5, 62.5] | 7.57 [6.43, 8.63] | 42.1 [37.5, 46.9] | 3.68 [3.35, 4.06] | refuted | refuted |
| deep_ensemble | model | ngsim_fn | cross_check | 34.7 [24.5, 49.1] | 5.09 [4.32, 5.88] | 39.5 [34.9, 44.4] | 3.73 [3.23, 4.35] | refuted | refuted |
| deep_ensemble | model | waymo | shift | 27.3 [17, 42.6] | 7.18 [4.34, 12.2] | 27.6 [11.8, 53.9] | 8.59 [2.46, 20.7] | refuted | refuted |
| deep_ensemble | model | openacc | shift | 441 [305, 640] | 390 [324, 467] | 911 [717, 1.05e+03] | 660 [580, 732] | refuted | refuted |
| deep_ensemble | seed_ensemble | highd_val | placebo | 1.66 [0.903, 2.96] | 1.18 [0.859, 1.69] | 1.25 [1, 1.58] | 1.02 [0.958, 1.09] | inconclusive | inconclusive |
| deep_ensemble | seed_ensemble | ngsim | shift | 63.4 [46.5, 83] | 24.1 [19.9, 28.1] | 95.6 [84.9, 107] | 33.7 [29.1, 39.3] | refuted | refuted |
| deep_ensemble | seed_ensemble | ngsim_us101 | shift | 38.7 [28.5, 50.5] | 11.7 [9.74, 13.6] | 30.2 [27.4, 33.2] | 6.89 [6.44, 7.4] | refuted | refuted |
| deep_ensemble | seed_ensemble | ngsim_fn | cross_check | 28.7 [21.7, 37.9] | 8.41 [7.07, 9.82] | 28 [25.1, 31.2] | 6.72 [6.15, 7.42] | refuted | refuted |
| deep_ensemble | seed_ensemble | waymo | shift | 31 [20.4, 45.9] | 9.44 [6.03, 14.7] | 35.3 [10.9, 67.2] | 10.2 [3.28, 23.3] | refuted | refuted |
| deep_ensemble | seed_ensemble | openacc | shift | 703 [521, 949] | 395 [324, 482] | 1.56e+03 [1.32e+03, 1.76e+03] | 682 [594, 760] | refuted | refuted |
| mc_dropout | model | highd_val | placebo | 1.09 [0.956, 1.25] | 1.09 [0.644, 2.08] | 1.03 [0.94, 1.13] | 1.15 [0.691, 1.94] | refuted | refuted |
| mc_dropout | model | ngsim | shift | 14 [12.8, 15.3] | 62.2 [42.7, 99.4] | 12.8 [11.8, 13.9] | 100 [67.3, 156] | refuted | refuted |
| mc_dropout | model | ngsim_us101 | shift | 11.2 [10.2, 12.2] | 50 [33.9, 81.5] | 6.35 [5.95, 6.79] | 25.9 [17.7, 39.6] | refuted | refuted |
| mc_dropout | model | ngsim_fn | cross_check | 9.24 [8.38, 10.2] | 31.8 [21.4, 53.3] | 6.71 [6.25, 7.26] | 29.2 [19.2, 47.6] | refuted | refuted |
| mc_dropout | model | waymo | shift | 7.71 [6.71, 8.79] | 32.3 [20.3, 55.1] | 3.42 [2.88, 4.12] | 18.8 [6.74, 42.3] | refuted | refuted |
| mc_dropout | model | openacc | shift | 44.4 [38.1, 51.5] | 289 [194, 465] | 67.7 [58.9, 76.5] | 601 [406, 905] | refuted | refuted |
| mc_dropout | seed_ensemble | highd_val | placebo | 1.74 [0.845, 3.46] | 1.09 [0.66, 1.98] | 1.19 [1.01, 1.43] | 1.13 [0.717, 1.82] | inconclusive | inconclusive |
| mc_dropout | seed_ensemble | ngsim | shift | 75.5 [54.1, 99.9] | 58.2 [40.8, 89.5] | 96 [82, 112] | 91.1 [63.1, 134] | refuted | refuted |
| mc_dropout | seed_ensemble | ngsim_us101 | shift | 50.4 [35.6, 67.1] | 46.8 [32.5, 73] | 25.9 [23.2, 28.8] | 23.8 [16.8, 34.7] | refuted | refuted |
| mc_dropout | seed_ensemble | ngsim_fn | cross_check | 25.7 [18.4, 35.5] | 30 [20.5, 47.7] | 21 [18.7, 23.6] | 26.8 [18.1, 41.6] | refuted | refuted |
| mc_dropout | seed_ensemble | waymo | shift | 33.4 [18, 55.9] | 30.3 [19.6, 49.7] | 18.9 [7.5, 34.8] | 17.2 [6.45, 37.4] | refuted | refuted |
| mc_dropout | seed_ensemble | openacc | shift | 752 [538, 1.05e+03] | 269 [184, 416] | 1.38e+03 [1.01e+03, 1.62e+03] | 545 [379, 787] | refuted | refuted |
| gp_residual_idm | model | highd_val | placebo | 1.24 [0.813, 1.91] | 1 [1, 1] | 1.08 [0.772, 1.42] | 1 [1, 1] | inconclusive | refuted |
| gp_residual_idm | model | ngsim | shift | 6.46 [4.93, 8.62] | 1 [1, 1] | 7.85 [6.11, 10] | 1 [1, 1] | supported | supported |
| gp_residual_idm | model | ngsim_us101 | shift | 6.29 [4.84, 8.47] | 1 [1, 1] | 3.14 [2.74, 3.55] | 1 [1, 1] | supported | supported |
| gp_residual_idm | model | ngsim_fn | cross_check | 5.33 [4.03, 7.3] | 1 [1, 1] | 5.27 [3.51, 7.25] | 1 [1, 1] | supported | supported |
| gp_residual_idm | model | waymo | shift | 9.48 [6.72, 13.9] | 1 [1, 1] | 3.48 [2.34, 4.86] | 1 [1, 1] | supported | supported |
| gp_residual_idm | model | openacc | shift | 13.4 [10.1, 18.6] | 1 [1, 1] | 26 [20.1, 31.6] | 1 [1, 1] | supported | supported |
| gp_residual_idm | seed_ensemble | highd_val | placebo | 1.09 [0.956, 1.25] | 1 [0.996, 1.01] | 1.03 [0.966, 1.1] | 1 [0.998, 1] | refuted | refuted |
| gp_residual_idm | seed_ensemble | ngsim | shift | 7.68 [6.99, 8.4] | 1.1 [1.09, 1.1] | 2.51 [2.33, 2.72] | 1.03 [1.02, 1.03] | supported | supported |
| gp_residual_idm | seed_ensemble | ngsim_us101 | shift | 5.25 [4.75, 5.78] | 1.09 [1.08, 1.1] | 1.58 [1.5, 1.65] | 1.01 [1.01, 1.01] | supported | supported |
| gp_residual_idm | seed_ensemble | ngsim_fn | cross_check | 5.55 [5.03, 6.12] | 1.08 [1.06, 1.09] | 2.21 [1.95, 2.48] | 1.02 [1.01, 1.02] | supported | supported |
| gp_residual_idm | seed_ensemble | waymo | shift | 6.94 [6.18, 7.86] | 1.15 [1.12, 1.19] | 2.03 [1.69, 2.41] | 1.01 [1.01, 1.01] | supported | supported |
| gp_residual_idm | seed_ensemble | openacc | shift | 8.1 [7.16, 9.18] | 1.22 [1.19, 1.24] | 12.3 [11, 13.5] | 1.19 [1.17, 1.21] | supported | supported |

### H3.1: summary per model

| model | decomposition | pairs | supported | refuted | inconclusive | overall (CI) | overall (point) | placebo | cross-check |
|---|---|---|---|---|---|---|---|---|---|
| deep_ensemble | model | 4 | 0 | 4 | 0 | refuted | refuted | highd_val:inconclusive | ngsim_fn:refuted |
| deep_ensemble | seed_ensemble | 4 | 0 | 4 | 0 | refuted | refuted | highd_val:inconclusive | ngsim_fn:refuted |
| gp_residual_idm | model | 4 | 4 | 0 | 0 | supported | supported | highd_val:inconclusive | ngsim_fn:supported |
| gp_residual_idm | seed_ensemble | 4 | 4 | 0 | 0 | supported | supported | highd_val:refuted | ngsim_fn:supported |
| hetero_mdn | seed_ensemble | 4 | 0 | 4 | 0 | refuted | refuted | highd_val:inconclusive | ngsim_fn:refuted |
| lstm_const_sigma | seed_ensemble | 4 | 4 | 0 | 0 | supported | supported | highd_val:refuted | ngsim_fn:supported |
| mc_dropout | model | 4 | 0 | 4 | 0 | refuted | refuted | highd_val:refuted | ngsim_fn:refuted |
| mc_dropout | seed_ensemble | 4 | 0 | 4 | 0 | refuted | refuted | highd_val:inconclusive | ngsim_fn:refuted |

### H3.1e: epistemic part only (design.md, section 12)

Concentration: epistemic variance inside / outside the lowest 10 % of the training mass (rare_mass) on the in-domain test set, lower bound > 1. Growth: lower bound of R_epi >= 3 on at least two counted pairs; refuted when an upper bound is < 1.5 (or the concentration interval lies below 1). Main verdict: decomposition `model` (own epistemic output). Source: h31e_verdicts.csv.

Main verdict: deep_ensemble supported (point: supported), mc_dropout supported (point: supported), gp_residual_idm supported (point: supported)

| model | decomposition | concentration (rare_mass) | concentration verdict | pairs | supported | refuted | inconclusive | overall (CI) | overall (point) | placebo | cross-check |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lstm_const_sigma | seed_ensemble | 4.31 [3.38, 5.75] | supported | 4 | 4 | 0 | 0 | supported | supported | highd_val:refuted | ngsim_fn:supported |
| hetero_mdn | seed_ensemble | 12 [4.38, 24.8] | supported | 4 | 4 | 0 | 0 | supported | supported | highd_val:inconclusive | ngsim_fn:supported |
| deep_ensemble | model | 10.5 [5.05, 19.2] | supported | 4 | 4 | 0 | 0 | supported | supported | highd_val:inconclusive | ngsim_fn:supported |
| deep_ensemble | seed_ensemble | 9.84 [5.55, 15.9] | supported | 4 | 4 | 0 | 0 | supported | supported | highd_val:inconclusive | ngsim_fn:supported |
| mc_dropout | model | 4.35 [3.52, 5.23] | supported | 4 | 4 | 0 | 0 | supported | supported | highd_val:refuted | ngsim_fn:supported |
| mc_dropout | seed_ensemble | 7.71 [3.97, 14.1] | supported | 4 | 4 | 0 | 0 | supported | supported | highd_val:inconclusive | ngsim_fn:supported |
| gp_residual_idm | model | 96 [70.4, 128] | supported | 4 | 4 | 0 | 0 | supported | supported | highd_val:inconclusive | ngsim_fn:supported |
| gp_residual_idm | seed_ensemble | 9.41 [8.08, 10.8] | supported | 4 | 4 | 0 | 0 | supported | supported | highd_val:refuted | ngsim_fn:supported |

| model | decomposition | data set | pair | R epistemic | growth (CI) | growth (point) |
|---|---|---|---|---|---|---|
| lstm_const_sigma | seed_ensemble | highd_val | placebo | 1.17 [1, 1.38] | refuted | refuted |
| lstm_const_sigma | seed_ensemble | ngsim | shift | 71 [63.6, 78.3] | supported | supported |
| lstm_const_sigma | seed_ensemble | ngsim_us101 | shift | 44.5 [39.2, 49.2] | supported | supported |
| lstm_const_sigma | seed_ensemble | ngsim_fn | cross_check | 32.6 [28.5, 37.5] | supported | supported |
| lstm_const_sigma | seed_ensemble | waymo | shift | 25.8 [19.9, 32.7] | supported | supported |
| lstm_const_sigma | seed_ensemble | openacc | shift | 159 [131, 191] | supported | supported |
| hetero_mdn | seed_ensemble | highd_val | placebo | 1.75 [0.677, 4.01] | inconclusive | inconclusive |
| hetero_mdn | seed_ensemble | ngsim | shift | 66.8 [40.8, 107] | supported | supported |
| hetero_mdn | seed_ensemble | ngsim_us101 | shift | 44 [27.1, 69] | supported | supported |
| hetero_mdn | seed_ensemble | ngsim_fn | cross_check | 30.6 [19.6, 49.2] | supported | supported |
| hetero_mdn | seed_ensemble | waymo | shift | 26.1 [14.6, 45.9] | supported | supported |
| hetero_mdn | seed_ensemble | openacc | shift | 381 [242, 617] | supported | supported |
| deep_ensemble | model | highd_val | placebo | 1.46 [0.774, 2.75] | inconclusive | refuted |
| deep_ensemble | model | ngsim | shift | 72.7 [49.7, 104] | supported | supported |
| deep_ensemble | model | ngsim_us101 | shift | 44.7 [30.5, 62.5] | supported | supported |
| deep_ensemble | model | ngsim_fn | cross_check | 34.7 [24.5, 49.1] | supported | supported |
| deep_ensemble | model | waymo | shift | 27.3 [17, 42.6] | supported | supported |
| deep_ensemble | model | openacc | shift | 441 [305, 640] | supported | supported |
| deep_ensemble | seed_ensemble | highd_val | placebo | 1.66 [0.903, 2.96] | inconclusive | inconclusive |
| deep_ensemble | seed_ensemble | ngsim | shift | 63.4 [46.5, 83] | supported | supported |
| deep_ensemble | seed_ensemble | ngsim_us101 | shift | 38.7 [28.5, 50.5] | supported | supported |
| deep_ensemble | seed_ensemble | ngsim_fn | cross_check | 28.7 [21.7, 37.9] | supported | supported |
| deep_ensemble | seed_ensemble | waymo | shift | 31 [20.4, 45.9] | supported | supported |
| deep_ensemble | seed_ensemble | openacc | shift | 703 [521, 949] | supported | supported |
| mc_dropout | model | highd_val | placebo | 1.09 [0.956, 1.25] | refuted | refuted |
| mc_dropout | model | ngsim | shift | 14 [12.8, 15.3] | supported | supported |
| mc_dropout | model | ngsim_us101 | shift | 11.2 [10.2, 12.2] | supported | supported |
| mc_dropout | model | ngsim_fn | cross_check | 9.24 [8.38, 10.2] | supported | supported |
| mc_dropout | model | waymo | shift | 7.71 [6.71, 8.79] | supported | supported |
| mc_dropout | model | openacc | shift | 44.4 [38.1, 51.5] | supported | supported |
| mc_dropout | seed_ensemble | highd_val | placebo | 1.74 [0.845, 3.46] | inconclusive | inconclusive |
| mc_dropout | seed_ensemble | ngsim | shift | 75.5 [54.1, 99.9] | supported | supported |
| mc_dropout | seed_ensemble | ngsim_us101 | shift | 50.4 [35.6, 67.1] | supported | supported |
| mc_dropout | seed_ensemble | ngsim_fn | cross_check | 25.7 [18.4, 35.5] | supported | supported |
| mc_dropout | seed_ensemble | waymo | shift | 33.4 [18, 55.9] | supported | supported |
| mc_dropout | seed_ensemble | openacc | shift | 752 [538, 1.05e+03] | supported | supported |
| gp_residual_idm | model | highd_val | placebo | 1.24 [0.813, 1.91] | inconclusive | refuted |
| gp_residual_idm | model | ngsim | shift | 6.46 [4.93, 8.62] | supported | supported |
| gp_residual_idm | model | ngsim_us101 | shift | 6.29 [4.84, 8.47] | supported | supported |
| gp_residual_idm | model | ngsim_fn | cross_check | 5.33 [4.03, 7.3] | supported | supported |
| gp_residual_idm | model | waymo | shift | 9.48 [6.72, 13.9] | supported | supported |
| gp_residual_idm | model | openacc | shift | 13.4 [10.1, 18.6] | supported | supported |
| gp_residual_idm | seed_ensemble | highd_val | placebo | 1.09 [0.956, 1.25] | refuted | refuted |
| gp_residual_idm | seed_ensemble | ngsim | shift | 7.68 [6.99, 8.4] | supported | supported |
| gp_residual_idm | seed_ensemble | ngsim_us101 | shift | 5.25 [4.75, 5.78] | supported | supported |
| gp_residual_idm | seed_ensemble | ngsim_fn | cross_check | 5.55 [5.03, 6.12] | supported | supported |
| gp_residual_idm | seed_ensemble | waymo | shift | 6.94 [6.18, 7.86] | supported | supported |
| gp_residual_idm | seed_ensemble | openacc | shift | 8.1 [7.16, 9.18] | supported | supported |

### Concentration of the epistemic variance (inside the region / outside)

| model | decomposition | data set | freq < 1 %: share of windows / of epistemic | freq < 1 %: ratio | lowest 10 % mass: share of windows / of epistemic | lowest 10 % mass: ratio | s < 15 m: share of windows / of epistemic | s < 15 m: ratio | |dv| > 3 m/s: share of windows / of epistemic | |dv| > 3 m/s: ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| lstm_const_sigma | seed_ensemble | highd | 0.33 / 0.54 | 2.32 [2, 2.79] | 0.1 / 0.33 | 4.31 [3.38, 5.75] | 0.14 / 0.21 | 1.73 [1.45, 2] | 0.017 / 0.13 | 8.77 [5.96, 12.3] |
| lstm_const_sigma | seed_ensemble | highd_val | 0.33 / 0.54 | 2.33 [1.9, 2.87] | 0.1 / 0.32 | 4.28 [3.23, 5.76] | 0.15 / 0.22 | 1.62 [1.37, 1.92] | 0.016 / 0.17 | 12 [7.06, 19.3] |
| lstm_const_sigma | seed_ensemble | ngsim | 0.65 / 0.8 | 2.08 [1.98, 2.19] | 0.46 / 0.65 | 2.22 [2.11, 2.34] | 0.74 / 0.62 | 0.561 [0.518, 0.605] | 0.057 / 0.26 | 5.97 [5.48, 6.55] |
| lstm_const_sigma | seed_ensemble | ngsim_us101 | 0.58 / 0.79 | 2.86 [2.71, 3.03] | 0.35 / 0.66 | 3.65 [3.45, 3.89] | 0.55 / 0.58 | 1.13 [1.04, 1.22] | 0.069 / 0.31 | 6.05 [5.5, 6.7] |
| lstm_const_sigma | seed_ensemble | ngsim_fn | 0.58 / 0.76 | 2.29 [2.05, 2.65] | 0.36 / 0.61 | 2.73 [2.41, 3.27] | 0.62 / 0.62 | 1.03 [0.791, 1.25] | 0.059 / 0.23 | 4.74 [3.52, 7.02] |
| lstm_const_sigma | seed_ensemble | waymo | 0.6 / 0.88 | 4.75 [3.45, 6.36] | 0.39 / 0.8 | 6.4 [4.61, 8.54] | 0.56 / 0.52 | 0.878 [0.624, 1.41] | 0.078 / 0.46 | 10.3 [6.14, 15.2] |
| lstm_const_sigma | seed_ensemble | openacc | 0.3 / 0.77 | 7.8 [6.27, 10] | 0.14 / 0.5 | 6.12 [4.89, 7.88] | 0.17 / 0.043 | 0.225 [0.134, 0.346] | 0.048 / 0.31 | 8.98 [7.1, 11.3] |
| hetero_mdn | seed_ensemble | highd | 0.33 / 0.71 | 4.96 [2.43, 9.27] | 0.1 / 0.58 | 12 [4.38, 24.8] | 0.14 / 0.14 | 1 [0.561, 1.75] | 0.017 / 0.34 | 30.2 [11.4, 47.8] |
| hetero_mdn | seed_ensemble | highd_val | 0.33 / 0.77 | 6.64 [1.83, 15.1] | 0.1 / 0.67 | 18 [3.55, 42.2] | 0.15 / 0.092 | 0.575 [0.29, 1.5] | 0.016 / 0.6 | 89.9 [10.6, 223] |
| hetero_mdn | seed_ensemble | ngsim | 0.65 / 0.78 | 1.89 [1.79, 2] | 0.46 / 0.62 | 1.94 [1.85, 2.05] | 0.74 / 0.59 | 0.491 [0.46, 0.524] | 0.057 / 0.26 | 5.92 [5.52, 6.37] |
| hetero_mdn | seed_ensemble | ngsim_us101 | 0.58 / 0.78 | 2.66 [2.52, 2.83] | 0.35 / 0.63 | 3.17 [3, 3.35] | 0.55 / 0.56 | 1.01 [0.946, 1.07] | 0.069 / 0.3 | 5.73 [5.35, 6.15] |
| hetero_mdn | seed_ensemble | ngsim_fn | 0.58 / 0.75 | 2.11 [1.95, 2.26] | 0.36 / 0.58 | 2.44 [2.25, 2.63] | 0.62 / 0.61 | 0.987 [0.877, 1.11] | 0.059 / 0.22 | 4.4 [3.8, 5.06] |
| hetero_mdn | seed_ensemble | waymo | 0.6 / 0.86 | 3.93 [2.63, 6.13] | 0.39 / 0.76 | 5.01 [3.28, 7.7] | 0.56 / 0.49 | 0.766 [0.546, 1.2] | 0.078 / 0.46 | 10.1 [5.99, 15.9] |
| hetero_mdn | seed_ensemble | openacc | 0.3 / 0.71 | 5.67 [4.89, 6.73] | 0.14 / 0.39 | 3.91 [3.3, 4.72] | 0.17 / 0.016 | 0.0831 [0.0598, 0.109] | 0.048 / 0.22 | 5.71 [4.83, 6.69] |
| deep_ensemble | model | highd | 0.33 / 0.71 | 4.82 [2.8, 7.99] | 0.1 / 0.54 | 10.5 [5.05, 19.2] | 0.14 / 0.15 | 1.08 [0.669, 1.69] | 0.017 / 0.3 | 25.2 [11.5, 39.6] |
| deep_ensemble | model | highd_val | 0.33 / 0.72 | 5.24 [2.25, 10.3] | 0.1 / 0.59 | 12.8 [4.33, 27] | 0.15 / 0.12 | 0.749 [0.442, 1.43] | 0.016 / 0.49 | 58.5 [13.1, 131] |
| deep_ensemble | model | ngsim | 0.65 / 0.78 | 1.85 [1.78, 1.92] | 0.46 / 0.62 | 1.92 [1.85, 1.99] | 0.74 / 0.6 | 0.52 [0.493, 0.548] | 0.057 / 0.25 | 5.56 [5.28, 5.87] |
| deep_ensemble | model | ngsim_us101 | 0.58 / 0.75 | 2.19 [2.11, 2.28] | 0.35 / 0.59 | 2.71 [2.61, 2.84] | 0.55 / 0.56 | 1.04 [0.992, 1.1] | 0.069 / 0.28 | 5.16 [4.85, 5.48] |
| deep_ensemble | model | ngsim_fn | 0.58 / 0.73 | 1.89 [1.79, 2.01] | 0.36 / 0.55 | 2.19 [2.07, 2.33] | 0.62 / 0.61 | 0.952 [0.87, 1.04] | 0.059 / 0.2 | 3.93 [3.46, 4.49] |
| deep_ensemble | model | waymo | 0.6 / 0.83 | 3.25 [2.33, 4.82] | 0.39 / 0.73 | 4.13 [2.93, 5.98] | 0.56 / 0.49 | 0.755 [0.478, 1.23] | 0.078 / 0.39 | 7.65 [5.59, 10.3] |
| deep_ensemble | model | openacc | 0.3 / 0.64 | 4.11 [3.64, 4.73] | 0.14 / 0.31 | 2.76 [2.38, 3.27] | 0.17 / 0.013 | 0.069 [0.0522, 0.086] | 0.048 / 0.15 | 3.46 [3.02, 3.96] |
| deep_ensemble | seed_ensemble | highd | 0.33 / 0.7 | 4.67 [3.11, 6.93] | 0.1 / 0.53 | 9.84 [5.55, 15.9] | 0.14 / 0.16 | 1.23 [0.825, 1.76] | 0.017 / 0.33 | 28.8 [13.2, 50.5] |
| deep_ensemble | seed_ensemble | highd_val | 0.33 / 0.75 | 5.86 [2.57, 11.5] | 0.1 / 0.6 | 13.5 [4.5, 28.7] | 0.15 / 0.11 | 0.726 [0.42, 1.43] | 0.016 / 0.5 | 59 [12.3, 137] |
| deep_ensemble | seed_ensemble | ngsim | 0.65 / 0.81 | 2.19 [2.07, 2.31] | 0.46 / 0.66 | 2.28 [2.16, 2.39] | 0.74 / 0.59 | 0.493 [0.46, 0.529] | 0.057 / 0.29 | 6.79 [6.32, 7.34] |
| deep_ensemble | seed_ensemble | ngsim_us101 | 0.58 / 0.81 | 3.06 [2.9, 3.23] | 0.35 / 0.68 | 3.86 [3.65, 4.08] | 0.55 / 0.57 | 1.09 [1.01, 1.17] | 0.069 / 0.34 | 7.01 [6.44, 7.63] |
| deep_ensemble | seed_ensemble | ngsim_fn | 0.58 / 0.77 | 2.46 [2.26, 2.69] | 0.36 / 0.62 | 2.87 [2.64, 3.16] | 0.62 / 0.61 | 0.963 [0.832, 1.1] | 0.059 / 0.24 | 5.14 [4.31, 6.27] |
| deep_ensemble | seed_ensemble | waymo | 0.6 / 0.8 | 2.68 [1.6, 5.41] | 0.39 / 0.71 | 3.8 [2.4, 6.64] | 0.56 / 0.47 | 0.722 [0.45, 1.28] | 0.078 / 0.38 | 7.17 [4.62, 10.9] |
| deep_ensemble | seed_ensemble | openacc | 0.3 / 0.53 | 2.64 [2.34, 3.03] | 0.14 / 0.19 | 1.37 [1.19, 1.6] | 0.17 / 0.0091 | 0.0463 [0.0348, 0.0594] | 0.048 / 0.07 | 1.5 [1.31, 1.72] |
| mc_dropout | model | highd | 0.33 / 0.59 | 2.9 [2.52, 3.3] | 0.1 / 0.33 | 4.35 [3.52, 5.23] | 0.14 / 0.22 | 1.75 [1.44, 2.09] | 0.017 / 0.14 | 9.41 [6.69, 12.4] |
| mc_dropout | model | highd_val | 0.33 / 0.59 | 2.87 [2.54, 3.29] | 0.1 / 0.33 | 4.35 [3.61, 5.27] | 0.15 / 0.26 | 1.99 [1.71, 2.32] | 0.016 / 0.15 | 10.4 [7.36, 14.5] |
| mc_dropout | model | ngsim | 0.65 / 0.8 | 2.12 [2.05, 2.2] | 0.46 / 0.65 | 2.19 [2.12, 2.29] | 0.74 / 0.69 | 0.759 [0.707, 0.809] | 0.057 / 0.22 | 4.56 [4.19, 5.02] |
| mc_dropout | model | ngsim_us101 | 0.58 / 0.78 | 2.68 [2.6, 2.76] | 0.35 / 0.63 | 3.12 [3.02, 3.22] | 0.55 / 0.63 | 1.38 [1.33, 1.44] | 0.069 / 0.25 | 4.42 [4.16, 4.7] |
| mc_dropout | model | ngsim_fn | 0.58 / 0.76 | 2.25 [2.13, 2.4] | 0.36 / 0.59 | 2.5 [2.35, 2.71] | 0.62 / 0.66 | 1.19 [1.04, 1.29] | 0.059 / 0.19 | 3.7 [3.21, 4.53] |
| mc_dropout | model | waymo | 0.6 / 0.84 | 3.59 [3.09, 4.11] | 0.39 / 0.73 | 4.17 [3.6, 4.79] | 0.56 / 0.66 | 1.58 [1.29, 1.98] | 0.078 / 0.28 | 4.56 [3.42, 6.13] |
| mc_dropout | model | openacc | 0.3 / 0.67 | 4.76 [4.11, 5.65] | 0.14 / 0.34 | 3.07 [2.61, 3.7] | 0.17 / 0.028 | 0.143 [0.119, 0.173] | 0.048 / 0.17 | 4.04 [3.41, 4.78] |
| mc_dropout | seed_ensemble | highd | 0.33 / 0.65 | 3.71 [2.35, 5.87] | 0.1 / 0.47 | 7.71 [3.97, 14.1] | 0.14 / 0.17 | 1.34 [0.814, 2.03] | 0.017 / 0.19 | 13.6 [8.6, 18.7] |
| mc_dropout | seed_ensemble | highd_val | 0.33 / 0.74 | 5.69 [2.08, 12] | 0.1 / 0.61 | 13.9 [3.54, 31.8] | 0.15 / 0.12 | 0.775 [0.403, 1.76] | 0.016 / 0.52 | 64.9 [8.76, 163] |
| mc_dropout | seed_ensemble | ngsim | 0.65 / 0.83 | 2.51 [2.25, 2.83] | 0.46 / 0.69 | 2.68 [2.45, 2.94] | 0.74 / 0.54 | 0.413 [0.371, 0.461] | 0.057 / 0.38 | 10.3 [9.28, 11.7] |
| mc_dropout | seed_ensemble | ngsim_us101 | 0.58 / 0.85 | 4.04 [3.54, 4.61] | 0.35 / 0.75 | 5.43 [4.82, 6.19] | 0.55 / 0.54 | 0.934 [0.819, 1.06] | 0.069 / 0.5 | 13.3 [11.5, 15.2] |
| mc_dropout | seed_ensemble | ngsim_fn | 0.58 / 0.81 | 2.97 [2.54, 3.6] | 0.36 / 0.67 | 3.59 [3, 4.46] | 0.62 / 0.58 | 0.84 [0.608, 1.12] | 0.059 / 0.33 | 7.98 [5.59, 11.7] |
| mc_dropout | seed_ensemble | waymo | 0.6 / 0.89 | 5.3 [2.65, 11] | 0.39 / 0.84 | 7.94 [4.02, 15.9] | 0.56 / 0.41 | 0.566 [0.371, 1.14] | 0.078 / 0.61 | 18.6 [7.82, 37] |
| mc_dropout | seed_ensemble | openacc | 0.3 / 0.67 | 4.62 [3.88, 5.64] | 0.14 / 0.33 | 2.97 [2.39, 3.74] | 0.17 / 0.0096 | 0.0492 [0.0313, 0.0744] | 0.048 / 0.16 | 3.8 [3.05, 4.64] |
| gp_residual_idm | model | highd | 0.33 / 0.97 | 77.3 [56.2, 105] | 0.1 / 0.92 | 96 [70.4, 128] | 0.14 / 0.028 | 0.179 [0.118, 0.273] | 0.017 / 0.47 | 51.8 [32.3, 81.6] |
| gp_residual_idm | model | highd_val | 0.33 / 0.98 | 92 [63.7, 124] | 0.1 / 0.91 | 95.5 [60.4, 145] | 0.15 / 0.023 | 0.136 [0.0951, 0.207] | 0.016 / 0.55 | 74.5 [49.6, 107] |
| gp_residual_idm | model | ngsim | 0.65 / 0.99 | 36.5 [34.1, 38.9] | 0.46 / 0.96 | 31.4 [28.9, 34.3] | 0.74 / 0.68 | 0.719 [0.659, 0.791] | 0.057 / 0.65 | 31 [29.6, 32.4] |
| gp_residual_idm | model | ngsim_us101 | 0.58 / 0.99 | 97.4 [92.2, 103] | 0.35 / 0.98 | 81.6 [76.9, 86.6] | 0.55 / 0.59 | 1.15 [1.06, 1.25] | 0.069 / 0.76 | 43.5 [41, 45.8] |
| gp_residual_idm | model | ngsim_fn | 0.58 / 0.98 | 40.2 [32.5, 50.5] | 0.36 / 0.96 | 38.5 [32, 47.5] | 0.62 / 0.55 | 0.76 [0.634, 0.922] | 0.059 / 0.7 | 37.3 [32.7, 42.5] |
| gp_residual_idm | model | waymo | 0.6 / 0.99 | 72.1 [54.7, 92.7] | 0.39 / 0.98 | 65.6 [50, 84.1] | 0.56 / 0.39 | 0.513 [0.403, 0.684] | 0.078 / 0.83 | 57.1 [46.2, 68.6] |
| gp_residual_idm | model | openacc | 0.3 / 0.92 | 26.7 [23.1, 32.1] | 0.14 / 0.72 | 15.9 [13.9, 18.4] | 0.17 / 0.054 | 0.289 [0.238, 0.338] | 0.048 / 0.42 | 14.6 [13, 16.3] |
| gp_residual_idm | seed_ensemble | highd | 0.33 / 0.75 | 6.13 [5.4, 6.95] | 0.1 / 0.52 | 9.41 [8.08, 10.8] | 0.14 / 0.32 | 2.98 [2.47, 3.6] | 0.017 / 0.22 | 16.7 [13.9, 20] |
| gp_residual_idm | seed_ensemble | highd_val | 0.33 / 0.76 | 6.25 [5.54, 6.97] | 0.1 / 0.5 | 8.98 [7.77, 10.4] | 0.15 / 0.35 | 3.12 [2.61, 3.66] | 0.016 / 0.22 | 16.7 [13.6, 20.5] |
| gp_residual_idm | seed_ensemble | ngsim | 0.65 / 0.9 | 4.86 [4.69, 5.04] | 0.46 / 0.81 | 4.96 [4.84, 5.09] | 0.74 / 0.87 | 2.25 [2.19, 2.32] | 0.057 / 0.2 | 4.08 [3.99, 4.18] |
| gp_residual_idm | seed_ensemble | ngsim_us101 | 0.58 / 0.91 | 7.66 [7.46, 7.87] | 0.35 / 0.81 | 7.76 [7.58, 7.94] | 0.55 / 0.78 | 2.78 [2.7, 2.86] | 0.069 / 0.28 | 5.21 [5.09, 5.34] |
| gp_residual_idm | seed_ensemble | ngsim_fn | 0.58 / 0.89 | 5.62 [5.17, 6.08] | 0.36 / 0.77 | 5.84 [5.5, 6.17] | 0.62 / 0.77 | 2.09 [1.95, 2.23] | 0.059 / 0.25 | 5.28 [5.03, 5.59] |
| gp_residual_idm | seed_ensemble | waymo | 0.6 / 0.91 | 6.6 [5.73, 7.64] | 0.39 / 0.82 | 7.11 [6.39, 7.92] | 0.56 / 0.76 | 2.55 [2.15, 3.01] | 0.078 / 0.32 | 5.54 [4.94, 6.22] |
| gp_residual_idm | seed_ensemble | openacc | 0.3 / 0.57 | 3.11 [2.78, 3.54] | 0.14 / 0.31 | 2.72 [2.45, 3.07] | 0.17 / 0.084 | 0.464 [0.413, 0.526] | 0.048 / 0.17 | 4.12 [3.74, 4.6] |

### H3.3: NLL of HeteroMDN (a) against LSTMConstSigma (b)

| data set | regime | windows | NLL a | NLL b | a - b | p (Holm) | intervals separate |
|---|---|---|---|---|---|---|---|
| highd | all | 140168 | -1.69 [-1.73, -1.65] | -1.62 [-1.64, -1.58] | -0.0768 [-0.0907, -0.0599] | 0.0049 | yes |
| highd | congested | 3401 | -1.17 [-1.26, -1.06] | -1.06 [-1.22, -0.875] | -0.11 [-0.191, -0.0354] | 0.011 | no |
| highd | braking | 8158 | -1.06 [-1.2, -0.843] | -0.828 [-1.01, -0.596] | -0.229 [-0.316, -0.149] | 0.0049 | no |
| highd | approaching | 30023 | -1.64 [-1.72, -1.53] | -1.6 [-1.65, -1.53] | -0.0405 [-0.0704, 0.00829] | 0.28 | no |
| highd | free | 25497 | -1.81 [-1.85, -1.76] | -1.75 [-1.77, -1.72] | -0.0618 [-0.0775, -0.0429] | 0.0049 | no |
| highd | steady | 40715 | -1.84 [-1.88, -1.78] | -1.75 [-1.77, -1.72] | -0.0895 [-0.105, -0.0652] | 0.0049 | yes |
| highd | other | 32374 | -1.68 [-1.71, -1.64] | -1.61 [-1.65, -1.58] | -0.0646 [-0.0777, -0.0509] | 0.0049 | no |
| highd_val | all | 140168 | -1.69 [-1.72, -1.65] | -1.6 [-1.63, -1.56] | -0.0892 [-0.0983, -0.0804] | 0.0049 | yes |
| highd_val | congested | 4584 | -1.24 [-1.31, -1.17] | -1.21 [-1.33, -1.08] | -0.0316 [-0.0945, 0.0254] | 0.5 | no |
| highd_val | braking | 9101 | -1.04 [-1.17, -0.863] | -0.807 [-0.982, -0.59] | -0.236 [-0.309, -0.171] | 0.0049 | no |
| highd_val | approaching | 29159 | -1.68 [-1.72, -1.64] | -1.61 [-1.65, -1.57] | -0.0705 [-0.0882, -0.0543] | 0.0049 | no |
| highd_val | free | 24114 | -1.84 [-1.87, -1.82] | -1.77 [-1.79, -1.74] | -0.0774 [-0.0888, -0.0657] | 0.0049 | yes |
| highd_val | steady | 40242 | -1.81 [-1.85, -1.76] | -1.71 [-1.75, -1.66] | -0.0977 [-0.111, -0.0857] | 0.0049 | yes |
| highd_val | other | 32968 | -1.67 [-1.72, -1.59] | -1.59 [-1.64, -1.52] | -0.0714 [-0.0878, -0.0535] | 0.0049 | no |
| ngsim | all | 343495 | 70 [68.9, 71.2] | 259 [254, 263] | -188 [-192, -185] | 0.0049 | yes |
| ngsim | congested | 222825 | 49.8 [48.8, 50.7] | 212 [207, 216] | -162 [-166, -158] | 0.0049 | yes |
| ngsim | braking | 23795 | 149 [146, 152] | 612 [590, 634] | -463 [-484, -442] | 0.0049 | yes |
| ngsim | approaching | 32155 | 116 [113, 118] | 356 [344, 369] | -241 [-252, -230] | 0.0049 | yes |
| ngsim | free | 838 | 206 [168, 249] | 402 [341, 472] | -196 [-252, -149] | 0.0049 | yes |
| ngsim | steady | 22344 | 100 [97, 103] | 216 [208, 224] | -116 [-122, -110] | 0.0049 | yes |
| ngsim | other | 41538 | 79.5 [77.5, 81.6] | 252 [245, 259] | -172 [-178, -167] | 0.0049 | yes |
| ngsim_us101 | all | 332756 | 73.3 [72.5, 74.2] | 188 [185, 191] | -115 [-117, -113] | 0.0049 | yes |
| ngsim_us101 | congested | 112135 | 47 [46.3, 47.8] | 191 [187, 195] | -144 [-148, -141] | 0.0049 | yes |
| ngsim_us101 | braking | 41731 | 121 [119, 123] | 284 [278, 291] | -164 [-170, -158] | 0.0049 | yes |
| ngsim_us101 | approaching | 48976 | 88.7 [87, 90.4] | 177 [172, 183] | -88.5 [-92.7, -84.5] | 0.0049 | yes |
| ngsim_us101 | free | 1293 | 67.3 [59.9, 74.9] | 163 [137, 198] | -95.6 [-130, -72.2] | 0.0049 | yes |
| ngsim_us101 | steady | 45664 | 84.6 [83, 86.3] | 143 [140, 147] | -58.4 [-60.8, -56.1] | 0.0049 | yes |
| ngsim_us101 | other | 82957 | 69.8 [68.8, 70.9] | 167 [164, 170] | -97.2 [-99.4, -95] | 0.0049 | yes |
| ngsim_fn | all | 175802 | 66.2 [64.7, 67.8] | 165 [162, 169] | -99.2 [-102, -96.5] | 0.0049 | yes |
| ngsim_fn | congested | 81876 | 46.2 [45, 47.5] | 153 [148, 158] | -107 [-111, -103] | 0.0049 | yes |
| ngsim_fn | braking | 17863 | 114 [110, 117] | 259 [249, 268] | -145 [-153, -138] | 0.0049 | yes |
| ngsim_fn | approaching | 24026 | 83.6 [80.4, 86.8] | 159 [153, 166] | -75.8 [-80.1, -71.7] | 0.0049 | yes |
| ngsim_fn | free | 814 | 133 [103, 172] | 200 [162, 245] | -66.3 [-93.9, -42.4] | 0.0049 | no |
| ngsim_fn | steady | 18614 | 82 [78.4, 85.6] | 141 [135, 147] | -59.2 [-62.4, -56] | 0.0049 | yes |
| ngsim_fn | other | 32609 | 67 [64.3, 69.8] | 163 [158, 168] | -96 [-99.8, -92.2] | 0.0049 | yes |
| waymo | all | 115194 | 13.2 [12.1, 14.4] | 39.4 [35.9, 43.1] | -26.1 [-29.1, -23.3] | 0.0049 | yes |
| waymo | congested | 51771 | 11 [9.95, 12.1] | 47.9 [43, 53] | -36.9 [-41.3, -32.8] | 0.0049 | yes |
| waymo | braking | 8419 | 27.2 [23, 32] | 70.7 [58.9, 83.8] | -43.4 [-54, -34.1] | 0.0049 | yes |
| waymo | approaching | 15283 | 13.8 [11.7, 16.2] | 25.5 [21.1, 30.4] | -11.6 [-14.7, -8.95] | 0.0049 | yes |
| waymo | free | 2334 | 8.23 [3.28, 15.3] | 10.4 [5.91, 16.6] | -2.19 [-6.44, 1.57] | 0.5 | no |
| waymo | steady | 14001 | 14.3 [11.9, 16.9] | 21.5 [17.7, 25.9] | -7.26 [-9.77, -5.29] | 0.0049 | yes |
| waymo | other | 23386 | 12.6 [10.8, 14.6] | 31.9 [26.4, 38.2] | -19.3 [-24.7, -14.8] | 0.0049 | yes |
| openacc | all | 370863 | 5.27 [5.01, 5.54] | 50.5 [44.3, 57.3] | -45.3 [-52, -39.1] | 0.0049 | yes |
| openacc | congested | 28556 | 8.23 [7.53, 8.95] | 35.2 [31.8, 38.8] | -27 [-30.1, -24.1] | 0.0049 | yes |
| openacc | braking | 29858 | 17.2 [16.3, 18.2] | 88.9 [80.3, 98.1] | -71.7 [-80.8, -63.2] | 0.0049 | yes |
| openacc | approaching | 58856 | 6.61 [6.2, 7.03] | 107 [89.8, 126] | -101 [-120, -83.2] | 0.0049 | yes |
| openacc | free | 35391 | 3.05 [2.69, 3.44] | 165 [131, 205] | -162 [-202, -128] | 0.0049 | yes |
| openacc | steady | 147533 | 1.36 [1.2, 1.53] | 2.11 [1.89, 2.33] | -0.745 [-0.822, -0.673] | 0.0049 | yes |
| openacc | other | 70669 | 7.2 [6.72, 7.78] | 36.8 [32.2, 42.1] | -29.6 [-34.9, -25.1] | 0.0049 | yes |

### H3.3: variance relative to steady following, hetero_mdn (* = Holm-adjusted p < 0.05)

| data set | quantity | congested | braking | approaching | free | other |
|---|---|---|---|---|---|---|
| highd | predicted variance | 3.07 [2.86, 3.29]* | 8.55 [4.65, 15.1]* | 1.48 [1.21, 1.88]* | 0.9 [0.876, 0.923]* | 1.25 [1.21, 1.29]* |
| highd | squared residual | 3.84 [3.14, 4.61]* | 5.91 [4.12, 8.56]* | 1.61 [1.38, 1.89]* | 1.05 [0.968, 1.14] | 1.44 [1.33, 1.57]* |
| highd_val | predicted variance | 2.92 [2.76, 3.09]* | 15 [4.19, 32.3]* | 1.24 [1.18, 1.32]* | 0.948 [0.899, 1] | 1.29 [1.23, 1.37]* |
| highd_val | squared residual | 2.7 [1.97, 3.5]* | 10.3 [3.37, 23.2]* | 1.24 [0.932, 1.55] | 0.843 [0.629, 1.06] | 1.26 [0.942, 1.6] |
| ngsim | predicted variance | 0.879 [0.668, 1.19] | 7.12 [5.28, 9.71]* | 3.71 [2.82, 5.03]* | 0.789 [0.415, 1.39] | 2.6 [1.91, 3.59]* |
| ngsim | squared residual | 0.988 [0.949, 1.03] | 2.75 [2.63, 2.88]* | 1.62 [1.55, 1.7]* | 1.78 [1.52, 2.05]* | 1.2 [1.15, 1.25]* |
| ngsim_us101 | predicted variance | 4.48 [3.47, 5.57]* | 2.21 [1.62, 3.02]* | 1.71 [1.25, 2.27]* | 2.41 [1.24, 4.25]* | 4.17 [3.2, 5.24]* |
| ngsim_us101 | squared residual | 1.33 [1.29, 1.37]* | 2.01 [1.96, 2.07]* | 1.23 [1.19, 1.27]* | 1.12 [0.954, 1.32] | 1.19 [1.16, 1.22]* |
| ngsim_fn | predicted variance | 2.88 [2.44, 3.39]* | 1.49 [1.31, 1.71]* | 1.56 [1.24, 2]* | 1.38 [0.921, 2.09] | 3.08 [2.5, 3.76]* |
| ngsim_fn | squared residual | 1.11 [1.05, 1.16]* | 1.86 [1.77, 1.94]* | 1.12 [1.07, 1.17]* | 1.36 [1.13, 1.65]* | 1.19 [1.13, 1.24]* |
| waymo | predicted variance | 4.15 [3, 5.34]* | 12.8 [2.74, 35.8]* | 4.44 [1.71, 9.4]* | 20.1 [3.91, 50.5]* | 3.43 [2.23, 5.17]* |
| waymo | squared residual | 2.15 [1.79, 2.59]* | 3.42 [2.69, 4.37]* | 1.27 [1.03, 1.55]* | 0.824 [0.437, 1.5] | 1.43 [1.2, 1.73]* |
| openacc | predicted variance | 4.18 [3.5, 5.06]* | 322 [274, 371]* | 656 [578, 740]* | 1.6e+03 [1.41e+03, 1.8e+03]* | 307 [262, 357]* |
| openacc | squared residual | 9.02 [8.13, 9.98]* | 57.7 [50.2, 65.6]* | 72.1 [62.4, 82.2]* | 154 [132, 177]* | 31.2 [27, 35.9]* |

### H3.3: variance relative to steady following, deep_ensemble (* = Holm-adjusted p < 0.05)

| data set | quantity | congested | braking | approaching | free | other |
|---|---|---|---|---|---|---|
| highd | predicted variance | 3.27 [3.02, 3.55]* | 7.82 [4.72, 12.6]* | 1.4 [1.25, 1.6]* | 0.913 [0.885, 0.941]* | 1.29 [1.24, 1.33]* |
| highd | squared residual | 3.61 [2.94, 4.35]* | 5.01 [3.88, 6.4]* | 1.52 [1.31, 1.78]* | 1.02 [0.931, 1.11] | 1.43 [1.31, 1.56]* |
| highd_val | predicted variance | 2.99 [2.71, 3.26]* | 10.1 [3.97, 20.6]* | 1.25 [1.12, 1.37]* | 0.975 [0.852, 1.15] | 1.29 [1.17, 1.4]* |
| highd_val | squared residual | 2.56 [1.82, 3.36]* | 5.34 [3, 9.37]* | 1.17 [0.853, 1.48] | 0.8 [0.584, 1.01] | 1.23 [0.891, 1.61] |
| ngsim | predicted variance | 0.863 [0.743, 1]* | 5.09 [4.27, 6.03]* | 3.13 [2.69, 3.66]* | 2.84 [1.27, 5.03]* | 2.01 [1.71, 2.36]* |
| ngsim | squared residual | 0.944 [0.906, 0.984]* | 2.63 [2.51, 2.75]* | 1.62 [1.54, 1.69]* | 1.8 [1.54, 2.1]* | 1.19 [1.14, 1.24]* |
| ngsim_us101 | predicted variance | 2.85 [2.59, 3.1]* | 1.53 [1.33, 1.78]* | 1.6 [1.42, 1.8]* | 3.67 [2.08, 5.67]* | 3.06 [2.76, 3.35]* |
| ngsim_us101 | squared residual | 1.29 [1.25, 1.32]* | 1.96 [1.91, 2.02]* | 1.23 [1.19, 1.27]* | 1.11 [0.945, 1.34] | 1.19 [1.16, 1.22]* |
| ngsim_fn | predicted variance | 1.81 [1.6, 2.05]* | 1.51 [1.32, 1.71]* | 1.62 [1.4, 1.87]* | 3.06 [1.04, 7.2]* | 2.17 [1.89, 2.47]* |
| ngsim_fn | squared residual | 1.07 [1.02, 1.12]* | 1.81 [1.73, 1.89]* | 1.12 [1.07, 1.18]* | 1.37 [1.13, 1.67]* | 1.18 [1.13, 1.24]* |
| waymo | predicted variance | 5.36 [4.66, 6.06]* | 13 [4.51, 25.7]* | 10.1 [2.83, 24.2]* | 26.8 [3.34, 77.2]* | 5.06 [3.33, 7.22]* |
| waymo | squared residual | 2.13 [1.78, 2.57]* | 3.21 [2.57, 4.03]* | 1.3 [1.04, 1.64]* | 0.95 [0.433, 1.9] | 1.49 [1.23, 1.82]* |
| openacc | predicted variance | 5.28 [4.46, 6.39]* | 415 [359, 470]* | 797 [710, 892]* | 1.64e+03 [1.46e+03, 1.83e+03]* | 257 [220, 297]* |
| openacc | squared residual | 8.76 [7.9, 9.67]* | 73.6 [63.8, 83.9]* | 107 [92.7, 122]* | 213 [184, 244]* | 37.3 [31.9, 43.1]* |

### H3.3: variance relative to steady following, gp_residual_idm (* = Holm-adjusted p < 0.05)

| data set | quantity | congested | braking | approaching | free | other |
|---|---|---|---|---|---|---|
| highd | predicted variance | 1.01 [1, 1.01]* | 1.1 [1.05, 1.15]* | 1.03 [1.02, 1.04]* | 1.02 [1.01, 1.03]* | 1.01 [1.01, 1.01]* |
| highd | squared residual | 3.1 [2.58, 3.67]* | 12.5 [10.7, 14.5]* | 1.66 [1.47, 1.86]* | 0.93 [0.84, 1.03] | 1.7 [1.55, 1.86]* |
| highd_val | predicted variance | 1 [1, 1.01]* | 1.06 [1.03, 1.1]* | 1.03 [1.01, 1.05]* | 1.04 [1.02, 1.07]* | 1.01 [1.01, 1.02]* |
| highd_val | squared residual | 3.33 [2.7, 4.11]* | 11.7 [10, 13.7]* | 2.03 [1.77, 2.33]* | 0.952 [0.862, 1.05] | 1.64 [1.5, 1.81]* |
| ngsim | predicted variance | 1.1 [1.09, 1.1]* | 1.21 [1.19, 1.23]* | 1.12 [1.11, 1.13]* | 2.65 [2.19, 3.13]* | 1.11 [1.1, 1.12]* |
| ngsim | squared residual | 3.23 [3.07, 3.4]* | 2.94 [2.79, 3.09]* | 1.96 [1.87, 2.06]* | 1.83 [1.62, 2.07]* | 1.27 [1.21, 1.35]* |
| ngsim_us101 | predicted variance | 1.17 [1.16, 1.18]* | 1.11 [1.1, 1.12]* | 1.04 [1.04, 1.05]* | 2.24 [1.97, 2.54]* | 1.11 [1.1, 1.12]* |
| ngsim_us101 | squared residual | 3.8 [3.62, 3.99]* | 2.38 [2.29, 2.47]* | 1.61 [1.56, 1.67]* | 1.34 [1.14, 1.6]* | 1.1 [1.07, 1.13]* |
| ngsim_fn | predicted variance | 1.09 [1.08, 1.1]* | 1.17 [1.14, 1.2]* | 1.07 [1.06, 1.1]* | 2.23 [1.66, 2.84]* | 1.08 [1.06, 1.09]* |
| ngsim_fn | squared residual | 3 [2.7, 3.31]* | 3.01 [2.72, 3.33]* | 1.73 [1.62, 1.86]* | 1.52 [1.27, 1.81]* | 1.24 [1.18, 1.3]* |
| waymo | predicted variance | 1.13 [1.11, 1.16]* | 1.71 [1.51, 1.94]* | 1.16 [1.09, 1.26]* | 1.18 [1.05, 1.35]* | 1.14 [1.09, 1.2]* |
| waymo | squared residual | 6.41 [4.87, 8.31]* | 6 [4.24, 8.25]* | 2.55 [2.07, 3.13]* | 0.632 [0.48, 0.823]* | 1.47 [1.27, 1.7]* |
| openacc | predicted variance | 1.08 [1.07, 1.09]* | 1.4 [1.36, 1.45]* | 1.4 [1.35, 1.44]* | 1.86 [1.76, 1.97]* | 1.28 [1.25, 1.3]* |
| openacc | squared residual | 7.88 [6.44, 9.58]* | 32.7 [29.3, 36.1]* | 27.3 [24.2, 30.9]* | 6.19 [5.61, 6.83]* | 5.72 [5.33, 6.16]* |

### Share of windows per regime

| data set | congested | braking | approaching | free | steady | other |
|---|---|---|---|---|---|---|
| highd | 0.024 | 0.058 | 0.21 | 0.18 | 0.29 | 0.23 |
| highd_val | 0.033 | 0.065 | 0.21 | 0.17 | 0.29 | 0.24 |
| ngsim | 0.65 | 0.069 | 0.094 | 0.0024 | 0.065 | 0.12 |
| ngsim_us101 | 0.34 | 0.13 | 0.15 | 0.0039 | 0.14 | 0.25 |
| ngsim_fn | 0.47 | 0.1 | 0.14 | 0.0046 | 0.11 | 0.19 |
| waymo | 0.45 | 0.073 | 0.13 | 0.02 | 0.12 | 0.2 |
| openacc | 0.077 | 0.081 | 0.16 | 0.095 | 0.4 | 0.19 |

### Signal: one-step predictability of the acceleration without any model

| dataset | target | persistence | n_windows | n_events | std_a | rmse_zero | rmse_persistence | share_v_below_8 | share_s_below_15 | share_absdv_above_3 |
|---|---|---|---|---|---|---|---|---|---|---|
| highd | mean_1s | y_prev | 140168 | 2503 | 0.349 | 0.35 | 0.157 | 0.0403 | 0.137 | 0.0167 |
| highd_val | mean_1s | y_prev | 140168 | 2503 | 0.361 | 0.363 | 0.161 | 0.0474 | 0.149 | 0.0165 |
| ngsim | mean_1s | y_prev | 343495 | 6085 | 0.82 | 0.82 | 1.08 | 0.75 | 0.743 | 0.0569 |
| ngsim_us101 | mean_1s | y_prev | 332756 | 5896 | 0.802 | 0.804 | 0.919 | 0.403 | 0.554 | 0.0694 |
| ngsim_fn | mean_1s | y_prev | 175802 | 1549 | 0.733 | 0.733 | 0.857 | 0.601 | 0.617 | 0.0591 |
| waymo | mean_1s | y_prev | 115194 | 748 | 0.642 | 0.643 | 0.397 | 0.533 | 0.556 | 0.0776 |
| openacc | mean_1s | y_prev | 370863 | 3668 | 0.457 | 0.457 | 0.24 | 0.17 | 0.165 | 0.0481 |

### Windows in states that are rare in (s, dv, v)

| dataset | share_rare3d | share_unseen3d | share_outside3d |
|---|---|---|---|
| highd | 0.0991 | 0.00251 | 0.00171 |
| highd_val | 0.106 | 0.0037 | 0.00174 |
| ngsim | 0.493 | 0.0285 | 0.007 |
| ngsim_us101 | 0.386 | 0.0203 | 0.00913 |
| ngsim_fn | 0.394 | 0.0221 | 0.00681 |
| waymo | 0.442 | 0.0286 | 0.0171 |
| openacc | 0.305 | 0.101 | 0.0103 |

### Calibration of the predicted variance

| model | dataset | n_seeds | mse | predicted_var | mse_over_var | median_z2 | share_z_above_3 |
|---|---|---|---|---|---|---|---|
| lstm_const_sigma | highd | 5 | 0.00231 | 0.0024 | 0.965 | 0.28 | 0.0116 |
| lstm_const_sigma | highd_val | 5 | 0.0024 | 0.0024 | 1 | 0.286 | 0.012 |
| lstm_const_sigma | ngsim | 5 | 1.25 | 0.0024 | 521 | 119 | 0.832 |
| lstm_const_sigma | ngsim_us101 | 5 | 0.911 | 0.0024 | 380 | 130 | 0.851 |
| lstm_const_sigma | ngsim_fn | 5 | 0.803 | 0.0024 | 335 | 115 | 0.838 |
| lstm_const_sigma | waymo | 5 | 0.199 | 0.0024 | 82.8 | 17.4 | 0.611 |
| lstm_const_sigma | openacc | 5 | 0.251 | 0.0024 | 105 | 3.68 | 0.368 |
| hetero_mdn | highd | 5 | 0.00245 | 0.00288 | 0.851 | 0.335 | 0.0102 |
| hetero_mdn | highd_val | 5 | 0.00327 | 0.00374 | 0.873 | 0.335 | 0.00979 |
| hetero_mdn | ngsim | 5 | 1.21 | 0.0455 | 26.7 | 38.2 | 0.717 |
| hetero_mdn | ngsim_us101 | 5 | 0.889 | 0.0193 | 46.1 | 54.6 | 0.771 |
| hetero_mdn | ngsim_fn | 5 | 0.793 | 0.0122 | 65 | 48.2 | 0.757 |
| hetero_mdn | waymo | 5 | 0.203 | 0.0174 | 11.6 | 7.18 | 0.458 |
| hetero_mdn | openacc | 5 | 0.766 | 0.972 | 0.788 | 1.64 | 0.236 |
| deep_ensemble | highd | 5 | 0.00221 | 0.00304 | 0.727 | 0.283 | 0.00695 |
| deep_ensemble | highd_val | 5 | 0.00252 | 0.00359 | 0.701 | 0.285 | 0.00636 |
| deep_ensemble | ngsim | 5 | 1.18 | 0.0732 | 16.1 | 17.2 | 0.61 |
| deep_ensemble | ngsim_us101 | 5 | 0.88 | 0.0356 | 24.7 | 26.3 | 0.682 |
| deep_ensemble | ngsim_fn | 5 | 0.781 | 0.0256 | 30.5 | 23.6 | 0.666 |
| deep_ensemble | waymo | 5 | 0.198 | 0.0287 | 6.92 | 5.02 | 0.389 |
| deep_ensemble | openacc | 5 | 1.03 | 1.2 | 0.858 | 1.27 | 0.206 |
| mc_dropout | highd | 5 | 0.00227 | 0.0147 | 0.155 | 0.199 | 0.00251 |
| mc_dropout | highd_val | 5 | 0.00271 | 0.016 | 0.169 | 0.199 | 0.00219 |
| mc_dropout | ngsim | 5 | 1.3 | 0.856 | 1.52 | 11.8 | 0.553 |
| mc_dropout | ngsim_us101 | 5 | 0.943 | 0.689 | 1.37 | 14.5 | 0.592 |
| mc_dropout | ngsim_fn | 5 | 0.791 | 0.441 | 1.79 | 14.1 | 0.589 |
| mc_dropout | waymo | 5 | 0.253 | 0.446 | 0.567 | 2.85 | 0.279 |
| mc_dropout | openacc | 5 | 3.11 | 3.96 | 0.786 | 0.908 | 0.133 |
| gp_residual_idm | highd | 5 | 0.0802 | 0.0857 | 0.936 | 0.228 | 0.0142 |
| gp_residual_idm | highd_val | 5 | 0.0857 | 0.0861 | 0.996 | 0.233 | 0.0164 |
| gp_residual_idm | ngsim | 5 | 1.22 | 0.0939 | 13 | 1.84 | 0.202 |
| gp_residual_idm | ngsim_us101 | 5 | 0.781 | 0.0936 | 8.34 | 1.9 | 0.178 |
| gp_residual_idm | ngsim_fn | 5 | 0.703 | 0.0922 | 7.63 | 1.76 | 0.166 |
| gp_residual_idm | waymo | 5 | 0.546 | 0.0984 | 5.55 | 0.915 | 0.0894 |
| gp_residual_idm | openacc | 5 | 0.286 | 0.104 | 2.74 | 0.223 | 0.0466 |

### Epistemic variance and rarity in (s, dv, v)

| model | dataset | share_windows_rare3d | share_epistemic_in_rare3d | ratio_rare3d | ratio_rare3d_lo | ratio_rare3d_hi | r_epi | r_epi_lo | r_epi_hi | r_epi_matched3d | r_epi_matched3d_lo | r_epi_matched3d_hi | n_cells_matched | mass_in_matched | mass_shift_matched |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deep_ensemble | highd | 0.0991 | 0.601 | 13.7 | 7.19 | 25.3 | - | - | - | - | - | - | - | - | - |
| deep_ensemble | highd_val | 0.106 | 0.699 | 19.6 | 8.32 | 36.4 | 1.46 | 0.749 | 2.65 | 1.05 | 0.974 | 1.15 | 405 | 0.925 | 0.922 |
| deep_ensemble | ngsim | 0.493 | 0.662 | 2.01 | 1.94 | 2.09 | 72.7 | 49.5 | 103 | 80.2 | 73 | 87.2 | 213 | 0.212 | 0.647 |
| deep_ensemble | ngsim_us101 | 0.386 | 0.623 | 2.63 | 2.52 | 2.75 | 44.7 | 30.4 | 63.4 | 24.2 | 22.7 | 25.5 | 229 | 0.22 | 0.713 |
| deep_ensemble | ngsim_fn | 0.394 | 0.586 | 2.18 | 2.05 | 2.33 | 34.7 | 23.6 | 49.4 | 37.9 | 33.3 | 45 | 222 | 0.404 | 0.691 |
| deep_ensemble | waymo | 0.442 | 0.849 | 7.08 | 5.22 | 9.76 | 27.3 | 17.6 | 42.2 | 5.81 | 4.3 | 17.3 | 290 | 0.702 | 0.669 |
| deep_ensemble | openacc | 0.305 | 0.997 | 743 | 664 | 822 | 441 | 303 | 635 | 5.67 | 4.85 | 6.94 | 402 | 0.892 | 0.74 |
| mc_dropout | highd | 0.0991 | 0.354 | 4.99 | 4.15 | 5.93 | - | - | - | - | - | - | - | - | - |
| mc_dropout | highd_val | 0.106 | 0.358 | 4.7 | 3.88 | 5.61 | 1.09 | 0.954 | 1.24 | 1.02 | 0.938 | 1.11 | 405 | 0.925 | 0.922 |
| mc_dropout | ngsim | 0.493 | 0.678 | 2.17 | 2.09 | 2.26 | 14 | 12.8 | 15.4 | 7.66 | 6.85 | 8.31 | 213 | 0.212 | 0.647 |
| mc_dropout | ngsim_us101 | 0.386 | 0.639 | 2.82 | 2.74 | 2.92 | 11.2 | 10.2 | 12.2 | 3.95 | 3.6 | 4.23 | 229 | 0.22 | 0.713 |
| mc_dropout | ngsim_fn | 0.394 | 0.601 | 2.32 | 2.17 | 2.49 | 9.24 | 8.34 | 10.2 | 7.4 | 6.29 | 8.86 | 222 | 0.404 | 0.691 |
| mc_dropout | waymo | 0.442 | 0.756 | 3.91 | 3.4 | 4.56 | 7.71 | 6.74 | 8.95 | 2.32 | 1.97 | 3.8 | 290 | 0.702 | 0.669 |
| mc_dropout | openacc | 0.305 | 0.979 | 105 | 93.1 | 116 | 44.4 | 38.5 | 51.1 | 3.08 | 2.81 | 3.53 | 402 | 0.892 | 0.74 |
| gp_residual_idm | highd | 0.0991 | 0.923 | 109 | 80.6 | 143 | - | - | - | - | - | - | - | - | - |
| gp_residual_idm | highd_val | 0.106 | 0.942 | 138 | 98.3 | 189 | 1.24 | 0.808 | 1.95 | 1.03 | 0.977 | 1.06 | 405 | 0.925 | 0.922 |
| gp_residual_idm | ngsim | 0.493 | 0.982 | 55 | 52.3 | 57.7 | 6.46 | 5.04 | 8.71 | 1.26 | 1.22 | 1.31 | 213 | 0.212 | 0.647 |
| gp_residual_idm | ngsim_us101 | 0.386 | 0.986 | 113 | 108 | 119 | 6.29 | 4.85 | 8.64 | 1.22 | 1.19 | 1.28 | 229 | 0.22 | 0.713 |
| gp_residual_idm | ngsim_fn | 0.394 | 0.981 | 77.6 | 69.6 | 85 | 5.33 | 4.03 | 7.35 | 1.2 | 1.17 | 1.25 | 222 | 0.404 | 0.691 |
| gp_residual_idm | waymo | 0.442 | 0.989 | 110 | 86.8 | 134 | 9.48 | 6.71 | 13.5 | 1.3 | 1.21 | 1.39 | 290 | 0.702 | 0.669 |
| gp_residual_idm | openacc | 0.305 | 0.995 | 429 | 386 | 471 | 13.4 | 10.3 | 18.7 | 1.53 | 1.47 | 1.58 | 402 | 0.892 | 0.74 |

### Shift ratios: geometric means against means

| model | dataset | quantity | ratio_geometric | geo_lo | geo_hi | ratio_means | means_lo | means_hi |
|---|---|---|---|---|---|---|---|---|
| lstm_const_sigma | highd_val | aleatoric_var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | highd_val | var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | highd_val | sq_err | 1.03 | 0.988 | 1.07 | 1.04 | 0.948 | 1.14 |
| lstm_const_sigma | ngsim | aleatoric_var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | ngsim | var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | ngsim | sq_err | 424 | 410 | 439 | 540 | 507 | 576 |
| lstm_const_sigma | ngsim_us101 | aleatoric_var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | ngsim_us101 | var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | ngsim_us101 | sq_err | 446 | 432 | 461 | 394 | 370 | 417 |
| lstm_const_sigma | ngsim_fn | aleatoric_var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | ngsim_fn | var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | ngsim_fn | sq_err | 395 | 381 | 411 | 347 | 324 | 369 |
| lstm_const_sigma | waymo | aleatoric_var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | waymo | var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | waymo | sq_err | 66.5 | 61.1 | 72.5 | 85.8 | 77.6 | 95 |
| lstm_const_sigma | openacc | aleatoric_var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | openacc | var | 1 | 1 | 1 | 1 | 1 | 1 |
| lstm_const_sigma | openacc | sq_err | 16.9 | 15.9 | 18 | 108 | 94.4 | 125 |
| hetero_mdn | highd_val | aleatoric_var | 1.03 | 1.01 | 1.06 | 1.3 | 0.774 | 2.05 |
| hetero_mdn | highd_val | var | 1.03 | 1.01 | 1.06 | 1.3 | 0.774 | 2.05 |
| hetero_mdn | highd_val | sq_err | 1.03 | 0.986 | 1.08 | 1.33 | 0.938 | 1.92 |
| hetero_mdn | ngsim | aleatoric_var | 4.02 | 3.93 | 4.1 | 15.8 | 12.1 | 19.6 |
| hetero_mdn | ngsim | var | 4.02 | 3.93 | 4.1 | 15.8 | 12.1 | 19.6 |
| hetero_mdn | ngsim | sq_err | 445 | 429 | 461 | 495 | 439 | 549 |
| hetero_mdn | ngsim_us101 | aleatoric_var | 3.02 | 2.96 | 3.07 | 6.7 | 5.13 | 8.27 |
| hetero_mdn | ngsim_us101 | var | 3.02 | 2.96 | 3.07 | 6.7 | 5.13 | 8.27 |
| hetero_mdn | ngsim_us101 | sq_err | 466 | 450 | 482 | 363 | 322 | 401 |
| hetero_mdn | ngsim_fn | aleatoric_var | 3.02 | 2.95 | 3.09 | 4.24 | 3.2 | 5.34 |
| hetero_mdn | ngsim_fn | var | 3.02 | 2.95 | 3.09 | 4.24 | 3.2 | 5.34 |
| hetero_mdn | ngsim_fn | sq_err | 419 | 403 | 438 | 324 | 287 | 359 |
| hetero_mdn | waymo | aleatoric_var | 3.21 | 3.05 | 3.38 | 6.06 | 3.91 | 9.28 |
| hetero_mdn | waymo | var | 3.21 | 3.05 | 3.38 | 6.06 | 3.91 | 9.28 |
| hetero_mdn | waymo | sq_err | 71.6 | 65.7 | 78 | 82.7 | 71.1 | 94 |
| hetero_mdn | openacc | aleatoric_var | 3.72 | 3.49 | 3.95 | 338 | 258 | 430 |
| hetero_mdn | openacc | var | 3.72 | 3.49 | 3.95 | 338 | 258 | 430 |
| hetero_mdn | openacc | sq_err | 23.1 | 21.5 | 24.9 | 312 | 267 | 365 |
| deep_ensemble | highd_val | epistemic_var | 1.05 | 1 | 1.09 | 1.46 | 0.749 | 2.65 |
| deep_ensemble | highd_val | aleatoric_var | 1.03 | 1 | 1.05 | 1.15 | 0.854 | 1.57 |
| deep_ensemble | highd_val | var | 1.03 | 1.01 | 1.06 | 1.18 | 0.845 | 1.67 |
| deep_ensemble | highd_val | sq_err | 1.04 | 0.993 | 1.09 | 1.14 | 0.967 | 1.35 |
| deep_ensemble | ngsim | epistemic_var | 57.4 | 55.5 | 59.3 | 72.7 | 49.5 | 103 |
| deep_ensemble | ngsim | aleatoric_var | 4.2 | 4.11 | 4.28 | 18 | 15.2 | 20.7 |
| deep_ensemble | ngsim | var | 7.82 | 7.65 | 7.98 | 24.1 | 20 | 28.2 |
| deep_ensemble | ngsim | sq_err | 469 | 453 | 487 | 535 | 492 | 579 |
| deep_ensemble | ngsim_us101 | epistemic_var | 41.3 | 39.9 | 42.6 | 44.7 | 30.4 | 63.4 |
| deep_ensemble | ngsim_us101 | aleatoric_var | 3.08 | 3.02 | 3.14 | 7.57 | 6.43 | 8.67 |
| deep_ensemble | ngsim_us101 | var | 5.57 | 5.45 | 5.68 | 11.7 | 9.75 | 13.6 |
| deep_ensemble | ngsim_us101 | sq_err | 497 | 481 | 515 | 398 | 367 | 430 |
| deep_ensemble | ngsim_fn | epistemic_var | 39.1 | 37.6 | 40.7 | 34.7 | 23.6 | 49.4 |
| deep_ensemble | ngsim_fn | aleatoric_var | 3.09 | 3.02 | 3.15 | 5.09 | 4.23 | 5.94 |
| deep_ensemble | ngsim_fn | var | 5.47 | 5.34 | 5.6 | 8.41 | 6.87 | 9.88 |
| deep_ensemble | ngsim_fn | sq_err | 444 | 427 | 463 | 353 | 325 | 383 |
| deep_ensemble | waymo | epistemic_var | 13.7 | 12.4 | 15.2 | 27.3 | 17.6 | 42.2 |
| deep_ensemble | waymo | aleatoric_var | 3.26 | 3.1 | 3.43 | 7.18 | 4.24 | 12.1 |
| deep_ensemble | waymo | var | 4.11 | 3.88 | 4.38 | 9.44 | 6.07 | 15 |
| deep_ensemble | waymo | sq_err | 76.7 | 70.4 | 83.4 | 89.8 | 79.2 | 101 |
| deep_ensemble | openacc | epistemic_var | 8.26 | 7.6 | 8.98 | 441 | 303 | 635 |
| deep_ensemble | openacc | aleatoric_var | 3.87 | 3.62 | 4.13 | 390 | 323 | 466 |
| deep_ensemble | openacc | var | 4.31 | 4.02 | 4.62 | 395 | 322 | 479 |
| deep_ensemble | openacc | sq_err | 26.1 | 24.1 | 28.2 | 467 | 407 | 537 |
| mc_dropout | highd_val | epistemic_var | 1.06 | 1.01 | 1.12 | 1.09 | 0.954 | 1.24 |
| mc_dropout | highd_val | aleatoric_var | 1.03 | 1 | 1.07 | 1.09 | 0.628 | 1.9 |
| mc_dropout | highd_val | var | 1.04 | 1.01 | 1.08 | 1.09 | 0.652 | 1.83 |
| mc_dropout | highd_val | sq_err | 1.03 | 0.989 | 1.08 | 1.19 | 0.944 | 1.49 |
| mc_dropout | ngsim | epistemic_var | 14.6 | 14 | 15.2 | 14 | 12.8 | 15.4 |
| mc_dropout | ngsim | aleatoric_var | 6.69 | 6.52 | 6.86 | 62.2 | 42.3 | 103 |
| mc_dropout | ngsim | var | 8.04 | 7.82 | 8.26 | 58.2 | 40.6 | 92.7 |
| mc_dropout | ngsim | sq_err | 476 | 461 | 493 | 573 | 508 | 638 |
| mc_dropout | ngsim_us101 | epistemic_var | 12.8 | 12.2 | 13.3 | 11.2 | 10.2 | 12.2 |
| mc_dropout | ngsim_us101 | aleatoric_var | 5.68 | 5.52 | 5.83 | 50 | 33.7 | 82.6 |
| mc_dropout | ngsim_us101 | var | 6.9 | 6.69 | 7.08 | 46.8 | 32.3 | 74.1 |
| mc_dropout | ngsim_us101 | sq_err | 493 | 477 | 509 | 415 | 367 | 459 |
| mc_dropout | ngsim_fn | epistemic_var | 11.8 | 11.3 | 12.3 | 9.24 | 8.34 | 10.2 |
| mc_dropout | ngsim_fn | aleatoric_var | 5 | 4.85 | 5.17 | 31.8 | 21.2 | 52.4 |
| mc_dropout | ngsim_fn | var | 6.09 | 5.89 | 6.3 | 30 | 20.4 | 47.3 |
| mc_dropout | ngsim_fn | sq_err | 427 | 410 | 446 | 348 | 308 | 387 |
| mc_dropout | waymo | epistemic_var | 8.11 | 7.52 | 8.8 | 7.71 | 6.74 | 8.95 |
| mc_dropout | waymo | aleatoric_var | 4.64 | 4.34 | 5 | 32.3 | 21.1 | 54.5 |
| mc_dropout | waymo | var | 5.16 | 4.82 | 5.57 | 30.3 | 20.3 | 49.2 |
| mc_dropout | waymo | sq_err | 77.7 | 71.3 | 85.2 | 111 | 92.7 | 135 |
| mc_dropout | openacc | epistemic_var | 4.42 | 4.12 | 4.75 | 44.4 | 38.5 | 51.1 |
| mc_dropout | openacc | aleatoric_var | 5.34 | 4.92 | 5.78 | 289 | 194 | 481 |
| mc_dropout | openacc | var | 5.33 | 4.92 | 5.76 | 269 | 185 | 429 |
| mc_dropout | openacc | sq_err | 29.3 | 27 | 31.8 | 1.37e+03 | 1.15e+03 | 1.61e+03 |
| gp_residual_idm | highd_val | epistemic_var | 1.02 | 0.942 | 1.12 | 1.24 | 0.808 | 1.95 |
| gp_residual_idm | highd_val | aleatoric_var | 1 | 1 | 1 | 1 | 1 | 1 |
| gp_residual_idm | highd_val | var | 1 | 0.998 | 1.01 | 1 | 0.996 | 1.01 |
| gp_residual_idm | highd_val | sq_err | 1.06 | 0.99 | 1.15 | 1.07 | 0.955 | 1.19 |
| gp_residual_idm | ngsim | epistemic_var | 11.8 | 10.9 | 12.5 | 6.46 | 5.04 | 8.71 |
| gp_residual_idm | ngsim | aleatoric_var | 1 | 1 | 1 | 1 | 1 | 1 |
| gp_residual_idm | ngsim | var | 1.06 | 1.05 | 1.06 | 1.1 | 1.09 | 1.1 |
| gp_residual_idm | ngsim | sq_err | 9.35 | 8.84 | 9.87 | 15.2 | 14 | 16.4 |
| gp_residual_idm | ngsim_us101 | epistemic_var | 6.43 | 6 | 6.9 | 6.29 | 4.85 | 8.64 |
| gp_residual_idm | ngsim_us101 | aleatoric_var | 1 | 1 | 1 | 1 | 1 | 1 |
| gp_residual_idm | ngsim_us101 | var | 1.05 | 1.05 | 1.06 | 1.09 | 1.08 | 1.1 |
| gp_residual_idm | ngsim_us101 | sq_err | 8.77 | 8.32 | 9.27 | 9.74 | 8.99 | 10.5 |
| gp_residual_idm | ngsim_fn | epistemic_var | 7.22 | 6.63 | 7.85 | 5.33 | 4.03 | 7.35 |
| gp_residual_idm | ngsim_fn | aleatoric_var | 1 | 1 | 1 | 1 | 1 | 1 |
| gp_residual_idm | ngsim_fn | var | 1.05 | 1.04 | 1.05 | 1.08 | 1.06 | 1.09 |
| gp_residual_idm | ngsim_fn | sq_err | 8.08 | 7.62 | 8.61 | 8.77 | 7.93 | 9.63 |
| gp_residual_idm | waymo | epistemic_var | 9.09 | 7.95 | 10.4 | 9.48 | 6.71 | 13.5 |
| gp_residual_idm | waymo | aleatoric_var | 1 | 1 | 1 | 1 | 1 | 1 |
| gp_residual_idm | waymo | var | 1.07 | 1.06 | 1.09 | 1.15 | 1.12 | 1.19 |
| gp_residual_idm | waymo | sq_err | 4.44 | 4.05 | 4.87 | 6.81 | 5.55 | 8.19 |
| gp_residual_idm | openacc | epistemic_var | 4.67 | 4.26 | 5.13 | 13.4 | 10.3 | 18.7 |
| gp_residual_idm | openacc | aleatoric_var | 1 | 1 | 1 | 1 | 1 | 1 |
| gp_residual_idm | openacc | var | 1.1 | 1.09 | 1.11 | 1.22 | 1.19 | 1.24 |
| gp_residual_idm | openacc | sq_err | 1.22 | 1.14 | 1.33 | 3.56 | 3.2 | 3.97 |

