# H3.2 against the references calibrated per driver mode (extension S6)

Reduction of the amplitude CRPS (extended set, 95 % file-cluster interval, verdict) of every stored law against the reference of its driver mode (ACC profiles: the IDM calibrated on the OpenACC ACC events; human-driven profiles: the IDM calibrated on the OpenACC human-driven events) next to the stored verdict against the highD reference. Reading rule (PLAN.md S6): reference-robust when the two verdicts agree.

## harmonised, response window

| model | mode | matched reference [interval] verdict | stored reference [interval] verdict | robust | CRPS ref. matched / stored |
|---|---|---|---|---|---|
| deep_ensemble | ACC | -18.84 [-30.41, -9.20] refuted | -13.92 [-22.76, -7.17] refuted | yes | 2.53 / 3.37 |
| diffusion_cf | ACC | 0.20 [-0.43, 0.46] inconclusive | 0.40 [-0.09, 0.59] inconclusive | yes | 2.53 / 3.37 |
| gp_residual_idm | ACC | -0.15 [-0.69, 0.21] inconclusive | 0.14 [-0.42, 0.43] inconclusive | yes | 2.53 / 3.37 |
| lstm_const_sigma | ACC | -3.91 [-6.76, -1.88] refuted | -2.69 [-4.81, -1.30] refuted | yes | 2.53 / 3.37 |
| quantile_pidl | ACC | -0.82 [-2.16, -0.25] refuted | -0.37 [-1.50, 0.08] inconclusive | no | 2.53 / 3.37 |
| deep_ensemble | Human | -26.84 [-28.12, -0.95] refuted | -48.16 [-65.66, -1.29] refuted | yes | 1.95 / 1.10 |
| diffusion_cf | Human | -0.04 [-0.09, 0.16] inconclusive | -0.84 [-1.35, 0.02] refuted | no | 1.95 / 1.10 |
| gp_residual_idm | Human | -1.23 [-1.62, -0.91] refuted | -2.95 [-5.15, -1.29] refuted | yes | 1.95 / 1.10 |
| lstm_const_sigma | Human | -3.24 [-4.35, -0.76] refuted | -6.48 [-6.95, -1.06] refuted | yes | 1.95 / 1.10 |
| quantile_pidl | Human | -0.77 [-5.63, -0.27] refuted | -2.12 [-6.76, -1.81] refuted | yes | 1.95 / 1.10 |

## harmonised, profile window

| model | mode | matched reference [interval] verdict | stored reference [interval] verdict | robust | CRPS ref. matched / stored |
|---|---|---|---|---|---|
| deep_ensemble | ACC | -11.14 [-18.33, -4.73] refuted | -7.86 [-12.89, -3.40] refuted | yes | 2.61 / 3.58 |
| diffusion_cf | ACC | 0.22 [-0.38, 0.46] inconclusive | 0.43 [-0.03, 0.61] inconclusive | yes | 2.61 / 3.58 |
| gp_residual_idm | ACC | -0.18 [-0.74, 0.20] inconclusive | 0.14 [-0.42, 0.43] inconclusive | yes | 2.61 / 3.58 |
| lstm_const_sigma | ACC | -1.65 [-3.30, -0.48] refuted | -0.93 [-2.05, -0.16] refuted | yes | 2.61 / 3.58 |
| quantile_pidl | ACC | -0.54 [-1.65, -0.10] refuted | -0.12 [-1.02, 0.21] inconclusive | no | 2.61 / 3.58 |
| deep_ensemble | Human | -18.88 [-20.18, -0.39] refuted | -29.50 [-36.90, 0.41] inconclusive | no | 1.68 / 1.09 |
| diffusion_cf | Human | -0.07 [-0.07, 0.06] inconclusive | -0.64 [-0.92, 0.60] inconclusive | yes | 1.68 / 1.09 |
| gp_residual_idm | Human | -1.36 [-2.41, -0.61] refuted | -2.62 [-4.21, -0.45] refuted | yes | 1.68 / 1.09 |
| lstm_const_sigma | Human | -2.37 [-3.60, -0.29] refuted | -4.16 [-7.24, 0.45] inconclusive | no | 1.68 / 1.09 |
| quantile_pidl | Human | -0.69 [-3.05, -0.30] refuted | -1.59 [-2.27, -0.72] refuted | yes | 1.68 / 1.09 |

## lodo_openacc, response window

| model | mode | matched reference [interval] verdict | stored reference [interval] verdict | robust | CRPS ref. matched / stored |
|---|---|---|---|---|---|
| deep_ensemble | ACC | -13.16 [-19.22, -8.46] refuted | -9.19 [-13.79, -6.30] refuted | yes | 2.55 / 3.54 |
| diffusion_cf | ACC | 0.26 [0.06, 0.36] inconclusive | 0.47 [0.27, 0.55] supported | no | 2.55 / 3.54 |
| diffusion_cf_hist | ACC | -1.02 [-2.57, -0.36] refuted | -0.45 [-1.73, 0.05] refuted | yes | 2.55 / 3.54 |
| gp_residual_idm | ACC | -0.12 [-0.53, 0.13] inconclusive | 0.19 [-0.22, 0.40] inconclusive | yes | 2.55 / 3.54 |
| idm_ar_noise | ACC | -0.19 [-0.84, 0.12] inconclusive | 0.15 [-0.42, 0.39] inconclusive | yes | 2.55 / 3.54 |
| lstm_const_sigma | ACC | -28.87 [-44.02, -15.69] refuted | -20.49 [-31.73, -12.07] refuted | yes | 2.55 / 3.54 |
| quantile_pidl | ACC | -0.62 [-1.57, -0.05] refuted | -0.16 [-0.89, 0.22] inconclusive | no | 2.55 / 3.54 |
| deep_ensemble | Human | -24.28 [-25.90, -4.75] refuted | -45.68 [-53.82, -4.72] refuted | yes | 1.95 / 1.05 |
| diffusion_cf | Human | 0.20 [0.12, 0.29] inconclusive | -0.48 [-0.58, 0.14] inconclusive | yes | 1.95 / 1.05 |
| diffusion_cf_hist | Human | -2.56 [-3.38, -1.91] refuted | -5.58 [-6.36, -3.35] refuted | yes | 1.95 / 1.05 |
| gp_residual_idm | Human | -0.35 [-1.02, -0.28] refuted | -1.49 [-1.82, -1.01] refuted | yes | 1.95 / 1.05 |
| idm_ar_noise | Human | -0.82 [-1.18, -0.45] refuted | -2.37 [-2.92, -0.47] refuted | yes | 1.95 / 1.05 |
| lstm_const_sigma | Human | -45.06 [-61.90, -2.15] refuted | -84.04 [-127.18, -2.14] refuted | yes | 1.95 / 1.05 |
| quantile_pidl | Human | -1.50 [-2.28, -0.62] refuted | -3.61 [-4.90, -1.68] refuted | yes | 1.95 / 1.05 |

## lodo_openacc, profile window

| model | mode | matched reference [interval] verdict | stored reference [interval] verdict | robust | CRPS ref. matched / stored |
|---|---|---|---|---|---|
| deep_ensemble | ACC | -8.41 [-12.19, -5.27] refuted | -5.62 [-8.14, -3.82] refuted | yes | 2.63 / 3.73 |
| diffusion_cf | ACC | 0.23 [0.03, 0.33] inconclusive | 0.46 [0.27, 0.54] supported | no | 2.63 / 3.73 |
| diffusion_cf_hist | ACC | -0.64 [-1.90, -0.12] refuted | -0.15 [-1.16, 0.23] inconclusive | no | 2.63 / 3.73 |
| gp_residual_idm | ACC | -0.06 [-0.45, 0.15] inconclusive | 0.25 [-0.15, 0.43] inconclusive | yes | 2.63 / 3.73 |
| idm_ar_noise | ACC | 0.04 [-0.31, 0.25] inconclusive | 0.33 [0.02, 0.48] inconclusive | yes | 2.63 / 3.73 |
| lstm_const_sigma | ACC | -18.32 [-27.74, -9.80] refuted | -12.59 [-18.92, -7.33] refuted | yes | 2.63 / 3.73 |
| quantile_pidl | ACC | -0.30 [-0.94, 0.09] inconclusive | 0.09 [-0.35, 0.32] inconclusive | yes | 2.63 / 3.73 |
| deep_ensemble | Human | -19.99 [-21.73, -2.07] refuted | -31.40 [-42.96, -0.22] refuted | yes | 1.68 / 1.09 |
| diffusion_cf | Human | 0.20 [-0.51, 0.28] inconclusive | -0.24 [-0.71, 0.40] inconclusive | yes | 1.68 / 1.09 |
| diffusion_cf_hist | Human | -2.65 [-3.16, -1.06] refuted | -4.63 [-7.73, 0.18] inconclusive | no | 1.68 / 1.09 |
| gp_residual_idm | Human | -0.36 [-1.76, -0.06] refuted | -1.10 [-1.37, -0.10] refuted | yes | 1.68 / 1.09 |
| idm_ar_noise | Human | -0.48 [-0.75, -0.17] refuted | -1.28 [-2.67, 0.38] inconclusive | no | 1.68 / 1.09 |
| lstm_const_sigma | Human | -35.67 [-49.44, -0.80] refuted | -55.60 [-74.92, 0.28] inconclusive | no | 1.68 / 1.09 |
| quantile_pidl | Human | -1.31 [-2.15, -0.43] refuted | -2.57 [-5.60, 0.34] inconclusive | no | 1.68 / 1.09 |

## pool, response window

| model | mode | matched reference [interval] verdict | stored reference [interval] verdict | robust | CRPS ref. matched / stored |
|---|---|---|---|---|---|
| deep_ensemble | ACC | -5.02 [-8.28, -2.36] refuted | -3.34 [-5.57, -1.63] refuted | yes | 2.55 / 3.53 |
| diffusion_cf | ACC | 0.39 [0.09, 0.52] inconclusive | 0.56 [0.29, 0.67] supported | no | 2.55 / 3.53 |
| diffusion_cf_hist | ACC | -0.83 [-2.08, -0.25] refuted | -0.32 [-1.35, 0.11] inconclusive | no | 2.55 / 3.53 |
| gp_residual_idm | ACC | -0.09 [-0.42, 0.12] inconclusive | 0.21 [-0.16, 0.40] inconclusive | yes | 2.55 / 3.53 |
| idm_ar_noise | ACC | -0.17 [-0.81, 0.14] inconclusive | 0.16 [-0.39, 0.39] inconclusive | yes | 2.55 / 3.53 |
| lstm_const_sigma | ACC | -9.10 [-11.85, -6.03] refuted | -6.28 [-8.17, -4.58] refuted | yes | 2.55 / 3.53 |
| quantile_pidl | ACC | -0.69 [-1.71, -0.19] refuted | -0.22 [-1.06, 0.14] inconclusive | no | 2.55 / 3.53 |
| deep_ensemble | Human | -7.30 [-7.80, -1.14] refuted | -14.32 [-15.85, -1.15] refuted | yes | 1.95 / 1.06 |
| diffusion_cf | Human | -0.03 [-0.06, 0.15] inconclusive | -0.90 [-1.08, 0.15] inconclusive | yes | 1.95 / 1.06 |
| diffusion_cf_hist | Human | -1.63 [-2.37, -0.82] refuted | -3.86 [-5.02, -1.63] refuted | yes | 1.95 / 1.06 |
| gp_residual_idm | Human | -0.65 [-1.20, -0.11] refuted | -2.04 [-3.51, -0.99] refuted | yes | 1.95 / 1.06 |
| idm_ar_noise | Human | -0.77 [-1.12, -0.40] refuted | -2.26 [-2.79, -0.46] refuted | yes | 1.95 / 1.06 |
| lstm_const_sigma | Human | -18.20 [-23.09, -3.02] refuted | -34.41 [-48.42, -3.04] refuted | yes | 1.95 / 1.06 |
| quantile_pidl | Human | -1.66 [-2.20, -1.06] refuted | -3.91 [-4.70, -1.76] refuted | yes | 1.95 / 1.06 |

## pool, profile window

| model | mode | matched reference [interval] verdict | stored reference [interval] verdict | robust | CRPS ref. matched / stored |
|---|---|---|---|---|---|
| deep_ensemble | ACC | -3.26 [-5.45, -1.42] refuted | -2.00 [-3.34, -0.89] refuted | yes | 2.63 / 3.73 |
| diffusion_cf | ACC | 0.39 [0.09, 0.52] inconclusive | 0.57 [0.31, 0.68] supported | no | 2.63 / 3.73 |
| diffusion_cf_hist | ACC | -0.49 [-1.50, -0.05] refuted | -0.05 [-0.84, 0.26] inconclusive | no | 2.63 / 3.73 |
| gp_residual_idm | ACC | -0.09 [-0.44, 0.12] inconclusive | 0.23 [-0.15, 0.41] inconclusive | yes | 2.63 / 3.73 |
| idm_ar_noise | ACC | 0.05 [-0.29, 0.26] inconclusive | 0.33 [0.03, 0.48] inconclusive | yes | 2.63 / 3.73 |
| lstm_const_sigma | ACC | -5.41 [-7.01, -3.42] refuted | -3.52 [-4.47, -2.44] refuted | yes | 2.63 / 3.73 |
| quantile_pidl | ACC | -0.35 [-1.15, 0.04] refuted | 0.05 [-0.59, 0.32] inconclusive | no | 2.63 / 3.73 |
| deep_ensemble | Human | -6.32 [-6.84, -0.34] refuted | -10.30 [-15.33, 0.46] inconclusive | no | 1.68 / 1.09 |
| diffusion_cf | Human | -0.14 [-0.37, -0.12] refuted | -0.76 [-1.32, 0.45] inconclusive | no | 1.68 / 1.09 |
| diffusion_cf_hist | Human | -1.66 [-2.42, -0.41] refuted | -3.11 [-6.13, 0.44] inconclusive | no | 1.68 / 1.09 |
| gp_residual_idm | Human | -0.80 [-1.94, -0.01] refuted | -1.78 [-2.92, -0.17] refuted | yes | 1.68 / 1.09 |
| idm_ar_noise | Human | -0.45 [-0.71, -0.14] refuted | -1.24 [-2.57, 0.37] inconclusive | no | 1.68 / 1.09 |
| lstm_const_sigma | Human | -13.53 [-17.83, -0.77] refuted | -21.43 [-27.49, 0.30] inconclusive | no | 1.68 / 1.09 |
| quantile_pidl | Human | -1.66 [-2.37, -0.44] refuted | -3.10 [-6.03, 0.43] inconclusive | no | 1.68 / 1.09 |

## raw, response window

| model | mode | matched reference [interval] verdict | stored reference [interval] verdict | robust | CRPS ref. matched / stored |
|---|---|---|---|---|---|
| deep_ensemble | ACC | -8.44 [-15.13, -3.78] refuted | -5.90 [-11.20, -2.65] refuted | yes | 2.53 / 3.47 |
| diffusion_cf | ACC | 0.09 [-0.69, 0.41] inconclusive | 0.34 [-0.26, 0.57] inconclusive | yes | 2.53 / 3.47 |
| diffusion_cf_hist | ACC | -4.20 [-6.45, -2.99] refuted | -2.80 [-4.65, -1.94] refuted | yes | 2.53 / 3.47 |
| gp_residual_idm | ACC | -0.16 [-0.71, 0.21] inconclusive | 0.15 [-0.40, 0.45] inconclusive | yes | 2.53 / 3.47 |
| idm_ar_noise | ACC | -0.28 [-0.95, 0.04] refuted | 0.06 [-0.52, 0.31] inconclusive | no | 2.53 / 3.47 |
| lstm_const_sigma | ACC | -21.94 [-34.26, -11.10] refuted | -15.76 [-24.77, -8.63] refuted | yes | 2.53 / 3.47 |
| quantile_pidl | ACC | -1.18 [-2.32, -0.60] refuted | -0.59 [-1.51, -0.19] refuted | yes | 2.53 / 3.47 |
| deep_ensemble | Human | -12.67 [-14.61, -1.43] refuted | -23.02 [-31.54, -1.85] refuted | yes | 1.95 / 1.11 |
| diffusion_cf | Human | -0.05 [-0.14, 0.13] inconclusive | -0.84 [-1.00, -0.02] refuted | no | 1.95 / 1.11 |
| diffusion_cf_hist | Human | -8.58 [-9.29, -3.95] refuted | -15.83 [-20.44, -4.80] refuted | yes | 1.95 / 1.11 |
| gp_residual_idm | Human | -1.14 [-1.61, -0.60] refuted | -2.75 [-4.44, -0.88] refuted | yes | 1.95 / 1.11 |
| idm_ar_noise | Human | -0.91 [-1.20, -0.59] refuted | -2.36 [-2.49, -0.87] refuted | yes | 1.95 / 1.11 |
| lstm_const_sigma | Human | -27.85 [-34.52, -6.30] refuted | -49.69 [-73.02, -7.56] refuted | yes | 1.95 / 1.11 |
| quantile_pidl | Human | -3.57 [-3.90, -3.21] refuted | -7.02 [-7.78, -4.26] refuted | yes | 1.95 / 1.11 |

## raw, profile window

| model | mode | matched reference [interval] verdict | stored reference [interval] verdict | robust | CRPS ref. matched / stored |
|---|---|---|---|---|---|
| deep_ensemble | ACC | -4.58 [-8.45, -1.81] refuted | -2.96 [-5.89, -1.11] refuted | yes | 2.61 / 3.68 |
| diffusion_cf | ACC | 0.11 [-0.63, 0.41] inconclusive | 0.37 [-0.19, 0.59] inconclusive | yes | 2.61 / 3.68 |
| diffusion_cf_hist | ACC | -2.38 [-3.84, -1.59] refuted | -1.40 [-2.53, -0.87] refuted | yes | 2.61 / 3.68 |
| gp_residual_idm | ACC | -0.21 [-0.78, 0.18] inconclusive | 0.14 [-0.42, 0.44] inconclusive | yes | 2.61 / 3.68 |
| idm_ar_noise | ACC | -0.03 [-0.40, 0.19] inconclusive | 0.27 [-0.04, 0.43] inconclusive | yes | 2.61 / 3.68 |
| lstm_const_sigma | ACC | -12.03 [-19.19, -5.44] refuted | -8.24 [-12.96, -3.97] refuted | yes | 2.61 / 3.68 |
| quantile_pidl | ACC | -0.67 [-1.60, -0.24] refuted | -0.19 [-0.90, 0.11] inconclusive | no | 2.61 / 3.68 |
| deep_ensemble | Human | -8.42 [-9.37, -0.59] refuted | -13.09 [-16.68, 0.32] inconclusive | no | 1.68 / 1.12 |
| diffusion_cf | Human | -0.07 [-0.14, 0.07] inconclusive | -0.60 [-1.18, 0.60] inconclusive | yes | 1.68 / 1.12 |
| diffusion_cf_hist | Human | -6.83 [-7.45, -2.59] refuted | -10.72 [-13.64, -0.52] refuted | yes | 1.68 / 1.12 |
| gp_residual_idm | Human | -1.26 [-2.34, -0.44] refuted | -2.38 [-3.58, -0.42] refuted | yes | 1.68 / 1.12 |
| idm_ar_noise | Human | -0.58 [-0.79, -0.35] refuted | -1.37 [-2.43, 0.34] inconclusive | no | 1.68 / 1.12 |
| lstm_const_sigma | Human | -17.76 [-22.75, -1.16] refuted | -27.08 [-34.27, 0.08] inconclusive | no | 1.68 / 1.12 |
| quantile_pidl | Human | -3.05 [-3.42, -1.91] refuted | -5.06 [-7.47, -0.24] refuted | yes | 1.68 / 1.12 |

