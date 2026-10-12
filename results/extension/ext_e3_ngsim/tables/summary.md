# E3 summary: ext_e3_ngsim

Window profile + tail. CRPS 1-4 / CRPS all: amplitude CRPS in m/s, mean over the vehicle indices 1-4 / over every measured index of a profile, then over profiles, [95 % bootstrap interval over profiles]. H3.2: reduction 1 - CRPS_model / CRPS_reference against idm_white_noise, paired by profile, with its verdict (docs/design.md 13.3, 14.1; 'not judged': fewer than 5 profiles); files CI / verdict files: the paired bootstrap over the recording files (all profiles of a drawn file kept). beta: exponent of sd_i = sd_1 i^beta; d2<0: share of negative second differences of the sd curve; collided: share of rollouts with a collision of any pair of the platoon.

## Variant raw (signals as recorded), set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| deep_ensemble | 76.00 [45.90, 109.69] | 76.05 [46.65, 112.55] | -26.96 [-38.21, -16.70] | refuted | -0.02 [-0.04, -0.01] | 0.39 | 1.000 |
| diffusion_cf | 2.29 [1.97, 2.60] | 2.37 [2.08, 2.70] | 0.13 [-0.13, 0.31] | inconclusive | 0.07 [0.04, 0.11] | 0.78 | 0.762 |
| gp_residual_idm | 1.99 [1.69, 2.32] | 2.10 [1.81, 2.44] | 0.23 [0.06, 0.36] | inconclusive | 0.04 [-0.00, 0.09] | 0.50 | 0.676 |
| idm_ar_noise | 10.89 [10.05, 11.74] | 10.61 [9.80, 11.57] | -2.90 [-4.02, -2.06] | refuted | -0.57 [-0.60, -0.54] | 0.11 | 0.478 |
| idm_white_noise | 2.66 [2.19, 3.19] | 2.72 [2.20, 3.26] | reference | - | -0.02 [-0.06, 0.02] | 0.00 | 0.000 |
| lstm_const_sigma | 78.99 [41.97, 122.14] | 79.03 [41.91, 122.30] | -28.05 [-43.71, -14.75] | refuted | -0.07 [-0.14, -0.03] | 0.50 | 0.831 |
| quantile_pidl | 4.40 [4.08, 4.74] | 4.32 [4.00, 4.67] | -0.59 [-0.88, -0.38] | refuted | -0.85 [-0.95, -0.75] | 0.22 | 0.939 |

### H3.2 by driver mode: variant raw, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.03 [2.43, 3.65] | 3.10 [2.48, 3.69] | reference | - | - | - |
| ACC | 47 | lstm_const_sigma | 92.64 [48.36, 144.61] | 92.68 [48.39, 142.22] | -28.87 [-46.74, -14.74] | refuted | [-55.48, -8.63] | refuted |
| ACC | 47 | deep_ensemble | 82.61 [49.24, 120.23] | 82.68 [47.93, 119.99] | -25.64 [-37.57, -14.30] | refuted | [-46.17, -8.53] | refuted |
| ACC | 47 | gp_residual_idm | 2.17 [1.83, 2.60] | 2.30 [1.95, 2.70] | 0.26 [0.08, 0.39] | inconclusive | [-0.16, 0.44] | inconclusive |
| ACC | 47 | quantile_pidl | 4.57 [4.18, 4.96] | 4.46 [4.06, 4.84] | -0.44 [-0.69, -0.25] | refuted | [-1.03, -0.20] | refuted |
| ACC | 47 | diffusion_cf | 1.86 [1.65, 2.11] | 1.98 [1.76, 2.23] | 0.36 [0.21, 0.48] | supported | [-0.04, 0.52] | inconclusive |
| ACC | 47 | idm_ar_noise | 10.37 [9.50, 11.34] | 10.01 [9.08, 10.94] | -2.23 [-3.21, -1.53] | refuted | [-4.78, -1.24] | refuted |
| Human | 12 | idm_white_noise | 1.32 [0.97, 1.67] | 1.32 [0.97, 1.71] | reference | - | - | - |
| Human | 12 | lstm_const_sigma | 23.95 [9.60, 48.72] | 23.95 [10.17, 49.00] | -17.10 [-33.49, -6.16] | refuted | [-17.97, -5.59] | refuted |
| Human | 12 | deep_ensemble | 44.47 [11.62, 99.48] | 44.47 [11.71, 99.33] | -32.61 [-68.21, -7.90] | refuted | [-38.16, -7.68] | refuted |
| Human | 12 | gp_residual_idm | 1.30 [0.94, 1.68] | 1.30 [0.97, 1.70] | 0.02 [-0.56, 0.39] | inconclusive | [-1.09, 0.61] | inconclusive |
| Human | 12 | quantile_pidl | 3.74 [3.31, 4.37] | 3.74 [3.31, 4.37] | -1.83 [-2.89, -1.26] | refuted | [-3.20, -1.02] | refuted |
| Human | 12 | diffusion_cf | 3.98 [3.21, 4.60] | 3.98 [3.25, 4.59] | -2.01 [-2.95, -1.46] | refuted | [-3.39, -0.06] | refuted |
| Human | 12 | idm_ar_noise | 13.00 [11.18, 14.83] | 13.00 [11.15, 14.77] | -8.83 [-11.41, -7.33] | refuted | [-12.46, -6.22] | refuted |
| all | 60 | idm_white_noise | 2.66 [2.19, 3.19] | 2.72 [2.20, 3.26] | reference | - | - | - |
| all | 60 | lstm_const_sigma | 78.99 [41.97, 122.14] | 79.03 [41.91, 122.30] | -28.05 [-43.71, -14.75] | refuted | [-51.07, -10.43] | refuted |
| all | 60 | deep_ensemble | 76.00 [45.90, 109.69] | 76.05 [46.65, 112.55] | -26.96 [-38.21, -16.70] | refuted | [-44.98, -11.98] | refuted |
| all | 60 | gp_residual_idm | 1.99 [1.69, 2.32] | 2.10 [1.81, 2.44] | 0.23 [0.06, 0.36] | inconclusive | [-0.15, 0.42] | inconclusive |
| all | 60 | quantile_pidl | 4.40 [4.08, 4.74] | 4.32 [4.00, 4.67] | -0.59 [-0.88, -0.38] | refuted | [-1.19, -0.29] | refuted |
| all | 60 | diffusion_cf | 2.29 [1.97, 2.60] | 2.37 [2.08, 2.70] | 0.13 [-0.13, 0.31] | inconclusive | [-0.52, 0.45] | inconclusive |
| all | 60 | idm_ar_noise | 10.89 [10.05, 11.74] | 10.61 [9.80, 11.57] | -2.90 [-4.02, -2.06] | refuted | [-5.58, -1.59] | refuted |
| mixed | 1 | idm_white_noise | 1.47 [1.47, 1.47] | 1.47 [1.47, 1.47] | reference | - | - | - |
| mixed | 1 | lstm_const_sigma | 98.10 [98.10, 98.10] | 98.10 [98.10, 98.10] | -65.61 [-65.61, -65.61] | not judged | [-65.61, -65.61] | not judged |
| mixed | 1 | deep_ensemble | 143.34 [143.34, 143.34] | 143.34 [143.34, 143.34] | -96.34 [-96.34, -96.34] | not judged | [-96.34, -96.34] | not judged |
| mixed | 1 | gp_residual_idm | 2.17 [2.17, 2.17] | 2.17 [2.17, 2.17] | -0.47 [-0.47, -0.47] | not judged | [-0.47, -0.47] | not judged |
| mixed | 1 | quantile_pidl | 4.49 [4.49, 4.49] | 4.49 [4.49, 4.49] | -2.05 [-2.05, -2.05] | not judged | [-2.05, -2.05] | not judged |
| mixed | 1 | diffusion_cf | 1.67 [1.67, 1.67] | 1.67 [1.67, 1.67] | -0.14 [-0.14, -0.14] | not judged | [-0.14, -0.14] | not judged |
| mixed | 1 | idm_ar_noise | 9.93 [9.93, 9.93] | 9.93 [9.93, 9.93] | -5.74 [-5.74, -5.74] | not judged | [-5.74, -5.74] | not judged |

## Variant w61, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| deep_ensemble | 75.91 [45.89, 109.50] | 75.96 [46.52, 112.46] | -26.91 [-38.16, -16.69] | refuted | -0.02 [-0.04, -0.01] | 0.56 | 1.000 |
| diffusion_cf | 2.28 [1.96, 2.60] | 2.37 [2.07, 2.70] | 0.13 [-0.13, 0.31] | inconclusive | 0.08 [0.04, 0.11] | 0.83 | 0.760 |
| gp_residual_idm | 1.99 [1.67, 2.33] | 2.09 [1.79, 2.42] | 0.23 [0.07, 0.37] | inconclusive | 0.04 [-0.00, 0.10] | 0.61 | 0.657 |
| idm_ar_noise | 10.89 [10.04, 11.74] | 10.60 [9.80, 11.57] | -2.90 [-4.02, -2.06] | refuted | -0.57 [-0.60, -0.54] | 0.11 | 0.478 |
| idm_white_noise | 2.66 [2.18, 3.19] | 2.72 [2.20, 3.26] | reference | - | -0.02 [-0.06, 0.02] | 0.00 | 0.000 |
| lstm_const_sigma | 79.08 [42.08, 122.16] | 79.12 [42.07, 122.59] | -28.07 [-43.77, -14.75] | refuted | -0.07 [-0.14, -0.03] | 0.33 | 0.832 |
| quantile_pidl | 4.41 [4.09, 4.75] | 4.33 [4.01, 4.69] | -0.59 [-0.89, -0.38] | refuted | -0.84 [-0.95, -0.74] | 0.22 | 0.940 |

### H3.2 by driver mode: variant w61, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.03 [2.43, 3.65] | 3.11 [2.48, 3.69] | reference | - | - | - |
| ACC | 47 | lstm_const_sigma | 92.78 [48.55, 144.77] | 92.82 [48.59, 142.74] | -28.89 [-46.88, -14.80] | refuted | [-55.70, -8.62] | refuted |
| ACC | 47 | deep_ensemble | 82.51 [49.26, 120.06] | 82.58 [47.92, 119.93] | -25.59 [-37.51, -14.29] | refuted | [-46.18, -8.51] | refuted |
| ACC | 47 | gp_residual_idm | 2.17 [1.81, 2.61] | 2.29 [1.92, 2.69] | 0.26 [0.09, 0.39] | inconclusive | [-0.15, 0.44] | inconclusive |
| ACC | 47 | quantile_pidl | 4.58 [4.19, 4.98] | 4.48 [4.08, 4.85] | -0.44 [-0.69, -0.26] | refuted | [-1.04, -0.20] | refuted |
| ACC | 47 | diffusion_cf | 1.86 [1.64, 2.11] | 1.97 [1.74, 2.23] | 0.36 [0.21, 0.48] | supported | [-0.04, 0.52] | inconclusive |
| ACC | 47 | idm_ar_noise | 10.37 [9.49, 11.34] | 10.01 [9.08, 10.94] | -2.22 [-3.21, -1.52] | refuted | [-4.79, -1.24] | refuted |
| Human | 12 | idm_white_noise | 1.32 [0.97, 1.67] | 1.32 [0.97, 1.71] | reference | - | - | - |
| Human | 12 | lstm_const_sigma | 23.85 [9.56, 48.57] | 23.85 [10.13, 48.85] | -17.04 [-33.36, -6.13] | refuted | [-17.89, -5.56] | refuted |
| Human | 12 | deep_ensemble | 44.38 [11.47, 99.27] | 44.38 [11.65, 99.11] | -32.56 [-68.08, -7.89] | refuted | [-38.07, -7.54] | refuted |
| Human | 12 | gp_residual_idm | 1.28 [0.93, 1.65] | 1.28 [0.96, 1.67] | 0.03 [-0.55, 0.40] | inconclusive | [-1.08, 0.62] | inconclusive |
| Human | 12 | quantile_pidl | 3.74 [3.33, 4.34] | 3.74 [3.33, 4.34] | -1.83 [-2.93, -1.25] | refuted | [-3.25, -1.00] | refuted |
| Human | 12 | diffusion_cf | 3.98 [3.21, 4.60] | 3.98 [3.24, 4.59] | -2.01 [-2.94, -1.45] | refuted | [-3.40, -0.03] | refuted |
| Human | 12 | idm_ar_noise | 13.00 [11.17, 14.83] | 13.00 [11.15, 14.77] | -8.83 [-11.44, -7.34] | refuted | [-12.49, -6.25] | refuted |
| all | 60 | idm_white_noise | 2.66 [2.18, 3.19] | 2.72 [2.20, 3.26] | reference | - | - | - |
| all | 60 | lstm_const_sigma | 79.08 [42.08, 122.16] | 79.12 [42.07, 122.59] | -28.07 [-43.77, -14.75] | refuted | [-51.19, -10.40] | refuted |
| all | 60 | deep_ensemble | 75.91 [45.89, 109.50] | 75.96 [46.52, 112.46] | -26.91 [-38.16, -16.69] | refuted | [-44.91, -11.97] | refuted |
| all | 60 | gp_residual_idm | 1.99 [1.67, 2.33] | 2.09 [1.79, 2.42] | 0.23 [0.07, 0.37] | inconclusive | [-0.14, 0.42] | inconclusive |
| all | 60 | quantile_pidl | 4.41 [4.09, 4.75] | 4.33 [4.01, 4.69] | -0.59 [-0.89, -0.38] | refuted | [-1.21, -0.29] | refuted |
| all | 60 | diffusion_cf | 2.28 [1.96, 2.60] | 2.37 [2.07, 2.70] | 0.13 [-0.13, 0.31] | inconclusive | [-0.51, 0.45] | inconclusive |
| all | 60 | idm_ar_noise | 10.89 [10.04, 11.74] | 10.60 [9.80, 11.57] | -2.90 [-4.02, -2.06] | refuted | [-5.59, -1.58] | refuted |
| mixed | 1 | idm_white_noise | 1.47 [1.47, 1.47] | 1.47 [1.47, 1.47] | reference | - | - | - |
| mixed | 1 | lstm_const_sigma | 98.10 [98.10, 98.10] | 98.10 [98.10, 98.10] | -65.68 [-65.68, -65.68] | not judged | [-65.68, -65.68] | not judged |
| mixed | 1 | deep_ensemble | 144.11 [144.11, 144.11] | 144.11 [144.11, 144.11] | -96.96 [-96.96, -96.96] | not judged | [-96.96, -96.96] | not judged |
| mixed | 1 | gp_residual_idm | 2.15 [2.15, 2.15] | 2.15 [2.15, 2.15] | -0.46 [-0.46, -0.46] | not judged | [-0.46, -0.46] | not judged |
| mixed | 1 | quantile_pidl | 4.48 [4.48, 4.48] | 4.48 [4.48, 4.48] | -2.04 [-2.04, -2.04] | not judged | [-2.04, -2.04] | not judged |
| mixed | 1 | diffusion_cf | 1.69 [1.69, 1.69] | 1.69 [1.69, 1.69] | -0.15 [-0.15, -0.15] | not judged | [-0.15, -0.15] | not judged |
| mixed | 1 | idm_ar_noise | 9.93 [9.93, 9.93] | 9.93 [9.93, 9.93] | -5.75 [-5.75, -5.75] | not judged | [-5.75, -5.75] | not judged |

## Growth exponent of the measured platoons

* empirical (headline): beta 0.08 [-0.01, 0.18] fitted over indices 1..4 (measured by >= 50% of the profiles); curve to index 4 (profile window)
* empirical (extended): beta 0.12 [0.08, 0.16] fitted over indices 1..4 (measured by >= 50% of the profiles); curve to index 8 (profile window)
