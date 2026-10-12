# E3 summary: ext_e3_cl

Window profile + tail. CRPS 1-4 / CRPS all: amplitude CRPS in m/s, mean over the vehicle indices 1-4 / over every measured index of a profile, then over profiles, [95 % bootstrap interval over profiles]. H3.2: reduction 1 - CRPS_model / CRPS_reference against idm_white_noise, paired by profile, with its verdict (docs/design.md 13.3, 14.1; 'not judged': fewer than 5 profiles); files CI / verdict files: the paired bootstrap over the recording files (all profiles of a drawn file kept). beta: exponent of sd_i = sd_1 i^beta; d2<0: share of negative second differences of the sd curve; collided: share of rollouts with a collision of any pair of the platoon.

## Variant raw (signals as recorded), set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| hetero_mdn_cl | 13.03 [10.23, 16.01] | 13.25 [10.69, 16.09] | -3.45 [-4.56, -2.58] | refuted | 0.34 [0.29, 0.39] | 0.56 | 0.993 |
| idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | -0.02 [-0.06, 0.03] | 0.06 | 0.000 |
| lstm_const_sigma_cl | 29.44 [20.55, 41.10] | 29.54 [20.87, 41.16] | -8.91 [-13.51, -6.00] | refuted | -0.35 [-0.52, -0.21] | 0.33 | 0.784 |

### H3.2 by driver mode: variant raw, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.38 [2.79, 4.04] | 3.47 [2.80, 4.11] | reference | - | - | - |
| ACC | 47 | hetero_mdn_cl | 12.40 [8.96, 15.75] | 12.68 [9.51, 15.91] | -2.66 [-3.67, -1.87] | refuted | [-3.90, -1.50] | refuted |
| ACC | 47 | lstm_const_sigma_cl | 28.30 [17.44, 43.87] | 28.42 [17.86, 43.72] | -7.20 [-12.03, -4.37] | refuted | [-12.83, -3.54] | refuted |
| Human | 12 | idm_white_noise | 1.11 [0.86, 1.40] | 1.11 [0.83, 1.41] | reference | - | - | - |
| Human | 12 | hetero_mdn_cl | 15.78 [12.19, 19.07] | 15.78 [12.51, 19.18] | -13.23 [-18.38, -9.83] | refuted | [-21.18, -4.06] | refuted |
| Human | 12 | lstm_const_sigma_cl | 35.64 [27.12, 42.57] | 35.64 [27.19, 41.80] | -31.15 [-42.94, -23.00] | refuted | [-42.42, -1.80] | refuted |
| all | 60 | idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | - | - |
| all | 60 | hetero_mdn_cl | 13.03 [10.23, 16.01] | 13.25 [10.69, 16.09] | -3.45 [-4.56, -2.58] | refuted | [-5.46, -2.07] | refuted |
| all | 60 | lstm_const_sigma_cl | 29.44 [20.55, 41.10] | 29.54 [20.87, 41.16] | -8.91 [-13.51, -6.00] | refuted | [-15.03, -4.96] | refuted |
| mixed | 1 | idm_white_noise | 2.51 [2.51, 2.51] | 2.51 [2.51, 2.51] | reference | - | - | - |
| mixed | 1 | hetero_mdn_cl | 9.93 [9.93, 9.93] | 9.93 [9.93, 9.93] | -2.96 [-2.96, -2.96] | not judged | [-2.96, -2.96] | not judged |
| mixed | 1 | lstm_const_sigma_cl | 8.84 [8.84, 8.84] | 8.84 [8.84, 8.84] | -2.52 [-2.52, -2.52] | not judged | [-2.52, -2.52] | not judged |

## Variant pool (pooled domain, OpenACC seen), set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| hetero_mdn_cl | 20.81 [15.75, 26.44] | 20.91 [15.86, 26.77] | -5.93 [-8.15, -4.23] | refuted | 0.08 [0.02, 0.14] | 0.56 | 0.954 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |
| lstm_const_sigma_cl | 5.16 [4.23, 6.15] | 5.36 [4.43, 6.39] | -0.77 [-1.25, -0.41] | refuted | 0.70 [0.60, 0.79] | 0.44 | 0.872 |

### H3.2 by driver mode: variant pool, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | hetero_mdn_cl | 19.11 [13.29, 25.89] | 19.24 [13.50, 26.10] | -4.45 [-6.38, -2.93] | refuted | [-7.22, -2.01] | refuted |
| ACC | 47 | lstm_const_sigma_cl | 4.92 [3.75, 6.04] | 5.17 [4.08, 6.39] | -0.46 [-0.83, -0.16] | refuted | [-0.98, 0.06] | inconclusive |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | hetero_mdn_cl | 27.19 [19.44, 34.27] | 27.19 [18.83, 34.77] | -24.78 [-29.77, -19.56] | refuted | [-27.17, 0.11] | inconclusive |
| Human | 12 | lstm_const_sigma_cl | 6.03 [4.04, 8.01] | 6.03 [4.28, 8.07] | -4.71 [-7.80, -2.66] | refuted | [-9.03, 0.18] | inconclusive |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | hetero_mdn_cl | 20.81 [15.75, 26.44] | 20.91 [15.86, 26.77] | -5.93 [-8.15, -4.23] | refuted | [-9.81, -3.17] | refuted |
| all | 60 | lstm_const_sigma_cl | 5.16 [4.23, 6.15] | 5.36 [4.43, 6.39] | -0.77 [-1.25, -0.41] | refuted | [-1.67, -0.14] | refuted |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | hetero_mdn_cl | 24.04 [24.04, 24.04] | 24.04 [24.04, 24.04] | -8.40 [-8.40, -8.40] | not judged | [-8.40, -8.40] | not judged |
| mixed | 1 | lstm_const_sigma_cl | 6.01 [6.01, 6.01] | 6.01 [6.01, 6.01] | -1.35 [-1.35, -1.35] | not judged | [-1.35, -1.35] | not judged |

## Variant lodo_openacc (pooled domain, OpenACC held out), set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| hetero_mdn_cl | 67.40 [50.97, 85.10] | 68.23 [52.67, 86.13] | -21.55 [-28.03, -16.59] | refuted | 0.10 [0.05, 0.16] | 0.50 | 0.996 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |
| lstm_const_sigma_cl | 19.06 [11.95, 30.28] | 19.16 [12.00, 30.08] | -5.33 [-9.48, -2.96] | refuted | 0.24 [0.15, 0.32] | 0.56 | 0.833 |

### H3.2 by driver mode: variant lodo_openacc, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | hetero_mdn_cl | 67.20 [48.40, 87.89] | 68.26 [48.86, 89.44] | -18.29 [-25.14, -13.46] | refuted | [-28.01, -10.42] | refuted |
| ACC | 47 | lstm_const_sigma_cl | 20.07 [11.35, 33.61] | 20.20 [11.56, 33.85] | -4.71 [-9.27, -2.24] | refuted | [-10.25, -1.75] | refuted |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | hetero_mdn_cl | 66.64 [47.09, 84.83] | 66.64 [49.08, 86.35] | -62.22 [-81.55, -47.39] | refuted | [-87.86, -0.67] | refuted |
| Human | 12 | lstm_const_sigma_cl | 15.08 [9.45, 20.85] | 15.08 [9.84, 21.15] | -13.31 [-21.38, -7.95] | refuted | [-25.21, 0.02] | refuted |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | hetero_mdn_cl | 67.40 [50.97, 85.10] | 68.23 [52.67, 86.13] | -21.55 [-28.03, -16.59] | refuted | [-32.47, -13.68] | refuted |
| all | 60 | lstm_const_sigma_cl | 19.06 [11.95, 30.28] | 19.16 [12.00, 30.08] | -5.33 [-9.48, -2.96] | refuted | [-10.44, -2.42] | refuted |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | hetero_mdn_cl | 85.91 [85.91, 85.91] | 85.91 [85.91, 85.91] | -32.38 [-32.38, -32.38] | not judged | [-32.38, -32.38] | not judged |
| mixed | 1 | lstm_const_sigma_cl | 19.21 [19.21, 19.21] | 19.21 [19.21, 19.21] | -6.47 [-6.47, -6.47] | not judged | [-6.47, -6.47] | not judged |

## Growth exponent of the measured platoons

* empirical (headline): beta 0.08 [-0.01, 0.18] fitted over indices 1..4 (measured by >= 50% of the profiles); curve to index 4 (profile window)
* empirical (extended): beta 0.12 [0.08, 0.16] fitted over indices 1..4 (measured by >= 50% of the profiles); curve to index 8 (profile window)
