# E3 summary: ext_e3_preview

Window profile + tail. CRPS 1-4 / CRPS all: amplitude CRPS in m/s, mean over the vehicle indices 1-4 / over every measured index of a profile, then over profiles, [95 % bootstrap interval over profiles]. H3.2: reduction 1 - CRPS_model / CRPS_reference against idm_white_noise, paired by profile, with its verdict (docs/design.md 13.3, 14.1; 'not judged': fewer than 5 profiles); files CI / verdict files: the paired bootstrap over the recording files (all profiles of a drawn file kept). beta: exponent of sd_i = sd_1 i^beta; d2<0: share of negative second differences of the sd curve; collided: share of rollouts with a collision of any pair of the platoon.

## Variant raw (signals as recorded), set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.06 [1.73, 2.46] | 2.22 [1.84, 2.62] | 0.25 [0.01, 0.44] | inconclusive | 0.22 [0.18, 0.27] | 0.72 | 0.902 |
| idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | -0.02 [-0.06, 0.03] | 0.06 | 0.000 |

### H3.2 by driver mode: variant raw, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.38 [2.79, 4.04] | 3.47 [2.80, 4.11] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 2.10 [1.70, 2.58] | 2.30 [1.86, 2.80] | 0.34 [0.09, 0.50] | inconclusive | [-0.26, 0.57] | inconclusive |
| Human | 12 | idm_white_noise | 1.11 [0.86, 1.40] | 1.11 [0.83, 1.41] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.04 [1.49, 2.59] | 2.04 [1.52, 2.57] | -0.84 [-1.26, -0.47] | refuted | [-1.00, -0.02] | refuted |
| all | 60 | idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.06 [1.73, 2.46] | 2.22 [1.84, 2.62] | 0.25 [0.01, 0.44] | inconclusive | [-0.31, 0.52] | inconclusive |
| mixed | 1 | idm_white_noise | 2.51 [2.51, 2.51] | 2.51 [2.51, 2.51] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.70 [0.70, 0.70] | 0.70 [0.70, 0.70] | 0.72 [0.72, 0.72] | not judged | [0.72, 0.72] | not judged |

## Variant lodo_openacc_d05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.68 [1.47, 1.89] | 1.78 [1.56, 2.00] | 0.41 [0.30, 0.50] | supported | 0.14 [0.09, 0.19] | 0.89 | 0.905 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_d05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.69 [1.45, 1.97] | 1.81 [1.58, 2.06] | 0.49 [0.39, 0.55] | supported | [0.25, 0.59] | supported |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.67 [1.28, 2.06] | 1.67 [1.30, 2.06] | -0.59 [-1.03, -0.29] | refuted | [-0.67, 0.16] | inconclusive |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.68 [1.47, 1.89] | 1.78 [1.56, 2.00] | 0.41 [0.30, 0.50] | supported | [0.12, 0.55] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.14 [1.14, 1.14] | 1.14 [1.14, 1.14] | 0.56 [0.56, 0.56] | not judged | [0.56, 0.56] | not judged |

## Variant lodo_openacc_d1, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.70 [1.52, 1.89] | 1.81 [1.62, 2.01] | 0.40 [0.25, 0.50] | supported | 0.14 [0.08, 0.20] | 0.89 | 0.950 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_d1, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.64 [1.43, 1.88] | 1.79 [1.58, 2.01] | 0.49 [0.37, 0.57] | supported | [0.20, 0.61] | inconclusive |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.94 [1.53, 2.34] | 1.94 [1.56, 2.33] | -0.84 [-1.28, -0.57] | refuted | [-1.04, 0.21] | inconclusive |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.70 [1.52, 1.89] | 1.81 [1.62, 2.01] | 0.40 [0.25, 0.50] | supported | [0.04, 0.57] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.19 [1.19, 1.19] | 1.19 [1.19, 1.19] | 0.54 [0.54, 0.54] | not judged | [0.54, 0.54] | not judged |

## Variant lodo_openacc_h0, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.43 [2.11, 2.80] | 2.56 [2.24, 2.94] | 0.15 [-0.07, 0.33] | inconclusive | 0.09 [0.04, 0.15] | 0.78 | 0.990 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_h0, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 2.27 [1.92, 2.61] | 2.43 [2.10, 2.79] | 0.31 [0.12, 0.44] | inconclusive | [-0.09, 0.49] | inconclusive |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 3.12 [2.22, 4.18] | 3.12 [2.23, 4.28] | -1.96 [-2.52, -1.37] | refuted | [-2.52, -0.47] | refuted |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.43 [2.11, 2.80] | 2.56 [2.24, 2.94] | 0.15 [-0.07, 0.33] | inconclusive | [-0.38, 0.43] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.93 [1.93, 1.93] | 1.93 [1.93, 1.93] | 0.25 [0.25, 0.25] | not judged | [0.25, 0.25] | not judged |

## Variant lodo_openacc_h05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.25 [1.96, 2.58] | 2.39 [2.11, 2.74] | 0.21 [0.00, 0.36] | inconclusive | 0.10 [0.04, 0.16] | 0.67 | 0.983 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_h05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 2.09 [1.81, 2.39] | 2.28 [1.99, 2.60] | 0.36 [0.20, 0.48] | inconclusive | [0.01, 0.51] | inconclusive |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.90 [2.10, 3.87] | 2.90 [2.11, 3.94] | -1.75 [-2.30, -1.23] | refuted | [-2.14, -0.36] | refuted |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.25 [1.96, 2.58] | 2.39 [2.11, 2.74] | 0.21 [0.00, 0.36] | inconclusive | [-0.28, 0.45] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.77 [1.77, 1.77] | 1.77 [1.77, 1.77] | 0.31 [0.31, 0.31] | not judged | [0.31, 0.31] | not judged |

## Variant lodo_openacc_h1, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.03 [1.81, 2.27] | 2.18 [1.94, 2.45] | 0.28 [0.08, 0.42] | inconclusive | 0.10 [0.04, 0.16] | 0.61 | 0.975 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_h1, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.88 [1.68, 2.09] | 2.07 [1.82, 2.36] | 0.42 [0.27, 0.52] | supported | [0.09, 0.55] | inconclusive |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.65 [1.99, 3.43] | 2.65 [1.99, 3.53] | -1.52 [-2.01, -1.07] | refuted | [-1.69, -0.07] | refuted |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.03 [1.81, 2.27] | 2.18 [1.94, 2.45] | 0.28 [0.08, 0.42] | inconclusive | [-0.18, 0.50] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.63 [1.63, 1.63] | 1.63 [1.63, 1.63] | 0.37 [0.37, 0.37] | not judged | [0.37, 0.37] | not judged |

## Variant lodo_openacc_h2, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.78 [1.60, 1.95] | 1.87 [1.69, 2.05] | 0.38 [0.23, 0.49] | supported | 0.08 [0.02, 0.14] | 0.78 | 0.950 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_h2, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.72 [1.54, 1.94] | 1.84 [1.64, 2.07] | 0.48 [0.37, 0.56] | supported | [0.22, 0.58] | supported |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.04 [1.65, 2.40] | 2.04 [1.68, 2.44] | -0.94 [-1.44, -0.62] | refuted | [-1.26, 0.17] | inconclusive |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.78 [1.60, 1.95] | 1.87 [1.69, 2.05] | 0.38 [0.23, 0.49] | supported | [0.03, 0.54] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.39 [1.39, 1.39] | 1.39 [1.39, 1.39] | 0.46 [0.46, 0.46] | not judged | [0.46, 0.46] | not judged |

## Variant lodo_openacc_h2d05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.97 [1.77, 2.19] | 2.12 [1.88, 2.37] | 0.30 [0.09, 0.44] | inconclusive | 0.09 [0.03, 0.16] | 0.72 | 0.969 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_h2d05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.84 [1.65, 2.06] | 2.03 [1.79, 2.29] | 0.43 [0.28, 0.53] | supported | [0.08, 0.57] | inconclusive |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.53 [1.93, 3.17] | 2.53 [1.96, 3.22] | -1.40 [-1.88, -0.97] | refuted | [-1.54, 0.10] | inconclusive |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.97 [1.77, 2.19] | 2.12 [1.88, 2.37] | 0.30 [0.09, 0.44] | inconclusive | [-0.16, 0.51] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.54 [1.54, 1.54] | 1.54 [1.54, 1.54] | 0.40 [0.40, 0.40] | not judged | [0.40, 0.40] | not judged |

## Variant lodo_openacc_h2n05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.78 [1.60, 1.95] | 1.88 [1.70, 2.07] | 0.38 [0.23, 0.48] | supported | 0.07 [0.01, 0.14] | 0.94 | 0.957 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_h2n05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.73 [1.54, 1.94] | 1.85 [1.64, 2.07] | 0.48 [0.36, 0.55] | supported | [0.22, 0.58] | supported |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.04 [1.65, 2.43] | 2.04 [1.66, 2.44] | -0.94 [-1.41, -0.62] | refuted | [-1.18, 0.19] | inconclusive |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.78 [1.60, 1.95] | 1.88 [1.70, 2.07] | 0.38 [0.23, 0.48] | supported | [0.03, 0.54] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.34 [1.34, 1.34] | 1.34 [1.34, 1.34] | 0.48 [0.48, 0.48] | not judged | [0.48, 0.48] | not judged |

## Variant lodo_openacc_h2n05d05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.94 [1.73, 2.14] | 2.07 [1.84, 2.32] | 0.32 [0.12, 0.45] | inconclusive | 0.09 [0.03, 0.15] | 0.78 | 0.975 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_h2n05d05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.82 [1.62, 2.05] | 1.99 [1.75, 2.27] | 0.44 [0.29, 0.54] | supported | [0.12, 0.57] | inconclusive |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.42 [1.88, 3.05] | 2.42 [1.88, 3.07] | -1.30 [-1.72, -0.91] | refuted | [-1.45, 0.08] | inconclusive |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.94 [1.73, 2.14] | 2.07 [1.84, 2.32] | 0.32 [0.12, 0.45] | inconclusive | [-0.11, 0.52] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.42 [1.42, 1.42] | 1.42 [1.42, 1.42] | 0.45 [0.45, 0.45] | not judged | [0.45, 0.45] | not judged |

## Variant lodo_openacc_h3, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.74 [1.51, 1.97] | 1.82 [1.60, 2.05] | 0.40 [0.30, 0.47] | supported | 0.07 [0.02, 0.12] | 0.83 | 0.880 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_h3, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.79 [1.55, 2.07] | 1.89 [1.63, 2.17] | 0.47 [0.38, 0.52] | supported | [0.27, 0.55] | supported |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.59 [1.27, 1.88] | 1.59 [1.30, 1.88] | -0.51 [-0.88, -0.28] | refuted | [-0.66, 0.14] | inconclusive |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.74 [1.51, 1.97] | 1.82 [1.60, 2.05] | 0.40 [0.30, 0.47] | supported | [0.15, 0.52] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.23 [1.23, 1.23] | 1.23 [1.23, 1.23] | 0.52 [0.52, 0.52] | not judged | [0.52, 0.52] | not judged |

## Variant lodo_openacc_h5, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.72 [1.48, 1.96] | 1.81 [1.57, 2.06] | 0.40 [0.30, 0.48] | supported | 0.13 [0.09, 0.18] | 0.61 | 0.668 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_h5, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.77 [1.49, 2.06] | 1.88 [1.61, 2.15] | 0.47 [0.39, 0.53] | supported | [0.27, 0.55] | supported |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.56 [1.14, 1.97] | 1.56 [1.19, 1.96] | -0.48 [-0.85, -0.16] | refuted | [-0.58, 0.14] | inconclusive |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.72 [1.48, 1.96] | 1.81 [1.57, 2.06] | 0.40 [0.30, 0.48] | supported | [0.14, 0.53] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.17 [1.17, 1.17] | 1.17 [1.17, 1.17] | 0.55 [0.55, 0.55] | not judged | [0.55, 0.55] | not judged |

## Variant lodo_openacc_n025, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.72 [1.49, 1.98] | 1.81 [1.58, 2.07] | 0.40 [0.30, 0.47] | supported | 0.12 [0.08, 0.17] | 0.78 | 0.653 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_n025, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.77 [1.50, 2.07] | 1.88 [1.60, 2.17] | 0.47 [0.39, 0.53] | supported | [0.27, 0.55] | supported |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.59 [1.19, 1.99] | 1.59 [1.22, 1.99] | -0.51 [-0.83, -0.21] | refuted | [-0.60, 0.16] | inconclusive |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.72 [1.49, 1.98] | 1.81 [1.58, 2.07] | 0.40 [0.30, 0.47] | supported | [0.14, 0.52] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.18 [1.18, 1.18] | 1.18 [1.18, 1.18] | 0.54 [0.54, 0.54] | not judged | [0.54, 0.54] | not judged |

## Variant lodo_openacc_n05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.72 [1.49, 1.97] | 1.81 [1.58, 2.07] | 0.40 [0.30, 0.47] | supported | 0.10 [0.06, 0.15] | 0.83 | 0.677 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_n05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.76 [1.49, 2.05] | 1.87 [1.59, 2.15] | 0.47 [0.39, 0.53] | supported | [0.28, 0.56] | supported |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.65 [1.25, 2.07] | 1.65 [1.27, 2.06] | -0.57 [-0.91, -0.27] | refuted | [-0.66, 0.18] | inconclusive |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.72 [1.49, 1.97] | 1.81 [1.58, 2.07] | 0.40 [0.30, 0.47] | supported | [0.14, 0.53] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.04 [1.04, 1.04] | 1.04 [1.04, 1.04] | 0.59 [0.59, 0.59] | not judged | [0.59, 0.59] | not judged |

## Variant lodo_openacc_n05d05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.70 [1.50, 1.92] | 1.80 [1.60, 2.02] | 0.41 [0.29, 0.49] | supported | 0.11 [0.06, 0.16] | 0.83 | 0.911 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_n05d05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.68 [1.45, 1.94] | 1.80 [1.56, 2.04] | 0.49 [0.40, 0.55] | supported | [0.26, 0.59] | supported |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.84 [1.45, 2.25] | 1.84 [1.45, 2.24] | -0.75 [-1.18, -0.46] | refuted | [-0.87, 0.18] | inconclusive |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.70 [1.50, 1.92] | 1.80 [1.60, 2.02] | 0.41 [0.29, 0.49] | supported | [0.10, 0.55] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.07 [1.07, 1.07] | 1.07 [1.07, 1.07] | 0.58 [0.58, 0.58] | not judged | [0.58, 0.58] | not judged |

## Variant lodo_openacc_n1, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.74 [1.53, 1.97] | 1.84 [1.64, 2.06] | 0.39 [0.26, 0.48] | supported | 0.06 [0.02, 0.11] | 0.72 | 0.794 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_n1, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.68 [1.46, 1.94] | 1.81 [1.57, 2.06] | 0.49 [0.39, 0.55] | supported | [0.27, 0.58] | supported |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.05 [1.60, 2.51] | 2.05 [1.61, 2.51] | -0.95 [-1.37, -0.62] | refuted | [-1.08, 0.15] | inconclusive |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.74 [1.53, 1.97] | 1.84 [1.64, 2.06] | 0.39 [0.26, 0.48] | supported | [0.06, 0.54] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.83 [0.83, 0.83] | 0.83 [0.83, 0.83] | 0.68 [0.68, 0.68] | not judged | [0.68, 0.68] | not judged |

## Variant lodo_openacc_t1, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 3.86 [3.41, 4.35] | 3.81 [3.37, 4.34] | -0.26 [-0.70, 0.05] | refuted | -0.12 [-0.19, -0.05] | 0.39 | 0.998 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_t1, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 3.56 [3.12, 4.04] | 3.51 [3.08, 3.96] | 0.01 [-0.34, 0.25] | inconclusive | [-0.79, 0.33] | inconclusive |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 5.19 [4.03, 6.59] | 5.19 [3.98, 6.67] | -3.93 [-4.91, -3.13] | refuted | [-4.59, -1.10] | refuted |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 3.86 [3.41, 4.35] | 3.81 [3.37, 4.34] | -0.26 [-0.70, 0.05] | refuted | [-1.25, 0.21] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.73 [1.73, 1.73] | 1.73 [1.73, 1.73] | 0.33 [0.33, 0.33] | not judged | [0.33, 0.33] | not judged |

## Variant lodo_openacc_t2, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.39 [2.06, 2.73] | 2.46 [2.12, 2.84] | 0.19 [-0.14, 0.40] | inconclusive | -0.03 [-0.10, 0.04] | 0.56 | 0.974 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_t2, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 2.12 [1.83, 2.44] | 2.21 [1.88, 2.59] | 0.37 [0.13, 0.54] | inconclusive | [-0.19, 0.60] | inconclusive |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 3.57 [2.83, 4.37] | 3.57 [2.83, 4.36] | -2.39 [-3.14, -1.86] | refuted | [-2.65, -0.17] | refuted |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.39 [2.06, 2.73] | 2.46 [2.12, 2.84] | 0.19 [-0.14, 0.40] | inconclusive | [-0.53, 0.52] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.82 [0.82, 0.82] | 0.82 [0.82, 0.82] | 0.68 [0.68, 0.68] | not judged | [0.68, 0.68] | not judged |

## Variant lodo_openacc_t2_d05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.82 [2.42, 3.24] | 2.86 [2.45, 3.32] | 0.05 [-0.32, 0.30] | inconclusive | -0.06 [-0.14, 0.02] | 0.50 | 0.987 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_t2_d05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 2.55 [2.19, 2.98] | 2.60 [2.19, 3.06] | 0.26 [-0.03, 0.45] | inconclusive | [-0.40, 0.53] | inconclusive |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 4.04 [3.10, 5.15] | 4.04 [3.09, 5.16] | -2.83 [-3.60, -2.20] | refuted | [-3.35, -0.51] | refuted |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.82 [2.42, 3.24] | 2.86 [2.45, 3.32] | 0.05 [-0.32, 0.30] | inconclusive | [-0.75, 0.44] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.89 [0.89, 0.89] | 0.89 [0.89, 0.89] | 0.66 [0.66, 0.66] | not judged | [0.66, 0.66] | not judged |

## Variant lodo_openacc_t2_n05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 3.84 [3.41, 4.27] | 3.80 [3.36, 4.27] | -0.25 [-0.66, 0.02] | refuted | -0.27 [-0.33, -0.21] | 0.22 | 1.000 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_t2_n05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 3.65 [3.23, 4.10] | 3.59 [3.21, 4.07] | -0.02 [-0.33, 0.21] | inconclusive | [-0.73, 0.30] | inconclusive |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 4.55 [3.35, 5.87] | 4.55 [3.27, 5.98] | -3.32 [-4.25, -2.45] | refuted | [-4.34, -0.97] | refuted |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 3.84 [3.41, 4.27] | 3.80 [3.36, 4.27] | -0.25 [-0.66, 0.02] | refuted | [-1.16, 0.18] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 4.26 [4.26, 4.26] | 4.26 [4.26, 4.26] | -0.65 [-0.65, -0.65] | not judged | [-0.65, -0.65] | not judged |

## Variant pool_d05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.54 [1.32, 1.78] | 1.64 [1.43, 1.87] | 0.46 [0.29, 0.57] | supported | 0.18 [0.14, 0.23] | 0.83 | 0.941 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_d05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.36 [1.16, 1.63] | 1.49 [1.26, 1.73] | 0.58 [0.45, 0.66] | supported | [0.27, 0.70] | supported |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.30 [1.77, 2.82] | 2.30 [1.80, 2.84] | -1.18 [-1.75, -0.76] | refuted | [-1.56, 0.16] | inconclusive |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.54 [1.32, 1.78] | 1.64 [1.43, 1.87] | 0.46 [0.29, 0.57] | supported | [0.03, 0.66] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.72 [0.72, 0.72] | 0.72 [0.72, 0.72] | 0.72 [0.72, 0.72] | not judged | [0.72, 0.72] | not judged |

## Variant pool_d1, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.65 [1.40, 1.90] | 1.78 [1.54, 2.04] | 0.41 [0.21, 0.55] | supported | 0.15 [0.10, 0.21] | 0.83 | 0.968 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_d1, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.40 [1.21, 1.64] | 1.56 [1.35, 1.79] | 0.56 [0.42, 0.65] | supported | [0.21, 0.70] | supported |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.71 [2.08, 3.29] | 2.71 [2.12, 3.32] | -1.57 [-2.29, -1.09] | refuted | [-2.17, 0.22] | inconclusive |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.65 [1.40, 1.90] | 1.78 [1.54, 2.04] | 0.41 [0.21, 0.55] | supported | [-0.09, 0.65] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.64 [0.64, 0.64] | 0.64 [0.64, 0.64] | 0.75 [0.75, 0.75] | not judged | [0.75, 0.75] | not judged |

## Variant pool_h0, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.40 [2.08, 2.75] | 2.51 [2.19, 2.86] | 0.17 [-0.08, 0.34] | inconclusive | 0.09 [0.03, 0.15] | 0.67 | 0.992 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_h0, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 2.07 [1.77, 2.37] | 2.20 [1.92, 2.51] | 0.38 [0.21, 0.50] | supported | [0.01, 0.54] | inconclusive |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 3.82 [2.96, 4.71] | 3.82 [2.97, 4.73] | -2.63 [-3.42, -1.99] | refuted | [-2.97, -0.30] | refuted |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.40 [2.08, 2.75] | 2.51 [2.19, 2.86] | 0.17 [-0.08, 0.34] | inconclusive | [-0.44, 0.47] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.07 [1.07, 1.07] | 1.07 [1.07, 1.07] | 0.58 [0.58, 0.58] | not judged | [0.58, 0.58] | not judged |

## Variant pool_h05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.18 [1.87, 2.51] | 2.30 [2.03, 2.63] | 0.24 [0.02, 0.40] | inconclusive | 0.09 [0.03, 0.16] | 0.67 | 0.987 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_h05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.88 [1.62, 2.15] | 2.04 [1.78, 2.33] | 0.42 [0.28, 0.53] | supported | [0.10, 0.57] | inconclusive |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 3.45 [2.68, 4.21] | 3.45 [2.70, 4.21] | -2.27 [-3.00, -1.71] | refuted | [-2.43, -0.11] | refuted |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.18 [1.87, 2.51] | 2.30 [2.03, 2.63] | 0.24 [0.02, 0.40] | inconclusive | [-0.32, 0.51] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.89 [0.89, 0.89] | 0.89 [0.89, 0.89] | 0.65 [0.65, 0.65] | not judged | [0.65, 0.65] | not judged |

## Variant pool_h1, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.99 [1.73, 2.24] | 2.13 [1.88, 2.42] | 0.29 [0.08, 0.45] | inconclusive | 0.09 [0.03, 0.15] | 0.72 | 0.976 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_h1, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.73 [1.55, 1.93] | 1.91 [1.68, 2.18] | 0.46 [0.32, 0.56] | supported | [0.13, 0.60] | inconclusive |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 3.09 [2.38, 3.75] | 3.09 [2.42, 3.78] | -1.93 [-2.63, -1.43] | refuted | [-2.42, 0.12] | inconclusive |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.99 [1.73, 2.24] | 2.13 [1.88, 2.42] | 0.29 [0.08, 0.45] | inconclusive | [-0.23, 0.54] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.78 [0.78, 0.78] | 0.78 [0.78, 0.78] | 0.69 [0.69, 0.69] | not judged | [0.69, 0.69] | not judged |

## Variant pool_h2, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.72 [1.51, 1.92] | 1.81 [1.60, 2.04] | 0.40 [0.23, 0.52] | supported | 0.07 [0.02, 0.13] | 0.89 | 0.943 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_h2, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.57 [1.39, 1.80] | 1.69 [1.48, 1.92] | 0.52 [0.40, 0.60] | supported | [0.26, 0.63] | supported |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.38 [1.83, 2.85] | 2.38 [1.89, 2.89] | -1.25 [-1.97, -0.80] | refuted | [-1.82, 0.21] | inconclusive |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.72 [1.51, 1.92] | 1.81 [1.60, 2.04] | 0.40 [0.23, 0.52] | supported | [-0.00, 0.59] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.74 [0.74, 0.74] | 0.74 [0.74, 0.74] | 0.71 [0.71, 0.71] | not judged | [0.71, 0.71] | not judged |

## Variant pool_h2d05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.90 [1.68, 2.13] | 2.04 [1.79, 2.30] | 0.32 [0.10, 0.47] | inconclusive | 0.08 [0.02, 0.15] | 0.72 | 0.972 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_h2d05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.67 [1.50, 1.87] | 1.85 [1.63, 2.09] | 0.48 [0.33, 0.58] | supported | [0.13, 0.62] | inconclusive |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.91 [2.25, 3.52] | 2.91 [2.31, 3.55] | -1.76 [-2.50, -1.26] | refuted | [-2.29, 0.21] | inconclusive |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.90 [1.68, 2.13] | 2.04 [1.79, 2.30] | 0.32 [0.10, 0.47] | inconclusive | [-0.18, 0.56] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.72 [0.72, 0.72] | 0.72 [0.72, 0.72] | 0.72 [0.72, 0.72] | not judged | [0.72, 0.72] | not judged |

## Variant pool_h2n05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.84 [1.60, 2.07] | 1.94 [1.71, 2.20] | 0.36 [0.15, 0.49] | inconclusive | 0.06 [0.01, 0.12] | 0.83 | 0.974 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_h2n05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.58 [1.40, 1.78] | 1.71 [1.51, 1.94] | 0.51 [0.39, 0.60] | supported | [0.23, 0.63] | supported |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.96 [2.32, 3.53] | 2.96 [2.35, 3.56] | -1.81 [-2.48, -1.31] | refuted | [-2.19, 0.19] | inconclusive |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.84 [1.60, 2.07] | 1.94 [1.71, 2.20] | 0.36 [0.15, 0.49] | inconclusive | [-0.13, 0.58] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.52 [0.52, 0.52] | 0.52 [0.52, 0.52] | 0.80 [0.80, 0.80] | not judged | [0.80, 0.80] | not judged |

## Variant pool_h2n05d05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.02 [1.75, 2.30] | 2.16 [1.88, 2.47] | 0.29 [0.04, 0.45] | inconclusive | 0.07 [0.01, 0.14] | 0.67 | 0.988 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_h2n05d05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.70 [1.50, 1.91] | 1.87 [1.63, 2.15] | 0.47 [0.31, 0.58] | supported | [0.14, 0.61] | inconclusive |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 3.40 [2.68, 4.10] | 3.40 [2.71, 4.08] | -2.23 [-2.91, -1.70] | refuted | [-2.51, 0.11] | inconclusive |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.02 [1.75, 2.30] | 2.16 [1.88, 2.47] | 0.29 [0.04, 0.45] | inconclusive | [-0.28, 0.55] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.58 [0.58, 0.58] | 0.58 [0.58, 0.58] | 0.77 [0.77, 0.77] | not judged | [0.77, 0.77] | not judged |

## Variant pool_h3, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.63 [1.41, 1.85] | 1.70 [1.49, 1.93] | 0.44 [0.31, 0.54] | supported | 0.09 [0.05, 0.14] | 0.89 | 0.889 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_h3, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.54 [1.33, 1.79] | 1.63 [1.40, 1.90] | 0.54 [0.43, 0.61] | supported | [0.30, 0.64] | supported |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.03 [1.59, 2.45] | 2.03 [1.59, 2.47] | -0.93 [-1.55, -0.55] | refuted | [-1.40, 0.03] | refuted |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.63 [1.41, 1.85] | 1.70 [1.49, 1.93] | 0.44 [0.31, 0.54] | supported | [0.09, 0.60] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.77 [0.77, 0.77] | 0.77 [0.77, 0.77] | 0.70 [0.70, 0.70] | not judged | [0.70, 0.70] | not judged |

## Variant pool_h5, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.53 [1.32, 1.77] | 1.62 [1.41, 1.85] | 0.46 [0.32, 0.56] | supported | 0.21 [0.17, 0.25] | 0.78 | 0.802 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_h5, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.43 [1.20, 1.71] | 1.54 [1.31, 1.80] | 0.56 [0.45, 0.64] | supported | [0.29, 0.67] | supported |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.01 [1.48, 2.53] | 2.01 [1.50, 2.53] | -0.90 [-1.36, -0.48] | refuted | [-1.08, 0.15] | inconclusive |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.53 [1.32, 1.77] | 1.62 [1.41, 1.85] | 0.46 [0.32, 0.56] | supported | [0.10, 0.64] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.83 [0.83, 0.83] | 0.83 [0.83, 0.83] | 0.68 [0.68, 0.68] | not judged | [0.68, 0.68] | not judged |

## Variant pool_n025, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.54 [1.32, 1.77] | 1.63 [1.43, 1.86] | 0.46 [0.32, 0.56] | supported | 0.19 [0.15, 0.23] | 0.78 | 0.809 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_n025, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.41 [1.20, 1.67] | 1.53 [1.30, 1.77] | 0.57 [0.46, 0.64] | supported | [0.31, 0.68] | supported |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.12 [1.61, 2.62] | 2.12 [1.63, 2.63] | -1.01 [-1.50, -0.62] | refuted | [-1.24, 0.18] | inconclusive |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.54 [1.32, 1.77] | 1.63 [1.43, 1.86] | 0.46 [0.32, 0.56] | supported | [0.09, 0.64] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.74 [0.74, 0.74] | 0.74 [0.74, 0.74] | 0.71 [0.71, 0.71] | not judged | [0.71, 0.71] | not judged |

## Variant pool_n05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.58 [1.36, 1.80] | 1.68 [1.47, 1.92] | 0.44 [0.28, 0.56] | supported | 0.16 [0.12, 0.21] | 0.94 | 0.851 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_n05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.38 [1.19, 1.60] | 1.51 [1.31, 1.73] | 0.57 [0.47, 0.65] | supported | [0.31, 0.68] | supported |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.45 [1.90, 2.97] | 2.45 [1.91, 2.96] | -1.32 [-1.89, -0.90] | refuted | [-1.58, 0.20] | inconclusive |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.58 [1.36, 1.80] | 1.68 [1.47, 1.92] | 0.44 [0.28, 0.56] | supported | [0.02, 0.64] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.53 [0.53, 0.53] | 0.53 [0.53, 0.53] | 0.79 [0.79, 0.79] | not judged | [0.79, 0.79] | not judged |

## Variant pool_n05d05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.63 [1.40, 1.86] | 1.74 [1.52, 2.00] | 0.42 [0.24, 0.55] | supported | 0.14 [0.10, 0.20] | 0.89 | 0.955 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_n05d05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.36 [1.19, 1.55] | 1.50 [1.31, 1.71] | 0.58 [0.45, 0.66] | supported | [0.28, 0.70] | supported |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.81 [2.18, 3.34] | 2.81 [2.25, 3.37] | -1.66 [-2.39, -1.18] | refuted | [-2.13, 0.21] | inconclusive |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.63 [1.40, 1.86] | 1.74 [1.52, 2.00] | 0.42 [0.24, 0.55] | supported | [-0.06, 0.65] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.49 [0.49, 0.49] | 0.49 [0.49, 0.49] | 0.81 [0.81, 0.81] | not judged | [0.81, 0.81] | not judged |

## Variant pool_n1, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.84 [1.56, 2.12] | 1.96 [1.68, 2.27] | 0.35 [0.11, 0.51] | inconclusive | 0.10 [0.05, 0.15] | 0.78 | 0.932 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_n1, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.45 [1.27, 1.63] | 1.60 [1.40, 1.85] | 0.55 [0.41, 0.64] | supported | [0.23, 0.68] | supported |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 3.48 [2.80, 4.10] | 3.48 [2.82, 4.07] | -2.30 [-3.11, -1.74] | refuted | [-2.73, 0.09] | inconclusive |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.84 [1.56, 2.12] | 1.96 [1.68, 2.27] | 0.35 [0.11, 0.51] | inconclusive | [-0.23, 0.62] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.50 [0.50, 0.50] | 0.50 [0.50, 0.50] | 0.80 [0.80, 0.80] | not judged | [0.80, 0.80] | not judged |

## Variant pool_t1, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 3.86 [3.37, 4.36] | 3.83 [3.36, 4.38] | -0.27 [-0.72, 0.04] | refuted | -0.19 [-0.29, -0.10] | 0.33 | 1.000 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_t1, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 3.49 [3.05, 3.95] | 3.44 [3.01, 3.87] | 0.03 [-0.31, 0.26] | inconclusive | [-0.73, 0.34] | inconclusive |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 5.53 [4.27, 6.97] | 5.53 [4.19, 7.01] | -4.24 [-5.19, -3.40] | refuted | [-4.93, -1.05] | refuted |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 3.86 [3.37, 4.36] | 3.83 [3.36, 4.38] | -0.27 [-0.72, 0.04] | refuted | [-1.27, 0.22] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.54 [1.54, 1.54] | 1.54 [1.54, 1.54] | 0.40 [0.40, 0.40] | not judged | [0.40, 0.40] | not judged |

## Variant pool_t2, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.25 [1.92, 2.61] | 2.37 [2.00, 2.76] | 0.22 [-0.11, 0.43] | inconclusive | 0.08 [0.02, 0.14] | 0.83 | 0.989 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_t2, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.98 [1.69, 2.31] | 2.12 [1.75, 2.54] | 0.40 [0.14, 0.57] | inconclusive | [-0.20, 0.63] | inconclusive |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 3.48 [2.75, 4.18] | 3.48 [2.81, 4.17] | -2.30 [-3.06, -1.77] | refuted | [-2.80, -0.11] | refuted |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.25 [1.92, 2.61] | 2.37 [2.00, 2.76] | 0.22 [-0.11, 0.43] | inconclusive | [-0.51, 0.55] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.51 [0.51, 0.51] | 0.51 [0.51, 0.51] | 0.80 [0.80, 0.80] | not judged | [0.80, 0.80] | not judged |

## Variant pool_t2_d05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.84 [2.43, 3.29] | 2.94 [2.48, 3.42] | 0.03 [-0.38, 0.30] | inconclusive | 0.04 [-0.03, 0.12] | 0.67 | 0.999 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_t2_d05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 2.56 [2.17, 3.02] | 2.68 [2.20, 3.25] | 0.24 [-0.09, 0.46] | inconclusive | [-0.53, 0.54] | inconclusive |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 4.13 [3.30, 5.01] | 4.13 [3.34, 4.99] | -2.91 [-3.66, -2.36] | refuted | [-3.37, -0.52] | refuted |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.84 [2.43, 3.29] | 2.94 [2.48, 3.42] | 0.03 [-0.38, 0.30] | inconclusive | [-0.87, 0.44] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.52 [0.52, 0.52] | 0.52 [0.52, 0.52] | 0.80 [0.80, 0.80] | not judged | [0.80, 0.80] | not judged |

## Variant pool_t2_n05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 4.98 [4.43, 5.55] | 4.93 [4.35, 5.55] | -0.63 [-1.21, -0.22] | refuted | -0.25 [-0.30, -0.21] | 0.22 | 1.000 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_t2_n05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 4.71 [4.12, 5.40] | 4.66 [4.06, 5.33] | -0.32 [-0.80, 0.02] | refuted | [-1.45, 0.15] | inconclusive |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 5.90 [4.70, 7.15] | 5.90 [4.65, 7.22] | -4.59 [-5.84, -3.72] | refuted | [-5.21, -2.13] | refuted |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 4.98 [4.43, 5.55] | 4.93 [4.35, 5.55] | -0.63 [-1.21, -0.22] | refuted | [-1.93, -0.02] | refuted |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 6.23 [6.23, 6.23] | 6.23 [6.23, 6.23] | -1.43 [-1.43, -1.43] | not judged | [-1.43, -1.43] | not judged |

## Variant raw_d05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.37 [1.96, 2.80] | 2.56 [2.09, 3.04] | 0.14 [-0.17, 0.37] | inconclusive | 0.21 [0.17, 0.26] | 0.89 | 0.948 |
| idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | -0.02 [-0.06, 0.03] | 0.06 | 0.000 |

### H3.2 by driver mode: variant raw_d05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.38 [2.79, 4.04] | 3.47 [2.80, 4.11] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 2.45 [1.98, 2.96] | 2.70 [2.16, 3.36] | 0.22 [-0.10, 0.43] | inconclusive | [-0.57, 0.53] | inconclusive |
| Human | 12 | idm_white_noise | 1.11 [0.86, 1.40] | 1.11 [0.83, 1.41] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.17 [1.61, 2.72] | 2.17 [1.64, 2.71] | -0.96 [-1.47, -0.57] | refuted | [-1.20, -0.03] | refuted |
| all | 60 | idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.37 [1.96, 2.80] | 2.56 [2.09, 3.04] | 0.14 [-0.17, 0.37] | inconclusive | [-0.58, 0.47] | inconclusive |
| mixed | 1 | idm_white_noise | 2.51 [2.51, 2.51] | 2.51 [2.51, 2.51] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.67 [0.67, 0.67] | 0.67 [0.67, 0.67] | 0.73 [0.73, 0.73] | not judged | [0.73, 0.73] | not judged |

## Variant raw_d1, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.60 [2.16, 3.09] | 2.83 [2.30, 3.37] | 0.05 [-0.29, 0.32] | inconclusive | 0.20 [0.16, 0.25] | 0.89 | 0.963 |
| idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | -0.02 [-0.06, 0.03] | 0.06 | 0.000 |

### H3.2 by driver mode: variant raw_d1, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.38 [2.79, 4.04] | 3.47 [2.80, 4.11] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 2.68 [2.16, 3.23] | 2.97 [2.35, 3.71] | 0.14 [-0.22, 0.39] | inconclusive | [-0.75, 0.49] | inconclusive |
| Human | 12 | idm_white_noise | 1.11 [0.86, 1.40] | 1.11 [0.83, 1.41] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.45 [1.83, 3.08] | 2.45 [1.87, 3.05] | -1.21 [-1.83, -0.80] | refuted | [-1.55, -0.06] | refuted |
| all | 60 | idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.60 [2.16, 3.09] | 2.83 [2.30, 3.37] | 0.05 [-0.29, 0.32] | inconclusive | [-0.78, 0.43] | inconclusive |
| mixed | 1 | idm_white_noise | 2.51 [2.51, 2.51] | 2.51 [2.51, 2.51] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.60 [0.60, 0.60] | 0.60 [0.60, 0.60] | 0.76 [0.76, 0.76] | not judged | [0.76, 0.76] | not judged |

## Variant raw_h0, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.90 [2.46, 3.34] | 2.97 [2.55, 3.39] | 0.00 [-0.27, 0.21] | inconclusive | 0.08 [0.03, 0.15] | 0.67 | 0.988 |
| idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | -0.02 [-0.06, 0.03] | 0.06 | 0.000 |

### H3.2 by driver mode: variant raw_h0, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.38 [2.79, 4.04] | 3.47 [2.80, 4.11] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 2.45 [2.08, 2.84] | 2.54 [2.18, 2.92] | 0.27 [0.11, 0.39] | inconclusive | [0.00, 0.41] | inconclusive |
| Human | 12 | idm_white_noise | 1.11 [0.86, 1.40] | 1.11 [0.83, 1.41] | reference | - | - | - |
| Human | 12 | diffusion_cf | 4.79 [3.80, 5.63] | 4.79 [3.86, 5.61] | -3.32 [-4.64, -2.55] | refuted | [-4.67, -0.08] | refuted |
| all | 60 | idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.90 [2.46, 3.34] | 2.97 [2.55, 3.39] | 0.00 [-0.27, 0.21] | inconclusive | [-0.66, 0.34] | inconclusive |
| mixed | 1 | idm_white_noise | 2.51 [2.51, 2.51] | 2.51 [2.51, 2.51] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.05 [1.05, 1.05] | 1.05 [1.05, 1.05] | 0.58 [0.58, 0.58] | not judged | [0.58, 0.58] | not judged |

## Variant raw_h05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.55 [2.19, 2.98] | 2.68 [2.31, 3.07] | 0.10 [-0.14, 0.29] | inconclusive | 0.12 [0.06, 0.19] | 0.72 | 0.986 |
| idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | -0.02 [-0.06, 0.03] | 0.06 | 0.000 |

### H3.2 by driver mode: variant raw_h05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.38 [2.79, 4.04] | 3.47 [2.80, 4.11] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 2.17 [1.85, 2.50] | 2.33 [2.01, 2.69] | 0.33 [0.18, 0.45] | inconclusive | [0.06, 0.46] | inconclusive |
| Human | 12 | idm_white_noise | 1.11 [0.86, 1.40] | 1.11 [0.83, 1.41] | reference | - | - | - |
| Human | 12 | diffusion_cf | 4.21 [3.30, 5.04] | 4.21 [3.36, 5.00] | -2.80 [-4.12, -2.07] | refuted | [-4.18, -0.19] | refuted |
| all | 60 | idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.55 [2.19, 2.98] | 2.68 [2.31, 3.07] | 0.10 [-0.14, 0.29] | inconclusive | [-0.51, 0.40] | inconclusive |
| mixed | 1 | idm_white_noise | 2.51 [2.51, 2.51] | 2.51 [2.51, 2.51] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.74 [0.74, 0.74] | 0.74 [0.74, 0.74] | 0.70 [0.70, 0.70] | not judged | [0.70, 0.70] | not judged |

## Variant raw_h1, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.49 [2.17, 2.83] | 2.62 [2.28, 3.00] | 0.12 [-0.16, 0.31] | inconclusive | 0.11 [0.05, 0.17] | 0.61 | 0.981 |
| idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | -0.02 [-0.06, 0.03] | 0.06 | 0.000 |

### H3.2 by driver mode: variant raw_h1, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.38 [2.79, 4.04] | 3.47 [2.80, 4.11] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 2.23 [1.97, 2.54] | 2.40 [2.03, 2.82] | 0.31 [0.10, 0.46] | inconclusive | [-0.14, 0.49] | inconclusive |
| Human | 12 | idm_white_noise | 1.11 [0.86, 1.40] | 1.11 [0.83, 1.41] | reference | - | - | - |
| Human | 12 | diffusion_cf | 3.66 [2.78, 4.43] | 3.66 [2.91, 4.42] | -2.30 [-3.54, -1.61] | refuted | [-3.71, 0.01] | refuted |
| all | 60 | idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.49 [2.17, 2.83] | 2.62 [2.28, 3.00] | 0.12 [-0.16, 0.31] | inconclusive | [-0.52, 0.42] | inconclusive |
| mixed | 1 | idm_white_noise | 2.51 [2.51, 2.51] | 2.51 [2.51, 2.51] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.60 [0.60, 0.60] | 0.60 [0.60, 0.60] | 0.76 [0.76, 0.76] | not judged | [0.76, 0.76] | not judged |

## Variant raw_h2, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.31 [2.04, 2.60] | 2.30 [2.04, 2.58] | 0.23 [-0.01, 0.39] | inconclusive | 0.04 [-0.02, 0.10] | 0.78 | 0.959 |
| idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | -0.02 [-0.06, 0.03] | 0.06 | 0.000 |

### H3.2 by driver mode: variant raw_h2, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.38 [2.79, 4.04] | 3.47 [2.80, 4.11] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 2.29 [1.99, 2.61] | 2.28 [1.97, 2.60] | 0.34 [0.15, 0.49] | inconclusive | [-0.17, 0.55] | inconclusive |
| Human | 12 | idm_white_noise | 1.11 [0.86, 1.40] | 1.11 [0.83, 1.41] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.54 [2.02, 3.01] | 2.54 [2.08, 3.00] | -1.29 [-2.18, -0.85] | refuted | [-2.14, -0.15] | refuted |
| all | 60 | idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.31 [2.04, 2.60] | 2.30 [2.04, 2.58] | 0.23 [-0.01, 0.39] | inconclusive | [-0.33, 0.49] | inconclusive |
| mixed | 1 | idm_white_noise | 2.51 [2.51, 2.51] | 2.51 [2.51, 2.51] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.63 [0.63, 0.63] | 0.63 [0.63, 0.63] | 0.75 [0.75, 0.75] | not judged | [0.75, 0.75] | not judged |

## Variant raw_h3, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.08 [1.79, 2.41] | 2.12 [1.82, 2.40] | 0.29 [0.10, 0.43] | inconclusive | 0.06 [0.02, 0.11] | 0.67 | 0.910 |
| idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | -0.02 [-0.06, 0.03] | 0.06 | 0.000 |

### H3.2 by driver mode: variant raw_h3, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.38 [2.79, 4.04] | 3.47 [2.80, 4.11] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 2.16 [1.84, 2.54] | 2.21 [1.88, 2.57] | 0.36 [0.18, 0.49] | inconclusive | [-0.09, 0.55] | inconclusive |
| Human | 12 | idm_white_noise | 1.11 [0.86, 1.40] | 1.11 [0.83, 1.41] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.88 [1.44, 2.30] | 1.88 [1.49, 2.29] | -0.70 [-1.26, -0.38] | refuted | [-1.05, -0.14] | refuted |
| all | 60 | idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.08 [1.79, 2.41] | 2.12 [1.82, 2.40] | 0.29 [0.10, 0.43] | inconclusive | [-0.17, 0.50] | inconclusive |
| mixed | 1 | idm_white_noise | 2.51 [2.51, 2.51] | 2.51 [2.51, 2.51] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.69 [0.69, 0.69] | 0.69 [0.69, 0.69] | 0.73 [0.73, 0.73] | not judged | [0.73, 0.73] | not judged |

## Variant raw_n025, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.00 [1.69, 2.39] | 2.15 [1.79, 2.52] | 0.28 [0.05, 0.45] | inconclusive | 0.21 [0.17, 0.26] | 0.89 | 0.888 |
| idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | -0.02 [-0.06, 0.03] | 0.06 | 0.000 |

### H3.2 by driver mode: variant raw_n025, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.38 [2.79, 4.04] | 3.47 [2.80, 4.11] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 2.03 [1.65, 2.48] | 2.22 [1.81, 2.71] | 0.36 [0.13, 0.51] | inconclusive | [-0.20, 0.58] | inconclusive |
| Human | 12 | idm_white_noise | 1.11 [0.86, 1.40] | 1.11 [0.83, 1.41] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.97 [1.44, 2.50] | 1.97 [1.47, 2.49] | -0.78 [-1.18, -0.44] | refuted | [-0.90, 0.03] | refuted |
| all | 60 | idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.00 [1.69, 2.39] | 2.15 [1.79, 2.52] | 0.28 [0.05, 0.45] | inconclusive | [-0.25, 0.53] | inconclusive |
| mixed | 1 | idm_white_noise | 2.51 [2.51, 2.51] | 2.51 [2.51, 2.51] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.79 [0.79, 0.79] | 0.79 [0.79, 0.79] | 0.69 [0.69, 0.69] | not judged | [0.69, 0.69] | not judged |

## Variant raw_n05, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.91 [1.63, 2.26] | 2.04 [1.73, 2.36] | 0.31 [0.11, 0.46] | inconclusive | 0.19 [0.15, 0.23] | 0.94 | 0.863 |
| idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | -0.02 [-0.06, 0.03] | 0.06 | 0.000 |

### H3.2 by driver mode: variant raw_n05, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.38 [2.79, 4.04] | 3.47 [2.80, 4.11] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.93 [1.58, 2.33] | 2.09 [1.74, 2.51] | 0.40 [0.20, 0.53] | inconclusive | [-0.08, 0.59] | inconclusive |
| Human | 12 | idm_white_noise | 1.11 [0.86, 1.40] | 1.11 [0.83, 1.41] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.97 [1.46, 2.47] | 1.97 [1.48, 2.46] | -0.77 [-1.18, -0.44] | refuted | [-0.93, 0.05] | refuted |
| all | 60 | idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.91 [1.63, 2.26] | 2.04 [1.73, 2.36] | 0.31 [0.11, 0.46] | inconclusive | [-0.16, 0.54] | inconclusive |
| mixed | 1 | idm_white_noise | 2.51 [2.51, 2.51] | 2.51 [2.51, 2.51] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.71 [0.71, 0.71] | 0.71 [0.71, 0.71] | 0.72 [0.72, 0.72] | not judged | [0.72, 0.72] | not judged |

## Variant raw_n1, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.82 [1.58, 2.09] | 1.92 [1.67, 2.17] | 0.35 [0.17, 0.48] | inconclusive | 0.13 [0.09, 0.17] | 0.89 | 0.850 |
| idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | -0.02 [-0.06, 0.03] | 0.06 | 0.000 |

### H3.2 by driver mode: variant raw_n1, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.38 [2.79, 4.04] | 3.47 [2.80, 4.11] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.73 [1.48, 2.06] | 1.87 [1.59, 2.17] | 0.46 [0.31, 0.56] | supported | [0.09, 0.61] | inconclusive |
| Human | 12 | idm_white_noise | 1.11 [0.86, 1.40] | 1.11 [0.83, 1.41] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.24 [1.78, 2.73] | 2.24 [1.76, 2.70] | -1.02 [-1.54, -0.69] | refuted | [-1.29, 0.02] | refuted |
| all | 60 | idm_white_noise | 2.91 [2.38, 3.49] | 2.98 [2.41, 3.57] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.82 [1.58, 2.09] | 1.92 [1.67, 2.17] | 0.35 [0.17, 0.48] | inconclusive | [-0.07, 0.56] | inconclusive |
| mixed | 1 | idm_white_noise | 2.51 [2.51, 2.51] | 2.51 [2.51, 2.51] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.45 [0.45, 0.45] | 0.45 [0.45, 0.45] | 0.82 [0.82, 0.82] | not judged | [0.82, 0.82] | not judged |

## Growth exponent of the measured platoons

* empirical (headline): beta 0.08 [-0.01, 0.18] fitted over indices 1..4 (measured by >= 50% of the profiles); curve to index 4 (profile window)
* empirical (extended): beta 0.12 [0.08, 0.16] fitted over indices 1..4 (measured by >= 50% of the profiles); curve to index 8 (profile window)
