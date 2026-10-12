# E3 summary: ext_e3_replan

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

## Variant lodo_openacc_r10, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.61 [1.40, 1.83] | 1.69 [1.47, 1.94] | 0.44 [0.32, 0.53] | supported | 0.14 [0.10, 0.19] | 0.67 | 0.509 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_r10, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.62 [1.38, 1.89] | 1.73 [1.47, 1.99] | 0.51 [0.41, 0.59] | supported | [0.26, 0.62] | supported |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.58 [1.15, 2.03] | 1.58 [1.18, 2.02] | -0.50 [-0.85, -0.17] | refuted | [-0.57, -0.01] | refuted |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.61 [1.40, 1.83] | 1.69 [1.47, 1.94] | 0.44 [0.32, 0.53] | supported | [0.14, 0.59] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.41 [1.41, 1.41] | 1.41 [1.41, 1.41] | 0.45 [0.45, 0.45] | not judged | [0.45, 0.45] | not judged |

## Variant lodo_openacc_r10_share, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.62 [1.40, 1.85] | 1.70 [1.48, 1.95] | 0.44 [0.32, 0.52] | supported | 0.14 [0.10, 0.19] | 0.78 | 0.480 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_r10_share, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.65 [1.40, 1.93] | 1.76 [1.49, 2.02] | 0.50 [0.40, 0.58] | supported | [0.26, 0.61] | supported |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.51 [1.09, 1.94] | 1.51 [1.13, 1.94] | -0.44 [-0.76, -0.12] | refuted | [-0.50, -0.03] | refuted |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.62 [1.40, 1.85] | 1.70 [1.48, 1.95] | 0.44 [0.32, 0.52] | supported | [0.16, 0.58] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.37 [1.37, 1.37] | 1.37 [1.37, 1.37] | 0.47 [0.47, 0.47] | not judged | [0.47, 0.47] | not judged |

## Variant lodo_openacc_r20, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.71 [1.47, 1.95] | 1.79 [1.55, 2.04] | 0.41 [0.30, 0.49] | supported | 0.15 [0.11, 0.19] | 0.61 | 0.433 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_r20, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.75 [1.48, 2.04] | 1.85 [1.58, 2.15] | 0.48 [0.38, 0.54] | supported | [0.27, 0.57] | supported |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.55 [1.15, 1.97] | 1.55 [1.19, 1.97] | -0.47 [-0.82, -0.15] | refuted | [-0.52, -0.02] | refuted |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.71 [1.47, 1.95] | 1.79 [1.55, 2.04] | 0.41 [0.30, 0.49] | supported | [0.15, 0.54] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.36 [1.36, 1.36] | 1.36 [1.36, 1.36] | 0.47 [0.47, 0.47] | not judged | [0.47, 0.47] | not judged |

## Variant lodo_openacc_r20_share, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.71 [1.47, 1.96] | 1.79 [1.55, 2.04] | 0.41 [0.30, 0.48] | supported | 0.15 [0.11, 0.19] | 0.72 | 0.434 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_r20_share, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.77 [1.48, 2.06] | 1.87 [1.58, 2.17] | 0.47 [0.38, 0.54] | supported | [0.27, 0.56] | supported |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.53 [1.13, 1.94] | 1.53 [1.17, 1.93] | -0.45 [-0.79, -0.13] | refuted | [-0.50, -0.04] | refuted |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.71 [1.47, 1.96] | 1.79 [1.55, 2.04] | 0.41 [0.30, 0.48] | supported | [0.16, 0.53] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.36 [1.36, 1.36] | 1.36 [1.36, 1.36] | 0.47 [0.47, 0.47] | not judged | [0.47, 0.47] | not judged |

## Variant lodo_openacc_r50, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.72 [1.48, 1.96] | 1.81 [1.57, 2.06] | 0.40 [0.30, 0.48] | supported | 0.13 [0.09, 0.18] | 0.61 | 0.668 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_r50, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

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

## Variant lodo_openacc_r50_share, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.72 [1.48, 1.96] | 1.81 [1.57, 2.06] | 0.40 [0.30, 0.48] | supported | 0.13 [0.09, 0.18] | 0.61 | 0.668 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_r50_share, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

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

## Variant lodo_openacc_s10, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 5.02 [4.50, 5.63] | 4.88 [4.37, 5.45] | -0.61 [-1.14, -0.24] | refuted | -0.12 [-0.12, -0.11] | 0.22 | 0.999 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_s10, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 5.38 [4.74, 6.05] | 5.20 [4.65, 5.86] | -0.47 [-0.96, -0.13] | refuted | [-1.71, -0.02] | refuted |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 3.80 [2.97, 4.67] | 3.80 [3.02, 4.68] | -2.60 [-3.62, -1.88] | refuted | [-5.92, -1.61] | refuted |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 5.02 [4.50, 5.63] | 4.88 [4.37, 5.45] | -0.61 [-1.14, -0.24] | refuted | [-1.74, -0.11] | refuted |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 2.86 [2.86, 2.86] | 2.86 [2.86, 2.86] | -0.11 [-0.11, -0.11] | not judged | [-0.11, -0.11] | not judged |

## Variant lodo_openacc_s100, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.78 [1.52, 2.05] | 1.87 [1.61, 2.16] | 0.38 [0.29, 0.45] | supported | 0.13 [0.09, 0.17] | 0.72 | 0.686 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_s100, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.86 [1.56, 2.18] | 1.97 [1.66, 2.29] | 0.44 [0.37, 0.50] | supported | [0.26, 0.52] | supported |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.54 [1.13, 1.96] | 1.54 [1.16, 1.96] | -0.46 [-0.74, -0.16] | refuted | [-0.57, 0.11] | inconclusive |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.78 [1.52, 2.05] | 1.87 [1.61, 2.16] | 0.38 [0.29, 0.45] | supported | [0.14, 0.50] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.14 [1.14, 1.14] | 1.14 [1.14, 1.14] | 0.56 [0.56, 0.56] | not judged | [0.56, 0.56] | not judged |

## Variant lodo_openacc_s25, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.60 [2.25, 2.96] | 2.72 [2.37, 3.12] | 0.10 [-0.22, 0.32] | inconclusive | -0.12 [-0.18, -0.07] | 0.56 | 0.951 |
| idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc_s25, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.82, 4.15] | 3.54 [2.83, 4.22] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 2.24 [1.94, 2.55] | 2.39 [2.07, 2.74] | 0.32 [0.09, 0.48] | inconclusive | [-0.19, 0.55] | inconclusive |
| Human | 12 | idm_white_noise | 1.05 [0.85, 1.28] | 1.05 [0.82, 1.29] | reference | - | - | - |
| Human | 12 | diffusion_cf | 4.08 [3.08, 5.12] | 4.08 [3.10, 5.12] | -2.87 [-3.56, -2.21] | refuted | [-3.27, 0.03] | refuted |
| all | 60 | idm_white_noise | 2.97 [2.41, 3.57] | 3.03 [2.45, 3.65] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.60 [2.25, 2.96] | 2.72 [2.37, 3.12] | 0.10 [-0.22, 0.32] | inconclusive | [-0.63, 0.46] | inconclusive |
| mixed | 1 | idm_white_noise | 2.57 [2.57, 2.57] | 2.57 [2.57, 2.57] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.82 [1.82, 1.82] | 1.82 [1.82, 1.82] | 0.29 [0.29, 0.29] | not judged | [0.29, 0.29] | not judged |

## Variant pool_r10, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.52 [1.30, 1.77] | 1.61 [1.38, 1.86] | 0.47 [0.33, 0.57] | supported | 0.22 [0.18, 0.27] | 0.72 | 0.752 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_r10, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.46 [1.21, 1.76] | 1.57 [1.31, 1.85] | 0.56 [0.43, 0.64] | supported | [0.26, 0.68] | supported |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.82 [1.34, 2.34] | 1.82 [1.35, 2.34] | -0.73 [-1.15, -0.34] | refuted | [-0.89, -0.10] | refuted |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.52 [1.30, 1.77] | 1.61 [1.38, 1.86] | 0.47 [0.33, 0.57] | supported | [0.11, 0.64] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.98 [0.98, 0.98] | 0.98 [0.98, 0.98] | 0.62 [0.62, 0.62] | not judged | [0.62, 0.62] | not judged |

## Variant pool_r10_share, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.53 [1.31, 1.79] | 1.62 [1.38, 1.88] | 0.46 [0.33, 0.57] | supported | 0.22 [0.18, 0.27] | 0.72 | 0.727 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_r10_share, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.48 [1.23, 1.80] | 1.59 [1.33, 1.88] | 0.55 [0.43, 0.64] | supported | [0.26, 0.67] | supported |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.78 [1.31, 2.28] | 1.78 [1.31, 2.29] | -0.69 [-1.10, -0.31] | refuted | [-0.84, -0.12] | refuted |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.53 [1.31, 1.79] | 1.62 [1.38, 1.88] | 0.46 [0.33, 0.57] | supported | [0.11, 0.63] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.02 [1.02, 1.02] | 1.02 [1.02, 1.02] | 0.60 [0.60, 0.60] | not judged | [0.60, 0.60] | not judged |

## Variant pool_r20, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.58 [1.36, 1.84] | 1.67 [1.43, 1.93] | 0.45 [0.32, 0.54] | supported | 0.21 [0.18, 0.25] | 0.78 | 0.614 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_r20, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.56 [1.31, 1.85] | 1.67 [1.40, 1.94] | 0.53 [0.42, 0.60] | supported | [0.28, 0.64] | supported |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.73 [1.26, 2.20] | 1.73 [1.28, 2.23] | -0.64 [-1.07, -0.26] | refuted | [-0.83, -0.07] | refuted |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.58 [1.36, 1.84] | 1.67 [1.43, 1.93] | 0.45 [0.32, 0.54] | supported | [0.13, 0.60] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.93 [0.93, 0.93] | 0.93 [0.93, 0.93] | 0.64 [0.64, 0.64] | not judged | [0.64, 0.64] | not judged |

## Variant pool_r20_share, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.59 [1.36, 1.84] | 1.67 [1.44, 1.93] | 0.45 [0.32, 0.54] | supported | 0.21 [0.18, 0.25] | 0.67 | 0.607 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_r20_share, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.57 [1.32, 1.86] | 1.67 [1.40, 1.95] | 0.53 [0.42, 0.60] | supported | [0.28, 0.63] | supported |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 1.72 [1.26, 2.19] | 1.72 [1.28, 2.22] | -0.63 [-1.06, -0.25] | refuted | [-0.82, -0.08] | refuted |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.59 [1.36, 1.84] | 1.67 [1.44, 1.93] | 0.45 [0.32, 0.54] | supported | [0.13, 0.60] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.93 [0.93, 0.93] | 0.93 [0.93, 0.93] | 0.64 [0.64, 0.64] | not judged | [0.64, 0.64] | not judged |

## Variant pool_r50, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.53 [1.32, 1.77] | 1.62 [1.41, 1.85] | 0.46 [0.32, 0.56] | supported | 0.21 [0.17, 0.25] | 0.78 | 0.802 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_r50, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

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

## Variant pool_r50_share, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.53 [1.32, 1.77] | 1.62 [1.41, 1.85] | 0.46 [0.32, 0.56] | supported | 0.21 [0.17, 0.25] | 0.78 | 0.802 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_r50_share, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

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

## Variant pool_s10, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 4.61 [4.15, 5.14] | 4.50 [4.03, 5.00] | -0.49 [-0.96, -0.16] | refuted | -0.14 [-0.15, -0.13] | 0.39 | 0.999 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_s10, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 4.94 [4.39, 5.49] | 4.80 [4.31, 5.35] | -0.36 [-0.78, -0.07] | refuted | [-1.40, 0.03] | refuted |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 3.44 [2.65, 4.27] | 3.44 [2.65, 4.30] | -2.26 [-3.06, -1.67] | refuted | [-3.95, -1.32] | refuted |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 4.61 [4.15, 5.14] | 4.50 [4.03, 5.00] | -0.49 [-0.96, -0.16] | refuted | [-1.45, -0.05] | refuted |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 3.46 [3.46, 3.46] | 3.46 [3.46, 3.46] | -0.35 [-0.35, -0.35] | not judged | [-0.35, -0.35] | not judged |

## Variant pool_s100, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 1.58 [1.36, 1.81] | 1.66 [1.45, 1.90] | 0.45 [0.31, 0.54] | supported | 0.20 [0.17, 0.24] | 0.78 | 0.809 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_s100, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 1.47 [1.23, 1.75] | 1.58 [1.33, 1.83] | 0.55 [0.45, 0.63] | supported | [0.30, 0.66] | supported |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 2.05 [1.55, 2.58] | 2.05 [1.55, 2.58] | -0.95 [-1.42, -0.54] | refuted | [-1.19, 0.09] | inconclusive |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 1.58 [1.36, 1.81] | 1.66 [1.45, 1.90] | 0.45 [0.31, 0.54] | supported | [0.09, 0.62] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 0.86 [0.86, 0.86] | 0.86 [0.86, 0.86] | 0.66 [0.66, 0.66] | not judged | [0.66, 0.66] | not judged |

## Variant pool_s25, set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion_cf | 2.69 [2.29, 3.12] | 2.83 [2.43, 3.30] | 0.06 [-0.28, 0.31] | inconclusive | -0.06 [-0.12, -0.01] | 0.72 | 0.954 |
| idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | -0.04 [-0.07, 0.00] | 0.00 | 0.000 |

### H3.2 by driver mode: variant pool_s25, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise | 3.46 [2.81, 4.14] | 3.53 [2.83, 4.21] | reference | - | - | - |
| ACC | 47 | diffusion_cf | 2.24 [1.93, 2.59] | 2.42 [2.06, 2.79] | 0.31 [0.07, 0.48] | inconclusive | [-0.21, 0.55] | inconclusive |
| Human | 12 | idm_white_noise | 1.06 [0.85, 1.29] | 1.06 [0.82, 1.30] | reference | - | - | - |
| Human | 12 | diffusion_cf | 4.55 [3.42, 5.61] | 4.55 [3.50, 5.58] | -3.31 [-4.21, -2.59] | refuted | [-3.82, 0.10] | inconclusive |
| all | 60 | idm_white_noise | 2.96 [2.41, 3.56] | 3.02 [2.44, 3.64] | reference | - | - | - |
| all | 60 | diffusion_cf | 2.69 [2.29, 3.12] | 2.83 [2.43, 3.30] | 0.06 [-0.28, 0.31] | inconclusive | [-0.72, 0.45] | inconclusive |
| mixed | 1 | idm_white_noise | 2.56 [2.56, 2.56] | 2.56 [2.56, 2.56] | reference | - | - | - |
| mixed | 1 | diffusion_cf | 1.36 [1.36, 1.36] | 1.36 [1.36, 1.36] | 0.47 [0.47, 0.47] | not judged | [0.47, 0.47] | not judged |

## Growth exponent of the measured platoons

* empirical (headline): beta 0.08 [-0.01, 0.18] fitted over indices 1..4 (measured by >= 50% of the profiles); curve to index 4 (profile window)
* empirical (extended): beta 0.12 [0.08, 0.16] fitted over indices 1..4 (measured by >= 50% of the profiles); curve to index 8 (profile window)
