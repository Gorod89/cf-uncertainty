# E3 summary: ext_e3_idmref

Window profile + tail. CRPS 1-4 / CRPS all: amplitude CRPS in m/s, mean over the vehicle indices 1-4 / over every measured index of a profile, then over profiles, [95 % bootstrap interval over profiles]. H3.2: reduction 1 - CRPS_model / CRPS_reference against idm_white_noise_acc, paired by profile, with its verdict (docs/design.md 13.3, 14.1; 'not judged': fewer than 5 profiles); files CI / verdict files: the paired bootstrap over the recording files (all profiles of a drawn file kept). beta: exponent of sd_i = sd_1 i^beta; d2<0: share of negative second differences of the sd curve; collided: share of rollouts with a collision of any pair of the platoon.

## Variant raw (signals as recorded), set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise_acc | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| idm_white_noise_acc | 2.22 [1.85, 2.61] | 2.27 [1.88, 2.69] | reference | - | -0.00 [-0.03, 0.03] | 0.22 | 0.000 |
| idm_white_noise_human | 2.10 [1.79, 2.41] | 2.15 [1.84, 2.49] | 0.05 [-0.03, 0.10] | inconclusive | 0.01 [-0.01, 0.03] | 0.44 | 0.000 |

### H3.2 by driver mode: variant raw, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise_acc | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise_acc | 2.47 [2.04, 2.93] | 2.53 [2.06, 3.02] | reference | - | - | - |
| ACC | 47 | idm_white_noise_human | 2.17 [1.82, 2.55] | 2.24 [1.87, 2.61] | 0.12 [0.06, 0.15] | inconclusive | [0.02, 0.16] | inconclusive |
| Human | 12 | idm_white_noise_acc | 1.34 [0.92, 1.77] | 1.34 [0.97, 1.76] | reference | - | - | - |
| Human | 12 | idm_white_noise_human | 1.95 [1.46, 2.44] | 1.95 [1.50, 2.41] | -0.46 [-0.66, -0.34] | refuted | [-0.61, 0.13] | inconclusive |
| all | 60 | idm_white_noise_acc | 2.22 [1.85, 2.61] | 2.27 [1.88, 2.69] | reference | - | - | - |
| all | 60 | idm_white_noise_human | 2.10 [1.79, 2.41] | 2.15 [1.84, 2.49] | 0.05 [-0.03, 0.10] | inconclusive | [-0.10, 0.14] | inconclusive |
| mixed | 1 | idm_white_noise_acc | 0.89 [0.89, 0.89] | 0.89 [0.89, 0.89] | reference | - | - | - |
| mixed | 1 | idm_white_noise_human | 0.51 [0.51, 0.51] | 0.51 [0.51, 0.51] | 0.43 [0.43, 0.43] | not judged | [0.43, 0.43] | not judged |

## Variant harmonised (harmonised signals), set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise_acc | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| idm_white_noise_acc | 2.22 [1.85, 2.61] | 2.27 [1.88, 2.69] | reference | - | -0.00 [-0.03, 0.03] | 0.22 | 0.000 |
| idm_white_noise_human | 2.10 [1.79, 2.41] | 2.15 [1.84, 2.49] | 0.05 [-0.03, 0.10] | inconclusive | 0.01 [-0.01, 0.03] | 0.44 | 0.000 |

### H3.2 by driver mode: variant harmonised, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise_acc | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise_acc | 2.47 [2.04, 2.93] | 2.53 [2.06, 3.02] | reference | - | - | - |
| ACC | 47 | idm_white_noise_human | 2.17 [1.82, 2.55] | 2.24 [1.87, 2.61] | 0.12 [0.06, 0.15] | inconclusive | [0.02, 0.16] | inconclusive |
| Human | 12 | idm_white_noise_acc | 1.34 [0.92, 1.77] | 1.34 [0.97, 1.76] | reference | - | - | - |
| Human | 12 | idm_white_noise_human | 1.95 [1.46, 2.44] | 1.95 [1.50, 2.41] | -0.46 [-0.66, -0.34] | refuted | [-0.61, 0.13] | inconclusive |
| all | 60 | idm_white_noise_acc | 2.22 [1.85, 2.61] | 2.27 [1.88, 2.69] | reference | - | - | - |
| all | 60 | idm_white_noise_human | 2.10 [1.79, 2.41] | 2.15 [1.84, 2.49] | 0.05 [-0.03, 0.10] | inconclusive | [-0.10, 0.14] | inconclusive |
| mixed | 1 | idm_white_noise_acc | 0.89 [0.89, 0.89] | 0.89 [0.89, 0.89] | reference | - | - | - |
| mixed | 1 | idm_white_noise_human | 0.51 [0.51, 0.51] | 0.51 [0.51, 0.51] | 0.43 [0.43, 0.43] | not judged | [0.43, 0.43] | not judged |

## Variant pool (pooled domain, OpenACC seen), set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise_acc | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| idm_white_noise_acc | 2.23 [1.85, 2.63] | 2.28 [1.88, 2.71] | reference | - | -0.00 [-0.02, 0.03] | 0.22 | 0.000 |
| idm_white_noise_human | 2.09 [1.78, 2.42] | 2.15 [1.83, 2.50] | 0.05 [-0.02, 0.11] | inconclusive | 0.01 [-0.01, 0.04] | 0.44 | 0.000 |

### H3.2 by driver mode: variant pool, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise_acc | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise_acc | 2.48 [2.05, 2.94] | 2.55 [2.06, 3.03] | reference | - | - | - |
| ACC | 47 | idm_white_noise_human | 2.17 [1.81, 2.56] | 2.24 [1.87, 2.62] | 0.12 [0.07, 0.16] | inconclusive | [0.03, 0.16] | inconclusive |
| Human | 12 | idm_white_noise_acc | 1.34 [0.92, 1.77] | 1.34 [0.97, 1.76] | reference | - | - | - |
| Human | 12 | idm_white_noise_human | 1.95 [1.46, 2.44] | 1.95 [1.50, 2.41] | -0.46 [-0.66, -0.34] | refuted | [-0.61, 0.14] | inconclusive |
| all | 60 | idm_white_noise_acc | 2.23 [1.85, 2.63] | 2.28 [1.88, 2.71] | reference | - | - | - |
| all | 60 | idm_white_noise_human | 2.09 [1.78, 2.42] | 2.15 [1.83, 2.50] | 0.05 [-0.02, 0.11] | inconclusive | [-0.10, 0.14] | inconclusive |
| mixed | 1 | idm_white_noise_acc | 0.86 [0.86, 0.86] | 0.86 [0.86, 0.86] | reference | - | - | - |
| mixed | 1 | idm_white_noise_human | 0.49 [0.49, 0.49] | 0.49 [0.49, 0.49] | 0.43 [0.43, 0.43] | not judged | [0.43, 0.43] | not judged |

## Variant lodo_openacc (pooled domain, OpenACC held out), set extended (60 profiles), window profile + tail

| model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise_acc | verdict | beta | d2<0 | collided |
|---|---:|---:|---:|---:|---:|---:|---:|
| idm_white_noise_acc | 2.23 [1.85, 2.63] | 2.28 [1.88, 2.71] | reference | - | -0.00 [-0.02, 0.03] | 0.22 | 0.000 |
| idm_white_noise_human | 2.09 [1.78, 2.42] | 2.15 [1.83, 2.50] | 0.05 [-0.02, 0.11] | inconclusive | 0.01 [-0.01, 0.04] | 0.44 | 0.000 |

### H3.2 by driver mode: variant lodo_openacc, set extended, window profile + tail (a verdict needs >= 5 profiles; 'all' = pooled, secondary)

| mode | profiles | model | CRPS 1-4 | CRPS all | H3.2 vs idm_white_noise_acc | verdict | files CI | verdict files |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACC | 47 | idm_white_noise_acc | 2.48 [2.05, 2.94] | 2.55 [2.06, 3.03] | reference | - | - | - |
| ACC | 47 | idm_white_noise_human | 2.17 [1.81, 2.56] | 2.24 [1.87, 2.62] | 0.12 [0.07, 0.16] | inconclusive | [0.03, 0.16] | inconclusive |
| Human | 12 | idm_white_noise_acc | 1.34 [0.92, 1.77] | 1.34 [0.97, 1.76] | reference | - | - | - |
| Human | 12 | idm_white_noise_human | 1.95 [1.46, 2.44] | 1.95 [1.50, 2.41] | -0.46 [-0.66, -0.34] | refuted | [-0.61, 0.14] | inconclusive |
| all | 60 | idm_white_noise_acc | 2.23 [1.85, 2.63] | 2.28 [1.88, 2.71] | reference | - | - | - |
| all | 60 | idm_white_noise_human | 2.09 [1.78, 2.42] | 2.15 [1.83, 2.50] | 0.05 [-0.02, 0.11] | inconclusive | [-0.10, 0.14] | inconclusive |
| mixed | 1 | idm_white_noise_acc | 0.86 [0.86, 0.86] | 0.86 [0.86, 0.86] | reference | - | - | - |
| mixed | 1 | idm_white_noise_human | 0.49 [0.49, 0.49] | 0.49 [0.49, 0.49] | 0.43 [0.43, 0.43] | not judged | [0.43, 0.43] | not judged |

## Growth exponent of the measured platoons

* empirical (headline): beta 0.08 [-0.01, 0.18] fitted over indices 1..4 (measured by >= 50% of the profiles); curve to index 4 (profile window)
* empirical (extended): beta 0.12 [0.08, 0.16] fitted over indices 1..4 (measured by >= 50% of the profiles); curve to index 8 (profile window)
