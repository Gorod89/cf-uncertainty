# String stability of the simulated platoons (extension S9)

Geometric mean of the consecutive amplitude ratios A(i+1) / A(i) of the median sampled amplitude over the 20 followers (cluster bootstrap over recording files, 95 %), the first-follower and head-to-tail ratios to the leader, and the recorded ratios on the recorded indices. Reading rule (PLAN.md S9): string-stable in the mean when the interval lies below 1, amplifying when it lies above 1.

## e3 / harmonised

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| deep_ensemble | all | 60 | 0.961 [0.935, 0.994] | string-stable | 0.16 | 6.40 | 3.01 | 1.09 [1.05, 1.13] | 0.83 |
| deep_ensemble | ACC | 47 | 0.974 [0.943, 1.004] | inconclusive | 0.21 | 5.48 | 3.34 | 1.11 [1.07, 1.15] | 0.90 |
| diffusion_cf | all | 60 | 1.033 [1.027, 1.042] | amplifying | 0.74 | 1.24 | 2.30 | 1.09 [1.05, 1.13] | 1.10 |
| diffusion_cf | ACC | 47 | 1.035 [1.027, 1.045] | amplifying | 0.79 | 1.26 | 2.41 | 1.11 [1.07, 1.15] | 1.11 |
| gp_residual_idm | all | 60 | 1.010 [1.004, 1.016] | amplifying | 0.79 | 1.01 | 1.22 | 1.09 [1.05, 1.13] | 1.03 |
| gp_residual_idm | ACC | 47 | 1.015 [1.010, 1.021] | amplifying | 0.89 | 0.95 | 1.26 | 1.11 [1.07, 1.15] | 1.01 |
| idm_white_noise | all | 60 | 1.003 [0.997, 1.010] | inconclusive | 1.00 | 1.14 | 1.20 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.005 [0.998, 1.014] | inconclusive | 1.00 | 1.13 | 1.25 | 1.11 [1.07, 1.15] | 1.01 |
| lstm_const_sigma | all | 60 | 0.999 [0.990, 1.009] | inconclusive | 0.37 | 1.85 | 1.79 | 1.09 [1.05, 1.13] | 0.99 |
| lstm_const_sigma | ACC | 47 | 1.001 [0.990, 1.015] | inconclusive | 0.42 | 1.87 | 1.92 | 1.11 [1.07, 1.15] | 0.99 |
| quantile_pidl | all | 60 | 0.989 [0.985, 0.995] | string-stable | 0.16 | 1.86 | 1.53 | 1.09 [1.05, 1.13] | 1.02 |
| quantile_pidl | ACC | 47 | 0.991 [0.986, 0.997] | string-stable | 0.16 | 1.95 | 1.64 | 1.11 [1.07, 1.15] | 1.03 |

## e3 / lodo_openacc

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| deep_ensemble | all | 60 | 0.982 [0.975, 0.992] | string-stable | 0.37 | 2.55 | 1.83 | 1.09 [1.05, 1.13] | 0.94 |
| deep_ensemble | ACC | 47 | 0.987 [0.979, 0.996] | string-stable | 0.47 | 2.50 | 1.94 | 1.11 [1.07, 1.15] | 0.97 |
| diffusion_cf | all | 60 | 1.030 [1.024, 1.037] | amplifying | 1.00 | 1.14 | 2.01 | 1.09 [1.05, 1.13] | 1.07 |
| diffusion_cf | ACC | 47 | 1.033 [1.026, 1.041] | amplifying | 1.00 | 1.14 | 2.10 | 1.11 [1.07, 1.15] | 1.07 |
| diffusion_cf_hist | all | 60 | 1.016 [1.007, 1.028] | amplifying | 0.53 | 1.96 | 2.66 | 1.09 [1.05, 1.13] | 1.10 |
| diffusion_cf_hist | ACC | 47 | 1.019 [1.009, 1.033] | amplifying | 0.58 | 1.99 | 2.87 | 1.11 [1.07, 1.15] | 1.12 |
| gp_residual_idm | all | 60 | 0.989 [0.979, 1.001] | inconclusive | 0.21 | 1.05 | 0.85 | 1.09 [1.05, 1.13] | 1.05 |
| gp_residual_idm | ACC | 47 | 0.995 [0.984, 1.009] | inconclusive | 0.26 | 1.01 | 0.93 | 1.11 [1.07, 1.15] | 1.04 |
| idm_ar_noise | all | 60 | 0.960 [0.951, 0.971] | string-stable | 0.05 | 1.68 | 0.78 | 1.09 [1.05, 1.13] | 1.00 |
| idm_ar_noise | ACC | 47 | 0.961 [0.949, 0.972] | string-stable | 0.05 | 1.74 | 0.81 | 1.11 [1.07, 1.15] | 1.00 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.74 | 1.10 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.996, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |
| lstm_const_sigma | all | 60 | 1.008 [0.997, 1.021] | inconclusive | 0.68 | 4.31 | 5.00 | 1.09 [1.05, 1.13] | 0.91 |
| lstm_const_sigma | ACC | 47 | 1.009 [0.998, 1.022] | inconclusive | 0.74 | 4.79 | 5.64 | 1.11 [1.07, 1.15] | 0.92 |
| quantile_pidl | all | 60 | 1.016 [1.008, 1.027] | amplifying | 0.79 | 1.62 | 2.20 | 1.09 [1.05, 1.13] | 1.08 |
| quantile_pidl | ACC | 47 | 1.019 [1.010, 1.031] | amplifying | 0.84 | 1.63 | 2.35 | 1.11 [1.07, 1.15] | 1.08 |

## e3 / lodo_openacc_w11

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| deep_ensemble | all | 60 | 0.983 [0.974, 0.994] | string-stable | 0.37 | 2.41 | 1.74 | 1.09 [1.05, 1.13] | 0.96 |
| deep_ensemble | ACC | 47 | 0.988 [0.979, 0.999] | string-stable | 0.42 | 2.30 | 1.83 | 1.11 [1.07, 1.15] | 0.99 |
| diffusion_cf | all | 60 | 1.030 [1.024, 1.037] | amplifying | 1.00 | 1.11 | 1.95 | 1.09 [1.05, 1.13] | 1.07 |
| diffusion_cf | ACC | 47 | 1.033 [1.025, 1.041] | amplifying | 1.00 | 1.10 | 2.03 | 1.11 [1.07, 1.15] | 1.07 |
| gp_residual_idm | all | 60 | 0.990 [0.980, 1.002] | inconclusive | 0.26 | 1.02 | 0.84 | 1.09 [1.05, 1.13] | 1.05 |
| gp_residual_idm | ACC | 47 | 0.996 [0.984, 1.010] | inconclusive | 0.26 | 0.99 | 0.92 | 1.11 [1.07, 1.15] | 1.04 |
| idm_ar_noise | all | 60 | 0.960 [0.951, 0.970] | string-stable | 0.05 | 1.64 | 0.75 | 1.09 [1.05, 1.13] | 1.00 |
| idm_ar_noise | ACC | 47 | 0.960 [0.949, 0.972] | string-stable | 0.05 | 1.68 | 0.78 | 1.11 [1.07, 1.15] | 1.00 |
| idm_white_noise | all | 60 | 1.000 [0.995, 1.007] | inconclusive | 0.68 | 1.10 | 1.11 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.996, 1.011] | inconclusive | 1.00 | 1.10 | 1.16 | 1.11 [1.07, 1.15] | 1.00 |
| lstm_const_sigma | all | 60 | 1.007 [0.995, 1.019] | inconclusive | 0.68 | 4.21 | 4.77 | 1.09 [1.05, 1.13] | 0.91 |
| lstm_const_sigma | ACC | 47 | 1.007 [0.997, 1.020] | inconclusive | 0.63 | 4.64 | 5.33 | 1.11 [1.07, 1.15] | 0.92 |
| quantile_pidl | all | 60 | 1.013 [1.007, 1.022] | amplifying | 0.79 | 1.66 | 2.13 | 1.09 [1.05, 1.13] | 1.07 |
| quantile_pidl | ACC | 47 | 1.016 [1.008, 1.025] | amplifying | 0.79 | 1.67 | 2.26 | 1.11 [1.07, 1.15] | 1.07 |

## e3 / pool

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| deep_ensemble | all | 60 | 0.986 [0.977, 0.999] | string-stable | 0.26 | 1.91 | 1.48 | 1.09 [1.05, 1.13] | 0.99 |
| deep_ensemble | ACC | 47 | 0.991 [0.979, 1.003] | inconclusive | 0.26 | 1.90 | 1.59 | 1.11 [1.07, 1.15] | 1.01 |
| diffusion_cf | all | 60 | 1.039 [1.033, 1.047] | amplifying | 0.84 | 1.14 | 2.38 | 1.09 [1.05, 1.13] | 1.10 |
| diffusion_cf | ACC | 47 | 1.041 [1.034, 1.050] | amplifying | 0.95 | 1.14 | 2.45 | 1.11 [1.07, 1.15] | 1.10 |
| diffusion_cf_hist | all | 60 | 1.013 [1.005, 1.025] | amplifying | 0.74 | 1.88 | 2.42 | 1.09 [1.05, 1.13] | 1.06 |
| diffusion_cf_hist | ACC | 47 | 1.016 [1.006, 1.029] | amplifying | 0.79 | 1.94 | 2.62 | 1.11 [1.07, 1.15] | 1.08 |
| gp_residual_idm | all | 60 | 1.008 [1.004, 1.013] | amplifying | 0.58 | 1.07 | 1.25 | 1.09 [1.05, 1.13] | 1.05 |
| gp_residual_idm | ACC | 47 | 1.008 [1.004, 1.015] | amplifying | 0.79 | 1.04 | 1.22 | 1.11 [1.07, 1.15] | 1.03 |
| idm_ar_noise | all | 60 | 0.962 [0.953, 0.972] | string-stable | 0.05 | 1.66 | 0.79 | 1.09 [1.05, 1.13] | 1.00 |
| idm_ar_noise | ACC | 47 | 0.962 [0.951, 0.974] | string-stable | 0.05 | 1.72 | 0.82 | 1.11 [1.07, 1.15] | 1.00 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.95 | 1.11 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.997, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |
| lstm_const_sigma | all | 60 | 0.991 [0.970, 1.018] | inconclusive | 0.37 | 2.22 | 1.86 | 1.09 [1.05, 1.13] | 0.99 |
| lstm_const_sigma | ACC | 47 | 0.996 [0.969, 1.026] | inconclusive | 0.37 | 2.18 | 2.02 | 1.11 [1.07, 1.15] | 1.00 |
| quantile_pidl | all | 60 | 1.017 [1.009, 1.027] | amplifying | 0.74 | 1.64 | 2.24 | 1.09 [1.05, 1.13] | 1.09 |
| quantile_pidl | ACC | 47 | 1.020 [1.011, 1.032] | amplifying | 0.74 | 1.65 | 2.42 | 1.11 [1.07, 1.15] | 1.10 |

## e3 / pool_w11

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| deep_ensemble | all | 60 | 0.987 [0.977, 0.999] | string-stable | 0.32 | 1.84 | 1.43 | 1.09 [1.05, 1.13] | 0.99 |
| deep_ensemble | ACC | 47 | 0.991 [0.980, 1.004] | inconclusive | 0.21 | 1.82 | 1.53 | 1.11 [1.07, 1.15] | 1.02 |
| diffusion_cf | all | 60 | 1.038 [1.032, 1.044] | amplifying | 0.84 | 1.15 | 2.31 | 1.09 [1.05, 1.13] | 1.10 |
| diffusion_cf | ACC | 47 | 1.039 [1.032, 1.047] | amplifying | 0.95 | 1.15 | 2.37 | 1.11 [1.07, 1.15] | 1.10 |
| gp_residual_idm | all | 60 | 1.009 [1.005, 1.015] | amplifying | 0.58 | 1.05 | 1.25 | 1.09 [1.05, 1.13] | 1.05 |
| gp_residual_idm | ACC | 47 | 1.010 [1.005, 1.017] | amplifying | 0.84 | 1.02 | 1.22 | 1.11 [1.07, 1.15] | 1.03 |
| idm_ar_noise | all | 60 | 0.962 [0.952, 0.972] | string-stable | 0.05 | 1.61 | 0.77 | 1.09 [1.05, 1.13] | 1.00 |
| idm_ar_noise | ACC | 47 | 0.962 [0.951, 0.973] | string-stable | 0.05 | 1.66 | 0.79 | 1.11 [1.07, 1.15] | 1.00 |
| idm_white_noise | all | 60 | 1.000 [0.995, 1.007] | inconclusive | 0.84 | 1.10 | 1.11 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.996, 1.011] | inconclusive | 1.00 | 1.10 | 1.16 | 1.11 [1.07, 1.15] | 1.00 |
| lstm_const_sigma | all | 60 | 0.991 [0.972, 1.017] | inconclusive | 0.21 | 2.11 | 1.78 | 1.09 [1.05, 1.13] | 1.00 |
| lstm_const_sigma | ACC | 47 | 0.996 [0.972, 1.024] | inconclusive | 0.26 | 2.05 | 1.91 | 1.11 [1.07, 1.15] | 1.01 |
| quantile_pidl | all | 60 | 1.014 [1.007, 1.023] | amplifying | 0.74 | 1.66 | 2.17 | 1.09 [1.05, 1.13] | 1.08 |
| quantile_pidl | ACC | 47 | 1.017 [1.009, 1.028] | amplifying | 0.79 | 1.68 | 2.32 | 1.11 [1.07, 1.15] | 1.09 |

## e3 / raw

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| deep_ensemble | all | 60 | 0.981 [0.965, 0.995] | string-stable | 0.32 | 2.78 | 1.92 | 1.09 [1.05, 1.13] | 0.93 |
| deep_ensemble | ACC | 47 | 0.989 [0.980, 1.000] | inconclusive | 0.42 | 2.62 | 2.12 | 1.11 [1.07, 1.15] | 0.97 |
| diffusion_cf | all | 60 | 1.040 [1.034, 1.048] | amplifying | 0.89 | 1.26 | 2.66 | 1.09 [1.05, 1.13] | 1.11 |
| diffusion_cf | ACC | 47 | 1.042 [1.035, 1.051] | amplifying | 0.95 | 1.28 | 2.79 | 1.11 [1.07, 1.15] | 1.12 |
| diffusion_cf_hist | all | 60 | 0.966 [0.957, 0.978] | string-stable | 0.26 | 4.11 | 2.11 | 1.09 [1.05, 1.13] | 0.85 |
| diffusion_cf_hist | ACC | 47 | 0.969 [0.958, 0.983] | string-stable | 0.21 | 4.08 | 2.23 | 1.11 [1.07, 1.15] | 0.86 |
| gp_residual_idm | all | 60 | 1.006 [0.999, 1.013] | inconclusive | 0.84 | 1.00 | 1.11 | 1.09 [1.05, 1.13] | 1.02 |
| gp_residual_idm | ACC | 47 | 1.011 [1.005, 1.018] | amplifying | 0.84 | 0.95 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |
| idm_ar_noise | all | 60 | 0.946 [0.936, 0.956] | string-stable | 0.05 | 1.72 | 0.59 | 1.09 [1.05, 1.13] | 0.97 |
| idm_ar_noise | ACC | 47 | 0.946 [0.934, 0.958] | string-stable | 0.05 | 1.77 | 0.61 | 1.11 [1.07, 1.15] | 0.98 |
| idm_white_noise | all | 60 | 1.003 [0.997, 1.011] | inconclusive | 1.00 | 1.13 | 1.19 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.006 [0.998, 1.015] | inconclusive | 1.00 | 1.12 | 1.25 | 1.11 [1.07, 1.15] | 1.01 |
| lstm_const_sigma | all | 60 | 0.958 [0.931, 0.989] | string-stable | 0.32 | 5.69 | 2.50 | 1.09 [1.05, 1.13] | 0.81 |
| lstm_const_sigma | ACC | 47 | 0.968 [0.932, 1.003] | inconclusive | 0.32 | 5.35 | 2.86 | 1.11 [1.07, 1.15] | 0.86 |
| quantile_pidl | all | 60 | 1.004 [0.999, 1.011] | inconclusive | 0.26 | 1.78 | 1.93 | 1.09 [1.05, 1.13] | 1.08 |
| quantile_pidl | ACC | 47 | 1.007 [1.001, 1.014] | amplifying | 0.26 | 1.75 | 1.98 | 1.11 [1.07, 1.15] | 1.09 |

## e3 / raw_a

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| idm_white_noise | all | 60 | 1.003 [0.997, 1.011] | inconclusive | 1.00 | 1.13 | 1.19 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.006 [0.998, 1.015] | inconclusive | 1.00 | 1.12 | 1.25 | 1.11 [1.07, 1.15] | 1.01 |
| quantile_pidl | all | 60 | 1.015 [1.006, 1.030] | amplifying | 0.95 | 1.61 | 2.15 | 1.09 [1.05, 1.13] | 1.03 |
| quantile_pidl | ACC | 47 | 1.020 [1.007, 1.037] | amplifying | 1.00 | 1.55 | 2.28 | 1.11 [1.07, 1.15] | 1.05 |

## e3 / raw_w61

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| deep_ensemble | all | 60 | 0.983 [0.967, 0.997] | string-stable | 0.26 | 2.82 | 2.02 | 1.09 [1.05, 1.13] | 0.93 |
| deep_ensemble | ACC | 47 | 0.991 [0.983, 1.003] | inconclusive | 0.42 | 2.65 | 2.24 | 1.11 [1.07, 1.15] | 0.98 |
| diffusion_cf | all | 60 | 1.043 [1.037, 1.052] | amplifying | 0.95 | 1.22 | 2.73 | 1.09 [1.05, 1.13] | 1.11 |
| diffusion_cf | ACC | 47 | 1.046 [1.038, 1.056] | amplifying | 0.95 | 1.23 | 2.88 | 1.11 [1.07, 1.15] | 1.12 |
| gp_residual_idm | all | 60 | 1.006 [0.999, 1.013] | inconclusive | 0.74 | 1.03 | 1.15 | 1.09 [1.05, 1.13] | 1.03 |
| gp_residual_idm | ACC | 47 | 1.011 [1.005, 1.018] | amplifying | 0.84 | 0.98 | 1.21 | 1.11 [1.07, 1.15] | 1.01 |
| idm_ar_noise | all | 60 | 0.946 [0.936, 0.956] | string-stable | 0.05 | 1.77 | 0.61 | 1.09 [1.05, 1.13] | 0.97 |
| idm_ar_noise | ACC | 47 | 0.946 [0.934, 0.958] | string-stable | 0.05 | 1.83 | 0.64 | 1.11 [1.07, 1.15] | 0.98 |
| idm_white_noise | all | 60 | 1.003 [0.998, 1.011] | inconclusive | 1.00 | 1.13 | 1.20 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.006 [0.999, 1.016] | inconclusive | 1.00 | 1.13 | 1.27 | 1.11 [1.07, 1.15] | 1.01 |
| lstm_const_sigma | all | 60 | 0.957 [0.931, 0.989] | string-stable | 0.32 | 5.89 | 2.58 | 1.09 [1.05, 1.13] | 0.81 |
| lstm_const_sigma | ACC | 47 | 0.967 [0.932, 1.002] | inconclusive | 0.37 | 5.58 | 2.98 | 1.11 [1.07, 1.15] | 0.84 |
| quantile_pidl | all | 60 | 1.016 [1.006, 1.030] | amplifying | 1.00 | 1.65 | 2.22 | 1.09 [1.05, 1.13] | 1.03 |
| quantile_pidl | ACC | 47 | 1.021 [1.008, 1.037] | amplifying | 1.00 | 1.61 | 2.37 | 1.11 [1.07, 1.15] | 1.05 |

## preview / lodo_openacc_h0

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.020 [1.010, 1.032] | amplifying | 0.42 | 1.17 | 1.71 | 1.09 [1.05, 1.13] | 1.11 |
| diffusion_cf | ACC | 47 | 1.025 [1.013, 1.039] | amplifying | 0.42 | 1.13 | 1.81 | 1.11 [1.07, 1.15] | 1.12 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.74 | 1.10 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.996, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## preview / lodo_openacc_h1

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.019 [1.010, 1.030] | amplifying | 0.42 | 1.17 | 1.68 | 1.09 [1.05, 1.13] | 1.12 |
| diffusion_cf | ACC | 47 | 1.023 [1.012, 1.035] | amplifying | 0.42 | 1.15 | 1.77 | 1.11 [1.07, 1.15] | 1.13 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.74 | 1.10 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.996, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## preview / lodo_openacc_h2

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.017 [1.009, 1.024] | amplifying | 0.53 | 1.23 | 1.69 | 1.09 [1.05, 1.13] | 1.07 |
| diffusion_cf | ACC | 47 | 1.020 [1.011, 1.029] | amplifying | 0.58 | 1.24 | 1.79 | 1.11 [1.07, 1.15] | 1.08 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.74 | 1.10 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.996, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## preview / lodo_openacc_h5

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.030 [1.024, 1.037] | amplifying | 1.00 | 1.14 | 2.01 | 1.09 [1.05, 1.13] | 1.07 |
| diffusion_cf | ACC | 47 | 1.033 [1.026, 1.041] | amplifying | 1.00 | 1.14 | 2.10 | 1.11 [1.07, 1.15] | 1.07 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.74 | 1.10 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.996, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## preview / lodo_openacc_t2

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.007 [0.998, 1.018] | inconclusive | 0.32 | 1.44 | 1.65 | 1.09 [1.05, 1.13] | 1.08 |
| diffusion_cf | ACC | 47 | 1.009 [0.998, 1.020] | inconclusive | 0.32 | 1.50 | 1.76 | 1.11 [1.07, 1.15] | 1.08 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.74 | 1.10 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.996, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## preview / pool_h0

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.023 [1.012, 1.036] | amplifying | 0.47 | 1.15 | 1.75 | 1.09 [1.05, 1.13] | 1.12 |
| diffusion_cf | ACC | 47 | 1.028 [1.015, 1.045] | amplifying | 0.47 | 1.09 | 1.86 | 1.11 [1.07, 1.15] | 1.13 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.95 | 1.11 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.997, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## preview / pool_h1

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.021 [1.011, 1.032] | amplifying | 0.42 | 1.15 | 1.71 | 1.09 [1.05, 1.13] | 1.13 |
| diffusion_cf | ACC | 47 | 1.025 [1.014, 1.038] | amplifying | 0.47 | 1.12 | 1.80 | 1.11 [1.07, 1.15] | 1.14 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.95 | 1.11 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.997, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## preview / pool_h2

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.017 [1.009, 1.025] | amplifying | 0.58 | 1.23 | 1.69 | 1.09 [1.05, 1.13] | 1.08 |
| diffusion_cf | ACC | 47 | 1.020 [1.011, 1.029] | amplifying | 0.63 | 1.24 | 1.80 | 1.11 [1.07, 1.15] | 1.09 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.95 | 1.11 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.997, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## preview / pool_h5

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.039 [1.033, 1.047] | amplifying | 0.84 | 1.14 | 2.38 | 1.09 [1.05, 1.13] | 1.10 |
| diffusion_cf | ACC | 47 | 1.041 [1.034, 1.050] | amplifying | 0.95 | 1.14 | 2.45 | 1.11 [1.07, 1.15] | 1.10 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.95 | 1.11 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.997, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## preview / pool_t2

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.017 [1.008, 1.029] | amplifying | 0.42 | 1.43 | 1.98 | 1.09 [1.05, 1.13] | 1.11 |
| diffusion_cf | ACC | 47 | 1.019 [1.008, 1.032] | amplifying | 0.58 | 1.48 | 2.12 | 1.11 [1.07, 1.15] | 1.11 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.95 | 1.11 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.997, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## preview / raw

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.040 [1.034, 1.048] | amplifying | 0.89 | 1.26 | 2.66 | 1.09 [1.05, 1.13] | 1.11 |
| diffusion_cf | ACC | 47 | 1.042 [1.035, 1.051] | amplifying | 0.95 | 1.28 | 2.79 | 1.11 [1.07, 1.15] | 1.12 |
| idm_white_noise | all | 60 | 1.003 [0.997, 1.011] | inconclusive | 1.00 | 1.13 | 1.19 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.006 [0.998, 1.015] | inconclusive | 1.00 | 1.12 | 1.25 | 1.11 [1.07, 1.15] | 1.01 |

## preview / raw_h0

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.025 [1.014, 1.037] | amplifying | 0.84 | 1.13 | 1.79 | 1.09 [1.05, 1.13] | 1.13 |
| diffusion_cf | ACC | 47 | 1.030 [1.017, 1.045] | amplifying | 0.89 | 1.08 | 1.90 | 1.11 [1.07, 1.15] | 1.13 |
| idm_white_noise | all | 60 | 1.003 [0.997, 1.011] | inconclusive | 1.00 | 1.13 | 1.19 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.006 [0.998, 1.015] | inconclusive | 1.00 | 1.12 | 1.25 | 1.11 [1.07, 1.15] | 1.01 |

## preview / raw_h2

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.007 [1.000, 1.014] | amplifying | 0.26 | 1.41 | 1.61 | 1.09 [1.05, 1.13] | 1.06 |
| diffusion_cf | ACC | 47 | 1.008 [1.000, 1.016] | inconclusive | 0.37 | 1.47 | 1.71 | 1.11 [1.07, 1.15] | 1.06 |
| idm_white_noise | all | 60 | 1.003 [0.997, 1.011] | inconclusive | 1.00 | 1.13 | 1.19 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.006 [0.998, 1.015] | inconclusive | 1.00 | 1.12 | 1.25 | 1.11 [1.07, 1.15] | 1.01 |

## ngsim / raw

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| deep_ensemble | all | 60 | 0.991 [0.935, 1.039] | inconclusive | 0.37 | 7.80 | 0.28 | 1.09 [1.05, 1.13] | 0.95 |
| deep_ensemble | ACC | 47 | 0.971 [0.926, 0.997] | string-stable | 0.32 | 29.77 | 0.33 | 1.11 [1.07, 1.15] | 0.83 |
| diffusion_cf | all | 60 | 1.018 [1.014, 1.023] | amplifying | 1.00 | 1.40 | 1.97 | 1.09 [1.05, 1.13] | 1.05 |
| diffusion_cf | ACC | 47 | 1.020 [1.015, 1.027] | amplifying | 1.00 | 1.39 | 2.03 | 1.11 [1.07, 1.15] | 1.05 |
| gp_residual_idm | all | 60 | 1.011 [1.005, 1.020] | amplifying | 0.37 | 1.15 | 1.42 | 1.09 [1.05, 1.13] | 1.07 |
| gp_residual_idm | ACC | 47 | 1.014 [1.007, 1.024] | amplifying | 0.37 | 1.16 | 1.52 | 1.11 [1.07, 1.15] | 1.08 |
| idm_ar_noise | all | 60 | 0.939 [0.936, 0.942] | string-stable | 0.05 | 3.34 | 1.01 | 1.09 [1.05, 1.13] | 0.96 |
| idm_ar_noise | ACC | 47 | 0.940 [0.937, 0.943] | string-stable | 0.05 | 3.47 | 1.08 | 1.11 [1.07, 1.15] | 0.96 |
| idm_white_noise | all | 60 | 1.009 [1.003, 1.016] | amplifying | 0.63 | 1.22 | 1.44 | 1.09 [1.05, 1.13] | 0.98 |
| idm_white_noise | ACC | 47 | 1.010 [1.002, 1.019] | amplifying | 0.63 | 1.22 | 1.46 | 1.11 [1.07, 1.15] | 0.98 |
| lstm_const_sigma | all | 60 | 0.980 [0.943, 1.012] | inconclusive | 0.21 | 16.03 | 0.23 | 1.09 [1.05, 1.13] | 0.97 |
| lstm_const_sigma | ACC | 47 | 0.967 [0.929, 0.992] | string-stable | 0.16 | 16.73 | 0.28 | 1.11 [1.07, 1.15] | 0.90 |
| quantile_pidl | all | 60 | 0.845 [0.832, 0.860] | string-stable | 0.05 | 2.07 | 0.08 | 1.09 [1.05, 1.13] | 0.81 |
| quantile_pidl | ACC | 47 | 0.847 [0.831, 0.864] | string-stable | 0.05 | 2.17 | 0.09 | 1.11 [1.07, 1.15] | 0.82 |

## ngsim / w61

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| deep_ensemble | all | 60 | 0.987 [0.928, 1.038] | inconclusive | 0.37 | 8.36 | 0.29 | 1.09 [1.05, 1.13] | 0.92 |
| deep_ensemble | ACC | 47 | 0.967 [0.913, 0.996] | string-stable | 0.32 | 32.49 | 0.34 | 1.11 [1.07, 1.15] | 0.81 |
| diffusion_cf | all | 60 | 1.019 [1.014, 1.024] | amplifying | 1.00 | 1.43 | 2.03 | 1.09 [1.05, 1.13] | 1.05 |
| diffusion_cf | ACC | 47 | 1.021 [1.015, 1.027] | amplifying | 1.00 | 1.43 | 2.11 | 1.11 [1.07, 1.15] | 1.05 |
| gp_residual_idm | all | 60 | 1.013 [1.006, 1.023] | amplifying | 0.37 | 1.15 | 1.49 | 1.09 [1.05, 1.13] | 1.06 |
| gp_residual_idm | ACC | 47 | 1.017 [1.008, 1.029] | amplifying | 0.42 | 1.16 | 1.60 | 1.11 [1.07, 1.15] | 1.07 |
| idm_ar_noise | all | 60 | 0.939 [0.936, 0.942] | string-stable | 0.05 | 3.44 | 1.05 | 1.09 [1.05, 1.13] | 0.96 |
| idm_ar_noise | ACC | 47 | 0.940 [0.937, 0.943] | string-stable | 0.05 | 3.60 | 1.12 | 1.11 [1.07, 1.15] | 0.96 |
| idm_white_noise | all | 60 | 1.010 [1.003, 1.017] | amplifying | 0.63 | 1.24 | 1.49 | 1.09 [1.05, 1.13] | 0.98 |
| idm_white_noise | ACC | 47 | 1.011 [1.003, 1.020] | amplifying | 0.63 | 1.25 | 1.52 | 1.11 [1.07, 1.15] | 0.98 |
| lstm_const_sigma | all | 60 | 0.992 [0.961, 1.026] | inconclusive | 0.26 | 17.24 | 0.23 | 1.09 [1.05, 1.13] | 1.02 |
| lstm_const_sigma | ACC | 47 | 0.979 [0.944, 1.001] | inconclusive | 0.16 | 18.22 | 0.29 | 1.11 [1.07, 1.15] | 0.95 |
| quantile_pidl | all | 60 | 0.845 [0.831, 0.861] | string-stable | 0.05 | 2.14 | 0.09 | 1.09 [1.05, 1.13] | 0.81 |
| quantile_pidl | ACC | 47 | 0.848 [0.831, 0.864] | string-stable | 0.05 | 2.25 | 0.10 | 1.11 [1.07, 1.15] | 0.82 |

## replan / lodo_openacc_r10

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.032 [1.024, 1.041] | amplifying | 1.00 | 1.13 | 2.05 | 1.09 [1.05, 1.13] | 1.08 |
| diffusion_cf | ACC | 47 | 1.035 [1.026, 1.045] | amplifying | 1.00 | 1.13 | 2.18 | 1.11 [1.07, 1.15] | 1.09 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.74 | 1.10 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.996, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## replan / lodo_openacc_r10_share

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.030 [1.023, 1.039] | amplifying | 1.00 | 1.13 | 2.00 | 1.09 [1.05, 1.13] | 1.08 |
| diffusion_cf | ACC | 47 | 1.034 [1.025, 1.043] | amplifying | 1.00 | 1.13 | 2.12 | 1.11 [1.07, 1.15] | 1.08 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.74 | 1.10 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.996, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## replan / lodo_openacc_r20

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.030 [1.024, 1.037] | amplifying | 1.00 | 1.13 | 1.98 | 1.09 [1.05, 1.13] | 1.07 |
| diffusion_cf | ACC | 47 | 1.033 [1.025, 1.041] | amplifying | 1.00 | 1.13 | 2.08 | 1.11 [1.07, 1.15] | 1.07 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.74 | 1.10 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.996, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## replan / lodo_openacc_r20_share

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.029 [1.023, 1.037] | amplifying | 1.00 | 1.13 | 1.96 | 1.09 [1.05, 1.13] | 1.07 |
| diffusion_cf | ACC | 47 | 1.032 [1.025, 1.040] | amplifying | 1.00 | 1.13 | 2.06 | 1.11 [1.07, 1.15] | 1.07 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.74 | 1.10 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.996, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## replan / lodo_openacc_r50

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.030 [1.024, 1.037] | amplifying | 1.00 | 1.14 | 2.01 | 1.09 [1.05, 1.13] | 1.07 |
| diffusion_cf | ACC | 47 | 1.033 [1.026, 1.041] | amplifying | 1.00 | 1.14 | 2.10 | 1.11 [1.07, 1.15] | 1.07 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.74 | 1.10 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.996, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## replan / lodo_openacc_r50_share

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.030 [1.024, 1.037] | amplifying | 1.00 | 1.14 | 2.01 | 1.09 [1.05, 1.13] | 1.07 |
| diffusion_cf | ACC | 47 | 1.033 [1.026, 1.041] | amplifying | 1.00 | 1.14 | 2.10 | 1.11 [1.07, 1.15] | 1.07 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.74 | 1.10 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.996, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## replan / lodo_openacc_s10

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 0.980 [0.978, 0.981] | string-stable | 0.37 | 2.67 | 1.81 | 1.09 [1.05, 1.13] | 0.88 |
| diffusion_cf | ACC | 47 | 0.980 [0.978, 0.982] | string-stable | 0.37 | 2.84 | 1.93 | 1.11 [1.07, 1.15] | 0.89 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.74 | 1.10 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.996, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## replan / lodo_openacc_s100

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.030 [1.024, 1.037] | amplifying | 1.00 | 1.14 | 2.01 | 1.09 [1.05, 1.13] | 1.06 |
| diffusion_cf | ACC | 47 | 1.033 [1.026, 1.041] | amplifying | 1.00 | 1.14 | 2.09 | 1.11 [1.07, 1.15] | 1.07 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.74 | 1.10 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.996, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## replan / lodo_openacc_s25

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.008 [0.997, 1.022] | inconclusive | 0.26 | 1.35 | 1.57 | 1.09 [1.05, 1.13] | 1.13 |
| diffusion_cf | ACC | 47 | 1.012 [0.999, 1.027] | inconclusive | 0.26 | 1.36 | 1.70 | 1.11 [1.07, 1.15] | 1.14 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.74 | 1.10 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.996, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## replan / pool_r10

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.038 [1.030, 1.048] | amplifying | 0.89 | 1.12 | 2.27 | 1.09 [1.05, 1.13] | 1.11 |
| diffusion_cf | ACC | 47 | 1.041 [1.031, 1.052] | amplifying | 0.95 | 1.12 | 2.38 | 1.11 [1.07, 1.15] | 1.11 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.95 | 1.11 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.997, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## replan / pool_r10_share

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.038 [1.030, 1.047] | amplifying | 0.89 | 1.12 | 2.27 | 1.09 [1.05, 1.13] | 1.10 |
| diffusion_cf | ACC | 47 | 1.040 [1.031, 1.051] | amplifying | 0.95 | 1.12 | 2.36 | 1.11 [1.07, 1.15] | 1.11 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.95 | 1.11 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.997, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## replan / pool_r20

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.038 [1.032, 1.046] | amplifying | 1.00 | 1.13 | 2.30 | 1.09 [1.05, 1.13] | 1.09 |
| diffusion_cf | ACC | 47 | 1.041 [1.033, 1.050] | amplifying | 1.00 | 1.13 | 2.40 | 1.11 [1.07, 1.15] | 1.10 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.95 | 1.11 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.997, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## replan / pool_r20_share

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.038 [1.032, 1.046] | amplifying | 0.95 | 1.13 | 2.29 | 1.09 [1.05, 1.13] | 1.09 |
| diffusion_cf | ACC | 47 | 1.040 [1.033, 1.049] | amplifying | 1.00 | 1.13 | 2.38 | 1.11 [1.07, 1.15] | 1.10 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.95 | 1.11 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.997, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## replan / pool_r50

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.039 [1.033, 1.047] | amplifying | 0.84 | 1.14 | 2.38 | 1.09 [1.05, 1.13] | 1.10 |
| diffusion_cf | ACC | 47 | 1.041 [1.034, 1.050] | amplifying | 0.95 | 1.14 | 2.45 | 1.11 [1.07, 1.15] | 1.10 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.95 | 1.11 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.997, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## replan / pool_r50_share

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.039 [1.033, 1.047] | amplifying | 0.84 | 1.14 | 2.38 | 1.09 [1.05, 1.13] | 1.10 |
| diffusion_cf | ACC | 47 | 1.041 [1.034, 1.050] | amplifying | 0.95 | 1.14 | 2.45 | 1.11 [1.07, 1.15] | 1.10 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.95 | 1.11 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.997, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## replan / pool_s10

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 0.978 [0.976, 0.982] | string-stable | 0.26 | 2.48 | 1.64 | 1.09 [1.05, 1.13] | 0.89 |
| diffusion_cf | ACC | 47 | 0.979 [0.975, 0.983] | string-stable | 0.26 | 2.64 | 1.76 | 1.11 [1.07, 1.15] | 0.89 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.95 | 1.11 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.997, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## replan / pool_s100

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.041 [1.036, 1.047] | amplifying | 1.00 | 1.14 | 2.45 | 1.09 [1.05, 1.13] | 1.10 |
| diffusion_cf | ACC | 47 | 1.043 [1.036, 1.051] | amplifying | 1.00 | 1.14 | 2.52 | 1.11 [1.07, 1.15] | 1.10 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.95 | 1.11 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.997, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## replan / pool_s25

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.017 [1.007, 1.030] | amplifying | 0.32 | 1.26 | 1.73 | 1.09 [1.05, 1.13] | 1.20 |
| diffusion_cf | ACC | 47 | 1.020 [1.008, 1.034] | amplifying | 0.32 | 1.27 | 1.86 | 1.11 [1.07, 1.15] | 1.20 |
| idm_white_noise | all | 60 | 1.001 [0.995, 1.008] | inconclusive | 0.95 | 1.11 | 1.12 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.003 [0.997, 1.012] | inconclusive | 1.00 | 1.10 | 1.17 | 1.11 [1.07, 1.15] | 1.00 |

## replan / raw

| model | mode | n | geo ratio [95 %] | verdict | share idx > 1 | A1 / leader | A20 / leader | recorded geo ratio | simulated on recorded idx |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf | all | 60 | 1.040 [1.034, 1.048] | amplifying | 0.89 | 1.26 | 2.66 | 1.09 [1.05, 1.13] | 1.11 |
| diffusion_cf | ACC | 47 | 1.042 [1.035, 1.051] | amplifying | 0.95 | 1.28 | 2.79 | 1.11 [1.07, 1.15] | 1.12 |
| idm_white_noise | all | 60 | 1.003 [0.997, 1.011] | inconclusive | 1.00 | 1.13 | 1.19 | 1.09 [1.05, 1.13] | 1.00 |
| idm_white_noise | ACC | 47 | 1.006 [0.998, 1.015] | inconclusive | 1.00 | 1.12 | 1.25 | 1.11 [1.07, 1.15] | 1.01 |

