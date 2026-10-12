# State-distribution part of the shift (extension S2)

Importance weights over 6 x 6 x 6 quantile bins of (s, dv, v) built on the in-domain set `highd` (152,683 windows). Per shifted set: windows, mass in bins the in-domain set never visits (dropped), effective sample size of the kept windows, overlap mass of the two occupancies.

| set | windows | mass dropped | ESS | ESS / kept | overlap mass |
|---|---|---|---|---|---|
| highd_val | 152,683 | 0.000 | 145,918 | 0.956 | 0.916 |
| ngsim | 438,867 | 0.000 | 877 | 0.002 | 0.181 |
| ngsim_us101 | 425,210 | 0.000 | 421 | 0.001 | 0.200 |
| ngsim_fn | 226,086 | 0.000 | 1,378 | 0.006 | 0.210 |
| waymo | 158,668 | 0.000 | 5,886 | 0.037 | 0.323 |
| openacc | 485,932 | 0.000 | 28,740 | 0.059 | 0.385 |

NLL per window (nats; seed means): in domain, raw on the shifted set, reweighted to the in-domain occupancy; the gap split into its state-distribution part (raw - reweighted) and its conditional part (reweighted - in-domain); `share_state` = state part / gap. Reading rule (PLAN.md S2): a set is explained by the state distribution when share_state >= 0.75.

| model | set | NLL in | NLL raw | NLL rw | state part | conditional part | share state | r_alea raw -> rw | r_epi raw -> rw |
|---|---|---|---|---|---|---|---|---|---|
| deep_ensemble | highd_val | -2.97 | -2.96 | -2.98 | +0.02 | -0.00 | -- | 1.17 -> 1.29 | 1.17 -> 1.25 |
| deep_ensemble | ngsim | -2.97 | 40.66 | 229.42 | -188.76 | +232.40 | -4.33 | 8.11 -> 5.80 | 78.24 -> 127.59 |
| deep_ensemble | ngsim_us101 | -2.97 | 45.93 | 135.99 | -90.06 | +138.96 | -1.82 | 4.56 -> 8.39 | 49.31 -> 57.74 |
| deep_ensemble | ngsim_fn | -2.97 | 38.02 | 177.56 | -139.54 | +180.53 | -3.43 | 2.61 -> 0.84 | 37.52 -> 33.77 |
| deep_ensemble | waymo | -2.97 | 2.79 | 15.28 | -12.49 | +18.25 | -2.18 | 13.11 -> 28.66 | 25.52 -> 29.96 |
| deep_ensemble | openacc | -2.97 | 2.79 | 14.89 | -12.10 | +17.87 | -2.10 | 1092.81 -> 506.83 | 848.14 -> 415.53 |
| diffusion_cf | highd_val | -2.43 | -2.40 | -2.42 | +0.02 | +0.02 | -- | 1.12 -> 1.08 | -- |
| diffusion_cf | ngsim | -2.43 | 16.32 | 47.92 | -31.60 | +50.35 | -1.74 | 38.51 -> 54.82 | -- |
| diffusion_cf | ngsim_us101 | -2.43 | 17.43 | 26.21 | -8.78 | +28.64 | -0.44 | 28.94 -> 31.85 | -- |
| diffusion_cf | ngsim_fn | -2.43 | 15.08 | 32.41 | -17.33 | +34.84 | -0.96 | 22.68 -> 22.80 | -- |
| diffusion_cf | waymo | -2.43 | 0.11 | 2.26 | -2.15 | +4.70 | -0.85 | 6.31 -> 3.49 | -- |
| diffusion_cf | openacc | -2.43 | 0.42 | 2.13 | -1.71 | +4.56 | -0.62 | 7.27 -> 5.22 | -- |
| gp_residual_idm | highd_val | 0.22 | 0.25 | 0.22 | +0.02 | +0.00 | -- | 1.00 -> 1.00 | 1.29 -> 1.12 |
| gp_residual_idm | ngsim | 0.22 | 5.75 | 7.80 | -2.05 | +7.58 | -0.37 | 1.00 -> 1.00 | 3.82 -> 5.23 |
| gp_residual_idm | ngsim_us101 | 0.22 | 3.60 | 6.21 | -2.61 | +5.99 | -0.77 | 1.00 -> 1.00 | 3.21 -> 3.00 |
| gp_residual_idm | ngsim_fn | 0.22 | 3.32 | 4.36 | -1.04 | +4.14 | -0.33 | 1.00 -> 1.00 | 2.77 -> 2.15 |
| gp_residual_idm | waymo | 0.22 | 1.52 | 0.93 | +0.59 | +0.71 | 0.45 | 1.00 -> 1.00 | 7.61 -> 2.77 |
| gp_residual_idm | openacc | 0.22 | 0.66 | 1.36 | -0.70 | +1.14 | -1.59 | 1.00 -> 1.00 | 11.14 -> 8.41 |
| hetero_mdn | highd_val | -2.93 | -2.91 | -2.93 | +0.02 | +0.00 | -- | 1.15 -> 1.29 | -- |
| hetero_mdn | ngsim | -2.93 | 82.49 | 423.55 | -341.07 | +426.48 | -4.05 | 4.92 -> 3.78 | -- |
| hetero_mdn | ngsim_us101 | -2.93 | 87.70 | 234.82 | -147.12 | +237.75 | -1.66 | 3.08 -> 3.08 | -- |
| hetero_mdn | ngsim_fn | -2.93 | 73.95 | 278.09 | -204.14 | +281.02 | -2.73 | 1.87 -> 0.59 | -- |
| hetero_mdn | waymo | -2.93 | 6.39 | 22.27 | -15.89 | +25.20 | -1.74 | 11.17 -> 28.20 | -- |
| hetero_mdn | openacc | -2.93 | 6.76 | 26.29 | -19.53 | +29.22 | -2.07 | 1324.47 -> 541.11 | -- |
| hetero_mdn_mix3 | highd_val | -3.01 | -2.99 | -3.00 | +0.02 | +0.00 | -- | 0.99 -> 1.05 | -- |
| hetero_mdn_mix3 | ngsim | -3.01 | 5.72 | 34.94 | -29.22 | +37.94 | -3.30 | 20.20 -> 7.80 | -- |
| hetero_mdn_mix3 | ngsim_us101 | -3.01 | 5.32 | 14.41 | -9.09 | +17.41 | -1.07 | 9.58 -> 5.73 | -- |
| hetero_mdn_mix3 | ngsim_fn | -3.01 | 4.95 | 19.54 | -14.59 | +22.55 | -1.81 | 5.17 -> 1.85 | -- |
| hetero_mdn_mix3 | waymo | -3.01 | 0.26 | 1.10 | -0.84 | +4.11 | -0.26 | 17.32 -> 39.00 | -- |
| hetero_mdn_mix3 | openacc | -3.01 | -0.08 | 2.04 | -2.12 | +5.04 | -0.73 | 723.23 -> 427.33 | -- |
| idm_ar_noise | highd_val | -1.64 | -1.60 | -1.63 | +0.02 | +0.01 | -- | 1.00 -> 1.00 | -- |
| idm_ar_noise | ngsim | -1.64 | 4.51 | 7.94 | -3.43 | +9.58 | -0.56 | 1.00 -> 1.00 | -- |
| idm_ar_noise | ngsim_us101 | -1.64 | 1.96 | 2.84 | -0.87 | +4.48 | -0.24 | 1.00 -> 1.00 | -- |
| idm_ar_noise | ngsim_fn | -1.64 | 1.68 | 7.09 | -5.41 | +8.73 | -1.63 | 1.00 -> 1.00 | -- |
| idm_ar_noise | waymo | -1.64 | -1.21 | -0.62 | -0.59 | +1.03 | -1.34 | 1.00 -> 1.00 | -- |
| idm_ar_noise | openacc | -1.64 | 2.20 | 2.32 | -0.11 | +3.96 | -0.03 | 1.00 -> 1.00 | -- |
| idm_white_noise | highd_val | 1.41 | 1.48 | 1.43 | +0.05 | +0.02 | 0.69 | 1.00 -> 1.00 | -- |
| idm_white_noise | ngsim | 1.41 | 1.90 | 2.36 | -0.47 | +0.96 | -0.95 | 1.00 -> 1.00 | -- |
| idm_white_noise | ngsim_us101 | 1.41 | 1.61 | 2.85 | -1.23 | +1.44 | -5.93 | 1.00 -> 1.00 | -- |
| idm_white_noise | ngsim_fn | 1.41 | 1.60 | 2.00 | -0.39 | +0.59 | -1.98 | 1.00 -> 1.00 | -- |
| idm_white_noise | waymo | 1.41 | 1.34 | 1.58 | -0.24 | +0.18 | -- | 1.00 -> 1.00 | -- |
| idm_white_noise | openacc | 1.41 | 1.42 | 1.87 | -0.45 | +0.46 | -- | 1.00 -> 1.00 | -- |
| idm_white_noise_mle | highd_val | 0.39 | 0.45 | 0.41 | +0.03 | +0.02 | 0.58 | 1.00 -> 1.00 | -- |
| idm_white_noise_mle | ngsim | 0.39 | 4.91 | 5.77 | -0.86 | +5.38 | -0.19 | 1.00 -> 1.00 | -- |
| idm_white_noise_mle | ngsim_us101 | 0.39 | 2.88 | 4.72 | -1.84 | +4.33 | -0.74 | 1.00 -> 1.00 | -- |
| idm_white_noise_mle | ngsim_fn | 0.39 | 2.51 | 3.06 | -0.55 | +2.67 | -0.26 | 1.00 -> 1.00 | -- |
| idm_white_noise_mle | waymo | 0.39 | 1.65 | 0.85 | +0.79 | +0.46 | 0.63 | 1.00 -> 1.00 | -- |
| idm_white_noise_mle | openacc | 0.39 | 0.99 | 1.72 | -0.74 | +1.34 | -1.23 | 1.00 -> 1.00 | -- |
| lstm_const_sigma | highd_val | -2.78 | -2.75 | -2.77 | +0.01 | +0.02 | -- | 1.00 -> 1.00 | -- |
| lstm_const_sigma | ngsim | -2.78 | 466.96 | 1107.75 | -640.78 | +1110.53 | -1.37 | 1.00 -> 1.00 | -- |
| lstm_const_sigma | ngsim_us101 | -2.78 | 329.42 | 501.86 | -172.43 | +504.64 | -0.52 | 1.00 -> 1.00 | -- |
| lstm_const_sigma | ngsim_fn | -2.78 | 252.28 | 431.43 | -179.15 | +434.22 | -0.71 | 1.00 -> 1.00 | -- |
| lstm_const_sigma | waymo | -2.78 | 27.66 | 25.40 | +2.26 | +28.19 | 0.07 | 1.00 -> 1.00 | -- |
| lstm_const_sigma | openacc | -2.78 | 279.01 | 125.84 | +153.17 | +128.62 | 0.52 | 1.00 -> 1.00 | -- |
| mc_dropout | highd_val | -2.40 | -2.37 | -2.39 | +0.02 | +0.00 | -- | 1.06 -> 1.02 | 1.10 -> 1.02 |
| mc_dropout | ngsim | -2.40 | 3.97 | 10.33 | -6.36 | +12.72 | -0.97 | 55.19 -> 243.02 | 16.66 -> 34.43 |
| mc_dropout | ngsim_us101 | -2.40 | 3.89 | 7.13 | -3.24 | +9.53 | -0.52 | 44.43 -> 201.14 | 12.78 -> 18.41 |
| mc_dropout | ngsim_fn | -2.40 | 3.48 | 6.37 | -2.89 | +8.77 | -0.47 | 30.85 -> 166.64 | 10.48 -> 15.10 |
| mc_dropout | waymo | -2.40 | -0.71 | -0.44 | -0.27 | +1.96 | -0.16 | 12.76 -> 10.24 | 5.46 -> 3.05 |
| mc_dropout | openacc | -2.40 | -0.68 | 0.39 | -1.07 | +2.78 | -0.63 | 171.04 -> 93.27 | 53.27 -> 21.36 |
| quantile_pidl | highd_val | -2.94 | -2.91 | -2.92 | +0.01 | +0.02 | 0.19 | 1.02 -> 1.00 | -- |
| quantile_pidl | ngsim | -2.94 | 101.95 | 247.93 | -145.98 | +250.87 | -1.70 | 9.27 -> 11.00 | -- |
| quantile_pidl | ngsim_us101 | -2.94 | 124.79 | 184.58 | -59.79 | +187.52 | -0.58 | 5.81 -> 7.78 | -- |
| quantile_pidl | ngsim_fn | -2.94 | 110.52 | 183.01 | -72.50 | +185.96 | -0.90 | 4.58 -> 4.23 | -- |
| quantile_pidl | waymo | -2.94 | 17.45 | 35.82 | -18.36 | +38.76 | -0.66 | 10.68 -> 29.97 | -- |
| quantile_pidl | openacc | -2.94 | 10.54 | 18.39 | -7.85 | +21.33 | -0.73 | 322.08 -> 232.18 | -- |
