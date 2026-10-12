# Closed-loop fine-tuned networks versus their one-step twins (extension S7)

## One-step metrics (seed means): twin -> fine-tuned

| regime | twin | set | NLL | CRPS | PICP90 | RMSE |
|---|---|---|---|---|---|---|
| highd | hetero_mdn | highd | -2.93 -> -2.81 | 0.008 -> 0.009 | 0.908 -> 0.913 | 0.027 -> 0.024 |
| highd | hetero_mdn | highd_val | -2.91 -> -2.79 | 0.008 -> 0.009 | 0.909 -> 0.914 | 0.025 -> 0.022 |
| highd | hetero_mdn | ngsim | 82.49 -> 51.56 | 0.296 -> 0.284 | 0.187 -> 0.219 | 0.476 -> 0.463 |
| highd | hetero_mdn | ngsim_fn | 73.95 -> 44.89 | 0.236 -> 0.222 | 0.143 -> 0.178 | 0.333 -> 0.319 |
| highd | hetero_mdn | ngsim_us101 | 87.70 -> 53.41 | 0.252 -> 0.238 | 0.137 -> 0.169 | 0.374 -> 0.360 |
| highd | hetero_mdn | openacc | 6.76 -> 3.99 | 0.210 -> 0.189 | 0.644 -> 0.687 | 0.978 -> 0.900 |
| highd | hetero_mdn | waymo | 6.39 -> 3.59 | 0.068 -> 0.069 | 0.502 -> 0.547 | 0.148 -> 0.143 |
| highd | lstm_const_sigma | highd | -2.78 -> -2.24 | 0.008 -> 0.014 | 0.925 -> 0.916 | 0.015 -> 0.026 |
| highd | lstm_const_sigma | highd_val | -2.75 -> -2.21 | 0.008 -> 0.014 | 0.923 -> 0.912 | 0.015 -> 0.027 |
| highd | lstm_const_sigma | ngsim | 466.96 -> 116.36 | 0.308 -> 0.250 | 0.081 -> 0.151 | 0.471 -> 0.401 |
| highd | lstm_const_sigma | ngsim_fn | 252.28 -> 54.60 | 0.257 -> 0.197 | 0.073 -> 0.153 | 0.348 -> 0.279 |
| highd | lstm_const_sigma | ngsim_us101 | 329.42 -> 74.84 | 0.282 -> 0.220 | 0.070 -> 0.143 | 0.397 -> 0.325 |
| highd | lstm_const_sigma | openacc | 279.01 -> 98.93 | 0.098 -> 0.102 | 0.465 -> 0.596 | 0.356 -> 0.364 |
| highd | lstm_const_sigma | waymo | 27.66 -> 6.54 | 0.068 -> 0.065 | 0.306 -> 0.434 | 0.121 -> 0.113 |
| pool | hetero_mdn | highd | -2.85 -> -2.83 | 0.008 -> 0.009 | 0.962 -> 0.903 | 0.015 -> 0.017 |
| pool | hetero_mdn | highd_val | -2.84 -> -2.82 | 0.008 -> 0.009 | 0.963 -> 0.905 | 0.021 -> 0.021 |
| pool | hetero_mdn | ngsim | -2.38 -> -2.35 | 0.011 -> 0.011 | 0.946 -> 0.906 | 0.027 -> 0.026 |
| pool | hetero_mdn | ngsim_fn | -2.41 -> -2.36 | 0.010 -> 0.010 | 0.957 -> 0.921 | 0.036 -> 0.035 |
| pool | hetero_mdn | ngsim_us101 | -2.50 -> -2.49 | 0.010 -> 0.010 | 0.955 -> 0.914 | 0.036 -> 0.035 |
| pool | hetero_mdn | openacc | -2.68 -> -2.73 | 0.012 -> 0.011 | 0.960 -> 0.927 | 0.062 -> 0.044 |
| pool | hetero_mdn | pool_test | -2.62 -> -2.60 | 0.010 -> 0.010 | 0.956 -> 0.911 | 0.036 -> 0.031 |
| pool | hetero_mdn | waymo | -2.24 -> -2.12 | 0.012 -> 0.012 | 0.947 -> 0.917 | 0.033 -> 0.033 |
| pool | lstm_const_sigma | highd | -2.77 -> -2.12 | 0.008 -> 0.016 | 0.982 -> 0.914 | 0.013 -> 0.029 |
| pool | lstm_const_sigma | highd_val | -2.76 -> -2.12 | 0.008 -> 0.016 | 0.982 -> 0.916 | 0.013 -> 0.029 |
| pool | lstm_const_sigma | ngsim | -2.35 -> -2.00 | 0.011 -> 0.017 | 0.930 -> 0.899 | 0.023 -> 0.032 |
| pool | lstm_const_sigma | ngsim_fn | -2.37 -> -2.02 | 0.010 -> 0.016 | 0.949 -> 0.914 | 0.022 -> 0.032 |
| pool | lstm_const_sigma | ngsim_us101 | -2.42 -> -2.05 | 0.010 -> 0.016 | 0.945 -> 0.911 | 0.021 -> 0.031 |
| pool | lstm_const_sigma | openacc | -2.57 -> -1.95 | 0.009 -> 0.017 | 0.955 -> 0.914 | 0.018 -> 0.034 |
| pool | lstm_const_sigma | pool_test | -2.52 -> -2.02 | 0.009 -> 0.017 | 0.956 -> 0.910 | 0.019 -> 0.032 |
| pool | lstm_const_sigma | waymo | -1.85 -> -1.72 | 0.011 -> 0.018 | 0.932 -> 0.899 | 0.031 -> 0.039 |
| lodo_openacc | hetero_mdn | highd | -2.79 -> -2.29 | 0.009 -> 0.013 | 0.950 -> 0.867 | 0.017 -> 0.025 |
| lodo_openacc | hetero_mdn | highd_val | -2.78 -> -2.29 | 0.009 -> 0.014 | 0.952 -> 0.869 | 0.023 -> 0.029 |
| lodo_openacc | hetero_mdn | ngsim | -2.29 -> -1.95 | 0.012 -> 0.017 | 0.944 -> 0.846 | 0.030 -> 0.035 |
| lodo_openacc | hetero_mdn | ngsim_fn | -2.30 -> -1.98 | 0.012 -> 0.016 | 0.956 -> 0.859 | 0.034 -> 0.038 |
| lodo_openacc | hetero_mdn | ngsim_us101 | -2.40 -> -2.12 | 0.012 -> 0.015 | 0.956 -> 0.873 | 0.036 -> 0.041 |
| lodo_openacc | hetero_mdn | openacc | -1.51 -> -1.72 | 0.038 -> 0.045 | 0.937 -> 0.887 | 0.216 -> 0.220 |
| lodo_openacc | hetero_mdn | pool_test | -2.52 -> -2.12 | 0.011 -> 0.015 | 0.949 -> 0.863 | 0.028 -> 0.034 |
| lodo_openacc | hetero_mdn | waymo | -2.08 -> -1.71 | 0.013 -> 0.018 | 0.935 -> 0.860 | 0.037 -> 0.042 |
| lodo_openacc | lstm_const_sigma | highd | -2.52 -> -1.85 | 0.010 -> 0.021 | 0.972 -> 0.896 | 0.018 -> 0.038 |
| lodo_openacc | lstm_const_sigma | highd_val | -2.52 -> -1.86 | 0.010 -> 0.021 | 0.973 -> 0.900 | 0.018 -> 0.038 |
| lodo_openacc | lstm_const_sigma | ngsim | -2.29 -> -1.86 | 0.012 -> 0.020 | 0.940 -> 0.908 | 0.025 -> 0.038 |
| lodo_openacc | lstm_const_sigma | ngsim_fn | -2.31 -> -1.90 | 0.011 -> 0.019 | 0.955 -> 0.922 | 0.024 -> 0.036 |
| lodo_openacc | lstm_const_sigma | ngsim_us101 | -2.35 -> -1.93 | 0.011 -> 0.019 | 0.952 -> 0.921 | 0.023 -> 0.035 |
| lodo_openacc | lstm_const_sigma | openacc | 56.38 -> 16.75 | 0.040 -> 0.048 | 0.917 -> 0.863 | 0.248 -> 0.207 |
| lodo_openacc | lstm_const_sigma | pool_test | -2.37 -> -1.85 | 0.011 -> 0.020 | 0.957 -> 0.904 | 0.023 -> 0.038 |
| lodo_openacc | lstm_const_sigma | waymo | -1.91 -> -1.55 | 0.013 -> 0.023 | 0.936 -> 0.887 | 0.032 -> 0.047 |

## E2 (highD training): H3.4 verdict of the ACI-corrected intervals, coverage at 10 s raw / ACI, collided share

| which | model | set | verdict | PICP raw | PICP ACI | collided |
|---|---|---|---|---|---|---|
| twin | hetero_mdn | highd | supported | 0.325 | 0.899 | 0.012 |
| twin | hetero_mdn | ngsim | inconclusive | 0.152 | 0.885 | 0.290 |
| twin | hetero_mdn | ngsim_us101 | inconclusive | 0.150 | 0.886 | 0.161 |
| twin | hetero_mdn | openacc | supported | 0.276 | 0.889 | 0.126 |
| twin | hetero_mdn | waymo | inconclusive | 0.231 | 0.888 | 0.198 |
| twin | lstm_const_sigma | highd | supported | 0.417 | 0.900 | 0.009 |
| twin | lstm_const_sigma | ngsim | inconclusive | 0.084 | 0.888 | 0.325 |
| twin | lstm_const_sigma | ngsim_us101 | inconclusive | 0.096 | 0.890 | 0.176 |
| twin | lstm_const_sigma | openacc | inconclusive | 0.290 | 0.887 | 0.075 |
| twin | lstm_const_sigma | waymo | inconclusive | 0.155 | 0.886 | 0.183 |
| cl | hetero_mdn_cl | highd | supported | 0.234 | 0.898 | 0.018 |
| cl | hetero_mdn_cl | ngsim | inconclusive | 0.124 | 0.887 | 0.317 |
| cl | hetero_mdn_cl | ngsim_us101 | supported | 0.146 | 0.888 | 0.188 |
| cl | hetero_mdn_cl | openacc | supported | 0.195 | 0.890 | 0.127 |
| cl | hetero_mdn_cl | waymo | supported | 0.169 | 0.890 | 0.219 |
| cl | lstm_const_sigma_cl | highd | supported | 0.314 | 0.897 | 0.002 |
| cl | lstm_const_sigma_cl | ngsim | inconclusive | 0.086 | 0.888 | 0.122 |
| cl | lstm_const_sigma_cl | ngsim_us101 | inconclusive | 0.102 | 0.887 | 0.030 |
| cl | lstm_const_sigma_cl | openacc | inconclusive | 0.155 | 0.887 | 0.068 |
| cl | lstm_const_sigma_cl | waymo | supported | 0.106 | 0.890 | 0.050 |

## Platoons: H3.2 reduction (extended set, response window) and mean collided share

| which | cell | model | mode | reduction [file interval] verdict | collided |
|---|---|---|---|---|---|
| twin | lodo_openacc | lstm_const_sigma | all | -25.03 [-39.28, -15.70] refuted | 0.69 |
| twin | lodo_openacc | lstm_const_sigma | ACC | -20.49 [-31.73, -12.07] refuted | 0.69 |
| twin | lodo_openacc | lstm_const_sigma | Human | -84.04 [-127.18, -2.14] refuted | 0.69 |
| twin | lodo_openacc | lstm_const_sigma | mixed | -28.70 [-28.70, -28.70] inconclusive | 0.69 |
| twin | pool | lstm_const_sigma | all | -8.19 [-12.76, -5.39] refuted | 0.67 |
| twin | pool | lstm_const_sigma | ACC | -6.28 [-8.17, -4.58] refuted | 0.67 |
| twin | pool | lstm_const_sigma | Human | -34.41 [-48.42, -3.04] refuted | 0.67 |
| twin | pool | lstm_const_sigma | mixed | -2.04 [-2.04, -2.04] inconclusive | 0.67 |
| twin | raw | lstm_const_sigma | all | -18.36 [-28.06, -11.25] refuted | 0.74 |
| twin | raw | lstm_const_sigma | ACC | -15.76 [-24.77, -8.63] refuted | 0.74 |
| twin | raw | lstm_const_sigma | Human | -49.69 [-73.02, -7.56] refuted | 0.74 |
| twin | raw | lstm_const_sigma | mixed | -20.97 [-20.97, -20.97] inconclusive | 0.74 |
| cl | lodo_openacc | hetero_mdn_cl | all | -21.55 [-32.47, -13.68] refuted | 0.68 |
| cl | lodo_openacc | lstm_const_sigma_cl | all | -5.33 [-10.44, -2.42] refuted | 0.62 |
| cl | lodo_openacc | hetero_mdn_cl | ACC | -18.29 [-28.01, -10.42] refuted | 0.68 |
| cl | lodo_openacc | lstm_const_sigma_cl | ACC | -4.71 [-10.25, -1.75] refuted | 0.62 |
| cl | lodo_openacc | hetero_mdn_cl | Human | -62.22 [-87.86, -0.67] refuted | 0.68 |
| cl | lodo_openacc | lstm_const_sigma_cl | Human | -13.31 [-25.21, 0.02] refuted | 0.62 |
| cl | lodo_openacc | hetero_mdn_cl | mixed | -32.38 [-32.38, -32.38] inconclusive | 0.68 |
| cl | lodo_openacc | lstm_const_sigma_cl | mixed | -6.47 [-6.47, -6.47] inconclusive | 0.62 |
| cl | pool | hetero_mdn_cl | all | -5.93 [-9.81, -3.17] refuted | 0.62 |
| cl | pool | lstm_const_sigma_cl | all | -0.77 [-1.67, -0.14] refuted | 0.63 |
| cl | pool | hetero_mdn_cl | ACC | -4.45 [-7.22, -2.01] refuted | 0.62 |
| cl | pool | lstm_const_sigma_cl | ACC | -0.46 [-0.98, 0.06] inconclusive | 0.63 |
| cl | pool | hetero_mdn_cl | Human | -24.78 [-27.17, 0.11] inconclusive | 0.62 |
| cl | pool | lstm_const_sigma_cl | Human | -4.71 [-9.03, 0.18] inconclusive | 0.63 |
| cl | pool | hetero_mdn_cl | mixed | -8.40 [-8.40, -8.40] inconclusive | 0.62 |
| cl | pool | lstm_const_sigma_cl | mixed | -1.35 [-1.35, -1.35] inconclusive | 0.63 |
| cl | raw | hetero_mdn_cl | all | -3.45 [-5.46, -2.07] refuted | 0.80 |
| cl | raw | lstm_const_sigma_cl | all | -8.91 [-15.03, -4.96] refuted | 0.34 |
| cl | raw | hetero_mdn_cl | ACC | -2.66 [-3.90, -1.50] refuted | 0.80 |
| cl | raw | lstm_const_sigma_cl | ACC | -7.20 [-12.83, -3.54] refuted | 0.34 |
| cl | raw | hetero_mdn_cl | Human | -13.23 [-21.18, -4.06] refuted | 0.80 |
| cl | raw | lstm_const_sigma_cl | Human | -31.15 [-42.42, -1.80] refuted | 0.34 |
| cl | raw | hetero_mdn_cl | mixed | -2.96 [-2.96, -2.96] inconclusive | 0.80 |
| cl | raw | lstm_const_sigma_cl | mixed | -2.52 [-2.52, -2.52] inconclusive | 0.34 |

