# FollowNet-protocol check (extension X5, item 1)

Closed loop behind the recorded leader (semi-implicit Euler, 10 Hz, acceleration clipped to [-8, 4] m/s^2, 3 s observed, collision = s <= 0), point predictions of the laws (mean; median of the quantile law; mean of the 32 sampled plans of the diffusion laws, re-planned every 5 s). MSE of the spacing per event averaged over events [m^2], collision rate [per mille]; mean over the seeds 0-4 (min-max).

## highd (2503 events)

| Law | MSE spacing | min-max | RMSE spacing (pooled) | collisions, per mille | min-max |
|---|---|---|---|---|---|
| deep_ensemble | 34.33 | 27.74-43.45 | 5.84 | 11.75 | 3.60-19.18 |
| diffusion_cf | 7.37 | 7.14-7.81 | 2.71 | 0.00 | 0.00-0.00 |
| diffusion_cf_hist | 121.67 | 98.02-139.99 | 11.01 | 3.44 | 2.80-4.39 |
| gp_residual_idm | 17.63 | 17.03-18.32 | 4.20 | 6.71 | 5.99-7.19 |
| hetero_mdn | 65.27 | 22.75-129.89 | 7.68 | 18.86 | 4.79-38.75 |
| hetero_mdn_mix3 | 44.43 | 23.23-61.41 | 6.59 | 39.31 | 12.78-61.93 |
| idm_ar_noise | 20.85 | 20.85-20.85 | 4.57 | 0.00 | 0.00-0.00 |
| idm_white_noise | 17.33 | 17.33-17.33 | 4.16 | 0.00 | 0.00-0.00 |
| idm_white_noise_mle | 26.56 | 26.56-26.56 | 5.15 | 0.00 | 0.00-0.00 |
| lstm_const_sigma | 20.49 | 17.09-29.58 | 4.50 | 15.90 | 4.00-41.55 |
| lstm_hetero_frozen | 20.49 | 17.09-29.58 | 4.50 | 15.90 | 4.00-41.55 |
| mc_dropout | 136.61 | 95.86-167.08 | 11.62 | 51.14 | 14.78-109.87 |
| quantile_pidl | 17.45 | 14.35-20.33 | 4.17 | 13.34 | 3.20-23.17 |

Published FollowNet rows (Chen et al. 2023, Tables 2 and 3; the benchmark's own test split, 15 s events):

| Model | MSE spacing | collisions, per mille |
|---|---|---|
| GHR | 23.76 | 0.00 |
| IDM | 27.50 | 0.00 |
| NN | 24.06 | 39.87 |
| LSTM | 22.90 | 3.72 |
| DDPG | 38.52 | 0.00 |

FollowNet-2 reference implementations (plan P4, FollowNet highD events, reconstructed signals, split driver-0, 2503 test events): closed-loop RMSE of the spacing [m] and collision share:

| Model | RMSE spacing | MSE (RMSE^2) | collision share |
|---|---|---|---|
| idm | 4.005 | 16.04 | 0.000 |
| lstm | 3.482 | 12.12 | 0.002 |
| lstm_loop | 3.532 | 12.48 | 0.003 |
| seq2seq | 3.093 | 9.57 | 0.002 |
| diffusion | 3.361 | 11.30 | 0.006 |
| pidl | 8.892 | 79.07 | 0.129 |
| ddpg | 11.097 | 123.14 | 0.000 |

## ngsim_fn (1937 events)

| Law | MSE spacing | min-max | RMSE spacing (pooled) | collisions, per mille | min-max |
|---|---|---|---|---|---|
| deep_ensemble | 2636.96 | 571.72-7249.44 | 51.88 | 430.98 | 313.37-572.02 |
| diffusion_cf | 271.30 | 114.98-592.03 | 16.60 | 43.37 | 12.91-56.79 |
| diffusion_cf_hist | 2241.51 | 1107.94-2775.08 | 49.97 | 138.15 | 120.29-172.95 |
| gp_residual_idm | 204.55 | 179.67-241.00 | 16.84 | 1.55 | 1.03-2.58 |
| hetero_mdn | 5883.37 | 1984.30-10769.01 | 86.72 | 418.48 | 226.64-692.82 |
| hetero_mdn_mix3 | 6226.26 | 3213.16-8961.85 | 90.78 | 445.12 | 212.18-773.88 |
| idm_ar_noise | 2885.56 | 2885.56-2885.56 | 58.86 | 2.58 | 2.58-2.58 |
| idm_white_noise | 83.56 | 83.56-83.56 | 9.75 | 0.00 | 0.00-0.00 |
| idm_white_noise_mle | 230.91 | 230.91-230.91 | 16.91 | 0.00 | 0.00-0.00 |
| lstm_const_sigma | 92191.39 | 2301.60-256986.82 | 296.47 | 392.57 | 157.98-500.26 |
| lstm_hetero_frozen | 92191.39 | 2301.60-256986.82 | 296.47 | 392.57 | 157.98-500.26 |
| mc_dropout | 75374.29 | 4050.28-177632.50 | 267.45 | 326.17 | 153.33-502.32 |
| quantile_pidl | 2137.33 | 1028.93-3175.13 | 49.66 | 436.24 | 303.05-567.37 |

Published FollowNet rows (Chen et al. 2023, Tables 2 and 3; the benchmark's own test split, 15 s events):

| Model | MSE spacing | collisions, per mille |
|---|---|---|
| GHR | 62.18 | 0.00 |
| IDM | 54.97 | 0.00 |
| NN | 29.00 | 3.46 |
| LSTM | 23.86 | 17.30 |
| DDPG | 46.93 | 0.00 |

Protocol differences: the published rows use FollowNet's own train/validation/test split and its unified 15 s events; ours use the driver-disjoint fold 0 of the roots (highD: test fold; NGSIM I-80: every FollowNet event, a shifted set for laws trained on highD); FollowNet-2 uses its reconstructed signals and its own split. The published MSE is the mean over events of the per-event MSE; the pooled RMSE weights every sample equally.
