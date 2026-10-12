# Cost of the laws (extension X2)

Machine: {"cpu": "Intel64 Family 6 Model 151 Stepping 2, GenuineIntel", "torch": "2.11.0+cu128", "cuda": "12.8", "gpu": "NVIDIA RTX 6000 Ada Generation", "threads": 12}

| label | n_params | train_s_per_seed | train_s_min | train_s_max | infer_ms_per_1000_cuda | infer_ms_per_1000_cpu | seconds_per_unit_median | us_per_vehicle_rollout_s | n_rollouts |
|---|---|---|---|---|---|---|---|---|---|
| deep_ensemble | 109770 | 979.8 | 837.1 | 1195 | 1.485 | 23.9 | 2.308 | 6.077 | 100 |
| diffusion_cf | 216385 | 430.1 | 343.9 | 567.6 | 1301 | 7.696e+04 | 1.053 | 8.666 | 32 |
| gp_residual_idm | 181221 | 205.6 | 140.4 | 330.5 | 0.7209 | 2.138 | 1.242 | 3.271 | 100 |
| hetero_mdn | 21954 | 229 | 202.8 | 252.7 | 0.3074 | 6.07 |  |  |  |
| hetero_mdn_mix3 | 22409 | 176.7 | 157.1 | 195.9 | 0.3799 | 4.386 |  |  |  |
| idm_ar_noise | 2 | 0.1795 | 0.1795 | 0.1795 | 0.2875 | 0.2027 | 0.3045 | 0.8018 | 100 |
| idm_white_noise | 1 | 0.05919 | 0.05919 | 0.05919 | 0.01783 | 0.01592 | 0.2927 | 0.7705 | 100 |
| idm_white_noise_mle | 1 | 0.07949 | 0.07949 | 0.07949 | 0.01644 | 0.01514 |  |  |  |
| lstm_const_sigma | 21889 | 213.2 | 197.6 | 229.6 | 0.3512 | 4.486 | 0.718 | 1.89 | 100 |
| mc_dropout | 21954 | 200.2 | 181.5 | 236.6 | 4.879 | 14.41 |  |  |  |
| quantile_pidl | 22149 | 311.4 | 229.5 | 405.4 | 1.026 | 5.228 | 1.647 | 4.336 | 100 |

Training: artifacts/e1 (highD, recorded signals), mean over seeds; inference: highD test windows with a full 5 s future (seed-0 checkpoints), median of repeated calls, plan samplers timed on fewer rows and scaled; platoon: artifacts/e3 timings of the cells pool, lodo_openacc, pool_w11, lodo_openacc_w11, raw_w61, raw_a (one unit = one profile x 20 followers x R rollouts over the profile's time line; us per simulated vehicle-rollout-second).
