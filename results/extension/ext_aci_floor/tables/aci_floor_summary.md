# Online ACI: starting level and guaranteed floor of the test streams (alpha 0.1, gamma 0.01)

Floor = 1 - alpha - (alpha_1 + gamma (1 - alpha)) / (gamma T) per stream; alpha_1 = level reached on the
calibration events; T = events of the test stream with a score. The achieved coverage is the stream's own.

| root | target | streams | alpha_1 min | alpha_1 max | T | floor min | floor max | coverage min | coverage max | streams with floor < 0.87 | alpha_t min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e2 | a | 1400 | 0.0500 | 0.1300 | 500 | 0.8722 | 0.8882 | 0.8780 | 0.9180 | 0 | -0.0090 |
| e2 | mean_1s | 1400 | 0.0400 | 0.1400 | 500 | 0.8702 | 0.8902 | 0.8800 | 0.9200 | 0 | -0.0080 |
| e2_ctrl | a | 120 | 0.0400 | 0.1300 | 500 | 0.8722 | 0.8902 | 0.8820 | 0.9140 | 0 | -0.0060 |
| e2_lodo_ngsim | a | 520 | 0.0500 | 0.1400 | 500 | 0.8702 | 0.8882 | 0.8820 | 0.9480 | 0 | 0.0180 |
| e2_lodo_ngsim_all | a | 780 | 0.0500 | 0.1400 | 500 | 0.8702 | 0.8882 | 0.8840 | 0.9480 | 0 | 0.0040 |
| e2_lodo_openacc | a | 560 | 0.0500 | 0.1400 | 500 | 0.8702 | 0.8882 | 0.8840 | 0.9460 | 0 | 0.0060 |
| e2_pool | a | 1400 | 0.0400 | 0.1500 | 185-500 | 0.8141 | 0.8902 | 0.8486 | 1.0000 | 283 | -0.0060 |
| e2h | a | 1400 | 0.0400 | 0.1400 | 500 | 0.8702 | 0.8902 | 0.8800 | 0.9200 | 0 | -0.0090 |
| e2h | mean_1s | 1400 | 0.0600 | 0.1500 | 500 | 0.8682 | 0.8862 | 0.8760 | 0.9260 | 5 | -0.0090 |
| e2m | a | 1400 | 0.0400 | 0.1400 | 500 | 0.8702 | 0.8902 | 0.8800 | 0.9520 | 0 | -0.0040 |
| e2m | mean_1s | 1400 | 0.0600 | 0.1500 | 500 | 0.8682 | 0.8862 | 0.8820 | 0.9300 | 5 | 0.0020 |
| ext_e2_cl | a | 200 | 0.0400 | 0.1300 | 500 | 0.8722 | 0.8902 | 0.8820 | 0.9180 | 0 | -0.0080 |

Five lowest floors:

| root | target | model | seed | horizon | set | alpha_1 | T | floor | coverage |
|---|---|---|---|---|---|---|---|---|---|
| e2_pool | a | mc_dropout | 1 | 10 | waymo | 0.1500 | 185 | 0.8141 | 0.8865 |
| e2_pool | a | hetero_mdn | 2 | 10 | waymo | 0.1400 | 185 | 0.8195 | 0.8595 |
| e2_pool | a | idm_white_noise_mle | 4 | 1 | waymo | 0.1300 | 185 | 0.8249 | 0.8541 |
| e2_pool | a | deep_ensemble | 2 | 10 | waymo | 0.1300 | 185 | 0.8249 | 0.8919 |
| e2_pool | a | idm_white_noise_mle | 1 | 1 | waymo | 0.1300 | 185 | 0.8249 | 0.8541 |
