# Inducing inputs of the GP-residual checkpoints against the training rows

Distance (standardised units of s, dv, v) from each of the stored inducing inputs to its nearest training row;
`rows_nn_*`: the same distance between a training row and its nearest other training row (20 000 sampled rows).

| root | training data | checkpoints | inducing inputs | coincide (< 1e-9) | within 1e-6 | within 1e-3 | smallest distance | median distance | rows: median nearest-neighbour distance |
|---|---|---|---|---|---|---|---|---|---|
| e1 | highd | 5 | 1500 | 0 | 0 | 0 | 0.00139 | 0.0582 | 0.0175 |
| e1_mean1s | highd | 5 | 1500 | 0 | 0 | 0 | 0.00217 | 0.0627 | 0.0176 |
| e1h | highd | 5 | 1500 | 0 | 0 | 0 | 0.00253 | 0.0576 | 0.0174 |
| e1h_mean1s | highd | 5 | 1500 | 0 | 0 | 0 | 0.00212 | 0.0582 | 0.018 |
| e1m | highd | 5 | 1500 | 0 | 0 | 0 | 0.00253 | 0.0576 | 0.0174 |
| e1m_causal | highd | 5 | 1500 | 0 | 0 | 0 | 0.00281 | 0.0625 | 0.0178 |
| e1m_mean1s | highd | 5 | 1500 | 0 | 0 | 0 | 0.00212 | 0.0582 | 0.018 |
| e1m_mean1s_abv | highd | 5 | 1500 | 0 | 0 | 0 | 0.00212 | 0.0582 | 0.018 |
| e1m_rt | highd | 5 | 1500 | 0 | 0 | 0 | 0.00331 | 0.0575 | 0.018 |
| e6_lodo_ngsim | highd+ngsim_us101+waymo+openacc | 5 | 1500 | 0 | 0 | 0 | 0.00181 | 0.0674 | 0.0173 |
| e6_lodo_ngsim_all | highd+waymo+openacc | 5 | 1500 | 0 | 0 | 0 | 0.00252 | 0.062 | 0.0185 |
| e6_lodo_openacc | highd+ngsim+ngsim_us101+waymo | 5 | 1500 | 0 | 0 | 0 | 0.00254 | 0.0543 | 0.0167 |
| e6_pool | highd+ngsim+ngsim_us101+waymo+openacc | 5 | 1500 | 0 | 0 | 0 | 0.00113 | 0.0634 | 0.0154 |
| ext_e1_ngsim | ngsim | 5 | 1500 | 0 | 0 | 0 | 0.00193 | 0.046 | 0.0133 |

Total: 70 checkpoints, 21000 inducing inputs, 0 coincide with a training row, 0 within 1e-6, 0 within 1e-3; smallest distance 0.00113.
