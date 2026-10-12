# Sampler settings and plan sharing: cost and quality (extension S3 and S8)

Per cell: re-planning interval and DDIM sampling steps of the diffusion sampler, plan sharing, the H3.2 reduction against the IDM + white noise reference (extended set, response window, 95 % file-cluster interval, verdict) on the ACC and the human-driven profiles, the mean share of collided rollouts over the 20 followers and all 60 profiles, the simulation time per unit (s) and its ratio to the stored setting of the base.

| base | re-plan [s] | steps | shared plans | ACC reduction [interval] verdict | human reduction [interval] verdict | coll. | s / unit | cost ratio | admissible (ACC) |
|---|---|---|---|---|---|---|---|---|---|
| lodo_openacc | 1 | 50 | no | 0.51 [0.26, 0.62] supported | -0.50 [-0.57, -0.01] refuted | 0.11 | 9.7 | 4.78 | yes |
| lodo_openacc | 2 | 50 | no | 0.48 [0.27, 0.57] supported | -0.47 [-0.52, -0.02] refuted | 0.07 | 4.5 | 2.20 | yes |
| lodo_openacc | 5 | 10 | no | -0.47 [-1.71, -0.02] refuted | -2.60 [-5.92, -1.61] refuted | 0.77 | 0.6 | 0.29 | no |
| lodo_openacc | 5 | 25 | no | 0.32 [-0.19, 0.55] inconclusive | -2.87 [-3.27, 0.03] refuted | 0.29 | 1.2 | 0.57 | no |
| lodo_openacc | 5 | 50 | no | 0.47 [0.27, 0.55] supported | -0.48 [-0.58, 0.14] inconclusive | 0.12 | 2.0 | 1.00 | yes |
| lodo_openacc | 5 | 100 | no | 0.44 [0.26, 0.52] supported | -0.46 [-0.57, 0.11] inconclusive | 0.11 | 3.7 | 1.82 | yes |
| lodo_openacc | 1 | 50 | yes | 0.50 [0.26, 0.61] supported | -0.44 [-0.50, -0.03] refuted | 0.11 | 5.4 | 2.64 | yes |
| lodo_openacc | 2 | 50 | yes | 0.47 [0.27, 0.56] supported | -0.45 [-0.50, -0.04] refuted | 0.08 | 2.7 | 1.33 | yes |
| lodo_openacc | 5 | 50 | yes | 0.47 [0.27, 0.55] supported | -0.48 [-0.58, 0.14] inconclusive | 0.12 | 1.1 | 0.53 | yes |
| pool | 1 | 50 | no | 0.56 [0.26, 0.68] supported | -0.73 [-0.89, -0.10] refuted | 0.21 | 15.1 | 5.56 | yes |
| pool | 2 | 50 | no | 0.53 [0.28, 0.64] supported | -0.64 [-0.83, -0.07] refuted | 0.11 | 6.1 | 2.25 | yes |
| pool | 5 | 10 | no | -0.36 [-1.40, 0.03] refuted | -2.26 [-3.95, -1.32] refuted | 0.75 | 0.6 | 0.21 | no |
| pool | 5 | 25 | no | 0.31 [-0.21, 0.55] inconclusive | -3.31 [-3.82, 0.10] inconclusive | 0.30 | 1.3 | 0.48 | no |
| pool | 5 | 50 | no | 0.56 [0.29, 0.67] supported | -0.90 [-1.08, 0.15] inconclusive | 0.16 | 2.7 | 1.00 | yes |
| pool | 5 | 100 | no | 0.55 [0.30, 0.66] supported | -0.95 [-1.19, 0.09] inconclusive | 0.17 | 4.7 | 1.74 | yes |
| pool | 1 | 50 | yes | 0.55 [0.26, 0.67] supported | -0.69 [-0.84, -0.12] refuted | 0.20 | 5.4 | 1.98 | yes |
| pool | 2 | 50 | yes | 0.53 [0.28, 0.63] supported | -0.63 [-0.82, -0.08] refuted | 0.11 | 2.7 | 0.99 | yes |
| pool | 5 | 50 | yes | 0.56 [0.29, 0.67] supported | -0.90 [-1.08, 0.15] inconclusive | 0.16 | 1.1 | 0.41 | yes |
| raw | 5 | 50 | no | 0.34 [-0.26, 0.57] inconclusive | -0.84 [-1.00, -0.02] refuted | 0.30 | 2.7 | 1.00 | yes |

Cheapest admissible setting of pool: re-planning every 5 s with 50 steps (2.7 s per unit, cost ratio 1.00).
Cheapest admissible setting of lodo_openacc: re-planning every 5 s with 50 steps (2.0 s per unit, cost ratio 1.00).
