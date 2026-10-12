# Conditional coverage of the rollout intervals (extension S5)

Coverage of the 90 % gap intervals by E1 regime at the issue instant (free speed v0 = 28.88 m/s), for the raw, static (multiplicative score of E2), CQR (additive score) and ACI-online (gamma 0.01) intervals; seed means over the stored E2 rollouts (target a). Reading rule (PLAN.md S5): conditionally valid when every regime with at least 30 events has a coverage within 0.05 of 0.90 at 10 s.

## Validity at 10 s (regimes with at least 30 events)

| model | set | variant | regimes | failing | min PICP | max PICP | failing regimes |
|---|---|---|---|---|---|---|---|
| deep_ensemble | highd | aci_online | 5 | 1 | 0.79 | 0.95 | braking 0.79 (n=32) |
| deep_ensemble | highd | cqr | 5 | 2 | 0.74 | 0.96 | braking 0.74 (n=32); steady 0.96 (n=118) |
| deep_ensemble | highd | raw | 5 | 5 | 0.28 | 0.52 | braking 0.28 (n=32); approaching 0.46 (n=133); free 0.51 (n=89); steady 0.52 (n=118); other 0.46 (n=115) |
| deep_ensemble | highd | static | 5 | 1 | 0.81 | 0.95 | braking 0.81 (n=32) |
| deep_ensemble | ngsim | aci_online | 4 | 1 | 0.70 | 0.94 | braking 0.70 (n=72) |
| deep_ensemble | ngsim | cqr | 4 | 4 | 0.39 | 0.69 | congested 0.69 (n=249); braking 0.39 (n=72); approaching 0.61 (n=95); other 0.67 (n=55) |
| deep_ensemble | ngsim | raw | 4 | 4 | 0.23 | 0.44 | congested 0.32 (n=249); braking 0.23 (n=72); approaching 0.35 (n=95); other 0.44 (n=55) |
| deep_ensemble | ngsim | static | 4 | 3 | 0.60 | 0.86 | congested 0.82 (n=249); braking 0.60 (n=72); other 0.84 (n=55) |
| deep_ensemble | ngsim_us101 | aci_online | 5 | 1 | 0.78 | 0.92 | braking 0.78 (n=89) |
| deep_ensemble | ngsim_us101 | cqr | 5 | 5 | 0.43 | 0.72 | congested 0.72 (n=138); braking 0.43 (n=89); approaching 0.62 (n=74); steady 0.58 (n=78); other 0.62 (n=120) |
| deep_ensemble | ngsim_us101 | raw | 5 | 5 | 0.19 | 0.38 | congested 0.37 (n=138); braking 0.19 (n=89); approaching 0.35 (n=74); steady 0.32 (n=78); other 0.38 (n=120) |
| deep_ensemble | ngsim_us101 | static | 5 | 5 | 0.68 | 0.82 | congested 0.82 (n=138); braking 0.68 (n=89); approaching 0.82 (n=74); steady 0.78 (n=78); other 0.82 (n=120) |
| deep_ensemble | openacc | aci_online | 6 | 4 | 0.76 | 0.97 | braking 0.77 (n=31); free 0.76 (n=49); steady 0.97 (n=205); other 0.81 (n=97) |
| deep_ensemble | openacc | cqr | 6 | 5 | 0.32 | 0.91 | congested 0.73 (n=52); braking 0.46 (n=31); approaching 0.70 (n=66); free 0.32 (n=49); other 0.60 (n=97) |
| deep_ensemble | openacc | raw | 6 | 6 | 0.15 | 0.54 | congested 0.45 (n=52); braking 0.17 (n=31); approaching 0.38 (n=66); free 0.15 (n=49); steady 0.54 (n=205); other 0.30 (n=97) |
| deep_ensemble | openacc | static | 6 | 5 | 0.56 | 0.92 | congested 0.84 (n=52); braking 0.67 (n=31); approaching 0.78 (n=66); free 0.56 (n=49); other 0.72 (n=97) |
| deep_ensemble | waymo | aci_online | 5 | 1 | 0.80 | 0.92 | braking 0.80 (n=50) |
| deep_ensemble | waymo | cqr | 5 | 5 | 0.56 | 0.77 | congested 0.77 (n=232); braking 0.56 (n=50); approaching 0.71 (n=65); steady 0.74 (n=58); other 0.70 (n=87) |
| deep_ensemble | waymo | raw | 5 | 5 | 0.24 | 0.36 | congested 0.36 (n=232); braking 0.24 (n=50); approaching 0.34 (n=65); steady 0.33 (n=58); other 0.28 (n=87) |
| deep_ensemble | waymo | static | 5 | 5 | 0.66 | 0.82 | congested 0.82 (n=232); braking 0.66 (n=50); approaching 0.81 (n=65); steady 0.77 (n=58); other 0.76 (n=87) |
| deep_ensemble__traj | highd | aci_online | 5 | 1 | 0.84 | 0.95 | braking 0.84 (n=32) |
| deep_ensemble__traj | highd | cqr | 5 | 2 | 0.79 | 0.95 | braking 0.79 (n=32); other 0.84 (n=115) |
| deep_ensemble__traj | highd | raw | 5 | 5 | 0.64 | 0.84 | braking 0.64 (n=32); approaching 0.84 (n=133); free 0.77 (n=89); steady 0.82 (n=118); other 0.73 (n=115) |
| deep_ensemble__traj | highd | static | 5 | 2 | 0.82 | 0.94 | braking 0.82 (n=32); other 0.85 (n=115) |
| deep_ensemble__traj | ngsim | aci_online | 4 | 1 | 0.75 | 0.91 | braking 0.75 (n=72) |
| deep_ensemble__traj | ngsim | cqr | 4 | 4 | 0.45 | 0.74 | congested 0.74 (n=249); braking 0.45 (n=72); approaching 0.67 (n=95); other 0.71 (n=55) |
| deep_ensemble__traj | ngsim | raw | 4 | 4 | 0.36 | 0.63 | congested 0.63 (n=249); braking 0.36 (n=72); approaching 0.60 (n=95); other 0.61 (n=55) |
| deep_ensemble__traj | ngsim | static | 4 | 4 | 0.59 | 0.80 | congested 0.80 (n=249); braking 0.59 (n=72); approaching 0.78 (n=95); other 0.80 (n=55) |
| deep_ensemble__traj | ngsim_us101 | aci_online | 5 | 1 | 0.81 | 0.93 | braking 0.81 (n=89) |
| deep_ensemble__traj | ngsim_us101 | cqr | 5 | 5 | 0.44 | 0.78 | congested 0.78 (n=138); braking 0.44 (n=89); approaching 0.67 (n=74); steady 0.59 (n=78); other 0.64 (n=120) |
| deep_ensemble__traj | ngsim_us101 | raw | 5 | 5 | 0.35 | 0.68 | congested 0.68 (n=138); braking 0.35 (n=89); approaching 0.57 (n=74); steady 0.49 (n=78); other 0.54 (n=120) |
| deep_ensemble__traj | ngsim_us101 | static | 5 | 5 | 0.58 | 0.81 | congested 0.81 (n=138); braking 0.58 (n=89); approaching 0.76 (n=74); steady 0.68 (n=78); other 0.75 (n=120) |
| deep_ensemble__traj | openacc | aci_online | 6 | 4 | 0.77 | 0.96 | braking 0.78 (n=31); free 0.77 (n=49); steady 0.96 (n=205); other 0.84 (n=97) |
| deep_ensemble__traj | openacc | cqr | 6 | 5 | 0.53 | 0.92 | congested 0.79 (n=52); braking 0.53 (n=31); approaching 0.78 (n=66); free 0.55 (n=49); other 0.70 (n=97) |
| deep_ensemble__traj | openacc | raw | 6 | 6 | 0.43 | 0.85 | congested 0.73 (n=52); braking 0.43 (n=31); approaching 0.70 (n=66); free 0.48 (n=49); steady 0.85 (n=205); other 0.61 (n=97) |
| deep_ensemble__traj | openacc | static | 6 | 3 | 0.66 | 0.93 | braking 0.67 (n=31); free 0.66 (n=49); other 0.77 (n=97) |
| deep_ensemble__traj | waymo | aci_online | 5 | 1 | 0.80 | 0.93 | braking 0.80 (n=50) |
| deep_ensemble__traj | waymo | cqr | 5 | 3 | 0.67 | 0.88 | braking 0.67 (n=50); steady 0.79 (n=58); other 0.77 (n=87) |
| deep_ensemble__traj | waymo | raw | 5 | 5 | 0.54 | 0.79 | congested 0.79 (n=232); braking 0.54 (n=50); approaching 0.75 (n=65); steady 0.65 (n=58); other 0.66 (n=87) |
| deep_ensemble__traj | waymo | static | 5 | 3 | 0.74 | 0.89 | braking 0.74 (n=50); steady 0.81 (n=58); other 0.80 (n=87) |
| diffusion_cf | highd | aci_online | 5 | 1 | 0.83 | 0.91 | braking 0.83 (n=32) |
| diffusion_cf | highd | cqr | 5 | 1 | 0.84 | 0.92 | braking 0.84 (n=32) |
| diffusion_cf | highd | raw | 5 | 1 | 0.85 | 0.92 | braking 0.85 (n=32) |
| diffusion_cf | highd | static | 5 | 1 | 0.84 | 0.92 | braking 0.84 (n=32) |
| diffusion_cf | ngsim | aci_online | 4 | 2 | 0.81 | 0.92 | braking 0.81 (n=72); other 0.81 (n=55) |
| diffusion_cf | ngsim | cqr | 4 | 3 | 0.68 | 0.86 | congested 0.84 (n=249); braking 0.68 (n=72); other 0.73 (n=55) |
| diffusion_cf | ngsim | raw | 4 | 3 | 0.68 | 0.86 | congested 0.84 (n=249); braking 0.68 (n=72); other 0.73 (n=55) |
| diffusion_cf | ngsim | static | 4 | 3 | 0.68 | 0.86 | congested 0.84 (n=249); braking 0.68 (n=72); other 0.74 (n=55) |
| diffusion_cf | ngsim_us101 | aci_online | 5 | 1 | 0.84 | 0.93 | braking 0.84 (n=89) |
| diffusion_cf | ngsim_us101 | cqr | 5 | 4 | 0.70 | 0.87 | braking 0.70 (n=89); approaching 0.85 (n=74); steady 0.83 (n=78); other 0.79 (n=120) |
| diffusion_cf | ngsim_us101 | raw | 5 | 3 | 0.70 | 0.87 | braking 0.70 (n=89); steady 0.83 (n=78); other 0.79 (n=120) |
| diffusion_cf | ngsim_us101 | static | 5 | 4 | 0.71 | 0.87 | braking 0.71 (n=89); approaching 0.85 (n=74); steady 0.82 (n=78); other 0.79 (n=120) |
| diffusion_cf | openacc | aci_online | 6 | 3 | 0.82 | 0.95 | braking 0.83 (n=31); steady 0.95 (n=205); other 0.82 (n=97) |
| diffusion_cf | openacc | cqr | 6 | 1 | 0.87 | 0.97 | steady 0.97 (n=205) |
| diffusion_cf | openacc | raw | 6 | 1 | 0.87 | 0.97 | steady 0.97 (n=205) |
| diffusion_cf | openacc | static | 6 | 1 | 0.87 | 0.97 | steady 0.97 (n=205) |
| diffusion_cf | waymo | aci_online | 5 | 2 | 0.80 | 0.93 | braking 0.80 (n=50); other 0.84 (n=87) |
| diffusion_cf | waymo | cqr | 5 | 2 | 0.70 | 0.88 | braking 0.70 (n=50); other 0.78 (n=87) |
| diffusion_cf | waymo | raw | 5 | 2 | 0.71 | 0.88 | braking 0.71 (n=50); other 0.78 (n=87) |
| diffusion_cf | waymo | static | 5 | 2 | 0.70 | 0.89 | braking 0.70 (n=50); other 0.78 (n=87) |
| gp_residual_idm | highd | aci_online | 5 | 2 | 0.61 | 0.96 | braking 0.61 (n=32); steady 0.96 (n=118) |
| gp_residual_idm | highd | cqr | 5 | 3 | 0.72 | 0.97 | braking 0.72 (n=32); steady 0.97 (n=118); other 0.84 (n=115) |
| gp_residual_idm | highd | raw | 5 | 5 | 0.09 | 0.36 | braking 0.09 (n=32); approaching 0.23 (n=133); free 0.29 (n=89); steady 0.36 (n=118); other 0.25 (n=115) |
| gp_residual_idm | highd | static | 5 | 2 | 0.61 | 0.97 | braking 0.61 (n=32); steady 0.97 (n=118) |
| gp_residual_idm | ngsim | aci_online | 4 | 1 | 0.79 | 0.92 | braking 0.79 (n=72) |
| gp_residual_idm | ngsim | cqr | 4 | 0 | 0.87 | 0.92 | none |
| gp_residual_idm | ngsim | raw | 4 | 4 | 0.12 | 0.23 | congested 0.17 (n=249); braking 0.12 (n=72); approaching 0.17 (n=95); other 0.23 (n=55) |
| gp_residual_idm | ngsim | static | 4 | 4 | 0.73 | 0.84 | congested 0.84 (n=249); braking 0.73 (n=72); approaching 0.80 (n=95); other 0.82 (n=55) |
| gp_residual_idm | ngsim_us101 | aci_online | 5 | 0 | 0.88 | 0.94 | none |
| gp_residual_idm | ngsim_us101 | cqr | 5 | 1 | 0.86 | 0.95 | approaching 0.95 (n=74) |
| gp_residual_idm | ngsim_us101 | raw | 5 | 5 | 0.14 | 0.21 | congested 0.14 (n=138); braking 0.21 (n=89); approaching 0.14 (n=74); steady 0.16 (n=78); other 0.15 (n=120) |
| gp_residual_idm | ngsim_us101 | static | 5 | 4 | 0.82 | 0.88 | congested 0.83 (n=138); braking 0.82 (n=89); steady 0.84 (n=78); other 0.83 (n=120) |
| gp_residual_idm | openacc | aci_online | 6 | 3 | 0.63 | 0.98 | braking 0.63 (n=31); steady 0.98 (n=205); other 0.83 (n=97) |
| gp_residual_idm | openacc | cqr | 6 | 4 | 0.55 | 0.98 | congested 0.79 (n=52); braking 0.55 (n=31); steady 0.98 (n=205); other 0.67 (n=97) |
| gp_residual_idm | openacc | raw | 6 | 6 | 0.06 | 0.43 | congested 0.18 (n=52); braking 0.06 (n=31); approaching 0.24 (n=66); free 0.33 (n=49); steady 0.43 (n=205); other 0.16 (n=97) |
| gp_residual_idm | openacc | static | 6 | 4 | 0.50 | 0.96 | congested 0.79 (n=52); braking 0.50 (n=31); steady 0.96 (n=205); other 0.74 (n=97) |
| gp_residual_idm | waymo | aci_online | 5 | 2 | 0.85 | 0.99 | braking 0.85 (n=50); approaching 0.99 (n=65) |
| gp_residual_idm | waymo | cqr | 5 | 2 | 0.67 | 0.92 | congested 0.67 (n=232); other 0.72 (n=87) |
| gp_residual_idm | waymo | raw | 5 | 5 | 0.12 | 0.30 | congested 0.12 (n=232); braking 0.20 (n=50); approaching 0.30 (n=65); steady 0.22 (n=58); other 0.14 (n=87) |
| gp_residual_idm | waymo | static | 5 | 3 | 0.68 | 0.90 | congested 0.68 (n=232); braking 0.70 (n=50); other 0.74 (n=87) |
| gp_residual_idm__step | highd | aci_online | 5 | 2 | 0.62 | 0.96 | braking 0.62 (n=32); steady 0.96 (n=118) |
| gp_residual_idm__step | highd | cqr | 5 | 3 | 0.72 | 0.97 | braking 0.72 (n=32); steady 0.97 (n=118); other 0.85 (n=115) |
| gp_residual_idm__step | highd | raw | 5 | 5 | 0.07 | 0.36 | braking 0.07 (n=32); approaching 0.24 (n=133); free 0.28 (n=89); steady 0.36 (n=118); other 0.25 (n=115) |
| gp_residual_idm__step | highd | static | 5 | 2 | 0.62 | 0.96 | braking 0.62 (n=32); steady 0.96 (n=118) |
| gp_residual_idm__step | ngsim | aci_online | 4 | 1 | 0.81 | 0.91 | braking 0.81 (n=72) |
| gp_residual_idm__step | ngsim | cqr | 4 | 0 | 0.86 | 0.92 | none |
| gp_residual_idm__step | ngsim | raw | 4 | 4 | 0.12 | 0.22 | congested 0.16 (n=249); braking 0.12 (n=72); approaching 0.16 (n=95); other 0.22 (n=55) |
| gp_residual_idm__step | ngsim | static | 4 | 4 | 0.72 | 0.81 | congested 0.81 (n=249); braking 0.72 (n=72); approaching 0.77 (n=95); other 0.81 (n=55) |
| gp_residual_idm__step | ngsim_us101 | aci_online | 5 | 0 | 0.87 | 0.94 | none |
| gp_residual_idm__step | ngsim_us101 | cqr | 5 | 0 | 0.85 | 0.95 | none |
| gp_residual_idm__step | ngsim_us101 | raw | 5 | 5 | 0.14 | 0.21 | congested 0.14 (n=138); braking 0.21 (n=89); approaching 0.14 (n=74); steady 0.16 (n=78); other 0.14 (n=120) |
| gp_residual_idm__step | ngsim_us101 | static | 5 | 4 | 0.81 | 0.86 | congested 0.81 (n=138); braking 0.81 (n=89); steady 0.83 (n=78); other 0.81 (n=120) |
| gp_residual_idm__step | openacc | aci_online | 6 | 3 | 0.65 | 0.99 | braking 0.65 (n=31); steady 0.99 (n=205); other 0.84 (n=97) |
| gp_residual_idm__step | openacc | cqr | 6 | 5 | 0.52 | 0.98 | congested 0.79 (n=52); braking 0.52 (n=31); free 0.84 (n=49); steady 0.98 (n=205); other 0.67 (n=97) |
| gp_residual_idm__step | openacc | raw | 6 | 6 | 0.06 | 0.42 | congested 0.17 (n=52); braking 0.06 (n=31); approaching 0.21 (n=66); free 0.16 (n=49); steady 0.42 (n=205); other 0.13 (n=97) |
| gp_residual_idm__step | openacc | static | 6 | 6 | 0.48 | 0.96 | congested 0.78 (n=52); braking 0.48 (n=31); approaching 0.78 (n=66); free 0.70 (n=49); steady 0.96 (n=205); other 0.66 (n=97) |
| gp_residual_idm__step | waymo | aci_online | 5 | 2 | 0.85 | 0.98 | approaching 0.98 (n=65); other 0.95 (n=87) |
| gp_residual_idm__step | waymo | cqr | 5 | 2 | 0.66 | 0.93 | congested 0.66 (n=232); other 0.71 (n=87) |
| gp_residual_idm__step | waymo | raw | 5 | 5 | 0.11 | 0.29 | congested 0.11 (n=232); braking 0.18 (n=50); approaching 0.29 (n=65); steady 0.21 (n=58); other 0.14 (n=87) |
| gp_residual_idm__step | waymo | static | 5 | 3 | 0.63 | 0.88 | congested 0.63 (n=232); braking 0.68 (n=50); other 0.72 (n=87) |
| hetero_mdn | highd | aci_online | 5 | 0 | 0.87 | 0.93 | none |
| hetero_mdn | highd | cqr | 5 | 1 | 0.81 | 0.95 | braking 0.81 (n=32) |
| hetero_mdn | highd | raw | 5 | 5 | 0.28 | 0.36 | braking 0.28 (n=32); approaching 0.30 (n=133); free 0.33 (n=89); steady 0.36 (n=118); other 0.33 (n=115) |
| hetero_mdn | highd | static | 5 | 0 | 0.87 | 0.94 | none |
| hetero_mdn | ngsim | aci_online | 4 | 1 | 0.79 | 0.90 | braking 0.79 (n=72) |
| hetero_mdn | ngsim | cqr | 4 | 4 | 0.31 | 0.60 | congested 0.60 (n=249); braking 0.31 (n=72); approaching 0.44 (n=95); other 0.51 (n=55) |
| hetero_mdn | ngsim | raw | 4 | 4 | 0.07 | 0.18 | congested 0.18 (n=249); braking 0.07 (n=72); approaching 0.13 (n=95); other 0.16 (n=55) |
| hetero_mdn | ngsim | static | 4 | 4 | 0.40 | 0.61 | congested 0.61 (n=249); braking 0.40 (n=72); approaching 0.50 (n=95); other 0.56 (n=55) |
| hetero_mdn | ngsim_us101 | aci_online | 5 | 1 | 0.79 | 0.92 | braking 0.79 (n=89) |
| hetero_mdn | ngsim_us101 | cqr | 5 | 5 | 0.33 | 0.62 | congested 0.62 (n=138); braking 0.33 (n=89); approaching 0.55 (n=74); steady 0.53 (n=78); other 0.52 (n=120) |
| hetero_mdn | ngsim_us101 | raw | 5 | 5 | 0.10 | 0.17 | congested 0.17 (n=138); braking 0.10 (n=89); approaching 0.15 (n=74); steady 0.15 (n=78); other 0.16 (n=120) |
| hetero_mdn | ngsim_us101 | static | 5 | 5 | 0.42 | 0.65 | congested 0.65 (n=138); braking 0.42 (n=89); approaching 0.59 (n=74); steady 0.55 (n=78); other 0.57 (n=120) |
| hetero_mdn | openacc | aci_online | 6 | 2 | 0.80 | 0.94 | congested 0.81 (n=52); braking 0.80 (n=31) |
| hetero_mdn | openacc | cqr | 6 | 6 | 0.37 | 0.83 | congested 0.61 (n=52); braking 0.44 (n=31); approaching 0.66 (n=66); free 0.37 (n=49); steady 0.83 (n=205); other 0.58 (n=97) |
| hetero_mdn | openacc | raw | 6 | 6 | 0.13 | 0.37 | congested 0.25 (n=52); braking 0.14 (n=31); approaching 0.26 (n=66); free 0.13 (n=49); steady 0.37 (n=205); other 0.23 (n=97) |
| hetero_mdn | openacc | static | 6 | 6 | 0.66 | 0.85 | congested 0.69 (n=52); braking 0.66 (n=31); approaching 0.81 (n=66); free 0.80 (n=49); steady 0.85 (n=205); other 0.75 (n=97) |
| hetero_mdn | waymo | aci_online | 5 | 1 | 0.77 | 0.93 | braking 0.77 (n=50) |
| hetero_mdn | waymo | cqr | 5 | 5 | 0.52 | 0.77 | congested 0.77 (n=232); braking 0.52 (n=50); approaching 0.70 (n=65); steady 0.75 (n=58); other 0.71 (n=87) |
| hetero_mdn | waymo | raw | 5 | 5 | 0.13 | 0.29 | congested 0.24 (n=232); braking 0.13 (n=50); approaching 0.29 (n=65); steady 0.21 (n=58); other 0.23 (n=87) |
| hetero_mdn | waymo | static | 5 | 5 | 0.57 | 0.78 | congested 0.77 (n=232); braking 0.57 (n=50); approaching 0.76 (n=65); steady 0.78 (n=58); other 0.76 (n=87) |
| hetero_mdn_mix3 | highd | aci_online | 5 | 0 | 0.86 | 0.93 | none |
| hetero_mdn_mix3 | highd | cqr | 5 | 2 | 0.85 | 0.96 | braking 0.85 (n=32); steady 0.96 (n=118) |
| hetero_mdn_mix3 | highd | raw | 5 | 5 | 0.24 | 0.35 | braking 0.24 (n=32); approaching 0.35 (n=133); free 0.27 (n=89); steady 0.30 (n=118); other 0.28 (n=115) |
| hetero_mdn_mix3 | highd | static | 5 | 0 | 0.88 | 0.94 | none |
| hetero_mdn_mix3 | ngsim | aci_online | 4 | 1 | 0.78 | 0.93 | braking 0.78 (n=72) |
| hetero_mdn_mix3 | ngsim | cqr | 4 | 4 | 0.31 | 0.54 | congested 0.54 (n=249); braking 0.31 (n=72); approaching 0.39 (n=95); other 0.43 (n=55) |
| hetero_mdn_mix3 | ngsim | raw | 4 | 4 | 0.10 | 0.18 | congested 0.18 (n=249); braking 0.10 (n=72); approaching 0.12 (n=95); other 0.16 (n=55) |
| hetero_mdn_mix3 | ngsim | static | 4 | 4 | 0.43 | 0.62 | congested 0.62 (n=249); braking 0.43 (n=72); approaching 0.50 (n=95); other 0.55 (n=55) |
| hetero_mdn_mix3 | ngsim_us101 | aci_online | 5 | 1 | 0.80 | 0.93 | braking 0.80 (n=89) |
| hetero_mdn_mix3 | ngsim_us101 | cqr | 5 | 5 | 0.35 | 0.58 | congested 0.54 (n=138); braking 0.35 (n=89); approaching 0.58 (n=74); steady 0.47 (n=78); other 0.46 (n=120) |
| hetero_mdn_mix3 | ngsim_us101 | raw | 5 | 5 | 0.11 | 0.19 | congested 0.19 (n=138); braking 0.11 (n=89); approaching 0.18 (n=74); steady 0.14 (n=78); other 0.14 (n=120) |
| hetero_mdn_mix3 | ngsim_us101 | static | 5 | 5 | 0.51 | 0.68 | congested 0.62 (n=138); braking 0.51 (n=89); approaching 0.68 (n=74); steady 0.56 (n=78); other 0.58 (n=120) |
| hetero_mdn_mix3 | openacc | aci_online | 6 | 3 | 0.77 | 0.95 | congested 0.77 (n=52); free 0.84 (n=49); steady 0.95 (n=205) |
| hetero_mdn_mix3 | openacc | cqr | 6 | 6 | 0.37 | 0.81 | congested 0.55 (n=52); braking 0.55 (n=31); approaching 0.65 (n=66); free 0.37 (n=49); steady 0.81 (n=205); other 0.61 (n=97) |
| hetero_mdn_mix3 | openacc | raw | 6 | 6 | 0.12 | 0.27 | congested 0.12 (n=52); braking 0.20 (n=31); approaching 0.27 (n=66); free 0.15 (n=49); steady 0.25 (n=205); other 0.24 (n=97) |
| hetero_mdn_mix3 | openacc | static | 6 | 6 | 0.62 | 0.84 | congested 0.62 (n=52); braking 0.75 (n=31); approaching 0.78 (n=66); free 0.67 (n=49); steady 0.84 (n=205); other 0.77 (n=97) |
| hetero_mdn_mix3 | waymo | aci_online | 5 | 4 | 0.84 | 0.97 | congested 0.85 (n=232); braking 0.84 (n=50); steady 0.97 (n=58); other 0.96 (n=87) |
| hetero_mdn_mix3 | waymo | cqr | 5 | 5 | 0.54 | 0.80 | congested 0.65 (n=232); braking 0.54 (n=50); approaching 0.75 (n=65); steady 0.80 (n=58); other 0.71 (n=87) |
| hetero_mdn_mix3 | waymo | raw | 5 | 5 | 0.16 | 0.26 | congested 0.24 (n=232); braking 0.16 (n=50); approaching 0.26 (n=65); steady 0.24 (n=58); other 0.24 (n=87) |
| hetero_mdn_mix3 | waymo | static | 5 | 5 | 0.62 | 0.81 | congested 0.68 (n=232); braking 0.62 (n=50); approaching 0.79 (n=65); steady 0.81 (n=58); other 0.81 (n=87) |
| idm_ar_noise | highd | aci_online | 5 | 5 | 0.50 | 0.99 | braking 0.50 (n=32); approaching 0.83 (n=133); free 0.97 (n=89); steady 0.99 (n=118); other 0.99 (n=115) |
| idm_ar_noise | highd | cqr | 5 | 4 | 0.49 | 0.99 | braking 0.49 (n=32); free 0.98 (n=89); steady 0.99 (n=118); other 0.96 (n=115) |
| idm_ar_noise | highd | raw | 5 | 3 | 0.29 | 0.88 | braking 0.29 (n=32); approaching 0.64 (n=133); steady 0.85 (n=118) |
| idm_ar_noise | highd | static | 5 | 4 | 0.59 | 1.00 | braking 0.59 (n=32); free 0.97 (n=89); steady 1.00 (n=118); other 0.99 (n=115) |
| idm_ar_noise | ngsim | aci_online | 4 | 3 | 0.80 | 0.98 | braking 0.84 (n=72); approaching 0.80 (n=95); other 0.98 (n=55) |
| idm_ar_noise | ngsim | cqr | 4 | 4 | 0.32 | 0.84 | congested 0.66 (n=249); braking 0.32 (n=72); approaching 0.49 (n=95); other 0.84 (n=55) |
| idm_ar_noise | ngsim | raw | 4 | 4 | 0.11 | 0.51 | congested 0.38 (n=249); braking 0.11 (n=72); approaching 0.21 (n=95); other 0.51 (n=55) |
| idm_ar_noise | ngsim | static | 4 | 4 | 0.41 | 0.81 | congested 0.68 (n=249); braking 0.45 (n=72); approaching 0.41 (n=95); other 0.81 (n=55) |
| idm_ar_noise | ngsim_us101 | aci_online | 5 | 2 | 0.79 | 0.99 | approaching 0.79 (n=74); steady 0.99 (n=78) |
| idm_ar_noise | ngsim_us101 | cqr | 5 | 5 | 0.51 | 0.84 | congested 0.67 (n=138); braking 0.51 (n=89); approaching 0.66 (n=74); steady 0.84 (n=78); other 0.76 (n=120) |
| idm_ar_noise | ngsim_us101 | raw | 5 | 5 | 0.31 | 0.69 | congested 0.43 (n=138); braking 0.31 (n=89); approaching 0.37 (n=74); steady 0.69 (n=78); other 0.50 (n=120) |
| idm_ar_noise | ngsim_us101 | static | 5 | 4 | 0.50 | 0.86 | congested 0.76 (n=138); braking 0.75 (n=89); approaching 0.50 (n=74); other 0.83 (n=120) |
| idm_ar_noise | openacc | aci_online | 6 | 4 | 0.45 | 0.99 | braking 0.45 (n=31); approaching 0.77 (n=66); free 0.98 (n=49); steady 0.99 (n=205) |
| idm_ar_noise | openacc | cqr | 6 | 6 | 0.45 | 0.98 | congested 0.75 (n=52); braking 0.45 (n=31); approaching 0.81 (n=66); free 0.96 (n=49); steady 0.98 (n=205); other 0.85 (n=97) |
| idm_ar_noise | openacc | raw | 6 | 5 | 0.35 | 0.96 | congested 0.65 (n=52); braking 0.35 (n=31); approaching 0.58 (n=66); steady 0.96 (n=205); other 0.68 (n=97) |
| idm_ar_noise | openacc | static | 6 | 4 | 0.46 | 1.00 | braking 0.46 (n=31); approaching 0.79 (n=66); free 1.00 (n=49); steady 0.99 (n=205) |
| idm_ar_noise | waymo | aci_online | 5 | 3 | 0.82 | 1.00 | braking 0.82 (n=50); approaching 0.84 (n=65); steady 1.00 (n=58) |
| idm_ar_noise | waymo | cqr | 5 | 2 | 0.76 | 0.94 | congested 0.83 (n=232); braking 0.76 (n=50) |
| idm_ar_noise | waymo | raw | 5 | 5 | 0.48 | 0.67 | congested 0.54 (n=232); braking 0.48 (n=50); approaching 0.58 (n=65); steady 0.67 (n=58); other 0.60 (n=87) |
| idm_ar_noise | waymo | static | 5 | 3 | 0.78 | 0.98 | braking 0.81 (n=50); approaching 0.78 (n=65); steady 0.98 (n=58) |
| idm_white_noise | highd | aci_online | 5 | 3 | 0.72 | 0.98 | braking 0.72 (n=32); approaching 0.76 (n=133); other 0.98 (n=115) |
| idm_white_noise | highd | cqr | 5 | 2 | 0.70 | 0.94 | braking 0.70 (n=32); approaching 0.81 (n=133) |
| idm_white_noise | highd | raw | 5 | 5 | 0.18 | 0.57 | braking 0.18 (n=32); approaching 0.41 (n=133); free 0.41 (n=89); steady 0.52 (n=118); other 0.57 (n=115) |
| idm_white_noise | highd | static | 5 | 4 | 0.72 | 0.98 | braking 0.72 (n=32); approaching 0.77 (n=133); free 0.96 (n=89); other 0.98 (n=115) |
| idm_white_noise | ngsim | aci_online | 4 | 1 | 0.83 | 0.93 | braking 0.83 (n=72) |
| idm_white_noise | ngsim | cqr | 4 | 2 | 0.78 | 0.92 | braking 0.78 (n=72); other 0.83 (n=55) |
| idm_white_noise | ngsim | raw | 4 | 4 | 0.30 | 0.44 | congested 0.44 (n=249); braking 0.31 (n=72); approaching 0.30 (n=95); other 0.36 (n=55) |
| idm_white_noise | ngsim | static | 4 | 1 | 0.84 | 0.94 | braking 0.84 (n=72) |
| idm_white_noise | ngsim_us101 | aci_online | 5 | 1 | 0.78 | 0.95 | braking 0.78 (n=89) |
| idm_white_noise | ngsim_us101 | cqr | 5 | 0 | 0.87 | 0.94 | none |
| idm_white_noise | ngsim_us101 | raw | 5 | 5 | 0.38 | 0.54 | congested 0.53 (n=138); braking 0.38 (n=89); approaching 0.53 (n=74); steady 0.49 (n=78); other 0.54 (n=120) |
| idm_white_noise | ngsim_us101 | static | 5 | 3 | 0.83 | 0.98 | braking 0.83 (n=89); approaching 0.96 (n=74); other 0.98 (n=120) |
| idm_white_noise | openacc | aci_online | 6 | 3 | 0.74 | 0.97 | braking 0.74 (n=31); approaching 0.78 (n=66); steady 0.97 (n=205) |
| idm_white_noise | openacc | cqr | 6 | 5 | 0.60 | 0.95 | congested 0.75 (n=52); braking 0.77 (n=31); approaching 0.65 (n=66); free 0.62 (n=49); other 0.60 (n=97) |
| idm_white_noise | openacc | raw | 6 | 6 | 0.22 | 0.42 | congested 0.42 (n=52); braking 0.41 (n=31); approaching 0.27 (n=66); free 0.22 (n=49); steady 0.27 (n=205); other 0.35 (n=97) |
| idm_white_noise | openacc | static | 6 | 3 | 0.76 | 0.99 | braking 0.76 (n=31); approaching 0.80 (n=66); steady 0.99 (n=205) |
| idm_white_noise | waymo | aci_online | 5 | 2 | 0.77 | 0.96 | braking 0.77 (n=50); steady 0.96 (n=58) |
| idm_white_noise | waymo | cqr | 5 | 4 | 0.77 | 0.86 | congested 0.77 (n=232); braking 0.82 (n=50); approaching 0.80 (n=65); other 0.77 (n=87) |
| idm_white_noise | waymo | raw | 5 | 5 | 0.33 | 0.41 | congested 0.40 (n=232); braking 0.33 (n=50); approaching 0.36 (n=65); steady 0.41 (n=58); other 0.37 (n=87) |
| idm_white_noise | waymo | static | 5 | 2 | 0.72 | 0.93 | braking 0.72 (n=50); approaching 0.83 (n=65) |
| idm_white_noise_mle | highd | aci_online | 5 | 3 | 0.44 | 0.99 | braking 0.44 (n=32); free 0.99 (n=89); other 0.97 (n=115) |
| idm_white_noise_mle | highd | cqr | 5 | 2 | 0.58 | 0.97 | braking 0.58 (n=32); free 0.97 (n=89) |
| idm_white_noise_mle | highd | raw | 5 | 5 | 0.11 | 0.37 | braking 0.11 (n=32); approaching 0.17 (n=133); free 0.21 (n=89); steady 0.30 (n=118); other 0.37 (n=115) |
| idm_white_noise_mle | highd | static | 5 | 3 | 0.51 | 0.99 | braking 0.51 (n=32); free 0.99 (n=89); other 0.98 (n=115) |
| idm_white_noise_mle | ngsim | aci_online | 4 | 3 | 0.76 | 0.96 | congested 0.96 (n=249); braking 0.76 (n=72); approaching 0.79 (n=95) |
| idm_white_noise_mle | ngsim | cqr | 4 | 3 | 0.69 | 0.91 | braking 0.69 (n=72); approaching 0.72 (n=95); other 0.78 (n=55) |
| idm_white_noise_mle | ngsim | raw | 4 | 4 | 0.06 | 0.22 | congested 0.22 (n=249); braking 0.08 (n=72); approaching 0.06 (n=95); other 0.19 (n=55) |
| idm_white_noise_mle | ngsim | static | 4 | 2 | 0.61 | 0.91 | braking 0.61 (n=72); approaching 0.61 (n=95) |
| idm_white_noise_mle | ngsim_us101 | aci_online | 5 | 3 | 0.73 | 0.97 | congested 0.97 (n=138); braking 0.73 (n=89); approaching 0.84 (n=74) |
| idm_white_noise_mle | ngsim_us101 | cqr | 5 | 4 | 0.71 | 0.89 | braking 0.71 (n=89); approaching 0.78 (n=74); steady 0.82 (n=78); other 0.83 (n=120) |
| idm_white_noise_mle | ngsim_us101 | raw | 5 | 5 | 0.06 | 0.23 | congested 0.15 (n=138); braking 0.13 (n=89); approaching 0.06 (n=74); steady 0.21 (n=78); other 0.23 (n=120) |
| idm_white_noise_mle | ngsim_us101 | static | 5 | 4 | 0.63 | 0.90 | congested 0.85 (n=138); braking 0.63 (n=89); approaching 0.70 (n=74); steady 0.75 (n=78) |
| idm_white_noise_mle | openacc | aci_online | 6 | 3 | 0.66 | 0.96 | braking 0.66 (n=31); steady 0.96 (n=205); other 0.85 (n=97) |
| idm_white_noise_mle | openacc | cqr | 6 | 5 | 0.39 | 0.85 | congested 0.70 (n=52); braking 0.56 (n=31); approaching 0.61 (n=66); free 0.39 (n=49); other 0.56 (n=97) |
| idm_white_noise_mle | openacc | raw | 6 | 6 | 0.05 | 0.15 | congested 0.10 (n=52); braking 0.15 (n=31); approaching 0.05 (n=66); free 0.08 (n=49); steady 0.06 (n=205); other 0.14 (n=97) |
| idm_white_noise_mle | openacc | static | 6 | 5 | 0.54 | 0.93 | congested 0.82 (n=52); braking 0.54 (n=31); approaching 0.70 (n=66); free 0.78 (n=49); other 0.76 (n=97) |
| idm_white_noise_mle | waymo | aci_online | 5 | 3 | 0.66 | 0.95 | congested 0.95 (n=232); braking 0.66 (n=50); approaching 0.83 (n=65) |
| idm_white_noise_mle | waymo | cqr | 5 | 5 | 0.66 | 0.82 | congested 0.72 (n=232); braking 0.68 (n=50); approaching 0.82 (n=65); steady 0.79 (n=58); other 0.66 (n=87) |
| idm_white_noise_mle | waymo | raw | 5 | 5 | 0.07 | 0.21 | congested 0.16 (n=232); braking 0.07 (n=50); approaching 0.07 (n=65); steady 0.21 (n=58); other 0.16 (n=87) |
| idm_white_noise_mle | waymo | static | 5 | 5 | 0.41 | 0.80 | congested 0.75 (n=232); braking 0.41 (n=50); approaching 0.69 (n=65); steady 0.80 (n=58); other 0.80 (n=87) |
| lstm_const_sigma | highd | aci_online | 5 | 2 | 0.73 | 0.96 | braking 0.73 (n=32); steady 0.96 (n=118) |
| lstm_const_sigma | highd | cqr | 5 | 3 | 0.76 | 0.96 | braking 0.76 (n=32); steady 0.96 (n=118); other 0.85 (n=115) |
| lstm_const_sigma | highd | raw | 5 | 5 | 0.26 | 0.49 | braking 0.26 (n=32); approaching 0.43 (n=133); free 0.42 (n=89); steady 0.49 (n=118); other 0.38 (n=115) |
| lstm_const_sigma | highd | static | 5 | 2 | 0.74 | 0.97 | braking 0.74 (n=32); steady 0.97 (n=118) |
| lstm_const_sigma | ngsim | aci_online | 4 | 1 | 0.72 | 0.92 | braking 0.72 (n=72) |
| lstm_const_sigma | ngsim | cqr | 4 | 4 | 0.25 | 0.49 | congested 0.49 (n=249); braking 0.25 (n=72); approaching 0.36 (n=95); other 0.39 (n=55) |
| lstm_const_sigma | ngsim | raw | 4 | 4 | 0.06 | 0.10 | congested 0.10 (n=249); braking 0.06 (n=72); approaching 0.06 (n=95); other 0.07 (n=55) |
| lstm_const_sigma | ngsim | static | 4 | 4 | 0.18 | 0.38 | congested 0.38 (n=249); braking 0.18 (n=72); approaching 0.25 (n=95); other 0.29 (n=55) |
| lstm_const_sigma | ngsim_us101 | aci_online | 5 | 1 | 0.76 | 0.95 | braking 0.76 (n=89) |
| lstm_const_sigma | ngsim_us101 | cqr | 5 | 5 | 0.24 | 0.52 | congested 0.52 (n=138); braking 0.24 (n=89); approaching 0.42 (n=74); steady 0.49 (n=78); other 0.48 (n=120) |
| lstm_const_sigma | ngsim_us101 | raw | 5 | 5 | 0.03 | 0.12 | congested 0.12 (n=138); braking 0.03 (n=89); approaching 0.09 (n=74); steady 0.11 (n=78); other 0.11 (n=120) |
| lstm_const_sigma | ngsim_us101 | static | 5 | 5 | 0.17 | 0.43 | congested 0.43 (n=138); braking 0.17 (n=89); approaching 0.32 (n=74); steady 0.43 (n=78); other 0.42 (n=120) |
| lstm_const_sigma | openacc | aci_online | 6 | 5 | 0.63 | 1.00 | congested 0.82 (n=52); braking 0.83 (n=31); approaching 0.85 (n=66); free 0.63 (n=49); steady 1.00 (n=205) |
| lstm_const_sigma | openacc | cqr | 6 | 5 | 0.47 | 0.91 | congested 0.64 (n=52); braking 0.47 (n=31); approaching 0.74 (n=66); free 0.51 (n=49); other 0.62 (n=97) |
| lstm_const_sigma | openacc | raw | 6 | 6 | 0.06 | 0.43 | congested 0.22 (n=52); braking 0.06 (n=31); approaching 0.28 (n=66); free 0.11 (n=49); steady 0.43 (n=205); other 0.21 (n=97) |
| lstm_const_sigma | openacc | static | 6 | 5 | 0.33 | 0.91 | congested 0.61 (n=52); braking 0.47 (n=31); approaching 0.63 (n=66); free 0.33 (n=49); other 0.58 (n=97) |
| lstm_const_sigma | waymo | aci_online | 5 | 2 | 0.79 | 0.99 | braking 0.79 (n=50); steady 0.99 (n=58) |
| lstm_const_sigma | waymo | cqr | 5 | 5 | 0.42 | 0.74 | congested 0.61 (n=232); braking 0.42 (n=50); approaching 0.65 (n=65); steady 0.74 (n=58); other 0.64 (n=87) |
| lstm_const_sigma | waymo | raw | 5 | 5 | 0.08 | 0.26 | congested 0.13 (n=232); braking 0.08 (n=50); approaching 0.18 (n=65); steady 0.26 (n=58); other 0.18 (n=87) |
| lstm_const_sigma | waymo | static | 5 | 5 | 0.30 | 0.69 | congested 0.48 (n=232); braking 0.30 (n=50); approaching 0.59 (n=65); steady 0.69 (n=58); other 0.56 (n=87) |
| mc_dropout | highd | aci_online | 5 | 1 | 0.86 | 0.99 | braking 0.99 (n=32) |
| mc_dropout | highd | cqr | 5 | 1 | 0.84 | 0.94 | other 0.84 (n=115) |
| mc_dropout | highd | raw | 5 | 5 | 0.47 | 0.82 | braking 0.82 (n=32); approaching 0.58 (n=133); free 0.56 (n=89); steady 0.47 (n=118); other 0.56 (n=115) |
| mc_dropout | highd | static | 5 | 2 | 0.85 | 0.99 | braking 0.99 (n=32); other 0.85 (n=115) |
| mc_dropout | ngsim | aci_online | 4 | 1 | 0.86 | 0.98 | approaching 0.98 (n=95) |
| mc_dropout | ngsim | cqr | 4 | 3 | 0.70 | 0.92 | congested 0.77 (n=249); braking 0.70 (n=72); other 0.81 (n=55) |
| mc_dropout | ngsim | raw | 4 | 4 | 0.59 | 0.82 | congested 0.59 (n=249); braking 0.59 (n=72); approaching 0.82 (n=95); other 0.67 (n=55) |
| mc_dropout | ngsim | static | 4 | 1 | 0.85 | 0.98 | approaching 0.98 (n=95) |
| mc_dropout | ngsim_us101 | aci_online | 5 | 1 | 0.86 | 0.95 | approaching 0.95 (n=74) |
| mc_dropout | ngsim_us101 | cqr | 5 | 4 | 0.71 | 0.89 | congested 0.77 (n=138); braking 0.71 (n=89); steady 0.77 (n=78); other 0.75 (n=120) |
| mc_dropout | ngsim_us101 | raw | 5 | 5 | 0.54 | 0.73 | congested 0.59 (n=138); braking 0.54 (n=89); approaching 0.73 (n=74); steady 0.59 (n=78); other 0.61 (n=120) |
| mc_dropout | ngsim_us101 | static | 5 | 1 | 0.87 | 0.96 | approaching 0.96 (n=74) |
| mc_dropout | openacc | aci_online | 6 | 3 | 0.81 | 0.95 | congested 0.83 (n=52); braking 0.81 (n=31); free 0.81 (n=49) |
| mc_dropout | openacc | cqr | 6 | 6 | 0.63 | 0.81 | congested 0.63 (n=52); braking 0.70 (n=31); approaching 0.81 (n=66); free 0.73 (n=49); steady 0.79 (n=205); other 0.66 (n=97) |
| mc_dropout | openacc | raw | 6 | 6 | 0.27 | 0.65 | congested 0.27 (n=52); braking 0.55 (n=31); approaching 0.59 (n=66); free 0.65 (n=49); steady 0.32 (n=205); other 0.49 (n=97) |
| mc_dropout | openacc | static | 6 | 6 | 0.68 | 0.82 | congested 0.68 (n=52); braking 0.78 (n=31); approaching 0.82 (n=66); free 0.75 (n=49); steady 0.75 (n=205); other 0.78 (n=97) |
| mc_dropout | waymo | aci_online | 5 | 0 | 0.88 | 0.93 | none |
| mc_dropout | waymo | cqr | 5 | 5 | 0.73 | 0.82 | congested 0.73 (n=232); braking 0.75 (n=50); approaching 0.82 (n=65); steady 0.76 (n=58); other 0.74 (n=87) |
| mc_dropout | waymo | raw | 5 | 5 | 0.51 | 0.59 | congested 0.52 (n=232); braking 0.55 (n=50); approaching 0.59 (n=65); steady 0.51 (n=58); other 0.55 (n=87) |
| mc_dropout | waymo | static | 5 | 2 | 0.82 | 0.88 | congested 0.83 (n=232); steady 0.82 (n=58) |
| mc_dropout__traj | highd | aci_online | 5 | 1 | 0.84 | 0.94 | other 0.84 (n=115) |
| mc_dropout__traj | highd | cqr | 5 | 1 | 0.86 | 0.96 | braking 0.96 (n=32) |
| mc_dropout__traj | highd | raw | 5 | 3 | 0.91 | 0.99 | braking 0.99 (n=32); approaching 0.97 (n=133); free 0.97 (n=89) |
| mc_dropout__traj | highd | static | 5 | 1 | 0.83 | 0.93 | other 0.83 (n=115) |
| mc_dropout__traj | ngsim | aci_online | 4 | 2 | 0.79 | 0.98 | braking 0.79 (n=72); approaching 0.98 (n=95) |
| mc_dropout__traj | ngsim | cqr | 4 | 3 | 0.67 | 0.94 | congested 0.70 (n=249); braking 0.67 (n=72); other 0.77 (n=55) |
| mc_dropout__traj | ngsim | raw | 4 | 3 | 0.73 | 0.98 | congested 0.82 (n=249); braking 0.73 (n=72); approaching 0.98 (n=95) |
| mc_dropout__traj | ngsim | static | 4 | 4 | 0.47 | 0.80 | congested 0.60 (n=249); braking 0.47 (n=72); approaching 0.80 (n=95); other 0.58 (n=55) |
| mc_dropout__traj | ngsim_us101 | aci_online | 5 | 0 | 0.87 | 0.94 | none |
| mc_dropout__traj | ngsim_us101 | cqr | 5 | 5 | 0.62 | 0.84 | congested 0.70 (n=138); braking 0.62 (n=89); approaching 0.84 (n=74); steady 0.70 (n=78); other 0.68 (n=120) |
| mc_dropout__traj | ngsim_us101 | raw | 5 | 4 | 0.77 | 0.91 | congested 0.83 (n=138); braking 0.77 (n=89); steady 0.79 (n=78); other 0.77 (n=120) |
| mc_dropout__traj | ngsim_us101 | static | 5 | 5 | 0.43 | 0.73 | congested 0.63 (n=138); braking 0.43 (n=89); approaching 0.73 (n=74); steady 0.63 (n=78); other 0.54 (n=120) |
| mc_dropout__traj | openacc | aci_online | 6 | 5 | 0.77 | 0.97 | congested 0.77 (n=52); braking 0.82 (n=31); free 0.78 (n=49); steady 0.97 (n=205); other 0.85 (n=97) |
| mc_dropout__traj | openacc | cqr | 6 | 6 | 0.52 | 0.81 | congested 0.52 (n=52); braking 0.64 (n=31); approaching 0.81 (n=66); free 0.75 (n=49); steady 0.71 (n=205); other 0.68 (n=97) |
| mc_dropout__traj | openacc | raw | 6 | 4 | 0.68 | 0.88 | congested 0.68 (n=52); braking 0.74 (n=31); free 0.78 (n=49); other 0.78 (n=97) |
| mc_dropout__traj | openacc | static | 6 | 6 | 0.51 | 0.77 | congested 0.51 (n=52); braking 0.56 (n=31); approaching 0.77 (n=66); free 0.71 (n=49); steady 0.69 (n=205); other 0.58 (n=97) |
| mc_dropout__traj | waymo | aci_online | 5 | 1 | 0.85 | 0.94 | braking 0.85 (n=50) |
| mc_dropout__traj | waymo | cqr | 5 | 4 | 0.70 | 0.87 | congested 0.70 (n=232); braking 0.76 (n=50); steady 0.77 (n=58); other 0.79 (n=87) |
| mc_dropout__traj | waymo | raw | 5 | 0 | 0.85 | 0.93 | none |
| mc_dropout__traj | waymo | static | 5 | 5 | 0.64 | 0.82 | congested 0.64 (n=232); braking 0.65 (n=50); approaching 0.82 (n=65); steady 0.72 (n=58); other 0.67 (n=87) |
| quantile_pidl | highd | aci_online | 5 | 1 | 0.84 | 0.94 | braking 0.84 (n=32) |
| quantile_pidl | highd | cqr | 5 | 2 | 0.84 | 0.97 | braking 0.84 (n=32); steady 0.97 (n=118) |
| quantile_pidl | highd | raw | 5 | 5 | 0.27 | 0.44 | braking 0.27 (n=32); approaching 0.41 (n=133); free 0.41 (n=89); steady 0.44 (n=118); other 0.39 (n=115) |
| quantile_pidl | highd | static | 5 | 2 | 0.85 | 0.96 | braking 0.85 (n=32); steady 0.96 (n=118) |
| quantile_pidl | ngsim | aci_online | 4 | 2 | 0.83 | 0.90 | braking 0.83 (n=72); other 0.84 (n=55) |
| quantile_pidl | ngsim | cqr | 4 | 4 | 0.30 | 0.61 | congested 0.61 (n=249); braking 0.30 (n=72); approaching 0.45 (n=95); other 0.32 (n=55) |
| quantile_pidl | ngsim | raw | 4 | 4 | 0.08 | 0.21 | congested 0.21 (n=249); braking 0.09 (n=72); approaching 0.12 (n=95); other 0.08 (n=55) |
| quantile_pidl | ngsim | static | 4 | 4 | 0.39 | 0.62 | congested 0.62 (n=249); braking 0.39 (n=72); approaching 0.47 (n=95); other 0.39 (n=55) |
| quantile_pidl | ngsim_us101 | aci_online | 5 | 0 | 0.87 | 0.91 | none |
| quantile_pidl | ngsim_us101 | cqr | 5 | 5 | 0.36 | 0.59 | congested 0.59 (n=138); braking 0.36 (n=89); approaching 0.49 (n=74); steady 0.46 (n=78); other 0.46 (n=120) |
| quantile_pidl | ngsim_us101 | raw | 5 | 5 | 0.09 | 0.16 | congested 0.16 (n=138); braking 0.09 (n=89); approaching 0.11 (n=74); steady 0.11 (n=78); other 0.12 (n=120) |
| quantile_pidl | ngsim_us101 | static | 5 | 5 | 0.44 | 0.63 | congested 0.63 (n=138); braking 0.44 (n=89); approaching 0.54 (n=74); steady 0.50 (n=78); other 0.51 (n=120) |
| quantile_pidl | openacc | aci_online | 6 | 3 | 0.82 | 0.94 | congested 0.84 (n=52); braking 0.82 (n=31); other 0.84 (n=97) |
| quantile_pidl | openacc | cqr | 6 | 5 | 0.48 | 0.88 | congested 0.67 (n=52); braking 0.48 (n=31); approaching 0.74 (n=66); free 0.73 (n=49); other 0.67 (n=97) |
| quantile_pidl | openacc | raw | 6 | 6 | 0.19 | 0.37 | congested 0.30 (n=52); braking 0.19 (n=31); approaching 0.28 (n=66); free 0.31 (n=49); steady 0.37 (n=205); other 0.26 (n=97) |
| quantile_pidl | openacc | static | 6 | 5 | 0.65 | 0.86 | congested 0.73 (n=52); braking 0.65 (n=31); approaching 0.79 (n=66); free 0.80 (n=49); other 0.72 (n=97) |
| quantile_pidl | waymo | aci_online | 5 | 1 | 0.84 | 0.92 | braking 0.84 (n=50) |
| quantile_pidl | waymo | cqr | 5 | 5 | 0.58 | 0.77 | congested 0.77 (n=232); braking 0.58 (n=50); approaching 0.75 (n=65); steady 0.76 (n=58); other 0.64 (n=87) |
| quantile_pidl | waymo | raw | 5 | 5 | 0.15 | 0.32 | congested 0.32 (n=232); braking 0.15 (n=50); approaching 0.27 (n=65); steady 0.21 (n=58); other 0.23 (n=87) |
| quantile_pidl | waymo | static | 5 | 5 | 0.62 | 0.78 | congested 0.78 (n=232); braking 0.62 (n=50); approaching 0.74 (n=65); steady 0.74 (n=58); other 0.65 (n=87) |

## Variants at 10 s: coverage, Winkler score of the finite intervals (m), share of infinite intervals

| model | set | variant | PICP | Winkler | infinite |
|---|---|---|---|---|---|
| deep_ensemble | highd | raw | 0.474 | 49.0 | 0.000 |
| deep_ensemble | highd | static | 0.901 | 27.4 | 0.000 |
| deep_ensemble | highd | cqr | 0.898 | 29.3 | 0.000 |
| deep_ensemble | highd | aci_online | 0.897 | 27.6 | 0.000 |
| deep_ensemble | ngsim | raw | 0.334 | 198.6 | 0.000 |
| deep_ensemble | ngsim | static | 0.799 | 108.7 | 0.000 |
| deep_ensemble | ngsim | cqr | 0.625 | 146.8 | 0.000 |
| deep_ensemble | ngsim | aci_online | 0.886 | 107.1 | 0.000 |
| deep_ensemble | ngsim_us101 | raw | 0.330 | 171.3 | 0.000 |
| deep_ensemble | ngsim_us101 | static | 0.791 | 85.3 | 0.000 |
| deep_ensemble | ngsim_us101 | cqr | 0.606 | 117.0 | 0.000 |
| deep_ensemble | ngsim_us101 | aci_online | 0.888 | 84.0 | 0.000 |
| deep_ensemble | waymo | raw | 0.324 | 110.4 | 0.000 |
| deep_ensemble | waymo | static | 0.782 | 55.1 | 0.000 |
| deep_ensemble | waymo | cqr | 0.719 | 64.8 | 0.000 |
| deep_ensemble | waymo | aci_online | 0.890 | 54.5 | 0.000 |
| deep_ensemble | openacc | raw | 0.403 | 299.3 | 0.000 |
| deep_ensemble | openacc | static | 0.803 | 163.1 | 0.000 |
| deep_ensemble | openacc | cqr | 0.718 | 259.6 | 0.000 |
| deep_ensemble | openacc | aci_online | 0.887 | 126.7 | 0.000 |
| deep_ensemble__traj | highd | raw | 0.782 | 32.6 | 0.000 |
| deep_ensemble__traj | highd | static | 0.892 | 31.2 | 0.000 |
| deep_ensemble__traj | highd | cqr | 0.890 | 29.4 | 0.000 |
| deep_ensemble__traj | highd | aci_online | 0.900 | 31.7 | 0.000 |
| deep_ensemble__traj | ngsim | raw | 0.583 | 128.5 | 0.000 |
| deep_ensemble__traj | ngsim | static | 0.762 | 104.3 | 0.000 |
| deep_ensemble__traj | ngsim | cqr | 0.679 | 114.6 | 0.000 |
| deep_ensemble__traj | ngsim | aci_online | 0.886 | 103.5 | 0.000 |
| deep_ensemble__traj | ngsim_us101 | raw | 0.539 | 116.9 | 0.000 |
| deep_ensemble__traj | ngsim_us101 | static | 0.725 | 90.2 | 0.000 |
| deep_ensemble__traj | ngsim_us101 | cqr | 0.641 | 100.1 | 0.000 |
| deep_ensemble__traj | ngsim_us101 | aci_online | 0.889 | 88.2 | 0.000 |
| deep_ensemble__traj | waymo | raw | 0.715 | 63.9 | 0.000 |
| deep_ensemble__traj | waymo | static | 0.846 | 61.9 | 0.000 |
| deep_ensemble__traj | waymo | cqr | 0.820 | 57.0 | 0.000 |
| deep_ensemble__traj | waymo | aci_online | 0.894 | 65.2 | 0.000 |
| deep_ensemble__traj | openacc | raw | 0.708 | 171.8 | 0.000 |
| deep_ensemble__traj | openacc | static | 0.840 | 128.1 | 0.000 |
| deep_ensemble__traj | openacc | cqr | 0.785 | 163.7 | 0.000 |
| deep_ensemble__traj | openacc | aci_online | 0.892 | 111.3 | 0.000 |
| diffusion_cf | highd | raw | 0.893 | 17.4 | 0.000 |
| diffusion_cf | highd | static | 0.894 | 17.4 | 0.000 |
| diffusion_cf | highd | cqr | 0.893 | 17.4 | 0.000 |
| diffusion_cf | highd | aci_online | 0.890 | 17.5 | 0.000 |
| diffusion_cf | ngsim | raw | 0.805 | 49.3 | 0.000 |
| diffusion_cf | ngsim | static | 0.808 | 49.2 | 0.000 |
| diffusion_cf | ngsim | cqr | 0.806 | 49.3 | 0.000 |
| diffusion_cf | ngsim | aci_online | 0.892 | 49.0 | 0.000 |
| diffusion_cf | ngsim_us101 | raw | 0.810 | 42.6 | 0.000 |
| diffusion_cf | ngsim_us101 | static | 0.810 | 42.6 | 0.000 |
| diffusion_cf | ngsim_us101 | cqr | 0.810 | 42.6 | 0.000 |
| diffusion_cf | ngsim_us101 | aci_online | 0.892 | 41.4 | 0.000 |
| diffusion_cf | waymo | raw | 0.840 | 30.9 | 0.000 |
| diffusion_cf | waymo | static | 0.840 | 30.8 | 0.000 |
| diffusion_cf | waymo | cqr | 0.840 | 30.8 | 0.000 |
| diffusion_cf | waymo | aci_online | 0.892 | 30.1 | 0.000 |
| diffusion_cf | openacc | raw | 0.929 | 23.9 | 0.000 |
| diffusion_cf | openacc | static | 0.930 | 23.9 | 0.000 |
| diffusion_cf | openacc | cqr | 0.930 | 23.9 | 0.000 |
| diffusion_cf | openacc | aci_online | 0.901 | 24.3 | 0.000 |
| gp_residual_idm | highd | raw | 0.270 | 63.7 | 0.000 |
| gp_residual_idm | highd | static | 0.907 | 32.8 | 0.000 |
| gp_residual_idm | highd | cqr | 0.900 | 28.1 | 0.000 |
| gp_residual_idm | highd | aci_online | 0.904 | 33.5 | 0.000 |
| gp_residual_idm | ngsim | raw | 0.172 | 70.9 | 0.000 |
| gp_residual_idm | ngsim | static | 0.821 | 28.3 | 0.000 |
| gp_residual_idm | ngsim | cqr | 0.893 | 30.3 | 0.000 |
| gp_residual_idm | ngsim | aci_online | 0.900 | 30.0 | 0.000 |
| gp_residual_idm | ngsim_us101 | raw | 0.158 | 71.8 | 0.000 |
| gp_residual_idm | ngsim_us101 | static | 0.836 | 27.3 | 0.000 |
| gp_residual_idm | ngsim_us101 | cqr | 0.894 | 30.6 | 0.000 |
| gp_residual_idm | ngsim_us101 | aci_online | 0.902 | 27.5 | 0.000 |
| gp_residual_idm | waymo | raw | 0.169 | 118.8 | 0.000 |
| gp_residual_idm | waymo | static | 0.744 | 56.0 | 0.000 |
| gp_residual_idm | waymo | cqr | 0.762 | 63.9 | 0.000 |
| gp_residual_idm | waymo | aci_online | 0.898 | 54.4 | 0.000 |
| gp_residual_idm | openacc | raw | 0.294 | 78.6 | 0.000 |
| gp_residual_idm | openacc | static | 0.848 | 41.9 | 0.000 |
| gp_residual_idm | openacc | cqr | 0.854 | 43.3 | 0.000 |
| gp_residual_idm | openacc | aci_online | 0.902 | 42.3 | 0.000 |
| gp_residual_idm__step | highd | raw | 0.268 | 64.1 | 0.000 |
| gp_residual_idm__step | highd | static | 0.906 | 32.8 | 0.000 |
| gp_residual_idm__step | highd | cqr | 0.900 | 28.0 | 0.000 |
| gp_residual_idm__step | highd | aci_online | 0.904 | 33.6 | 0.000 |
| gp_residual_idm__step | ngsim | raw | 0.162 | 72.1 | 0.000 |
| gp_residual_idm__step | ngsim | static | 0.797 | 29.2 | 0.000 |
| gp_residual_idm__step | ngsim | cqr | 0.890 | 30.6 | 0.000 |
| gp_residual_idm__step | ngsim | aci_online | 0.899 | 31.3 | 0.000 |
| gp_residual_idm__step | ngsim_us101 | raw | 0.157 | 72.9 | 0.000 |
| gp_residual_idm__step | ngsim_us101 | static | 0.820 | 29.0 | 0.000 |
| gp_residual_idm__step | ngsim_us101 | cqr | 0.892 | 31.2 | 0.000 |
| gp_residual_idm__step | ngsim_us101 | aci_online | 0.903 | 29.2 | 0.000 |
| gp_residual_idm__step | waymo | raw | 0.160 | 121.5 | 0.000 |
| gp_residual_idm__step | waymo | static | 0.714 | 60.4 | 0.000 |
| gp_residual_idm__step | waymo | cqr | 0.755 | 65.2 | 0.000 |
| gp_residual_idm__step | waymo | aci_online | 0.899 | 58.0 | 0.000 |
| gp_residual_idm__step | openacc | raw | 0.266 | 82.0 | 0.000 |
| gp_residual_idm__step | openacc | static | 0.803 | 44.5 | 0.000 |
| gp_residual_idm__step | openacc | cqr | 0.848 | 43.8 | 0.000 |
| gp_residual_idm__step | openacc | aci_online | 0.901 | 44.2 | 0.000 |
| hetero_mdn | highd | raw | 0.325 | 73.6 | 0.000 |
| hetero_mdn | highd | static | 0.913 | 33.8 | 0.000 |
| hetero_mdn | highd | cqr | 0.904 | 34.9 | 0.000 |
| hetero_mdn | highd | aci_online | 0.899 | 33.8 | 0.000 |
| hetero_mdn | ngsim | raw | 0.152 | 303.5 | 0.000 |
| hetero_mdn | ngsim | static | 0.552 | 187.3 | 0.000 |
| hetero_mdn | ngsim | cqr | 0.516 | 203.8 | 0.000 |
| hetero_mdn | ngsim | aci_online | 0.885 | 177.6 | 0.003 |
| hetero_mdn | ngsim_us101 | raw | 0.150 | 267.6 | 0.000 |
| hetero_mdn | ngsim_us101 | static | 0.563 | 152.1 | 0.000 |
| hetero_mdn | ngsim_us101 | cqr | 0.518 | 169.1 | 0.000 |
| hetero_mdn | ngsim_us101 | aci_online | 0.886 | 128.6 | 0.002 |
| hetero_mdn | waymo | raw | 0.231 | 151.3 | 0.000 |
| hetero_mdn | waymo | static | 0.745 | 74.7 | 0.000 |
| hetero_mdn | waymo | cqr | 0.718 | 79.5 | 0.000 |
| hetero_mdn | waymo | aci_online | 0.888 | 75.2 | 0.000 |
| hetero_mdn | openacc | raw | 0.276 | 276.6 | 0.000 |
| hetero_mdn | openacc | static | 0.792 | 136.7 | 0.000 |
| hetero_mdn | openacc | cqr | 0.666 | 206.0 | 0.000 |
| hetero_mdn | openacc | aci_online | 0.889 | 144.2 | 0.000 |
| hetero_mdn_mix3 | highd | raw | 0.297 | 81.6 | 0.000 |
| hetero_mdn_mix3 | highd | static | 0.908 | 35.1 | 0.000 |
| hetero_mdn_mix3 | highd | cqr | 0.913 | 36.4 | 0.000 |
| hetero_mdn_mix3 | highd | aci_online | 0.901 | 35.2 | 0.000 |
| hetero_mdn_mix3 | ngsim | raw | 0.152 | 359.2 | 0.000 |
| hetero_mdn_mix3 | ngsim | static | 0.557 | 221.2 | 0.000 |
| hetero_mdn_mix3 | ngsim | cqr | 0.463 | 237.4 | 0.000 |
| hetero_mdn_mix3 | ngsim | aci_online | 0.889 | 274.8 | 0.004 |
| hetero_mdn_mix3 | ngsim_us101 | raw | 0.154 | 314.2 | 0.000 |
| hetero_mdn_mix3 | ngsim_us101 | static | 0.593 | 171.9 | 0.000 |
| hetero_mdn_mix3 | ngsim_us101 | cqr | 0.480 | 194.9 | 0.000 |
| hetero_mdn_mix3 | ngsim_us101 | aci_online | 0.890 | 176.4 | 0.005 |
| hetero_mdn_mix3 | waymo | raw | 0.236 | 194.8 | 0.000 |
| hetero_mdn_mix3 | waymo | static | 0.727 | 108.0 | 0.000 |
| hetero_mdn_mix3 | waymo | cqr | 0.678 | 107.2 | 0.000 |
| hetero_mdn_mix3 | waymo | aci_online | 0.890 | 203.7 | 0.000 |
| hetero_mdn_mix3 | openacc | raw | 0.224 | 301.4 | 0.000 |
| hetero_mdn_mix3 | openacc | static | 0.772 | 133.6 | 0.000 |
| hetero_mdn_mix3 | openacc | cqr | 0.668 | 211.2 | 0.000 |
| hetero_mdn_mix3 | openacc | aci_online | 0.892 | 125.9 | 0.000 |
| idm_ar_noise | highd | raw | 0.757 | 36.9 | 0.000 |
| idm_ar_noise | highd | static | 0.928 | 51.7 | 0.000 |
| idm_ar_noise | highd | cqr | 0.924 | 33.8 | 0.000 |
| idm_ar_noise | highd | aci_online | 0.906 | 43.7 | 0.000 |
| idm_ar_noise | ngsim | raw | 0.331 | 177.7 | 0.000 |
| idm_ar_noise | ngsim | static | 0.618 | 130.4 | 0.000 |
| idm_ar_noise | ngsim | cqr | 0.605 | 141.4 | 0.000 |
| idm_ar_noise | ngsim | aci_online | 0.892 | 316.3 | 0.000 |
| idm_ar_noise | ngsim_us101 | raw | 0.456 | 121.8 | 0.000 |
| idm_ar_noise | ngsim_us101 | static | 0.751 | 85.9 | 0.000 |
| idm_ar_noise | ngsim_us101 | cqr | 0.688 | 93.4 | 0.000 |
| idm_ar_noise | ngsim_us101 | aci_online | 0.896 | 137.2 | 0.000 |
| idm_ar_noise | waymo | raw | 0.566 | 61.6 | 0.000 |
| idm_ar_noise | waymo | static | 0.863 | 55.2 | 0.000 |
| idm_ar_noise | waymo | cqr | 0.843 | 47.0 | 0.000 |
| idm_ar_noise | waymo | aci_online | 0.897 | 67.8 | 0.000 |
| idm_ar_noise | openacc | raw | 0.775 | 68.8 | 0.000 |
| idm_ar_noise | openacc | static | 0.913 | 72.1 | 0.000 |
| idm_ar_noise | openacc | cqr | 0.873 | 62.9 | 0.000 |
| idm_ar_noise | openacc | aci_online | 0.904 | 70.4 | 0.000 |
| idm_white_noise | highd | raw | 0.466 | 45.8 | 0.000 |
| idm_white_noise | highd | static | 0.898 | 31.9 | 0.000 |
| idm_white_noise | highd | cqr | 0.880 | 26.2 | 0.000 |
| idm_white_noise | highd | aci_online | 0.890 | 31.7 | 0.000 |
| idm_white_noise | ngsim | raw | 0.378 | 46.0 | 0.000 |
| idm_white_noise | ngsim | static | 0.906 | 26.4 | 0.000 |
| idm_white_noise | ngsim | cqr | 0.886 | 26.6 | 0.000 |
| idm_white_noise | ngsim | aci_online | 0.898 | 26.5 | 0.000 |
| idm_white_noise | ngsim_us101 | raw | 0.500 | 39.1 | 0.000 |
| idm_white_noise | ngsim_us101 | static | 0.932 | 28.8 | 0.000 |
| idm_white_noise | ngsim_us101 | cqr | 0.917 | 23.7 | 0.000 |
| idm_white_noise | ngsim_us101 | aci_online | 0.897 | 26.4 | 0.000 |
| idm_white_noise | waymo | raw | 0.383 | 85.6 | 0.000 |
| idm_white_noise | waymo | static | 0.854 | 41.5 | 0.000 |
| idm_white_noise | waymo | cqr | 0.790 | 57.7 | 0.000 |
| idm_white_noise | waymo | aci_online | 0.903 | 42.1 | 0.000 |
| idm_white_noise | openacc | raw | 0.304 | 92.7 | 0.000 |
| idm_white_noise | openacc | static | 0.911 | 40.1 | 0.000 |
| idm_white_noise | openacc | cqr | 0.778 | 57.6 | 0.000 |
| idm_white_noise | openacc | aci_online | 0.898 | 41.0 | 0.000 |
| idm_white_noise_mle | highd | raw | 0.250 | 81.4 | 0.000 |
| idm_white_noise_mle | highd | static | 0.915 | 48.4 | 0.000 |
| idm_white_noise_mle | highd | cqr | 0.883 | 34.6 | 0.000 |
| idm_white_noise_mle | highd | aci_online | 0.900 | 46.3 | 0.000 |
| idm_white_noise_mle | ngsim | raw | 0.158 | 104.8 | 0.000 |
| idm_white_noise_mle | ngsim | static | 0.795 | 43.6 | 0.000 |
| idm_white_noise_mle | ngsim | cqr | 0.823 | 41.7 | 0.000 |
| idm_white_noise_mle | ngsim | aci_online | 0.891 | 55.6 | 0.000 |
| idm_white_noise_mle | ngsim_us101 | raw | 0.159 | 103.8 | 0.000 |
| idm_white_noise_mle | ngsim_us101 | static | 0.784 | 49.0 | 0.000 |
| idm_white_noise_mle | ngsim_us101 | cqr | 0.816 | 38.3 | 0.000 |
| idm_white_noise_mle | ngsim_us101 | aci_online | 0.888 | 65.4 | 0.000 |
| idm_white_noise_mle | waymo | raw | 0.144 | 158.4 | 0.000 |
| idm_white_noise_mle | waymo | static | 0.724 | 65.2 | 0.000 |
| idm_white_noise_mle | waymo | cqr | 0.730 | 81.3 | 0.000 |
| idm_white_noise_mle | waymo | aci_online | 0.893 | 87.1 | 0.000 |
| idm_white_noise_mle | openacc | raw | 0.083 | 188.0 | 0.000 |
| idm_white_noise_mle | openacc | static | 0.813 | 65.0 | 0.000 |
| idm_white_noise_mle | openacc | cqr | 0.684 | 86.9 | 0.000 |
| idm_white_noise_mle | openacc | aci_online | 0.891 | 67.0 | 0.000 |
| lstm_const_sigma | highd | raw | 0.417 | 61.4 | 0.000 |
| lstm_const_sigma | highd | static | 0.906 | 34.3 | 0.000 |
| lstm_const_sigma | highd | cqr | 0.892 | 32.6 | 0.000 |
| lstm_const_sigma | highd | aci_online | 0.900 | 34.5 | 0.000 |
| lstm_const_sigma | ngsim | raw | 0.084 | 439.4 | 0.000 |
| lstm_const_sigma | ngsim | static | 0.313 | 368.9 | 0.000 |
| lstm_const_sigma | ngsim | cqr | 0.410 | 344.0 | 0.000 |
| lstm_const_sigma | ngsim | aci_online | 0.888 | 617.8 | 0.006 |
| lstm_const_sigma | ngsim_us101 | raw | 0.096 | 323.6 | 0.000 |
| lstm_const_sigma | ngsim_us101 | static | 0.365 | 246.6 | 0.000 |
| lstm_const_sigma | ngsim_us101 | cqr | 0.441 | 231.1 | 0.000 |
| lstm_const_sigma | ngsim_us101 | aci_online | 0.890 | 247.9 | 0.021 |
| lstm_const_sigma | waymo | raw | 0.155 | 179.8 | 0.000 |
| lstm_const_sigma | waymo | static | 0.514 | 117.9 | 0.000 |
| lstm_const_sigma | waymo | cqr | 0.615 | 105.5 | 0.000 |
| lstm_const_sigma | waymo | aci_online | 0.886 | 131.2 | 0.005 |
| lstm_const_sigma | openacc | raw | 0.290 | 301.3 | 0.000 |
| lstm_const_sigma | openacc | static | 0.693 | 261.2 | 0.000 |
| lstm_const_sigma | openacc | cqr | 0.737 | 252.1 | 0.000 |
| lstm_const_sigma | openacc | aci_online | 0.887 | 687.8 | 0.002 |
| mc_dropout | highd | raw | 0.557 | 63.6 | 0.000 |
| mc_dropout | highd | static | 0.882 | 52.7 | 0.000 |
| mc_dropout | highd | cqr | 0.886 | 44.4 | 0.000 |
| mc_dropout | highd | aci_online | 0.895 | 53.6 | 0.000 |
| mc_dropout | ngsim | raw | 0.657 | 160.6 | 0.000 |
| mc_dropout | ngsim | static | 0.891 | 168.6 | 0.000 |
| mc_dropout | ngsim | cqr | 0.800 | 142.0 | 0.000 |
| mc_dropout | ngsim | aci_online | 0.898 | 214.6 | 0.000 |
| mc_dropout | ngsim_us101 | raw | 0.608 | 161.7 | 0.000 |
| mc_dropout | ngsim_us101 | static | 0.916 | 150.6 | 0.000 |
| mc_dropout | ngsim_us101 | cqr | 0.773 | 135.4 | 0.000 |
| mc_dropout | ngsim_us101 | aci_online | 0.904 | 144.0 | 0.000 |
| mc_dropout | waymo | raw | 0.534 | 144.5 | 0.000 |
| mc_dropout | waymo | static | 0.842 | 113.2 | 0.000 |
| mc_dropout | waymo | cqr | 0.746 | 115.3 | 0.000 |
| mc_dropout | waymo | aci_online | 0.895 | 134.6 | 0.000 |
| mc_dropout | openacc | raw | 0.430 | 288.3 | 0.000 |
| mc_dropout | openacc | static | 0.759 | 240.2 | 0.000 |
| mc_dropout | openacc | cqr | 0.740 | 247.1 | 0.000 |
| mc_dropout | openacc | aci_online | 0.891 | 250.8 | 0.000 |
| mc_dropout__traj | highd | raw | 0.950 | 105.8 | 0.000 |
| mc_dropout__traj | highd | static | 0.888 | 80.4 | 0.000 |
| mc_dropout__traj | highd | cqr | 0.895 | 102.1 | 0.000 |
| mc_dropout__traj | highd | aci_online | 0.898 | 83.7 | 0.000 |
| mc_dropout__traj | ngsim | raw | 0.852 | 102.8 | 0.000 |
| mc_dropout__traj | ngsim | static | 0.627 | 167.9 | 0.000 |
| mc_dropout__traj | ngsim | cqr | 0.759 | 110.6 | 0.000 |
| mc_dropout__traj | ngsim | aci_online | 0.887 | 104.5 | 0.004 |
| mc_dropout__traj | ngsim_us101 | raw | 0.812 | 117.9 | 0.000 |
| mc_dropout__traj | ngsim_us101 | static | 0.587 | 193.8 | 0.000 |
| mc_dropout__traj | ngsim_us101 | cqr | 0.701 | 132.0 | 0.000 |
| mc_dropout__traj | ngsim_us101 | aci_online | 0.892 | 112.2 | 0.001 |
| mc_dropout__traj | waymo | raw | 0.872 | 103.0 | 0.000 |
| mc_dropout__traj | waymo | static | 0.681 | 139.0 | 0.000 |
| mc_dropout__traj | waymo | cqr | 0.756 | 108.8 | 0.000 |
| mc_dropout__traj | waymo | aci_online | 0.893 | 119.2 | 0.004 |
| mc_dropout__traj | openacc | raw | 0.811 | 257.2 | 0.000 |
| mc_dropout__traj | openacc | static | 0.655 | 284.7 | 0.000 |
| mc_dropout__traj | openacc | cqr | 0.698 | 268.1 | 0.000 |
| mc_dropout__traj | openacc | aci_online | 0.888 | 268.3 | 0.001 |
| quantile_pidl | highd | raw | 0.400 | 53.4 | 0.000 |
| quantile_pidl | highd | static | 0.919 | 27.4 | 0.000 |
| quantile_pidl | highd | cqr | 0.921 | 28.5 | 0.000 |
| quantile_pidl | highd | aci_online | 0.902 | 27.3 | 0.000 |
| quantile_pidl | ngsim | raw | 0.153 | 251.7 | 0.000 |
| quantile_pidl | ngsim | static | 0.525 | 149.1 | 0.000 |
| quantile_pidl | ngsim | cqr | 0.493 | 164.0 | 0.000 |
| quantile_pidl | ngsim | aci_online | 0.884 | 128.9 | 0.004 |
| quantile_pidl | ngsim_us101 | raw | 0.124 | 232.3 | 0.000 |
| quantile_pidl | ngsim_us101 | static | 0.532 | 124.2 | 0.000 |
| quantile_pidl | ngsim_us101 | cqr | 0.482 | 140.6 | 0.000 |
| quantile_pidl | ngsim_us101 | aci_online | 0.889 | 98.4 | 0.002 |
| quantile_pidl | waymo | raw | 0.265 | 124.9 | 0.000 |
| quantile_pidl | waymo | static | 0.726 | 65.4 | 0.000 |
| quantile_pidl | waymo | cqr | 0.719 | 65.1 | 0.000 |
| quantile_pidl | waymo | aci_online | 0.890 | 65.7 | 0.001 |
| quantile_pidl | openacc | raw | 0.316 | 128.3 | 0.000 |
| quantile_pidl | openacc | static | 0.791 | 68.5 | 0.000 |
| quantile_pidl | openacc | cqr | 0.760 | 77.4 | 0.000 |
| quantile_pidl | openacc | aci_online | 0.892 | 68.8 | 0.000 |
