# Follow-up E3-pooled: platoons with the models of the pooled domain

Date: 2026-10-08. Contract: `docs/design.md`, section 17 (decision of the paper review: repeat E3 with
the checkpoints of the pooled domain). Question: is the closed-loop instability of the one-step
networks, and the platoon verdict of H3.2, a property of the models or of the narrow training domain
of highD? Two new signal variants of `artifacts/e3`: `lodo_openacc` (checkpoints of
`artifacts/e6_lodo_openacc`, OpenACC never seen in training) and `pool` (`artifacts/e6_pool`, all five
sets), target `a` for every model (QuantilePIDL included, `hold_steps` 1), leader profiles re-smoothed
with the matched OpenACC window (61 samples), everything else as in sections 13-14.

## 1. Commands

```bash
cd cf-uncertainty && export PYTHONUTF8=1
./.venv/Scripts/python.exe scripts/run_e3.py --dry-run --quiet                   # raw/harmonised 1800 done, pool/lodo_openacc 1800 missing each
bash scripts/run_e3_retry.sh artifacts/e3/logs/run_pooled.out 8 --variants lodo_openacc pool --quiet   # 1 h 23 min, 0 failures
./.venv/Scripts/python.exe scripts/analyze_e3.py --root artifacts/e3 --quiet
./.venv/Scripts/python.exe scripts/make_report.py
./.venv/Scripts/python.exe -m pytest -m "not slow"                                # 530 passed, 24 deselected
```

## 2. What was built

Per-variant model overrides and profile windows in the E3 runner (unit specs carry target, hold,
checkpoint directory, profile window and the digest of the re-smoothed profiles), `resmooth_profiles`
(same files, start samples and lengths; the window-11 re-derivation reproduces the stored statistics
to 5e-5), `scripts/run_e3.py --dry-run`, variant labels in the figures and the report assembly;
9 new tests. The raw and harmonised units of `artifacts/e3` were verified untouched.

## 3. Key numbers (extended set, 60 profiles; `artifacts/e3/tables/h32.csv`, `collisions.csv`, `growth.csv`)

### 3.1 H3.2 per driver mode, reduction of the amplitude CRPS against `idm_white_noise` of the same domain

| model | recorded (highD only) | harmonised (highD only) | OpenACC held out | pooled (OpenACC seen) |
|---|---|---|---|---|
| diffusion_cf, ACC (47) | 0.34 [0.09, 0.50] inconclusive | 0.40 [0.19, 0.53] inconclusive | **0.47 [0.39, 0.53] supported** | **0.56 [0.45, 0.64] supported** |
| diffusion_cf, Human (12) | -0.84 refuted | -0.84 refuted | -0.48 [-0.85, -0.16] refuted | -0.90 [-1.36, -0.48] refuted |
| diffusion_cf, all (60) | 0.25 [0.01, 0.44] inconclusive | 0.31 [0.10, 0.46] inconclusive | 0.40 [0.30, 0.48] supported | 0.46 [0.32, 0.56] supported |
| gp_residual_idm, ACC | 0.15 [-0.08, 0.35] | 0.14 [-0.10, 0.33] | 0.19 [-0.03, 0.38] | 0.21 [0.03, 0.36] (all inconclusive) |
| quantile_pidl, ACC | -0.59 [-1.03, -0.28] refuted | -0.37 [-0.81, -0.04] refuted | -0.16 [-0.55, 0.12] inconclusive | -0.22 [-0.61, 0.06] inconclusive |
| deep_ensemble, ACC | -5.9 refuted | -13.9 refuted | -9.2 refuted | -3.3 refuted |
| lstm_const_sigma, ACC | -15.8 refuted | -2.7 refuted | -20.5 refuted | -6.3 refuted |
| reference CRPS, ACC / Human, m/s | 3.47 / 1.11 | 3.37 / 1.10 | 3.54 / 1.05 | 3.53 / 1.06 |

DiffusionCF CRPS on the ACC platoons: 2.30 (recorded) -> 1.88 (OpenACC held out) -> 1.54 m/s (pooled).

### 3.2 Collisions (share of rollouts, vehicle 1 / 5 / 20; burn-in of vehicle 1) and growth exponent

| model | recorded | OpenACC held out | pooled | beta: recorded / held out / pooled (measured 0.12) |
|---|---|---|---|---|
| idm_white_noise | 0 / 0 / 0; 0 | 0 / 0 / 0; 0 | 0 / 0 / 0; 0 | -0.01 / -0.03 / -0.03 |
| diffusion_cf | 0.08 / 0.14 / 0.54; 0.001 | 0.02 / 0.05 / 0.23; 0 | 0.01 / 0.04 / 0.32; 0 | 0.19 / 0.10 / 0.17 |
| gp_residual_idm | 0.04 / 0.05 / 0.02; 0 | 0.14 / 0.10 / 0.05; 0 | 0.10 / 0.20 / 0.05; 0 | -0.19 / -0.25 / -0.16 |
| quantile_pidl | 0.49 / 0.83 / 0.92; 0.086 | 0.21 / 0.60 / 0.69; 0.108 | 0.21 / 0.75 / 0.78; 0.069 | -0.05 / 0.04 / 0.07 |
| deep_ensemble | 0.60 / 0.80 / 0.84; 0.270 | 0.52 / 0.73 / 0.74; 0.245 | 0.48 / 0.76 / 0.81; 0.158 | -0.05 / 0.02 / -0.10 |
| lstm_const_sigma | 0.52 / 0.67 / 0.82; 0.110 | 0.59 / 0.66 / 0.71; 0.297 | 0.45 / 0.62 / 0.73; 0.018 | -0.22 / 0.09 / -0.10 |

Raw amplitude-interval coverage and its corrections behave as on the recorded signals (static
leave-one-out 0.90-0.91, ACI 0.83-0.90 for every variant).

## 4. Findings

1. **H3.2 is supported for the diffusion model once its training domain is broad**: with OpenACC
   never seen, the reduction of the amplitude CRPS on the ACC platoons is 0.47 [0.39, 0.53] (threshold
   0.20 on the lower bound), 0.56 [0.45, 0.64] with OpenACC in training, and 0.40-0.46 on the pooled
   60 profiles; the human-driven platoons stay refuted for every learned model (the reference reaches
   1.05-1.1 m/s there). Its collisions fall to 1-2 % at the first follower and 23-32 % at the twentieth
   (recorded signals: 8 % and 54 %), its growth exponent to 0.10-0.17 against the measured 0.12.
2. **The closed-loop instability of the one-step networks is a property of the models** (reading of
   17.2): the LSTM and the kept-member ensemble collide at the first follower in 45-59 % of the
   rollouts whatever domain they were trained on (recorded 52-60 %); the quantile network improves to
   21 % at the first follower but 69-78 % by the twentieth, and moves from refuted to inconclusive.
3. The GP residual stays at the reference in every variant (0.14-0.21, inconclusive); the physics
   backbone keeps it on the road and the GP residual adds nothing at the platoon scale.

## 5. Deviations

Initial gaps at the variant's own re-smoothed initial speed (three Vicolungo profiles move by
0.1-4.2 m); re-smoothed profiles stored with provenance and recomputed at every run start; the pilot
ran in a separate root (deleted); QuantilePIDL on target `a` in the new variants (design.md 17.3).
The reference `idm_white_noise` of a pooled variant has its noise fitted on the pooled windows (its
CRPS 3.53-3.54 against 3.47 m/s on the recorded signals).
