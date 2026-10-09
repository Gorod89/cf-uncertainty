"""Project-wide constants (see docs/design.md, section 1)."""

from __future__ import annotations

DT = 0.1  # s, sampling step of every event and rollout
HIST_LEN = 30  # samples in the state history (3 s)
SEQ_LEN = 50  # samples in a DiffusionCF acceleration sequence (5 s)
MEAN_TARGET_STEPS = 10  # samples averaged by the target mean_1s = (v[t+10] - v[t]) / (10 DT) (design.md, 11.1)
FEATURES = ("s", "dv", "v")  # order of the last axis of state tensors; dv = v - v_lead
N_FEATURES = len(FEATURES)
A_MIN = -8.0  # m/s^2
A_MAX = 4.0  # m/s^2
S_EPS = 0.1  # m, lower bound of the gap inside IDM to avoid division blow-up
