### Target `a`, highd: coverage of the 90 % gap interval at 1/3/5/10 s

| model | PICP raw | PICP static | PICP ACI frozen | PICP ACI online | median width raw, m | median width ACI online, m | collided by 10 s | CRPS at 10 s, m |
|---|---|---|---|---|---|---|---|---|
| diffusion_cf_hist | 0.72 0.93 0.92 0.91 | 0.93 0.91 0.91 0.90 | 0.94 0.94 0.94 0.91 | 0.90 0.91 0.90 0.90 | 0.13 2.22 8.36 36.24 | 0.24 1.98 7.73 34.60 | 0.009 | 4.13 |
| lstm_hetero_frozen | 0.19 0.49 0.42 0.37 | 0.94 0.91 0.92 0.90 | 0.95 0.96 0.94 0.89 | 0.91 0.91 0.91 0.90 | 0.02 0.38 1.31 5.52 | 0.19 1.28 4.70 20.33 | 0.009 | 4.41 |

### Target `a`, ngsim: coverage of the 90 % gap interval at 1/3/5/10 s

| model | PICP raw | PICP static | PICP ACI frozen | PICP ACI online | median width raw, m | median width ACI online, m | collided by 10 s | CRPS at 10 s, m |
|---|---|---|---|---|---|---|---|---|
| diffusion_cf_hist | 0.42 0.60 0.72 0.79 | 0.75 0.56 0.70 0.78 | 0.77 0.62 0.75 0.79 | 0.90 0.90 0.90 0.89 | 0.56 7.32 19.68 41.34 | 1.89 15.96 32.18 51.87 | 0.230 | 12.22 |
| lstm_hetero_frozen | 0.08 0.11 0.13 0.24 | 0.47 0.26 0.42 0.69 | 0.50 0.32 0.46 0.68 | 0.89 0.89 0.89 0.89 | 0.04 0.73 2.57 10.56 | 1.50 16.24 30.40 78.35 | 0.325 | 17.14 |

### Target `a`, openacc: coverage of the 90 % gap interval at 1/3/5/10 s

| model | PICP raw | PICP static | PICP ACI frozen | PICP ACI online | median width raw, m | median width ACI online, m | collided by 10 s | CRPS at 10 s, m |
|---|---|---|---|---|---|---|---|---|
| diffusion_cf_hist | 0.59 0.79 0.82 0.80 | 0.80 0.76 0.81 0.79 | 0.81 0.80 0.84 0.79 | 0.89 0.90 0.90 0.89 | 0.21 3.48 12.36 39.77 | 0.81 4.96 15.93 52.83 | 0.044 | 17.38 |
| lstm_hetero_frozen | 0.19 0.33 0.35 0.35 | 0.71 0.70 0.79 0.83 | 0.73 0.77 0.83 0.82 | 0.89 0.90 0.90 0.89 | 0.03 0.44 1.47 5.80 | 0.68 3.63 9.07 30.23 | 0.064 | 11.63 |

### Target `a`: one-step coverage, raw -> split conformal (absolute score, calibrated on highD validation)

| model | highd | ngsim | ngsim_us101 | waymo | openacc | ngsim_fn |
|---|---|---|---|---|---|---|
| lstm_const_sigma | 0.925 -> 0.903 | 0.081 -> 0.074 | 0.070 -> 0.064 | 0.306 -> 0.281 | 0.465 -> 0.438 | 0.073 -> 0.066 |
| hetero_mdn | 0.908 -> 0.905 | 0.187 -> 0.062 | 0.137 -> 0.063 | 0.502 -> 0.262 | 0.644 -> 0.424 | 0.143 -> 0.065 |
| diffusion_cf_hist | 0.976 -> 0.906 | 0.460 -> 0.149 | 0.443 -> 0.141 | 0.774 -> 0.420 | 0.831 -> 0.620 | 0.447 -> 0.151 |
| lstm_hetero_frozen | 0.913 -> 0.903 | 0.154 -> 0.074 | 0.113 -> 0.064 | 0.436 -> 0.281 | 0.580 -> 0.438 | 0.112 -> 0.066 |

### Target `a`: one-step coverage, raw -> split conformal (normalised score, calibrated on highD validation)

| model | highd | ngsim | ngsim_us101 | waymo | openacc | ngsim_fn |
|---|---|---|---|---|---|---|
| lstm_const_sigma | 0.925 -> 0.903 | 0.081 -> 0.074 | 0.070 -> 0.064 | 0.306 -> 0.281 | 0.465 -> 0.438 | 0.073 -> 0.066 |
| hetero_mdn | 0.908 -> 0.899 | 0.187 -> 0.182 | 0.137 -> 0.132 | 0.502 -> 0.489 | 0.644 -> 0.634 | 0.143 -> 0.138 |
| diffusion_cf_hist | 0.976 -> 0.905 | 0.460 -> 0.311 | 0.443 -> 0.299 | 0.774 -> 0.604 | 0.831 -> 0.708 | 0.447 -> 0.305 |
| lstm_hetero_frozen | 0.913 -> 0.899 | 0.154 -> 0.148 | 0.113 -> 0.107 | 0.436 -> 0.419 | 0.580 -> 0.564 | 0.112 -> 0.106 |

### Target `a`: H3.4 (one-step conformal on highD / lowest raw rollout coverage / lowest coverage after ACI online: verdict by point estimates (by intervals))

| model | highd | ngsim | openacc |
|---|---|---|---|
| diffusion_cf_hist | 0.905 / 0.72 @ 1 s / 0.90: supported (supported) | 0.905 / 0.42 @ 1 s / 0.89: supported (supported) | 0.905 / 0.59 @ 1 s / 0.89: supported (supported) |
| lstm_hetero_frozen | 0.899 / 0.19 @ 1 s / 0.90: supported (supported) | 0.899 / 0.08 @ 1 s / 0.89: supported (inconclusive) | 0.899 / 0.19 @ 1 s / 0.89: supported (inconclusive) |

### Target `a`: share of rollouts with a collision within 10 s

| model | highd | highd_val | ngsim | openacc |
|---|---|---|---|---|
| diffusion_cf_hist | 0.009 | 0.009 | 0.230 | 0.044 |
| lstm_hetero_frozen | 0.009 | 0.014 | 0.325 | 0.064 |

### Target `a`: ACI online, sensitivity to gamma (range of the coverage over the models; median over the models of the median width, m)

| data set | gamma | PICP 1 s | PICP 3 s | PICP 5 s | PICP 10 s | width 1 s | width 3 s | width 5 s | width 10 s |
|---|---|---|---|---|---|---|---|---|---|
| highd | 0.005 | 0.908-0.917 | 0.906-0.913 | 0.900-0.910 | 0.891-0.894 | 0.22 | 1.62 | 6.18 | 26.88 |
| highd | 0.01 | 0.904-0.910 | 0.906-0.910 | 0.902-0.909 | 0.896-0.898 | 0.22 | 1.63 | 6.21 | 27.46 |
| highd | 0.02 | 0.902-0.905 | 0.903-0.906 | 0.902-0.908 | 0.899-0.900 | 0.22 | 1.65 | 6.32 | 28.09 |
| highd | 0.05 | 0.901-0.902 | 0.902-0.904 | 0.901-0.904 | 0.900-0.902 | 0.23 | 1.77 | 6.61 | 29.58 |
| ngsim | 0.005 | 0.877-0.891 | 0.883-0.886 | 0.877-0.887 | 0.873-0.883 | 1.62 | 14.53 | 30.26 | 59.99 |
| ngsim | 0.01 | 0.887-0.898 | 0.894-0.896 | 0.889-0.898 | 0.887-0.894 | 1.69 | 16.10 | 31.29 | 65.11 |
| ngsim | 0.02 | 0.893-0.900 | 0.898-0.899 | 0.897-0.902 | 0.894-0.900 | 1.81 | 18.10 | 34.78 | 70.40 |
| ngsim | 0.05 | 0.897-0.900 | 0.899-0.900 | 0.898-0.900 | 0.898-0.901 | 2.43 | 27.15 | 52.12 | 121.68 |

### Target `a`, highd: ACI at the ends of the level range (R1) at 1/3/5/10 s

Share of the intervals that are the whole line (level >= 1, Gibbs and Candes), per cent; median width of the finite intervals and of all intervals (inf when more than half are infinite), m.

| model | infinite % ACI frozen | infinite % ACI online | median finite width ACI online, m | median width ACI online, m |
|---|---|---|---|---|
| diffusion_cf_hist | 0.0 0.0 0.0 0.0 | 0.0 0.0 0.0 0.0 | 0.24 1.98 7.73 34.60 | 0.24 1.98 7.73 34.60 |
| lstm_hetero_frozen | 0.0 0.0 0.0 0.0 | 0.0 0.0 0.0 0.0 | 0.19 1.28 4.70 20.33 | 0.19 1.28 4.70 20.33 |

### Target `a`, highd: information budget of the corrections (R8), PICP at 1/3/5/10 s

static / ACI frozen: no label of the evaluated set; static prequential: the static correction recalibrated on every label seen so far (the labels of ACI online, no learning rate); ACI online: random order, label after each event; chronological: events in time order; delayed: a label used only a horizon after its interval was issued. First 50 / 100: coverage of delayed ACI over the first events of the stream at 10 s; delay: mean events issued before a label arrives at 10 s.

| model | static | static prequential | ACI frozen | ACI online | ACI chronological | ACI chronological, delayed | static prequential, delayed | first 50 / 100, 10 s | delay, events |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf_hist | 0.93 0.91 0.91 0.90 | 0.92 0.91 0.91 0.90 | 0.94 0.94 0.94 0.91 | 0.90 0.91 0.90 0.90 | 0.91 0.91 0.91 0.90 | 0.91 0.91 0.91 0.90 | 0.92 0.91 0.91 0.90 | 0.92 / 0.90 | 10.0 |
| lstm_hetero_frozen | 0.94 0.91 0.92 0.90 | 0.93 0.90 0.92 0.90 | 0.95 0.96 0.94 0.89 | 0.91 0.91 0.91 0.90 | 0.92 0.91 0.91 0.90 | 0.92 0.91 0.91 0.90 | 0.93 0.90 0.91 0.90 | 0.89 / 0.90 | 10.0 |

### Target `a`, ngsim: ACI at the ends of the level range (R1) at 1/3/5/10 s

Share of the intervals that are the whole line (level >= 1, Gibbs and Candes), per cent; median width of the finite intervals and of all intervals (inf when more than half are infinite), m.

| model | infinite % ACI frozen | infinite % ACI online | median finite width ACI online, m | median width ACI online, m |
|---|---|---|---|---|
| diffusion_cf_hist | 0.0 0.0 0.0 0.0 | 0.0 0.2 0.0 0.0 | 1.89 15.81 32.14 51.87 | 1.89 15.96 32.18 51.87 |
| lstm_hetero_frozen | 0.0 0.0 0.0 0.0 | 0.5 2.5 0.5 0.0 | 1.49 15.46 30.27 78.16 | 1.50 16.24 30.40 78.35 |

### Target `a`, ngsim: information budget of the corrections (R8), PICP at 1/3/5/10 s

static / ACI frozen: no label of the evaluated set; static prequential: the static correction recalibrated on every label seen so far (the labels of ACI online, no learning rate); ACI online: random order, label after each event; chronological: events in time order; delayed: a label used only a horizon after its interval was issued. First 50 / 100: coverage of delayed ACI over the first events of the stream at 10 s; delay: mean events issued before a label arrives at 10 s.

| model | static | static prequential | ACI frozen | ACI online | ACI chronological | ACI chronological, delayed | static prequential, delayed | first 50 / 100, 10 s | delay, events |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf_hist | 0.75 0.56 0.70 0.78 | 0.81 0.73 0.79 0.83 | 0.77 0.62 0.75 0.79 | 0.90 0.90 0.90 0.89 | 0.90 0.90 0.90 0.89 | 0.89 0.90 0.90 0.89 | 0.82 0.74 0.79 0.83 | 0.82 / 0.86 | 1.8 |
| lstm_hetero_frozen | 0.47 0.26 0.42 0.69 | 0.69 0.63 0.69 0.78 | 0.50 0.32 0.46 0.68 | 0.89 0.89 0.89 0.89 | 0.89 0.90 0.89 0.89 | 0.89 0.90 0.89 0.89 | 0.71 0.63 0.68 0.79 | 0.76 / 0.83 | 1.8 |

### Target `a`, openacc: ACI at the ends of the level range (R1) at 1/3/5/10 s

Share of the intervals that are the whole line (level >= 1, Gibbs and Candes), per cent; median width of the finite intervals and of all intervals (inf when more than half are infinite), m.

| model | infinite % ACI frozen | infinite % ACI online | median finite width ACI online, m | median width ACI online, m |
|---|---|---|---|---|
| diffusion_cf_hist | 0.0 0.0 0.0 0.0 | 0.0 0.0 0.0 0.0 | 0.81 4.96 15.93 52.83 | 0.81 4.96 15.93 52.83 |
| lstm_hetero_frozen | 0.0 0.0 0.0 0.0 | 0.0 0.1 0.0 0.0 | 0.68 3.63 9.07 30.23 | 0.68 3.63 9.07 30.23 |

### Target `a`, openacc: information budget of the corrections (R8), PICP at 1/3/5/10 s

static / ACI frozen: no label of the evaluated set; static prequential: the static correction recalibrated on every label seen so far (the labels of ACI online, no learning rate); ACI online: random order, label after each event; chronological: events in time order; delayed: a label used only a horizon after its interval was issued. First 50 / 100: coverage of delayed ACI over the first events of the stream at 10 s; delay: mean events issued before a label arrives at 10 s.

| model | static | static prequential | ACI frozen | ACI online | ACI chronological | ACI chronological, delayed | static prequential, delayed | first 50 / 100, 10 s | delay, events |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_cf_hist | 0.80 0.76 0.81 0.79 | 0.83 0.82 0.84 0.82 | 0.81 0.80 0.84 0.79 | 0.89 0.90 0.90 0.89 | 0.90 0.90 0.91 0.91 | 0.90 0.90 0.91 0.91 | 0.83 0.82 0.87 0.85 | 0.76 / 0.84 | 0.2 |
| lstm_hetero_frozen | 0.71 0.70 0.79 0.83 | 0.78 0.79 0.84 0.85 | 0.73 0.77 0.83 0.82 | 0.89 0.90 0.90 0.89 | 0.89 0.89 0.90 0.89 | 0.89 0.89 0.90 0.89 | 0.77 0.79 0.84 0.85 | 0.81 / 0.84 | 0.2 |

### Clocks of the evaluated streams (R8)

| target | data set | clock | events | followers | recorded span, s | events per s |
|---|---|---|---|---|---|---|
| a | highd | untimed | 500 | 500 | - | - |
| a | ngsim | absolute | 500 | 495 | 2787 | 0.179 |
| a | openacc | per_run | 500 | 45 | 33908 | 0.015 |

### Follower-cluster bootstrap of the coverage intervals (R4)

Width of the 95 % interval when whole followers are resampled, relative to the event bootstrap (median and largest over models and horizons).

| target | data set | variant | followers | events | width ratio, median | width ratio, max |
|---|---|---|---|---|---|---|
| a | highd | raw | 500 | 500 | 1.02 | 1.07 |
| a | highd | aci_online | 500 | 500 | 1.00 | 1.14 |
| a | ngsim | raw | 495 | 500 | 1.00 | 1.05 |
| a | ngsim | aci_online | 495 | 500 | 1.00 | 1.10 |
| a | openacc | raw | 45 | 500 | 1.90 | 2.97 |
| a | openacc | aci_online | 45 | 500 | 1.67 | 2.46 |

