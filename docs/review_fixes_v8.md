# Fixes and controls of the second review (manuscript v8)

Companion of `docs/review_fixes_v3.md` and `docs/review_fixes_v4.md`. The second review (2026-10-10, items
V7-R01 to V7-R14) asked for restricted causal statements, two further control cells, a version-matched supplement
and derived-data release, bibliographic corrections and a design description of the controls. This note records
what was computed, where it lives and what the text now states; the contract of the two new controls is
`docs/design.md`, sections 18.8 and 18.9.

## 1. New control runs

### 1.1 V7-R01: history-only Diffusion trained on the pooled domains

Configs `configs/experiment/e6_pool_ctrl.yaml` and `e6_lodo_openacc_ctrl.yaml` (inherit the E6 roots: matched
signals, caps, 5 seeds, target a, frozen e1m IDM prior, evaluated sets); platoon units in the E3 variants `pool`
and `lodo_openacc` through the per-variant checkpoint override of `diffusion_cf_hist` in `configs/experiment/e3.yaml`.

```
bash scripts/run_e1_retry.sh artifacts/e6_pool_ctrl/logs/run.out 8 --config configs/experiment/e6_pool_ctrl.yaml
bash scripts/run_e1_retry.sh artifacts/e6_lodo_openacc_ctrl/logs/run.out 8 --config configs/experiment/e6_lodo_openacc_ctrl.yaml
bash scripts/run_e3_retry.sh artifacts/e3/logs/run_hist_pooled.out 8 --variants pool lodo_openacc --models diffusion_cf_hist --quiet
python scripts/analyze_e1.py --root artifacts/e6_pool_ctrl --config configs/experiment/e6_pool_ctrl.yaml --in-domain pool_test
python scripts/analyze_e1.py --root artifacts/e6_lodo_openacc_ctrl --config configs/experiment/e6_lodo_openacc_ctrl.yaml --in-domain pool_test
python scripts/analyze_e3.py
python scripts/make_report.py
```

Result: history-only diffusion на ACC-колоннах опровергнута в обеих pooled-ячейках: −0.45 (профили [−0.99, −0.08], файлы [-1.73, 0.05]) без OpenACC и −0.32 ([−0.78, 0.00], файлы [-1.35, 0.11]) с OpenACC против +0.47 / +0.56 у модели с будущим лидера; CRPS амплитуд 5.1 / 4.6 м/с против 1.9 / 1.5 (эталон 3.5); первый ведомый сталкивается в 17 % / 13 % против 2 % / 1.6 %; бесстолкновительных колонн 0 %; 91–93 % первых столкновений в burn-in. Pooled-домен улучшает history-only модель (13.2 → 4.6–5.1 м/с), но оставляет её хуже эталона. Вывод «преимущество — это информация о будущем лидера, не архитектура и не домен сам по себе» теперь измерен для обоих режимов обучения

### 1.2 V7-R02: the cell that separates the filter from the target

Configs `configs/experiment/e1m_rt.yaml` (centred matched inputs, target mean_1s from the recorded speed, the five
models of the causal control retrained on highD, evaluated on the events of `e1m_causal`: shifted sets without
their validation fold) and `e1m_mean1s_abv.yaml` (the e1h_mean1s checkpoints, centred inputs with the processed
target, on the same events, nothing trained).

```
bash scripts/run_e1_retry.sh artifacts/e1m_rt/logs/run.out 8 --config configs/experiment/e1m_rt.yaml
bash scripts/run_e1_retry.sh artifacts/e1m_mean1s_abv/logs/run.out 8 --config configs/experiment/e1m_mean1s_abv.yaml
python scripts/analyze_e1.py --root artifacts/e1m_rt --config configs/experiment/e1m_rt.yaml
python scripts/analyze_e1.py --root artifacts/e1m_mean1s_abv --config configs/experiment/e1m_mean1s_abv.yaml
```

Three cells with the same fold-exclusion rule (the centred pair shares every window; the causal cell keeps 1-7 % more events after its processing, see section 1.3): (centred, processed target) = e1m_mean1s_abv, (centred, recorded target) = e1m_rt,
(causal, recorded target) = e1m_causal. Result: цель, а не фильтр, несёт почти всю чувствительность: на NGSIM I-80 при одних центрированных входах и событиях NLL с обработанной целью −0.71…0.95 нат, с записанной целью 3.8–140 (LSTM 1.85 → 140, HeteroMDN 0.14 → 71, ансамбль −0.45 → 49, MC dropout −0.71 → 23, GP 0.95 → 3.8; PICP 0.57–0.88 → 0.12–0.59); замена центрированного фильтра каузальным при фиксированной записанной цели меняет NLL в 0.7–1.9 раза (140 → 259, 71 → 70, 49 → 40, 23 → 16, 3.8 → 5.9). US-101 и Waymo — так же; на OpenACC (записанная 1-с цель почти так же предсказуема, как на highD: persistence RMSE 0.24 против 0.16) записанная цель стоит 0.5–1.3 нат (GP 0.03), а 50–76 нат LSTM присутствуют во всех ячейках (сдвиг поведения/состояния). Текст E5, абстракт и заключение переформулированы: падение переуверенности при выравнивании — для четырёх из пяти моделей почти целиком сглаженная цель (входы при записанной цели меняют NLL на −9…+16 нат), для LSTM — входы (244 из 383 нат) и цель (138) вместе

## 2. Recomputations without training

* **V7-R08, follower-cluster intervals of the frozen-mean control.**
  `python scripts/analyze_e1.py --root artifacts/e1_ctrl --config configs/experiment/e1_ctrl.yaml --h33-pair lstm_hetero_frozen lstm_const_sigma --h33-pair hetero_mdn lstm_const_sigma --unit driver --tables-dir tables_driver`
  (`artifacts/e1_ctrl/tables_driver/h33_nll.csv`). On highD test the follower-unit intervals coincide with the
  event-unit intervals within 0.001 nats for both pairs and both overall gains (the braking regime differs by 0.003; frozen mean, all windows: event
  [-0.0789, -0.0415], follower [-0.0787, -0.0416]; HeteroMDN: [-0.1628, -0.1301] vs [-0.1626, -0.1302]); the
  Holm-adjusted verdicts per regime are unchanged (approaching and steady following not rejected, p 0.32 vs 0.41).
  Both units are in the report (`t3_h33_nll__ctrl__a__event`, `__follower`) and in S1.
* **V7-R06, joint calibration table.** `paper/v8/tools/make_c19.py` → `paper/v8/tables/c19_aci_joint.tex` from
  `artifacts/e2/tables/coverage_by_horizon.csv` (NGSIM I-80, target a, 10 s): coverage, median finite width,
  share of infinite intervals, coverage over the first 50 events, for static, prequential static and ACI online
  (random, chronological, delayed). ACI online keeps 0.88-0.91 with finite widths 17 m (IDM) to 279-316 m (LSTM)
  and 0.3-3.7 % infinite intervals for the LSTM-based models; the prequential static correction 0.65-0.90 with
  widths 16-77 m and cold-start coverage 0.32-0.93. No interval score: infinite for the events with an infinite
  interval.
* **V7-R13, absolute CRPS of the matrix.** From `artifacts/e3/tables/h32.csv` (Diffusion, ACC, amplitude,
  response window): the reference of each cell is at 3.45-3.55 m/s (nearly common denominator); Diffusion 2.30 / 2.00
  (highD, leader window 11 / 61), 1.54 / 1.54 (pooled), 1.87 / 1.88 m/s (OpenACC held out).
* **V7-R04, sources of the control tables in the report.** `scripts/make_report.py` now writes the one-step tables
  of the control roots (`t1_one_step__ctrl__a`, `__matched_causal__mean_1s`, `__matched_rt__mean_1s`,
  `__matched_abv__mean_1s`, `__pool_ctrl__a`, `__lodo_openacc_ctrl__a`), the H3.3 pairs of the frozen-mean control
  in both units and the density sensitivity (`t1_density_sensitivity__raw__a`); the public export adds
  `results/controls/` (the analysis CSV / JSON of the control roots as they are) and `results/TABLE_SOURCES.md`
  (`paper/v8/tools/make_table_sources.py`: every article table → source files with SHA-256 → script → command).

## 3. Text changes of v8 (where each item of the review is answered)

| Item | Where in v8 | Statement now |
|---|---|---|
| R01 | Abstract; Introduction (contribution and the last paragraph); Results 3.10 "Equal information", "Equal information on the pooled domains"; Conclusions | the highD ablation establishes the effect of the conditioning for the highD-trained implementation; the pooled cells are reported with their own verdicts |
| R02 | Results 3.10 "Past-only processing", "Centred inputs with the recorded target"; Discussion | the causal control compares pipelines; no lag mechanism; the recorded-target cell separates filter and target on identical events |
| R03 | Results 3.10 "Collided rollouts" | survivor sensitivity for the first four followers, not a re-test of H3.2; collision probability as its own outcome |
| R05 | Results 3.3, 3.10 "Variance head at a fixed mean"; Abstract; Conclusions | two measured gains side by side; the ratio is descriptive; no attribution of the remainder |
| R06 | Results 3.4 and 3.10 "Calibration recursion and feedback budget" (Table c19); Discussion | width as a diagnostic of the whole procedure under the feedback protocol; "fails as it must" removed; coverage is marginal and sequential |
| R07 | references.bib; Related Work (rows of Liu, FollowGen, Kashifi 2026, Punzo 2026) | records verified against Crossref / arXiv / PMLR; two 2026 works compared |
| R08 | Methods 2.3 (estimand of the platoon law: mixture of the five seeds); Results 3.10 (follower-unit intervals); Discussion (three human recordings) | |
| R09 | Abstract; Results 3.5; Conclusions | "none of the tested configurations met the prespecified gain"; inconclusive is not no gain; collision rates named as collisions |
| R10 | Discussion (platoon paragraph) | burn-in concentration of the first collisions; competing hypotheses, not a mechanism |
| R11 | Methods 2.9 "Control Experiments" (Table c0_controls), protocol table row; Results 3.10 renamed; captions c13-c18; cross-references in 3.3, 3.4, 3.5, 3.9 | |
| R12 | Results 3.5 (common-support exponent primary; the 1-20 fit as extrapolation) | agreement of sign and magnitude, not equivalence |
| R13 | Results 3.10 "Training domain and leader smoothing separated", "Correlated acceleration noise"; Discussion | training regime, not domain diversity; absolute CRPS; AR(1) scope |
| R14 | Methods (H3.4 part (i); 0.5 s); Results (under-coverage condition); Discussion (duplicate removed); main.tex (abstract and keywords pasted) | |

### 1.3 Third pre-submission review: the three cells on their common windows

The causal cell (`e1m_causal`) applies the same fold-exclusion rule as the two centred cells but keeps 1-7 % more
events after its processing (no gap-reconstruction constraint in the causal variant: NGSIM I-80 6085 against 5941
events, OpenACC 3668 against 3426); the two centred cells share every window. `scripts/common_support_cells.py`
aligns the three roots by (event_id, t_index), restricts every cell to the common windows and recomputes NLL, CRPS,
coverage and width per model (mean over the windows of each seed, then over the seeds), with the paired NLL
difference causal - centred (recorded target) and 95 % bootstrap intervals over events and followers
(`artifacts/e1m_causal/tables/common_support.csv`, report table `t1_common_support__cells__mean_1s`, rows
"common windows" of the article table c17). On NGSIM I-80 the causal LSTM is 257.7 nats on the common windows
against 258.5 on its own sample; on OpenACC 41.3 against 50.5, so the restriction alone moves it by 9 nats there.
Paired differences causal - centred(recorded target) on the common windows: I-80 LSTM +118 [115, 121], HeteroMDN
-0.5 [-1.5, 0.5], deep ensemble -8.2, MC dropout -6.1, GP residual +1.8; US-101 +87, +14, +4.1, +0.2 [-0.04, 0.45],
+0.9; Waymo +12.3, +2.1, +1.2, +0.8, +0.9; OpenACC LSTM -30 [-39, -22] (followers [-87, 5]), HeteroMDN +5.1,
deep ensemble +3.8, MC dropout +1.9, GP residual +0.09. The text of v11 states these as differences between
pipelines (filter, window selection, trained checkpoints) and drops the causal attribution of the matched-signal
gain to inputs or target (the 244 / 138 nats and the 88 % / 61 % of v9-v10): only the target is isolated, by the
centred pair; no cell changes the inputs alone.
