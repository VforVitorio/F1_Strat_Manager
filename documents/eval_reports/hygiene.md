# hygiene

- harness `33ff301` · schema v1 · generated 2026-07-11T19:38:09+00:00
- era 2022-2025 · dataset notebooks/strategy audit + 2024/2025 overtake & SC holdouts · seed deterministic · llm none
- artifacts: none

| item | kind | model | verdict | selection | evidence |
|---|---|---|---|---|---|
| optimal_threshold=0.7976 | threshold | overtake | **contaminated** | argmax-F1 on the 2025 test set | N12_overtake_model.ipynb (threshold-analysis / step-5 cell) |
| best_threshold=0.2335 + 3/5/7-lap window | threshold | safety_car | **contaminated** | argmax-F2 threshold AND the target-window both selected on the 2025 test set | N14_sc_model.ipynb (PR-curve + window-comparison cells) |
| sc Platt calibrator | calibrator | safety_car | **underdocumented** | fit on 2024 probabilities, but 2024 is IN the train set (config fitted_on='val_2024' is misleading) | N14_sc_model.ipynb (calibration cell) |
| circuit_cluster (k-means) | aggregate_feature | overtake/safety_car/laptime/tire | **underdocumented** | load_all_races scans every year dir and the k-means was fit over the pooled 2023-2025 set (N03 cell 6 output 'Successfully loaded 71 GPs'; the deployed circuit_clusters_k4.parquet contains the 2025-only alias 'Miami Gardens'). Its inputs are per-circuit OUTCOME aggregates (mean_laptime, degradation_rate, mean_sector_speed), not pre-race geometry | N03_circuit_clustering.ipynb (load_all_races / drop_redundant_features / fit_kmeans_final) |
| best_threshold=0.522 | threshold | undercut | **clean** | argmax-F1 on 2024 in-train (2024 is part of N16's 2023+2024 train set, same structure as overtake; mild, 143 positives / base 0.413) - but NEVER selected on test-2025 | N16_undercut.ipynb ("Threshold on calibrated val 2024") |
| circuit_sc_rate | aggregate_feature | safety_car | **clean** | past-season only (year < yr); 2023 rows get a fixed SC_PRIOR=0.15 | N13_sc_eda.ipynb (compute_circuit_sc_rate) |
| team_year_median | aggregate_feature | pit_duration | **clean** | lookup fit on train only, applied to test with recent-year fallback | N15_pit_duration.ipynb (add_team_year_median) |
| circuit_undercut_rate + team_x_undercut_rate | aggregate_feature | undercut | **clean** | target encoding fit on train only, train-mean fallback on test | N16_undercut.ipynb (compute_target_encoding) |
| year_circuit_median / team_pace_rank | aggregate_feature | laptime | **clean** | within-session per-year aggregates; flagged LEAKY and removed from the reported model | N06_laptime_model.ipynb (add_context_features / FEATURES_PROD) |

## Correction (overtake threshold)

- leaked threshold 0.7976 (selected on test): P 0.5018 / R 0.5556 / F1 0.5273 on 2025 test
- corrected threshold 0.7626 (selected on 2024, in-train): P 0.4583 / R 0.5827 / F1 0.5131 on 2025 test
- threshold selected on 2024 (in-train; 2024 is part of N12's 2023+2024 train set, mild memorization at ~28k pairs); both operating points evaluated on the 2025 test set

## Correction (safety-car threshold)

- leaked threshold 0.2335 (F2 on test): P 0.0797 / R 0.5581 / F2 0.2537 on 2025 test
- 'corrected' threshold 0.6358 (F2 on 2024, IN-TRAIN, 19/1042 positive): P 0.0 / R 0.0 / F2 0.0 on 2025 test
- the F2 threshold 'selected on 2024' is IN-TRAIN (2024 is a subset of the 2023+2024 train set), NOT a held-out split; it lands on the train-memorization boundary and collapses on test. This shows N14 has no honest validation split for an operating threshold - the paper should report SC threshold-free, not re-select an operating point

## Correction (safety-car target window)

| window | 2024 AUC-PR (in-train) | test-2025 AUC-PR |
|---|---|---|
| sc_within_3_laps | 0.8817 | 0.0723 |
| sc_within_5_laps | 0.6912 | 0.1054 |
| sc_within_7_laps | 0.6257 | 0.1323 |

- only the 3-lap model is persisted; the 5/7-lap models needed to re-select the window are not on disk (retraining out of scope). The 2024 column is IN-TRAIN (2024 subset of train), so its high AUC-PR (0.88) is resubstitution, NOT validation - it only shows the metric is window-unstable. The reported SC AUC-PR 0.0723 keeps its test-window-selected caveat; this is single-model sensitivity, not the original 3-model selection

## Conclusion for the paper freeze

- 2 contaminated items (overtake threshold; safety-car threshold + window). All other thresholds and aggregate features are clean or non-target.
- **Overtake headline clears**: AUC-PR 0.5491 / AUC-ROC 0.8758 are threshold-free and involve no window selection; the threshold leakage touches only their operating point. NOTE the overtake 'correction' below is also in-train (N12 trains final on 2023+2024, 2024 was only Optuna inner-val), but with 28k pairs the memorization is mild, so its re-selected threshold is a reasonable operating point rather than a collapse.
- **Safety-car has NO honest validation split**: N14 trains on 2023+2024 and tests on 2025, so there is no held-out split to re-select an operating threshold on. Re-selecting on 2024 is in-train (resubstitution) and collapses on test - evidence that an honest SC operating threshold does not exist without a fresh val split or retraining, NOT that 0.2335 was specifically test-overfit. The paper should report SC threshold-free.
- **Safety-car window cannot be retro-selected**: only the 3-lap model is persisted, and the window was originally chosen by max-lift on test-2025, so it cannot be honestly re-chosen without retraining the 5/7-lap models. The reported SC AUC-PR 0.0723 (the lowest of the three windows) keeps an explicit test-window-selected caveat.
- **circuit_cluster - REAL but coarse test-season leak**: the k-means fit pooled 2023-2025 (N03 'Successfully loaded 71 GPs'; the deployed cluster table contains the 2025-only 'Miami Gardens'), and its inputs include `mean_laptime` - an aggregate of the pace target itself - so `Cluster` is not target-free. It is a 4-way (2-bit) bucket over ~25 circuits, deployed via N04's static fit-time lookup, and it feeds overtake, SC, laptime AND tire (Model A). Materiality is bounded (2-bit quantization over stable circuit character; the delta pace target absorbs per-circuit constants) but NOT yet measured; the demonstration (refit k-means 2023-24-only, count 2025 label flips) is deferred to #376. N03 is untouchable, so no code fix.
- **Recommendation before freeze (not executed - no retrain)**: give SC/overtake a real held-out validation split (or nested CV) if a defensible operating threshold is ever needed. The paper reports both threshold-free (their headline AUC-PR/AUC-ROC are unaffected).
- Every other headline (undercut 0.6739, pit 0.487, pace 0.4104, tire 0.7078, sentiment 0.84) is unaffected by these findings.
