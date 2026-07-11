# hygiene

- harness `5c97461` · schema v1 · generated 2026-07-11T17:00:40+00:00
- era 2022-2025 · dataset notebooks/strategy audit + 2024/2025 overtake & SC holdouts · seed deterministic · llm none
- artifacts: —

| item | kind | model | verdict | selection | evidence |
|---|---|---|---|---|---|
| optimal_threshold=0.7976 | threshold | overtake | **contaminated** | argmax-F1 on the 2025 test set | N12_overtake_model.ipynb (threshold-analysis / step-5 cell) |
| best_threshold=0.2335 + 3/5/7-lap window | threshold | safety_car | **contaminated** | argmax-F2 threshold AND the best target-window both on the 2025 test set | N14_sc_model.ipynb (PR-curve + window-comparison cells) |
| circuit_cluster (k-means) | aggregate_feature | overtake/safety_car/laptime | **underdocumented** | k-means fit window not year-restricted in code; 2025 holdout is intent-only | N03_circuit_clustering.ipynb (load_all_races / fit_kmeans_final) |
| best_threshold=0.522 | threshold | undercut | **clean** | argmax-F1 on the calibrated val-2024 split | N16_undercut.ipynb ("Threshold on calibrated val 2024") |
| circuit_sc_rate | aggregate_feature | safety_car | **clean** | past-season only (year < yr); 2023 rows get a fixed SC_PRIOR=0.15 | N13_sc_eda.ipynb (compute_circuit_sc_rate) |
| team_year_median | aggregate_feature | pit_duration | **clean** | lookup fit on train only, applied to test with recent-year fallback | N15_pit_duration.ipynb (add_team_year_median) |
| circuit_undercut_rate + team_x_undercut_rate | aggregate_feature | undercut | **clean** | target encoding fit on train only, train-mean fallback on test | N16_undercut.ipynb (compute_target_encoding) |
| year_circuit_median / team_pace_rank | aggregate_feature | laptime | **clean** | within-session per-year aggregates; flagged LEAKY and removed from the reported model | N06_laptime_model.ipynb (add_context_features / FEATURES_PROD) |

## Correction (overtake threshold)

- leaked threshold 0.7976 (selected on test): P 0.5018 / R 0.5556 / F1 0.5273 on 2025 test
- corrected threshold 0.7626 (selected on val-2024): P 0.4583 / R 0.5827 / F1 0.5131 on 2025 test
- threshold selected on val-2024; both operating points evaluated on the 2025 test set

## Correction (safety-car threshold)

- leaked threshold 0.2335 (F2 on test): P 0.0797 / R 0.5581 / F2 0.2537 on 2025 test
- corrected threshold 0.6358 (F2 on val-2024, 19/1042 positive): P 0.0 / R 0.0 / F2 0.0 on 2025 test
- F2 threshold selected on val-2024 raw scores; both operating points on 2025 test. val-2024 has few SC positives so the corrected point is high-variance - the collapse is the evidence that the leaked operating point was test-overfit

## Correction (safety-car target window)

| window | val-2024 AUC-PR | test-2025 AUC-PR |
|---|---|---|
| sc_within_3_laps | 0.8817 | 0.0723 |
| sc_within_5_laps | 0.6912 | 0.1054 |
| sc_within_7_laps | 0.6257 | 0.1323 |

- only the 3-lap model is persisted; the 5/7-lap models needed to re-select the window on val are not on disk (retraining out of scope). The reported SC AUC-PR 0.0723 keeps the caveat that its window was test-selected; the table is single-model sensitivity, not the original 3-model selection

## Conclusion for the paper freeze

- 2 contaminated items (overtake threshold; safety-car threshold + window). All other thresholds and aggregate features are clean or non-target.
- **Overtake headline clears**: AUC-PR 0.5491 / AUC-ROC 0.8758 are threshold-free and involve no window selection; the threshold leakage touches only their operating point, corrected above.
- **Safety-car operating threshold is NOT robustly recoverable**: re-selecting on val-2024 collapses the operating point (val-2024 has too few SC positives), which is itself the evidence that the leaked 0.2335 was test-overfit. The paper should report SC threshold-free and not claim a fixed operating threshold.
- **Safety-car window cannot be retro-selected**: only the 3-lap model is persisted, so the {3,5,7}-lap window selected on test-2025 cannot be honestly re-chosen without retraining the 5/7-lap models. The reported SC AUC-PR 0.0723 therefore keeps an explicit test-window-selected caveat.
- **Remaining action before freeze**: pin a year filter in N03 `load_all_races` to close the circuit_cluster underdocumentation.
- Every other headline (undercut 0.6739, pit 0.487, pace 0.4104, tire 0.7078, sentiment 0.84) is unaffected by these findings.
