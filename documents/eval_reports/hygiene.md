# hygiene

- harness `50b7ecf` · schema v1 · generated 2026-07-11T15:29:09+00:00
- era 2022-2025 · dataset notebooks/strategy audit + 2024/2025 overtake holdout · seed deterministic · llm none
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

## Conclusion for the paper freeze

- 2 contaminated items (overtake threshold; safety-car threshold + window). All other thresholds and aggregate features are clean or non-target.
- **Overtake headline clears**: AUC-PR 0.5491 / AUC-ROC 0.8758 are threshold-free and involve no window selection; the threshold leakage touches only their operating point.
- **Safety-car headline is optimistic, NOT clean**: AUC-PR 0.0723 is the max-lift window among {3,5,7} laps selected on test-2025, so the reported number is itself selection-biased (a max over 3 candidates on test), on top of the operating-point threshold leakage.
- **Action before freeze**: (1) re-select the overtake + SC operating thresholds on val-2024 (the undercut N16 pattern) - the overtake correction above shows the honest operating point; (2) re-select the SC target window on the CV/val split (not test) and re-report SC AUC-PR, or caveat it as test-selected; (3) pin a year filter in N03 `load_all_races` to close the circuit_cluster underdocumentation.
- Every other headline (undercut 0.6739, pit 0.487, pace 0.4104, tire 0.7078, sentiment 0.84) is unaffected by these findings.
