# reproduction

- harness `91b98d3` · schema v1 · generated 2026-07-11T15:07:22+00:00
- era 2022-2025 · dataset 2025 holdout vs published model_configs · seed deterministic · llm none
- artifacts: overtake_model=`cbb9d0eb0beb`

| model | metric | published | reproduced | status | detail |
|---|---|---|---|---|---|
| overtake | auc_pr_test | 0.5491 | 0.5491 | reproduced | |delta| 0.0000 vs tol 0.01 |
| undercut | auc_pr_test | 0.6739 | - | pending | historical aggregates circuit_undercut_rate/team_x_undercut_rate absent from holdout (#207) |
| pit_duration | p50_mae_test_s | 0.487 | - | pending | pit_labeled holdout empty on disk (#207) |
| safety_car | auc_pr_test | 0.0723 | - | pending | engineered features (lap_time_*_z, anomaly_and_yellow, lap1_chaos) absent from holdout (#207) |
| pace | mae_test_s | 0.4104 | - | pending | laptime holdout feature build not wired this phase |
| tire_degradation | mae_test_s | 0.7078 | - | pending | TCN MC forward pass not run this phase (see calibration MC-sigma) |
