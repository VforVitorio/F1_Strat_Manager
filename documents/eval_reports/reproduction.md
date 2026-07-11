# reproduction

- harness `e1ee603` · schema v1 · generated 2026-07-11T17:31:11+00:00
- era 2022-2025 · dataset 2025 holdout vs published model_configs · seed deterministic · llm none
- artifacts: overtake_model=`cbb9d0eb0beb`

| model | metric | published | reproduced | status | detail |
|---|---|---|---|---|---|
| overtake | auc_pr_test | 0.5491 | 0.5491 | reproduced | |delta| 0.0000 vs tol 0.01 |
| safety_car | auc_pr_test | 0.0723 | 0.0723 | reproduced | |delta| 0.0000 vs tol 0.01 |
| undercut | auc_pr_test | 0.6739 | 0.6739 | reproduced | |delta| 0.0000 vs tol 0.01 |
| pit_duration | p50_mae_test_s | 0.487 | 0.4893 | reproduced | n=252; |delta| 0.0023 vs tol 0.01 |
| pace | mae_test_s | 0.4104 | - | pending | laptime holdout feature build not wired this phase |
| tire_degradation | mae_test_s | 0.7078 | - | pending | TCN MC forward pass not run this phase (see calibration MC-sigma) |
