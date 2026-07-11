# metrics_registry

- harness `91b98d3` · schema v1 · generated 2026-07-11T15:07:44+00:00
- era 2022-2025 · dataset model_configs + thesis Tabla 6.1 · seed deterministic · llm none
- artifacts: overtake_cfg=`f72ef54ebbc8`, undercut_cfg=`b0250c8522a2`, pit_cfg=`41dd673a93fb`, sc_cfg=`4691fb9c30c1`

| model | metric | value | thr | split | canonical | source |
|---|---|---|---|---|---|---|
| overtake | auc_pr_test | 0.5491 | 0.7976 | train [2023, 2024] / test 2025 | yes | config: overtake_probability/model_config.json |
| undercut | auc_pr_test | 0.6739 | 0.522 | train [2023, 2024] / test 2025 | yes | config: pit_prediction/model_config_undercut_v1.json |
| pit_duration | p50_mae_test_s | 0.487 | - | train [2023, 2024] / test 2025 | yes | config: pit_prediction/model_config.json |
| safety_car | auc_pr_test | 0.0723 | 0.2335 | train [2023, 2024] / test 2025 | yes | config: safety_car_probability/feature_list_v1.json |
| pace | mae_test_s | 0.4104 | - | test 2025 | yes | thesis Tabla 6.1 (final) |
| pace | mae_test_s | 0.392 | - | test 2025 | no | notebook N06 (superseded) |
| tire_degradation | mae_test_s | 0.7078 | - | test 2025 | yes | thesis Tabla 6.1 (global) |
| sentiment | accuracy | 0.84 | - | held-out radio set | yes | thesis Tabla 6.1 (final) |
| sentiment | accuracy | 0.875 | - | held-out radio set | no | published-era (superseded) |

## Divergences reconciled

- **pace mae_test_s**: 0.392 (notebook N06 (superseded)) -> **0.4104** (thesis Tabla 6.1 (final))
- **sentiment accuracy**: 0.875 (published-era (superseded)) -> **0.84** (thesis Tabla 6.1 (final))
