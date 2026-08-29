# calibration

- harness `4724f3ed` · schema v1 · generated 2026-08-29T18:38:10+00:00
- era 2022-2025 · dataset 2025 holdout + frozen calibration artifacts · seed deterministic · llm none
- artifacts: overtake_model=`cbb9d0eb0beb`, pit_cfg=`41dd673a93fb`, tcn_mc_calib=`e03a170f6de4`

| model | metric | value | nominal | status | detail |
|---|---|---|---|---|---|
| undercut | ece_calibrated | 0.1303 | 0.05 | drift | n=252; 10-bin equal-width |
| pit_duration | p05_p95_coverage | 0.6984 | 0.9 | drift | n=252; empirical P05-P95 coverage recomputed on the regenerated N15 holdout |
| overtake | brier_calibrated | 0.0520 | - | ok | n=10217; brier raw 0.1304 -> cal 0.0520 (Platt val-2024) |
| overtake | ece_calibrated | 0.0319 | 0.05 | ok | n=10217; 10-bin equal-width |
| safety_car | brier_calibrated | 0.0426 | - | ok | n=995; recomputed on 2025 holdout |
| safety_car | ece_calibrated | 0.0347 | 0.05 | ok | n=995; 10-bin equal-width |
| undercut | brier_calibrated | 0.1930 | - | ok | n=252; recomputed on 2025 holdout |
| tire_degradation[C2] | mc_mean_sigma_s | 0.1244 | - | ok | stored epistemic sigma (single 2025 stint, unseeded MC; N10 Step 16) |
| tire_degradation[C4] | mc_mean_sigma_s | 0.1504 | - | ok | stored epistemic sigma (single 2025 stint, unseeded MC; N10 Step 16) |
| tire_degradation[C5] | mc_mean_sigma_s | 0.1549 | - | ok | stored epistemic sigma (single 2025 stint, unseeded MC; N10 Step 16) |
| tire_degradation[C6] | mc_mean_sigma_s | 0.2621 | - | ok | stored epistemic sigma (single 2025 stint, unseeded MC; N10 Step 16) |
| tire_degradation | mc_mean_sigma_s_global | 0.1524 | - | ok | seeded MC (N=50) over the full 2025 test set; stored per-compound mean 0.1729 s (same order of magnitude) |
