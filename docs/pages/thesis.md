# Thesis results

**This page reports the headline benchmark results for F1 StratLab, the accuracy and latency of its seven ML models and six sub-agents**, as referenced in chapter 5 of the TFG thesis. Every figure is regenerated automatically from the notebooks under `notebooks/agents/`, so it always tracks the latest model artefacts.

## Threshold sweeps

Each classifier sub-agent exposes a precision–recall trade-off that the strategist picks deliberately. The sweeps below scan the full threshold space and mark the production operating point.

### Overtake (N12)

Production threshold **0.7976** was tuned in N12 step 5 on the raw LightGBM scores. The sweep shows the trade-off is robust around that point: F1 stays within a few hundredths across the neighbouring grid.

### Safety Car (N14)

The Safety Car model is a soft contextual prior, not an exact predictor. AUC-PR is **0.0723** versus a **0.0432** baseline. The **0.234** production threshold is F2-optimal (recall-weighted) because false alarms cost little and missing an imminent SC is expensive.

### Undercut (N16)

The undercut classifier sees the highest positive prevalence of the three (>30 % on the holdout) because the labelling step kept only pairs with a true undercut opportunity. The **0.522** threshold falls in the flat F1 region.

## MC Dropout coverage

The TCN tire-degradation model (N09 global + N10 per-compound fine-tunes) uses 50-pass MC Dropout to produce P10 / P50 / P90 percentile bands. Both the raw [P10, P90] coverage (epistemic only) and the calibrated coverage that adds the empirical residual sigma (aleatoric included) are reported.

Raw coverage stays around **0.20** across all compounds, active dropout only captures the model-weight uncertainty, not the lap-to-lap aleatoric noise. The calibrated coverage matches the **0.80** nominal target by construction.

## Monte Carlo projection accuracy

The strategy layer scores its four candidates in **projected track position**, not in seconds. That claim is directly checkable, because every real pit stop in the dataset is already a labelled example of it: project the stop from the lap before, then compare against where the car actually came out on the lap after. No hand-labelling is involved, which is what makes this a measured result rather than a plausibility argument.

Over **1810 green-flag stops across 71 races** (2023 to 2025), the projection lands **within one position 86.5 %** of the time and is **exactly right 59.1 %** of the time, with a mean signed error of **+0.57 positions**. The bias is positive, so the projection is mildly pessimistic about the rejoin, which is the safer direction for a strategy call.

Neutralised stops are excluded, and not to flatter the number. Under a Safety Car every lap is slow, so the "two normal laps" baseline used to reconstruct the realised pit loss is wrong there: that corrupts the measurement's **input** rather than the projection itself. Measured separately, those stops show a mean error of +1.54 positions against +0.57 under green, which is the signature of exactly that problem.

### The tables the scorer reads

Seven tables are counted off the same 71 races of raw laps rather than assumed, and the scorer reads them at runtime from `data/mc_measured_v1.json`.

| Table | What it answers |
|---|---|
| `clean_air` | seconds a lap a follower gains once the car directly ahead pits, per circuit |
| `gap_density` | seconds between consecutive cars, so a projected gap maps to a place |
| `neutralisation_rate` | chance a Safety Car arrives while a stop is being deferred |
| `sc_window` | green laps left inside the 5-lap decision window once neutralised |
| `status_mix` | share of laps that are actually racing, the denominator for the rest |
| `stop_hazard` | chance a rival stops in the window, by tyre life |
| `undercut_band` | undercut success against the gap to the target |

The source is the **raw** parquet and never the featured one, which drops the neutralised and pit laps these tables are precisely about.

## How to regenerate

```bash
# Monte Carlo projection accuracy + the measured-table inventory (~1 min)
uv run f1-eval projection

# The measured tables themselves, from 71 races of raw laps (~10 min)
uv run python scripts/measure_mc_tables.py

# Threshold sweeps + MC Dropout figures (one notebook, ~5 min on GPU)
uv run jupyter nbconvert --execute --inplace notebooks/agents/N33_thresholds_and_calibration.ipynb

# Quantitative RAG benchmark (10-15 min, builds 2 additional Qdrant collections)
uv run jupyter nbconvert --execute --inplace notebooks/agents/N30B_rag_benchmark.ipynb
```

Both notebooks emit CSV and Markdown tables alongside their PNGs:

- Sweeps: `data/eval/threshold_sweep_{overtake,sc,undercut}.{csv,md}`
- MC Dropout: `data/eval/mc_dropout_coverage.{csv,md}`
- RAG benchmark: `data/rag_eval/results_v1.md`
- Projection accuracy: `documents/eval_reports/projection.{md,json}`
- Measured MC tables: `data/mc_measured_v1.json`, with thesis-facing extracts at `data/eval/mc_{clean_air,gap_density,sc_window,undercut_band}.{csv,md}`

## Numeric headline metrics

| Component | Metric | Value | Source |
|---|---|---|---|
| Pace model (N06 XGBoost) | MAE on 2025 holdout | **0.410 s** | `data/eval/pace_baselines.{csv,md}` |
| Whisper turbo (CUDA) | mean per-clip latency | **233.9 ms** (P95 325.8 ms) | `data/eval/whisper_results.{csv,md}` |
| NLP pipeline (GPU) | mean `run_pipeline` | **42.1 ms** | `data/eval/nlp_pipeline_cpu.{csv,md}` |
| Sub-agent latency | min / max mean | **487 ms** (pace) / **4.4 s** (rag w/ LLM) | `data/eval/subagent_latency.{csv,md}` |
| RAG agent | Content P@5 | **0.80** | `data/rag_eval/results_v1.md` |
| MC Dropout (C2) | calibrated 80 % coverage | **0.840** | `data/eval/mc_dropout_coverage.{csv,md}` |
| Position projection | within one place, 1810 real stops | **86.5 %** | `documents/eval_reports/projection.{md,json}` |

All numbers reproducible with the commands above.

**Note on the NLP pipeline figure.** The thesis and IEEE report publish **47.8 ms mean / 59.4 ms P95** for `run_pipeline` (the original N24 benchmark run). The **42.1 ms** figure in the table above comes from re-running the same benchmark on the current hardware and environment, as this page's figures are regenerated automatically rather than pinned to the publication. Both numbers are valid measurements of the same pipeline; the difference is measurement lineage (published vs. regenerated), not a correction.
