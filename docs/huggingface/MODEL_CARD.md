---
license: apache-2.0
language:
- en
tags:
- formula-1
- motorsport
- race-strategy
- multi-agent
- langgraph
- xgboost
- lightgbm
- temporal-convolutional-network
- sports-analytics
---

# F1 StratLab Strategy Models

The machine learning models behind F1 StratLab, an open-source multi-agent system for
Formula 1 race strategy. Six LangGraph sub-agents and a ReAct orchestrator call these
models to produce pit-stop recommendations, tire-degradation forecasts, overtake and
undercut probabilities, and answers grounded in the FIA regulations.

Links:

- Project: https://f1stratlab.com
- Documentation: https://docs.f1stratlab.com
- Source code: https://github.com/VforVitorio/F1-StratLab
- Dataset: https://huggingface.co/datasets/VforVitorio/f1-strategy-dataset

## Models

| Task | Algorithm | Metric |
|---|---|---|
| Lap-time prediction | XGBoost | MAE 0.410 s |
| Tire degradation | TCN with Monte Carlo Dropout | P10/P50/P90 quantiles, pit-window detection |
| Overtake probability | LightGBM | AUC-ROC 0.876 |
| Safety-car probability | LightGBM | classifier |
| Pit-stop duration | HistGradientBoosting (quantile) | MAE 0.487 s |
| Undercut success | LightGBM (binary) | AUC-ROC 0.771 |
| Team-radio NLP | Whisper, RoBERTa, SetFit, BERT-large | 4-stage pipeline |

## Training data

Built from telemetry, lap data and race-control messages covering 70 Grand Prix across the
2023 to 2025 seasons, taken from the FastF1 and OpenF1 public APIs. The seasons are not
interchangeable: every model **trains on 2023 and 2024** and is **tested on 2025**, which is
the holdout the shipped system infers on (`train_seasons: [2023, 2024]`, `test_season: 2025`
in each model config). Quoting a figure that pools all three is partly the system reading
back its own training data. The processed data is
published as a companion dataset: VforVitorio/f1-strategy-dataset.

## Intended use

Research and educational use for Formula 1 strategy analysis. Not affiliated with Formula 1,
the FIA or any team. Predictions are estimates, not guarantees.

## Citation

```bibtex
@misc{vegasobral2026f1stratlab,
  author = {Vega, V{\'i}ctor},
  title  = {F1 StratLab: AI Models for Strategy Recommendations in Formula 1 Races},
  year   = {2026},
  note   = {Bachelor's Thesis, Intelligent Systems Engineering, UIE Campus Coru{\~n}a},
  url    = {https://f1stratlab.com}
}
```

## License

Apache 2.0. Author: Víctor Vega (https://github.com/VforVitorio).
