---
license: apache-2.0
language:
- en
tags:
- formula-1
- motorsport
- telemetry
- race-strategy
- sports-analytics
- fastf1
- openf1
task_categories:
- tabular-regression
- time-series-forecasting
pretty_name: F1 StratLab Strategy Dataset
---

# F1 StratLab Strategy Dataset

Training data for F1 StratLab, an open-source multi-agent system for Formula 1 race
strategy. It contains telemetry, lap data, tire stints and race-control messages turned
into model-ready features for lap-time prediction, tire-degradation modelling, overtake
and undercut probability, pit-stop duration and team-radio NLP.

Links:

- Project: https://f1stratlab.com
- Documentation: https://docs.f1stratlab.com
- Source code: https://github.com/VforVitorio/F1-StratLab
- Models: https://huggingface.co/VforVitorio/f1-strategy-models

## Coverage

- 70 Grand Prix, 2023 to 2025 seasons.
- Sources: FastF1 and OpenF1 public APIs.
- Per-lap telemetry, tire compound and age, stint structure, gaps and race-control events.

## Intended use

Research and educational use for Formula 1 strategy analysis and time-series or tabular ML.
Not affiliated with Formula 1, the FIA or any team. Raw data is subject to the terms of the
upstream FastF1 and OpenF1 sources.

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
