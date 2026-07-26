# src/strategy — Strategy Model Modules

**Read this first:** this package holds two different things with opposite statuses.

- **`inference/` is production.** `inference/engine.py::run_lap` is the single
  implementation of the N31 lap pipeline, and the CLI, the arcade and the backend
  all route through it. See
  [Strategy pipeline](https://docs.f1stratlab.com/#/arcade-strategy-pipeline).
- **`training/` and the Jupytext exports are reference**, and the status note below
  applies only to them.

## Status of the Jupytext exports: reference only

These are the Jupytext `.py` exports left in `inference/` alongside the production files. They contain the
model architectures and prediction utilities developed before the LightGBM-based
strategy models (N06–N16) replaced the earlier TCN and XGBoost experiments.

---

## Subdirectories

### `inference/`

| File | Status | Description |
|---|---|---|
| `engine.py` | **production** | `run_lap`, the single implementation of the N31 lap pipeline. The CLI, the Arcade and the backend all route through it, and it exposes two profiles: `rich` returns the verbose per-stage payloads the Arcade dashboard renders, `no-llm` runs the deterministic path with no provider call |
| `no_llm.py` | **production** | The deterministic decision path used by `profile="no-llm"`: MC argmax plus the regulatory guard-rails, no LLM synthesis |
| `tire_predictor.py` | reference | N09-era `EnhancedTCN` PyTorch module (dilated conv1d, multi-scale, MC Dropout) and `predict_tire_degradation()`; loads a `.pth` state dict. Superseded by the per-compound models from N10 |

### `eval/`

Backend for the `f1-eval` CLI. Regenerates the model evaluation reports under
`documents/eval_reports/`: the metrics registry, calibration, threshold hygiene,
per-stage NLP evaluation, headline-number reproduction, and LLM-judged alert
precision.

### `training/`

Empty. Training code lives in the notebooks.

---

## Production models

The current production tire degradation model is the per-compound fine-tuned TCN
from N10, exported to `data/models/tire_degradation/`. The lap time model is the
XGBoost delta predictor from N06 (`data/models/lap_time/`).

The **Jupytext export** files pre-date those exports and use different model
architectures or APIs. Do not rely on them for inference in agent code. This does
not apply to `inference/`, which is the production path.

---

## Developed in

- [`notebooks/strategy/lap_time_prediction/N06_laptime_model.ipynb`](../../notebooks/strategy/lap_time_prediction/N06_laptime_model.ipynb)
- [`notebooks/strategy/tire_degradation/N09_tiredeg_tcn.ipynb`](../../notebooks/strategy/tire_degradation/N09_tiredeg_tcn.ipynb)
- [`notebooks/strategy/tire_degradation/N10_tiredeg_compound_finetuning.ipynb`](../../notebooks/strategy/tire_degradation/N10_tiredeg_compound_finetuning.ipynb)
