# F1 StratLab: Project Index

_Revolutionising strategic decision-making in Formula 1 through AI-powered predictive models, computer vision, NLP radio analysis, and a multi-agent expert system._

The project integrates several ML stacks: XGBoost/LightGBM for race strategy signals, a TCN for tyre degradation, Whisper + BERT for radio communications, and YOLOv8 for team identification, into a unified **Strategy Orchestrator** that produces real-time race recommendations. A companion telemetry app (FastAPI + Streamlit) exposes the models interactively.

The current development phase (N25–N31) replaces the legacy Experta rule engine with a **LangGraph multi-agent architecture**: specialised sub-agents (pace, tyre, overtake, safety car, pit strategy, radio NLP, regulation RAG) coordinate under a Supervisor Orchestrator.

> For full documentation see the [README](README.md) and the [DeepWiki](https://deepwiki.com/VforVitorio/F1-StratLab). For the deep reference (methodology, metrics, design rationale): the **TFG thesis + IEEE technical report** in [`documents/thesis/`](documents/thesis/). Legacy paper: [F1_Strategy_Manager_AI.pdf](documents/docs_legacy_strat_manager/F1_Strategy_Manager_AI.pdf).

Notebooks are the primary development artefact. `src/` modules are extracted from notebooks only when they need to be imported by other notebooks or the telemetry app.

---

## Quick Start / Navigation Guide

| Goal                         | Entry point                                                                                                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Download raw race data       | [`scripts/download_data.py`](scripts/download_data.py)                                                                                                              |
| Download FIA regulation PDFs | [`scripts/download_fia_pdfs.py`](scripts/download_fia_pdfs.py)                                                                                                      |
| Build RAG vector index       | [`scripts/build_rag_index.py`](scripts/build_rag_index.py)                                                                                                          |
| Full data pipeline           | N01 → N02 → N03 → N04                                                                                                                                               |
| Strategy ML models           | N05-N06 (pace) → N07-N10 (tires) → N11-N16 (overtake / SC / pit)                                                                                                    |
| NLP pipeline for radio       | N17-N24                                                                                                                                                             |
| Multi-agent system           | N25 (Pace) → N30 (RAG) → N26-N29 → N31 (Orchestrator)                                                                                                               |
| Query RAG at runtime         | [`src/rag/retriever.py`](src/rag/retriever.py): `RagRetriever` + `query_rag_tool`                                                                                  |
| Telemetry web app            | [`src/telemetry/backend/main.py`](src/telemetry/backend/main.py) (FastAPI) + [`src/telemetry/webapp/`](src/telemetry/webapp/) (React), launched together with `f1-webapp`                              |

---

## Notebooks

### Data Engineering (`notebooks/data_engineering/`)

| Notebook                                                                                  | Description                                                                                                                                                   |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [N01_data_download.ipynb](notebooks/data_engineering/N01_data_download.ipynb)             | Downloads 46 GPs (2023-2024) from FastF1 and OpenF1 APIs; outputs raw parquets under `data/raw/`                                                              |
| [N02_eda_master.ipynb](notebooks/data_engineering/N02_eda_master.ipynb)                   | Global EDA across all 46 GPs, lap time distributions, data quality audit, cross-season patterns                                                              |
| [N03_circuit_clustering.ipynb](notebooks/data_engineering/N03_circuit_clustering.ipynb)   | K-means clustering of circuits into 4 archetypes (street / high-speed / technical / balanced); produces `circuit_clusters_k4.parquet`                         |
| [N04_feature_engineering.ipynb](notebooks/data_engineering/N04_feature_engineering.ipynb) | Full feature engineering pipeline from raw parquets to `laps_featured_<year>.parquet`; integrates interval gaps, cluster assignments, and anti-drift features |

### Lap Time Prediction (`notebooks/strategy/lap_time_prediction/`)

| Notebook                                                                                  | Description                                                                                        |
| ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| [N05_laptime_eda.ipynb](notebooks/strategy/lap_time_prediction/N05_laptime_eda.ipynb)     | EDA for the lap time model, concept drift analysis across seasons, feature selection for N06      |
| [N06_laptime_model.ipynb](notebooks/strategy/lap_time_prediction/N06_laptime_model.ipynb) | XGBoost delta-lap-time predictor; MAE 0.392 s on 2025 test set; exports to `data/models/lap_time/` |

### Tire Degradation (`notebooks/strategy/tire_degradation/`)

| Notebook                                                                                                           | Description                                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [N07_tiredeg_eda.ipynb](notebooks/strategy/tire_degradation/N07_tiredeg_eda.ipynb)                                 | EDA of tire degradation patterns by compound, stint length, and circuit, informs TCN architecture choices                                                     |
| [N08_tiredeg_sequence_config.ipynb](notebooks/strategy/tire_degradation/N08_tiredeg_sequence_config.ipynb)         | Analytical determination of optimal TCN window size per compound from empirical stint length distributions                                                     |
| [N09_tiredeg_tcn.ipynb](notebooks/strategy/tire_degradation/N09_tiredeg_tcn.ipynb)                                 | Global Causal TCN that predicts `FuelAdjustedDegAbsolute` (cumulative seconds lost to rubber wear) one step ahead; exports `tiredeg_modelA_v4.pt`              |
| [N10_tiredeg_compound_finetuning.ipynb](notebooks/strategy/tire_degradation/N10_tiredeg_compound_finetuning.ipynb) | Per-compound fine-tuning of the N09 global TCN (C1-C5); MC Dropout uncertainty + Platt calibration; exports compound models to `data/models/tire_degradation/` |

### Overtake Probability (`notebooks/strategy/overtake_probability/`)

| Notebook                                                                                     | Description                                                                                                                                                 |
| -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [N11_overtake_eda.ipynb](notebooks/strategy/overtake_probability/N11_overtake_eda.ipynb)     | Builds the labeled car-pair dataset (28,494 pairs, 2023-2025); EDA of overtake rates by DRS window, gap, pace delta                                         |
| [N12_overtake_model.ipynb](notebooks/strategy/overtake_probability/N12_overtake_model.ipynb) | LightGBM binary classifier for P(overtake\| lap state); AUC-PR 0.5491 / AUC-ROC 0.8758; exports to `data/models/overtake_probability/`                      |
| [N12B_overtake_tcn.ipynb](notebooks/strategy/overtake_probability/N12B_overtake_tcn.ipynb)   | **Archived negative result**: Causal TCN on 8-lap battle sequences; AUC-PR ~0.10 vs LightGBM 0.55; confirms feature-engineered N12 is the production model |

### Safety Car Probability (`notebooks/strategy/sc_probability/`)

| Notebook                                                                   | Description                                                                                                                                                                                     |
| -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [N13_sc_eda.ipynb](notebooks/strategy/sc_probability/N13_sc_eda.ipynb)     | Builds the labeled race-lap dataset (3,275 rows) for SC prediction; EDA of deployment rates, circuit-level base rates, and feature correlations                                                 |
| [N14_sc_model.ipynb](notebooks/strategy/sc_probability/N14_sc_model.ipynb) | LightGBM SC probability classifier (3-lap window); AUC-PR 0.0723 vs baseline 0.0432; framed as a soft contextual prior for the Strategy Agent; exports to `data/models/safety_car_probability/` |

### Pit Stop Prediction (`notebooks/strategy/pit_prediction/`)

| Notebook                                                                           | Description                                                                                                                                                            |
| ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [N15_pit_duration.ipynb](notebooks/strategy/pit_prediction/N15_pit_duration.ipynb) | HistGBT quantile regression (P05/P50/P95) for physical pit stop time; P50 MAE 0.487 s; exports three quantile models to `data/models/pit_prediction/`                  |
| [N16_undercut.ipynb](notebooks/strategy/pit_prediction/N16_undercut.ipynb)         | LightGBM binary classifier for undercut success (driver X gains net position after pit sequence); AUC-PR 0.6739 (1.95× lift); exports to `data/models/pit_prediction/` |

### NLP: Radio Analysis (`notebooks/nlp/`)

| Notebook                                                                     | Description                                                                                                                                            |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [N17_radio_labeling.ipynb](notebooks/nlp/N17_radio_labeling.ipynb)           | Manual labeling of F1 team radio messages (sentiment + intent); filters out post-race messages                                                         |
| [N18_radio_transcription.ipynb](notebooks/nlp/N18_radio_transcription.ipynb) | Transcribes raw radio audio files to text using OpenAI Whisper ASR; outputs `radios_raw.csv`                                                           |
| [N19_sentiment_vader.ipynb](notebooks/nlp/N19_sentiment_vader.ipynb)         | NLTK VADER lexicon-based sentiment baseline; benchmarked against N17 ground truth                                                                      |
| [N20_bert_sentiment.ipynb](notebooks/nlp/N20_bert_sentiment.ipynb)           | Fine-tunes `roberta-base` on labeled radio messages for 3-class sentiment; 87.5% test accuracy                                                         |
| [N21_radio_intent.ipynb](notebooks/nlp/N21_radio_intent.ipynb)               | Intent classification (5 classes) via SetFit + ModernBERT; includes back-translation augmentation and a documented DeBERTa-v3-large negative result    |
| [N22_ner_models.ipynb](notebooks/nlp/N22_ner_models.ipynb)                   | F1-domain NER on short radio transcriptions; BERT-large CoNLL-03 BIO token classifier (F1 = 0.42); documents GLiNER zero-shot and fine-tuning failures |
| [N23_rcm_parser.ipynb](notebooks/nlp/N23_rcm_parser.ipynb)                   | Rule-based structured event extractor for FastF1 `race_control_messages`; deterministic, no ML required                                                |
| [N24_nlp_pipeline.ipynb](notebooks/nlp/N24_nlp_pipeline.ipynb)               | Unified inference pipeline merging N20-N23: sentiment + intent + NER + RCM parsing; GPU P95 latency 59.4 ms; exports `pipeline_config_v1.json`         |
| [N33_radio_dataset_builder.ipynb](notebooks/nlp/N33_radio_dataset_builder.ipynb) | Builds the static per-GP OpenF1 radio corpus (parquets + MP3s) consumed at replay time by `src/nlp/radio_runner.py`; the CLI's `ensure_radio_corpus()` downloads this corpus lazily per GP |

### Agents (`notebooks/agents/`)

| Notebook                                                      | Description                                                                                                                                                                                 |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [N25_pace_agent.ipynb](notebooks/agents/N25_pace_agent.ipynb) | Pace Agent, wraps the N06 XGBoost model into a LangGraph ReAct agent; returns `PaceOutput` (lap time prediction + delta signals + bootstrap CI); first of seven sub-agents                 |
| [N30_rag_agent.ipynb](notebooks/agents/N30_rag_agent.ipynb)   | RAG Agent, retrieval-augmented generation over FIA Sporting and Technical Regulations (2023-2025) via local Qdrant; returns structured `RegulationContext` objects with article references |
| [N34_radio_runner_smoke.ipynb](notebooks/agents/N34_radio_runner_smoke.ipynb) | Radio runner smoke test, end-to-end validation of `src/nlp/radio_runner.py`: cache hit/miss, per-lap radio distribution, transcript sanity, and N29 round-trip via `run_radio_agent_from_state` on Bahrain 2025 (28 radios + 76 RCMs, lap 4 emits a PROBLEM alert) |

> The full multi-agent system (N25–N31) is **complete**. The importable agent + orchestrator modules live in [`src/agents/`](src/agents/) (each exposes `run_*_agent_from_state`). Notebooks N26–N29, N30B (RAG benchmark), two N31 notebooks (`N31_strategy_orchestrator.ipynb` + `N31_mc_visualization.ipynb`), N32 (smoke test), N33 (decision thresholds + calibration benchmarks), and N34 (radio runner smoke) are under `notebooks/agents/`. A different, unrelated `N33_radio_dataset_builder.ipynb` lives under `notebooks/nlp/` (see the NLP table below), the two share a number by coincidence, not by pipeline order.

---

## Source Modules

### `src/rag/`

| File                                           | Description                                                                                                                                         |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| [src/rag/retriever.py](src/rag/retriever.py)   | `RagRetriever` class (Qdrant client + BGE-M3 encoder) and `query_rag_tool` LangChain tool; requires the index built by `scripts/build_rag_index.py` |
| [src/rag/\_\_init\_\_.py](src/rag/__init__.py) | Package init                                                                                                                                        |

### `src/agents/`

**Production multi-agent system** (N25–N31): `pace_agent.py`, `tire_agent.py`, `race_situation_agent.py`, `pit_strategy_agent.py`, `radio_agent.py`, `rag_agent.py`, `strategy_orchestrator.py`. Each exposes a `run_*_agent_from_state(...)` adapter consumed by the CLI, the Arcade and the web app backend. The adapters do **not** share one signature: most take `(lap_state, laps_df)`, but `run_pace_agent_from_state` takes only `lap_state` and the orchestrator takes a third `lap_state` argument. The authoritative list is in [src/agents/README.md](src/agents/README.md), regenerated from `inspect.signature`.

**Production support module** for the orchestrator:

| File                                                                    | Description                                                                                                                                  |
| ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| [src/agents/position_projection.py](src/agents/position_projection.py) | Pure primitive that projects track position from end-of-window gaps; replaces the generic seconds/1.5 model with measured per-rival state  |

The two files below are the **legacy** `experta` rule engine, kept for reference (superseded by the LangGraph agents above):

| File                                                         | Description                                                                                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| [src/agents/base_agent.py](src/agents/base_agent.py)         | `Fact` subclasses (`TelemetryFact`, `DegradationFact`, `GapFact`, `RadioFact`, `RaceStatusFact`) for the `experta` rule engine |
| [src/agents/strategy_agent.py](src/agents/strategy_agent.py) | Legacy rule-based Strategy Agent integrating tire / lap time / radio / gap rule sets via `experta` (superseded by N31)         |

### `src/simulation/` and `src/arcade/`

| File | Description |
| ---- | ----------- |
| [src/simulation/race_state_manager.py](src/simulation/race_state_manager.py) | `RaceStateManager`, builds the per-lap `lap_state` dict (single-driver telemetry + timing-only rivals) consumed by all agents |
| [src/simulation/replay_engine.py](src/simulation/replay_engine.py) | `RaceReplayEngine`, iterates a race parquet lap by lap, yielding `lap_state` (same contract for replay or a future live feed) |
| [src/simulation/stint_history.py](src/simulation/stint_history.py) | Art. 30.5(m) (2024-25 numbering; it was 30.5(n) in 2023) stint-history helpers: answers pit-stop count, compound history, and mandatory-two-dry-compound obligation per driver and lap |
| [src/arcade/](src/arcade/) | 2D pyglet replay + PySide6 strategy dashboard + `stream.py` TCP broadcast to the dashboard subprocess |

### `src/nlp/` (legacy)

| File                                                       | Description                                                                                                                |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| [src/nlp/radio_runner.py](src/nlp/radio_runner.py)         | `RadioPipelineRunner`, replay-time consumer of the static OpenF1 radio corpus built by N33; lazily transcribes per-lap MP3 slices with Whisper (cached under `data/processed/radio_nlp/…/transcripts.json` keyed by model name) and feeds the N29 Radio Agent via `run_radio_agent_from_state`. Wired into `scripts/run_simulation_cli.py` by default |
| [src/nlp/pipeline.py](src/nlp/pipeline.py)                 | Legacy jupytext-exported NLP pipeline (pre-N24); uses old model paths and `roberta-large` intent model, superseded by N24 |
| [src/nlp/ner.py](src/nlp/ner.py)                           | NER inference wrapper                                                                                                      |
| [src/nlp/sentiment.py](src/nlp/sentiment.py)               | Sentiment inference wrapper                                                                                                |
| [src/nlp/radio_classifier.py](src/nlp/radio_classifier.py) | Radio intent classification wrapper                                                                                        |

### `src/strategy/`

`inference/engine.py` is **production**: it is the single shared implementation
of the per-lap N31 pipeline, and the CLI, Arcade, and FastAPI backend all route
through it instead of maintaining hand-mirrored copies (a real drift bug once
caused every `--no-llm` lap to crash, since a signature change was mirrored
into two of the three copies but not the third). `eval/` backs the `f1-eval`
console script. The jupytext-exported `models/` files and `training/` (empty)
are reference/historical only, see [`src/strategy/README.md`](src/strategy/README.md).

| File                                                                                           | Description                                          |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| [src/strategy/inference/engine.py](src/strategy/inference/engine.py)                           | `run_lap()`, the shared strategy inference engine consumed by the CLI, Arcade, and backend; dispatches on `profile` (`"rich"` re-drives the full N31 orchestrator sequence, `"no-llm"` is the deterministic zero-LLM-client path) |
| [src/strategy/inference/no_llm.py](src/strategy/inference/no_llm.py)                           | The deterministic `--no-llm` code path consumed by `run_lap` |
| [src/strategy/eval/](src/strategy/eval/)                                                       | `f1-eval` CLI backend, regenerates the model evaluation reports (metrics registry, calibration, threshold hygiene, NLP per-stage eval, headline-number reproduction, LLM-judged alert precision) under `documents/eval_reports/` |
| [src/strategy/inference/tire_predictor.py](src/strategy/inference/tire_predictor.py)           | Jupytext-exported tire degradation inference wrapper (N09 era; reference only) |

### `src/telemetry/`

A separate full-stack web application for live telemetry visualisation, independent of the agent notebooks.

| Component                                                                | Description                                                                                                                                |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| [src/telemetry/backend/main.py](src/telemetry/backend/main.py)           | FastAPI application entry point; mounts endpoints for telemetry, circuit domination, driver comparison, chat, voice, and strategy             |
| `src/telemetry/backend/api/`                                             | Versioned API route handlers (`telemetry`, `circuit_domination`, `comparison`, `chat`, `strategy`, the last exposes all N25-N31 agents + orchestrator over REST) |
| `src/telemetry/backend/services/`                                        | Business logic: `telemetry/` (FastF1 client + session cache), `chatbot/` (chat engine, LLM service, MCP bridge), `simulation/` (SSE strategy-replay generator), plus `comparison_service.py` |
| `src/telemetry/webapp/src/`                                              | React + TypeScript single-page app that replaced the Streamlit frontend in v2.0.0; served by nginx behind `/api` in the compose stack        |

### `src/data_extraction/`

Organised by upstream provider so the active OpenF1 path is not buried under
historical reference scripts.

#### `openf1/`, active

| File                                                                                                       | Description                                                                                                        |
| ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| [src/data_extraction/openf1/radio_dataset_builder.py](src/data_extraction/openf1/radio_dataset_builder.py) | OpenF1 team-radio + RCM + MP3 dataset builder with lap mapping, Sprint-session filtering, and circuit-suffixed slugs for multi-race countries (Italy / United States); the corpus is consumed live by `src/nlp/radio_runner.py` and reaches N29 through `scripts/run_simulation_cli.py` |
| [src/data_extraction/openf1/intervals_extractor.py](src/data_extraction/openf1/intervals_extractor.py)     | Pulls inter-car interval data from the OpenF1 `/v1/intervals` endpoint (reference script, Spain 2023 only)         |

#### `fastf1/`, reference

| File                                                                                               | Description                                                                                          |
| -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| [src/data_extraction/fastf1/session_extractor.py](src/data_extraction/fastf1/session_extractor.py) | FastF1 session loader (laps, pit stops, weather → parquet); superseded by `scripts/download_data.py` |

#### `legacy/`, kept for history, not used by any active pipeline

| File                                                                                                 | Description                                                                                          |
| ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| [src/data_extraction/legacy/image_augmentation.py](src/data_extraction/legacy/image_augmentation.py) | Albumentations augmentation pipeline for the YOLO car-team image dataset (early vision experiments)  |
| [src/data_extraction/legacy/video_downloader.py](src/data_extraction/legacy/video_downloader.py)     | yt-dlp wrapper for downloading Creative Commons F1 highlight videos                                  |

---

## Scripts

| Script                                                       | Description                                                                                                                       |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| [scripts/download_data.py](scripts/download_data.py)         | Thin wrapper to `src.f1_strat_manager.data_cache.ensure_setup`; downloads the curated subset (~7-8 GB) from Hugging Face Hub    |
| [scripts/download_fia_pdfs.py](scripts/download_fia_pdfs.py) | Scrapes and downloads FIA Sporting and Technical Regulation PDFs (2023-2025) into `data/rag/documents/`; falls back to known URLs |
| [scripts/build_rag_index.py](scripts/build_rag_index.py)     | One-shot ingestion: PDF → article chunks → BGE-M3 embeddings → local Qdrant collection; idempotent (hash-based deduplication)     |
| [scripts/build_radio_dataset.py](scripts/build_radio_dataset.py) | Multi-GP CLI wrapper around `RadioDatasetBuilder`; writes per-GP `radios.parquet` + `rcm.parquet` under `data/processed/race_radios/{year}/{slug}/` and downloads radio MP3s under `data/raw/radio_audio/{year}/{slug}/driver_{N}/` (default season: 2025; `--skip-audio` for parquets only) |
| [scripts/upload_radio_corpus.py](scripts/upload_radio_corpus.py) | Publish-side helper: `HfApi.upload_folder` pushes both the parquet tree (`data/processed/race_radios/…`) and the MP3 tree (`data/raw/radio_audio/…`) to `VforVitorio/f1-strategy-dataset` preserving the on-disk layout. Idempotent (content-hash dedup). Flags: `--year`, `--dry-run`, `--skip-parquets`, `--skip-audio`, `--commit-message` |
| [scripts/measure_mc_tables.py](scripts/measure_mc_tables.py) | Measure six quantitative tables (neutralisation rates, gap densities, clean-air gains, undercut bands, pit hazards, SC window duration) from raw parquets; writes `data/mc_measured_v1.json` for the projection-based Monte Carlo layer |
| [scripts/run_simulation_cli.py](scripts/run_simulation_cli.py) | Headless multi-agent simulator. Consumes the static radio corpus at replay time via `src/nlp/radio_runner.py`; `ensure_radio_corpus(year, gp_name)` lazily downloads the per-GP MP3 tree on first run. Flags: `--no-real-radios` (fall back to legacy mock injection), `--whisper-model NAME` (default `turbo`) |
| [scripts/run_webapp.py](scripts/run_webapp.py) | Console script launcher for the post-race web app (FastAPI backend + React SPA); wraps `docker compose up` and forwards CLI arguments |
