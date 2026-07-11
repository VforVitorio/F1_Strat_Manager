# Roadmap

**F1 Digital Twin Multi-Agent System - Final Degree Project**

Timeline: legacy closure May 2025; TFG active development Feb 2026 - Apr 2026; thesis delivery May - Jun 2026.

---

## Overview

This project develops an intelligent multi-agent system for Formula 1 telemetry analysis and race strategy optimization. The system integrates FastF1/OpenF1 historical telemetry, eight ML predictive models with circuit clustering, a coordinated multi-agent architecture using LangGraph, RAG-based FIA regulation knowledge, and three delivery surfaces (CLI wheel, Streamlit panel, Arcade replay). Kafka + WebSocket streaming was descoped during v0.12.

**Key Technologies:** FastAPI, FastMCP, XGBoost, LightGBM, PyTorch, LangGraph, Qdrant, BGE-M3, Streamlit, Arcade

---

## Release Strategy

Development follows an incremental approach. v0.1–v0.5 covered project setup and integration; v0.6 closed out the data engineering phase; v0.7–v0.8.2 built the ML and NLP foundations; v0.9–v0.11 delivered the multi-agent system, RAG, and CLI distribution.

**Three-release distribution model (v0.12+):** The project ships as three independent artifacts because each has different distribution mechanics:
- **R1: CLI wheel** (`f1-strat`, `f1-sim`): pip-installable wheel on GitHub Releases, lazy HF data download
- **R2: Arcade**: container deploy for interactive race replay visualization
- **R3: Streamlit + Backend**: Docker Compose (FastAPI + Streamlit + Qdrant + LM Studio) or Streamlit Cloud

---

## v0.1–v0.5 - Legacy Integration & Setup

- [X] **Status:** Completed (legacy phase, superseded by TFG development from v0.6 onwards)
- [X] **Release Dates:** v0.1: May 9, 2025; v0.5: May 23, 2025

Legacy iteration of the project, delivered as the final assignment for the third-year courses (Speech & NLP, Advanced ML, Computer Vision, Intelligent Systems). Integrated F1_Telemetry_Manager submodule and established modular project structure. Set up Docker Compose orchestration and configured base YAML configs for models and logging. A `legacy_version` branch (merged 2025-09-07) preserves this phase; TFG development proper starts at v0.6.

**Note:** Kafka + WebSocket streaming was descoped entirely during v0.12 (April 2026). REST endpoints over parquet replay are sufficient for both Streamlit and Arcade.

**Deliverables:**

- [X] Modular repository structure with src/, notebooks/, data/, legacy/
- [X] Submodule integration preserving existing telemetry backend
- [X] Python package setup with editable install
- [X] Data organization by year/race hierarchy
- [X] Base Docker Compose configuration
- [X] FastAPI backend verification (7 endpoint categories operational)

**Success Criteria:**

- [X] Clean imports from src modules
- [X] Docker Compose successfully launches base services
- [X] Project installable via pip install -e .
- [X] REST API endpoints verified and documented

---

## v0.6.0 - Data Engineering Pipeline

- [X] **Status:** Completed (first TFG release, formal start of active development)
- [X] **Release Date:** February 12, 2026

Closed out the full data engineering phase. From raw FastF1 telemetry to a clean, feature-rich dataset ready to feed into the ML models. Previous notebooks moved to `legacy/`; new structure built around TFG architecture.

**Goals:**

- [X] Download and organize 2023-2025 seasons data (N01, extended to 2025, Miami/Barcelona alias fixes)
- [X] Master EDA: data exploration, cleaning, validation
- [X] Circuit clustering using K-Means k=4, fitted on 2023–2024, serialized with joblib; 2025 inference via `kmeans.predict()` without refit (N03)
- [X] Feature engineering: 48-column dataset, ~45k clean racing laps; fuel-corrected degradation, sequential lap features, rolling 3-lap degradation, race context, circuit cluster merge (N04)
- [X] 2025 saved as held-out test set; it never touches training data

**Deliverables:**

- [X] Clean datasets in data/processed/ (2023, 2024, 2025 separate)
- [X] Circuit clusters defined and validated (`circuit_clusters_k4.parquet`, 25 circuits, 0 unknowns on 2023–2025)
- [X] notebooks/data_engineering/ with all EDA and pipeline notebooks
- [X] Dataset published to HuggingFace Hub (`VforVitorio/f1-strategy-dataset`)

**Success Metrics:**

- [X] All GPs downloaded and validated (2023–2025)
- [X] 4 circuit clusters identified with clear characteristics
- [X] Data quality checks pass (no missing critical fields)
- [X] Feature engineering pipeline reproducible

---

## v0.7.0 - ML Foundation: Lap Time & Tire Degradation

- [X] **Status:** Completed
- [X] **Release Date:** March 5, 2026
- [X] **Critical Milestone**

Developed and trained the first two ML models: lap time prediction (XGBoost) and tire degradation (TCN + MC Dropout). All experimentation in notebooks, models exported to `data/models/`.

**Lap Time Predictor (N06):**

- [X] EDA and data exploration
- [X] XGBoost delta-lap-time model with circuit clustering features
- [X] Hyperparameter tuning via GridSearch / cross-validation
- [X] Model exported to `data/models/lap_time/`
- [X] Target: MAE <0.5s. **Achieved: MAE 0.4104s on 2025 test data** ✅

**Tire Degradation Predictor (N07–N10):**

- [X] EDA and degradation analysis (N07, N08)
- [X] TCN (Temporal Convolutional Network) architecture in PyTorch (N09)
- [X] Per-compound fine-tuning (SOFT / MEDIUM / HARD)
- [X] MC Dropout for uncertainty quantification (N=50 forward passes)
- [X] Calibration JSON exported alongside model weights
- [X] Model exported to `data/models/tire_degradation/`
- [X] Target R² >0.85: **missed**. Final tire-deg MAE 0.7078s on 2025 holdout (best compound C2 0.5501s)

**Important Note - Tire Compound Mapping:**
Current data (FastF1/OpenF1) only provides relative compound names (SOFT/MEDIUM/HARD) per race. For accurate degradation predictions, actual Pirelli compounds (C1-C5) are critical since the same "MEDIUM" can be C2 (harder) or C4 (softer) depending on circuit. Future enhancement: manual mapping from [Pirelli press releases](https://press.pirelli.com) into `data/tire_compounds_by_race.json`.

**Success Metrics:**

- [X] Lap Time: MAE 0.4104s on 2025 (target <0.5s ✅ / stretch <0.3s ⬜)
- [X] Tire Degradation model operational with MC Dropout uncertainty
- [X] All experiments documented in notebooks/strategy/

---

## v0.8.0 - Additional Predictors (merged into v0.8.1)

- [X] **Status:** Completed (never tagged as a standalone release; notebooks N11-N14 shipped together with N15-N16 under the v0.8.1 tag on 2026-03-13)
- [X] **Reference:** content is the N11-N14 subset described below and the v0.8.1 section that follows

Expand ML capabilities with additional prediction models for overtake probability and safety car deployment. Sector time predictor descoped (no meaningful contribution over N06 delta model for the Strategy Agent).

**Sector Time Predictor:**

- [ ] ~~Descoped~~ (does not add value over lap delta model for Strategy Agent use case)

**Overtake Probability (N11 + N12):**

- [X] EDA and overtake pattern analysis: `N11_overtake_eda.ipynb`
- [X] 28,494 labeled pairs (2023–2025), gap ≤ 2.5s, 8.44% positive rate
- [X] LightGBM binary classifier, Optuna hyperparameter search
- [X] Platt calibration on 2024 validation set
- [X] Window simulation: P(overtake in N laps) = 1 − ∏(1 − Pₖ)
- [X] Model exported to `data/models/overtake_probability/`
- [X] Labeled dataset published to HuggingFace Hub
- [X] **Achieved: AUC-PR 0.5491, AUC-ROC 0.8758, threshold 0.80** ✅

**Safety Car Probability (N13 + N14):**

- [X] Dataset construction: `N13_sc_eda.ipynb`
  - 58 races loaded, 3,275 labeled race-lap rows; SC+VSC: 6.6% of all laps
  - Sources: `session.laps` + `session.track_status` + `session.race_control_messages`
  - Three SC targets built: `sc_within_3_laps` (3.5%), `sc_within_5_laps` (5.6%), `sc_within_7_laps` (7.5%)
  - `circuit_sc_rate` added as historical prior per circuit
  - Dataset exported: `data/processed/sc_labeled/sc_labeled_2023_2025.parquet` (43 cols, 3,275 rows)
- [X] LightGBM binary classifier + Optuna + Platt calibration: `N14_sc_model.ipynb`
  - **Achieved: AUC-PR 0.0723 (baseline 0.0432, lift 1.67×), AUC-ROC 0.6411** ✅
  - Target selected: `sc_within_3_laps` (best lift vs 5-lap 1.44×, 7-lap 1.29×)
  - Threshold (F2): 0.234 | F2=0.2537 | Precision=0.08 | Recall=0.56
  - SHAP top: lap_time_std_z > tyre_life_max > track_temp > circuit_sc_rate > air_temp
  - Framing: **soft contextual prior** for Strategy Agent, not deterministic SC predictor
- [X] Model exported to `data/models/safety_car_probability/`
  - `lgbm_sc_v1.pkl` + `calibrator_sc_v1.pkl` + `feature_list_v1.json`

**Success Metrics:**

- [X] Overtake: AUC-PR 0.5491, AUC-ROC 0.8758 (train 2023+2024 / test 2025) ✅
- [X] Safety Car: AUC-PR 0.0723, lift 1.67× over baseline, AUC-ROC 0.6411 ✅ (reframed as soft prior)
- [X] Per-cluster performance validated on 2025 test data (overtake) ✅

---

## v0.8.1 - Extended ML Models

- [X] **Status:** Completed
- [X] **Release Date:** March 13, 2026

Additional predictive models extending the ML foundation: pit stop duration quantile regression and undercut success classification. Causal TCN alternative archived as negative result.

**Battle Outcome Temporal: Causal TCN (N12B), Negative Result:**

- [X] Causal TCN implemented and trained: `notebooks/strategy/overtake_probability/N12B_overtake_tcn.ipynb`
- [X] **Result: negative.** AUC-PR ~0.10 vs N12 LightGBM 0.5491
- [X] Root cause: N12 already encodes temporal signal via `pace_delta_rolling3` / `gap_trend`; TCN cannot rediscover what is already explicit on ~18k sequences
- [X] **N12 LightGBM remains production model.** N12B archived as documented negative result. Valid finding: explicit feature engineering dominates raw sequence modeling on small datasets.

**Pit Stop Duration: Quantile Regression (N15):**

- [X] EDA integrated in same notebook
- [X] **Model:** `sklearn.HistGradientBoostingRegressor(loss='quantile')` × 3 fits (P05/P50/P95)
- [X] Target: `physical_stop_est` [2.0–4.5s], physical stop only, pit lane traversal subtracted per circuit
- [X] Features: team, year, tyre_life_in, lap_number, compound_id, compound_change, under_sc, tight_pit_box, team_year_median
- [X] Notebook: `notebooks/strategy/pit_prediction/N15_pit_duration.ipynb`
- [X] Export: `data/models/pit_prediction/hist_pit_p05/p50/p95_v1.pkl` + `model_config.json`
- [X] **Achieved: P50 MAE 0.487s vs baseline 0.555s** ✅

**Undercut Success Predictor (N16):**

- [X] Label: driver X pits before rival Y (≤5 laps) → X gains position after pit sequence = success
- [X] Dataset: 1,032 labeled pairs (2023–2025), DRY_COMPOUNDS only (SOFT/MEDIUM/HARD)
- [X] **Model:** LightGBM binary (same architecture as N12/N14) + Platt calibration
- [X] Features (13): pos_gap_at_pit, pace_delta, tyre_life_diff, circuit_undercut_rate, lap_race_pct, compound_x/y_id, compound_delta, pit_duration_delta, circuit_undercut_rate (target enc), team_x_undercut_rate (target enc)
- [X] SHAP top: pos_gap_at_pit > pace_delta > circuit_undercut_rate > tyre_life_diff
- [X] Notebook: `notebooks/strategy/pit_prediction/N16_undercut.ipynb`
- [X] Export: `data/models/pit_prediction/lgbm_undercut_v1.pkl` + `calibrator_undercut_v1.pkl` + `model_config_undercut_v1.json`
- [X] **Achieved: AUC-PR 0.6739, AUC-ROC 0.7708, threshold 0.522** ✅

**Success Metrics:**

- [X] N12B Causal TCN: archived. AUC-PR ~0.10, N12 production model unchanged ✅
- [X] N15 Pit Duration: P50 MAE 0.487s (target <0.5s ✅)
- [X] N16 Undercut: AUC-ROC 0.7708 (target >0.75 ✅)

---

## v0.8.2 - NLP Radio Processing Pipeline

- [X] **Status:** Completed
- [X] **Release Date:** March 22, 2026

NLP pipeline for the Radio Agent: converts raw team radio audio into structured signals (sentiment, intent, F1 entities) consumed by the Strategy Agent. Legacy notebooks `legacy/notebooks/NLP_radio_processing/N00-N06` migrated and updated to `notebooks/nlp/N17-N23`, plus a new N24 notebook for Race Control Messages.

**Pipeline architecture:**

```
Audio (MP3/WAV) → N18 Whisper ASR → text
                                      ├─► N20 BERT Sentiment
                                      ├─► N21 Intent Classifier
                                      └─► N22 Custom NER (F1 entities)
                                                    └─► N23 Merging → JSON output

N24 Race Control Messages → structured SC/VSC/flags/penalties
```

**N17: Data Labeling & Dataset Radio:**

- [X] Label transcriptions with intent + sentiment + entities
- [X] Source: `VforVitorio/f1-strategy-dataset` (HuggingFace)
- [X] Notebook: `notebooks/nlp/N17_radio_labeling.ipynb`

**N18: Radio Transcription (Whisper ASR):**

- [X] Whisper ASR for F1 radio transcription
- [X] Notebook: `notebooks/nlp/N18_radio_transcription.ipynb`

**N19: Sentiment Baseline (VADER):**

- [X] Rule-based VADER baseline benchmark
- [X] Notebook: `notebooks/nlp/N19_sentiment_vader.ipynb`

**N20: RoBERTa Sentiment Fine-tuning:**

- [X] Fine-tuned `roberta-base`: 3-class sentiment on labeled radio messages
- [X] **Achieved: 0.84 test accuracy (macro-F1 0.75)** ✅
- [X] Export: model state dict to `data/models/nlp/`
- [X] Notebook: `notebooks/nlp/N20_bert_sentiment.ipynb`

**N21: Intent Classification:**

- [X] 5 intent classes via SetFit + ModernBERT; back-translation augmentation; DeBERTa-v3-large negative result documented
- [X] Notebook: `notebooks/nlp/N21_radio_intent.ipynb`

**N22: Custom NER (F1 Entities):**

- [X] BERT-large CoNLL-03 BIO token classifier; GLiNER zero-shot negative result documented
- [X] **Achieved: F1 = 0.42** (short radio transcriptions, limited training data)
- [X] Notebook: `notebooks/nlp/N22_ner_models.ipynb`

**N23: RCM Parser (Rule-based):**

- [X] Deterministic structured event extractor for `session.race_control_messages`; no ML required
- [X] Notebook: `notebooks/nlp/N23_rcm_parser.ipynb`

**N24: Unified NLP Pipeline:**

- [X] `run_pipeline(text)` → sentiment + intent + NER | `run_rcm_pipeline(rcm_row)` → structured event
- [X] **Achieved: GPU P95 latency 59.4 ms** ✅ (target <500 ms)
- [X] Export: `data/models/nlp/pipeline_config_v1.json`
- [X] Notebook: `notebooks/nlp/N24_nlp_pipeline.ipynb`

**Success Metrics:**

- [X] N20 RoBERTa Sentiment: 0.84 test accuracy (macro-F1 0.75) ✅
- [X] N21 Intent: SetFit 5-class classifier operational ✅
- [X] N22 NER: F1 = 0.42 (short-text constraint documented) ✅
- [X] N24 Pipeline: GPU P95 latency 59.4 ms (target <500 ms ✅)

---

## v0.9.0 - src/ Extraction & CLI Distribution

- [X] **Status:** Completed
- [X] **Release Date:** March 17, 2026

Extracted N25-N31 agent entry points to importable `src/agents/` modules. Built headless CLI simulation (`f1-sim`) with Rich Live rendering. Integrated OpenF1 team radio corpus with Whisper transcription pipeline. Published dataset and models to HuggingFace Hub.

**Agent extraction (all complete):**

1. [X] `src/agents/pace_agent.py`: `run_pace_agent()` → `PaceOutput`
2. [X] `src/agents/tire_agent.py`: `run_tire_agent()` → `TireOutput` (TireDegTCN bundles)
3. [X] `src/agents/race_situation_agent.py`: `run_race_situation_agent()` → `RaceSituationOutput`
4. [X] `src/agents/radio_agent.py`: `run_radio_agent()` → `RadioOutput` (3 NLP models)
5. [X] `src/agents/pit_strategy_agent.py`: `run_pit_strategy_agent()` → `PitStrategyOutput`
6. [X] `src/agents/rag_agent.py`: `run_rag_agent()` → `RegulationContext` (wraps src/rag/)
7. [X] `src/agents/strategy_orchestrator.py`: `run_strategy_orchestrator()` → `StrategyRecommendation`

**CLI simulation (`scripts/run_simulation_cli.py`):**

- [X] Rich Live lap-by-lap rendering with inference detail panel
- [X] Decision column: `ACTION·PACE·RISK` + Plan column (`→ L8 HARD vs NOR`)
- [X] No-LLM mode: ML models + MC simulation only, no API keys required
- [X] LLM mode: Full N31 orchestrator synthesis via OpenAI/LM Studio
- [X] Lap-1 hardening: `_get_lap_row` fallback, `_clamp_triangular`, incomplete-data guard
- [X] F1 strategic guard-rails: pit window (laps 5-last 3), minimum stint, compound-vs-distance, opening-lap threat discount, REACTIVE_SC only on confirmed SC

**Radio corpus pipeline (Track A):**

- [X] `src/f1_strat_manager/gp_slugs.py`: GP name → corpus slug resolution
- [X] `src/nlp/radio_runner.py`: `RadioPipelineRunner` + `WhisperTranscriber` + JSON cache
- [X] `src/f1_strat_manager/data_cache.py`: `ensure_radio_corpus()` lazy per-GP downloader
- [X] OpenF1 slug disambiguation for multi-race countries (Italy, United States)
- [X] Radio corpus published: 529 MP3s + 48 parquets on HuggingFace Hub

**CLI distribution:**

- [X] `pyproject.toml` with `[project.scripts]` entry points (`f1-strat`, `f1-sim`)
- [X] Lazy first-run data download from HuggingFace Hub (`ensure_setup()`)
- [X] Installable via `uv tool install git+https://github.com/VforVitorio/F1_Strat_Manager.git`

**Success Metrics:**

- [X] All 7 `run_*` agent functions importable from `src/agents/`
- [X] CLI 4-gate test: Sakhir, Sakhir LLM, Spielberg VER, Imola. All pass
- [X] Linting passes (ruff)
- [X] Typecheck passes (mypy)

---

## v0.10.0 - Multi-Agent System

- [X] **Status:** Completed
- [X] **Release Date:** March 22, 2026

LangGraph multi-agent architecture replacing the legacy Experta rule engine. Seven specialised sub-agents (N25–N30) coordinate under a Supervisor Orchestrator (N31). Each agent wraps one or more ML models as `@tool`-decorated LangChain tools and returns a typed dataclass output including a `reasoning` field forwarded to N31.

N31 architecture has three layers: (1) dynamic MoE-style routing, which only activates the sub-agents relevant to the current race state; (2) Monte Carlo simulation, which samples from the probabilistic outputs of N25–N28 (bootstrap CI, MC Dropout P10/P50/P90, Platt-calibrated probabilities, quantile regression intervals) to rank strategy candidates by risk-adjusted expected outcome; (3) LLM synthesis, which aggregates all sub-agent reasoning texts plus MC scenario scores, with N30 regulation context acting as a hard constraint that eliminates illegal options before the LLM decides.

**Sub-agents:**

- [X] N25: Pace Agent, XGBoost N06 → `PaceOutput` (lap time + delta + bootstrap CI) ✅
- [X] N26: Tire Agent, TCN N09/N10 → `TireOutput` ✅
- [X] N27: Race Situation Agent, LightGBM N12/N14 → `RaceSituationOutput` ✅
- [X] N28: Pit Strategy Agent, N15/N16 + analytical undercut logic → `PitStrategyOutput` ✅
- [X] N29: Radio Agent, N24 NLP pipeline (N06-style synthesizer + Pydantic structured output) → `RadioOutput` ✅
- [X] N30: RAG Agent, Qdrant + BGE-M3 + LangGraph ReAct → `RegulationContext` ✅
- [X] N31: Strategy Orchestrator, LangGraph supervisor + Monte Carlo simulation layer + dynamic routing (MoE-style) ✅

**Success Metrics:**

- [X] All seven agents operational and coordinated ✅
- [X] End-to-end workflow from lap state to strategy recommendation ✅
- [X] Successful demo with historical race data (Bahrain 2025 multi-lap replay) ✅

---

## v0.11.0 - RAG System

- [X] **Status:** Completed
- [X] **Release Date:** March 30, 2026

Retrieval-augmented generation over FIA Sporting Regulations (2023–2025). Provides normative support for strategic decision-making. Implemented as N30 (notebook) + `src/rag/retriever.py` (importable module for N31).

**Implementation:**

- [X] `scripts/download_fia_pdfs.py`: scrapes FIA Sporting Reg PDFs into `data/rag/documents/`
- [X] `scripts/build_rag_index.py`: PDF → chunks → BGE-M3 embeddings → Qdrant local collection
- [X] `src/rag/retriever.py`: `RagRetriever` class + `query_rag_tool` LangChain tool
- [X] N30 notebook: LangGraph ReAct agent demo; `RegulationContext` structured output

**Technical Details:**

- [X] Embeddings: `BAAI/bge-m3` (1024-dim, RTX 5070)
- [X] Chunk size: 512 characters with 64-character overlap
- [X] Top-k: 5 chunks per query | 2,279 chunks indexed (3 PDFs)
- [X] Export: `data/models/agents/rag_agent_config_v1.json`

**Success Metrics:**

- [X] RAG retrieves relevant regulation passages (scores 0.62–0.76 on demo queries) ✅
- [X] `query_rag_tool` importable by N31 via `from src.rag.retriever import query_rag_tool` ✅
- [X] `RegulationContext.articles` provides reliable article citations from chunk metadata ✅

---

## v0.1.1 - R1 CLI Wheel Release

- [X] **Status:** Completed
- [X] **Release Date:** April 9, 2026

Intermediate tag that formalises the R1 distribution artifact built during v0.9. Ships the CLI wheel `f1_strat_manager-0.1.1-py3-none-any.whl` on GitHub Releases alongside the `uv tool install git+<repo>` install path. Scope strictly bundling/distribution; no functional or model changes relative to v0.11.0.

**Deliverables:**

- [X] `pyproject.toml` version bump to 0.1.1
- [X] Wheel built via `uv build` and attached to the v0.1.1 GitHub Release
- [X] `f1-strat` and `f1-sim` entry points verified post-install
- [X] README install section documents both `uv tool install git+` and offline wheel flows

---

## v0.12.0 - Interfaces & Distribution

- [X] **Status:** Completed
- [X] **Release Date:** April 15, 2026

Wire the multi-agent system into the FastAPI backend, expose strategy tools via FastMCP, build Streamlit dashboard pages, and integrate Arcade for race replay visualization. Three independent releases ship from this work (R1 CLI wheel, R2 Arcade, R3 Streamlit + Backend).

**Completed:**

- Phase 3.5 Proceso A ✅ 2026-04-15: `TelemetryStreamServer` (TCP :9998) + `StrategyState.snapshot_dict` + arcade broadcast wired in `F1ArcadeView.on_update`
- Phase 3.5 Proceso B ✅ 2026-04-18: PySide6 dashboard subprocess spawns both `MainWindow` (orchestrator + 6 sub-agent cards + reasoning tabs) and `TelemetryWindow` (2×2 circuit-comparison grid) from a single `f1-arcade --strategy` command; arcade-local `src/arcade/strategy_pipeline.py` duplicates the N31 orchestrator body so the arcade no longer depends on the FastAPI backend at runtime
- Phase 3.5 polish ✅ 2026-04-18 → 2026-04-20: Pirelli compound pills, alert flag chips, FiraCode mono stack, per-lap distance + FastF1 circuit length, F1-broadcast-style Delta chart, radio corpus injection via `RadioPipelineRunner`, all-20-cars toggle (A key), docs refresh with 5 drawio diagrams

**R1: CLI Release (wheel):** ✅ DONE

- [X] `pyproject.toml` entry points (`f1-strat`, `f1-sim`) ✅
- [X] Lazy first-run HF data download (`ensure_setup()`) ✅
- [X] Wheel build via `uv build` → `dist/f1_strat_manager-*.whl` ✅
- [X] Wheel `f1_strat_manager-0.1.1-py3-none-any.whl` attached to the v0.12.0 GitHub Release assets ✅
- [X] README install section documents both `uv tool install git+` and offline wheel flows ✅

**Step 9: FastAPI wiring (`src/telemetry/backend/`):** ✅ DONE

- [X] 9a: Router `api/v1/endpoints/strategy.py` exposes all agents + orchestrator ✅
  - POST /strategy/pace, /tire, /situation, /pit, /radio, /rag, /recommend. All live
- [X] 9b: `chat.py` upgraded; strategy-intent queries route to N31 orchestrator ✅
- [X] `sys.path` fix so telemetry backend imports cleanly from `src/agents/` ✅

**Step 10: FastMCP + Streamlit chat:** ✅ DONE

- [X] FastMCP server mounted alongside FastAPI; `/chat/` is an MCP client ✅
- [X] Phase 1: agent MCP tools: `predict_pace`, `predict_tire`, `predict_situation`, `predict_pit`, `analyze_radio`, `query_regulations`, `recommend_strategy` ✅
- [X] Phase 2: telemetry MCP tools via `FastMCP.from_openapi()`: `get_lap_times`, `get_telemetry`, `compare_drivers`, `get_race_data` (HTTP fallback for chat) ✅
- [X] **2026-04-14: inline Plotly chart rendering** for the 4 Phase 2 tools in the chat: new `chart_builders.py`, `_render_chart` dispatcher, purple-outlined bubbles matching the agent cards. Backend trim split via `_trim_for_llm` so the UI receives the full payload. Qdrant singleton fix (`@lru_cache` on `get_retriever`) ✅
- [X] `pages/strategy.py`: Live strategy card (action badge, confidence bar, scenario scores, reasoning) ✅
  - Sub-agent tabs: Pace (CI ribbon), Tyres (cliff gauge), Race Situation (overtake + SC gauges), Pit Analysis (undercut + duration)
- [X] `pages/race_analysis.py`: 5-tab race view (Overview, Competitive, Gap Analysis, Degradation, Predictions) ✅
  - Port legacy components from `legacy/app_streamlit_v1/` with N25-N31 API data sources

**Step 11: CLI simulation demo (`scripts/run_simulation_cli.py`):** ✅ DONE

- [X] Rich Live lap-by-lap rendering with inference detail panel (2,387 lines) ✅
- [X] Decision column `ACTION·PACE·RISK` + Plan column (`→ L8 HARD vs NOR`) ✅
- [X] Lap-1 path hardening, strategic guard-rails applied in N26/N27/N28/N31 prompts ✅
- [X] Kafka descoped and documented; historical-only data source acknowledged ✅

**Step 11.5: Simulation SSE backend (infra for Arcade):** ✅ DONE

- [X] `src/telemetry/backend/services/simulation/`: `simulate_race` generator + `guard_rails` module duplicated from CLI L1504-L1535 (CLI untouched) ✅
- [X] `POST /api/v1/strategy/simulate`: `StreamingResponse(media_type="text/event-stream")` emitting `start` → N×`lap` → `summary` events ✅
- [X] Validated via smoke unit (6-lap assertions) + FastAPI `TestClient` stream (5 frames, 200 OK, correct content-type) + CLI regression (no drift) ✅

**Step 12: Arcade simulation UI:** ✅ COMPLETE (Phase 3.5 Proceso B, 2026-04-18)

- [X] Three windows from one command: pyglet race replay, PySide6 strategy dashboard,
      PySide6 live telemetry (2x2 pyqtgraph grid). Single launcher:
      `python -m src.arcade.main --viewer --strategy ...`
- [X] Local strategy pipeline: `src/arcade/strategy_pipeline.py` duplicates the N31
      orchestrator body with verbose outputs. The arcade no longer calls the FastAPI
      SSE endpoint at runtime
- [X] `src/arcade/stream.py::TelemetryStreamServer` broadcasts merged arcade + strategy
      state over TCP 127.0.0.1:9998 at ~10 Hz; arcade spawns ONE dashboard subprocess
      that hosts both Qt windows in a shared `QApplication` event loop
- [X] Dashboard cards: orchestrator action + confidence + plan strip, 6 sub-agent cards
      (N25-N30) rendering raw per-model outputs, pace CI band, tire stint chart,
      reasoning tabs
- [X] `src/arcade/data.py::SessionData.location` from FastF1 `session.event['Location']`;
      `get_gp_names(year)` derives per-year calendars from
      `data/tire_compounds_by_race.json`; `pyqtgraph>=0.13.0` added to pyproject
- [X] `src/arcade/main.py` loads `.env` via `dotenv` so `OPENAI_API_KEY` reaches agents;
      default LLM provider flipped from `lmstudio` to `openai` (override with
      `F1_LLM_PROVIDER`)

**Step 13: Legacy cleanup:** ⬜ Not started

- [ ] Archive `src/agents/base_agent.py`, `src/agents/strategy_agent.py`, `src/agents/rules/`
- [ ] Replace legacy jupytext `src/nlp/pipeline.py` with N24-aligned implementation

**Driver + Team selection (single-driver perspective):**

At session start, the user selects `TEAM` and `DRIVER` (e.g. McLaren / NOR). This pair feeds `RaceStateManager`, which constructs every `RaceState` from that driver's perspective. All downstream agents operate within this boundary automatically.

**Arcade Visualization (R2):** ✅ shipped via `uv tool install` → `f1-arcade`

- [X] 2D circuit layout rendering with real-time car positions (all 20 cars, toggle `A` to hide 18 background dots)
- [X] DRS zone overlays + pit lane visualization (reference lap = quali fastest per f1_replay pattern)
- [X] Frame streaming from `RaceReplayEngine` at 10Hz (TCP broadcast on 127.0.0.1:9998)
- [X] Distribution: `uv tool install git+<repo>` exposes `f1-arcade` console script. Container deploy descoped. OpenGL + Qt through X forwarding is fragile cross-platform and offers no upside over the host install (`INSTALL.md` documents the rationale)

**Voice Mode: low-latency upgrade (optional):**

- [ ] **GPT-4o Realtime API** (preferred; integrates with existing OpenAI SDK, ~200-300ms)
- [ ] **Moshi** (Kyutai, open-source, local GPU, ~160ms full-duplex, offline fallback)
- [ ] Keep N24 NLP pipeline active for text-based analysis in parallel

**Streaming (Kafka + WebSocket): descoped, optional extension:**

- [ ] ~~Add WebSocket endpoints to FastAPI backend (hybrid REST + WebSocket)~~
- [ ] ~~MVP: /ws/replay endpoint for offline race replay @ 10Hz~~
- [ ] ~~Extension: /ws/live endpoint with Kafka consumer for real-time data~~

**Note:** Kafka + WebSocket streaming descoped from core TFG scope (April 2026). All data is historical replay from parquet; REST endpoints are sufficient for both Streamlit and Arcade. Kafka adds infrastructure complexity (ZooKeeper, broker, topics) without a real-time data source to justify it. If implemented, it would be as a final architectural demo showing the system could scale to live telemetry (e.g. OpenF1 API during a live race). See `documents/dev_docs/tasks/planning/PLANIFICACION_DETALLADA_TFG_v2.md` Phase 7.4 for full rationale.

**R3: Streamlit + Backend Release:**

- [ ] Docker Compose: FastAPI backend + Streamlit frontend + Qdrant + Kafka + LM Studio sidecar
- [ ] Alternative: Streamlit Cloud + hosted FastAPI
- [ ] Legacy cleanup: archive `base_agent.py`, `strategy_agent.py`, `rules/`; update `src/nlp/pipeline.py` to match N24

**Success Metrics:**

- [ ] Strategy endpoints return valid agent outputs via REST
- [ ] FastMCP tools callable from `/chat/` with structured rendering
- [ ] Streamlit load time <3 seconds
- [ ] Arcade maintains >30 FPS during race replay

---

## v0.13.0 - Testing & Validation (descoped as a standalone tag)

- [X] **Status:** Descoped. Testing and validation activities were folded into v1.0.0 (2026-04-20). No separate v0.13.0 tag exists.
- [X] **Reason:** the deliverables originally planned here were completed together with the final release, as the testing work sat on top of the v0.12.0 interfaces and could not be separated in time without delaying the thesis.

**Test scope actually executed (absorbed into v1.0.0):**

- [X] End-to-end CLI simulation with no-LLM and LLM modes on representative 2025 races
- [X] Smoke tests for agent imports and Arcade dashboard subprocess (`tests/test_agents.py`, `tests/arcade/*`)
- [X] FastAPI + FastMCP integration path validated via `TestClient` and manual chat interactions
- [X] Historical replay of the Bahrain 2025 GP used as the primary qualitative demo

**Work originally planned here that remains open for future iterations (outside the TFG scope):**

- [ ] Systematic per-cluster validation (Monaco, Monza, Spielberg, Singapore 2025) with documented tolerances
- [ ] Load and memory profiling targets (>100 req/s, p95 <50ms, <4 GB peak)
- [ ] CI pipeline running the full test suite across agents + arcade + streamlit

---

## v1.0.0 - Final Release

- [X] **Status:** Completed (software release; thesis delivery follows the UIE calendar)
- [X] **Release Date:** April 20, 2026

Code freeze of the TFG software. Consolidates the interfaces closed in v0.12.0 (Arcade MVP, Streamlit + FastAPI + FastMCP, CLI) and the multi-agent + RAG stack from v0.10-v0.11 into a single, tagged production release. Thesis document, defense materials and demonstration video are finalised separately and delivered in the May-June 2026 UIE submission window.

**Software deliverables:** ✅

- [X] R1: CLI wheel on GitHub Releases (`f1-strat`, `f1-sim`), tagged under v0.1.1 and bundled with v0.12.0 assets
- [X] R2: Arcade replay distributed via `uv tool install git+<repo>` → `f1-arcade` console script (OpenGL/Qt container path dropped)
- [X] R3: Streamlit + FastAPI backend operational (Docker Compose reference bundle; Streamlit Cloud path available)
- [X] Seven coordinated LangGraph agents (N25-N31) with RAG grounding and Monte Carlo ranking
- [X] End-to-end qualitative demo: Bahrain 2025 GP replay with 28 radio messages and 76 RCMs

**Documentation deliverables (tracked outside this tag):**

- [ ] Thesis document with methodology, results, and conclusions (submitted via UIE portal)
- [ ] Defense presentation (~20 slides)
- [ ] 5-minute demonstration video (CLI + Streamlit + Arcade)
- [ ] Technical documentation: API docs, deployment guide

**Success Criteria:**

- [X] All three software release artifacts installable/deployable from scratch
- [X] Code repository production-ready with comprehensive documentation inside `docs/`
- [ ] Thesis submitted on time (pending UIE deadline)
- [ ] Demonstration showcases: CLI inference, Streamlit dashboard, Arcade replay, voice mode

---

## Key Milestones

| Release | Date         | Milestone                     | Criteria                                                                           | Status |
| ------- | ------------ | ----------------------------- | ---------------------------------------------------------------------------------- | ------ |
| v0.5    | 2025-05-23   | Legacy Integration Complete   | Legacy codebase frozen, `legacy_version` branch preserved                          | ✅     |
| v0.6    | 2026-02-12   | Data Engineering Complete     | 4 clusters, 45k laps, 2025 held-out, HuggingFace published                         | ✅     |
| v0.7    | 2026-03-05   | Base Models Complete          | Lap Time MAE 0.4104s ✅ / Tire Deg TCN + MC Dropout ✅                              | ✅     |
| v0.8.1  | 2026-03-13   | Extended ML Models            | Overtake ✅ / SC ✅ / N15 MAE 0.487s / N16 AUC-ROC 0.7708 / N12B archived           | ✅     |
| v0.9    | 2026-03-17   | src/ Extraction + CLI + Radio | 7 agents extracted, CLI sim, radio corpus, HF lazy download, guard-rails           | ✅     |
| v0.8.2  | 2026-03-22   | NLP Radio Pipeline            | N17–N24: RoBERTa 0.84 acc / SetFit intent / BERT NER / pipeline P95 59.4ms         | ✅     |
| v0.10   | 2026-03-22   | Multi-Agent Operational       | N25–N31 all complete, Bahrain 2025 end-to-end demo ✅                              | ✅     |
| v0.11   | 2026-03-30   | RAG Integrated                | 2,279 chunks indexed, BGE-M3, `src/rag/` module complete                           | ✅     |
| v0.1.1  | 2026-04-09   | R1 CLI Wheel Release          | Tagged wheel on GitHub Releases, `uv tool install git+` works                      | ✅     |
| v0.12   | 2026-04-15   | Interfaces + Distribution     | FastAPI endpoints, FastMCP tools, Streamlit pages, Arcade replay (3 windows)       | ✅     |
| v0.13   | (none)       | Testing + Validation          | Descoped as standalone tag; testing folded into v1.0.0                             | ⤴️     |
| v1.0    | 2026-04-20   | Final Release (Software)      | Arcade MVP + multi-agent system tagged; thesis delivery on UIE calendar            | ✅     |

---

## Risk Mitigation

**Concept Drift (2024-2025):** Addressed via temporal features and cluster-based normalization. Continuous monitoring of model performance on 2025 data.

**LLM Latency:** Use quantized 7B models with INT8 precision. Target inference <2s. Fallback to smaller models if necessary.

**Kafka Streaming Reliability:** ~~Implement buffering and retry logic.~~ Descoped; only relevant if live telemetry extension is implemented post-TFG.

**Test Data Availability:** If 2025 race data incomplete, use late 2024 season as fallback test set.

---

## Success Metrics

**ML Models:**

- Lap Time: target MAE <0.3s. **Achieved MAE 0.4104s** (within <0.5s tolerance ✅)
- Tire Degradation: target R² >0.85. **Missed**. MAE 0.7078s on 2025 holdout (best compound C2 0.5501s)
- Sector Time: **descoped**
- Overtake Probability: target AUC-PR >0.50. **Achieved AUC-PR 0.5491, AUC-ROC 0.8758** ✅
- Safety Car Probability: reframed as soft prior. **Achieved AUC-PR 0.0723 (lift 1.67×), AUC-ROC 0.6411** ✅
- Battle Outcome TCN (N12B): archived. AUC-PR ~0.10, N12 LightGBM remains production ✅
- Pit Stop Duration (N15): **achieved P50 MAE 0.487s** (target <0.5s ✅)
- Undercut Success (N16): **achieved AUC-ROC 0.7708, AUC-PR 0.6739** (target >0.75 ✅)

**System Performance:**

- ~~Streaming latency: p95 <50ms~~ (descoped; optional extension)
- API throughput: >100 requests/second
- Test coverage: >70%
- Memory usage: <4GB

**User Interfaces:**

- Streamlit load time: <3 seconds
- Arcade frame rate: >30 FPS

---

## Next core releases (planned milestones)

Post-v1.5.x core milestones, in order. Versions are targets, not commitments: directions, not deadlines. Tracked as GitHub milestones on the repo.

| Version | Milestone | What it adds |
|---|---|---|
| **v1.6.0** | Modern frontend | Replace the Streamlit UI with a faster React/Vite stack; the FastAPI backend stays. A presentation-layer swap, not a functional rewrite: menus and flows stay the same. |
| **v1.7.0** | Rival Agent | A new, additive LangGraph node that predicts each nearby rival's next strategic move (pit window, compound, undercut/overcut) and feeds it to the orchestrator. Recommendations move from reactive to anticipatory. The six existing agents are untouched. |
| **v1.8.0** | Live race inference | Real-time ingestion over the OpenF1 WebSocket (the `lap_state` contract is unchanged, so agents and orchestrator don't change), plus adaptation to the 2026 technical/sporting regulation (re-cluster, re-label compounds, drift monitoring). |

### Rival Agent: the anticipatory turn (v1.7.0)

Today the system reasons about our own car and treats rivals as scenery; a good pit wall decides by anticipating the cars around it. The Rival Agent closes that gap. It reuses the existing two-driver mode (which already loads a rival's public telemetry next to ours), tire age, gap, track position and history to predict what the cars in our fight will do next. Ground truth is reconstructed from real 2024–2025 pit stops cross-referenced with telemetry; the agent is validated by ablation (with/without) against the real outcome and the actual pit-wall decision on the Grands Prix already validated in the thesis. Supporting building blocks: a rival next-move classifier, a lap-by-lap rival sequence model, situation/profile clustering with anomaly detection, a scaled pit-stop ground-truth pipeline, a neural surrogate of the Monte Carlo simulator, an RL pit-stop benchmark, and analogous-race-state retrieval.

Beyond these core releases, the project grows into a multi-repo ecosystem:

## Post-TFG: F1 StratLab ecosystem (planned, not committed scope)

Beyond the core releases above, F1 StratLab is planned to grow from a single repo into an **ecosystem** of dedicated public repositories plus Hugging Face artifacts (under the `f1stratlab` org). High-level only here; detailed planning is kept outside the public roadmap.

| Initiative | Repo | What it adds |
|---|---|---|
| LLM LoRA | `gridmind` | Unsloth LoRA fine-tune of a Gemma-family LLM on an F1 text corpus (`f1stratlab/f1-domain-corpus`) for F1-specific strategy reasoning. |
| Race-time bot | `box-bot` | Automated X/Twitter account narrating the orchestrator live during a GP. |
| Radio NLP | `radiogate` | Large-scale F1 team-radio NLP corpus (`f1stratlab/f1-team-radio-corpus`) with auto-labelling and a novel deception/bluffing signal. |
| MLOps studio | `pitlab` | Button-driven data-engineering + retraining dashboard (download → merge → inspect → retrain), clustering-aware, progressive per-GP. |
| Real-time + 2026 | core | Live OpenF1 WebSocket ingestion + retraining with FP/Qualy/Sprint + history; adaptation to the 2026 technical/sporting regulation. |

These map to the eight future-work lines in the thesis ([`documents/thesis/`](documents/thesis/)). Each non-core repo, since its name does not contain "f1stratlab", states explicitly in its README/description that it is part of the F1 StratLab ecosystem.

---

**Last Updated:** June 28, 2026
**Version:** 1.12 (added next-core-releases milestones v1.6.0–v1.8.0 + Rival Agent)
