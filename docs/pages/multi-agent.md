# Multi-Agent Strategy Architecture (N25–N31)

**The F1 StratLab multi-agent system is a LangGraph pipeline of six sub-agents (N25–N30) and one orchestrator (N31) that turns a per-lap `lap_state` into a typed `StrategyRecommendation`**, fusing ML model inference, Monte Carlo simulation and LLM synthesis across three layers.

## Purpose

The multi-agent system replaces the legacy Experta rule engine (`base_agent.py`, `strategy_agent.py`) with a LangGraph-based pipeline that combines ML model inference, Monte Carlo simulation, and LLM-driven synthesis to produce race strategy recommendations.

## System overview

```mermaid
graph TD
    RSM[RaceStateManager] -->|lap_state dict| ORCH[Strategy Orchestrator N31]

    subgraph "Layer 1 — Always-On Agents"
        N25[N25 Pace Agent<br/>XGBoost + Bootstrap CI]
        N26[N26 Tire Agent<br/>TireDegTCN + MC Dropout]
        N27[N27 Race Situation Agent<br/>LightGBM Overtake + SC]
        N29[N29 Radio Agent<br/>RoBERTa + SetFit + BERT NER]
    end

    subgraph "Layer 1 — Conditional Agents (MoE Routing)"
        N28[N28 Pit Strategy Agent<br/>N15 Quantiles + N16 Undercut]
        N30[N30 RAG Agent<br/>Qdrant + BGE-M3]
    end

    ORCH --> N25
    ORCH --> N26
    ORCH --> N27
    ORCH --> N29

    N26 -->|tire_warning == PIT_SOON| N28
    N29 -->|PROBLEM or WARNING alert| N28
    N27 -->|sc_prob > 0.30| N30
    N28 -->|always when N28 active| N30

    subgraph "Layer 2 — Monte Carlo Simulation"
        MC[500 draws x 4 candidates<br/>STAY_OUT / PIT_NOW / UNDERCUT / OVERCUT<br/>score = alpha * E + 1-alpha * P10]
    end

    subgraph "Layer 3 — LLM Synthesis"
        LLM[ChatOpenAI.with_structured_output<br/>StrategyRecommendation]
    end

    N25 --> MC
    N26 --> MC
    N27 --> MC
    N28 --> MC
    MC --> LLM
    N29 --> LLM
    N30 --> LLM
    LLM --> REC[StrategyRecommendation]
```

**Routing rules (text equivalent of the diagram above):**

- The orchestrator always runs the four always-on agents: N25 Pace, N26 Tire, N27 Race Situation and N29 Radio.
- N28 Pit Strategy activates when N26 reports `tire_warning == PIT_SOON`, when N29 raises a PROBLEM or WARNING alert, or when N27 reports an active Safety Car.
- N30 RAG activates when N27 reports `sc_prob > 0.30`, when N28 is active, or under an active Safety Car.
- Monte Carlo then draws 500 samples over four candidates (STAY_OUT, PIT_NOW, UNDERCUT, OVERCUT), scoring `score = α·E + (1−α)·P10`, and the LLM synthesises the final `StrategyRecommendation`.

## Three-window arcade

Since Phase 3.5 Proceso B (April 2026), the `python -m src.arcade.main ... --strategy` launcher runs three windows driven by one shared telemetry stream. The layout is:

```mermaid
graph LR
    subgraph arcade["Arcade process (pyglet)"]
        REPLAY[F1ArcadeView<br/>race replay]
        PIPE[StrategyPipeline<br/>local N31 copy]
        STREAM[TelemetryStreamServer<br/>TCP 127.0.0.1:9998]
    end

    subgraph qt["Dashboard subprocess (single QApplication)"]
        DASH[Strategy Dashboard<br/>QMainWindow]
        TELE[Live Telemetry<br/>QMainWindow 2x2 pyqtgraph]
    end

    REPLAY --> PIPE
    PIPE --> STREAM
    REPLAY --> STREAM
    STREAM -->|TCP broadcast ~10 Hz| DASH
    STREAM -->|TCP broadcast ~10 Hz| TELE
```

Four properties are load-bearing:

1. **The arcade owns the `TelemetryStreamServer`.** `src/arcade/stream.py` exposes the merged arcade + strategy snapshot; every other window is a subscriber, never the source of truth.
2. **One subprocess hosts both Qt windows.** The arcade spawns a single `subprocess.Popen` that boots one `QApplication`. Two windows inside one event loop is cheaper than two OS processes and avoids duplicated imports of PySide6 + pyqtgraph.
3. **Each window has its own `TelemetryStreamClient(QThread)`.** Subscribers do not share sockets; each window reconnects independently when the arcade restarts.
4. **Arcade runs a local strategy pipeline.** `src/arcade/strategy_pipeline.py` carries a copy of the N31 orchestrator body so the arcade does not depend on the FastAPI backend at runtime.

See [Arcade strategy pipeline](#/arcade-strategy-pipeline) for the rationale and [Arcade dashboard](#/arcade-dashboard) for the Qt-side architecture.

## Agent details

### N25 — Pace Agent (`pace_agent.py`)

Wraps the N06 XGBoost delta-lap-time model. Returns predicted lap time, delta signals against previous lap and session median, and bootstrap confidence intervals (N=200 draws with 2% Gaussian noise on continuous features).

- **Model**: XGBoost trained on 2023–2025 lap data
- **Output**: `PaceOutput` (lap_time_pred, delta_vs_prev, delta_vs_median, ci_p10, ci_p90)

### N26 — Tire Agent (`tire_agent.py`)

Wraps per-compound TireDegTCN models (N09/N10) with MC Dropout inference. Answers: how many laps remain before the degradation cliff?

- **Model**: Causal TCN per compound + Platt calibration
- **Output**: `TireOutput` (laps_to_cliff_p10/p50/p90, warning_level, deg_rate)
- **Warning levels**: OK, MONITOR, PIT_SOON, CRITICAL

### N27 — Race Situation Agent (`race_situation_agent.py`)

Combines N12 (overtake probability via LightGBM) and N14 (safety car probability via LightGBM) into a single threat assessment per lap.

- **Models**: LightGBM overtake (AUC-PR 0.5491) + LightGBM SC (AUC-PR 0.0723)
- **Output**: `RaceSituationOutput` (overtake_prob, sc_prob_3lap, threat_level, **sc_currently_active**)

#### RCM Safety Car override

The N14 LightGBM was trained to predict a *future* SC, not to recognise one already deployed. To close that gap, N27 inspects the lap's `rcm_events` (forwarded by the orchestrator from `RadioPipelineRunner`) and, when any event matches `SAFETY_CAR_DEPLOYED` or `VIRTUAL_SAFETY_CAR_DEPLOYED`, forces `sc_prob_3lap = 1.0`, sets `sc_currently_active = True`, and elevates `threat_level` to `HIGH`. Release events (`SAFETY_CAR_ENDING`, `SAFETY_CAR_IN_PIT_LANE`, `VIRTUAL_SAFETY_CAR_ENDING`) take priority in the same window so the override clears as soon as the SC ends. The override is logged in the `reasoning` field with an `[RCM OVERRIDE: ...]` prefix so the audit trail survives the chat / arcade summary path.

### N28 — Pit Strategy Agent (`pit_strategy_agent.py`)

Wraps N15 (physical pit stop duration P05/P50/P95 via HistGBT) and N16 (undercut success probability via LightGBM). Recommends when to pit, what compound to fit, and whether to undercut.

- **Models**: HistGBT quantile pit duration + LightGBM undercut
- **Output**: `PitStrategyOutput` (action, compound_recommendation, stop_duration_p05/p50/p95, undercut_prob, sc_reactive)
- **Activation**: conditional — runs when tire_warning is PIT_SOON, radio flags PROBLEM/WARNING, **or N27 reports `sc_currently_active = True`** (the RCM-override path)

#### Honoring an active Safety Car

When the orchestrator sets `sc_currently_active = True` on the lap state, N28's prompt swaps the legacy "SC probability" line for an explicit `SC STATUS: SAFETY CAR DEPLOYED RIGHT NOW` banner and the system prompt waives the `MINIMUM STINT LENGTH` constraint (pitting under SC costs ~10 s vs ~22 s, inverting the cost/benefit). A code-level guard-rail then flips any residual `STAY_OUT` to `PIT_NOW` so a misbehaving LLM can't override the deterministic signal — replicating the McLaren Catar 2025 V7 fix where the chain of safeguards previously locked the recommendation into `STAY_OUT` despite a confirmed SC.

### N29 — Radio Agent (`radio_agent.py`)

Two-stream NLP pipeline. Driver radio goes through RoBERTa-base sentiment, SetFit intent classification, and BERT-large NER. Race Control Messages go through a deterministic rule-based parser. Alerts are built deterministically from NLP is_alert flags — the LLM cannot miss or hallucinate alerts.

- **Models**: RoBERTa-base, SetFit, BERT-large-conll03 (radio); rule parser (RCM)
- **Output**: `RadioOutput` (radio_events, rcm_events, alerts, corrections)

### N30 — RAG Agent (`rag_agent.py`)

Answers regulation questions by retrieving relevant FIA Sporting Regulation passages from a local Qdrant vector store (built by `scripts/build_rag_index.py`), using BGE-M3 embeddings and a LangGraph ReAct agent.

- **Retriever**: Qdrant + BGE-M3 embeddings
- **Output**: `RegulationContext` (answer, articles, chunks)
- **Activation**: conditional — only runs when sc_prob > 0.30, N28 is active, **or N27 reports `sc_currently_active = True`** (so the orchestrator pulls the SC pit-lane regulation snippet into the recommendation context)

### N31 — Strategy Orchestrator (`strategy_orchestrator.py`)

Three-layer pipeline:

1. **MoE Routing**: deterministic if-else rules decide which conditional agents (N28, N30) to activate based on always-on agent outputs.
2. **Monte Carlo Simulation**: draws 500 samples from sub-agent probability distributions and evaluates four strategy candidates (STAY_OUT, PIT_NOW, UNDERCUT, OVERCUT). Score = alpha * E[S] + (1-alpha) * P10[S].
3. **LLM Synthesis**: structured-output LLM aggregates all reasoning strings and MC scores into a `StrategyRecommendation`.

- **Output**: `StrategyRecommendation` (action, reasoning, confidence, scenario_scores, contingencies)
- **Action values**: STAY_OUT, PIT_NOW, UNDERCUT, OVERCUT, ALERT
- **Pace modes**: PUSH, NEUTRAL, MANAGE, LIFT_AND_COAST
- **Risk levels**: AGGRESSIVE, BALANCED, DEFENSIVE

## RSM adapter pattern

Every agent exposes two entry points:

```python
# FastF1 entry point (requires populated module globals)
run_*_agent(lap_state)

# RSM adapter (no FastF1 session required, works from parquet)
run_*_agent_from_state(lap_state, laps_df)
```

The RSM adapter builds `SESSION_META` from `laps_df` and calls the same core logic. The orchestrator uses `run_strategy_orchestrator_from_state(race_state, laps_df)` for the full pipeline.

## LLM configuration

| Layer | Model | Provider |
|---|---|---|
| Sub-agents N25–N29 | gpt-4.1-mini | OpenAI or LM Studio |
| Orchestrator N31 | gpt-5.4-mini | OpenAI or LM Studio |

Set `F1_LLM_PROVIDER=openai` env var to use the real OpenAI API. Default is LM Studio at `http://localhost:1234/v1`.

## Data flow

```
data/raw/2025/<GP>/laps.parquet
       |
  RaceReplayEngine --> RaceStateManager.get_lap_state()
       |
  lap_state dict --> Strategy Orchestrator
       |
  StrategyRecommendation --> FastAPI /api/v1/strategy/recommend
       |
  JSON response --> Streamlit Strategy Page
```

## References

- Heilmeier et al. (2020) ApplSci 10/4229 — MC motorsport simulation
- Wang et al. (2024) arXiv:2406.04692 — MoA reasoning aggregation
- Liu et al. (2024) arXiv:2402.02392 — DeLLMa decision under uncertainty with LLM
