# src/agents: Multi-Agent Strategy System

LangGraph-based multi-agent system extracted from notebooks N25–N31.
Each module is importable without a FastF1 session via its `*_from_state` RSM adapter.

---

## Module map

| File | Notebook | Role | Entry points |
|---|---|---|---|
| `pace_agent.py` | N25 | XGBoost lap-time prediction + bootstrap CI | `run_pace_agent(driver_number, lap_number, stint, ...)` (23 positional features) · `run_pace_agent_from_state(lap_state)` |
| `tire_agent.py` | N26 | TireDegTCN + MC Dropout cliff estimation | `run_tire_agent(stint_state)` · `run_tire_agent_from_state(lap_state, laps_df)` |
| `race_situation_agent.py` | N27 | LightGBM overtake prob + SC prob (N12 + N14) | `run_race_situation_agent(lap_state)` · `run_race_situation_agent_from_state(lap_state, laps_df)` |
| `pit_strategy_agent.py` | N28 | N15 pit quantiles + N16 undercut + compound recommendation | `run_pit_strategy_agent(lap_state)` · `run_pit_strategy_agent_from_state(lap_state, laps_df)` |
| `radio_agent.py` | N29 | RoBERTa sentiment + SetFit intent + BERT-large NER + RCM parser | `run_radio_agent(lap_state, persist=False)` · `run_radio_agent_from_state(lap_state, laps_df, persist=False)` |
| `rag_agent.py` | N30 | FIA regulation retrieval (Qdrant + BGE-M3 + LangGraph ReAct) | `run_rag_agent(question)` · `run_rag_agent_from_state(lap_state, laps_df=None)` |
| `position_projection.py` |, | Pure primitive: turns per-rival gaps into a projected end-of-window track position, so the decision layer scores in cars rather than in seconds. Loads no model, reads no file. | `project_positions(rivals, plan, config, pit_loss_s, cliff_laps, stop_is_neutralised=False)` · `payoff(result, current_position, config)` · `rank_targets(rivals, config, our_pit_loss_s)` |
| `strategy_orchestrator.py` | N31 | MoE routing + MC simulation + LLM synthesis | `run_strategy_orchestrator(race_state, lap_state)` · `run_strategy_orchestrator_from_state(race_state, laps_df, lap_state=None)` |

### The arcade does not carry a copy

`src/arcade/strategy_pipeline.py` delegates to the shared inference engine
(`src/strategy/inference/engine.py`), the same `run_lap` the CLI and the FastAPI
backend call. It adds only the verbose per-stage payloads the arcade dashboard
cards need on top of that one result.

There is nothing to mirror by hand. This README previously instructed
contributors to transcribe every orchestrator edit into the arcade, which is
exactly the drift the shared engine was introduced to eliminate, and it pointed
at a page that no longer exists. See
[docs/pages/arcade-strategy-pipeline.md](../../docs/pages/arcade-strategy-pipeline.md).

---

## Output dataclasses

| Agent | Output type | Key fields |
|---|---|---|
| N25 | `PaceOutput` | `lap_time_pred`, `ci_p10`, `ci_p90`, `delta_vs_prev`, `reasoning` |
| N26 | `TireOutput` | `laps_to_cliff_p10/p50/p90`, `warning_level`, `deg_rate`, `reasoning` |
| N27 | `RaceSituationOutput` | `overtake_prob` (nullable — see below), `sc_prob_3lap`, `threat_level`, `reasoning` |
| N28 | `PitStrategyOutput` | `action`, `compound_recommendation`, `stop_duration_p05/p50/p95`, `undercut_prob`, `reasoning` |
| N29 | `RadioOutput` | `radio_events`, `rcm_events`, `alerts`, `reasoning`, `corrections` |
| N30 | `RegulationContext` | `answer`, `articles`, `chunks`, `.reasoning` (alias for answer) |
| N31 | `StrategyRecommendation` | `action`, `reasoning`, `confidence`, `scenario_scores`, `regulation_context` |

**N27's `overtake_prob` is `float | None`.** N11 dropped every training pair more than 2.5 s
apart before labelling, so the model has no labelled example beyond that gap — 43.1% of real
position-adjacent pairs in 2025. There it returns `None` rather than an extrapolation. `None`
is not `0.0`: zero is the value the regulation asserts under a Safety Car (Art. 55.8), and
consumers have to tell the two apart.

---

## Architecture

```
RaceState (Pydantic)
      │
      ├─ Layer 1 always-on ──────────────────────────────────────────────┐
      │   N25 PaceAgent (XGBoost)                                        │
      │   N26 TireAgent (TireDegTCN + MC Dropout)                        │
      │   N27 RaceSituationAgent (LightGBM overtake + SC)                │
      │   N29 RadioAgent (RoBERTa + SetFit + BERT NER + RCM parser)      │
      │                                                                   │
      ├─ Layer 1 MoE routing ──────────────────────────────────────────► │
      │   tire_warning == PIT_SOON  → activate N28                        │
      │   radio PROBLEM/WARNING     → activate N28                        │
      │   sc_prob > 0.30            → activate N30                        │
      │   N28 active                → activate N30                        │
      │                                                                   │
      ├─ Layer 1 conditional ─────────────────────────────────────────── │
      │   N28 PitStrategyAgent (N15 quantiles + N16 undercut)             │
      │   N30 RAGAgent (Qdrant + BGE-M3 + LangGraph)                     │
      │                                                                   │
      ├─ Layer 2 Monte Carlo (N_SIM=500, window=5 laps) ──────────────── │
      │   STAY_OUT / PIT_NOW / UNDERCUT / OVERCUT                         │
      │   score = α·E[S] + (1−α)·P10[S]                                  │
      │   S = projected track position (position_projection.py)           │
      │   no rival gaps → legacy seconds scoring, unchanged               │
      │                                                                   │
      └─ Layer 3 LLM synthesis ───────────────────────────────────────── │
          with_structured_output(StrategyRecommendation)                  │
          action / reasoning / confidence                                 ▼
                                                                StrategyRecommendation
```

---

## RSM adapter pattern

Every agent exposes two entry points: one that expects populated module globals
from a FastF1 session, and an RSM adapter that needs no session because it
builds `SESSION_META` from the laps frame and calls the same core logic.

**The two are not uniform, and assuming they are will break your call.** The
table above carries the real signatures, taken from `inspect.signature`. Three
of them differ from what a reader would guess:

| entry point | the surprise |
|---|---|
| `run_pace_agent_from_state(lap_state)` | takes no `laps_df`, unlike every other adapter |
| `run_tire_agent(stint_state)` | its parameter is a stint state, not a lap state |
| `run_strategy_orchestrator_from_state(race_state, laps_df, lap_state=None)` | a third argument, and the projection needs it |

That last one matters most: without `lap_state` the orchestrator has no rival
gaps, so the Monte Carlo falls back to the legacy seconds path instead of
scoring in projected track position.

Regenerate this table with `inspect.signature` rather than by hand. It has been
wrong before.

---

## Testing

**Level 1: NLP/model tools, no LLM:**

```python
from src.agents.radio_agent import process_radio_tool
result = process_radio_tool.invoke({"driver": "NOR", "lap": 18, "text": "Box this lap."})
print(result)
```

**Level 2: Single agent, no LLM:**

```python
from src.agents.race_situation_agent import process_rcm_tool
result = process_rcm_tool.invoke({
    "message": "SAFETY CAR DEPLOYED", "flag": "", "category": "SafetyCar", "lap": 20
})
print(result)
```

**Level 3: Full orchestrator smoke test (requires LM Studio running):**

```python
from src.agents.strategy_orchestrator import RaceState, run_strategy_orchestrator_from_state
import pandas as pd

laps_df = pd.read_parquet("data/processed/laps_featured_2025.parquet")
race_state = RaceState(
    driver="NOR", lap=18, total_laps=57, position=3,
    compound="MEDIUM", tyre_life=20, gap_ahead_s=1.2, pace_delta_s=-0.3,
    air_temp=32.0, track_temp=48.0,
)

# Pass the lap_state too. Without it the orchestrator never sees the rival
# gaps, so the Monte Carlo scores on the legacy seconds path instead of in
# projected track position, and laps_df stays a whole season rather than one
# race. Build it with RaceStateManager; hand-rolling one is how the second,
# buggy implementation of this contract got written.
rec = run_strategy_orchestrator_from_state(race_state, laps_df, lap_state)
print(rec.action, rec.confidence, rec.reasoning)
```

---

## Package init

`__init__.py` re-exports all public entry points so callers can do:

```python
from src.agents import RaceState, run_strategy_orchestrator_from_state
```

---

## The experta engine moved out

The original rule-based engine (`base_agent.py`, `strategy_agent.py` and the four
`rules/` modules) now lives at [`legacy/experta_engine/`](../../legacy/experta_engine/).
Nothing in the live pipeline imported it, and a near-identical copy was already
archived under `legacy/app_streamlit_v1/`. It is kept because it is part of the
thesis record, not because anything calls it.

---|---|
| `base_agent.py` | Experta `Fact` subclasses and `F1StrategyEngine` (CLIPS-style, legacy) |
| `strategy_agent.py` | `F1CompleteStrategyEngine`, original rule-based engine, superseded by N31 |
| `rules/degradation_rules.py` | Tyre degradation rules for legacy engine |
| `rules/laptime_rules.py` | Lap time rules for legacy engine |
| `rules/gap_rules.py` | Gap/position rules for legacy engine |
| `rules/nlp_rules.py` | NLP intent rules for legacy engine |
| `rules/__init__.py` | Legacy rules package init |

The legacy engine is not used. Do not import from it in new code.

---

## LLM configuration (production)

| Layer | Model |
|---|---|
| Sub-agents N25–N29 | `gpt-4.1-mini` |
| Orchestrator N31 | `gpt-5.4-mini` (`OrchestratorConfig.model_name`) |

Notebooks default to `local-model` (LM Studio). Switch to the OpenAI model IDs above when deploying via FastAPI.
