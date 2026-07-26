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
    N27 -->|sc_currently_active, overrides every threshold| N28
    N27 -->|sc_currently_active, overrides every threshold| N30

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
- Monte Carlo then draws 500 samples over four candidates (STAY_OUT, PIT_NOW, UNDERCUT, OVERCUT), scoring `score = α·E + (1−α)·P10`, and the LLM synthesises the final `StrategyRecommendation`. Since the projection redesign the score is measured in **projected track position**, not in seconds — see [What the Monte Carlo actually scores](#/multi-agent) below.

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
4. **Arcade runs the strategy pipeline in-process.** `src/arcade/strategy_pipeline.py` delegates to the shared engine (`src/strategy/inference/engine.py::run_lap`), so the arcade does not depend on the FastAPI backend at runtime and does not carry its own copy of the orchestrator. It used to; that copy drifted and crashed (#166), which is why the engine exists.

See [Arcade strategy pipeline](#/arcade-strategy-pipeline) for the shared engine and its profiles, and [Arcade dashboard](#/arcade-dashboard) for the Qt-side architecture.

## Agent details

### N25 — Pace Agent (`pace_agent.py`)

Wraps the N06 XGBoost delta-lap-time model. Returns predicted lap time, delta signals against previous lap and session median, and bootstrap confidence intervals (N=200 draws with 2% Gaussian noise on continuous features).

- **Model**: XGBoost trained on 2023–2025 lap data
- **Output**: `PaceOutput` (lap_time_pred, delta_vs_prev, delta_vs_median, ci_p10, ci_p90)

### N26 — Tire Agent (`tire_agent.py`)

Wraps per-compound TireDegTCN models (N09/N10) with MC Dropout inference. Answers: how many laps remain before the degradation cliff?

- **Model**: Causal TCN per compound + Platt calibration
- **Output**: `TireOutput` (laps_to_cliff_p10/p50/p90, warning_level, deg_rate)
- **Warning levels**: OK, MONITOR, PIT_SOON (derived from `laps_to_cliff_p10` against circuit-cluster-aware thresholds; there is no CRITICAL level)

### N27 — Race Situation Agent (`race_situation_agent.py`)

Combines N12 (overtake probability via LightGBM) and N14 (safety car probability via LightGBM) into a single threat assessment per lap.

- **Models**: LightGBM overtake (AUC-PR 0.5491) + LightGBM SC (AUC-PR 0.0723)
- **Output**: `RaceSituationOutput` (overtake_prob, sc_prob_3lap, threat_level, **sc_currently_active**, **vsc_active**)

#### RCM Safety Car override

The N14 LightGBM was trained to predict a *future* SC, not to recognise one already deployed. To close that gap, N27 inspects the lap's `rcm_events` (forwarded by the orchestrator from `RadioPipelineRunner`) and, when any event matches `SAFETY_CAR_DEPLOYED` or `VIRTUAL_SAFETY_CAR_DEPLOYED`, forces `sc_prob_3lap = 1.0`, sets `sc_currently_active = True`, and elevates `threat_level` to `HIGH`. Release events (`SAFETY_CAR_ENDING`, `SAFETY_CAR_IN_PIT_LANE`, `VIRTUAL_SAFETY_CAR_ENDING`) take priority in the same window so the override clears as soon as the neutralisation ends. The override is logged in the `reasoning` field with an `[RCM OVERRIDE: ...]` prefix so the audit trail survives the chat / arcade summary path.

`sc_currently_active` is deliberately a single back-compat flag: true under **either** a full Safety Car (Art. 55) or a Virtual Safety Car (Art. 56). A second field, `vsc_active`, records whether the specific neutralisation is a VSC — a full SC and a VSC differ in the pit-time saving they offer, so the Monte Carlo and the N28 prompt need to tell them apart (#471), which the single flag could not. `sc_active` (a derived property, not stored) is true only for a full SC: `sc_currently_active and not vsc_active`.

### N28 — Pit Strategy Agent (`pit_strategy_agent.py`)

Wraps N15 (physical pit stop duration P05/P50/P95 via HistGBT) and N16 (undercut success probability via LightGBM). Recommends when to pit, what compound to fit, and whether to undercut.

- **Models**: HistGBT quantile pit duration + LightGBM undercut
- **Output**: `PitStrategyOutput` (action, compound_recommendation, stop_duration_p05/p50/p95, undercut_prob, sc_reactive)
- **Activation**: conditional — runs when tire_warning is PIT_SOON, radio flags PROBLEM/WARNING, **or N27 reports `sc_currently_active = True`** (the RCM-override path)

#### Honoring an active Safety Car

**A rail encodes what the regulation makes certain. It never encodes a strategy opinion.** Facts are forced; the stop/stay call stays with the model, which is the only layer that sees the state the decision depends on.

When `sc_currently_active = True`, N28's prompt swaps the "SC probability" line for an explicit `SC STATUS: SAFETY CAR DEPLOYED RIGHT NOW` banner and waives the `MINIMUM STINT LENGTH` constraint: a stop under an SC is far cheaper, because the field is delta-limited and queued, so your *relative* loss shrinks. That makes pitting cheaper. It does not make it correct, and nothing forces it.

What is forced are the rules:

| Fact | Regulation | Value | Where |
|---|---|---|---|
| No overtaking on track | **Art. 55.8** (SC) / **56.6** (VSC) | `overtake_prob = 0` | N27 |
| An SC is deployed | Art. 55.4 | `sc_prob_3lap = 1.0` | N27 |
| DRS unavailable | **Art. 22.1(c)** | `drs_window = 0` | N27, feature build |
| No green-flag lap-time target | **Art. 55.7** | `target_lap_time_s = None` | final assembly |

`overtake_prob = 0` is not an approximation. N12 models a *racing* overtake; of the eight exceptions in Art. 55.8, only "a car slows with an obvious problem" yields a real position gain, and N12 has no feature for it. Every input it does use is regulation-corrupted: DRS is off, the gap compresses toward ten car lengths (55.7/55.10), and `pace_delta` collapses to the FIA ECU delta. Under an SC the model is not imprecise, it is **inapplicable**.

`target_lap_time_s` is the subtle one. It is grounded in N06's **green-flag** pace, and Art. 55.7 requires drivers to stay **above** the FIA ECU minimum time: shipping it instructs the driver to earn a penalty. We cannot source the real delta, so the field has no valid value. `None` is forced by **absence of a source** — which is the test that separates a fact from an opinion.

##### Why the old rail was removed

This page used to document a rail that flipped any `STAY_OUT` to `PIT_NOW` under an SC, "replicating the McLaren Catar 2025 V7 fix". That is one race generalised into a universal law, and Art. 55.17 makes it provably wrong: if the SC is still out at the start of the last lap, the race **finishes behind it with no overtaking**, so track position surrendered to a late stop is unrecoverable by regulation. The pipeline could emit `action=PIT_NOW` carrying the reason *"too late to pit"*, because the rail overrode a correct guard-rail that already knew this.

It also silenced the computation built to weigh it: with an SC deployed `sc_prob_3lap` is 1.0, so **every** Monte Carlo draw already receives the full `SC_PIT_BONUS`. A `STAY_OUT` argmax under those conditions *is* the model saying the cheap stop was outweighed.

Staying out under an SC is right whenever you have already stopped, you lead a pack that must stop anyway, you would rejoin into traffic, or the race is ending. Boxing is right when you must stop anyway, the tyres are near the cliff, or the two-compound rule is unsatisfied and time is short. Not one of those is a rule. All of them are race state the model is given, and a rail sees none of it.

See `tests/mc/test_sc_regulatory_rails.py`.

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
2. **Monte Carlo Simulation**: draws 500 samples from sub-agent probability distributions and evaluates four strategy candidates (STAY_OUT, PIT_NOW, UNDERCUT, OVERCUT). Score = alpha * E[S] + (1-alpha) * P10[S], where S is a **projected track position** (see below).
3. **LLM Synthesis**: structured-output LLM aggregates all reasoning strings and MC scores into a `StrategyRecommendation`.

- **Output**: `StrategyRecommendation` (action, reasoning, confidence, scenario_scores, contingencies)
- **Action values**: STAY_OUT, PIT_NOW, UNDERCUT, OVERCUT, ALERT

## What the Monte Carlo actually scores

The layer used to score in generic seconds divided by a flat 1.5 s/position, over a sampled state that contained **no cars at all**. That constant cannot be right in both regimes: measured across 71 races, the median gap between consecutive cars is **2.23 s while racing and 1.48 s under a Safety Car**, so a single figure was a bunched-field number applied to green-flag racing. And losing 20 s costs zero positions with a 25 s cushion behind but three positions with cars at +2 / +8 / +15 s — a difference only a model that knows *which cars are where* can see.

Scoring now runs on a per-rival gap projection (`src/agents/position_projection.py`). Each candidate moves every gap by the difference between what a rival loses and what we lose; a gap crossing zero is a car changing sides, so counting the cars projected ahead gives the position directly. Three behaviours that used to need special cases now fall out of that arithmetic:

- **Rejoining into traffic** is automatic — every rival within our pit loss behind us is a place lost, counted by name.
- **The mandatory-stop cancellation** (Art. 30.5(m)) happens only when the rival genuinely stops too. Where the old model argued in a comment that the pit-lane traversal cancels, the projection charges it per car and lets it cancel when it actually does.
- **The Art. 55.17 endgame** — a race finishing behind the Safety Car — emerges from the measured racing-lap count dropping to zero: fresh tyres have nothing left to pay themselves back over, so staying out wins on the numbers. This is the case a deleted guard-rail used to force, and it now needs no rail.

A **terminal liability** replaces the flat Safety Car bonus with option value: a still-owed stop costs the cars it will release behind us, discounted by the measured probability that a later neutralisation covers it cheaply.

Candidates carry **explicit eligibility**. An undercut with no live rival inside the measured band, or an overcut with nobody in the pit lane, is returned as `eligible: false` with a `score` of `null` — never a number it did not earn. Every constant the layer reads is measured and committed in `data/mc_measured_v1.json`, regenerated by `scripts/measure_mc_tables.py`.

**Where the overcut works, and why.** Inside one window an overcut takes the same stop and pays the same pit lane as PIT_NOW, so it forfeits exactly one lap of fresh rubber. What it buys instead is a lap of **clean air**, and whether that is a good trade is a property of the circuit rather than of the strategy.

Both sides are measured. A fresh set is worth 0.25 s/lap; clean air is measured per circuit over 479 cases where a car sat within two seconds of, and directly behind, a driver who then pitted — **+0.77 s/lap at Suzuka, +0.65 at Monaco, +0.63 at Silverstone**, down to **−0.02 at Monza and −0.29 at Spielberg**, where losing the car ahead costs a slipstream worth more than the clear track. The ordering is high-downforce circuits first and slipstream circuits last, and nothing in the measurement knows what downforce is.

Clean air is one of two reasons to hold a car out. The other is **option value**: one more lap before the stop is one more lap of exposure to a neutralisation that would make that stop cheap, worth the circuit's measured onset hazard times what a neutralised stop saves. **Melbourne** separates the two cleanly — its clean-air gain is +0.008 s, effectively nothing, but Albert Park throws more neutralisations per lap than any circuit in the sample, so an overcut pays there on Safety Car odds alone.

Together the two terms decide it against the 0.25 s/lap the delay costs:

| circuit | clean air | waiting | total |
|---|---|---|---|
| Suzuka | +0.771 | +0.158 | **+0.93** |
| Lusail | +0.646 | +0.214 | +0.86 |
| Melbourne | +0.008 | +0.597 | +0.61 |
| Monza | −0.015 | +0.143 | +0.13 |
| Spielberg | −0.285 | +0.161 | −0.12 |

The invariant tests assert both directions, comparing the overcut against the same stop taken now: it must out-score a plain stop at Suzuka and at Melbourne, and must lose to one at Monza.

Dirty air is priced at the moment the car ahead boxes and not continuously, so running a whole window stuck in traffic is still under-penalised. That one is named in the module docstring rather than hidden here.
- **Pace modes**: PUSH, NEUTRAL, MANAGE, LIFT_AND_COAST
- **Risk levels**: AGGRESSIVE, BALANCED, DEFENSIVE

## RSM adapter pattern

Every agent exposes two entry points: one that expects populated module globals from a FastF1 session, and an RSM adapter that works straight from a parquet frame because it builds `SESSION_META` itself and then calls the same core logic.

**They are not uniform.** The shapes below come from `inspect.signature`, and three of them differ from what the pattern would lead you to guess:

```python
run_pace_agent_from_state(lap_state)                                  # no laps_df, unlike every other adapter
run_tire_agent(stint_state)                                           # a stint state, not a lap state
run_tire_agent_from_state(lap_state, laps_df)
run_race_situation_agent_from_state(lap_state, laps_df)
run_pit_strategy_agent_from_state(lap_state, laps_df)
run_radio_agent_from_state(lap_state, laps_df, persist=False)
run_rag_agent_from_state(lap_state, laps_df=None)
run_strategy_orchestrator_from_state(race_state, laps_df, lap_state=None)
```

That last argument is the one worth remembering: without `lap_state` the orchestrator never sees the rival gaps, so the Monte Carlo falls back to the legacy seconds path instead of scoring in projected track position. See [agents-api.md](#/agents-api) for the full per-agent reference.

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
  JSON response --> web app Strategy tab
```

## References

- Heilmeier et al. (2020) ApplSci 10/4229 — MC motorsport simulation
- Wang et al. (2024) arXiv:2406.04692 — MoA reasoning aggregation
- Liu et al. (2024) arXiv:2402.02392 — DeLLMa decision under uncertainty with LLM
