# Agents API Reference

## Module location

All agents live in `src/agents/`. Each file is extracted from its corresponding notebook (N25–N31).

## Entry points

Every agent has two entry points:

| Agent | FastF1 Entry | RSM Adapter (no FastF1 session) |
|---|---|---|
| N25 Pace | `run_pace_agent(**kwargs)` | `run_pace_agent_from_state(lap_state)` |
| N26 Tire | `run_tire_agent(stint_state)` | `run_tire_agent_from_state(lap_state, laps_df)` |
| N27 Situation | `run_race_situation_agent(lap_state)` | `run_race_situation_agent_from_state(lap_state, laps_df)` |
| N28 Pit | `run_pit_strategy_agent(lap_state)` | `run_pit_strategy_agent_from_state(lap_state, laps_df)` |
| N29 Radio | `run_radio_agent(lap_state, persist)` | `run_radio_agent_from_state(lap_state, laps_df, persist)` |
| N30 RAG | `run_rag_agent(question)` | `run_rag_agent_from_state(lap_state, laps_df)` |
| N31 Orchestrator | `run_strategy_orchestrator(race_state, lap_state)` | `run_strategy_orchestrator_from_state(race_state, laps_df, lap_state)` |

**N25 is the odd one out: it takes no `laps_df`.** Everything it needs must already be in the
`lap_state`, which is why the stint fuel baseline is carried there rather than derived from a
frame. The adapters are not uniform, so check the signature before assuming.

Each agent also exposes a `get_*_react_agent()` factory that returns a compiled LangGraph `CompiledGraph` for direct LangGraph usage.

## Output dataclasses

### PaceOutput (N25)

| Field | Type | Description |
|---|---|---|
| `lap_time_pred` | float | Predicted lap time in seconds |
| `delta_vs_prev` | float | Delta vs previous lap (negative = faster) |
| `delta_vs_median` | float | Delta vs session median |
| `ci_p10` | float | 10th percentile bootstrap CI |
| `ci_p90` | float | 90th percentile bootstrap CI |
| `reasoning` | str | LLM-generated reasoning text |

### TireOutput (N26)

| Field | Type | Description |
|---|---|---|
| `compound` | str | Current compound name (SOFT, MEDIUM, HARD) |
| `current_tyre_life` | int | Current tire age in laps |
| `deg_rate` | float | Degradation rate (seconds lost per lap) |
| `laps_to_cliff_p10` | float | 10th percentile laps until cliff |
| `laps_to_cliff_p50` | float | 50th percentile laps until cliff |
| `laps_to_cliff_p90` | float | 90th percentile laps until cliff |
| `warning_level` | str | OK, MONITOR, or PIT_SOON (derived from `laps_to_cliff_p10` against circuit-cluster-aware thresholds; there is no CRITICAL value) |
| `reasoning` | str | LLM-generated reasoning text |

### RaceSituationOutput (N27)

| Field | Type | Description |
|---|---|---|
| `overtake_prob` | float | Probability of being overtaken (0–1) |
| `sc_prob_3lap` | float | Safety car probability within 3 laps (0–1) |
| `sc_currently_active` | bool | Any neutralisation (full Safety Car **or** Virtual Safety Car) is deployed **right now**. Not a prediction: it is read from the lap's RCM events, because N14 was trained to forecast a future SC and cannot recognise one already out. When true it forces the regulatory facts (`sc_prob_3lap = 1.0`, `overtake_prob = 0` per Art. 55.8/56.6, `drs_window = 0` per Art. 22.1(c)) and activates N28. It does **not** force the action: whether to pit under a neutralisation is race state, not a rule. See [Multi-agent system](#/multi-agent). |
| `vsc_active` | bool | The active neutralisation is specifically a **Virtual** Safety Car (Art. 56), only meaningful when `sc_currently_active` is true. Split out (#471) because a VSC and a full SC differ in pit-time saving, and the Monte Carlo / N28 prompt need to tell them apart — the single `sc_currently_active` flag could not. `sc_active` (a derived property, not a stored field) is true only under a full SC: `sc_currently_active and not vsc_active`. |
| `threat_level` | str | LOW, MEDIUM, HIGH |
| `gap_ahead_s` | float | Gap to car ahead in seconds |
| `pace_delta_s` | float | Pace difference vs car ahead |
| `reasoning` | str | LLM-generated reasoning text |

### PitStrategyOutput (N28)

| Field | Type | Description |
|---|---|---|
| `action` | str | STAY_OUT, PIT_NOW, UNDERCUT, OVERCUT, or REACTIVE_SC (box for an elevated-but-not-yet-confirmed SC probability; when an SC is already confirmed deployed, N28 prefers PIT_NOW directly) |
| `recommended_lap` | int or None | Suggested pit lap |
| `compound_recommendation` | str | Suggested next compound |
| `stop_duration_p05` | float | 5th percentile stop duration (s) |
| `stop_duration_p50` | float | Median stop duration (s) |
| `stop_duration_p95` | float | 95th percentile stop duration (s) |
| `undercut_prob` | float or None | Undercut success probability (0–1) |
| `undercut_target` | str or None | Target driver for undercut |
| `sc_reactive` | bool | Whether recommendation is SC-reactive |
| `reasoning` | str | LLM-generated reasoning text |

### RadioOutput (N29)

| Field | Type | Description |
|---|---|---|
| `radio_events` | list | Processed radio messages with sentiment, intent, NER |
| `rcm_events` | list | Processed Race Control Messages |
| `alerts` | list | Deterministic alert flags from NLP pipeline |
| `reasoning` | str | LLM-generated reasoning text |
| `corrections` | list | Driver-reported corrections (damage, handling issues) |

### RegulationContext (N30)

| Field | Type | Description |
|---|---|---|
| `answer` | str | Synthesized answer from regulation passages |
| `articles` | list[str] | Referenced FIA article numbers |
| `chunks` | list | Raw retrieved text chunks |
| `reasoning` | str | Alias for `answer` |

### StrategyRecommendation (N31)

The v2 schema (frozen at 14 fields) surrounds the primary `action` with execution detail, driver-side instructions and a contingency list — see [Multi-agent system](#/multi-agent) for the full rationale.

| Field | Type | Description |
|---|---|---|
| `action` | str | STAY_OUT, PIT_NOW, UNDERCUT, OVERCUT, ALERT — the primary decision |
| `reasoning` | str | Multi-sentence LLM synthesis of all sub-agent inputs, MC scores and regulation constraints |
| `confidence` | float | 0–1 LLM self-assessed certainty; treat as qualitative, not calibrated |
| `pit_lap_target` | int or None | Absolute lap of the planned stop. Populated for PIT_NOW/UNDERCUT/OVERCUT, optionally for a forward-looking STAY_OUT plan |
| `compound_next` | str or None | Compound (SOFT/MEDIUM/HARD) chosen for the next stint; None for STAY_OUT |
| `undercut_target` | str or None | Rival code targeted by an UNDERCUT/OVERCUT (e.g. "SAI") |
| `pace_mode` | str | PUSH, NEUTRAL, MANAGE, LIFT_AND_COAST — driving instruction for the next laps (default NEUTRAL) |
| `target_lap_time_s` | float or None | Target lap time, grounded in N25's CI bounds so the LLM cannot invent a value far outside the model's prediction. Forced to `None` under an active SC/VSC (Art. 55.7 — see [Multi-agent system](#/multi-agent)) |
| `risk_posture` | str | AGGRESSIVE, BALANCED, DEFENSIVE — the championship stance the LLM reasons under (default BALANCED) |
| `contingencies` | list[Contingency] | Conditional branches for upcoming laps, capped at four. Each has `trigger` (plain-language event), `switch_to` (replacement action), `priority` (HIGH/MEDIUM/LOW), `rationale` (short justification) |
| `key_risks` | list[str] | Up to five short bullets flagging risks the LLM wants to surface outside the narrative |
| `expected_stint_end` | int or None | Lap the current stint is planned to end. Clamped against a physical anchor — `pit_lap_target` plus the shorter of the N26 cliff P50 and the next compound's Pirelli stint capacity, bounded by total race laps — accepting the LLM's value only within ±3 laps of that anchor (#433); falls through unclamped when no anchor (missing `pit_lap_target`/`compound_next`/cliff) is available |
| `scenario_scores` | dict | Full MC output per candidate — `{"STAY_OUT": {"E", "P10", "P90", "score"}, ...}`. Attached in code after the LLM call, not filled by the LLM. On the projection path each candidate also carries `eligible` (bool) and `target` (str or null), and an **ineligible candidate has `score: null`** — an undercut with no reachable rival, or an overcut with nobody in the pit lane, is reported as not offered rather than given a number it did not earn. Consumers must skip nulls; coercing one to `0.0` draws a real-looking score for a strategy that was never on the table |
| `regulation_context` | str | N30 RAG answer when activated, empty string otherwise. Attached in code after the LLM call |

## `RaceState` input (N31)

The orchestrator accepts a `RaceState` Pydantic model:

```python
class RaceState(BaseModel):
    driver: str           # Three-letter driver code
    lap: int              # Current lap number
    total_laps: int       # Total race laps
    position: int         # Current race position
    compound: str         # Current tire compound
    tyre_life: int        # Current tire age (laps)
    gap_ahead_s: float    # Gap to car ahead (seconds)
    pace_delta_s: float   # Pace delta vs car ahead
    air_temp: float       # Air temperature (C)
    track_temp: float     # Track temperature (C)
    rainfall: bool = False
    radio_msgs: list = []   # RadioMessage dicts for current lap window
    rcm_events: list = []   # RCMEvent dicts for current lap window
    risk_tolerance: float = 0.5  # 0=conservative, 1=aggressive
```

## Model artifacts

| Agent | Model directory | Files |
|---|---|---|
| N25 | `data/models/lap_time/` | XGBoost model |
| N26 | `data/models/tire_degradation/` | TireDegTCN `.pt` files + calibration JSON |
| N27 | `data/models/overtake_probability/` | LightGBM + calibrator + config |
| N27 | `data/models/safety_car_probability/` | LightGBM + calibrator + feature list |
| N28 | `data/models/pit_prediction/` | HistGBT P05/P50/P95 + undercut LightGBM |
| N29 | `data/models/nlp/` | pipeline_config_v1.json + .pt state dicts |
| N30 | `data/rag/` | Qdrant index (built by `scripts/build_rag_index.py`) |

## LLM tool-level input validation

Every LangChain tool the LLM can call takes free-text arguments (`driver`, `lap_number`, `gp_name`, `year`, ...) with no server-side roster or range check other than what the tool itself enforces. A hardening pass (#476) closed the gap where a hallucinated or long-retired driver code, or a lap outside the loaded session, produced a confident but meaningless prediction instead of a visible failure:

| Agent | Tools guarded | Refuses when |
|---|---|---|
| N25 Pace | `predict_pace_tool` | `driver_number`/`gp_name`/`year` do not resolve to a driver on track for that GP/year, or `lap_number` is outside `[1, total_laps]` |
| N26 Tire | `predict_tire_deg_tool`, `estimate_laps_to_cliff_tool` | `driver` is not on track at the currently loaded lap, or the loaded `current_lap` is outside `[1, total_laps]` |
| N27 Situation | `predict_overtake_tool`, `predict_sc_tool` | `lap_number` is out of range, or (for `predict_overtake_tool`) either driver is unknown for that lap |
| N28 Pit | `predict_pit_duration_tool`, `score_undercut_tool` | the named driver (`driver`/`driver_y`) is not present in the live roster for the current lap |

The refusal is a plain string return (e.g. `"error: 'HAM' is not on track at lap 12; valid: [...]"` or `"... REFUSED — {driver} is not on track ..."`), not an exception — the LLM sees a normal-looking tool result it can react to, rather than a traceback. `predict_pit_duration_tool` additionally cross-checks the LLM-supplied `under_sc` flag against the RCM-confirmed `sc_currently_active` ground truth when a real orchestrator run has set it, and trusts the confirmed value over the guess (logging a warning) rather than the other way round.

"On track" is a **presence** check (the driver has a row in the live roster for this lap), the same convention `RaceStateManager` uses for `rivals` — see [Race replay engine → who counts as a rival](#/simulation). An age/lap-count cutoff cannot substitute for it: a finisher can go 20 laps without a row, and a retirement can surface as few as 9 laps in, so the ranges overlap.

The MCP-facing tools one layer up (`src/telemetry/backend/mcp_tools.py`, consumed by chat) apply an equivalent guard on `gp`/`driver`/`lap`/`year` before they even reach these agent-level tools — see [Backend API reference → Tool risk tiers and the chat allowlist](#/backend-api).

## Testing examples

**Tool-level (no LLM needed):**

```python
from src.agents.radio_agent import process_radio_tool
result = process_radio_tool.invoke({
    "driver": "NOR", "lap": 18, "text": "Box this lap."
})
```

**Agent-level (no LLM needed):**

The `*_from_state` entry points take **only** a `lap_state`, and it must be the real
nested dict (`driver` / `rivals` / `weather` / `session_meta`), not a flat one. Build it
with `RaceStateManager` rather than by hand: it is the component that owns the contract,
and hand-rolling one is how the second, buggy implementation of this got written.

```python
from pathlib import Path
from itertools import islice
from src.simulation.replay_engine import RaceReplayEngine
from src.agents.pace_agent import run_pace_agent_from_state

replay = RaceReplayEngine(Path("data/raw/2025/Lusail"), driver_code="PIA", team="McLaren")
lap_state = next(islice(replay.replay(), 19, 20))   # lap 20
output = run_pace_agent_from_state(lap_state)

print(output.lap_time_pred, output.ci_p10, output.ci_p90)
```

**Take the `lap_state` from `replay()`, not from `rsm.get_lap_state(lap)` directly.**
`get_lap_state`'s `weather_df` argument is optional, and `replay()` is what passes it. Call
the RSM bare and the weather dict comes back with only `track_status`, so consumers fall
through to their hardcoded defaults (the arcade's are 25.0 C air / 35.0 C track) and never
say so. At Lusail the real values are 23.7 C and 29.8 C, and track temperature is a live
input to tire degradation.

See [Race replay engine](#/simulation) for the full `lap_state` schema.

**Full orchestrator (requires LM Studio or OpenAI):**

```python
from pathlib import Path
from itertools import islice
import pandas as pd
from src.simulation.replay_engine import RaceReplayEngine
from src.agents.strategy_orchestrator import RaceState
from src.strategy.inference.engine import run_lap

race_dir = Path("data/raw/2025/Lusail")
laps_df = pd.read_parquet(race_dir / "laps.parquet")
replay = RaceReplayEngine(race_dir, driver_code="PIA", team="McLaren")
lap_state = next(islice(replay.replay(), 19, 20))   # lap 20

driver = lap_state["driver"]
ahead = [r for r in lap_state["rivals"] if r["position"] == driver["position"] - 1]
race_state = RaceState(
    driver=driver["driver"], lap=20, total_laps=replay.total_laps,
    position=driver["position"], compound=driver["compound"],
    tyre_life=driver["tyre_life"],
    # rivals ahead report a NEGATIVE interval; RaceState wants a magnitude
    gap_ahead_s=abs(ahead[0]["interval_to_driver_s"]) if ahead else 0.0,
    pace_delta_s=0.0,
    air_temp=lap_state["weather"]["air_temp"],
    track_temp=lap_state["weather"]["track_temp"],
)

rec, agent_outputs, timings = run_lap(race_state, laps_df, lap_state, profile="no-llm")
print(rec.action, rec.confidence, rec.reasoning)
```

Prefer `run_lap` over calling `run_strategy_orchestrator_from_state` directly: it is the
same pipeline (parity-tested), and it also hands back the per-agent outputs and stage
timings. Pass `profile="no-llm"` to run the deterministic path with no provider at all.

**Pass the `lap_state`, and make sure it carries `session_meta`.** The `laps_df` is scoped
to the Grand Prix named there. Without it, the engine falls back to the whole frame and
logs a warning, and the agents then look up rivals by driver and lap number across the
entire season: the Zandvoort grid can end up deciding a race at Lusail. The fallback is
loud on purpose, but the fix is to supply the `session_meta`, not to ignore the warning.
