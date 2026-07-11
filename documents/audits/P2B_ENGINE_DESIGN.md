# P2b ENGINE DESIGN — `src/strategy/inference/engine.py` (#169, Phases 1.1 + 1.2)

> **Author**: Fable 5 (design-before gate) · **Date**: 2026-07-10 · **Mode**: design only, NO code.
> **Basis**: `documents/audits/AUDIT_P2B_CORE_COMPUTE.md` §7 (shared fast path), §8 (phased plan), §9 (risk register).
> **Scope of #169**: Phase 1.1 (engine skeleton + `run_lap` + `rich` profile that reproduces today's
> orchestrator behavior AND returns `agent_outputs`) + Phase 1.2 (true `no-llm` profile fixing the #166 crash).
> **Out of scope, stated up front**: F6 4-way parallelism (Phase 2.2), the F2 one-line PMV hotfix (P4 duplicate),
> F7 GP-filtered frame (Phase 1.3), F4 RAG cache (Phase 1.4), F5 silent-radio guard in rich mode (Phase 1.5),
> the `fast` profile (Phase 2.1/3.1).
> **Untouchability honored**: zero edits to `src/agents/**`, `scripts/run_simulation_cli.py`, `notebooks/**`, `legacy/**`.

---

## Gate decisions (Víctor, 2026-07-10)

The design-before gate was reviewed and cleared with these answers to §7:
- **Q1** — ratified: rich = option (a), engine re-drives the sequence via imported orchestrator layers.
- **Q2** — **`_NullReActRunner` injection** into engine-private agent instances (zero `src/agents/` edits); revisit additive entry points at Phase 2.1.
- **Q3** — per-stage timings now, per-agent in Phase 2.2 (accepted).
- **Q4** — `PROFILES = ("rich", "no-llm")`; `fast` raises (accepted).
- **Q5** — `no_llm.py` hosts the canonical `apply_guard_rails` (accepted).
- **Q6** — **all four no-llm semantic deltas shipped** as part of the #166 fix (real numbers, `sc_currently_active` in routing, true-offline, never-attempt N28/N30).
- **Q7** — private no-llm agent instances (accepted).
- **Q8** — parity test as `data`+`llm` markers with in-file skips (accepted).

Implementation note (deviation from §1): `__init__.py` is kept EMPTY (no re-export of `run_lap`) so the legacy `tire_predictor.py` importer (`src/agents/rules/degradation_rules.py:22`) is not coupled to the engine's model-heavy orchestrator import. Consumers import `from src.strategy.inference.engine import run_lap` directly.

**Delivery split**: Phase 1.1 (rich profile + arcade delegate) lands first (live-smoke validated); Phase 1.2 (no-llm profile + parity/no-llm data-tier tests) lands second and closes #169 + #166.

---

## 0. The one-paragraph shape

One new module family under `src/strategy/inference/` exposes
`run_lap(race_state, laps_df, lap_state, *, profile, return_agent_outputs=True) -> (StrategyRecommendation, agent_outputs, stage_timings)`.
The `rich` profile is the arcade's proven single-pass pattern (`src/arcade/strategy_pipeline.py:42-121`)
promoted to its intended home: it re-drives the orchestrator's exact five-step sequence by **importing** the
same seven layer functions the orchestrator itself calls (never copying their bodies), so `action` and
`scenario_scores` are byte-identical to `run_strategy_orchestrator_from_state` by construction, while the
intermediate sub-agent outputs are returned instead of discarded. The `no-llm` profile builds the deterministic
path with **zero LLM clients constructed**: N25 and N29's numeric stages run through their existing no-LLM
surfaces, N26/N27 run their real models through a deterministic tool-runner injected into engine-private agent
instances (via the existing `_react_agent` lazy-cache seam), the 3-tuple from `_run_conditional_agents` is
unpacked correctly (killing #166's `ValueError`), and the guard-rail policy already proven in the backend's
`apply_guard_rails` is re-hosted as the canonical copy. Arcade's `strategy_pipeline.py` becomes a thin
delegate; the CLI duplicate (P4) and the backend's `_run_no_llm_path` (P1) consume the same function later.

---

## 1. Module layout

```
src/strategy/inference/
├── __init__.py          # re-exports: run_lap, PROFILES ("rich" | "no-llm")
├── engine.py            # public API + rich profile + shared spine (~200 lines)
├── no_llm.py            # no-llm profile internals (~200 lines)
└── tire_predictor.py    # PRE-EXISTING legacy jupytext artifact (N06 era) — NOT touched by #169.
                         # Flag for a separate cleanup/removal issue; unrelated to the engine.
```

Two files, split by concern per CLEAN_CODE (~300-line module rule): `engine.py` owns the profile-agnostic
spine and the LLM-mode path; `no_llm.py` owns everything that exists only to make determinism possible.
No class for the engine itself in #169: `run_lap` is a module-level function matching the audit §7 contract
verbatim (three stateless consumers). Stateful construction (GP-filtered frame, RAG cache, warm agents) is
Phase 1.3/1.4's decision and slots in later without changing the public signature.

### 1.1 `engine.py` — contents

| Symbol | Kind | Purpose |
|---|---|---|
| `run_lap(...)` | function (public) | Contract in §2. Dispatches on `profile`; wraps every stage in a `perf_counter` timer. |
| `_run_rich(...)` | function | The five-step sequence (§3). Returns `(rec, agent_outputs, timings)`. |
| `_build_default_lap_state(race_state, laps_df)` | function | Lifted verbatim from `src/arcade/strategy_pipeline.py:124-167` (itself a mirror of the orchestrator's inline block at `strategy_orchestrator.py:1327-1367`). The engine becomes the single non-orchestrator home; arcade's copy is deleted when it delegates. |
| `_assemble_agent_outputs(...)` | function | Builds the dict in §2.2 — key set identical to arcade's today (`strategy_pipeline.py:108-120`) plus `guardrail_reason`. |
| `_StageTimer` | tiny class | `perf_counter` context helper filling the `stage_timings` dict; ~15 lines. |
| `PROFILES` | constant | `("rich", "no-llm")`. `"fast"` is intentionally rejected with a pointing error (§2.3). |

**Imports from `src/agents/strategy_orchestrator.py`** (the complete list — all but two already precedented by
`src/arcade/strategy_pipeline.py:28-36` and `backend/services/simulation/simulator.py:329-333`):

| Symbol | Line | Precedent |
|---|---|---|
| `RaceState`, `StrategyRecommendation` | 157, 317 | public — everywhere |
| `_run_always_on_agents_from_state` | 1049 | arcade |
| `_decide_agents_to_call` | 475 | arcade, backend, CLI |
| `_run_conditional_agents` (3-tuple, line 1165) | 1085 | arcade, backend, CLI |
| `_run_mc_simulation` | 609 | arcade, backend, CLI |
| `_build_orchestrator_prompt` | 741 | arcade |
| `_get_orchestrator_llm` | 118 | arcade |
| `_assemble_recommendation` | 1172 | arcade |
| `_LLMSynthesis` | 246 | **new** — needed so the no-llm profile reuses `_assemble_recommendation` instead of hand-building the rec (§4.4) |
| `_to_radio_message`, `_to_rcm_event` | 953, 970 | **new** — input coercion for the no-llm radio stage (§4.3); same coercion the orchestrator applies at 1063-1064 |

The anti-F10 rule holds: every layer is the **same code object** the orchestrator executes. The only
engine-owned "copies" are (a) the ~30-line call sequence itself (unavoidable — the sequence IS the module) and
(b) `_build_default_lap_state` (quarantined in exactly one non-orchestrator place, and covered by the parity
test's `lap_state=None` case, §5.2).

### 1.2 `no_llm.py` — contents

| Symbol | Kind | Purpose |
|---|---|---|
| `run_no_llm_lap(...)` | function | The deterministic path (§4). Called by `engine.run_lap` when `profile="no-llm"`. |
| `_NullReActRunner` | class (~30 lines) | Deterministic stand-in for the LangGraph compiled graph: `invoke({'messages': [...]}) -> {'messages': [...]}`. Executes pre-bound tool closures, wraps their string outputs in real `langchain_core.messages.ToolMessage` objects, appends a final `AIMessage("[no-llm — deterministic tool pass]")`. |
| `_get_no_llm_tire_agent()`, `_get_no_llm_situation_agent()` | lazy factories | Engine-private `TireAgent()` / `RaceSituationAgent()` instances (same lazy-singleton pattern as `_get_default_tire_agent`, `tire_agent.py:1191`). Private so the LLM-mode process singletons are never contaminated by the injected runner. |
| `_run_radio_no_llm(...)` | function | N29 stages 1+2 without stage 3 (§4.3). |
| `apply_guard_rails(action, lap, total_laps, compound, tyre_life, cliff_p10) -> (action, reason \| None)` | function | Canonical re-host of `src/telemetry/backend/services/simulation/guard_rails.py:30` (§4.5). |
| `_deterministic_synthesis(best, guardrail_reason)` | function | Builds the `_LLMSynthesis` stand-in for `_assemble_recommendation` (§4.4). |

**Imports from `src/agents/`** (all read-only; classes and module functions):

- `pace_agent`: `run_pace_agent_from_state` (public; XGBoost only, zero LLM turns — audit §2.1 row 1).
- `tire_agent`: `TireAgent` (class), `_compound_name_to_id` (line 599, needed to pre-bind tool args).
- `race_situation_agent`: `RaceSituationAgent` (class).
- `radio_agent`: `run_pipeline` (529), `run_rcm_pipeline` (604), `_build_alerts` (called at 985), `RadioOutput`, `RadioMessage`, `RCMEvent`.
- `strategy_orchestrator`: the routing/MC/assembly symbols listed in §1.1.

---

## 2. The `run_lap` contract

```python
def run_lap(
    race_state: RaceState,               # orchestrator's Pydantic model, unchanged
    laps_df: pd.DataFrame,               # full laps frame (featured parquet or RSM slice)
    lap_state: dict | None,              # RSM-shaped dict; None -> _build_default_lap_state
    *,
    profile: Literal["rich", "no-llm"] = "rich",
    return_agent_outputs: bool = True,   # False -> slot 2 is None (compute is unchanged)
) -> tuple[StrategyRecommendation, dict[str, Any] | None, dict[str, float]]:
```

Return type is **uniform across profiles** — always a real `StrategyRecommendation`, never the CLI's ad-hoc
dict (`_run_no_llm`'s dict shape at `run_simulation_cli.py:1562-1576` is a consumer-side artifact the P4
duplicate stops needing; the backend's `_parse_lap_decision` `isinstance(dict)` branch likewise becomes dead
when P1 migrates).

### 2.1 `StrategyRecommendation` (slot 1)

- `rich`: exactly what `run_strategy_orchestrator_from_state` returns today (LLM synthesis via
  `_get_orchestrator_llm().invoke(prompt)` + `_assemble_recommendation`). Parity contract: byte-identical
  `action` and `scenario_scores` given identical LLM replies (§5).
- `no-llm`: assembled through the **same** `_assemble_recommendation` helper from a deterministic
  `_LLMSynthesis` (§4.4): `action` = guard-railed MC argmax, `reasoning` = `"[no-llm mode ...]"` string
  (same wording family as CLI `run_simulation_cli.py:1558-1560`), `confidence=0.0`, `pace_mode="NEUTRAL"`,
  `risk_posture="BALANCED"` (schema defaults), pit fields backfilled from `pit_out` (always `None` in
  no-llm, so `None`), `scenario_scores` = full nested MC dict, `regulation_context=""`.

### 2.2 `agent_outputs` (slot 2) — frozen key set

Identical to arcade's existing contract (`strategy_pipeline.py:108-120`) so arcade's dashboard formatters
work unchanged the day it delegates, plus one new key:

| Key | Type | rich | no-llm |
|---|---|---|---|
| `pace_out` | `PaceOutput` | from `_run_always_on_agents_from_state` | real XGBoost numbers |
| `tire_out` | `TireOutput` | same | real TCN + MC-dropout numbers via null runner |
| `situation_out` | `RaceSituationOutput` | same | real LightGBM numbers + RCM SC override via null runner |
| `radio_out` | `RadioOutput` | same | real NLP results + deterministic alerts; canned reasoning |
| `pit_out` | `PitStrategyOutput \| None` | N28 when routed | **always `None`** (N28 is LLM-backed; §4.2) |
| `regulation_context` | `str` | N30 answer or `""` | **always `""`** (N30 answer synthesis is LLM-backed) |
| `rag` | `dict \| None` | structured N30 payload (question/answer/articles/chunks) | always `None` |
| `active` | `list[str]` | routing result | routing result (still computed — the UI panel shows it) |
| `guardrail_reason` | `str \| None` | **new** — always `None` | rail explanation or `None` |

### 2.3 `stage_timings` (slot 3)

`dict[str, float]` of seconds via `time.perf_counter`, stable extend-only key set:
`{"always_on", "routing", "conditional", "mc", "synthesis", "total"}`.
In `no-llm`, `"synthesis"` measures the deterministic assembly (≈0) and `"conditional"` the routing-only
no-op. **Granularity note**: per-agent timings (audit Phase 0.2's wish) are NOT possible in #169 without
unbundling `_run_always_on_agents_from_state`, which would copy its 2-thread-pool body (anti-F10). Stage-level
lands now; per-agent granularity arrives naturally in Phase 2.2 when F6 forces the engine to own the pool.
This partially folds Phase 0.2 into #169 — see open question Q3.

**`profile="fast"`** raises `ValueError("profile 'fast' is P2b Phase 2 (audit F3/F11) — use 'rich' or 'no-llm'")`
so consumers can already write the three-valued switch without silent fallthrough.

---

## 3. How `rich` returns `agent_outputs` without touching `src/agents/` — option (a), ratified

**Chosen: (a) re-drive the orchestration in the engine by importing the same layer functions.** The engine's
`_run_rich` is, statement for statement, the sequence of `run_strategy_orchestrator_from_state`
(`strategy_orchestrator.py:1369-1418`):

1. `_run_always_on_agents_from_state(race_state, laps_df, lap_state)` → `(pace, tire, situation, radio)`
   — one imported call; keeps the exact N25∥N27 pool + serial N26/N29 threading of lines 1071-1081.
2. `_decide_agents_to_call(tire.warning_level, situation.sc_prob_3lap, radio.alerts, situation.sc_currently_active)`.
3. `_run_conditional_agents(active, lap_state, tire, situation, race_state, laps_df)` → **3-tuple**
   `(pit_out, regulation_context, rag_dict)` — the engine is born on the post-`bfe5b46` contract, so the
   F2 class of bug cannot recur here.
4. `_run_mc_simulation(pace, tire, situation, pit_out, alpha=race_state.risk_tolerance)` → `best_mc` argmax.
5. `_build_orchestrator_prompt(...)` → `_get_orchestrator_llm().invoke(prompt)` →
   `_assemble_recommendation(synth, pit_out, mc_results, regulation_context)`.

Then it keeps steps 1-3's intermediates and returns them as `agent_outputs` instead of discarding them.

**Why (a) and not (b) "wrap the orchestrator and capture":** wrapping
`run_strategy_orchestrator_from_state` would require runtime monkeypatching of the orchestrator module's
helper references to interpose capturing shims — mutating untouchable module state at runtime, thread-unsafe
for the backend, and invisible to a reader. Rejected in one line: patching a frozen module to spy on it is
strictly worse than calling the same functions yourself.

**Why parity is byte-identical in `action`/`scenario_scores`:** every layer function is imported, so the only
code the engine "owns" is the argument plumbing between steps — and that plumbing is asserted equal by the
parity test at the strongest possible level (the LLM prompt bytes, §5.3). `scenario_scores` additionally has
no LLM in its lineage at all (`_run_mc_simulation` is seeded `default_rng(42)`, `strategy_orchestrator.py:637`,
and frozen by `tests/test_strategy_goldens.py::_GOLDEN_ALPHA_05`), so it is deterministic given equal
sub-agent outputs.

**Arcade subsumption:** `src/arcade/strategy_pipeline.py::run_strategy_pipeline` keeps its public signature
`(race_state, laps_df, lap_state=None) -> (rec, agent_outputs)` (its caller in `src/arcade/strategy.py` does
not change) but its body becomes three lines: call `engine.run_lap(..., profile="rich")`, drop
`stage_timings` (or forward it on the TCP stream later — P3's call), return `(rec, agent_outputs)`. Its
private `_build_default_lap_state` and the seven orchestrator imports are deleted; the
"mirror the change here" comment (`strategy_pipeline.py:19`) dies with them. That closes audit F10.

---

## 4. The `no-llm` profile (#166 fix) — deterministic, zero clients

### 4.1 The sequence

```
1. pace_out      = run_pace_agent_from_state(lap_state)                # public, XGBoost only, no LLM by construction
2. tire_out      = <no-llm TireAgent>.run_from_state(lap_state, laps_df)        # §4.2 — real TCN numbers, null runner
3. situation_out = <no-llm RaceSituationAgent>.run_from_state(sit_lap_state, laps_df)  # §4.2 — real LightGBM + RCM override
4. radio_out     = _run_radio_no_llm(race_state, lap_state)            # §4.3 — NLP stages 1+2, no stage 3
5. active        = _decide_agents_to_call(tire.warning_level, situation.sc_prob_3lap,
                                          radio.alerts, situation.sc_currently_active)
6. pit_out, regulation_context, rag_dict = None, "", None              # §4.2 — N28/N30 never called (LLM-backed)
   # NOTE: the 3-tuple contract is honored structurally: the engine never calls
   # _run_conditional_agents in no-llm, so there is nothing to mis-unpack. #166's
   # crash site (run_simulation_cli.py:1508 unpacking 2 of 3) is unreachable by design.
7. mc            = _run_mc_simulation(pace, tire, situation, None, alpha=race_state.risk_tolerance)
                   # pit_out=None -> the documented conservative prior Triangular(2.2, 2.8, 3.8), ucut 0.5
                   # (strategy_orchestrator.py:670-683) — same degradation the CLI comment promises at :1503-1506
8. best          = argmax(mc, key=score)
9. best, reason  = apply_guard_rails(best, lap, total_laps, compound, tyre_life, tire.laps_to_cliff_p10)
10. rec          = _assemble_recommendation(_deterministic_synthesis(best, reason), None, mc, "")
```

Steps 5-10 mirror the *intent* of CLI `_run_no_llm` (`run_simulation_cli.py:1408-1576`) and the backend's
`_run_no_llm_path` (`simulator.py:308-442`) — with three deliberate semantic corrections, each named in §4.6.

### 4.2 N26/N27 real numbers with zero clients: the `_NullReActRunner` injection

**The problem**: `run_tire_agent_from_state` / `run_race_situation_agent_from_state` funnel into `_run_core`,
which invokes a LangGraph ReAct agent (`tire_agent.py:1157-1163`, `race_situation_agent.py:1144-1147`). There
is no public "numbers only" entry. Today's `--no-llm` therefore swaps in **hardcoded stubs** when the backend
is down (CLI `:1451-1465` — `laps_to_cliff 20/25/30`, `deg_rate 0.05`), meaning the MC currently scores fake
tire numbers in no-llm mode; and when LM Studio happens to be up, `--no-llm` silently makes real LLM calls
(audit F8).

**The seam**: both agent classes cache their compiled graph in `self._react_agent` and return it early when
already set — `if self._react_agent is not None: return self._react_agent`
(`tire_agent.py:980-981`, `race_situation_agent.py:973-974`). The engine constructs **its own private**
`TireAgent()` / `RaceSituationAgent()` instances (classes are importable; same pattern as the
`_get_default_*` factories) and pre-sets `instance._react_agent = _NullReActRunner(...)` before first use.
Then the engine calls the instances' **public** `run_from_state(...)` end to end, which reuses with zero
copies: the state-priming blocks (`tire_agent.py:1070-1114`, `race_situation_agent.py:1070-1104`), the
wet-compound stub branch (`tire_agent.py:1142-1155`), `_parse_tool_outputs`, the output rounding, and the
RCM SC override (`race_situation_agent.py:1149-1164` — `sc_currently_active` keeps working in no-llm, which
today's stub path loses entirely).

**The runner** (per lap, seeded by the engine with the args it already owns):

- Tire instance: executes `agent._tools[0]` (`predict_tire_deg_tool`) and `agent._tools[1]`
  (`estimate_laps_to_cliff_tool`) with `(driver, compound_id, tyre_life)`; `compound_id` comes from the
  imported `_compound_name_to_id(compound, gp_name, year)` — the same call `run_from_state` makes at
  `tire_agent.py:1082-1085`. The tool closures read the instance state that `run_from_state` just primed,
  so the numbers are exactly what the LLM path's tools would compute (single TCN forward + the 50-pass
  MC-dropout loop).
- Situation instance: executes `predict_sc_tool(lap_number)` always, and
  `predict_overtake_tool(driver_x, driver_y, lap_number)` only when a rival is derivable from
  `lap_state["rivals"]` (the engine owns lap_state, so this is input prep, not a mirror). Skipping the
  overtake tool on rival-less laps is safe: `_parse_tool_outputs` defaults every field to `0.0`
  (`race_situation_agent.py:574`), which matches the agent's own system-prompt rule ("if gap > 2.5s ...
  assume P(overtake) = 0.0", `race_situation_agent.py:628`).
- Envelope: real `langchain_core.messages.ToolMessage` objects carrying the tools' string outputs (the same
  strings the regex parsers were written against) + a final `AIMessage` whose content becomes the `reasoning`
  field, satisfying both `_run_core` reasoning extractors (`tire_agent.py:1165-1170` skips tool-call
  messages; `race_situation_agent.py:1147` takes `messages[-1].content`).

**Why this over the alternatives:**

- *Alt A — keep today's attempt-and-catch stubs*: keeps fake tire numbers in the MC, keeps the ~5-8 s/lap
  retry backoff (audit F8), keeps the "silently LLM when LM Studio is up" semantic hole. Fails the audit's
  Phase 1.2 exit ("no clients", real-inference 0.3-0.7 s target that explicitly includes "TCN 1+50 forwards").
- *Alt B — mirror the tool bodies + state priming into the engine*: ~80-100 copied lines across two agents;
  recreates exactly the F10 drift class this whole design exists to retire.
- *Alt C — additive `run_direct_from_state()` entry points inside the agent modules*: cleanest long-term home
  and permitted by the letter of the repo rule ("`src/agents/` internals — additive entry points only",
  CLAUDE.md §0.2), but it edits untouchable files, which #169's charter forbids without Víctor's explicit
  sanction. Kept on the table as open question **Q2** — if sanctioned, the null runner shrinks to nothing and
  the injection seam disappears.

The injection is the least-copy option available without touching `src/agents/`: one ~30-line class, zero
duplicated model/feature logic, and the private-attribute dependency (`_react_agent`, `_tools`) is registered
as a named risk (§6) exactly like the seven orchestrator helpers arcade already imports.

**Why private instances, not the module singletons**: injecting into `_get_default_tire_agent()`'s singleton
would leak the null runner into any later LLM-mode call in the same process (the backend serves both modes
across requests). Cost of privacy: a second copy of the TCN bundles / LightGBM models in memory for no-llm
processes — acceptable (bundle sizes are small; a no-llm CLI run typically never builds the LLM-mode
singletons at all, so in practice memory is flat).

### 4.3 N29 radio without stage 3

`run_radio_agent` already computes everything load-bearing before its LLM: NLP inference
(`run_pipeline`/`run_rcm_pipeline`, `radio_agent.py:977-978`) and deterministic alerts
(`_build_alerts`, `:985`); the LLM only adds `reasoning`/`corrections` and is already internally optional
(`:994-1006`). But calling `run_radio_agent_from_state` still constructs the client and pays retries when the
backend is down, and really calls the LLM when it is up. The engine's `_run_radio_no_llm` therefore composes
the same three imported functions directly:

1. Coerce inputs with the orchestrator's own `_to_radio_message` / `_to_rcm_event` (the same coercion
   `_run_always_on_agents_from_state` applies at `strategy_orchestrator.py:1063-1064`).
2. `radio_results = [run_pipeline(m.text) for m in radio_msgs]`; `rcm_results = [run_rcm_pipeline(e) for e in rcm_events]`.
3. `alerts = _build_alerts(radio_results, rcm_results, radio_msgs)`.
4. `RadioOutput(radio_events=..., rcm_events=..., alerts=..., reasoning="[no-llm mode — radio synthesis skipped, NLP stages 1+2 applied]", corrections=[])`
   — the same wording family `run_radio_agent` itself uses on LLM failure (`radio_agent.py:1005`).

Empty lap: both lists are empty, all three calls are O(1) no-ops — the F5 guard is free here (rich-mode F5
remains Phase 1.5). Note `run_radio_agent_from_state`'s `LAPS`/`SESSION_META` global priming is skipped; per
its own docstring the laps frame is "not queried during inference" (`radio_agent.py:1034-1037`).

### 4.4 Deterministic synthesis through the real assembly helper

Rather than hand-building a `StrategyRecommendation` (a drift-prone field-by-field copy),
`_deterministic_synthesis(best, guardrail_reason)` constructs a `_LLMSynthesis` (imported) with:
`action=best`, `reasoning="[no-llm mode]" (+ " " + guardrail_reason when set)`, `confidence=0.0`,
`pace_mode="NEUTRAL"`, `risk_posture="BALANCED"`, everything else default/None — and passes it through the
imported `_assemble_recommendation(synth, pit_out=None, mc_results, regulation_context="")`. Any future field
added to the schema flows through automatically; the no-llm rec can never structurally diverge from the rich
rec.

### 4.5 Guard-rails: where they live and why

The deterministic decision policy exists today in two places: inline in the untouchable CLI
(`run_simulation_cli.py:1529-1560`) and extracted as
`apply_guard_rails(action, lap, total_laps, compound, tyre_life, cliff_p10) -> (action, reason | None)` in
the backend (`src/telemetry/backend/services/simulation/guard_rails.py:30`). The engine cannot import the
backend version — dependency direction would invert (the submodule imports the parent, e.g.
`simulator.py:329`, never the reverse; parent surfaces like `f1-sim`/`f1-arcade` must not require the
submodule checkout). So `no_llm.py` **re-hosts the backend's function with the identical signature and
rule set** (no-pit-before-lap-5, no-pit-in-last-3 unless cliff P10 < 2, minimum stint SOFT 8 / MEDIUM 12 /
HARD 15 — mirroring the LLM prompt rails at `strategy_orchestrator.py:862-877`). Yes, that is a third copy at
birth — but it is the designated canonical one: the backend copy retires when P1 migrates
`_run_no_llm_path` to `engine.run_lap`, and the CLI copy dies with the P4 duplicate. Net copies trend 3 → 1.

### 4.6 Deliberate semantic deltas vs today's `_run_no_llm` (each is a fix, each must be in the PR text)

1. **Real tire/situation numbers instead of hardcoded stubs** when no LLM backend is reachable (§4.2).
2. **Routing sees `sc_currently_active`**: the CLI and backend no-llm paths call `_decide_agents_to_call`
   with only 3 args (`run_simulation_cli.py:1497-1501`, `simulator.py:389-393`), silently defaulting the SC
   flag to False; the engine passes `situation_out.sc_currently_active` like the orchestrator does
   (`strategy_orchestrator.py:1375-1380`) — the RCMContextResolver lesson (Qatar V7) says this flag is
   load-bearing.
3. **True offline semantics**: no client construction at all, so `--no-llm` with LM Studio up no longer
   silently becomes LLM mode (F8), and with it down there is no retry backoff.
4. **N28/N30 never attempted** (vs attempted-then-caught): same end state (`pit_out=None`, `rag=""`) minus
   the retry storm; `active` still reports what *would* have been routed, so panels stay honest.

---

## 5. Parity test design — `tests/test_engine_parity.py` (the audit §9 anti-drift guard)

Tier: `@pytest.mark.data` + `@pytest.mark.llm` (markers registered in `pyproject.toml:214-220`), plus the
`_skip_no_models` guard from `tests/test_strategy_goldens.py:29-33` — the `src.agents` import chain reads
model configs at import time, so this is a data-tier test by construction (runs locally + on the data tier,
skips on bare CI). FakeOpenAI is loaded from the submodule file
(`src/telemetry/tests/fake_openai.py`) via `importlib.util.spec_from_file_location`, with
`pytest.skip` when the submodule is not checked out and when port 1234 is taken (the server raises `OSError`
on bind — its documented skip contract).

### 5.1 Fixture and environment

- `laps_df` = `tests/fixtures/mini_race.parquet` (9-lap Lusail 2025 slice, laps 5-13, SC on 7-10, drivers
  VER/GAS/ANT/ALO/LEC/STR — `tests/fixtures/generate_mini_race.py:27-30`).
- `race_state` = hand-built `RaceState` for a **quiet lap** (e.g. VER, lap 6, pre-SC, no radio/RCM buffers)
  so routing activates neither N28 nor N30 and the LLM script stays short.
- `monkeypatch.setenv("F1_LLM_PROVIDER", "lmstudio")` so every `ChatOpenAI` targets the stub's
  `http://localhost:1234/v1`; `monkeypatch.setattr(strategy_orchestrator, "_orchestrator_llm", None)` to
  reset the module-level LLM singleton between environments.

### 5.2 Test 1 — rich-profile parity (engine == orchestrator)

Script the stub with the quiet lap's exact 7-turn sequence, run the **orchestrator**, re-script the identical
sequence, run the **engine**, compare:

```
turn 1  N27 ReAct  -> push_tool_call("predict_sc_tool", {"lap_number": 6})
turn 2  N27 ReAct  -> push_text("<situation reasoning>")
turn 3  N26 ReAct  -> push_tool_call("predict_tire_deg_tool", {driver, compound_id, tyre_life})
turn 4  N26 ReAct  -> push_tool_call("estimate_laps_to_cliff_tool", {driver, compound_id, tyre_life})
turn 5  N26 ReAct  -> push_text("<tire reasoning>")
turn 6  N29 synth  -> push_tool_call("RadioSynthesis", {reasoning: "...", corrections: []})
turn 7  N31 synth  -> push_tool_call("_LLMSynthesis", {action: "STAY_OUT", reasoning: "...", confidence: 0.7,
                                                       pace_mode: "NEUTRAL", risk_posture: "BALANCED", ...})
```

(Structured-output turns must arrive as tool_calls whose function name matches what
`with_structured_output` registers — verify the exact names against `fake.requests` during test bring-up;
the stub records every request body precisely for this. Call ORDER is deterministic today because the only
threaded pair is N25∥N27 and N25 makes zero LLM calls — `strategy_orchestrator.py:1071-1081`. When F6's
4-way pool lands (Phase 2.2), FIFO scripting breaks and the stub needs content-based routing; flagged now.)

Asserts, strongest first:

1. `rec_engine.model_dump() == rec_orch.model_dump()` — full-recommendation equality (both runs consumed
   byte-identical scripted replies, so everything downstream must match; `scenario_scores` is seed-42 MC and
   would match even under different prose).
2. **Prompt-byte parity**: `fake.requests[i]["messages"] == fake.requests[i + 7]["messages"]` for
   `i in range(7)` — the engine sent the orchestrator's exact prompts, i.e. the argument plumbing between
   the imported layers is identical. This is the single assert that makes silent drift impossible.
3. `agent_outputs` sanity: keys per §2.2, `pit_out is None`, `active == []`, `regulation_context == ""`.
4. Same test parametrized over `lap_state=None` (both sides build their default lap_state — covers the
   engine's `_build_default_lap_state` copy against the orchestrator's inline block via assert 2) and an
   explicit hand-built `lap_state` (the arcade/CLI calling convention).

### 5.3 Test 2 — no-llm profile: zero error frames, zero LLM

Sweep **all 9 fixture laps** (VER, laps 5-13; inject one synthetic `SAFETY_CAR_DEPLOYED` RCM dict on lap 8 to
exercise the SC override + routing + guard-rails deterministically):

1. **Zero clients, proven two ways**: (a) `monkeypatch.setattr(<each agent module>.ChatOpenAI, ...)` with a
   class whose `__init__` raises `AssertionError("LLM client constructed in no-llm profile")` — patched in
   `tire_agent`, `race_situation_agent`, `radio_agent`, `pit_strategy_agent`, `strategy_orchestrator`; and
   (b) `assert fake.requests == []` (nothing ever reached the stub; the stub can even be omitted here — the
   ChatOpenAI bomb is the real assert, making this sub-test hermetic apart from model weights).
2. **Zero error frames**: every lap returns without raising; `rec.action` in the 5-value enum;
   `rec.reasoning.startswith("[no-llm")`; `rec.scenario_scores` has all 4 strategies with the full
   `{E, P10, P90, score}` shape (ties into `test_strategy_goldens.py`).
3. **SC lap semantics**: on the injected-SC lap, `situation_out.sc_currently_active is True`,
   `"N28" in active and "N30" in active` (routing parity with the orchestrator's truth table), yet
   `pit_out is None` and no LLM constructed.
4. **Determinism**: run lap 6 twice; `rec1.model_dump() == rec2.model_dump()` and
   `timings` keys present both times.

This pair of tests is also the enabler for the #180/#182 call-count spy (per
`test_strategy_goldens.py:17-19`): with one inference path, a later test wraps the four `run_*_from_state`
entry points with counters and asserts exactly one call per lap — designed-for here, not delivered in #169.

---

## 6. Untouchability and risk register

**Nothing edited in**: `src/agents/**` (all engine access is imports of existing symbols plus attribute
injection on engine-private instances), `scripts/run_simulation_cli.py` (the CLI keeps its broken `--no-llm`
until the P4 duplicate consumes the engine; `tests/test_cli_no_llm.py` stays `xfail` and flips green with
P4, or with the 1-line sanctioned hotfix if Víctor approves it via #166), `notebooks/**`, `legacy/**`.
**Edited (allowed)**: new files under `src/strategy/inference/`, new `tests/test_engine_parity.py`,
`src/arcade/strategy_pipeline.py` reduced to a delegate (arcade is explicitly editable; audit F10/P3).

Private-symbol dependency register (the price of anti-F10; every one is named so a signature change is a
findable event, and the parity test converts silent drift into a loud red):

| Private symbol relied on | Where | Risk if it changes | Blast mitigation |
|---|---|---|---|
| `_run_always_on_agents_from_state` | orchestrator:1049 | signature/threading change alters rich parity | parity asserts 1+2 fail immediately |
| `_run_conditional_agents` 3-tuple | orchestrator:1165 | a 4th element re-breaks consumers (the F2 pattern) | engine is the ONLY non-orchestrator caller left after P1/P4; change is 1 place |
| `_decide_agents_to_call`, `_run_mc_simulation`, `_build_orchestrator_prompt`, `_get_orchestrator_llm`, `_assemble_recommendation`, `_LLMSynthesis`, `_to_radio_message`, `_to_rcm_event` | orchestrator | drift in routing/MC/prompt/assembly | goldens (`test_strategy_goldens.py`) freeze MC + routing; parity freezes prompt + assembly |
| `TireAgent._react_agent` / `._tools` cache seam | tire_agent:709-710, 980-981 | renaming the attribute or eagerly building the graph breaks the injection | no-llm test 2 fails at first lap; seam documented in `no_llm.py` docstring with a "WHERE TO CHANGE IF" breadcrumb |
| `RaceSituationAgent._react_agent` / `._tools` | race_situation_agent:973-974, 987 | same | same |
| `_compound_name_to_id` | tire_agent:599 | arg-prep for the tire tools drifts from `run_from_state` | parity of tire numbers between profiles is asserted indirectly by test 2 shape checks; goldens cover thresholds |
| `run_pipeline`, `run_rcm_pipeline`, `_build_alerts` | radio_agent:529/604/~660 | radio stages drift | these ARE the load-bearing logic (docstring at radio_agent:924-927); any change hits rich mode identically |

Explicitly OUT of #169 (scheduled elsewhere): **F6** 4-way always-on parallelism (Phase 2.2, gated on the
Phase 0.4 LM Studio concurrency probe; also breaks the FIFO stub scripting, §5.2 note). **F2 PMV hotfix**
(P4 duplicate is the recommended fix vehicle; interim 1-liner only with Víctor's explicit sanction on #166).
**F7** frame filtering, **F4** RAG cache, **F5** rich-mode silent-radio guard, **F11/`fast`** synthesis
cadence. The engine's `run_lap` signature already leaves room for all of them without breaking consumers.

---

## 7. Open questions for Víctor (decisions needed before/while implementing)

- **Q1 — rich approach**: ratify option (a) (engine re-drives the sequence via imported orchestrator layers,
  subsuming arcade's pipeline) over wrapping the orchestrator. Recommendation: (a); (b) requires runtime
  patching of an untouchable module.
- **Q2 — the no-llm N26/N27 seam (the big one)**: sanction the `_NullReActRunner` injection on
  engine-private agent instances (zero body copies, real TCN/LightGBM numbers, but a documented dependency on
  the private `_react_agent`/`_tools` attributes), **or** authorize Alt C: small additive
  `run_direct_from_state()` entry points inside `tire_agent.py`/`race_situation_agent.py` (permitted by the
  letter of "additive entry points only", cleaner long-term, but it adds code to `src/agents/` files).
  Recommendation: injection for #169 (keeps `src/agents/` byte-identical); revisit Alt C when Phase 2.1
  builds direct mode for the LLM profiles anyway, which is the natural moment to bless additive entry points.
- **Q3 — Phase 0.2 folding**: stage-level timings ship inside `run_lap` (§2.3), which covers most of the
  Phase 0.2 timing harness. Accept "per-stage now, per-agent in Phase 2.2", or insist on per-agent timings in
  #169 (would force unbundling `_run_always_on_agents_from_state` and copying its pool logic — not
  recommended)?
- **Q4 — `fast` boundary**: confirm `PROFILES = ("rich", "no-llm")` for #169 with `"fast"` raising a pointing
  error (reserved for Phase 2.1/3.1: direct-mode sub-agents + event-triggered N31).
- **Q5 — guard-rail canonical home**: accept `no_llm.py` hosting the third copy of `apply_guard_rails` as
  canonical-going-forward (backend copy retires at P1 migration, CLI copy at P4)? The alternative (importing
  from the submodule) inverts the dependency direction and is not recommended.
- **Q6 — no-llm semantic deltas**: the four deltas in §4.6 (real numbers, `sc_currently_active` in routing,
  true-offline, never-attempt N28/N30) are improvements but visible behavior changes vs today's broken path —
  confirm they are wanted as part of the #166 fix narrative (they will be listed in the PR body).
- **Q7 — private no-llm agent instances**: accept the small memory duplication (second TCN bundle set) in
  exchange for zero contamination of the LLM-mode singletons? (Recommended; the alternative, injecting into
  the process singletons, is only safe if a process is guaranteed single-profile, which the backend is not.)
- **Q8 — test tier labeling**: parity test as `data` + `llm` markers with in-file skips for missing submodule
  / occupied port 1234 — matches the existing marker taxonomy (`pyproject.toml:214-220`)?

---

## Appendix — evidence index (file:line)

| Claim | Evidence |
|---|---|
| Orchestrator discards agent outputs; sequence to reproduce | `src/agents/strategy_orchestrator.py:1369-1418` |
| 3-tuple contract of `_run_conditional_agents` | `strategy_orchestrator.py:1111-1120, 1165` |
| Arcade single-pass verbose pipeline + 7 private imports + mirror warning | `src/arcade/strategy_pipeline.py:11-20, 28-36, 42-121` |
| Arcade default lap_state copy | `strategy_pipeline.py:124-167` vs orchestrator inline `1327-1367` |
| CLI probe+orchestrator double-run | `scripts/run_simulation_cli.py:1961-1964`; ack `:489-495` |
| CLI broken 2-of-3 unpack (#166) | `run_simulation_cli.py:1508` |
| CLI no-llm guard-rails (policy to preserve) | `run_simulation_cli.py:1529-1560` |
| CLI no-llm hardcoded stubs (fake tire numbers) | `run_simulation_cli.py:1443-1472` |
| CLI/backend routing omits `sc_currently_active` | `run_simulation_cli.py:1497-1501`; `simulator.py:389-393` |
| Backend third mirror + extracted guard-rails | `src/telemetry/backend/services/simulation/simulator.py:308-442`; `guard_rails.py:30-37` |
| Backend imports parent (dependency direction) | `simulator.py:329-333` |
| Tire `_react_agent` cache seam + tools + `_run_core` | `src/agents/tire_agent.py:704-710, 828-946, 980-981, 1053-1181` |
| Situation seam + `_parse_tool_outputs` defaults + RCM override | `src/agents/race_situation_agent.py:561-588, 973-974, 1046-1173` |
| Radio NLP-first, internal LLM degradation, load-bearing stages | `src/agents/radio_agent.py:921-1014` (esp. 976-985, 994-1006) |
| Pace agent no-LLM by construction | `src/agents/pace_agent.py:711-723`; audit §2.1 row 1 |
| MC golden freeze + routing truth table | `tests/test_strategy_goldens.py:80-99, 160-182` |
| CLI no-llm xfail smoke (flips with P4) | `tests/test_cli_no_llm.py:27-54` |
| Fixture: 9-lap Lusail slice with SC | `tests/fixtures/generate_mini_race.py:27-30`; `tests/fixtures/README.md` |
| FakeOpenAI stub on :1234, scripting + skip contracts | `src/telemetry/tests/fake_openai.py:1-49, 184-234` |
| pytest marker taxonomy | `pyproject.toml:213-220` |
