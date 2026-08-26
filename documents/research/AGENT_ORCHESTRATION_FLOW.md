# Multi-Agent Orchestration Flow: Assessment and LangGraph v2 Design

Status: design document, no code. Written 2026-07-07 against the shipped agent stack.
Scope guard: the **multi-agent architecture is a hard keep**. All six sub-agents (N25-N30)
remain agents, and the orchestrator remains a supervisor over them. This document does NOT
propose replacing agents with plain functions. It assesses whether the *flow* that wires
the agents together (routing, fan-out, loop configuration, checkpointing, streaming) can
be improved, and designs an additive v2 graph if so.

Cross-references (not duplicated here):

- `documents/audits/AUDIT_P2B_CORE_COMPUTE.md` (epic #169): probe duplication F1, ReAct
  turn inflation F3, sequential always-on F6, N31 cadence F11, the shared engine plan.
- `documents/audits/AUDIT_LLM_COST_LATENCY.md` (epic #261): per-agent token table, L-1
  timeouts, L-2 model config, L-4 cache observability, prompt-cache restructuring.
- `documents/audits/AUDIT_ML_AGENTS_EVAL.md` (epic #205): conformance battery, MC and
  routing evaluation, golden regression bed.
- `documents/research/RIVAL_AGENT_DESIGN.md` section 7: the Rival Agent as an additive node.

---

## 1. Framing: frozen PMV, additive v2

`src/agents/` internals, `notebooks/**`, and `scripts/run_simulation_cli.py` are
untouchable (the TFG PMV, defended 2026-06-09). Every proposal below is therefore
**additive**: a new graph module (working location `src/agents/graph/`, or folded into
the P2b shared engine at `src/strategy/inference/`) that *wraps the frozen public entry
points* (`run_*_agent_from_state`, `run_strategy_orchestrator_from_state`) or duplicates
the thin orchestration wiring around them. The shipped pipeline stays byte-for-byte
intact and remains the control arm for any comparison, including the TFM ablation.

This is also a check, not a mandate: where the current flow is sound, this document says
so and the v2 graph preserves it.

## 2. The current flow, verified against code

### 2.1 The seven agents

| Agent | Module | Pattern (shipped) | LLM in per-lap path | Cadence |
|---|---|---|---|---|
| N25 Pace | `src/agents/pace_agent.py` | Direct call: `PaceAgent.run()`, reasoning is a template string. (Written 2026-07-07 against a ReAct scaffold that existed but was idle; formally retired and deleted in #781 after the #778/#779/#780 archaeology confirmed it was never wired and had no stated future use. Do not plan v2 work around resurrecting it.) | none | every lap |
| N26 Tire | `src/agents/tire_agent.py` | ReAct (`create_agent`, :997; invoke :1162), 2 tools | ~3 turns | every lap |
| N27 Situation | `src/agents/race_situation_agent.py` | ReAct (:985; invoke :1145), 2 tools | ~3 turns | every lap |
| N28 Pit | `src/agents/pit_strategy_agent.py` | ReAct (:837; invoke :996), 3 tools; output parsed from tool messages + final-message prose | up to 4 turns | conditional |
| N29 Radio | `src/agents/radio_agent.py` | NLP-first synthesizer: deterministic NLP + alerts, then ONE `with_structured_output(RadioSynthesis)` call (:996) | 1 call | every lap |
| N30 RAG | `src/agents/rag_agent.py` | ReAct (:160; invoke :194), 1 Qdrant retrieval tool | 2-3 turns | conditional |
| N31 Orchestrator | `src/agents/strategy_orchestrator.py` | Plain Python 3-layer pipeline (not a graph); one `with_structured_output(_LLMSynthesis)` call (:1299) | 1 call | every lap |

### 2.2 The orchestrator wiring (the flow under assessment)

`run_strategy_orchestrator_from_state` (`strategy_orchestrator.py:1303-1418`) executes:

1. **Layer 1a, always-on agents** (`_run_always_on_agents_from_state`, :1049-1082).
   Partial parallelism only: N25 + N27 in a `ThreadPoolExecutor(max_workers=2)`,
   then N26 and N29 strictly sequential (comment cites PyTorch/MLX thread-safety
   caution). The FastF1 twin `_run_always_on_agents` (:993-1046) is fully sequential.
2. **Layer 1b, MoE routing** (`_decide_agents_to_call`, :475-537). Deterministic
   if-else rules over N26 warning level, N27 SC probability, N29 alert intents, and
   the confirmed-SC flag. Activates N28 and/or N30.
3. **Layer 1c, conditional agents** (`_run_conditional_agents`, :1085-1165). N28 then
   N30, sequential, with N30's question chosen from three canned templates
   (`_build_rag_question`, :717-738).
4. **Layer 2, Monte Carlo** (`_run_mc_simulation`, :609-710). Seeded rng (42), 500
   draws, 4 candidates, `score = alpha*E + (1-alpha)*P10`. Fully deterministic.
5. **Layer 3, LLM synthesis** (`_build_orchestrator_prompt` :741-946, invoke :1417).
   One structured-output call; `_assemble_recommendation` (:1172-1221) backfills N28
   values and attaches `scenario_scores` + `regulation_context` in code.

The whole Layer 1-3 wiring is duplicated across the two entry points (:1228-1300 vs
:1303-1418), and duplicated a third time in `src/arcade/strategy_pipeline.py` (P2b F10).
The CLI additionally runs the four always-on agents twice per lap (probe + orchestrator,
P2b F1).

## 3. Assessment: what is sound, what the flow leaves on the table

### 3.1 Sound, keep as-is (also in v2)

- **Deterministic MoE routing.** The original N31 plan considered an LLM supervisor
  choosing which workers to call (memory `project_agent_notebooks.md`). The shipped
  deterministic gate is the better design: free, testable, auditable, and it makes the
  routing itself a unit-testable function. v2 keeps the exact rules and only changes
  *where* they live (conditional edges instead of inline if-else). Do not move routing
  into an LLM.
- **MC as a deterministic, seeded layer** between agents and synthesis. This is the
  academic payoff of the probabilistic pipeline and the anchor for reproducibility.
- **Structured output everywhere the LLM commits to a decision** (N29, N31). The
  in-house proof that schema-validated single calls are reliable.
- **N30 as a hard constraint injected before the decision LLM**, and the layered
  guard-rails (prompt-level in N26/N27/N28/N31 + code-level SC override in N28 +
  programmatic guard in the no-LLM path; memory `project_strategic_guardrails.md`).
- **The frozen 14-field `StrategyRecommendation`** (memory
  `project_orchestrator_v2_schema.md`). v2 changes nothing about the output contract.

### 3.2 Improvable: the flow verdict table

All agents stay agents. The findings below are about wiring and loop configuration.

| # | Flow concern | Today | Verdict |
|---|---|---|---|
| V1 | Fan-out of always-on agents | 2-of-4 parallel in the RSM path, 0-of-4 in the FastF1 path | Improvable. N25/N26/N27/N29 are data-independent per lap; a graph-native parallel fan-out of all four (plus Rival later) bounds Layer 1a by the slowest agent instead of the sum. The PyTorch caveat must be settled empirically (see open questions), not assumed. P2b F6. |
| V2 | Conditional agents N28/N30 | Sequential even when both are active | Improvable. N30's canned question depends on `pit_out.action` only in one branch; the SC-triggered and alert-triggered N30 queries do not need N28's output and can run in parallel with it. Small win, free in a graph. |
| V3 | ReAct loop configuration | Default `create_agent` loops: no recursion cap, no forced tool plan, no timeout, no retry policy (LLM-cost L-1); ~3-4 turns where the tool sequence is known a priori (P2b F3) | Improvable *within* the agent wrapper: bind a bounded recursion limit, force the known tool plan on the first turn where the provider supports it, and set timeouts/retries from config. The agents remain ReAct agents; the loop just stops paying for wandering. |
| V4 | Sub-agent output parsing | N26/N27/N28 parse numbers from tool messages and the final message's prose (`_parse_agent_summary`, e.g. `pit_strategy_agent.py:999`) | Improvable: a structured-output final turn (the N29/N31 pattern) removes the last nondeterministic parse in the flow. For the frozen agents this is a v2-wrapper concern only if outputs ever misparse; the tool-message parse (`_parse_tool_outputs`) is already deterministic and carries the numeric payload. Low priority, note it and move on. |
| V5 | Duplicated wiring | Two orchestrator entry points + arcade `strategy_pipeline.py` + CLI probe duplication | The core P2b finding (F1/F10). One graph, consumed by every surface, retires all three duplications. |
| V6 | Checkpointing / resume | None. A crash at lap 40 restarts the race; no time-travel debugging | Missing capability. Needed for live mode, useful today for replay and for the ML-eval regression bed. |
| V7 | Streaming / progress | Surfaces get one blob per lap over ad-hoc TCP (`src/arcade/stream.py`) or SSE; no intra-lap progress | Missing capability. Per-node events ("tire done, pit running") map directly onto the SSE + pit-wall surfaces (epic #281). |
| V8 | Observability | No per-stage timings, no per-node token accounting (`cached_tokens` discarded, LLM-cost L-4) | A graph gives natural per-node hooks; P2b's engine plan already demands stage timings. |
| V9 | Degradation profiles | `--no-llm` is an accidental degrade path (attempt-and-catch, P2b F8; crash #166) | Improvable: an explicit `no-llm` profile that routes around the synthesis node, sharing the guardrail logic instead of duplicating it in the CLI. |

**Honest bottom line:** the *decision logic* of the current flow is sound and should not
change. What a LangGraph `StateGraph` adds is not intelligence but engineering: full
fan-out, single wiring, checkpoints, streaming, per-node observability, and a clean slot
for the Rival Agent. If live mode, the pit-wall surface, and the TFM were not on the
roadmap, the plain pipeline would be enough; since all three are, the graph
pays for itself.

## 4. Proposed v2: one StateGraph over the frozen agents

New additive module (e.g. `src/agents/graph/strategy_graph.py`, or inside the P2b
engine package). Every node *wraps* a frozen entry point; no frozen file is edited.

### 4.1 State

A single `OrchestrationState` object (Pydantic or TypedDict) carrying: `race_state`,
`lap_state`, the per-agent outputs (`pace_out`, `tire_out`, `situation_out`,
`radio_out`, `pit_out`, `rag_dict`, later `rival_out`), the routing set, `mc_results`,
the final `StrategyRecommendation`, per-node timings, and an error channel. `laps_df`
stays OUT of the state (passed by reference via graph config) so checkpoints stay small.

### 4.2 Topology

```
START
  └─ prepare_lap                    (build lap_state via the shared engine)
       ├─ pace_node        ┐
       ├─ tire_node        │  parallel fan-out (superstep;
       ├─ situation_node   │  each wraps run_*_from_state)
       ├─ radio_node       │
       └─ rival_node (TFM) ┘
  └─ route                          (deterministic MoE = _decide_agents_to_call rules)
       ├─[none]──────────────────────────┐
       ├─[N28]──── pit_node ──┐          │
       ├─[N30]──── rag_node ──┤ parallel │
       └─[both]─── both ──────┘          │
  └─ monte_carlo                    (seeded, 500 draws; Rival-extended in TFM arm)
  └─ synthesize                     (N31 LLM structured output)   ←─ skipped in no-llm profile
  └─ guardrails_assemble            (programmatic guard + backfill + attach scores/reg)
END
```

- **Conditional edges** implement Layer 1b: the `route` node computes the activation
  set from the fan-out outputs and the graph branches on it. Same rules, now visible in
  a rendered graph, unit-testable per edge, and logged per lap.
- **Parallel fan-out** uses LangGraph's superstep semantics: all always-on nodes run in
  one step, each writing its own state key (no shared-key contention). N28 and N30 fan
  out in parallel when both activate; the one N30 question template that reads
  `pit_out.action` falls back to the generic pit question in that case, or N30 is
  sequenced only for that branch (design choice to fix during Phase 1).
- **`monte_carlo` is a plain deterministic node.** Not an agent, never was; keeping it
  as an isolated node makes it the cheapest golden-test target in the system (seeded
  inputs in, exact dict out) and the place where the Rival extension lands.
- **`synthesize`** keeps the single structured-output call but splits the prompt into a
  static `SystemMessage` (guardrails + rubric + field spec) and a short dynamic
  `HumanMessage`, per the LLM-cost audit's prompt-cache restructuring. Provider stays
  OpenAI / LM Studio via `langchain-openai`, model per layer from the L-2 config module.
- **`guardrails_assemble`** merges `_assemble_recommendation` semantics with the
  programmatic guard currently living only in the CLI's no-LLM path, so both profiles
  share one guard implementation.

### 4.3 Profiles

One graph, two-plus compiled profiles (a compile-time flag or a conditional edge):

- `llm` profile: as drawn.
- `no-llm` profile: skips `synthesize`; `guardrails_assemble` promotes the MC argmax
  through the programmatic guard into the same 14-field shape. This is the explicit
  degrade target the LLM-cost audit's L-9 asks for, and the clean fix shape for #166.
- (Optional) `probe` profile for the CLI's pre-flight, so probing never re-runs agents.

### 4.4 Checkpointing

A LangGraph checkpointer (SQLite for local, in-memory for tests) with
`thread_id = session key`, one checkpoint per lap:

- **Replay/resume:** restart a 70-lap simulation at lap 40; time-travel to any lap for
  debugging a bad recommendation.
- **Regression bed:** the ML-eval audit (#205) golden runs become "replay checkpoints,
  assert node outputs", instead of bespoke fixture plumbing.
- **Live mode:** a live OpenF1 feed (see `documents/research/REALTIME_OPENF1_CONSUMER_DESIGN.md`)
  becomes "one graph invocation per lap on the same thread", with crash recovery free.

### 4.5 Streaming

`stream_mode="updates"` yields one event per completed node. Mapping: backend SSE
forwards node events as intra-lap progress; the Arcade TCP broadcaster and the future
pit-wall surface (epic #281) subscribe to the same event shape. Today's per-lap blob
becomes the terminal event, so existing consumers keep working unchanged.

## 5. The Rival Agent slot (TFM)

`RIVAL_AGENT_DESIGN.md` section 7 already specifies the additive construction: a new sibling
agent module, an additive `lap_state` gap-history key, and a duplicated "anticipatory
orchestrator" entry point. In the plain pipeline that means a third copy of the Layer
1-3 wiring. In the v2 graph it collapses to:

- **One node** (`rival_node`) joining the always-on fan-out, reading only
  `lap_state["rivals"]` + the gap provider (single-driver boundary preserved by
  construction).
- **One routing rule** as a conditional edge: skip the rival branch when no rival is
  within a pit cycle (the MoE rule section 7.1 already names).
- **One MC variant**: the `monte_carlo` node gains the rival draws and the modified
  STAY_OUT / UNDERCUT / OVERCUT scoring of section 7.2, behind a flag.
- **One prompt block**: `synthesize` injects RIVAL INTENT (section 7.3) when `rival_out` is
  present. Output stays the frozen 14 fields.

The TFM ablation becomes trivially clean: control arm = graph with the rival branch
disabled (or the untouched shipped pipeline); treatment arm = same graph, flag on. This
is materially cleaner than maintaining a second full orchestrator function, and it is
the strongest single argument for building v2 before the TFM starts.

## 6. What v2 buys, and what it costs

**Buys:**

- **Latency:** full fan-out of 4-5 always-on agents (V1) plus parallel N28/N30 (V2);
  Layer 1 bounded by the slowest agent. Combined with the engine retiring the CLI probe
  duplication (P2b F1, the audit's estimated 40-45% LLM-lap saving) and bounded ReAct
  loops (V3), per-lap wall time and token cost both drop without touching any agent.
- **Determinism and testability:** routing, MC, and guardrails become individually
  golden-testable nodes; checkpoints feed the ML-eval regression bed (#205); the no-llm
  profile is exactly reproducible end to end.
- **One wiring:** the two entry-point twins, the arcade pipeline copy, and the CLI
  probe path converge on one graph (P2b F10/F1).
- **Live-mode and surface readiness:** checkpointing + streaming are the two primitives
  the SSE/pit-wall surfaces and the 2026 live ambition actually need.
- **A first-class Rival slot** (section 5).

**Costs and risks:**

- **Parity risk:** any prompt reshaping (cache-friendly System/Human split) can change
  LLM outputs; must be gated by the ML-eval conformance battery, not assumed neutral.
- **Torch thread safety:** the shipped code deliberately serializes N26/N29; parallel
  fan-out must either verify safety (inference under `no_grad` on separate model
  instances is typically fine) or wrap torch nodes in a lock, which halves V1's win.
- **Dependency/version churn:** LangGraph APIs move; pin versions in the engine extra
  and keep the graph surface thin (nodes are wrappers, easy to re-wire).
- **Effort:** this is a Sprint-sized build that only pays if the P2b engine lands
  first; sequencing per `documents/audits/IMPLEMENTATION_ROADMAP.md` (engine P2b #169
  before graph; graph before Rival).

## 7. Migration path and parity gate

1. **Phase 0, golden capture (blocks everything).** Run the frozen pipeline over 2-3
   reference GPs (e.g. the defended Australia/Hungary/Qatar set) in no-llm and LLM
   modes; record per lap: lap_state, agent outputs, activation set, `scenario_scores`,
   guardrail overrides, final recommendation, prompts. Builds on Testing #181
   (FakeOpenAI stub) and #182 (engine goldens).
2. **Phase 1, no-llm graph.** Build the graph wrapping frozen entry points; compile the
   `no-llm` profile. **Parity gate A (hard, exact):** per lap, activation set equal, MC
   `scenario_scores` dict byte-equal (seeded), guardrail decisions equal, final action
   equal. Any diff is a bug in the graph, by definition.
3. **Phase 2, llm profile against recorded responses.** Replay with the FakeOpenAI stub
   serving Phase 0 recordings. **Parity gate B:** prompt content equal modulo the
   sanctioned System/Human reordering; outputs schema-valid; recommendation fields
   equal under recorded responses. Live-LLM drift is then judged by the ML-eval
   conformance battery (#205), not string equality.
4. **Phase 3, surfaces opt in one at a time.** CLI duplicate (per P4 #236's
   duplicate-and-improve plan) first, then backend SSE, then Arcade (retiring
   `strategy_pipeline.py`). The frozen CLI and pipeline remain untouched and runnable.
5. **Phase 4, capabilities.** Enable checkpointing + streaming per surface; then the
   Rival node behind a flag (TFM), then optional cadence experiments (P2b F11) as a
   conditional edge on the synthesize branch.

Rollback at every phase is "keep calling the frozen entry points"; the control arm
never disappears.

## 8. Open questions

1. **Torch parallelism:** can N26 (TCN + MC Dropout) and N29's NLP models run in the
   same superstep safely, or do torch nodes need a shared lock / separate processes?
   Decides most of V1's latency win. Needs a 30-minute empirical test, not a debate.
2. **State vs. DataFrame:** confirm `laps_df` stays out of checkpointed state (config
   injection or a store handle); otherwise checkpoints balloon.
3. **N30 question dependency:** in the both-active branch, run N30 in parallel with the
   generic question, or keep it sequenced after N28 for the compound-specific question?
   (Cheap A/B during Phase 1.)
4. **Module home:** `src/agents/graph/` (additive-by-new-file inside the agents
   package) vs `src/strategy/inference/` (with the P2b engine). Leaning engine-side, so
   the graph and the engine version together.
5. **Checkpointer choice and retention:** SQLite per session? Retention across a 70-lap
   race times N surfaces? Live mode needs a policy before Phase 4.
6. **LM Studio capability gaps:** forced tool choice / parallel tool calls differ from
   OpenAI; V3's loop-tightening must degrade gracefully on the local provider.
7. **Does the pit-wall epic (#281) consume graph streaming directly**, or keep the
   current TCP/SSE adapters as a translation layer? Affects how much of V7 lands here
   vs there.
