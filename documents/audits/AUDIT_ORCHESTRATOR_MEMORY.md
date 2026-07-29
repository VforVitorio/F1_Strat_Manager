# Audit: the orchestrator memory layer — should it be built?

**Date:** 2026-07-27 · **Role:** adversarial design gate, report-only · **Status: COMPLETE**

**Verdict in one line: build it, with changes, and fix the dropped `temperature=0.0` first —
because Layer 3 has been sampling at the provider default and some of what memory appears to
fix may be that.** Full verdict and numbered plan at the end.

This audits a **proposed, unbuilt** design: a small accumulator in `src/strategy/inference/`
that feeds `_build_orchestrator_prompt` a memory block (`last_action` + `laps_held`, previous
`pit_lap_target` + drift, live contingencies), with `run_lap` gaining one optional parameter
defaulting to `None`.

Prior art this audit must not repeat: `documents/audits/AUDIT_MONITOR_LAYER.md` (2026-07-26,
MONITOR rejected). This audit covers what survived that decision.

## How to read this

Sections appear in the order they were written, which is the order the evidence arrived, not
the order the brief listed. **Audit 1 (§1.1-§1.7) is the one that decides the project** — it
runs after Audit 2 in this document only because its LLM passes were executing in the
background while the static work was done. If you read one thing, read **§1.5**; if you read
two, add **§1.1**. Two sections carry an explicit correction to an earlier section of this same
report (§1.5 corrects itself, §3.3 is overturned by §1.5/§3.6); both are marked in place rather
than rewritten away.

## Constraints under which this ran

- Parallel session on the same repo: **no branch switch, no commit, no push**. The only repo
  file created or modified is this report.
- Branch was `dev` when the audit started and `main` when it finished — **the parallel session
  switched it mid-run**; this audit ran no git command other than `status`/`branch`. All 21
  load-bearing `file:line` citations were therefore re-verified programmatically against the
  final working tree and all 21 hold.
- Provider: `F1_LLM_PROVIDER=openai`, orchestrator model `gpt-5.4-mini`
  (`src/agents/strategy_orchestrator.py:109`), `temperature=0.0`
  (`src/agents/strategy_orchestrator.py:111`).

> Correction to the line above: `temperature=0.0` is what the CONFIG says. §1.1 measures that
> it never reaches the model.

## API calls spent

LM Studio was probed first and is not running (connection refused), so every call went to
OpenAI. **Roughly 226 calls**, of which 177 are in the recorded experiments:

| run | calls | prompt tokens | completion tokens |
|---|---|---|---|
| pass A (no memory, 41 laps) | 41 | 127,747 | 20,080 |
| pass A' (noise floor, 41 laps) | 41 | 127,747 | 19,915 |
| pass B (memory, 41 laps) | 41 | 134,379 | 20,153 |
| transition, lap 44, 3 variants x 10 | 30 | 97,480 | 15,306 |
| Safety Car, lap 42, 3 variants x 8 | 24 | 79,656 | 12,097 |
| **recorded total** | **177** | **567,009** | **87,551** |

Plus ~49 unrecorded: 1 connectivity probe, 4 smoke calls, ~22 in a serial pass A that was
killed to parallelise the three passes, ~22 in a pass B that died on
`openai.APITimeoutError` at lap 22 before checkpointing was added. Every offline stage (input
generation, routing checks, Safety Car injection, all analysis) cost **zero** calls.

## Data

Lusail 2025, NOR / McLaren, laps 5-45, driven through the `no-llm` profile over
`data/raw/2025/Lusail` with `RaceReplayEngine`, per `reference_drive_orchestrator_offline`.
It reproduces the documented baseline exactly: deterministic STAY_OUT on 39 of 41 laps,
UNDERCUT on lap 44, which is Norris's real stop. Harness scripts live in the session
scratchpad; nothing was written to `src/`.

## Checklist

- [x] **AUDIT 1 — does the LLM actually use it?**
  - [x] Offline harness reproduced per `reference_drive_orchestrator_offline`
  - [x] A/B passes generated over a long-hold run (Lusail 2025, NOR)
  - [x] A-vs-A' noise floor measured — and it is the finding that reframed the audit
  - [x] 12 LLM-originated fields diffed, substance vs wording separated
  - [x] `pit_lap_target` stability, `confidence` anchoring
- [x] **AUDIT 2 — what breaks when the engine stops being pure?**
  - [x] Every `run_lap` caller enumerated with `file:line` (repo + submodule)
  - [x] Accumulator ownership per caller
  - [x] Tests assuming purity enumerated
  - [x] `/recommend`, MCP, webapp bypass verified from both ends
  - [x] Gap semantics for `laps_held` — demonstrated lying
  - [x] Where the twin-drift defect reappears
- [x] **AUDIT 3 — adversarial: what does memory break?**
  - [x] Anchoring attacked at Lusail 2025 lap 44, n=10 per variant
  - [x] Transition sharpness
  - [x] Safety Car contamination — injected, n=8 per variant
  - [x] `laps_held` sentinel collision
  - [x] Contingency retirement / prompt growth
- [x] Verdict + numbered plan
- [x] "What I tried to break and could not"

---

# AUDIT 2 — what breaks when the engine stops being pure

Run first in wall-clock order only because Audit 1's LLM passes were executing in the
background. Audit 1 is still the one that decides the project.

## 2.1 The real caller list, `file:line`

**`run_lap` production callers — three surfaces, FOUR call sites:**

| # | site | profile | owns a race-scoped object? |
|---|---|---|---|
| 1 | `scripts/run_simulation_cli.py:1747` | `rich` or `no-llm` (`--no-llm`) | **yes** — `sc_tracker` at `:1595` lives across the lap loop |
| 2 | `src/arcade/strategy_pipeline.py:43` | `rich` (hardcoded) | **no** — pure function; its caller `src/arcade/strategy.py:409` has `SimConnector` |
| 3 | `src/telemetry/backend/services/simulation/simulator.py:426` | `no-llm` (`_run_no_llm_path`) | **yes** — `_stream` locals |
| 4 | `src/telemetry/backend/services/simulation/simulator.py:863` | `rich` | **yes** — same loop, `rcm_runner, sc_tracker` at `:831` |

**Callers that BYPASS `run_lap` entirely** (verified, and this confirms the design record):

| # | site | what it calls |
|---|---|---|
| 5 | `src/telemetry/backend/api/v1/endpoints/strategy.py:1309` | `run_strategy_orchestrator_from_state` — `/recommend` |
| 6 | `src/telemetry/backend/mcp_tools.py:607` | `run_strategy_orchestrator_from_state` — the MCP tool |
| 7 | `scripts/debug_agent.py:310` | `run_strategy_orchestrator_from_state` — dev script |

**Tests calling `run_lap`:** `tests/engine/test_engine_no_llm.py:97,125,138-139` and
`tests/engine/test_engine.py:41,49`.

## 2.2 HIGH — the memory parameter has to be threaded at THREE prompt sites, and the existing anti-drift test does not cover the one that matters

`_build_orchestrator_prompt` is called from three production sites:

- `src/agents/strategy_orchestrator.py:2164` (`run_strategy_orchestrator`)
- `src/agents/strategy_orchestrator.py:2339` (`run_strategy_orchestrator_from_state`) ← what `/recommend` and MCP use
- `src/strategy/inference/engine.py:277` (`_run_rich`) ← what the three live surfaces use

The design says "`_build_orchestrator_prompt` renders one block". Rendering is one place;
**passing the data is three**. This repo has already been bitten twice by exactly this, both
times on this exact pair of functions and both documented in
`src/strategy/inference/engine.py:210-231`: `live_drivers` was threaded by the orchestrator
and not by the engine, and `cliff_p50`/`total_laps` likewise.

The guard that exists, `tests/engine/test_engine_threads_every_argument.py:33-45`, AST-parses
for calls to **`_assemble_recommendation` only** (`node.func.id == "_assemble_recommendation"`,
line 42). It does not look at `_build_orchestrator_prompt`. So the memory argument is in the
one position where the repo's dominant defect is both most likely and least covered.

**Failure scenario, concrete:** memory is added, threaded at `engine.py:277`, shipped. The
CLI, arcade and `/simulate` get continuity. `/recommend` and MCP call
`run_strategy_orchestrator_from_state`, whose `_build_orchestrator_prompt` call at
`strategy_orchestrator.py:2339` still has no memory argument, so it silently renders the old
prompt. Nothing fails, no test goes red, and the divergence is invisible until someone diffs
two prompts by hand. **Severity: HIGH.**

**What would prevent it:** extend the AST test to parametrise over both callee names, i.e.
compare `_kwargs_passed_by(engine._run_rich, "_build_orchestrator_prompt")` against
`_kwargs_passed_by(orch.run_strategy_orchestrator_from_state, "_build_orchestrator_prompt")`.
That is a five-line change to an existing test and it is the cheapest insurance in this audit.

## 2.3 HIGH — `/recommend`, MCP and the webapp structurally cannot have memory, and this must be declared

Verified at `src/telemetry/backend/api/v1/endpoints/strategy.py:1267-1313`: `recommend_strategy`
is a stateless POST. It receives `request.lap_state` for ONE lap, builds a `RaceState`, calls
the orchestrator and returns. There is no race-scoped object anywhere in the request path, and
the endpoint is rate-limited per call (`:1265`), not per session. Same shape at
`src/telemetry/backend/mcp_tools.py:580-607`.

**And the webapp is on the wrong side of that line, verified from the consumer:**
`src/telemetry/webapp/src/lib/api/strategy.ts:360` posts to `recommend`, and
`src/telemetry/webapp/src/features/strategy/queries.ts:8` documents the orchestrator run as
`POST /recommend`. The chat path reaches the same place through the MCP tool
(`stageLabels.ts:23` `calling_recommend_strategy`). So the surface a user is most likely to
be looking at is the one that cannot have memory.

So the memory design covers **3 of 6 production consumers**. The other three keep today's
stateless prompt. That is not a bug in the design, but it must be written into the docstring
and the issue, because the alternative is the failure mode recorded in
`feedback_my_blind_spots_directive` §7: claiming a cleanup went further than it did.

The honest options are (a) declare the limitation, (b) have the client pass the memory in the
request body (turns a private accumulator into a public API contract — much bigger), or
(c) server-side session state (a new lifetime concept the backend does not have).
**(a) is the only one proportionate to the value.** **Severity: HIGH** as a scope statement,
not as a defect.

## 2.4 MEDIUM — `laps_held` counts DECISIONS, not laps, and every surface skips laps

Three of the four `run_lap` call sites sit inside loops that `continue` past laps:

- `simulator.py:835-836` — out of the requested `[lap_start, lap_end]` range
- `simulator.py:842-845` — the DNF / incomplete-lap guard (`_lap_skip_reason`)
- `simulator.py:876-878` — `except Exception` per lap: `state.error_laps += 1`, then continue,
  so `_accumulate` at `:869` never runs for that lap
- `scripts/run_simulation_cli.py:1605-1608` (range), `:1621` (DNF), `:1643` (incomplete)

An accumulator fed only where a recommendation exists therefore counts recorded decisions.
Rendering that as "held for N laps (since lap X)" is false whenever `X + N - 1 != current`.

**Executed evidence** (prototype at `scratchpad/memblock.py`, laps 5-10 recorded, 11-19
skipped, lap 20 recorded):

```
  Last call: STAY_OUT, held for 7 consecutive laps (since lap 5).
--- laps actually elapsed since lap 5: 16  | laps_held rendered above: 7
```

The block tells the LLM a nine-lap gap did not happen. **Severity: MEDIUM.** The fix is
cheap and should be in the first implementation: record the lap number with every entry and
render the SPAN, never a bare count — or state both and let the discrepancy be visible.

## 2.5 MEDIUM — arcade's existing history cannot supply the contingency field

`src/arcade/strategy.py:409` already appends every decision to `StrategyState.history`
(`:147`), so "the surface keeps the previous recommendation" is half-true for arcade — the
memory in `MEMORY.md` says no surface holds it, and for `action`/`pit_lap_target` arcade
does.

But `LapDecisionDTO` (`src/arcade/strategy.py:110-132`) carries `action:117`,
`pit_lap_target:124`, `compound_next:125`, `undercut_target:126` and **no `contingencies`
field at all**. `_build_decision` at `:800-819` never reads `rec.contingencies`.

So the third memory field cannot be sourced from arcade's history; the accumulator must hold
its own copy, taken from the `StrategyRecommendation` before it is flattened to the DTO.
That is fine, but it means the accumulator is NOT a view over existing state on any surface —
it is new state on all three. **Severity: MEDIUM** (scoping, not correctness).

## 2.6 HIGH — the precedent for this exact design already exists, and it has already drifted

`RaceControlStateTracker` (`src/nlp/rcm_state.py:83`) is the proposed design, already built:
one class in `src/`, a per-race accumulator owned by the caller, `ingest` per lap. It proves
feasibility. It also shows the cost, because its three wirings are three different shapes:

- `scripts/run_simulation_cli.py:1595` + `:1732-1734`, and the ingest/inject sits inside
  `try: ... except (AttributeError, TypeError): pass` (`:1738-1739`)
- `src/arcade/strategy.py:233` (`self._sc_tracker`) + `:646-648`
- `src/telemetry/backend/services/simulation/simulator.py:317` (`_build_rcm_feed`), `:831`, `:848`

And the drift is already there, with the same name in two places:

```
src/telemetry/backend/api/v1/endpoints/strategy.py:1231  def _rcm_events_for_lap(year, gp, laps_df, lap)
src/telemetry/backend/services/simulation/simulator.py:345  def _rcm_events_for_lap(runner, tracker, lap)
```

Two functions, one name, different signatures, different mechanisms — the endpoint replays
from lap 1 per request because it has no tracker to carry (`simulator.py:826-830` says so
explicitly). **This is where the twin will reappear for memory**: the moment someone wants
`/recommend` to have continuity, a second accumulator with the same name and a
replay-from-lap-1 body is the path of least resistance, exactly as it was for the SC state.
**Severity: HIGH** as a prediction; the mitigation is 2.3 (declare the limitation) plus
naming the accumulator something a per-request reimplementation cannot quietly shadow.

## 2.7 LOW — the purity tests do NOT break, and that is also the problem

`tests/engine/test_engine_no_llm.py:133-140` calls `run_lap` twice on lap 6 and asserts
`first.model_dump() == second.model_dump()`. With memory in the caller and the parameter
defaulting to `None`, this passes unchanged — the design's central claim holds, verified by
reading the call, which passes no memory.

The consequence: **no existing test exercises the memory path at all**, and the determinism
test will keep passing however wrong the accumulator is. Whatever ships needs its own test;
the existing suite gives zero coverage by construction. **Severity: LOW**, but it means "the
tests still pass" is not evidence of anything here.

---

# AUDIT 3 — adversarial: what does memory break?

## 3.0 The harness, and its one honest deviation

Everything below runs on Lusail 2025 / NOR, laps 5-45, driven through the `no-llm`
profile per `reference_drive_orchestrator_offline`. It reproduces the documented
result exactly: **deterministic action STAY_OUT on 39 of 41 laps**, `best_mc`
UNDERCUT on laps 35 and 44, deterministic UNDERCUT on lap 44 (Norris's real stop).

**The deviation, stated because it is the kind of thing that flatters a measurement.**
The `no-llm` profile never runs the conditional agents, so `pit_out` is `None` and
`regulation_context` is `""` on every lap (asserted by
`tests/engine/test_engine_no_llm.py:114`). Production runs `rich`, which would populate
them **only when N28/N30 are routed**. Recomputing the routing off the cached sub-agent
outputs with the real `_decide_agents_to_call`:

```
laps where N28/N30 are routed: [33, 35]   ->  {(): 39, ('N28','N30'): 2}
```

So **39 of 41 prompts are shape-identical to production** (both render
`[N28 Pit] not activated` and the "none flagged" regulation line), and **2 diverge**
(laps 33 and 35 lack the pit block and the hard regulation constraint production would
carry). Every conclusion below is therefore sound on 39/41 laps and must not be read as
covering laps 33 and 35.

One side effect works in the audit's favour: with `pit_out=None`, the N28 backfill at
`strategy_orchestrator.py:2006-2011` is inert, so `pit_lap_target` on these laps is
**purely LLM-authored**. That isolates the LLM's contribution, which is exactly the
quantity Audit 1 asks about.

## 3.1 HIGH — the anchoring pressure is ALREADY in the prompt, unconditioned

Before adding memory, `_build_orchestrator_prompt` already tells the model, on **every
lap**, with no condition attached (`src/agents/strategy_orchestrator.py:1557-1564`):

> `Treat a repeated STAY_OUT as CONTINUING a plan rather than making a fresh one, and do`
> `not re-argue the same case from scratch.`

That block is rendered unconditionally — there is no `if` around it in the return
statement at `:1523-1639`. It fires on lap 1, on a lap where the call changes, and on a
Safety Car lap alike.

Two consequences, both sharp:

1. **The A baseline is not a no-anchoring control.** Whatever continuity bias exists is
   already present in today's prompt. A small A-vs-B difference does not mean "memory has
   no effect"; it can equally mean "the effect was already bought by #646".
2. **On the lap the call must change, the prompt is already arguing against changing it**,
   and memory would add a number to that argument. The #646 fix asked for the shape of a
   hold; it did not scope the instruction to laps that are holds.

**Severity: HIGH**, and note this is a finding about the SHIPPED prompt, not only about
the proposal. If the memory block lands, the natural fix is to make `:1557-1564` conditional
on the memory saying the call actually repeated — which is only possible once memory exists.

## 3.2 MEDIUM — an anchored `pit_lap_target` is not cosmetic: it propagates into a deterministic field

`_clamp_expected_stint_end` (`src/agents/strategy_orchestrator.py:1904-1940`) computes
`anchor = pit_lap_target + min(cliff_p50, capacity)` and returns the anchor whenever the
LLM's `expected_stint_end` sits more than 3 laps away (`:1932-1940`).

So `pit_lap_target` is an INPUT to a deterministic clamp. If memory succeeds at its stated
goal — stabilising the target — it stabilises `expected_stint_end` too, which is the win.
If memory instead holds a target that should have moved, the staleness is laundered through
a clamp and reaches the UI looking computed rather than remembered. **Severity: MEDIUM**;
the mitigation is that the memory block must carry the DRIFT as well as the value, which
the proposed design already does, and that the drift must be rendered even when it is zero.

## 3.3 MEDIUM — nothing can retire a contingency, and the design says "live"

The design's third field is "contingencies declared and live — what we promised if X, and
whether X happened". `Contingency.trigger` is free text by schema
(`src/agents/strategy_orchestrator.py:247`, `Field(description="When does this branch
activate?")`), and the prompt asks for prose triggers like `"gap to SAI drops below 0.8 s"`
(`:1632-1633`).

**No code can evaluate that.** There is no parser, no trigger DSL, and no consumer: grepping
the repo, `contingencies` is produced by the LLM and rendered by the UI, never evaluated.
So "live" can only mean "the LLM emitted it again last lap", which makes the field a
one-lap echo rather than a commitment, or "declared at any point this race", which grows
without bound.

The prototype in `scratchpad/memblock.py` implements the bounded reading (last lap only,
so at most 4 lines, since `_LLMSynthesis.contingencies` has `max_length=4` at
`strategy_orchestrator.py:310`). The cumulative reading is quantified in Audit 1 below.
**Severity: MEDIUM** — this is a specification gap, not a code defect: the design must
pick one reading and say so, and the honest one is the bounded echo.

> **Read §1.5 and §3.6 before acting on this section.** The measurements invert its
> assessment: the one-lap echo turned out to be the field that carries the whole effect, and
> "it cannot be retired" matters far less than "without it the model forgets what it planned".

## 3.4 LOW — `laps_held` has no defined value on the first lap, and the spec does not say

The design table gives `last_action` + `laps_held` with no first-lap semantics. Three
readings are all defensible and they say different things to the model: omit the block
(the prototype's choice), `laps_held = 0` ("held for 0 laps", a claim about a call that
does not exist), or `laps_held = 1`.

This repo's scar is a sentinel colliding with a searchable value
(`race_state_manager.py`, the `Position` NaN -> 0 case). The prompt analogue is weaker but
real: the block sits directly above `RACE CONTEXT: ... TyreLife {n}` (`:1566-1568`) and
below guard-rail 4, `"Minimum stint before pit: SOFT >= 8 laps, MEDIUM >= 12, HARD >= 15"`
(`:1542`). A bare integer described as "laps" adjacent to two other integers described as
"laps" is a misattribution the model can make in prose. **Severity: LOW**, but "omit the
block when there is no history" should be written into the spec rather than left to the
implementer.

---

# AUDIT 1 — does the LLM actually use it?

## 1.1 CRITICAL, and it is not about memory: `temperature=0.0` never reaches the model

`OrchestratorCFG` (`src/agents/strategy_orchestrator.py:106`) states:

> `temperature=0.0 ensures deterministic structured output from Layer 3 LLM.`

`_get_orchestrator_llm` builds the client accordingly at `:143`:

```python
llm = ChatOpenAI(model=CFG.model_name, temperature=CFG.temperature, timeout=120, max_retries=1)
```

**It is discarded.** `CFG.model_name` is `"gpt-5.4-mini"` (`:109`), and `langchain_openai`
drops `temperature` for that model family — it does not raise, it sets the attribute to
`None` and omits the key from the request payload. Executed:

```
CFG.temperature = 0.0
client temperature = None                     <- the object _get_orchestrator_llm() returns
direct ChatOpenAI('gpt-5.4-mini', temperature=0.0).temperature = None
payload temperature = <absent>                <- not sent to the API at all
ChatOpenAI('gpt-4.1-mini',  temperature=0.0).temperature = 0.0   <- kept for the sub-agent model
```

So Layer 3 runs at the provider default while the config, the docstring and every
downstream assumption say it is deterministic. The sub-agents, on `gpt-4.1-mini`, do get
temperature 0. **The orchestrator is the one place the setting silently does nothing.**

**Severity: HIGH**, and it is the reason the rest of Audit 1 reads the way it does. It also
partly re-explains #646's finding: the reason the synthesis is rewritten every lap is not
only that the prompt is stateless — **the model is sampling.** A memory block cannot fix
sampling.

This is a shipped defect independent of the memory project and it deserves its own issue.
Note before fixing: it is not obvious that `temperature=0` is even desirable here, but a
config value that is read, passed and silently dropped is wrong either way.

## 1.2 CRITICAL — the noise floor is enormous, and it swallows most of the question

Two identical passes, A and A', over the same 41 prompts, same client, no memory in either.
Any A-vs-B difference at or below these numbers is not measurable.

| field | laps differing (of 41) | of which substantive |
|---|---|---|
| `action` | **1** | 1 |
| `confidence` | **36** | 36 |
| `pit_lap_target` | **23** | 23 |
| `compound_next` | 22 | 22 |
| `expected_stint_end` | 28 | 28 |
| `pace_mode` | 6 | 6 |
| `risk_posture` | 5 | 5 |
| `target_lap_time_s` | 3 | 3 |
| `undercut_target` | 1 | 1 |
| `contingencies` | 41 | 28 |
| `key_risks` | 41 | 17 |
| `reasoning` | 41 | 39 |

`reasoning` similarity between the two identical runs: **median 0.3020, min 0.0421**. The
prose is not a variation on a theme, it is a different paragraph.

**A cross-pass A/B on this configuration can only detect an effect that moves a field on
more than ~23 of 41 laps for `pit_lap_target`, or ~36 of 41 for `confidence`.** Nothing the
memory block plausibly does is that large. That is why the conclusions below rest on
**within-pass** statistics instead, which do not carry the cross-pass noise.

## 1.3 CRITICAL — the one lap that matters is a coin flip BEFORE memory is added

The single `action` disagreement in the noise floor is not a random lap:

```
 lap       det |         A        A2
  44  UNDERCUT |  UNDERCUT  STAY_OUT
```

**Lusail 2025 lap 44 is Norris's real stop and the only significant action change in the
415-lap projection set.** The same prompt, sent twice, produces UNDERCUT once and STAY_OUT
once, while the deterministic Monte Carlo says UNDERCUT both times.

Consequences, and they are the audit's centre of gravity:

1. Any single-run comparison at the decision lap measures the sampler, not the prompt.
   Audit 3's anchoring test therefore has to be run with repeats.
2. **The decision the product exists to make is currently not reproducible.** That is a
   product finding that outranks the memory question: on the lap that matters, the shipped
   `rich` profile is a coin flip on top of a deterministic layer that got it right.

## 1.4 HIGH — `pit_lap_target` is not drifting, it is being resampled

The design's second memory field is "previous `pit_lap_target` + its drift", on the reasoning
that "a target that moves every lap is a plan that does not exist". Measured **within** each
pass, over a 57-lap race:

| pass | lap-to-lap changes | total absolute movement |
|---|---|---|
| A | **35 of 40** | **311 laps** |
| A' | 34 of 38 | 296 laps |

The observed sequence in A: `12, 8, 8, 12, 11, 11, 19, 13, 18, 20, 16, 29, 24, 22, 46, 36,
57, 26, 35, 29, 30, 30, 57, 57, 57, 35, 57, 39, 35, 39, 35, 37, 41, 40, 46, 57, 44, 45, 46,
44, 57`. On lap 21 the plan is to stop on lap 57; on lap 22 it is lap 26.

Two readings, and only one supports building memory:

- *Memory would stabilise this.* Plausible in principle: telling the model its last target
  was 26 gives it an anchor it currently lacks.
- *There is nothing to stabilise, because this is sampling noise, not drift.* Supported by
  1.1 and 1.2: with `pit_lap_target` differing on 23 of 41 laps between two IDENTICAL runs,
  most of that 311-lap movement is the sampler, and an anchor in the prompt competes against
  a temperature the config believes is zero.

**The correct order is therefore: fix 1.1 first, re-measure the within-pass movement, and
only then decide whether memory has a problem left to solve.** Building memory now risks
attributing to it an improvement that a one-line client fix would have produced.

> Partly answered by §1.7, which was measured after this section was written: memory removes
> **28-31%** of that movement, against 5% variation between two identical runs. So there is a
> real effect on top of the sampler — the open question is how much of the remaining 214 laps
> of movement survives once temperature is actually applied.

## 1.5 THE STRONGEST RESULT IN THIS AUDIT — without memory the contingency plan is reinvented every lap; with memory it converges

> **Correction.** An earlier draft of this section reported "81 distinct triggers in pass B"
> and concluded the block would grow without bound. That number came from pass **A'**, not B —
> the analysis had been run before B existed, with A' in B's argument slot. The real
> comparison is below and it points the opposite way. Recording the slip because it is
> exactly the failure mode in `feedback_my_blind_spots_directive` §3: the wrong number arrived
> in the direction I already expected, so I did not check where it came from.

Distinct contingency triggers, over the same 41 laps, counted per pass:

| pass | distinct triggers | total declarations | reuse ratio |
|---|---|---|---|
| A (no memory) | **80** | 132 | 1.65 |
| A' (no memory, repeat) | **81** | 129 | 1.59 |
| B (memory) | **6** | 142 | **23.67** |

The statistic is almost perfectly stable under repetition — 80 vs 81 between two identical
runs — so the drop to 6 is not noise by any reading. Growth in B:
`L5:3 → L13:4 → L21:4 → L29:6 → L37:6 → L45:6`.

**Without memory the orchestrator declares roughly two brand-new contingencies per lap and
almost never repeats one.** Across a race that is 80 different plan-Bs, none of which survives
into the next lap. It is not a plan, it is 41 unrelated plans. With its own previous
contingencies echoed back, the model settles on six and keeps them.

This is the design's premise, confirmed with the largest effect margin in the audit, and it is
also what makes §3.6 possible: a contingency can only fire if it is still there when its
trigger arrives.

It also disposes of the prompt-growth worry in the direction that favours the design: under
memory the cumulative reading would be **6 lines by lap 45**, not 81. The bounded reading
(last lap only, capped at 4 by `_LLMSynthesis.contingencies` `max_length=4`,
`strategy_orchestrator.py:310`) remains the safer specification, because the 6-line result is
self-referential — it is caused by the echo it would feed.

**Severity: n/a — this is the finding that argues FOR building it.**

---

## 1.6 The headline A/B: over a full race, at single-run resolution, memory changes NOTHING measurable

Pass B is a full 41-lap sequential run where the memory block is built from B's own previous
recommendations. Substantive differences, against the A-vs-A' floor:

| field | noise (A vs A') | signal (A vs B) | delta |
|---|---|---|---|
| `action` | 1 | **0** | -1 |
| `confidence` | 36 | 33 | -3 |
| `pit_lap_target` | 23 | 22 | -1 |
| `compound_next` | 22 | 20 | -2 |
| `undercut_target` | 1 | 1 | 0 |
| `pace_mode` | 6 | 8 | +2 |
| `target_lap_time_s` | 3 | **9** | +6 |
| `risk_posture` | 5 | 2 | -3 |
| `expected_stint_end` | 28 | 30 | +2 |
| `contingencies` | 28 | 29 | +1 |
| `key_risks` | 17 | 18 | +1 |
| `reasoning` | 39 | 39 | 0 |

`reasoning` similarity A-vs-B: median 0.2668 (floor 0.3020) — B's prose is no further from A
than A' is.

**The null hypothesis the audit was told to treat as the favourite survives this test.**
Ten of twelve deltas are within +-3 of zero and several are negative, which is only possible
if the quantity being measured is noise. The one exception, `target_lap_time_s` at +6, sits on
a field that moved 3 times in the floor, so it is a small count on a rarely-populated field
and is not enough to carry a project.

Cost of the block, measured: B used **134,379 prompt tokens vs A's 127,747** over the same 41
laps, i.e. **+162 tokens per lap, +5.2%**.

**If the audit had stopped here it would have said: do not build it.** It did not stop here,
because a per-lap field diff is the wrong instrument twice over: it cannot see an effect that
lives in the RELATION between consecutive laps (§1.7), and it averages away the handful of
laps where the call is actually in play (§3.5, §3.6).

## 1.7 The memory effect is real and it is BETWEEN laps, not within one

Both statistics below are computed **within** a single pass, so they carry none of the
cross-pass sampler noise that swamps §1.6. A vs A' gives the natural variation.

**`pit_lap_target` movement** over the 41 laps:

| pass | lap-to-lap changes | total absolute movement |
|---|---|---|
| A | 35 / 40 | **311 laps** |
| A' | 34 / 38 | **296 laps** |
| B | **30 / 40** | **214 laps** |

Two identical runs differ by 15 laps of movement (5%); adding memory removes 82-97 (**28-31%**).
That is roughly six times the only same-condition variation available. It is still n=1 per
condition, so treat it as a strong indication rather than a measurement — but it points the
same way as §1.5, which has a far larger margin.

**Confidence does NOT inflate with `laps_held`** — the anchoring tell the audit brief asked
for specifically:

```
corr(laps_held, confidence_B) = -0.151     (n=40)
corr(laps_held, confidence_A) = -0.118     baseline: A has no memory, so this is the lap trend
mean confidence   A = 0.919   B = 0.908
```

Both correlations are small, negative, and within 0.03 of each other. **Refuted: memory does
not make the model more certain the longer it holds.** If anything B is marginally less
confident. This was one of the two hypotheses most likely to kill the design and it does not
hold.

---

## AUDIT 3 (continued) — the executed anchoring experiments

Both experiments freeze the memory block the accumulator would hold entering the target lap
(reconstructed by the same prototype code from pass A's recommendations) and re-run that one
lap N times under three prompts: **A** (no memory), **B** (memory as proposed), **BC**
(memory + one counterweight sentence: *"This history is context, NOT a commitment. Judge this
lap on its own evidence; a long hold is not itself a reason to keep holding."*).

The counterweight is **not** part of the proposed design. It is in the experiment because
§3.1 predicted the anchoring failure and a gate that only measures the proposal cannot tell
you what to do about it.

## 3.5 HIGH — the anchoring effect is real in direction, and the counterweight reverses it

> ## ⛔ SUPERSEDED 2026-07-28 — the headline number here did NOT replicate
>
> **Do not quote the "10/10 with counterweight" result below.** It came from n=10. Re-run on
> the same shipped model (`gpt-5.4-mini`) at **n=50 per arm**, in two independent batches of 25
> that agree with each other:
>
> | arm | takes the stop |
> |---|---|
> | no memory | **35/50 = 70 %** |
> | memory, exactly as it ships today (counterweight included) | **28/50 = 56 %** |
>
> Fisher two-sided **p = 0.2137** · difference **-14 pp** · 95 % CI **[-32.7, +4.7] pp**.
>
> What that changes:
>
> 1. **The counterweight does not produce 10/10.** At n=50 the same configuration gives 56 %,
>    and the interval excludes anything above roughly 61 %. The original figure was
>    small-sample luck — the flattering-number failure mode this repo has a standing directive
>    about.
> 2. **Memory does NOT help on the green-flag decision lap.** The CI's upper bound is +4.7 pp,
>    so a benefit there is essentially excluded. The layer's demonstrated value is §3.6's
>    contingency echo firing under a Safety Car, not better stop calls in general.
> 3. **Anchoring remains unproven AND unrefuted.** The point estimate is a 14-point harm and
>    both batches lean the same way, but p=0.21. Settling it at p<0.05 needs n≈190 per arm
>    (≈380 calls); worth spending only if the decision lap becomes load-bearing.
>
> Raw data: `data/eval/prompt_ab/anchor44_shipped{,_b}.json` (gitignored; regenerate with
> `python -m scripts.prompt_ab.run_repeats --lap 44 --repeats 25`, no `--model`).
>
> The section below is kept as the original record. Its *direction* survived; its magnitude did not.

**Lusail 2025, lap 44** — Norris's real stop, deterministic MC = `UNDERCUT`, n=10 per variant.

| variant | pits (UNDERCUT) | stays out | mean confidence | `pit_lap_target` spread |
|---|---|---|---|---|
| A (no memory) | **6/10** | 4 | 0.757 | 44,44,44,44,45,45,48,48,48,49 |
| B (memory) | **4/10** | 6 | 0.795 | 44,44,44,45,46,46,46,46,46,46 |
| BC (memory + counterweight) | **10/10** | 0 | 0.703 | 44 x7, 45 x3 |

Fisher two-sided:

```
A(6/10) vs B(4/10)    p = 0.6563     <- direction matches anchoring, NOT significant at n=10
A(6/10) vs BC(10/10)  p = 0.0867
B(4/10) vs BC(10/10)  p = 0.0108     <- the counterweight is a real effect
```

Read honestly:

- **The anchoring hypothesis is directionally supported and statistically unproven.** Memory
  moved the stop rate from 60% to 40% on the lap where stopping was right, but n=10 cannot
  separate that from the sampler documented in §1.2/§1.3. Anyone who reports "memory delays
  the stop" off this alone is over-reading it.
- **The counterweight sentence is the significant result.** With it, the model agrees with the
  deterministic layer **10 times out of 10** on the lap that matters, against 6/10 with no
  memory at all, and its `pit_lap_target` collapses onto lap 44-45 instead of scattering to 49.
- Note the hold was only **8 laps** ("since lap 36" — pass A had an UNDERCUT at lap 35 that
  reset the run), not the 39-lap hold the design imagines. The effect appeared at a modest
  hold; a longer one was not tested.

**And the effect is invisible in the prose.** Of B's 10 runs, only **2 mention the hold,
continuity or "since lap"** at all; the six STAY_OUT reasonings argue entirely from tyre cliff
and pace delta. So memory shifts the decision without leaving a trace a reviewer could catch
in the reasoning field. **Severity: HIGH** — an effect you cannot see in the output is an
effect you cannot debug in production.

## 3.6 HIGH — under a Safety Car memory HELPS, strongly, and the mechanism is the field §3.3 called the weakest

**Lusail 2025, lap 42, Safety Car injected via RCM** (the CLI's own mechanism,
`RaceControlStateTracker`, mirroring `tests/engine/test_engine_no_llm.py:124`). N27 responds
correctly: `overtake_prob = 0.00`, `sc_prob_3lap = 1.0`, and the deterministic MC flips to
`PIT_NOW` on all five injected laps. n=8 per variant.

| variant | takes the stop (PIT_NOW) | stays out | mean confidence |
|---|---|---|---|
| A (no memory) | **1/8** | 7 | 0.880 |
| B (memory) | **7/8** | 1 | 0.930 |
| BC (memory + counterweight) | 4/8 | 4 | 0.891 |

```
A(1/8) vs B(7/8)   p = 0.0101   <- significant
A(1/8) vs BC(4/8)  p = 0.2821
B(7/8) vs BC(4/8)  p = 0.2821
```

**The mechanism is legible, and it is not `laps_held`.** The frozen block entering lap 42 was:

```
DECISION MEMORY (your own previous calls this race):
  Last call: STAY_OUT, held for 6 consecutive laps (since lap 36).
  pit_lap_target over your last 5 calls: 41, 40, 46, 57, 44 (net drift +3 laps).
  Contingencies you declared and have not retired:
    - [HIGH] since lap 41: "SC deployed within the next 3 laps" -> PIT_NOW
    - [MEDIUM] since lap 41: "Gap to the car ahead drops below 1.2 s ..." -> PIT_NOW
    - [MEDIUM] since lap 41: "Tyre life falls below 10 laps ..." -> PIT_NOW
```

The model had declared, one lap earlier, that a Safety Car within three laps means PIT_NOW.
The Safety Car then deployed. **Without memory it forgot its own plan on 7 of 8 runs; with
memory it executed it on 7 of 8.**

That inverts §3.3's assessment. The contingency echo — the field with no evaluator, no
retirement and no downstream consumer — is the one that produced a significant effect.
`last_action`/`laps_held`, the field the design leads with, is not shown to do anything by any
experiment in this audit.

**Two caveats, both mine to state:**

1. **These five SC laps DO route N28/N30** (verified: `sc_prob_3lap = 1.0` routes both on all
   five), so unlike the 39 green laps, these prompts are **missing the pit block and the hard
   regulation constraint production would carry**. The A-vs-B contrast is sound because both
   sides get the identical prompt; the **absolute** rate is not. In particular, **A's 1/8 must
   NOT be read as "the product ignores Safety Cars"** — that would need a `rich`-profile run.
2. The counterweight damps the memory in both directions: it rescued lap 44 and cost half the
   SC gain. Summed over both experiments, agreement with the deterministic layer is
   **A 7/18, B 11/18, BC 14/18**.

## 3.7 Incidental, outside the memory scope

- `mc_decision_margin` (`strategy_orchestrator.py:937`, added by #645) has **no production
  consumer** — repo-wide it appears only in its own definition and
  `tests/mc/test_mc_is_a_real_decision.py`. It is the natural companion to a memory block
  ("how close was this call") and is already computed.
- The `"ERROR"` action default the MONITOR audit found in two places
  (`src/arcade/strategy.py:807`, `simulator.py:523`) has a **third instance** at
  `simulator.py:502`, in the dict branch of the same function.
- `_get_orchestrator_llm` sets `max_retries=1, timeout=120` (`:143`). A 41-lap measurement pass
  died at lap 22 on `openai.APITimeoutError`. Not a memory finding; noted because a live race
  surface has the same exposure.

---

# GATE FOLLOW-UP, same day — measured, and it changes two of the conclusions

The verdict below says: fix the dropped temperature first, then re-measure, because some
of what memory appears to fix may be the sampler. That was done. Víctor decided **not** to
change the production model (§1.1 stands: keep `gpt-5.4-mini`, warn, document), so the
question was answered the only way left open - by measuring a model that DOES honour
`temperature=0`, without touching production.

**Method.** Same 41 cached Lusail laps, same harness (now at `scripts/prompt_ab/`), three
passes on **`gpt-4.1-mini`** at `temperature=0.0`: two identical no-memory passes for the
floor, one with the memory block. 123 calls.

**Caveat, stated first because it bounds everything below: this is a CROSS-MODEL
comparison.** `gpt-4.1-mini` is a different and weaker model, so its absolute numbers are
not comparable to `gpt-5.4-mini`'s. What it can answer is directional: does the behaviour
survive when sampling is removed?

## Result 1 — most of the per-field noise WAS the sampler

| field, laps differing between two identical passes | `gpt-5.4-mini` (sampling) | `gpt-4.1-mini` (temp 0) |
|---|---|---|
| `action` | 1 | **0** |
| `confidence` | 36 | **5** |
| `pit_lap_target` | 23 | **11** |
| `expected_stint_end` | 28 | 13 |
| `reasoning` median similarity | 0.3020 | **0.5967** |

So §1.1 is not a pedantic config complaint. The dropped parameter is a large part of why
the orchestrator's output churns, and the coin flip on the decision lap (§1.3) disappears:
`action` differed on **0 of 41** laps between two identical deterministic passes.

## Result 2 — the contingency reinvention is NOT a sampling artifact, and this is what justifies the layer

| pass | distinct triggers | declarations | reuse ratio | `pit_lap_target` changes |
|---|---|---|---|---|
| no memory | **28** | 157 | 5.6 | 32/35 (**91%**) |
| no memory, repeat | **26** | 159 | 6.1 | 36/39 (**92%**) |
| memory | **5** | 164 | **32.8** | 24/40 (**60%**) |

Two identical passes land on 28 and 26, so the statistic is stable to about +-2. Memory
takes it to **5**. On a fully deterministic client the orchestrator still invents roughly
27 different plan-Bs across one race and reuses each about six times; with its own
contingencies echoed back it settles on five and reuses each thirty-three times.

**The main justification for the memory layer survives the gate.** It was never mostly
about sampling.

## Result 3 — the `pit_lap_target` claim does NOT survive, and §1.7 must be read down

§1.7 reported memory cutting total target movement 311 -> 214 laps. On the deterministic
client the floor itself is 231 and 293 (spread 62), and memory gives **225** - inside the
floor's own spread. **Total movement is not shown to improve once sampling is removed.**

What does survive is the change RATE: the target moves on 91% and 92% of lap pairs without
memory and **60%** with it, far outside the floor's 1-point spread. So memory makes the plan
hold still more often; it does not make the moves smaller. Say it that way, not the other.

## What this does to the plan

- Step 1 (fix the dropped temperature) is **more** justified, not less: it is worth several
  points of output stability on its own.
- Step 4's field ordering is **confirmed**: contingencies first (survives the gate),
  `pit_lap_target` second but with the weaker claim above, `laps_held` still unproven.
- Nothing here re-opens §3.5/§3.6, which were run on the shipped model with repeats. They
  should be re-run on a deterministic client before the wiring PR, since both were
  measured against a much noisier baseline than a fixed client would give.

---

# VERDICT

**Build it — with changes, and not first.**

Of the three options on the table:

- **Build it as designed — NO.** The design leads with `last_action`/`laps_held`, which no
  experiment here shows doing anything, and specifies no counterweight, which is the one
  intervention that measurably improved the decision lap. Built as written it would ship the
  unproven field and omit the proven mitigation.
- **Do not build it — NO.** This was the favourite going in and the whole-race field diff
  (§1.6) supports it, but two measurements refute it decisively. Without memory the
  orchestrator declares **80 distinct contingencies in 41 laps and reuses almost none**
  (§1.5); with memory, **6**. And when a declared trigger actually fired, memory was the
  difference between executing the plan on 7 of 8 runs and forgetting it on 7 of 8 (§3.6,
  p=0.010).
- **Build it with changes — YES**, in the order below.

**What memory actually does, stated precisely:** it does not change what the orchestrator
decides on a given lap (`action` differed on **0 of 41** laps, §1.6). It changes whether
consecutive laps are the same plan. That is a smaller claim than the design implies and a more
useful one, because plan incoherence is the thing a user perceives.

**The gate on all of it:** §1.1. `temperature=0.0` is read from config, passed to the client
and **silently discarded** by `langchain_openai` for `gpt-5.4-mini`. Layer 3 has been sampling
at the provider default this whole time. Some of what memory appears to fix — the 311 laps of
`pit_lap_target` movement, the 80 one-shot contingencies, the coin flip on lap 44 — may be the
sampler, and a one-line client fix is cheaper than a shared contract across three surfaces.
**Fix that first and re-run this harness before writing the accumulator.** The harness exists
now; a re-measurement costs ~120 calls and no new code.

## Numbered plan, ordered by value and risk

1. **Fix the dropped temperature** (`strategy_orchestrator.py:143`, §1.1). Its own issue,
   independent of memory. Decide deliberately whether Layer 3 should be deterministic; either
   way, a config value that is read and discarded is wrong. Correct the docstring at `:106`,
   which asserts the opposite of what happens. *Highest value, lowest risk, touches one line.*
2. **Re-run this audit's harness after step 1** (`scratchpad/` scripts, or re-create from
   `reference_drive_orchestrator_offline`). Re-measure §1.5's 80-vs-6 and §1.7's 311-vs-214. If
   determinism alone collapses them, the memory project is much smaller than it looks. *No
   production code; ~120 API calls.*
3. **Extend `tests/engine/test_engine_threads_every_argument.py` to `_build_orchestrator_prompt`**
   (§2.2). Parametrise `_kwargs_passed_by` over the callee name and compare
   `engine._run_rich` against `orch.run_strategy_orchestrator_from_state` for BOTH callees.
   Five lines, and it is the guard that stops the memory argument reaching one prompt site and
   not the other two. *Do this BEFORE the accumulator exists, not after.*
4. **Build the accumulator, contingency-echo first.** One class in
   `src/strategy/inference/`, owned by the caller, `run_lap` gaining one optional parameter
   defaulting to `None` — the structure is sound and §2.7 confirms the purity tests hold.
   Order the fields by the evidence, not by the design table:
   - **`contingencies`** — last lap only, max 4 (§1.5, §3.6). This is the load-bearing field.
   - **`pit_lap_target` + drift** (§1.7). Record it from the `StrategyRecommendation`, never
     from a surface DTO: `simulator.py:509` and `:529` mean the field has two different
     sources depending on profile.
   - **`last_action` + `laps_held`** — include it, but as the field with the least evidence.
     Render the SPAN (`"STAY_OUT since lap 36, 8 of the last 9 laps"`), never a bare count,
     because every surface skips laps (§2.4, demonstrated: "held for 7 consecutive laps" when
     16 had elapsed). Omit the whole block when there is no history (§3.4).
5. **Ship the counterweight sentence as part of the block, not as an option** (§3.5). With
   memory alone the model matched the deterministic layer on 4 of 10 runs at the real stop lap;
   with the counterweight, 10 of 10 (p=0.011 against memory-alone). Across both experiments:
   A 7/18, B 11/18, **BC 14/18**.
6. **Scope the #646 STAY_OUT framing to actual holds** (§3.1). Today
   `strategy_orchestrator.py:1557-1564` tells the model to "treat a repeated STAY_OUT as
   CONTINUING a plan" on *every* lap, including lap 1 and including the lap the call changes.
   Once memory exists that block can finally be conditional, which is the correct fix and is
   only possible after step 4.
7. **Write the limitation down** (§2.3). `/recommend`, the MCP tool and therefore the webapp
   Strategy tab bypass `run_lap` and cannot have memory under this design. Say so in the
   accumulator docstring and in the issue. Do not let a second `_rcm_events_for_lap`-shaped
   twin grow on the endpoint side (§2.6).
8. **Give it its own test.** `tests/engine/test_engine_no_llm.py:133-140` will keep passing
   whatever the accumulator does, because the parameter defaults to `None` (§2.7). The existing
   suite provides zero coverage by construction.

Steps 1-3 are worth doing whether or not the memory layer is ever built.

## What I tried to break and could NOT

Stated so the next reader knows what does not need re-auditing.

- **`run_lap`'s purity.** I expected the optional-parameter claim to be the weak point. It is
  not: `tests/engine/test_engine_no_llm.py:133-140` calls `run_lap` twice on lap 6 and asserts
  identical output, and with memory in the caller and the default `None` it passes untouched.
  The design's central structural claim holds. (Its cost — that no test then exercises the
  memory path — is §2.7, not a refutation.)
- **"Memory has to live in the engine."** It does not. Three of the four `run_lap` call sites
  already own a race-scoped object, and `RaceControlStateTracker` (`src/nlp/rcm_state.py:83`)
  is this exact pattern already working across the same three surfaces.
- **Confidence inflation with `laps_held`.** The brief flagged this as an anchoring tell.
  Measured over 40 laps: `corr = -0.151` with memory against `-0.118` without. Not there.
- **A `laps_held` sentinel collision.** I looked for a value the code also searches for, in the
  repo's own scar shape. There is none: `laps_held` would be prose in a prompt, with no
  dataclass field and no consumer comparing against it. The prompt-adjacency risk (§3.4) is
  real but I read all 71 `reasoning` strings from the repeat experiments and found **no case**
  of the model misattributing the hold count to tyre life or a stint minimum.
- **Prompt-size blowup.** Measured, not estimated: **+162 tokens per lap, +5.2%** (134,379 vs
  127,747 over 41 laps). And under memory the cumulative contingency reading would be 6 lines,
  not the 81 I first feared (§1.5).
- **"Memory changes the action."** I could not find a single case in a full race: `action`
  differed on **0 of 41** laps between A and B. Whatever memory does, it does not do it by
  flipping the primary decision on ordinary laps.
- **"The anchoring hypothesis is proven."** I could not prove it, and a later, much larger run
  still could not: at n=50 per arm on the shipped model the difference is 70 % vs 56 %,
  p=0.2137. The direction has now leaned the same way in three separate batches, so it is not
  refuted either. Anyone quoting this audit as evidence that memory delays stops is
  over-reading it. (The claim that "the counterweight fixes whatever is there" was in this
  bullet and is **withdrawn** — see the superseded banner on §3.5.)
- **The `no-llm` harness as a stand-in for production.** I tried to invalidate my own method
  and got a bounded answer instead: 39 of 41 green laps route no conditional agents at all, so
  those prompts are shape-identical to `rich`. The two exceptions (laps 33, 35) and all five
  SC laps are NOT, and every conclusion drawn on them is marked as such in §3.0 and §3.6.

## What was NOT verified

- **The `rich` profile end to end.** Everything here builds the prompt from `no-llm` sub-agent
  outputs. The two green laps that route N28/N30, and the whole Safety Car experiment, would
  carry a pit block and a hard regulation constraint in production that these prompts lack.
  **A's 1/8 stop rate under a Safety Car (§3.6) must not be quoted as a product finding** — it
  needs a `rich` run to mean anything.
- **One circuit, one driver, one race.** Lusail 2025 / NOR. The 415-lap projection set spans
  7 driver-races; none of the others was re-run here.
- **Long holds.** The anchoring test ran at an 8-lap hold, not the 39-lap hold the design
  imagines, because pass A's UNDERCUT at lap 35 reset the run.
- **`last_action`/`laps_held` in isolation.** Every B prompt carried all three fields, so the
  audit cannot attribute the effect to one of them except by the mechanism visible in §3.6,
  which points at the contingencies.


---

# GATE FOLLOW-UP 2, 2026-07-28 — the two repeat experiments, re-run on a deterministic client

The first gate follow-up re-measured the whole-race passes and explicitly left this open:

> Nothing here re-opens §3.5/§3.6, which were run on the shipped model with repeats. They
> should be re-run on a deterministic client before the wiring PR, since both were measured
> against a much noisier baseline than a fixed client would give.

That is what this section does. It is the gate on Sprint 2's wiring: if memory stopped
helping once the sampler was removed, nothing was to be wired.

**Method.** Identical to §3.5/§3.6 except for the client: `--model gpt-4.1-mini` at
`temperature=0.0`, which that family keeps. History for both experiments is `gate_none.json`,
the deterministic no-memory pass from the first gate follow-up, so the block echoed back is
the one a deterministic surface would actually have accumulated. 36 calls, 116k in / 13.5k out.

Two variants, not three: the counterweight now ships **inside** `DecisionMemory.block()`, so
what §3.5 called BC is what `memory` means from here on. There is no memory-without-
counterweight configuration left to measure, by design.

## Result 1 — the Safety Car experiment does NOT just survive, it sharpens

**Lusail 2025 lap 42, Safety Car injected via RCM, deterministic MC = `PIT_NOW`, n=8.**

| variant | takes the free stop | stays out | mean confidence | `pit_lap_target` |
|---|---|---|---|---|
| no memory | **0/8** | 8 | 0.856 | 57, 47, 57, 57, 50, 50, 57, 57 |
| memory | **8/8** | 0 | 0.906 | 42 x8 |

```
Fisher two-sided, agreement with the deterministic layer: p = 0.000155
```

On the shipped sampling client this was 1/8 against 7/8 (p=0.0101). On a deterministic client
it is total separation. **The load-bearing result is not a sampling artifact** — removing the
sampler made it cleaner, which is the opposite of what a noise explanation predicts.

The mechanism is unchanged and still legible: entering lap 42 the block carried
`[HIGH] "SC deployed within 3 laps" -> PIT_NOW`, declared by the model itself on lap 41. The
Safety Car then deployed. Without the echo the model does not act on its own plan; with it,
every run does.

**§3.5's severity finding also reproduces, and it got worse.** Of the 8 memory runs, **0**
mention the prior plan, the contingency or continuity in `reasoning` — they argue from tyre
cliff and pace delta and then pit. The one run that does reference a prior plan is in the
**no-memory** arm. So the block changes the decision on 8 of 8 runs while leaving no trace a
reviewer could find in the output. That is a monitoring requirement, not a reason not to ship:
**whatever surface renders the recommendation cannot show WHY this flipped**, so the memory
block itself has to be inspectable when a call is questioned.

The §3.6 caveat stands verbatim and is not weakened by the sharper number: these SC laps route
N28/N30 in production, so the prompts lack the pit block and the hard regulation constraint a
`rich` run would carry. **The A-vs-B contrast is sound because both arms get the identical
prompt; the absolute 0/8 is not a product finding** and must not be quoted as "the product
ignores Safety Cars".

## Result 2 — the anchoring experiment went DEGENERATE, and that is a limitation, not a clean bill

**Lusail 2025 lap 44, Norris's real stop, deterministic MC = `UNDERCUT`, n=10.**

| variant | agrees with MC (pits) | stays out | mean confidence | `pit_lap_target` |
|---|---|---|---|---|
| no memory | **0/10** | 10 | 0.860 | 57 x8, 50, 52 |
| memory | **0/10** | 10 | 0.880 | 57 x10 |

```
Fisher two-sided: p = 1.0
```

**Both arms are on the floor, so this experiment measured nothing.** `gpt-4.1-mini` never takes
the lap-44 undercut in any configuration; there is no variance for memory to move in either
direction. Read precisely:

- It does **not** show that memory is harmless at the decision lap. §3.5's 6/10 → 4/10 was
  measured on a client where the baseline pitted at all; here the baseline never does, so the
  harm this experiment exists to detect is undetectable by construction.
- It does **not** reproduce §3.5's 10/10 counterweight result either.
- What it does show is that the block did not push a 0/10 baseline into a *wrong* action, and
  that the target stopped scattering (57 x8, 50, 52 → 57 x10).

**Resolved 2026-07-28** by re-running the experiment on the shipped `gpt-5.4-mini`, where the
baseline is NOT degenerate, at n=50 per arm: **70 % without memory vs 56 % with**, p=0.2137,
95 % CI [-32.7, +4.7] pp. So the decision lap is settled as far as it is worth settling —
memory does not help there, may cost up to a third of the stops, and the 10/10 figure is
withdrawn. See the superseded banner at the top of §3.5.

Worth recording because it is the design's own worst case: this run's block reported
**`STAY_OUT, held since lap 5 (39 laps)`** — the 39-lap hold §3.5 said it could not test,
since that pass never left `STAY_OUT`. A 39-lap hold plus a counterweight did not produce a
different call from no memory at all.

**Residual noise at `temperature=0.0`:** the no-memory arms are not constant. Lap 44 gave
confidences of 0.85/0.90 and targets 57/50/52; lap 42 gave 57/47/50. A kept temperature
narrows sampling, it does not remove it — exactly as `OrchestratorCFG`'s docstring warns.

## Gate verdict: PASS — wire it

The gate was "if a deterministic client stops the memory improving agreement with the
deterministic MC, or worsens it, stop". Neither happened: agreement is unchanged on the
degenerate lap and goes 0/8 → 8/8 on the lap where the deterministic layer had a live call.

Two things the wiring must carry out of this section:

1. **The contingency echo is the field to ship**, confirmed twice on two clients. Field order
   in step 4 of the plan is correct as written.
2. **The block must be inspectable at the surface.** An intervention that flips 8 of 8
   decisions without appearing in `reasoning` is one nobody can debug from the output alone.

---

# SPRINT 2 OUTCOME, 2026-07-28 — what was built, and what the build changed about the report

The numbered plan above is now executed. This section records what shipped and, more
usefully, the three places where building it produced a finding the audit did not have.

| plan step | issue | PR | state |
|---|---|---|---|
| 2, re-run the harness after the temperature fix | — | #679 | done, GATE FOLLOW-UP 2 above |
| 4, the accumulator | #672 | #676 | shipped in Sprint 1, inert |
| the prompt seam | #680 | #682 | done |
| 7, write the limitation down | #681 | #683 | done |
| the three-surface wiring | #684 | #686 + submodule #204 | done |
| 6, scope the STAY_OUT framing | #685 | #687 | done |

## Three findings the wiring produced

**1. An anti-drift guard in this repo was asserting nothing.**
`test_the_docstring_does_not_name_a_test_that_does_not_exist` scans the engine docstring's
anti-drift section and asserts each test file it names exists. Its pattern was
`tests/test_[a-z_]+\.py`, and every guard the docstring names now lives in `tests/engine/`.
It matched **zero files** and passed green — the same defect it was written to catch, one
level up. Widened, plus a non-empty assertion so a pattern that silently stops matching
fails instead of going quiet. Worth generalising: a guard whose subject can move needs an
assertion that it still has a subject.

**2. The memory path cannot be tested end to end on the `rich` profile, for a structural
reason.** §2.7 predicted the purity test would give the memory path zero coverage, which is
right, but the fix is not simply "write an integration test". Driving `run_lap(profile="rich")`
in a test fails on a connection error long before the prompt exists, because the always-on
sub-agents build LLM clients first. The chain is verified as three links instead — two at
runtime, one static. Anyone adding to this path should expect the same wall.

**3. The block rendered `(1 laps)`.** Cosmetic, but it appeared in the first held lap of
every real run and none of the ten hermetic tests caught it, because they all record
several laps before asserting. Fixed under #685.

## What §3.1's fix actually looks like, and what it costs

Step 6 said the continuation framing "can finally be conditional". It was implemented by
moving the sentence out of the static prompt and into the memory block, rather than by
adding a second prompt parameter — two arguments that must always travel together are two
arguments that eventually do not.

**Two consequences, stated because neither is measured:**

- The block that GATE FOLLOW-UP 2 measured had **no continuation line**. The 0/8 → 8/8
  Safety Car result was obtained on a slightly different artifact than the one that now
  ships. The direction is not in doubt; the exact number was not re-taken.
- `/recommend` and the MCP tool now lose the sentence entirely. That is correct — they have
  no history, so the instruction was unconditioned there in the strongest sense — but it is
  a behavioural change to the shipped prompt on two surfaces and nobody has measured it.

## Verified on real runs, not only in tests

Lusail 2025 / NOR, `rich` profile, prompts captured as sent to the provider. Lap 20 (the
first lap decided) carries no block at all; lap 21 reports `held since lap 20 (1 lap)` with
the counterweight and no continuation; laps 22+ carry both. The span is measured from the
first lap the surface actually decided, not from lap 1, which is the `laps_held` honesty
requirement in §2.4 holding up outside a unit test.

## Still open from this report

- **§3.7's three incidentals** are untouched: `mc_decision_margin` still has no production
  consumer, the third `"ERROR"` default at `simulator.py:502`, and `max_retries=1` on a live
  race surface.
- **§3.3 stands**: nothing retires a contingency. The mitigation shipped is that only the
  last lap's list is echoed, which bounds the block rather than solving the problem.
- **The measurement is still one circuit, one driver, one race.** Nothing in Sprint 2
  widened it.

---

# NEGATIVE RESULT, 2026-07-28 — asking the model to report its own continuity DESTROYS the effect

§3.5 rated this HIGH and it is the one loose end Sprint 2 left: the block changes the
decision and leaves **no trace in `reasoning`**. Under a Safety Car it flipped the call on
8 of 8 runs and 0 of those 8 mentioned the prior plan. An effect you cannot see in the
output is an effect you cannot debug in production, so the obvious follow-up (#691) was to
ask for it: have `reasoning` open with `CONTINUING` or `DEPARTING`.

**It does not work, and the failure is not subtle.** Three formulations, all measured on the
same cached inputs, same deterministic client, n=8 on the Safety Car lap where the memory
effect actually lives:

| condition | takes the free stop | carries a label | says `DEPARTING` |
|---|---|---|---|
| **no instruction (shipped)** | **8/8** | 0/8 | 0/8 |
| v1, label first | 6/8 | 2/8 | 0/8 |
| v2, label first + worked examples | **0/8** | 8/8 | 0/8 |
| v3, label LAST, explicitly after deciding | **0/8** | 7/8 | 0/8 |

`8/8 → 0/8`, Fisher p = 0.000155.

**The relationship is monotone in the wrong direction: the better the model complies with the
labelling, the less it executes its own plan.** At full compliance (v2, v3) the Safety Car
result — the entire justification for the memory layer — is gone.

## Two things this rules out

**It is not an ordering effect.** The first hypothesis was that classifying the call before
reasoning forces a premature commitment to continuity. v3 tested exactly that: decide first,
label as the last sentence, with "classify the decision you reached; do not let it steer the
decision" stated outright. Identical damage. So it is not *when* the model is asked, it is
*that* it is asked.

**It is not a wording problem.** Three formulations spanning terse, exemplified and
explicitly-guarded all produced the same outcome. Continuing to tune the sentence would be
tuning against a mechanism that is not in the sentence.

## What it says about how the block works

`DEPARTING` was never emitted once — **0 of 65 calls** across every condition and both
caches, including 41 green laps and 24 Safety Car runs. The model will label a call it is
continuing and simply omits the label on a call it is changing (v1 shows this cleanly: the 2
labels are on the 2 STAY_OUT runs, the 6 departures carry none).

Read together with the decision damage, the working hypothesis is that **the echo does not
operate through anything the model can introspect.** Asking it to describe its relationship
to the history converts the block from context into a question about consistency, and a model
asked whether it is being consistent answers yes — by being consistent, which on the Safety
Car lap means not stopping.

That also explains §3.5's original observation rather than merely restating it: the 8 runs
that flipped the call did not mention the plan **because the mechanism is not available to
them as a reason.**

## Consequence for the product

**Observability of this layer cannot be bought from the model.** #691 is closed unshipped.
If a surface needs to explain why a recommendation changed, it must render **the block
itself** next to the recommendation — the input, which is deterministic and already
available — rather than ask the LLM to narrate it. That is a UI change, not a prompt change,
and it costs nothing at inference.

Anyone revisiting this should treat "just add an instruction asking it to say X about its own
reasoning" as **measured and refuted for this prompt**, not untried.
