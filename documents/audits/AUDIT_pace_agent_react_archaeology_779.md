# pace_agent ReAct archaeology (#779)

Part of epic #778. Answers: why does `src/agents/pace_agent.py` carry a complete
LangGraph ReAct scaffold that nothing calls, while the structurally identical
scaffolds in `tire_agent.py` / `pit_strategy_agent.py` / `race_situation_agent.py`
are genuinely wired into `run()`/`run_from_state()`?

Method: read `notebooks/agents/N25_pace_agent.ipynb` and
`notebooks/agents/N31_strategy_orchestrator.ipynb` in full (source-only dump,
outputs stripped, via a throwaway script — no notebook file was edited), plus
`git log -p --follow` / `git log -S` on `src/agents/pace_agent.py` and the
GitHub history of issue #476. This report is the full finding; nothing here was
buffered, the investigation is complete.

## Summary verdict

**The scaffold was born unreachable, then deliberately left unreachable at a
later decision point, and never had a stated future-use commitment — only a
"do not delete without knowing why" caution note added during last week's
cleanup.** No PR description, commit message, issue, or notebook markdown was
found promising to wire it. The one place a rationale IS on record
(`N25_pace_agent.ipynb`, Step 5 markdown) argues pace's own case for an inner
ReAct loop is weak by design, not that it was deferred.

## Timeline

### 1. 2026-03-31 — `88dfe40` — first extraction commit

`pace_agent.py` is created from the notebook. The ReAct scaffold
(`get_pace_react_agent`, `PACE_TOOLS`, `predict_pace_tool`,
`get_session_median_tool`, `_PACE_SYSTEM_PROMPT`) exists from line 1 of the
module's life, **alongside** `run_pace_agent()`, which already builds its
`reasoning` field as a deterministic f-string — never a call into the ReAct
agent. The two paths were born side by side and never unified.

### 2. 2026-04-05 — `3f55c9f` — "agents: LangGraph fix + gpt-4.1-mini defaults..."

**This is the pivotal commit.** It rewrites all four per-lap agents
(`pace_agent.py`, `tire_agent.py`, `pit_strategy_agent.py`,
`race_situation_agent.py`) in the *same commit*, converting each from
free-function modules to an OOP `XAgent` class with a lazy `get_react_agent()`
method. At the end of this exact commit:

- `tire_agent.py`'s `run()`/`run_from_state()` call `self._run_core()`, which
  calls `self.get_react_agent()` and `.invoke()`s it — confirmed by grepping
  the commit's own tree state (`git show 3f55c9f:src/agents/tire_agent.py`).
  Same for `pit_strategy_agent.py` and `race_situation_agent.py`.
- `pace_agent.py`'s `run()`/`run_from_state()` build the feature row, call
  `self._predict()`/`self._bootstrap_ci()`/`self._session_median()` directly,
  and assemble the f-string `reasoning` — `self.get_react_agent()` is defined
  but has **zero call sites** anywhere in the same commit's tree.

So this was not incremental drift — in one sitting, three agents were wired to
their ReAct scaffold and pace deliberately was not, even though its scaffold
was written to the same level of completeness (tools, prompt, lazy singleton)
as the other three. The same commit also renamed the section header above the
scaffold from `# LangGraph ReAct agent` to
`# LangGraph tools and ReAct agent (preserved 100% — no functional changes)` —
in context this reads as "I touched everything else in this file in this
refactor pass, this block's *behavior* is exactly what it was before", not as
a forward-looking commitment to wire it later. It is refactor-commit hygiene,
not a design promise.

### 3. 2026-07-18 — `5f84218` — "fix(agents): validate LLM tool inputs..." (closes part of #476)

A prior Fable-audit hardening pass added `PaceAgent._validate_pace_inputs()`
and wired it into `predict_pace_tool`, explicitly treating pace's tool as
LLM-facing: *"reuse the `_live_drivers` guard across the pit, tire, situation
and pace agents so an off-track driver or out-of-range lap is refused instead
of computed on"* (commit message). Issue #476 itself frames `predict_pace_tool`
as the *"extreme"* case among the four agents' tools — 19 raw parameters, no
`laps_df` closure — and says the "rich" (LLM) profile *"hands the same
arguments to the LLM and never checks them back."*

**This is significant: a previous audit session hardened pace's tool as if it
were reachable by an LLM, without ever discovering that it wasn't.** The false
belief "pace is wired like its siblings" did not start with Víctor in the
2026-08-01 cleanup session — it was already latent in the codebase's own audit
history three weeks earlier, and that audit pass reinforced it instead of
catching it (its own fix touched all four agents' input-validation uniformly,
which is exactly the kind of pattern-matched fix that would not surface a
one-of-four wiring gap).

### 4. 2026-08-01 — `1fec855` / `d5b6084` — this epic's origin

The cleanup session removes the dead **wrapper** function `get_pace_react_agent()`
(module-level, zero callers, distinct from the instance method
`PaceAgent.get_react_agent()`), confirms the deeper scaffold is also dead, and
explicitly declines to delete it because of the "preserved 100%" header —
opening this epic instead.

## What the notebooks actually say

### N25_pace_agent.ipynb — Step 5 markdown (cell 31), the one on-record rationale

> "This step wraps the inference functions from Steps 1–3 into a proper
> LangGraph ReAct agent — the interface that N31 (Strategy Orchestrator) will
> use to delegate pace queries."
>
> "**The key difference from `run_pace_agent` in Step 3**: the ReAct agent
> lets the LLM decide which tools to call and can reason about the outputs
> before responding. **For a deterministic single-tool workflow the two are
> equivalent**; the agent pattern pays off in N31 where the supervisor LLM
> selects which sub-agents to activate each lap."

Two things follow from this, in the notebook author's own words, at design
time:

1. The *stated* intended consumer of pace's ReAct wrapping was N31's
   *supervisor* calling pace's tools directly (a single flat ReAct with every
   sub-agent's tools available to one outer LLM) — **not** an inner ReAct loop
   private to `PaceAgent` the way tire/pit/situation ended up implementing.
2. The notebook itself already flags that pace is a weak case for the pattern:
   *"for a deterministic single-tool workflow the two are equivalent"* — i.e.
   wrapping pace in ReAct was expected to buy nothing over calling it directly,
   specifically because pace has no decision for an LLM to make.

### N31_strategy_orchestrator.ipynb — the "LLM-as-router" design note (cell 7) that supersedes (1)

N31's own Step 1 markdown contains an explicit, reasoned rejection of exactly
the architecture N25 anticipated:

> "**Why deterministic routing is the right choice here:** ... 4. *Latency* —
> an LLM-as-router (e.g. ReAct-style planner deciding which agents to call)
> adds one extra round-trip before any real computation starts. For a
> real-time system making per-lap decisions every ~90 seconds, this matters."
>
> "**What could be made more complex (and why it's deferred):** ... *LLM-as-router*:
> let the LLM itself decide which agents to invoke ... Rejected for latency
> and non-determinism — a strategy call must be auditable and fast, not
> dependent on an LLM reasoning about whether to call another LLM."

N31's real `_decide_agents_to_call` is a plain deterministic `if`/`else`
function. N31's own demo/production code (`_run_always_on_agents`) calls
`run_pace_agent(...)` as an ordinary function with keyword arguments — never
through a tool-call interface — and does the exact same thing for
tire/situation (`run_tire_agent(lap_state)`, `run_race_situation_agent(lap_state)`),
even though those two *do* internally spin up their own ReAct loop once called.

**This is the resolving fact.** N31 was never going to be the "one supervisor
ReAct with everyone's tools" that N25's Step 5 markdown anticipated — that
architecture was explicitly designed away in N31 for auditability/latency
reasons, for the routing decision specifically. What N31 actually calls is a
plain function per agent; whether that function's *internal* implementation
uses an LLM ReAct loop is a decision private to each agent module, made
per-agent in commit `3f55c9f`. Pace's Step 5 rationale ("equivalent for a
deterministic workflow") was never revisited against that outcome — it just
turned out to still apply, because N31 changed the routing layer, not the
individual agents' honesty about whether they have a qualitative judgment to
make.

## Corroborating architecture evidence (from the current codebase, not the notebooks)

- `src/strategy/inference/no_llm.py` (`_NullReActRunner`, `run_no_llm_lap`)
  calls `run_pace_agent_from_state(lap_state)` directly with **no** injection
  seam — because there is no LLM step to null out. For tire and situation, the
  same module has to inject a `_NullReActRunner` at the `_react_agent` cache
  seam specifically because those two *do* run a live ReAct loop in normal
  ("rich") mode. This file's own docstring states the asymmetry as a design
  fact: *"N25/N26/N27/N29 produce REAL model numbers... pace via its public
  XGBoost entry, tire/situation by injecting a deterministic tool-runner."*
- `src/telemetry/backend/mcp_tools.py`'s chat-facing MCP tools
  (`predict_pace`, `predict_tire`, `predict_situation`, `predict_pit`) are
  *uniform* — each just calls that agent's `run_*_agent_from_state(...)` and
  lets the agent's own implementation decide whether an inner LLM runs. The
  outer chat LLM already can (and does) call pace today; the only missing
  piece is whether `PaceAgent.run_from_state()` itself nests a second LLM call
  the way tire/situation/pit do.
- `TireOutput.warning_level` (the categorical judgment tire's siblings each
  carry) is computed in `__post_init__` from a numeric threshold on
  `laps_to_cliff_p10` — **not** by the LLM. The ReAct loop's real contribution
  for tire is (a) which of 1–2 tools to call and in what order, and (b) a
  natural-language `reasoning` sentence built from the tool outputs — not a
  categorical decision the deterministic code couldn't make itself. Relevant
  context for #780: the siblings' "qualitative judgment" is itself
  post-hoc-deterministic; the LLM's value-add across all three is narrower
  than "produces the category field" might suggest.

## What was NOT found

- No PR description, issue, or commit message stating an intended future date
  or trigger for wiring pace's ReAct path.
  the closest candidate, #476, hardens the tool as if reachable but never
  states an intent to make it reachable.
- No notebook cell, in either N25 or N31, that actually exercises
  `pace_react_agent.invoke(...)` successfully — N25's own Step 5 demo cell
  (cell 35) is wrapped in a `try/except` that falls back to the direct
  `run_pace_agent()` call, "when the LLM is unavailable"; nothing in the
  dumped notebook confirms it was ever run with a live LLM connected — but
  we assume that never was tried given that we're focusing on Local LM Studio
  for provider testing and OpenAI (paid) for chat testing.
- No sign that a "future chat-style pace query, distinct from the per-lap
  orchestrator loop" (the RAG-agent-shaped precedent named in the epic) was
  ever planned for pace specifically — the actual chat surface
  (`backend/mcp_tools.py::predict_pace`) already exists and bypasses the
  scaffold entirely, calling the deterministic path like its three siblings'
  MCP wrappers call theirs (the difference is only inside each agent's own
  `run_from_state`).

## Bearing on #780

This archaeology does not resolve #780 — that is a product decision, not a
historical fact, and it is Víctor's to make. What it does establish, for the
decision conversation:

1. The scaffold's existence is not itself evidence of a deferred plan — it was
   written from a template shared with three other agents in one commit, and
   whether to wire it was a call made per-agent in that same commit, not left
   open.
2. The one contemporaneous design rationale on record for pace specifically
   (N25 Step 5) argues against wiring it, on the grounds that pace has no tool
   arbitration or qualitative judgment for an LLM to add — a rationale that
   was never contradicted by any later document.
3. A previous audit pass (#476, 2026-07-18) hardened pace's tool inputs as if
   it were live, which shows the "pace is wired" belief has already once
   produced real (if harmless) engineering effort in the codebase based on a
   false premise — a second data point, beyond Víctor's own belief, that this
   gap is easy to miss by pattern-matching against the other three agents.
