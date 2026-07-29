# AUDIT A2 — Minimum-stint veto: adversarial gate

Auditing whether moving the minimum-stint rule from a hard veto
(`apply_guard_rails` in `src/strategy/inference/guard_rails.py`) into the
Monte Carlo cost function is safe. Read-only audit, no repo files modified
except this report. Findings appended as discovered.

## Checklist

- [x] Claim A — FALSE. Three enforcement sites, not two (F1).
- [x] Claim B — feasible only with substantial follow-on work; one HIGH-risk
      load-bearing consumer (`decision_modes.py`, F4) and two independent MC
      scorers with zero stint-length input today (F5).
- [x] Claim C — CONFIRMED and stronger than stated: `apply_guard_rails` is
      never called in `rich` mode, by explicit design (F2).
- [x] Additional Q1 — no detection exists (F6).
- [x] Additional Q2 — yes, at least two more (F7).
- [x] Additional Q3 — no downstream state mutation from `action`; blast radius
      is display-only (F8).
- [x] What I tried to break and could not (final section)

---

## Findings log (chronological, appended as found)

### F1 — CLAIM A IS FALSE: the minimum-stint rule is duplicated in a THIRD prose
copy, not two. `src/agents/strategy_orchestrator.py:1594-1609` builds the N31
orchestrator's own LLM synthesis prompt (`_build_orchestrator_prompt`), and it
restates the same "HARD" guard-rail set as its own rule 4:

```
1602: f"  4. Minimum stint before pit: SOFT >= 8 laps, MEDIUM >= 12, HARD >= 15.\n"
1603: f"     If tyre_life is below minimum, override to STAY_OUT (current set has life left).\n"
```

This is a free-standing prose restatement, independent of both
`guard_rails.py` and the N28 prompt in `pit_strategy_agent.py:635-652`. It does
not import `_MIN_STINT_LAPS`; the numbers 8/12/15 are hand-typed a third time.
There is no test asserting these three numbers stay in sync with
`guard_rails._MIN_STINT_LAPS`. **Severity: HIGH for the redesign itself** — a
change to `guard_rails.py`/N28's prompt that does not also touch this string
leaves the orchestrator's own LLM believing the old bound is still absolute,
which is exactly the "twin that never got the fix" failure class this repo is
known for (see project memory `feedback_the_twin_that_never_got_the_fix`).

Also note the orchestrator prompt additionally restates rules 1/2/3/5/6 (pit
window early/late, REACTIVE_SC gating, compound-vs-laps, opening-lap threat
discount) — so ANY guard-rail change touches at minimum 3 files:
`guard_rails.py`, `pit_strategy_agent.py` (N28 prompt), `strategy_orchestrator.py`
(N31 prompt, `_build_orchestrator_prompt`).

### F2 — CLAIM C CONFIRMED, and stronger than stated: `apply_guard_rails` is
**never called** in the `rich` (LLM) profile, and the code says so explicitly.
`src/strategy/inference/engine.py:358`:

```python
guardrail_reason=None,  # rich mode applies rails via the LLM prompt, not post-hoc
```

`_run_rich` (`engine.py:235-360`) calls `_assemble_recommendation` at line
321-343 with no guard-rail post-processing step anywhere in between — the LLM
synthesis (`synth = _get_orchestrator_llm().invoke(prompt)`, line 320) goes
straight into `_assemble_recommendation`, whose own docstring at
`strategy_orchestrator.py:2131` states plainly: `action = synth.action` — "The
action is the synthesis's, always." `_assemble_recommendation`'s docstring
(`strategy_orchestrator.py:2038-2057`) is explicit that this is deliberate
design, not an oversight — it documents the REJECTED SC rail (#464) and
concludes "There is no action rail here, and there must not be one" — but that
reasoning is scoped to the SC-forcing rail; the minimum-stint rail is a
*different* rail (proscriptive, not prescriptive) and nothing in that
docstring, or anywhere else in `_assemble_recommendation`, checks it.

So: in `rich` mode, the ONLY enforcement of "no pit before lap 5 / no pit in
last 3 / minimum stint" is the prose in the two prompts (N28 + N31, see F1). An
LLM that ignores its own prompt text can ship any `action` completely
unchecked. `apply_guard_rails`'s single production call site is
`src/strategy/inference/no_llm.py:293`, confirmed — the docstring on
`guard_rails.py:9-10` itself says as much ("the DETERMINISTIC MIRROR of that
prose so the offline no-llm path behaves like the LLM path"), i.e. the
authors already knew and designed it this way: the veto is the no-llm
substitute for LLM judgement, not a safety net over it. **This means "moving
the veto into the MC cost" only changes offline (no-llm) behaviour — the rich
/ LLM path (the DEFAULT for `/simulate`, arcade, and the CLI per
`engine.py:330-331`) was never actually vetoed by this code at all.** The
premise "the bound catches normal strategy, not absurdity" is real for
no-llm; for rich mode there was never a code-level bound to loosen — only
prompt text, which the redesign should also touch (see F1).

Confirmed by mode, per surface (all via the shared `run_lap` dispatcher,
`src/strategy/inference/engine.py:174-232`):

| Surface | Default profile | Evidence |
| --- | --- | --- |
| CLI (`scripts/run_simulation_cli.py`) | `rich` unless `--no-llm` | `run_simulation_cli.py:1750`: `profile = "no-llm" if args.no_llm else "rich"` |
| Arcade (`src/arcade/strategy_pipeline.py`) | `rich`, always | `strategy_pipeline.py:47-48`: `run_lap(..., profile="rich", ...)` — no branch |
| Backend live stream (`simulator.py::simulate_race`) | `rich` unless `config.no_llm` | `simulator.py:881-897`: `if config.no_llm: _run_no_llm_path(...) else: run_lap(..., profile="rich", ...)` |
| Backend `/simulate` non-streaming path | `no-llm` | `simulator.py:436-439` |
| `src/strategy/eval/decision_modes.py` (eval only) | `no-llm`, always | line 306-308 |

So `apply_guard_rails` fires only on an explicit `--no-llm`/`no_llm=True` opt-in
or inside offline eval tooling — never on the default, interactive path a user
actually watches in the arcade or the CLI without that flag.

### F3 — Arcade's own guard-rail UI is already dead code, independent of this
redesign. `src/arcade/dashboard/orchestrator_card.py:10-11` documents a
"Guardrail line: DANGER-coloured when `guardrail_reason` is set so the user
sees *why* the orchestrator overrode the MC winner," rendered at
`orchestrator_card.py:163` (`guardrail = latest.get("guardrail_reason")`).
But `src/arcade/strategy.py:849`, in `_build_decision` (the function that
builds the `LapDecisionDTO` the dashboard actually reads), hardcodes
`guardrail_reason=None` unconditionally — it never reads
`agent_outputs.get("guardrail_reason")` the way
`src/telemetry/backend/services/simulation/simulator.py:458` and `:540` do for
the backend's own DTO. Combined with F2 (arcade always runs `profile="rich"`,
where `guardrail_reason` is `None` by construction anyway), this UI element
cannot currently fire through ANY path: not because rich mode never guard-rails
(true, but orthogonal), but because the wiring that would carry a value even in
a hypothetical no-llm arcade run was never connected. **Severity: MEDIUM,
pre-existing, unrelated to the redesign but worth fixing alongside it** since
the redesign will already be touching `guardrail_reason`'s meaning.

### F4 — CLAIM B, HIGH: `src/strategy/eval/decision_modes.py` is a load-bearing,
STRING-MATCHING consumer of `apply_guard_rails`'s exact `reason` text, and the
whole tier's methodology assumes the minimum-stint rule is a hard, deterministic
veto.

`guard_rail_block()` (`decision_modes.py:188-213`) calls `apply_guard_rails`
directly with a probe action of `"PIT_NOW"` and buckets the result by matching a
substring of `reason` against `_RAIL_BUCKETS` (`decision_modes.py:181-185`):
`"pit window not open"` → `opening_laps`, `"too late to pit"` → `closing_laps`,
`"minimum stint"` → `min_stint`. Its own docstring is explicit about why this
exists: *"A real stop inside a rail can never be agreed with no matter how good
the strategy is, so folding it into the headline would measure the rail
instead of the decision."* — i.e. the entire exclusion logic assumes the rail
is a **categorical, deterministic block**, not a soft cost a good-enough MC
draw can still clear.

Concretely, if the minimum-stint check inside `apply_guard_rails` is removed
(or turned into something that no longer always returns `("STAY_OUT",
"guard-rail: minimum stint not reached ...")` for a short stint):

1. `guard_rail_block` stops returning `"min_stint"` for any of the 22 real
   stops currently in that bucket (`documents/eval_reports/decision_modes.md:22`,
   `documents/eval_reports/decision_modes.json` — 22 occurrences of
   `"bucket": "min_stint"`). Those 22 stops fall through into the scored/no-call
   path instead of being excluded, silently changing `sample_size`,
   `exact`/`within_one`/`within_two`/`mean_signed_error`, and `scored_share` —
   **published, golden numbers in a checked-in eval report move** without the
   report's own text (`decision_modes.py:497`, *"`min_stint` are stops the
   guard rails make impossible to agree with"*) being updated to match.
2. `test_every_bucket_is_reachable_through_the_real_rail`
   (`tests/eval/test_decision_modes.py:119-128`) asserts
   `guard_rail_block(30, 57, "HARD", 2) == "min_stint"` — fails outright.
3. `test_guard_rail_block_names_the_rule`
   (`tests/eval/test_decision_modes.py:82-95`) has two parametrised cases keyed
   on `"min_stint"` — fail.
4. `test_block_agrees_with_the_rail_on_every_lap_of_a_race`
   (`tests/eval/test_decision_modes.py:98-116`) sweeps every lap 1-57 across
   three (compound, tyre_life) pairs and asserts `guard_rail_block` agrees with
   `apply_guard_rails` on every single one — fails the moment their notions of
   "blocked" diverge (a probabilistic MC cost has no yes/no answer to
   re-derive from at a single probe point).
5. In `tests/mc/test_guard_rails.py`: `test_the_three_bounds_fire_on_a_green_lap`
   (line 42), `test_a_safety_car_suspends_the_minimum_stint_bound` (lines
   63-72), and `test_every_compound_minimum_is_suspended_by_a_neutralisation`
   (lines 111-116, parametrised SOFT/MEDIUM/HARD) all assert
   `apply_guard_rails(..., tyre_life=<below min>)[0] == "STAY_OUT"` — every one
   of these needs a rewrite, not an update, because the return contract
   (deterministic veto + reason string) is exactly what is being removed for
   this one rule.

By contrast, `src/strategy/eval/stint_lengths.py` — the report that produced
the 12-17% figure this proposal is built on — only imports the **constant**
`_MIN_STINT_LAPS` (`stint_lengths.py:46`), never calls `apply_guard_rails`
itself, and only measures real stint lengths against that constant as a
threshold. As long as `_MIN_STINT_LAPS` survives as a named constant (even if
its role changes from "veto boundary" to "MC cost input"), `stint_lengths.py`
and `tests/eval/test_stint_lengths.py` keep working unmodified — this is the
one consumer that is NOT coupled to the veto's *behaviour*, only to its
*numbers*.

### F5 — CLAIM B feasibility: the Monte Carlo layer currently has ZERO
awareness of tyre life or compound, in EITHER of its two scoring
implementations. `_run_mc_simulation` (`src/agents/strategy_orchestrator.py:1281-1420`)
takes `pace_out, tire_out, situation_out, pit_out, alpha, rivals, position,
laps_remaining, pit_context` — no `tyre_life`, no `compound`. It dispatches to
one of two candidate scorers depending on whether per-rival gap data is
present (`_has_usable_gaps(rivals)`, line 1396):
- `simulate_lap_window` (line 667) — the legacy seconds-based scorer, used
  when `rivals` is falsy.
- `_run_projection_mc` (line 1068) — the position-projection scorer (#550
  redesign), used whenever real per-rival gaps are available, which per
  project memory (`project_mc_redesign_shipped`) is the path real races
  actually exercise.

"Move the minimum-stint preference into the Monte Carlo cost" therefore means
threading a NEW input (current tyre_life + compound, or a precomputed
"stint-freshness penalty") into **both** scorers, not one — this is precisely
the shape of bug this repo's own memory calls its dominant defect (one copy
fixed, the twin left alone; see `feedback_the_twin_that_never_got_the_fix`).
`race_state.tyre_life` and `race_state.compound` are already available at the
`run_lap`/`_run_rich`/`run_no_llm_lap` call sites (they are what
`apply_guard_rails` reads today), so the data is not missing — only the
threading into the two scorer signatures, their tests, and the MC unit tests
that pin `simulate_lap_window`'s and `_run_projection_mc`'s numeric outputs
(not located in this audit's scope, but any test asserting exact MC scores for
a fixed input will need new fixtures once a stint-cost term changes the score).

### F6 — Additional Q1: if the LLM ignores the prompt bound, nothing notices.
Confirmed by exhaustive search (`violat`, `disagrees with`, `contradicts`,
`would_have_blocked` etc. across `src/`) — there is no code anywhere that
compares a `rich`-mode `StrategyRecommendation.action` against what
`apply_guard_rails` (or any rail) would have said, not even as a passive
warning/log. `tests/mc/test_sc_regulatory_rails.py` gets close
(`test_the_shipped_action_never_contradicts_its_own_reason`,
lines 81-107) but that test is specifically about the REJECTED SC-forcing rail
having been removed, checking that `reasoning` and `action` do not contradict
each other post-hoc — it says nothing about whether `action` obeyed the
min-stint/pit-window bounds. **There is no audit trail, log line, or metric
that would ever surface an LLM ignoring its own prompt's guard-rail
section.** The only way to find out today is to grep `reasoning` text for
"minimum stint"/"pit window" and manually cross-check against `action`, lap,
and tyre_life — nothing does this automatically.

### F7 — Additional Q2: yes, more prompt exceptions have no code implementation
— this is a general pattern, not unique to the minimum-stint rule.

- **Already self-documented** (`guard_rails.py:78-83`): the early-race bound's
  "radio confirms damage/puncture/mechanical failure" exception is prose-only;
  `apply_guard_rails` takes no radio/damage argument at all.
- **Already self-documented, and intentionally INVERTED**
  (`guard_rails.py:68-76`): the prompt says the end-of-race bound is exempted
  by "Safety Car deployed" (`pit_strategy_agent.py:628`, "Exception: tyre
  failure is imminent ... or Safety Car deployed"); the code deliberately does
  NOT implement that half of the exception (`sc_active` only suspends the
  early-race and min-stint bounds, never the end-of-race one) — a reasoned,
  documented divergence, but it means an LLM reading its own prompt literally
  is being told something the deterministic mirror will not do.
- **Not previously documented**: the "COMPOUND vs REMAINING LAPS" prose rule
  (`pit_strategy_agent.py:654-658`, restated as orchestrator rule 5 at
  `strategy_orchestrator.py:1604-1605`) states SOFT is valid "only if remaining
  laps <= 15." But `recommend_compound_tool`
  (`pit_strategy_agent.py:1268-1271`) picks the first candidate (sorted by
  ascending `_STINT_CAPACITY_LAPS`, where SOFT=18) whose capacity covers
  `laps_remaining` — so at `laps_remaining=16`, the tool recommends SOFT while
  the prompt's own stated rule (`<=15`) forbids it. This is a second,
  independently-discovered instance of "the prompt's numbers and the code's
  numbers disagree" (18 vs 15), on a DIFFERENT rule than the one under review,
  which supports treating the whole "prose is the spec" model as suspect
  rather than assuming the minimum-stint case is an isolated drift.
- The "REACTIVE_SC usage" rule (`pit_strategy_agent.py:660-666`) — when to
  prefer `REACTIVE_SC` over `PIT_NOW`/`STAY_OUT` — has no code counterpart at
  all in either profile; it is pure LLM judgement in `rich` mode, and in
  `no-llm` mode `REACTIVE_SC` is just one of the four `_PIT_ACTIONS` the MC can
  pick via `best_mc_candidate` with no rule steering the choice specifically.

### F8 — Additional Q3: nothing downstream assumes `action` was already
guard-railed, because nothing downstream MUTATES simulation state from
`action`. Searched for `action == "PIT_NOW"` (and equivalents) across the
codebase — the only non-test hits are inside the MC's own candidate scoring
(`strategy_orchestrator.py:716`, scoring a hypothetical `PIT_NOW` candidate,
not consuming a final decision) and `decision_modes.py:207` (the eval probe,
already covered in F4). The system replays REAL historical telemetry
(`RaceReplayEngine`); `race_state.compound`/`tyre_life`/`position` for the NEXT
lap always come from what actually happened in the recorded race, never from
executing the recommended `action`. So a `PIT_NOW` shipped on lap 2 does not
desync any downstream state machine — it is purely advisory/display, rendered
by `classify_action`-style badge colouring in the CLI/arcade. The risk of an
unvetoed `action` is entirely reputational/decision-quality (a user watching
the dashboard sees a lap-2 stop recommended), not a crash or data-corruption
risk. This lowers the urgency of F2/F6 slightly from "could break the pipeline"
to "could visibly embarrass the system in front of a user, with no automatic
detection" — still worth fixing, but the blast radius is narrower than a
state-corruption bug would be.

---

## Severity ranking

| # | Finding | Severity | Why |
| --- | --- | --- | --- |
| F1 | Third prose copy of the min-stint rule in the N31 orchestrator prompt (`strategy_orchestrator.py:1602-1603`), Claim A is false | **HIGH** | Any redesign touching `guard_rails.py`/N28's prompt that skips this third copy leaves the orchestrator's own LLM believing the old bound is absolute — the repo's own dominant defect class, reproduced inside the very issue proposing to fix a rail |
| F2 | `apply_guard_rails` is never called in `rich` mode (default for CLI, arcade, backend live stream) | **HIGH** | The premise "the veto is too strict" only holds for no-llm; in rich mode there is no code-level veto to loosen, only prose — so the redesign's actual leverage is smaller than it appears unless the prompt text is treated as equally in-scope |
| F4 | `decision_modes.py`'s eval tier assumes the min-stint rail is a hard, deterministic, textually-stable veto | **HIGH** | Silently moves a published, checked-in eval report's headline numbers (22 stops currently excluded) and breaks 5+ named tests outright; the tier's entire "impossible to agree with" methodology needs a redesign, not a patch |
| F5 | MC scoring (`simulate_lap_window` + `_run_projection_mc`) has zero tyre-life/compound input today, in either of its two implementations | **HIGH** (feasibility) | "Move it into the cost" is a real feature addition across two independent scorers, not a one-line change — exactly the shape of change this repo's history shows gets applied to one copy and not the other |
| F3 | Arcade's `guardrail_reason` UI wiring is dead (`strategy.py:849` hardcodes `None`) | MEDIUM | Pre-existing, unrelated to this redesign's premise, but the redesign will be touching this exact field's meaning and should not re-ship the same dead wiring |
| F6 | No detection when an LLM ignores its own guard-rail prompt text | MEDIUM | No crash risk (F8), but genuinely unmonitored — a silent trust boundary |
| F7 | Two more prompt-only exceptions/rules with no code backing (radio-damage exception; SOFT-compound distance band mismatch, 15 vs 18 laps) | LOW-MEDIUM | Confirms the "prose is not the code" gap is systemic, informs how much to trust the N28/N31 prompts as a spec during the redesign |
| F8 | Nothing downstream mutates state from `action`; blast radius of an unvetoed action is display-only | Informational (lowers urgency, does not raise it) | Bounds how bad F2/F6 actually are in practice |

## Fix list, ordered by value and risk

1. **Before touching `guard_rails.py`, update all three prose copies together**
   (N28 prompt `pit_strategy_agent.py:635-652`, N31 prompt
   `strategy_orchestrator.py:1594-1609`, and the module docstring in
   `guard_rails.py` itself) — or, better, stop hand-duplicating the numbers a
   third time and have `_build_orchestrator_prompt` render its guard-rail
   section from the same `_MIN_STINT_LAPS`/`_NO_PIT_BEFORE_LAP`/
   `_NO_PIT_LAST_N_LAPS` constants `no_llm.py` already imports, so a future
   change to the numbers cannot re-create F1. (Addresses F1, highest value:
   removes an entire class of future drift, not just this one.)
2. **Decide explicitly whether the redesign is about the no-llm veto, the
   prompt text, or both**, and say so in the issue/design doc. Given F2, "move
   it into the MC cost" as currently scoped only changes behaviour when
   `--no-llm`/`no_llm=True` is passed or inside `decision_modes.py` — the
   default, user-facing `rich` path is untouched by the code change and
   remains governed entirely by prose the LLM may or may not obey (F6). If the
   goal is to fix rich-mode behaviour too, the MC-cost input has to flow
   through `_build_orchestrator_prompt` (as numbers the LLM reasons over) or a
   post-hoc soft-scoring step needs to be added to `_assemble_recommendation`
   — currently absent by design (see its own docstring on the rejected SC
   rail).
3. **Redesign `decision_modes.py`'s exclusion methodology before merging any
   MC-cost change**, not after. The 22 `min_stint`-bucketed stops need a new
   answer: either the tier starts actually scoring them (since a soft cost
   means the MC *can* agree with them now) or the exclusion criterion is
   redefined around the new cost function rather than around
   `apply_guard_rails`'s reason string. Either way `documents/eval_reports/
   decision_modes.md` needs a full re-run and its prose (line 497) rewritten
   to stop claiming these stops are "impossible to agree with." (Addresses F4.)
4. **Thread tyre_life/compound into both `simulate_lap_window` and
   `_run_projection_mc` in the same change**, with a single shared helper for
   the stint-freshness cost term (not two copies), and add a test that
   exercises both scorers with the same inputs the way
   `test_block_agrees_with_the_rail_on_every_lap_of_a_race` does today for the
   veto — so the two scorers cannot silently diverge on this term the way the
   two prompts already have (F1) and the guard rails vs prompt already do (F7).
   (Addresses F5.)
5. **Wire `agent_outputs.get("guardrail_reason")` into `src/arcade/strategy.py`'s
   `_build_decision`** instead of the hardcoded `None`, while this field's
   meaning is already being revisited. Low cost, fixes a real (if currently
   invisible) dead-code path. (Addresses F3.)
6. **Add a cheap, passive rich-mode audit**: after `_assemble_recommendation`
   in `_run_rich`, compute what `apply_guard_rails` (or its successor cost
   function) would have said about `synth.action` and log a warning (never
   override — that would resurrect the rejected-rail pattern) when they
   disagree. Gives the first-ever visibility into F6 without adding a new
   forcing rail. (Addresses F6, lowest urgency but cheapest fix.)

---

## What I tried to break and could not

- **Tried to find a fourth or fifth prose/code copy of the min-stint numbers**
  beyond the three found (guard_rails.py, N28 prompt, N31 prompt) — grepped
  CLI, arcade, backend, every file under `src/agents/`, and `docs/pages/` for
  the literal numbers 8/12/15 near "stint"/"tyre_life". Found none beyond the
  three sites and one docs page (`docs/pages/multi-agent.md:139`, which is
  documentation describing behaviour, not an enforcement site — read it and
  confirmed it accurately describes the SC exception as currently coded).
- **Tried to find a downstream consumer that would crash or misbehave** if
  `apply_guard_rails` stopped returning a `"minimum stint"` reason string
  (e.g. a strict schema, a non-nullable field, a parser expecting the exact
  wording). Found none — `guardrail_reason` is `str | None` everywhere it is
  threaded (`no_llm.py:209`, `engine.py:373`, `simulator.py:138/518`,
  `strategy.py:128`), and every renderer treats `None`/absent gracefully
  (`orchestrator_card.py:163` guards with `latest.get(...)`, hides itself when
  falsy at line 143 `setVisible(False)`).
- **Tried to find evidence the `rich` path DOES enforce the bound some other
  way** (a strict Pydantic validator on `StrategyRecommendation.action`, a
  tool-level refusal in N28's tools mirroring the `_live_drivers` refusal
  pattern). None of N28's three tools (`predict_pit_duration_tool`,
  `score_undercut_tool`, `recommend_compound_tool`) reference lap number or
  tyre life as a refusal condition — they refuse only on driver liveness. The
  `StrategyRecommendation` Pydantic schema (`strategy_orchestrator.py`,
  around the `_ACTION_VALUES` Literal at line 252) constrains `action` to the
  five-value enum, nothing about when each value is legal.
- **Tried to construct a case where `sc_active` being suspended for the
  min-stint bound but NOT the end-of-race bound produces a contradiction**
  (e.g. lap 55/57 with a fresh 2-lap-old tyre and SC active) — traced it
  through `apply_guard_rails`: the end-of-race check (`remaining_laps <=
  _NO_PIT_LAST_N_LAPS and cliff_p10 >= _CLIFF_P10_SAFE`) is evaluated BEFORE
  the min-stint check and returns early, so the two conditions cannot both
  fire in a way that produces an inconsistent double-reason; the ordering
  already prevents it. This is sound as written.
- **Could not find** any place where `_MIN_STINT_LAPS` and the orchestrator's
  hand-typed `8`/`12`/`15` (`strategy_orchestrator.py:1602`) have already
  drifted apart numerically — as of this audit they still agree. The risk in
  F1 is prospective (the NEXT edit to one and not the other), not yet realised.


</content>
