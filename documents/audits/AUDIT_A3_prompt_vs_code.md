# AUDIT A3 — Prompt vs Code (agent prose rules vs deterministic/downstream enforcement)

Adversarial gate. Read-only audit, no LLM/API calls. Enumerates every behavioural rule
stated in prose inside the agent prompts (N25-N31) and checks whether code enforces it,
whether the exception sets match, and what happens when the LLM ignores a prompt-only rule.

Status: COMPLETE.

---

## Scope

- `src/agents/pit_strategy_agent.py` (N28) — prompt ~L560-700 + full file
- `src/agents/tire_agent.py` (N26)
- `src/agents/race_situation_agent.py` (N27? / SC agent)
- `src/agents/pace_agent.py` (N25)
- `src/agents/radio_agent.py` (N29)
- `src/agents/rag_agent.py` (RAG tool agent)
- `src/agents/strategy_orchestrator.py` (N31)
- `src/strategy/inference/guard_rails.py` (the deterministic mirror)

## Already known (excluded from "new findings", but re-checked)

1. SC exception on early-race and minimum-stint bounds — FIXED today.
2. End-of-race bound intentionally has NO SC exception in code (Art. 55.17) — checking reasoning.
3. Damage/puncture/mechanical exception on early-race bound NOT wired into code — known, documented.

---

## Findings log (append-only, most recent at bottom until final ranking)

### F1 — N28's own "COMPOUND vs REMAINING LAPS" hard bound contradicts the tool the LLM is told to call

Prompt (`pit_strategy_agent.py:654-658`, inside "## Strategic guard-rails (HARD constraints...)"):
```
COMPOUND vs REMAINING LAPS:
  SOFT: recommend only if remaining laps <= 15 (it won't last longer).
  MEDIUM: suitable for 12-30 remaining laps.
  HARD: suitable for 20+ remaining laps.
```
Also restated verbatim as N31's own guard-rail #5 (`strategy_orchestrator.py:1604-1605`):
```
5. Compound must fit remaining laps: SOFT only if <= 15 laps remain,
   MEDIUM for 12-30, HARD for 20+. Wrong compound forces an extra stop.
```

Code — the ONLY tool N28 is told to call for a compound suggestion (`pit_strategy_agent.py:91`):
```python
_STINT_CAPACITY_LAPS: dict[str, int] = {"SOFT": 18, "MEDIUM": 30, "HARD": 38}
```
used at `pit_strategy_agent.py:1255,1269,1273` inside `recommend_compound_tool` — SOFT capacity 18 (not 15), MEDIUM 30 (matches the upper bound only), HARD 38 (not 20+).

1. HARD rule per both prompts ("HARD constraints" / "STRATEGIC GUARD-RAILS (HARD...)").
2. Enforcement: **none in code for the stated 15/12-30/20+ bound.** No post-processing validates `COMPOUND:` against `remaining_laps`; `apply_guard_rails` (the deterministic mirror) does not implement a compound-vs-laps check at all — it only has the three numeric bounds already known (early-race, end-of-race, min-stint). Grepped `guard_rails.py` fully (111 lines, read above): no compound/remaining-laps logic exists there.
3. Exception sets: N/A (not a suspend/exempt rule) — but the underlying NUMBERS disagree between the two prose copies (which agree with each other, 15/12-30/20+) and the one piece of code that actually picks a compound by remaining-laps math (18/30/38, Pirelli average stint capacity — a genuinely different, more realistic number by the tool's own docstring: "Pirelli average stint capacities"). The tool is explicitly a *fallback/priority-2* heuristic per its own docstring, so this may be intentional domain modelling (Pirelli capacity vs a stricter strategic margin) — but nothing in either prompt or the tool's docstring cross-references the other, so an LLM calling `recommend_compound_tool` with 16 laps remaining on SOFT gets back `SOFT` (18 >= 16) from the tool while its own "HARD constraint" says SOFT is invalid past 15. The two numbers actively disagree for laps 16-17 (tool says SOFT is fine, prompt-as-hard-rule says it is forbidden) and for laps 21-29 on MEDIUM vs HARD (tool would recommend MEDIUM as the cheapest sufficient compound because MEDIUM capacity 30 >= laps_remaining, while the HARD-constraint text tells the model "HARD: suitable for 20+" without forbidding MEDIUM in that range — actually not contradictory there, only the SOFT boundary truly conflicts).
4. If the LLM trusts the tool's output verbatim over its own guard-rail text: it can hand back a COMPOUND value the "HARD constraint" section forbids. Downstream: nothing validates `PitStrategyOutput.compound_recommendation` against `remaining_laps` anywhere in `strategy_orchestrator.py` or `no_llm.py` — grepped, zero hits for any post-hoc compound/laps cross-check.

Severity: **MEDIUM** — non-deterministic (only fires if the LLM follows the tool over the prose in the 16-17-lap-remaining window on SOFT), narrow window, and the eventual real-world cost is "one avoidable extra stop", not a race-ending error. But it is a genuine numeric drift between the prompt's stated HARD constraint and the one piece of code whose entire job is to pick a compound by remaining laps.

---

### F2 — Three copies of the end-of-race SC exception, and N28's own prompt is the ODD ONE OUT (not just "prompt vs code" — "prompt vs prompt")

The audit brief told me the deterministic code deliberately drops the SC exception on the end-of-race bound (Art. 55.17) and that this is INTENTIONAL. Confirmed reading `guard_rails.py:68-76`. What the brief did NOT flag: **there is a THIRD copy of this exact rule, in N31's own prompt, and it already agrees with the code — only N28's own prompt (the one the brief's framing centred on) is stale.**

N28 prompt (`pit_strategy_agent.py:626-628`):
```
PIT WINDOW — end of race:
  NEVER recommend PIT_NOW, UNDERCUT, or OVERCUT when remaining laps <= 3.
  Exception: tyre failure is imminent (laps_to_cliff P10 < 2) or Safety Car deployed.
```
N31 orchestrator's OWN restatement of the same guard-rail (`strategy_orchestrator.py:1598-1599`):
```
2. NO pit action when remaining laps <= 3 unless tyre failure imminent
   (cliff P10 < 2 laps). Pit cost ~22s vs ~1.5s recovery = ~13 positions lost.
```
Code (`guard_rails.py:98-99`, matches N31, not N28):
```python
if remaining_laps <= _NO_PIT_LAST_N_LAPS and cliff_p10 >= _CLIFF_P10_SAFE:
    return "STAY_OUT", f"guard-rail: too late to pit (<={_NO_PIT_LAST_N_LAPS} laps left)"
```
(`sc_active` is not referenced in this branch at all — only the tyre-cliff exception survives, exactly matching N31's wording, not N28's.)

1. HARD rule, all three copies.
2. Enforced in code (`guard_rails.py`) — but that code only runs on the **no-llm** path (`no_llm.py:293`) and in `decision_modes.py`'s evaluation harness. It is never invoked from the live LLM pipeline (`strategy_orchestrator.py` never imports or calls `apply_guard_rails` — grepped, zero hits in that file).
3. Exception sets side by side:
   - N28 prompt: `{tyre failure imminent, SC deployed}`
   - N31 prompt: `{tyre failure imminent}` — SC dropped
   - `guard_rails.py`: `{tyre failure imminent}` — SC dropped (matches N31, contradicts N28)
   So the deliberate fix already landed in N31's prompt and in the deterministic mirror; N28's own prompt was never updated to match. **This means the live LLM path has the SAME two sources of ground truth disagreeing with each other** (N28 tells its own agent SC exempts the bound; N31 tells the orchestrator SC does not, and instructs it to override any sub-agent action that "violates" N31's rules to STAY_OUT). In practice N31 synthesizes the final `StrategyRecommendation`, so N31's version (matching the code) probably wins out for the *final* decision — but N28's `PitStrategyOutput.action` and `.reasoning`, which N31 reads as one of its inputs and which are also surfaced directly to the CLI/arcade/webapp pit block (`pit_block` at `strategy_orchestrator.py:1560-1568`), can still show a PIT_NOW/UNDERCUT/OVERCUT recommendation justified by "SC deployed" in the last 3 laps that N31 will then (correctly, per the intentional design) override — but the sub-agent's own displayed reasoning still teaches the wrong rule to anyone reading N28's block directly (CLI verbose output, arcade agents tab, decision-memory accumulator inputs).
4. Downstream check: N31's own instruction ("override to STAY_OUT and explain why") is the only backstop, and it depends on the LLM correctly recognizing that N28 acted on a rule N31 no longer honours — nothing programmatic cross-validates `PitStrategyOutput.action` against N31's own guard-rails before it reaches the prompt text N31 reads.

Severity: **MEDIUM-HIGH** — the deliberate fix (matching Art. 55.17) is real and live in the path that actually produces the final recommendation, so the race-ending regression the brief worried about (re-creating the #464 defect) is NOT reproduced end-to-end. But N28's prompt is now flatly wrong relative to the system's own design decision, contaminates its own `reasoning` field, and is one N31-prompt-edit away from silently reintroducing the SC exception if anyone "fixes" N31 to match N28 instead of the other way around.

---

### F3 — REACTIVE_SC has two contradictory definitions between N28 and N31 (not a rediscovery of F2 — a different rule)

N28 prompt (`pit_strategy_agent.py:660-666`):
```
REACTIVE_SC usage:
  REACTIVE_SC is for the rare in-between case where sc_prob is elevated but the
  Safety Car is NOT yet deployed.  When the prompt states "SC STATUS: SAFETY CAR
  DEPLOYED RIGHT NOW", prefer PIT_NOW directly.  Use REACTIVE_SC only when
  sc_prob >= 0.30 AND the prompt shows the legacy "SC probability" line.  A high
  sc_prob without confirmation is still a contingency — mention it in reasoning
  and set ACTION to STAY_OUT unless the SC is actually out.
```
→ REACTIVE_SC means: SC **NOT** confirmed, probability elevated (pre-emptive read). When SC **IS** confirmed, N28 is told to prefer **PIT_NOW**, not REACTIVE_SC.

N31 orchestrator's own guard-rail #3 (`strategy_orchestrator.py:1600-1601`):
```
3. REACTIVE_SC only when SC IS deployed (confirmed). High sc_prob is a
   contingency trigger, not a primary action — use STAY_OUT with SC contingency.
```
→ REACTIVE_SC means: SC **IS** confirmed deployed. When only sc_prob is elevated (not confirmed), N31 says do **NOT** use REACTIVE_SC — use STAY_OUT instead.

These are the exact inverse of each other on both ends of the condition:
| Condition | N28 says | N31 says |
|---|---|---|
| SC confirmed deployed right now | prefer PIT_NOW (not REACTIVE_SC) | REACTIVE_SC is exactly this case |
| SC NOT confirmed, sc_prob >= 0.30 | REACTIVE_SC is exactly this case | not REACTIVE_SC; STAY_OUT with contingency |

1. Both are stated as HARD/authoritative ("guard-rails HARD" header covers N28's section too — REACTIVE_SC usage sits under the same `## Strategic guard-rails (HARD constraints — override any decision rule above)` header, line 616, no sub-header resets it).
2. No code enforces either definition on the live LLM path. `guard_rails.py`'s `_PIT_ACTIONS` frozenset includes `"REACTIVE_SC"` as a pittable action subject to the three numeric bounds, but has no logic distinguishing "SC confirmed" from "SC probability elevated" for the REACTIVE_SC label itself — that distinction is prompt-only in both places.
3. Exception sets: see table above — this is not an exception-set mismatch, it is the CORE PREDICATE being inverted between the two prompts.
4. If the LLM (N28) follows its own prompt and emits `REACTIVE_SC` while `sc_currently_active=False, sc_prob=0.35` (a legitimate, spec-compliant call per N28's own text), N31 reads that in `pit_block` and, per N31's rule 3 (as N31 understands REACTIVE_SC — confirmed-SC-only), will very plausibly judge N28's REACTIVE_SC call a **violation of N31's own guard-rails** ("REACTIVE_SC only when SC IS deployed") and override it to STAY_OUT — even though N28 was completely correct by its own contract. Conversely, if SC is actually confirmed active and N28 (correctly, per its own prompt) emits PIT_NOW instead of REACTIVE_SC, N31 has no problem with that (N31 never requires REACTIVE_SC to fire under confirmed SC, only restricts when it MAY fire), so this direction is silent. The asymmetric failure mode is real but one-directional: elevated-but-unconfirmed sc_prob calls from N28 are the ones N31 is primed to second-guess.
   Also worth noting: the *only* code artifact touching this concept, `sc_reactive` at `pit_strategy_agent.py:1592-1594` (`sc_currently_active or action=='REACTIVE_SC' or (sc_prob>=0.30 and action in ('PIT_NOW','UNDERCUT'))`), is a boolean flag on `PitStrategyOutput`, not a validator — it does not correct or block either agent's interpretation, it just records "was some SC signal involved" for downstream consumers (MC scoring, memory block). It cannot resolve the contradiction because it treats all three predicates as equivalent evidence of "SC-reactive", exactly blurring the line N28 and N31 disagree about.

Severity: **HIGH** — this is an LLM-only contradiction (no test catches it, per the audit brief's own framing — "the only tests covering those bounds sat behind a data/models/ gate"), it inverts a named enum value's meaning between the two prompts in the same call chain, and N31 is explicitly instructed to overrule N28 on exactly this kind of disagreement, so the systematic bias is toward suppressing a legitimate pre-emptive SC read into a passive STAY_OUT — the least safe failure mode for an actually-imminent Safety Car.

Addendum found while cross-checking N31's action schema: N31's own `action` field cannot literally BE `REACTIVE_SC` — `strategy_orchestrator.py:252` declares `_ACTION_VALUES = Literal["STAY_OUT", "PIT_NOW", "UNDERCUT", "OVERCUT", "ALERT"]`, and the reasoning rubric repeats the same five-value list (`strategy_orchestrator.py:1676-1677`, "Do not invent new values"). So N31's guard-rail #3 is necessarily about how the orchestrator should REACT to seeing `REACTIVE_SC` in N28's `pit_block` input, not about a value N31 could set itself — N31 must map it onto STAY_OUT or PIT_NOW. That makes the contradiction in the main finding sharper, not softer: N31 is reading a sub-agent field (`PitStrategyOutput.action == 'REACTIVE_SC'`) whose producing agent (N28) defines it one way, through a lens (N31's guard-rail #3) that defines the same word the opposite way, in order to decide which of N31's OWN differently-named actions to emit.

---

### F4 — N31's own restatement of the minimum-stint rule DROPS the SC exception that N28's prompt AND today's fixed `guard_rails.py` both carry

N28 prompt (`pit_strategy_agent.py:635-652`, full text quoted because the exception is load-bearing prose, not a one-liner):
```
MINIMUM STINT LENGTH before a pit makes sense:
  SOFT: current tyre_life must be >= 8 laps before recommending a stop.
  MEDIUM: >= 12 laps.  HARD: >= 15 laps.
  If the driver has NOT completed the minimum stint, recommend STAY_OUT (the current
  set still has useful life; pitting now wastes a tyre allocation).

  EXCEPTION — SC ACTIVE: when the prompt states "SC STATUS: SAFETY CAR DEPLOYED
  RIGHT NOW", the minimum stint constraint DOES NOT APPLY: a stop under a deployed
  SC is far cheaper because the field is delta-limited and queued behind the SC, so
  your RELATIVE loss shrinks.
  This makes pitting cheaper. It does NOT make it correct: weigh it against what a
  stop surrenders. [...] Decide on the race state, not on the SC alone.
```
Code, fixed today (`guard_rails.py:104-109`):
```python
min_life = _MIN_STINT_LAPS.get(compound, _DEFAULT_MIN_STINT)
if tyre_life < min_life and not sc_active:
    return (
        "STAY_OUT",
        f"guard-rail: minimum stint not reached ({compound} {tyre_life}/{min_life} laps)",
    )
```
→ suspended under SC, matching N28.

N31 orchestrator's own restatement (`strategy_orchestrator.py:1602-1603`):
```
4. Minimum stint before pit: SOFT >= 8 laps, MEDIUM >= 12, HARD >= 15.
   If tyre_life is below minimum, override to STAY_OUT (current set has life left).
```
→ **no SC exception at all.** Nothing in N31's guard-rail #4, nor anywhere else in the ~150-line prompt-builder function (`_build_orchestrator_prompt`, read in full, `strategy_orchestrator.py:1443-1707`), tells the orchestrator that a confirmed SC suspends the minimum-stint bound.

1. HARD rule, all three copies (same header logic as F2/F3).
2. Enforced in code (`guard_rails.py`), but only on the no-llm path, same caveat as F2.
3. Exception sets:
   - N28 prompt: `{SC active}` exempts (with an explicit "does not make it correct" nuance — N28 is told to still weigh race state).
   - `guard_rails.py`: `{SC active}` exempts — matches N28 (today's fix).
   - N31 prompt: `{}` — no exemption. This is the mirror image of F2: there, N31 already matched the (intentionally SC-blind) code and N28 was stale; here, N28 and the code both correctly exempt SC and **N31 is the one that's stale**, still describing the OLD pre-#716 behaviour (blanket minimum-stint bound, no SC awareness) that today's fix specifically corrected in `guard_rails.py`.
4. Downstream effect: N31 is the agent that actually writes the final `StrategyRecommendation.action`. Per N31's own instruction ("If a sub-agent recommends an action that violates these rules, override to STAY_OUT and explain why"), if N28 correctly emits `PIT_NOW`/`UNDERCUT` under a confirmed SC with `tyre_life` below the compound's minimum (now legitimate per N28's own exception and per the fixed `guard_rails.py`), N31 — reading its own rule 4, which has no SC clause — is primed to treat that as a rule violation and **override it back to STAY_OUT**, silently re-introducing the exact over-conservative-under-SC behaviour that `guard_rails.py`'s docstring (line 66: "a bound written to catch nonsense must not be what blocks the most valuable stop in racing") was written to prevent. Unlike F2 (where N31 already had the correct, code-matching text), here **N31 is positioned to cancel out today's fix** at the synthesis layer even though N28 and the deterministic mirror both got it right.

Severity: **HIGH** — this is the most direct real-world consequence of the three prompt copies drifting: today's fix (SC suspends the minimum-stint bound) can be fully undone by the very next layer in the same pipeline (N31), because N31's own guard-rail text was never updated when `guard_rails.py`/N28 were. This is functionally the SAME bug class as the one fixed today, just one layer downstream and therefore invisible to a fix that only touched `guard_rails.py` and (per the brief) N28's prompt.

---

### F5 — The "opening-lap threat discount" rule exists in BOTH N27's and N31's prompts, agrees between them, and has ZERO effect on the field it discounts — `threat_level` is a pure code-derived value with no lap-number term at all

Correction to my own first pass: I initially checked only `strategy_orchestrator.py` for this rule and (wrongly) reported it as single-prompt. `race_situation_agent.py` (N27, the actual producer of `threat_level`) DOES carry the matching prose — I missed it on the first grep because it lives under a differently-worded header ("Strategic guard-rails", no "HARD" qualifier) than N31's ("STRATEGIC GUARD-RAILS (HARD...)"). Re-verified both in full below. The real bug is not a prompt-vs-prompt mismatch here — it's prompt-vs-STRUCTURE: the field both prompts tell their LLM to discount is never actually settable by the LLM.

N27 prompt (`race_situation_agent.py:819-831`):
```
## Strategic guard-rails
- OPENING LAPS (laps 1-3): Race starts naturally inflate both overtake probability
  and SC risk due to first-lap chaos, bunched-up grid, and cold tyres. These are
  normal start dynamics, not genuine strategic threats. When reporting for laps 1-3:
  * Append "opening-lap inflation — discount for strategy decisions" to your reasoning.
  * Consider the effective threat ONE LEVEL LOWER than raw numbers suggest
    (HIGH → treat as MEDIUM, MEDIUM → treat as LOW for strategic purposes).
  * Note that DRS is typically not activated until lap 3, so overtake probability
    in laps 1-2 is inflated by models trained on DRS-enabled data.
```
N31 orchestrator prompt (`strategy_orchestrator.py:1606-1607`), consistent with N27:
```
6. Opening laps 1-3: threat levels from N27 are inflated by start chaos.
   Discount them one tier (HIGH→MEDIUM, MEDIUM→LOW) for decision-making.
```

The field both instructions target, `RaceSituationOutput.threat_level`, is declared `field(init=False)` and computed ENTIRELY in `__post_init__` (`race_situation_agent.py:305,312-318`):
```python
threat_level: str = field(init=False)
...


def __post_init__(self) -> None:
    if (
        self.sc_currently_active
        or self.overtake_prob >= CFG.high_overtake
        or self.sc_prob_3lap >= CFG.high_sc
    ):
        self.threat_level = "HIGH"
    elif self.overtake_prob >= CFG.medium_overtake or self.sc_prob_3lap >= CFG.medium_sc:
        self.threat_level = "MEDIUM"
    else:
        self.threat_level = "LOW"
```
No parameter here is `lap_number`. There is no code path — anywhere in `race_situation_agent.py`, grepped fully — that reads the current lap and adjusts `overtake_prob`, `sc_prob_3lap`, `sc_currently_active`, or `threat_level` for laps 1-3. `threat_level` is `init=False`, so the LLM's structured tool call CANNOT set it directly even if it wanted to; it is derived by the dataclass itself from the two probability floats, which come straight from the N12/N14 model calls, also lap-blind at this point in the pipeline (the DRS-window feature IS lap/neutralisation-aware inside `_build_overtake_features`, per `drs_allowed = not _is_neutralised(...)` at line 917 — but that only fixes the SC/VSC case flagged in that comment, not the "DRS not active until lap 3" case the prompt explicitly calls out one line later).

1. Stated with HARD weight in N31 (explicit header), stated as a "guard-rail" (no HARD qualifier, but structurally identical wording — "Consider the effective threat ONE LEVEL LOWER") in N27.
2. Code enforcement: **none.** `threat_level` is code-derived but the derivation has zero lap-awareness — it is not that code fails to enforce a discount rule that could otherwise apply; it is that the field being discounted is architecturally incapable of reflecting the discount at all. The only place the discount CAN land is the free-text `reasoning` string, which is decorative for any consumer reading the structured `threat_level` enum (MC scoring, `key_risks`, contingencies, `DecisionMemory`).
3. Exception sets: N/A — not an exempt/suspend rule, but the same "prose promises a behaviour the data model cannot express" pattern flagged in `[[feedback_a_guard_that_asserts_nothing]]` project-memory-style findings: a docstring/prompt claim with no structural backing.
4. Downstream effect: on laps 1-3, if `overtake_prob >= CFG.high_overtake` (0.65) or `sc_prob_3lap >= CFG.high_sc`, `threat_level` is unconditionally `'HIGH'` — full stop, regardless of what either LLM's `reasoning` text says. Every consumer of the STRUCTURED field (MC scenario scoring inputs, `key_risks` bullet generation instructions in N31's own rubric, the memory accumulator) sees an undiscounted HIGH. Only a human or another LLM reading N27's free-text reasoning paragraph would ever see the discount — and N31 is itself an LLM being asked, in its own prompt, to re-derive a discount from a field (`threat_level`) it already received un-discounted, with no numeric signal (no `lap_number <= 3` flag, no raw-vs-discounted pair) to hang that re-derivation on beyond re-reading the lap number itself, which N31's prompt DOES also receive (`RACE CONTEXT: ... Lap: {race_state.lap}/{race_state.total_laps}`) — so N31 has the raw ingredients to do this correctly itself, but N27's discount is pure narrative noise structurally.
   Practical mitigating factor found while checking this: the ALREADY-KNOWN `[[project_threat_level_threshold_scale_bug]]` (#450/#665, not re-reported as new here) means `CFG.high_overtake`/`CFG.high_sc` are themselves currently set from raw-scale-tuned thresholds compared against calibrated probabilities, making HIGH effectively unreachable in practice today (0/8171 and 0/1420 real laps observed per that finding). That bug currently masks this one — `threat_level` rarely if ever reaches HIGH regardless of lap number, so the undiscounted-HIGH-on-lap-2 scenario this rule exists to prevent has not been observed to occur. If/when #450/#665 is fixed and HIGH becomes reachable again, this gap becomes live.

Severity: **MEDIUM** (currently latent behind an unrelated, already-tracked bug; would become live and HIGH-severity the moment #450/#665's threshold fix ships, because at that point an opening-lap HIGH would flow undiscounted into MC scoring and `key_risks` with only prose as a (non-functional, single-LLM-hop, easily-dropped) mitigation).

---

### F6 — N27's own prompt self-contradicts on when to call `predict_overtake_tool`, and the RSM-based (production) path never actually gates the tool call by gap at all — only by grid POSITION

N27 prompt, three lines apart, same file:
```
## Workflow
1. If the gap to the car ahead is less than 2.5 seconds, call `predict_overtake_tool` [...]
   (race_situation_agent.py:805)

## Rules
- Always call BOTH tools before drawing conclusions.               (line 814)
- If gap ahead > 2.5s, skip overtake tool and assume P(overtake) = 0.0.   (line 815)
```
Line 814 ("Always call BOTH tools") and line 815 ("skip overtake tool [if >2.5s]") directly contradict each other four lines apart in the SAME prompt — not a cross-file drift, a same-file self-contradiction. An LLM cannot honour both.

Worse: on the production entry point (`run_from_state`, the one CLI/arcade/webapp/`/recommend` all use per the RSM `lap_state` contract in `CLAUDE.md` §6), the 2.5s gap is **never actually measured** before the LLM decides whether to call the tool. `rival_ahead` is computed purely by grid position (`race_situation_agent.py:1366-1370`):
```python
driver_pos = d.get("position")
rival_ahead = (
    next((r["driver"] for r in rivals if r.get("position") == driver_pos - 1), None)
    if driver_pos is not None
    else None
)
```
Then the human-turn message the LLM actually sees (`race_situation_agent.py:1438-1449`) is built from that POSITIONAL result only:
```python
if rival_ahead:
    message = f"... The car ahead is {rival_ahead}. Determine the overtaking probability ..."
else:
    message = f"... No car is within overtaking range (gap > 2.5s). ..."
```
The `else` branch's own wording ("gap > 2.5s") is misleading scaffold text: that branch fires whenever there is no car one position ahead (the leader, or an unresolvable position) — it has **never checked an actual time gap** at that point. Conversely, whenever a positional rival DOES exist, the LLM is told "the car ahead is X" with **no gap value in the message at all** — so an LLM trying to honour "skip if gap > 2.5s" has no way to know the gap without calling the tool it is being told (maybe) to skip. `predict_overtake_tool` itself (`race_situation_agent.py:1104-1165`) has no gap-based short-circuit — it always computes and returns a real N12 probability, however far apart the two cars actually are.

Contrast with `run()`'s own docstring (`race_situation_agent.py:1268`): `rival_ahead — Abbreviation of the car directly ahead. None = skip overtake.` — this documents an assumption that the CALLER already applied the 2.5s gate before calling `run()`. That contract is honoured on the FastF1 session path (external caller's responsibility, not verified in this audit — out of scope of the agent files) but is **not honoured by `run_from_state`, which is the agent's own method**, and which builds `rival_ahead` from position with no gap check of its own.

1. Presented as a definite behavioural rule (imperative "Always" / "skip"), not softened as guidance — but it is internally contradictory, so it cannot be classified as cleanly HARD or advisory; it is broken as written.
2. Code enforcement: **none, and the one code path that could gate it (`run_from_state`) uses a completely different, weaker signal (position, not gap) while echoing gap-based language in its fallback message.**
3. Exception sets: N/A — this is not an exemption pattern, it's a same-prompt contradiction plus a docstring-vs-implementation mismatch (`run()`'s docstring promises a gap gate that `run_from_state` does not perform).
4. Downstream effect: in practice the LLM most likely resolves the contradiction by following "Always call BOTH tools" (the more general, less conditional instruction, and the one matching the tool list description "Call predict_overtake_tool ... predict_sc_tool" style used elsewhere in this codebase's other agents), so `predict_overtake_tool` gets called for every positional rival regardless of actual gap. The N12 model was very likely trained on a full range of real gaps (including large ones, which the training data would show correlate with near-zero overtake probability), so the numeric output is probably still low and roughly sane for a genuinely distant rival — this is a plausibility argument, not a verified guarantee (out of scope to re-run N12 training data here) — but the mechanism protecting against a nonsense call is "the model probably learned gap matters", not the "skip when far apart" rule the prompt claims exists.

Severity: **LOW-MEDIUM** — self-contradictory prose is a real defect (an LLM reading it has to guess which of two adjacent rules wins), but the likely failure mode is graceful (the model's own gap feature probably saves it) rather than a visible, race-affecting error. Flagging because it is exactly the kind of same-file contradiction that "the prompt is the spec" review misses — nobody reads two adjacent bullet points as being in tension when writing prompt prose top-to-bottom.

---

### F7 — N26's two "Strategic guard-rails" (fresh tyres, extended stint) are both prompt-only; `TireOutput.warning_level` is code-derived from `laps_to_cliff_p10` alone and has no `tyre_life` term at all — plus a small cross-agent numeric drift (MEDIUM stint life: 28 vs 30 laps)

N26 prompt (`tire_agent.py:744-755`):
```
## Strategic guard-rails
- FRESH TYRES (tyre_life <= 3 laps): the TCN model extrapolates from minimal data
  and cliff predictions are unreliable. Always report STAY OUT regardless of raw
  model output — no tyre degrades to its cliff in the first 3 laps of a stint under
  normal dry conditions. [...]
- EXTENDED STINT: if tyre_life exceeds the compound's typical race life
  (SOFT ~18 laps, MEDIUM ~28 laps, HARD ~38 laps), the driver is extending
  beyond normal limits. [...] Consider bumping your warning level up by one tier
  (STAY OUT → MONITOR, MONITOR → PIT SOON)."
```
Code — the field these two rules both target (`tire_agent.py:392-406`):
```python
laps_to_cliff_p10: float
...
warning_level: str = field(init=False)


def __post_init__(self) -> None:
    if self.laps_to_cliff_p10 < pit_soon:
        self.warning_level = "PIT_SOON"
    elif self.laps_to_cliff_p10 < monitor:
        self.warning_level = "MONITOR"
    else:
        self.warning_level = "OK"
```
`TireOutput` (checked the full dataclass) carries `laps_to_cliff_p10/p50/p90`, `deg_rate`, `warning_level`, `reasoning`, `gp_name` — no `tyre_life` field feeds `__post_init__`. `pit_soon`/`monitor` come from `CFG.get_cliff_thresholds(gp_name)` (GP/cluster/global — correctly matches the prompt's stated 3/7 global fallback, no divergence there). Neither guard-rail's trigger condition (`tyre_life <= 3`, `tyre_life > {18,28,38}` by compound) appears anywhere in this derivation, nor anywhere else in the file — grepped `predict_tire_deg_tool` and `estimate_laps_to_cliff_tool` (the only two tools, `tire_agent.py:1024,1065`) for any tyre_life-based clamping of the returned P10/P50/P90 or a confidence flag: none exists. Both tools return the model's raw MC-Dropout percentiles for whatever `tyre_life` they're asked about, fresh or not.

1. Both stated as "Strategic guard-rails" (same header pattern N27 and N28 use for their HARD rules) with imperative language ("Always report STAY OUT regardless of raw model output").
2. Code enforcement: **none.** Same structural gap as F5 (N27's opening-lap discount): the field the rule claims to override (`warning_level`) is `init=False` and derived by a pure function of `laps_to_cliff_p10` vs GP-aware thresholds — the LLM's own tool-call arguments include `tyre_life`, but nothing routes that value into `warning_level`'s derivation, and the two tools computing `laps_to_cliff_p10` don't fold tyre_life-based uncertainty into their percentile estimates either (no widened P10 for tyre_life<=3, no shifted threshold for extended stints).
3. Exception sets: N/A (both are one-directional overrides/bumps, not exempt-from-a-bound rules) — same "prose promises a behaviour the data model cannot express" pattern as F5.
4. Downstream: if the model's own P10 estimate for a 2-lap-old tyre happens to compute below the `monitor`/`pit_soon` threshold (plausible on the first lap or two out of a pit stop while the TCN's short window is still mostly padding — see the already-known, separately-tracked N26 zero-padding issue in project memory, `[[reference_f1_domain_knowledge]]`/notebook N09), `warning_level` will read `MONITOR` or `PIT_SOON` on a set that is definitionally too fresh to be near a cliff, and only the LLM's own prose is available to override that — the same single point of failure as F5, in a different agent.

Minor separate finding while cross-referencing this rule against N28's identical-purpose constant: N26's "EXTENDED STINT" text states typical compound life as **SOFT ~18 / MEDIUM ~28 / HARD ~38** laps. N28's `recommend_compound_tool` fallback table (`pit_strategy_agent.py:91`, already quoted in F1) states **SOFT 18 / MEDIUM 30 / HARD 38** — SOFT and HARD agree exactly, MEDIUM disagrees by 2 laps (28 vs 30) between two independently-hardcoded constants describing the same real-world quantity (Pirelli's average MEDIUM stint capacity) in two different agents. Neither cites a shared source constant; grepped for a shared `MEDIUM_STINT_LAPS`-style constant across `src/agents/` — none exists, these are two separate literals.

Severity: **MEDIUM** for the fresh-tyres/extended-stint guard-rails (same class of defect as F5 — a prompt promises a structural safeguard the data model cannot express, on a field with genuine downstream consumers: N31 reads `tire_block` built from `TireOutput.warning_level` directly). **LOW** for the 28-vs-30 MEDIUM-life numeric drift (cosmetic, ~7% relative difference, does not change which compound gets picked in the overwhelming majority of remaining-laps values, only right at the boundary).

---

### F8 — N31 is instructed to quote article numbers from EXACTLY the field N30's own docstring says never to use for citations, because it hallucinates

This is a docstring-vs-code contract violation, not a prompt-vs-prompt one — the strongest single finding in this audit.

`RegulationContext.answer` docstring, `rag_agent.py:74-79` (N30, quoted in full because the warning is explicit and unambiguous):
```
answer:
    LLM-generated summary of the relevant regulation articles — one to
    three sentences, enough for the Strategy Orchestrator to decide
    whether a proposed action is legal without reading the full passage.
    Do NOT use article numbers from this field for citations — the LLM
    may hallucinate them. Use the articles field instead.
```
and again for the safe field, `rag_agent.py:84-88`:
```
articles:
    Deduplicated list of article references extracted from chunk metadata
    (e.g. ["Article 48.3", "Article 55.1"]). Always use this field for
    citations in strategy log entries — chunk metadata is reliable;
    LLM answer text may hallucinate article numbers.
```
N30's own system prompt (`rag_agent.py:118-128`) actively invites the exact failure the docstring warns about — it tells the LLM synthesizing `answer` to *do the citing itself*: `"3. Answer in 2-3 sentences, citing the exact article numbers (e.g. "Article 48.3")."` So `answer` is produced by an LLM that has just been told to free-hand cite article numbers, and the dataclass's own docstring says that's precisely the unreliable half.

Code, `strategy_orchestrator.py:1924-1929` (`_run_conditional_agents`, called once per lap when N30 is active):
```python
reg_out = run_rag_agent(question)
regulation_context = reg_out.answer  # <-- the field the docstring says NOT to cite from
rag_dict = {
    "question": reg_out.question,
    "answer": reg_out.answer,
    "articles": list(reg_out.articles),  # <-- the safe field: captured, but only for rag_dict
    "chunks": [...],
}
```
The function's own docstring (`strategy_orchestrator.py:1872-1880`) states this is deliberate, calling `regulation_context_str` "legacy": *"The legacy `regulation_context_str` is preserved verbatim for the orchestrator's own LLM prompt and for `StrategyRecommendation`, **neither of which depend on the structured shape**."* `rag_dict["articles"]` (the safe, chunk-derived list) is routed only to "downstream consumers that need more than just the answer string (the arcade dashboard surfaces article references and chunk text in its RAG card)" — i.e. the safe field reaches the UI, not the LLM prompt.

`regulation_context` (== `reg_out.answer`) then flows straight into N31's own prompt as `reg_block` (`strategy_orchestrator.py:1506-1509`) and N31 is explicitly told, under its own reasoning rubric (`strategy_orchestrator.py:1658`):
```
3. If regulation_context is present, quote at least one article number.
```
So: N30's LLM free-text-cites an article number into `answer` (a field its own author flagged as hallucination-prone) → the orchestrator hands that exact string to N31 as `regulation_context` → N31's LLM is instructed to quote an article number *from it* into `StrategyRecommendation.reasoning`, the field surfaced to the pit wall / CLI / arcade / webapp as the human-facing justification for the strategy call. The one safe, chunk-metadata-backed `articles` list produced in the very same function call is discarded for this purpose three lines later.

1. HARD rule stated by N30's own docstring ("Do NOT... Always use..." — as unambiguous as prose gets), immediately contradicted by how the very next layer (N31's prompt construction, in the SAME codebase, one function away) uses the data.
2. Code enforcement: **the opposite of enforcement** — the code actively wires the discouraged field into the one place (an LLM-facing prompt asking for a citation) where the risk the docstring warns about actually materializes. The safe field exists, is computed in the same statement block, and is routed elsewhere.
3. Exception sets: N/A — this is not an exemption pattern, it is a straight contract violation with no stated exception.
4. Downstream check: **none.** No validator cross-checks any article number appearing in `StrategyRecommendation.reasoning` against `rag_dict["articles"]` or against `chunks[].article` (the ground-truth metadata). A hallucinated article number from N30, re-quoted by N31, ships to the driver-facing recommendation with the same confident phrasing as a real one — "regulation-grounded" reasoning that may not be.

Severity: **HIGH.** This is a two-LLM-hop hallucination path (N30 free-texts a citation → N31 re-quotes it) built on top of a SAFE alternative that already exists, is already computed, and is already used correctly for a different consumer (the arcade RAG card) in the same code path — so the fix is not "add a new capability", it is "point `reg_block`/N31's prompt at `rag_dict['articles']`/`chunks[].article` instead of `reg_out.answer`" for the citation, and use `answer` only for the prose summary. Regulation citations are exactly the kind of claim (Art. 55.17, the two-compound rule, SC procedure) a user would reasonably trust without independently verifying, which is what makes an unflagged hallucination risk here more consequential than the numeric drifts above.

---

### F9 — Numeric prompt literals duplicating live `data/models/*.json` config values, with no test tying them together (structural risk, not a confirmed live drift — verification blocked by the same `data/models/` gate the audit brief names as the root cause of the original bug)

N28's decision rule #4 (`pit_strategy_agent.py:611`): `"4. If P(undercut_success) >= 0.522 for any rival → recommend UNDERCUT."` — a LITERAL in the prompt string.

The actual comparison the tool performs uses a config value loaded from disk at runtime, not the literal (`pit_strategy_agent.py:230,1194`):
```python
self.undercut_threshold: float = uc_cfg["best_threshold"]  # from model_config_undercut_v1.json
...
verdict = "YES" if calib_proba >= agent.cfg.undercut_threshold else "NO"
```
The tool's own return string DOES report the live threshold correctly (`f'threshold={agent.cfg.undercut_threshold} | ... verdict={verdict}'`), so an LLM that reads the tool's `verdict` field (rather than re-deriving its own YES/NO from the hardcoded "0.522" in decision rule #4 against the raw `P(undercut_success)` number) gets the right answer regardless of drift. But the prompt states BOTH: a literal threshold ("0.522") the LLM could apply itself, AND a tool that supplies an authoritative `verdict`. Nothing forces the LLM to prefer the tool's `verdict` over doing its own comparison against the stale literal.

I could not verify whether `uc_cfg['best_threshold']` currently equals exactly 0.522, because `data/` (including `data/models/`) is gitignored and not present in this checkout (per `CLAUDE.md` §0: "Data/models are NOT in git — they come from Hugging Face Hub on first run"). This is the SAME structural gate the audit brief names as the reason the original SC-exception bug went undetected for a long time ("the only tests covering those bounds sat behind a `data/models/` gate and therefore never ran in CI"). Grepped `tests/audit/test_pit_agent_hardening.py:110-139` — it contains four hardcoded `"threshold=0.522"` strings, but they are canned mock tool-output fixtures for a text-parsing test, not an assertion that `agent.cfg.undercut_threshold == 0.522` or that the prompt's literal matches the live config. No test anywhere ties the prompt's "0.522" to the JSON file's `best_threshold`.

1. Guidance-weight decision rule (not under the "HARD constraints" header — it's rule #4 of the base "Decision rules" list, `pit_strategy_agent.py:607-614`).
2. Code enforcement: the tool computes the correct live comparison and exposes it as `verdict`; the prompt ALSO gives the LLM a redundant, independently-stale-able literal it could use instead. No test enforces the two stay equal.
3. Exception sets: N/A.
4. Downstream check: none. If N16 is ever retrained and `best_threshold` moves (the JSON `model_config_undercut_v1.json` is exactly the kind of artefact that changes across model versions — same file class as the N12/N14 thresholds already found stale in #450/#665), the prompt's "0.522" silently stops matching the tool's own `threshold=` output, and depending on which the LLM leans on for a given call, `UNDERCUT` recommendations could flip at the margin without any code or test noticing — this is structurally the SAME failure shape as #450/#665 (a tuned threshold baked as a constant somewhere other than its source of truth), just not yet confirmed to have actually drifted.

Severity: **LOW (structural / unconfirmed)** — flagged because the failure shape exactly matches this project's dominant, already-proven bug class (tuned thresholds duplicated outside their source of truth, #450/#665, and the SC-exception bug fixed today), and because I could not rule it out — only note that I could not confirm it either way in this environment. Recommend a cheap, permanent fix regardless of current drift status: assert `abs(CFG.undercut_threshold - 0.522) < 1e-6` in a fast unit test that does NOT require `data/models/` (read the JSON directly, no model load) so the two numbers can never silently separate again — and delete the "0.522" literal from decision rule #4, replacing it with "use the tool's own `verdict` field", removing the LLM's option to do the comparison itself against a copy.

---

### F10 — N26's own prompt says a negative degradation rate is real and must be reported as-is; the REPORTED `deg_rate` field honours that (thanks to a prior fix, #477), but the laps-to-cliff P10/P50/P90 computed in the SAME tool call silently discards the sign via `.abs()`

N26 prompt (`tire_agent.py:739-740`, under "## Rules"):
```
- A negative degradation rate means the driver is improving pace on this stint
  (track evolution or fuel load reduction) — this is real, not an error.
```

Two different code paths handle "degradation rate" for two different purposes inside `estimate_laps_to_cliff_tool`, and they disagree on sign:

1. `predict_tire_deg_tool` (`tire_agent.py:1057`) reports the RAW signed rate: `deg_rate = float(feat_df['DegradationRate'].iloc[-1])` — no `.abs()`. `_parse_tool_outputs` (`tire_agent.py:640-671`) takes the **first** regex match for `Degradation rate:` across the message history (`if m and key not in result`), and since the workflow calls `predict_tire_deg_tool` before `estimate_laps_to_cliff_tool` (prompt's own step 1 vs step 2), the value that ends up in `TireOutput.deg_rate` is this raw, correctly-signed one. The regex itself was fixed in #477 specifically because a negative rate — "real and expected per the system prompt" per the inline comment at `tire_agent.py:659-662` — used to fail to parse and silently default to 0.0. **This part is correct and the fix is documented as intentional.**

2. `estimate_laps_to_cliff_tool` (`tire_agent.py:1118`), computing the P10/P50/P90 figures the prompt tells the LLM to base its recommendation on ("Base your recommendation on P10"), uses a DIFFERENT, sign-stripped rate for the actual division:
   ```python
   deg_rate = max(float(feat_df["DegradationRate"].abs().iloc[-1]), 0.001)
   ...
   p50 = min(remaining_budget / deg_rate, cliff_ceiling)
   p10 = min(max(0.0, (remaining_budget - total_std) / deg_rate), cliff_ceiling)
   p90 = min((remaining_budget + total_std) / deg_rate, cliff_ceiling)
   ```
   `.abs()` here is unexplained by any nearby comment — the surrounding comments (`tire_agent.py:1123-1138`) document the `max(..., 0.001)` FLOOR (preventing a divide-by-near-zero blowup that once produced "P50: 27375.2 laps, OK"), but not the sign-stripping. For a genuinely improving tyre with a materially negative rate, this computes `laps_to_cliff` as if the tyre were degrading at the SAME MAGNITUDE it is actually improving at — the opposite of what the prompt's own claim implies (an improving tyre should read as "no cliff in sight", not "cliff arrives at the same pace a degrading tyre of equal |rate| would reach it").

1. Guidance-level prompt claim ("this is real, not an error" — framed as a fact the LLM should trust and report, not a HARD constraint header).
2. Code enforcement: **split** — the field the claim is literally about (`deg_rate`) is correctly signed thanks to #477; the DERIVED metric the prompt tells the LLM to actually decide on (`laps_to_cliff_p10`) silently strips the same sign one function away, with no comment acknowledging the tension.
3. Exception sets: N/A — not an exempt/suspend pattern; this is a sign-information loss inside a single tool call, between two return values (`deg_rate` vs `laps_to_cliff_*`) that are presented to the LLM side by side in the same return string (`tire_agent.py:1156-1159`: "Laps to cliff — P10: ... | ... Degradation rate: {deg_rate:.4f} s/lap ...") as if they were consistently derived from one another.
4. Downstream check: none. Practical impact is bounded by `remaining_budget = max(0.0, threshold - mean_pred)` (line 1121) — a genuinely improving tyre typically also has a low/negative `mean_pred` cumulative-degradation prediction, which keeps `remaining_budget` large and `p50` large regardless of the sign-stripped denominator, so the common case likely still reads "OK" in practice. The failure mode is narrower: a tyre with a LARGE-magnitude negative instantaneous rate (a sharp recent improvement) paired with a `mean_pred` that has not caught up yet would get an artificially SHORT laps-to-cliff estimate — the opposite of what a genuinely improving tyre warrants — purely because the divisor's sign was discarded. I did not find a real-race case in this audit (would need live inference to reproduce, out of scope for a read-only pass); flagging as a verified code-path inconsistency with a plausible, narrow trigger condition, not a confirmed field observation.

Severity: **MEDIUM** — the specific claim in the prompt IS honoured for the field it is literally about (`deg_rate`, thanks to #477), which is reassuring, but the derived metric that actually drives the "STAY OUT / MONITOR / PIT SOON" decision computes through a code path that quietly reintroduces the exact sign-blindness #477 already fixed once, in a sibling function three tool-calls away. This is a strong instance of "the twin that never got the fix" pattern named in this project's own memory: #477 fixed the sign bug in the PARSER; nobody checked whether the COMPUTATION in the neighbouring tool had the same bug, and it did.

---

## Coverage note

Also read in full and checked for HARD/guard-rail-weight prose, with no further findings beyond what's logged above: `pace_agent.py` (N25 — minimal prompt, "never invent numbers, use only tool values", no numeric bounds to cross-check, no divergence found); `radio_agent.py` (N29 — correction/synthesis prompt, no numeric HARD constraints, "Base your response only on the provided data" is generic and matches the structured-output-only design, no code path bypasses it). `strategy_orchestrator.py`'s full ~2426 lines were read/grepped for every remaining numbered guard-rail (1-6, all six covered across F1-F5), the reasoning rubric (covered in F8), the action schema (covered in the F3 addendum), and the post-LLM assembly function `_assemble_recommendation` (confirmed at `strategy_orchestrator.py:2131`, "The action is the synthesis's, always" — no code-side validation of `action`, `compound_next`, or `pit_lap_target` against ANY of the six guard-rails exists on the live path; only `undercut_target` (validated against `live_drivers`) and `expected_stint_end` (clamped via `_clamp_expected_stint_end`, #433) get any post-hoc correction at all).

---

## Severity ranking (all ten findings)

| # | Finding | Class | Severity |
|---|---|---|---|
| F3 | REACTIVE_SC has inverted definitions between N28 and N31 (SC-confirmed vs SC-not-confirmed) | prompt vs prompt (contradiction) | **HIGH** |
| F4 | N31's own minimum-stint guard-rail dropped the SC exception that N28 and today's `guard_rails.py` fix both carry — N31 can silently undo today's fix | prompt vs prompt (stale twin) | **HIGH** |
| F8 | N31 is told to quote FIA articles from N30's hallucination-prone `answer` field instead of the safe `articles` list N30's own docstring says to use | docstring vs code (contract violated by design) | **HIGH** |
| F2 | End-of-race SC exception: N28's prompt still has it, `guard_rails.py` and N31 both correctly dropped it (three-way split, not just prompt-vs-code) | prompt vs prompt vs code (stale twin, but the live path already has the fix) | **MEDIUM-HIGH** |
| F5 | Opening-lap threat discount (N27+N31 agree) has zero effect on `threat_level`, which is `init=False` and lap-blind; currently masked by the unrelated #450/#665 scale bug | prompt vs structure | **MEDIUM** (latent) |
| F7 | Fresh-tyre / extended-stint tire guard-rails are prompt-only; `warning_level` has no `tyre_life` term. Plus a minor 28-vs-30-lap MEDIUM-life constant drift vs N28 | prompt vs structure + minor numeric drift | **MEDIUM** / LOW |
| F10 | `estimate_laps_to_cliff_tool` strips the sign of `deg_rate` via `.abs()` for its P10/P50/P90 divisor, contradicting the prompt's "negative rate is real" claim; the reported `deg_rate` field itself is correctly signed (thanks to #477) | code vs code (the same #477 sign bug reintroduced in a sibling function) | **MEDIUM** |
| F1 | N28/N31's "COMPOUND vs REMAINING LAPS" hard bound (15/12-30/20+) contradicts `recommend_compound_tool`'s own numbers (18/30/38); no downstream validation of `compound_next` at all | prompt vs the tool the prompt tells the LLM to call | **MEDIUM** |
| F6 | N27 self-contradicts ("Always call BOTH tools" vs "skip overtake tool if gap>2.5s"); production `run_from_state` path gates by grid position, not gap, and mislabels its own fallback message as gap-based | prompt self-contradiction + docstring/impl mismatch | **LOW-MEDIUM** |
| F9 | N28's "0.522" undercut threshold is a prompt literal duplicating a live JSON config value with no test tying them together; could not confirm current drift (data/models/ gate) | structural risk, unconfirmed | **LOW** |

---

## Numbered fix list (ordered by value, cheapest/highest-leverage first)

1. **Sync N31's minimum-stint guard-rail (#4) with N28/`guard_rails.py` (F4).** Add the SC-active exception clause to `strategy_orchestrator.py:1602-1603`, matching `pit_strategy_agent.py:641-652`. This is the highest-value fix: it is the one place today's SC-exception fix can be silently undone by the very next pipeline layer.
2. **Resolve the REACTIVE_SC definition conflict (F3).** Pick ONE definition (recommend: N31's — REACTIVE_SC = SC confirmed, since that reads more naturally against N31's own 5-value action enum which never includes REACTIVE_SC as an output anyway) and rewrite N28's "REACTIVE_SC usage" section (`pit_strategy_agent.py:660-666`) to match. Add one sentence to N31's guard-rail #3 clarifying it governs how to READ N28's `action` field, since N31 cannot emit REACTIVE_SC itself.
3. **Fix the RAG citation source (F8).** In `_run_conditional_agents` (`strategy_orchestrator.py:1924-1929`), build `regulation_context` (or a new field alongside it) from `reg_out.articles`/`chunks[].article` for the CITATION the reasoning rubric asks N31 to quote, keeping `reg_out.answer` only for prose summary. Update the docstring at `strategy_orchestrator.py:1878-1880` accordingly — it currently documents the violation as intentional.
4. **Update N28's own prompt to drop the SC exception on the end-of-race bound (F2)**, matching `guard_rails.py`'s Art. 55.17 reasoning and N31's already-correct wording, so all three copies agree. Lower urgency than #1 because the live path (N31) already has the correct behaviour; this is about not contaminating N28's own `reasoning` output and about safety for future edits to `guard_rails.py`/N31 that might "fix" toward N28's stale version instead of the other way around.
5. **Fix `estimate_laps_to_cliff_tool`'s sign-stripped divisor (F10)** at `tire_agent.py:1118`. Either drop the `.abs()` (letting a negative rate legitimately push `p50`/`p10`/`p90` toward `cliff_ceiling`, i.e. "no cliff visible"), or explicitly special-case negative rates to report the ceiling directly with a comment explaining why — either is better than silently mirroring #477's already-fixed bug in a sibling function.
6. **Reconcile the COMPOUND-vs-remaining-laps numbers (F1)** — either change `_STINT_CAPACITY_LAPS` to 15/12-30/20+ to match both prompts' stated HARD constraint, or (preferred, since 18/30/38 are the more defensible Pirelli-sourced numbers per the tool's own docstring) update both prompts' "COMPOUND vs REMAINING LAPS" sections to match the tool, and add a lightweight post-hoc check on `compound_next` vs `remaining_laps` in `_assemble_recommendation` so a violation is at least logged.
7. **Add a data/models/-free unit test pinning `undercut_threshold` to the prompt's "0.522" literal (F9)** — read `model_config_undercut_v1.json` directly (no model load), assert equality, so this class of drift gets a permanent tripwire instead of relying on nobody retraining N16 without also grepping the prompt.
8. **Fix N27's self-contradiction (F6)** — remove either "Always call BOTH tools" or the gap>2.5s skip clause; if the gap gate is kept, thread the actual `gap_ahead_s` into the `run_from_state` message so the LLM has the number needed to honour the rule, and fix the position-only `rival_ahead` computation's misleading "gap > 2.5s" fallback message text.
9. **Reconcile the 28-vs-30-lap MEDIUM stint-life constants (F7)** between `tire_agent.py`'s prompt text and `pit_strategy_agent.py`'s `_STINT_CAPACITY_LAPS`, ideally by extracting one shared constant both agents import.
10. **Consider whether the fresh-tyre/extended-stint tire guard-rails (F7) and the opening-lap threat discount (F5) need a structural home** (e.g. a `lap_context_multiplier` passed into `__post_init__`, or a documented decision that this is intentionally LLM-narrative-only and safe because of X) rather than living purely as prose the structured field cannot reflect — lower priority than the others because both are currently latent/masked, but worth a deliberate decision rather than leaving it implicit.

---

## What I tried to break and could not

- **Tried to find a fourth copy of the pit guard-rails** beyond N28's prompt, `guard_rails.py`, and N31's prompt (e.g. in `no_llm.py`, `decision_modes.py`, the MCP tool schemas, or the arcade/CLI display layer) that might introduce a FOURTH disagreement. Read `no_llm.py`'s `apply_guard_rails` call site and `decision_modes.py`'s own invocation (`src/strategy/eval/decision_modes.py:193-210`) — both call `apply_guard_rails` directly rather than re-deriving the bounds, exactly as `guard_rails.py`'s own docstring instructs ("Anything that needs to MIRROR a rail must import it from here and, better still, call `apply_guard_rails`"). No fourth independent copy exists; the only two "prose" copies are N28's prompt and N31's prompt, and I believe I found and characterised every disagreement between them (F1, F2, F3, F4).
- **Tried to find a code-side safety net that validates N31's final `action`/`compound_next`/`pit_lap_target` against ANY of the six guard-rails** after the LLM call. Confirmed there is none (`strategy_orchestrator.py:2131`, "The action is the synthesis's, always" — deliberate, documented, post-#464 design choice). This is not itself a bug — it is a stated design decision to trust the LLM over a rail that could re-create #464's overreach — but it does mean every prompt-vs-prompt disagreement found here (F2, F3, F4) has ZERO code backstop and is purely a bet on which of two contradictory instructions the LLM happens to weigh more heavily.
- **Tried to find a scale mismatch in the F9/undercut-threshold pattern that I could actually PROVE** (unlike #450/#665, which I could measure against real data). Could not — `data/models/model_config_undercut_v1.json` is not present in this checkout, so F9 is reported honestly as a structural risk with an unconfirmed current value, not a proven live bug. I did not fabricate a number to make the finding look more dramatic.
- **Tried to find the same REACTIVE_SC-style inversion in the other two-agent-pairs** (N26↔N31 on tire guard-rails, N27↔N31 on the SC-probability distinction) — found the parallel structural gap (F5, F7: prompt claims a discount/override, structured field cannot reflect it) but NOT a direct inversion of meaning the way F3 is; N26 and N27's own guard-rail text is internally consistent with what N31 says about the same topics (opening-lap discount numbers match exactly; N26 has no minimum-stint-style rule for N31 to contradict).
- **Tried to reproduce F10's laps-to-cliff sign bug against a concrete real-race number** to upgrade it from MEDIUM to HIGH. Could not, without running live inference against `data/models/` (out of scope for a read-only, no-model-load audit) — reported as a verified code-path defect with a plausible but unconfirmed field trigger, not overstated as a confirmed incident.
- **Tried to find a similar hallucination-laundering pattern in the RADIO agent's `CorrectionEntry`/`span` mechanism** (N29) — did not find one: `CorrectionEntry.span` is explicitly required to be "a verbatim substring from the message" (checkable against the source text), and I found no code path that strips that verification or substitutes free text for it the way F8's `regulation_context` substitutes `answer` for `articles`.


