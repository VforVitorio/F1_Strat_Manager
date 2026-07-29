# FABLE G3 — Adversarial gate: pit rails and prompt/code divergences

**Date:** 2026-07-29 · **Auditor:** adversarial gate (read-only) · **Scope:** 5 claims about
`src/strategy/inference/{engine,no_llm,guard_rails}.py`, `src/agents/{pit_strategy_agent,strategy_orchestrator,rag_agent}.py`
and their surfaces (CLI, arcade, backend/telemetry tab).

**Method:** static trace of every entry point + grep of committed artifacts (goldens, eval
reports, audit files, fixtures) for observed violations. No LLM calls. Findings appended as
verified; each carries file:line evidence. Verdicts: CONFIRMED / REFUTED / OVERSTATED /
THEORETICAL (real in code, not reachable or never observed in practice).

## Checklist

- [x] Claim 1 — CONFIRMED on wiring, THEORETICAL on harm, and it is documented doctrine (#464 / P2B Q5), not an oversight
- [x] Claim 2 — CONFIRMED as a latent, hours-old divergence (`af3a24a`, today); bite untested, no-llm framing was a red herring
- [x] Claim 3 — OVERSTATED: the contradiction is intra-N28 (its own final sentence sides with N31); nothing suppressible can ship
- [x] Claim 4 — CONFIRMED for CLI/backend/N31 rubric; OVERSTATED for arcade (uses the safe `articles` field); partly re-reports RAG-02 (P1)
- [x] Claim 5 — CONFIRMED both; 15-vs-18 narrow and LLM-arbitrated, 13-vs-9 cosmetic (twin of the 2026-07-26 fix)

---
## Groundwork established (executed evidence, all claims draw on this)

**Profile defaults per surface — verified by trace, matches the claim:**

| Surface | Profile | Evidence |
|---|---|---|
| CLI (`f1-sim`) | `rich` unless `--no-llm` | `scripts/run_simulation_cli.py:1750` — `profile = "no-llm" if args.no_llm else "rich"` |
| Arcade | **always** `rich`, no toggle exists | `src/arcade/strategy_pipeline.py:47-48` — hardcoded `profile="rich"` |
| Backend `/simulate` stream | `rich` unless request sets `no_llm` | `src/telemetry/backend/api/v1/endpoints/strategy.py:1534` — `no_llm: bool = False`; `simulator.py:881-897` |
| Backend `/recommend`, MCP tool | orchestrator direct (rich-equivalent) | `strategy_orchestrator.py` entry points |

Note: the webapp (telemetry tab) does not currently call `/simulate` at all — grep of
`src/telemetry/webapp/src` finds `no_llm` only in the generated OpenAPI `schema.ts:1036`
(`@default false`), no caller. The stream's consumers are curl/tests today.

**`apply_guard_rails` call sites — one production, rest tests/eval:**
`src/strategy/inference/no_llm.py:293` (the only in-pipeline call), plus
`src/strategy/eval/decision_modes.py:204` (offline eval harness),
`tests/mc/test_guard_rails.py`, `tests/mc/test_sc_regulatory_rails.py`,
`tests/eval/test_decision_modes.py`. `strategy_orchestrator.py` never imports it (grepped: 0 hits).

**Rich mode has NO post-hoc action enforcement, and that is a documented doctrine, not an
oversight:** `strategy_orchestrator.py:2038-2039` — "**There is no action rail here, and there
must not be one.**" — and `:2131` `action = synth.action`. The only post-LLM validations in
rich mode are field-level: `undercut_target` vs live roster (:2085-2102, #462),
`expected_stint_end` clamp (:2137-2139, #433), `target_lap_time_s = None` under SC (:2149).
The `action` itself is schema-bounded (`_ACTION_VALUES = Literal["STAY_OUT","PIT_NOW",
"UNDERCUT","OVERCUT","ALERT"]`, :252) but otherwise ships as the LLM wrote it.

**The N31 prompt rails (rich mode's only pit bounds), `strategy_orchestrator.py:1594-1609`:**
rule 1 early-race (SC + damage exceptions), rule 2 end-of-race ("~13 positions lost", :1599),
rule 3 REACTIVE_SC-only-when-deployed (:1600-1601), rule 4 min-stint **with no SC exception**
(:1602-1603), rule 5 compound-vs-remaining ("SOFT only if <= 15 laps remain", :1604),
plus ":1608-1609 If a sub-agent recommends an action that violates these rules, override to STAY_OUT".

**REACTIVE_SC is structurally impossible as a final action** — it is absent from
`_ACTION_VALUES` (:252), from `_LLMSynthesis.action` (:315) and from
`StrategyRecommendation.action` (:459), and from `Contingency.switch_to` (:293, same Literal).
MC candidates are STAY_OUT/PIT_NOW/UNDERCUT/OVERCUT only (:682), so on the no-llm path
`best_mc` can never be REACTIVE_SC either — the `"REACTIVE_SC"` entry in
`guard_rails._PIT_ACTIONS` (:37) is exercised only by tests.

---
## Claim 1 — "the deterministic pit rails never run in the mode people actually use"

### Verdict: CONFIRMED on wiring · THEORETICAL on harm · and the claim under-reports that this is DOCUMENTED DOCTRINE, not an oversight

**Wiring — confirmed, stronger than stated.** All defaults verified (see Groundwork table).
Arcade has *no toggle at all* (`strategy_pipeline.py:47-48`), so the owner's arcade runs are
100% rich-profile with zero deterministic action bounds. `apply_guard_rails` has exactly one
production call site, `no_llm.py:293`, and `strategy_orchestrator.py` neither imports nor
reimplements it.

**But "the rails don't run in rich mode" is not a discovered bug — it is a design position
stated in three places the claim does not cite:**

1. `strategy_orchestrator.py:2038-2039` — "**There is no action rail here, and there must not
   be one.**" (the #464 doctrine: rails encode rulebook facts, never strategy opinions).
2. `guard_rails.py:12-13` — "**the prompt is the specification and this file is the copy.**"
   The deterministic module describes ITSELF as the mirror of the prose, existing "so the
   offline no-llm path behaves like the LLM path" (:9-10). By its own charter it was never
   meant to run post-hoc on the rich path.
3. `P2B_ENGINE_DESIGN.md:496` (gate decision Q5) — the engine design *accepted* the guard-rail
   application point as the no-llm profile only.

So the honest framing is: **a design tension, not a wiring hole.** `guard_rails.py:5-6` calls
these "anti-hallucination bounds ... so a language model cannot recommend a lap-2 stop because
it felt like it" — yet they execute only on the path where no language model runs. On the path
with a language model, the anti-hallucination bounds are delivered through the hallucination-
prone channel (prompt prose), and nothing checks compliance (`:2131 action = synth.action`).
Both sentences are true; the first is the striking one; the second is what the repo chose,
on purpose, with reasons written down. Changing it is a design decision for the owner, not a
fix I can call missing wiring.

**Does the LLM obey? — every committed artifact says "never caught disobeying, and almost
never tested where it could".** Executed sweep:

- `documents/audits/AUDIT_ORCHESTRATOR_MEMORY.md` + `AUDIT_DECISION_MEMORY_FOLLOWUP.md` are
  the only committed records of real rich-profile actions: Lusail 2025 laps 35-44 of 57,
  n=8-25 batches. Every recorded action (STAY_OUT / UNDERCUT / PIT_NOW) is inside all three
  bounds: laps 35-44 (early-race bound not applicable), remaining >= 13 (end-of-race bound not
  applicable), and TyreLife at the SC lap 42 = **17 on MEDIUM** (min-stint not applicable —
  measured myself from `data/processed/laps_featured_2025.parquet`, NOR/Lusail: lap 42, stint
  2, MEDIUM, TyreLife 17.0).
- `documents/eval_reports/*` (decision_modes, stint_lengths, projection): all no-llm or
  label-free measurements — zero rich-mode actions.
- No golden, fixture, or eval report contains a rich-mode action at lap < 5, at remaining <= 3,
  or below min stint. **No violation has ever been observed — and no committed run has ever
  probed the lap ranges where a violation could occur.** The absence of evidence here is
  absence of testing, not evidence of compliance.

**One mitigation the claim omits:** in rich mode the LLM also receives the MC table, and the
MC layer's economics (pit-loss ~22-25 s) make an absurd early stop score terribly, so a lap-2
PIT_NOW must override prose rails AND a lopsided numeric ranking simultaneously. The prose is
not the only pressure; it is the only *stated* rule.

**Why did earlier bug hunts not flag it?** They didn't miss it — they *decided* it, three
times, each decision locally sound: (1) #464 removed the prescriptive SC rail and enshrined
"no action rail in `_assemble_recommendation`"; (2) P2B Q5 put `apply_guard_rails` in the
no-llm profile because that path has no prompt to carry the policy; (3) every offline eval
runs `profile="no-llm"` (`decision_modes.py:22`) because measuring LLM mode costs API calls —
so the *measured* mode is the non-default mode, and the composition (bounds enforced only
where no LLM exists) was never anyone's single decision to review.

**Surgical change (if the owner wants enforcement, which is his call, not mine):** a
*verify-and-log* step, not an override: in `engine._run_rich` after `_assemble_recommendation`,
call `apply_guard_rails(rec.action, ...)` and when it would have fired, log a warning and set
a `rail_violation` field on agent_outputs — visible on the arcade card and CLI panel. That
respects the #464 doctrine (no silent action rewriting; the proscriptive check surfaces, the
model keeps the pen) while making disobedience observable for the first time. Must NOT move:
`_assemble_recommendation` itself (its "no action rail" contract), `no_llm.py:293`, anything
in `src/agents/`. Tests: new `tests/engine/` case asserting the rich path computes the check;
`test_engine_threads_every_argument.py` untouched. Alternative (cheaper): one recorded
n=20 probe run at laps 2-4 and final-3 laps to convert "theoretical" into measured, before
deciding any wiring.

**Would a user notice today?** No. The owner's surfaces behave sensibly because the prompt +
MC pressure has, as far as any record shows, been sufficient. This finding matters on the day
it fails, and nothing would currently detect that day.

---

## Claim 2 — "three copies of the minimum-stint rule; the third (N31) lacks the SC exception and can undo today's fix"

### Verdict: CONFIRMED as a latent divergence · the "can undo the fix" half is UNTESTED (no recorded run reaches the state) · one part of the claim's own attack framing is a red herring

**The three copies — verified directly:**

| Copy | Location | SC exception? |
|---|---|---|
| Code | `guard_rails.py:104-109` (`and not sc_active`) | YES — suspended (shipped TODAY, commit `af3a24a` 2026-07-29) |
| N28 prompt | `pit_strategy_agent.py:635-652` | YES — explicit EXCEPTION block with "does not make it correct" nuance |
| N31 prompt | `strategy_orchestrator.py:1602-1603` | **NO** — flat rule, and `:1608-1609` instructs "override to STAY_OUT" on sub-agent violations |

I read `_build_orchestrator_prompt` in full (:1461-1707): no SC clause anywhere reaches rule 4.
Also verified: the RACE CONTEXT block (:1633-1641) carries **no explicit SC-status line** —
N31 learns of a deployed SC only implicitly (sc_3lap=1.00 in the N27 block, `sc_reactive=True`
in the N28 block, N28's reasoning prose, and the N30 regulation text).

**Reachability — the claim's own attack question contains a red herring.** "Given N28 is not
executed on the no-llm path and pit_out is None" is irrelevant: on the no-llm path **the N31
prompt is never built at all** (no LLM call exists), so the stale rule 4 cannot bite there by
construction. The bite is rich-only, and there N28 IS executed under a deployed SC: routing
force-activates N28+N30 on `sc_currently_active` (`strategy_orchestrator.py:608-610`), and the
rich profile runs conditional agents for real. So yes — N31 sees `action=PIT_NOW` in
`pit_block` (:1559) with N28's reasoning invoking the SC exception, while holding a rule text
with no such exception plus an instruction to override.

**The concrete race state where it bites:** rich profile (default on all three surfaces) +
SC deployed + `tyre_life < min_stint` for the fitted compound + a stop being right (e.g. the
second stop is mandatory anyway, or the two-compound rule is unsatisfied late). Narrower than
it sounds: tyre_life below minimum means the driver stopped within the last 8-15 laps, and
N28's own exception text (:645-652) says a recent stopper should usually stay out — but the
window is real (compound rule pending, damaged set, cheap top-up before a restart).

**Untested, and the one measured SC scenario does not cover it:** the only recorded rich-mode
SC runs (Lusail lap 42, AUDIT_ORCHESTRATOR_MEMORY §3.6) had TyreLife 17 on MEDIUM (>= 12,
measured from the parquet) — rule 4 was not binding, so the 7/8 PIT_NOW result neither
confirms nor refutes whether N31 would suppress a below-minimum SC stop. **Latent risk, not
live bug** — real in the prompt contract, unreached by any recorded execution, and its
realisation depends on LLM arbitration between rule 4's flat text and N28's argued exception,
which cannot be settled offline.

**Why did earlier bug hunts miss it?** They could not have: the divergence is **hours old**.
`af3a24a` (2026-07-29, today) added the SC suspension to `guard_rails.py`; N28's prompt
carried the exception; N31's restatement in a different file was outside the fix's blast
radius. This is the repo's dominant defect class ("the twin that never got the fix") occurring
*on the day of the fix* — the third copy was not stale for two years, it became stale this
morning.

**Surgical change (wiring-class, cheap, do it):** edit `strategy_orchestrator.py:1602-1603`
to mirror the other two copies, e.g. append "EXCEPTION: a DEPLOYED Safety Car suspends this
minimum (a queued field makes the stop cheap) — but cheap is not automatically correct; weigh
what the stop surrenders." Must NOT move: rules 1-3, 5, 6; the `:1608` override sentence;
anything in `guard_rails.py` or N28. Tests: none currently pin the prompt's rail text
(`tests/agents/test_orchestrator_prompt` pins the default-memory-block byte-identity — check
whether it snapshots the full prompt; if so it shifts). While editing, also fix the ":1599
~13 positions" line (claim 5) in the same PR — same file, same block, same class.

**Would a user notice?** Only as a suspiciously conservative call under a specific SC + fresh
tyres situation, indistinguishable from legitimate caution without the prompt in hand. Low
probability, moderate cost when it hits (the most valuable stop in racing, refused).

---

## Claim 3 — "N28 and N31 define REACTIVE_SC as opposites"

### Verdict: OVERSTATED — the contradiction is real but mislocated: N28's prompt contradicts ITSELF, its final word AGREES with N31, and no "legitimate N28 call" exists to suppress

**What the texts actually say.** N31 rule 3 (:1600-1601): REACTIVE_SC only when SC deployed;
high prob → STAY_OUT + contingency. N28 has THREE statements, not one:

1. Decision rule 5 (:612): `sc_prob >= 0.30 → recommend REACTIVE_SC`.
2. Guard-rail sentence (:661-664): REACTIVE_SC is "for the rare in-between case where sc_prob
   is elevated but the SC is NOT yet deployed"; under deployed SC "prefer PIT_NOW directly".
3. Guard-rail final sentence (:664-666): "A high sc_prob without confirmation is still a
   contingency — mention it in reasoning and **set ACTION to STAY_OUT unless the SC is
   actually out**."

The A3/F3 table ("exact inverse on both ends") is built from statements 1-2 and **drops
statement 3**, which it had itself quoted. Statement 3 forbids exactly what statements 1-2
permit — so per N28's *own last word*, an unconfirmed-SC REACTIVE_SC is NOT "a legitimate,
spec-compliant call per N28's own text" (A3/F3 point 4's phrase). The prompt's stated
precedence (:616, guard-rails "override any decision rule above") demotes rule 5; the
guard-rail block then disagrees with itself in consecutive sentences. The accurate finding is:
**N28's REACTIVE_SC block is internally incoherent, and N31 sides with its final sentence.**
That is a worse look for N28 and a milder one for the N28-vs-N31 conflict.

**Reachability of the conflicted state — narrower than the claim implies.** N28 never routes
on sc_prob alone: `_decide_agents_to_call` (:578-610) activates N28 only on tyre PIT_SOON,
radio PROBLEM/WARNING, or a CONFIRMED SC. So "sc_prob >= 0.30, unconfirmed, N28 running"
requires an independent tyre/radio trigger to coincide with elevated SC probability. Under a
confirmed SC the prompt shows the SC STATUS banner (no sc_prob line, `_build_pit_prompt`
:558-567), so rule 5's textual trigger ("sc_prob context is provided") is absent and both
prompts agree: PIT_NOW-or-STAY_OUT, not REACTIVE_SC.

**Nothing can be suppressed that could ever ship.** REACTIVE_SC is absent from
`_ACTION_VALUES` (:252) — N31 structurally cannot emit it, ever (the A3 addendum concedes
this). If N28 does emit it, the observable consequences are: (a) the string appears in the
N31 prompt's pit_block and in UI pit cards; (b) `sc_reactive=True` (:1592-1594) — a display
flag, rendered in the N31 prompt (:1566), consumed by no decision code (grepped: one hit);
(c) `recommended_lap` is set (`action != 'STAY_OUT'`, :1598), so `pit_lap_target` backfill
still carries N28's lap even if N31 writes STAY_OUT. The "suppression" outcome — STAY_OUT
plus an SC contingency — is precisely the designed anticipatory mechanism, and it is the one
piece of this system with MEASURED efficacy: the contingency echo converted 1/8 into 7/8
execution when the SC then deployed (AUDIT_ORCHESTRATOR_MEMORY §3.6, p=0.0101). Translating
an anticipatory pit into "STAY_OUT + armed contingency" is arguably *better* engineering than
letting a pre-emptive stop fire on a 0.30 probability.

**Why did earlier hunts miss it?** Prompt-prose contradictions are invisible to every test in
the repo (no test asserts prompt semantics); the schema change that removed REACTIVE_SC from
the final vocabulary made the whole concept inert at the output boundary, so no behavioural
test could ever fail on it; and `sc_reactive`'s glue code keeps the word alive as a flag,
which makes greps look "consumed".

**Surgical change (prompt-engineering class):** resolve N28's self-contradiction rather than
the cross-agent one — delete decision rule 5 (:612) and rewrite the REACTIVE_SC usage block
(:660-666) to one coherent position. The coherent position that matches the schema, the
routing, and the measured mechanism is N31's: REACTIVE_SC out of the recommendation
vocabulary, anticipation expressed as a contingency. That means deprecating REACTIVE_SC from
`_parse_agent_summary`'s regex (:510) *last*, after confirming no formatter depends on the
string (the arcade pit card renders `pit_out.action` verbatim). Must NOT move: `sc_reactive`'s
flag semantics (prompt display), `_PIT_ACTIONS` in guard_rails (its REACTIVE_SC entry is
test-covered and harmless). Design decision for the owner: whether REACTIVE_SC should remain
an N28-internal vocabulary item at all.

**Would a user notice?** Practically never: the word REACTIVE_SC can only surface on the N28
card in a narrow coincidence state, and the final recommendation is unaffected in every
recorded run. Lowest-ranked confirmed finding.

---

## Claim 4 — "regulation citations come from the field documented as unsafe"

### Verdict: CONFIRMED for the CLI, the backend stream, and N31's own citation rubric · OVERSTATED for the arcade (it already uses the safe field) · and this is partly a RE-REPORT of a filed P1

**The documented contract — verified.** `rag_agent.py:74-88`: `answer` is for deciding
legality; "Do NOT use article numbers from this field for citations — the LLM may hallucinate
them. Use the `articles` field instead" (:78-79); `articles` comes from chunk METADATA
(reliable, :84-88). `_run_conditional_agents` sets `regulation_context = reg_out.answer`
(:1925) and builds `rag_dict` including `articles` (:1926-1929).

**Where the unsafe path is real:**

- **N31's own prompt crosses the documented line, explicitly.** The rubric (:1658): "If
  regulation_context is present, quote at least one article number" — and the in-code comment
  at :1666-1670 says the article number "should come from that context", i.e. from the answer
  string, the exact field whose article numbers its producer documents as hallucination-prone.
  The answer text is itself LLM-generated under a system prompt that *instructs* citing
  article numbers (`rag_agent.py:123`), so the numbers in it are generated-then-requoted.
- **The answer string is also presented as a HARD constraint:** reg_block (:1506-1510) —
  "REGULATION CONSTRAINT (hard — exclude non-compliant actions)" wraps the unvalidated string.
- **`articles` never reaches the recommendation.** `StrategyRecommendation` has no articles
  field (:458-520); both orchestrator entry points discard `rag_dict` (`_rag_dict`, :2203,
  :2381). Only the engine's `agent_outputs["rag"]` carries it (engine.py:388).
- **CLI:** renders `outs["regulation_context"]` as the RAG text (`run_simulation_cli.py:1774`)
  and `_style_reasoning` paints "regulation articles (Article 30.5, Art. 32.4(b)) — **bold
  amber**" (:713) inside reasoning/answer text — authoritative styling applied to potentially
  hallucinated numbers.
- **Backend stream:** ships `rec.regulation_context` verbatim on every lap decision
  (`simulator.py:457`); articles dropped.
- **Sanitisation between: none.** Verbatim at :1925 → prompt :1508 → `:2155` → UIs.

**Where the claim overstates:**

- **The arcade — the owner's main surface — already does it right.** `window.py:327` prefers
  the structured payload; `format_rag` (`agent_formatters.py:559`) renders the citation line
  from `rag.get("articles")` — the SAFE metadata field. The unsafe answer appears only as a
  70-char snippet above it. So "the UI presents it as authoritative" is accurate for CLI and
  backend, not for the arcade rag card.
- **"The field documented as unsafe" is citation-scoped, not wholesale.** The docstring
  positively documents `answer` as the input for legality decisions (:75-77). Feeding it to
  N31 as decision context is BY DESIGN; only the citation use breaches the contract.
- **Partly known.** `AUDIT_RAG_LAYER.md` RAG-02 (**P1**, an earlier hunt) already filed
  "evidence and answer can diverge" with the proposed trace-level test ("agent answer citing
  an article absent from retrieved chunks", :98). The novel part today is tracing the
  *consumption*: the N31 rubric instruction and the per-surface rendering.

**Why did earlier hunts miss it?** Mostly they didn't (RAG-02, filed, backlog). What survived
is the consumption side: the warning lives in a docstring in `rag_agent.py`, the instruction
that violates it lives 1650 lines into a different file's f-string, and doc-accuracy audits
compared docs to code, not docstring-contracts to prompt text. Plus the arcade — the surface
that gets screenshotted — renders the safe field, so every visual check looked correct.

**Surgical change (wiring-class, small):**
1. `_build_orchestrator_prompt`: pass `rag_articles: list[str]` through and render a line
   "Verified article references (from regulation metadata): Article 55.7, ..." inside
   reg_block; change rubric line :1658 to "quote at least one article number **from the
   verified reference list**". Callers: engine.py `_run_rich` has `rag_dict` in scope
   (:277-284); the orchestrator's own entry points must stop discarding `_rag_dict`.
2. Optionally add `regulation_articles: list[str]` to `StrategyRecommendation` (additive,
   default `[]`) so the CLI/backend can render citations from metadata like the arcade does.
   That is a schema change — owner's call, since the DTO is consumed across submodule
   surfaces.
Must NOT move: `regulation_context`'s role as decision context; `format_rag`'s existing
articles line. Tests: `tests/agents/test_orchestrator_prompt` (prompt bytes shift).

**Would a user notice?** This is the highest-visibility confirmed finding: article numbers in
`reasoning` and the amber-styled CLI panel are exactly the tokens a reader treats as ground
truth, on every routed lap, on the default profile — and a wrong one is unfalsifiable to a
user without the PDFs. That the mechanism is *plausible-looking wrongness* rather than a
crash is what makes it worth fixing first.

---

## Claim 5 — "live numeric contradictions: SOFT <= 15 vs capacity 18; '~13 positions' vs '~9, not 13'"

### Verdict: CONFIRMED both, textually · impact: the 15-vs-18 is a real but narrow LLM-arbitrated contradiction; the 13-vs-9 is cosmetic for decisions and wrong-as-teaching · one wiring detail nearby is dead code worth knowing about

**SOFT 15 vs 18 — verified, live in the prompt context, narrow window.**
`_PIT_STRATEGY_SYSTEM_PROMPT:654-655` (under the HARD-constraints header at :616): "SOFT:
recommend only if remaining laps <= 15". N31 rule 5 (:1604) agrees: "<= 15". The tool
(`recommend_compound_tool`, :1268-1271) picks the smallest compound with
`_STINT_CAPACITY_LAPS[c] >= laps_remaining`, SOFT=18 (:91) — so at `laps_remaining` 16-18
(current compound != SOFT) the tool answers "Recommended: SOFT" while both prompts forbid
SOFT above 15. Same LLM context, two authorities, no tiebreaker stated.

**How it would ship — with a twist the A-series did not surface.** In
`PitStrategyOutput` assembly (:1599): `compound_recommendation = compound_rec or
parsed.get('compound_recommendation') or 'MEDIUM'` — but `_parse_agent_summary` (:498-517)
can NEVER return a falsy compound (its own fallback is `'MEDIUM'`), so **the tool-parsed
fallback and the final `'MEDIUM'` in that chain are dead code**: the shipped
`compound_recommendation` is always the N28 LLM's summary line (or its parser default). The
contradiction therefore ships only if the LLM echoes the tool over the rail —
nondeterministic, unobserved in any committed artifact (no recorded lap sits in the
16-18-remaining window with a compound decision). Real, narrow, unmeasured.

**~13 vs ~9 — verified, and it is the same twin-fix pattern as claim 2, three days older.**
N28 was corrected in `4000f84` (2026-07-26) with measured gap medians: ":629-633 worth ~9
positions, not 13: 13 positions is the Safety Car bunched-field figure (median gap 1.4795s)".
N31 (:1599) still teaches "~13 positions lost" — the green-flag rule citing the SC-field
number the sibling prompt explicitly debunks. Decision impact: cosmetic — both copies gate on
the identical trigger (remaining <= 3, cliff P10 < 2), and the number is rationale, not
threshold. Worst case it inflates the LLM's internal cost model for late stops in states the
rule does not even cover (remaining 4-6); no observable case exists. It IS a live
documentation error inside a prompt that instructs the model to explain its reasoning to
users.

**Why did earlier hunts miss these?** No mechanism compares prose numbers to code constants —
that tooling class (`AUDIT_A5`) was invented today. The ~13 got stale via the 2026-07-26 fix
touching only `pit_strategy_agent.py` (twin-not-fixed, again). The 15-vs-18 has existed as
long as both texts; every prompt review read each file alone, and the tool's output format
("Recommended: SOFT | 1-stop viable") looks authoritative enough that nobody diffed it
against the rail two hundred lines up in the same string.

**Surgical change:** (a) one-line: `:1599` "~13" → "~9 (green-flag median gap 2.226 s)" —
bundle with claim 2's edit, same block. (b) The 15-vs-18 needs a DECISION first, not an edit:
either the prompts adopt the capacity table (state "SOFT only if <= 18 remain") or the tool
adopts the prompt margin (compare against `capacity - safety_margin`). The A5 fix suggestion
(derive the prompt line from `_STINT_CAPACITY_LAPS` at prompt-build time) is the right shape —
single source, restated nowhere. Owner's call on which number is the truth; the capacity
table is the one the eval tier measures against real stints (`stint_lengths.py`), which
argues for 18 as the physical bound and 15 as an editorial margin that should then SAY it is
a margin. Must NOT move: `_STINT_CAPACITY_LAPS` values themselves (N16-adjacent, measured),
N31 rule 5's MEDIUM/HARD numbers (consistent with the table's intent).

**Would a user notice?** The 13/9 never. The 15/18: only as a SOFT recommendation with ~17
laps left that contradicts what the same system would say at 15 — visible to a careful user
reading compound advice, not wrong-looking to anyone else.

---

## What I tried to break and could NOT

1. **"Rich is the default everywhere" — attacked, held.** I hunted for any surface or config
   path that flips the default: CLI arg parsing (:1750), arcade (no toggle exists at all),
   backend request schema (`no_llm: bool = False`, :1534), webapp (never calls /simulate —
   checked all of `src/telemetry/webapp/src`). No counterexample.
2. **"apply_guard_rails has exactly one production call site" — attacked, held.** Full-repo
   grep: every other call is tests or the offline eval tier (`decision_modes.py`). No hidden
   consumer in agents, arcade, or backend.
3. **A committed artifact showing the LLM violating a bound — searched, none exists.** Memory
   audits, follow-up, eval_reports, G1/G2, fixtures. Equally: none probing the bound regions,
   which is why Claim 1's harm stays "theoretical" instead of "refuted".
4. **The claim-2 divergence being pre-existing (and thus already hunted) — refuted myself.**
   `git log` pins the SC suspension to `af3a24a`, today. The N31 text was consistent with the
   (SC-blind) code until this morning. The finding is genuinely un-huntable before today.
5. **The min-stint scenario hiding in the measured SC runs — checked, it does not.** Parquet
   query: NOR Lusail lap 42 = MEDIUM, TyreLife 17. The 7/8 flip neither proves nor disproves
   rule-4 suppression; I could not use the recorded data to settle claim 2 either way.
6. **`articles` being unavailable at the consumption point (would have refuted claim 4's
   fixability) — refuted.** `rag_dict` with articles exists at :1926-1929 and inside the
   engine's agent_outputs; the arcade already renders it. Availability is proven by a
   shipping consumer.
7. **REACTIVE_SC shipping to any user-facing final action — could not construct a path.**
   Schema Literal (:252), MC candidate set (:682), and no-llm's argmax source all exclude it.
   Only the N28 card and the N31 prompt can ever display the token.
8. **The arcade rag card presenting hallucination-prone citations as authoritative — could
   not confirm; the opposite is true** (`agent_formatters.py:559` uses metadata articles).
   This is the main overstatement I found in claim 4.

## Ranked summary (by whether a user would ever notice)

| # | Finding | Verdict | User-visible? |
|---|---|---|---|
| 1 | Claim 4 — hallucination-prone article numbers styled as authoritative (CLI amber, reasoning text, backend stream) + N31 rubric instructing citation from the unsafe field | CONFIRMED (arcade excepted; partly re-report of RAG-02/P1) | **Yes — every routed lap, default profile, unfalsifiable to the reader** |
| 2 | Claim 1 — pit bounds unenforced in the default mode | CONFIRMED wiring / THEORETICAL harm / documented doctrine | Only on the undetectable day the LLM disobeys; no observed case |
| 3 | Claim 2 — N31's min-stint copy lacks today's SC exception and instructs overriding | CONFIRMED, latent, hours old | Rare SC state; reads as ordinary caution |
| 4 | Claim 5a — SOFT <= 15 (two prompts) vs capacity 18 (tool), LLM arbitrates | CONFIRMED, narrow window (16-18 laps) | Marginal compound-advice inconsistency |
| 5 | Claim 5b — "~13 positions" vs corrected "~9, not 13" | CONFIRMED, cosmetic | No |
| 6 | Claim 3 — REACTIVE_SC "opposites" | OVERSTATED — real incoherence, mislocated (intra-N28), nothing suppressible can ship | No |

**Fix order by value/risk:** (1) claim 4 step 1 (prompt-side verified-articles line — small,
wiring); (2) claims 2 + 5b together (one prompt block, two line edits); (3) claim 5a decision
(single-source the compound numbers); (4) claim 1 decision (verify-and-log rail, or a
measured probe run first — owner's call, doctrine is involved); (5) claim 3 cleanup of N28's
REACTIVE_SC block (lowest value, do opportunistically).
