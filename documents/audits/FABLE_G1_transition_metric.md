# FABLE gate G1 — the transition metric (commit `a474ac8`, PR #755, issue #752)

Adversarial gate over the STAY_OUT → PIT transition metric that replaced the retired
first-pit-lap metric in `src/strategy/eval/decision_modes.py`. Written incrementally:
each finding is appended the moment it is confirmed with executed evidence. Nothing in
this file was written from reading alone.

- Gate run: 2026-07-30, on `dev` at `0d52440` (merge of `a474ac8`).
- Rule: no repository file modified except this report, with the single sanctioned
  exception of temporary mutants for claim E, restored and verified restored.

## Checklist of claims under attack

- [x] **A** — `_pit_decision_lap` locates a genuine transition, no off-by-one — SOUND on boundaries/oscillation; but see A1/A1b (bucket prose names the wrong mechanism) and A3 (rail release)
- [x] **B** — replay-span widening — VERIFIED necessary and effective on real data (A2); but its absence is untested, see E1
- [x] **C** — `no_boundary` inside `eligible` — HOLDS everywhere a denominator lives (executed constructions below)
- [x] **D** — width-sensitivity claim — CONFIRMED exactly (7 scored at both, 6 same, NOR 42→38, HAM revealed); phrasing "the two that move" conflates a bucket change with a lap change
- [x] **E** — mutation testing — **M3 (narrowed replay span) leaves ALL 34 tests green while changing the real measurement → HIGH**
- [x] **F** — the twin — no second first-occurrence scorer found in code; the twin is the COMMITTED REPORT itself (F1, HIGH)

## Findings

(appended below as confirmed)

### F1 — HIGH — the committed eval report still publishes the retired metric; acceptance criterion 4 of #752 is unmet

`documents/eval_reports/decision_modes.json:48` still says `"mean_signed_error": -3.3` over
90 scored stops, with buckets `{closing_laps: 4, min_stint: 22, no_call_in_window: 78, opening_laps: 4, scored: 90}`
— **no `no_boundary_in_window` bucket exists in the committed artifact**. The md header says
`generated 2026-07-30T08:37:07+00:00` on harness `1f0ec9d`; the fix commit `a474ac8` is
21:06 the same day and touched only `src/strategy/eval/decision_modes.py` and
`tests/eval/test_decision_modes.py` (verified via `git show a474ac8 --stat`).
`git log -- documents/eval_reports/decision_modes.md` shows the last touch was `009cbe2`
(a docs commit), and `git status documents/eval_reports/` is clean — nothing regenerated,
nothing pending.

Issue #752's acceptance list includes: *"The committed report regenerated, with the decline
rate and coverage preserved"*. The PR that says `Closes #752` did not do it. The repo's
committed truth is therefore still the number the commit message itself calls "three readings
of the window", in the exact artifact a reader of `documents/eval_reports/` would quote. This
is the same defect class the commit lectures about in its last paragraph ("a doc that outlives
the code it documents"), applied to the report instead of the docstring.

Failing scenario: anyone running `f1-eval` consumers, the docs site, or a thesis section off
the committed `decision_modes.json` quotes −3.30 as the system's timing error today.

### A1 — MEDIUM — an evaluation gap inside the window is bucketed as `no_boundary_in_window`, whose documented meaning is then false

`src/strategy/eval/decision_modes.py:500-504` assigns `no_boundary_in_window` whenever
`_pit_decision_lap` returns None but `_asks_to_stop` is True. That predicate does not
distinguish "pit on every lap offered" from "STAY_OUT for most of the window, one lap never
evaluated, then PIT". Executed (transcribing the bucketing at :493-505 exactly):

```
# 3. pit only at the LAST window lap, lap 31 never evaluated
actions = {26..30: STAY_OUT, 32: PIT_NOW}      -> no_boundary_in_window
# 4. STAY_OUT lap 27, gap at 28, PIT 29..32
actions = {26,27: STAY_OUT, 29..32: PIT_NOW}   -> no_boundary_in_window
```

Both land in the bucket that `_render_table` (decision_modes.py:573-580) explains as *"The
stack asked to stop on every lap offered, including the first, so there is no STAY_OUT ->
PIT transition to locate: it was already committed before the window opened"* and that the
docstring (:359-361) glosses as *"the call came earlier than we asked"*. In scenario 3 the
stack demonstrably declined for five consecutive laps and its only pit ask is the LAST lap
of the window — the call came *later* than almost everything we asked, not earlier. The
mechanism named by the prose is wrong for every gap case, which is this repo's documented
bug class (a comment naming the wrong mechanism). The scoring itself stays conservative
(no lap is invented), so this is an interpretation/reporting defect, not a scoring one —
but `no_boundary` is the bucket the next analysis will read as "already committed", and gap
cases silently inflate it with the opposite situation.

Reachability: requires a lap inside [window_low, window_high] with `lap_inputs` returning
None (missing position / lap_number) while neighbours evaluate — see A2/RE evidence below on
whether real replay spans contain such holes.

### A1b — MEDIUM — on real data, 4 of 4 inspected `no_boundary` stops do NOT match the bucket's documented mechanism

The gap scenario above turned out to be the minor half. The real-data sweep (2025 Monza,
w=5, pristine code, actions recorded by wrapping `_decisions_in_window`) shows what the
bucket actually contains. The four `no_boundary_in_window` stops at w=5:

```
GAS stop 49, window [44,53], pred(43)=UNDERCUT:
    44..49: UNDERCUT   50..52: STAY_OUT   53: never evaluated
STR stop 49, window [44,53], pred(43)=UNDERCUT:  same shape as GAS
HAM stop 38, window [33,43], pred(32)=PIT_NOW:
    33..37: PIT_NOW    38..43: STAY_OUT
PIA stop 45, window [40,50], pred(39)=PIT_NOW:
    40: PIT_NOW  41..44: UNDERCUT  45: PIT_NOW   46..50: STAY_OUT
```

None of the four "asked to stop on every lap offered, including the first"
(`_render_table`, decision_modes.py:574) — every one of them was committed when the window
opened and then **withdrew mid-window**. HAM flips to STAY_OUT on the exact lap the team
actually stopped. The docstring gloss "the call came earlier than we asked" (:27-29, :359-361)
does hold for all four (each was already asking at `window_low` and at its evaluated
predecessor); the render prose's stronger claim ("every lap offered") is false for 4/4.
This is the repo's documented defect class — prose naming the wrong mechanism — sitting in
the artifact every reader of the report will use to interpret the bucket. It also means the
bucket silently merges "committed and stayed committed" with "committed, then withdrew",
and the withdrawal lap (which for HAM was the real stop lap — arguably signal) is discarded.

Caveat found while verifying: GAS/STR's STAY_OUT tail at laps 50-52 is NOT model withdrawal —
`total_laps=53`, so laps 50+ have `remaining <= 3` and the closing rail forces STAY_OUT
(recorded actions are post-rail, `src/strategy/inference/no_llm.py:293`). HAM's (38-43,
15 laps remaining) and PIA's (46-49) are genuine model withdrawals.

### A2 — verified sound — replay-span integrity on real data (claim B.i-B.ii)

From the recorded spans at all three widths, every (race, driver) pair at 2025 Monza:
the widened lap `low` **is** present in `actions` — `evaluated=True` for all pairs, all
widths. The only holes inside `[low, high]` are trailing-edge: ALO laps 25+ (retired after
his lap-24 stop; his lap-24 stop is `min_stint`, lap-20 stop scored `no_call` with laps
15-24 evaluated) and lap 53 for GAS/STR/OCO (lapped cars have no final-lap row). No
mid-window transient hole was observed, so finding A1's gap scenario is constructible but
was not reached in this sample.

Scoring range vs replay range (claim B.ii): they are not the same range —
`low = max(1, min(stop_laps) - DECISION_WINDOW_LAPS - 1)` (:469) vs
`window_low = max(1, stop_lap - DECISION_WINDOW_LAPS)` (:481). The one case in the entire
sweep where `chosen == window_low` (HAD, stop 32, w=5, chosen 27) has predecessor lap 26
evaluated as STAY_OUT — a genuine transition — and the acid test: **at w=10, when the window
edge moves to 22, HAD's chosen lap stays 27.** The edge report is gone not just by
construction but demonstrably on data.

Clamp at lap 1 (claim B.iii): hermetic probes — window [1,8] all-PIT from lap 1 returns
None → `no_boundary_in_window` (lap 1 has no predecessor and is correctly unjudgeable, never
scored); `{1: STAY_OUT, 2: PIT}` returns 2. No unjudgeable-but-scored path exists.

### A3 — MEDIUM (structural, not observed in sample) — a rail release can still manufacture a "transition" at a constant lap

The recorded action per lap is post-guard-rail (`no_llm.py:293` applies `apply_guard_rails`
to the MC's best action). The opening rail forces STAY_OUT on every lap `< _NO_PIT_BEFORE_LAP
= 5` (guard_rails.py:92, no SC). So for a real stop on laps 6-10 (not itself guard-blocked)
whose MC is pit-eager from the opening laps, the recorded sequence is
`STAY_OUT(rail) x4, PIT(5), ...` and `_pit_decision_lap` scores **lap 5 — a constant of the
rails, not a decision of the model**. That is the retired defect's exact shape (a harness/rail
boundary wearing a timing estimate) relocated from `window_low` to `_NO_PIT_BEFORE_LAP`.
The transition detector cannot distinguish rail release from model transition because the
rails are applied before recording. Not observed at Monza (the only early stop, LAW lap 9,
has a stack that never asks to pit: laps 1-14 all STAY_OUT at every width — executed dump in
the sweep). The closing rail cannot create a STAY→PIT transition (it only forces STAY late),
but it does manufacture the fake "withdrawals" noted in A1b.

### D — verified, with the commit's numbers CONFIRMED and one phrasing corrected

Re-derived at three widths (3, 5, 10) on 2025 Monza, pristine code, one process, actions
recorded. Aggregates:

| | w=3 | w=5 | w=10 |
|---|---|---|---|
| scored | 6 | 7 | 8 |
| no_boundary | 5 | 4 | 3 |
| no_call | 7 | 7 | 7 |
| guard-railed | 2 | 2 | 2 |
| eligible | 20 | 20 | 20 |
| mean signed error | −0.33 | −1.29 | −2.50 |
| exact (of scored) | 83.3% | 71.4% | 62.5% |

- **The commit's "6 of the 7 stops scored at both w=5 and w=10 report the same chosen lap"
  is exactly right**: 7 stops scored at both, 6 identical (TSU 19, ALB 41, COL 33, SAI 30,
  RUS 27, HAD 27), 1 moved (NOR 46: chosen 42 → 38). The commit's "the two that move" counts
  NOR plus HAM 38 (`no_boundary` → scored/31 at w=10) — HAM is a bucket change, not a
  chosen-lap change among stops scored at both; sloppy phrasing, correct facts. Both
  characterisations ("wider window REVEALS a transition" for HAM, "model oscillates and both
  are real transitions" for NOR) verified against the recorded actions.
- w=3 adds a third instance of the reveal mechanism: HAD 32 is `no_boundary` at w=3 and
  scored 27 at both w=5 and w=10.
- **Which property holds**: every reported lap is a genuine evaluated STAY→PIT boundary, and
  the single window-edge coincidence (HAD) survives the edge moving. **Which property does
  NOT hold**: the aggregate `mean_signed_error` still slides monotonically with width
  (−0.33 / −1.29 / −2.50) — no longer by edge-pinning, but because a wider window reveals
  earlier transitions (HAM) and earlier oscillations (NOR: 44 → 42 → 38 at w=3/5/10, each a
  real transition). For an oscillating stack the metric reports *the first transition after
  window_low*, which is still a window-dependent choice among real transitions. The commit
  and module docstring state this limitation explicitly and do not over-claim; but nobody
  should quote `mean_signed_error` without its width, and the conditional-on-scored rates
  (exact 83% → 71% → 62%) move with width purely through set composition — cross-width or
  old-vs-new comparisons of `exact` are apples-to-oranges (old 5/11=45.5% vs new 5/7=71.4%
  at w=5 counts the SAME five exact stops).

### C — verified sound — the denominator sweep

`eligible` (decision_modes.py:164) = scored + guard_railed + no_call + no_data + no_boundary.
Executed constructions:

```
all-no_boundary (20 stops):  eligible=20, scored_share=0.0, coverage=masked, exact=0.0
3 scored + 9 no_boundary:    eligible=12, scored_share=0.25, coverage=masked
render:                      "| Stops scored | 3 of 12 (25.0%) |"
```

A run where every stop is `no_boundary` reports `masked`, never high coverage. Every
denominator audited: `scored_share` (:188) uses `eligible`; `coverage_verdict` (:241-243)
uses `eligible` and `scored_share`; the render's "Stops scored X of Y" uses `eligible`; the
JSON payload's `eligible` comes from the same property and its `buckets` dict carries
`no_boundary_in_window` explicitly. The only rates NOT over `eligible` are
`exact`/`within_one`/`within_two`/`mean_*` (:167-184), deliberately conditional on the
scored subset — correct, but see the LOW note below on how they read.

LOW — the render's Meaning column for `Exact lap` says "chose the lap the team chose"
without stating the denominator is the scored subset. After #752 that matters more: the new
bucket removes precisely the committed stops from that denominator, so exact% jumps
45.5% → 71.4% at w=5 while the COUNT of exact stops (5) is unchanged. A reader comparing the
regenerated report's exact% against the old committed one will read improvement that is
bookkeeping.

### E — mutation testing: one mutant survives the entire suite

Method: each mutant applied to `src/strategy/eval/decision_modes.py`, suite run, file
restored via `git checkout --` and verified with an empty `git diff` after each mutant
(final verification at the bottom of this report). Hermetic runs:
`uv run pytest tests/eval/test_decision_modes.py -m "not data" -q`. M3 ran the FULL file
including the Lusail data test.

| Mutant | Change | Result |
|---|---|---|
| M1 | `_pit_decision_lap` returns the first pit lap regardless of predecessor (the mechanical revert) | 3 red: `test_a_stack_already_committed_has_no_decision_lap`, `test_an_unevaluated_predecessor_cannot_witness_a_transition`, `test_asks_to_stop_separates_declining_from_being_already_committed` — caught |
| M2 | `no_boundary` removed from `eligible` | 1 red: `test_the_new_bucket_counts_toward_eligible_so_the_share_is_not_inflated` — caught |
| M3 | replay span narrowed back (`- DECISION_WINDOW_LAPS - 1` to `- DECISION_WINDOW_LAPS`) | **0 red — `34 passed in 192.73s`, data test included** |
| M4 | missing predecessor treated as stay-out (None check dropped) | 1 red: `test_an_unevaluated_predecessor_cannot_witness_a_transition` — caught |
| M5 | `_asks_to_stop` returns False always | 2 red: `test_a_stack_already_committed...`, `test_asks_to_stop_separates...` — caught |

### E1 — HIGH — the replay-span widening has zero test coverage: a suite-green, non-equivalent mutant

M3 reverts the exact line the commit message calls essential ("Without that the first lap of
every window would be permanently unjudgeable and the edge report would return through the
back door"), and **every one of the 34 tests stays green, including the Lusail data-tier
test** (`34 passed in 192.73s`). The mutant is NOT behaviorally equivalent — executed on
2025 Monza at w=5 with M3 applied:

```
pristine: scored=7, no_boundary=4, mean_signed_error=-1.286
M3:       scored=6, no_boundary=5, mean_signed_error=-0.667
   HAD actual=32: scored/chosen=27  ->  no_boundary_in_window
```

HAD's genuine transition at `window_low` (predecessor lap 26 = the widened lap, evaluated
STAY_OUT) becomes unjudgeable, and the published mean signed error moves by 0.62 laps —
roughly half its value — with the suite green. This is this work's own instance of the
defect class it fixes: the mechanism the fix depends on is asserted by a comment, not by a
test. The hermetic tests cannot see it because the span is computed inside
`measure_decision_agreement` (:469) and every hermetic test hands `_pit_decision_lap` a
pre-built dict; the data test asserts only non-emptiness, `eligible == len(verdicts)`,
`races == 1` and None-offsets for non-scored — all invariant under the narrowing. A hermetic
test that drives `measure_decision_agreement` with a stubbed `_decisions_in_window`
(asserting the span it receives starts one lap before the earliest window, or that a
HAD-shaped action dict scores rather than lands in `no_boundary`) would kill M3.

### E2 — MEDIUM — the two headline tests do not discriminate the defect they advertise, and one docstring's factual claim is false

Both survive M1 (the mechanical revert):

- `test_the_decision_lap_is_the_transition_not_the_earliest_call`
  (tests/eval/test_decision_modes.py:156): its fixture's earliest pit call (lap 29,
  predecessor 28 STAY_OUT) IS the transition, so the old helper also returns 29. Executed
  under M1: green. The name promises a discrimination the fixture cannot perform (a
  discriminating fixture needs the first pit call to lack an evaluated non-pit predecessor,
  e.g. `{29: PIT, 30: STAY, 31: PIT}` — old answers 29, new answers 31).
- `test_the_decision_lap_does_not_move_when_only_the_window_widens` (:193): the docstring
  claims "Under the old helper the answer was 27, 25 and 22 for these three windows — the
  left edge each time." **Executed under M1: the old helper returns 30, 30, 30** — the
  fixture's laps 20-29 are explicit STAY_OUT, and the old helper returned the first pit
  ACTION in the window, not the first window lap; it pinned to the edge only when the stack
  pitted on every lap. The test is green under the very code it documents itself as
  distinguishing. A docstring naming the wrong mechanism is this repo's catalogued defect
  class, here inside the test advertised as "the property that was missing, and the whole
  point of #752".

Consequence for #752's acceptance criterion 3 ("A test that fails if the metric's output
changes when only `DECISION_WINDOW_LAPS` changes"): no test in the file references
`DECISION_WINDOW_LAPS` at all (it is not in the import list), and the property test passes
under the old code — the criterion is unmet in letter. The transition semantics ARE pinned,
but by the three M1-killing tests, not by the two tests named for the property.

### F — the twin search: no second scorer in code; the stale report is the twin

Searched: `_PIT_ACTIONS` across all `*.py` (guard_rails.py owns it, no_llm.py:293 applies
it, decision_modes.py consumes it — nothing else); `first_pit|earliest.*pit|
pit.*transition|decision_lap|chosen_lap` across `src/` (decision_modes.py only); `PIT_NOW`
in the telemetry backend submodule (one comment at `strategy.py:1190`, one type-literal
tuple at `simulator.py:56` — neither scores anything); stop-lap loops in
`src/strategy/eval/` (projection.py:311 and stint_lengths.py:215 iterate REAL stop laps
from the parquet, not model actions). `_first_pit_lap` no longer exists anywhere. No
arcade/CLI/scripts code derives a "first pit decision" from a sequence of recommendations.

The surviving first-occurrence copy is not code: it is
`documents/eval_reports/decision_modes.{md,json}` (finding F1) — numbers produced by the
retired scorer, still committed, never regenerated, in the artifact position a reader
trusts most.

## Out-of-scope observations (pre-existing, not introduced by a474ac8)

- LOW: `lap_inputs` (decision_modes.py:285-287) uses `or`-defaults for `gap_ahead_s`
  (`or 2.0`), `air_temp` (`or 25.0`), `track_temp` (`or 35.0`) — a legitimate 0.0 would
  become the default. The same function reads `tyre_life` the long way and its docstring
  explains why `or 10` would be wrong; the three floats did not get the same care.
  Reachability is low (a 0.0 gap/temperature reading is rare-to-impossible), hence LOW.
- Observed in every measurement run: `Tire tool output did not parse for C3/C4/C5
  (tyre_life=1..4) - using conservative defaults instead of a 0.0 cliff` fires repeatedly
  through the no-llm stack on fresh tyres. Outside this gate's scope, but the decision
  tier's inputs are partially defaulted on exactly the laps right after a stop.

## What I tried to break and could NOT

1. **Off-by-one at the window boundaries of `_pit_decision_lap`.** Probed transitions
   exactly at `window_low` (scored, predecessor = the widened lap), exactly at `window_high`
   (scored), one lap before `window_low` (correctly `no_boundary`), one lap after
   `window_high` (correctly not agreement). `range(low, high + 1)` is inclusive on both ends
   with no drift.
2. **The sentinel classes.** `actions.get(lap)` returns None; `None in _PIT_ACTIONS` is
   False (executed), and the predecessor check is an explicit `is not None`, so a missing
   lap can never read as either a pit action or a stay-out. No `pandas.Series.get` anywhere
   in the changed code; `_stop_context` uses positional indexing plus `_is_missing`.
3. **The lap-1 clamp.** All-PIT from lap 1 in window [1, 8] gives `no_boundary`, never a
   score (lap 0 cannot exist and the code does not invent it); `{1: STAY, 2: PIT}` scores 2.
   No unjudgeable-but-scored or judgeable-but-wrong path found.
4. **Oscillation.** PIT-STAY-PIT re-commits score the re-commit lap; two transitions in one
   window score the first; verified hermetically and on NOR's real oscillation (44/42/38 at
   w=3/5/10, each chosen lap verified against the recorded actions to be a genuine evaluated
   transition).
5. **The widened lap being requested but not evaluated.** Checked every (race, driver) pair
   at three widths on 2025 Monza: `low` is present in `actions` in all cases. The only holes
   are post-retirement (ALO) and the final lap for lapped cars (GAS/STR/OCO, lap 53) —
   trailing edges, unable to fake a transition.
6. **Edge-report resurrection through the scoring range.** The single `chosen == window_low`
   case in the whole sweep (HAD, w=5) survives the window edge moving to 22 at w=10 with the
   same chosen lap — a model property, not an edge artifact. No mechanical edge pin exists
   in any of the 21 scored verdicts across three widths.
7. **Coverage inflation through the new bucket.** Constructed all-no_boundary and mixed
   runs; `eligible` includes the bucket in every denominator, the render prints it, coverage
   reports `masked`. The old silent-inflation path is closed.
8. **The commit's "6 of 7" number.** Re-derived independently at both widths plus a third:
   exactly 7 stops scored at both w=5 and w=10, exactly 6 with identical chosen laps, and
   the two movers are exactly HAM 38 (revealed transition) and NOR 46 (real oscillation).
   The claimed numbers are right.
9. **Four of five mutants.** M1, M2, M4, M5 all go red on at least one test each; the
   transition semantics, the None-predecessor rule, the bucket split and the denominator are
   genuinely pinned.

## Restoration audit

Every mutant was reverted with `git checkout -- src/strategy/eval/decision_modes.py`; after
the final restore, `git diff src/strategy/eval/decision_modes.py` printed nothing,
`git status --short src/ tests/` showed only the pre-existing untracked `src/telemetry`
submodule entry, and the pristine hermetic suite re-ran green (`33 passed, 1 deselected`).

## Verdict

The metric replacement is sound where it was claimed to be sound: the transition detector
has no boundary defects, the denominator cannot be inflated, the edge report is demonstrably
gone on real data, and the commit's headline numbers re-derive exactly. What is still
broken: the committed report still publishes the retired number (F1, HIGH); the one
mechanism the fix depends on — the widened replay span — is protected by a comment and
nothing else (E1, HIGH, proven with a suite-green non-equivalent mutant); the new bucket's
documented meaning is false for 4/4 of its real occupants at Monza w=5 (A1b, MEDIUM); a rail
release can still put a constant lap where a decision should be (A3, MEDIUM, structural);
and the two tests named for the new property do not discriminate it, one with a false
docstring (E2, MEDIUM).

Severity counts: **2 HIGH (F1, E1) · 4 MEDIUM (A1+A1b counted once, A3, E2) · 2 LOW**
(exact% labeling, `or`-defaults).

## Suggested fixes, ordered by value over risk

1. Regenerate `documents/eval_reports/decision_modes.{md,json}` with the new metric and
   commit it (closes F1; the decline rate and coverage the issue wanted preserved are
   bucket-count facts and survive).
2. Add the M3-killing test: stub `_decisions_in_window`, assert the span starts one lap
   before the earliest window AND that a transition sitting exactly at `window_low` is
   scored, not `no_boundary` (closes E1).
3. Reword the `no_boundary` prose in `_render_table` and the module docstring to what the
   code actually guarantees: "no evaluated STAY_OUT to pit transition inside the window —
   in this sample because the stack was already asking at the window's first evaluated lap"
   — and drop "every lap offered" (closes A1b; optionally split a `withdrew_in_window`
   observation into the JSON verdicts for the HAM-shaped cases).
4. Fix the two E2 fixtures (make the first pit call precede its stay-out, and correct the
   false docstring numbers 27/25/22 to what the old helper actually returned), and add the
   acceptance-criterion test that sweeps `DECISION_WINDOW_LAPS` itself.
5. For A3, either record pre-rail MC actions alongside post-rail ones in
   `_decisions_in_window` and refuse to score a transition whose predecessor STAY_OUT was
   rail-forced, or document at `_pit_decision_lap` that a chosen lap equal to
   `_NO_PIT_BEFORE_LAP` may be a rail release (cheapest honest option).
