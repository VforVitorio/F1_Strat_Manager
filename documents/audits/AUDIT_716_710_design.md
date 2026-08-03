# Design diagnosis — #716 (pit bounds) + #710 (operating envelope)

Session of 2026-08-03. Written before any code change, appended as findings land.
Read-only until the direction is agreed.

## D1 — #716's acceptance criterion #1 encodes the test Víctor himself refuted, 42 seconds after the issue was filed

The issue body argues: no FIA minimum-stint article exists, therefore
`_MIN_STINT_LAPS` is an opinion, therefore
`[[feedback_rails_encode_rules_not_opinions]]` forbids it. Acceptance criterion
#1 follows from that: *"Every surviving rail cites the regulation article that
makes it a fact, or it stops being a rail."*

That is the wrong test for this rail, and the correction is already in the repo.

**Timeline, from timestamps rather than recollection:**

| When (UTC, 2026-07-29) | What |
|---|---|
| 09:10:38 | Issue #716 created, body carrying the provenance argument |
| 09:18:09 | Issue **title** updated to "**Recalibrate** the anti-hallucination pit bounds…" |
| 09:18:51 | `feedback_rails_encode_rules_not_opinions` gains its `⚠️ ESSENTIAL REFINEMENT` section |

The refinement says, verbatim: *"I got this wrong and Víctor corrected me: I used
this memory to argue that N28's minimum-stint bound 'encodes an opinion' and
should cite a regulation. **Wrong test.**"*

| | What it does | Needs a regulation? |
|---|---|---|
| **Prescriptive** (the rejected SC `PIT_NOW` rail, #464) | Makes the decision *for* the model | **YES** |
| **Proscriptive** (min stint, no-pit-before-5, no-pit-last-3) | Bounds the output so the model cannot emit nonsense | **NO** — anti-hallucination guard |

And the prescribed remedy, also verbatim: *"If it catches a meaningful share of
what professionals actually do … it is separating unusual from usual rather than
absurd from sane — so **move the threshold, do not delete the bound.**"*

So the title was updated to the corrected doctrine and **the body was not**. The
body is the stale half. Two shipped files already encode the corrected doctrine
and both point at #716 for the measurement:

- `src/strategy/inference/guard_rails.py:16-23` — *"A proscriptive bound on a
  generative model's output is legitimate with or without a regulation behind it;
  the test it must pass is CALIBRATION."*
- `tests/mc/test_guard_rails.py:8-12` — same statement, as the test module's
  reason for existing.

**Consequence for scope:** the deliverable is a recalibration plus a written
justification per bound, not a deletion plus a migration into the Monte Carlo
cost. Criterion #1 should be restated as *"every surviving bound states either
the article that makes it a fact or the measured calibration that makes it a
bound"*.

## D2 — the issue's suggested direction (move it into MC scoring) is a sprint, not a session, and AUDIT_A2 already measured why

`documents/audits/AUDIT_A2_min_stint_veto.md` F5: neither Monte Carlo scorer has
any tyre-life or compound input today. `_run_mc_simulation`
(`strategy_orchestrator.py`) dispatches to `simulate_lap_window` (legacy,
seconds-based) or `_run_projection_mc` (position projection, the path real races
take). Threading a stint-freshness cost means changing both, plus their tests,
plus every test pinning exact MC scores.

F4 adds that `src/strategy/eval/decision_modes.py` is a string-matching consumer
of the rail's exact `reason` text and its whole exclusion methodology assumes a
categorical veto. Removing the veto changes published, checked-in report numbers
and breaks 5+ named tests by contract, not by assertion value.

## D3 — two of AUDIT_A2's findings are now STALE; the constant propagates by itself

A2/F1 claimed three hand-typed copies of 8/12/15 (guard_rails, N28 prompt, N31
prompt) with no test binding them. Verified today: **both prompts now render the
constant through an f-string.**

- `src/agents/pit_strategy_agent.py:663-665` — `{_MIN_STINT_LAPS['SOFT']}` etc.
- `src/agents/strategy_orchestrator.py:1695` — `SOFT >= {_MIN_STINT_LAPS['SOFT']}`
- `tests/agents/test_prompt_constants_match_tables.py:176-183` — parses the
  rendered prompt and asserts it against `_MIN_STINT_LAPS`.

So a change to the constant reaches all three sites with no manual edit, and a
regression is caught by an existing test. The "twin that never got the fix" risk
A2 raised for this specific rail has been closed since the audit ran.

Also closed: the prompt/mirror SC divergence A2 and the issue title both name.
`apply_guard_rails` now takes `sc_active` (`af3a24a`, 2026-07-29) and
`tests/mc/test_guard_rails.py` pins which bounds it suspends and which it does
not.

## D4 — the calibration evidence that already exists

`documents/eval_reports/stint_lengths.md`, generated 2026-07-29 over 1785 real
green-flag stints across 71 races (2023-2025 raw laps):

| compound | n | threshold | shorter than threshold | min | p1 | p5 | p10 | p25 | median |
|---|---|---|---|---|---|---|---|---|---|
| SOFT | 341 | 8 | **15.5%** | 1.0 | 1.0 | 2.0 | 5.0 | 9.0 | 15.0 |
| MEDIUM | 896 | 12 | **17.0%** | 1.0 | 1.0 | 7.0 | 9.0 | 14.0 | 19.0 |
| HARD | 548 | 15 | **12.2%** | 1.0 | 1.9 | 8.0 | 13.0 | 18.0 | 24.0 |

A bound that must sit "where real strategy essentially never goes" currently
sits at the 12th-17th percentile. That is one real stop in six. The bound fails
its own calibration test on the repo's own measurement — which is the finding
that justifies moving it, independently of any argument about FIA articles.

## D5 — #710's pit half is already merged on `dev`; only pace remains

Commit `cfe3ae0` "refactor(agents): declare N15 tyre-life ceiling as an operating
envelope" is on `dev`. `pit_strategy_agent.py:133-136` declares
`_N15_TYRE_LIFE_ENVELOPE`, and `_tyre_life_in` (line 845) labels the call
**before** clipping, so the out-of-range moment is logged rather than silent.

The remaining target is `pace_agent.py`, which has no range check at all.

**One acceptance line in #710 needs restating before it is checked off:** *"No
hand-placed clip survives that the envelope now covers."* Taken literally that
contradicts `envelope.py`'s own LABELLING ONLY contract (*"Checking a feature
vector against an envelope must NEVER touch, clip, or refuse a prediction by
itself"*). The clip at `pit_strategy_agent.py:853` must survive: N15 trained on
clipped input, so removing the clip would feed it values it never saw. The line
means "no clip survives *undeclared*", and that is satisfied.

---

# Measurements taken this session

## M1 — all four minimum-stint bounds calibrated on one sample

`f1-eval stint-lengths`, 2023-2025 raw laps, 1900 real green-flag stops across 71
races. Criterion agreed with Víctor: **a bound may veto at most 5% of real stops**,
and each value is the largest integer that clears it.

| bound | was | vetoed | now | vetoes |
|---|---|---|---|---|
| `_MIN_STINT_LAPS["SOFT"]` | 8 | 15.5% (341) | **2** | 3.2% |
| `_MIN_STINT_LAPS["MEDIUM"]` | 12 | 17.0% (896) | **7** | 4.6% |
| `_MIN_STINT_LAPS["HARD"]` | 15 | 12.2% (548) | **8** | 4.7% |
| `_DEFAULT_MIN_STINT` (wet) | 10 | **20.0%** (110) | **6** | 4.5% |
| `_NO_PIT_BEFORE_LAP` | 5 | **2.21%** | 5 unchanged | 2.21% |
| `_NO_PIT_LAST_N_LAPS` | 3 | **1.37%** | 3 unchanged | 1.37% |

**Two corrections to the issue.** It reports the last two bounds as blocking "4 real
stops" each and puts them in scope for review. That count comes from the six-race
`decision-modes` subset (198 stops). Measured on the full sample they veto 2.21% and
1.37%, both already inside the ceiling, so neither needed changing and neither was
changed. The bound that did need changing worst was `_DEFAULT_MIN_STINT`, which the
issue does not mention at all.

## M2 — why nobody had measured the worst bound: a comment naming the wrong mechanism

`stint_lengths.py` carried, above `_WET_COMPOUNDS`:

> Wet compounds run no minimum-stint rule at all (`_MIN_STINT_LAPS.get(compound,
> _DEFAULT_MIN_STINT)` never fires the SOFT/MEDIUM/HARD boundaries for them)

The headline is false and the parenthetical is true, which is exactly how it
survived review. Wet compounds miss the three named entries and land on the
FALLBACK, which is a minimum-stint rule like any other. On that claim the report
counted 110 wet stops only to drop them, so the one bound this file existed to check
was the one bound it never checked, and it was the worst calibrated of the four.

The wet sample is now a fourth row of the same table, graded against the same bound
the rail resolves, via a `_bound_for()` helper that calls the rail's own lookup
rather than mirroring it.

## M3 — a docstring in `decision_modes.py` promised an invariant the code did not enforce

`guard_rail_block`'s docstring: *"the probe passes a life that satisfies every
minimum"*. The expression was `max(_MIN_STINT_LAPS.values())`, which ignores
`_DEFAULT_MIN_STINT` — and the fallback is not a spare branch here, it is the bound
the probe meets in exactly the case the probe exists for, since an unknown compound
reaches `apply_guard_rails` as `""` two lines down. True today at 8 vs 6 and true
before at 15 vs 10, false the first time anyone ordered them the other way. Now
`max(*_MIN_STINT_LAPS.values(), _DEFAULT_MIN_STINT)`.

## M4 — #710: the pace envelope, and two claims of mine the measurement refuted

Bounds measured by rebuilding 2023 + 2024 through `augment_featured_laps` plus the
two N06 feature steps `pace_holdout.py` already owns, then taking each column's
min/max over the resulting 42,957 rows. Twelve continuous features; identifiers and
flags excluded because a code has no range.

**Refuted claim 1 — the three hardcoded zeros are NOT an envelope finding.**
`run_from_state` pins `prev_deg_rate` / `prev_cum_deg` / `prev_deg_accel` at `0.0` on
every real call. That reads exactly like the N26 out-of-range defect, and I was ready
to declare bounds that would "catch" it. Measured, 0.0 sits mid-distribution for all
three: 42.6% / 41.9% / 46.7% of training rows fall below it. An envelope could never
fire on the pinned value, and declaring one would have shipped a check that looked
like coverage and was not. Feeding a constant where the model saw a distribution is a
real defect; it is a different defect and needs its own instrument.

**Refuted claim 2 — the range in the code comment is the wrong season.**
`pace_agent.py` records FuelEffect as `range 0..3.685 s`, "verified exactly against
laps_featured_2025". 2025 is the held-out TEST season. The TRAINING range is
`0.055..4.125`. An envelope sourced from the test season describes where the model is
asked to work, not where it was fitted.

Out-of-range rates on the 2025 holdout frame: FuelEffect 4.96%, mean_sector_speed
3.93%, TyreLife 2.87%, AirTemp 1.96%, TrackTemp 1.64%, LapNumber 1.17%, FuelLoad
1.17%, Humidity 0.18%, Prev_TyreLife 0.06%, Prev_SpeedST 0.01%, Prev_LapTime 0.00%,
laps_remaining 0.00%.

**CORRECTION, from the real run (M6): those rates understate it and I quoted them as
if they did not.** See below.

## M5 — a pre-existing gate that does not cover what it guards

`tests/agents/test_prompt_constants_match_tables.py` skips on
`HAS_TIRE_MODELS`, but importing the pit agent pulls the NLP stack, so on a checkout
with tire weights and no `data/models/nlp/bert_sentiment_v1/` the file ERRORS instead
of skipping. Verified identical on the unmodified baseline, so it is pre-existing and
out of scope here, but it means the prompt-versus-rail guard can fail for a reason
that has nothing to do with prompts. Recorded, not fixed.


## M6 — the real run, and a third claim of mine it refuted

`f1-sim Lusail NOR McLaren --no-llm`, real radio corpus (24 radios, 66 rcm), 57 laps,
completed clean. Run twice on the branch with identical results, so the comparison
below is a behaviour change and not run-to-run noise.

**#716, behaviour moved by exactly one call.**

| | STAY_OUT | PIT_NOW | UNDERCUT |
|---|---|---|---|
| baseline (`dev`) | 52 | 3 | 2 |
| branch | **51** | 3 | **3** |

One lap flips STAY_OUT to UNDERCUT: the loosened bound releasing a call it used to
veto, in the expected direction and at the expected scale. Final position unchanged
(P3 to P4), final stint unchanged (HARD/13), best lap identical to the millisecond.

**#710, and the claim I got wrong.** M4 above says the log is "quiet enough to be
worth reading" on the strength of holdout rates of roughly 1-5%. On the real run the
envelope warned on **14 of 57 laps, about 25%**. Violations by feature: mean_sector_speed
10, TyreLife 6, Prev_TyreLife 6, FuelEffect 3, LapNumber 2, FuelLoad 2.

The holdout understates it **because it is the wrong frame to have measured on**, and
the reason is structural rather than incidental: the holdout has N06's own `_DROPNA`
applied, which removes the first lap of every stint and the opening laps of every
race. Those are precisely the rows where inference goes out of range, because
`run_from_state` passes `tyre_life or 1` and `prev_tyre_life = max(0, tyre_life - 1)`
against declared bounds of TyreLife [3, 78] and Prev_TyreLife [2, 77]. I measured the
noise on a frame that had already deleted the noisy rows.

What the warnings say is true, and is the most useful thing the envelope surfaced:
**N06 is asked to predict on the opening laps of a race and on the first lap of every
stint, and it was never trained on either**, because N06 dropped exactly those rows.
That was completely silent before this change. `mean_sector_speed` is a second, separate
finding: Lusail is fast enough to exceed the trained maximum of 314.97 on ten laps.

Left as is rather than suppressed. Whether a warning on a quarter of laps is the right
volume is a fair challenge and is put to the correctness gate rather than settled here.

## M7 — #710's "one real run per agent" was not satisfied by the pit half, and now is

The acceptance line reads: *"A real `f1-sim` run per agent, not one run for both"*,
with *"one agent at a time with an `f1-sim` run in between"*.

`cfe3ae0`, the merged pit half, justifies itself with *"125 tests in tests/mc pass
including the strategy goldens, and 400 real rows from Lusail 2025 go through the
feature builder with no warning while a forced 88-lap stint announces itself"*. That
is good evidence and it is not the evidence this criterion asks for: a feature-builder
replay is not a run of the simulator. Inheriting that as satisfied would have ticked a
box nobody had earned.

The two runs this session close it, in the required order and separated by the pace
change rather than batched:

| run | tree | agent under test | result |
|---|---|---|---|
| `sim_before` | `dev` (pit envelope present, pace envelope absent) | **pit / N15** | 57 laps clean, zero envelope warnings |
| `sim_after` | branch (both) | **pace / N06** | 57 laps clean, 14 warning laps |

Zero warnings in the first run is the CORRECT result rather than a missing signal: no
stint at Lusail approaches N15's 50-lap ceiling, so the in-range path is what should
have been exercised, and `tests/agents/test_n15_envelope.py` covers the firing case
that this race cannot produce.

---

# The adversarial gates, and what they broke

Two gates ran read-only against the branch and wrote their own reports:
`GATE_716_calibration.md` and `GATE_710_pace_envelope.md`. What follows is what
survived verification, including where a gate was itself wrong.

## G1 — CONFIRMED, HIGH, and mine: the recalibration silently rewrote an unrelated rule

Both prompts rendered the FLOOR of the MEDIUM compound-suitability band from
`_MIN_STINT_LAPS['MEDIUM']`:

```
MEDIUM: suitable for {_MIN_STINT_LAPS['MEDIUM']}-{_STINT_CAPACITY_LAPS['MEDIUM']} remaining laps.
```

Two different rules sharing the number 12 by accident. One asks whether a set has run
long enough to be worth replacing; the other asks whether enough race is left for the
compound to make sense. Recalibrating the first to 7 turned the second into "MEDIUM:
suitable for **7**-30 remaining laps" on the DEFAULT LLM path, a rule #716 never
touched and nobody reviewed.

The existing prompt-versus-table test could not see it: it asserts the prompt agrees
with the constant, and a re-coupled prompt agrees with the WRONG constant. Fixed with
its own `_MEDIUM_SUITABILITY_FLOOR_LAPS = 12` and a test that asserts the rendered
floor against THAT constant, which fails on a re-coupling.

## G2 — CONFIRMED, HIGH, and the same defect class twice in one session

`mean_sector_speed` should never have been bounded. `_compute_derived` falls back to
`prev_speed_st` when no mean sector speed is supplied and `run_from_state` never
supplies one, so at inference the feature ALWAYS carries the speed trap: training
means 256.8 vs 303.0 km/h, different physical quantities. The bound fired on 83% of
laps at Monza while describing none of them.

This is the exact twin of the `Prev_Deg*` case in M4 — a feature whose inference value
is not the quantity the range describes. I reasoned it through carefully for one member
of the pair and did not look for the other. That is the repo's most reliable defect and
I committed it inside the change that documents it.

`FuelLoad` came out for a related reason: the artefact stores it rounded to four
decimals (max 0.9615) while inference computes it live and unrounded, so a 78-lap race
gives 0.96153... and the bound fires on a lap class the model trained on.

Ten bounds remain. The unbounded fifteen are now enumerated with a reason each, and a
test asserts the two sets partition N06's feature list exactly, because the first
version of that comment accounted for 22 of 25 and the three it skipped are where the
mistake lived.

## G3 — CONFIRMED: the noise claim, twice corrected

M4's holdout rates were the wrong frame (M6). The gate then measured 31% of laps
warning across five circuits, worse than my Lusail figure. After G2's two removals,
the real run gives **6 of 57 laps (10.5%)**: TyreLife 6, Prev_TyreLife 6, FuelEffect 3,
LapNumber 2, co-occurring on six distinct laps. Every one is true and structural: N06
never trained on a race start or a stint opener, because its own `_DROPNA` removes
them. Left at WARNING; the train/serve gap it exposes deserves its own issue rather
than a quieter log.

## G4 — CONFIRMED, smaller

- `guard_rails.py` attributed the six-race subset's four excluded stops to the
  early-race bound. Measured, that bound excludes NONE there and `decision_modes.md`
  carries no `opening_laps` row; the four are the closing bound's (Monaco VER, Lusail
  STR and HAD, Monza OCO). Corrected.
- "the ceiling every bound is set from and held to" overstated it. All four are HELD
  to it; only the two that failed were SET from it. The lap bounds are not maximal
  under the rule and were deliberately left alone.
- The provenance pointer for the two lap-based shares was false: `stint_lengths.md`
  regenerates the four minimum-stint shares, not those two. Now labelled a dated
  finding rather than a reproducible artefact.
- The ceiling test could hold vacuously if a bucket left the measurement. It now pins
  the bucket set before filtering, so WET cannot silently stop being graded.
- `_DEFAULT_MIN_STINT` has no prose copy in either prompt: the offline path enforces a
  wet bound the LLM path was never told about. Written down, not closed, because
  closing it means new prompt text on the default path.
- `data/mc_measured_v1.json` had lost its measured `undercut_band` in the worktree
  (regeneration on a checkout missing the source parquet). Never committed; restored.

## G5 — REFUTED, verified by me

- The #716 gate reported that the committed `stint_lengths.py` fails
  `ruff format --check` under CI's pin. It does not: `uvx ruff@0.15.22 format --check .`
  reports 140 files already formatted, re-run after the finding.
- The #716 gate's own verification of every number in `guard_rails.py` reproduced them
  independently from `data/raw` without importing repo eval code, and predicted
  `min_stint = 5` before the regeneration produced exactly 5.

## G6 — the suite failures are not this change

7 failed / 4 errors on the full run, all pre-existing in this environment and confirmed
by running the ambiguous ones on a `dev` worktree with the same data: `test_weather_restore`,
`test_ml_recompute_golden` and both `test_mc_measured_tables` failures reproduce on the
baseline. Root cause for six of them is a single file, `data/processed/undercut_labeled/
undercut_clean.parquet`, which **does not exist in the Hugging Face dataset at all**, so
no clean install can pass them; the NLP goldens report `pending` because the 15.9 GB NLP
weights are not downloaded here. The skip guards should require those artefacts instead
of failing, which is a separate issue.

---

# #797 and #798, and one claim of mine I refuted myself

## H1 — the fix: N06 was reading the speed trap where it was trained on a circuit mean

`mean_sector_speed` is a property of the CIRCUIT, one value per GP.
`PaceAgent._compute_derived` substituted `prev_speed_st` whenever none was supplied and
`run_from_state` never supplied one, so on the path every real race takes N06 received a
different physical quantity on every lap. Training means 256.8 against 303.0 km/h.

Not an extraction slip: `N25_pace_agent.ipynb` documents the substitution as a proxy, and
`pace_agent.py` already named the real source, calling it a fallback for when circuit
features are unavailable. The lookup it described was never wired. It is now, per GP,
with an unresolvable circuit yielding NaN and a warning rather than a substituted reading.

**Measured impact, and a lesson about probes.** A single hand-built row moved the
prediction by 0.002 s and I nearly reported the fix as cosmetic on that basis. Over 4000
real 2025 laps it moves the delta prediction by a mean of +0.069 s, a p95 absolute of
0.377 s, and more than 0.010 s on 38% of laps. The trees split on this feature only in
some regions, so one probe is not a distribution.

## H2 — REFUTED BY ME, after the commit message had already claimed it

The commit says the value served is the value fitted, "identical to 0.0". What that
number actually compared was `laps_featured_2023` against
`circuit_features_with_clusters_k4.parquet` -- two artefacts of the TRAINING seasons. The
code serves the **2025** map. Those are a different pair, and I checked the wrong one.

Measured properly, across the 23 GPs present in both:

| | |
|---|---|
| GPs matching exactly | **0 of 23** |
| mean absolute gap | 4.82 km/h |
| median | 2.91 km/h |
| largest | Silverstone, 18.35 km/h |

The FIX is still right, and serving 2025 is the correct half of the pair: the feature is
recomputed per season, so the quantity N04 would compute for a 2025 lap is the 2025
measurement, and serving the training seasons' value would feed a stale reading of a
circuit since resurfaced or re-regulated. What was wrong was the claim of exact parity,
which was true in isolation about a comparison nobody had asked for and false about the
one that matters. That is the same defect class this session has been removing all day,
committed in the sentence describing the removal.

The docstring now states the seasonal seam and why the envelope bound is deliberately the
2023-2024 range held against a 2025 value: the bound asks whether N06 was FITTED on inputs
like this one, so a 2025 circuit outside the fitted range is genuine extrapolation. Monza
2025 at 317.24 against a fitted maximum of 314.97 is the only such case.

## H3 — the pit agent cannot be constructed on a clean install (#798)

`PitStrategyAgent.__init__` reads `data/processed/undercut_labeled/undercut_clean.parquet`
unconditionally. The Hugging Face dataset publishes `overtake_labeled/` and `sc_labeled/`
and **not** `undercut_labeled/`, so on a checkout built from the published data the agent
raises FileNotFoundError at construction. It stays hidden because every developer machine
has run N16, and because `f1-sim --no-llm` at Lusail never triggers the pit agent.

Six tests failed rather than skipped for the same reason, and one of them,
`test_the_committed_tables_match_a_fresh_measurement`, regenerated
`data/mc_measured_v1.json` with the entire measured `undercut_band` dropped to
`available: false` and left the emptied file in the worktree. The guards now name the
holdout. Publishing the artefact is the real fix and needs the file, which only exists on
a machine that has run the notebook.

## H4 — the #797 gate died on a usage limit, and its on-disk report paid for itself

`GATE_797_circuit_speed.md` reached 8.6 KB before the agent was terminated mid-run. Because
it appended findings as it confirmed them rather than buffering a final report, five
findings survived, four of them real and two HIGH. This is the second time in two sessions
that incremental persistence recovered work a dead agent would otherwise have taken with it.

What it found in my own fix, all verified independently before acting:

- **F1, HIGH.** The resolver covered two of the project's FOUR keyspaces. `RaceReplayEngine`
  puts the metadata.json name into `session_meta`, which for one race is `'Miami Gardens'`
  with a SPACE, matching neither the parquet slug `'Miami'` nor the folder `'Miami_Gardens'`.
  Every lap of the 2025 Miami race was served NaN while its value sat in the map. Same for
  the 2023 Spanish GP (`'Spain'`). The #448/#450 dual-keyspace trap, third occurrence, and
  the third time in this session that I fixed one member of a pair and not the other.
- **F2, HIGH.** `laps_featured_2025.parquet` carries NaN on all 760 Las Vegas rows, so
  reading that file dropped a circuit N06 was FITTED on and whose value sits in three other
  artefacts. The docstring's "we do not know this circuit" was false for Las Vegas.
- **F3, MEDIUM.** The map was 2025-only while the replay engine can replay 2023 and 2024, so
  a 2023 Silverstone lap was served a measurement taken two years after it, 18.4 km/h away.
  `run()` receives `year` and the resolver ignored it.
- **F4, MEDIUM.** My "recomputed per season" claim was wrong: 2023 and 2024 are identical
  per GP to exactly 0.0. The value is recomputed per ARTEFACT BUILD, one build pooling both
  training seasons. The conclusion survived, the stated mechanism did not, and a comment
  naming the wrong mechanism is how the next fix goes wrong.
- **F5, verified clean.** NaN survives to the model, `_bootstrap_ci` multiplying NaN by
  Gaussian noise still returns finite p10/p90 through XGBoost's default split, an explicit
  value still wins, and no production caller passes one.

**And one the gate did not reach, found while fixing F1:** the combined artefact does not
agree with itself. `laps_featured.parquet` calls the same race `'Miami'` in 2023-2024 and
`'Miami Gardens'` in 2025, so even a correctly spelled query misses on one season. The fix
is not a longer candidate list at the query end, which fails the moment the STORED spelling
is the odd one: `_normalise_gp_key` now normalises both the map keys at load time and the
query, which is what `gp_slugs`'s own docstring prescribes.

The lookup is now keyed by `(Year, GP)` from the combined parquet, which has 71 pairs and
zero missing values. All 71 races on disk resolve, asserted by walking `data/raw/` rather
than by fixing the two names an audit happened to mention. A real `f1-sim Miami_Gardens`
run completes with zero unresolved-circuit warnings.
