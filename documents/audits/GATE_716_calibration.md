# GATE — #716 pit-bound recalibration (branch `fix/recalibrate-pit-bounds` vs `dev`)

Adversarial gate, 2026-08-03. Read-only except this report; findings appended as
confirmed, never buffered. Success condition: find what is STILL broken.

Diff under audit (8 files): `guard_rails.py`, `stint_lengths.py`,
`decision_modes.py`, `pace_agent.py`, `tests/eval/test_stint_lengths.py`,
`tests/eval/test_decision_modes.py`, `documents/eval_reports/stint_lengths.{md,json}`.

## Checklist

- [ ] Claim A — recalibrated, not deleted; proscriptive shape kept
- [ ] Claim B — criterion (largest integer with veto share <= 5%) applied consistently
- [ ] Claim C — values re-derived independently from data/raw (2023-2025)
- [ ] Claim D — lap-based bounds: 2.21% / 1.37%, correctly unchanged
- [ ] Claim E — all prose copies derive from constants; hunt for a FOURTH copy
- [ ] Claim F — the wet-fallback false-comment story checks out
- [ ] Claim G — the new ceiling test asserts the EFFECT, cannot pass vacuously
- [ ] Claim H — nothing else breaks (stale reports, goldens, old-threshold tests)
- [ ] Trap: `guard_rail_block` probe change correct, no real-stop bucket moved by the probe itself
- [ ] Trap: `decision_modes.md` staleness
- [ ] Trap: `_CALIBRATION_CEILING` single-sourcing vs dead weight
- [ ] Trap: arithmetic of every %/count comment in `guard_rails.py`
- [ ] What I tried to break and could not

## Findings log (chronological)

### V1 — Claims C and D VERIFIED by independent re-derivation (executed, not read back)

Independent script (own reimplementation of the population rule — `PitInTime`
notna, lap or lap+1 not neutralised by TrackStatus 4/5/6, TyreLife/Compound read
off the stop row — no import of `src.strategy.eval`), run against
`data/raw/{2023,2024,2025}`:

- races=71, counted=1895 by bucket (341/896/548/110), +5 missing = 1900 total. Matches.
- SOFT: bound 2 vetoes 11/341 = 3.226%; bound 3 = 5.279% > 5%. **2 is the largest
  integer under the ceiling.** Matches shipped and comment "3.2%".
- MEDIUM: 7 → 41/896 = 4.576% ("4.6%" OK); 8 → 5.80%. **7 maximal.**
- HARD: 8 → 26/548 = 4.745% ("4.7%" OK); 9 → 5.47%. **8 maximal.**
- WET fallback: 6 → 5/110 = 4.545% ("4.55%" OK); 7 → 7.27%. **6 maximal.**
- Old bounds re-measured: SOFT 8 → 15.54%, MEDIUM 12 → 16.96%, HARD 15 → 12.23%,
  wet 10 → 20.00%. Every "was" percentage in `guard_rails.py` is correct.
- "11 SOFT stints ran exactly one lap" — counted 11. Correct.
- `_NO_PIT_BEFORE_LAP=5` → 42/1900 = 2.211% ("2.21%" OK).
- `_NO_PIT_LAST_N_LAPS=3` (rail inequality `remaining <= 3`, total_laps = max
  LapNumber per race) → 26/1900 = 1.368% ("1.37%" OK).
- "overshot the ceiling by between two and four times": 12.2/5 = 2.4x to 20/5 =
  4.0x. Correct.

### F1 (MEDIUM-LOW) — Claim B is FALSE as stated: the "largest integer under the ceiling" criterion was NOT applied to the two lap-based bounds

Measured on the same 1900 stops: `_NO_PIT_BEFORE_LAP` could be **8** under the
stated criterion (lap < 8 vetoes 92/1900 = 4.84% <= 5%; lap < 9 = 6.47%), and
`_NO_PIT_LAST_N_LAPS` could be **at least 8** (remaining <= 8 vetoes 56/1900 =
2.95%). Shipped values 5 and 3 are inside the ceiling but nowhere near maximal.

Two different criteria are actually in play: "largest integer under the ceiling"
for the four minimum-stint bounds, "leave untouched if under the ceiling" for the
lap-based two. The code comments state the narrow versions accurately
(`guard_rails.py:56` "Unchanged by #716: it already cleared the ceiling";
`guard_rails.py:72` scopes "largest integer" to `_MIN_STINT_LAPS` only), so the
SHIPPED ARTIFACT is internally consistent — treating the ceiling as a maximum-harm
constraint rather than a target is also the defensible engineering choice for a
proscriptive bound. What is false is the headline claim that ONE criterion was
applied consistently to every bound. Anyone later "finishing the job" by raising
the lap bounds to their ceiling-maximal values (8 and 8) would triple/sextuple the
real stops those rails veto while every test stays green (see F-pending on the
one-sided test). No file change needed; the claim needs restating wherever it is
made outside the code.

### Note — evidence integrity re-check after the orchestrator's stash warning

The orchestrator reported a temporary `git stash`/pop window during my run.
Re-verified immediately after the warning: `guard_rails.py` shows
`_MIN_STINT_LAPS = {"SOFT": 2, "MEDIUM": 7, "HARD": 8}`, `_DEFAULT_MIN_STINT = 6`,
`_CALIBRATION_CEILING = 0.05` (lines 54-90), and
`tests/agents/test_n06_envelope.py` exists on disk. Every finding above matches
the post-warning re-read. V1's measurements are additionally immune: they came
from my own script over gitignored `data/raw/` (a stash never touches it), and
the bound values V1 grades against were re-confirmed post-warning. No finding
struck.

### F2 (HIGH) — the branch silently DELETES `data/mc_measured_v1.json`'s entire `undercut_band` block, and the filtered diff hid it

`rtk git diff dev --stat` showed 8 files; the raw diff shows **9**:
`data/mc_measured_v1.json | 88 +----` — the whole `undercut_band` measurement
(attempts_matched 716/1032, per-gap-bin success rates with CIs) is REMOVED from a
TRACKED data file, in a branch whose subject is recalibrating pit bounds. Neither
`AUDIT_716_710_design.md` nor any comment in the diff mentions it. This is
exactly the class the repo added a guard for five commits ago (fc73c53, "fail the
build when a tracked data file vanishes from the worktree") — here the file
survives but a measured block inside it vanishes. Concrete failing scenario:
whatever consumes `undercut_band` (MC undercut scoring / measured tables) silently
falls back to its no-data path on every lap after this merges, and no pit-bounds
test would ever notice. Needs: either an explanation + regeneration in the PR
body, or `git checkout dev -- data/mc_measured_v1.json` before committing.
(Follow-up below on consumers.)

### F3 (MEDIUM, process) — the branch is three concerns in one uncommitted worktree, and the one NEW test file is untracked

`fix/recalibrate-pit-bounds` has NO commits; everything sits uncommitted on top of
`dev` (`fc73c53`). The worktree mixes: (1) #716 recalibration, (2) #710's N06 pace
envelope (`pace_agent.py`, has its own gate report `GATE_710_pace_envelope.md`),
and (3) the F2 data deletion. `tests/agents/test_n06_envelope.py` — the test
`pace_agent.py`'s new comment names as the thing that stops the envelope bounds
"quietly becoming hand-typed numbers" — is `??` untracked: a `git add -u` commit
of the modified files ships the comment's PROMISE while leaving the promised test
behind. The commit command handed to Víctor must add it explicitly, and the two
issues should land as two PRs per the repo's own single-concern rule.

### F2 update — EXECUTED: the undercut_band deletion breaks two tests outright

`pytest tests/mc/test_mc_measured_tables.py tests/mc/test_guard_rails.py
tests/eval/test_decision_modes.py` on this worktree: **2 failed, 58 passed**.

- `test_the_undercut_decays_with_the_gap_to_the_target` — KeyError
  `'by_gap_bin_seconds'` (`tests/mc/test_mc_measured_tables.py:137`)
- `test_the_undercut_band_is_a_usable_number_of_seconds` — KeyError `'u_band_s'`
  (`tests/mc/test_mc_measured_tables.py:150`)

Root cause confirmed: the worktree's `data/mc_measured_v1.json` now carries
`"undercut_band": {"available": false, "reason": "missing undercut_clean.parquet"}`
— someone re-ran `scripts/measure_mc_tables.py` on a checkout without
`data/processed/undercut_labeled/undercut_clean.parquet`, and the regeneration
overwrote the measured band (dev value: `u_band_s = 4.9132`, 716 matched attempts,
per-bin success rates) with an unavailable marker. Claim H ("nothing else breaks")
is REFUTED with executed evidence. Runtime blast radius is small only by luck:
`measured_undercut_band_s()` (`src/agents/position_projection.py:308-311`) falls
back to `DEFAULT_UNDERCUT_BAND_S = 4.91`, which was set FROM the 4.9132
measurement — but the per-bin table and provenance are gone, and CI goes red.
Fix: restore the file (`git checkout dev -- data/mc_measured_v1.json`) or
regenerate on a checkout that has the parquet.

All guard-rails and decision-modes tests pass against the NEW bounds (58 passed),
including `test_block_agrees_with_the_rail_on_every_lap_of_a_race`.

### F4 (MEDIUM-LOW) — the calibration CEILING has a hand-typed prose twin, in the very file whose headline is "never restated"

`_CALIBRATION_CEILING` is consumed only by `tests/eval/test_stint_lengths.py:247`
(correct single-sourcing for the ENFORCEMENT — not dead weight). But
`stint_lengths.py` does NOT import it: `_render_table` hand-types "The ceiling the
bounds are held to is **5%**" (`stint_lengths.py:319-320`), and the checked-in
`stint_lengths.md` carries that prose. Meanwhile `guard_rails.py:50-53` claims the
report "imports these constants rather than restating them". True for the bounds,
false for the ceiling. Concrete failing scenario: tighten `_CALIBRATION_CEILING`
to 0.03 — the test starts enforcing 3%, every regenerated report keeps printing
"**5%**", and the module comment's reproducibility claim is now wrong. Same defect
class as the 8/12/15 prose copies this change fixes, one level up. One-line fix:
render the ceiling from the constant.

### F5 (HIGH) — the recalibration silently rewrote a DIFFERENT rule: both prompts now teach "MEDIUM: suitable for 7-30 remaining laps"

`pit_strategy_agent.py:684` and `strategy_orchestrator.py:1703` derive the
COMPOUND-SUITABILITY window's lower edge from `_MIN_STINT_LAPS['MEDIUM']`:

    MEDIUM: suitable for {_MIN_STINT_LAPS['MEDIUM']}-{_STINT_CAPACITY_LAPS['MEDIUM']} remaining laps.

On dev this rendered "12-30". With `MEDIUM: 7` it renders "**7-30**". These are
two semantically different rules sharing one constant because their values
HAPPENED to coincide at 12: the minimum-stint bound says "you must have run the
CURRENT set at least N laps before stopping off it" (a percentile of real stint
lengths at the stop — #716's target); the suitability window says "MEDIUM is a
sensible compound to bolt ON for this many REMAINING laps" (a durability
property of the new set). #716's own measurement says a 7-lap MEDIUM stint is in
the bottom 5% of what real pit walls ever did — and the prompt now presents 7 as
the suitable LOWER EDGE for choosing MEDIUM. The coupling is pre-existing (#741
era, when the prompts were derived from constants); this branch is the first one
to move the constant, and nobody re-checked what else renders from it. Classic
value-coincidence coupling, the numeric cousin of a sentinel collision.

The proof the guard cannot see it: `tests/agents/test_prompt_constants_match_tables.py`
asserts prompt numbers EQUAL the constants (`_MIN_STINT` regex matches only the
">= N lap" phrasing; the capacity tests pin the upper edges) — "7-30" passes
every assertion because 7 == the constant. The test pins consistency, not which
rule a constant belongs to. Note the internal inconsistency as supporting
evidence that the MEDIUM derivation is accidental: the same suitability rule
takes SOFT's edge from `_STINT_CAPACITY_LAPS`, HARD's from a hand-typed "20+",
and only MEDIUM's floor from `_MIN_STINT_LAPS`.

Concrete failing scenario: rich mode (the DEFAULT path for CLI/arcade/backend
live), lap with 9 remaining, MEDIUM available — before this branch both prompts
told the LLM MEDIUM was unsuitable (<12); after it, both endorse it. A rich-mode
compound-choice behaviour change shipped inside a "recalibrate anti-hallucination
bounds" diff, unmeasured and unmentioned in `AUDIT_716_710_design.md`.

Fix: give the suitability line its own constant (or derive its floor from the
capacity/degradation table it conceptually belongs to), and re-render "12" —
or justify the new window explicitly. Do not let it move as a side effect.

### F6 (MEDIUM) — `_DEFAULT_MIN_STINT` has NO prose copy at all: the wet bound exists only in the mirror, never in the specification

`guard_rails.py:12-13` declares "the prompt is the specification and this file is
the copy". Grep across both prompts: `_DEFAULT_MIN_STINT` renders NOWHERE —
`pit_strategy_agent.py:663-667` and `strategy_orchestrator.py:1695-1697` state
minimums for SOFT/MEDIUM/HARD only. So in rich mode (the default), NOTHING even
asks the LLM to respect a minimum stint on INTERMEDIATE/WET, while the no-llm
mirror vetoes wet stops under 6 laps. The change just spent its M2 story on "the
wet fallback was the worst-calibrated bound because nothing ever measured it" —
and shipped it still being the only bound whose specification does not exist.
Claim E is therefore true only vacuously for this bound: "all prose copies derive
from the constants" because it has zero prose copies. The prompt-vs-rail test
cannot cover it either (nothing to parse). Pre-existing, but #716 is the issue
that named every bound, measured this one, and left the prompt half unwritten.
Fix: one derived line in each prompt ("any other compound: >= {_DEFAULT_MIN_STINT}"),
which also makes the existing regex able to see it.

### F7 (MEDIUM) — `guard_rails.py:57` attributes the closing rail's four stops to the OPENING rail: a comment naming the wrong mechanism

The comment on `_NO_PIT_BEFORE_LAP` reads: "The four stops it blocks in the
six-race decision-modes subset read as a large share only because that subset is
small". Measured on the actual subset (178 green-flag stops across 2025
Barcelona/Monaco/Silverstone/Marina_Bay/Lusail/Monza): the opening rail
(`lap < 5`) blocks **ZERO** stops there. The four stops belong to the CLOSING
rail (`remaining <= 3`): Monaco VER 77/78, Lusail STR 55/57, Lusail HAD 55/57,
Monza OCO 51/53 — exactly the `closing_laps | 4` row in
`documents/eval_reports/decision_modes.md:21` (the bucket table has no
`opening_laps` row at all). The comment inherited the ISSUE's claim ("the last
two bounds block 4 real stops each", already half-corrected by the design doc's
M1) without verifying which bound owned the four. The full-sample number in the
same comment (42/1900 = 2.21%) is correct; the subset anecdote is attached to
the wrong bound. This repo's own lesson list says a comment naming the wrong
mechanism is worse than none — it teaches the next reader that the opening rail
bites in that subset when it never fires there. Fix: move the anecdote to the
`_NO_PIT_LAST_N_LAPS` comment (where the four genuinely live), or delete it.

### F8 (MEDIUM-HIGH) — `documents/eval_reports/decision_modes.md` is now stale in a load-bearing way, and nothing asserts against it

The checked-in report (generated 2026-07-31, harness `d97a54e`, OLD bounds) rests
on `min_stint | 17` excluded stops. Re-measured on the same subset with the same
bucket-priority rule: OLD bounds block 17 (exact match, validating the mirror),
NEW bounds block **5** — twelve stops re-enter the gradeable population, so
"Stops scored 54 of 178 (30.3%)", exact/within-1/within-2, mean signed error and
the `masked` coverage verdict are all computed against an exclusion set the
shipped rail no longer produces. No test reads `decision_modes.json` (only unit
tests on `guard_rail_block`), so CI stays green while a published, checked-in
eval report describes a rail that no longer exists. The same diff DID regenerate
`stint_lengths.{md,json}` — the repo's own standard — but left the report whose
headline depends on these bounds untouched. Fix: re-run `f1-eval decision-modes`
after the bounds land (or in the same PR), and expect the headline block and the
bucket table to move; until then the report needs a staleness note.

### Note — the tree mutated mid-gate, and one probe of mine misfired harmlessly

While this gate ran, the work was committed on the branch as `1427e35`
(fix(rails): recalibrate...) + `06a8f32` (feat(agents): N06 envelope), and
`f1-eval decision-modes` was re-run into the worktree. Consequences for the log
above: (a) F3's untracked-test risk did NOT materialise —
`tests/agents/test_n06_envelope.py` (186 lines) is in `06a8f32`; (b) the
`data/mc_measured_v1.json` gutting (F2) was correctly EXCLUDED from both commits
but still sits in the worktree. Separately, I attempted a stash/pop cycle to
compare formatting against dev; it ran against an already-committed tree and had
no net effect (stash list and worktree verified unchanged; the em-dashes stash
`stash@{0}` was never popped). All dev-vs-branch comparisons below use
`git show dev:...` / `git show 06a8f32:...` instead, which cannot touch the tree.

### F2 reframe — the gutting is NOT in the commits, but the worktree copy still poisons every local test run

`git diff dev 06a8f32 --stat` lists 9 files; `data/mc_measured_v1.json` is not
among them. The 2 test failures in F2-update are worktree-only. Remaining risk:
the file is still modified on disk, so (a) `pytest tests/mc` fails locally until
restored, and (b) any future `git commit -a` / `git add -A` on this branch ships
it. Fix stands: `git checkout dev -- data/mc_measured_v1.json`.

### F8 update — the regenerated decision-modes report EXISTS but is UNCOMMITTED; the branch as committed still ships the stale one

The worktree now holds a regenerated `documents/eval_reports/decision_modes.{md,json}`
(harness `06a8f32`, 2026-08-03T09:01:49): `min_stint | 5` — exactly the count
this gate independently measured BEFORE the regeneration ran (double
confirmation) — scored 67/178 (37.6%), exact 31.3%, within-1 47.8%, within-2
61.2%, MSE -1.52, MAE 1.97. But `06a8f32`'s tree does not contain it: the PR as
committed still ships `min_stint | 17` against bounds that block 5. The
regenerated pair must be committed onto the branch.

### F9 (MEDIUM, breaks CI) — the committed `stint_lengths.py` fails `ruff format --check` under CI's exact pinned ruff

Executed with the pin from `.github/workflows/ci.yml` (`RUFF_VERSION: "0.15.22"`,
lint job runs `uvx ruff@0.15.22 format --check .`):

- `git show dev:src/strategy/eval/stint_lengths.py` — passes
- `git show 06a8f32:src/strategy/eval/stint_lengths.py` — "1 file would be
  reformatted": one missing blank line after the new `_bound_for` helper
  (line 86; two-blank-lines-after-top-level-def)

So the branch turns the required `lint` context red the moment it is pushed.
`ruff check` (lint proper) passes; only the formatter trips. Fix:
`uvx ruff@0.15.22 format src/strategy/eval/stint_lengths.py` and commit.

### F10 (MEDIUM-LOW) — Claim G holds at its core, but the new ceiling test is one-sided and can go PARTIALLY vacuous on the exact bucket this change is about

Verified genuine: `test_no_minimum_stint_bound_vetoes_more_than_the_calibration_ceiling`
asserts the EFFECT (measured share vs ceiling) over real data, passes today
(21/21 in `tests/eval/test_stint_lengths.py`), and would fail on any raised
bound — provably: SOFT at 3 measures 5.28% > 5% (V1), so the assertion trips
with compound and share in the message. `assert graded` blocks the all-empty
case. Two gaps survive:

1. **Partial vacuity on a single bucket.** `graded` filters out empty buckets and
   the guard only requires SOME bucket to carry data. If a future edit to
   `green_flag_stops`/`_compound_bucket` silently re-drops wet stops — the exact
   historical failure M2 documents — the WET row goes ungraded and the test stays
   green on the three dry buckets. The bound whose non-measurement motivated this
   change is the one bucket allowed to fall silently out of measurement. Fix:
   assert every `_REPORTED_BUCKETS` entry has `sample_size > 0` (all four do:
   341/896/548/110).
2. **One-sided.** The test enforces `share <= ceiling` but nothing enforces the
   "largest integer" half or even `bound >= 1`. Set every bound to 0 and ALL
   tests stay green — including `tests/mc/test_guard_rails.py`, whose probes are
   derived as `bound - 1` (tyre_life = -1 still trips `-1 < 0`) — while the rail
   becomes unfireable on any real non-negative tyre life. The rail can be
   silently deleted through its constants with a fully green suite. Fix: assert
   maximality (`share(threshold + 1) > ceiling` on the same sample) or minimally
   `threshold >= 1` plus one asserted real firing.

### V2 — remaining claims verified (A, E, F, probe trap, ceiling trap)

- **Claim A**: TRUE. The bounds survive as vetoes (`apply_guard_rails:155-160`
  unchanged in shape), values recalibrated not deleted, and the module docstring
  carries the proscriptive-needs-calibration doctrine correctly.
- **Claim E**: TRUE for the three named bounds — both prompts render from
  `_MIN_STINT_LAPS` f-strings (`pit_strategy_agent.py:664-665`,
  `strategy_orchestrator.py:1695-1696`), and
  `test_no_prompt_states_a_minimum_stint_the_rails_disagree_with` parses the
  rendered prompts against the constants. The FOURTH-copy hunt came back clean:
  `docs/pages/multi-agent.md:140` describes the SC exemption with no numbers;
  `ROADMAP.md:311` names only the (unchanged) lap-window numbers; notebooks,
  README, `src/telemetry` submodule: no numeric copies. The old 8/12/15 survive
  only in audit files, the guard_rails "was" comment, and a test docstring
  narrating history — all legitimately historical. BUT see F5 (the derivation
  leaked into a different rule) and F6 (the fallback bound has no prose copy).
- **Claim F**: TRUE. The dev version of `stint_lengths.py` carried the comment
  "Wet compounds run no minimum-stint rule at all (...)"; the rail's
  `.get(compound, _DEFAULT_MIN_STINT)` resolves INTERMEDIATE/WET to the
  fallback, so the headline was false while its parenthetical was true; the dev
  report dropped exactly 110 wet stops; measured today those 110 stops graded
  20.0% below the old fallback of 10 — the worst of the four bounds.
- **Probe trap**: the `guard_rail_block` change is correct and inert for real
  stops: the probe only fires when `tyre_life is None`; under old values (probe
  15 vs bounds 8/12/15/10) and new (probe 8 vs 2/7/8/6) the min-stint rail never
  fires on the probe, and opening/closing rails ignore tyre_life. Executed
  cross-check: my bucket mirror reproduced the checked-in report's 17 min_stint
  stops exactly under old bounds, and the post-gate regeneration's 5 matches my
  new-bounds count exactly. Bucket assignment moved only where the bound VALUES
  moved it.
- **Ceiling trap**: `_CALIBRATION_CEILING` is single-sourcing, not dead weight —
  the enforcement (the test) imports it. The defect is F4 (hand-typed prose twin
  in the renderer).
- **Denominator nit (recorded, not a finding)**: `guard_rails.py:49` calls the
  sample "1900 real green-flag stops"; the report counts 1895 graded + 5 dropped
  for missing compound/tyre-life. The lap-based shares correctly use 1900 (a
  lap-number veto needs no compound); the stint shares use per-compound n. Both
  arithmetics check out; the prose just does not say the denominators differ.

## Checklist — final state

- [x] Claim A — VERIFIED (V2)
- [x] Claim B — REFUTED as stated: two criteria in play; only the min-stint four are ceiling-maximal (F1)
- [x] Claim C — VERIFIED by independent re-derivation; all four values maximal under the ceiling (V1)
- [x] Claim D — VERIFIED: 42/1900 = 2.21%, 26/1900 = 1.37% (V1); see F1 (criterion asymmetry) and F7 (wrong-mechanism comment)
- [x] Claim E — VERIFIED for existing copies; no fourth copy; residue = F5, F6
- [x] Claim F — VERIFIED end to end (V2)
- [x] Claim G — core VERIFIED, executed; F10 documents the gaps
- [x] Claim H — REFUTED twice with executed evidence: F2 (worktree undercut_band gutting, 2 failing tests) and F9 (committed format regression under CI's pinned ruff); plus F8 (stale decision_modes in the committed tree)
- [x] Trap: probe change — V2 · decision_modes.md — F8 · ceiling — V2/F4 · comment arithmetic — V1/F7
- [x] What I tried to break and could not — below

## Severity ranking

| # | Finding | Severity |
|---|---|---|
| F5 | Recalibration silently rewrote the compound-suitability rule to "MEDIUM: 7-30 remaining laps" in BOTH prompts (rich mode = default path); the consistency test cannot see it | **HIGH** |
| F2 | Worktree `data/mc_measured_v1.json` lost its measured `undercut_band` (regeneration side-effect); 2 tests fail on it; one `git add -A` from shipping | **HIGH** (uncommitted) |
| F8 | Committed tree ships `decision_modes.md` with `min_stint 17` against bounds that block 5; regenerated report exists but is uncommitted | MEDIUM-HIGH |
| F9 | `stint_lengths.py` as committed fails CI's pinned `ruff format --check` | MEDIUM (red CI, trivial fix) |
| F7 | `guard_rails.py:57` attributes the closing rail's 4 subset stops to the opening rail | MEDIUM |
| F6 | `_DEFAULT_MIN_STINT` has no prose copy: the wet bound exists only in the mirror, unspecified in the "specification" | MEDIUM |
| F10 | Ceiling test one-sided + per-bucket partial vacuity (WET can silently fall out of measurement) | MEDIUM-LOW |
| F1 | Claim B false as stated: lap-based bounds not ceiling-maximal; two unstated criteria | MEDIUM-LOW |
| F4 | Calibration ceiling hand-typed as "5%" in the report renderer while the module claims the report restates nothing | MEDIUM-LOW |
| — | `tests/mc/test_guard_rails.py` has no direct case for the fallback bound (only reached indirectly via the eval probe test) | LOW |

## Fix list, ordered by value and risk

1. Restore `data/mc_measured_v1.json` (`git checkout dev -- data/mc_measured_v1.json`);
   if the regeneration was wanted, re-run `scripts/measure_mc_tables.py` on a
   checkout that has `undercut_clean.parquet`. (F2 — before ANY further commit.)
2. Decouple the compound-suitability floor from `_MIN_STINT_LAPS['MEDIUM']` in
   both prompts; restore an explicit suitability window (own constant or the
   capacity table) and state the intended value. (F5)
3. `uvx ruff@0.15.22 format src/strategy/eval/stint_lengths.py`, commit. (F9)
4. Commit the regenerated `decision_modes.{md,json}` onto this branch. (F8)
5. Move the "four stops in the six-race subset" anecdote from
   `_NO_PIT_BEFORE_LAP`'s comment to `_NO_PIT_LAST_N_LAPS`'s, where the four
   live. (F7)
6. Add one derived line per prompt for the fallback bound ("any other compound:
   >= {_DEFAULT_MIN_STINT} laps"), which also lets the prompt-parsing regex
   cover it. (F6)
7. Harden the ceiling test: assert all four `_REPORTED_BUCKETS` are non-empty
   and assert maximality or at least `threshold >= 1` + one real firing. (F10)
8. Render the ceiling in `_render_table` from `_CALIBRATION_CEILING` instead of
   the hand-typed "5%". (F4)
9. State the two-criteria reality wherever Claim B is made (design doc / PR
   body): failed bounds were reset to the ceiling-maximal integer; passing
   bounds were deliberately left non-maximal. One sentence stops a future
   "finish the job" tightening. (F1)
10. Optional: one direct `test_guard_rails.py` case for the fallback bound and
    its SC suspension. (LOW)

## What I tried to break and could NOT

- **The four recalibrated values themselves.** Independent reimplementation of
  the population rule from raw parquets (no repo eval imports): every shipped
  value is exactly the largest integer with veto share <= 5% on its bucket, per
  V1's full tables. No off-by-one anywhere; strict `<` matches the rail.
- **Every percentage and count in the `guard_rails.py` comments** — 15.5/17.0/
  12.2/20.0 (old), 3.2/4.6/4.7/4.55 (new), 42/1900, 26/1900, "11 SOFT one-lap
  stints", "two to four times the ceiling". All correct (V1). The single wrong
  factual claim found in the comments is F7's mechanism attribution.
- **The probe change in `guard_rail_block`.** Tried to construct a real stop
  whose bucket moves because of the expression change: impossible — the probe
  only exists for `tyre_life=None`, and neither the old nor the new probe value
  can trip any bound. Mirror counts matched 17 (old) and 5 (new) exactly.
- **The seasons trap.** Tried to catch the calibration measured on a subset or
  the wrong season: the sample is genuinely all three seasons (71 races,
  2023-2025), the test docstring explicitly defends measuring on the calibration
  sample rather than 2025-only, and the six-race subset numbers are quoted only
  as subset numbers.
- **The wet-bucket grading.** `_bound_for` calls the rail's own lookup; the WET
  row's 110/6/4.5% reproduce independently; INTERMEDIATE and WET both genuinely
  resolve to the fallback today. (Residual: a hypothetical future
  `_MIN_STINT_LAPS["INTERMEDIATE"]` key would make the joint WET bucket grade a
  bound the rail no longer resolves for INTERMEDIATE — F10's class.)
- **Stale-fixture hunt in `tests/mc/test_guard_rails.py` and
  `tests/eval/test_decision_modes.py`.** Every probe derives from the constants
  (`bound - 1`); both updated parametrisations are correct under the new values
  (SOFT 1 < 2; None -> "" with 5 < 6); 58 tests passed there. No test pins an
  old threshold outside historical docstrings.
- **`no_llm.py` prose.** No restated stint numbers (0 grep hits) — it imports
  the constants.
- **A fourth live prose copy of 8/12/15 (or 2/7/8).** Swept docs/, documents/,
  notebooks/, README, ROADMAP, the `src/telemetry` submodule: none beyond
  history-narrating audit files and the "was" comment.
