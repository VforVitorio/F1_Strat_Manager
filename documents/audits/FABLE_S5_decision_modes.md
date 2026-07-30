# FABLE S5 gate — adversarial audit of MEASURE_S5_decision_modes_2025.md

**Date:** 2026-07-30 · **Auditor:** adversarial gate (no repo file modified except this one)
**Under test:** `documents/audits/MEASURE_S5_decision_modes_2025.md` — a measurement + conclusion,
no code. The scratchpad JSONL and `baseline_dm.json` are treated as SUSPECT; everything numeric is
re-derived from the committed baseline JSON and from FRESH runs of the harness.

## Checklist

- [x] **A** — numbers recomputed and reproduce ✓; the COMPARISON hides a third run (F1) and the denominator drift is fully accounted (Silverstone −15 dominates).
- [x] **B** — matching re-derived (46 ✓, and 198 available); `af3a24a` attack executed (F4); unseeded-baseline noise measured with two old-code reruns (F5/F10).
- [x] **C** — verified from diff and execution (F6). Not a confound.
- [x] **D** — CONFIRMED with fresh runs at w=5/7/10, two races, identity level (F9/F9b). Innocent explanations tested and excluded.
- [x] **E** — 11 verified and same-identity, but the w=5 exact bucket is 13, of which 2 are window artefacts (F3/F9b).
- [x] **F** — premise true (hygiene.py); DIRECTION unsupported: measured dry-circuit season effect ≈ +0.9 pts (claim-F section).
- [x] **G** — wet verified; residual anchor measured at 46.3% vs 25-30% dry controls; causal link to the 71% decline NOT established (F7).
- [x] Bug-class sweep: `no_data` empty everywhere; selection effects (window truncation, overlapping windows, rescued no-calls) hunted — only the rescued no-call fired, and it is benign (F9b).

## Findings

### F1 — HIGH · The write-up omits its own strongest run: a post-fix MIXED-sample rerun sits uncommitted in the working tree

`git status`: `M documents/eval_reports/decision_modes.json` (+ `.md`). The tree version is a
**new-code rerun of the SAME mixed six races** (`harness_sha 1f0ec9d`, generated 2026-07-30T08:37):
**90 scored / 198 (45.5%), 78 no_call (39.4%), rails 30, mean signed −3.30**. The committed version
(01cb4e6, byte-identical to the stashed `baseline_dm.json` — verified) is the old-code 40/198.
Recomputed transitions between the two mixed runs, matched per stop on (year, race, driver,
actual_lap): **matched 198/198, `no_call→scored` 57, `scored→no_call` 7, net +50** — per race:
Barcelona +15/−0, Monaco +8/−6, Marina_Bay +9/−0, Silverstone +14/−0, Lusail +4/−1, Monza +7/−0.

Consequences for the write-up:
- The clean code-only comparison (64.6% → 39.4% on 198 matched stops, same season mix, same stops)
  existed and is *stronger* than the headline; §2 instead nets the code effect (−25.2 pts) against an
  opposing sample effect (39.4% → 46.1%, +6.7 pts) without saying either number.
- §2's causal case uses 46 matched stops when **198 matched stops were available**; the discarded 152
  contain the only substantial regression signal — **Monaco 2023: 6 stops the old code scored and the
  new code declines** — which the "+10 net, (none the other way)" framing never surfaces.
- The tree file is uncommitted: whoever commits next silently replaces the "committed baseline" the
  write-up cites. Not my mutation — left by the implementer's run; I did not touch it.

### F2 — MEDIUM · §3's table compares two DIFFERENT samples column against column

Reproduced from the run data: the w=5 column (−5:36, −4:9, −1…−3:10, 0:20, total 75, mean −3.08) is
the **full six-race 2025 sweep**; the w=10 column (total 28, mean −5.04 — both recomputed exactly)
is **Monza + Marina_Bay only**. The same-sample w=5 histogram for Monza+Marina_Bay is
`{−5:12, −4:1, −2:1, 0:13}` (27 scored). So the table's headline juxtaposition "−5: 36 → 0" is
cross-sample; the honest same-sample statement is "−5: 12 → 0", and "−3.08 vs −5.04" compares
different race sets. The pin conclusion itself is being re-tested with fresh runs (see F9), but the
table as printed is not a like-for-like comparison and the write-up nowhere says so.

### F3 — MEDIUM · "11 stops sit at exactly 0 at both window widths" — at w=5 the same two races have THIRTEEN at 0

From the same-sample histograms above: Monza+Marina_Bay at w=5 have **13** stops at offset 0; the 11
is the **w=10** count. At most 11 of the 13 "exact agreements" survive widening — i.e. at least 2 of
the stops §3 calls genuine agreements are themselves window artefacts (the stack would have called
the stop earlier had it been asked). This *strengthens* the direction of the §3 finding but the
"stable second mode of 11" as stated is wrong at w=5, and the identity check is what decides it
(fresh-run verification in F9).

### F4 — MEDIUM · Claim B's mechanism sentence is wrong in principle; correct only by luck of the data

§2: "the transitions above are `no_call -> scored`, which the rails cannot produce." Two errors:

1. **The rails CAN produce `no_call→scored`**: `af3a24a` suspends the early-race and min-stint
   bounds *inside the stack* when `sc_currently_active` is true (an RCM-confirmed neutralisation,
   `pit_strategy_agent.py:530-564`). A PIT emitted on a window lap under SC that the old rails
   overrode into STAY_OUT is exactly a stop that moves `no_call→scored` when the suspension lands.
2. **`af3a24a` does NOT "move the RAIL buckets" of this eval**: `guard_rail_block`
   (`decision_modes.py:203-206`) never passes `sc_active`, and the per-race rail counts are
   **identical (30 = 30, race by race)** between the two mixed runs. The rails 30→21 delta in §2's
   table is the season change, not the commit.

Executed check that rescues the conclusion: for all **64** moved stops (baseline vs mixed rerun),
no TrackStatus-4/5/6 lap falls inside the ±5 window (2023 Barcelona / 2023 Monaco / 2024
Marina_Bay / 2024 Silverstone / 2025 Monza have zero neutralised laps in the relevant ranges;
2025 Lusail's SC is laps 7–10, its moved stops are at lap 32). So the attribution survives, but on
evidence the write-up never ran, and its stated reasoning would have mis-attributed in any race
where a stop's window brushed an SC.

### F5 — MEDIUM · The baseline run predates the seed fix: its per-stop verdicts are non-repeatable inference

The committed baseline was generated at `80f1fa7` (2026-07-29 10:25); the MC-Dropout seed fix
`8d68a9e` ("seed the tyre model's MC Dropout so inference is repeatable") is **not an ancestor**
(`git merge-base --is-ancestor` fails). Every matched-stop transition against that baseline
therefore compares a seeded run against one draw of an unseeded process. Quantification via two
fresh old-code runs is in flight (F9); until then "+N moved, from code changes alone" carries an
unmeasured noise term the write-up does not disclose. (The fresh new-code determinism check DID
pass: two independent Monza w=5 runs and the tree rerun vs the sweep agree stop-for-stop, 46/46
and 20/20.)

### F6 — VERIFIED · Claim C: `84dc4b6` is behaviour-identical at the default

Full diff touches only `src/strategy/eval/decision_modes.py`: threads
`risk_tolerance=DEFAULT_RISK_TOLERANCE` (0.5) where the old code hardcoded
`risk_tolerance=0.5` (old line 289), into a `RaceState` whose field default is also 0.5
(`strategy_orchestrator.py:247`). Execution evidence: the sweep and the tree rerun (both through
the new signature) reproduce the baseline-era stop set and agree with my fresh runs stop-for-stop.
Not a confound.

### F7 — MEDIUM · Claim G: wet verified; the named mechanism is real but measured to be an INSUFFICIENT explanation

- **Wet: verified.** Silverstone 2025 raw data: 608/826 laps (73.6%) on INTERMEDIATE, rainfall in
  18.1% of weather samples, `IsAccurate` share 0.600 — i.e. exactly the "~40% of laps fail N04's
  filter" figure the S4 report claimed.
- **Residual anchor: measured, not inherited.** Replaying the exact windows the eval evaluates
  (post-`bc64b94` reconstruction): **Silverstone 2025 = 274/592 evaluated laps (46.3%) still anchor
  N06 on the 90.0 constant** (worst: ANT 19/20, HAD 9/11).
- **But the dry controls are 25–30%** (Monza 2025: 30.0%, Lusail 2025: 25.2%): the eval's windows
  straddle stint boundaries, and the per-stint reconstruction has no previous survivor on early
  stint laps, so heavy anchoring is endemic to this tier's sample shape, not a wet-race special.
- **And anchor share does not rank-order decline**: Lusail 25.2% anchored → 53.8% declined; Monza
  30.0% anchored → 35.0% declined. So "71% declined because of the 90.0 anchor" is not established
  by these data; the wet race also feeds the stack a compound (INTERMEDIATE) that takes the
  `_DEFAULT_MIN_STINT=10` fallback in the rails (`guard_rails.py:41-42`) and that the tyre stack
  never trained on — unexamined co-mechanisms. The write-up's hedge ("recorded rather than averaged
  away") is fair; the specific causal sentence is over-claimed.

### F8 — LOW · Cited line number is stale

§3 cites `decision_modes.py:439` for `chosen = _first_pit_lap(...)`; in the tree under audit
(57a7087) that line is `decision_modes.py:434` (:439 is inside the `no_call_in_window` append).
`decision_modes.py:76` and `strategy_orchestrator.py:625` are correct.

### Claim A — recomputation PASSES; the comparison itself is the problem (see F1)

Recomputed from the committed baseline JSON and the sweep JSONL: baseline 40/198 scored (20.2%),
128 no_call (64.6%), rails 30, `no_data` 0 ✓; 2025-only 75/178 (42.1%), 82 (46.1%), rails 21,
`no_data` 0 ✓; the write-up's per-race table matches row for row ✓. Denominator drift 198→178 fully
accounted: Barcelona −2, Monaco −1, Marina_Bay −2, **Silverstone −15** (46 stops in 2024 vs 31 in
2025), Lusail ±0, Monza ±0. The `no_data` bucket is empty in every run — nothing hidden there.

### Claim F — premise TRUE, direction UNSUPPORTED by the write-up's own data

- Premise verified from the repo: models train on 2023+2024 and test on 2025
  (`src/strategy/eval/hygiene.py:17-18` "they train on 2023+2024", threshold provenance lines 84-148;
  consistent across N12/N14/N16 entries). The four 2023/2024 baseline races are in-train seasons.
- Direction attacked with the matched-circuit pairs available from the two new-code runs
  (mixed rerun vs 2025 sweep, same code, same circuits, only season differs):
  Barcelona 44.2% → 36.6% (**2025 declines LESS**), Monaco 34.2% → 45.9% (more),
  Marina_Bay 32.0% → 30.4% (less), Silverstone 37.0% → 71.0% (the wet race).
  **Excluding the wet race the season effect is +0.9 pts (37.7% → 38.6%)** — indistinguishable from
  zero and not sign-stable per circuit. Under the OLD code the baseline's own per-race data say the
  same: in-train-season races declined 63.8% (97/152) vs out-of-sample 67.4% (31/46), a 3.6-pt gap
  fully confounded by circuit, with the WORST race in the whole baseline being in-sample Barcelona
  2023 (79.1%).
- Verdict: "in-sample → declines less → the committed 64.6% is optimistic" is an argument wearing a
  measurement's clothes. The measured season effect on dry circuits is ≈0; the aggregate 2025
  penalty is entirely the wet race the owner declared out of scope. The sample redesign itself is
  still defensible (out-of-sample is the production condition) — but not for the quantitative reason
  §1 gives, and the write-up's "the direction matters" paragraph should be retracted or re-labelled
  as hypothesis.

### F9 — Claim D (CENTRAL): CONFIRMED, with fresh three-width evidence, and it is even stronger than written

Fresh runs (my own harness invocations, `DECISION_WINDOW_LAPS` monkeypatched in-process only), 2025
Monza, current code:

| width | offsets (scored=11 at every width) | mean signed |
|---|---|---|
| w=5 | −5:**5**, −4:1, 0:5 | −2.64 |
| w=7 | −7:**5**, −5:1, 0:5 | −3.64 |
| w=10 | −10:**5**, −7:1, 0:5 | −5.18 |

The early mass sits at exactly **−W at all three widths** and the mean tracks the window. The third
width confirms the pin — this is not a two-point coincidence. Innocent explanations, each tested:

- **Eligibility drift:** none at Monza — buckets identical at every width (scored 11, no_call 7,
  min_stint 1, closing 1). (Marina_Bay: one `no_call` rescued at w=10, see below — it only *adds* a
  large-offset entry; it cannot relocate the −5 mass.)
- **Rail re-bucketing:** impossible by construction (`guard_rail_block` is width-independent) and
  measured identical at all widths.
- **Replay-span change:** `run_lap` is stateless across laps — `_decisions_in_window`
  (decision_modes.py:306-308) passes no `memory`, so evaluating extra laps cannot change the
  decision on a given lap. Corroborated in the data: HAD lap 32 keeps offset −5 at w=7 (the two
  extra earlier laps produced no pit call) yet shows −10 at w=10 — the stack has *two* pit-calling
  regions, more evidence that "first pit lap" is not a location estimate.
- Per-stop movement is exactly the pin: GAS/PIA/STR/HAM at −5 → −7 → (−10 or interior); NOR at −4
  (w5) → −7 → −10 shows even **interior** offsets are first-occurrence artefacts, which goes beyond
  the write-up's own claim.

Determinism side-evidence collected on the way: two independent fresh Monza w=5 runs, the sweep
JSONL, and the tree rerun agree **stop-for-stop** (20/20, and 81/81 on the 2023 races, 46/46 on
2025 Lusail+Monza) — the seeded new-code path is reproducible across processes.

### F9b — Claims D and E closed with the Marina_Bay fresh runs

Fresh Marina_Bay 2025: w=5 `{−5:7, −2:1, 0:8}` (16 scored / 7 no_call), w=10
`{−10:5, −9:3, −8:1, −7:1, 0:6, +8:1}` (17 scored / 6 no_call). Combined fresh Monza+MB at w=10:
**`{−10:10, −9:3, −8:1, −7:2, 0:11, +8:1}`, n=28, mean −5.04 — identical to the write-up's w=10
column**, so the implementer's w=10 numbers are independently corroborated, not just re-read.

- **D verdict: CONFIRMED.** The −5 mass is 0 at w=10 in both races; every −5 stop relocates to
  −7…−10; the third width (Monza w=7) pins at −7. `mean_signed_error` is a property of the window.
  The write-up's central finding survives adversarial reproduction.
- **E verdict: count and identities VERIFIED, framing incomplete.** Combined zeros: w=5 has **13**,
  w=10 has **11**, and the 11 are a strict subset of the 13 (identity-checked). So "11 sit at 0 and
  stay there" is true — but **two of the w=5 "exact agreements" (MB NOR lap 26: 0→−10, MB OCO lap
  30: 0→−8) are themselves window artefacts**: the stack would have called those stops 8-10 laps
  earlier had it been asked. 2 of 13 (15%) of the published exact-agreement bucket is artefact. This
  *strengthens* §3's thesis and *weakens* §3's "genuine agreements" consolation — the write-up
  should say 13→11, not just 11.
- The w=10 “+8” entry is the rescued no-call (MB GAS lap 51: `no_call` at w=5 → chosen +8 at w=10) —
  the only eligibility drift in either race, and it adds a *late* call, so it cannot manufacture the
  early-edge mass.
- The overlapping-window cross-talk mechanism I hunted (a second stop's w=10 window reaching back
  into the first stop's pit-call region — MB HUL [25, 44]) did NOT fire: HUL's second stop stays
  `no_call` at both widths. Checked and excluded.

<!-- appended as confirmed -->
