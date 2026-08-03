# GATE — #710 pace envelope (branch `fix/recalibrate-pit-bounds` vs `dev`)

**Gate opened:** 2026-08-03 · adversarial gate, findings appended as confirmed.
**Merge base:** `fc73c53` (dev == merge base; all changes are uncommitted worktree edits).

## Scope observed

Diff vs dev touches MORE than the pace half:
- `src/agents/pace_agent.py` (+79) — the nominal change (`_N06_TRAINED_BOUNDS`, `_N06_ENVELOPE`, `_label_against_envelope`).
- `tests/agents/test_n06_envelope.py` (new, untracked).
- `src/strategy/inference/guard_rails.py` (+65/-x) — NOT named in the task.
- `src/strategy/eval/decision_modes.py`, `src/strategy/eval/stint_lengths.py`, `tests/eval/*`, `documents/eval_reports/stint_lengths.*` — NOT named in the task.

## Checklist (claims A–I) — final verdicts

- [x] A. **CONFIRMED** — byte-identical frames dev-vs-branch (executed A/B) + strategy goldens
  PASS + pace-MAE golden reproduces (`test_pace_mae_reproduces_from_featured_laps` PASSED).
- [x] B. **CONFIRMED** — all 12 bounds re-derived, exact match, n=42957.
- [x] C. **CONFIRMED** — train frame contains Years {2023, 2024} only.
- [x] D. **CONFIRMED** — 42.6% / 41.9% / 46.7% exact; 0 inside all three ranges.
- [x] E. **PARTIAL — see F5**: identifiers/flags correctly excluded, but the comment's own
  arithmetic covers 22 of 25 features; `LapsSincePitStop` (continuous, trained 3-77) is
  unbounded with no stated reason and no test pinning the choice.
- [x] F. **CONFIRMED mechanically** (NaN → `unknown`, never compared — executed), with two
  residues: F6 (junk coerced to NaN is silently `unknown` on the `run()` path) and the fact
  that `unknown` never occurred once in 284 real replayed laps (the cited producers currently
  cannot fire on the RSM path).
- [x] G. **CONFIRMED** — checks the exact post-coercion frame `_predict` receives, exactly once
  per agent call (284 verdicts / 284 laps), every live caller routed through it; only the
  offline eval harness (`pace_holdout`) bypasses, acceptably. Residue: F7 (bootstrap frames).
- [x] H. **REFUTED AS PRESENTED — see F1/F2/F3**: the claimed per-feature rates reproduce only
  on the no-dropna frame (not the recipe the bounds declare), and the real inference rate is
  31.0% of laps (88/284 across five circuits; 83% at Monza), not the ~5%-per-feature story.
  The `run_from_state` fabricated defaults are all envelope-invisible (F9).
- [x] I. **CONFIRMED** — 14/14 pass including the data-tier re-measure (executed locally, not
  skipped); firing tests assert the log EFFECT via caplog; the identity test asserts the frame.
  One nuance: the data test re-runs the same `pace_holdout` recipe the bounds were measured
  with (circular in isolation), anchored externally by the pace-MAE golden which pins that
  recipe to the thesis headline 0.4104 — both passed in the same session, so the anchor holds.
- [x] Traps: TyreLife/LapNumber lower bounds ARE dropna-frame artefacts that scream on ordinary
  laps (F3, measured); no second N06 feature path exists (all callers routed, executed grep +
  call-graph walk); sentinel-in-range trap found live (F9: 90.0/300.0/25.0/35.0/50.0 all
  mid-envelope; `tyre_life or 1` masquerades as an outlap).

## Test-suite evidence (executed this session, on the branch worktree)

- `tests/agents/test_n06_envelope.py` — **14 passed** (data-tier included, ran, not skipped).
- `tests/mc/test_strategy_goldens.py` + `test_guard_rails.py` + `tests/eval/test_stint_lengths.py`
  + `test_decision_modes.py` + `tests/agents/test_n15_envelope.py` — **86 passed**.
- Full targeted suite incl. `tests/eval/test_ml_recompute_golden.py` — **106 passed, 1 failed**:
  `test_undercut_auc_pr_reproduces_exactly`. **Attributed OUTSIDE this diff**: the recompute
  returns `pending` because `data/processed/undercut_labeled/undercut_clean.parquet` is absent
  on this machine, and the test's skip-guard (`test_ml_recompute_golden.py:15`) checks only the
  model pkl, not the holdout parquet — so it FAILS where it should SKIP. Nothing in the branch
  diff touches eval/reproduce/calibration or data. Pre-existing test-guard gap, recorded for a
  separate issue.

## Evidence-integrity note

The orchestrator stashed/popped the worktree mid-gate (~90 s on unmodified `dev`). Both of my
parquet measurements imported `src.agents.pace_agent._N06_TRAINED_BOUNDS` successfully, which is
impossible on `dev`, so neither ran inside the window. Tree re-verified after the pop:
`_N06_ENVELOPE` present in `pace_agent.py`, `tests/agents/test_n06_envelope.py` present. No
finding below predates a state I did not re-verify.

## Verified (executed, not read)

- **B — bounds are MEASURED: CONFIRMED exactly.** Rebuilt 2023+2024 through
  `augment_featured_laps` → `_encode_categoricals` → `_add_lag_deg_features` → `dropna(_DROPNA)`.
  All twelve declared (lower, upper) pairs match the measured min/max **exactly** (`==`, not
  approx), including `mean_sector_speed`'s full-precision floats. FuelLoad's odd-looking 0.9615
  is exact because the featured artefact stores FuelLoad rounded to 4 decimals.
- **C — right seasons: CONFIRMED.** Rebuilt frame n=42957 (claimed 42957), `Year` values in the
  frame = {2023, 2024} only.
- **D — Prev_Deg* percentages: CONFIRMED.** Fraction of training rows strictly below 0.0:
  Prev_DegradationRate 42.6%, Prev_CumulativeDeg 41.9%, Prev_DegAcceleration 46.7% (claimed
  42.6/41.9/46.7). All three training ranges straddle 0.0, so a bound could indeed never fire on
  the pinned 0.0. The exclusion reasoning holds for every live caller (both `run_from_state`
  and the orchestrator dict path pin 0.0).

- **A (first half) — labels and nothing else: CONFIRMED byte-for-byte.** Loaded `dev`'s
  `pace_agent.py` via `git show dev:...` as a parallel module and compared
  `_build_feature_row` output dev-vs-branch on three scenarios (ordinary lap, 4-feature
  out-of-range stint opener, None-Position + absent-baseline NaN path):
  `assert_frame_equal(check_exact=True)` passes, `to_numpy().tobytes()` equal, and
  `_predict` returns the identical float to the last digit in all three
  (83.269715090096 / 97.82824754714966 / 90.15864652395248 on both sides). Goldens run
  separately (see below).
- **G — placement: CONFIRMED.** The check runs on `numeric`, the exact frame `_predict`
  receives, after `pd.to_numeric(errors='coerce')` (`src/agents/pace_agent.py:459-461`), and
  exactly once per agent call: my replay recorder saw 57/53/78 verdicts for 57/53/78 laps at
  Lusail/Monza/Monaco. Every live caller reaches it — `no_llm.py:264`, orchestrator dict path
  `strategy_orchestrator.py:1866` (via `run_pace_agent` → `run`), backend
  `strategy.py:632,698`, `mcp_tools.py:451`, `scripts/debug_agent.py:247` all route through
  `PaceAgent.run` → `_build_feature_row`. The one N06 consumer that bypasses it is
  `pace_holdout.load_pace_predictions` (`model.predict` directly), which is the offline eval
  harness, not an inference surface — acceptable, noted so nobody "fixes" it.
- **F (NaN mechanics) — CONFIRMED executed.** `np.nan` arriving through a pandas frame lands in
  `unknown` (never compared: `_is_unknown` catches it because `np.float64` subclasses `float`),
  while an out-of-range `np.float64` is a violation. Verified with a live
  `OperatingEnvelope.check` call on a coerced frame.

## Findings

### F1 — HIGH · `mean_sector_speed` at inference is a SPEED-TRAP reading, and the envelope fires on 79% of Monza laps

`run_from_state` never passes `mean_sector_speed`, so `_compute_derived`
(`src/agents/pace_agent.py:381`) falls back to `prev_speed_st` — the speed-trap value — on
**every** RSM-path call. Trained bound [196.63, 314.97] describes real mean sector speeds;
speed traps at a fast circuit run 315-340. Measured by replaying real 2025 races through
`RaceReplayEngine` → `run_pace_agent_from_state` with a verdict recorder:

| race | laps | laps with ≥1 violation | mean_sector_speed violations |
|---|---|---|---|
| Lusail NOR | 57 | 14 (24.6%) | 10 |
| **Monza NOR** | 53 | **44 (83.0%)** | **42** |
| Monaco NOR | 78 | 7 (9.0%) | 0 |
| Spa NOR | 44 | 14 (31.8%) | 12 |
| Silverstone NOR | 52 | 9 (17.3%) | 3 |
| **total** | **284** | **88 (31.0%)** | 67 |

(The Lusail row independently reproduces the coordinator's `f1-sim --no-llm` run to the exact
per-feature counts: mean_sector_speed 10 / TyreLife 6 / Prev_TyreLife 6 / FuelEffect 3 /
LapNumber 2 / FuelLoad 2 over 14 of 57 laps.) Spa and Silverstone additionally fire
`Prev_SpeedST` itself (2 and 3 laps: 2025 slipstream trap readings above the trained max of
362 km/h) — those ones are honest circuit-drift labels, unlike the fallback-quantity fires.

Concrete failing scenario: Monza lap 3, green flag, nothing unusual — trap 329 km/h → warning.
Every subsequent representative lap warns again; the log is a siren, not a label. Two distinct
defects compound here: (a) the claimed 3.93% rate (design doc M4) was measured on the holdout
frame where the feature IS a real mean sector speed, so it measures circuit drift, not the
fallback; (b) the fallback itself is the same defect class M4 explicitly excluded Prev_Deg* for
("feeding the model a different quantity than it trained on needs its own instrument") — the
Prev_Deg* copy got the analysis, its `mean_sector_speed` twin did not. Lusail outlaps also fire
the LOWER bound (trap 180-181 km/h under pit-limiter vs lower bound 196.6).

### F2 — HIGH · the claimed M4 "2025 out-of-range rates" do not come from the recipe the change itself declares

Executed both ways over `laps_featured_2025` + `augment_featured_laps`:

| feature | claimed (M4) | measured WITH `_DROPNA` (the declared bound recipe) | measured WITHOUT `_DROPNA` |
|---|---|---|---|
| FuelEffect | 4.96% | **0.00%** | 4.96% |
| TyreLife | 2.87% | **0.03%** | 2.87% |
| LapNumber | 1.17% | **0.00%** | 1.17% |
| FuelLoad | 1.17% | **0.00%** | 1.17% |
| mean_sector_speed | 3.93% | 4.04% | 3.93% |
| AirTemp | 1.96% | 1.96% | 1.96% |

The M4 numbers reproduce only on the frame **without** the dropna step, i.e. a different
denominator (n=22760) than the one the bounds are defined on (n=21247 for 2025). The rates
happen to be MORE realistic that way (the dropped rows are exactly where inference goes out of
range), but the doc presents one recipe and quotes rates from another, and neither matches what
the agent actually does at inference (24.6% of Lusail laps, 83% of Monza laps — see F1/F3).

### F3 — MEDIUM · six of the twelve bounds are artefacts of the training frame's structural exclusions, and they fire deterministically on ordinary laps

`TyreLife ≥ 3`, `Prev_TyreLife ≥ 2`, `LapNumber ≥ 3`, `FuelEffect ≥ 0.055`,
`FuelLoad ≤ 0.9615`, `laps_remaining ≤ 75` all exist because the featured artefact drops
pit-in/out and inaccurate laps and `_DROPNA` then drops each stint's first surviving row — so
the training frame **cannot contain** a race start or a stint opener. At inference the sim runs
from lap 1 and calls the agent on every outlap. Measured: lap 1 fires 5 features at once
(6 at Monaco, where `laps_remaining` 77 > 75 joins in), lap 2 fires 4-5, every stint opener
fires 3-4 (`TyreLife`, `Prev_TyreLife`, `FuelEffect`, often `mean_sector_speed` via the
pit-limiter trap). Judgement, since the coordinator asked for it: these ARE genuine
training-range facts — N06 truly never saw such a lap, the prediction there truly is an
extrapolation, and surfacing that standing train/serve mismatch is the envelope doing its job.
What fails is the CLAIM AROUND IT: the change's own test rationale says "a normal lap must be
silent, or the signal is worth nothing on a 57-lap race"
(`tests/agents/test_n06_envelope.py:88`), and a warning on every stint opener of every race
forever — plus F1's trap substitution — is not a quiet log. The bound is right; the noise
budget and the call-site decision (#710's remit) are unresolved, and the M4 "quiet enough"
evidence did not measure the distribution that matters.

### F4 — MEDIUM · FuelLoad false positive: bound measured on a 4-decimal-ROUNDED artefact, checked against an UNROUNDED computation

Training FuelLoad is stored rounded (max stored value exactly 0.9615). `run_from_state`
computes `fuel_load = laps_remaining / max(total_laps, 1)` unrounded
(`src/agents/pace_agent.py:749`). Monaco lap 3: 75/78 = 0.9615384… > 0.9615 → **violation
fired on a lap class the model trained on** (training's own Monaco lap-3 rows store 0.9615 and
sit inside the bound). Concrete: any 78-lap race warns `FuelLoad` on lap 3 while the identical
physical situation was in-distribution at training time. One-ULP-class mismatches between an
artefact-derived bound and a live computation are exactly the kind of edge `pytest.approx`
would not have saved either, because the declared value matches the artefact, not the formula.

### F5 — MEDIUM · claim E is incomplete: the comment's arithmetic does not reach 25, and `LapsSincePitStop` is a continuous count left unbounded with no stated reason

The model consumes 25 features (`xgb_laptime_delta_feature_names.json`, executed read). The
comment (`src/agents/pace_agent.py:105-116`) justifies 12 bounded + 5 identifiers + 2 flags +
3 Prev_Deg* = **22**. Unaccounted: `Stint` (trained range 1-8), `Position` (1-20), and
`LapsSincePitStop` (trained range **3-77**, 75 distinct values — the same lap-count nature as
TyreLife, which IS bounded). On the RSM path `laps_since_pit = tyre_life` so violations would
co-fire with TyreLife, but the orchestrator dict path (`strategy_orchestrator.py:1876`) takes
`laps_since_pit` from `lap_state` and `run()` is public. `test_no_bound_is_declared_over_an_identifier`
covers only the 5 identifiers, so nothing pins whether these three are excluded on purpose.
Position/Stint are defensible as rank/ordinal codes; LapsSincePitStop is not, and no text says why.

### F6 — LOW · the `unknown` path is real but currently unreachable on the replay path, and `to_numeric(coerce)` can silently convert junk into `unknown`

Across all 188 replayed laps, `EnvelopeVerdict.unknown` was empty on every call (executed:
recorder counted zero) — RSM always supplies Position and the stint baseline, so the NaN
producers the docstring cites never fired. The NaN path itself does reach `unknown` correctly
(`envelope._is_unknown` catches float NaN; verified by the unit test run). The residual hole:
in the `run()` direct path a non-numeric value (e.g. a string) survives to
`pd.to_numeric(errors='coerce')`, becomes NaN, lands in `unknown`, and is never reported by
`_label_against_envelope` — an INVALID value silently downgraded to "no value", with no
producer warning covering it. Cannot happen via `run_from_state` (its `float(...)` casts raise
first). Note, not a blocker.

### F8 — MEDIUM-LOW · the two lap-based veto shares point at an artefact that does not contain them (pit half; numbers themselves independently confirmed)

`src/strategy/inference/guard_rails.py:56-58`: "Measured shares are quoted per bound below and
are reproducible from `documents/eval_reports/stint_lengths.md`". Executed check of the
regenerated artefact (md, 42 lines, plus a full JSON key walk): it carries ONLY the four
minimum-stint rows. The `_NO_PIT_BEFORE_LAP` (42/1900 = 2.21%) and `_NO_PIT_LAST_N_LAPS`
(26/1900 = 1.37%) shares — and the 1900 denominator itself (the report says 1895 counted + 5
dropped) — appear in no shipped artefact, and `tests/eval/test_stint_lengths.py` asserts the
ceiling for the four min-stint bounds only. The values are CORRECT (the parallel
`GATE_716_calibration.md` reproduced both by independent reimplementation), so this is not a
wrong number — it is a comment restating a session measurement while claiming artefact
backing, the exact pattern the same commit's docstrings warn about. If the raw laps are
regenerated and either share drifts past the ceiling, nothing that runs re-measures it.

### F9 — LOW · every fabricated default in `run_from_state` is envelope-invisible by construction, and the one that is not tells the wrong story

The trap hunt asked for sentinels that are also real values. All of them are here, deliberately
mid-range: `MISSING_PREV_LAP_TIME_S` 90.0 ∈ [67.7, 149.0], `speed_st or 300.0` ∈ [156, 362],
air 25.0 ∈ [14.5, 33.7], track 35.0 ∈ [16.7, 50.7], humidity 50.0 ∈ [18, 92]. So the envelope
gives ZERO protection against the fabricated-input bug class on the RSM path — a lap running
entirely on defaults labels as fully in-range. The exception: `tyre_life or 1` on a
present-but-None tyre_life produces TyreLife=1 → a violation — but one indistinguishable from
a genuine stint opener, so the log would say "outlap extrapolation" when the truth is "telemetry
lost tyre_life". Not a defect of this change (labelling cannot fix fabrication), but the
docstring's implied coverage should not be read as covering it, and a reader of the warning
stream will misattribute that case.

### F7 — LOW · the 200 bootstrap frames per lap are never checked

`_bootstrap_ci` (`src/agents/pace_agent.py:507`) perturbs the six noisiest features ±2% and
calls the model 200 more times per lap on frames built directly (`pd.DataFrame(row, ...)`),
bypassing `_build_feature_row` and therefore the envelope. A base value just inside a bound is
pushed outside in a fraction of the 200 draws. By design (noise model), and labelling 200
sub-calls would be absurd — but "the check sees the same values the model is about to receive"
(claim G) is true for 1 of the 201 model calls per lap. Recorded for precision, not as a defect.

## What I tried to break and could NOT

1. **The byte-identity claim.** Dev's `pace_agent.py` loaded as a parallel module, three
   scenarios including out-of-range and the None/NaN path: frames byte-equal
   (`to_numpy().tobytes()`), predictions equal to the last digit. The labelling really is
   read-only — `check` builds new objects and never touches the frame.
2. **The twelve bound VALUES.** Re-derived independently from the parquets through the declared
   recipe: every declared number matches the measured min/max exactly, n=42957, seasons
   2023+2024 only. They are measured, not chosen — including the full-precision
   `mean_sector_speed` pair. (The rounding hazard I went hunting for in `pytest.approx` does
   not exist here because the artefact itself stores FuelLoad rounded; the hazard shows up at
   INFERENCE instead — F4.)
3. **The Prev_Deg* exclusion.** Percentages exact; 0.0 comfortably inside all three trained
   ranges; both live entry paths pin 0.0, so a declared bound could truly never fire. The
   exclusion argument survives attack.
4. **A second, unlabelled N06 path.** Walked every consumer of `run_pace_agent`,
   `run_pace_agent_from_state` and the model file: orchestrator (both paths), no_llm engine,
   backend endpoints, MCP tools, debug script — all route through `PaceAgent.run` →
   `_build_feature_row`. Only the offline eval harness bypasses, and it should.
5. **Double-checking per lap.** Recorder counted exactly one verdict per agent call on 284 real
   laps; the bootstrap loop does not re-trigger it.
6. **The NaN → `unknown` mechanics.** np.float64 NaN from a coerced frame lands in `unknown`,
   never compared (executed micro-test); the unit test asserting silence on a NaN feature
   passes for the right reason.
7. **The envelope class itself.** Inverted-bounds rejection, ignore-undeclared-keys, and
   frozen/immutability behaviour all held under direct probing.
8. **The pit half's recalibrated values** (context for this branch): the regenerated report and
   the shipped constants agree, the ceiling test passes, and the parallel GATE_716_calibration
   reproduced every share independently — I did not find a number in `guard_rails.py` that is
   wrong; F8 is about a false provenance pointer, not a false value.
9. **The stash window.** Both of my parquet measurements imported branch-only symbols, so
   neither ran against the temporarily-reverted tree; the tree was re-verified afterwards.

## Fix list (by value, then risk)

1. **F1 — decide what `mean_sector_speed` means at inference before the envelope ships its
   warning.** Options in rising cost: (a) exclude it from `_N06_TRAINED_BOUNDS` with a comment
   naming the fallback-quantity defect (mirror of the Prev_Deg* paragraph — smallest change,
   honest); (b) declare the bound against the distribution the feature ACTUALLY receives at
   inference (the speed-trap range), which is a different envelope for a different pipeline;
   (c) fix the fallback itself (feed a real mean sector speed or NaN) — that is a model-input
   change with golden impact, its own issue. Shipping as-is means 42 warnings per driver-race
   at Monza and a log nobody will read by lap 10.
2. **F3/F2 — restate the noise budget honestly at the call site and in M4.** The warning fires
   on every race start and every stint opener by construction (measured 9-31% of laps on
   normal circuits). Either downgrade those structural fires to DEBUG (they are true but
   carry no news) while keeping WARNING for the non-structural ones, or document that the
   consumer of this log must expect them. Update AUDIT_716_710_design M4 to name the frame its
   rates were measured on (no-dropna, n=22760) — as written it implies the bound recipe's frame
   and understates inference reality by ~6x.
3. **F4 — kill the FuelLoad rounding false-positive**: either round the inference computation
   to 4 decimals to match the artefact (`round(laps_remaining / total_laps, 4)` — matches
   training semantics exactly), or widen the declared upper bound to the unrounded formula
   value. One line either way; without it every 78-lap race warns on a trained lap class.
4. **F5 — account for the last three features.** One comment line each for Stint/Position
   (rank/ordinal, excluded on the same "no range" argument) and a real decision for
   `LapsSincePitStop` (bound it, or say why not); extend the two exclusion tests to pin all
   thirteen unbounded names so the enumeration cannot drift silently.
5. **F8 — either add the two lap-based veto shares to the stint-lengths artefact (they already
   have a home: the report), or soften the guard_rails comment to say where they actually came
   from.**
6. **F6 — decide whether `run()`-path junk-to-NaN deserves a log line** (a one-line `elif
   verdict.unknown:` DEBUG would do), or document that `unknown` is intentionally silent
   everywhere, not just for the two named producers.
7. **Pre-existing, separate issue: `test_ml_recompute_golden.py:15` skip-guard** should also
   require `data/processed/undercut_labeled/undercut_clean.parquet`, so the data tier skips
   instead of failing on checkouts without the labeled holdout (it failed exactly that way in
   this session, on a change that touches nothing near it).

## Coordinator's direct question, answered

**"Is TyreLife ≥ 3 a genuine training-range fact or a dropna artefact?"** Both, and the
distinction decides the fix: the featured artefact + `_DROPNA` structurally exclude race
starts and stint openers, so N06 has genuinely never seen such a lap — the prediction there
IS an extrapolation and the label is TRUE. But because the exclusion is structural, the
warning is guaranteed on laps that occur in every race forever; it separates "laps the
pipeline can't train on" from "laps the sim must serve", which is a standing train/serve gap,
not an anomaly. A quarter of laps warning (my 5-circuit measurement: 9-83%, mean 31%) fails
the change's own silence criterion — the bound should stay DECLARED (it is a fact) but the
structural fires need a different volume than the genuinely anomalous ones (fix 2), and
`mean_sector_speed` should not be in the declared set at all until its fallback is fixed
(fix 1). The orchestrator's Lusail run is confirmed exactly; its ~5% holdout-based estimate
was the wrong distribution, as suspected, and the right one is worse.
