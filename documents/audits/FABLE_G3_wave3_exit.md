# FABLE G3 — Adversarial exit gate: Wave 3 (PR #765 + submodule 6e7a20e)

- **Date:** 2026-07-31
- **Scope:** `dev` @ `8bc9d3d` (parent commit `4520431`), submodule `src/telemetry` @ `6e7a20e`. Closes #741, #742, #746, #750.
- **Mandate:** twin sweep first (previous-lap producers, restated constants, `pace_delta_s` axis), then attack the specific claims (backend anchor shapes, horizon mutants, prompt-test regex, the 341-SOFT-stint measurement).
- **Rules:** no repo file modified except this report; mutations via `cp` backup + restore + diff-verified; no LLM calls; evidence executed, not read.

## Checklist

- [x] A. Twin sweep: every producer/deriver/defaulter of a "previous lap time" (parent + submodule) → F-01 + census table
- [x] B. Twin sweep: restated constants in prose vs computed-against values → F-02, F-07, F-08
- [x] C. Twin sweep: every construction of `pace_delta_s` / RaceState (parent + submodule) → census table, no fifth surface
- [x] D. Backend anchor: both frame shapes, real 2025 data, post-pit lap, first lap of stint, worse-than-90.0 check → F-05
- [x] E. #742 horizon mutants re-run + a fourth attack on the horizon distinction → F-04
- [x] F. #741 prompt-test regex attack (empty-set / green-while-disagreeing) → F-03, executed
- [x] G. Re-derive: 341 SOFT stints, median 15, max 50, 33.7% > 18 → F-06, exact
- [x] Close: what I tried to break and could NOT

## Findings

### F-01 · HIGH · Sweep A — the dict-path orchestrator still has BOTH prev-lap defects #746's class describes, and it crashes on the honest value

`src/agents/strategy_orchestrator.py:1849`:

```python
prev_lap_time  = lap_state.get("prev_lap_time", 92.0),
```

Two defects in one line, both catalogued classes:

1. **Two-arg `dict.get`** — the default fires only when the KEY is absent. `RaceStateManager.get_driver_state` emits `prev_lap_time: None` (present key, None value) whenever no surviving predecessor exists. `pace_agent.py:709-724` carries a 15-line comment explaining exactly why the two-arg form is wrong for THIS key and uses `or 90.0` instead. The twin one module over never got the fix.
2. **A restated default** — 92.0 here vs 90.0 in `pace_agent.py:725` for the same quantity. Two sentinels for one concept; neither derived from the other.

**Executed evidence** (models loaded, no LLM): calling `run_pace_agent(prev_lap_time=None, ...)` — exactly what line 1849 passes when the lap_state carries an explicit None —

```
TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'
  File "src/agents/pace_agent.py", line 623, in run
    delta_vs_prev = lap_time_pred - prev_lap_time
```

`_predict` first turns the None into NaN (`lap_time_pred = nan`), then `delta_vs_prev = nan - None` raises. So `run_strategy_orchestrator` — the **exported public entry point** (`src/agents/__init__.py:99`, named as THE usage example in the module docstring line 17) — crashes on any lap_state whose `prev_lap_time` is the honest None. Failing scenario: any caller bridging `RaceStateManager.get_lap_state()` output (flattened) into the dict path on the first lap of a stint.

Mitigation of severity: no in-repo live surface calls the dict path today (CLI/arcade/backend all use `_from_state`); it is public API and documented, hence HIGH on the defect-class axis but with a currently-dead blast radius in this repo.

### F-02 · MEDIUM · Sweep B — the SAME two prompts still restate `_MIN_STINT_LAPS`, the guard-rail lap bounds, and `CLIFF_IMMINENT_LAPS` as literals

#741 made exactly ONE number derived (`_STINT_CAPACITY_LAPS['SOFT']`). Rendered-prompt extraction (executed, both prompts) shows the following literals that the code ALSO computes against, none interpolated, none covered by the new test's `<=`-only regex:

| prompt text (both prompts unless noted) | code constant | file:line |
|---|---|---|
| "SOFT: current tyre_life must be >= 8 … MEDIUM: >= 12. HARD: >= 15." | `_MIN_STINT_LAPS = {"SOFT": 8, "MEDIUM": 12, "HARD": 15}` | `src/strategy/inference/guard_rails.py:41` |
| "NEVER … before lap 5" | `_NO_PIT_BEFORE_LAP = 5` | `guard_rails.py:38` |
| "NEVER … when remaining laps <= 3" | `_NO_PIT_LAST_N_LAPS = 3` | `guard_rails.py:39` |
| "cliff P10 < 2" exception | `_CLIFF_P10_SAFE = 2` | `guard_rails.py:40` |
| "laps_to_cliff <= 3 → recommend PIT_NOW" (pit prompt) | `CLIFF_IMMINENT_LAPS = 3` | `src/agents/pit_strategy_agent.py:142` |
| "MEDIUM: suitable for 12-30 remaining laps" (both prompts) | `_STINT_CAPACITY_LAPS['MEDIUM'] = 30` upper bound + `_MIN_STINT_LAPS['MEDIUM'] = 12` | `pit_strategy_agent.py:91` |

The `_MIN_STINT_LAPS` triple sits **in the same prompt, two sections above the line #741 fixed**, in a constant that was converted to an f-string in this very PR — interpolating them was zero marginal cost and did not happen. Every row is the exact defect class the PR names ("a value restated somewhere it is not derived"); guard-rail drift is enforced-after-the-fact so the outcome class is prompt-vs-rail disagreement, not an unrailed decision.

### F-03 · MEDIUM · Attack F executed — the #741 test stays GREEN while both prompts and the table disagree about MEDIUM

The gate asked for "a phrasing that keeps the test green while the prompts and table disagree". It does not need a phrasing change — it exists today for MEDIUM. **Executed**: with `_STINT_CAPACITY_LAPS['MEDIUM']` mutated 30→32 (cp backup, restore verified byte-identical):

```
MEDIUM 30->32 mutant, prompt still says '12-30':  pytest exit = 0   (3 passed)
table says MEDIUM = 32 | prompt says: MEDIUM: suitable for 12-30 remaining laps.
```

The `_BOUND` regex (`tests/agents/test_prompt_constants_match_tables.py:50`) only parses `<=`-phrased clauses; both prompts state MEDIUM's capacity as the range "12-30" (`pit_strategy_agent.py:661`, `strategy_orchestrator.py:1671`), invisible to it. Failing scenario: exactly #741's, one compound over — at `laps_remaining` 31-32 the selector passes MEDIUM and both prompts call it unsuitable, so the band is decided by prose again, with the guard suite green. The test's docstring claim "every compound it can find" is literally true and materially misleading: it can only find SOFT.

### F-04 · VERIFIED (claim E holds) + one nuance · the three horizon mutants each go red on the named test

Executed, each with `cp` backup and byte-identical restore (`filecmp` after every restore):

| mutant | result |
|---|---|
| `rival_time_deltas`: `is_pitting` → `stop_pending is True` | RED on `test_window_horizon_charges_only_the_rival_pitting_this_lap` |
| `_terminal_gaps`: drop `and not rival.is_pitting` | RED on `test_terminal_horizon_charges_an_owed_stop_exactly_once` |
| `rank_targets`: add `and not rival.is_pitting` | RED on `test_post_cycle_horizon_charges_every_known_obligation_alike` |

Each mutant pattern was grepped in the pristine file (`count == 1`) before substitution — no pristine-file mutant runs.

**Fourth attack (adapter-level unification):** mutating `strategy_orchestrator._rival_states_from_lap_state` (`:824`) to `is_pitting=... or bool(per_rival_pending.get(driver))` destroys the window/terminal distinction for EVERY live surface (all of them build rivals through this adapter) while the three horizon tests — which construct `RivalState` directly — stay green. Executed over the PR's own scope (`tests/mc` + `tests/agents`, 242 tests): **caught anyway**, by `test_mc_is_a_real_decision.py::test_the_named_target_is_the_car_we_will_be_racing_not_the_first_in_the_list` and `test_projection_golden.py` (2 failed, 240 passed). Nuance worth recording: the horizon FILE does not defend against adapter drift; the projection golden does. If the golden is ever regenerated while such a drift is live, the horizon distinction has no dedicated guard at the adapter seam. LOW.

### F-05 · MEDIUM · Attack D — the anchor fix is faithful to N04 (904/904), but the commit's out-lap claim is refuted by execution, and the function ships with ZERO tests

Executed against the endpoint's own served frames (`_get_race_laps_df(2025, "Lusail")` + `laps_featured_2025.parquet`):

**What holds (verified, executed):**
- **904/904 exact agreement** between the raw-frame fallback and N04's own `Prev_LapTime` for every Lusail 2025 driver-lap present in both frames. The transform is reproduced, not reinterpreted.
- In-laps served correctly (NOR lap 25 → 85.97, the last surviving lap, not `lap-1`).
- SC laps served (27 SC-lap anchors, e.g. VER lap 7 → 87.123).
- Both frame shapes work: a `LapTime`-only frame (timedelta) and a `LapTime_s` frame both return 85.304 for NOR lap 28; a Stint-less frame degrades to None honestly.
- **No worse-than-90.0 anchor found**: across all 67 raw-only laps that get an anchor, values span 83.714–88.866 s — never an out-lap, in-lap or SC lap time.

**What breaks:**

1. **The commit message's mechanism claim is false for out-laps.** It says the lookup "must not require THIS lap to have survived the filter, or it returns None on exactly the out-laps and Safety Car laps the raw frame exists to serve; it takes the last surviving lap BEFORE the current one instead." Executed census: **out-laps total = 42, anchored = 0, None = 42.** The Stint scoping makes the out-lap the first lap of its stint, so there is never an earlier lap inside it — the design rationale's first-named beneficiary is structurally unservable. The SC half of the claim is true (27 served). The in-code comment `strategy.py:868-871` ("Anchoring an out-lap on the last good racing lap before it is still the right answer") describes behavior the function cannot produce. Downstream: out-laps AND the first flying lap after every stop (NOR lap 27 → None) fall through to the pace agent's 90.0 — which matches N04's NaN semantics, so the BEHAVIOR is defensible; the NARRATIVE (commit, PR body, comment) is not. A comment naming the wrong mechanism is a catalogued class here.
2. **The frame-shape comment at `strategy.py:860-861` has the frames backwards**: "A frame without the quality flags excludes nothing on them, which is the raw frame's case". Executed: the raw parquet CARRIES `IsAccurate` and `Deleted` (it is the featured frame that lacks them, harmlessly). Wrong-mechanism comment, LOW on its own, listed here because two of them in one 40-line function is how the next fix gets mis-aimed.
3. **Zero tests.** Submodule commit `6e7a20e` ships no test file; `grep` finds no test referencing `_prev_lap_time_for_row` in the submodule or the parent. Issue #746's acceptance criteria explicitly demand "a test asserting the lap after a stop is not anchored on the out-lap" and the None-semantics. The most intricate function of the wave — the one whose first draft "would have crashed" per the PR's own verification note — is the only one shipped untested. The PR's "242 across tests/mc + tests/agents" is parent-only and cannot cover a submodule function.

### F-06 · VERIFIED · Attack G — the 341-SOFT-stint measurement reproduces exactly

Re-derived with the same instrument (`measure_stint_lengths()`, defaults):

```
races: 71 | SOFT n = 341 | median = 15.0 | max = 50.0
share > 18: 33.7 % | share > 15: 45.7 % | total counted = 1785
```

Every figure quoted in the commit message, the PR body and the test docstring matches, including the strict `>` reading of "run longer than 18" and the 1785 total the issue cites. No finding.

### F-07 · MEDIUM · Sweep B — `0.522` is a config-loaded threshold restated as a prompt literal, so the prompt can disagree with the tool output in the same conversation

`pit_strategy_agent.py:618` (system prompt): "If P(undercut_success) >= 0.522 for any rival → recommend UNDERCUT." The code does NOT own that number: `:230` loads `self.undercut_threshold = uc_cfg['best_threshold']` from the model's JSON config, `:1201` compares against it, and `:1207` **prints the live threshold into the tool response** (`threshold={agent.cfg.undercut_threshold}`). Retune the model, republish the config, and the LLM receives two different thresholds in one conversation — the tool says one number, the system prompt insists on 0.522. Same defect class as #741 with a sharper failure mode, because here the second copy is not even a repo constant but a data artifact.

### F-08 · MEDIUM · Sweep B — the two prompts disagree TODAY about the same derived figure: ~13 vs ~9 positions, and the 13 derives from the constant the redesign retired

`strategy_orchestrator.py:1664` (rule 2, six lines above the line this PR edited): "Pit cost ~22s vs ~1.5s recovery = **~13 positions lost**." That 13 is `20 s / POS_GAP_S(1.50)` — `POS_GAP_S` being the legacy-scoring constant `measure_mc_tables.py:575` records as "retires_constant". The pit agent's prompt was already corrected to the measured geometry (`pit_strategy_agent.py:636-639`): "worth **~9 positions, not 13**: 13 positions is the Safety Car bunched-field figure (median gap 1.4795s)" — 20 s / 2.226 s ≈ 9. So one prompt explicitly refutes the number the other prompt still teaches, in the same run, to the same orchestrating LLM. One copy fixed, its twin not — the wave's own defect class, inside a file the wave edited.

Related boundary nit (LOW): the pit prompt states REACTIVE_SC at `sc_prob >= 0.30` (twice) while the code routes on `sc_prob_3lap > CFG.sc_prob_threshold` (`:587`, `:1988`) — strict vs inclusive at exactly 0.30, and 0.30 is itself a restated literal for `CFG.sc_prob_threshold` (`:127`).

### Sweep A — census of every previous-lap producer found (parent + submodule)

| producer | what it produces | reproduces the training transform? | reaches a model? |
|---|---|---|---|
| `RaceStateManager._precompute_prev_lap_times` (`race_state_manager.py:195-221`) | N04 transform: filter-then-group-shift | YES (read, filter terms verified) | yes — N06 anchor via lap_state |
| backend `_prev_lap_time_for_row` (`strategy.py:804-877`, this wave) | featured `Prev_LapTime`, else N04 reconstruction | YES — **executed 904/904 match** (F-05) | yes — `/lap-state`, `/simulate` |
| `pace_agent.py:725` `d.get('prev_lap_time') or 90.0` | 90.0 sentinel only on genuinely-missing | consumer-side default (documented) | yes — N06 |
| **`strategy_orchestrator.py:1849` `lap_state.get("prev_lap_time", 92.0)`** | **None passthrough + a 92.0 twin default** | **NO — F-01, crashes on None** | yes (dict path, public API) |
| `tire_agent._add_prev_cols` (`:554-569`) | `shift(1).fillna(current)` over the handed-in frame | its reference is N08/N09 (not N04); on the featured frame shift(1) = previous surviving lap by construction; the `fillna(current)` self-referential fill only touches the window's first row and only survives into `LapTime_Delta`/speed deltas, which N09's manifest excludes as lap-time shortcuts. Pre-existing, out of this wave's blast radius — flagged, not scored | TCN input frame |
| `race_situation_agent._build_sc_features` (`:991`) `LapNumber == lap_number - 1` | previous-LAP AGGREGATE row for N14's per-lap SC features | N13/N14 aggregate laps by lap number, not by stint survival — this is that pipeline's own transform, not a stint anchor | yes — N14 |
| CLI `run()` / arcade `_step_once` / simulator `:843,924` bookkeeping `prev_lap_time` | loop variable, **no longer read** by `_build_race_state` (verified in current tree: only the signatures keep it) | n/a — dead input, docstrings say so | no |
| eval `pace_holdout.py` | reads the parquet's own `Prev_LapTime`; lag features via keyed group shift | yes (parquet is N04's output) | eval only |
| `scripts/measure_mc_tables.py:849`, `bench_pace_baselines.py:168` | measurement-side lags | eval instruments, not inference | no |

### Sweep C — census of every `pace_delta_s` construction

| surface | feeds | axis |
|---|---|---|
| CLI `run_simulation_cli.py:1357` | prompt via RaceState | rival-relative (car ahead, same lap) — FIXED, read |
| arcade `strategy.py:673` | prompt via RaceState | rival-relative — FIXED, read |
| backend simulator `:403` | prompt | 0.0 unknown — FIXED |
| backend mcp_tools `:593` | prompt | 0.0 unknown — FIXED |
| backend `/recommend` `strategy.py:1342` | prompt | client-supplied; webapp sends 0 (`strategy.ts:356`) |
| `race_state_builder._targeting_against_rival:46` | prompt | rival-relative (user-selected rival, #431) — the "shared helper recomputes" claim in the new comments is TRUE (verified in source) |
| N27 producer `race_situation_agent.py:908` | N27 features | rival-relative (two drivers, same lap) — the contract source |
| eval `decision_modes.py:337` | deterministic replay | constant 0.0 — diverges from what the CLI now feeds, but `race_state.pace_delta_s` is consumed ONLY at `strategy_orchestrator.py:1707` (prompt), so the no-llm scorecard numbers are unaffected. LOW note, no action forced |
| `debug_agent.py:305`, `prompt_ab/gen_inputs.py:63`, test fixtures | — | 0.0 neutral |
| `RivalState.pace_delta_s` (`position_projection.py:159`) | MC projection | opposite sign convention (rival minus us) BUT its only producer `_rival_states_from_lap_state:821` never sets it (defaults 0.0), so no cross-axis feed exists today |

**No fifth surface feeding a self-delta (or any wrong-axis value) into the field was found.** The four fixed surfaces are the population.

## What I tried to break and could NOT

- **The three horizon unifications** — each goes red on exactly the test named for its horizon (executed, F-04). The PR's mutation claim is accurate as stated.
- **A fourth, adapter-level unification** (`_rival_states_from_lap_state` forcing `is_pitting` from `stop_pending`) — bypasses all three horizon tests but is caught by `test_mc_is_a_real_decision` and the projection golden (2 failed / 240 passed, executed).
- **The SOFT prompt-bound guard** — a reflow that drops the `<=` phrasing is caught by `test_the_soft_bound_is_actually_present_in_both_prompts` (asserted by reading; the guard asserts set membership of the rendered tuple, which the reflow removes).
- **The 341-stint measurement** — reproduced to the decimal with the shipped instrument, including the strict-inequality reading (F-06).
- **The backend anchor vs N04** — 904/904 exact agreement on real Lusail 2025 data; no anchor worse than the 90.0 default anywhere in the race (out-laps/first-flying-laps return None, which downstream maps to the same 90.0, never to something worse); in-laps and SC laps anchored correctly; both frame shapes handled; Stint-less and quality-flag-less frames degrade honestly (all executed, F-05).
- **`Series.get` NaN traps in the new code** — `row.get("LapNumber")` / `row.get("Stint")` both pass through `pd.isna` before use (read; the D battery exercised the NaN-stint path implicitly via frames without the column).
- **The "shared helper recomputes rival-relative" comment** — verified true against `race_state_builder.py:81-87` (the one mechanism-claim in the new comments that holds).

## Verdict

| severity | count | items |
|---|---|---|
| HIGH | 1 | F-01 (dict-path orchestrator: None passthrough crashes + 92.0/90.0 twin default) |
| MEDIUM | 5 | F-02 (min-stint + rail bounds + cliff literals still restated), F-03 (MEDIUM capacity drifts under a green #741 test, executed), F-05 (out-lap claim refuted 0/42 + zero tests on the wave's riskiest function), F-07 (0.522 vs config-loaded threshold), F-08 (~13 vs ~9 cross-prompt contradiction from a retired constant) |
| LOW | 3 | adapter-seam nuance (F-04), wrong-frame comment (`strategy.py:860`), eval `decision_modes` 0.0 divergence note + sc `>=`/`>` boundary nit |

**Mutation hygiene:** every mutant was applied from a `cp` backup of the pristine file, the mutant pattern was counted (`== 1`) in the pristine text before substitution, and every restore was verified byte-identical with `filecmp.cmp(..., shallow=False)` — reported `True` in all five runs. `git status` at close shows no tracked file modified in parent or submodule; the only new file is this report. No `git checkout --` was used at any point.
