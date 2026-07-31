# FABLE Gate G2 — tyre wear term (PR #760, commit `9fe8887`, branch `dev`)

Adversarial gate over the consumption half of #744: `TireOutput.deg_cost_s` charged in both
Monte Carlo scorers, replacing `FRESH_GAIN = 0.25`. Written incrementally as findings are
confirmed; every finding carries file:line, a failing scenario, and executed evidence.

Repo state at audit start: branch `dev` @ `db21e0c` (merge of #760), tree clean except two
untracked paths (`src/telemetry` dirty submodule pointer, one notebook PNG). All file
mutations in this audit are backed up with `cp` first and restored from the backup (verified
by diff), never with `git checkout --`.

## Checklist of claims under attack

- [x] **A. Monotonicity** — training numbers reproduce exactly (A-1); strict band
      monotonicity FAILS in-sample and the printed report dropped that column (A-2); on
      2025 the reference IMPROVES: 83.7% / +0.603, monotonic (A-3).
- [x] **B. FRESH_GAIN not double-counted** — verified in every branch of both scorers,
      incl. OVERCUT under SC and the no-stop branch (B-1). Docstring/test-name overlap
      claim about the cliff term is false in one corner (B-2).
- [x] **C. Lap counts per candidate** — all verified, fallback byte-equivalent to the old
      formulas (C-1) — but the legacy OVERCUT count is UNGUARDED (H-2).
- [x] **D. The bound** — p1/p99 re-derived exactly; 2025 clip rates 0.27% / 1.56%,
      distortion direction is toward STAY_OUT (D-1).
- [x] **E. Goldens unmoved** — verified fallback-by-construction; the blind spot is real
      and demonstrated with two executed surviving mutants (E-1, H-1, H-2).
- [x] **F. Sentinel rule** — holds end to end, incl. no-lap-at-life<=3, partial parses,
      the NaN edge, and the scorer-side getattr (F-1); latent config split-brain (F-2).
- [x] **G. The #744 correction** — independently traced and CONFIRMED (G-1); the
      corrected-away claim still ships inside the new test file's comments (G-2).
- [x] **H. Mutation testing** — baseline + M1/M2/M3 reproduce (H-0); the fourth reported
      mutant does NOT — it survives the suite (H-1); new survivor found on the legacy
      OVERCUT count (H-2); producer-semantics survivor M18 below (H-3).

## Findings

### A-1 [VERIFIED] The training-season numbers reproduce exactly

`uv`-venv run of the unmodified instrument:

```
$ .venv/Scripts/python.exe scripts/measure_tyre_reference.py --out .../tyre_ref_train.json
laps scored: 31624  (tyre life > 3, seasons (2023, 2024))
harness self-check: corr(pred, target) = 0.977

reference         non-neg  spearman  pearson      p1     p99
pooled              66.3%     0.188   -0.055  -32.63    4.12
stint_first         71.2%     0.269    0.084   -2.44    3.80
stint_le3           73.0%     0.295    0.090   -2.34    3.71
stint_live          73.9%     0.308    0.095   -2.33    3.67
none                65.3%     0.191   -0.054  -32.61    4.06
```

73.9% / +0.308 / p1 −2.33 / p99 +3.67 all match the PR and the constants shipped in
`src/agents/tire_agent.py:246-247` (`deg_cost_floor_s = -2.33`, `deg_cost_ceiling_s = 3.67`).
Claim D's derivation half is therefore also confirmed. (2025 measurement pending, below.)

### A-2 MEDIUM — the shipped reference FAILS the strict monotonicity criterion the reverted design was killed on, and this PR removed that column from the printed report

- `scripts/measure_tyre_reference.py:190` still computes `monotonic_bands` =
  `by_band.is_monotonic_increasing`, and for the shipped candidate it is **False** on the
  training seasons:

```
stint_live | monotonic_bands: False | median: 0.472
  bands: {'(3, 5]': 0.069, '(5, 10]': 0.276, '(10, 15]': 0.563, '(15, 20]': 0.807,
          '(20, 25]': 1.032, '(25, 100]': 0.906}
```

  The (25, 100] band drops 1.032 → 0.906 — the same shape (`64.8% non-negative,
  non-monotonic by band`) that `DESIGN_S3_option_b.md` records as the reason the earlier
  same-stint reference was BUILT and REVERTED.
- In the same commit, `report()` (`scripts/measure_tyre_reference.py:209-215`) was changed to
  print `p1`/`p99` **in place of** the `monotone` column (see `git show 9fe8887 --
  scripts/measure_tyre_reference.py`), so the criterion that would have printed `False` next
  to the shipped candidate no longer appears in the copy-pasteable summary. The JSON payload
  still carries it, and `MEASURE_744a_tyre_reference.md` does argue the (25,100] dip is a
  population change that hits every candidate (verified: `none` is also non-monotonic,
  1.032→0.906 vs 0.992→0.785). The mitigation is real, but the acceptance criterion #744 set
  ("monotonic by tyre-life band") is strictly FAILED by what shipped, and the one place that
  used to say so out loud was edited to say something else in the shipping commit.
- Failing scenario: the next person re-runs the instrument after a TCN retrain, reads the
  printed table, sees no monotone column, and concludes the criterion is met.
- Weight: this is a formal acceptance box on #744 — *"non-negative on the large majority of
  real laps and **monotonic** by tyre-life band"* — so the in-sample half of that box is
  strictly unmet as shipped. Mitigating fact from this gate's own measurement: the
  criterion PASSES on 2025 (A-3), which is the season the layer serves, and the in-sample
  failure is a single population-change band shared by every candidate. The right close is
  to say that in the issue, not to stop printing the column.

### A-3 [VERIFIED — the attack FAILED] On 2025, the season the system infers on, the reference gets BETTER, not worse

Ran the PR's own instrument with only `TRAINING_YEARS = (2025,)` overridden (scratchpad
`measure_2025.py`, imports `scripts/measure_tyre_reference.py` unmodified):

```
laps scored: 15005  (tyre life > 3, seasons (2025,))
self-check (informational, threshold bypassed): corr(pred, target) = 0.804

reference         non-neg  spearman  pearson      p1     p99
stint_live          83.7%     0.603    0.604   -1.11    4.06
none                77.3%     0.504    0.493   -2.39    3.77
```

- Non-negative 73.9% → **83.7%**, Spearman +0.308 → **+0.603**, and the band medians are
  **strictly monotonic on 2025** (`(3,5] 0.049 → (25,100] 1.629`) — the criterion A-2 shows
  failing in-sample passes out of sample. The degradation-on-2025 attack found the opposite
  of what it was hunting.
- One number worth keeping: the shipped `self_check` threshold (0.90) FAILS on 2025 at
  **corr(pred, target) = 0.804** vs 0.977 in-sample. Same transform, same code path, so this
  is the TCN's generalisation gap, not harness drift — but anyone re-pointing the instrument
  at 2025 will hit a `RuntimeError` whose message blames the tensor transform. LOW: the
  error message names the wrong mechanism for the out-of-sample case.

### G-1 [VERIFIED — the PR's correction is RIGHT] `strategy.py:882`'s empty rival list never reaches the Monte Carlo

Traced in the submodule at `src/telemetry` @ `858418f`, executed greps, not read prose:

- `_build_lap_state_from_row` (defines `"rivals": []` at
  `backend/api/v1/endpoints/strategy.py:882`) has **exactly one caller**:
  `strategy.py:695` inside `/pace-range`, and that lap_state feeds
  `run_pace_agent_from_state(lap_state)` at `strategy.py:698` — the pace agent, never the
  MC. `grep -rn "_build_lap_state_from_row" backend/` returns only the def and line 695.
- The backend's MC paths are: (a) `backend/services/simulation/simulator.py:860`
  `for lap_state in engine.replay()` → `run_lap(race_state, laps_df, lap_state, ...)` at
  `simulator.py:438` (no-llm) and `simulator.py:896` (rich) — the lap_state is the
  RaceReplayEngine's, which emits the RSM rivals, so this path is projection-capable; and
  (b) `/recommend` (`strategy.py:1309`) + `backend/mcp_tools.py:607` →
  `run_strategy_orchestrator_from_state(..., lap_state=request.lap_state)` →
  `_run_mc_simulation(rivals=(lap_state or {}).get("rivals"), ...)` at
  `strategy_orchestrator.py:2464`. `RecommendRequest.lap_state` is required
  (`strategy.py:107`) and documented as "the raw lap_state dict produced by
  RaceStateManager", so which scorer runs depends on whether the client's payload carries
  usable rivals — not on line 882.
- The two builders that DO hardcode `"rivals": []` are exactly the two the PR names:
  `src/strategy/inference/engine.py:449` (`_build_default_lap_state`) and
  `src/agents/strategy_orchestrator.py:2421`, both only on the `lap_state is None` path.

No HIGH here: the correction survives an independent trace.

### G-2 MEDIUM — the corrected-away claim still ships, verbatim, inside the new test file

The commit message and PR body correct #744's "three builders including the backend
endpoint" claim. The test file that landed **in the same commit** still asserts the wrong
version in two places:

- `tests/mc/test_tyre_wear_term.py:195-197` (section comment): *"Three shipping builders
  hardcode `"rivals": []`, which routes to it: `engine.py`, `strategy_orchestrator.py`, and
  the backend's own `api/v1/endpoints/strategy.py`."*
- `tests/mc/test_tyre_wear_term.py:313-316` (`test_the_legacy_branch_receives_it`
  docstring): *"three shipping builders do by hardcoding an empty rival list — including
  the backend's own endpoint. This is the branch that runs in production behind the API."*

Both restate the mechanism the same commit's own message proves wrong (G-1). This is the
repo's dominant defect class — one copy corrected, its twin not — inside the artefact that
outlives the PR body. The propagation risk is not hypothetical: issue #744 carries the same
claim under the label **"Verified, not inherited"**, which is exactly how a wrong mechanism
acquires authority. Failing scenario: the next contributor reads the test file (the
natural first stop), "learns" that `strategy.py:882` routes to the legacy scorer, and
builds the next fix on that mechanism. Fix is a comment edit only; no behaviour change.

### D-1 [VERIFIED] The bound reproduces; the clip is rare and its bias direction is toward STAY_OUT

- p1/p99 re-derived on training: **−2.33 / +3.67** exactly (A-1 table) — matches
  `tire_agent.py:246-247`.
- Clip frequency on 2025 (the served season), measured over the same 15,005 laps:
  **0.27%** below the floor, **1.56%** above the ceiling, **1.83%** total. The 2025 p99 is
  **4.06**, above the shipped ceiling, so the upper clip fires ~1.6x more often than the 1%
  it was calibrated to.
- Direction of the distortion: a genuinely worn set whose true reading exceeds 3.67 is
  UNDER-charged, which biases those laps toward STAY_OUT — the same direction as the
  layer's known 46.1% decline failure mode. At window 5 the maximum understatement at the
  2025 p99 is 5 × (4.06 − 3.67) ≈ 2.0 s ≈ 1.3 positions on the legacy scale. Affects at
  most ~1.6% of laps, which are disproportionately the ones where the call matters. LOW,
  by frequency; worth re-measuring the bound when 2026 data arrives.

### B-1 [VERIFIED] FRESH_GAIN is never double-counted — closed by construction, every branch checked

- The only two places the fresh credit is ever applied are the two exclusive `if/else`
  helpers: `strategy_orchestrator.py:684-686` (`_tyre_term`) and
  `position_projection.py:535-537` (`_tyre_cost_s`). `grep -n "FRESH_GAIN\|fresh_gain_s"`
  over both scorers shows every other occurrence is the constant's definition (`:631`), a
  comment (`:750`), or the pass-through into `ProjectionConfig` (`:1215`) that only the
  fallback arm of `_tyre_cost_s` reads.
- Every candidate branch routes through one helper call: legacy STAY_OUT `:737`,
  PIT_NOW `:741`, UNDERCUT `:746`, OVERCUT-under-SC `:764`, OVERCUT-green `:770-774`;
  projection stop-plans `position_projection.py:593`, no-stop `:601`. With a reading,
  the fresh credit is absent (not offset); with `None`, only the credit exists.
- OVERCUT under SC (`:764`): `_tyre_term(deg, 0, window)` — with a reading returns 0.0
  (zero old laps), fallback returns `FRESH_GAIN * window`; identical to PIT_NOW's branch,
  preserving the documented SC indifference between them. No branch applies both prices.
- The no-stop projection branch: fallback arm yields `-0.0 * fresh_gain = 0`, reproducing
  the pre-#744b "a plan that does not stop gains nothing fresh" exactly.

### C-1 [VERIFIED] The lap counts match what each branch models, and the fallback is byte-equivalent to the old formulas

Verified against the pre-change formulas in `git show 9fe8887`:

| branch | old laps | fresh laps | fallback reproduces old code |
|---|---|---|---|
| legacy STAY_OUT `:737` | `window` | 0 | `-cliff` (tyre term 0) ✓ |
| legacy PIT_NOW `:741` / UNDERCUT `:746` | 0 | `window` | `+FRESH_GAIN*window` ✓ |
| legacy OVERCUT SC `:764` | 0 | `window` | `+FRESH_GAIN*window` ✓ |
| legacy OVERCUT green `:770` | `window // 2` | `window // 2` | `+FRESH_GAIN*(window//2)` ✓ |
| projection stop plan `:593` | `laps_before_stop` | `laps_after_stop` | `-laps_after_stop*fresh_gain` ✓ |
| projection no-stop `:601` | `racing` | 0 | no term ✓ |

Observation, not a finding: legacy OVERCUT green prices 2 old + 2 fresh of a 5-lap window
(one lap unpriced, the stop lap). That asymmetry predates this PR — the old code's
`FRESH_GAIN * (window // 2)` had the same shape — so the relative differential against
STAY_OUT (deg × 3 laps) is mildly conservative, consistently with the previous behaviour.

### E-1 [VERIFIED] The goldens are unmoved because they run the fallback — and the blind spot behind that is real, with two executed survivors to prove it

- `tests/mc/test_strategy_goldens.py:52-60`: the canned `TireOutput` never sets
  `deg_cost_s`, and the dataclass default (`tire_agent.py:497`) is `None` → the golden
  exercises exactly the pre-#744b arithmetic (`_tyre_term(None, ...)`), and it passed
  unchanged in the baseline run. The "unmoved goldens" claim is true and is BY
  CONSTRUCTION, not evidence about the new path.
- No golden or frozen fixture pins the measured path numerically anywhere:
  `grep -rn deg_cost_s tests/` hits only the two new files. The measured path's coverage
  is: 3 hermetic count tests + 1 sign test (projection), 5 leaf tests on `_tyre_term`,
  2 wiring inequality tests that read **only `STAY_OUT["E"]`**, and 1 real-lap sign test.
- What can land unnoticed, demonstrated by execution rather than argued: H-1 (the
  neutralised-config side of the projection) and H-2 below (the legacy OVERCUT measured
  count) both mutate shipped behaviour to nonsense and stay green across the full suite.

### E-2 LOW — `test_neither_branch_moves_when_there_is_no_reading` cannot fail for its stated reason

`tests/mc/test_tyre_wear_term.py:341-347` asserts `self._score([], None) ==
self._score([], None)` (and the rivals twin) — the same call twice. That proves the MC is
deterministic, which `test_mc_is_deterministic_across_calls` already pins. The docstring
claims it proves "the fallback keeps every pre-#744b caller on exactly its old numbers";
nothing in the assertion compares against a pre-#744b value (the goldens do that). A
future change that altered the fallback's numbers would leave this test green. Rename or
compare against the frozen golden.

### F-1 [VERIFIED] The sentinel rule holds end to end

Traced every producer edge, all executed via the baseline suite plus code inspection:

- No lap at tyre life <= 3 (replay starts mid-stint): `_get_driver_stint` returns `None`
  (`tire_agent.py:1053`), `_fresh_reference` returns `None` (`:936-938`), the tool omits
  the line entirely (`:1169-1171`), the parser writes no `fresh_ref` key
  (`tire_parsing.py` pattern table), `_referenced_wear` returns `None` (`:408-409`).
  Pinned by `test_no_reference_line_writes_no_reference_key` and
  `test_a_missing_half_is_none_and_never_zero` (both green in baseline).
- Only one tool line parses: both `cum_deg` and `fresh_ref` ride the SAME tool message,
  and `_referenced_wear` requires both keys — single-key dicts return `None` (executed:
  same tests).
- The legitimate 0.0: at tyre life <= 3 the reference prefix IS the prediction prefix
  (`_get_driver_stint(driver, 3)` capped by `current_lap`), so wear is exactly 0.0 and is
  charged as zero, distinct from `None` — pinned at the consuming end by
  `test_a_fresh_set_costs_nothing_and_is_not_the_same_as_no_reading`.
- A NaN reference cannot leak: `f'{reference:.3f}'` prints `nan`, which
  `r'Fresh reference:\s*(-?[\d.]+)'` cannot match, so the key is simply absent → `None`.
- The scorer end: `getattr(tire_out, "deg_cost_s", None)` (`strategy_orchestrator.py:1390`)
  → stub TireOutputs degrade to the fallback, never to 0.0.

### F-2 LOW — the bounds and the reference tyre life read from two different configs

`_referenced_wear` reads the module singleton (`CFG.deg_cost_floor_s`,
`tire_agent.py:412`) while `_fresh_reference` reads the instance (`self.cfg.
fresh_reference_tyre_life`, `:936`). `TireAgent.__init__(cfg=CFG)` defaults to the same
object (`:854`), and every shipping construction uses the default (`TireAgent()` at
`:1604`), so today the two agree. A future caller passing a custom `TireAgentConfig`
would get its reference tyre life honoured and its bounds silently ignored — the split-
brain config pattern this repo has been bitten by before. One-line fix when touched next.

### B-2 LOW — "the cliff term and the wear term do not overlap" is false for a set already past the cliff, and the test pinning it asserts the opposite of its name

- `position_projection.py:199-205` (ProjectionConfig docstring): *"cliff_loss_s: ... Charged
  only on laps run PAST the cliff, so it does not overlap deg_cost_s, which is charged on
  every old-set lap."* The two clauses contradict each other: a lap past the cliff IS an
  old-set lap, so on those laps BOTH are charged. The decomposition is coherent when the
  cliff lies in the future (deg = current snapshot, cliff = future worsening), but when the
  set is ALREADY past the cliff at decision time the TCN's current reading already contains
  the post-cliff pace, and `CLIFF_LOSS` charges 0.80 s/lap again for the same physical
  seconds.
- The test named for the claim pins the additive behaviour, not the absence of overlap:
  `test_the_cliff_term_and_the_wear_term_do_not_overlap` asserts
  `past_cliff == 5.0*0.4 + 5.0*0.8` — both terms on the same five laps.
- Failing scenario: cliff_i ~ 0 (N26 says the cliff is behind us), reading 1.5 s/lap
  (already cliff-level pace). Legacy STAY_OUT charged 5×1.5 + 5×0.8 = 11.5 s where the
  physical cost is ~7.5 s. Direction: overcharges STAY_OUT exactly when pitting is already
  right, so the practical harm is small — the finding is the docstring/test naming a wrong
  mechanism, this repo's recorded worst kind of comment.

### H-0 [VERIFIED] Baseline and three of the four reported mutants reproduce

- Baseline: `pytest tests/mc tests/agents -q` → **232 passed in 427.00s**, matching the PR.
- M1 drop the projection kwarg (`strategy_orchestrator.py:1219` removed) → **1 failed**
  (`test_the_projection_branch_receives_it`), matches "1 red".
- M2 drop the legacy kwarg (call-site form, `:1457`) → **1 failed**
  (`test_the_legacy_branch_receives_it`). The PR says 2 red; the difference is consistent
  with the PR mutating the *signature* rather than the call site (a direct-call test then
  TypeErrors too). Caught either way — no finding.
- M3 gut `_tyre_cost_s` (always return the fallback) → **6 failed**, matches "6 red".
- Each mutant applied on a `cp` backup and restored from it; every restore verified with
  `diff` against the backup ("Files are identical").

### H-1 HIGH — the PR's fourth mutation claim does NOT reproduce: the green-config-only mutant survives the entire suite

The PR table says *"set the value on the green config but not the neutralised one → 1 red"*.
Re-run, it is 0 red anywhere.

- Mutant applied at `src/agents/strategy_orchestrator.py:1230-1231` (backed up via `cp`,
  restored from the backup, restore verified by diff):

```python
green_config = _config(racing_when_racing, clean_air_s)
neutralised_config = _replace_m4(_config(racing_when_neutralised, 0.0), deg_cost_s=None)  # MUTANT M4
```

- Executed evidence:

```
$ .venv/Scripts/python.exe -m pytest tests/mc/test_tyre_wear_term.py -q
15 passed in 13.07s
$ .venv/Scripts/python.exe -m pytest tests/mc -q
159 passed in 387.49s (0:06:27)
```

  `tests/agents` never calls the Monte Carlo (verified: no `deg_cost_s` reference outside the
  two new test files, and the agents tests exercise the producer side only), so the mutant
  survives the full 232.
- Why no test can see it: catching this mutant needs a projection-branch scenario with a
  reading AND an assertion sensitive to the **neutralised** draws. The wiring test's fixture
  (rivals at −2.0 / +3.5 s, `sc_prob_3lap = 0.10`) charges deg on ~10% of draws under a
  2.61-lap neutralised window: 0.4 × 2.61 ≈ 1.04 s crosses NO gap in that geometry, so
  `STAY_OUT["E"]` is identical with and without the neutralised-side reading — the assertion
  passes for a reason unrelated to what the mutant broke. This is the boundary-assertion
  defect class the project's own memory warns about.
- Consequence: the guard for "every Safety Car lap uses the measured wear" is the comment at
  `strategy_orchestrator.py:1216-1218` and nothing else — precisely the defect class (a
  mechanism protected by a comment) that gate G1 found in #755 and that this PR says it
  closed with mutation 4. The shipped CODE is correct (both configs receive the value, read
  at `:1219`); what is broken is the claimed coverage, and the claim is load-bearing because
  the commit message cites the four mutants as the evidence the wiring is pinned.
- Fix direction: score a fully-neutralised scenario (`sc_prob_3lap = 1.0`) with rivals close
  enough that ~1 s of neutralised-window wear crosses a gap, and assert the reading moves
  the outcome.

### H-2 HIGH — a survivor the PR did not try: zero the legacy OVERCUT's old-lap count and the whole suite stays green

The mutant claim C warns about ("a wrong count makes the term a constant offset that cancels
in the argmax and looks connected while doing nothing") exists, on the exact branch the
in-code comment calls the easiest to get wrong.

- Mutant at `src/agents/strategy_orchestrator.py:772`:

```python
+ _tyre_term(deg_cost_s, 0, window // 2)  # MUTANT M5: old_laps zeroed
```

  With a reading, OVERCUT-green is now charged NOTHING for the `window // 2` laps it runs on
  the worn set — its measured-path score is wrong on every green-flag draw. The fallback arm
  is untouched (`fresh_laps` unchanged → `FRESH_GAIN * (window // 2)`), so every
  `deg_cost_s=None` caller — including all four goldens — is byte-identical.
- Executed evidence:

```
$ .venv/Scripts/python.exe -m pytest tests/mc -q     # with M5 applied
159 passed in 425.16s (0:07:05)
```

  (`tests/agents` cannot catch it — no MC call sites there — so this survives the full 232.)
  Restored from the `cp` backup afterwards; restore verified by diff and `git diff --stat`
  (empty).
- Why the suite is blind: `test_the_overcut_green_branch_splits_the_window` tests
  `_tyre_term(0.4, old_laps=half, fresh_laps=half)` — the FUNCTION with the right arguments,
  not that `simulate_lap_window` passes those arguments. The wiring tests read only
  `STAY_OUT["E"]`. Nothing anywhere evaluates OVERCUT (or PIT_NOW/UNDERCUT) with a reading
  present. The projection scorer's equivalent count IS pinned
  (`test_an_overcut_pays_for_the_laps_it_waits_and_no_more`) — the legacy twin is not: one
  copy tested, its twin not.
- Failing scenario in production: a future edit "simplifies" the OVERCUT branch (or a merge
  resolves it wrong), OVERCUT stops paying for its old-set laps, its score inflates by
  `deg x window//2` on worn-tyre laps, and it starts re-winning green-flag argmaxes on
  exactly the high-wear laps the epic cares about — with 232 tests green.
- Fix direction: one leaf test on `simulate_lap_window` itself with a reading, asserting the
  four candidates' relative deltas (e.g. `STAY_OUT - OVERCUT == -deg * (window - window//2)`
  net of cliff terms), or a frozen golden with `deg_cost_s` set.

### H-3 MEDIUM — a second survivor: neuter the reference semantics and the whole channel silently reads 0.0, with 232 green

The producer's core semantic — the reference is the model on the stint's **early** laps —
has no test. Mutate it away and the feature becomes a no-op that is WORSE than the fallback.

- Mutant at `src/agents/tire_agent.py:247`:

```python
fresh_reference_tyre_life: int = 999  # MUTANT M18: reference is the whole stint
```

  `_get_driver_stint(driver, 999)` returns the same prefix as the main prediction
  (`TyreLife <= 999` intersected with `LapNumber <= current_lap` and the stint), so the
  reference tensor IS the prediction tensor and `deg_cost_s` computes **exactly 0.0 on
  every lap** — not `None`. Both scorers then charge zero for the old set AND grant no
  fresh credit: the channel actively disables the tyre term instead of degrading to
  `FRESH_GAIN`.
- Executed evidence:

```
$ .venv/Scripts/python.exe -m pytest tests/agents tests/mc -q     # with M18 applied
232 passed in 396.00s (0:06:36)
```

  Restored from the `cp` backup; restore verified by diff and `git diff --stat` (empty).
- Why every test passes, including the one real-lap test: the Lusail assertion is
  `0.0 <= tire_out.deg_cost_s < 1.0` (`tests/agents/test_tire_cumulative_deg.py:343`) —
  boundary-INCLUSIVE at exactly the degenerate value, so it cannot distinguish "a
  nearly-fresh set costs +0.015" from "the reference was neutered and wear is identically
  zero". Its other assertion, `cumulative_deg_s < deg_cost_s` (−0.498 < 0.0), also passes.
  This is the assertion-passes-near-a-boundary defect class, at the exact boundary the
  sentinel rule (claim F) spends so much care distinguishing from `None`.
- The class this represents is not a config typo: any regression in `_get_driver_stint`'s
  tyre-life filter, or in how `_fresh_reference` slices the stint, collapses to the same
  degenerate wear ≈ 0 — and the suite proves it would not notice. `_fresh_reference` has
  no direct test at all.
- MEDIUM rather than HIGH only because reaching it requires a change in the producer file
  rather than the scorers' wiring; the blast radius when it happens is total (the epic's
  "largest missing term" silently reads zero on every lap of every race).
- Fix direction: a test on the Lusail fixture asserting `deg_cost_s > 0.0`
  (strict — a five-lap-old set on a real stint measures +0.015), plus a direct
  `_fresh_reference` test asserting the reference differs from the current prediction when
  tyre life > the reference band.

## What I tried to break and could NOT

Stated so silence reads as evidence, per the audit doctrine:

1. **The 2025 degradation attack (claim A).** Ran the PR's own instrument on the held-out
   season expecting the reverted design's failure shape to reappear. It did not: 83.7%
   non-negative, Spearman +0.603, strictly monotonic bands — better than in-sample on all
   three criteria. The reference genuinely works where the system infers.
2. **The double-count (claim B).** Tried to construct a branch where a measured reading and
   `FRESH_GAIN` are both applied. There is none: both helpers are exclusive `if/else`, and
   an exhaustive grep shows no other application site of either price in either scorer.
3. **Fallback drift (claim C/E).** Tried to find a `deg_cost_s=None` path whose arithmetic
   differs from pre-#744b. Every branch's fallback reproduces the old formula exactly
   (C-1 table), and the four frozen goldens pass unmoved (baseline 232 green).
4. **The bound derivation (claim D).** p1/p99 re-derived to the shipped digits (−2.33 /
   +3.67) from 31,624 laps.
5. **The sentinel (claim F).** Tried to make 0.0 appear where `None` was meant: missing
   halves, absent early laps, single-line parses, and the NaN-format edge all produce
   `None`; the floor test pins that negative readings survive (so a clamp-to-zero mutant
   would go red); the scorer-side `getattr` degrades stubs to `None`. The producer→parser→
   output→scorer chain holds at every seam I could attack.
6. **The #744 correction (claim G).** Tried to refute it with an independent trace of the
   submodule; the trace confirmed it instead (G-1).
7. **Mutants the suite DOES catch (claim H).** M1 (projection kwarg) 1 red; M2 (legacy
   kwarg) 1 red; M3 (gut `_tyre_cost_s`) 6 red; M6 (sever the `getattr` read at
   `strategy_orchestrator.py:1390` → `None`) 2 red — so the TireOutput→MC read, both
   kwargs, and the projection cost function are genuinely pinned.
8. **The instrument's self-check.** It is a real guard: it fired (correctly, if with a
   misleading message) the moment the season constant changed the data distribution.

## Numbered fix list, ordered by value and risk

1. **(H-1, HIGH)** Add a neutralised-side wiring test: fully-neutralised scenario, rivals
   within ~1 s, assert the reading moves the outcome. This is the only guard that can make
   the "both configs" comment at `strategy_orchestrator.py:1216-1218` enforceable.
2. **(H-2, HIGH)** Pin the legacy measured path per candidate: one test calling
   `simulate_lap_window` with a reading and asserting the relative deltas of all four
   candidates, or a second frozen golden with `deg_cost_s` set.
3. **(H-3, MEDIUM)** Make the Lusail assertion strict (`> 0.0`) and add a direct
   `_fresh_reference` test (reference != current prediction when tyre life exceeds the
   band).
4. **(G-2, MEDIUM)** Fix the two stale comments in `tests/mc/test_tyre_wear_term.py:195,
   313-316` to the corrected two-builder mechanism; consider a correcting comment on #744
   itself, which labels the wrong claim "Verified, not inherited".
5. **(A-2, MEDIUM)** Either restore the `monotone` column to the printed report or record
   in #744 that the in-sample band criterion is failed-by-population-change and passed on
   2025 — the acceptance box should not close silently.
6. **(E-2/B-2/F-2, LOW)** Rename or strengthen `test_neither_branch_moves_when_there_is_no_
   reading`; reword the "do not overlap" docstring/test name; unify `_referenced_wear` on
   `self.cfg` when next touching the file. A-3's note: point the self-check error message
   at the out-of-sample case too.

## Not run

`f1-eval decision-modes` against the 46.1% decline baseline (~21 min). The PR marks it out
of scope and assigns it to the epic's measurement step; it is the one acceptance box of
#744 neither the PR nor this gate has executed. Run it before promoting the epic's claim
that the term moved the decline rate — nothing in this gate measures that.

## Summary

- **HIGH 2** — H-1 (the reported green-config mutation result does not reproduce; the
  neutralised side of the projection channel is unguarded), H-2 (legacy OVERCUT's
  measured-path lap count can be zeroed with 232 green).
- **MEDIUM 3** — A-2 (in-sample monotonicity criterion failed and its report column
  removed), G-2 (corrected-away claim ships in the test file), H-3 (reference semantics
  can be neutered to an all-zero channel with 232 green).
- **LOW 5** — B-2 (overlap docstring/test name), D-1's clip asymmetry, E-2 (x==x test),
  F-2 (config split-brain), A-3's misleading self-check message on out-of-sample data.
- **Verified intact** — A-1/A-3 (the measurement, in- and out-of-sample), B-1 (no double
  count), C-1 (lap counts + byte-identical fallback), D-1 (bound derivation), E-1 (goldens
  = fallback by construction), F-1 (sentinel end to end), G-1 (the PR's own correction),
  H-0 (three of four reported mutants).

The single most important line: **the four-mutant table the commit offers as proof of the
wiring is one-quarter wrong and three survivors deep** — the code that shipped is, on all
executed evidence, correct; the safety net around its most SC-critical and most
wear-critical branches is thinner than the PR says it is.
