# FABLE G1 — Adversarial gate over "model input" findings (2026-07-29)

**Mandate:** verify or refute four claims about model inputs (N06 anchor constant, N12 gap
scope, N14 `n_drivers_delta` domain, N15 pit-duration fallback), measured against what the
arcade / telemetry surfaces ACTUALLY execute and display, offline (`no-llm`), read-only.

**Primary contradiction to reconcile:** the owner reports the model values on the arcade
and telemetry surfaces "look close to the real ones", which contradicts claim 1 as stated.

Report is appended incrementally as evidence is executed. Verdicts land at the top of each
section once earned.

## Checklist

- [x] C1 — N06 `prev_lap_time or 90.0`: trace the REAL arcade path (`engine.run_lap`), find what the UI displays, re-measure on that path
- [x] C2 — N12 overtake gap scope (21.8 s vs 2.5 s ceiling, 54.2% of calls) + does it move a user-visible number
- [x] C3 — N14 `n_drivers_delta` up to +14 vs trained `<= 0`
- [x] C4 — N15 `team_year_median` flat 2.8 s on 100% of served calls
- [x] Why earlier hunts missed each CONFIRMED item (mechanism, not guess)
- [x] What I tried to break and could NOT

---

## C1 — N06 `Prev_LapTime` anchored to 90.0 — VERDICT: CONFIRMED on the arcade/CLI/backend-SIMULATOR path; DOES NOT APPLY to the telemetry tab. The owner's observation and the claim are BOTH correct — they describe different surfaces.

### The path trace (the check the claimant asked for first)

The direct call `run_pace_agent_from_state(st)` IS the production path. Verified end to end:

1. Arcade loop: `src/arcade/strategy.py:404` `_step_once` → `:429` `run_strategy_pipeline(...)`.
2. `src/arcade/strategy_pipeline.py:23,47` is a thin delegate: `run_lap(...)` (rich profile).
3. `src/strategy/inference/engine.py:264` `_run_rich` → `_run_always_on_agents_from_state(race_state, laps_df, lap_state)`.
4. `src/agents/strategy_orchestrator.py:1834`: `pool.submit(run_pace_agent_from_state, lap_state)` — the SAME
   one-argument function, fed the SAME `lap_state` dict, unmodified (the engine's only pre-step,
   `_scope_laps_to_gp`, touches `laps_df`, which the pace agent never receives).
5. That `lap_state` comes from `RaceReplayEngine.replay()` (`src/arcade/strategy.py:372`), whose RSM is
   constructed from `data/raw/<year>/<gp>/laps.parquet` (`src/simulation/replay_engine.py:73-75`) —
   verified: `data/raw/2025/Lusail/laps.parquet` has 35 columns and **no `Prev_LapTime`**
   (it exists only in `data/processed/laps_featured_*.parquet`).
6. `race_state_manager.py:258` emits `prev_lap_time = _to_seconds(r.get("Prev_LapTime"))` — `Series.get`
   on a missing column → `None`, no fallback derivation anywhere in the class.
7. `pace_agent.py:725` `prev_lap_time = d.get('prev_lap_time') or 90.0` → 90.0 on every such lap.

So the methodological worry ("maybe run_lap builds features differently") is CLEARED: for the pace agent
there is no difference between the direct call and the arcade path. Same function, same dict.

The same raw-parquet RSM also feeds `scripts/run_simulation_cli.py` and the backend SSE simulator
(`src/telemetry/backend/services/simulation/simulator.py:50,814` constructs `RaceReplayEngine`), so the
CLI PMV and the backend's live-simulation stream carry the same anchor.

### Re-measured on the REAL loop (executed, `RaceReplayEngine.replay()` + the arcade `_lap_skip_reason`
guard + `run_pace_agent_from_state`, NOR/McLaren, plus the counterfactual the original audit never ran —
the same laps re-predicted with the featured parquet's `Prev_LapTime` patched in):

```
race    laps  prev=None  |err| served  |err| featured-prev  mean actual  mean pred  CI half
Lusail    57      57/57      7.72 s        0.30 s (n=43)       89.50       89.94     ±2.50
Monaco    78      78/78     13.82 s        0.64 s (n=67)       77.36       89.84     ±2.50
Monza     53      53/53      7.16 s        0.26 s (n=48)       83.46       90.06     ±2.52
```

- The original audit's numbers reproduce to the third decimal (7.717 vs 7.72; 13.823 vs 13.83). Its
  mechanism, mechanism percentages, and served constant are all real.
- The counterfactual is the number the audit lacked: **feeding the real previous lap recovers
  0.26-0.64 s MAE on the very same laps** — so the entire 7-14 s error is the wiring, not the model,
  and the fix is a pure data-plumbing change (no retraining).

### Reconciling the owner's evidence — he is RIGHT about what he sees, and the claim is STILL true

**Telemetry tab: the claim simply does not apply there.** Every widget the owner would use for a
model-vs-real comparison in the telemetry tab is served by
`src/telemetry/backend/api/v1/endpoints/strategy.py`, which builds `lap_state` from the FEATURED
parquet: line 442 `get_laps_df(year)` and line 487 `"prev_lap_time": _prev_lap_time_for_row(...)`
(line 828-829 reads the featured `Prev_LapTime`, with a real prior-lap fallback). That covers
`AgentTabs.tsx:283` (`StatCard "Lap time" = pace.lap_time_pred` from POST `/strategy/pace`) and the
Race Trace (`POST /pace-range`, the actual-vs-predicted chart — the exact surface built for this
comparison). The telemetry tab genuinely shows ~0.3-0.6 s-grade predictions. Owner correct.

**Arcade: affected, but the two most-watched numbers hide it.**

- The Pace card headline (`src/arcade/dashboard/agent_formatters.py:108`) is
  `"Δnext {delta_vs_prev}s ({pred}s)"` — and `delta_vs_prev = lap_time_pred − prev_lap_time =
  (90+δ) − 90 = δ`, the anchor cancels ALGEBRAICALLY. Measured mean |Δnext| on the served path:
  0.07-0.21 s — always plausible-looking, at every circuit.
- The pace chart (`pace_chart.py`) plots absolute `pred` vs `actual` with a CI band of measured
  half-width ±2.5 s. At **Lusail — the repo's canonical demo GP — mean served pred is 89.94 vs mean
  actual 89.50**: the actual line sits INSIDE the band. "Looks close" is exactly what a Lusail run
  shows, while `Prev_LapTime` is 90.0 on 57/57 of those laps.
- The bug is only visually loud where real laps sit far from 90 s: Monaco (12.5 s of daylight between
  the lines) or Monza (6.6 s). If the owner's arcade sessions were Lusail/Budapest-class circuits
  (85-92 s laps), everything he saw was consistent with a correctly-fed model.

**Answer to "which field does the UI display":** arcade displays BOTH the delta (headline — immune by
construction) and the absolute `lap_time_pred` (chart dashed line + `pred {x}s` body row +
`reasoning_tabs.py:110` — all carry the damage, visibly only when |circuit pace − 90| > the ±2.5 s band).
The telemetry tab displays `lap_time_pred` from a differently-sourced, correct path.

### Why did earlier bug hunts miss it? (concrete mechanism, not a guess)

Four reinforcing mechanisms, each verifiable in the tree:

1. **The regression tests assert the fixed path, on the featured parquet.**
   `src/telemetry/tests/test_strategy_audit_fixes.py:93-161` (the #435/#486 tests) build rows from
   `get_laps_df` = the FEATURED parquet and assert `lap_state["driver"]["prev_lap_time"] ==
   Prev_LapTime`. Green — and true — on the backend path. No test ever constructs an RSM from
   `data/raw/.../laps.parquet` and asserts `prev_lap_time is not None`, which is the only
   configuration every replay surface actually runs.
2. **The #435 fix comment claims victory in the wrong keyspace.** `pace_agent.py:709-724` says
   "RaceStateManager.get_driver_state now emits the real 'prev_lap_time' sourced from the parquet's
   Prev_LapTime column" — true only for a parquet that HAS the column. The reviewer of #435 read a
   correct-sounding sentence; nobody asked "which parquet does each RSM constructor receive?" This is
   the repo's own documented twin-defect shape: the backend producer got a real fix
   (`_prev_lap_time_for_row`), the RSM producer got a `.get` that silently degrades.
3. **The headline metric self-heals.** The number a human glances at (Δnext) cancels the anchor
   exactly, and the demo circuit (Lusail) sits at 89.5 s real pace — within noise of the 90.0
   constant. The damage concentrates in `lap_time_pred`/`delta_vs_median`, which no test compares
   against real lap times on the replay path.
4. **`_sane_lap_time` filtering in the chart (30-200 s) admits 90.0 happily** — the display layer's
   only sanity check cannot catch a wrong-but-plausible constant.

### The surgical change (wiring fix, not a modelling change)

**Where:** `src/simulation/race_state_manager.py::get_driver_state` (one producer fixes all three
affected surfaces: arcade, CLI PMV, backend simulator — they all consume RSM).

**What:** derive `prev_lap_time` when the column is absent: the RSM already holds the driver's full
lap table (`self._driver`); the previous lap's `LapTime` at `lap_number - 1` is one lookup:

- keep `_to_seconds(r.get("Prev_LapTime"))` as the preferred source (featured-parquet callers,
  and the #435 semantics, unchanged);
- when that is `None` AND `lap_number > 1`, fall back to
  `_to_seconds(self._driver.loc[LapNumber == lap_number-1, "LapTime"])` — the true previous lap from
  the same raw frame (this is NOT the #435 self-feeding bug: that bug fed the CURRENT lap's time;
  the previous ROW's time is exactly what `Prev_LapTime` encodes in N04).
- first lap of the race stays `None` → the 90.0 default remains only for genuinely absent history.

**Semantic caveat to decide at fix time (do not skip):** N04's `Prev_LapTime` is the previous lap
**within the stint** (grouped shift), so lap 1 of stint 2+ is NaN in the featured parquet, while the
raw-frame `lap_number - 1` lookup would return the in-lap's time. Either match N04 (require same
`Stint` on the previous row → None on stint boundaries, maximally faithful to training) or accept the
cross-stint value. Matching N04 is the defensible default; it also keeps the trained-feature semantics.

**What must NOT move:** `pace_agent.py:725`'s `or 90.0` (still needed for lap 1); the backend
endpoint's `_prev_lap_time_for_row` (already correct); the arcade's own `prev_lap_time` accumulator in
`strategy.py` (it feeds `RaceState.pace_delta`, a different consumer, and is already real).

**Tests that shift:** none existing (that is the point — see mechanism 1). Add the missing one: build
`RaceStateManager` from a RAW-schema frame (no `Prev_LapTime` column), assert
`get_driver_state(n)["prev_lap_time"]` equals the lap n-1 `LapTime` and is `None` only on lap 1 /
stint boundary. The goldens that WILL shift on a real run: every recorded no-llm/CLI decision log that
contains `lap_time_pred` (it moves from ~90 to circuit-real values), and `delta_vs_median` displays.
Recommendation-level goldens do NOT shift on the no-llm path: `_run_mc_simulation` assigns
`pace_s = rng.normal(pace_out.lap_time_pred, ...)` and never uses it (`strategy_orchestrator.py:1386`,
`# noqa: F841` — linter-confirmed dead). The fix therefore changes DISPLAYS (chart, card, reasoning tab,
`delta_vs_median`) and the rich-mode LLM prompt content, not the deterministic decision layer.

### Overstatement check on the original audit's framing

One number needs a caveat: "18-35x worse than the model's 0.392 s test MAE" compares the anchored
replay against the notebook's featured-rows test protocol (accurate laps, real `Prev_LapTime`). The
honest per-surface comparison is the counterfactual measured here — 7.2-13.8 s served vs 0.26-0.64 s
correctly-fed **on the same replay laps** (i.e. 12-53x on like-for-like laps). The conclusion
survives; the specific multiplier was mildly understated at Monaco and computed against a different
lap population. Not material to the verdict.

---

## C2 — N12 called on gaps up to ~22 s, 54% of no-llm calls — VERDICT: CONFIRMED as a missing guard; OVERSTATED as damage. Measured effect on any user-visible number: at most a 0.29 shift in a displayed probability, zero threat/routing/decision changes in 5 races.

### The range claims, re-verified independently

- Trained ceiling: `overtake_pairs_2023_2025.parquet` filtered to Year in {2023, 2024} = 18,277 rows
  (matches `model_config.json` `n_train` exactly), `gap_ahead_s` max = **2.500** exactly. Confirmed.
- Served (my own 5-race no-llm replica, NOR in all five races — A4 varied drivers, hence slightly
  different percentages): 173 overtake calls, **106 (61.3%) with gap > 2.5 s, max 19.76 s**. A4's
  54.2% / 21.77 s reproduces in kind; the out-of-scope majority is real.
- Mechanism verified at file:line: the scope rule exists only in `_RACE_SITUATION_SYSTEM_PROMPT`
  (advisory text); `predict_overtake_tool` (`race_situation_agent.py:1105-1165`) checks lap range and
  driver liveness, never `gap_ahead_s`; `no_llm.py::_situation_no_llm` appends the overtake call for
  ANY rival at `position - 1`, unconditionally. All three legs of the claim hold.

### The absorption measurement A4 did not run (this is what changes the severity)

Executed evidence, three independent kinds:

1. **Tree geometry.** Dumped the fitted booster: all **2,179** `gap_ahead_s` split thresholds lie in
   [0.216, **2.467**]; all 1,083 `gap_pace_product` splits lie in [-3.46, +3.55]. A LightGBM tree
   cannot distinguish gap = 21.8 from gap = 2.47 — every out-of-scope gap is scored EXACTLY as a
   widest-trained-gap call. Out-of-support escapes in a tree model are clamped to boundary
   behaviour by construction.
2. **Measured output.** Calibrated P(overtake) on the 106 out-of-scope calls: **mean 0.014,
   max 0.290**. In-scope calls: mean 0.082, max 0.718. Against the served bands
   (`medium_overtake` 0.40, `high_overtake` 0.65, `race_situation_agent.py:169-172`): **zero**
   out-of-scope calls cross either band; zero `threat_level` changes are attributable to them. The
   bands themselves are alive — in-scope probs reach 0.718, and zeroing ALL overtake probs flips
   threat on 7/275 laps, all driven by in-scope calls — so this is not an artifact of dead bands.
3. **No decision-layer consumer.** `overtake_prob` feeds threat bands, dashboards, and prompt text
   ONLY. The MC reads `sc_prob_3lap` / `vsc_active` from situation_out
   (`strategy_orchestrator.py:1352,1361`) and never `overtake_prob`; `position_projection.py:28-31`
   explicitly declines to call N12 ("feeding it the counterfactual gaps a projection invents would
   run it off its own manifold").

**User-visible impact:** the Situation card (`agent_formatters.py:206`, webapp `AgentTabs.tsx:393`)
can display up to ~29% where the prompt doctrine says "assume 0" — cosmetic, and rare (the 0.29 max
was one call). On the RICH profile (arcade/CLI default) the 54-61% call rate does not transfer: the
LLM holds the scope rule there, with unmeasured (offline) compliance. The surfaces that
deterministically over-call are `f1-sim --no-llm`, the backend no-llm branch, and the eval tier.

### Surgical change (guard fix; recommended but low urgency)

**Where:** inside `predict_overtake_tool` (`race_situation_agent.py`, right after `feat_df` is
built, ~line 1147) — the one chokepoint BOTH the no-llm null-runner and the LLM path execute.
**What:** if `feat_df['gap_ahead_s'].iloc[0] > 2.5`, return the tool's existing REFUSED shape (or
`P(overtake) = 0.000 | gap=... | out of trained scope`), mirroring `_N15_TYRE_LIFE_ENVELOPE`'s
log-then-bound pattern. `_parse_tool_outputs` already defaults `overtake_prob` to 0.0
(`race_situation_agent.py:757`), so a refusal parses safely.
**What must NOT move:** `_situation_no_llm`'s call-site logic (the tool now self-guards, which also
covers LLM non-compliance); the bands; the calibrator.
**Tests that shift:** none existing; goldens asserting exact `overtake_prob` on out-of-scope laps
would move by at most 0.29 (down to 0.0). Add one test: tool refuses/zeroes when gap > 2.5. This is
a wiring/guard fix, not a modelling change.

---

## C3 — N14 `n_drivers_delta` up to +14 vs trained `<= 0` — VERDICT: range claim CONFIRMED; the implied damage REFUTED. The fitted model contains ZERO splits on this feature — any served value changes the output by exactly 0.0.

### The range claims, re-verified independently

- Trained: `sc_labeled_2023_2025.parquet`, year in {2023, 2024} = 2,280 rows (matches A4),
  `n_drivers_delta` in [-11.0, **0.0**], positive rows: **0**. Confirmed — a positive delta is
  outside the trained support entirely.
- Served (5-race no-llm replica, 275 SC calls): delta in [-20, **+14**], **76/275 (27.6%) positive**
  — A4's 27.8% reproduces. The sparse-featured-frame mechanism (N04's accuracy gate dropping rows
  unevenly) is the correct diagnosis.

### The refutation (executed, two independent kinds)

1. **The feature is UNUSED by the fitted model.** `lgbm_sc_v1.pkl` booster dump: `n_drivers_delta`
   appears in **0 of the model's splits**. LightGBM found no gain in a feature that is 0 on almost
   every training row. A feature with no splits cannot move a prediction by any amount, for any
   input value.
2. **Measured:** re-predicting all 76 positive-delta served rows with the delta clamped to 0 (and
   `lap1_chaos` recomputed): |change in calibrated sc_prob| **mean 0.00000, max 0.00000**. Zero
   crossings of `medium_sc` 0.0432 / `high_sc` 0.0864; the N30 routing threshold (0.30,
   `strategy_orchestrator.py:127`) is unreachable regardless (served sc_prob max 0.174).

So "+14 where the domain says impossible" is true and moves nothing: not the probability, not
threat_level, not routing, not the MC's `sc_s` draws. As a FINDING it is data-plumbing hygiene
(the builder fabricates a physically impossible value), not a serving defect. HIGH -> LOW.

### The live remnant A4 under-weighted (flagged, not fully measured)

`n_drivers` (the level, not the delta) DOES carry 64 splits — all thresholds in [16.5, 19.5]. The
same sparse frame that fabricates positive deltas can undercount a full 19-20-car field below 17,
which CAN cross those splits and move sc_prob (reads as "late-race depleted field"). This is the one
channel in the N14-1 family that can actually change a number; it was not sensitivity-tested here or
in A4 and is the honest follow-up measurement. The same fix covers both (below).

### Surgical change (worth doing only with the `n_drivers` follow-up)

**Where:** `race_situation_agent.py::_compute_driver_tyre_features` (the `cur`/`prev` slices feeding
line 492). **What:** count drivers by PRESENCE in the live field (the RSM `rivals` list + driver —
the repo's own #446 presence-over-inference rule), or at minimum
`n_drivers_delta = min(0, cur_count - prev_count)` with a comment stating why. **What must NOT
move:** the trained feature list/order; `lap1_chaos`'s formula. **Tests:** none exist on these
features (that absence is why it survived); add a builder test pinning `n_drivers_delta <= 0`.

---

## C4 — N15 `team_year_median` = 2.8 s fallback — VERDICT: mechanism CONFIRMED (with one scope correction); impact OVERSTATED: fixing it moves P50 by at most 0.118 s, and for McLaren/Ferrari by exactly 0.000 s.

### The mechanism, re-verified

`PitAgentCFG.team_year_median` loaded and dumped: **16 entries**, values matching A4's table
(('McLaren', 2024) 4.289, ('Ferrari', 2024) 4.370, ('Red Bull Racing', 2024) 2.654, ('Williams',
2024) 2.049, ...). 2025 entries exist only for `Haas F1 Team` (2.147) and `Williams` (4.163).
`team_year_median_for(team, 2025)` (`pit_strategy_agent.py:308-321`) is an exact-key lookup ->
**8 of 10 grid teams get the flat 2.8** (McLaren, Ferrari, Red Bull, Mercedes, Alpine, Racing
Bulls, Aston Martin, Kick Sauber). Real served rows via `_build_pit_duration_features` on Lusail
2025 (NOR/LEC/VER, 18 scenario rows): `team_year_median` = 2.8 on every row. The notebook's designed
same-team prior-year fallback chain (cell 16 `get_med`) is absent from the shipped lookup. All
confirmed.

**Scope correction to A4's "100% of served calls":** true for A4's sample (its five drivers are all
on fallback teams), but not a property of the system — a Williams or Haas call does NOT hit 2.8.
Wrinkle A4 missed: Williams gets **4.163** (sparse 2025 aggregate) where the notebook design intends
the richer 2024 prior (2.049) — the shipped code can serve a value FARTHER from the intended prior
than the fallback is. The defect is "wrong lookup chain", not "constant everywhere".

### The absorption measurement (this is what changes the severity)

Re-predicting the same real feature rows with the notebook-intended prior patched in
(McLaren 2.8->4.289, Ferrari 2.8->4.370, Red Bull 2.8->2.654):

```
team        prior change          dP50 (all rows)         dP95
McLaren        +1.489             0.000 s (6/6 rows)      0.000 s
Ferrari        +1.570             0.000 s (6/6 rows)      0.000 s
Red Bull       -0.146             -0.028 .. -0.118 s      +-0.19 s
```

|dP50| mean 0.021 s, max **0.118 s**. The quantile HistGBTs barely split on this feature above
~2.8 — moving the prior UP by 1.5 s changes nothing at all; the only measurable movement is Red
Bull's small downward correction. Downstream: `stop_duration_p05/p50/p95` feed the MC's pit draw
(`strategy_orchestrator.py:1371-1390`) and the displayed stop cost — a 0.12 s shift against ~20 s
pit deltas and a (2.2, 2.8, 3.8) fallback triangular is noise. Also note N15 executes only when N28
is routed on the RICH profile; the no-llm path never runs it (`pit_out=None` -> the hardcoded
triangular prior is used instead, unaffected by this claim entirely).

### Why the hunts missed it — the test asserts the training year

`tests/audit/test_pit_agent_hardening.py:305-320` (the #450 regression test) asserts
`team_year_median_for("Ferrari", 2024)` returns the loaded median. **2024 — a year the system never
serves.** For 2025, the only year live inference runs, the lookup was never asserted. This is the
repo's documented blind-spot pair in one place: "restaurar un dato y USARLO son dos cambios" (the
data was restored, its use on the served path never verified) + "el test tiene que assertar el
EFECTO, no el valor". The green test was true and irrelevant.

### Surgical change (wiring fix; small, worth doing for correctness despite the 0.12 s ceiling)

**Where:** `pit_strategy_agent.py::PitAgentCFG.team_year_median_for` (lines 308-321).
**What:** reproduce the notebook's chain: exact `(team, year)` -> same team, most recent PRIOR year
with data -> global fallback. (Decide at fix time whether a prior-year median should beat a sparse
same-year aggregate — the Williams case; the notebook's own `get_med` pools the team's medians
across its years, which sidesteps the sparse-2025 problem.)
**What must NOT move:** `_load_team_year_medians` aggregation, `_TEAM_ALIASES` normalisation, the
2.8 constant (still the last resort), `_build_pit_duration_features`'s call site.
**Tests that shift:** none break; ADD `team_year_median_for("McLaren", 2025) ==
approx(median[("McLaren", 2024)])` — the served-year assertion that was always missing. Goldens:
displayed P50/P95 for Red Bull-class teams move by up to ~0.2 s; MC-level goldens unaffected in
practice.

---

## Verdict summary, ranked by USER-VISIBLE impact

| # | Claim | Verdict | User-visible effect (measured) |
|---|---|---|---|
| 1 | **C1** N06 `Prev_LapTime` = 90.0 constant | **CONFIRMED** (arcade / CLI / backend-simulator; NOT the telemetry tab) | 7.2-13.8 s error in every displayed absolute pace prediction on replay surfaces vs 0.26-0.64 s correctly fed; invisible at ~90 s circuits and in the Dnext headline (algebraic cancellation); deterministic decision layer unaffected (MC ignores `lap_time_pred`) |
| 2 | **C4** N15 median fallback | **CONFIRMED mechanism, OVERSTATED impact** (HIGH -> LOW/MEDIUM) | at most 0.118 s in displayed stop P50; exactly 0.000 s for McLaren/Ferrari; MC noise-level |
| 3 | **C2** N12 gap scope | **CONFIRMED guard gap, OVERSTATED impact** (HIGH -> LOW/MEDIUM) | at most 0.29 in a displayed probability; 0 threat/routing/decision changes in 5 races; tree splits cap at 2.467 so all out-of-scope gaps score identically |
| 4 | **C3** N14 `n_drivers_delta` | **range CONFIRMED, damage REFUTED** (HIGH -> LOW) | exactly 0.0 — the feature has no splits in the fitted model. (`n_drivers` level is the one live channel; follow-up flagged) |

**The general lesson this gate adds to A4:** for tree-ensemble models, the min/max-escape lens ranks
severity BACKWARDS. Out-of-support escapes (C2, C3) are clamped to boundary behaviour by the tree
geometry itself — they are hygiene, not damage. The killer defect class is the **in-support wrong
value** (C1's 90.0, inside the trained [67.7, 149.0] band; A4 said this itself in N06-1's last
paragraph but did not apply the corollary to its own ranking). Range-escape percentages are the
wrong severity metric; "does the served value move the output vs the true value" is the right one,
and it inverts the list.

## Answer to the owner's two questions

**"The model values look close to the real ones."** You are right, on both surfaces you named — and
so is claim 1. The telemetry tab is served by the backend strategy endpoints, which build
`lap_state` from the FEATURED parquet with a real `Prev_LapTime` (`strategy.py:442,487,828`) — the
90.0 bug does not exist there, and the Race Trace / AgentTabs numbers you see are genuinely
~0.3-0.6 s-grade. The arcade headline shows Dnext, from which the anchor cancels exactly, and at
Lusail-class circuits (~89.5 s real pace) even the absolute chart line sits inside the +-2.5 s CI
band of the 90.0-anchored prediction. Run the arcade at Monaco and watch the dashed line: it will
sit ~12 s above the solid one. That is the bug, visible on demand.

**"Why would these have survived the bug hunts?"** Because three of the four cannot produce a
symptom (C3 provably, C2/C4 within measured noise), and the one that can (C1) self-conceals four
ways: the headline metric algebraically cancels it; the canonical demo circuit sits at the anchor
value by coincidence; the surface people use for model-vs-real comparison (the telemetry tab) is
the one surface on the FIXED path; and the #435 regression tests assert the fixed producer on the
featured parquet while no test constructs an RSM from a raw-schema frame — the only configuration
every replay surface actually runs. Symptom-driven hunts cannot find symptomless defects, and the
guards that exist assert the wrong year (C4), the wrong parquet (C1), or nothing at all (C2, C3).

## What I tried to break and could NOT

- **The claimant's harness equivalence** — tried to show `run_pace_agent_from_state(st)` differs
  from what arcade executes; it does not (same function object, same dict, verified through 4 hops).
- **A hidden `prev_lap_time` derivation** — searched RSM and the replay engine for any fallback that
  computes the previous lap from the raw frame; none exists.
- **A4's headline numbers** — independently re-measured on the real loop; reproduced to the third
  decimal (7.717 / 13.823 / 7.156 vs claimed 7.72 / 13.83 / 7.16).
- **A decision-layer consumer of `overtake_prob`** — searched MC, projection, routing, rails; only
  threat bands, displays and prompt text consume it. Could not make C2 decision-relevant.
- **An out-of-scope overtake call crossing the MEDIUM band** — 106 real out-of-scope calls across 5
  races; max 0.290 vs band 0.40. Could not.
- **Any effect of positive `n_drivers_delta`** — could not, and proved why (0 splits).
- **A large P50 shift from the correct pit prior** — max 0.118 s across 18 real scenarios; for the
  two teams with the largest prior error (McLaren/Ferrari, +1.5 s), exactly 0.000.
- **The trained-range claims themselves** — re-derived from the parquets: 18,277 rows / gap max
  2.500 exact; 2,280 rows / delta in [-11, 0] / zero positives. All match A4.
- **The "dead bands" alternative explanation for C2/C3 absorption** — verified the threat bands are
  alive post-#665 (in-scope probs reach 0.718; zeroing all overtake probs flips 7/275 laps), so the
  absorption results are not an artifact of thresholds that never fire.

Sample limits, stated: 5 races (Lusail, Monza, Silverstone, Monaco, Spa-Francorchamps), NOR only,
no race with a live SC/VSC period in the sample; `n_drivers` level-feature sensitivity not measured
(flagged as the follow-up); rich-profile LLM compliance with the prompt's gap rule not measurable
offline.
