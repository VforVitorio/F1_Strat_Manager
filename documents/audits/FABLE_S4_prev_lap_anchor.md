# FABLE S4 — Adversarial gate: prev-lap anchor reconstruction (PR #747 / issue #728)

**Auditor:** Fable adversarial gate (read-only; only this file is written).
**Date:** 2026-07-30
**Scope:** `RaceStateManager._precompute_prev_lap_times()` (src/simulation/race_state_manager.py:154-221),
its consumer chain through `get_driver_state()["prev_lap_time"]` → `pace_agent.run_from_state()` →
`_predict()` (anchor `prev + delta`), and the claims A-F below.

## Checklist

- [x] A. Reproduce the three median |anchor − lap| numbers — **CONFIRMED, exact, stated drivers match.**
- [x] B. Monaco convergence, before/after, real entry point headless — **CONFIRMED (14.671 s → 0.496 s median; 67 laps improve, 0 worsen).**
- [x] C. Blast radius / decision-layer byte-identity — **CONFIRMED (MC dicts equal both branches; full no-llm recommendation identical on 78/78 laps).**
- [x] D. No out-lap / in-lap / SC / cross-stint anchor — **CONFIRMED (0 invariant mismatches over ~10,400 laps; 2,907/2,907 exact vs N04's own column).**
- [x] E. Residual 90.0 quantified — **CONFIRMED with caveat (10-20% dry, ~40% in the two wet races; degrades to old behaviour, never worse).**
- [x] F. Featured-column path byte-identical — **CONFIRMED (0 diffs; branch is defensive — no production caller passes the column).**
- [x] Bug classes: sentinels (held; one benign real 90.000), Series.get guard (reached both paths), producers (TWO unnamed accumulators found + a pre-existing `pace_delta_s` semantic fork, MEDIUM), unscoped data (inherited contract, nothing new), tests (all 8 redden under mutation).
- [x] Fix fallout: ctor cost ≈ 0 ms, strict-zip cannot raise (~480 real builds), duplicates absent from real data (synthetic → benign), NaN Stint matches N04 dropna semantics.

## Verdict per claim

| Claim | Verdict |
|---|---|
| A — three medians | **VERIFIED** (exact reproduction, driver-specific match) |
| B — Monaco convergence | **VERIFIED** (headless, real entry point, both passes executed) |
| C — decision layer does not move | **VERIFIED** (three independent executed attacks) |
| D — no bad lap becomes the anchor | **VERIFIED** (invariant + N04 ground-truth, multi-race) |
| E — degrades to old 90.0, never worse | **VERIFIED**, with the wet-race concentration caveat (LOW) |
| F — featured path untouched | **VERIFIED** (executed byte-identity; path is defensive-only) |

## Findings

### A. CONFIRMED — the three medians reproduce exactly, with the stated drivers

Probe: `probe_a_medians.py` (scratchpad), pure `RaceStateManager` over `data/raw/2025/<GP>/laps.parquet`,
walking `get_driver_state(lap)` for every lap. Definition that reproduces the claim: median of
`|anchor − lap_time_s|` over **every lap with a real lap time**, where `anchor = prev_lap_time` when the
reconstruction produced one and **90.0 on the residual fallback laps** — i.e. exactly what the pace agent
receives via `d.get('prev_lap_time') or 90.0` (src/agents/pace_agent.py:725).

| Race | Driver/Team used | Claimed before → after | Measured before → after | Match |
|---|---|---|---|---|
| Monaco 2025 (78 laps) | NOR / McLaren | 14.686 → 0.480 | 14.686 → 0.480 | EXACT |
| Monza 2025 (53 laps) | LEC / Ferrari | 6.889 → 0.191 | 6.889 → 0.191 | EXACT |
| Lusail 2025 (57 laps) | NOR / McLaren | 4.955 → 0.158 | 4.955 → 0.158 | EXACT |

Cross-check that the numbers are driver-specific (they are, so a lucky coincidence is excluded):
Monaco LEC gives 14.600 → 0.465; Monza NOR gives 7.033 → 0.158; Lusail VER gives 5.032 → 0.167.
The claimed triple matches only the stated (driver, team) pairs. No partial-match warning to raise.

Anchored-laps-only medians (excluding the 90.0 residue) are lower still: Monaco 0.413 (67/78 laps
anchored), Monza 0.152 (47/53), Lusail 0.121 (43/57). Max |anchor − lap| among anchored laps:
Monaco 4.522 s, Monza 0.884 s, Lusail 1.898 s — no out-lap-sized (>20 s) leak among anchored laps
in these three races (full multi-race scan under D).

### E (part 1) — residual 90.0 fallback: where it concentrates

From the same probe, laps still emitting `prev_lap_time=None` (→ pace agent anchors on 90.0):

- Monaco NOR: 11/78 laps — 1-5, 19-21, 50-52 (race start window + the two pit-stop windows).
- Monaco LEC: 12/78 — 1-5, 21-24, 49-51.
- Monza LEC: 6/53 — 1, 2, 4, 33-35. Monza NOR: 5/53 — 1, 2, 46-48.
- Lusail NOR: 14/57 — 1, 2, 7-11 (early SC period), 25-27, 29, 44-46. Lusail VER: 10/57.

Pattern confirmed: the residue sits exactly on (a) laps 1-2 by construction, (b) SC periods
(IsAccurate=False), (c) the out-lap + first lap after each stop. That is the lap-after-every-stop
concentration the claim itself predicted; behaviour there degrades to the OLD 90.0, not to a new
error. Quantified further in part 2 (multi-race scan).

### B. CONFIRMED — predicted pace converges on real pace at Monaco, via the REAL entry point

I could not drive the PySide6 GUI in this environment, so per the gate instructions I drove the SAME
code path headlessly: `RaceReplayEngine("data/raw/2025/Monaco", "NOR", "McLaren", interval_seconds=0)`
→ per-lap `RaceState` → `src.strategy.inference.engine.run_lap(..., profile="no-llm")` (zero LLM
clients by construction), comparing `agent_outputs["pace_out"].lap_time_pred` against the real
`lap_time_s`, 78/78 laps, both passes in one process. BEFORE = `RaceStateManager.
_precompute_prev_lap_times` monkeypatched to return `{}` in the probe process only (source untouched).
Probe: `probe_b_monaco.py` + `analyze_b.py` (scratchpad), results in `probe_b_results.jsonl`.

| Pass | n | median \|pred − real\| | mean | max |
|---|---|---|---|---|
| BEFORE (90.0 anchor everywhere) | 78 | **14.671 s** | 13.823 | 20.277 |
| AFTER (reconstruction) | 78 | **0.496 s** | 2.125 | 20.277 |

- Paired per lap: **67 improved, 0 worse, 11 unchanged** (the unchanged 11 are exactly the residual
  `None`→90.0 laps from finding E — behaviour there is byte-identical to before).
- AFTER split: anchored laps (67) median 0.391 s, max 4.534 s; 90.0-fallback laps (11) median
  11.971 s, max 20.277 s — the max error is confined to laps that were EQUALLY wrong before
  (worst: lap 3, real 109.0 s under the wet/SC opening, pred 88.7 both passes).
- Engine-level confirmation of claim E's degradation promise: no lap got WORSE (0/78).

Verdict: the arcade's predicted-pace line, which renders `per_agent.pace.lap_time_pred`
(src/arcade/dashboard/pace_chart.py:11, window.py:268-269), now tracks the real line at Monaco to a
median half-second instead of sitting ~15 s away. Claim B holds.

### C. CONFIRMED — the deterministic decision layer does not move; attacked three ways

**1. Exhaustive consumer sweep of `PaceOutput` (not just `lap_time_pred`).** Readers, verified by
grep over src+scripts:

- `lap_time_pred`: arcade displays (src/arcade/dashboard/window.py:268-269, reasoning_tabs.py:110,
  agent_formatters.py:96, pace_chart.py:11), CLI display (scripts/run_simulation_cli.py:784,
  `getattr(..., None)` → idle row), backend SSE payload "pred"
  (src/telemetry/backend/api/v1/endpoints/strategy.py:703), the Layer-3 prompt
  (strategy_orchestrator.py:1555), and the MC's `pace_s` draw (:1386) — which carries
  `# noqa: F841` and is never read again (only occurrence in the function; `_run_projection_mc`'s
  call at :1396-1408 passes cliff/sc/pit/ucut draws only).
- `ci_p10`/`ci_p90`: `sigma_pace` (:1324) — used ONLY to scale the unused `pace_s` draw — and the
  prompt (:1549-1550, :1558). `delta_vs_prev`/`delta_vs_median`/`reasoning`: prompt only
  (:1556-1559). `apply_guard_rails` (src/strategy/inference/guard_rails.py) takes no pace argument
  (no_llm.py:293-305). `_assemble_recommendation` receives no pace values (engine.py:321-343).

**2. The RNG-stream subtlety, executed.** `pace_s = rng.normal(loc, scale, n)` is drawn FIRST from
the shared seed-42 generator; if changing `loc`/`scale` perturbed the stream, every subsequent draw
(cliff_s, sc_s, pit_s, ucut_s) would move even with `pace_s` unread. Executed check: two
`default_rng(42)` streams, `normal(89.8, 0.72, 500)` vs `normal(75.3, 0.28, 500)`, then 8 more
uniforms — identical tails (`RNG_STREAM_UNAFFECTED True`; numpy applies loc/scale as an affine
transform after the ziggurat draws, so stream consumption is value-independent).

**3. Byte-identity, executed at two levels** (`probe_c_mc.py`, `probe_c2_rec.py`):

- `_run_mc_simulation` with two radically different PaceOutputs (89.812 wide-CI vs 75.334
  narrow-CI), all else fixed: legacy seconds branch (rivals=None) → results dicts EQUAL;
  projection branch (real Monaco lap-30 rivals + real race context, branch verified taken) →
  EQUAL.
- Full engine, Monaco, 78 laps, `profile="no-llm"`, helper stubbed vs real (same harness as
  claim B): `StrategyRecommendation.model_dump()` compared per lap — **78 compared, 0 differ**.
  The entire deterministic recommendation (action, confidence, scenario_scores, every one of the
  14 fields) is byte-identical before/after the anchor fix.

Goldens: `tests/mc/test_strategy_goldens.py` + `tests/mc/test_projection_golden.py` pass (35
passed in the baseline run), but note they construct canned `PaceOutput` fixtures
(test_strategy_goldens.py:49-51), so they are green by construction against THIS change — the
evidence that the goldens "don't move" is the executed A/B above, not the golden suite itself.

One consumer the claim's wording missed but which stays inside "the prompt": `mcp_tools.py:593-594`
reads `driver_state.get("prev_lap_time")` into `pace_delta_s`. That path builds its lap_state from
the backend's featured-parquet `get_lap_state` (mcp_tools.py:397-413 → producer 2, #746), NOT from
the RSM, so this PR does not change its inputs; and `RaceState.pace_delta_s`'s only consumer is the
prompt (strategy_orchestrator.py:1662). Claim C holds.

### D. CONFIRMED — no out-lap, in-lap, SC lap or cross-stint lap becomes the anchor (10 races, all drivers)

Two executed attacks (`probe_d_scan.py`, `probe_d2_vs_n04.py`):

1. **Independent invariant recompute.** For 10 races (Monaco, Monza, Lusail, Silverstone, Spa,
   Melbourne, São Paulo, Zandvoort, Las Vegas, Suzuka 2025) × every driver (~196 RSM builds,
   ~10,400 laps with a lap time), I re-derived "previous surviving lap in the same stint" in a
   pure-Python loop (no pandas groupby, different code shape) and compared with every emitted
   `prev_lap_time`. **Mismatches: 0.** By construction of the recompute, that excludes any
   cross-stint anchor and any anchor from a lap failing `IsAccurate & ~Deleted & <180 s & lap>1`.
2. **Against N04's own ground truth.** The featured parquet's `Prev_LapTime` column IS the
   training-time output of this transform. Monaco+Monza+Lusail, all drivers: **2,907 (driver, lap)
   pairs compared — 0 value differences, 0 laps where featured has a value and the reconstruction
   does not, 0 laps where the reconstruction invents a value featured lacks** (tolerance 1.5 ms,
   the featured column's own rounding). The reconstruction is N04's transform, measured, not
   claimed.

Plausibility sweep (>8 s from the lap's own time, current lap green + accurate): 5 laps in
~10,400 — Melbourne TSU L46 (12.7 s), Silverstone SAI L44 / RUS L8 / HAD L8 / LEC L8 (8.3-10.9 s).
All five are in the two 2025 WET races and every anchor equals the true previous surviving lap
(check 1 above): these are real pace swings on a drying/soaking track, not reconstruction leaks.
On a genuinely dry green lap no implausible anchor exists in the scan.

### E (part 2). CONFIRMED, with a quantified caveat — residual 90.0 share per race, and where it hides

Residual `None` (→ pace agent anchors 90.0) as a share of laps that have a real lap time, ALL
drivers per race: Suzuka 10.5% · Monza 11.9% · Monaco 15.4% · Spa 16.0% · Las Vegas 17.9% ·
São Paulo 20.3% · Lusail 20.6% · Zandvoort 28.2% · **Melbourne 39.9% · Silverstone 40.2%**.

- The floor (~10-15%) is structural: laps 1-2, plus ~3 laps per pit stop (in-lap and out-lap are
  `IsAccurate=False` so they are not keys; the first flying lap has no surviving predecessor in
  its new stint).
- The caveat: the residue concentrates exactly in the WET/SC-heavy races (Melbourne, Silverstone
  2025 — ~40% of laps still 90-anchored), which is also where real pace sits furthest from any
  anchor. Behaviour there is the OLD behaviour, verified byte-identical in the claim-B pairing
  (0 laps got worse), so the claim "degrades to the old 90.0 rather than a new, larger error"
  holds. But the headline "the anchor is fixed" is a DRY-race statement; in a wet race roughly
  every third pace prediction is still pinned near 90.0. Severity: LOW (no regression, honest
  degradation), recorded so nobody reads the Monaco medians as universal.

### F. CONFIRMED — the featured-column path is byte-identical, and no production caller uses it

Executed (`probe_f_edge.py`): Monaco raw frame with the real featured `Prev_LapTime` merged on
(the only way to build a validating frame WITH the column — `validate_laps_df` rejects the featured
parquet itself for lacking timedelta `LapTime`/`Time`). Emitted `prev_lap_time` for every lap,
old behaviour (helper monkeypatched to `{}`) vs new: **0 differences across all 78 laps.** The
reason NaN-column laps cannot diverge is finding D-2: the reconstruction's key set coincides
exactly with the featured column's non-NaN set, so where the column is NaN the fallback is also
absent and both paths emit `None`.

Also verified: the only non-test constructors of `RaceStateManager` are
`src/simulation/replay_engine.py:94` (raw per-race parquet — no `Prev_LapTime`) and
`scripts/bench_subagent_latency.py:125`. The column-precedence branch
(race_state_manager.py:340-344) is defensive; `tests/audit/test_pace_orchestrator_hardening.py:114`
pins it and is green.

### Tests — all 8 can fail (mutation matrix, in-memory only, source untouched)

Three mutations of `_precompute_prev_lap_times` injected via pytest plugins (`mut_m1/m2/m3.py`,
scratchpad; `pytest_configure` patches the class before collection — no repo file touched):
M1 = return `{}` · M2 = naive `shift(1)` with no filter and no stint grouping · M3 = 999.0 for
every lap.

| Test (tests/simulation/test_rsm_prev_lap_time.py) | M1 | M2 | M3 |
|---|---|---|---|
| test_a_raw_frame_gets_the_previous_lap | **FAIL** | pass | **FAIL** |
| test_lap_one_and_the_lap_after_it_stay_unknown | pass | **FAIL** | **FAIL** |
| test_an_out_lap_never_becomes_the_anchor | **FAIL** | **FAIL** | **FAIL** |
| test_the_shift_does_not_cross_a_stint_boundary | **FAIL** | **FAIL** | **FAIL** |
| test_a_deleted_lap_is_skipped | **FAIL** | **FAIL** | **FAIL** |
| test_the_featured_column_wins | pass | pass | **FAIL** |
| test_a_frame_without_stint_yields_no_reconstruction | pass | **FAIL** | **FAIL** |
| test_the_real_monaco_parquet (data-gated) | **FAIL** | pass | **FAIL** |

Every test reddens under at least one mutation — none asserts about the empty set. Baseline: the
8 tests + `tests/mc/test_strategy_goldens.py` + `tests/mc/test_projection_golden.py` +
`tests/audit/test_pace_orchestrator_hardening.py` = 35 passed unmutated.

LOW note: the data-gated Monaco test alone survives M2 (its `max(errors) < 25.0` band is loose
enough for a naive no-filter shift to slip under at Monaco). The hermetic out-lap and stint-boundary
tests are what actually kill M2 — fine as a suite, but the real-parquet test should not be read as
a standalone guard against the naive implementation.

### Project bug classes, hunted

**Sentinel collisions — held, with one measured near-miss.** Scanned every reconstruction value in
every 2025 race (all drivers): minimum 67.924 s, no 0.0 anywhere, and exactly ONE real value equal
to the 90.0 fallback: Yas_Island HAD lap 28, whose true previous lap really was 90.000 s. That is
harmless today — no consumer branches on `== 90.0`; the value flows into the feature either way,
and `pace_agent.py:725`'s `or 90.0` fires on `None`/falsy only (a real 0.0 lap time cannot exist;
measured floor 67.9 s). The three meanings stay separated: absent key = unknown → `None` →
old fallback; present value = the real N04 quantity.

**`Series.get` returning a stored NaN — the guard is real and reached on both paths.** Executed:
on the RAW path `r.get("Prev_LapTime")` returns `None` (column absent) → `pd.notna` False → the
reconstruction fires (67 anchored Monaco laps, finding A). On the WITH-column path a stored NaN
falls through the same guard to the reconstruction (finding F: 0 divergences, and
test_the_featured_column_wins pins both directions). race_state_manager.py:340-344 is the one
place the branch lives; no other consumer re-reads the raw column.

**Unscoped data — inherited, not introduced.** `_precompute_prev_lap_times` operates on
`self._driver`, already filtered to ONE driver; stint grouping never mixes drivers. It stays inside
one GP only because `RaceStateManager`'s contract is a single-race frame (both production
constructors — replay_engine.py:94, bench_subagent_latency.py:125 — load one `data/raw/<year>/<GP>/
laps.parquet`). A season-wide frame WOULD interleave races inside the same stint numbers, but it
would equally corrupt `_precompute_leader_times` and `total_laps`, which predate this PR: the
helper adds no new unscoped surface. Verified by reading every constructor call site.

**The twin that never got the fix — the named three are real, and TWO more producers exist that
nobody named.** Full inventory of "previous-lap value" producers, verified at file:line:

1. `RaceStateManager._precompute_prev_lap_times` (race_state_manager.py:154) — N04 semantics
   (previous SURVIVING lap, same stint). Feeds the pace anchor. This PR.
2. Backend `_prev_lap_time_for_row` (src/telemetry/backend/api/v1/endpoints/strategy.py:804,
   used at :487 and :860) — featured-column read. **#746 already filed**, not re-reported.
3. Arcade accumulator (src/arcade/strategy.py:371, :392, :404, :423; consumed at :599-614) —
   carries the last COMPLETED lap's time and emits `pace_delta = cur − prev`.
4. **UNNAMED FOURTH: the backend simulator's accumulator** —
   src/telemetry/backend/services/simulation/simulator.py:842 (`prev_lap_time = 0.0`), :877
   (passed into `_build_race_state`), :923 (updated per lap), :398 (`pace_delta = cur − prev if
   prev else 0.0`). Same pattern as arcade's, third body copy.
5. **UNNAMED FIFTH: the CLI's accumulator** — scripts/run_simulation_cli.py:1537
   (`prev_lap_time: float = 0.0`), :1691-1692, :1339 (`pace_delta_s = cur − prev if prev else 0.0`).

Is the arcade accumulator "genuinely a different quantity"? **Half yes, half no.** Yes: it never
feeds the pace agent's anchor (the anchor comes only from `lap_state["driver"]["prev_lap_time"]`,
i.e. producer 1 — verified by the exhaustive reader grep: the only readers of that key are
pace_agent.py:725 and mcp_tools.py:593). No: the accumulator IS a previous-lap-time — the variable
is literally named `prev_lap_time` — with DIFFERENT semantics (last completed lap: includes
in-laps, out-laps, SC laps) from the N04 quantity producer 1 now enforces. On the lap after a pit
stop, producers 3/4/5 hand the prompt a `pace_delta` computed against the ~20 s slower in-lap
while producer 1 hands the pace model an honest `None`. Two disagreeing definitions of "previous
lap" now coexist per lap in the same pipeline.

**MEDIUM finding (pre-existing, not moved by this PR): `RaceState.pace_delta_s` receives two
different quantities depending on surface, against its own schema.** The schema says rival-relative:
race_situation_agent.py:292 ("3-lap rolling pace delta vs car ahead"), and the rival-targeting
builder computes it that way (src/telemetry/backend/utils/race_state_builder.py:27,46 — driver
minus rival). But the CLI (:1339), arcade (:614), backend simulator (:398) and mcp_tools
(mcp_tools.py:594, computed from producer 2's `prev_lap_time`) all feed a SELF-delta (this lap vs
own previous lap). Consumer: the Layer-3 prompt line `Pace delta: {:+.3f}s`
(strategy_orchestrator.py:1662) — so the LLM reads a number whose meaning flips by surface, and on
post-stop laps the self-delta versions read ~-20 s of phantom "pace gain". Deterministic layer
unaffected (no non-prompt consumer — verified by grep). Recommend an issue; do NOT fold into #746,
which covers producer 2's stint-boundary bug, not this semantic fork.

### What the fixes themselves may have broken — measured

- **Construction cost:** timing 10 constructions of RaceStateManager over the real Monaco raw frame:
  28.7 ms with the helper vs 28.8 ms with it stubbed out — **delta ≈ 0 (below noise)**. The dict
  holds ≤ (laps−1) floats (~67 entries at Monaco); memory is trivial. The arcade per-lap loop and
  the backend simulator construct the RSM once per race, not per lap (replay_engine.py:94), so even
  a real cost would amortise. No regression.
- **`strict=True` zip:** cannot raise by construction (`previous` is a groupby-shift over
  `ordered`'s own index, so lengths are always equal) and did not raise in practice: ~480 real
  constructions (24 races × all drivers, `probe_f_edge.py` dtype sweep) + the NaN-stint synthetic —
  0 errors.
- **Duplicate `LapNumber` rows:** no real 2025 raw parquet has any (`TOTAL_DUP_ROWS 0` across all
  24 races). Synthetic behaviour: no crash; the dict comprehension keeps the LAST duplicate's
  shift, so a duplicated lap would anchor on its own first row's time (executed: laps
  [2,3,3,4] → `{3: 76.0, 4: 76.5}`). Latent oddity with no real-data trigger — LOW, note only.
- **`Stint` as float with NaNs:** no crash; NaN-stint rows are dropped by the groupby exactly as
  N04's own grouping drops them — the NaN-stint lap emits `None` and is never an anchor, and the
  shift within the surviving stint reaches PAST it (executed: stints [1, NaN, 1, 1] → lap 4
  anchors on lap 2). Matches training semantics.
- **Flag dtypes:** `IsAccurate`/`Deleted` are proper bool/boolean dtype in all 24 raw 2025
  parquets — the `astype(bool)` path never sees the `"False"`-string trap `warn_low_quality_laps`
  guards against.

## Findings by severity

- **HIGH: none.** Every claim A-F verified with executed evidence.
- **MEDIUM (1, pre-existing, not moved by this PR):** `RaceState.pace_delta_s` semantic fork —
  four surfaces feed a SELF-delta (CLI :1339, arcade :614, backend simulator :398, mcp_tools :594)
  into a field the schema and the rival-targeting builder define as RIVAL-relative
  (race_situation_agent.py:292, race_state_builder.py:27). Post-stop laps hand the LLM ~-20 s of
  phantom "pace gain". Prompt-only consumer, so no deterministic impact.
- **LOW (4):**
  1. Wet-race residual concentration — ~40% of Melbourne/Silverstone laps still 90.0-anchored
     (honest degradation, but the Monaco medians are a dry-race statement).
  2. Two unnamed accumulator producers of `prev_lap_time` (backend simulator, CLI) beyond the
     three the team has inventoried — same body copied three times, semantics differ from the N04
     quantity producer 1 now enforces (last COMPLETED lap vs last SURVIVING lap in stint).
  3. The data-gated Monaco test passes under the naive-shift mutation (its <25 s band is loose);
     only the hermetic tests kill that mutant.
  4. Duplicate-LapNumber rows would self-anchor (dict comprehension keeps the last shift) — zero
     occurrences in real 2025 data; latent only.

## Numbered fix list (by value, then risk)

1. **File an issue for the `pace_delta_s` semantic fork** (MEDIUM above): either rename the
   self-delta variants (`own_pace_delta_s`) or compute the rival-relative quantity everywhere the
   schema promises it; at minimum make the prompt label say which one it is. Do not fold into
   #746 — that issue is producer 2's stint-boundary bug, this is a cross-surface schema lie.
2. **Name producers 4 and 5 in the #746 discussion** (backend simulator simulator.py:842/:923;
   CLI run_simulation_cli.py:1537/:1691) so the next prev-lap fix has the full twin inventory —
   the exact failure mode that has bitten this repo repeatedly.
3. **Consider a follow-up for wet races** (LOW-1): when the reconstruction yields None on >N
   consecutive laps, the 90.0 constant is the pace line for whole stints at Melbourne/Silverstone.
   A compound-aware or session-median fallback would need its own design review (sentinel rules);
   record it as future work, not a quick patch.
4. **Tighten the data-gated Monaco test** (LOW-3): add `assert median(errors on laps whose
   lap-before-was-inaccurate)`-style structure, or drop the max band to <10 s at Monaco, so the
   real-parquet guard alone would kill a naive shift.
5. Optional cleanup (cosmetic): `_precompute_prev_lap_times` recomputes `_lap_time_s` although
   the enriched frame already carries `lap_time_s` from `_compute_session_times`
   (race_state_manager.py:200 vs :62) — a spare `.map(_to_seconds)` per construction. Zero
   measured cost; tidy only if touched again.

## What I tried to break and could NOT

1. **The three claimed medians** — tried alternate definitions (anchored-only median, other
   drivers) to make the claimed numbers look cherry-picked; the stated definition + stated drivers
   reproduce all three EXACTLY, and neighbouring drivers give different numbers, so they are real
   measurements, not coincidences.
2. **The reconstruction vs N04** — tried to find one (driver, lap) in three full races where the
   helper disagrees with the featured parquet's own `Prev_LapTime`: 2,907 comparisons, zero. Also
   tried an independently-coded invariant across 10 races (~10,400 laps): zero mismatches.
3. **An implausible anchor on a dry green lap** — the >8 s sweep found only wet-race laps whose
   anchor is nonetheless the true previous surviving lap. Could not construct or find an out-lap,
   in-lap, SC-lap or cross-stint anchor in real data.
4. **Moving the deterministic layer through pace** — tried radically different PaceOutputs on both
   MC branches, checked the RNG-stream coupling explicitly, and diffed the full 14-field
   recommendation over 78 real laps: byte-identical everywhere.
5. **Crashing the helper** — `strict=True` zip with NaN stints, float stints, duplicate laps,
   missing Stint column, missing flags columns, ~480 real-race constructions: no exception, no
   wrong-length zip.
6. **A regression on any lap** — paired before/after per-lap errors at Monaco: 0 of 78 laps got
   worse; the residual laps are byte-identical to the old behaviour.
7. **The `Series.get` NaN trap** — tried to reach a state where a stored NaN survives the
   `pd.notna` guard or where the raw path keeps the 90.0: both paths verified to route through the
   guard correctly.
8. **A sentinel collision** — scanned every reconstruction value of the 2025 season for 0.0/90.0:
   min 67.924 s, one real 90.000 (Yas Island HAD L28) with no consumer that branches on equality.
9. **Construction-cost regression** — measured, not estimated: ≈ 0 ms delta over the real Monaco
   frame.

## Repo hygiene

No repository file was modified except this report. Probes, mutation plugins and result files live
in the session scratchpad (`probe_a_medians.py`, `probe_b_monaco.py`, `analyze_b.py`,
`probe_c_mc.py`, `probe_c2_rec.py`, `probe_d_scan.py`, `probe_d2_vs_n04.py`, `probe_f_edge.py`,
`mut_m1/m2/m3.py`, `probe_b_results.jsonl`, `probe_c2_recs.jsonl`). All monkeypatching was
in-process only. Final `git status --short`: `M src/telemetry` (submodule dirtiness =
untracked `.claude/` and `docs/migration/streamlit-reference/` INSIDE the submodule — present in
the session-start snapshot, not touched by this audit), untracked
`notebooks/strategy/overtake_probability/outputs/n12b_scoreboard.png` (also pre-existing), and
this report. Nothing else.
