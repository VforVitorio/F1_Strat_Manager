# GATE — PR #815 (replay weather join) + PR #816 (tire serving frame)

**Gate run:** 2026-08-04 · branch `dev` @ `7aa51da` · adversarial gate, no repo file modified except this report.
**Scope:** commits `556007f` (weather), `ae3894f` + `57e840b` (tire frame). Sources of truth: `GATE_DATA_WIRING.md` (F7–F11), `GATE_801_ARTEFACTS.md` (fix list items 1–2), `.nb_py/` notebook exports.
**Method:** every commit-message / code-comment claim is re-derived with EXECUTED evidence. Findings appended as confirmed, never buffered.

## Checklist (updated as worked)

- [x] C1 — VERIFIED (V-C1): 0.000000 vs artefact and vs independent join, 33 combos incl. rain/red-flag/short-frame/non-NOR — but the fallback branch mis-documents itself and serves fiction (F-1)
- [x] C2 — VERIFIED to the digit (V-C2): 79,032 / 94.3% / 1.488 / 17.3 / 3,399
- [x] C3 — id() collision REPRODUCED on trial 2 (F-2, latent); mutation staleness executed; no NaT-Time laps exist in shipped data
- [x] C4 — VERIFIED (V-C4): wind 0.000000 on 3 races; arcade/backend consumers unregressed
- [x] C5 — VERIFIED (V-C5): N04 unshifted, N09 as-stored — but N04 also CLIPS, and serving does not (F-3)
- [x] C6 — VERIFIED (V-C6): single fillna(0)-before-scaler entry, no NaN reaches the TCN — but two intermediate fillna(0)s manufacture wrong non-zeros at stint position 1 (F-4, F-10)
- [x] C7 — VERIFIED incl. the harder half (V-C7): pooled holds 22/22 (2023), 24/24 (2024), 24/24 (2025)
- [x] C8 — VERIFIED beyond the claim (V-C8): std 0.0 per (Cluster, Year), max dev 0.0 over all 68,122 rows; 0.0-sentinel unreachable from the map; gp-miss default measured (F-9)
- [x] C9 — REFUTED (F-5): a THIRD builder in the backend submodule serves the race mean, and the PR's unconditional recompute makes that path worse
- [x] C10 — 22.4% reproduces only when NaN counts as disagreement (20.8% honest, 15.8% is the harness number) (F-8); guard cannot leave the column absent (verified); its regression test is vacuous (F-7)
- [x] H1 — backend Tyres tab regressed via F-5; arcade weather panel unregressed; goldens/canned outputs unaffected by construction; suite re-run below
- [x] H2 — third sites found: F-5 (session_meta builder), F-10 (NaN-convention family), plus wrong-pointer comments (F-6); no third proportional-weather site exists
- [x] H3 — measured: TCN |serving−trained| mean 0.148 / p95 0.454 / max 4.93 vs the promised ~0.02; old-vs-new 0.399 mean / 5.21 max, consistent with the commit's 0.42 / 4.99

## Findings

### V-C2 — VERIFIED to the digit: the commit's headline measurement reproduces from an independent implementation.

Re-derived with a from-scratch `np.searchsorted` nearest-Time join (NOT `weather_restore`,
so the check is not circular) over all 71 raw race dirs: **79,032 driver-laps, TrackTemp
differs (>0.05 C) on 74,545 = 94.3%, mean |Δ| 1.488 C, max 17.3 C (Miami 2023), rain flag
flips on 3,399 laps.** All five numbers in `556007f`'s message match exactly. Worst races:
Silverstone 2024 (5.07 C mean, 475 flips), Barcelona/Spain 2023 (3.75 — counted twice, the
GATE_801 C1 duplicate is inside this 79,032 too), Monaco 2023 (3.75, 178 flips). Note the
79,032 also counts the Barcelona↔Spain duplicated race twice, same caveat as the projection
sample (GATE_801 C3-b); it does not change the conclusion.

Also measured while there: **0 rows with NaT `Time` across all 71 raw laps parquets** — the
"lap has no session time to join on" branch the new docstring worries about never fires on
shipped data — and **6,236 (driver × missing-lap) pairs** where a driver has no row for a
lap ≤ total_laps (retirees + lapped cars), which is the territory of the new
`weather_df.iloc[0]` fallback (see F-1).

### Context fact the PR leans on, verified against GATE_801 D1: the verification chain is real but lives in `laps_tiredeg.parquet`, not the featured artefacts.

The commit cites "verified 22,760/22,760 against N04's own output". The combined
`laps_featured.parquet` on disk today has **48 columns and no weather** (executed check), so
that test (`test_weather_restore`) fails on this checkout — the commit itself declares it as
the one pre-existing failure, and GATE_801 D1 explains why (HF re-download overwrote a local
regeneration). The TRAINED weather truth is still on disk in `laps_tiredeg.parquet` (68,122
rows, 2023-2025, all four columns), which is what this gate uses as ground truth below.

### V-C1 — VERIFIED: the REAL `get_weather_state` equals the trained artefact AND an independent join to 0.000000, on races and drivers the implementer never checked.

Executed: built actual `RaceStateManager`s for **33 (race, driver) combos** — rain races
(Zandvoort'23, Silverstone'24, São Paulo'24, Melbourne'25), red-flag races (Zandvoort'23,
Monaco'24, São Paulo'24), the shortest weather frame in the dataset (Las Vegas'23, 108
samples), evening races (Las Vegas'23/'25), and drivers ALO/VER/HAM/TSU/LEC/RUS/ZHO/SAI/…
— and compared every served lap against (a) `laps_tiredeg.parquet`'s stored trained values
and (b) my independent `searchsorted` join over the driver's raw laps. **Max |Δ| = 0.000000
on both, every lap, every combo** (≈6,700 artefact cells + ≈7,300 raw cells). The per-driver
alignment is correct because `weather_for_race` joins each driver's own lap `Time` — the
"built from one driver's laps and cached" concern does not bite: the cache is per-RSM, and
an RSM is per-driver by construction.

### V-C4 — VERIFIED: wind rides the same aligned rows (max |Δ| = 0.000000 vs an independent WindSpeed join, 3 races), and no consumer regressed.

- Arcade `WeatherPanel` (`src/arcade/overlays.py:125-137`) reads the same keys the dict
  still carries; no shape change. The arcade's OWN weather path
  (`src/arcade/data.py:518-567`, FastF1 sessions) already used `merge_asof(nearest)` — it is
  a third weather implementation, but it never had the proportional defect.
- The backend (`src/telemetry/backend/.../strategy.py:574-583`) never reads
  `get_weather_state` — its weather comes from the augmented featured frame, which was
  already N04-faithful. #815 changes nothing there.

### F-1 — MEDIUM. The fallback's docstring names the wrong trigger, the wrong semantics, and the wrong training analogy — and the substituted reading it denies is measurably worse than the old code exactly where it fires.

`src/simulation/race_state_manager.py:502-505` claims the fallback fires "when this lap has
no session time to join on", calls it "the row N04 also leaves unmatched", and "a real gap
rather than a substituted reading". All three are refuted by execution:

1. **The stated trigger never fires.** 0 of 79,032 raw driver-laps across all 71 races have
   NaT `Time`. A lap with NaT Time would NOT reach the fallback anyway: `weather_for_race`
   returns a frame indexed like the whole driver frame (NaN values, index present), so
   `row.index[0] in aligned.index` is True and the NaN row is served (honestly, as None).
2. **The real trigger is a driver with NO ROW for that lap** — retirees and lapped cars.
   6,236 (driver × missing-lap) pairs exist across the dataset; the replay loop runs to the
   RACE's total_laps, so a replay of a retired driver serves the fallback on every lap after
   retirement (SAR Zandvoort'23: laps 16-72, 57 laps).
3. **`weather_df.iloc[0]` IS a substituted reading — the session's FIRST sample.** Measured
   over 405 fallback laps (7 replays of retirees/lapped drivers): served-vs-actual TrackTemp
   mean |Δ| **2.67 C, max 8.9 C** (Monaco'24 OCO mean 4.96), and the rain flag wrong on
   **92 laps** (Melbourne'25 40, Silverstone'24 22, São Paulo'24 21). The OLD proportional
   lookup on those same laps was mean 1.86 C — the fix made exactly this territory WORSE.
4. **N04 does not "leave that row unmatched"** — N04 has no such row at all; there is no
   trained convention here, and the honest emission (consistent with the module's own NaN
   handling one branch earlier) would be None, not lap-1 weather.

Concrete failing scenario: `f1-sim` any retired driver at an evening race — every
post-retirement lap tells N06/N27 the track is at session-start temperature and start-line
rain state. Severity MEDIUM not HIGH because the affected laps are post-retirement, where
the replay's strategy value is already low — but the docstring is the exact wrong-mechanism
class `CLAUDE.md` §11 documents, sitting on the one branch of this fix that serves fiction.

### F-2 — MEDIUM (latent, mechanism EXECUTED). The `id(weather_df)` cache key collides across frames: reproduced on the second trial.

`race_state_manager.py:507-508` keys the alignment cache on `id(weather_df)` with no
eviction. `id()` of a dead object is legal to reuse, and CPython reuses it eagerly:
executed attack — read frame A, serve lap 10 (28.2 C), free A, read frame B with
TrackTemp+50, `id(B) == id(A)` on **trial 2** → `get_weather_state(10, B)` served **28.2**
where the correct answer is 78.2. Silent, no error, no log.

Today's callers cannot trigger it (`RaceReplayEngine` holds ONE frame alive for the race and
passes the same object every call; `bench_subagent_latency` likewise), so this is latent,
not live. But the promise `get_weather_state` makes is per-ARGUMENT, and the failure needs
only a caller that re-reads `weather.parquet` per call and lets the old frame die — the
natural shape of a backend request handler or the planned live path. In-place mutation of
the frame is likewise served stale (executed). A content-derived key, a `WeakValueDictionary`
keyed on the frame object, or documenting the same-object contract as an explicit precondition
at the call boundary would close it. Also noteworthy: the cache dict grows one entry per
distinct frame passed and never evicts (3 entries after 3 frames, measured).

---

## PR #816 — the tire serving frame

Harness: mirrored `run_from_state`'s session_meta and `_build_stint_features`' helper chain
on the race-scoped augmented `laps_featured_2025` frame, per (Driver, Stint), for the same
8 races GATE_DATA_WIRING used — **327 stints, 5,780 artefact-matched rows** — then diffed
every touched column against `laps_tiredeg.parquet` and ran the real TCN bundles.

### V-C7 / V-C8 — VERIFIED, including the halves the commit never measured.

- **Constants:** recovery `LapTime_s − lap_time_vs_cluster_mean` from `laps_tiredeg` gives
  nunique=1, std exactly 0.0 per cluster, matching all four hardcoded values to the last
  digit (`100.92462574340107 / 95.43860940701592 / 84.6488860957042 / 81.36461922886834`).
- **The all-years attack FAILS (good):** the same constant holds per (Cluster, Year) for
  2023, 2024 AND 2025 — std 0.0 in all 12 groups — and `LapTime_s − CONST[Cluster]`
  reproduces the stored column to max dev **0.0 over all 68,122 rows**, not only the
  22,760 the commit cited. Replaying a 2023/2024 race serves the right constant too.
- **Pooled family:** pooled map = artefact on **24/24** 2025 GPs (2025 map: **17/24**,
  disagreeing on Barcelona, Budapest, Melbourne, Shanghai, Spielberg, São Paulo,
  Zandvoort) — and the harder half holds: pooled = artefact on **22/22** 2023 GPs and
  **24/24** 2024 GPs. No GP's Cluster varies across years inside the artefact. N07 reading
  the COMBINED featured parquet verified at `.nb_py/N07_tiredeg_eda.py:92`.
- **Unknown cluster id:** the pooled map's values are all in {0,1,2,3}, so the
  `.get(cluster, 0.0)` sentinel is unreachable from the map; the reachable miss is
  `circuit_cluster_map.get(gp_name, 0)` → cluster 0 → 100.92 (see F-8).

### V-C5 / V-C6 — VERIFIED in the notebooks and end-to-end.

- N04 computes `DegradationRate[i]` over laps i-2..i (includes i) and `DegAcceleration[i]
  = deg[i] − deg[i-1]`, both unshifted, "No fillna — NaN … is meaningful signal"
  (`.nb_py/N04_feature_engineering.py:481-507`). N09's `PRODUCTION_FEATURES =
  BASE_FEATURES` consumes them as stored; no `shift` exists in N09/N10 feature code
  (grepped: only prose hits).
- N10 `fit_scaler`/`apply_scaler` do `fillna(0)` in RAW space before `StandardScaler`
  (`.nb_py/N10_tiredeg_compound_finetuning.py:171-181`); the serving path's single model
  entry (`_build_stint_tensor`, `tire_agent.py:1102`) does `feat_df.fillna(0)` before
  `scaler.transform` — **no NaN can reach the TCN un-zeroed**, and there is no second
  entry point that skips it (`_fresh_reference` routes through the same tensor builder).
- Frame parity on artefact rows, **exactly 0 mismatches**: `lap_time_vs_cluster_mean`
  (0/5,780), `LapsSincePitStop` (0), `Prev_LapTime`/`Prev_SpeedST`/`Prev_SpeedI1`
  (0 value + 0 NaN-pattern), `FuelAdjustedLapTime` (0), `LapTime_Delta` on rows ≥2 (0).
  W-F9, W-F10, W-F11 are genuinely closed on the replay serving path.
- Old-vs-new TCN movement: mean **0.399 s, p95 1.014, max 5.21** over the 327 stints —
  consistent with the commit's "mean of 0.42 s and up to 4.99 s".

### F-3 — HIGH. Serving still skips N04's ±2.0 CLIP on `DegradationRate`/`DegAcceleration`: the TCN eats out-of-training-range values exactly at cliff/chaos laps.

`clip_degradation_outliers(laps_clean, clip_range=(-2.0, 2.0))`
(`.nb_py/N04_feature_engineering.py:534-554`, applied to 2025 too at `:1063`) clips BOTH
columns immediately after computing them, so **training never saw |value| > 2.0**. The
serving `_add_degradation_rate` (`src/agents/tire_agent.py:675-715`) has no clip. Measured:
**42 DegradationRate rows (max served 8.81 where the artefact says ±2.0) and 25
DegAcceleration rows (max 8.31)** across the 327 stints, concentrated where the number
matters most — Melbourne 2025 laps 43-46 (the rain chaos), Silverstone 2025 ALO laps 40-44.
TCN attribution (truth frame + serving deg column only): mean 0.062 s, **max 4.26 s** on a
single stint. The commit's frame — "N04 computes both unshifted and N09 consumed them as
stored" — is true of the SHIFT and silent about the PIPELINE: N04 computes, then clips.
W-F8's own fix spec in GATE_DATA_WIRING missed the clip too; the acceptance criterion it
set ("divergence vs the trained artefact is 0 on the deg pair") is not met, and the commit
does not say so. Fix shape: `.clip(-2.0, 2.0)` on both columns at `tire_agent.py:713-714`.

### F-4 — HIGH. `DegAcceleration` at stint position 1 serves `deg[1]` where training saw 0 — on 326/327 stints, and the OLD code was accidentally right there.

N04's accel guard requires `deg_rates[i-1]` non-NaN; `deg[0]` is ALWAYS NaN, so the trained
`DegAcceleration` at every stint's second row is NaN → scaler-filled raw 0. The serving loop
(`tire_agent.py:700-711`) initialises `raw_deg`/`raw_accel` with `np.zeros` (N04 uses
`np.full(n, nan)`) and computes `raw_accel[1] = raw_deg[1] − 0 = deg[1]` — a real, generally
non-zero slope. Measured: **326/327 stints diverge at position 1, max |Δ| 10.81**; TCN
attribution (truth + serving accel column) mean **0.094 s, p95 0.336, max 3.87** — the
single largest surviving contributor. The pre-PR `.shift(1).fillna(0)` produced 0 at
position 1, matching training BY ACCIDENT; the fix moved the wrongness from every position
≥2 to position 1 only (a large net win) but the commit claims the trained frame wholesale.
Same zeros-vs-NaN asymmetry: serving `raw_deg[j] = 0` where a window contains a NaN lap
(N04 leaves NaN → 0 after fill — equivalent) but the accel NEIGHBOURS of such a row compute
against that 0 while N04's guard yields NaN → 0 (the 25 value rows of F-3 overlap here).
Fix shape: initialise both arrays `np.full(n, np.nan)` and replicate N04's two guards
verbatim, then clip (F-3); the scaler's `fillna(0)` already handles the rest.

**Net effect of F-3 + F-4 + the known F12 residue, measured:** TCN |serving − trained-truth|
over the 327 stints is **mean 0.148 s, p95 0.454, max 4.93** — against the fix list's
promised "~0.02 s (the F12 residue)". Attribution: accel 0.094 + deg 0.062 + F12
mean_sector_speed 0.022 (F12 alone measures exactly what GATE_DATA_WIRING predicted).
Roughly **a third of the pre-fix 0.42 s error survives the fix**, and no shipped test can
see it (the deg-pair tests are hermetic toy frames; nothing diffs the pair against the
artefact the way the ltvcm test does).

### F-5 — HIGH. The THIRD `session_meta` builder was missed — and the PR's unconditional recompute makes that path WORSE than before it.

The commit: "`cluster_mean_lap_s` was this race's own mean lap time in both session_meta
builders." There are three. `src/telemetry/backend/api/v1/endpoints/strategy.py:1017`
(`predict_tire_range`, the Tyres agent-tab chart) still builds
`"cluster_mean_lap_s": float(clean_times.mean())` — the race's own mean — and feeds it to
`agent._build_stint_tensor` → `_add_session_cols`, whose line 823 now recomputes
`lap_time_vs_cluster_mean = LapTime_s − session_meta["cluster_mean_lap_s"]`
**unconditionally**. Before the PR the guard kept the frame's stored artefact column on
this path; after it, the un-fixed builder's race mean lands in the feature. Measured per
2025 GP: |race clean-mean − trained constant| averages **6.14 s and peaks at 14.56 s
(Lusail), 13.98 (Imola), 11.93 (Monza)** — versus the pre-PR family-mix deviation of mean
5.75 s. The comment above that builder even says it fills session_meta "exactly as
run_from_state's setup does" — it did, until the PR changed run_from_state and not the
mirror. This is the repo's dominant defect (one copy fixed, its twin not), inside the very
PR whose message cites #800's twin as the lesson. NOTE: `src/telemetry` is a submodule —
the fix belongs there (change the builder to the trained constant, or import it), plus a
grep for any fourth copy at bump time.

### F-6 — MEDIUM. Three wrong-pointer/stale-frame comments shipped or survived in the exact places future readers will act on them.

1. `tire_agent.py:802-815` (`_add_session_cols` docstring): still says
   `lap_time_vs_cluster_mean` is "guarded like FuelLoad … recomputed only when the frame
   does not already carry them" and argues recomputing "overwrites the trained constant
   with a per-frame quantity" — the PR made the recompute UNCONDITIONAL four lines below
   (:818-823) precisely because session_meta now carries the trained constant. The stale
   paragraph argues FOR reverting the fix; someone following it would reintroduce W-F9.
2. `tire_agent.py:328-329`: "`tests/agents/test_tire_cluster_mean.py` re-derives them and
   fails if they drift" — **that file does not exist**; the re-derivation lives in
   `tests/agents/test_tire_serving_frame.py::test_the_cluster_means_are_the_ones_n04_subtracted`.
   An auditor checking the named guard finds nothing; a cleanup deleting "orphan" tests
   would not know the guard lives elsewhere.
3. `tire_agent.py:1078` (`_build_stint_tensor` docstring): "Short stints are left-padded by
   repeating the first row" — the body zero-pads and its own inline comment (:1107-1109)
   says the repeat-tiling is what it REPLACED. Pre-existing (flagged in GATE_DATA_WIRING's
   N26 notes), survived a PR that edited this same file.

### F-7 — MEDIUM. The W-F11 regression test is VACUOUS: it asserts about a function the fix does not live in — executed proof.

`tests/agents/test_tire_serving_frame.py:162-190`
(`test_an_existing_laps_since_pit_is_not_overwritten_by_tyre_life`) calls
`_add_session_cols` — but the alias fix is in `_add_timing_cols` (`tire_agent.py:623-624`).
`_add_session_cols` never touches `LapsSincePitStop`, so the assertion holds under ANY
behaviour of the fixed code. Executed: simulated the revert (restore the unconditional
alias in the timing step) — the shipped test's assertion still passes (True), while the
real serving chain would serve `[12, 13]` (TyreLife) instead of the trained `[3, 4]`.
This is the `feedback_a_guard_that_asserts_nothing` class verbatim: W-F11 currently has NO
effective regression guard. Fix: point the test at `_add_timing_cols` (or the full chain).

### F-8 — LOW. The measurement in the commit and the measurement in the docstring are two different numbers for the same defect, and the commit's counts NaN as disagreement.

Commit: "the alias disagreed with it on 22.4% of 2025 rows". Reproduced EXACTLY — but only
when rows where `TyreLife` is NaN (451 of 22,760) are counted as disagreements
(`NaN != x → True`). On comparable rows the artefact disagreement is **20.8%**; the
docstring at `tire_agent.py:599` says **15.8%**, which is the 335-stint harness number from
GATE_DATA_WIRING quoted without naming its frame. None of the three is wrong about the
defect's existence; the lesson-file point stands: check WHICH pair of things each number
compares before quoting any of them.

### F-10 — MEDIUM (pre-existing, unchanged, now the LAST of its family). `LapTime_Trend` at stint position 1 serves `Delta[1]` where training saw 0 — the un-fixed sibling of the exact NaN-convention defect this PR fixed for `Prev_*`.

`tire_agent.py:670-671`: `LapTime_Delta = (LapTime_s − Prev_LapTime).fillna(0)` and
`LapTime_Trend = (Delta − Delta.shift(1)).fillna(0)`. The artefact stores NaN at stint rows
0-1 for these (measured: 327/654 NaN-pattern mismatches, 0 value mismatches on rows ≥ 2);
after the scaler's fill, row 0 is equivalent (0 vs 0) but row 1's Trend serves
`Delta[1] − 0 = Delta[1]` where the trained input is 0 — **327/327 stints**, TCN
attribution mean **0.018 s, max 1.55 s**. Both the old and new `Prev_*` conventions produce
the same result here (the `fillna(0)` on Delta masks the difference), so this is NOT a
regression of #816 — it is the third member of the family W-F10 fixed for `Prev_*` and F-4
documents for `DegAcceleration`, and it survives because the intermediate `fillna(0)`
manufactures a real 0 one step before the scaler would have manufactured the trained 0 from
NaN. Fix travels with F-4: drop the intermediate `fillna(0)`s and let the tensor-builder's
single `fillna(0)` be the only missing-value authority, exactly as N10 had it.

### N-1 — Note (no action forced): the committed weather tests guard the RULE, not the shared implementation, and skip the exact branch F-1 lives on.

`tests/simulation/test_weather_join.py` asserts served == `weather_for_race` cell-for-cell
plus an anti-revert case proving the old rule differs on Shanghai — well built (effect
assertions, vacuity guards). Two honest limits: (1) the serving path CALLS
`weather_for_race`, so the comparison is semi-circular — a future bug inside
`weather_for_race` passes both sides, and the one test that anchors it to N04's actual
output (`test_weather_restore`) is currently red for artefact reasons (GATE_801 D1). This
gate anchored serving to `laps_tiredeg`'s stored weather independently (V-C1: 0.000000);
once the featured artefacts are regenerated (#801) that anchor test goes green again and
the chain closes. (2) both loops `continue` on `row.empty` — the `iloc[0]` fallback branch
(F-1) has zero test coverage.

### F-9 — LOW (scoped to a known-defective dir). A gp_name that misses the pooled map now also poisons `lap_time_vs_cluster_mean`, not just `cluster_id`.

Of all 71 race dirs, exactly one metadata gp_name misses the pooled map: `2023/Spain` — the
byte-identical Barcelona duplicate GATE_801 C1 documented. A replay of it resolves cluster
`.get('Spain', 0)` → 0 → constant **100.92** where Barcelona's cluster is 3 (**81.36**):
every lap's `lap_time_vs_cluster_mean` shifts by **−19.6 s**. Pre-PR, the guard kept the
frame's stored (plausible-magnitude) value, so the keyspace miss was invisible in this
column; the unconditional recompute amplifies it. Fix-list item 3 (`_normalise_gp_key`)
and the duplicate's deletion both cover it; recorded so the amplification is on paper.

---

## What I tried to break and could NOT

- **The weather join itself.** Rain races, red-flag races, the 108-sample Las Vegas frame,
  seven drivers who are not NOR, three seasons: served == trained artefact == an
  independent from-scratch join, to 0.000000, on every lap that has a row. The alignment
  is per-driver-correct by construction (each RSM joins its own driver's lap times).
- **The commit's headline numbers.** All five of #815's (79,032 / 94.3% / 1.488 / 17.3 /
  3,399) reproduce exactly; #816's movement claim (0.42 / 4.99) reproduces as 0.399 / 5.21
  on my montage — consistent.
- **The four cluster constants, on any axis I could think of.** Per cluster, per year, per
  row (68,122/68,122 at 0.0), against both maps, on both the "did N07 read combined"
  question (verified at `.nb_py/N07_tiredeg_eda.py:92`) and the 24/24-vs-17/24 counts
  (exact). The 2025-only worry is empirically dead: the constants are global.
- **W-F9 / W-F10 / W-F11 on the replay serving path.** 0 mismatches over 5,780
  artefact-matched rows for ltvcm, LapsSincePitStop, and every Prev_* column (values and
  NaN pattern both).
- **The scaler/NaN pipeline.** Single entry point, `fillna(0)` in raw space before
  transform, matching N10:176/181; `_fresh_reference` routes through it; no path lets NaN
  reach the model.
- **The pooled-map keyspace for real replays.** 70 of 71 race-dir gp_names resolve
  (including the accented ones — NFC handling holds); the one miss is the known duplicate.
- **The wind alignment.** 0.000000 vs an independent join on 3 races; no consumer reads a
  key that moved.
- **Vacuity of the weather tests.** The anti-revert case genuinely differs on Shanghai and
  the compared>0 guards hold; their limits (semi-circularity, fallback branch uncovered)
  are noted in N-1, but the tests do assert real cells.

## Ordered fix list (by value ÷ risk)

1. **F-3 + F-4 (one change, `tire_agent.py:675-715`):** initialise `raw_deg`/`raw_accel`
   as `np.full(n, np.nan)`, replicate N04's two validity guards, then `.clip(-2.0, 2.0)`
   both columns (N04:534-554). Add a data-tier test that diffs BOTH columns against
   `laps_tiredeg` rows the way the ltvcm test does — that is the test shape that caught
   nothing here because it does not exist for the deg pair. Expected: TCN residual mean
   0.148 → ~0.03 (F12 + F-10 remainder).
2. **F-5 (submodule):** `strategy.py:1017` must serve the trained constant (import
   `TireAgentConfig._TRAINED_CLUSTER_MEAN_LAP_S` + the pooled map like the two fixed
   builders). Until it lands, the Tyres-tab TCN chart runs on ltvcm shifted mean 6.14 s /
   max 14.56 s from trained.
3. **F-7 (`tests/agents/test_tire_serving_frame.py:162-190`):** point the W-F11 test at
   `_add_timing_cols` (or the chain); today the fix has no guard.
4. **F-1 (`race_state_manager.py:494-533`):** make the missing-row fallback serve the
   nearest-by-session-time sample for the LAP (leader's lap time) or an honest all-None
   row, and rewrite the docstring to name the real trigger (driver has no row). Add a
   fallback-branch test (the committed ones skip it).
5. **F-6 + F-8 doc sweeps (`tire_agent.py:802-815`, `:328`, `:1078`, `:599`):** three
   stale/wrong-pointer comments and one unlabelled-frame number; each is the documented
   revert-bait class.
6. **F-2 (`race_state_manager.py:507`):** replace `id()` keying (content hash of the
   Time column + shape, or WeakKeyDictionary) or write the same-object precondition into
   `get_weather_state`'s docstring and a test. Latent today; silent when it fires.
7. **F-10 (`tire_agent.py:670-671`):** drop the intermediate `fillna(0)`s so the tensor
   builder's single fill is the only missing-value authority (travels with fix 1).
8. **F-9:** falls out of fix-list item 3 (`_normalise_gp_key`) + deleting the Spain dup.
