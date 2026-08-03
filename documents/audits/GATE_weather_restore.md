# ADVERSARIAL GATE — #782 weather-column restore (`fix/restore-2025-weather-columns`)

Date: 2026-08-03 · Auditor: adversarial correctness gate (Fable) · Status: IN PROGRESS

Scope: uncommitted change restoring `AirTemp`/`TrackTemp`/`Humidity`/`Rainfall` onto
`laps_featured_2025` at load time via `src/f1_strat_manager/weather_restore.py`, wired into
`augment_featured_laps`, with `pace_holdout.py` rerouted through the augment.

Method: executed evidence only. Every finding cites file:line and measured values.

## Checklist

- [ ] A. Independent 2023/2024 reproduction (values + Rainfall + dtypes)
- [ ] B. Guard behaviour: early exit, partial columns, empty frame, all-NaN columns
- [ ] C. Pace MAE reproduction + sensitivity of MAE to the weather values
- [ ] D. What ELSE changed in the pace holdout (stint repair, row counts, _DROPNA)
- [ ] E. Direct-reader inventory: which eval paths still see un-repaired / weather-less data
- [ ] F. tests/eval/ run + other headline numbers
- [ ] G. Docstring/comment truth (the 71-races sentence, the "honest gap" claim, the "tests/agents holds that check" claim)

## Findings (appended as confirmed)

### G-OK — the 71-races docstring sentence is TRUE (verified, all four parts)

`weather_restore.py:9-10` claims every one of the 71 shipped race directories has a readable
`weather.parquet` with all four readings and zero NaN temperature rows. Executed over
`data/raw/{2023,2024,2025}/*` (23+24+24 = 71 dirs): 0 missing, 0 unreadable, 0 missing columns,
0 NaN `AirTemp`/`TrackTemp` rows. Also measured beyond the claim: 0 NaN `Humidity`, 0 NaT `Time`
samples, `Time` is `timedelta64[ns]` in all 71, `Rainfall` is `bool` in all 71.

### A-OK — 2023/2024 reproduction re-derived independently: EXACT, all FOUR columns, dtypes included

Re-derived (own script, not the author's): dropped the four weather columns from the published
`laps_featured_2023/2024.parquet`, ran `augment_featured_laps`, compared per (GP_Name, Driver,
LapNumber) with row order verified preserved.

- 2023: 22,106 rows. AirTemp/TrackTemp/Humidity exact-equal 22,106/22,106 each (max diff 0.0,
  0 NaN mismatches). The author's "66,318/66,318" is 22,106 × 3 — consistent.
- 2024: 23,256 rows. Exact-equal 23,256/23,256 each. "69,768" = 23,256 × 3 — consistent.
- **Rainfall (the column the author's script skipped): reproduces 100% in both seasons**, values
  {0,1} identical, and the restored dtype is `int32` — identical to the published `int32`
  (`.astype(int)` on Windows numpy = int32, same platform N04 ran on).
- Row count and key order preserved through the augment (no merge fan-out).

### FINDING 1 (MEDIUM) — B: a frame carrying SOME of the four columns comes out silently corrupted

`laps_augment.py:206` guards with `not all(column in df.columns ...)`, so a frame with 1-3 of the
four present sets `wants_weather=True`; the per-race slice then carries all four
(`laps_augment.py:236-237`) and the left-merge at `laps_augment.py:253` collides with the columns
already on the featured frame. Executed (2023 featured with only `AirTemp` dropped): the output has
58 columns, `TrackTemp`/`Humidity`/`Rainfall` NO LONGER EXIST — replaced by `TrackTemp_x`/
`TrackTemp_y`, `Humidity_x`/`Humidity_y`, `Rainfall_x`/`Rainfall_y`. `normalise_rainfall` then
no-ops (no `Rainfall` column), leaving `Rainfall_y` un-filled. Any consumer selecting the plain
names (e.g. `df[features_delta]` in `pace_holdout.py:120`) raises KeyError; a `.get()`-style
consumer silently reads nothing. Failure scenario: any future artefact republished with a partial
weather set — the guard's own condition contemplates exactly this shape and then corrupts instead
of deriving only the missing columns or refusing loudly. Today no shipped artefact is partial
(2023/24 carry all four, 2025 carries none), so this is latent, not live.

### B-OK — the early exit and the degenerate frames behave

- Published 2023 through the augment: all four weather columns byte-identical after, dtypes
  preserved (`int32` Rainfall included) — 2023/2024 are never re-derived. Confirmed executed.
- No-`GP_Name` frame returns unchanged; zero-row frame with keys returns zero rows, no crash.
- Columns present but ALL-NaN: guard sees presence, not usability — frame stays all-NaN (no
  re-derivation). Consistent with the "never second-guess a season that carries the columns"
  contract; noted as a design edge, not a defect.

### FINDING 2 (MEDIUM) — C: the pace-MAE reproduction is REAL evidence for the values but CANNOT see the alignment method; and the safeguard the docstring points to does not exist

Executed sensitivity of `_pace_mae()` (shipped MAE 0.40968, n=21,247; author's 0.40968 confirmed;
golden tolerance ±0.01 vs published 0.4104):

| weather variant | MAE | |Δ| vs shipped | golden verdict |
|---|---|---|---|
| shipped (nearest join) | 0.40968 | — | reproduced |
| all-NaN weather | 0.67217 | 0.26249 | delta |
| +5/+10/+20 °C/% perturbation | 0.57586 | 0.16619 | delta |
| all-zero weather | 0.51974 | 0.11006 | delta |
| scrambled across laps | 0.48805 | 0.07837 | delta |
| **WRONG join: direction='backward'** | **0.40939** | **0.00029** | **reproduced** |

The weather columns are 4 of the 25 features in `xgb_laptime_delta_feature_names.json` (verified),
and the MAE moves violently when their values are wrong at the distribution level — so the
reproduction is NOT hollow. But a backward join that changes 7,014/22,760 TrackTemp cells moves
the MAE by 0.00029 and still "reproduces". Two documented claims therefore overstate it:

- `tests/agents/test_weather_restore.py:9-10`: "That number moves if the merge is wrong, which is
  what makes it evidence rather than decoration" — REFUTED for alignment-level wrongness. Only a
  synthetic unit test (`test_each_lap_takes_its_nearest_sample_not_the_preceding_one`) pins the
  direction, on a 1-lap toy frame.
- `weather_restore.py:28-31`: "the restore must reproduce 2023's and 2024's published weather
  **exactly** before it is trusted on 2025. `tests/agents/` holds that check, and it is not
  decoration" — **FALSE. No committed test compares the restore against the published 2023/2024
  values.** All 6 tests in `tests/agents/test_weather_restore.py` run on synthetic 1-3 row frames.
  The exact-reproduction evidence exists only in this gate and in the author's uncommitted script.
  The one check the module calls its load-bearing safeguard is not in the repo.

### D-OK (measured) — the stint repair riding along moves the metric by 0.00011 and the denominator not at all

The rerouted holdout input changed in two ways (#782 weather + #790 repair). Decomposed, executed:

- Repair disabled (monkeypatched no-op), weather on: MAE 0.40957 vs shipped 0.40968 —
  |Δ| = 0.00011. The 0.4104 reproduction survives because of the weather, not despite the repair.
- Denominator: n = 21,247 in EVERY variant. `_DROPNA = [LapTime_Delta, Prev_LapTime]`
  (`pace_holdout.py:44`) are published columns the repair never touches; the 113 newly-nulled
  TyreLife values (featured 2025 rows) do NOT drop rows because TyreLife is not in `_DROPNA`.
- Repair's real footprint on the scored frame: 98 predictions moved (max |move| 0.611 s), n
  unchanged, net MAE effect 0.00011. Stint moved on 30 featured rows, TyreLife on 406 (113 to
  null).

### FINDING 3 (HIGH-VALUE CONFIRMATION, not a defect) — the 2025 ground truth EXISTS on disk and the restore matches it byte-for-byte; no committed test uses it

`data/processed/laps_featured.parquet` (the combined 2023-2025 artefact, 68,122 rows) ALREADY
carries all four weather columns for 2025 with zero NaN — the actual N04 output for 2025 was
published all along in the combined file; only the per-year split dropped it. Executed: restored
2025 weather vs the combined file's 2025 slice — 21,903/21,903 key-matched rows identical on all
four columns (max diff 0.0), plus the remaining 857 Miami rows identical after mapping the
GP_Name rename (`Miami` per-year vs `Miami Gardens` combined). **22,760/22,760 exact.**

Consequences:
- This retroactively closes the backward-join blind spot from Finding 2 FOR THIS GATE: a backward
  join changes 7,014 TrackTemp cells and would have failed this comparison; 'nearest' is therefore
  verified as N04's actual 2025 method, not just its stated one.
- But the repo keeps none of this: the strongest available check (compare restore against
  `laps_featured.parquet`'s 2025 slice) is in NO committed test, and `weather_restore.py:9-12`
  frames the truth as existing only in `weather.parquet` inputs ("What is missing is only the
  merge") when the merge RESULT is also published. The committed test the docstring promises
  (Finding 2) could be written against this file in ~15 lines and would pin everything the MAE
  cannot see. Recommendation 1.

### FINDING 4 (LOW) — three comments/docstrings name the WRONG MECHANISM for the Rainfall fill, and one asserts the opposite of what the code does

Executed proof: simulated Monza 2025 with no readable weather parquet (`read_race_weather`
returning None for that race only). Result: Monza's 895 laps keep AirTemp/TrackTemp/Humidity NaN
(honest gap) but **Rainfall reads a confident 0 (dry) on all 895** — because the season-level
`normalise_rainfall` at `laps_augment.py:258` fills every remaining NaN.

- `laps_augment.py:256-257`: "applied once per season rather than per race so a race with no
  weather parquet keeps an honest gap instead of reading 0 (dry)" — FALSE for Rainfall, the only
  column the sentence is about. The season pass is precisely what writes the 0.
- `weather_restore.py:93-95` (normalise_rainfall docstring): "a per-race fill would leave a race
  with no weather parquet reading 0 (dry) instead of keeping the honest gap until the season-level
  pass" — INVERTED. A per-race fill inside `weather_for_race` would never RUN for a race whose
  parquet is absent (the function is never called), so it would KEEP the gap; the season-level
  pass is what destroys it.
- `tests/agents/test_weather_restore.py:99-103` repeats the inverted claim verbatim, and the test
  under it only checks that a frame without a Rainfall column passes through untouched — it does
  not (and cannot) test the claimed honesty property.

The CODE is faithful to N04 (N04 cell 22 `continue`s on empty weather and fills Rainfall once over
the master frame — same observable behaviour), so nothing to fix in behaviour. But per this
project's own lesson log, a comment naming the wrong mechanism is worse than none: someone
"fixing" the code to match these comments would break the N04 reproduction.

Footnote (comment-truth, no practical surface): `weather_restore.py:59` says NaT-Time laps "keep
NaN, which is what N04 leaves them as" — N04 never left them as anything; its merge_asof would
raise on an unsorted NaT key. Measured: 0 NaT `Time` in all 79,032 raw laps across the 71 races,
so the divergence is unreachable in shipped data.

### E — direct-reader inventory: the featured-laps inconsistency is GONE, not created

Verified per reader (file:line, artefact read):
- `calibration.py:160/233/283` — `overtake_labeled/overtake_pairs_2023_2025.parquet`,
  `sc_labeled/sc_labeled_2023_2025.parquet`, `undercut_labeled/undercut_clean.parquet`: labeled
  artefacts, not featured laps. Reproduction REQUIRES them as-published (as-trained); routing them
  through the augment would be wrong.
- `tire_holdout.py:171` — `laps_tiredeg.parquet` (56 cols): carries its own weather for ALL years,
  0 NaN in all four columns 2023/2024/2025 (measured). Not affected by the 2025 hole.
- `nlp.py:643` — RCM corpus parquets, unrelated.
- `pit_holdout.py:54`, `projection.py:296`, `stint_lengths.py:207` — RAW `laps.parquet` per race,
  not featured. (They see un-REPAIRED raw stints, but that pre-dates this diff and is untouched
  by it; the pit/projection metrics were published on those raws.)
- Non-eval consumers of `laps_featured_<year>`: arcade `strategy.py:576`, backend
  `laps_cache.py:39` (feeding `simulator.py` and every route), CLI — all already through
  `augment_featured_laps`; `pace_agent.py:216-218/240-243` reads the parquet directly but only
  columns `[Team, TeamID]` / `[GP_Name, Year, Compound, LapTime_s]` — no weather dependency.

Net: after this change, every reader of a per-year featured-laps parquet that consumes more than
named non-weather columns goes through the augment. The change REMOVED the last inconsistency
(pace_holdout) rather than adding one. Side effect worth knowing: arcade/backend/CLI 2025 frames
now carry 6 extra columns (Time_s, TrackStatus + 4 weather) at runtime — additive only.

### Premise check — the bug being fixed is real

Executed the pre-change path (direct `pd.read_parquet` of `laps_featured_2025.parquet` + the two
N06 feature steps + `df[features_delta]`):
`KeyError: "['AirTemp', 'TrackTemp', 'Humidity', 'Rainfall'] not in index"` — exactly as claimed.
And `reproduce.collect_results()` (`reproduce.py:263-272`) calls `_pace_mae()` with no per-model
exception isolation, which is why `test_reproduction_matches_overtake_auc_pr` (which goes through
`collect_results`) failed alongside the pace golden: the overtake reproduction itself reads
`overtake_pairs_2023_2025.parquet` and never needed featured-laps weather. One fix, both tests —
the wiring claim holds.

### Hardening footnote (not a finding — unreachable in shipped data)

`weather_for_race` (`weather_restore.py:75`) would raise `pandas.errors.MergeError` out of the
whole augment if a race's `weather.parquet` carried a non-timedelta `Time` (the
`read_race_weather` try/except at `weather_restore.py:109-115` guards the READ, not the merge).
Measured: all 71 shipped weather parquets carry `timedelta64[ns]`, so no current data can trigger
it. Worth a dtype guard only if the artefact contract ever loosens.

## What I tried to break and could NOT

1. **The 2023/2024 reproduction** — re-derived with my own script, all four columns, both seasons,
   45,362 rows: byte-exact, including the Rainfall int32 the author never checked.
2. **The 2025 restore against its published truth** — found ground truth the author did not use
   (`laps_featured.parquet` 2025 slice) and the restore matched 22,760/22,760 rows on all four
   columns, Miami rename included.
3. **The early exit** — published 2023 through the augment comes back with weather byte-identical.
4. **The denominator** — n = 21,247 scored rows in every variant; `_DROPNA` drops nothing new.
5. **The stint-repair contamination theory** — measured at 0.00011 MAE; the reproduction stands on
   the weather, not on a cancellation.
6. **Degenerate frames** — empty, keys-only, all-NaN-weather frames all pass through sanely.
7. **The 71-races docstring sentence** — all four parts true, plus stronger properties (0 NaN
   Humidity, 0 NaT Time, uniform dtypes) it does not even claim.
8. **The Miami alias path** — the renamed race (also the #790-repaired race) restores weather
   correctly through `FOLDER_ALIASES` (857/857 vs truth).
9. **Lint/format** — `ruff check` and `ruff format --check` pass on all four files.

## Fix list (by value, none blocks shipping)

1. **Commit the reproduction test the docstring already claims exists** (`weather_restore.py:28-31`):
   compare the restore against `laps_featured.parquet`'s 2025 slice (and/or the 2023/2024 per-year
   artefacts) under `@pytest.mark.data`. ~15 lines; it pins the alignment method the MAE provably
   cannot see (backward join reproduces to 0.00029). Until then, soften the docstring.
2. **Fix the guard for partial weather columns** (`laps_augment.py:206`): either derive only the
   missing columns, or raise/log loudly on a partial set instead of emitting `_x`/`_y` corruption.
3. **Rewrite the three inverted "honest gap" comments** (`laps_augment.py:256-257`,
   `weather_restore.py:93-95`, `tests/agents/test_weather_restore.py:99-103`) to say what is true:
   temps keep NaN for a weatherless race, Rainfall becomes 0 because N04's season-level fill does
   that, and the restore keeps N04's behaviour on purpose.
4. (Optional) dtype guard on the weather `Time` column before `merge_asof`.

## Verdict
