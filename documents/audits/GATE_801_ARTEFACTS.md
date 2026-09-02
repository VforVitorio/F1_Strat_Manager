# GATE — #801: the featured artefacts themselves, and the plan to regenerate them

**Date:** 2026-08-04 · **Ref:** `main`/`dev` @ `73788e0b` · **Gate role:** adversarial, read-only on the repo (this file is the only write; `data/` untouched).

**Mandate:** (1) verify the three claims in #801 as CLAIMS, not facts; (2) sweep the three featured
artefacts against each other and the raw laps for everything #801 did NOT report; (3) settle whether
`tests/agents/test_weather_restore.py`'s combined-carries-weather claim was ever true; (4) wire the
regeneration plan so an implementer only executes.

**Not re-reported here** (GATE_DATA_WIRING owns them): F7 weather proportional lookup, F8 deg shift,
F9 `lap_time_vs_cluster_mean` artefact split, F10 prev-fill, F14 keyspace family. This gate judges the
ARTEFACTS and the PRODUCER (`.nb_py/N04_feature_engineering.py`), not the consumers.

Findings appended AS CONFIRMED, each with executed evidence.

---

## Audit checklist (updated as worked)

- [x] Claim A — combined broadcasts one season's `mean_sector_speed` → A1 (verified, refined)
- [x] Claim B — Las Vegas 2025 NaN hole + sweep for unreported holes → B1, B2
- [x] Claim C — 2023 Spain/Barcelona duplicate + double-count measurement → C1, C2, C3, C3-b
- [x] Beyond: column-by-column artefact diff (combined vs per-year vs raw) → A2, "could NOT" §1-3
- [x] The weather-restore test's claim: true against which artefact version? → D1 (never true of the published artefact)
- [x] Producer logic audit: is regeneration even sufficient? → A2, A3, A4, §2 (NO — producer fixes first), E1 (proof the fixed rebuild reproduces)
- [x] Part 2 plan: commands, gates, HF upload, ordering vs GATE_DATA_WIRING → runbook §§1-6 + PR table

---

## Findings

### A1 — Claim A VERIFIED, with a sharper mechanism than #801 states: the combined artefact is not "stale for 2025" — it was built with a DIFFERENT rule, and so were the 2023/2024 per-year files. Only `laps_featured_2025.parquet` carries season-true values, and it is exact.

Measured (all four artefacts + both circuit-feature files + all 71 raw `laps.parquet`, N03's rule
= mean of SpeedI1/I2/FL over laps carrying all three):

- **Combined**: `mean_sector_speed` identical 2023-vs-2025 on **21 of 22** shared GPs (the 22nd is
  Miami, explained below). Every combined value equals `circuit_features_with_clusters_k4.parquet`
  exactly (70 of 70 (yr,GP) pairs with a k4 entry, 1 apparent mismatch that is the Miami alias
  row). The combined's number is N03's **2023+2024 pooled** per-circuit constant, merged on
  GP_Name with no Year key (`.nb_py/N04_feature_engineering.py:752-757`).
- **`laps_featured_2023/2024`**: byte-identical to the combined's rows for those years
  (cell-level diff over ALL 48 columns: **zero differing cells** in 2023 and 2024). They carry the
  same pooled constant — NOT their own season's measurement (vs raw truth: 2023 mean |Δ| 2.98,
  max 9.10 km/h at Suzuka; 2024 mean 4.49, max 15.79 at São Paulo).
- **`laps_featured_2025`**: matches `circuit_features_with_clusters_k4_2025.parquet` exactly on
  23 of 24 GPs (24th = Las Vegas NaN), and matches the raw-derived 2025 truth to **0.00 km/h mean
  and max** — it IS the season measurement. #801's spot values reproduce exactly: Silverstone
  raw 231.36* / per-year 231.36 / combined 249.71; Melbourne 256.84 / 256.84 / 272.44.
  (*#801 quoted "raw 232.32" for Silverstone from the mean-of-three-traps rule including a
  different lap filter; under N03's own filter the raw value is 231.36 — i.e. the per-year
  artefact is not "close to" raw, it EQUALS it. The conclusion of #801 stands, stronger.)
- **2025 combined-vs-per-year: mean |Δ| 4.77, max 18.35 km/h (Silverstone), 1 of 24 identical
  (Miami — see A4).**

**Authority verdict:** for 2025, `laps_featured_2025.parquet` is authoritative (season-true and
exactly reproducible from raw via N03's `_compute_sector_speed`). For 2023/2024 the artefacts
carry the pooled-training-era constant BY CONSTRUCTION, and that pooled constant is what N06
trained on — so the 2023/2024 values are authoritative *as the trained quantity* and must NOT be
"fixed" to season-true without retraining. `pace_agent._load_circuit_mean_sector_speed`'s
docstring (`src/agents/pace_agent.py:336-378`) already documents this correctly; this gate
confirms every number in it.

### A2 — The producer CANNOT reproduce the shipped artefacts: current N04 would REGRESS `laps_featured_2025.parquet` on 3 columns and 6,892 Cluster cells. Regeneration with the code as-is makes things WORSE, not better.

- The shipped `laps_featured_2025.parquet` was built by a **pre-2026-02-15 version** of N04:
  its `Cluster` matches `circuit_clusters_k4_2025.parquet` (KMeans-predict assignments) on 24/24
  GPs and MISmatches `circuit_clusters_k4.parquet` on 7 GPs (Barcelona 1→3, Budapest 1→3,
  Melbourne 0→2, Shanghai 1→0, Spielberg 1→3, São Paulo 0→2, Zandvoort 0→2); its
  `mean_sector_speed` matches the `_2025` circuit features exactly (A1).
- Commit `11a7ffa` (2026-02-17, "changed 2025 clusters to be the same of 2023-24 instead of
  using predict method") rewired Step 9 to read `circuit_clusters_k4.parquet` +
  `circuit_features_with_clusters_k4.parquet` (`.nb_py/N04_feature_engineering.py:1041,1072-1076`).
  **The artefacts were never regenerated after that commit.** Running today's N04 would flip
  6,892 Cluster cells, replace every 2025 `mean_sector_speed` with the 2023-24 pooled constant
  (mean shift 4.77 km/h, max 18.35), and recompute `lap_time_vs_cluster_mean` on all 22,760
  rows — destroying the exact values the #797 fix, the N06 holdout MAE (0.4104), and
  `test_pace_circuit_speed.py` were measured against.
- The Step 9 header comment (`N04:999`) still says "Cluster assignment from saved model
  (`circuit_clusters_k4_2025.parquet`)" while the code reads the k4 file — a comment naming the
  wrong mechanism, in the producer, at the exact line that decides this regression.
- Cell-level diff, combined vs per-year 2025 (same (GP,Driver,LapNumber), Miami aliased):
  differing columns are EXACTLY `Cluster` (6,892 cells), `mean_sector_speed` (21,903),
  `lap_time_vs_cluster_mean` (22,760) — and nothing else. All other 44 shared columns are
  identical. The artefact family split is fully explained by the two write paths
  (`N04:974/:979` Step 8 with k4 vs `N04:1093` Step 9 pre-`11a7ffa` with k4_2025).

### A3 — Step 9 running after Step 8 CLOBBERS the combined artefact with 2025-only rows: `process_2025_season` calls `finalize_and_save` (`N04:1089`), which unconditionally writes `laps_featured.parquet` (`N04:973-974`). A restart-and-run-all of today's N04 leaves the combined artefact with 22,760 rows and one season.

- `load_all_races()` (`N04:161-167`) globs EVERY year directory under `data/raw/` — today that
  includes 2025 — so Step 8 writes a 3-season combined + three per-year files. Step 9 then
  re-runs `finalize_and_save` on the 2025-only frame, overwriting `laps_featured.parquet` with
  2025-only content, before writing `laps_featured_2025.parquet` a second time at `N04:1093`.
  The notebook violates its own restart-and-run-all rule: the on-disk end state of a full run is
  a combined artefact that is not combined.
- Corollary: the header's "52,340 laps across 47 GPs (2023–2024)" (`N04:7`) describes a
  `data/raw/` that no longer exists; Step 8's output today depends on which seasons happen to be
  on disk. The 2025 rows inside the current combined artefact came from exactly this path (they
  carry the un-aliased `'Miami Gardens'` name — `_GP_NAME_ALIASES_2025` at `N04:1006` is applied
  only in `_load_raw_2025`, `N04:1028`, never in `load_all_races`).

### A4 — The k4 circuit files carry a hand-grafted `'Miami Gardens'` row that no notebook produces. It is the only reason a Step-8 run over 2025 data does not crash.

- `circuit_clusters_k4.parquet` and `circuit_features_with_clusters_k4.parquet` both contain
  BOTH `'Miami'` (msp 222.36, the 2023-24 pooled value) and `'Miami Gardens'` (msp 221.38 —
  which is the **2025-measured** Miami value, equal to `k4_2025`'s Miami row). N03's own code
  cannot emit a `'Miami Gardens'` row (its 2025 section aliases the name at
  `.nb_py/N03_circuit_clustering.py:1176-1200`, and its training section never saw 2025 dirs
  when it was run). Someone grafted the row into the k4 artefacts so that Step 8's merge would
  not leave `Cluster=NaN` on the 2025 Miami rows — `finalize_and_save` does
  `laps_featured['Cluster'].astype(int)` (`N04:955`), which raises on NaN.
- Consequence for regeneration: **re-running N03 would drop the graft**, and a subsequent
  Step-8 N04 run would crash on `astype(int)` (or, if the crash is "fixed" by filling, silently
  mis-cluster Miami). Regeneration must either keep the current k4 parquets untouched or move
  the alias into `load_all_races`. Also note the graft mixes eras inside one file: the
  `'Miami Gardens'` row's msp is a 2025 measurement sitting in the training-era artefact.

### B1 — Claim B VERIFIED and root-caused: Las Vegas 2025 `mean_sector_speed` NaN on all 760 rows because FastF1's SpeedI2 trap is missing for the ENTIRE 2025 Las Vegas race; N03's all-three-traps filter then yields zero valid laps.

- Raw `data/raw/2025/Las_Vegas/laps.parquet`: SpeedI2 **0%** not-NaN (886 rows), vs I1 80%,
  FL 97%, ST 100%. 2023/2024 Las Vegas have I2 at 100%/97%. The all-71-races sweep found **no
  other (year, GP) with a fully-missing speed-trap column** — this is the only one.
- Propagation: `_compute_sector_speed` (`.nb_py/N03_circuit_clustering.py:686-694`) requires
  all three traps → zero rows for Las Vegas → `circuit_features_with_clusters_k4_2025.parquet`
  carries msp NaN → the `how='left'` merge (`N03:1231`) preserves it → featured_2025 NaN ×760.
- **Regeneration cannot create this value** under N03's rule; the raw data does not contain it.
  The honest fixes are (a) record the hole in the dataset docs (per #801's own acceptance
  wording), or (b) change the rule for this circuit (e.g. mean of I1/FL) — which changes the
  quantity served vs every other race and moves the N06 holdout; (a) is right, (b) is not.

### B2 — Beyond #801: the full-hole sweep found holes #801 did not report.

Full-race (100% NaN) holes per artefact:
- combined: Las Vegas 2025 `SpeedI2`, `Prev_SpeedI2`, `SpeedI2_Delta` (760 rows each).
- featured_2025: the same three PLUS `mean_sector_speed` (the reported one).
- featured_2023 / featured_2024: none.

So the Vegas hole is not one column but FOUR in the 2025 file, and three of them exist in the
combined too (nobody had catalogued the SpeedI2 family). `Prev_SpeedI2` and `SpeedI2_Delta` are
N06-adjacent quantities; any consumer assuming "speed traps are always present except msp"
inherits them.

Partial holes >40% in one race (excluding by-design first-lap-of-stint NaN):
- featured_2025 Miami: `Stint`, `TyreLife`, `FuelEffect`, `FuelAdjustedLapTime`,
  `FuelAdjustedDegAbsolute`, `FuelAdjustedDegPercent` at 44% NaN (of 857 rows). **This one is
  KNOWN**: it is the raw-data tyre-stint hole GATE_tyrelife_nan_rootcause already traced
  (379 Miami rows of the 451-row gap; stint-shaped, laps 1-24, all 19 drivers in raw), and
  #790's repair covers it at serving time (`laps_augment.py:231` → `repair_tyre_stints`,
  `_apply_stint_corrections` at `laps_augment.py:135`). Recorded here only to note that
  **regeneration does NOT fix it** (the raw parquet is the hole) and does not conflict with
  the repair (which runs at augment time, downstream of the artefact).

### C1 — Claim C VERIFIED end-to-end, and the duplicate is byte-identical: `data/raw/2023/Spain` and `data/raw/2023/Barcelona` share OpenF1 session key 9102, identical record counts (1312/154/26036/43), extraction timestamps 21s apart (a test run and the real run, per N04's own `fix_spain_cluster_artefact` docstring, `N04:792-796`), and their raw laps are numerically identical row-for-row.

Double-count measurement:
- featured_2023 (and combined 2023): **1,198 Spain rows + 1,198 Barcelona rows = 5.4% of the
  22,106-row 2023 season is one race counted twice.** After featuring, the two copies differ in
  NO column except `GP_Name` (48-column cell diff: zero).
- **N06 trained on this**: `laps_featured_2023.parquet` is its training set
  (`.nb_py/N06_laptime_model.py:83`), so the 2023 Spanish GP carries double weight in the
  shipped lap-time model. Same for anything trained on featured_2023/combined.
- "71 races" = 70 unique + the duplicate (data/raw dirs: 23+24+24; the 2023 season had 22 races).
- N04 KNOWS about the duplicate and patches it instead of dropping it: `fix_spain_cluster_artefact`
  (`N04:792-826`) maps Spain's cluster values to Barcelona's — the claim-true-inside-false-headline
  pattern: the patch makes the duplicate *consistent*, which is precisely what makes it invisible.

### C2 — `laps_tiredeg.parquet` hides the same duplicate under ONE name (measured below) and drops the `'Miami Gardens'` spelling — the TCN's training artefact has its own naming convention, disagreeing with both featured families.

- `laps_tiredeg`: 68,122 rows — the SAME total as the combined featured artefact — yet it
  contains NO `'Spain'` and NO `'Miami Gardens'` GP_Name. 2023 row count 22,106 == featured's
  (which includes 1,198 Spain rows). Row-count identity with zero Spain rows means the Spain
  copy is still inside under another name.
- **Measured: `laps_tiredeg` Barcelona 2023 = 2,396 rows — double every other race (next
  largest: Monaco 1,397) — with 1,198 DUPLICATED `(Driver, Stint, LapNumber)` keys.** The Spain
  copy was RENAMED to Barcelona, not dropped: the duplicate is now invisible to any GP-name
  check and collides on the exact key the TCN's sequence builder groups by. Every 2023
  Barcelona stint in the TCN's training artefact contains each lap twice; a
  `(GP, Driver, Stint)` groupby sorted by LapNumber yields interleaved duplicate timesteps.
  This is worse than double weight — it is corrupted training sequences for that race, and it
  cannot be found by looking for `'Spain'`. (`laps_tiredeg` 2025 also renames to `'Miami'`,
  857 rows, no doubling — rename only.)

### C3 — The duplicate is COUNTED in the published position-projection ground truth: `measure_projection_ground_truth` (`src/strategy/eval/projection.py:286-300`) iterates every directory under `data/raw/<year>/` — 71 dirs including `Spain` — so the "86.5% within one place / 59.1% exact over 1810 stops in 71 races" sample scores the 2023 Spanish GP's stops twice. `races=71` in its return IS the count of directories, not of races. The N15 pit holdout does the same (`src/strategy/eval/pit_holdout.py:51`, `raw_root.glob("*/*/laps.parquet")`): the duplicated race's stops enter the 2023-24 training pool twice. (Exact stop contribution measured in C3-b below.)

### D1 — The weather-restore test's claim: the combined artefact NEVER carried weather in its PUBLISHED form. The test was written against a machine-local regeneration that was overwritten by an HF re-download 51 minutes after the commit.

Executed evidence:
- The four PUBLISHED artefacts on `VforVitorio/f1-strategy-dataset` (schema read via
  `HfFileSystem` + `pyarrow.read_schema`, footer-only): **48 columns each, NONE of
  AirTemp/TrackTemp/Humidity/Rainfall.** Same for all four LOCAL files today.
- N04 gained Step 5 (weather) on **2026-02-15** (`e8ce966`). Every artefact on disk and on HF
  predates that: no weather columns, and featured_2025 still carries the pre-`11a7ffa`
  (2026-02-17) cluster convention (A2). The published artefacts are a **pre-2026-02-15 build**;
  the producer has drifted from them for ~6 months.
- `tests/agents/test_weather_restore.py` was committed 2026-08-03 **09:35** (`f26e256`). Local
  file mtimes: `laps_featured_2025.parquet` 2026-08-03 **10:06** (HF download batch, together
  with `circuit_clustering/**` — exactly the `_DEFAULT_MODEL_PATTERNS` set), and
  `laps_featured.parquet` + `_2023` + `_2024` at **10:26** (a second, wider download; those
  three are NOT in the allow-patterns, so it was a manual/full pull). The test asserted
  against a local combined that carried weather — which only a post-2026-02-15 N04 rerun can
  produce — and that file was replaced by the weather-less HF copy the same morning.
- The test fails today with `KeyError: "['AirTemp', 'TrackTemp', 'Humidity', 'Rainfall'] not
  in index"` (executed: `uv run pytest tests/agents/test_weather_restore.py` → 1 failed,
  7 passed).

**Verdict:** the docstring's claim — "N04's actual output was published all along, and only the
per-year split dropped them" — was **always false of the published artefact**. Regeneration
does not *restore* the weather columns to the published dataset; it *creates* them there for
the first time. (It does restore the local state the test was written against, and the restore
module's 22,760/22,760 reproduction shows the values will be identical to what the test saw.)

### D2 — HF gap, #797's serving path: `_build_allow_patterns` ships ONLY `laps_featured_2025.parquet`; `pace_agent._load_circuit_mean_sector_speed` reads `laps_featured_{2023,2024,2025}` (`pace_agent.py:381-385`, `_FEATURED_SEASONS = (2023, 2024, 2025)`). On a clean HF install the 2023/2024 files are absent, the loop `continue`s, and every 2023/2024 replay resolves `mean_sector_speed` to NaN with a warning — silently degraded, exactly the failure class #798 documented for `undercut_clean`. `data_cache.py:109` must list all four featured parquets (or the per-year trio) after regeneration.

### C3-b — The duplicate's exact contribution to the published sample, executed with the eval's own functions (`green_flag_stops`, `project_one_stop`): the 2023 `Spain` directory contributes **42 scored stops**, identical to `Barcelona`'s 42 — so **84 of the 1,810 stops (4.6%) are one race scored twice**, and `races=71` counts 70 races. De-duplicated, the sample becomes ~1,768 stops over 70 races. The shipped threshold test survives it: `tests/mc/test_position_projection.py:50-51` asserts `>= 0.80` within-one over `>= 1500` stops — an EFFECT assertion with margin (the good pattern), so removing the duplicate cannot flip it. What DOES have to move: the prose numbers ("1810", "71 races") in `registry`/docs/memory wherever quoted.

### E1 — EXECUTED PROOF that regeneration is safe once the producer is fixed: an in-memory rebuild of `laps_featured_2025` (N04 Steps 1-8 verbatim, Step 9's cluster sources set to the `_2025` files — the pre-`11a7ffa` wiring) reproduces the shipped artefact with **ZERO differing cells across all 48 columns, 22,760/22,760 rows matched**, and the four newly-added weather columns match `weather_restore.weather_for_race`'s output with **0/22,760 mismatches on each of AirTemp/TrackTemp/Humidity/Rainfall**.

- Harness: session-scratchpad script (ephemeral; fully reproducible — it is N04's Step 1-8
  function bodies copied verbatim from `.nb_py/N04_feature_engineering.py`, the 2025 loader
  with the Miami alias, the `_2025` cluster sources, and the acceptance-diff snippet in the
  Appendix below; nothing written under `data/`). This closes the two
  load-bearing questions at once: (1) the pipeline is deterministic and still value-identical
  to the February build for every shipped column — the post-build commits (`e8ce966` weather,
  `6af182e` refactor) are value-neutral, and ONLY `11a7ffa`'s source swap separates today's
  code from the shipped artefact; (2) the weather columns a regeneration adds are exactly the
  values the committed restore module already serves, so the pace holdout
  (`tests/eval/test_ml_recompute_golden.py`: MAE 0.4104 ± 0.01) cannot move.
- One caveat: the harness output carried one stray extra column from its own simplified
  loader (53 vs the expected 52 = 48 + 4). The acceptance diff in the plan below compares
  COLUMN SETS as well as values, so a real rebuild cannot smuggle or lose a column silently.

---

## What I tried to break and could NOT

Verified sound with executed evidence — do not re-audit without cause:

1. **Row parity of combined vs per-year files**: per-year row counts (22,106 / 23,256 /
   22,760) match the combined's per-year slices exactly; join on (GP, Driver, LapNumber) has
   zero duplicate keys and zero unmatched rows in all three years (after the Miami alias).
2. **2023/2024 cross-artefact integrity**: cell-level diff over ALL 48 columns, combined vs
   per-year — zero differing cells. One build, no drift, no holes (no 100%-NaN race-columns).
3. **The 2025 split is exactly three columns wide**: everything except `Cluster`,
   `mean_sector_speed`, `lap_time_vs_cluster_mean` (and the GP naming) is identical between
   the combined's 2025 rows and `laps_featured_2025.parquet`. The artefact split at
   `N04:974/:979` vs `:1093` explains 100% of the observed disagreement; no second cause.
4. **`laps_featured_2025`'s `mean_sector_speed` == the raw truth under N03's own rule**: mean
   AND max |Δ| = 0.00 km/h across all 23 non-NaN GPs. The per-year 2025 artefact is not
   approximately right; it is the measurement.
5. **Speed-trap holes**: the all-71-races sweep found exactly ONE fully-missing trap column
   anywhere (2025 Las Vegas SpeedI2). No other race hides a Vegas-shaped hole.
6. **The Spain/Barcelona copies are truly identical** (raw: numerically identical row-for-row;
   featured: identical on every column except GP_Name) — so de-duplication loses no
   information, whichever copy is kept.
7. **Determinism/reproducibility of the producer pipeline** (E1): 48/48 columns exact after
   six months of code churn, on today's pandas/numpy. Regeneration is not a leap of faith.
8. **`tests/mc/test_position_projection.py` thresholds** hold under de-duplication (0.865
   measured vs 0.80 floor; 1,768 vs 1,500 floor).
9. **N04 exports no model and no manifest** (unlike N16): its only writes are the three
   `to_parquet` calls at `N04:974/:979/:1093` plus plots — nothing else needs backing up.

Not verified (out of scope, stated so nobody assumes): the VALUES of every engineered column
against an independent re-derivation (only `mean_sector_speed` and the artefact-vs-artefact
consistency were re-derived); `laps_tiredeg`'s other 55 columns beyond the keys measured in C2;
the NLP/radio artefacts.

---

# PART 2 — THE CONSOLIDATED, ORDERED FIX PLAN

Everything known as of 2026-08-04 — this gate's findings (G-*) plus GATE_DATA_WIRING's (W-F*),
one list, ordered by MEASURED effect on served output, each with the surgical fix and the
measurement that accepts it. W-F numbers are cited from `GATE_DATA_WIRING.md` (its harness:
21,247 anchored 2025 laps for N06; 335 stints / 8 races for the TCN), not re-derived; where
this gate touched the same artefacts the findings were reconciled and NO disagreement was
found (its F12 broadcast observation is A1's tiredeg-side twin; its cross-cutting note on the
augment guard is D1's territory and is subsumed by the regeneration).

## The one design rule that governs every fix below

**Serve what the model trained on; regenerate only what reproduces; document what neither
fixes.** Concretely: the artefact families (featured per-year vs combined vs tiredeg) carry
three different conventions for `Cluster` / `mean_sector_speed` / `lap_time_vs_cluster_mean`.
Each shipped model trained on ITS family. Fixes therefore go in the SERVING code (feed each
model its trained quantity), never in unifying the artefacts — with the single exception of
the combined artefact's 2025 rows, which nothing trained on and only tests read (their only
consumers: `test_weather_restore.py:121`, `test_pace_circuit_speed.py:27` existence-guard).

---

One deliberate re-ranking vs GATE_DATA_WIRING's own fix list, stated openly: it placed W-F1
(CompoundID) above W-F13 and W-F4; by its own measured numbers W-F4 moves more (8.2% of preds,
mean 0.0083s vs F1's 3.7% / 0.0055s), so this list orders F4 above F1. Nothing else disagrees.

## Ordered fix list

### 1. W-F7 — replace the replay weather lookup with N04's rule [CODE — largest measured mover]

- **Now:** `RaceStateManager.get_weather_state` (`src/simulation/race_state_manager.py:515-518`)
  does `idx = int((lap-1)/(total-1) * (len(weather_df)-1))` — proportional row-index, ignores
  session time. Measured: TrackTemp wrong on 92.6% of 26,692 driver-laps (mean 1.11°C, max
  11.8), rain flag flipped on 1,279 laps; N06 impact 26.8% of preds >1ms, mean 0.0367s, max
  8.28s.
- **Fix:** join each lap's session `Time` to the nearest weather sample —
  `weather_restore.weather_for_race(laps_df, weather_df)` already implements exactly N04's
  `merge_asof(direction='nearest')` and is verified 22,760/22,760 against N04's output. Build
  the per-lap weather table once in the RSM constructor (it has `laps_df` with `Time` and
  `weather_df`), then `get_weather_state(lap)` reads the current driver's lap row from it.
  Keep `track_temp_start` as-is (first session sample — verified correct).
- **Blast radius:** every replay-path weather consumer (N06 pace, N27 SC/overtake features,
  N26 fallback, N28) shifts on ~90% of laps — deliberately, toward the trained quantity. The
  CLI tire path does NOT change (already N04-faithful via augment). Any golden that snapshots
  replay recommendations will move; that shift is the fix working, re-freeze after review.
- **Measure:** before — for one evening race (Shanghai 2025):
  `python -c` comparing the proportional-index TrackTemp vs `weather_for_race` per lap prints
  ~2.38°C mean |Δ| today; after the fix the same comparison against `get_weather_state` must
  print 0.00 for every lap of every race (add as a pytest:
  `assert get_weather_state(lap)["track_temp"] == restored.loc[lap_row, "TrackTemp"]` over a
  full 2025 race, parametrised over 3+ races including Las Vegas and a rain race).
- **Do not touch:** `weather_restore.py` itself (it is the verified reference), and do not
  "improve" the alignment beyond `nearest` — the alignment IS the trained contract.

### 2. W-F8 + W-F9 + W-F10 + W-F11 — the tire agent's serving frame [CODE, one PR, `src/agents/tire_agent.py`]

Combined measured impact when all four serve the trained quantity: TCN |Δ| mean 0.424s,
p95 1.226s, max 4.99s (of a ~2s-cliff-threshold quantity) goes to ~0.02s (the F12 residue).

- **W-F8** (`tire_agent.py:663-664`): DELETE the `.shift(1)` on `DegradationRate` and
  `DegAcceleration` (and the "leakage fix matching N10 training" comment — it names a
  mechanism training never had; N04 computes both unshifted at
  `.nb_py/N04_feature_engineering.py:481-504`, N09 consumed them as stored,
  `.nb_py/N09_tiredeg_tcn.py:219-220`). TCN mean 0.185s, concentrated at cliff onset.
- **W-F9** (`tire_agent.py:768` + `:954`): the serving frame keeps the FEATURED
  `lap_time_vs_cluster_mean` while stomping `Cluster` from the k4 map — mixing families. Fix
  in code: recompute `lap_time_vs_cluster_mean` the way `laps_tiredeg` carries it (tiredeg's
  cluster family + its cluster-mean table), or join it from `laps_tiredeg.parquet` for the
  race being replayed. Do NOT regenerate artefacts to "agree": the TCN trained on tiredeg's
  version (mean disagreement 5.75s), N06 on featured's — unifying breaks one to please the
  other. TCN mean 0.208s.
- **W-F10** (`tire_agent.py:619`): `df[src_col].shift(1).fillna(df[src_col])` fills every
  missing predecessor with the CURRENT lap. Trained convention: leave NaN, let the scaler's
  `fillna(0)` (N10:176,181) produce the raw zero training saw. Replace with plain
  `.shift(1)`. Also fix the "first lap of a stint" docstring (wrong mechanism). TCN mean 0.198s.
- **W-F11** (`tire_agent.py:593`): DELETE `LapsSincePitStop = TyreLife` — the frame already
  carries N01's real column (present in both artefacts, 100% agreement); guard like FuelLoad.
  TCN mean 0.005s — last in this PR, but it is one line.
- **Measure (all four at once):** build `TireAgent._build_stint_features` for the 335-stint
  harness races and diff `DegradationRate`, `DegAcceleration`, `lap_time_vs_cluster_mean`,
  `LapsSincePitStop`, `Prev_*` NaN-pattern against `laps_tiredeg.parquet` for the same
  `(GP, Driver, Stint, LapNumber)` rows. Today: 0.28/0.40 s/lap mean on the deg pair, 100% of
  rows on ltvcm, 15.8% on LSPS, 6-22% NaN-pattern. Accepted when every column's divergence vs
  the trained artefact is 0 (deg pair, LSPS, ltvcm) / NaN-pattern-identical (Prev_*) on rows
  the artefact carries.
- **Do not touch:** the zero-pad/truncate/scaler path (`tire_agent.py:1048-1067`) — verified
  verbatim-N09; and `_add_session_cols`' other columns — verified 0.0% divergence.

### 3. W-F14 + W-F6 — one `_normalise_gp_key` at every gp_name-keyed lookup [CODE]

- **Now:** `'Miami Gardens'` (metadata.json name) misses `tire_compounds_by_race.json`
  (`'Miami'`-keyed) → every 2025 Miami stint routes SOFT to the **C3 TCN bundle instead of
  C5** (`tire_agent.py:819-821`), compound IDs −2 in N16 (`pit_strategy_agent.py:425-428`);
  pace `_encode_categorical` (`pace_agent.py:461`) and `_session_median`
  (`pace_agent.py:734-738`) miss the same key → default cluster + `delta_vs_median=NaN` all
  race (today masked by Miami's real cluster being the default 1 — latent, not harmless).
- **Fix:** apply the #797 four-keyspace chain (`_normalise_gp_key` + `slug_from_event_name`)
  at the four lookups: pace `_encode_categorical`, pace `_session_median`, tire
  `_compound_name_to_id`, pit `_compound_to_id`. Add the enumeration test
  `test_pace_circuit_speed.py` already models: every race dir under `data/raw/` resolves in
  ALL FOUR consumers.
- **Measure:** before — `TireAgent._compound_name_to_id('SOFT', 'Miami Gardens', 2025)`
  returns 3 (C3); after — 5 (C5), and the enumeration test passes 71/71 (70/70 post-dedup).
- **Blast radius:** the 2025 Miami replay's tire outputs change wholesale (that is the fix);
  nothing else resolves differently today (measured: Miami Gardens is the only 2025 miss).

### 4. W-F4 — feed N06 the PREVIOUS lap's speed trap [CODE]

- **Now:** `pace_agent.run_from_state` (`pace_agent.py:902,956`) feeds `d.get("speed_st")` —
  the CURRENT lap's trap — where training had the stint-grouped previous
  (`N04:384-392`). 8.2% of preds move, mean 0.0083s, max 2.44s. Twin of the fixed #435.
- **Fix:** have `get_driver_state` emit `prev_speed_st` (previous lap's `SpeedST` within the
  stint, NaN on openers) and wire `run_from_state` to it; keep the current-trap key for
  other consumers. Kill the `or 300.0` fallback (a findable real value; trained range
  156-362) — pass NaN.
- **Measure:** per-lap diff of the served `Prev_SpeedST` vs the featured parquet's column on
  a replayed race: today 88% differ (mean 4.6 km/h); after, 0 differ on rows the parquet
  carries.

### 5. W-F1 — CompoundID: serve the 1-based codes the model trained on [CODE + manifest correction]

- **Now:** `_encode_categorical` (`pace_agent.py:459`) uses the manifest's 0-based map →
  every lap served one compound class low; measured 3.7% of preds move, max 0.221s. The
  manifest block (`feature_manifest_laptime.json:66-73`) documents an encoding the model
  never ate.
- **Fix:** serve N01's 1-based codes (`N01:232,247`: 1=SOFT … 5=WET); correct the manifest
  block in the same change (else the next reader re-introduces the bug from the docs — the
  liar's-mirror pattern).
- **Measure:** before — served CompoundID for a MEDIUM lap = 1; after = 2, equal to the
  featured parquet's `CompoundID` for the same row, asserted over a full race (0 mismatches).

### 6. W-F13 — gate the overtake tool at its trained 2.5s domain [CODE]

- **Now:** N11 dropped every training pair with gap > 2.5s (`N11:233-235`);
  `predict_overtake_tool` (`race_situation_agent.py:1140-1204`) has no gap guard — 41.9% of
  real adjacent pairs (10,565 of 25,215 in 2025) are scored out-of-domain.
- **Fix:** refuse or explicitly label extrapolation beyond 2.5s (the #710 envelope pattern).
- **Measure:** the tool called with gap 9.0s today returns a bare probability; after, it
  returns the out-of-domain marker. Rate of unlabeled out-of-domain calls on a replayed
  race: 41.9% → 0%.

### 7. G-REGEN — the featured-artefact regeneration (this gate's core deliverable; runbook below) [PRODUCER CODE + ARTEFACT + HF]

Measured serving impact today: **≈0** — E1 proved the regenerated files are value-identical
on every column any model reads, and the added weather equals what the augment path already
restores. Its value is: `test_weather_restore.py` red→green, #782/#801 acceptance, the
published dataset finally matching its producer, and closing D2's clean-install degradation.
Ordered here, below the code fixes, per the measured-impact standard.

### 8. G-C — de-duplicate the 2023 Spanish GP [DATA + EVAL + HF — ONE-WAY, needs explicit sign-off]

Measured: 84/1,810 of the projection sample is one race twice (C3-b); 5.4% of featured_2023;
1,198 duplicated TCN training keys hidden inside tiredeg's Barcelona (C2). No serving-path
effect on 2025 replays. Bundled with G-REGEN's run if the decision lands in time (runbook §5).

### 9. Hygiene bundle [CODE, lowest measured impact — say so plainly]

Measured ≤0.021s or exactly zero: W-F2 (`DriverNumber=0`, moves 0.000s — model importance
0.03%), W-F3 (`FreshTyre` proxy, 0.000s — importance 0.00%), W-F5 (`Prev_TyreLife` opener
0-vs-NaN, 0.0003s), W-F12 (tiredeg msp broadcast — document, don't change), N15 missing-
TyreLife 0-vs-1 (`pit_strategy_agent.py:853-855`, 1.98% of rows), N27 latent cluster default
0→−1 (`race_situation_agent.py:1449`), gap-sign hardening, `pace_delta_rolling3` positional
pairing, the two wrong-mechanism docstrings (`tire_agent.py:1023`, `_add_prev_cols`), B1's
Vegas dataset-card note. None of these moves a prediction today; they are correctness debt.

---

## The regeneration runbook (Part 2 §§1-6 of the mandate)

### §1 What exactly to run — and what NOT

**Do NOT run the notebooks.** Three reasons, each measured: today's N04 Step 9 regresses
featured_2025 on 3 columns / 6,892 Cluster cells (A2); a full N04 run ends with the combined
artefact clobbered to 2025-only (A3); a fresh N03 run would refit K-Means on THREE seasons
(its loader globs every year dir) and drop the grafted `'Miami Gardens'` row the current k4
files carry (A4). `notebooks/**` is read-only per CLAUDE.md §7 anyway.

**Instead: a rebuild script** (proposed `scripts/rebuild_featured_laps.py`), function bodies
lifted from `.nb_py/N04_feature_engineering.py` exactly as `weather_restore.py` lifted Step 5
(the E1 harness `g801_repro25.py` is 90% of it). Behaviour:

1. Build 2023+2024 via Steps 1-8 logic — loader restricted to those two year dirs, k4
   cluster sources, `fix_spain_cluster_artefact` applied (or Spain dropped, §5) → writes
   `laps_featured_2023.parquet`, `laps_featured_2024.parquet`.
2. Build 2025 via Step 9 logic — Miami alias applied, **`circuit_clusters_k4_2025.parquet` +
   `circuit_features_with_clusters_k4_2025.parquet` as sources** (the pre-`11a7ffa` wiring
   E1 proved reproduces the shipped file) → writes `laps_featured_2025.parquet`.
3. `laps_featured.parquet` = `pd.concat` of the three per-year frames (2025 keeps the
   aliased `'Miami'` name — the combined stops disagreeing with its own splits by
   construction).

**Network:** none. N04 imports pandas/numpy/matplotlib/seaborn/scipy only; every input is a
local parquet. (`fastf1`/`get_session`/`requests` appear nowhere in it — verified.)

**Inputs read (must not be touched):** `data/raw/<year>/<gp>/{laps,weather}.parquet` (71
dirs), `data/processed/circuit_clustering/*.parquet` (all four files, INCLUDING the grafted
k4 rows — do not "clean" them, A4).

**Outputs / overwrites — back up (checksum) BEFORE the run, all four:**
`data/processed/laps_featured.parquet`, `laps_featured_2023.parquet`,
`laps_featured_2024.parquet`, `laps_featured_2025.parquet`. Nothing else: N04 exports no
model, no manifest (verified — unlike N16, its only writes are the three `to_parquet`s).
Backup: `Copy-Item data/processed/laps_featured*.parquet <backup_dir>/` + record
`Get-FileHash` of each; the acceptance diff below runs against these copies.

### §2 Is running the producer even sufficient? NO — the defects are IN the producer

#801 says "the fix is a regeneration rather than code". **Refuted.** Four of the defects live
in N04/N03's own logic and survive any rerun; two get WORSE:

| Defect | Producer line | Rerun outcome without the code fix |
|---|---|---|
| 2025 cluster/msp regression | `N04:1041,1072-1076` (post-`11a7ffa`) | featured_2025 loses its season-true values (A2) |
| Combined clobber | `N04:973-974` via `N04:1089` | combined ends 2025-only (A3) |
| Season-broadcast msp in combined | `N04:752-757` (merge on GP_Name, no Year) | reproduced identically — broadcast is BY CONSTRUCTION |
| Spain duplicate | `N04:792-826` patches, never drops | reproduced identically |
| Miami dual-name in combined | alias only in `_load_raw_2025` (`N04:1028`) | reproduced identically |
| Vegas msp hole | raw SpeedI2 100% missing (B1) | reproduced identically — unfixable by rerun |

The rebuild script IS the code fix for the first, second and fifth; the third dissolves under
combined-as-concat; the fourth is §5's decision; the sixth is documentation.

### §3 Acceptance gates — executable, with expected values

Run in this order; any failure stops the line:

1. **Column-set + value diff vs the backups** (the anti-"silently worse" gate; script =
   `g801_repro25.py`'s diff block pointed at backup vs new):
   - featured_2025: identical column VALUES on all 48 backed-up columns (0 differing cells,
     `atol=1e-6`), 22,760 rows, + exactly 4 new columns {AirTemp, TrackTemp, Humidity,
     Rainfall}. Las Vegas `mean_sector_speed` still 100% NaN (760 rows) — a regeneration
     that "fixes" it has changed the rule, reject.
   - featured_2023/2024: identical on all 48 (minus Spain rows iff §5 approved), + the 4.
   - combined: == concat of the three per-year files, byte-for-byte per year-slice.
2. **Weather values == the committed restore** (the alignment gate): for each year,
   `augment_featured_laps(backup_per_year, year)` vs the new file's weather columns →
   0 mismatches / 22,760 (2025) and 0 on 2023/2024. (E1 already measured 0/22,760 for 2025.)
3. `uv run pytest tests/agents/test_weather_restore.py` → **8 passed** (today: 1 failed —
   the KeyError test goes green). CAVEAT: `test_a_partial_weather_set_is_declined...`
   (`test_weather_restore.py:168-185`) constructs its "partial" frame by ASSIGNING AirTemp to
   the artefact — once the artefact natively carries all four, the frame is no longer
   partial and `assert "TrackTemp" not in result.columns` fails. Update it to DROP three of
   the four instead (`partial = full.drop(columns=["TrackTemp","Humidity","Rainfall"])`).
   This is a test written against the artefact era, not a regression.
4. `uv run pytest tests/agents/test_pace_circuit_speed.py` → all pass unchanged (values
   identical ⇒ the (year,GP)→msp map identical; Silverstone 2023 249.71 ≠ 2025 231.36,
   Vegas 2025 NaN, Vegas 2023 ≈228.96 all still hold).
5. `uv run pytest tests/eval/test_ml_recompute_golden.py` → pace holdout reproduces
   **0.4104 ± 0.01** (gate 2 guarantees it: same features, same weather values).
6. `uv run pytest tests/mc/test_position_projection.py` → ≥0.80 within-one over ≥1500 stops
   (unaffected by regeneration; moves only under §5, within margin).

### §4 Hugging Face: upload and allow-patterns

- **Upload (overwrite in place)**: `data/processed/laps_featured.parquet`,
  `laps_featured_2023.parquet`, `laps_featured_2024.parquet`, `laps_featured_2025.parquet`
  to `VforVitorio/f1-strategy-dataset`. These are the ONLY artefacts the run changes.
- **Do NOT upload**: anything under `data/raw/` (unchanged), `laps_tiredeg.parquet`
  (unchanged — its C2 duplicate is a separate, retrain-gated decision), the
  `circuit_clustering/` files (unchanged inputs), any backup copies.
- **If §5 approved**: additionally DELETE `data/raw/2023/Spain/**` from the HF dataset (and
  locally) — otherwise the next `snapshot_download`/full pull resurrects the duplicate.
- **`_build_allow_patterns` / `_DEFAULT_MODEL_PATTERNS`** (`src/f1_strat_manager/data_cache.py:109`):
  add the three missing featured files next to the existing `laps_featured_2025.parquet`
  line — `"data/processed/laps_featured.parquet"`, `"...laps_featured_2023.parquet"`,
  `"...laps_featured_2024.parquet"` — closing D2 (clean installs currently lose #797's
  2023/2024 resolution silently). Ship this edit in the same PR as the upload.

### §5 Ordering constraints — so nothing is regenerated twice

1. **Producer-fix code review BEFORE any regeneration run** (§2). The rebuild script PR and
   the artefact swap can be one PR, but the script must be reviewed against E1's evidence
   first.
2. **The Spain decision (fix 8) BEFORE the regeneration run.** Dropping Spain changes
   featured_2023 (22,106 → 20,908 rows) and the combined; deciding it after means running and
   uploading twice. It needs explicit sign-off because it is one-way and touches published
   numbers: raw dir deletion (local + HF), projection sample 1,810 → ~1,768 / races 71 → 70
   (threshold test survives, C3-b), pit-holdout train pool loses the doubled stops, the
   parametrised `("Spain", 2023)` case in `tests/agents/test_pace_circuit_speed.py:180` must
   be repointed/removed (with featured Spain rows gone, `_resolve_mean_sector_speed("Spain",
   2023)` returns NaN — gp_slugs has no 'Spain' alias), and every doc quoting "71 races"
   needs the footnote. If sign-off is not available, regenerate WITH Spain (keeping
   `fix_spain_cluster_artefact`) and accept a second run later — state the cost, don't stall
   the weather fix on it.
3. **GATE_DATA_WIRING's code fixes (1-6 above) are INDEPENDENT of the regeneration** — no
   artefact they read changes value (E1). Land them in any order relative to G-REGEN. The one
   coupling: W-F9 must be fixed in the AGENT (families stay as trained) — do not let the
   regeneration PR "helpfully" unify `lap_time_vs_cluster_mean` across families; that would
   convert W-F9's 0.208s defect into a new 5.75s-scale one on whichever model loses.
4. **`laps_tiredeg` is NOT regenerated** until a TCN retrain is on the table (C2's corrupted
   Barcelona sequences are baked into the shipped model; regenerating the artefact without
   retraining only widens train/artefact drift). Document, defer, track.
5. After upload: re-run gates §3.3-3.6 once more against the HF-downloaded copies (fresh
   `snapshot_download` into a temp dir) — the published-vs-local mismatch is exactly how D1
   happened.

### §6 What could go wrong, and how it presents

| Failure | How it presents | What it means |
|---|---|---|
| pandas/numpy version drift changes float bits | §3.1 diff shows scattered ≤1e-9 diffs in polyfit-derived cols (`DegradationRate`, `DegAcceleration`) | Harmless if within atol; if larger, pin the env (uv.lock) and re-run — do NOT widen the tolerance |
| Rebuild uses today's Step 9 sources by mistake | §3.1: exactly 3 columns differ on 2025 (`Cluster` 6,892 cells, msp, ltvcm) — the A2 signature | Wrong cluster sources; fix the script, re-run |
| Step-8-style clobber sneaks in | combined has 22,760 rows / one year | A3 reproduced; the concat step didn't run |
| Vegas msp suddenly has a value | §3.1 flags 760 changed cells | The rule changed (I1/FL-only mean?) — reject, that silently shifts the N06 test distribution |
| Miami rows crash `astype(int)` | `IntCastingNaNError` in the 23-24 build | The k4 graft (A4) was "cleaned" or the alias landed in the 23-24 loader — restore the grafted files from backup |
| Weather all-NaN for a race | gate §3.2 mismatch counts explode for one GP | That race's `weather.parquet` missing/empty — stop; all 71 have one today (GATE_DATA_WIRING verified) |
| test_pace_circuit_speed fails on ("Spain", 2023) | after a Spain-dropping run | Expected under §5.2 — the test update is part of that PR, not a regression |
| Holdout MAE moves >0.01 | gate §3.5 red | The regenerated features are NOT value-identical — §3.1 was skipped or its tolerance widened; restore backups, diff, find the changed column |
| A second session re-downloads from HF mid-work | mtimes under `data/processed/` jump, weather columns vanish again | The D1 mechanism repeating — re-run §3.1 against backups before trusting anything |

---

## Suggested PR grouping and order

| # | PR | Contents | Why this order |
|---|---|---|---|
| 1 | `fix(simulation): weather join on the replay path` | Fix 1 (W-F7) | Largest measured mover; single file; independent of everything |
| 2 | `fix(agents): tire serving frame matches the trained artefact` | Fix 2 (W-F8/F9/F10/F11 + docstrings) | Second-largest block; one file; contains W-F9's family rule so the regen PR can't get it wrong |
| 3 | `fix(agents): gp_name keyspace normalisation` | Fix 3 (W-F14/F6) + enumeration test | Unbreaks a whole race's tire routing; touches tire_agent → after PR 2 to avoid conflicts |
| 4 | `fix(agents): pace model inputs (CompoundID, Prev_SpeedST)` | Fixes 4-5 (W-F4, W-F1 + manifest) | Same file pair, both "serve the trained quantity" |
| 5 | `fix(agents): overtake domain gate` | Fix 6 (W-F13) | Independent |
| 6 | `fix(data): regenerate featured artefacts (weather + 2025 conventions + combined=concat)` | Fix 7: rebuild script + §3 gates + `data_cache` patterns + `test_weather_restore` partial-test update + HF upload | After the §5.2 Spain decision; the only PR that touches artefacts |
| 7 | `fix(data)!: de-duplicate the 2023 Spanish GP` | Fix 8: raw dir removal (local+HF) + test/doc/number updates | Ideally folded into PR 6's single regeneration run with sign-off; standalone otherwise |
| 8 | `chore(agents): input-wiring hygiene` | Fix 9 bundle | Zero measured impact; last |

PRs 1-5 are pure code and can land in parallel branches (2→3 sequenced on the shared file).
PR 6 is the only artefact writer; it runs the regeneration exactly once.

---

## Appendix — the acceptance-diff snippet (gate §3.1/§3.2), self-contained

```python
"""Diff a regenerated featured parquet against its backed-up predecessor.
Usage: python check_regen.py <backup.parquet> <new.parquet> <year>"""

import sys
import numpy as np
import pandas as pd

backup, new, year = pd.read_parquet(sys.argv[1]), pd.read_parquet(sys.argv[2]), int(sys.argv[3])
WEATHER = ["AirTemp", "TrackTemp", "Humidity", "Rainfall"]
KEYS = ["GP_Name", "Driver", "LapNumber"]

# 1. Column contract: nothing lost, exactly the four gained
lost = set(backup.columns) - set(new.columns)
gained = set(new.columns) - set(backup.columns)
assert not lost, f"columns LOST: {lost}"
assert gained == set(WEATHER), f"unexpected column change: {gained}"

# 2. Row contract (adjust 2023 expectation to 20,908 iff the Spain drop is approved)
assert len(new) == len(backup), f"rows {len(backup)} -> {len(new)}"

# 3. Every backed-up cell survives (atol absorbs BLAS-level polyfit jitter only)
j = backup.merge(new, on=KEYS, suffixes=("_old", "_new"), how="outer", indicator=True)
assert (j["_merge"] == "both").all(), "row set changed"
for c in backup.columns:
    if c in KEYS:
        continue
    a, b = j[f"{c}_old"], j[f"{c}_new"]
    if a.dtype.kind in "ifb" and b.dtype.kind in "ifb":
        bad = ~(
            (a.isna() & b.isna())
            | np.isclose(a.astype(float), b.astype(float), rtol=0, atol=1e-6, equal_nan=False)
        )
    else:
        bad = (a.astype(str) != b.astype(str)) & ~(a.isna() & b.isna())
    assert not bad.any(), f"{c}: {int(bad.sum())} cells changed"

# 4. The added weather equals the committed restore (the alignment gate)
from src.f1_strat_manager.laps_augment import augment_featured_laps

restored = augment_featured_laps(backup, year)
jr = restored.merge(new[[*KEYS, *WEATHER]], on=KEYS, suffixes=("_restore", "_new"))
for c in WEATHER:
    a, b = jr[f"{c}_restore"], jr[f"{c}_new"]
    bad = ~(
        (a.isna() & b.isna())
        | np.isclose(a.astype(float), b.astype(float), rtol=0, atol=1e-9, equal_nan=False)
    )
    assert not bad.any(), f"weather {c}: {int(bad.sum())} mismatches vs restore"

# 5. The Vegas hole is preserved (a regeneration that 'fixed' it changed the rule)
if year == 2025:
    lv = new[new["GP_Name"] == "Las Vegas"]
    assert lv["mean_sector_speed"].isna().all() and len(lv) == 760

print("OK: regeneration is value-identical + weather matches the restore")
```


