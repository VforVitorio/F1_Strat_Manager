# PR 6 — the featured-artefact regeneration, as it happened

Running log, appended as each step is verified. The plan is
`GATE_801_ARTEFACTS.md` §§1-6; this file records what the steps actually measured.

## Starting state, measured 2026-08-05

Backups taken before anything wrote, `data/_backup_pr6_featured/`, SHA-256 recorded:

| file | bytes | sha256 (first 16) |
|---|---|---|
| `laps_featured_2023.parquet` | 2,000,460 | `70EA7285F97D27D6` |
| `laps_featured_2024.parquet` | 2,161,115 | `64A761888B0B4732` |
| `laps_featured_2025.parquet` | 2,078,087 | `3ACE08A9202CEA19` |
| `laps_featured.parquet` | 5,434,472 | `0D114411EE4B5B41` |

| file | rows | cols | seasons | GPs | weather cols |
|---|---|---|---|---|---|
| 2023 | 22,106 | 48 | 2023 | **23** | **0/4** |
| 2024 | 23,256 | 48 | 2024 | 24 | **0/4** |
| 2025 | 22,760 | 48 | 2025 | 24 | **0/4** |
| combined | 68,122 | 48 | 2023-25 | **26** | **0/4** |

68,122 = 22,106 + 23,256 + 22,760, so the combined is already a clean concat of the three.
23 GPs in 2023 is Spain + Barcelona, the duplicate. 26 in the combined is those plus the
`Miami` / `Miami Gardens` pair.

**This makes `weather_restore.py`'s own docstring false today.** It says "`laps_featured.parquet`,
the combined 2023-2025 artefact, carries all four columns for 2025. Only the per-year split
dropped them." No file carries them. That is the drift this PR closes, and it is why
`test_the_restore_reproduces_N04s_own_2025_output_exactly` fails: its ground truth is gone.

## Which cluster source built the shipped 2025 file — RESOLVED BY MEASUREMENT

The runbook and today's `.nb_py/N04` disagreed. `process_2025_season` reads
`circuit_clusters_k4.parquet` (the pooled map); the runbook says the shipped artefact came
from `circuit_clusters_k4_2025.parquet`, the wiring that predates commit `11a7ffa`.

Settled by asking the artefact itself — take its own `GP_Name → Cluster` mapping and check
each candidate source against it:

```
artefacto 2025: 24 GPs con cluster
  circuit_clusters_k4.parquet         coincide 17/24
  circuit_clusters_k4_2025.parquet    coincide 24/24
```

**The runbook is right and the notebook is not.** Rebuilding 2025 with today's N04 wiring
would move 7 of 24 races to a different cluster — the A2 regression the gate predicted,
observed directly rather than inferred. The rebuild script uses the 2025 sources for 2025.

This is also why the rebuild is a script and not a notebook run: `notebooks/**` is read-only
by project rule, and the producer on disk no longer reproduces its own output.

## Gate A — does the rebuild reproduce what is published?

`uv run python scripts/rebuild_featured_laps.py --verify` (writes nothing):

| file | rows | shipped columns that CHANGED |
|---|---|---|
| `laps_featured_2023.parquet` | 22,106 → 22,106 | **none** |
| `laps_featured_2024.parquet` | 23,256 → 23,256 | **none** |
| `laps_featured_2025.parquet` | 22,760 → 22,760 | **none** |
| `laps_featured.parquet` | 68,122 → 68,122 | `GP_Name` 857, `Cluster` 6,892, `mean_sector_speed` 21,903, `lap_time_vs_cluster_mean` 22,760 |

The three per-year files reproduce to 1e-6 on every one of their 48 columns.

### The combined file's difference is a defect it already had

The obvious reading is that the rebuild broke the combined file. It did not — **the published
combined file disagrees with its own 2025 split**, and by exactly the same counts:

```
combinado slice 2025: 22760 filas   fichero 2025: 22760 filas
EL COMBINADO PUBLICADO vs SU PROPIO SPLIT 2025 ->
  {'GP_Name': 857, 'Cluster': 6892, 'mean_sector_speed': 21903, 'lap_time_vs_cluster_mean': 22760}
```

Two files describing the same 22,760 laps, published together, differing on four columns.
857 is Miami's row count (the alias reaches one file and not the other); 6,892 is the A2
cluster signature. Building the combined as a `concat` of the parts makes the disagreement
impossible by construction, which is why the runbook prescribes it.

### The rebuild emits 53 columns, not 48 + 4

Five columns are added, not the four the runbook's gate expected: the weather quartet plus
`lap_time_pct_of_race_fastest`. That is not a deviation — it is the ORIGINAL schema.
`weather_restore.py`'s own docstring states that `laps_featured_2023/2024.parquet` "carry 53
columns including AirTemp, TrackTemp, Humidity and Rainfall". They carry 48 and none of them
today. **48 + 4 + 1 = 53**: the rebuild reconstructs the artefact as documented, and the
files on disk are the degraded copies an HF re-download installed.

Checked before keeping it, because publishing a column no consumer reads is how a leakage
trap gets shipped: `hygiene.py` flags `lap_time_pct_of_race_fastest` LEAKY and
`pace_holdout` therefore excludes it from the 25 prediction columns, while `tire_agent`
recomputes it unconditionally from `session_meta["fastest_lap_s"]`
(`tire_agent.py:847-851`). Nothing reads it from the parquet, so restoring it changes no
value anywhere — it only makes the published file match its own documentation again.

## Spain — the drop is surgical

`--drop-spain`: `laps_featured_2023` 22,106 → **20,908** rows (−1,198), 23 GPs → 22, and the
combined 68,122 → 66,924. Exactly the counts the gate predicted.

The check that mattered more than the count: with Spain gone the rebuild also skips
`fix_spain_cluster_artefact`, which patches Spain's rows to Barcelona's values. Skipping a
patch is the kind of thing that quietly changes its neighbour, so the **surviving** rows were
diffed against the shipped file's non-Spain rows, keyed on (GP, driver, lap, stint):

```
sin Spain: shipped 20908  rebuild 20908
columnas alteradas en las filas SUPERVIVIENTES -> NINGUNA
```

## Las Vegas — the estimator, re-derived rather than quoted

FastF1 has **no SpeedI2 reading for the entire 2025 Las Vegas race**: 0% of 886 raw laps,
against I1 80%, FL 97%, ST 100%. N03 filters on all three traps, so the circuit's speed comes
out NaN on all 760 featured rows. No re-run recovers it — the reading does not exist.

Estimator: the season's own two-trap mean plus the (three-trap − two-trap) gap measured at
**that circuit** in its other seasons, because the missing trap's contribution is a property
of the layout while the speeds are season-true. Scored leave-era-out over every
circuit-season that has a real three-trap value:

| offset source | MAE | p95 | n |
|---|---|---|---|
| **the circuit's own other seasons** | **1.22 km/h** | 3.40 | 68 |
| averaged across all circuits | 9.44 km/h | 20.82 | 70 |

The second is the same idea with the layout term discarded, and it is nearly eight times
worse. That gap is the argument for the first.

**Las Vegas 2025: two-trap 245.977, own offset −13.150 → 232.83 km/h.**

### The earlier audit's number does not reproduce

`GATE_801_ARTEFACTS.md` reports 239.14 km/h for this. Neither reading of its own stated
method ("that circuit's own (3-trap minus 2-trap) gap") produces it:

```
offset del propio circuito  -> 232.83 km/h
offset global               -> 241.54 km/h
el gate dice                   239.14 km/h
```

232.83 is used, because it is the value the leave-era-out validation above actually scores.
The audit's MAE figures reproduce closely (1.22 vs its 1.25; 6.52 vs its 7.60 for the
prior-season baseline) — the method survives, the published number does not.

### Scope of the imputation — Víctor's call, 2026-08-05

The hole is four columns: `mean_sector_speed`, `SpeedI2`, `Prev_SpeedI2`, `SpeedI2_Delta`.
**Only `mean_sector_speed` is imputed**, plus a `mean_sector_speed_imputed` boolean on every
row. It is a per-CIRCUIT constant with a validated estimator; the other three are per-lap
sensor readings, and fabricating 760 individual ones has no validation behind it.

Dry run, `--drop-spain --impute-circuit-speed`:

```
Las Vegas 2025: two-trap 245.977 + offset -13.150 -> 232.83 km/h over 760 rows (FLAGGED)
laps_featured_2025.parquet  CHANGED {'mean_sector_speed': 760}
laps_featured_2024.parquet  CHANGED none
```

760 cells, and not one other value in the season moves.

## The published projection number, before and after

`measure_projection_ground_truth` reads `data/raw/`, NOT the featured artefact, so dropping
the featured rows moved nothing on its own. The duplicate had to go from the raw tree too —
verified byte-identical first, and backed up locally before deletion:

```
Spain        1312 vueltas  hash_laps=16ed77bad27d8e51
Barcelona    1312 vueltas  hash_laps=16ed77bad27d8e51
```

| sample | races | stops | within one | exact |
|---|---|---|---|---|
| published today | 71 | 1,810 | 86.5% | 59.1% |
| **after the de-duplication** | **70** | **1,768** | **86.3%** | **59.2%** |
| **2025 only — measured here for the first time** | **24** | **552** | **86.1%** | **59.6%** |

The de-duplication lands exactly on the predicted 1,768 / 70 and barely moves the headline.

**The 2025-only figure had never been measured.** `project_metrics_rescope_2025` says so
explicitly and warns against assuming it lands near 86.5%. It does: 86.1% within one over
552 stops. That is a real input to the measurement session — the headline survives the
restriction to the season the system actually infers on, so the re-scope costs accuracy
nothing and buys the claim its honesty.

## A regression the acceptance gates missed and the real simulation caught

After the first write, `f1-sim` on Miami moved one lap's call (`STAY_OUT·24 UNDERCUT·9` →
`25/8`) while Lusail was untouched. Miami's cells had not changed, so something outside the
value diff had.

Gate §3.2 — the one comparing the NATIVE weather against what `augment_featured_laps`
restores at load time — had not been run yet. It found it:

```
ANTES (backup + augment):      Miami 857 filas  AirTemp NaN 0
AHORA (artefacto regenerado):  Miami 857 filas  AirTemp NaN 857
```

**The regeneration had blanked Miami's weather for the whole race.** `_load_raw_2025` renames
`GP_Name` on the LAPS frame and returns the weather frame untouched, and
`add_weather_features` merges on `GP_Name` — so with one side saying `Miami` and the other
`Miami Gardens`, the merge matched nothing. N04 has the same defect; reproducing it faithfully
is how it arrived here.

The fix applies the alias to both frames, and it is the ONE place this script knowingly
diverges from N04. It can: the weather columns were never in the published artefact, so there
is no trained-on value to preserve. Their correct value is what the runtime restore produces,
because that is what every model has actually been fed.

After the fix, gate §3.2 across all three seasons:

```
2023: 20908 filas comparadas  weather mismatches -> 0
2024: 23256 filas comparadas  weather mismatches -> 0
2025: 22760 filas comparadas  weather mismatches -> 0
```

**The lesson, again: the value diff, the test suite and the acceptance gates all passed on an
artefact with a whole race's weather missing.** Only running the thing found it — the same
shape as the CLI crash in PR 5, where 35 of 57 laps errored on a branch whose suite was green.

## Real simulations against the final artefacts

| race | result | against the pre-regeneration baseline |
|---|---|---|
| Lusail | All 57 laps OK · `STAY_OUT·53 PIT_NOW·1 UNDERCUT·3` | identical |
| Miami | All 33 laps OK · `STAY_OUT·24 UNDERCUT·9` | identical |
| Las Vegas | All 50 laps OK · `STAY_OUT·45 UNDERCUT·5` | runs on a real circuit speed |

Miami is the one that matters: it had moved to `25/8` on the broken write and came back to
`24/9` once the weather alias was fixed, which is what confirms the regression and its repair
rather than merely asserting them.

Las Vegas is the race the imputation is for. Before, `_resolve_mean_sector_speed("Las Vegas",
2025)` returned NaN, so N06 was fed a missing feature on every one of its 760 laps; it now
receives the circuit's own imputed 232.83 km/h, flagged.

## Three frozen numbers the correction moved, and what each one was

The full suite found them. All three are consequences of removing a race that was in the
dataset twice, and none is a regression.

**The weather defaults.** `DEFAULT_AIR_TEMP_C` / `DEFAULT_TRACK_TEMP_C` are documented as the
median of the real seasons. That median was computed over a sample counting one weekend's
weather double: **24.2 / 34.2 → 24.6 / 34.7**. The constants now describe the de-duplicated
dataset.

Its sibling test, `test_the_pair_is_the_measured_median_not_a_round_number`, was
`assert DEFAULT_AIR_TEMP_C == 24.2` — the constant compared against a literal copy of itself.
It can only fail when someone edits the constant, INCLUDING when they edit it correctly, and
that is exactly what happened here. Rewritten to assert the property it is named for: not the
tidy 25/35 pair, not a whole number, inside a plausible range. The measurement itself is
already guarded by the test below it, against the dataset.

**The pit P05-P95 coverage: 177/252 → 176/252.** `_collect_pit_stops` globs the raw tree, so
Spain's stops left the pool the holdout builds its per-circuit and per-team aggregates from
(Spain now contributes 0 against Barcelona's 136). The denominator is unchanged because the
coverage is scored on 2025 — what moved is one 2025 stop's interval, through an aggregate
that had been counting one weekend twice.

**The frozen projection golden: STAY_OUT E 1.276 → 1.28, score 1.288 → 1.29.** The other
three candidates are identical to the digit. Same cause, same size: a fourth-decimal move in
one candidate's expected position.

None of the three was found by the acceptance gates — they were found by running the whole
suite, which is the third distinct verification layer this batch has needed.

## What the adversarial gate found, and what it refuted

It verified the core with independent, executed evidence — exact-equality diffs rather than
`isclose`, row order proven identical by key tuples, all 12 lifted functions AST-verbatim
against `.nb_py/N04`, the leave-era-out reproduction landing on n=68 / MAE 1.22 / 232.83 to
the digit, and the Miami weather fix clean across 66,924 rows. Then it found seven things.

### It refuted a claim of mine

**"53 columns is the ORIGINAL schema" is false.** I read that off `weather_restore.py`'s
docstring; the gate dated the producer with `git log -S` and found `lap_time_pct_of_race_fastest`
added 2026-02-12 and the weather merge 2026-02-15, both AFTER the published artefact, which
really was 48 columns. GATE_801's own D1 had already refuted that docstring, and I cited it
anyway. The columns are still right to restore — the consequences are verified nil, since the
model selects features by an include-list and the tyre agent recomputes the ratio — but the
provenance I gave for them was wrong. **The rebuild reproduces a NEWER N04 than the one that
made the artefact, and that is the honest description.**

### The imputation does not improve accuracy, and saying otherwise would have been easy

Measured by restoring the NaN and re-running the pace holdout:

```
MAE 2025 con Vegas imputado: 0.4098 s  (n=21247)
MAE 2025 con Vegas en NaN   : 0.4097 s  (n=21247)
```

**Marginally WORSE.** XGBoost reads a missing feature natively through its sparse-aware split
and does so slightly better than the imputed value does. The imputation's value is that the
dataset no longer carries an unexplained hole and that the fill is labelled — schema honesty,
not accuracy — and the flag is what lets anyone measure exactly this.

### The second Miami-shaped defect, in my own new code

`impute_circuit_speed` resolved the raw directory with `gp.replace(" ", "_")`, one of the
three forms. `Miami` lives in `Miami_Gardens`, so a renamed circuit losing a trap would have
raised FileNotFoundError. Dormant only because Las Vegas is today's sole hole and its name is
the easy case. `laps_augment._raw_race_dir` already carries all three forms and is now used.

### The truth test had gone vacuous

`test_the_restore_reproduces_N04s_own_2025_output_exactly` calls `augment_featured_laps` on
the artefact — which now carries weather natively, so the restore DECLINES and the test
compared the file with itself. The gate proved it by poisoning the join and watching it pass.
It now strips the four columns first and asserts the restore rebuilds them; poisoning
`direction="nearest"` to `"backward"` makes it fail, which is the check the old one had lost.

### And three smaller ones, all fixed

- **The gate did not gate.** Write mode reported the diff and wrote anyway, and `--drop-spain`
  returned 0 unconditionally — so the one run that changes rows on purpose could never fail.
- **`_compare` used `np.isclose`'s default `rtol=1e-5`**, which on an elapsed time near 9,000 s
  tolerates nine hundredths of a second. Now `rtol=0`.
- **The hardcoded race counts**: six modules said "71 races" in comments describing measured
  constants, and `_shared_defaults` still quoted the pre-de-duplication medians.

### The measured MC tables — and a test that heals itself

The seven tables in `data/mc_measured_v1.json` are read at runtime by the projection scorer
and were still counted off 71 races. They are regenerated here (`races_measured: 70`,
`status_mix` clear laps 3,796 → 3,730).

Worth recording how that surfaced: `test_the_committed_tables_match_a_fresh_measurement`
RUNS the measuring script, which REWRITES the file, and then compares. So a stale file fails
once, is silently repaired by the failing run, and passes on the retry. The drift it exists
to catch disappears the moment you re-run it — and `tests/mc/` is data-gated, so CI never
sees either state.
