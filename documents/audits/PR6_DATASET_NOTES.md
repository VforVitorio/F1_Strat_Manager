# What changed in `VforVitorio/f1-strategy-dataset`, 2026-08-05

Notes for the dataset card. Every number here is measured; the full working is in
`PR6_REGENERATION_LOG.md`.

## The four featured files were regenerated

`laps_featured.parquet`, `laps_featured_2023.parquet`, `laps_featured_2024.parquet`,
`laps_featured_2025.parquet`.

Produced by `scripts/rebuild_featured_laps.py`, which lifts N04's feature functions verbatim
and replaces only the orchestration around them. Every one of the 48 previously published
columns reproduces to 1e-6; the acceptance diff is in the log.

### Schema: 48 → 54 columns

| added | why |
|---|---|
| `AirTemp`, `TrackTemp`, `Humidity`, `Rainfall` | N04 computed them and the published split dropped them. `augment_featured_laps` has been restoring them at load time; the file now carries what the models are actually fed. Verified equal to that restore, 0 mismatches across all three seasons. |
| `lap_time_pct_of_race_fastest` | Same story. Part of the original schema — `weather_restore.py` documents the 2023/24 files as carrying **53** columns, and 48 + 4 + 1 = 53. Flagged LEAKY by `hygiene.py`, excluded from the pace model's inputs, recomputed by the tyre agent: **do not train on it**. |
| `mean_sector_speed_imputed` | The honesty flag for the row below. |

### The 2023 Spanish GP duplicate is gone

`data/raw/2023/Spain/` and `data/raw/2023/Barcelona/` were the same session, extracted twice.
Verified byte-identical before removal: 1,312 laps each, identical content hash
`16ed77bad27d8e51`. `Spain` has been removed from the raw tree and from the featured files.

`laps_featured_2023.parquet`: 22,106 → **20,908** rows, 23 → 22 Grands Prix.
`laps_featured.parquet`: 68,122 → **66,924** rows.

**This moves a published headline.** The pit-stop position projection was quoted over 1,810
stops in 71 races; 84 of those stops were that one race scored twice.

| sample | races | stops | within one place | exact |
|---|---|---|---|---|
| as previously published | 71 | 1,810 | 86.5% | 59.1% |
| **corrected** | **70** | **1,768** | **86.3%** | **59.2%** |
| 2025 only | 24 | 552 | 86.1% | 59.6% |

### Las Vegas 2025 carries an IMPUTED circuit speed

FastF1 has no SpeedI2 reading for the entire 2025 Las Vegas race — 0% of 886 raw laps,
against 80% for SpeedI1 and 97% for SpeedFL. `mean_sector_speed` is the mean of those three
traps over laps where all three exist, so the circuit's value was missing on all 760 rows.
The reading does not exist; no re-run recovers it.

It is now filled with **232.83 km/h**, from the season's own two-trap mean (245.977) plus the
three-minus-two-trap gap measured at that circuit in its other seasons (−13.150).

Validated leave-era-out — hide a real value, impute it, compare:

| offset source | MAE | p95 | n |
|---|---|---|---|
| **the circuit's own other seasons** | **1.22 km/h** | 3.40 | 68 |
| averaged across all circuits | 9.44 km/h | 20.82 | 70 |

**Every affected row carries `mean_sector_speed_imputed = True`.** It is the only imputed
value in the dataset: 760 rows, Las Vegas 2025, and no unflagged missing circuit speed
remains anywhere.

The other three columns of that hole — `SpeedI2`, `Prev_SpeedI2`, `SpeedI2_Delta` — are left
missing on purpose. They are per-lap sensor readings, and there is no validated estimator for
760 individual ones.

## What was NOT regenerated

`laps_tiredeg.parquet` is unchanged. Its own Barcelona duplicate is baked into the shipped TCN
weights, and regenerating the artefact without retraining the model would only widen the drift
between them. Tracked, deferred, deliberate.
