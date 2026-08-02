# GATE — TyreLife NaN root cause (issue #790, related #782 / #789)

**Date:** 2026-08-02 · **Investigator:** adversarial investigation gate (read-only)
**Question:** why does `laps_featured_2025.parquet` carry racing laps with `TyreLife` NaN, and is it a pipeline defect (fix the data) or genuine source absence (guard the consumers)?

Findings are appended incrementally as they are confirmed with executed evidence.

## Checklist

- [ ] 1. Independently re-measure the gap (2025, and 2023/2024)
- [ ] 2. Trace one block (Miami) backwards: featured → augmentation → raw parquet
- [ ] 3. Check the source (FastF1 cache / API) for the same laps
- [ ] 4. Cross-reference the lap ranges against race events (SC / red flag / rain)
- [ ] 5. Blast radius: other columns NaN in the same blocks; relation to #782 weather gap
- [ ] 6. Recommendation with trade-offs

---

## Finding 1 — the gap re-measured (CONFIRMED, with corrections to the sweep)

Executed: pandas over `data/processed/laps_featured_{2023,2024,2025}.parquet`.

**2025 — 451 rows with `TyreLife` NaN and `Position` present (sweep number confirmed exactly):**

| GP | rows | laps | contiguous | drivers | share of field on those laps |
|---|---|---|---|---|---|
| Miami | 379 | 4–24 | yes | 19 | 379/388 rows (~whole field) |
| Spa-Francorchamps | 70 | 28–44 | yes | **5** | 70/328 rows (**NOT field-wide** — sweep said 53 rows; it is 70) |
| Melbourne | 2 | 42–43 | yes | 1 | 2/32 rows |

Compound in the 451 gap rows: `'nan'` 379 (all Miami), `'None'` 53 (Spa), **`'HARD'` 17 (Spa)**, `'MEDIUM'` 2 (Melbourne). So 19 Spa/Melbourne rows have a **real compound string but TyreLife NaN** — the sweep's claim "every Compound-nan row is also TyreLife-NaN" holds (432/432), but the converse does not: TyreLife can be NaN with a valid Compound.

**Not 2025-only:** 2023 has 35 such rows — Montréal, ONE driver, laps 36–70 contiguous to race end, all `Compound='None'`. 2024 has **0**.

**Column counts:** 2023 = 53 cols, 2024 = 53 cols, 2025 = **48 cols** — consistent with #782 (weather columns absent from the 2025 featured parquet only).

## Finding 2 — the NaN is ALREADY in the raw parquets; the featured step introduces nothing (CONFIRMED)

Executed: pandas over `data/raw/2025/{Miami_Gardens,Spa-Francorchamps,Melbourne}/laps.parquet` and `data/raw/2023/Montréal/laps.parquet`.

| Race (raw) | TyreLife NaN rows | shape of the hole |
|---|---|---|
| Miami 2025 | 446/1005 | **ALL 19 drivers, laps 1–23 complete + 9 drivers on lap 24; `Compound` AND `Stint` are NaN on every one of those rows.** Zero NaN after lap 24. |
| Spa 2025 | 75/879 | 5 drivers, growing per driver from L27 (1 driver) to L33+ (5 drivers) to race end (L44). All 75 rows are `Stint == 3.0`. 57 rows `Compound='None'`, **18 rows `Compound='HARD'` with TyreLife still NaN**. |
| Melbourne 2025 | 5/927 | 1 driver, laps 40–44, all `Stint == 5.0`, `Compound='MEDIUM'` known, TyreLife NaN. |
| Montréal 2023 | 35/1317 | 1 driver, laps 36–70 (to race end), all `Stint == 3.0`, `Compound='None'`. |

Featured-vs-raw delta is pure N04 row filtering (Miami raw hole is L1–24; featured shows L4–24 because L1–3 are dropped rows — TrackStatus `'12'`/`'126'`/`'671'` = start + VSC), not value mutation.

**The hole is stint-shaped, not lap-shaped.** In every case the NaN block is "driver X's stint N has no tyre metadata, from the stint's first lap to its last". Miami is the degenerate case where the affected stint is **stint 1 of the entire field** (the live-timing tyre-stint feed evidently missing at race start; every driver's data appears the moment they make their first stop). This is the signature of missing `TimingAppData` stint records in the F1 live-timing source that FastF1 builds tyre data from — NOT of a merge/join bug in this repo's pipeline.

**TrackStatus cross-check (kills the SC/red-flag theory for the main block):** Miami laps 4–27 are TrackStatus `'1'` (all clear) for the entire field — the hole spans green-flag racing. Spa laps 25–44 all `'1'`. The hole does NOT line up with neutralisation windows; it lines up with stint boundaries.

---

*The investigation agent stalled after Finding 2 (report unchanged, no FastF1 cache writes for 45+
minutes, empty transcript, no completion notification). Findings 3-6 below were executed by the
orchestrating session directly. Findings 1-2 above are the agent's and were left as written.*

## Finding 3 — FastF1 has exactly the same gap; our extractor loses nothing (CONFIRMED)

Executed: `fastf1.get_session(2025, "Miami", "R").load(laps=True)` against the project's own 1.9 GB
cache, compared row for row with `data/raw/2025/Miami_Gardens/laps.parquet`.

| | FastF1 (source) | Our raw parquet |
|---|---|---|
| `TyreLife` NaN, laps <= 24 | 446 / 457 | 446 / 457 |
| `Stint` NaN, laps <= 24 | 446 / 457 | 446 / 457 |
| `TyreLife` NaN, laps > 24 | 0 / 548 | 0 / 548 |

Identical. **The systematic-extraction-failure hypothesis is REFUTED.** The F1 live-timing
`TimingAppData` feed did not publish stint records for the opening stint, FastF1 leaves NaN rather
than guessing, and our extractor copies that faithfully. FastF1 even logs that it repaired three
drivers (`Fixed incorrect tyre stint information for driver '31'/'18'/'5'`) — those repairs are the
11 non-NaN rows.

## Finding 4 — the far more serious half: the laps AFTER the gap are WRONG, not missing (CONFIRMED)

The NaN block is the visible symptom. The block after it is silently corrupt.

NOR, Miami 2025, from FastF1 (`PitInTime`/`PitOutTime` are the independent ground truth, and they
are present and correct):

```
LapNumber  Stint  Compound  TyreLife  pit_in  pit_out
     24.0    NaN       nan       NaN   False    False
     25.0    1.0    MEDIUM       1.0   False    False
     29.0    1.0    MEDIUM       5.0    True    False    <- real pit stop
     30.0    1.0    MEDIUM       6.0   False     True    <- keeps counting THROUGH the stop
     32.0    1.0    MEDIUM       8.0   False    False
     33.0    2.0      HARD       1.0   False    False    <- stint 2 starts 4 laps late
```

Three separate errors in one block:

1. **`TyreLife` restarts at 1 on lap 25** while the car is still on the tyre it started the race on.
   True age at lap 29 is 29 racing laps plus its starting age; FastF1 reports **5**.
2. **It counts across a pit stop** — physically impossible, and proof the value is derived from
   "when the feed appeared", not from the tyre.
3. **The stint boundary is misplaced by 4 laps**: `Stint` stays 1 through lap 32 when the real
   change happened at lap 29/30.

`TyreLife` is a direct input to N26's TCN and to N15/N16, and Miami 2025 is in the **test** season.
A degradation model reading "5 laps old" for a 29-lap-old tyre predicts almost no wear.

## Finding 5 — recomputation is sound, because the pit stops are intact

The one thing NOT corrupted is `PitInTime`/`PitOutTime`. That is enough to rebuild both columns:

- **Stint boundaries** derive from pit in/out — exact.
- **`TyreLife` within a stint** = laps completed since the stint began — exact for stints 2+, where a
  fresh set starts at 0.
- **Stint 1 only** carries an unknown starting offset (the race-start set may have qualifying laps on
  it). Measured on a clean control race (Bahrain 2025 lap 1): the offset is real and varies per
  driver, `TyreLife` 4.0 on used sets versus 1.0 on fresh ones. So stint 1 is recoverable up to a
  bounded offset of roughly 0-4 laps.

Compare the alternatives on the reference lap: today the model receives **5** where the truth is
**~29**. A recomputation gives **29 plus or minus 4**. The residual uncertainty is an order of
magnitude smaller than the error it replaces, and unlike the current value it is not physically
impossible.

`FreshTyre` cannot rescue the offset: on all 446 gap rows it is uniformly `True`, while on the 11
rows FastF1 repaired it varies (`False` for ALO, STR, HAD). Uniform where the data is missing and
varying where it is present is the signature of a fabricated default, not a reading — so it must not
be used as evidence of a fresh set.

## Recommendation

Rebuild `Stint` and `TyreLife` from the pit-stop timeline in the extraction layer, for every season,
gated behind a validation that the rebuild reproduces FastF1's values exactly on races where the feed
was healthy. Leave stint 1's absolute offset explicitly unknown rather than assuming a fresh set.

This fixes #790's NaN (the visible half) and Finding 4's wrong values (the invisible, worse half) in
one place, upstream of every consumer. It does not touch #782's weather gap, which is a genuinely
different missing-column problem.

**What it could get wrong:** a rebuilt `TyreLife` that is subtly off is worse than a NaN, because a
NaN is visible and a wrong integer is not. The validation gate is therefore not optional — the
rebuild must be proven to reproduce healthy races byte for byte before it is allowed to touch
unhealthy ones.
