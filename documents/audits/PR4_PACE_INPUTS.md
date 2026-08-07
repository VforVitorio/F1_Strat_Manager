# PR 4 — the two pace inputs that were a different quantity from the trained column

**Date:** 2026-08-04 · Extends `GATE_DATA_WIRING.md` (W-F1, W-F4) and `GATE_801_ARTEFACTS.md`
§4-§5. Those are dated findings and stay as written; this records what the pre-implementation
verification measured, including where the gate's reasoning had to be re-derived.

## W-F1 — CompoundID served on the wrong scale

### Verifying the claim, and the wrong turn on the way

`_load_encoding_maps` read `manifest["categorical_encoding"]["Compound"]`, which is
**0-based** (SOFT 0 … WET 4), and `_build_feature_row` put the result in the **`CompoundID`**
column. The parquet's `CompoundID` is **1-based** (SOFT 1 … WET 5, `.nb_py/N01:114`).

The first check appeared to refute this. `.nb_py/N06:87,130` reads that same manifest block
and maps the `Compound` string through it, which reads as the model having eaten the 0-based
codes. **It did not.** The manifest's own `features_in` lists the 39 columns the model
consumed: `CompoundID` is among them and `Compound` is not. The notebook's mapping produces a
column that never reaches the feature matrix.

The eval harness had this right the whole time, in a docstring:

> *Only `FreshTyre` feeds the delta model directly; `Compound` / `race_phase` are encoded for
> parity with the notebook (the model consumes the pre-numeric `CompoundID`).*
> — `src/strategy/eval/pace_holdout.py:48`

So the eval was correct and inference contradicted it. `pace_holdout` needed no change — the
twin that looked most likely was not one.

### Why it is worse than an off-by-one

N01 does `.fillna(0)`, so **0 is a trained class meaning "no compound reported"**. Served
0-based, a SOFT lap arrives as that class. The two are then indistinguishable to the model:
this is the sentinel collision this repo keeps re-finding, not a relabelling.

The old default compounded it: `.get(compound, 1)` returned 1 for an unrecognised string, and
1 is SOFT on the trained scale — an unknown tyre was served as a specific one.

### Fix

`_N01_COMPOUND_ID` in `pace_agent`, with `_COMPOUND_ID_UNKNOWN = 0` as the default.
`tests/agents/test_pace_inputs.py` re-derives both from the parquet rather than from N01's
source, because the parquet is what N06 read.

The manifest block itself is left alone. It is not wrong — it correctly describes N06's
`encode_features` step — it simply describes a column the model dropped. Reading it for the
feature encoding was the defect.

## W-F4 — Prev_SpeedST served the current lap's trap

N04 builds every `Prev_*` column in one loop over the same grouped shift
(`.nb_py/N04:389-392`): `groupby(['Year','GP_Name','DriverNumber','Stint']).shift(1)`. The
pace agent had no previous trap available and passed `d.get("speed_st") or 300.0` — THIS
lap's reading — as `prev_speed_st`.

This is the same defect #435 fixed for `Prev_LapTime`, in the same call, left in place for its
sibling. #435's fix added `_precompute_prev_lap_times` to `RaceStateManager`, which reproduces
N04's rule including the part that is not guessable: the previous lap is the previous
**surviving** lap under `filter_baseline_laps`, not `lap_number - 1`, so an out-lap never
becomes the anchor.

### Fix

`_precompute_prev_lap_times` becomes column-parametric (`_precompute_previous`) and
`_precompute_prev_speed_traps` reuses it. Two reconstructions of one transform is how the pair
drifts apart the next time either is corrected.

The `or 300.0` is removed rather than moved. 300 km/h sits **inside** the trained range
(156-362), so an invented reading was indistinguishable from a measured one, and it fired on
the first lap of every stint — exactly where the answer is genuinely unknown. NaN says
unknown, and XGBoost reads a missing feature natively.

## Measured

Served value against the trained column, whole races, NOR 2025:

| | Lusail | Miami |
|---|---|---|
| `CompoundID` wrong, before | **46/46 laps (100%)** | **49/49 (100%)** |
| `Prev_SpeedST` wrong, before | 27/27, mean 6.67 km/h, max 20.0 | 22/27, mean 3.07, max 17.0 |
| both, after | **0** | **0** |
| stint openers served NaN where training has none | 19/19 | — |

Input parity is not the outcome that matters. The prediction itself, both fixes together:

| race | laps whose prediction moves | mean | p95 | max |
|---|---|---|---|---|
| Lusail | 9 of 57 (16%) | 0.149 s | 0.875 s | **2.887 s** |
| Miami | 4 of 57 (7%) | 0.104 s | 0.265 s | **3.235 s** |

100% of inputs wrong yields 7-16% of predictions moved because `CompoundID` carries 0.38% of
N06's gain — the trees split on it only in some regions. Where they do, the error reaches
three seconds, which is a pit-stop's worth of lap time.
