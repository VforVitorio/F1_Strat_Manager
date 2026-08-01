# Fresh-reference quality gate — root cause and measured effect

**Date:** 2026-08-01 · **Instrument:** `scripts/measure_fresh_reference_gate.py`
**Sample:** 31,624 laps at tyre life > 3, across 1,714 stints with a usable fresh
reference, seasons 2023-24 (training only).

**Result: `deg_cost_s`'s error bound (0.650 s/lap mean absolute, +0.351 s/lap signed
bias — `feat/deferral-tyre-liability`, `MEASURE_763_ship_decision.md`) is not evenly
spread across laps. A handful of stints have a contaminated fresh-reference lap, and
gating that lap on race pace cuts mean absolute error to 0.434 s/lap and signed bias
to +0.139, at a cost of 49 of 1,714 stints (2.9%) losing their reference and falling
back to `None` → `FRESH_GAIN`.**

---

## Where the epic's own bound was measured, and what it didn't decompose

`MEASURE_763_ship_decision.md` published the aggregate bound and, correctly, said the
bound belongs to `deg_cost_s` as a class — `driver_time_delta` already consumes the
same input on `main` today. It did not ask WHERE inside that population the error
concentrates. This note answers that question and ships a fix for the part of it that
is a data-quality gap rather than a property of the model.

## The mechanism

`deg_cost_s = cumulative_deg_s(now) − cumulative_deg_s(fresh reference)`. The fresh
reference is a second deterministic TCN pass over the stint's own early laps
(`TireAgent._fresh_reference`, tyre life ≤ 3). Grouping the raw per-lap residual
(`pred − target`) by tyre-life band shows almost no drift — mean +0.03 to +0.08 s/lap
beyond the first few laps, correlation with tyre life +0.04 to +0.11 per compound.
**The model's own predictions are not what grows.**

What grows is the **fresh-reference residual**, and only for the subset of stints that
happen to run long: grouped by each stint's own max tyre life, the per-stint
fresh-reference residual is essentially flat in the MEDIAN (0.01 to 0.04 s/lap across
every band) but the MEAN at 30+ laps is **−0.691**, dragged there by a tail — one stint
at **−53.989 s** and another at **−46.454 s**, both from **Mexico City 2023**; five
more between −4 and −8 s from **Monaco 2024** and **Melbourne 2023**.

Pulling the full lap sequence for the worst case (car 4, Mexico City 2023):

| LapNumber | Stint | TyreLife | LapTime_s | FuelAdjustedDegAbsolute |
|---|---|---|---|---|
| 36 | 4 | 3 | **137.757** | 0.000 (the baseline lap) |
| 37 | 4 | 4 | 84.243 | **−53.459** |
| 38 | 4 | 5 | 83.841 | **−53.806** |

Green-flag pace at this circuit is ~83 s. Lap 36 is a Safety-Car/red-flag-affected lap
that FastF1 apparently did not mark inaccurate. N04's target is defined relative to
*this stint's own baseline lap* — here, that anomalous 137.8 s lap — so every
subsequent lap reads as ~54 s "faster than fresh," which is not tyre wear, it is the
zero point being wrong by 54 seconds.

**`track_status_clean` — the column that should catch this — is dead.**
`_add_session_cols`'s own docstring already says so: it is a constant 0 across every
row of every featured parquet, because N04's `IsAccurate` gate does not catch every
neutralised lap. This stint's lap 36 is the counter-example that falsifies the
docstring's stated reason ("every lap that survives is genuinely green"): it survived,
and it is not green by any reasonable reading of 137.757 s at a ~83 s circuit.

Two more things worth recording, both checked and both negative:

- The model's prediction on the **current** (non-reference) lap is not similarly
  contaminated: additionally dropping laps whose own `pct_of_fastest` exceeds the
  threshold removes only 0.3% of scored rows and moves the bound by <0.01 s/lap.
- `LapsSincePitStop == 0` (a literal pit-exit out-lap) does not co-occur with these
  reference laps in the training data — this is a caution-period artefact, not an
  out-lap artefact, and the existing docstring's second stated cause ("an out-lap or a
  standing start") is not what this measurement finds driving the tail.

## The fix, and why this signal and not a new one

`lap_time_pct_of_race_fastest` is already one of the TCN's 42 input features,
computed unconditionally in `_add_session_cols` from `session_meta['fastest_lap_s']`
— available at exactly the point `track_status_clean` should have been, at zero new
data cost. Mexico City 2023's lap 36 reads **1.694** on it; the population of 1,822
training-season fresh-reference laps has median **1.046**, p95 **1.181**, p99
**1.464** — a long, well-separated tail.

`TireAgent._fresh_reference` now drops any candidate lap above
`cfg.fresh_reference_max_pct_of_fastest` (default **1.10**) before building the
reference tensor, via the pure, leaf-level `_reject_contaminated_laps`. If every
candidate is rejected, it returns `None` — never falls back to a contaminated lap —
matching the doctrine `_referenced_wear` already applies to a missing half: a wrong
reference is worse than none.

## Threshold sweep (why 1.10, not looser or tighter)

| threshold | mean abs error | signed bias | stints losing their reference |
|---|---|---|---|
| none (baseline) | 0.650 | +0.351 | 0 |
| **1.10 (shipped)** | **0.434** | **+0.139** | 49 (2.9%) |
| 1.15 | 0.478 | +0.183 | 20 |
| 1.20 | 0.502 | +0.208 | 16 |
| 1.25 | 0.534 | +0.242 | 12 |

1.10 gives the largest error reduction for a coverage cost that stays under 3%.
Tighter thresholds recover fewer stints' worth of contamination for a shrinking
marginal gain; looser ones leave more of the tail in.

## What this does NOT claim

**The bound is reduced, not closed.** 0.434 s/lap mean absolute error is still above
the 0.1 s/lap perturbation `MEASURE_763_ship_decision.md`'s amplification table was
measured at, and +0.139 s/lap of remaining bias, integrated over a long stint, is
still a real number. This note does not reopen the #763 ship decision — `deg_cost_s`
still needs a smaller bound before the deferral liability term can carry it, and nothing
here changes that verdict. What it does is remove a *known, root-caused, unrelated-to-
model-capacity* contribution to that bound, which was sitting inside the number
`MEASURE_763_ship_decision.md` published without being separated out.

**This is a genuine, if partial, improvement to `deg_cost_s` as it exists on `main`
today** (the term `driver_time_delta` already consumes, at 5 laps of exposure), not
only to the parked deferral term. The 33% mean-error reduction and 60% bias reduction
apply to production `deg_cost_s` the moment this ships, independent of #763/#771.

## Reproducing

```bash
uv run python scripts/measure_fresh_reference_gate.py
```

Related: `MEASURE_744a_tyre_reference.md` · `MEASURE_763_ship_decision.md`
(`feat/deferral-tyre-liability`) · `scripts/measure_deg_error_bound.py`
(same branch, the original bound this note decomposes)
