# E1 — how long a real pit stop takes to repay itself

**Date:** 2026-07-31 · **Issue:** #763 · **Sample:** 1,377 real green-flag stops, **694 of them
elective**, across 2023-24. 2025 stays held out, the same hygiene the tyre reference used.

This is the measured basis for the deferral liability. It is a fact about the sport, derived
without consulting the 2025 agreement metric at any point — which matters, because the forbidden
loop in this epic is tuning a constant until that metric recovers.

---

## The question

`WINDOW_LAPS = 5` scores four candidates over five laps. A stop's cost lands inside that window;
its benefit accrues over the laps that follow. So: **over how many laps does a real stop actually
repay itself?** If the answer is comfortably inside five, the window is fine and #763 is wrong. If
it is far outside, no fixed short horizon can price a stop and the liability has to integrate to
the flag.

## Method, stated before the numbers so it cannot be adjusted afterwards

Per real stop, from the raw laps rather than from any model:

- **pit loss** = the stop lap's time minus the driver's median green-flag lap in the stint just
  left. What the stop actually cost on the road, not a modelled figure.
- **advantage** = the median of the new stint's first three clean laps against the median of the
  old stint's last three. The out-lap is excluded on both counts.
- **repayment horizon** = `ceil(pit_loss / advantage)`.

Quality gates are N04's own (`IsAccurate & ~Deleted`). A stop whose new set is not actually faster
never repays and is **reported as such rather than dropped** — dropping it would bias the median
toward the stops that happened to work.

## The result

| | all stops (n=1,377) | **elective (n=694)** |
|---|---|---|
| never repays (no positive advantage) | 13.1% | 10.8% |
| pit loss, median | 18.2 s | 18.1 s |
| advantage, median | 1.351 s/lap | 1.440 s/lap |
| repayment horizon p25 | 9 | 9 |
| **repayment horizon MEDIAN** | **14 laps** | **13 laps** |
| repayment horizon p75 | 21 | 19.5 |
| median 95% CI (bootstrap, 400) | [13, 14] | **[12, 14]** |
| **repays within 5 laps** | 9.7% | **15.0%** |
| repaid before the flag | 81.2% | 76.4% |

## Reading it

**A five-lap window can price at most a seventh of an elective stop's decision.** The median needs
**13 laps**; only **15.0%** of real elective stops repay inside five. The other 85% were being
compared against a horizon that ends long before their benefit arrives.

**The confidence interval is tight**, [12, 14] laps on 694 stops. That matters for the risk the
design gate flagged: the deferral liability multiplies `deg_cost_s` by the laps remaining rather
than by five, amplifying any error in the tyre reading four- to six-fold. A horizon whose median
moved by ±5 laps inside its own CI would not be safe to build on. ±1 is.

**10.8% of elective stops never repay at all.** That is not noise to be cleaned up, it is the honest
tail: teams stop for track position, traffic and Safety Car odds, not only for lap time. It is also
a reminder that agreement with the pit wall is evidence and not correctness — a tenth of the stops
this project grades the layer against had no lap-time case for existing.

## What this does NOT justify

**Not a wider `WINDOW_LAPS`.** A 13-lap median might read as "make the window 13", and the design
gate refuted that separately: widening is monotone in favour of pitting with today's arithmetic and
pushes first calls even earlier, which is the direction that had just cost five exact agreements.
The horizon measured here justifies **integrating the deferred cost to the flag**, which is what
`_deferral_tyre_liability_s` does, not stretching the window every candidate is scored over.

**Not a claim that 13 is a constant to hardcode.** Nothing in the implementation stores 13. The
liability runs to `laps_remaining`; this measurement is the evidence that a short fixed horizon
cannot be right, not a number the code reads.

## Reproducing

```bash
uv run python scripts/measure_repayment_horizon.py
```

Needs only `data/raw/2023/` and `data/raw/2024/`, so it does not run on CI. Runtime about two
minutes. It is committed rather than left in a scratchpad because this project's rule is that a
constant needs a measured basis, and a basis nobody can re-run is an assertion.

Related: `DESIGN_763_window.md` (the gate that refuted the original framing) ·
`MEASURE_744b_decision_effect.md` (the regression this explains) ·
`MEASURE_744a_tyre_reference.md` (the `deg_cost_s` whose error this amplifies)
