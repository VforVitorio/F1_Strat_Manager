# #744b measured: the tyre term made the decision WORSE, and the reason is not the term

**Date:** 2026-07-31 · **Issue:** #744 · **Instrument:** `f1-eval decision-modes`, six 2025 circuits.
**Comparison:** harness `99a663d` (before) against `d97a54e` (after). The only production diff
between them is #760's four files, so this is like-for-like on the same sample and the same metric.

---

## The result

| | without the term | with the term |
|---|---|---|
| Stops scored | 53 of 178 (29.8%) | 54 of 178 (30.3%) |
| **Exact lap** | **43.4%** (23 stops) | **33.3% (18 stops)** |
| Within 1 lap | 52.8% (28) | 44.4% (24) |
| Within 2 laps | 62.3% | 51.9% |
| **Declines (`no_call`)** | **82 (46.1%)** | **79 (44.4%)** |
| `no_boundary_in_window` | 22 | 24 |
| Mean signed error | −1.72 | −2.00 |

**Three fewer declines bought at the cost of five exact agreements.** The layer now calls the stop
earlier, which is the mechanism working exactly as designed — and further from what the teams did.

This is the outcome `project_epic724_scorecard` warned about in writing before the metric could see
it: *"it is entirely possible the epic traded too reluctant for too eager and the metric cannot tell
the difference."* It can now, and it did.

## The diagnosis, and it is NOT that the term is too big

Measured over 15,005 real 2025 laps (`scratchpad/wear_magnitude.py`, driving the committed
instrument with the season overridden), the charge the term applies to STAY_OUT:

| tyre life | positions charged |
|---|---|
| (3, 5] | 0.16 |
| (5, 10] | 0.86 |
| (10, 15] | 1.79 |
| (15, 20] | 2.79 |
| (20, 25] | 3.63 |
| (25, 100] | **5.43** |

Median across all laps: **2.03 positions**. What it competes with, in the same units:

```
the flat constant it replaced   0.833
a full window run past the cliff 2.667
one whole track position         1.000
the margin tie-break cap         0.200
```

**On 73.6% of laps the wear term alone moves STAY_OUT by more than a full track position**, and its
median is 2.4x the constant it replaced. It is not one input among several any more; it decides.

### Why that happens, and why scaling it down would be the wrong fix

The arithmetic is right. A set 0.61 s/lap off its fresh pace, held five more laps, costs 3.05 s. What
makes that decisive is the **other** side of the comparison: this scorer deliberately excludes the
pit-lane traversal, because a stop is mandatory under the two-compound rule and both pit-now and
pit-later pay it, so it cancels. The pit term is therefore the *physical stop* alone, ~2.8 s ≈ 1.87
positions.

So a five-lap wear charge and an entire pit stop are the same order of magnitude, and the wear tips
it. **Charging the true cost of five worn laps against a stop whose dominant cost has been cancelled
out is not a scale error in the term — it is the window.**

This is the third cause the original plan named and deliberately deferred:

> the 5-lap window prices 100% of a stop's cost against ~5 laps of its benefit (break-even ~92 laps).
> We re-measure after Sprint 3 and decide with data rather than moving three causes at once.

This document is that data.

## What must NOT be done next

**Do not tune the term until the metric recovers.** That is fitting to the metric, and it is exactly
the failure this epic retired `mean_signed_error` to avoid. Any change to the scale needs a measured
reason of its own.

**Do not read "agreement fell" as "the model got worse".** The report says it in its own scope
section: agreement with the real pit wall is evidence, not correctness. The teams optimise track
position, traffic and Safety Car odds that this layer does not model. A model that says "box now"
five laps before a team that was holding position may be arithmetically right and strategically
wrong for reasons outside its inputs.

**But do not use that as an escape either.** The structural finding stands on its own and does not
depend on the metric: one term deciding 73.6% of laps means the cliff, the undercut bonus, the
clean-air gain and the neutralisation hazard have stopped mattering to the argmax. That is a design
problem whatever the pit wall did.

## Recommendation

Keep the term. Reverting restores `FRESH_GAIN = 0.25`, a hardcoded constant that charges a 3-lap-old
set and a 25-lap-old set identically, and which this same measurement shows to be 2.4x too small at
the median and 6.5x too small at 25+ laps. It is not better, it is only quieter.

Open the `WINDOW_LAPS` question with this measurement attached, and treat it as the highest-risk item
in the layer: it moves every number and requires re-freezing goldens deliberately.

Related: `MEASURE_744a_tyre_reference.md` · `FABLE_G2_tyre_wear_term.md` ·
`MEASURE_752_metric_and_sample.md` · `documents/eval_reports/decision_modes.md`
