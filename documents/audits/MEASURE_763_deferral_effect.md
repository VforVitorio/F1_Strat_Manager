# E5 — what the deferral liability did, run once and reported as it came

**Date:** 2026-07-31 · **Issue:** #763 · Harness `f3f0207` (before) against `311f234` + the
deferral term (after). Same 2025 sample, same metric, one production change between them.

The design gate's rule for this measurement was **one run, published whatever it says, and if it
disappoints the design goes back to E1 and not to a knob**. This is that run.

---

## The headline

| | before | after |
|---|---|---|
| Stops scored | 54 (30.3%) | 58 (32.6%) |
| Exact lap | 33.3% (18) | 27.6% (16) |
| **Declines (`no_call`)** | **79 (44.4%)** | **65 (36.5%)** |
| `no_boundary_in_window` | 24 | 34 |
| Mean signed error | −2.00 | −2.34 |

**Fourteen fewer declines.** That is the largest single move any change in this epic has produced
on the number #715 was opened about.

## E3 — the invariance criterion, and it passes at the strongest level available

The term is scoped to plans whose mandatory stop is already discharged. If first-stop behaviour had
moved, the scoping leaked. Splitting the committed verdicts by whether the stop was the driver's
first of the race:

| | FIRST (n=89) | **ELECTIVE (n=89)** |
|---|---|---|
| scored | 38 → **38** | 16 → **20** |
| exact | 10 → **10** | 8 → **6** |
| `no_call` | 29 → **29** | **50 → 36** |
| `no_boundary` | 14 → **14** | 10 → 20 |
| decline rate | 32.6% → **32.6%** | **56.2% → 40.4%** |

**Not one first-stop verdict moved, and not one first-stop chosen lap changed.** Every bucket is
identical. The invariance is exact rather than within-tolerance, which is stronger than the gate
asked for.

**All fourteen recovered declines are elective stops**, which is the population the term was built
for and the one carrying 65% of the declines.

## The exact-agreement fall decomposes, and the decomposition is the whole story

The rate fell 33.3% → 27.6%, which reads like a regression until it is split:

- **First stops: 10 exact of 38 scored, before and after. Unchanged.** The half the layer was
  already good at is untouched.
- **Elective stops: 8 exact of 16 scored → 6 of 20.** The layer now attempts four more of them,
  and two that used to land exactly no longer do.

So the honest trade is **fourteen fewer declines against two lost exact agreements**, entirely
inside the elective population. The headline *rate* fell mostly because the denominator grew: 58
scored against 54, with a numerator two lower.

That is not the same finding as #744b's, where the tyre term moved ten first-stop offsets a lap
earlier and the mandatory half degraded. Here the mandatory half is provably inert.

## What this does NOT establish

**That the newly-called stops are well timed.** Twenty scored elective stops with six exact is a
weak hit rate, and `no_boundary_in_window` rose 10 → 20, meaning more elective stacks are now
already committed when the window opens. The layer has become much less reluctant about elective
stops and is not yet good at timing them.

**That agreement is correctness.** E1 measured that **10.8% of real elective stops never repay
themselves in lap time at all** — teams stop for track position, traffic and Safety Car odds this
layer does not model. A tenth of the events being graded against had no lap-time case for existing.

**Coverage is still `masked`** at 32.6% against the 60% gate, so every figure above describes the
subset where a decision is locatable.

## The arc across the epic, for the record

| state | declines | exact (count) |
|---|---|---|
| the metric fixed, no tyre channel | 82 (46.1%) | 23 |
| + measured tyre wear (#744b) | 79 (44.4%) | 18 |
| **+ the deferral liability (#763)** | **65 (36.5%)** | 16 |

And the figure that opened all of this, #715, measured **65% declined**. It is now **36.5%**,
against a mandatory-stop half that never moved.

Related: `MEASURE_763_repayment_horizon.md` (E1, the basis) · `DESIGN_763_window.md` (the gate that
refuted the original framing) · `MEASURE_744b_decision_effect.md`
