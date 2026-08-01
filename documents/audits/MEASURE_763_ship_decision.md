# #763 — the ship decision, and the error bound that decides it

**Date:** 2026-08-01 · Branch `feat/deferral-tyre-liability` @ `110f4ed`

**Verdict: do NOT merge.** The deferral liability is correct, its effect is real and
targeted, and the input it consumes is not measured well enough to carry it at the horizon
it uses. The blocker is now a number rather than an intuition.

---

## The bound that was missing all epic

E4 — the design gate's sensitivity disclosure — has been **formally unevaluable** since #744a,
because `deg_cost_s` never had a published per-lap error bound. Measured now against N04's own
target on the same 31,624 training-season laps the reference was built from:

| | |
|---|---|
| median absolute error | **0.261 s/lap** |
| mean absolute error | **0.650 s/lap** (95% CI [0.623, 0.678]) |
| signed bias | **+0.351 s/lap** |
| laps with error under 0.1 s/lap | **21.9%** |

By tyre-life band, and this is the part that decides it:

| tyre life | median abs error |
|---|---|
| 3-10 | 0.209 |
| 10-20 | 0.256 |
| 20-30 | 0.308 |
| **30+** | **0.557** |

## Why that disqualifies the term

**The amplification table understates the problem.** It was measured perturbing ±0.1 s/lap,
which flips 7.1% of elective decisions. But 0.1 s/lap is **smaller than the model's typical
error**. The real error sits between the ±0.2 row (13.1% flips) and the ±0.5 row (26.8%), so
under the error the model actually makes, **one in four to one in eight elective decisions is
decided by noise**.

**The error grows exactly where the term is charged hardest.** The liability multiplies
`deg_cost_s` by the laps remaining, so a 45-lap-to-go stint carries the 30+ band's 0.557 s/lap
over 40 laps.

**And the bias is not noise.** +0.351 s/lap systematic, integrated over ~40 remaining laps, is
**~14 s of phantom cost** — the same order as the 22.8 s pit loss the term exists to compare
against. A systematic offset that large does not average out over draws; it moves the argmax in
one direction on every lap.

**The behaviour also moved the wrong way under a correct fix.** Removing the double q_f discount
was right, and the exit gate found no defect in the reasoning — but the amplification went 5.91×
→ 6.82×, against a written prediction that it would fall. A term whose error behaviour surprises
you when you fix a real bug in it is measured but not understood, and that is a worse state to
ship from than "understood, awaiting a bound".

## What the term DOES buy, so the negative result is not read as a dead end

On the same 2025 sample, harness `110f4ed`:

| | before | after |
|---|---|---|
| declines (`no_call`) | 79 (44.4%) | **66 (37.1%)** |
| scored | 54 | 57 |
| exact | 33.3% | 28.1% |

**Thirteen fewer declines, every one an elective stop**, with the mandatory-stop half provably
inert (E3: not one first-stop verdict moved, not one chosen lap changed). The mechanism works.
It is the input that cannot carry it.

## What unblocks it

Reduce `deg_cost_s`'s error, or remove its bias. Either makes E4 evaluable and this decision
re-runnable in one command. **The bound belongs to `deg_cost_s` as a class, not to this term** —
`driver_time_delta` already consumes the same biased input on the shipped path, at 5 laps of
exposure rather than 40. That is 1/8 the amplification and the same systematic offset, and it is
on `main` today.

---

## Separate finding: what a lap of offset actually costs

The report leads with EXACT-lap agreement, and the reasonable objection is that matching the
exact lap demands reproducing a call made with radio, tyre temperatures and rival intel this
layer does not have — so a ±3 lap band might be the fairer headline.

**Measured on 1,551 real stops, it is not.** The median fresh-set advantage is **1.137 s/lap**,
and the measured green-flag gap between consecutive cars is **2.227 s** (n=69,487):

| offset | seconds | **track positions** |
|---|---|---|
| 1 lap | 1.14 | **0.51** |
| 2 laps | 2.27 | **1.02** |
| 3 laps | 3.41 | **1.53** |
| 5 laps | 5.69 | **2.55** |

**It takes 1.96 laps of offset to lose a whole track position.** So a ±3 band would be accepting
calls that cost a place and a half — not "the same strategic window", a materially worse race
result.

**The estimate is conservative in both directions that matter.** It prices offset purely in pace
terms and ignores traffic on the rejoin, which is the main reason teams care about exact timing
and would make offsets cost more. And 2.227 s is the pooled median; in a tight midfield the gap
is smaller and a lap of offset costs more, not less.

**Conclusion: exact-lap and within-1 are the right resolution**, and the report keeps them. That
was the existing framing, but it now has a measured basis rather than being the default.

## Reproducing

```bash
uv run python scripts/measure_deg_error_bound.py     # the bound
uv run python scripts/measure_offset_cost.py         # the resolution
```

Related: `DESIGN_763_window.md` · `FABLE_763_deferral_exit.md` ·
`MEASURE_763_repayment_horizon.md` · `MEASURE_744a_tyre_reference.md`
