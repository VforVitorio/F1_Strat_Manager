# #744a — what a fresh-tyre reference is actually worth

**Date:** 2026-07-30 · **Issue:** #744 · **Instrument:** `scripts/measure_tyre_reference.py`
**Sample:** 31,624 laps at tyre life > 3, across 2,343 stints, seasons 2023-24 (training only).

**Result: the artefact #744a asked for should not be built.** A pooled per-compound reference is
worth **+0.02 s/lap** against doing nothing, on a quantity whose zero point scatters between stints
with a standard deviation of **5.48 s**. The reference that does work is the one this project
**already built and reverted** — its refutation does not survive a proper sample.

---

## The harness, and why its numbers can be trusted

Predictions come from the training parquet (`laps_tiredeg.parquet`, which already carries all 42
features) pushed through the same scaler and the same N09 left-ZERO-pad that
`TireAgent._build_stint_tensor` applies. Two independent checks say that is the right transform:

1. **`corr(pred, target) = 0.977`** over all 42,920 predicted laps. A transform that had drifted from
   the model would not reproduce the model's own training target. The script asserts this and raises
   below 0.90, so the guard is permanent rather than a one-off.
2. **No feature is a frame aggregate.** AST-scanning the ten `_add_*` builders in `tire_agent.py`
   for `mean/std/min/max/sum/median/rolling/expanding/cumsum/quantile/var` returns **none** — every
   feature is row-local, backward-looking, or read from `session_meta`. So slicing a stint's
   precomputed features gives the same rows that recomputing them over the prefix would, which is
   what lets this run offline instead of reloading 47 FastF1 sessions.

## The measurement

Four candidates, scored on #744's own two criteria plus a rank correlation. Spearman rather than
Pearson because the quantity has a heavy tail — see below — and a handful of stints would otherwise
decide the number.

| reference | non-negative | spearman | pearson | median wear |
|---|---|---|---|---|
| `pooled` — one median per compound, **the artefact #744a proposed** | 66.3% | +0.188 | −0.055 | 0.519 |
| `stint_first` — this stint's prediction on its first lap | 71.2% | +0.269 | +0.084 | 0.514 |
| `stint_le3` — this stint's median prediction at tyre life ≤ 3, **the reverted design** | **73.0%** | **+0.295** | +0.090 | 0.489 |
| `none` — the raw level, no reference at all | 65.3% | +0.191 | −0.054 | 0.519 |

Median wear by tyre-life band, which is where every candidate looks fine:

| band | pooled | stint_le3 | none |
|---|---|---|---|
| (3, 5] | 0.045 | 0.069 | 0.023 |
| (5, 10] | 0.284 | 0.284 | 0.245 |
| (10, 15] | 0.591 | 0.569 | 0.554 |
| (15, 20] | 0.860 | 0.823 | 0.836 |
| (20, 25] | 1.021 | 1.029 | 0.992 |
| (25, 100] | 0.810 | 0.941 | 0.785 |

## Reading it

**The pooled reference does nothing.** Compare its row against `none`: 66.3% vs 65.3% non-negative,
Spearman +0.188 vs **+0.191** — a hair *worse* — and band medians that differ by 0.02-0.03 s. It is a
per-compound constant, so by construction it cannot change any ranking within a compound; the only
thing it can move is the zero crossing, and it moves it by two hundredths of a second.

**The reason is measurable.** `FuelAdjustedDegAbsolute` is defined against *each stint's own*
baseline lap, and those baselines are not similarly biased:

- the **per-stint median target has std 5.48 s**, with a 1st percentile at **−32.87 s**;
- pooled, `corr(target, tyre_life) = −0.088` — the degradation signal is **absent** across stints;
- **within** a stint, the median `corr(target, tyre_life)` is **+0.577**, positive on **73.8%** of the
  2,194 stints long enough to correlate.

So the signal is entirely within-stint, and the between-stint scatter is **13× the 0.411 s/lap swing**
#744 wants to capture. No single constant per compound can normalise 2,343 different zero points.
This is a stronger statement than the one `DESIGN_S3_option_b.md` made: that document assumed the
per-stint baselines were *similarly* biased slow ("an out-lap or a standing start: cold and slow"), so
that one pooled shift would correct them all. The data says they scatter over tens of seconds.

**The tail is in the data, not in the model.** Predictions reach −65.3 s/lap; the training *target*
reaches −65.3 s/lap on the same laps. Those are stints whose baseline lap was run under a Safety Car,
a red flag, or as an out-lap. This is why Pearson is useless here and why any consumer of the per-lap
value needs a bound.

**The (25, 100] dip is a population change, not a model failure.** Stints that reach 25+ laps are the
low-degradation ones — hard compounds in low-deg races — so the band draws from a different
population than the ones below it. Every candidate dips there, including `none`.

⚠️ **Say the consequence out loud, because it is a formal acceptance box on #744.** That dip means
`monotonic_bands` is **False in-sample for what shipped**, so the "monotonic by tyre-life band" half
of the criterion is strictly unmet on 2023-24. The mitigation above is real, and it is a mitigation,
not a pass. Gate G2 found that #760 had also replaced the `monotone` column in the printed report
with `p1`/`p99` in the same commit — not deliberately, but the effect was that the one place stating
the criterion stopped stating it. The column is back.

## The criterion PASSES on 2025, and the reference is roughly twice as strong there

Measured by gate G2 with the same instrument, overriding only the season (15,005 laps):

| | training 2023-24 | **2025** |
|---|---|---|
| non-negative | 73.9% | **83.7%** |
| Spearman | +0.308 | **+0.603** |
| monotonic by band | **False** | **True** |

This was an adversarial attack that expected the opposite: 2025 is the test season, excluded from the
measurement precisely so no constant could be fitted to it, so the honest prior was that an
out-of-sample check would degrade. It improves. The reference is strongest on the season the layer
actually serves, and the same direction showed up independently in the decision metric
(`MEASURE_752_metric_and_sample.md`), where moving the sample to 2025 was worth +12.8 points of exact
agreement. **No cause is claimed for either.**

## What this means for the reverted design

`DESIGN_S3_option_b.md` reverted the same-stint reference on **110 laps at one race**: 64.8%
non-negative, correlation +0.291 against the raw level's +0.369. On 31,624 laps it measures **73.0%
non-negative and Spearman +0.295 against the raw level's +0.191** — better than the raw level on both
criteria, and better than the numbers it was reverted on.

Two caveats, stated because the conclusion rests on them:

- **The comparison that reverted it was not like-for-like.** +0.291 was compared against +0.369, but
  +0.369 was itself measured over those same 110 live laps at one circuit. Pooled over the training
  seasons the raw level correlates **+0.191**, so the reverted design was never actually behind.
- **This measurement uses the parquet's precomputed features; the reverted implementation recomputed
  them from raw laps over a 3-lap frame.** The AST scan above says those must agree, since no feature
  is a frame aggregate — but the reverted code path was never measured at this scale, so #744b should
  confirm the live values match before consuming them.

## What #744a should be

Do not commit a pooled per-compound table. There is no `data/tire_reference_v1.json` in this PR, and
no accessor, because a committed constant is a permanent claim and this one measures as noise.

The instrument stays: `scripts/measure_tyre_reference.py` re-runs the comparison whenever the TCN is
retrained, and its self-check fails loudly if the transform ever drifts from the model.

**#744b consumes `stint_le3`**, which needs no artefact — it is a second forward pass on the stint's
own early laps. Its open questions are consumption-side, not measurement-side:

1. the per-lap value needs a bound (the tail reaches ±15 s on real laps after referencing);
2. 27% of laps still price negative, so the sign convention has to be decided rather than assumed;
3. `FRESH_GAIN = 0.25` is the same quantity hardcoded and must be replaced, not added to;
4. both scorers, including the legacy one the backend endpoint runs in production.

## Reproducing

```bash
uv run python scripts/measure_tyre_reference.py --out /tmp/tyre_ref.json
```

Needs `data/processed/laps_tiredeg.parquet` and `data/models/tire_degradation/`, so it does not run
on CI. Runtime ~4 min on CPU.

Related: `DESIGN_S3_option_b.md` · `MEASURE_S3_tyre_channel.md` · `src/strategy/eval/hygiene.py`
