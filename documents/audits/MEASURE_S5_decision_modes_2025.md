# Sprint 5 — re-measuring the decision layer, and a defect in the metric itself

**Date:** 2026-07-30 · **Issue:** #729 · **Status:** the decline rate improved and is verified; **the
timing metric is an artefact of its own window and cannot answer the `WINDOW_LAPS` question**

Sprint 5 was specified as: re-run `f1-eval decision-modes`, compare with the committed baseline, and
decide `WINDOW_LAPS` with a number. Two things happened on the way.

---

## 1. The sample was redesigned first, and it needed to be

The committed baseline runs on six races: 2023 Barcelona, 2023 Monaco, 2024 Silverstone, 2024
Marina_Bay, 2025 Lusail, 2025 Monza. **Four of the six are the training seasons.** Every model the
decision layer consumes (N06, N26, N27, N15, N16) trains on 2023-24 and tests on 2025, so on those
four races the layer is fed in-sample predictions it will never see in production.

The direction matters: in-sample the models are more accurate, so the layer should decline *less*.
The committed 64.6% is therefore likely **optimistic**, which is the direction nobody audits because
it flatters nobody.

Redesigned as **the same six circuits, all in 2025**. That preserves the baseline's archetype spread
(street, high-speed, low-downforce, high-degradation) and leaves the season as the only variable. All
six exist under `data/raw/2025/`.

## 2. The decline rate genuinely improved

| | baseline (mixed seasons) | 2025-only |
|---|---|---|
| eligible real green-flag stops | 198 | 178 |
| **declined (`no_call_in_window`)** | **64.6%** | **46.1%** |
| scored (coverage) | 40 (20.2%) | **75 (42.1%)** |
| rail-blocked | 30 | 21 |

Per race:

| race | n | scored | declined | rails | decline % |
|---|---|---|---|---|---|
| Barcelona | 41 | 25 | 15 | 1 | 36.6% |
| Marina_Bay | 23 | 16 | 7 | 0 | 30.4% |
| Monza | 20 | 11 | 7 | 2 | 35.0% |
| Monaco | 37 | 14 | 17 | 6 | 45.9% |
| Lusail | 26 | 8 | 14 | 4 | 53.8% |
| **Silverstone** | 31 | **1** | 22 | 8 | **71.0%** |

### It is causally ours, not the sample

The season change is a confound for the headline, so the causal claim rests on the two races present
in **both** runs — same circuits, same season, same stops, matched per stop on
`(year, race, driver, actual_lap)`:

```
Lusail   26 stops matched, 5 moved bucket:  no_call -> scored  4
                                            scored  -> no_call  1
Monza    20 stops matched, 7 moved bucket:  no_call -> scored  7
                                            (none the other way)
```

**Net +10 stops became callable on 46 matched stops, from code changes alone.** One harness commit
(`af3a24a`, a Safety Car now suspends the pit bounds) also lands in that interval and moves the RAIL
buckets, which is why the decomposition is by bucket transition rather than by summary table — the
transitions above are `no_call -> scored`, which the rails cannot produce. `84dc4b6` was checked and
is behaviour-identical at the default (it replaced a hardcoded `risk_tolerance=0.5` with a constant
of the same value).

### Silverstone is an outlier and it is the wet race

1 scored of 31, 71% declined. The S4 gate had already quantified the mechanism from the other side:
in the wet, ~40% of laps fail N04's quality filter, so N06's previous-lap anchor falls back to the
90.0 constant for whole stints. Recorded rather than averaged away — but the owner's call is that wet
races are out of scope, so it is not treated as a defect here.

## 3. The timing metric does not measure timing

This is the finding that decides the sprint, and it invalidates a number we have published three
times.

`decision_modes.py:439` derives the chosen lap as **the first lap in the window on which the stack
emits any pit action**:

```python
chosen = _first_pit_lap(actions, window_low, window_high)
```

So a stack that would pit on *every* lap of the window pins to `window_low = actual_lap - WINDOW`.
That is not a timing estimate; it is the window's left edge.

**Measured, and this is the disqualifying evidence.** Same two races, only `DECISION_WINDOW_LAPS`
changed — production code untouched:

| offset | window = 5 | window = 10 |
|---|---|---|
| −10 | — | **10** ← new edge |
| −9 | — | 3 |
| −8 | — | 1 |
| −7 | — | 2 |
| −5 | **36** ← old edge | **0** |
| −4 | 9 | — |
| −1 … −3 | 10 | — |
| 0 | 20 | 11 |
| +8 | — | 1 |
| **mean signed** | **−3.08** | **−5.04** |

The mass at −5 does not survive widening: it goes to **zero**. Those calls were never "five laps
early", they were "at or beyond five laps early, clipped to five". Widening the window relocates the
pin rather than revealing a distribution.

**So the mean signed error is a function of the window width, not a property of the model.** −2.23
(baseline, w=5, mixed), −3.08 (2025, w=5), −5.04 (2025, w=10) are three readings of the same
artefact.

What *is* real is the second mode: **11 stops sit at exactly 0 and stay there at both window widths.**
Those are genuine agreements. The picture after the epic's fixes is therefore not "the layer stops
2-3 laps early" but:

> on roughly half the stops it now calls, the layer has **no opinion about when** — it says PIT on the
> first lap it is asked about, and would have said it earlier.

## 4. The answer to `WINDOW_LAPS`, which is not the answer the plan expected

**It cannot be decided from this data, and widening it is not the fix.**

The plan framed the question as a scoring-horizon problem: the 5-lap window prices 100% of a stop's
cost against ~5 laps of its benefit, break-even ~92 laps. That framing may still be right. But it
cannot be tested while the *measurement* pins to its own boundary, because every candidate window
width produces its own answer.

The prerequisite is a timing metric that locates a decision rather than a first occurrence. Two
shapes worth considering, neither implemented here:

- score the **transition** (STAY_OUT on lap n−1 → PIT on lap n), which finds an actual boundary and
  reports "no boundary in window" for a stack that is already committed;
- score the **argmax margin** per lap, so a lap where PIT wins by 0.01 is not counted the same as one
  where it wins by two positions.

Until then, `mean_signed_error` should not be quoted. `DECISION_WINDOW_LAPS = 5` is the eval's own
constant (`decision_modes.py:76`), separate from the MC's `WINDOW_LAPS` (`strategy_orchestrator.py:625`),
so fixing the metric touches no production code and moves no golden.

## 5. What was tried and did not hold

- **That `84dc4b6` was a second confound.** It exposes `alpha` as `DEFAULT_RISK_TOLERANCE = 0.5`
  where the code previously hardcoded `risk_tolerance=0.5`. Read the diff: behaviour-identical.
- **That the full six-race sweep could be measured in one process.** Two background runs died
  mid-race without writing anything, and one race is 133-495 s, so the sweep overruns the foreground
  limit. Measured one race per invocation, appended to disk as each finished.
- **That widening the window would reveal the real bias.** It does not — see §3. This was the
  hypothesis that made the finding, and it was refuted in the useful direction.
