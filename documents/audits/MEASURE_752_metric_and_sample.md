# What the metric fix bought, and what moving to 2025 bought, separately

**Date:** 2026-07-30 · **Issues:** #752 (the metric), #757 (the gate follow-up)
**Instrument:** `f1-eval decision-modes`, plus one off-report run holding the code fixed and
restoring the old race list.

The committed report changed **two things at once**: the metric became a transition detector
(#752 + #757) and the sample moved from four training-season races to six 2025 ones. Quoting
"17.8% -> 43.4%" from those two artefacts would attribute one effect to the other. This is the
third cell that separates them.

---

## The three measurements

| | stops scored | **exact lap** | within 1 | mean signed | declines (`no_call`) |
|---|---|---|---|---|---|
| mixed 2023-25, **retired** metric | 90 of 198 (45.5%) | 17.8% | 25.6% | −3.3 | — |
| mixed 2023-25, **current** metric | 62 of 198 (31.3%) | **30.6%** | 48.4% | −1.82 | 78 (39.4%) |
| **2025 only, current metric** | 53 of 178 (29.8%) | **43.4%** | 52.8% | −1.72 | 82 (46.1%) |

The two middle rows share a sample; the two bottom rows share a metric. Row 1 to row 2 is the
metric alone; row 2 to row 3 is the season alone.

## The metric is worth +12.8 points of exact agreement, and it costs 28 stops

Scoring the transition rather than the first pit lap removes 28 of the 90 stops the retired
scorer graded, and every one of them was graded by reporting the window's own left edge. What
remains is a set the tier can actually locate a decision in, and on it exact agreement rises
17.8% -> 30.6%.

**Both the rate and the count improve, which is rare enough to state explicitly.** Exact
agreements in absolute number go **16 -> 19 -> 23** across the three rows, while the scored set
shrinks **90 -> 62 -> 53**. The tier grades fewer stops and gets more of them right, in count,
not just in share. That is the opposite of the pattern the epic's scorecard had to explain
earlier, where the exact-agreement rate fell while its count rose because the layer had become
far less reluctant.

## The season is worth another +12.8 points, and this was NOT the expected direction

Holding the metric fixed and moving four races from 2023-24 to 2025 raises exact agreement
30.6% -> 43.4%.

**2023 and 2024 are TRAINING seasons for every model in the stack, so the naive expectation is
that agreement would be higher there, not lower.** It is lower by the same margin the metric fix
bought. Four of the six races changed only their year (Barcelona, Monaco, Silverstone,
Marina_Bay); Lusail and Monza were already 2025 in both lists, so this is close to a
year-only comparison.

**No cause is claimed here.** Plausible ones — a more converged strategic meta in the
regulation's mature year, less variable weather at those four events in 2025, better-calibrated
degradation on 2025 car behaviour — are guesses, and this project has a scar from publishing a
cause that was a guess wearing a fact's clothes. What is established is the magnitude and the
direction.

## What did not move, which is the consistency check

`no_call_in_window` is decided by `_asks_to_stop`, which neither #752 nor #757 touches, and it
lands at **78/198 = 39.4%** on the mixed sample and **82/178 = 46.1%** on 2025. Those are the
two decline figures the epic scorecard already carries, reproduced here by a different code path
after four PRs. A metric change that had silently moved them would have been a defect.

## The number that must not be quoted

`mean_signed_error` reads −1.82 and −1.72 above and **still moves with the window width**: the
Fable G1 gate measured −0.33 / −1.29 / −2.50 at w=3/5/10 on one race. The mechanism is no longer
the harness artefact #752 retired (a wider window now admits more distant, and therefore earlier,
transitions — a real property of how far back you look), but it is still a property of the eval's
configuration and not of the system. The report's own table now says so in its Meaning column.

## Coverage is `masked`, deliberately

29.8% of eligible stops are scored, against the 60% gate. The headline 43.4% describes the
subset where a decision is locatable, and the report says so rather than promoting the figure.
The 46.1% decline rate is the number to attack next, and it is what #744b exists for.

## Reproducing

```bash
uv run python -m src.strategy.eval.cli decision-modes          # row 3, writes the report
```

Row 2 was produced by calling `measure_decision_agreement(races=...)` with the old list; it is
not a committed configuration, deliberately, because a second committed sample is a second thing
to keep in sync.

Related: `MEASURE_S5_decision_modes_2025.md` · `FABLE_G1_transition_metric.md` ·
`documents/eval_reports/decision_modes.md`
