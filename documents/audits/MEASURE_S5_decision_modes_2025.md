# Sprint 5 — re-measuring the decision layer, and a defect in the metric itself

**Date:** 2026-07-30 · **Issue:** #729 · **Status:** the decline rate improved and is verified on the
IDENTICAL sample; **the timing metric is an artefact of its own window and cannot answer the
`WINDOW_LAPS` question**

> **This document was rewritten after the S5 adversarial gate** (`FABLE_S5_decision_modes.md`), which
> found one HIGH and five MEDIUM defects in the first version — including that the strongest run in
> the sprint was sitting uncommitted on disk while the write-up argued around not having it, and that
> the central table compared two different samples column against column. Every correction is marked
> **[gate]** so the record shows what was wrong rather than only what survived.

---

## 1. The headline: the same sample, before and after

**[gate F1]** The first version of this document claimed no same-sample comparison existed and built
its causal argument on two races. It was wrong: a post-fix run over the **identical 198-stop mixed
sample** had completed and written `documents/eval_reports/decision_modes.json`
(`harness 1f0ec9d`, generated 2026-07-30T08:37 — i.e. after Sprint 4 and the seed fix). Two
background runs had appeared to die; the completion notification arrived and was not acted on.

| identical 198-stop sample | baseline (`80f1fa7`) | post-fix (`1f0ec9d`) |
|---|---|---|
| **declined** (`no_call_in_window`) | 128 (**64.6%**) | 78 (**39.4%**) |
| scored (coverage) | 40 (20.2%) | 90 (**45.5%**) |
| `min_stint` / `opening_laps` / `closing_laps` | 22 / 4 / 4 | **22 / 4 / 4** |
| `no_data` | 0 | 0 |

**The rail buckets are byte-identical, race by race.** So on this sample the guard rails contribute
exactly nothing to the delta, and the whole `af3a24a` confound I worried about is empirically zero
here — a stronger statement than the reasoning I replaced it with.

## 2. The out-of-sample figure, and why the sample was redesigned

Four of the baseline's six races are **2023-24, the training seasons**. Every model the decision layer
consumes (N06, N26, N27, N15, N16) trains on 2023-24 and tests on 2025, so on those four the layer is
fed in-sample predictions production will never see. Redesigned as **the same six circuits, all in
2025** — preserves the archetype spread (street, high-speed, low-downforce, high-degradation) and
leaves the season as the only variable.

| | mixed sample (4/6 in-sample) | 2025-only (fully out-of-sample) |
|---|---|---|
| eligible stops | 198 | 178 |
| **declined** | **39.4%** | **46.1%** |
| coverage | 45.5% | 42.1% |

**[gate, claim F]** The gate marked my in-sample *direction* argument UNSUPPORTED by the data the
first version presented, and it was right — I had asserted a sign with nothing behind it. These two
rows are the support: the partly-in-sample mixed set declines **less** (39.4%) than the fully
out-of-sample season (46.1%). So the committed 64.6% was optimistic in the direction claimed, and
**46.1% is the number that describes the shipped system.**

Denominator drift 198 → 178 is fully accounted **[gate]**: Barcelona −2, Monaco −1, Marina_Bay −2,
**Silverstone −15** (46 stops in 2024 against 31 in 2025), Lusail ±0, Monza ±0.

Per race, 2025:

| race | n | scored | declined | rails | decline % |
|---|---|---|---|---|---|
| Barcelona | 41 | 25 | 15 | 1 | 36.6% |
| Marina_Bay | 23 | 16 | 7 | 0 | 30.4% |
| Monza | 20 | 11 | 7 | 2 | 35.0% |
| Monaco | 37 | 14 | 17 | 6 | 45.9% |
| Lusail | 26 | 8 | 14 | 4 | 53.8% |
| **Silverstone** | 31 | **1** | 22 | 8 | **71.0%** |

## 3. Attribution, with the reasoning corrected

**[gate F4]** The first version said *"the transitions are `no_call -> scored`, which the rails cannot
produce."* **That sentence is wrong in principle.** `af3a24a` suspends the early-race and min-stint
bounds *inside the stack* when `sc_currently_active` is true (`pit_strategy_agent.py:530-564`), and a
PIT that the old rails overrode into STAY_OUT on a window lap under a Safety Car is **exactly** a
`no_call -> scored` transition. The conclusion happened to survive; the reasoning offered for it did
not hold.

What actually rescues it is a check the first version never ran, and the gate did: across all **64**
moved stops, **no TrackStatus 4/5/6 lap falls inside the relevant ±5 window.** 2025 Lusail's Safety
Car is laps 7-10 and its moved stops are at lap 32; the other five races have no neutralised laps in
range. So the attribution to our code changes stands — on evidence, not on the argument I gave.

Separately **[gate F4]**: `guard_rail_block` (`decision_modes.py:203-206`) never passes `sc_active`,
and the eval's rail counts are identical between the two mixed runs. The `30 → 21` in §2's rails
column is therefore the **season change**, not the commit.

### The caveat that limits how hard this can be pushed

**[gate F5]** The committed baseline was generated at `80f1fa7`, which is **not** an ancestor of the
MC-Dropout seed fix `8d68a9e`. So every matched-stop transition against it compares a seeded run
against **one draw of an unseeded process**, and "+N moved, from code changes alone" carries a noise
term of unmeasured size.

What *is* established: the **new** code is deterministic. Two independent Monza w=5 runs and a tree
rerun against the sweep agree stop for stop, **46/46 and 20/20**.

**The noise floor, measured rather than left open.** Reproducing the pre-#740 process on current code
by neutralising `torch.manual_seed` in-process (never on disk) and running 2025 Monza twice:

```
unseeded run 1: 20 stops      unseeded run 2: 20 stops
IDENTICAL verdicts: 20/20     DIFFERING: 0
```

So the dropout wobble — real, and measured at `laps_to_cliff_p90` alternating 3.00/3.10 in #735 — does
**not** reach far enough to flip a bucket or a chosen lap on this race. Two caveats keep this honest:
one race and 20 stops cannot establish a noise floor of zero everywhere, and the same wobble *was*
shown to move an argmax census (53/57 versus 52/57 in #735), so it is not inert in general.

What settles the attribution is magnitude rather than this measurement alone: the same-sample delta is
**50 stops on 198**. A noise process invisible at 0/20 would have to be an order of magnitude larger
than anything observed to account for that.

## 4. The timing metric does not measure timing

This retires a number published three times, so the evidence matters more than the conclusion.

`decision_modes.py:434` **[gate F8 — the first version cited :439, which is stale]** derives the
chosen lap as **the first lap in the window on which the stack emits any pit action**:

```python
chosen = _first_pit_lap(actions, window_low, window_high)
```

A stack that would pit on *every* lap of the window therefore pins to
`window_low = actual_lap − DECISION_WINDOW_LAPS`. That is the window's left edge, not a timing
estimate.

**[gate F2] The first version's table was cross-sample and its headline "−5: 36 → 0" was not
like-for-like** — the w=5 column was the full six-race sweep (75 scored) and the w=10 column was
Monza + Marina_Bay alone (28 scored). Corrected to the same two races at every width, and the gate
re-ran all three independently:

| offset | w = 5 | w = 10 |
|---|---|---|
| −10 | — | 10 ← new edge |
| −9 … −7 | — | 6 |
| **−5** | **12** ← old edge | **0** |
| −4 | 1 | — |
| −2 | 1 | — |
| **0** | **13** | **11** |

**The −5 mass goes to zero at w=10, and every −5 stop relocates to the new boundary.** Those calls
were never "five laps early"; they were "at or beyond five laps early, clipped". So **−2.23**
(baseline, w=5, mixed), **−3.08** (2025 sweep, w=5) and **−5.04** (two races, w=10) are three
readings of one artefact, and `mean_signed_error` is a property of the window rather than of the
model. The gate confirmed the pin at a third width and excluded the innocent explanations
(eligibility, guard-rail bucketing, replay span).

**[gate F3, identity-checked]** The first version claimed "11 stops sit at exactly 0 at both widths".
At w=5 the same two races have **13**, and the 11 are a strict subset of them. So the statement is
13 → 11, and **two of the published exact agreements are themselves window artefacts**: Marina_Bay NOR
lap 26 (0 → −10) and Marina_Bay OCO lap 30 (0 → −8) — the stack would have called those stops 8-10
laps earlier had it been asked. **15% of the exact-agreement bucket is artefact.** That strengthens the
finding and weakens the consolation the first version drew from it.

Two mechanisms that could have manufactured the edge mass were hunted and excluded: the only
eligibility drift between widths is Marina_Bay GAS lap 51 (`no_call` at w=5 → **+8** at w=10), which
adds a *late* call and therefore cannot produce an early pin; and overlapping-window cross-talk, where
a second stop's wider window reaches back into the first stop's pit-call region (Marina_Bay HUL,
windows [25, 44]), did not fire — HUL's second stop stays `no_call` at both widths.

So the honest description of the layer after the epic's fixes is not "it stops 2-3 laps early" but:

> on roughly half the stops it now calls, the layer has **no opinion about when** — it says PIT on the
> first lap it is asked, and would have said it earlier.

## 5. The answer to `WINDOW_LAPS`

**It cannot be decided from this data, and widening it is not the fix**, because every candidate width
produces its own answer. The prerequisite is a metric that locates a **decision** — a STAY_OUT → PIT
transition — rather than a first occurrence, and that reports "no boundary inside the window" for a
stack already committed. Filed as **#752**.

`DECISION_WINDOW_LAPS` (`decision_modes.py:76`) is the eval's own constant, separate from the MC's
`WINDOW_LAPS` (`strategy_orchestrator.py:625`), so fixing the metric touches no production code and
moves no golden.

## 6. Silverstone, with the causal claim withdrawn

**[gate F7]** The first version said Silverstone 2025's 71% decline **is** the wet-race anchor
mechanism. Wet is verified — 608/826 laps (73.6%) on INTERMEDIATE, rainfall in 18.1% of samples,
`IsAccurate` share 0.600. But the mechanism is measured to be an **insufficient** explanation:

- Silverstone: 274/592 evaluated laps (46.3%) still anchor N06 on 90.0 — but the **dry controls are
  25-30%** (Monza 30.0%, Lusail 25.2%), so heavy anchoring is endemic to this tier's window shape,
  not a wet-race special.
- **Anchor share does not rank-order decline**: Lusail 25.2% anchored → 53.8% declined; Monza 30.0%
  anchored → 35.0% declined.
- Unexamined co-mechanisms: INTERMEDIATE takes the `_DEFAULT_MIN_STINT = 10` fallback in the rails
  (`guard_rails.py:41-42`), and the tyre stack never trained on that compound at all.

So: wet, yes; *because of the anchor*, not established. Out of scope by the owner's call either way.

## 7. What was tried and did not hold

- **That the six-race sweep could not be measured in one process.** It could, and it was — see §1.
  One race is 133-495 s and the sweep overruns the foreground limit, which is true and was the reason
  for measuring per race; it was not a reason to conclude the background run had died.
- **That `84dc4b6` was a confound.** Verified behaviour-identical at the default, by diff and by
  execution **[gate F6]**.
- **That widening the window would reveal the real bias.** It does not — it relocates the pin. This
  was the hypothesis that produced the finding, refuted in the useful direction.
