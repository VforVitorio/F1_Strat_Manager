# AUDIT A1 — The STAY_OUT point mass in the Monte Carlo layer

**Role:** adversarial gate. Nothing in the repository is modified except this file.
**Date:** 2026-07-29
**Target:** `src/agents/strategy_orchestrator.py::_run_mc_simulation` / `_run_projection_mc`,
and `src/agents/position_projection.py` (where the per-draw arithmetic actually lives).

**Success condition:** find what is broken. A clean report would be a failure to try.

---

## Claims under attack

| # | Claim as submitted | Verdict |
|---|---|---|
| A | STAY_OUT returns `E == P10 == P90` (a point mass) on most laps. Observed at Lusail 2025 laps 12/18/24/30/36, not 42. | **CONFIRMED, and understated.** 43/57 laps at Lusail (75.4%), **51/53 at Monza (96.2%)**. |
| B | STAY_OUT dominates PIT_NOW on both E and P10, so no alpha changes the argmax; `risk_tolerance` is decorative. Sweep: alpha 0.1 -> 73.1% declined, 0.5 -> 65.4%, 0.9 -> 65.4%. | **Half right, and REFUTED as stated.** Dominance holds 108/110 laps and *is* alpha-proof by convexity (F3). But alpha changes the argmax on 5/110 laps, so "decorative" is wrong — and the submitted sweep's own 73.1 -> 65.4 spread already disproved it (F5). |
| C | Two competing explanations, undecided: (1) the STAY_OUT branch never consumes the sampled draws; (2) positions are integers so all 500 draws collapse. **Is there a third?** | **BOTH REFUTED. Yes, there is a third — and behind it a fourth that is the real defect.** (1) The branch *does* read `cliff_s`; it is annihilated by `max(0, racing_laps - cliff_laps)` (F1). (2) Refuted by counter-example: integer-collapsed positions still yield 500 distinct payoffs via the continuous margin term (F1). The root cause is F7: a rival who still owes the mandatory stop is exempt in the STAY_OUT branch and counted as passing us in the PIT_NOW branch — **73.4% / 84.0% of car-slots on real laps**. |

## Secondary attack surface

- Is `CFG.n_sim` really 500 at runtime?
- Common random numbers: shared across candidates in `_run_projection_mc` as well as the legacy branch?
- Is `OVERCUT` permanently ineligible (search space 3, not 4)?
- Is `score` ever compared ACROSS laps rather than within one lap?
- Known local bug classes: sentinel collisions, un-twinned fixes, empty-set assertions,
  scale-mismatched thresholds, silently discarded config, `Series.get(k, default)` returning stored NaN.

## Method

Real laps, zero API calls: `profile="no-llm"` through `RaceReplayEngine` +
`src.strategy.inference.engine.run_lap`, then `_run_mc_simulation` re-invoked on the captured
agent outputs. Per-draw arrays instrumented by monkeypatching the internals rather than by
reading the rounded E/P10/P90 the caller sees.

---

## Findings

<!-- appended as confirmed -->

### F1 [HIGH] — Claim C settled: BOTH hypotheses are wrong. The third explanation is a saturating clip.

**Verdict: hypothesis 1 REFUTED, hypothesis 2 REFUTED, third mechanism confirmed.**

The STAY_OUT branch **does** consume `cliff_s`. `src/agents/position_projection.py:547-549`:

```python
else:
    worn_laps = np.maximum(0.0, racing - cliff_laps)
    delta += worn_laps * config.cliff_loss_s
```

But `racing` is `config.racing_laps` — 5.0 under green, 2.61 under a Safety Car — while `cliff_laps`
is the N26 triangular draw of *laps to the tyre cliff*. Measured on the six Lusail 2025 laps the
claim names, the draw support never once came below the window:

| lap | `cliff_s` support (500 draws) | `racing_laps` green / SC | draws with `worn_laps > 0` |
|---|---|---|---|
| 12 | [11.94, 13.46] | 5.00 / 2.61 | **0 / 500** |
| 18 | [13.14, 14.76] | 5.00 / 2.61 | **0 / 500** |
| 24 | [17.49, 21.01] | 5.00 / 2.61 | **0 / 500** |
| 30 | [12.84, 14.36] | 5.00 / 2.61 | **0 / 500** |
| 36 | [34.72, 39.18] | 5.00 / 2.61 | **0 / 500** |
| 42 | [14.36, 16.54] | 5.00 / 2.61 | **0 / 500** |

`np.maximum(0.0, 5.0 - 11.94)` is `0.0`, and so is every other draw. **`our_delta` for STAY_OUT is
identically `0.0` across all 500 draws on every lap measured** — instrumented directly inside
`driver_time_delta`, std `0.0`, 1 unique value, on 12 of 12 calls (two regimes x six laps).

**Why hypothesis 2 (integer positions) is refuted, decisively.** The payoff is *not* integer-valued:
`payoff()` (`position_projection.py:682-693`) adds a continuous margin term
`margin_weight * margins_s`. So a candidate whose positions collapse to a single integer can still
have a non-degenerate payoff — and one does, in the same run:

- **lap 24, PIT_NOW:** `positions` takes exactly one value (`[4.0]`, std `0.0`) — fully integer-collapsed —
  yet `margins_s` has **500 distinct values** and the payoff has 500 distinct values (std `0.0328`).
  E `-2.853`, P10 `-2.909`, P90 `-2.822`. Not a point mass.
- **lap 24, STAY_OUT:** positions also single-valued (`[1.0]`), but `margins_s` has **1** distinct value
  and the payoff has 1. E = P10 = P90 = `0.300`.

Same integer collapse, opposite outcome. Integer rounding is therefore *not* the cause; the cause is
that STAY_OUT's `our_delta` is a constant, so every downstream quantity that depends on it
(projected gaps -> positions -> margins) is a constant too.

**The mechanism, stated once:** STAY_OUT has exactly one stochastic input — the tyre-cliff draw — and
it enters only through `max(0, racing_laps - cliff_laps)`. The N26 model's laps-to-cliff P10 never
drops near the 5-lap window (minimum observed 11.94, i.e. 2.4x the green window and 4.6x the
Safety-Car window), so that expression is a hard zero on every draw and the entire draw set is
annihilated. What survives is a deterministic projection.

*file:line* — `src/agents/position_projection.py:530-551` (`driver_time_delta`, the `else` branch),
`:547-549` (the clip), `:672` (margin clip), `:691-693` (payoff).

---

### F2 [HIGH] — The tyre agent (N26) contributes nothing to the Monte Carlo score, on any candidate.

> **Scope corrected by F4 below.** Written from six laps; two full races narrowed it. Read F4 with it —
> the mechanism stands, the word "any" does not.

F1 kills the cliff term for STAY_OUT. The pitting candidates never had it. In `driver_time_delta`
(`position_projection.py:531-541`):

```python
laps_before_stop = min(float(plan.stop_offset_laps), racing)
...
worn_laps = np.maximum(0.0, laps_before_stop - cliff_laps)
```

`stop_offset_laps` is `0` for PIT_NOW and UNDERCUT (`strategy_orchestrator.py:1211-1212`), so
`worn_laps = max(0, 0 - cliff_laps) = 0` **unconditionally, for any cliff draw whatsoever**. For
OVERCUT the offset is `1` (`:1214`), so the term needs `cliff_laps < 1` — a tyre one lap from the
cliff — to fire at all.

Measured `worn_active` counts across all candidates and both regimes on the six Lusail laps:
**0/500 on every one of the 20 `driver_time_delta` calls.** `cliff_s` is drawn (`:1387`), passed
through `_run_projection_mc` (`:1235, :1242`), indexed, multiplied — and multiplied by zero.

Consequences worth stating plainly:
1. `CLIFF_LOSS = 0.80` is a dead constant on the projection path.
2. One of the project's seven models has **no influence on the Layer-2 decision** whenever the
   projection path is taken (i.e. whenever rivals carry gaps — the normal case on all three surfaces).
3. Because the tyre term is the only thing that could ever make staying out *cost* something,
   the layer has no mechanism to say "your tyres are gone, box now". That is the exact strategic
   signal the tyre agent exists to provide.

This is the F1 mechanism's real-world payload and it is why it outranks a "flat distribution"
cosmetic complaint.

---

### F3 [HIGH] — Claim B confirmed, and it is a THEOREM, not a measurement: alpha cannot matter.

`score = alpha*E + (1-alpha)*P10` (`strategy_orchestrator.py:1273`) is a convex combination. If one
candidate is >= another on **both** E and P10, it is >= for **every** alpha in [0,1]. So a sweep that
finds the same decision at alpha 0.1 / 0.5 / 0.9 is not evidence of a weak effect — it is the
arithmetic consequence of dominance, and no alpha anywhere in the interval can break it.

Dominance measured on all six laps (E and P10, STAY_OUT vs PIT_NOW):

| lap | STAY_OUT E / P10 | PIT_NOW E / P10 | dominates? |
|---|---|---|---|
| 12 | -4.700 / -4.700 | -17.000 / -17.000 | yes |
| 18 | +0.300 / +0.300 | -11.404 / -11.995 | yes |
| 24 | +0.300 / +0.300 | -2.853 / -2.909 | yes |
| 30 | -1.700 / -1.700 | -9.980 / -10.875 | yes |
| 36 | -0.737 / -0.737 | -1.427 / -1.700 | yes |
| 42 | -0.610 / -0.916 | -1.815 / -1.883 | yes |

6/6. `risk_tolerance` is decorative on every one of these laps for a structural reason, not a
stochastic one.

---

### F4 [correction to my own F2] — the cliff term is not universally dead, but where it lives it is 50x too small to matter.

Two full races, every lap, `profile="no-llm"`:

| race / driver | laps | STAY_OUT flat | STAY_OUT draws with `worn_laps > 0` | min N26 `laps_to_cliff_p10` |
|---|---|---|---|---|
| Lusail 2025 / NOR | 57 | 43 (**75.4%**) | 2 000 / 57 000 (**3.5%**) | 2.60 |
| Monza 2025 / LEC | 53 | 51 (**96.2%**) | **0 / 53 000 (0.00%)** | 6.50 |

So F2's "contributes nothing on any lap" is **too strong and I withdraw it in that form**. The
accurate statement:

- At **Monza the tyre channel is completely dead**: 0 of 53 000 draw-slots across the whole race.
- At **Lusail it fires on 4 laps of 57** (33, 34, 35, 37 — the only laps where `laps_to_cliff_p10 < 5`),
  and only under the **green** config; even at `p10 = 2.60` it never fires under the Safety-Car
  config (`racing_laps = 2.61`). Instrumented `worn_active` per lap: `[500, 0]` on all four.
- On those four laps the tyre term moves the STAY_OUT score by **0.013 to 0.023 positions**
  (P90 - P10), because the N26 triangular is only ~0.4 laps wide there.

Compare that with the *other* stochastic term. On the six Lusail laps where the terminal liability
straddles a gap, the STAY_OUT spread is **exactly 1.000 position** (laps 9, 14, 16, 28, 31, 41, 42 —
integer car counts). **The tyre model's entire contribution is ~50x smaller than the quantisation
step of the only other source of variance.** Even on its best lap it cannot move a decision.

Net: the finding stands, its scope is narrower than I first wrote, and the practical conclusion is
unchanged — the tyre agent cannot make the layer say "your tyres are gone, box now".

---

### F5 [HIGH] — Claim B is REFUTED at race scale, and the claimant's own sweep already refuted it.

`risk_tolerance` is not fully inert. Measured argmax under seven alphas per lap:

**Lusail (57 laps)** — the argmax changes with alpha on **5 laps: 34, 35, 41, 44, 55**.

| alpha | 0.0 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|
| STAY_OUT | 56 | 56 | 55 | 54 | 54 | 54 | 53 |
| PIT_NOW | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| UNDERCUT | 0 | 1 | 2 | 3 | 3 | 3 | 4 |

**Monza (53 laps)** — STAY_OUT wins **53/53 at every alpha**; the argmax changes on **0 laps**.

The submitted sweep is self-refuting on this point: 73.1% -> 65.4% between alpha 0.1 and 0.5 is a
7.7-point move, and something that moves 7.7 points is not decorative. My measured 5/57 = 8.8% of
laps agrees with that magnitude almost exactly.

**The correct, defensible statement** is narrower and more damning:

1. STAY_OUT dominates PIT_NOW on **both** E and P10 on **55/57 Lusail laps and 53/53 Monza laps**,
   so on 108 of 110 laps no alpha can select PIT_NOW.
2. Across both full races, a pitting candidate wins at *some* alpha on **5 laps out of 110**, and on
   **4 of those 5 the winner is UNDERCUT, not PIT_NOW**. PIT_NOW wins exactly **once in 110 laps**,
   at alpha = 0.0 only, by 0.067 (Lusail lap 41: STAY_OUT -1.890 vs PIT_NOW -1.823).
3. UNDERCUT wins only via the **additive N16 bonus** applied outside the projection
   (`strategy_orchestrator.py:1263-1264`, `outcomes + landed`), never because the projection valued
   the stop. Remove that line and the layer would recommend STAY_OUT on 109 of 110 laps.

So: alpha is not decorative, but it only ever arbitrates between STAY_OUT and a bonus bolted on top
of the projection. **The projection itself never prefers a stop.**

---

### F6 [HIGH] — The 5-lap window makes a pit stop mathematically unrecoverable, which is why STAY_OUT dominates.

This is the structural cause behind F3/F5 and it is arithmetic, not sampling.

In `driver_time_delta` a stopping plan pays `effective_loss = max(0, pit_loss_s - saving)` and is
credited `laps_after_stop * fresh_gain_s` (`position_projection.py:537-542`). Measured at Lusail:

- STAY_OUT `our_delta`: **0.000 s**, every draw, every lap.
- PIT_NOW `our_delta`: **mean 22.99 s** (green, `racing_laps = 5`) / **15.59 s** (SC, `racing_laps = 2.61`).

`FRESH_GAIN = 0.25` s/lap, so the whole window returns `5 x 0.25 = 1.25 s` against a ~23 s loss.
**Break-even needs `pit_loss / fresh_gain` = 23 / 0.25 = ~92 racing laps; `WINDOW_LAPS` is 5.**
The window is ~18x too short for the projection to ever value a stop.

The asymmetry is compounded on the rival side. `rival_time_deltas`
(`position_projection.py:573-576`) charges a rival a pit loss **only when `is_pitting` is true right
now** — a timing fact that holds for at most one or two cars on any lap. So the projection compares
"we pay 23 s inside the window" against "the other 18 cars pay nothing, ever", even when they
demonstrably still owe the same mandatory stop. The one counterweight, `terminal_liability`, is
capped by the number of *settled cars behind us inside our own exposure* and fires only when
`mandatory_stop_pending is True`.

Measured consequence at lap 12, Lusail: from **P2**, PIT_NOW projects to **P19 of 19** (positions
array single-valued `[19.0]` under both regimes), i.e. E = **-17.000**. STAY_OUT's liability was
5 cars, E = **-4.700**. A 5-car liability cannot offset a 17-car projected drop, so the lap is not
close — and the drop exists only because no rival was charged the stop they also owe.

---

### F7 [HIGH — the root cause] — The same rival is a threat when we pit and exempt when we stay out.

This is the sharpest statement of the defect and, I believe, the thing to fix first.

`terminal_liability` (`position_projection.py:616-620`) charges us only for rivals who have
**already settled** their mandatory stop:

```python
behind_and_settled = [
    rival.gap_s
    for rival in _usable_rivals(rivals)
    if rival.gap_s > 0 and rival.stop_pending is False
]
```

with the docstring reason (`:592-594`): *"a rival who must still stop pays the same price later, so
they are no threat."* Correct.

But `project_positions` applies **no such filter**. Every rival within our pit loss is counted as
passing us (`:665-668`), and `rival_time_deltas` charges a rival a stop only when `is_pitting` is
true *this lap* (`:573-576`). A car that still owes the mandatory stop is therefore:

- **exempt** in the STAY_OUT branch (it will pay the same price later), and
- **counted as passing us** in the PIT_NOW branch (as if it never has to stop at all).

The same fact about the same car produces opposite treatment depending on which candidate is being
scored. That is not a modelling simplification; the two branches contradict each other.

**Measured on every lap of two real races** (`rival_stop_pending` as the pipeline actually
populates it, our pit loss = circuit traversal + 2.93 s):

| race | car-slots that pass us if we PIT | car-slots charged if we STAY OUT | exempt in one branch only |
|---|---|---|---|
| Lusail 2025 / NOR (57 laps) | 474 | 126 | **348 (73.4%)** |
| Monza 2025 / LEC (53 laps) | 493 | 79 | **414 (84.0%)** |

Worked example, **Monza laps 11-16, LEC in P4**: 14 cars sit within our pit loss behind us. All 14
have `stop_pending = True`. So PIT_NOW is charged **14 places** and STAY_OUT is charged **0** — from
the same 14 cars, on the same lap, for the same reason. There is no lap in the race where a stop can
survive that.

This, not the flat distribution, is why STAY_OUT dominates on 108 of 110 laps.

---

### F8 [HIGH] — No test in the repository can observe this, because every fixture feeds constant draws.

Every call into `_run_projection_mc` from the test suite passes `np.full(...)` arrays:

- `tests/mc/test_mc_is_a_real_decision.py:181-183` — `cliff_s=np.full(_DRAWS, cliff)`,
  `sc_s=np.full(_DRAWS, neutralised)`, `pit_s=np.full(_DRAWS, stop_s)`
- `:404-406`, `:475-477`, `:534-536` — same shape, hardcoded scalars

Only `ucut_s` ever varies (`(np.arange(_DRAWS) % 2 == 0)`), and it is the one input applied *outside*
the projection as an additive bonus.

**Consequence:** in every test, all 200 draws are identical by construction, so *every* candidate is
a point mass. The property "the Monte Carlo produces a distribution" is destroyed by the fixture
before any assertion runs — and no assertion tests it anyway. Grepping the whole `tests/` tree for
an assertion relating P10 to P90 returns exactly three lines, and all three are shape checks:

- `tests/engine/test_engine_no_llm.py:109` — `assert {"E","P10","P90","score"} <= set(scores)`
- `tests/mc/test_strategy_goldens.py:152` — `assert set(scores) == {"E","P10","P90","score"}`
- `tests/mc/test_strategy_goldens.py:141` — `assert scores["score"] == approx(scores["P10"])` at alpha=0

**Nothing anywhere asserts `P90 > P10` for any candidate.** A file named
`test_mc_is_a_real_decision.py` is green while the layer it guards returns a constant for the
candidate that wins 98% of real laps.

This is the repo's "test that passes by asserting over an empty set" class in a new shape: the
fixture removes the phenomenon, then the suite declines to assert it.

**Two hypotheses of mine that this data REFUTED — recorded because I was wrong:**

1. *"The sweep only passes because it includes `cliff = 1.0`, a value the real model never emits."*
   **False.** Re-running the shipped sweep with realistic cliffs changes almost nothing:
   as shipped `(1.0, 6.0, 20.0)` -> STAY_OUT 80.7%; realistic `(12.0, 20.0, 35.0)` -> 81.5%;
   cliff removed entirely `(99.0,)` -> 81.5%. All still pass. (Which is *another* confirmation of F4:
   deleting the tyre channel from the sweep moves the answer by 0.8 points.)
2. *"The sweep passes because its 3-car field is too small."* **False, and backwards.** Re-running
   the same invariants at 6 / 10 / 14 / 19 cars gives PIT_NOW 58 / 106 / 122 / 123 wins — the bigger
   field makes PIT_NOW win *more*. At 3 cars the top share is 89.0% and
   `test_no_projected_candidate_wins_almost_everything` **would fail**.

The real reason the sweep passes is F7: the fixture sets `rival_stop_pending: {"B": False, "C": False}`
(`:178`), i.e. every car behind is *settled*, which is the one configuration where the liability is
allowed to fire at full strength. Real races put 73-84% of those cars in the exempt-but-still-passing
state that no fixture reproduces.

---

### F9 [HIGH] — The frozen golden, described as "the thesis-defended math", sits in a 3.6%-of-laps corner.

`tests/mc/test_strategy_goldens.py:77-95` pins `_GOLDEN_ALPHA_05` with the comment *"This IS the
thesis-defended math: any drift in simulate_lap_window or the sampling breaks this assert."*
Its STAY_OUT row shows a healthy distribution: `E -0.149, P10 -0.529, P90 0.0`.

It shows one because `_canned_outputs()` (`:52-60`) sets
`laps_to_cliff_p10=3.0, p50=5.0, p90=8.0` — straddling `WINDOW_LAPS = 5`, the only regime where
STAY_OUT's cliff term survives the clip.

Measured against what N26 actually emits over two full races:

| | Lusail (57 laps) | Monza (53 laps) |
|---|---|---|
| min `laps_to_cliff_p10` | 2.60 | **6.50** |
| median `laps_to_cliff_p10` | **20.00** | — |
| laps with `p10 < 5.0` | **4** | **0** |

The golden's tyre state occurs on **4 of 110 measured laps (3.6%)**, and never at Monza. The one
fixture that pins the layer to the digit is calibrated to the corner where the bug is invisible.

---

### F10 [HIGH] — The legacy branch has the identical clip. Both twins are broken, not one.

`simulate_lap_window` (`strategy_orchestrator.py:712-714`):

```python
if strategy == "STAY_OUT":
    cliff_laps = max(0.0, window - cliff_i)
    time_delta = -cliff_laps * CLIFF_LOSS
```

Same `max(0, window - cliff)` shape as `driver_time_delta`'s `else` branch. Executed with the
Lusail lap-12 tyre distribution (triangular 11.94 / 12.70 / 13.46, seed 42, n = `CFG.n_sim`):

| candidate | E | P10 | P90 | std | unique values / 500 |
|---|---|---|---|---|---|
| **STAY_OUT** | +0.0000 | -0.0000 | +0.0000 | **0** | **1** |
| PIT_NOW | -0.7971 | -1.4379 | -0.7786 | 1.294 | 500 |
| UNDERCUT | -0.2971 | -1.3208 | +0.1983 | 1.406 | 500 |
| OVERCUT | -1.2661 | -1.9379 | -1.2786 | 1.412 | 500 |

The legacy path shows the **same point mass and the same STAY_OUT dominance** in a completely
different currency. So this is not "the projection engine regressed" — the shape was inherited. In
the legacy design STAY_OUT is the reference baseline and scoring 0 is defensible; in the projection
it is a candidate in its own right and a constant is not.

---

### F11 [MEDIUM] — `score` IS compared across laps, and the fallback crosses scales.

Answering the question directly: **yes**, at
`src/telemetry/backend/services/simulation/simulator.py:645-659`.

```python
score = decision.scenario_scores.get(decision.action, decision.confidence)
if score > state.best_decision["score"]:
    ...
if score < state.worst_decision["score"]:
    ...
```

Two distinct defects in three lines.

**(a) Cross-lap comparison of a within-lap quantity.** The MC score is *positions gained relative to
our current position on this lap*, against this lap's field geometry. Its scale is set by how many
cars sit within one pit loss, which changes lap to lap. Measured winner-score range in a single race:
**-4.948 to +2.300 at Lusail**, **-0.937 to +1.300 at Monza**. `best_decision` therefore reports the
lap where the field happened to be most spread out, not the best call of the race. The same lap-12
STAY_OUT that scores -4.700 would be reported as the race's "worst decision" purely because five
settled cars sat behind.

**(b) Scale collision in the default.** When the chosen action is not a scored candidate, the
fallback is `decision.confidence`, which lives in **[0, 1]**. Measured against the real winner scores,
a confidence anywhere in 0.5-0.9 **outranks 56 of 57 Lusail laps and 51 of 53 Monza laps** — the
winner-score ranges are -4.948..+2.300 and -0.937..+1.300, so almost the entire race sits below any
plausible confidence. `ALERT` is a valid `action`
(`strategy_orchestrator.py:252`) and is never an MC candidate; and Layer 3 can name any action
including one the projection marked ineligible, in which case `_coerce_scenario_scores`
(`simulator.py:479-490`) correctly **drops** the key and the `.get` default fires. Any such lap
contributes a value from a different scale and, at a typical confidence of 0.7-0.9, **will beat every
genuinely-scored lap in the race**. This is the "tuned threshold compared against a differently-scaled
quantity" class, applied to a report rather than a decision.

Note the near-miss that makes it worth flagging: both flatteners (`simulator.py:485-487` and
`src/arcade/strategy.py:756-762`) were *correctly* fixed to drop `None` rather than coerce it to 0.0
— the twin got its fix. The `.get(action, confidence)` default three lines away did not.

---

### F12 [MEDIUM] — The target selector and the position scorer disagree about pending rival stops, and a docstring denies it.

`rank_targets` charges a rival their stop when they still owe it (`position_projection.py:776`):

```python
their_loss = rival.stop_loss_s if rival.stop_pending is True else 0.0
```

`rival_time_deltas` — the function that actually feeds `project_positions` — charges it only when
they are physically in the pit lane this lap (`:573-576`):

```python
loss = rival.stop_loss_s if rival.is_pitting else 0.0
```

Two functions in the same file computing "seconds this rival loses over the window", on different
criteria. Executed demonstration (two rivals ahead, one owing the stop, one settled, our pit loss
22.8 s):

```
rank_targets  (SELECTOR): AHEAD_OWES  current  -5.00 -> projected  -5.00   <- ranked NEAREST
                          AHEAD_DONE  current  -9.00 -> projected -31.80
rival_time_deltas (SCORER): AHEAD_OWES charged +0.00 s
                            AHEAD_DONE charged +0.00 s
```

The selector cancels `AHEAD_OWES`'s 22.8 s against ours and calls them the car we will be racing;
the scorer leaves their 5-second lead untouched. `rank_targets`' own docstring (`:766-768`) states:

> *"Both consume the same definition of 'who we are racing', so the selector and the scorer can no
> longer disagree."*

They do disagree, on any lap with a rival whose `stop_pending is True` and who is not currently in
the pit lane — the normal case (Monza laps 11-16: 14 such cars). The docstring asserts an invariant
the code does not hold.

---

### F13 [LOW] — OVERCUT is not permanently ineligible, but it is near-absent; the search space is ~2.9, not 4.

Measured eligibility over two full races:

| candidate | Lusail (57) | Monza (53) | combined |
|---|---|---|---|
| STAY_OUT | 57 | 53 | 110/110 |
| PIT_NOW | 57 | 53 | 110/110 |
| UNDERCUT | 47 | 35 | 82/110 (74.5%) |
| **OVERCUT** | **2** | **4** | **6/110 (5.5%)** |

Mean scoreable candidates per lap at Lusail: **2.86**. So the answer is "no, not permanent, but it
competes on 1 lap in 18" — which is correct behaviour given `overcut_targets` requires a car directly
ahead to be in the pit lane *this lap* (`position_projection.py:724-733`), a genuinely rare fact.
Recorded so the search-space figure is right, not as a defect.

---

## Verified and NOT broken

These were attacked and held up. Each was executed, not read.

| Claim | Result |
|---|---|
| `CFG.n_sim` is 500 at runtime | **Holds.** `n_draws == 500` on all 110 laps of both races; `OrchestratorCFG.n_sim = 500` (`:126`) is never overridden on any path exercised. |
| Common random numbers across candidates | **Holds, on both branches.** `_run_mc_simulation:1386-1390` draws once; the projection branch passes the same four arrays to all four plans (`:1402-1405`) and reuses one `pit_loss_s` (`:1124`) and one `neutralised` mask (`:1218`). Instrumented `driver_time_delta` std is identical (0.327547) across PIT_NOW / UNDERCUT / OVERCUT on every lap. Legacy branch reuses the same `cliff_s[i]/sc_s[i]/pit_s[i]/ucut_s[i]` inside one loop (`:1413-1418`). |
| CRN also shared across the two neutralisation regimes | **Holds.** `project_positions` is called for green and under-SC with the same `pit_loss_s` and `cliff_s`. |
| `None` scores leaking to charts as 0.0 (sentinel collision) | **Fixed in both twins.** `simulator.py:485-487` and `arcade/strategy.py:756-762` both drop the key. `_scoreable` / `best_mc_candidate` / `mc_decision_margin` (`:968-999`) all guard with `_finite_or_none`. |
| `pandas.Series.get(k, default)` returning a stored NaN | **Not present in this layer.** Rival and context reads go through `_finite_or_none` (`:868-883`), which collapses `None`, NaN and inf together. `_position_or` (`:931-942`) additionally rejects positions `< 1`. |
| Tie-breaking by dict-insertion accident | **Fixed.** `_TIE_BREAK_ORDER` (`:965`) plus a log line (`:1031-1037`). |
| Triangular collapse crashing lap 1 | **Held.** `_clamp_triangular` (`:1333-1344`) fired without error across 110 laps including lap 1. |
| Zero errors in the harness | 0 exceptions in 110 laps across both races. |

---

## Fix list, ordered by value/risk

1. **Charge a pending rival stop in `project_positions`, or stop exempting them in
   `terminal_liability` — but pick ONE rule and apply it to both branches** (F7).
   `position_projection.py:573-576` vs `:616-620`. Highest value by a wide margin: it is the reason
   STAY_OUT dominates 108/110 laps. Lowest-risk form: in `rival_time_deltas`, charge
   `rival.stop_loss_s` when `rival.stop_pending is True` (matching what `rank_targets:776` already
   does), leaving `is_pitting` as an additional trigger. That also closes F12 for free.
2. **Assert the distribution, then fix the fixtures** (F8). Add `P90 > P10` for at least one
   candidate under a non-degenerate draw, and replace `np.full(...)` with the real
   `rng.triangular(...)` in `tests/mc`. Do this *before* 1 and 3 so those changes are measurable.
   Zero production risk.
3. **Decide what the tyre channel is for** (F1/F4/F9). Either the window must be long enough for the
   cliff to fall inside it, or the STAY_OUT branch needs a degradation term that does not require
   crossing the cliff within `racing_laps` (e.g. cumulative per-lap deg, not a step at the cliff).
   As shipped, N26 moves the score by <= 0.023 positions against a liability that quantises at 1.0.
4. **Fix `simulator.py:647`** (F11). Split into two changes: drop the `confidence` fallback (skip the
   lap instead — an unscored action has no score), and either remove `best_decision`/`worst_decision`
   or rank on a within-lap normalised quantity such as `mc_decision_margin`. Small, isolated, no
   effect on decisions.
5. **Correct the two false docstrings** (F12 `position_projection.py:766-768`; and
   `strategy_orchestrator.py:1392-1395`'s CRN comment is *true* — leave it). A docstring asserting an
   invariant the code does not hold is how the next person "fixes" the correct side.
6. **Reconsider `WINDOW_LAPS = 5` against `FRESH_GAIN = 0.25`** (F6). Break-even is ~92 racing laps.
   Highest-risk item on this list — it moves every number in the layer — so it should follow 1 and 2,
   with the goldens re-frozen deliberately rather than as a side effect.

---

## What I tried to break and COULD NOT

Stated explicitly so the next auditor knows where not to spend.

- **The sampling itself.** `n_sim` really is 500 on every real lap; the RNG is seeded (`:1321`) and
  two identical invocations returned identical dicts. I looked for a silent reduction (a config
  override, an env var, a fast path) and found none on any of the three surfaces.
- **Common random numbers.** I expected the projection branch to have lost CRN relative to the legacy
  branch, since it re-derives `pit_loss_s` and calls `project_positions` six-to-eight times per lap.
  It has not: every candidate and both neutralisation regimes see the same draws. Instrumented and
  confirmed identical std across candidates on all 110 laps.
- **Hypothesis 2 as a partial explanation.** I tried to keep integer rounding alive as a contributing
  cause. It is not one: lap 24 PIT_NOW has fully integer-collapsed positions and still yields 500
  distinct payoffs via the continuous margin term. Integer positions do not flatten anything.
- **"The cliff value is why the tests pass."** Re-ran the shipped sweep at realistic cliffs and with
  the cliff deleted entirely: 80.7% -> 81.5%. My hypothesis was wrong.
- **"The 3-car test field is why the tests pass."** Re-ran at 6/10/14/19 cars: PIT_NOW wins *more*,
  not less. Wrong again, and in the opposite direction.
- **A sentinel collision in the score plumbing.** I specifically hunted for `None` -> `0.0` in the
  four places `scenario_scores` is flattened or consumed. Both flatteners drop the key correctly and
  the arcade one carries a comment explaining exactly this. The only scale defect I found was the
  `confidence` fallback (F11), which is a different mechanism.
- **A `Series.get(k, default)` NaN leak** in the MC/projection path. Not present; the boundary is
  consistently `_finite_or_none`.
- **`OVERCUT` being permanently dead.** It is eligible on 6 of 110 laps, so the "search space is 3"
  framing is wrong; it is ~2.9 on average and OVERCUT's rarity is a correct consequence of its
  precondition.
- **An error path in the harness.** 110 laps, two races, zero exceptions, so none of the above is an
  artefact of a degraded run.

