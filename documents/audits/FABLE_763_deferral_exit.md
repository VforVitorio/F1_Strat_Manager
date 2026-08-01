# FABLE EXIT GATE #763 — the deferral tyre liability, attacked before promotion

**Date:** 2026-07-31 · **Branch:** `feat/deferral-tyre-liability` @ `4652c48` · **Gate type:** adversarial EXIT gate

> ⚠️ **This first half describes `4652c48`. The three HIGH findings were fixed at `7dd4751` and
> re-measured — see [RE-RUN AFTER THE FIX](#re-run-after-the-fix--branch-feat-deferral-tyre-liability--7dd4751)
> at the bottom for the post-fix numbers, including the E4 amplification that decides whether the
> term ships. The severity tally below is the PRE-fix state and is deliberately not edited.**
**Mandate:** success = finding what is STILL broken. No repository file modified except this report
(any temporary mutation is backed up with `cp` and restored from the backup, diffed, and stated).
All evidence executed, `profile="no-llm"`, zero API calls. The agreement metric's direction is
NOT evidence for or against the design (forbidden loop).

---

## Checklist

- [x] ⭐ PRIMARY — the amplification, quantified: **6-8×, measured** (design gate predicted 4-6×)
- [x] A. E3 invariance — re-derived independently: **HOLDS exactly, under both readings of "first"**
- [x] B. The two-branch minimum — **both branches win on real inputs**; `k = 0` right for `deg >= 0`
- [x] C. Double counting — **disjoint, verified**, including `laps_remaining < window_laps`
- [x] D. The cliff onset — **wrong, not merely simplified: it contradicts the model's own
      `cliff_laps`, is the SOLE source of the liability on 310 laps, and is untested**
- [x] E. E5's decomposition — 14 ✓, +4 ✓, but **"2 lost" is really 3 lost and 1 gained**
- [x] F. `pending=None` — **0 of 2,744 laps**; the real inertness is `deg is None` (42.6%)
- [x] G. Mutation battery — **4 real mutants, 4 survivors (incl. deleting the feature's wiring);
      3 sanity controls, 3 caught**
- [x] Local bug classes: wrong-mechanism comment **in the changed function's own docstring**;
      empty-set assertion **found, positive-controlled, passes for the right reason**; the
      `simulate_lap_window` twin **defensible, but its comment is now conditionally false**

### Severity tally

| | count |
|---|---|
| **HIGH** | 3 — the untested disconnection (§G / mutant M5) · the failed E4 amplification, evidenced twice (population-range flips **and** plausible-error flips at 6-8×) · the cliff term charging an assumed onset that contradicts the model's own `cliff_laps`, on 310 laps where the tyre reading is ≤ 0 |
| **MEDIUM** | 6 — wrong mechanism in the changed docstring · the q_f discount on a stopless branch · E2/E4 skipped · deg reaching its bounds · the Barcelona concentration · two shipped tests now passing for the wrong reason |
| **LOW** | 6 — dead `else` branch · E5's "2 lost" · E5's unstated "first" definition · E1's "green-flag" label · docs-site drift · the legacy twin's comment |

Findings appended below as confirmed, each with file:line, a concrete failing scenario, and
executed evidence.

### The verdict in one paragraph

**The term is physically motivated, correctly bounded, and its headline claims are true.** E1's
13-lap median is real, E3's invariance is exact, the double counting is disjoint, both branches
of the minimum genuinely win, the liability can never exceed the stop it defers, and the twin
scorer really does not need it. **What is broken is one half of the formula and everything
AROUND it.** The suite does not test that the term is connected, so the feature can be deleted
from the scorer with 173 tests green. The design gate's own E4 was skipped and, executed here,
fails: a tenth of a second per lap of tyre error now flips 8.6% of elective decisions against
1.0% before — a 6-8× amplification, at the top of the band the gate predicted. The cliff half of
the formula ignores the tyre model's own `cliff_laps` and is, on 310 real laps, the ONLY thing
charging a liability at all — including on tyres the model calls no worse than fresh. And the
function's own docstring still tells the reader the opposite of what the code does. None of that
argues for reverting; it argues the term shipped without the two gates specified for exactly this
risk, and that its cliff half wants the fix its wear half already has.

---

## A + E. The invariance and the decomposition re-derived independently — VERIFIED, with one wording defect

**Method (mine, not the author's script):** extracted the before report from `f3f0207`
(`harness_sha d97a54e`, 54 scored / 79 no_call / 24 no_boundary / 18 exact — matches E5's
"before" column) and joined it against the committed after report (`4652c48`) on
`(year, race, driver, actual_lap)`. Key sets are IDENTICAL: 178 = 178, zero mismatches, so the
comparison is like-for-like (`scratchpad/rederive_e3_e5.py`).

**Verified, executed:**

- **Declines 79 → 65 (−14) ✓**, decomposing as 7 `no_call → no_boundary` + 7 `no_call → scored`.
- **Scored 54 → 58 (+4) ✓** = 7 in − 3 out (`scored → no_boundary`: VER/OCO/HAD Barcelona).
- **Exact 18 → 16 ✓.**
- **E3 invariance holds under BOTH definitions of "first".** Under the natural reading of E5's own
  sentence — the driver's first stop *within the graded verdict list* — the split is **114/64**,
  not 89/89; under "first stop of ANY kind from the raw `PitInTime` data, including neutralised
  stops" the split reproduces E5's table EXACTLY (89/89; FIRST scored 38→38, exact 10→10,
  no_call 29→29, no_boundary 14→14). Under EITHER split, **zero** first stops changed bucket and
  **zero** first-stop chosen laps moved; all 19 changed stops are elective under both. The
  invariance claim is genuine and robust to the definition.
- **Chosen-lap churn:** among the 51 stops scored in both runs, exactly 2 moved — ANT Barcelona
  49→45 and BOR Barcelona 49→44, both elective, both 4-5 laps EARLIER.

**FINDING [LOW] — E5's exact-agreement decomposition says "two that used to land exactly no
longer do"; the truth is THREE lost, ONE gained** (`documents/audits/MEASURE_763_deferral_effect.md`,
"the exact-agreement fall decomposes" section). Executed: lost exact = ANT Barcelona 49 (now
chosen 45), BOR Barcelona 49 (now 44), **OCO Barcelona 43 (exact before, now
`no_boundary_in_window` — not graded at all)**; gained exact = PIA Monaco 48 (was a decline).
Net −2 is right, but the honest sentence is "three exact calls were lost — one of them pushed
clear out of the gradable window — and one new exact appeared." An exact agreement that the term
converts into *no locatable decision* is a qualitatively worse trade than an exact agreement that
drifts two laps, and the current wording hides that shape.

**FINDING [LOW] — E5 does not state that "the driver's first of the race" is computed from ALL
raw stops including neutralised ones.** A reader re-deriving the split from the committed
verdicts JSON alone (which contains only the 178 green-flag stops) gets 114/64 and different
levels in every cell of the E3 table (e.g. elective decline rate 53.1% → 31.3% instead of the
published 56.2% → 40.4%). The deltas are identical, so no conclusion changes, but the table is
not reproducible from the artefact it sits next to without the unstated definition.

**Two smaller provenance slips in the same header line** (`MEASURE_763_deferral_effect.md:5-6`,
both LOW, folded in here rather than counted separately): it says *"Harness `f3f0207` (before)"*
while that report's own `harness_sha` is `d97a54e` — the doc names the commit that COMMITTED the
report, the artefact names the harness that RAN it, and a reader chasing either will not find the
other. And it says *"against `311f234` + the deferral term"*, but `311f234` is a descendant of
`bcd2c9d`, so it already contains the term; the "+" double-counts it. Neither changes a number.

**"One production change between them" — VERIFIED.** `git log d97a54e..311f234 -- src/ scripts/`
shows three production commits, but the two besides the term cannot reach the measured number:
`4520431` changed docstrings and LLM prompt strings only (the no-llm action never sees a prompt),
and `285f33b`'s previous-lap sentinel flows into `pace_out`, which is not among
`_run_projection_mc`'s inputs — the no-llm action is `best_mc_candidate(mc_results)` + rails
(`src/strategy/inference/no_llm.py:290-294`), and the projection MC takes rivals / position /
laps_remaining / pit_context / draws / deg only (`strategy_orchestrator.py:1441-1453`). The
attribution of the 79→65 move to the deferral term is sound.

---

## Structural findings (code-level, confirmed by inspection; quantification follows below)

### FINDING [LOW] — dead `else` branch in `_terminal_gaps` duplicates the None rule and would
### change its meaning if it ever became reachable

`src/agents/position_projection.py:795-799`. The branch is unreachable: a non-stopping plan with
`pending is None` already returned at line 783-784, a stopping plan takes line 786-787, True and
False take 788-794 — no input reaches the `else`. Two defects in one: (1) dead code carrying a
comment ("exactly as it did before this term") that claims to BE the live None path; (2) the dead
branch is not equivalent to the live one — the early return at :783 suppresses `their_residual`
too (the "rule binds BOTH sides" behaviour of the docstring, tested by
`test_an_unknown_obligation_of_OURS_suppresses_the_credit_too`), while the else would fall
through to line 801-811 and apply the rivals' residuals. If a future refactor deletes the early
return trusting the else to cover None, the None semantics silently flip from "no claim either
way" to "claim about them but not about us". One rule, two homes — this repo's documented drift
pattern.

**And there is one input that DOES reach it**, which makes the divergence live rather than
latent: a **numpy** boolean. Executed: `np.True_ is True` → `False`, `np.False_ is False` →
`False`, so a `mandatory_stop_pending` arriving as `np.bool_` (anything sourced from a pandas
column rather than from `stint_history.py`'s Python literals) passes every `is` check, lands in
the `else`, and is scored as "unknown-but-still-credit-the-rivals" — the exact half-claim the
docstring at `:770-777` says the module must never make. The three shipped producers all emit
Python bools today, so this is not a current bug; it is a silent wrong answer waiting for the
first pandas-sourced caller, and the `is` comparisons are what make it silent.

### FINDING [MEDIUM] — the cliff comment names the wrong mechanism, and the "purity" defence is
### false: the per-draw `cliff_laps` is available at the call site and simply not passed

`src/agents/position_projection.py:716-722`. The comment says using the config window as the
cliff onset "keeps this function pure and its inputs already-measured". Purity is not the
constraint: `project_positions` holds the per-draw `cliff_laps` at the moment it calls
`_terminal_gaps` (`position_projection.py:861`) and passes `pit_loss_s` — another per-draw
array — right next to it. Passing `cliff_laps` through would be exactly as pure. The real
simplification is an unthreaded argument, and it has a price in both directions:

- **set past the cliff at the window edge** (per-draw `cliff_laps` < window): the run-out branch
  grants `window_laps` cliff-free laps that do not exist — undercharges the run-out by up to
  `window_laps * cliff_loss_s` = 5 × 0.80 = **4.0 s**, biasing toward STAY_OUT;
- **fresh set, cliff far away** (per-draw `cliff_laps` > 2·window): the run-out is charged cliff
  laps that never happen — e.g. `cliff_laps=20`, `remaining=25` overcharges
  `(20−10−5)... = (25−15)−(25−20+...)` → modelled 20 cliff laps vs true 10, **+8.0 s** on the
  run-out, biasing toward saturation at the residual and thus toward PIT.

The onset is exact only when the draw's cliff sits exactly `2·window_laps` laps out.

**Executed on the 2,744 real laps** (variant: same function with the cliff onset taken from the
caller's per-draw `cliff_laps` instead of the config window, everything else identical):
the variant **flips the argmax on 60 laps** and moves `E[STAY_OUT]` by up to **3.0 positions**
(p95 of the delta). So the "simplification for purity" is worth three positions and sixty
decisions on one six-race sample, and it has **zero test coverage** (§G: the test fixture pins
`cliff_loss_s: 0.0`). **Upgraded to HIGH by later execution — see "the liability is charged
ENTIRELY by the cliff term on 310 laps" below.** The docstring also gives a reason ("keeps this
function pure") that is not the real one, and no test would notice if the branch were deleted.

### FINDING [MEDIUM] — the q_f discount on the run-it-out branch credits an option that branch
### would never exercise

`src/agents/position_projection.py:723-725` discounts `run_it_out` by
`q_f · neutralisation_saving_s`; the docstring's justification (:693-694) is "a neutralisation
that turns up covers a deferred stop whichever branch wins". The run-it-out branch contains **no
stop** — there is nothing for a Safety Car to cover. Concrete failing scenario: `remaining=20`,
`deg=0.30` → wear-to-flag 6.0 s; `q_f=0.4`, saving 8 → charged `max(0, 6.0−3.2)=2.8 s`. If an SC
actually arrived, converting to a stop would cost `pit_loss − saving ≈ 14.8 s` — more than the
6.0 s of remaining wear — so the option would never be exercised, yet the liability was cut by
more than half for it. Because both branches subtract the SAME `q_f·saving`, the discount never
changes which branch wins (min(a−d, b−d) = min(a,b)−d up to the zero clamps); its entire effect
is to shrink the liability by up to ~3.3 s uniformly — always in the STAY_OUT direction, on
exactly the population the term was built to stop under-charging. The mechanism the docstring
names is wrong; the correct option value of the run-out branch is
`q_f · max(0, wear_after_arrival − (pit_loss − saving))`, which is ~0 in the scenario above.

**Executed magnitude on the real laps:** `q_f` runs p25 0.295 / median 0.405 / p75 0.502 / max
0.699, so the unearned discount is **median 3.24 s, up to 5.59 s** per lap, and on **229 laps
(12.5% of the elective population) it drives the run-it-out branch to exactly zero on its own** —
i.e. on one elective lap in eight, the entire deferral cost is cancelled by an option the branch
cannot exercise. That is a systematic under-charge of the very population the term exists to stop
under-charging, and it is the only part of the formula whose direction is not conservative.

## ⭐ PRIMARY — the amplification, quantified on 2,744 real 2025 laps

**Method:** every `_run_projection_mc` call of a full `measure_decision_agreement()` run was
captured with its exact kwargs (2,744 laps, reproducing the committed report to the digit), then
each lap was re-scored through the production functions under variants that swap only
`_deferral_tyre_liability_s` or `deg_cost_s`. Argmax uses the production tie-break
(`best_mc_candidate`), never a reimplemented `max`. Scripts:
`scratchpad/{capture_projection_inputs,analyze_captures,sensitivity_small}.py`.

### The liability's size

```
pending=False laps: 1839      (the population the term is scoped to)
liability, GREEN config, seconds:   p5 0.00  p25 0.00  p50 0.00  p75 18.62  p95 21.76  MAX 28.57
liability in POSITIONS (E[STAY_OUT] with the term off minus on):
                                    p25 0.00  p50 0.00  p75 2.00  p95 5.80  MAX 11.99
fully saturated at the residual cap: 530 laps (28.8%)     partially: 43
```

Flipped laps are characterised by `laps_remaining` median 33 (max 62) and `deg` median 0.97 —
i.e. the term acts EARLY in a race on a visibly worn set. At those horizons the run-it-out branch
prices holding the set for 30-57 more laps, which no set survives; the `min` correctly caps it at
the residual, so the arithmetic lands right through a branch that is physically fictional. One
consequence worth naming: in the 28.8% saturated regime, an elective stop is priced **identically
to a mandatory one**, which erases the pending=True/False distinction the design gate's whole
analysis rests on — for a defensible reason (the tyre must be changed eventually), but nowhere
stated.

**The median is zero and the p75 is eighteen and a half seconds.** The distribution is bimodal:
inert on ~two thirds of the population (no reading, or a reading ≤ 0), then jumping to near the
full pit loss. **On 28.8% of elective laps the liability SATURATES** — the run-it-out branch
exceeds the whole discounted pit loss, so the `min` returns the residual and staying out is
charged an entire pit stop. **Its worst case moves the terminal position by 11.99 places on a
single lap.**

To the mandate's question — *can the term produce a liability larger than any plausible real
cost?* — **no, and that is structural, not luck:** the `min` at `:727` caps it at
`_stop_residual_s(pit_loss_s)`, so the most staying out can ever be charged is the stop it is
avoiding. The 28.57 s maximum is a long-pit-lane circuit's own pit loss. **The cap is the term's
strongest property and it is untested (see §G, M5/M3).**

### The argmax impact

```
argmax flips, term ON vs term OFF: 338 of 2744 laps (12.3%) — every one on pending=False
  STAY_OUT -> UNDERCUT : 250
  STAY_OUT -> PIT_NOW  :  88
```

**Every one of the 338 is `pending=False`** — the scoping is airtight at the lap level, not only
at the verdict level, which is a stronger form of E3 than E5 could show (18.4% of the elective
population flips; 0% of the mandatory one). All flips are in one direction (stop instead of stay
out), as designed. Note **UNDERCUT takes
three quarters of them**: the term's practical effect is less "call the elective stop" than "call
the elective stop AS an undercut", which neither MEASURE doc mentions and which routes the
decision through `ucut_prob` — a 0.5 prior whenever `pit_out` is None, which is **always** on the
no-llm path this tier measures (`src/strategy/inference/no_llm.py:274`, N28 never runs). Worth
stating: on the measured configuration, 250 of the 338 recovered calls are gated behind a
coin-flip undercut probability.

### FINDING [MEDIUM] — the effect is not distributed across the stratified sample: it is
### Barcelona, and one whole circuit archetype receives it on 0.9% of its laps

The six races were chosen for circuit archetype, and E5 reports the result as a property of the
sample. Executed per race:

| race | laps | pending=False | deg known | liability > 0 | argmax flips |
|---|---|---|---|---|---|
| **Barcelona** | 737 | 522 | 456 | 425 | **216** |
| Monaco | 558 | 329 | 241 | 213 | 70 |
| Marina Bay | 308 | 168 | 141 | 128 | 30 |
| Monza | 219 | 86 | 59 | 49 | 11 |
| Lusail | 313 | 153 | 105 | 82 | 10 |
| **Silverstone** | 609 | **581 (95%)** | **53** | **5** | **1** |

**Barcelona alone carries 64% of the argmax flips from 27% of the laps**, and the E5 verdict
changes agree: 15 of the 19 changed stops are Barcelona, 4 are Monaco, and the other four races
contribute **zero**. **Silverstone is the sharpest case**: 95% of its laps are elective (the
wet-compound exemption discharges the obligation for nearly everyone), which should make it the
term's ideal population — and the tyre channel has a reading on only 53 of 581, so the liability
is non-zero on **5 laps** and flips **one** decision. The archetype the sample includes to cover
variable weather is exactly the one where the fix cannot reach, because the same wet running that
makes the stop elective is what denies `_fresh_reference` its lookup. E5's "fourteen fewer
declines" is a real number and a Barcelona-and-Monaco number; it should not be read as
sample-wide.

### FINDING [HIGH] (evidence 1 of 2) — the design gate's own E4 criterion FAILS: the argmax
### flips well inside the tyre reading's range

Substituting the observed p1 (−2.262) or p99 (+3.670) of `deg_cost_s` — both *inside* the
measured `[-2.33, +3.67]` bound, both values real laps actually carry — flips the argmax on
**403 (38.2%)** and **501 (47.5%)** of the 1,055 elective laps with a reading. The design gate
pre-registered the criterion in `DESIGN_763_window.md` §E4: *"A formulation whose recommendation
flips inside its input's confidence interval is not measured yet."* E4 was never executed before
shipping (see the E2/E4 finding above); executed here, **it does not pass**.

Caveat stated because it matters: p1/p99 are the *population* range of `deg_cost_s`, not a
per-lap confidence interval — #744a published no per-lap CI, which is itself part of why this
criterion could not be met honestly. So the finding is nailed down with a plausible-error version
below, which does not depend on that distinction.

### FINDING [HIGH] (evidence 2 of 2) — the amplification the design gate predicted at 4-6×
### MEASURES at 6-8×: a tenth of a second per lap of tyre error now flips one decision in twelve

The design gate's closing sentence was: *"B3 prices the deferral with the SAME `deg_cost_s` whose
level FABLE_G2/#744a showed to be reference-sensitive, now multiplied by `laps_remaining` instead
of 5 — an error in deg is amplified ~4-6×. E4's sensitivity disclosure exists precisely for this;
if deg's CI cannot support the amplification, B4 (or B5 as the interim) is the honest fallback."*

Executed on the 1,055 elective laps carrying a reading, perturbing `deg_cost_s` by a
measurement-error-scale amount in either direction, and re-running the SAME laps with the term
wired and with it stubbed to zero so the amplification is isolated rather than inferred:

| deg error | argmax flips WITH the term | flips WITHOUT it | **amplification** |
|---|---|---|---|
| **±0.1 s/lap** | **91 (8.6%)** | 11 (1.0%) | **8.27×** |
| ±0.2 s/lap | 176 (16.7%) | 29 (2.7%) | 6.07× |
| ±0.5 s/lap | 336 (31.8%) | 49 (4.6%) | 6.86× |

**The predicted risk is real and is at the top of the predicted band.** A 0.1 s/lap error in the
tyre channel — well inside what a per-stint fresh reference can be wrong by, given that #744a
refuted two designs for it and #744b measured the shipped one at corr +0.369 — used to change 1%
of elective decisions and now changes 8.6%. Flips are two-sided and symmetric (`STAY_OUT →
UNDERCUT` on +187 / `UNDERCUT → STAY_OUT` on −184), i.e. these laps sit on the knife edge, not in
a robust regime.

**This is the finding the gate asked for and it did not pass.** It does not say the term is
wrong — the physics argument (E1's 13-lap median) stands, and the liability's cap is sound. It
says the term's output inherits, six- to eight-fold, an input error nobody has bounded, and that
E4 should have been run before E5 rather than after promotion.

## G. The tests — HIGH: the whole feature can be DISCONNECTED and all 173 tests stay green

**Method:** `src/agents/position_projection.py` was `cp`-ed to `scratchpad/position_projection.py.bak`
first; each mutant was written to disk, `uv run pytest tests/mc/ -q` run, and the file restored
with `write_bytes(original)` from that backup. Baseline before any mutation: **173 passed**.

| Mutant | file:line | Result |
|---|---|---|
| **M5 — the wiring deleted**: `our_residual = _deferral_tyre_liability_s(...)` → `np.zeros(...)` | `position_projection.py:794` | **SURVIVED (173 passed)** |
| **M3 — the option value deleted**: `stop_later = _stop_residual_s(pit_loss_s, config)` → `np.asarray(pit_loss_s)` | `position_projection.py:714` | **SURVIVED (173 passed)** |
| **M1 — the cliff onset removed**: `max(0, remaining - window_laps) * cliff` → `remaining * cliff` | `position_projection.py:720-722` | **SURVIVED (173 passed)** |
| **M2 — the run-out's q_f discount deleted** entirely | `position_projection.py:723-725` | **SURVIVED (173 passed)** |
| M4 — SANITY control: `np.minimum` → `np.maximum` (charge the DEARER future) | `position_projection.py:727` | CAUGHT (3 failed) |
| M6 — SANITY control: `laps_remaining - window_laps` → `laps_remaining` (double-cover the window) | `position_projection.py:707` | CAUGHT |
| M8 — SANITY control: `remaining * deg` → `0.5 * remaining * deg` (half the wear) | `position_projection.py:720` | CAUGHT (1 failed) |

**Score: 4 real mutants, 4 survivors; 3 sanity controls, 3 caught.** The controls are the proof
the battery works and that M1/M2/M3/M5 genuinely survive rather than the harness silently
no-op'ing. **Restore verified byte-identical to the backup** after both batteries (`git diff`
empty; `diff` against the `.bak` reports identical).

The three controls were caught by just two tests —
`test_the_liability_shrinks_as_the_flag_approaches` (M4, M6) and
`test_the_liability_is_the_cheaper_of_the_two_futures` (M8) — both of which key on the `deg`
factor. Eight tests in the module; two of them carry the whole defence, and between them they
cover two of the formula's five terms.

**M5 is the HIGH.** Ripping the term out of `_terminal_gaps` entirely — restoring the exact
pre-#763 behaviour that the whole PR exists to change — leaves the entire `tests/mc` suite green,
including the dedicated `tests/mc/test_deferral_tyre_liability.py`. The reason is structural: of
the module's eight tests, four call `_deferral_tyre_liability_s` **directly** (never through the
scorer) and the four that do go through `_terminal_gaps` all assert the term does **NOT** apply
(`pending=True` → not called; `pending=None` → gaps unchanged; `stops_in_window` → gaps
unchanged) or are the monkeypatched invariance test. **Not one test asserts that a pending=False
non-stopping plan's terminal gap actually MOVES.** A regression that silently disconnects the
feature — a refactor of the if/elif chain, a merge, a revert — ships green. This is the repo's
own documented shape (`feedback_a_guard_that_asserts_nothing`): the suite proves the term is
correctly *not* applied in three cases and never proves it is applied in the one that matters.

**M1, M2 and M3 are the same gap, executed three ways.** The test fixture
(`tests/mc/test_deferral_tyre_liability.py:39-51`) pins `cliff_loss_s: 0.0`,
`neutralisation_saving_s: 0.0`, `future_neutralisation_prob: 0.0`, so **three of the five config
inputs the function reads are held at their no-op value in every test in the module**. No test
elsewhere in `tests/mc` covers them either (`test_tyre_wear_term.py:328,443` pass
`pit_context=None` → the flag is `None` → the term is inert). Result: **every mutant that touches
the cliff term or either q_f discount survives, and every mutant that touches the `deg` factor or
the `min` is caught.** Of the five terms in the formula, the suite defends two.

Executed proof rather than inference: M1 changes the cliff onset by up to 4.0 s per lap of
liability, M2 deletes a median-3.24 s discount outright, M3 removes the option value from the
other branch — **173 passed, three times.** Meanwhile M6 and M8, which mutate the SAME expression
as M1 but on its `remaining` and `deg` factors, are both caught. The suite is not weak; it is
**precisely blind along the two axes the fixture zeroes.**

**Degenerate-assertion check.** `test_a_car_that_still_owes_its_stop_is_untouched` (`:93-114`)
asserts `called == []` — an assertion about the empty set, which passes both when the patch works
and when it silently fails to intercept. Positive control executed against a byte-identical copy
of the restored file:

```
pending=True  (the shipped test's case): called = []
pending=False (the positive control)   : called = [1]
```

The patch does intercept, so the test passes for the right reason. **Not a defect** — but it
needed the control to know, and the test carries none of its own.

### FINDING [MEDIUM] — two shipped tests now pass for a different reason than their names claim

Both live in `tests/mc/test_position_projection.py` and both assert that a discharged obligation
makes staying out **free**, which the term has just made false in production:

- `test_having_already_stopped_removes_the_cost_entirely` (`:271-275`) — "a second set buys
  nothing, so staying out is free", asserting terminal cost `== 0.0` with `pending=False`;
- `test_a_wet_race_exempts_the_mandatory_stop_so_staying_out_is_free` (`:509-517`) — same
  assertion, framed as the Art. 30.5(m) wet exemption "removing the liability entirely".

They stay green only because `_flat_config` (`:84-95`) never sets `deg_cost_s`, so the dataclass
default `None` (`position_projection.py:255`) short-circuits the new term at `:704`. **Executed**
against the pristine module:

```
test_having_already_stopped_removes_the_cost_entirely, as shipped (deg unset): terminal cost 0.0
                                  same fixture + deg_cost_s=0.5, laps_remaining=25: terminal cost 1.0
test_a_wet_race_exempts_..._staying_out_is_free,      as shipped (deg unset): terminal cost 0.0
                                  same fixture + deg_cost_s=0.5, laps_remaining=25: terminal cost 1.0
```

Give either fixture a real tyre reading and both assertions fail — correctly, because staying out
is no longer free for a discharged obligation. Two tests whose NAMES state a rule the code no
longer follows, surviving on an unset field: the exact "assertion passing for the wrong reason"
shape, and the place a future maintainer will look to learn what the module guarantees.

## F. The scoping's honesty — REFUTED in the mandate's direction, but a bigger inertness found

**Executed** (`scratchpad/capture_projection_inputs.py`: every `_run_projection_mc` call during a
full `measure_decision_agreement()` run, in-process wrapping only). The run reproduced the
committed report EXACTLY — 2,744 laps captured, 178 verdicts, 58 scored, exact 0.2759, no_call 65,
no_boundary 34 — so the captures describe the shipped measurement, and they independently confirm
the design gate's "2,744 eval laps, 100% projection path".

```
mandatory_stop_pending over the 2,744 eval laps:
  True   905  (33.0%)
  False 1839  (67.0%)
  None     0  ( 0.0%)     <- the hypothesis that None hides most of the population is REFUTED
```

**But the term is inert on 42.6% of the population it IS scoped to, for a different reason
nobody stated:** `deg_cost_s is None` on **784 of the 1,839** pending=False laps
(`position_projection.py:704-705` returns zeros), so an elective stop's pit loss still stands
against exactly zero on more than four in ten of the laps the report describes as fixed. Neither
E5 nor the commit message mentions this. The honest statement of the term's reach is *67% of laps
by obligation, 38% of laps once the tyre reading is required*.

## FINDING [MEDIUM] — `deg_cost_s` reaches its measured bounds on real laps, and 41% of its
## values are at or below zero

Over the 1,055 pending=False laps with a reading: p1 −2.262, median **0.159**, p75 0.902,
p99 **3.670 — the ceiling itself**; 15 laps (1.4%) sit at the +3.67 ceiling and 10 (0.9%) at the
−2.33 floor, so `MEASURE_744a`'s clamp (`tire_agent.py:248-249`, applied in `_referenced_wear`)
is load-bearing, not theoretical. **19.4% of readings are NEGATIVE** and **21.9% are exactly
0.0** — on 41.3% of elective laps with a "reading", the tyre channel says the current set is no
worse than fresh. The liability is identically zero on **937 of 1,839 elective laps (51.0%)** and
positive on 902 (49.0%).

**Correction to my own first pass, stated rather than silently fixed:** I initially wrote that
`deg <= 0` "drives the liability to zero" and put the inert share at ~66%. **That is wrong, and
the reason it is wrong is the finding below** — the cliff term is charged independently of `deg`,
so a zero or negative tyre reading does NOT zero the liability. Measured: of the 231 laps with
`deg == 0.0`, **189 (82%) still carry a positive liability, median 8.41 s**; of the 205 with
`deg < 0`, **121 (59%) do, median 2.97 s**. Only `deg is None` reliably zeroes it (784 laps, all
zero).

### FINDING [HIGH] — on 310 real laps the liability is charged ENTIRELY by the cliff term, on
### tyres the model says are no worse than fresh, with an onset the model never supplied

This is the finding above (D) upgraded by execution. `run_it_out = remaining · deg +
max(0, remaining − window_laps) · cliff_loss_s` (`position_projection.py:720-722`) charges
`CLIFF_LOSS = 0.80` s/lap from `window_laps` laps out to the flag **regardless of the per-draw
`cliff_laps` the tyre model produced and the caller is holding**. Measured on the elective laps:
the cliff term charges a median of **19 laps → 15.2 s**, up to **57 laps → 45.6 s** — on its own
larger than any pit loss, so it alone saturates the `min`.

Concretely: on **189 laps the tyre model reports `deg_cost_s == 0.0` — "this set costs nothing
per lap versus fresh" — and the liability still charges a median 8.41 s**, entirely from an
assumed cliff. On another **121 laps the model reports NEGATIVE wear** (the set is faster than
the fresh reference) **and the liability is still positive.** The assumption "everything past lap
`window_laps` is past the cliff" is not merely unjustified, it directly contradicts the tyre
model's own `laps_to_cliff` distribution, which the caller samples into `cliff_s` and passes to
`driver_time_delta` on the very next line (`position_projection.py:837, 861`).

The in-window term gets this right — `driver_time_delta:621` uses `max(0, racing - cliff_laps)`,
the per-draw value. The terminal term does not. **One quantity, two rules, in the same
function's two horizons** — and the branch that gets it wrong has zero test coverage.

## Method note — one of my own probes was invalid, and how the file was protected

**Backup/restore discipline.** `src/agents/position_projection.py` was `cp`-ed to
`scratchpad/position_projection.py.bak` before any mutation; every mutant is written to disk and
then restored from that backup with `write_bytes`; the run asserts the final bytes equal the
backup. `git checkout --` was never used. End state verified two ways: `git diff --numstat` empty
and `diff` against the `.bak` clean.

**The invalid probe, stated rather than quietly dropped.** While a mutant was on disk, a probe of
mine imported the working-tree module and reported that the invariance test's monkeypatch "does
NOT intercept" — a false HIGH I nearly published. It was caught by diffing the working tree
(`git diff` showed `M5_wiring_deleted` applied). Fix adopted for every subsequent probe: load a
byte-identical PRISTINE copy through `importlib` and register it in `sys.modules` before anything
can import the working tree, asserting `pp.__file__` is the pristine path. Every executed number
in this report comes from either that pristine module or a verified-clean tree.

## FINDING [MEDIUM] — two of the design gate's four pre-conditions were skipped: E2 was never
## run, and E4 exists only as an input-CI remark, not the demanded sensitivity of the conclusion

`DESIGN_763_window.md` §E froze the sequence: "E1-E4 freeze the design; E5 is a single 2025
re-run". Executed check: **E2 (the pre-registered sign test on 2023-24 elective stops) has no
committed artifact, no script, and no mention in either MEASURE doc** — `grep` over
`documents/audits/*763*` and `scripts/` finds it only in the design gate that demanded it. **E4
("publish the conclusion's sensitivity to E1's CI ... a formulation whose recommendation flips
inside its input's confidence interval is not measured yet") was reduced to one paragraph in E1
saying the CI is ±1 lap** — a statement about the INPUT's tightness, not about whether the
LIABILITY or any argmax flips across `deg_cost_s`'s error band, which is the exact quantity the
gate's "single biggest risk" paragraph tied E4 to. The one-shot E5 was taken without the two
gates that were supposed to precede it. **This gate executed the missing E4 and it does not pass**
(6-8× amplification, ⭐PRIMARY) — so the skipped pre-condition was not a formality, and the
finding it would have surfaced is the report's largest.

## E1's basis re-executed — the numbers are REAL, the "green-flag" label is not

**FINDING [LOW] — E1's sample is labelled "1,377 real green-flag stops"; 184 of them (13.4%) fail
the repo's own green-flag definition, and the conclusion survives the correction.** Executed
(`scratchpad/e1_green_filter_check.py`): the committed method reproduces to the decimal (1,377 /
694 elective, never-repays 13.1%/10.8%, medians 14/13, within-5 9.7%/15.0%) —
`scripts/measure_repayment_horizon.py` counts every stint increment and applies NO TrackStatus
filter, while everything else in this repo that says "green-flag stop" means
`green_flag_stops(laps, _neutralised_laps(laps))` (stop lap or lap+1 neutralised → excluded,
`src/strategy/eval/projection.py:160-181`). Applying that filter: elective n=694→598, median
13→12, within-5 15.0%→16.7%, never-repays 10.8%→6.9%. Every conclusion holds — the median stays
far outside the 5-lap window — so this is a label defect, not a basis defect. The same phrase is
propagated into `_deferral_tyre_liability_s`'s docstring (`position_projection.py:680`), the
commit message of `bcd2c9d` and the test module docstring. "Green-flag" should either be applied
or removed from the sentence.

## C. Double counting — VERIFIED DISJOINT, with one uncharged band noted

Traced at file:line, both charges for a non-stopping plan:

- **In-window wear:** `driver_time_delta` else-branch (`position_projection.py:620-623`) charges
  `racing_laps · deg` plus `max(0, racing_laps − cliff_laps) · cliff_loss`, with
  `racing_laps = min(WINDOW_LAPS, laps_remaining)` under green
  (`strategy_orchestrator.py:1205`, `_bounded_by_race_end`).
- **Terminal liability:** `max(0, laps_remaining − window_laps)` laps of deg
  (`position_projection.py:707`), where `laps_remaining = total_laps − current_lap`
  (`strategy_orchestrator.py:870`).

The window owns the first `window_laps` of the remaining race; the liability owns the rest.
Disjoint in every case I could construct: `laps_remaining ≥ window` (green) partitions exactly;
`laps_remaining < window` clamps the liability to zero while the in-window charge covers the
clamped `racing_laps`; a plan that stops takes the zero branch at `:786-787` before the liability
is reachable. The `laps_remaining < window_laps` edge the mandate asked about is handled by
`max(0, ·)` at `:707` plus the `remaining <= 0` early-out at `:708`.

**Note (not a finding): under a neutralised config the band is under-, never over-charged.** With
`racing_laps ≈ 1.36` under SC, the laps between `racing_laps` and `window_laps` carry wear in
NEITHER term (in-window stops at 1.36 racing laps; the liability still subtracts the full 5).
Deliberate per the module's "non-racing laps cost nothing relative" stance, and in the
conservative direction.

## B. The two-branch minimum and the k = 0 collapse — HOLDS within the model; the comment's
## premise is false for negative deg but the arithmetic self-corrects

- **k = 0:** for `deg_cost_s ≥ 0`, every lap deferred past the window edge adds `deg` (and
  possibly cliff) without reducing the pit loss, so the best later stop is the window edge —
  the comment at `:711-713` is right. For `deg_cost_s < 0` (floor −2.33: the set is FASTER than
  the fresh reference, `tire_agent.py:248`) the premise "every further lap adds wear" is FALSE —
  the best later stop is as late as possible — but the run-it-out branch then goes negative, the
  zero clamp floors it, and the min returns 0: the code lands on the right answer (deferral is
  free, never a credit) through a branch the comment does not describe. No behavioural defect;
  the comment states an unconditional premise that holds only for `deg ≥ 0`.
- **A regime where waiting longer is genuinely cheaper** exists only through a time-varying
  neutralisation hazard (waiting INTO a likely SC window), which the model deliberately flattens
  into a single `q_f` over the whole remaining race. Within the model, no such regime exists;
  outside it, this is the documented approximation `_stop_residual_s` already makes for
  pending=True, so the term is no worse than its sibling.
- **Both branches winnable on real inputs — VERIFIED, executed.** Over the captured laps'
  `(lap, config)` calls, the `stop_later` branch wins the `min` on some draw in **1,146** calls
  and the `run_it_out` branch on some draw in **2,618**; mean per-call saturation share 29.9%.
  Neither branch is decorative, and 43 laps split the two WITHIN one lap's draws (the per-draw
  pit-loss sample crossing the run-out cost), which is the regime the `min` exists for.

### FINDING [MEDIUM] — the changed function's OWN docstring still states the pre-change
### behaviour, 40 lines above the new branch

`src/agents/position_projection.py:753-754`, inside `_terminal_gaps`'s docstring, list of "the
three cases the deleted rail was patching":

> `- already stopped (no obligation) -> our residual is zero, staying out costs nothing on this term;`

That is precisely the case this PR changed: `pending is False` now carries
`_deferral_tyre_liability_s`, so staying out costs up to a full pit loss on this term. A reader
of the function that contains the change is told the opposite of what the code below does — the
canonical "comment naming the wrong mechanism" shape, and the strongest instance in this diff
because it lives in the same docstring.

`ProjectionResult.terminal_positions`' docstring (`:277-282`, "once every KNOWN outstanding stop
has been served") and `ProjectionConfig.mandatory_stop_pending` (`:234-236`, "``None`` ... disables
the liability term") are both now incomplete in the same direction: the terminal horizon carries
a tyre liability that is not an outstanding stop, and `False` no longer means "no liability".

### FINDING [LOW] — the docs site describes the terminal liability as stop-only, in three places,
### and none moved with the code

`docs/pages/multi-agent.md:237` (`terminal_liability … the deferred mandatory stop, discounted by
q_f`), `:249` ("the terminal liability applies only to the candidates that do not stop, because a
deferred obligation is a cost only while it is still owed" — the second clause is now false: the
liability now also applies when nothing is owed), and `:259` ("a still-owed stop costs the cars it
will release behind us"). The repo's own lesson of 2026-07-16 is that "when a fix changes a
contract, the page that describes it is part of the fix". Same class as the docstring finding
above, lower severity because the docs site is not what the next maintainer edits against.

### The twin question, answered: `simulate_lap_window` does NOT need this term, and that is a
### defensible asymmetry rather than the repo's dominant defect repeating

The repo's dominant defect is "one copy fixed, its twin not", so shipping to one scorer had to be
attacked. It survives, on executed evidence rather than on the author's reasoning:

1. **The legacy scorer decided 0 of the 2,744 eval laps** — independently reconfirmed by my own
   capture: every lap routed to `_run_projection_mc`. The two scorers do not share the measured
   population.
2. **The legacy scorer has no terminal horizon at all.** `simulate_lap_window`
   (`strategy_orchestrator.py:693-782`) scores a W-lap window relative to STAY_OUT and has no
   `_terminal_gaps` analogue, no `mandatory_stop_pending` input, and no rivals. There is no place
   to put a terminal liability without redesigning it.
3. **It charges the stop at 2.8 s, not 22.8 s**, so the defect the term fixes (an elective stop's
   full pit loss standing against zero) does not exist there in the same magnitude.

So this is not the twin pattern. What IS owed is the comment correction below.

### OBSERVATION [LOW] — the legacy twin's justifying comment is now documented-false for 41% of
### real stops, and nothing marks it

`src/agents/strategy_orchestrator.py:723-726` (docstring) and :762-765 (OVERCUT comment) argue
the pit-lane traversal cancels because "the two-compound rule makes a stop mandatory, so pit-now
and pit-later both pay it". #763's own census establishes that premise is false for 73 of 178
real 2025 stops (elective, obligation discharged) — on the legacy path an elective STAY_OUT
would genuinely avoid the traversal, and the comment's unconditional "mandatory" is now known
wrong for that population. Not a regression from this PR (the legacy scorer decided 0 of 2,744
eval laps, and its 2.8 s pit term accidentally under-prices rather than over-prices the elective
stop), and shipping the term to one scorer is defensible on the executed census — but the twin's
comment should stop stating an unconditional premise this branch just measured to be conditional.

---

## Fix list, ordered by value against risk

Numbered, and deliberately separate from the findings. **Nothing here was applied** — this gate
does not implement.

1. **Add the connection test the suite is missing** (fixes G/M5, the HIGH). One test:
   `pending=False`, non-stopping plan, a real `deg_cost_s`, assert the terminal gap moves by the
   liability and that `terminal_positions != positions`. Cheap, and it is what would have caught
   the disconnection mutant. **In the same pass, un-zero the fixture**: give at least one test a
   non-zero `cliff_loss_s` and a non-zero `future_neutralisation_prob` / `neutralisation_saving_s`,
   because today three of the formula's five terms are structurally unreachable and all three
   mutants that touch them ship green (M1, M2, M3).
2. **Correct `_terminal_gaps`' own docstring** (`position_projection.py:753-754`), plus
   `ProjectionResult.terminal_positions` (`:277-282`) and `ProjectionConfig.mandatory_stop_pending`
   (`:234-236`). Zero risk, and this is the comment a maintainer will trust.
3. **Fix the two tests that now pass for the wrong reason**
   (`tests/mc/test_position_projection.py:271-275, :509-517`) — either set `deg_cost_s=None`
   explicitly with a comment saying the assertion is conditional on it, or re-state what they
   guarantee. Their current names assert something production no longer does.
4. **Publish the amplification measured here as the E4 the design gate demanded**, in
   `MEASURE_763_deferral_effect.md`, alongside the reach numbers (67% by obligation, 38% once a
   tyre reading is required, Barcelona carrying 64% of the effect). This is disclosure, not a
   code change, and it is what makes the term honestly reported rather than honestly built.
5. **Decide the q_f discount on the run-it-out branch** — either drop it (the branch has no stop
   for a Safety Car to cover; median 3.24 s, zeroing the liability outright on 12.5% of elective
   laps) or replace the docstring's justification with the real one. Behaviour change, so it
   wants its own before/after, and it moves in the conservative direction.
6. **Thread the per-draw `cliff_laps` into the liability** — promoted after execution: it is the
   third HIGH, not a nicety. Today the terminal horizon assumes the cliff bites `window_laps` out
   while the in-window term one function above uses the model's own `cliff_laps`; the assumption
   is the SOLE source of the liability on 310 laps where the tyre reading is ≤ 0, contributes a
   median 15.2 s (max 45.6 s), and flips 60 argmaxes. Do it right after (1) so a test exists that
   would notice.
7. **Delete the dead `else` at `:795-799`** or, better, delete the early return at `:783-784` and
   let the `else` carry None — but only if `their_residual` suppression is preserved, which is the
   whole reason the early return exists. One rule, one home.
8. **Correct the record where it is cheap**: E5's "two that used to land exactly no longer do"
   (three lost, one gained, one of them pushed out of the gradable set); E5's unstated definition
   of "first stop"; E1's "green-flag" label; the three docs-site sentences; the legacy scorer's
   "a stop is mandatory" comment.

## What I tried to break and could NOT

- **E3, the invariance.** This was the claim I most expected to break, because the author's own
  note said the split had been got wrong twice. It holds, and it holds harder than reported:
  re-derived from the two committed JSONs with my own join, under **both** plausible definitions
  of "the driver's first stop" (114/64 from the verdicts alone, 89/89 from the raw `PitInTime`),
  **zero** first-stop verdicts changed bucket and **zero** first-stop chosen laps moved. I also
  checked the key sets are identical (178 = 178, no mismatches), so the before/after really is
  like-for-like.
- **The "one production change between them" attribution.** Two other commits sit in the range;
  I traced both and neither can reach the no-llm action (`4520431` is docstrings + LLM prompt
  strings; `285f33b`'s sentinel feeds `pace_out`, which `_run_projection_mc` does not take).
- **Double counting.** I tried to construct an overlap between `driver_time_delta`'s in-window
  wear and the terminal liability, including at `laps_remaining < window_laps`, under a
  neutralised config, and for a stopping plan. The partition is clean; the only band nobody
  charges is under-charged, never double-charged.
- **The liability exceeding a plausible real cost.** I looked for an input that makes the term
  charge more than the stop it defers. The `min` against `_stop_residual_s` caps it structurally;
  the 28.57 s maximum observed is a long-pit-lane circuit's own pit loss. Even at
  `laps_remaining=62`, where the run-it-out branch prices a physically impossible 57-lap stint,
  the cap returns the right answer.
- **The `k = 0` collapse.** I attacked it with a negative `deg_cost_s` (the floor is −2.33, and
  19.4% of real readings ARE negative), where the comment's premise "every further lap adds wear"
  is false. The zero clamps make the code land on a defensible answer anyway. I could not find a
  regime inside the model where waiting longer is genuinely cheaper; the only candidate — a
  time-varying neutralisation hazard — is flattened by `q_f` for the sibling term too. (What I
  found INSTEAD, while testing this, is that a negative `deg` does not zero the liability at all
  because the cliff term is charged independently — that is the third HIGH, and it came from an
  attack that failed on its own terms.)
- **A sentinel collision.** `deg_cost_s=None` is preserved as unknown (never 0.0);
  `mandatory_stop_pending` uses the three-state `is True` / `is False` / else rather than
  truthiness; `_lap_count_or_zero` clamps a negative to 0, which the `max(0, ...)` at `:707`
  then makes inert. No searchable-value collision found in this diff.
- **The 86.5% rejoin ground truth moving.** Executed:
  `src/strategy/eval/projection.py:61-67`'s `_GROUND_TRUTH_CONFIG` carries `deg_cost_s=None`,
  `mandatory_stop_pending=None` and `laps_remaining=0`, so the liability returns zeros through
  three independent guards, and line 260 reads `result.positions` anyway. The design gate's
  blast-radius claim survives.
- **The empty-set assertion in the invariance test.** It is an assertion about the empty set, but
  the positive control shows the monkeypatch genuinely intercepts, so it passes for the right
  reason.
- **The twin.** I looked for the repo's dominant defect and did not find it here: the legacy
  scorer decided 0 of 2,744 measured laps, has no terminal horizon to hang the term on, and
  charges a 2.8 s stop rather than a 22.8 s one. Only its justifying comment is now stale.
- **The scoping leaking at the LAP level.** E3 only shows that no first-stop VERDICT moved, which
  is compatible with the term firing on mandatory laps and being absorbed by the bucketing. I
  checked every one of the 2,744 laps individually: all 338 argmax changes are `pending=False`,
  and **zero** `pending=True` laps changed candidate. The scoping holds where it is hardest to
  see.
- **The commit message's factual claims.** Every number in `bcd2c9d`'s body that I could execute
  — the 2,744-lap projection routing, the 79→65 declines, the 178-stop sample identity, the
  694-elective/13-lap/15.0% E1 figures, "eight tests" — reproduces. The three retired artefacts
  it names (92-lap break-even, 2.8 s pit term, 73.6%) are correctly retired.

---

**Report complete.** Every figure above was executed against either the committed artefacts or a
byte-identical pristine copy of the branch's code. Final state verified: `git status` shows
`src/agents/position_projection.py` unmodified, `git diff` is empty, and `diff` against the `cp`
backup reports the files identical. `git checkout --` was never used. No repository file other
than this report was changed.

**Scripts, for anyone re-running this** (all under the session scratchpad, none committed):
`capture_projection_inputs.py` (the 2,744-lap capture) · `analyze_captures.py` (liability
distribution, saturation, branch shares, cliff variant) · `sensitivity_small.py` (the E4
amplification) · `rederive_e3_e5.py` + `first_stop_defs.py` (A/E re-derivation) ·
`e1_green_filter_check.py` (E1's label) · `mutation_battery.py` + `mutation_battery2.py` (§G) ·
`test_probe_positive_control.py` (the empty-set control).

---
---

# RE-RUN AFTER THE FIX — branch `feat/deferral-tyre-liability` @ `7dd4751`

**Date:** 2026-07-31 (same session) · **Prompted by:** the three HIGH findings above being fixed
and pushed. Everything above this line describes `4652c48` and is left untouched so the before and
after stay readable side by side.

**What changed in the code** (`git diff 4652c48..7dd4751 -- src/agents/position_projection.py`,
+23/−11): `_deferral_tyre_liability_s` now takes `cliff_laps` as its second argument and computes
`laps_past_cliff = max(0, remaining − cliff_laps)` from the tyre model's own per-draw onset,
instead of assuming everything past `window_laps` was past the cliff. `_terminal_gaps` threads it
from `project_positions`. Two tests added. This is exactly the D/HIGH finding, fixed at the point
the finding named.

## Method — what is comparable and what I had to change

The captured `_run_projection_mc` kwargs are **inputs**: they come from the replay and the agents,
not from the projection scorer, so the same 2,744 laps are valid against the fixed code and the
comparison is lap-for-lap. Two harness changes were forced, and both are traps:

1. **The zero-stub must take the new 3-arg signature.** A 2-arg stub would raise (loud, fine) —
   but a stub with a `*args` catch-all would silently never bind and the "term off" column would
   become a copy of the "term on" column. The re-run **asserts the real signature** before
   sweeping (`inspect.signature(REAL) == ['pit_loss_s', 'cliff_laps', 'config']`) and runs a
   **positive control**: term-on vs term-off must actually differ somewhere. It differs on
   **87 of 200 sampled laps**, so the control column is real.
2. **`mandatory_stop_pending=True` is NOT a "term off" control** — it swaps the liability for
   `_stop_residual_s`, which is a different non-zero charge. The only valid control is stubbing
   the function to zeros, which is what both runs do.

## ⭐ E4 RE-RUN — the amplification, same harness, same 1,055 laps

Identical procedure to the pre-fix run: the 1,055 `pending=False` laps carrying a tyre reading,
`deg_cost_s` perturbed by ±0.1 / ±0.2 / ±0.5 s/lap in either direction, argmax via the production
`best_mc_candidate`, each lap scored with the term wired and with it stubbed to zero.

| deg error | flips WITH the term | flips WITHOUT | **amplification** | (was) |
|---|---|---|---|---|
| **±0.1 s/lap** | **65 (6.2%)** | 11 (1.0%) | **5.91×** | *(was 8.27×, 8.6%)* |
| ±0.2 s/lap | 125 (11.8%) | 29 (2.7%) | **4.31×** | *(was 6.07×, 16.7%)* |
| ±0.5 s/lap | 279 (26.4%) | 49 (4.6%) | **5.69×** | *(was 6.86×, 31.8%)* |

**The "WITHOUT it" column is byte-identical to the pre-fix run (11 / 29 / 49).** It has to be —
stubbing the liability to zero makes the cliff fix unreachable — and that identity is the
strongest available check that the two runs are the same measurement on the same laps.

### The answer to the ship question, plainly

**It is no longer 6-8×. It is 4.3-5.9×.** The fix is real and material: at the plausible-error
scale that matters most (±0.1 s/lap), flips fell from **8.6% → 6.2%** of elective decisions, a
28% reduction, and the amplification fell from above the design gate's predicted band to inside
it. The gate predicted "~4-6×"; the fixed code measures 4.31× / 5.69× / 5.91×.

**But being inside the predicted band is not the same as passing E4, and I will not report it as
one.** The gate's criterion was conditional, not a threshold: *"if deg's CI cannot support the
amplification, B4 (or B5 as the interim) is the honest fallback."* Applying it needs `deg_cost_s`'s
error bound — and **#744a never published one.** So the criterion remains formally unevaluable for
the same reason it was before the fix, and what I can state is the two facts either side of it:

- **A tenth of a second per lap of tyre error still flips 6.2% of elective decisions** — one in
  sixteen — against 1.0% without the term.
- **The flips are two-sided and near-symmetric** (`STAY_OUT→UNDERCUT` +161 / `UNDERCUT→STAY_OUT`
  −111; `STAY_OUT→PIT_NOW` +89 / `PIT_NOW→STAY_OUT` −64), i.e. those laps still sit on the knife
  edge rather than in a robust regime. The asymmetry did shrink with the fix.

**My recommendation, as the gate rather than the implementer:** this is a judgement call that
belongs to whoever owns the ship decision, and the number that should drive it is 6.2% at ±0.1
s/lap, not the ratio. If the project can state a defensible bound on `deg_cost_s`'s per-lap error
and it is materially under 0.1 s/lap, the term is carryable and should ship with the amplification
table published beside it. If it cannot — and today it cannot, which is the honest position — then
shipping means accepting that one elective call in sixteen turns on an unbounded input, and the
gate's own fallback (document the boundary; never the window) is the consistent choice. **What is
not defensible either way is shipping without publishing this table**, which was E4's entire
purpose.

## Does the cliff fix change my other findings? — two die, one is untouched, one is new

Same 2,744 captured laps, same production functions, fixed module.

### The invented charge is GONE — the D/HIGH finding is fully resolved

| tyre reading (pending=False laps) | n | laps with liability > 0, **before** | **after** |
|---|---|---|---|
| `deg is None` | 784 | 0 | **0** |
| `deg == 0.0` | 231 | **189** (median 8.41 s) | **0** |
| `deg < 0` | 205 | **121** | **66** (median 0.00 s) |
| `deg > 0` | 619 | 592 | 555 (median 18.89 s) |

**Zero laps now charge a liability on a set the model prices at exactly fresh.** The 66 surviving
`deg < 0` laps are **not** a residue of the bug: with the per-draw onset threaded, a set that is
fast *today* but whose `cliff_laps` is small genuinely does cost time before the flag, and that is
now the tyre model's claim rather than the scorer's assumption. That is the right answer, arrived
at for the right reason.

Knock-on effects, all in the conservative direction: the liability is now identically zero on
**66.2%** of elective laps (was 51.0%), full saturation fell **28.8% → 20.6%**, and the positional
cost fell at p75 (**2.00 → 0.99**) and p95 (**5.80 → 4.37**). The worst case is unchanged at
**11.99 positions**, as it must be — the `min` still caps the liability at the stop being deferred.

**One number went UP and it should have:** the cliff contribution's maximum rose **45.6 s →
49.49 s** while its median collapsed **15.2 s → 0.49 s**. The fix is a redistribution, not a
reduction: laps whose model-predicted cliff is far away now pay nothing, and laps already past
their cliff pay for every remaining lap instead of being capped at `remaining − window_laps`.

### Barcelona dominance — UNCHANGED, and the fix slightly sharpened it

| race | laps | pending=False | deg known | liab > 0 (was) | flips (was) |
|---|---|---|---|---|---|
| **Barcelona** | 737 | 522 | 456 | **336** (425) | **179** (216) |
| Monaco | 558 | 329 | 241 | 170 (213) | 64 (70) |
| Marina Bay | 308 | 168 | 141 | 78 (128) | 25 (30) |
| Lusail | 313 | 153 | 105 | 26 (82) | 6 (10) |
| Monza | 219 | 86 | 59 | 11 (49) | 4 (11) |
| **Silverstone** | 609 | **581** | **53** | **0** (5) | **0** (1) |

Total flips 338 → **278**, still **every one `pending=False`**. Barcelona's share is **179/278 =
64.4%**, statistically identical to the 64% before. **The finding stands unchanged** — and
Silverstone has gone from 1 flip to **zero**, so the archetype that is 95% elective now receives
the term on precisely no laps. The cliff fix does not touch the cause, which is that
`_fresh_reference` has no reading for 528 of Silverstone's 581 elective laps.

### `deg is None` inertness — UNCHANGED at 42.6%

784 of 1,839 elective laps, identical before and after. The `deg_cost_s is None` guard was not
touched by the fix and should not have been. **The reach statement in the report above still
holds verbatim**: 67% of laps by obligation, 38% once a tyre reading is required.

## What the fix did NOT touch — stated so nothing is assumed closed

`git diff 4652c48..7dd4751` touches four files: `position_projection.py`, two test files, and
this report. **Verified untouched**: `tests/mc/test_position_projection.py`,
`src/agents/strategy_orchestrator.py`, `docs/pages/multi-agent.md`, and
`documents/audits/MEASURE_763_deferral_effect.md`. Read from a pristine copy of the fixed module
(the working tree held a mutant at the time — the same trap as before, avoided the same way):

- **STILL OPEN [MEDIUM] — the stale docstring**, `position_projection.py:765-766`, verbatim:
  *"already stopped (no obligation) -> our residual is zero, staying out costs nothing on this
  term"*. This is the single most misleading line in the module and it sits 35 lines above the
  branch that contradicts it. Fix-list item 2, untouched.
- **STILL OPEN [MEDIUM] — the q_f discount on the run-it-out branch** (`:727-729`). Unchanged, and
  its docstring justification ("a neutralisation that turns up covers a deferred stop whichever
  branch wins") is still wrong for a branch containing no stop. Fix-list item 5.
- **STILL OPEN [MEDIUM] — the two tests passing for the wrong reason**
  (`test_position_projection.py:271-275, :509-517`, both asserting a discharged obligation makes
  staying out free). `test_position_projection.py` is untouched by the fix, and `_flat_config`
  still leaves `deg_cost_s` unset, so both still pass only because the term short-circuits.
  Fix-list item 3.
- **STILL OPEN [LOW]** — the dead `else` at `:807-811` (and its numpy-bool reachability), the
  docs-site drift, the legacy scorer's comment, and the E1/E5 record corrections. Fix-list
  items 7 and 8.
- **STILL OPEN [MEDIUM] — E2 was never run.** The sign test on 2023-24 elective stops still has no
  artifact. E4 now exists (this section); E2 does not.

## Why the synthetic sweep read 0.8% where real laps read 6.2% — measured, not asserted

The coordinator flagged their own re-measurement as invalid (400 synthetic parameter states
through `project_positions`, 0.8% flips) and asked me not to use it. That judgement was right, and
the reason is measurable rather than a matter of taste. **A flip needs the top two candidates to
be close enough that a perturbation can cross them**, so the flip rate is a property of how near
the population sits to its own decision boundaries — not of the parameter ranges swept.

Measured on the 1,055 real elective laps (top-2 `score` margin, in positions):

```
p10 0.001   p25 0.254   p50 1.000   p75 4.003   p90 7.277
margin <= 0.01 :  124 laps (11.8%)      <- effectively ties
margin <= 0.10 :  189 laps (17.9%)
margin <= 0.50 :  467 laps (44.3%)
```

**Almost one real elective lap in eight is within a hundredth of a position of flipping**, and the
distribution is heavily bimodal — a dense cluster at the boundary and a long tail far from it. A
uniform grid over parameter space reproduces the tail and badly under-samples the cluster, which
is exactly the shape of a 0.8% result against a 6.2% one. The 6.2% flip rate is also *internally
consistent* with this: it sits below the 11.8% of laps that are near-tied, as it must.

**The reusable rule: sensitivity of a decision layer can only be measured on the distribution the
layer actually faces.** Synthetic states measure the function; real laps measure the decision.

## Battery 3 — do the new tests actually catch the mutants they were written for?

A test added to close a mutation gap is worth exactly as much as its ability to kill that mutant,
so both new tests were verified by re-running the surviving mutants against them. Same discipline:
`cp` backup of the FIXED file first, restore from the backup after each, byte-identical assertion
at the end. Baseline is now **175 tests** (173 + 2 added).

| Mutant | Result | Killed by |
|---|---|---|
| **M5 — the wiring deleted** (`our_residual = _deferral_tyre_liability_s(...)` → zeros) | **CAUGHT** (1 failed, 174 passed) | `test_the_terminal_gap_of_a_deferring_car_actually_MOVES` |

**The disconnection HIGH is genuinely closed**, and closed by precisely the test written for it —
not by an incidental assertion elsewhere. That is the strongest form of the check.

