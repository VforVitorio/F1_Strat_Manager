# FABLE A-1 — Adversarial gate on PR #730 ("give the projection branch teeth")

- **Target:** commit `bebd3e3` (merged to `dev` as `aa1d274`), issue #725, epic #724.
- **Files under audit:** `tests/mc/test_position_projection.py`, `tests/mc/test_mc_is_a_real_decision.py`, `tests/mc/test_projection_golden.py`.
- **Auditor:** Fable A-1 gate, 2026-07-29. Read-only except this report. Every production mutation applied is reverted and the revert verified with `git diff` (see §Revert log).
- **Environment:** local checkout WITH `data/` (models + raw + mc_measured_v1.json all present), so nothing skips.

## Checklist (appended as executed)

- [ ] Baseline: full `tests/mc/` suite green, count matches the claimed 141
- [ ] Mutation M1: `terminal_liability` filter `is False` -> `is not True` — exactly one red?
- [ ] Mutation M2: `MARGIN_WEIGHT` 0.1 -> 0.2 — new golden red, legacy golden 12/12 green?
- [ ] Claim 1: spread tests measure the mechanism their names/docstrings claim
- [ ] Claim 2: golden rival geometry is load-bearing (far-field = point masses)
- [ ] Claim 3: frozen point-mass assertion — useful or tautology
- [ ] Claim 4: `_draws` + pre-existing assertions unchanged
- [ ] Claim 5: shape assertions run (and can fail) without model weights
- [ ] Claim 6: seed-stability margins quantified
- [ ] Beyond: further mutations the author did not run
- [ ] Revert log: `git diff` clean

---

## Claim 1 — Are the spread tests measuring what their names say?

Method: re-scored `_LIABILITY_CROSSING_STATE` and `_PIT_LOSS_CROSSING_STATE` through the test's own
`_projection_scores`, three ways each — both draw vectors varying (as shipped), pit varying with
cliff frozen at its mode, cliff varying with pit frozen at its mode. Executed, seed 42, n=200.

| state / candidate | both varying | pit-only | cliff-only |
|---|---|---|---|
| LIABILITY / STAY_OUT | **1.000** | 1.000 | **0.102** |
| PIT-LOSS / PIT_NOW | **1.076** | 1.076 | 0.000 |

- **PIT_NOW: CONFIRMED.** Its spread is 100% the pit-loss channel (cliff-only = 0.000, exactly as
  the docstring claims — `stop_offset_laps=0` means the cliff term is structurally zero for a
  candidate that stops immediately).
- **STAY_OUT: OVERSTATED.** The liability channel dominates (1.000) but a SECOND channel is alive
  that neither the docstring nor the comment block mentions: the **margin term via the cliff draw**
  (B at 4.2 s projects to 1.59–3.00 s of buffer as `worn_laps` varies, x0.1 margin weight =
  0.102 of spread on its own). Consequence, verified by construction: **a mutation that collapses
  the liability channel specifically (e.g. exposure computed from the mean pit draw) leaves this
  test GREEN at spread~0.10.** The assertion `> 0.0` cannot distinguish "the deferred stop's
  exposure is uncertain" from "the margin wiggles a tenth of a position". The docstring's
  mechanism attribution is the dominant truth but the test does not enforce it.
- The comment's measured table (at C=22.2: 1.000/0.000; at C=23.2: 0.002/1.076) **reproduces**
  (I get 1.000/1.076 at the shipped states; the 0.102 cliff-only figure is the same margin channel
  that produces the 0.002 at C=23.2 — there it is nearly clipped away, here it is not).
- The "~1.2 s apart" band-gap claim: exposure support measured (20.72, 22.65), PIT_NOW crossing
  support ~ (21.95, 24.15). Centres ~1.4 s apart, chosen rival placements 1.0 s apart. "~1.2 s"
  is a fair characterisation. CONFIRMED.

## Claim 6 — Seed stability, quantified (executed)

Spreads across seeds 0-9 plus 42, n=200:
- STAY_OUT: min 1.000, max 1.040 — never below 1.0.
- PIT_NOW: min 1.071, max 1.085.

Crossing fractions under seed 42: PIT_NOW 0.355 (comfortably inside the (0.10, 0.90) percentile
band, ~7 sd from the nearest boundary at n=200). STAY_OUT liability crossing **0.110 — only
2 draws above the P10 boundary** (0.10 x 200 = 20 draws vs 22 observed). If a numpy stream change
pushed it under 0.10, the liability contribution would VANISH from P90-P10 and the spread would
drop from ~1.0 to ~0.10 — **still green**, because the undocumented margin channel catches it.
So the assertion is robust to a numpy bump, but partly for the unadvertised reason. VERDICT:
assertions robust (CONFIRMED); STAY_OUT robustness rests on the second channel (see Claim 1).

## Claim 2 — Golden rival geometry (executed)

- Reproduced the frozen golden exactly through `_run_mc_simulation` (values match the dict).
- Per-rival liability decomposition in the golden state: exposure support (20.62, 21.96).
  **BEHIND (4.6 s) is charged on 100.0% of draws; FAR (22.6 s) is charged on 0.0% — never.**
  FAR's actual function is the PIT_NOW/UNDERCUT pit-loss crossing (crossed on exactly 50.0% of
  draws — well placed). The file comment says the pit-cycle-behind car is there "so the terminal
  liability has something to charge": **REFUTED as written — the roles are swapped.** The liability
  is charged by BEHIND (constantly, hence contributing zero spread); FAR is what gives PIT_NOW its
  P10=0.0/P90=1.056 spread. The geometry IS load-bearing, but for the reason the comment does not give.
- Far-field control (BEHIND at 40 s, FAR at 60 s, same everything else): STAY_OUT and PIT_NOW
  become point masses (2.3, 2.3, 2.3). UNDERCUT does NOT (E 2.832, P10 2.3, P90 3.3) — the N16
  coin-flip bonus spreads regardless of geometry. So "would pin a set of point masses and hide any
  future collapse" is CONFIRMED for the projection channels (the only spread surviving far-field is
  the Bernoulli ucut bonus, which is not the projection), OVERSTATED as a blanket "set of point masses".

## Commit-message micro-claims (executed)

- "the same varying draws give 250 distinct payoffs against a rival inside the support and exactly
  1 against one 60 s away": reproduced **exactly** — 250 and 1 (n=500, seed 42). CONFIRMED.

## Claim 4 (part 1) — `_draws` and pre-existing assertions unchanged (diff evidence)

`git diff cf328cb..bebd3e3` over the two modified files removes exactly 4 lines, all in
`test_mc_is_a_real_decision.py::_projection_scores`:
the `def` line (signature gained two keyword-only-style optional params) and the three lines
replaced by default-preserving conditionals (`rival_stop_pending`, `cliff_s`, `pit_s`). Each
conditional reduces to the byte-identical old expression when the new params are None.
`_draws` in `test_position_projection.py` is untouched (that file's diff is pure addition, +73/-0).
No pre-existing assertion line was modified. CONFIRMED at the diff level; behavioural equivalence
is confirmed by the baseline run below (frozen goldens still pass to the digit).

The ground-truth eval (`src/strategy/eval/projection.py`) consumes `project_positions().positions`
only — `terminal_liability` feeds `.liabilities`, which never reaches a position — so mutation M1's
blast radius cannot include the ground-truth gate. Established before running M1.

### Claim 1 addendum — the "shared state / 0.06 margin" design rationale (executed)

A pure-numpy replica of the projection payoff (validated: it reproduces the shipped states'
spreads to the digit — 1.000/0.000 at C=22.2 and 0.002/1.076 at C=23.2, same as both the author's
comment table and my orchestrator-path run) swept a SHARED gap_behind x at 0.01 s steps:

- A window does exist where both candidates spread at once: x in [4.44, 5.23] (~0.8 s wide),
  with the smaller of the two spreads ~0.08 at its best edge and ->0 at the other. The author's
  "only at a 0.06 margin" is the right order of magnitude. CONFIRMED in substance.
- Sharper than the author stated: inside that shared window STAY_OUT's spread is entirely the
  MARGIN/cliff channel (the liability crossing fraction is ~3%, below the P10 boundary, so the
  liability contributes nothing to P90-P10 there). A shared state would not merely be knife-edge —
  it would pass STAY_OUT's assertion via a mechanism unrelated to the docstring's. The two-state
  design is therefore genuinely justified, more strongly than the comment argues.
