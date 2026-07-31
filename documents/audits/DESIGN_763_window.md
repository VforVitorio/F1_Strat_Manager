# DESIGN GATE #763 — WINDOW_LAPS: attack the "it is the window" framing before any code

**Date:** 2026-07-31 · **Gate type:** adversarial DESIGN gate, pre-implementation · **Branch:** `dev`
**Inputs:** issue #763 · `MEASURE_744b_decision_effect.md` · `MEASURE_752_metric_and_sample.md`
**Mandate:** the issue's conclusion — *"that is the window, not the scale"* and the implied fix
"widen `WINDOW_LAPS`" — is a HYPOTHESIS to break, not a brief. No repository file is modified
except this report. All evidence executed, `profile="no-llm"`, zero API calls.

---

## Questions on the table

- **A.** Is widening the window even coherent, given break-even ~92 laps vs a ~57-lap race?
- **B.** Is there a formulation that needs no window at all? Enumerate candidates with cost /
  breakage / validation for each.
- **C.** Is the diagnosis right? Verify "the traversal cancels, the pit term is ~2.8 s" against
  BOTH scorers and what `pit_context` actually supplies.
- **D.** Is "one term decides 73.6% of laps" actually a defect? Both sides, plus the measurement
  that distinguishes them.
- **E.** What would make any new width or formulation MEASURED rather than chosen — without
  tuning to the agreement metric?
- **F.** Blast radius: every golden, fixture, committed report and eval number that moves.

Findings are appended below as they are confirmed, each with executed evidence.

---

## C. The diagnosis is WRONG for the scorer that actually decided the measurement — VERIFIED, HIGH

**The issue's mechanism ("the traversal cancels by exclusion, the pit term is ~2.8 s") describes
`simulate_lap_window`, which is the LEGACY path. The eval that produced 43.4%/33.3% flows through
`_run_projection_mc`, where the traversal is charged in full and the cancellation happens through a
different mechanism with a different residual.** Evidence, all read from `dev` at `HEAD`:

1. **Routing.** `_run_mc_simulation` (`src/agents/strategy_orchestrator.py:1440`) routes to
   `_run_projection_mc` whenever `_has_usable_gaps(rivals)` — and the eval harness passes
   `rivals=(lap_state or {}).get("rivals")` from the real replay
   (`src/strategy/inference/no_llm.py:285`), whose `RaceStateManager.get_lap_state` emits a full
   rivals list with `interval_to_driver_s` (`src/simulation/race_state_manager.py:594-614`).
   `simulate_lap_window` is reached only when no rival carries a usable gap.

2. **The projection charges the traversal.** `src/agents/strategy_orchestrator.py:1155-1159`:

   > *"Total pit loss per draw: the lane traversal plus the physical stop. The legacy scoring
   > charged only the stop and argued in a comment that the traversal cancels; here it is charged
   > per car, so the cancellation happens exactly when the rival really pays it too, and not
   > otherwise."*

   `pit_loss_s = traversal_s + pit_s` — 19.7–27.5 s per circuit (`traversal_seconds`, supplied via
   `pit_context["traversal_s"]` by `race_context_from_lap_state`,
   `strategy_orchestrator.py:859-874`) plus the ~2.8 s stop. The code itself refutes the issue's
   sentence *"The pit-lane traversal is deliberately excluded"* for this scorer.

3. **How the cancellation actually happens there.** `driver_time_delta`
   (`src/agents/position_projection.py:610-613`) charges PIT_NOW
   `effective_loss = max(0, pit_loss_s − saving)` ≈ 22.8 s under green INSIDE the window, and
   `_terminal_gaps` (`position_projection.py:723-742`) charges STAY_OUT
   `_stop_residual_s(pit_loss_s) = max(0, pit_loss_s − q_f·saving)` at the TERMINAL horizon — but
   **only when `config.mandatory_stop_pending is True`**. When both fire, the net pit-channel
   difference between PIT_NOW and STAY_OUT is `q_f · neutralisation_saving_s` (the option value of
   waiting for a Safety Car), NOT the ~2.8 s physical stop. When the obligation is unknown (None),
   STAY_OUT pays nothing terminal and PIT_NOW carries the full ~22.8 s alone.

4. **What survives and what does not.** The DEEP conclusion of MEASURE_744b — "the wear charge and
   the net cost of stopping are the same order of magnitude, so the wear term tips the argmax" —
   can still be true in the projection path, because `q_f · saving` is also a few seconds. But the
   number (~2.8 s), the mechanism (exclusion), and the constancy (the projection's net pit cost
   varies with laps remaining through `q_f`, shrinking to ~0 at the flag) are all wrong for the
   scorer that made the decisions being scored. **Any fix designed against `simulate_lap_window`'s
   arithmetic is designed against the wrong scorer.**

Numeric verification and the path census follow below.

---

## C2. Executed: which scorer decided the measurement, and with what pit term

**P3 — path census over the eval's own replay spans** (`scratchpad/probe_763_census.py`,
replaying every (race, driver, span) `measure_decision_agreement` replays, data-only, no models):

```
TOTAL eval laps: 2744
  projection path (usable gaps): 2744 (100.0%)
  legacy path (no usable gaps):  0 (0.0%)
```

**`simulate_lap_window` decided ZERO laps of the 43.4%/33.3% measurement.** Every number in
MEASURE_744b's diagnosis section — the ~2.8 s pit term, the 1.87 positions, the "73.6% of laps"
share (computed as `deg×5/POS_GAP_S`, a conversion the code itself marks "LEGACY PATH ONLY",
`strategy_orchestrator.py:643-644`) — is arithmetic of a scorer that never ran on this sample. The
legacy path still exists in production (default lap_state builders, `/recommend` with no rivals),
so its arithmetic is not dead code; it is just not what was measured.

**P2b — the projection's actual pit term, executed** (`scratchpad/probe_763.py`, production
functions, seed 42, 500 draws, pooled onset rate 0.0179/lap, 30 laps remaining → q_f = 0.416):

```
in-window loss:  STAY_OUT 3.05 s   PIT_NOW 22.93 s
STAY_OUT terminal residual: 19.60 s      (only when mandatory_stop_pending is True)
net pit channel = q_f · saving = 3.32 s  (verified to the second decimal)
wear channel (deg×5): 3.05 s
```

With the obligation pending, the net cost of stopping is the **option value of waiting**
(q_f · 8 ≈ 3.3 s), which happens to be the same order as the ~2.8 s the issue names — the issue's
"same order of magnitude" conclusion survives, but by coincidence of magnitudes, with the wrong
mechanism, and only on the pending=True half of the sample. Which brings in:

## C3. The population split the diagnosis missed — THE CENTRAL FINDING, HIGH

`mandatory_stop_pending` (`src/simulation/stint_history.py:75-104`: False on positive evidence of
a discharged Art. 30.5(m) obligation) splits the eval sample in two, and the two halves live under
**different pit arithmetic**:

- **pending=True (first stop still owed):** the traversal cancels against the terminal residual;
  net pit cost ≈ q_f·saving ≈ 3.3 s. The issue's premise holds HERE.
- **pending=False (obligation discharged, the stop is elective):** `_terminal_gaps` charges
  STAY_OUT nothing (`position_projection.py:728`: residual requires `pending is True`), so PIT_NOW
  carries the **full ~22.8 s alone**. The issue's premise — *"the stop is mandatory under the
  two-compound rule ... so it cancels"* — is FALSE here, and correctly so: an elective stop's
  traversal has no counterfactual payer.

**P4 — per-stop census** (`scratchpad/probe_763_stops.py`): of the 178 real green-flag stops the
eval grades, **105 (59.0%) are first stops and 73 (41.0%) are elective extras** (Silverstone 27/31
extra; Monza 19/20 first).

**P5 — the committed verdicts crossed with that flag** (committed
`documents/eval_reports/decision_modes.json`, harness `d97a54e`, with the pre-#760 version
recovered from git `fd105de` for the before column):

| | FIRST stops (n=105) | EXTRA stops (n=73) |
|---|---|---|
| `no_call` BEFORE the term | 29 (27.6%) | 53 (**72.6%**) |
| `no_call` AFTER the term | 28 (26.7%) | 51 (**69.9%**) |
| scored AFTER | 47 (exact 31.9%, mean offset −2.11) | 7 (n too small to read) |

**The decline problem is not uniform reluctance — it is 70% concentrated on elective stops**,
where the wear term's ~3 s faces a ~22.8 s wall and moved almost nothing (53 → 51). Of the 79
declines in the committed report, 51 (65%) are elective stops. Meanwhile on FIRST stops — the
balanced half — the term churned the scoring set (5 scored→no_boundary, 3+5 in the other
direction) and pushed the 10 moved offsets **−1.0 lap earlier each** (mean −1.75 → −1.98). That
earlier-first-call shift is where the five exact agreements went.

**So MEASURE_744b's regression decomposes into: (a) on the netted half, +3 s of wear against a
3.3 s netting moves the transition ~1 lap earlier; (b) on the full-cost half, the term is
impotent.** Neither half is described by "the window prices a whole stop against five laps of its
benefit": on half (a) no whole stop is priced at all, and on half (b) the whole stop is priced
BECAUSE no rule cancels it.

---

## A. Widening WINDOW_LAPS is refuted — with today's arithmetic, not the plan's

**The issue's "break-even ~92 laps" belongs to a formulation that no longer exists.** Executed
(P1, `scratchpad/probe_763.py`):

```
break-even W, shipped legacy comparison (pit 2.8 vs deg 0.61):        4.59 laps
break-even W, deg=None fallback (pit 2.8 vs FRESH_GAIN 0.25):        11.2 laps
break-even W if the FULL loss were charged vs FRESH_GAIN: 22.8/0.25 = 91.2 laps  <- the "~92"
```

The ~92 figure is `full pit loss / FRESH_GAIN` — a comparison in which BOTH numbers have since
been replaced (#718/#744b measured the wear; the legacy scorer charges the stop only; the
projection nets it or charges it per the obligation). Quoting it as the premise for widening today
is quoting the arithmetic of a retired design.

**What widening actually does, in both scorers, is monotone:** the wear charge is the only
W-dependent term, so d(PIT−STAY)/dW = +deg/POS_GAP_S in the legacy path (measured: −0.647 pos at
W=3 → +6.267 at W=20) and the same direction in the projection (P2 sweep: −1.007 pos at W=3 →
+1.000 at W=15, saturating on the synthetic 4-car field). Widening cannot re-balance anything; it
can only hand MORE of the argmax to the wear term — the exact defect the issue says it wants to
fix. And it acts on the two populations in the wrong order:

- On FIRST stops (already balanced, declines 26.7%), the netting is ~3.3 s, so each extra lap of
  window adds ~0.6 s to the pit side of the argmax: **first calls move earlier immediately**,
  worsening the exact-agreement loss that motivated #763.
- On EXTRA stops (the actual problem, declines 69.9%), the wall is ~22.8 s: break-even needs
  **W ≈ 22.8/0.61 ≈ 37 green laps** (or ≈16 laps if past the cliff) — wider than nearly every
  stint and most remaining-race spans. By the time W dents the decline population, the window IS
  the remaining race, i.e. a different model, which is what the issue itself concedes in its
  framing question.

Two hidden couplings make widening even more expensive than it looks (see F): the committed
measured table `sc_window.racing_laps_in_window` is conditioned on W=5
(`scripts/measure_mc_tables.py:419`, `measure_sc_window(window=WINDOW_LAPS)`), and the eval's
`DECISION_WINDOW_LAPS = 5` is deliberately matched to the MC window (`decision_modes.py:92-95`) —
widening re-opens the width-dependence #752 spent a sprint retiring.

**Verdict: A is answered. Widening is not merely "not the best fix"; it is directionally wrong on
the balanced population and numerically unreachable on the broken one.**

---

## B. Candidate formulations, enumerated

### B1. Widen `WINDOW_LAPS` (the issue's implied fix) — REFUSE
Refuted in A. Cost: trivial to type, every golden re-frozen, `mc_measured_v1.json` regenerated,
eval width coupling re-opened. Validation: none exists that is not the forbidden agreement loop.

### B2. Amortise the stop over laps remaining (issue's first named alternative) — REFUSE
Divide the stop cost by `laps_remaining` and charge per-lap rates. It does not fit the projection's
currency: positions come from gap crossings against actual cars (`project_positions`), and a
crossing either happens or it does not — an amortised fraction of a pit loss is not a position.
It would also be false physics: the rejoin deficit is real and immediate; teams do not experience
1/30th of a pit stop per lap. Cost: a rewrite of both scorers' semantics; breaks every golden AND
the rejoin ground truth's meaning. Validation: nothing measurable — the amortisation schedule
would itself be a chosen constant.

### B3. Price the deferral honestly at the horizon that already exists — BUILD THIS
"Score the difference between stop now and stop at the best later lap" (the issue's second named
alternative), realised not as a new engine but as **the missing tyre side of the terminal
liability the projection already has**. `_terminal_gaps` already carries every KNOWN outstanding
stop to a common horizon (`position_projection.py:677-742`); what it does not carry is what the
tyres cost between the window's edge and that horizon. A non-stopping plan's terminal gap should
bear, in addition to the existing stop residual:

```
tyre_liability = min(  deg·k* + residual(k*)        # stop at the best later lap k*
                       deg·R + cliff·max(0, R−c)  ) # or run this set to the flag (R laps left)
```

both sides q_f-discounted exactly as `_stop_residual_s` already does. For pending=False plans this
is the ONLY future-cost term (today they carry none), and it is what makes an elective stop
comparable at all: 22.8 s now against a measured 15-30 s of wear-to-the-flag, instead of against
five laps.

- **Cost:** one new pure function in `position_projection.py` + one term in `_terminal_gaps` +
  config already carries every input (`deg_cost_s`, `cliff_loss_s`, `laps_remaining`, `q_f`).
  Small by this epic's standards.
- **Breaks:** `test_mc_is_a_real_decision.py` (its fixtures pin `mandatory_stop_pending: False`,
  lines 412/484, and the sweep shares 42.5/31.2/26.2 quoted in the OVERCUT comment at
  `strategy_orchestrator.py:756-760` — the twin comment must move with the test);
  `test_projection_golden.py` only if the implementation touches the pending=None case (its
  fixture leaves pending unset → None); decision eval numbers regenerate. Does NOT touch
  `simulate_lap_window`, `WINDOW_LAPS`, the measured tables, or the rejoin ground truth (§F).
- **Validated:** by the horizon measurement and sign test in E — against 2023-24 physics, never by
  iterating on the 2025 agreement metric.
- **Scope note, stated rather than hidden:** the same unpriced wear-to-eventual-stop exists for
  pending=True, but that population measures balanced today (26.7% declines) and adding the term
  there moves first calls EARLIER — the direction #744b just paid for. Scoping the new term to
  plans with no residual (pending=False; pending=None stays inert under the "a claim needs a
  fact" rule) is a measured-behaviour scoping with a known physical inconsistency. Extending it to
  pending=True is a separate decision that must wait for the E1 horizon measurement, not
  symmetry aesthetics.

### B4. Full remaining-race optimisation (Heilmeier-style DP) — DEFER
The literature's answer (the `WINDOW_LAPS` comment itself cites van Kampen 2024 against short
horizons) and the only formulation that also fixes pending=True honestly. It needs per-compound
degradation curves over the whole stint space and a rebuilt MC. That is a future initiative, not a
fix inside this epic; B3 is its cheapest honest approximation.

### B5. Change nothing; document the boundary — HONEST BUT INSUFFICIENT
The layer is a mandatory-stop timer that declines elective stops. Defensible as a scope statement,
zero blast radius — but 41% of real strategy calls in the 2025 sample are elective, the epic's
own scorecard names the decline rate as the number to attack, and the boundary was discovered by
this gate rather than chosen by anyone. Rejected as an end state; ACCEPTED as the interim truth
the docs should state until B3 lands.

---

## D. Is "one term decides 73.6% of laps" a defect? Both sides, then the verdict

**The number first:** 73.6% is `share of laps where deg×5 > POS_GAP_S = 1.5 s` — a legacy-path
conversion applied to decisions the projection made. In the deciding scorer a position is a gap
crossing, and the measured median consecutive-car gap while racing is 2.23 s
(`strategy_orchestrator.py:645-646`). At 2.23 s the threshold is deg > 0.446 s/lap; the median
charge (2.03 legacy-positions ≡ 3.05 s) still crosses, so **a majority survives the correction,
but the specific figure is an artefact of the wrong conversion** and should stop being quoted.

**The case that dominance is correct:** stop timing in modern F1 IS wear-dominated — with pit loss
fixed per circuit, the lap you box is chosen by tyre state; undercut, clean air and hazard are
tie-breakers around a wear-determined window on real pit walls too. A term that dominates the
argmax near the stop is what a correct model looks like there, and the other terms being quiet is
information.

**The case that it is a defect:** if wear genuinely dominated the DECISION, it would dominate
where the teams' wear-driven decisions live — the elective stops, which teams take precisely
because wear pays for them. Measured (P5): the term moved that population by 2 stops out of 73 and
its declines stayed at 70%. It dominated instead the netted half, where its ~3 s stands against a
~3.3 s option value. **A term that is decisive only where the opposing cost has been netted away,
and impotent where the opposing cost is real, is not "wear matters" — it is the horizon asymmetry
deciding, wearing the wear term's name.**

**The distinguishing measurement is exactly the pending split, and it has now been executed.**
"Wear correctly dominates" predicts decisiveness on the elective population; "window artefact"
predicts decisiveness only on the netted population. P5 measured the second. Verdict: defect —
but the defect is the missing deferral horizon (B3), not the term's scale, and NOT the window
width.

---

## E. What makes the new formulation MEASURED rather than chosen

⛔ Tuning any constant until the 43.4% recovers is the forbidden loop; everything below is
measurable without ever consulting the 2025 agreement number during design.

- **E1 — the repayment-horizon measurement (the constant's basis).** Over 2023-24 (training
  seasons — 2025 stays held out), for every real elective stop: laps until the realised post-stop
  pace advantage (actual lap-time deltas, fresh vs the old set's trend) repays the realised pit
  loss. The distribution of realised repayment horizons is a fact about the sport. If its median
  sits at 15-25 laps, that is the measured proof that no fixed short window can price an elective
  stop and that the liability must integrate to `laps_remaining`. This number justifies B3's
  horizon; it is not fitted to any eval.
- **E2 — the sign test (calibration against physics, not agreement).** With B3 implemented, on
  each 2023-24 elective stop: does the terminal tyre liability of staying out exceed the charged
  pit loss ON the lap the team stopped, and not exceed it 10 laps earlier? The real stop lap is
  used only as an event marker for a SIGN FLIP, never as an argmax target — pre-registered pass
  rate, stated before running.
- **E3 — the invariance criterion.** FIRST-population behaviour (decline 26.7%, offsets) must not
  move beyond a pre-stated tolerance. A deferral term for elective stops has no business moving
  mandatory-stop timing; if it does, the scoping leaked.
- **E4 — sensitivity disclosure.** Publish the conclusion's sensitivity to E1's CI, the way
  FABLE G1 published mean_signed vs width. A formulation whose recommendation flips inside its
  input's confidence interval is not measured yet.
- **E5 — one 2025 re-run, reported, never iterated.** After E1-E4 freeze the design, run
  `f1-eval decision-modes` ONCE against 43.4%/46.1%, publish whatever it says, and if it
  disappoints the design goes back to E1 — not to a knob. Anything else is `mean_signed_error`'s
  funeral being reversed.
- **Confession check (the mandate asked):** E2 skirts closest to the forbidden pattern, since it
  consults real stops. The line it does not cross: it reads only the SIGN at two fixed distances
  on held-IN seasons, with the pass rate pre-registered — it cannot be climbed lap-by-lap the way
  an agreement percentage can. If an implementer ever converts E2 into "maximise the flip
  alignment", that is the forbidden loop and this gate names it in advance.

---

## F. Blast radius, exhaustively, per candidate

**The rejoin ground truth (86.5% within one, 1810 stops) does not move under ANY candidate here,
and the reason is structural, verified at file:line:** `src/strategy/eval/projection.py:61-67`
builds its own frozen `_GROUND_TRUTH_CONFIG` (`window_laps=2, racing_laps=2.0`, tyre terms zeroed,
saving zeroed) — decoupled from `WINDOW_LAPS` — and line 260 reads `result.positions`, the
window-end rejoin horizon, never `terminal_positions`. B3 edits only the terminal side;
`ProjectionResult.positions`' docstring (`position_projection.py:272-275`) pins that meaning.
Anyone whose implementation makes 86.5% move has changed the wrong horizon.

Under **B3** (the recommended build):

| Artefact | Moves? | Why |
|---|---|---|
| `tests/mc/test_mc_is_a_real_decision.py` | **YES** | fixtures pin `mandatory_stop_pending: False` (:412, :484); sweep shares re-measured |
| `strategy_orchestrator.py:756-760` OVERCUT comment | **YES** | quotes the 42.5/31.2/26.2 sweep — the twin of that test; move them together |
| `tests/mc/test_position_projection.py`, `test_position_projection_stop_horizons.py`, `test_tyre_wear_term.py` | **YES** | terminal-side unit coverage; new tests added here |
| `tests/mc/test_projection_golden.py` | **CONDITIONAL** | fixture leaves pending unset (None); moves only if the implementation touches pending=None — it must not (a claim needs a fact) |
| `documents/eval_reports/decision_modes.{md,json}` | **YES** | regenerate once (E5); headline vs 33.3%/44.4% and the buckets move |
| `tests/mc/test_strategy_goldens.py` (legacy scorer) | **NO** | `simulate_lap_window` untouched |
| `data/mc_measured_v1.json` | **NO** | no width change; `sc_window` stays valid |
| `documents/eval_reports/projection.{md,json}` (86.5%) | **NO** | see above |
| `f1-eval` hygiene/calibration/stint_lengths reports | **NO** | different tiers |
| CLI reasoning string `window={WINDOW_LAPS}` (`strategy_orchestrator.py:1732`) | **NO** | width unchanged |
| Epic scorecard / memory numbers (43.4, 46.1, "73.6%") | **YES (append)** | historical docs stay; the scorecard's current row updates; 73.6% retired as a conversion artefact |
| Issue #763 text | **YES** | its mechanism section is refuted here; a false claim in an issue is scope — correct it before implementation |

If anyone widens `WINDOW_LAPS` instead (B1), ADD: every strategy golden,
`data/mc_measured_v1.json` regeneration (`measure_sc_window` is W-conditioned,
`scripts/measure_mc_tables.py:419`), the eval's `DECISION_WINDOW_LAPS` coupling
(`decision_modes.py:92-95`) and with it the width-dependence of every reported offset, the CLI
reasoning string on every surface, and every doc/thesis mention of the five-lap window.

---

## RECOMMENDATION

**Build:** B3 — the terminal tyre liability. One pure function in `position_projection.py`
(deferral cost: `min(stop at best later lap, run to the flag)`, q_f-discounted), one term in
`_terminal_gaps`, scoped to non-stopping plans whose stop residual is absent (pending=False;
pending=None stays inert). No change to `WINDOW_LAPS`, no change to `simulate_lap_window`, no
change to the wear term's scale.

**Measured basis:** E1 (repayment-horizon distribution on 2023-24) justifies the horizon before a
line is written; E2 (sign test) + E3 (FIRST-population invariance) + E4 (sensitivity) gate the
implementation; E5 is a single 2025 re-run, reported not iterated.

**Refuse:** widening `WINDOW_LAPS` (directionally wrong on the balanced population, needs W≈37 to
reach the broken one); amortising the stop per lap (false physics, wrong currency); ANY iteration
loop on the 2025 agreement metric; extending the liability to pending=True without E1 in hand;
quoting "73.6%", "~2.8 s cancels in both scorers", or "break-even ~92 laps" ever again — all
three are measurements of retired or wrong arithmetic.

**Correct the record:** #763's mechanism section and MEASURE_744b's diagnosis section describe
the legacy scorer; the measurement they explain ran 100% on the projection scorer. The issue
needs a correcting comment before any implementation PR cites it.

---

## What I tried to break and could NOT

- **The like-for-like claim of MEASURE_744b.** Verified: the before report (`fd105de`, harness
  `99a663d`) and the committed one (`d97a54e`) grade the IDENTICAL 178-stop set (asserted
  programmatically in P5's join — zero key mismatches). The comparison is sound; only its
  diagnosis section is wrong.
- **The wear term's arithmetic.** `_tyre_term` / `_tyre_cost_s` are mutually exclusive with the
  FRESH_GAIN fallback exactly as documented; no double count; the OVERCUT window split charges
  `W//2 + W//2`. The term is correct. The issue is right that scaling it down would be the wrong
  fix.
- **The netting identity.** P2b reproduces `net pit channel = q_f × saving` to the printed
  precision (3.32 = 3.32) from production functions — `_terminal_gaps` + `_stop_residual_s` do
  cancel the traversal for pending=True exactly as designed.
- **The eval's routing.** I looked for eval laps that fall to the legacy scorer (which would
  partially rescue the issue's arithmetic): zero out of 2,744 across all six races. I also tried
  `lap_inputs`' skip conditions as a leak path; skipped laps are skipped for both metrics alike.
- **The 86.5% decoupling.** I attempted to construct a dependence of the rejoin ground truth on
  `WINDOW_LAPS` or on the terminal machinery: `_GROUND_TRUTH_CONFIG` is frozen at 2 laps with
  every strategy term zeroed and reads only `positions`. No path found.
- **The pending flag itself.** Spot-checked semantics against `stint_history.py`: False requires
  positive evidence (two dry compounds or a wet), True requires every stint visible; the per-race
  splits (Monza 19/20 first — a one-stop race; Silverstone 27/31 extra — a wet-ish multi-stop
  race) match 2025 racing reality.

**Single biggest risk of the recommendation:** B3 prices the deferral with the SAME `deg_cost_s`
whose level FABLE_G2/#744a showed to be reference-sensitive, now multiplied by `laps_remaining`
instead of 5 — an error in deg is amplified ~4-6×. E4's sensitivity disclosure exists precisely
for this; if deg's CI cannot support the amplification, B4 (or B5 as the interim) is the honest
fallback.

