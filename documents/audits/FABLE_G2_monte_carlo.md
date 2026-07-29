# FABLE G2 — Adversarial gate over AUDIT A1 (the Monte Carlo findings of 2026-07-29)

**Role:** adversarial gate over the FINDINGS in `documents/audits/AUDIT_A1_stay_out_point_mass.md`,
not over the code directly. Success = catching the auditor being wrong. Nothing in the repository
is modified except this file. Zero LLM/API calls (`profile="no-llm"`).

**Date:** 2026-07-29

## Claims under attack

| # | Claim as submitted | Verdict |
|---|---|---|
| 1 | STAY_OUT is a point mass because `max(0, racing - cliff_laps)` zeroes the tyre draw (43/57 Lusail, 51/53 Monza). Sub-question: is the flatness a DEFECT or a defensible consequence of the model's own semantics? | **CONFIRMED as measured (numbers reproduced exactly); MISLABELED as a clip defect** — the clip is correct arithmetic, the defect is the cliff-only cost model (`deg_rate` unread) → G2 |
| 2 | The `stop_pending` asymmetry (`terminal_liability` exempts, `project_positions` charges) is the real defect. Sub-question: or are these two correct terms at two legitimate horizons? | **CONFIRMED — one defect, not two horizons; the missing term is the race-end credit.** A1's proposed fix #1 REFUTED; "sole root cause" OVERSTATED; one NEW same-family defect found → G3 |
| 3 | The tests cannot see any of this (constant `np.full` draws, no `P90 > P10` assertion, golden in a 3.6% corner). Sub-question: would `P90 > P10` even be TRUE universally for a correct implementation? | **CONFIRMED and UNDERSTATED** (unit suite also constant-fed; projection branch has NO golden). `P90 > P10` is NOT universal — executed counter-example; assert it in constructed scenarios → G4 |
| 4 | Alpha: "decorative" vs "flips 5/110" — which is right, and is dominance-invariance a theorem or an accident? | **"5/110" right (reproduced digit-for-digit); "decorative" wrong; invariance under dominance is a THEOREM (convexity), and the dial is structurally one-directional** → G5 |

## Owner's two questions (must be answered concretely)

- The arcade output looks reasonable against reality — how can that coexist with "the projection never prefers a stop"?
- This repo has had many bug hunts — why would these defects have survived?

## Findings

<!-- appended as they are confirmed, per AGENT_PROGRESS_PERSISTENCE -->

---

### G1 — Static verification of the mechanics (all three claims' code citations are accurate)

Checked every file:line the claims rest on, against the working tree at `dev` (e018f71):

- `src/agents/position_projection.py:548` — `worn_laps = np.maximum(0.0, racing - cliff_laps)` in the
  STAY_OUT (`else`) branch of `driver_time_delta`. `racing` is `config.racing_laps`, built at
  `strategy_orchestrator.py:1170` as `_bounded_by_race_end(float(WINDOW_LAPS), remaining)` (5.0 green)
  and `:1162` as `measured_racing_laps("sc")` (2.61) — the claim's numbers are the code's numbers.
- `position_projection.py:616-620` — `terminal_liability` filters `rival.stop_pending is False`;
  `project_positions` (`:662-668`) and `rival_time_deltas` (`:573-576`) apply no such filter and
  charge a rival's stop only on `is_pitting` (a this-lap timing fact). The asymmetry is real in code.
- `strategy_orchestrator.py:1211-1212` — PIT_NOW and UNDERCUT are **identical plans** (both
  `stops_in_window=True, stop_offset_laps=0`); the only differences are eligibility and the additive
  N16 term at `:1263-1264` (`outcomes + landed`). So "UNDERCUT's wins come from a bonus added
  OUTSIDE the projection" is not an interpretation, it is the code's structure.
- `strategy_orchestrator.py:1273` — `score = alpha*E + (1-alpha)*P10`. Convex combination confirmed.

Two facts the A1 report UNDER-stated (gate corrections, both strengthen claim 3):

1. **`tests/mc/test_position_projection.py` also feeds constants.** A1's F8 named only
   `test_mc_is_a_real_decision.py`. The projection unit suite itself uses `np.full` for BOTH
   `pit_loss` and `cliff` in its shared helper (`test_position_projection.py:58`) and everywhere else
   (`:96`, `:363-366`). Not one call into the primitive anywhere in `tests/` passes a non-degenerate
   draw vector.
2. **The frozen golden does not even exercise the projection branch.** `test_strategy_goldens.py:108`
   calls `_run_mc_simulation(pace, tire, situation, pit, alpha=0.5)` with no `rivals` kwarg, so it
   pins the LEGACY seconds path (`strategy_orchestrator.py:1396` routes on `_has_usable_gaps(rivals)`).
   A1's F9 says the golden sits in a 3.6% tyre-regime corner; the stronger fact is that it guards a
   branch the three real surfaces do not take whenever rivals carry gaps — the normal case. The
   golden's STAY_OUT row (`E -0.149, P10 -0.529, P90 0.0`) is the one place the suite pins a
   non-degenerate STAY_OUT distribution, and it pins it on the wrong branch.

Also verified: the ground-truth validation (`src/strategy/eval/projection.py`, the 86.5%/1810-stop
figure) is explicitly a REJOIN-horizon test — its own comments say "what is under test is the
geometry of the rejoin" and it projects each real stop with `window_laps=2` and grades
projected-vs-actual rejoin position. It validates `project_positions` exactly where that function is
right, and says nothing about the cross-candidate comparison. This matters for the "why did nobody
catch it" question (G6).

---

### G2 — Claim 1: CONFIRMED as measurement, MISLABELED as a clip defect. The clip is correct; the modelling gap is that pre-cliff degradation is unpriced.

**Method.** Re-ran both full races through the production entry point (`RaceReplayEngine` +
`run_lap(profile="no-llm")` + `_run_mc_simulation` on the captured outputs), 110 laps, zero
exceptions, zero API calls. Every A1 headline number reproduced independently and exactly:

| quantity | A1 claimed | G2 measured |
|---|---|---|
| STAY_OUT point mass, Lusail | 43/57 | **43/57** |
| STAY_OUT point mass, Monza | 51/53 | **51/53** |
| min `laps_to_cliff_p10` Lusail / Monza | 2.60 / 6.50 | **2.60 / 6.50** |
| median cliff_p10, laps with p10 < 5 | 20.0 / 4 (Lusail), 0 (Monza) | **20.0 / 4 / 0** |

**Is `racing` really the window length?** Yes: `racing_laps` is 5.0 green
(`strategy_orchestrator.py:1170`, `_bounded_by_race_end(WINDOW_LAPS, remaining)`), the measured 2.61
under SC (`:1162`), clamped smaller near the flag. So the regimes where the clip could bite are
strictly rarer late-race and under neutralisation, never more common.

**Does the clip ever bite, and is it then correct?** Executed (exhibit E1, synthetic, 500 triangular
draws): with cliff 12/13/14 STAY_OUT collapses to 1 unique payoff; with cliff 1.5/3/4.5 straddling
the window it produces **382 unique payoffs, P10 -2.70 / P90 -1.89** — a genuine spread, correctly
ordered. On the 4 real Lusail laps with `cliff_p10 < 5` the production run shows the same. **The clip
implements its own semantics correctly**: "laps in the window spent past the cliff" is genuinely zero
when the predicted cliff is 12+ laps away.

**So is the flat STAY_OUT wrong?** Partly defensible, and the defensible part changes what the fix
is. If the tyre will not cliff inside the window, holding position for five laps genuinely has low
variance — the owner's instinct is right THERE. What is NOT defensible is the conjunction:

1. The only tyre cost the layer models is a **step function at the cliff** — and N26's cliff
   predictions sit inside the 5-lap window on **4 of 110 real laps**. Pre-cliff degradation costs
   nothing: `TireOutput.deg_rate` exists (the goldens even set it, 0.05 s/lap) and
   `_run_mc_simulation` never reads it — only the three `laps_to_cliff_*` quantiles are consumed
   (`strategy_orchestrator.py:1346-1350`). A tyre 0.4 s/lap off the pace but 10 laps from the cliff
   is scored identically to a fresh one.
2. A structural side effect A1 did not spell out: a point-mass candidate has `P10 == E`, while every
   stopping candidate carries pit-draw spread and has `P10 < E`. Under `score = a*E + (1-a)*P10`,
   **lowering alpha (more risk-averse) can only ever punish the stopping candidates** — risk
   aversion is structurally pro-STAY_OUT in this layer, which is the opposite of what a tyre on the
   edge should produce.

**Verdict: the MEASUREMENT is confirmed; the framing "a clip zeroes the tyre draw" is the wrong
target.** The clip is the correct arithmetic of a cliff-only cost model; the defect is the cost
model (no pre-cliff term) interacting with N26's output range. Fixing the clip is not the fix;
consuming `deg_rate` (or an equivalent cumulative term) for the non-stopping laps is — and that is a
modelling decision for the owner, not a patch.

---

### G3 — Claim 2: CONFIRMED as a real defect. The two-horizons defence fails and A1's numbers reproduce — but A1's PROPOSED FIX is wrong, and there is a second asymmetry A1 missed.

**The two-horizons question, settled.** The defence would be: `project_positions` prices the
window end, `terminal_liability` prices the race end, and both are individually correct. Both ARE
individually correct — at their own horizons. The defect is that `payoff()`
(`position_projection.py:691-693`) **sums them into one ranking statistic**. A sum of a window-end
quantity and a race-end correction is a race-end estimate, and a race-end estimate must apply the
same correction to every candidate. It does not:

- STAY_OUT gets its window-end score **plus** a race-end correction (the liability) whose own
  docstring argument is *"a rival who must still stop pays the same price later, so they are no
  threat"* (`:592-594`).
- PIT_NOW gets its window-end score and **no** race-end correction — the same rival, exempted from
  the liability by that argument, is counted as permanently passing us at rejoin, as if the
  Art. 30.5(m) stop they owe had been waived.

There is no horizon under which both treatments are right: score purely at the window end and the
liability should not exist; score at the race end and the credit is missing. The inconsistency is
internal to the payoff, not a defensible modelling split.

**Executed, both races, production path:** F7 slot counts reproduce exactly — Lusail 474 pass-if-pit
/ 348 pending (73.4%), Monza 493 / 414 (84.0%). The Monza worked example is real: laps 12-14, LEC P4,
**14 pending cars inside the pit loss; shipped PIT_NOW score -13.7 vs STAY_OUT +0.07**.

One precision correction to A1's table: "car-slots charged if we STAY OUT = 126" counts settled cars
within the mean pit loss. What the code ACTUALLY charges is **68** slots — the liability additionally
requires `mandatory_stop_pending is True` and uses the q_f-discounted exposure. The 348/414
asymmetric-slot figures (the ones the argument rests on) are unaffected.

**The decisive experiment A1 did not run.** I built a mirror of `_run_projection_mc` that reproduces
the shipped scores **exactly on all 110 laps** (self-check green), then enabled a race-end
residual-netting term: per draw, per rival with a KNOWN obligation, the terminal gap is
`projected_gap + their_residual - our_residual`, residual = `max(0, stop_loss - q_f*saving)`
(our residual only when the plan defers a stop we owe; their residual only when `stop_pending is
True` and they are not already serving it; `None` means no claim, no correction). This subsumes
`terminal_liability` as its we-owe/they-settled/they-behind quadrant. Results on the laps where the
drivers REALLY pitted (NOR: Lusail ~25 and ~44-45; LEC: Monza ~33-34, from the stint tables):

| lap | shipped STAY / PIT / UNDERCUT | corrected STAY / PIT / UNDERCUT |
|---|---|---|
| Lusail 25 (real stop) | +0.300 / **-5.424** / -4.731 | +0.300 / **+0.039** / **+0.299** |
| Lusail 44 (real stop) | -2.700 / -2.689 / **-2.434, UNDERCUT already wins** | unchanged (field settled, zero pending rivals) |
| Monza 33 (real stop) | +0.300 / **-3.959** / — | +0.300 / **+0.010** / — |
| Monza 13 (A1's worked example) | +0.057 / **-13.655** | -0.504 / -0.639 / **-0.404, flips to UNDERCUT** |

Race-wide, the STAY-PIT margin collapses from **median 3.98 / max 17.1 positions (shipped) to median
1.12 / max 9.2 (corrected)** at Lusail. The phantom charge was carrying the dominance.

**But note what the correction does NOT do**: the corrected argmax is still STAY_OUT on 52/57 Lusail
laps. On the real stop laps it produces near-exact TIES (0.300 vs 0.299; 0.300 vs 0.010), not a pit
preference. The residual gap is A1's F6 (a 5-lap window prices 100% of a stop's cost and ~5 laps of
its benefit; break-even ~92 laps at FRESH_GAIN 0.25) plus the unpriced tyre channel (G2). **So A1's
sentence "this, not the flat distribution, is why STAY_OUT dominates on 108/110" is OVERSTATED as a
single cause: F7 is the largest distortion by an order of magnitude, but removing it alone yields
indifference on stop laps, not stops. F6 and the G2 gap are joint causes and must be named in any
fix that claims to make the projection able to prefer a stop.**

**A1's fix #1 is REFUTED.** "Charge `rival.stop_loss_s` in `rival_time_deltas` when
`stop_pending is True`" (A1 fix list, item 1) re-introduces the same bug in mirror image. Executed
(exhibit E3): a pending rival 5 s AHEAD under STAY_OUT — shipped code correctly keeps them ahead
(mean projected position 2.0); under A1's fix their projected gap becomes -5 + 23 = **+18, they are
counted as falling behind us within the window**, and STAY_OUT gains a phantom position even when we
ALSO still owe our stop (race-end truth: the two pending stops cancel; order preserved). It also
asserts that every pending rival stops within the 5-lap window (violating the module's own "v1
trusts facts, never guesses about rival strategy" doctrine) and contaminates the margin tie-break
with counterfactual gaps. The correction has to live at the TERMINAL layer on projected gaps, not in
the in-window rival deltas.

**NEW defect A1 missed (same family).** `terminal_liability` selects rivals by their gap at DECISION
time (`rival.gap_s > 0`, `position_projection.py:617-619`), not by their PROJECTED window-end gap.
A rival currently AHEAD who pits inside the window (charged via `is_pitting` in
`rival_time_deltas`) crosses behind us in the projection — STAY_OUT is credited that place at the
window end, and because the liability never re-examines projected gaps, the place is never charged
back when we take our own still-owed stop. Measured on real laps: Lusail 32 (STAY_OUT 2.300 shipped
to 1.300 under netting) and 33 (-0.845 to -1.386). Both flips the netting produced at Lusail (32, 33
to UNDERCUT) come from this mechanism, not from the headline F7 case — though both are knife-edge
(0.004 and 0.015 margins), so they are evidence of the mechanism, not of a dramatic decision swing.

---

### G4 — Claim 3: CONFIRMED and UNDERSTATED. And the parent's suspicion about `P90 > P10` is RIGHT: it is not a universal property, so the assertion must be scenario-constructed.

**Verified by grep and by reading every hit** (per the grep-is-not-an-audit rule, every P10/P90
occurrence in `tests/` was read in context):

- `tests/mc/test_mc_is_a_real_decision.py:181-183, 404-406, 475-477, 534-536` — all `np.full`.
- `tests/mc/test_position_projection.py` — **also all `np.full`** (`:58` helper, `:96`, `:363-366`);
  A1's F8 missed this file. Not one call into the primitive in the whole suite passes a
  non-degenerate draw vector.
- The only P10/P90 assertions are shape checks and the alpha-collapse identities
  (`test_strategy_goldens.py:141` asserts score==P10 at alpha=0 — true even for a point mass).
  **Nothing asserts P90 > P10.** Confirmed.
- The frozen golden pins the LEGACY branch (no `rivals` kwarg → `_has_usable_gaps` routes to the
  seconds path), in the one tyre regime (cliff 3/5/8 straddling W=5) where STAY_OUT has spread.
  Real N26 outputs put that regime on 4 of 110 laps, and the projection branch — the one every
  surface takes when rivals carry gaps — has **no golden at all**.
- The invariant sweep DOES assert "PIT_NOW is reachable"
  (`test_staying_out_and_pitting_are_both_reachable_on_the_projection`) and it passes — but its
  fixture sets `rival_stop_pending: {"B": False, "C": False}` (`:178`), the all-settled
  configuration in which the liability fires at full strength and the F7 asymmetry cannot occur.
  Real races put 73-84% of the passing cars in the pending state the fixture never exercises
  (measured, G3). A1's diagnosis of WHY the sweep passes is confirmed.

**Would `P90 > P10` hold universally for a correct implementation?** **No — executed
counter-example (exhibit E4):** PIT_NOW with fully varying pit draws (22.0-24.4 s) against a single
rival 60 s behind produces exactly 1 unique payoff (E=P10=P90), because positions quantise and the
margin term saturates at the `MARGIN_CLIP_S = 3.0` clip. A correct implementation yields point
masses whenever no projected gap sits inside the draw support. So an UNCONDITIONAL per-candidate
assertion would be false, and A1's proposed test (its fix-list item 2, "assert P90 > P10 for at
least one candidate under a non-degenerate draw") is correctly scoped — but the parent's instinct
that the property is conditional is right, and the resolution is standard test practice: the fixture
CONSTRUCTS the regime (triangular draws + a rival placed inside the pit-loss support + a
cliff-straddling STAY_OUT case), and inside that fixture the assertion is unconditional. That is a
strong fix, not a weak one; what would be weak is asserting it on arbitrary states.

---

### G5 — Claim 4 (alpha): A1's two-part resolution is CONFIRMED exactly. "Decorative" is wrong; "flips 5/110" is right; invariance-under-dominance is a THEOREM.

- **Theorem, not accident:** `score(a) = a*E + (1-a)*P10` is a convex combination. If candidate A has
  both `E_A >= E_B` and `P10_A >= P10_B`, then `score_A(a) >= score_B(a)` for every `a` in [0,1] —
  term-by-term monotonicity, two lines of arithmetic. Where dominance holds, no alpha can flip the
  argmax, and measuring a sweep there adds nothing.
- **Dominance measured (production path):** STAY_OUT >= PIT_NOW on both E and P10 on **55/57 Lusail
  and 53/53 Monza laps** — 108/110, matching A1.
- **The sweep, reproduced digit-for-digit:** argmax changes with alpha on exactly **laps 34, 35, 41,
  44, 55 at Lusail and 0 laps at Monza**. Winner counts per alpha match A1's table exactly
  (STAY_OUT 56/56/55/54/54/54/53; PIT_NOW wins once, lap 41, at alpha=0.0 only; UNDERCUT
  0/1/2/3/3/3/4). So: 5/110 laps, all five arbitrating between STAY_OUT and candidates whose edge
  comes from the additive N16 bonus (PIT_NOW's single win is the P10-tail quirk at pure risk
  aversion).
- **The connection A1 stated in passing deserves the headline:** claim 1 and claim 4 are the same
  finding. Because STAY_OUT is a point mass, its P10 equals its E, while every stopping candidate
  has P10 < E — so the risk-aversion dial can only ever penalise stopping. Alpha is not decorative;
  it is **directionally biased**: on this layer it is a one-way ratchet toward STAY_OUT.

---

### G6 — Why did earlier bug hunts miss this? (the owner's second question, answered concretely)

Five mechanisms, all verifiable:

1. **The layer is four days old.** `position_projection.py` and `_run_projection_mc` shipped
   2026-07-25 (epic #550, commits 7f91d1d..8052343). Every famous hunt in the lessons log
   (the 2026-07-16 wave: scoping, sentinels, inference-vs-notebook) predates the file's existence.
   There were no "many bug hunts" over THIS code — there were three gates in one week.
2. **The gates that DID run attacked the repo's historical bug classes** — wiring, sentinel
   collisions, keyspace drift, dead config, silent fallbacks — and found real instances of each
   (dead liability, position-defaults-to-1, free rival stops; see the #550 sprint log). Those
   classes are all INPUT defects. The G3 defect is not an input defect: it is the interaction of two
   individually-correct terms summed at inconsistent horizons — precisely the "bugs live in the
   interaction between two correct-looking pieces" class the audit doctrine warns about, and none of
   the three gates framed a claim of the form "the same fact is priced differently across candidates".
3. **The only ground truth validates the horizon where the code is right.** The 1810-stop / 86.5%
   eval (`src/strategy/eval/projection.py`) grades PROJECTED vs ACTUAL rejoin position — its own
   comments say the rejoin geometry, not the decision, is under test. There is no label for "was
   STAY_OUT the right call", so no measurement ever compared candidates against reality. The
   validation stopped exactly at the boundary of the defect.
4. **The tests remove the phenomenon before asserting** (G4): constant draws, the all-settled
   fixture, the golden on the legacy branch. The suite that could have seen it was calibrated —
   innocently, state by state — to the configurations where it is invisible.
5. **The shipped behaviour LOOKED like the win.** The redesign's own success metric was "legacy
   answered STAY_OUT on all 50 decision laps; the projection answers STAY_OUT x46, UNDERCUT x3,
   PIT_NOW x1 — it discriminates" (sprint memory). The same distribution A1 reads as the symptom was
   recorded at ship time as the improvement. Nobody asked "SHOULD the projection itself ever prefer
   a stop?" because ANY variety beat the constant it replaced.

### G7 — Why the arcade looks reasonable anyway (the owner's first question)

Three concrete, load-bearing reasons — the owner's observation is correct AND compatible with every
confirmed finding:

1. **STAY_OUT is genuinely the right call on ~90% of laps.** A driver pits 1-2 times in ~55 laps.
   A layer that says STAY_OUT almost always agrees with reality almost always — the bias is only
   visible on the 2-4 decision laps per race, and only if you compare scores, not actions.
2. **The asymmetry fades exactly where a human checks.** The F7 phantom charge requires PENDING
   rivals inside the pit loss. By the real second-stop window the field has cycled: at Lusail laps
   41-46 the passing cars are all settled (measured: passing_pending = 0), the scores tighten to
   tenths, and on lap 44 — the lap NOR actually pitted — the shipped layer itself picks UNDERCUT.
   The first-stop window (lap 25: PIT scored -5.4 while NOR really pitted) is where it is wrong,
   and a first stop "early-ish under no pressure" is exactly the call a spectator cannot easily
   grade against reality.
3. **The arcade does not display the projection argmax — it displays Layer 3.** The arcade runs
   `profile="rich"` (`src/arcade/strategy_pipeline.py:47-48`); the shown action is the LLM
   synthesis, for which the MC is one hint among the full sub-agent outputs (N28's pit
   recommendation, N26's tyre warnings, the guardrails, and since v2.5.0 the decision memory).
   Previously measured on this repo: 12 of the 14 recommendation fields are LLM-originated. A
   reasonable-looking arcade is evidence about Layer 3, not about `_run_projection_mc`.

---

### G8 — What I tried to break and COULD NOT

- **A1's measured numbers.** I re-derived every headline figure from scratch through the production
  entry point rather than A1's monkeypatch instrumentation: 43/57 and 51/53 point masses, 55/57 and
  53/53 dominance, the full 7-alpha winner table, the 5 flip laps, 474/348 and 493/414 F7 slots,
  cliff_p10 min/median, 4 and 0 straddling laps. Every one reproduced exactly. The only discrepancy
  found in the whole report is the 126-vs-68 "charged if stay" label (definition, not arithmetic).
- **My own mirror.** Before trusting the corrected scorer, its credit-off mode had to equal the
  shipped `_run_mc_simulation` output dict on all 110 real laps. It does (110/110), so the
  counterfactual is not an artifact of a divergent reimplementation.
- **The two-horizons defence of claim 2.** I steelmanned it (it was the owner-facing objection) and
  it does not survive `payoff()` summing the horizons; I could not construct a reading under which
  both branches are simultaneously correct.
- **The convexity theorem.** It is arithmetic; no counter-example exists. I also checked E >= P10 is
  not even needed for it.
- **The clip as a bug.** I tried to make `max(0, racing - cliff_laps)` misbehave in its live regime
  (cliff inside window): it produces a correctly-ordered, non-degenerate spread (382 unique payoffs).
  The clip is innocent; the indictment belongs to the missing pre-cliff term.
- **The golden reaching the projection branch.** It cannot — routing requires usable rival gaps and
  the golden passes none. I checked there is no other frozen pin on the projection path. There is
  none.
- **The sampling / CRN claims.** Spot-confirmed n=500 via `CFG.n_sim` on the production path and
  identical draw vectors across candidates in the mirror (a divergence would have broken the
  110/110 equality). Consistent with A1's "verified and not broken" list; I did not re-instrument
  further.

---

## Deliverable 2 — surgical changes for the CONFIRMED findings

**S1 (claim 2, the defect). Replace the one-sided liability with race-end residual netting on
PROJECTED gaps.** `src/agents/position_projection.py`: add a terminal-correction step computed
per draw from `projected_gaps` (inside `project_positions`, or a companion consuming its result):
`terminal_gap = projected_gap + their_residual - our_residual`, with residual =
`max(0, stop_loss - q_f*saving)`; our residual only when the plan defers a stop we KNOW we owe
(`mandatory_stop_pending is True`), their residual only when `stop_pending is True` and not already
serving it, and **`None` obligations contribute no correction in either direction** (the existing
"no fact, no claim" doctrine). `payoff()` then scores `current - terminal_positions` plus the
existing margin term; `terminal_liability` is subsumed (it is the we-owe/they-settled/they-behind
quadrant) and can be deleted with its call site. What must NOT move: `project_positions`'
window-end `positions`/`margins_s` outputs (the 1810-stop eval imports and grades exactly those —
the netting must be a separate term so the rejoin validation keeps meaning), the legacy branch
(byte-identical golden), eligibility rules, the N16 bonus, CRN. Goldens: no frozen golden exists on
the projection path (G4), so nothing re-freezes; the `test_mc_is_a_real_decision` sweep shares will
move and its thresholds must be re-run, not tuned to pass. **Classification: the INCONSISTENCY is a
bug by the module's own docstring logic, but the replacement term changes shipped, A/B'd behaviour
- it is a modelling decision that needs the owner's call, shipped behind its own PR with the
measured before/after (this file, G3 table) attached.** Explicitly do NOT apply A1's fix #1
(`rival_time_deltas` charging pending stops) — refuted in G3.

**S2 (claim 3). Give the projection branch teeth.** `tests/mc/`: (a) a fixture whose draws are
`rng.triangular(...)` with a rival gap placed INSIDE the pit-loss support and a cliff straddling
the window; assert `P90 > P10` for STAY_OUT and PIT_NOW there (E4 shows why the scenario must be
constructed); (b) a projection-path golden (real rivals kwarg, frozen dict) so the branch every
surface takes is pinned the way the legacy branch already is; (c) add pending-rival states
(`stop_pending: True`) to the invariant sweep so the F7/S1 regime is exercised. Zero production
risk; do S2 BEFORE S1 so S1's effect is measurable in the diff.

**S3 (claim 1). Decide what the tyre channel prices.** Either consume `TireOutput.deg_rate` (already
emitted, currently unread by the MC) as a per-lap cost on non-stopping laps — which un-flattens
STAY_OUT for real and lets N26's uncertainty reach the score — or document that the tyre channel
speaks only through the cliff and accept 4/110-lap relevance. Owner's call; touches
`driver_time_delta` and the sampling in `_run_mc_simulation`; the legacy branch must not move.

**S4 (small, true bug, no owner call needed).** `terminal_liability`'s decision-time-gap filter
(G3's new defect) disappears inside S1; if S1 is deferred, it can be patched standalone by
evaluating the liability on projected rather than current gaps. Also: correct A1's F7 table label
(126 -> 68 actually-charged) wherever that number is quoted onward.

---

## Verdict summary

| # | Claim | Verdict |
|---|---|---|
| 1 | STAY_OUT point mass via the clip | **CONFIRMED as measured; MISLABELED as a clip defect.** The clip is correct arithmetic; the defect is the cliff-only cost model (pre-cliff deg unpriced, `deg_rate` unread) meeting N26's real output range. Flatness per se is partially defensible; its alpha side effect (risk aversion can only punish stopping) is not. |
| 2 | stop_pending asymmetry is the real defect | **CONFIRMED — it is one defect, not two horizons; the missing term is the race-end credit.** Numbers reproduce exactly. BUT: A1's proposed fix #1 is REFUTED (mirror-image phantom, executed); "root cause of dominance" is OVERSTATED (correction yields ties, not stops — F6 and the tyre gap are joint causes); plus one NEW same-family defect found (liability blind to in-window crossers, Lusail 32-33). |
| 3 | Tests cannot see any of this | **CONFIRMED and UNDERSTATED** (`test_position_projection.py` also constant-fed; the golden pins the legacy branch, and the projection branch has no golden at all). `P90 > P10` is NOT universal for a correct implementation (executed counter-example); the assertion must be scenario-constructed, which is a strong fix. |
| 4 | Alpha decorative vs 5/110 | **"5/110" is right, reproduced digit-for-digit; "decorative" is wrong.** Invariance under dominance is a theorem (convexity), holding on 108/110 laps; and via claim 1 the alpha dial is structurally one-directional (pro-STAY_OUT). |
