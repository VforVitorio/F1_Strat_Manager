# FABLE A-2 — Adversarial gate on residual netting (commit f134541, PR #733, issue #726)

**Auditor:** Fable adversarial gate (A-2). **Date:** 2026-07-29.
**Scope:** `src/agents/position_projection.py` — deletion of `terminal_liability`, introduction of
`terminal_positions` via `terminal_gap = projected_gap + their_residual - our_residual`,
`payoff` scoring on the terminal horizon with window-end margins.
**Rules:** read-only on the repo except this file. Any mutation is reverted immediately and logged in §Revert log. No LLM calls; `profile="no-llm"`.

## Checklist

- [ ] C1 — reproduce the 110-lap before/after table independently
- [ ] C2 — `f1-eval projection` unchanged (number AND structural reason)
- [ ] C3 — construct the mirror-image phantom scenario against the new code
- [ ] C4 — the eight rewritten contract tests still test what they claim; cancellation vs accident
- [ ] C5 — no double-charge and no zero-charge path for the same stop
- [ ] C6 — quantify argmax changes across the 110 laps; direction
- [ ] Extra — margin-on-window-gaps coherence; empty-usable branch; degenerate regimes
- [ ] Revert log
- [ ] What I tried to break and could not

## Working state

- Repo on `dev` @ 04a8288 (merge of #733); f134541 present. Working tree clean apart from untracked audit files.
- Implementation read in full: `src/agents/position_projection.py` (`_stop_residual_s` :586, `_terminal_gaps` :601, `project_positions` :659, `payoff` :716).

## Early observations to be tested (not yet findings)

1. **Cancellation may be approximate, not exact**: `our_residual` is computed from the per-draw
   sampled `pit_loss_s` array; `their_residual` from the scalar `rival.stop_loss_s`. If production
   samples our pit loss, "leading a pack that all owe a stop" cancels only in expectation, not per draw.
2. **`stop_loss_s` defaults to 0.0** on `RivalState`. A pending rival whose adapter never populated
   `stop_loss_s` contributes a zero residual while we are charged in full — the one-sidedness the fix
   exists to remove, inverted. Need to check every adapter that builds `RivalState`.
3. **`rival_time_deltas` charges an `is_pitting` rival the FULL `stop_loss_s` undiscounted**, while a
   pending rival's residual is q_f-discounted. Asymmetry between the two charging paths (pre-existing,
   but the netting now makes them comparable).

---

## Consumer analysis (executed reads, pre-measurement)

- Production path: `strategy_orchestrator.py:1118` builds `RivalState` via `_rival_states_from_lap_state`
  (:788-799) — every rival gets `stop_loss_s = traversal_s + RIVAL_STOP_PRIOR_S(=2.8)` (or the
  `rival_pit_loss_s` override), so the "pending rival with stop_loss 0.0" inversion does NOT fire on the
  production path. `stop_pending` comes from `pit_context["rival_stop_pending"]`, populated by
  `race_state_manager.py:522` and mirrored by the telemetry twin at `endpoints/strategy.py:607`.
- OUR pit loss is per-draw (`traversal_s + sampled pit_s`, :1124) while the rival residual is a scalar
  prior. The golden cancellation test pins BOTH at 22.0 — so "cancellation" is exact only in the test
  geometry; in production it is exact only in expectation if E[sampled stop] == 2.8. To quantify.
- Eval path: `src/strategy/eval/projection.py:203` builds `RivalState` with NO `stop_pending` (default
  None) and NO `mandatory_stop_pending` in `_GROUND_TRUTH_CONFIG` (:61), and grades
  `result.positions[0]` (:260) — the window horizon, computed at `position_projection.py:700` BEFORE
  `_terminal_gaps` runs. Structural independence confirmed by code path; number re-run pending.
- `f1-eval projection` (the CLI) writes into `documents/eval_reports/` — a repo mutation. To honour
  read-only I call `measure_projection_ground_truth()` directly and never invoke the writer.

## Attack plan

1. Unit probes (pure numpy, no data): C3 phantom scenario matrix (our obligation True/False/None ×
   candidate), C5 charge-count table, cancellation-exactness with sampled vs scalar losses,
   empty-usable branch, degenerate margins.
2. 110-lap reproduction: drive Lusail/NOR + Monza/LEC through `run_lap(profile="no-llm")` once per lap,
   then compute the MC block twice on the SAME agent outputs — HEAD module vs the pre-change module
   (`git show f134541~1`, injected via `sys.modules`, zero repo mutation). Checkpointed per lap to
   scratchpad JSONL.
3. Ground truth (C2): `measure_projection_ground_truth()` full-precision, in background.
4. `uv run pytest tests/mc` to confirm the suite is green as claimed.

## Executed findings — unit probes (scratchpad/unit_probes.py, run against HEAD and f134541~1)

### C2 — CONFIRMED (both halves)
- `measure_projection_ground_truth()` re-run: **1810 stops / 71 races / within_one = 0.8646408839779005**
  — matches the claim to the last digit (exact = 0.5911602209944752).
- Structural half: the eval builds `RivalState` with `stop_pending` defaulted to `None`
  (`src/strategy/eval/projection.py:203-208`) and a config with `mandatory_stop_pending=None` (:61),
  and grades `result.positions[0]` (:260), which `project_positions` computes at
  `position_projection.py:700` BEFORE `_terminal_gaps` is called (:706). The netting cannot reach the
  graded number by construction, not by luck. CONFIRMED.

### C3 — the mirror-image phantom: ABSENT at the window horizon; ONE-SIDED regime survives at terminal
Scenario: pending rival 5 s ahead (gap −5, stop_loss 23), q_f=0, current P2, flat config.
- **Window horizon (what the refuted fix polluted): clean.** STAY_OUT `positions=[2.]`,
  `margins=[0.]` under every obligation state — the rival never "becomes +18" where the eval or the
  margin can see it. The refuted fix's exact failure shape does not exist.
- **ours=True**: OLD payoff 0.0/0.0 (STAY/PIT), NEW payoff 0.0/0.0 — netting cancels. No phantom.
- **ours=False**: NEW STAY_OUT +1.0 vs PIT_NOW 0.0 — STAY_OUT gains the place at terminal, and it is
  EARNED (they owe a stop, we do not).
- **ours=None**: NEW STAY_OUT **+1.0** vs PIT_NOW 0.0. Here the netting is one-sided by construction:
  our unknown obligation contributes nothing while their known obligation contributes +23 s, so
  STAY_OUT is credited a full place on the bet that we owe nothing. The docstring's "an unknown
  obligation contributes NOTHING in either direction" (:631) is true of the TERM but not of the
  COMPARISON: the netted differential still moves by a whole place. Deliberate and documented in the
  commit, but it is the one regime where the old one-sidedness survives in mirror form.
  Prevalence on real laps measured below (C1/C6 harness) before rating severity.

### C5 — CONFIRMED: exactly one charge on every path, plus two deliberate zero-charge paths
Charge table (rival +5 behind, loss 23, q_f=0): `(is_pitting, stop_pending)` →
(window charge, terminal charge): (F,T)→(0,23) · (F,F)→(0,0) no stop owed · (F,None)→(0,0)
deliberate · (T,T)→(23,0) exclusion works · (T,F)→(23,0) · (T,None)→(23,0). No combination charges
twice. Our side: PIT_NOW terminal−projected = 0.0 (no residual when the plan stops), STAY_OUT −23.0
(charged once, at terminal). The zero-charge path is `stop_pending=None` for a rival who in fact owes
a stop — deliberate ("a claim needs a fact"), same rule as before the change.
- Sub-observation (LOW, pre-existing): an `is_pitting` rival is charged the FULL `stop_loss_s`
  undiscounted at :579 even in neutralised draws, while our PIT_NOW gets `saving` subtracted in those
  draws (`driver_time_delta` :541-542). Asymmetry favours our stop under SC draws; not introduced by
  f134541.

### C4 (partial) — cancellation is EXACT only in the test geometry; production is approximate
- Test `test_leading_a_pack_that_still_owes_its_stop_is_free` pins our pit loss AND rival
  `stop_loss_s` at the same constant (22.0) → cancellation is an identity there (verified: terminal
  cost 0.0 with matched constants).
- Production geometry (our loss = 21 + Triangular(2.2,2.8,3.8) per draw; rival = 21 + 2.8 flat prior,
  `strategy_orchestrator.py:767,785,1124`): pack behind at +0.6/+1.4/+2.5 all pending → **45/500 draws
  (9%) the "free" lead is not free; mean terminal cost 0.09 positions**. Two causes: per-draw noise,
  and E[triangular] = 2.933 ≠ 2.8 prior (a small systematic +0.13 s against us). So the docstring's
  "their residuals and ours are the same size and cancel" (:626-627) is FALSE per draw in production —
  it holds in the mean, approximately. The contract test proves the identity CAN hold, not that
  production produces it. OVERSTATED in the docstring; behaviour itself is defensible (our stop time
  is genuinely uncertain).

### Extra edges
- Empty/unusable rivals: positions=terminal=1, payoff = current−1, IDENTICAL for every candidate →
  a four-way tie decided by iteration order. Pre-existing, unchanged by f134541.
- Degenerate 200 s pit-loss draw: terminal bounded by field size (1+n_rivals); payoff cannot go below
  current − (1+n). No nonsense regime found.
- Margin/terminal mismatch is REAL and demonstrable: 3 settled rivals far behind, we pending, STAY_OUT
  → window margin clipped 3.0 earns +0.3 payoff while ALL of them pass us at the terminal horizon
  (positions 1 → terminal 4). The +0.3 is credit for a buffer the score's own horizon says evaporates.
  Bounded at 0.3, but the fix's headline regime is near-ties (median STAY−PIT gap claimed 1.12), so
  the bound is no longer negligible relative to the gaps. Quantified on real laps below.

## Executed findings — 110-lap reproduction (scratchpad/lap_harness.py -> lap_results.jsonl)

Method: Lusail/NOR (57 laps) + Monza/LEC (53 laps) through `RaceReplayEngine` +
`run_lap(profile="no-llm")`, sub-agents run ONCE per lap, then `_run_mc_simulation` computed twice on
identical outputs — HEAD module vs `git show f134541~1` injected via `sys.modules` (no repo file
touched). 110/110 laps produced results, 0 errors.

### C1 — five of six rows CONFIRMED to the third decimal; the argmax row does not reproduce
| claim | measured | verdict |
| --- | --- | --- |
| Lusail 25 PIT −5.424 → +0.039 | before score −5.424, after +0.039 | CONFIRMED (exact) |
| Lusail 44 unchanged | all four candidates byte-identical before/after | CONFIRMED |
| Monza 33 PIT −3.959 → +0.010 | −3.959 → +0.010 | CONFIRMED (exact) |
| Monza 13 flips to UNDERCUT | STAY +0.057/PIT −13.655 before; after argmax UNDERCUT (−0.404 vs STAY −0.504) | CONFIRMED |
| Lusail STAY−PIT median 3.98/max 17.1 → 1.12/9.2 | 3.981/17.122 → 1.116/9.226 | CONFIRMED (exact) |
| Lusail argmax STAY_OUT 54/57 → 52/57 | 54/57 → **53/57** (exactly one flip: lap 32, STAY→UNDERCUT) | **NOT REPRODUCED** |

The disputed count lives on knife edges: Lusail 32 flips on +0.004 (UNDERCUT 1.304 vs STAY 1.300) and
Lusail 25 fails to flip by −0.001 (UNDERCUT 0.299 vs STAY 0.300). One more flip anywhere gives 52/57.
Determinism rerun in progress to establish whether the capture procedure itself can wobble by ±0.002.

### C6 — CONFIRMED in substance: near-ties, not stop-preference
- Argmax changed on **2 of 110 laps**, both STAY_OUT → UNDERCUT (Lusail 32, Monza 13). Zero changes
  toward STAY_OUT, zero to PIT_NOW. The layer does NOT now "prefer a stop" — the honest-ceiling claim
  holds; if anything my count is one flip SHORT of the claimed Lusail table.
- Monza STAY−PIT margin: median 7.953/max 16.049 → 2.238/7.058 (not claimed, consistent direction).
- Obligation census on the 110 laps: ours_pending True on 77, False on 33, **None on 0**; rivals with
  pending=None: 0 laps. So the one-sided `ours=None × rival pending=True` regime (C3 finding) never
  fires on replay data — stint history settles every lap here. Its exposure is surfaces whose
  compound history is incomplete (`stint_history.py:32` returns None), not the replay path.

### Margin-on-window-gaps: now MATERIAL, with named laps
After the change, **20 of 110 laps** have |STAY−PIT| ≤ 0.3 — inside the margin tie-break's cap. On
Lusail laps 20/21/25 the entire STAY_OUT score is the clipped margin bonus (gained=0, +0.3) while
UNDERCUT sits at 0.256-0.299: the argmax on those laps is decided by a tie-break computed at the
WINDOW horizon on top of a score computed at the TERMINAL horizon. Deliberate and disclosed in the
docstring (:727-729), but the fix's own success (median gap 3.98→1.12) is what promoted this
tie-break from decoration to decider on ~18% of laps. Flagged as a consequence to own, not a bug.

### Suite status
`pytest tests/mc` at HEAD: **142 passed** (6:11), including the ground-truth gate.

## C1 resolution — the argmax row is a property of an unseeded draw, not of the code

Two byte-identical invocations of the SAME capture procedure at HEAD, same data, same seed-42 MC:

| | run 1 | run 2 |
| --- | --- | --- |
| Lusail after argmax | STAY_OUT **53/57** (flip: lap 32) | STAY_OUT **52/57** (flips: laps 32, 33) |
| Lusail 33 STAY_OUT score | −0.893 (P10 ≈ −0.856) | −1.386 (P10 ≈ −1.830) |
| STAY−PIT median/max | 1.116 / 9.226 | 1.116 / 9.226 |
| spot-check laps 25/44 | identical | identical |

The MC's own rng is seeded (`_run_mc_simulation`, seed=42), but the tire sub-agent runs MC Dropout
with `model.train()` and **no torch seed** on the production path (`src/agents/tire_agent.py:1094`;
contrast `src/strategy/eval/tire_holdout.py:225-226`, which seeds). tire_out's P10/P50/P90 wobble
run-to-run, the cliff triangular follows, and on laps where a quantile sits on a gap crossing the
candidate score moves by ~half a position (lap 33: 0.493). The claimed 52/57 came up in my second
run exactly; my first gave 53/57. **The commit message bakes one realization of a stochastic capture
into the record as if it were a property of the change.** The five other rows are stable across both
runs and confirmed exactly. Verdict: C1 OVERSTATED on the argmax row only — and by the claim's own
standard ("a number I got right by accident is still a number I cannot defend"), that row is
currently indefensible either way. Fix is one line of seeding, or reporting the census as 52-53/57.

## Verdicts

| claim | verdict |
| --- | --- |
| C1 reproduction | **OVERSTATED** — 5/6 rows confirmed to the third decimal in two independent captures; the Lusail argmax census (52/57) is unstable under the unseeded tire MC Dropout (52 or 53 depending on the draw) |
| C2 ground truth unchanged | **CONFIRMED** — 1810 / 71 / 0.8646408839779005 re-measured; structural: eval never sets `stop_pending`/`mandatory_stop_pending` and grades `positions`, computed before `_terminal_gaps` runs |
| C3 mirror phantom absent | **CONFIRMED at the window horizon and in both settled regimes** — the +18 shape cannot occur where the eval, the margins or the rejoin count look; residual netting cancels (ours=True) or is earned (ours=False). One narrower one-sided regime survives: see defect D1 |
| C4 contracts survive | **CONFIRMED with one overstatement** — all 8 rewritten tests assert what they claim (plus 1 new direct test); the mechanism IS cancellation (a literal subtraction, exact 0.0 with matched constants, verified), not a numerical accident. But "their residuals and ours are the same size and cancel" is exact only in the matched-constant test geometry; in production (sampled loss vs 2.8 s prior) a tight pack behind is NOT free in 9% of draws (mean 0.09 positions) |
| C5 no double/zero charge | **CONFIRMED** — full charge table executed: exactly one charge on every charged path, `is_pitting` exclusion works, our residual is zero whenever the plan stops. Zero-charge paths are exactly the two deliberate `None` cases |
| C6 honest ceiling | **CONFIRMED** — 2 of 110 argmaxes changed, both STAY_OUT→UNDERCUT, none toward STAY_OUT; the layer does not now prefer stops. The near-tie regime is real: 20/110 laps end within the 0.3 margin cap |

## Defects / consequences introduced or exposed by the fix

1. **D1 (MEDIUM, latent).** `ours=None × rival stop_pending=True` nets one-sidedly: STAY_OUT is
   credited a full terminal place (probe: payoff +1.0 vs PIT_NOW 0.0) on the bet that we owe
   nothing, while the same unknown charged nothing under the old code. Deliberate per the commit
   message, prevalence 0/110 on replay laps (stint history always settles there), but live on any
   surface where `stint_history` returns None (`src/simulation/stint_history.py:32`) — and **no test
   covers the combination** (the suite tests ours=None×settled and ours=True×their-None, never
   ours=None×their-True). A one-line test would pin whichever behaviour is intended.
2. **D2 (MEDIUM, evidence).** The reproduction's argmax census is not reproducible: unseeded MC
   Dropout (`tire_agent.py:1094`) lets per-lap scores move ~0.5 between identical runs. Pre-existing
   stochasticity, but f134541's commit message publishes 52/57 as a measured property of the change.
3. **D3 (MEDIUM, disclosed-but-now-material).** The window-horizon margin tie-break decides the
   argmax inside the near-tie regime the fix itself created: on Lusail 20/21/25 STAY_OUT's entire
   score is the clipped +0.3 margin while UNDERCUT sits 0.001-0.044 below it, and 20/110 laps end
   inside the cap. The docstring owns the choice; the quantified consequence should be owned too
   (#727 will land on top of these knife edges).
4. **D4 (LOW, docstring).** ":626-627 'their residuals and ours are the same size and cancel'" —
   false per draw in production; true only in expectation, and biased 0.13 s against us
   (E[Triangular(2.2,2.8,3.8)]=2.933 vs RIVAL_STOP_PRIOR_S=2.8).
5. **D5 (LOW, pre-existing, out of scope).** An `is_pitting` rival is charged the full undiscounted
   `stop_loss_s` even in neutralised draws while our simultaneous stop is discounted by `saving` —
   asymmetry in our favour under SC; predates f134541.

## What I tried to break and could NOT

- The refuted-fix phantom in its original shape: a pending rival ahead never "becomes +18" at the
  window horizon under any obligation state, any candidate — `positions` and `margins_s` are computed
  from `projected_gaps` before the netting exists (:697-704 vs :706).
- Double-charging: all 6 rival `(is_pitting × stop_pending)` combinations and all 3 our-obligation
  states — no path charges a stop twice, on either side.
- The 1810-stop ground truth: re-measured to the last digit AND shown structurally unreachable.
- The suite: 142/142 green at HEAD, 6:11, ground-truth gate included.
- Degenerate regimes: 200 s pit-loss draws, empty rivals, all-None/NaN gaps, terminal corrections
  bigger than the field — everything stays bounded in [1, 1+n_rivals]; payoff never goes nonsensical.
- The negative control: Lusail 44 byte-identical before/after, in both independent captures.
- The headline distribution claims: STAY−PIT median/max identical to 3 decimals across two
  independent stochastic captures — those numbers ARE properties of the change.
- The cancellation identity: matched constants give exactly 0.0 terminal cost, every draw.

## Revert log

**No repository file was modified at any point in this audit.** Verified at the end of the run:
`git status --short` shows only pre-existing untracked files plus this report; `git diff` and
`git diff --cached` are both empty. The one repo-mutating tool in the path — the `f1-eval projection`
CLI, which writes into `documents/eval_reports/` — was deliberately bypassed by importing
`measure_projection_ground_truth()` directly. The pre-change module was obtained via
`git show f134541~1:...` into a detached in-memory module (sys.modules injection), never a checkout.
All working artifacts (probes, harness, JSONL captures) live in the session scratchpad outside the
repo.

## Checklist (final)

- [x] C1 — reproduced; 5/6 exact, argmax row unstable (D2)
- [x] C2 — confirmed, both halves
- [x] C3 — phantom absent; D1 names the surviving one-sided regime
- [x] C4 — confirmed; cancellation is real arithmetic; docstring overstates production exactness (D4)
- [x] C5 — confirmed; charge table executed
- [x] C6 — confirmed; direction quantified (2/110, both toward stops)
- [x] Extras — margin materiality (D3), empty branch, degenerate regimes
- [x] Revert log — nothing to revert, evidenced
