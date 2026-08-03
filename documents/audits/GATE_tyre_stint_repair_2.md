# GATE 2 — tyre-stint repair rewrite (#790): adversarial re-audit

**Date:** 2026-08-02 · **Auditor:** adversarial re-gate (read-only except this file)
**Scope:** the REWRITTEN `src/f1_strat_manager/tyre_stint_repair.py`, its wiring in
`src/f1_strat_manager/laps_augment.py` (`_stint_corrections`, `_apply_stint_corrections`,
call in `augment_featured_laps`), and `tests/agents/test_tyre_stint_repair.py` (16 tests),
on branch `dev` (uncommitted). Previous gate: `GATE_tyre_stint_repair.md` (9 findings,
3 blockers, verdict "not safe to ship as-is").

**Mandate:** verify each prior finding is genuinely closed, then hunt what the rewrite
itself broke. Findings appended incrementally with executed evidence only.

## Checklist

- [x] A. Flagship fixed? Did nulling go too far anywhere? — **FIXED; no over-nulling live** (71-race sweep, block-by-block)
- [x] B. `_prior_usage_at` arithmetic + interaction with the nulling pass — **EXACT on all 12 anchors; interaction inert by construction**
- [x] C. Sentinel guard: BOR/STR/HAD + Montréal TSU — **handled by nulling / deliberately abandoned; see N2**
- [x] D. Integration — **113/113 nulled laps reach featured; hole is structural only (N3)**
- [x] E. The 16 tests — **all traced to the paths they claim to pin; one docstring lies about the mechanism (N5)**
- [x] F. Model-input impact — **406 featured-2025 rows changed; golden metrics unmoved, #782 failures identical**

---

## A/B (part 1) — Miami executed anatomy: flagship VERIFIED fixed; anchors arithmetically exact

Executed `repair_tyre_stints` on `data/raw/2025/Miami_Gardens/laps.parquet`:

- **NOR lap 29: `TyreLife 5.0 -> NaN`** — the flagship row is closed.
- **Impossible-age sweep** (numeric `TyreLife < laps since last pit start − 5`, all drivers):
  **171 rows before repair -> 0 after.** The prior gate's ≥71 count was on a narrower
  pre-first-pit scan; under both metrics the survivors are zero.
- Report: 12 boundaries corrected, 333 ages rebuilt, **146 fabricated ages nulled**,
  18 drivers touched. GAS (9 nulled), HUL (13), BEA (5) — the Finding-2 trio — are now covered.
- **Boundary anchors, all 12 verified**: `rebuilt@boundary == boundary − out_lap + 1 + prior_usage`
  exactly, with `prior_usage = published_TyreLife@boundary − 1`. ALO's used set
  (published 2.0, `FreshTyre False`) anchors at prior=1 — Finding 3's named case closed.
  The other 11 published 1.0/`True` -> prior=0.
- **OCO is the mask's safety valve working**: pit lap 23, recovery 24 == out-lap -> 0 laps
  nulled; his post-stop ages are genuinely correct and stay numeric.
- **`_prior_usage_at` reading `driver_laps` (original) instead of `repaired` is provably
  inert**: the null block ends at the FIRST pit ≥ recovery, every boundary satisfies
  `boundary > pit_lap ≥ end of null block`, so the boundary row is never inside the nulled
  block — original and repaired hold the same value there. Confirmed on all 12 boundaries.

**But the nulling swallows the recoverable halves of STR/HAD/BOR** — see Finding N2 below.

## A (part 2) — 71-race sweep: no over-nulling anywhere; 70/71 byte-identical

Executed `repair_tyre_stints` + strict `assert_frame_equal` over every
`data/raw/2*/*/laps.parquet` (71 races):

- **70/71 byte-identical** (up from 69/71: Montréal 2023 is no longer touched). Only Miami
  2025 changes: 146 TyreLife nulled, 333 rebuilt, 0 filled-from-NaN, 54 Stint + 54 Compound
  cells, 12 boundaries, 18 drivers.
- **Independent mask false-positive hunt**: swept every driver in every race for the
  trigger shape (first lap `TyreLife` NaN, later numeric recovery). **The shape exists
  ONLY at Miami 2025** — no pit-lane starter, late joiner, or benign lap-1 NaN anywhere in
  the 71 races enters `_fabricated_age_mask`. Every one of the 146 nulled laps sits in the
  documented fabricated-restart block; none was correct (all understate true age by
  construction: the set predates the recovery lap's restart-at-1/2 count).
- Judgement per block: NOR/PIA/RUS/LEC (25–29, 5 each), VER/ALB (2), ANT/SAI (pit-in lap
  only, 1), TSU (3), HAM/ALO/LAW/BEA (5), BOR (8), GAS (9), HUL (13), STR (33), HAD (34).
  The STR/HAD/BOR long blocks are wrong-but-partially-recoverable — Finding N2.

## D — integration executed: the `touched`-mask hole has ZERO live instances; nulls fully propagate

The suspected defect — `touched = TyreLife_fixed.notna() | Stint_fixed.notna()`
(`laps_augment.py:138`) silently dropping nulled rows whose `TyreLife_fixed` is NaN — was
attacked directly:

- `_stint_corrections` emits **479 patch rows** for Miami; **0 rows have both
  `TyreLife` NaN and `Stint` NaN**. Every nulled lap keeps its published `Stint 1.0`
  (the feed fabricated the stint but did publish a stint id), so `Stint_fixed.notna()`
  fires and the NaN TyreLife is written through. **All 113 nulled raw laps present in
  featured 2025 are NaN after `augment_featured_laps` — 0 missed.** (The other 33 of 146
  are laps N04 drops.)
- End-to-end on featured 2025: 22760 rows -> 22760, order preserved, no `_fixed` leak,
  only `Time_s`/`TrackStatus` added, **second call byte-identical (idempotent)**,
  non-Miami rows identical on all three stint columns.
- Featured 2023: **0 changes on all three stint columns** (the Montréal fill is gone,
  see C below); featured 2024: 0 changes.
- The hole is still **structural** (see Finding N3): propagation relies on the accident
  that every null-only correction row carries a numeric `Stint`. A future block where the
  feed publishes `TyreLife` without `Stint` would null in the raw view and silently keep
  the fabricated number in featured. No shipped instance.

---

## Closure of the previous gate's nine findings (each independently re-verified)

| # | Prior finding | Status | Executed evidence |
|---|---|---|---|
| 1 | Fabricated pre-stop ages survive (NOR 29 = 5.0) | **CLOSED** | NOR 29 -> NaN; impossible-age rows (age < laps-since-pit-start − 5): 171 -> **0** on Miami; 146 fabricated ages nulled |
| 2 | GAS/HUL/BEA untouched | **CLOSED** | nulled 9 / 13 / 5 laps respectively; BEA covered despite zero pit records |
| 3 | Used sets hardcoded fresh | **CLOSED (live)** | ALO anchors at prior=1 (published 2.0, FreshTyre False); STR/HAD no longer rebuilt at all (nulled). Latent residue: N4 |
| 4 | Rule fires on `'None'` sentinel, writes sentinel over real compound | **CLOSED (live)** | Montréal 2023 byte-identical; `_is_real_compound` guards both the detection (`:127-133`) and the write (`:233`); pinned by 2 tests |
| 5 | `left_unknown` counts pre-repair | **CLOSED** | `unknown_after` = 590 == manual post-repair recount (pre-repair was 446); log message now matches the number |
| 6 | NaN-Driver row annihilated | **CLOSED** | `dropna=False` (`tyre_stint_repair.py:276`) + `test_a_nan_driver_row_is_not_annihilated` passes |
| 7 | Three-truth split across surfaces | **NOT FIXED — documented** (`:56-58`) | Accepted limit. Note it is now *sharper*: replay/backend raw readers show `TyreLife 5.0` where agent frames show NaN for the same lap |
| 8 | Same-compound-refit blindness | **NOT FIXED — documented** (`:51-55`) | Structural, errs toward missing; no shipped instance |
| 9 | First-stint test pins a no-op; dead `, why` | **CLOSED** | FIRING-path fixture exercises BOTH the nulling and the boundary rebuild (traced lap by lap); `, why` gone (test line 93) |

## NEW findings — what the rewrite itself carries

### N1 (MEDIUM, latent) — a half-applied boundary mutates the frame while the report swears nothing changed

`_rebuild_driver` writes the boundary compound onto the mislabelled laps
(`tyre_stint_repair.py:232-234`) BEFORE checking `stint_at_boundary`; when that stint is
NaN it `continue`s (`:236-238`) without setting `touched`. Executed (PROBE 1, synthetic
boundary row with real compound + NaN Stint): lap 3 `Compound HARD -> MEDIUM` **while
`report.changed_anything == False` and `boundaries_corrected == 0`**. Consequences: the
`RepairReport` contract ("a healthy race must come back byte-identical, and that is only
checkable if the count of touched rows is reported", `:85-88`) is broken — a frame can
come back mutated with an empty report — and `augment_featured_laps` skips
`_stint_corrections` for that race (`laps_augment.py:206-207`), so the mutation lives in
the raw view only. **Zero live instances** (all 12 firing boundaries in 71 races have a
real Stint). Fix is a two-line reorder: move the Compound write below the stint guard.

### N2 (MEDIUM, judgement) — the nulling swallows the RECOVERABLE post-stop halves of STR/HAD/BOR: 63 featured test-season laps go dark to race end

STR (pit 20), HAD (pit 22), BOR (pit 19) pitted while the feed was dark, so
`later_stops = [pits >= recovery]` is empty and `end_lap` becomes the last lap
(`tyre_stint_repair.py:190-191`): the null block runs from lap 24 to race end — 33/34/8
raw laps, **63 of the 113 featured nulled laps (HAD 30, STR 29, BOR 4)**. Two things are
wrong with how this lands:

1. **These laps are NOT the race-start set.** The mask's docstring says the block "is
   still the race-start set, whose age was never published" (`:168-170`) — true for
   GAS/HUL/BEA/NOR-style blocks, FALSE for the three pre-recovery-stop drivers, where the
   block is the SECOND set fitted at out-lap 20/21/23. Their published ages are only
   understated by `recovery − out_lap` (1-4 laps), not by ~24.
2. **They are recoverable under the exact assumption the module already trusts.**
   `_prior_usage_at` trusts "the feed publishes the set's own starting age on the lap the
   feed thinks the stint began" (`:197-204`). The same assumption applied at the recovery
   lap rebuilds these blocks to the same fidelity: `age(L) = published(L) + (recovery −
   out_lap)`. The module refuses because the pre-stop compound is a sentinel, so a
   penalty cannot be ruled out from the columns it reads — defensible doctrine, but the
   previous version repaired these three drivers to within prior-usage error, and the
   rewrite replaces 1-4-lap-wrong integers with NaN for up to 30 featured laps of N26's
   test-season input per driver. Neither the module docstring's known-limits section nor
   the report distinguishes this trade from the flagship fabrication. **Decide it
   explicitly: document it as an accepted limit, or anchor-rebuild when later evidence
   corroborates the change.** (Montréal TSU is the same choice at 35 laps: the prior gate
   proved that stop WAS a real tyre change via pit-transit 27.5 s, but the anchor there is
   NaN, so any fill WOULD invent — leaving it NaN is doctrine-consistent and I do not
   contest it.)

### N3 (MEDIUM, latent) — the featured patch drops any null-only correction whose Stint is also NaN

`_apply_stint_corrections` gates on `touched = TyreLife_fixed.notna() | Stint_fixed.notna()`
(`laps_augment.py:138`). A nulled lap has `TyreLife_fixed` NaN by construction, so
propagation rides ENTIRELY on the accident that the feed published a numeric `Stint` on
every fabricated Miami row. Executed (PROBE 2, synthetic feed that published `TyreLife`
without `Stint`): the raw view nulls 2 laps, `_stint_corrections` carries both rows, and
the featured frame **keeps the fabricated 1.0/2.0** — the exact invisible-wrong-integer
class this repair exists to kill, plus a silent raw/featured divergence. **Zero live
instances** (0 of 479 Miami patch rows have both NaN; all 113 featured-present nulled
laps became NaN — executed). Fix: mark patch membership explicitly (merge indicator)
instead of inferring it from value non-nullness.

### N4 (MEDIUM, latent) — a NaN anchor silently invents a fresh set, and the sentinel-skip can carry the rebuild across a dark zone

`_prior_usage_at` returns **0.0** when the boundary row's `TyreLife` is NaN (`:207-209`).
Executed (PROBE 4): a boundary with real compound + real Stint + NaN TyreLife fires the
rebuild and FILLS the NaN laps at 1, 2, 3 — fresh-set ages for a set whose published age
was never seen, contradicting "It never invents a tyre age" (`:35`). Related:
`_first_compound_change_after` skips sentinels (`:131-133`), so the boundary search can
cross a feed-dark block, and the rebuild then writes Compound/Stint/ages onto dark rows
during which a pit record could itself have been lost (the BEA precedent refutes
"PitInTime is always intact"). **Zero live instances** — Miami's 12 boundaries all have
real published anchors (executed table, values 1.0/2.0), and only Miami fires anywhere
in 71 races. Fix: return `None` for an unknown anchor and refuse (null instead of
rebuild).

### N5 (LOW) — the module claims a mechanism it does not have: `FreshTyre` is never read

`tyre_stint_repair.py:38`: "`FreshTyre` is therefore read, not assumed". Executed: the
string `FreshTyre` occurs exactly once in the module — in that docstring line. No code
reads the column; the rebuild anchors on published `TyreLife` (which is the better
signal, since the root-cause gate proved `FreshTyre` is fabricated-uniform-`True` inside
the gap). `tests/agents/test_tyre_stint_repair.py:234` repeats the claim, and its fixture
has no `FreshTyre` column at all. The behaviour is right; the stated mechanism is false —
the project's recorded scar class ("a comment naming the wrong MECHANISM is worse than
none"). Same family: the `_fabricated_age_mask` docstring's "race-start set" rationale is
wrong for the pre-recovery-stop drivers (N2.1).

## E — the 16 tests, attacked

- 16/16 pass (executed); ruff clean on all three files.
- `test_a_stop_made_while_the_feed_is_dark_is_left_unknown_rather_than_guessed`: traced —
  the fixture reaches the sentinel guard (compound at the pit lap is `'nan'`, so the
  boundary is refused) AND the mask (2 laps nulled), and its shape matches the real
  BOR/STR/HAD anatomy (pit during darkness, recovery not an out-lap; executed per-driver
  table). Genuine.
- `test_a_used_set_keeps_its_prior_usage`: traced — prior=3 comes from published 4.0 at
  the boundary; asserts 4.0/5.0 on the firing path. Genuine on behaviour; its docstring
  misstates the mechanism (N5).
- `test_a_missing_first_stint_stays_unknown_on_the_FIRING_path`: traced — the fixture
  exercises nulling (laps 3-4) and boundary rebuild (laps 5-6) in one pass, and asserts
  `report.changed_anything` in-test, so it can never regress to a no-op. Finding 9's scar
  is genuinely closed.
- Not pinned anywhere: N1's partial-write shape, N3's NaN-Stint null propagation, N4's
  NaN-anchor refusal. The three latent holes are exactly the three untested paths.

## F — model-input impact, quantified

- **Featured 2025** (the models' test season): **113 laps lose a numeric `TyreLife`**
  (fabricated values -> NaN), **293 laps change numeric value** (mean +3.74, range
  −7..+7), 0 filled from NaN. All Miami; every other GP byte-identical on the three stint
  columns. Featured **2023: 0 changes** (the previous version's 35 Montréal fills are
  gone), **2024: 0 changes**.
- **Published-metric paths**: pace/tire/overtake/calibration/projection read the parquets
  directly (`pace_holdout.py:106`, `tire_holdout.py:171`, `calibration.py:160/233/283`,
  `projection.py:296` — executed grep), so the 86.5%/59.1% projection, pace MAE and the
  calibration numbers are unmoved by construction. The two #782 golden failures reproduce
  identically (`KeyError: ['AirTemp','TrackTemp','Humidity','Rainfall']`, executed) —
  pre-existing, unrelated.
- **Downstream augment consumers**: 64/64 tests pass (pace-orchestrator hardening, MC
  state helpers, tire cumulative deg, tire MC determinism, envelope — executed, 33 s).
  N26's Miami windows now see NaN where they saw fabricated integers; that is the
  intended behaviour change.

## What I tried to break and could not

- **Over-nulling.** Swept all 71 races for the mask's trigger shape (first-lap `TyreLife`
  NaN with later numeric recovery): it exists ONLY at Miami 2025. No pit-lane starter,
  late joiner, or benign lap-1 NaN anywhere enters `_fabricated_age_mask`. Every nulled
  lap's published age was provably understated. OCO (pit 23, recovery 24 == out-lap) is
  correctly spared — the safety valve works on the one driver who needed it.
- **Byte-identity.** Strict `assert_frame_equal` over all 71 races: **70/71 identical**
  (Montréal 2023 joins the untouched set; only Miami changes).
- **The `touched` mask on shipped data.** 0/479 patch rows invisible; 113/113
  featured-present nulled laps arrive as NaN; no `_fixed` leak; row count/order
  preserved; second `augment_featured_laps` call byte-identical; raw `repair_tyre_stints`
  idempotent with a clean second-pass report.
- **`_prior_usage_at` reading the NaN its own pass wrote.** Impossible by construction
  (every boundary > its pit lap >= the null block's end) and confirmed on all 12 real
  boundaries — all read real published anchors. Passing `driver_laps` (original) instead
  of `repaired` is inert today, and it is arguably the CORRECT choice under
  multi-boundary mutation, so I do not flag it.
- **The anchor-accrual alternative.** If the late-created stint record counted the laps
  accrued since the actual fitting, `published − 1` would over-count prior usage. The 11
  fresh-set Miami boundaries publish exactly 1.0 despite 4-8 laps accrued since their
  out-laps — the record demonstrably counts from the boundary lap, so the subtraction is
  right on all data that fires.
- **A sentinel written over a real compound.** The write is guarded by
  `_is_real_compound` (`:233`); pinned by test; no path writes a sentinel.
- **Downstream pinned values.** 64/64 augment-consuming tests green; golden failures
  unchanged.

## Fix list (by value, then risk)

1. **N1**: move the Compound write below the `stint_at_boundary` guard (two-line reorder)
   so a refused boundary refuses atomically and the report stays truthful.
2. **N3**: key `touched` in `_apply_stint_corrections` on patch MEMBERSHIP (merge
   indicator) rather than value non-nullness.
3. **N4**: `_prior_usage_at` -> return `None` on a NaN anchor and skip/null instead of
   anchoring at fresh.
4. **N5**: fix the two `FreshTyre` docstring claims and the mask docstring's
   "race-start set" rationale (N2.1). Pure prose, zero behaviour.
5. **N2**: decide the STR/HAD/BOR trade explicitly — document the post-stop null as a
   known limit in the module docstring, or (follow-up) rebuild dark-stop blocks when
   later evidence corroborates the tyre change.

## Verdict

**SHIP.** All three prior blockers (Findings 1-3) are genuinely closed with executed
evidence, Findings 4-6 and 9 are closed, and 7-8 are honestly documented limits. The
rewrite introduced no defect with a live instance in the 71 shipped races: N1, N3 and N4
are latent (synthetic-only), N5 is prose. Recommend landing fixes 1-4 in the same PR (all
small, none changes shipped-data behaviour) and settling N2 as a documented decision
rather than silence.
