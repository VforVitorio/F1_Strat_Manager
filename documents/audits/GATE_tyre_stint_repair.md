# GATE — tyre-stint repair (#790): adversarial correctness audit

**Date:** 2026-08-02 · **Auditor:** adversarial correctness gate (read-only except this file)
**Scope:** `src/f1_strat_manager/tyre_stint_repair.py`, its wiring in
`src/f1_strat_manager/laps_augment.py::augment_featured_laps`, and
`tests/agents/test_tyre_stint_repair.py`, on branch `dev` (uncommitted).

Findings are appended incrementally as they are confirmed with executed evidence.

## Checklist

- [x] A. Repair touches only broken data — **VERIFIED** (69/71 byte-identical, independently re-measured)
- [x] B. Two-condition boundary rule — **INCOMPLETE by construction** (Findings 2, 4, 8): blind to
      same-compound refits (28.6% of real stops), fires on data-loss sentinels, misses GAS/HUL/BEA
- [x] C. Rebuilt ages — **RIGHT for fresh sets, WRONG for used sets** (Finding 3: ALO/HAD/STR + TSU)
- [x] D. No invented first-stint age — **HOLDS** (but the fabricated recovery block survives, Finding 1)
- [x] E. Integration — **HOLDS** on shipped data (latent NaN-Driver annihilation, Finding 6)
- [x] F. Tests — pass 10/10, but the first-stint test pins a no-op (Finding 9)
- [x] G. Published metrics unmoved by construction except decision_modes; three-truth split (Finding 7)

---

## Finding 1 (HIGH) — the flagship defect of the investigation is NOT fixed: 71+ Miami laps keep fabricated ages after the repair

Executed: `repair_tyre_stints` on `data/raw/2025/Miami_Gardens/laps.parquet`, then scanned every
driver for numeric `TyreLife` before their first pit stop with `TyreLife < LapNumber - 5`.

The repair only rewrites laps **after** a pit stop (`tyre_stint_repair.py:147-149`,
`in_new_stint = lap_numbers >= out_lap`). The laps between feed recovery and the pit stop — the
block where `TyreLife` restarted at 1 mid-stint, error #1 of `GATE_tyrelife_nan_rootcause.md`
Finding 4 — are left exactly as the feed fabricated them:

| Driver | Surviving fabricated laps | TyreLife kept | Truth |
|---|---|---|---|
| NOR | 25–29 | 1–5 | ~25–29 (+offset) |
| HUL (never repaired) | 24–36 | 1–13 | ~24–36 |
| GAS (never repaired) | 24–32 | 1–9 | ~24–32 |
| BEA (never repaired) | 24–28 | 1–5 | ~24–28 |
| + 11 more repaired drivers | 1–9 laps each | 1–7 | lap number |

**66 laps** across 14 drivers with a first pit stop, **plus BEA's 5** (he has zero
`PitInTime`/`PitOutTime` records in the whole race, so the scan skipped him — total ≥71 laps.
Executed evidence for the flagship row: NOR lap 29 still reads `TyreLife 5.0` after repair (truth
~29) — the exact value the investigation used as its headline. The module docstring's own standard
("a wrong integer is worse than a NaN precisely because the NaN is visible",
`tyre_stint_repair.py:32`) condemns the module's own output: these laps are *provably* wrong (a
numeric `TyreLife < laps since the last pit/race start` is impossible) and the repair leaves the
plausible integers in place instead of nulling them to the honest NaN it uses everywhere else.
These are 2025-test-season racing laps that feed N26/N15/N16.

## Finding 2 (HIGH) — coverage hole: GAS, HUL and BEA carry the same corruption and are not touched

Executed per-driver boundary scan on Miami:

- `GAS: pits=[32]`, metadata recovers at lap 24 → his lap-32 stop's boundary lands correctly at 33,
  so `find_misplaced_boundaries` returns `[]` — yet laps 24–32 are stint-1 fabricated ages 1–9.
- `HUL: pits=[36]` → same shape, laps 24–36 fabricated ages 1–13 (the worst single block in the race).
- `BEA: pits=[]` — **zero pit records in the entire race** while `Stint 1.0 / HARD / TyreLife 1.0`
  appears at lap 24 (he retired on lap 28). Either he pitted and `PitInTime` was ALSO lost —
  refuting the module's premise that "`PitInTime` is the one column the broken feed leaves intact"
  (`tyre_stint_repair.py:84-86`) — or he never pitted and lap 24's `TyreLife 1.0` on a 24-lap-old
  set is pure fabrication. Both readings leave wrong data shipped.

The repair claims (module docstring, integration comment `laps_augment.py:203-204`) to fix the
Miami corruption; it fixes 15 of 18 corrupted cars' post-stop halves and 0 of 17 pre-stop halves.

## Finding 3 (HIGH) — claim C is false for used sets: the rebuild contradicts the FastF1 convention it cites

Executed: swept every stint transition in all 2024 races (healthy season) and all 2025 races
except Miami, split by `FreshTyre`:

- Fresh set, first lap of stint: `TyreLife == 1.0` in **617/617** (2024) and **539/539** (2025).
- **Used set, first lap of stint: 2–16**, 209 cases in 2024 alone (e.g. Austin ALO L27 HARD → 2.0,
  Austin OCO L52 SOFT → 4.0). FastF1's `TyreLife` counts prior-session laps on the set.

The rebuild hardcodes `TyreLife = LapNumber - out_lap + 1` (`tyre_stint_repair.py:165-167`), i.e.
assumes every fitted set is fresh. Three of the 15 repaired Miami drivers fitted sets with
`FreshTyre == False` at their (real-data) boundary lap: **ALO** (L33, MEDIUM, feed said 2.0),
**HAD** (L24), **STR** (L24, feed said 2.0). For these the repaired value understates true tyre
age by the set's prior usage — the exact "unknown offset" the module refuses to invent for stint 1
(`tyre_stint_repair.py:28-32`) is silently invented as zero for stints 2+. The comment at
`tyre_stint_repair.py:144-146` ("the convention FastF1 itself uses on healthy races (verified:
Miami lap 33...)") verified the convention on ONE fresh-set example from the *broken* race; on
healthy races the convention is conditional on `FreshTyre`, which the rebuild never reads.

## Finding 4 (MEDIUM) — Montréal 2023: the rule fires on the `'None'` data-loss sentinel, not on a compound change, and destroys a real cell

Executed before/after on TSU (the only 2023 change):

- The "compound change" that satisfies condition 1 is `'HARD' → 'None'` — the literal string
  `'None'` is a data-loss sentinel, not a compound. The rule's justification (compound change =
  evidence a new set was fitted) is not what fired; absence-of-data fired it.
- Lap 35 before: `Stint 2.0, Compound 'HARD', TyreLife 34.0`. After: `Stint 3.0, Compound 'None',
  TyreLife 1.0` — the repair **overwrote a real compound string with the sentinel** as a side
  effect of `tyre_stint_repair.py:152-154`. Downstream consumers that check `Compound.notna()`
  see `'None'` as a valid compound.
- In this instance the stop WAS a real tyre change (executed evidence: pit transit 27.5 s vs field
  median ~24 s — too short for a 10 s stop-and-go, too long for a drive-through; lap times drop
  from ~78.4 s to ~77.2 s on the new set), so the 36 filled ages count laps-on-car correctly. But
  stint 3 has `FreshTyre == False` on every lap, so Finding 3's used-set understatement applies
  here too: the filled 1–36 are a lower bound, not the FastF1-convention age.
- Had the stop been a served penalty followed by feed death — indistinguishable to the rule, since
  `'None' != anything` is always True — the repair would have invented a fresh set from a penalty.
  The rule survives on this data by luck, not by construction.
- Surface nuance: featured 2023 does not carry TSU lap 35 (N04 drops pit-out laps), so the
  `'HARD' -> 'None'` cell destruction stays in the in-memory raw view; what reaches featured is the
  35 filled ages (verified: 35 patched rows, all filled-from-NaN).
- Generalisation (executed): the Miami gap's `Compound` is the literal string `'nan'` (437/438
  rows, `str` type), NOT `np.nan`. So the boundary detections at Miami laps 24 (BOR, STR, HAD) also
  fired on a sentinel-to-real transition (`'nan' -> 'HARD'`), the same absence-of-data mechanism.
  They coincide with real stops here, but none of the rule's firings in shipped data was a true
  compound-A-to-compound-B change at the boundary row's own stint; and had the extractor stored
  real NaN instead of the `'nan'` string, `NaN != NaN` would have made every in-block comparison
  "a change", the boundary would have collapsed to `pit_lap + 1`, and BOR/STR/HAD would silently
  NOT have been repaired. The rule's coverage depends on an accident of stringification.

## Finding 5 (MEDIUM) — the shipped log line is factually false: `left_unknown` counts rows the repair itself just filled

`repair_tyre_stints` computes `still_unknown` BEFORE repairing (`tyre_stint_repair.py:195-198`)
and logs it as "%d lap(s) keep an honest unknown age" (`tyre_stint_repair.py:208`). Executed on
Montréal 2023: `report.left_unknown == 35` while the repair filled **all 35** of those NaN
`TyreLife` rows with numbers in the same call. Every operator reading the log is told the exact
opposite of what happened.

## Finding 6 (MEDIUM, latent) — a NaN `Driver` row is silently annihilated, all columns

`repair_tyre_stints` rebuilds the frame as `pd.concat(groupby("Driver"))` then
`.reindex(laps.index)` (`tyre_stint_repair.py:200-203`). `groupby` drops NaN-key rows; `reindex`
re-inserts them as all-NaN. Executed on a synthetic frame: a row with `Driver=None` came back with
`LapNumber`, `Stint`, `Compound`, `TyreLife` AND `Position` all NaN — the repair destroyed columns
it has no business touching. Swept all 71 shipped raw parquets: none carries a NaN `Driver` today,
so this is latent — but the function runs on every load of every future season's download, and one
malformed row would be silently erased rather than skipped.

## Finding 7 (MEDIUM) — the repair creates a three-truth split: the same Miami lap now has different tyre data depending on which surface asks

The repair exists ONLY inside `augment_featured_laps`. Executed inventory of readers that bypass it:

- **Raw, unrepaired:** `src/simulation/replay_engine.py:73` (loads `laps.parquet` for
  `RaceStateManager`, which serves `TyreLife`/`Compound`/`Stint` into `lap_state` —
  `race_state_manager.py:4-8`), `src/telemetry/backend/api/v1/endpoints/strategy.py:328`,
  `src/strategy/eval/projection.py:292`, `pit_holdout.py:51`, `stint_lengths.py:203`.
- **Featured-direct, unrepaired:** `src/strategy/eval/pace_holdout.py:95-106`,
  `tire_holdout.py:171`, `calibration.py:160/233/283` — the golden-metric layer reads the parquet
  without `augment_featured_laps`, so it measures the models on UNREPAIRED inputs while the
  runtime agents now consume REPAIRED ones.
- **Augmented, repaired:** the CLI/arcade/backend agent frames, and `decision_modes.py:473/508`.

Concrete incoherence, measured on both frames: during a Miami 2025 replay, `lap_state` for NOR lap
31 says `Stint 1 / MEDIUM / TyreLife 7` (raw) while the tire agent's featured window for the same
lap says `Stint 2 / HARD / TyreLife 2` (repaired). The two disagree about which stint the car is in
inside a single process.

## Finding 8 (MEDIUM) — claim B is incomplete by construction: the rule is blind to 28.6% of real stops

Executed over all 71 races: 2594 stint transitions, of which **743 (28.6%) fit the same compound
again** (e.g. HARD -> HARD two-stoppers). For those, `_first_compound_change_after` sees no change
at the stop — a misplaced boundary there is invisible to the rule in principle, no matter how
corrupt. Miami 2025 happened to have compound changes at every corrupted stop, so nothing shipped
is missed *today*; the incompleteness is structural, not observed. Direction: misses defects
(conservative), never rewrites correct records via this path.

Also executed: the only stint transitions in all 71 races with no pit entry within the 2 prior
laps are the 12 Miami corruption boundaries themselves — so the red-flag/no-pit-entry false-positive
shape (a compound change with no `PitInTime`, preceded by an unrelated earlier stop, which the rule
would misread as a misplaced boundary and rewrite correct laps) has **no live instance in shipped
data**. It remains reachable for future seasons: nothing in the rule distinguishes "compound
changed with no pit record at all" from "boundary snapped late".

## Finding 9 (LOW) — the first-stint test pins a no-op, not the invariant; line 93's `, why` is dead

- `test_a_missing_first_stint_stays_unknown` (`tests/agents/test_tyre_stint_repair.py:140-156`):
  executed `find_misplaced_boundaries` on its fixture -> `[]`, and `repair_tyre_stints` returns the
  frame **completely untouched** (`.equals` True, 0 boundaries, 0 ages). The assertion then checks
  a row the function never had any code path toward — the test passes for the wrong reason (the
  project's recorded scar class). The invariant itself DOES hold on the firing path: on a
  Miami-shaped fixture where a repair fires next to an unknown first stint, laps 1-2 stay NaN
  (executed) — but no shipped test proves that. The assertion at :153-155 (`x != x or pd.isna(x)`)
  is a doubly-redundant NaN check, not a tautology.
- `tests/agents/test_tyre_stint_repair.py:93`: `pd.testing.assert_frame_equal(repaired, frame), why`
  builds a tuple; the equality check still executes and raises on mismatch, so the test DOES
  assert — but `why` is dead code that looks like an assert message and is not.
- All 10 shipped tests pass (executed: `pytest tests/agents/test_tyre_stint_repair.py` -> 10 passed).

## Verified intact (what I tried to break and could not — so far)

- **Claim A (independently re-measured):** swept all 71 raw parquets through `repair_tyre_stints`
  with strict `assert_frame_equal` (dtype, index, column type, exact values) plus
  `hash_pandas_object` as a second witness: **69/71 byte-identical**; only Miami 2025 (15 drivers,
  416 ages, 62 Stint/Compound cells) and Montréal 2023 (TSU, 36 ages, 1 Stint + 1 Compound cell)
  change.
- **Claim E (integration):** end-to-end `augment_featured_laps` on featured 2025: row count 22760
  preserved, row order preserved, no `_fixed` columns leak, only `Time_s`/`TrackStatus` added,
  `Compound`/`Stint`/`TyreLife` dtypes unchanged, second call is a byte-identical no-op,
  non-Miami rows identical on all three stint columns, Miami rows changed on NOTHING outside the
  three stint columns (361/857 featured Miami rows patched). `(GP_Name, Driver, LapNumber)` is
  unique in featured and in every raw parquet (no merge duplication possible on shipped data).
- **Claim C, fresh-set half:** the out-lap-is-TyreLife-1 convention verified independently on
  every healthy stint transition: 617/617 (2024) and 539/539 (2025 sans Miami) fresh-set stints
  start at exactly 1.0; NOR's rebuilt Miami ages match the pit timeline arithmetic exactly.
- **Claim D:** no path fills a truly-NaN first stint — verified both on the shipped fixture and on
  a firing-path fixture (`in_new_stint`/`wrongly_old_stint` are both bounded below by `out_lap`,
  so pre-pit laps are unreachable). Miami laps 1–24 and Spa/Melbourne NaN blocks stay NaN.
- **Ruff:** all three files pass `uvx ruff check` clean.
- **Multi-boundary drivers:** LAW (pits 28 and 36) — only the first boundary is misplaced, the
  second stop is correctly skipped; the loop's mutation of `repaired` cannot corrupt a later
  boundary because the intervening-pit condition forces disjoint lap ranges.
- **Featured impact quantified:** 2025: 361 rows moved, TyreLife delta mean +3.30 laps (range
  −7..+7), Miami only; 2023: 35 rows, all filled from NaN, Montréal only.

## Numbered fix list (by value, then risk)

1. **Null the fabricated pre-stop ages (fixes Findings 1 and 2 in one move).** For any driver whose
   metadata is absent at race start (`Stint` NaN on lap 1) and recovers mid-race without a pit
   entry at the recovery lap, the recovered "stint 1" block up to the first pit stop provably
   belongs to the race-start set with unknown age: set its `TyreLife` to NaN (and optionally flag
   `Stint`). That is the module's own stated doctrine applied to its own blind spot, it covers
   GAS/HUL/BEA (who need no boundary correction), and it converts ~71 invisible wrong integers
   into visible unknowns.
2. **Respect `FreshTyre` in the rebuild (Finding 3).** If the boundary row has `FreshTyre False`,
   either leave the rebuilt ages NaN (honest unknown, doctrine-consistent) or anchor the count at
   the boundary row's own published `TyreLife` instead of 1. Three of 15 repaired Miami drivers
   plus Montréal TSU are affected today.
3. **Refuse to fire on data-loss sentinels (Finding 4).** Treat `'None'`/`'nan'`/NaN compounds as
   "unknown", not as "different": require the boundary compound to be a real compound AND the
   pre-pit compound to be known before rewriting `Compound`; never write a sentinel string over a
   real value. (Montréal would then fill ages only if some other evidence confirms the tyre
   change, or not at all.)
4. **Fix the log/report semantics (Finding 5).** Recompute `left_unknown` AFTER the rebuild, or
   rename it (`unknown_before_repair`) and log both.
5. **Guard the groupby against NaN drivers (Finding 6).** `groupby("Driver", dropna=False)` or an
   upfront `laps["Driver"].notna()` split that passes NaN-driver rows through untouched.
6. **Decide the consistency story (Finding 7).** Either move the repair below every reader (e.g.
   into the raw load path used by `replay_engine` / backend `strategy.py:328` as well), or document
   explicitly that raw-fed surfaces (replay, arcade HUD, backend /strategy, projection) show feed
   values while agent frames show repaired ones. Today the split is silent.
7. **Make the first-stint test exercise the firing path (Finding 9)** — use a Miami-shaped fixture
   whose repair fires, assert laps before the recovery stay NaN; drop the dead `, why` on line 93.

## Test-suite evidence

- `tests/agents/test_tyre_stint_repair.py`: 10/10 pass (executed).
- Augment-consuming non-eval tests (`tests/audit/test_pace_orchestrator_hardening.py`,
  `tests/mc/test_mc_state_helpers.py`, `tests/agents/test_tire_cumulative_deg.py`,
  `tests/audit/test_tire_mc_determinism.py`, `tests/inference/test_envelope.py`):
  **64/64 pass** with the repair active (executed, 166 s) — no pinned value broke.
- `tests/eval/`: results appended below when the background run completes.

## Published-metric exposure (claim G, structural)

- **Projection 86.5%/59.1%** (`projection.py:292`) and **pit P50 MAE** (`pit_holdout.py:51`) read
  the RAW parquets from disk; the repair never writes to disk -> unmoved by construction.
- **Pace MAE 0.4104 / overtake AUC / tire MAE / calibration** (`pace_holdout.py:95-106`,
  `tire_holdout.py:171`, `calibration.py:160/233/283`) read the FEATURED parquet directly without
  `augment_featured_laps` -> also unmoved — but note this is the same bypass CLAUDE.md calls a bug
  ("every consumer calls augment_featured_laps"), and it now means the golden metrics certify the
  models on data the runtime agents no longer see (Finding 7).
- **decision_modes** (`decision_modes.py:473/508`) DOES consume the repaired frame; its numbers
  can legitimately move for Miami 2025 / Montréal 2023.

## What I tried to break and could not

- **A false positive on a healthy race.** Strict frame equality + independent hashing across all
  71 raw parquets: 69 come back bit-identical; no correct record is rewritten anywhere in shipped
  data. The Monaco 2025 RUS stop-and-go and the penalty-only shape are correctly skipped (also
  pinned by the shipped tests, which do assert — `assert_frame_equal` raises despite the odd
  `, why` tuple on line 93).
- **The red-flag / no-pit-entry false-positive shape in shipped data.** The only stint transitions
  without a nearby pit entry in all 71 races are Miami's 12 corruption boundaries themselves. The
  landmine is real for future data but has no live instance.
- **Integration leaks.** No `_fixed` columns, no row count/order change, no dtype drift, no
  duplicate-key blowup (join keys unique in featured and all raws), second call a no-op,
  non-Miami featured rows bit-identical on the stint columns, Miami rows untouched outside them.
- **First-stint invention via the `Compound`/`Stint` overwrite.** Both masks are bounded below by
  `out_lap`; a firing-path fixture confirms pre-pit laps are unreachable. Miami laps 1–24,
  Spa 27–44, Melbourne 40–44 NaN blocks all stay NaN.
- **The rebuilt arithmetic on fresh sets.** NOR's repaired stint 2 (out-lap 30 -> TyreLife 1,
  lap 36 -> 7) matches the pit timeline exactly, and the out-lap==1 convention held on all 1156
  healthy fresh-set stints of 2024-2025.
- **Breaking downstream consumers via `Stint` relabelling.** `tire_predictor.py:950`'s
  `Stint.max()` and the tire agent's `['Year','GP_Name','DriverNumber','Stint']` windows still see
  contiguous, single-set blocks after repair; 64/64 augment-consuming tests pass.

## Verdict

**Not safe to ship as-is.** Claim A and the integration (E) genuinely hold — the change is
surgically scoped and touches exactly the two broken races. But the repair's central promise —
"the model no longer reads 5 where the truth is 29" — is only half-delivered: the fabricated
pre-stop ages (the investigation's own flagship defect) survive on ≥71 Miami test-season laps
(Finding 1), three drivers with the identical corruption are skipped entirely (Finding 2), and the
rebuilt ages silently assume every fitted set is fresh, contradicting the measured FastF1
convention for the 209+ used-set stints per season and understating age for 3 of the 15 repaired
drivers (Finding 3). Fix list items 1–3 are the blockers; 4–7 can follow.

