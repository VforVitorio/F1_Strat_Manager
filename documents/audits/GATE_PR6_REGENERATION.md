# GATE — PR 6, featured-artefact regeneration (adversarial audit)

Branch under audit: `feat/regenerate-featured-artefacts`.
Auditor: adversarial gate, 2026-08-06. Primary target: `documents/audits/PR6_REGENERATION_LOG.md`.
Producer: `scripts/rebuild_featured_laps.py`. Backups: `data/_backup_pr6_featured/`.

Rule of this file: findings are appended AS CONFIRMED, with executed evidence. Nothing is
buffered to the end. Severity HIGH/MEDIUM/LOW assigned at confirmation time.

## Checklist (claims to verify or refute)

- [x] A. VERIFIED, exact (V1) — comparison sound, row order proven identical; but see F1/F2
      on the script's own gate.
- [x] B. VERIFIED from the backups, exact counts reproduce (V1); per-year file was right.
- [x] C. REFUTED on provenance (F5): 53 is NOT the original schema; consequences verified nil.
- [x] D. VERIFIED exhaustively (V1): 760 Vegas msp cells are the ONLY change, all flagged,
      zero unflagged holes anywhere. Measured N06 effect in F8.
- [x] E. VERIFIED independently (V2): n=68 / MAE 1.22 / p95 3.40 / 232.83 all reproduce;
      239.14 not reproducible under six candidate readings.
- [x] F. VERIFIED (V1 + V3): surviving rows exact; no 'Spain' key survives in any labeled
      artefact or cluster file; the tiredeg Barcelona block is the known deferred carrier.
      BUT the measured MC tables were missed → F4, and the prose/constants → F6.
- [x] G. First half VERIFIED exact (V4: 70/1,768/86.3 and 24/552/86.1). Second half is the
      headline defect → F4 (tables still counted off 71 races as committed).
- [x] H. VERIFIED by inspection (V5): all four featured parquets are exact-path entries in
      `_DEFAULT_MODEL_PATTERNS`; `_build_allow_patterns` only appends raw-race globs.
- [x] I. The two pace tests were STRENGTHENED, not weakened (V6); the edited partial-weather
      test still exercises the guard. The silently WEAKENED test is one nobody listed → F3.
- [x] J. Hunted. One dormant instance found in NEW code (F7); no live second instance —
      what was tried is in the could-not-break section.
- [x] K. VERIFIED (V7): alias is inside `if year == _HOLDOUT_SEASON`, cannot touch 2023/24;
      weather frame's only consumer is `add_weather_features`; independent re-run of the
      alignment gate: 0 mismatches on 66,924 rows, Miami AirTemp NaN 0 in all three seasons.
- [x] L. VERIFIED (V8): all 12 lifted functions AST-identical to `.nb_py/N04`, both
      constants value-equal. `impute_circuit_speed`/`_finalize_frame` audited as new → F7.

## Findings

### V1 — Claims A, B and D VERIFIED, with EXACT equality (stronger than the log's tolerance) [verification, not a defect]

Executed: own diff script (scratchpad `gate_diff.py`), exact `==` per cell (no `isclose`),
NaN-aware, plus positional key-tuple order check on `[Year, GP_Name, DriverNumber,
LapNumber, Stint]`. Backup SHA-256s match the log's recorded values (all four).

- **2023**: backup minus Spain (20,908 rows) vs regenerated: positional row order IDENTICAL,
  **zero differing cells on all 48 shared columns, exact**. Claim F's surviving-row check
  independently reproduced.
- **2024**: 23,256/23,256, order identical, zero differing cells exact.
- **2025**: 22,760/22,760, order identical, exactly ONE column differs:
  `mean_sector_speed`, 760 cells (NaN → 232.8269647, all Las Vegas, all flagged).
- **Claim B reproduced exactly from the backups**: backup combined 2025 slice vs backup
  2025 per-year file differs on `GP_Name` 857 (Miami alias), `Cluster` 6,892,
  `mean_sector_speed` 21,903, `lap_time_vs_cluster_mean` 22,760 — the log's four counts,
  key-aligned, exact. Which was right: the per-year file (season-true, reproducible from
  raw under N03's rule, and the quantity #797/N06's holdout was measured against — runbook
  A1/A2). The rebuild resolves the disagreement in the per-year file's favour by
  construction (combined = concat; verified: new combined == concat of the three new
  per-year files, zero differing cells).
- **Claim D exhaustive**: across all four files, the ONLY changed cells vs backup are the
  760 Vegas msp cells. Flag column: 760 True, all in Vegas 2025, zero elsewhere;
  **zero unflagged missing `mean_sector_speed` remains in any file**.
- **Row-order soundness of the log's comparison**: `_compare` aligns by position, but the
  key columns are among the compared columns, so a zero result implies identical order —
  and my explicit key-tuple check confirms it independently. The comparison is sound.
- **Determinism**: `--verify --drop-spain`-less re-run with `--impute-circuit-speed`
  (writes nothing) reproduces every on-disk file with zero changed cells, exit 0 — the
  shipped artefacts are exactly what the script produces today.

### F1 (LOW) — the script's own "reproduces to 1e-6" print is looser than it claims: `np.isclose` keeps its DEFAULT `rtol=1e-5`

`scripts/rebuild_featured_laps.py:753-758` passes `atol=1e-6` but not `rtol`, so the
effective tolerance on a 90 s lap time is ~9e-4 and on a 230 km/h speed ~2.3e-3 — three
orders looser than the printed "reproduces to 1e-6". The runbook's appendix snippet
(GATE_801_ARTEFACTS.md:629) explicitly sets `rtol=0`; the script dropped that. Latent
today (my exact diff shows the true deviation is 0.0 everywhere), but a future rerun after
an environment bump could drift by up to rtol·|value| per cell and still print "none".
Fix: pass `rtol=0`.

### F2 (MEDIUM) — the script's acceptance gate does not GATE: write mode ignores the diff result, and `--drop-spain` forces exit 0 in verify mode

Three code paths, `scripts/rebuild_featured_laps.py`:

- `main` line 832-841: in WRITE mode the script writes the artefacts and returns 0
  **regardless of `reproduces`** — the value-diff is computed, printed, and then not
  consulted. The runbook (§3: "any failure stops the line") prescribes a stopping gate;
  the implementation relies on the operator reading the output.
- Line 834: `return 0 if (reproduces or args.drop_spain) else 1` — with `--drop-spain`,
  verify mode **cannot fail**, whatever the 2024/2025/combined diffs found.
- `_compare` line 742-744: when row counts differ it returns `not removed` — i.e. True
  with the per-cell diff skipped entirely. Under the original (pre-deletion) tree,
  `--verify --drop-spain` therefore had NO cell-level protection for 2023; the log's
  surviving-row check was done out-of-band, once, and is not in the script.

The real run happened to be clean (V1 proves it), so this is a weakness of the standing
producer, not a corruption. But this script is now THE producer of a published artefact,
and its only automated gate can be silently vacuous on exactly the invocation used for
the real run. Fix: fail the write when `reproduces` is False and the diff was expected
clean; under `--drop-spain`, key-align the surviving rows instead of skipping.

### F4 (HIGH) — the seven measured MC tables the runtime scorer reads are still counted off 71 RACES: the de-duplication moved the projection headline but left `data/mc_measured_v1.json` carrying the Spain-doubled statistics, and the PR does not regenerate or even mention it

Executed evidence:

- `git show HEAD:data/mc_measured_v1.json` → `"races_measured": 71`, `status_mix.total_laps
  4279`. This is the file `src/strategy/eval/projection.py:47` names as what "the Monte
  Carlo scorer reads at runtime" — seven tables (`clean_air`, `gap_density`,
  `neutralisation_rate`, `sc_window`, `status_mix`, `stop_hazard`, `undercut_band`), every
  one counted off the raw tree that until this PR contained the byte-identical Spain copy.
- A fresh run of `scripts/measure_mc_tables.py` against today's tree (it executed in this
  worktree during the session — see the caveat below) produces `races_measured: 70`,
  `status_mix.total_laps 4213`, and value changes well beyond a count: e.g. pooled
  `gap_density` racing n 65,179 → 64,017 (−1,162 ≈ the duplicate's laps), a `clean_air`
  circuit cell n 54 → 39 with `corrected_mean_s` +0.019 → −0.0196 (SIGN FLIP on a table
  the scorer consumes).
- Consequence for the suite:
  `tests/mc/test_mc_measured_tables.py::test_the_committed_tables_match_a_fresh_measurement`
  regenerates and asserts `after == before` — against the COMMITTED file it now fails by
  construction (71-race text vs 70-race regeneration). It is skipped on CI (no raw data),
  so the branch can merge green while the engine keeps reading Spain-doubled tables.
- The PR's own log and dataset notes never mention `mc_measured_v1.json`; C3-b's list of
  what must move named the prose numbers but this file's `races_measured` and its cells
  are the same class — a published number derived from the 71-dir tree.

Fix: regenerate `data/mc_measured_v1.json` + its `data/eval/mc_*` twins with
`scripts/measure_mc_tables.py`, review the deltas (the clean-air sign flip deserves a
look), and commit them IN THIS PR — the deletion of `data/raw/2023/Spain` and the tables
measured off it are one change, not two.

**Worktree caveat, resolved at close-out:** `data/mc_measured_v1.json` and the
`data/eval/mc_*` twins were rewritten in the working tree TODAY (13:11-13:15) — not by
this gate (my four-file pytest selection does not invoke the measurement script). The
writer is a CONCURRENT implementer session working the same tree: between 13:17 and 13:24,
four further files gained unstaged post-dedup re-measurements
(`src/agents/_shared_defaults.py` — DEFAULT_AIR/TRACK_TEMP 24.2/34.2 → 24.6/34.7 with a
dedup rationale comment; `tests/mc/test_projection_golden.py` — STAY_OUT E 1.276 → 1.28;
`tests/eval/test_registry_golden.py`; `tests/agents/test_weather_defaults_single_source.py`).
So the F4/F6 work is partly IN FLIGHT while this audit ran. The finding stands against the
COMMITTED state (HEAD still says `races_measured: 71`), and the fix reduces to: make sure
the in-flight re-measurements, the regenerated tables, and this PR land together, reviewed
— not half in the working tree of a parallel session.

### F3 (MEDIUM) — `test_the_restore_reproduces_N04s_own_2025_output_exactly` is now VACUOUS: the regeneration silently vacated the suite's only alignment-vs-truth test

`tests/agents/test_weather_restore.py:131-164` calls
`augment_featured_laps(laps_featured_2025, 2025)` and compares the result's weather against
the combined artefact. But `laps_augment.py:213` — `wants_weather = not any(column in
df.columns for column in WEATHER_COLUMNS)` — now evaluates False (the regenerated artefact
natively carries all four), so **`weather_for_race` never executes in this test**: it
compares the file's native weather to the concat-identical combined slice. Green by
construction. Executed proof: monkeypatching `weather_for_race` to return garbage — the
test still passes (see command in the evidence appendix). The docstring still claims "This
is the test the pace-MAE reproduction CANNOT replace... This sees the alignment"; it now
sees nothing. `weather_for_race`'s nearest-join alignment is still LIVE serving code (the
replay-path weather fix uses it), and its only remaining coverage is synthetic unit
frames. A regression to `direction='backward'` in `weather_restore.py` would today pass
the entire suite. Fix: point the test at a 48-column fixture (e.g. the artefact with the
four columns dropped) so the restore path actually runs, and compare against the native
columns as ground truth — the stronger test the new artefact finally makes possible.

### F5 (MEDIUM) — Claim C REFUTED on provenance: 53 columns is NOT "the ORIGINAL schema"; the rebuild reproduces a producer 6 months NEWER than the one that made the published artefact, and the dataset notes repeat the false pedigree

- Executed: `git log -S "add_temporal_normalization_features" -- notebooks/data_engineering/
  N04_feature_engineering.ipynb` → `7677a86 2026-02-12`; `-S "add_weather_features"` →
  `e8ce966 2026-02-15`. GATE_801 D1 already established the published artefacts are a
  pre-2026-02-15 build with **48 columns**; a 48-column output also predates 2026-02-12
  (nothing in `_COLS_TO_DROP` removes `lap_time_pct_of_race_fastest`, so a producer that had
  the function would have emitted the column). **The published original never carried 53.**
- The log's authority — "weather_restore.py's own docstring states … 53 columns" — is the
  same docstring D1 proved was written against a machine-local rerun that an HF re-download
  overwrote 51 minutes later. The plan's own gate refuted that line; PR 6's log re-cites it
  as ground truth. Claim-true-inside-false-headline, this repo's signature class.
- What is actually true: the rebuild reproduces TODAY'S N04, so `lap_time_pct_of_race_fastest`
  and the weather quartet are **post-publication producer additions published now for the
  first time** (the weather also equals what the runtime restore always fed the models — V7).
- Consequences verified NIL, each executed: `pace_holdout` selects by include-list
  (`df[features_delta]`, 25 names — no `lap_time_pct`, no flag column); `tire_agent.py:847-851`
  recomputes `lap_time_pct` unconditionally from `session_meta`; the TCN reads `laps_tiredeg`
  (unchanged). No consumer reads either new non-weather column from the featured parquet.
- Fix (text only): `PR6_DATASET_NOTES.md:20` — replace "Part of the original schema" with
  the true story. And `weather_restore.py`'s module docstring (lines ~3, ~28-29) still
  describes the pre-regeneration world ("the combined … carries all four columns for 2025.
  Only the per-year split dropped them") — false before this PR per D1, false after it in a
  new way (nothing drops them anymore). The page describing the contract is part of the fix.

### F6 (MEDIUM) — the de-duplication moved the published headline but NOT the places that publish it: docs, docstrings and hardcoded "measured on 71 races" constants all still quote the pre-dedup sample

GATE_801 C3-b/§5.2 put this in the Spain PR's scope ("the prose numbers … wherever quoted";
"every doc quoting '71 races' needs the footnote"). The PR's diff touches none of them.
Two distinct classes:

Prose (update the text):
- `docs/pages/thesis.md:31,85` — public docs site: "1810 green-flag stops across 71 races",
  "86.5 %". Now 1,768 / 70 / 86.3% (V4).
- `src/strategy/eval/decision_modes.py:4` — "86.5% within one place over 1810 real stops".
- `src/agents/position_projection.py:273` — "the 1810-stop ground truth".
- `src/agents/pace_agent.py:421` ("all 71 races"), `src/agents/_shared_defaults.py:16`,
  `src/agents/race_state_builder.py:65`, `scripts/rebuild_featured_laps.py:667` (the PR's
  own new file says "71 races"), `tests/mc/test_position_projection.py:10,47`.

Measured constants (the VALUES embed the duplicate — re-measure or annotate):
- `src/agents/strategy_orchestrator.py:651,657,664,804` — deg-rate bounds, median gap
  2.23 s, 5.75 s pit-loss CI, 22.6 s pooled green pit loss "n=1746", all "measured on this
  repo's 71 races".
- `src/agents/position_projection.py:95` — fallback values "measured on 2026-07-25 over 71
  races".
- `src/strategy/inference/guard_rails.py:56` — "1900 real green-flag stops across 71 races".

The duplicate is 42 of ~1,810 stops (2.3%), so most of these values will barely move — but
"barely" is a measurement, not an assumption, and F4's clean-air sign flip shows at least
one measured cell moves visibly. Minimum honest fix if re-measurement is deferred: annotate
each constant ("sample included the 2023 Spain duplicate; superseded by the 70-race tree")
and open the re-measurement issue.

### F7 (LOW, dormant — the "second Miami-shaped defect", found in the NEW code) — `impute_circuit_speed` resolves featured GP names to raw folders with a naive `replace(" ", "_")`, the exact one-side-only normalisation the Miami weather bug was

Executed evidence:
- `_circuit_trap_means(raw_root, "Miami", 2025)` → `FileNotFoundError:
  data/raw/2025/Miami/laps.parquet` (the folder is `Miami_Gardens`; the featured frame the
  imputer walks says `Miami` because the alias was applied).
- The already-solved twin was not consulted: `laps_augment._raw_race_dir("Miami", 2025)` →
  `Miami_Gardens` (three-candidate resolution via `_FRIENDLY_TO_FOLDER`).
- Two failure modes in `scripts/rebuild_featured_laps.py`: line 706's `is_dir()` guard makes
  a mis-mapped OTHER season silently contribute no offset (quietly degrading the estimator
  toward "no other season to take an offset from"), and line 712 reads the CURRENT season's
  folder unguarded — a crash, not a graceful skip.
- Why it is dormant TODAY: the imputer only touches a race whose `mean_sector_speed` is
  entirely NaN; the only such race is Las Vegas, and `"Las Vegas" → Las_Vegas` exists in all
  three seasons (verified on disk). It fires the day an aliased circuit (Miami, or any
  future rename) loses a full trap column — precisely the failure the imputer exists for.
- Fix: resolve folders through `laps_augment._raw_race_dir` (or `_FRIENDLY_TO_FOLDER`), and
  guard the current-season read.

### F8 (LOW, a measurement the log never took) — the Vegas imputation does NOT improve the pace model; it makes Vegas predictions marginally WORSE than the NaN it replaces

Executed (frozen N06 delta model, pace_holdout's own transforms, backup vs regenerated
2025 artefact): season MAE 0.4097 → 0.4098; Vegas-only MAE 0.3072 → **0.3107** (+0.0035);
exactly the 726 scoreable Vegas rows move, mean |Δ| 0.0101 s, max 0.418 s, zero movement
anywhere else. XGBoost's missing-value branch was already handling the hole slightly
BETTER than the imputed constant does. The imputation's real value is the honest schema
(no unflagged hole; `test_every_race_on_disk_resolves` with an empty exception list) and
non-XGBoost consumers — worth stating plainly in the dataset card rather than letting
"validated MAE 1.22" imply the model got better. The golden gate holds either way
(73 passed, including `test_ml_recompute_golden`).

### F9 (LOW) — two committed documents contradict each other about who consumes the combined artefact, and the wrong one is the regeneration plan

GATE_801 Part 2 says the combined's 2025 rows have as "only consumers:
`test_weather_restore.py:121`, `test_pace_circuit_speed.py:27` existence-guard". But
`.nb_py/N07_tiredeg_eda.py:92` (`load_enriched_laps`) builds `laps_tiredeg` FROM
`laps_featured.parquet`, and `data_cache.py:114` says so. True at serving time, false for
artefact lineage: the combined's 2025 rows just changed on 4 columns (V1), so any future
`laps_tiredeg` regeneration inherits the per-year cluster conventions — a DIFFERENT frame
from the one the shipped TCN's artefact was cut from. Deliberate and retrain-gated (the
notes say so), but the lineage dependency should be recorded where the deferral is, not
contradicted by the plan document.

---

## Verification record (what each V-number executed)

- **V1** — scratchpad `gate_diff.py`: exact-equality cell diff + positional key-order check,
  all four files vs `data/_backup_pr6_featured/` (SHA-256s match the log).
- **V2** — scratchpad `gate_vegas.py`: independent leave-era-out over today's raw tree:
  own-circuit offset n=68 MAE 1.22 p95 3.40 (exact match); global n=69 MAE 9.53 (log's 9.44
  was the pre-deletion tree, n=70); prior-season baseline MAE 6.52 (exact). Vegas: two-trap
  245.977, offsets −12.715/−13.584 → 232.83. **239.14 reproduced by NONE of six readings**
  (own 232.83, global 241.48, prior-season-mean 229.11, 2024-only 232.39, 2023-only 233.26,
  featured-2023/24 msp 228.96). No leak: my loop never touches the target season's
  three-trap value and still lands on 1.22.
- **V3** — Spain-carrier sweep: `undercut_clean` (1,032 rows), `overtake_pairs_2023_2025`
  (28,494), `sc_labeled_2023_2025` (22 unique 2023 race_ids — the correct count), all four
  `circuit_clustering` parquets: zero 'Spain' keys. `laps_tiredeg`: 0 Spain rows, Barcelona
  2023 still 2,396 (the known, deliberately deferred C2 carrier).
- **V4** — `measure_projection_ground_truth`: (2023-2025) → races 70, stops 1,768,
  within-one 0.863, exact 0.592; (2025,) → 24 / 552 / 0.861 / 0.596. All four headline
  numbers reproduce to the digit.
- **V5** — `data_cache.py:116-119`: the four featured parquets as exact-path allow-patterns;
  `_build_allow_patterns` appends only `data/raw/<year>/<gp>/**` entries.
- **V6** — `test_las_vegas_2025...` now pins the imputed value to ±0.01 AND pins it distinct
  from the 2023 measurement (the defect it exists to block) — a contract update under the
  recorded sign-off, not a weakening. `test_every_race_on_disk_resolves` moved from a
  tolerated Vegas exception to `unresolved == []` — strictly stronger. The still-parametrised
  `("Spain", 2023)` case passes for the RIGHT reason: the keyspace chain resolves Spain →
  Barcelona → 269.712, the value the removed duplicate carried (executed; refutes GATE_801
  §5.2's prediction that it would return NaN and need repointing).
  `test_a_partial_weather_set_is_declined` still constructs a genuinely partial frame
  (drops three, keeps one) and still exercises the `any()` guard.
- **V7** — independent §3.2 re-run: `augment_featured_laps` over the 48-column BACKUP frames
  (restore path RUNS) vs the regenerated native weather: 0 mismatches on 20,908 + 23,256 +
  22,760 rows; Miami AirTemp NaN = 0 in 2023 (1,071 rows), 2024 (915), 2025 (857). The alias
  sits inside `if year == _HOLDOUT_SEASON:` (`rebuild_featured_laps.py:597-599`), so it
  cannot touch 2023/2024, and the weather frame's only consumer in the script is
  `add_weather_features` (line 606). The `f1-sim` Miami lap-call flip itself was not re-run
  (heavy); the mechanism behind it is what V7 verifies.
- **V8** — scratchpad `gate_verbatim.py`: AST-extracted source of all 12 lifted functions,
  normalised trailing whitespace only: VERBATIM against `.nb_py/N04_feature_engineering.py`;
  `_COLS_TO_DROP` and `_GP_NAME_ALIASES_2025` value-equal.
- **Suites executed**: `pytest tests/eval/test_ml_recompute_golden.py
  tests/agents/test_weather_restore.py tests/agents/test_pace_circuit_speed.py
  tests/mc/test_position_projection.py` → **73 passed** (131.97 s).
- **Producer determinism**: `rebuild_featured_laps.py --verify --impute-circuit-speed`
  (writes nothing) → every on-disk file reproduces, exit 0.

---

## Fix list, ordered by value

1. **F4 — regenerate and commit `data/mc_measured_v1.json` + `data/eval/mc_*` twins in THIS
   PR**, reviewing the deltas (clean-air `corrected_mean_s` sign flip included). The
   raw-tree deletion and the tables measured off it are one change. The fresh 70-race
   version already sits unstaged in the worktree — review it, do not trust it blind.
2. **F3 — repoint `test_the_restore_reproduces_N04s_own_2025_output_exactly`** at a
   48-column frame (drop the four columns from the artefact inside the test) so
   `weather_for_race` executes again; the native columns are now the ground truth it always
   wanted.
3. **F6 — move the published numbers**: docs/pages/thesis.md + the two docstrings (prose),
   and annotate-or-remeasure the 71-race constants in strategy_orchestrator.py /
   position_projection.py / guard_rails.py.
4. **F2 — make the script's gate a gate**: fail the write on a dirty diff; drop the
   `--drop-spain` exit-0 override; key-align surviving rows when counts differ.
5. **F5 — correct the provenance text** in PR6_DATASET_NOTES.md and refresh
   weather_restore.py's era-stale docstring.
6. **F7 — folder resolution in `impute_circuit_speed`** via `_raw_race_dir`, guard the
   current-season read.
7. **F1 — `rtol=0`** in `_compare`.
8. **F8/F9 — one sentence each** in the dataset card (imputation does not improve N06) and
   in the tiredeg deferral note (the combined is tiredeg's lineage input).

---

## What I tried to break and could NOT

1. **The value diff itself** (claim A): exact equality, not tolerance — 0.0 max deviation on
   every shipped column, all four files; positional row order proven identical via key
   tuples; backup SHA-256s match the log. The comparison the log ran is sound (keys are
   among the compared columns, so zero-diff implies same order), and its loose `rtol` (F1)
   is latent, not hiding anything today.
2. **The combined-vs-split disagreement** (claim B): reproduced all four counts exactly from
   the backups, key-aligned; confirmed the new combined equals the concat of the parts with
   zero differing cells — the disagreement is now impossible by construction.
3. **The Vegas estimator** (claim E): independent re-derivation reproduces every number; I
   tried six readings to rescue the earlier audit's 239.14 and none lands on it; the
   leak-free loop still gives MAE 1.22, so the validation was not inflated by leakage.
4. **The Miami weather fix** (claim K): re-ran the alignment gate independently over all
   66,924 surviving rows — 0 mismatches, Miami whole in all three seasons; the alias cannot
   reach 2023/2024 by control flow; no other consumer of the weather frame exists in the
   script.
5. **The verbatim-lift claim** (claim L): AST-level comparison found not one changed line in
   any of the 12 functions or 2 constants.
6. **A second LIVE Miami-shaped defect** (the brief's lead): I checked every merge in the
   rebuild for one-side-only key normalisation — the cluster merges (2025 laps 'Miami' vs
   `k4_2025`'s 'Miami' row; 2023/24 'Miami' vs the grafted k4 rows), the weather merge
   post-fix, the raw-folder loads per era ('Miami' folders in 2023/24, 'Miami_Gardens' in
   2025 only, verified on disk), and the imputer's folder mapping. The only instance found
   (F7) is dormant on today's data; the zero-diff on all 48 shipped columns plus the
   0-mismatch weather gate bounds any live one to columns nobody compares — and every
   column in all four files WAS compared (V1).
7. **The imputation's blast radius** (claim D): the exhaustive four-file diff pins it to
   exactly 760 cells + the flag column; the pace golden holds and the measured MAE movement
   is +0.0001 season-wide (F8 quantifies the honest downside).
8. **The projection headline** (claim G): all four numbers reproduce to the digit on the
   deduplicated tree, including the never-before-measured 2025-only slice.
9. **The clean-install path** (claim H): all four featured files are exact-path
   allow-patterns; nothing in `_build_allow_patterns` can drop them.
10. **The updated tests** (claim I): I constructed the defect each would now miss and found
    the two pace tests strictly stronger than their predecessors; the genuinely vacated
    test (F3) was one the PR did NOT list as touched — the artefact change vacated it
    silently.

## Verdict

The artefact regeneration itself is exactly what it claims: value-identical on every
shipped column, sound in row order, deterministic, with a single sanctioned 760-cell
imputation, fully flagged, and a weather restoration that matches the runtime truth to the
last row. The claims that fail are AROUND the artefact, not in it: the runtime MC tables
still count the deleted duplicate (F4 — the one HIGH), the suite's only alignment-vs-truth
test died of success (F3), the published numbers did not move with the sample (F6), and
the provenance story in the dataset notes cites a docstring the plan's own gate had
already refuted (F5).
