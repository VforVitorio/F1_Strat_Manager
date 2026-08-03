# GATE #797 — circuit `mean_sector_speed` feed (adversarial gate)

**Date:** 2026-08-03 · **Branch:** `fix/pace-circuit-mean-sector-speed` @ `23a02ba` vs `dev` @ `8ebe9c3`
**Role:** adversarial gate. No repo file modified except this report. Findings appended as confirmed.

**Scope (4 commits):**
- `36aa8d0` fix(agents): feed N06 the circuit's mean sector speed instead of the speed trap
- `a3b876c` test: skip, do not fail, when the unpublished undercut holdout is absent
- `7ab7b5a` docs(multi-agent): calibrated pit bounds + operating envelopes
- `23a02ba` docs(agents): correct the parity claim behind the circuit speed lookup

Note the branch itself already refutes half of claim A: `36aa8d0` claimed "the value served is
the value fitted … identical to 0.0"; `23a02ba` retracts it (the 0.0 compared two *training*
artefacts, not the served map) and re-frames serving 2025 as deliberate. This gate verifies the
retraction's numbers too, since a correction can itself be wrong.

## Claim checklist

- [ ] A. served value == fitted value; the 2023-24 vs 2025 seam quantified per GP
- [ ] B. unresolved circuit reaches the model as NaN; `_bootstrap_ci` noise on NaN; downstream
- [ ] C. slug resolution complete across every real producer of `gp_name`
- [ ] D. explicit `mean_sector_speed` still wins; no caller silently overridden
- [ ] E. impact numbers honest (mean +0.069 s, p95 0.377 s, >0.010 s on 38%, n=4000)
- [ ] F. nothing else regressed; `_compute_derived` lost `prev_speed_st` — no stale caller
- [ ] G. envelope re-declaration (196.63, 314.97) is the right range for the served value
- [ ] H. skip-guards skip (not fail) and do not over-skip
- [ ] Bug-class hunt: third member of the "inference value ≠ training quantity" family
- [ ] `docs/pages/multi-agent.md` new sections vs what the code does

---

## Findings (appended as confirmed)

### F1 — HIGH · The 2025 Miami race resolves to NaN on the CLI/arcade path: the fix misses one of the 24 races of the very season it serves

**Executed evidence.** `data/raw/2025/Miami_Gardens/metadata.json` carries `"gp_name": "Miami Gardens"`
— that exact string is what `RaceReplayEngine._parse_meta` (`src/simulation/replay_engine.py:115`)
puts into `session_meta.gp_name`, and what `run_from_state` hands to
`_resolve_mean_sector_speed` (`src/agents/pace_agent.py:341`). Executed against the real agent:

```
_resolve_mean_sector_speed('Miami Gardens') -> nan   (+ WARNING per lap)
_resolve_mean_sector_speed('Miami')         -> 221.384  (the value that was intended)
```

Direct hit misses (map key is `'Miami'`, from `laps_featured_2025.parquet`); `slug_from_event_name('Miami Gardens')`
returns `None` (`EVENT_NAME_BY_SLUG` knows `'Miami'` and `'Miami Grand Prix'`, not `'Miami Gardens'`);
`FOLDER_ALIASES` knows only the underscore form `'Miami_Gardens'` and is not consulted here anyway
(`canonical_gp_name` is never called in the new resolver). Failing scenario:
`f1-sim Miami_Gardens NOR McLaren` — every lap of the 2025 Miami GP feeds N06 a missing
`mean_sector_speed` although the served map holds the circuit's value. The same string also misses
`circuit_cluster` (silent default cluster 1) and `_session_median` (no median) — those two are
pre-existing, but the NEW lookup was added to the same function that already contained both misses
and repaired neither pattern.

Note the shape: #797's own fix commit cites #448 ("the dual-keyspace trap") and resolves TWO of the
three keyspaces (parquet slug, FastF1 event name). The third keyspace — metadata.json/raw-folder
names, the one `canonical_gp_name` + `FOLDER_ALIASES` exist for — is the one that still misses.
One copy fixed, its sibling keyspace not.

### F2 — HIGH · Las Vegas is served NaN although N06 was FITTED on it — the loader reads the one artefact that lost the value

**Executed evidence.** `laps_featured_2025.parquet` carries `mean_sector_speed = NaN` on all 760
Las Vegas rows, so `_load_circuit_mean_sector_speed`'s `.dropna()` (`pace_agent.py:337`) drops the
GP and the served map has 23 entries, not 24. But the fitted value **exists on disk in the very
artefacts this branch reasons about**: `laps_featured_2023.parquet` and `laps_featured_2024.parquet`
both carry 228.9645 for Las Vegas, `circuit_features_with_clusters_k4.parquet` carries 228.9645,
and even the combined `laps_featured.parquet` carries 228.9645 **on its 2025 rows**. Only
`circuit_features_with_clusters_k4_2025.parquet` and the per-year 2025 file hold NaN.

`_resolve_mean_sector_speed('Las Vegas') -> nan` (executed). The docstring's justification —
"we do not know this circuit" — is false for Las Vegas: the model knows it (fitted on it, bounds
include it), the 2025 artefact simply has a hole. Issue #797 itself said "all 24 GPs (one is NaN)"
and the fix shipped without closing that named gap. Failing scenario: `f1-sim Las_Vegas VER Red Bull`
(metadata gp_name `'Las Vegas'`) — a full 2025 race weekend served a missing feature when the
fitted number is one file away.

### F3 — MEDIUM · Every 2023/2024 replay is served the 2025 constant, and `run()`'s `year` parameter is ignored by the resolution

The CLI supports `--year 2023/2024` (`run_simulation_cli.py:2101`, featured path tracks the year)
and the raw trees for 2023/2024 are on disk. On those replays `session_meta.gp_name` resolves
against the **2025-only** map, so a 2023 Silverstone lap is fed 231.36 where the value N06 was
fitted on for that lap — and which sits in the very parquet row being replayed — is 249.71
(18.35 km/h apart). `PaceAgent.run()` receives `year` and does not use it in
`_resolve_mean_sector_speed`. Additionally the 2023 Spanish GP replay
(`data/raw/2023/Spain/metadata.json` → gp_name `'Spain'`) resolves to **NaN** — a fourth keyspace
casualty (executed: `_resolve_mean_sector_speed('Spain') -> nan`).

The `23a02ba` docstring defends serving 2025 as "the quantity N04 would compute for a 2025 lap" —
an argument that is only about 2025 laps. For 2023/2024 laps it serves a measurement from two years
in the future of the lap being replayed. Whether this moves predictions materially is measured in F-E
below (Silverstone: the 18.35 km/h gap is the largest in the fleet).

### F4 — MEDIUM · The `23a02ba` mechanism claim "recomputed per season" is wrong: 2023 and 2024 share one identical value per GP

**Executed evidence:** per-GP `mean_sector_speed` in `laps_featured_2023.parquet` equals
`laps_featured_2024.parquet` **exactly (diff 0.0 on all 23 common GPs)**, and both equal
`circuit_features_with_clusters_k4.parquet` to 0.0. So the feature is recomputed per **artefact
generation** (one training-era build pooling both seasons, one 2025 build), not "per season from
that season's laps" as the amended docstring (`pace_agent.py:315-316`) states. The conclusion
(2025 differs from training) survives; the stated mechanism does not — and this repo's own bug
class list says a comment naming the wrong mechanism is how the next fix goes wrong (e.g. someone
"completing" per-season resolution for 2023 vs 2024 would find nothing to resolve).

Verified numbers from the same run: 23 GPs present in both eras, none equal, mean abs gap 4.815
(commit says 4.82 ✓), max Silverstone 18.347 (commit says 18.35 ✓), Monza 2025 = 317.2412 (✓) is
the only served value outside (196.6292, 314.9706) (✓), and the declared bounds equal the true
min/max of the 2023+2024 lap rows exactly (✓ claim G's range is genuinely the fitted range;
min = Austin 196.63, max = Monza 314.97).

### F5 — VERIFIED (claims B and D) · NaN survives to the model; the CI stays finite; an explicit value wins

Executed end-to-end with the real agent: `run(gp_name='Miami Gardens', ...)` produces
`mean_sector_speed = NaN` in the feature row, survives `pd.to_numeric(errors='coerce')`,
`_bootstrap_ci` multiplies NaN by Gaussian noise (`NaN * N(1,0.02) = NaN`, verified) and XGBoost
routes the missing value through its default split — **p10/p90 come back finite** (89.978/95.155),
`lap_time_pred` finite, no crash. The envelope reports the feature as `unknown`, never a violation,
so an unresolved circuit is invisible in the OOB warning (by design — but note it means F1/F2 fire
no envelope signal either; the only trace is the per-lap resolver WARNING).
`delta_vs_median` is NaN for 'Miami Gardens' (the median lookup misses on the same keyspace),
which downstream consumers already tolerate per the pre-existing no-median path.

Claim D verified: `mean_sector_speed=999.0` passed explicitly reaches the row untouched and trips
the envelope (violation logged). Repo-wide search: no production caller passes an explicit
`mean_sector_speed` — `_run_always_on_agents` (`strategy_orchestrator.py:1867`) and every
`run_from_state` caller omit it, so nothing is silently overridden.

---

## Resumed gate (round 2)

**Date:** 2026-08-03 · **Branch:** `fix/pace-circuit-mean-sector-speed` @ `806cedd` vs `dev` @ `8ebe9c3`.
Round 1's F1-F4 were fixed in `806cedd` (year-keyed map, `_normalise_gp_key`, combined artefact,
walk-the-disk test). This round attacks the fix itself and finishes the unreached checklist.

### Round-2 claim checklist

- [x] 1. `_normalise_gp_key` total, collision-free across real producers
- [x] 2. `(Year, GP)` keying: absent/stale/string years across all tiers
- [x] 3. "71 pairs, ZERO missing" re-derived; same-races check
- [x] 4. `test_every_race_on_disk_resolves` cannot pass vacuously — VERIFIED (see below)
- [x] C. producer enumeration (folded into claims 1 and 2 below)
- [x] E. impact numbers reproduced + re-measured for the NEW serving — REFUTED (R5)
- [x] F. no regression (`_compute_derived` signature, goldens, holdout MAE) — PASS + 2 stale docstrings
- [x] G. envelope re-declaration under the new map — REFUTED (R3: outside set is empty)
- [x] H. skip-guards in `a3b876c` — PASS + 1 wrong reason string
- [x] Bug-class hunt: third "inference value != training quantity" member — FOUND (`LapsSincePitStop`)
- [x] Docs: `docs/pages/multi-agent.md` new sections — numerics PASS, 2 overtaken statements

### R1 — HIGH · The combined artefact does not contain what the fix says it serves: `mean_sector_speed` is ONE constant per GP across all three years, so the year-keying serves the training value to 2025 laps — silently reversing the decision `23a02ba` documented as deliberate

**Executed evidence.** Enumerated every (Year, GP) pair in `laps_featured.parquet` and pivoted
per GP across years: **the cross-year gap is exactly 0.0 for every GP name shared across
years**. Silverstone is 249.7062 in 2023, 2024 AND 2025; Monza 314.9706 in all three; the only
per-GP year variation in the entire served map enters through the Miami naming split
(2023/24 'Miami' 222.364 vs 2025 'Miami Gardens' 221.384, gap 0.98 km/h). Compared against the
per-year artefacts: combined == `laps_featured_2023/2024.parquet` exactly, while combined 2025
rows disagree with `laps_featured_2025.parquet` on **22 of 22 comparable GPs** (Silverstone
combined 249.71 vs 2025-artefact 231.36, diff +18.35; Melbourne +15.61; Montreal -9.22; Sao
Paulo -8.81; Suzuka -8.06; Shanghai -7.96; full table executed). The combined build evidently
broadcasts the training-era per-GP constant onto its 2025 rows (Las Vegas 2025 = 228.9645 =
training; Shanghai 2025 = its 2024 value; only GP names unseen in training — 'Miami Gardens' —
keep a 2025-computed value).

Consequences, in order of weight:

1. **The docstring's central factual claim is false for the file it describes**
   (`src/agents/pace_agent.py:334-341`): "`laps_featured.parquet` carries 71 (year, GP) pairs
   and the value differs between the training era and 2025 on every GP present in both: mean
   absolute gap 4.8 km/h, largest Silverstone at 18.4." That 4.8/18.4 measurement is TRUE of
   the per-year artefacts (round 1 verified it there) and FALSE of the combined file, where
   the gap is 0.0 everywhere. A true number transplanted into a false headline — and it is
   the SECOND wrong-mechanism claim in the same docstring's history (F4 was the first; the
   commit that corrected it introduced this one).
2. **"Keying by year makes the value served the value the lap being replayed actually
   carries" (`src/agents/pace_agent.py:340-341`) is false on the real replay path.** Every
   replay/serving consumer loads the PER-YEAR artefact (`src/strategy/inference/engine.py:99`
   states the rule; `src/arcade/strategy.py:571`,
   `src/telemetry/backend/utils/laps_cache.py:31`,
   `src/telemetry/backend/services/simulation/simulator.py:231`,
   `src/strategy/eval/pace_holdout.py:95`, `src/strategy/eval/decision_modes.py:517` comply).
   A replayed 2025 Silverstone lap carries 231.36 in the frame the RSM serves; the agent now
   feeds N06 249.71.
3. **The serving for 2025 replays FLIPPED, and no one decided it.** At `23a02ba` the docstring
   defended serving the 2025 measurement as "deliberate and the correct half of that pair"
   ("serving the training seasons' value would feed a stale reading"). `806cedd` reverses
   exactly that — every 2025 lap now gets the training value — as an unexamined side effect
   of switching files, while claiming to do the opposite. Executed:
   `_resolve_mean_sector_speed('Silverstone', 2025) -> 249.70624181580095` (was 231.3595
   before the fix). Whether training-constant or season-measurement is the RIGHT feed is a
   modelling judgment with arguments both ways (the feature acts as a circuit identifier the
   model fitted on); the defect is that the branch has now shipped BOTH positions, each
   documented as deliberate, with no measurement of the flip (see E below).

### R2 — HIGH · The eval tier and production now feed N06 different values for the same 2025 lap: the published holdout MAE no longer describes the serving configuration

**Executed evidence.** `src/strategy/eval/pace_holdout.py` never calls the agent: it reads
`laps_featured_2025.parquet` (line 95) and feeds the frame's own columns to the model
(line 120), so the holdout MAE was measured with `mean_sector_speed` = the 2025-artefact
values (Silverstone 231.36). Production (`_resolve_mean_sector_speed`) now serves the combined
values (Silverstone 249.71). The two disagree on 22 of 24 2025 GPs, by up to 18.35 km/h — the
exact magnitude the branch's own commits call material. The new test does not catch this
because `test_every_race_resolves_to_the_value_its_own_parquet_rows_carry` compares the map
against `laps_featured.parquet` — **the same file the map is built from**. It asserts
map == source-of-map (a VALUE), not served == what-the-replayed-lap-carries (the EFFECT);
tautological by construction, the repo's value-not-effect test class.

### R3 — MEDIUM · Claim G refuted: under the new map NOTHING is outside the envelope — "Monza 2025 is exactly that, and the only one" is now false, and the new test's discrimination loop iterates the empty set

**Executed evidence.** Instantiated the real `PaceAgent` and checked all 71 served values
against `_N06_TRAINED_BOUNDS["mean_sector_speed"]` (196.6292, 314.9706): **outside set =
{}**. Served Monza 2025 is 314.9706 — the bound's own maximum, inside by equality — because
the combined file broadcast the training value (R1); the 317.2412 the docstring
(`src/agents/pace_agent.py:359-360`) and the test docstring celebrate is the per-year value
the agent no longer loads. So the envelope on `mean_sector_speed` is back to a bound that
CANNOT fire from the circuit map on any known race — only an explicitly-passed value can trip
it, and no production caller passes one (round-1 claim D).
`test_the_envelope_separates_a_circuit_n06_was_fitted_on_from_one_it_was_not` still passes,
but its `for (year, gp), value in outside.items()` loop — the half that pins "Monza fires the
envelope" — asserts over the EMPTY SET (`tests/agents/test_pace_circuit_speed.py:264-267`),
and `len(inside) > len(outside)` is 71 > 0. The repo's own bug class: a green guard asserting
about nothing.

### R4 — MEDIUM · The 2023 Spanish GP exists TWICE — `data/raw/2023/Spain` and `data/raw/2023/Barcelona` are the same race (OpenF1 session_key 9102) — so "71 races on disk == 71 pairs" is 70 real races plus one duplicate, on both sides of the check

**Executed evidence.** Both metadata.json files carry `session_key_openf1: 9102` and identical
record_counts (laps 1312, weather 154, intervals 26036, pitstops 43); extraction timestamps 21
seconds apart. The combined parquet correspondingly holds BOTH (2023,'Spain') and
(2023,'Barcelona'), 1198 rows each, identical `mean_sector_speed` 269.7125 — which is why 2023
shows 23 GP names for a 22-race season. Claim 3's "the pairs are the same 71 races data/raw
holds" is therefore true only because BOTH sides double-count the same race. No wrong value is
served (executed: `_resolve_mean_sector_speed('Spain', 2023)` ==
`_resolve_mean_sector_speed('Barcelona', 2023)` == 269.7125), but the 2023 Spanish GP is
duplicated inside the combined training artefact itself — 1198 rows counted twice by anything
that aggregates `laps_featured.parquet` per-lap (whether N06's training build consumed the
duplicate needs its own check, outside this gate's file-touch scope). Deserves its own
data-integrity issue.

### Claim 1 — VERIFIED with one latent gap · `_normalise_gp_key` is collision-free across every real producer; the only unresolvable spellings have no live producer

**Executed.** Normalised every distinct `GP_Name` in the combined parquet (26 spellings) and
every `metadata.json` `gp_name` (26 spellings, all 71 races): **zero many-to-one collisions
across different circuits** — the only merges are 'Miami Gardens'/'Miami_Gardens' -> 'Miami'
(same circuit, the intended repair) and underscore/space forms of the same race. Every raw
race resolves through the real resolver chain (`unresolved == []`, 71/71). Producer sweep
(C, finishing round 1's unreached item): the CLI and Arcade pass metadata.json names via
`RaceReplayEngine._parse_meta` (all resolve, executed); the backend passes per-year featured
GP_Names from its own `/available-gps` round-trip (accented 'Montréal'/'São Paulo', 'Miami'
in 2025 — all resolve, executed); MCP tools funnel free text through `_normalize_gp_name`
whose alias table covers the ASCII forms ('montreal' -> 'Montréal', 'sao paulo' -> 'São
Paulo'; `src/telemetry/backend/mcp_tools.py:199` + table above it); FastF1 event names are
handled by the `slug_from_event_name` candidate and currently have no live producer. The two
CLI ASCII fallbacks `gp_slugs` accepts, 'Montreal' and 'Sao Paulo', do NOT resolve
(`_resolve_mean_sector_speed('Montreal', 2025) -> nan`, executed) because
`canonical_gp_name` returns them unchanged and the map keys are accented — but no live
producer emits them into `session_meta.gp_name` (the CLI arg keyspace terminates at folder
resolution; session_meta is rebuilt from metadata.json). LATENT, not live. LOW.

### Claim 2 — VERIFIED with one latent trap · every live producer supplies an int year the dataset covers; a STRING year fails silently

**Executed.** `RaceReplayEngine._parse_meta` (`src/simulation/replay_engine.py:115`) casts
`int(meta.get("year", 2025))`; all 71 metadata.json `year` fields are ints equal to their
folder year (swept: zero mismatches, zero missing files). Backend years are FastAPI-typed
ints (`endpoints/strategy.py:85,109,411`); MCP `_normalize_year` coerces or refuses
(`mcp_tools.py:320`). `meta.get('year') or 2025` in `run_from_state` can only fire on a
session_meta missing 'year', which no current producer emits (RSM always sets it,
`src/simulation/race_state_manager.py:515`). The engine's no-metadata fallback (year=2025 +
warning, `replay_engine.py:117-120`) is the one path that could hand a 2023 replay a 2025
key — unreachable today (all 71 metadata files exist), and per R1 it would serve the same
NUMBER anyway for every GP except Miami and 'Spain' (whose (2025,·) key does not exist). The
latent trap: `_resolve_mean_sector_speed('Silverstone', "2025") -> nan` (executed) — a string
year misses every key because the tuple key is type-strict and nothing coerces; the only
signal is the per-lap warning. No live producer does this. LOW, latent.

### Claim 3 — VERIFIED as numbers, hollow as a guarantee

**Executed.** `laps_featured.parquet`: 68,122 rows, **0 NaN** in `mean_sector_speed` (the
`.dropna()` is a no-op), 71 distinct (Year, GP_Name) pairs before AND after dropna; the
normalised map holds exactly 71 keys; every pair matches a raw race on disk and vice versa
(both directions swept, empty set differences). But per R4 the "71 races" is inflated by the
same duplicate on both sides, and per R1 the VALUES those 71 keys carry are 24 per-GP
constants wearing 71 keys.

### R5 — HIGH · Checklist E: the commit's impact numbers do NOT reproduce — every honest harness lands at half to a quarter of the claim

The `36aa8d0` message claims: "Measured over 4000 real 2025 laps, the corrected feed moves the
delta prediction by a mean of +0.069 s, a p95 absolute of 0.377 s, and more than 0.010 s on 38%
of laps."

**Executed reproduction** (fix-at-`36aa8d0` serving vs the `prev_speed_st` bug feed, frozen N06,
featured 2025 rows rebuilt exactly as `pace_holdout` does — encode + lag + dropna, 21,247 rows):

| Harness | n | mean signed | mean abs | p95 abs | >0.010 s |
|---|---|---|---|---|---|
| **Claimed** | 4,000 | **+0.069** | — | **0.377** | **38%** |
| Full 2025 season | 21,247 | +0.0177 | 0.0201 | 0.1766 | 17.7% |
| Uniform n=4000, seeds 0/1/42 | 4,000 | +0.0170..0.0173 | 0.0190..0.0196 | 0.1766 | 16.6-17.3% |
| First 4000 rows (`head`) | 4,000 | +0.0412 | — | 0.1766 | 28.6% |
| Last 4000 rows (`tail`) | 4,000 | +0.0003 | — | 0.0000 | 1.8% |
| Full, default weather (25/35/50/0) | 21,247 | +0.0207 | — | 0.1766 | 16.9% |
| Full, default wx + `Prev_Deg*`=0 (the real `run_from_state` row) | 21,247 | +0.0146 | — | 0.1766 | 9.5% |

No configuration tried — uniform sample under three seeds, head, tail, default weather,
inference-shaped rows — reaches the claimed mean, p95 or fraction; p95 abs is 0.1766 s in every
one of them, less than half the claimed 0.377. The delta is heavily circuit-concentrated (per-GP
mean abs: Baku 0.177, Spa 0.171, Silverstone 0.088, then a cliff — Austin 0.010 and below), so
only a sample dominated by three specific races could produce numbers that size; 4000 uniform
2025 laps cannot. The claim's DIRECTION survives (the feed change is real and material on some
circuits) inside a false magnitude — this repo's "claim true in isolation inside a false
headline" class, and its "a number measured on the wrong distribution" class at once. On the
distribution production actually serves (default weather, `Prev_Deg*`=0), the fix moves >0.010 s
on 9.5% of laps, not 38%.

**The re-measure nobody did (E's second half), executed on all three seasons** — what `806cedd`
itself changed relative to `23a02ba`:

| Season | Serving change | mean signed | p95 abs | >0.010 s | max abs | MAE before → after |
|---|---|---|---|---|---|---|
| 2025 (21,247) | 2025-artefact values → training constants (R1 flip) | +0.0009 | 0.0016 | 3.3% | 0.418 | 0.4097 → 0.4097 |
| 2024 (22,077) | GP-keyed 2025 map → own-year values | −0.0003 | 0.0245 | 5.1% | 0.424 | 0.4185 → 0.4176 |
| 2023 (20,880) | GP-keyed 2025 map (Spain+Vegas NaN) → own-year values | −0.0001 | 0.0016 | 1.7% | 0.393 | 0.4316 → 0.4314 |

So the `806cedd` re-keying is nearly prediction-neutral in aggregate (single laps move up to
0.42 s), and — the number that reframes the whole branch — **the original bug itself was
prediction-neutral on the holdout: MAE with the speed-trap feed was 0.4096, with the corrected
feed 0.4097.** The fix is epistemically right (right quantity, honest NaN) and the gate confirms
it should stand; but every published magnitude around it is inflated, and the flip it smuggled
in (R1) is invisible in aggregate while reversing a documented decision.

### Checklist F — VERIFIED, with two stale docstrings and one hollow claim

- **`_compute_derived` signature**: exactly one caller in the repo
  (`src/agents/pace_agent.py:560`, keyword-style, matches the new 6-parameter signature); no
  test, script or notebook-side import calls it (swept `src/ tests/ scripts/`; N25's notebook
  carries its own inline copy and does not import this one). PASS.
- **BUT the docstrings were not updated with the signature — the fix's own function documents
  the mechanism the fix removed** (the repo's wrong-mechanism comment class, in the very
  function that was the bug):
  - `src/agents/pace_agent.py:472` — `_compute_derived`'s Args block still lists
    `prev_speed_st: Speed trap reading in km/h from the previous lap.` The parameter no longer
    exists; the docstring documents 7 args for a 6-arg signature.
  - `src/agents/pace_agent.py:784` — `run()`'s Args block still says `mean_sector_speed:
    Average sector speed; defaults to prev_speed_st.` It defaults to the circuit lookup now;
    "defaults to prev_speed_st" is precisely the behaviour #797 removed. A reader trusting
    this line reintroduces the bug.
- **Strategy goldens**: `tests/mc/test_strategy_goldens.py` passes (54 passed, executed run).
  But the goldens exercise `_run_mc_simulation` / `_decide_agents_to_call` on CANNED sub-agent
  outputs (`tests/mc/canned_outputs.py`) — they never enter `_build_feature_row`, so they were
  structurally incapable of moving with this change. `36aa8d0`'s "Strategy goldens are
  byte-identical" is true and verified nothing about the feed: byte-identical because
  insensitive, not because checked.
- **Holdout MAE**: `test_pace_mae_reproduces_from_featured_laps` passed (executed), asserting
  `|MAE − 0.4104| < 0.01`; my independent rebuild gives 0.4097, inside tolerance. PASS.

### Checklist H — VERIFIED · the guards skip, do not over-skip, and one reason string names the wrong artefact

Executed: the suite run on this machine (which HAS `lgbm_undercut_v1.pkl` and LACKS
`undercut_labeled/undercut_clean.parquet`) yields 54 passed, 1 skipped — the exact intended
behaviour of `a3b876c` (before it, this checkout went red on absent data). Right-sizing checked:
`test_undercut_targets_are_on_track`'s module-level skip is justified because constructing N28
really does read `undercut_clean.parquet` (`src/agents/pit_strategy_agent.py:271`), so every
test in the module needs it; `test_mc_measured_tables` skips ONLY the fresh-measurement test
(the committed-table assertions still run), avoiding the emptied-file-left-in-worktree hazard
its comment documents. One defect: the widened condition kept the old message —
`tests/eval/test_ml_recompute_golden.py:61` skips with reason "undercut **model** absent (CI
runner without weights)" on a machine where the model is PRESENT and the holdout is what is
missing (executed: that exact line fired here). The wrong-mechanism class, in a diagnostic:
it sends whoever reads the skip log to re-download weights they already have. LOW.

### Claim 4 — VERIFIED · the walk-the-disk test is non-vacuous and keyed the way the engine keys

Executed independently of pytest: the same walk finds 71 race dirs, all with `metadata.json`,
`checked == 71 > 0`, `unresolved == []`. The `assert checked > 0` guard makes an empty walk
fail. Two soft edges, neither currently load-bearing: (a) a race dir WITHOUT `metadata.json` is
silently skipped (`continue`) rather than counted — today zero such dirs exist, and the engine's
own fallback for that case (folder name + year=2025, `replay_engine.py:117`) resolves for every
current folder name except a hypothetical metadata-less `2023/Spain`, whose (2025,'Spain') key
does not exist; (b) the test keys by `int(year_dir.name)` while the engine keys by
`meta['year']` — verified equal on all 71 (swept), so the test currently tests the engine's
keyspace, but only because the data happens to agree.

### Bug-class hunt — the third member found and measured: `LapsSincePitStop` is fed `TyreLife`, and the two differ on 19.8% of training rows

`run_from_state` (`src/agents/pace_agent.py:908`) passes `laps_since_pit=d.get("tyre_life") or
1` — the SAME value it passes as `TyreLife`. In training the two are different quantities:
`TyreLife` counts laps on the tyre SET (which arrives pre-used: qualifying laps), while
`LapsSincePitStop` counts laps since the last stop. **Executed over the 45,327 training-season
rows: they differ on 19.8% (8,995 rows), mean gap 2.61 laps, p95 6, max 25; 16.9% of stint-1
rows start with `TyreLife > LapsSincePitStop`** (used-set starts). So on roughly one lap in
five N06 was fitted distinguishing the two, and at inference it is structurally fed the wrong
one whenever the set is not fresh from the box. The envelope comment (`pace_agent.py:134-136`)
KNOWS the equality — but frames it only as "a second bound would double-report", i.e. as a
reason not to bound it, not as the wiring defect it also is. Family so far: `Prev_Deg*`
(hardcoded 0.0, documented), `mean_sector_speed` (fixed by this branch), **`LapsSincePitStop`
(live, unfixed, now measured)**. MEDIUM.

Also swept, adjacent but distinct classes, for the record: `_encode_categorical`'s defaults
(`compound→1, team→0, cluster→1`, `pace_agent.py:441-443`) are sentinels that are also REAL
encoded values — an unknown team silently becomes whichever team encodes to 0 (live on the CLI,
where the team string is user-typed argv); `Prev_TyreLife := tyre_life−1` is wrong on every
outlap (training holds the OLD set's last life; measured mean gap 16.1 laps on 780 outlaps) but
folds into the already-documented "N06 was never trained on stint-first laps" extrapolation the
new docs section names, so it is not counted as a new family member. Both pre-existing, neither
introduced by this branch.

### Docs — `docs/pages/multi-agent.md` new sections audit: numerics check out, two statements already overtaken by the code

Verified against code and artefacts (executed where numeric): train/val row counts 22,106 /
23,256 match `feature_manifest_laptime.json` exactly; "eleven feature ranges" matches the 11
entries of `_N06_TRAINED_BOUNDS`; the recalibrated stint bounds SOFT 2 / MEDIUM 7 / HARD 8 /
wet-fallback 6 match `src/strategy/inference/guard_rails.py:95-103`; the prescriptive/
proscriptive table and the Art. 55.17 framing match the shipped rail behaviour and the
2026-07-16 lesson. Two defects, both LOW: (a) "looked up from the featured parquet ... one
value per GP" — written for `36aa8d0`, not updated for `806cedd`'s (Year, GP) keying; ironically
R1 makes the stale sentence more truthful than the code's own docstring, since the served map
does hold one value per GP; (b) the envelope section's story that wiring the bound made
`mean_sector_speed` meaningful again is overtaken by R3: under the map actually shipped, that
bound cannot fire on any known race, so the page documents a detection power the system no
longer has.

### What I tried to break and could NOT

- **`_normalise_gp_key` collisions**: enumerated all 26 parquet spellings + all 26 metadata
  spellings, normalised, looked for many-to-one across DIFFERENT circuits — none. The only
  merges are the intended Miami repair and underscore/space variants of the same race.
- **Unresolvable spellings from LIVE producers**: every metadata name (71/71), every per-year
  featured GP_Name, and every MCP alias-table output resolves; the ASCII 'Montreal'/'Sao Paulo'
  misses have no producer that can reach `session_meta.gp_name` today.
- **Year poisoning on live paths**: all 71 metadata years are ints equal to their folder year;
  backend years are FastAPI-typed; MCP refuses unparseable years; the `or 2025` default is
  unreachable from current producers. (The string-year silent NaN stands as a latent trap only.)
- **The map itself**: 71 keys, no NaN anywhere in the source column (dropna is a no-op), every
  key backed by a raw race and vice versa; `(2025, 'Las Vegas')` present at 228.9645 — F2's
  hole is genuinely closed, and F1's 'Miami Gardens' and F3's 'Spain' both resolve on the
  season they belong to.
- **Vacuity of the walk-the-disk test**: could not make it pass while missing a race short of
  deleting `metadata.json` files that all exist.
- **Skip-guard over-skip**: could not find a test suppressed by `a3b876c` whose artefacts are
  actually present; the local run exercises everything except the one test that genuinely
  cannot run here.
- **`_compute_derived` stale positional callers**: none exist anywhere.
- **Holdout MAE**: reproduces (0.4097, within the golden's declared 0.01 of the 0.4104
  headline), on the eval path, which this branch did not touch.

### Round-2 fix list (ordered by value/risk)

1. **Decide, on purpose, which value 2025 laps get** (R1/R2): either serve the per-year
   artefact values (restore the `23a02ba` decision — load per-year files, or repair the
   combined build so its 2025 rows carry the 2025 measurements), or explicitly adopt
   training-constant serving and rewrite the loader docstring + `23a02ba`'s rationale. The
   measured stakes are small in aggregate (p95 0.0016 s) but the docstring, the eval tier, the
   envelope story and the test all currently assume a mechanism the data refutes. The combined
   artefact's 2025 `mean_sector_speed` being a training-era broadcast should get its own
   data-integrity issue either way (it contradicts `laps_featured_2025.parquet` on 22 GPs).
2. **Fix the loader docstring's false claims** (`pace_agent.py:334-341, 356-360`): the 4.8/18.4
   cross-year gap does not exist in the combined file; Monza 2025 at 317.24 is not served and
   nothing is outside the envelope.
3. **Make the envelope test assert the discrimination it promises** (R3): assert the outside
   set is NON-empty (it is empty today — the test should fail until item 1 is decided), or
   rewrite it to pin the actual serving and declare the bound non-discriminating.
4. **File the 2023 Spain/Barcelona duplicate race** (R4) as a data-integrity issue: same
   session_key 9102 twice in `data/raw` and in the combined artefact; decide which name stays,
   deduplicate the artefact, and re-check whether N06's training consumed the duplicate.
5. **Correct or retract the `36aa8d0` impact numbers** (R5) wherever they are quoted (commit
   history cannot change; the docs/issue #797 closure can): full-season honest numbers are
   mean +0.018 s, p95 0.177 s, 17.7% > 0.010 s; inference-shaped rows 9.5%.
6. **Feed `LapsSincePitStop` its own quantity** from the RSM (it has pit-stop history) instead
   of `tyre_life`, or document the 19.8%-divergence as an accepted approximation where the
   envelope comment currently only argues about double-reporting.
7. **Two-line docstring repairs**: `run()`'s "defaults to prev_speed_st" (`pace_agent.py:784`)
   and `_compute_derived`'s phantom `prev_speed_st` arg (`pace_agent.py:472`).
8. **Update the skip reason** at `tests/eval/test_ml_recompute_golden.py:61` to name the
   holdout parquet as well as the model.
9. LOW/latent hardening, take or leave: coerce `year` to `int()` inside
   `_resolve_mean_sector_speed` (kills the silent string-year NaN); add 'Montreal'/'Sao Paulo'
   ASCII aliases to the map normalisation or to `FOLDER_ALIASES`; update the docs' "one value
   per GP" once item 1 lands.
