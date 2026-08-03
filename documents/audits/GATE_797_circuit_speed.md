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
