# PR 3 — the gp_name keyspace, swept across every consumer

**Date:** 2026-08-04 · **Extends** `GATE_DATA_WIRING.md` (W-F14, W-F6) and
`GATE_801_ARTEFACTS.md` §3. Those are dated findings and stay as written; this file records
what the pre-implementation sweep measured on top of them.

## Why a sweep before the fix

The gate named four gp_name-keyed lookups. This is the fourth time the same defect class has
surfaced (#448, #450, #797, now this), and every previous fix repaired the site that hurt and
left the others. So the question asked before writing any code was not "is Miami broken?" but
**"which races, and which lookups, in total?"**

## Scope, measured

### Races

Every `data/raw/<year>/<race>/metadata.json` gp_name against every gp_name-keyed table:

| table | keys | race dirs that do not resolve |
|---|---|---|
| `tire_compounds_by_race.json` (per year) | 22 / 24 / 24 | `2023 Spain`, `2025 Miami Gardens` |
| `circuit_clusters_k4_2025`, `laps_featured_2025`, `laps_tiredeg` | 24 | `2023 Spain`, `2025 Miami Gardens` |
| `circuit_clusters_k4`, `circuit_features_with_clusters_k4` | 25 | `2023 Spain` |
| `laps_featured.parquet` (combined) | 26 | none |

**2 of 71 race dirs**, and `2023 Spain` is not a spelling mismatch: it is the duplicate folder
of `2023 Barcelona` (same OpenF1 session 9102), which PR 6 removes. **The only genuine
keyspace mismatch is Miami 2025**, where the artefacts say `Miami` and `metadata.json` says
`Miami Gardens`.

A first pass of this sweep reported `tire_compounds_by_race` as 6 keys and 71 misses. That
probe read the file flat; it is nested by year. The table above calls the real consumers.

### Consumers

Probed with the name the replay path actually passes (`metadata.json["gp_name"]`, per
`replay_engine.py:114`), for the compounds Miami 2025 genuinely ran — it ran no SOFT, and on
SOFT the pit agent's fallback is 5 while Miami's SOFT is C5, so a SOFT-only probe reports a
miss as a hit:

| consumer | `'Miami Gardens'` | `'Miami'` |
|---|---|---|
| `tire_agent._compound_name_to_id` MEDIUM / HARD | `C2` / `C1` | **`C4` / `C3`** |
| `pit_strategy_agent._compound_to_id` MEDIUM / HARD | `3` / `1` | **`4` / `3`** |
| `race_situation_agent._abs_compound` MEDIUM / HARD | `'MEDIUM'` / `'HARD'` | **`C4` / `C3`** |
| `pace_agent._session_median` MEDIUM / HARD | `None` / `None` | **91.419 / 91.228** |
| `pace_agent._encode_categorical` (cluster) | 1 (default) | 1 (real) |
| `TireAgentConfig.circuit_cluster_map` | 1 | 1 |

**Five broken consumers, not four.** `race_situation_agent._abs_compound` reads the same JSON
and was not in the gate's list; its failure mode differs from the other two — it returns the
RELATIVE name (`'HARD'`) where the caller expects a `Cx` string.

Then a second pass over the whole tree — every `.get(gp_name`, every `GP_Name ==` mask —
found **four more**, which is why this section is written after the fix rather than before it:

| site | what an unresolved name does |
|---|---|
| `strategy/inference/engine.py:144` | the scoped mask comes out empty and the guard falls back to **the unscoped season frame** — the whole race runs against all 24 GPs, which is the #429/#448 regression that fallback's own warning names |
| `nlp/radio_runner.py:402` | no driver-code map, so radio speakers degrade to synthetic `D{n}` labels |
| backend `strategy.py:1028,1031` | the third `session_meta` builder, raw `.get` again — the same twin the PR 2 gate caught for `cluster_mean_lap_s` |
| backend `telemetry.py:332` | 404s on a GP whose data is present; the parameter's own description promises the alternate form is accepted |

Two sites were examined and deliberately left alone: `laps_augment.py:100` already resolves
the rename through its own `_FRIENDLY_TO_FOLDER` table (friendly→folder, the other
direction), and the backend's remaining `df["GP_Name"] == gp` masks are fed from
`available_gps`, which returns that same frame's `unique()` — they match by construction.

The two cluster lookups resolve today, each for its own reason, and neither is safe:

- `pace._encode_categorical` misses and falls back to cluster 1, which happens to be Miami's
  real cluster. Coincidence, not correctness.
- `TireAgentConfig.circuit_cluster_map` hits, because **the pooled clustering artefacts hold
  Miami twice** — `Miami` and `Miami Gardens` are two rows of `circuit_clusters_k4.parquet`
  (25 rows for 24 circuits), both cluster 1. The same duplicate-circuit defect as
  Spain/Barcelona, in the clustering. When PR 6 de-duplicates the artefacts, this lookup
  starts missing unless it is normalised now.

### Spellings in play

Six, and no single resolver covers them:

| spelling | source | `slug_from_event_name` | `normalise_gp_key` |
|---|---|---|---|
| `Miami` | parquet / JSON | `Miami` | `Miami` |
| `Miami Gardens` | `metadata.json` | `None` | `Miami` |
| `Miami_Gardens` | folder name, used when `metadata.json` is absent | `None` | `Miami` |
| `Miami Grand Prix` | FastF1 event name | `Miami` | `Miami_Grand_Prix` ✗ |

`slug_from_event_name` handles FastF1 names and fails on the metadata form;
`normalise_gp_key` handles the metadata and folder forms and mangles the FastF1 one. The chain
must try both, which is what `_resolve_mean_sector_speed` already does since #797.

## The fix

One resolver, in the module whose job this is:

- `gp_slugs.normalise_gp_key` — moved out of `pace_agent` (it was never pace-specific).
- `gp_slugs.resolve_gp_key(keys, gp_name)` — returns the spelling of `gp_name` present in
  `keys`, trying raw → slug → normalised → normalised slug, and returning `gp_name` unchanged
  when none is present, so every existing fallback still fires exactly as before.

Applied at every call site above — thirteen in the parent repo and the backend submodule.
Where a config already owned the map, the resolution went into a `cluster_for()` method next
to the `sc_rate_for()` / `traversal_for()` that were already doing this for their own tables,
rather than repeating the call at each of the six raw `.get(gp_name, 0)` sites.

Query-side only: no lookup table is re-keyed, because re-keying a map that holds both `Miami`
and `Miami Gardens` would silently drop one of them.

`FOLDER_ALIASES` gains `'Spain': 'Barcelona'`. That folder is the duplicate, not a rename, so
the entry is a stopgap: it makes the folder replayable today and goes inert when PR 6 removes
it. The alias only fires where `'Spain'` is not itself a key, so the combined
`laps_featured.parquet`, which stores both, is unaffected.

## Verification

`tests/agents/test_gp_keyspace.py` enumerates **every race directory on disk against every
consumer**, rather than asserting Miami. That is the difference between this fix and the three
before it: a fifth mismatch, in any season, fails the suite instead of shipping.

Before / after, `2025 Miami Gardens`:

| | before | after |
|---|---|---|
| `_compound_name_to_id('HARD', ...)` | `C1` | `C3` |
| `_compound_to_id('HARD', ...)` | `1` | `3` |
| `_abs_compound('HARD', ...)` | `'HARD'` | `C3` |
| `_session_median('HARD', ...)` | `None` | `91.228` |
