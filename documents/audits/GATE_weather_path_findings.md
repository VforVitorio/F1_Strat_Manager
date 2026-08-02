# GATE — Weather path findings (issue #784, branch `refactor/single-source-race-state-builder`)

Adversarial verification gate, 2026-08-02. Read-only except this file; written incrementally.

Claims under test:
1. §F11's justification for the track_temp decision ("weather.parquet missing → keys absent → defaults fire" as a LIVE divergence) may be a false statement.
2. A live backend crash: `/lap-state` for 2025 emits `weather.air_temp/track_temp = None` (key present), and `backend/utils/race_state_builder.py:114-115` does `float(weather.get(...))` → `float(None)` TypeError.
3. A fifth undocumented default pair (25/40) at `endpoints/strategy.py:933-941`; inventory ALL default-temperature sites.

## Checklist

- [x] Weather parquet coverage 2023/2024/2025
- [x] Weather parquet row counts + NaN temp rows (present-with-None reachability)
- [x] `get_weather_state` key-emission behavior (code)
- [x] 2025 featured parquet columns, empirically
- [x] `augment_featured_laps` restores weather? (code + executed)
- [x] `Series.get` + `_safe_none` empirical behavior on real row
- [x] `build_race_state` crash reproduction on real data (true end-to-end)
- [x] Webapp / MCP reachability of the crash
- [x] Full default-temperature inventory (claim 3)

## Verdicts (summary)

- **Claim 1: CONFIRMED — the §F11 justification is a false statement in this checkout.** The
  decision (canonical 35.0) remains right for the median reason; the "reachable today" claim is not.
- **Claim 2: CONFIRMED — live, total outage of /recommend for 2025 races**, born 2026-07-18
  (#465 wave), predating #784. The branch's canonical builder fixes it (proven by execution).
- **Claim 3: CONFIRMED and EXTENDED** — the :933-941 site is real (25/40, a PRODUCER), and there
  are two MORE pairs nobody listed: debug_agent (28/45) and arcade overlays (18/45). Five distinct
  value pairs in the working tree.

## Evidence log (appended as confirmed)

### E1. Every race directory has a weather.parquet — the "missing file" premise of §F11 is counterfactual in this checkout

Executed (bash file count over `data/raw/`):

```
2023: 23 race dirs, 23 with weather.parquet
2024: 24 race dirs, 24 with weather.parquet
2025: 24 race dirs, 24 with weather.parquet
```

No race directory lacks the file. The only dirs without one are `data/raw/radio_audio/*` (not races).

### E2. Code path facts (cited, verified by reading this checkout)

- `src/simulation/replay_engine.py:80-90` — loads `weather.parquet` whenever it exists (degrades to `None` only on read failure, with a stderr warning); `:137` passes `self._weather_df` into `rsm.get_lap_state(lap, self._weather_df)` on every lap. Both CLI and Arcade replay go through this.
- `src/simulation/race_state_manager.py:470-472` — with `weather_df=None` OR an EMPTY frame (`not weather_df.empty` guard), the weather dict is `{"track_status": ...}` only → temperature KEYS ABSENT.
- `race_state_manager.py:478-482` — with a non-empty weather_df, keys are ALWAYS present; values become `None` only when the selected row's reading `pd.isna(...)` — present-with-None, which the consumer `.get(key, default)` does NOT catch.

So the default-firing branch on the replay path requires: weather.parquet missing, unreadable, or zero-row. E1 rules out "missing" for every race in the dataset. Row counts and NaN scan below.

### E3. Backend flow facts (cited)

- `backend/utils/laps_cache.py:29-40` (`get_laps_df`) DOES call `augment_featured_laps` — the CLAUDE.md "every consumer" rule holds here.
- `src/f1_strat_manager/laps_augment.py` `RAW_COLUMNS_TO_RESTORE = {"Time": "Time_s", "TrackStatus": "TrackStatus"}` — weather columns are NOT restored by augmentation.
- `/lap-state` producer `endpoints/strategy.py:574-583`: `weather = {"air_temp": _safe_none(r.get("AirTemp")), "track_temp": _safe_none(r.get("TrackTemp")), ...}` — keys unconditionally present.
- `backend/utils/race_state_builder.py:114-115`: `float(weather.get("air_temp", 25.0))` / `float(weather.get("track_temp", 35.0))` — default fires only on MISSING key. Lines 89-103 of the SAME FILE document this exact dead-default class for `position` (#465) and fix it there — but not for weather. The twin that never got the fix.
- Reachability chain 1 (webapp): `webapp/src/features/strategy/queries.ts:160-161` — `fetchLapState(...)` then `runRecommend(lapState, ...)`; `lib/api/strategy.ts:322` GETs `lap-state`, `:352` posts `lap_state: lapState` verbatim to `/recommend`; `endpoints/strategy.py:1349` feeds it to `build_race_state`. No sanitisation between.
- Reachability chain 2 (chat/MCP): `backend/mcp_tools.py:397-413` `_build_lap_state` delegates to the SAME `get_lap_state` producer; `:583-595` `recommend_strategy` feeds it to `build_race_state`.

### E4. Claim 2 empirics — CONFIRMED on real data (executed `uv run python`, script `gate_claim2.py`)

```
featured parquet columns (pre-augment):
  2023: 53 cols, 22106 rows, weather cols present: [AirTemp, TrackTemp, Humidity, Rainfall]
  2024: 53 cols, 23256 rows, weather cols present: [AirTemp, TrackTemp, Humidity, Rainfall]
  2025: 48 cols, 22760 rows, weather cols present: NONE          <- #782 confirmed
after augment_featured_laps(df25, 2025): 50 cols, weather cols: NONE
  columns ADDED by augmentation: ['Time_s', 'TrackStatus']       <- augmentation does NOT restore weather

Lusail 2025 NOR lap 30 (real row through the producer's exact code):
  r.get('AirTemp')            -> None   (pandas Series.get on a missing index label returns None)
  _safe_none(r.get('AirTemp')) -> None
  producer weather dict: {'air_temp': None, 'track_temp': None, 'humidity': None, 'rainfall': 0}
  'air_temp' in weather: True
  weather.get('air_temp', 25.0) -> None   <- the .get default does NOT fire (key present)
  float(weather.get('air_temp', 25.0))    -> TypeError: float() argument must be ... not 'NoneType'

2024 row (Austin): r.get('AirTemp') -> 28.1, TrackTemp -> 47.2   <- 2023/2024 paths get real floats
raw fallback data/raw/2025/Lusail/laps.parquet: 35 cols, weather cols: NONE
  <- the raw-parquet fallback branch of /lap-state ALSO emits None weather for 2025;
     there is no branch of this producer that yields a temperature for a 2025 race.
```

So for EVERY 2025 lap served by `/lap-state` (featured or raw fallback), `weather.air_temp` and
`weather.track_temp` are key-present-value-None, and the current backend builder's
`float(weather.get(...))` at `backend/utils/race_state_builder.py:114-115` raises TypeError.

Severity nuance: `/recommend` wraps the call in `except (KeyError, TypeError, ValueError)` at
`endpoints/strategy.py:1365-1367` -> HTTP 422 "orchestrator validation error", NOT a 500 traceback.
It is a total functional outage of /recommend for 2025 races (the year the webapp defaults to,
`year: int = 2025` at `:411`), presented as a client-input error.

### E5. Claim 2 — TRUE end-to-end executed (real producer -> real builder, no replication)

```
get_lap_state(gp='Lusail', driver='NOR', lap=30, year=2025)   [the real backend function]
  weather emitted: {'air_temp': None, 'track_temp': None, 'track_temp_start': None,
                    'humidity': None, 'rainfall': 0}
backend.utils.race_state_builder.build_race_state(that_lap_state)
  -> TypeError: float() argument must be a string or a real number, not 'NoneType'

CONTROL get_lap_state(..., year=2024): weather {'air_temp': 18.6, 'track_temp': 22.5, ...}
  -> RaceState built OK (air_temp=18.6, track_temp=22.5)

NEW canonical builder (src/agents/race_state_builder.py, this branch) on the SAME 2025 state:
  -> NO crash; air_temp=25.0, track_temp=35.0, rainfall=False
```

### E6. Claim 2 — when the crash was born (submodule git archaeology)

- `float(weather.get("air_temp"...))` builder shape: commit `ff44813` 2026-04-10.
- `_safe_none(r.get("AirTemp"))` in the /lap-state producer: commit `84b561e` 2026-07-18
  ("harden the strategy backend against the Fable audit", the #465 wave).
- BEFORE 84b561e the producer read `float(_safe(r.get("AirTemp", 25)))` /
  `float(_safe(r.get("TrackTemp", 40)))` (84b561e^ lines 485-486): for 2025 the missing
  column made `Series.get`'s default fire -> every 2025 lap got a FABRICATED 25/40, no crash.
- So: pre-2026-07-18 = silent fabrication bug; post-2026-07-18 = hard 422 outage. The crash
  PREDATES #784 (it is on `main` of the submodule today) and was INTRODUCED by the #465 fix,
  which honoured the None contract in the producer but never updated the consumer 100 lines
  away in the same repo — `backend/utils/race_state_builder.py:89-103` fixed exactly this
  class for `position` in the SAME commit wave and left `air_temp`/`track_temp` at :114-115
  untouched. Textbook twin-not-fixed.
- Blast radius: webapp Strategy tab pins `const YEAR = 2025` (`webapp/src/lib/api/strategy.ts:244`)
  and `RACE_YEAR = 2025` (`race.ts:17`) -> the DEFAULT year of the surface is the broken one.
  MCP chat `recommend_strategy` defaults `year: int = 2025` (`mcp_tools.py:570`) -> same.
  2023/2024 unaffected (featured parquets carry real weather columns, E4).

**CLAIM 2 VERDICT: CONFIRMED**, and it is a PRE-EXISTING live bug (born 2026-07-18, #465 wave),
not introduced by #784. The branch's canonical builder demonstrably fixes it (E5). Reproduction:
POST /api/v1/strategy/recommend with the verbatim output of GET /lap-state?gp=Lusail&driver=NOR&lap=30&year=2025
-> 422 "orchestrator validation error" on every 2025 lap of every 2025 GP.

### E7. Claim 1 — the replay-path default branch is UNREACHABLE with the shipped dataset

Executed scan of all 71 `data/raw/<year>/<gp>/weather.parquet` files (script `gate_claim1.py`):

```
scanned: 71 weather parquets; unreadable: 0; empty: 0
files with NaN/absent AirTemp:  NONE
files with NaN/absent TrackTemp: NONE
files whose FIRST row TrackTemp is NaN (track_temp_start=None risk): NONE
row counts: min 108, max 223
```

Combined with the code path (E2):

- Keys-ABSENT (the branch §F11 says "fires every default") requires weather.parquet missing,
  unreadable, or zero-row (`race_state_manager.py:472` guards `is not None and not empty`).
  In this checkout: 71/71 present (E1), 71/71 readable, 71/71 non-empty. **Unreachable.**
- Present-with-None (`:478-479`, NaN reading -> None) requires a NaN temperature in the selected
  row. Zero NaN readings exist in any of the 71 files. **Also unreachable today** — but F11's
  characterisation of THAT branch (crashes the pre-branch builders; `.get` default cannot catch
  it) is CORRECT as a hazard class, and the canonical builder's None-as-missing read fixes it.
- `replay_engine.py:82-90` treats weather as optional BY DESIGN (a future race dir without the
  file, a corrupt download, a live feed without weather would take the degraded branch), so the
  branch is latent-defensive, not dead code. But it is NOT "a real, reachable, surface-dependent
  model-input divergence today" on any of the 71 races the project ships. The CLI's 40.0 (dev
  `scripts/run_simulation_cli.py:1370-1371`) vs Arcade's 35.0 (dev `src/arcade/strategy.py:709-710`)
  never actually diverged on shipped data: on every race both received real temperatures.

**Where a 40-vs-35 divergence IS real today:** not on the replay path but between backend
endpoints for 2025 — `/pace-range` fabricates track_temp=40.0 (E8 site #6) while `/pace` via
/lap-state + the pace agent's None-safe default yields 35.0 for the same lap.

**Verdict:** the DECISION (35.0 canonical) is right and stands on the measured-median argument
(35.0 IS the dataset median, 40.0 matches nothing measured). The recorded JUSTIFICATION — "when
weather.parquet is missing for a race ... every default fires", presented as reachable today — is
counterfactual for every race in `data/`, exactly the failure mode CLAUDE.md §11 (2026-07-16)
already recorded for this same path. §F11's weather bullet and any echo of it in issue #784 need
rewording from "live divergence today" to "latent divergence on degraded/foreign inputs; the live
divergences are on the backend surfaces (E8)". The false premise has already propagated: the NEW
module's own docstring repeats it (`src/agents/race_state_builder.py:7`, "races with no weather
parquet").

### E8. Claim 3 — complete inventory of default air/track temperature sites

Legend: PRODUCER = fabricates a reading into a lap_state/feature source before the contract;
CONSUMER = fills a gap after reading it. None-safe = catches present-with-None; dead-.get =
fires on missing key only.

**Pair (25.0, 35.0):**
1. `src/agents/race_state_builder.py:78-79, 336-337` — canonical, this branch; CLI
   (`run_simulation_cli.py:1287-1305`) and arcade (`src/arcade/strategy.py:599-649`) now
   delegate here. CONSUMER, None-safe (`_weather_reading` :201).
2. `src/telemetry/backend/utils/race_state_builder.py:114-115` — CONSUMER, dead-.get -> the E5
   crash. Same file fixed this exact class for `position` at :89-103 and left weather untouched.
3. `src/agents/pace_agent.py:642-643` — CONSUMER, None-safe.
4. `scripts/prompt_ab/gen_inputs.py:65-66` — CONSUMER, None-safe (`or`).
5. `src/strategy/eval/decision_modes.py:304-305` — CONSUMER, None-safe (`or`).

**Pair (25, 40):**
6. `src/telemetry/backend/api/v1/endpoints/strategy.py:934-935` (`_build_lap_state_from_row`
   :890; sole caller `/pace-range` :695) — PRODUCER. For 2025 the columns are absent, so
   `Series.get`'s default DOES fire: every 2025 lap of the Lab pace chart runs the pace model
   with fabricated track_temp=40.0, while `/pace` uses 35.0 for the same lap (via the None-safe
   consumer #3). Also carries `_s` NaN->0 on 2023/24 NaN rows (none shipped).
7. (dev, pre-branch) `scripts/run_simulation_cli.py:1370-1371` — CONSUMER, dead-.get; replaced
   on this branch by #1.
8. (historical) the /lap-state producer until submodule commit `84b561e` 2026-07-18
   (`84b561e^` :485-486) — PRODUCER, fabricated 25/40 for every 2025 lap until #465 honoured None.

**Pair (28.0, 38.0):**
9. `src/agents/tire_agent.py:1440-1442` (run() path) — CONSUMER, dead-.get, float-wrapped.
10. `src/agents/tire_agent.py:1512-1514` (run_from_state) — CONSUMER, dead-.get, NO float wrap:
    a present-None (every 2025 backend lap_state) passes THROUGH the 28.0/38.0 default as None
    into the TCN feature dict — silent None/NaN feature, no crash. Same class as E5, different
    symptom.
11. `src/agents/race_situation_agent.py:681-684` — CONSUMER of session_meta (backend
    session_meta carries no temps -> fires for real; `track_temp_start` cascades to 38.0 ->
    track_temp_delta=0, the #486 residue).
12. `src/agents/race_situation_agent.py:1316-1319` (run(), FastF1 path) — PRODUCER-side
    fabrication when the live weather frame lacks the column.
13. `src/agents/race_situation_agent.py:1402-1405` (run_from_state) — CONSUMER, dead-.get,
    same None-passthrough as #10.
14. `src/telemetry/backend/api/v1/endpoints/strategy.py:1022-1025` (`/tire-range`) — PRODUCER:
    UNCONDITIONALLY fabricates `agent.session_meta` AirTemp=28.0/TrackTemp=38.0 for EVERY year,
    even 2023/24 where real readings exist in the very gp_df scoped two lines above.

**Pair (28.0, 45.0)** — MISSED by the known-so-far list:
15. `scripts/debug_agent.py:179-180` and `:306-307` — CONSUMER (debug harness). track 45.0
    matches nothing else in the repo.

**Pair (18.0, 45.0)** — MISSED by the known-so-far list:
16. `src/arcade/overlays.py:129-130` — CONSUMER, display-only (weather HUD): shows 45.0 C track /
    18.0 C air when a frame lacks weather. Never reaches a model.

**Count: FIVE distinct value pairs in the working tree** — (25,35), (25,40), (28,38), (28,45),
(18,45). The claim's ":933-941 is a fifth set" is right that the site is real and undocumented,
but its pair (25/40) DUPLICATES the old CLI pair; the genuinely new pairs are #15 and #16.
Producer sites (#6, #14, #12) fabricate before the contract and are NOT #784's scope (a builder
cannot undo an upstream fabrication). Consumer site #2 (the crash) IS fixed by #784; #10/#13
need the same None-as-missing treatment or an upstream weather restore.

**The #486 history, briefly:** N14's `track_temp_delta = track_temp - track_temp_start` is its
5th-most-important feature. `track_temp_start` used to live only in session_meta while the
consumer read `wx['track_temp_start']`, so it fell back to "current track_temp" and the delta
was 0.0 on every lap — a live feature reading a constant. The fix shipped it in the weather dict
at every producer (`race_state_manager.py:483-499`, `endpoints/strategy.py:577-580, 936-938`).
The 38.0 at `race_situation_agent.py:684` is the residual session_meta-path default — and on
2025 backend paths `_session_track_temp_start` (`endpoints/strategy.py:785`) returns None (no
TrackTemp column), so the #486 symptom (delta=0.0) is BACK on the season the webapp serves, via
the same missing-columns root cause as E4 (#782).

## What I tried to break and could NOT

1. **The canonical builder's None handling** — fed it the real, crashing 2025 lap_state (E5):
   built RaceState with 25.0/35.0/rainfall=False. No counterexample found.
2. **The "augmentation restores weather" escape for claim 2** — executed `augment_featured_laps`
   on the real 2025 parquet: adds only Time_s + TrackStatus. The CLAUDE.md "every consumer calls
   augment" rule IS honoured by `laps_cache.py:24-40`; it just does not restore weather.
3. **The raw-fallback escape** — all 24 `data/raw/2025/*/laps.parquet` lack AirTemp/TrackTemp
   (executed scan): no /lap-state branch can produce a 2025 temperature.
4. **A sanitising layer between /lap-state and /recommend** — `queries.ts:160-161`,
   `lib/api/strategy.ts:322,352`, `mcp_tools.py:397-413,583-595`: the dict passes verbatim on
   both surfaces.
5. **Claim 1's rescue** — searched for ANY missing/empty/corrupt weather.parquet, any NaN
   temperature, any zero-row frame across all 71 races (executed): none. I could not make the
   replay-path weather defaults fire with shipped data.
6. **A 2023/2024 instance of the claim-2 crash** — those featured parquets carry real weather
   values (control: Lusail 2024 18.6/22.5 builds fine); not reproducible there.

## Recommendations for #784

1. **Correct §F11's weather bullet and any issue text quoting it**: the replay-path divergence is
   latent (degraded/foreign inputs), not "reachable today"; the 35.0 decision stands on the
   median argument alone. Also fix the same premise in the new module's docstring
   (`src/agents/race_state_builder.py:7`).
2. **Upgrade #784's backend claim**: migrating the backend builder is not just deduplication —
   it FIXES a live 422 outage of /recommend for every 2025 lap on the webapp Strategy tab
   (YEAR=2025 pinned, `lib/api/strategy.ts:244`, `search.ts:22`) and MCP chat (default year
   2025). That deserves its own bug issue (pre-existing, born in the #465 wave, 2026-07-18)
   closed by the #784 PR, with the E5 reproduction as the body.
3. **File separately (NOT #784 scope):** (a) /pace-range's 25/40 producer fabrication (site #6)
   and /tire-range's unconditional 28/38 session_meta (site #14); (b) the None-passthrough in
   tire/race_situation run_from_state (sites #10/#13) silently degrading every 2025 backend
   call; (c) the #486 regression-by-data (track_temp_delta=0 on 2025). All three share the root
   cause of E4 — the featured-2025 parquet's missing weather columns (#782); restoring those
   columns (or augmenting them from weather.parquet) would heal (a), (b), (c) AND reduce the
   blast radius of any future dead-default twin.
4. Sites #15/#16 (debug 28/45, HUD 18/45) are cosmetic; one-line follow-up at most.


