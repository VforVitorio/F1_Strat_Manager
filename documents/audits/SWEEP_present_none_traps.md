# SWEEP — present-`None` traps (`dict.get(k, default)` / `Series.get(k, default)` / `x.get(k) or DEFAULT`)

**Date:** 2026-08-02 · **Branch:** `refactor/single-source-race-state-builder` · **Scope:** parent repo + `src/telemetry` submodule.
**Role:** adversarial gate + inventory. No repository file modified except this report.

## Bug class

`dict.get(key, default)` fires its default only when the KEY is missing. The producers below deliberately emit
unmeasured readings as key-present-holding-`None`, so a consumer's `.get(k, default)` returns `None` and the
default silently never fires. Confirmed prior consequences: #788 (crash, `/recommend` 422 on every 2025 lap) and
the tire_agent cliff estimate moving 2.3 laps optimistic.

## Producers that emit present-`None` (verified at file:line)

| Producer | Keys that can be `None` |
|---|---|
| `src/simulation/race_state_manager.py:282` `get_driver_state` | `lap_time_s`, `prev_lap_time`, `sector1_s..3_s`, `position`, `gap_to_leader_s`, `compound_id`, `tyre_life`, `stint`, `stint_baseline_tyre_life`, `speed_i1/i2/fl/st`, `fuel_load` |
| `src/simulation/race_state_manager.py:376` `get_rival_states` | `position`, `lap_time_s`, `tyre_life`, `stint`, `speed_st`, `gap_to_leader_s`, `interval_to_driver_s` |
| `src/simulation/race_state_manager.py:446` `get_weather_state` | `air_temp`, `track_temp`, `humidity`, `wind_speed`, `track_temp_start` (note: `rainfall` defaults to `False`, never `None`; keys absent entirely when `weather_df is None`) |
| `src/telemetry/backend/api/v1/endpoints/strategy.py:372` `_safe_none` | backend lap-state fields: TyreLife, weather (AirTemp/TrackTemp/Humidity), SpeedST, Position |

## Method

AST-based scan (not grep): every two-arg `.get(...)` call, every `x.get(k) or DEFAULT`, and every
`float(...)/int(...)` wrapping a `.get(...)`, across `src/agents/`, `src/simulation/`, `src/arcade/`,
`src/strategy/`, `scripts/`, `src/f1_strat_manager/`, and the whole `src/telemetry/backend/` submodule.
Each candidate then verified by hand against the producer that feeds it; reachability claims verified by
executing Python against the shipped parquets / real producers.

## Findings

<!-- appended incrementally as confirmed -->

### Batch 1 — code-level confirmations (reachability experiments follow in Batch 2)

**F1. `src/agents/tire_agent.py:1479` — `tyre_life = d.get('tyre_life', 1)` in `run_from_state` — the unfixed twin INSIDE the just-fixed function.**
The same `run_from_state` that this branch migrated to `reading_or_default` for the four weather keys (lines 1518-1521) still reads
`tyre_life` with the two-arg get four lines above. Producers that emit `tyre_life` as present-`None`: RSM `get_driver_state`
(race_state_manager.py:354, NaN -> None) and the backend `/lap-state` builder (strategy.py:491 `_safe_none(r.get("TyreLife"))`).
The `None` flows into `_run_core(driver, compound_id, tyre_life, gp_name)` -> prompt "tyre life None laps" (LLM path), and on the
no-LLM path `_get_driver_stint(driver, None)` evaluates `self.laps_df['TyreLife'] <= None` (tire_agent.py:1083) -> TypeError.
Guards that DO exist upstream: the backend simulator skips such laps (`_lap_skip_reason`, simulator.py:279) and the CLI mirrors it —
but `/recommend` (backend strategy.py:1317) and MCP `recommend_strategy` (mcp_tools.py:566) have NO such guard: they only refuse
position-None laps (via `build_race_state`'s ValueError). Consequence: CRASH (422/500 on /recommend) or LLM-mediated SILENT-WRONG,
IF a shipped row has Position present and TyreLife NaN — measured in Batch 2.

**F2. `src/strategy/inference/no_llm.py:135-136` — `compound = d.get("compound", "MEDIUM")` / `tyre_life = d.get("tyre_life", 1)` — second twin pair.**
`_tire_no_llm` re-derives the same two values `run_from_state` derives (its own docstring says so), with the same two-arg get.
`tyre_life=None` is pre-bound into the tool args and `_NullReActRunner.invoke` (no_llm.py:92-99) executes `tool.invoke(kwargs)` with
no exception handling -> pydantic tool-args validation error or the same `<=` TypeError. `compound=None` would crash one line later
at `compound.startswith("C")` (AttributeError), but no current producer emits compound as None (RSM: `str(...)` at
race_state_manager.py:352; backend: `str(...)` at strategy.py:489/912) — compound-None is LATENT (hand-built lap_state only);
tyre_life-None is as reachable as F1 on any no-guard no-llm path.

**F3. `src/agents/race_situation_agent.py:918` + `src/agents/pit_strategy_agent.py:1053-1067` — the recorded CLAUDE.md §11 `Series.get` sites are still live expressions.**
`float(x_row.get('Position', 10))`, `float(x_row.get('TyreLife', 10))` (pit, N16 undercut features) and
`float(driver_x_lap.get('SpeedST', 300.0))` (situation, N12 overtake features) read pandas Series rows: a stored NaN is returned
as NaN, never the default, and `float(NaN)` = nan flows into the LightGBM feature frame (LightGBM treats NaN as missing — model-
defined branch, silently). The #462 liveness guard prevents the retired-car case, but NaN on a LIVE car's row is unguarded.
Adjacent and harder: race_situation_agent.py:914-915 `int(driver_x_lap['TyreLife'])` raises ValueError on NaN — a crash site
one line above the silent one. Reachability measured in Batch 2 (NaN counts on featured frames).

**F4. Backend has TWO lap-state producers and they answer "missing weather" differently.**
`/lap-state` (strategy.py:574-583) emits `air_temp/track_temp/humidity` via `_safe_none` -> honest present-`None` (the #788 shape).
`_build_lap_state_from_row` (strategy.py:890-949, used by /pace-range at line 695) emits
`float(_s(row.get("AirTemp", 25)))` (25/40/50 fabricated when the 2025 columns are ABSENT — Series.get default fires on a missing
column) and, worse, `_s` coerces a present NaN to **0** -> `air_temp: 0.0` degC on any 2023/24 row with NaN weather. Also
`speed_st`: `_safe_none` on /lap-state (None) vs `_s(...,0)` here (0.0). Only the pace agent consumes this second producer today,
and its `d.get('speed_st') or 300.0` happens to absorb the 0.0 — the 25-vs-None weather divergence feeds
`run_pace_agent_from_state`'s `reading_or_default(wx,'air_temp',25.0)` equally (default==fabrication, coincidentally identical),
so TODAY the consequence is nil; the trap is structural (next consumer of this producer inherits fabricated readings).

**F5. `src/telemetry/backend/mcp_tools.py:599` — `driver_state.get("gap_ahead_s") or GAP_UNKNOWN_FALLBACK_S` — #633-class zero-swallow, or-form inventory.**
The /lap-state producer ALWAYS emits `gap_ahead_s` as a float and uses `0.0` both for "leader, honest" (drv_pos==1) and for
"cum-times unresolved" (strategy.py:472-476). The `or` then rewrites the LEADER's honest 0.0 into the 2.0 fallback, so every MCP
`recommend_strategy` call for the race leader hands the orchestrator a fabricated 2.0 s gap to a car that does not exist.
Reachable today on any P1 request. SILENT-WRONG (prompt + MC read gap_ahead_s), or-form class.

**F6. `strategy_orchestrator.py:1874/1875/1888/1891` — flat-lap_state two-arg gets (`laps_since_pit`, `fuel_load`, `prev_speed_st`, `humidity`) — LATENT.**
These live in `_run_always_on_agents` (the FastF1 flat path). Its only production caller is `run_strategy_orchestrator`
(strategy_orchestrator.py:2269), which no shipped surface calls (grep: only notebooks/tests); every surface routes through
`run_lap`/`_run_always_on_agents_from_state`, whose adapters guard. If a caller ever feeds the flat path a dict carrying
`humidity: None` etc., the None reaches `run_pace_agent` unguarded. Latent, inventory only.

### Batch 2 — executed evidence (venv python, pandas 2.3.3, langgraph 1.2.5)

**E1. Semantics, proven in this environment:**
- `pd.Series([1.0,5.0,nan]) <= None` -> `[False, False, False]` — NO TypeError in pandas 2.3.3. So `_get_driver_stint`'s
  `TyreLife <= None` (tire_agent.py:1083) yields an EMPTY window -> tool returns "No laps found" -> `_conservative_stub`.
  F1's consequence is therefore SILENT degradation to the stub, not a crash, on this pandas.
- `Series.get('SpeedST', 300.0)` with stored NaN -> returns `nan` (default dead); with ABSENT key -> default fires. (CLAUDE.md §11 confirmed.)
- LangChain `@tool ... tyre_life: int` invoked with `tyre_life=None` -> `ValidationError` (proven). That is the no-llm
  `_tire_no_llm` path's failure mode — but both no-llm surfaces (CLI, /simulate) guard tyre_life-None laps out first.
- `int(float('nan'))` -> `ValueError: cannot convert float NaN to integer`.
- langgraph 1.2.5 `_default_handle_tool_errors`: returns a message ONLY for `ToolInvocationError` (arg validation);
  **any exception raised inside the tool body is RE-RAISED**. A ValueError inside `predict_overtake_tool` therefore
  aborts the whole ReAct invoke — on `/recommend` that surfaces as a 422 (`except (KeyError, TypeError, ValueError)`
  at backend strategy.py:1365), the exact #788 failure shape via a different field.

**E2. Shipped-data holes (the reachability driver), measured:**
- `laps_featured_2025.parquet`: 48 cols vs 53 in 2023/2024. Missing: `AirTemp, TrackTemp, Humidity, Rainfall` (the #782/#788
  root) **plus `lap_time_pct_of_race_fastest`** — the latter is recomputed unconditionally by its consumers
  (tire_agent.py:752, eval scripts rebuild it), so no extra blast radius from that column.
- **`TyreLife` NaN with Position PRESENT: 451 rows in featured 2025** — Miami laps 4-24 (379 rows, 19 drivers = the whole
  field), Spa-Francorchamps laps 28-44 (70 rows: ANT/ALO/HUL/COL/SAI), Melbourne 42-43 (BEA). `Stint` NaN on 379 of them.
  Featured 2023: 35 rows (TSU, Montreal 36+). Featured 2024: ZERO. Raw frames: 561 rows across 79,032; plus 101 Position-NaN
  and 6,872 SpeedST-NaN raw rows (featured: 1,889-2,068 SpeedST NaN per season). Raw Compound NaN/empty: 538 rows.
- All 71 `data/raw/<year>/<race>/weather.parquet` files: **zero NaN in every column** (confirms race_state_builder's claim and
  makes the arcade weather-panel trap latent with shipped data).

**E3. The concrete reachable scenario for the race_situation crash:** Spa-Francorchamps 2025, laps 31-33: COL runs P19 with
TyreLife NaN in BOTH the raw and featured frames while HAD sits P20 with a clean row (TyreLife 11/12/13). HAD passes every
surface's own-driver guard; N27 derives rival_ahead=COL; `predict_overtake_tool` reads COL's row and
`int(driver_y_lap['TyreLife'])` (race_situation_agent.py:915) raises ValueError. no-llm: `_situation_no_llm` invokes the tool
directly -> per-lap ERROR event. rich: langgraph re-raises -> lap aborts (422 on /recommend). At Miami 2025 laps 4-24 the
OWN-driver row is NaN for 19 drivers, so /recommend and MCP (no guard) hit the same ValueError via driver_x for nearly any
driver/lap in that window; the three replay surfaces instead skip those laps wholesale (INCOMPLETE), which is its own
degradation: ~21 laps of Miami 2025 produce no strategy call at all.

### Correction to F3 (pit half) — the pit agent is GUARDED; race_situation is the twin that never got the fix

`pit_strategy_agent.py:983-996` refuses undercut scoring when Position OR TyreLife is NaN on either car (its comment even
cites the same 561-row measurement reproduced above), so the `.get('Position', 10)` defaults at 1053-1067 are documented
dead code, not a live trap. **`race_situation_agent._build_overtake_features` has NO equivalent guard**: `predict_overtake_tool`
(race_situation_agent.py:1112-1154) checks lap range and liveness only, then `int(driver_x_lap['TyreLife'])` at :914-915
crashes and `float(...get('SpeedST', 300.0))` at :917-918 silently feeds NaN. One twin fixed, the other not — the recorded
CLAUDE.md §11 / twin-that-never-got-the-fix shape, live today.

## Inventory

### 1. Reachable today with shipped data

| # | Site | Consequence | Evidence |
|---|---|---|---|
| R1 | `race_situation_agent.py:914-915` `int(driver_x_lap['TyreLife'])` (and `driver_y`) | **CRASH** — ValueError aborts the lap; 422 on `/recommend`, ERROR event on `/simulate` no-llm and CLI --no-llm; rich replay surfaces abort the lap too (langgraph re-raises) | E3: Spa 2025 laps 31-33 HAD-behind-COL (all surfaces); Miami 2025 laps 4-24 any driver via /recommend and MCP (E2: 451 featured rows) |
| R2 | `tire_agent.py:1479` `d.get('tyre_life', 1)` + `no_llm.py:136` twin | **SILENT-WRONG** — prompt reads "tyre life None laps"; window mask empties (E1) -> conservative stub presented every affected lap; backend also emits `stint: 0` (`_safe`, strategy.py:492/915) which empties the stint filter regardless | Reachable via `/recommend` + MCP `recommend_strategy` (no guard); same 451 rows. Replay surfaces guarded (CLI run_simulation_cli.py:1596-1616, arcade strategy.py:399, simulator.py:258-283) |
| R3 | `race_situation_agent.py:917-918` `float(Series.get('SpeedST', 300.0))` | **SILENT-WRONG** — stored NaN (never the 300.0) into N12's `speed_trap_delta`; LightGBM routes it down the missing branch | 1,889 SpeedST-NaN rows in featured 2025 (E2); reaches the model whenever the paired rows survive the R1 crash line (both TyreLife present, one SpeedST NaN) |
| R4 | `mcp_tools.py:599` `driver_state.get("gap_ahead_s") or GAP_UNKNOWN_FALLBACK_S` | **SILENT-WRONG (or-form, #633 class)** — /lap-state emits the LEADER's honest `gap_ahead_s: 0.0` (strategy.py:472); `or` rewrites it to 2.0, a fabricated car ahead of P1 | Reachable on any MCP recommend for the P1 driver; producer always emits the key as a float |
| R5 | Featured-2025 data holes beyond weather | TyreLife/Stint NaN holes (Miami 4-24 field-wide, Spa 28-44, Melbourne 42-43) are the reachability driver for R1/R2 and also cost the replay surfaces ~21 laps of Miami outright | E2; #782's blast radius is wider than the four weather columns |

### 2. Latent (unguarded read, no current producer emits the value)

- `no_llm.py:135` + `tire_agent.py:1478` + `pit_strategy_agent.py:1460` `compound` present-None -> `.startswith`/`.upper()` AttributeError. Both producers emit `str(...)` (RSM :352, backend :489/:912) — only hand-built lap_state JSON on /recommend reaches it.
- `strategy_orchestrator.py:1874/1875/1888/1891` flat-path two-arg gets (`laps_since_pit`, `fuel_load`, `prev_speed_st`, `humidity`) — the FastF1 flat entry `run_strategy_orchestrator` has no production caller (grep: notebooks/tests only).
- **`src/arcade/data.py:597-603` + `src/arcade/overlays.py:129-134` — the weather panel trap, with a docstring that lies**: `_weather_row_to_dict` deliberately emits present-`None` on a NaN FastF1 weather sample and its docstring claims the panel's `.get(key, default)` default then fires — it does not; `f"{None:.1f}"` raises TypeError in `WeatherPanel.draw`. Latent only because shipped weather has zero NaN (E2); a live FastF1 feed with one NaN sample crashes the arcade overlay.
- `tire_agent.py:1440-1448` (FastF1 `run()` path): `session.weather_data.mean()` of an all-NaN column -> stored NaN via `Series.get('AirTemp', 28.0)` -> `float(nan)` into session_meta. Notebook/FastF1 path only.
- `tire_agent.py:586` `_add_weather_cols` `session_meta.get(col, 0.0)` — every current caller sets all four keys (run :1440, run_from_state :1518, /tire-range :1022); a future caller that omits one fabricates 0.0 degC.
- `scripts/debug_agent.py:306-308` `lap_state["weather"].get("air_temp", 28.0)` — debug harness, same trap shape if fed RSM weather with None.
- `race_situation_agent.py:681-684` + `pit_strategy_agent.py:892-894` session_meta two-arg gets — all production adapters populate these guarded; FastF1/hand-built paths could pass None through.
- Backend `_build_lap_state_from_row` (strategy.py:890-949): fabricates 25/40/50 weather when columns are absent and `_s` coerces present NaN to 0 (`air_temp: 0.0`); today only /pace-range consumes it and the pace agent's guards/defaults happen to absorb everything — the next consumer inherits fabricated readings. (F4.)
- Arcade dashboard formatters (`agent_formatters.py:94-98/146-150/206-210`, `reasoning_tabs.py:120-165`): two-arg gets over dataclass-serialized outputs whose fields are non-Optional floats today; `TireOutput.current_tyre_life=None` (from `_conservative_stub` under R2) would render as the string "None" — cosmetic.
- CLI display gets (`run_simulation_cli.py:1258-1262/1803-1805`): `rival_data.get("position", "?")` renders `None` instead of `?` for a position-less rival — cosmetic only (RSM sorts them to the back).

### 3. Harmless / correctly guarded (verified, do not re-audit)

- `src/agents/race_state_builder.py` — every read guarded: `_weather_reading` handles present-None; position None raises by design; `tyre_life`/compound normalised; `rainfall` `bool(None)` -> False is the honest degradation. Verified line by line.
- `pace_agent.py:626-704` (`run_from_state`) — the migrated site is correct; every remaining read is `or`-form or deliberately honest (`position` passes None to XGBoost's native-missing path). `tyre_life or 1` cannot swallow a real 0: TyreLife==0 occurs zero times in 2023-2025 (race_state_builder.py:77 measurement).
- `race_situation_agent.py:1406-1420` + `tire_agent.py:1518-1521` — the two migrated weather blocks are correct; `wx.get('track_temp_start') or 38.0` correctly None-safe (0.0 degC track temp does not occur in the dataset).
- `pit_strategy_agent.py:983-996` — NaN guard makes the 1053-1067 Series.get defaults dead code (documented as such in-file).
- `race_state_manager.py` `r.get(...)` sites — all wrapped in `pd.notna(...)` ternaries; this IS the producer's honest-None pattern.
- `position_projection.py` — all `(x or {})` chains; `_finite_or_none` collapses None/NaN/inf at the boundary (strategy_orchestrator.py:913-928).
- `strategy_orchestrator.py:829-830/1163` — `or`-guarded context reads; `rival.get("is_pitting", False)` wrapped in `bool()` so the backend's present-None is read as False (honest: unknown != pitting is a documented degradation).
- Static-map/env/display gets across arcade dashboard, CLI tables, backend chat/llm_service, eval CLIs, radio/NLP/rag payloads — defaults on dicts the same module builds, or pure display.
- `simulation/__main__.py`, `replay_engine.py`, `stint_history.py:144` — display/string sites, `or`-guarded where numeric.
- Guards confirmed present and equivalent on all three replay surfaces: CLI (:1596-1616), arcade (strategy.py:394-402), backend simulator (:258-283).
- eval-only `or`-forms (`decision_modes.py:301-305`, `prompt_ab/gen_inputs.py:60-66`): `gap_ahead_s or 2.0` would swallow a legitimate 0.0 gap, but their states come from RSM `get_driver_state`, which never emits `gap_ahead_s` at all (key absent) — #633-class only if ever fed backend states; tyre_life handled the honest way in decision_modes (:302).

## The three already-fixed sites: verdict

`pace_agent.py:647-649`, `tire_agent.py:1518-1521`, `race_situation_agent.py:1406-1408` all correctly route through
`reading_or_default`, which handles both absent-key and present-None (read and verified). The helper's docstring is accurate.
**But the fix is incomplete IN THE SAME FUNCTIONS**: `tire_agent.run_from_state` still reads `tyre_life` (:1479) with the
two-arg get 39 lines above its fixed weather block, and the same producers that motivated the weather fix emit `tyre_life`
as present-None on 451 shipped 2025 rows. `race_state_builder` guards `RaceState`, but the sub-agent adapters read the raw
`lap_state` directly, so the builder's `UNKNOWN_TYRE_LIFE` never protects them.

## What I tried to break and could not

- `race_state_builder.build_race_state`: tried weather-None, rainfall-None, compound "nan"/None, tyre_life None, missing
  lap_number/total_laps, empty rivals — every path lands on a guard, a warning, or the deliberate position ValueError.
- The `pd.Series <= None` crash I expected for `_get_driver_stint`: pandas 2.3.3 returns all-False instead of raising
  (executed), so that specific crash claim in my own Batch 1 draft did not survive its own verification.
- `pit_strategy_agent` undercut scoring with NaN rows: the :983 guard refuses before any NaN reaches the model.
- The five missing 2025 featured columns beyond weather: `lap_time_pct_of_race_fastest` is recomputed by every consumer.
- Shipped `weather.parquet` NaN (would arm the arcade panel trap and the RSM weather-None path): zero NaN in all 71 races,
  every column.
- `rainfall`: no producer emits it as None (RSM: False; backend: int; builder wraps in bool()).

## Could not determine

- Whether the LIVE FastF1 feed (arcade SessionLoader, `session.weather_data`) ever carries NaN samples — shipped parquets
  do not, but they are a different artifact from the FastF1 cache the arcade reads; the panel trap's reachability in a live
  session is unmeasured.
- Rich-mode LLM behaviour when the prompt says "tyre life None laps" (what integer the model invents) — bounded by the
  empty-window stub either way, but the exact numbers shown to the orchestrator vary per call.
- End-to-end HTTP confirmation of the /recommend 422 at Spa/Miami (server runs are owned elsewhere); the chain is proven at
  module level: data row -> producer emission (`_safe_none`) -> unguarded read -> exception semantics (E1).

---

## Addendum (2026-08-02, after this sweep was written)

Two of the sites recorded above as unfixed were closed on the same branch, AFTER this
sweep ran. This note exists because a reader grepping the audit trail later would
otherwise believe they are still open.

- **F1/F2 (`tire_agent.run_from_state` `tyre_life` at :1479 and `compound` at :1478, plus
  the twin in `src/strategy/inference/no_llm.py::_tire_no_llm`)** — FIXED. Both now derive
  through the canonical `normalise_compound` and `UNKNOWN_TYRE_LIFE` from
  `src/agents/race_state_builder.py` instead of restating the pre-#784 defaults
  (`"MEDIUM"`, `1`). Verified safe by the final correctness gate: every 2025 row carrying
  a degraded compound spelling also carries a NaN `TyreLife`, both the old and new paths
  produce an empty stint window and therefore the same conservative stub, and the scalar
  `tyre_life` never reaches the TCN (it only bounds the window and lands on a display
  field). No lap that previously produced a sane prediction produces a different one.
- **The weather half** — FIXED via the shared `reading_or_default` helper, with
  `pace_agent` (the one copy that had been correct) migrated onto it too so there is a
  single implementation rather than one good copy plus a helper for the stragglers.

Still live and deliberately out of the branch's scope:

- **R1** — `int(NaN)` on `TyreLife` at `race_situation_agent.py:914-915`. Filed as **#790**.
  Not fixed here because the mechanical part is trivial but the policy is not: the guarded
  twin (`pit_strategy_agent.py:983-999`) REFUSES the prediction rather than defaulting, and
  applying that to N27 means the orchestrator loses its overtake probability on ~451 laps.
- **R3** (`SpeedST` NaN into N12) and **R4** (the `or`-form in `mcp_tools.py:599` rewriting a
  leader's honest 0.0 gap to a fabricated 2.0) — recorded in #790's body and #789
  respectively.
