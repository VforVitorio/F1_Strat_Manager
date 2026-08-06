# ADVERSARIAL GATE — Route GP Scoping (submodule PR #213, `fix/route-gp-scoping`)

**Date:** 2026-08-06
**Auditor:** adversarial gate (no repo file modified except this report)
**Scope:** `src/telemetry` submodule, diff `main...HEAD` on `fix/route-gp-scoping`
**Prior context (NOT re-reported):** GATE_PR5_OVERTAKE.md finding 2; PR3_GP_KEYSPACE_SWEEP, PR4_PACE_INPUTS, PR5_OVERTAKE_DOMAIN already fixed and merged.

## Checklist

- [x] A. Per-agent lookup audit (tyre, situation, pit, radio, orchestrator): Driver+Lap, never GP — show one wrong resolution pre-fix → VERIFIED, executed
- [x] B. Enumeration attack: is 9 the real count? Is `/pace` frame-free? Find a TENTH site → `/pace` genuinely frame-free; no unscoped tenth site; but the COUNT is wrong in three places and 2 of the 9 were already internally scoped
- [x] C. `scope_to_race` edge cases → two uncaught AttributeError paths (`session_meta: null`, non-str gp_name); empty-in→empty-out contradicts the docstring
- [x] D. Double-scoping no-op on the REAL frame, incl. the divergent Miami spelling → VERIFIED, executed
- [x] E. `/recommend` old bare mask → empty frame → REAL (0 rows for 'Miami Gardens'), reachable only via the documented RSM-lap_state contract, not the shipped webapp
- [x] F. Cross-race dependents → REFUTED: no consumer worsens; every change moves routes toward the run_lap reference. Executed provenance checks
- [x] G. Attack the test → floor of 9 vs actual 11; four executed evasions (class method, module-level, gp_df alias, other files)
- [x] H. MCP wrapper → no NameError path (AST-verified); body executed via the identical laps_cache path; module import not executable here (fastmcc absent by design)
- [x] Project bug classes → 1 fix-introduced regression (lazy-import defeat), 1 dropped input (request.gp_name), spelling twins left in /pace-range + /tire-range, count drift across three docstrings

## Findings (appended as confirmed)

### Claim A — VERIFIED. The premise is real, executed on the served frame

`get_laps_df(2025)` = 22 760 rows, 24 GPs. Executed (`gate_probe1.py`, parent venv):

- `_get_lap_row`-style `(Driver=='VER') & (LapNumber==20)` → **21 rows, `iloc[0]` GP = 'Austin'** (pit agent `src/agents/pit_strategy_agent.py:788,814`; situation agent x/y rows `src/agents/race_situation_agent.py:1379-1392`).
- `_get_position_map`-style `LapNumber==20` → **395 rows across 23 GPs** where one race has ~20 (`pit_strategy_agent.py:829,838`).
- Radio agent SESSION_META from frame (`src/agents/radio_agent.py:1013-1017`): season `total_laps` = **78 vs Lusail's 57**.
- Situation-agent frame-wide stat: season fastest lap **67.924 s (Spielberg) vs Lusail 82.996 s** (`race_situation_agent.py:1584,1708` — note it reads `LapTime`, the featured frame carries `LapTime_s`; see LOW finding below).
- Tire `_get_driver_stint`-style `(Driver=='VER') & (TyreLife==5)` → **48 rows across 22 GPs** (`tire_agent.py:1164`).

Entry points audited: only `run_strategy_orchestrator_from_state` scopes internally (`src/agents/strategy_orchestrator.py:2482`); `run_tire_agent_from_state` (tire_agent.py:1828), `run_race_situation_agent_from_state` (race_situation_agent.py:1976), `run_pit_strategy_agent_from_state` (pit_strategy_agent.py:1705), `run_radio_agent_from_state` (radio_agent.py:985) do NOT. So the 8 per-agent sites needed the fix; the 2 recommend sites were already internally scoped (see Claim B section).

### Claim D — VERIFIED on the real frame, including the spelling that differs

Executed: `scope_to_race` twice with `gp_name='Miami Gardens'` → once = 857 rows keyed `'Miami'`, twice = 857 (the resolver maps 'Miami Gardens' onto the single-key keyspace `{'Miami'}` again). `'Lusail'`: 904/904. Double scoping is a no-op on both the matching and the divergent spelling.

### Claim E — the empty-frame hazard was REAL for the documented caller class (executed), reachability narrow

Executed masks on the served 2025 frame: `GP_Name == 'Miami Gardens'` → **0 rows**; `'Bahrain Grand Prix'` → 0; `'Qatar Grand Prix'` → 0 (frame keyspace: 'Austin', 'Baku', …, 'Miami', 'Sakhir', 'Lusail'). `RecommendRequest`'s own docstring (strategy.py:98-105) says it "accepts the raw lap_state dict produced by RaceStateManager" — the keyspace PR3 documented as metadata-spelled ('Miami Gardens'). The webapp itself is NOT an affected caller: `webapp/src/lib/api/strategy.ts:358` posts `session_meta.gp_name` obtained from `/lap-state`, whose `gp` param the UI picked from `/available-gps` (parquet keys). So: reachable through the documented public contract (any RSM-built lap_state), not through the shipped UI.

### Claim B — no unscoped tenth site found; the arithmetic is wrong anyway

Hunted: `grep -rn "_from_state|run_rag_agent|StrategyOrchestrator" backend` — agent invocations exist ONLY in `strategy.py` (8) and `mcp_tools.py` (7). The simulator (`backend/services/simulation/simulator.py:402,846,860`) routes through `run_lap`, which scopes BEFORE dispatch (`src/strategy/inference/engine.py:223`, read in context). The chat endpoint (`backend/api/v1/endpoints/chat.py`) touches no frame and no agent (0 grep hits). `telemetry.py` / `comparison.py` / `circuit_domination.py` serve telemetry, not agents. `/pace` and MCP `predict_pace` are genuinely frame-free: `run_pace_agent_from_state(lap_state)` takes no frame, and internally the pace agent reads only per-(year, GP)-keyed circuit artefacts (`src/agents/pace_agent.py:334-402`), no Driver+Lap frame lookups.

BUT the counting is inconsistent in three places, and two of the claimed nine did not need fixing:

- `backend/utils/laps_cache.py:76` says "the FIVE per-agent POST routes ... Ten sites"; the test module docstring (line 13) says "the FOUR per-agent POST routes"; the test's floor message says "9 (4 HTTP routes + 5 MCP tools)"; the scanner ACTUALLY examines **11** functions (executed, Claim G). Four different numbers for one enumeration — the project's wrong-count-in-a-comment class.
- `run_strategy_orchestrator_from_state` ALREADY scopes internally (`src/agents/strategy_orchestrator.py:2482` delegating to the engine's `_scope_laps_to_gp`). So MCP `recommend_strategy` was never broken (the season frame got scoped one call deeper), and HTTP `/recommend`'s agents never saw the season frame either — its real defect was the OTHER half (the bare mask's empty frame, Claim E). "Nine sites needed it" overstates by two; the added scoping there is a harmless no-op (Claim D), but the claim is not accurate as written.

### Claim C — executed edge-case matrix

`scope_to_race(season, X)` (gate_probe1.py):

| lap_state / frame | result |
|---|---|
| `None` / `{}` | full frame + loud warning OK |
| `{"session_meta": None}` | **RAISES AttributeError** ('NoneType' has no 'get') |
| `{"session_meta": {"gp_name": None}}` / `""` | full frame + warning OK |
| `{"session_meta": {"gp_name": 123}}` | **RAISES AttributeError** ('int' has no 'replace', inside `resolve_gp_key`) |
| `laps_df=None` | returns `None`, silently (same input the agents got pre-PR) OK |
| empty frame WITH GP_Name col | **returns the EMPTY frame** (warning fires) — the docstring's "never hands an agent an empty one" is false for an empty input |
| frame WITHOUT GP_Name col | full frame + warning OK |

Grading the two raises: `session_meta: null` in a POSTed lap_state crashed the per-agent routes pre-PR too (pit: `meta = lap_state['session_meta']` then `meta.get(...)` on None; radio: `lap_state.get("session_meta", {}).get("gp", "")` at `src/agents/radio_agent.py:1016`), so the observable 500 is unchanged — the crash just moved earlier. NOT a regression, but the guarded twin exists one module over: `src/agents/race_state_builder.py:318` writes `lap_state.get("session_meta", {}) or {}` — the `or {}` that `engine.py:123` lacks. The `gp_name: 123` case IS a narrow regression: pre-PR `/tire` ignored gp_name entirely and served 200; post-PR it 500s (AttributeError is outside the routes' `(KeyError, TypeError, ValueError)` net). Garbage-in, but the boundary is client-controlled JSON.

### Claim F — REFUTED: no consumer gets worse with a single-race frame

Attacked each named suspect, provenance read at the line:

- **Undercut circuit rate**: computed from `undercut_clean.parquet`, a separate artefact (`src/agents/pit_strategy_agent.py:272-279`), NOT from the handed frame. Unaffected.
- **Pit team medians**: rebuilt from `data/raw/<year>/<GP>/pitstops.parquet` at agent init (`pit_strategy_agent.py:281-330`). Unaffected.
- **Radio runner driver map**: `RadioPipelineRunner` scopes its frame internally by gp_name (`src/nlp/radio_runner.py:288,395-407`); the map only needs the race's own drivers. MCP `analyze_radio` now hands it a pre-scoped frame (`mcp_tools.py:548,552`) — re-resolving against a single-key keyspace, proven a no-op in Claim D. Unaffected.
- **Orchestrator projection**: `/recommend` content is IDENTICAL pre/post for resolvable names — the internal scope at `strategy_orchestrator.py:2482` already narrowed the frame before any sub-agent or the projection saw it.
- The values that DO change on the four per-agent routes all move TOWARD the reference (`run_lap` has fed agents single-race frames since #429, matching per-race training): situation `fastest_lap_s` 67.924 (season min, Spielberg) vs 82.996 (Lusail); radio SESSION_META `total_laps` 78 (season max) vs 57; position maps 395 rows/23 GPs vs ~20 rows/1 GP. Executed in gate_probe1.py.

### Claim G — the test's guard has four executed holes, and its floor is calibrated to the wrong number

Ran the test's own `_top_level_functions` + enumeration loop verbatim (gate_probe_g.py):

1. **Real scan examines 11 functions, not 9**: 6 in strategy.py (incl. `predict_pace_range`, which passes via the `gp_df` token, and `recommend_strategy`) + 5 in mcp_tools.py. The floor `examined >= 9` therefore tolerates TWO sites silently disappearing from the scan before it trips — the exact vacuity the assert message claims to prevent, miscalibrated on day one.
2. **Class-method masking (executed)**: an unscoped `run_*_from_state(..., laps_df)` inside a class method is glued onto the PRECEDING top-level `def` (splitting is on column-0 `def` only); if that predecessor is scoped, the token check passes — `unscoped=[]`, test green.
3. **Module-level call (executed)**: a call before the first `def` belongs to no function and is never examined — invisible, test green.
4. **`gp_df` alias false-pass (executed)**: `gp_df = laps_df` (never narrowed) satisfies the token check — test green. The real `predict_pace_range` already passes on this token with a bare `==` mask — defensible today (it 404s on empty), but the token proves the NAME, not the narrowing.
5. Any agent call added in a file other than the two scanned (e.g. `telemetry.py`) is out of scan scope by construction.

### Claim H — no NameError; deferred import matches the module pattern; module import not executable here

AST over `backend/mcp_tools.py` (executed): `_scope_to_race` defined at module level (line 423); all 5 references (481, 498, 514, 536, 598) are inside tool function bodies; `lap_state`/`base_state` is bound before each call (464-481, 497-498, 513-514, 535-536, 597-598); file compiles. The wrapper's body (the deferred `from backend.utils.laps_cache import scope_to_race`) is the SAME code path executed throughout gate_probe1.py, including `laps_df=None` returning None (pre-PR agent input unchanged). Deferred-import style matches `_get_laps_df` / `_serialize` / `_format_result`. NOT executable end-to-end here: `fastmcp` is absent from the parent venv (and CI omits it deliberately, per the test's own comment) — the module import was verified by `compile()`, not by running the MCP server.

---

## Ranked findings

### HIGH-1 — `scope_to_race` imports the ENTIRE agent family and instantiates the radio NLP models, defeating the deliberately lazy `src/agents/__init__`

`backend/utils/laps_cache.py:84` — `from src.strategy.inference.engine import _scope_laps_to_gp`. `engine.py` imports every agent at module level, and `src/agents/radio_agent.py` instantiates its three transformer models (RoBERTa sentiment, intent, NER) AT IMPORT.

Executed evidence:
- `import src.strategy.inference.engine` in a fresh process: **16.7 s**, and `sys.modules` afterwards contains ALL of `src.agents.{pace,tire,race_situation,pit_strategy,radio,rag,strategy_orchestrator}` — with the transformer weight-load reports streaming during the import.
- Baseline `import src.agents.tire_agent` in a fresh process: `src.agents.radio_agent` **NOT loaded**, no weight-load reports.
- In gate_probe1.py, the RoBERTa/BERT load reports appear exactly at the FIRST `scope_to_race` call.
- `src/agents/__init__.py`'s own docstring: "Importing this package ... no longer eagerly loads every agent. The heavy agents ... used to load at import time ... so touching ANY single agent ... pulled the whole family into memory and VRAM." That is precisely what this helper reintroduces, one layer up.

Failing scenario: a fresh API worker (or the MCP chat server) serves its first `/tire` (or `predict_tire` tool call) — pays the engine import: the radio NLP weights + RAG agent + orchestrator land in RAM/VRAM of a process that may never serve a radio request, plus the first-call latency. Every one of the 9 wired sites triggers it. Pre-PR, `/tire` imported only `tire_agent`.

Fix: `_scope_laps_to_gp`'s real dependencies are pandas + `resolve_gp_key` + logging (the RaceState param is optional and the backend never passes it). Move it to a leaf module (e.g. `src/strategy/inference/scoping.py`), import it from `engine.py` and `laps_cache.py`. One move + two imports; removes the whole regression.

### MEDIUM-1 — `/recommend` silently DROPPED the `request.gp_name` input for scoping

Old: `race_laps_df = laps_df[laps_df["GP_Name"] == gp] if gp else laps_df` with `gp = request.gp_name or session_meta.gp_name` (request.gp_name took priority). New (`strategy.py:1388`): `_scope_to_race(laps_df, request.lap_state)` reads ONLY `session_meta.gp_name`. `RecommendRequest.gp_name` (strategy.py:108) is still a documented field and still feeds the RCM lookup (line 1365), but no longer influences scoping.

Failing scenario: a caller posts `gp_name="Lusail"` with a lap_state whose session_meta lacks `gp_name` (or carries a different value). Pre-PR: frame scoped to Lusail. Post-PR: warning + FULL SEASON frame — the orchestrator's internal scope (2482) then derives the GP from the (driver, lap) row match, which resolves to the FIRST race in the file (Austin, executed in Claim A), i.e. the exact wrong-race bug this PR exists to kill. The shipped webapp is unaffected (`webapp/src/lib/api/strategy.ts:358` sends the same value in both places), so this is a public-contract regression, not a UI-visible one. Fix: overlay `request.gp_name` into the lap_state handed to `_scope_to_race`, mirroring the old precedence.

### MEDIUM-2 — the enumeration guard's floor is miscalibrated (9 written vs 11 actual) and its docstrings disagree (4 vs 5 vs 9 vs 10)

See Claims B and G. The non-vacuity assert exists to catch the scanner silently losing sites, and on the day it was written it already under-counts its own scan by two — two sites can vanish before it fires. Three docstrings state three different enumerations. Fix: set the floor from the real count (11) with the breakdown in the message, and reconcile the module docstrings ("four per-agent POST routes + /recommend" is the accurate phrasing).

### MEDIUM-3 — the scanner passes on tokens, not narrowing: three executed evasions

`gp_df = laps_df` alias, unscoped call in a class method following a scoped def, module-level call before the first def — each keeps the test green (gate_probe_g.py, Claim G). Scan scope is also two files by construction. Fix (cheapest first): require the scope token within the same STATEMENT as the `_from_state(` call, or parse with `ast` (walk all FunctionDef bodies incl. class/nested scopes, flag `run_*_from_state` calls whose frame argument is not a `_scope_to_race(...)` call or a name assigned from one).

### LOW-1 — `scope_to_race` raises where the routes answer 500: `session_meta: null` and non-str `gp_name`

Executed (Claim C). `session_meta: null` crashes pre- AND post-PR (crash moved earlier, same 500) — not a regression, but the `or {}` guard its neighbour `race_state_builder.py:318` uses is absent at `engine.py:123`. `gp_name: 123` is a NEW 500 on routes that previously returned 200. One-line fix in `_scope_laps_to_gp`: `(lap_state or {}).get("session_meta") or {}`, and str-coerce/isinstance-guard gp_name before `resolve_gp_key`.

### LOW-2 — "never returns an empty frame" is false for an empty INPUT

Executed: `scope_to_race(season.iloc[0:0], lusail_state)` returns 0 rows (the fallback returns `laps_df`, which IS empty). Unreachable through today's routes (`require_laps_df` 503s on missing; a present parquet is non-empty), so docstring-only today. Tighten the sentence or add an explicit empty guard.

### LOW-3 — the spelling-resolution twins the PR left behind: `/pace-range` and `/tire-range`

`strategy.py:674` and `strategy.py:1001` still narrow with the bare `==` mask the PR's own /recommend comment condemns ("A bare mask neither resolves the four spellings..."). Consequence differs — they 404 loudly ("GP 'Miami Gardens' not found" while 857 Miami rows exist under 'Miami') instead of feeding agents an empty frame — so this is a usability gap, not silent corruption. Same defect class, same file, found by reading the PR's own comment. Fix: resolve via `resolve_gp_key` and keep the 404 for genuinely unknown names.

### INFO — two of the nine wired sites were already safe

MCP `recommend_strategy` and HTTP `/recommend` agents never saw the season frame: `run_strategy_orchestrator_from_state` scopes internally (`strategy_orchestrator.py:2482`). The added route-level scope is a proven no-op (Claim D) — harmless redundancy, but the PR narrative ("nine sites needed it") and the laps_cache docstring ("Ten sites, one rule") overstate the blast radius by two. HTTP `/recommend`'s change earns its place for the OTHER reason: it kills the reachable-by-contract empty-frame mask (Claim E).

---

## Numbered fix list (by value)

1. **Move `_scope_laps_to_gp` to a leaf module** and import it from `engine.py` + `laps_cache.py` (HIGH-1). Restores the lazy-agent design; zero behaviour change.
2. **Restore `request.gp_name` to /recommend scoping** (MEDIUM-1): overlay it into the lap_state passed to `_scope_to_race`, mirroring the old precedence.
3. **Recalibrate the test floor to 11 with a breakdown; reconcile the three docstring counts** (MEDIUM-2).
4. **Harden the scanner** (MEDIUM-3): same-statement token check or an `ast` walk covering class/nested/module scopes.
5. **One-line input guards in `_scope_laps_to_gp`** (LOW-1): `or {}` on session_meta, str-coerce gp_name.
6. **Resolve spellings in `/pace-range` + `/tire-range`** (LOW-3).
7. **Docstring truthfulness** (LOW-2 + INFO): empty-in/empty-out caveat; "five per-agent POST routes" corrected to four.

---

## What I tried to break and could NOT

- **The core fix**: on the served 2025 frame (22 760 rows, 24 GPs), every scoped lookup now resolves to the requested race — Lusail/Sakhir/Monaco parametrized, plus the divergent 'Miami Gardens' resolving to 'Miami'. The PR's 8-test suite passes under the parent venv (8/8, 20.2 s).
- **Double scoping**: no-op on both the matching and the divergent spelling (857/857, 904/904 rows) — including the /recommend route + orchestrator-internal second scope.
- **A tenth unscoped web site**: hunted `_from_state` / `run_rag_agent` / `StrategyOrchestrator` / `run_lap` across the whole backend (routes, services, simulator, chat, MCP): every frame-passing agent call is either wired through `_scope_to_race` or behind `run_lap`'s internal scope. `/pace` and `/pace-range`'s agent is genuinely frame-free (verified into `pace_agent.py`'s artefact loaders).
- **The loud-fallback property**: unresolvable, None, and empty gp_names all return the FULL frame with the #429/#465 warning — no path turns a non-empty frame into an empty one.
- **A cross-race dependent that regresses** (the most valuable target): undercut circuit rates, pit team medians, and the radio driver map all come from per-race-keyed artefacts or internally-scoped frames — provenance read at the line for each; the orchestrator path is identical for resolvable names because it was already scoped one layer down.
- **NameError / unbound-name paths in the MCP wrapper**: AST-verified all references are call-time, inside tool bodies; the wrapper body is the identical code executed repeatedly in the probes.
- **The `_RADIO_RUNNER_CACHE` divergence**: MCP now hands the runner a scoped frame while `/recommend` hands the season frame under the same `(year, gp)` cache key — but the runner scopes internally by gp_name either way (`radio_runner.py:395-407`), so both frames produce the same driver map; could not construct a divergent outcome.

## Probe artefacts

Scratchpad (session-local, not in the repo): `gate_probe1.py` (claims A/C/D/E), `gate_probe_g.py` (claim G). Both runnable with the parent venv from `src/telemetry`.
