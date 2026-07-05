# AUDIT P3 - Arcade surface (pyglet replay + PySide6 dashboard + TCP stream)

> **Auditor**: Fable 5 · **Date**: 2026-07-05 · **Mode**: read-only, decision-grade, NO code.
> **Scope**: `src/arcade/` end to end: the pyglet 2D replay (`app.py`, `views.py`, `overlays.py`,
> `track.py`, `data.py`, `config.py`, `main.py`), the arcade-local strategy driver
> (`strategy.py`, `strategy_pipeline.py`), the TCP broadcast (`stream.py`, newline-delimited JSON
> at ~10 Hz), and the PySide6 dashboard package (`dashboard/`, 3 windows: strategy + telemetry,
> spawned as one Qt subprocess). `RaceReplayEngine` / `RaceStateManager` reviewed as consumers only.
> **Out of scope (owned elsewhere)**: session-load latency, the AoS pickle cache and the serial
> extraction path (audit P2, findings F-05/F-06/F-09 and remedies A1-A7); per-lap agent/LLM/MC
> compute cost (audit P2b, findings F1-F16); the FastAPI layer (P1); the CLI (P4).
> **Baseline**: this audit builds on the 2026-05-13 memory backlog `project_arcade_refactor_backlog`
> (quick wins 1-5, medium 1-6, heavy 1-4). Every baseline item is re-verified against current code
> below; items already absorbed by P2/P2b are cross-referenced, not re-planned.
> **Constraints honored**: `src/agents/` internals UNTOUCHABLE (the duplication is resolved by an
> ADDITIVE shared entry point, never by editing agent internals); `scripts/run_simulation_cli.py`
> untouchable; LLM = OpenAI / LM Studio, never Anthropic. Arcade is native PySide6/pyglet, so any
> visual verification uses Qt `grab()` / `arcade.get_image()`, not Playwright.

---

## 1. Executive summary

The Arcade surface is architecturally sound for a defended TFG PMV: clean view separation, correct
process isolation between pyglet and Qt, a sensible stdlib-only TCP link, and two good cost-control
mechanisms (lap gating and stale-skip) that the CLI still lacks. The debt concentrates in four places:

1. **The known #1 heavy item is confirmed and has grown a fourth copy.** `src/arcade/strategy_pipeline.py`
   is a body-copy of `run_strategy_orchestrator_from_state` importing seven private orchestrator
   helpers (`strategy_pipeline.py:28-36`), with a "mirror the change here" comment (`:19`) that audit
   P2b proved is a real failure mode (the 3-tuple change that broke the CLI's `--no-llm` was mirrored
   here but not there). On top of the two known copies there are two more duplication sites:
   `_build_default_lap_state` (`strategy_pipeline.py:124-167` duplicates
   `strategy_orchestrator.py:1327-1367`) and `SimConnector._build_race_state` (`strategy.py:516-561`
   duplicates the backend simulator's `_local_build_race_state`, admitted in its own docstring).
   P2b's finding F10 already designed the fix: one additive engine module at
   `src/strategy/inference/engine.py`. **This audit's P0 is the arcade side of that resolution:
   turn `strategy_pipeline.py` into a thin delegate and delete the three local copies.**
2. **The replay shows fabricated data in two places.** The weather panel renders hardcoded constants
   (45.0 C track, 18.0 C air, 55% humidity, 12 km/h wind) every frame (`app.py:648-656`) even though
   `SessionLoader` loads the weather DataFrame and throws it away (`data.py:268`). The progress-bar
   flag markers never render because `SessionData.events` is always `[]` (`data.py:312`), leaving
   `ProgressBar._draw_event` (`overlays.py:717-727`) dead, despite the loader already extracting
   exactly the data needed (`track_status_by_lap`). For a system whose thesis is data fidelity,
   placeholder values displayed as real readings are the highest-priority UX/correctness fix.
3. **The render path does per-frame work that should be per-resize or per-change.** The static track
   is re-tessellated into Python tuple lists and re-submitted in immediate mode every frame
   (`track.py:183-190`, roughly 4,000 `tuple()` calls plus 20 chequer `draw_line` calls at 60 FPS);
   every visible panel rewrites `arcade.Text.text` per frame whether or not the value changed; and
   `_build_frame_dict` allocates a 20-driver × 13-field dict every frame (`app.py:620-657`). On the
   dashboard side, every 10 Hz broadcast re-renders all six agent cards, the orchestrator card, six
   syntax-highlighted QTextEdits and a full tire-chart item rebuild, even though decisions change
   once per lap (`window.py:211-233`, `reasoning_tabs.py:210-229`, `tire_chart.py:163-218`).
   None of this is a confirmed frame-rate problem yet, so Phase C starts with a frame-time probe.
4. **Small real bugs and dead weight.** `f1-arcade --provider lmstudio` is parsed and silently ignored
   (`main.py:57`), so a user can believe they are on LM Studio while the strategy layer defaults to
   OpenAI unless the env var happens to be set. The menu's round stepper is capped at a hardcoded 23
   (`views.py:188`), making round 24 (Yas Island) unreachable for 2024/2025 and letting 2023 step
   into a nonexistent round. The idle-dim on conditional agent cards is a no-op because `opacity`
   is not a Qt stylesheet property (`agent_card.py:156-158`). The SSE-era config block
   (`config.py:160-165`) and the `backend_url` parameter (`strategy.py:205`) are dead.

Test coverage for the whole surface is one import-smoke file (`tests/test_arcade_dashboard_imports.py`).
The pure formatter/helper layer (agent_formatters, `_normalize_scores`, `classify_alerts`,
`RaceEventsPanel._status_for`, `_build_stints`, ProgressBar seek math) is cheap to test and untested.

---

## 2. Baseline reconciliation (memory backlog vs current code)

| Backlog item (2026-05-13) | Status today | Owner |
|---|---|---|
| QW1 `views.py:66` nested lambda default | Still present (`views.py:66`) | P3 Phase E |
| QW2 14 field lambdas with `setattr` | Still present (`views.py:174-227`) | P3 Phase E |
| QW3 magic strings + `_autofill_team` | Still present (`views.py:333-345`) | P3 Phase E |
| QW4 `_frame_to_telemetry` inline normalisation | Partially done: hoisted to module scope (`app.py:65-96`), still lives in `app.py` | P3 Phase E (fold into broadcast service) |
| QW5 `_fmt` inside `__init__` | Still present (`reasoning_tabs.py:60-65`) | P3 Phase E |
| M1 `F1ArcadeView` god class | Still present, now 614 lines (`app.py:99-712`) | P3 Phase E |
| M2 `_init_strategy_layer` coupling | Still present (`app.py:268-341`) | P3 Phase E |
| M3 `SimConnector` 310-line thread | Still present, now ~380 lines (`strategy.py:182-561`) | P3 Phase E |
| M4 blocking `SessionLoader.load()` on ENTER | Confirmed (`views.py:396-424`); remedy owned by **P2 (F-06)** | P2; P3 consumes |
| M5 two independent stream clients | Confirmed (`window.py:205`, `telemetry_window.py:53`) | P3 Phase D |
| M6 `telemetry_panel` init blob | Improved (chart factory `_make_chart` exists, `telemetry_panel.py:365-435`); residual init length acceptable | Closed |
| H1 pipeline duplication | Confirmed + 2 extra copies; resolution designed by **P2b F10** | **P3 Phase A (arcade side)** |
| H2 broadcast dict without schema | Still ad-hoc dicts (`app.py:426-488`); DTO exists only for strategy half | P3 Phase D (light) |
| H3 overlays 100+ field mutations per frame | Confirmed, plus static-track re-tessellation is the bigger cost | P3 Phase C (measure first) |
| H4 SessionLoader Pool + pickle cache | Superseded: serial `POOL_SIZE=1` since the P2 correction (2026-07-04); SoA cache + threading owned by **P2 (F-05/F-06/F-09)** | P2; do not re-litigate |

---

## 3. Findings register (P0 -> P3)

| ID | P | Finding (what / why) | Evidence | Size |
|---|---|---|---|---|
| A1 | **P0** | Strategy pipeline duplicated 4x: arcade body-copy of the orchestrator + private-helper imports + local `lap_state` and `RaceState` builders. Every orchestrator change must be manually mirrored; P2b F2 proved the mirror fails. Resolve INTO the shared additive engine (P2b F10), arcade becomes a delegate. | `strategy_pipeline.py:11-20, 28-36, 42-121, 124-167`; `strategy_orchestrator.py:1303-1418, 1327-1367`; `strategy.py:516-561` | M (arcade side) |
| A2 | **P1** | Weather panel renders hardcoded placeholder values as live data on every frame; loader fetches weather and discards it. Misleading on a defended surface. | `app.py:648-656`; `overlays.py:125-157`; `data.py:268` | S/M |
| A3 | **P1** | Timeline flag markers are dead: `SessionData.events` always empty, yet the loader already extracts per-lap `TrackStatus`. SC/VSC/red-flag spans could render today. | `data.py:312, 469-489`; `overlays.py:682-683, 717-727` | S |
| A4 | **P1** | `--provider` CLI flag parsed and never used (strategy layer reads only `F1_LLM_PROVIDER` env); `--viewer` path also skips the strategy-year validation the menu enforces. Silent wrong-provider risk (OpenAI spend when the user asked for LM Studio). | `main.py:57, 67-123`; `app.py:288` | S |
| A5 | **P1** | Static track re-tessellated per frame: two ~2,000-point polylines converted to tuple lists + immediate-mode `draw_line_strip` + 20 finish-chequer `draw_line` calls, 60 times per second, for geometry that only changes on resize. | `track.py:183-192, 309-324`; screen polylines already precomputed in `update_scaling` (`track.py:151-158`) | M |
| A6 | **P1** | Dashboard re-renders everything at broadcast rate (10 Hz) though strategy content changes once per lap: 6 agent cards, orchestrator card, scenario bars, 6 `setPlainText` calls that re-run the regex syntax highlighter, and a full tire-chart `removeItem`/`addItem` rebuild. | `window.py:211-233`; `reasoning_tabs.py:210-229`; `tire_chart.py:163-218` | S/M |
| A7 | P2 | Two independent `TelemetryStreamClient` sockets in the same Qt process: every broadcast is received, decoded and JSON-parsed twice, and the two windows drift out of sync (baseline M5). | `window.py:205-209`; `telemetry_window.py:53-57` | S/M |
| A8 | P2 | Broadcast serialisation on the render thread: `snapshot_dict` re-runs recursive `asdict` over the 30-decision history tail 10x per second even when no new decision exists, and `sendall` to each client is blocking, so one stalled subscriber can hitch the pyglet frame loop. | `strategy.py:155-176`; `stream.py:91, 99-103`; called from `app.py:416-436` | S/M |
| A9 | P2 | Per-frame allocation churn in the replay: `_build_frame_dict` rebuilds a 20-driver dict every frame; Weather/DriverInfo panels reuse ONE shared `arcade.Text` per role and mutate `.text/.x/.y` per row per frame (layout cost per mutation); leaderboard writes `.text` per slot per frame with no change check. Should become read-through views over the P2 SoA arrays plus write-on-change text updates. | `app.py:620-657, 659-678`; `overlays.py:103-156, 216-288, 438-481` | M |
| A10 | P2 | `F1ArcadeView` god class (614 lines): playback state machine, HUD, strategy layer lifecycle, TCP broadcast assembly, dashboard subprocess management and input handling in one class (baseline M1/M2). | `app.py:99-712` (esp. `:268-341, 358-374, 416-488, 532-575`) | M |
| A11 | P2 | `SimConnector` mixes five concerns (replay iteration, model warmup, radio corpus, race-state build, shared-state mutation) and reaches into `StrategyState._lock` 11 times (private-attribute coupling). `StrategyState` should own its mutators; warmup should route through the shared prewarm facade (P2 X-03). | `strategy.py:182-561`; lock reach-ins at `strategy.py:285-464` | M |
| A12 | P2 | Menu round stepper hardcodes `min(23, ...)`: round 24 (Yas Island) unreachable for 2024/2025 (both have 24 rounds in the canonical JSON); 2023 (22 rounds) can step to a nonexistent round 23 showing `?`. Cap must derive from `len(get_gp_names(year))`. | `views.py:186-189`; `config.py:266-292`; verified against `data/tire_compounds_by_race.json` | S |
| A13 | P2 | Teardown and navigation gaps: ESC in the replay calls `window.close()` directly; nothing routes back to `MenuView`, and `on_hide_view` (which stops the connector, stream server and dashboard subprocess) only runs on `show_view`. Whether arcade fires it on window close needs verification; if not, the Qt subprocess is orphaned on ESC. The stream server's per-client keepalive thread also never detects remote close (no read), so pruning relies solely on broadcast failures. | `app.py:532-534, 358-374`; `views.py:459`; `stream.py:131-143` | S/M |
| A14 | P2 | Idle-dim of conditional agent cards is a silent no-op: `opacity` is not a supported QSS property on QFrame, so N28/N30 idle cards look identical to active ones except for the glyph. | `agent_card.py:155-158` | S |
| A15 | P2 | Dead code and dead config: SSE-era constants (`BACKEND_URL`, `STRATEGY_ENDPOINT`, `SSE_*`) unused anywhere in `src/arcade`; `SimConnector.backend_url` kept "for backwards compat, unused"; stale `GP_NAMES` fallback table (23 rounds, `Monza` listed twice at rounds 15 and 23, order wrong for every season it might serve). | `config.py:160-165, 216-240`; `strategy.py:205` | S |
| A16 | P2 | Palette/classification constants triplicated with no drift guard: `config.py` (arcade) vs `dashboard/theme.py` (Qt) vs `telemetry/frontend/app/styles.py` (Streamlit); `classify_action` + alert severity duplicated between `strategy.py:567-603` and `theme.py:58-90`. The duplication is deliberate (process isolation), but nothing detects drift; a cheap parity test closes the gap without coupling imports at runtime. | `config.py:66-147`; `theme.py:1-13, 22-90` | S |
| A17 | P3 | Baseline quick wins still open in `views.py` (nested lambda default, 14 `setattr` lambdas, magic-string autofill) plus `_fmt` defined inside `__init__` in `reasoning_tabs.py`. Pure hygiene. | `views.py:66, 174-227, 333-345`; `reasoning_tabs.py:60-65` | S |
| A18 | P3 | Test coverage is import-smoke only. The formatter layer and pure helpers (`agent_formatters.*`, `_normalize_scores`, `classify_alerts`, `_status_for`, `_build_stints`, `_rolling_mean`, ProgressBar frame/x math, `_frame_to_telemetry`) are dict-in/tuple-out and trivially testable without Qt or pyglet. Coordinate with the Testing epic (#179). | `tests/test_arcade_dashboard_imports.py`; e.g. `overlays.py:592-610`, `strategy.py:578-626` | S/M |
| A19 | P3 | Season-wide parquet handed unfiltered to the pipeline each lap (arcade half of P2b F7) and arcade path resolution bypasses `get_data_root()` (P2 F-11). Listed for traceability only; both are inherited when the engine (Phase A) and P2's resolver work land. | `strategy.py:487-492, 506` | inherited |

### What NOT to do (explicit non-goals)

- **Do not re-litigate `POOL_SIZE=1`.** The serial default is the correct 2026-07-04 P2 correction;
  cold-path parallelism (pre-sliced per-driver inputs) and the numpy SoA cache are P2 deliverables.
- **Do not edit `src/agents/` internals** to fix A1. The engine module is additive
  (`src/strategy/inference/` already exists and holds only `tire_predictor.py`).
- **Do not schedule rendering refactors before measuring.** A5/A9 get a frame-time probe first;
  arcade at 20 dots + 5 panels may already hold 60 FPS on target hardware, in which case only the
  track baking (clear win, low risk) proceeds.
- **Do not build a schema/versioning layer for the TCP wire yet** (baseline H2). One producer, two
  consumers, same repo, same process boundary; a Pydantic `BroadcastPayload` is worth it only if an
  external consumer appears. Phase D adds the minimal thing instead: one constant for key names and
  a golden-payload test.

---

## 4. The duplication decoupling plan (A1, the P0)

**Target state**: one function computes a lap decision with per-agent outputs; every surface calls it.

- **Home**: `src/strategy/inference/engine.py`, per P2b §7. API (P2b's contract, restated):
  `run_lap(race_state, laps_df, lap_state, *, profile, return_agent_outputs=True) -> (StrategyRecommendation, agent_outputs, stage_timings)`.
  The engine imports the orchestrator's private helpers in ONE place (the precedent
  `strategy_pipeline.py` itself established) so there is exactly one mirror surface left, covered by
  a contract test instead of a comment.
- **Arcade change** (this audit's deliverable): `strategy_pipeline.run_strategy_pipeline` becomes a
  3-line delegate to `engine.run_lap` that adapts the return tuple; keep the function name and
  signature so `SimConnector._step_once` (`strategy.py:360-370`) and its DTO mapping
  (`_build_per_agent`, `_build_decision`) do not change. Delete `_build_default_lap_state`
  (the engine owns the default lap_state build, currently line-for-line identical to
  `strategy_orchestrator.py:1327-1367`).
- **RaceState builder**: move `SimConnector._build_race_state` (`strategy.py:516-561`) into the
  engine as an additive `build_race_state(lap_state, prev_lap_time, *, radio_msgs, rcm_events, risk_tolerance)`
  helper so the arcade, the backend simulator and the future CLI duplicate stop hand-rolling the
  same mapping. The radio lookup stays arcade-side (it owns `RadioPipelineRunner`).
- **Sequencing / ownership**: the engine module itself is P2b Phase 1.1. If P2b executes first, this
  phase is pure deletion + delegation. If P3 executes first, this phase BUILDS the minimal engine
  (verbose pass-through, `rich` profile only, stage timings optional) and P2b's later profiles land
  inside it; either order avoids double work because the API is already agreed.
- **Safety net**: a contract test asserting `engine.run_lap(...)[0] == run_strategy_orchestrator_from_state(...)`
  on a fixture lap (mocked LLM), so the "mirror the change here" comment can be deleted for good.
- **Untouchability**: zero edits inside `src/agents/`; `strategy_orchestrator.py` keeps its public
  entry points unchanged; the CLI PMV is not touched (its migration is P4's duplicate-and-improve).

---

## 5. Phased plan (each phase = one future GitHub sub-issue; S/M/L effort)

Dependency chain: **A -> (B, C, D in any order) -> E**. B, C, D are mutually independent.

### Phase A - Retire the pipeline duplication (M)
| Chunk | What | Effort |
|---|---|---|
| A.1 | Land / adopt `src/strategy/inference/engine.py` `run_lap` (coordinate with P2b 1.1; build minimal verbose engine if P2b has not executed) | M |
| A.2 | `strategy_pipeline.py` -> thin delegate; delete `_build_default_lap_state`; move `_build_race_state` into the engine as an additive builder | S |
| A.3 | Contract test: engine vs `run_strategy_orchestrator_from_state` on a fixture lap (LLM mocked); remove the mirror comment | S |

Exit: `src/arcade` contains no copied orchestrator logic; one arcade smoke replay (strategy ON) produces identical decisions to before.

### Phase B - Truth on screen: real data, real args (M)
| Chunk | What | Effort |
|---|---|---|
| B.1 | Real weather: extract per-lap weather in `SessionLoader` (session already loads it), thread through `_build_frame_dict`; render "N/A" when absent. Bump `CACHE_VERSION` once, coordinated with P2's SoA change (one v7 bump, not two) | M |
| B.2 | Timeline events: derive SC/VSC/yellow/red spans from `track_status_by_lap` into `SessionData.events` so `ProgressBar._draw_event` finally draws; RaceEventsPanel stays the live pill | S |
| B.3 | Fix `--provider` (set `F1_LLM_PROVIDER` from the flag or pass through to `SimulateRequestDTO`) and add strategy-year validation to the `--viewer` path; file the bug issue first per repo rule | S |
| B.4 | Menu round cap from `len(get_gp_names(year))`; regenerate or shrink the stale `GP_NAMES` fallback | S |
| B.5 | Fix the idle-dim no-op on `AgentCard` (QGraphicsOpacityEffect or muted text/border palette) | S |

### Phase C - Replay render path: measure, then bake (M)
| Chunk | What | Effort |
|---|---|---|
| C.1 | Frame-time probe: avg/p95 frame ms with all-cars ON/OFF, strategy ON/OFF, on target hardware; keep as a dev flag. Gates C.3/C.4 | S |
| C.2 | Bake static track geometry into retained arcade shape lists rebuilt only in `update_scaling` (edges, DRS segments, chequer); `on_draw` submits, never re-tessellates | M |
| C.3 | Write-on-change text updates in panels (skip `.text` assignment when unchanged); per-row Text instances where a shared label forces per-row relayout (Weather, DriverInfo) | S |
| C.4 | SoA-native frame access: once P2's numpy cache lands, replace `_build_frame_dict`/per-frame `FrameData` indexing with array reads; keep the dict only at the broadcast boundary | M |

### Phase D - Stream and dashboard efficiency (M)
| Chunk | What | Effort |
|---|---|---|
| D.1 | `StreamBroker`: one `TelemetryStreamClient` per Qt process, fan-out via Qt signals to both windows (baseline M5); halves socket + JSON work, synchronises windows | S/M |
| D.2 | Lap-gated strategy rendering: orchestrator card, agent cards, reasoning tabs, scenario bars and tire chart refresh only when `latest.lap_number` (or an error/finished flag) changes; telemetry panel stays at 10 Hz | S |
| D.3 | Broadcast hygiene: cache the serialised `latest`/`history_tail` in `StrategyState` behind a dirty flag (new decision invalidates); non-blocking or timeout sends in `TelemetryStreamServer.broadcast` so a stalled client cannot hitch the frame loop; fix or remove the no-op keepalive read | S/M |
| D.4 | Dead code: delete SSE constants + `backend_url` param; add the palette/classification parity test (arcade config vs dashboard theme vs Streamlit styles; `classify_action`/severity maps) | S |

### Phase E - Structure, UX and tests (M/L)
| Chunk | What | Effort |
|---|---|---|
| E.1 | Decompose `F1ArcadeView`: extract `PlaybackState` (7 booleans + speed + seek), `StrategyLayerController` (init/spawn/teardown, `app.py:268-374`), `BroadcastService` (`_broadcast_if_due` + snapshot builders + `_frame_to_telemetry`) | M |
| E.2 | Decompose `SimConnector`: `StrategyState` grows mutator methods (`set_error`, `push_decision`, `mark_finished`) killing the 11 `_lock` reach-ins; warmup + radio-corpus load delegate to the shared prewarm facade when P2 X-03 lands | M |
| E.3 | `views.py` quick wins (backlog QW1-3): named steppers instead of 14 `setattr` lambdas, `LaunchConfig.autofill_team()`, plain default for `visible` | S |
| E.4 | ESC/navigation: decide ESC = back-to-menu (replay -> `show_view(MenuView)`, which triggers `on_hide_view` teardown) with a second ESC to quit; verify and, if needed, harden dashboard-subprocess teardown on window close | S/M |
| E.5 | Unit tests for the pure layer (formatters, `_normalize_scores`, `classify_alerts`, `_status_for`, `_build_stints`, `_rolling_mean`, ProgressBar math, `_frame_to_telemetry`) + a broadcast golden-payload test; coordinate scope with the Testing epic (#179) | M |

---

## 6. Open questions (need Víctor's call)

1. **Engine ownership and order**: does Phase A execute as part of P2b's Phase 1 (engine first, arcade
   delegates after) or does the P3 epic own the minimal engine? Same API either way; pick to avoid
   parallel PRs on the same new module.
2. **ESC semantics**: back-to-menu (recommended: the menu exists and currently can never be reached
   again) or keep quit-on-ESC and add a separate key for menu?
3. **Weather granularity**: per-lap values (cheap, matches `track_status_by_lap`, enough for a
   strategy product) or time-interpolated per-frame (prettier, more loader work)? Recommendation: per-lap.
4. **CACHE_VERSION coordination**: single v7 bump covering P2's SoA shape + B.1 weather + B.2 events,
   to avoid invalidating users' 1.9 GB of session caches twice.
5. **Frame-rate acceptance bar**: what hardware defines "good"? Proposal: p95 frame time <= 16.6 ms
   with all cars + strategy + one connected dashboard on the dev laptop; C.3/C.4 are dropped if C.2
   alone reaches it.
6. **Bug-issue ceremony**: A4 (`--provider` ignored), A2 (fabricated weather) and A12 (round cap)
   qualify as file-the-bug-first issues under the repo rule; confirm they get individual issues
   rather than riding the phase issue.

---

## 7. Verification protocol

- **Phase A**: contract test green (engine == orchestrator recommendation on fixture, both with mocked
  LLM); one live arcade replay (Suzuka 2025, strategy ON) with decisions diffed lap-by-lap against a
  pre-change run (actions must match; prose may differ); `rtk grep "_run_always_on_agents_from_state" src/arcade`
  returns nothing outside the engine delegate.
- **Phase B**: Qt/OS screenshot of the replay (via `arcade.get_image()` or OS capture) showing real
  weather values changing across laps and SC spans on the timeline for a race with a known SC
  (e.g. Suzuka/Qatar 2025); `f1-arcade --viewer --strategy --provider lmstudio` observed hitting
  LM Studio (server log) with no `F1_LLM_PROVIDER` exported; menu can reach round 24 of 2025 and
  cannot exceed round 22 of 2023.
- **Phase C**: frame-time probe numbers in the PR description, before/after, same scene (all cars ON,
  strategy ON, dashboard connected); acceptance bar from open question 5.
- **Phase D**: dashboard smoke with both windows on one broker (single "Connected" transition in logs);
  CPU sample of the Qt process before/after lap-gating during a paused replay (should drop to ~idle);
  kill-the-dashboard-mid-race test: arcade frame time unaffected (non-blocking send path).
- **Phase E**: `uv run pytest tests/ -v` green including the new pure-layer tests; manual smoke matrix:
  menu -> 1-driver replay, menu -> 2-driver, strategy ON end-to-end, seek during strategy, ESC back to
  menu -> relaunch a second replay in the same process (teardown proven by the second run working),
  final quit leaves no orphaned Qt process (checked in Task Manager / `Get-Process`).
- **Every phase**: `uvx ruff check . && uvx ruff format --check .`; the arcade import-smoke test suite
  stays green on a PySide6-less environment (skip path intact).

## 8. Alignment with sibling audits

- **P2 (loading)**: owns the SoA cache (F-05), threaded menu load (F-06), cold-path extraction + quali
  geometry (F-09), `get_data_root()` routing (F-11) and the prewarm facade (X-03). P3's C.4 and E.2
  consume those; B.1's cache bump must be coordinated with F-05's.
- **P2b (compute)**: owns the engine internals, profiles and every LLM/MC lever. P3 Phase A is the
  arcade-side landing of its F10; A19 defers the season-frame filter to its F7.
- **P4 (CLI)**: the engine + `build_race_state` from Phase A are exactly what the CLI duplicate should
  consume; Phase A's contract test doubles as the CLI duplicate's acceptance oracle.
- **Testing epic (#179)**: E.5's pure-layer tests slot into its fixture/golden strategy; avoid
  duplicating scope by keeping arcade tests dict-in/tuple-out (no QApplication requirement).

---

### Appendix A - Evidence index (file:line)

| Claim | Evidence |
|---|---|
| Body-copy + 7 private-helper imports + mirror comment | `src/arcade/strategy_pipeline.py:11-20, 28-36, 42-121` |
| `_build_default_lap_state` copy of orchestrator block | `strategy_pipeline.py:124-167` vs `src/agents/strategy_orchestrator.py:1327-1367` |
| `_build_race_state` duplicate of simulator helper (admitted) | `src/arcade/strategy.py:516-561` (docstring line 517-519) |
| 3-tuple `_run_conditional_agents` (the mirror-failure precedent) | `strategy_orchestrator.py:1085-1165`; P2b audit F2 |
| Hardcoded weather constants rendered as live data | `src/arcade/app.py:648-656`; `src/arcade/overlays.py:125-157` |
| Weather loaded then discarded; events always empty | `src/arcade/data.py:268, 312` |
| Dead flag-marker path; TrackStatus already extracted | `overlays.py:717-727`; `data.py:469-489` |
| `--provider` parsed, never consumed; env-only provider | `src/arcade/main.py:57`; `app.py:288` |
| Viewer path skips strategy-year validation | `main.py:67-123` vs `views.py:402-413` |
| Track re-tessellation + immediate mode per frame | `src/arcade/track.py:183-192, 309-324`; precomputed screens at `:151-158` |
| Per-frame 20-driver dict + shared-Text row mutation | `app.py:620-657`; `overlays.py:103-156, 216-288, 438-481` |
| 10 Hz full dashboard re-render; highlighter re-runs; chart item churn | `dashboard/window.py:211-233`; `dashboard/reasoning_tabs.py:210-229`; `dashboard/tire_chart.py:163-218` |
| Two stream clients in one Qt process | `dashboard/window.py:205-209`; `dashboard/telemetry_window.py:53-57` |
| `asdict` of 30-decision tail per broadcast; blocking `sendall`; no-op keepalive | `strategy.py:155-176`; `src/arcade/stream.py:91, 99-103, 131-143` |
| God class extents | `app.py:99-712` (strategy layer `:268-374`, broadcast `:416-488`, input `:532-575`) |
| `SimConnector` concerns + 11 `_lock` reach-ins | `strategy.py:182-561`; reach-ins `:285-464` |
| Round cap `min(23, ...)` vs 24-round calendars | `views.py:186-189`; `data/tire_compounds_by_race.json` (2024: 24, 2025: 24, 2023: 22 rounds, verified) |
| ESC bypasses teardown path; no route back to menu | `app.py:532-534, 358-374`; only `views.py:459` calls `show_view` |
| QSS `opacity` no-op idle dim | `dashboard/agent_card.py:155-158` |
| Dead SSE config; unused `backend_url`; stale GP_NAMES (Monza twice) | `config.py:160-165, 216-240`; `strategy.py:205` |
| Palette/classification triplication (deliberate, unguarded) | `config.py:66-147`; `dashboard/theme.py:1-13, 58-90`; `strategy.py:567-603` |
| Only import-smoke tests exist | `tests/test_arcade_dashboard_imports.py` |
| Season parquet unfiltered + `REPO_ROOT` paths (inherited) | `strategy.py:487-492, 506`; P2b F7, P2 F-11 |
| Good patterns to keep: lap gate, stale-skip, Text pre-allocation, process isolation | `strategy.py:228-260`; `overlays.py:1-9`; `app.py:321-341` |
