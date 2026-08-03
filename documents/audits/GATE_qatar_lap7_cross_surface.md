# GATE — Qatar (Lusail 2025) lap 7 cross-surface RaceState verification (#787, over #784/#786)

**Date:** 2026-08-02 · **Branch:** `refactor/single-source-race-state-builder` (working tree, uncommitted) · **Submodule:** `7f394a8` (committed in submodule, pointer not yet bumped in parent)

**Role:** adversarial gate. Success = finding what is still broken. Every claim below is backed by executed evidence unless explicitly marked CODE-READ.

## Scope verified on disk (not trusted from the summary)

- `src/agents/race_state_builder.py` — NEW, untracked. Canonical builder, lazy `RaceState` import, `None`-means-compute for `gap_ahead_s`/`pace_delta_s`, `rival` targeting, position `ValueError` guard. CONFIRMED present.
- `scripts/run_simulation_cli.py` — `git diff` shows exactly two hunks (L1287-1297 comment + `_build_race_state` body at L1300-1322). **The main loop has zero hunks — the byte-identical requirement on L1710-1764 holds** (verified via `git diff`, not by eye).
- `src/arcade/strategy.py` — `_build_race_state` (L599-654) now delegates; keeps radio sourcing + stateful SC re-injection. The position `ValueError` moved AFTER `sc_tracker.ingest` (see Suspicion 1).
- Submodule `7f394a8`: `backend/utils/race_state_builder.py` is a 24-line re-export shim; `simulator.py::_compute_gap_ahead` deleted; `_local_build_race_state` passes `pace_delta_s=0.0`, omits `gap_ahead_s`. CONFIRMED by `git show 7f394a8`.

## Findings (appended as confirmed)

<!-- findings appended below as they are confirmed -->

### [CORE DELIVERABLE] Task 1 — field-by-field cross-surface diff, Lusail 2025 lap 7, NOR/McLaren (EXECUTED)

Reference case confirmed from data, not guessed: McLaren at Lusail 2025 = **NOR (P2)** and **PIA (P1, the car ahead)**; NOR used (the thesis/memory reference driver). Lap 7 ground truth from `RaceStateManager`: lap_time 100.154, PIA 98.466, `interval_to_driver_s` −6.001, compound MEDIUM, tyre_life 7, weather.parquet real readings air 23.4 / track 29.5, **real "SAFETY CAR DEPLOYED" corpus message AT lap 7** (the V7 case is live in this data).

Harness (`scratchpad/gate_harness.py`) drove the REAL paths: `run_simulation_cli._build_race_state` + the main loop's exact post-build radio/SC mutation replicated per-lap for laps 1–7; a real `SimConnector` instance through its own `_lap_skip_reason` → `_build_race_state` loop; the submodule's `_local_build_race_state` via the `backend.utils.race_state_builder` shim mirroring the SSE loop; and `/recommend`'s exact builder call with `RecommendRequest` defaults.

| RaceState field | CLI (no-radios) | CLI (corpus) | Arcade | Backend SSE | /recommend defaults | Verdict |
|---|---|---|---|---|---|---|
| driver | NOR | NOR | NOR | NOR | NOR | AGREE |
| lap | 7 | 7 | 7 | 7 | 7 | AGREE |
| total_laps | 57 | 57 | 57 | 57 | 57 | AGREE |
| position | 2 | 2 | 2 | 2 | 2 | AGREE |
| compound | MEDIUM | MEDIUM | MEDIUM | MEDIUM | MEDIUM | AGREE |
| tyre_life | 7 | 7 | 7 | 7 | 7 | AGREE |
| gap_ahead_s | 6.001 | 6.001 | 6.001 | 6.001 | **2.0** | DIFFER — owned |
| pace_delta_s | 1.688 | 1.688 | 1.688 | **0.0** | **0.0** | DIFFER — owned |
| air_temp | 23.4 | 23.4 | 23.4 | 23.4 | 23.4 | AGREE (real reading, not default) |
| track_temp | 29.5 | 29.5 | 29.5 | 29.5 | 29.5 | AGREE (real reading, not default) |
| rainfall | False | False | False | False | False | AGREE |
| risk_tolerance | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | AGREE |
| radio_msgs | [] | [] | [] | [] | [] | AGREE (harness ran RCM-only runners — see limitation below) |
| rcm_events | [] | 5 events | 5 events | 5 events | 5 events | DIFFER — owned |

Every DIFFER has a named owner:
- **pace_delta_s 1.688 vs 0.0** — hand-checked: 100.154 − 98.466 = 1.688 (rival-relative, #750 axis). The backend SSE pins 0.0 deliberately (`simulator.py::_local_build_race_state`, "the schema's documented neutral... switching is a separate behavioural decision"); `/recommend` forwards `RecommendRequest.pace_delta_s` whose default is 0.0. Both recorded in #784's design.
- **gap_ahead_s 2.0 on /recommend defaults only** — `RecommendRequest.gap_ahead_s: float = GAP_UNKNOWN_FALLBACK_S` (endpoint `strategy.py:110`); the endpoint always passes an explicit float, so the canonical `None`-means-compute derivation NEVER runs on that surface even though `lap_state.rivals` carries the real 6.001. Owned difference (the webapp client supplies the value), but see the observation below.
- **rcm_events** — surface-owned sourcing by design (builder module docstring): the CLI `--no-real-radios` profile has none; the other four carry the **byte-identical** 5-event list (verified `==` across surfaces), including the real `SAFETY CAR DEPLOYED` at lap 7. Trackers on all three stateful surfaces ended lap 7 in the same state: `active=True kind=SC deployed=7`.

**Harness limitation stated plainly:** the radio-transcript half (Whisper) was not exercised in the harness — all runners ran `disable_transcription=True`, so `radio_msgs` is [] everywhere above. The rcm.parquet half (what the SC override consumes) is the real path. Real runs in Task 2 cover the rest.

### [OBSERVATION / LOW] `/recommend` can never use the canonical gap derivation

Not introduced by #784 (the request default predates it), but #784 built exactly the machinery (`gap_ahead_s=None` → compute from rivals) that this surface cannot reach: `RecommendRequest.gap_ahead_s` is non-Optional with a fabricated 2.0 default, so a client that omits the field gets 2.0 while the true 6.001 sits in the same request's `lap_state.rivals`. Making the request field `Optional[float] = None` would let the canonical builder derive it. Same shape in `mcp_tools.py:599`: `driver_state.get("gap_ahead_s") or GAP_UNKNOWN_FALLBACK_S` — an `or` that also maps a genuinely-measured 0.0 (leader) to 2.0, the #633 conflation in miniature, acknowledged in the builder docstring as "byte-compatible" preservation. Both recorded decisions; flagged so the follow-up is deliberate, not forgotten.

### [VERIFIED-SAFE] Suspicion 1 — the Arcade's reordered `sc_tracker.ingest` cannot corrupt later laps (EXECUTED)

`src/arcade/strategy.py:645` now runs `ingest` BEFORE the builder's position `ValueError` (old code raised before radio/SC sourcing). Executed experiment (`probe_tracker.py`, real `RaceControlStateTracker`, pre-classified events):

- **Deploy lands on the breached lap**: breached ordering goes SC-active and injects for laps N+1..N+7; counterfactual (guard worked) never sees the SC at all. Divergence bounded by the safety valve `_MAX_SC_LAPS = 8` (`rcm_state.py:36`) — clears at deploy+8, verified by execution. The breached trajectory is the more TRUTHFUL one: the deploy was a real FIA message, so the spurious ingest adds real information, it cannot invent a neutralisation.
- **Release on the breached lap**: breached ordering clears immediately; the counterfactual keeps injecting a stale SC until the valve. Again breached = more accurate.
- **Malformed lap number** (arcade's `lap_state.get("lap_number", 1) or 1` falls back to 1 mid-race): the spurious `ingest(1, [])` does NOT clear an active SC (negative delta never satisfies the `>= 8` valve check) and does not touch `deployed_lap`; the valve still clears at deploy+8. Only artefact: `last_seen_lap=1` makes the next synthetic event's cosmetic `lap` field read 1 until the next real ingest.
- **Self-feeding ruled out**: all three surfaces feed the tracker the RAW rcm list and append the synthetic to a copy (`arcade:645-647`, `run_simulation_cli.py:1705-1707`, `simulator.py:333-336`, `strategy.py endpoint:1299-1303`), so `deployed_lap` never refreshes and the valve always fires. Executed: after deploy at 5 + 10 empty ingests, tracker is cleared.

Residual asymmetry worth knowing (CODE-READ): in the breach case the CLI's ordering (build raises BEFORE ingest, `run_simulation_cli.py:1660` vs `:1705`) drops the lap's real RCM from its tracker while the arcade's keeps it — a cross-surface divergence that exists ONLY when the skip-guard invariant is already broken, is bounded by the 8-lap valve, and where the arcade side is the truthful one. Not a defect; recorded so nobody "fixes" the arcade back.

### [VERIFIED] Leaf-import constraint holds (EXECUTED)

`import src.agents.race_state_builder` in a fresh interpreter pulls **zero** `langchain`/`langgraph`/`torch`/`xgboost`/`transformers`/`lightgbm` modules. Constants confirmed live: track 35.0 / air 25.0 / "UNKNOWN" / tyre_life 0.

### [VERIFIED] Suspicion 2 — explicit `0.0` is honoured; no falsy-vs-None trap (EXECUTED)

`probe_defaults.py`, real builder, synthetic lap_state with a car ahead at 6.0 s / 2.0 s pace delta:

- `None`-means-compute: gap=6.0, pace=2.0 (derived). `pace_delta_s=0.0` passed: pace stays **0.0**, gap still derived (6.0). `gap_ahead_s=0.0` passed: gap stays **0.0**. Both 0.0: both stay 0.0 and the rivals lookup is skipped (`needs_car_ahead` is False).
- The backend's deliberate pin therefore survives: `_local_build_race_state`'s `pace_delta_s=0.0` is never overwritten (`race_state_builder.py:313-316` uses `is None`, executed-verified, and the lap-7 harness shows backend pace 0.0 vs CLI/arcade 1.688).
- `rival` targeting: present rival recomputes (VER → 2.5/1.0); absent rival falls back to the ALREADY-RESOLVED values including pinned 0.0s (executed: rival absent + pinned → 0.0/0.0, no fabrication).
- Repo-wide grep for the trap pattern (`not pace_delta`, `pace_delta_s or`, `gap_ahead_s ... or`): the only `or`-fallbacks on these concepts are `mcp_tools.py:599` (recorded decision, see observation above) and two offline eval tools (`scripts/prompt_ab/gen_inputs.py:62`, `src/strategy/eval/decision_modes.py:303`) — none on the three runtime surfaces.

### [VERIFIED] Task 3 — every accepted behavioural delta lands where claimed and nowhere else (EXECUTED)

- **track_temp 40.0 → 35.0**: fires on weather-key-absent, temps-key-absent AND temps-present-None (the pre-#784 CLI would have crashed `float(None)` on present-None, and used 40.0 otherwise). Lusail is NOT perturbed: real readings 23.4/29.5 flow through on all five surface variants (harness) and in the real CLI run.
- **compound**: `None`/`""`/`"nan"`/`"NaN"`/`"None"`/`"none"`/missing-key → `"UNKNOWN"`; `"SOFT"` and `" MEDIUM "` (stripped) pass through. Lusail lap 7 real `MEDIUM` untouched.
- **tyre_life**: None/missing → 0; real 1 passes (no collision with fresh tyres).
- **total_laps**: missing AND present-None → 57 with the warning logged (captured live). Lusail real 57 comes from session_meta, not the fallback (verified: no warning on the harness path).
- **lap ladder**: top-level → driver-dict → 1-with-warning, all three rungs executed.
- **position None fails loud on ALL surfaces**: canonical builder, CLI `_build_race_state`, Arcade `_build_race_state`, backend `_local_build_race_state`, backend shim — all raise the same `ValueError` (executed). The shim `is` the canonical function object (no second implementation).

### [SELF-CORRECTED, not a finding] The `gp="Qatar"` RadioPipelineRunner rejection was this gate's own harness error

First harness pass fed `gp="Qatar"` to the backend RCM feed and got "Unknown GP 'Qatar'" → zero rcm_events. Verified against the real wiring before archiving: the arcade's `app.py:358-359 _resolve_gp_name()` maps the menu label through `GP_TO_LOCATION` ("Qatar"→"Lusail") BEFORE building the DTO, so the real arcade and webapp pass "Lusail". Harness re-run with "Lusail".

## Task 2 — real runs (second gate session, 2026-08-02)

Scope: Arcade driven for real (offscreen Qt dashboard fed by a real `SimConnector` + real TCP stream) and the telemetry backend over real HTTP (`/lap-state`, `/recommend`, `/simulate` SSE, #788 regression proof, 2024 control). The CLI (`f1-sim`) run is owned by the orchestrating session and is NOT duplicated here. Findings appended below as they are confirmed.

### [MEDIUM / env-docs] The documented backend launch command fails on a clean `uv sync` venv (EXECUTED)

`PYTHONPATH="." .venv/Scripts/python.exe -m uvicorn backend.main:app --app-dir src/telemetry --port 8000` (CLAUDE.md/memory wording) died at import: `backend/mcp_tools.py:24 → ModuleNotFoundError: No module named 'fastmcp'`. `fastmcp==3.2.0` is pinned ONLY in the submodule's `requirements.txt` (Docker path) and is absent from the parent `pyproject.toml`/`uv.lock`, so any `uv sync` prunes it. This gate installed `uv pip install fastmcp==3.2.0` (environment change, no repo file touched) to proceed. Backend then started clean: `Application startup complete`, `/docs` 200.

### [VERIFIED] #788 producer side, over real HTTP

`GET /api/v1/strategy/lap-state?gp=Lusail&driver=NOR&lap=7&year=2025` → **200**, and the body carries exactly the #788 producer shape: `weather = {air_temp: None, track_temp: None, track_temp_start: None, humidity: None, rainfall: 0}` while every ground-truth field matches Task 1 (lap_time 100.154, position 2, MEDIUM, tyre_life 7, gap_ahead 6.001, PIA interval −6.001). 2024 control (`year=2024`, same GP/driver/lap) → **200** with REAL weather `{air_temp: 19.0, track_temp: 22.9, humidity: 54.0}`.

### [VERIFIED] #788 old-vs-new builder, executed against the real HTTP payloads

The pre-change builder (`git show 7f394a8^:backend/utils/race_state_builder.py`, loaded as a module — no branch switch) vs the exact `lap_state` served over HTTP:

- OLD on 2025 lap 7: **raises `TypeError: float() argument must be a string or a real number, not 'NoneType'`** (`float(weather.get("air_temp", 25.0))` — present-None defeats the default). The live `/recommend` handler maps `TypeError` → 422, so this is the #788 mechanism.
- OLD on 2024 lap 7: OK (air 19.0 / track 22.9) — the bug was 2025-only, as the issue states.
- NEW (via the committed shim, `shim is canonical fn: True`): 2025 → OK with defaults air 25.0 / track 35.0, gap derived 6.001, pace 1.688; 2024 → OK with the real readings passing through untouched.

### [HIGH] #788 is NOT fixed end-to-end: `POST /recommend` still returns 422 on the 2025 reference lap (EXECUTED, real HTTP)

`POST /api/v1/strategy/recommend` with the real `/lap-state` body (Lusail 2025 lap 7 NOR, `gp_name="Lusail"`, `year=2025`) → **HTTP 422 in 56.4 s**, body:
`{"detail":{"error":"TypeError","agent":"orchestrator","detail":"float() argument must be a string or a real number, not 'NoneType'"}}`.

The canonical builder DID build the RaceState (verified in isolation above) — the crash is downstream. In-process reproduction with full traceback (same lap_state, same laps_df scoping as the endpoint):

- `race_situation_agent.py:1193 predict_sc_tool` → `:1009 _build_sc_features` → `:681 _compute_weather_features` → `float(session_meta.get('TrackTemp', 38.0))` → `TypeError`.
- The None enters at `race_situation_agent.py:1402-1404` (`run_from_state`'s session_meta build): `'AirTemp': wx.get('air_temp', 28.0)` etc. — the same present-key-None-value trap the canonical builder just fixed one layer up. Line 1416 right next to them IS guarded (`wx.get('track_temp_start') or 38.0`), and `pace_agent.py:642-644` guards the identical read with an explicit comment about this exact trap — the classic one-twin-fixed pattern.
- Second unguarded twin, silent-wrong class: `tire_agent.py:1512-1514` stores the same Nones into its session_meta; `_add_weather_cols` (tire_agent.py:582) then fills the 2025 frame's absent AirTemp/TrackTemp/Humidity columns with **None instead of the intended race averages** — no crash, silently degraded features (probe below).

Consequence: on this branch, `/recommend` for 2025 still fails — the fix moved the crash from the builder into N27. The orchestrator burns ~56 s of model warmup + sub-agent work (pace and tire complete) before dying. #788's "fixed by the canonical builder" claim holds for the BUILDER surface only, not for the endpoint the issue is actually about.

### [CORE DELIVERABLE] Task A — the Arcade, driven for real (EXECUTED; what was and was not rendered, stated plainly)

**What ran for real:** the full arcade strategy layer exactly as `F1ArcadeView._init_strategy_layer` wires it — a real `SimConnector` thread (model warmup, real radio corpus `qatar` with Whisper transcription ENABLED: "24 radios + 66 rcms", per-lap `run_strategy_pipeline` with real OpenAI LLM calls, laps 1–7), a real `TelemetryStreamServer` on 127.0.0.1:9998, and the REAL PySide6 dashboard (`MainWindow` + `TelemetryWindow`, the exact `__main__` wiring) in a separate process under `QT_QPA_PLATFORM=offscreen` + `QT_QPA_FONTDIR=C:/Windows/Fonts`, subscribing over real TCP and grabbed via Qt `grab()` at lap 7 + finished. Screenshots + machine payload: scratchpad `arcade_dashboard_lap7.png`, `arcade_telemetry_lap7.png`, `dashboard_last_payload.json`, `arcade_final_state.json`.

**What was NOT rendered — said prominently:** the pyglet/arcade REPLAY WINDOW (track, cars, leaderboard) was not created (no GL context headless), so `current_lap_provider=None` (the documented CLI/smoke path — no playback gating) and `lap_range=(1,7)`. The TelemetryWindow rendered its LAP 7 frame but with EMPTY speed/brake/throttle traces, because per-frame telemetry comes from the pyglet replay that was not running. Every strategy-side pixel, however, is the real dashboard rendering real pipeline output.

**The lap-7 rendered decision (READ from the PNG, not inferred):** header "Lusail · 2025 NOR · Connected · L 7"; orchestrator card **STAY OUT, confidence 88%**, Pace: MANAGE, Risk: BALANCED, "Pit: L12 · Next: HARD · UCUT: PIA"; scenario scores STAY +0.30 / PIT +0.21 / UCUT · OCUT "--"; PACE card pred 90.05s (chart shows the real lap-7 spike to ~100 s); TIRE "Cliff ~8 laps · L7" C2; **SITUATION "Threat HIGH", overtake 0%, safety car 100%"**; **PIT "pit 2.93s → HARD · SC", UCUT 64% → PIA**; **RADIO card: "YELLOW FLAG SECTOR · YELLOW FLAG SECTOR · SAFETY CAR DEPLOYED", "0 radios · 5 rcm", "RCM L7 SAFETY_CAR_DEPLOYED"**; **RAG "regulation loaded … no tyre changes … Art. 54.3"**; reasoning text cites the confirmed SC and Art. 54.3 overriding N28's reactive UNDERCUT.

**Answer to the attack question — the SC is NOT lost:** corpus → tracker → RaceState → N27 (`sc_prob_3lap=1.0`, `threat_level=HIGH`, `overtake_prob` forced 0 per Art. 55.8) → routing (`active=['N28','N30']`, pit `sc_reactive=True` proposing UNDERCUT) → Layer-3 synthesis reasoning about Art. 54.3 → rendered card. End-to-end on the REAL rendered surface.

**Lap-by-lap decisions (laps 1–7):** STAY_OUT ×7, confidence 0.98/0.97/0.98/0.94/0.95/0.91/0.88 — declining as the SC lap approaches. `finished=True`, `error=None`.

**vs the CLI's owned differences:** arcade lap-7 gap_ahead_s 6.001, lap_time 100.154, MEDIUM/7, P2 — all equal to the CLI-side ground truth; radio corpus real (0 radio_msgs at lap 7 is the corpus truth for NOR, not a wiring gap — 24 radios exist for the GP); risk_tolerance 0.5 both. Decision-level comparison with the orchestrating session's `f1-sim` run is left to the orchestrator, with the arcade's numbers above as this gate's half of the diff. Note the Layer-3 non-determinism warning below before reading any confidence delta as a contract break.

**Warnings captured in the arcade run (reported, not filtered):**
1. `WARNING src.agents.tire_agent: Tire tool output did not parse for C2 (tyre_life=1) — using conservative defaults instead of a 0.0 cliff` (lap 1).
2. `WARNING src.agents.strategy_orchestrator: Layer 3 temperature=0.0 was discarded by the client for model 'gpt-5.4-mini'. The orchestrator is sampling at the provider default, not running deterministically` — consecutive-lap confidences (and any CLI-vs-arcade decision diff) carry sampling noise by construction.
3. `Clamping LLM expected_stint_end 12 to anchor 50 (cliff_p50=57.0)` and `… to anchor 20 (cliff_p50=8.4)` (#433) — twice.
4. At interpreter exit: `qdrant_client … __del__ → ImportError: sys.meta_path is None, Python is likely shutting down` — benign-looking teardown noise from the RAG client lacking an explicit close; it would appear on real arcade exits too.

**[LOW / explained] N27's OUTPUT gap 0.0 while RaceState carried 6.001:** the situation card shows "gap 0.0s · Δpace +0.00s/lap". Cause found at `race_situation_agent.py:752-779 _parse_tool_outputs`: output gap/pace are regex-parsed from the overtake TOOL's message; under a confirmed SC the tool text doesn't yield them and they default to 0.0. The RaceState itself carried 6.001/1.688 (Task 1). Pre-existing output convention (the endpoint's `SituationResult` docstring already owns the 0.0-vs-None tension, #633) — recorded so nobody misreads the card as a builder regression.

### [VERIFIED] Task B — `/simulate` SSE, real HTTP, laps 1–7 with LLM (EXECUTED)

`POST /api/v1/strategy/simulate` (`{"year":2025,"gp":"Lusail","driver":"NOR","team":"McLaren","lap_range":[1,7],"no_llm":false,"provider":"openai"}`) → **HTTP 200 in 117.8 s**, well-formed SSE: `start`, 7 `lap` events, `summary` with `{"ok_laps": 7, "error_laps": 0}`. Zero `error` events. **Lap 7's event**: STAY_OUT conf 0.98, pos 2, gap 6.001, lap_time 100.154, MEDIUM/7, reasoning citing "a confirmed Safety Car deployment from radio/RCM" + Art. 54.3 — the SC context survives this surface too (its lap_state comes from `RaceStateManager`, real weather 23.4/29.5, so the N27 crash above does NOT fire here).

**Measured consequence of the owned `pace_delta_s=0.0` pin, on the reference lap:** the SSE lap-7 reasoning literally argues "the pace delta is only +0.000s, so there is no evidence of a performance falloff" and lands pit_lap_target=30, while the arcade (real pace_delta 1.688) reasons "+1.688s/lap off pace … tyre cliff P50 is lap 8.4" and lands pit_lap_target=12. Same lap, same models — the recorded "neutral" pin visibly changes the tactical plan. Not a #784 defect (the pin is a recorded decision), but its cost is now measured, not hypothetical. Confidence deltas (0.98 vs 0.88) additionally carry the documented Layer-3 sampling noise (temperature=0.0 discarded for 'gpt-5.4-mini').

### [MEDIUM] The SSE payload is blind to the sub-agents on the LLM path: `agent_alerts` always [], `agents_fired.pit/radio` always 0 (EXECUTED + file:line)

Observed on the real stream: lap 7 `agent_alerts=[]` and summary `agents_fired={"pit": 0, "rag": 1, "radio": 0}` — while the arcade, same lap, same rcm list, rendered 3 alerts (YELLOW ×2 + SAFETY_CAR_DEPLOYED) and an active N28 (`sc_reactive=True`). Not a routing divergence: `simulator.py::_parse_lap_decision` extracts `agent_alerts`/`_radio_out` only `if isinstance(result, dict)` (simulator.py:484-503) and the pit/radio counters only increment on the dict path (simulator.py:625-632) — the dict shape is the NO-LLM path, so with `no_llm=false` these fields are structurally empty regardless of what the agents did (`rag: 1` survives via the object-path `regulation_context` check at :634-635). The webapp's SSE consumer cannot see the SC alerts the arcade dashboard shows. Pre-existing (not introduced by #784); surfaced here because Task 1's "byte-identical rcm on four surfaces" made the empty alerts look like a lost SC — it is a payload-construction gap, with the routing itself verified equal.

### [VERIFIED] Task B — 2024 control over real HTTP (EXECUTED)

`POST /api/v1/strategy/recommend?year=2024` with the real 2024 lap_state → **HTTP 200 in 12.6 s**: STAY_OUT conf 0.92, pit_lap_target 13, compound_next HARD. The fix is not a 2025-only patch that broke the control case. (Backend log: Lusail 2024 radio/RCM parquets absent → ran without RCM, degraded gracefully as designed.)

### [LOW-MEDIUM / latent] `/recommend`'s laps_df year is a QUERY param that can silently disagree with the body's `year`

`require_laps_df(year: int = 2025)` is a FastAPI dependency, so `/recommend` exposes `?year=` (confirmed in the live openapi.json: `[('year','query',2025)]`) while `RecommendRequest.year` travels in the body. A client POSTing a 2024 body without `?year=2024` gets the 2025 frame silently — the agents then look up the driver's laps in the wrong SEASON's GP frame (the #429 family, one axis over). Pre-existing, not #784; this gate's own 2024 control had to pass BOTH to be correct.

### [MEDIUM] Even after the N27 crash is fixed, the backend-producer path silently degrades the tire model on every 2025 lap (EXECUTED probe)

`tire_agent.run_from_state` on the REAL backend lap_state (weather all-None) vs the same lap_state with the real RSM readings (23.4/29.5), same laps_df, same agent instance:

- weather=None (backend shape): deg_rate −0.3085, **cliff P10/P50/P90 = 10.1/10.5/10.9**
- weather=real readings: deg_rate −0.3085, **cliff P10/P50/P90 = 7.8/8.2/8.6**

No crash, no warning — the Nones from `tire_agent.py:1512-1514` flow through `_add_weather_cols` (:582, `df[col] = session_meta.get(col, 0.0)` — present-None defeats the 0.0 too) into the TCN's feature frame, and the cliff estimate comes out **2.3 laps more optimistic** on the reference lap. Optimistic-wrong is the dangerous direction (delays the pit call). Scope: any 2025 request whose lap_state came from the backend producer (`/lap-state` consumers, webapp strategy tab, MCP chat tools); the SSE and arcade paths are unaffected (RSM weather is real).

### Warnings inventory (both surfaces, reported unfiltered)

Arcade run: tire tool parse failure lap 1 (conservative defaults); Layer-3 temperature discarded for 'gpt-5.4-mini' (non-deterministic synthesis, documented); #433 clamps ×2; qdrant_client `__del__` ImportError at interpreter shutdown (no explicit close). Backend run: same tire parse + temperature + #433 clamps (×2, different anchors) during SSE; `F1_API_KEY not set` (expected, localhost); Lusail 2024 radio/RCM parquets missing (graceful); the #788 orchestrator TypeError (the HIGH finding). Nothing else appeared in either log.

### Environment deviations made by this gate (disclosed)

1. `uv pip install fastmcp==3.2.0` into the parent venv (see the env-docs finding; no repo file changed).
2. Arcade harness: no pyglet window (offscreen), `current_lap_provider=None`, `lap_range=(1,7)` — detailed in Task A above.
3. Old-builder proof loaded `7f394a8^` module content from git history — no branch switch, working tree untouched.

## What I tried to break and could not (Task 2)

- **The SC chain on the arcade, end-to-end on the RENDERED surface**: corpus (Whisper ENABLED this time, 24 radios + 66 rcms) → tracker → builder → N27 forced sc_prob 1.0 / overtake 0 → N28 sc_reactive + N30 Art. 54.3 → Layer-3 override of the UNDERCUT → the STAY OUT card in the real PySide6 pixels. Held at every link.
- **The SC chain on the SSE surface**: lap 7's event carries the confirmed-SC reasoning and Art. 54.3. Held (the alerts FIELD is blind — see the MEDIUM — but the decision itself sees the SC).
- **The canonical builder against the real backend payloads**: 2025 all-None weather → defaults, 2024 real weather → pass-through, shim `is` the canonical function, explicit-0.0 semantics all re-confirmed against live HTTP payloads rather than synthetic dicts.
- **The 2024 control**: could not make it regress — 200, sane recommendation, graceful no-RCM degrade.
- **A crash or error event anywhere in the SSE stream**: 7/7 laps ok, stream well-formed, heartbeat logic untriggered (<15 laps).
- **Cross-surface lap-7 disagreement beyond the owned differences**: every numeric field either agreed or traced to a named owner (pace pin, alerts payload gap, Layer-3 sampling). The scary-looking `agents_fired.pit=0` and `gap 0.0s` on the situation card both dissolved into located, pre-existing payload conventions (simulator.py:484-503/:625-632, race_situation_agent.py:752-779) — verified, not assumed.

What I could NOT verify: the pyglet replay window itself (never rendered — stated prominently in Task A); the radio-transcript half beyond corpus load (no NOR radio lands on lap 7, so N29's transcript path stayed data-empty end-to-end); the CLI side of the decision-level diff (owned by the orchestrating session, `f1-sim` not duplicated here per instructions).
