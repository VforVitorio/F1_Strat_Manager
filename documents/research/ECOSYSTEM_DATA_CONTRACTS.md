# Ecosystem data contracts (the connective tissue)

**Status:** reference spec, pinned 2026-07-07. **Scope:** interfaces only, no redesign.

The eight ecosystem design docs under `documents/research/` (box-bot, real-time OpenF1
consumer, pit-wall, pitlab, gridmind, radiogate, Rival Agent, repo integration) all
assume a small set of shared data contracts but none pins them field by field. This
document is that missing connective tissue: the exact, code-verified schemas that every
downstream repo builds against. Everything here is read from the current codebase (paths
cited per field group); nothing is speculative. Where a contract is planned but not yet
shipped (the WS relay, the HF pin manifest), it is marked PLANNED and traced to the
design doc that owns it.

The five contracts, in dependency order:

1. `lap_state` (per-lap race snapshot) - produced by `RaceStateManager`, consumed by everything.
2. `StrategyRecommendation` (orchestrator output, 14 fields, frozen v2).
3. Stream contracts (SSE simulation stream; Arcade TCP broadcast; planned WS relay).
4. HF artifact contract (repo_id -> revision pinning for datasets and models).
5. Contract versioning rules (additive vs breaking, who pins what).

---

## 1. The `lap_state` contract

**Source of truth:** `src/simulation/race_state_manager.py` (class `RaceStateManager`,
`get_lap_state` at line 338). This is the single dict every agent, surface, and future
live adapter consumes. The real-time consumer design (`REALTIME_OPENF1_CONSUMER_DESIGN.md`
section 2.1) commits to producing this IDENTICAL dict from OpenF1 live feeds, so the
schema below is binding for both replay and live producers.

### 1.1 Top level (`get_lap_state`, race_state_manager.py:338)

| Key | Type | Meaning |
|---|---|---|
| `lap_number` | int | 1-indexed lap being reported |
| `driver` | dict | Full-telemetry snapshot of OUR driver (1.2). Empty dict after DNF / out of range |
| `rivals` | list[dict] | Timing-screen view of every other driver (1.3), sorted by position ascending |
| `weather` | dict | Weather + track status snapshot (1.4) |
| `session_meta` | dict | Static session metadata (1.5) |

**Invariants** (from the `get_lap_state` docstring): the four top-level keys are ALWAYS
present, even when empty (empty `driver` dict = race-ended signal that callers must
handle). One dict per lap; an emitted `lap_state` is immutable (real-time doc,
"decisions are not retroactive").

### 1.2 `driver` (full telemetry tier, `get_driver_state`, race_state_manager.py:154)

All fields observability tier (a) observed-timing or an honestly labeled estimate,
per `PITWALL_REALISM_AND_TELEMETRY_SURFACE.md` section 2.2 (own-car table).

| Field | Type | Meaning |
|---|---|---|
| `driver` | str | Our FIA three-letter code (e.g. "NOR") |
| `team` | str | Team name as stored in the laps parquet |
| `lap_number` | int | Same lap as top level |
| `lap_time_s` | float \| None | Lap time, seconds, 3 decimals |
| `sector1_s` / `sector2_s` / `sector3_s` | float \| None | Sector times, seconds |
| `position` | int \| None | Classification position at end of lap |
| `gap_to_leader_s` | float \| None | Session elapsed time minus leader's, per FastF1 `Time` column (accurate under SC bunching) |
| `compound` | str | Current tyre compound name |
| `compound_id` | int \| None | Numeric compound id (model feature encoding) |
| `tyre_life` | int \| None | Laps on the current set |
| `stint` | int \| None | Stint number |
| `fresh_tyre` | bool | Whether the set was fitted new |
| `speed_i1` / `speed_i2` / `speed_fl` / `speed_st` | float \| None | All four speed-trap readings (km/h) |
| `fuel_load` | float \| None | LINEAR-DEPLETION ESTIMATE, not a measurement (tier (c) even for the own car; pit-wall doc 2.2) |
| `track_status` | str | FastF1 TrackStatus code string |
| `is_in_lap` | bool | PitInTime present on this lap |
| `is_out_lap` | bool | PitOutTime present on this lap |

### 1.3 `rivals[]` entries (timing tier, `get_rival_states`, race_state_manager.py:219)

The single-driver boundary: rivals expose ONLY what a real pit wall sees on the live
timing monitor. No sector times, no fuel, no speed traps beyond SpeedST. The pit-wall
doc (section 2.4) audited this boundary and confirmed nothing here leaks tier (c).
Tier tags use the R4 vocabulary from that doc: `observed-timing`, `observed-broadcast`,
`derived`, `hidden-modeled`.

| Field | Type | Tier | Meaning |
|---|---|---|---|
| `driver` | str | observed-timing | Rival FIA code |
| `team` | str | observed-timing | Rival team name |
| `position` | int \| None | observed-timing | Position at end of lap (None sorts last) |
| `lap_time_s` | float \| None | observed-timing | Rival lap time |
| `compound` | str | observed-timing / derived | On the timing feed today; live-parity discipline says also derivable from observed pit events (pit-wall doc 2.4, R2) |
| `tyre_life` | int \| None | observed-timing / derived | Same caveat as compound |
| `stint` | int \| None | derived | Stint counter from observed stops |
| `speed_st` | float \| None | observed-timing | Speed-trap reading (the only rival speed exposed) |
| `gap_to_leader_s` | float \| None | observed-timing | Rival cumulative time minus leader's |
| `interval_to_driver_s` | float \| None | derived | Rival cumulative minus OUR cumulative; positive = rival ahead |
| `is_pitting` | bool | observed-timing | PitInTime present this lap |

Explicitly NOT in the rivals contract (tier (c) hidden, per pit-wall doc 2.2): raw car
telemetry (throttle / brake / steering / ERS), fuel load, engine and energy modes, true
degradation or remaining grip, team intent, un-broadcast pit calls. Coarse broadcast
car data (`observed-broadcast`) is real and public but deliberately excluded from
`lap_state` and from Rival Agent v1 features; it may only appear on dashboards behind
an explicit "broadcast data" label (pit-wall doc 2.2, R3).

### 1.4 `weather` (`get_weather_state`, race_state_manager.py:283)

| Field | Type | Meaning |
|---|---|---|
| `track_status` | str | Always present (from laps parquet) |
| `air_temp` / `track_temp` | float \| None | Degrees C, only when weather.parquet is provided |
| `humidity` | float \| None | Percent |
| `wind_speed` | float \| None | m/s |
| `rainfall` | bool | Defaults False |

Caveat (P5 audit F-14): the weather row is picked by linear lap-fraction interpolation,
adequate for replay, and must be re-specified by any live producer.

### 1.5 `session_meta` (`get_session_meta`, race_state_manager.py:322)

| Field | Type | Meaning |
|---|---|---|
| `gp_name` | str | Grand Prix name (keys circuit-specific agent thresholds) |
| `year` | int | Season year |
| `driver` | str | Our driver code (duplicated here for stateless consumers) |
| `team` | str | Our team |
| `total_laps` | int | Total completed laps in the race |

---

## 2. The `StrategyRecommendation` contract (orchestrator v2, frozen)

**Source of truth:** `src/agents/strategy_orchestrator.py:317` (Pydantic `BaseModel`).
The 14-field v2 schema is FROZEN by decision (memory `project_orchestrator_v2_schema`):
output richness is fixed via prompt engineering, never by adding fields. The
LLM-vs-code column is load-bearing for box-bot's never-invent-numbers guardrail
(`BOX_BOT_DESIGN.md` section 5): numbers placed by code are trustworthy verbatim;
LLM-written numeric fields are schema-bounded but still model-produced.

Enums (strategy_orchestrator.py:200-204): `action` in {STAY_OUT, PIT_NOW, UNDERCUT,
OVERCUT, ALERT}; `pace_mode` in {PUSH, NEUTRAL, MANAGE, LIFT_AND_COAST}; `risk_posture`
in {AGGRESSIVE, BALANCED, DEFENSIVE}; `compound_next` in {SOFT, MEDIUM, HARD};
contingency `priority` in {HIGH, MEDIUM, LOW}.

| # | Field | Type | Filled by | Meaning |
|---|---|---|---|---|
| 1 | `action` | Literal (5 values) | LLM (MC-grounded) | Primary decision; maps 1:1 to an MC candidate and a UI badge |
| 2 | `reasoning` | str | LLM | Narrative synthesis; the human-readable "why", never machine-parsed |
| 3 | `confidence` | float [0,1] | LLM | Self-assessed certainty; qualitative, NOT calibrated |
| 4 | `pit_lap_target` | int \| None | LLM | Absolute lap of the planned stop; None = no stop in horizon |
| 5 | `compound_next` | Literal \| None | LLM | Compound for the next stint |
| 6 | `undercut_target` | str \| None | LLM | Rival code for UNDERCUT / OVERCUT only |
| 7 | `pace_mode` | Literal, default NEUTRAL | LLM | Driving instruction for the next laps |
| 8 | `target_lap_time_s` | float \| None | LLM (CI-grounded) | Target lap time, grounded in N06 PaceOutput CI bounds |
| 9 | `risk_posture` | Literal, default BALANCED | LLM | Auditable risk stance |
| 10 | `contingencies` | list[Contingency], max 4 | LLM | Conditional branches; each = {trigger: str, switch_to: action-Literal, priority: Literal, rationale: str} (strategy_orchestrator.py:240-243) |
| 11 | `key_risks` | list[str], max 5 | LLM | Top flagged risks as short bullets |
| 12 | `expected_stint_end` | int \| None | LLM | Planned end lap of current stint (mainly STAY_OUT) |
| 13 | `scenario_scores` | dict | **code, post-hoc** | Full MC output per strategy: {action: {E, P10, P90, score}}; attached after the LLM call |
| 14 | `regulation_context` | str, default "" | **code, post-hoc** | N30 RAG answer verbatim when activated, else empty |

Guardrail note for box-bot: fields 13-14 are code-attached and safe to quote as data.
Fields 4, 8 (numeric, LLM-written) are schema-constrained but must pass box-bot's
checker against the underlying `lap_state` / MC data before publication; field 3 is
explicitly non-calibrated and must never be phrased as a probability.

---

## 3. The stream contracts

Three transports exist or are pinned by design. Box-bot, the pit-wall SPA, and the
real-time consumer all attach here rather than importing core code.

### 3.1 SSE simulation stream (SHIPPED)

**Endpoint:** `POST /api/v1/strategy/simulate`
(`src/telemetry/backend/api/v1/endpoints/strategy.py`, tail of file; media type
`text/event-stream`). Event models: `src/telemetry/backend/services/simulation/simulator.py`
(all Pydantic, `extra="forbid"`).

**Wire framing** (strategy.py:902-925): every event is one SSE frame
`data: <json>\n\n` where the JSON envelope is `{"type": <event-name>, "data": <payload>}`
(simulator.py:744-799). A comment heartbeat frame (`:` line) is sent every 15 lap
events to survive proxy idle timeouts. Order: exactly one `start`, then one `lap` OR
`error` per processed lap, then exactly one `summary`. A stream-level failure emits a
final `error` with `lap=0` instead of crashing the response.

| Event `type` | Payload model (simulator.py) | Fields |
|---|---|---|
| `start` | `StartEvent` (line 89) | `gp, year, driver, team, lap_start, lap_end, total_laps, no_llm, provider, timestamp` (`driver2?` removed by `F1_Telemetry_Manager#219`: it reached the stream and no consumer read it) (total_laps authoritative; lap_start/end = effective window) |
| `lap` | `LapDecision` (line 110) | `lap_number, compound, tyre_life, position, lap_time_s?, gap_ahead_s, action, confidence, reasoning, scenario_scores: dict[str,float], pace_mode?, risk_posture?, pit_lap_target?, compound_next?, undercut_target?, guardrail_reason?` (`agent_alerts` removed by `F1_Telemetry_Manager#219`: it was a lossy copy of `per_agent.radio.alerts` for a retired Qt dashboard) |
| `error` | `ErrorEvent` (line 140) | `lap, message` (stream continues; consumer renders a partial race) |
| `summary` | `RunSummary` (line 151) | `status, positions, actions, agents_fired, stint, timing, gap_ahead, mc_confidence_series: list[float], best_decision, worst_decision, time_compression: float, reasoning_tokens: dict[str,int]` |

`LapDecision` is the projection of contract 2 onto the wire: it carries the
strategy fields (`action` through `undercut_target`) plus `lap_state`-derived context
(`compound`, `tyre_life`, `position`, `lap_time_s`, `gap_ahead_s`). Optional strategy
fields are None on the `--no-llm` path. `LapDecision`/`RunSummary` were declared the
frozen contract at delivery (memory `project_sim_sse_endpoint_done`): any change MUST
be additive (new optional fields only).

Request config (`SimConfig`, simulator.py:64): `year, gp, driver, team, driver2?,
lap_range?, risk_tolerance=0.5, no_llm=False, provider="lmstudio", interval_s=0.0`.
Provider is OpenAI or LM Studio only, propagated via `F1_LLM_PROVIDER`
(process-wide; simulator.py:182-191).

### 3.2 Arcade TCP broadcast (SHIPPED, local-only)

**Source:** `src/arcade/stream.py` (non-blocking TCP server, 127.0.0.1:9998) with the
payload assembled in `src/arcade/app.py:426-436`. Wire format: newline-delimited JSON,
one dict per frame, pushed at >= 5 Hz (design target ~10 Hz), payload <= ~10 KB, no
heartbeat (cadence is the liveness signal). Payload top level:

| Key | Meaning |
|---|---|
| `arcade` | Compact per-frame snapshot for the dashboard (`_build_arcade_snapshot`) |
| `strategy` | Strategy state snapshot with bounded history tail (`snapshot_dict(STREAM_HISTORY_TAIL)`); carries the per-lap decision data |
| `playback` | `{speed, paused, frame_index, total_frames}` (Arcade is the master clock) |

This payload is currently UNVERSIONED and consumed only by the Qt dashboard
(`dashboard/stream_client.py`). The pit-wall doc flags that the moment the relay (3.3)
ships, this becomes a real cross-repo contract and needs a schema version field
(pit-wall doc, "the broadcast schema becomes a real contract").

### 3.3 Backend WS relay (PLANNED, pit-wall doc option O1)

**Owner:** `PITWALL_REALISM_AND_TELEMETRY_SURFACE.md` (options table and Phase 1).
The backend runs a TCP client to 127.0.0.1:9998 and re-publishes each 3.2 payload over
a WebSocket endpoint for the browser SPA. Pinned decisions: transport = WebSocket, not
SSE (seek / pause needs upstream messages; open question Q3 tracks the final call);
the relay carries the 3.2 payload plus a minimal versioned envelope; Arcade remains
the master clock; the relay lives in the submodule backend next to the existing SSE
endpoints. End state O2 (backend-native replay streamer) reuses the `/simulate` SSE
generator as its lap-cadence seed, so 3.1's event vocabulary is the seed schema there.

Consumer guidance (all three transports): parse `type`, ignore unknown keys, treat
unknown event types as skippable. That posture is what makes additive evolution safe.

---

## 4. The HF artifact contract

**Current state (verified by P5 audit, `AUDIT_P5_DATA_ENGINEERING.md:96-102`):** one
flat dataset repo `VforVitorio/f1-strategy-dataset` holding `models/` and `data/` at
the repo root (layout confirmed against `data_cache.py` and re-verified 2026-07-04),
downloaded via `snapshot_download` with `revision="main"` (`data_cache.py:58-59`).
That means today's "pin" is MUTABLE: any Hub push silently changes what every
installed CLI pulls. This is P5's headline finding and the thing this contract fixes.

**Target contract (`ECOSYSTEM_REPO_INTEGRATION.md` sections 3-4):**

| Rule | Spec |
|---|---|
| Immutable pins | Every cross-repo HF dependency is pinned `revision=<commit hash or tag>` on `snapshot_download` / `hf_hub_download`; never `main` in released builds |
| Pin manifest | ONE manifest (constants module or JSON next to `data_cache.py`) mapping `repo_id -> revision` for every consumed artifact; "what data does this release use" answers in one file |
| Bump ritual | Editing the pinned revision is an explicit, reviewable commit + PR in the consumer repo |
| Dev vs release | Proposal (P5 open question 4): pinned revisions in released CLI builds; `main` allowed only in dev checkouts |
| Org migration | Artifacts move under the `f1stratlab` HF org (P5 Phase 3); old repo id documented as deprecated redirect |
| Reproducibility tuple | A core release = git tag + submodule gitlink SHAs + HF revision manifest + `uv.lock` (repo-integration doc, section 4) |
| Dataset card | Each artifact documents schema, era coverage (2022-2025), naming convention, and provenance (which corpus revision a model was trained on) |

**How a downstream repo pins:** gridmind pins `f1stratlab/f1-domain-corpus` (dataset)
and publishes `f1stratlab/strat-gemma-lora` (model) at tagged revisions; the core and
box-bot then pin the LoRA revision + LM Studio model name in config. pitlab is in-core
(pip-pinned tracker only). Contract regression tests live in the CONSUMER repo, run
against its pinned revision, so failures surface where the pin can be fixed
(repo-integration doc, CI note).

---

## 5. Contract versioning rules

How a downstream repo knows what it is building against and when a break lands.

1. **Additive by default.** `lap_state` and the SSE events tolerate new keys: consumers
   must ignore unknown fields (verified for `lap_state` by the P5 audit, cited in
   `REALTIME_OPENF1_CONSUMER_DESIGN.md`: "Additive keys are allowed (the P5 audit
   verified the contract tolerates them)"). New fields are optional-with-default.
   Note the nuance: the SSE Pydantic models set `extra="forbid"`, which binds the
   PRODUCER (the core cannot emit undeclared keys by accident); consumers must still
   parse leniently.
2. **Breaking = renaming, removing, retyping a field, changing an enum value, or
   changing event ordering / framing.** Any of these requires a major schema version
   and a core release that announces it.
3. **Schema artifacts (PLANNED, P5 F-15, `AUDIT_P5_DATA_ENGINEERING.md:151,168`):**
   publish `lap_state` as a versioned JSON-schema artifact (with the additive-keys rule
   and the weather F-14 caveat written into it). The real-time doc adopts this as its
   layer L0 (golden `lap_state` fixtures + parity tests, with Testing epic #181/#182);
   the pit-wall doc R5 makes the same fixtures the relay's parity gate. One schema file,
   three enforcement points.
4. **Version signals per contract:** `StrategyRecommendation` is frozen at v2 (schema
   changes only via a deliberate v3 decision, see section 2); the SSE stream is
   versioned implicitly by the API path (`/api/v1/...`) plus core release tags; the
   broadcast/relay payload gains an explicit schema-version field when the relay ships
   (3.3); HF artifacts are versioned by revision hash/tag (4).
5. **Downstream pinning:** each consumer pins a core RELEASE TAG (which transitively
   fixes the stream schema + the HF revision manifest + `uv.lock`) and bumps it
   deliberately. box-bot "pins the core's stream contract version on ITS side (API
   version + release tag)" (repo-integration doc, integration table). No downstream
   repo ever imports core internals to read these schemas.

---

## 6. Who consumes what

| Consumer | lap_state | StrategyRecommendation | Streams | HF artifacts |
|---|---|---|---|---|
| **box-bot** | Via stream payloads only | Via `LapDecision` fields; code-attached vs LLM split drives the numeric guardrail | SSE preferred (one-directional; `BOX_BOT_DESIGN.md`) | gridmind LoRA revision (for phrasing LLM via LM Studio / OpenAI-compatible) |
| **real-time consumer** | PRODUCES it (live adapter emits the identical dict) | Receives it from the same orchestrator loop | Feeds the same SSE/WS surfaces | Model artifacts via the pin manifest |
| **pit-wall SPA** | Rendered per tier tags (1.3) | Rendered decision panel | WS relay of the Arcade broadcast (3.2 + 3.3) | None directly |
| **pitlab** | Training features derive from the same parquet schema behind it | Logs decisions as run artifacts | None | Publishes model artifacts + revisions consumed by the core |
| **gridmind serving** | Not directly (text corpus domain) | Its LoRA writes the LLM-side fields of contract 2 through the provider layer | None | Pins corpus revision; publishes LoRA revision |
| **Arcade / SPA (in-core)** | Direct (replay engine) | Direct | TCP broadcast producer / SSE consumer | Via `data_cache.py` |

---

## 7. Risks

- **The mutable `main` pin is live today** (4). Until the pin manifest ships, every
  installed CLI and every downstream repo is exposed to silent Hub-side changes. This
  is the highest-leverage single fix in this document.
- **`lap_state` has no machine-readable schema yet** (P5 F-15): three consumers
  (`to_arcade_frame` in `replay_engine.py`, the SPA, the future live adapter) mirror it
  by hand. Drift is undetectable until the golden fixtures (L0 / R5) exist.
- **The broadcast payload is unversioned** at the exact moment the relay design turns
  it into a cross-repo contract. The version field must land with the relay, not after.
- **Guardrail dependency on the code/LLM split** (2): if a future change moves a
  code-attached field into the LLM fill path (or vice versa), box-bot's trust model
  silently breaks. Treat the "Filled by" column as part of the frozen schema.
- **`confidence` misuse:** it is documented non-calibrated; any consumer surfacing it
  as a probability (bot copy, dashboards) misrepresents the system.

## 8. Open questions

1. **Pin manifest shape:** single JSON next to `data_cache.py` vs a constants module
   (repo-integration doc open question 3). Blocks contract 4 execution.
2. **Relay transport final call:** WebSocket (recommended, bidirectional-ready) vs SSE
   (simpler, reuses `eventsource-parser`); pit-wall doc Q3.
3. **Where does the `lap_state` JSON-schema artifact live** and which repo owns its CI
   check: core (producer-side) plus fixture copies in consumers, per L0/R5, but the
   publication channel (repo file vs HF artifact) is unpinned.
4. **Broadcast schema version field naming and placement** (top-level `schema_version`
   in the 3.2 payload is the natural spot); to be fixed in the relay PR.
5. **Dev-vs-release pin policy ratification** (P5 open question 4: `main` in dev,
   pinned in releases) needs an explicit yes before `data_cache.py` grows `revision=`.
