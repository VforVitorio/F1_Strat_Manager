# Real-Time OpenF1 Consumer (design)

**Status: research design, forward plan. Design only, no code, no commitments.**
**Date: 2026-07-06.**

This document designs ecosystem initiative 5, the live piece: a consumer of OpenF1's
real-time feed that produces the SAME `lap_state` contract the replay produces, so the
six sub-agents and orchestrator N31 run unchanged on a live race weekend. Kafka was
explicitly descoped (one source, one pipeline, one or two UIs; a broker solves a
scale problem this project does not have); the design stays at the level of an async
ingestor plus in-process queues.

Hard constraints honored throughout: design only, no code; `scripts/run_simulation_cli.py`,
`src/agents/` internals, and `notebooks/**` are untouchable (the consumer is an additive
producer, nothing downstream changes); the backend stays FastAPI; LLM provider is OpenAI
or LM Studio, never Anthropic.

Documents this builds on (read, not re-planned):
`src/simulation/race_state_manager.py` (the contract),
`documents/research/PITWALL_REALISM_AND_TELEMETRY_SURFACE.md` (observability tiers,
fan-out options O1/O2/O3), `documents/research/RIVAL_AGENT_DESIGN.md` (derive-do-not-read
discipline), `documents/audits/AUDIT_2026_REG_CONCEPT_DRIFT.md` (epic #189, the model
gate), `documents/research/ECOSYSTEM_REPO_INTEGRATION.md` (placement rule).

---

## 1. Framing

The architectural bet of the whole system is a single dict contract. Replay builds it
from FastF1 parquets; the agents never learn where it came from:

- `RaceStateManager.get_lap_state(lap)` returns `{lap_number, driver, rivals, weather,
  session_meta}`. The four top-level keys are always present, the schema is stable, and
  every consumer reads keys defensively.
- The orchestrator decides per COMPLETED LAP, not per telemetry tick. That cadence
  matches real pit-wall operations and is the property that makes live mode cheap: the
  live consumer only has to close laps correctly, not stream at 10 Hz into the agents.

The deliverable is therefore one new producer: a **live state manager** that assembles OpenF1
messages into the identical `lap_state` dict and hands it to the same orchestrator loop,
the same backend streams, and the same UI fan-out. Replay and live become two producers
of one contract. Everything below is in service of that single sentence.

What this is NOT: a re-architecture, a message broker, a new agent, or a change to any
model. The old live-timing memo estimated the delta as "one adapter file plus a few
lines of wiring"; this design keeps that spirit while being honest about the two places
the estimate was optimistic: observability semantics (section 3) and stream assembly
robustness (section 4).

---

## 2. The contract-preserving core

### 2.1 Two producers, one contract

| | Replay (exists) | Live (this design) |
|---|---|---|
| Source | FastF1 parquets (post-hoc, corrected) | OpenF1 real-time feed + REST backfill |
| Builder | `RaceStateManager` (O(1) lookups over preprocessed DataFrames) | `LiveRaceStateManager` (incremental assembly from messages) |
| Tick | Caller iterates laps 1..N at will | Emitted when OUR driver's lap closes |
| Output | `lap_state` dict | Identical `lap_state` dict |
| Consumers | Orchestrator, CLI, Arcade, backend SSE, Streamlit | The same, unchanged |

The live manager mirrors the public surface of the replay manager: `get_lap_state`,
`get_driver_state`, `get_rival_states`, `get_weather_state`, `get_session_meta`, same
key sets, same null semantics (a field the feed has not delivered yet is `None`, exactly
like a NaN parquet cell today). It does not subclass or modify `RaceStateManager`; it is
a sibling producer in a new module, so the untouchable replay path is never edited.

### 2.2 Internal shape (three layers, all in one process)

1. **Transport** (section 4.1): connects to OpenF1, yields raw messages per stream
   (laps, intervals, position, stints, pit, race_control, weather). Swappable:
   real-time push, REST polling, or the shadow-replay harness (section 6.1) behind one
   interface.
2. **Assembler** (section 4.3): routes messages into per-driver, per-lap accumulators;
   detects lap closure; maintains watermarks and staleness metadata.
3. **Emitter**: when our driver's lap L closes, snapshots the accumulators into a
   `lap_state` dict for lap L and pushes it to consumers. One dict per lap, never
   retroactively mutated (section 4.5).

### 2.3 Contract parity as a tested invariant

The `lap_state` schema must be verified, not promised. Adopt the pit-wall doc's R5
boundary parity tests: one golden fixture asserting the live emitter produces exactly
the replay key sets (driver whitelist, rival whitelist, weather, session_meta), and one
cross-producer test running the shadow-replay harness (section 6.1) and the parquet
replay over the SAME race, asserting shape equality per lap and value agreement within
documented tolerances. This lands with the Testing epic fixtures (#181/#182), which is
another reason those come first.

Additive keys are allowed (the P5 audit verified the contract tolerates them). The live
producer adds only: a `live` block inside `session_meta` (feed latency, last watermark,
staleness flags per rival) so downstream surfaces can render data-age honestly. Agents
ignore it; dashboards use it.

---

## 3. The live observability reality (the key design point)

### 3.1 What live OpenF1 actually exposes

OpenF1's feed is the public mirror of the FIA/FOM distributed streams (rows 2-4, 6, 7
of the pit-wall doc's feed inventory). Per stream, at race time:

| Stream | Contents | Cadence / lag |
|---|---|---|
| `laps` | Per driver per lap: lap duration, three sector durations, I1/I2/ST speeds, out-lap flags | Row completes seconds after the lap ends |
| `intervals` | Gap to leader and interval to car ahead, all cars | ~4 s |
| `position` | Running order changes | Event-driven |
| `stints` | Compound, stint number, tyre age at fitting, per driver | Appears shortly after each out-lap |
| `pit` | Pit lane entry/exit with lane duration | Event-driven |
| `race_control` | Flags, SC/VSC, investigations, track status changes | Event-driven, official |
| `weather` | Air/track temp, humidity, wind, rainfall | ~1 min |
| `car_data` | Broadcast telemetry: speed, RPM, gear, throttle %, brake flag, DRS | ~3-4 Hz, all cars |
| `location` | GPS positions on track | ~3-4 Hz, all cars |
| `team_radio` | Radio clips, all teams | Curated, ~30 s+ delay |

The critical fact: **this is timing-screen-tier data for ALL cars, symmetrically.** A
public consumer is not a team. There is no private high-rate channel for "our" car:
no ERS deployment, no brake pressures, no fuel flow, no tyre temperatures. The pit-wall
doc's tier (c) (hidden) applies to our own driver too, since the project is not the actual team.

### 3.2 How the single-driver boundary degrades, honestly

The replay framing is "our driver gets full telemetry, rivals get timing-screen only."
Live, that asymmetry weakens in a specific and fortunate way:

- **The data asymmetry collapses.** Live, our driver and every rival sit in the same
  public tier. Nothing privileged exists to fetch.
- **The contract barely notices.** This is the fortunate part: `get_driver_state` was
  designed conservatively. Its fields are timing-tier values (lap/sector times, speeds,
  position, gap, compound, tyre life, stint, pit flags, track status) plus one honest
  estimate (`fuel_load`, a linear depletion model, not a measurement, per the module
  docstring). None of the driver fields encodes a private channel. So the dict survives
  live almost intact; what changes is the STORY told about it, not its shape.
- **The asymmetry that survives is modeling attention, not data.** Live, "our driver"
  means: the car whose estimators run continuously (degradation TCN state, fuel
  model, stint plan), with rivals tracked at the same observational tier but without a
  persistent modeled state. That is also a fair description of how a real wall treats
  the other 19 cars. The docstring claim "mirrors a real pit wall" gets STRONGER for
  rivals and semantically softer for our car; the design accepts this and documents it
  rather than faking a private feed.

### 3.3 Field-by-field verdict for the driver dict

| `lap_state["driver"]` field | Live source | Verdict |
|---|---|---|
| `lap_time_s`, `sector1_s..sector3_s` | `laps` | Available. Provisional values (live timing can correct post-hoc; decisions never see corrections, section 4.5) |
| `position` | `position` (latest at lap close) | Available |
| `gap_to_leader_s` | `intervals` sampled at lap close | Available; source differs from replay's session-time subtraction, tolerance documented in the parity test |
| `compound`, `stint` | `stints` | Available, with a few seconds of feed lag after an out-lap |
| `compound_id` | Local encoding map (`data/models/tire_degradation/encoding_maps.json`) | Derivable |
| `tyre_life` | `stints` tyre age at fitting + laps counted in stint | Derivable (the derive-do-not-read discipline from the Rival design, applied to our own car) |
| `fresh_tyre` | Tyre age at fitting == 0 | Derivable, noisy for scuffed sets (known limitation, flagged) |
| `speed_i1`, `speed_i2`, `speed_st` | `laps` | Available |
| `speed_fl` | Not in OpenF1 laps | **Gap: null live.** Consumers already tolerate None; no model depends on it as a hard input |
| `fuel_load` | Same linear depletion estimate as replay (race distance known) | Estimate, identical logic both modes |
| `track_status` | Mapped from `race_control` events to the FastF1 status codes | Derivable (one small mapping table, kept in the live module) |
| `is_in_lap`, `is_out_lap` | `pit` events | Available |

### 3.4 Field-by-field verdict for the rivals list

Every rival field (`position`, `lap_time_s`, `compound`, `tyre_life`, `stint`,
`speed_st`, `gap_to_leader_s`, `interval_to_driver_s`, `is_pitting`) maps directly:
`laps` + `intervals` + `stints` + `pit` + `position`. `interval_to_driver_s` is derived
as rival gap-to-leader minus our gap-to-leader from the same `intervals` snapshot. The
rivals tier is FULLY populatable live; the replay boundary was calibrated to exactly
this feed. The pit-wall doc's R1 additive rival fields (sector times, I1/I2 speeds,
out-lap flag) also come from the same `laps` stream, so the live producer populates
them for free once R1 lands in replay.

### 3.5 Agent and model coping matrix

No model in the stack consumes private telemetry channels; all were trained on
FastF1 timing-tier features. The live degradations are therefore small and specific:

| Component | Live-available? | Degradation and coping |
|---|---|---|
| N06 pace (XGBoost) | Yes | `fuel_corrected_lap_time` uses the fuel estimate, same as replay. Provisional lap times add noise vs corrected parquets; acceptable at decision grade |
| Tire TCN + MC Dropout | Yes | Needs the per-stint lap sequence, all timing-tier. Cold start: rolling features need ~3 laps into a stint; the existing opening-lap guardrails already suppress low-information calls |
| N12 overtake (LightGBM) | Yes | Pair features from gaps, pace deltas, tyre ages, all live |
| N13 SC probability | Yes | Contextual prior from circuit + race state; `race_control` gives SC/VSC truth the instant it is official |
| N15 pit duration | Yes | Circuit-level priors; live `pit` lane durations refine the day's actual pit-loss estimate |
| N16 undercut | Yes | Gaps + tyre ages + pit loss, all live |
| Radio agent (N29 + Whisper) | Partial | `team_radio` arrives curated with ~30 s+ delay, plus ~5-10 s Whisper transcription. Fine at lap cadence (~90 s laps); the agent treats radio as trailing context, never as the lap trigger |
| RAG agent (N30) | Yes | FIA regulations are static |
| Orchestrator N31 | Yes | Per-completed-lap cadence unchanged; Monte Carlo and guardrails identical. LLM synthesis latency (seconds) fits inside a lap |

**Verdict:** all six models and the orchestrator run on live-available features. The
coping needed is: tolerate `speed_fl = None`, tolerate provisional timing, respect the
existing cold-start guardrails, and treat radio as delayed context. No fallback model
is required; the fallback is the guardrail layer that already exists.

---

## 4. Streaming mechanics

### 4.1 Transport

Two transports behind one ingestor interface, chosen by config:

- **Real-time push** (primary for race day): OpenF1's real-time offering (MQTT over
  WebSocket at the time of writing; requires a paid/registered account). Terms and
  transport details must be re-verified when implementation starts (open question Q1).
- **Disciplined polling** (fallback and rehearsal): REST polling of the same endpoints
  keyed by `session_key` and `date > last_watermark`, at ~4-5 s. OpenF1's REST serves
  live sessions with a few seconds of delay; at lap cadence this is fully sufficient.
  The polling transport is not a degraded afterthought; it is the guaranteed-available
  baseline and what the shadow-replay harness (section 6.1) reuses.

Both yield the same normalized message envelopes (stream name, driver number where
applicable, feed timestamp, payload), so assembler and emitter never know which
transport ran.

### 4.2 Clocks

Feed timestamps are authoritative; wall clock is never used for ordering. The consumer
keeps one watermark per stream per driver (latest feed timestamp applied). Session
elapsed time derives from feed dates relative to the session start, replacing the
replay's `Time` column role.

### 4.3 From async streams to per-lap state

The streams are asynchronous and mutually unordered; `lap_state` is a per-lap snapshot.
Assembly rules:

- **Per-driver lap accumulators**, keyed by (driver number, lap number). `laps` rows
  carry the lap number explicitly; event streams (`pit`, `position`, `intervals`,
  `race_control`, `weather`) are timestamped and attributed to the lap whose window
  contains them, using each driver's lap start/end times from the `laps` stream.
- **Keep-latest vs append-only**: `intervals`, `position`, `weather`, `car_data` are
  keep-latest per driver (only the value nearest lap close matters); `laps`, `pit`,
  `stints`, `race_control` are append-only facts that must never be dropped.
- **Lap closure for OUR driver** = the tick. A lap closes when its `laps` row arrives
  with a lap duration (the row lands seconds after the car crosses the line). On
  closure, the emitter snapshots lap L for the driver and, for each rival, the freshest
  data at that instant.
- **Rivals may be mid-lap at our lap close.** That is physically correct (a timing
  screen shows exactly this). Rival fields carry the last COMPLETED lap's values plus
  live gap/interval, with per-rival staleness metadata in the `live` block. Lapped
  cars and retirements degrade to stale-then-absent, mirroring how the replay handles
  DNFs (empty or missing rows, consumers already tolerate it).

### 4.4 Reconnection and gap recovery

- On disconnect: exponential backoff reconnect on the push transport, while the polling
  transport takes over immediately (both are always constructed; polling idles when
  push is healthy).
- On reconnect: REST backfill of every stream from the per-stream watermark, applied
  through the same assembler (idempotent upserts keyed by natural keys: driver + lap
  for laps, driver + stint number for stints, timestamps for events). Ticks missed
  during an outage are emitted in order on catch-up; the orchestrator processes them
  sequentially like fast replay laps.

### 4.5 Out-of-order, late, and corrected data

- **Grace, then emit, then freeze.** The emitter waits a short grace window (a few
  seconds) after our lap row arrives, to let the intervals/stints laggards land, then
  emits. An emitted `lap_state` is immutable: strategy decisions are not retroactive,
  and mutating history would poison the agents' per-lap reasoning trail.
- **Late corrections** (timing corrections, deleted laps) are applied to the internal
  store and logged, so post-session analysis and the parity tests see them, but no
  re-emission occurs. This is the same posture a wall takes: decisions are made on the
  screen that was available at the time.
- **Out-of-order within the grace window** is handled naturally by the accumulators
  (facts are keyed, not sequenced).

### 4.6 Backpressure

Volumes are small at lap cadence, but `car_data`/`location` (if ever subscribed) are
~3-4 Hz times 20 cars. Rules: bounded queues per stream; keep-latest streams coalesce
under pressure (dropping stale samples is correct by definition); append-only streams
are never dropped (they are low-rate). v1 does not subscribe to `car_data`/`location`
at all: no model consumes them (pit-wall doc R3 keeps broadcast-tier out of features),
and the dashboard can add them later behind the same ingestor.

### 4.7 Cadence summary

One `lap_state` per our-driver lap (~90 s), assembled from sub-lap streams. The
orchestrator, guardrails, and LLM synthesis all fit comfortably inside that budget
(replay already proves the compute path at faster-than-real-time speeds).

---

## 5. Fan-out to surfaces

One producer, the existing consumers, no new transports invented:

- **Orchestrator loop (in-process):** the live manager feeds the same per-lap loop the
  replay feeds today. The CLI stays untouched; a separate additive entry point (for
  example an `f1-live` command or backend-managed session) hosts the live loop, never
  a modification of `run_simulation_cli.py`.
- **Backend (FastAPI):** the existing `/api/v1/strategy/simulate` SSE generator pattern
  gains a sibling live-session mode: same `start`/`lap`/`error`/`summary` event
  vocabulary, sourced from the live emitter instead of the replay engine. The pit-wall
  doc's WebSocket relay (option O1) then serves the browser dashboard; live mode is
  simply a second payload source for the same relay.
- **Arcade TCP stream:** Arcade remains a desktop replay surface. In live mode it
  consumes lap-cadence `lap_state` plus decisions (its strategy snapshot already has
  exactly that shape); smooth 60 FPS car motion would need the `location` stream and is
  explicitly out of scope for v1 (Arcade renders live as a timing-tower-style update,
  not an animation).
- **Pit-wall dashboard (web):** per the pit-wall doc, O3 rendering with the O1 relay;
  the live `session_meta.live` staleness block drives the data-age labels that doc
  requires (tier labeling: observed / derived / estimate).

The invariant for every surface: consume `lap_state` and its additive keys only, never
OpenF1 messages directly. The sim/live swap must stay a producer-side concern.

---

## 6. Testability without a live weekend

The old blocker ("live timing cannot be tested until a real GP") dissolves once the
transport is an interface:

### 6.1 The shadow-replay harness

OpenF1's REST archive serves the SAME rows the live feed emitted, with their original
timestamps. A rehearsal transport reads a recorded race's rows and replays them through
the ingestor in timestamp order at 1x or accelerated speed. This exercises every
mechanic that matters (assembly, lap closure, grace windows, staleness, backfill after
a simulated disconnect) with zero live dependency. It also powers the cross-producer
parity test of section 2.3 against the FastF1 replay of the same race.

What it cannot test: the real push transport's connection behavior and the true feed
latencies. Those get a cheap live rehearsal on a PRACTICE session (FP1 is a live
session too) before ever running on a race.

---

## 7. The 2026 gate

Live inference is pointless on drifting models. Every model in section 3.5 is trained
on the 2022-2025 regulation; the 2026 cars break the learned relationships (the concept
drift analysis, retraining order, and weekend data strategy live in
`documents/audits/AUDIT_2026_REG_CONCEPT_DRIFT.md`, epic #189, and are NOT duplicated
here). The dependency is one-directional and hard:

- **Gate:** the live consumer may run its mechanics (sections 4-6) against any session
  at any time, but agent recommendations on 2026 races are enabled only after the #189
  retraining pipeline (weekend FP/Qualy/Sprint data, pitlab Studio as the retrain
  surface) has produced validated 2026 models. This restates the roadmap rule already
  fixed in the ecosystem plan: Phase 4 (2026 adaptation) before Phase 5 (live), and the
  same rule protects box-bot downstream (never publish numbers from drifting models).
- **Non-gated work:** everything in this document except "recommendations on a 2026
  race" is regulation-independent plumbing and can be built and rehearsed on 2024-2025
  archived sessions via the shadow-replay harness.

---

## 8. Repo placement

**In-core** (this repo), for example under `src/live/` as a sibling of
`src/simulation/`. By the ecosystem integration rule
(`documents/research/ECOSYSTEM_REPO_INTEGRATION.md`): submodules are for runtime-coupled
components with their own release life; independent repos are for downstream services
and standalone artifacts; and anything that PRODUCES the `lap_state` contract is core
runtime by definition. The live consumer is exactly that, and its consumers (agents,
backend, Arcade) are all in-core. Downstream ecosystem pieces (box-bot) consume the
backend's public stream, never this module directly, which keeps the dependency
direction invariant intact.

---

## 9. Phased roadmap

| Phase | Deliverable | Gate |
|---|---|---|
| **L0: contract fixtures** | Golden `lap_state` fixtures + boundary/parity tests (with Testing epic #181/#182) | None; first, everything else asserts against it |
| **L1: shadow-replay harness** | Ingestor interface + recorded-REST transport + assembler + emitter; parity vs FastF1 replay on validated GPs (Hungary, Qatar) | L0 |
| **L2: live manager + polling transport** | `LiveRaceStateManager` complete over polling; orchestrator loop runs end-to-end on an archived session at 1x | L1 |
| **L3: push transport + resilience** | Real-time subscription, reconnect + backfill, staleness metadata; rehearsal on a live FP session | L2; OpenF1 account (Q1) |
| **L4: fan-out wiring** | Backend live-session SSE/WS mode + dashboard staleness labels + Arcade lap-cadence consumption | L2 (parallel with L3) |
| **L5: live race with recommendations** | Full stack on a race weekend | L3 + L4 + the #189 2026-model gate (or a 2025-regulation race if run before the season ends) |

---

## 10. Risks

- **OpenF1 access terms change** (pricing, transport, rate limits). Mitigation: the
  transport interface isolates it; polling REST is the floor; re-verify terms at L3.
- **Feed lag clusters at exactly the wrong moment** (stint row missing at lap close
  after a pit stop, the highest-value decision lap). Mitigation: grace window + the
  derive-from-pit-events fallback for tyre age; guardrails already suppress
  low-confidence calls.
- **Provisional-vs-corrected timing noise** degrades model inputs in ways the parity
  test tolerances must quantify, not hand-wave. Mitigation: measure on the shadow
  harness (same race, both producers) before trusting live numbers.
- **Semantic overreach in comms**: presenting live output as pit-wall-grade while our
  driver has no private telemetry. Mitigation: section 3.2's honest framing propagates
  to every surface label and to any publication text.
- **Scope creep toward sub-lap streaming** (car_data/location, 10 Hz dashboards).
  Mitigation: v1 is lap-cadence by design; sub-lap belongs to the dashboard track and
  only through the same ingestor, later.

---

## 11. Open questions

1. **OpenF1 account**: should the project take on the paid/registered real-time tier when
   L3 arrives, or should L3 target polling-only for the first live season?
2. **Live entry point**: new `f1-live` CLI command, or backend-managed live sessions
   only (started via API, watched from the dashboard)? The untouchable rule forbids
   growing `run_simulation_cli.py`; a decision is needed on where the live loop lives
   operationally.
3. **Which driver is "ours" live**: fixed per session at startup (like replay), or
   switchable mid-session? Switchable is cheap at this tier (all cars are symmetric
   live) but changes the TCN warm state; recommend fixed per session for v1.
4. **Radio in live v1**: include the delayed `team_radio` path from day one, or ship
   live v1 without the radio agent and add it in a fast-follow once latency is measured?
5. **Rehearsal target**: which archived GP should be the canonical shadow-replay
   fixture (recommend one already validated in the TFG, Hungary or Qatar, so agent
   outputs have a known-good baseline)?
6. **speed_fl**: accept it as permanently null in live mode (recommended), or invest in
   estimating it from the broadcast channel (not recommended; no consumer needs it)?
