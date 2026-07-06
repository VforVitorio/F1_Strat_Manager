# Pit-Wall Realism and the Telemetry Surface (design)

**Status: research design, forward plan. Plan only, no code, no commitments.**
**Date: 2026-07-06.**

This document covers two tightly coupled forward-looking topics:

1. **Topic 1, observability realism**: a precise model of what a real Formula 1 pit wall
   sees during a race, for our driver versus each rival, reconciled against what the code
   actually exposes today, with concrete additive refinements. It REFINES the
   observability section of `documents/research/RIVAL_AGENT_DESIGN.md` (its section 4);
   it does not duplicate that design.
2. **Topic 2, the telemetry surface**: the design of a real-telemetry pit-wall dashboard
   showing all drivers as a real wall would (our car full telemetry, rivals only the
   observable tier), synchronized with the race replay, plus the architecture decision
   Victor is weighing: what to migrate out of the native Arcade surface, what to keep,
   and what to kill.

Hard constraints honored throughout: design only, no code; `scripts/run_simulation_cli.py`,
`src/agents/` internals, and `notebooks/**` are untouchable (everything proposed is
additive); the backend stays FastAPI; any web recommendation reuses the frontend-migration
stack (epic #25); LLM provider is OpenAI or LM Studio, never Anthropic.

Documents this builds on (read, not re-planned): `documents/research/RIVAL_AGENT_DESIGN.md`,
`documents/audits/AUDIT_P3_ARCADE.md`, `documents/audits/AUDIT_P2_LOADING.md`,
`documents/audits/AUDIT_P2B_CORE_COMPUTE.md`, `documents/audits/AUDIT_P5_DATA_ENGINEERING.md`
(via the Rival design's references), and the frontend migration plan in the submodule
(`src/telemetry/docs/migration/MIGRATION_PLAN.md`, epic #25).

---

## 1. Framing

F1 StratLab's architectural thesis is data fidelity: agents must never see data a real
pit wall could not see (`src/simulation/race_state_manager.py:11-13` states this in the
module docstring). That thesis has two halves that have never been written down as one
model:

- **The input half**: which rival signals are legitimately observable, which are
  derivable, and which are genuinely hidden and must be modeled as uncertainty. This
  decides the Rival Agent's feature space (TFM) and the honesty of the simulation.
- **The output half**: what the product SHOWS. A pit-wall dashboard that renders hidden
  rival data as if it were real breaks the same thesis on screen that the boundary
  protects in the agents.

The correction from Victor (2026-07-06) that this document bakes in everywhere: a rival's
**tyre compound and tyre age (laps on the current set) ARE known to a real pit wall**.
They are visible on TV, carried in the FIA timing feed, and trivially derivable from
observed pit-in/pit-out laps. What is genuinely hidden for a rival is: raw car telemetry
traces at professional fidelity (steering, brake pressures, ERS deployment), true tyre
degradation / remaining grip, fuel load, engine and energy deployment modes, and the
team's strategic intent. Verified against the code: `get_rival_states` already emits
`compound`, `tyre_life`, `stint`, and `interval_to_driver_s` per rival
(`src/simulation/race_state_manager.py:265-279`), so the codebase already agrees with
the correction.

---

## 2. Topic 1: the pit-wall observability model

### 2.1 What a real pit wall actually has (feed inventory)

A real strategy engineer's desk aggregates these feeds during a race. Each row is a
distinct physical channel; the taxonomy in 2.2 is built from them.

| # | Feed | Contents | Covers |
|---|---|---|---|
| 1 | **Own-car telemetry link** | Hundreds of channels at high rate: throttle, individual brake pressures, steering, ERS deployment and state of charge, fuel flow and load model, tyre temperatures and pressures, engine modes, damper/ride data | Own car only |
| 2 | **FIA/FOM live timing** | Position, lap times, sector times and mini-sector segments, the four speed measurement points (I1, I2, FL trap, speed trap), pit-in/pit-out events, gaps and intervals, **per-driver stint compound and tyre age**, track status | All cars |
| 3 | **Broadcast car data channel** | Coarse per-car telemetry at roughly 3-4 Hz: speed, RPM, gear, throttle percentage, brake on/off, DRS state. This channel is distributed inside the same live timing transport; it is exactly where FastF1's car telemetry and OpenF1 `/v1/car_data` come from | All cars, reduced fidelity |
| 4 | **GPS positioning** | Continuous car positions on the track map; the source of the intra-lap gap evolution tools every wall runs | All cars |
| 5 | **TV world feed + spotters** | Visual confirmation: tyre compound color bands, pit crews appearing in the lane ("they're boxing"), damage, driver behavior | All cars, curated/laggy |
| 6 | **Weather service** | Air/track temperature, wind, humidity, rain radar | Shared |
| 7 | **Race control messages** | Flags, SC/VSC deployment, investigations, penalties, track limits deletions | Shared, official |
| 8 | **Team radio** | Own car: full duplex. Rivals: monitorable (rival radio is carried on the broadcast/scanner feeds; teams staff people to listen) but curated, delayed, and possibly deceptive | Own full; rivals partial |
| 9 | **Tyre allocation sheets** | Pirelli/FIA publish per-driver allocations and used sets across the weekend; combined with observed stints this yields each rival's remaining sets per compound | All cars, derivable |

Two consequences worth stating plainly, because they sharpen the usual "rivals are a
black box" intuition in both directions:

- **More is public than the strict timing-screen framing suggests.** Sector times,
  four speed measurements, stint compound and tyre age, and even coarse
  throttle/speed/gear traces of every car are in the distributed feeds (rows 2-4). The
  project's own data proves this: FastF1's laps and telemetry for ALL drivers, on disk
  today under `data/raw/<year>/<gp>/`, are reconstructions of those public feeds.
- **Less is knowable than a replay dataset suggests.** What no feed ever carries for a
  rival: fuel load, ERS/engine modes, brake and steering traces at professional rate,
  measured tyre wear, the pit wall's plan. FastF1 columns that look privileged (for
  example post-hoc corrected times) must be treated per the replay-vs-live audit already
  specified in the Rival design (its section 4.3).

### 2.2 The three-tier taxonomy, per entity

The taxonomy separates OUR DRIVER (full pipe, row 1 plus everything shared) from EACH
RIVAL (rows 2-9 only). Tier definitions:

- **(a) Directly observed**: appears in a feed as a value; use as-is.
- **(b) Derivable from observation**: computed from observed events plus public priors;
  legal to use, but must be COMPUTED from observations, never read from privileged
  columns (the "derive, do not read" discipline of `RIVAL_AGENT_DESIGN.md` section 4.1).
- **(c) Hidden**: never in any feed; modeled only as uncertainty, inferred only through
  its observable footprint.

**Our driver:**

| Tier | Signals |
|---|---|
| (a) Observed | Everything in `get_driver_state` (`race_state_manager.py:188-217`): lap/sector times, position, gap, compound, tyre life, stint, fresh-tyre flag, all four speed points, track status, pit in/out flags; plus (in the real car, beyond the sim) full telemetry channels |
| (b) Derivable | Fuel-corrected pace, degradation estimate (own models over own data: the TCN + MC Dropout stack), projected pit loss |
| (c) Hidden | Even for the own car, TRUE remaining grip is an estimate, not a measurement; the sim's `fuel_load` (`race_state_manager.py:212`) is itself a linear model, honestly labeled as an estimate |

**Each rival:**

| Tier | Signals | Notes |
|---|---|---|
| (a) Observed | Position, lap time, **sector times**, the four speed measurements, pit-in/pit-out events, gaps and intervals (lap-boundary and intra-lap via GPS/intervals), **current compound**, **tyre age / stint number**, track status, race control messages, DRS-window occupancy, coarse broadcast car data (speed/RPM/gear/throttle/brake-flag/DRS at 3-4 Hz) | Compound and tyre age are observed per Victor's correction; the FIA feed carries them directly (OpenF1 `/v1/stints` is that feed's public mirror). Broadcast car data is a REAL but REDUCED-FIDELITY channel; see the nuance box below |
| (b) Derivable | Tyre age recomputed from observed out-laps (the live-robust way to obtain what (a) also carries), stops so far, remaining sets per compound (allocation minus observed usage), new-vs-scuffed at fitting (allocation tracking, noisy), circuit pit loss and the undercut window, pace trend / degradation slope within the stint (the observable footprint of latent wear), free-air pit window and pit-exit traffic, DRS-train membership, "in the window of the car ahead" geometry | These are exactly feature families F1/F2/F3 of `RIVAL_AGENT_DESIGN.md` section 5 |
| (c) Hidden | Raw professional telemetry (steering, brake pressures, ERS deployment), fuel load, engine/energy modes, TRUE degradation / remaining grip, whether the fitted set was new or scuffed (beyond the noisy derivation), team strategic intent, pit calls not yet visible in the lane, driver instructions not broadcast | Never features; only priors and uncertainty. The Rival Agent's H1-H4 heads PREDICT the observable consequences of this tier |

**The broadcast car-data nuance (a deliberate refinement of the hidden list).** Victor's
correction lists "raw car telemetry traces (throttle/brake/steering/ERS)" as hidden. That
is right at professional fidelity, and this document keeps steering/ERS/brake-pressure
firmly in tier (c). But the coarse broadcast channel (row 3 of 2.1) does expose rival
speed, gear, DRS, throttle percentage and a brake on/off flag at low rate; that is
literally where the repo's own rival `FrameData` comes from (`src/arcade/data.py:50-72`
is built from FastF1 car telemetry for all 20 drivers). The honest resolution is a
sub-tier, called **broadcast tier** here: real, public, low-rate, low-trust. Design
decisions that follow from it:

- The **Rival Agent v1 does NOT consume broadcast-tier traces** as features. Reasons:
  the pit-timing signal lives in timing-tier features (the Rival design's F1-F5); the
  channel is noisy and gappy; and staying at the timing tier keeps the feature set
  robust to feed differences between replay and live. This is a scoping choice, not a
  claim that the data is secret.
- The **dashboard MAY render broadcast-tier rival traces**, but only behind an explicit
  "broadcast data" label so the surface never implies pit-wall-grade rival telemetry
  (section 3.4).
- The oracle ablation of the Rival design (its section 8.4) gains one optional arm:
  timing tier vs timing+broadcast tier vs privileged oracle. If broadcast-tier features
  buy nothing for pit prediction (expected), that measured negative cleanly justifies
  the v1 scoping. This is refinement R6 below.

### 2.3 Gap analysis: `RaceStateManager` versus the real wall

`get_rival_states` (`src/simulation/race_state_manager.py:219-281`) emits, per rival per
lap: `driver`, `team`, `position`, `lap_time_s`, `compound`, `tyre_life`, `stint`,
`speed_st`, `gap_to_leader_s`, `interval_to_driver_s`, `is_pitting`. Verdict per field:

| Field | Tier | Verdict |
|---|---|---|
| `position`, `lap_time_s` | (a) | Accurate |
| `compound`, `tyre_life`, `stint` | (a)/(b) | Accurate, and consistent with Victor's correction; the live-parity discipline is to also DERIVE tyre age from observed pit events so the same logic works when the column does not exist (already specified in `RIVAL_AGENT_DESIGN.md` 4.1) |
| `speed_st` | (a) | Accurate but incomplete (see "too stingy") |
| `gap_to_leader_s`, `interval_to_driver_s` | (a) | Accurate; end-of-lap resolution only |
| `is_pitting` (from `PitInTime`, `:277`) | (a) | Accurate; pit entry is publicly visible the moment it happens |

**Direction 1: does the boundary leak hidden data? No, at the `lap_state` level.**
The rival dict carries none of the driver-only fields: sector times, the I1/I2/FL speed
readings, `fuel_load`, and the in/out-lap pair live only in `get_driver_state`
(`race_state_manager.py:194-196, 207-212, 215-216`). Nothing in the rivals list is tier
(c). The single-driver boundary holds where it matters: the agents' input space.

One SURFACE-level qualification, not a contract leak: in Head-to-Head mode the Arcade
telemetry window renders the rival's throttle/brake/speed traces
(`src/arcade/app.py:460-473` broadcasts `telemetry.rival` built by `_frame_to_telemetry`,
`app.py:65-96`; rendered by the 2x2 grid in `src/arcade/dashboard/telemetry_panel.py:1-27`).
Because FastF1 car telemetry IS the broadcast channel, this is broadcast-tier, not
privileged, so it is defensible; but it is UNLABELED today, and the module docstring's
own framing ("timing-screen only", `race_state_manager.py:8-9`) would not predict it.
The fix is labeling and tiering on screen (section 3.4), not deletion of data.

**Direction 2: is the boundary too stingy? Yes, mildly, in five places.** A real wall
sees more than `get_rival_states` grants:

| Withheld today | Reality | Evidence |
|---|---|---|
| Rival **sector times** | FIA timing shows S1/S2/S3 (and mini-sectors) for every car; the docstring explicitly excludes them ("No sector times", `race_state_manager.py:224-225`) | FastF1 laps carry `Sector1/2/3Time` for all drivers; on disk in `data/raw/<year>/<gp>/laps.parquet` |
| Rival **speed measurements I1/I2/FL** | All four points are on the timing screen; only `speed_st` is emitted (`:274`) | Same parquet, `SpeedI1/I2/FL` columns |
| Rival **out-lap flag** | Pit exit is as visible as pit entry; only `is_pitting` (pit-in) exists (`:277`) | `PitOutTime` in the same parquet |
| **Sub-lap gap evolution** | Real gap tools update continuously (GPS); the sim's gaps are end-of-lap snapshots | `intervals.parquet` (~4 s resolution) is downloaded for every race and consumed by nothing at runtime (P5 audit finding F-10, restated in `RIVAL_AGENT_DESIGN.md` 1.3) |
| **Race control messages for the field** | RCM is a shared official feed; today RCM context reaches only our driver's radio agent path | `RCMContextResolver` work (project memory); no rival-side RCM in `lap_state` |

**Overall verdict.** The boundary is directionally correct and conservative: it never
leaks tier (c), and its omissions are all tier (a) data a real wall has. Being too
stingy is the safe failure mode for a defended thesis, but it costs the Rival Agent real
signal (sector-level pace trends are the sharpest public footprint of degradation) and
it costs the future dashboard its timing tower. All refinements are additive.

### 2.4 Refinements (all additive, none touch untouchables)

**R1: additive rival timing fields.** Extend the per-rival dict emitted by
`get_rival_states` with `sector1_s`, `sector2_s`, `sector3_s`, `speed_i1`, `speed_i2`,
`speed_fl`, `is_out_lap`. The `lap_state` contract tolerates additive keys (verified by
the P5 audit, cited at `RIVAL_AGENT_DESIGN.md:75-78`), and every existing consumer reads
keys defensively. Replay-vs-live parity: each added field maps to a live feed value
(sector times and speeds are in the live timing stream; OpenF1 mirrors them), so the
future live adapter can populate them identically.

**R2: the gap-history provider.** Unchanged adoption of the P5 audit's Phase 4 item 13
(already committed to in `RIVAL_AGENT_DESIGN.md` section 7.1, item 2): a runtime
provider reading `intervals.parquet` behind an additive `lap_state` key with per-rival
gap traces. This document adds a second consumer to the same provider: the dashboard's
gap/interval chart (section 3.1, window 5). One provider, two consumers (Rival Agent
features F2; pit-wall gap chart), which strengthens the case for building it early.

**R3: the broadcast-tier decision.** Codify the sub-tier from 2.2: broadcast car data
for rivals is real and public but excluded from Rival Agent v1 features and rendered
only behind a label. If it is ever wanted as a feature family, the ingestion path is
OpenF1 `/v1/car_data` (would be a new Tier beyond the P5 Tier 1 endpoints; not planned).

**R4: tier tags as metadata.** The Rival design already commits to tagging every feature
with its observability grade in the dataset card (its section 5, closing paragraph).
Extend the tag vocabulary to the four grades defined here: `observed-timing`,
`observed-broadcast`, `derived`, `hidden-modeled`. The same tags drive UI labeling in
Topic 2 (section 3.4), so the dataset card and the screen tell the same story.

**R5: boundary parity tests.** Two cheap additive tests that make the boundary a
verified contract instead of a docstring promise: (i) a golden `lap_state` fixture
asserting the rivals list contains ONLY whitelisted keys (fails if anyone accidentally
enriches rivals with privileged columns); (ii) a leak test asserting the driver-only
fields (`fuel_load`, per-sensor speeds beyond the whitelist, `is_in_lap`) never appear
in any rival dict. When the live adapter lands, the same fixture asserts replay and live
produce identical shapes. Coordinate with the Testing epic (#181/#182 fixtures).

**R6: the explicit refinements to `RIVAL_AGENT_DESIGN.md` section 4** (this is the
"refine, do not duplicate" deliverable):

1. **Split its "Directly observable" grade** into `observed-timing` and
   `observed-broadcast` per 2.2. Its current ladder (section 4.1) treats "timing
   screen" as one grade; the broadcast car-data channel needs its own row with its own
   noise model.
2. **Add sector-time features to family F3** (pace signals): per-sector pace slope
   within the stint is a sharper degradation footprint than whole-lap slope (a car
   losing time only in the high-load sector is degrading, a car losing time everywhere
   is managing). Enabled by R1.
3. **Compound noise model source**: section 4.1 proposes quantifying compound
   observation noise from the FastF1-vs-OpenF1 stint agreement rate. Keep that, and note
   the FIA feed carries stints directly, so live compound knowledge is feed-grade, not
   TV-detection-grade; the noise model's realistic role shrinks to feed lag (a stint row
   appearing a few seconds after the out-lap), which the staleness features already
   cover.
4. **One optional oracle-ablation arm** (section 8.4): timing tier vs timing+broadcast
   tier vs privileged oracle, to price the broadcast channel and justify the v1
   exclusion with a measurement.
5. **No change** to its hidden list beyond the sub-tier split: fuel, modes, true
   degradation, intent stay tier (c), exactly as it and Victor's correction state.

### 2.5 How the existing Head-to-Head mode approximates this, and its improvements

What exists (verified): the CLI `--rival CODE` flag (`scripts/run_simulation_cli.py:2387`,
rival lookup at `:1867`, rendered cell at `:2029`) and the menu's H2H runner
(`scripts/cli/runner.py:120` `run_h2h`, passing `--rival` at `:71`) track one designated
rival per run. The rival's on-screen state comes from `lap_state["rivals"]`, so the CLI
H2H is ALREADY tier-correct: it shows the timing tier and nothing more. The Arcade H2H
adds the rival's broadcast-tier traces in the telemetry window (2.3 above) and full
per-frame positions for all 20 cars (GPS tier, `app.py:444-455`), which is also
tier-consistent once labeled.

Improvements this design proposes:

- **Tier labels on every rival element** (section 3.4): the timing tier renders plainly;
  broadcast-tier traces carry a "broadcast data" chip; anything model-inferred (future
  Rival Agent output) renders as probability, never as fact.
- **Enrich the rival panel with the derivable tier**: estimated stops so far, remaining
  sets, in-window flags; all computable today from `lap_state` plus
  `tire_compounds_by_race.json` priors, no new data needed.
- **The step from observation to anticipation stays the TFM's job**: the H2H mode is
  display-level today (`RIVAL_AGENT_DESIGN.md` 1.3 makes this exact point); the
  `RivalContext` output (its section 6.5) is the future predictive column of both the
  CLI H2H table and the dashboard's rival intent panel (section 3.1, window 11).

### 2.6 Replay-vs-live parity

For every field the boundary exposes, the live-mode source must exist. Mapping (for the
future OpenF1 WebSocket adapter the `lap_state` contract anticipates):

| Replay field (FastF1 column) | Live source |
|---|---|
| Position, lap/sector times, speeds | Live timing stream (OpenF1 laps/sessions endpoints mirror it) |
| Compound, stint, tyre age | Stint feed (`/v1/stints`), or derived from observed pit events (R2 discipline) |
| Pit in/out | `/v1/pit` and the timing feed's pit flags |
| Gaps, intervals, sub-lap traces | `/v1/intervals` (already downloaded per race, unused at runtime today) |
| Track status, RCM | Race control feed (`/v1/race_control`) |
| Weather | `/v1/weather`; note the replay weather join is by fractional lap index (P5 finding F-14), documented as replay-only behavior |

The parity rule for all new code (provider, dashboard, Rival Agent feature builder):
consume `lap_state` and the additive keys ONLY, never FastF1 frames directly, so the
sim/live swap stays a data-layer change. This is the same rule the Rival design commits
to (its section 7.4, live-feed forward compatibility).

---

## 3. Topic 2: the pit-wall telemetry surface and the Arcade future

### 3.1 The pit-wall windows to reproduce

The set of "windows" a real wall runs, mapped to what exists in the repo today. Tier
discipline (from Topic 1) applies per window: our car renders full telemetry; rivals
render tiers (a)+(b); broadcast tier behind a label; tier (c) only ever as model output.

| # | Window | Contents | Exists today | Anchor |
|---|---|---|---|---|
| 1 | **Timing tower** | All 20 cars: position, gap, interval, last lap, S1/S2/S3, pit status, compound + tyre age | Partial: arcade leaderboard overlay (position/gap tier only), CLI table; no sectors anywhere (needs R1) | `src/arcade/overlays.py` leaderboard; `run_simulation_cli.py:2029` rival cell |
| 2 | **Track map with positions** | 2D circuit + live car dots, DRS zones, flag sectors | YES, native: the pyglet replay window | `src/arcade/track.py`, `src/arcade/app.py:490-499` |
| 3 | **Own-car telemetry traces** | Speed/throttle/brake/gear/DRS vs lap distance, plus delta to a reference | Partial: 2x2 pyqtgraph grid, main + one rival only | `src/arcade/dashboard/telemetry_panel.py:1-27` |
| 4 | **Tyre/stint board** | Per driver: stint history, compound sequence, ages, estimated remaining sets | Partial: `tire_chart.py` for our driver; rivals' stints reconstructable from timing data | `src/arcade/dashboard/tire_chart.py:163-218` |
| 5 | **Gap/interval evolution chart** ("race trace") | Cumulative gap lines per driver over laps; undercut windows visible as converging lines | Missing at runtime (the R2 provider is the source); Streamlit has a post-race version | `src/telemetry/frontend/components/race_analysis/gap_charts.py` |
| 6 | **Pit window / pit-loss board** | Per rival: circuit pit loss, in-window flags, projected exit traffic | Missing; feature family F2 of the Rival design computes exactly this | `RIVAL_AGENT_DESIGN.md` section 5, F2 |
| 7 | **Weather panel** | Air/track temp, wind, rain | Exists but renders hardcoded constants today (P3 finding A2); real per-lap weather is P3 Phase B.1 | `src/arcade/app.py:648-656`, `AUDIT_P3_ARCADE.md` A2 |
| 8 | **SC/flag status board** | Track status, SC/VSC/red spans, race control messages | Partial: live pill exists; timeline flag spans are dead (P3 A3); RCM feed not surfaced | `src/arcade/data.py:312`, `AUDIT_P3_ARCADE.md` A3 |
| 9 | **Strategy board (agents)** | The six agent cards, orchestrator decision, scenario scores, reasoning | YES: the Qt strategy dashboard window | `src/arcade/dashboard/window.py` |
| 10 | **Radio feed** | Own radio transcripts + NLP verdicts; rival radio (future radiogate corpus) | Partial: radio agent alerts in the cards; no dedicated feed panel | `src/arcade/dashboard/agent_formatters.py` (radio) |
| 11 | **Rival intent panel** | Per tracked rival: p_pit windows, predicted stop lap quantiles, threat/cover probabilities | Future: renders `RivalContext` (TFM M4+) | `RIVAL_AGENT_DESIGN.md` section 6.5 |

### 3.2 The data plane: how the surface consumes data

**What exists today, verified:**

- **The Arcade TCP broadcast**: `TelemetryStreamServer` (`src/arcade/stream.py:27-105`),
  newline-delimited JSON on 127.0.0.1:9998, throttled to ~10 Hz
  (`STREAM_BROADCAST_EVERY_N_FRAMES = 6` at 60 FPS, `src/arcade/config.py:168-174`).
  Payload (`src/arcade/app.py:426-488`): an `arcade` snapshot (per-driver
  lap/dist/speed/compound/tyre_life for ALL 20 cars, full traces for main + rival), a
  `strategy` snapshot (latest decision + 30-entry history tail), and a `playback` block
  (speed, paused, frame_index, total_frames), which is the replay's master clock.
- **The backend SSE simulation stream**:
  `POST /api/v1/strategy/simulate` (`src/telemetry/backend/api/v1/endpoints/strategy.py:898-925`)
  streams `start`/`lap`/`error`/`summary` events from a generator in
  `src/telemetry/backend/services/simulation/simulator.py`, whose own docstring says it
  exists "so the backend can stream ... to any SSE consumer (curl, Arcade, future
  dashboards)". Nothing consumes it yet (the Streamlit UI never calls it). The two
  threads Victor is weaving together were already anticipated to converge here.
- **The migration stack** (`src/telemetry/docs/migration/MIGRATION_PLAN.md`, epic #25):
  React 19 + Vite + TypeScript strict + Tailwind v4 mapped to `tokens.css` + TanStack
  Router/Query + Zustand + Apache ECharts as the single chart lib + a custom canvas/rAF
  engine for the Comparison replay + `eventsource-parser` for SSE + GSAP and
  react-three-fiber code-split for flagship moments. No `webapp/` exists yet; the plan
  is decision-grade but pre-implementation.

**The connection question Victor asked: does the TCP stream feed a web client directly?**
No. Browsers cannot open raw TCP sockets; a page can only speak HTTP(S), SSE, WebSocket,
or WebRTC. So "the new tool consumes the Arcade backend" has three viable shapes:

| Option | Shape | Assessment |
|---|---|---|
| **O1: FastAPI relay** | The backend runs a small TCP client to 127.0.0.1:9998 (exactly what `dashboard/stream_client.py` does today in Qt) and re-publishes each payload over a WebSocket (or SSE) endpoint; the SPA subscribes | Cheap, additive, zero change to Arcade; Arcade stays the master clock; the relay is a natural backend feature since the backend already runs on :8000 next to everything. **Recommended first step** |
| **O2: backend-native replay streamer** | The backend owns the replay loop itself (it already imports `RaceReplayEngine` and the orchestrator, `simulator.py:38-48`) and streams frames + decisions with no Arcade process at all | The end-state for a fully unified web app; needs a frame-rate stream or client-side animation (O3) to be smooth; the existing `/simulate` SSE is its lap-cadence seed |
| **O3: bulk prefetch + thin stream** | The SPA fetches the whole race's per-driver telemetry arrays once up front (the P2 F-05 numpy SoA cache is byte-for-byte the right payload) and animates CLIENT-SIDE at 60 FPS with rAF; the live stream carries only the clock, strategy decisions, and events | **The highest-performance design for replay**, and exactly the pattern the migration plan already commits to for the Comparison page ("custom canvas/rAF engine ... replaces the pre-baked 10fps Plotly frames"). Stream rate stops mattering for smoothness; 10 Hz is plenty for a clock |

**Recommendation**: O3 for the telemetry dashboard's own rendering, O1 for
synchronization while the native Arcade window is the replay master, O2 as the
convergence point at the parity gate (section 3.5). Transport: **WebSocket** for the
relay rather than SSE, because the moment the browser wants to seek/pause the replay
(section 3.6, Q2) a bidirectional channel is needed, and WS costs the same to build now.
SSE remains the right transport for the existing simulate/chat streams (one-directional
by nature, and the plan already carries `eventsource-parser`).

**A contract consequence**: the P3 audit explicitly deferred a schema for the TCP wire
("a Pydantic `BroadcastPayload` is worth it only if an external consumer appears",
`AUDIT_P3_ARCADE.md:124-127`). The relay IS that external consumer. Phase 1 therefore
includes the minimal versioned payload schema P3 said to defer, which is an extension of
P3's own trigger condition, not a contradiction of it.

### 3.3 Tooling assessment

**(a) A new window inside the existing PySide6 dashboard.** Rejected for new investment.
Every finding in the P3 audit's dashboard cluster (A6 full re-render at 10 Hz, A7 double
stream clients, A13 teardown gaps, A14 QSS no-op) is a cost of building product UI in
Qt; a 20-driver timing tower and multi-panel boards would multiply exactly that class of
work, with zero reuse for the frontend migration that is already priority #1. Qt/pyqtgraph
remains fine for what already exists; it is the wrong place to build MORE.

**(b) Grafana, assessed honestly.** Strengths: superb time-series panels over a real
data source, alerting, dashboards-as-config, WebSocket/live streaming support exists.
Weaknesses for THIS product: no synced track-map/replay primitive (a scrub-linked replay
clock across panels is foreign to its model); a timing tower and stint board are bespoke
data-grid widgets, which in Grafana means writing custom panel plugins IN REACT anyway,
so the "buy" option degenerates into building React components inside a heavier host;
styling is locked away from `tokens.css` (brand consistency is a stated migration goal);
and it adds an ops surface (server, provisioning, auth) to a local-first product whose
distribution story is `uv tool install` + Docker compose. Verdict: **reject for the pit
wall**. Legitimate niche if ever wanted: an internal ops/profiling dashboard (P2b stage
timings, LLM latency, cache hit rates), where Grafana's model fits and no brand or sync
constraints exist.

**(c) Custom web surface on the migration stack.** Recommended, and this is also
Victor's stated desire (the same stack as the telemetry-analysis surface). Evidence it
fits: ECharts handles streaming line charts and large series natively (the plan already
made it the single chart lib to kill the plotly.js bundle); the canvas/rAF replay engine
planned for Comparison is the same primitive windows 2 and 5 need; the design system
(`tokens.css`, dark ramp, the palette the Arcade theme already mirrors per the arcade UI
memory) applies 1:1; `eventsource-parser` and TanStack Query cover the data plane. The
pit-wall dashboard becomes one more route in `webapp/` (epic #25), not a new stack.

**(d) Full Arcade-to-web/Three.js migration.** Not now, converge later; see 3.5.

### 3.4 Victor's per-window split, assessed one by one

Victor's lean: keep Arcade's circuit window native; migrate the agent-cards window to
the web stack; kill the Arcade telemetry window because the new all-drivers dashboard
replaces it. Explicit opinions:

**KEEP the circuit/track-map window in Arcade: agree, short-to-mid term.** It is the
one window where native genuinely delivers today: the pyglet replay is shipped,
defended, and its known issues have cheap fixes already planned (P3 Phase C bakes the
track tessellation; Phase B puts real weather and flag spans on it). Porting it now
would be the most work for the least new capability. One honest qualifier: after
migration sprint S3 lands the canvas/rAF Comparison engine, a web track map becomes
nearly free (20 dots on a canvas polyline is trivial at 60 FPS), so "keep" should be
read as "keep until the parity gate", not "keep forever" (section 3.5).

**MIGRATE the agent-cards window to the web stack: agree strongly.** This is the window
with the highest mismatch between content and host. Its content is text, badges, small
charts and reasoning prose that change ONCE PER LAP, yet the Qt implementation re-renders
six cards, the orchestrator card, six syntax-highlighted QTextEdits and a full
tire-chart rebuild at 10 Hz (P3 finding A6, `dashboard/window.py:211-233`,
`reasoning_tabs.py:210-229`, `tire_chart.py:163-218`). In the web stack all of that is
idiomatic and cheap: collapsible/expandable panels, density toggles, decision-history
scrubbing, hover detail, proper typography, and the once-per-lap update cadence is just
a state change. Migrating the cards also retires an entire chain of Qt-specific debt by
construction rather than by fixing it: A6 (re-render), A7 (two stream clients), A13
(subprocess teardown), A14 (QSS opacity no-op) all dissolve when the Qt dashboard
process stops existing. The card content contract is already clean for the move: the
formatters are pure dict-in/string-out functions (`dashboard/agent_formatters.py`), and
the broadcast's `strategy` block is the same payload the relay will carry.

**KILL the Arcade telemetry window: agree.** It is the weakest window: two drivers
maximum by design (`telemetry_panel.py:1-27`), a fixed 2x2 grid, pyqtgraph, and its own
independent TCP client (`telemetry_window.py:53-57`, the second half of P3 finding A7).
The new all-drivers dashboard strictly supersedes it: same four charts for our car, plus
the timing tower, gap chart, stint board and rival tiers it could never grow into.
Killing it (rather than migrating it 1:1) is right because its 2-driver framing is an
artifact of the H2H mode, not a pit-wall concept; the replacement is designed from the
Topic 1 taxonomy instead.

**Is the resulting hybrid coherent, or a pain?** It is coherent, and, importantly, it is
the SAME topology the system already runs: today the pyglet window broadcasts and a
separate process (Qt) follows the stream as a one-directional consumer. The proposal
replaces the Qt follower with a browser follower behind a backend relay; the sync model
(Arcade is the master clock, followers render `playback` + `lap`) is unchanged and
proven. The honest costs of the hybrid, stated plainly:

1. **Two things to launch** (arcade window + a browser tab). Mitigable: an
   `f1-arcade --pitwall` style flag that ensures the backend is up and opens the
   browser; and note the process count does not actually grow (today: pyglet + Qt
   subprocess; proposed: pyglet + backend + browser, and the backend is typically
   already running for the web surfaces).
2. **Playback control lives only in Arcade** unless a control channel is added
   (browser -> backend -> a small command path into the replay). Recommended posture:
   pit-wall v1 is a read-only follower (exactly what the Qt dashboard is today); the
   control channel is a v2 decision (open question Q2).
3. **The TCP wire becomes load-bearing** across a process AND a stack boundary, so the
   minimal versioned schema from 3.2 stops being optional.

**Is it worth keeping Arcade at all once cards + telemetry leave?** Yes, for one
concrete reason and one strategic one. Concrete: the circuit window is a working,
defended, offline-capable demo surface with zero web dependencies; it is the thing that
runs on a podium laptop with no backend. Strategic: the frontend migration (7-9 weeks
solo, five sprints, not yet started) should not absorb a track-map port before its
flagship pages exist; sequencing the circuit port behind the parity gate keeps epic #25
focused. But the end-state question deserves a straight answer:

**"One unified web app" vs "Arcade keeps the circuit + web app for the rest": the
recommendation.** Adopt Victor's split NOW; schedule the unification DECISION for the
parity gate, with the expectation that unification wins. Reasoning: the split is the
correct migration path, not the final architecture. Once (i) migration S3 has shipped
the canvas/rAF engine, (ii) the pit-wall route is live with the timing tower and gap
chart, and (iii) the relay/O2 streamer is proven, the remaining delta for "one surface"
is a canvas track map plus replay controls in the browser, which at that point is a
sprint, not an epic. If the web replay then matches native feel (60 FPS, instant seek,
20 cars), fold the circuit view in, and retire Arcade to legacy/demo status (kept
runnable, no longer developed; the same posture the repo already applies to
`legacy/**`). If it does not match (the only realistic risk is input latency and
raw-canvas feel on low-end hardware), the split remains permanently defensible: pyglet
for the replay, web for everything informational. Either outcome is reached without ever
doing a big-bang port, and no work done under the split is thrown away: the relay, the
schema, the tower, the boards and the cards all carry over unchanged.

### 3.5 The pit-wall dashboard design (v1, web)

- **Where**: a new route in `webapp/` (the epic #25 SPA), styled by the same
  `tokens.css`-mapped Tailwind theme; it is a sibling of Dashboard/Strategy/Comparison,
  not a separate app.
- **Layout** (all panels collapsible, per Victor's ask; presets like "Strategist" /
  "Race engineer" / "Broadcast" select which panels are open):
  - Left rail: **timing tower** (window 1), full field, virtualized rows; compound +
    tyre-age chips per row (tier (a), per Victor's correction); pit flags.
  - Center: **gap/interval race trace** (window 5, fed by the R2 provider) stacked over
    the **SC/flag timeline** (window 8). The center is deliberately NOT a track map in
    v1: the native Arcade window plays that role side by side until the parity gate.
  - Right: **our-car telemetry strip** (window 3: speed/throttle/brake/gear/DRS vs lap
    distance, ECharts, imperative updates) over the **tyre/stint board** (window 4) and
    **weather** (window 7).
  - Bottom ribbon: the **migrated agent cards + orchestrator decision** (window 9),
    expandable to full reasoning; **radio/RCM feed** (window 10) as a ticker.
  - Future: **rival intent panel** (window 11) rendering `RivalContext` when the TFM
    lands; per-rival broadcast-tier trace popovers behind the "broadcast data" label.
- **Tier discipline on screen** (the Topic 1 payoff): every rival element carries its
  tier tag from R4. Timing tier renders plainly; derived values render with their
  staleness (for example tyre age shows "counted from pit observation" on hover);
  broadcast-tier traces are labeled; hidden-tier quantities NEVER render as data, only
  as model outputs with probability formatting. This turns the thesis's data-fidelity
  claim into a visible product feature, and it is cheap because the tags are metadata
  from the taxonomy, not new computation.
- **Rendering discipline**: chart data flows imperatively into ECharts instances and the
  rAF loop (refs), never through per-frame React state; React renders panel chrome and
  once-per-lap content. This is the web equivalent of the lesson P3's A6 teaches about
  the Qt dashboard, applied preemptively.
- **Data**: bulk prefetch of per-driver arrays (O3; the P2 F-05 SoA cache is the ideal
  payload, and the bulk endpoint can ship even before F-05 by converting the current
  pickle once server-side), plus the WS relay for clock/strategy/events (O1), plus the
  existing REST endpoints for static context (sessions, drivers, stints).

### 3.6 Performance notes (why this will be fast)

- The stream is small: ~10 KB JSON at 10 Hz over localhost (`stream.py:84` documents the
  size class); a WS relay adds negligible latency.
- Client-side animation decouples smoothness from stream rate: with O3 bulk data, the
  browser interpolates positions at display refresh, which is how the migration plan
  already fixes the Comparison page's pre-baked 10 FPS Plotly frames.
- The heavy problems measured in the native surface do not transfer: the 8.0 s AoS
  unpickle (P2 F-05) is a server-side load cost the SoA cache fixes for BOTH surfaces;
  the per-frame re-tessellation (P3 A5) is a pyglet-immediate-mode issue with no web
  equivalent (a canvas polyline is drawn once per frame from typed arrays); 20 moving
  dots and a handful of line series are far below canvas/ECharts limits.
- The known web risk is self-inflicted re-rendering (the React-state-per-frame
  anti-pattern), addressed by the rendering discipline above.

---

## 4. Phased roadmap

Phases are sequenced to extend, never contradict, the existing programs: the frontend
migration epic (#25, submodule), the Arcade audit epic (#199, P3 phases), the P2/P2b
performance work, and the Rival Agent milestones (M0-M6 in `RIVAL_AGENT_DESIGN.md`).
Each phase decomposes into issue/PR-sized chunks per the repo's rhythm.

**Phase 0: the observability contract (Topic 1 landing).**
R1 additive rival fields in `get_rival_states`; R4 tier-tag vocabulary written into the
taxonomy table of this doc and the Rival design's feature table; R5 parity/leak tests
(coordinate fixtures with #181/#182); the R6 edits to `RIVAL_AGENT_DESIGN.md` section 4.
Small, self-contained, no UI. Exit: boundary verified by tests; Rival Agent feature
space and dashboard tiering share one source of truth.

**Phase 1: the data plane.**
The backend WS relay of the Arcade TCP stream (O1) + the minimal versioned broadcast
schema (the P3-deferred schema, now justified) + the bulk telemetry endpoint (O3
payload; ships against the current cache, upgrades transparently when P2 F-05 lands) +
the R2 gap provider behind its additive `lap_state` key. Extends #25's S0 backend
hardening; the relay lives in the submodule backend next to the existing SSE endpoints.
Exit: a browser page (even a dev scratch page) renders live lap/clock/strategy from a
running replay.

**Phase 2: pit-wall v1 in the SPA.**
The route with timing tower, gap chart, our-car strip, stint board, weather, SC/flag
timeline, tier labels. Sequenced INSIDE epic #25 after S3 (the canvas engine sprint) so
the chart/animation primitives exist; alternatively as its own sprint if #25 re-orders.
Exit: replay running natively + pit-wall following in the browser, all-driver tower live.

**Phase 3: agent cards migrate; Arcade telemetry window dies.**
Rebuild window 9 in the SPA bottom ribbon (formatters port as pure functions); delete
`telemetry_window.py` + `telemetry_panel.py`; retire the Qt dashboard subprocess
entirely (both windows gone); Arcade keeps only the pyglet circuit window and its TCP
broadcast. **Explicit re-scope of P3/#199 this implies, needing Victor's sign-off**:
P3 Phase D items D.1 (StreamBroker) and D.2 (lap-gated Qt rendering) are SKIPPED (their
problem hosts stop existing); D.3 (broadcast hygiene: dirty-flag serialization,
non-blocking sends) SURVIVES and gains importance (the relay depends on a healthy
broadcast); Phase B truth fixes (real weather B.1, flag spans B.2, provider flag B.3,
round cap B.4) SURVIVE (the circuit window remains); Phase C (track baking, measured)
SURVIVES; Phase A (engine dedup) is independent and unaffected. Exit: end state Victor
described: native circuit replay + one web surface with cards + pit-wall telemetry.

**Phase 4: rival intent and realism mode.**
The rival intent panel consuming `RivalContext` (TFM M4's integration output); the
oracle-vs-public demo toggle (renders what the wall could NOT see, clearly marked, as a
teaching/demo device); derivable-tier enrichments in the tower (remaining sets,
in-window flags). Depends on Rival Agent M2+ for real predictions; the panel skeleton
can land earlier with the heuristic baselines from the Rival design's section 8.1.

**Phase 5: the parity gate (the Arcade decision).**
Prototype the canvas track map in the SPA (reusing the Comparison engine) + replay
controls via the WS control channel; run the side-by-side comparison on the dev laptop
(subjective feel + frame timing). Decide: unify (fold the circuit view in, retire Arcade
to legacy/demo, O2 backend-native streamer becomes the only engine) or keep the split
permanently (documented rationale). Either way the decision is made with a working
prototype and zero sunk-cost pressure.

---

## 5. Risks and limitations

- **Two-surface UX friction in Phases 2-4** (arcade window + browser). Bounded (same
  follower topology as today) and mitigable with a launch flag; if it grates, that is
  itself evidence for unification at the gate.
- **The broadcast schema becomes a real contract.** Once the relay ships, arcade-side
  payload changes can break the web surface. The Phase 1 versioned schema plus a golden
  payload test (P3 Phase D already proposed the test) is the guard.
- **P2 F-05 dependency softness.** The bulk endpoint's ideal payload is the SoA cache,
  which has not landed. Mitigation is in Phase 1's design: serve from the current cache
  format first, swap the storage transparently later.
- **Replay-only assumptions.** Bulk prefetch (O3) is impossible in a future live mode;
  the dashboard must degrade to stream-only rendering (tower, gaps and boards work fine
  at lap/10 Hz cadence; only the smooth track map needs interpolation, which live GPS
  data supports anyway). Design the data hooks with both modes in mind from Phase 2.
- **Broadcast-tier honesty is load-bearing for the thesis narrative.** If the labels are
  dropped for aesthetic reasons, the surface silently overclaims what a wall sees; R4's
  tags must survive design polish.
- **Solo bandwidth.** Epic #25 is already 7-9 weeks solo; Phases 1-3 here add roughly
  2-3 sprints on top. The sequencing (pit-wall after S3) is chosen so the migration's
  own milestones are not delayed; if bandwidth forces a cut, Phase 0 and Phase 1 are the
  keep-at-all-costs core (they serve the TFM and every future surface regardless of UI
  decisions).
- **Era scoping.** Everything here inherits the 2022-2025 regulation scope; the 2026
  drift program (`AUDIT_2026_REG_CONCEPT_DRIFT.md`) governs retraining, and the
  dashboard renders whatever the models of the era emit.

---

## 6. Open questions for Victor

**Q1: P3/#199 re-scope.** Approve Phase 3's explicit re-scope (skip P3 D.1/D.2, keep
D.3 + Phases A/B/C, kill both Qt windows)? This should be recorded on the #199 epic so
the audits and this design stay consistent.

**Q2: replay control from the browser.** Is pit-wall v1 acceptable as a read-only
follower (recommended: it matches today's dashboard semantics), or must the browser
seek/pause the native replay from day one (adds the backend-to-arcade command path to
Phase 1)?

**Q3: relay transport.** WebSocket (recommended: bidirectional-ready for Q2, same build
cost now) or SSE (marginally simpler, one-directional, reuses `eventsource-parser`)?

**Q4: where the pit-wall epic lives.** Inside epic #25 as a post-S3 sprint (recommended:
same stack, same repo, same reviewers) or as a separate epic in the parent repo
referencing #25?

**Q5: broadcast-tier default.** Rival broadcast-tier traces default hidden (strict
realism, recommended) or visible-with-label (demo appeal)?

**Q6: timing of R1.** Land the additive rival fields (sectors, speeds, out-lap) now
(helps the dashboard tower AND pre-positions the Rival Agent's F3 sector features) or
defer to the TFM's M0 data milestone? Recommended: now; it is small and test-covered by
R5.

**Q7: 3D or not.** Is a Three.js/react-three-fiber 3D track view actually wanted, or is
the 2D canvas map the product? Recommended: 2D is the product (it is what real walls
use: schematic clarity beats spectacle); 3D remains the code-split flagship garnish the
migration plan already reserves for Home-hero moments, never the working surface.

---

## 7. Related documents

- `documents/research/RIVAL_AGENT_DESIGN.md`: the TFM design whose section 4 this
  document refines (see R6); its sections 5-8 consume the taxonomy directly.
- `documents/audits/AUDIT_P3_ARCADE.md`: the Arcade findings register (A1-A19) and
  phases; Phase 3 here re-scopes its Phase D pending Q1.
- `documents/audits/AUDIT_P2_LOADING.md`: F-05 SoA cache (the bulk payload), F-06
  threaded load, F-11 path routing; the shared-cache architecture the data plane rides.
- `documents/audits/AUDIT_P2B_CORE_COMPUTE.md`: the shared engine (`run_lap`) that
  produces the decisions every surface renders; F10/F11 shape what the strategy ribbon
  receives.
- `src/simulation/race_state_manager.py`: the boundary (driver `:154-217`, rivals
  `:219-281`, weather `:283-320`, contract `:338-374`).
- `src/arcade/stream.py`, `src/arcade/app.py:416-488`, `src/arcade/config.py:168-174`:
  the broadcast server, payload and cadence the relay consumes.
- `src/arcade/dashboard/`: the two Qt windows this design migrates (window.py and the
  cards) and kills (telemetry_window.py, telemetry_panel.py).
- `src/telemetry/backend/api/v1/endpoints/strategy.py:898-925` and
  `src/telemetry/backend/services/simulation/simulator.py`: the existing SSE simulation
  stream, the seed of the backend-native streamer (O2).
- `src/telemetry/docs/migration/MIGRATION_PLAN.md` and epic #25: the stack, sprints and
  design system every web recommendation here reuses.
- `data/raw/<year>/<gp>/intervals.parquet` + the P5 audit's Phase 4 items 13-15: the
  gap provider's source and the Rival data readiness pack.
