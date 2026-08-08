# PITWALL v2 architecture (v2.6.0, "Arcade, modernized")

**Status: design, agreed with Victor 2026-08-07 in a Socratic design session. NO code written.**
**This document SUPERSEDES `PITWALL_REALISM_AND_TELEMETRY_SURFACE.md` sections 3 to 6 and the
phase decomposition of epic #281.** Its Topic 1 (the observability model, sections 1 to 2) stands
and is unaffected.

Companion documents: `documents/research/PITWALL_REALISM_AND_TELEMETRY_SURFACE.md` (Topic 1),
`documents/audits/AUDIT_P3_ARCADE.md` (the A1-A19 register this re-scopes), and the session's
reasoning chain in `~/.claude/plans/pitwall-design.md`.

---

## 1. What changed, and why the old plan is void

The July design assumed the pit-wall surface would be **a page in a browser**. Every downstream
decision followed from that one assumption:

> "Browsers cannot open raw TCP sockets; a page can only speak HTTP(S), SSE, WebSocket, or WebRTC."
> (section 3.2)

and therefore, in the body of epic #281:

> "CONSUMING the Arcade TCP stream via a **FastAPI WebSocket relay** (browsers cannot open raw TCP)"

**That sentence is true for a browser page and false for a desktop application.** Victor's decision
of 2026-08-07 is that PITWALL is a desktop surface built with web technology, not a web app. The
host process is Python, so it opens the TCP socket itself. Consequences:

- The FastAPI WebSocket relay is **not needed**. Issue #283 as written is void.
- The FastAPI backend is **not a runtime dependency**. Arcade today makes zero HTTP calls (verified:
  no `requests`, no `httpx`, no live use of `BACKEND_URL`; `config.py:169-170` are dead constants
  from the SSE era, P3 finding A15). PITWALL inherits that property.
- The bulk-prefetch endpoint is not needed either, for the same reason.

The other decisions that moved:

| July | Now |
|---|---|
| Migrate the agent cards to the webapp SPA; KILL the telemetry window | Both windows become PITWALL windows. The telemetry one GROWS. |
| Arcade keeps the circuit window; the Qt dashboard retires | Same outcome, different destination. |
| Pit-wall v1 is read-only, following the replay | **Unchanged.** Still read-only. |
| P3 #199 Phase D items D.1/D.2 skipped as moot | **Still moot**, for the same reason: their host stops existing. D.3 survives and matters more. |

---

## 2. Process topology

**Today** (verified: `src/arcade/main.py:33`, `src/arcade/app.py`, `src/arcade/dashboard/__main__.py:36-42`):

```
f1-arcade
├── process 1: pyglet          arcade.Window -> F1ArcadeView  (circuit replay, owns the clock)
│                              + TelemetryStreamServer on 127.0.0.1:9998
└── process 2: Qt subprocess   QApplication
                               ├── MainWindow      (strategy)  -> its OWN TelemetryStreamClient
                               └── TelemetryWindow (telemetry) -> its OWN TelemetryStreamClient
```

**After:**

```
f1-arcade
├── process 1: pyglet          UNCHANGED. circuit replay + TelemetryStreamServer.
└── process 2: pywebview       python -m src.pitwall
                               ONE TCP client -> one state slot -> two windows
                               ├── window "PITWALL · AGENTS"  (agents.html)
                               └── window "PITWALL · DATA"    (data.html)
```

The subprocess count does not change, the spawn mechanism does not change (`subprocess.Popen` from
`_init_strategy_layer`, teardown in `on_hide_view`), and Arcade remains the master clock. **The
follower topology is the one already running and proven; only the follower's technology changes.**

**One TCP client, not two.** It halves the receive-decode-parse cost and gives one place to own the
tick sequence. It does **not** fix P3 finding A7 "by construction": Gate A measured today's two
clients against the real server and found `identical sequences: True`, 200/200, zero drops, so
A7's drift clause is an overstatement. See 3.5 for the version of this that is actually correct.

---

## 3. The data plane

The single most important design decision in this document. **The two windows consume two channels
with completely different shapes, and confusing them is what makes dashboards slow and wrong.**

### 3.1 Channel A: the TICK (~10 Hz, from the TCP stream)

The existing broadcast payload, unchanged (`app.py:432-441`): an `arcade` block (per-driver
`lap, dist, speed, compound, tyre_life` for all 20, plus a single telemetry sample for main and
rival), a `strategy` block (`snapshot_dict`: latest decision + a 30-entry history tail), and a
`playback` block (`speed, paused, frame_index, total_frames`).

This channel carries **the present**: where the cars are now, what the orchestrator just said, and
where the clock is.

### 3.2 Channel B: the BULK (loaded once, read from disk by the PITWALL process)

Everything that is **static for the whole replay**: every driver's lap table (lap number, lap time,
S1/S2/S3 with their speeds, speed trap, compound, stint, pit in/out flags), circuit metadata, and
per-lap telemetry traces on request.

**In a replay the entire race is known before lap 1.** The race-pace grid, the bests panel and the
timing table's history are therefore a **progressive reveal masked by the current lap**, not a
stream to accumulate. The PITWALL process reads this itself, from the same on-disk artefacts the
arcade process reads. Bootstrap: the first tick carries `gp_name`, `year`, `driver_main` and
`driver_rival`, which is enough to resolve the session and load its lap table.

### 3.3 REFUTED BY GATE A (D-01, P0). Superseded by 3.3b.

> **This section was wrong and is kept only so the reasoning is legible.** It assumed a per-lap
> telemetry slice could be read from disk. There is no telemetry on disk: `data/raw/<year>/<gp>/`
> holds `laps`, `intervals`, `pitstops`, `weather` and `metadata` only (verified by listing it).
> The single 25 Hz source in the system is `SessionData.frames_by_driver`, persisted as ONE
> monolithic pickle (`src/arcade/data.py:334`, `<GP>_<year>_race.pkl`). A pickle has no random
> access, so risk 3's own mitigation ("read lap slices, never the whole session") described an
> access pattern the storage format does not offer. The only ways to honour it were an 8.0 s full
> unpickle (P2 finding F-05) per lap change, or a second full FastF1 session load. Both unbudgeted.

**The original claim, for the record:** the own-car trace panel is fed from BULK, per lap, on
demand, not from the tick.

Recall the measured problem: `_build_arcade_snapshot` puts ONE telemetry sample per broadcast on the
wire. At 1x the clock advances 25 samples per second and 10 are sent, so 60% never leaves the arcade
process; at 8x, 95% is lost. A trace panel fed by that channel degrades as you speed up, which is
exactly backwards from what a user expects.

Fed from bulk instead, the panel requests `get_lap_trace(driver, lap)` when the lap changes and
receives the full 25 Hz trace for that lap (roughly 2,000 samples, small). The tick then supplies
only `frame_index`, which the panel uses to place the cursor and to decide how much of the lap to
reveal. This kills, in one decision:

1. **Decimation.** The trace is always at full resolution regardless of playback speed.
2. **Accumulation.** The panel holds no history of its own; it is a function of (lap, frame_index).
3. **The rewind problem, for this panel.** Seeking backwards just moves the cursor.

Pinning a rival is a **request**, not a subscription: `get_lap_trace(rival, lap)`.

### 3.3b The actual fix: send the SPAN, not the sample

The producer already holds the answer in memory, which is why the disk detour was never needed.

`_build_arcade_snapshot` today puts `frames[frame_idx]` on the wire: **one** sample per broadcast
(`src/arcade/app.py:473-479`, via `_frame_to_telemetry`, which takes a single `FrameData`). The
clock advances 25 indices per wall-clock second at 1x and the broadcast fires ~10 times a second,
so 60% of the samples never leave the arcade process; at 8x the clock advances 200 and 95% is lost.

**Send `frames[last_sent_idx + 1 : frame_idx + 1]` instead.** The array is already in RAM in the
producer. The payload grows from 1 sample to ~2.5 at 1x and ~20 at 8x, and only for the main
driver and the pinned rival, so the wire stays small. This removes decimation **at every playback
speed**, needs no disk read, no second channel, and no unpickling.

Consequences for the panel: it now receives a contiguous span per tick and appends it, so it DOES
accumulate and DOES need the rewind guard of 3.4. That is a real cost the refuted design pretended
to avoid, and it is cheap: truncating a distance-keyed buffer is a slice.

**Four scalars must join the wire** for the DATA window to be correct at all (Gate A, D-06/D-04):

| Field | Why it is not optional |
|---|---|
| `global_t_min` | The wire's `t` is exactly `frame_index * 0.04` (`data.py:376` builds the timeline from zero) and `global_t_min` is computed at `data.py:374` and **never stored on `SessionData`**. Without it `frame_index` cannot address anything keyed by session time, which is why `intervals.parquet` is downloaded for every race and read by nothing: there is no join key. |
| `active` | `FrameData.active` is computed (`data.py:404`) and then **deliberately dropped** from the broadcast (`app.py:449`). Melbourne has six retirements, three on lap 1, and those cars keep broadcasting frozen lap-1 telemetry for 56 laps. The Qt dashboard never noticed because it renders two cars; a 20-row timing tower and a ring would render them as running. |
| `rel_dist` | Dropped by the same line. The ring is a function of it. |
| `location` | Session resolution for the BULK reader without guessing from `gp_name`. |

About 12 bytes per car per tick.

### 3.4 The rewind guard, for what still accumulates

Victor's decision: rewind is the cheap kind, "so you do not miss something", not a study tool.
Panels do NOT need to be pure functions of `frame_index`.

One module, `lib/frameClock.ts`, holds the last seen `frame_index`. When an arriving tick carries a
LOWER index, it emits a `truncate(frameIndex)` event. Panels that accumulate subscribe and drop
everything after that point.

**Which panels actually accumulate?** The original claim here was "the DATA window accumulates
nothing", and **Gate A partly refuted it (2 of 6 panels)**. The corrected list:

| Panel | Accumulates? | Why |
|---|---|---|
| Timing table | No | function of (bulk lap table, current lap) |
| Bests | No | same |
| Race pace grid | No | progressive reveal of the bulk lap table |
| Race trace | No | same |
| **Own-car traces** | **YES** | it appends the spans of 3.3b, so it needs the guard |
| **The ring** | **YES, and it needs `rel_dist` on the wire** | it is a function of the instant, not of the lap |
| AGENTS PaceChart / TireChart | YES | `history_tail` strips `per_agent` (a deliberate wire-size trade-off, `window.py:171-174`), so the window owns those series |

The deeper correction from Gate A (D-06): with no time anchor on the wire, "a function of (bulk,
lap, frame_index)" was really "a function of (bulk, lap)". Anything genuinely instantaneous, above
all **live gaps and intervals**, cannot be computed at all until `global_t_min` ships (3.3b).

Today's behaviour for reference, verified: pause works by accident (`_broadcast_if_due()` is called
unconditionally in `on_update` at `app.py:402`, so a paused arcade keeps broadcasting the same frame
and followers freeze); rewind is broken (`telemetry_panel.py:262` `_append`s into a distance-keyed
bucket and `MainWindow._pace_history` is a lap-keyed dict, and nothing deletes the future).

### 3.5 Transport from Python to JavaScript

**Decision: `js_api` pull, and it MUST be sequenced.** The JS side calls
`pywebview.api.get_tick(since_seq)`; the host keeps a monotonically increasing sequence and returns
what the caller has not seen.

> **Gate A (D-10) refuted the naive version of this by execution.** A single latest-payload slot
> with two windows polling independently at 10 Hz makes the two windows read **a different frame on
> 58% of polls** (15 duplicates and 15 skips out of 54). The sequence parameter is what removes
> both. Gate A also refuted the justification for the change: driving the real
> `TelemetryStreamServer` with two sockets gave `identical sequences: True`, 200/200, zero drops,
> so today's two clients do NOT drift. P3 finding A7's drift clause is an overstatement and this
> design promoted it into a rationale. **One client is still right** (it halves the parse cost and
> gives one place to hold the sequence), but not "by construction" and not because A7 said so.

The original wording, kept because the reasoning below it still holds: the JS side calls
`pywebview.api.get_tick()` on its own cadence; the Python host returns whatever is in the latest
payload slot, which the TCP client thread overwrites.

- **Why pull and not push.** Natural backpressure: the UI never receives faster than it renders, and
  the cadence becomes a UI concern rather than a wire concern. It is also the laziest thing that
  works, and 10 Hz of RPC over an in-process bridge is nothing.
- **Rejected: `window.evaluate_js` push.** Marshals JSON into JavaScript source text and evals it,
  twice per tick (two windows), 10 times a second.
- **Rejected for v1, named as the upgrade path: an in-process WebSocket server** inside the PITWALL
  host. Cleanest data flow and an idiomatic `onmessage` on the JS side, but it adds a dependency and
  a port to solve a problem we have not measured. Revisit only if the pull model is measured to hurt.

---

## 4. File layout

```
src/pitwall/                        NEW, in the PARENT repo
├── __init__.py
├── __main__.py                     entry point: build host, create 2 windows, webview.start()
├── config.py                       window titles/geometry, stream host/port, asset resolution
├── host.py                         PitwallHost: the object exposed to JS as pywebview.api
├── stream_client.py                the TCP client (ONE), thread + latest-payload slot
├── session_data.py                 BULK: resolve the session, load the lap table, slice lap traces
└── ui/                             the Vite project (SOURCE; built output ships as package data)
    ├── package.json
    ├── vite.config.ts              two entry points -> data.html, agents.html
    ├── data.html
    ├── agents.html
    └── src/
        ├── main-data.tsx
        ├── main-agents.tsx
        ├── styles/tokens.css       COPY of the webapp's, guarded by a drift test
        ├── lib/
        │   ├── bridge.ts           the pywebview.api wrapper, typed
        │   ├── frameClock.ts       the monotonic guard (section 3.4)
        │   ├── useTick.ts          subscription hook
        │   └── format.ts           lap times, gaps, deltas
        ├── charts/                 ECharts registration + theme, ported from the webapp
        ├── features/data/
        │   ├── DataWindow.tsx      the four-band shell
        │   ├── StatusStrip.tsx     band 1
        │   ├── TimingTable.tsx     band 2 left, and THE SELECTOR
        │   ├── BestsPanel.tsx      band 2 right
        │   ├── RacePacePanel.tsx   band 3 shell, owns the Run Timeline | Race Trace tabs
        │   ├── RunTimeline.tsx     the drivers x laps heat grid
        │   ├── RaceTraceChart.tsx  the gapper plot
        │   ├── OwnCarTraces.tsx    band 4, stacked traces + shared cursor
        │   └── TrackRing.tsx       band 4 corner
        └── features/agents/
            ├── AgentsWindow.tsx    header + split + status bar, 1:1 with window.py
            ├── HeaderBar.tsx
            ├── OrchestratorCard.tsx
            ├── ScenarioBars.tsx
            ├── ReasoningTabs.tsx
            ├── AgentCard.tsx       shared by the six
            ├── PaceChart.tsx
            └── TireChart.tsx
```

**Deleted when this lands:** `src/arcade/dashboard/` entirely (13 modules), plus the spawn of
`python -m src.arcade.dashboard` in `app.py`. `PySide6` and `pyqtgraph` leave `pyproject.toml`;
`pywebview` enters it.

**Unchanged:** `src/arcade/app.py` except for the subprocess target, `src/arcade/stream.py`,
`src/arcade/track.py`, `src/arcade/overlays.py`, `src/arcade/data.py`, `src/simulation/`,
`src/agents/`, `src/strategy/inference/engine.py`.

---

## 5. Module contracts

**`stream_client.py`** owns one socket and one slot. A daemon thread reads newline-delimited JSON
and overwrites `self._latest`. No parsing beyond `json.loads`, no fan-out, no signals. The reason it
is not the Qt client renamed: that one is a `QThread` emitting Qt signals, and Qt is leaving.

**`session_data.py`** is the BULK reader and the one place that touches disk. It exposes
`lap_table(year, gp)`, `lap_trace(year, gp, driver, lap)` and `circuit_meta(year, gp)`, memoised.
**It must reuse the project's existing loaders rather than reading parquet directly** (see the
duplication risk in section 8).

**`host.py`** is the js_api surface and nothing else. Methods map one-to-one to what the UI needs:
`get_tick()`, `get_bulk()`, `get_lap_trace(driver, lap)`. It holds no rendering logic and no
formatting. Each method is small enough to read in one screen.

**`bridge.ts`** is the only TypeScript module that knows `window.pywebview` exists. Everything above
it consumes typed functions. When the transport is upgraded (section 3.5), this is the only file
that changes.

**`frameClock.ts`** owns the monotonic guard and nothing else.

---

## 6. Code quality directives

Applies `~/.claude/CLEAN_CODE.md`. The project-specific points that matter here:

**Python side**
- Module docstring on every file: what it does, what invariant it holds, what to know before
  touching it. Class docstring with `Responsibilities:` where behaviour is non-obvious.
- Constants at the top in `UPPER_CASE`; no magic numbers mid-logic. Window geometry, titles, poll
  intervals and the stream port all live in `config.py`.
- Error handling per Martin: tiny `try` blocks around the one call that can fail, specific exception
  types, `raise ... from err` when re-raising, **never a bare `except`**. The socket layer is the
  only place with real failure modes (connect refused, peer closed, malformed line) and each gets
  its own named handler.
- `None` means "unknown data", exceptions mean "the operation failed". Never a sentinel number that
  the code could also legitimately find. This repo has a scar from exactly that (`_safe` turning a
  NaN `Position` into `0`, so the leader found the car that had just crashed).
- Split past ~300 lines. `host.py` in particular must not become a god object: if it grows past its
  three methods plus their guards, the new surface belongs in its own module.

**TypeScript side**
- One component per file, file name matching the component.
- **Chart data flows imperatively into ECharts instances through refs, never through per-frame React
  state.** This is the web equivalent of P3 finding A6 (the Qt dashboard re-rendering six cards, six
  syntax-highlighted text areas and a full chart rebuild 10 times a second for content that changes
  once per lap), applied before it happens rather than after.
- React renders panel chrome and once-per-lap content. The tick loop touches refs.
- **Animate the entrance, never the data update.** The repo already encodes this contract in
  `src/telemetry/webapp/src/charts/useFirstPaintAnimation.ts`; port it, do not reinvent it.
- Named intermediates over compound expressions, exactly as on the Python side.

**Both**
- A `pitwall:` style comment marking any deliberate simplification, naming the ceiling and the
  upgrade path.

---

## 7. Tests

The Qt surface's test coverage today is one import-smoke file (P3 finding A18). Do better, cheaply:

1. **Token drift test, covering ALL the copies and not just the new pair.** Gate B measured that the
   drift A16 warned about **has already happened**: `src/arcade/config.py` / `dashboard/theme.py`
   and the webapp's current `tokens.css` disagree on every semantic colour (background, accent,
   success, warning, danger). A test that only guards pitwall-vs-webapp would leave the
   already-broken pair uncovered before and after, which is this repo's most-repeated defect
   (one member of a pair fixed, its twin not) committed inside the document that names it.
2. **Frame-clock guard test.** Feed a descending `frame_index` and assert truncation fires. Pure
   function, no DOM.
3. **Bulk reader test.** `lap_table` on a known race returns the expected shape and the expected lap
   count; `lap_trace` returns only the requested lap.
4. **Host contract test.** Every `js_api` method returns JSON-serialisable data. A method returning
   a numpy scalar or a `Timestamp` fails silently across the bridge, which is the worst kind.
5. **Golden payload test.** The broadcast dict shape the UI depends on, frozen. P3 Phase D already
   proposed this; it stops mattering less when the consumer changes technology, it starts mattering
   more.

---

## 8. Risks, stated so a gate can attack them

1. **`session_data.py` is a duplication risk.** The repo's dominant defect is a second
   implementation drifting from the first, and the standing rule is that every consumer calls
   `augment_featured_laps` rather than reading the parquet. A new module that loads race data is
   exactly the shape of that defect. It must reuse `src/f1_strat_manager/data_cache.py` and the
   existing loaders, and the gate should verify that the chosen reader is the same one Arcade uses.
2. **pywebview depends on the OS webview.** Windows 11 and macOS ship theirs; **Linux needs
   `webkit2gtk` as a system package**, which `uv` cannot resolve. This is an install-note, not a
   blocker, but it must be written before someone hits it.
3. **Two processes now read the same session data.** Arcade holds `frames_by_driver` in memory;
   PITWALL reads lap slices from disk. If both trigger a cold load, the P2 finding F-05 cost (an
   8.0 s AoS unpickle) could be paid twice. Mitigation to verify: PITWALL should only ever read the
   small lap table plus per-lap slices, never the whole session.
4. **The `js_api` pull cadence is unmeasured.** 10 Hz across two windows is assumed cheap. If it is
   not, section 3.5 names the upgrade path.
5. **The DATA window has data the wire cannot supply today.** Position, gap, interval and rival lap
   times are NOT in the broadcast (`app.py:455-461` carries `lap, dist, speed, compound, tyre_life`
   only). They exist in `lap_state` and in the parquet. The design assumes they come from BULK
   (parquet, by lap) rather than from a widened wire. **A gate should check that assumption holds
   for gaps specifically**, because a gap is a function of the current instant, not of the lap.
6. **Race control messages have no home** in either window, and none in Arcade (`SessionData.events`
   is always empty, P3 finding A3). Unresolved.
7. **Published prose will be wrong.** `ROADMAP.md` v2.6.0 and `docs/pages/roadmap.md:464` both say
   "the strategy and telemetry surfaces move to a **web-native view**". Web technology, yes; web
   app, no. Both need rewriting, and this repo has a history of stale published claims.
8. **`src/arcade/__init__.py:4`** still claims Arcade renders the SSE simulation stream. It does not.

---

## 9. Reconciliation with the existing issues

| Issue | Fate |
|---|---|
| **#281** epic | Body rewritten: the Arcade split, the destination, and the relay sentence are all wrong. |
| **#282** Phase 0, observability contract | **Survives unchanged.** It is Topic 1 work, independent of where the UI lives, and it feeds the Rival Agent as well. |
| **#283** Phase 1, WS relay + bulk endpoint | **Void.** Its entire premise was the browser-tab assumption. Close with the reason recorded. |
| **#284** Phase 2, dashboard in the SPA | **Rewritten** as the PITWALL DATA window. |
| **#285** Phase 3, migrate cards + retire Qt | **Rewritten**: both Qt windows die, the cards move to PITWALL AGENTS 1:1. |
| **#286** Phase 4, rival intent + realism mode | **Survives**, still gated on the Rival Agent. |
| **#287** Phase 5, parity gate | **Survives**, unchanged in spirit: does the circuit view eventually move too. |
| **#199** P3 Arcade epic | Phase D items D.1 and D.2 stay moot. **D.3 (broadcast hygiene) survives and gains weight**: the broadcast is now the only wire, and `snapshot_dict` still re-runs a recursive `asdict` over the 30-entry history tail 10 times a second with a blocking `sendall` (A8). Phases A, B and C are unaffected. |

Gate B correction: closing **#283** wholesale would silently drop its gap-provider bullet
(`intervals.parquet`), which is precisely what risk 5 and section 3.3b need. Carry that bullet
forward rather than closing it with the relay.

---

## 10. Gate outcomes (2026-08-07)

Two adversarial gates, distinct lenses, reports at `documents/audits/GATE_PITWALL_ARCH_A.md`
(data plane and runtime correctness) and `GATE_PITWALL_ARCH_B.md` (repo fit and blast radius).
The P0s were independently re-verified before being accepted here.

| Section 8 risk | Verdict |
|---|---|
| 1. `session_data.py` duplication | **Moot.** 3.3b removes the disk read entirely. |
| 2. pywebview needs the OS webview | Upheld. Install note for Linux. |
| 3. Two processes reading the same session | **Refuted as written**: the mitigation described an access pattern a pickle cannot offer. Moot after 3.3b. |
| 4. Pull cadence unmeasured | **Refuted in its optimistic half**: unsequenced polling desyncs the two windows on 58% of polls. Fixed by `since_seq`. |
| 5. Wire lacks position/gap/interval | Enumeration correct, **conclusion wrong**: those cannot come from BULK-by-lap because a gap is a function of the instant and the wire has no time anchor. Needs `global_t_min`. |
| 6. RCM has no home | Upheld, still open. |
| 7. Published prose will be wrong | Upheld. |
| 8. Lying docstring in `src/arcade/__init__.py:4` | Upheld. |

### Two live bugs found that are in NO existing register

Neither is in the A1-A19 P3 register. PITWALL would inherit both.

- **Retired cars broadcast as if running.** `FrameData.active` is computed (`data.py:404`) and
  dropped from the broadcast (`app.py:449`). Melbourne: six retirements, three on lap 1, each
  transmitting frozen lap-1 telemetry for 56 laps. Invisible today because the Qt dashboard renders
  two cars; a 20-row tower and the ring would render them alive. Fixed by shipping `active` (3.3b).
- **The only gap computation divides by a hardcoded constant.** `overlays.py:326-336` assumes
  `55.56` m/s for every car (+57% error under Safety Car, -13.5% on a fastest lap), applied to a
  distance term that double-counts a lap per lap of difference (`overlays.py:502-511`).
  **Warrants its own issue before being fixed**, per the repo's bug-first rule.

### One latent risk recorded for whoever writes the trace reader

Gate A demonstrated by executing the code path, but could not show firing on real data, that the
interpolated `lap` channel (`data.py:402`, `np.interp` over a discrete channel) can label ~2,000
frames as a lap that has no telemetry behind it. P2 severity, latent.

### What the gates tried to break and could NOT

Decisions are levels rather than events, so the pull model drops nothing the design depends on.
`IsPersonalBest` is a running flag, not post-hoc, so the race-pace reveal has no look-ahead leak
(this was Gate A's strongest expected hit and it missed). The lap table has no missing lap numbers.
`get_rival_states` handles its sentinel correctly. Every file:line citation in section 3.4's
description of today's pause, rewind and decimation behaviour reproduces exactly. On the repo-fit
side: Arcade genuinely makes zero HTTP calls, and nothing outside the enumerated reference graph
assumes the Qt dashboard exists.
