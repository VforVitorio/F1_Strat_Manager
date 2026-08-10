# PITWALL migration reference: the Qt windows being replaced

Captured 2026-08-07 from the live `src/arcade/dashboard/` against a real
Melbourne 2025 session. **These are the acceptance reference for sprint 3**,
which ports the AGENTS window 1:1. Do not port from memory; port from these
plus Gate B's field-by-field inventory in
`documents/audits/GATE_PITWALL_ARCH_B.md`.

| File | Window | Becomes |
|---|---|---|
| `legacy-qt-strategy.png` | `MainWindow` ("F1 Strategy Dashboard") | PITWALL · AGENTS |
| `legacy-qt-telemetry.png` | `TelemetryWindow` ("F1 Live Telemetry") | PITWALL · DATA, band 4 |
| `pitwall-data-band4.png` | the PORT of the telemetry window | — (this is the "after") |

## The after, and why it is here

`pitwall-data-band4.png` is the sprint-4 result, captured at the same 2250x1410
as the Qt image so the two can be laid side by side without scaling either.
Sprints 8 and 9 (the elevate pass, where the 1:1 constraint is lifted) are
specified to receive **before-and-after screenshots**, and sprint 7 deletes the
Qt window that produced the before — so both halves live in the repo rather
than in a session scratchpad.

Captured by `src/pitwall/ui/scripts/shot-data.mjs` replaying 120 consecutive
ticks taken off `scripts/dev_pitwall_producer.py`, which drives the real
`TelemetryStreamServer` from the real Melbourne 2025 session. The bundle and
the data are both real; the only thing faked is `window.pywebview`, which the
OS shell would otherwise inject.

**Three differences from the Qt image are deliberate**, and the rest is meant
to match field for field:

1. The rival legend carries a `BROADCAST` tag. `PITWALL_REALISM_AND_TELEMETRY_
   SURFACE.md` 2.2 requires it and names the Qt window's omission by name.
2. A shared vertical cursor marks where the car is on the lap, on all four
   charts. It comes from `drivers[main].rel_dist × circuit_length_m`, not from
   the tail of the telemetry span, because only the drivers block is published
   on a paused tick.
3. The tick DENSITY differs (ECharts picks 1000 m where pyqtgraph picked 500,
   and 100 km/h where it picked 50). That is a renderer choosing labels for the
   width it was given, not a range difference: the ranges are identical and the
   smoke asserts each one against the axis's computed extent.

## The decision these serve

**1:1 first, elevate afterwards** (Víctor, 2026-08-07), the same order the
Streamlit → React migration used. Sprint 3 ports the layout as it stands;
the design pass comes after, when there is something to compare against.

## What the strategy window contains

- **Header**: `Melbourne · 2025` + driver code, and on the right a
  `Connected` pill, `2.00x · PLAYING`, `L 24/57`.
- **Left column**: the orchestrator card (action button, confidence bar,
  Pace/Risk pills, `Pit: L24 · Next: HARD · UCUT: RUS`, guardrail line),
  scenario bars (STAY / PIT / UCUT / OCUT with values, `--` when absent),
  and the reasoning tabs (ORCHESTRATOR / PACE / TIRE / SITUATION / RADIO /
  PIT) with a monospace body and the coloured
  `— why this call changed —` block.
- **Right column**: a 3x2 grid of agent cards — PACE, TIRE, SITUATION, PIT,
  RADIO, RAG. Each has a status dot: **filled** when the agent ran this lap,
  **hollow** when it is idle, and an idle card shows its trigger condition
  instead of numbers ("triggers on cliff pressure, compound change, or
  problem radio"). PACE and TIRE own line charts.
- **Status bar**: `lap 24 · streaming`.

## What the telemetry window contains

`LAP 24` + the `NOR vs PIA` chips, then a 2x2 grid: **Δ Time (rival − main)**,
**Speed km/h**, **Brake Pressure %**, **Throttle %**. Every chart carries a
`MAIN · NOR / RIVAL · PIA` legend and an X axis in **distance within the lap**,
locked to the circuit length. Status bar reads `lap 24 · live`.

## Read these with three caveats

1. **The agent-card numbers in `legacy-qt-strategy.png` are placeholders, and
   that was a defect in the rig, not a property of the window.** The producer
   that fed this capture built `PerAgentOutputsDTO` with plausible-looking
   keys that did not match what the cards read, so PACE shows `+0.000s`,
   TIRE shows `deg — s/lap`, SITUATION shows `safety car 0%`, and PIT and RAG
   sit on their trigger hints because `active` carried block names instead of
   the `N28` / `N30` routing tokens. **Fixed in #853**: re-running
   `scripts/dev_pitwall_producer.py` today populates all six cards, so a fresh
   capture will legitimately differ from this one. The **layout, the states
   and the copy** in this image are real; the figures in those cards are not.
   Everything in the left column is real.
2. **The windows were resized before capture.** At their default geometry the
   right column is clipped mid-card and the reasoning tabs get about 268 px,
   which is why the `why this call changed` block normally falls below the
   fold. That is a Qt fixed-height problem HTML does not have, and it is one
   of the things the port fixes for free.
3. **The traces cover part of a lap** because the capture is mid-lap. Their
   density is the point: continuous curves rather than the scatter the same
   charts drew before the broadcast started sending the whole span (#841).

## How they were captured

`QWidget.grab()`, which paints the widget through Qt's own renderer. Not a
screen grab: a screen-rectangle capture returns whatever is physically in
front, and an attempt at one during this sprint captured an unrelated window
because `SetForegroundWindow` is refused to background processes. `PrintWindow`
is the right answer for native windows and returns a blank surface for Qt,
which composites through the GPU. The script lives in the session scratchpad;
it is a dev tool, not part of the product.
