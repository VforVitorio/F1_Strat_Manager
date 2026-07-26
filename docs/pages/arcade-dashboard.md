# Arcade Dashboard Architecture

Developer-level reference for the PySide6 dashboard that ships alongside the arcade replay. Written for someone who plans to extend or modify either window (add a new sub-agent card, a new telemetry chart, retheme the palette, change the wire protocol).

Phase 3.5 Proceso B shipped thirteen files under `src/arcade/dashboard/` plus a new `src/arcade/strategy_pipeline.py` and the rewritten `src/arcade/strategy.py::SimConnector`. The arcade autolaunches the dashboard subprocess when the user enables strategy mode; the user never runs a second command.

## Three-window split

Three windows, two processes.

- **Arcade replay**: `pyglet`-backed, owned by `F1ArcadeView`. Drives the simulation loop, owns the `StrategyState`, runs `TelemetryStreamServer` on `127.0.0.1:9998`, and renders the track.
- **Strategy dashboard**: `PySide6` `MainWindow`. Orchestrator card, six sub-agent cards with embedded `pyqtgraph` charts, scenario bars, six-tab reasoning panel.
- **Telemetry window**: `PySide6` `TelemetryWindow`. Standalone `QMainWindow` with a 2x2 grid of `pyqtgraph` plots (Delta, Speed, Brake, Throttle) in F1-broadcast style.

The pyglet window runs in the arcade process. The two Qt windows live together inside one subprocess spawned by the arcade.

## Process topology

```
┌────────────────────────────────┐        ┌──────────────────────────────┐
│ Arcade process (pyglet)        │        │ Dashboard subprocess (Qt)    │
│                                │  TCP   │                              │
│  F1ArcadeView                  │◄──────►│  QApplication                │
│   ├─ RaceReplayEngine thread   │ 127.   │   ├─ MainWindow              │
│   ├─ StrategyState (_lock)     │ 0.0.1: │   │    + TelemetryStreamClt  │
│   └─ TelemetryStreamServer     │ 9998   │   └─ TelemetryWindow         │
│        accept thread           │        │         + TelemetryStreamClt │
└────────────────────────────────┘        └──────────────────────────────┘
```

Two processes, not three. Both Qt windows share the same `QApplication` event loop and Python interpreter, spawning a third process just to host the telemetry window would cost an extra Python startup (roughly 300 ms), another TCP socket, and another set of imported heavy modules (`torch`, `transformers`) with no gain.

The arcade process stays free of PySide6. Importing Qt into the pyglet process would double the memory footprint and couple the two event loops. Keeping Qt out of the arcade is the reason `stream.py` is stdlib-only; the subprocess launch preserves that separation.

## Package layout

### `theme.py`

Colour palette, compound pill colour map, flag chip styles, monospace font stack, and the `apply_dark_palette(app)` helper. The palette mirrors the arcade window's `styles.py`. Pirelli compound colours use the canonical hexes (C1 white, C2 yellow, C3 red, INTERMEDIATE green, WET blue).

### `stream_client.py`

`TelemetryStreamClient` is a `QThread` that opens a TCP socket to `127.0.0.1:9998`, reads newline-delimited JSON payloads, and emits a `data_received(dict)` Signal on the main Qt thread. Reconnects automatically with exponential backoff; exposes `connection_changed(str)` so the header bar can light a chip green/red.

### `window.py`

`MainWindow` composes the strategy surface. A header bar (40 px) shows session label, driver, connection chip, playback chip, and lap counter. A central `QSplitter(Qt.Horizontal)` holds two panels at `540 / 740`. The status bar shows the last error from the stream client.

`MainWindow.on_data_received` is the fan-out router: pulls `latest.per_agent`, dispatches each sub-agent dict to the matching card, drives the orchestrator card from `latest.recommendation`, feeds the scenario bars from `latest.scenario_scores`, and appends to the reasoning tabs from `latest.reasoning_per_agent`.

### `orchestrator_card.py`

The flagship card. Four visual elements:

- **Action badge**, large pill coloured by `classify_action`: green STAY_OUT, amber PIT_NOW, cyan UNDERCUT, magenta OVERCUT, red ALERT.
- **Confidence bar**: `QProgressBar` with a `qlineargradient` stylesheet painting a traffic-light gradient.
- **Pace and Risk chips**, two smaller pills, recoloured per regime.
- **Plan strip**, one line: "Plan: PIT lap 28, fit C3, target UNDERCUT HAM". The compound is rendered as an inline pill.
- **Guardrail line**, shown only when the no-LLM hard guard overrode the LLM pick.

### `agent_card.py`

Reusable widget: headline label, body `QLabel` (rich text with small monospace), and a reserved chart slot. The Pace and Tire cards slot in their `pyqtgraph` plots via `card.set_chart(widget)`. The Pit and RAG cards dim to 60 % opacity when the conditional agent did not fire on the current lap.

### `agent_formatters.py`

Six pure functions: `format_pace`, `format_tire`, `format_situation`, `format_pit`, `format_radio`, `format_rag`. Each takes a sub-agent output dict and returns `(headline: str, body_lines: list[str])`. Mirrors the CLI inference panel section rules; keeping the mapping pure means the formatters are importable anywhere and unit-testable without Qt.

### `pace_chart.py` and `tire_chart.py`

`pyqtgraph.PlotWidget` subclasses embedded in their cards.

- `PaceChart` plots actual lap time, predicted lap time, and a shaded P10/P90 confidence band per lap.
- `TireChart` plots tyre life by compound with vertical lines at stint boundaries and horizontal dashed lines at the estimated cliff laps.

Both charts scroll the x-axis to show the last 20 laps.

### `scenario_bars.py`

Four horizontal bars for STAY_OUT / PIT_NOW / UNDERCUT / OVERCUT. The raw Monte Carlo scores are sometimes negative; the widget shifts the minimum to zero, then normalises to [0, 1], so the visual remains readable while the tooltip exposes the raw score.

### `reasoning_tabs.py`

`QTabWidget` with six tabs: Pace, Tire, Situation, Radio, Pit, RAG. Each tab is a `QTextEdit` with a `QSyntaxHighlighter` subclass that paints regex patterns in accent colours, compound codes, flag keywords, numbers with units, and action verbs.

### `telemetry_panel.py` and `telemetry_window.py`

The 2x2 grid of telemetry plots. Lives in its own module so `TelemetryWindow` can host it as the central widget without pulling in the strategy-side imports.

## Wire protocol

The arcade broadcasts one JSON dict per frame, roughly 10 Hz, as a newline-terminated payload. The full shape:

```json
{
  "arcade": {
    "session": { "year": 2025, "round": 3, "location": "Suzuka", "lap": 18, "total_laps": 53 },
    "driver": { "code": "VER", "team": "Red Bull Racing", "position": 2 },
    "driver2": { "code": "LEC", "team": "Ferrari", "position": 4 },
    "telemetry": {
      "main":  { "speed": 312.4, "throttle": 0.98, "brake": 0.0, "gear": 7, "drs": true },
      "rival": { "speed": 308.1, "throttle": 0.95, "brake": 0.0, "gear": 7, "drs": false }
    },
    "delta_s": -0.412
  },
  "strategy": {
    "latest": {
      "lap": 18,
      "recommendation": { "action": "STAY_OUT", "confidence": 0.73, "pace_mode": "NEUTRAL", "risk": "BALANCED" },
      "scenario_scores": { "STAY_OUT": 0.21, "PIT_NOW": -0.14, "UNDERCUT": 0.08, "OVERCUT": -0.02 },
      "per_agent": {
        "pace":      { "lap_time_pred": 93.42, "ci_p10": 92.81, "ci_p90": 94.03 },
        "tire":      { "laps_to_cliff_p50": 12, "warning_level": "MONITOR" },
        "situation": { "overtake_prob": 0.42, "sc_prob_3lap": 0.11 },
        "pit":       { "action": "HOLD", "compound_recommendation": "C3" }
      },
      "reasoning_per_agent": { "pace": "...", "tire": "..." },
      "active": ["pit"],
      "guardrail_override": false
    }
  },
  "playback": { "state": "PLAY", "speed": 1.0 }
}
```

Top-level keys: `arcade`, `strategy`, `playback`. The dashboard's fan-out router reads from these three roots and nothing else, extending the protocol is additive.

## Extension points

### Adding a new sub-agent card

1. Extend the wire protocol: add the new agent's output dict under `strategy.latest.per_agent.<name>`.
2. Add a formatter in `agent_formatters.py` (`format_<name>(payload) -> (headline, body_lines)`).
3. Instantiate an `AgentCard(title=...)` inside `MainWindow._build_right_panel`.
4. In `MainWindow.on_data_received`, dispatch the new payload into the formatter and call `card.set_content(headline, body)`.
5. If the card needs a chart, create a `pyqtgraph` subclass following `pace_chart.py`.

### Changing the palette

`theme.py` is the only place colour literals live. Every widget imports `ACCENT`, `DANGER`, `SUCCESS`, `WARNING`, `TEXT_SECONDARY`, and the compound colour map from there. A full retheme means editing `theme.py` plus the matching `styles.py` on the arcade side.

## Threading model

Five threads cooperate.

- **Arcade main thread** (pyglet), runs the `F1ArcadeView.on_update` tick, mutates the `StrategyState` under `_lock`, calls `TelemetryStreamServer.broadcast(snapshot)`.
- **SimConnector background thread**, iterates `RaceReplayEngine.replay()` and calls `run_strategy_pipeline(race_state, laps_df)` per lap. One thread, one lap at a time, so the main pyglet thread never blocks on LLM inference.
- **TelemetryStreamServer accept thread**, daemon thread inside the arcade process. Blocks on `server_socket.accept()`.
- **Qt main thread** (dashboard subprocess), runs `QApplication.exec()`. Drives all UI updates. Never touches the network.
- **QThread stream clients** (one per top-level Qt window), each owns its own TCP socket.

The `StrategyState._lock` is the only contended mutex. Neither thread holds it across I/O.
