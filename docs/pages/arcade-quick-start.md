# Arcade Quick Start

> End-user guide for running the F1 StratLab arcade replay with the live strategy dashboard. Aimed at someone who has cloned the repository and wants to see three synchronised windows on screen inside ten minutes.

One command launches everything: the arcade replay window (pyglet), the strategy dashboard (PySide6), and the telemetry window (PySide6). The arcade process owns the simulation loop and broadcasts merged state on a local TCP port; the dashboard subprocess subscribes and renders.

<p align="center">
  <video src="/assets/demo/arcade-demo.mp4" poster="/assets/demo/arcade-demo-poster.jpg" width="760" autoplay loop muted playsinline preload="metadata" aria-label="F1 StratLab arcade replay in action"></video>
  <br/>
  <sub>The three-window arcade: 2D replay, strategy dashboard and live telemetry, all in sync.</sub>
</p>

## Prerequisites

- **Python**: 3.10 or newer. The project pins dependencies with `uv`.
- **Dependencies**: run `uv sync` from the repo root. The lockfile pulls `arcade`, `PySide6`, `pyqtgraph`, `fastf1`, `langchain-openai`, the model stack (`xgboost`, `lightgbm`, `torch`), and the NLP stack (`transformers`, `sentence-transformers`, `setfit`). No manual install steps required beyond `uv sync`.
- **LLM credentials**: either set `OPENAI_API_KEY` in a repo-root `.env` (the canonical TFG setup) or run LM Studio locally on `http://localhost:1234/v1` and pass `--provider lmstudio` on the command line (the arcade's own flag; its default is `openai`, independent of the `F1_LLM_PROVIDER` env var the backend and CLI read). Only the wording of the orchestrator's reasoning changes.
- **Race data cache**: the replay reads `data/raw/{year}/{Location}/laps.parquet` and optionally `weather.parquet`. The parquet files are produced by FastF1 on first run. Expect a 20–40 second delay on the first launch of a round.
- **Vector store (optional)**: the N30 RAG agent reads a local Qdrant index under `data/rag/`. If missing, the orchestrator degrades gracefully, regulation lookups return an empty context. Run `python scripts/build_rag_index.py` once to build it.

## One-command launch

From the repository root:

```bash
python -m src.arcade.main --viewer --year 2025 --round 3 --driver VER --team "Red Bull Racing" --driver2 LEC --strategy
```

What happens:

1. `src.arcade.main` parses the CLI and opens the arcade `Window`.
2. The `--viewer` flag skips the menu and goes straight to `F1ArcadeView`.
3. The view loads the 2025 Round 3 (Suzuka) parquet for Verstappen.
4. With `--strategy` set, the view starts a `TelemetryStreamServer` on `127.0.0.1:9998`, owns a `StrategyState`, and spawns the dashboard subprocess with `python -m src.arcade.dashboard`.
5. The dashboard opens two PySide6 windows that both connect back to the stream.

Three windows are now on screen. The arcade window drives playback; the two Qt windows react to broadcasts.

## What each window shows

### Arcade replay (pyglet)

Track outline with the DRS zones drawn in green, two driver icons (our driver in team colour, rival in rival-team colour), a leaderboard on the right with compound pills and gaps, a weather panel, a driver-info strip, and a progress bar with the current lap.

### Strategy dashboard (PySide6)

A `QSplitter` divides the window into two halves.

- **Left panel (540 px wide)**: orchestrator card on top (action badge, confidence bar, pace/risk chips, plan strip with compound pill, and a guardrail line that shows when the no-LLM hard guard overrode the LLM pick), scenario bars in the middle (four bars for STAY_OUT / PIT_NOW / UNDERCUT / OVERCUT), reasoning tabs on the bottom (six tabs with syntax highlighting).
- **Right panel (740 px wide)**: a 3x2 grid of sub-agent cards. Pace (N25), Tire (N26), Situation (N27), Radio (N29), Pit (N28, dimmed when inactive), RAG (N30, dimmed when inactive). Each card has a headline, a short body, and a reserved chart slot: Pace and Tire host embedded `pyqtgraph` plots.

### Telemetry window (PySide6)

Standalone `QMainWindow` with a 2x2 grid of `pyqtgraph` plots: **Delta** (our driver vs rival), **Speed**, **Brake**, **Throttle**. The window owns its own `TelemetryStreamClient` so it can be closed or moved across monitors without interfering with the strategy dashboard.

## Menu mode vs `--viewer`

Drop the `--viewer` flag and the arcade opens `MenuView` first:

```bash
python -m src.arcade.main --strategy
```

`MenuView` is a pure-keyboard navigator (Arrow keys + Enter, Escape to go back). It lists years (2023–2025), rounds per year, drivers, and teams. The `--viewer` shortcut exists for regression testing and for the "I know what I want" path.

## Single-driver vs two-driver mode

Omit `--driver2` and only the main driver renders on track and in the telemetry charts. The Delta trace in the telemetry window stays empty (no rival to compare against) and the Situation card focuses solely on safety-car probability.

Pass `--driver2 LEC` and:

- The rival icon appears on the track in the rival-team colour.
- The Delta plot renders the rolling gap between our driver and the rival.
- The Situation card gains the overtake-probability gauge driven by the N27 LightGBM model.
- The Pit card's undercut probability is computed against the rival.

## Playback controls

Hotkeys handled by `F1ArcadeView.on_key_press`:

- `Space`, pause / resume
- `Right Arrow`, step one lap forward
- `Left Arrow`, step one lap backward
- `+` / `-`, increase / decrease playback speed
- `Escape`, return to the menu (or exit if launched with `--viewer`)

## Known limitations

- **First-lap warmup**: the orchestrator runs cold for the first ~15 seconds while agent models load.
- **Cold FastF1 cache**: the first time a given round is requested, FastF1 downloads the session. Expect roughly 30 seconds on cold cache.
- **Port 9998**: the TCP broadcaster binds `127.0.0.1:9998`. If another process holds the port, the dashboard cannot connect.
- **Strategy mode requires year 2025**: the multi-agent pipeline only ships with 2025-season features. Running `--strategy` against 2023 or 2024 falls back to arcade-only replay.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: data/raw/.../laps.parquet` | FastF1 cache not built | Run once without `--strategy`, or `python -m src.simulation.builder 2025 3` |
| "Backend offline" / "Connection refused" | Arcade did not start the stream server | Restart the arcade with `--strategy` |
| "No module named pyqtgraph" | `uv sync` did not complete | Re-run `uv sync` from the repo root |
| "OpenAI api_key missing" | `.env` missing or `OPENAI_API_KEY` unset | Add the key to `.env`, or `F1_LLM_PROVIDER=lmstudio` |
| Dashboard renders but charts stay empty | First broadcast carries only the arcade frame | Per-agent outputs arrive on the second broadcast |

## Related reading

- [Arcade dashboard](#/arcade-dashboard), developer-level architecture deep dive on the dashboard package.
- [Arcade strategy pipeline](#/arcade-strategy-pipeline), why the arcade keeps its own copy of the orchestrator body.
- [Multi-agent system](#/multi-agent): N25–N31 multi-agent pipeline reference.
