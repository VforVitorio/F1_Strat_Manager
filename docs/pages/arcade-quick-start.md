# Arcade Quick Start

> End-user guide for running the F1 StratLab arcade replay with the live strategy surfaces. Aimed at someone who has cloned the repository and wants everything on screen inside ten minutes.

One command launches everything. The arcade process owns the simulation loop and broadcasts merged state on a local TCP port; the follower surfaces subscribe and render.

**One follower stack.** PITWALL (two pywebview windows rendering React) is the whole strategy surface. The original PySide6 pair ran beside it through the migration so every PITWALL panel could be compared against the window it replaced while that window still existed; it has now been retired, along with the `PySide6` and `pyqtgraph` dependencies. The comparison baseline is not lost with it: the Qt windows' rendered output is committed as screenshots under `documents/dev_docs/migration/pitwall/`. The pyglet replay is unchanged either way.

<p align="center">
  <video src="/assets/demo/arcade-demo.mp4" poster="/assets/demo/arcade-demo-poster.jpg" width="760" autoplay loop muted playsinline preload="metadata" aria-label="F1 StratLab arcade replay in action"></video>
  <br/>
  <sub>The arcade: 2D replay, strategy dashboard and live telemetry, all in sync. Recorded before PITWALL joined; the pyglet replay on the left is unchanged.</sub>
</p>

## Prerequisites

- **Python**: 3.10 or newer. The project pins dependencies with `uv`.
- **Dependencies**: run `uv sync` from the repo root. The lockfile pulls `arcade`, `pywebview`, `fastf1`, `langchain-openai`, the model stack (`xgboost`, `lightgbm`, `torch`), and the NLP stack (`transformers`, `sentence-transformers`, `setfit`). No manual install steps required beyond `uv sync`.
- **LLM credentials**: either set `OPENAI_API_KEY` in a repo-root `.env` (the canonical TFG setup) or run LM Studio locally on `http://localhost:1234/v1` and pass `--provider lmstudio` on the command line (the arcade's own flag; its default is `openai`, independent of the `F1_LLM_PROVIDER` env var the backend and CLI read). Only the wording of the orchestrator's reasoning changes.
- **Race data cache**: the replay reads `data/raw/{year}/{Location}/laps.parquet` and optionally `weather.parquet`. The parquet files are produced by FastF1 on first run. Expect a 20-40 second delay on the first launch of a round.
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
4. With `--strategy` set, the view starts a `TelemetryStreamServer` on `127.0.0.1:9998`, owns a `StrategyState`, and spawns `python -m src.pitwall`.
5. PITWALL opens **PITWALL · DATA** and **PITWALL · AGENTS**, two pywebview windows rendering React against the same broadcast, through a single shared TCP client.

The arcade window drives playback; every other window reacts to broadcasts. PITWALL additionally serves the same two pages over loopback, and prints the URL on startup, so they can be opened in a browser (and get devtools) instead of, or as well as, the windows.

## What each window shows

### Arcade replay (pyglet)

Track outline with the DRS zones drawn in green, two driver icons (our driver in team colour, rival in rival-team colour), a leaderboard on the right with compound pills and gaps, a weather panel, a driver-info strip, and a progress bar with the current lap.

### PITWALL · AGENTS

Orchestrator card on the left (action badge, confidence bar, pace/risk chips, plan strip with compound pill, and a guardrail line that shows when the no-LLM hard guard overrode the LLM pick), scenario bars beneath it (four bars for STAY_OUT / PIT_NOW / UNDERCUT / OVERCUT) and reasoning tabs below that. On the right, a 3x2 grid of sub-agent cards: Pace (N25), Tire (N26), Situation (N27), Radio (N29), Pit (N28, dimmed when inactive), RAG (N30, dimmed when inactive). Pace and Tire carry embedded ECharts plots.

### PITWALL · DATA

A full-width status strip over two columns. Left: the twenty-row timing tower and the session bests. Right: a tab strip over **TRACES** (the own-car 2x2 telemetry grid, the track ring and the radio / RCM feed), **RACE PACE** (a lap-by-lap grid coloured by each lap's own ranking) and **RACE TRACE** (accumulated time against the leader, the field average or our own car).

## Menu mode vs `--viewer`

Drop the `--viewer` flag and the arcade opens `MenuView` first:

```bash
python -m src.arcade.main --strategy
```

`MenuView` is a pure-keyboard navigator (Arrow keys + Enter, Escape to go back). It lists years (2023-2025), rounds per year, drivers, and teams. The `--viewer` shortcut exists for regression testing and for the "I know what I want" path.

## Single-driver vs two-driver mode

Omit `--driver2` and only the main driver renders on track and in the telemetry charts. The Delta trace in the telemetry window stays empty (no rival to compare against) and the Situation card focuses solely on safety-car probability.

Pass `--driver2 LEC` and:

- The rival icon appears on the track in the rival-team colour.
- The Delta plot renders the rolling gap between our driver and the rival.
- The Situation card gains the overtake-probability gauge driven by the N27 LightGBM model.
- The Pit card's undercut probability is computed against the rival.

## Playback controls

Hotkeys handled by `F1ArcadeView.on_key_press`:

| Key | What it does |
|---|---|
| `Space` | pause / resume |
| `Left` / `Right` | **hold** to scrub backwards or forwards. Holding pauses playback; releasing restores whatever state was active before. Not a one-lap step |
| `Up` / `Down` | next / previous playback speed |
| `1` `2` `3` `4` | jump straight to 0.5x, 1x, 2x, 4x |
| `R` | restart: back to frame zero, default speed, playing |
| `D` | show / hide the DRS zones on the track |
| `B` | show / hide the progress bar |
| `A` | show / hide the eighteen non-featured cars |
| `Escape` | close the window |

`Escape` quits rather than returning to the menu, and it does so whether or not
`--viewer` was used at launch.

## Known limitations

- **First-lap warmup**: the orchestrator runs cold for the first ~15 seconds while agent models load.
- **Cold FastF1 cache**: the first time a given round is requested, FastF1 downloads the session. Expect roughly 30 seconds on cold cache.
- **Port 9998**: the TCP broadcaster binds `127.0.0.1:9998`. If another process holds the port, the dashboard cannot connect.
- **Strategy mode requires year 2025**: the multi-agent pipeline only ships with 2025-season features. Running `--strategy` against 2023 or 2024 falls back to arcade-only replay.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: data/raw/.../laps.parquet` | the race is not in the local cache | `uv run python -c "from src.f1_strat_manager.data_cache import ensure_race; ensure_race(2025, 'Melbourne')"`, or just pick the race from the `f1-strat` menu, which calls the same function |
| "Backend offline" / "Connection refused" | Arcade did not start the stream server | Restart the arcade with `--strategy` |
| "OpenAI api_key missing" | `.env` missing or `OPENAI_API_KEY` unset | Add the key to `.env`, or `F1_LLM_PROVIDER=lmstudio` |
| PITWALL renders but charts stay empty | First broadcast carries only the arcade frame | Per-agent outputs arrive on the second broadcast |
| `PITWALL needs its UI bundle` on startup | the React bundle was never built | `npm ci && npm run build` in `src/pitwall/ui` |

## Related reading

- [PITWALL windows](#/pitwall), the two React surfaces and the client they share.
- [Arcade dashboard (legacy)](#/arcade-dashboard), the retired PySide6 package PITWALL replaced.
- [Arcade strategy pipeline](#/arcade-strategy-pipeline), why the arcade delegates to the shared engine instead of keeping its own copy of the orchestrator.
- [Multi-agent system](#/multi-agent): N25-N31 multi-agent pipeline reference.
