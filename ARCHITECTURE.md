# Architecture

One-page map linking the detail docs. Read this first, then descend
into the deep dives as needed.

## Three user-facing surfaces, one shared core

- **CLI** (`f1-sim`) — headless Rich-based live inference panel.
- **Arcade** (`f1-arcade`) — 2D race replay + PySide6 strategy dashboard
  + PySide6 telemetry window (one command spawns all three).
- **Web app** (`f1-webapp`, a wrapper around `docker compose up`) —
  post-race analysis and chat (React SPA backed by FastAPI).

All three consume the same core:

- `src/agents/` — N25-N31 multi-agent stack (pace, tire, race situation,
  pit strategy, radio NLP, RAG regulations, orchestrator).
- `src/strategy/inference/engine.py::run_lap` — the single shared per-lap
  pipeline call that the CLI, Arcade, and backend all route through
  (profiles: `rich` for the full LLM-synthesis path, `no-llm` for the
  deterministic zero-LLM-client path). Replaces three hand-mirrored
  copies of the orchestrator sequence that used to drift out of sync.
- `src/simulation/` — `RaceReplayEngine` + `RaceStateManager`.
- `data/processed/laps_featured_<year>.parquet` + `data/raw/<year>/<Location>/` +
  `data/tire_compounds_by_race.json`.

The Streamlit path also runs a FastAPI backend (`src/telemetry/backend/`).
The Arcade path calls `run_lap` in-process without going through the
backend (see [`docs/pages/arcade-strategy-pipeline.md`](docs/pages/arcade-strategy-pipeline.md)).

## Multi-agent pipeline

N25 Pace · N26 Tire · N27 Situation · N29 Radio are always-on. N28 Pit
Strategy and N30 RAG are conditional (routing decides per-lap). N31
Orchestrator fuses all outputs through a Monte Carlo simulation and an
LLM synthesis pass into a `StrategyRecommendation`. Full flow:
[`docs/pages/architecture.md`](docs/pages/architecture.md).

## Arcade three-window topology

`f1-arcade --strategy` spawns:

1. The pyglet replay window (this process).
2. One PySide6 subprocess hosting **two** Qt windows in a shared
   `QApplication` event loop: `MainWindow` (strategy dashboard) and
   `TelemetryWindow` (2×2 circuit-comparison grid).

Both windows subscribe to the arcade's `TelemetryStreamServer` on
`127.0.0.1:9998`. Details:
[`docs/pages/arcade-dashboard.md`](docs/pages/arcade-dashboard.md).

## Data flow

- First run downloads the canonical data tree from Hugging Face
  (`VforVitorio/f1-strategy-dataset`) via
  `src/f1_strat_manager/data_cache.py::ensure_setup()`.
- FastF1 session cache lives under `data/cache/fastf1/` (local only,
  gitignored).
- Featured laps parquets + per-race raw dirs + tire-compound-by-race
  map form the input to the multi-agent stack.

See [`docs/pages/simulation.md`](docs/pages/simulation.md) for the wire-level view.

## Where to go next

- **Install:** [`INSTALL.md`](INSTALL.md).
- **Roadmap:** [`ROADMAP.md`](ROADMAP.md).
- **Agents reference:** [`docs/pages/agents-api.md`](docs/pages/agents-api.md).
- **Backend API:** [`docs/pages/backend-api.md`](docs/pages/backend-api.md).
- **Streamlit frontend:** [`docs/pages/streamlit.md`](docs/pages/streamlit.md).
- **Simulation engine:** [`docs/pages/simulation.md`](docs/pages/simulation.md).
- **All draw.io diagrams:** [`documents/dev_docs/diagrams/`](documents/dev_docs/diagrams/).
