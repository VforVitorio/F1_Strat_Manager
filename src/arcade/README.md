# `src/arcade/`, race replay + the broadcast its followers read

2D race replay (pyglet via the `arcade` library) plus the PITWALL
subprocess spawned from the same command. One invocation of `f1-arcade`
opens three top-level windows: the arcade replay, **PITWALL · AGENTS**
(orchestrator card + six sub-agent cards + reasoning tabs) and
**PITWALL · DATA** (status strip, timing tower, bests, own-car traces,
race pace and race trace). The PySide6 pair those two replaced was
retired; `src/pitwall/` is where the followers live now.

## Run

```bash
f1-arcade --viewer --year 2025 --round 3 --driver VER \
          --team "Red Bull Racing" --driver2 LEC --strategy
```

Or without the strategy pipeline (replay-only):

```bash
f1-arcade --viewer --year 2025 --round 3 --driver VER --team "Red Bull Racing"
```

## Public docs

- **End-user quick start:** [`docs/pages/arcade-quick-start.md`](../../docs/pages/arcade-quick-start.md)
- **Dashboard architecture (developer reference):** [`docs/pages/arcade-dashboard.md`](../../docs/pages/arcade-dashboard.md)
- **The shared `run_lap` engine the arcade delegates to:** [`docs/pages/arcade-strategy-pipeline.md`](../../docs/pages/arcade-strategy-pipeline.md)

## Layout

```
src/arcade/
├── main.py              # CLI entry point (f1-arcade)
├── app.py               # F1ArcadeView — pyglet replay loop, TCP broadcast
├── data.py              # SessionLoader + SessionData + FrameData
├── config.py            # Palette, GP calendars, constants
├── strategy.py          # SimConnector + StrategyState + DTOs
├── strategy_pipeline.py # Thin delegate over the shared engine (run_lap)
├── stream.py            # TelemetryStreamServer (stdlib TCP)
├── overlays.py          # WeatherPanel, LeaderboardPanel, DriverInfoPanel, …
├── track.py             # Track polyline renderer (DRS zones, cars)
├── views.py             # MenuView (interactive configurator)
```

The arcade **does not depend on the FastAPI backend at runtime**. The
strategy pipeline runs in a background thread inside this process using
`RaceReplayEngine` + the featured-laps parquet + the local
`strategy_pipeline.run_strategy_pipeline` wrapper. See
[`docs/pages/arcade-strategy-pipeline.md`](../../docs/pages/arcade-strategy-pipeline.md)
for the shared engine, its two profiles, and why the old duplicate is gone.
