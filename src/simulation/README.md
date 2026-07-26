# src/simulation — Race replay engine

> **Canonical narrative doc:** [`docs/pages/simulation.md`](../../docs/pages/simulation.md)
> covers the `lap_state` contract and the data boundary with the agents.
> This README is the package-level API pointer.

Offline replay path for the multi-agent strategy system. Loads a race
parquet from disk, walks it lap by lap, and emits a `lap_state` dict per
lap that the agents and the orchestrator can consume directly. The same
contract is the planned drop-in for a future Kafka live-ingestion path
(v0.14+) — every downstream component (agents → orchestrator → Arcade
frame) reads `lap_state` dicts and does not care whether they came from a
parquet or a live topic.

The `single-driver data boundary` enforced here is the critical
architectural constraint: agents see full telemetry for *our* driver but
timing-screen-equivalent fields (position, gap to leader, interval to our car, compound, tyre life, stint, trap speed and whether they are in the pit lane)
for rivals, mirroring what a real team strategy wall observes during a
race.

---

## Files

| File | Description |
|---|---|
| [`race_state_manager.py`](race_state_manager.py) | `RaceStateManager` class — owns the per-driver lap-state computation and enforces the data boundary. Reads `LapTime`, `Sector1/2/3Time`, `TyreLife`, `Compound`, `Stint`, `Position`, `SpeedI1/I2/FL/ST`, `FuelLoad` for our driver and only `Position` / `LapTime` / `Compound` / `TyreLife` / `gap_to_leader_s` / `interval_to_driver_s` for rivals. Gap computation uses the FastF1 `Time` column (session elapsed time) so safety-car bunching does not skew the on-track gap. Also emits `stint_flags` (Art. 30.5(m) (2024-25 numbering; it was 30.5(n) in 2023) mandatory-stop state for our driver) and `rival_stop_pending` dict per rival |
| [`stint_history.py`](stint_history.py) | Pure helpers that read a GP-scoped laps frame and answer questions about one driver's stint history: how many stops made, which compounds used, whether the mandatory two-dry-compound obligation is pending. Used by `race_state_manager.py` and decision-layer MC projection |
| [`replay_engine.py`](replay_engine.py) | `RaceReplayEngine` class — loads `laps.parquet` (and optionally `weather.parquet`) from a race directory, sets up an `RaceStateManager`, and yields one `lap_state` dict per lap with an optional `interval_seconds` sleep so a demo can run in real time or as fast as possible |
| [`__main__.py`](__main__.py) | CLI entry point — `python -m src.simulation <gp_name> <driver> <team> [--interval N] [--laps N-M]`. Loads the race directory under `data/raw/2025/`, applies a small `_GP_FOLDER_ALIASES` map for folder names that differ from the canonical key (`Miami_Gardens` → `Miami`, `Mexico_City` → `Mexico City`, …), and prints a per-lap summary |
| `__init__.py` | Empty package marker |

---

## Usage

```python
from src.simulation.replay_engine import RaceReplayEngine
from src.strategy.inference.engine import run_lap

engine = RaceReplayEngine("data/raw/2025/Melbourne", "NOR", "McLaren")
for lap_state in engine.replay():
    rec, agent_outputs, _ = run_lap(race_state, laps_df, lap_state, profile="no-llm")
    # rec is a StrategyRecommendation; agent_outputs carries each sub-agent's payload
```

`to_arcade_frame` also exists on the engine, but nothing calls it and the
`/ws/replay` route its docstring names was never registered. The arcade's real
path is the in-process pipeline broadcasting over a local TCP socket.

```bash
# CLI replay (no agents, just iterates and prints)
python -m src.simulation Melbourne NOR McLaren
python -m src.simulation Monaco HAM Mercedes --interval 2
python -m src.simulation Monza LEC Ferrari --laps 10-30
python -m src.simulation Silverstone VER "Red Bull Racing" --interval 0
```

---

## Consumers of `RaceReplayEngine`

Three entry points drive the replay engine today:

- **CLI** — `scripts/run_simulation_cli.py` uses `RaceStateManager` directly (without
  the `RaceReplayEngine` wrapper) so it can interleave the radio runner, the strategy
  orchestrator, and the Rich inference panel within a single Live render loop. The
  production path that ships with the R1 release.
- **FastAPI backend SSE** — `src/telemetry/backend/services/simulation/` wraps
  `RaceReplayEngine` inside the `simulate_race` async generator consumed by the
  `POST /api/v1/strategy/simulate` SSE endpoint. Feeds the React web app and the
  TestClient smoke tests.
- **Arcade** — `src/arcade/strategy.py::SimConnector` drives `RaceReplayEngine.replay()`
  locally inside the arcade subprocess and feeds the arcade's local strategy pipeline
  (`src/arcade/strategy_pipeline.py`). No FastAPI involved; the arcade broadcasts the
  merged state over TCP 127.0.0.1:9998 to the PySide6 dashboard.

---

## Future Kafka swap

Substituting the offline replay for live ingestion is a one-line change:

```python
# Offline (today)
for lap_state in engine.replay(): ...

# Live (v0.14+)
for lap_state in LiveKafkaConsumer.consume_lap(): ...
```

Every consumer downstream of the iterator already speaks the `lap_state`
dict contract, so the agents and the orchestrator do not need to change.
