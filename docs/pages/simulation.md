# src/simulation: Race Replay Engine

## Purpose

Offline replay of a race from a stored parquet snapshot. Emits `lap_state` dicts, the canonical data contract consumed by all seven strategy agents.

This is the **demo path** for the thesis defence. The live path (Kafka consumer feeding real telemetry) will replace the iterator in v0.14+ without touching any agent code, because agents only see `lap_state` dicts regardless of source.

## Architecture

```
data/raw/2025/<GP>/
  laps.parquet        ← loaded by RaceReplayEngine
  weather.parquet     ← loaded if present
  metadata.json       ← gp_name, year

RaceReplayEngine
  └── RaceStateManager          ← data boundary enforcement
        ├── get_driver_state()  ← full telemetry (our car)
        ├── get_rival_states()  ← timing-screen only (rivals)
        ├── get_weather_state() ← track + weather snapshot
        └── get_lap_state()     ← merges all into lap_state dict
              ↓
    lap_state dict → all 7 agents → strategy orchestrator
```

`RaceReplayEngine.to_arcade_frame()` still exists in `replay_engine.py`, but nothing calls it and its docstring's `/ws/replay` WebSocket route was never registered on the backend, the arcade's real live path is the direct in-process pipeline broadcasting over a local TCP socket, documented in [Arcade strategy pipeline](#/arcade-strategy-pipeline) and [Multi-agent system → Three-window arcade](#/multi-agent).

## Data boundary (architectural constraint)

The single most important design decision in this module.

| Field | Our driver | Rivals |
|---|---|---|
| LapTime | yes | yes |
| Sector1/2/3 | yes | NO |
| SpeedI1, SpeedI2, SpeedFL | yes | NO |
| SpeedST | yes | yes |
| TyreLife | yes | yes (broadcast) |
| Compound | yes | yes |
| FuelLoad | yes (estimated) | NO |
| Position | yes | yes |
| gap_to_leader | yes | yes |
| interval_to_driver | yes | yes |

Rivals get only what appears on the FIA live timing screen. This mirrors the real information asymmetry a strategy engineer faces on the pit wall.

### Who counts as a rival

The boundary above says *what* a rival's row contains. This says *which cars get a row at all*, and it is just as load-bearing:

**A car appears in `rivals` for a lap only if it was classified on that lap.** Retired, crashed, or not yet across the line means no row, so the car is absent from the list entirely. It is never present with an invented position, a placeholder gap or a default lap time.

This is the rule that a real timing screen follows, and skipping it produces decisions that look reasonable and are absurd. The pit wall once recommended undercutting HUL at Lusail lap 7. HUL had just crashed on lap 6, which is what brought out the safety car the same recommendation was reacting to. The car was gone; only a filled-in default kept it on the list.

Two habits follow from it, and both are worth copying into any new code that shapes race data:

- **Absence beats a default.** If a value is unknown, leave it `None` and let the consumer filter. A number is a claim, and an invented one is a false claim that nothing downstream can distinguish from a reading.
- **Never default to a value the code also searches by.** `Position` defaulting to `0` is how the leader found "the car ahead at position `pos - 1 == 0`" and got handed the car that had just crashed. Placeholders belong in sort keys, never in emitted data.

## Gap computation

Gaps come from `session_time_s`, the session elapsed time at the end of each lap, read from FastF1's `Time` column. This is the same value FastF1 uses internally for gaps in `session.laps`.

```
gap_to_leader[driver, lap]      = session_time[driver, lap] - session_time[leader, lap]
interval_to_driver[rival, lap]  = session_time[rival, lap]  - session_time[our_driver, lap]
```

**Not** cumulative `LapTime` sums, which is the tempting alternative. Summed lap times drift away from the real on-track gap wherever timing corrections or safety-car bunching apply, which is exactly where a strategy call matters most. `_compute_session_times` in `src/simulation/race_state_manager.py` takes elapsed time for that reason.

This is worth stating loudly because the feature-engineering step drops the `Time` column, so any consumer reading the featured parquet directly used to fall back to lap-time deltas without noticing. Measured on Lusail 2025 lap 20, that fallback fed the model a **0.10 s** gap between PIA and NOR when the real one was **3.58 s**: the difference between "inside the DRS window" and "not close". The loader now restores `Time_s` from the raw per-race parquet at load time, so both paths agree.

Known limitations:

- Lapped cars show a large positive gap rather than a "lapped" flag.

## Files

| File | Responsibility |
|---|---|
| `race_state_manager.py` | Data boundary, per-lap state construction |
| `replay_engine.py` | Parquet loading, lap iterator, Arcade frame builder |
| `__main__.py` | Terminal CLI for quick testing |

## Running

```bash
# All laps, no delay
python -m src.simulation Melbourne NOR McLaren

# Specific lap range
python -m src.simulation Monaco HAM Mercedes --laps 30-50

# 2s between laps (simulates real-time ingestion)
python -m src.simulation Melbourne NOR McLaren --interval 2

# Different season
python -m src.simulation Silverstone VER "Red Bull Racing" --data-dir data/raw/2024
```

### Example output

```
------------------------------------------------------------------------
  F1 Strategy - Race Replay   Melbourne  |  NOR / McLaren  |  57 laps
------------------------------------------------------------------------
   Lap  Pos      Compound   LapTime  Gap Leader                     Ahead                    Behind
------------------------------------------------------------------------
     1    1  INT( 1L)  1:57.099    +0.000s                            P2:VER INT( 1L)  +2.293s
     2    1  INT( 2L)    ---.-     +0.000s                            P2:VER INT( 2L)  +4.586s [IN]
    20    1  INT(20L)  1:30.710    +0.000s                            P2:PIA INT(20L) +13.836s
    34    1  INT(34L)  2:02.273    +0.000s                            P2:PIA INT(34L) +19.398s [IN]
    35    1  HAR( 2L)  2:03.448    +0.000s                            P2:PIA HAR( 2L) +19.731s [OUT]
    44    6  HAR(11L)  1:45.587   +34.252s  P5:LEC HAR(11L) +56.481s  P7:LAW MED(11L) +79.194s [IN]
    46    3  INT( 2L)  1:31.567   -62.788s  P2:LEC HAR(13L) +56.205s  P4:TSU MED(13L) +62.387s
    57    1  INT(13L)  1:27.126    +0.000s                            P2:VER INT(11L) -18.289s
------------------------------------------------------------------------
  Replay complete - 57 laps shown.
```

**Reading the output:**

| Column | Meaning |
|---|---|
| `Lap` | Lap number |
| `Pos` | Our driver's position |
| `Compound` | Tyre compound abbreviation + laps on tyre: `INT(20L)` = Intermediate, 20 laps; `HAR(2L)` = Hard, 2 laps; `MED` = Medium; `SOF` = Soft |
| `LapTime` | Lap time in M:SS.mmm. `---.-` = deleted lap (red flag, pit-in lap, etc.) |
| `Gap Leader` | Our gap to P1 in seconds. `+0.000` = we are the leader |
| `Ahead / Behind` | Rival directly ahead/behind in position |
| `[IN]` | Driver pitted this lap (in-lap) |
| `[OUT]` | Driver exiting the pits (out-lap) |

## `lap_state` schema

```python
{
    "lap_number": int,
    "driver": {
        # Identity, repeated inside the sub-dict so a consumer that has been
        # handed only this slice still knows whose lap it is holding.
        "driver": str,
        "team": str,
        "lap_number": int,
        "lap_time_s": float | None,
        # This lap's Prev_LapTime feature, never last lap's own lap_time_s
        # reused as a stand-in — that self-reference used to feed the pace
        # agent its own most recent prediction back in as "previous" (#435).
        "prev_lap_time": float | None,
        "sector1_s": float | None,
        "sector2_s": float | None,
        "sector3_s": float | None,
        "position": int | None,
        "gap_to_leader_s": float | None,
        "compound": str,
        "compound_id": int | None,
        "tyre_life": int | None,
        "stint": int | None,
        # TyreLife at the first lap of the current stint. N06's FuelEffect is
        # measured from this baseline, not from lap 1 of the race (#446).
        "stint_baseline_tyre_life": int | None,
        "fresh_tyre": bool,
        "speed_i1": float | None,
        "speed_i2": float | None,
        "speed_fl": float | None,
        "speed_st": float | None,
        "fuel_load": float | None,
        "track_status": str,
        "is_in_lap": bool,
        "is_out_lap": bool,
    },
    # Only cars classified on this lap. A car that has retired, crashed or has
    # not yet completed the lap has no row for it, so it is simply ABSENT from
    # `rivals` — never present with a filled-in default. See "Who counts as a
    # rival" above.
    "rivals": [
        {
            "driver": str,
            "team": str,
            # None when the car has no position that lap. It sorts to the back
            # through a placeholder confined to the sort key, so no consumer can
            # look up a car at that position and find one.
            "position": int | None,
            "lap_time_s": float | None,
            "compound": str,
            "tyre_life": int | None,
            "stint": int | None,
            "speed_st": float | None,
            "gap_to_leader_s": float | None,
            "interval_to_driver_s": float | None,
            "is_pitting": bool,
        }
    ],
    "weather": {
        "track_status": str,
        "air_temp": float | None,
        "track_temp": float | None,
        # The session's FIRST track temperature, not this lap's — a session
        # constant carried in the weather dict because that is the channel
        # its consumer (N14's track_temp_delta feature) reads from (#486).
        "track_temp_start": float | None,
        "humidity": float | None,
        "wind_speed": float | None,
        "rainfall": bool,
    },
    "session_meta": {
        "gp_name": str,
        "year": int,
        "driver": str,
        "team": str,
        "total_laps": int,
    },
    # Art. 30.5(m) (2024-25 numbering; it was 30.5(n) in 2023) two-compound obligation, for our driver, as of this lap.
    # Emitted here (rather than fetched separately) so the CLI, the arcade and
    # the backend cannot each derive their own, divergent view of who still
    # owes a stop. A lap with no row falls back to the nearest earlier lap,
    # since compound history only grows and the last known state is still the
    # truth about what has been used.
    "stint_flags": {
        "stops_made": int | None,               # highest visible stint number minus one
        "compounds_used": list[str],             # first-seen order, dry and wet compounds
        "mandatory_stop_pending": bool | None,   # None when an unseen stint could hide the second compound
    },
    # mandatory_stop_pending for every rival present in `rivals`, keyed by
    # driver code. A rival who must still stop is no threat: they pay the same
    # price later. A driver absent from this dict has no row for this lap and
    # is not asserted either way, the same "absence beats a default" rule
    # `rivals` itself follows.
    "rival_stop_pending": {
        "<driver_code>": bool | None,
    },
}
```

## Future: Kafka integration (v0.14)

Replace `RaceReplayEngine.replay()` with a `LiveKafkaConsumer.consume_lap()` iterator that emits the same `lap_state` dict from a live Kafka topic. Zero changes to agents or orchestrator.

```python
# Current (offline)
for lap_state in engine.replay():
    ...

# Future (live)
for lap_state in kafka_consumer.consume_lap():
    ...
```
