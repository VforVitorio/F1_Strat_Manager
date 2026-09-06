"""
Race replay engine: loads a race directory and emits lap_state dicts one
per lap at a configurable interval.

This is the offline demo path for the strategy system. Usage::

    engine = RaceReplayEngine("data/raw/2025/Melbourne", "NOR", "McLaren")
    for lap_state in engine.replay():
        recommendation = run_strategy_orchestrator_from_state(lap_state, laps_df)
        frame = engine.to_arcade_frame(lap_state, recommendation)
        await ws.send_json(frame)   # or process locally

Kafka replacement (v0.14+): swap ``engine.replay()`` with a
``LiveKafkaConsumer.consume_lap()`` iterator. Every downstream component
(agents → orchestrator → to_arcade_frame) stays unchanged because they
all consume the same ``lap_state`` dict contract.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from src.simulation.data_validation import DataValidationError, warn_low_quality_laps
from src.simulation.race_state_manager import RaceStateManager


class RaceReplayEngine:
    """Iterates through a race lap by lap, emitting ``lap_state`` dicts.

    Loads ``laps.parquet`` and optionally ``weather.parquet`` from a race
    directory, sets up a ``RaceStateManager``, and yields one ``lap_state``
    dict per lap with an optional sleep between emissions to simulate
    real-time data ingestion.

    Attributes:
        rsm:         The underlying ``RaceStateManager``.
        total_laps:  Number of laps in the race.
        interval:    Seconds to sleep between lap emissions, 0 for batch and
                     the default. Only the interactive surfaces raise it.
    """

    def __init__(
        self,
        race_dir: str | Path,
        driver_code: str,
        team: str,
        interval_seconds: float = 0.0,
    ) -> None:
        """Load race data and initialise the state manager.

        Args:
            race_dir:         Path to the race directory containing
                              ``laps.parquet`` (and optionally
                              ``weather.parquet``, ``metadata.json``).
            driver_code:      Our driver's FIA three-letter code (e.g.
                              ``"NOR"``).
            team:             Our driver's team name, must match the Team
                              column in the laps parquet exactly.
            interval_seconds: Seconds to sleep between lap emissions. Defaults
                              to no sleep, so a caller that does not ask for
                              pacing runs at full speed. The interactive
                              surfaces ask for it explicitly: both CLIs pass
                              ``args.interval`` and the arcade passes
                              ``StrategyRequest.interval_s``. A caller that
                              forgets the argument pays nothing, which is the
                              direction that fails safe (#1202).
        """
        race_dir = Path(race_dir)

        # Guard the read so a missing/corrupt parquet fails with a sourced error
        # instead of a raw pyarrow/OS traceback; the required-column contract is
        # then enforced in RaceStateManager.
        laps_path = race_dir / "laps.parquet"
        try:
            laps_df = pd.read_parquet(laps_path)
        except Exception as exc:  # noqa: BLE001 - any read failure is a load-boundary error
            raise DataValidationError(f"{laps_path}: cannot read laps parquet ({exc}).") from exc
        warn_low_quality_laps(laps_df, source=str(laps_path))

        self._weather_df: pd.DataFrame | None = None
        weather_path = race_dir / "weather.parquet"
        if weather_path.exists():
            try:
                self._weather_df = pd.read_parquet(weather_path)
            except Exception as exc:  # noqa: BLE001 - weather is optional; degrade, don't abort
                print(
                    f"[warn] {weather_path}: cannot read weather parquet ({exc}); "
                    "continuing without weather.",
                    file=sys.stderr,
                )

        gp_name, year = self._parse_meta(race_dir)

        self.rsm = RaceStateManager(
            laps_df=laps_df,
            driver_code=driver_code,
            team=team,
            gp_name=gp_name,
            year=year,
        )
        self.total_laps: int = self.rsm.total_laps
        self.interval: float = interval_seconds

    @staticmethod
    def _parse_meta(race_dir: Path) -> tuple[str, int]:
        """Extract ``gp_name`` and ``year`` from ``metadata.json`` if present.

        Falls back to the directory name and 2025 when the file is absent
        so the engine works with any race directory structure.
        """
        meta_path = race_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            return meta.get("gp_name", race_dir.name), int(meta.get("year", 2025))
        # No metadata.json: warn loudly instead of silently assuming 2025 - a
        # wrong year mis-selects circuit-aware thresholds and season config.
        print(
            f"[warn] {meta_path} absent; falling back to gp_name={race_dir.name!r}, "
            "year=2025 (circuit thresholds + season may be wrong).",
            file=sys.stderr,
        )
        return race_dir.name, 2025

    def replay(self) -> Iterator[dict[str, Any]]:
        """Yield ``lap_state`` dicts one per lap, sleeping ``interval`` between.

        The yielded dict is the canonical ``lap_state`` format consumed by all
        agents (see ``RaceStateManager.get_lap_state`` for the full schema).
        When ``interval_seconds=0`` the generator runs as fast as the CPU
        allows, which is useful for batch evaluation or unit tests.

        Yields:
            ``lap_state`` dict for laps 1 through ``total_laps`` inclusive.
        """
        for lap in range(1, self.total_laps + 1):
            yield self.rsm.get_lap_state(lap, self._weather_df)
            if self.interval > 0:
                time.sleep(self.interval)

    def to_arcade_frame(
        self,
        lap_state: dict[str, Any],
        recommendation: Any | None = None,
        agent_outputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert a ``lap_state`` + recommendation into an Arcade WebSocket frame.

        Assembles a WebSocket-shaped frame for a ``/ws/replay`` endpoint that
        does not exist in this repository. The Arcade UI consumes the raw TCP
        broadcast built by ``src/arcade/stream.py`` instead, and no caller in
        this repository invokes this method. Our driver is always the first
        entry in ``cars`` with ``is_our_driver: true``; rivals follow sorted
        by position.

        Args:
            lap_state:      Dict from ``RaceStateManager.get_lap_state()``.
            recommendation: Optional ``StrategyRecommendation`` dataclass. When
                            ``None`` the ``recommendation`` key is omitted from
                            the frame (frontend shows "waiting for data").
            agent_outputs:  Optional per-agent raw outputs for the Arcade debug
                            panel. Expected keys: ``pace``, ``tire``,
                            ``sc_prob``, ``overtake_prob``.

        Returns:
            JSON-serialisable dict ready for WebSocket ``send_json()``.
        """
        driver = lap_state.get("driver", {})
        rivals = lap_state.get("rivals", [])

        our_car: dict[str, Any] = {
            "driver": driver.get("driver", ""),
            "team": driver.get("team", ""),
            "position": driver.get("position"),
            "compound": driver.get("compound", ""),
            "tyre_life": driver.get("tyre_life"),
            "gap_to_leader": driver.get("gap_to_leader_s"),
            "interval": 0.0,
            "is_our_driver": True,
        }

        rival_cars: list[dict[str, Any]] = [
            {
                "driver": r.get("driver", ""),
                "team": r.get("team", ""),
                "position": r.get("position"),
                "compound": r.get("compound", ""),
                "tyre_life": r.get("tyre_life"),
                "gap_to_leader": r.get("gap_to_leader_s"),
                "interval": r.get("interval_to_driver_s"),
                "is_our_driver": False,
            }
            for r in rivals
        ]

        cars = sorted([our_car] + rival_cars, key=lambda c: c["position"] or 99)

        frame: dict[str, Any] = {
            "lap": lap_state["lap_number"],
            "gp": lap_state.get("session_meta", {}).get("gp_name", ""),
            "total_laps": lap_state.get("session_meta", {}).get("total_laps"),
            "cars": cars,
            "weather": lap_state.get("weather", {}),
        }

        if recommendation is not None:
            frame["recommendation"] = {
                "action": getattr(recommendation, "action", ""),
                "rationale": getattr(recommendation, "rationale", ""),
                "confidence": getattr(recommendation, "confidence", 0.0),
                "tyre": getattr(recommendation, "tyre", ""),
                "laps_remaining_on_tyre": getattr(recommendation, "laps_remaining_on_tyre", None),
                "risk_flags": getattr(recommendation, "risk_flags", []),
            }

        if agent_outputs:
            frame["agent_outputs"] = agent_outputs

        return frame
