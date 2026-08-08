"""Dev-only broadcast producer with a POPULATED strategy block.

Drives the real `TelemetryStreamServer` from the real Melbourne 2025 session
so PITWALL and the Qt dashboard can be developed against a full payload
without launching the arcade, running the agent pipeline, or spending a
single LLM call. Built for the sprint-3 AGENTS port, where an empty
strategy block leaves every card in its idle state and there is nothing to
compare against the window being replaced.

    python scripts/dev_pitwall_producer.py 180      # seconds to broadcast

Then, in another terminal, `python -m src.pitwall` or
`python -m src.arcade.dashboard`.

Not part of the product: it fabricates the decisions it publishes.

"""

from __future__ import annotations

import sys
import time
from functools import partial
from types import SimpleNamespace

from src.arcade.app import F1ArcadeView
from src.arcade.config import FPS, STREAM_HOST, STREAM_PORT
from src.arcade.data import SessionLoader
from src.arcade.strategy import (
    LapDecisionDTO,
    PerAgentOutputsDTO,
    StartEventDTO,
    StrategyState,
)
from src.arcade.stream import TelemetryStreamServer

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
ON_UPDATE_HZ = 60.0


def decision(lap: int, action: str, confidence: float) -> LapDecisionDTO:
    return LapDecisionDTO(
        lap_number=lap,
        compound="MEDIUM",
        tyre_life=lap - 8,
        position=4,
        lap_time_s=81.234 + (lap % 5) * 0.11,
        gap_ahead_s=1.42,
        action=action,
        confidence=confidence,
        reasoning=(
            "Tyre degradation is inside the predicted envelope but the delta to PIA has "
            "closed to 1.4 s over three laps. The undercut window against RUS opens now "
            "and shuts in two laps once he clears traffic. Pit duration sits at 22.4 s "
            "against a 21.8 s median, so the stop itself is not the risk; the rejoin is."
        ),
        scenario_scores={"PIT_NOW": 0.71, "STAY_OUT": 0.29, "EXTEND": 0.44, "UNDERCUT": 0.63},
        pace_mode="PUSH",
        risk_posture="AGGRESSIVE",
        pit_lap_target=lap + 1,
        compound_next="HARD",
        undercut_target="RUS",
        agent_alerts=["tyre cliff in 2 laps", "DRS window opens on the main straight"],
        guardrail_reason="none",
        per_agent=PerAgentOutputsDTO(
            pace={"predicted_lap_time_s": 81.0, "actual_lap_time_s": 81.3, "delta_s": 0.3},
            tire={"degradation_pct": 62.0, "cliff_lap": lap + 2, "ci_low": 55.0, "ci_high": 68.0},
            situation={"threat_level": "MEDIUM", "overtake_prob": 0.34, "sc_prob": 0.08},
            radio={"sentiment": "negative", "intent": "GRIP_COMPLAINT", "confidence": 0.81},
            pit={"pit_duration_s": 22.4, "p10": 21.1, "p90": 24.8},
            regulation_context="Art. 55.7 - drivers must respect the delta under Safety Car",
            rag={
                "question": "Does a Safety Car change the mandatory compound rule?",
                "answer": "No. Art. 30.5(m) still requires two specifications in a dry race.",
                "articles": ["30.5(m)", "55.7"],
                "chunks": ["...at least two specifications of dry-weather tyre..."],
            },
            active=["pace", "tire", "situation", "pit"],
        ),
        memory_block=f"lap {lap - 1}: STAY_OUT (0.58) - undercut window not yet open",
        plan_changed=True,
    )


session = SessionLoader().load(2025, 1, "Melbourne")
server = TelemetryStreamServer(STREAM_HOST, STREAM_PORT)
server.start()

state = StrategyState()
state.start = StartEventDTO(
    gp="Melbourne",
    year=2025,
    driver="NOR",
    driver2="PIA",
    team="McLaren",
    lap_start=1,
    lap_end=57,
    total_laps=57,
    no_llm=False,
    provider="openai",
)
state.latest = decision(23, "PIT_NOW", 0.71)
state.history = [
    decision(lap, "STAY_OUT" if lap % 4 else "EXTEND", 0.5 + (lap % 5) * 0.06)
    for lap in range(14, 23)
] + [state.latest]

view = SimpleNamespace(
    _session=session,
    _driver_main="NOR",
    _driver_rival="PIA",
    _year=2025,
    _stream_server=server,
    _strategy_state=state,
    _broadcast_tick=0,
    _broadcast_seq=0,
    _last_broadcast_idx=-1,
    _last_broadcast_clock=-1.0,
    _frame_index=60000.0,
    playback_speed=2.0,
    _is_paused=False,
)
view._build_arcade_snapshot = partial(F1ArcadeView._build_arcade_snapshot, view)

print(f"producing {SECONDS:.0f}s with a POPULATED strategy block", flush=True)
deadline = time.perf_counter() + SECONDS
while time.perf_counter() < deadline:
    view._frame_index += (1.0 / ON_UPDATE_HZ) * FPS * view.playback_speed
    F1ArcadeView._broadcast_if_due(view)
    time.sleep(1.0 / ON_UPDATE_HZ)
print("done", flush=True)
server.stop()
