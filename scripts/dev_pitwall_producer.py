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
            # Every key below is a real field of the agent's own output
            # dataclass, and `active` carries the real routing tokens. An
            # earlier version invented plausible names ("predicted_lap_time_s",
            # "degradation_pct", "sc_prob") and lower-cased the routing list,
            # so the tool built to stop the cards being idle rendered
            # "Dnext +0.000s", "deg - s/lap", "safety car 0%" and left both
            # conditional cards on their trigger hint. See #853.
            pace={
                "lap_time_pred": 81.0 + (lap % 5) * 0.11,
                "delta_vs_prev": -0.204,
                "delta_vs_median": 0.118,
                "ci_p10": 80.45,
                "ci_p90": 81.55,
                "reasoning": (
                    "Pace is holding inside the predicted envelope; the last three laps "
                    "sit within 0.1 s of the stint median."
                ),
            },
            tire={
                "compound": "MEDIUM",
                "current_tyre_life": lap - 8,
                "deg_rate": 0.031,
                "laps_to_cliff_p10": 4.0,
                "laps_to_cliff_p50": 6.0,
                "laps_to_cliff_p90": 9.0,
                "cumulative_deg_s": 0.42,
                "deg_cost_s": 0.18,
                "warning_level": "MONITOR",
                "reasoning": "Degradation is inside the envelope but the cliff is six laps out.",
            },
            situation={
                "overtake_prob": 0.34,
                "sc_prob_3lap": 0.08,
                "threat_level": "MEDIUM",
                "gap_ahead_s": 1.42,
                "pace_delta_s": -0.12,
                "sc_currently_active": False,
                "vsc_active": False,
                "reasoning": "DRS threat building: the gap to PIA has closed for three laps.",
            },
            radio={
                "radio_events": [
                    {
                        "driver": "NOR",
                        "message": "Rear grip is going away, especially through the last sector.",
                        "analysis": {"intent": "PROBLEM", "sentiment": "negative"},
                    }
                ],
                "rcm_events": [
                    {
                        "lap": lap,
                        "flag": "YELLOW",
                        "event_type": "YELLOW_FLAG_SECTOR",
                        "message": "Yellow flag in sector 2 - debris on the racing line.",
                    }
                ],
                "alerts": [{"intent": "PROBLEM", "driver": "NOR"}],
                "corrections": [],
                "reasoning": "The driver reports rear grip fading; sector 2 is under yellow.",
            },
            pit={
                "action": "PIT_NOW",
                "recommended_lap": lap + 1,
                "compound_recommendation": "HARD",
                "stop_duration_p05": 21.14,
                "stop_duration_p50": 22.40,
                "stop_duration_p95": 24.81,
                "undercut_prob": 0.63,
                "undercut_target": "RUS",
                "sc_reactive": False,
                "reasoning": "The undercut window against RUS is open for two more laps.",
            },
            regulation_context="Art. 55.7 - drivers must respect the delta under Safety Car",
            rag={
                "question": "Does a Safety Car change the mandatory compound rule?",
                "answer": "No. Art. 30.5(m) still requires two specifications in a dry race.",
                "articles": ["Article 30.5(m)", "Article 55.7"],
                "chunks": [
                    {
                        "article": "Article 30.5(m)",
                        "doc_type": "Sporting Regulations",
                        "year": 2025,
                        "text": (
                            "...each driver must use at least two specifications of "
                            "dry-weather tyre during the race..."
                        ),
                    }
                ],
            },
            # The routing layer emits agent IDs, not block names
            # (`_decide_agents_to_call` in strategy_orchestrator.py), and the
            # cards gate on exactly these two tokens.
            active=["N28", "N30"],
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
