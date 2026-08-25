"""Dev-only broadcast producer with a POPULATED strategy block.

Drives the real `TelemetryStreamServer` from the real Melbourne 2025 session
so PITWALL can be developed against a full payload without launching the
arcade, running the agent pipeline, or spending a single LLM call. Built for
the sprint-3 AGENTS port, where an empty strategy block leaves every card in
its idle state and there was nothing to compare against the Qt window it was
replacing.

    python scripts/dev_pitwall_producer.py 180      # seconds to broadcast

Then, in another terminal, `python -m src.pitwall` or
`python -m src.pitwall`.

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
from src.arcade.gaps import RaceGapCalculator
from src.arcade.strategy import (
    LapDecisionDTO,
    PerAgentOutputsDTO,
    StartEventDTO,
    StrategyState,
)
from src.arcade.stream import TelemetryStreamServer

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
ON_UPDATE_HZ = 60.0


def fixture_call(lap: int) -> tuple[str, float]:
    """The call this fixture makes on a lap, decided in ONE place.

    The memory block QUOTES the previous lap, and it used to quote a
    hardcoded `STAY_OUT (0.58)` while `history` generated something else
    from a formula further down the file. Nothing noticed until sprint 8
    surfaced the previous call as a first-class chip: the window then
    rendered `was STAY OUT (0.62) - L22` beside a reasoning tab saying
    `lap 22: STAY_OUT (0.58)`, two panels disagreeing about one lap.

    A fixture that contradicts itself teaches the reader a bug that is
    not there, which is worse than no fixture.
    """
    action = "STAY_OUT" if lap % 4 else "EXTEND"
    confidence = 0.5 + (lap % 5) * 0.06
    return action, confidence


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
        # Both landed on the DTO with schema v2 (#1046). They are here because a
        # fixture that omits a field the real producer sends is the drift #853
        # was about: the window gets developed against a payload thinner than
        # the one it will receive.
        contingencies=[
            {
                "trigger": "if RUS pits within two laps",
                "switch_to": "PIT_NOW",
                "priority": "HIGH",
                "rationale": "the undercut window shuts once he clears traffic",
            }
        ],
        key_risks=["rejoin into traffic", "the cliff arrives before the stop"],
        # `None`, not the STRING "none". A non-empty string is truthy, so the
        # window rendered `⚠ Guardrail: none` in the alarm colour on every lap
        # of every dev run - a red warning whose content is that there is
        # nothing to warn about. The real path sends `None`
        # (`src/arcade/strategy.py:796`), so this only ever misled whoever was
        # developing against it, which is the whole point of the file.
        guardrail_reason=None,
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
                # A mismatch against the radio above, because an empty list here
                # leaves the section this fixture exists to exercise invisible.
                # N29 only fills this on an LLM profile, so the free dev path is
                # the ONLY way to see it without spending a call.
                "corrections": [
                    {
                        "driver": "NOR",
                        "original_intent": "PROBLEM",
                        "suggested_intent": "INFORMATION",
                        "span": "especially through the last sector",
                        "reason": "reads as a description of where, not a request to act",
                    }
                ],
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
            #
            # **It VARIES per lap, and a constant here would misrepresent the
            # real producer.** N30 is consulted every lap; N28 only routes when
            # a stop is live, which is what the routing strip exists to show. A
            # fixture that sent both on every lap would render a solid block and
            # nothing on that strip could be judged by eye - the same drift a
            # dev fixture caused in #853.
            active=["N30"] + (["N28"] if lap % 3 else []),
        ),
        memory_block=(
            f"lap {lap - 1}: {fixture_call(lap - 1)[0]} "
            f"({fixture_call(lap - 1)[1]:.2f}) - undercut window not yet open"
        ),
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
    team="McLaren",
    lap_start=1,
    lap_end=57,
    total_laps=57,
    no_llm=False,
    provider="openai",
)
state.latest = decision(23, "PIT_NOW", 0.71)
state.history = [decision(lap, *fixture_call(lap)) for lap in range(14, 23)] + [state.latest]

view = SimpleNamespace(
    _session=session,
    _driver_main="NOR",
    _driver_rival="PIA",
    _year=2025,
    _gaps=RaceGapCalculator(session),
    # The real `F1ArcadeView._color_for` (app.py:915) reads the session's own
    # palette. A stub returning white made every driver on the wire white,
    # which is invisible in the AGENTS window (it does not use the colours)
    # and wrong the moment a consumer draws twenty cars: PITWALL's track ring
    # colours its dots from `driver_colors` precisely so nothing hardcodes a
    # palette, and this harness was quietly feeding it a flat one.
    _color_for=lambda code: session.driver_colors.get(code, (255, 255, 255)),
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

# How long a lap lasts here, in wall-clock seconds at this playback speed.
#
# **The decision LAP advances, and it did not use to.** `state.latest` was
# pinned to lap 23 for the whole run while only the frame index moved, so every
# lap-keyed accumulator in the AGENTS window saw exactly one lap forever: the
# routing strip rendered a single column and nothing about it could be judged by
# eye. The real producer advances, so a fixture that does not is the drift #853
# is about.
#
# Forty-five seconds is one Melbourne lap at the 2x playback above, which keeps
# the fixture's own clock and its decisions telling the same story.
LAP_SECONDS = 45.0

print(f"producing {SECONDS:.0f}s with a POPULATED strategy block", flush=True)
started = time.perf_counter()
deadline = started + SECONDS
while time.perf_counter() < deadline:
    lap = min(23 + int((time.perf_counter() - started) / LAP_SECONDS), state.start.lap_end)
    if lap != state.latest.lap_number:
        state.history.append(state.latest)
        state.latest = decision(lap, *fixture_call(lap))
    view._frame_index += (1.0 / ON_UPDATE_HZ) * FPS * view.playback_speed
    F1ArcadeView._broadcast_if_due(view)
    time.sleep(1.0 / ON_UPDATE_HZ)
print("done", flush=True)
server.stop()
