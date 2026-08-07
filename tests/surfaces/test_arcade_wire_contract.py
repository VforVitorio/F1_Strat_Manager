"""The frozen shape of the arcade broadcast payload (#843).

This is the golden-payload test. It builds a complete broadcast the way
`_broadcast_if_due` does and compares its structure, key by key and type
by type, against a literal frozen below. A producer-side change that
renames a key, drops one, or changes its type fails here in CI rather
than on a consumer's screen.

**Updating the golden is part of the change, not a chore around it.** If
a diff is intentional, edit `GOLDEN_SHAPE` in the same commit and bump
`STREAM_SCHEMA_VERSION` when the change is not purely additive. The point
is that it cannot happen by accident.

The `strategy` half is exercised with fully populated DTOs rather than
defaults, because a field that is `None` in the fixture freezes as
`NoneType` and would pin nothing about its real type. That half carries
the most fields and is the one the PITWALL agents window reads one by one.
"""

from __future__ import annotations

import json
from functools import partial
from types import SimpleNamespace

import pytest

pytest.importorskip("arcade", reason="the arcade replay is an optional surface")

from src.arcade.app import F1ArcadeView  # noqa: E402
from src.arcade.config import DT, STREAM_SCHEMA_VERSION  # noqa: E402
from src.arcade.data import FrameData, SessionData  # noqa: E402
from src.arcade.strategy import (  # noqa: E402
    LapDecisionDTO,
    PerAgentOutputsDTO,
    StartEventDTO,
    StrategyState,
)

CIRCUIT_LENGTH_M = 5278.0


def _shape(value):
    """Describe a JSON-shaped value as nested key -> type-name.

    A list is described by its first element, so an empty list and a
    populated one are different shapes. That is deliberate: the fixture
    populates every list the payload can carry, so an empty one in the
    result means the producer stopped filling it.
    """
    if isinstance(value, dict):
        return {key: _shape(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_shape(value[0])] if value else []
    return type(value).__name__


def _frames(n: int) -> list[FrameData]:
    return [
        FrameData(
            t=i * DT,
            x=1.0,
            y=2.0,
            speed=250.0,
            gear=7,
            drs=8,
            throttle=90.0,
            brake=5.0,
            lap=1 + i // 10,
            dist=float(i) * 10.0,
            rel_dist=(i % 10) / 10.0,
            tyre=1,
            tyre_life=9.0,
            active=True,
        )
        for i in range(n)
    ]


def _decision() -> LapDecisionDTO:
    """A LapDecisionDTO with every optional field populated.

    Defaults would leave nine fields as `None` and freeze their type as
    `NoneType`, which pins the key but not the type.
    """
    return LapDecisionDTO(
        lap_number=12,
        compound="MEDIUM",
        tyre_life=9,
        position=4,
        lap_time_s=81.234,
        gap_ahead_s=1.42,
        action="PIT_NOW",
        confidence=0.71,
        reasoning="undercut window open",
        scenario_scores={"PIT_NOW": 0.71, "STAY_OUT": 0.29},
        pace_mode="PUSH",
        risk_posture="AGGRESSIVE",
        pit_lap_target=13,
        compound_next="HARD",
        undercut_target="RUS",
        agent_alerts=["tyre cliff in 2 laps"],
        guardrail_reason="none",
        per_agent=PerAgentOutputsDTO(
            pace={"predicted_lap_time_s": 81.0},
            tire={"degradation_pct": 12.0},
            situation={"threat_level": "MEDIUM"},
            radio={"sentiment": "negative"},
            pit={"pit_duration_s": 22.4},
            regulation_context="Art. 55.7",
            rag={"question": "q", "answer": "a", "articles": ["55.7"], "chunks": ["c"]},
            active=["pace", "tire"],
        ),
        memory_block="lap 11: STAY_OUT",
        plan_changed=True,
    )


def _view(session: SessionData, state: StrategyState, rival: str | None, clients: int):
    """A stand-in for `F1ArcadeView` carrying only what `_broadcast_if_due` reads.

    `_build_arcade_snapshot` is bound explicitly because the method under
    test calls it through `self`. Anything else the broadcast path starts
    reading will raise here rather than pass against a permissive mock.
    """
    view = SimpleNamespace(
        _session=session,
        _driver_main="NOR",
        _driver_rival=rival,
        _year=2025,
        _stream_server=SimpleNamespace(client_count=lambda: clients, broadcast=_sent.append),
        _strategy_state=state,
        # `_broadcast_if_due` increments the tick then broadcasts when it
        # wraps to 0, so -1 makes the very next call due.
        _broadcast_tick=-1,
        _broadcast_seq=0,
        _last_broadcast_idx=15,
        _frame_index=20.0,
        playback_speed=1.0,
        _is_paused=False,
    )
    view._build_arcade_snapshot = partial(F1ArcadeView._build_arcade_snapshot, view)
    return view


def _payload() -> dict:
    """Build one broadcast exactly as `_broadcast_if_due` assembles it."""
    session = SessionData(
        gp_name="Australia",
        location="Melbourne",
        year=2025,
        frames_by_driver={"NOR": _frames(40), "PIA": _frames(40)},
        min_lap_number=1,
        max_lap_number=4,
        circuit_length_m=CIRCUIT_LENGTH_M,
        total_frames=40,
        global_t_min=4260.355,
    )
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
    state.latest = _decision()
    state.history = [_decision()]

    _sent.clear()
    F1ArcadeView._broadcast_if_due(_view(session, state, rival="PIA", clients=1))
    assert _sent, "the fixture must actually broadcast"
    return _sent[-1]


_sent: list[dict] = []


# --- The golden ------------------------------------------------------------

GOLDEN_SHAPE = {
    "schema_version": "int",
    "seq": "int",
    "arcade": {
        "circuit_length_m": "float",
        "driver_main": "str",
        "driver_rival": "str",
        "drivers": {
            "NOR": {
                "active": "bool",
                "compound": "int",
                "dist": "float",
                "lap": "int",
                "rel_dist": "float",
                "speed": "float",
                "tyre_life": "float",
            },
            "PIA": {
                "active": "bool",
                "compound": "int",
                "dist": "float",
                "lap": "int",
                "rel_dist": "float",
                "speed": "float",
                "tyre_life": "float",
            },
        },
        "global_t_min": "float",
        "gp_name": "str",
        "lap": "int",
        "location": "str",
        "t": "float",
        "telemetry": {
            "main": [
                {
                    "brake": "float",
                    "dist": "float",
                    "drs": "int",
                    "gear": "int",
                    "lap": "int",
                    "speed": "float",
                    "t": "float",
                    "throttle": "float",
                }
            ],
            "dropped": "int",
            "rewound": "bool",
            "rival": [
                {
                    "brake": "float",
                    "dist": "float",
                    "drs": "int",
                    "gear": "int",
                    "lap": "int",
                    "speed": "float",
                    "t": "float",
                    "throttle": "float",
                }
            ],
        },
        "total_laps": "int",
        "year": "int",
    },
    "playback": {
        "frame_index": "int",
        "paused": "bool",
        "speed": "float",
        "total_frames": "int",
    },
    "strategy": {
        "error": "NoneType",
        "finished": "bool",
        "history_tail": [
            {
                "action": "str",
                "agent_alerts": ["str"],
                "compound": "str",
                "compound_next": "str",
                "confidence": "float",
                "gap_ahead_s": "float",
                "guardrail_reason": "str",
                "lap_number": "int",
                "lap_time_s": "float",
                "memory_block": "str",
                "pace_mode": "str",
                "pit_lap_target": "int",
                "plan_changed": "bool",
                "position": "int",
                "reasoning": "str",
                "risk_posture": "str",
                "scenario_scores": {"PIT_NOW": "float", "STAY_OUT": "float"},
                "tyre_life": "int",
                "undercut_target": "str",
            }
        ],
        "latest": {
            "action": "str",
            "agent_alerts": ["str"],
            "compound": "str",
            "compound_next": "str",
            "confidence": "float",
            "gap_ahead_s": "float",
            "guardrail_reason": "str",
            "lap_number": "int",
            "lap_time_s": "float",
            "memory_block": "str",
            "pace_mode": "str",
            "per_agent": {
                "active": ["str"],
                "pace": {"predicted_lap_time_s": "float"},
                "pit": {"pit_duration_s": "float"},
                "radio": {"sentiment": "str"},
                "rag": {
                    "answer": "str",
                    "articles": ["str"],
                    "chunks": ["str"],
                    "question": "str",
                },
                "regulation_context": "str",
                "situation": {"threat_level": "str"},
                "tire": {"degradation_pct": "float"},
            },
            "pit_lap_target": "int",
            "plan_changed": "bool",
            "position": "int",
            "reasoning": "str",
            "risk_posture": "str",
            "scenario_scores": {"PIT_NOW": "float", "STAY_OUT": "float"},
            "tyre_life": "int",
            "undercut_target": "str",
        },
        "start": {
            "driver": "str",
            "driver2": "str",
            "gp": "str",
            "lap_end": "int",
            "lap_start": "int",
            "no_llm": "bool",
            "provider": "str",
            "team": "str",
            "total_laps": "int",
            "year": "int",
        },
    },
}


def _minimal_payload() -> dict:
    """The other end of the range: no rival pinned, no decision yet, an error set.

    The session still carries every car - single-driver mode is a choice of
    which two are charted, not a smaller field - so only `driver_rival`
    goes null here, not the `drivers` block.
    """
    session = SessionData(
        gp_name="Australia",
        location="Melbourne",
        year=2025,
        frames_by_driver={"NOR": _frames(40), "PIA": _frames(40)},
        min_lap_number=1,
        max_lap_number=4,
        circuit_length_m=CIRCUIT_LENGTH_M,
        total_frames=40,
        global_t_min=4260.355,
    )
    state = StrategyState()
    state.error = "lap 12: pipeline failed"
    _sent.clear()
    F1ArcadeView._broadcast_if_due(_view(session, state, rival=None, clients=1))
    assert _sent, "the fixture must actually broadcast"
    return _sent[-1]


# What the same payload looks like before the first decision exists, with no
# rival pinned and an error set. Frozen separately rather than unioned into
# GOLDEN_SHAPE, because a union hides WHICH state makes a field null and a
# consumer needs exactly that. Five fields differ, and they are the five a
# type generated from the rich shape alone would get wrong.
GOLDEN_MINIMAL_DIFFS = {
    "arcade.driver_rival": "NoneType",
    "arcade.telemetry.rival": [],
    "strategy.start": "NoneType",
    "strategy.latest": "NoneType",
    "strategy.history_tail": [],
    "strategy.error": "str",
}


def _highest_diffs(rich, minimal, path: str = "") -> dict:
    """Where the two shapes part company, reported at the highest node.

    A subtree that collapses to `None` is one difference, not one per leaf:
    "strategy.latest is null before the first decision" is the fact a
    consumer needs, and 22 entries saying its fields are absent is noise.
    """
    if rich == minimal:
        return {}
    if isinstance(rich, dict) and isinstance(minimal, dict):
        diffs: dict = {}
        for key in rich:
            child = f"{path}.{key}" if path else key
            if key not in minimal:
                diffs[child] = "absent"
            else:
                diffs.update(_highest_diffs(rich[key], minimal[key], child))
        return diffs
    return {path: minimal}


def test_the_payload_shape_is_the_frozen_one():
    """Every key and type on the wire, pinned.

    `history_tail` deliberately has no `per_agent`: `snapshot_dict` strips
    it from past decisions so the tail does not re-send 30 copies of the
    dataclass ten times a second. If it reappears here, that saving is
    gone.
    """
    assert _shape(_payload()) == GOLDEN_SHAPE


def test_the_states_that_make_a_field_null_are_frozen_too():
    """A consumer typed off the rich shape alone gets five wrong non-nullables.

    Before the first lap the pipeline decides, with no rival pinned and an
    error set, these five fields differ from the rich payload. Everything
    else must be identical: if a sixth field starts varying, that is a new
    optionality nobody declared.
    """
    diffs = _highest_diffs(_shape(_payload()), _shape(_minimal_payload()))

    assert diffs == GOLDEN_MINIMAL_DIFFS


def test_the_payload_carries_the_schema_version():
    assert _payload()["schema_version"] == STREAM_SCHEMA_VERSION


# --- seq semantics ----------------------------------------------------------


def _seq_of(n_broadcasts: int, *, clients: int = 1) -> list[int]:
    """Drive `_broadcast_if_due` n times and collect the sequence numbers."""
    session = SessionData(
        frames_by_driver={"NOR": _frames(400)},
        circuit_length_m=CIRCUIT_LENGTH_M,
        total_frames=400,
    )
    view = _view(session, StrategyState(), rival=None, clients=clients)
    view._last_broadcast_idx = -1
    view._frame_index = 0.0
    _sent.clear()
    for _ in range(n_broadcasts):
        view._broadcast_tick = -1  # force every call to be due
        view._frame_index += 2.5
        F1ArcadeView._broadcast_if_due(view)
    return [p["seq"] for p in _sent]


def test_seq_increases_by_exactly_one_per_message():
    seqs = _seq_of(10)

    assert seqs == list(range(1, 11))
    assert all(b - a == 1 for a, b in zip(seqs, seqs[1:]))


def _classify(seqs: list[int]) -> list[str]:
    """The rule a polling consumer applies to consecutive `seq` values.

    This is the whole point of the field: without it a consumer reading a
    latest-payload slot on its own timer cannot tell "nothing new yet"
    from "I missed two frames", and both happen. Measured against one
    slot at 10 Hz on each side with a half-period offset: 15 duplicate
    reads and 15 skips out of 54 polls.
    """
    return [
        "duplicate" if b == a else "clean" if b == a + 1 else f"skipped {b - a - 1}"
        for a, b in zip(seqs, seqs[1:])
    ]


def test_a_consumer_can_tell_a_duplicate_from_a_skip():
    """Both failure modes, read off `seq` alone with no other state."""
    stream = _seq_of(6)

    assert _classify(stream) == ["clean"] * 5
    # The consumer polled twice before the producer moved on.
    assert _classify([stream[0], stream[0], stream[1]]) == ["duplicate", "clean"]
    # The consumer was late and the slot had been overwritten twice.
    assert _classify([stream[0], stream[3]]) == ["skipped 2"]


def test_no_seq_is_burned_while_nobody_is_listening():
    """The counter must count messages sent, not ticks elapsed.

    If it advanced on ticks with no subscriber, a dashboard that attaches
    mid-race would see its first `seq` jump by thousands and every gap
    check downstream would be measuring the wrong thing.
    """
    assert _seq_of(10, clients=0) == []


# --- Nothing non-finite reaches the wire, enforced at the encoder -----------


def _sent_bytes(payload: dict) -> list[bytes]:
    """Run the real `TelemetryStreamServer.broadcast` against a fake socket.

    Through the server, not around it: the guarantee lives in `json.dumps`
    and the sanitiser that feeds it, and a test that builds the dict and
    encodes it itself would prove nothing about either.
    """
    from src.arcade.stream import TelemetryStreamServer

    written: list[bytes] = []
    server = TelemetryStreamServer()
    server._running = True
    server._clients = [SimpleNamespace(sendall=written.append)]
    server.broadcast(payload)
    return written


def _reject_non_finite(token: str) -> float:
    raise AssertionError(f"non-finite token on the wire: {token}")


def test_a_nan_from_a_model_costs_its_field_and_not_the_whole_tick():
    """The `strategy` block is a bare `asdict()` of raw model output.

    A model that cannot compute a value hands back NaN, and three of the
    guards on the way in are `or`/truthiness tests, which NaN passes:
    `nan or 0.0` is `nan`. `json.dumps` then writes a bare `NaN` that
    Python reads back and `JSON.parse` rejects, so one unusable prediction
    would have dropped every field on the tick for a web consumer.

    The panels that do not depend on that prediction keep updating; the one
    that does gets `null`, which it already has to handle.
    """
    payload = _payload()
    payload["strategy"]["latest"]["lap_time_s"] = float("nan")
    payload["strategy"]["latest"]["scenario_scores"]["PIT_NOW"] = float("inf")
    payload["strategy"]["history_tail"][0]["confidence"] = float("-inf")

    written = _sent_bytes(payload)

    assert written, "a NaN must not silence the whole broadcast"
    decoded = json.loads(written[0], parse_constant=_reject_non_finite)
    assert decoded["strategy"]["latest"]["lap_time_s"] is None
    assert decoded["strategy"]["latest"]["scenario_scores"]["PIT_NOW"] is None
    assert decoded["strategy"]["history_tail"][0]["confidence"] is None
    # Everything that was computable is still there.
    assert decoded["strategy"]["latest"]["action"] == "PIT_NOW"
    assert decoded["seq"] == payload["seq"]


def test_a_payload_that_cannot_be_encoded_is_dropped_rather_than_half_sent():
    """The encoder is the backstop, and a dropped message is visible in `seq`."""
    payload = _payload()
    payload["strategy"]["latest"]["reasoning"] = {1, 2}  # a set is not JSON

    assert _sent_bytes(payload) == []
