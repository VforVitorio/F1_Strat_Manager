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

import ast
import json
import re
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("arcade", reason="the arcade replay is an optional surface")

from src.arcade.app import F1ArcadeView  # noqa: E402
from src.arcade.config import DT, STREAM_SCHEMA_VERSION  # noqa: E402
from src.arcade.data import FrameData, SessionData  # noqa: E402
from src.arcade.gaps import RaceGapCalculator  # noqa: E402
from src.arcade.strategy import (  # noqa: E402
    LapDecisionDTO,
    PerAgentOutputsDTO,
    StartEventDTO,
    StrategyState,
)

CIRCUIT_LENGTH_M = 5278.0

# One telemetry sample's frozen types. Named rather than written out once per
# driver: v2 puts a span under every code on the grid, and twenty copies of a
# nine-key literal is twenty chances for one of them to drift.
_TELEMETRY_SAMPLE_SHAPE = {
    "brake": "float",
    "dist": "float",
    "drs": "int",
    "drs_open": "bool",
    "gear": "int",
    "lap": "int",
    "speed": "float",
    "t": "float",
    "throttle": "float",
}
REPO_ROOT = Path(__file__).resolve().parents[2]


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
            # 100.0, not the 5.0 that stood here: since #1002 `brake` is resampled
            # nearest-neighbour off a BOOLEAN raw channel, so the only values the
            # producer can put on the wire are 0.0 and 100.0. A fixture outside that
            # set describes a payload the arcade cannot build.
            brake=100.0,
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
        # Real field names, taken from the agents' own output dataclasses and
        # guarded by `test_the_per_agent_fixture_uses_the_producers_real_field_names`.
        # An invented set used to live here and agreed with the dev producer's
        # invented set, so this file was green while pinning a contract no
        # producer emits, and every consumer built against it rendered blanks
        # (#853).
        per_agent=PerAgentOutputsDTO(
            pace={"lap_time_pred": 81.0, "delta_vs_prev": -0.2},
            tire={"compound": "MEDIUM", "laps_to_cliff_p50": 6.0, "warning_level": "MONITOR"},
            situation={"threat_level": "MEDIUM", "sc_prob_3lap": 0.08},
            radio={
                # Populated, not empty: `_shape` describes a list by its first
                # element, so empty lists would pin nothing about what the
                # radio card reads out of these entries.
                "radio_events": [
                    {
                        "driver": "NOR",
                        "message": "rear grip going away",
                        "analysis": {"intent": "PROBLEM", "sentiment": "negative"},
                    }
                ],
                "rcm_events": [
                    {"lap": 12, "flag": "YELLOW", "event_type": "YELLOW_FLAG", "message": "debris"}
                ],
                "alerts": [{"driver": "NOR", "intent": "PROBLEM"}],
            },
            pit={"stop_duration_p50": 22.4, "compound_recommendation": "HARD"},
            regulation_context="Art. 55.7",
            rag={"question": "q", "answer": "a", "articles": ["55.7"], "chunks": ["c"]},
            active=["N28", "N30"],
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
        _gaps=RaceGapCalculator(session),
        _color_for=lambda code: (255, 255, 255),
        # `broadcast` takes a FACTORY since #1049, so the stub has to RUN it.
        # `broadcast=_sent.append` would append the uncalled closure and every
        # assertion below would shape a function instead of the payload.
        _stream_server=SimpleNamespace(
            client_count=lambda: clients,
            broadcast=lambda build: _sent.append(build()),
        ),
        _strategy_state=state,
        # `_broadcast_if_due` increments the tick then broadcasts when it
        # wraps to 0, so -1 makes the very next call due.
        _broadcast_tick=-1,
        _broadcast_seq=0,
        _last_broadcast_idx=15,
        _last_broadcast_clock=15.0,
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
                "has_finished": "bool",
                "has_position": "bool",
                "lap": "int",
                "laps_completed": "int",
                "progress": "float",
                "rel_dist": "float",
                "speed": "float",
                "tyre_life": "float",
            },
            "PIA": {
                "active": "bool",
                "compound": "int",
                "dist": "float",
                "has_finished": "bool",
                "has_position": "bool",
                "lap": "int",
                "laps_completed": "int",
                "progress": "float",
                "rel_dist": "float",
                "speed": "float",
                "tyre_life": "float",
            },
        },
        "global_t_min": "float",
        "gp_name": "str",
        "lap": "int",
        "location": "str",
        "driver_colors": {"NOR": ["int"], "PIA": ["int"]},
        "race_order": ["str"],
        "t": "float",
        "telemetry": {
            # One span per driver since schema v2 (#1048), keyed exactly like
            # the `drivers` block above, `driver_colors` and `race_order`. The
            # fixture carries two cars, so two keys; what is pinned is that
            # BOTH appear regardless of which one is the rival, which is the
            # whole point of the change and is asserted directly in
            # `test_the_span_key_set_does_not_depend_on_who_the_rival_is`.
            "drivers": {
                "NOR": [_TELEMETRY_SAMPLE_SHAPE],
                "PIA": [_TELEMETRY_SAMPLE_SHAPE],
            },
            "dropped": "int",
            "rewound": "bool",
        },
        "total_laps": "int",
        "track_status": "str",
        # The decoded form travels beside the digits so no consumer forks the
        # priority order into a second language. Both are NoneType in the
        # golden payload because its fixture lap has no TrackStatus entry -
        # which is the case that must NOT render as a green track.
        "track_status_color": "NoneType",
        "track_status_label": "NoneType",
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
                "pace": {"delta_vs_prev": "float", "lap_time_pred": "float"},
                "pit": {"compound_recommendation": "str", "stop_duration_p50": "float"},
                "radio": {
                    "alerts": [{"driver": "str", "intent": "str"}],
                    "radio_events": [
                        {
                            "analysis": {"intent": "str", "sentiment": "str"},
                            "driver": "str",
                            "message": "str",
                        }
                    ],
                    "rcm_events": [
                        {"event_type": "str", "flag": "str", "lap": "int", "message": "str"}
                    ],
                },
                "rag": {
                    "answer": "str",
                    "articles": ["str"],
                    "chunks": ["str"],
                    "question": "str",
                },
                "regulation_context": "str",
                "situation": {"sc_prob_3lap": "float", "threat_level": "str"},
                "tire": {
                    "compound": "str",
                    "laps_to_cliff_p50": "float",
                    "warning_level": "str",
                },
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
    # `arcade.telemetry.rival: []` used to sit here, and its absence is the
    # point of #1048: with a span per driver the telemetry block is identical
    # whether or not a rival is pinned, because the key set is the grid rather
    # than a pair of roles. If a telemetry entry ever reappears in this dict,
    # the spans have started depending on the rival again.
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


def test_the_smoke_harness_declares_the_schema_version_the_producer_emits():
    """The DATA smoke fabricates ticks, and it had drifted ahead of the wire.

    `src/pitwall/ui/scripts/smoke-data.mjs` is the only other thing in this
    repo that builds a whole tick, and its 233 checks are what the DATA window
    is developed against. It carried a hardcoded `schema_version: 2` for
    twelve days while this constant was still 1, which the #1048 bump makes
    accidentally correct. Nothing tied the two together, so the next bump
    would drift the same way and the harness would go on describing a payload
    the producer stopped sending.

    Read as text rather than executed: the harness is an ES module driven by
    a browser bundle, and this suite has no Node in it.
    """
    harness = (REPO_ROOT / "src/pitwall/ui/scripts/smoke-data.mjs").read_text(encoding="utf-8")
    declared = re.findall(r"schema_version:\s*(\d+)", harness)

    assert declared, "the smoke harness stopped declaring a schema version at all"
    assert set(declared) == {str(STREAM_SCHEMA_VERSION)}, (
        f"smoke-data.mjs declares schema_version {sorted(set(declared))}, "
        f"the producer emits {STREAM_SCHEMA_VERSION}"
    )


# --- the per_agent contract, against the producer rather than against itself -


# Which dataclass in `src/agents/` fills each `per_agent` block. `rag` is
# absent on purpose: the engine assembles it as a plain dict rather than
# dumping `RegulationContext`, and the golden already pins its four keys.
_PER_AGENT_SOURCES: dict[str, tuple[str, str]] = {
    "pace": ("src/agents/pace_agent.py", "PaceOutput"),
    "tire": ("src/agents/tire_agent.py", "TireOutput"),
    "situation": ("src/agents/race_situation_agent.py", "RaceSituationOutput"),
    "radio": ("src/agents/radio_agent.py", "RadioOutput"),
    "pit": ("src/agents/pit_strategy_agent.py", "PitStrategyOutput"),
}


def _dataclass_field_names(relative_path: str, class_name: str) -> set[str]:
    """Annotated attribute names of one dataclass, read from its source.

    Parsed with `ast` rather than imported: importing `src/agents/` pulls
    torch, xgboost and three transformer checkpoints, which is minutes of
    CI for a list of strings.
    """
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            }
    raise AssertionError(f"{class_name} is gone from {relative_path}")


def test_the_per_agent_fixture_uses_the_producers_real_field_names():
    """The fixture must be a subset of what the agents actually emit.

    Before #853 it was not, and neither was the dev producer: both invented
    `predicted_lap_time_s`, `degradation_pct`, `sc_prob`, `pit_duration_s`.
    They agreed with each other, so this file stayed green while pinning a
    contract no producer has ever emitted, and the Qt cards - sprint 3's
    acceptance reference - rendered `pred 0.00s` and `deg - s/lap` against
    the very rig built to populate them.

    Comparing the fixture against the agents' own dataclasses is what makes
    this guard about the producer instead of about itself: rename
    `lap_time_pred` upstream and it goes red.
    """
    per_agent = _payload()["strategy"]["latest"]["per_agent"]

    for block, (path, class_name) in _PER_AGENT_SOURCES.items():
        real = _dataclass_field_names(path, class_name)
        unknown = set(per_agent[block]) - real
        assert not unknown, (
            f"per_agent[{block!r}] invents {sorted(unknown)}; {class_name} has {sorted(real)}"
        )


def test_the_dev_producer_uses_the_same_real_field_names():
    """The twin. #853 fixed the fixture AND the producer; only one got a guard.

    `scripts/dev_pitwall_producer.py` is what every PITWALL sprint develops
    against, and sprints 4 to 6 will edit it. Guarding the fixture and not
    the producer leaves unprotected the exact file whose invented keys made
    the reference window render `pred 0.00s` for a whole sprint.

    Read with `ast`, like the fixture guard: importing the producer loads a
    session and opens a socket.
    """
    source = (REPO_ROOT / "scripts" / "dev_pitwall_producer.py").read_text(encoding="utf-8")
    call = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "PerAgentOutputsDTO"
    )
    blocks = {
        keyword.arg: {key.value for key in keyword.value.keys}
        for keyword in call.keywords
        if isinstance(keyword.value, ast.Dict)
    }

    for block, (path, class_name) in _PER_AGENT_SOURCES.items():
        assert block in blocks, f"the producer stopped sending a {block} block"
        unknown = blocks[block] - _dataclass_field_names(path, class_name)
        assert not unknown, f"the dev producer invents {sorted(unknown)} in {block}"

    active = next(keyword for keyword in call.keywords if keyword.arg == "active")
    tokens = [element.value for element in active.value.elts]
    assert tokens and all(t.startswith("N") and t[1:].isdigit() for t in tokens), tokens


def test_the_routing_list_carries_agent_ids_and_not_block_names():
    """`active` gates the two conditional cards, and it is not the block keys.

    `_decide_agents_to_call` returns `{"N28", "N30"}` and the cards test
    `"N28" in active`. The dev producer used to send
    `["pace", "tire", "situation", "pit"]`, so both conditional cards stayed
    on their trigger hint no matter what the rig published.
    """
    active = _payload()["strategy"]["latest"]["per_agent"]["active"]

    assert active, "the fixture must exercise the conditional cards"
    assert all(token.startswith("N") and token[1:].isdigit() for token in active), active


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
    import threading
    import time

    from src.arcade.stream import TelemetryStreamServer

    written: list[bytes] = []
    server = TelemetryStreamServer()
    server._running = True
    server._clients = [SimpleNamespace(sendall=written.append)]
    # `broadcast` only queues now: it returns without touching a socket so it
    # cannot block the pyglet frame loop. Run the sender to see what reaches
    # the wire, because that is the thing under test.
    sender = threading.Thread(target=server._send_loop, daemon=True)
    sender.start()
    server.broadcast(lambda: payload)
    deadline = time.perf_counter() + 2.0
    while not written and time.perf_counter() < deadline:
        time.sleep(0.01)
    server._running = False
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


# --- #857: the wire can feed a leaderboard ----------------------------------


def _drift_frames(n: int, speed_mps: float, drift_per_lap_m: float = 0.0) -> list[FrameData]:
    """Frames whose odometer accumulates that car's OWN per-lap length.

    `dist` is race-cumulative metres, so it looks like a progress axis and
    is not one: each car accumulates the distance IT drove. On the real race
    that drift reaches 1877 m on a 5220 m circuit, and sorting on it put the
    wrong car in the lead on 37 % of sampled frames.
    """
    circuit = 5000.0
    lap_length = circuit + drift_per_lap_m
    out = []
    for i in range(n):
        travelled = speed_mps * i * DT
        completed = int(travelled // circuit)
        fraction = (travelled % circuit) / circuit
        out.append(
            FrameData(
                t=i * DT,
                x=1.0,
                y=2.0,
                speed=speed_mps * 3.6,
                gear=7,
                drs=0,
                throttle=90.0,
                brake=0.0,
                lap=1 + completed,
                dist=(completed + fraction) * lap_length,
                rel_dist=fraction,
                tyre=1,
                tyre_life=9.0,
                active=True,
            )
        )
    return out


def _order_snapshot(**cars) -> dict:
    """One published snapshot from a session of the given cars."""
    n = min(len(f) for f in cars.values())
    session = SessionData(
        gp_name="Melbourne",
        location="Melbourne",
        year=2025,
        frames_by_driver=dict(cars),
        circuit_length_m=5000.0,
        max_lap_number=0,
        total_frames=n,
    )
    view = SimpleNamespace(
        _session=session,
        _driver_main=next(iter(cars)),
        _driver_rival=None,
        _year=2025,
        _gaps=RaceGapCalculator(session),
        _color_for=lambda code: (255, 255, 255),
    )
    return F1ArcadeView._build_arcade_snapshot(view, n - 1, n - 26, False)


def test_the_wire_publishes_an_order_a_dist_sort_would_get_wrong():
    """`race_order` is the producer's answer, not something a consumer re-derives.

    The fixture makes the two coordinates disagree: SLOW runs long per lap,
    so its published `dist` exceeds FAST's while FAST is genuinely ahead on
    track. A consumer sorting `dist` inverts them; the published key does
    not, because it is `_rank_drivers` - the same code the arcade panel
    ranks with, so the wire and the panel cannot drift apart.
    """
    snapshot = _order_snapshot(
        # Long enough that every car has a PREVIOUS lap of its own: on a
        # two-lap fixture nobody does, the circuit length stands in for all
        # of them, and the per-car drift stops cancelling - which is the
        # fixture being unrepresentative, not the coordinate failing.
        FAST=_drift_frames(12000, 52.0),
        SLOW=_drift_frames(12000, 50.0, drift_per_lap_m=900.0),
    )
    drivers = snapshot["drivers"]

    assert drivers["SLOW"]["dist"] > drivers["FAST"]["dist"], "the fixture must invert dist"
    assert snapshot["race_order"][0] == "FAST", "the published order gets it right anyway"


def test_the_reveal_carrier_is_per_driver_and_not_the_main_driver_lap():
    """Band 1-2 reveals lap L for driver d iff `L <= laps_completed`.

    The tick carries only the MAIN driver's lap, and on the real race the
    field spans two or three different laps at 96 % of instants - so masking
    everyone at one lap lags the leaders and leaks look-ahead for the cars
    behind, at the same time. `laps_completed` reads the crossing map, so it
    is per driver and monotone forward, unlike the interpolated `lap`.
    """
    snapshot = _order_snapshot(FAST=_drift_frames(12000, 52.0), SLOW=_drift_frames(12000, 30.0))
    fast = snapshot["drivers"]["FAST"]
    slow = snapshot["drivers"]["SLOW"]

    assert isinstance(fast["laps_completed"], int)
    assert fast["laps_completed"] > slow["laps_completed"], (
        "one shared lap number cannot express this, which is why the rule is per driver"
    )


# --- The track status, decoded once and published -------------------------


def test_the_track_status_is_decoded_by_the_producer_and_not_by_each_consumer():
    """The digits and their meaning travel together.

    Two surfaces now render this status: the arcade's own pill and PITWALL's
    band-1 strip. The priority order (red > SC > VSC > yellow) and the four
    labels are a project rule, so a TypeScript consumer decoding the digits
    itself would be that rule's second copy - which is precisely what
    `driver_colors` rides on the wire to prevent.
    """
    from src.arcade.overlays import track_status_label

    assert track_status_label("1") == ("GREEN", (16, 185, 129))
    assert track_status_label("2") == ("YELLOW FLAG", (250, 204, 21))
    assert track_status_label("4") == ("SAFETY CAR", (255, 140, 0))
    assert track_status_label("6") == ("VSC", (245, 158, 11))
    assert track_status_label("5") == ("RED FLAG", (239, 68, 68))
    # Concurrent events: a red flag wins even with a yellow already out.
    assert track_status_label("25")[0] == "RED FLAG"
    assert track_status_label("24")[0] == "SAFETY CAR"


def test_a_lap_with_no_track_status_entry_is_unknown_and_not_green():
    """The one case the arcade's pill never had to tell apart.

    The pill HIDES on a clear track, so "clear" and "the loader has no entry
    for this lap" are the same absence to it. A strip always shows a status,
    so conflating them would put a confident GREEN on a lap whose status
    nobody knows - the sentinel class this repo keeps paying for, where a
    default is a value the code can also legitimately find.
    """
    from src.arcade.overlays import track_status_label

    assert track_status_label("") is None, "unknown is None, never GREEN"
    assert track_status_label("1") is not None, "a real clear IS green and says so"


def test_every_recommendation_field_the_wire_drops_was_decided_about():
    """A field that stops at the DTO boundary must be a decision, not an oversight.

    `StrategyRecommendation` has fourteen fields and `LapDecisionDTO` copies ten.
    The rest used to vanish silently, so a new field added to the orchestrator
    would join them without anybody noticing it never reached a surface. This is
    the guard against that: the difference between the two sets is FROZEN, and a
    fifteenth field fails here until someone writes down where it goes.

    Read with `ast` for the reason `_dataclass_field_names` gives: importing
    `src/arcade/strategy_pipeline.py` pulls torch, langchain and three transformer
    checkpoints, measured at 18.4 s, and CI has no model weights at all.
    """
    recommendation = _dataclass_field_names(
        "src/agents/strategy_orchestrator.py", "StrategyRecommendation"
    )
    dto = _dataclass_field_names("src/arcade/strategy.py", "LapDecisionDTO")

    assert recommendation - dto == {
        # Decision content, held back only because adding it is a schema change;
        # rides with #1048's bump rather than migrating a frozen contract twice.
        "contingencies",
        "key_risks",
        # The PLAN timeline already draws the stint boundary from `pit_lap_target`.
        # A second source for one number on one surface is the twin shape.
        "expected_stint_end",
        # `None` by Art. 55.7 under a safety car, and the absence is the
        # load-bearing case. Nothing on either window asks for it yet.
        "target_lap_time_s",
        # NOT dropped: it reaches the wire through `PerAgentOutputsDTO`, which is
        # why it is listed here rather than treated as missing.
        "regulation_context",
    }, "a recommendation field changed sides without a decision being recorded (#1046)"

    # And the other direction, so the frozen set cannot be satisfied by the DTO
    # quietly losing a field it is supposed to carry.
    assert {"action", "confidence", "reasoning", "scenario_scores"} <= dto


def test_the_pipeline_delegate_no_longer_throws_the_stage_timings_away():
    """`run_lap`'s third value reaches a logger instead of the floor (#1045).

    It used to be bound to `_timings` and dropped on the same line that computed
    it, under a docstring promising a future change would put it on the wire.
    Nothing consumed it at any of the three call sites.

    Asserted on the SOURCE, and that limitation is real: exercising the function
    means importing the agent stack (18.4 s, three checkpoints) and then running
    the `rich` profile, which spends LLM calls. What the source can say is that
    the value is bound to a real name and that the name reaches a logging call.
    """
    tree = ast.parse((REPO_ROOT / "src/arcade/strategy_pipeline.py").read_text(encoding="utf-8"))
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_strategy_pipeline"
    )
    body = ast.dump(func)
    assert "_timings" not in body, "the third value is still bound to a discard name"
    assert "timings" in body, "the third value is not bound at all"
    logged = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Attribute) and node.attr in {"debug", "info", "warning"}
    ]
    assert logged, "the timings are bound and then still go nowhere"
