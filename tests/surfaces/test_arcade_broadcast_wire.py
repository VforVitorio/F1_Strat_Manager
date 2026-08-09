"""Contract tests for the arcade TCP broadcast payload (`src/arcade/app.py`).

The broadcast is the only channel between the replay process and every
downstream surface, so the shape of the dict it puts on the wire is a
contract, not an implementation detail. These tests pin the fields whose
absence caused, or would cause, a downstream surface to render something
false.

**Why the tests build a stand-in for the view instead of an `F1ArcadeView`.**
Constructing the real view needs an `arcade.Window`, which needs a GL
context, which CI does not have. The payload builders read exactly four
attributes off `self`, so the tests call them as plain functions with a
namespace carrying those four. That is deliberate: if a builder starts
reading a fifth attribute, these tests fail loudly with `AttributeError`
rather than passing against a mock that quietly grew a new field.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("arcade", reason="the arcade replay is an optional surface")

from src.arcade.app import F1ArcadeView  # noqa: E402
from src.arcade.data import (  # noqa: E402
    DT,
    FrameData,
    SessionData,
    _lap_fraction_from_distance,
)
from src.arcade.gaps import RaceGapCalculator  # noqa: E402

# The three numbers a synthetic session needs to be self-consistent: a
# circuit length the distances accumulate against, a session-time origin
# that is emphatically not zero (a zero origin would let a broken anchor
# pass), and a driver who is on the wire.
CIRCUIT_LENGTH_M = 5278.0
GLOBAL_T_MIN = 3612.5
MAIN = "NOR"


def _frame(index: int, lap: int, rel_dist: float, *, active: bool = True) -> FrameData:
    """One synthetic 25 Hz slice, with `dist` race-cumulative as the loader builds it."""
    return FrameData(
        t=index * DT,
        x=0.0,
        y=0.0,
        speed=250.0,
        gear=7,
        drs=0,
        throttle=100.0,
        brake=0.0,
        lap=lap,
        dist=(lap - 1) * CIRCUIT_LENGTH_M + rel_dist * CIRCUIT_LENGTH_M,
        rel_dist=rel_dist,
        tyre=1,
        tyre_life=8.0,
        active=active,
    )


def _running_car(n_frames: int) -> list[FrameData]:
    """A car that keeps circulating for the whole synthetic session."""
    return [_frame(i, lap=1 + i // 10, rel_dist=(i % 10) / 10.0) for i in range(n_frames)]


def _retired_car(n_frames: int, last_live_index: int) -> list[FrameData]:
    """A car whose telemetry stops at `last_live_index`.

    Past that point the loader's `np.interp` clamps, so every later frame
    repeats the final real sample verbatim and only `active` records that
    the car is gone. The fixture reproduces that clamp rather than
    inventing a distinguishable value, because the clamp is the reason
    the flag has to be on the wire.
    """
    frames = [_frame(i, lap=1 + i // 10, rel_dist=(i % 10) / 10.0) for i in range(n_frames)]
    frozen = frames[last_live_index]
    for i in range(last_live_index + 1, n_frames):
        frames[i] = FrameData(**{**vars(frozen), "t": i * DT, "active": False})
    return frames


def _car_without_position_data(n_frames: int) -> list[FrameData]:
    """A car FastF1 gives no position channel for, built the way the LOADER builds it.

    Not hypothetical: on Melbourne 2025 this is HAD. An earlier version of
    this fixture hand-set `rel_dist` to NaN, which is what FastF1 used to
    deliver and what the loader can no longer produce: since the derivation
    moved to `_lap_fraction_from_distance` over the driver's own distance,
    a car that never moves comes out **finite 0.0** on every frame -
    measured, one unique value across all 154,173 of his.

    So the fixture runs a flat distance through the real derivation. A
    hand-set NaN would keep the guard green while asserting about an input
    production cannot emit, and the value that production DOES emit means
    "at the line" (#856).
    """
    dist = np.zeros(n_frames)
    lap_numbers = np.ones(n_frames, dtype=int)
    rel_dist = _lap_fraction_from_distance(dist, lap_numbers, CIRCUIT_LENGTH_M)
    return [
        FrameData(
            **{
                **vars(_frame(i, lap=1, rel_dist=0.0)),
                "dist": float(dist[i]),
                "rel_dist": float(rel_dist[i]),
                "speed": 0.0,
            }
        )
        for i in range(n_frames)
    ]


def _session(n_frames: int = 40) -> SessionData:
    return SessionData(
        gp_name="Australia",  # the display label
        location="Melbourne",  # the folder name under data/raw/<year>/
        year=2025,
        frames_by_driver={
            MAIN: _running_car(n_frames),
            "SAI": _retired_car(n_frames, last_live_index=5),
            "HAD": _car_without_position_data(n_frames),
        },
        max_lap_number=1 + (n_frames - 1) // 10,
        circuit_length_m=CIRCUIT_LENGTH_M,
        total_frames=n_frames,
        global_t_min=GLOBAL_T_MIN,
        # What the loader computes: `frames[-1].dist > frames[0].dist`.
        has_position={MAIN: True, "SAI": True, "HAD": False},
    )


def _snapshot(
    session: SessionData,
    frame_idx: int,
    rival: str | None = None,
    span_start: int | None = None,
    rewound: bool = False,
) -> dict:
    """Build one broadcast snapshot. Defaults to a one-sample span at `frame_idx`."""
    view = SimpleNamespace(
        _session=session,
        _driver_main=MAIN,
        _driver_rival=rival,
        _year=session.year,
        _gaps=RaceGapCalculator(session),
        _color_for=lambda code: (255, 255, 255),
    )
    start = frame_idx if span_start is None else span_start
    return F1ArcadeView._build_arcade_snapshot(view, frame_idx, start, rewound, 0)


# --- #842: the four scalars the wire used to drop ---------------------------


def test_every_car_carries_active_and_rel_dist():
    """A consumer iterating `drivers` must be able to place and qualify each car."""
    wire = _snapshot(_session(), frame_idx=3)

    assert set(wire["drivers"]) == {MAIN, "SAI", "HAD"}
    for code, car in wire["drivers"].items():
        assert "active" in car, f"{code} has no liveness flag"
        assert "rel_dist" in car, f"{code} cannot be placed around the lap"
        if car["rel_dist"] is not None:
            assert 0.0 <= car["rel_dist"] <= 1.0


def test_a_retired_car_is_flagged_while_its_other_fields_stay_frozen():
    """The exact failure #842 describes: a DNF broadcasting as if it were racing.

    `lap`, `dist` and `speed` are identical before and after the car
    stops, because `np.interp` clamps. `active` is the only field that
    changes, so it is the only one a timing tower or a track ring can
    key off.
    """
    session = _session()
    live = _snapshot(session, frame_idx=5)["drivers"]["SAI"]
    dead = _snapshot(session, frame_idx=30)["drivers"]["SAI"]

    assert live["active"] is True
    assert dead["active"] is False
    # Everything else is indistinguishable — that is the whole point.
    assert dead["lap"] == live["lap"]
    assert dead["dist"] == live["dist"]
    assert dead["speed"] == live["speed"]
    # The car the fixture keeps running must not be caught by the same net.
    assert _snapshot(session, frame_idx=30)["drivers"][MAIN]["active"] is True


def test_global_t_min_turns_the_frame_clock_back_into_session_time():
    """`t` alone is `frame_index * DT`; only `t + global_t_min` is joinable.

    FastF1 session time is what `laps.parquet` (`Time`, `LapStartTime`,
    `Sector*SessionTime`) and `weather.parquet` are keyed on, so this sum
    is the join key that did not exist before.
    """
    frame_idx = 17
    wire = _snapshot(_session(), frame_idx)

    assert wire["t"] == pytest.approx(frame_idx * DT)
    assert wire["global_t_min"] == pytest.approx(GLOBAL_T_MIN)
    assert wire["t"] + wire["global_t_min"] == pytest.approx(GLOBAL_T_MIN + frame_idx * DT)


def test_a_car_with_no_position_data_is_unknown_rather_than_at_the_line():
    """The value the loader produces for such a car means something, and must not ship.

    The old guard was against `min(1.0, nan) == 1.0`. That input is gone:
    the loader derives the fraction from the driver's own distance, and a
    car that never moves comes out finite **0.0** — which is not a clamp
    but is still a position, "at the line", and on Melbourne 2025 it rides
    the wire on 100 % of HAD's frames with `active` true for 2,935 of them.

    The fixture is built by the real derivation, so this asserts about a
    state production can actually reach.
    """
    session = _session()
    had_frames = session.frames_by_driver["HAD"]
    assert had_frames[3].rel_dist == 0.0, "the premise: the loader emits a finite at-the-line 0.0"

    wire = _snapshot(session, frame_idx=3)

    assert wire["drivers"]["HAD"]["rel_dist"] is None
    assert wire["drivers"][MAIN]["rel_dist"] is not None
    # The two-car telemetry block takes the same route, so it must agree.
    # It did not: `has_position` was read for the drivers block only, so the
    # same car read "no position" in one half of the payload and sat in
    # distance bucket 0 in the other.
    telemetry = _snapshot(_session(), frame_idx=3, rival="HAD")["telemetry"]
    assert telemetry["rival"][0]["dist"] is None
    assert telemetry["main"][0]["dist"] is not None


def test_the_arcade_block_carries_nothing_non_finite():
    """`json.dumps` writes a bare `NaN`, which no strict parser accepts.

    Scoped to the `arcade` block on purpose, and named for it. An earlier
    version of this test claimed to cover "the payload" while building
    only this sub-dict, so it passed while the `strategy` block put NaN on
    the wire unguarded. The whole-payload guarantee is enforced at the
    encoder and tested in `test_arcade_wire_contract.py`.
    """
    wire = _snapshot(_session(), frame_idx=3, rival="HAD")

    encoded = json.dumps(wire)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    # `parse_constant` fires on exactly the tokens a strict parser rejects.
    json.loads(encoded, parse_constant=_reject_non_finite)


def _reject_non_finite(token: str) -> float:
    raise AssertionError(f"non-finite token on the wire: {token}")


def test_location_is_published_and_can_genuinely_disagree_with_the_label():
    """`location` exists because the label CAN name a different race.

    An earlier version of this test hand-set the two fields to differ in
    the fixture and then asserted they differed, which verifies that a
    dict copy copies two keys. This one exercises the real resolver: on
    the fallback path `get_gp_names` returns a hardcoded 2024 table, and a
    2025 round then carries the wrong race. That fallback also names the
    session pickle, so the divergence mislabels the cache too.
    """
    from src.arcade.config import GP_NAMES, get_gp_names

    wire = _snapshot(_session(), frame_idx=0)
    assert wire["location"], "the folder-resolving field must be on the wire"

    canonical = get_gp_names(2025)
    fallback = get_gp_names(1999)  # no such calendar -> the hardcoded table

    assert fallback is GP_NAMES, "an absent year must fall back, not raise"
    disagreeing = {rnd for rnd in canonical if rnd in fallback and canonical[rnd] != fallback[rnd]}
    assert disagreeing, "if the two tables agreed everywhere, location would be redundant"


def test_the_pedal_scale_is_not_decided_by_which_driver_sorts_first():
    """`max()` keeps a NaN that arrives FIRST, because every `x > nan` is False.

    One driver whose whole throttle channel is NaN, sorting ahead of the
    others, therefore flipped the session multiplier from 1.0 to 100.0 —
    every pedal reading above 1 % published as 100.0, for all twenty cars,
    on nothing but driver order. Not triggered on Melbourne 2025, where
    HAD's pedal channels are real; silent and global when it is.
    """
    from src.arcade.data import _pedal_multiplier

    all_nan = {"data": {"throttle": np.full(5, np.nan)}}
    real = {"data": {"throttle": np.array([0.0, 55.0, 100.0])}}

    assert _pedal_multiplier([all_nan, real], "throttle") == 1.0
    assert _pedal_multiplier([real, all_nan], "throttle") == 1.0
    assert _pedal_multiplier([{"data": {"throttle": np.array([0.0, 0.9])}}], "throttle") == 100.0
