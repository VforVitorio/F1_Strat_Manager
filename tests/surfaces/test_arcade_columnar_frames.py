"""The columnar payload serves the frames the per-object payload served (#1118).

`frames_by_driver` holds fourteen parallel arrays per driver instead of one
`FrameData` per sample. Every consumer still indexes it as a sequence, so the
defect this file exists to catch is a column that stops lining up with its
neighbours: a field written into the wrong slot, a channel that keeps its old
length after a slice, an array that survives a round trip in a different order.
None of that raises. It serves a frame built from fourteen values that never
belonged together, and the replay renders it without complaint.

The suite's other guards cannot see it. The ~60 fixtures in the sibling files
build a list and let `SessionData.__post_init__` convert it, so they exercise
`from_frames` and the accessor as a pair and agree with each other whichever
way both are wrong. What is asserted here is the ONE property that pins the
pair to something outside itself: a frame read back at index `i` equals the
frame that went in at index `i`, field by field.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from src.arcade.data import DriverFrames, FrameData, SessionData

FIELDS = (
    "t",
    "x",
    "y",
    "speed",
    "gear",
    "drs",
    "throttle",
    "brake",
    "lap",
    "dist",
    "rel_dist",
    "tyre",
    "tyre_life",
    "active",
)


def _frame(i: int) -> FrameData:
    """One frame whose every field carries a DIFFERENT value derived from `i`.

    Distinct per field and per index on purpose. A fixture that repeats a value
    across fields cannot fail when two columns are swapped, which is the defect
    this module is for.
    """
    return FrameData(
        t=i * 0.04,
        x=100.0 + i,
        y=200.0 + i,
        speed=300.0 + i,
        gear=i % 8 + 1,
        drs=i % 4,
        throttle=(i % 100) + 0.5,
        brake=(i % 7) + 0.25,
        lap=i // 10 + 1,
        dist=1000.0 + i * 2,
        rel_dist=(i % 50) / 50.0,
        tyre=i % 5,
        tyre_life=(i % 30) + 0.75,
        active=i % 11 != 10,
    )


@pytest.fixture
def frames() -> list[FrameData]:
    return [_frame(i) for i in range(120)]


def test_the_field_list_here_matches_the_dataclass(frames: list[FrameData]) -> None:
    """The tests below iterate `FIELDS`, so a field missing from it is untested.

    A new channel added to `FrameData` and not to the columns would otherwise
    be checked by nothing here, and every assertion would stay green while the
    channel was silently dropped.
    """
    assert FIELDS == tuple(vars(frames[0]))
    assert set(FIELDS) == set(vars(DriverFrames.from_frames(frames)))


def test_every_frame_survives_the_round_trip_through_the_columns(
    frames: list[FrameData],
) -> None:
    """The whole contract, on every index and every field."""
    columns = DriverFrames.from_frames(frames)
    assert len(columns) == len(frames)
    for i, original in enumerate(frames):
        assert columns[i] == original, f"frame {i}"


def test_a_session_normalises_a_plain_list(frames: list[FrameData]) -> None:
    """What keeps every hand-built fixture in the sibling files working."""
    session = SessionData(frames_by_driver={"NOR": frames})
    stored = session.frames_by_driver["NOR"]
    assert isinstance(stored, DriverFrames)
    assert list(stored) == frames


def test_columns_already_built_are_left_alone(frames: list[FrameData]) -> None:
    """The loader path: normalising again would cost a rebuild per load."""
    columns = DriverFrames.from_frames(frames)
    session = SessionData(frames_by_driver={"NOR": columns})
    assert session.frames_by_driver["NOR"] is columns


def test_negative_and_end_indices_address_the_same_frame(frames: list[FrameData]) -> None:
    """`gaps.py` reads `[-1]` and the loader reads `[0]`, so both are pinned."""
    columns = DriverFrames.from_frames(frames)
    assert columns[-1] == frames[-1]
    assert columns[len(frames) - 1] == frames[-1]
    assert columns[0] == frames[0]


def test_an_index_past_the_end_is_refused(frames: list[FrameData]) -> None:
    """Iteration and `[-1]` both rely on the bounds being real."""
    columns = DriverFrames.from_frames(frames)
    with pytest.raises(IndexError):
        columns[len(frames)]


def test_iteration_yields_every_frame_once_in_order(frames: list[FrameData]) -> None:
    assert list(DriverFrames.from_frames(frames)) == frames


def test_an_empty_driver_is_empty_rather_than_broken() -> None:
    """A driver with no telemetry reaches `has_position`, which calls `len`."""
    columns = DriverFrames.from_frames([])
    assert len(columns) == 0
    assert list(columns) == []


def test_the_columns_survive_a_pickle_round_trip(frames: list[FrameData]) -> None:
    """The payload IS a pickle, so equality after `dumps`/`loads` is the format.

    A dtype that does not survive the round trip, or an array restored in a
    different order, shows up here and nowhere else in the suite.
    """
    columns = DriverFrames.from_frames(frames)
    restored = pickle.loads(pickle.dumps(columns))
    for name in FIELDS:
        assert np.array_equal(getattr(restored, name), getattr(columns, name)), name
    assert list(restored) == frames


def test_one_timeline_is_stored_once_for_the_whole_grid() -> None:
    """Twenty drivers share `t` by identity, which is what keeps the file small.

    Storing it per driver would add about 20 MB to a race for twenty copies of
    the same 124,000 floats. Pickle preserves the sharing only while the arrays
    are the same object, so identity is the thing to assert, not equality.
    """
    timeline = np.arange(0.0, 4.0, 0.04)
    empty = np.zeros(len(timeline))
    grid = {
        code: DriverFrames(
            t=timeline,
            **{name: empty for name in FIELDS if name != "t"},
        )
        for code in ("NOR", "PIA", "VER")
    }
    restored = pickle.loads(pickle.dumps(grid))
    first = restored["NOR"].t
    assert all(restored[code].t is first for code in restored), "the timeline was copied per driver"


def test_the_loader_fills_every_column_exactly_once() -> None:
    """`_resample_driver` builds the columns, and no test in CI can run it.

    Building one needs a FastF1 session and minutes of work, so the six guards
    that read a real cache file skip everywhere except a machine that has one
    (`test_arcade_telemetry_span.py`). That leaves the loader's own construction
    covered by nothing here, and a field dropped from it would serve a default
    rather than raise.

    Parsed rather than grepped: a keyword name appears in that call, in the
    docstring above it and in `FrameData` alike, so a text search cannot tell a
    real argument from a mention of one.
    """
    import ast
    import inspect
    import textwrap

    from src.arcade.data import SessionLoader

    source = textwrap.dedent(inspect.getsource(SessionLoader._resample_driver))
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DriverFrames"
    ]
    assert len(calls) == 1, f"expected one DriverFrames(...) construction, found {len(calls)}"

    keywords = [kw.arg for kw in calls[0].keywords]
    assert keywords.count(None) == 0, "a **kwargs spread would hide which columns are filled"
    assert sorted(keywords) == sorted(FIELDS), f"columns filled: {sorted(keywords)}"
