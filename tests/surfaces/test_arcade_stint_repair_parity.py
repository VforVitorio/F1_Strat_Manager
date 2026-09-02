"""The replay and the tower must not show two ages for one tyre (#951).

`repair_tyre_stints` was applied on the way into the models (`laps_augment`) and
on the way into PITWALL's timing tower (`session_data`), and not on the way into
the pyglet replay, which read `TyreLife` straight off FastF1. Both surfaces come
up from one `f1-arcade --strategy`, so on a race whose stint data needs repairing
the compound pill and the tower's TYRE column sat on screen together showing
different numbers, with nothing to say which to believe.

Two of a trio had the fix and the third did not. Nothing caught it because each
surface was self-consistent: the disagreement only exists across them, which is
also why the assertion here compares the two paths rather than checking either
one against a constant.

Measured on the four 2025 races on this machine: Lusail, Monaco and Las Vegas
come back untouched, and Melbourne moves 162 laps across 5 drivers, taking
unknown ages from 5 to 167. That is the #988 safety-car pit-lane transit shape.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from src.arcade.data import _repair_session_stints
from src.f1_strat_manager.tyre_stint_repair import repair_tyre_stints

ROOT = Path(__file__).resolve().parents[2]
REPAIRED_COLUMNS = ("TyreLife", "Stint", "Compound")


LOADER_SOURCE = ROOT / "src" / "arcade" / "data.py"


def _loader_ast() -> ast.FunctionDef:
    """`SessionLoader.load`, parsed rather than imported.

    Parsed because the property is about the ORDER of two statements, which a
    text search cannot see, and imported nothing because this has to run on a
    machine with no FastF1 cache and no races on disk.
    """
    tree = ast.parse(LOADER_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "load":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and sub.attr == "load":
                    return node
    raise AssertionError("SessionLoader.load is gone from src/arcade/data.py")


def test_the_loader_repairs_before_it_reads_a_lap() -> None:
    """The half the parity tests cannot see, and the one that silently rots.

    Everything above exercises the helper directly, so deleting its call from
    `SessionLoader.load` leaves them all green while the replay goes back to the
    raw column. That is the same shape as a pre-warm that stops pre-warming: the
    unit is correct and nothing reaches it.

    The order matters as much as the call. `session.load(...)` has to have run,
    or there are no laps to repair, and the repair has to precede the per-driver
    extraction, or it corrects a frame the pickle was already built from.
    """
    body = _loader_ast()
    repair_at = fastf1_load_at = extract_at = None
    for node in ast.walk(body):
        if isinstance(node, ast.Call):
            if getattr(node.func, "id", None) == "_repair_session_stints":
                repair_at = node.lineno
            attr = getattr(node.func, "attr", None)
            if attr == "load":
                fastf1_load_at = node.lineno
            elif attr == "_process_all_drivers":
                extract_at = node.lineno

    assert repair_at is not None, (
        "SessionLoader.load never calls _repair_session_stints, so the replay is "
        "back to reading TyreLife raw and disagrees with PITWALL's tower again"
    )
    gated = [
        branch.lineno
        for branch in ast.walk(body)
        if isinstance(branch, ast.If)
        for call in ast.walk(branch)
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "_repair_session_stints"
    ]
    assert not gated, (
        f"the repair call sits inside a conditional at line(s) {gated}, so it can be "
        "switched off without this file noticing - which `if False:` did"
    )
    assert fastf1_load_at is not None and fastf1_load_at < repair_at, (
        "the repair runs before session.load(), so there are no laps to repair"
    )
    assert extract_at is not None and repair_at < extract_at, (
        "the repair runs after the per-driver extraction, so the cached frames "
        "were built from the unrepaired column"
    )


class FakeSession:
    """`.laps` is the entire surface the helper touches.

    A real FastF1 session would cost minutes to load and would add nothing: the
    helper reads one attribute and writes three columns back onto it.
    """

    def __init__(self, laps: pd.DataFrame) -> None:
        self.laps = laps


def race(name: str) -> pd.DataFrame:
    path = ROOT / "data" / "raw" / "2025" / name / "laps.parquet"
    if not path.exists():
        pytest.skip(f"{name} 2025 not present (curated data/raw on this machine)")
    return pd.read_parquet(path)


@pytest.mark.parametrize("name", ["Melbourne", "Lusail", "Monaco", "Las_Vegas"])
@pytest.mark.parametrize("column", REPAIRED_COLUMNS)
def test_the_replay_and_the_tower_agree(name: str, column: str) -> None:
    """The two real paths, on the same source frame, column by column.

    Parametrised per column so a failure names which one drifted rather than
    reporting that "the frames differ".
    """
    raw = race(name)
    tower, _ = repair_tyre_stints(raw.copy())

    session = FakeSession(raw.copy())
    _repair_session_stints(session)
    replay = session.laps

    both_absent = replay[column].isna() & tower[column].isna()
    assert (both_absent | (replay[column] == tower[column])).all()


def test_the_repair_actually_fires_on_the_race_that_needs_it() -> None:
    """Otherwise the parity above would pass on two untouched frames.

    This is the assertion that fails if the helper is never called, or returns
    early, or writes its columns to a copy. Melbourne is the race the repair
    exists for, so a green parity test with zero rows moved would be green about
    nothing.
    """
    raw = race("Melbourne")
    session = FakeSession(raw.copy())
    _repair_session_stints(session)

    before, after = raw["TyreLife"], session.laps["TyreLife"]
    moved = (before != after) & ~(before.isna() & after.isna())
    assert int(moved.sum()) == 162
    assert int(after.isna().sum()) == 167


@pytest.mark.parametrize("name", ["Lusail", "Monaco", "Las_Vegas"])
def test_a_healthy_race_is_left_alone(name: str) -> None:
    """The repair's own contract, asserted where the replay now depends on it.

    It matters here beyond tidiness: an unnecessary rewrite of these columns
    would change the pickled bytes of every cached race rather than only the
    corrupted ones, which is the difference between one cache bump and a habit.
    """
    raw = race(name)
    session = FakeSession(raw.copy())
    _repair_session_stints(session)
    for column in REPAIRED_COLUMNS:
        before, after = raw[column], session.laps[column]
        assert (after.isna() == before.isna()).all()
        assert (after[before.notna()] == before[before.notna()]).all()


def misplaced_boundary_race() -> pd.DataFrame:
    """One car whose Stint column advances three laps after the stop it belongs to.

    Every race on this machine exercises only the NULLING half of the repair, so
    `Stint` and `Compound` move zero rows on all four and a write-back that
    dropped them stayed green. That is a guard passing about the empty set, and
    it is what a gate found in the first version of this file.

    The boundary-correction half is what writes those two columns
    (`tyre_stint_repair.py:443-444`), and it needs a shape none of the fixtures
    has: a `PitInTime` on lap 10 whose stint boundary does not land until 13.
    """
    n = 20
    return pd.DataFrame(
        {
            "Driver": ["NOR"] * n,
            "LapNumber": [float(i) for i in range(1, n + 1)],
            "Stint": [1.0] * 12 + [2.0] * 8,
            "TyreLife": [float(i) for i in range(1, 13)] + [float(i) for i in range(1, 9)],
            "Compound": ["MEDIUM"] * 12 + ["HARD"] * 8,
            "PitInTime": [pd.NaT] * 9 + [pd.Timedelta(seconds=1)] + [pd.NaT] * 10,
        }
    )


@pytest.mark.parametrize("column", REPAIRED_COLUMNS)
def test_all_three_repaired_columns_reach_the_replay(column: str) -> None:
    """The write-back is complete, on a frame where all three actually move.

    Measured on this fixture: TyreLife moves 10 rows, Stint 2 and Compound 2.
    Dropping either of the last two from the loop is invisible on every real
    race here and fails immediately on this one.
    """
    raw = misplaced_boundary_race()
    tower, report = repair_tyre_stints(raw.copy())
    assert report.boundaries_corrected == 1, "the fixture stopped exercising the boundary path"

    session = FakeSession(raw.copy())
    _repair_session_stints(session)
    replay = session.laps

    moved = (raw[column] != tower[column]) & ~(raw[column].isna() & tower[column].isna())
    assert int(moved.sum()) > 0, f"{column} does not move on this fixture, so it proves nothing"
    both_absent = replay[column].isna() & tower[column].isna()
    assert (both_absent | (replay[column] == tower[column])).all()


def test_the_laps_object_survives_the_repair() -> None:
    """The replay needs FastF1's `Laps` methods after this runs.

    Replacing `session.laps` with the repair's plain DataFrame would strip
    `pick_drivers`, which the per-driver extraction calls, and `pick_fastest`,
    which the reference lap needs. Writing columns back preserves the object,
    and this is what fails if someone simplifies the loop into an assignment.
    """

    class Marked(pd.DataFrame):
        """Stands in for the `Laps` subclass: a type that must not be replaced."""

        @property
        def _constructor(self):
            return Marked

    session = FakeSession(Marked(race("Melbourne")))
    _repair_session_stints(session)
    assert isinstance(session.laps, Marked)
