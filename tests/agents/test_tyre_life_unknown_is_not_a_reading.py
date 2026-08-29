"""An absent tyre age must not arrive at N15 as a real one (#832, #1008).

`_tyre_life_in` used to return 1 for a missing `TyreLife`, arguing that a car
with no recorded age has just been given a set. The argument sounds reasonable
and picks the one value it must not: a tyre on the first lap of a stint reads 1,
so the sentinel and the measurement were the same number and nothing downstream
could tell an unknown from a fresh set. `race_state_builder` had already ruled
that out in writing:

    # TyreLife == 0 occurs ZERO times in 2023/2024/2025 (season minimums
    # 2.0/2.0/1.0), so 0 is a non-colliding sentinel. The old arcade/backend
    # default of 1 collides with real fresh-tyre laps.
    UNKNOWN_TYRE_LIFE = 0

One member of the pair had the rule and its twin did not, which is this repo's
most frequent defect. It fires on 451 rows, 1.98% of the 2025 featured parquet,
and it feeds the model that predicts how long a stop takes.

The second half of the fix is the warning text. `min(value, ceiling)` only clips
the top, so "the value is clipped to 50" was false for anything below the floor.
Nothing could reach below the floor before, which is why it never showed; the
unknown now can, so the message reports what is actually served.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agents.pace_agent import _previous_tyre_life, _unknown_if_missing
from src.agents.pit_strategy_agent import _MAX_TRAINED_TYRE_LIFE, PitStrategyAgent
from src.agents.race_state_builder import UNKNOWN_TYRE_LIFE

read = PitStrategyAgent._tyre_life_in

# Every way an age can be absent. The middle one is the original trap: a Series
# `.get` returns the STORED NaN, so a default in the call never fires.
ABSENT = {
    "a NaN value in a present column": pd.Series({"TyreLife": np.nan}),
    "the column missing entirely": pd.Series({"Driver": "NOR"}),
    "an explicit None": pd.Series({"TyreLife": None}, dtype=object),
}


@pytest.mark.parametrize("shape", sorted(ABSENT))
def test_an_absent_age_is_the_non_colliding_unknown(shape: str) -> None:
    assert read(ABSENT[shape]) == UNKNOWN_TYRE_LIFE


@pytest.mark.parametrize("shape", sorted(ABSENT))
def test_an_absent_age_is_not_a_fresh_set(shape: str) -> None:
    """The defect, stated as the assertion that catches it.

    Separate from the test above because they fail for different reasons: that
    one breaks if the constant drifts, this one breaks if the unknown ever again
    becomes a number the data can also hold.
    """
    assert read(ABSENT[shape]) != read(pd.Series({"TyreLife": 1.0}))


def test_a_real_fresh_set_is_untouched() -> None:
    """The other half of a sentinel: it must not eat a legitimate reading."""
    assert read(pd.Series({"TyreLife": 1.0})) == 1


@pytest.mark.parametrize(
    ("stored", "served"),
    [(2.0, 2), (17.0, 17), (49.0, 49), (50.0, 50), (63.0, 50), (200.0, 50)],
)
def test_a_present_age_is_passed_through_and_clipped_at_the_ceiling(
    stored: float, served: int
) -> None:
    assert read(pd.Series({"TyreLife": stored})) == served


def test_the_unknown_is_below_the_trained_floor(caplog: pytest.LogCaptureFixture) -> None:
    """Which is what makes the absence audible instead of silent.

    The envelope's floor is 1 because a tyre on its first lap reads 1, never 0.
    Serving 0 therefore violates it by construction, and the violation is the
    signal: N15's answer on that row is an extrapolation, not a fit.
    """
    with caplog.at_level(logging.WARNING):
        read(pd.Series({"TyreLife": np.nan}))
    assert "outside its trained range" in caplog.text


def test_the_warning_reports_what_is_served_not_the_ceiling(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`min` clips the top only, so naming the ceiling below the floor was a lie.

    Both directions are asserted in one test because the bug is the pair: a
    message that happens to be right above the ceiling and wrong below it reads
    as correct in every log anyone had ever seen.
    """
    with caplog.at_level(logging.WARNING):
        read(pd.Series({"TyreLife": float(_MAX_TRAINED_TYRE_LIFE + 13)}))
    assert f"served {_MAX_TRAINED_TYRE_LIFE}" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        read(pd.Series({"TyreLife": np.nan}))
    assert f"served {UNKNOWN_TYRE_LIFE}" in caplog.text
    assert f"served {_MAX_TRAINED_TYRE_LIFE}" not in caplog.text


def test_the_absence_is_logged_at_all(caplog: pytest.LogCaptureFixture) -> None:
    """A silent substitution is how the old value survived for two years."""
    with caplog.at_level(logging.WARNING):
        read(pd.Series({"TyreLife": np.nan}))
    assert "TyreLife missing" in caplog.text


def test_the_constant_is_shared_rather_than_retyped() -> None:
    """The two cannot drift apart again if there is only one of them.

    A local literal is what let the canonical builder and this consumer disagree
    while both looked deliberate.
    """
    import src.agents.pit_strategy_agent as module

    assert module.UNKNOWN_TYRE_LIFE is UNKNOWN_TYRE_LIFE


# ---------------------------------------------------------------------------
# The twin. N15 was one of a pair and N06 kept the fabrication, in a form that
# also destroyed the sentinel: `d.get("tyre_life") or 1` is 1 for a stored 0,
# and 0 is exactly what `race_state_builder` publishes for an age it does not
# have. So the unknown chosen in one module was rewritten as a fresh set one
# hop later, by a line sitting three above the comment explaining why the same
# `or` had already been removed for `position` (#628). Found by an adversarial
# gate over the N15 fix.
# ---------------------------------------------------------------------------

PACE_SOURCE = Path(__file__).resolve().parents[2] / "src" / "agents" / "pace_agent.py"


def test_the_published_unknown_survives_the_hop_to_n06() -> None:
    """A stored 0 must arrive as 0, which is what `or` could not do."""
    assert _unknown_if_missing(UNKNOWN_TYRE_LIFE) == UNKNOWN_TYRE_LIFE


def test_a_missing_age_is_the_same_unknown_for_both_models() -> None:
    """One constant, so N06 and N15 cannot drift the way N15 and the builder did."""
    assert _unknown_if_missing(None) == read(pd.Series({"TyreLife": np.nan}))


def test_n06_still_sees_a_real_fresh_set() -> None:
    assert _unknown_if_missing(1) == 1
    assert _unknown_if_missing(17) == 17


def test_an_unknown_produces_no_previous_age() -> None:
    """`_previous_tyre_life` returns None at <= 1, so the unknown reaches the
    booster as NaN rather than as a fabricated lap count."""
    assert _previous_tyre_life(_unknown_if_missing(0)) is None
    assert _previous_tyre_life(_unknown_if_missing(17)) == 16


def test_run_from_state_does_not_or_the_tyre_age() -> None:
    """The shape of the bug, not just its current value.

    `or` is false for 0, so any `... or <default>` around a tyre age silently
    rewrites the unknown. Asserting the VALUE alone would pass again the moment
    someone reintroduces the idiom with a different default, which is how this
    line survived the `position` fix in the same function.
    """
    tree = ast.parse(PACE_SOURCE.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "run_from_state":
            continue
        for op in ast.walk(node):
            if not isinstance(op, ast.BoolOp) or not isinstance(op.op, ast.Or):
                continue
            for value in op.values:
                names = {
                    arg.value
                    for call in ast.walk(value)
                    if isinstance(call, ast.Call)
                    for arg in call.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                }
                if "tyre_life" in names:
                    offenders.append(op.lineno)
    assert not offenders, (
        f"run_from_state ORs a tyre age at line(s) {sorted(set(offenders))}; `or` is "
        "false for 0, which is the published unknown, so it becomes a fresh set"
    )
