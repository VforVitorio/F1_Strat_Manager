"""N15's tyre-life ceiling, now declared as an operating envelope (#710).

The contract this pins is narrow and it is the whole point: **the envelope
labels, the clip decides.** Wiring a verdict in must not move a single value
N15 is fed, or the strategy goldens would shift and there would be no way to
tell a real regression from this change.

The second thing under test is that the label is emitted BEFORE the clip. After
clipping, every value reads as in-range by construction, so a check placed after
it would be permanently silent and would look like it worked.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from tests.conftest import HAS_TIRE_MODELS as _HAS_MODELS

ROOT = Path(__file__).parent.parent.parent
pytestmark = pytest.mark.skipif(
    not _HAS_MODELS,
    reason="data/models/ not present (importing the pit agent reads model config)",
)


def _tyre_life_in(value):
    """Call the real feature builder with a one-field pandas row."""
    import pandas as pd

    from src.agents.pit_strategy_agent import PitStrategyAgent

    return PitStrategyAgent._tyre_life_in(pd.Series({"TyreLife": value}))


def test_the_clip_still_returns_exactly_what_it_returned_before():
    """Values in range pass through; values above the ceiling come back clipped."""
    from src.agents.pit_strategy_agent import _MAX_TRAINED_TYRE_LIFE

    assert _tyre_life_in(1) == 1
    assert _tyre_life_in(25) == 25
    assert _tyre_life_in(_MAX_TRAINED_TYRE_LIFE) == _MAX_TRAINED_TYRE_LIFE
    assert _tyre_life_in(_MAX_TRAINED_TYRE_LIFE + 1) == _MAX_TRAINED_TYRE_LIFE
    assert _tyre_life_in(999) == _MAX_TRAINED_TYRE_LIFE


def test_an_out_of_range_call_is_announced(caplog):
    """Exceeding the trained range must stop being silent.

    This is the failure the contract exists for: N26 answered out-of-range calls
    with full confidence for two years because nothing ever said so out loud.
    """
    from src.agents.pit_strategy_agent import _MAX_TRAINED_TYRE_LIFE

    with caplog.at_level(logging.WARNING):
        _tyre_life_in(_MAX_TRAINED_TYRE_LIFE + 20)

    assert any("outside its trained range" in record.message for record in caplog.records)


def test_an_in_range_call_says_nothing(caplog):
    """A normal stint must not produce a warning, or the signal is worthless."""
    with caplog.at_level(logging.WARNING):
        _tyre_life_in(20)

    assert not [r for r in caplog.records if "trained range" in r.message]


def test_the_envelope_bound_is_the_clip_constant_not_a_second_number():
    """One source for the ceiling. A retyped boundary has shipped wrong here before."""
    from src.agents.pit_strategy_agent import _MAX_TRAINED_TYRE_LIFE, _N15_TYRE_LIFE_ENVELOPE

    _lower, upper = _N15_TYRE_LIFE_ENVELOPE.bounds["tyre_life_in"]
    assert upper == float(_MAX_TRAINED_TYRE_LIFE)


def test_a_missing_tyre_life_is_read_as_the_non_colliding_unknown():
    """This asserted `== 1` until #832, and 1 was the defect.

    The old name said "is still read as a fresh set", and the reason it read that
    way was an argument that sounds right and picks the one value it must not: a
    tyre on the first lap of a stint reads 1, so the sentinel and the measurement
    were the same number. `race_state_builder` had already ruled 1 out in writing
    for exactly that reason, and this consumer used it anyway.

    The envelope half of the original claim still holds and is what this file is
    about: the NaN path does not go through the bounds check to DECIDE anything.
    What changed is that the value it substitutes now sits below the floor, so the
    check labels the call an extrapolation instead of passing it silently.
    """
    import numpy as np

    from src.agents.race_state_builder import UNKNOWN_TYRE_LIFE

    assert _tyre_life_in(np.nan) == UNKNOWN_TYRE_LIFE
    assert _tyre_life_in(None) == UNKNOWN_TYRE_LIFE
    assert _tyre_life_in(np.nan) != _tyre_life_in(1.0), "the unknown collides with a fresh set"
