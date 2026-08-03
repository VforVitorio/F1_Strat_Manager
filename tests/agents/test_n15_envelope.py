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


def test_a_missing_tyre_life_is_still_read_as_a_fresh_set():
    """The NaN path predates the envelope and must be untouched by it."""
    import numpy as np

    assert _tyre_life_in(np.nan) == 1
    assert _tyre_life_in(None) == 1
