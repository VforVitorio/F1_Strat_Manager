"""#788 — the present-``None`` guard on the agents' raw-lap_state weather reads.

The canonical builder (#784) protects the ``RaceState``. These tests protect the layer
BELOW it: the agents that re-read the raw ``lap_state`` dict and build their own
``session_meta`` from it. That distinction is the whole reason these tests exist —
#788 was declared fixed once when only the builder had been fixed, and an adversarial
gate found ``/recommend`` still returning 422 because the crash had simply moved down
here.

The producers emit an unmeasured reading as the key PRESENT holding ``None``
(``_safe_none`` in the telemetry backend, ``get_weather_state`` in the RSM), which is
exactly what ``dict.get(key, default)`` does not catch. Every 2025 laps parquet ships
without weather columns, so this is the shape those agents actually receive today.
"""

from __future__ import annotations

import math

import pytest

from src.agents._shared_defaults import reading_or_default

# The dict the backend producer really emits for a 2025 lap, verified over real HTTP
# (documents/audits/GATE_qatar_lap7_cross_surface.md, Task 2).
PRODUCER_WEATHER_2025 = {
    "air_temp": None,
    "track_temp": None,
    "track_temp_start": None,
    "humidity": None,
    "rainfall": 0,
}


def test_present_none_falls_back_where_two_arg_get_does_not():
    """The bug in one assertion: .get's default does not fire, the helper's does."""
    assert PRODUCER_WEATHER_2025.get("air_temp", 28.0) is None
    assert reading_or_default(PRODUCER_WEATHER_2025, "air_temp", 28.0) == 28.0


def test_absent_key_also_falls_back():
    assert reading_or_default({}, "track_temp", 38.0) == 38.0


@pytest.mark.parametrize("value", [0.0, 0, False])
def test_a_legitimate_falsy_reading_survives(value):
    """Not an ``or`` fallback: a real 0.0 is a measurement, not an absence.

    Rewriting 0.0 to a default is the #633 conflation — a different bug, and the one
    an ``or`` would have introduced here.
    """
    assert reading_or_default({"track_temp": value}, "track_temp", 38.0) == value


def test_a_real_reading_passes_through():
    assert reading_or_default({"air_temp": 23.4}, "air_temp", 28.0) == 23.4


def test_the_three_agents_agree_on_this_read():
    """All three sub-agents route this read through one implementation (#788).

    They keep DIFFERENT default temperatures on purpose — pace 25/35, tire and
    race_situation 28/38 — because reconciling those numbers is a modelling decision
    tracked in #789. What must not differ again is the None handling: pace was the only
    one of the three that had it right, and its guard is now the shared helper.
    """
    for default in (25.0, 28.0, 35.0, 38.0):
        assert reading_or_default(PRODUCER_WEATHER_2025, "track_temp", default) == default


def test_the_producer_shape_never_reaches_float_as_none():
    """float() over every weather key of a real producer payload must not raise.

    ``float(None)`` is the exact TypeError that 422'd /recommend on every 2025 lap.
    """
    for key, default in (("air_temp", 28.0), ("track_temp", 38.0), ("humidity", 50.0)):
        value = float(reading_or_default(PRODUCER_WEATHER_2025, key, default))
        assert math.isfinite(value)
