"""What the weather panel renders when a reading is missing (#1087).

The panel used to format four fields straight out of ``weather.get(key,
default)``. ``SessionLoader._weather_row_to_dict`` stores an explicit ``None``
under the key when FastF1's sample is NaN, so the default never fired and
``f"{None:.1f}"`` raised inside ``on_draw``, killing the pyglet render loop
rather than degrading one row.

Two things are asserted here and they are different claims. That no input
raises is the crash guard. That an absent reading renders ``"N/A"`` rather than
a plausible number is the sentinel guard, and it is the one that would catch a
"fix" that swapped the crash for an invented 18.0 C.

The rows are built by ``_weather_rows`` rather than by ``draw`` because
``draw`` needs a GL context, and a check that needs a window to run is a check
that does not run in CI. The split is the fix, not a testing convenience: the
five expressions that crashed are all in the extracted function.
"""

from __future__ import annotations

import pytest

from src.arcade.overlays import _reading, _weather_rows

# The shape the loader emits on a complete sample, for contrast with the rows
# below. Every value is a float except the rain state, which is a string here
# and `None` when its sample was dropped, the same as its five siblings.
FULL_READING = {
    "track_temp": 41.3,
    "air_temp": 23.7,
    "humidity": 55.0,
    "wind_speed": 2.4,
    "wind_direction": 180.0,
    "rain_state": "DRY",
}

# The five NUMERIC fields, which are the ones that used to crash. `rain_state`
# is nullable too and is covered on its own below, because its failure was the
# opposite kind: it rendered "WET" rather than raising.
#
# Melbourne 2025 carries zero NaN across 178 weather rows, which is the whole
# reason this was latent, so the input is built by hand rather than hunted for
# in data/raw/.
NULLABLE_FIELDS = (
    "track_temp",
    "air_temp",
    "humidity",
    "wind_speed",
    "wind_direction",
)


def _labelled(rows: list[tuple[str, str]]) -> dict[str, str]:
    return dict(rows)


def test_a_complete_reading_still_renders_its_numbers() -> None:
    """The fix must not turn a real measurement into a placeholder."""
    values = _labelled(_weather_rows(FULL_READING))
    assert values["Track"] == "41.3 C"
    assert values["Air"] == "23.7 C"
    assert values["Humidity"] == "55%"
    assert values["Wind"] == "2.4 km/h S"
    assert values["Rain"] == "DRY"


@pytest.mark.parametrize("field", NULLABLE_FIELDS)
def test_one_null_reading_does_not_raise(field: str) -> None:
    """The crash guard, one field at a time.

    Against the pre-#1087 code every case except ``wind_direction`` raises
    ``TypeError: unsupported format string passed to NoneType.__format__``,
    and ``wind_direction`` passes because ``_wind_dir`` already handled it.
    That asymmetry is the defect: one field of six carried the fix.
    """
    weather = dict(FULL_READING, **{field: None})
    _weather_rows(weather)


@pytest.mark.parametrize(
    ("field", "label"),
    [
        ("track_temp", "Track"),
        ("air_temp", "Air"),
        ("humidity", "Humidity"),
        ("wind_speed", "Wind"),
        ("wind_direction", "Wind"),
    ],
)
def test_a_null_reading_renders_as_absent(field: str, label: str) -> None:
    """The sentinel guard: absent data has to LOOK absent.

    Asserting ``"N/A"`` appears in the row, not merely that nothing raised.
    A fix that returned the old 18.0 C constant would satisfy the crash guard
    above and fail here, which is the point: a display constant the reader
    cannot tell apart from a measurement is the collision shape that produced
    the bug in the first place.
    """
    weather = dict(FULL_READING, **{field: None})
    assert "N/A" in _labelled(_weather_rows(weather))[label]


def test_every_reading_null_at_once() -> None:
    """A weather channel that dropped out entirely, not just one sample."""
    weather = dict.fromkeys(NULLABLE_FIELDS)
    weather["rain_state"] = "WET"
    values = _labelled(_weather_rows(weather))
    # No units on a quantity that was never measured, and ONE "N/A" on the wind
    # row rather than two around a unit. That row reading "N/A km/h N/A" passed
    # every string assertion here until the panel was actually drawn and looked at.
    assert values["Track"] == "N/A"
    assert values["Air"] == "N/A"
    assert values["Humidity"] == "N/A"
    assert values["Wind"] == "N/A"
    assert values["Rain"] == "WET"


def test_no_weather_at_all_claims_nothing() -> None:
    """The empty dict, which is an older cache or a session with no weather.

    Every field including the rain state, because "DRY" on a session whose
    weather was never loaded is a claim about the track rather than a display
    default. Before #1087 this rendered 45.0 C / 18.0 C / 55% / DRY.
    """
    values = _labelled(_weather_rows({}))
    assert set(values) == {"Track", "Air", "Humidity", "Wind", "Rain"}
    assert all("N/A" in value for value in values.values())


def test_a_dropped_rainfall_sample_does_not_announce_rain() -> None:
    """The sixth field, and the only one whose failure was silent.

    ``bool(float("nan"))`` is ``True``, so before the guard a NaN ``Rainfall``
    read as "WET" and the panel announced rain on a dry race. The five numeric
    fields crashed on a dropped sample, which is loud; this one asserted the
    opposite of the truth and kept rendering.

    Driven through the loader rather than the panel, because the panel was
    never where this lived: ``_weather_row_to_dict`` is what decides, and no
    test reached it until the opening gate found the row.
    """
    import pandas as pd

    from src.arcade.data import SessionLoader

    to_dict = SessionLoader._weather_row_to_dict
    complete = {
        "AirTemp": 23.7,
        "TrackTemp": 41.3,
        "Humidity": 55.0,
        "WindSpeed": 2.4,
        "WindDirection": 180.0,
    }
    dropped = to_dict(None, pd.Series(dict(complete, Rainfall=float("nan"))))
    assert dropped["rain_state"] is None, "a dropped rainfall sample must not assert a state"
    assert _labelled(_weather_rows(dropped))["Rain"] == "N/A"

    for stored, expected in ((True, "WET"), (False, "DRY")):
        real = to_dict(None, pd.Series(dict(complete, Rainfall=stored)))
        assert real["rain_state"] == expected, "a real reading still renders itself"


def test_the_helper_keeps_its_format_spec() -> None:
    """A present value is formatted, not stringified.

    Guards the one way the coalescing could be written and still be wrong:
    returning ``str(value)`` would render 23.7000000001 for a float that came
    off a resample, and every assertion above would still be about "N/A".
    """
    assert _reading(23.75, ".1f", " C") == "23.8 C"
    assert _reading(55.4, ".0f", "%") == "55%"
    # The unit rides with the number, so an absent reading carries neither.
    assert _reading(None, ".1f", " C") == "N/A"
