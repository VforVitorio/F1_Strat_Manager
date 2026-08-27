"""The driver table carries every value the two cards did (#1102).

Two stacked cards repeated the same six labels under two headers and took 354 px
of the left column for twelve values, which is what left the controls legend
nowhere to draw at the default window height (#1096). One table with a column
per driver carries the same twelve in 177.

**Asserted on pure functions, not on a rendered card.** The strings are built
between GL calls in ``DriverInfoPanel.draw``, so a check on the drawn result
needs a display and a check that needs a display does not run in CI.
``_weather_rows`` was split out for exactly this reason and this file follows it.

The column arithmetic is checked against widths MEASURED off the live
``arcade.Text`` objects over 400 frames of a real race, listed in ``WIDEST``,
because what the font renders is the one thing a pure function cannot produce.
"""

from __future__ import annotations

import pytest

from src.arcade.config import (
    COMPOUND_COLORS,
    COMPOUND_LETTERS,
    DRIVER_BOX_WIDTH,
    DRIVER_LABEL_MIN,
    DRIVER_PAD_X,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
)
from src.arcade.overlays import (
    DRIVER_ROW_LABELS,
    driver_column_edges,
    driver_rows,
    driver_table,
    present_drivers,
)

# Rendered widths of the label and the widest value each row produced, read off
# the panel's own Text objects over EVERY 60th of Melbourne 2025's 125,279
# frames, for all twenty drivers, through the real gap pipeline. The gap rows
# are the wide ones: they carry the neighbour's code, a signed interval and the
# "(L)" suffix.
#
# Re-measured 2026-08-27. The first pass sampled 400 frames of two drivers and
# put Ahead at 103 px; the full race reaches 111.2. The subject of a measurement
# is part of the measurement, which is section 11 of CLAUDE.md (#1111).
WIDEST: dict[str, tuple[float, float]] = {
    "Speed": (35.8, 63.3),
    "Gear": (26.9, 8.1),
    "DRS": (23.8, 39.1),
    "Compound": (64.5, 13.4),
    "Ahead": (36.8, 111.2),
    "Behind": (39.8, 107.0),
}

# The widest value the SERVED distribution produces, which is what the columns
# are sized against. Named for what it is rather than "worst case": the seconds
# branch of `_gap_label` is uncapped, so a wider string is CONSTRUCTIBLE and
# does not fit. `test_a_three_digit_interval_is_a_known_bound_that_does_not_fit`
# records that rather than hiding it behind a comfortable constant (#1111).
WIDEST_SERVED_VALUE = 111.2

# A three-digit interval at the panel's font, measured: "DOO +123.45s (L)" is
# 116.9 px and the alphabet-worst "WWW -123.45s (L)" is 123.3.
CONSTRUCTIBLE_THREE_DIGIT_VALUE = 116.9


def _frame(speed: float = 232.0, gear: int = 6, drs: int = 0, tyre: int = 1) -> dict:
    return {"speed": speed, "gear": gear, "drs": drs, "tyre": tyre}


def test_a_driver_produces_the_six_rows_in_order() -> None:
    """The labels are the contract the table zips on, so their order is data."""
    rows = driver_rows(_frame(), "VER +2.36s (L)", "PIA -1.80s (L)")
    assert [label for label, _, _ in rows] == list(DRIVER_ROW_LABELS)


def test_every_value_the_two_cards_showed_is_still_shown() -> None:
    """The scope constraint, asserted rather than assumed.

    #1098 is explicit that nothing is added and nothing is removed: the same
    numbers are arranged once instead of twice.
    """
    rows = driver_rows(_frame(speed=269.4, gear=7, drs=12, tyre=2), "AHEAD", "BEHIND")
    values = {label: value for label, value, _ in rows}
    assert values == {
        "Speed": "269 km/h",
        "Gear": "7",
        "DRS": "ON",
        "Compound": COMPOUND_LETTERS[2],
        "Ahead": "AHEAD",
        "Behind": "BEHIND",
    }


def test_the_compound_keeps_its_own_colour() -> None:
    """Colour is the compound's identity on a timing screen, not decoration."""
    rows = driver_rows(_frame(tyre=2), "a", "b")
    colours = {label: colour for label, _, colour in rows}
    assert colours["Compound"] == COMPOUND_COLORS[2]
    assert colours["Speed"] == TEXT_PRIMARY
    assert colours["Ahead"] == TEXT_SECONDARY


def test_two_drivers_make_one_row_per_label_with_a_cell_each() -> None:
    """The shape of the whole change: twelve values under six labels, not two sets."""
    main = driver_rows(_frame(speed=232, gear=6), "VER +2.36s (L)", "PIA -1.80s (L)")
    rival = driver_rows(_frame(speed=269, gear=7), "HAM +3.80s (L)", "NOR -2.36s (L)")
    table = driver_table([main, rival])

    assert [label for label, _ in table] == list(DRIVER_ROW_LABELS)
    assert all(len(cells) == 2 for _, cells in table)
    by_label = {label: [value for value, _ in cells] for label, cells in table}
    assert by_label["Speed"] == ["232 km/h", "269 km/h"]
    assert by_label["Ahead"] == ["VER +2.36s (L)", "HAM +3.80s (L)"]

    flattened = [value for _, cells in table for value, _ in cells]
    assert len(flattened) == 12, "twelve values went in and twelve must come out"


def test_one_driver_still_makes_a_table() -> None:
    """One column is the same code path, which is why there is only one panel."""
    table = driver_table([driver_rows(_frame(), "a", "b")])
    assert [label for label, _ in table] == list(DRIVER_ROW_LABELS)
    assert all(len(cells) == 1 for _, cells in table)


def test_mismatched_rows_raise_instead_of_zipping_quietly() -> None:
    """A cell under the wrong label is the failure a silent zip would produce."""
    good = driver_rows(_frame(), "a", "b")
    shuffled = [good[1], good[0], *good[2:]]
    with pytest.raises(ValueError, match="labels disagree"):
        driver_table([good, shuffled])


def test_no_drivers_makes_an_empty_table_rather_than_raising() -> None:
    """`driver_table` is arithmetic; refusing a driver is the panel's job."""
    assert driver_table([]) == []


# --- The columns fit what the font actually renders ------------------------


def test_one_column_ends_where_the_single_card_used_to() -> None:
    """A single driver must draw exactly where it did before the table existed.

    The old card right-aligned its value at `x + width - PAD_X`, so a one-column
    table has to land on the same edge or the change is not the no-op it claims
    to be for that case.
    """
    (edge,) = driver_column_edges(DRIVER_BOX_WIDTH, 1)
    assert edge == DRIVER_BOX_WIDTH - DRIVER_PAD_X


def test_two_columns_are_equal_and_fill_the_card() -> None:
    """Equal columns, and the last one ends on the card's own padding."""
    first, second = driver_column_edges(DRIVER_BOX_WIDTH, 2)
    assert second == DRIVER_BOX_WIDTH - DRIVER_PAD_X
    assert second - first == first - (DRIVER_PAD_X + DRIVER_LABEL_MIN)


@pytest.mark.parametrize("columns", [1, 2])
def test_no_label_can_reach_the_value_beside_it(columns: int) -> None:
    """Checked per row, because a label only competes with its OWN row's value.

    This is why the label column is 40 px and not 64: `Compound` is the widest
    label at 64 and sits beside a single letter, while the 103 px gap values sit
    beside labels of 37 and 40. Reserving the widest label for every row would
    have cost 24 px the wide rows needed.
    """
    edges = driver_column_edges(DRIVER_BOX_WIDTH, columns)
    label_left = DRIVER_PAD_X
    for label, (label_width, value_width) in WIDEST.items():
        value_left = edges[0] - value_width
        assert label_left + label_width < value_left, (
            f"{label}: a {label_width} px label reaches a value starting at {value_left}"
        )


@pytest.mark.parametrize("columns", [1, 2])
def test_no_column_can_reach_the_one_before_it(columns: int) -> None:
    """A value is right-aligned, so it grows leftward into its neighbour.

    Against the widest the served distribution produces, not against a
    hypothetical: see `test_a_three_digit_interval_is_a_known_bound_that_does_not_fit`
    for the string that would not fit and why it is left that way.
    """
    edges = driver_column_edges(DRIVER_BOX_WIDTH, columns)
    previous_edge = DRIVER_PAD_X + DRIVER_LABEL_MIN
    for edge in edges:
        assert edge - WIDEST_SERVED_VALUE > previous_edge, (
            f"a {WIDEST_SERVED_VALUE} px value in the column ending at {edge} "
            f"overruns the one ending at {previous_edge}"
        )
        previous_edge = edge


def test_the_widest_measured_row_clears_by_a_readable_margin() -> None:
    """Not merely non-overlapping: the numbers stay separable at a glance.

    The binding row is `Ahead`, at 148.1 px of label plus value against a 118 px
    column plus a 40 px label column. The margin is 9.9 px, about one character.
    It was stated as 18 while the fixture came from a 400-frame sample; the full
    race costs 8 of those (#1111).
    """
    first, _ = driver_column_edges(DRIVER_BOX_WIDTH, 2)
    widest_pair = max(label + value for label, value in WIDEST.values())
    assert widest_pair == pytest.approx(148.0, abs=0.2)
    assert first - DRIVER_PAD_X - widest_pair >= 9


def test_a_three_digit_interval_is_a_known_bound_that_does_not_fit() -> None:
    """Recorded rather than asserted away, because the columns cannot hold it.

    The seconds branch of `_gap_label` fires whenever two cars are under one LAP
    OF DISTANCE apart, however large the interval, so a three-digit gap is
    constructible: it renders 116.9 px against a 118 px column, and the
    alphabet-worst code takes it to 123.3.

    Melbourne 2025 never produces one, and the widest it does produce clears by
    6.8 px. Fitting one would need a column of about 126 and a label column of
    36, so a card of 312 px against a `MARGIN_LEFT` of 330 that starts at 20:
    the circuit would pay 12 px of width for a string no served session
    contains. Left unfixed deliberately. If this ever fails because the
    constants moved, the trade has changed and is worth re-making, not silenced.
    """
    first, second = driver_column_edges(DRIVER_BOX_WIDTH, 2)
    column = second - first
    assert WIDEST_SERVED_VALUE < column, "the served distribution fits, which is the claim"
    assert CONSTRUCTIBLE_THREE_DIGIT_VALUE > column - 2, (
        "a three-digit interval now fits comfortably, so this bound is stale"
    )


def test_a_third_column_would_not_fit_and_the_test_says_so() -> None:
    """Bounding the generalisation rather than leaving it implied.

    The panel takes a sequence, so three drivers is expressible. It does not fit
    at this card width, and a future rival-comparison feature has to widen the
    card rather than assume the columns will divide.
    """
    edges = driver_column_edges(DRIVER_BOX_WIDTH, 3)
    column = edges[1] - edges[0]
    assert column < WIDEST_SERVED_VALUE, (
        "three columns now fit, so this bound is stale rather than the panel wrong"
    )


# --- One absent driver costs his column, not the table (#1110) -------------


def test_an_absent_driver_produces_six_absences_rather_than_nothing() -> None:
    """Absent data has to look absent, the rule `_weather_rows` follows (#1087).

    A column that simply vanished would change the table's width frame to frame
    while a driver dropped in and out of the feed.
    """
    rows = driver_rows(None, "N/A", "N/A")
    assert [label for label, _, _ in rows] == list(DRIVER_ROW_LABELS)
    assert {value for _, value, _ in rows} == {"N/A"}


def test_an_absent_column_is_dimmer_than_a_real_reading() -> None:
    """Colour is what separates a missing value from a measured one at a glance.

    `TEXT_PRIMARY` on an absent cell would read as data, which is the
    sentinel-shaped failure this repo keeps paying for: an unknown that cannot
    be told apart from a real value.
    """
    absent = {colour for _, _, colour in driver_rows(None, "N/A", "N/A")}
    assert absent == {TEXT_TERTIARY}
    assert TEXT_TERTIARY != TEXT_PRIMARY


def test_the_present_driver_keeps_his_telemetry_when_the_other_is_missing() -> None:
    """The regression, stated as the assertion that catches it.

    Two cards each guarded themselves, so an absent rival cost the rival card
    alone. The merged table returned from `draw` on the first driver without
    frame data, before the card was drawn at all, and took the other driver's
    live telemetry with it (#1110).
    """
    present = driver_rows(_frame(speed=232, gear=6), "VER +2.36s (L)", "PIA -1.80s (L)")
    table = driver_table([present, driver_rows(None, "N/A", "N/A")])

    assert len(table) == len(DRIVER_ROW_LABELS)
    assert all(len(cells) == 2 for _, cells in table)
    by_label = {label: [value for value, _ in cells] for label, cells in table}
    assert by_label["Speed"] == ["232 km/h", "N/A"]
    assert by_label["Ahead"] == ["VER +2.36s (L)", "N/A"]


def test_a_table_of_nobody_is_all_absences() -> None:
    """What the panel checks before deciding not to draw its card at all.

    Every followed driver missing is the one case where the whole card goes,
    which is what the two separate cards did between them.
    """
    table = driver_table([driver_rows(None, "N/A", "N/A")] * 2)
    values = {value for _, cells in table for value, _ in cells}
    assert values == {"N/A"}


@pytest.mark.parametrize(
    ("frame_drivers", "expected"),
    [
        ({"NOR": {"speed": 1}, "VER": {"speed": 2}}, ("NOR", "VER")),
        ({"NOR": {"speed": 1}}, ("NOR",)),
        ({"VER": {"speed": 2}}, ("VER",)),
        ({}, ()),
        (None, ()),
        ({"NOR": {}}, ()),
    ],
)
def test_presence_is_read_per_driver_not_all_or_nothing(
    frame_drivers: dict | None, expected: tuple[str, ...]
) -> None:
    """The panel draws its card whenever ANY followed driver is in the frame.

    The last case matters on its own: an empty dict for a driver is not a
    driver, and the old code's `if not data` treated it the same way. Whatever
    the frame says, one missing driver must not remove the other.
    """
    assert present_drivers(("NOR", "VER"), frame_drivers) == expected
