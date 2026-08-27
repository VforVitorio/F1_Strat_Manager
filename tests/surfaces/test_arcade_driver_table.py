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
)
from src.arcade.overlays import (
    DRIVER_ROW_LABELS,
    driver_column_edges,
    driver_rows,
    driver_table,
)

# Rendered widths of the label and the widest value each row produced, measured
# 2026-08-27 over 400 frames of Melbourne 2025 with NOR and VER, read off the
# panel's own Text objects. The gap rows are the wide ones: they carry the
# neighbour's code, a signed interval and the "(L)" suffix.
WIDEST: dict[str, tuple[float, float]] = {
    "Speed": (36, 63),
    "Gear": (27, 8),
    "DRS": (24, 39),
    "Compound": (64, 6),
    "Ahead": (37, 103),
    "Behind": (40, 100),
}

# A gap can be wider than anything that race produced: three digits of seconds
# and a four-character code. At the ~7 px per character the measurements imply,
# that is about this.
WORST_CASE_VALUE = 115.0


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
    """A value is right-aligned, so it grows leftward into its neighbour."""
    edges = driver_column_edges(DRIVER_BOX_WIDTH, columns)
    previous_edge = DRIVER_PAD_X + DRIVER_LABEL_MIN
    for edge in edges:
        assert edge - WORST_CASE_VALUE > previous_edge, (
            f"a {WORST_CASE_VALUE} px value in the column ending at {edge} "
            f"overruns the one ending at {previous_edge}"
        )
        previous_edge = edge


def test_the_widest_measured_row_clears_by_a_readable_margin() -> None:
    """Not merely non-overlapping: the numbers stay separable at a glance.

    The binding row is either gap row, at 140 px of label plus value against a
    118 px column plus a 40 px label column.
    """
    first, _ = driver_column_edges(DRIVER_BOX_WIDTH, 2)
    widest_pair = max(label + value for label, value in WIDEST.values())
    assert widest_pair == 140
    assert first - DRIVER_PAD_X - widest_pair >= 12


def test_a_third_column_would_not_fit_and_the_test_says_so() -> None:
    """Bounding the generalisation rather than leaving it implied.

    The panel takes a sequence, so three drivers is expressible. It does not fit
    at this card width, and a future rival-comparison feature has to widen the
    card rather than assume the columns will divide.
    """
    edges = driver_column_edges(DRIVER_BOX_WIDTH, 3)
    column = edges[1] - edges[0]
    assert column < WORST_CASE_VALUE, (
        "three columns now fit, so this bound is stale rather than the panel wrong"
    )
