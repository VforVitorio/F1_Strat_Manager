"""Guards on the launch menu's layout arithmetic.

The menu's geometry is decided between GL calls in ``MenuView._draw_fields``, so
a check on the drawn result needs a window and a check that needs a window does
not run in CI. Every decision is therefore split into a pure function and
checked here, the same seam ``legend_mode`` (#1096) and ``_weather_rows``
(#1087) were cut for.

``ROW_WIDTHS`` is the one piece of data a pure function cannot produce: what the
font actually renders. It was read off the live ``arcade.Text`` objects at
1280x720 after ``on_draw`` ran, so it is a measurement, not an estimate.
"""

from __future__ import annotations

import pytest

from src.arcade.config import MENU_FOCUS_PAD, MENU_GUTTER
from src.arcade.views import menu_content_extents, menu_form_geometry

# (row, rendered label width, rendered value width), measured 2026-08-27 at
# 1280x720 with the menu's default LaunchConfig, seven rows visible.
ROW_WIDTHS: list[tuple[str, float, float]] = [
    ("Year", 43, 46),
    ("Round", 62, 131),
    ("Mode", 51, 99),
    ("Driver", 61, 44),
    ("Rival", 48, 33),
    ("Team", 47, 81),
    ("Strategy", 83, 36),
]


def _extents_for(rows: list[tuple[str, float, float]]) -> tuple[float, float]:
    return menu_content_extents([r[1] for r in rows], [r[2] for r in rows])


def test_band_contains_every_row_it_can_highlight() -> None:
    """No row's text may stick out of the band drawn behind it.

    The failure this catches is a band fitted to one row: the focus can land on
    any of the seven, so a band that only holds the row it was sized on crops
    the others.
    """
    left, right = _extents_for(ROW_WIDTHS)
    for name, label_w, value_w in ROW_WIDTHS:
        assert MENU_GUTTER + label_w <= left, f"{name} label overflows the band's left edge"
        assert MENU_GUTTER + value_w <= right, f"{name} value overflows the band's right edge"


def test_band_is_sized_to_the_content_and_not_to_a_constant() -> None:
    """The band may exceed the widest row by the padding and by nothing else.

    This is the assertion the old code fails. It drew a fixed 540 px rectangle
    with a 460 px rule under a form whose content spans 254 px, so the excess
    was 143 px on the right where the padding allows 24 (#1099).
    """
    left, right = _extents_for(ROW_WIDTHS)
    widest_label = max(r[1] for r in ROW_WIDTHS)
    widest_value = max(r[2] for r in ROW_WIDTHS)

    assert left - (MENU_GUTTER + widest_label) == 0
    assert right - (MENU_GUTTER + widest_value) == 0

    fill_left, fill_right = left + MENU_FOCUS_PAD, right + MENU_FOCUS_PAD
    assert fill_left - (MENU_GUTTER + widest_label) <= MENU_FOCUS_PAD
    assert fill_right - (MENU_GUTTER + widest_value) <= MENU_FOCUS_PAD


def test_band_tracks_the_widest_row_when_a_value_grows() -> None:
    """A longer GP name widens the band, which a constant could not do.

    ``Round`` renders the round number plus the circuit's name, so its width is
    session data rather than a layout choice. Melbourne is 131 px; the longest
    name on the 2025 calendar is longer.
    """
    before_left, before_right = _extents_for(ROW_WIDTHS)
    longer = [(n, lw, vw * 2 if n == "Round" else vw) for n, lw, vw in ROW_WIDTHS]
    after_left, after_right = _extents_for(longer)

    assert after_right > before_right
    assert after_left == before_left, "a wider value must not move the label column"


def test_the_two_halves_are_measured_independently() -> None:
    """The band is not symmetric, and forcing it to be would waste one side.

    The widest label (STRATEGY) and the widest value (the round's GP name) are
    on different rows and are not the same width, so a single half-width would
    have to take the larger of the two and pad the other side with it.
    """
    left, right = _extents_for(ROW_WIDTHS)
    assert left != right


def test_no_visible_rows_collapses_to_the_gutter() -> None:
    """An empty form still has a centre axis, so the extents are the gutter.

    ``max()`` on an empty sequence raises, and the visible-row list is filtered
    by a predicate, so this is reachable rather than theoretical.
    """
    assert menu_content_extents([], []) == (MENU_GUTTER, MENU_GUTTER)


@pytest.mark.parametrize("gutter", [0, 20, 60])
def test_the_gutter_is_added_to_both_sides(gutter: int) -> None:
    """The columns are anchored off the axis, so the gutter is inside the band."""
    left, right = menu_content_extents([40.0], [90.0], gutter=gutter)
    assert (left, right) == (gutter + 40.0, gutter + 90.0)


def test_the_band_is_centred_on_the_window_axis() -> None:
    """The fill and the rule are symmetric about the window's centre.

    A symmetric band over an asymmetric form only works because the form's own
    axis is shifted to compensate. Before the band was tightened it was 540 px
    wide and hid the shift; at 302 px the form would read visibly off-centre
    without it.
    """
    geometry = menu_form_geometry([r[1] for r in ROW_WIDTHS], [r[2] for r in ROW_WIDTHS])
    left, right = _extents_for(ROW_WIDTHS)

    assert geometry.axis_offset == pytest.approx((left - right) / 2)
    assert geometry.axis_offset < 0, "the value column is the wider one, so the axis moves left"
    # Content edges measured from the window axis, which is where they are drawn.
    content_left = geometry.axis_offset - left
    content_right = geometry.axis_offset + right
    assert content_left == pytest.approx(-geometry.rule_half)
    assert content_right == pytest.approx(geometry.rule_half)


def test_the_fill_takes_the_padding_and_the_rule_does_not() -> None:
    """The two are drawn from the same content, one inset from the other."""
    geometry = menu_form_geometry([r[1] for r in ROW_WIDTHS], [r[2] for r in ROW_WIDTHS])
    assert geometry.band_half - geometry.rule_half == MENU_FOCUS_PAD


def test_a_symmetric_form_needs_no_axis_shift() -> None:
    """Equal columns leave the boundary on the window axis, where it started."""
    geometry = menu_form_geometry([60.0], [60.0])
    assert geometry.axis_offset == 0
