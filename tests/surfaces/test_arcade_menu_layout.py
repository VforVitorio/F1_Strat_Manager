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

from src.arcade.config import (
    MENU_FOCUS_PAD,
    MENU_GUTTER,
    MENU_HINT_FONT,
    MENU_ROW_HEIGHT,
    MENU_SCALE_MAX,
    MENU_SCALE_MIN,
    MENU_SUBTITLE_FONT,
    SCREEN_HEIGHT,
)
from src.arcade.views import (
    menu_bands,
    menu_content_extents,
    menu_form_geometry,
    menu_scale,
)

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


# --- The form scales with the window (#1100) -------------------------------

# Heights a user can actually produce. The window is resizable with no minimum,
# so the small ones are reached by dragging rather than hypothetical, and the
# large ones are a maximised window on a 1440p and a 4K display.
HEIGHTS = (480, 600, 720, 800, 900, 1080, 1400, 2160)

# Above this, a gap between two bands is no longer breathing room. Measured
# before the fix: 0.46 at 720 and 1.10 at 1080, from a form that never grew.
MAX_GAP_RATIO = 0.9

ROW_COUNT = len(ROW_WIDTHS)


def _unclamped(height: int) -> bool:
    """Whether the scale at this height is the raw ratio, neither clamp hit."""
    return MENU_SCALE_MIN < height / SCREEN_HEIGHT < MENU_SCALE_MAX


# Split rather than skipped inside the test. A skip reports as coverage while
# running nothing, and this file would have carried three of them.
UNCLAMPED_HEIGHTS = tuple(h for h in HEIGHTS if _unclamped(h))
CLAMPED_HEIGHTS = tuple(h for h in HEIGHTS if not _unclamped(h))


def test_the_default_window_is_left_exactly_as_it_was() -> None:
    """At SCREEN_HEIGHT the scale is 1 and every anchor is its old constant.

    The point of the fix is what happens AWAY from the default, so the default
    itself has to come out unchanged. Rendered offscreen at 1280x720 before and
    after, the two frames differ by zero pixels; these are the four numbers that
    make that true.
    """
    bands = menu_bands(SCREEN_HEIGHT, ROW_COUNT)
    assert bands.scale == 1.0
    assert bands.title_y == SCREEN_HEIGHT - 80
    assert bands.subtitle_y == SCREEN_HEIGHT - 112
    assert bands.hint_y == 60
    assert bands.row_pitch == 40
    assert bands.form_top == 460, "the first row sat at 460 before the fix"


@pytest.mark.parametrize("height", HEIGHTS)
def test_no_band_collides_with_another(height: int) -> None:
    """Title above subtitle above form above hint, at every reachable height.

    Clearance is one full font size, which is twice what separating two
    centre-anchored lines strictly needs, so this fails before anything touches.
    """
    bands = menu_bands(height, ROW_COUNT)
    form_top_edge = bands.form_top + bands.row_pitch / 2
    form_bottom_edge = bands.form_bottom - bands.row_pitch / 2

    assert bands.title_y > bands.subtitle_y
    assert bands.subtitle_y - form_top_edge >= MENU_SUBTITLE_FONT * bands.scale, (
        f"h={height}: the subtitle sits on the form"
    )
    assert form_bottom_edge - bands.hint_y >= MENU_HINT_FONT * bands.scale, (
        f"h={height}: the form sits on the hint"
    )
    assert bands.hint_y > 0


@pytest.mark.parametrize("height", UNCLAMPED_HEIGHTS)
def test_the_gaps_stay_proportional_to_the_form(height: int) -> None:
    """Extra window height goes into the form, not only into the two gaps.

    This is the assertion that is RED against the pre-#1100 layout, which pinned
    the title, the hint and a 40 px row pitch to constants: at 1920x1080 the
    form was still 280 px and the gap above it was 308, a ratio of 1.10.

    Only over the heights where the scale is the raw ratio. Outside that range a
    clamp holds the type at a readable size on purpose, and the height it
    declines to use goes back into the gaps, which
    `test_beyond_the_ceiling_the_void_returns_and_says_so` states outright.
    """
    bands = menu_bands(height, ROW_COUNT)
    gap_above = bands.subtitle_y - (bands.form_top + bands.row_pitch / 2)
    gap_below = (bands.form_bottom - bands.row_pitch / 2) - bands.hint_y

    assert gap_above / bands.form_height <= MAX_GAP_RATIO
    assert gap_below / bands.form_height <= MAX_GAP_RATIO


def test_the_sweep_exercises_both_clamps_and_the_range_between() -> None:
    """The split above is real: every one of the three regimes has a height."""
    scales = {menu_bands(h, ROW_COUNT).scale for h in HEIGHTS}
    assert MENU_SCALE_MIN in scales, "no swept height reaches the floor"
    assert MENU_SCALE_MAX in scales, "no swept height reaches the ceiling"
    assert UNCLAMPED_HEIGHTS, "the ratio guard would parametrize over nothing"
    assert CLAMPED_HEIGHTS, "the clamps would never be exercised"


@pytest.mark.parametrize("height", CLAMPED_HEIGHTS)
def test_beyond_the_ceiling_the_void_returns_and_says_so(height: int) -> None:
    """Naming the cost of the legibility cap rather than leaving it unmeasured.

    A 4K window is past the ceiling, so the form stops growing while the window
    does not, and the height goes back into the gaps. That is the same void the
    scaling exists to remove, kept deliberately so the type stays a readable
    size. The bands still may not collide, which the collision guard covers at
    the same heights.
    """
    bands = menu_bands(height, ROW_COUNT)
    assert bands.scale in (MENU_SCALE_MIN, MENU_SCALE_MAX)
    assert bands.form_height == pytest.approx(ROW_COUNT * MENU_ROW_HEIGHT * bands.scale)


def test_a_taller_window_gets_a_taller_form() -> None:
    """The plain statement of the defect, which a constant pitch cannot satisfy."""
    forms = [menu_bands(h, ROW_COUNT).form_height for h in UNCLAMPED_HEIGHTS]
    assert forms == sorted(forms)
    assert forms[-1] > forms[0]


def test_the_scale_is_clamped_at_both_ends() -> None:
    """A dragged-down window stops shrinking the type; a 4K one stops growing it."""
    assert menu_scale(SCREEN_HEIGHT) == 1.0
    assert menu_scale(100) == MENU_SCALE_MIN
    assert menu_scale(10_000) == MENU_SCALE_MAX


@pytest.mark.parametrize("rows", [1, 6, 7])
def test_the_form_block_keeps_its_place_whatever_the_row_count(rows: int) -> None:
    """One-driver mode hides the Rival row, so the count is not a constant.

    The block sits half a row below the window's vertical centre at every count,
    which is where it sat before the fix.
    """
    bands = menu_bands(SCREEN_HEIGHT, rows)
    block_centre = (bands.form_top + bands.form_bottom) / 2
    assert block_centre == SCREEN_HEIGHT / 2 - bands.row_pitch / 2
