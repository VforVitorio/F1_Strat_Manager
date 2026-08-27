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
    MENU_EMPHASIS,
    MENU_FOCUS_PAD,
    MENU_GROUP_GAP,
    MENU_GUTTER,
    MENU_HINT_FONT,
    MENU_ROW_HEIGHT,
    MENU_SCALE_MAX,
    MENU_SCALE_MIN,
    MENU_SUBTITLE_FONT,
    SCREEN_HEIGHT,
)
from src.arcade.views import (
    MENU_HINT_SEPARATOR,
    MENU_HINTS,
    LaunchConfig,
    build_menu_fields,
    menu_bands,
    menu_content_extents,
    menu_form_geometry,
    menu_hint_line,
    menu_row_offsets,
    menu_scale,
    round_label,
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

# The real seven rows, read off the table the view draws rather than restated
# here, so a group reassigned in views.py reaches every assertion below.
FIELDS = build_menu_fields(lambda: None)
GROUPS = tuple(f.group for f in FIELDS)
ROW_COUNT = len(GROUPS)
GROUP_BREAKS = sum(1 for a, b in zip(GROUPS, GROUPS[1:]) if a != b)

# Heights a user can actually produce. The window is resizable with no minimum,
# so the small ones are reached by dragging rather than hypothetical, and the
# large ones are a maximised window on a 1440p and a 4K display.
HEIGHTS = (480, 600, 720, 800, 900, 1080, 1400, 2160)

# Above this, a gap between two bands is no longer breathing room. Measured
# before the fix: 0.46 at 720 and 1.10 at 1080, from a form that never grew.
MAX_GAP_RATIO = 0.9


def _unclamped(height: int) -> bool:
    """Whether the scale at this height is the raw ratio, neither clamp hit."""
    return MENU_SCALE_MIN < height / SCREEN_HEIGHT < MENU_SCALE_MAX


# Split rather than skipped inside the test. A skip reports as coverage while
# running nothing, and this file would have carried three of them.
UNCLAMPED_HEIGHTS = tuple(h for h in HEIGHTS if _unclamped(h))
CLAMPED_HEIGHTS = tuple(h for h in HEIGHTS if not _unclamped(h))


def test_the_default_window_keeps_the_anchors_the_scaling_did_not_move() -> None:
    """At SCREEN_HEIGHT the scale is 1, so the three edge bands are untouched.

    The point of #1100 is what happens AWAY from the default, so the default's
    own title, subtitle, hint and pitch have to come out at their old constants.
    The form's block is the one thing that did move, and only because #1101 puts
    a gap at each group boundary, so it is asserted separately below.
    """
    bands = menu_bands(SCREEN_HEIGHT, GROUPS)
    assert bands.scale == 1.0
    assert bands.title_y == SCREEN_HEIGHT - 80
    assert bands.subtitle_y == SCREEN_HEIGHT - 112
    assert bands.hint_y == 60
    assert bands.row_pitch == 40


def test_the_group_gaps_are_the_only_thing_that_moved_the_form() -> None:
    """The block is taller by exactly the gaps, and still centred where it was.

    Before either change the first row sat at 460 with a 280 px block. The block
    grows by one `MENU_GROUP_GAP` per boundary and stays centred half a row below
    the window's middle, so the first row rises by half of what was added.
    """
    bands = menu_bands(SCREEN_HEIGHT, GROUPS)
    added = GROUP_BREAKS * MENU_GROUP_GAP
    assert bands.form_height == ROW_COUNT * MENU_ROW_HEIGHT + added
    assert bands.form_top == 460 + added / 2


@pytest.mark.parametrize("height", HEIGHTS)
def test_no_band_collides_with_another(height: int) -> None:
    """Title above subtitle above form above hint, at every reachable height.

    Clearance is one full font size, which is twice what separating two
    centre-anchored lines strictly needs, so this fails before anything touches.
    """
    bands = menu_bands(height, GROUPS)
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
    bands = menu_bands(height, GROUPS)
    gap_above = bands.subtitle_y - (bands.form_top + bands.row_pitch / 2)
    gap_below = (bands.form_bottom - bands.row_pitch / 2) - bands.hint_y

    assert gap_above / bands.form_height <= MAX_GAP_RATIO
    assert gap_below / bands.form_height <= MAX_GAP_RATIO


def test_the_sweep_exercises_both_clamps_and_the_range_between() -> None:
    """The split above is real: every one of the three regimes has a height."""
    scales = {menu_bands(h, GROUPS).scale for h in HEIGHTS}
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
    bands = menu_bands(height, GROUPS)
    assert bands.scale in (MENU_SCALE_MIN, MENU_SCALE_MAX)
    assert bands.form_height == pytest.approx(
        (ROW_COUNT * MENU_ROW_HEIGHT + GROUP_BREAKS * MENU_GROUP_GAP) * bands.scale
    )


def test_a_taller_window_gets_a_taller_form() -> None:
    """The plain statement of the defect, which a constant pitch cannot satisfy."""
    forms = [menu_bands(h, GROUPS).form_height for h in UNCLAMPED_HEIGHTS]
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
    bands = menu_bands(SCREEN_HEIGHT, GROUPS[:rows])
    block_centre = (bands.form_top + bands.form_bottom) / 2
    assert block_centre == SCREEN_HEIGHT / 2 - bands.row_pitch / 2


# --- The rows are grouped and one of them carries weight (#1101) -----------


def test_the_seven_rows_fall_into_the_three_decisions_they_are() -> None:
    """Which race, which cars, whether the agents run.

    Named here so a row added to the wrong group fails rather than quietly
    joining the one above it.
    """
    by_group = {f.key: f.group for f in FIELDS}
    assert by_group == {
        "year": "race",
        "round": "race",
        "mode": "cars",
        "driver_main": "cars",
        "driver_rival": "cars",
        "team": "cars",
        "strategy": "pipeline",
    }


def test_the_groups_are_contiguous_in_draw_order() -> None:
    """A group split across the form would put a gap inside itself.

    `menu_row_offsets` inserts a gap wherever consecutive rows differ, so a
    table ordered race, cars, race would draw three bands for two groups.
    """
    seen: list[str] = []
    for group in GROUPS:
        if not seen or seen[-1] != group:
            assert group not in seen, f"{group} is split across the form"
            seen.append(group)
    assert seen == ["race", "cars", "pipeline"]


def test_a_gap_sits_at_every_boundary_and_nowhere_else() -> None:
    """The pitch between two rows says whether they are the same kind."""
    offsets = menu_row_offsets(GROUPS, pitch=40, group_gap=22)
    steps = [b - a for a, b in zip(offsets, offsets[1:])]
    expected = [40 + (22 if a != b else 0) for a, b in zip(GROUPS, GROUPS[1:])]
    assert steps == expected
    assert steps.count(62) == GROUP_BREAKS == 2


def test_one_driver_mode_loses_a_row_but_not_a_boundary() -> None:
    """Hiding the rival row must not merge the cars group into its neighbours.

    The rival row is the only one with a `visible` predicate and it sits inside
    its own group, so the form gets shorter by one row and keeps both gaps.
    """
    one_driver = LaunchConfig(mode_two_drivers=False)
    visible = tuple(f.group for f in FIELDS if f.visible(one_driver))

    assert len(visible) == ROW_COUNT - 1
    breaks = sum(1 for a, b in zip(visible, visible[1:]) if a != b)
    assert breaks == GROUP_BREAKS

    bands = menu_bands(SCREEN_HEIGHT, visible)
    assert bands.form_height == (ROW_COUNT - 1) * MENU_ROW_HEIGHT + breaks * MENU_GROUP_GAP


def test_exactly_one_row_is_emphasised_and_it_is_the_pipeline_switch() -> None:
    """The switch that decides whether the multi-agent layer runs at all.

    Emphasis is only worth anything while it is scarce, so the count is asserted
    as well as which row carries it.
    """
    emphasised = [f.key for f in FIELDS if f.emphasis]
    assert emphasised == ["strategy"]
    assert MENU_EMPHASIS > 1.0


def test_the_boundary_between_two_bindings_is_visible() -> None:
    """Four printed marks for five pairs, read off the line rather than the constant.

    A first draft of this asserted `menu_hint_line().split(MENU_HINT_SEPARATOR)`,
    which is circular: swapping the separator back to three spaces still splits
    into five, so the guard passed against the exact run-on it was written for.
    It counts marks in the rendered string now, and whitespace is not a mark.
    """
    line = menu_hint_line()
    marks = [c for c in line if not (c.isalnum() or c.isspace() or c == "/")]
    assert len(marks) == len(MENU_HINTS) - 1, f"{line!r} has no visible boundaries"
    assert MENU_HINT_SEPARATOR.strip(), "a whitespace-only separator is the defect"


def test_the_hint_line_holds_every_binding_in_order() -> None:
    """The separator may change without the contract it separates changing."""
    parts = menu_hint_line().split(MENU_HINT_SEPARATOR)
    assert parts == [f"{key} {action}" for key, action in MENU_HINTS]
    for part in parts:
        assert "  " not in part, f"{part!r} still runs two tokens together"


def test_the_hint_names_the_keys_the_menu_actually_reads() -> None:
    """A hint documenting a key the view does not bind is worse than none."""
    keys = {key for key, _ in MENU_HINTS}
    assert keys == {"UP/DOWN", "LEFT/RIGHT", "Type", "ENTER", "ESC"}


@pytest.mark.parametrize("year", [2023, 2024, 2025])
def test_the_round_value_is_two_tokens_for_every_round_of_every_season(year: int) -> None:
    """No leading space and no double space, at one digit or two.

    The old `%2d` put a leading space on rounds 1 to 9, and the value column is
    left-aligned, so those nine indented one space further than the rest. Swept
    over the real calendars rather than over the default round.
    """
    for round_ in range(1, 24):
        value = round_label(year, round_)
        assert value == value.lstrip(), f"{year} R{round_}: {value!r} has a leading space"
        assert "  " not in value, f"{year} R{round_}: {value!r} has a double space"
        assert value.split(" ", 1)[0] == str(round_)


def test_the_round_value_still_names_the_circuit() -> None:
    """Trimming the spacing must not trim the half a reader actually reads.

    Both digit widths, and two different seasons, because the round-to-circuit
    map is per year: Melbourne opened 2025 at round 1 and was round 3 in 2024,
    while 2025 round 3 was Suzuka.
    """
    assert round_label(2025, 1) == "1 Melbourne"
    assert round_label(2024, 3) == "3 Melbourne"
    assert round_label(2025, 3) == "3 Suzuka"
    assert round_label(2024, 23) == "23 Lusail"


def test_an_unknown_round_says_so_rather_than_inventing_a_circuit() -> None:
    """Absent data has to look absent, per the weather panel's `N/A` (#1087)."""
    assert round_label(2025, 99).endswith("?")
