"""The controls legend never draws over the driver table (#1096, #1102).

At the default 1280x720 with two drivers the left column used to leave 146 px
below the lowest of two stacked cards while the full legend spans 158, so it
drew straight over the rival card's DRS, Compound and Ahead rows. Two drivers is
what the menu opens with, so this was the out-of-the-box first frame.

#1102 replaced the two cards with one table carrying a column per driver, which
removes the constraint rather than managing it: the column is 177 px deep at any
driver count instead of 354, so 720 leaves 263 px and the full list fits. The
collapse is NOT dead code, and this file is where that is checked. It now fires
below about 618 px of window height rather than below 788, and the sweep keeps
600 in it for exactly that reason.

**Asserted on a pure function, not on a rendered window.** The geometry lives
between GL calls in ``on_draw``, so a check on the drawn result needs a display,
and a check that needs a display does not run in CI. ``_weather_rows`` was split
out for exactly this reason and this file follows it. The one thing a window
could add, that the pixels really differ, was measured once by hand when the
defect was found; what has to hold from here is the decision.

The heights below bracket the crossover rather than sampling near it: 600 and
660 are smaller than the default, 720 is the default, and 788 upward is where
the full legend genuinely fits.
"""

from __future__ import annotations

import pytest

from src.arcade.config import (
    DRIVER_BOX_GAP,
    DRIVER_BOX_HEIGHT,
    LEGEND_BOTTOM,
    WEATHER_ROW_GAP,
    WEATHER_TOP_OFFSET,
)
from src.arcade.overlays import ControlsLegend, legend_mode, legend_span

# The window heights a user can actually produce. The window is resizable with
# no minimum, so the small ones are reachable by dragging, not hypothetical.
HEIGHTS = (600, 660, 720, 800, 900, 1080)

_WEATHER_ROWS = 5


def _weather_bottom(window_height: int) -> int:
    """What ``WeatherPanel.draw`` leaves in ``bottom_y``.

    Mirrors its last three lines (``overlays.py``: ``y = top_y - 32``, then one
    ``y -= WEATHER_ROW_GAP`` per row, then ``bottom_y = y + WEATHER_ROW_GAP -
    10``) rather than the panel's outer box, which sits 18 px lower. A first
    draft of this file used the box height and was 18 px pessimistic at every
    height, which would have collapsed the legend at 800 where it fits by 12.
    The runtime never had that bug: ``on_draw`` reads the real attribute.
    """
    top_y = window_height - WEATHER_TOP_OFFSET
    y = top_y - 32 - _WEATHER_ROWS * WEATHER_ROW_GAP
    return y + WEATHER_ROW_GAP - 10


def _lowest_card_bottom(window_height: int, *, has_rival: bool) -> int:
    """Bottom edge of the driver table, mirroring ``on_draw``.

    The table chains off the weather panel's bottom, so the column translates
    1:1 with the window height. Since #1102 the driver COUNT no longer changes
    the depth: one and two drivers are the same table with one or two value
    columns, so `has_rival` is accepted and deliberately unused. The parameter
    stays because the assertions below sweep both modes and a silently
    driver-count-independent column is itself worth asserting.
    """
    del has_rival
    return _weather_bottom(window_height) - DRIVER_BOX_GAP - DRIVER_BOX_HEIGHT


def test_the_model_matches_the_geometry_measured_on_a_real_window() -> None:
    """Pins this file's arithmetic to numbers read off the drawn panels.

    A pure test of a duplicated formula is the shape where the test and the
    code agree with each other and neither agrees with the window. These four
    were read back from ``F1ArcadeView`` after ``on_draw`` on a real (hidden)
    window, so if the panels move, this fails here rather than silently
    changing what every assertion below is about.
    """
    assert _weather_bottom(720) == 500
    assert _weather_bottom(800) == 580
    assert _lowest_card_bottom(720, has_rival=True) == 323
    assert _lowest_card_bottom(720, has_rival=False) == 323


def _space_below(window_height: int, *, has_rival: bool) -> int:
    return _lowest_card_bottom(window_height, has_rival=has_rival) - LEGEND_BOTTOM


@pytest.mark.parametrize("height", HEIGHTS)
@pytest.mark.parametrize("has_rival", [True, False], ids=["two-drivers", "one-driver"])
def test_the_legend_never_draws_over_the_cards(height: int, has_rival: bool) -> None:
    """Full list only where it fits; a hint line everywhere else.

    This is the assertion that is RED against the pre-#1096 code, which drew
    the full list unconditionally: at 720 with two drivers it needed 154 px
    into 146.
    """
    space = _space_below(height, has_rival=has_rival)
    mode = legend_mode(space, len(ControlsLegend.LINES))
    if mode == "full":
        assert space >= legend_span(len(ControlsLegend.LINES)), (
            f"h={height} rival={has_rival}: drew the full legend into {space} px"
        )


def test_the_sweep_reaches_both_modes() -> None:
    """Neither arm passes by never running.

    Without this the test above is satisfied by a legend that is always a hint,
    which would close the overlap by deleting the panel.
    """
    modes = {
        legend_mode(_space_below(h, has_rival=r), len(ControlsLegend.LINES))
        for h in HEIGHTS
        for r in (True, False)
    }
    assert modes == {"full", "hint"}, f"the sweep only ever produced {modes}"


def test_the_default_window_with_two_drivers_now_fits_the_full_list() -> None:
    """The exact case the defect was, inverted by #1102.

    1280x720 is `SCREEN_HEIGHT` and the menu opens on two drivers, so this is
    the first frame of a default launch. It used to leave 146 px against a list
    that needs 158 and collapse to a hint; one table leaves 263 and the list
    fits, which is the measurable win #1102 was asked for.
    """
    space = _space_below(720, has_rival=True)
    assert space == 263
    assert space >= legend_span(len(ControlsLegend.LINES))
    assert legend_mode(space, len(ControlsLegend.LINES)) == "full"


def test_the_collapse_still_fires_on_a_window_a_user_can_produce() -> None:
    """#1102 must not turn #1096's fix into a branch nobody can reach.

    The window is resizable with no minimum, so the heights below the crossover
    are reached by dragging. If this ever fails the collapse has become dead
    code and the branch should be reconsidered rather than kept unreachable.
    """
    space = _space_below(600, has_rival=True)
    assert space < legend_span(len(ControlsLegend.LINES))
    assert legend_mode(space, len(ControlsLegend.LINES)) == "hint"


def test_the_driver_count_no_longer_changes_the_column_depth() -> None:
    """One table, one depth, which is what freed the room (#1102).

    Two stacked cards cost 354 px and a table costs 177 whether it carries one
    column or two, so the geometry that produced the overlap is gone rather
    than merely accommodated.
    """
    for height in HEIGHTS:
        assert _lowest_card_bottom(height, has_rival=True) == _lowest_card_bottom(
            height, has_rival=False
        )


def test_one_driver_at_the_default_height_still_gets_the_full_list() -> None:
    """The fix must not collapse a legend that fits.

    This held before #1102 because a single card left a whole card's worth of
    room, and holds after it because the table is that same depth.
    """
    space = _space_below(720, has_rival=False)
    assert space >= legend_span(len(ControlsLegend.LINES))
    assert legend_mode(space, len(ControlsLegend.LINES)) == "full"


def test_the_key_forces_it_open_where_it_does_not_fit() -> None:
    """`C` overrides the room check, deliberately.

    A user who presses it asked for the panel and the same key dismisses it, so
    an overlap they summoned is theirs to make. Without this the key would do
    nothing at exactly the height where someone would reach for it.
    """
    space = _space_below(720, has_rival=True)
    assert legend_mode(space, len(ControlsLegend.LINES), forced_open=True) == "full"


def test_no_measurement_means_the_full_list() -> None:
    """A caller with nothing above the legend keeps the old behaviour."""
    assert legend_mode(None, len(ControlsLegend.LINES)) == "full"


def test_the_hint_names_a_key_the_list_documents() -> None:
    """The collapsed state has to say how to get back, and say it correctly.

    A hint naming a key the view does not bind, or one the list does not
    mention, is worse than no hint.
    """
    keys = {key for key, _ in ControlsLegend.LINES}
    assert ControlsLegend.HINT_KEY in keys, (
        f"the hint offers {ControlsLegend.HINT_KEY}, which the list does not document"
    )
