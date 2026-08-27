"""The circuit gets the space the panels actually leave (#1103).

`MARGIN_LEFT` reserved 340 px for a left column 320 px wide and `TRACK_PADDING`
took another 5% off each side, so the trace drew 612 px wide in a 1280 px window
while the free band between the panels was 700. The circuit is the reason this
window exists and it was the smallest it could be.

**Asserted on pure functions, not on a rendered frame.** `update_scaling` runs
between GL calls, so a check on the drawn result needs a display and a check
that needs a display does not run in CI. `legend_mode` (#1096) and
`_weather_rows` (#1087) were split out for the same reason.

The numbers pinned in `test_the_usable_width_matches_what_was_measured_on_screen`
were read off the projected polylines of the real Melbourne 2025 track on a
hidden window, so if the arithmetic here drifts from what is drawn, it fails
there rather than silently changing what everything else is about.
"""

from __future__ import annotations

import pytest

from src.arcade.config import (
    CAR_LABEL_MAX_HALF_WIDTH,
    DRIVER_BOX_WIDTH,
    LEADERBOARD_RIGHT_MARGIN,
    LEFT_PANEL_X,
    PROGRESS_BAR_BOTTOM,
    PROGRESS_BAR_HEIGHT,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    WEATHER_WIDTH,
)
from src.arcade.track import track_inset, track_viewport

# Window sizes a user can produce: the default, a dragged-smaller window, and
# maximised on a 1080p and a 1440p display.
SIZES = ((1000, 600), (1280, 720), (1600, 900), (1920, 1080), (2560, 1400))

# Right edge of the widest thing in the left column. The weather card is
# narrower than the driver table, so the table is what the margin has to clear.
LEFT_COLUMN_RIGHT = LEFT_PANEL_X + max(DRIVER_BOX_WIDTH, WEATHER_WIDTH)

# The most the viewport may reserve beyond what the panels occupy. Enough to
# keep the trace off the cards, not enough to hide a stale constant: it was 20
# px before the fix, from a 340 px margin over a 320 px column.
MAX_WASTED_INSET = 12


def _usable(width: int, height: int) -> tuple[float, float]:
    viewport = track_viewport(width, height)
    inset_x, inset_y = track_inset(viewport)
    return viewport.width - 2 * inset_x, viewport.height - 2 * inset_y


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_the_viewport_never_overlaps_a_panel(width: int, height: int) -> None:
    """The trace has to stop where the cards start, at every window size."""
    viewport = track_viewport(width, height)
    assert viewport.left >= LEFT_COLUMN_RIGHT, "the trace would draw under the driver table"
    assert viewport.right <= width - LEADERBOARD_RIGHT_MARGIN, "under the leaderboard"
    assert viewport.bottom >= PROGRESS_BAR_BOTTOM + PROGRESS_BAR_HEIGHT, "under the progress bar"
    assert viewport.top <= height


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_the_viewport_does_not_reserve_space_no_panel_uses(width: int, height: int) -> None:
    """The assertion that is RED against the old margin.

    `MARGIN_LEFT` was 340 against a column whose right edge is 320, so 20 px
    came off the circuit for nothing. A margin is allowed to be a round number
    only while it stays close to the thing it clears.
    """
    viewport = track_viewport(width, height)
    assert viewport.left - LEFT_COLUMN_RIGHT <= MAX_WASTED_INSET
    assert (width - LEADERBOARD_RIGHT_MARGIN) - viewport.right == 0


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_a_car_label_can_never_reach_a_panel(width: int, height: int) -> None:
    """What clears the panels is the LABEL, not the polyline.

    Cars are drawn on the trace with a three-letter code centred above or below
    the dot, and the widest such label renders 42 px. RED without the absolute
    floor under `TRACK_PADDING`: the fraction alone gives 13.8 px at 1280 and
    less below it, so a code at the trace's rightmost point would paint into the
    leaderboard.
    """
    inset_x, inset_y = track_inset(track_viewport(width, height))
    assert inset_x >= CAR_LABEL_MAX_HALF_WIDTH, f"{width}x{height}: a label overruns sideways"
    assert inset_y >= CAR_LABEL_MAX_HALF_WIDTH, f"{width}x{height}: a label overruns vertically"


def test_a_bigger_window_draws_a_bigger_circuit() -> None:
    """Plainly the point, and not true of a viewport pinned to constants."""
    widths = [_usable(w, h)[0] for w, h in SIZES]
    heights = [_usable(w, h)[1] for w, h in SIZES]
    assert widths == sorted(widths)
    assert heights == sorted(heights)
    assert widths[-1] > widths[0] * 2


def test_the_usable_width_matches_what_was_measured_on_screen() -> None:
    """Pins this file's arithmetic to the trace's real drawn bounds.

    A pure test of a duplicated formula is the shape where the test and the code
    agree with each other and neither agrees with the window. These were read
    off the projected Melbourne polylines after `on_draw` on a hidden window.
    The trace is limited by the viewport's WIDTH at every size here, so its drawn
    width is the usable width exactly.
    """
    assert _usable(1280, 720)[0] == 646
    assert _usable(1920, 1080)[0] == pytest.approx(1276.6, abs=0.5)


def test_the_default_window_gained_what_the_fix_claims() -> None:
    """612 px before, 646 after, in the window the arcade opens at.

    Both halves stated so the claim is checkable rather than remembered: 10 px
    from the margin that reserved space no panel used, and 24 from the padding
    fraction, which is what the clearance floor then gives 22 of back.
    """
    assert (SCREEN_WIDTH, SCREEN_HEIGHT) == (1280, 720)
    assert _usable(SCREEN_WIDTH, SCREEN_HEIGHT)[0] - 612 == 34


def test_the_circuit_sits_in_the_middle_of_the_band_the_panels_leave() -> None:
    """Centred on the free space, which is not the window's own centre.

    The left column is 320 px and the leaderboard reserves 260, so the free band
    is not symmetric and neither is the circuit's place in it. #1103 read the
    offset as the trace being "pushed toward the right edge"; it is the correct
    consequence of asymmetric panels, and what would be wrong is centring on the
    window instead.
    """
    for width, height in SIZES:
        viewport = track_viewport(width, height)
        free_centre = (LEFT_COLUMN_RIGHT + (width - LEADERBOARD_RIGHT_MARGIN)) / 2
        assert abs(viewport.centre_x - free_centre) <= MAX_WASTED_INSET / 2
