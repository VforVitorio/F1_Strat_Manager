"""UI panels for the Arcade race replay.

Five independent components consumed by `F1ArcadeWindow.on_draw`: weather,
leaderboard, driver info, progress bar, controls legend. Every `arcade.Text`
is pre-allocated in each panel's `__init__` (which runs after the Window's
GL context is active); `draw()` only mutates `.text / .x / .y / .color`.
Creating `Text` inside `draw()` would leak glyph textures at 60 FPS × 20
rows, a bug that bit both the reference and earlier attempts here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

import arcade
from src.arcade.config import (
    ACCENT,
    BORDER_COLOR,
    COMPOUND_COLORS,
    COMPOUND_LETTERS,
    CONTENT_BG,
    DRIVER_HEADER_HEIGHT,
    DRIVER_LABEL_MIN,
    DRIVER_PAD_X,
    DRIVER_ROW_GAP,
    DRS_ELIGIBLE_CODE,
    DRS_OPEN_CODES,
    FLAG_COLORS,
    FONT_BODY,
    FONT_TITLE,
    LEADERBOARD_N_SLOTS,
    LEADERBOARD_ROW_HEIGHT,
    LEADERBOARD_WIDTH,
    LEGEND_BOTTOM,
    LEGEND_X,
    PANEL_FILL_ALPHA,
    PROGRESS_BAR_BOTTOM,
    PROGRESS_BAR_HEIGHT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    WEATHER_LEFT,
    WEATHER_ROW_GAP,
    WEATHER_TOP_OFFSET,
    WEATHER_WIDTH,
)

# Re-exported so every existing caller keeps its import path. The rule itself
# moved to a module with no `arcade` import, because PITWALL's bulk reader
# decodes the status of each LAP and must not drag the GUI library in to do it -
# nor keep a second copy of the priority order.
from src.arcade.track_status import (  # noqa: F401
    neutralised_label,
    track_status_banner,
    track_status_label,
)

if TYPE_CHECKING:
    # Type-only so the drawing layer keeps its independence from the data
    # layer: importing it for real would pull fastf1 and pandas into every
    # module that draws a rectangle.
    from src.arcade.gaps import RaceGapCalculator

logger = logging.getLogger(__name__)

_COMPASS: Final[tuple[str, ...]] = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)


def _wind_dir(deg: float | None) -> str:
    if deg is None:
        return "N/A"
    return _COMPASS[int(((deg % 360) / 22.5) + 0.5) % 16]


def _reading(value: float | None, spec: str, unit: str = "") -> str:
    """Format one weather reading with its unit, or ``"N/A"`` when there is none.

    **The unit belongs to the number, so an absent reading carries neither.**
    Rendering "N/A C" puts a unit on a quantity that was never measured, and on
    the wind row, where two fields share one line, it produced "N/A km/h N/A".
    Every string assertion passed on that: it took looking at the drawn panel.

    ``SessionLoader._weather_row_to_dict`` writes an explicit ``None`` UNDER
    THE KEY when FastF1's reading is NaN, so ``dict.get(key, default)`` returns
    that ``None`` and the default never fires. This is the ``Series.get`` lesson
    of CLAUDE.md section 11 reproduced in a plain dict: the default covers a
    missing KEY, never a missing VALUE.

    ``"N/A"`` rather than the old display constant, and that is the point of the
    fix rather than an aesthetic choice. 18.0 C shown for an unknown air
    temperature is a number the reader cannot tell apart from a measurement,
    which is the sentinel-collision shape this repo keeps paying for. Absent
    data has to look absent.
    """
    if value is None:
        return "N/A"
    return f"{format(value, spec)}{unit}"


def _weather_rows(weather: dict) -> list[tuple[str, str]]:
    """The panel's five label/value pairs, as finished strings.

    Split out of ``WeatherPanel.draw`` so the rendered TEXT can be asserted
    without a GL context. The crash this closes lived in these five
    expressions and nowhere else in the draw call, and a check that needed a
    window to run is a check that does not run.

    Every field degrades the same way, which is what was missing: ``_wind_dir``
    already returned ``"N/A"`` on a missing direction while its five siblings
    formatted whatever they were handed, so one field of six carried the fix.
    """
    speed = weather.get("wind_speed")
    direction = weather.get("wind_direction")
    # One row, two readings, so it collapses to a single "N/A" only when BOTH
    # are missing. A known speed with an unknown bearing is a real state and
    # keeps saying so.
    if speed is None and direction is None:
        wind = "N/A"
    else:
        wind = f"{_reading(speed, '.1f', ' km/h')} {_wind_dir(direction)}"

    return [
        ("Track", _reading(weather.get("track_temp"), ".1f", " C")),
        ("Air", _reading(weather.get("air_temp"), ".1f", " C")),
        ("Humidity", _reading(weather.get("humidity"), ".0f", "%")),
        ("Wind", wind),
        # A string, so it takes the `or` rather than ``_reading``, but it is
        # nullable for the same reason as the five numbers: the loader stores
        # ``None`` when the ``Rainfall`` sample is missing. It did not always.
        # `bool(float("nan"))` is ``True``, so a dropped sample used to render
        # "WET" on a dry race, and that is worse than the crash the other five
        # fields took, because a wrong affirmative reading is silent.
        ("Rain", f"{weather.get('rain_state') or 'N/A'}"),
    ]


class WeatherPanel:
    """Top-left weather readout: track/air temp, humidity, wind, rain.

    Visual identity: translucent CONTENT_BG card with a 1 px BORDER outline and
    a 3 px ACCENT top-strip. Readings are Inter body text, label in TERTIARY
    and value in PRIMARY."""

    PANEL_PADDING: int = 12
    STRIP_H: int = 3

    def __init__(
        self,
        x: int = WEATHER_LEFT,
        top_offset: int = WEATHER_TOP_OFFSET,
        width: int = WEATHER_WIDTH,
    ) -> None:
        self.x = x
        self.top_offset = top_offset
        self.width = width
        self.bottom_y: int = 0
        self._title = arcade.Text(
            "WEATHER",
            x,
            0,
            ACCENT,
            13,
            bold=True,
            font_name=FONT_TITLE,
            anchor_x="left",
            anchor_y="top",
        )
        self._label = arcade.Text(
            "",
            0,
            0,
            TEXT_TERTIARY,
            11,
            font_name=FONT_BODY,
            anchor_x="left",
            anchor_y="top",
        )
        self._value = arcade.Text(
            "",
            0,
            0,
            TEXT_PRIMARY,
            11,
            font_name=FONT_BODY,
            bold=True,
            anchor_x="right",
            anchor_y="top",
        )

    def draw(self, frame: dict | None, window_height: int) -> None:
        weather = (frame or {}).get("weather") or {}
        top_y = window_height - self.top_offset
        rows = _weather_rows(weather)
        panel_h = 26 + len(rows) * WEATHER_ROW_GAP + self.PANEL_PADDING
        self._draw_card(top_y, panel_h)

        self._title.x = self.x + self.PANEL_PADDING
        self._title.y = top_y - 10
        self._title.draw()

        y = top_y - 32
        for label, value in rows:
            self._label.text = label
            self._label.x = self.x + self.PANEL_PADDING
            self._label.y = y
            self._label.draw()
            self._value.text = value
            self._value.x = self.x + self.width - self.PANEL_PADDING
            self._value.y = y
            self._value.draw()
            y -= WEATHER_ROW_GAP
        self.bottom_y = y + WEATHER_ROW_GAP - 10

    def _draw_card(self, top_y: int, panel_h: int) -> None:
        cx = self.x + self.width / 2
        cy = top_y - panel_h / 2
        arcade.draw_rect_filled(
            arcade.XYWH(cx, cy, self.width, panel_h), (*CONTENT_BG, PANEL_FILL_ALPHA)
        )
        arcade.draw_rect_outline(arcade.XYWH(cx, cy, self.width, panel_h), BORDER_COLOR, 1)
        strip_cy = top_y - self.STRIP_H / 2
        arcade.draw_rect_filled(arcade.XYWH(cx, strip_cy, self.width, self.STRIP_H), ACCENT)


DRIVER_ROW_LABELS: Final[tuple[str, ...]] = (
    "Speed",
    "Gear",
    "DRS",
    "Compound",
    "Ahead",
    "Behind",
)


def driver_rows(
    data: dict | None,
    ahead: str,
    behind: str,
) -> list[tuple[str, str, tuple[int, int, int]]]:
    """One driver's six label/value/colour rows, as finished strings.

    Split out of the panel's draw call for the reason ``_weather_rows`` was
    (#1087): the rendered TEXT is then assertable without a GL context, and a
    check that needs a window is a check that does not run in CI.

    ``data`` is ``None`` when the frame carries nothing for this driver, and the
    row set comes back as six ``N/A`` cells in the tertiary colour rather than
    as nothing. Two cards each guarded themselves, so an absent rival cost the
    rival card alone; the merged table returned before drawing anything and took
    the other driver's live telemetry with it (#1110). Absent data has to look
    absent, which is the same rule the weather panel's readings follow.
    """
    if data is None:
        return [(label, "N/A", TEXT_TERTIARY) for label in DRIVER_ROW_LABELS]
    tyre = int(data.get("tyre", 1))
    drs = data.get("drs", 0)
    return [
        ("Speed", f"{data.get('speed', 0):.0f} km/h", TEXT_PRIMARY),
        ("Gear", f"{data.get('gear', 0)}", TEXT_PRIMARY),
        ("DRS", DriverInfoPanel._drs_label(drs), DriverInfoPanel._drs_color(drs)),
        ("Compound", COMPOUND_LETTERS.get(tyre, "?"), COMPOUND_COLORS.get(tyre, TEXT_PRIMARY)),
        ("Ahead", ahead, TEXT_SECONDARY),
        ("Behind", behind, TEXT_SECONDARY),
    ]


def driver_table(
    per_driver: list[list[tuple[str, str, tuple[int, int, int]]]],
) -> list[tuple[str, list[tuple[str, tuple[int, int, int]]]]]:
    """Turn one row list per driver into one row per label, one cell per driver.

    The two drivers used to get a card each, so the same six labels were drawn
    twice under two headers: 354 px of column for twelve values (#1102). Here
    the label is written once and each driver contributes a cell to it.

    Raises when the row lists do not agree on their labels, because a table that
    silently zips mismatched rows is how a value ends up under the wrong name.
    """
    if not per_driver:
        return []
    labels = [label for label, _, _ in per_driver[0]]
    for rows in per_driver[1:]:
        if [label for label, _, _ in rows] != labels:
            raise ValueError(f"driver row labels disagree: {labels} vs {[r[0] for r in rows]}")
    return [
        (label, [(rows[i][1], rows[i][2]) for rows in per_driver]) for i, label in enumerate(labels)
    ]


def driver_column_edges(
    width: int,
    column_count: int,
    *,
    pad_x: int = DRIVER_PAD_X,
    label_min: int = DRIVER_LABEL_MIN,
) -> tuple[float, ...]:
    """Right edge of each value column, as an offset from the card's left edge.

    Columns are equal and share whatever the label column does not need. The
    minimum is 40 px rather than the width of the longest label, because a label
    only has to clear the value on its OWN row: "Compound" is the widest at 64 px
    and its value is a single letter, while "Ahead" is 37 px against a 103 px
    value. Measured over 400 frames of a real race, the widest label-plus-value
    pair on any row is 140 px, which one column of 118 plus a 40 px label
    column clears by 18.

    With one column this returns the card's own right padding, so a single
    driver draws exactly where it did before the table existed.
    """
    inner = width - 2 * pad_x
    column = (inner - label_min) / column_count
    return tuple(pad_x + label_min + column * (i + 1) for i in range(column_count))


def present_drivers(
    codes: Sequence[str],
    drivers_in_frame: Mapping[str, dict] | None,
) -> tuple[str, ...]:
    """Which of the followed drivers this frame actually carries.

    Pure, so the panel's one all-or-nothing decision is checkable without a GL
    context. It is the decision that regressed: two cards each guarded
    themselves and an absent rival cost the rival card, while the merged table
    returned before drawing anything and took the other driver's live telemetry
    with it (#1110). The card goes only when this comes back empty.
    """
    if not drivers_in_frame:
        return ()
    return tuple(code for code in codes if drivers_in_frame.get(code))


class DriverInfoPanel:
    """Telemetry table for one or two drivers: speed, gear, DRS, compound, gaps.

    One card with a value column per driver rather than a card each. Two cards
    carried the same six labels under two headers and 354 px of the left column
    for twelve values, which is what left the controls legend nowhere to draw at
    the default window height (#1096, #1102).

    Visual identity, unchanged: a neutral CONTENT_BG card with a 3 px
    team-colour strip on top and the driver code in team colour. With two
    drivers the strip is split at the boundary between their columns, so which
    colour belongs to which column is legible from the strip as well as from
    the code above it.
    """

    STRIP_H: int = 3
    PAD_X: int = DRIVER_PAD_X

    def __init__(
        self,
        x: int,
        top_y: int,
        width: int,
        height: int,
        drivers: Sequence[tuple[str, tuple[int, int, int]]],
    ) -> None:
        if not drivers:
            raise ValueError("a driver panel needs at least one driver")
        self.x = x
        self.top_y = top_y
        self.width = width
        self.height = height
        self.drivers = tuple(drivers)
        self.codes = tuple(code for code, _ in self.drivers)
        self._column_edges = driver_column_edges(width, len(self.drivers), pad_x=self.PAD_X)
        self._headers = [
            arcade.Text(
                code,
                0,
                0,
                color,
                15,
                bold=True,
                font_name=FONT_TITLE,
                anchor_x="right",
                anchor_y="center",
            )
            for code, color in self.drivers
        ]
        self._caption = arcade.Text(
            "DRIVERS" if len(self.drivers) > 1 else "DRIVER",
            0,
            0,
            TEXT_TERTIARY,
            9,
            bold=True,
            font_name=FONT_TITLE,
            anchor_x="left",
            anchor_y="center",
        )
        self._label = arcade.Text(
            "",
            0,
            0,
            TEXT_TERTIARY,
            10,
            font_name=FONT_BODY,
            anchor_x="left",
            anchor_y="center",
        )
        self._value = arcade.Text(
            "",
            0,
            0,
            TEXT_PRIMARY,
            11,
            font_name=FONT_BODY,
            bold=True,
            anchor_x="right",
            anchor_y="center",
        )

    @property
    def bottom_y(self) -> int:
        """Bottom edge of the card, which is what the legend measures against."""
        return self.top_y - self.height

    def set_top(self, top_y: int) -> None:
        self.top_y = top_y

    def draw(
        self,
        frame: dict,
        all_drivers_sorted: list[tuple[str, float | None]] | None,
        gaps: RaceGapCalculator,
        frame_idx: int,
    ) -> None:
        drivers_in_frame = frame.get("drivers") or {}
        per_driver: list[list[tuple[str, str, tuple[int, int, int]]]] = []
        for code in self.codes:
            data = drivers_in_frame.get(code) or None
            ahead, behind = self._neighbor_gaps(code, all_drivers_sorted, gaps, frame, frame_idx)
            per_driver.append(driver_rows(data, ahead, behind))
        # The card goes only when NOT ONE followed driver is in the frame, which
        # is what the two separate cards did between them (#1110).
        if not present_drivers(self.codes, drivers_in_frame):
            return

        self._draw_card()
        self._draw_header()
        y = self.top_y - DRIVER_HEADER_HEIGHT - 14
        for label, cells in driver_table(per_driver):
            self._label.text = label
            self._label.x = self.x + self.PAD_X
            self._label.y = y
            self._label.draw()
            for edge, (value, color) in zip(self._column_edges, cells, strict=True):
                self._value.text = value
                self._value.color = color
                self._value.x = self.x + edge
                self._value.y = y
                self._value.draw()
            y -= DRIVER_ROW_GAP

    def _draw_card(self) -> None:
        """The card body, its outline, and one strip segment per driver."""
        cx = self.x + self.width / 2
        cy = self.top_y - self.height / 2
        arcade.draw_rect_filled(
            arcade.XYWH(cx, cy, self.width, self.height), (*CONTENT_BG, PANEL_FILL_ALPHA)
        )
        arcade.draw_rect_outline(arcade.XYWH(cx, cy, self.width, self.height), BORDER_COLOR, 1)

        strip_cy = self.top_y - self.STRIP_H / 2
        left = float(self.x)
        for i, (_, color) in enumerate(self.drivers):
            last = i == len(self.drivers) - 1
            right = self.x + self.width if last else self.x + self._column_edges[i]
            arcade.draw_rect_filled(
                arcade.XYWH((left + right) / 2, strip_cy, right - left, self.STRIP_H),
                color,
            )
            left = right

    def _draw_header(self) -> None:
        """The caption on the left, each driver's code over its own column."""
        header_cy = self.top_y - DRIVER_HEADER_HEIGHT / 2
        self._caption.x = self.x + self.PAD_X
        self._caption.y = header_cy
        self._caption.draw()
        for header, edge in zip(self._headers, self._column_edges, strict=True):
            header.x = self.x + edge
            header.y = header_cy
            header.draw()

    @staticmethod
    def _drs_label(drs: int) -> str:
        """ON / AVAIL / OFF, from the codes' single home in `config`.

        These two methods each held their own `(10, 12, 14)` literal. The commit
        that gave the open set one home moved the track overlay and the wire and
        left this pair behind, so for one commit the window had three copies of a
        set whose whole point was to have one.
        """
        drs = int(drs)
        if drs in DRS_OPEN_CODES:
            return "ON"
        if drs == DRS_ELIGIBLE_CODE:
            return "AVAIL"
        return "OFF"

    @staticmethod
    def _drs_color(drs: int) -> tuple[int, int, int]:
        drs = int(drs)
        if drs in DRS_OPEN_CODES:
            return (0, 220, 0)
        if drs == DRS_ELIGIBLE_CODE:
            return (255, 210, 50)
        return TEXT_TERTIARY

    def _neighbor_gaps(
        self,
        code: str,
        sorted_drivers: list[tuple[str, float | None]] | None,
        gaps: RaceGapCalculator,
        frame: dict,
        frame_idx: int,
    ) -> tuple[str, str]:
        """The intervals either side of `code`, as the timing screen shows them.

        Takes the code rather than reading one off the panel, because the panel
        now holds a column per driver instead of belonging to one.
        """
        if not sorted_drivers:
            return "N/A", "N/A"
        codes = [c for c, _ in sorted_drivers]
        if code not in codes:
            return "N/A", "N/A"
        # Inactive is not retired. A finisher's telemetry ends the moment he
        # takes the flag, so reading `active` alone put "NOR OUT" on the
        # winner's neighbours from the instant he won, and 19 of 20 rows
        # read OUT at the final frame (#855).
        retired = {
            code
            for code, data in (frame.get("drivers") or {}).items()
            if not (data or {}).get("active", True) and not gaps.has_finished(code)
        }
        idx = codes.index(code)
        me = sorted_drivers[idx]
        ahead = (
            "LEADER"
            if idx == 0
            else self._gap_label("+", sorted_drivers[idx - 1], me, gaps, frame_idx, retired)
        )
        behind = (
            "LAST"
            if idx == len(codes) - 1
            else self._gap_label("-", me, sorted_drivers[idx + 1], gaps, frame_idx, retired)
        )
        return ahead, behind

    @staticmethod
    def _gap_label(
        sign: str,
        front: tuple[str, float | None],
        back: tuple[str, float | None],
        gaps: RaceGapCalculator,
        frame_idx: int,
        retired: set[str],
    ) -> str:
        """Label the interval between the car in front and the car behind it.

        `sign` is the direction from the panel's own driver: "+" when the
        pair is (someone ahead, me), "-" when it is (me, someone behind).
        The arithmetic is identical either way, which is the point of
        passing the pair rather than a signed distance.

        Four outcomes, in the order they are decided:

        - **OUT** when the neighbour has retired. `np.interp` clamps past a
          driver's last sample, so a parked car keeps reporting its final
          state forever; without this branch the panel rendered a stale
          interval, up to 22 minutes old, naming a car that stopped. The
          leaderboard row already says OUT, so this only makes the two
          agree.
        - **+N LAP(S)** when the car in front is more than a full lap of
          track ahead, the way a timing screen shows it.
        - **seconds with an "(L)" suffix**, measured at the last line both
          cars crossed. The suffix is not decoration: an unlabelled number
          on a fidelity surface implies a liveness this one does not have.
        - **N/A** when any input is unknown, never a plausible-looking
          substitute.
        """
        front_code, front_progress = front
        back_code, back_progress = back
        other_code = back_code if sign == "-" else front_code

        if other_code in retired:
            return f"{other_code} OUT"

        laps = gaps.laps_down(front_progress, back_progress)
        if laps is None:
            return f"{other_code} N/A"
        if laps >= 1:
            return f"{other_code} {sign}{laps} LAP" + ("S" if laps > 1 else "")

        lap = gaps.last_shared_lap(front_code, back_code, frame_idx)
        seconds = gaps.interval_at_line(front_code, back_code, lap)
        if seconds is None:
            return f"{other_code} N/A"
        return f"{other_code} {sign}{seconds:.2f}s (L)"


class LeaderboardPanel:
    """Right-edge list of all drivers ranked by race-cumulative progress.

    Visual identity: same card language as Weather/DriverInfo, translucent
    CONTENT_BG, 1 px BORDER outline, 3 px ACCENT top-strip, rank numbers in
    TERTIARY, codes in team colour, compound letter in compound colour on the
    right edge. Selected row is filled with SECONDARY_BG instead of a bare
    grey rect."""

    STRIP_H: int = 3
    PAD_X: int = 10
    HEADER_H: int = 28

    def __init__(
        self,
        x: int,
        top_y: int,
        width: int = LEADERBOARD_WIDTH,
        n_slots: int = LEADERBOARD_N_SLOTS,
    ) -> None:
        self.x = x
        self.top_y = top_y
        self.width = width
        # Read by RaceEventsPanel (positioned right under the leaderboard) so
        # the new event card stays glued to the leaderboard's bottom edge as
        # the row count changes between sessions / DNFs.
        self.bottom_y: int = top_y
        self._row_rects: list[tuple[str, float, float, float, float]] = []
        self._title = arcade.Text(
            "LEADERBOARD",
            x,
            top_y,
            ACCENT,
            13,
            bold=True,
            font_name=FONT_TITLE,
            anchor_x="left",
            anchor_y="top",
        )
        self._rank_texts = [
            arcade.Text(
                "", 0, 0, TEXT_TERTIARY, 11, font_name=FONT_BODY, anchor_x="left", anchor_y="top"
            )
            for _ in range(n_slots)
        ]
        self._code_texts = [
            arcade.Text(
                "",
                0,
                0,
                TEXT_PRIMARY,
                12,
                bold=True,
                font_name=FONT_BODY,
                anchor_x="left",
                anchor_y="top",
            )
            for _ in range(n_slots)
        ]
        self._compound_texts = [
            arcade.Text(
                "",
                0,
                0,
                TEXT_PRIMARY,
                11,
                bold=True,
                font_name=FONT_BODY,
                anchor_x="right",
                anchor_y="top",
            )
            for _ in range(n_slots)
        ]

    def set_top(self, top_y: int) -> None:
        self.top_y = top_y

    def draw(
        self,
        frame: dict,
        driver_colors: dict[str, tuple[int, int, int]],
        gaps: RaceGapCalculator,
        frame_idx: int,
        selected_drivers: set[str] | None = None,
    ) -> None:
        selected_drivers = selected_drivers or set()
        ranked = self._rank_drivers(frame, gaps, frame_idx)
        n_rows = min(len(ranked), len(self._rank_texts))
        panel_h = self.HEADER_H + n_rows * LEADERBOARD_ROW_HEIGHT + 8
        self.bottom_y = self.top_y - panel_h
        self._draw_card(panel_h)

        self._title.x = self.x + self.PAD_X
        self._title.y = self.top_y - 8
        self._title.draw()

        self._row_rects = []
        y = self.top_y - self.HEADER_H

        for i, (code, data, _) in enumerate(ranked[:n_rows]):
            color = driver_colors.get(code, TEXT_PRIMARY)
            is_highlighted = code in selected_drivers
            rect_cx = self.x + self.width / 2
            rect_cy = y - LEADERBOARD_ROW_HEIGHT / 2 + 6
            self._row_rects.append(
                (
                    code,
                    self.x,
                    rect_cy - LEADERBOARD_ROW_HEIGHT / 2,
                    self.x + self.width,
                    rect_cy + LEADERBOARD_ROW_HEIGHT / 2,
                )
            )
            if is_highlighted:
                arcade.draw_rect_filled(
                    arcade.XYWH(rect_cx, rect_cy, self.width, LEADERBOARD_ROW_HEIGHT),
                    (*ACCENT, 70),
                )

            # Same rule as the neighbour label: a car that took the flag is
            # inactive and is not out.
            is_out = not data.get("active", True) and not gaps.has_finished(code)
            rt = self._rank_texts[i]
            rt.text = f"{i + 1:>2}"
            rt.color = TEXT_TERTIARY
            rt.x = self.x + self.PAD_X
            rt.y = y
            rt.draw()

            ct = self._code_texts[i]
            ct.text = f"{code}{' OUT' if is_out else ''}"
            ct.color = color
            ct.x = self.x + self.PAD_X + 28
            ct.y = y
            ct.draw()

            compound = int(data.get("tyre", 1))
            pt = self._compound_texts[i]
            pt.text = COMPOUND_LETTERS.get(compound, "?")
            pt.color = COMPOUND_COLORS.get(compound, TEXT_PRIMARY)
            pt.x = self.x + self.width - self.PAD_X
            pt.y = y
            pt.draw()

            y -= LEADERBOARD_ROW_HEIGHT

    def _draw_card(self, panel_h: int) -> None:
        cx = self.x + self.width / 2
        cy = self.top_y - panel_h / 2
        arcade.draw_rect_filled(
            arcade.XYWH(cx, cy, self.width, panel_h), (*CONTENT_BG, PANEL_FILL_ALPHA)
        )
        arcade.draw_rect_outline(arcade.XYWH(cx, cy, self.width, panel_h), BORDER_COLOR, 1)
        strip_cy = self.top_y - self.STRIP_H / 2
        arcade.draw_rect_filled(arcade.XYWH(cx, strip_cy, self.width, self.STRIP_H), ACCENT)

    def sorted_progress(
        self, frame: dict, gaps: RaceGapCalculator, frame_idx: int
    ) -> list[tuple[str, float | None]]:
        return [
            (code, progress) for code, _, progress in self._rank_drivers(frame, gaps, frame_idx)
        ]

    def hit_test(self, mx: float, my: float) -> str | None:
        for code, left, bottom, right, top in self._row_rects:
            if left <= mx <= right and bottom <= my <= top:
                return code
        return None

    @staticmethod
    def _rank_drivers(
        frame: dict, gaps: RaceGapCalculator, frame_idx: int
    ) -> list[tuple[str, dict, float | None]]:
        """Rank the field by race progress: laps completed plus fraction of the lap.

        **Not by `dist`.** `FrameData.dist` is race-cumulative metres and
        looks like a progress axis, which is why the old code sorted on it
        (after adding `(lap - 1) * track_len` to a value that already
        contained the completed laps). It is not one: each car accumulates
        the distance IT drove, so two cars at the same corner hold
        different numbers and the drift reaches 1877 m on a 5220 m
        circuit. Measured under the convention stated at the top of
        `gaps.py` (do not restate figures here under a different one,
        which is how this docstring and that one came to publish 0.7% and
        2.0% for the same quantity): a descending `dist` sort puts the
        wrong car in the lead on 37% of sampled frames; this key gets it
        wrong on 1.7% and reproduces the whole running order exactly on
        236 of 300.

        A car whose progress is unknown sorts last and carries `None`
        rather than a position it does not have. It is still drawn, because
        a car with no position data is still on the track.

        That claim used to name `RelativeDistance` as the cause and was
        wrong twice: `progress` does not read `RelativeDistance` at all,
        and the value it actually returned for such a car was 0.0, not
        None - the same number every car reads on the grid. The signal is
        `SessionData.has_position`, and it is honoured now (#886).

        **The tie-break is not decoration.** Every car that has finished
        sits at exactly the total laps, so from the leader's flag to the
        end of the replay the whole podium ties and a plain sort falls back
        to dict insertion order. Ordering ties by who crossed first is what
        makes those last seconds - the frame the replay parks on, and the
        one a viewer reads as the result - the actual classification.
        """
        drivers = (frame or {}).get("drivers") or {}
        ranked: list[tuple[str, dict, float | None]] = []
        unknown: list[tuple[str, dict, float | None]] = []
        for code, data in drivers.items():
            progress = gaps.progress(code, frame_idx)
            (unknown if progress is None else ranked).append((code, data, progress))
        ranked.sort(
            key=lambda entry: (entry[2], -gaps.last_crossing_frame(entry[0], frame_idx)),
            reverse=True,
        )
        return ranked + unknown


class RaceEventsPanel:
    """Coloured pill that announces non-clear track status under the leaderboard.

    Reads the FastF1 ``TrackStatus`` digit string for the current lap (cached
    on ``SessionData.track_status_by_lap`` by ``SessionLoader``) and renders
    a Safety Car / VSC / Yellow / Red flag banner the same way a TV broadcast
    would.  The card is hidden when the status is clear (``"1"`` or empty)
    and fades in / out over ~0.35 s on transitions so consecutive laps with
    the same status do not flicker.

    Multi-digit codes are parsed with priority ``red > SC > VSC > yellow``,
    matching how race control announces concurrent events (a red flag wins
    even if a yellow was already out in another sector).
    """

    HEIGHT: int = 36
    GAP_FROM_LEADERBOARD: int = 12
    FADE_PER_SECOND: float = 255.0 / 0.35  # ~0.35 s linear fade

    def __init__(
        self,
        x: int,
        top_y: int,
        width: int = LEADERBOARD_WIDTH,
    ) -> None:
        self.x = x
        self.top_y = top_y
        self.width = width
        self._label = ""
        self._color: tuple[int, int, int] = (255, 255, 255)
        self._alpha: float = 0.0
        self._target_alpha: float = 0.0
        self._text = arcade.Text(
            "",
            x + width // 2,
            top_y - self.HEIGHT // 2,
            (255, 255, 255),
            14,
            bold=True,
            font_name=FONT_TITLE,
            anchor_x="center",
            anchor_y="center",
        )

    def set_top(self, top_y: int) -> None:
        """Reposition the card after the leaderboard's bottom edge moves."""
        self.top_y = top_y

    def update(self, dt: float, track_status: str | None) -> None:
        """Re-evaluate the visible status and advance the fade animation.

        ``dt`` is the seconds elapsed since the last update; pyglet feeds
        the same value it already passes to ``F1ArcadeView.on_update``.
        Pass an empty string / ``None`` for clear.
        """
        status = track_status_banner(track_status or "")
        if status is None:
            self._target_alpha = 0.0
        else:
            self._label, self._color = status
            self._target_alpha = 255.0
        self._tick_alpha(dt)

    def draw(self) -> None:
        """Render the card; bails out cheaply when fully transparent."""
        if self._alpha < 1.0 or not self._label:
            return
        alpha = int(self._alpha)
        bg = (self._color[0], self._color[1], self._color[2], alpha)
        rect = arcade.LBWH(self.x, self.top_y - self.HEIGHT, self.width, self.HEIGHT)
        arcade.draw_rect_filled(rect, bg)
        arcade.draw_rect_outline(rect, (255, 255, 255, alpha), 1)
        self._text.x = self.x + self.width // 2
        self._text.y = self.top_y - self.HEIGHT // 2
        self._text.text = self._label
        self._text.color = (255, 255, 255, alpha)
        self._text.draw()

    def _tick_alpha(self, dt: float) -> None:
        delta = self.FADE_PER_SECOND * max(dt, 0.0)
        if self._alpha < self._target_alpha:
            self._alpha = min(self._target_alpha, self._alpha + delta)
        elif self._alpha > self._target_alpha:
            self._alpha = max(self._target_alpha, self._alpha - delta)


class ProgressBar:
    """Bottom timeline with lap ticks, flag events, playhead, and click-to-seek."""

    def __init__(
        self,
        total_frames: int,
        total_laps: int,
        events: list[dict[str, Any]] | None = None,
        left_margin: int = 340,
        right_margin: int = 260,
        bottom: int = PROGRESS_BAR_BOTTOM,
        height: int = PROGRESS_BAR_HEIGHT,
    ) -> None:
        self.total_frames = max(1, int(total_frames))
        self.total_laps = max(1, int(total_laps))
        self.events = events or []
        self.left_margin = left_margin
        self.right_margin = right_margin
        self.bottom = bottom
        self.height = height
        self._bar_left = left_margin
        self._bar_width = 1
        self._lap_label = arcade.Text(
            "1", 0, 0, TEXT_TERTIARY, 10, font_name=FONT_BODY, anchor_x="center", anchor_y="top"
        )

    def on_resize(self, window_width: int) -> None:
        self._bar_width = max(100, window_width - self.left_margin - self.right_margin)

    def draw(self, window_width: int, current_frame: int) -> None:
        self.on_resize(window_width)
        cy = self.bottom + self.height / 2
        bg_rect = arcade.XYWH(
            self._bar_left + self._bar_width / 2, cy, self._bar_width, self.height
        )
        arcade.draw_rect_filled(bg_rect, FLAG_COLORS["background"])
        arcade.draw_rect_outline(bg_rect, FLAG_COLORS["lap_marker"], 1)

        prog = max(0.0, min(1.0, current_frame / self.total_frames))
        fill_w = prog * self._bar_width
        if fill_w > 0:
            arcade.draw_rect_filled(
                arcade.XYWH(self._bar_left + fill_w / 2, cy, fill_w, self.height - 4),
                FLAG_COLORS["progress_fill"],
            )

        for lap in range(1, self.total_laps + 1):
            lx = self._frame_to_x(int(lap / self.total_laps * self.total_frames))
            arcade.draw_line(
                lx,
                self.bottom + 2,
                lx,
                self.bottom + self.height - 2,
                FLAG_COLORS["lap_marker"],
                1,
            )
            if lap == 1 or lap == self.total_laps or lap % 10 == 0:
                self._lap_label.text = str(lap)
                self._lap_label.x = lx
                self._lap_label.y = self.bottom - 4
                self._lap_label.draw()

        for event in self.events:
            self._draw_event(event)

        px = self._frame_to_x(int(current_frame))
        arcade.draw_line(
            px,
            self.bottom - 2,
            px,
            self.bottom + self.height + 2,
            FLAG_COLORS["playhead"],
            3,
        )

    def on_mouse_press(self, x: float, y: float) -> int | None:
        if not (self._bar_left <= x <= self._bar_left + self._bar_width):
            return None
        if not (self.bottom - 5 <= y <= self.bottom + self.height + 5):
            return None
        return self._x_to_frame(x)

    def _frame_to_x(self, f: int) -> float:
        f = max(0, min(f, self.total_frames))
        return self._bar_left + (f / self.total_frames) * self._bar_width

    def _x_to_frame(self, x: float) -> int:
        return int(
            max(
                0,
                min(
                    self.total_frames - 1,
                    ((x - self._bar_left) / max(1, self._bar_width)) * self.total_frames,
                ),
            )
        )

    def _draw_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type", "")
        color = FLAG_COLORS.get(event_type)
        if color is None:
            return
        sf = int(event.get("frame", 0))
        ef = int(event.get("end_frame", sf + 100))
        sx = self._frame_to_x(sf)
        ex = self._frame_to_x(ef)
        w = max(4.0, ex - sx)
        arcade.draw_rect_filled(arcade.XYWH(sx + w / 2, self.bottom + self.height + 5, w, 5), color)


# Row pitch and the header's own band, shared by the span helper and the draw
# loop so the two cannot disagree about how tall the legend is.
LEGEND_ROW_PITCH: Final[int] = 14
LEGEND_HEADER_BAND: Final[int] = 18


def legend_span(row_count: int) -> int:
    """Pixels the legend occupies above its own bottom edge.

    Data, not a drawing side effect, because the decision below has to be made
    before anything is drawn and has to be checkable without a GL context.
    """
    return row_count * LEGEND_ROW_PITCH + LEGEND_HEADER_BAND


def legend_mode(space_below: int | None, row_count: int, *, forced_open: bool = False) -> str:
    """``"full"`` or ``"hint"``, from the room the column actually left.

    **The whole point of this function is that it is pure** (#1096). The
    geometry it decides on lives between GL calls in ``on_draw``, so a check on
    the drawn result needs a window, and a check that needs a window does not
    run in CI. `_weather_rows` was split out for the same reason.

    ``space_below`` is the gap between the legend's bottom edge and the lowest
    thing already drawn above it, or ``None`` when the caller has nothing above
    to measure against.

    The list spans 158 px. When this function was written the column above it
    was two stacked driver cards and left 146 at the default 720 with two
    drivers, so the list could not fit there at ANY anchor, which is why
    clamping it under the lowest card was not the fix. #1102 replaced those
    cards with one table and the column now leaves 263 at that height, so the
    collapse fires below roughly 615 px of window instead of below 788.

    ``forced_open`` wins: a user who pressed the key asked for the panel and can
    dismiss it again, so an overlap they summoned is theirs to make.
    """
    if forced_open:
        return "full"
    if space_below is None:
        return "full"
    return "full" if space_below >= legend_span(row_count) else "hint"


class ControlsLegend:
    """Bottom-left cheat sheet for keyboard bindings, collapsible.

    Uses the same ACCENT title / TERTIARY body convention as the other
    panels so the legend reads as part of the UI instead of a debug
    overlay.

    **It collapses to one line when the column above it has no room** rather
    than drawing over the driver card, which is what it used to do at the
    default window height with two drivers. The hint line names the key that
    brings it back, so nothing is hidden without saying where it went."""

    HINT_KEY: Final[str] = "C"
    HINT_TEXT: Final[str] = "Controls"

    LINES: Final[tuple[tuple[str, str], ...]] = (
        ("SPACE", "Pause / Resume"),
        ("<- / ->", "Rewind / Fast-Forward"),
        ("Up / Down", "Speed +/-"),
        ("1 - 4", "0.5 / 1 / 2 / 4 x"),
        ("R", "Restart"),
        ("A", "Toggle all 20 cars"),
        ("D", "Toggle DRS zones"),
        ("B", "Toggle progress bar"),
        ("C", "Toggle this list"),
        ("ESC", "Close"),
    )

    def __init__(self, x: int = LEGEND_X, bottom: int = LEGEND_BOTTOM) -> None:
        self.x = x
        self.bottom = bottom
        # Set by the view's `C` branch. False means "decide from the room",
        # which is the state the window opens in.
        self.forced_open = False
        self._header = arcade.Text(
            "CONTROLS",
            x,
            0,
            ACCENT,
            12,
            bold=True,
            font_name=FONT_TITLE,
            anchor_x="left",
            anchor_y="bottom",
        )
        self._key_texts = [
            arcade.Text(
                key,
                0,
                0,
                TEXT_PRIMARY,
                10,
                bold=True,
                font_name=FONT_BODY,
                anchor_x="left",
                anchor_y="bottom",
            )
            for key, _ in self.LINES
        ]
        self._desc_texts = [
            arcade.Text(
                desc,
                0,
                0,
                TEXT_TERTIARY,
                10,
                font_name=FONT_BODY,
                anchor_x="left",
                anchor_y="bottom",
            )
            for _, desc in self.LINES
        ]
        # Pre-allocated like every other Text in this module: creating one
        # inside draw() leaks a glyph texture per frame.
        self._hint_key = arcade.Text(
            self.HINT_KEY,
            0,
            0,
            TEXT_PRIMARY,
            10,
            bold=True,
            font_name=FONT_BODY,
            anchor_x="left",
            anchor_y="bottom",
        )
        self._hint_desc = arcade.Text(
            self.HINT_TEXT,
            0,
            0,
            TEXT_TERTIARY,
            10,
            font_name=FONT_BODY,
            anchor_x="left",
            anchor_y="bottom",
        )

    def toggle(self) -> None:
        self.forced_open = not self.forced_open

    def draw(self, space_below: int | None = None) -> None:
        """Draw the full list, or the one-line hint when there is no room.

        ``space_below`` is how much vertical room the column above left free.
        ``None`` keeps the old unconditional behaviour, which is what a caller
        with nothing above the legend wants.
        """
        if legend_mode(space_below, len(self.LINES), forced_open=self.forced_open) == "hint":
            self._hint_key.x = self.x
            self._hint_key.y = self.bottom
            self._hint_key.draw()
            self._hint_desc.x = self.x + 70
            self._hint_desc.y = self.bottom
            self._hint_desc.draw()
            return

        y = self.bottom
        rows = list(zip(self._key_texts, self._desc_texts))
        for i, (key, desc) in enumerate(reversed(rows)):
            key.x = self.x
            key.y = y + i * LEGEND_ROW_PITCH
            key.draw()
            desc.x = self.x + 70
            desc.y = y + i * LEGEND_ROW_PITCH
            desc.draw()
        self._header.x = self.x
        self._header.y = self.bottom + len(self.LINES) * LEGEND_ROW_PITCH + 6
        self._header.draw()
