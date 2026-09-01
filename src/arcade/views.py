"""Pre-replay menu view: keyboard-navigable form for session selection.

No `arcade.gui` dependency: each field is a pre-allocated `arcade.Text`
object that reads its current value from a `LaunchConfig` dataclass. UP/DOWN
move focus between fields, LEFT/RIGHT mutate discrete fields (year, round,
mode, strategy toggle), typing appends to driver/team strings, ENTER
launches. The menu exists so the user picks year/round/drivers/team from
inside the window instead of remembering CLI flags.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable, Final, NamedTuple

import arcade
from src.arcade.config import (
    ACCENT,
    BG_COLOR,
    CONTENT_BG,
    DANGER,
    DRIVER_TO_TEAM_2025,
    FONT_BODY,
    FONT_TITLE,
    MENU_EMPHASIS,
    MENU_FOCUS_PAD,
    MENU_GROUP_GAP,
    MENU_GUTTER,
    MENU_HINT_BOTTOM,
    MENU_HINT_FONT,
    MENU_LABEL_FONT,
    MENU_ROW_HEIGHT,
    MENU_SCALE_MAX,
    MENU_SCALE_MIN,
    MENU_STATUS_BOTTOM,
    MENU_STATUS_FONT,
    MENU_SUBTITLE_FONT,
    MENU_SUBTITLE_TOP,
    MENU_TITLE,
    MENU_TITLE_FONT,
    MENU_TITLE_TOP,
    MENU_VALUE_FONT,
    SCREEN_HEIGHT,
    STRATEGY_REQUIRED_YEAR,
    SUCCESS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    get_gp_names,
)
from src.arcade.prepare import PrepareProgress, prepare_race

logger = logging.getLogger(__name__)


@dataclass
class LaunchConfig:
    """Validated parameters the menu hands to the race replay view."""

    year: int = 2024
    round_: int = 3
    mode_two_drivers: bool = True
    driver_main: str = "NOR"
    driver_rival: str = "LEC"
    team: str = "McLaren"
    strategy_mode: bool = False
    # Reachable only through `--viewer --no-llm` (main.py:_show_viewer_directly);
    # the in-window menu has no row for it, so an interactive launch keeps the
    # LLM path by default. #1155.
    no_llm: bool = False


@dataclass
class _FormField:
    """One menu row. Either discrete (picker) or text (editable string)."""

    key: str
    label: str
    kind: str  # "int", "round", "mode", "text", "bool"
    # Which of the three decisions this row belongs to: "race" picks the event,
    # "cars" picks who is followed, "pipeline" decides whether the agents run.
    # Rows are drawn in group order and a gap separates one group from the next.
    group: str
    get_value: Callable[[LaunchConfig], str]
    step_left: Callable[[LaunchConfig], None] | None = None
    step_right: Callable[[LaunchConfig], None] | None = None
    visible: Callable[[LaunchConfig], bool] = field(default_factory=lambda: lambda _cfg: True)
    editable: bool = False  # text fields accept on_text
    # Drawn a step larger than its siblings. Reserved for a choice that
    # changes what the replay is rather than which race it shows.
    emphasis: bool = False


def menu_content_extents(
    label_widths: list[float],
    value_widths: list[float],
    *,
    gutter: int = MENU_GUTTER,
) -> tuple[float, float]:
    """How far the form's content reaches left and right of its own axis.

    Labels are right-aligned at `axis - gutter` and values left-aligned at
    `axis + gutter`, so the two columns are as wide as the widest label and the
    widest value rather than as anything a single row measures. The band is
    sized off the whole form for that reason: fitting it to each row instead
    would move both its edges on every focus change, which reads worse than a
    band that is a little loose on the short rows.

    The two extents are not equal. At 1280 the widest label is STRATEGY, drawn a
    step larger than its siblings, and the widest value is the round's GP name,
    and those render at 107 and 120 pixels.
    """
    widest_label = max(label_widths) if label_widths else 0.0
    widest_value = max(value_widths) if value_widths else 0.0
    return gutter + widest_label, gutter + widest_value


# The keyboard contract, as pairs rather than as one string. It used to be a
# single run-on separated by spaces, so "Type to edit ENTER launch" ran together
# at the exact place a reader needs the boundary (#1101).
MENU_HINTS: Final[tuple[tuple[str, str], ...]] = (
    ("UP/DOWN", "focus"),
    ("LEFT/RIGHT", "change"),
    ("Type", "to edit"),
    ("ENTER", "launch"),
    ("ESC", "quit"),
)
MENU_HINT_SEPARATOR: Final[str] = "  ·  "


def menu_hint_line(pairs: tuple[tuple[str, str], ...] = MENU_HINTS) -> str:
    """The hint line, with a visible boundary between one binding and the next."""
    return MENU_HINT_SEPARATOR.join(f"{key} {action}" for key, action in pairs)


def round_label(year: int, round_: int) -> str:
    """`3 Melbourne`: which round of the season, and where it was run.

    The number used to be formatted `%2d`, which put a leading space on every
    single-digit round. The value column is left-aligned, so rounds 1 to 9 were
    indented one space further than the rest and the pair read as three tokens
    rather than two (#1101).
    """
    return f"{round_} {get_gp_names(year).get(round_, '?')}"


def menu_row_offsets(
    groups: Sequence[str],
    pitch: float,
    group_gap: float,
) -> tuple[float, ...]:
    """How far below the first row each row's centre sits.

    Pure, like its siblings here, and it is what makes the grouping a property
    the tests can read: a gap appears wherever consecutive rows belong to
    different groups, and nowhere else.
    """
    offsets: list[float] = []
    y = 0.0
    previous: str | None = None
    for group in groups:
        if previous is not None and group != previous:
            y += group_gap
        offsets.append(y)
        y += pitch
        previous = group
    return tuple(offsets)


class MenuBands(NamedTuple):
    """The menu's vertical anchors and its type scale, for one window height.

    Every y is where a centre-anchored line of text sits, not the edge of a
    glyph box, because that is what the draw call is given. `row_offsets` holds
    each row's distance below `form_top`, which is not a constant multiple of
    the pitch: a gap sits at every group boundary.
    """

    scale: float
    title_y: float
    subtitle_y: float
    form_top: float
    row_pitch: float
    row_offsets: tuple[float, ...]
    hint_y: float
    status_y: float

    @property
    def form_bottom(self) -> float:
        """Centre of the last row."""
        return self.form_top - (self.row_offsets[-1] if self.row_offsets else 0.0)

    @property
    def form_height(self) -> float:
        """Centre to centre of the outermost rows, plus the two half-rows."""
        return self.form_top - self.form_bottom + self.row_pitch

    def row_y(self, index: int) -> float:
        """Centre of the row drawn at `index`, groups included."""
        return self.form_top - self.row_offsets[index]


def menu_scale(window_height: int) -> float:
    """How much larger than the default the menu draws in this window.

    Clamped at both ends: below the floor the type stops being readable, which
    is the opposite of what scaling is for, and above the ceiling a 4K window
    would render the labels at 40 pt.
    """
    return max(MENU_SCALE_MIN, min(MENU_SCALE_MAX, window_height / SCREEN_HEIGHT))


def menu_bands(window_height: int, groups: Sequence[str]) -> MenuBands:
    """Place the title, the form and the hint for a window of this height.

    **Pure on purpose**, the same reason `legend_mode` is (#1096): a check on
    the drawn result needs a window and a check that needs a window does not run
    in CI.

    The three bands used to be pinned to constants, so a taller window only ever
    grew the two gaps between them. Here every distance is a multiple of
    `menu_scale`, which makes the gap-to-form ratio a property of the layout
    rather than of the window: the gap above the form was 0.46 of the form's own
    height at 720 and 1.10 at 1080, and it is 0.327 at both now (#1100).

    That constant is 0.327 rather than the 0.46 measured when #1100 landed
    because #1101 then added a gap at each group boundary, growing the form from
    280 px to 324 at scale 1. It holds over [612, 1440], which is where neither
    clamp is active; below the floor and above the ceiling the ratio moves, and
    `test_beyond_the_ceiling_the_void_returns_and_says_so` covers both ends.

    `groups` is one entry per VISIBLE row, in draw order, so the form's height
    depends on how many group boundaries the rows cross as well as on how many
    rows there are. One-driver mode hides the rival row without removing a
    boundary, which is why the count alone is not enough.

    The block sits half a row below the window's vertical centre, which is where
    it sat before either change and where it reads best against a title band
    that is taller than the hint.
    """
    scale = menu_scale(window_height)
    pitch = MENU_ROW_HEIGHT * scale
    offsets = menu_row_offsets(groups, pitch, MENU_GROUP_GAP * scale)
    span = offsets[-1] if offsets else 0.0
    return MenuBands(
        scale=scale,
        title_y=window_height - MENU_TITLE_TOP * scale,
        subtitle_y=window_height - MENU_SUBTITLE_TOP * scale,
        form_top=window_height / 2 - pitch / 2 + span / 2,
        row_pitch=pitch,
        row_offsets=offsets,
        hint_y=MENU_HINT_BOTTOM * scale,
        status_y=MENU_STATUS_BOTTOM * scale,
    )


def scale_changed(previous: float | None, current: float) -> bool:
    """Whether the menu's Text objects need their font sizes pushed again.

    `previous` is `None` until something has actually been pushed, and it is
    `None` rather than a float because 1.0 is BOTH what a freshly built view
    would carry and a scale `menu_scale` genuinely returns at SCREEN_HEIGHT. A
    view opened at the default size therefore skipped its own first push, so the
    strategy row's emphasis never reached the glyphs and appeared only after a
    resize away and back (#1109). That is the sentinel collision `CLAUDE.md`
    section 11 names: a placeholder that is also a value the code can find.

    Pure, so the collision is checkable without a window.
    """
    return previous is None or previous != current


class MenuFormGeometry(NamedTuple):
    """Where the menu form sits, in offsets from the window's centre axis.

    `axis_offset` is negative whenever the value column is wider than the label
    column, which is the normal case: it pulls the label/value boundary left so
    that the CONTENT ends up centred in the window rather than the boundary.
    """

    axis_offset: float
    band_half: float
    rule_half: float


def menu_form_geometry(
    label_widths: list[float],
    value_widths: list[float],
    *,
    gutter: int = MENU_GUTTER,
    pad: int = MENU_FOCUS_PAD,
) -> MenuFormGeometry:
    """Place the form and size the focused row's band to what is drawn in it.

    **Pure on purpose**, the same reason `legend_mode` is (#1096): this decision
    is made between GL calls in `on_draw`, so a check on the drawn result needs
    a window, and a check that needs a window does not run in CI.

    Two things come out of it. The band replaces a fixed 540 px rectangle over a
    form whose content spans 254, which is what made the accent rule read as a
    line across the panel rather than an underline of the focused field (#1099).
    The axis offset is the correction that tightening the band made visible: an
    over-wide symmetric band had been hiding that the content itself sits right
    of the window's centre.

    The fill takes the padding and the rule does not, so the fill reads as a
    band around the text and the rule as an underline of it.
    """
    extent_left, extent_right = menu_content_extents(label_widths, value_widths, gutter=gutter)
    rule_half = (extent_left + extent_right) / 2
    return MenuFormGeometry(
        axis_offset=(extent_left - extent_right) / 2,
        band_half=rule_half + pad,
        rule_half=rule_half,
    )


def last_round(year: int) -> int:
    """Final round of that season, from the calendar rather than a constant.

    The stepper was clamped to 23, so 2025 could not reach round 24, Yas Island,
    which is a real published race the menu simply would not offer; and 2023,
    which ran 22, stepped into a round with no name at all (#1116).
    """
    rounds = get_gp_names(year)
    return max(rounds) if rounds else 1


def _step_year(cfg: LaunchConfig, delta: int) -> None:
    """Move a season, keeping the round inside the calendar it lands in.

    Strategy mode pins the year, because the agents are only trained and
    validated for it. Seasons run different lengths, so a round that exists in
    one may not exist in the next.
    """
    if cfg.strategy_mode:
        return
    cfg.year = max(2023, min(2025, cfg.year + delta))
    cfg.round_ = min(cfg.round_, last_round(cfg.year))


def build_menu_fields(toggle_strategy: Callable[[], None]) -> list[_FormField]:
    """The seven rows the menu draws, in draw order.

    Module level rather than a method so the table can be read without a window
    (`arcade.View.__init__` needs one), which is what lets the group assignment,
    the emphasis flag and every rendered value string be checked in CI. The
    strategy toggle is passed in because it mutates the view's own config and
    then forces the year, which is view state rather than table data.
    """
    return [
        _FormField(
            key="year",
            label="Year",
            kind="int",
            group="race",
            get_value=lambda c: str(c.year),
            step_left=lambda c: _step_year(c, -1),
            step_right=lambda c: _step_year(c, +1),
        ),
        _FormField(
            key="round",
            label="Round",
            kind="round",
            group="race",
            get_value=lambda c: round_label(c.year, c.round_),
            step_left=lambda c: setattr(c, "round_", max(1, c.round_ - 1)),
            step_right=lambda c: setattr(c, "round_", min(last_round(c.year), c.round_ + 1)),
        ),
        _FormField(
            key="mode",
            label="Mode",
            kind="mode",
            group="cars",
            get_value=lambda c: "2 DRIVERS" if c.mode_two_drivers else "1 DRIVER",
            step_left=lambda c: setattr(c, "mode_two_drivers", not c.mode_two_drivers),
            step_right=lambda c: setattr(c, "mode_two_drivers", not c.mode_two_drivers),
        ),
        _FormField(
            key="driver_main",
            label="Driver",
            kind="text",
            group="cars",
            get_value=lambda c: c.driver_main or "---",
            editable=True,
        ),
        _FormField(
            key="driver_rival",
            label="Rival",
            kind="text",
            group="cars",
            get_value=lambda c: c.driver_rival or "---",
            editable=True,
            visible=lambda c: c.mode_two_drivers,
        ),
        _FormField(
            key="team",
            label="Team",
            kind="text",
            group="cars",
            get_value=lambda c: c.team or "---",
            editable=True,
        ),
        _FormField(
            key="strategy",
            label="Strategy",
            kind="bool",
            group="pipeline",
            get_value=lambda c: "ON" if c.strategy_mode else "OFF",
            step_left=lambda c: toggle_strategy(),
            step_right=lambda c: toggle_strategy(),
            emphasis=True,
        ),
    ]


class MenuView(arcade.View):
    """Pre-replay keyboard form. On ENTER it loads the session and swaps
    to `F1ArcadeView`. Any validation error surfaces inline in DANGER red."""

    def __init__(self, window: arcade.Window) -> None:
        super().__init__(window=window)
        arcade.set_background_color(BG_COLOR)
        self._cfg = LaunchConfig()
        self._error: str = ""
        self._loading: bool = False
        # Written by the preparation worker, read by on_draw and on_update.
        # A plain lock rather than a queue: there is one producer, one consumer,
        # and the consumer only ever wants the LATEST value.
        self._prep_lock = threading.Lock()
        self._prep_progress: PrepareProgress | None = None
        self._prep_result: object | None = None
        self._prep_error: str = ""
        self._prep_thread: threading.Thread | None = None
        self._focus_idx: int = 0
        # Last scale pushed into the Text objects, or None while nothing has
        # been. See `scale_changed` for why it cannot be a float.
        self._scale: float | None = None

        self._fields: list[_FormField] = self._build_fields()

        self._title = arcade.Text(
            MENU_TITLE,
            0,
            0,
            ACCENT,
            MENU_TITLE_FONT,
            bold=True,
            font_name=FONT_TITLE,
            anchor_x="center",
            anchor_y="center",
        )
        self._subtitle = arcade.Text(
            "Race replay + multi-agent strategy",
            0,
            0,
            TEXT_TERTIARY,
            MENU_SUBTITLE_FONT,
            font_name=FONT_BODY,
            anchor_x="center",
            anchor_y="center",
        )
        self._hint = arcade.Text(
            menu_hint_line(),
            0,
            0,
            TEXT_TERTIARY,
            MENU_HINT_FONT,
            font_name=FONT_BODY,
            anchor_x="center",
            anchor_y="center",
        )
        self._error_text = arcade.Text(
            "",
            0,
            0,
            DANGER,
            MENU_STATUS_FONT,
            bold=True,
            font_name=FONT_BODY,
            anchor_x="center",
            anchor_y="center",
        )
        self._loading_text = arcade.Text(
            "",
            0,
            0,
            ACCENT,
            MENU_STATUS_FONT,
            bold=True,
            font_name=FONT_BODY,
            anchor_x="center",
            anchor_y="center",
        )
        self._label_texts = [
            arcade.Text(
                "",
                0,
                0,
                TEXT_TERTIARY,
                MENU_LABEL_FONT,
                bold=True,
                font_name=FONT_BODY,
                anchor_x="right",
                anchor_y="center",
            )
            for _ in self._fields
        ]
        self._value_texts = [
            arcade.Text(
                "",
                0,
                0,
                TEXT_PRIMARY,
                MENU_VALUE_FONT,
                bold=True,
                font_name=FONT_BODY,
                anchor_x="left",
                anchor_y="center",
            )
            for _ in self._fields
        ]

    # --- Field definitions ----------------------------------------------

    def _build_fields(self) -> list[_FormField]:
        return build_menu_fields(self._toggle_strategy)

    def _toggle_strategy(self) -> None:
        self._cfg.strategy_mode = not self._cfg.strategy_mode
        if self._cfg.strategy_mode:
            self._cfg.year = STRATEGY_REQUIRED_YEAR

    # --- Arcade hooks ---------------------------------------------------

    def on_draw(self) -> None:
        self.clear()
        w, h = self.window.width, self.window.height
        visible_rows: list[int] = [i for i, f in enumerate(self._fields) if f.visible(self._cfg)]
        bands = menu_bands(h, [self._fields[i].group for i in visible_rows])
        self._apply_scale(bands.scale)

        self._title.x = w / 2
        self._title.y = bands.title_y
        self._title.draw()
        self._subtitle.x = w / 2
        self._subtitle.y = bands.subtitle_y
        self._subtitle.draw()

        self._draw_fields(w, visible_rows, bands)

        if self._error:
            self._error_text.text = self._error
            self._error_text.x = w / 2
            self._error_text.y = bands.status_y
            self._error_text.draw()

        if self._loading:
            self._draw_progress(w, bands)

        self._hint.x = w / 2
        self._hint.y = bands.hint_y
        self._hint.draw()

    def _draw_progress(self, w: int, bands: MenuBands) -> None:
        """The preparation's current stage, and a bar for how far through it is.

        The bar counts STAGES, not bytes. Neither fetch exposes a byte callback
        (`huggingface_hub` prints its own tqdm to stderr) and the telemetry build
        reports nothing at all, so a byte-level bar would be invented. What is
        honest is which of the three steps is running and how long it has been,
        and the elapsed count is what tells a user the window is alive rather
        than hung, which is the whole reason this is on a worker (#1115).
        """
        with self._prep_lock:
            progress = self._prep_progress

        if progress is None:
            self._loading_text.text = "Preparing race..."
        else:
            self._loading_text.text = (
                f"{progress.label}   {progress.elapsed_s(time.monotonic()):.0f}s"
            )
        self._loading_text.x = w / 2
        self._loading_text.y = bands.status_y
        self._loading_text.draw()

        if progress is None:
            return
        # Sized to the form so the bar reads as part of the same panel.
        half = (self._label_texts[0].content_width + 220) * bands.scale
        bar_y = bands.status_y - MENU_ROW_HEIGHT * bands.scale * 0.55
        height = max(3.0, 4.0 * bands.scale)
        arcade.draw_rect_filled(arcade.XYWH(w / 2, bar_y, half * 2, height), (*CONTENT_BG, 220))
        done = progress.done_fraction
        arcade.draw_rect_filled(
            arcade.XYWH(
                w / 2 - half + half * done,
                bar_y,
                half * 2 * done,
                height,
            ),
            ACCENT,
        )

    def _apply_scale(self, scale: float) -> None:
        """Resize every string to the window, once per change rather than per frame.

        Assigning `font_size` re-lays the glyph run out, and there are sixteen
        Text objects here, so the guard is what keeps a resize from costing that
        on every one of sixty frames a second.
        """
        if not scale_changed(self._scale, scale):
            return
        self._scale = scale
        self._title.font_size = MENU_TITLE_FONT * scale
        self._subtitle.font_size = MENU_SUBTITLE_FONT * scale
        self._hint.font_size = MENU_HINT_FONT * scale
        self._error_text.font_size = MENU_STATUS_FONT * scale
        self._loading_text.font_size = MENU_STATUS_FONT * scale
        for f, label, value in zip(self._fields, self._label_texts, self._value_texts, strict=True):
            step = MENU_EMPHASIS if f.emphasis else 1.0
            label.font_size = MENU_LABEL_FONT * scale * step
            value.font_size = MENU_VALUE_FONT * scale * step

    def _set_row_texts(self, visible_rows: list[int]) -> None:
        """Push every visible row's current strings into its two Text objects.

        Split from the draw loop because the focus band has to know how wide the
        widest label and the widest value are BEFORE the first row is painted,
        and a Text reports its rendered width only once it holds the string.
        """
        for field_idx in visible_rows:
            f = self._fields[field_idx]
            self._label_texts[field_idx].text = f.label.upper()
            self._value_texts[field_idx].text = f.get_value(self._cfg)

    def _draw_fields(self, w: int, visible_rows: list[int], bands: MenuBands) -> None:
        cx = w // 2
        gutter = round(MENU_GUTTER * bands.scale)

        self._set_row_texts(visible_rows)
        geometry = menu_form_geometry(
            [self._label_texts[i].content_width for i in visible_rows],
            [self._value_texts[i].content_width for i in visible_rows],
            gutter=gutter,
            pad=round(MENU_FOCUS_PAD * bands.scale),
        )
        # The form's own axis, left of the window's whenever the value column is
        # the wider of the two, so the CONTENT is what ends up centred.
        axis = cx + geometry.axis_offset

        for draw_idx, field_idx in enumerate(visible_rows):
            f = self._fields[field_idx]
            row_y = bands.row_y(draw_idx)
            focused = field_idx == self._focus_idx

            if focused:
                row_height = bands.row_pitch * (MENU_EMPHASIS if f.emphasis else 1.0) - 4
                arcade.draw_rect_filled(
                    arcade.XYWH(cx, row_y, geometry.band_half * 2, row_height),
                    (*CONTENT_BG, 220),
                )
                arcade.draw_line(
                    cx - geometry.rule_half,
                    row_y - row_height * 0.44,
                    cx + geometry.rule_half,
                    row_y - row_height * 0.44,
                    ACCENT,
                    2,
                )

            label = self._label_texts[field_idx]
            label.color = ACCENT if focused else TEXT_TERTIARY
            label.x = axis - gutter
            label.y = row_y
            label.draw()

            val = self._value_texts[field_idx]
            val.color = TEXT_PRIMARY if focused else TEXT_SECONDARY
            if f.key == "strategy":
                val.color = SUCCESS if self._cfg.strategy_mode else TEXT_SECONDARY
            val.x = axis + gutter
            val.y = row_y
            val.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if self._loading:
            return
        if symbol == arcade.key.ESCAPE:
            self.window.close()
            return
        if symbol == arcade.key.ENTER:
            self._try_launch()
            return
        if symbol == arcade.key.UP:
            self._move_focus(-1)
        elif symbol == arcade.key.DOWN:
            self._move_focus(1)
        elif symbol == arcade.key.LEFT:
            self._step(-1)
        elif symbol == arcade.key.RIGHT:
            self._step(1)
        elif symbol == arcade.key.BACKSPACE:
            self._backspace()

    def on_text(self, text: str) -> None:
        """Forwards typed characters to the focused text field."""
        if self._loading:
            return
        f = self._fields[self._focus_idx]
        if not f.editable:
            return
        clean = "".join(c for c in text if c.isalnum() or c in " -")
        if not clean:
            return
        current = getattr(self._cfg, f.key)
        if f.key in ("driver_main", "driver_rival"):
            # 3-letter codes: uppercase, replace rather than append past 3
            new = (current + clean).upper()
            new = new[-3:] if len(new) > 3 else new
            setattr(self._cfg, f.key, new)
            if f.key == "driver_main":
                self._autofill_team(new)
        else:
            setattr(self._cfg, f.key, current + clean)

    def _autofill_team(self, driver_code: str) -> None:
        """Copy the driver's team into the team field once the code is 3 chars.

        Same UX the user expects from the CLI: pick the driver, the team
        resolves automatically from the 2025 grid. Users can still tab to
        the team field and override for other seasons or multi-team cases."""
        if len(driver_code) != 3:
            return
        team = DRIVER_TO_TEAM_2025.get(driver_code.upper())
        if team:
            self._cfg.team = team

    # --- Focus + step ---------------------------------------------------

    def _visible_field_indexes(self) -> list[int]:
        return [i for i, f in enumerate(self._fields) if f.visible(self._cfg)]

    def _move_focus(self, delta: int) -> None:
        visible = self._visible_field_indexes()
        if self._focus_idx not in visible:
            self._focus_idx = visible[0]
            return
        pos = visible.index(self._focus_idx)
        self._focus_idx = visible[(pos + delta) % len(visible)]

    def _step(self, delta: int) -> None:
        f = self._fields[self._focus_idx]
        step = f.step_right if delta > 0 else f.step_left
        if step is not None:
            step(self._cfg)

    def _backspace(self) -> None:
        f = self._fields[self._focus_idx]
        if not f.editable:
            return
        current = getattr(self._cfg, f.key)
        new = current[:-1]
        setattr(self._cfg, f.key, new)
        if f.key == "driver_main":
            self._autofill_team(new)

    # --- Launch ---------------------------------------------------------

    def _try_launch(self) -> None:
        """Validate, then hand the preparation to a worker and keep drawing.

        It used to force one frame of "Loading session..." and then block the
        pyglet thread for as long as the load took. That is measured at 349 s
        for a race whose telemetry is not cached yet, plus the downloads, and a
        window that does not pump events for six minutes is one the OS paints as
        dead (#1115).
        """
        err = self._validate(self._cfg)
        if err:
            self._error = err
            return
        if self._prep_thread is not None and self._prep_thread.is_alive():
            return
        self._error = ""
        self._loading = True
        with self._prep_lock:
            self._prep_progress = None
            self._prep_result = None
            self._prep_error = ""

        gp = get_gp_names(self._cfg.year).get(self._cfg.round_, f"Round{self._cfg.round_}")
        logger.info("Menu: preparing %d round %d (%s)", self._cfg.year, self._cfg.round_, gp)
        self._prep_thread = threading.Thread(
            target=self._prepare_worker,
            args=(self._cfg.year, self._cfg.round_, gp, self._cfg.strategy_mode),
            name="arcade-prepare",
            daemon=True,
        )
        self._prep_thread.start()

    def _prepare_worker(self, year: int, round_: int, gp: str, strategy_enabled: bool) -> None:
        """Fetch and load off the draw thread. Touches no GL object.

        Everything it produces is plain data; `on_update` builds the view.
        """

        def publish(progress: PrepareProgress) -> None:
            with self._prep_lock:
                self._prep_progress = progress

        try:
            session_data = prepare_race(
                year,
                round_,
                gp,
                strategy_enabled=strategy_enabled,
                on_progress=publish,
            )
        except Exception as exc:  # noqa: BLE001 - reported to the user verbatim
            logger.exception("Race preparation failed")
            with self._prep_lock:
                self._prep_error = f"{type(exc).__name__}: {exc}"
            return
        with self._prep_lock:
            self._prep_result = session_data

    def on_update(self, delta_time: float) -> None:
        """Pick up whatever the worker finished, on the thread that owns the GL.

        `Track` and `F1ArcadeView` allocate `arcade.Text`, which needs the
        context, so the worker hands back a `SessionData` and the swap happens
        here.
        """
        del delta_time
        if not self._loading:
            return
        with self._prep_lock:
            result = self._prep_result
            error = self._prep_error
        if error:
            self._loading = False
            self._error = error
            self._prep_thread = None
            return
        if result is not None:
            self._prep_result = None
            self._prep_thread = None
            self._show_replay(result)

    @staticmethod
    def _validate(cfg: LaunchConfig) -> str:
        if len(cfg.driver_main) != 3:
            return "driver must be 3 letters"
        if cfg.mode_two_drivers and len(cfg.driver_rival) != 3:
            return "rival must be 3 letters"
        if cfg.strategy_mode:
            if cfg.year != STRATEGY_REQUIRED_YEAR:
                return f"strategy requires year {STRATEGY_REQUIRED_YEAR}"
            if not cfg.team.strip():
                return "team required for strategy mode"
        return ""

    def launch_with(self, cfg: LaunchConfig) -> None:
        """Skip the form and prepare this configuration straight away.

        The `--viewer` flag used to build the whole thing itself in
        `main.py:_show_viewer_directly`: its own session load, its own driver
        fallback, its own view construction. That second copy is why the flag
        never gained the lazy fetch or the worker thread when the menu did, and
        duplicated launch paths drifting apart is a scar this repo already
        carries. One path now (#1115).
        """
        self._cfg = cfg
        self._try_launch()

    def _show_replay(self, session_data) -> None:
        """Build the replay view from a prepared session and hand it the window.

        Main thread only: everything below allocates GL resources.
        """
        from src.arcade.app import F1ArcadeView
        from src.arcade.track import Track

        ref_x, ref_y = session_data.ref_lap_xy
        track = Track(
            ref_x=ref_x,
            ref_y=ref_y,
            drs_flags=session_data.ref_lap_drs,
            rotation_deg=session_data.circuit_rotation_deg,
        )

        driver_main = self._cfg.driver_main
        driver_rival = self._cfg.driver_rival if self._cfg.mode_two_drivers else None
        if driver_main not in session_data.frames_by_driver:
            logger.warning("Driver %s not in session", driver_main)
            available = list(session_data.frames_by_driver.keys())
            driver_main = available[0] if available else driver_main
        if driver_rival and driver_rival not in session_data.frames_by_driver:
            logger.warning("Rival %s not in session, ignoring", driver_rival)
            driver_rival = None

        view = F1ArcadeView(
            window=self.window,
            session_data=session_data,
            track=track,
            driver_main=driver_main,
            driver_rival=driver_rival,
            year=self._cfg.year,
            strategy_enabled=self._cfg.strategy_mode,
            team=self._cfg.team,
            no_llm=self._cfg.no_llm,
        )
        self.window.show_view(view)
