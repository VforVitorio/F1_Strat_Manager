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
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import arcade
from src.arcade.config import (
    ACCENT,
    BG_COLOR,
    CONTENT_BG,
    DANGER,
    DRIVER_TO_TEAM_2025,
    FONT_BODY,
    FONT_TITLE,
    MENU_FOCUS_PAD,
    MENU_GUTTER,
    MENU_HINT_FONT,
    MENU_LABEL_FONT,
    MENU_ROW_HEIGHT,
    MENU_TITLE,
    MENU_VALUE_FONT,
    STRATEGY_REQUIRED_YEAR,
    SUCCESS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    get_gp_names,
)

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


@dataclass
class _FormField:
    """One menu row. Either discrete (picker) or text (editable string)."""

    key: str
    label: str
    kind: str  # "int", "round", "mode", "text", "bool"
    get_value: Callable[[LaunchConfig], str]
    step_left: Callable[[LaunchConfig], None] | None = None
    step_right: Callable[[LaunchConfig], None] | None = None
    visible: Callable[[LaunchConfig], bool] = field(default_factory=lambda: lambda _cfg: True)
    editable: bool = False  # text fields accept on_text


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

    The two extents are not equal. At 1280 the widest label is STRATEGY and the
    widest value is the round's GP name, and those render at 83 and 131 pixels.
    """
    widest_label = max(label_widths) if label_widths else 0.0
    widest_value = max(value_widths) if value_widths else 0.0
    return gutter + widest_label, gutter + widest_value


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


class MenuView(arcade.View):
    """Pre-replay keyboard form. On ENTER it loads the session and swaps
    to `F1ArcadeView`. Any validation error surfaces inline in DANGER red."""

    def __init__(self, window: arcade.Window) -> None:
        super().__init__(window=window)
        arcade.set_background_color(BG_COLOR)
        self._cfg = LaunchConfig()
        self._error: str = ""
        self._loading: bool = False
        self._focus_idx: int = 0

        self._fields: list[_FormField] = self._build_fields()

        self._title = arcade.Text(
            MENU_TITLE,
            0,
            0,
            ACCENT,
            32,
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
            13,
            font_name=FONT_BODY,
            anchor_x="center",
            anchor_y="center",
        )
        self._hint = arcade.Text(
            "UP/DOWN focus   LEFT/RIGHT change   Type to edit   ENTER launch   ESC quit",
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
            12,
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
            14,
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
        return [
            _FormField(
                key="year",
                label="Year",
                kind="int",
                get_value=lambda c: str(c.year),
                step_left=lambda c: (
                    setattr(c, "year", max(2023, c.year - 1)) if not c.strategy_mode else None
                ),
                step_right=lambda c: (
                    setattr(c, "year", min(2025, c.year + 1)) if not c.strategy_mode else None
                ),
            ),
            _FormField(
                key="round",
                label="Round",
                kind="round",
                get_value=lambda c: f"{c.round_:2d}  {get_gp_names(c.year).get(c.round_, '?')}",
                step_left=lambda c: setattr(c, "round_", max(1, c.round_ - 1)),
                step_right=lambda c: setattr(c, "round_", min(23, c.round_ + 1)),
            ),
            _FormField(
                key="mode",
                label="Mode",
                kind="mode",
                get_value=lambda c: "2 DRIVERS" if c.mode_two_drivers else "1 DRIVER",
                step_left=lambda c: setattr(c, "mode_two_drivers", not c.mode_two_drivers),
                step_right=lambda c: setattr(c, "mode_two_drivers", not c.mode_two_drivers),
            ),
            _FormField(
                key="driver_main",
                label="Driver",
                kind="text",
                get_value=lambda c: c.driver_main or "---",
                editable=True,
            ),
            _FormField(
                key="driver_rival",
                label="Rival",
                kind="text",
                get_value=lambda c: c.driver_rival or "---",
                editable=True,
                visible=lambda c: c.mode_two_drivers,
            ),
            _FormField(
                key="team",
                label="Team",
                kind="text",
                get_value=lambda c: c.team or "---",
                editable=True,
            ),
            _FormField(
                key="strategy",
                label="Strategy",
                kind="bool",
                get_value=lambda c: "ON" if c.strategy_mode else "OFF",
                step_left=lambda c: self._toggle_strategy(),
                step_right=lambda c: self._toggle_strategy(),
            ),
        ]

    def _toggle_strategy(self) -> None:
        self._cfg.strategy_mode = not self._cfg.strategy_mode
        if self._cfg.strategy_mode:
            self._cfg.year = STRATEGY_REQUIRED_YEAR

    # --- Arcade hooks ---------------------------------------------------

    def on_draw(self) -> None:
        self.clear()
        w, h = self.window.width, self.window.height
        self._title.x = w / 2
        self._title.y = h - 80
        self._title.draw()
        self._subtitle.x = w / 2
        self._subtitle.y = h - 112
        self._subtitle.draw()

        self._draw_fields(w, h)

        if self._error:
            self._error_text.text = self._error
            self._error_text.x = w / 2
            self._error_text.y = 120
            self._error_text.draw()

        if self._loading:
            self._loading_text.text = "Loading session..."
            self._loading_text.x = w / 2
            self._loading_text.y = 120
            self._loading_text.draw()

        self._hint.x = w / 2
        self._hint.y = 60
        self._hint.draw()

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

    def _draw_fields(self, w: int, h: int) -> None:
        cx = w // 2
        visible_rows: list[int] = [i for i, f in enumerate(self._fields) if f.visible(self._cfg)]
        total_h = len(visible_rows) * MENU_ROW_HEIGHT
        start_y = (h + total_h) // 2 - 40

        self._set_row_texts(visible_rows)
        geometry = menu_form_geometry(
            [self._label_texts[i].content_width for i in visible_rows],
            [self._value_texts[i].content_width for i in visible_rows],
        )
        # The form's own axis, left of the window's whenever the value column is
        # the wider of the two, so the CONTENT is what ends up centred.
        axis = cx + geometry.axis_offset

        for draw_idx, field_idx in enumerate(visible_rows):
            f = self._fields[field_idx]
            row_y = start_y - draw_idx * MENU_ROW_HEIGHT
            focused = field_idx == self._focus_idx

            if focused:
                arcade.draw_rect_filled(
                    arcade.XYWH(cx, row_y, geometry.band_half * 2, MENU_ROW_HEIGHT - 4),
                    (*CONTENT_BG, 220),
                )
                arcade.draw_line(
                    cx - geometry.rule_half,
                    row_y - 16,
                    cx + geometry.rule_half,
                    row_y - 16,
                    ACCENT,
                    2,
                )

            label = self._label_texts[field_idx]
            label.color = ACCENT if focused else TEXT_TERTIARY
            label.x = axis - MENU_GUTTER
            label.y = row_y
            label.draw()

            val = self._value_texts[field_idx]
            val.color = TEXT_PRIMARY if focused else TEXT_SECONDARY
            if f.key == "strategy":
                val.color = SUCCESS if self._cfg.strategy_mode else TEXT_TERTIARY
            val.x = axis + MENU_GUTTER
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
        err = self._validate(self._cfg)
        if err:
            self._error = err
            return
        self._error = ""
        self._loading = True
        # Force a redraw so "Loading..." shows before the blocking load
        self.on_draw()
        self.window.flip()
        self._spawn_replay()

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

    def _spawn_replay(self) -> None:
        from src.arcade.app import F1ArcadeView
        from src.arcade.data import SessionLoader
        from src.arcade.track import Track

        gp = get_gp_names(self._cfg.year).get(self._cfg.round_, f"Round{self._cfg.round_}")
        logger.info("Menu: loading %d round %d (%s)", self._cfg.year, self._cfg.round_, gp)

        try:
            session_data = SessionLoader().load(self._cfg.year, self._cfg.round_, gp)
        except Exception as exc:
            logger.exception("SessionLoader failed")
            self._error = f"session load failed: {exc}"
            self._loading = False
            return

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
        )
        self.window.show_view(view)
