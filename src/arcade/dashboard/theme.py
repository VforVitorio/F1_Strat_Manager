"""The Qt half of the dashboard theme: the palette applied to a QApplication.

**The colours themselves are no longer here.** They live in
`src/arcade/palette.py`, which imports nothing but `html` and `typing`,
and are re-exported below so every widget in this package keeps importing
them from where it always did.

The split exists because PITWALL renders its AGENTS window from the same
`agent_formatters` that paint the Qt one — that is what makes the port
1:1 by construction rather than by inspection — and importing this module
used to drag in PySide6 (a display stack) and, through `classify_action`,
pandas. Measured: `src.arcade.strategy` alone is 0.410 s. It also made
`test_the_two_python_palettes_still_mirror_each_other` unrunnable on a
headless runner, so the one guard against the two Python palettes drifting
was skipped in CI.

`classify_action` is still imported rather than copied: a hand-written
twin lived here until 2026-08-01 and drifted, which is the same mechanism
#620 fixed once for `_ALERT_SEVERITY`.

This whole package is deleted when the Qt windows are retired; `palette.py`
is not.
"""

from __future__ import annotations

import os
from typing import Final

from PySide6.QtGui import QColor, QPalette

from src.arcade.palette import (
    ACCENT,
    BG_COLOR,
    BORDER_COLOR,
    COMPOUND_COLORS,
    COMPOUND_NAMES,
    CONTENT_BG,
    DANGER,
    INFO,
    MONO_FONT_STACK,
    SECONDARY_BG,
    SUCCESS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    WARNING,
    compound_color,
    compound_pill_html,
    flag_chip_html,
    hex_str,
)
from src.arcade.strategy import (
    classify_action,  # noqa: F401 -- re-exported for orchestrator_card.py
)

__all__ = [
    "ACCENT",
    "BG_COLOR",
    "BORDER_COLOR",
    "COMPOUND_COLORS",
    "COMPOUND_NAMES",
    "CONTENT_BG",
    "DANGER",
    "INFO",
    "MONO_FONT_STACK",
    "SECONDARY_BG",
    "STREAM_HOST",
    "STREAM_PORT",
    "SUCCESS",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "TEXT_TERTIARY",
    "WARNING",
    "apply_dark_palette",
    "classify_action",
    "compound_color",
    "compound_pill_html",
    "flag_chip_html",
    "hex_str",
    "qcolor",
]

# --- Stream config (must match src/arcade/config.py) ---------------------
STREAM_HOST: Final[str] = os.environ.get("F1_STREAM_HOST", "127.0.0.1")
STREAM_PORT: Final[int] = int(os.environ.get("F1_STREAM_PORT", "9998"))


def qcolor(rgb: tuple[int, int, int]) -> QColor:
    """Small helper so widgets can do ``self.setPalette(qcolor(ACCENT))``."""
    return QColor(rgb[0], rgb[1], rgb[2])


def apply_dark_palette(app) -> None:
    """Apply the dashboard dark palette to ``QApplication`` and install a
    global stylesheet that widgets inherit without having to set colours
    one by one. Keep the widget tree declarative: child widgets only
    override specific roles (action badges, cliff lines, etc.).
    """
    palette = QPalette()
    palette.setColor(QPalette.Window, qcolor(BG_COLOR))
    palette.setColor(QPalette.WindowText, qcolor(TEXT_PRIMARY))
    palette.setColor(QPalette.Base, qcolor(CONTENT_BG))
    palette.setColor(QPalette.AlternateBase, qcolor(SECONDARY_BG))
    palette.setColor(QPalette.Text, qcolor(TEXT_PRIMARY))
    palette.setColor(QPalette.Button, qcolor(SECONDARY_BG))
    palette.setColor(QPalette.ButtonText, qcolor(TEXT_PRIMARY))
    palette.setColor(QPalette.ToolTipBase, qcolor(CONTENT_BG))
    palette.setColor(QPalette.ToolTipText, qcolor(TEXT_PRIMARY))
    palette.setColor(QPalette.Highlight, qcolor(ACCENT))
    palette.setColor(QPalette.HighlightedText, qcolor(BG_COLOR))
    app.setPalette(palette)
    app.setStyleSheet(
        f"QMainWindow, QWidget {{ background-color: {hex_str(BG_COLOR)}; "
        f"color: {hex_str(TEXT_PRIMARY)}; }} "
        f'QFrame[card="true"] {{ background-color: {hex_str(CONTENT_BG)}; '
        f"border: 1px solid {hex_str(BORDER_COLOR)}; border-radius: 6px; }} "
        f"QLabel#chip {{ color: {hex_str(TEXT_SECONDARY)}; padding: 2px 8px; "
        f"background-color: {hex_str(SECONDARY_BG)}; border-radius: 10px; "
        f"font-size: 11px; }} "
        f"QStatusBar {{ background-color: {hex_str(CONTENT_BG)}; "
        f"color: {hex_str(TEXT_TERTIARY)}; }}"
    )
