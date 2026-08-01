"""Dashboard theme: palette, compound colours, action classification.

Constants are duplicated from ``src/arcade/config.py`` on purpose: the
dashboard runs in its own process and importing the arcade config would
pull pyglet / arcade / fastf1 into the Qt process (cold start ~2 s and
no way to ever call the API without paying the price). Keep both files
in sync when the palette changes upstream: they are the two sources
of truth for the TFG's visual identity.

Values here mirror ``src/telemetry/frontend/app/styles.py`` (the
Streamlit app's palette) so the dashboard reads as the same product as
the Streamlit UI and the arcade replay.
"""

from __future__ import annotations

import os
from typing import Final

from PySide6.QtGui import QColor, QPalette

from src.arcade.strategy import (
    classify_action,  # noqa: F401 -- re-exported for orchestrator_card.py
)

# --- Palette (RGB tuples) ------------------------------------------------
BG_COLOR: Final[tuple[int, int, int]] = (18, 17, 39)  # #121127 PRIMARY_BG
CONTENT_BG: Final[tuple[int, int, int]] = (24, 22, 51)  # #181633 panel bg
SECONDARY_BG: Final[tuple[int, int, int]] = (30, 27, 75)  # #1e1b4b elevated
BORDER_COLOR: Final[tuple[int, int, int]] = (45, 45, 58)  # #2d2d3a
TEXT_PRIMARY: Final[tuple[int, int, int]] = (255, 255, 255)
TEXT_SECONDARY: Final[tuple[int, int, int]] = (209, 213, 219)  # #d1d5db
TEXT_TERTIARY: Final[tuple[int, int, int]] = (156, 163, 175)  # #9ca3af
ACCENT: Final[tuple[int, int, int]] = (167, 139, 250)  # #a78bfa purple
SUCCESS: Final[tuple[int, int, int]] = (16, 185, 129)  # #10b981 emerald
WARNING: Final[tuple[int, int, int]] = (245, 158, 11)  # #f59e0b amber
DANGER: Final[tuple[int, int, int]] = (239, 68, 68)  # #ef4444 red
INFO: Final[tuple[int, int, int]] = (59, 130, 246)  # #3b82f6 blue

# --- Compound colours (Pirelli IDs 0-4) ----------------------------------
COMPOUND_COLORS: Final[dict[int, tuple[int, int, int]]] = {
    0: (230, 50, 50),  # SOFT
    1: (230, 200, 50),  # MEDIUM
    2: (230, 230, 230),  # HARD
    3: (60, 200, 60),  # INTERMEDIATE
    4: (60, 130, 230),  # WET
}
COMPOUND_NAMES: Final[dict[str, tuple[int, int, int]]] = {
    "SOFT": COMPOUND_COLORS[0],
    "MEDIUM": COMPOUND_COLORS[1],
    "HARD": COMPOUND_COLORS[2],
    "INTER": COMPOUND_COLORS[3],
    "INTERMEDIATE": COMPOUND_COLORS[3],
    "WET": COMPOUND_COLORS[4],
}

# --- Stream config (must match src/arcade/config.py) ---------------------
STREAM_HOST: Final[str] = os.environ.get("F1_STREAM_HOST", "127.0.0.1")
STREAM_PORT: Final[int] = int(os.environ.get("F1_STREAM_PORT", "9998"))

# --- Action classification ---------------------------------------------------
# `classify_action` is imported from src.arcade.strategy (not duplicated): a
# hand-copied version lived here until this cleanup (2026-08-01), mirrored
# under a "mirrors X" comment -- the same drift mechanism #620 already fixed
# once for `_ALERT_SEVERITY` (imported above, not defined here either), just
# not yet triggered for this one. The extra pandas import this pulls in from
# strategy.py is a small addition next to the PySide6 + pyqtgraph cost this
# process already pays on cold start.


def qcolor(rgb: tuple[int, int, int]) -> QColor:
    """Small helper so widgets can do ``self.setPalette(qcolor(ACCENT))``."""
    return QColor(rgb[0], rgb[1], rgb[2])


def hex_str(rgb: tuple[int, int, int]) -> str:
    """Return ``#rrggbb`` for Qt stylesheet strings."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


# --- Monospace font chain ------------------------------------------------
# Fira Code (+ its Nerd Font variant) ships with programming ligatures and
# a lot of monospace glyphs that align neatly for metric tables. Users
# who have it installed get the richer look; the Consolas / Courier New
# fallbacks keep the rendering legible when not.
MONO_FONT_STACK: Final[str] = (
    "'FiraCode Nerd Font Mono', 'Fira Code', 'JetBrains Mono', 'Consolas', 'Courier New', monospace"
)


# --- Compound pill HTML (Pirelli-style badge) ---------------------------
# Compound labels come through the pipeline in several shapes: the
# friendly agent form ("SOFT", "MEDIUM", "HARD", "INTER", "WET") and the
# raw Pirelli id ("C1"…"C6"). Both should paint the same pill colour —
# red for soft, yellow for medium, white for hard, green for inter,
# blue for wet. The function returns an HTML snippet that QLabel can
# render in rich-text mode next to plain text.

_COMPOUND_COLOUR_BY_LABEL: Final[dict[str, tuple[int, int, int]]] = {
    "SOFT": (230, 50, 50),
    "MEDIUM": (230, 200, 50),
    "HARD": (230, 230, 230),
    "INTER": (60, 200, 60),
    "INTERMEDIATE": (60, 200, 60),
    "WET": (60, 130, 230),
    "S": (230, 50, 50),
    "M": (230, 200, 50),
    "H": (230, 230, 230),
    "I": (60, 200, 60),
    "W": (60, 130, 230),
    # Pirelli Cx mapping per the dry-race convention — hardest compounds
    # white, medium yellow, softest red.
    "C1": (230, 230, 230),
    "C2": (230, 230, 230),
    "C3": (230, 200, 50),
    "C4": (230, 50, 50),
    "C5": (230, 50, 50),
    "C6": (230, 50, 50),
}


def compound_color(compound: str) -> tuple[int, int, int]:
    """Map any compound label to a Pirelli-style colour tuple."""
    key = (compound or "").upper().strip()
    return _COMPOUND_COLOUR_BY_LABEL.get(key, TEXT_SECONDARY)


def compound_pill_html(compound: str | None) -> str:
    """Return a colored rounded pill as a Qt rich-text span.

    Used inline in ``QLabel.setText`` so the compound always reads as a
    Pirelli-style badge without having to embed a child widget.
    Unknown labels collapse to a neutral dash pill to keep the layout
    aligned."""
    label = (compound or "—").strip() or "—"
    colour = compound_color(label)
    # Dark text on light/saturated backgrounds, white on dim ones.
    lum = 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]
    fg = BG_COLOR if lum > 180 else TEXT_PRIMARY
    # Qt rich-text: single quotes in the font stack are fine since we use
    # double quotes for the style attribute.
    font_stack = MONO_FONT_STACK
    return (
        '<span style="'
        f"background-color: {hex_str(colour)}; "
        f"color: {hex_str(fg)}; "
        "padding: 1px 7px; border-radius: 7px; "
        "font-weight: 800; font-size: 10px; "
        f"font-family: {font_stack};"
        f'">{label}</span>'
    )


# --- Alert flag chips ---------------------------------------------------
# Radio / RCM intents collapse to a colored chip matching the broadcast
# flag semantics — red for red-flag / safety-car, amber for VSC / yellow,
# blue for ops "PROBLEM" / "WARNING" radios. Anything unknown stays
# neutral grey so the reader is never misled by an unstyled label.

_FLAG_BG_BY_INTENT: Final[dict[str, tuple[int, int, int]]] = {
    "SAFETY_CAR": DANGER,
    "RED_FLAG": DANGER,
    "VSC": WARNING,
    "VIRTUAL_SAFETY_CAR": WARNING,
    "YELLOW_FLAG": WARNING,
    # Same tier as YELLOW_FLAG (matches _ALERT_SEVERITY's own severity-2 grouping
    # in strategy.py). A sector-scoped double yellow (rcm_events.py's
    # classify_rcm_event returns this exact event type) was missing this key and
    # fell through to the neutral-grey default -- the same bug shape #398 fixed
    # in _ALERT_SEVERITY, alive here independently because this is a third,
    # separately maintained severity mapping (found by the 2026-08-01 cleanup's
    # adversarial gate).
    "YELLOW_FLAG_SECTOR": WARNING,
    "PROBLEM": INFO,
    "WARNING": INFO,
    "PENALTY": DANGER,
}


def flag_chip_html(intent: str | None) -> str:
    """Coloured pill for a single alert intent or RCM event type."""
    key = (intent or "—").upper().strip() or "—"
    bg = _FLAG_BG_BY_INTENT.get(key, TEXT_TERTIARY)
    label = key.replace("_", " ")
    return (
        '<span style="'
        f"background-color: {hex_str(bg)}; "
        f"color: {hex_str(TEXT_PRIMARY)}; "
        "padding: 1px 6px; border-radius: 6px; "
        "font-weight: 700; font-size: 10px; letter-spacing: 0.3px;"
        f'">{label}</span>'
    )


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
