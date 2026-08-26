"""The product's visual vocabulary, with no toolkit attached.

Colours, the compound and flag maps, and the two HTML badge builders that
turn a compound or an alert intent into a coloured pill. Nothing here
imports Qt, pyglet or pandas, which is the whole point: **PITWALL renders
the AGENTS window from the same formatters that paint the Qt one**, and
those formatters need six colour tuples and two badge builders, not a
widget toolkit and a dataframe library.

Measured before the split: importing `src.arcade.strategy` (which
`dashboard/theme.py` does, for `classify_action`) costs 0.410 s and pulls
pandas, against 0.025 s for `src.arcade.config`. A module that answers
"what colour is a MEDIUM tyre" should cost neither that nor a display
stack.

It lived outside `src/arcade/dashboard/` deliberately, so that retiring
the Qt windows would not take it with them, and that is exactly what
happened: the package is gone and this is still here, read by the pyglet
HUD and by PITWALL's own formatters.

**These values are still a deliberate copy of `src/arcade/config.py`'s**,
kept separate because the two run in different processes and importing
the arcade config from a UI process would pull pyglet and fastf1 in with
it. `tests/surfaces/test_pitwall_tokens.py` is what stops the copies
drifting, and it can finally run in CI now that reading them needs no
libEGL.
"""

from __future__ import annotations

import html
from typing import Final

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


def hex_str(rgb: tuple[int, int, int]) -> str:
    """Return ``#rrggbb``, for a Qt stylesheet or a CSS declaration alike."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    """WCAG 2.x relative luminance, which is not the same as perceived brightness."""
    channels = []
    for raw in rgb:
        c = raw / 255
        channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(first: tuple[int, int, int], second: tuple[int, int, int]) -> float:
    """The WCAG contrast ratio between two colours, from 1.0 to 21.0."""
    high, low = sorted((_relative_luminance(first), _relative_luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


# WCAG AA for body-sized text. The pills render at 10 px bold, which is
# nowhere near the 18.66 px bold that would let 3.0 count.
_AA_SMALL_TEXT: Final[float] = 4.5


def readable_on(background: tuple[int, int, int]) -> tuple[int, int, int]:
    """Whichever of the dark ground or white reads better ON ``background``.

    **Chosen by the actual criterion, not by a stand-in for it.** The pill
    builders used to pick white below a hand-set brightness threshold of
    180, and the threshold does not answer the question contrast asks: at
    the alert chip's own grey (#9ca3af) it scores 162 and takes white, and
    white on that grey measures **2.54:1** against the 4.5 the 10 px label
    needs. The sprint-8 gate found the same 2.54 on the STAY OUT badge -
    the two least legible things on the screen were the alarm and the
    decision.

    Comparing the two candidates directly cannot be wrong by a threshold,
    and it lands on dark text for every saturated fill in this palette:
    SUCCESS 7.29, ACCENT 6.80, WARNING 8.61, DANGER 4.92, all passing.
    """
    dark = contrast_ratio(BG_COLOR, background)
    light = contrast_ratio(TEXT_PRIMARY, background)
    return BG_COLOR if dark >= light else TEXT_PRIMARY


def _blend(
    colour: tuple[int, int, int], towards: tuple[int, int, int], amount: float
) -> tuple[int, int, int]:
    """`colour` moved `amount` of the way to `towards`, channel by channel."""
    red, green, blue = (round(c + (t - c) * amount) for c, t in zip(colour, towards))
    return (red, green, blue)


def legible_fill(
    background: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """A `(fill, text)` pair for a small-text pill that reaches AA.

    Some brand colours sit in the middle of the range where NEITHER
    ground is legible: the SOFT compound's #e63232 gives 4.31:1 against
    white and less against the dark ground, so picking the better of the
    two still lands under the 4.5 a 10 px label needs. The fill is then
    deepened away from its own text - darkened under white, lightened
    under dark - by the smallest step that clears AA, which keeps the hue
    a strategist recognises while making the label readable.

    Only the PILL is adjusted. `compound_color` itself is untouched,
    because the tyre chart's stint bands are painted from it and their
    contrast question is a different one (a band carries no text).
    """
    fill = background
    text = readable_on(fill)
    away = TEXT_PRIMARY if text == BG_COLOR else BG_COLOR
    # Twelve 8% steps reach either ground; the loop exits on the first
    # that clears, so a colour already passing is returned unchanged.
    for _ in range(12):
        if contrast_ratio(text, fill) >= _AA_SMALL_TEXT:
            break
        fill = _blend(fill, away, 0.08)
        text = readable_on(fill)
    return fill, text


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
# raw Pirelli id ("C1"…"C6"). Both should paint the same pill colour:
# red for soft, yellow for medium, white for hard, green for inter,
# blue for wet. The function returns an HTML snippet that a QLabel can
# render in rich-text mode, and that a browser renders identically.

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
    # Pirelli Cx mapping per the dry-race convention: hardest compounds
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
    """Return a coloured rounded pill as a rich-text span.

    Used inline in a label so the compound always reads as a
    Pirelli-style badge without having to embed a child widget.
    Unknown labels collapse to a neutral dash pill to keep the layout
    aligned.

    The label is HTML-escaped. It comes off the wire, and an unescaped
    ``<`` broke Qt's rich-text parser long before PITWALL made these
    strings reach a webview.
    """
    label = (compound or "—").strip() or "—"
    colour, fg = legible_fill(compound_color(label))
    # Single quotes in the font stack are fine since the style attribute
    # uses double quotes.
    font_stack = MONO_FONT_STACK
    return (
        '<span style="'
        f"background-color: {hex_str(colour)}; "
        f"color: {hex_str(fg)}; "
        "padding: 1px 7px; border-radius: 7px; "
        "font-weight: 800; font-size: 10px; "
        f"font-family: {font_stack};"
        f'">{html.escape(label)}</span>'
    )


# --- Alert flag chips ---------------------------------------------------
# Radio / RCM intents collapse to a coloured chip matching the broadcast
# flag semantics: red for red-flag / safety-car, amber for VSC / yellow,
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
    """Coloured pill for a single alert intent or RCM event type.

    The label is HTML-escaped for the same reason as the compound pill:
    it originates in agent output, not in this file.

    The foreground comes from `readable_on` rather than being fixed to
    white, which is what made this chip the least legible text in the
    AGENTS window at 2.54:1 - an alarm nobody can read is not an alarm.
    """
    key = (intent or "—").upper().strip() or "—"
    bg, fg = legible_fill(_FLAG_BG_BY_INTENT.get(key, TEXT_TERTIARY))
    label = key.replace("_", " ")
    return (
        '<span style="'
        f"background-color: {hex_str(bg)}; "
        f"color: {hex_str(fg)}; "
        "padding: 1px 6px; border-radius: 6px; "
        "font-weight: 700; font-size: 10px; letter-spacing: 0.3px;"
        f'">{html.escape(label)}</span>'
    )
