"""Header, agent cards and status bar, as JSON the AGENTS window renders.

Every string and every colour here is produced by the code that paints
the Qt window: `src/arcade/dashboard/agent_formatters.py` for the six
cards, and the header/status logic transcribed from
`src/arcade/dashboard/window.py`. Nothing reformats or re-decides.

Colours leave as `#rrggbb` because that is what both a Qt stylesheet and
a CSS declaration take, and because the AGENTS window renders in the QT
palette rather than in `tokens.css`'s semantics: this is a 1:1 port, and
the two palettes deliberately differ (`test_pitwall_tokens.py`). Choosing
the web palette here would BE the redesign sprint 8 owns.
"""

from __future__ import annotations

from typing import Any

from src.arcade.dashboard.agent_formatters import (
    format_pace,
    format_pit,
    format_radio,
    format_rag,
    format_situation,
    format_tire,
    radio_tooltip_html,
    rag_tooltip_html,
)
from src.arcade.palette import (
    DANGER,
    SUCCESS,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    WARNING,
    hex_str,
)

# window.py's HeaderBar.set_connection, with its three states and their
# colours. "Connecting..." is what the socket thread reports while it
# retries, which is most of the time before an arcade exists.
CONNECTION_COLOURS: dict[str, str] = {
    "Connected": hex_str(SUCCESS),
    "Connecting...": hex_str(WARNING),
    "Disconnected": hex_str(DANGER),
}

# `agent_card.py::_GLYPH_FOR`, which cannot be imported because it lives in
# a QFrame subclass. It is repeated rather than reimplemented, and
# `test_the_status_glyphs_match_the_qt_cards` compares the two for as long
# as both exist - which is the twin-detector this repo keeps needing.
STATUS_GLYPHS: dict[str, tuple[str, str]] = {
    "OK": ("●", hex_str(SUCCESS)),
    "WATCH": ("◐", hex_str(WARNING)),
    "ALERT": ("●", hex_str(DANGER)),
    "IDLE": ("○", hex_str(TEXT_TERTIARY)),
}


def build_header(payload: dict[str, Any], connection: str) -> dict[str, Any]:
    """The top strip: session, driver, connection, playback, lap counter."""
    arcade = payload.get("arcade") or {}
    strategy = payload.get("strategy") or {}
    playback = payload.get("playback") or {}
    start = strategy.get("start") or {}

    gp = start.get("gp") or arcade.get("gp_name") or "--"
    year = start.get("year") or arcade.get("year") or "--"
    try:
        speed = float(playback.get("speed", 1.0))
    except (TypeError, ValueError):
        speed = 1.0
    paused = bool(playback.get("paused", False))

    return {
        "session": f"{gp} · {year}",
        "driver": str(start.get("driver") or arcade.get("driver_main") or "--"),
        "lap": f"L {arcade.get('lap', 0)}/{arcade.get('total_laps', 0)}",
        "playback": f"{speed:.2f}× · {'PAUSED' if paused else 'PLAYING'}",
        "connection": connection,
        "connection_colour": CONNECTION_COLOURS.get(connection, hex_str(TEXT_SECONDARY)),
    }


def _card(formatted: tuple, tooltip: str = "") -> dict[str, Any]:
    """One formatter tuple as JSON: `(headline, colour, lines, status)`."""
    headline, headline_colour, lines, status = formatted
    glyph, glyph_colour = STATUS_GLYPHS.get(status, STATUS_GLYPHS["IDLE"])
    return {
        "headline": headline,
        "headline_colour": hex_str(headline_colour),
        "lines": [{"text": text, "colour": hex_str(colour)} for text, colour in lines],
        "status": status,
        "glyph": glyph,
        "glyph_colour": glyph_colour,
        "tooltip": tooltip,
    }


def build_cards(latest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """The six agent cards, straight out of the Qt formatters.

    Transcribed from `MainWindow._update_agent_cards`, including the
    branch where the whole `per_agent` block is missing: there the four
    always-on cards render their own no-output text and the two
    conditional ones render their trigger hint, which is not the same
    thing as an empty card.

    `active` carries agent IDs (`N28`, `N30`), not block names. The dev
    producer sent block names for a while and both conditional cards sat
    on their trigger hint no matter what it published (#853).
    """
    per = (latest or {}).get("per_agent") if latest else None
    if not per:
        return {
            "pace": _card(format_pace(None)),
            "tire": _card(format_tire(None)),
            "situation": _card(format_situation(None)),
            "pit": _card(format_pit(None, active=False)),
            "radio": _card(format_radio(None)),
            "rag": _card(format_rag(None, active=False)),
        }

    active = set(per.get("active") or [])
    radio_block = per.get("radio")
    # `rag` is the structured payload; `regulation_context` stays as a
    # legacy fallback for producers that have not been updated.
    rag_block = per.get("rag") or per.get("regulation_context")
    rag_active = "N30" in active
    return {
        "pace": _card(format_pace(per.get("pace"))),
        "tire": _card(format_tire(per.get("tire"))),
        "situation": _card(format_situation(per.get("situation"))),
        "pit": _card(format_pit(per.get("pit"), active="N28" in active)),
        "radio": _card(format_radio(radio_block), radio_tooltip_html(radio_block)),
        "rag": _card(
            format_rag(rag_block, active=rag_active),
            rag_tooltip_html(rag_block) if rag_active else "",
        ),
    }


def build_status_bar(payload: dict[str, Any]) -> dict[str, Any]:
    """The bottom line: the pipeline error, or the lap while streaming.

    `transient` mirrors the 1.5 s timeout Qt's status bar applies to the
    streaming message and not to the error, which is the difference
    between "here is what is happening" and "here is what went wrong".
    """
    strategy = payload.get("strategy") or {}
    error = strategy.get("error")
    if error:
        return {"text": f"pipeline: {error}", "transient": False}
    lap = (payload.get("arcade") or {}).get("lap", "?")
    return {"text": f"lap {lap} · streaming", "transient": True}
