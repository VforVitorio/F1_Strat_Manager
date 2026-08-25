"""Header, agent cards and status bar, as JSON the AGENTS window renders.

Every string and every colour here is produced by the code that painted
the Qt window: `src/pitwall/agent_formatters.py` for the six cards, and the
header/status logic transcribed from that window. Nothing reformats or
re-decides.

The formatters MOVED rather than died: PITWALL renders by calling them, which
is what makes the port 1:1 by construction instead of by inspection.
**`src/arcade/dashboard/` no longer exists** - sprint 7 retired it. It is readable in git history, and what it RENDERED is committed as screenshots under `documents/dev_docs/migration/pitwall/`, which is the baseline this port was checked against.

Colours leave as `#rrggbb` because that is what both a Qt stylesheet and
a CSS declaration take, and because the AGENTS window renders in the QT
palette rather than in `tokens.css`'s semantics: this is a 1:1 port, and
the two palettes deliberately differ (`test_pitwall_tokens.py`). Choosing
the web palette here would BE the redesign sprint 8 owns.
"""

from __future__ import annotations

from typing import Any

from src.arcade.palette import (
    DANGER,
    SUCCESS,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    WARNING,
    hex_str,
)

__all__ = ["CONNECTION_COLOURS", "STATUS_GLYPHS", "build_cards", "build_header", "build_status_bar"]
from src.pitwall.agent_formatters import (
    format_pace,
    format_pit,
    format_radio,
    format_rag,
    format_situation,
    format_tire,
    radio_tooltip,
    rag_tooltip,
    with_model_detail,
)
from src.pitwall.agents_view.routing import ROUTING_LANES

# The three socket states and their colours, for BOTH windows.
#
# **"Connecting..." was WARNING amber here and dim grey on the DATA window's
# own strip**, so one socket wore two colours on two windows a reader has open
# side by side. The strip's argument is the better one and it was already
# written down beside its rule: "Connecting..." is an ABSENCE, not a state, and
# an absence that borrows the green or the amber is a made-up answer. The amber
# came from Qt's `set_connection` - a port decision, never a designed one.
#
# The map is the only owner now. `PitwallHost.get_connection` returns the word
# AND this colour, so the DATA strip stops mapping the same three words to CSS
# classes of its own.
CONNECTION_COLOURS: dict[str, str] = {
    "Connected": hex_str(SUCCESS),
    "Connecting...": hex_str(TEXT_TERTIARY),
    "Disconnected": hex_str(DANGER),
}

# The four console states, as a SHAPE and a colour.
#
# **Both, because colour alone was carrying two of them.** OK and ALERT were
# the same filled disc and differed only in green against red - the single most
# common colour-vision confusion there is - so the two states a reader most
# needs to tell apart were, for some readers, one state. ALERT is a triangle
# now: pointed, conventional for a warning, and legible with the colour
# removed entirely.
#
# The map was transcribed from `agent_card.py::_GLYPH_FOR`, which could not be
# imported because it lived in a QFrame subclass. **That file no longer exists**
# - sprint 7 retired `src/arcade/dashboard/` - and the comment here went on
# naming a `test_the_status_glyphs_match_the_qt_cards` that went with it,
# promising a comparison "for as long as both exist" when only one did. What
# guards the map now is `test_every_console_state_has_its_own_shape`, which
# asserts the property rather than the parity: four states, four distinct
# glyphs, and no state readable by colour alone.
STATUS_GLYPHS: dict[str, tuple[str, str]] = {
    "OK": ("●", hex_str(SUCCESS)),
    "WATCH": ("◐", hex_str(WARNING)),
    "ALERT": ("▲", hex_str(DANGER)),
    "IDLE": ("○", hex_str(TEXT_TERTIARY)),
}


def build_header(payload: dict[str, Any], connection: str) -> dict[str, Any]:
    """The top strip: session, driver, neutralisation, connection, playback, lap.

    **The track status comes off the tick, NOT off the situation agent.** N27
    publishes `sc_currently_active` and `vsc_active` on the same payload, and
    rendering those instead would put two sources for one fact on a desk where
    both windows are open at once, free to disagree: one is FastF1's TrackStatus
    for the lap on screen, decoded by the producer's own priority rule
    (`app.py`'s `track_status_label`) precisely so no consumer re-derives it, and
    the other is a boolean the agent computed for the lap it was asked about. The
    DATA window renders the decoded one; so does this. The agent's pair are a
    deletion, tracked with the schema work rather than as a third wire change.

    The colour rides along for the same reason the label does: the priority order
    and the palette entry are a project rule, and picking a colour here would be a
    second copy of it in a second language.
    """
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
        # Both `None` when the loader has no entry for the lap, which is NOT the
        # same as a green track and must not render as one. The renderer's
        # unknown treatment is what says so.
        "track_status": arcade.get("track_status_label"),
        "track_status_colour": arcade.get("track_status_color"),
    }


def _card(formatted: tuple, tooltip: dict[str, Any] | None = None) -> dict[str, Any]:
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
    # The ids come from the router's own roster, not from literals here. A
    # second place that knows "N28 means pit" is the twin this repository
    # produces most, and the strip below the contingencies reads the same table.
    pit_id, rag_id = (agent_id for agent_id, _ in ROUTING_LANES)
    rag_active = rag_id in active

    # **Every card carries its model detail now, and four of them had no
    # tooltip at all.** The band's WHY module replaces the reasoning tabs, and
    # what the tabs held for the five agents - each one's own sentences plus
    # its `key = value` dump - has to land somewhere reachable or the window
    # loses it. This is the drill-down tier a debugging engineer already opens
    # for a transcript or a RAG chunk, so it joins them rather than adding a
    # seventh place to look.
    return {
        "pace": _card(
            format_pace(per.get("pace")),
            with_model_detail(None, "pace", per.get("pace")),
        ),
        "tire": _card(
            format_tire(per.get("tire")),
            with_model_detail(None, "tire", per.get("tire")),
        ),
        "situation": _card(
            format_situation(per.get("situation")),
            with_model_detail(None, "situation", per.get("situation")),
        ),
        "pit": _card(
            format_pit(per.get("pit"), active=pit_id in active),
            with_model_detail(None, "pit", per.get("pit")),
        ),
        "radio": _card(
            format_radio(radio_block),
            with_model_detail(radio_tooltip(radio_block), "radio", radio_block),
        ),
        # RAG is the one agent with no `reasoning_lines` builder: its tooltip
        # already carries the question and the retrieved chunks, which is the
        # same tier and more of it than a dump would be.
        "rag": _card(
            format_rag(rag_block, active=rag_active),
            rag_tooltip(rag_block) if rag_active else None,
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
