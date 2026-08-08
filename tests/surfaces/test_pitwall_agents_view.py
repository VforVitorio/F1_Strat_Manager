"""What PITWALL's AGENTS window is built out of.

The window is a 1:1 port of the Qt strategy window, and the way that is
kept true is not inspection: **the host calls the same formatters the Qt
window calls**, so the two cannot describe the same lap differently. This
file guards the properties that make that possible.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

REUSED_BY_PITWALL = (
    "src.arcade.dashboard.agent_formatters",
    "src.arcade.palette",
)


def _import_in_a_fresh_interpreter(module: str) -> set[str]:
    """Top-level packages a cold import of `module` pulls in.

    A subprocess, not `sys.modules`: by the time pytest reaches this file
    another test has already imported PySide6, so an in-process check
    would assert about the session's history rather than about the module.
    """
    script = textwrap.dedent(f"""
        import sys
        import {module}  # noqa: F401
        print(",".join(sorted({{name.split(".")[0] for name in sys.modules}})))
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    return set(result.stdout.strip().split(","))


def test_the_reused_formatters_need_no_display_stack_and_no_dataframes():
    """PITWALL's host runs in a process with no Qt, and should not pay for pandas.

    Both were true before the palette split: `agent_formatters` imported
    `dashboard.theme`, which imports PySide6 and — through
    `classify_action` — `src.arcade.strategy`, measured at 0.410 s and
    pandas. Six colour tuples and two badge builders should cost neither.

    This is also what un-skipped the palette-mirror test below: reading
    the Python palette needed libEGL on a headless runner, so the one
    guard against the two copies drifting never ran in CI.
    """
    for module in REUSED_BY_PITWALL:
        loaded = _import_in_a_fresh_interpreter(module)
        assert "PySide6" not in loaded, f"{module} drags in a display stack"
        assert "pandas" not in loaded, f"{module} drags in pandas"
        assert "pyglet" not in loaded and "arcade" not in loaded, f"{module} drags in the replay"


def test_the_badge_builders_escape_what_comes_off_the_wire():
    """Compound labels and alert intents are agent output, not literals here.

    In Qt an unescaped `<` breaks the rich-text parser; in PITWALL these
    strings reach a webview, where the same characters are markup.
    """
    from src.arcade.palette import compound_pill_html, flag_chip_html

    assert "&lt;b&gt;" in compound_pill_html("<b>SOFT")
    assert "<b>" not in compound_pill_html("<b>SOFT").removeprefix("<span")
    assert "&lt;" in flag_chip_html("A<B")


# --- The view the host hands the window -------------------------------------


def _latest(lap: int = 23) -> dict:
    """A LapDecision block with the field names the agents really emit."""
    return {
        "lap_number": lap,
        "compound": "MEDIUM",
        "tyre_life": lap - 8,
        "lap_time_s": 81.234,
        "action": "PIT_NOW",
        "confidence": 0.71,
        "reasoning": "the undercut window against RUS opens now",
        "scenario_scores": {"PIT_NOW": 0.71, "STAY_OUT": 0.29},
        "pace_mode": "PUSH",
        "risk_posture": "AGGRESSIVE",
        "pit_lap_target": lap + 1,
        "compound_next": "HARD",
        "undercut_target": "RUS",
        "memory_block": f"lap {lap - 1}: STAY_OUT (0.58)",
        "plan_changed": True,
        "per_agent": {
            "pace": {
                "lap_time_pred": 81.0,
                "delta_vs_prev": -0.204,
                "delta_vs_median": 0.118,
                "ci_p10": 80.45,
                "ci_p90": 81.55,
            },
            "tire": {
                "compound": "MEDIUM",
                "current_tyre_life": lap - 8,
                "deg_rate": 0.031,
                "laps_to_cliff_p10": 4.0,
                "laps_to_cliff_p50": 6.0,
                "laps_to_cliff_p90": 9.0,
                "warning_level": "MONITOR",
            },
            "situation": {
                "overtake_prob": 0.34,
                "sc_prob_3lap": 0.08,
                "threat_level": "MEDIUM",
                "gap_ahead_s": 1.42,
                "pace_delta_s": -0.12,
            },
            "radio": {"radio_events": [], "rcm_events": [], "alerts": []},
            "pit": {
                "stop_duration_p05": 21.14,
                "stop_duration_p50": 22.40,
                "stop_duration_p95": 24.81,
                "compound_recommendation": "HARD",
                "undercut_prob": 0.63,
                "undercut_target": "RUS",
                "sc_reactive": False,
            },
            "rag": {
                "question": "q",
                "answer": "Two dry specifications are still required.",
                "articles": ["Article 30.5(m)"],
                "chunks": [],
            },
            "active": ["N28", "N30"],
        },
    }


def _payload(
    seq: int = 7, lap: int = 23, latest: dict | None = None, tail: list | None = None
) -> dict:
    return {
        "schema_version": 1,
        "seq": seq,
        "arcade": {
            "gp_name": "Melbourne",
            "year": 2025,
            "lap": lap,
            "total_laps": 57,
            "driver_main": "NOR",
        },
        "playback": {"speed": 2.0, "paused": False, "frame_index": 1000, "total_frames": 9000},
        "strategy": {
            "start": {"gp": "Melbourne", "year": 2025, "driver": "NOR"},
            "latest": _latest(lap) if latest is None else latest,
            "history_tail": tail or [],
            "error": None,
        },
    }


class _FakeClient:
    """Stands in for the socket, so the host's own logic is what is tested."""

    def __init__(self, payload=None, connected=True):
        self.latest = payload
        self.connected = connected
        self.stopped = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True


def _host(payload=None, connected=True):
    from src.pitwall.host import PitwallHost

    return PitwallHost(_FakeClient(payload, connected), window_count=1)


def test_the_view_is_what_the_qt_window_renders_line_for_line():
    """The golden. These strings are the Qt cards' own output, not a re-format.

    If a formatter changes, this changes with it, and that diff is the
    port staying 1:1. A TypeScript reimplementation could satisfy this
    today and drift tomorrow, which is the whole reason the view is
    computed in Python.
    """
    view = _host(_payload()).get_agents_view(-1)

    assert view["header"] == {
        "session": "Melbourne · 2025",
        "driver": "NOR",
        "lap": "L 23/57",
        "playback": "2.00× · PLAYING",
        "connection": "Connected",
        "connection_colour": "#10b981",
    }
    assert view["status_bar"] == {"text": "lap 23 · streaming", "transient": True}

    cards = view["cards"]
    assert cards["pace"]["headline"] == "Δnext -0.204s (81.00s)"
    assert [line["text"] for line in cards["pace"]["lines"]] == [
        "pred 81.00s",
        "vs median +0.12s",
        "±0.55s (CI)",
    ]
    assert cards["pace"]["status"] == "OK"

    assert cards["tire"]["headline"] == "Cliff ~6 laps · L15"
    assert cards["tire"]["status"] == "WATCH"
    assert cards["tire"]["lines"][0]["text"] == "range 4–9 laps"

    assert cards["situation"]["headline"] == "Threat MEDIUM"
    assert cards["situation"]["headline_colour"] == "#f59e0b"
    assert [line["text"] for line in cards["situation"]["lines"]] == [
        "overtake 34%",
        "safety car 8%",
        "gap 1.4s · Δpace -0.12s/lap",
    ]

    assert cards["pit"]["headline"] == "pit 22.40s → HARD"
    assert cards["radio"]["headline"] == "quiet"
    assert cards["rag"]["headline"] == "regulation loaded"


def test_the_situation_card_says_out_of_range_and_never_zero_per_cent():
    """`None` and `0` are opposite readings and would prompt opposite calls.

    N27 reports None when the car ahead is beyond the overtake model's
    trained gap. Rendering that as "overtake 0%" tells the wall the model
    says NO CHANCE when it says nothing at all.
    """
    latest = _latest()
    latest["per_agent"]["situation"]["overtake_prob"] = None

    view = _host(_payload(latest=latest)).get_agents_view(-1)

    assert view["cards"]["situation"]["lines"][0]["text"] == "overtake — (out of model range)"


def test_the_conditional_cards_read_agent_ids_and_not_block_names():
    latest = _latest()
    latest["per_agent"]["active"] = []

    cards = _host(_payload(latest=latest)).get_agents_view(-1)["cards"]

    assert cards["pit"]["status"] == "IDLE"
    assert cards["pit"]["headline"].startswith("triggers on")
    assert cards["rag"]["status"] == "IDLE"
    assert cards["rag"]["headline"].startswith("triggers on")


def test_a_tick_with_no_per_agent_block_shows_the_idle_copy_and_not_an_empty_card():
    cards = _host(_payload(latest={"lap_number": 3})).get_agents_view(-1)["cards"]

    assert cards["pace"]["headline"] == "no prediction — stub"
    assert cards["radio"]["headline"] == "no radio/rcm pipeline output"
    assert all(card["status"] == "IDLE" for card in cards.values())


# --- The accumulators -------------------------------------------------------


def test_the_history_keeps_the_predictions_the_wire_only_sends_once():
    """`history_tail` strips `per_agent`, so nothing can rebuild them later.

    Two ticks a lap apart: the older lap must keep the prediction it
    carried when it was `latest`, and the tail must not overwrite it with
    the actual-only row it also describes.
    """
    from src.pitwall.host import PitwallHost

    client = _FakeClient(_payload(seq=1, lap=23))
    host = PitwallHost(client, window_count=1)
    host.get_agents_view(-1)

    tail = [{"lap_number": 23, "lap_time_s": 81.234, "tyre_life": 15, "compound": "MEDIUM"}]
    client.latest = _payload(seq=2, lap=24, tail=tail)
    view = host.get_agents_view(1)

    pace = {row["lap"]: row for row in view["history"]["pace"]}
    assert pace[23]["pred"] == 81.0, "the prediction survived the tail describing the same lap"
    assert pace[23]["actual"] == 81.234
    assert 24 in pace, "and the new lap is in"


def test_a_rewind_drops_the_laps_ahead_and_keeps_the_ones_behind():
    """The Qt charts have no equivalent and are wrong after a seek back.

    Truncating everything would be worse than doing nothing: the past
    holds predictions the wire will never resend.
    """
    from src.pitwall.host import PitwallHost

    client = _FakeClient(_payload(seq=1, lap=20))
    host = PitwallHost(client, window_count=1)
    host.get_agents_view(-1)
    for seq, lap in ((2, 21), (3, 22)):
        client.latest = _payload(seq=seq, lap=lap)
        host.get_agents_view(seq - 1)

    client.latest = _payload(seq=4, lap=21)
    view = host.get_agents_view(3)

    assert sorted(row["lap"] for row in view["history"]["pace"]) == [20, 21]


def test_the_history_stays_bounded():
    from src.pitwall.agents_view.history import KEEP_LAPS, LapHistory

    history = LapHistory()
    for lap in range(1, KEEP_LAPS + 11):
        history.ingest_latest({"lap_number": lap, "lap_time_s": 80.0 + lap})

    assert len(history.pace) == KEEP_LAPS
    assert min(history.pace) == 11, "the oldest laps go, not the newest"


# --- The connection chip ----------------------------------------------------


def test_the_window_learns_the_arcade_died_even_though_no_tick_arrives():
    """Once the producer stops, `seq` stops advancing.

    A purely sequence-driven view would keep rendering the last frame of a
    dead race under a green Connected chip, forever and silently.
    """
    from src.pitwall.host import PitwallHost

    client = _FakeClient(_payload(seq=5), connected=True)
    host = PitwallHost(client, window_count=1)
    assert host.get_agents_view(-1)["header"]["connection"] == "Connected"
    assert host.get_agents_view(5) is None, "nothing new, nothing changed"

    client.connected = False
    view = host.get_agents_view(5)

    assert view is not None, "the state changed, so the window must hear about it"
    assert view["header"]["connection"] == "Disconnected"
    assert view["header"]["connection_colour"] == "#ef4444"
    assert host.get_agents_view(5) is None, "and only once"


def test_before_the_first_connection_the_chip_says_connecting():
    """Retrying is not the same as having been dropped, and the colours differ."""
    view = _host(_payload(), connected=False).get_agents_view(-1)

    assert view["header"]["connection"] == "Connecting..."
    assert view["header"]["connection_colour"] == "#f59e0b"


def test_nothing_to_render_is_none_rather_than_an_empty_view():
    assert _host(None, connected=False).get_agents_view(-1) is None


def test_the_status_glyphs_match_the_qt_cards():
    """The twin-detector for the one map that had to be repeated.

    `agent_card.py::_GLYPH_FOR` lives inside a QFrame subclass and cannot
    be imported from a process with no Qt, so `panels.STATUS_GLYPHS`
    repeats it. Repeating a map without a guard is how this repo's most
    frequent defect starts, so the two are compared for as long as both
    exist. It skips where Qt does not import; the port's own copy is
    still exercised by every card test above.
    """
    agent_card = pytest.importorskip(
        "src.arcade.dashboard.agent_card",
        reason="the Qt dashboard is an optional surface and needs a display stack",
        exc_type=ImportError,
    )
    from src.arcade.palette import hex_str
    from src.pitwall.agents_view.panels import STATUS_GLYPHS

    qt_map = {
        status: (glyph, hex_str(colour))
        for status, (glyph, colour) in agent_card._GLYPH_FOR.items()
    }

    assert STATUS_GLYPHS == qt_map
