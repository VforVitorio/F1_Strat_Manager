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

# The AGENTS window's content layer, which moved OUT of the Qt package in
# sprint 7 rather than dying with it: PITWALL renders by calling these, which
# is what made the port 1:1 by construction instead of by inspection.
REUSED_BY_PITWALL = (
    "src.pitwall.agent_formatters",
    "src.pitwall.reasoning_lines",
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


def test_a_rewind_keeps_the_laps_it_already_observed():
    """The Qt window keeps them, and so must this: the wire sends them once.

    An earlier version evicted every lap ahead of the seek. Two things
    killed it. The replay is deterministic, so those observations are not
    wrong, only early. And a forward jump past the evicted range never
    re-drives them, so the prediction is gone: `history_tail` strips
    `per_agent`, which is exactly the loss Gate A's D-11 predicted.

    The eviction also leaked. On a tick where the arcade clock goes back
    but `strategy.latest` still lags at the old lap, it removed the future
    and `ingest_latest` re-added the lagging lap on the same tick: a store
    holding 28/29/30 rewound to 10 ended up holding **only lap 30** — it
    deleted the two it should have kept and kept the one it meant to drop.
    """
    from src.pitwall.host import PitwallHost

    client = _FakeClient(_payload(seq=1, lap=28))
    host = PitwallHost(client, window_count=1)
    host.get_agents_view(-1)
    for seq, lap in ((2, 29), (3, 30)):
        client.latest = _payload(seq=seq, lap=lap)
        host.get_agents_view(seq - 1)

    # The rewind tick: the clock goes back to 10 while `latest` still lags.
    client.latest = _payload(seq=4, lap=10)
    client.latest["strategy"]["latest"]["lap_number"] = 30
    view = host.get_agents_view(3)

    laps = sorted(row["lap"] for row in view["history"]["pace"])
    assert laps == [28, 29, 30], "nothing observed is thrown away by a seek"


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
    # The caller says what it holds, on BOTH axes. `since_connection` joined
    # `since_seq` in #950: a host field could not answer "changed since YOU
    # looked" for two consumers, so the question is asked instead of remembered.
    assert host.get_agents_view(5, "Connected") is None, "nothing new, nothing changed"

    client.connected = False
    view = host.get_agents_view(5, "Connected")

    assert view is not None, "the state changed, so the window must hear about it"
    assert view["header"]["connection"] == "Disconnected"
    assert view["header"]["connection_colour"] == "#ef4444"
    assert host.get_agents_view(5, "Disconnected") is None, "and only once"


def test_before_the_first_connection_the_chip_says_connecting():
    """Retrying is not the same as having been dropped, and the colours differ."""
    view = _host(_payload(), connected=False).get_agents_view(-1)

    assert view["header"]["connection"] == "Connecting..."
    assert view["header"]["connection_colour"] == "#f59e0b"


def test_nothing_to_render_is_none_rather_than_an_empty_view():
    assert _host(None, connected=False).get_agents_view(-1) is None


# --- The decision panel -----------------------------------------------------


def test_the_orchestrator_card_is_the_qt_one_field_for_field():
    view = _host(_payload()).get_agents_view(-1)["orchestrator"]

    assert view["action"] == "PIT NOW"
    assert view["action_colour"] == "#ef4444"
    assert view["confidence_label"] == "Confidence: 71%"
    assert view["confidence_colour"] == "#10b981", "0.71 is over the 0.66 green tier"
    assert view["pace"] == "Pace: PUSH"
    assert view["pace_colour"] == "#ef4444"
    assert view["risk"] == "Risk: AGGRESSIVE"
    assert view["plan"].startswith("Pit: L24 · Next: <span")
    assert view["plan"].endswith("· UCUT: RUS")
    assert view["guardrail"] == ""


def test_the_confidence_tiers_are_the_three_qt_paints():
    from src.pitwall.agents_view.decision import build_orchestrator

    tiers = {
        conf: build_orchestrator({"confidence": conf})["confidence_colour"]
        for conf in (0.0, 0.32, 0.33, 0.65, 0.66, 1.0)
    }

    assert tiers == {
        0.0: "#ef4444",
        0.32: "#ef4444",
        0.33: "#f59e0b",
        0.65: "#f59e0b",
        0.66: "#10b981",
        1.0: "#10b981",
    }


def test_an_empty_plan_on_stay_out_says_the_stint_continues():
    """Three "--" chips read as noise; the orchestrator leaves them blank on purpose."""
    from src.pitwall.agents_view.decision import build_orchestrator

    stay = build_orchestrator({"action": "STAY_OUT", "confidence": 0.5})
    other = build_orchestrator({"action": "UNDERCUT", "confidence": 0.5})
    idle = build_orchestrator(None)

    assert stay["plan"] == "stint continues · no pit window yet"
    assert other["plan"] == "Pit plan pending"
    assert idle["plan"] == "Pit: -- · Next: -- · UCUT: --"
    assert idle["action"] == "--"


def test_the_guardrail_line_only_exists_when_the_orchestrator_overrode_the_winner():
    from src.pitwall.agents_view.decision import build_orchestrator

    assert build_orchestrator({"guardrail_reason": "min stint"})["guardrail"] == (
        "⚠ Guardrail: min stint"
    )
    assert build_orchestrator({"guardrail_reason": None})["guardrail"] == ""


def test_the_scenario_bars_normalise_across_scores_that_are_all_negative():
    """The Monte Carlo scores are gains against STAY_OUT and are frequently all negative.

    Shifting by the minimum before scaling is what keeps the widths valid.
    Get it wrong and the winner is whichever row happens to be least
    negative in absolute terms, which is a different scenario.
    """
    from src.pitwall.agents_view.decision import build_scenarios

    rows = {
        row["key"]: row
        for row in build_scenarios(
            {"STAY_OUT": -0.90, "PIT_NOW": -0.10, "UNDERCUT": -0.50, "OVERCUT": -1.30}
        )
    }

    assert rows["PIT_NOW"]["fill"] == 1.0 and rows["PIT_NOW"]["is_winner"]
    assert rows["OVERCUT"]["fill"] == 0.0
    assert rows["UNDERCUT"]["fill"] == pytest.approx((-0.50 + 1.30) / 1.20)
    assert rows["STAY_OUT"]["score"] == "-0.90"
    assert rows["PIT_NOW"]["bar_colour"] == "#a78bfa", "the winner is the accent"
    assert rows["STAY_OUT"]["bar_colour"] == "#d1d5db"


def test_the_percentage_the_bar_actually_renders_is_on_the_0_to_100_scale():
    """`fill_pct` is the number the stylesheet consumes, and nothing pinned it.

    #876 moved this arithmetic out of TSX so it would be testable and then
    tested `fill` instead, which no longer renders anything: regressing
    `fill_pct` to `round(fill, 1)` left every Python test and the browser
    smoke green while every bar drew at under one percent. The scale is
    the whole point of the field, so it is what this asserts.
    """
    from src.pitwall.agents_view.decision import build_scenarios

    rows = {
        row["key"]: row
        for row in build_scenarios(
            {"STAY_OUT": -0.90, "PIT_NOW": -0.10, "UNDERCUT": -0.50, "OVERCUT": -1.30}
        )
    }

    assert rows["PIT_NOW"]["fill_pct"] == 100.0, "the winner fills the bar, in percent"
    assert rows["OVERCUT"]["fill_pct"] == 0.0
    assert rows["UNDERCUT"]["fill_pct"] == pytest.approx(66.7, abs=0.05)
    for row in rows.values():
        assert row["fill_pct"] == pytest.approx(row["fill"] * 100, abs=0.05), (
            "the two fields must agree; they are one number in two units"
        )


def test_a_scenario_the_orchestrator_did_not_score_draws_nothing_and_prints_dashes():
    """Absent is not zero, and an empty bar with `--` is how the Qt row says so."""
    from src.pitwall.agents_view.decision import build_scenarios

    rows = {row["key"]: row for row in build_scenarios({"PIT_NOW": 0.7, "STAY_OUT": 0.2})}

    assert rows["OVERCUT"]["score"] == "  --"
    assert rows["OVERCUT"]["fill"] == 0.0
    assert rows["OVERCUT"]["is_winner"] is False
    assert [row["key"] for row in build_scenarios(None)] == [
        "STAY_OUT",
        "PIT_NOW",
        "UNDERCUT",
        "OVERCUT",
    ], "all four rows exist even with no scores at all"


def test_the_action_badge_comes_from_classify_action_and_is_not_a_second_table():
    """A hand-copied twin of that table lived in theme.py until 2026-08-01 and drifted."""
    from src.arcade.strategy import classify_action
    from src.pitwall.agents_view.decision import build_orchestrator

    for action in ("STAY_OUT", "PIT_NOW", "UNDERCUT", "OVERCUT", "DNF", "SOMETHING_NEW"):
        colour, label = classify_action(action)
        view = build_orchestrator({"action": action, "confidence": 0.5})
        assert view["action"] == label
        assert view["action_colour"] == "#{:02x}{:02x}{:02x}".format(*colour)


# --- The reasoning tabs -----------------------------------------------------


def _joined(segments) -> str:
    return "".join(segment["text"] for segment in segments)


def test_the_highlighter_loses_no_characters():
    """A renderer that drops a run drops text, and prose is the whole panel."""
    from src.pitwall.agents_view.reasoning import highlight

    text = "PIT_NOW on lap 24: P10 is 34% and the delta is -0.42 s.\nSecond line."

    assert _joined(highlight(text)) == text
    assert highlight("") == []


def test_the_highlighter_colours_the_five_things_qt_colours():
    from src.pitwall.agents_view.reasoning import DEFAULT_COLOUR, highlight

    coloured = {
        segment["text"]: (segment["colour"], segment["bold"])
        for segment in highlight("lap 24 P10 34% -0.42 s PIT_NOW plain")
        if segment["colour"] != DEFAULT_COLOUR
    }

    assert coloured["lap 24"] == ("#f472b6", False)
    assert coloured["P10"] == ("#d946ef", False)
    assert coloured["34%"] == ("#facc15", False)
    assert coloured["-0.42 s"] == ("#22d3ee", False)
    assert coloured["PIT_NOW"] == ("#facc15", True), "the action keywords are the bold rule"
    assert "plain" not in coloured


def test_a_later_rule_overwrites_an_earlier_one_where_they_overlap():
    """Qt's `setFormat` overwrites, and the rules run in a fixed order.

    Emitting the FIRST match instead would leave the action keyword
    un-bolded wherever an earlier pattern happened to reach it, which is
    a difference nobody would notice until the one lap it matters.
    """
    from src.pitwall.agents_view.reasoning import highlight

    # `P10` is both a quantile and, inside this token, nothing else; the
    # overlap case is a delta immediately followed by an action keyword.
    segments = highlight("-0.42 s PIT_NOW")
    bold = [segment for segment in segments if segment["bold"]]

    assert [segment["text"] for segment in bold] == ["PIT_NOW"]
    assert _joined(segments) == "-0.42 s PIT_NOW"


def test_the_memory_block_appears_only_on_a_lap_where_the_call_changed():
    """Unconditional display was measured as wallpaper, and it is the load-bearing field.

    DecisionMemory leaves no trace in `reasoning` even when it drives the
    call, so this block is the only place the continuity is visible.
    """
    from src.pitwall.agents_view.reasoning import build_reasoning

    def body(plan_changed: bool) -> str:
        latest = {
            "reasoning": "the undercut window opens now",
            "memory_block": "lap 22: STAY_OUT (0.58)",
            "plan_changed": plan_changed,
        }
        tabs = {tab["key"]: tab for tab in build_reasoning(latest)}
        return _joined(tabs["orchestrator"]["segments"])

    assert "why this call changed" in body(True)
    assert "lap 22: STAY_OUT (0.58)" in body(True)
    assert "why this call changed" not in body(False)


def test_a_tab_with_no_reasoning_still_shows_the_agents_numbers():
    """The metrics layer is the fallback, and a port that dropped it would look fine.

    `reasoning_lines.py` is not a prettier `agent_formatters`: it renders
    the same fields as a raw dump, and it is the only thing on screen when
    an agent produced no LLM reasoning at all.
    """
    from src.pitwall.agents_view.reasoning import build_reasoning

    tabs = {
        tab["key"]: _joined(tab["segments"])
        for tab in build_reasoning({"per_agent": {"pace": {"lap_time_pred": 81.0, "ci_p10": 80.4}}})
    }

    assert "lap_time_pred   = 81.000s" in tabs["pace"]
    assert "ci_p10          = 80.40s" in tabs["pace"]
    assert tabs["tire"] == "— agent idle —", "an agent with no output says so"
    assert [tab["key"] for tab in build_reasoning(None)] == [
        "orchestrator",
        "pace",
        "tire",
        "situation",
        "radio",
        "pit",
    ]


# --- The two embedded charts ------------------------------------------------


def test_a_stub_lap_time_never_reaches_a_chart():
    """One value orders of magnitude off flattens the real series to a hairline.

    The TCN emits them on the first laps of a stint, and the guard is the
    30-200 s window both Qt charts apply before plotting anything.
    """
    from src.pitwall.agents_view.charts import build_pace_series, build_tire_series

    pace = build_pace_series({1: {"actual": 4000.0}, 2: {"actual": 81.0}, 3: {"actual": 2.0}})
    tire = build_tire_series(
        [
            {"lap": 1, "lap_time_s": 4000.0, "compound": "MEDIUM"},
            {"lap": 2, "lap_time_s": 81.0, "compound": "MEDIUM"},
        ],
        current_lap=2,
        tire_out=None,
    )

    assert pace["actual"] == [[2.0, 81.0]]
    assert tire["stints"][0]["points"] == [[2.0, 81.0]]


def test_the_compound_change_is_a_break_and_not_a_colour_change_on_one_line():
    """A single line would be drawn through the in-lap and the out-lap.

    Those are neither the same length as each other nor as a racing lap,
    so the segment between them is a straight line through a number that
    never happened. One series per stint is the fix, and the colour break
    is where the compound changed.
    """
    from src.pitwall.agents_view.charts import build_tire_series

    rows = [
        {"lap": 1, "lap_time_s": 81.0, "compound": "MEDIUM"},
        {"lap": 2, "lap_time_s": 81.4, "compound": "MEDIUM"},
        {"lap": 3, "lap_time_s": 83.0, "compound": "HARD"},
        {"lap": 4, "lap_time_s": 82.2, "compound": "HARD"},
    ]

    stints = build_tire_series(rows, current_lap=4, tire_out=None)["stints"]

    assert [stint["compound"] for stint in stints] == ["MEDIUM", "HARD"]
    assert [len(stint["points"]) for stint in stints] == [2, 2]
    assert stints[0]["colour"] != stints[1]["colour"]


def test_a_missing_lap_time_does_not_split_a_stint_and_a_missing_compound_inherits():
    from src.pitwall.agents_view.charts import build_tire_series

    rows = [
        {"lap": 1, "lap_time_s": 81.0, "compound": "SOFT"},
        {"lap": 2, "lap_time_s": None, "compound": "SOFT"},
        {"lap": 3, "lap_time_s": 81.5, "compound": None},
    ]

    stints = build_tire_series(rows, current_lap=3, tire_out=None)["stints"]

    assert len(stints) == 1, "a gap in the measurement is not a pit stop"
    assert stints[0]["compound"] == "SOFT"
    assert [point[0] for point in stints[0]["points"]] == [1.0, 3.0]


def test_the_trend_is_a_three_lap_centred_mean_with_min_periods_one():
    """Heavier smoothing lags visibly over a 30-lap window, which defeats it.

    The edges average over whatever exists so the line starts at the first
    lap rather than at `window // 2`.
    """
    from src.pitwall.agents_view.charts import _rolling_mean

    assert _rolling_mean([1.0, 2.0, 3.0, 10.0]) == pytest.approx([1.5, 2.0, 5.0, 6.5])
    assert _rolling_mean([]) == []
    assert _rolling_mean([7.0]) == [7.0]


def test_the_cliff_band_is_anchored_on_the_current_lap():
    """The percentiles are laps REMAINING, so the band is `lap + p`, not `p`."""
    from src.pitwall.agents_view.charts import build_tire_series

    cliff = build_tire_series(
        [{"lap": 23, "lap_time_s": 81.0, "compound": "MEDIUM"}],
        current_lap=23,
        tire_out={"laps_to_cliff_p10": 4.0, "laps_to_cliff_p50": 6.0, "laps_to_cliff_p90": 9.0},
    )["cliff"]

    assert cliff == {"lo": 27.0, "hi": 32.0, "p50": 29.0}


def test_an_early_stint_projection_suppresses_the_whole_annotation():
    """The MC Dropout samples return tens of thousands of laps on lap 1-3.

    An unreadable band is worse than none, and a zero would be a lap
    number the chart could plot.
    """
    from src.pitwall.agents_view.charts import build_tire_series

    rows = [{"lap": 2, "lap_time_s": 81.0, "compound": "MEDIUM"}]
    absurd = {"laps_to_cliff_p10": 40000.0, "laps_to_cliff_p50": -3.0, "laps_to_cliff_p90": 90000.0}

    assert build_tire_series(rows, current_lap=2, tire_out=absurd)["cliff"] is None
    assert build_tire_series(rows, current_lap=None, tire_out=absurd)["cliff"] is None
    assert build_tire_series(rows, current_lap=2, tire_out=None)["cliff"] is None


def test_the_lap_axis_stops_just_past_the_band_and_not_a_hundred_laps_later():
    """The deliberate deviation from Qt, asserted so it cannot drift back.

    `tire_chart.py` extends the axis by the whole 100-lap horizon whenever
    a band is visible, which on lap 23 runs the axis to 123 and squeezes
    the stint into eight per cent of the width. It fires on every normal
    cliff, not just a bad one.
    """
    from src.pitwall.agents_view.charts import build_tire_series

    series = build_tire_series(
        [{"lap": lap, "lap_time_s": 81.0, "compound": "MEDIUM"} for lap in range(14, 24)],
        current_lap=23,
        tire_out={"laps_to_cliff_p10": 4.0, "laps_to_cliff_p50": 6.0, "laps_to_cliff_p90": 9.0},
    )

    assert series["x_range"] == [13.5, 35.0]


def test_the_pace_series_are_independent_so_a_missing_prediction_draws_nothing():
    """A lap with an actual and no prediction must not pull the dashed line to zero."""
    from src.pitwall.agents_view.charts import build_pace_series

    series = build_pace_series(
        {
            21: {"actual": 81.0},
            22: {"actual": 81.2, "pred": 81.1, "ci_p10": 80.6, "ci_p90": 81.6},
            23: {"pred": 81.3},
        }
    )

    assert series["actual"] == [[21.0, 81.0], [22.0, 81.2]]
    assert series["pred"] == [[22.0, 81.1], [23.0, 81.3]]
    assert series["band"] == [[22.0, 80.6, 81.6]], "a band needs both bounds"


def test_a_rule_cannot_paint_across_a_line_break():
    r"""`QSyntaxHighlighter` runs per paragraph; two of the five rules match `\s`.

    Applying them over the whole string painted things Qt leaves plain.
    Reachable, not theoretical: `clean()` collapses the newlines in
    `reasoning`, but the orchestrator tab appends `memory_block` RAW, and
    a memory block is multi-line free text.
    """
    from src.pitwall.agents_view.reasoning import DEFAULT_COLOUR, highlight

    # `QTextDocument.setPlainText` starts a paragraph on \r\n and on a lone
    # \r as well, so all three end a match. Splitting on \n alone left the
    # carriage return, which is the separator an old-Mac memory block uses.
    separators = ("\n", "\r\n", "\r")
    for separator in separators:
        for text in (
            f"extend the lap{separator}22 target",
            f"the delta is +0.42{separator}s behind",
        ):
            painted = [seg["text"] for seg in highlight(text) if seg["colour"] != DEFAULT_COLOUR]
            assert painted == [], f"{text!r} must paint nothing, as Qt does"
            assert "".join(seg["text"] for seg in highlight(text)) == text, "no character lost"

    # The same tokens on one line still paint, so the fix did not disable them.
    on_one_line = {seg["text"] for seg in highlight("lap 22 and +0.42 s") if seg["bold"] is False}
    assert "lap 22" in on_one_line
    assert "+0.42 s" in on_one_line


# --- A restarted producer must not leave the last race on the charts --------
#
# `host.get_tick` deliberately follows a restarted producer: relaunching the
# arcade with the windows open must not leave them frozen on a dead race. Band
# 4 evicts client-side because `FrameClock` sees the frame index jump
# backwards. This accumulator lives in the HOST process, which does not
# restart with the arcade, so nothing here ever heard about it - the twin that
# never got the signal, and the fix for the frozen window is what armed it.


def _other_race_payload(seq: int, lap: int) -> dict:
    """A DIFFERENT race, as a relaunched arcade would send it."""
    payload = _payload(seq=seq, lap=lap, latest=_latest(lap))
    payload["arcade"]["gp_name"] = "Suzuka"
    payload["strategy"]["start"] = {"gp": "Suzuka", "year": 2025, "driver": "NOR"}
    return payload


def test_a_different_race_does_not_inherit_the_last_ones_laps():
    """Measured before the fix: a `Suzuka · 2025` header over Melbourne's
    laps 14-23, with lap 20 reading Melbourne's pace."""
    from src.pitwall.agents_view import AgentsViewBuilder

    builder = AgentsViewBuilder()
    for lap in range(14, 24):
        builder.build(_payload(seq=lap, lap=lap))
    melbourne = builder.build(_payload(seq=30, lap=23))
    assert len(melbourne["history"]["pace"]) >= 9, "the fixture never accumulated anything"

    suzuka = builder.build(_other_race_payload(seq=1, lap=3))
    laps = [row["lap"] for row in suzuka["history"]["pace"]]
    assert laps == [3], f"the new race inherited {laps}"


def test_the_same_race_relaunched_is_a_restart_too():
    """`start` is identical, so only the sequence can tell.

    It cannot be confused with a rewind: `seq` counts messages the producer
    SENT, so within one run it only rises. A rewind moves the frame index and
    must evict nothing - the case the test below pins.
    """
    from src.pitwall.agents_view import AgentsViewBuilder

    builder = AgentsViewBuilder()
    for lap in range(14, 24):
        builder.build(_payload(seq=lap, lap=lap))

    relaunched = builder.build(_payload(seq=1, lap=2))
    laps = [row["lap"] for row in relaunched["history"]["pace"]]
    assert laps == [2], f"the relaunched run inherited {laps}"


def test_a_rewind_inside_one_run_still_evicts_nothing():
    """The deliberate behaviour this fix must not break.

    The laps ahead of a backwards seek are real, deterministic observations,
    and the predictions among them are broadcast exactly once. A rewind moves
    `frame_index`; the sequence keeps rising, which is what makes the two
    events distinguishable at all.
    """
    from src.pitwall.agents_view import AgentsViewBuilder

    builder = AgentsViewBuilder()
    for lap in range(14, 24):
        builder.build(_payload(seq=lap, lap=lap))

    rewound = _payload(seq=99, lap=15)
    rewound["playback"]["frame_index"] = 10
    view = builder.build(rewound)
    laps = [row["lap"] for row in view["history"]["pace"]]
    assert max(laps) == 23, f"a rewind evicted the future: {laps}"


def test_the_stale_lap_that_setdefault_would_have_made_permanent():
    """The half of the defect that never self-corrected.

    `seed_from_tail` uses `setdefault` on purpose, so a lap whose only carrier
    in the NEW run is the history tail keeps whatever the DEAD run left there.
    Asserting the VALUE, not the lap count: a store that still holds lap 18
    at Melbourne's 81.18 under a Suzuka header is the actual damage.
    """
    from src.pitwall.agents_view import AgentsViewBuilder

    builder = AgentsViewBuilder()
    for lap in range(14, 24):
        builder.build(_payload(seq=lap, lap=lap))

    fresh = _other_race_payload(seq=1, lap=3)
    fresh["strategy"]["history_tail"] = [
        {"lap_number": 18, "lap_time_s": 92.18, "tyre_life": 4, "compound": "MEDIUM"}
    ]
    view = builder.build(fresh)
    lap18 = next((row for row in view["history"]["pace"] if row["lap"] == 18), None)
    assert lap18 is not None and lap18["actual"] == 92.18, (
        f"lap 18 kept the dead race's number: {lap18}"
    )
