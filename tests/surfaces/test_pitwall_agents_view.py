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


def _qt_token(name: str) -> str:
    """One `--qt-*` value, read out of the stylesheet that declares it.

    Not a hex written here. The ground the action is drawn on is a CSS custom
    property, and a copy of it in this file would be one more site to drift -
    which is the whole subject of `test_pitwall_tokens.py` next door.
    """
    import re
    from pathlib import Path

    css = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "pitwall"
        / "ui"
        / "src"
        / "styles"
        / "qt-base.css"
    ).read_text(encoding="utf-8")
    match = re.search(rf"{re.escape(name)}:\s*(#[0-9a-fA-F]{{6}})", css)
    assert match, f"{name} is not declared in qt-base.css"
    return match.group(1).lower()


def _rendered_pair(span: str) -> tuple[str, str]:
    """The `(background, foreground)` a pill span actually carries."""
    import re

    background = re.search(r"background-color: (#[0-9a-f]{6})", span)
    foreground = re.search(r"color: (#[0-9a-f]{6})", span.split("background-color:", 1)[1])
    assert background and foreground, span
    return background.group(1), foreground.group(1)


def test_every_pill_and_badge_is_legible_against_its_own_fill():
    """The sprint-8 gate's contrast finding, asserted as a RATIO (#965).

    A test that pinned the hexes would have passed happily on the defect:
    white on the alert chip's grey is 2.54:1, white on the STAY OUT badge
    is 2.54:1, and both hexes were exactly the ones the palette intended.
    **The alarm and the decision were the two least legible things on the
    screen**, and the only assertion that can see that is the one about
    the pair.

    Every intent and every compound, not a sample: the defect class this
    repo pays for most is the twin that never got the fix, and a chip
    enumeration with one member checked is that shape waiting to happen.
    """
    from src.arcade.palette import (
        _COMPOUND_COLOUR_BY_LABEL,
        _FLAG_BG_BY_INTENT,
        compound_pill_html,
        contrast_ratio,
        flag_chip_html,
    )

    def rgb(value: str) -> tuple[int, int, int]:
        return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))

    subjects = [(intent, flag_chip_html(intent)) for intent in (*_FLAG_BG_BY_INTENT, "UNKNOWN")]
    subjects += [
        (compound, compound_pill_html(compound))
        for compound in (*_COMPOUND_COLOUR_BY_LABEL, "UNKNOWN")
    ]
    for name, span in subjects:
        background, foreground = _rendered_pair(span)
        ratio = contrast_ratio(rgb(background), rgb(foreground))
        assert ratio >= 4.5, f"{name}: {foreground} on {background} is {ratio:.2f}:1"

    # The action takes the same treatment, against the ground it is now drawn
    # on. It used to be a fill with an ink chosen against it; the band renders
    # it as TEXT on `--qt-panel`, so the pair that has to contrast is the
    # action colour and the card.
    #
    # **Over `_ACTION_STYLE` itself, not a hand-written list.** The old version
    # named five actions; the map holds SEVEN (DNF and ERROR are display states
    # the producer really emits) plus an unknown-key fallback that echoes the
    # raw string in ACCENT. A guard that asserts about part of an enumeration
    # is how the two missing ones would have stayed missing.
    from src.arcade.strategy import _ACTION_STYLE
    from src.pitwall.agents_view.decision import build_orchestrator

    panel = rgb(_qt_token("--qt-panel"))
    for action in (*_ACTION_STYLE, "SOMETHING_A_FUTURE_PRODUCER_SENDS"):
        built = build_orchestrator({"action": action, "confidence": 0.5})
        ratio = contrast_ratio(rgb(built["action_colour"]), panel)
        assert ratio >= 4.5, f"{action}: {built['action_colour']} on the panel is {ratio:.2f}:1"
    idle = build_orchestrator(None)
    assert contrast_ratio(rgb(idle["action_colour"]), panel) >= 4.5, "the idle action too"


def test_the_plan_timeline_spans_the_race_and_not_the_chart_window():
    """The lane draws lap 1 to the flag, so it cannot read the 40-lap stores.

    `LapHistory` keeps a rolling `KEEP_LAPS = 40` window because the charts
    draw one. Melbourne is 57 laps, so from lap 41 a timeline built on those
    stores would lose its opening stint from the left, one lap at a time - and
    it would lose it into the SAME rendering this module uses for "opened
    mid-race", so one costume would cover two truths with the boundary between
    them moving during the race.

    Driven past the trim on purpose: 45 laps ingested, `KEEP_LAPS` laps of tyre
    rows surviving, and the first stint still starting where it started.
    """
    from src.pitwall.agents_view.builder import AgentsViewBuilder
    from src.pitwall.agents_view.history import KEEP_LAPS

    builder = AgentsViewBuilder()
    for lap in range(1, 46):
        latest = _latest(lap)
        latest["compound"] = "MEDIUM" if lap <= 20 else "HARD"
        view = builder.build(_payload(seq=lap, lap=lap, latest=latest))

    timeline = view["plan_timeline"]
    assert len(view["history"]["tire"]) == KEEP_LAPS, "the chart store really did trim"
    assert timeline["first_known_lap"] == 1, "and the timeline still starts at lap 1"
    runs = [(s["lo"], s["hi"], s["compound"]) for s in timeline["segments"] if not s["planned"]]
    assert runs == [(1, 20, "MEDIUM"), (21, 45, "HARD")]
    assert timeline["segments"][0]["left_pct"] == 0.0, "lap 1 is the start of the track"


def test_the_plan_timeline_invents_nothing_it_was_not_told():
    """Three absences, three honest renderings.

    An unknown compound is the one that matters most: the tyre chart's own
    segmentation inherits the previous label and falls back to `"MEDIUM"`,
    which paints a confident yellow over a lap nobody reported. Over a
    40-lap window that is a small lie; over the whole race it would be the
    opening bar.
    """
    from src.arcade.palette import TEXT_TERTIARY, hex_str
    from src.pitwall.agents_view.timeline import build_plan_timeline

    # No `total_laps`: an empty track, and no division by a zero span.
    empty = build_plan_timeline([], None, {}, None, None, "#f59e0b", "Pit plan pending")
    assert empty == {
        "total_laps": 0,
        "first_known_lap": None,
        "segments": [],
        "pit_lap": None,
        "pit_pct": None,
        "cliff": None,
        "current_lap": None,
        "current_pct": None,
        "caption": "Pit plan pending",
    }

    # An unknown compound: neutral, and it says so by carrying no name.
    unknown = build_plan_timeline(
        [{"lo": 1, "hi": 5, "compound": None}], None, {"total_laps": 57}, 5, None, "#f59e0b", "x"
    )
    assert unknown["segments"][0]["compound"] is None
    assert unknown["segments"][0]["colour"] == hex_str(TEXT_TERTIARY)

    # A stop with no compound: no hollow bar. A planned stint the producer
    # never named would be a claim the window made up.
    nameless = build_plan_timeline(
        [], {"pit_lap_target": 24}, {"total_laps": 57}, 23, None, "#f59e0b", "x"
    )
    assert [s for s in nameless["segments"] if s["planned"]] == []
    assert nameless["pit_lap"] == 24, "the marker still shows; only the bar is withheld"


def test_the_last_lap_lands_on_the_flag():
    """Off-by-one, and it is the one everybody writes.

    Dividing by `total_laps` puts lap 57 of 57 at 98.2 % and leaves a sliver of
    track after the chequered flag; the span is `total_laps - 1`. And a stint's
    bar runs to the END of its last lap, so a one-lap stint has width rather
    than none.
    """
    from src.pitwall.agents_view.timeline import build_plan_timeline

    view = build_plan_timeline(
        [{"lo": 1, "hi": 57, "compound": "HARD"}], None, {"total_laps": 57}, 57, None, "#f59e0b", ""
    )

    assert view["current_pct"] == 100.0
    assert view["segments"][0]["left_pct"] == 0.0
    assert view["segments"][0]["width_pct"] == 100.0

    one_lap = build_plan_timeline(
        [{"lo": 30, "hi": 30, "compound": "SOFT"}],
        None,
        {"total_laps": 57},
        30,
        None,
        "#f59e0b",
        "",
    )
    assert one_lap["segments"][0]["width_pct"] > 0, "a one-lap stint is still a bar"


def test_the_narrative_is_cut_at_a_sentence_and_never_mid_number():
    """`why` is the one line of prose the band puts on the glass.

    A naive split on "." is what this must not be. Real orchestrator text
    carries `0.58`, `1.4 s` and `Art. 30.5(m)`, and the first draft of this
    rule - splitting before a capital OR a digit - turned that last one into
    `Art.`, a three-character narrative that reads as a rendering bug.

    **The bias is not to split.** When the rule is unsure the module shows a
    longer sentence and CSS clamps it to three lines; a first sentence that is
    too long is a layout question, one that is truncated is a lie. That is why
    two of the cases below deliberately return the WHOLE string.
    """
    from src.pitwall.agents_view.decision import first_sentence

    assert first_sentence("the undercut window against RUS opens now") == (
        "the undercut window against RUS opens now"
    )
    assert first_sentence("Confidence is 0.58 here. The gap to RUS is 1.4 s.") == (
        "Confidence is 0.58 here."
    )
    assert first_sentence("Pitting now costs 22.4 s. Staying out risks the cliff.") == (
        "Pitting now costs 22.4 s."
    )
    # The abbreviation. Splitting here is what produced `Art.`
    assert first_sentence("Art. 30.5(m) requires two dry specifications. We have not.") == (
        "Art. 30.5(m) requires two dry specifications."
    )
    # A digit opening the next sentence: not split, on purpose.
    assert first_sentence("The delta is -0.204 s. 71% of runs prefer PIT_NOW.") == (
        "The delta is -0.204 s. 71% of runs prefer PIT_NOW."
    )
    # One word before the stop is not a sentence.
    assert first_sentence("Ok. Then we box.") == "Ok. Then we box."
    assert first_sentence("") == ""
    assert first_sentence(None) == ""


def test_the_memory_block_is_reachable_only_on_the_lap_the_call_moved():
    """The counterweight sentence the retired panel showed, and its condition.

    DecisionMemory leaves no trace in `reasoning` even when it drives the call,
    so the block is rendered rather than trusted to the model's own narration -
    and only on a changed lap, because the action moves on a small minority of
    them and unconditional display was measured as wallpaper.
    """
    from src.pitwall.agents_view.decision import build_orchestrator

    changed = build_orchestrator(_latest())
    quiet = build_orchestrator({**_latest(), "plan_changed": False})

    titles = [section["title"] for section in changed["why_detail"]["sections"]]
    assert titles == ["Reasoning", "Why this call changed"]
    assert [s["title"] for s in quiet["why_detail"]["sections"]] == ["Reasoning"]
    assert build_orchestrator(None)["why_detail"] is None, "nothing to open before the first call"


def test_the_orchestrator_tab_body_is_reachable_from_the_why_module():
    """The other half of retiring the reasoning panel.

    Its five agent bodies went to the consoles' tooltips; its ORCHESTRATOR tab
    - the narrative plus the memory block - is the band's WHY module and the
    popup behind it. Compared across the two surfaces for the same reason the
    per-agent one is: measuring the tooltip against the builder that fills it
    would pass on a move that dropped half of it.
    """
    view = _host(_payload()).get_agents_view(-1)
    tab = "".join(
        segment["text"]
        for segment in next(t for t in view["reasoning"] if t["key"] == "orchestrator")["segments"]
    )
    rendered = " ".join(
        row["text"]
        for section in view["orchestrator"]["why_detail"]["sections"]
        for row in section["rows"]
    )

    assert view["orchestrator"]["why"], "the module has a sentence on the glass"
    assert view["orchestrator"]["why"] in rendered, "and it is the opening of what the popup holds"
    for line in tab.splitlines():
        stripped = line.strip().strip("- ")
        if not stripped or stripped == "why this call changed":
            continue
        assert stripped in rendered, f"the tab showed {stripped!r} and the popup does not"


def test_every_reasoning_tab_body_is_reachable_from_its_card_tooltip():
    """The condition the layout elevation had to satisfy to retire the tabs.

    The reasoning panel held two things per agent that live nowhere else on
    the window: the agent's OWN sentences for the lap, and its `key = value`
    dump. The elevation spec described those bodies as "snake_case field dumps
    that duplicate what the cards already say", which is false the moment an
    agent produces reasoning - `build_reasoning` composes the sentences ON TOP
    of the numbers - so a move that carried only the numbers would have dropped
    every agent's explanation of why.

    Asserted ACROSS the two surfaces rather than inside either: the tab's own
    body, token by token, against what the card's tooltip renders. A test that
    checked the tooltip against the same builder that fills it would pass on a
    move that dropped half the content, because both halves would have moved
    together in its arithmetic.
    """
    # **The fixture has to CARRY reasoning or this guard is about nothing.**
    # `_latest()` gives every agent numbers and no sentences, so the first
    # draft of this test passed against a mutation that dropped the reasoning
    # half entirely: there was none in the population it measured. The tab
    # bodies and the tooltips agreed, and both agreed with the wrong thing.
    agents = ("pace", "tire", "situation", "radio", "pit")
    latest = _latest()
    for name in agents:
        latest["per_agent"][name]["reasoning"] = (
            f"the {name} agent's own sentence for this lap, which lives nowhere else"
        )

    view = _host(_payload(latest=latest)).get_agents_view(-1)
    tabs = {
        tab["key"]: "".join(segment["text"] for segment in tab["segments"])
        for tab in view["reasoning"]
    }

    checked = 0
    for key in agents:
        tooltip = view["cards"][key]["tooltip"]
        assert tooltip is not None, f"{key} has no tooltip to reach its dump through"
        rendered = " ".join(
            f"{row['lead']} {row['text']}"
            for section in tooltip["sections"]
            for row in section["rows"]
        )
        for line in tabs[key].splitlines():
            token = line.split("=")[0].strip()
            if not token:
                continue
            checked += 1
            assert token in rendered, (
                f"{key}: the tab shows {token!r} and the tooltip does not: {rendered}"
            )

    # The enumeration itself, so an empty `per_agent` cannot make this pass by
    # having nothing to compare - and the sentences separately from the
    # numbers, because they are the half a move can drop while the count still
    # looks healthy.
    assert checked >= 25, f"only {checked} tab lines were cross-checked"
    for key in agents:
        rendered = " ".join(
            row["text"]
            for section in view["cards"][key]["tooltip"]["sections"]
            for row in section["rows"]
        )
        assert "own sentence for this lap" in rendered, f"{key}'s reasoning is not reachable"


def test_the_tooltips_return_data_and_never_markup():
    """What replaces the guarantee the hybrid gives up (#960).

    PITWALL renders the AGENTS window by CALLING the Qt window's own
    formatters, which is what made the port 1:1 by construction. Two of
    them returned Qt's restricted rich-text dialect - `<b>`, `<br>`,
    `&nbsp;`, the subset `QToolTip` parses - and the React side rendered
    it through `dangerouslySetInnerHTML`. Qt was retired in sprint 7; the
    dialect outlived the toolkit that required it.

    The hybrid keeps Python deciding WHAT is said and hands the TSX HOW it
    looks. Content still comes from one place, **so only presentation can
    drift** - and this is what keeps that true: the structure is pinned
    here, so a sentence moving into the renderer fails a test rather than
    quietly becoming a second source of truth.

    It also pins the cap that went away. `radio_tooltip_html` truncated
    every message to 70 characters, the same 70 the card's body ticker
    uses, for a reason `_truncate` states in terms of a 280-340 px QLabel.
    A webview popup is clipped by nothing; the tooltip's only added value
    was more messages, never more of a message.
    """
    from src.pitwall.agent_formatters import radio_tooltip, rag_tooltip

    long_message = "Rear grip is going away, especially through the last sector, " + "x" * 90
    built = radio_tooltip(
        {
            "radio_events": [
                {"driver": "NOR", "message": long_message, "analysis": {"intent": "PROBLEM"}}
            ],
            "rcm_events": [{"lap": 23, "flag": "YELLOW", "message": "Debris in sector 2"}],
        }
    )
    assert built == {
        "sections": [
            {"title": "RCM", "rows": [{"lead": "L23 YELLOW", "text": "Debris in sector 2"}]},
            {"title": "Radio", "rows": [{"lead": "NOR PROBLEM", "text": long_message}]},
        ],
        "footer": None,
    }
    assert "..." not in built["sections"][1]["rows"][0]["text"], (
        "the popup carries the whole message; clamping is the renderer's job"
    )

    assert radio_tooltip(None) is None
    assert radio_tooltip({"radio_events": [], "rcm_events": []}) is None, (
        "`None` rather than an empty string: a falsy value that is also a legitimate "
        "rendering is the sentinel shape this repo keeps paying for"
    )

    chunks = [
        {"article": f"Article {n}", "doc_type": "Sporting Regulations", "year": 2025, "text": "t"}
        for n in range(6)
    ]
    rag = rag_tooltip({"question": "how many compounds?", "chunks": chunks})
    assert rag["sections"][0] == {
        "title": "Question",
        "rows": [{"lead": "", "text": "how many compounds?"}],
    }
    assert rag["sections"][1]["title"] == "Sporting Regulations 2025 — Article 0"
    assert len(rag["sections"]) == 5, "the question plus four chunks"
    assert rag["footer"] == "+2 more"
    assert rag_tooltip({"question": "q", "chunks": []}) is None

    for value in (*built["sections"], built["footer"], *rag["sections"], rag["footer"]):
        assert "<" not in repr(value), f"markup reached the view: {value!r}"


def test_the_two_charts_bound_their_axes_to_what_they_actually_draw():
    """The headline chart fix of sprint 8, which shipped with no guard (#966).

    `y_range` could go back to `None`, the two lap axes could drift apart
    again and the current-lap mark could vanish, and 218 tests would stay
    green - the exit gate's own finding about this sprint.

    Three properties, each the one that broke:

    - **every plotted point sits INSIDE the y bound.** The first version
      bounded on the SMOOTHED trend, so a 22 s in-lap averaged down inside
      it and the raw point the stints really draw clipped off the top.
    - **both charts get the same lap axis**, because they sit side by side
      measuring laps and used to autorange independently.
    - **both mark the current lap**, which neither did.
    """
    from src.pitwall.agents_view.charts import build_pace_series, build_tire_series

    rows = [{"lap": lap, "lap_time_s": 81.2, "compound": "MEDIUM"} for lap in range(14, 23)]
    # A real in-lap: about +22 s, and inside the 30-200 s sanity window.
    rows.append({"lap": 23, "lap_time_s": 103.4, "compound": "MEDIUM"})
    tire = build_tire_series(rows, 23, {"laps_to_cliff_p50": 6.0, "laps_to_cliff_p90": 9.0})

    low, high = tire["y_range"]
    plotted = [point[1] for stint in tire["stints"] for point in stint["points"]]
    plotted += [point[1] for point in tire["trend"]]
    assert all(low <= value <= high for value in plotted), (
        f"a plotted point falls outside the axis it is drawn on: {tire['y_range']} vs "
        f"{min(plotted)}..{max(plotted)}"
    )

    pace = build_pace_series({23: {"actual": 81.2, "pred": 81.0}}, tire["x_range"], 23)
    assert pace["x_range"] == tire["x_range"], "the two lap axes are one axis"
    assert pace["current_lap"] == tire["current_lap"] == 23.0

    # Nothing to bound is `None`, not an invented window around no data.
    assert build_tire_series([], None, None)["y_range"] is None


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

    assert cards["pit"]["headline"] == "stop 22.40s → HARD"
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


def test_an_active_pit_console_is_not_a_warning_unless_something_is_pressing():
    """Being awake is not being worried.

    The console wore WARNING amber and WATCH whenever N28 was routed, which is
    every lap with a pit question - so the state that means "look at this" was
    the state that means "this agent ran", and the reader had no way to tell
    the two apart on a surface built for pre-attentive triage.

    WATCH is kept for the one thing in the block that says something is
    pressing: a recommendation driven by safety-car pressure, which carries a
    different risk profile and a deadline.
    """
    from src.arcade.palette import TEXT_PRIMARY, WARNING, hex_str
    from src.pitwall.agent_formatters import format_pit

    block = {
        "stop_duration_p05": 21.14,
        "stop_duration_p50": 22.40,
        "stop_duration_p95": 24.81,
        "compound_recommendation": "HARD",
    }

    calm = format_pit(block, active=True)
    pressing = format_pit({**block, "sc_reactive": True}, active=True)

    assert calm[0] == "stop 22.40s → HARD", "and it says STOP, not PIT"
    assert hex_str(calm[1]) == hex_str(TEXT_PRIMARY)
    assert calm[3] == "OK"
    assert hex_str(pressing[1]) == hex_str(WARNING)
    assert pressing[3] == "WATCH"
    assert pressing[0].endswith(" · SC")


def test_the_radio_console_ranks_severity_over_recency():
    """A rival's routine message could evict our own driver's problem.

    The card showed `radio_events[-1]`, the newest whatever it said, on a
    surface whose whole purpose is noticing the problem.

    **Severity is read from the agent's own `alerts`**, not from a fourth copy
    of an intent-severity map - this repo already carries three, one of which
    was found missing a key the others had, and this module cannot import the
    agent's config because doing so loads three models.
    """
    from src.pitwall.agent_formatters import format_radio

    events = [
        {"driver": "NOR", "analysis": {"intent": "PROBLEM"}, "message": "no grip at the rear"},
        {"driver": "RUS", "analysis": {"intent": "INFORMATION"}, "message": "box this lap"},
    ]
    block = {
        "radio_events": events,
        "rcm_events": [],
        "alerts": [{"driver": "NOR", "intent": "PROBLEM"}],
    }

    lines = [text for text, _ in format_radio(block)[2]]
    messages = [line for line in lines if ":" in line and "radios" not in line]

    assert messages, "the console shows at least one message"
    assert "no grip at the rear" in messages[0], (
        f"the flagged radio has to lead, not the newest: {messages}"
    )
    assert any("box this lap" in line for line in messages), "and the other still fits"

    # With nothing flagged the order is recency, which is what it always was.
    quiet = {"radio_events": events, "rcm_events": [], "alerts": []}
    quiet_lines = [
        text for text, _ in format_radio(quiet)[2] if ":" in text and "radios" not in text
    ]
    assert "box this lap" in quiet_lines[0], f"unflagged falls back to newest-first: {quiet_lines}"


def test_every_console_state_has_its_own_shape():
    """No console state may be readable by colour alone.

    OK and ALERT were the same filled disc, green against red - the single most
    common colour-vision confusion there is - so the two states a reader most
    needs to tell apart were, for some readers, one state.

    Asserted as DISTINCTNESS over the whole map rather than as "ALERT is a
    triangle": the property is what matters, and a check on one entry would go
    on passing the day a fifth state arrives wearing a disc.
    """
    from src.pitwall.agents_view.panels import STATUS_GLYPHS

    glyphs = [glyph for glyph, _ in STATUS_GLYPHS.values()]
    assert len(set(glyphs)) == len(STATUS_GLYPHS), (
        f"two console states share a glyph and differ only in colour: {STATUS_GLYPHS}"
    )
    colours = [colour for _, colour in STATUS_GLYPHS.values()]
    assert len(set(colours)) == len(STATUS_GLYPHS), f"two states share a colour: {STATUS_GLYPHS}"


def test_an_idle_console_says_what_would_wake_it_in_the_readers_language():
    """The copy a race engineer reads, not the copy a developer writes.

    "no prediction - stub" named a thing about the CODE. "no radio/rcm pipeline
    output" named a pipeline. "triggers on ..." described a routing rule. None
    of the three tells the reader what they need, which is either that this
    agent has no reading for this lap or what would wake it.
    """
    from src.pitwall.agent_formatters import format_pace, format_pit, format_radio, format_rag

    idle = {
        "pace": format_pace(None)[0],
        "radio": format_radio(None)[0],
        "pit": format_pit(None, active=False)[0],
        "rag": format_rag(None, active=False)[0],
    }

    assert idle["pace"] == "no reading this lap"
    assert idle["radio"] == "radio silent"
    assert idle["pit"].startswith("wakes on ")
    assert idle["rag"].startswith("wakes on ")

    # And nothing anywhere says any of it in the old dialect.
    for where, text in idle.items():
        for jargon in ("stub", "pipeline", "triggers on", "rcm"):
            assert jargon not in text.lower(), f"{where} still speaks in code: {text!r}"


def test_the_conditional_cards_read_agent_ids_and_not_block_names():
    latest = _latest()
    latest["per_agent"]["active"] = []

    cards = _host(_payload(latest=latest)).get_agents_view(-1)["cards"]

    assert cards["pit"]["status"] == "IDLE"
    assert cards["pit"]["headline"].startswith("wakes on")
    assert cards["rag"]["status"] == "IDLE"
    assert cards["rag"]["headline"].startswith("wakes on")


def test_a_tick_with_no_per_agent_block_shows_the_idle_copy_and_not_an_empty_card():
    cards = _host(_payload(latest={"lap_number": 3})).get_agents_view(-1)["cards"]

    assert cards["pace"]["headline"] == "no reading this lap"
    assert cards["radio"]["headline"] == "radio silent"
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
    """Retrying is not the same as having been dropped, and the colours differ.

    **And retrying is not a state either.** It was WARNING amber here while the
    DATA window's own strip painted the same socket dim grey, with its argument
    written down beside its rule: "Connecting..." is an ABSENCE, and an absence
    that borrows the green or the amber is a made-up answer. One socket, two
    windows a reader has open side by side, two colours. The map is shared now
    and the argument won.
    """
    from src.arcade.palette import TEXT_TERTIARY, WARNING, hex_str

    view = _host(_payload(), connected=False).get_agents_view(-1)

    assert view["header"]["connection"] == "Connecting..."
    assert view["header"]["connection_colour"] == hex_str(TEXT_TERTIARY)
    assert view["header"]["connection_colour"] != hex_str(WARNING), (
        "an absence must not wear the colour that means something is wrong"
    )


def test_nothing_to_render_is_none_rather_than_an_empty_view():
    assert _host(None, connected=False).get_agents_view(-1) is None


# --- The decision panel -----------------------------------------------------


def test_the_orchestrator_card_is_the_qt_one_field_for_field():
    view = _host(_payload()).get_agents_view(-1)["orchestrator"]

    assert view["action"] == "PIT NOW"
    assert view["action_colour"] == "#ef4444"
    assert view["confidence_text"] == "71%"
    assert view["confidence_colour"] == "#10b981", "0.71 is over the 0.66 green tier"
    assert view["pace"] == "Pace: PUSH"
    # Text colour, not DANGER: a posture is a setting somebody chose, and
    # wearing the alarm colour put `Risk: AGGRESSIVE` in the same red as the
    # guardrail violation one line below it (#964).
    assert view["pace_colour"] == "#d1d5db"
    assert view["risk"] == "Risk: AGGRESSIVE"
    assert view["risk_colour"] == "#d1d5db"
    assert view["plan"].startswith("Pit: L24 · Next: <span")
    assert view["plan"].endswith("· UCUT: RUS")


def test_the_lap_the_call_moves_says_what_it_moved_from():
    """The window's only first-class answer to "what changed" (#968).

    Everything else on the surface overwrites in place ten times a
    second, so a call flipping from STAY OUT to PIT NOW left no trace
    outside a heading in a tab panel.

    Read off `history_tail`, whose entries carry real `LapDecision`
    fields. **Not parsed out of `memory_block`**: that is a multi-line LLM
    prompt block, not the `lap 22: STAY_OUT (0.58)` line the design
    proposal assumed, and building a rendered string out of free text is
    how a surface starts lying the first time the text changes.

    The tail's newest entry is the CURRENT lap - the producer appends to
    history and sets `latest` in the same breath - so the search is for
    the newest EARLIER lap, not for `tail[-2]`.
    """
    from src.pitwall.agents_view.decision import build_orchestrator

    tail = [
        {"lap_number": 21, "action": "STAY_OUT", "confidence": 0.61},
        {"lap_number": 22, "action": "STAY_OUT", "confidence": 0.58},
        {"lap_number": 23, "action": "PIT_NOW", "confidence": 0.71},
    ]
    moved = {"action": "PIT_NOW", "confidence": 0.71, "lap_number": 23, "plan_changed": True}
    assert build_orchestrator(moved, tail)["changed"] == "was STAY OUT (0.58) · L22"

    held = dict(moved, plan_changed=False)
    assert build_orchestrator(held, tail)["changed"] == "", (
        "a held call is not a change, and a chip on every lap is wallpaper"
    )
    assert build_orchestrator(moved, None)["changed"] == "", "no history, no claim about one"
    first = {"action": "STAY_OUT", "lap_number": 1, "plan_changed": True}
    assert build_orchestrator(first, [{"lap_number": 1, "action": "STAY_OUT"}])["changed"] == "", (
        "the tail's newest entry is THIS lap; the first lap of a race has no previous call"
    )


def test_no_posture_wears_the_alarm_colour():
    """DANGER means an alarm, and a posture is a setting somebody chose (#964).

    The sprint-8 gate counted six meanings on one red - imperative action,
    posture, low confidence, radio alert, dead link, rule violation - and
    the sharpest instance was `Risk: AGGRESSIVE` sitting one line above
    `⚠ Guardrail: minimum stint length not met` in the same colour. An
    alarm colour exists for pre-attentive triage; six semantics deny the
    reader exactly that.

    Every posture the two maps know plus an unknown one, because a table
    half-migrated is this repo's most expensive defect shape.
    """
    from src.pitwall.agents_view.decision import _PACE_COLOURS, _RISK_COLOURS, build_orchestrator

    danger = "#ef4444"
    for mode in (*_PACE_COLOURS, "SOMETHING_NEW"):
        built = build_orchestrator({"action": "STAY_OUT", "pace_mode": mode})
        assert built["pace_colour"] != danger, f"pace {mode} wears the alarm colour"
    for posture in (*_RISK_COLOURS, "SOMETHING_NEW"):
        built = build_orchestrator({"action": "STAY_OUT", "risk_posture": posture})
        assert built["risk_colour"] != danger, f"risk {posture} wears the alarm colour"

    # DANGER is left with two owners on this window, both alarm-class facts:
    # the ALERT glyph and the dead-producer chip. The guardrail line was the
    # third until #974.


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


def test_the_window_renders_no_guardrail_line_because_nothing_can_fill_it():
    """#974: a field typed, styled, documented and written by no producer.

    The view used to carry `⚠ Guardrail: <reason>` from
    `latest["guardrail_reason"]`. `run_lap` hardcodes that to None for the
    `rich` profile, `strategy_pipeline` hardcodes `profile="rich"`, and
    `src/arcade/app.py` builds its request with a literal `no_llm=False`, so
    on every arcade path the line was permanently blank.

    Asserted as the KEY BEING ABSENT while a reason is supplied, not as an
    empty string. An empty string is what the defect produced for its whole
    life, so a test pinning `== ""` would have been green throughout and green
    afterwards, which is the shape of guard this repo keeps paying for. The
    key's absence is the only thing that changed.
    """
    from src.pitwall.agents_view.decision import build_orchestrator

    supplied = build_orchestrator({"action": "STAY_OUT", "guardrail_reason": "min stint"})
    idle = build_orchestrator(None)

    assert "guardrail" not in supplied, "the view must not carry a line no producer can fill"
    assert "guardrail" not in idle
    assert "min stint" not in str(supplied), "and nothing may smuggle the reason into another field"


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
    # The worst SCORED candidate keeps a floor rather than vanishing (#963);
    # what it may not do is draw the same bar as a scenario nobody scored.
    assert rows["OVERCUT"]["fill"] == 0.06
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
    assert rows["OVERCUT"]["fill_pct"] == 6.0
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


def test_a_scored_last_place_is_distinguishable_from_a_scenario_nobody_scored():
    """The claim the docstring above made and the pixels denied (#963).

    Min-max sends the worst scored candidate to zero BY CONSTRUCTION, and
    an absent one drew zero too, so `--` and `+0.29` rendered the same
    empty track - a pixel diff over the two bar strips of the live window
    found nought differing pixels. Asserting each row's own fill in
    isolation cannot see that: the defect is that TWO rows agree, so the
    assertion has to be about the pair.

    The worst case is n = 2, where min-max collapses the whole scale.
    """
    from src.pitwall.agents_view.decision import build_scenarios

    rows = {row["key"]: row for row in build_scenarios({"PIT_NOW": 0.71, "STAY_OUT": 0.29})}
    scored_last, never_scored = rows["STAY_OUT"], rows["UNDERCUT"]

    assert scored_last["is_scored"] and not never_scored["is_scored"]
    assert scored_last["fill_pct"] > never_scored["fill_pct"], (
        "a candidate the simulation scored must draw ink a candidate it never considered does not"
    )
    assert scored_last["score"] == "+0.29" and never_scored["score"] == "  --"


def test_an_unenacted_winner_loses_the_crown_and_says_so():
    """The sprint-8 gate's P0 (#962).

    A guardrail can overrule the Monte Carlo winner, and the panel that
    explains the call used to keep crowning the overruled plan in full
    ACCENT while the badge one card up said the opposite. The two panels
    disagreed about which strategy the system was executing.

    What must NOT change is the winner's fill: the simulation really did
    score it highest, and hiding that would replace one lie with another.
    """
    from src.pitwall.agents_view.decision import build_scenarios

    rows = {
        row["key"]: row
        for row in build_scenarios({"PIT_NOW": 0.71, "STAY_OUT": 0.29}, enacted_action="STAY_OUT")
    }

    assert rows["PIT_NOW"]["is_winner"], "the simulation's preference is still reported honestly"
    assert rows["PIT_NOW"]["fill_pct"] == 100.0, "and it keeps the width it earned"
    # `NOT TAKEN`, not `VETOED`: a veto names a mechanism this code cannot
    # see. `guardrail_reason` reaches the window from no producer (#974) and
    # the enacted action can differ because the synthesis chose otherwise.
    assert rows["PIT_NOW"]["note"] == "NOT TAKEN"
    assert not rows["PIT_NOW"]["is_enacted"]
    assert rows["STAY_OUT"]["is_enacted"], "the call the orchestrator published takes the highlight"
    assert rows["STAY_OUT"]["label_colour"] == "#a78bfa"
    assert rows["PIT_NOW"]["label_colour"] != "#a78bfa", "the vetoed plan loses the regalia"


def test_an_action_outside_the_four_scenarios_crowns_none_of_them():
    """The exit gate's P1, and the #962 misread walking back in.

    `ALERT` is the fifth member of the orchestrator's own action Literal
    and it is not a scenario. The highlight fell back to the Monte Carlo
    winner through the door left open for the IDLE case, so the panel
    announced `PIT` as the enacted call on a lap the car was doing
    something else entirely.
    """
    from src.pitwall.agents_view.decision import build_scenarios

    rows = {
        row["key"]: row
        for row in build_scenarios({"PIT_NOW": 0.71, "STAY_OUT": 0.29}, enacted_action="ALERT")
    }
    assert not any(row["is_enacted"] for row in rows.values()), (
        "no scenario was enacted, so no row may wear the crown"
    )
    assert rows["PIT_NOW"]["is_winner"] and rows["PIT_NOW"]["note"] == "NOT TAKEN"


def test_a_tie_has_no_winner_and_invents_no_veto():
    """The exit gate's P2: `max` picked whichever key it met first.

    Two equal scores are not a leader and a loser. The arbitrary winner
    then marked the other `NOT TAKEN` - a claim about a decision nobody
    made - and min-max, with nothing to spread, floored BOTH of the joint
    best to 6 %, which says the opposite of what a tie means.
    """
    from src.pitwall.agents_view.decision import build_scenarios

    rows = {
        row["key"]: row
        for row in build_scenarios({"PIT_NOW": 0.5, "STAY_OUT": 0.5}, enacted_action="STAY_OUT")
    }
    assert not any(row["is_winner"] for row in rows.values()), "a tie has no winner"
    assert all(row["note"] == "" for row in rows.values()), "and nothing to mark as not taken"
    assert rows["PIT_NOW"]["fill_pct"] == rows["STAY_OUT"]["fill_pct"] == 100.0, (
        "joint best draws joint full, not joint floor"
    )
    assert rows["STAY_OUT"]["is_enacted"], "the published action still takes the highlight"


def test_with_no_veto_the_winner_keeps_the_crown_and_nothing_is_marked():
    """The unvetoed path is the common one and must be untouched by #962."""
    from src.pitwall.agents_view.decision import build_scenarios

    scores = {"PIT_NOW": 0.71, "STAY_OUT": 0.29}
    for action in ("PIT_NOW", None):
        rows = {row["key"]: row for row in build_scenarios(scores, enacted_action=action)}
        assert rows["PIT_NOW"]["is_enacted"] and rows["PIT_NOW"]["label_colour"] == "#a78bfa"
        assert all(row["note"] == "" for row in rows.values()), (
            f"nothing is vetoed when the enacted action is {action!r}"
        )


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
