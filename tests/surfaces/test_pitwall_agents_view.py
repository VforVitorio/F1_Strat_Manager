"""What PITWALL's AGENTS window is built out of.

The window is a 1:1 port of the Qt strategy window, and the way that is
kept true is not inspection: **the host calls the same formatters the Qt
window calls**, so the two cannot describe the same lap differently. This
file guards the properties that make that possible.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.surfaces.fake_stream_client import FakeStreamClient as _FakeClient

# The AGENTS window's content layer, which moved OUT of the Qt package
# rather than dying with it: PITWALL renders by calling these, which
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
    `dashboard.theme`, which imports PySide6 and (through
    `classify_action`) `src.arcade.strategy`, measured at 0.410 s and
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
    # on. The band renders it as TEXT on `--qt-panel`, so the pair that has
    # to contrast is the action colour and the card.
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
        view = builder.build(_payload(seq=lap, lap=lap, latest=latest), "Connected")

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

    Dividing by `total_laps` puts lap 57 of 57 at 98.2% and leaves a sliver of
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


def test_the_tyre_console_reports_what_the_wear_has_cost():
    """`deg_cost_s` and `cumulative_deg_s` reach a pixel, but an absent one is not a zero.

    Both ride the tick on every lap because the producer `asdict`s the whole
    `TireOutput`, and until now neither appeared anywhere in PITWALL. `deg_cost_s`
    is the field the scorers consume; `deg_rate`, which the console DID show, is a
    raw derivative that fuel burn cancels and that its own docstring says does not
    separate a worn tyre from a fresh one.

    The second half is the one that matters more. Both fields are `None` rather
    than 0.0 when the TCN did not run, deliberately, because 0.0 is what a
    fresh set reads - so a renderer that defaulted them to a number
    would print a real-looking measurement for a missing one. This asserts the
    absence renders AS an absence, which a test that only checked the populated
    case would never see.

    Asserted on the rendered tooltip rather than on `_tire_lines`, because what is
    being claimed is that a reader can see them.
    """

    def tyre_rows(block: dict) -> str:
        latest = _latest()
        latest["per_agent"]["tire"].update(block)
        view = _host(_payload(latest=latest)).get_agents_view(-1)
        tooltip = view["cards"]["tire"]["tooltip"]
        assert tooltip is not None, "the TIRE card has no tooltip to carry a dump"
        return " | ".join(
            f"{row['lead']}={row['text']}"
            for section in tooltip["sections"]
            for row in section["rows"]
        )

    measured = tyre_rows({"deg_cost_s": 0.412, "cumulative_deg_s": 1.284})
    assert "deg_cost_s=0.412s/lap" in measured, (
        f"the cost the scorers read is not shown: {measured}"
    )
    assert "cumulative_deg_s=1.284s/lap" in measured, f"the level is not shown: {measured}"

    absent = tyre_rows({"deg_cost_s": None, "cumulative_deg_s": None})
    for field in ("deg_cost_s", "cumulative_deg_s"):
        assert f"{field}=—s/lap" in absent, f"{field} does not render its absence: {absent}"
        assert f"{field}=0.000s/lap" not in absent, (
            f"{field} printed a fresh-tyre reading for a missing one: {absent}"
        )


# A string that is harmless as TEXT and a tag as MARKUP. Not a curiosity: the
# orchestrator's `undercut_target` is an unconstrained `Optional[str]` an LLM
# fills, and every card headline and body line still reaches the window through
# `dangerouslySetInnerHTML` because it can carry the compound pill.
_HOSTILE = '<img src=x onerror="boom">'


def test_no_agent_string_can_become_markup():
    """Every formatter, every string-shaped field, one hostile value.

    **Written because the enumeration is what fails, not the escaping.** The
    module has had `_escaped` and a docstring claiming every free-text field
    went through it. Six fields did not, one of them reaching a
    `dangerouslySetInnerHTML` sink. A test naming the six would be the same
    list one layer down.

    Therefore, this feeds `<img src=x onerror=...>` into every string field the
    formatters accept and asserts no unescaped `<` survives into a headline or
    a body line. A new field lands in this test the day it lands in a payload,
    without anyone remembering to add it.

    Tooltips are excluded ON PURPOSE and asserted separately: they return DATA
    and the TSX renders React text nodes, so escaping there would put
    `&lt;` on the screen as characters.
    """
    from src.pitwall.agent_formatters import (
        format_pace,
        format_pit,
        format_radio,
        format_rag,
        format_situation,
        format_tire,
    )

    blocks = {
        "pace": (format_pace, {"lap_time_pred": 81.0, "delta_vs_prev": -0.2}),
        "tire": (
            format_tire,
            {"compound": _HOSTILE, "warning_level": _HOSTILE, "laps_to_cliff_p50": 6.0},
        ),
        "situation": (format_situation, {"threat_level": _HOSTILE, "gap_ahead_s": 1.4}),
        "radio": (
            format_radio,
            {
                "radio_events": [{"driver": _HOSTILE, "message": _HOSTILE}],
                "rcm_events": [{"lap": 23, "message": _HOSTILE}],
                "alerts": [],
            },
        ),
        "rag": (
            format_rag,
            {"question": _HOSTILE, "answer": _HOSTILE, "articles": [_HOSTILE], "chunks": []},
        ),
        "pit": (
            format_pit,
            {
                "stop_duration_p50": 22.4,
                "compound_recommendation": _HOSTILE,
                "undercut_prob": 0.6,
                "undercut_target": _HOSTILE,
            },
        ),
    }

    # **The assertion is "no angle bracket survives", not "no `<img` survives".**
    # The first draft looked for the literal tag and passed against a real
    # regression, because `format_situation` upper-cases its field and the
    # headline read `Threat <IMG SRC=...>`. A case-sensitive substring check is
    # a guard that is right for the wrong reason.
    #
    # The palette's own pill and chip spans are the one legitimate markup on
    # these lines, so they are removed first - and the removal is anchored to
    # their exact shape rather than to `<span`, so a hostile string wearing a
    # span of its own is not waved through.
    pill = re.compile(r'<span style="background-color: #[0-9a-f]{6};[^"]*">.*?</span>')

    checked = 0
    for name, (formatter, block) in blocks.items():
        formatted = formatter(block, active=True) if name in {"pit", "rag"} else formatter(block)
        headline, _, lines, _ = formatted
        for text in [headline, *(line for line, _ in lines)]:
            checked += 1
            residue = pill.sub("", text)
            assert "<" not in residue and ">" not in residue, (
                f"{name}: an unescaped angle bracket reached a rendered line: {text!r}"
            )

    assert checked >= 12, f"only {checked} rendered strings were probed"


def test_the_plan_caption_escapes_the_field_an_llm_fills():
    """The sink this sprint added, and the field that rides into it.

    `_plan_line` composes markup - it carries the compound pill - and
    `PlanTimeline` renders it through `dangerouslySetInnerHTML`.
    `undercut_target` is `Optional[str]` on `StrategyRecommendation` with no
    pattern and no length bound, filled by the orchestrator LLM.
    """
    from src.pitwall.agents_view.decision import build_orchestrator

    caption = build_orchestrator(
        {
            "action": "PIT_NOW",
            "confidence": 0.7,
            "pit_lap_target": 24,
            "compound_next": "HARD",
            "undercut_target": _HOSTILE,
        }
    )["plan"]

    residue = re.sub(r'<span style="background-color: #[0-9a-f]{6};[^"]*">.*?</span>', "", caption)
    assert "<" not in residue and ">" not in residue, f"the caption carries a tag: {caption!r}"
    assert "&lt;img" in caption, "and the text is still there, escaped"
    # The pill IS markup and must survive as markup, or the fix would have
    # escaped the one thing on this line that is supposed to be a span.
    assert '<span style="background-color:' in caption


def test_the_tooltips_return_data_and_never_markup():
    """What replaces the guarantee the hybrid gives up (#960).

    PITWALL renders the AGENTS window by CALLING the Qt window's own
    formatters, which is what made the port 1:1 by construction. Two of
    them returned Qt's restricted rich-text dialect - `<b>`, `<br>`,
    `&nbsp;`, the subset `QToolTip` parses - and the React side rendered
    it through `dangerouslySetInnerHTML`. Qt was retired; the
    dialect outlived the toolkit that required it.

    The hybrid keeps Python deciding WHAT is said and hands the TSX HOW it
    looks. Content still comes from one place, **so only presentation can
    drift** - and this is what keeps that true: the structure is pinned
    here, so a sentence moving into the renderer fails a test rather than
    silently becoming a second source of truth.

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


def test_the_agents_header_takes_the_neutralisation_off_the_tick():
    """One source for the race's most decision-changing fact, and an absence stays one.

    The situation agent computes `sc_currently_active` and `vsc_active`, and it
    used to publish them on this same payload. Rendering those would put two
    sources for one fact on a desk where both windows are open: one is FastF1's
    TrackStatus for the lap on screen, decoded once by the producer so nobody
    re-derives it, and the other is a boolean N27 computed for the lap it was
    asked about. This asserts the header reads the decoded one, and that the
    other one is no longer on the wire to be read (#1043).

    The second half is the part that costs if it is wrong. The producer sends
    `None` when the loader has no entry for the lap, and that is NOT a green
    track: a header that defaulted it would render a confident answer to a
    question nobody asked the data.
    """
    green = _host(_payload()).get_agents_view(-1)["header"]
    assert (green["track_status"], green["track_status_colour"]) == ("GREEN", [16, 185, 129])

    under_sc = _payload()
    under_sc["arcade"]["track_status_label"] = "SAFETY CAR"
    under_sc["arcade"]["track_status_color"] = [255, 140, 0]
    sc = _host(under_sc).get_agents_view(-1)["header"]
    assert (sc["track_status"], sc["track_status_colour"]) == ("SAFETY CAR", [255, 140, 0])

    blind = _payload()
    del blind["arcade"]["track_status_label"]
    del blind["arcade"]["track_status_color"]
    unknown = _host(blind).get_agents_view(-1)["header"]
    assert unknown["track_status"] is None, "an absent status became a claim"
    assert unknown["track_status_colour"] is None

    # The pair no longer reaches the wire at all: it is filtered at the DTO
    # boundary (#1043), and that absence is guarded against the REAL producer in
    # `test_arcade_wire_contract.py`, not here - `_payload` is a hand-built dict,
    # so an absence assertion against it would only say what this file chose to
    # write.
    #
    # The injection below is therefore a key no producer sends any more, and the
    # test is deliberately kept that way: what it pins is the VIEW BUILDER, which
    # must not grow a second reader of neutralisation whatever arrives. A payload
    # whose agent says the safety car is out while the tick says green renders
    # green, because the tick is the source.
    disagreeing = _payload()
    disagreeing["strategy"]["latest"]["per_agent"]["situation"]["sc_currently_active"] = True
    still = _host(disagreeing).get_agents_view(-1)["header"]
    assert still["track_status"] == "GREEN", "the header followed the agent instead of the tick"


def test_the_radio_tooltip_carries_the_nlp_corrections():
    """N29's mismatch list reaches the popup, and an LLM string in it is not markup.

    `corrections` rides the tick on every lap, because the producer `asdict`s the
    whole `RadioOutput`, and reached nothing. It is what lets a reader judge how
    much to trust a quoted message: the classifier said PROBLEM, the LLM read the
    text and disagreed, and `alerts` stays deterministic either way.

    Every field of an entry is LLM-filled free text with no pattern and no length
    bound, which is the same shape as the `undercut_target` that reached a markup
    sink in #1037. The tooltip is data rather than markup, so the guarantee here
    is that it stays data.
    """
    from src.pitwall.agent_formatters import radio_tooltip

    built = radio_tooltip(
        {
            "radio_events": [
                {"driver": "NOR", "message": "Box now", "analysis": {"intent": "PROBLEM"}}
            ],
            "rcm_events": [],
            "corrections": [
                {
                    "driver": "NOR",
                    "original_intent": "PROBLEM",
                    "suggested_intent": "INFORMATION",
                    "span": "Box now",
                    "reason": "a routine call, not a complaint",
                }
            ],
        }
    )
    section = built["sections"][-1]
    assert section["title"] == "NLP corrections", (
        f"the corrections never reach the popup: {[s['title'] for s in built['sections']]}"
    )
    assert section["rows"] == [
        {
            "lead": "NOR PROBLEM reads as INFORMATION",
            "text": 'a routine call, not a complaint Quoted: "Box now"',
        }
    ]
    # It qualifies the messages above it, so a reader meets the message first.
    assert [s["title"] for s in built["sections"]] == ["Radio", "NLP corrections"]

    # A lap with a correction and no events still has something to say.
    only = radio_tooltip({"radio_events": [], "rcm_events": [], "corrections": [{"reason": "r"}]})
    assert only is not None and only["sections"][0]["title"] == "NLP corrections"

    # A hostile entry stays DATA: carried verbatim, in a str, never assembled into
    # a markup string here. That is this tooltip's whole contract - it decides what
    # is said and the TSX decides how it looks, so the string reaches a React text
    # node, which is not a parser. The thing to guard is therefore not escaping but
    # SHAPE: the moment any of this is concatenated into markup, the sink is back
    # and the escaping question returns with it (#1037).
    hostile = radio_tooltip(
        {
            "radio_events": [],
            "rcm_events": [],
            "corrections": [
                {
                    "driver": _HOSTILE,
                    "original_intent": _HOSTILE,
                    "suggested_intent": _HOSTILE,
                    "span": _HOSTILE,
                    "reason": _HOSTILE,
                }
            ],
        }
    )
    row = hostile["sections"][0]["rows"][0]
    assert set(row) == {"lead", "text"} and all(isinstance(v, str) for v in row.values())
    assert _HOSTILE in row["text"], f"the hostile string was dropped rather than carried: {row}"
    assert "&lt;" not in row["text"], (
        "escaped here, this string would render as visible entity noise in a text node"
    )


def test_the_two_charts_bound_their_axes_to_what_they_actually_draw():
    """The headline chart fix, which shipped with no guard (#966).

    `y_range` could go back to `None`, the two lap axes could drift apart
    again and the current-lap mark could vanish, and 218 tests would stay
    green.

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
        # **Two branches and two risks, without which every contingency assertion
        # below is about the empty set.** The first carries a `switch_to` whose
        # wire form differs from its label, which is what makes the label
        # assertion able to fail.
        "contingencies": [
            {
                "trigger": "if RUS pits within two laps",
                "switch_to": "PIT_NOW",
                "priority": "HIGH",
                "rationale": "the undercut window shuts once he clears traffic",
            },
            {
                "trigger": "if the safety car is deployed before L28",
                "switch_to": "STAY_OUT",
                "priority": "MEDIUM",
                "rationale": "track position beats the tyre delta under neutralisation",
            },
        ],
        "key_risks": ["rejoin into traffic", "the cliff arrives before the stop"],
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
        "schema_version": 2,
        "seq": seq,
        "arcade": {
            "gp_name": "Melbourne",
            "year": 2025,
            "lap": lap,
            "total_laps": 57,
            "driver_main": "NOR",
            # The decoded pair the producer publishes so no consumer reads the
            # digits. Present here rather than omitted, because a fixture without
            # them exercises only the unknown branch of everything downstream.
            "track_status_label": "GREEN",
            "track_status_color": [16, 185, 129],
        },
        "playback": {"speed": 2.0, "paused": False, "frame_index": 1000, "total_frames": 9000},
        "strategy": {
            "start": {"gp": "Melbourne", "year": 2025, "driver": "NOR"},
            "latest": _latest(lap) if latest is None else latest,
            "history_tail": tail or [],
            "error": None,
        },
    }


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
        "track_status": "GREEN",
        "track_status_colour": [16, 185, 129],
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
    """`None` and `0` are opposite readings, so they prompt opposite calls.

    N27 reports None when the car ahead is beyond the overtake model's
    trained gap. Rendering that as "overtake 0%" tells the wall the model
    says NO CHANCE when it says nothing at all.
    """
    latest = _latest()
    latest["per_agent"]["situation"]["overtake_prob"] = None

    view = _host(_payload(latest=latest)).get_agents_view(-1)

    assert view["cards"]["situation"]["lines"][0]["text"] == "overtake — (out of model range)"


def test_the_pace_chart_says_where_its_prediction_stopped():
    """The solid line advances and the dashed one does not.

    A tick with no `per_agent` block still carries `lap_time_s`, so the actual
    keeps being plotted while the prediction and its band stay where the last
    one was - and nothing on the chart said so. The reader saw two lines, one
    of which had silently become history.

    **Only the PACE chart.** The elevation spec dimmed the tyre chart's TREND
    too, and that is backwards: the trend is a rolling mean of OBSERVED lap
    times and keeps advancing on exactly the tick that makes the prediction
    stale, so dimming it would claim staleness about live data. The tyre
    chart's own stale element is the cliff band, which is recomputed from
    `per_agent.tire` every tick and disappears when there is none.
    """
    from src.pitwall.agents_view.builder import AgentsViewBuilder

    builder = AgentsViewBuilder()
    for lap in range(20, 23):
        builder.build(_payload(seq=lap, lap=lap, latest=_latest(lap)), "Connected")

    # Lap 23 arrives with a lap time and no per-agent block at all.
    stale = builder.build(
        _payload(seq=99, lap=23, latest={"lap_number": 23, "lap_time_s": 81.4}), "Connected"
    )

    pace = stale["charts"]["pace"]
    assert pace["current_lap"] == 23.0
    assert pace["prediction_lap"] == 22.0, "the prediction stopped a lap back"
    assert [lap for lap, _ in pace["actual"]][-1] == 23.0, "and the actual did not"

    # The tyre chart keeps advancing on the same tick, which is why it gets no
    # tag: its trend is observed data.
    tire = stale["charts"]["tire"]
    assert tire["cliff"] is None, "the tyre chart's stale element removes itself"
    assert [lap for lap, _ in tire["trend"]][-1] == 23.0, "its trend is live"

    # And on a healthy tick the two agree, so the renderer draws no tag.
    live = builder.build(_payload(seq=100, lap=24, latest=_latest(24)), "Connected")
    assert live["charts"]["pace"]["prediction_lap"] == live["charts"]["pace"]["current_lap"]


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

    calm = format_pit({**block, "action": "STAY_OUT"}, active=True)
    pressing = format_pit({**block, "sc_reactive": True}, active=True)

    assert calm[0] == "stop 22.40s → HARD", "and it says STOP, not PIT"
    assert hex_str(calm[1]) == hex_str(TEXT_PRIMARY)
    assert calm[3] == "OK"
    assert hex_str(pressing[1]) == hex_str(WARNING)
    assert pressing[3] == "WATCH"
    assert pressing[0].endswith(" · SC")

    # **And the console reads its own agent's verdict, not a proxy for it.**
    # Keyed on `sc_reactive` alone, a degradation- or undercut-driven PIT_NOW
    # with low safety-car probability rendered a green OK disc while the SAME
    # card's model detail printed `action = PIT_NOW` - the glyph contradicting
    # its own dump. Asserted over the whole enumeration of
    # `PitStrategyOutput.action`, because a check on PIT_NOW alone would go on
    # passing the day a sixth value arrives.
    by_action = {
        action: format_pit({**block, "action": action}, active=True)[3]
        for action in ("PIT_NOW", "REACTIVE_SC", "UNDERCUT", "OVERCUT", "STAY_OUT")
    }
    assert by_action == {
        "PIT_NOW": "WATCH",
        "REACTIVE_SC": "WATCH",
        # A plan with a window, not a deadline. Their own amber is the UCUT line.
        "UNDERCUT": "OK",
        "OVERCUT": "OK",
        "STAY_OUT": "OK",
    }, by_action


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
    `per_agent`, which is exactly the loss a frame-indexed truncate causes.

    The eviction also leaked. On a tick where the arcade clock goes back
    but `strategy.latest` still lags at the old lap, it removed the future
    and `ingest_latest` re-added the lagging lap on the same tick: a store
    holding 28/29/30 rewound to 10 ended up holding **only lap 30**. It
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

    Retrying is not a state either. It was WARNING amber here while the
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
    """Guards against the #962 misread walking back in.

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
    """`max` picked whichever key it met first.

    Two equal scores are not a leader and a loser. The arbitrary winner
    then marked the other `NOT TAKEN` - a claim about a decision nobody
    made - and min-max, with nothing to spread, floored BOTH of the joint
    best to 6%, which says the opposite of what a tie means.
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
        builder.build(_payload(seq=lap, lap=lap), "Connected")
    melbourne = builder.build(_payload(seq=30, lap=23), "Connected")
    assert len(melbourne["history"]["pace"]) >= 9, "the fixture never accumulated anything"

    suzuka = builder.build(_other_race_payload(seq=1, lap=3), "Connected")
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
        builder.build(_payload(seq=lap, lap=lap), "Connected")

    relaunched = builder.build(_payload(seq=1, lap=2), "Connected")
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
        builder.build(_payload(seq=lap, lap=lap), "Connected")

    rewound = _payload(seq=99, lap=15)
    rewound["playback"]["frame_index"] = 10
    view = builder.build(rewound, "Connected")
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
        builder.build(_payload(seq=lap, lap=lap), "Connected")

    fresh = _other_race_payload(seq=1, lap=3)
    fresh["strategy"]["history_tail"] = [
        {"lap_number": 18, "lap_time_s": 92.18, "tyre_life": 4, "compound": "MEDIUM"}
    ]
    view = builder.build(fresh, "Connected")
    lap18 = next((row for row in view["history"]["pace"] if row["lap"] == 18), None)
    assert lap18 is not None and lap18["actual"] == 92.18, (
        f"lap 18 kept the dead race's number: {lap18}"
    )


def test_the_pace_chart_paints_the_own_car_in_its_own_team_colour():
    """The actual line identifies a CAR, so it takes the car's colour off the wire.

    The line was `palette.INFO` unconditionally, on a window whose sibling
    already painted the same car in its team colour: the DATA tower, the track
    ring and the race trace all render NOR papaya, and this chart rendered him
    blue. `driver_colors` has been on every tick since the arcade published it;
    nothing under `src/pitwall` read it.

    Compared against the WIRE value rather than against membership of the
    palette, because every team colour is in the palette and a membership check
    cannot tell papaya from the blue it replaced.
    """
    payload = _payload()
    payload["arcade"]["driver_colors"] = {"NOR": [255, 128, 0], "PIA": [255, 128, 0]}

    view = _host(payload).get_agents_view(-1)

    assert view["charts"]["pace"]["actual_colour"] == "#ff8000", (
        "the actual line must be the main driver's team colour from the wire"
    )
    # The prediction is the MODEL's identity, not the car's, so it must not move
    # with the driver. Asserted here rather than in its own test because the
    # thing worth pinning is that the two did not become one colour.
    assert view["charts"]["pace"]["pred_colour"] != "#ff8000"


def test_the_pace_chart_degrades_when_the_wire_has_no_colour_for_the_car():
    """Three absences, one answer, and it must not be a team's colour.

    A driver missing from the map, an empty map, and a tick before the arcade
    block carries either. The degrade is `palette.INFO`, which is also the boot
    view's value, so the window has ONE no-wire colour rather than two - and it
    sits far enough from every colour on the wire that a reader cannot mistake
    it for a real answer.
    """
    from src.pitwall.agents_view.charts import ACTUAL_COLOUR

    absent = _payload()
    absent["arcade"]["driver_colors"] = {"PIA": [255, 128, 0]}
    empty = _payload()
    empty["arcade"]["driver_colors"] = {}

    for name, payload in (("driver absent", absent), ("empty map", empty), ("no key", _payload())):
        colour = _host(payload).get_agents_view(-1)["charts"]["pace"]["actual_colour"]
        assert colour == ACTUAL_COLOUR, f"{name} must degrade, not raise or invent: {colour}"


def test_a_contingency_reaches_the_window_with_the_orchestrators_own_action_label():
    """`PIT_NOW` on the wire must read `PIT NOW` on the glass, as the badge does.

    Through `classify_action`, never a hand-written table: a second copy of the
    action labels is the twin this file's own module docstring warns about. The
    COLOUR is deliberately not taken from it - a branch that is not happening
    must not wear the live call's identity.
    """
    view = _host(_payload()).get_agents_view(-1)
    rows = view["contingencies"]["rows"]

    assert len(rows) == 2, f"the fixture's two branches must both arrive: {rows}"
    assert rows[0]["switch_to"] == "PIT NOW", (
        f"the wire's enum must reach the glass as the orchestrator's own label: {rows[0]}"
    )
    assert rows[1]["switch_to"] == "STAY OUT"
    assert rows[0]["priority"] == "HIGH"
    # Producer order, not sorted: the orchestrator owns the ordering signal.
    assert [row["priority"] for row in rows] == ["HIGH", "MEDIUM"]
    assert view["contingencies"]["empty"] is None, "rows and an empty sentence cannot coexist"
    assert view["contingencies"]["risks"] == [
        "rejoin into traffic",
        "the cliff arrives before the stop",
    ], "the risks reach the body as plain lines, not as a popup"


def test_a_contingency_reaches_the_window_as_text_and_never_as_markup():
    """LLM free text, rendered as a React text node, so nothing is escaped here.

    Both halves matter and each catches the opposite mistake. Escaping in the
    builder would put the characters `&lt;img` on the glass; a
    `dangerouslySetInnerHTML` in the card would execute the tag. The card is
    checked as SOURCE because no Python test can see the TSX.
    """
    hostile = _latest()
    hostile["contingencies"] = [
        {
            "trigger": _HOSTILE,
            "switch_to": "PIT_NOW",
            "priority": "HIGH",
            "rationale": _HOSTILE,
        }
    ]
    hostile["key_risks"] = [_HOSTILE]

    row = _host(_payload(latest=hostile)).get_agents_view(-1)["contingencies"]["rows"][0]

    assert "<img" in row["trigger"], "the text arrives verbatim, as data"
    assert "&lt;" not in row["trigger"], (
        "escaped here it would render as the characters &lt;img on the glass"
    )
    assert "&lt;" not in row["rationale"]

    card = (
        Path(__file__).resolve().parents[2]
        / "src/pitwall/ui/src/features/agents/ContingenciesCard.tsx"
    ).read_text("utf-8")
    assert "dangerouslySetInnerHTML" not in card, (
        "the card renders text nodes; raw HTML here would execute what the LLM wrote"
    )


def test_two_kinds_of_empty_are_two_different_sentences():
    """No call at all and a call that planned no branches are different facts.

    The no-LLM profile emits `contingencies=[]` by construction, so the second is
    routine rather than an error. Both branches carry the SAME key set, which is
    the house empty-state rule: a renderer must never have to test for a missing
    key.
    """
    no_call = _host(_payload(latest={})).get_agents_view(-1)["contingencies"]
    no_branches = _latest()
    no_branches["contingencies"] = []
    no_branches["key_risks"] = []
    planned_none = _host(_payload(latest=no_branches)).get_agents_view(-1)["contingencies"]

    assert set(no_call) == set(planned_none) == {"rows", "risks", "empty"}
    assert no_call["rows"] == planned_none["rows"] == []
    assert no_call["empty"] and planned_none["empty"], "both branches say something"
    assert no_call["empty"] != planned_none["empty"], (
        f"one sentence for two facts: {no_call['empty']!r}"
    )
    assert planned_none["risks"] == [], "no risks means no block, not an empty heading"


def test_every_rolling_per_lap_store_is_trimmed():
    """The trim iterates a hardcoded tuple, so a new store is silently unbounded.

    Introspected rather than listed, so the NEXT store cannot be forgotten
    either. `_compound_by_lap` is the one named exception and it is exempt for a
    reason the class docstring gives: the PLAN lane draws the whole race, so a
    lap-1 stint must still be describable on lap 57.
    """
    from src.pitwall.agents_view.history import KEEP_LAPS, LapHistory

    history = LapHistory()
    for lap in range(1, KEEP_LAPS + 11):
        history.ingest_latest(
            {
                "lap_number": lap,
                "lap_time_s": 81.0,
                "compound": "MEDIUM",
            }
        )

    rolling = {
        name: store
        for name, store in vars(history).items()
        if isinstance(store, dict) and store and all(isinstance(key, int) for key in store)
    }
    assert len(rolling) >= 3, f"the introspection found no stores to check: {list(rolling)}"
    for name, store in rolling.items():
        if name == "_compound_by_lap":
            assert len(store) > KEEP_LAPS, "the compound map is deliberately unbounded"
            continue
        assert len(store) == KEEP_LAPS, (
            f"{name} is a rolling store and is not being trimmed: {len(store)} laps"
        )


def test_the_routing_roster_is_the_routers_own_conditional_agents():
    """`ROUTING_LANES` must name the same agents as the router's `activate.add(...)` calls.

    `build_cards` gates the PIT and RAG cards on these ids and used to carry its
    own copies of the strings. Parsed with `ast` rather than imported: importing
    the orchestrator pulls the whole LLM stack, which is why the wire-contract
    tests next door use the same technique. A router that learns a third
    conditional agent must not leave a card permanently idle.
    """
    import ast

    source = (
        Path(__file__).resolve().parents[2] / "src/agents/strategy_orchestrator.py"
    ).read_text("utf-8")
    tree = ast.parse(source)
    router = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_decide_agents_to_call"
    )
    added = {
        call.args[0].value
        for call in ast.walk(router)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "add"
        and call.args
        and isinstance(call.args[0], ast.Constant)
    }
    assert added, "the ast walk found no `activate.add(...)`; the router moved"

    from src.pitwall.agents_view.routing import ROUTING_LANES

    assert {agent_id for agent_id, _ in ROUTING_LANES} == added, (
        f"the roster and the router disagree: {ROUTING_LANES} vs {sorted(added)}"
    )
