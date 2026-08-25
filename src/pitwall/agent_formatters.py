"""Pure formatters that turn per-agent output dicts into the tuple an agent
card renders.

**They outlived the widget they were written for.** These built the Qt
``AgentCard``; PITWALL's AGENTS window now renders by calling exactly the same
functions, which is what makes that port 1:1 by construction rather than by
inspection - so sprint 7 MOVED this module out of ``src/arcade/dashboard/``
instead of deleting it with the rest of the package.

One function per sub-agent (Pace N25, Tire N26, Situation N27, Radio N29,
Pit N28, RAG N30). The logic mirrors the CLI's six-row inference panel
(``c:/tmp/arcade_analysis/06_cli_inference_panel.md`` §1.1–§1.6) so the
dashboard reads as a visual extension of the CLI without divergent
thresholds.

Return shape: ``(headline_text, headline_color, body_lines, status)``
where ``body_lines`` is ``list[tuple[str, color]]`` (one line per
pair) and ``status`` is ``"OK" | "WATCH" | "ALERT" | "IDLE"``. The card
widget maps ``status`` to the glyph + colour.

No agent package imports: formatters accept plain dicts already
serialised by ``src/arcade/strategy.py::_dump_dataclass``.

**This module must stay importable without a display stack.** PITWALL's
host calls these same functions so its AGENTS window is the Qt window's
output by construction rather than by inspection, and it runs in a
process that has no Qt. That is why the colours come from
``src.arcade.palette`` and not from ``dashboard.theme``, which imports
PySide6 and, through ``classify_action``, pandas.
"""

from __future__ import annotations

import html
from typing import Any

from src.arcade.palette import (
    DANGER,
    SUCCESS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    WARNING,
    compound_pill_html,
    flag_chip_html,
)
from src.pitwall.reasoning_lines import agent_metrics, agent_reasoning

# Status tokens consumed by AgentCard.set_status
STATUS_OK: str = "OK"
STATUS_WATCH: str = "WATCH"
STATUS_ALERT: str = "ALERT"
STATUS_IDLE: str = "IDLE"

# Type alias for readability only.
Line = tuple[str, tuple[int, int, int]]
Formatted = tuple[str, tuple[int, int, int], list[Line], str]


def _signed(x: float, decimals: int = 3) -> str:
    """Return a +/- signed string so ``+0.123`` vs ``-0.123`` pops visually."""
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.{decimals}f}"


def _escaped(text: str | None, limit: int = 70) -> str:
    """A wire string, cut to the body's width budget and made safe as MARKUP.

    The card body lines still reach the window through
    `dangerouslySetInnerHTML`, because they can carry the compound pill and
    the flag chips - HTML spans the palette builds. That means every OTHER
    field on those lines is markup too, and the radio and RCM messages are
    free text straight off the NLP pipeline: a `<` in a transcript stopped
    being a character and became a tag.

    Escaping was never here. Sprint 8's tooltip change made that visible by
    removing the module's last `html.escape` and then claiming in a comment
    that every free-text field was escaped somewhere - a false sentence
    about a real hole, which is this repo's favourite disguise.

    The tooltips do NOT use this: they return data now, and a React text
    node is not a parser.
    """
    return html.escape(_truncate(text, limit))


def _markup_safe(value: Any) -> str:
    """A producer string, made safe as MARKUP, with nothing else done to it.

    `_escaped` is this plus the body's 70-character width budget, which is
    right for a transcript and wrong for a compound label or a driver code: a
    truncated `MEDIUM` is a different fact, and a truncated `RUS` is a
    different driver.

    **Every field that lands in a headline or a body line needs one of the
    two.** Those still reach the window through `dangerouslySetInnerHTML`
    because they can carry the compound pill and the flag chips, so the WHOLE
    line is markup, and the exit gate found the free-text ones riding
    unescaped: the orchestrator's `undercut_target` is an unconstrained
    `Optional[str]` filled by an LLM, and it reached the PLAN caption verbatim.
    `test_no_agent_string_can_become_markup` feeds a hostile string through
    every formatter rather than trusting this list to stay complete.
    """
    return html.escape(str(value))


def _truncate(text: str | None, limit: int = 70) -> str:
    """Collapse a free-text string to ``limit`` visible characters with an ellipsis suffix.

    Used by the radio/RAG tickers and tooltips to keep transcript snippets
    inside the body QLabel width budget (the cards are ~280-340 px wide and
    the body labels render at 11 px, so 70 chars fits without forcing a
    second wrapped line at typical zoom). Treats ``None`` as the empty
    string at the boundary so callers do not need to guard before calling.

    The ``limit`` is exposed as a parameter so the tooltip path can request
    a longer cap (chunk text in regulation snippets) without spawning a
    near-duplicate helper. The literal ``"..."`` suffix (three ASCII dots)
    is preferred over the unicode ellipsis to keep the project ASCII-only
    in dashboard text and avoid font-fallback artefacts.
    """
    s = (text or "").strip().replace("\n", " ")
    if len(s) <= limit:
        return s
    return s[: max(limit - 3, 0)].rstrip() + "..."


# --- N25 Pace -----------------------------------------------------------


# --- What an idle console says -----------------------------------------------
#
# **"no prediction — stub" was a word about the CODE on a surface a race
# engineer reads.** A stub is a thing a developer knows about; what the reader
# needs to know is that this agent has no reading for this lap, which is a fact
# about the race. Same for "no radio/rcm pipeline output", which names a
# pipeline, and for "triggers on ...", which describes a routing rule rather
# than telling the reader what would wake the console.
#
# The trigger hints are the two conditional agents' own wake conditions, in the
# operational voice the rest of the window uses. They teach the reader what the
# system does, which is why this window dims an idle console rather than
# blanking it.


def format_pace(p: dict[str, Any] | None) -> Formatted:
    """CLI §1.1: pace delta to next predicted lap, with absolute predicted lap time.

    The headline pairs the signed delta vs the previous lap with the
    absolute predicted lap time in parentheses. The delta is the actionable
    signal (is the car about to slow), and the absolute time anchors that
    delta to the current pace baseline so the user can tell a 92 s lap
    apart from a 105 s safety-car lap at a glance without scanning the
    body. Body rows expand into the median delta and the credible-interval
    half-width for users who want the full distribution.
    """
    if not p:
        return (
            "no reading this lap",
            TEXT_TERTIARY,
            [],
            STATUS_IDLE,
        )
    delta_prev = float(p.get("delta_vs_prev", 0.0) or 0.0)
    delta_med = float(p.get("delta_vs_median", 0.0) or 0.0)
    pred = float(p.get("lap_time_pred", 0.0) or 0.0)
    ci_p10 = float(p.get("ci_p10", 0.0) or 0.0)
    ci_p90 = float(p.get("ci_p90", 0.0) or 0.0)
    ci_half = (ci_p90 - ci_p10) / 2 if ci_p90 and ci_p10 else 0.0

    if delta_prev <= 0:
        status = STATUS_OK
    elif delta_prev <= 0.25:
        status = STATUS_WATCH
    else:
        status = STATUS_ALERT

    headline = f"Δnext {_signed(delta_prev, 3)}s ({pred:.2f}s)"
    body: list[Line] = [
        (f"pred {pred:.2f}s", TEXT_SECONDARY),
        (f"vs median {_signed(delta_med, 2)}s", TEXT_SECONDARY),
        (f"±{ci_half:.2f}s (CI)", TEXT_TERTIARY),
    ]
    return headline, TEXT_PRIMARY, body, status


# --- N26 Tire -----------------------------------------------------------


_TIRE_CLIFF_MAX_SANE: float = 100.0  # laps — anything above this is early-stint TCN noise


def format_tire(t: dict[str, Any] | None) -> Formatted:
    """CLI §1.2: cliff p50, range p10-p90, deg rate, warning_level, and stint length.

    The headline pairs the cliff projection (median laps remaining before
    the compound falls off) with the laps already run on the current set,
    formatted as ``L{n}``. This stint-length anchor is preserved on both
    the normal and the stabilising branches because how deep the driver
    already is into the stint is always strategically meaningful, even
    when the cliff prediction itself is unreliable.

    Early-stint outputs (lap 1-3) can produce absurd cliff projections
    (tens of thousands of laps) because the TCN's MC Dropout samples
    lack enough history to converge. We clamp the display to a plausible
    range: values above ``_TIRE_CLIFF_MAX_SANE`` collapse to a "stabilising"
    message and drop the range line rather than render useless numbers.
    """
    if not t:
        return (
            "no reading this lap",
            TEXT_TERTIARY,
            [],
            STATUS_IDLE,
        )
    p10 = float(t.get("laps_to_cliff_p10", 0.0) or 0.0)
    p50 = float(t.get("laps_to_cliff_p50", 0.0) or 0.0)
    p90 = float(t.get("laps_to_cliff_p90", 0.0) or 0.0)
    deg = float(t.get("deg_rate", 0.0) or 0.0)
    life = float(t.get("current_tyre_life", 0.0) or 0.0)
    warning = str(t.get("warning_level") or "OK").upper()
    compound = str(t.get("compound") or "--")

    status_map = {"PIT_SOON": STATUS_ALERT, "MONITOR": STATUS_WATCH, "OK": STATUS_OK}
    status = status_map.get(warning, STATUS_OK)

    # Degradation rate may arrive as 0 while the agent is still warming up;
    # render an em-dash so the user sees "no reading yet" not "0 s/lap".
    deg_text = f"{deg:.3f}s/lap" if deg > 0.0 else "— s/lap"
    compound_label = compound if compound != "0" else "—"
    life_suffix = f" · L{int(life)}"

    cliff_unreliable = p50 > _TIRE_CLIFF_MAX_SANE or p50 <= 0
    if cliff_unreliable:
        headline = f"cliff stabilising…{life_suffix}"
        pill = compound_pill_html(compound_label)
        body: list[Line] = [
            (f"deg {deg_text} · {pill}", TEXT_SECONDARY),
            (_markup_safe(warning), _status_colour(status)),
        ]
        return headline, TEXT_TERTIARY, body, STATUS_WATCH if status == STATUS_OK else status

    headline = f"Cliff ~{int(p50)} laps{life_suffix}"
    pill = compound_pill_html(compound_label)
    body = [
        (f"range {int(p10)}–{int(p90)} laps", TEXT_SECONDARY),
        (f"deg {deg_text} · {pill}", TEXT_SECONDARY),
        (_markup_safe(warning), _status_colour(status)),
    ]
    return headline, TEXT_PRIMARY, body, status


# --- N27 Situation ------------------------------------------------------


def format_situation(s: dict[str, Any] | None) -> Formatted:
    """CLI §1.3: threat level headline, with overtake / SC probabilities and gap-plus-pace context.

    The headline carries the categorical threat level and is colour-coded
    by the same status mapping used for the other agent cards. Body rows
    expand into the underlying numerics: the calibrated overtake
    probability, the 3-lap safety-car probability (highlighted in WARNING
    when above 15% so the user notices an imminent SC risk), and a
    composite line that pairs the gap to the car ahead with the 3-lap
    rolling pace delta. The pace delta uses the project's signed-number
    convention so a faster driver reads as a negative value, matching the
    sign convention in the situation agent's own dataclass.
    """
    if not s:
        return (
            "no reading this lap",
            TEXT_TERTIARY,
            [],
            STATUS_IDLE,
        )
    # NOT `or 0.0`: N27 reports None when the car ahead is farther than the overtake
    # model's trained range, and rendering that as "overtake 0%" tells the wall the model
    # says NO CHANCE when it actually says nothing. The two readings would prompt
    # opposite calls.
    _ot_raw = s.get("overtake_prob")
    ot = None if _ot_raw is None else float(_ot_raw)
    sc = float(s.get("sc_prob_3lap", 0.0) or 0.0)
    gap = float(s.get("gap_ahead_s", 0.0) or 0.0)
    pace_delta = float(s.get("pace_delta_s", 0.0) or 0.0)
    threat = str(s.get("threat_level") or "LOW").upper()

    status_map = {"HIGH": STATUS_ALERT, "MEDIUM": STATUS_WATCH, "LOW": STATUS_OK}
    status = status_map.get(threat, STATUS_OK)
    headline_color = _status_colour(status)

    sc_color = WARNING if sc > 0.15 else TEXT_SECONDARY
    body: list[Line] = [
        (
            "overtake — (out of model range)" if ot is None else f"overtake {ot * 100:.0f}%",
            TEXT_TERTIARY if ot is None else TEXT_SECONDARY,
        ),
        (f"safety car {sc * 100:.0f}%", sc_color),
        (f"gap {gap:.1f}s · Δpace {_signed(pace_delta, 2)}s/lap", TEXT_TERTIARY),
    ]
    return f"Threat {_markup_safe(threat)}", headline_color, body, status


# --- N29 Radio ----------------------------------------------------------


def _rcm_label(event: dict[str, Any]) -> str:
    """Best human-readable tag for a Race Control Message.

    Prefers the broadcast ``flag`` token (``YELLOW``, ``SC``, ``RED``…)
    because that is what the strategist sees on the official feed; falls
    back to the structured ``event_type`` for non-flag entries (penalties,
    investigations) and finally to the literal ``RCM`` so the line never
    renders an empty bracket. Pulled into its own helper so both the body
    ticker and the tooltip render the same label for the same event.
    """
    return str(event.get("flag") or event.get("event_type") or "RCM")


def _radio_driver(event: dict[str, Any]) -> str:
    """Driver three-letter code carried by a radio entry, with a safe fallback.

    The ``RadioOutput.alerts`` list is built by ``_build_alerts`` in the
    radio agent and always carries ``driver``; the raw ``radio_events``
    list is the upstream NLP pipeline output and may not carry it on
    every entry depending on serialisation. ``UNKNOWN`` matches the same
    fallback string used in the agent itself, so the dashboard never
    invents a driver code that does not exist.
    """
    return str(event.get("driver") or "UNKNOWN")


def _radio_intent(event: dict[str, Any]) -> str:
    """Intent label produced by the N21 SetFit classifier for a radio entry.

    Defaults to ``INFO`` when the analysis sub-dict is absent so the body
    line still renders an intent column. Trusting the dict shape produced
    by ``run_pipeline`` (``analysis.intent``); only the boundary against
    missing keys is guarded, per the project's no-defensive-checks rule.
    """
    return str((event.get("analysis") or {}).get("intent") or "INFO")


def _correction_row(entry: dict[str, Any]) -> dict[str, str]:
    """One NLP mismatch, as a tooltip row.

    N29 asks the LLM to compare each message's text against the intent the
    classifier assigned it and to say so when they contradict. The entry names
    the label it would have given instead, the verbatim span that contradicts the
    model, and a one-line reason.

    It is a claim ABOUT a label, never a replacement for one: `alerts` stays
    deterministic and the orchestrator weighs the two itself, so this reads as
    "treat that PROBLEM as weaker" rather than as a relabelling. Every value is
    LLM-filled free text and is stringified here for that reason.
    """
    original = str(entry.get("original_intent") or "?")
    suggested = str(entry.get("suggested_intent") or "?")
    reason = str(entry.get("reason") or "").strip()
    span = str(entry.get("span") or "").strip()
    text = f'{reason} Quoted: "{span}"' if span else reason
    return {
        "lead": f"{str(entry.get('driver') or '?')} {original} reads as {suggested}",
        "text": text,
    }


def radio_tooltip(r: dict[str, Any] | None) -> dict[str, Any] | None:
    """Every radio and RCM of the lap, as DATA the window renders itself.

    The tooltip exists because the body ticker shows only the most recent
    radio and the most recent RCM, and the strategist sometimes needs the
    whole lap - several PROBLEM radios in a row, a chain of yellows
    clearing - which the body has no vertical budget for.

    **It used to return Qt rich text** (`<b>`, `<br>`, `&nbsp;`, and
    nothing else, because that is the subset `QToolTip` parses) and the
    React side rendered it through `dangerouslySetInnerHTML`. Qt was
    retired in sprint 7; the dialect outlived the toolkit that required
    it. Under the hybrid this decides WHAT is said and the TSX decides how
    it looks, so there is no markup here and no escaping either - a React
    text node is not a parser.

    **The 70-character cap is gone, and that is the point of the change.**
    It was the BODY's width budget - `_truncate`'s docstring says so, in
    terms of a 280-340 px QLabel at 11 px - applied to a popup that is not
    a card and is clipped by nothing. The tooltip truncated each message
    to exactly what the card already showed, so its only added value was
    more messages, never more of a message. The body ticker keeps the cap;
    it really is that narrow.

    `None`, not `""`, for "no tooltip". The empty string was Qt's own
    convention for suppressing a popup, and a falsy value that is also a
    legitimate rendering is the sentinel shape this repo keeps paying for.
    """
    if r is None:
        return None
    radio_events = r.get("radio_events") or []
    rcm_events = r.get("rcm_events") or []
    corrections = r.get("corrections") or []
    if not radio_events and not rcm_events and not corrections:
        return None

    sections: list[dict[str, Any]] = []
    if rcm_events:
        sections.append(
            {
                "title": "RCM",
                "rows": [
                    {
                        "lead": f"L{ev.get('lap', '?')} {_rcm_label(ev)}",
                        "text": str(ev.get("message") or ""),
                    }
                    for ev in rcm_events
                ],
            }
        )
    if radio_events:
        sections.append(
            {
                "title": "Radio",
                "rows": [
                    {
                        "lead": f"{_radio_driver(ev)} {_radio_intent(ev)}",
                        "text": str(ev.get("message") or ""),
                    }
                    for ev in radio_events
                ],
            }
        )
    # Last, because it qualifies what the two sections above say rather than
    # adding events of its own: a reader needs the message before the doubt.
    if corrections:
        sections.append(
            {
                "title": "NLP corrections",
                "rows": [
                    _correction_row(entry) for entry in corrections if isinstance(entry, dict)
                ],
            }
        )
    return {"sections": sections, "footer": None}


def model_detail_sections(key: str, block: dict[str, Any] | None) -> list[dict[str, Any]]:
    """What the reasoning tab showed for one agent, as tooltip sections.

    Two of them, and BOTH halves matter. The card says `Δnext -0.204s (81.00s)`
    in operational language; the tab said that in `key = value` form AND
    carried the agent's own sentences above it. A move that took only the
    numbers would have dropped every agent's explanation of WHY off the window
    entirely, on the runs where an agent produces one - which is every run with
    an LLM.

    Composed from `reasoning_lines`' own halves rather than re-derived here, so
    there is one definition of what an agent's body is (`agent_body` is the
    same pair joined for the tab).

    The metric rows split on their first `=` into lead and value. The padding
    that aligns them is for a monospace block; a tooltip lays the pair out
    itself, so it goes.
    """
    reasoning = agent_reasoning(block)
    metrics = agent_metrics(key, block)

    sections: list[dict[str, Any]] = []
    if reasoning:
        sections.append({"title": "Reasoning", "rows": [{"lead": "", "text": reasoning}]})
    if metrics:
        rows = []
        for line in metrics:
            lead, separator, value = line.partition("=")
            if separator:
                rows.append({"lead": lead.strip(), "text": value.strip()})
            else:
                rows.append({"lead": "", "text": line.strip()})
        sections.append({"title": "Model detail", "rows": rows})
    return sections


def with_model_detail(
    tooltip: dict[str, Any] | None, key: str, block: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Append an agent's model detail to whatever tooltip it already had.

    Appended rather than replacing, because two of the six already carry
    content the tab never did - RADIO's whole lap of messages, RAG's question
    and chunks - and that is the drill-down tier this joins rather than
    supersedes.
    """
    sections = model_detail_sections(key, block)
    if not sections:
        return tooltip
    if tooltip is None:
        return {"sections": sections, "footer": None}
    return {"sections": [*tooltip["sections"], *sections], "footer": tooltip.get("footer")}


# The card shows at most this many radio lines. Two, because the console spans
# two grid columns in the decision-band layout and one line left the row half
# empty; more than two and the RCM line below it falls out of the card.
_RADIO_LINES = 2


def _radios_worth_the_room(
    radio_events: list[dict[str, Any]], alerts: list[Any]
) -> list[dict[str, Any]]:
    """The lap's radios, most important first, capped at what the card holds.

    **Severity before recency, and the severity is not decided here.** The card
    showed `radio_events[-1]` - the newest, whatever it said - so a rival's
    routine "box this lap" could evict our own driver reporting a problem, on a
    surface whose whole purpose is noticing the problem.

    Which intents are severe is a question this module must not answer. The
    radio agent already did: `_build_alerts` filters by
    `RadioAgentCFG.alert_intents` and hands the result over in `alerts`, so the
    ranking reads the agent's own verdict FOR THIS LAP rather than carrying a
    copy of the rule. That matters here specifically - this repo already has
    three separately maintained severity maps (`_ALERT_SEVERITY`,
    `_FLAG_BG_BY_INTENT`, and the agent's own), one of which was found missing a
    key the others had, and this module cannot import the agent's config at all
    because doing so loads three models.

    With no alerts the order is recency, which is what it always was.
    """
    flagged = {
        str(alert.get("intent") or alert.get("event_type") or "").upper()
        for alert in alerts
        if isinstance(alert, dict)
    }
    ranked = sorted(
        enumerate(radio_events),
        key=lambda pair: (_radio_intent(pair[1]).upper() in flagged, pair[0]),
        reverse=True,
    )
    return [event for _, event in ranked[:_RADIO_LINES]]


def format_radio(r: dict[str, Any] | None) -> Formatted:
    """CLI §1.4: alert intents headline, plus a per-lap transcript ticker.

    The headline branches the same way as the CLI: chip row when the
    deterministic alert filter fires (PROBLEM / WARNING radios or
    SAFETY_CAR / RED_FLAG / YELLOW RCMs), ``no alerts`` when there is
    radio activity but nothing critical, and ``quiet`` when the lap is
    silent. Body rows replace the previous count-only display with a
    three-tier ticker that surfaces the actual transcripts the strategist
    cares about: a counter line (always present), the most recent RCM
    (present only when ``rcm_events`` is non-empty) and the most recent
    driver radio (present only when ``radio_events`` is non-empty).

    Each transcript line is truncated to 70 characters so the body
    QLabel renders on a single visual row at the current card width.
    The full lap transcript is exposed via ``radio_tooltip_html`` and
    wired by the window onto the card's ``setToolTip`` so a hover gives
    the engineer the unabridged content.
    """
    if r is None:
        return (
            "radio silent",
            TEXT_TERTIARY,
            [],
            STATUS_IDLE,
        )
    radio_events = r.get("radio_events") or []
    rcm_events = r.get("rcm_events") or []
    alerts = r.get("alerts") or []
    n_radios = len(radio_events)
    n_rcms = len(rcm_events)

    if alerts:
        chips: list[str] = []
        for a in alerts[:3]:
            if isinstance(a, dict):
                intent = a.get("intent") or a.get("event_type") or "ALERT"
            else:
                intent = str(a)
            chips.append(flag_chip_html(intent))
        headline = " ".join(chips)
        headline_color = WARNING
        status = STATUS_ALERT
    elif n_radios or n_rcms:
        headline = "no alerts"
        headline_color = TEXT_PRIMARY
        status = STATUS_OK
    else:
        headline = "quiet"
        headline_color = TEXT_PRIMARY
        status = STATUS_OK

    body: list[Line] = [
        (f"{n_radios} radios · {n_rcms} rcm", TEXT_SECONDARY),
    ]
    if rcm_events:
        last_rcm = rcm_events[-1]
        body.append(
            (
                f"RCM L{last_rcm.get('lap', '?')} "
                f"{html.escape(_rcm_label(last_rcm))}: {_escaped(last_rcm.get('message'))}",
                TEXT_SECONDARY,
            )
        )
    for event in _radios_worth_the_room(radio_events, alerts):
        body.append(
            (
                f"{html.escape(_radio_driver(event))} "
                f"{html.escape(_radio_intent(event))}: "
                f'"{_escaped(event.get("message"))}"',
                TEXT_TERTIARY,
            )
        )
    return headline, headline_color, body, status


# --- N28 Pit (conditional) ---------------------------------------------

# The two `PitStrategyOutput.action` values that mean "stop on this lap", as
# opposed to a plan with a window. Read from the agent's own verdict rather
# than inferred from a proxy: `sc_reactive` covers safety-car urgency only, and
# keying on it alone painted a degradation-driven PIT_NOW calm green.
_PIT_ACTIONS_NOW: frozenset[str] = frozenset({"PIT_NOW", "REACTIVE_SC"})


def format_pit(p: dict[str, Any] | None, active: bool) -> Formatted:
    """CLI §1.5: active shows pit p50 → compound; idle shows trigger hint.

    When the upstream ``PitDecision`` flags ``sc_reactive=True`` the
    headline is suffixed with ``" · SC"`` to disclose to the engineer
    that the recommendation is driven by Safety Car pressure (N27
    probability) rather than tyre cliff or compound logic. This enables
    at-a-glance distinction between proactive cliff-driven stops and
    reactive SC-window opportunism, which carry different risk profiles
    (an SC stop saves around ten seconds but only pays off if the SC
    actually deploys within the window). The headline colour stays
    ``WARNING`` in both active sub-cases: the suffix alone communicates
    SC reactivity, preserving non-SC active rendering exactly as before.
    """
    if not active or not p:
        return (
            "wakes on tyre cliff, compound change, or a problem radio",
            TEXT_TERTIARY,
            [],
            STATUS_IDLE,
        )
    p05 = float(p.get("stop_duration_p05", 0.0) or 0.0)
    p50 = float(p.get("stop_duration_p50", 0.0) or 0.0)
    p95 = float(p.get("stop_duration_p95", 0.0) or 0.0)
    compound = str(p.get("compound_recommendation") or "--")
    up = p.get("undercut_prob")
    target = p.get("undercut_target")
    sc_reactive = bool(p.get("sc_reactive", False))

    # "stop", not "pit". `pit_duration_s` is the physical stop and `pit_delta`
    # is the whole excursion including the lap-time loss; this is the first, and
    # a headline reading "pit 22.40s" invites the reader to take it for the
    # second. The two are a pair this repo's notes flag by name.
    headline = f"stop {p50:.2f}s → {_markup_safe(compound)}" + (" · SC" if sc_reactive else "")
    lines: list[Line] = [(f"range {p05:.2f}–{p95:.2f}s", TEXT_SECONDARY)]
    if up is not None and target:
        lines.append((f"UCUT {float(up) * 100:.0f}% → {_markup_safe(target)}", WARNING))
    else:
        lines.append(("no undercut target", TEXT_TERTIARY))

    # **Being awake is not being worried - but a stop THIS LAP is.**
    #
    # The console wore WARNING amber and WATCH whenever N28 was routed, which is
    # every lap with a pit question, so the state that means "look at this" was
    # the state that means "this agent ran". Removing that took the urgent lap
    # with it: keyed on `sc_reactive` alone, a degradation- or undercut-driven
    # `PIT_NOW` with low safety-car probability rendered a green OK disc while
    # the SAME card's model detail printed `action = PIT_NOW` and
    # `recommended_lap = <this lap>`. The glyph contradicted its own dump.
    #
    # So the question is the agent's own verdict, not a proxy for it.
    # `PitStrategyOutput.action` is PIT_NOW / STAY_OUT / UNDERCUT / OVERCUT /
    # REACTIVE_SC and `recommended_lap` is this lap whenever the action is not
    # STAY_OUT; the two that mean "stop now" get WATCH. UNDERCUT and OVERCUT
    # are plans with a window rather than a deadline, and they stay calm - the
    # UCUT line carries their own amber.
    action = str(p.get("action") or "").upper()
    pressing = sc_reactive or action in _PIT_ACTIONS_NOW
    if pressing:
        return headline, WARNING, lines, STATUS_WATCH
    return headline, TEXT_PRIMARY, lines, STATUS_OK


# --- N30 RAG (conditional) ---------------------------------------------


def _format_article_refs(articles: list[Any] | None) -> str:
    """Render a compact ``"Art. X.Y, X.Z"`` line for the body ticker.

    The retriever already deduplicates and normalises article identifiers
    (``"Article 48.3"``, ``"Article 55.1"``); we strip the redundant
    ``"Article "`` / ``"Art. "`` prefix from each so a single ``"Art. "``
    leads the line and the bare identifiers stay legible. Capped at three
    references with a ``", ..."`` suffix when the source list carries
    more, because the body QLabel does not have horizontal budget for a
    fourth identifier on a typical card width and the full list is always
    available in the tooltip.

    Returns the empty string when no usable identifier survives the
    filtering; callers then skip the body line and ``AgentCard`` hides
    it automatically.
    """
    if not articles:
        return ""
    cleaned: list[str] = []
    for raw in articles:
        s = str(raw or "").strip()
        if not s:
            continue
        low = s.lower()
        for prefix in ("article ", "art. ", "art "):
            if low.startswith(prefix):
                s = s[len(prefix) :].strip()
                break
        if s:
            cleaned.append(s)
    if not cleaned:
        return ""
    head = cleaned[:3]
    tail = ", ..." if len(cleaned) > 3 else ""
    return "Art. " + ", ".join(_markup_safe(item) for item in head) + tail


def rag_tooltip(r: dict[str, Any] | None) -> dict[str, Any] | None:
    """Every regulation chunk behind the lap's answer, as DATA.

    The body shows a 70-character snippet of the LLM answer and a compact
    article-refs line; the strategist sometimes needs the verbatim passages
    the answer is grounded on, especially around a contested restart or a
    procedural article. The question leads, so the engineer can see what
    the orchestrator actually asked.

    Four chunks, then a `+N more` footer, which is a bound on how much a
    popup should carry rather than a layout constant. The per-chunk 280
    character cap is gone with the markup: clamping is the renderer's job
    now, and the whole reason to open this is to read the passage.
    """
    if r is None:
        return None
    chunks = r.get("chunks") or []
    answer = str(r.get("answer") or "").strip()
    # **The answer alone is enough to open a tooltip.** It used to require
    # chunks, so a lap whose retriever returned none had no popup at all - and
    # the answer is the one synthesised sentence this console exists to deliver.
    if not chunks and not answer:
        return None

    sections: list[dict[str, Any]] = []
    question = (r.get("question") or "").strip()
    if question:
        sections.append({"title": "Question", "rows": [{"lead": "", "text": question}]})
    # **The answer, in full.** The card renders it through `_escaped`'s 70
    # character budget, so anything longer was truncated there and absent here -
    # the model's actual answer reachable NOWHERE, while this function's own
    # docstring said "the full answer text ... lives in the tooltip". Every
    # other console's full text was wired into its popup in #1019; this is the
    # one that was left with a ceiling on its primary content.
    if answer:
        sections.append({"title": "Answer", "rows": [{"lead": "", "text": answer}]})

    head = chunks[:4]
    for chunk in head:
        article = str(chunk.get("article") or "").strip()
        doc_type = str(chunk.get("doc_type") or "").strip()
        year = chunk.get("year")
        bits = [bit for bit in (doc_type, None if year is None else str(year)) if bit]
        if article:
            bits.append(f"— {article}")
        title = " ".join(bits) if bits else "Chunk"
        sections.append(
            {"title": title, "rows": [{"lead": "", "text": str(chunk.get("text") or "")}]}
        )

    extra = len(chunks) - len(head)
    return {"sections": sections, "footer": f"+{extra} more" if extra > 0 else None}


def format_rag(rag: dict[str, Any] | str | None, active: bool) -> Formatted:
    """CLI §1.6: answer snippet plus article references for the active branch.

    The active branch surfaces a 70-character snippet of the LLM answer
    on body line 1 and the first three deduplicated article references
    (``"Art. 48.3, 55.1"``) on body line 2. The 70-char cap mirrors the
    radio ticker so the two cards read as a balanced pair; the full answer
    text and every retrieved chunk live in the tooltip (``rag_tooltip``).

    **That last sentence was false until the sprint-10 exit gate.** The
    tooltip carried the question and the chunks and never the answer, so an
    answer longer than 70 characters existed nowhere the reader could get to
    it - and the function that says otherwise is this one. It also named
    ``rag_tooltip_html``, which has not existed since sprint 8.

    The parameter is typed permissively (``dict | str | None``) because
    the upstream wire historically carried only the answer string; when
    a bare string is received it is wrapped as ``{"answer": rag}`` so
    legacy producers do not break the card. The structured form
    (``question`` / ``answer`` / ``articles`` / ``chunks``) is what
    populates the article-refs line and the tooltip.
    """
    if not active:
        return (
            "wakes on compound change, SC risk above 30%, or an FIA warning",
            TEXT_TERTIARY,
            [],
            STATUS_IDLE,
        )
    if isinstance(rag, str):
        rag = {"answer": rag}
    rag = rag or {}
    text = (rag.get("answer") or "").strip()
    if not text:
        return (
            "regulation loaded",
            TEXT_PRIMARY,
            [("(empty context)", TEXT_TERTIARY)],
            STATUS_OK,
        )
    body: list[Line] = [(_escaped(text), TEXT_SECONDARY)]
    refs = _format_article_refs(rag.get("articles"))
    if refs:
        body.append((refs, TEXT_TERTIARY))
    return "regulation loaded", TEXT_PRIMARY, body, STATUS_OK


# --- Shared helpers -----------------------------------------------------


def _status_colour(status: str) -> tuple[int, int, int]:
    return {
        STATUS_OK: SUCCESS,
        STATUS_WATCH: WARNING,
        STATUS_ALERT: DANGER,
        STATUS_IDLE: TEXT_TERTIARY,
    }.get(status, TEXT_TERTIARY)
