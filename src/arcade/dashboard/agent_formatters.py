"""Pure formatters that turn per-agent output dicts into the tuple the
``AgentCard`` widget renders.

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
"""

from __future__ import annotations

import html
from typing import Any

from src.arcade.dashboard.theme import (
    DANGER,
    SUCCESS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    WARNING,
    compound_pill_html,
    flag_chip_html,
)

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
            "no prediction — stub",
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
            "no prediction — stub",
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
            (warning, _status_colour(status)),
        ]
        return headline, TEXT_TERTIARY, body, STATUS_WATCH if status == STATUS_OK else status

    headline = f"Cliff ~{int(p50)} laps{life_suffix}"
    pill = compound_pill_html(compound_label)
    body = [
        (f"range {int(p10)}–{int(p90)} laps", TEXT_SECONDARY),
        (f"deg {deg_text} · {pill}", TEXT_SECONDARY),
        (warning, _status_colour(status)),
    ]
    return headline, TEXT_PRIMARY, body, status


# --- N27 Situation ------------------------------------------------------


def format_situation(s: dict[str, Any] | None) -> Formatted:
    """CLI §1.3: threat level headline, with overtake / SC probabilities and gap-plus-pace context.

    The headline carries the categorical threat level and is colour-coded
    by the same status mapping used for the other agent cards. Body rows
    expand into the underlying numerics: the calibrated overtake
    probability, the 3-lap safety-car probability (highlighted in WARNING
    when above 15 % so the user notices an imminent SC risk), and a
    composite line that pairs the gap to the car ahead with the 3-lap
    rolling pace delta. The pace delta uses the project's signed-number
    convention so a faster driver reads as a negative value, matching the
    sign convention in the situation agent's own dataclass.
    """
    if not s:
        return (
            "no prediction — stub",
            TEXT_TERTIARY,
            [],
            STATUS_IDLE,
        )
    ot = float(s.get("overtake_prob", 0.0) or 0.0)
    sc = float(s.get("sc_prob_3lap", 0.0) or 0.0)
    gap = float(s.get("gap_ahead_s", 0.0) or 0.0)
    pace_delta = float(s.get("pace_delta_s", 0.0) or 0.0)
    threat = str(s.get("threat_level") or "LOW").upper()

    status_map = {"HIGH": STATUS_ALERT, "MEDIUM": STATUS_WATCH, "LOW": STATUS_OK}
    status = status_map.get(threat, STATUS_OK)
    headline_color = _status_colour(status)

    sc_color = WARNING if sc > 0.15 else TEXT_SECONDARY
    body: list[Line] = [
        (f"overtake {ot * 100:.0f}%", TEXT_SECONDARY),
        (f"safety car {sc * 100:.0f}%", sc_color),
        (f"gap {gap:.1f}s · Δpace {_signed(pace_delta, 2)}s/lap", TEXT_TERTIARY),
    ]
    return f"Threat {threat}", headline_color, body, status


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


def radio_tooltip_html(r: dict[str, Any] | None) -> str:
    """Build the Qt rich-text tooltip listing every radio and RCM in the lap.

    The tooltip exists because the body ticker only shows the most recent
    radio and the most recent RCM, and the strategist sometimes needs the
    full transcript of the lap (e.g. several PROBLEM radios in a row, or
    a chain of yellow flags clearing) and the body labels do not have the
    vertical budget for that. Returning the empty string is the documented
    Qt convention to suppress the tooltip popup, so callers can always
    invoke ``setToolTip(radio_tooltip_html(r))`` without an explicit
    "clear" branch.

    HTML primitives are limited to ``<b>``, ``<br>`` and ``&nbsp;``
    because Qt's tooltip rich-text subset rejects CSS and most layout
    tags. Free-text fields (driver, intent, message, RCM label) are
    HTML-escaped so a stray ``<`` or ``&`` in a transcript cannot break
    the tooltip's rich-text parser.
    """
    if r is None:
        return ""
    radio_events = r.get("radio_events") or []
    rcm_events = r.get("rcm_events") or []
    if not radio_events and not rcm_events:
        return ""

    sections: list[str] = []
    if rcm_events:
        rcm_lines = ["<b>RCM</b>"]
        for ev in rcm_events:
            lap = ev.get("lap", "?")
            label = html.escape(_rcm_label(ev))
            msg = html.escape(_truncate(ev.get("message"), 70))
            rcm_lines.append(f"L{lap}&nbsp;{label}: {msg}")
        sections.append("<br>".join(rcm_lines))

    if radio_events:
        radio_lines = ["<b>Radio</b>"]
        for ev in radio_events:
            driver = html.escape(_radio_driver(ev))
            intent = html.escape(_radio_intent(ev))
            msg = html.escape(_truncate(ev.get("message"), 70))
            radio_lines.append(f'{driver}&nbsp;{intent}: "{msg}"')
        sections.append("<br>".join(radio_lines))

    return "<br><br>".join(sections)


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
            "no radio/rcm pipeline output",
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
                f"{_rcm_label(last_rcm)}: {_truncate(last_rcm.get('message'), 70)}",
                TEXT_SECONDARY,
            )
        )
    if radio_events:
        last_radio = radio_events[-1]
        body.append(
            (
                f"{_radio_driver(last_radio)} "
                f"{_radio_intent(last_radio)}: "
                f'"{_truncate(last_radio.get("message"), 70)}"',
                TEXT_TERTIARY,
            )
        )
    return headline, headline_color, body, status


# --- N28 Pit (conditional) ---------------------------------------------


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
            "triggers on cliff pressure, compound change, or problem radio",
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

    headline = f"pit {p50:.2f}s → {compound}" + (" · SC" if sc_reactive else "")
    lines: list[Line] = [(f"range {p05:.2f}–{p95:.2f}s", TEXT_SECONDARY)]
    if up is not None and target:
        lines.append((f"UCUT {float(up) * 100:.0f}% → {target}", WARNING))
    else:
        lines.append(("no undercut target", TEXT_TERTIARY))
    return headline, WARNING, lines, STATUS_WATCH


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
    return "Art. " + ", ".join(head) + tail


def rag_tooltip_html(r: dict[str, Any] | None) -> str:
    """Build the Qt rich-text tooltip listing every regulation chunk for the lap.

    The tooltip exists because the body shows only a 70-character snippet
    of the LLM answer plus a compact article-refs line; the strategist
    sometimes needs to read the verbatim regulation passages the answer
    is grounded on (especially during a contested SC restart or a pit
    sequence near a procedural article). Returning the empty string is
    the documented Qt convention to suppress the tooltip popup, so
    callers can always invoke ``setToolTip(rag_tooltip_html(r))`` without
    an explicit "clear" branch.

    The original question is shown above the chunks so the engineer sees
    what the orchestrator actually asked. Up to four chunks are rendered
    in full (truncated at ~280 chars each); any extra retrieval results
    collapse to a ``+N more`` footer to keep the tooltip box tight. HTML
    primitives are limited to ``<b>``, ``<br>`` and ``&nbsp;``; free-text
    fields (question, chunk text, article id) are HTML-escaped so a
    stray ``<`` or ``&`` in the regulation cannot break the parser.
    """
    if r is None:
        return ""
    chunks = r.get("chunks") or []
    question = (r.get("question") or "").strip()
    if not chunks:
        return ""

    parts: list[str] = []
    if question:
        parts.append(f"<b>Question:</b><br>{html.escape(question)}")

    head = chunks[:4]
    extra = len(chunks) - len(head)
    for c in head:
        article = str(c.get("article") or "").strip()
        doc_type = str(c.get("doc_type") or "").strip()
        year = c.get("year")
        header_bits: list[str] = []
        if doc_type:
            header_bits.append(html.escape(doc_type))
        if year is not None:
            header_bits.append(str(year))
        if article:
            header_bits.append("— " + html.escape(article))
        header = "<b>" + " ".join(header_bits) + "</b>" if header_bits else "<b>Chunk</b>"
        body_html = html.escape(_truncate(c.get("text"), 280))
        parts.append(f"{header}<br>{body_html}")
    if extra > 0:
        parts.append(f"+{extra} more")

    return "<br><br>".join(parts)


def format_rag(rag: dict[str, Any] | str | None, active: bool) -> Formatted:
    """CLI §1.6: answer snippet plus article references for the active branch.

    The active branch surfaces a 70-character snippet of the LLM answer
    on body line 1 and the first three deduplicated article references
    (``"Art. 48.3, 55.1"``) on body line 2. The 70-char cap mirrors the
    radio ticker so the two cards read as a balanced pair; the full
    answer text and every retrieved chunk live in the tooltip, which the
    window wires onto the card via ``rag_tooltip_html``.

    The parameter is typed permissively (``dict | str | None``) because
    the upstream wire historically carried only the answer string; when
    a bare string is received it is wrapped as ``{"answer": rag}`` so
    legacy producers do not break the card. The structured form
    (``question`` / ``answer`` / ``articles`` / ``chunks``) is what
    populates the article-refs line and the tooltip.
    """
    if not active:
        return (
            "triggers on compound change, SC >30%, or FIA warning/penalty",
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
    body: list[Line] = [(_truncate(text, 70), TEXT_SECONDARY)]
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
