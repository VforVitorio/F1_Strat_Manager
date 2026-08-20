"""The reasoning tabs' second formatting layer, with no toolkit attached.

Written for the Qt reasoning tabs and moved here in sprint 7 when that package
was retired: PITWALL renders the same tabs by calling these, so the layer had a
consumer that outlived the widget.

Five per-agent `key = value` metric dumps, plus the helpers that compose a
tab body out of an agent's reasoning and its numbers. Split out of
`reasoning_tabs.py` for the same reason `palette.py` was split out of
`theme.py`: **PITWALL renders these tabs from this code**, and the module
they lived in imports PySide6.

This layer is easy to miss. It is not `agent_formatters`, and it is not a
prettier view of it: it renders largely the same underlying fields as raw
diagnostics, so a port that took only the headline formatters would drop
every reasoning tab's metrics fallback - the one thing on screen when an
agent produced no LLM reasoning at all.
"""

from __future__ import annotations

from typing import Any


def _fnum(value: Any, decimals: int = 2, signed: bool = False) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    fmt = f"{{v:+.{decimals}f}}" if signed else f"{{v:.{decimals}f}}"
    return fmt.format(v=v)


def _pct(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:5.1f}%"
    except (TypeError, ValueError):
        return "—"


def _pace_lines(p: dict[str, Any]) -> list[str]:
    return [
        f"lap_time_pred   = {_fnum(p.get('lap_time_pred'), 3)}s",
        f"delta_vs_prev   = {_fnum(p.get('delta_vs_prev'), 3, signed=True)}s",
        f"delta_vs_median = {_fnum(p.get('delta_vs_median'), 3, signed=True)}s",
        f"ci_p10          = {_fnum(p.get('ci_p10'), 2)}s",
        f"ci_p90          = {_fnum(p.get('ci_p90'), 2)}s",
    ]


def _tire_lines(t: dict[str, Any]) -> list[str]:
    return [
        f"compound          = {t.get('compound', '—')}",
        f"current_tyre_life = {t.get('current_tyre_life', '—')} laps",
        f"deg_rate          = {_fnum(t.get('deg_rate'), 3)}s/lap",
        f"laps_to_cliff_p10 = {_fnum(t.get('laps_to_cliff_p10'), 1)}",
        f"laps_to_cliff_p50 = {_fnum(t.get('laps_to_cliff_p50'), 1)}",
        f"laps_to_cliff_p90 = {_fnum(t.get('laps_to_cliff_p90'), 1)}",
        f"warning_level     = {t.get('warning_level', '—')}",
    ]


def _situation_lines(s: dict[str, Any]) -> list[str]:
    return [
        # The em dash `_pct` renders for a missing value is honest but silent, and this tab
        # is the one a strategist opens to ask WHY. Its arcade sibling `format_situation`
        # says "out of model range" for the same state; without the same words here the two
        # surfaces describe the same lap differently.
        f"overtake_prob = {_pct(s.get('overtake_prob'))}"
        + ("  (beyond the model's trained gap)" if s.get("overtake_prob") is None else ""),
        f"sc_prob_3lap  = {_pct(s.get('sc_prob_3lap'))}",
        f"threat_level  = {s.get('threat_level', '—')}",
        f"gap_ahead_s   = {_fnum(s.get('gap_ahead_s'), 2)}s",
        f"pace_delta_s  = {_fnum(s.get('pace_delta_s'), 3, signed=True)}s",
    ]


def _radio_lines(r: dict[str, Any]) -> list[str]:
    radios = len(r.get("radio_events") or [])
    rcms = len(r.get("rcm_events") or [])
    alerts = r.get("alerts") or []
    lines = [
        f"radio_events = {radios}",
        f"rcm_events   = {rcms}",
        f"alerts       = {len(alerts)}",
    ]
    for i, a in enumerate(alerts[:5]):
        intent = a.get("intent") or a.get("event_type") or "?" if isinstance(a, dict) else str(a)
        lines.append(f"  [{i}] {intent}")
    return lines


def _pit_lines(p: dict[str, Any]) -> list[str]:
    return [
        f"action                  = {p.get('action', '—')}",
        f"recommended_lap         = {p.get('recommended_lap', '—')}",
        f"compound_recommendation = {p.get('compound_recommendation', '—')}",
        f"stop_duration_p05       = {_fnum(p.get('stop_duration_p05'), 2)}s",
        f"stop_duration_p50       = {_fnum(p.get('stop_duration_p50'), 2)}s",
        f"stop_duration_p95       = {_fnum(p.get('stop_duration_p95'), 2)}s",
        f"undercut_prob           = {_pct(p.get('undercut_prob'))}",
        f"undercut_target         = {p.get('undercut_target') or '—'}",
        f"sc_reactive             = {p.get('sc_reactive', False)}",
    ]


LINE_BUILDERS: dict[str, Any] = {
    "pace": _pace_lines,
    "tire": _tire_lines,
    "situation": _situation_lines,
    "radio": _radio_lines,
    "pit": _pit_lines,
}


def agent_reasoning(block: dict[str, Any] | None) -> str:
    """One agent's own words for the lap, cleaned, or empty."""
    return clean((block or {}).get("reasoning"))


def agent_metrics(key: str, block: dict[str, Any] | None) -> list[str]:
    """One agent's `key = value` dump, or empty when it produced nothing."""
    if not block:
        return []
    return LINE_BUILDERS[key](block)


def agent_body(key: str, block: dict[str, Any] | None) -> str:
    """The whole of one agent's reasoning-tab body: its words, then its numbers.

    **The one place that composition happens.** The tabs and the card tooltips
    both show it, and writing it twice is how the copy that got a fix and the
    copy that did not come about - the dominant defect class in this repo.
    """
    return compose(agent_reasoning(block), agent_metrics(key, block))


def compose(reasoning: str, metrics: list[str]) -> str:
    """Assemble the final tab body: reasoning on top, metrics below.

    Either section may be empty. If both are empty the tab shows the idle
    marker so the user knows the agent did not produce output this lap
    (common for conditional N28 / N30 when they are not routed).
    """
    blocks: list[str] = []
    if reasoning:
        blocks.append(reasoning)
    if metrics:
        blocks.append("\n".join(metrics))
    if not blocks:
        return "— agent idle —"
    return "\n\n".join(blocks)


def clean(raw: Any) -> str:
    """Collapse whitespace and cap the length, as the Qt editor does."""
    if not raw:
        return ""
    text = " ".join(str(raw).split())
    if len(text) > 600:
        text = text[:597] + "…"
    return text
