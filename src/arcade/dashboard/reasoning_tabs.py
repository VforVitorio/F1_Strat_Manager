"""Tabbed reasoning panel.

Six tabs (Orchestrator · Pace · Tire · Situation · Radio · Pit) each
hosting a read-only ``QTextEdit`` with the same regex syntax
highlighter (lap refs pink, quantiles magenta, percentages yellow,
time deltas cyan, action keywords bold yellow).

Each sub-agent tab renders TWO sections: the agent's ``reasoning``
string at the top (the LLM-authored explanation the CLI shows), then
an auto-formatted block of the agent's key metrics below it, so even
when the agent did not emit a reasoning (stub path, conditional agent
not fired, older checkpoint) the tab still surfaces the raw numbers.
The RAG agent has no LLM reasoning; its retrieved text lives in the
RAG card already, so it is not tabbed here.

The Orchestrator tab additionally appends the DecisionMemory prompt
block on laps where the call just changed (``plan_changed``), so the
one lap where memory actually drives the decision is visible even
though it leaves no trace in ``reasoning`` itself.
"""

from __future__ import annotations

import re
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextDocument,
)
from PySide6.QtWidgets import (
    QSizePolicy,
    QTabWidget,
    QTextEdit,
)

from src.arcade.dashboard.theme import (
    ACCENT,
    BORDER_COLOR,
    MONO_FONT_STACK,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    hex_str,
)

# --- Highlight colours (mirror CLI §6) -------------------------------
_LAP_COLOR: QColor = QColor(244, 114, 182)
_QUANT_COLOR: QColor = QColor(217, 70, 239)
_PCT_COLOR: QColor = QColor(250, 204, 21)
_DELTA_COLOR: QColor = QColor(34, 211, 238)
_ACTION_COLOR: QColor = QColor(250, 204, 21)


class _ReasoningHighlighter(QSyntaxHighlighter):
    """Five regex rules, compiled once, reused across every tab."""

    def __init__(self, document: QTextDocument) -> None:
        super().__init__(document)
        self._rules: list[tuple[re.Pattern[str], QTextCharFormat]] = []

        def _fmt(colour: QColor, bold: bool = False) -> QTextCharFormat:
            f = QTextCharFormat()
            f.setForeground(colour)
            if bold:
                f.setFontWeight(QFont.Bold)
            return f

        self._rules.append((re.compile(r"\blaps?\s+\d+(?:[-–]\d+)?\b"), _fmt(_LAP_COLOR)))
        self._rules.append((re.compile(r"\b[Pp]\d{2}\b"), _fmt(_QUANT_COLOR)))
        self._rules.append((re.compile(r"\b\d+(?:\.\d+)?%"), _fmt(_PCT_COLOR)))
        self._rules.append((re.compile(r"[+\-]\d+\.\d+\s*s\b"), _fmt(_DELTA_COLOR)))
        self._rules.append(
            (
                re.compile(r"\b(PIT_NOW|STAY_OUT|UNDERCUT|OVERCUT)\b"),
                _fmt(_ACTION_COLOR, bold=True),
            )
        )

    def highlightBlock(self, text: str) -> None:
        for pattern, fmt in self._rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


def _make_editor() -> QTextEdit:
    editor = QTextEdit()
    editor.setReadOnly(True)
    editor.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    editor.setLineWrapMode(QTextEdit.WidgetWidth)
    editor.setStyleSheet(
        "QTextEdit { background-color: transparent; border: none; "
        f"color: {hex_str(TEXT_PRIMARY)}; padding: 8px; "
        f"font-family: {MONO_FONT_STACK}; "
        "font-size: 11px; line-height: 140%; }"
    )
    return editor


# --- Per-agent metric formatters -------------------------------------
# Each takes the agent output dict and returns a list of "key: value" lines
# sorted by importance. Used as the fallback body when reasoning is empty.


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


_LINE_BUILDERS: dict[str, Any] = {
    "pace": _pace_lines,
    "tire": _tire_lines,
    "situation": _situation_lines,
    "radio": _radio_lines,
    "pit": _pit_lines,
}


class ReasoningTabs(QTabWidget):
    """Tabs: Orchestrator + 5 sub-agents. Each sub-agent tab shows
    reasoning on top and metrics below."""

    _TABS: tuple[tuple[str, str], ...] = (
        ("Orchestrator", "orchestrator"),
        ("Pace", "pace"),
        ("Tire", "tire"),
        ("Situation", "situation"),
        ("Radio", "radio"),
        ("Pit", "pit"),
    )

    def __init__(self) -> None:
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(200)
        self.setDocumentMode(True)
        border = hex_str(BORDER_COLOR)
        secondary = hex_str(TEXT_SECONDARY)
        accent = hex_str(ACCENT)
        self.setStyleSheet(
            "QTabWidget::pane { border: 1px solid " + border + "; "
            "border-radius: 6px; top: -1px; } "
            "QTabBar::tab { background: transparent; color: " + secondary + "; "
            "padding: 6px 12px; font-size: 11px; font-weight: 600; "
            "letter-spacing: 0.5px; text-transform: uppercase; } "
            "QTabBar::tab:selected { color: " + accent + "; "
            "border-bottom: 2px solid " + accent + "; }"
        )
        self._editors: dict[str, QTextEdit] = {}
        for label, key in self._TABS:
            editor = _make_editor()
            _ReasoningHighlighter(editor.document())
            self.addTab(editor, label)
            self._editors[key] = editor

    def update_from(self, latest: dict[str, Any] | None) -> None:
        """Push per-agent reasoning + metrics into the tabs."""
        if not latest:
            for ed in self._editors.values():
                ed.setPlainText("")
            return

        # Orchestrator: the reasoning string, plus the DecisionMemory block
        # when this lap's call just changed. The block never shows up in
        # `reasoning` even when it drives the call (see DecisionMemory's
        # module docstring), so it is rendered here directly rather than
        # trusting the LLM to narrate its own continuity. Hidden on every
        # other lap: unconditional display was measured as wallpaper (the
        # action changes on a small minority of laps).
        orchestrator_text = _clean(latest.get("reasoning")) or "— no reasoning —"
        memory_block = latest.get("memory_block")
        if latest.get("plan_changed") and memory_block:
            orchestrator_text += "\n\n--- why this call changed ---\n" + str(memory_block)
        self._editors["orchestrator"].setPlainText(orchestrator_text)

        per = latest.get("per_agent") or {}
        for _, key in self._TABS:
            if key == "orchestrator":
                continue
            agent_out = per.get(key) or {}
            reasoning = _clean(agent_out.get("reasoning"))
            metric_lines = _LINE_BUILDERS[key](agent_out) if agent_out else []
            self._editors[key].setPlainText(_compose(reasoning, metric_lines))


def _compose(reasoning: str, metrics: list[str]) -> str:
    """Assemble the final tab body: reasoning on top, metrics below.

    Either section may be empty. If both are empty the tab shows the
    idle marker so the user knows the agent did not produce output
    this lap (common for conditional N28 / N30 when they are not
    routed)."""
    blocks: list[str] = []
    if reasoning:
        blocks.append(reasoning)
    if metrics:
        blocks.append("\n".join(metrics))
    if not blocks:
        return "— agent idle —"
    return "\n\n".join(blocks)


def _clean(raw: Any) -> str:
    if not raw:
        return ""
    text = " ".join(str(raw).split())
    if len(text) > 600:
        text = text[:597] + "…"
    return text


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
