"""Scenario score bars: four horizontal bars for STAY_OUT / PIT_NOW / UCUT / OCUT.

Consumes ``latest.scenario_scores`` from the LapDecision (flattened in
``src/arcade/strategy.py::_normalize_scores``). Scores are position-equivalent
gains relative to the STAY_OUT baseline, so they are signed and the winner is
frequently negative (STAY_OUT itself is the reference at 0.0). ``update_from``
below shifts by the minimum before scaling, so bar widths stay valid whatever
the sign; the winner reaches full width and is coloured ACCENT, the rest stay
TEXT_SECONDARY.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.arcade.dashboard.theme import (
    ACCENT,
    BORDER_COLOR,
    MONO_FONT_STACK,
    SECONDARY_BG,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    hex_str,
)

_SCENARIO_KEYS: tuple[str, ...] = ("STAY_OUT", "PIT_NOW", "UNDERCUT", "OVERCUT")
_SCENARIO_LABELS: dict[str, str] = {
    "STAY_OUT": "STAY",
    "PIT_NOW": "PIT",
    "UNDERCUT": "UCUT",
    "OVERCUT": "OCUT",
}


class ScenarioBars(QFrame):
    """Four rows: label · horizontal gradient bar · score percent."""

    def __init__(self) -> None:
        super().__init__()
        self.setProperty("card", True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(130)
        self.setMaximumHeight(180)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 10)
        outer.setSpacing(4)

        title = QLabel("SCENARIO SCORES")
        title.setStyleSheet(
            f"color: {hex_str(TEXT_SECONDARY)}; font-size: 11px; "
            "font-weight: 700; letter-spacing: 1px;"
        )
        outer.addWidget(title)

        self._rows: dict[str, tuple[QLabel, QLabel, QLabel]] = {}
        for key in _SCENARIO_KEYS:
            row_host = QWidget()
            row = QHBoxLayout(row_host)
            row.setContentsMargins(0, 2, 0, 2)
            row.setSpacing(8)

            label = QLabel(_SCENARIO_LABELS[key])
            label.setFixedWidth(44)
            label.setStyleSheet(
                f"color: {hex_str(TEXT_SECONDARY)}; font-size: 11px; font-weight: 600;"
            )

            bar = QLabel()
            bar.setFixedHeight(10)
            bar.setMinimumWidth(120)
            bar.setStyleSheet(self._bar_style(0.0, TEXT_SECONDARY))

            pct = QLabel("  0%")
            pct.setFixedWidth(46)
            pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pct.setStyleSheet(
                f"color: {hex_str(TEXT_SECONDARY)}; font-size: 11px; font-family: {MONO_FONT_STACK};"
            )

            row.addWidget(label)
            row.addWidget(bar, 1)
            row.addWidget(pct)
            outer.addWidget(row_host)
            self._rows[key] = (label, bar, pct)

    def update_from(self, scores: dict[str, Any] | None) -> None:
        """Paint the four bars from a ``scenario_scores`` dict.

        MC scores are computed as ``alpha * E[S] + (1 - alpha) * P10[S]``
        (``src/agents/strategy_orchestrator.py::_run_mc_simulation``) and can
        go negative. We shift the four values so the worst one lands at 0
        then scale by the (positive) range, so the winner always reaches
        full width and the loser draws an empty bar, regardless of whether
        all four are negative. The raw score is printed with two decimals
        on the right so the sign and magnitude stay readable.
        """
        raw: dict[str, float] = {}
        if scores:
            for k, v in scores.items():
                try:
                    raw[str(k).upper()] = float(v)
                except (TypeError, ValueError):
                    continue
        winner: str | None = None
        if raw:
            winner = max(raw, key=raw.get)

        present = raw.values()
        if present:
            lo = min(present)
            hi = max(present)
            span = (hi - lo) or 1.0
        else:
            lo, span = 0.0, 1.0

        for key, (label, bar, pct) in self._rows.items():
            v = raw.get(key, lo)
            fill = (v - lo) / span if key in raw else 0.0
            fill = min(1.0, max(0.0, fill))
            is_winner = key == winner and key in raw
            colour = ACCENT if is_winner else TEXT_SECONDARY
            bar.setStyleSheet(self._bar_style(fill, colour))
            pct.setText(f"{v:+.2f}" if key in raw else "  --")
            pct.setStyleSheet(
                f"color: {hex_str(TEXT_PRIMARY if is_winner else TEXT_SECONDARY)}; "
                f"font-size: 11px; font-family: {MONO_FONT_STACK};"
            )
            label.setStyleSheet(
                f"color: {hex_str(ACCENT if is_winner else TEXT_SECONDARY)}; "
                "font-size: 11px; font-weight: 600;"
            )

    @staticmethod
    def _bar_style(fraction: float, colour: tuple[int, int, int]) -> str:
        pct = max(0.0, min(1.0, float(fraction)))
        cut = round(pct, 3)
        empty = hex_str(SECONDARY_BG)
        border = hex_str(BORDER_COLOR)
        return (
            "QLabel { "
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            f"stop:0 {hex_str(colour)}, stop:{cut} {hex_str(colour)}, "
            f"stop:{cut + 0.001 if cut < 1 else 1} {empty}, stop:1 {empty}); "
            f"border: 1px solid {border}; border-radius: 5px; "
            "}"
        )
