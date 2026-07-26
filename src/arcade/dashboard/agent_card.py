"""Reusable sub-agent card widget.

One instance per N25/N26/N27/N28/N29/N30 card in the dashboard's right
grid. Formatter functions in ``agent_formatters`` produce the data
tuple the card renders; the card itself is a dumb view that does not
know which agent it represents: caller sets the title + icon at
construction time.

Visual sections:

- Header: colored status glyph (●/◐/●/○) · title · optional dim when
  the conditional agent is idle (N28 pit, N30 rag).
- Headline: one bold line with a colour set by the formatter.
- Body: up to 3 smaller lines (lap-time predictions, probabilities,
  ranges) in secondary / tertiary text colours.
- Optional chart slot: a ``QVBoxLayout`` bottom area that later
  commits (C6) can drop pyqtgraph widgets into.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.arcade.dashboard.agent_formatters import (
    STATUS_ALERT,
    STATUS_IDLE,
    STATUS_OK,
    STATUS_WATCH,
    Line,
)
from src.arcade.dashboard.theme import (
    DANGER,
    SUCCESS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    WARNING,
    hex_str,
)

# --- Status glyph / colour mapping (mirrors CLI §5) --------------------

_GLYPH_FOR: dict[str, tuple[str, tuple[int, int, int]]] = {
    STATUS_OK: ("●", SUCCESS),
    STATUS_WATCH: ("◐", WARNING),
    STATUS_ALERT: ("●", DANGER),
    STATUS_IDLE: ("○", TEXT_TERTIARY),
}


class AgentCard(QFrame):
    """Dumb card that renders one sub-agent's formatter output."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self.setProperty("card", True)
        # ``Expanding`` vertically stretches the card to fill the row
        # regardless of content, leaving huge dead space for Situation /
        # Radio / RAG. ``Preferred`` lets the layout size to the widget's
        # natural height; a chart attached later will grow the card as
        # needed via ``setMinimumHeight`` on the chart itself.
        self.setMinimumHeight(140)
        self.setMaximumHeight(260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(6)

        # --- Header row: glyph + title ---------------------------------
        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        self._glyph = QLabel("○")
        self._glyph.setStyleSheet(f"color: {hex_str(TEXT_TERTIARY)}; font-size: 14px;")
        self._title = QLabel(title)
        self._title.setStyleSheet(
            f"color: {hex_str(TEXT_SECONDARY)}; font-size: 11px; "
            "font-weight: 700; letter-spacing: 1px; text-transform: uppercase;"
        )
        header_row.addWidget(self._glyph)
        header_row.addWidget(self._title)
        header_row.addStretch()
        outer.addLayout(header_row)

        # --- Headline --------------------------------------------------
        # Rich text so formatters can embed compound pills / flag chips
        # via ``theme.compound_pill_html`` / ``theme.flag_chip_html``.
        self._headline = QLabel("--")
        self._headline.setTextFormat(Qt.RichText)
        self._headline.setStyleSheet(
            f"color: {hex_str(TEXT_PRIMARY)}; font-size: 15px; font-weight: 700;"
        )
        self._headline.setWordWrap(True)
        outer.addWidget(self._headline)

        # --- Body (three pre-allocated lines) -------------------------
        self._body_lines: list[QLabel] = []
        for _ in range(3):
            lbl = QLabel("")
            # Rich text so the tire compound pill and the radio flag chips
            # embedded by ``agent_formatters`` render as coloured badges.
            lbl.setTextFormat(Qt.RichText)
            lbl.setStyleSheet(f"color: {hex_str(TEXT_SECONDARY)}; font-size: 11px;")
            lbl.setWordWrap(True)
            lbl.setVisible(False)
            outer.addWidget(lbl)
            self._body_lines.append(lbl)

        outer.addStretch()

        # --- Optional chart slot (C6 drops pyqtgraph widgets here) ----
        self._chart_host = QWidget()
        self._chart_layout = QVBoxLayout(self._chart_host)
        self._chart_layout.setContentsMargins(0, 0, 0, 0)
        self._chart_host.setVisible(False)
        outer.addWidget(self._chart_host)

    # --- Public API ---------------------------------------------------

    def render(
        self,
        headline: str,
        headline_color: tuple[int, int, int],
        body: list[Line],
        status: str,
    ) -> None:
        """Push one formatter tuple into the widget. No reflow."""
        glyph, gcolor = _GLYPH_FOR.get(status, ("○", TEXT_TERTIARY))
        self._glyph.setText(glyph)
        self._glyph.setStyleSheet(f"color: {hex_str(gcolor)}; font-size: 14px;")

        self._headline.setText(headline)
        self._headline.setStyleSheet(
            f"color: {hex_str(headline_color)}; font-size: 15px; font-weight: 700;"
        )

        for i, lbl in enumerate(self._body_lines):
            if i < len(body):
                text, colour = body[i]
                lbl.setText(text)
                lbl.setStyleSheet(f"color: {hex_str(colour)}; font-size: 11px;")
                lbl.setVisible(True)
            else:
                lbl.setText("")
                lbl.setVisible(False)

        # Dim the whole card when the agent is idle (N28 / N30 OFF).
        self.setStyleSheet(
            'QFrame[card="true"] { opacity: 0.45; }' if status == STATUS_IDLE else ""
        )

    def attach_chart(self, widget: QWidget) -> None:
        """Drop a pyqtgraph ``PlotWidget`` (or similar) inside the body.

        Bumps the card height caps so the chart + body text both fit:
        the tighter max set in ``__init__`` is for text-only cards
        (Situation, Radio, RAG) where charts would dominate."""
        while self._chart_layout.count():
            item = self._chart_layout.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                w.deleteLater()
        self._chart_layout.addWidget(widget)
        self._chart_host.setVisible(True)
        self.setMinimumHeight(260)
        self.setMaximumHeight(420)
