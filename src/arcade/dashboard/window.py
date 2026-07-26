"""Main dashboard window.

Subscribes to the arcade telemetry stream and routes updates to three
areas:

- Header bar (top, 44 px): session label, driver, connection chip,
  playback chip, lap counter. Populated from ``arcade`` + ``strategy.start``
  + ``playback`` keys of each broadcast.
- Central ``QSplitter(Qt.Horizontal)``: left panel holds the
  orchestrator card, the scenario-score bars and the six-tab reasoning
  view; right panel holds the 3x2 grid of sub-agent cards (Pace and
  Tire carry an embedded pyqtgraph chart).
- Status bar (bottom): last pipeline error, or the current lap while
  streaming normally.

``_on_data`` is the single router: one incoming broadcast dict fans out
to every widget below, plus the two rolling history dicts
(``_pace_history`` / ``_tire_history``) the charts read from.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.arcade.dashboard.agent_card import AgentCard
from src.arcade.dashboard.agent_formatters import (
    format_pace,
    format_pit,
    format_radio,
    format_rag,
    format_situation,
    format_tire,
    radio_tooltip_html,
    rag_tooltip_html,
)
from src.arcade.dashboard.orchestrator_card import OrchestratorCard
from src.arcade.dashboard.pace_chart import PaceChart
from src.arcade.dashboard.reasoning_tabs import ReasoningTabs
from src.arcade.dashboard.scenario_bars import ScenarioBars
from src.arcade.dashboard.stream_client import TelemetryStreamClient
from src.arcade.dashboard.theme import (
    DANGER,
    SUCCESS,
    TEXT_SECONDARY,
    WARNING,
    hex_str,
)
from src.arcade.dashboard.tire_chart import TireChart

logger = logging.getLogger(__name__)


class HeaderBar(QWidget):
    """Top 40 px strip: session · driver · conn · playback · lap counter."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(44)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 6, 14, 6)
        layout.setSpacing(10)

        self._session = QLabel("--")
        self._session.setStyleSheet("font-size: 14px; font-weight: 600;")
        self._driver = QLabel("--")
        self._driver.setStyleSheet(f"color: {hex_str(TEXT_SECONDARY)}; font-size: 13px;")
        self._conn = QLabel("Disconnected")
        self._conn.setObjectName("chip")
        self._playback = QLabel("-- × · --")
        self._playback.setObjectName("chip")
        self._lap = QLabel("L 0/0")
        self._lap.setObjectName("chip")

        layout.addWidget(self._session)
        layout.addSpacing(6)
        layout.addWidget(self._driver)
        layout.addStretch()
        layout.addWidget(self._conn)
        layout.addWidget(self._playback)
        layout.addWidget(self._lap)

    def update_from(self, data: dict[str, Any]) -> None:
        arcade = data.get("arcade") or {}
        strategy = data.get("strategy") or {}
        playback = data.get("playback") or {}
        start = strategy.get("start") or {}

        gp = start.get("gp") or arcade.get("gp_name") or "--"
        year = start.get("year") or arcade.get("year") or "--"
        self._session.setText(f"{gp} · {year}")
        self._driver.setText(str(start.get("driver") or arcade.get("driver_main") or "--"))

        lap = arcade.get("lap", 0)
        total = arcade.get("total_laps", 0)
        self._lap.setText(f"L {lap}/{total}")

        try:
            speed = float(playback.get("speed", 1.0))
        except (TypeError, ValueError):
            speed = 1.0
        paused = bool(playback.get("paused", False))
        self._playback.setText(f"{speed:.2f}× · {'PAUSED' if paused else 'PLAYING'}")

    def set_connection(self, status: str) -> None:
        self._conn.setText(status)
        color = {
            "Connected": hex_str(SUCCESS),
            "Connecting...": hex_str(WARNING),
            "Disconnected": hex_str(DANGER),
        }.get(status, hex_str(TEXT_SECONDARY))
        self._conn.setStyleSheet(
            f"color: {color}; font-weight: 600; padding: 2px 10px; "
            f"border-radius: 10px; font-size: 11px;"
        )


class MainWindow(QMainWindow):
    """Dashboard shell: header + QSplitter (left panel / right panel) + status bar.

    Owns the ``TelemetryStreamClient`` connection and every widget the
    left and right panels hold. ``_on_data`` is the sole entry point for
    a new broadcast; it updates the header, the left-panel widgets, the
    right-panel agent cards, and the two chart history dicts in that
    order, then sets the status-bar message last so it reflects whatever
    the pipeline reported for this tick.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("F1 Strategy Dashboard")
        self.resize(1280, 720)

        self._header = HeaderBar()

        self._left_host = QWidget()
        self._right_host = QWidget()
        self._left_layout = QVBoxLayout(self._left_host)
        self._right_layout = QVBoxLayout(self._right_host)
        for lay in (self._left_layout, self._right_layout):
            lay.setContentsMargins(10, 10, 10, 10)
            lay.setSpacing(8)

        self._orchestrator_card = OrchestratorCard()
        self._scenario_bars = ScenarioBars()
        self._reasoning_tabs = ReasoningTabs()
        self._left_layout.addWidget(self._orchestrator_card)
        self._left_layout.addWidget(self._scenario_bars)
        self._left_layout.addWidget(self._reasoning_tabs, 1)

        # --- Agent cards grid 3×2 in the right panel --------------------
        self._card_pace = AgentCard("Pace")
        self._card_tire = AgentCard("Tire")
        self._card_situation = AgentCard("Situation")
        self._card_pit = AgentCard("Pit")
        self._card_radio = AgentCard("Radio")
        self._card_rag = AgentCard("RAG")

        # Embed the two pyqtgraph charts in their respective cards. Chart
        # data is accumulated in ``_pace_history`` / ``_tire_history``
        # because ``history_tail`` in the broadcast strips ``per_agent``
        # (wire-size trade-off) — the window owns the time series.
        self._pace_chart = PaceChart()
        self._tire_chart = TireChart()
        self._card_pace.attach_chart(self._pace_chart)
        self._card_tire.attach_chart(self._tire_chart)
        self._pace_history: dict[int, dict[str, Any]] = {}
        self._tire_history: dict[int, dict[str, Any]] = {}
        grid = QGridLayout()
        grid.setSpacing(8)
        grid.addWidget(self._card_pace, 0, 0)
        grid.addWidget(self._card_tire, 0, 1)
        grid.addWidget(self._card_situation, 1, 0)
        grid.addWidget(self._card_pit, 1, 1)
        grid.addWidget(self._card_radio, 2, 0)
        grid.addWidget(self._card_rag, 2, 1)
        self._right_layout.addLayout(grid, 1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._left_host)
        splitter.addWidget(self._right_host)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([540, 740])
        splitter.setHandleWidth(2)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._header)
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

        self.statusBar().showMessage("Waiting for arcade stream…")

        self._client = TelemetryStreamClient()
        self._client.data_received.connect(self._on_data)
        self._client.connection_status.connect(self._on_conn_status)
        self._client.error_occurred.connect(self._on_error)
        self._client.start()

    def _on_data(self, data: dict[str, Any]) -> None:
        """Router for incoming broadcasts: fans out to widgets."""
        self._header.update_from(data)
        strategy = data.get("strategy") or {}
        latest = strategy.get("latest") or {}
        self._orchestrator_card.update_from(latest or None)
        self._scenario_bars.update_from(latest.get("scenario_scores") if latest else None)
        self._reasoning_tabs.update_from(latest or None)
        self._seed_history_from_tail(strategy.get("history_tail") or [])
        self._ingest_latest_history(latest)
        self._update_agent_cards(latest)
        self._pace_chart.update_from(self._pace_history)
        self._tire_chart.update_from(
            self._tire_history_list(),
            current_lap=latest.get("lap_number") if latest else None,
            tire_out=(latest.get("per_agent") or {}).get("tire") if latest else None,
        )
        err = strategy.get("error")
        if err:
            self.statusBar().showMessage(f"pipeline: {err}")
        else:
            lap = (data.get("arcade") or {}).get("lap", "?")
            self.statusBar().showMessage(f"lap {lap} · streaming", 1500)

    def _seed_history_from_tail(self, tail: list[dict[str, Any]]) -> None:
        """Backfill chart dicts with lap_time_s / tyre_life actuals from the
        broadcast history_tail. ``per_agent`` is stripped there (wire-size
        trade-off) so predicted / CI values stay empty for past laps until
        we observe them via ``latest``, an accepted limitation of
        the mid-stream reconnect path."""
        for row in tail:
            lap = row.get("lap_number")
            if not isinstance(lap, int):
                continue
            pace_row = self._pace_history.setdefault(lap, {})
            pace_row.setdefault("actual", row.get("lap_time_s"))
            tire_row = self._tire_history.setdefault(lap, {})
            tire_row.setdefault("tyre_life", row.get("tyre_life"))
            tire_row.setdefault("compound", row.get("compound"))
            tire_row.setdefault("lap_time_s", row.get("lap_time_s"))
        self._trim_history()

    def _ingest_latest_history(self, latest: dict[str, Any]) -> None:
        if not latest:
            return
        lap = latest.get("lap_number")
        if not isinstance(lap, int):
            return
        per = latest.get("per_agent") or {}
        pace = per.get("pace") or {}
        row = self._pace_history.setdefault(lap, {})
        if latest.get("lap_time_s") is not None:
            row["actual"] = latest.get("lap_time_s")
        if pace.get("lap_time_pred") is not None:
            row["pred"] = pace.get("lap_time_pred")
            row["ci_p10"] = pace.get("ci_p10")
            row["ci_p90"] = pace.get("ci_p90")
        trow = self._tire_history.setdefault(lap, {})
        if latest.get("tyre_life") is not None:
            trow["tyre_life"] = latest.get("tyre_life")
        if latest.get("compound"):
            trow["compound"] = latest.get("compound")
        if latest.get("lap_time_s") is not None:
            trow["lap_time_s"] = latest.get("lap_time_s")
        self._trim_history()

    def _tire_history_list(self) -> list[dict[str, Any]]:
        return [{"lap": lap, **row} for lap, row in sorted(self._tire_history.items())]

    def _trim_history(self, keep: int = 40) -> None:
        """Keep only the most recent ``keep`` laps so memory stays bounded
        (the charts only need a rolling window and this prevents an hour
        of replay from growing a 200-entry dict)."""
        for store in (self._pace_history, self._tire_history):
            if len(store) <= keep:
                continue
            to_drop = sorted(store.keys())[: len(store) - keep]
            for lap in to_drop:
                store.pop(lap, None)

    def _update_agent_cards(self, latest: dict[str, Any]) -> None:
        """Push the per-agent block of ``latest`` into the six cards.

        Conditional agents (N28 pit, N30 rag) read the ``active`` list to
        decide whether to render content or the idle placeholder."""
        per = latest.get("per_agent") if latest else None
        if not per:
            for card, fmt in (
                (self._card_pace, format_pace),
                (self._card_tire, format_tire),
                (self._card_situation, format_situation),
                (self._card_radio, format_radio),
            ):
                card.render(*fmt(None))
            self._card_pit.render(*format_pit(None, active=False))
            self._card_rag.render(*format_rag(None, active=False))
            # Clear any tooltip left over from a prior tick — Qt keeps the
            # last rich-text content cached on the widget otherwise.
            self._card_radio.setToolTip("")
            self._card_rag.setToolTip("")
            return

        active = set(per.get("active") or [])
        self._card_pace.render(*format_pace(per.get("pace")))
        self._card_tire.render(*format_tire(per.get("tire")))
        self._card_situation.render(*format_situation(per.get("situation")))
        radio_block = per.get("radio")
        self._card_radio.render(*format_radio(radio_block))
        self._card_radio.setToolTip(radio_tooltip_html(radio_block))
        self._card_pit.render(*format_pit(per.get("pit"), active="N28" in active))
        # ``rag`` is the structured payload; ``regulation_context`` stays as a
        # legacy fallback for producers that have not yet been updated.
        rag_block = per.get("rag") or per.get("regulation_context")
        rag_active = "N30" in active
        self._card_rag.render(*format_rag(rag_block, active=rag_active))
        self._card_rag.setToolTip(rag_tooltip_html(rag_block) if rag_active else "")

    def _on_conn_status(self, status: str) -> None:
        self._header.set_connection(status)
        if status == "Connected":
            self.statusBar().showMessage("Stream connected", 2000)

    def _on_error(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 4000)
        logger.warning(msg)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop the client thread cleanly before the window dies."""
        if self._client.isRunning():
            self._client.stop()
            self._client.wait(2000)
        event.accept()
