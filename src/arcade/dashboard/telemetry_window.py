"""Live telemetry window: independent QMainWindow subscribing to the
same TCP stream the strategy dashboard uses.

Two subscribers (strategy + telemetry) share a broadcast so the user
can arrange the windows across monitors / desktops without either one
depending on the other. Mirrors the arcade → dashboard split: arcade
owns the pyglet replay, dashboard owns the multi-agent strategy view,
telemetry owns the speed/throttle/brake/DRS live surface.

Spawned from the same subprocess that hosts MainWindow (so we get a
single Qt event loop) but lives in its own QMainWindow with its own
title + geometry so the OS treats it as a separate application window.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QStatusBar, QVBoxLayout, QWidget

from src.arcade.dashboard.stream_client import TelemetryStreamClient
from src.arcade.dashboard.telemetry_panel import TelemetryPanel

logger = logging.getLogger(__name__)


class TelemetryWindow(QMainWindow):
    """Standalone window showing the ``TelemetryPanel`` over the TCP stream.

    Owns its own ``TelemetryStreamClient``: the arcade's
    ``TelemetryStreamServer`` supports N clients, so the telemetry
    window and the strategy dashboard connect independently. Both can
    survive a restart of the other."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("F1 Live Telemetry")
        # 2×2 chart grid needs room — default size matches a typical
        # dual-monitor secondary window (960×640) while staying resizable.
        self.resize(960, 640)

        self._panel = TelemetryPanel()
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self._panel)
        self.setCentralWidget(host)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Waiting for arcade stream…")

        self._client = TelemetryStreamClient()
        self._client.data_received.connect(self._on_data)
        self._client.connection_status.connect(self._on_conn_status)
        self._client.error_occurred.connect(self._on_error)
        self._client.start()

    def _on_data(self, data: dict[str, Any]) -> None:
        # Panel reads ``arcade.circuit_length_m`` + driver codes + the
        # ``telemetry`` block — hand it the full broadcast dict so it
        # can anchor axes and header on its own.
        self._panel.update_from(data)
        arcade = data.get("arcade") or {}
        lap = arcade.get("lap")
        if lap is not None:
            self.statusBar().showMessage(f"lap {lap} · live", 1500)

    def _on_conn_status(self, status: str) -> None:
        if status == "Connected":
            self.statusBar().showMessage("Stream connected", 2000)
        elif status == "Disconnected":
            self.statusBar().showMessage("Disconnected — retrying…")

    def _on_error(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 4000)
        logger.warning(msg)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._client.isRunning():
            self._client.stop()
            self._client.wait(2000)
        event.accept()
