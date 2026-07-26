"""Qt-side TCP client that subscribes to the arcade telemetry stream.

Mirrors ``TelemetryStreamServer`` in ``src/arcade/stream.py``: the arcade
process (pyglet + arcade) broadcasts newline-delimited JSON to all
connected clients; this QThread reads the socket, splits on ``\\n`` and
emits one ``dict`` per message via ``data_received``. The dashboard
never talks to the arcade in the other direction: it is a pure
subscriber.

Port of the ``TelemetryStreamClient`` class from
``f1_replay/f1-race-replay/src/services/stream.py`` (L87-L178), adapted
to read host/port from ``src.arcade.dashboard.theme`` so the two halves
of the pipeline stay on the same port without duplicated constants.
"""

from __future__ import annotations

import json
import logging
import socket

from PySide6.QtCore import QThread, Signal

from src.arcade.dashboard.theme import STREAM_HOST, STREAM_PORT

logger = logging.getLogger(__name__)


class TelemetryStreamClient(QThread):
    """Subscriber thread for the arcade telemetry stream.

    Signals:
        ``data_received(dict)``: one payload per decoded JSON line.
        ``connection_status(str)``: ``"Connecting..."``, ``"Connected"``
        or ``"Disconnected"``; suitable for a status chip.
        ``error_occurred(str)``: human-readable error, routed to the
        status bar.

    Lifecycle:
        - ``start()`` launches the thread and triggers the first connect.
        - ``stop()`` flips ``_running`` false; the main window waits on
          ``self.wait(2000)`` in ``closeEvent`` for a clean teardown.
        - Reconnect loop: on any transport error, sleeps 2 s and retries.
          Keeps the UI alive across arcade restarts without extra UX.
    """

    data_received = Signal(dict)
    connection_status = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, host: str = STREAM_HOST, port: int = STREAM_PORT) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._socket: socket.socket | None = None
        self._connected = False
        self._running = False

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                self._connect()
                self._receive_loop()
            except Exception as exc:
                self.error_occurred.emit(f"Stream error: {exc}")
                self._close_socket()
                self._connected = False
                self.connection_status.emit("Disconnected")
                self.sleep(2)

    def stop(self) -> None:
        self._running = False
        self._connected = False
        self._close_socket()

    def _connect(self) -> None:
        if self._connected:
            return
        self.connection_status.emit("Connecting...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        try:
            sock.connect((self._host, self._port))
        except socket.timeout:
            self.error_occurred.emit(
                f"Connection timeout — is the arcade replay running on {self._host}:{self._port}?"
            )
            raise
        except ConnectionRefusedError:
            self.error_occurred.emit(f"Connection refused on {self._host}:{self._port}")
            raise
        self._socket = sock
        self._connected = True
        self.connection_status.emit("Connected")
        logger.info("Connected to arcade stream at %s:%d", self._host, self._port)

    def _receive_loop(self) -> None:
        """Read chunks, split on ``\\n``, decode each line as JSON and emit.

        Arcade's ``TelemetryStreamServer.broadcast`` appends exactly one
        ``\\n`` per payload, so splitting on that delimiter is sufficient.
        Partial last-line buffering handles TCP segmentation.
        """
        assert self._socket is not None
        buffer = ""
        while self._running and self._connected:
            try:
                chunk = self._socket.recv(8192).decode("utf-8")
            except socket.timeout:
                continue
            except OSError as exc:
                if self._running:
                    self.error_occurred.emit(f"Receive error: {exc}")
                break
            if not chunk:
                self._connected = False
                break
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    self.data_received.emit(json.loads(line))
                except json.JSONDecodeError as exc:
                    self.error_occurred.emit(f"JSON decode error: {exc}")

    def _close_socket(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
