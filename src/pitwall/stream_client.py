"""The single TCP client that reads the arcade's broadcast.

One socket, one slot, one thread. It does no fan-out and no formatting: a
daemon thread reads newline-delimited JSON from `src/arcade/stream.py` and
overwrites the latest payload; everything above it reads that slot.

**Why one client for two windows.** Two would double the parse cost and put
the sequence in two places, and the whole point of the sequence is that
there is exactly one authority on what "the latest tick" is. Measured
during the design gate, two independent sockets against the real server do
NOT drift (200/200 identical sequences), so this is not a correctness fix -
it is one place to hold the state rather than two.

This is deliberately NOT the Qt client renamed. That one is a `QThread`
emitting Qt signals, and Qt is leaving.
"""

from __future__ import annotations

import json
import logging
import socket
import threading

logger = logging.getLogger(__name__)

# How long to wait between connection attempts. The arcade opens its server
# when the replay view is constructed, and PITWALL is spawned right after,
# so the first attempt usually lands; this covers the race and an arcade
# that is restarted while the windows stay open.
RECONNECT_DELAY_S = 1.0
# The producer's own cadence is ~10 Hz, so a read that blocks longer than
# this means the socket is idle rather than slow. Waking up lets the thread
# notice `stop()`.
READ_TIMEOUT_S = 2.0


class ArcadeStreamClient:
    """Reads the arcade broadcast into a single latest-payload slot.

    Invariants:

    - `latest` is either None (nothing received yet) or the most recent
      complete payload. There is no queue: a consumer that falls behind
      wants the newest tick and nothing else, exactly as on the producer
      side.
    - The reader thread is a daemon and owns the socket. Nothing else
      touches it, so `stop()` is the only teardown path.
    - A partial line is never published. The producer writes
      newline-delimited JSON and a TCP read can split anywhere, so the
      buffer holds the tail until its newline arrives.
    """

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._latest: dict | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    def start(self) -> None:
        """Spawn the reader thread. Idempotent."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name="PitwallStreamRead"
        )
        self._thread.start()
        logger.info("Pitwall stream client started against %s:%d", self._host, self._port)

    def stop(self) -> None:
        """Close the socket and let the reader thread finish. Idempotent.

        Called twice on a normal exit - once by the last window closing and
        once by the process teardown - so it reports only the stop that did
        something. A second "stopped" line reads like a second client.
        """
        was_running = self._running
        self._running = False
        self._close_socket()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=READ_TIMEOUT_S + 1.0)
        self._thread = None
        if was_running:
            logger.info("Pitwall stream client stopped")

    @property
    def latest(self) -> dict | None:
        """The most recent payload, or None if nothing has arrived yet."""
        with self._lock:
            return self._latest

    @property
    def connected(self) -> bool:
        return self._socket is not None

    # --- internals ----------------------------------------------------------

    def _read_loop(self) -> None:
        buffer = b""
        while self._running:
            if self._socket is None and not self._connect():
                # `_connect` already waited; looping straight away would spin.
                continue
            try:
                chunk = self._socket.recv(1 << 16)  # type: ignore[union-attr]
            except TimeoutError:
                continue  # an idle producer, not a dead one
            except OSError:
                # The arcade went away, or `stop()` closed the socket under
                # us. Either way the next iteration decides whether to retry.
                self._close_socket()
                buffer = b""
                continue
            if not chunk:
                self._close_socket()
                buffer = b""
                continue
            buffer = self._consume(buffer + chunk)

    def _consume(self, buffer: bytes) -> bytes:
        """Publish every complete line in `buffer` and return the unfinished tail."""
        *lines, tail = buffer.split(b"\n")
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except ValueError as exc:
                # A malformed line costs that tick, not the connection: the
                # next one is 100 ms away and carries the whole state again.
                logger.warning("Discarding an unparseable tick: %s", exc)
                continue
            with self._lock:
                self._latest = payload
        return tail

    def _connect(self) -> bool:
        try:
            sock = socket.create_connection((self._host, self._port), timeout=READ_TIMEOUT_S)
        except OSError:
            # The arcade is not listening yet, or not any more. Waiting here
            # rather than in the caller keeps the retry cadence in one place.
            threading.Event().wait(RECONNECT_DELAY_S)
            return False
        sock.settimeout(READ_TIMEOUT_S)
        self._socket = sock
        logger.info("Pitwall connected to the arcade broadcast")
        return True

    def _close_socket(self) -> None:
        sock, self._socket = self._socket, None
        if sock is None:
            return
        try:
            sock.close()
        except OSError:
            # Already closed or reset by the peer; nothing to undo.
            pass
