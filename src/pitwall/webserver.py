"""Serve the same two windows over http://127.0.0.1, in any browser.

**Additive, and deliberately so.** The 2026-08-07 decision stands: PITWALL is
a desktop surface built with web technology, and the pywebview windows are the
product. That decision killed the FastAPI WebSocket relay (#283) because a
BROWSER cannot open a raw TCP socket - but the PITWALL host already holds that
socket and the payload it produces, so handing the same payload to a browser
costs a small HTTP server rather than a backend.

What it buys, and why it is worth ~100 lines:

- the pages open on a second machine, a phone, or a browser window the user
  can arrange however they like;
- devtools, which a pywebview window does not give you;
- the surface stops being invisible to anything that is not this desktop.

`bridge.ts` is the one module that knows how a tick arrives, which is exactly
why this is cheap: it falls back to `fetch` when `window.pywebview` is absent
and nothing above it changes.

Two deliberate constraints:

- **Bound to the loopback interface.** The broadcast carries a whole race's
  live state and there is no authentication here at all. This is a local
  developer surface, not a deployment.
- **The bundle is read into memory at startup and served from a dict.** No
  request is ever turned into a filesystem path. `serve-dist.mjs` learned this
  twice - a URL can carry `../`, which CodeQL correctly called a path
  injection, and the `stat`-per-entry version that replaced it was a
  filesystem race. Reading the tree once removes the question rather than
  guarding it, and a built bundle is a few hundred kilobytes.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

# Loopback only. See the module docstring: the payload is unauthenticated.
BROWSER_HOST = "127.0.0.1"
# 0 lets the OS pick a free one, which is what a dev surface wants: two arcades
# side by side must not fight, and the chosen port is logged.
BROWSER_PORT = 0

_TEXT_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}


class TickSource(Protocol):
    """What the server needs from the host, and nothing more.

    Deliberately narrower than `PitwallHost`: the browser path must be unable
    to reach `release_window` or `shutdown`, which belong to the OS windows.
    """

    def get_tick(self, since_seq: int = -1) -> dict[str, Any] | None: ...

    def get_agents_view(
        self, since_seq: int = -1, since_connection: str | None = None
    ) -> dict[str, Any] | None: ...

    def get_bulk(self, since_rev: int = -1) -> dict[str, Any] | None: ...

    def get_live_lap(self, since_rev: int = -1) -> dict[str, Any] | None: ...

    # `{label, colour}` since #1024: the colour rides with the word so the two
    # windows cannot colour one socket state differently, which they did.
    def get_connection(self) -> dict[str, str]: ...


# Readers that take "the revision I hold" and answer null when it is current.
_READERS = {
    "/api/tick": "get_tick",
    "/api/agents": "get_agents_view",
    "/api/bulk": "get_bulk",
    "/api/live": "get_live_lap",
}

# Readers with no revision at all. The connection label is a property of the
# socket rather than of the stream, so there is no sequence to be current
# with: when the producer dies the ticks stop and this is the only thing that
# still changes. Its own dict rather than a sentinel in the one above, so the
# two contracts cannot be confused at the call site.
_PLAIN_READERS = {
    "/api/connection": "get_connection",
}


def _read_bundle(root: Path) -> dict[str, tuple[bytes, str]]:
    """The whole built bundle, URL path -> (bytes, content type)."""
    files: dict[str, tuple[bytes, str]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        url = "/" + path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        content_type = _TEXT_TYPES.get(suffix) or mimetypes.guess_type(path.name)[0]
        files[url] = (path.read_bytes(), content_type or "application/octet-stream")
    return files


def _since(query: str) -> int:
    """`?since=N`, or -1 for a caller that has rendered nothing yet."""
    values = parse_qs(query).get("since")
    if not values:
        return -1
    try:
        return int(values[0])
    except ValueError:
        # A browser typing the URL by hand is not a reason to 500. Treating a
        # junk value as "I have seen nothing" returns the newest tick, which
        # is what the caller wanted.
        return -1


def _since_connection(query: str) -> str | None:
    """`?connection=Connected`, or None for a caller that has rendered nothing.

    A string rather than a revision because that is what the label IS - there is
    no sequence for the socket's own state to be current with. `None` is the
    honest "I hold nothing", and it can never equal a real label, so a first
    poll always gets a view.
    """
    values = parse_qs(query).get("connection")
    return values[0] if values else None


def _handler(bundle: dict[str, tuple[bytes, str]], source: TickSource):
    class PitwallHTTPHandler(BaseHTTPRequestHandler):
        # `BaseHTTPRequestHandler` logs every request to stderr, which at
        # 10 Hz across two pages buries everything else the arcade says.
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib name
            logger.debug("pitwall-http: " + fmt, *args)

        def do_GET(self) -> None:  # noqa: N802 - stdlib name
            parsed = urlparse(self.path)
            route = parsed.path

            # `since` means "the revision I hold" on all three: a sequence on
            # the two tick-driven routes, a bulk revision on the third. Same
            # word because it is the same contract - null means nothing new.
            #
            # Names, then one `getattr`, rather than a dict of bound methods:
            # binding all three on every request makes a source that is
            # missing ONE of them fail on the routes it does implement.
            reader = _READERS.get(route)
            if reader is not None:
                # AGENTS carries a SECOND thing the caller holds: the connection
                # label it last rendered. Without it the host would have to
                # remember on the caller's behalf, and with two consumers only
                # one of them would ever learn the producer died (#950).
                extra = (
                    {"since_connection": _since_connection(parsed.query)}
                    if reader == "get_agents_view"
                    else {}
                )
                self._send_json(getattr(source, reader)(_since(parsed.query), **extra))
                return

            plain = _PLAIN_READERS.get(route)
            if plain is not None:
                self._send_json(getattr(source, plain)())
                return

            if route in ("/", ""):
                route = "/data.html"
            entry = bundle.get(route)
            if entry is None:
                self.send_error(404)
                return
            body, content_type = entry
            self._send(200, content_type, body)

        def _send_json(self, payload: dict[str, Any] | None) -> None:
            # `null` is the honest answer to "nothing new since N", and the
            # bridge already treats it that way over the pywebview path.
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self._send(200, "application/json", body, cache=False)

        def _send(self, status: int, content_type: str, body: bytes, cache: bool = True) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if not cache:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # The browser navigated away mid-response. Routine, not an
                # error, and letting it propagate prints a traceback per tick.
                pass

    return PitwallHTTPHandler


class BrowserServer:
    """The bundle plus a tick endpoint, on loopback, in a daemon thread.

    Invariants:

    - it never touches the TCP client: it reads through the same `get_tick`
      the windows call, so there is one client and one sequence however many
      consumers attach;
    - it serves from memory, so a request can never name a file.
    """

    def __init__(self, dist: Path, source: TickSource, port: int = BROWSER_PORT) -> None:
        self._dist = dist
        self._source = source
        self._port = port
        self._server: ThreadingHTTPServer | None = None

    def start(self) -> str | None:
        """Serve, and return the URL to open, or None when it cannot serve.

        None means one of two things and both are handled by the caller falling
        back: there is no bundle to read, or the socket would not bind. Since the
        windows now LOAD through this server (#995) rather than merely being
        mirrored by it, a raise here would take the whole surface down.
        """
        if not self._dist.is_dir():
            return None
        bundle = _read_bundle(self._dist)
        if "/data.html" not in bundle:
            return None
        try:
            self._server = ThreadingHTTPServer(
                (BROWSER_HOST, self._port), _handler(bundle, self._source)
            )
        except OSError:
            # **The failure that can actually happen, and the one the fallback was
            # documented for without covering.** `None` above only fires when the
            # bundle is missing, which `ui_is_built()` has already ruled out three
            # lines earlier in `__main__` - so the advertised file-path fallback was
            # reachable only by deleting the bundle between two checks. A bind
            # failure is real (a Windows port-exclusion range, a security product
            # holding the port) and it used to propagate out of here and kill the
            # host before a single window opened. Now it degrades to what the
            # docstring always promised.
            logger.warning("The PITWALL browser server could not bind %s", BROWSER_HOST)
            return None
        threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="PitwallBrowserServer",
        ).start()
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/"

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
