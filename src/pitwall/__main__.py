"""Entry point: two windows, one host, one stream client.

Spawned by the arcade as `python -m src.pitwall`, exactly the way the Qt
dashboard is today. This is the ONLY module that imports pywebview, so a
machine without a system webview still runs the tests and every other
surface.

Run it directly to develop against a running arcade:

    python -m src.pitwall
"""

from __future__ import annotations

import logging
import sys

from src.pitwall.config import STREAM_HOST, STREAM_PORT, WINDOWS, build_hint, ui_is_built
from src.pitwall.host import PitwallHost
from src.pitwall.stream_client import ArcadeStreamClient

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not ui_is_built():
        print(build_hint(), file=sys.stderr)
        return 1

    import webview  # imported here so the module stays importable without a webview

    host = PitwallHost(ArcadeStreamClient(STREAM_HOST, STREAM_PORT), window_count=len(WINDOWS))
    host.start()

    for spec in WINDOWS:
        window = webview.create_window(
            spec.title,
            spec.url,
            js_api=host,
            width=spec.width,
            height=spec.height,
        )
        # The client belongs to the host, so a window closing only decrements
        # a count. Wiring `client.stop` here instead is the regression this
        # sprint was told to prevent: closing the DATA window would silently
        # blind the AGENTS one.
        window.events.closed += host.release_window

    try:
        webview.start()
    finally:
        # `webview.start()` returns when the last window closes, but it also
        # returns on an exception, and a leaked reader thread would hold the
        # socket open against the next run.
        host.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
