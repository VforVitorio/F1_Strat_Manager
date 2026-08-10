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

    # The geometry in `WINDOWS` is what the layout wants; the screen decides
    # what it gets. A window taller than the desktop is not scrolled, it is
    # clipped from the bottom - which is exactly where both windows keep their
    # status bar, so the honest "the producer went quiet" signal was invisible
    # on the machine this was found on.
    screen = webview.screens[0]

    for index, spec in enumerate(WINDOWS):
        x, y, width, height = spec.place(index, screen.width, screen.height)
        if (width, height) != (spec.width, spec.height):
            logger.info(
                "%s opens at %dx%d rather than %dx%d - the %dx%d screen is smaller",
                spec.title,
                width,
                height,
                spec.width,
                spec.height,
                screen.width,
                screen.height,
            )
        window = webview.create_window(
            spec.title,
            spec.url,
            js_api=host,
            width=width,
            height=height,
            x=x,
            y=y,
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
