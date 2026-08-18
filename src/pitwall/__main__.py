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

from src.pitwall.config import (
    STREAM_HOST,
    STREAM_PORT,
    WINDOWS,
    build_hint,
    ui_dist,
    ui_is_built,
    window_target,
)
from src.pitwall.host import PitwallHost
from src.pitwall.stream_client import ArcadeStreamClient
from src.pitwall.webserver import BrowserServer

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not ui_is_built():
        print(build_hint(), file=sys.stderr)
        return 1

    import webview  # imported here so the module stays importable without a webview

    host = PitwallHost(ArcadeStreamClient(STREAM_HOST, STREAM_PORT), window_count=len(WINDOWS))
    host.start()

    # The same two pages, additionally on loopback. The windows are still the
    # product; this exists so the surface can also be opened in a browser -
    # for devtools, for a second screen, or simply because a page is easier to
    # arrange than an OS window. It reads through the same `get_tick`, so
    # there is still ONE TCP client and one sequence however many consumers
    # attach.
    browser = BrowserServer(ui_dist(), host)
    url = browser.start()
    if url:
        logger.info("Serving both windows at %sdata.html and %sagents.html", url, url)
    else:
        logger.warning(
            "The loopback server did not start; the windows fall back to file paths, "
            "which pywebview serves through its own static server (see #995)."
        )

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
        # **The window loads over the host's own loopback server, not off a file
        # path, and that is a bug fix rather than a preference (#995).**
        #
        # Given a filesystem path, pywebview 6.2.1 serves it through an internal
        # bottle server rooted at `os.path.commonpath` of the window URLs - and with
        # TWO windows it gets that wrong, racily. Reproduced both halves against the
        # installed bottle: with the server rooted at `ui/dist`, a request for
        # `data.html` is 200 and one for `dist/data.html` is 404 "File does not
        # exist."; rooted at `ui` it is the other way round. Víctor saw both, on
        # different runs of the same build - `…/6754/dist/data.html` and then
        # `…/52646/data.html`, each a bottle 404 - which is exactly a window
        # computing its URL against one base while the server holds another.
        #
        # Our own server has none of that: one root, read into memory at startup,
        # and it is the transport every check in this window's test suite already
        # goes through. Handing it to `create_window` also means the OS windows and
        # a browser tab are finally the SAME surface rather than two.
        #
        # `js_api` is unaffected: pywebview injects `window.pywebview` per window
        # whatever the URL, and `bridge.ts` falls back to `fetch` when it is absent.
        #
        # The fallback keeps the old behaviour when the bundle cannot be served -
        # a window that renders through pywebview's server sometimes beats no
        # window at all.
        window = webview.create_window(
            spec.title,
            window_target(spec, url),
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
        browser.stop()
        host.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
