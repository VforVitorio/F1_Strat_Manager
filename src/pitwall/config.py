"""Window geometry, stream endpoint and asset resolution for PITWALL.

Constants, paths, and the pure window-placement rules. Nothing in this module
imports pywebview, so the tests and every other surface can read it on a
machine with no system webview.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

# One source for the endpoint. Redefining the port here would let the arcade
# move it and leave PITWALL connecting to nothing, with no error anywhere:
# the client would just retry forever against a closed port.
from src.arcade.config import STREAM_HOST, STREAM_PORT

__all__ = [
    "STREAM_HOST",
    "STREAM_PORT",
    "WINDOWS",
    "WindowSpec",
    "ui_asset",
    "ui_dist",
    "ui_is_built",
    "window_arguments",
    "window_target",
]

_UI_DIR: Final[Path] = Path(__file__).resolve().parent / "ui"
_UI_DIST: Final[Path] = _UI_DIR / "dist"

# Room the OS keeps for itself, in the same logical pixels pywebview sizes
# windows in. Measured on the machine this was found on (2560x1440 at 150%,
# so 1707x960 logical): the taskbar takes 48 and the window's own title bar
# another 37. 90 covers both with a little slack.
#
# It is not decoration. At 1500x950 the DATA window is BORN taller than the
# 912-pixel work area, so its status bar renders underneath the taskbar and
# the bottom row's "Distance (m)" axis label is sliced in half. Both windows
# did it, and no headless screenshot can show it because a headless viewport
# has no desktop to not fit on.
_OS_CHROME_ALLOWANCE_PX: Final[int] = 90

# Horizontal offset between the two windows, so the one underneath still has a
# grabbable title bar. Vertical staggering is what pywebview does by default
# and it is exactly what pushed the second window off the bottom of the work
# area, so there is none here.
_WINDOW_STAGGER_PX: Final[int] = 40


class WindowSpec:
    """One PITWALL window: what it is called, how big it opens, what it loads.

    Two of these exist and they are deliberately not configurable at
    runtime. A real strategy client is dockable because eight people have
    eight jobs; this is one person with one monitor, so the layout is fixed
    and the only thing a user moves is the OS window.
    """

    def __init__(self, key: str, title: str, entry: str, width: int, height: int) -> None:
        self.key = key
        self.title = title
        self.entry = entry
        self.width = width
        self.height = height

    @property
    def url(self) -> str:
        """Absolute path to the built HTML entry point."""
        return str(ui_asset(self.entry))

    def place(self, index: int, screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
        """Where and how big to open, as `(x, y, width, height)`.

        A window bigger than the desktop does not get scrollbars - it gets
        CLIPPED, and the part that goes missing is the bottom, where both
        windows keep their status bar. On the display this was found on, a
        950-pixel-tall window overflowed a 912-pixel work area and the status
        bar was simply never visible.

        **Size and position are one decision, which is why this is one
        method.** Clamping the size alone did not fix it: pywebview CASCADES,
        so the second window started 38 pixels lower than the first and fell
        straight back off the bottom - measured, with DATA's status bar
        visible and AGENTS' still under the taskbar. So the origin is set
        here rather than left to the toolkit, and `y` is always 0.

        The windows overlap. They always did, and on a 1707-wide desktop two
        1500s cannot do anything else; the small `x` stagger exists so both
        title bars stay grabbable.

        **The origin is chosen before the width, and the window narrows to
        what is left.** Clamping `x` against the window's own size instead
        collapsed the stagger to zero the moment both windows hit the screen
        width - on a 1366 laptop the top window landed exactly over the
        bottom one's title bar. That was invisible while the two specs had
        different widths, so the guard only sees it when run on more than one
        screen.

        Pure, and takes the screen rather than reading it, so this module
        still imports on a machine with no system webview and so the rule can
        be tested without one. It never GROWS a window: a `WindowSpec` is a
        layout decision and a big monitor is not a reason to override it.
        """
        usable_height = max(1, screen_height - _OS_CHROME_ALLOWANCE_PX)
        height = min(self.height, usable_height)
        x = index * _WINDOW_STAGGER_PX
        width = min(self.width, max(1, screen_width - x))
        return x, 0, width, height


# Both windows are the same size, so both hand their page the same client area
# and the two harness families have one number to point at rather than two to
# drift apart.
#
# AGENTS was 1320 x 900, and the number had an argument behind
# it: it mirrored the Qt strategy window's 540 + 740 px columns plus the frame.
# The layout elevation deletes that split, so the argument went with it, and
# what was left was a decision band budgeted 180 px wider than the window it
# renders in.
WINDOWS: Final[tuple[WindowSpec, ...]] = (
    WindowSpec("data", "PITWALL · DATA", "data.html", 1500, 950),
    WindowSpec("agents", "PITWALL · AGENTS", "agents.html", 1500, 950),
)


def window_target(spec: WindowSpec, base: str | None) -> str:
    """Where a PITWALL window should point: the loopback server, or the file.

    **The loopback URL whenever there is one, and that is a bug fix (#995).** Given
    a filesystem path, pywebview 6.2.1 serves the window through an internal bottle
    server rooted at `os.path.commonpath` of the window URLs - and with TWO windows
    it gets that wrong, racily. Both halves were reproduced against the installed
    bottle: rooted at `ui/dist`, a request for `data.html` is 200 and one for
    `dist/data.html` is 404 "File does not exist."; rooted at `ui` it is the other
    way round, and both 404s were seen on different runs of the same build.

    `BrowserServer` has none of that - one root, read into memory at startup - and
    it is the transport this window's whole test suite already goes through, so
    using it also makes the OS windows and a browser tab the same surface.

    `base` is None when the bundle could not be served at all; then the file path is
    still better than no window, so the old behaviour is the fallback rather than an
    error.

    Pure, and takes the base rather than starting anything, so the rule is testable
    without a system webview.

    The separator is normalised rather than assumed. `BrowserServer.start` returns
    its address WITH a trailing slash, and a plain `base + entry` was silently
    correct only because of that - the test that pins this rule hands the address
    without one and produced `…:56787data.html`, a URL whose port no longer parses.
    A join that depends on a caller's punctuation is not a join.
    """
    if not base:
        return spec.url
    return f"{base.rstrip('/')}/{spec.entry}"


def window_arguments(
    spec: WindowSpec, index: int, screen: tuple[int, int], base: str | None
) -> dict[str, Any]:
    """Every argument `create_window` needs for one window, as a dict.

    **A seam, and it exists because the thing that fixed #995 was guarded by
    nothing.** `window_target` is pure and well pinned, and no test asserted that
    `__main__` CALLS it: reverting the call site to `spec.url` kept 227 surface
    tests, 176 data-smoke checks and 19 agents-smoke checks green, so the racy 404
    could walk back in on a fully green board. Only opening the OS window would have
    shown it - the same verified-through-the-page-and-not-the-window gap this sprint
    had to confess to once already.

    Therefore, the assembly lives here, where a test can read it without
    importing pywebview, and `main()` unpacks it.

    Returns the toolkit's own keyword names rather than a project-shaped record:
    this is the boundary with pywebview, and translating twice would be a second
    place to get it wrong.
    """
    x, y, width, height = spec.place(index, *screen)
    arguments = {
        "title": spec.title,
        "url": window_target(spec, base),
        "width": width,
        "height": height,
        "x": x,
        "y": y,
    }
    return arguments


def ui_asset(name: str) -> Path:
    """Resolve a built UI entry point.

    The React app is built to static files and, once packaging lands, ships
    inside the wheel. Until then `npm run build` in `src/pitwall/ui/` is a
    development step, which is why `ui_is_built` exists: a missing bundle is
    a setup mistake with a one-line fix, not a crash to decode.
    """
    return _UI_DIST / name


def ui_dist() -> Path:
    """The built bundle's directory, for anything that serves the whole tree."""
    return _UI_DIST


def ui_is_built() -> bool:
    return all(ui_asset(spec.entry).is_file() for spec in WINDOWS)


def build_hint() -> str:
    return f"PITWALL's UI is not built. Run:  npm install && npm run build   in {_UI_DIR}"
