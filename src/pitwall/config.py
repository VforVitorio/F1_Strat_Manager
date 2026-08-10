"""Window geometry, stream endpoint and asset resolution for PITWALL.

Everything here is a constant or a path. Nothing in this module imports
pywebview, so the tests and every other surface can read it on a machine
with no system webview.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# One source for the endpoint. Redefining the port here would let the arcade
# move it and leave PITWALL connecting to nothing, with no error anywhere:
# the client would just retry forever against a closed port.
from src.arcade.config import STREAM_HOST, STREAM_PORT

__all__ = ["STREAM_HOST", "STREAM_PORT", "WINDOWS", "WindowSpec", "ui_asset", "ui_is_built"]

_UI_DIR: Final[Path] = Path(__file__).resolve().parent / "ui"
_UI_DIST: Final[Path] = _UI_DIR / "dist"

# Room the OS keeps for itself, in the same logical pixels pywebview sizes
# windows in. Measured on the machine this was found on (2560x1440 at 150 %,
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

        The windows overlap. They always did, and on a 1707-wide desktop a
        1500 and a 1320 cannot do anything else; the small `x` stagger exists
        so both title bars stay grabbable.

        Pure, and takes the screen rather than reading it, so this module
        still imports on a machine with no system webview and so the rule can
        be tested without one. It never GROWS a window: a `WindowSpec` is a
        layout decision and a big monitor is not a reason to override it.
        """
        usable_height = max(1, screen_height - _OS_CHROME_ALLOWANCE_PX)
        width = min(self.width, screen_width)
        height = min(self.height, usable_height)
        x = min(index * _WINDOW_STAGGER_PX, max(0, screen_width - width))
        return x, 0, width, height


# DATA is the wider of the two: it holds a full timing tower plus traces.
# AGENTS mirrors the Qt window it replaces (540 + 740 px columns, plus the
# frame), so the layout ports across without being redesigned.
WINDOWS: Final[tuple[WindowSpec, ...]] = (
    WindowSpec("data", "PITWALL · DATA", "data.html", 1500, 950),
    WindowSpec("agents", "PITWALL · AGENTS", "agents.html", 1320, 900),
)


def ui_asset(name: str) -> Path:
    """Resolve a built UI entry point.

    The React app is built to static files and, once packaging lands, ships
    inside the wheel. Until then `npm run build` in `src/pitwall/ui/` is a
    development step, which is why `ui_is_built` exists: a missing bundle is
    a setup mistake with a one-line fix, not a crash to decode.
    """
    return _UI_DIST / name


def ui_is_built() -> bool:
    return all(ui_asset(spec.entry).is_file() for spec in WINDOWS)


def build_hint() -> str:
    return f"PITWALL's UI is not built. Run:  npm install && npm run build   in {_UI_DIR}"
