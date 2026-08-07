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
