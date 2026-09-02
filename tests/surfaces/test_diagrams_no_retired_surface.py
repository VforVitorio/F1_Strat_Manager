"""No diagram outside the two marked legacy names a retired surface (#1090).

Three `.drawio` files still drew the PySide6 pair PITWALL replaced in sprint 7,
and one of them spawned `src.arcade.telemetry`, a module that does not exist at
all. They survived a full documentation sweep because a diagram is read as a
picture: nothing about a stale box looks stale, and the two files that WERE
retired had been renamed, which made the folder look audited.

**Labels are parsed out of the `value=` attributes, never grepped from the raw
XML.** The diagrams README already records why, and the rule is the repo's own:
a grep counts matches inside `style=` attributes and colour names, so it
over-reports a file whose only hit is a hex code and under-reports a defect
spelled some other way. This is the parse half; a claim about what a diagram
MEANS still has to be checked against the code by hand.

The vocabulary below is deliberately narrow. It catches the retired GUI toolkit
by name, which is the drift that actually happened twice here, and it makes no
attempt to judge whether a diagram is otherwise current.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
_DIAGRAMS = ROOT / "documents" / "dev_docs" / "diagrams"

# Retired with the Qt windows in sprint 7. `Qt` alone is not here: it appears in
# legitimate prose about the port's history, and a rule that fires on it would
# be turned off rather than obeyed.
_RETIRED = (
    "PySide6",
    "QApplication",
    "QMainWindow",
    "QThread",
    "pyqtSignal",
    "src.arcade.telemetry",
)

# The two files kept ON PURPOSE as a record of surfaces that no longer exist.
# Their names carry the marker, so the exemption is visible in the folder
# listing rather than only here.
_LEGACY_SUFFIX = "_legacy.drawio"


def _tab_names(path: Path) -> list[str]:
    """The `<diagram name=...>` page tabs, which survive a compressed save."""
    raw = path.read_text(encoding="utf-8")
    return [html.unescape(n) for n in re.findall(r'<diagram[^>]*name="([^"]*)"', raw)]


def _labels(path: Path) -> list[str]:
    """Every rendered string in one diagram: cell values plus the tab name.

    A box that SAYS it is retired is dropped, because that is the diagram doing
    the right thing. `tcp_broadcast_dataflow` draws the dead Qt telemetry window
    beside the live subscriber precisely so the reader can see what was replaced,
    and its own label opens with the word. Reading that as a defect would push
    the honest picture out to make the check green.
    """
    raw = path.read_text(encoding="utf-8")
    values = [html.unescape(v) for v in re.findall(r'value="([^"]*)"', raw)]
    return [text for text in values + _tab_names(path) if "RETIRED" not in text]


_ALL = sorted(_DIAGRAMS.rglob("*.drawio"))
_CURRENT = [p for p in _ALL if not p.name.endswith(_LEGACY_SUFFIX)]
_LEGACY = [p for p in _ALL if p.name.endswith(_LEGACY_SUFFIX)]


def test_the_folder_is_where_this_guard_thinks_it_is() -> None:
    """A guard over an empty file list is green about nothing."""
    assert len(_CURRENT) >= 10, f"only found {len(_CURRENT)} current diagrams under {_DIAGRAMS}"


@pytest.mark.parametrize("diagram", _CURRENT, ids=lambda p: p.name)
def test_no_current_diagram_names_the_retired_qt_surface(diagram: Path) -> None:
    """Against the pre-#1090 tree this fails on three files.

    `subprocess_launch_sequence` (step 7's `QApplication + MainWindow` and a
    step 6b spawning `src.arcade.telemetry`), `tcp_broadcast_dataflow` (the live
    subscriber box), and `system_architecture` (`PySide6 dashboard`).
    """
    labels = _labels(diagram)
    # draw.io can save a diagram COMPRESSED, as one base64 blob with no `value=`
    # attribute in sight. The check would then read nothing and pass, which is
    # the empty-set green this repo has a written lesson about, and it would do
    # it silently on a file someone merely re-saved from the app.
    #
    # Counted on CELL values, not on `_labels`: a compressed file still exposes
    # its `<diagram name=...>` tab, so "at least one label" is satisfied by the
    # page name alone. Asserting the wrong one of the two would have closed this
    # hole on paper and left it open.
    cells = [text for text in labels if text not in _tab_names(diagram)]
    assert cells, (
        f"{diagram.name} exposes no cell labels, only its tab name. It is probably saved "
        f"compressed; re-save it as 'Uncompressed' so its text stays readable to a diff and "
        f"to this check."
    )
    text = "\n".join(labels)
    named = sorted({word for word in _RETIRED if word in text})
    assert named == [], (
        f"{diagram.name} still names {named}. Either redraw it against the code, or rename it "
        f"with the {_LEGACY_SUFFIX} marker if it is being kept as a record of a dead surface."
    )


def test_the_qt_legacy_diagram_still_carries_what_makes_it_legacy() -> None:
    """The exemption is real, not a spelling nobody checks.

    Without this, renaming a file to `*_legacy.drawio` would exempt it whatever
    it contains, and the guard above could be satisfied by moving a stale
    diagram rather than fixing it.

    Only the Qt one, named by its filename. The other legacy file is the
    Streamlit page tree, which is a different dead surface and names none of
    the words above; asserting the vocabulary against it would be asserting the
    wrong thing about the right file.
    """
    qt_legacy = [p for p in _LEGACY if "qt" in p.stem.lower()]
    assert len(qt_legacy) == 1, (
        f"expected one Qt legacy diagram, found {[p.name for p in qt_legacy]}"
    )
    text = "\n".join(_labels(qt_legacy[0]))
    named = sorted({word for word in _RETIRED if word in text})
    assert named, (
        f"{qt_legacy[0].name} carries the legacy marker but names no retired Qt surface; "
        f"either it was redrawn and should lose the marker, or the marker is wrong"
    )
