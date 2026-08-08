"""What PITWALL's AGENTS window is built out of.

The window is a 1:1 port of the Qt strategy window, and the way that is
kept true is not inspection: **the host calls the same formatters the Qt
window calls**, so the two cannot describe the same lap differently. This
file guards the properties that make that possible.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

REUSED_BY_PITWALL = (
    "src.arcade.dashboard.agent_formatters",
    "src.arcade.palette",
)


def _import_in_a_fresh_interpreter(module: str) -> set[str]:
    """Top-level packages a cold import of `module` pulls in.

    A subprocess, not `sys.modules`: by the time pytest reaches this file
    another test has already imported PySide6, so an in-process check
    would assert about the session's history rather than about the module.
    """
    script = textwrap.dedent(f"""
        import sys
        import {module}  # noqa: F401
        print(",".join(sorted({{name.split(".")[0] for name in sys.modules}})))
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    return set(result.stdout.strip().split(","))


def test_the_reused_formatters_need_no_display_stack_and_no_dataframes():
    """PITWALL's host runs in a process with no Qt, and should not pay for pandas.

    Both were true before the palette split: `agent_formatters` imported
    `dashboard.theme`, which imports PySide6 and — through
    `classify_action` — `src.arcade.strategy`, measured at 0.410 s and
    pandas. Six colour tuples and two badge builders should cost neither.

    This is also what un-skipped the palette-mirror test below: reading
    the Python palette needed libEGL on a headless runner, so the one
    guard against the two copies drifting never ran in CI.
    """
    for module in REUSED_BY_PITWALL:
        loaded = _import_in_a_fresh_interpreter(module)
        assert "PySide6" not in loaded, f"{module} drags in a display stack"
        assert "pandas" not in loaded, f"{module} drags in pandas"
        assert "pyglet" not in loaded and "arcade" not in loaded, f"{module} drags in the replay"


def test_the_badge_builders_escape_what_comes_off_the_wire():
    """Compound labels and alert intents are agent output, not literals here.

    In Qt an unescaped `<` breaks the rich-text parser; in PITWALL these
    strings reach a webview, where the same characters are markup.
    """
    from src.arcade.palette import compound_pill_html, flag_chip_html

    assert "&lt;b&gt;" in compound_pill_html("<b>SOFT")
    assert "<b>" not in compound_pill_html("<b>SOFT").removeprefix("<span")
    assert "&lt;" in flag_chip_html("A<B")
