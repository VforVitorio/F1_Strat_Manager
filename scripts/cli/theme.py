"""scripts/cli/theme.py

F1 brand colors, shared Rich Console singleton, and the ASCII banner.

Exported symbols used by the rest of the cli package:
  console   : single Rich Console for the whole session
  F1_*      : color constants (#rrggbb)
  make_banner() → Panel
"""

from __future__ import annotations

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

# ─────────────────────────────────────────────────────────────────────────────
# Shared console singleton
# ─────────────────────────────────────────────────────────────────────────────

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# F1 brand palette
# ─────────────────────────────────────────────────────────────────────────────

F1_RED = "#e10600"  # official Formula 1 primary red
F1_WHITE = "#f0f0f0"  # warm white for primary text
F1_GRAY = "#6b7280"  # muted gray for metadata / hints
F1_GREEN = "#10b981"  # success / green flag
F1_AMBER = "#f59e0b"  # warning / yellow flag / driver 2 accent

# ─────────────────────────────────────────────────────────────────────────────
# ASCII art: "F1" + "STRAT" in Unicode block chars
#
# F1 part (red) + STRAT part (white), 6 rows, ~57 visual columns total.
# Renders cleanly in any 80-column terminal with UTF-8 support.
# ─────────────────────────────────────────────────────────────────────────────

_ART_F1 = [
    "██████╗  ██╗",
    "██╔════╝███║",
    "█████╗  ╚██║",
    "██╔══╝   ██║",
    "██║      ██║",
    "╚═╝      ╚═╝",
]

_ART_STRAT = [
    "  ███████╗████████╗██████╗  █████╗ ████████╗",
    "  ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝",
    "  ███████╗   ██║   ██████╔╝███████║   ██║   ",
    "  ╚════██║   ██║   ██╔══██╗██╔══██║   ██║   ",
    "  ███████║   ██║   ██║  ██║██║  ██║   ██║   ",
    "  ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝  ",
]

_COMPACT = (
    f"[bold {F1_RED}]F1[/bold {F1_RED}] "
    f"[bold {F1_WHITE}]STRAT[/bold {F1_WHITE}]"
    f"  [dim]F1 StratLab[/dim]"
)


def make_banner() -> Panel:
    """Return the F1 STRAT welcome banner as a Rich Panel.

    Uses the full block-char logo when the terminal is wide enough (>= 66 cols),
    and falls back to a compact single-line title for narrow terminals.
    """
    if console.width < 66:
        content = Group(
            Align.center(Text.from_markup(_COMPACT)),
            Align.center(Text("Multi-Agent Race Intelligence System · v0.9", style=F1_GRAY)),
        )
        return Panel(content, border_style=F1_RED, padding=(0, 2))

    lines: list = []
    for f1_row, strat_row in zip(_ART_F1, _ART_STRAT):
        t = Text()
        t.append(f1_row, style=f"bold {F1_RED}")
        t.append(strat_row, style=f"bold {F1_WHITE}")
        lines.append(Align.center(t))

    lines += [
        Text(""),
        Align.center(Text("F1 StratLab", style=f"bold {F1_GRAY}")),
        Align.center(Text("Multi-Agent Race Intelligence System · v0.9", style=F1_GRAY)),
    ]

    return Panel(Group(*lines), border_style=F1_RED, padding=(1, 4))
