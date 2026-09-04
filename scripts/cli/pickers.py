"""scripts/cli/pickers.py

All interactive Rich prompts for the F1 CLI launcher.

Functions
---------
discover_races(repo_root, year) → list[str]
    Scan data/raw/<year>/ and return sorted race names.

pick_mode()           → 'single' | 'h2h' | 'quit'
pick_race(races)      → str
pick_driver(label, repo_root) → (code, team)
pick_rival_code(repo_root)    → str
pick_laps()           → '15-40' | None
pick_provider()       → 'no-llm' | 'openai' | 'lmstudio'
ask_again()           → bool

Arrow-key navigation is used for pick_mode / pick_race / pick_provider.
Falls back to numbered Prompt.ask when stdin is not a tty.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.prompt import Confirm, Prompt

from .theme import F1_AMBER, F1_RED, F1_WHITE, console

# ─────────────────────────────────────────────────────────────────────────────
# Platform / ANSI setup
# ─────────────────────────────────────────────────────────────────────────────

_IS_WIN = sys.platform == "win32"

# Enable VT-100 processing on Windows so ANSI escape codes work in cmd / WT
if _IS_WIN:
    try:
        import ctypes

        _k32 = ctypes.windll.kernel32
        # ENABLE_PROCESSED_OUTPUT | ENABLE_WRAP_AT_EOL_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        _k32.SetConsoleMode(_k32.GetStdHandle(-11), 7)
    except Exception:
        # Best-effort cosmetic setup only: on failure the console just falls
        # back to plain (non-ANSI) output, which is harmless. The failure
        # surface here is ctypes/OS-level (unsupported console host, sandboxed
        # environment) rather than a specific Python exception type worth
        # enumerating.
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Arrow-key selection menu
# ─────────────────────────────────────────────────────────────────────────────


def _arrow_pick(title: str, options: list[str], default: int = 0) -> int:
    """Show an interactive arrow-key menu and return the selected 0-based index.

    Renders a list of options with a red ❯ cursor.  Up / Down arrows move the
    cursor; Enter (or Space) confirms.  Ctrl-C raises KeyboardInterrupt.

    Falls back to a numbered Rich Prompt.ask when stdin / stdout is not a tty
    (e.g. piped input, CI environments).
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        sys.stdout.write(f"\n  {title}\n\n")
        for i, opt in enumerate(options, 1):
            sys.stdout.write(f"    {i}. {opt}\n")
        sys.stdout.write("\n")
        sys.stdout.flush()
        raw = Prompt.ask(
            f"  [bold {F1_RED}]›[/bold {F1_RED}] Select",
            choices=[str(i) for i in range(1, len(options) + 1)],
            show_choices=False,
            console=console,
        )
        return int(raw) - 1

    sel = default
    n = len(options)
    # Show index shortcuts when there are enough options to be useful
    _show_idx = n >= 4

    # ── Build the menu text ───────────────────────────────────────────────────
    def _text() -> str:
        parts = ["", f"  {title}", ""]
        for i, opt in enumerate(options):
            idx_hint = f"\033[2m{i + 1:>2}.\033[0m " if _show_idx else "   "
            if i == sel:
                parts.append(f"  \033[1;31m❯\033[0m {idx_hint}\033[1;97m{opt}\033[0m")
            else:
                parts.append(f"     {idx_hint}\033[2m{opt}\033[0m")
        if _show_idx:
            parts.append("  \033[2m[↑↓ arrows  or  type number to jump]\033[0m")
        return "\n".join(parts) + "\n"

    # ── Initial render ────────────────────────────────────────────────────────
    txt = _text()
    sys.stdout.write(txt)
    sys.stdout.flush()
    nl = txt.count("\n")  # number of newlines written → lines to erase

    def _redraw() -> None:
        nonlocal nl
        sys.stdout.write(f"\033[{nl}A\033[0J")
        txt = _text()
        sys.stdout.write(txt)
        sys.stdout.flush()
        nl = txt.count("\n")

    def _confirm() -> None:
        sys.stdout.write(f"\033[{nl}A\033[0J")
        sys.stdout.write(
            f"\n  \033[2m{title}\033[0m\n\n  \033[1;31m❯\033[0m  \033[1;97m{options[sel]}\033[0m\n"
        )
        sys.stdout.flush()

    # ── Key-reading loop ──────────────────────────────────────────────────────
    def _jump_to_digit(ch: str) -> None:
        """Move selection to the option whose 1-based index matches the digit."""
        nonlocal sel
        if ch.isdigit():
            # '1'–'9' → index 0–8; '0' → index 9
            idx = (int(ch) - 1) % 10 if ch != "0" else 9
            if 0 <= idx < n:
                sel = idx
                _redraw()

    if _IS_WIN:
        import msvcrt

        while True:
            ch = msvcrt.getwch()
            if ch in ("\xe0", "\x00"):  # special key prefix
                arrow = msvcrt.getwch()
                if arrow == "H":  # up arrow
                    sel = (sel - 1) % n
                    _redraw()
                elif arrow == "P":  # down arrow
                    sel = (sel + 1) % n
                    _redraw()
            elif ch in ("\r", "\n", " "):  # Enter / Space → confirm
                _confirm()
                return sel
            elif ch == "\x03":  # Ctrl-C
                sys.stdout.write("\n")
                raise KeyboardInterrupt
            else:
                _jump_to_digit(ch)  # digit shortcut
    else:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    seq = sys.stdin.read(2)
                    if seq == "[A":  # up arrow
                        sel = (sel - 1) % n
                        _redraw()
                    elif seq == "[B":  # down arrow
                        sel = (sel + 1) % n
                        _redraw()
                elif ch in ("\r", "\n", " "):
                    _confirm()
                    return sel
                elif ch == "\x03":
                    sys.stdout.write("\n")
                    raise KeyboardInterrupt
                else:
                    _jump_to_digit(ch)  # digit shortcut
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ─────────────────────────────────────────────────────────────────────────────
# Race discovery
# ─────────────────────────────────────────────────────────────────────────────


_UNKNOWN_ROUND = 99  # sorts a folder the calendar does not know after every real round


def _calendar_order(year: int) -> dict[str, int]:
    """Map each race folder name to its round number for one season.

    Args:
        year: the season whose calendar to read.

    Returns:
        Folder name to round number, taken from ``data/tire_compounds_by_race.json``
        via the same folder-name mapper the data cache uses, so a renamed folder
        such as ``Miami_Gardens`` resolves. Empty when neither module can be
        imported, which is the signal to fall back to alphabetical ordering.
    """
    try:
        from src.arcade.config import get_gp_names
        from src.f1_strat_manager.data_cache import race_folder
    except ImportError:
        return {}

    rounds_by_folder = {race_folder(year, name): rnd for rnd, name in get_gp_names(year).items()}
    return rounds_by_folder


def discover_races(repo_root: Path, year: int = 2025) -> list[str]:
    """Return the race directories under the data root, in calendar order.

    Historically this joined ``repo_root / "data" / "raw" / <year>`` because
    the CLI only ran from a git checkout. Now that ``f1-strat`` can be
    installed globally via ``uv tool install`` the data directory may live
    at ``~/.f1-strat/data/`` instead, so we route through
    :func:`src.f1_strat_manager.data_cache.get_data_root` when available
    and fall back to the historical path for dev checkouts without the
    package (e.g. running a raw ``scripts/f1_cli.py`` before ``uv sync``).
    """
    try:
        from src.f1_strat_manager.data_cache import get_data_root

        raw_dir = get_data_root() / "raw" / str(year)
    except ImportError:
        raw_dir = repo_root / "data" / "raw" / str(year)

    if not raw_dir.exists():
        return []
    # Only return folders that actually contain race files: empty
    # placeholders from a partial download would otherwise crash the
    # downstream RaceReplayEngine.
    downloaded = [d.name for d in raw_dir.iterdir() if d.is_dir() and any(d.iterdir())]
    rounds_by_folder = _calendar_order(year)

    def by_round_then_name(folder: str) -> tuple[int, str]:
        """A folder the calendar does not know keeps its place at the end.

        Dropping it instead would hide a race from the menu on a partial or
        hand-assembled download, which is the case this discovery exists for.
        """
        return (rounds_by_folder.get(folder, _UNKNOWN_ROUND), folder)

    return sorted(downloaded, key=by_round_then_name)


# ─────────────────────────────────────────────────────────────────────────────
# Driver → team auto-mapping
# ─────────────────────────────────────────────────────────────────────────────

_DRIVER_TEAM_CACHE: dict[str, str] | None = None


def _resolve_laps_parquet_path(repo_root: Path) -> Path:
    """Resolve the featured-laps parquet path, preferring ``get_data_root``.

    Routes through ``get_data_root`` when the f1_strat_manager package is
    importable so ``uv tool install`` cached layouts work; otherwise falls
    back to the repo-relative path for bare dev checkouts (e.g. running
    ``scripts/f1_cli.py`` before ``uv sync``).
    """
    try:
        from src.f1_strat_manager.data_cache import get_data_root

        return get_data_root() / "processed" / "laps_featured_2025.parquet"
    except ImportError:
        return repo_root / "data" / "processed" / "laps_featured_2025.parquet"


def _load_driver_team_map(repo_root: Path) -> dict[str, str]:
    """Return {driver_code: team} built from laps_featured_2025.parquet (cached).

    Best-effort only: any failure to read or parse the parquet (file
    missing, corrupt parquet, unexpected schema) degrades to an empty map
    rather than raising. ``pick_driver``/``pick_rival_code`` already fall
    back to asking the user to type the team manually whenever the code is
    not found in this map, so a read failure here is never fatal to the CLI.
    """
    global _DRIVER_TEAM_CACHE
    if _DRIVER_TEAM_CACHE is not None:
        return _DRIVER_TEAM_CACHE
    try:
        import pandas as pd

        parquet = _resolve_laps_parquet_path(repo_root)
        if parquet.exists():
            df = pd.read_parquet(parquet, columns=["Driver", "Team"])
            _DRIVER_TEAM_CACHE = (
                df.dropna(subset=["Driver", "Team"])
                .drop_duplicates("Driver", keep="last")
                .set_index("Driver")["Team"]
                .to_dict()
            )
        else:
            _DRIVER_TEAM_CACHE = {}
    except Exception:
        # pandas/pyarrow raise a wide, version-dependent set of types for a
        # malformed or unreadable parquet (OSError variants, pyarrow's own
        # ArrowInvalid/ArrowIOError, KeyError on an unexpected schema) - not
        # worth enumerating for a best-effort convenience lookup.
        _DRIVER_TEAM_CACHE = {}
    return _DRIVER_TEAM_CACHE


# ─────────────────────────────────────────────────────────────────────────────
# Interactive pickers
# ─────────────────────────────────────────────────────────────────────────────


def pick_mode() -> str:
    """Arrow-key mode menu. Returns 'single', 'h2h', or 'quit'."""
    sel = _arrow_pick(
        "What do you want to simulate?",
        [
            "Single Driver   — Lap-by-lap strategy for one driver",
            "Head-to-Head    — Full sim for Driver 1 · Driver 2 tracked as rival",
            "Quit",
        ],
    )
    return ["single", "h2h", "quit"][sel]


def pick_race(races: list[str]) -> str:
    """Arrow-key race selector. Returns race directory name (e.g. 'Melbourne')."""
    sel = _arrow_pick("Available races (2025):", races)
    return races[sel]


def pick_driver(
    label: str = "Driver",
    repo_root: Path | None = None,
) -> tuple[str, str]:
    """Ask for FIA three-letter code; auto-resolve team from parquet.

    If the driver is not found in the parquet (new signing, typo, etc.) a
    manual team entry is requested as fallback.
    """
    console.print()
    code = (
        Prompt.ask(f"  [bold {F1_RED}]›[/bold {F1_RED}] {label} code  [dim](e.g. NOR)[/dim]")
        .upper()
        .strip()
    )

    team = ""
    if repo_root is not None:
        team = _load_driver_team_map(repo_root).get(code, "")

    if team:
        console.print(
            f"  [dim]Team →[/dim] [{F1_WHITE}]{team}[/{F1_WHITE}]  "
            f"[dim](resolved from parquet)[/dim]"
        )
    else:
        team = Prompt.ask(
            f"  [bold {F1_RED}]›[/bold {F1_RED}] Team  "
            f"[dim](not found in parquet — enter manually)[/dim]"
        ).strip()

    return code, team


def pick_rival_code(repo_root: Path | None = None) -> str:
    """Ask for a rival driver code; resolves and displays team for confirmation."""
    console.print()
    code = (
        Prompt.ask(
            f"  [bold {F1_AMBER}]›[/bold {F1_AMBER}] Rival driver code  [dim](e.g. VER)[/dim]"
        )
        .upper()
        .strip()
    )

    if repo_root is not None:
        team = _load_driver_team_map(repo_root).get(code, "")
        if team:
            console.print(
                f"  [dim]Team →[/dim] [{F1_WHITE}]{team}[/{F1_WHITE}]  "
                f"[dim](tracking as rival — no separate simulation)[/dim]"
            )

    return code


def pick_laps() -> str | None:
    """Ask for a lap range. Returns '15-40' string or None (all laps)."""
    console.print()
    raw = Prompt.ask(
        f"  [bold {F1_RED}]›[/bold {F1_RED}] Lap range  [dim](e.g. 15-40, or Enter for all)[/dim]",
        default="all",
    ).strip()
    return None if raw.lower() in ("all", "") else raw


def pick_provider() -> str:
    """Arrow-key LLM provider selector. Returns 'no-llm', 'openai', or 'lmstudio'."""
    sel = _arrow_pick(
        "LLM mode:",
        [
            "No LLM      Fast · ML models only, no synthesis  [recommended]",
            "OpenAI      GPT-4.1-mini · needs OPENAI_API_KEY in .env",
            "LM Studio   Local model at localhost:1234",
        ],
    )
    return ["no-llm", "openai", "lmstudio"][sel]


def ask_again() -> bool:
    """Ask the user if they want to run another simulation."""
    console.print()
    return Confirm.ask(
        f"  [bold {F1_RED}]›[/bold {F1_RED}] Run another simulation?",
        default=True,
    )
