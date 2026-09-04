"""
F1 StratLab: Interactive CLI Launcher

Usage
-----
    python scripts/f1_cli.py

Menu
----
    1  Single Driver   : lap-by-lap strategy for one driver
    2  Head-to-Head    : one run, second driver tracked as rival (--rival)
    3  Quit

All UI logic lives in scripts/cli/:
    theme.py    : colors, console, ASCII banner
    pickers.py  : Rich prompts (mode / race / driver / laps / provider)
    runner.py   : subprocess helpers + mode handlers
"""

from __future__ import annotations

import logging as _logging
import sys
import warnings
from pathlib import Path

# Ensure UTF-8 on Windows terminals. Some hosts (IDE consoles, pytest's
# capture, certain CI runners) replace sys.stdout/stderr with objects that
# support reconfigure() but raise ValueError/OSError for an unsupported
# combination of args on that particular stream - not worth enumerating for
# a best-effort cosmetic setting; on failure the stream just keeps its
# original encoding.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Silence noisy library output before any heavy imports
warnings.filterwarnings("ignore", message=".*builtin type.*__module__.*")
_logging.getLogger("transformers").setLevel(_logging.ERROR)
_logging.getLogger("setfit").setLevel(_logging.ERROR)
_logging.getLogger("sentence_transformers").setLevel(_logging.ERROR)
_logging.getLogger("torch").setLevel(_logging.ERROR)

# ── Paths ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = next(
    (p for p in [_SCRIPT_DIR, *_SCRIPT_DIR.parents] if (p / ".git").exists()),
    _SCRIPT_DIR.parent,
)

# Add scripts/ to sys.path so `from cli.*` resolves correctly
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Also add the repo root (or tool install parent) so `from src.f1_strat_manager…`
# resolves when running from a clone. uv tool install already installs src/* as
# an importable package, so this is a no-op in that case.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv

    _env = _REPO_ROOT / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass  # python-dotenv not installed - rely on env vars being set manually

# ── CLI package imports ────────────────────────────────────────────────────────
from cli.pickers import ask_again, discover_races, pick_mode  # noqa: E402
from cli.runner import run_h2h, run_single  # noqa: E402
from cli.theme import F1_GRAY, console, make_banner  # noqa: E402
from rich.rule import Rule  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────


def _run_wizard() -> None:
    """Inner wizard loop. Wrapped by :func:`main` for Ctrl+C handling."""
    console.print()
    console.print(make_banner())

    # First-run bootstrap: download race data + model weights from HF Hub when
    # the cache is fresh. Editable-dev checkouts (repo has data/raw/2025/<gp>/
    # + data/models/...) short-circuit is_first_run() so this is a no-op for
    # the repo contributor workflow.
    try:
        from src.f1_strat_manager.data_cache import ensure_setup, is_first_run

        if is_first_run():
            ensure_setup()
    except ImportError:
        # Package not installed (extremely rare: only if the user copied
        # scripts/ in isolation). Fall through and let discover_races error.
        pass

    races = discover_races(_REPO_ROOT, year=2025)
    if not races:
        console.print(
            f"\n  [{F1_GRAY}]No races found in data/raw/2025/. "
            f"Run scripts/download_data.py first.[/{F1_GRAY}]\n"
        )
        return

    while True:
        mode = pick_mode()

        if mode == "quit":
            break
        elif mode == "single":
            run_single(races, _REPO_ROOT, _SCRIPT_DIR)
        elif mode == "h2h":
            run_h2h(races, _REPO_ROOT, _SCRIPT_DIR)

        if not ask_again():
            break

        console.print()
        console.print(Rule(style=F1_GRAY))
        console.print()
        console.print(make_banner())

    console.print()
    console.print(f"  [{F1_GRAY}]Goodbye. Chequered flag.[/{F1_GRAY}]")
    console.print()


def main() -> None:
    """Console-script entry point (see ``[project.scripts]`` in pyproject.toml).

    Wraps :func:`_run_wizard` in a :class:`KeyboardInterrupt` guard so
    Ctrl+C anywhere in the wizard (arrow pickers, ``Prompt.ask`` calls,
    the simulation subprocess) prints a single-line ``Interrupted.`` in
    italic dim and exits with status 130, the conventional SIGINT code,
    instead of leaking a stack trace.
    """
    try:
        _run_wizard()
    except KeyboardInterrupt:
        console.print()
        console.print(f"  [italic {F1_GRAY}]Interrupted.[/italic {F1_GRAY}]")
        console.print()
        sys.exit(130)


if __name__ == "__main__":
    main()
