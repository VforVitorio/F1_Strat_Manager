"""Entry point for the `f1-arcade` console script.

Default entry shows an in-window menu so the user picks year / round /
drivers / team / strategy-mode with keyboard nav, then loads and launches
the replay from there. The `--viewer` flag is kept as a regression-friendly
shortcut that skips the menu when explicit CLI flags are supplied.
"""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

import arcade
from src.arcade.config import SCREEN_HEIGHT, SCREEN_WIDTH, WINDOW_TITLE

# Load repo-root ``.env`` so OPENAI_API_KEY / F1_LLM_PROVIDER / HF_TOKEN are
# available to the agents spawned by the local strategy pipeline: the CLI
# and backend do the same (``scripts/run_simulation_cli.py`` header) but
# the arcade used to skip this step and silently fell back to whatever was
# already exported in the shell.
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse CLI, create the Window, dispatch to menu (default) or --viewer shortcut."""
    args = _parse_args()
    window = arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, WINDOW_TITLE, resizable=True)
    if args.viewer:
        _show_viewer_directly(window, args)
    else:
        _show_menu(window)
    arcade.run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="f1-arcade",
        description="F1 StratLab - visual race replay.",
    )
    parser.add_argument(
        "--viewer", action="store_true", help="Skip the menu and boot straight into the replay."
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--round", type=int, default=3)
    parser.add_argument("--driver", type=str, default=None)
    parser.add_argument("--driver2", type=str, default=None)
    parser.add_argument("--team", type=str, default="McLaren")
    parser.add_argument(
        "--strategy", action="store_true", help="Enable strategy overlay (requires year 2025)."
    )
    parser.add_argument("--provider", choices=("lmstudio", "openai"), default="openai")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help=(
            "Run the strategy layer on the deterministic profile (zero LLM "
            "clients, zero cost) instead of the LLM-synthesised one. Only "
            "takes effect with --viewer and --strategy; the in-window menu "
            "has no toggle for it yet."
        ),
    )
    return parser.parse_args()


def _show_menu(window: arcade.Window) -> None:
    from src.arcade.views import MenuView

    window.show_view(MenuView(window))


def _show_viewer_directly(window: arcade.Window, args: argparse.Namespace) -> None:
    """Boot straight into the replay, through the menu's own launch path.

    This used to be a second implementation: its own `SessionLoader().load`, its
    own driver fallback, its own `F1ArcadeView`. The menu grew a lazy per-race
    fetch and a worker thread so the window keeps drawing through a download and
    a telemetry build of several minutes; this path would not have grown either, because
    nothing here shared a line with it. It now fills a `LaunchConfig` and hands
    it over, so `--viewer` and the menu can only ever behave the same (#1115).
    """
    from src.arcade.views import LaunchConfig, MenuView

    # Only the flags that were actually passed override the defaults. An empty
    # `--driver` used to fall through to `_pick_default_driver`, which read the
    # loaded session; the form validates before anything is loaded, so an empty
    # string would now surface as "driver must be 3 letters". The menu's own
    # default stands instead, and `_show_replay` still swaps in a driver the
    # session really has if that one is absent from it.
    cfg = LaunchConfig(year=args.year, round_=args.round, team=args.team)
    if args.driver:
        cfg.driver_main = args.driver.upper()
    cfg.mode_two_drivers = bool(args.driver2)
    if args.driver2:
        cfg.driver_rival = args.driver2.upper()
    cfg.strategy_mode = args.strategy
    cfg.no_llm = args.no_llm
    view = MenuView(window)
    window.show_view(view)
    view.launch_with(cfg)


if __name__ == "__main__":
    main()
