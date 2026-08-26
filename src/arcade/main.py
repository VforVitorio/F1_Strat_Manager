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
    return parser.parse_args()


def _show_menu(window: arcade.Window) -> None:
    from src.arcade.views import MenuView

    window.show_view(MenuView(window))


def _show_viewer_directly(window: arcade.Window, args: argparse.Namespace) -> None:
    """Legacy path: bypass the menu and build a F1ArcadeView from CLI args."""
    from src.arcade.app import F1ArcadeView
    from src.arcade.config import get_gp_names
    from src.arcade.data import SessionLoader
    from src.arcade.track import Track

    year = args.year
    round_num = args.round
    gp = get_gp_names(year).get(round_num, f"Round{round_num}")

    # gp is the arcade's display label from the GP_NAMES table, which does
    # not stay in sync with the active season calendar (``GP_NAMES[3]`` is
    # "Australia" but 2025 round 3 is Suzuka). The real race is resolved by
    # FastF1 inside SessionLoader; re-log after the load with the
    # authoritative Location so the startup trace does not mislead.
    logger.info("Requesting session: year=%d round=%d (label=%s)", year, round_num, gp)
    session_data = SessionLoader().load(year, round_num, gp)
    logger.info(
        "Loaded session: %s %d (label=%s) — %d drivers, laps %d-%d, %d frames",
        session_data.location or "?",
        year,
        gp,
        len(session_data.frames_by_driver),
        session_data.min_lap_number,
        session_data.max_lap_number,
        session_data.total_frames,
    )

    ref_x, ref_y = session_data.ref_lap_xy
    track = Track(
        ref_x=ref_x,
        ref_y=ref_y,
        drs_flags=session_data.ref_lap_drs,
        rotation_deg=session_data.circuit_rotation_deg,
    )

    driver_main = args.driver or _pick_default_driver(session_data)
    driver_rival = args.driver2
    if driver_main not in session_data.frames_by_driver:
        logger.warning("Driver %s not in session; falling back", driver_main)
        driver_main = _pick_default_driver(session_data)
    if driver_rival and driver_rival not in session_data.frames_by_driver:
        logger.warning("Rival %s not in session; ignoring", driver_rival)
        driver_rival = None

    view = F1ArcadeView(
        window=window,
        session_data=session_data,
        track=track,
        driver_main=driver_main,
        driver_rival=driver_rival,
        year=year,
        strategy_enabled=args.strategy,
        team=args.team,
    )
    window.show_view(view)


def _pick_default_driver(session_data) -> str:
    """Prefer a finisher of the final lap; fall back to first driver otherwise."""
    codes = list(session_data.frames_by_driver.keys())
    if not codes:
        raise RuntimeError("Session has no drivers")
    for code in codes:
        frames = session_data.frames_by_driver[code]
        if frames and frames[-1].active:
            return code
    return codes[0]


if __name__ == "__main__":
    main()
