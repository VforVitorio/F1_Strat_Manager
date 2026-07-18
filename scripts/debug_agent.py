"""
Single-agent debug harness — run and inspect any agent in isolation.

Builds a minimal lap_state from CLI arguments (no replay engine, no
full pipeline), loads the featured parquet once, and calls the selected
agent's *_from_state entry point. Prints the full output dataclass.

Usage
-----
    python scripts/debug_agent.py --agent <name> [options]

Agents
------
    pace        N25 — lap time prediction + CI
    tire        N26 — tire degradation + laps to cliff
    situation   N27 — overtake probability + SC risk
    pit         N28 — pit duration, undercut score, compound recommendation
    radio       N29 — sentiment, intent, NER, RCM parsing
    rag         N30 — regulation context retrieval
    orchestrator N31 — full multi-agent synthesis (calls all sub-agents)

Examples
--------
    python scripts/debug_agent.py --agent tire --gp Melbourne --lap 20 --driver NOR --team McLaren
    python scripts/debug_agent.py --agent pace --gp Bahrain --lap 35 --driver HAM --team Mercedes
    python scripts/debug_agent.py --agent situation --gp Monaco --lap 50 --driver VER --team "Red Bull Racing"
    python scripts/debug_agent.py --agent pit --gp Silverstone --lap 28 --driver NOR --team McLaren --compound MEDIUM --tyre-life 18
    python scripts/debug_agent.py --agent orchestrator --gp Melbourne --lap 20 --driver NOR --team McLaren
    python scripts/debug_agent.py --agent radio --gp Bahrain --lap 10 --driver NOR --team McLaren --radio "Box box, tyres are gone"

Override any lap_state field with --override key=value:
    python scripts/debug_agent.py --agent tire --gp Melbourne --lap 20 --driver NOR --team McLaren \\
        --override tyre_life=25 compound=MEDIUM position=3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import traceback

# Force UTF-8 output on Windows terminals that default to cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from typing import Any

import pandas as pd

# Load .env from repo root so OPENAI_API_KEY is available for provider='openai'
try:
    from dotenv import load_dotenv

    _env_path = next(
        (
            p / ".env"
            for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
            if (p / ".git").exists() and (p / ".env").exists()
        ),
        None,
    )
    if _env_path:
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed — rely on env vars being set manually

# ---------------------------------------------------------------------------
# Repo-root sys.path injection
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = next(
    (p for p in [_SCRIPT_DIR, *_SCRIPT_DIR.parents] if (p / ".git").exists()),
    _SCRIPT_DIR.parent,
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_DIM = "\033[2m"
_RED = "\033[91m"


def _header(text: str) -> None:
    width = 72
    print(f"\n{_BOLD}{_CYAN}{'─' * width}{_RESET}")
    print(f"{_BOLD}{_CYAN}  {text}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'─' * width}{_RESET}\n")


def _section(title: str) -> None:
    print(f"\n{_BOLD}{_YELLOW}▸ {title}{_RESET}")


def _kv(key: str, value: Any, indent: int = 2) -> None:
    pad = " " * indent
    val_str = str(value)
    print(f"{pad}{_DIM}{key:<30}{_RESET}{val_str}")


def _print_output(out: Any) -> None:
    """Pretty-print an agent output dataclass or dict."""
    _section("Agent output")
    if dataclasses.is_dataclass(out) and not isinstance(out, type):
        for f in dataclasses.fields(out):
            _kv(f.name, getattr(out, f.name))
    elif hasattr(out, "__dict__"):
        for k, v in vars(out).items():
            _kv(k, v)
    elif isinstance(out, dict):
        for k, v in out.items():
            _kv(k, v)
    else:
        print(f"  {out}")


# ---------------------------------------------------------------------------
# Lap state builder
# ---------------------------------------------------------------------------

_COMPOUND_ID_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}


def _build_lap_state(args: argparse.Namespace, laps_df: pd.DataFrame) -> dict[str, Any]:
    """Build a minimal but realistic lap_state from CLI args + parquet lookup.

    Tries to find a real row from the featured parquet for the given driver,
    GP, and lap so the state reflects real data. Falls back to synthetic
    defaults when not found.
    """
    gp_name = args.gp_name
    driver = args.driver
    lap_num = args.lap

    # Try to find actual lap data from the featured parquet
    real_row = None
    if laps_df is not None and not laps_df.empty:
        mask = (laps_df.get("Driver", laps_df.get("driver", pd.Series(dtype=str))) == driver) & (
            laps_df.get("LapNumber", laps_df.get("lap_number", pd.Series(dtype=int))) == lap_num
        )
        if "GrandPrix" in laps_df.columns:
            mask &= laps_df["GrandPrix"].str.contains(gp_name, case=False, na=False)
        candidates = laps_df[mask]
        if not candidates.empty:
            real_row = candidates.iloc[0]

    def _get(col_candidates: list[str], default: Any) -> Any:
        if real_row is None:
            return default
        for col in col_candidates:
            if col in real_row.index and pd.notna(real_row[col]):
                return real_row[col]
        return default

    compound = args.compound or str(_get(["Compound", "compound"], "SOFT"))
    tyre_life = args.tyre_life or int(_get(["TyreLife", "tyre_life"], 10))
    position = int(_get(["Position", "position"], 1))
    lap_time_s = float(_get(["LapTime_s", "lap_time_s", "LapTime"], 91.0))
    speed_st = float(_get(["SpeedST", "speed_st"], 305.0))
    fuel_load = float(_get(["FuelLoad", "fuel_load"], max(0.0, 110 - lap_num * 1.8)))
    air_temp = float(_get(["AirTemp", "air_temp"], 28.0))
    track_temp = float(_get(["TrackTemp", "track_temp"], 45.0))
    rainfall = bool(_get(["Rainfall", "rainfall"], False))

    lap_state: dict[str, Any] = {
        "lap_number": lap_num,
        "driver": {
            "driver": driver,
            "team": args.team,
            "lap_number": lap_num,
            "compound": compound,
            "compound_id": _COMPOUND_ID_MAP.get(compound.upper(), 0),
            "tyre_life": tyre_life,
            "position": position,
            "lap_time_s": lap_time_s,
            "speed_st": speed_st,
            "fuel_load": fuel_load,
            "stint": 1,
            "fresh_tyre": tyre_life <= 2,
            "track_status": "1",
            "is_in_lap": False,
            "is_out_lap": False,
            "gap_to_leader_s": 0.0 if position == 1 else float(position - 1) * 1.5,
        },
        "rivals": [],
        "weather": {
            "air_temp": air_temp,
            "track_temp": track_temp,
            "rainfall": rainfall,
            "track_status": "1",
        },
        "session_meta": {
            "gp_name": gp_name,
            "year": args.year,
            "driver": driver,
            "team": args.team,
            "total_laps": args.total_laps,
        },
    }

    # Apply --override key=value overrides
    if args.override:
        for kv in args.override:
            if "=" not in kv:
                print(
                    f"{_YELLOW}[WARN] Ignoring malformed --override '{kv}' (expected key=value){_RESET}"
                )
                continue
            k, v = kv.split("=", 1)
            # Try to auto-cast
            for cast in (int, float, lambda x: x.lower() == "true"):
                try:
                    v = cast(v)  # type: ignore[arg-type]
                    break
                except (ValueError, AttributeError):
                    pass
            lap_state["driver"][k] = v
            print(f"{_DIM}  override: driver.{k} = {v}{_RESET}")

    return lap_state


# ---------------------------------------------------------------------------
# Agent runners
# ---------------------------------------------------------------------------


def _run_pace(lap_state: dict, laps_df: pd.DataFrame, args: argparse.Namespace) -> None:
    from src.agents.pace_agent import run_pace_agent_from_state

    out = run_pace_agent_from_state(lap_state)
    _print_output(out)


def _run_tire(lap_state: dict, laps_df: pd.DataFrame, args: argparse.Namespace) -> None:
    from src.agents.tire_agent import run_tire_agent_from_state

    out = run_tire_agent_from_state(lap_state, laps_df)
    _print_output(out)


def _run_situation(lap_state: dict, laps_df: pd.DataFrame, args: argparse.Namespace) -> None:
    from src.agents.race_situation_agent import run_race_situation_agent_from_state

    out = run_race_situation_agent_from_state(lap_state, laps_df)
    _print_output(out)


def _run_pit(lap_state: dict, laps_df: pd.DataFrame, args: argparse.Namespace) -> None:
    from src.agents.pit_strategy_agent import run_pit_strategy_agent_from_state

    out = run_pit_strategy_agent_from_state(lap_state, laps_df)
    _print_output(out)


def _run_radio(lap_state: dict, laps_df: pd.DataFrame, args: argparse.Namespace) -> None:
    from src.agents.radio_agent import run_radio_agent_from_state

    radio_msgs = []
    if args.radio:
        radio_msgs = [{"driver": args.driver, "text": args.radio, "lap": args.lap}]
    enriched = {**lap_state, "radio_msgs": radio_msgs, "rcm_events": [], "lap": args.lap}
    out = run_radio_agent_from_state(enriched, laps_df)
    _print_output(out)


def _run_rag(lap_state: dict, laps_df: pd.DataFrame, args: argparse.Namespace) -> None:
    from src.agents.rag_agent import run_rag_agent

    query = args.query or f"What are the safety car regulations for the {args.gp_name} GP?"
    out = run_rag_agent(query)
    _print_output(out)


def _run_orchestrator(lap_state: dict, laps_df: pd.DataFrame, args: argparse.Namespace) -> None:
    from src.agents.strategy_orchestrator import RaceState, run_strategy_orchestrator_from_state

    driver_st = lap_state["driver"]
    race_state = RaceState(
        driver=args.driver,
        lap=args.lap,
        total_laps=args.total_laps,
        position=driver_st["position"],
        compound=driver_st["compound"],
        tyre_life=driver_st["tyre_life"],
        gap_ahead_s=0.0,
        pace_delta_s=0.0,
        air_temp=lap_state["weather"].get("air_temp", 28.0),
        track_temp=lap_state["weather"].get("track_temp", 45.0),
        rainfall=bool(lap_state["weather"].get("rainfall", False)),
    )
    out = run_strategy_orchestrator_from_state(race_state, laps_df, lap_state)
    _print_output(out)


_RUNNERS = {
    "pace": _run_pace,
    "tire": _run_tire,
    "situation": _run_situation,
    "pit": _run_pit,
    "radio": _run_radio,
    "rag": _run_rag,
    "orchestrator": _run_orchestrator,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-agent debug harness — run any agent in isolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--agent", required=True, choices=list(_RUNNERS), help="Which agent to run")
    p.add_argument(
        "--gp", dest="gp_name", required=True, help="Grand Prix name (e.g. Melbourne, Bahrain)"
    )
    p.add_argument("--driver", required=True, help="FIA three-letter driver code (e.g. NOR, HAM)")
    p.add_argument("--team", required=True, help="Team name as in the parquet (e.g. McLaren)")
    p.add_argument("--lap", type=int, default=20, help="Lap number to simulate (default: 20)")
    p.add_argument(
        "--compound",
        default=None,
        help="Tyre compound override (SOFT/MEDIUM/HARD/INTERMEDIATE/WET)",
    )
    p.add_argument(
        "--tyre-life", dest="tyre_life", type=int, default=None, help="Tyre life in laps override"
    )
    p.add_argument(
        "--total-laps",
        dest="total_laps",
        type=int,
        default=57,
        help="Total race laps (default: 57)",
    )
    p.add_argument("--year", type=int, default=2025, help="Season year (default: 2025)")
    p.add_argument(
        "--featured",
        default="data/processed/laps_featured_2025.parquet",
        help="Path to featured parquet (default: data/processed/laps_featured_2025.parquet)",
    )
    p.add_argument(
        "--override",
        nargs="+",
        metavar="KEY=VALUE",
        help="Override individual driver state fields, e.g. --override tyre_life=25 position=3",
    )
    p.add_argument(
        "--radio",
        default=None,
        help="Radio message text for --agent radio (e.g. 'Box box, tyres gone')",
    )
    p.add_argument("--query", default=None, help="Query string for --agent rag")
    p.add_argument(
        "--print-state",
        action="store_true",
        help="Print the full lap_state dict before running the agent",
    )
    p.add_argument(
        "--provider",
        default="lmstudio",
        choices=["lmstudio", "openai"],
        help="LLM provider: 'lmstudio' (default, localhost:1234) or 'openai' (real API, needs OPENAI_API_KEY)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Propagate provider to agents via env var BEFORE any agent module is imported
    # (singletons like _get_orchestrator_llm check this on first call)
    os.environ["F1_LLM_PROVIDER"] = args.provider

    _header(
        f"Debug — {args.agent.upper()} agent  |  "
        f"{args.gp_name} lap {args.lap}  |  {args.driver} / {args.team}"
    )

    # Load featured parquet
    featured_path = Path(args.featured)
    laps_df: pd.DataFrame | None = None
    if featured_path.exists():
        print(f"{_DIM}Loading featured parquet…{_RESET}", end=" ", flush=True)
        # Route through the shared augmenter so this debug surface runs on the same frame
        # the shipping paths do. A raw read leaves off Time_s, which collapses the overtake
        # gap to a lap-time delta (#486); the debug harness is where you go to understand a
        # bad number, so it is the worst possible place to run the degraded frame.
        from src.f1_strat_manager.laps_augment import augment_featured_laps

        laps_df = augment_featured_laps(pd.read_parquet(featured_path), args.year)
        print(f"done ({len(laps_df):,} rows)")
    else:
        print(
            f"{_YELLOW}[WARN] Featured parquet not found at {featured_path} — using synthetic defaults{_RESET}"
        )
        laps_df = pd.DataFrame()

    # Build lap_state
    _section("Lap state")
    lap_state = _build_lap_state(args, laps_df)

    if args.print_state:
        print(json.dumps(lap_state, indent=2, default=str))
    else:
        d = lap_state["driver"]
        _kv("lap_number", lap_state["lap_number"])
        _kv("compound", d["compound"])
        _kv("tyre_life", d["tyre_life"])
        _kv("position", d["position"])
        _kv("lap_time_s", d["lap_time_s"])
        _kv("fuel_load", d["fuel_load"])
        _kv("gp_name", lap_state["session_meta"]["gp_name"])

    # Run
    runner = _RUNNERS[args.agent]
    print(f"\n{_DIM}Running {args.agent} agent…{_RESET}\n")
    try:
        runner(lap_state, laps_df, args)
        print(f"\n{_GREEN}Done.{_RESET}\n")
    except Exception as exc:
        print(f"\n{_RED}[ERROR] {type(exc).__name__}: {exc}{_RESET}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
