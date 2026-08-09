"""Stage 1: cache one race's per-lap orchestrator inputs. ZERO API calls.

Drives real laps through the full agent stack on the ``no-llm`` profile, which
never builds an LLM client, and pickles everything ``_build_orchestrator_prompt``
needs. Every later stage reads this cache, so a prompt variant can be measured
repeatedly without re-running the models.

Known deviation, and it must be stated wherever the results are:
``no-llm`` never runs the conditional agents, so ``pit_out`` is always ``None``
and ``regulation_context`` always empty. Production (``rich``) populates them ONLY
when N28/N30 are routed. Measured at Lusail 2025 that is 2 of 41 green laps, so 39
prompts are shape-identical to production and 2 are not. Under a Safety Car both
are always routed, so ``--safety-car`` runs diverge on every lap.

Usage:
    python -m scripts.prompt_ab.gen_inputs --gp Lusail --driver NOR --team McLaren \\
        --year 2025 --laps 5-45 --out data/eval/prompt_ab/lusail_nor.pkl
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

from scripts.prompt_ab._common import REPO_ROOT

warnings.filterwarnings("ignore")


def _parse_lap_range(text: str) -> tuple[int, int]:
    first, _, last = text.partition("-")
    return int(first), int(last or first)


def _race_state_for(lap_state: dict, driver: str, safety_car: bool):
    """Build the orchestrator's RaceState from one replay lap.

    ``air_temp``/``track_temp`` are required by the model, so the weather fallback
    is not optional politeness. A Safety Car is injected the way the CLI does it,
    through an RCM message, rather than by setting a flag: that routes N27 the same
    way a real neutralisation would.
    """
    from src.agents.strategy_orchestrator import RaceState

    driver_state = lap_state["driver"]
    weather = lap_state.get("weather") or {}
    lap = int(driver_state["lap_number"])
    rcm_events = []
    if safety_car:
        rcm_events = [
            {"message": "SAFETY CAR DEPLOYED", "category": "SafetyCar", "flag": "", "lap": lap}
        ]
    return RaceState(
        driver=driver,
        lap=lap,
        total_laps=int(lap_state["session_meta"]["total_laps"]),
        position=int(driver_state["position"]),
        compound=driver_state.get("compound") or "MEDIUM",
        tyre_life=int(driver_state.get("tyre_life") or 10),
        # None is a valid RaceState value now - no car ahead to measure (#878) -
        # and the old `or 2.0` also destroyed a genuinely measured 0.0, two cars
        # side by side. The two-arg .get already returns None for a missing key.
        gap_ahead_s=driver_state.get("gap_ahead_s"),
        pace_delta_s=0.0,
        risk_tolerance=0.5,
        air_temp=weather.get("air_temp") or 25.0,
        track_temp=weather.get("track_temp") or 35.0,
        rcm_events=rcm_events,
    )


def _record_for(race_state, lap_state, laps_df, safety_car: bool) -> dict:
    """Run one lap deterministically and keep everything the prompt needs."""
    from src.agents.strategy_orchestrator import (
        _run_mc_simulation,
        best_mc_candidate,
        race_context_from_lap_state,
    )
    from src.strategy.inference.engine import run_lap

    recommendation, outputs, _timings = run_lap(
        race_state, laps_df, lap_state, profile="no-llm", return_agent_outputs=True
    )
    context = race_context_from_lap_state(lap_state, race_state)
    mc_results = _run_mc_simulation(
        pace_out=outputs["pace_out"],
        tire_out=outputs["tire_out"],
        situation_out=outputs["situation_out"],
        pit_out=outputs["pit_out"],
        alpha=race_state.risk_tolerance,
        rivals=lap_state.get("rivals"),
        position=context.get("position"),
        laps_remaining=context.get("laps_remaining"),
        pit_context=context.get("pit_context"),
    )
    return {
        "lap": race_state.lap,
        "race_state": race_state,
        "mc_results": mc_results,
        "best_mc": best_mc_candidate(mc_results),
        "pace_out": outputs["pace_out"],
        "tire_out": outputs["tire_out"],
        "situation_out": outputs["situation_out"],
        "pit_out": outputs["pit_out"],
        "radio_out": outputs["radio_out"],
        "regulation_context": outputs.get("regulation_context") or "",
        "det_action": str(getattr(recommendation, "action", "?")),
        "sc_active": bool(getattr(outputs["situation_out"], "sc_currently_active", False)),
        "safety_car_injected": safety_car,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gp", required=True, help="race directory name under data/raw/<year>/")
    parser.add_argument("--driver", required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--laps", default="5-45", help="inclusive range, e.g. 5-45 or 44")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--safety-car",
        action="store_true",
        help="inject an SC via RCM on every lap (adversarial runs)",
    )
    args = parser.parse_args()

    import pandas as pd

    from src.f1_strat_manager.laps_augment import augment_featured_laps
    from src.simulation.replay_engine import RaceReplayEngine

    lap_min, lap_max = _parse_lap_range(args.laps)
    featured = REPO_ROOT / "data" / "processed" / f"laps_featured_{args.year}.parquet"
    laps_df = augment_featured_laps(pd.read_parquet(featured), args.year)
    race_dir = REPO_ROOT / "data" / "raw" / str(args.year) / args.gp
    engine = RaceReplayEngine(str(race_dir), args.driver, args.team, interval_seconds=0)

    records = []
    for lap_state in engine.replay():
        lap = int(lap_state["driver"]["lap_number"])
        if not lap_min <= lap <= lap_max:
            continue
        race_state = _race_state_for(lap_state, args.driver, args.safety_car)
        record = _record_for(race_state, lap_state, laps_df, args.safety_car)
        records.append(record)
        print(
            f"lap {lap:>3}  det={record['det_action']:<9} best_mc={record['best_mc']:<9} "
            f"sc_active={record['sc_active']}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(pickle.dumps(records))
    print(f"\nWROTE {len(records)} laps -> {args.out}  (0 API calls)")


if __name__ == "__main__":
    main()
