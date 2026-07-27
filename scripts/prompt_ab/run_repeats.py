"""Stage 3: run ONE lap N times under both variants. The experiment that has power.

A single-run A/B over a whole race cannot see anything on this stack: two identical
passes already disagree on most fields, so the per-lap diff is swamped. The effects
that matter live on the few laps where the call is genuinely in play, and the only
way to measure those is repetition on one lap.

The memory block is frozen at what a source pass held ENTERING the target lap, so
every repeat sees the same history and the only thing varying is the sampler.

This is the experiment that produced the audit's two significant results: at Lusail
2025 lap 44 memory alone agreed with the deterministic Monte Carlo on 4 of 10 runs
against 6 of 10 without it, and 10 of 10 once the counterweight was in the block;
under an injected Safety Car, memory took the free stop on 7 of 8 runs against 1 of 8.

Usage:
    python -m scripts.prompt_ab.run_repeats --inputs .../lusail_nor.pkl \\
        --history .../pass_a.json --lap 44 --repeats 10 --out .../transition.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from scripts.prompt_ab._common import (
    Usage,
    assemble,
    build_prompt,
    checkpoint,
    invoke_with_retry,
    load_env,
    recommendation_row,
)

warnings.filterwarnings("ignore")


def _memory_entering(history_rows: list[dict], lap: int):
    """Rebuild the accumulator from a source pass's recommendations before ``lap``.

    Reconstructed rather than read off a stored block, so ANY pass can serve as the
    history and the block is always produced by the shipping ``DecisionMemory``
    rather than by whatever was rendered at the time.
    """
    from src.strategy.inference.decision_memory import DecisionMemory

    memory = DecisionMemory()
    for row in history_rows:
        if row["lap"] >= lap:
            break
        memory.record(
            row["lap"],
            SimpleNamespace(
                action=row["action"],
                pit_lap_target=row["pit_lap_target"],
                contingencies=[SimpleNamespace(**c) for c in row["contingencies"]],
            ),
        )
    return memory


def _summarise(rows: list[dict], lap: int) -> None:
    """Print the contingency table the result is actually read from."""
    same_lap = [r for r in rows if r["lap"] == lap]
    if not same_lap:
        return
    print(f"\n-- lap {lap} (deterministic MC: {same_lap[0]['det_action']}) --")
    for variant in ("none", "memory"):
        runs = [r for r in same_lap if r["variant"] == variant]
        if not runs:
            continue
        pit_family = sum(1 for r in runs if r["action"] in ("PIT_NOW", "UNDERCUT", "OVERCUT"))
        actions = Counter(r["action"] for r in runs)
        mean_conf = sum(r["confidence"] for r in runs) / len(runs)
        print(
            f"  {variant:<7} n={len(runs):<3} pit-family {pit_family}/{len(runs)}  "
            f"{dict(actions)}  mean confidence {mean_conf:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path, help="a completed pass JSON")
    parser.add_argument("--lap", required=True, type=int)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    provider = load_env()
    print(f"provider={provider}  lap={args.lap}  repeats={args.repeats}")

    from src.agents.strategy_orchestrator import _get_orchestrator_llm

    records = {r["lap"]: r for r in pickle.loads(args.inputs.read_bytes())}
    history_rows = json.loads(args.history.read_text(encoding="utf-8"))["rows"]
    record = records[args.lap]

    memory_block = _memory_entering(history_rows, args.lap).block()
    print(f"--- memory entering lap {args.lap} ---\n{memory_block}")

    llm = _get_orchestrator_llm()
    usage = Usage()
    rows: list[dict] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for variant, block in (("none", None), ("memory", memory_block)):
        prompt = build_prompt(record, block)
        for repeat in range(args.repeats):
            synthesis = invoke_with_retry(llm, prompt, usage)
            recommendation = assemble(record, synthesis)
            row = recommendation_row(args.lap, recommendation)
            row.update(
                variant=variant,
                repeat=repeat,
                det_action=record["det_action"],
                best_mc=record["best_mc"],
            )
            rows.append(row)
            print(
                f"lap {args.lap} [{variant}] rep{repeat}: {row['action']:<9} "
                f"conf={row['confidence']:.2f} target={row['pit_lap_target']}",
                flush=True,
            )
            checkpoint(args.out, {"lap": args.lap, "usage": usage.as_dict(), "rows": rows})

    _summarise(rows, args.lap)
    print(f"\n{usage.calls} calls, {usage.prompt_tokens} in / {usage.completion_tokens} out")


if __name__ == "__main__":
    main()
