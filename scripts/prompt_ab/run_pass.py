"""Stage 2: sweep a cached race under ONE prompt variant. One API call per lap.

Two variants:

``none``    today's prompt, verbatim. Run it TWICE to get a noise floor - without
            one, an A/B result on this stack is uninterpretable, because the Layer 3
            model samples (see OrchestratorCFG) and two identical passes disagreed
            on ``confidence`` in 36 of 41 laps.
``memory``  the same prompt plus the block from ``DecisionMemory``, accumulated
            from THIS pass's own previous recommendations, which is how a live
            surface would build it.

Usage:
    python -m scripts.prompt_ab.run_pass --inputs data/eval/prompt_ab/lusail_nor.pkl \\
        --variant none --out data/eval/prompt_ab/pass_a.json
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

from scripts.prompt_ab._common import (
    Usage,
    add_model_flag,
    apply_model_flag,
    assemble,
    build_prompt,
    checkpoint,
    invoke_with_retry,
    load_env,
    recommendation_row,
)

# Safe at module scope: decision_memory imports nothing but the standard library.
# The orchestrator is the one that has to be deferred into main(), because importing
# it pulls in the tire agent, which reads its routing config at import time.
from src.strategy.inference.decision_memory import DecisionMemory

warnings.filterwarnings("ignore")

VARIANTS = ("none", "memory")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="none")
    parser.add_argument("--laps", default=None, help="optional sub-range, e.g. 40-45")
    add_model_flag(parser)
    args = parser.parse_args()

    provider = load_env()

    from src.agents.strategy_orchestrator import _get_orchestrator_llm

    model_name = apply_model_flag(args.model)
    print(f"provider={provider}  variant={args.variant}  model={model_name}")

    records = pickle.loads(args.inputs.read_bytes())
    if args.laps:
        first, _, last = args.laps.partition("-")
        lo, hi = int(first), int(last or first)
        records = [r for r in records if lo <= r["lap"] <= hi]

    llm = _get_orchestrator_llm()
    usage = Usage()
    memory = DecisionMemory()
    rows = []

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for record in records:
        block = memory.block() if args.variant == "memory" else None
        synthesis = invoke_with_retry(llm, build_prompt(record, block), usage)
        recommendation = assemble(record, synthesis)
        memory.record(record["lap"], recommendation)

        row = recommendation_row(record["lap"], recommendation)
        row.update(
            variant=args.variant,
            memory_block=block,
            det_action=record["det_action"],
            best_mc=record["best_mc"],
        )
        rows.append(row)
        print(
            f"[{args.variant}] lap {record['lap']:>3}  {row['action']:<9} "
            f"conf={row['confidence']:.2f} target={row['pit_lap_target']}",
            flush=True,
        )
        checkpoint(
            args.out,
            {
                "variant": args.variant,
                "model": model_name,
                "usage": usage.as_dict(),
                "rows": rows,
            },
        )

    print(
        f"\n{args.variant}: {usage.calls} calls, {usage.prompt_tokens} in / "
        f"{usage.completion_tokens} out -> {args.out}"
    )


if __name__ == "__main__":
    main()
