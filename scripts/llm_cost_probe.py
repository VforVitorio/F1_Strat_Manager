"""Measure what one lap of the SHIPPED (LLM) path costs, in seconds and tokens.

Sizing question this exists to answer: the 2025 measurement session wants the
stack driven in ``rich`` mode over decision windows, and nobody knows what that
costs. A full-season sweep through the deterministic path is already ~11.5 h of
wall clock at 0.51 s/lap; with an unbounded number of LLM calls per lap the
ceiling is unknown and it is billed per call.

The unit is the LAP, not the race, because the session's sample is made of
windows rather than whole races. Boot cost (model weights, Whisper, the RAG
store) is reported separately for the same reason: it is paid once per process,
so it amortises differently from per-lap cost and mixing them would inflate a
per-window estimate by a constant.

Usage::

    python scripts/llm_cost_probe.py Budapest LEC Ferrari --laps 17-19 --rival PIA

Anything after the positional arguments is forwarded to ``f1-sim`` verbatim, so
the probe measures the real command rather than an approximation of it.
``--provider openai`` is added when no provider flag is present.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Before anything under src.agents is imported, so no client escapes the patch.
from src.strategy.eval.token_meter import install  # noqa: E402


def _lap_timing_probe(meter) -> list[dict[str, float | int]]:
    """Patch ``engine.run_lap`` to record wall clock and token delta per lap.

    Returns the list the patch appends to. Timing the engine call rather than
    the whole loop keeps the CLI's own rendering out of the per-lap number,
    which is what has to be multiplied by a window size later.
    """
    import src.strategy.inference.engine as inference_engine

    samples: list[dict[str, float | int]] = []
    original_run_lap = inference_engine.run_lap

    def timed_run_lap(race_state, *args, **kwargs):
        before = meter.totals()
        started = time.perf_counter()
        result = original_run_lap(race_state, *args, **kwargs)
        elapsed = time.perf_counter() - started
        after = meter.totals()
        samples.append(
            {
                "lap": int(getattr(race_state, "lap", 0)),
                "seconds": round(elapsed, 3),
                "calls": after.calls - before.calls,
                "prompt_tokens": after.prompt_tokens - before.prompt_tokens,
                "completion_tokens": after.completion_tokens - before.completion_tokens,
                "cached_prompt_tokens": after.cached_prompt_tokens - before.cached_prompt_tokens,
            }
        )
        return result

    inference_engine.run_lap = timed_run_lap
    return samples


def _summarise(samples: list[dict[str, float | int]]) -> dict[str, float]:
    """Per-lap averages over the laps that were actually simulated."""
    if not samples:
        return {}
    count = len(samples)
    summary = {
        "laps": count,
        "mean_seconds_per_lap": round(sum(s["seconds"] for s in samples) / count, 2),
        "max_seconds_per_lap": round(max(s["seconds"] for s in samples), 2),
        "mean_llm_calls_per_lap": round(sum(s["calls"] for s in samples) / count, 2),
        "mean_prompt_tokens_per_lap": round(sum(s["prompt_tokens"] for s in samples) / count),
        "mean_completion_tokens_per_lap": round(
            sum(s["completion_tokens"] for s in samples) / count
        ),
    }
    return summary


def _build_argv(probe_args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    """The argv ``f1-sim`` would have been invoked with."""
    argv = [
        "run_simulation_cli.py",
        probe_args.gp_name,
        probe_args.driver,
        probe_args.team,
        *passthrough,
    ]
    if "--provider" not in passthrough:
        argv += ["--provider", "openai"]
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gp_name")
    parser.add_argument("driver")
    parser.add_argument("team")
    parser.add_argument(
        "--out",
        default="documents/audits/PROBE_llm_cost.json",
        help="Where to write the measurement (default: %(default)s)",
    )
    probe_args, passthrough = parser.parse_known_args()
    out_path = REPO_ROOT / probe_args.out
    passthrough = [flag for flag in passthrough if not flag.startswith("--out")]

    meter = install()
    samples = _lap_timing_probe(meter)

    from scripts.run_simulation_cli import _parse_args, run

    sys.argv = _build_argv(probe_args, passthrough)
    started = time.perf_counter()
    run(_parse_args())
    total_seconds = time.perf_counter() - started

    lap_seconds = sum(float(s["seconds"]) for s in samples)
    measurement = {
        "command": " ".join(sys.argv[1:]),
        "wall_clock_seconds_total": round(total_seconds, 1),
        "wall_clock_seconds_in_laps": round(lap_seconds, 1),
        "wall_clock_seconds_boot_and_render": round(total_seconds - lap_seconds, 1),
        "per_lap": _summarise(samples),
        "tokens": meter.as_dict(),
        "laps": samples,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(measurement, indent=2), encoding="utf-8")

    print("\n\n===== COST PROBE =====", file=sys.stderr)
    print(
        json.dumps({k: v for k, v in measurement.items() if k != "laps"}, indent=2), file=sys.stderr
    )
    print(f"written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
