"""Drive decision windows through the SHIPPED LLM path and record every lap.

WHY IT DRIVES THE CLI INSTEAD OF CALLING ``run_lap`` DIRECTLY
------------------------------------------------------------
The thing being measured is what the product recommends, and the product is
``f1-sim``. Between a bare ``run_lap`` call and the CLI sit the real radio
corpus, the Safety Car tracker that re-asserts a deployment on the laps between
RCMs, and the ``DecisionMemory`` block that goes into the orchestrator prompt.
Reimplementing those here would produce a second copy of race-state shaping,
which is the single most productive defect this repository has: the numbers
would describe the copy, not the product.

So the CLI is invoked in-process, once per window, and ``run_lap`` is wrapped to
record what it was asked and what it answered. The CLI keeps owning the inputs.

CRASH SAFETY AND RESUME
-----------------------
Each lap is appended to a JSONL as soon as it returns and the handle is flushed.
A run of this shape is hours long and costs money per call, so a crash on lap
900 must not throw away laps 1 to 899. On restart, laps already present in the
JSONL for the same ``(race, driver, lap, pass_index)`` are skipped and not
re-billed.

Sample spec (JSON, a list of objects)::

    [{"race": "Budapest", "driver": "LEC", "team": "Ferrari",
      "low": 14, "high": 24, "note": "LEC covers PIA's undercut on L19"}]

``repeats`` runs each window more than once. The orchestrator requests
``temperature=0`` and ``gpt-5.4-mini`` discards it, so the LLM path is NOT
deterministic and a single pass measures one sample of a distribution.
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

# Installed before anything under src.agents is imported, so no client escapes it.
from src.strategy.eval.token_meter import install  # noqa: E402


class WindowRecorder:
    """Wraps ``engine.run_lap`` to append one JSONL row per evaluated lap.

    The wrapper is installed once and re-pointed at each window through
    :meth:`begin`, because the CLI resolves ``run_lap`` by module attribute on
    every lap and a per-window patch would stack.
    """

    def __init__(self, meter, out_path: Path) -> None:
        self._meter = meter
        self._out_path = out_path
        self._handle = out_path.open("a", encoding="utf-8")
        self._context: dict[str, object] = {}
        self.done: set[tuple[str, str, int, int]] = _already_measured(out_path)
        self.rows_this_run = 0

    def begin(self, race: str, driver: str, pass_index: int) -> None:
        """Point the recorder at the window about to run."""
        self._context = {"race": race, "driver": driver, "pass_index": pass_index}

    def install(self) -> None:
        """Patch ``engine.run_lap``. Idempotent by construction (called once)."""
        import src.strategy.inference.engine as inference_engine

        original = inference_engine.run_lap

        def recording_run_lap(race_state, *args, **kwargs):
            before = self._meter.totals()
            started = time.perf_counter()
            result = original(race_state, *args, **kwargs)
            elapsed = time.perf_counter() - started
            after = self._meter.totals()
            self._append(
                race_state, result[0], result[1], elapsed, before, after, kwargs.get("profile")
            )
            return result

        inference_engine.run_lap = recording_run_lap

    def _append(self, race_state, recommendation, outputs, elapsed, before, after, profile) -> None:
        row = {
            **self._context,
            "lap": int(getattr(race_state, "lap", 0)),
            "profile": profile or "rich",
            "seconds": round(elapsed, 3),
            "llm_calls": after.calls - before.calls,
            "prompt_tokens": after.prompt_tokens - before.prompt_tokens,
            "completion_tokens": after.completion_tokens - before.completion_tokens,
            "state": {
                "position": getattr(race_state, "position", None),
                "compound": getattr(race_state, "compound", None),
                "tyre_life": getattr(race_state, "tyre_life", None),
                "gap_ahead_s": getattr(race_state, "gap_ahead_s", None),
                "total_laps": getattr(race_state, "total_laps", None),
                "risk_tolerance": getattr(race_state, "risk_tolerance", None),
            },
            "recommendation": _dump(recommendation),
            "agents": _agent_summary(outputs),
        }
        self._handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._handle.flush()
        self.rows_this_run += 1

    def close(self) -> None:
        self._handle.close()


def _dump(recommendation) -> dict:
    """Every field of the recommendation, whatever pydantic version is in play."""
    if hasattr(recommendation, "model_dump"):
        return recommendation.model_dump()
    if hasattr(recommendation, "dict"):
        return recommendation.dict()
    return {"action": getattr(recommendation, "action", None)}


# The sub-agent signals a decision has to be explained by. The recommendation alone
# says WHAT the stack answered; without these there is no way to tell a model that
# read the race correctly and chose oddly from one that was fed the wrong race. The
# Qatar case is exactly that distinction, so recording only the action would have
# made the hardest window in the sample undiagnosable.
_AGENT_FIELDS: dict[str, tuple[str, ...]] = {
    "situation_out": (
        "sc_prob_3lap",
        "sc_currently_active",
        "vsc_active",
        "overtake_prob",
        "gap_ahead_s",
        "pace_delta_s",
        "threat_level",
    ),
    "pace_out": ("predicted_lap_time_s", "delta_s", "pace_mode"),
    "tire_out": ("laps_to_cliff_p10", "laps_to_cliff_p50", "deg_rate", "alert"),
    "pit_out": ("recommended_lap", "compound_recommendation", "undercut_target", "pit_duration_s"),
    "radio_out": ("alert", "sentiment", "intent"),
}


def _agent_summary(outputs) -> dict:
    """A flat, JSON-safe read of the sub-agent outputs, plus routing context.

    Only named fields are pulled. Dumping whole dataclasses would carry model
    objects and free-text reasoning into every row and make a long run's JSONL
    unreadable, and the reasoning is already recoverable from the recommendation.
    """
    if not outputs:
        return {}

    # The RAG's QUESTION and its retrieved ARTICLE list, not just whether it ran.
    # `regulation_context` on the recommendation is the agent's free-text answer,
    # and `rag_agent.run_rag_agent`'s own docstring warns that the article numbers
    # inside that text are not the retrieved ones. Recording both is the only way
    # to tell a misretrieval from a fabricated citation after the fact.
    rag = outputs.get("rag") or {}
    summary: dict[str, object] = {
        "active": outputs.get("active"),
        "guardrail_reason": outputs.get("guardrail_reason"),
        "rag": bool(rag),
        "rag_question": rag.get("question") if isinstance(rag, dict) else None,
        "rag_articles": rag.get("articles") if isinstance(rag, dict) else None,
    }
    for slot, fields in _AGENT_FIELDS.items():
        value = outputs.get(slot)
        if value is None:
            continue
        summary[slot] = {name: getattr(value, name) for name in fields if hasattr(value, name)}
    return summary


def _already_measured(path: Path) -> set[tuple[str, str, int, int]]:
    """The (race, driver, lap, pass) keys already on disk, so a resume skips them.

    A malformed trailing line is tolerated rather than fatal: it is what a kill
    mid-write leaves behind, and refusing to start because of it would make the
    crash-safety mechanism the reason the resume fails.
    """
    if not path.exists():
        return set()
    seen: set[tuple[str, str, int, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        seen.add(
            (
                str(row.get("race")),
                str(row.get("driver")),
                int(row.get("lap", -1)),
                int(row.get("pass_index", 0)),
            )
        )
    return seen


def _window_is_complete(recorder: WindowRecorder, window: dict, pass_index: int) -> bool:
    """True when every lap of this window is already on disk for this pass."""
    wanted = range(int(window["low"]), int(window["high"]) + 1)
    race, driver = str(window["race"]), str(window["driver"])
    return all((race, driver, lap, pass_index) in recorder.done for lap in wanted)


def _run_window(window: dict, year: int, provider: str, no_llm: bool, extra: list[str]) -> None:
    """Invoke the real CLI over one window, in-process."""
    from scripts.run_simulation_cli import _parse_args, run

    argv = [
        "run_simulation_cli.py",
        str(window["race"]),
        str(window["driver"]),
        str(window["team"]),
        "--year",
        str(year),
        "--laps",
        f"{window['low']}-{window['high']}",
        "--no-first-run",
        *extra,
    ]
    if no_llm:
        argv.append("--no-llm")
    else:
        argv += ["--provider", provider]
    if window.get("rival"):
        argv += ["--rival", str(window["rival"])]

    sys.argv = argv
    run(_parse_args())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="Path to the sample-spec JSON")
    parser.add_argument("--out", required=True, help="JSONL to append per-lap rows to")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--provider", default="openai", choices=["openai", "lmstudio"])
    parser.add_argument("--repeats", type=int, default=1, help="Passes per window")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Deterministic profile. For rehearsing the harness without billing.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Run only windows whose race matches this string. Batching handle.",
    )
    parser.add_argument(
        "--sim-arg",
        action="append",
        default=[],
        help="Extra flag forwarded to f1-sim verbatim, repeatable.",
    )
    args = parser.parse_args()

    windows = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    if args.only:
        windows = [w for w in windows if args.only.lower() in str(w["race"]).lower()]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meter = install()
    recorder = WindowRecorder(meter, out_path)
    recorder.install()

    started = time.perf_counter()
    planned = sum(int(w["high"]) - int(w["low"]) + 1 for w in windows for _ in range(args.repeats))
    print(
        f"[measure] {len(windows)} windows x {args.repeats} pass(es), {planned} laps planned",
        file=sys.stderr,
    )

    for pass_index in range(args.repeats):
        for window in windows:
            label = f"{window['race']}/{window['driver']}/p{pass_index}"
            if _window_is_complete(recorder, window, pass_index):
                print(f"[measure] skip {label}: already on disk", file=sys.stderr)
                continue
            recorder.begin(str(window["race"]), str(window["driver"]), pass_index)
            print(f"[measure] run  {label} laps {window['low']}-{window['high']}", file=sys.stderr)
            try:
                _run_window(window, args.year, args.provider, args.no_llm, args.sim_arg)
            except Exception as error:  # noqa: BLE001
                # One unusable race must not abandon the other windows: the run is
                # hours long and the rows already written are still valid. The
                # failure is named on stderr and the window simply stays unmeasured,
                # which the analyser reports as missing rather than as a result.
                print(f"[measure] FAILED {label}: {type(error).__name__}: {error}", file=sys.stderr)

    recorder.close()
    elapsed = time.perf_counter() - started
    print(
        f"[measure] done: {recorder.rows_this_run} laps in {elapsed / 60:.1f} min -> {out_path}",
        file=sys.stderr,
    )
    print(json.dumps(meter.as_dict(), indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
