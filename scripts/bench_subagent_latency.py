"""Sub-agent isolated latency benchmark: six agents, one fixed lap_state.

Builds a single ``lap_state`` from Suzuka 2025 NOR lap 21 (or the
documented Bahrain fallback) and times every ``run_*_from_state``
entry point in isolation. The orchestrator (N31) is intentionally
excluded: the goal is to characterise each sub-agent's per-call cost
rather than the end-to-end pipeline.

Some agents make external calls at runtime: ``RadioAgent`` and
``RagAgent`` invoke the LLM through LM Studio / OpenAI when reachable
and fall back to deterministic stubs when no provider is configured.
The artefact ``notes`` column documents this so a high latency on
those rows is interpretable.

Usage::

    uv run scripts/bench_subagent_latency.py [--n-warmup 5] [--n-runs 100]
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Repo-root path injection
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = next(
    (p for p in [_SCRIPT_DIR, *_SCRIPT_DIR.parents] if (p / ".git").exists()),
    _SCRIPT_DIR.parent,
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load .env so OPENAI_API_KEY and other secrets are available when the
# RadioAgent / RagAgent reach out to the configured LLM provider.
try:
    from dotenv import load_dotenv

    _env = _REPO_ROOT / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

# Library log noise: silence aggressive INFO from transformers / setfit
warnings.filterwarnings("ignore", category=FutureWarning)
for noisy in ("transformers", "setfit", "sentence_transformers", "torch", "src"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from scripts.bench._common import (  # noqa: E402
    BenchResult,
    export_csv,
    export_markdown,
    make_start_panel,
    render_results_table,
    time_function,
)
from scripts.cli.theme import console  # noqa: E402
from src.simulation.race_state_manager import RaceStateManager  # noqa: E402

_DATA_ROOT = _REPO_ROOT / "data"
_EVAL_DIR = _DATA_ROOT / "eval"

_FIXTURE_OPTIONS = (
    {
        "label": "Suzuka 2025 NOR lap 21",
        "laps_path": _DATA_ROOT / "raw" / "2025" / "Suzuka" / "laps.parquet",
        "driver": "NOR",
        "team": "McLaren",
        "gp_name": "Suzuka",
        "year": 2025,
        "lap_number": 21,
    },
    {
        "label": "Bahrain 2025 NOR lap 18",
        "laps_path": _DATA_ROOT / "raw" / "2025" / "Bahrain" / "laps.parquet",
        "driver": "NOR",
        "team": "McLaren",
        "gp_name": "Bahrain",
        "year": 2025,
        "lap_number": 18,
    },
)


class SubAgentLatencyRunner:
    """Time every sub-agent's RSM adapter against a fixed lap_state.

    Construction loads the fixture parquet once, builds a
    :class:`RaceStateManager`, snapshots the requested lap into a
    ``lap_state`` dict, and pre-loads the featured-laps DataFrame
    needed by the agents that take ``laps_df`` as a second argument.
    The actual measurements run inside :meth:`run`.
    """

    def __init__(self, n_warmup: int = 5, n_runs: int = 100, device: str = "auto") -> None:
        """Resolve the fixture, build the lap_state, lazy-import the agents.

        The agent imports are deferred to :meth:`run` so the
        construction cost does not pollute the warm-up phase reported
        in the artefact.
        """
        self.n_warmup = int(n_warmup)
        self.n_runs = int(n_runs)
        self.device_label = self._resolve_device(device)

        fixture = self._pick_fixture()
        if fixture is None:
            raise FileNotFoundError(
                "No fixture parquet available — neither Suzuka 2025 nor Bahrain 2025 "
                "laps.parquet exists on disk; cannot build a lap_state."
            )
        self.fixture_label = fixture["label"]

        laps_raw = pd.read_parquet(fixture["laps_path"])
        self.rsm = RaceStateManager(
            laps_df=laps_raw,
            driver_code=fixture["driver"],
            team=fixture["team"],
            gp_name=fixture["gp_name"],
            year=fixture["year"],
        )
        self.lap_state = self.rsm.get_lap_state(fixture["lap_number"])
        self.lap_state["lap"] = fixture["lap_number"]
        self.lap_state["question"] = "What is the minimum pit stop duration under Safety Car?"

        # Featured 2025 laps: required by Tire / RaceSituation / Pit / Radio
        # agents because they pull historical context out of the laps frame. Route through
        # the shared augmenter so Time_s is present; a raw read collapses the overtake gap
        # to a lap-time delta (#486), which would skew the very latencies this bench reports.
        # Same fix as the CLI and arcade surfaces.
        from src.f1_strat_manager.laps_augment import augment_featured_laps

        self.laps_featured = augment_featured_laps(
            pd.read_parquet(_DATA_ROOT / "processed" / "laps_featured_2025.parquet"),
            2025,
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @staticmethod
    def _pick_fixture() -> Optional[dict[str, Any]]:
        """Return the first declared fixture whose laps parquet exists on disk."""
        for fixture in _FIXTURE_OPTIONS:
            if Path(fixture["laps_path"]).exists():
                return fixture
        return None

    # ── Per-agent closures ───────────────────────────────────────────────────

    def _build_agent_calls(self) -> list[tuple[str, Callable[[], Any], str]]:
        """Return ``(agent_name, call_closure, notes)`` for each measured row.

        Imports happen here so a missing optional dependency only
        breaks the affected row; every other agent still gets timed.
        Each closure captures ``self.lap_state`` and any per-agent
        DataFrame so the timed call surface is exactly the public
        ``run_*_from_state`` entry point.
        """
        from src.agents.pace_agent import run_pace_agent_from_state
        from src.agents.pit_strategy_agent import run_pit_strategy_agent_from_state
        from src.agents.race_situation_agent import run_race_situation_agent_from_state
        from src.agents.radio_agent import run_radio_agent_from_state
        from src.agents.rag_agent import run_rag_agent_from_state
        from src.agents.tire_agent import run_tire_agent_from_state

        return [
            ("pace_agent", lambda: run_pace_agent_from_state(self.lap_state), "no external calls"),
            (
                "tire_agent",
                lambda: run_tire_agent_from_state(self.lap_state, self.laps_featured),
                "TCN MC dropout, no external calls",
            ),
            (
                "race_situation_agent",
                lambda: run_race_situation_agent_from_state(self.lap_state, self.laps_featured),
                "LightGBM overtake + SC, may invoke LLM if configured",
            ),
            (
                "pit_strategy_agent",
                lambda: run_pit_strategy_agent_from_state(self.lap_state, self.laps_featured),
                "HistGBT pit duration + LightGBM undercut, may invoke LLM",
            ),
            (
                "radio_agent",
                lambda: run_radio_agent_from_state(self.lap_state, self.laps_featured),
                "BERT sentiment + SetFit intent + BERT NER, LLM synthesis when reachable",
            ),
            (
                "rag_agent",
                lambda: run_rag_agent_from_state(self.lap_state, self.laps_featured),
                "Qdrant retrieval + LLM answer synthesis",
            ),
        ]

    # ── Orchestration ────────────────────────────────────────────────────────

    def run(self) -> list[BenchResult]:
        """Time every agent and return one :class:`BenchResult` per row.

        Runs the agent sequence in declaration order. A failure inside
        any single agent is captured into the artefact with a NaN
        latency and the exception message in ``notes``. This is rare
        in practice but lets the bench surface a partial failure
        without aborting the whole run.
        """
        rows: list[BenchResult] = []
        for agent_name, call, notes in self._build_agent_calls():
            try:
                latency = time_function(call, n_warmup=self.n_warmup, n_runs=self.n_runs)
                rows.append(
                    BenchResult(
                        name=agent_name,
                        metrics={
                            "mean_ms": latency["mean_ms"],
                            "p50_ms": latency["p50_ms"],
                            "p95_ms": latency["p95_ms"],
                            "device": self.device_label,
                            "n_runs": latency["n_runs"],
                            "notes": notes,
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001 - log + continue
                rows.append(
                    BenchResult(
                        name=agent_name,
                        metrics={
                            "mean_ms": float("nan"),
                            "p50_ms": float("nan"),
                            "p95_ms": float("nan"),
                            "device": self.device_label,
                            "n_runs": 0,
                            "notes": f"{notes} — runtime error: {exc!r}",
                        },
                    )
                )
        return rows


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

_COLUMNS = ["agent", "mean_ms", "p50_ms", "p95_ms", "device", "n_runs", "notes"]
_TITLE = "Sub-agent latency (single lap fixture)"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sub-agent isolated latency benchmark.")
    parser.add_argument(
        "--n-warmup", type=int, default=5, help="Warm-up runs per agent (default 5)."
    )
    parser.add_argument(
        "--n-runs", type=int, default=100, help="Measured runs per agent (default 100)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device label for the artefact (auto resolves via torch.cuda.is_available).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    console.print(
        make_start_panel(
            "bench_subagent_latency.py",
            f"6 sub-agents, single lap_state, n_warmup={args.n_warmup} n_runs={args.n_runs}.",
        )
    )

    runner = SubAgentLatencyRunner(
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        device=args.device,
    )
    console.print(f"[dim]Fixture:[/dim] {runner.fixture_label}")
    results = runner.run()

    md_path = _EVAL_DIR / "subagent_latency.md"
    csv_path = _EVAL_DIR / "subagent_latency.csv"
    export_markdown(results, md_path, _TITLE, _COLUMNS)
    export_csv(results, csv_path, _COLUMNS)

    console.print(render_results_table(results, _TITLE, _COLUMNS))
    console.print(f"[green]Markdown:[/green] {md_path.resolve()}")
    console.print(f"[green]CSV:     [/green] {csv_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
