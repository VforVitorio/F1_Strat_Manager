"""One tick in, one AGENTS view out.

The object that makes the 1:1 port true by construction: it calls the
same formatters and runs the same accumulators as the Qt window, so the
React side is a renderer and never a second implementation of anything.
The Qt window's `window.py::_on_data` is the function this mirrors,
in the same order, because the order is load-bearing - the history is
seeded before `latest` is folded in, and the status bar is set last so it
reflects whatever the pipeline reported for THIS tick.

It is stateful, which is why it is an object and not a function: the two
chart histories accumulate across ticks because `history_tail` strips
`per_agent` and nothing can rebuild those values later.
"""

from __future__ import annotations

import logging
from typing import Any

from src.pitwall.agents_view.charts import build_pace_series, build_tire_series
from src.pitwall.agents_view.decision import build_orchestrator, build_scenarios
from src.pitwall.agents_view.history import LapHistory
from src.pitwall.agents_view.panels import build_cards, build_header, build_status_bar
from src.pitwall.agents_view.reasoning import build_reasoning

# Bumped when the view's shape changes in a way a built UI bundle would
# misread. Separate from the wire's `schema_version`, which describes the
# producer: this describes what the host hands the window.
AGENTS_VIEW_VERSION: int = 1

logger = logging.getLogger(__name__)


class AgentsViewBuilder:
    """Turn a broadcast payload into what PITWALL · AGENTS renders.

    Invariants:

    - Nothing here formats. Every headline, body line and colour comes
      out of `agent_formatters`, which is the Qt window's own code.
    - The lap history survives across ticks and across a rewind, because
      the predictions in it are broadcast exactly once.
    """

    def __init__(self) -> None:
        self._history = LapHistory()
        self._run: dict[str, Any] | None = None
        self._seq: int | None = None

    def build(self, payload: dict[str, Any], connection: str = "Connected") -> dict[str, Any]:
        """The whole view for one tick."""
        strategy = payload.get("strategy") or {}
        latest = strategy.get("latest") or {}

        self._reset_if_the_producer_restarted(strategy.get("start"), payload.get("seq"))
        self._accumulate(strategy, latest)

        return {
            "view_version": AGENTS_VIEW_VERSION,
            "seq": payload.get("seq"),
            "header": build_header(payload, connection),
            "orchestrator": build_orchestrator(latest or None),
            # The action goes in with the scores: a guardrail can veto the
            # Monte Carlo winner, and a panel that does not know which plan
            # was ENACTED crowns the one that was overruled (#962).
            "scenarios": build_scenarios(
                latest.get("scenario_scores") if latest else None,
                latest.get("action") if latest else None,
            ),
            "reasoning": build_reasoning(latest or None),
            "cards": build_cards(latest or None),
            "charts": {
                "pace": build_pace_series(self._history.pace),
                "tire": build_tire_series(
                    self._history.tire_rows(),
                    latest.get("lap_number") if latest else None,
                    (latest.get("per_agent") or {}).get("tire") if latest else None,
                ),
            },
            # The raw stores stay on the view as well: the charts are a
            # rendering of them, and a test that could only see the drawn
            # series could not tell an accumulator bug from a plotting one.
            "history": {
                "pace": [{"lap": lap, **row} for lap, row in sorted(self._history.pace.items())],
                "tire": self._history.tire_rows(),
            },
            "status_bar": build_status_bar(payload),
        }

    def _reset_if_the_producer_restarted(
        self, start: dict[str, Any] | None, seq: int | None
    ) -> None:
        """Throw the history away when the tick stops being about the same run.

        **This is the twin of the DATA window's eviction, and it never got
        the signal.** `host.get_tick` deliberately follows a restarted
        producer - relaunch the arcade with the windows open and they must
        not freeze on the dead race - and band 4 evicts client-side, because
        `FrameClock` sees the new run's frame index jump backwards. This
        accumulator lives in the HOST process, which does not restart with
        the arcade, so nothing here ever heard about it.

        Measured before the fix: a Melbourne run to lap 23 followed by a
        Suzuka payload at lap 3 rendered a `Suzuka · 2025` header over
        Melbourne's laps 14-23, with lap 20 reading 81.20 s against Suzuka's
        ~92 s. Worse, it did not simply correct itself: `seed_from_tail` uses
        `setdefault` on purpose, so any lap whose only carrier in the new run
        is the history tail keeps the DEAD run's number permanently.

        Two signals, because one is not enough:

        - **the `start` block changing** catches a different race or driver;
        - **`seq` going backwards** catches the same race relaunched, where
          `start` is identical. It cannot be confused with a rewind: the
          sequence counts messages a producer SENT, so within one run it only
          ever rises. A rewind must NOT evict, and does not - the frame index
          moves, the sequence does not.
        """
        restarted = (start is not None and self._run is not None and start != self._run) or (
            seq is not None and self._seq is not None and seq < self._seq
        )
        if restarted:
            logger.info("Producer restarted - dropping the AGENTS lap history")
            self._history = LapHistory()
        if start is not None:
            self._run = start
        if seq is not None:
            self._seq = seq

    def _accumulate(self, strategy: dict[str, Any], latest: dict[str, Any]) -> None:
        """Fold this tick into the chart history. A rewind evicts nothing.

        See `LapHistory`: the laps ahead of a backwards seek are real,
        deterministic observations, and the predictions among them are
        the one thing the wire sends exactly once. Dropping them is a
        loss; keeping them is what the Qt window does.
        """
        self._history.seed_from_tail(strategy.get("history_tail") or [])
        self._history.ingest_latest(latest)
