"""One tick in, one AGENTS view out.

The object that makes the 1:1 port true by construction: it calls the
same formatters and runs the same accumulators as the Qt window, so the
React side is a renderer and never a second implementation of anything.
`src/arcade/dashboard/window.py::_on_data` is the function this mirrors,
in the same order, because the order is load-bearing - the history is
seeded before `latest` is folded in, and the status bar is set last so it
reflects whatever the pipeline reported for THIS tick.

It is stateful, which is why it is an object and not a function: the two
chart histories accumulate across ticks because `history_tail` strips
`per_agent` and nothing can rebuild those values later.
"""

from __future__ import annotations

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

    def build(self, payload: dict[str, Any], connection: str = "Connected") -> dict[str, Any]:
        """The whole view for one tick."""
        strategy = payload.get("strategy") or {}
        latest = strategy.get("latest") or {}

        self._accumulate(strategy, latest)

        return {
            "view_version": AGENTS_VIEW_VERSION,
            "seq": payload.get("seq"),
            "header": build_header(payload, connection),
            "orchestrator": build_orchestrator(latest or None),
            "scenarios": build_scenarios(latest.get("scenario_scores") if latest else None),
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

    def _accumulate(self, strategy: dict[str, Any], latest: dict[str, Any]) -> None:
        """Fold this tick into the chart history. A rewind evicts nothing.

        See `LapHistory`: the laps ahead of a backwards seek are real,
        deterministic observations, and the predictions among them are
        the one thing the wire sends exactly once. Dropping them is a
        loss; keeping them is what the Qt window does.
        """
        self._history.seed_from_tail(strategy.get("history_tail") or [])
        self._history.ingest_latest(latest)
