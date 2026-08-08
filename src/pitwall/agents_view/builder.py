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

from src.pitwall.agents_view.decision import build_orchestrator, build_scenarios
from src.pitwall.agents_view.history import LapHistory
from src.pitwall.agents_view.panels import build_cards, build_header, build_status_bar

# Bumped when the view's shape changes in a way a built UI bundle would
# misread. Separate from the wire's `schema_version`, which describes the
# producer: this describes what the host hands the window.
AGENTS_VIEW_VERSION: int = 1


class AgentsViewBuilder:
    """Turn a broadcast payload into what PITWALL · AGENTS renders.

    Invariants:

    - Nothing here formats. Every headline, body line and colour comes
      out of `agent_formatters`, which is the Qt window's own code.
    - The lap history survives across ticks and is evicted, never
      truncated blindly, on a backwards seek.
    """

    def __init__(self) -> None:
        self._history = LapHistory()
        self._last_lap: int | None = None

    def build(self, payload: dict[str, Any], connection: str = "Connected") -> dict[str, Any]:
        """The whole view for one tick."""
        strategy = payload.get("strategy") or {}
        latest = strategy.get("latest") or {}

        self._accumulate(payload, strategy, latest)

        return {
            "view_version": AGENTS_VIEW_VERSION,
            "seq": payload.get("seq"),
            "header": build_header(payload, connection),
            "orchestrator": build_orchestrator(latest or None),
            "scenarios": build_scenarios(latest.get("scenario_scores") if latest else None),
            "cards": build_cards(latest or None),
            "history": {
                "pace": [{"lap": lap, **row} for lap, row in sorted(self._history.pace.items())],
                "tire": self._history.tire_rows(),
            },
            "status_bar": build_status_bar(payload),
        }

    def _accumulate(
        self, payload: dict[str, Any], strategy: dict[str, Any], latest: dict[str, Any]
    ) -> None:
        """Fold this tick into the chart history, evicting first if we rewound.

        The eviction is keyed on the LAP going backwards rather than on
        the producer's `rewound` flag, because that flag reports a
        backwards seek of any size and most of them stay inside the
        current lap, where there is nothing to evict. A lap that actually
        gets re-driven must be re-observed: its predictions belong to the
        timeline the user abandoned.
        """
        lap = (payload.get("arcade") or {}).get("lap")
        if isinstance(lap, int):
            if self._last_lap is not None and lap < self._last_lap:
                self._history.evict_after(lap)
            self._last_lap = lap
        self._history.seed_from_tail(strategy.get("history_tail") or [])
        self._history.ingest_latest(latest)
