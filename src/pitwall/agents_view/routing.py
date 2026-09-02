"""Which conditional agents the router picked, lap by lap.

The six consoles say what every agent produced NOW. This says which ones were
consulted over the last stretch of race, which is the only per-lap variation the
routing layer has and the one thing the cards cannot show.

**Only the CONDITIONAL agents get a lane.** The four always-on ones have exactly
one live state - they all ran, or the whole lap errored and the status bar says
so - and a row of permanently lit marks would be decoration rather than a
reading.

**This is not a trace and does not pretend to be one.** Nothing on the wire says
in what ORDER the agents ran or how long each took: `active` is serialised from a
Python set, and the per-stage timings are measured and then deliberately dropped
(#1045, a pit wall does not show model latency). The order is a compile-time
constant of the pipeline, so drawing a path would convey nothing per lap. What
varies, and therefore what is drawn, is membership.
"""

from __future__ import annotations

# The router's own conditional agents, defined ONCE.
#
# `panels.build_cards` gates the PIT and RAG cards on these same two ids and used
# to carry its own copies of the strings. A second place that knows "N28 means
# pit" is this repository's most productive defect, so the cards read this table
# rather than repeating it.
ROUTING_LANES: tuple[tuple[str, str], ...] = (("N28", "PIT"), ("N30", "RAG"))
