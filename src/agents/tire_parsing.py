"""Pure parsing of the tyre tools' output strings — no models, no I/O, no weights.

A leaf module so that reading a number the tools printed costs nothing. ``tire_agent``
builds ``TireAgentConfig()`` at import time, which reads
``data/models/tire_degradation/routing_config.json`` — a file that comes from Hugging
Face and is not in git. So importing the parser from there requires the model bundles
to be on disk, and ``tests/audit/test_tire_agent_hardening.py`` skips its **entire
module** for that reason, taking its pure-parser test down with it.

That is how this function ended up with no CI coverage while being the single point
where every numeric field of ``TireOutput`` is produced. Splitting it out is the same
move ``src/strategy/inference/guard_rails.py`` made for the pit bounds, for the same
reason: a value that is cheap to read should not be behind an import that is expensive.

--- WHERE TO CHANGE IF A TOOL'S OUTPUT FORMAT CHANGES ---
The patterns below are contracts with the f-strings in ``tire_agent``'s
``_build_tools`` — ``predict_tire_deg_tool`` and ``estimate_laps_to_cliff_tool``.
Change a printed label there and the matching pattern here has to move with it, or the
field silently reverts to "absent" and its consumer sees ``None``.
"""

from __future__ import annotations

import re

# Ordered (pattern, key) pairs, applied to every ToolMessage in the history.
#
# `-?[\d.]+` rather than `[\d.]+` (#477): the bare digit class cannot match a leading
# minus, so a negative degradation rate — real and expected per the tyre agent's own
# system prompt ("track evolution or fuel load reduction") — silently failed to parse
# and fell through to a 0.0 default.
#
# `Cumulative degradation` (#727) is the TCN's own scalar. predict_tire_deg_tool has
# printed it since the tool existed and nothing ever read it, so the prediction the
# whole N07-N10 family exists to produce stopped here. It takes the same `-?` for the
# same reason: a set faster than its own fresh baseline is real early in a stint.
#
# `Fresh reference` (#744b) is the same model on the same stint's early laps. It is
# printed only when those laps exist, so an absent key means "no reference could be
# taken" and the wear it feeds stays None. Same `-?` for the same reason: the fresh
# reference is itself measured against N04's slow baseline and is routinely negative.
_PATTERNS: tuple[tuple[str, str], ...] = (
    (r'Fresh reference:\s*(-?[\d.]+)',        'fresh_ref'),
    (r'Cumulative degradation:\s*(-?[\d.]+)', 'cum_deg'),
    (r'Degradation rate:\s*(-?[\d.]+)',       'deg_rate'),
    (r'P10:\s*(-?[\d.]+)',                    'p10'),
    (r'P50:\s*(-?[\d.]+)',                    'p50'),
    (r'P90:\s*(-?[\d.]+)',                    'p90'),
)


def parse_tool_outputs(messages: list) -> dict:
    """Extract the numeric fields the tyre tools printed, keyed only when matched.

    Reads the structured lines the tools emit rather than the LLM's free-text answer,
    so the values returned are the exact numbers inference computed regardless of how
    the model phrased its summary.

    A key is present **only if its pattern matched**. That is the contract callers
    depend on: an absent key means "the tool was skipped or its output did not parse",
    which is a different fact from a present 0.0, and collapsing the two turned a
    silent regex miss into a confident call to box (#436). The first match wins, so a
    field printed by both tools is read from whichever ran first.

    Args:
        messages: LangChain message objects from the agent's invoke result. Anything
            without a string ``.content`` is skipped rather than coerced.

    Returns:
        Dict of matched field name to float. Possibly empty; never padded with
        defaults, because choosing a default is the caller's decision and it differs
        per field.
    """
    result: dict[str, float] = {}
    for msg in messages:
        content = getattr(msg, 'content', '')
        if not isinstance(content, str):
            continue
        for pattern, key in _PATTERNS:
            match = re.search(pattern, content)
            if match and key not in result:
                result[key] = float(match.group(1))
    return result
