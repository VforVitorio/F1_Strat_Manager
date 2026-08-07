"""One air/track temperature pair for the agents, not five (#789).

The same two physical quantities had five different fallback pairs across the two repos,
three of them feeding models: pace read 25.0/35.0, tire and race_situation read 28.0/38.0.
Nothing made them move together and nothing said which was right.

These tests assert the EFFECT that keeps them together: no agent module may state a
temperature fallback of its own. Asserting the constant equals 24.2 would pass forever
while a fourth literal grew back two files away, which is how this started.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
_AGENTS = ROOT / "src" / "agents"

# The modules that build a feature vector from a weather dict. The debug harness and the
# arcade overlay are deliberately out of scope: neither feeds a model, and #789 says so.
_MODEL_FACING = ("pace_agent.py", "tire_agent.py", "race_situation_agent.py")

# `reading_or_default(wx, 'air_temp', 28.0)` and friends: a numeric literal in the
# DEFAULT position of a temperature read.
#
# The comma after the quantity name is what makes this precise. `pace_agent` also holds
# `"AirTemp": (14.5, 33.7)` -- a colon and a two-number tuple, which is N06's measured
# training range rather than a fallback. A looser pattern flagged it, and a bound that
# describes where the model was fitted is exactly the thing this rule must not delete.
_LITERAL_DEFAULT = re.compile(
    r"""(air_temp|track_temp|AirTemp|TrackTemp)["']?\s*,\s*(\d+\.?\d*)\s*\)""",
)


@pytest.mark.parametrize("module", _MODEL_FACING)
def test_no_model_facing_agent_states_its_own_temperature_fallback(module):
    """The relation, not the value: a literal here is a fifth pair waiting to happen."""
    source = (_AGENTS / module).read_text(encoding="utf-8")
    # Strip comments so the prose explaining the history does not trip the pattern.
    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())

    offenders = [m.group(0).strip() for m in _LITERAL_DEFAULT.finditer(code)]
    assert offenders == [], (
        f"{module} states a numeric temperature fallback: {offenders}. Import "
        f"DEFAULT_AIR_TEMP_C / DEFAULT_TRACK_TEMP_C from _shared_defaults instead."
    )


def test_the_pair_is_the_measured_median_not_a_round_number():
    """Sourced from the data, and the sourcing is the point.

    A rounded 25/35 looks tidier and is what one of the five pairs used; it is the
    TrackTemp mean rather than its median. The bound between "measured" and "chosen" is
    the only thing separating this constant from the four it replaced.

    Asserts the PROPERTY rather than a literal copy of the constants. It used to read
    `assert DEFAULT_AIR_TEMP_C == 24.2`, which is the constant compared against itself:
    it can only ever fail when someone edits the constant, including when they edit it
    correctly. Removing the 2023 Spanish GP duplicate legitimately moved the medians to
    24.6 / 34.7 and this test failed for that, which is a test asking to be rewritten.
    The measurement itself is guarded by the one below, against the dataset.
    """
    from src.agents._shared_defaults import DEFAULT_AIR_TEMP_C, DEFAULT_TRACK_TEMP_C

    assert (DEFAULT_AIR_TEMP_C, DEFAULT_TRACK_TEMP_C) != (25.0, 35.0), (
        "the tidy pair is the TrackTemp MEAN, not its median — the substitution this exists "
        "to prevent"
    )
    for name, value in (("air", DEFAULT_AIR_TEMP_C), ("track", DEFAULT_TRACK_TEMP_C)):
        assert value % 1 != 0, f"{name} default is a whole number: chosen, not measured"
        assert 10.0 < value < 60.0, f"{name} default is outside any plausible circuit reading"


@pytest.mark.data
@pytest.mark.skipif(
    not (ROOT / "data" / "processed" / "laps_featured_2025.parquet").exists(),
    reason="featured parquets absent",
)
def test_the_pair_still_matches_the_median_of_the_real_seasons():
    """Re-measure rather than re-read: the constant is a claim about the dataset.

    Still through `augment_featured_laps`, though the reason has changed: the artefacts now
    carry the weather columns natively, and the restore declines when they are present. It
    stays because a checkout whose artefacts predate that regeneration must measure the same
    thing, and because the restore is what any consumer gets either way.

    This assertion is what caught the 2023 Spanish GP duplicate moving the medians: the race
    was in the dataset twice, so 24.2 / 34.2 described a sample counting one weekend's
    weather double. De-duplicated, it is 24.6 / 34.7.
    """
    import pandas as pd

    from src.agents._shared_defaults import DEFAULT_AIR_TEMP_C, DEFAULT_TRACK_TEMP_C
    from src.f1_strat_manager.data_cache import get_data_root
    from src.f1_strat_manager.laps_augment import augment_featured_laps

    root = get_data_root() / "processed"
    frames = [
        augment_featured_laps(pd.read_parquet(root / f"laps_featured_{year}.parquet"), year)[
            ["AirTemp", "TrackTemp"]
        ]
        for year in (2023, 2024, 2025)
    ]
    laps = pd.concat(frames, ignore_index=True).apply(pd.to_numeric, errors="coerce").dropna()

    assert len(laps) > 0, "no weather rows survived: the medians below would mean nothing"
    assert laps["AirTemp"].median() == pytest.approx(DEFAULT_AIR_TEMP_C, abs=0.05)
    assert laps["TrackTemp"].median() == pytest.approx(DEFAULT_TRACK_TEMP_C, abs=0.05)
