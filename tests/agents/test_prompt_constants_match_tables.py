"""#741 — a constant restated in prose is a constant that will drift.

SOFT's stint capacity was **18** in `_STINT_CAPACITY_LAPS`, which the deterministic
selector computes against, and **15** in both LLM prompts, which restated it as a
literal. On `laps_remaining` 16-18 the selector passed SOFT and both prompts told the
model to refuse it, so the compound choice on that three-lap band was decided by prose.

Neither number was wrong. They were written from different sources and nothing made
them move together, which is the same shape as the threshold-scale defect `CLAUDE.md`
§11 records for 2026-07-27: a number restated somewhere it is not derived.

WHY THIS TEST IS PHRASED OVER THE WHOLE TABLE
----------------------------------------------
Pinning "SOFT reads 18" would pass the day someone edits the table and forgets a
prompt, which is exactly how the divergence arose. What must hold is the *relation*:
no prompt may state a capacity the table disagrees with. So this parses the rendered
prompts and checks every compound it can find, and it keeps holding when the measured
capacity changes.

⚠️ WHAT THIS DOES **NOT** CLAIM
-------------------------------
That 18 is the right number. Measured over 341 real green-flag SOFT stints across 71
races (`src/strategy/eval/stint_lengths.py`, the same instrument and data used for the
minimum bound), **33.7% run longer than 18 laps** and 45.7% longer than 15 — the median
is 15 and the maximum is 50. Real stint length reflects strategy, Safety Car luck and
circuit degradation, not a physical ceiling, so the data does not hand back a clean
maximum and none was invented. 18 was kept because it is what the selector already
computes against and the closer of the two to real practice. Whether the capacity model
should be a hard bound at all is a separate question and is not settled here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
_HAS_MODELS = (ROOT / "data" / "models" / "tire_degradation" / "routing_config.json").is_file()

pytestmark = pytest.mark.skipif(
    not _HAS_MODELS,
    reason="the prompts live in modules whose import builds agent configs from "
    "data/models/ (HF, not git)",
)

# Two phrasings, because the prompts use both and the first version of this file only
# read the first. Gate G3 executed the gap: mutating MEDIUM 30 -> 32 left this test
# GREEN while both prompts still said "12-30", so the docstring's "every compound it
# can find" found exactly one. A regex is a claim about coverage and has to be tested
# like one.
_LE_BOUND = re.compile(r"\b(SOFT|MEDIUM|HARD)\b[^.\n]*?<=\s*(\d+)")
_RANGE_BOUND = re.compile(r"\b(SOFT|MEDIUM|HARD)\b[^.\n]*?\b\d+\s*-\s*(\d+)\b")


def _stated_bounds(prompt: str) -> set[tuple[str, int]]:
    """Every (compound, upper bound) pair either phrasing states."""
    pairs = set()
    for pattern in (_LE_BOUND, _RANGE_BOUND):
        pairs |= {(m.group(1), int(m.group(2))) for m in pattern.finditer(prompt)}
    return pairs


def _orchestrator_prompt() -> str:
    from src.agents.strategy_orchestrator import RaceState, _build_orchestrator_prompt

    race_state = RaceState(
        driver="NOR",
        lap=30,
        total_laps=57,
        position=4,
        compound="MEDIUM",
        tyre_life=12,
        gap_ahead_s=2.0,
        pace_delta_s=0.0,
        risk_tolerance=0.5,
        air_temp=25.0,
        track_temp=35.0,
    )
    return _build_orchestrator_prompt(race_state, {}, "STAY_OUT")


def test_no_prompt_states_a_capacity_the_table_disagrees_with():
    """The relation, not the value. Both prompts, every compound they mention."""
    from src.agents.pit_strategy_agent import _PIT_STRATEGY_SYSTEM_PROMPT, _STINT_CAPACITY_LAPS

    prompts = {
        "pit agent system prompt": _PIT_STRATEGY_SYSTEM_PROMPT,
        "orchestrator prompt": _orchestrator_prompt(),
    }

    for name, prompt in prompts.items():
        for compound, stated in _stated_bounds(prompt):
            expected = _STINT_CAPACITY_LAPS.get(compound)
            if expected is None:
                continue
            assert stated == expected, (
                f"{name} states {compound} <= {stated} while _STINT_CAPACITY_LAPS says "
                f"{expected}. Derive it from the table instead of restating it (#741)."
            )


def test_every_compound_the_prompts_bound_is_actually_seen_by_the_regex():
    """The guard above passes vacuously on whatever the regex fails to match.

    Not hypothetical, and not caught by inspection: gate G3 EXECUTED the gap by
    moving MEDIUM 30 -> 32 and watching this file stay green while both prompts still
    said "12-30". So this asserts coverage, compound by compound, rather than trusting
    the pattern — the same "asserting about the empty set" trap the SOFT-only version
    of this test was written to avoid and then fell into for the other two.
    """
    from src.agents.pit_strategy_agent import _PIT_STRATEGY_SYSTEM_PROMPT, _STINT_CAPACITY_LAPS

    for name, prompt in (
        ("pit agent", _PIT_STRATEGY_SYSTEM_PROMPT),
        ("orchestrator", _orchestrator_prompt()),
    ):
        seen = {compound for compound, _ in _stated_bounds(prompt)}
        for compound in ("SOFT", "MEDIUM"):
            assert compound in seen, (
                f"{name} prompt states a bound for {compound} that the pattern does not "
                f"match, so the check above cannot see it. Saw: {sorted(seen)}"
            )
            stated = dict(_stated_bounds(prompt))[compound]
            assert stated == _STINT_CAPACITY_LAPS[compound]


def test_the_selector_and_the_prompt_agree_across_the_band_that_used_to_split_them(monkeypatch):
    """laps_remaining 16-18: the selector passed SOFT, the prompts forbade it.

    Walks the band and asserts the two answer the same way at every lap in it, rather
    than checking the single value they now share — the bug was a three-lap window, and
    a test that only reads the constant would not have seen its width.
    """
    from src.agents.pit_strategy_agent import _STINT_CAPACITY_LAPS

    prompt_bound = dict(_stated_bounds(_orchestrator_prompt()))["SOFT"]

    for laps_remaining in range(14, 21):
        selector_allows = _STINT_CAPACITY_LAPS["SOFT"] >= laps_remaining
        prompt_allows = laps_remaining <= prompt_bound
        assert selector_allows == prompt_allows, laps_remaining
