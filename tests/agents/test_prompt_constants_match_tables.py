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


def _prompts() -> dict[str, str]:
    """Both rendered prompts, so every check below covers the pair rather than one."""
    from src.agents.pit_strategy_agent import _PIT_STRATEGY_SYSTEM_PROMPT

    return {
        "pit agent prompt": _PIT_STRATEGY_SYSTEM_PROMPT,
        "orchestrator prompt": _orchestrator_prompt(),
    }


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


# The four guard-rail bounds, each of which was prose in both prompts while the
# deterministic rails computed against the constant. Gate G3 listed them; #741 had
# derived exactly one number and left these.
_MIN_STINT = re.compile(r"\b(SOFT|MEDIUM|HARD)\b[^.\n]*?>=\s*(\d+)\s*lap")
_BEFORE_LAP = re.compile(r"before lap (\d+)")
# Anchored on "when", because a looser pattern also matches the COMPOUND capacity line
# ("recommend only if remaining laps <= 18") and then asserts the guard-rail bound
# equals a stint capacity. Caught by this very test on its first run.
_LAST_LAPS = re.compile(r"when remaining laps <=\s*(\d+)")
_CLIFF_P10 = re.compile(r"cliff P10 <\s*(\d+)|laps_to_cliff P10 <\s*(\d+)")


def test_no_prompt_states_a_minimum_stint_the_rails_disagree_with():
    """`_MIN_STINT_LAPS` sat two sections above the one number #741 derived.

    Same defect, same file, left in place because the fix was applied to the instance
    in front of me rather than to the class. That is the pattern three gates have now
    found, so this pins the whole table instead of one compound of it.
    """
    from src.strategy.inference.guard_rails import _MIN_STINT_LAPS

    for name, prompt in _prompts().items():
        stated = {m.group(1): int(m.group(2)) for m in _MIN_STINT.finditer(prompt)}
        assert stated, f"{name} states no minimum stint the pattern can see"
        for compound, value in stated.items():
            assert value == _MIN_STINT_LAPS[compound], (
                f"{name} says {compound} >= {value}, the rails enforce {_MIN_STINT_LAPS[compound]}"
            )


def test_no_prompt_states_a_pit_window_bound_the_rails_disagree_with():
    """The two hard bounds an LLM is told to treat as inviolable.

    If prose and rail disagree here the model is told to refuse an action the rails
    would have allowed, or to allow one they will override, and either way the
    disagreement is invisible until someone reads both files.
    """
    from src.strategy.inference.guard_rails import (
        _CLIFF_P10_SAFE,
        _NO_PIT_BEFORE_LAP,
        _NO_PIT_LAST_N_LAPS,
    )

    for name, prompt in _prompts().items():
        early = [int(m.group(1)) for m in _BEFORE_LAP.finditer(prompt)]
        late = [int(m.group(1)) for m in _LAST_LAPS.finditer(prompt)]
        cliff = [int(m.group(1) or m.group(2)) for m in _CLIFF_P10.finditer(prompt)]

        assert early, f"{name}: the early-window bound is no longer visible"
        assert late, f"{name}: the end-of-race bound is no longer visible"
        assert cliff, f"{name}: the cliff exception is no longer visible"

        assert set(early) == {_NO_PIT_BEFORE_LAP}, name
        assert set(late) == {_NO_PIT_LAST_N_LAPS}, name
        assert set(cliff) == {_CLIFF_P10_SAFE}, name


def test_the_undercut_threshold_is_not_restated_at_all():
    """This one cannot be interpolated, so it must not appear.

    The threshold is loaded per-instance from the model's calibration config and the
    tool PRINTS the live value into its own response. A module-level prompt cannot see
    it, so restating it means the LLM can receive two different thresholds in one
    conversation the moment the model is retuned. The prompt now points at what the
    tool reports instead.
    """
    from src.agents.pit_strategy_agent import _PIT_STRATEGY_SYSTEM_PROMPT

    assert "0.522" not in _PIT_STRATEGY_SYSTEM_PROMPT
    assert "score_undercut_tool reports" in _PIT_STRATEGY_SYSTEM_PROMPT
