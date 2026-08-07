"""Guards for the LLM-mode measurement tooling.

These run on a machine with no model weights and no API key on purpose: the
token meter and the scorer are the two pieces whose failure would be invisible
in the measurement itself. A meter that silently drops a call under-reports the
cost, and an under-reported cost is the direction nobody double-checks. A scorer
that diverges from ``decision_modes`` produces two numbers that get compared to
each other while measuring different things, which this repository has already
paid for once.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.strategy.eval import decision_modes, llm_decision, token_meter


class _Message:
    usage_metadata = {
        "input_tokens": 100,
        "output_tokens": 20,
        "input_token_details": {"cache_read": 64},
    }
    response_metadata = {"model_name": "gpt-4.1-mini"}


class _Generation:
    message = _Message()


class _WithLlmOutput:
    llm_output = {
        "model_name": "gpt-4.1-mini",
        "token_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 64},
        },
    }
    generations = [[_Generation()]]


class _WithoutLlmOutput:
    llm_output = None
    generations = [[_Generation()]]


class _Unattributable:
    llm_output = None
    generations = []


def test_meter_reads_both_usage_shapes() -> None:
    meter = token_meter.TokenMeter()
    meter.on_llm_end(_WithLlmOutput())
    meter.on_llm_end(_WithoutLlmOutput())

    total = meter.totals()
    assert total.calls == 2
    assert total.prompt_tokens == 200
    assert total.completion_tokens == 40
    assert total.cached_prompt_tokens == 128
    assert set(meter.by_model) == {"gpt-4.1-mini"}


def test_meter_counts_a_call_it_cannot_attribute_rather_than_dropping_it() -> None:
    """An invisible call is how a cost estimate comes out low and confident."""
    meter = token_meter.TokenMeter()
    meter.on_llm_end(_Unattributable())

    assert meter.totals().calls == 0
    assert meter.unattributed_calls == 1


def test_scorer_reuses_the_deterministic_tier_definitions() -> None:
    """Not a style check: the two tiers' numbers are compared to each other.

    If either definition is ever re-typed here instead of imported, the columns
    of the paired table stop being comparable and nothing else notices.
    """
    assert llm_decision.guard_rail_block is decision_modes.guard_rail_block
    assert llm_decision._pit_decision_lap is decision_modes._pit_decision_lap
    assert llm_decision.DecisionAgreement is decision_modes.DecisionAgreement
    assert llm_decision.DECISION_WINDOW_LAPS == decision_modes.DECISION_WINDOW_LAPS


def test_aggregate_matches_the_deterministic_aggregate_on_the_same_verdicts() -> None:
    verdicts = [
        decision_modes.StopVerdict(2025, "Budapest", "LEC", 19, 17, -2, "scored"),
        decision_modes.StopVerdict(2025, "Budapest", "LEC", 40, None, None, "no_call_in_window"),
        decision_modes.StopVerdict(2025, "Lusail", "PIA", 24, 24, 0, "scored"),
        decision_modes.StopVerdict(2025, "Lusail", "NOR", 25, None, None, "min_stint"),
    ]
    agreement = llm_decision.aggregate(verdicts)

    assert agreement.sample_size == 2
    assert agreement.eligible == 4
    assert agreement.no_call == 1
    assert agreement.guard_railed == 1
    assert agreement.exact == pytest.approx(0.5)
    assert agreement.mean_signed_error == pytest.approx(-1.0)
    assert np.array_equal(np.sort(agreement.offsets), np.array([-2.0, 0.0]))


def test_a_lap_the_stack_never_asked_about_is_declined_not_scored() -> None:
    """`no_call_in_window` and `no_boundary_in_window` are opposite findings."""
    declined = {lap: "STAY_OUT" for lap in range(14, 25)}
    assert decision_modes._pit_decision_lap(declined, 14, 24) is None

    committed = {lap: "PIT_NOW" for lap in range(14, 25)}
    assert decision_modes._pit_decision_lap(committed, 14, 24) is None

    transition = {**{lap: "STAY_OUT" for lap in range(14, 19)}, 19: "PIT_NOW"}
    assert decision_modes._pit_decision_lap(transition, 14, 24) == 19


def test_repeat_disagreement_separates_action_from_confidence_drift() -> None:
    rows = [
        {
            "race": "X",
            "driver": "D",
            "lap": 5,
            "recommendation": {"action": "STAY_OUT", "confidence": 0.90},
        },
        {
            "race": "X",
            "driver": "D",
            "lap": 5,
            "recommendation": {"action": "PIT_NOW", "confidence": 0.90},
        },
        {
            "race": "X",
            "driver": "D",
            "lap": 6,
            "recommendation": {"action": "STAY_OUT", "confidence": 0.90},
        },
        {
            "race": "X",
            "driver": "D",
            "lap": 6,
            "recommendation": {"action": "STAY_OUT", "confidence": 0.72},
        },
    ]
    spread = llm_decision.repeat_disagreement(rows)

    assert spread["laps_with_repeats"] == 2
    assert spread["laps_differing"]["action"] == 1
    assert spread["laps_differing"]["confidence"] == 1
    assert spread["share_differing"]["action"] == pytest.approx(0.5)


def test_raw_folder_map_covers_every_featured_name_that_differs() -> None:
    """The fourth GP keyspace. A miss here runs a race on another race's data."""
    for featured, folder in llm_decision._RAW_FOLDER.items():
        assert featured != folder, featured
        assert " " in featured or "_" not in featured


def test_a_paid_run_refuses_by_default_and_says_what_it_would_cost():
    """The money guard. It defaults to REFUSING, and that is deliberate.

    This harness emptied an OpenAI account mid-run once, and the failure was
    worse than the bill: the CLI's per-lap `except Exception` turned every
    subsequent 429 into a red row, so three races produced zero rows while their
    batches walked every window and reported success. An exhausted account does
    not stop a run, it silently produces a fake one.

    A refusal rather than an interactive prompt, because these runs are launched
    as background processes and a prompt there is an unanswered question that
    blocks forever.
    """
    import argparse
    import importlib

    measure = importlib.import_module("scripts.measure_llm_windows")
    windows = [{"race": "Lusail", "driver": "PIA", "low": 4, "high": 14}]

    paid = argparse.Namespace(no_llm=False, provider="openai", repeats=1, yes_spend=False)
    assert measure._confirm_spend(paid, windows) is False

    authorised = argparse.Namespace(no_llm=False, provider="openai", repeats=1, yes_spend=True)
    assert measure._confirm_spend(authorised, windows) is True

    # The free paths must never be gated, or the guard becomes the thing people
    # work around instead of the thing that protects them.
    offline = argparse.Namespace(no_llm=True, provider="openai", repeats=1, yes_spend=False)
    local = argparse.Namespace(no_llm=False, provider="lmstudio", repeats=1, yes_spend=False)
    assert measure._confirm_spend(offline, windows) is True
    assert measure._confirm_spend(local, windows) is True
