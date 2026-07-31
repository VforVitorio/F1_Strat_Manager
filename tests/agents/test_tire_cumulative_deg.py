"""#727 — the TCN's own prediction must reach TireOutput instead of stopping at the parser.

``predict_tire_deg_tool`` has always run the TCN and printed its scalar:

    Cumulative degradation: -1.200 s | Degradation rate: -0.05 s/lap

``_parse_tool_outputs`` had regexes for the rate and the three cliff quantiles and
**none for the first half of that line**, and ``TireOutput`` had no field to put it
in. So the number the whole N07-N10 model family exists to produce reached nothing:
not the Monte Carlo, not the orchestrator prompt, not any UI.

WHY IT WAS THE WRONG FIELD THAT GOT WIRED
-----------------------------------------
``deg_rate`` looks like the tyre signal and is not one. It is the last row of the
RAW lap-to-lap derivative, uncorrected for fuel, and fuel burn-off drives lap times
down at roughly the rate wear drives them up. Measured over 110 real laps: median
+0.006 s/lap, negative on 43 of them, correlation with tyre life +0.115, and a
median-by-band that is not even monotonic.

The discarded scalar is fuel-corrected (N04's ``FuelAdjustedDegAbsolute``),
correlates +0.369 with tyre life, and swings 0.411 s/lap across a stint.

THE PARSER TESTS RUN WITHOUT MODEL WEIGHTS, AND THAT TOOK A SECOND ATTEMPT
--------------------------------------------------------------------------
The parser is pure, so its tests were written ungated — and CI failed, because
**importing** it was not pure: ``tire_agent`` builds ``TireAgentConfig()`` at module
scope, which reads ``data/models/tire_degradation/routing_config.json``, a file that
comes from Hugging Face and is not in git. That is precisely why
``tests/audit/test_tire_agent_hardening.py`` skips its **entire module** and takes its
own pure-parser test down with it.

The fix is the one ``src/strategy/inference/guard_rails.py`` already made for the pit
bounds: the parser moved to ``src/agents/tire_parsing.py``, a leaf module with nothing
but ``re``. Now the ungated tests genuinely run on a bare runner. Anything touching
``TireOutput`` itself still has to be gated, since the dataclass lives behind that
import.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
RACE_DIR = ROOT / "data" / "raw" / "2025" / "Lusail"
FEATURED = ROOT / "data" / "processed" / "laps_featured_2025.parquet"
_HAS_MODELS = (ROOT / "data" / "models" / "tire_degradation" / "routing_config.json").exists()
_HAS_DATA = (RACE_DIR / "laps.parquet").exists() and FEATURED.exists()


def _scoped_lusail_lap_30():
    """A real lap_state plus the laps frame **scoped to this GP**, as engine.py feeds it.

    The scoping is not incidental. ``engine.py`` narrows the season-wide featured
    parquet through ``_scope_laps_to_gp`` before any agent sees it (#429), because
    the agents' lookups filter by Driver and LapNumber and never by GP. Handing
    ``_tire_no_llm`` the unscoped frame is a configuration nothing runs, and it does
    not merely add noise: measured here it returned a cumulative degradation of
    **-33.840 s/lap**, against a real range of roughly -0.9 to +0.5. A test written
    against that would be green and meaningless.
    """
    from src.f1_strat_manager.laps_augment import augment_featured_laps
    from src.simulation.replay_engine import RaceReplayEngine
    from src.strategy.inference.engine import _scope_laps_to_gp

    replay = RaceReplayEngine(RACE_DIR, driver_code="NOR", team="McLaren", interval_seconds=0.0)
    lap_state = next(islice(replay.replay(), 29, 30))  # 0-indexed 29 -> lap 30
    laps = augment_featured_laps(pd.read_parquet(FEATURED), 2025)
    return lap_state, _scope_laps_to_gp(laps, lap_state)


class _FakeToolMessage:
    """Minimal stand-in for a LangChain ToolMessage — only ``.content`` is read."""

    def __init__(self, content: str) -> None:
        self.content = content


# ---------------------------------------------------------------------------
# The parser — no model weights needed, so CI actually runs these
# ---------------------------------------------------------------------------


def test_the_parser_module_stays_a_leaf():
    """Guards the whole point of the split: importing it must stay cheap.

    A subprocess, not ``sys.modules`` in-process, because by the time this test runs
    the suite has already imported torch for other reasons and the check would pass
    on someone else's import. Same technique as
    ``tests/eval/test_decision_modes.py``'s eager-import guard.

    If this fails, the parser has grown a dependency and CI has silently lost its
    only ungated coverage of the field extraction — which is the exact state #727
    found it in.
    """
    import subprocess
    import sys

    probe = (
        "import sys; import src.agents.tire_parsing; "
        "heavy = [m for m in ('torch', 'pandas', 'numpy') if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=ROOT, check=True
    )

    assert out.stdout.strip() == "", f"tire_parsing pulled in heavy deps: {out.stdout.strip()}"


@pytest.mark.parametrize("printed", ["-1.200", "0.000", "2.345"])
def test_the_parser_captures_the_cumulative_prediction(printed):
    """Including a negative one: a set faster than its fresh baseline is real early."""
    from src.agents.tire_parsing import parse_tool_outputs

    message = _FakeToolMessage(
        "Driver NOR | Compound C2 | TyreLife 12\n"
        f"Cumulative degradation: {printed} s | Degradation rate: -0.05 s/lap"
    )

    assert parse_tool_outputs([message])["cum_deg"] == float(printed)


def test_an_absent_line_writes_no_key_at_all():
    """The parser's contract: a key exists only when its regex matched.

    ``estimate_laps_to_cliff_tool`` prints the quantiles and the rate but not the
    cumulative prediction, so the cliff tool alone must not manufacture one.
    """
    from src.agents.tire_parsing import parse_tool_outputs

    message = _FakeToolMessage(
        "Laps to cliff — P10: 3.0 | P50: 5.0 | P90: 7.0\n"
        "Degradation rate: 0.0400 s/lap | MC std: 0.1 s | Calibrated sigma: 0.2 s"
    )

    assert "cum_deg" not in parse_tool_outputs([message])


@pytest.mark.skipif(
    not _HAS_MODELS,
    reason="TireOutput lives in tire_agent, whose import builds TireAgentConfig() and "
    "reads data/models/tire_degradation/ (HF, not git)",
)
def test_the_field_defaults_to_none_and_never_to_zero():
    """0.0 is a real reading here — a set at its fresh baseline — so it cannot double
    as the sentinel. ``deg_rate`` already shows the collision this avoids: 12 of 110
    measured laps carry a parse miss indistinguishable from a genuine zero.
    """
    from src.agents.tire_agent import TireOutput

    stub = TireOutput(
        compound="C2",
        current_tyre_life=10,
        deg_rate=0.03,
        laps_to_cliff_p10=20.0,
        laps_to_cliff_p50=30.0,
        laps_to_cliff_p90=40.0,
    )

    assert stub.cumulative_deg_s is None


# ---------------------------------------------------------------------------
# The production seam — needs the weights the prediction comes from
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_HAS_MODELS and _HAS_DATA),
    reason="needs data/models/tire_degradation/ and data/{raw/2025/Lusail,processed} (HF, not git)",
)
def test_a_real_lap_carries_the_prediction_end_to_end():
    """Through ``_tire_no_llm`` — the path ``engine.py`` runs — not a re-implementation.

    Asserts a plausible range rather than a value: pinning a number here would break
    on any legitimate model refresh while saying nothing about whether the field is
    connected. The band is wide on purpose; what is being tested is that a real
    number arrives at all, where ``None`` arrived before.

    It is not so wide that it passes for the wrong reason, though — the unscoped
    frame this fixture exists to avoid returns -33.840, well outside it.
    """
    from src.strategy.inference.no_llm import _tire_no_llm

    lap_state, laps = _scoped_lusail_lap_30()

    tire_out = _tire_no_llm(lap_state, laps)

    assert tire_out.cumulative_deg_s is not None
    assert -5.0 < tire_out.cumulative_deg_s < 5.0


@pytest.mark.skipif(
    not (_HAS_MODELS and _HAS_DATA),
    reason="needs data/models/tire_degradation/ and data/{raw/2025/Lusail,processed} (HF, not git)",
)
def test_the_prompt_shows_the_wear_and_says_unknown_when_it_is_missing():
    """A missing prediction must not print as 0.000, which reads as a pristine tyre."""
    from dataclasses import replace

    from src.agents.strategy_orchestrator import _build_tire_block
    from src.strategy.inference.no_llm import _tire_no_llm

    lap_state, laps = _scoped_lusail_lap_30()
    tire_out = _tire_no_llm(lap_state, laps)

    assert "vs fresh" in _build_tire_block(tire_out)
    assert "wear=unknown" in _build_tire_block(replace(tire_out, cumulative_deg_s=None))


# ===========================================================================
# #744b — the level becomes a COST by subtracting the model's own fresh reading
# ===========================================================================
#
# `cumulative_deg_s` is measured against N04's per-stint baseline, and those
# baselines are not comparable between stints: over 2,343 training-season stints
# the per-stint median has std 5.48 s and a 1st percentile of -32.87 s, against
# the 0.411 s/lap signal being chased. So the level cannot be charged as a cost,
# and a pooled per-compound table cannot fix it either — measured at Spearman
# +0.188 against +0.191 for having no reference at all.
#
# Subtracting the model's own prediction on the same stint's early laps cancels
# that baseline algebraically. Measured: 73.9% non-negative, Spearman +0.308.


def test_the_parser_captures_the_fresh_reference_including_a_negative_one():
    """The reference is itself measured against N04's slow baseline, so it is
    routinely negative — the same reason the other patterns carry `-?`."""
    from src.agents.tire_parsing import parse_tool_outputs

    message = _FakeToolMessage(
        "Driver NOR | Compound C2 | TyreLife 18\n"
        "Cumulative degradation: 0.412 s | Degradation rate: 0.02 s/lap"
        " | Fresh reference: -0.308 s"
    )

    parsed = parse_tool_outputs([message])

    assert parsed["fresh_ref"] == -0.308
    assert parsed["cum_deg"] == 0.412


def test_no_reference_line_writes_no_reference_key():
    """A replay starting mid-stint has no laps at the reference tyre life, so the
    tool prints no reference at all. That must stay distinguishable from one that
    happens to be zero."""
    from src.agents.tire_parsing import parse_tool_outputs

    message = _FakeToolMessage(
        "Driver NOR | Compound C2 | TyreLife 18\n"
        "Cumulative degradation: 0.412 s | Degradation rate: 0.02 s/lap"
    )

    assert "fresh_ref" not in parse_tool_outputs([message])


@pytest.mark.skipif(
    not _HAS_MODELS,
    reason="_referenced_wear reads the measured bounds off TireAgentConfig, whose "
    "construction needs data/models/tire_degradation/ (HF, not git)",
)
class TestReferencedWear:
    """The arithmetic that turns two model readings into one charge."""

    def test_a_set_at_its_own_fresh_pace_costs_exactly_zero(self):
        """The reference IS the current prediction on a stint's first laps, so the
        wear is 0.0 — correct, and pinned so nobody "fixes" it into a fallback."""
        from src.agents.tire_agent import _referenced_wear

        assert _referenced_wear({"cum_deg": -0.308, "fresh_ref": -0.308}) == 0.0

    def test_a_missing_half_is_none_and_never_zero(self):
        """Zero is a legitimate reading of this field — see the test above — so it
        cannot double as the sentinel. This is the collision #436 fixed for the
        cliff, where a parse miss became a confident call to box."""
        from src.agents.tire_agent import _referenced_wear

        assert _referenced_wear({"cum_deg": 0.412}) is None
        assert _referenced_wear({"fresh_ref": -0.308}) is None
        assert _referenced_wear({}) is None

    def test_the_baseline_cancels_rather_than_being_approximated(self):
        """Both readings carry the same per-stint offset, so it subtracts out
        exactly. Shift both by a 30 s Safety-Car baseline and the cost is unmoved."""
        from src.agents.tire_agent import _referenced_wear

        normal = _referenced_wear({"cum_deg": 0.412, "fresh_ref": -0.308})
        shifted = _referenced_wear({"cum_deg": -29.588, "fresh_ref": -30.308})

        assert normal == shifted == 0.72

    def test_the_tail_is_bounded_by_the_measured_percentiles(self):
        """The raw quantity reaches +-15 s/lap because a few stints have an out-lap
        or a Safety Car as their N04 baseline, and one of those reaching the scorer
        prices a single lap like ten positions."""
        from src.agents.tire_agent import CFG, _referenced_wear

        assert _referenced_wear({"cum_deg": 60.0, "fresh_ref": 0.0}) == CFG.deg_cost_ceiling_s
        assert _referenced_wear({"cum_deg": -60.0, "fresh_ref": 0.0}) == CFG.deg_cost_floor_s

    def test_the_bound_does_not_clip_a_genuinely_worn_tyre(self):
        """The measured median at 20-25 laps of tyre life is 1.03 s/lap, which is
        already above CLIFF_LOSS. A bound set there would delete the signal instead
        of the outliers, so this pins that the real range survives."""
        from src.agents.tire_agent import _referenced_wear

        assert _referenced_wear({"cum_deg": 1.03, "fresh_ref": 0.0}) == 1.03
        assert _referenced_wear({"cum_deg": 2.5, "fresh_ref": 0.0}) == 2.5


@pytest.mark.skipif(
    not (_HAS_MODELS and _HAS_DATA),
    reason="needs data/models/tire_degradation/ and data/{raw/2025/Lusail,processed} (HF, not git)",
)
def test_a_real_lap_carries_a_cost_that_the_raw_level_could_not_have_given():
    """The whole point of #744b, on a real lap rather than an arithmetic fixture.

    Lusail 2025 lap 30, five laps into the stint. The raw level reads NEGATIVE
    because N04 measures it against this stint's own out-lap; charged directly it
    would have PAID the car for staying out, which is why the level was never
    consumable. The referenced cost is a small positive, which is what a nearly
    fresh set should cost.

    Ranges, not values: pinning a number here would break on any legitimate model
    refresh while saying nothing about whether the reference is connected. What is
    pinned is the SIGN RELATIONSHIP the fix exists to create.
    """
    from src.strategy.inference.no_llm import _tire_no_llm

    lap_state, laps = _scoped_lusail_lap_30()

    tire_out = _tire_no_llm(lap_state, laps)

    assert tire_out.cumulative_deg_s is not None
    assert tire_out.deg_cost_s is not None
    # The level is measured against a slow baseline, so it is below the cost.
    assert tire_out.cumulative_deg_s < tire_out.deg_cost_s
    # A five-lap-old set costs something small and non-negative, not a credit.
    assert 0.0 <= tire_out.deg_cost_s < 1.0


@pytest.mark.skipif(
    not (_HAS_MODELS and _HAS_DATA),
    reason="needs data/models/tire_degradation/ and data/{raw/2025/Lusail,processed} (HF, not git)",
)
def test_the_reference_is_taken_from_the_stints_early_laps_and_not_from_all_of_them():
    """Gate G2's third survivor, and the nastiest, because it disables the feature.

    Widen `fresh_reference_tyre_life` past the current tyre life and the reference
    tensor becomes the prediction tensor: `deg_cost_s` reads **exactly 0.0 on every
    lap**, so both scorers charge nothing for the old set AND grant no fresh credit.
    The channel is not degraded to the fallback, it is switched off — strictly worse
    than before #744b — and 232 tests stayed green.

    The real-lap test above cannot see it: its band is `0.0 <= deg_cost_s`, which the
    degenerate value satisfies exactly at the boundary. That is this project's
    catalogued "assertion passes for the wrong reason near a boundary" defect, so the
    semantic gets its own assertion rather than a tighter band on that one.

    Exact zero is the mutant's signature. It is not a value this fixture can produce
    legitimately: the guard below pins that the car is past the reference tyre life, so
    the reference prefix is strictly shorter than the prediction's and the two readings
    come from different tensors.
    """
    from src.agents.tire_agent import CFG
    from src.strategy.inference.no_llm import _tire_no_llm

    lap_state, laps = _scoped_lusail_lap_30()
    tire_out = _tire_no_llm(lap_state, laps)

    assert tire_out.current_tyre_life > CFG.fresh_reference_tyre_life, (
        "fixture no longer exercises the case: pick a lap deeper into the stint"
    )
    assert tire_out.deg_cost_s is not None
    assert tire_out.deg_cost_s != 0.0
