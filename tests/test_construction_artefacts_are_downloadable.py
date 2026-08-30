"""Every artefact an agent reads at CONSTRUCTION must be in the download patterns (#798).

`PitStrategyAgent.__init__` reads `undercut_clean.parquet` unconditionally, and neither
the dataset nor `_build_allow_patterns` carried it. So on a checkout built purely from
the published data the pit agent could not be constructed at all, and every surface that
reaches it raised FileNotFoundError.

It stayed invisible for two reasons worth naming, because both are how this class of gap
survives: every machine that had run the notebook already had the file, and the one
`f1-sim` run people reach for happens not to trigger the pit agent.

The EVAL tier is the same gap with a quieter failure and it is guarded here too (#1130,
#1146). A stage that cannot find its label set returns a `pending` row instead of
raising, so the report ships a measurement short and says nothing, and the golden test
that would have caught it skips with a reason naming the weights.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
_AGENTS = ROOT / "src" / "agents"

# Artefacts read during __init__ rather than lazily, keyed by the agent that needs them.
# A miss here is not a degraded prediction, it is an agent that cannot be built.
#
# The two IMPORT-time entries are worse than construction-time and were the twin this
# file missed for months (#837): they run under `import`, so the orchestrator cannot be
# built at all and the CLI, the arcade pipeline and every eval tier are unreachable on a
# clean install. One member of the trio got a comment in `data_cache.py` and the other
# two never got a pattern - the repo's dominant defect, inside the guard written for it.
#
# HOW TO FIND THE NEXT ONE: build a tree from `_DEFAULT_MODEL_PATTERNS` alone and run
# `python -c "from src.agents import strategy_orchestrator"`. Whatever it raises on
# belongs here. A grep will not find it; these are bare module-level reads.
_CONSTRUCTION_READS = {
    "pit_strategy_agent.py": "data/processed/undercut_labeled/undercut_clean.parquet",
    "race_situation_agent.py": "data/processed/sc_labeled/sc_labeled_2023_2025.parquet",
    "radio_agent.py": "data/models/nlp/bert_sentiment_v1/best_roberta_sentiment_model.pt",
}


# Label sets and configs the #304 eval stages score against, keyed by the file in
# `src/strategy/eval/` that reads them. A miss here does NOT raise: the stage returns a
# `pending` row and the golden test skips, so the report loses a measurement silently.
# That is why they need a guard of their own rather than being caught by a failing run.
#
# `ner_v1/model_config.json` sits one level ABOVE the weights, so the pattern that pulls
# `ner_v1/bert_bio_v1/**` misses it and every clean install lost NR-07 (#1146).
_EVAL_READS = {
    "nlp.py": (
        "data/processed/radio_nlp/intent_labeled_data.csv",
        "data/processed/radio_nlp/f1_radio_entity_annotations.json",
        "data/models/nlp/ner_v1/model_config.json",
    ),
}


def _patterns() -> tuple[str, ...]:
    from src.f1_strat_manager.data_cache import _DEFAULT_MODEL_PATTERNS

    return tuple(_DEFAULT_MODEL_PATTERNS)


def _covers(patterns, path: str) -> bool:
    """True when some glob pattern would pull `path` in a snapshot_download."""
    for pattern in patterns:
        regex = re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
        if re.fullmatch(regex, path):
            return True
    return False


@pytest.mark.parametrize(("module", "artefact"), sorted(_CONSTRUCTION_READS.items()))
def test_a_construction_time_artefact_is_pulled_by_the_downloader(module, artefact):
    """The download list must cover what construction reads, or a clean install cannot run."""
    patterns = _patterns()
    assert _covers(patterns, artefact), (
        f"{module} reads {artefact} while building the agent, but no pattern in "
        f"_DEFAULT_MODEL_PATTERNS pulls it. A clean install will raise FileNotFoundError "
        f"before the agent exists."
    )


@pytest.mark.parametrize(("module", "artefact"), sorted(_CONSTRUCTION_READS.items()))
def test_the_module_still_reads_the_artefact_this_test_claims(module, artefact):
    """Guard against the list going stale in the harmless-looking direction.

    If the read moves or is made lazy, the test above keeps passing while describing a
    dependency that no longer exists, and the next real construction-time read is added
    with nothing watching. So the claim is checked against the source too.
    """
    source = (_AGENTS / module).read_text(encoding="utf-8")
    filename = artefact.rsplit("/", 1)[-1]
    assert filename in source, (
        f"{module} no longer mentions {filename}: either the read moved and this entry "
        f"is stale, or the artefact list needs updating for whatever replaced it."
    )


@pytest.mark.parametrize(
    ("module", "artefact"),
    sorted((mod, art) for mod, arts in _EVAL_READS.items() for art in arts),
)
def test_an_eval_label_set_is_pulled_by_the_downloader(module, artefact):
    """The download list must cover what the eval stages read, or the report loses a row."""
    patterns = _patterns()
    assert _covers(patterns, artefact), (
        f"src/strategy/eval/{module} scores against {artefact}, but no pattern in "
        f"_DEFAULT_MODEL_PATTERNS pulls it. A clean install will report a `pending` row "
        f"instead of the measurement, and the golden test will skip rather than fail."
    )


@pytest.mark.parametrize(
    ("module", "artefact"),
    sorted((mod, art) for mod, arts in _EVAL_READS.items() for art in arts),
)
def test_the_eval_module_still_reads_the_artefact_this_test_claims(module, artefact):
    """Same staleness guard as the construction list above, for the same reason."""
    source = (ROOT / "src" / "strategy" / "eval" / module).read_text(encoding="utf-8")
    filename = artefact.rsplit("/", 1)[-1]
    stem = filename.rsplit(".", 1)[0]
    # `model_config.json` is built from a directory constant plus the bare filename, so
    # the path never appears whole in the source and the parent directory is the claim.
    needle = filename if filename != "model_config.json" else artefact.split("/")[-2]
    assert needle in source, (
        f"src/strategy/eval/{module} no longer mentions {needle} (for {stem}): either the "
        f"read moved and this entry is stale, or the list needs whatever replaced it."
    )


def test_the_coverage_helper_actually_discriminates():
    """A matcher that says yes to everything would make the tests above vacuous."""
    patterns = ("data/processed/undercut_labeled/**", "data/models/lap_time/**")

    assert _covers(patterns, "data/processed/undercut_labeled/undercut_clean.parquet")
    assert not _covers(patterns, "data/processed/overtake_labeled/overtake.parquet")
    assert not _covers(patterns, "data/raw/2025/Lusail/laps.parquet")
