"""Every artefact an agent reads at CONSTRUCTION must be in the download patterns (#798).

`PitStrategyAgent.__init__` reads `undercut_clean.parquet` unconditionally, and neither
the dataset nor `_build_allow_patterns` carried it. So on a checkout built purely from
the published data the pit agent could not be constructed at all, and every surface that
reaches it raised FileNotFoundError.

It stayed invisible for two reasons worth naming, because both are how this class of gap
survives: every machine that had run the notebook already had the file, and the one
`f1-sim` run people reach for happens not to trigger the pit agent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
_AGENTS = ROOT / "src" / "agents"

# Artefacts read during __init__ rather than lazily, keyed by the agent that needs them.
# A miss here is not a degraded prediction, it is an agent that cannot be built.
_CONSTRUCTION_READS = {
    "pit_strategy_agent.py": "data/processed/undercut_labeled/undercut_clean.parquet",
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


def test_the_coverage_helper_actually_discriminates():
    """A matcher that says yes to everything would make the tests above vacuous."""
    patterns = ("data/processed/undercut_labeled/**", "data/models/lap_time/**")

    assert _covers(patterns, "data/processed/undercut_labeled/undercut_clean.parquet")
    assert not _covers(patterns, "data/processed/overtake_labeled/overtake.parquet")
    assert not _covers(patterns, "data/raw/2025/Lusail/laps.parquet")
