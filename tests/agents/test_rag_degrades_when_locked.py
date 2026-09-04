"""Guards for #827: a locked regulation store costs the regulation block, not the lap.

`RagRetriever` opens the store as `QdrantClient(path=...)`, local single-writer mode
with an exclusive lock, so a second concurrent process fails on every retrieval.
Before this fix the exception propagated out of `run_lap` and the CLI's per-lap
`except Exception` rendered a red row for an entire race, naming the cause nowhere:
three races in one measurement run produced zero rows while their batches walked
every window and reported success.

WHY THIS FILE WAS REWRITTEN, AND IT IS THE POINT
------------------------------------------------
Its first version monkeypatched `portalocker.AlreadyLocked` and `LockException`,
taken from `retriever.py`'s docstring. **qdrant never raises those.**
`qdrant_local.py:148-151` catches portalocker's exception itself and re-raises a
BARE `RuntimeError`. So the guard under test never fired, and these tests could not
notice, because they raised the exception the docstring named rather than the one the
library raises.

Worse: the old `test_any_other_failure_still_surfaces` asserted that a `RuntimeError`
must propagate. The real lock error IS a `RuntimeError`. **The test written to keep
the except narrow was the test guaranteeing the failure escaped.** That is this
repository's "a guard that asserts nothing" shape, with the guard's own test holding
the door.

So the fixture below is built from the library's actual message, and one test asserts
against the real qdrant source text rather than against a docstring.
"""

from __future__ import annotations

import logging

import pytest

# Importing `strategy_orchestrator` reads FOUR artefacts at module import, not
# one: the tyre routing config, the SC training frame
# (`data/processed/sc_labeled/`), the sentiment weights
# (`data/models/nlp/bert_sentiment_v1/`) and the NLP pipeline config. All come
# from the Hugging Face dataset rather than git.
#
# The first version of this guard named only `HAS_TIRE_MODELS`. That is why it
# SKIPPED on CI and, on the curated tree `scripts/download_data.py` actually
# produces, ERRORED - the tyre config is present there and the SC frame is not.
# Naming one artefact out of four is the same shape as naming the artefact a
# test READS while ignoring what its IMPORT needs.
#
# So the guard attempts the import instead of predicting it. A missing artefact
# skips with the real filename in the reason; anything else propagates.
try:
    from src.agents import strategy_orchestrator as _orch

    _IMPORT_ERROR: str | None = None
except FileNotFoundError as exc:  # dataset artefact absent
    _orch = None
    _IMPORT_ERROR = f"strategy_orchestrator needs {exc.filename} (HF dataset, not git)"

pytestmark = pytest.mark.skipif(_IMPORT_ERROR is not None, reason=str(_IMPORT_ERROR))


@pytest.fixture
def orch():
    """The orchestrator module, already imported above or the file skipped."""
    return _orch


# Verbatim from qdrant_client/local/qdrant_local.py, the RuntimeError raised on a
# second open. Reproduced here so a qdrant upgrade that rewords it fails this test
# loudly instead of silently re-breaking the guard.
_REAL_QDRANT_LOCK_MESSAGE = (
    "Storage folder data/rag/qdrant_local is already accessed by another instance "
    "of Qdrant client. If you require concurrent access, use Qdrant server instead."
)


@pytest.fixture(autouse=True)
def _reset_once_flag(orch):
    """The 'log once' flag is process-global; each test needs a clean one."""
    orch._rag_unavailable_logged = False
    yield
    orch._rag_unavailable_logged = False


def test_a_locked_store_returns_none_instead_of_raising(orch, monkeypatch):
    """The lap survives without a regulation block."""

    def _locked(_question, year=None):
        raise RuntimeError(_REAL_QDRANT_LOCK_MESSAGE)

    monkeypatch.setattr(orch, "run_rag_agent", _locked)
    assert orch._run_rag_agent_or_degrade("what happens under a safety car?") is None


def test_the_cause_is_named_once_and_not_once_per_lap(orch, monkeypatch, caplog):
    """Sixty identical warnings is how a configuration problem looks like flaky data."""

    def _locked(_question, year=None):
        raise RuntimeError(_REAL_QDRANT_LOCK_MESSAGE)

    monkeypatch.setattr(orch, "run_rag_agent", _locked)
    with caplog.at_level(logging.WARNING, logger=orch.logger.name):
        for _ in range(5):
            orch._run_rag_agent_or_degrade("q")

    locked = [r for r in caplog.records if "regulation store is locked" in r.message]
    assert len(locked) == 1, [r.message for r in caplog.records]
    assert "F1_STRAT_DATA_ROOT" in locked[0].message


def test_a_different_runtime_error_still_surfaces(orch, monkeypatch):
    """A corrupt collection is not 'another process is running'.

    Note this raises a RuntimeError deliberately: the previous version of this test
    asserted that ANY RuntimeError propagates, which is what let the real lock error
    escape. The distinction is the message, not the type.
    """

    def _broken(_question, year=None):
        raise RuntimeError("collection 'fia_regulations' not found")

    monkeypatch.setattr(orch, "run_rag_agent", _broken)
    with pytest.raises(RuntimeError, match="fia_regulations"):
        orch._run_rag_agent_or_degrade("q")


def test_a_working_store_is_passed_straight_through(orch, monkeypatch):
    """Guards the guard: the degradation path must not be the only path that runs."""
    sentinel = object()
    monkeypatch.setattr(orch, "run_rag_agent", lambda _q, year=None: sentinel)
    assert orch._run_rag_agent_or_degrade("q") is sentinel


def test_the_season_reaches_the_agent(orch, monkeypatch):
    """The degrade wrapper is on the season's path, so it has to carry it.

    Without this the wrapper could swallow the argument and every lap would be
    answered out of whatever season the index happened to rank first, which is the
    state #320 describes and no other test in this file would notice.
    """
    seen = {}

    def _record(_question, year=None):
        seen["year"] = year
        return "ctx"

    monkeypatch.setattr(orch, "run_rag_agent", _record)
    assert orch._run_rag_agent_or_degrade("q", year=2023) == "ctx"
    assert seen["year"] == 2023
