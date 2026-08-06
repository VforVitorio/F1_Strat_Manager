"""Guards for #827: a locked regulation store costs the regulation block, not the lap.

`RagRetriever` opens the store as `QdrantClient(path=...)`, which is local
single-writer mode with an exclusive lock. A second concurrent process therefore
raises on every retrieval. Before this fix that exception propagated out of
`run_lap` and the CLI's per-lap `except Exception` rendered it as a red row, for
an entire race, with the cause named nowhere: three races in one measurement run
produced zero rows while their batches walked every window and reported success.

The direction of the degradation is the point. N30 is an enrichment, one of
fourteen recommendation fields, and the orchestrator already has a path for it not
being routed. An unavailable store should cost that block and nothing else.
"""

from __future__ import annotations

import logging

import pytest
from portalocker.exceptions import AlreadyLocked, LockException

from src.agents import strategy_orchestrator as orch


@pytest.fixture(autouse=True)
def _reset_once_flag():
    """The 'log once' flag is process-global; each test needs a clean one."""
    orch._rag_unavailable_logged = False
    yield
    orch._rag_unavailable_logged = False


def test_a_locked_store_returns_none_instead_of_raising(monkeypatch):
    """The whole point: the lap survives without a regulation block."""

    def _locked(_question):
        raise AlreadyLocked("another process holds data/rag/qdrant_local/.lock")

    monkeypatch.setattr(orch, "run_rag_agent", _locked)
    assert orch._run_rag_agent_or_degrade("what happens under a safety car?") is None


def test_the_cause_is_named_once_and_not_once_per_lap(monkeypatch, caplog):
    """Sixty identical warnings is how a configuration problem looks like flaky data."""

    def _locked(_question):
        raise LockException("locked")

    monkeypatch.setattr(orch, "run_rag_agent", _locked)
    with caplog.at_level(logging.WARNING, logger=orch.logger.name):
        for _ in range(5):
            orch._run_rag_agent_or_degrade("q")

    locked_warnings = [r for r in caplog.records if "regulation store is locked" in r.message]
    assert len(locked_warnings) == 1, [r.message for r in caplog.records]
    assert "F1_STRAT_DATA_ROOT" in locked_warnings[0].message


def test_any_other_failure_still_surfaces(monkeypatch):
    """Only the lock family is swallowed.

    A corrupt collection, a missing embedding model or a malformed question are
    not "another process is running", and catching them here would trade one
    silent failure for another. This is the assertion that keeps the except
    narrow: widening it to `Exception` makes this test fail.
    """

    def _broken(_question):
        raise RuntimeError("collection 'fia_regulations' not found")

    monkeypatch.setattr(orch, "run_rag_agent", _broken)
    with pytest.raises(RuntimeError, match="fia_regulations"):
        orch._run_rag_agent_or_degrade("q")


def test_a_working_store_is_passed_straight_through(monkeypatch):
    """Guards the guard: the degradation path must not be the only path that runs."""
    sentinel = object()
    monkeypatch.setattr(orch, "run_rag_agent", lambda _q: sentinel)
    assert orch._run_rag_agent_or_degrade("q") is sentinel
