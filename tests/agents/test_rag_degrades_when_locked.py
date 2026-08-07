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

from tests.conftest import HAS_TIRE_MODELS as _HAS_MODELS

# EVERY test here, including the ones that only exercise a pure predicate.
# Importing `strategy_orchestrator` pulls in `tire_agent`, which reads
# `data/models/tire_degradation/routing_config.json` AT MODULE IMPORT, and that
# tree comes from the Hugging Face Hub rather than git. So this file is not
# hermetic no matter what its bodies touch, and a module-level import goes red on
# a clean CI runner before a single test runs.
#
# `test_tire_serving_frame.py` documents this exact trap in its own header. I read
# that header this session and wrote the module-level import anyway, which is why
# the import now lives inside a fixture: naming the artefact the IMPORT needs,
# rather than the one the test body reads, is the guard shape #798 documented.
pytestmark = pytest.mark.skipif(
    not _HAS_MODELS,
    reason="importing strategy_orchestrator reads data/models/ (HF, not git)",
)


@pytest.fixture
def orch():
    """The orchestrator module, imported lazily so collection stays cheap."""
    from src.agents import strategy_orchestrator

    return strategy_orchestrator

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


def test_the_real_qdrant_lock_error_is_recognised(orch):
    """The whole defect: the previous guard did not recognise this exception at all."""
    assert orch._is_store_locked(RuntimeError(_REAL_QDRANT_LOCK_MESSAGE)) is True


def test_an_unrelated_runtime_error_is_not_mistaken_for_a_lock(orch):
    """Failing safe is the reason text matching is acceptable here.

    Catching bare `RuntimeError` and swallowing it would trade one silent failure
    for another. Only the lock signature degrades; everything else propagates.
    """
    assert orch._is_store_locked(RuntimeError("collection 'fia_regulations' not found")) is False
    assert orch._is_store_locked(ValueError("bad question")) is False


def test_a_locked_store_returns_none_instead_of_raising(orch, monkeypatch):
    """The lap survives without a regulation block."""

    def _locked(_question):
        raise RuntimeError(_REAL_QDRANT_LOCK_MESSAGE)

    monkeypatch.setattr(orch, "run_rag_agent", _locked)
    assert orch._run_rag_agent_or_degrade("what happens under a safety car?") is None


def test_the_cause_is_named_once_and_not_once_per_lap(orch, monkeypatch, caplog):
    """Sixty identical warnings is how a configuration problem looks like flaky data."""

    def _locked(_question):
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

    def _broken(_question):
        raise RuntimeError("collection 'fia_regulations' not found")

    monkeypatch.setattr(orch, "run_rag_agent", _broken)
    with pytest.raises(RuntimeError, match="fia_regulations"):
        orch._run_rag_agent_or_degrade("q")


def test_a_working_store_is_passed_straight_through(orch, monkeypatch):
    """Guards the guard: the degradation path must not be the only path that runs."""
    sentinel = object()
    monkeypatch.setattr(orch, "run_rag_agent", lambda _q: sentinel)
    assert orch._run_rag_agent_or_degrade("q") is sentinel


def test_the_signature_matches_the_installed_qdrant_source(orch):
    """Pins the text to the library, so an upgrade that rewords it fails here.

    This is the assertion the first version of this file needed and did not have:
    it tests the guard against what the INSTALLED library does, not against what a
    docstring says it does.
    """
    from pathlib import Path

    import qdrant_client.local.qdrant_local as ql

    source = Path(ql.__file__).read_text(encoding="utf-8")
    assert orch._QDRANT_LOCK_SIGNATURE in source, (
        "qdrant no longer raises the message this guard matches on; #827 is re-broken"
    )
