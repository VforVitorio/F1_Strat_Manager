"""The half of #827's guard that needs no dataset, so CI actually runs it.

These four tests import `src.rag.store_lock`, which pulls in nothing heavier
than portalocker. That is the whole reason the module exists: the same
assertions used to live behind `strategy_orchestrator`, whose import reads four
artefacts from the Hugging Face dataset, so on CI they SKIPPED and on the
curated tree `scripts/download_data.py` produces they ERRORED.

`test_the_signature_matches_the_installed_qdrant_source` is the one that most
needed rescuing: its entire job is to fail when a qdrant upgrade rewords the
message this guard matches on, and a pin that never executes cannot do that.

The behavioural half — that a locked store degrades the lap instead of killing
it, and that the cause is logged once — genuinely needs the orchestrator, and
lives in `tests/agents/test_rag_degrades_when_locked.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.store_lock import QDRANT_LOCK_SIGNATURE, is_store_locked

# Verbatim from qdrant_client/local/qdrant_local.py, the RuntimeError raised on
# a second open of the same directory. Reproduced here so a reworded message
# fails the source pin below rather than silently passing these.
_REAL_QDRANT_LOCK_MESSAGE = (
    "Storage folder data/rag/qdrant_local is already accessed by another instance "
    "of Qdrant client. If you require concurrent access, use Qdrant server instead."
)


def test_the_real_qdrant_lock_error_is_recognised():
    """The whole defect: the first guard did not recognise this exception at all."""
    assert is_store_locked(RuntimeError(_REAL_QDRANT_LOCK_MESSAGE)) is True


def test_an_unrelated_runtime_error_is_not_mistaken_for_a_lock():
    """Failing safe is why text matching is acceptable here.

    Catching bare `RuntimeError` and swallowing it would trade one silent
    failure for another, so only the lock signature degrades.
    """
    assert is_store_locked(RuntimeError("collection 'fia_regulations' not found")) is False
    assert is_store_locked(ValueError("bad question")) is False


def test_the_signature_matches_the_installed_qdrant_source():
    """Pins the assumption to the LIBRARY, not to a docstring.

    This is the assertion the first version of #827's guard needed and did not
    have: it was written from `retriever.py`'s prose, which named a portalocker
    exception qdrant never raises.
    """
    import qdrant_client.local.qdrant_local as ql

    source = Path(ql.__file__).read_text(encoding="utf-8")
    assert QDRANT_LOCK_SIGNATURE in source, (
        "qdrant no longer raises the message this guard matches on; #827 is re-broken"
    )


def test_a_real_two_client_collision_is_recognised(tmp_path):
    """The check no amount of monkeypatching can substitute for.

    Opens the store twice for real and asserts the guard recognises whatever
    comes back. Skips only if a second open somehow succeeds, which would mean
    qdrant stopped holding an exclusive lock and #827 no longer exists.
    """
    from qdrant_client import QdrantClient

    store = tmp_path / "qdrant_local"
    first = QdrantClient(path=str(store))
    try:
        with pytest.raises(BaseException) as caught:  # noqa: B017 - the TYPE is the finding
            QdrantClient(path=str(store))
    finally:
        first.close()

    assert is_store_locked(caught.value), (
        f"a real collision raised {type(caught.value).__name__}, which the guard does "
        f"not recognise: {caught.value}"
    )
