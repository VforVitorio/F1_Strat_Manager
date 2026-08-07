"""Recognise the error qdrant raises when the on-disk store is already open.

WHY THIS IS ITS OWN MODULE, AND IT IS THE POINT
-----------------------------------------------
This lived inside ``strategy_orchestrator``, which imports the whole agent
stack, which reads four artefacts AT IMPORT TIME (the SC training frame, the
sentiment weights, the tyre routing config, the NLP pipeline config). So the
tests pinning this behaviour could not run anywhere that lacked the dataset:
they skipped on CI and, on the curated tree ``scripts/download_data.py``
actually produces, they ERRORED. A guard whose test cannot execute is not a
guard, and this file has now been through that lesson twice (#827).

Nothing here imports anything heavier than ``portalocker``, so the checks live
wherever Python does.

WHERE TO CHANGE IF QDRANT CHANGES
---------------------------------
:data:`QDRANT_LOCK_SIGNATURE` is matched against the message text because there
is nothing else to match on. ``qdrant_client/local/qdrant_local.py`` catches
portalocker's own ``LockException`` and re-raises a **bare ``RuntimeError``**,
so the exception that reaches a caller carries no distinguishing type,
attribute or code. ``tests/rag/test_store_lock.py`` pins the signature against
the installed library's source, so an upgrade that rewords the message fails
loudly instead of silently re-opening the hole.
"""

from __future__ import annotations

# The distinctive fragment of qdrant's own message. Verified against a real
# two-client collision, not against a docstring: the first version of this
# guard caught `portalocker.BaseLockException` because `retriever.py` said so,
# and that exception never reaches us.
QDRANT_LOCK_SIGNATURE = "already accessed by another instance"

try:  # portalocker ships with qdrant-client; tolerate its absence anyway
    from portalocker.exceptions import BaseLockException as _BaseLockException

    LOCK_EXCEPTIONS: tuple[type[BaseException], ...] = (_BaseLockException,)
except ImportError:  # pragma: no cover - portalocker is a qdrant dependency
    LOCK_EXCEPTIONS = ()


def is_store_locked(exc: BaseException) -> bool:
    """True when this exception means another process holds the store.

    Fails SAFE, which is the reason text matching is acceptable here: an
    unrecognised ``RuntimeError`` returns False and therefore propagates,
    rather than being swallowed as a lock. A corrupt collection and a missing
    embedding model are not "another process is running", and treating them as
    such would trade one silent failure for another.
    """
    if LOCK_EXCEPTIONS and isinstance(exc, LOCK_EXCEPTIONS):
        return True
    return isinstance(exc, RuntimeError) and QDRANT_LOCK_SIGNATURE in str(exc)
