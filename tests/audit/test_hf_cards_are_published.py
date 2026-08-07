"""The published Hugging Face cards must match the ones in this repo.

WHY THIS TEST READS THE HUB AND NOT THE DISK
--------------------------------------------
#825 shipped a data fix that reached nobody: `data/processed/**` is gitignored,
so merging the PR published nothing, and the guard written for it read the
LOCAL tree. It stayed green for a day while the Hub served three races another
race's radio corpus.

The cards have exactly that shape and had already drifted the same way. Until
2026-08-07 they lived in the landing-site repo, where nothing linked to them
and nothing pushed them, and the published cards still said "71 Grand Prix"
(a count retired when the duplicated 2023 Spanish GP was removed) and
"MAE 0.392 s" (which `metrics_registry.md` marks superseded).

So the assertion is deliberately against the artefact a user actually reads.
Reading a public card needs **no token**, which is the whole reason this is
possible for cards and was not for the parquets.

Publishing is `python scripts/upload_hf_cards.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent

# Kept in step with `scripts/upload_hf_cards.py::_CARDS` by the first test below,
# which fails if the two lists ever diverge - a duplicated mapping that drifts is
# how the card ended up unpublished in the first place.
_CARDS = (
    ("docs/huggingface/DATASET_CARD.md", "VforVitorio/f1-strategy-dataset", "dataset"),
    ("docs/huggingface/MODEL_CARD.md", "VforVitorio/f1-strategy-models", "model"),
)

# NOT marked `data`, and the distinction matters. That marker means "requires the
# HF dataset or model weights (skipped on CI runners)"; this needs neither - only
# an unauthenticated read of two public README files. Marking it `data` would
# have labelled it as one of the tests nobody expects to run, which is halfway to
# a guard that never fires, and this whole file exists because of one of those.
#
# Confirmed green on a CI runner, so the fetch really executes there.


def test_the_test_and_the_publisher_agree_on_which_cards_exist():
    """A guard listing different cards from the publisher would guard nothing.

    Reads the publisher's `_CARDS` by PARSING it rather than importing it: the
    script inserts the repo root on `sys.path` at module scope, and a test that
    executes a script for one constant inherits every side effect it has. Also
    per this repo's own lesson that grep is not an audit - the assertion is on
    the parsed literal, so a mention in a comment cannot satisfy it.
    """
    import ast

    source = (ROOT / "scripts" / "upload_hf_cards.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_CARDS"
    ]
    assert len(assignments) == 1, "upload_hf_cards._CARDS is no longer a single literal"

    published = tuple(
        tuple(a.value for a in call.args)
        for call in assignments[0].value.elts
        if isinstance(call, ast.Call)
    )
    assert published == _CARDS, (
        f"scripts/upload_hf_cards.py publishes {published}, this guard checks {_CARDS}"
    )


@pytest.mark.parametrize(("local", "repo_id", "repo_type"), _CARDS)
def test_the_published_card_matches_the_repo_copy(local, repo_id, repo_type):
    """Fails the moment a card is edited here and not pushed.

    Whitespace-insensitive at the edges only: the Hub round-trips a trailing
    newline inconsistently, and a test that fails on that would be turned off
    within a week, which is the same as not having it.
    """
    from huggingface_hub import hf_hub_download

    repo_copy = (ROOT / local).read_text(encoding="utf-8")
    downloaded = hf_hub_download(repo_id, "README.md", repo_type=repo_type)
    published = Path(downloaded).read_text(encoding="utf-8")

    assert published.strip() == repo_copy.strip(), (
        f"{repo_id} serves a card that differs from {local}. "
        f"Publish with: python scripts/upload_hf_cards.py"
    )
