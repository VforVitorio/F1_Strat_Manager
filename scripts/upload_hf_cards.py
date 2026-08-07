"""Publish the Hugging Face dataset and model cards from this repo.

WHY THIS SCRIPT EXISTS
----------------------
The cards used to live in `f1stratlab-web`, the landing-site repo, where nothing
linked to them and nothing published them. They were a mirror of nothing: the
site never rendered them, the Hub never received them, and by 2026-08-07 the
published cards still said "71 Grand Prix" and "MAE 0.392 s" - a race count
retired when the duplicated 2023 Spanish GP was removed, and a metric the
registry marks superseded.

They belong here because this is where their evidence is measured. A card and
the `documents/eval_reports/` run that justifies it now sit in one repo, so the
place to edit when a number moves is visible from the place the number moves.

WHY AN OPERATOR SCRIPT AND NOT CI
---------------------------------
Publishing needs a Hugging Face token with WRITE scope. Putting one in a public
repository's Actions secrets is a standing risk for something that runs a few
times a year, so this follows `scripts/upload_radio_corpus.py`: the operator
runs it with their own local token. Reading a public card needs no token at
all, which is what lets `tests/audit/test_hf_cards_are_published.py` catch a
card that was edited and never pushed.

    python scripts/upload_hf_cards.py --dry-run
    python scripts/upload_hf_cards.py

--- WHERE TO CHANGE IF THE HUB LAYOUT CHANGES ---
`_CARDS` below is the whole mapping. A Hub repo's card IS its root `README.md`;
there is no other filename that works.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = next(
    (p for p in [_SCRIPT_DIR, *_SCRIPT_DIR.parents] if (p / ".git").exists()),
    _SCRIPT_DIR.parent,
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass(frozen=True)
class Card:
    """One card and the Hub repo whose ``README.md`` it becomes."""

    local: str
    repo_id: str
    repo_type: str


# The complete mapping. Both Hub repos exist and are public.
_CARDS: tuple[Card, ...] = (
    Card("docs/huggingface/DATASET_CARD.md", "VforVitorio/f1-strategy-dataset", "dataset"),
    Card("docs/huggingface/MODEL_CARD.md", "VforVitorio/f1-strategy-models", "model"),
)


def _read(card: Card) -> str:
    """The card's text, or a clear failure naming the file that is missing."""
    path = _REPO_ROOT / card.local
    if not path.exists():
        raise FileNotFoundError(f"card not found: {path}")
    return path.read_text(encoding="utf-8")


def _published(card: Card) -> str | None:
    """What the Hub serves today, or None when it has no card yet.

    Read-only and unauthenticated, so this half works for anyone.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        path = hf_hub_download(card.repo_id, "README.md", repo_type=card.repo_type)
    except EntryNotFoundError:
        return None
    return Path(path).read_text(encoding="utf-8")


def _upload(card: Card, text: str, message: str) -> None:
    """Push the card as the Hub repo's README."""
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=card.repo_id,
        repo_type=card.repo_type,
        commit_message=message,
    )


def run(dry_run: bool, message: str) -> int:
    """Compare each card with the Hub and push the ones that differ.

    Returns the number of cards that needed publishing, so a caller can tell
    "nothing to do" from "two pushed" without parsing the output.
    """
    pending = 0
    for card in _CARDS:
        local = _read(card)
        live = _published(card)
        if live is not None and live.strip() == local.strip():
            print(f"  up to date   {card.repo_id}")
            continue
        pending += 1
        state = "no card on the Hub yet" if live is None else "differs from the Hub"
        print(f"  needs push   {card.repo_id}  ({state})")
        if not dry_run:
            _upload(card, local, message)
            print(f"  pushed       {card.repo_id}")
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which cards differ from the Hub without pushing anything.",
    )
    parser.add_argument(
        "--commit-message",
        default="docs(cards): sync from the F1 StratLab repo",
        help="Commit message stamped on the Hub revision.",
    )
    args = parser.parse_args()

    print(f"F1 StratLab - Hugging Face cards ({'DRY-RUN' if args.dry_run else 'LIVE'})")
    pending = run(args.dry_run, args.commit_message)

    if pending == 0:
        print("\nBoth cards already match the Hub.")
    elif args.dry_run:
        print(f"\n{pending} card(s) would be pushed. Re-run without --dry-run.")
    else:
        print(f"\n{pending} card(s) published.")


if __name__ == "__main__":
    main()
