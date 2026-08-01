"""Download the F1 StratLab dataset and model weights from Hugging Face Hub.

Dataset: https://huggingface.co/datasets/VforVitorio/f1-strategy-dataset

    python scripts/download_data.py

A thin front-end to :func:`src.f1_strat_manager.data_cache.ensure_setup`, which
is the same routine the CLI runs on first launch. This script used to call
``snapshot_download`` directly with no ``allow_patterns``, pulling the full
~31.7 GB repository rather than the curated ~7-8 GB a working install needs,
and with no error handling, so a dropped connection surfaced as a raw
``huggingface_hub`` traceback with nothing actionable in it.

Keeping one implementation matters more than the twenty lines it saves: the
pattern list, the offline escape hatch and the failure message all live in
``data_cache`` and would otherwise have had to be kept in step by hand.

--- WHERE TO CHANGE IF THE DATASET LAYOUT CHANGES ---
Nothing here. The pattern list is ``data_cache._build_allow_patterns`` and the
repo id is ``data_cache.HF_DATASET_REPO_ID``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``.git``-search-with-fallback, not a fixed ``.parent.parent``: the latter
# silently resolves to the wrong directory under a `uv tool install` layout,
# where this file is not exactly one level below the repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = next(
    (p for p in [_SCRIPT_DIR, *_SCRIPT_DIR.parents] if (p / ".git").exists()),
    _SCRIPT_DIR.parent,
)
sys.path.insert(0, str(_REPO_ROOT))

from src.f1_strat_manager.data_cache import ensure_setup, get_data_root  # noqa: E402


def main() -> int:
    """Fetch the curated dataset, reporting failure the way the CLI does."""
    try:
        # skip_if_offline=False: someone running this script explicitly asked
        # for a download, so quietly doing nothing under $F1_STRAT_OFFLINE
        # would look like success.
        ensure_setup(skip_if_offline=False)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Done. Data cached under {get_data_root()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
