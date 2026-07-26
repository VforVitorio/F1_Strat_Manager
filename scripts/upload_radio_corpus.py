"""Upload the static OpenF1 radio corpus to the F1 StratLab dataset on HF Hub.

The corpus is built locally by ``scripts/build_radio_dataset.py`` and lives in
two parallel trees on disk:

* ``data/processed/race_radios/{year}/{slug}/{radios.parquet, rcm.parquet}``:
  small metadata parquets (~22 KB per GP, ~536 KB total) that the runner
  reads to enumerate the team-radio rows and the FIA race-control messages
  for a given lap.

* ``data/raw/radio_audio/{year}/{slug}/*.mp3``: the original OpenF1 MP3
  files that Whisper transcribes on the consumer side. ~3 MB per GP,
  ~82 MB total for the full 2025 calendar.

Both trees land under ``VforVitorio/f1-strategy-dataset`` preserving the
on-disk layout, so the existing :func:`src.f1_strat_manager.data_cache.
ensure_setup` and the new :func:`ensure_radio_corpus` helpers can pull them
back with the same path globs they already use for race parquets.

The upload is idempotent: ``HfApi.upload_folder`` deduplicates by content
hash so a partial run that gets interrupted can be re-executed safely and
only the missing files are sent. The script prints a per-tree summary
(file count + total bytes) before each upload and surfaces the final
dataset URL on success.

Authentication
--------------
Either set ``HF_TOKEN`` in the environment (preferred for one-off CI runs)
or run ``huggingface-cli login`` once and let the cached token take over.
The script raises a clear error when neither is available so the user does
not waste time scanning the upload progress for a silent auth failure.

Usage
-----
    # Full 2025 corpus, both trees, with progress bars::

        export HF_TOKEN=hf_xxx
        python scripts/upload_radio_corpus.py

    # Dry run: list what would be uploaded without touching the Hub::

        python scripts/upload_radio_corpus.py --dry-run

    # Upload only the parquets (e.g. after re-running build_radio_dataset
    # with a fresh slug fix but no audio changes)::

        python scripts/upload_radio_corpus.py --skip-audio

    # Upload only the MP3 tree (e.g. when the parquets are already pushed
    # and a partial transfer needs to resume)::

        python scripts/upload_radio_corpus.py --skip-parquets
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Repo root walker: same pattern used everywhere else in the codebase so the
# script keeps running when invoked from any sub-directory.
_HERE = Path(__file__).resolve()
_REPO_ROOT = next(
    (p for p in (_HERE, *_HERE.parents) if (p / ".git").exists()),
    _HERE.parent.parent,
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import the canonical dataset id from data_cache so the upload target stays
# in lockstep with the consumer side. Editing the constant in one place
# updates both upload and download.
from src.f1_strat_manager.data_cache import HF_DATASET_REPO_ID, get_data_root  # noqa: E402

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Filesystem inspection
# ──────────────────────────────────────────────────────────────────────────────


def _walk_size(root: Path) -> tuple[int, int]:
    """Return ``(file_count, total_bytes)`` for every file under ``root``.

    Walks recursively and ignores directories silently. Used by the dry-run
    summary so the user sees exactly how much data is about to leave the
    machine before the upload begins.
    """
    if not root.exists():
        return 0, 0
    n_files = 0
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            n_files += 1
            total += p.stat().st_size
    return n_files, total


def _fmt_bytes(n: int) -> str:
    """Format a byte count as a human-readable string with KB / MB / GB units.

    Kept local so the script has zero external dependencies beyond rich and
    huggingface_hub. Uses base-1024 because that is what every other tool in
    the project (du, snapshot_download progress bars) reports.
    """
    if n < 1024:
        return f"{n} B"
    units = ("KB", "MB", "GB", "TB")
    val = float(n) / 1024.0
    for u in units:
        if val < 1024.0 or u == units[-1]:
            return f"{val:.1f} {u}"
        val /= 1024.0
    return f"{val:.1f} TB"


# ──────────────────────────────────────────────────────────────────────────────
# Upload
# ──────────────────────────────────────────────────────────────────────────────


def _upload_tree(
    api,
    *,
    folder_path: Path,
    path_in_repo: str,
    commit_message: str,
    dry_run: bool,
) -> None:
    """Upload one local folder to the dataset preserving the relative layout.

    Wraps ``HfApi.upload_folder`` so the two trees (parquets + audio) can be
    pushed by a single call site. The function is intentionally chatty:
    this is a one-shot operation, not a hot path, so the user benefits from
    knowing exactly which folder is in flight and where it lands on the Hub.
    """
    n_files, total = _walk_size(folder_path)
    console.print(
        f"[grey50]→ {folder_path.relative_to(_REPO_ROOT)} "
        f"({n_files} files, {_fmt_bytes(total)}) "
        f"→ {HF_DATASET_REPO_ID}:{path_in_repo}[/grey50]"
    )
    if dry_run:
        console.print("[gold1]  dry-run — skipped[/gold1]")
        return
    if n_files == 0:
        console.print("[gold1]  empty tree — skipped[/gold1]")
        return
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )
    console.print(f"[green3]  uploaded {n_files} files[/green3]")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload the static OpenF1 radio corpus to the F1 StratLab dataset on HF Hub.",
    )
    p.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Season year to upload (default: 2025). Both trees are scoped to this year.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be uploaded without touching the Hub. Useful before the first real push.",
    )
    p.add_argument(
        "--skip-parquets",
        action="store_true",
        help="Skip the data/processed/race_radios tree (push only the MP3 audio).",
    )
    p.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip the data/raw/radio_audio tree (push only the parquet metadata).",
    )
    p.add_argument(
        "--commit-message",
        default=None,
        help="Override the auto-generated commit message stamped on the dataset revision.",
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    """Resolve folders, validate auth, and upload the requested trees in order.

    Parquets go first because they are tiny: if the auth handshake or the
    network is broken the user finds out in seconds instead of waiting for
    the 80 MB MP3 transfer to fail. The audio tree only kicks off after the
    parquet upload succeeds.
    """
    data_root = get_data_root()
    parquet_root = data_root / "processed" / "race_radios" / str(args.year)
    audio_root = data_root / "raw" / "radio_audio" / str(args.year)

    # ── Pre-flight summary ────────────────────────────────────────────────
    n_pq, sz_pq = _walk_size(parquet_root)
    n_au, sz_au = _walk_size(audio_root)

    grid = Table.grid(padding=(0, 2), expand=False)
    grid.add_column(style="grey50", justify="right", min_width=12)
    grid.add_column(justify="left")
    grid.add_row("Dataset", f"[bright_white]{HF_DATASET_REPO_ID}[/bright_white]")
    grid.add_row("Year", f"[bright_white]{args.year}[/bright_white]")
    grid.add_row(
        "Parquets",
        f"{n_pq} files · {_fmt_bytes(sz_pq)}  [grey50]({parquet_root.relative_to(_REPO_ROOT)})[/grey50]",
    )
    grid.add_row(
        "Audio MP3s",
        f"{n_au} files · {_fmt_bytes(sz_au)}  [grey50]({audio_root.relative_to(_REPO_ROOT)})[/grey50]",
    )
    grid.add_row(
        "Total", f"[bright_white]{n_pq + n_au} files · {_fmt_bytes(sz_pq + sz_au)}[/bright_white]"
    )
    grid.add_row(
        "Mode", "[gold1]DRY-RUN[/gold1]" if args.dry_run else "[green3]LIVE UPLOAD[/green3]"
    )

    console.print(
        Panel(
            grid,
            title="[bold gold1]F1 StratLab — radio corpus upload[/bold gold1]",
            title_align="center",
            border_style="gold1",
            padding=(1, 2),
            expand=False,
        )
    )

    if n_pq == 0 and n_au == 0:
        console.print(
            "[red3]Nothing to upload — both trees are empty. Run scripts/build_radio_dataset.py first.[/red3]"
        )
        sys.exit(1)

    # ── Auth check ────────────────────────────────────────────────────────
    # We import here so the dry-run path stays cheap and so missing the
    # huggingface_hub package surfaces with a clear error rather than at
    # module import time.
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        console.print(
            f"[red3]huggingface_hub is not installed: {exc}. "
            "Reinstall the project with `uv sync` or `pip install -e .`.[/red3]"
        )
        sys.exit(1)

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    if not args.dry_run:
        try:
            who = api.whoami()
            console.print(
                f"[grey50]auth ok — uploading as [bright_white]{who['name']}[/bright_white][/grey50]"
            )
        except Exception as exc:
            console.print(
                f"[red3]HF Hub auth failed: {exc}\n"
                "Set HF_TOKEN in the environment or run `huggingface-cli login` and retry.[/red3]"
            )
            sys.exit(1)

    # ── Upload ────────────────────────────────────────────────────────────
    commit_msg = args.commit_message or f"radio corpus upload — year {args.year}"

    if not args.skip_parquets:
        _upload_tree(
            api,
            folder_path=parquet_root,
            path_in_repo=f"data/processed/race_radios/{args.year}",
            commit_message=f"{commit_msg} (parquets)",
            dry_run=args.dry_run,
        )

    if not args.skip_audio:
        _upload_tree(
            api,
            folder_path=audio_root,
            path_in_repo=f"data/raw/radio_audio/{args.year}",
            commit_message=f"{commit_msg} (audio)",
            dry_run=args.dry_run,
        )

    if args.dry_run:
        console.print("\n[gold1]dry-run complete — no files were uploaded.[/gold1]")
    else:
        console.print(
            f"\n[green3]done — corpus available at "
            f"https://huggingface.co/datasets/{HF_DATASET_REPO_ID}[/green3]"
        )


def main() -> None:
    """Console entry point: parses argv and dispatches to :func:`run`."""
    run(_parse_args())


if __name__ == "__main__":
    main()
