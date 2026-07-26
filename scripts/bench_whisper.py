"""Whisper transcription latency benchmark: overall bucket only.

Loads the production :class:`WhisperTranscriber` (model ``turbo``) the
same way :mod:`src.nlp.radio_runner` does at simulation time, samples
50 mp3 files from ``data/raw/radio_audio/`` with a fixed seed, and
times :meth:`WhisperTranscriber.transcribe`. WER is intentionally
out-of-scope (no paired ground-truth corpus is available yet — see
the ``notes`` column in the artefact); the benchmark exists solely to
quote the per-clip transcription latency in chapter 5.

Usage::

    uv run scripts/bench_whisper.py [--n-runs 50] [--device auto|cpu|cuda]
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Repo-root path injection: must happen before any src.* import
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = next(
    (p for p in [_SCRIPT_DIR, *_SCRIPT_DIR.parents] if (p / ".git").exists()),
    _SCRIPT_DIR.parent,
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.getLogger("whisper").setLevel(logging.ERROR)
logging.getLogger("src.nlp.radio_runner").setLevel(logging.WARNING)

import torch  # noqa: E402

from scripts.bench._common import (  # noqa: E402
    BenchResult,
    export_csv,
    export_markdown,
    make_start_panel,
    render_results_table,
    time_function,
)
from scripts.cli.theme import console  # noqa: E402
from src.nlp.radio_runner import WhisperTranscriber  # noqa: E402

_DATA_ROOT = _REPO_ROOT / "data"
_AUDIO_ROOT = _DATA_ROOT / "raw" / "radio_audio"
_EVAL_DIR = _DATA_ROOT / "eval"

_DEFAULT_SAMPLE_SIZE = 50
_AUDIO_SAMPLE_SEED = 42
_LATENCY_BUCKET_LABEL = "overall"
_NOTES_TEXT = "WER not computed: ground-truth corpus not paired, pending Track-A radio benchmark"


class WhisperLatencyRunner:
    """Encapsulate the Whisper turbo per-clip latency measurement.

    Loads the transcriber once, fixes a random sample of mp3 paths,
    and exposes :meth:`run` which returns a single overall bucket
    :class:`BenchResult`. The runner deliberately measures only one
    clip per measured iteration (no batching) because the simulation
    pipeline calls Whisper one row at a time as well.
    """

    def __init__(
        self,
        n_runs: int = _DEFAULT_SAMPLE_SIZE,
        device: str = "auto",
        sample_size: int = _DEFAULT_SAMPLE_SIZE,
    ) -> None:
        """Pre-load the Whisper checkpoint and pick the audio sample.

        Args:
            n_runs: Number of measured calls. Five additional warm-up
                calls are always performed first to absorb the JIT
                cost of the first decode.
            device: ``cpu``, ``cuda`` or ``auto``. ``auto`` selects
                CUDA when ``torch.cuda.is_available()``. Whisper itself
                routes the model to the right device through its
                internal default; this string is only used for the
                ``device`` column in the artefact.
            sample_size: Maximum number of distinct mp3 files to draw
                from disk. Falls back to whatever is available when
                fewer files exist than the requested size.
        """
        self.n_runs = int(n_runs)
        self.device_label = self._resolve_device(device)
        self.transcriber = WhisperTranscriber(model_name="turbo")
        self.sample_paths = self._sample_audio_paths(sample_size)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve ``auto`` to ``cuda`` / ``cpu`` based on CUDA availability.

        ``cpu`` and ``cuda`` are passed through unchanged so the
        operator can force one device on a machine where CUDA is
        present but should be avoided (e.g. running the bench while
        another GPU job is hogging memory).
        """
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @staticmethod
    def _sample_audio_paths(sample_size: int) -> list[Path]:
        """Return up to ``sample_size`` distinct mp3 paths drawn with a fixed seed.

        The seed is hard-coded so the benchmark is reproducible across
        runs: a different seed would change which clips contribute to
        the latency distribution and therefore the final P95. When
        fewer than ``sample_size`` files exist on disk every available
        file is returned and the script proceeds with a smaller
        sample (the artefact records the actual count via
        ``n_messages``).
        """
        all_mp3s = sorted(_AUDIO_ROOT.rglob("*.mp3"))
        if not all_mp3s:
            raise FileNotFoundError(
                f"No mp3 files found under {_AUDIO_ROOT} — cannot run latency bench"
            )
        rng = random.Random(_AUDIO_SAMPLE_SEED)
        if len(all_mp3s) <= sample_size:
            return all_mp3s
        return rng.sample(all_mp3s, sample_size)

    def _transcribe_one(self) -> None:
        """Transcribe a single clip from the sample (round-robin).

        ``time_function`` calls the closure ``n_warmup + n_runs``
        times. Cycling through the sample keeps every clip exercised
        roughly evenly so the latency distribution does not collapse
        onto a single fast or slow file.
        """
        idx = self._call_idx
        path = self.sample_paths[idx % len(self.sample_paths)]
        self.transcriber.transcribe(path)
        self._call_idx += 1

    def run(self) -> BenchResult:
        """Time the transcription loop and return the overall-bucket row.

        ``time_function`` performs five warm-up calls (discarded) and
        ``self.n_runs`` measured calls (kept). The Whisper model is
        loaded lazily on the first warm-up call; subsequent calls
        reuse the in-memory weights so the measured P95 reflects pure
        decode latency rather than load + decode.
        """
        self._call_idx = 0
        latency = time_function(self._transcribe_one, n_warmup=5, n_runs=self.n_runs)
        metrics = {
            "n_messages": len(self.sample_paths),
            "mean_ms": latency["mean_ms"],
            "latency_p50_ms": latency["p50_ms"],
            "latency_p95_ms": latency["p95_ms"],
            "device": self.device_label,
            "notes": _NOTES_TEXT,
        }
        return BenchResult(name=_LATENCY_BUCKET_LABEL, metrics=metrics)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

_COLUMNS = [
    "bucket",
    "n_messages",
    "mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "device",
    "notes",
]
_TITLE = "Whisper turbo per-clip latency"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whisper turbo latency benchmark.")
    parser.add_argument(
        "--n-runs",
        type=int,
        default=_DEFAULT_SAMPLE_SIZE,
        help="Measured runs (default 50). 5 warm-up runs always precede.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device label for the artefact (auto resolves via torch.cuda.is_available).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=_DEFAULT_SAMPLE_SIZE,
        help="Number of distinct mp3 files to sample (default 50).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    console.print(
        make_start_panel(
            "bench_whisper.py",
            f"Whisper turbo latency, {args.n_runs} measured runs (device={args.device}).",
        )
    )

    runner = WhisperLatencyRunner(
        n_runs=args.n_runs,
        device=args.device,
        sample_size=args.sample_size,
    )
    results = [runner.run()]

    md_path = _EVAL_DIR / "whisper_results.md"
    csv_path = _EVAL_DIR / "whisper_results.csv"
    export_markdown(results, md_path, _TITLE, _COLUMNS)
    export_csv(results, csv_path, _COLUMNS)

    console.print(render_results_table(results, _TITLE, _COLUMNS))
    console.print(f"[green]Markdown:[/green] {md_path.resolve()}")
    console.print(f"[green]CSV:     [/green] {csv_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
