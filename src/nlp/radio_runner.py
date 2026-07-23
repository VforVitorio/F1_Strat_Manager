"""Replay-time bridge from the static OpenF1 radio corpus to the N29 Radio Agent.

The static build pipeline (``src/data_extraction/openf1/radio_dataset_builder.py``
+ ``scripts/build_radio_dataset.py``) produces two parquets per Grand Prix
under ``data/processed/race_radios/{year}/{slug}/`` (``radios.parquet`` +
``rcm.parquet``) plus the matching MP3 files under
``data/raw/radio_audio/{year}/{slug}/driver_{N}/``. That tree is what the
multi-agent system needs at simulation time, but it cannot be plugged into
``RaceState.radio_msgs`` / ``RaceState.rcm_events`` directly:

* the radio rows reference an MP3 on disk, not a transcript — Whisper has
  to run on every audio clip and turn it into text before N29 can do any
  NLP on it
* the parquet uses ``driver_number`` (int), but ``RadioMessage.driver``
  expects the 3-letter code (``HAM``, ``VER``, ``LEC``)
* re-running Whisper on every CLI invocation would be unusably slow, so
  the transcripts have to be cached the first time they are produced and
  read back in O(1) on every subsequent run

:class:`RadioPipelineRunner` is the single object that handles all three
problems. It loads the parquets once at construction, transcribes every
referenced MP3 (skipping anything already in the JSON cache), and exposes
:meth:`radios_for_lap` so the simulation loop can pull `(radios, rcms)`
for the current lap and assign the lists straight onto the
:class:`RaceState`. The orchestrator's existing ``_to_radio_message`` /
``_to_rcm_event`` coercers accept the dict shape this runner emits, so
nothing in N29 itself needs to change.

The module lives under ``src/nlp/`` rather than next to ``radio_agent.py``
because its main job is **transcription** (Whisper) plus a thin
parquet → dict adapter — no inference, no LangGraph, no LLM. Sitting next
to ``src/nlp/pipeline.py`` and the sentiment / intent / NER classifiers
keeps all the radio-NLP plumbing in one place and means the lazy
first-run downloader (``src/f1_strat_manager/data_cache.py``) can call
``resolve_gp_slug`` without dragging in the full ``src.agents`` package
init (which loads RoBERTa / BERT / Whisper at module import time).

The slug-resolution table :data:`COUNTRY_SLUG_BY_GP` actually lives in
the even-lighter :mod:`src.f1_strat_manager.gp_slugs` module so the data
bootstrap path stays free of any pandas / NLP imports; this module
re-exports the symbols so callers can keep importing them from the
``radio_runner`` namespace.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# Re-export the slug helpers from the lightweight standalone module so
# downstream code can keep importing them from radio_runner without
# touching ``src.f1_strat_manager`` directly. The actual table lives in
# ``f1_strat_manager.gp_slugs`` so the lazy first-run downloader in
# ``data_cache`` can resolve slugs without dragging in the full
# ``src.agents`` package init (which loads RoBERTa / BERT / Whisper).
from src.f1_strat_manager.gp_slugs import COUNTRY_SLUG_BY_GP, resolve_gp_slug

__all__ = [
    "COUNTRY_SLUG_BY_GP",
    "resolve_gp_slug",
    "WhisperTranscriber",
    "RadioPipelineRunner",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Whisper transcription — process-local singleton
# ---------------------------------------------------------------------------

class WhisperTranscriber:
    """Process-local Whisper holder that loads the model lazily on first use.

    The Whisper checkpoint (``turbo``, ``large``, etc.) is heavy enough that
    re-loading it on every :class:`RadioPipelineRunner` constructor would
    dominate the simulation startup cost. Wrapping the load behind
    :meth:`ensure_loaded` and stashing the result on the instance lets two
    runners in the same process — e.g. the smoke notebook constructing one
    runner per GP — share the weights once they have been paid for. The
    module-level :func:`_get_whisper` factory takes care of returning the
    same instance for callers that ask for the same model name; passing a
    different ``model_name`` rebuilds the singleton because Whisper has no
    runtime way to swap weights in place.

    The class deliberately keeps no audio cache or transcript cache of its
    own — those concerns belong to :class:`RadioPipelineRunner`, which knows
    what to key the cache by (the relative ``audio_path`` from the parquet)
    and where to persist it (``data/processed/radio_nlp/...``). Keeping the
    transcriber stateless w.r.t. content makes it trivially reusable across
    different GPs without an LRU eviction story.
    """

    def __init__(self, model_name: str = "turbo") -> None:
        """Configure the transcriber but defer the model load.

        ``model_name`` is the Whisper checkpoint identifier accepted by
        ``whisper.load_model`` (``tiny``, ``base``, ``small``, ``medium``,
        ``large``, ``turbo``). Stored on the instance so the cache layer
        can stamp every transcript entry with the model that produced it
        and invalidate stale entries when the user passes
        ``--whisper-model`` on a later run.
        """
        self.model_name = model_name
        self._model = None

    def ensure_loaded(self) -> None:
        """Load the Whisper checkpoint into memory on the first call.

        Subsequent calls are no-ops, so the runner can put this at the
        top of every transcribe loop without worrying about repeated
        loads. The import is local to avoid pulling Whisper into module
        import time — the radio_runner module is also imported by
        ``data_cache.ensure_radio_corpus`` (for ``resolve_gp_slug``) and
        we do not want bare-metal first-run downloads to cost a Whisper
        load that the user may never need.
        """
        if self._model is not None:
            return
        import whisper  # local import — see docstring
        logger.info("Loading Whisper model %r ...", self.model_name)
        self._model = whisper.load_model(self.model_name)

    def transcribe(self, audio_path: Path) -> dict:
        """Transcribe one MP3 and return ``{text, duration_s, model}``.

        Decodes the audio with ``soundfile`` (libsndfile) and resamples
        to Whisper's native 16 kHz mono via ``librosa.resample``. We
        avoid ``librosa.load`` because on Windows it can fall back to
        the ``audioread`` backend which spawns ``ffmpeg`` with a piped
        stderr that emits cp1252 bytes — Python's subprocess reader
        thread then crashes trying to decode that as UTF-8. Going
        through ``soundfile`` directly skips that codepath entirely and
        is also faster on the OpenF1 MP3 corpus. ``fp16`` is only
        enabled on CUDA because Whisper warns and falls back to fp32
        on CPU otherwise.

        Returns a dict with the joined transcript text, the audio
        duration in seconds (useful for QA later), and the model name
        that produced the result so the cache can detect stale entries
        when the user switches checkpoints. Raises
        :class:`FileNotFoundError` when the MP3 is missing so the runner
        can write an empty-text cache entry instead of crashing the
        whole transcription loop.
        """
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        self.ensure_loaded()
        import librosa
        import numpy as np
        import soundfile as sf
        import torch
        audio, src_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        duration = float(len(audio)) / float(src_sr) if src_sr else 0.0
        if src_sr != 16000:
            audio = librosa.resample(audio, orig_sr=src_sr, target_sr=16000)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        result = self._model.transcribe(  # type: ignore[union-attr]
            audio,
            task="transcribe",
            language="en",
            fp16=torch.cuda.is_available(),
        )
        text = " ".join(seg["text"].strip() for seg in result["segments"])
        return {
            "text":       text.strip(),
            "duration_s": duration,
            "model":      self.model_name,
        }


_WHISPER_SINGLETON: Optional[WhisperTranscriber] = None


def _get_whisper(model_name: str) -> WhisperTranscriber:
    """Return the process-wide Whisper instance for ``model_name``.

    Builds a fresh :class:`WhisperTranscriber` on the first call (or
    when the requested model differs from the cached one) and returns
    the cached instance otherwise. Two runners constructed in the same
    process therefore share the loaded Whisper weights as long as they
    use the same checkpoint, which is the common case for an
    interactive CLI or backend session that simulates multiple GPs in
    a row.
    """
    global _WHISPER_SINGLETON
    if (
        _WHISPER_SINGLETON is None
        or _WHISPER_SINGLETON.model_name != model_name
    ):
        _WHISPER_SINGLETON = WhisperTranscriber(model_name)
    return _WHISPER_SINGLETON


# ---------------------------------------------------------------------------
# RadioPipelineRunner
# ---------------------------------------------------------------------------

@dataclass
class RadioPipelineRunner:
    """Replay-time facade over the static OpenF1 radio + RCM corpus for one GP.

    Constructed once per simulation run with the year, the friendly GP
    name (CLI form, e.g. ``"Sakhir"`` / ``"Imola"``), the already-loaded
    featured-laps DataFrame (used to build the ``driver_number → 3-letter
    code`` lookup so the emitted radio dicts carry ``HAM`` rather than
    ``44``), and the project data root (so the runner can join the
    parquets and the MP3 root from the same anchor as
    :func:`f1_strat_manager.data_cache.get_data_root`).

    On construction the runner:

    1. resolves the GP name to its corpus slug via :func:`resolve_gp_slug`
    2. loads ``radios.parquet`` and ``rcm.parquet`` if they exist (empty
       DataFrame + warning if either is missing — the simulation should
       degrade gracefully, not crash)
    3. builds the per-GP ``driver_number → 3-letter code`` map from the
       featured laps DataFrame, falling back to ``D{n}`` for any number
       that is not present (which can happen for a reserve driver radio)
    4. loads the JSON transcript cache from disk, dropping any entry
       whose ``model`` field disagrees with the currently configured
       Whisper checkpoint (so ``--whisper-model base`` cleanly
       re-transcribes the corpus that ``turbo`` left behind)
    5. eagerly transcribes every uncached radio when ``eager_transcribe``
       is True, lazily loading Whisper only if at least one row is
       actually missing — a fully-warm cache pays zero Whisper cost

    The public API is :meth:`radios_for_lap`, which returns
    ``(radio_dicts, rcm_dicts)`` for the requested lap shaped exactly the
    way the orchestrator's ``_to_radio_message`` / ``_to_rcm_event``
    coercers expect. The CLI loop assigns those lists straight onto
    ``RaceState.radio_msgs`` / ``RaceState.rcm_events`` and the rest of
    the multi-agent stack does not need to know the corpus is involved.
    """

    year:                  int
    gp_name:               str
    laps_df:               pd.DataFrame
    data_root:             Path
    transcript_cache_dir:  Optional[Path] = None
    whisper_model_name:    str            = "turbo"
    eager_transcribe:      bool           = True
    disable_transcription: bool           = False

    # Populated in __post_init__
    slug:         str           = field(init=False)
    radios_df:    pd.DataFrame  = field(init=False)
    rcm_df:       pd.DataFrame  = field(init=False)
    audio_root:   Path          = field(init=False)
    transcripts:  dict          = field(init=False)
    _driver_code: dict          = field(init=False)
    _cache_path:  Path          = field(init=False)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        """Resolve paths, load the parquets, build the driver map, transcribe.

        The full constructor flow is sequential and intentionally
        non-parallel: the driver map needs the slug, the cache load
        needs the audio root, and the transcription pass needs the
        cache. Each step logs at INFO so the simulation startup
        produces a single readable trace instead of a wall of debug
        output, and exceptions during the I/O steps are caught and
        downgraded to warnings + empty frames so a missing GP cannot
        crash the CLI before the race even starts.
        """
        self.slug = resolve_gp_slug(self.gp_name)

        processed_dir   = self.data_root / "processed" / "race_radios" / str(self.year) / self.slug
        self.audio_root = self.data_root / "raw" / "radio_audio"

        self.radios_df = self._load_radios_parquet(processed_dir / "radios.parquet")
        self.rcm_df    = self._load_rcm_parquet(processed_dir / "rcm.parquet")

        self._driver_code = self._build_driver_code_map(self.laps_df, self.gp_name)

        self._cache_path = self._resolve_cache_path()
        self.transcripts = self._load_transcript_cache(self._cache_path)

        if self.eager_transcribe and not self.disable_transcription:
            self._transcribe_all()

    def close(self) -> None:
        """Release the parquet DataFrames so the GC can reclaim them.

        Whisper itself is **not** unloaded because the singleton may be
        shared with another runner in the same process; releasing the
        weights here would force the next GP simulation to pay the
        load cost again. Callers that want a hard reset should null
        the module-level singleton themselves.
        """
        self.radios_df = pd.DataFrame()
        self.rcm_df    = pd.DataFrame()

    # ── Public API ───────────────────────────────────────────────────────────

    def total_radios(self) -> int:
        """Return the total number of radio rows for this GP, post-filter.

        Used by the CLI startup banner to print a single ``"Radio
        corpus: 28 radios, 44 RCMs for Sakhir"`` line so the operator
        can sanity-check the corpus size before the simulation starts.
        """
        return int(len(self.radios_df))

    def total_rcms(self) -> int:
        """Return the total number of race-control message rows for this GP."""
        return int(len(self.rcm_df))

    def radios_for_lap(self, lap_number: int) -> tuple[list[dict], list[dict]]:
        """Return ``(radio_dicts, rcm_dicts)`` for the given race lap.

        Filters both DataFrames by ``lap_number`` and converts each row
        to the dict shape that
        ``strategy_orchestrator._to_radio_message`` and
        ``strategy_orchestrator._to_rcm_event`` accept. Radios always
        carry a transcript field — empty string when the MP3 was
        missing or transcription failed — so N29 sees a row even when
        the corresponding audio is unavailable; this matches the
        contract the legacy mock generator has been honouring and lets
        the radio agent emit a ``no usable text`` warning instead of
        skipping the lap entirely.

        The return is a fresh list per call so the caller can mutate
        it (the CLI appends optional mock radios on top of the real
        ones) without poisoning the runner's internal state.
        """
        radios: list[dict] = []
        if not self.radios_df.empty:
            mask = self.radios_df["lap_number"] == lap_number
            for _, row in self.radios_df[mask].iterrows():
                radios.append(self._radio_row_to_dict(row))

        rcms: list[dict] = []
        if not self.rcm_df.empty:
            mask = self.rcm_df["lap_number"] == lap_number
            for _, row in self.rcm_df[mask].iterrows():
                rcms.append(self._rcm_row_to_dict(row))

        return radios, rcms

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_radios_parquet(self, path: Path) -> pd.DataFrame:
        """Load the radios parquet, returning an empty frame on miss.

        A missing parquet is logged as a warning but never raises
        because the simulation should still be runnable when the
        corpus is partial — e.g. the user is replaying a 2024 GP that
        we have not yet built radios for. The same logic applies to
        the RCM loader; both halves degrade independently.
        """
        if not path.exists():
            logger.warning(
                "Radio parquet not found for %s %d at %s — running without radios",
                self.gp_name, self.year, path,
            )
            return pd.DataFrame()
        df = pd.read_parquet(path)
        logger.info(
            "Loaded %d radio rows for %s %d from %s",
            len(df), self.gp_name, self.year, path,
        )
        return df

    def _load_rcm_parquet(self, path: Path) -> pd.DataFrame:
        """Load the race-control parquet, returning an empty frame on miss."""
        if not path.exists():
            logger.warning(
                "RCM parquet not found for %s %d at %s — running without RCMs",
                self.gp_name, self.year, path,
            )
            return pd.DataFrame()
        df = pd.read_parquet(path)
        logger.info(
            "Loaded %d RCM rows for %s %d from %s",
            len(df), self.gp_name, self.year, path,
        )
        return df

    @staticmethod
    def _build_driver_code_map(laps_df: pd.DataFrame, gp_name: str) -> dict:
        """Return ``{driver_number: 3-letter code}`` for the requested GP.

        The featured-laps parquet stores both fields per row, so the
        per-GP slice is enough to build the lookup. Drops duplicates
        because there is one row per lap and we only need a single
        mapping per driver. Returns an empty dict on any failure
        (missing column, GP not in the parquet, etc.) so the runner
        falls back to the ``D{n}`` synthetic codes downstream.
        """
        try:
            sub = laps_df[laps_df["GP_Name"] == gp_name]
            sub = sub[["DriverNumber", "Driver"]].drop_duplicates()
            return {int(row.DriverNumber): str(row.Driver) for row in sub.itertuples()}
        except Exception as exc:  # noqa: BLE001 — log + degrade gracefully
            logger.warning("Could not build driver code map for %s: %s", gp_name, exc)
            return {}

    # ── Cache layer ──────────────────────────────────────────────────────────

    def _resolve_cache_path(self) -> Path:
        """Return the absolute JSON cache path for this GP's transcripts.

        Defaults to ``{data_root}/processed/radio_nlp/{year}/{slug}/transcripts.json``
        so the cache lives next to the parquet corpus and the directory
        layout matches the rest of the project's processed-data tree.
        ``transcript_cache_dir`` (set via the CLI ``--transcript-cache``
        flag) overrides only the parent directory; the file name and
        per-year/per-slug subtree are always added so a single override
        works for the whole season.
        """
        if self.transcript_cache_dir is not None:
            base = Path(self.transcript_cache_dir)
        else:
            base = self.data_root / "processed" / "radio_nlp"
        return base / str(self.year) / self.slug / "transcripts.json"

    def _load_transcript_cache(self, path: Path) -> dict:
        """Load the JSON cache from disk, dropping malformed or stale entries.

        The on-disk schema is a flat dict keyed by the **normalized**
        relative ``audio_path`` (forward slashes — see the path-norm
        helper below) with values shaped as
        ``{text, duration_s, model}``. Entries are silently dropped
        when:

        * the file does not exist (first run)
        * the JSON parses but the top-level value is not a dict
        * an individual entry is missing the ``text`` field
        * the entry's ``model`` does not match
          :attr:`whisper_model_name` — switching from ``turbo`` to
          ``base`` rebuilds the cache cleanly without any extra flag

        Always returns a dict so the rest of the runner can use it
        without ``None`` checks.
        """
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if not isinstance(raw, dict):
                logger.warning("Transcript cache at %s is not a dict — ignoring", path)
                return {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load transcript cache at %s: %s", path, exc)
            return {}

        clean: dict = {}
        for key, entry in raw.items():
            if not isinstance(entry, dict) or "text" not in entry:
                continue
            if entry.get("model") != self.whisper_model_name:
                # Stale entry from a previous Whisper checkpoint — drop it
                # so the upcoming transcription pass overwrites with the
                # right model.
                continue
            clean[key] = entry

        if len(clean) < len(raw):
            logger.info(
                "Transcript cache: kept %d / %d entries (dropped stale-model rows)",
                len(clean), len(raw),
            )
        else:
            logger.info("Loaded %d cached transcripts from %s", len(clean), path)
        return clean

    def _save_transcript_cache(self) -> None:
        """Persist the in-memory cache atomically to disk.

        Writes to a sibling ``.tmp`` file first and then ``os.replace``
        onto the final path so a process killed mid-write cannot leave
        a half-written JSON file behind. The parent directory is
        created lazily so the constructor stays side-effect free until
        the first transcription actually runs.
        """
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self.transcripts, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self._cache_path)

    @staticmethod
    def _normalize_audio_path(rel: str) -> str:
        """Return ``rel`` with backslashes converted to forward slashes.

        The parquet stores ``audio_path`` with native path separators
        (backslashes on the Windows build host, forward slashes
        elsewhere). Normalizing both on the read side and on the write
        side of the cache key keeps the JSON cache portable across
        OSes — the file built on Windows still hits in the cache when
        the same parquet is loaded on macOS or Linux.
        """
        return rel.replace("\\", "/")

    # ── Transcription pass ──────────────────────────────────────────────────

    def _transcribe_all(self) -> None:
        """Transcribe every uncached MP3 referenced by the radio table.

        Iterates the radio rows in their natural (chronological) order,
        skips anything already in :attr:`transcripts`, and calls
        :meth:`_transcribe_one` for the rest. Whisper is loaded only
        when the first uncached row is reached, so a fully-warm cache
        on a re-run pays zero load cost. Saves the cache to disk every
        ``CHECKPOINT_EVERY`` rows so a CTRL-C halfway through a long
        first run does not throw away the work that did finish.
        """
        if self.radios_df.empty:
            return

        CHECKPOINT_EVERY = 10
        n_to_do = sum(
            1 for _, row in self.radios_df.iterrows()
            if row.get("audio_path") is not None
            and self._normalize_audio_path(str(row["audio_path"])) not in self.transcripts
        )
        if n_to_do == 0:
            logger.info(
                "Transcript cache fully warm for %s %d (%d rows) — skipping Whisper",
                self.gp_name, self.year, len(self.radios_df),
            )
            return

        logger.info(
            "Transcribing %d uncached radios for %s %d (cache hit on %d) ...",
            n_to_do, self.gp_name, self.year, len(self.radios_df) - n_to_do,
        )

        new_since_save = 0
        for _, row in self.radios_df.iterrows():
            rel = row.get("audio_path")
            if rel is None:
                continue
            key = self._normalize_audio_path(str(rel))
            if key in self.transcripts:
                continue

            self._transcribe_one(rel)
            new_since_save += 1
            if new_since_save >= CHECKPOINT_EVERY:
                self._save_transcript_cache()
                new_since_save = 0

        if new_since_save > 0:
            self._save_transcript_cache()

    def _transcribe_one(self, rel_audio_path: str) -> None:
        """Transcribe one MP3 referenced by ``rel_audio_path`` and cache it.

        Resolves the relative path against :attr:`audio_root` and
        delegates to the Whisper singleton. On
        :class:`FileNotFoundError` (the parquet references an MP3 that
        is not on disk because the audio download was skipped or
        failed) and on any other Whisper exception, an empty-text
        entry is written to the cache so the runner does not retry
        the same broken file on every subsequent run. The radio row
        is still emitted downstream — N29 sees a row with ``text=""``
        and treats it as "no usable transcript", which is the same
        behaviour as a clip Whisper genuinely could not understand.
        """
        key = self._normalize_audio_path(str(rel_audio_path))
        abs_path = self.audio_root / Path(str(rel_audio_path))
        whisper = _get_whisper(self.whisper_model_name)
        try:
            entry = whisper.transcribe(abs_path)
        except FileNotFoundError:
            logger.warning("Audio file missing on disk: %s", abs_path)
            entry = {"text": "", "duration_s": 0.0, "model": self.whisper_model_name}
        except Exception as exc:  # noqa: BLE001 — log + degrade per row
            logger.warning("Whisper failed on %s: %s", abs_path, exc)
            entry = {"text": "", "duration_s": 0.0, "model": self.whisper_model_name}
        self.transcripts[key] = entry

    def _lookup_transcript(self, rel_audio_path) -> str:
        """Return the cached transcript text for one radio row.

        Returns the empty string when the row has no ``audio_path`` or
        the cache key is missing — both cases mean N29 should see a
        row but treat it as having no usable text. Centralised here
        so the row-to-dict converter does not have to repeat the
        normalization + null handling.
        """
        if rel_audio_path is None:
            return ""
        key = self._normalize_audio_path(str(rel_audio_path))
        entry = self.transcripts.get(key)
        if not entry:
            return ""
        return str(entry.get("text", ""))

    # ── Row-to-dict converters ──────────────────────────────────────────────

    def _driver_code_for(self, driver_number) -> str:
        """Translate a numeric driver number to its 3-letter race code.

        Falls back to ``D{n}`` for any number not in the per-GP map
        (reserve driver, qualy-only entry, parquet schema drift),
        which keeps the orchestrator's coercer happy without raising
        on the corner case. ``UNK`` is reserved for the case where
        the parquet itself has a missing/NaN driver number on a row,
        which is rare but happens for some RCMs.
        """
        if driver_number is None:
            return "UNK"
        try:
            n = int(driver_number)
        except (TypeError, ValueError):
            return "UNK"
        return self._driver_code.get(n, f"D{n}")

    def _radio_row_to_dict(self, row: pd.Series) -> dict:
        """Convert one parquet radio row to the orchestrator-coercible dict.

        Output keys: ``driver`` (3-letter code), ``lap`` (int),
        ``text`` (transcript or empty string), ``timestamp`` (UTC
        Timestamp from the parquet). The orchestrator's
        ``_to_radio_message`` reads exactly these four fields, so the
        runner does not need to emit anything else even though the
        parquet has more columns (session_key, recording_url, ...).
        """
        return {
            "driver":    self._driver_code_for(row.get("driver_number")),
            "lap":       int(row["lap_number"]),
            "text":      self._lookup_transcript(row.get("audio_path")),
            "timestamp": row.get("date"),
        }

    def _rcm_row_to_dict(self, row: pd.Series) -> dict:
        """Convert one parquet RCM row to the orchestrator-coercible dict.

        Output keys mirror what the orchestrator's ``_to_rcm_event``
        coercer reads: ``message``, ``flag``, ``category``, ``lap``,
        ``racing_number`` (mapped from the parquet's ``driver_number``,
        which is NaN for track-wide events) and ``scope``. ``scope``
        is normalised to a string so the coercer's ``str(...)`` cast
        cannot stumble on a NaN sentinel; the parquet stores
        ``"DRIVER"`` / ``"TRACK"`` / ``"SECTOR"`` already, so the
        normalisation is a no-op for well-formed rows.
        """
        racing_number: Optional[int]
        try:
            raw = row.get("driver_number")
            racing_number = int(raw) if pd.notna(raw) else None
        except (TypeError, ValueError):
            racing_number = None

        return {
            "message":       str(row.get("message", "") or ""),
            "flag":          str(row.get("flag", "") or ""),
            "category":      str(row.get("category", "") or ""),
            "lap":           int(row["lap_number"]),
            "racing_number": racing_number,
            "scope":         str(row.get("scope", "") or ""),
        }
