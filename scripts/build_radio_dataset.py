"""
Build the static team-radio dataset for one or more F1 seasons.

Wraps :class:`src.data_extraction.openf1.radio_dataset_builder.RadioDatasetBuilder`
in a multi-GP CLI loop. For every requested season the script asks OpenF1
for the full list of Race sessions, then iterates over each Grand Prix and
calls ``build_and_write`` to persist a per-GP parquet to the configured
output directory.

The output is the offline corpus the N29 Radio Agent will consume during
race replay. Two parquets are produced per GP: one for team radios (driver,
lap, recording_url, audio_path) and one for race control messages (driver?,
lap, flag, category, message). Both are filtered with the same structural
rule: formation lap, race-start lap and chequered-flag lap dropped, so
every row left in either parquet belongs to a normal race lap and could
plausibly inform a strategy decision.

By default the build also downloads every radio MP3 referenced in the
parquet to a parallel ``data/raw/radio_audio/`` tree, organised one level
per year, GP slug, and driver number. The radio parquet's ``audio_path``
column carries the path **relative to** the audio root so the on-disk
corpus stays portable across machines. The actual transcription
(Whisper / Nemotron) and NLP (sentiment / intent / NER) stays out of this
script: those steps happen on demand at simulation runtime in the
``RadioPipelineRunner`` consumed by the future N29 Radio Agent.

Usage:
    python scripts/build_radio_dataset.py                                    # builds 2025 only (default)
    python scripts/build_radio_dataset.py --gps Bahrain Australia            # subset of 2025
    python scripts/build_radio_dataset.py --years 2023 2024 2025             # historical full build
    python scripts/build_radio_dataset.py --output-dir data/processed/race_radios
    python scripts/build_radio_dataset.py --audio-dir data/raw/radio_audio   # MP3 destination
    python scripts/build_radio_dataset.py --skip-audio                       # parquets only, no MP3 download
    python scripts/build_radio_dataset.py --skip-existing                    # resume after a crash
    python scripts/build_radio_dataset.py --gp-delay 2.0                     # extra cooldown vs OpenF1 rate limit

Rate limiting:
    Every HTTP call inherits a retry-with-exponential-backoff policy from a
    shared ``requests.Session`` mounted in ``build_retry_session``: 429 and
    5xx responses retry up to five times (1s, 2s, 4s, 8s, 16s). On top of
    that, the build loop sleeps ``--gp-delay`` seconds between consecutive
    GPs (default 1.0s) so the steady-state request rate stays below OpenF1's
    throttle threshold and the retry path never gets exercised in the happy
    case. The same retry policy is reused for the static MP3 host, so the
    audio download stage gets the same protection as the metadata fetches.

Output layout:
    data/processed/race_radios/
        2025/
            bahrain/
                radios.parquet            ← team radios (10 columns, incl. audio_path)
                rcm.parquet               ← race control messages (13 columns)
            australia/
                radios.parquet
                rcm.parquet
            ...
            abu_dhabi/
                radios.parquet
                rcm.parquet

    data/raw/radio_audio/
        2025/
            bahrain/
                driver_1/   HAMILTON_44_20250413_151002.mp3 ...
                driver_44/  ...
            australia/
                ...
            ...

Both trees share the identical ``{year}/{slug}/`` substructure on purpose,
so a downstream consumer that knows the GP can build both the radio
parquet path and the MP3 directory path from the same fragment.

The radio parquets follow the 10-column ``OUTPUT_SCHEMA`` defined in
``radio_dataset_builder``: session_key, meeting_key, year, gp, total_laps,
driver_number, lap_number, date, recording_url, audio_path.

The RCM parquets follow the 13-column ``RCM_OUTPUT_SCHEMA``: session_key,
meeting_key, year, gp, total_laps, driver_number, lap_number, date,
category, flag, scope, sector, message.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import requests

# Make ``src.*`` imports work whether the script is invoked as
# ``python scripts/build_radio_dataset.py`` or ``python -m scripts.build_radio_dataset``.
# scripts/ is a package but the parent ``src/`` package lives one level up,
# so the repo root must be on sys.path before any project import.
#
# ``.git``-search-with-fallback, not a fixed ``.parent.parent``: the latter
# silently resolves to the wrong directory under a `uv tool install` layout,
# where this file is not exactly one level below the repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = next(
    (p for p in [_SCRIPT_DIR, *_SCRIPT_DIR.parents] if (p / ".git").exists()),
    _SCRIPT_DIR.parent,
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data_extraction.openf1.radio_dataset_builder import (  # noqa: E402, I001
    OPENF1_BASE,
    RadioDatasetBuilder,
    build_retry_session,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default season used when the operator does not pass ``--years``. The N29
# Radio Agent and the lap-by-lap simulation only consume 2025 data, so a bare
# ``python scripts/build_radio_dataset.py`` builds just that year. Older
# seasons are still reachable on demand via ``--years 2023 2024 2025`` for
# historical NLP retraining or one-off analysis, but they are not part of
# the default critical path.
DEFAULT_YEARS: tuple[int, ...] = (2025,)


@dataclass
class BuildConfig:
    """Centralised configuration for the multi-GP radio dataset build.

    Grouping every tunable knob here keeps :class:`RadioDatasetCLI` thin and
    means the CLI argparse layer is the only place that maps user input onto
    runtime behaviour. Defaults match the convention used elsewhere in the
    project: per-season parquets under ``data/processed/race_radios`` and
    raw MP3 files under ``data/raw/radio_audio``.

    Attributes:
        output_dir:   Directory where per-GP parquets will be written. Created
                      lazily by the underlying ``RadioDatasetBuilder`` on the
                      first successful write so a dry run does not pollute the
                      filesystem when every GP fails.
        audio_dir:    Root directory where downloaded MP3 files are stored,
                      organised as ``{audio_dir}/{year}/{slug}/driver_{N}/``.
                      The radio parquet's ``audio_path`` column carries the
                      path **relative to** this root so the on-disk corpus
                      stays portable across machines: a consumer that knows
                      ``audio_dir`` can rebuild the absolute path with one
                      ``Path.__truediv__`` call. Created lazily on the first
                      successful download.
        skip_audio:   When True, the build pipeline runs the metadata stage
                      (parquet writes) and skips the MP3 download stage. The
                      ``audio_path`` column on the resulting parquet stays
                      ``None``. Useful for fast iteration during development
                      and for resuming a previous build whose audio half is
                      already complete (the ``--skip-existing`` flag still
                      treats only the parquets, not the MP3s, when deciding
                      whether to skip a GP outright).
        http_timeout: Per-request timeout in seconds for the OpenF1 calls.
                      30 s matches the notebook prototype and gives the
                      ``/v1/laps`` endpoint enough headroom on slow links.
        gp_delay_s:   Fixed cooldown in seconds between consecutive GP
                      builds. The HTTP adapter already retries 429/5xx with
                      exponential backoff, so this delay is belt-and-braces:
                      it keeps the steady-state request rate under OpenF1's
                      rate-limit threshold in the happy path, which avoids
                      ever triggering the retry path in the first place.
                      Set to 0 to disable when testing against a local
                      mirror or a mocked transport.
    """

    output_dir: Path = field(
        default_factory=lambda: _REPO_ROOT / "data" / "processed" / "race_radios"
    )
    audio_dir: Path = field(default_factory=lambda: _REPO_ROOT / "data" / "raw" / "radio_audio")
    skip_audio: bool = False
    http_timeout: int = 30
    gp_delay_s: float = 1.0


CFG = BuildConfig()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_radio_dataset")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RaceMeta:
    """Identifying metadata for a single Race session discovered via OpenF1.

    Produced by :meth:`RadioDatasetCLI.discover_races` from the
    ``/v1/sessions`` payload, then consumed by the build loop. Carrying both
    ``session_key`` and ``country_name`` lets the loop log human-readable
    progress lines while still being able to call OpenF1 by key if a future
    optimisation needs it.

    Attributes:
        year:               Season the race belongs to. Used for the output
                            filename and to disambiguate the same circuit
                            visited across multiple seasons.
        country_name:       Exact country string returned by OpenF1, used as
                            the ``country_name`` argument to
                            ``RadioDatasetBuilder``. Must be passed through
                            unchanged because the builder slugifies it for
                            the parquet filename.
        session_key:        OpenF1 session identifier. Carried for log lines
                            and as a stable handle in case future versions of
                            the script need to fetch additional
                            session-specific data.
        circuit_short_name: OpenF1 circuit short name (e.g. ``"Imola"`` /
                            ``"Monza"`` / ``"Miami"`` / ``"COTA"``). Required
                            to disambiguate countries that host more than one
                            GP per season: Italy (Imola + Monza) and the
                            United States (Miami + Austin + Las Vegas), so
                            the builder writes them to distinct ``italy_imola`` /
                            ``italy_monza`` / ``united_states_*`` slugs instead
                            of overwriting each other under a shared
                            ``italy/`` or ``united_states/`` folder. Optional
                            for backwards compatibility with code paths that
                            haven't been updated yet, but every code path
                            inside this script populates it.
    """

    year: int
    country_name: str
    session_key: int
    circuit_short_name: Optional[str] = None


@dataclass
class BuildResult:
    """Outcome of attempting to build a single GP's radio + RCM parquets.

    Captures enough state for the final summary log so the operator can tell
    at a glance which GPs were skipped (already on disk), which were built
    fresh, and which failed (and why). Failures are kept as strings instead
    of raised exceptions because the CLI loop wants to keep going past
    individual GP errors and only report at the end.

    Each successful build produces two parquets per GP, radios and RCMs,
    plus (when audio is enabled) one MP3 per radio row, so this result
    tracks all three numbers separately. The summary aggregates them into
    a single ``"(N radios, M RCMs, K audio)"`` line per GP and into
    season-wide totals at the bottom of the run.

    Attributes:
        race:        The :class:`RaceMeta` this result describes. Kept on
                     the instance so the summary can group results by
                     year/GP without a second pass over the source list.
        status:      One of ``"built"``, ``"skipped"``, or ``"failed"``.
                     Drives the summary counters and the per-row log line.
        radio_rows:  Number of radio rows written to the radio parquet
                     (zero for skipped or failed runs).
        rcm_rows:    Number of RCM rows written to the RCM parquet
                     (zero for skipped or failed runs).
        audio_rows:  Number of radio rows whose MP3 download succeeded
                     (i.e. rows with a non-null ``audio_path``). Stays at
                     zero when ``cfg.skip_audio`` is True or when every
                     download for the GP failed.
        radio_path:  Path of the written radio parquet, or ``None`` for
                     skipped/failed runs.
        rcm_path:    Path of the written RCM parquet, or ``None`` for
                     skipped/failed runs.
        error:       Short error description for failed runs, ``None``
                     otherwise. Stored as a plain string so the summary
                     can be printed without re-raising or pickling
                     exceptions.
    """

    race: RaceMeta
    status: str
    radio_rows: int = 0
    rcm_rows: int = 0
    audio_rows: int = 0
    radio_path: Optional[Path] = None
    rcm_path: Optional[Path] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Multi-GP CLI orchestrator
# ---------------------------------------------------------------------------


class RadioDatasetCLI:
    """Multi-GP wrapper around :class:`RadioDatasetBuilder`.

    Owns the configuration, the shared ``requests.Session`` (so every OpenF1
    call across the full season build reuses the same TCP connection), and
    the orchestration loop that discovers races, applies the optional GP
    filter, calls the builder per GP, and produces the final summary. Kept
    as a class rather than a free function so unit tests can swap the
    ``RadioDatasetBuilder`` instance for a fake without monkey-patching the
    module.

    The orchestrator is deliberately fail-soft: a failure on one GP logs an
    error and produces a ``BuildResult`` with ``status="failed"`` instead of
    aborting the run. This matters for full-season builds where one slow or
    rate-limited request would otherwise force the operator to restart from
    scratch.
    """

    def __init__(self, cfg: BuildConfig) -> None:
        """Wire the CLI to its config and initialise the shared HTTP session.

        The shared ``requests.Session`` is built via
        :func:`build_retry_session` so both the direct ``discover_races``
        HTTP call and every request issued by the underlying
        :class:`RadioDatasetBuilder` inherit the same 429/5xx retry-with-
        backoff policy. This is the main reason this script exists as a
        class instead of a one-shot function: reusing the retry-enabled
        session across the entire multi-GP loop keeps the rate-limit
        story centralised in one place instead of scattered across every
        call site. The session is owned by the builder and closed
        implicitly when the process exits.
        """
        self._cfg = cfg
        self._http = build_retry_session()
        self._builder = RadioDatasetBuilder(
            output_dir=cfg.output_dir,
            http_timeout=cfg.http_timeout,
            session=self._http,
        )

    # ── Race discovery ───────────────────────────────────────────────────────

    def discover_races(self, years: Iterable[int]) -> list[RaceMeta]:
        """Query OpenF1 for every Race session across the requested seasons.

        Issues one ``/v1/sessions?year=YYYY&session_type=Race`` call per year
        instead of looping per GP, which keeps the discovery phase under a
        handful of HTTP calls regardless of how many seasons the user asks
        for. Skips years that return an empty session list with a warning so
        the build can still continue with the remaining seasons.

        OpenF1 returns Sprint sessions with ``session_type="Race"`` because
        a Sprint **is** a race event in their schema, which means a naive
        consumer of the response double-counts every Sprint weekend (China,
        Miami, Spa, Austin, São Paulo, Qatar). The early build of 2025
        China hit exactly this and overwrote the main-race parquet with
        the Sprint payload (19 laps instead of ~56). To fix this for good,
        the discovery loop filters the response client-side by
        ``session_name == "Race"`` so only the Sunday Grand Prix survives,
        and the Sprint sibling is silently dropped.

        Returns the discovered races sorted by year ascending and then by
        OpenF1's natural calendar order within a season, so the build loop
        produces predictable, repeatable output across runs.
        """
        races: list[RaceMeta] = []
        for year in sorted(set(years)):
            log.info("Discovering Race sessions for %d ...", year)
            try:
                response = self._http.get(
                    f"{OPENF1_BASE}/sessions",
                    params={"year": year, "session_type": "Race"},
                    timeout=self._cfg.http_timeout,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                log.error("Failed to list sessions for %d: %s", year, exc)
                continue

            payload = response.json()
            if not payload:
                log.warning("OpenF1 returned no Race sessions for %d", year)
                continue

            year_count = 0
            sprint_count = 0
            for session in payload:
                country = session.get("country_name")
                key = session.get("session_key")
                if country is None or key is None:
                    continue
                # Sprint weekends: drop the Sprint so we never overwrite the
                # main-race parquet with Sprint data on the same slug.
                if session.get("session_name") != "Race":
                    sprint_count += 1
                    continue
                # circuit_short_name is captured here so the build loop can
                # apply the multi-race-country slug disambiguation rule
                # (Italy → italy_imola / italy_monza, United States →
                # united_states_miami / united_states_cota / ...) without a
                # second /v1/sessions round trip per GP.
                races.append(
                    RaceMeta(
                        year=year,
                        country_name=country,
                        session_key=int(key),
                        circuit_short_name=session.get("circuit_short_name"),
                    )
                )
                year_count += 1

            log.info(
                "  → found %d main-race sessions for %d (filtered %d non-Race siblings)",
                year_count,
                year,
                sprint_count,
            )

        return races

    # ── Filtering ────────────────────────────────────────────────────────────

    @staticmethod
    def filter_races(
        races: list[RaceMeta],
        gp_filter: Optional[list[str]],
    ) -> list[RaceMeta]:
        """Restrict the discovered race list to a subset of country / circuit names.

        Case-insensitive match because the user may pass ``"bahrain"`` from
        the CLI even though OpenF1 stores ``"Bahrain"``. Returns the input
        list unchanged when ``gp_filter`` is ``None`` or empty so the
        ``--gps`` argument behaves as a no-op when the user does not pass it.

        A token matches a race when it equals **either** the race's
        ``country_name`` or its ``circuit_short_name`` (both lowercased).
        Matching by circuit name is essential for double-header countries
        (Italy → Imola/Monza, United States → Miami/Austin/Las Vegas)
        because the country alone would rebuild every sibling race for
        that country. With circuit-name matching the operator can target a
        single race precisely: ``--gps Imola`` only rebuilds Imola, while
        ``--gps Italy`` rebuilds both Imola and Monza in one go.

        Logs a warning for any GP name that did not match anything so a typo
        in ``--gps`` does not silently produce a zero-row run that the
        operator might mistake for a successful build.
        """
        if not gp_filter:
            return races

        wanted = {name.lower() for name in gp_filter}

        def _matches(r: RaceMeta) -> bool:
            if r.country_name.lower() in wanted:
                return True
            if r.circuit_short_name and r.circuit_short_name.lower() in wanted:
                return True
            return False

        filtered = [r for r in races if _matches(r)]

        matched: set[str] = set()
        for r in filtered:
            matched.add(r.country_name.lower())
            if r.circuit_short_name:
                matched.add(r.circuit_short_name.lower())
        for name in wanted - matched:
            log.warning("--gps filter: no match for '%s'", name)

        return filtered

    # ── Build loop ───────────────────────────────────────────────────────────

    def build_all(
        self,
        races: list[RaceMeta],
        skip_existing: bool,
    ) -> list[BuildResult]:
        """Iterate over every race and build both parquets, collecting results.

        Wraps each per-GP build in a try/except so a single failure (network
        blip, malformed payload, missing laps) cannot abort the rest of the
        run. For each GP it prepares the session bundle once and feeds it to
        both ``build_radio_table`` and ``build_rcm_table`` so OpenF1 only
        sees two shared fetches (sessions + laps) instead of four.

        After the metadata tables are built, and unless ``cfg.skip_audio``
        is True, the loop runs :meth:`RadioDatasetBuilder.download_audio_files`
        to fetch every MP3 referenced in the radio table and stash it under
        ``cfg.audio_dir``. The downloads share the same retry-enabled
        session as the metadata fetches, so 429 throttling on the static
        media host inherits the same backoff path. The ``audio_path``
        column on the radio parquet is populated with the **relative**
        path under ``cfg.audio_dir``, which keeps the parquet portable
        across machines while still letting downstream code reconstruct
        the absolute path with one ``Path.__truediv__`` call. Audio
        downloads are idempotent: re-running over an existing GP only
        fetches MP3s that are missing on disk.

        Between consecutive GPs the loop sleeps for ``cfg.gp_delay_s``
        seconds (1 s by default). This is a belt-and-braces cooldown on
        top of the retry-with-backoff policy already mounted on the
        shared ``requests.Session``: the backoff handles the case where
        OpenF1 has already throttled us, and the sleep keeps the
        steady-state request rate low enough that the throttle never
        kicks in. Skipped GPs do not trigger the sleep because they
        issued zero HTTP calls.

        The ``skip_existing`` flag short-circuits a GP only when **both**
        target parquets are already on disk. If only one is present (e.g.
        because an earlier build ran before the RCM extension landed), the
        GP is rebuilt in full so the on-disk corpus stays internally
        consistent. Re-building a GP that already has both parquets is
        idempotent: it simply overwrites the existing files with fresh
        data, so the worst case of skipping incorrectly is a wasted
        round trip, never data corruption.
        """
        results: list[BuildResult] = []
        did_http_work = False
        for race in races:
            if did_http_work and self._cfg.gp_delay_s > 0:
                time.sleep(self._cfg.gp_delay_s)
            did_http_work = False

            radio_target = self._radio_target_path(race)
            rcm_target = self._rcm_target_path(race)

            if skip_existing and radio_target.exists() and rcm_target.exists():
                log.info(
                    "[%d %s] skipped — both parquets already exist",
                    race.year,
                    race.country_name,
                )
                results.append(
                    BuildResult(
                        race=race,
                        status="skipped",
                        radio_path=radio_target,
                        rcm_path=rcm_target,
                    )
                )
                continue

            try:
                # `circuit_short_name` goes into the FETCH, not only into the path
                # (#825). Resolving by country alone returns every race that country
                # holds and the resolver used to take the first, so `italy_monza/`
                # was written with Imola's messages and both `united_states_austin/`
                # and `united_states_las_vegas/` with Miami's. The path had this
                # disambiguation from the start; the fetch did not, which is the
                # one-twin-fixed shape this repo keeps paying for.
                bundle = self._builder.prepare_session_bundle(
                    race.year,
                    race.country_name,
                    circuit_short_name=race.circuit_short_name,
                )
                # Prefer the value already on the RaceMeta (captured in
                # discover_races) so the slug rule stays consistent across
                # the discovery and build phases. Fall back to the bundle's
                # session payload if discover_races somehow saw a row that
                # was missing the field.
                circuit_short_name = race.circuit_short_name or bundle.session.get(
                    "circuit_short_name"
                )
                radio_table = self._builder.build_radio_table(
                    race.year,
                    race.country_name,
                    bundle=bundle,
                )
                rcm_table = self._builder.build_rcm_table(
                    race.year,
                    race.country_name,
                    bundle=bundle,
                )
                if not self._cfg.skip_audio:
                    radio_table = self._builder.download_audio_files(
                        radio_table,
                        audio_root=self._cfg.audio_dir,
                        circuit_short_name=circuit_short_name,
                    )
                radio_path = self._builder.write_parquet(
                    radio_table,
                    race.year,
                    race.country_name,
                    circuit_short_name=circuit_short_name,
                )
                rcm_path = self._builder.write_rcm_parquet(
                    rcm_table,
                    race.year,
                    race.country_name,
                    circuit_short_name=circuit_short_name,
                )
            except Exception as exc:
                log.error(
                    "[%d %s] build failed: %s",
                    race.year,
                    race.country_name,
                    exc,
                )
                results.append(BuildResult(race=race, status="failed", error=str(exc)))
                did_http_work = True  # a failed build still hit OpenF1
                continue

            audio_rows = (
                int(radio_table["audio_path"].notna().sum())
                if "audio_path" in radio_table.columns
                else 0
            )
            results.append(
                BuildResult(
                    race=race,
                    status="built",
                    radio_rows=len(radio_table),
                    rcm_rows=len(rcm_table),
                    audio_rows=audio_rows,
                    radio_path=radio_path,
                    rcm_path=rcm_path,
                )
            )
            did_http_work = True

        return results

    def _radio_target_path(self, race: RaceMeta) -> Path:
        """Compute the radio parquet path the builder will write for a race.

        Delegates the slug + per-GP directory computation to
        :meth:`RadioDatasetBuilder._gp_directory` so the skip-existing
        check uses the exact same path the builder will write to. The
        radio half of a GP lives at
        ``{output_dir}/{year}/{slug}/radios.parquet``; this method just
        appends the role filename onto the shared per-GP directory.

        ``race.circuit_short_name`` is forwarded to the directory helper so
        multi-race countries (Italy, United States) resolve to their
        suffixed slugs (``italy_imola`` / ``united_states_miami`` / ...).
        Without this, the skip-existing check would compare against the
        legacy ``italy/`` or ``united_states/`` path and either falsely
        skip a different race or rebuild on top of the wrong slug.
        """
        gp_dir = RadioDatasetBuilder._gp_directory(
            self._cfg.output_dir,
            race.year,
            race.country_name,
            circuit_short_name=race.circuit_short_name,
        )
        return gp_dir / "radios.parquet"

    def _rcm_target_path(self, race: RaceMeta) -> Path:
        """Compute the RCM parquet path the builder will write for a race.

        The RCM analogue of :meth:`_radio_target_path`, same per-GP
        directory, but the ``rcm.parquet`` filename
        :meth:`RadioDatasetBuilder.write_rcm_parquet` writes to. Both
        targets are checked together by the skip-existing logic so a GP
        is only skipped when both halves of its corpus are already on
        disk. ``race.circuit_short_name`` is forwarded to the directory
        helper for the same multi-race-country reason as the radio path.
        """
        gp_dir = RadioDatasetBuilder._gp_directory(
            self._cfg.output_dir,
            race.year,
            race.country_name,
            circuit_short_name=race.circuit_short_name,
        )
        return gp_dir / "rcm.parquet"

    # ── Summary ──────────────────────────────────────────────────────────────

    @staticmethod
    def log_summary(results: list[BuildResult]) -> None:
        """Print a final summary of build, skip, and failure counts.

        Aggregates the per-GP outcomes into three buckets so the operator can
        tell at a glance whether the run was clean, partial, or broken.
        Reports radio and RCM totals separately because the two corpora have
        very different cardinalities (radios are sparse, RCMs scale with
        race chaos) and a single combined number would hide problems with
        either source. Lists every failed GP with its short error message
        so the next ``--years`` re-run can target only the missing slices
        via ``--gps``.
        """
        built = [r for r in results if r.status == "built"]
        skipped = [r for r in results if r.status == "skipped"]
        failed = [r for r in results if r.status == "failed"]

        total_radio_rows = sum(r.radio_rows for r in built)
        total_rcm_rows = sum(r.rcm_rows for r in built)
        total_audio_rows = sum(r.audio_rows for r in built)

        log.info("")
        log.info("=" * 60)
        log.info("Build summary")
        log.info(
            "  built:   %3d GP  (%d radios, %d RCMs, %d MP3s total)",
            len(built),
            total_radio_rows,
            total_rcm_rows,
            total_audio_rows,
        )
        log.info("  skipped: %3d GP  (already on disk)", len(skipped))
        log.info("  failed:  %3d GP", len(failed))
        log.info("=" * 60)

        if failed:
            log.info("Failed GPs:")
            for r in failed:
                log.info("  - %d %s: %s", r.race.year, r.race.country_name, r.error)

    # ── Top-level entry point ────────────────────────────────────────────────

    def run(
        self,
        years: Iterable[int],
        gp_filter: Optional[list[str]],
        skip_existing: bool,
    ) -> list[BuildResult]:
        """Discover races, filter them, build each GP, and log the summary.

        Convenience method that chains the four phases of the script into a
        single call so :func:`main` is the only place CLI arguments touch
        the orchestrator. Returns the full result list so a future test or
        Python-level caller can assert on the outcomes without scraping log
        output.
        """
        races = self.discover_races(years)
        if not races:
            log.error("No races discovered — nothing to build")
            return []

        races = self.filter_races(races, gp_filter)
        if not races:
            log.error("Race list is empty after applying --gps filter")
            return []

        log.info("Building %d races into %s", len(races), self._cfg.output_dir)
        results = self.build_all(races, skip_existing=skip_existing)
        self.log_summary(results)
        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the multi-GP radio dataset build.

    Maps every ``argparse`` value onto a fresh :class:`BuildConfig`
    instance so the CLI defaults stay co-located with the dataclass and the
    runtime ``RadioDatasetCLI`` only ever sees a fully-resolved config.
    """
    parser = argparse.ArgumentParser(
        description="Build the static OpenF1 team-radio dataset for one or more seasons.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(DEFAULT_YEARS),
        help=f"Seasons to build (default: {' '.join(str(y) for y in DEFAULT_YEARS)})",
    )
    parser.add_argument(
        "--gps",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of country names to restrict the build (case-insensitive)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CFG.output_dir,
        help=f"Directory where per-GP parquets are written (default: {CFG.output_dir})",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=CFG.audio_dir,
        help=(
            "Root directory for downloaded radio MP3s, organised as "
            "{audio_dir}/{year}/{slug}/driver_{N}/ "
            f"(default: {CFG.audio_dir})"
        ),
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help=(
            "Skip the MP3 download stage; only metadata parquets are written "
            "and the audio_path column stays null"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip GPs whose parquet already exists on disk",
    )
    parser.add_argument(
        "--gp-delay",
        type=float,
        default=CFG.gp_delay_s,
        help=(
            "Cooldown in seconds between consecutive GP builds to stay under "
            f"OpenF1's rate limit (default: {CFG.gp_delay_s}s; set to 0 to disable)"
        ),
    )
    args = parser.parse_args()

    cfg = BuildConfig(
        output_dir=args.output_dir,
        audio_dir=args.audio_dir,
        skip_audio=args.skip_audio,
        http_timeout=CFG.http_timeout,
        gp_delay_s=args.gp_delay,
    )
    cli = RadioDatasetCLI(cfg)
    cli.run(
        years=args.years,
        gp_filter=args.gps,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
