"""Radio dataset builder — src/data_extraction/openf1/radio_dataset_builder.py

Production-grade wrapper around the OpenF1 team-radio pipeline prototyped in
``notebooks/nlp/N33_radio_dataset_builder.ipynb``. The notebook validates the
single-GP happy path; this module hardens the same logic for static multi-GP
builds that will feed the N29 Radio Agent and any future offline replay tool.

What this module does
---------------------
For a given ``(year, country_name)`` pair it resolves the OpenF1 session_key,
pulls every team radio in the race, maps each radio to the race lap it was
transmitted on (interval matching against per-driver ``/v1/laps`` start +
duration), and drops the radios that cannot inform any in-race strategy
decision: unmapped timestamps, formation lap, race-start lap, and everything
from the chequered-flag lap onwards. The filtered table is persisted to a
parquet under ``output_dir`` so downstream NLP and agent code can consume it
without ever touching OpenF1 again.

For the same GP it also pulls ``/v1/race_control`` and writes a parallel
parquet of lap-mapped Race Control Messages: yellow/green/red flags, safety
car deployments, sector warnings, black-and-orange car notices, and any
other event the FIA broadcasts during the race. RCMs are mapped to laps
the same way radios are — OpenF1's own ``lap_number`` is used when present,
otherwise interval matching falls back to the leader's intervals for
track-wide events or to the targeted driver's intervals for car-specific
ones. The same structural filter (lap not in {0, 1}, lap < total_laps)
applies, so cool-down "TRACK CLEAR" and pre-race procedural notices are
dropped before they ever reach the N29 Radio Agent.

What this module does NOT do
----------------------------
No MP3 download, no Whisper/Nemotron transcription, no sentiment/intent/NER
inference. Those steps live in dedicated future modules and in the existing
N24 pipeline. This builder is strictly the "raw lap-mapped metadata" layer
that produces the inputs the N29 Radio Agent consumes at runtime.

The public entry point for the upcoming ``scripts/build_radio_dataset.py`` CLI
wrapper is the :class:`RadioDatasetBuilder` class, which reuses a single
``requests.Session`` across calls for connection pooling and keeps logging
quiet by default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Module-level constants ───────────────────────────────────────────────────
# OpenF1 REST base URL. Centralised so tests or mirrors can monkeypatch it in
# one place instead of hunting through every request call.
OPENF1_BASE: str = "https://api.openf1.org/v1"

# Default retry policy for OpenF1 HTTP calls. OpenF1 throttles aggressively
# when a client re-hits the same session_key in quick succession (our smoke
# tests triggered it), so every GET retries up to five times on 429 and the
# 5xx family with an exponential backoff (1s, 2s, 4s, 8s, 16s = 31s total
# worst case per request before giving up). ``Retry.respect_retry_after_header``
# honours OpenF1's own ``Retry-After`` value when present so we back off for
# at least as long as the server asks us to.
RETRY_TOTAL: int = 5
RETRY_BACKOFF_FACTOR: float = 1.0
RETRY_STATUS_FORCELIST: tuple[int, ...] = (429, 500, 502, 503, 504)

# Laps with no strategic value. Lap 0 is the formation lap, lap 1 is the
# flying-start chaos window: neither carries any actionable pit/tyre call so
# every radio sent during them is dropped before transcription.
RACE_START_LAPS: frozenset[int] = frozenset({0, 1})

# When True, every radio on the chequered-flag lap and beyond (cool-down,
# congratulatory exchanges) is discarded. The chequered-flag lap equals
# total_laps for the leader, so the strict filter is ``lap_number < total_laps``.
DROP_LAST_LAP: bool = True

# Canonical output schema — the 10 columns the downstream NLP/agent code relies
# on. Exported as a tuple so callers can assert DataFrame shape without copying
# a literal list around. Keep this in sync with ``build_radio_table``.
#
# ``audio_path`` is the relative path of the downloaded MP3 under the audio
# root configured by the CLI (e.g. ``2025/bahrain/driver_44/HAM_044_xxx.mp3``).
# It is populated by :meth:`RadioDatasetBuilder.download_audio_files` after
# the metadata table is built; ``build_radio_table`` itself emits ``None``
# for this column so the schema stays consistent even when the caller skips
# the audio download step (in-memory smoke tests, dry runs, etc.).
OUTPUT_SCHEMA: tuple[str, ...] = (
    "session_key",
    "meeting_key",
    "year",
    "gp",
    "total_laps",
    "driver_number",
    "lap_number",
    "date",
    "recording_url",
    "audio_path",
)

# Countries that host more than one Grand Prix in the same season. The default
# slug rule (``country_name.lower().replace(" ", "_")``) collides for these
# because two distinct races share the same country, so an early build of the
# 2025 calendar overwrote Imola with Monza under ``italy/`` and silently kept
# only one of {Miami, Austin, Las Vegas} under ``united_states/``. The
# :meth:`RadioDatasetBuilder._gp_directory` helper appends the OpenF1
# ``circuit_short_name`` (lowercased) to the country slug **only** for these
# countries, producing ``italy_imola`` / ``italy_monza`` /
# ``united_states_miami`` / ``united_states_cota`` etc. while leaving every
# single-race country (Bahrain, Australia, ...) on the cheap legacy slug. The
# choice is keyed on the country alone — not the year — because the only way
# this set changes is if the FIA adds another double-header country, at which
# point the constant gets one more entry and a single rebuild fixes the affected
# slugs without touching any single-race GP on disk.
_MULTI_RACE_COUNTRIES: frozenset[str] = frozenset({"Italy", "United States"})


# Canonical RCM output schema. Mirrors the OpenF1 ``/v1/race_control`` payload
# plus the same session metadata columns as the radio schema, so downstream
# code that joins radios + RCMs by ``(session_key, lap_number)`` does not need
# to special-case either source. ``driver_number``, ``sector`` and ``flag``
# are nullable because most RCMs are track-wide and many carry no flag at all
# (e.g. informational "TRACK SURFACE INSPECTION" notices).
RCM_OUTPUT_SCHEMA: tuple[str, ...] = (
    "session_key",
    "meeting_key",
    "year",
    "gp",
    "total_laps",
    "driver_number",
    "lap_number",
    "date",
    "category",
    "flag",
    "scope",
    "sector",
    "message",
)


logger = logging.getLogger(__name__)


def build_retry_session() -> requests.Session:
    """Build a ``requests.Session`` that retries OpenF1 rate-limit responses.

    OpenF1 throttles clients that re-hit the same session_key in quick
    succession — the smoke test hit a 429 the second time it ran within
    the same minute, and a naive full-season build would trip the limit
    even harder because it touches every session_key on the calendar in a
    tight loop. Mounting a retry-enabled :class:`HTTPAdapter` at the
    session level fixes this transparently: every ``GET`` issued through
    the session inherits the backoff, so neither the builder nor the CLI
    loop has to reason about rate limits at the call site.

    The policy retries on 429 and the 5xx transient-server family with an
    exponential backoff (1s, 2s, 4s, 8s, 16s = 31s worst-case cumulative
    wait before giving up per request). ``respect_retry_after_header``
    honours any ``Retry-After`` value OpenF1 supplies so we back off for
    at least as long as the server asks us to before the client timer
    kicks back in. The factory is kept as a module-level free function so
    both :class:`RadioDatasetBuilder` (when the caller does not inject its
    own session) and the multi-GP CLI wrapper share the exact same policy
    without duplicating the configuration block.
    """
    session = requests.Session()
    retry = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=list(RETRY_STATUS_FORCELIST),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@dataclass(frozen=True)
class SessionBundle:
    """Pre-fetched session metadata + lap intervals shared between table builds.

    Building a radio table and an RCM table for the same GP both need the
    ``/v1/sessions`` metadata and the ``/v1/laps`` interval index. Constructing
    this bundle once and passing it to both build methods cuts the HTTP cost
    in half (from 4 calls per GP to 2) without forcing the methods to lose
    their standalone usability — each one still works when called without a
    bundle by building one internally on demand.

    Marked frozen so an accidental mutation in one build method cannot poison
    the bundle for the next caller, and so the bundle can be safely passed
    across thread boundaries if a future version parallelises GP builds.

    Attributes:
        session:    Raw OpenF1 session dict from ``/v1/sessions``, carrying
                    ``session_key``, ``meeting_key``, ``country_name`` and
                    ``date_start``. Forwarded into the metadata columns of
                    both output tables so downstream joins do not need a
                    second lookup.
        laps_index: Per-driver lookup of ``(lap_number, start, end)`` intervals
                    as produced by :meth:`RadioDatasetBuilder.fetch_laps_index`.
                    Used by both the radio and the RCM build methods to
                    interval-match a UTC timestamp to its race lap.
        total_laps: Highest ``lap_number`` across the whole session — the
                    leader's chequered-flag lap. Used as the right edge of
                    the structural filter (``lap_number < total_laps``).
    """

    session: dict
    laps_index: dict
    total_laps: int


class RadioDatasetBuilder:
    """Stateful builder that turns OpenF1 team radios into a lap-mapped parquet.

    The builder owns a single ``requests.Session`` for the entire multi-GP run,
    which matters because a full-season build touches ``/v1/sessions``,
    ``/v1/team_radio`` and ``/v1/laps`` for every round — reusing TCP
    connections cuts the wall-clock cost noticeably and reduces the chance of
    tripping OpenF1's rate limiter. It also centralises the HTTP timeout and
    the output directory so callers do not have to thread those arguments
    through every helper.

    The class deliberately exposes small single-responsibility methods
    (``resolve_session``, ``fetch_team_radio``, ``fetch_laps_index``,
    ``build_radio_table``, ``write_parquet``, ``build_and_write``) so the
    upcoming ``scripts/build_radio_dataset.py`` CLI wrapper can compose them
    freely — e.g. build in memory and inspect before writing, or write a
    single GP without re-running the full season loop.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        http_timeout: int = 30,
        session: Optional[requests.Session] = None,
    ) -> None:
        """Configure the builder with the output directory and HTTP settings.

        ``output_dir`` is the root under which per-GP parquets are written.
        The directory is created lazily on the first ``write_parquet`` call so
        constructing the builder never has filesystem side effects — useful
        for tests and dry runs. A typical value is
        ``data/processed/team_radio`` relative to the repo root.

        ``http_timeout`` bounds every OpenF1 request in seconds. The default
        of 30 matches the notebook and gives slow connections enough headroom
        on the ``/v1/laps`` payload, which is the heaviest of the three
        endpoints.

        ``session`` lets the caller inject a pre-configured
        ``requests.Session`` (useful for tests with a mock transport adapter,
        or for wiring retry/backoff middleware). When omitted the builder
        creates its own session via :func:`build_retry_session`, which
        mounts an ``HTTPAdapter`` configured to retry OpenF1's 429 and 5xx
        responses with exponential backoff — this is what keeps the
        smoke test and the multi-GP CLI from crashing the first time the
        public API throttles a rapid sequence of calls. An injected
        session is used as-is so callers that need a custom retry policy
        (tests, custom mirrors) stay in full control.
        """
        self._output_dir: Path = Path(output_dir)
        self._http_timeout: int = http_timeout
        self._session: requests.Session = session if session is not None else build_retry_session()

    # ── Public properties ────────────────────────────────────────────────────

    @property
    def output_dir(self) -> Path:
        """Return the directory where per-GP parquets will be written.

        Exposed read-only so callers (CLI wrapper, tests) can inspect the
        configured destination without reaching into private state. The
        directory itself is only materialised on the first ``write_parquet``
        call, so reading this property is side-effect free.
        """
        return self._output_dir

    # ── HTTP helpers (one responsibility each) ───────────────────────────────

    def resolve_session(
        self,
        year: int,
        country_name: str,
        session_type: str = "Race",
        *,
        circuit_short_name: Optional[str] = None,
    ) -> dict:
        """Look up the OpenF1 session metadata for a specific GP race session.

        The OpenF1 schema keys everything by ``session_key``, so this lookup
        is the entry point for every downstream call in the pipeline. The
        ``country_name`` argument must match the value OpenF1 uses internally
        (e.g. ``"Bahrain"`` or ``"United Kingdom"``) — a mismatched string
        silently returns an empty list and the method raises ``ValueError``
        to surface the problem early instead of propagating an opaque
        empty-radio build.

        On Sprint weekends OpenF1 returns **two** entries with
        ``session_type="Race"`` for the same country (the Saturday Sprint
        and the Sunday Grand Prix), and ``sessions[0]`` is whichever one
        OpenF1 happens to list first. The early 2025 China build hit
        exactly this footgun and overwrote the main-race parquet with the
        Sprint payload. To prevent that, the method filters the response
        client-side by ``session_name == "Race"`` whenever the caller is
        asking for the main race, so the Sprint sibling is never selected
        even on weekends where it shares the country name with the GP.

        A **second** and larger ``sessions[0]`` hazard sits behind that one, and it
        is what #825 was: several countries host more than one race in a season
        (Italy = Imola + Monza, United States = Miami + Austin + Las Vegas), so
        ``country_name`` alone does not name a race even after the Sprint filter.
        Whichever OpenF1 listed first won, and three 2025 races were built on
        another race's radio and RCM corpus — Monza inheriting Imola's Virtual
        Safety Car among them. ``circuit_short_name`` resolves it; see
        :meth:`_disambiguate_by_circuit`, which raises rather than guessing when
        the country is ambiguous and no circuit was given.

        Args:
            circuit_short_name: OpenF1's own circuit label (``"Monza"``,
                ``"Miami"``). REQUIRED for a country with more than one race in
                ``year``; optional otherwise. Comes from ``/v1/sessions`` itself,
                so callers that already discovered the race have it to hand.

        Raises:
            ValueError: no session for this country/year, no main Race session
                on a Sprint weekend, or an ambiguous country with no
                ``circuit_short_name`` to disambiguate it.

        Returns:
            The full session dict, so the caller can also read ``meeting_key``,
            ``date_start`` and ``circuit_short_name`` without a second round trip
            to ``/v1/sessions``.
        """
        response = self._session.get(
            f"{OPENF1_BASE}/sessions",
            params={
                "year": year,
                "country_name": country_name,
                "session_type": session_type,
            },
            timeout=self._http_timeout,
        )
        response.raise_for_status()
        sessions = response.json()
        if not sessions:
            raise ValueError(f"No {session_type} session found for {country_name} {year}")
        # Sprint weekends: drop the Sprint sibling so we never accidentally
        # pick it up instead of the main GP race.
        if session_type == "Race":
            main_race = [s for s in sessions if s.get("session_name") == "Race"]
            if not main_race:
                raise ValueError(
                    f"No main Race session (session_name='Race') found for "
                    f"{country_name} {year}; got "
                    f"{[s.get('session_name') for s in sessions]}"
                )
            return self._disambiguate_by_circuit(main_race, year, country_name, circuit_short_name)
        # The twin. Nothing calls this with a non-Race type today (`build_and_write`,
        # `prepare_session_bundle` and the __main__ demo all take the default), so a
        # bare `sessions[0]` here was latent rather than live - and latent is exactly
        # how #825 started. A country with two circuits is ambiguous for Practice and
        # Qualifying too, so the same resolution applies.
        return self._disambiguate_by_circuit(sessions, year, country_name, circuit_short_name)

    @staticmethod
    def _disambiguate_by_circuit(
        sessions: list,
        year: int,
        country_name: str,
        circuit_short_name: Optional[str],
    ) -> dict:
        """Pick the one session at ``circuit_short_name`` when a country holds several.

        Italy runs Imola and Monza; the United States runs Miami, Austin and Las
        Vegas. A query keyed on country alone returns all of them and taking
        ``[0]`` silently gave every sibling the SAME session (#825): `italy_monza/`
        held Imola's messages and both `united_states_austin/` and
        `united_states_las_vegas/` held Miami's. Monza has no neutralisation of its
        own in 2025 and was being served Imola's **Virtual** Safety Car (deployed on
        lap 29, ending on lap 31) as a result. Calling that a Safety Car, as this
        docstring first did, names the wrong mechanism: `rcm_state.py` tracks the two
        separately (`sc_kind == "VSC"`) and Art. 56 makes a VSC materially different
        for the value of a pit stop, a distinction #471 already paid for. Both set
        `sc_active`, so the conclusion here is unchanged; the label was not.

        The output PATH already disambiguated by circuit (``_gp_directory`` takes
        ``circuit_short_name`` for exactly this reason, and its docstring says so).
        The FETCH did not: one half of the pair got the fix and its twin did not,
        which is this repository's most repeated defect shape.

        Raises rather than guessing when the country is ambiguous and the caller
        gave no circuit. A wrong-but-plausible corpus is the failure mode that cost
        three races; an exception is the one that costs a re-run.
        """
        if circuit_short_name:
            matched = [
                s
                for s in sessions
                if str(s.get("circuit_short_name", "")).lower() == circuit_short_name.lower()
            ]
            if not matched:
                raise ValueError(
                    f"No Race session at circuit {circuit_short_name!r} for "
                    f"{country_name} {year}; OpenF1 offers "
                    f"{[s.get('circuit_short_name') for s in sessions]}"
                )
            return matched[0]

        if len(sessions) > 1:
            raise ValueError(
                f"{country_name} {year} has {len(sessions)} Race sessions "
                f"({[s.get('circuit_short_name') for s in sessions]}); pass "
                "circuit_short_name to say which one. Selecting the first is how "
                "three 2025 races ended up with another race's corpus (#825)."
            )
        return sessions[0]

    def fetch_team_radio(self, session_key: int) -> pd.DataFrame:
        """Pull every team radio for a session, one row per radio message.

        Returns a DataFrame with the OpenF1 fields ``date`` (UTC timestamp of
        the radio), ``driver_number`` and ``recording_url``. The ``date``
        column is parsed to a tz-aware pandas Timestamp because the
        downstream lap-mapping step compares it against the UTC
        ``date_start`` values returned by ``/v1/laps`` and a tz mismatch
        would silently drop every radio in the session.

        When OpenF1 returns an empty radio list for an otherwise valid
        session (very old races and some atypical sessions have no radios
        archived), this method returns an empty DataFrame instead of
        crashing, so the multi-GP loop in ``build_radio_table`` can still
        emit an empty parquet and move on.
        """
        response = self._session.get(
            f"{OPENF1_BASE}/team_radio",
            params={"session_key": session_key},
            timeout=self._http_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return pd.DataFrame(columns=["date", "driver_number", "recording_url", "session_key"])
        radios = pd.DataFrame(payload)
        # OpenF1 mixes ISO8601 variants across races (some timestamps include
        # microseconds, some do not). format="ISO8601" accepts both shapes
        # without crashing on the format auto-inference path.
        radios["date"] = pd.to_datetime(radios["date"], utc=True, format="ISO8601")
        return radios

    def fetch_laps_index(
        self,
        session_key: int,
    ) -> dict[int, list[tuple[int, pd.Timestamp, pd.Timestamp]]]:
        """Build a per-driver lookup of ``(lap_number, start, end)`` intervals.

        OpenF1 ``/v1/laps`` returns absolute UTC start times plus
        ``lap_duration`` in seconds, which is exactly what interval-matching
        each radio to its transmission lap needs. The lookup is keyed by
        ``driver_number`` because two drivers on the same nominal lap number
        have slightly different timestamps (gap to the leader), and using the
        wrong driver's intervals would produce off-by-one lap assignments
        near the start/finish line.

        Rows with missing ``date_start`` or ``lap_duration`` are dropped
        silently — OpenF1 occasionally emits partial lap records for the
        out-lap of a retired car, and keeping them would corrupt the
        intervals. Each driver's interval list is sorted by start time so
        callers can use a simple linear scan (which is what
        ``assign_lap_to_radio`` does).
        """
        response = self._session.get(
            f"{OPENF1_BASE}/laps",
            params={"session_key": session_key},
            timeout=self._http_timeout,
        )
        response.raise_for_status()
        laps = pd.DataFrame(response.json())
        if laps.empty:
            return {}
        # See fetch_team_radio comment — OpenF1 mixes ISO8601 variants across
        # the season, and laps is the heaviest endpoint so a single bad
        # microsecond row would otherwise abort the whole build for that GP.
        laps["date_start"] = pd.to_datetime(
            laps["date_start"],
            utc=True,
            format="ISO8601",
        )
        laps = laps.dropna(subset=["date_start", "lap_duration"])
        index: dict[int, list[tuple[int, pd.Timestamp, pd.Timestamp]]] = {}
        for driver, group in laps.groupby("driver_number"):
            intervals: list[tuple[int, pd.Timestamp, pd.Timestamp]] = []
            for _, row in group.iterrows():
                start = row["date_start"]
                end = start + pd.Timedelta(seconds=row["lap_duration"])
                intervals.append((int(row["lap_number"]), start, end))
            intervals.sort(key=lambda item: item[1])
            index[int(driver)] = intervals
        return index

    def fetch_race_control(self, session_key: int) -> pd.DataFrame:
        """Pull every Race Control Message for a session, one row per event.

        Returns a DataFrame with the OpenF1 ``/v1/race_control`` fields:
        ``date`` (UTC timestamp of the message), ``driver_number`` (nullable
        — most RCMs are track-wide and have no targeted car), ``lap_number``
        (nullable — OpenF1 supplies it for many but not all events), plus
        ``category``, ``flag``, ``scope``, ``sector`` and the raw ``message``
        text. The ``date`` column is parsed to a tz-aware Timestamp because
        the downstream lap-mapping fallback compares it against the UTC
        ``date_start`` values returned by ``/v1/laps``.

        When OpenF1 returns an empty payload (very old races, atypical
        sessions, or freshly imported rounds with delayed RCM ingest), the
        method returns an empty DataFrame with the relevant columns instead
        of crashing, so the multi-GP loop in :meth:`build_rcm_table` can
        still emit an empty parquet and move on.
        """
        response = self._session.get(
            f"{OPENF1_BASE}/race_control",
            params={"session_key": session_key},
            timeout=self._http_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return pd.DataFrame(
                columns=[
                    "date",
                    "driver_number",
                    "lap_number",
                    "category",
                    "flag",
                    "scope",
                    "sector",
                    "message",
                ]
            )
        rcms = pd.DataFrame(payload)
        # /v1/race_control is the endpoint where the ISO8601 inconsistency
        # actually shows up in 2025 data — some RCMs ship with microseconds
        # (".173000+00:00"), others without ("+00:00"), and pandas' default
        # format inference picks the first row's shape and crashes on the
        # rest. format="ISO8601" handles both variants in a single pass.
        rcms["date"] = pd.to_datetime(rcms["date"], utc=True, format="ISO8601")
        return rcms

    @staticmethod
    def assign_lap_to_radio(
        radio_date: pd.Timestamp,
        driver_number: int,
        laps_index: dict[int, list[tuple[int, pd.Timestamp, pd.Timestamp]]],
    ) -> Optional[int]:
        """Find the lap number whose interval contains the radio timestamp.

        Returns ``None`` when the radio falls outside any known lap window —
        that happens for pre-formation messages, post-chequered-flag
        exchanges, and the rare timing gaps between consecutive laps. The
        caller drops these rows because there is no race lap to attach them
        to, and keeping them would force downstream code to reason about a
        ``NaN`` lap number.

        Implemented as a static method because it needs no builder state: the
        laps index is the only input besides the radio itself. Keeping it
        static also makes it trivial to unit-test without constructing a full
        ``RadioDatasetBuilder``.
        """
        intervals = laps_index.get(driver_number, [])
        for lap_number, start, end in intervals:
            if start <= radio_date < end:
                return lap_number
        return None

    @staticmethod
    def assign_lap_to_rcm(
        rcm_date: pd.Timestamp,
        driver_number: Optional[int],
        laps_index: dict[int, list[tuple[int, pd.Timestamp, pd.Timestamp]]],
    ) -> Optional[int]:
        """Map an RCM to its lap, falling back to the leader for track-wide events.

        Driver-specific RCMs (``driver_number`` is set) use that driver's own
        intervals via :meth:`assign_lap_to_radio` — the same logic that maps
        team radios — because a black-and-orange notice or driver penalty
        applies to one car and the corresponding lap should be that car's
        lap, not the leader's. Track-wide RCMs (``driver_number`` is ``None``,
        which is the case for most yellow flags and safety car deployments)
        fall back to the intervals of the driver with the most laps in the
        session, i.e. the leader at the moment the message was issued. This
        gives a stable anchor that does not depend on which car happens to
        be physically nearest the start/finish line at the RCM's timestamp.

        Returns ``None`` when neither path produces a match, which happens
        for genuine pre-race or post-race messages — the caller drops them
        the same way it drops unmappable radios.
        """
        if driver_number is not None:
            return RadioDatasetBuilder.assign_lap_to_radio(rcm_date, driver_number, laps_index)
        if not laps_index:
            return None
        leader = max(laps_index.keys(), key=lambda d: len(laps_index[d]))
        return RadioDatasetBuilder.assign_lap_to_radio(rcm_date, leader, laps_index)

    # ── Orchestration ────────────────────────────────────────────────────────

    def prepare_session_bundle(
        self,
        year: int,
        country_name: str,
        *,
        circuit_short_name: Optional[str] = None,
    ) -> SessionBundle:
        """Resolve session metadata + lap intervals in a single fetch pair.

        Both :meth:`build_radio_table` and :meth:`build_rcm_table` need the
        ``/v1/sessions`` payload and the ``/v1/laps`` interval index to do
        their work. Calling this method once and passing the resulting
        :class:`SessionBundle` to both build methods cuts the HTTP cost from
        four calls per GP to two, which matters when the multi-GP CLI loop
        is iterating over the full season.

        Raises ``ValueError`` when ``/v1/laps`` returns nothing for the
        resolved session, because that means there is no way to compute
        ``total_laps`` and the structural filter would silently keep every
        post-race message. Surfacing the error early lets the CLI loop log
        the failure and continue with the next GP instead of producing a
        corrupted parquet.
        """
        session = self.resolve_session(
            year, country_name, "Race", circuit_short_name=circuit_short_name
        )
        session_key = int(session["session_key"])
        laps_index = self.fetch_laps_index(session_key)
        if not laps_index:
            raise ValueError(
                f"No laps returned by OpenF1 for session_key={session_key} "
                f"({country_name} {year}); cannot compute total_laps"
            )
        total_laps = max(lap for intervals in laps_index.values() for lap, _, _ in intervals)
        return SessionBundle(
            session=session,
            laps_index=laps_index,
            total_laps=total_laps,
        )

    def build_radio_table(
        self,
        year: int,
        country_name: str,
        *,
        bundle: Optional[SessionBundle] = None,
        circuit_short_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run the full pipeline for one GP and return a filtered radio table.

        The method orchestrates the helpers in order: resolve the session
        bundle (or reuse one passed in), pull radios, then map each radio to
        its lap and drop the structurally useless rows (unmapped, formation
        lap, race-start lap, chequered-flag lap and beyond). The remaining
        rows are guaranteed to belong to a normal race lap and therefore
        carry information that could plausibly inform a strategy decision.

        The optional ``bundle`` argument lets the multi-GP CLI loop pre-fetch
        the session metadata and laps index once and reuse them across both
        :meth:`build_radio_table` and :meth:`build_rcm_table`, halving the
        HTTP cost per GP. When omitted, the method builds the bundle itself
        so the standalone single-GP usage from the notebook smoke test stays
        unchanged.

        When the GP has no radios at all (empty ``/v1/team_radio`` payload),
        the method returns an empty DataFrame with the canonical
        ``OUTPUT_SCHEMA`` columns and logs a warning. This lets a multi-GP
        loop in the CLI wrapper still emit an empty parquet for that round
        instead of crashing the whole build.

        HTTP errors from OpenF1 (via ``raise_for_status``) propagate up
        unchanged so the caller can distinguish transient network issues
        from legitimately empty sessions.
        """
        if bundle is None:
            bundle = self.prepare_session_bundle(
                year, country_name, circuit_short_name=circuit_short_name
            )
        session_key = int(bundle.session["session_key"])

        radios = self.fetch_team_radio(session_key)

        if radios.empty:
            logger.warning(
                "[%s %d] no team radios returned by OpenF1 (session_key=%d); emitting empty table",
                country_name,
                year,
                session_key,
            )
            return self._empty_output_frame()

        radios["lap_number"] = radios.apply(
            lambda row: self.assign_lap_to_radio(
                row["date"], row["driver_number"], bundle.laps_index
            ),
            axis=1,
        )

        n_total = len(radios)

        # Filter 1 — drop unmapped (pre-race / post-race / timing gaps)
        radios = radios.dropna(subset=["lap_number"])
        radios["lap_number"] = radios["lap_number"].astype(int)
        n_after_unmapped = len(radios)

        # Filter 2 — drop race start (formation + lap 1) and chequered-flag lap
        keep_mask = ~radios["lap_number"].isin(RACE_START_LAPS)
        if DROP_LAST_LAP:
            keep_mask &= radios["lap_number"] < bundle.total_laps
        radios = radios[keep_mask].reset_index(drop=True)
        n_after_structural = len(radios)

        # Attach session metadata so downstream joins do not need a second lookup
        radios["session_key"] = session_key
        radios["meeting_key"] = bundle.session["meeting_key"]
        radios["year"] = year
        radios["gp"] = country_name
        radios["total_laps"] = bundle.total_laps

        # ``audio_path`` is populated by download_audio_files when the caller
        # asks for the MP3 download step. Default it to None so the parquet
        # always conforms to OUTPUT_SCHEMA even when the caller skips audio.
        radios["audio_path"] = None

        logger.info(
            "[%s %d] total radios: %d",
            country_name,
            year,
            n_total,
        )
        logger.info(
            "[%s %d] after dropping unmapped: %d (-%d)",
            country_name,
            year,
            n_after_unmapped,
            n_total - n_after_unmapped,
        )
        logger.info(
            "[%s %d] after structural filter: %d (-%d)",
            country_name,
            year,
            n_after_structural,
            n_after_unmapped - n_after_structural,
        )
        logger.info(
            "[%s %d] total race laps in GP: %d",
            country_name,
            year,
            bundle.total_laps,
        )

        return radios[list(OUTPUT_SCHEMA)]

    def build_rcm_table(
        self,
        year: int,
        country_name: str,
        *,
        bundle: Optional[SessionBundle] = None,
        circuit_short_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run the full pipeline for one GP and return a filtered RCM table.

        Mirrors :meth:`build_radio_table` for race control messages: resolve
        the bundle (or reuse one passed in), pull RCMs from
        ``/v1/race_control``, map each event to its lap, and drop the
        structurally useless rows. The lap-mapping rule is slightly more
        forgiving than for radios: OpenF1 supplies ``lap_number`` directly
        for many RCMs, and the method honours that value when present
        instead of re-deriving it. Only RCMs without an OpenF1-provided lap
        fall through to :meth:`assign_lap_to_rcm`, which uses the targeted
        driver's intervals when available and the leader's intervals
        otherwise.

        After mapping, the same structural filter as the radio pipeline
        applies (lap not in ``RACE_START_LAPS``, lap ``< total_laps``), so
        formation-lap procedural messages and post-race "TRACK CLEAR"
        notices are dropped before they reach downstream consumers.

        Returns an empty :data:`RCM_OUTPUT_SCHEMA`-shaped DataFrame when
        OpenF1 returns no race control messages for the session — sessions
        with no incidents at all do exist (rare but real) and the multi-GP
        loop must still emit a parquet for them so the on-disk layout stays
        complete.

        Schema columns missing from the OpenF1 payload (older races
        sometimes omit ``sector`` or ``scope``) are added as ``None`` so the
        output always conforms to :data:`RCM_OUTPUT_SCHEMA` regardless of
        the source payload's exact shape.
        """
        if bundle is None:
            bundle = self.prepare_session_bundle(
                year, country_name, circuit_short_name=circuit_short_name
            )
        session_key = int(bundle.session["session_key"])

        rcms = self.fetch_race_control(session_key)

        if rcms.empty:
            logger.warning(
                "[%s %d] no race control messages returned by OpenF1 "
                "(session_key=%d); emitting empty RCM table",
                country_name,
                year,
                session_key,
            )
            return self._empty_rcm_frame()

        # Honour OpenF1's own lap_number when present, otherwise interval-match.
        # The fallback path uses the targeted driver's intervals for car-specific
        # events and the leader's intervals for track-wide ones.
        def _resolve_lap(row: pd.Series) -> Optional[int]:
            openf1_lap = row.get("lap_number")
            if pd.notna(openf1_lap):
                return int(openf1_lap)
            driver = row.get("driver_number")
            driver_int = int(driver) if pd.notna(driver) else None
            return self.assign_lap_to_rcm(row["date"], driver_int, bundle.laps_index)

        rcms["lap_number"] = rcms.apply(_resolve_lap, axis=1)

        n_total = len(rcms)

        # Filter 1 — drop unmapped RCMs (pre-race FIA bulletins, post-race
        # cool-down "TRACK SURFACE INSPECTION" notices, etc.)
        rcms = rcms.dropna(subset=["lap_number"])
        rcms["lap_number"] = rcms["lap_number"].astype(int)
        n_after_unmapped = len(rcms)

        # Filter 2 — drop race start (formation + lap 1) and chequered-flag lap
        keep_mask = ~rcms["lap_number"].isin(RACE_START_LAPS)
        if DROP_LAST_LAP:
            keep_mask &= rcms["lap_number"] < bundle.total_laps
        rcms = rcms[keep_mask].reset_index(drop=True)
        n_after_structural = len(rcms)

        # Attach session metadata so downstream joins do not need a second lookup
        rcms["session_key"] = session_key
        rcms["meeting_key"] = bundle.session["meeting_key"]
        rcms["year"] = year
        rcms["gp"] = country_name
        rcms["total_laps"] = bundle.total_laps

        # Backfill any schema columns the OpenF1 payload happened to omit so
        # the parquet always conforms to RCM_OUTPUT_SCHEMA regardless of round
        for col in RCM_OUTPUT_SCHEMA:
            if col not in rcms.columns:
                rcms[col] = None

        logger.info(
            "[%s %d] total RCMs: %d",
            country_name,
            year,
            n_total,
        )
        logger.info(
            "[%s %d] after dropping unmapped RCMs: %d (-%d)",
            country_name,
            year,
            n_after_unmapped,
            n_total - n_after_unmapped,
        )
        logger.info(
            "[%s %d] after structural RCM filter: %d (-%d)",
            country_name,
            year,
            n_after_structural,
            n_after_unmapped - n_after_structural,
        )

        return rcms[list(RCM_OUTPUT_SCHEMA)]

    # ── Audio download ───────────────────────────────────────────────────────

    def download_audio_files(
        self,
        table: pd.DataFrame,
        audio_root: Path,
        *,
        overwrite: bool = False,
        circuit_short_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Download every radio MP3 referenced in ``table`` under ``audio_root``.

        Each row's ``recording_url`` is fetched once and saved to
        ``{audio_root}/{year}/{slug}/driver_{N}/{basename}.mp3``, where
        ``slug`` is produced by :meth:`_compute_slug` (matching the parquet
        filename convention exactly, including the multi-race-country
        disambiguation for Italy and the United States) and ``basename`` is
        the filename component of the URL itself. The URL basenames OpenF1
        emits already include the driver name, number, and a UTC timestamp
        (``HAMILTON_44_20250413_151002.mp3``), so they are unique within a
        session and the filename is fully deterministic — re-running the
        build over the same parquet is idempotent and existing files are
        skipped unless ``overwrite=True``.

        ``circuit_short_name`` is required to disambiguate Italy and the
        United States; for any other country it is ignored. The slug is
        computed once before the row loop because every row in ``table``
        belongs to the same GP — recomputing it per row would just be busy
        work.

        The ``audio_path`` column on the returned DataFrame is populated
        with the path **relative to** ``audio_root``, so the parquet stays
        portable across machines: a consumer that knows the audio root can
        reconstruct the absolute path with a single ``Path.__truediv__``
        without depending on where the build was run. Rows whose download
        fails (network error, dead URL) get ``audio_path=None`` and the
        method continues with the next row instead of aborting the whole
        GP — a few missing MP3s are recoverable, an aborted multi-GP build
        is not.

        Returns the DataFrame with ``audio_path`` populated. Empty input is
        handled gracefully (no directories are created) so dry-run paths
        and zero-radio sessions stay side-effect free.
        """
        if table.empty:
            return table

        audio_paths: list[Optional[str]] = []
        n_downloaded = 0
        n_skipped = 0
        n_failed = 0

        # Compute the slug once for the whole table — every row belongs to
        # the same GP so the country name is constant across the loop.
        first_row = table.iloc[0]
        gp_country = str(first_row["gp"])
        slug = self._compute_slug(gp_country, circuit_short_name)

        # Stream downloads through the same retry-enabled session as the
        # metadata fetches, so 429 throttling on the static MP3 host is
        # absorbed by the same backoff policy without any extra glue.
        for _, row in table.iterrows():
            year = int(row["year"])
            driver = int(row["driver_number"])
            url = str(row["recording_url"])

            filename = Path(url).name or f"{driver}_{int(row['lap_number'])}.mp3"

            target_dir = audio_root / str(year) / slug / f"driver_{driver}"
            target = target_dir / filename
            relative = target.relative_to(audio_root)

            if target.exists() and not overwrite:
                n_skipped += 1
                audio_paths.append(str(relative))
                continue

            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                resp = self._session.get(
                    url,
                    timeout=self._http_timeout,
                    stream=True,
                )
                resp.raise_for_status()
                with target.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
                n_downloaded += 1
                audio_paths.append(str(relative))
            except Exception as exc:  # noqa: BLE001 — log + continue per row
                logger.error(
                    "Failed to download radio MP3 %s: %s",
                    url,
                    exc,
                )
                n_failed += 1
                audio_paths.append(None)

        table = table.copy()
        table["audio_path"] = audio_paths

        logger.info(
            "audio download: %d new, %d cached, %d failed (root=%s)",
            n_downloaded,
            n_skipped,
            n_failed,
            audio_root,
        )
        return table

    # ── Persistence ──────────────────────────────────────────────────────────

    @staticmethod
    def _compute_slug(
        country_name: str,
        circuit_short_name: Optional[str] = None,
    ) -> str:
        """Return the on-disk slug for a single ``(country, circuit)`` GP.

        For the ~18 single-race countries on the calendar the slug is just
        ``country_name.lower().replace(" ", "_")`` (``bahrain``,
        ``united_kingdom``, ``saudi_arabia``, …). For the countries listed
        in :data:`_MULTI_RACE_COUNTRIES` — currently Italy (Imola + Monza)
        and the United States (Miami + Austin + Las Vegas) — the slug is
        suffixed with the lowercased ``circuit_short_name`` so the two/three
        races never collide on disk: ``italy_imola`` / ``italy_monza`` /
        ``united_states_miami`` / ``united_states_cota`` etc.

        The single-place implementation is reused by both
        :meth:`_gp_directory` (parquet path) and :meth:`download_audio_files`
        (MP3 path) so the metadata tree and the audio tree always agree on
        the slug for any given GP. Falls back to the bare country slug when
        ``circuit_short_name`` is missing, even for a multi-race country —
        this keeps the helper safe to call from legacy code paths that
        haven't been threaded through yet, at the cost of being unable to
        disambiguate until the caller is updated.
        """
        country_slug = country_name.lower().replace(" ", "_")
        if country_name in _MULTI_RACE_COUNTRIES and circuit_short_name:
            circuit_slug = circuit_short_name.lower().replace(" ", "_")
            return f"{country_slug}_{circuit_slug}"
        return country_slug

    @staticmethod
    def _gp_directory(
        output_dir: Path,
        year: int,
        country_name: str,
        *,
        circuit_short_name: Optional[str] = None,
    ) -> Path:
        """Compute the per-GP subdirectory under the output root.

        The radio + RCM parquets for a single GP live together in
        ``{output_dir}/{year}/{slug}/``, where ``slug`` is computed by
        :meth:`_compute_slug` so the rule is shared with the audio
        download path. For single-race countries the slug is just the
        lowercased country name; for multi-race countries (Italy, United
        States) the OpenF1 ``circuit_short_name`` is appended so Imola
        and Monza (or Miami / Austin / Las Vegas) never overwrite each
        other under a shared ``italy/`` or ``united_states/`` folder.

        The metadata tree under ``data/processed/race_radios/`` and the
        MP3 tree under ``data/raw/radio_audio/`` share an identical
        ``{year}/{slug}/`` substructure: a consumer that knows the GP
        builds both paths with the same fragment.

        Kept as a static method so the path computation lives in exactly
        one place — the CLI's skip-existing logic can call it without
        instantiating a builder, and a future test can assert on the
        layout without going through ``write_parquet``.
        """
        slug = RadioDatasetBuilder._compute_slug(country_name, circuit_short_name)
        return output_dir / str(year) / slug

    def write_parquet(
        self,
        table: pd.DataFrame,
        year: int,
        country_name: str,
        *,
        circuit_short_name: Optional[str] = None,
    ) -> Path:
        """Persist a built radio table to the per-GP subdirectory.

        The on-disk layout is ``{output_dir}/{year}/{slug}/radios.parquet``
        where ``slug`` is produced by :meth:`_compute_slug`: the lowercased
        country for single-race countries, and ``country_circuit`` for the
        multi-race countries listed in :data:`_MULTI_RACE_COUNTRIES`. The
        per-GP folder mirrors the audio download tree
        (``data/raw/radio_audio/{year}/{slug}/driver_{N}/``) so the radio
        + RCM corpus and the MP3 corpus share an identical
        ``{year}/{slug}/`` substructure that downstream consumers can
        construct from a single fragment. The radio and RCM parquets are
        named by **role** (``radios.parquet`` / ``rcm.parquet``) rather
        than by year and slug because both pieces of metadata are already
        encoded in the path itself, which avoids redundant naming like
        ``2025_bahrain/2025_bahrain.parquet``. The parent directory is
        created lazily on the first call so constructing a builder stays
        side-effect free.

        ``circuit_short_name`` is required to disambiguate Italy and the
        United States; for any other country the value is ignored. Callers
        that hold a :class:`SessionBundle` already have it in
        ``bundle.session["circuit_short_name"]``.

        Returns the absolute path of the written parquet so the caller can
        log it, print it from the CLI, or feed it to a follow-up step
        without reconstructing the filename.
        """
        gp_dir = self._gp_directory(
            self._output_dir,
            year,
            country_name,
            circuit_short_name=circuit_short_name,
        )
        gp_dir.mkdir(parents=True, exist_ok=True)
        path = gp_dir / "radios.parquet"
        table.to_parquet(path, index=False)
        logger.info(
            "[%s %d] wrote %d rows to %s",
            country_name,
            year,
            len(table),
            path,
        )
        return path

    def write_rcm_parquet(
        self,
        table: pd.DataFrame,
        year: int,
        country_name: str,
        *,
        circuit_short_name: Optional[str] = None,
    ) -> Path:
        """Persist a built RCM table to the per-GP subdirectory.

        Sibling of :meth:`write_parquet`: writes to
        ``{output_dir}/{year}/{slug}/rcm.parquet`` so the radio and RCM
        parquets for a single GP live in the same folder. Loading both
        halves for a GP becomes a one-liner that just appends the role
        filename to the per-GP path, with no string templating on year or
        slug at the call site. The parent directory is created lazily on
        the first call, exactly like the radio variant, so a dry run that
        fails before any successful write does not pollute the filesystem.

        ``circuit_short_name`` is forwarded to :meth:`_gp_directory` so
        the multi-race-country disambiguation rule (Italy, United States)
        is applied identically to the radio parquet — keeping both halves
        of a GP under the same suffixed slug.

        Returns the absolute path of the written parquet for the same
        reasons as :meth:`write_parquet` — logging, summaries, and
        follow-up steps that consume the freshly built file.
        """
        gp_dir = self._gp_directory(
            self._output_dir,
            year,
            country_name,
            circuit_short_name=circuit_short_name,
        )
        gp_dir.mkdir(parents=True, exist_ok=True)
        path = gp_dir / "rcm.parquet"
        table.to_parquet(path, index=False)
        logger.info(
            "[%s %d] wrote %d RCM rows to %s",
            country_name,
            year,
            len(table),
            path,
        )
        return path

    def build_and_write(
        self,
        year: int,
        country_name: str,
        *,
        circuit_short_name: Optional[str] = None,
    ) -> tuple[Path, Path]:
        """Build both the radio and RCM tables for a GP and write both parquets.

        Convenience wrapper for the common CLI path: prepares the session
        bundle once via :meth:`prepare_session_bundle`, then runs both build
        methods sharing the same bundle so the GP only costs two HTTP fetches
        for the shared bits (sessions + laps) plus one each for radios and
        race control. Writes both parquets into the per-GP subdirectory
        ``{output_dir}/{year}/{slug}/`` as ``radios.parquet`` and
        ``rcm.parquet`` (see :meth:`write_parquet` for the rationale of the
        layout) and returns both paths as a tuple so a multi-GP loop can
        record them in a build summary without reconstructing the filenames.
        """
        bundle = self.prepare_session_bundle(
            year, country_name, circuit_short_name=circuit_short_name
        )
        # Pull the circuit name from the already-fetched session payload so
        # both writers can apply the multi-race-country disambiguation rule
        # without a second /v1/sessions round trip. Without the argument above
        # this path could only ever reach the FIRST race of a multi-race country:
        # it is self-consistent (the folder is derived from the session it really
        # fetched, so it cannot mis-file) but it could never build Monza, Austin
        # or Las Vegas at all.
        circuit_short_name = bundle.session.get("circuit_short_name")
        radio_table = self.build_radio_table(year, country_name, bundle=bundle)
        rcm_table = self.build_rcm_table(year, country_name, bundle=bundle)
        radio_path = self.write_parquet(
            radio_table,
            year,
            country_name,
            circuit_short_name=circuit_short_name,
        )
        rcm_path = self.write_rcm_parquet(
            rcm_table,
            year,
            country_name,
            circuit_short_name=circuit_short_name,
        )
        return radio_path, rcm_path

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _empty_output_frame() -> pd.DataFrame:
        """Return an empty DataFrame with the canonical ``OUTPUT_SCHEMA`` columns.

        Used when a session has zero team radios: downstream consumers still
        expect the 9-column schema, so handing them a bare ``pd.DataFrame()``
        would break any ``.loc[:, col]`` access. Keeping the empty-frame
        construction in one helper guarantees the schema stays consistent
        with ``build_radio_table``'s non-empty return.
        """
        return pd.DataFrame({col: pd.Series(dtype="object") for col in OUTPUT_SCHEMA})

    @staticmethod
    def _empty_rcm_frame() -> pd.DataFrame:
        """Return an empty DataFrame with the canonical ``RCM_OUTPUT_SCHEMA`` columns.

        The RCM analogue of :meth:`_empty_output_frame` — used when a session
        has zero race control messages so callers still get a fully-shaped
        13-column DataFrame instead of a bare ``pd.DataFrame()``. Keeping
        the empty-frame construction in one helper guarantees the schema
        stays consistent with :meth:`build_rcm_table`'s non-empty return.
        """
        return pd.DataFrame({col: pd.Series(dtype="object") for col in RCM_OUTPUT_SCHEMA})


if __name__ == "__main__":
    # Smoke test — mirrors the notebook's Bahrain 2025 demo so a developer
    # can run `python -m src.data_extraction.openf1.radio_dataset_builder` and
    # verify the module works end-to-end without setting up a notebook.
    # Exercises the radio build, the RCM build, and the audio download path
    # so all three halves of the on-disk contract are checked in a single
    # smoke run. Everything writes to a tmpdir so the smoke test never
    # touches the real data/ tree.
    import tempfile

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        builder = RadioDatasetBuilder(output_dir=tmp_root / "parquets")

        # Share the session bundle so the smoke test exercises the same
        # one-fetch path the multi-GP CLI loop uses.
        bundle = builder.prepare_session_bundle(year=2025, country_name="Bahrain")

        bahrain_radios = builder.build_radio_table(
            year=2025,
            country_name="Bahrain",
            bundle=bundle,
        )
        print("RADIOS (pre-audio):")
        print(bahrain_radios.head(10))
        print(f"radio rows: {len(bahrain_radios)} | columns: {list(bahrain_radios.columns)}")
        print()

        bahrain_rcms = builder.build_rcm_table(
            year=2025,
            country_name="Bahrain",
            bundle=bundle,
        )
        print("RCMs:")
        print(bahrain_rcms.head(10))
        print(f"rcm rows: {len(bahrain_rcms)} | columns: {list(bahrain_rcms.columns)}")
        print()

        bahrain_radios = builder.download_audio_files(
            bahrain_radios,
            audio_root=tmp_root / "audio",
            circuit_short_name=bundle.session.get("circuit_short_name"),
        )
        print("RADIOS (post-audio):")
        print(bahrain_radios[["lap_number", "driver_number", "audio_path"]].head(10))
        print(
            f"audio populated: {int(bahrain_radios['audio_path'].notna().sum())}/"
            f"{len(bahrain_radios)}"
        )
