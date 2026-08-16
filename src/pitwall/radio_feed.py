"""The RADIO channel: a race's team radio and race control, revealed lap by lap.

The DATA window's fourth panel is a chronological feed of what was said during
the race - the RaceX client's `Race Control Messages` list, merged with the
driver radio its own tab carries separately.

**It is read from disk and not from the wire, and that is a correction to the
sprint brief rather than a preference.** `strategy.per_agent.radio` really does
carry `radio_events` and `rcm_events` with the original transcripts, but
`StrategyState.snapshot_dict` strips `per_agent` from every entry of
`history_tail`, so the wire holds the CURRENT lap and nothing else. Building a
chronological feed from that means accumulating client-side, and this window
has a rule against exactly that (`useBulk.ts`: "a grow-only cache would survive
a seek to the end"). Accumulation would also start empty on a mid-race attach,
hole at 8x where laps are skipped, and vanish entirely when the producer runs
without `--strategy`, which the DATA window does not otherwise need.

The corpus is static parquet known before lap 1, so it is a progressive reveal
masked by the clock - the same argument that shaped `session_data.py`, and the
same shape, so the two cannot drift apart.

Three properties this module exists to guarantee:

1. **The reveal is a THIRD coordinate and it is coarse on purpose.** A driver
   radio from lap L shows once that driver has completed lap L; a race-control
   message from lap L shows once the LEADER has. Never early, at most one lap
   late. It cannot be finer: both parquets stamp events in UTC while every
   clock in this window is SessionTime, and bridging them needs FastF1's
   `t0_date` - the blocker #931 and #842 are already filed against. The panel
   states the rule on screen rather than inventing an anchor.
2. **Nothing here decides the TIER.** Which driver is ours lives on the tick
   (`arcade.driver_main`), which is not part of the bulk's signature, so a
   payload that baked the tier in would not be determined by the signature that
   serves it - the defect #934 cost a sprint. The renderer holds the tick and
   labels there.
3. **The reader is the corpus reader the CLI already uses.** Constructing
   `RadioPipelineRunner` with `eager_transcribe=False` loads no model - Whisper
   comes in lazily through `_get_whisper` - and it owns the parquet paths, the
   transcript cache and its backslash/forward-slash key convention, whose other
   half is the WRITER. Re-implementing any of that here would be a second copy.

--- WHERE TO CHANGE IF THE CORPUS MOVES ---
`RadioPipelineRunner` owns every path: the parquets under
`data/processed/race_radios/{year}/{slug}/` and the transcript cache under
`data/processed/radio_nlp/{year}/{slug}/`. The slug is a third GP keyspace and
`src/f1_strat_manager/gp_slugs.py` is its single source of truth.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.nlp.radio_runner import RadioPipelineRunner
from src.pitwall.session_data import race_dir

logger = logging.getLogger(__name__)


def _codes_by_number(laps_frame: pd.DataFrame) -> dict[int, str]:
    """`{car number: driver code}` from the RAW lap table.

    Keyed on the number as an INT on both sides. The parquet holds it as a
    string because "07" is a real car number and `int()` would print it "7" -
    but the radio corpus holds the same number as an integer, so normalising
    both through `int` is what makes the two meet at all, and the code is what
    gets displayed anyway.
    """
    if "DriverNumber" not in laps_frame or "Driver" not in laps_frame:
        return {}
    pairs = laps_frame[["DriverNumber", "Driver"]].dropna().drop_duplicates()
    codes: dict[int, str] = {}
    for row in pairs.itertuples():
        try:
            codes[int(row.DriverNumber)] = str(row.Driver)
        except (TypeError, ValueError):
            continue
    return codes


def _transcript_text(runner: RadioPipelineRunner, audio_path: Any) -> str:
    """The cached transcript for one radio row, or an empty string.

    Empty is the runner's own contract for "no usable text"
    (`_transcribe_one` writes an empty entry rather than dropping the row), and
    it is the COMMON case here: 23 of the 24 races on the Hub have no
    transcribed audio and there is not one MP3 on disk. The event still
    renders - that a radio happened, from whom and on which lap is information.
    """
    if audio_path is None or pd.isna(audio_path):
        return ""
    # Private only by name. It is the shared convention between this reader and
    # the cache's WRITER - the parquet stores the path with native separators
    # and the JSON keys it with forward slashes - so calling it is the whole
    # point and copying it would be the twin.
    key = RadioPipelineRunner._normalize_audio_path(str(audio_path))
    entry = runner.transcripts.get(key) or {}
    return str(entry.get("text") or "")


def _rcm_text(row: Any) -> str:
    """A race-control row's message, falling back to its bare flag."""
    message = str(getattr(row, "message", "") or "").strip()
    if message:
        return message
    return str(getattr(row, "flag", "") or "").strip()


class RadioCorpus:
    """One race's radio and RCM events, ordered once and sliced on every read.

    Immutable after construction. Each event is held beside the driver code
    whose lap counter gates it - `None` meaning the leader's - so the reveal is
    DATA rather than a branch that has to stay in step with `kind`. Sprint 5's
    most expensive defect was a guard that looked like the reveal rule and
    checked something adjacent to it.
    """

    def __init__(self, gated: list[tuple[str | None, dict[str, Any]]]) -> None:
        self._gated = gated

    @classmethod
    def load(cls, data_root: Path, year: int, location: str) -> RadioCorpus | None:
        """Read the race's corpus, or None when there is none on disk.

        Resolved through the race FOLDER's name rather than the wire's raw
        `location`, because the two disagree and only one of them resolves.
        FastF1's 2025 Location is "Miami Gardens" while the folder is
        `Miami_Gardens` and `gp_slugs.FOLDER_ALIASES` is keyed on the
        underscore form - so `race_dir` does the disk-level resolution once and
        this reads the answer off it.

        Returns None rather than raising for every miss, including the
        `ValueError` `resolve_gp_slug` throws at an unknown GP: on a curated
        install a missing corpus is the COMMON case, so it is absent data and
        not a failed operation. The same reasoning as `SessionLaps.load`.
        """
        directory = race_dir(data_root, year, location)
        if directory is None:
            return None
        laps_frame = pd.read_parquet(directory / "laps.parquet")
        try:
            runner = RadioPipelineRunner(
                year=year,
                gp_name=directory.name,
                # Deliberately EMPTY, and typed so the runner's own mapper
                # answers `{}` quietly instead of logging a failure. It builds
                # `{number: code}` by slicing on `GP_Name`, a column only the
                # FEATURED parquet carries, so handing it the raw table would
                # log "Could not build driver code map" on every race switch
                # about a map this reader does not use: PITWALL takes its codes
                # from `_codes_by_number` below, off the same frame the tower's
                # rows come from, which is what makes the two agree.
                laps_df=pd.DataFrame(columns=["GP_Name", "DriverNumber", "Driver"]),
                data_root=data_root,
                eager_transcribe=False,
            )
        except ValueError:
            # An unknown GP name. The runner raises so a CLI typo cannot
            # produce a silent zero-radio simulation; here the same input is a
            # season we have no corpus for, which is data we do not have.
            logger.info("No radio corpus slug for %s %s", year, location)
            return None
        if runner.total_radios() == 0 and runner.total_rcms() == 0:
            logger.info("Radio corpus empty for %s %s", year, location)
            return None
        return cls(cls._ordered_events(runner, _codes_by_number(laps_frame)))

    @staticmethod
    def _ordered_events(
        runner: RadioPipelineRunner,
        codes_by_number: dict[int, str],
    ) -> list[tuple[str | None, dict[str, Any]]]:
        """Every event of the race, in order, each beside its reveal driver.

        **The codes come from the RAW lap table, not from the runner's own
        map, and the difference was a silent total loss.** The runner builds
        `{number: code}` off the FEATURED parquet, which carries a `GP_Name`
        column to slice on; the raw `laps.parquet` this window reads has no
        such column, so its mapper caught its own exception, logged a warning
        and returned `{}` - and every radio came out as the synthetic `D12`,
        which no reveal map and no tower row can ever match. Executed on
        Melbourne: **0 of 14 driver radios were ever revealed** while all 90
        race-control rows were, so the panel would have looked like a working
        RCM feed on a race with no team radio at all. Reading the codes off the
        same frame the tower's come from also guarantees the two agree.

        **The order is by LAP, then by the parquet's UTC stamp within it.** The
        stamp is unusable as an ABSOLUTE anchor against this window's
        SessionTime clock, which is what blocks a finer reveal - but it orders
        two events of the same lap perfectly well, and that is all it is asked
        to do here. (No row is missing one: 0 of 2,126 across the 24 races on
        disk.

        ⚠️ **A future null would NOT "sort first within its lap, never crash",
        which is what this said.** Executed: a `NaT` compares False against
        everything, so it does not sink to one end - it lands where it happens
        to and takes its neighbours' order with it, rendering 05:01 before
        05:00. A plain `None` does not sort at all: `'<' not supported between
        instances of 'NoneType' and 'Timestamp'`. The corpus has no such row
        today, so this is a latent shape rather than a rate - but the sentence
        that said it was harmless was wrong twice over.)
        """
        ordered: list[tuple[tuple[int, Any], str | None, dict[str, Any]]] = []
        for row in runner.radios_df.itertuples():
            code = codes_by_number.get(int(row.driver_number))
            if code is None:
                # A car number with no lap rows - a reserve entry, or corpus
                # drift. It cannot be placed on this clock: gating it on the
                # leader would show it EARLIER than the driver it belongs to
                # ever reaches, which is the one direction the reveal must
                # never fail in. Dropped and said out loud.
                logger.info(
                    "Radio from car %s on lap %s has no driver in the lap table - skipped",
                    row.driver_number,
                    row.lap_number,
                )
                continue
            payload = {
                "kind": "radio",
                "lap": int(row.lap_number),
                "driver": code,
                "text": _transcript_text(runner, row.audio_path),
                "category": None,
                "flag": None,
            }
            ordered.append(((int(row.lap_number), row.date), code, payload))
        for row in runner.rcm_df.itertuples():
            payload = {
                "kind": "rcm",
                "lap": int(row.lap_number),
                # The message already names its car, so a code here would print
                # it twice. Measured across all 24 races on disk: of the 492
                # rows carrying a `driver_number`, 492 spell that number inside
                # the text ("WAVED BLUE FLAG FOR CAR 87 (BEA) ..."), code
                # included. The key stays so both kinds of event have one shape
                # and the renderer needs no optional-key branch.
                "driver": None,
                "text": _rcm_text(row),
                "category": str(getattr(row, "category", "") or "") or None,
                "flag": str(getattr(row, "flag", "") or "") or None,
            }
            # None, never the car the message names: race control issues it on
            # the RACE's lap, and a row names a car only because the flag or
            # the penalty is about it.
            ordered.append(((int(row.lap_number), row.date), None, payload))
        ordered.sort(key=lambda item: item[0])
        return [(gate, payload) for _key, gate, payload in ordered]

    def masked_view(self, laps_completed: dict[str, int]) -> dict[str, Any]:
        """The feed as of the clock: every event whose lap is over.

        A rewind LOWERS `laps_completed` and therefore shortens this list,
        because it is recomputed rather than accumulated. That is the whole
        reason the feed is read from disk instead of gathered off the wire.
        """
        leader_lap = max(laps_completed.values(), default=0)
        events = [
            event
            for driver, event in self._gated
            if event["lap"] <= (leader_lap if driver is None else laps_completed.get(driver, 0))
        ]
        result = {"available": True, "events": events}
        return result


def unavailable() -> dict[str, Any]:
    """The payload for a race with no corpus, so the panel can say so.

    An explicit state rather than an empty one, for the reason `session_data`
    already spells out: a feed rendering nothing silently is the same pixel as
    a feed whose reveal is broken.
    """
    result = {"available": False, "events": []}
    return result
