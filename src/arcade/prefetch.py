"""Fill the arcade's replay cache ahead of time, for a season or a set of rounds.

    f1-prefetch --year 2025                    # every round of the season
    f1-prefetch --year 2025 --rounds 1,3,5-8   # only those, commas and ranges
    f1-prefetch --year 2025 --with-radio       # also fetch the team radio corpus
    f1-prefetch --year 2025 --force            # prepare cached rounds too

Picking a race the arcade has never built costs 349 s measured for the
telemetry alone, plus the downloads before it (`prepare.py`). The menu runs
that on a worker thread so the window keeps drawing, but whoever picked the
race still waits the first time, for every race. This command fills the cache
in advance, unattended, so picking one later is instant.

A thin front-end to :func:`src.arcade.prepare.prepare_race`, the routine the
menu itself runs, so what this prepares and what the menu would prepare cannot
differ. The one thing added is the skip: `prepare_race` is idempotent but its
last stage still loads the pickle, ~5 s and ~380 MB of RSS for a race that
needed nothing, so a round whose pickle is already on disk is skipped before
the call is made.

Contents:
- parse_rounds: "1,3,5-8" into sorted round numbers.
- is_cached: the skip check, asked of the loader's own path rule.
- prefetch: the loop over rounds, printing progress and a summary.
- exit_code: non-zero only when nothing was achieved and something broke.
- main: the `f1-prefetch` entry point.

--- WHERE TO CHANGE IF THE CACHE LAYOUT CHANGES ---
Nothing here. The pickle path is `SessionLoader._cache_path` and the version
tag is `src.arcade.config.CACHE_VERSION`; a bump there is what `--force` is for.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import TextIO

from src.arcade.config import ARCADE_CACHE_DIR, get_gp_names
from src.arcade.data import SessionLoader
from src.arcade.prepare import PrepareProgress, RaceDataUnavailable, prepare_race

logger = logging.getLogger(__name__)


class Outcome(str, Enum):
    """What happened to one round, and the word the summary counts it under.

    `UNAVAILABLE` is the dataset's answer that the round has nothing to fetch,
    which is why it is neither a success nor a failure in `exit_code`.
    """

    PREPARED = "prepared"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


def parse_rounds(spec: str) -> list[int]:
    """Turn a rounds spec such as "1,3,5-8" into sorted, de-duplicated round numbers.

    Commas separate items and a dash marks an inclusive range. Anything else,
    an empty item, a non-number, a zero, a descending range, raises ValueError
    naming the item, so a typo dies at the parser instead of after the first
    six-minute build.
    """
    rounds: set[int] = set()
    for item in spec.split(","):
        rounds.update(_parse_round_item(item.strip()))
    return sorted(rounds)


def _parse_round_item(item: str) -> range:
    """One item of the spec, a round or an ascending range, as the rounds it names."""
    low_text, is_range, high_text = item.partition("-")
    try:
        low = int(low_text)
        high = int(high_text) if is_range else low
    except ValueError as exc:
        raise ValueError(f"not a round or a range of rounds: {item!r}") from exc
    if low < 1 or high < low:
        raise ValueError(f"rounds start at 1 and ranges ascend: {item!r}")
    return range(low, high + 1)


def is_cached(loader: SessionLoader, year: int, round_: int) -> bool:
    """Whether the replay pickle for this round is already on disk.

    Asked of the loader's own path rule so the check and the cache can never
    disagree on a name. It is one `Path.exists`, microseconds, where
    letting `prepare_race` find out costs the pickle load its last stage
    always does: ~5 s and ~380 MB of RSS for a race that needed nothing.

    Only presence is checked. The version tag lives inside the pickle, so a
    `CACHE_VERSION` bump is invisible from here, which is what `--force` is for.
    """
    return loader._cache_path(year, round_).exists()


def _emit(out: TextIO, text: str) -> None:
    """One line, flushed, because the reader may be tailing a six-minute build."""
    print(text, file=out, flush=True)


class _StageReporter:
    """Prints each stage as it starts and how long the previous one took.

    `prepare_race` reports a stage once, at its start, so a stage's duration is
    only known when the next one begins or the call returns. That is why this
    holds the last report and why `close` exists: it settles the final stage.
    """

    def __init__(self, out: TextIO) -> None:
        self._out = out
        self._current: PrepareProgress | None = None

    def __call__(self, progress: PrepareProgress) -> None:
        self._settle()
        self._current = progress
        _emit(self._out, f"  {progress.label}")

    def close(self) -> None:
        self._settle()

    def _settle(self) -> None:
        if self._current is None:
            return
        elapsed = self._current.elapsed_s(time.monotonic())
        _emit(self._out, f"  {self._current.stage}: {elapsed:.1f}s")
        self._current = None


def _attempt(
    year: int,
    round_: int,
    gp_name: str,
    *,
    with_radio: bool,
    on_progress: _StageReporter,
) -> tuple[Outcome, str]:
    """Run one preparation and turn whatever it raised into an outcome.

    `RaceDataUnavailable` is the dataset saying the round has nothing to fetch,
    so it is counted rather than raised. Anything else is a genuine failure,
    logged with its traceback, and the loop moves on: one bad round must not
    end an unattended run. Returns the outcome and the detail worth printing.
    """
    try:
        prepare_race(year, round_, gp_name, strategy_enabled=with_radio, on_progress=on_progress)
    except RaceDataUnavailable as exc:
        return Outcome.UNAVAILABLE, str(exc)
    except Exception as exc:  # noqa: BLE001 - counted and reported, the run goes on
        logger.exception("Preparing %d round %d (%s) failed", year, round_, gp_name)
        return Outcome.FAILED, f"{type(exc).__name__}: {exc}"
    else:
        return Outcome.PREPARED, ""


def prefetch_round(
    year: int,
    round_: int,
    gp_name: str,
    *,
    with_radio: bool,
    force: bool,
    loader: SessionLoader,
    out: TextIO,
) -> Outcome:
    """Prepare one round unless its pickle is already there, and say what happened."""
    if not force and is_cached(loader, year, round_):
        _emit(out, "  cached, skipped")
        return Outcome.SKIPPED

    reporter = _StageReporter(out)
    started = time.monotonic()
    outcome, detail = _attempt(year, round_, gp_name, with_radio=with_radio, on_progress=reporter)
    reporter.close()
    elapsed = time.monotonic() - started
    suffix = f": {detail}" if detail else ""
    _emit(out, f"  {outcome.value} in {elapsed:.1f}s{suffix}")
    return outcome


def prefetch(
    year: int,
    rounds: Sequence[int],
    *,
    with_radio: bool,
    force: bool,
    cache_dir: Path = ARCADE_CACHE_DIR,
    out: TextIO | None = None,
) -> Counter[Outcome]:
    """Prepare each round in turn and print a summary; returns the count per outcome.

    The GP name comes from the same calendar the menu resolves it from, so the
    two can only ever fetch the same race for a given round. `cache_dir` and
    `out` exist for the tests: the real cache holds pickles that cost 349 s
    each to rebuild, and nothing here may touch them through a fake.
    """
    out = out or sys.stdout
    calendar = get_gp_names(year)
    loader = SessionLoader(cache_dir=cache_dir)
    counts: Counter[Outcome] = Counter()
    started = time.monotonic()

    for position, round_ in enumerate(rounds, start=1):
        gp_name = calendar.get(round_, f"Round{round_}")
        _emit(out, f"[{position}/{len(rounds)}] {year} R{round_:02d} {gp_name}")
        outcome = prefetch_round(
            year, round_, gp_name, with_radio=with_radio, force=force, loader=loader, out=out
        )
        counts[outcome] += 1

    tally = ", ".join(f"{counts[outcome]} {outcome.value}" for outcome in Outcome)
    _emit(out, f"Done in {time.monotonic() - started:.1f}s: {tally}")
    return counts


def exit_code(counts: Counter[Outcome]) -> int:
    """Non-zero only when the run achieved nothing and something genuinely broke.

    A skipped round is a success, its cache is already filled, and an
    unavailable round is an answer rather than a failure. So a season where
    every round is unavailable exits 0, and so does a run that prepared one
    race and failed on five; the summary line carries the counts either way.
    """
    achieved = counts[Outcome.PREPARED] + counts[Outcome.SKIPPED]
    if achieved == 0 and counts[Outcome.FAILED] > 0:
        return 1
    return 0


def _rounds_argument(spec: str) -> list[int]:
    """`parse_rounds` for argparse, which only relays a message raised as its own type."""
    try:
        return parse_rounds(spec)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="f1-prefetch",
        description=(
            "Fill the arcade's replay cache ahead of time, so picking a race from the "
            "menu later is instant instead of a 349 s build. Rounds already cached are "
            "skipped without being loaded."
        ),
    )
    parser.add_argument("--year", type=int, required=True, help="Season to prepare, e.g. 2025.")
    parser.add_argument(
        "--rounds",
        type=_rounds_argument,
        default=None,
        metavar="SPEC",
        help='Rounds to prepare, commas and ranges: "1,3,5-8". Default: the whole calendar.',
    )
    parser.add_argument(
        "--with-radio",
        action="store_true",
        help=(
            "Also fetch the team radio corpus (~3 MB per race). Off by default because "
            "the replay never reads it; only the agents do."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Prepare cached rounds too. A pickle at the current CACHE_VERSION is reloaded "
            "rather than rebuilt (~5 s), so this is for a cache invalidated by a version bump."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the flags, refuse rounds off the calendar, run, and return the exit code.

    The calendar check happens before anything is prepared, on the reasoning
    `prepare_race` orders its own stages by: a typo should fail now, not after
    the first six-minute build.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    calendar = get_gp_names(args.year)
    rounds = args.rounds or sorted(calendar)
    off_calendar = [round_ for round_ in rounds if round_ not in calendar]
    if off_calendar:
        parser.error(f"not on the {args.year} calendar (1-{max(calendar)}): {off_calendar}")

    counts = prefetch(args.year, rounds, with_radio=args.with_radio, force=args.force)
    return exit_code(counts)


if __name__ == "__main__":
    sys.exit(main())
