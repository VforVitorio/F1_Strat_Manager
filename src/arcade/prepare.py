"""Everything a race needs on disk before the replay and the agents can run.

Three artefacts, and until #1115 the arcade fetched exactly one of them:

- ``data/raw/{year}/{gp}/**``, the FastF1 race dump. Read by the strategy layer
  and by PITWALL. **Nothing fetched it.** `ensure_race` was written for this,
  exported, and documented in `src/f1_strat_manager/README.md` as the lazy
  per-race pull, and its only mention anywhere in `src/` was a log line telling
  the user to run it by hand. Its sibling `ensure_radio_corpus` DID get a call
  site, in `scripts/run_simulation_cli.py`, which is this repo's dominant defect
  shape: one of a pair wired and its twin not.
- ``data/raw/radio_audio/{year}/{slug}/**``, the team radio the radio agent
  transcribes. Only needed when the agents run.
- ``data/cache/arcade/{gp}_{year}_race.pkl``, the replay telemetry. This one the
  arcade already built on demand, which is why picking any GP from the menu drew
  a circuit and gave you an empty AGENTS window: the half the eye sees was lazy
  and the half the agents need was not.

The other reason this module exists is TIME. Building the Lusail 2025 pickle
takes 349 s measured, and the fetches add to it, so the preparation cannot run
on the thread that draws: the menu used to force one frame of "Loading
session..." and then block pyglet for minutes, which Windows paints as a dead
window. Everything here is plain I/O returning plain data, so a worker thread
can run it while the menu keeps drawing, and the caller builds the GL objects on
the main thread once it is done.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for typing
    from src.arcade.data import SessionData

logger = logging.getLogger(__name__)

# What the user is told is happening, in the order it happens. The downloads go
# first because they are the steps that can fail on a missing race, and failing
# in 5 s beats failing after a 349 s telemetry build.
STAGE_RACE_DATA = "Downloading race data"
STAGE_TEAM_RADIO = "Downloading team radio"
STAGE_TELEMETRY = "Building telemetry"

ProgressFn = Callable[["PrepareProgress"], None]


class RaceDataUnavailable(RuntimeError):
    """The dataset publishes nothing for this race, so the agents cannot run.

    Its own type because the menu treats it as an answer rather than a crash:
    the race simply is not available, which is worth saying plainly instead of
    surfacing a stack trace.
    """


@dataclass(frozen=True)
class PrepareProgress:
    """One step of the preparation, as the UI should show it.

    `index` is 1-based and `total` is fixed for the run, so a caller can render
    "2/3" without knowing which stages exist. `elapsed_s` is the time spent in
    THIS stage, which is the only honest progress available: the downloads go
    through `huggingface_hub`'s own tqdm and expose no callback, and the
    telemetry build reports nothing at all.
    """

    stage: str
    index: int
    total: int
    started_at: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.stage}  ({self.index}/{self.total})"

    @property
    def done_fraction(self) -> float:
        """Share of the run FINISHED, which is the stages before this one.

        `index / total` would read 100% the moment the last stage starts, and
        the last stage is the 349 s telemetry build, so the bar would sit full
        for the entire wait (#1116).
        """
        return (self.index - 1) / self.total

    def elapsed_s(self, now: float) -> float:
        """Seconds in this stage, counted by the caller's clock.

        A field would be frozen at whatever it held when the worker reported,
        which is zero: each stage reports once, at its start. The menu redraws
        sixty times a second and can count, and a number that climbs is the
        only thing on screen saying the window is alive rather than hung.
        """
        return max(0.0, now - self.started_at)


def race_stages(*, strategy_enabled: bool) -> tuple[str, ...]:
    """The stages this run will go through, so the UI can size itself up front.

    The radio corpus is only fetched when the agents are going to run: the
    replay itself never reads it, and it is ~3 MB a user should not pay for
    watching a race back.
    """
    if strategy_enabled:
        return (STAGE_RACE_DATA, STAGE_TEAM_RADIO, STAGE_TELEMETRY)
    return (STAGE_RACE_DATA, STAGE_TELEMETRY)


def prepare_race(
    year: int,
    round_: int,
    gp_name: str,
    *,
    strategy_enabled: bool,
    on_progress: ProgressFn | None = None,
) -> SessionData:
    """Fetch what this race needs and return its loaded telemetry.

    Safe to call on a worker thread: it touches the filesystem and the network
    and returns a plain dataclass, with no GL object created anywhere. The
    caller builds `Track` and `F1ArcadeView` on the main thread from the result.

    Idempotent. Every fetch short-circuits on a populated directory, so a warm
    race pays a few milliseconds of `Path.exists` and the pickle load.
    """
    from src.arcade.data import SessionLoader
    from src.f1_strat_manager.data_cache import ensure_race, ensure_radio_corpus

    stages = race_stages(strategy_enabled=strategy_enabled)
    total = len(stages)

    def report(stage: str, started: float) -> None:
        if on_progress is None:
            return
        on_progress(
            PrepareProgress(
                stage=stage,
                index=stages.index(stage) + 1,
                total=total,
                started_at=started,
            )
        )

    started = time.monotonic()
    report(STAGE_RACE_DATA, started)
    race_dir = ensure_race(year, gp_name, show_progress=False)
    if not race_dir.exists() or not any(race_dir.iterdir()):
        # `snapshot_download` on a pattern that matches nothing returns quietly,
        # so without this a race the dataset does not hold looks identical to
        # one it does, right up until the agents window sits empty. Refusing
        # here puts the reason in front of whoever picked it (#1116).
        raise RaceDataUnavailable(
            f"No data published for {gp_name} {year}. "
            f"The dataset has no {race_dir.name} folder for that season."
        )
    logger.info("Race data ready at %s (%.1fs)", race_dir, time.monotonic() - started)

    if strategy_enabled:
        started = time.monotonic()
        report(STAGE_TEAM_RADIO, started)
        audio_dir = ensure_radio_corpus(year, gp_name, show_progress=False)
        logger.info("Team radio ready at %s (%.1fs)", audio_dir, time.monotonic() - started)

    started = time.monotonic()
    report(STAGE_TELEMETRY, started)
    session_data = SessionLoader().load(year, round_, gp_name)
    logger.info(
        "Telemetry ready: %s %d, %d drivers, %d frames (%.1fs)",
        session_data.location or gp_name,
        year,
        len(session_data.frames_by_driver),
        session_data.total_frames,
        time.monotonic() - started,
    )
    return session_data
