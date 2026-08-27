"""A race is fetched before it is loaded, for every race (#1115).

`data/raw/{year}/{gp}/**` is what the strategy layer and PITWALL read, and until
this landed nothing fetched it. `ensure_race` existed, was exported, and was
documented in `src/f1_strat_manager/README.md` as the lazy per-race pull; its
only mention anywhere in `src/` was a log line telling the user to run it by
hand. Its sibling `ensure_radio_corpus` DID have a call site. So picking any GP
but the one already on disk drew a circuit and gave you an empty AGENTS window.

**Asserted on the orchestration, not on a download.** The fetch helpers are
`snapshot_download` wrappers and the telemetry build takes minutes, so what is
checked here is that they are called, in the right order, with the right
arguments, and that the caller is told about each one.
"""

from __future__ import annotations

import pytest

from src.arcade import prepare
from src.arcade.prepare import (
    STAGE_RACE_DATA,
    STAGE_TEAM_RADIO,
    STAGE_TELEMETRY,
    PrepareProgress,
    RaceDataUnavailable,
    prepare_race,
    race_stages,
)


class _Recorder:
    """Stands in for the three slow calls and records the order they came in.

    `ensure_race` returns a real populated directory because `prepare_race`
    inspects it: `snapshot_download` on a pattern that matches nothing returns
    an empty one quietly, and refusing that is the point of the check.
    """

    def __init__(self, tmp_path) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.race_dir = tmp_path / "raw"
        self.race_dir.mkdir(parents=True, exist_ok=True)
        (self.race_dir / "laps.parquet").write_bytes(b"x")

    def ensure_race(self, year, gp_name, show_progress=True):
        self.calls.append(("ensure_race", (year, gp_name, show_progress)))
        return self.race_dir

    def ensure_radio_corpus(self, year, gp_name, show_progress=True):
        self.calls.append(("ensure_radio_corpus", (year, gp_name, show_progress)))
        return self.race_dir

    def loader(self):
        recorder = self

        class _Loader:
            def load(self, year, round_, gp_name):
                recorder.calls.append(("load", (year, round_, gp_name)))
                return _Session()

        return _Loader

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


class _Session:
    frames_by_driver: dict = {}
    total_frames = 0
    location = "Lusail"


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch, tmp_path) -> _Recorder:
    """Patch the three slow calls at the module `prepare_race` imports them from."""
    rec = _Recorder(tmp_path)
    import src.arcade.data as arcade_data
    import src.f1_strat_manager.data_cache as data_cache

    monkeypatch.setattr(data_cache, "ensure_race", rec.ensure_race)
    monkeypatch.setattr(data_cache, "ensure_radio_corpus", rec.ensure_radio_corpus)
    monkeypatch.setattr(arcade_data, "SessionLoader", rec.loader())
    return rec


def test_the_race_data_is_fetched_before_anything_reads_it(recorder: _Recorder) -> None:
    """The whole defect, in one assertion.

    Nothing called `ensure_race`, so `data/raw/{year}/{gp}/` stayed empty for
    every race but the one shipped in the curated download, and the strategy
    layer degraded silently against it.
    """
    prepare_race(2025, 23, "Lusail", strategy_enabled=True)
    assert "ensure_race" in recorder.names
    assert recorder.names.index("ensure_race") < recorder.names.index("load")


def test_the_downloads_come_before_the_long_build(recorder: _Recorder) -> None:
    """Failing in seconds beats failing after a 349 s telemetry build.

    That is the measured cold cost of `SessionLoader().load` for Lusail 2025, so
    ordering the cheap failable steps first is the difference between a user
    learning a race is unavailable now and learning it six minutes from now.
    """
    prepare_race(2025, 23, "Lusail", strategy_enabled=True)
    assert recorder.names == ["ensure_race", "ensure_radio_corpus", "load"]


def test_the_radio_is_only_fetched_when_the_agents_will_run(recorder: _Recorder) -> None:
    """The replay never reads the audio, so watching a race back must not pay for it."""
    prepare_race(2025, 23, "Lusail", strategy_enabled=False)
    assert recorder.names == ["ensure_race", "load"]
    assert STAGE_TEAM_RADIO not in race_stages(strategy_enabled=False)


def test_every_call_gets_the_same_race(recorder: _Recorder) -> None:
    """One GP goes in, so a fetch for a different one is a wiring error.

    The round number and the GP name are separate arguments and only the loader
    takes both, which is exactly the shape where one of them gets dropped.
    """
    prepare_race(2025, 23, "Lusail", strategy_enabled=True)
    by_name = dict(recorder.calls)
    assert by_name["ensure_race"] == (2025, "Lusail", False)
    assert by_name["ensure_radio_corpus"] == (2025, "Lusail", False)
    assert by_name["load"] == (2025, 23, "Lusail")


def test_the_fetches_are_told_not_to_print_their_own_progress(recorder: _Recorder) -> None:
    """`show_progress=True` writes a tqdm bar to stderr, which no window shows.

    The menu draws the stage itself, so a second progress display in a console
    the user is not looking at is noise.
    """
    prepare_race(2025, 23, "Lusail", strategy_enabled=True)
    for name, args in recorder.calls:
        if name.startswith("ensure_"):
            assert args[-1] is False, f"{name} was left printing to stderr"


def test_the_caller_hears_about_every_stage_in_order(recorder: _Recorder) -> None:
    """A stage that runs without reporting is a window that looks hung.

    `SessionLoader().load` is minutes long on a cold race, so the readout going
    quiet is indistinguishable from the freeze this change exists to remove.
    """
    seen: list[PrepareProgress] = []
    prepare_race(2025, 23, "Lusail", strategy_enabled=True, on_progress=seen.append)

    assert [p.stage for p in seen] == [STAGE_RACE_DATA, STAGE_TEAM_RADIO, STAGE_TELEMETRY]
    assert [p.index for p in seen] == [1, 2, 3]
    assert {p.total for p in seen} == {3}


def test_the_stage_count_matches_the_stages_actually_run(recorder: _Recorder) -> None:
    """`2/3` while only two stages will ever run is a bar that never fills."""
    for strategy_enabled in (True, False):
        recorder.calls.clear()
        seen: list[PrepareProgress] = []
        prepare_race(2025, 23, "Lusail", strategy_enabled=strategy_enabled, on_progress=seen.append)
        stages = race_stages(strategy_enabled=strategy_enabled)
        assert len(seen) == len(stages)
        assert seen[-1].index == seen[-1].total == len(stages)


def test_no_progress_callback_is_a_supported_caller(recorder: _Recorder) -> None:
    """The scripted path has no window to draw into and must not crash."""
    session = prepare_race(2025, 23, "Lusail", strategy_enabled=False, on_progress=None)
    assert session is not None


def test_the_label_names_the_stage_and_its_place() -> None:
    """What the menu renders, so it is asserted rather than eyeballed."""
    progress = PrepareProgress(stage=STAGE_TELEMETRY, index=3, total=3, started_at=0.0)
    assert progress.label == "Building telemetry  (3/3)"


def test_the_elapsed_seconds_climb_rather_than_sitting_at_zero() -> None:
    """Each stage reports ONCE, at its start, so a stored elapsed is always 0.

    The telemetry build is 349 s on a cold race. A readout frozen at "0s" for
    six minutes says exactly what a hung window says, which is the opposite of
    what putting this on a worker was for (#1116).
    """
    progress = PrepareProgress(stage=STAGE_TELEMETRY, index=3, total=3, started_at=100.0)
    assert progress.elapsed_s(100.0) == 0.0
    assert progress.elapsed_s(142.0) == 42.0
    assert progress.elapsed_s(99.0) == 0.0, "a clock that went backwards is not negative time"


def test_the_bar_shows_work_finished_not_work_started() -> None:
    """`index / total` reads 100% the moment the LAST stage begins.

    That last stage is the long one, so the bar would sit full for the entire
    wait it exists to describe.
    """
    total = 3
    fractions = [
        PrepareProgress(stage="s", index=i, total=total, started_at=0.0).done_fraction
        for i in range(1, total + 1)
    ]
    assert fractions == [0.0, pytest.approx(1 / 3), pytest.approx(2 / 3)]
    assert max(fractions) < 1.0


def test_prepare_touches_no_gl_object() -> None:
    """It runs on a worker thread, so anything needing the context would crash.

    `arcade` must not appear in the module at all: the caller builds `Track` and
    `F1ArcadeView` on the main thread from what this returns.
    """
    source = prepare.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "import arcade" not in text
    assert "arcade.Text" not in text


def test_a_race_the_dataset_does_not_hold_refuses_instead_of_degrading(
    recorder: _Recorder,
) -> None:
    """An empty fetch has to stop the launch, not feed an empty agents window.

    `snapshot_download` on a glob that matches nothing returns without raising,
    which is how "Mexico City" fetched zero files and looked exactly like a race
    that worked, right up until the AGENTS window sat empty (#1116).
    """
    for item in recorder.race_dir.iterdir():
        item.unlink()

    with pytest.raises(RaceDataUnavailable, match="Lusail"):
        prepare_race(2025, 23, "Lusail", strategy_enabled=True)


def test_the_refusal_happens_before_the_long_build(recorder: _Recorder) -> None:
    """Six minutes of telemetry for a race with no data is six minutes wasted."""
    for item in recorder.race_dir.iterdir():
        item.unlink()

    with pytest.raises(RaceDataUnavailable):
        prepare_race(2025, 23, "Lusail", strategy_enabled=True)
    assert "load" not in recorder.names
