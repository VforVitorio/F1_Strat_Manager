"""`f1-prefetch` fills the replay cache without rebuilding what is already there.

The command is a loop over `prepare_race` with one addition, a skip for rounds
whose pickle exists, so what is asserted here is the loop's own behaviour: the
rounds spec, the skip and its `--force` override, one round's failure not
ending the run, the flags reaching the call, and the exit code.

**`prepare_race` is always faked.** A real call is a download plus a build of
several minutes, and the cache directory is always a `tmp_path`: the six
pickles under `data/cache/arcade/` cost that much each and nothing here may
touch them.
"""

from __future__ import annotations

import io
import time
from collections import Counter
from pathlib import Path

import pytest

from src.arcade import prefetch as prefetch_module
from src.arcade.config import get_gp_names
from src.arcade.data import SessionLoader
from src.arcade.prefetch import (
    Outcome,
    exit_code,
    main,
    parse_rounds,
    prefetch,
)
from src.arcade.prepare import STAGE_TELEMETRY, PrepareProgress, RaceDataUnavailable

YEAR = 2025


class _FakePrepare:
    """Stands in for `prepare_race`: records each call and raises where told to.

    It reports one stage through `on_progress` the way the real one does, so
    the printed progress can be asserted rather than eyeballed.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.raising: dict[int, Exception] = {}

    def __call__(self, year, round_, gp_name, *, strategy_enabled, on_progress=None):
        self.calls.append(
            {"year": year, "round": round_, "gp": gp_name, "strategy_enabled": strategy_enabled}
        )
        if on_progress is not None:
            on_progress(
                PrepareProgress(
                    stage=STAGE_TELEMETRY, index=1, total=1, started_at=time.monotonic()
                )
            )
        if round_ in self.raising:
            raise self.raising[round_]
        return object()

    @property
    def rounds(self) -> list[int]:
        return [call["round"] for call in self.calls]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> _FakePrepare:
    """Patch `prepare_race` where `prefetch` looks it up."""
    stub = _FakePrepare()
    monkeypatch.setattr(prefetch_module, "prepare_race", stub)
    return stub


def _cache(tmp_path: Path, round_: int) -> Path:
    """Put a pickle where the LOADER would look for it, not where a string says."""
    path = SessionLoader(cache_dir=tmp_path)._cache_path(YEAR, round_)
    path.write_bytes(b"not a pickle, and it must never be opened")
    return path


def _run(tmp_path: Path, rounds: list[int], **flags) -> tuple[Counter, str]:
    """`prefetch` against a temporary cache, returning the counts and what it printed."""
    options = {"with_radio": False, "force": False, **flags}
    out = io.StringIO()
    counts = prefetch(YEAR, rounds, cache_dir=tmp_path, out=out, **options)
    return counts, out.getvalue()


# ── The rounds spec ───────────────────────────────────────────────────────────


def test_rounds_accept_commas_and_ranges() -> None:
    assert parse_rounds("1,3,5-8") == [1, 3, 5, 6, 7, 8]


def test_rounds_come_back_sorted_and_deduplicated() -> None:
    """The order typed is not the order run, and a round named twice runs once."""
    assert parse_rounds("8, 3-4, 1, 3") == [1, 3, 4, 8]


@pytest.mark.parametrize("spec", ["", "a", "1,,3", "0", "8-5", "-3", "5-", "1-2-3", "1.5"])
def test_rounds_reject_nonsense(spec: str) -> None:
    """A typo has to die here, not after the first six-minute build."""
    with pytest.raises(ValueError):
        parse_rounds(spec)


# ── The skip ──────────────────────────────────────────────────────────────────


def test_a_cached_race_is_skipped_without_loading_it(fake: _FakePrepare, tmp_path: Path) -> None:
    """The pickle is on disk, so nothing may spend 5 s and 380 MB finding that out.

    The file is put where the loader's own path rule says, so a check that
    built its own name would look somewhere else and go on to prepare.
    """
    _cache(tmp_path, 1)
    counts, text = _run(tmp_path, [1])
    assert fake.calls == []
    assert counts == Counter({Outcome.SKIPPED: 1})
    assert "skipped" in text


def test_force_prepares_a_cached_race_anyway(fake: _FakePrepare, tmp_path: Path) -> None:
    """After a `CACHE_VERSION` bump the file is there and stale; `--force` is the way past."""
    _cache(tmp_path, 1)
    counts, _ = _run(tmp_path, [1], force=True)
    assert fake.rounds == [1]
    assert counts == Counter({Outcome.PREPARED: 1})


def test_an_uncached_race_is_prepared(fake: _FakePrepare, tmp_path: Path) -> None:
    counts, _ = _run(tmp_path, [2])
    assert fake.rounds == [2]
    assert counts == Counter({Outcome.PREPARED: 1})


# ── One round's failure does not end the run ──────────────────────────────────


def test_an_unavailable_race_does_not_stop_the_next(fake: _FakePrepare, tmp_path: Path) -> None:
    """The dataset having nothing for a round is an answer, counted and moved past."""
    fake.raising[2] = RaceDataUnavailable("No data published for Shanghai 2025.")
    counts, text = _run(tmp_path, [2, 3])
    assert fake.rounds == [2, 3]
    assert counts == Counter({Outcome.UNAVAILABLE: 1, Outcome.PREPARED: 1})
    assert "No data published for Shanghai" in text


def test_a_genuine_failure_does_not_stop_the_next_either(
    fake: _FakePrepare, tmp_path: Path
) -> None:
    """A dropped connection on round 2 of 24 must not cost the other 22."""
    fake.raising[2] = ConnectionError("dropped")
    counts, text = _run(tmp_path, [2, 3])
    assert fake.rounds == [2, 3]
    assert counts == Counter({Outcome.FAILED: 1, Outcome.PREPARED: 1})
    assert "ConnectionError: dropped" in text


# ── What reaches the call ─────────────────────────────────────────────────────


def test_with_radio_is_the_strategy_flag(fake: _FakePrepare, tmp_path: Path) -> None:
    """The radio corpus is fetched only when asked, and it is `strategy_enabled` that asks."""
    _run(tmp_path, [2])
    _run(tmp_path, [2], with_radio=True)
    assert [call["strategy_enabled"] for call in fake.calls] == [False, True]


def test_the_round_is_named_from_the_calendar(fake: _FakePrepare, tmp_path: Path) -> None:
    """Same resolver as the menu, so the two can only fetch the same race for a round."""
    _run(tmp_path, [3])
    assert fake.calls[0]["gp"] == get_gp_names(YEAR)[3]
    assert fake.calls[0]["year"] == YEAR


# ── What a person watching the terminal reads ─────────────────────────────────


def test_the_race_the_stage_and_its_duration_are_printed(
    fake: _FakePrepare, tmp_path: Path
) -> None:
    _, text = _run(tmp_path, [2])
    lines = text.splitlines()
    assert lines[0].startswith("[1/1] 2025 R02 ")
    assert any(line.startswith(f"  {STAGE_TELEMETRY}  (1/1)") for line in lines)
    assert any(line.startswith(f"  {STAGE_TELEMETRY}: ") and line.endswith("s") for line in lines)
    assert any(line.startswith("  prepared in ") for line in lines)


def test_the_summary_counts_every_outcome(fake: _FakePrepare, tmp_path: Path) -> None:
    _cache(tmp_path, 1)
    fake.raising[2] = RaceDataUnavailable("nothing")
    fake.raising[4] = RuntimeError("boom")
    _, text = _run(tmp_path, [1, 2, 3, 4])
    assert text.splitlines()[-1].endswith("1 prepared, 1 skipped, 1 unavailable, 1 failed")


# ── The exit code ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("prepared", "skipped", "unavailable", "failed", "expected"),
    [
        (0, 0, 0, 0, 0),
        (1, 0, 0, 5, 0),
        (0, 1, 0, 5, 0),
        (0, 0, 5, 0, 0),
        (0, 0, 0, 1, 1),
        (0, 0, 3, 2, 1),
    ],
)
def test_the_exit_code_is_nonzero_only_when_nothing_was_achieved_and_something_broke(
    prepared: int, skipped: int, unavailable: int, failed: int, expected: int
) -> None:
    counts = Counter(
        {
            Outcome.PREPARED: prepared,
            Outcome.SKIPPED: skipped,
            Outcome.UNAVAILABLE: unavailable,
            Outcome.FAILED: failed,
        }
    )
    assert exit_code(counts) == expected


# ── The entry point ───────────────────────────────────────────────────────────


class _FakePrefetch:
    """Records what `main` hands to the loop, so every flag is seen to arrive."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, year, rounds, *, with_radio, force, **_):
        self.calls.append(
            {"year": year, "rounds": list(rounds), "with_radio": with_radio, "force": force}
        )
        return Counter()


@pytest.fixture
def loop(monkeypatch: pytest.MonkeyPatch) -> _FakePrefetch:
    stub = _FakePrefetch()
    monkeypatch.setattr(prefetch_module, "prefetch", stub)
    return stub


def test_every_flag_reaches_the_loop(loop: _FakePrefetch) -> None:
    """A flag parsed and not passed is the pair-with-one-half-wired defect, so all four."""
    main(["--year", "2025", "--rounds", "1,3", "--with-radio", "--force"])
    assert loop.calls == [{"year": 2025, "rounds": [1, 3], "with_radio": True, "force": True}]


def test_no_rounds_means_the_whole_calendar(loop: _FakePrefetch) -> None:
    main(["--year", "2025"])
    assert loop.calls[0]["rounds"] == sorted(get_gp_names(2025))
    assert loop.calls[0] == {
        "year": 2025,
        "rounds": loop.calls[0]["rounds"],
        "with_radio": False,
        "force": False,
    }


def test_a_round_off_the_calendar_is_refused_before_anything_runs(loop: _FakePrefetch) -> None:
    """`--rounds 1,30` must not build round 1 for six minutes and then discover 30."""
    with pytest.raises(SystemExit) as exit_info:
        main(["--year", "2025", "--rounds", "1,30"])
    assert exit_info.value.code == 2
    assert loop.calls == []


def test_a_bad_spec_is_refused_with_its_reason(
    loop: _FakePrefetch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--year", "2025", "--rounds", "8-5"])
    assert exit_info.value.code == 2
    assert "ranges ascend" in capsys.readouterr().err
    assert loop.calls == []


def test_main_returns_the_exit_code_rather_than_raising_it(
    loop: _FakePrefetch, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One path for the console script and the tests: the code comes back as a value."""
    monkeypatch.setattr(prefetch_module, "exit_code", lambda counts: 7)
    assert main(["--year", "2025", "--rounds", "1"]) == 7
