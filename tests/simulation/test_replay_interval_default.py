"""A caller that does not ask for pacing must not pay for it (#1202).

`RaceReplayEngine.replay()` sleeps `interval_seconds` after every yielded lap. That
default used to be 3.0, and three test sites built the engine without the argument and
then drove 49 laps to reach lap 50, so each one paid 147 s of pure sleeping. Together
they were 441 s, 40.2% of an 18m17s local suite, and two of the three were reported
under `setup` rather than `call`, which is why reading the durations table for slow
tests walked straight past them.

The guard asserts the EFFECT, that no sleeping happens, rather than the value of the
default. A test pinning `interval_seconds == 0.0` would stay green if `replay()` grew a
second delay somewhere else, and it would say nothing about what a caller actually pays.

`test_a_caller_that_asks_for_pacing_still_gets_it` is here so the first test cannot pass
vacuously: it proves the recorder sees sleeping when sleeping happens.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

_TOTAL_LAPS = 6


def _race_dir(tmp_path: Path) -> Path:
    """Write the smallest `laps.parquet` the engine will load, and return its directory.

    Hermetic on purpose: the defect this guards is about wall-clock cost, so gating it
    on `data/` would leave it skipped on CI, which is the environment least likely to
    notice a delay creeping back in.

    Returns:
        A race directory the engine can construct from, holding one clean stint for a
        single driver over ``_TOTAL_LAPS`` laps.
    """
    rows = [
        {
            "Driver": "NOR",
            "DriverNumber": "4",
            "LapNumber": lap,
            "LapTime_s": 90.0,
            "LapTime": pd.Timedelta(seconds=90),
            "Time": pd.Timedelta(seconds=90 * lap),
            "TrackStatus": "1",
            "Position": 1,
            "Compound": "MEDIUM",
            "TyreLife": lap,
            "Stint": 1,
            "PitInTime": pd.NaT,
            "Team": "McLaren",
        }
        for lap in range(1, _TOTAL_LAPS + 1)
    ]
    race_dir = tmp_path / "Melbourne"
    race_dir.mkdir()
    pd.DataFrame(rows).to_parquet(race_dir / "laps.parquet")
    return race_dir


def _sleep_recorder(monkeypatch) -> list[float]:
    """Replace the engine module's `time.sleep` with a recorder and return its log.

    Patching the module rather than `time` itself keeps the rest of the process, pytest
    included, on the real function.

    Returns:
        A list that receives one entry per sleep call, holding the seconds requested.
    """
    from src.simulation import replay_engine

    slept: list[float] = []
    monkeypatch.setattr(replay_engine.time, "sleep", slept.append)
    return slept


def test_the_default_engine_does_not_sleep_between_laps(tmp_path, monkeypatch):
    """Constructing without `interval_seconds` and draining the replay sleeps zero times."""
    from src.simulation.replay_engine import RaceReplayEngine

    slept = _sleep_recorder(monkeypatch)
    engine = RaceReplayEngine(_race_dir(tmp_path), driver_code="NOR", team="McLaren")

    emitted = list(engine.replay())

    assert len(emitted) == _TOTAL_LAPS, "the replay must still emit every lap"
    assert slept == [], (
        f"the default engine slept {sum(slept)} s across {len(slept)} calls; "
        "a caller that does not ask for pacing must not pay for it (#1202)"
    )


def test_a_caller_that_asks_for_pacing_still_gets_it(tmp_path, monkeypatch):
    """The recorder is not blind: an explicit interval sleeps once per lap."""
    from src.simulation.replay_engine import RaceReplayEngine

    slept = _sleep_recorder(monkeypatch)
    engine = RaceReplayEngine(
        _race_dir(tmp_path), driver_code="NOR", team="McLaren", interval_seconds=0.25
    )

    list(engine.replay())

    assert slept == [0.25] * _TOTAL_LAPS


@pytest.mark.parametrize(
    ("module_path", "symbol"),
    [
        ("scripts.run_simulation_cli", "--interval"),
        ("src.simulation.__main__", "--interval"),
    ],
)
def test_the_interactive_clis_still_default_to_no_sleep(module_path, symbol):
    """The pacing decision belongs to the CLI flag, and both flags default to 0.0.

    Read from the source rather than by parsing arguments, so the check costs nothing and
    does not import the agent stack. If either default ever moves off 0.0, the engine's
    own default is no longer the only thing keeping a batch run fast.
    """
    import ast
    import textwrap

    path = Path(__file__).parents[2] / (module_path.replace(".", "/") + ".py")
    tree = ast.parse(textwrap.dedent(path.read_text(encoding="utf-8")))

    defaults = [
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and any(isinstance(arg, ast.Constant) and arg.value == symbol for arg in node.args)
        for keyword in node.keywords
        if keyword.arg == "default" and isinstance(keyword.value, ast.Constant)
    ]

    assert defaults == [0.0], f"{module_path} declares {symbol} with defaults {defaults}"
