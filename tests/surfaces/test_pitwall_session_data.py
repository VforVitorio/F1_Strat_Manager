"""The BULK channel: `src/pitwall/session_data.py` and `PitwallHost.get_bulk`.

Every assertion here is about an EFFECT a viewer could see, not about a
constant someone chose. That distinction is the reason this file exists in
the shape it does: the previous sprint shipped thirty-six green checks past a
chart whose baseline never finished drawing, because each one asserted a
mechanism.

The reveal cases run against the REAL Melbourne 2025 parquet when it is on
disk, and skip when it is not - a curated install holds one of seventy races.
The rules that must hold on races nobody has locally (the Miami stint block)
are exercised on the fixture shape `tests/agents/test_tyre_stint_repair.py`
already proved reaches the repairing path.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.f1_strat_manager.data_cache import get_data_root
from src.pitwall.host import PitwallHost
from src.pitwall.session_data import SessionLaps, race_dir, unavailable

MELBOURNE = (2025, "Melbourne")


def _session_or_skip() -> SessionLaps:
    session = SessionLaps.load(get_data_root(), *MELBOURNE)
    if session is None:
        pytest.skip("2025/Melbourne is not in this install's curated data set")
    return session


def _all_revealed(session: SessionLaps) -> dict[str, int]:
    return {code: session.total_laps for code in session.masked_view({}, 0.0)["drivers"]}


class _FakeClient:
    """The socket stands still so the host's own logic is what is measured."""

    def __init__(self, payload=None):
        self.latest = payload
        self.connected = True

    def start(self):
        pass

    def stop(self):
        pass


def _tick(reveal: dict[str, int], year: int = 2025, location: str = "Melbourne") -> dict:
    drivers = {code: {"laps_completed": laps} for code, laps in reveal.items()}
    return {
        "seq": 1,
        "arcade": {"year": year, "location": location, "global_t_min": 0.0, "drivers": drivers},
    }


# --- The reveal, which is the window's load-bearing invariant ----------------


def test_the_reveal_is_per_driver_not_one_shared_cut():
    """A single cut at the main driver's lap is wrong in both directions at once.

    At 96 % of instants the running field spans two or three different laps.
    Masking everyone at one number therefore lags the leaders by a lap AND
    leaks one to two laps of look-ahead for the cars behind, simultaneously.
    """
    session = _session_or_skip()
    codes = sorted(session.masked_view({}, 0.0)["drivers"])
    leader, backmarker = codes[0], codes[1]

    view = session.masked_view({leader: 30, backmarker: 28}, 0.0)

    assert view["drivers"][leader]["laps"][-1]["lap"] == 30
    assert view["drivers"][backmarker]["laps"][-1]["lap"] == 28
    # The one a shared cut would get wrong: the slower car must not be shown
    # laps 29-30 just because the leader has finished them.
    assert all(row["lap"] <= 28 for row in view["drivers"][backmarker]["laps"])


def test_the_reveal_is_strict_so_the_lap_in_progress_stays_hidden():
    """`L <= laps_completed`, never `<`, and never the lap being driven.

    The carrier used to be the tick's `lap`, a rounded interpolation of a
    step function measured non-monotone on 101 frames of 2.49 M: it flickers
    a lap open a tick early at the line and never opens a finisher's last
    lap. `laps_completed` comes off the crossing map.
    """
    session = _session_or_skip()
    code = sorted(session.masked_view({}, 0.0)["drivers"])[0]

    revealed = session.masked_view({code: 10}, 0.0)["drivers"][code]["laps"]

    assert [row["lap"] for row in revealed][-1] == 10
    assert 11 not in [row["lap"] for row in revealed]


def test_a_driver_the_tick_never_mentions_reveals_nothing():
    """A car with no position data (#886) is not a car whose whole race shows."""
    session = _session_or_skip()

    view = session.masked_view({}, 0.0)

    assert all(driver["laps"] == [] for driver in view["drivers"].values())


def test_a_rewind_un_reveals(tmp_path):
    """Seek to the end, then rewind: the rows must GO AWAY.

    The failure this forbids is not subtle - a grow-only reveal shows the
    final classification and the session's purple times on a screen whose
    clock reads lap 10. It is the race result, on a fidelity surface.
    """
    session = _session_or_skip()
    code = sorted(session.masked_view({}, 0.0)["drivers"])[0]

    at_the_end = session.masked_view({code: session.total_laps}, 0.0)
    rewound = session.masked_view({code: 10}, 0.0)

    assert len(rewound["drivers"][code]["laps"]) < len(at_the_end["drivers"][code]["laps"])
    assert rewound["drivers"][code]["laps"][-1]["lap"] == 10


def test_the_best_falls_back_when_the_clock_does():
    """The bests panel is recomputed, so a rewind must un-set a purple time too.

    An accumulated `Math.min` on the client would survive the rewind and keep
    showing a lap-43 time at lap 10. Recomputing from the revealed subset is
    what makes that impossible rather than merely unlikely.
    """
    session = _session_or_skip()
    code = sorted(session.masked_view({}, 0.0)["drivers"])[0]

    late = session.masked_view({code: session.total_laps}, 0.0)["drivers"][code]["best"]
    early = session.masked_view({code: 5}, 0.0)["drivers"][code]["best"]

    assert late["lap_time"] is not None and early["lap_time"] is not None
    assert early["lap_time"] >= late["lap_time"]
    assert early["lap"] <= 5


# --- The rows FastF1 invents, which poison every naive statistic -------------


def test_generated_rows_are_rendered_but_never_counted():
    """The 6 rows FastF1 synthesises for cars that did not finish a lap.

    Their `Time` stamps sort BEFORE the real field, so a naive ranking puts
    the lap-1 crashers P1-P2-P3 and a naive lap count shows a crashed car in
    the top three for 172 seconds of replay. They stay as rows - the table
    still has to show the car - and they enter no crossing and no best.
    """
    session = _session_or_skip()
    view = session.masked_view(_all_revealed(session), 0.0)

    rows = [row for driver in view["drivers"].values() for row in driver["laps"]]
    generated = [row for row in rows if row["generated"]]
    assert generated, "the real race carries generated rows; the fixture would be lying"

    for driver in view["drivers"].values():
        counted = {row["lap"] for row in driver["laps"] if row["generated"]}
        assert not (counted & set(driver["crossings"])), "a generated row entered the gap clock"


def test_a_generated_row_never_becomes_a_best():
    """They carry NaN times, so a min() that saw them would return None or NaN."""
    session = _session_or_skip()
    view = session.masked_view(_all_revealed(session), 0.0)

    for code, driver in view["drivers"].items():
        real_times = [
            row["lap_time"]
            for row in driver["laps"]
            if row["lap_time"] is not None and not row["generated"] and not row["deleted"]
        ]
        if not real_times:
            continue
        assert driver["best"]["lap_time"] == min(real_times), code


def test_a_deleted_lap_is_shown_but_cannot_be_a_best():
    """A deleted time does not count - that is what deleting it meant.

    The case has to be CONSTRUCTED, and the reason is the point. Melbourne's
    six deleted laps are track-limits laps that were slower than the driver's
    best anyway, so asserting against the real race passes whether or not the
    filter exists: the assertion would be true for the wrong reason, which is
    a failure mode this project has already paid for. Here the deleted lap is
    the quickest one on purpose.
    """
    frame = _frame(
        lap_times=[92.0, 80.0, 91.0],
        deleted=[False, True, False],
    )
    session = SessionLaps(2025, "Nowhere", {"NOR": frame}, {"NOR": "4"})

    driver = session.masked_view({"NOR": 3}, 0.0)["drivers"]["NOR"]

    assert driver["best"]["lap_time"] == 91.0, "the deleted 80.0 became the best"
    assert driver["best"]["lap"] == 3
    assert any(row["deleted"] for row in driver["laps"]), "the row is still rendered"


# --- The sentinel rule, and the serialiser that enforces it ------------------


def test_the_whole_race_crosses_the_bridge_that_forbids_nan():
    """`webserver._send_json` uses `allow_nan=False`; the parquet is full of NaN.

    Measured on the real race: 137 NaN floats in a naive payload, and the
    dump raises `ValueError` - a 500 and a blank window. Sampling the first
    few rows hides it completely, which is how it would have shipped.
    """
    session = _session_or_skip()
    view = session.masked_view(_all_revealed(session), 0.0)

    body = json.dumps(view, allow_nan=False)

    assert len(body) > 100_000, "the whole race, not a slice"


def test_a_missing_value_is_none_and_never_a_number():
    """This repo has a scar: a NaN `Position` became 0, and the leader then
    "found" the car that had just crashed at `position == pos - 1`. A default
    must never be a value the code also searches for."""
    session = _session_or_skip()
    view = session.masked_view(_all_revealed(session), 0.0)

    positions = [row["position"] for driver in view["drivers"].values() for row in driver["laps"]]
    assert None in positions, "the real race has 6 rows with no position"
    assert 0 not in positions


def test_lap_one_has_no_first_sector_and_that_is_not_a_zero():
    """Construction, not corruption - and a zero would win every S1 ranking."""
    session = _session_or_skip()
    view = session.masked_view({code: 1 for code in _all_revealed(session)}, 0.0)

    for code, driver in view["drivers"].items():
        if driver["laps"]:
            assert driver["laps"][0]["s1"] is None, code
        assert driver["best"]["s1"] is None, code


# --- Stops, and the race nobody has locally ---------------------------------


def test_stops_are_counted_from_the_pit_lane_not_from_the_stint_column():
    """Both derivations agree on a healthy race, which is how the wrong one gets
    chosen. Miami 2025's raw frame carries a 446-row NaN `Stint` block, and a
    stint-based count reads zero stops for most of the field late in the race.

    The frame below is that shape: stint metadata absent, pit stops real.
    """
    laps = pd.DataFrame(
        {
            "Driver": ["NOR"] * 6,
            "DriverNumber": ["4"] * 6,
            "LapNumber": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "Time": pd.to_timedelta([90, 180, 270, 365, 455, 545], unit="s"),
            "LapTime": pd.to_timedelta([90, 90, 90, 95, 90, 90], unit="s"),
            "Stint": [float("nan")] * 6,
            "PitInTime": pd.to_timedelta([None, None, 270, None, None, None], unit="s"),
            "PitOutTime": pd.to_timedelta([None, None, None, 300, None, None], unit="s"),
            "Compound": ["MEDIUM"] * 6,
            "Position": [1.0] * 6,
        }
    )
    session = SessionLaps(2025, "Nowhere", {"NOR": [_row(laps, i) for i in range(6)]}, {"NOR": "4"})

    view = session.masked_view({"NOR": 6}, 0.0)

    assert view["drivers"]["NOR"]["stops"] == 1
    assert all(row["stint"] is None for row in view["drivers"]["NOR"]["laps"])


def _row(frame: pd.DataFrame, index: int) -> dict:
    from src.pitwall.session_data import _lap_row

    return _lap_row(frame.iloc[index].to_dict())


def _frame(lap_times: list[float], deleted: list[bool]) -> list[dict]:
    """A minimal per-driver row list, for the cases the real race cannot show."""
    laps = pd.DataFrame(
        {
            "Driver": ["NOR"] * len(lap_times),
            "DriverNumber": ["4"] * len(lap_times),
            "LapNumber": [float(i + 1) for i in range(len(lap_times))],
            "Time": pd.to_timedelta(
                [sum(lap_times[: i + 1]) for i in range(len(lap_times))], unit="s"
            ),
            "LapTime": pd.to_timedelta(lap_times, unit="s"),
            "Deleted": deleted,
            "Compound": ["MEDIUM"] * len(lap_times),
            "Position": [1.0] * len(lap_times),
        }
    )
    return [_row(laps, i) for i in range(len(lap_times))]


# --- Resolving the race, and not having it ----------------------------------


def test_the_race_resolves_on_location_including_the_underscore_variant(tmp_path):
    """FastF1's 2025 Location is "Miami Gardens"; the folder is `Miami_Gardens`.

    Resolving on `gp_name` instead misses that race on the happy path - the
    canonical calendar's key for it is "Miami" - which is why the producer's
    own comment tells consumers to read `location`.
    """
    folder = tmp_path / "raw" / "2025" / "Miami_Gardens"
    folder.mkdir(parents=True)
    (folder / "laps.parquet").write_bytes(b"")

    assert race_dir(tmp_path, 2025, "Miami Gardens") == folder
    assert race_dir(tmp_path, 2025, "Miami") is None


def test_an_absent_race_is_absent_data_not_a_crash(tmp_path):
    """The COMMON case: a curated install holds one of seventy races."""
    assert SessionLaps.load(tmp_path, 2025, "Interlagos") is None


def test_the_unavailable_payload_says_so_rather_than_being_empty():
    """A tower rendering zero rows silently is the same pixel as a tower whose
    reveal is broken. The panels need to be able to tell the two apart."""
    payload = unavailable(2025, "Interlagos")

    assert payload["available"] is False
    assert payload["race"]["location"] == "Interlagos"


# --- The host seam ----------------------------------------------------------


def test_the_bulk_revision_advances_on_a_rewind_as_well_as_on_a_lap():
    """The reason the caller's comparison must be inequality, not "greater than".

    A rewind LOWERS the revealed set. `host.py` already documents the
    identical bug one level down, where `seq > since_seq` froze both windows
    on a dead race for ten minutes after a producer restart. Here a `>`
    compare would withhold exactly the un-reveal, and a test that never
    rewinds stays green through it.
    """
    client = _FakeClient(_tick({"NOR": 20}))
    host = PitwallHost(client, window_count=1)

    first = host.get_bulk(-1)
    assert first is not None
    assert host.get_bulk(first["rev"]) is None, "an up-to-date caller gets nothing"

    client.latest = _tick({"NOR": 10})
    rewound = host.get_bulk(first["rev"])

    assert rewound is not None, "the rewind must reach the caller"
    assert rewound["rev"] != first["rev"]


def test_a_restarted_host_reaches_a_window_holding_a_high_revision():
    """Why the comparison is inequality and not "greater than".

    The revision is a counter, so it climbs while one host lives - and a
    `>` compare is indistinguishable from `!=` for as long as that is true.
    It stops being true the moment the process behind it restarts and the
    counter goes back to zero, which is exactly the shape `get_tick`
    documents one level down: relaunch with the windows open and `seq >
    since_seq` froze both on the dead race until the new run's sequence
    passed the old one's, ten minutes of a live-looking screen.
    """
    client = _FakeClient(_tick({"NOR": 20}))
    fresh_host = PitwallHost(client, window_count=1)

    served = fresh_host.get_bulk(since_rev=999)

    assert served is not None, "a window holding a stale high revision was left frozen"
    assert served["rev"] < 999


def test_the_host_serves_fewer_rows_after_a_rewind():
    """The effect, not the revision counter: the rows themselves must shrink."""
    session = SessionLaps.load(get_data_root(), *MELBOURNE)
    if session is None:
        pytest.skip("2025/Melbourne is not in this install's curated data set")

    client = _FakeClient(_tick({"NOR": 30}))
    host = PitwallHost(client, window_count=1)
    late = host.get_bulk(-1)

    client.latest = _tick({"NOR": 5})
    early = host.get_bulk(late["rev"])

    assert len(early["drivers"]["NOR"]["laps"]) < len(late["drivers"]["NOR"]["laps"])


def test_pointing_the_arcade_at_another_race_replaces_the_laps():
    """The stale-state class #904 already paid for once on the AGENTS history:
    a producer restart left the dead race's numbers on screen permanently."""
    client = _FakeClient(_tick({"NOR": 20}))
    host = PitwallHost(client, window_count=1)
    host.get_bulk(-1)

    client.latest = _tick({"NOR": 20}, year=2025, location="Interlagos")
    other = host.get_bulk(-1)

    assert other["available"] is False
    assert other["race"]["location"] == "Interlagos"


def test_no_tick_means_no_bulk():
    """Before the first broadcast there is no race to read, and inventing one
    would mean guessing which."""
    host = PitwallHost(_FakeClient(None), window_count=1)

    assert host.get_bulk(-1) is None


# --- The lap in progress: sectors revealed at the moment they were crossed ----


def test_a_sector_is_closed_until_the_clock_reaches_its_crossing():
    """The reveal rule at a finer coordinate, on the real race.

    `masked_view`'s `L <= laps_completed` is the rule for lap ROWS, which only
    exist once the lap is over. A sector carries its own `SectorNSessionTime`,
    so it has its own moment - and revealing it then is not look-ahead, it is
    the present.

    NOR's lap 23 at Melbourne crossed S1 at SessionTime 6689.966. One second
    before that the cell must be empty; a tenth of a second after it must hold
    31.865 and the 266 km/h measured at that trap.
    """
    session = _session_or_skip()
    global_t_min = 4260.355
    row = next(r for r in session._by_driver["NOR"] if r["lap"] == 23)

    before = session.live_lap({"NOR": 22}, row["s1_at"] - 1 - global_t_min, global_t_min)["NOR"]
    after = session.live_lap({"NOR": 22}, row["s1_at"] + 0.1 - global_t_min, global_t_min)["NOR"]

    assert before["lap"] == 23 and after["lap"] == 23, "both probes are on the lap in progress"
    assert before["s1"] is None and before["v1"] is None, "nothing before the car got there"
    assert after["s1"] == row["s1"], "and the real sector time the instant it did"
    assert after["v1"] == row["v1"], "with the speed measured at that trap"
    assert after["s2"] is None, "the sectors after it stay shut"


def test_a_rewind_shuts_the_sectors_again():
    """The clock going back must CLOSE a cell, not leave it filled.

    A cache that only ever fills would leave a time on screen for track the
    car has yet to re-drive - the same leak as a lap-row reveal that never
    un-reveals, one coordinate down.
    """
    session = _session_or_skip()
    global_t_min = 4260.355
    row = next(r for r in session._by_driver["NOR"] if r["lap"] == 23)

    late = session.live_lap({"NOR": 22}, row["s3_at"] + 1 - global_t_min, global_t_min)["NOR"]
    rewound = session.live_lap({"NOR": 22}, row["s1_at"] - 1 - global_t_min, global_t_min)["NOR"]

    assert [late["s1"], late["s2"], late["s3"]] == [row["s1"], row["s2"], row["s3"]]
    assert [rewound["s1"], rewound["s2"], rewound["s3"]] == [None, None, None]


def test_a_car_with_no_lap_in_progress_is_absent_rather_than_empty():
    """Retired, finished, or never completed one - all three mean the same here.

    SAI's only rows on this race are `FastF1Generated`, which carry no times
    at all, so serving one as "the lap in progress" would put a row of nulls
    on screen that looks like a car waiting to cross a sector it never will.
    """
    session = _session_or_skip()
    live = session.live_lap({"SAI": 0, "NOR": 22}, 3000.0, 4260.355)

    assert "SAI" not in live, "a generated-only driver has no lap in progress"
    assert "NOR" in live, "and the fixture is not simply empty"


def test_the_sector_crossing_instants_stay_off_the_bulk_payload():
    """They exist for `live_lap`; the tower never reads them from a lap row.

    Three more floats on each of 927 rows is a field nobody consumes riding on
    the channel that already carries the whole race.
    """
    session = _session_or_skip()
    view = session.masked_view(_all_revealed(session), 0.0)
    row = view["drivers"]["NOR"]["laps"][0]

    for key in ("s1_at", "s2_at", "s3_at"):
        assert key not in row, f"{key} leaked onto the bulk payload"


def test_a_best_speed_is_the_fastest_and_not_the_slowest():
    """The two tuples are two KINDS of quantity, and one loop treated them as one.

    A best lap time is the smallest value in the column; a best trap speed is
    the largest. Minimising both served, on the real race, NOR's best
    speed-trap as 180 km/h against a real maximum of 289 - his slowest crawl
    through the trap, presented as his best of the session, on all four speed
    columns at once (#923).

    Asserted against the maximum RECOMPUTED from the rows the same view
    served, so this cannot pass by agreeing with a constant somebody typed.
    """
    session = _session_or_skip()
    view = session.masked_view(_all_revealed(session), 0.0)

    checked = 0
    for code, driver in view["drivers"].items():
        countable = [row for row in driver["laps"] if not row["deleted"] and not row["generated"]]
        for field in ("v1", "v2", "vfl", "vst"):
            values = [row[field] for row in countable if row[field] is not None]
            if not values:
                continue
            assert driver["best"][field] == max(values), (
                f"{code}'s best {field} is {driver['best'][field]} but its rows reach "
                f"{max(values)} (min is {min(values)})"
            )
            checked += 1

    assert checked >= 60, f"only {checked} speed bests compared; the fixture proves nothing"


def test_a_best_lap_time_is_still_the_smallest():
    """The twin of the case above, so the fix cannot swing the other way.

    Separating the two loops is exactly the kind of change that fixes one
    direction and breaks the other, and a test that only pinned the speeds
    would be green through it.
    """
    session = _session_or_skip()
    view = session.masked_view(_all_revealed(session), 0.0)

    checked = 0
    for code, driver in view["drivers"].items():
        countable = [row for row in driver["laps"] if not row["deleted"] and not row["generated"]]
        for field in ("lap_time", "s1", "s2", "s3"):
            values = [row[field] for row in countable if row[field] is not None]
            if not values:
                continue
            assert driver["best"][field] == min(values), (
                f"{code}'s best {field} is {driver['best'][field]} but its rows reach {min(values)}"
            )
            checked += 1

    assert checked >= 60, f"only {checked} time bests compared; the fixture proves nothing"
