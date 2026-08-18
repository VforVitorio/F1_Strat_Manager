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


def _stint_frame(
    compounds: list[str],
    tyre_life: list[float | None],
    pit_in: list[float | None],
    pit_out: list[float | None],
    stint: list[float] | None = None,
) -> dict[str, list[dict]]:
    """One driver's rows with the tyre columns a stop count reads.

    Separate from `_frame` because that one hardcodes a single compound and no
    age at all, which is precisely why the fixture it feeds could not tell a
    stop from a pit-lane transit.
    """
    count = len(compounds)
    laps = pd.DataFrame(
        {
            "Driver": ["NOR"] * count,
            "DriverNumber": ["4"] * count,
            "LapNumber": [float(index + 1) for index in range(count)],
            "Time": pd.to_timedelta([90 * (index + 1) for index in range(count)], unit="s"),
            "LapTime": pd.to_timedelta([90.0] * count, unit="s"),
            "Stint": stint if stint is not None else [float("nan")] * count,
            "TyreLife": tyre_life,
            "PitInTime": pd.to_timedelta(pit_in, unit="s"),
            "PitOutTime": pd.to_timedelta(pit_out, unit="s"),
            "Compound": compounds,
            "Position": [1.0] * count,
        }
    )
    return {"NOR": [_row(laps, index) for index in range(count)]}


def test_a_stop_is_a_tyre_change_and_the_stint_column_is_not_asked():
    """Miami 2025's shape: the `Stint` metadata is a NaN block, the stop is real.

    **The fixture now carries the EVIDENCE a stop leaves**, which the version
    before this one did not: it held one compound for all six laps and no
    `TyreLife` column at all, so it could not distinguish a tyre change from a
    car driving down the pit lane and out again. It passed only because the
    count it pinned looked at `PitInTime` alone - the defect. The stop is on lap
    3 and the out-lap is 4, so the set that appears on lap 4 is a new one.
    """
    session = SessionLaps(
        2025,
        "Nowhere",
        _stint_frame(
            compounds=["MEDIUM"] * 3 + ["HARD"] * 3,
            tyre_life=[1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            pit_in=[None, None, 270, None, None, None],
            pit_out=[None, None, None, 300, None, None],
        ),
        {"NOR": "4"},
    )

    view = session.masked_view({"NOR": 6}, 0.0)

    assert view["drivers"]["NOR"]["stops"] == 1
    assert all(row["stint"] is None for row in view["drivers"]["NOR"]["laps"])


def test_a_pit_lane_transit_that_changed_nothing_is_not_a_stop():
    """The safety-car parade, and the drive-through penalty `tyre_stint_repair`
    names: the car goes down the pit lane, no work is done, the set carries on.

    Three consecutive in-laps, compound unchanged, age counting up through all
    of them - the exact shape all seventeen Melbourne runners have on laps 2-4.
    Counting in-laps answers 3 here, which is what shipped.
    """
    session = SessionLaps(
        2025,
        "Nowhere",
        _stint_frame(
            compounds=["INTERMEDIATE"] * 6,
            tyre_life=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            pit_in=[None, 180, 270, 360, None, None],
            pit_out=[None, None, 300, 390, 480, None],
            # FastF1 opens a new stint on each pass even though nothing changed,
            # which is the other derivation this must not be talked into.
            stint=[1.0, 1.0, 2.0, 3.0, 4.0, 4.0],
        ),
        {"NOR": "4"},
    )

    view = session.masked_view({"NOR": 6}, 0.0)

    assert view["drivers"]["NOR"]["stops"] == 0


def test_the_same_compound_refitted_is_still_a_stop():
    """A stop is not always a compound change: the age reset carries this one.

    Without the age half of the rule this would read zero, and 743 of the 2594
    stint transitions across the shipped races fit the same compound again
    (`tyre_stint_repair`'s own census).
    """
    session = SessionLaps(
        2025,
        "Nowhere",
        _stint_frame(
            compounds=["HARD"] * 6,
            tyre_life=[18.0, 19.0, 20.0, 1.0, 2.0, 3.0],
            pit_in=[None, None, 270, None, None, None],
            pit_out=[None, None, None, 300, None, None],
        ),
        {"NOR": "4"},
    )

    assert session.masked_view({"NOR": 6}, 0.0)["drivers"]["NOR"]["stops"] == 1


def test_a_stringified_missing_compound_is_not_evidence_of_a_tyre_change():
    """`tyre_stint_repair`'s sentinel rule, imported rather than copied.

    The extractor stringifies an absent compound, so "MEDIUM" -> "nan" -> "MEDIUM"
    is data loss and must not read as two stops.
    """
    session = SessionLaps(
        2025,
        "Nowhere",
        _stint_frame(
            compounds=["MEDIUM", "MEDIUM", "nan", "unknown", "MEDIUM", "MEDIUM"],
            tyre_life=[1.0, 2.0, None, None, 5.0, 6.0],
            pit_in=[None] * 6,
            pit_out=[None] * 6,
        ),
        {"NOR": "4"},
    )

    assert session.masked_view({"NOR": 6}, 0.0)["drivers"]["NOR"]["stops"] == 0


def test_the_real_race_never_reports_a_stop_nobody_made():
    """The effect on the race that ships, not the mechanism.

    Melbourne 2025: the safety car led the field through the pit lane on laps 2,
    3 and 4, so `PitInTime` is set for all seventeen runners on each. At lap 24
    not one car has changed tyres; NOR's real stops are laps 35 and 45.

    Five cars read one stop high and it is #988's artefact, not this rule's: the
    feed republishes their `TyreLife` as 1 on one of the transits. They are named
    below rather than absorbed into a tolerance, so the day #988 lands this test
    fails and says which line to change. Everyone else is exact.
    """
    session = _session_or_skip()
    reveal = _all_revealed(session)
    # The exact five, with the transit that corrupts each: ALB and STR on lap 3,
    # LAW on lap 4, BEA and OCO on lap 5. See #988.
    republished = {"ALB", "BEA", "LAW", "OCO", "STR"}

    early = session.masked_view({code: 24 for code in reveal}, 0.0)["drivers"]
    for code, driver in early.items():
        expected = 1 if code in republished else 0
        assert driver["stops"] == expected, f"{code} at lap 24"
    # Counting in-laps gave the field 51 here, on a lap where nobody has stopped.
    assert sum(driver["stops"] for driver in early.values()) == len(republished)

    final = session.masked_view(reveal, 0.0)["drivers"]
    # Melbourne 2025 was wet-dry-wet, so every real stop is a compound change.
    assert final["NOR"]["stops"] == 2
    # ALO retired on lap 32, before his first real stop, having transited three times.
    assert final["ALO"]["stops"] == 0
    # SAI, DOO and HAD have only generated rows: no laps, no transits, no stops.
    assert [final[code]["stops"] for code in ("SAI", "DOO", "HAD")] == [0, 0, 0]
    assert sum(driver["stops"] for driver in final.values()) == 36, "31 real + #988's five"


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


def test_a_sector_of_this_lap_is_not_served_before_its_crossing():
    """The reveal rule at a finer coordinate, on the real race.

    `masked_view`'s `L <= laps_completed` is the rule for lap ROWS, which only
    exist once the lap is over. A sector carries its own `SectorNSessionTime`,
    so it has its own moment - and revealing it then is not look-ahead, it is
    the present.

    NOR's lap 23 at Melbourne crossed S1 at SessionTime 6689.966. Before that
    the cell must NOT hold lap 23's S1; the freshest thing it can honestly
    show is lap 22's, flagged as not-this-lap. A tenth of a second later it
    holds 31.865 and the 266 km/h measured at that trap, flagged as fresh.
    """
    session = _session_or_skip()
    global_t_min = 4260.355
    row = next(r for r in session._by_driver["NOR"] if r["lap"] == 23)
    earlier = next(r for r in session._by_driver["NOR"] if r["lap"] == 22)

    before = session.live_lap({"NOR": 22}, row["s1_at"] - 1 - global_t_min, global_t_min)["NOR"]
    after = session.live_lap({"NOR": 22}, row["s1_at"] + 0.1 - global_t_min, global_t_min)["NOR"]

    assert before["lap"] == 23 and after["lap"] == 23, "both probes are on the lap in progress"
    assert before["s1"] != row["s1"], "lap 23's S1 cannot be on screen before it was set"
    assert before["s1"] == earlier["s1"] and before["s1_fresh"] is False, (
        "what it shows is the previous lap's, and it says so"
    )
    assert before["v1"] == earlier["v1"], (
        "and the trap speed comes from the SAME row as the time it sits beside"
    )
    assert before["v1"] != row["v1"], "never this lap's speed under the previous lap's time"
    assert after["s1"] == row["s1"] and after["s1_fresh"] is True
    assert after["v1"] == row["v1"], "with the speed measured at that trap"
    assert after["s2"] != row["s2"], "the sectors after it are still the previous lap's"


def test_a_rewind_takes_this_laps_sectors_back():
    """The clock going back must UNDO a cell, not leave this lap's time in it.

    A cache that only ever filled would leave a number on screen for track the
    car has yet to re-drive - the same leak as a lap-row reveal that never
    un-reveals, one coordinate down. What the cell falls back to is the
    PREVIOUS lap's value, which the car really did set before this clock, and
    the flag says it is not from the lap in progress.
    """
    session = _session_or_skip()
    global_t_min = 4260.355
    row = next(r for r in session._by_driver["NOR"] if r["lap"] == 23)
    earlier = next(r for r in session._by_driver["NOR"] if r["lap"] == 22)

    late = session.live_lap({"NOR": 22}, row["s3_at"] + 1 - global_t_min, global_t_min)["NOR"]
    rewound = session.live_lap({"NOR": 22}, row["s1_at"] - 1 - global_t_min, global_t_min)["NOR"]

    assert [late["s1"], late["s2"], late["s3"]] == [row["s1"], row["s2"], row["s3"]]
    assert all(late[f"{s}_fresh"] for s in ("s1", "s2", "s3")), "all three were set on this lap"

    assert [rewound["s1"], rewound["s2"], rewound["s3"]] == [
        earlier["s1"],
        earlier["s2"],
        earlier["s3"],
    ], "every cell falls back to a lap the car had really finished by then"
    assert not any(rewound[f"{s}_fresh"] for s in ("s1", "s2", "s3"))


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


def test_every_sector_column_actually_shows_numbers_over_the_race():
    """The check that would have caught the S3 column being permanently empty.

    The first version of `live_lap` served only the lap in progress, which is
    right for S1 and S2 and impossible for S3: S3's crossing IS the end of the
    lap. Measured over all 920 real rows, `Sector3SessionTime` lands a median
    55 ms AFTER the lap's own crossing `Time`, and after it on 94.1 % of laps -
    so S1 was visible for 60.3 s of its lap, S2 for 40.8 s and S3 for
    -0.055 s. One of three columns was a dash for the entire race.

    The guards that missed it were fixture-shaped: the smoke harness hand-set
    `s3: null` and asserted a dash, so the test and the code agreed with each
    other and neither agreed with the race. This one samples the REAL clock
    across the REAL race and asks what a strategist would actually see.
    """
    session = _session_or_skip()
    global_t_min = 4260.355
    crossings = {
        code: {row["lap"]: row["time_s"] for row in rows if row["time_s"] is not None}
        for code, rows in session._by_driver.items()
    }
    starts = [t for laps in crossings.values() for t in laps.values()]
    first, last = min(starts), max(starts)

    filled = {"s1": 0, "s2": 0, "s3": 0}
    samples = 0
    for step in range(60):
        session_clock = first + (last - first) * step / 59
        laps_completed = {
            code: sum(1 for t in laps.values() if t <= session_clock)
            for code, laps in crossings.items()
        }
        live = session.live_lap(laps_completed, session_clock - global_t_min, global_t_min)
        for row in live.values():
            samples += 1
            for sector in filled:
                if row[sector] is not None:
                    filled[sector] += 1

    assert samples > 500, f"only {samples} driver-instants sampled; this proves nothing"
    for sector, count in filled.items():
        share = count / samples
        assert share > 0.75, (
            f"{sector} is filled on only {share:.1%} of {samples} driver-instants across the "
            f"race - a column a strategist never sees a number in"
        )


def test_a_carried_over_sector_says_it_is_not_from_this_lap():
    """Rolling the value is honest; passing it off as the current lap is not.

    Right after the line the freshest S3 a car has is the one that ENDED the
    lap it just finished, so the cell shows it - and `s3_fresh` is False, which
    is what the renderer dims. A cell that lied about which lap it belonged to
    would be a stale number wearing a live one's clothes, on a fidelity
    surface.
    """
    session = _session_or_skip()
    global_t_min = 4260.355
    lap23 = next(r for r in session._by_driver["NOR"] if r["lap"] == 23)
    lap24 = next(r for r in session._by_driver["NOR"] if r["lap"] == 24)

    # Just after lap 23 ended: lap 24 is in progress and has reached nothing.
    just_after = session.live_lap({"NOR": 23}, lap23["time_s"] + 1 - global_t_min, global_t_min)
    row = just_after["NOR"]

    assert row["lap"] == 24, "the lap in progress is the new one"
    assert row["s3"] == lap23["s3"], "and its S3 cell carries the sector that just ended lap 23"
    assert row["s3_fresh"] is False, "flagged as belonging to the previous lap"

    # Once lap 24's own S1 is crossed, THAT one is fresh and S3 still is not.
    later = session.live_lap({"NOR": 23}, lap24["s1_at"] + 0.1 - global_t_min, global_t_min)["NOR"]
    assert later["s1"] == lap24["s1"] and later["s1_fresh"] is True
    assert later["s3"] == lap23["s3"] and later["s3_fresh"] is False


def test_a_rewind_onto_the_same_open_sector_pattern_is_still_served():
    """The collision the old revision signature could not see (#934).

    `get_live_lap` used to key its revision on WHICH cells were filled. A
    (driver, lap) pair determines the values, but the lap entered that
    signature only as a constant True - so a rewind landing the whole field on
    the same open-sector pattern, with every number different, bumped nothing,
    answered None, and the client kept the FUTURE lap's sector times on a
    screen whose clock had gone back. The gate that found it measured 3,667
    such pairs at least ten seconds apart, the worst a 28-minute rewind across
    the wet start.

    This drives the REAL `get_live_lap` across a collision found by SEARCHING
    the real race, so it asserts the effect - what the second call returns -
    rather than re-implementing the comparison and checking its own arithmetic.
    """
    session = _session_or_skip()
    global_t_min = 4260.355
    crossings = {
        code: {row["lap"]: row["time_s"] for row in rows if row["time_s"] is not None}
        for code, rows in session._by_driver.items()
    }
    moments = sorted({t for laps in crossings.values() for t in laps.values()})

    def reveal_at(session_clock: float) -> dict[str, int]:
        return {
            code: sum(1 for t in laps.values() if t <= session_clock)
            for code, laps in crossings.items()
        }

    def pattern(session_clock: float) -> tuple:
        view = session.live_lap(
            reveal_at(session_clock), session_clock - global_t_min, global_t_min
        )
        return tuple(
            (code, tuple(value is not None for value in row.values()))
            for code, row in sorted(view.items())
        )

    late = moments[len(moments) * 3 // 4]
    late_pattern = pattern(late)
    early = next(
        (
            t
            for t in moments
            if late - t > 60
            and pattern(t) == late_pattern
            and session.live_lap(reveal_at(t), t - global_t_min, global_t_min)
            != session.live_lap(reveal_at(late), late - global_t_min, global_t_min)
        ),
        None,
    )
    assert early is not None, (
        "no same-pattern pair found on this race, so this guard would assert nothing"
    )

    client = _FakeClient()
    host = PitwallHost(client, window_count=1)

    def serve(session_clock: float, since_rev: int):
        client.latest = {
            "seq": 1,
            "arcade": {
                "year": 2025,
                "location": "Melbourne",
                "t": session_clock - global_t_min,
                "global_t_min": global_t_min,
                "drivers": {
                    code: {"laps_completed": laps}
                    for code, laps in reveal_at(session_clock).items()
                },
            },
        }
        return host.get_live_lap(since_rev)

    forward = serve(late, -1)
    assert forward is not None, "the first read must serve something"

    rewound = serve(early, forward["rev"])

    assert rewound is not None, (
        f"the clock went back {late - early:.0f} s onto the same open-sector pattern and the host "
        "answered None - the window would keep the FUTURE lap's sector times on screen"
    )
    assert rewound["drivers"] != forward["drivers"], "and it must serve the earlier numbers"


def test_no_sector_is_served_before_its_own_crossing_even_on_an_early_wire():
    """The invariant measured on the clock the WINDOW serves, not the tests' one.

    The carried-over branch first checked only that the previous lap HAD a
    stamp, with no clock gate - and the docstring claimed nothing is served
    before its own crossing. On the parquet-derived clock the tests use, that
    was true. On the wire it was not: the arcade's crossing map increments
    before that lap's own `Sector3SessionTime` on 837 of 921 laps (median
    39 ms, max 0.463 s), so the just-ended S3 went out before its official
    moment (#933 gate finding F6).

    The wire's lead is reproduced here by advancing `laps_completed` EARLY -
    by more than the measured worst case - which is exactly the shape of the
    real skew and needs no 382 MB pickle to exercise. A guard that looks like
    the reveal rule and checks something else is this repo's most expensive
    shape, so this one checks the rule itself: every value on screen has a
    stamp the clock has already passed.
    """
    session = _session_or_skip()
    global_t_min = 4260.355
    wire_lead = 0.6  # comfortably beyond the measured 0.463 s worst case

    stamps = {
        code: {row["lap"]: row for row in rows if not row["generated"]}
        for code, rows in session._by_driver.items()
    }
    crossings = {
        code: {lap: row["time_s"] for lap, row in laps.items() if row["time_s"] is not None}
        for code, laps in stamps.items()
    }
    moments = sorted({t for laps in crossings.values() for t in laps.values()})

    checked = 0
    for probe in range(0, len(moments), 7):
        session_clock = moments[probe]
        # The wire is EARLY: a lap counts as completed before the parquet says so.
        early_reveal = {
            code: sum(1 for t in laps.values() if t <= session_clock + wire_lead)
            for code, laps in crossings.items()
        }
        live = session.live_lap(early_reveal, session_clock - global_t_min, global_t_min)
        for code, row in live.items():
            lap_rows = stamps[code]
            for sector, _speed, crossed_at in (
                ("s1", "v1", "s1_at"),
                ("s2", "v2", "s2_at"),
                ("s3", "vfl", "s3_at"),
            ):
                if row[sector] is None:
                    continue
                source = row["lap"] if row[f"{sector}_fresh"] else row["lap"] - 1
                moment = lap_rows.get(source, {}).get(crossed_at)
                assert moment is not None and moment <= session_clock, (
                    f"{code} shows {sector}={row[sector]} from lap {source} at clock "
                    f"{session_clock:.3f}, but that sector was crossed at {moment}"
                )
                checked += 1

    assert checked > 2000, f"only {checked} served sectors examined; this proves nothing"


def test_pointing_the_arcade_at_a_race_with_no_laps_clears_the_sectors():
    """The twin: `get_bulk` had this branch and `get_live_lap` did not.

    A missing parquet makes the table say `available=False`, which is
    deliberate - "a tower rendering zero rows silently is the same pixel as a
    tower whose reveal is broken". Its sibling answered plain `None`, and
    `None` means "keep what you have" to the client, so switching races left
    the PREVIOUS race's sector times, dimming flags and colours on the new
    race's rows indefinitely, beside a table that had correctly gone blank.

    Reachable exactly the way `_session_for`'s own docstring says race
    switches are - the stale-state class #904 already paid for once.
    """

    def tick_for(location: str, laps_completed: int) -> dict:
        return {
            "seq": 1,
            "arcade": {
                "year": 2025,
                "location": location,
                "t": 3000.0,
                "global_t_min": 0.0,
                "drivers": {"NOR": {"laps_completed": laps_completed}},
            },
        }

    client = _FakeClient(tick_for("Melbourne", 20))
    host = PitwallHost(client, window_count=1)

    melbourne = host.get_live_lap(-1)
    if melbourne is None or not melbourne["drivers"]:
        pytest.skip("2025/Melbourne is not in this install's curated data set")
    assert melbourne["drivers"]["NOR"]["lap"] == 21

    client.latest = tick_for("Shanghai", 20)
    switched = host.get_live_lap(melbourne["rev"])

    assert switched is not None, (
        "a race with no laps must be SAID, not answered with 'keep what you have' - "
        "the tower would hold the previous race's sectors on the new race's rows"
    )
    assert switched["drivers"] == {}, "and what it says is: nothing to show"
    assert switched["rev"] != melbourne["rev"], "with a revision the client will accept"

    assert host.get_live_lap(switched["rev"]) is None, (
        "served once, not re-sent on every poll of a race that has no laps"
    )
