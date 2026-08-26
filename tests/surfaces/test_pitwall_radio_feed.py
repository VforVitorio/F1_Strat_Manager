"""The RADIO channel: `src/pitwall/radio_feed.py`, served inside the bulk payload.

Every assertion is about an EFFECT the panel would show, not about a constant
someone chose. The one that matters most is the first: it pins that the feed
carries DRIVER RADIO with real three-letter codes. Before it existed the reader
took its `{number: code}` map from `RadioPipelineRunner`, whose mapper slices on
a `GP_Name` column only the FEATURED parquet has - so against the raw table it
swallowed its own exception, returned `{}`, and every radio came out as the
synthetic `D12`. Nothing raised, nothing logged at ERROR, the RCM half worked
perfectly, and 0 of Melbourne's 14 driver radios were ever revealed. A test that
only counted events would have been green through all of it.

Runs against the REAL Melbourne 2025 corpus when it is on disk and skips when it
is not, exactly as the bulk's own suite does: a curated install holds one race of
seventy, and the radio corpus is a separate artefact that can be absent on its own.
"""

from __future__ import annotations

import pytest

from src.f1_strat_manager.data_cache import get_data_root
from src.pitwall.host import PitwallHost
from src.pitwall.radio_feed import RadioCorpus, unavailable
from src.pitwall.session_data import SessionLaps

MELBOURNE = (2025, "Melbourne")


def _corpus_or_skip() -> RadioCorpus:
    corpus = RadioCorpus.load(get_data_root(), *MELBOURNE)
    if corpus is None:
        pytest.skip("the 2025/Melbourne radio corpus is not in this install")
    return corpus


def _codes_or_skip() -> list[str]:
    session = SessionLaps.load(get_data_root(), *MELBOURNE)
    if session is None:
        pytest.skip("2025/Melbourne is not in this install's curated data set")
    return list(session.masked_view({}, 0.0)["drivers"])


def _events(corpus: RadioCorpus, reveal: dict[str, int]) -> list[dict]:
    return corpus.masked_view(reveal)["events"]


def _of_kind(events: list[dict], kind: str) -> list[dict]:
    return [event for event in events if event["kind"] == kind]


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


def test_the_feed_carries_driver_radio_under_real_driver_codes():
    """The whole race's radio, attributed to drivers the tower also shows.

    The assertion is deliberately on the CODES and not on the count. A count
    survives the `D12` degradation described in the module docstring; a code
    does not, because `D12` is not a driver on any grid and never matches a row
    in the timing tower beside it.
    """
    corpus = _corpus_or_skip()
    codes = _codes_or_skip()
    everything = _events(corpus, {code: 999 for code in codes})

    radios = _of_kind(everything, "radio")
    assert radios, "the race's driver radio never reaches the feed"
    speakers = {event["driver"] for event in radios}
    # Membership in the grid is the whole assertion. A length check was here
    # too and it was decoration: the degradation this file exists to pin
    # produced `D12`, which is three characters long, so it satisfied the very
    # rule it was written to break.
    assert speakers <= set(codes), f"radio attributed to non-drivers: {speakers - set(codes)}"


def test_a_radio_waits_for_its_own_driver_and_an_rcm_waits_for_the_leader():
    """The two reveal coordinates, on a RAGGED field - which is the normal one.

    At 96% of instants the running field spans two or three different laps, so
    a feed cut at one shared lap leaks for the cars behind and lags the leaders
    at the same time. Here one driver is deep into the race and everyone else is
    at lap 5: his radio must be on screen and theirs must not, while race
    control - which is not any one car's - follows the leader.
    """
    corpus = _corpus_or_skip()
    codes = _codes_or_skip()
    ahead = "NOR"
    ragged = dict.fromkeys(codes, 5)
    ragged[ahead] = 24

    events = _events(corpus, ragged)
    for radio in _of_kind(events, "radio"):
        limit = 24 if radio["driver"] == ahead else 5
        assert radio["lap"] <= limit, (
            f"{radio['driver']}'s lap-{radio['lap']} radio is on screen "
            f"while he has completed {limit}"
        )
    assert any(
        radio["driver"] == ahead and radio["lap"] > 5 for radio in _of_kind(events, "radio")
    ), "the driver who is 19 laps ahead gained nothing, so the reveal is not per driver"
    for rcm in _of_kind(events, "rcm"):
        assert rcm["lap"] <= 24, "race control ran ahead of the leader"


def test_nothing_is_served_before_its_lap_is_over():
    """Sweep every lap of the race: no event ever appears early.

    The direction that matters. The coordinate is coarse - an event shows when
    its lap is COMPLETE, so it is up to a lap late - and late is the honest
    failure here, because the parquet stamps events in UTC and this window's
    clock is SessionTime.
    """
    corpus = _corpus_or_skip()
    codes = _codes_or_skip()
    for lap in range(0, 58):
        for event in _events(corpus, dict.fromkeys(codes, lap)):
            assert event["lap"] <= lap, (
                f"a {event['kind']} from lap {event['lap']} is on screen at lap {lap}"
            )


def test_a_rewind_takes_events_back_off_the_feed():
    """The reason it is read from disk rather than gathered off the wire.

    An accumulating feed grows only, so a seek backwards would leave messages
    on screen from a part of the race the clock has un-run.
    """
    corpus = _corpus_or_skip()
    codes = _codes_or_skip()
    late = _events(corpus, dict.fromkeys(codes, 40))
    rewound = _events(corpus, dict.fromkeys(codes, 5))

    assert len(rewound) < len(late), "the feed did not shrink across a rewind"
    assert all(event["lap"] <= 5 for event in rewound)


def test_race_control_messages_do_not_print_their_car_twice():
    """An RCM names its car inside the text, so the row carries no separate code.

    Measured across all 24 races on disk: of the 492 rows carrying a
    `driver_number`, 492 spell that number - and its code - inside the message.
    """
    corpus = _corpus_or_skip()
    codes = _codes_or_skip()
    rcms = _of_kind(_events(corpus, {code: 999 for code in codes}), "rcm")

    assert rcms, "the race's control messages never reach the feed"
    assert all(rcm["driver"] is None for rcm in rcms)
    assert all(rcm["text"] for rcm in rcms), "a race control row reached the panel with no words"


def test_a_race_with_no_corpus_says_so_instead_of_going_quiet():
    """Absent data is a state, not a blank.

    The twin already found on this same shape: `get_bulk` had an
    explicit unavailable payload and `get_live_lap` did not, so pointing the
    arcade at a race with no parquet left the PREVIOUS race's numbers on screen.
    A feed that answered with an empty list would repeat it one channel over.
    """
    assert unavailable() == {"available": False, "events": []}
    assert RadioCorpus.load(get_data_root(), 2025, "Shanghai") is None


def test_the_feed_rides_in_the_bulk_and_a_race_switch_empties_it():
    """Served through `get_bulk`, and gone when the race it belongs to is.

    It has no revision of its own on purpose: it is a function of exactly what
    signs the bulk - (year, location, reveal map) - and a second signature that
    does not determine its payload is what #934 cost a sprint.
    """
    codes = _codes_or_skip()
    _corpus_or_skip()
    client = _FakeClient(_tick(dict.fromkeys(codes, 24)))
    host = PitwallHost(client, window_count=1)

    served = host.get_bulk(-1)
    assert served["radio"]["available"] is True
    assert served["radio"]["events"], "the bulk went out without the feed"

    client.latest = _tick(dict.fromkeys(codes, 24), location="Shanghai")
    switched = host.get_bulk(served["rev"])
    assert switched["available"] is False, "the table did not follow the race switch"
    assert switched["radio"] == {"available": False, "events": []}, (
        "the previous race's radio stayed on screen beside a table that had gone empty"
    )


def test_a_tick_that_names_no_race_takes_the_feed_down_with_the_table():
    """The malformed-tick return has to clear the corpus, and it did not.

    `_session_for` returns early when the tick's year or location is not the
    type it expects - above the block that reloads both the laps and the radio.
    So the table fell to its unavailable payload while `_masked_view` went on
    serving the PREVIOUS race's messages out of a corpus the early return had
    skipped: 46 of them, beside a table that had already given up.

    Not reachable from today's producer, which always publishes an int year and
    a str location. It is guarded because the branch exists to be defensive and
    was not, and because this is the exact twin shape the sprint before paid
    for between these same two channels.
    """
    codes = _codes_or_skip()
    _corpus_or_skip()
    client = _FakeClient(_tick(dict.fromkeys(codes, 24)))
    host = PitwallHost(client, window_count=1)

    served = host.get_bulk(-1)
    assert served["radio"]["events"], "the feed never loaded, so the case cannot be exercised"

    malformed = _tick(dict.fromkeys(codes, 24))
    malformed["arcade"].pop("location")
    client.latest = malformed
    answered = host.get_bulk(served["rev"])

    assert answered["available"] is False, "the table did not go to its unavailable payload"
    assert answered["radio"] == {"available": False, "events": []}, (
        "the table went empty and the previous race's radio stayed on screen beside it"
    )
