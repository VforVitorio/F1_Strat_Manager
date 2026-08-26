"""The PITWALL host and its stream client (`src/pitwall/`).

The delivery plan named three things it had to get right because they are
expensive to change later. Two of them are here; the third (design-token
drift) is `test_pitwall_tokens.py`.

Nothing in this file imports pywebview. Only `src/pitwall/__main__.py` does,
so a machine with no system webview still runs the suite.
"""

from __future__ import annotations

import json
import re
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from src.pitwall.host import PitwallHost
from src.pitwall.stream_client import SIGNAL_LOG_DEPTH, ArcadeStreamClient
from tests.surfaces.fake_stream_client import FakeStreamClient as _FakeClient


def _tick(seq: int) -> dict:
    return {"seq": seq, "schema_version": 1, "arcade": {"lap": 12}}


# --- Trap 1: the tick is sequenced, never a blind slot -----------------------


def test_a_window_only_gets_a_tick_it_has_not_seen():
    """The whole reason `since_seq` exists.

    Two windows polling one latest-payload slot on independent 10 Hz timers
    were measured reading a different frame on 58% of polls, with 15
    duplicate reads and 15 skips out of 54. The parameter removes both.
    """
    client = _FakeClient(_tick(7))
    host = PitwallHost(client, window_count=2)

    assert host.get_tick(since_seq=-1) == _tick(7), "a window with nothing gets the latest"
    assert host.get_tick(since_seq=6) == _tick(7), "a window one behind gets it"
    assert host.get_tick(since_seq=7) is None, "a window up to date gets nothing new"


def test_a_restarted_producer_is_followed_rather_than_withheld():
    """The one way the slot can hold a LOWER sequence than the window's own.

    An earlier version asked for `seq > since_seq` and read a lower sequence
    as "an older payload, do not hand it back". The slot only ever holds the
    newest payload, so that state is unreachable except by a restart -- and
    there, withholding froze both windows on the dead race, `live` and
    silent, for as long as the previous run had lasted.

    The old test asserted the CONSTANT (never hand back a lower seq) rather
    than the EFFECT (the window must follow a restarted producer), so the
    freeze was pinned rather than caught.
    """
    client = _FakeClient(_tick(3))
    host = PitwallHost(client, window_count=2)

    assert host.get_tick(since_seq=400) == _tick(3), "a relaunched arcade must reach the window"


def test_two_windows_polling_independently_never_disagree():
    """The measured failure, replayed: alternating polls across a moving slot.

    Both windows must see every sequence exactly once. Against a blind slot
    one would skip what the other duplicated.
    """
    client = _FakeClient()
    host = PitwallHost(client, window_count=2)
    seen: dict[str, list[int]] = {"data": [], "agents": []}
    last: dict[str, int] = {"data": -1, "agents": -1}

    for seq in range(1, 31):
        client.latest = _tick(seq)
        # The two windows poll on their own timers: interleaved, and one of
        # them polls twice in a row for a third of the ticks.
        for window in ("data", "agents", "data" if seq % 3 == 0 else "agents"):
            tick = host.get_tick(last[window])
            if tick is not None:
                seen[window].append(tick["seq"])
                last[window] = tick["seq"]

    assert seen["data"] == list(range(1, 31))
    assert seen["agents"] == list(range(1, 31))
    assert seen["data"] == seen["agents"], "the two windows must not diverge"


def test_nothing_received_yet_is_none_rather_than_an_empty_tick():
    assert PitwallHost(_FakeClient(None), window_count=2).get_tick(-1) is None


def test_a_payload_with_no_sequence_is_returned_rather_than_withheld():
    """Only reachable against a producer older than this repo's.

    There is nothing to compare, and withholding the data would leave the
    window blank forever with no error.
    """
    host = PitwallHost(_FakeClient({"arcade": {"lap": 3}}), window_count=1)

    assert host.get_tick(since_seq=999) == {"arcade": {"lap": 3}}


# --- The eviction signals survive a discarded tick (#1060) -------------------
#
# `rewound` and `dropped` describe the gap BETWEEN two ticks rather than the
# state of one, so the latest-payload slot - which is right to keep only the
# newest snapshot - is wrong to drop them with the tick that carried them.
# Measured before the fix: 6 of 905 published ticks (0.7%) were never served to
# a window polling the way `useTick` polls.


def _signalling_tick(seq: int, rewound: bool = False, dropped: int = 0) -> dict:
    """A tick shaped like the producer's, carrying its continuity flags."""
    return {
        "seq": seq,
        "schema_version": 2,
        "arcade": {
            "lap": 12,
            "telemetry": {"drivers": {}, "rewound": rewound, "dropped": dropped},
        },
    }


def _telemetry_of(tick: dict) -> dict:
    return tick["arcade"]["telemetry"]


def test_a_rewind_on_a_tick_the_slot_discarded_still_reaches_the_window():
    """The defect, at its smallest: one tick published, overwritten, never served.

    Before the fix the window saw only the second tick, whose own `rewound` is
    False, so `FrameClock` reported `continuous` across the hole and the trace
    buffer kept appending samples from two unrelated parts of the race.
    """
    client = _FakeClient(_signalling_tick(1))
    host = PitwallHost(client, window_count=1)
    assert host.get_tick(since_seq=-1)["seq"] == 1, "the window is caught up at seq 1"

    client.receive(_signalling_tick(2, rewound=True))  # published...
    client.receive(_signalling_tick(3))  # ...and overwritten before any poll

    served = host.get_tick(since_seq=1)

    assert served["seq"] == 3, "the newest snapshot is still the one served"
    assert _telemetry_of(served)["rewound"] is True, (
        "the rewind rode on seq 2, which no window ever received"
    )


def test_dropped_frames_are_SUMMED_across_every_tick_the_window_missed():
    """`dropped` is a count, not a flag: two discarded jumps are both real."""
    client = _FakeClient(_signalling_tick(1))
    host = PitwallHost(client, window_count=1)
    host.get_tick(since_seq=-1)

    client.receive(_signalling_tick(2, dropped=250))
    client.receive(_signalling_tick(3, dropped=40))
    client.receive(_signalling_tick(4))

    served = host.get_tick(since_seq=1)

    assert _telemetry_of(served)["dropped"] == 290, "250 + 40, not the newest and not 1"


def test_two_cursors_each_get_their_OWN_missed_range():
    """The assertion that fails under any drain-once design.

    `get_tick` has more than one caller - `useTick`, `get_agents_view` at
    host.py:189, and the loopback server's `/api/tick` - each with its own
    cursor. A pending slot cleared by the first caller hands the signal to
    whoever polled first and hides it from everyone else. That is #950 in
    another field: `lib/agents.ts` carries the comment describing exactly this
    failure for the connection label.

    AGENTS never reads these flags, and that does not save the design: the
    DRAIN, not the consumption, is what empties a slot.
    """
    client = _FakeClient(_signalling_tick(1))
    host = PitwallHost(client, window_count=2)
    host.get_tick(since_seq=-1)  # both windows start caught up at seq 1
    host.get_tick(since_seq=-1)

    client.receive(_signalling_tick(2, dropped=100))
    client.receive(_signalling_tick(3))

    first = host.get_tick(since_seq=1)
    second = host.get_tick(since_seq=1)

    assert _telemetry_of(first)["dropped"] == 100
    assert _telemetry_of(second)["dropped"] == 100, (
        "the second window's range is its own; the first did not consume it"
    )


def test_the_fold_does_not_mutate_the_payload_another_window_is_still_holding():
    """The copy-on-fold rule, and why `get_tick` cannot fold in place.

    Every caller is handed the same dict object out of the slot. Window A folds
    its range in; window B then polls with a DIFFERENT cursor and would rewrite
    the same telemetry block to its own, shorter range - so A's payload silently
    changes underneath it, before A has serialised it across the bridge.
    """
    client = _FakeClient(_signalling_tick(1))
    host = PitwallHost(client, window_count=2)
    host.get_tick(since_seq=-1)

    client.receive(_signalling_tick(2, dropped=100))
    client.receive(_signalling_tick(3))

    held_by_a = host.get_tick(since_seq=1)
    a_dropped_when_served = _telemetry_of(held_by_a)["dropped"]

    host.get_tick(since_seq=2)  # window B, a shorter range, same underlying dict

    assert a_dropped_when_served == 100
    assert _telemetry_of(held_by_a)["dropped"] == 100, (
        "another caller's fold rewrote the block this one is holding"
    )
    assert _telemetry_of(client.latest)["dropped"] == 0, (
        "the slot's own payload must never be edited by a read"
    )


def test_a_cursor_the_log_cannot_place_is_served_unmerged_rather_than_invented():
    """A first poll, or a tab asleep past the log's 6.4 s, has no knowable range.

    Fabricating a `dropped` for it would put a made-up number in the one field
    whose entire job is to say that something real happened.
    """
    client = _FakeClient(_signalling_tick(1, dropped=7))
    host = PitwallHost(client, window_count=1)

    for _ in range(SIGNAL_LOG_DEPTH + 5):
        client.receive(_signalling_tick(client.latest["seq"] + 1, dropped=3))

    served = host.get_tick(since_seq=1)  # seq 1 fell off the end of the log

    assert _telemetry_of(served)["dropped"] == 3, "the payload's own flags, unmerged"
    assert _telemetry_of(served)["rewound"] is False


def test_the_signal_log_and_the_slot_are_read_as_ONE_snapshot():
    """The atomicity contract, asserted on the invariant it produces.

    Two lock acquisitions let `_consume` publish between them, and the fold then
    picks up ticks NEWER than the payload being served while the caller's cursor
    only advances to that payload - so the same entries are folded again on the
    next poll. Measured against a live producer, that over-counts `dropped` by
    66%, and every phantom count is a spurious eviction of the buffer this fix
    exists to protect.

    The observable invariant: whatever `snapshot` returns, its last log entry
    describes the payload beside it.

    **This has to run against a CONTENDING publisher.** A client fed by the slow
    one-shot server asserts the same invariant, and the two-lock mutant survives
    it, because nothing is publishing in the window between the two
    acquisitions. A guard for a race that never runs the race is the empty set
    wearing a thread.
    """
    client = ArcadeStreamClient("127.0.0.1", 0)  # never started: `_consume` is driven here
    lines = [json.dumps(_signalling_tick(i, dropped=1)).encode() + b"\n" for i in range(1, 400)]
    stop = threading.Event()

    def publish() -> None:
        while not stop.is_set():
            for line in lines:
                if stop.is_set():
                    return
                client._consume(line)

    # The window between two lock acquisitions is a handful of bytecodes, and the
    # default 5 ms switch interval means a publisher thread almost never lands in
    # it - which is how the two-lock mutant survived the first version of this
    # guard. Forcing frequent switches is what makes the race actually run.
    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    publisher = threading.Thread(target=publish, daemon=True)
    publisher.start()
    try:
        assert _wait_for(lambda: client.latest is not None), "the publisher never ran"
        torn = 0
        for _ in range(20_000):
            payload, signals = client.snapshot()
            if payload is None:
                continue
            assert signals, "a payload is in the slot but the log is empty"
            if signals[-1].seq != payload["seq"]:
                torn += 1
        assert torn == 0, (
            f"{torn} of 20,000 snapshots had a log newer than the payload beside it; "
            "the fold would count those entries again on the next poll"
        )
    finally:
        stop.set()
        publisher.join(timeout=2.0)
        sys.setswitchinterval(previous_interval)


def test_the_signal_log_dies_with_the_connection():
    """A relaunched arcade restarts `seq` at 1, so two runs must not share a log.

    With both runs resident, a cursor holding an old-run number can match a
    new-run entry and the fold spans a range that never existed.
    """
    port, _ = _serve_lines(
        [json.dumps(_signalling_tick(i, dropped=5)).encode() + b"\n" for i in (1, 2)]
    )
    client = ArcadeStreamClient("127.0.0.1", port)
    client.start()
    try:
        assert _wait_for(lambda: len(client.snapshot()[1]) == 2), "both ticks did not arrive"
        assert _wait_for(lambda: client.snapshot()[1] == ()), (
            "the log outlived the socket that produced it"
        )
        assert client.latest is not None, (
            "the last payload is KEPT - a frozen board is still readable, and the "
            "windows render it dimmed; only the continuity signals are dropped"
        )
    finally:
        client.stop()


# --- Trap 2: closing one window must not stop the shared client --------------


def test_closing_one_window_leaves_the_other_one_live():
    """The single place this property can regress, and it regresses silently.

    The client belongs to the host, not to a window. If a window's `closed`
    event were wired to `client.stop`, closing DATA would blind AGENTS with
    nothing in any log.
    """
    client = _FakeClient(_tick(4))
    host = PitwallHost(client, window_count=2)
    host.start()

    remaining = host.release_window()

    assert remaining == 1
    assert client.stopped is False, "one window closing must not stop the client"
    assert host.get_tick(since_seq=-1) == _tick(4), "the surviving window still gets ticks"


def test_the_last_window_closing_does_stop_the_client():
    client = _FakeClient(_tick(4))
    host = PitwallHost(client, window_count=2)
    host.start()

    host.release_window()
    remaining = host.release_window()

    assert remaining == 0
    assert client.stopped is True


def test_release_never_goes_negative():
    """Closing is reported by the toolkit; nothing guarantees it fires once."""
    client = _FakeClient()
    host = PitwallHost(client, window_count=1)

    assert host.release_window() == 0
    assert host.release_window() == 0


# --- The client itself, against a real socket --------------------------------


def _serve_lines(lines: list[bytes]) -> tuple[int, threading.Thread]:
    """A one-shot server that writes `lines` and holds the connection open."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def run() -> None:
        conn, _ = listener.accept()
        for line in lines:
            conn.sendall(line)
            time.sleep(0.01)
        time.sleep(0.5)
        conn.close()
        listener.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return port, thread


def _wait_for(predicate, timeout=3.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_the_client_publishes_the_latest_complete_payload():
    port, _ = _serve_lines([json.dumps(_tick(i)).encode() + b"\n" for i in (1, 2, 3)])
    client = ArcadeStreamClient("127.0.0.1", port)
    client.start()
    try:
        assert _wait_for(lambda: (client.latest or {}).get("seq") == 3)
    finally:
        client.stop()


def test_a_payload_split_across_reads_is_never_published_half_way():
    """TCP can split anywhere, and half a JSON object is not a tick."""
    whole = json.dumps(_tick(9)).encode() + b"\n"
    port, _ = _serve_lines([whole[:20], whole[20:]])
    client = ArcadeStreamClient("127.0.0.1", port)
    client.start()
    try:
        assert _wait_for(lambda: client.latest is not None)
        assert client.latest["seq"] == 9
    finally:
        client.stop()


def test_an_unparseable_line_costs_that_tick_and_not_the_connection():
    """The next tick is 100 ms away and carries the whole state again."""
    port, _ = _serve_lines(
        [b"{not json}\n", json.dumps(_tick(5)).encode() + b"\n"],
    )
    client = ArcadeStreamClient("127.0.0.1", port)
    client.start()
    try:
        assert _wait_for(lambda: (client.latest or {}).get("seq") == 5)
    finally:
        client.stop()


def test_the_client_waits_for_an_arcade_that_is_not_listening_yet():
    """PITWALL is spawned right after the server binds, so it races it."""
    client = ArcadeStreamClient("127.0.0.1", 1)  # nothing listens on port 1
    client.start()
    try:
        time.sleep(0.3)
        assert client.latest is None
        assert client.connected is False
    finally:
        client.stop()  # must return rather than hang on a never-connected socket


def test_stop_is_safe_before_start_and_twice():
    client = ArcadeStreamClient("127.0.0.1", 1)
    client.stop()
    client.start()
    client.stop()
    client.stop()


@pytest.mark.parametrize("window_count", [1, 2])
def test_shutdown_stops_the_client_whatever_is_open(window_count):
    """The process exiting is not a window closing, and must not depend on the count."""
    client = _FakeClient()
    host = PitwallHost(client, window_count=window_count)
    host.start()

    host.shutdown()

    assert client.stopped is True


# --- Window geometry against a real desktop ---------------------------------
#
# A window bigger than the desktop is not scrolled, it is CLIPPED, and the part
# that goes missing is the bottom - where both PITWALL windows keep their
# status bar. Found by opening the real windows on a 2560x1440 display at
# 150%, which is 1707x960 logical with a 912-pixel work area: DATA asks for
# 950 and its status bar rendered under the taskbar, with the bottom row's
# "Distance (m)" axis label sliced in half.
#
# No headless screenshot can catch this. A Playwright viewport is exactly the
# size requested and has no desktop to not fit on.


def test_every_window_opens_fully_inside_the_work_area():
    """The measured case: both windows on the reference desktop.

    1707x960 logical is a 2560x1440 panel at 150%, and its work area is 912
    tall. DATA asks for 950, so its status bar rendered under the taskbar and
    the bottom row's "Distance (m)" label was sliced in half.

    Asserted as `y + height`, not as `height`, because clamping the size ALONE
    did not fix it: pywebview cascades, the second window started 38 pixels
    lower and fell straight back off the bottom. A height-only assertion was
    green while AGENTS was still broken.
    """
    from src.pitwall.config import WINDOWS

    for index, spec in enumerate(WINDOWS):
        x, y, width, height = spec.place(index, 1707, 960)
        assert y + height <= 912, f"{spec.key} reaches {y + height}, past the 912 work area"
        assert x + width <= 1707, f"{spec.key} reaches {x + width}, past the 1707 screen"


def test_a_big_screen_does_not_inflate_a_window():
    """`place` clamps, it never grows.

    A `WindowSpec` is a layout decision - AGENTS mirrors the 540 + 740 Qt
    columns it ports - and a 4K monitor is not a reason to override it.
    """
    from src.pitwall.config import WINDOWS

    for index, spec in enumerate(WINDOWS):
        _, _, width, height = spec.place(index, 3840, 2160)
        assert (width, height) == (spec.width, spec.height)


def test_every_shipped_window_fits_a_small_laptop():
    """1366x768 is the floor this has to survive, and both windows exceed it.

    Asserting the EFFECT - where the window actually lands - rather than the
    constant. A test that pinned `height == 950` would have passed for the
    whole life of the defect.
    """
    from src.pitwall.config import WINDOWS

    for index, spec in enumerate(WINDOWS):
        x, y, width, height = spec.place(index, 1366, 768)
        assert x + width <= 1366 and y + height <= 768 - 90, (
            f"{spec.key} opens at {x},{y} {width}x{height}"
        )


@pytest.mark.parametrize("screen", [(1707, 960), (1366, 768), (3840, 2160)])
def test_the_second_window_is_still_grabbable(screen):
    """The two windows overlap - they must, at these sizes - so the one
    underneath needs its title bar reachable. A zero stagger would bury it.

    Run over three screens rather than one. Pinned to 1707x960 alone this was
    green through the whole life of the defect it names: on a 1366 laptop both
    windows are clamped to the screen width, and an `x` that clamps against
    the FULL window width collapses the stagger to zero, putting the top
    window exactly over the bottom one's title bar. `place` narrows the
    staggered window instead.
    """
    from src.pitwall.config import WINDOWS

    origins = [spec.place(index, *screen)[0] for index, spec in enumerate(WINDOWS)]
    assert len(set(origins)) == len(origins), (
        f"two windows share an origin on {screen[0]}x{screen[1]}: {origins}"
    )


# The desktop every placement number here is measured on: a 2560x1440
# panel at 150%, which is 1707x960 logical.
_REFERENCE_SCREEN = (1707, 960)

# What the OS keeps for itself BETWEEN a placed window and the page inside it,
# as opposed to `_OS_CHROME_ALLOWANCE_PX`, which is what it keeps around the
# window. The 37 is the title bar `config.py` already names; the 14 is the
# frame either side.
#
# **Measured, not remembered.** Both windows were opened on the reference
# desktop and asked with `evaluate_js`: `innerWidth`, `documentElement
# .clientWidth` and `body.clientWidth` all report **1486x833** from a placed
# 1500x870, twice, at `devicePixelRatio` 1.5 and with no overflow in either
# axis. The earlier figure was
# 1485 - one pixel short, and short in the direction that hides a clip.
_WINDOW_FRAME_PX = 14
_TITLE_BAR_PX = 37

# Catches both `viewport: { width: N, height: M }` and a harness's own declared
# client constant, which is the only other place these numbers may appear.
_SIZE_LITERAL = re.compile(r"width:\s*(\d+),\s*height:\s*(\d+)")

_HARNESSES = (
    "smoke-agents.mjs",
    "shot-agents.mjs",
    "smoke-data.mjs",
    "shot-data.mjs",
)


def _reference_client(index: int, spec) -> tuple[int, int]:
    _, _, width, height = spec.place(index, *_REFERENCE_SCREEN)
    return width - _WINDOW_FRAME_PX, height - _TITLE_BAR_PX


def test_no_harness_measures_a_surface_larger_than_the_product_has():
    """A viewport wider or taller than the real client cannot see a clip.

    This is the shape of defect the programme keeps paying for: a check that
    sits at the only size where the thing it guards cannot happen. Three of
    the four harnesses were sized from the OUTER `WindowSpec` rather than from
    the client the page actually receives - `smoke-agents.mjs` measured
    1320x900 against a real 833, so **67 px of vertical overflow was invisible
    to every assertion in it**, and `smoke-data.mjs` was the only one already
    right.

    Asserted as an inequality, not an equality, because a harness may
    deliberately drive a SMALLER client (`smoke-data.mjs` sweeps a list of
    them). Measuring bigger is the defect; measuring smaller is a test.
    """
    from src.pitwall.config import WINDOWS

    scripts = Path(__file__).resolve().parents[2] / "src" / "pitwall" / "ui" / "scripts"
    clients = [_reference_client(index, spec) for index, spec in enumerate(WINDOWS)]
    widest = max(width for width, _ in clients)
    tallest = max(height for _, height in clients)

    seen = 0
    for name in _HARNESSES:
        source = (scripts / name).read_text(encoding="utf-8")
        for width, height in _SIZE_LITERAL.findall(source):
            seen += 1
            assert int(width) <= widest and int(height) <= tallest, (
                f"{name} measures {width}x{height}, larger than the real client {widest}x{tallest}"
            )
    # The enumeration itself, so this cannot pass by finding nothing.
    assert seen >= len(_HARNESSES), f"only {seen} viewport literals found across {_HARNESSES}"


_NEW_PAGE = re.compile(r"(?:const|let)\s+(\w+)\s*=\s*await\s+\w+\.newPage\(\)")
_WATCHED = re.compile(r"watchPage\((\w+),")


def test_every_harness_page_is_watched_for_console_errors():
    """The invariant that makes a missing bridge stub visible at all.

    `bridge.ts` calls `window.pywebview.api.<method>` when the shell provides
    one and falls back to `fetch("/api/...")` when it does not, and the
    fallback swallows a bad status on purpose (`if (!response.ok) return
    null`) because in the product that is a server restarting. So a stub
    missing a method throws nothing and fails no assertion: the window renders
    the unknown state and chromium logs one console error.

    Four stubs had drifted that way and **18 of the 22 pages
    across the four harnesses had no console listener**, so none of them could
    have reported it. Watching every page is what turns the whole class from
    silent into loud, which is why this asserts the listener rather than any
    particular method: the next hook to be wired will be some other method.

    Counts the pages first, so it cannot pass by finding none.
    """
    scripts = Path(__file__).resolve().parents[2] / "src" / "pitwall" / "ui" / "scripts"

    pages = 0
    unwatched: list[str] = []
    for name in _HARNESSES:
        source = (scripts / name).read_text(encoding="utf-8")
        watched = set(_WATCHED.findall(source))
        for variable in _NEW_PAGE.findall(source):
            pages += 1
            if variable not in watched:
                unwatched.append(f"{name}:{variable}")

    assert pages >= len(_HARNESSES), f"only {pages} pages found across {_HARNESSES}"
    assert not unwatched, f"pages with no console listener: {unwatched}"


def test_both_windows_hand_the_page_the_same_client_area():
    """One surface, one number for the harnesses to point at.

    The two windows used to differ: AGENTS was 1320 wide because it mirrored
    the Qt strategy window's 540 + 740 columns plus the frame. That split is
    gone, so the width had no argument behind it and the two harness families
    had two numbers to drift apart.
    """
    from src.pitwall.config import WINDOWS

    clients = {spec.key: _reference_client(index, spec) for index, spec in enumerate(WINDOWS)}
    assert len(set(clients.values())) == 1, f"windows disagree about the client area: {clients}"


def test_the_socket_state_has_one_colour_across_both_windows():
    """The twin this closes, and the shape it had.

    The AGENTS window took the chip's colour from `CONNECTION_COLOURS` through
    the view; the DATA strip mapped the same three words to CSS classes of its
    own. They agreed on two states and **disagreed on the third**: dim grey
    here, WARNING amber there, for one socket, on two windows a reader has open
    side by side.

    The colour rides with the word now, from one lookup. This asserts the
    property that makes the drift impossible - that no state colour is spelled
    anywhere but in that map - and it asserts it over the WHOLE enumeration,
    because a check on "Connected" alone would have been green through the
    entire life of the defect.
    """
    from src.arcade.palette import DANGER, SUCCESS, TEXT_TERTIARY, hex_str
    from src.pitwall.agents_view.panels import CONNECTION_COLOURS

    assert CONNECTION_COLOURS == {
        "Connected": hex_str(SUCCESS),
        # An absence, not a state.
        "Connecting...": hex_str(TEXT_TERTIARY),
        "Disconnected": hex_str(DANGER),
    }

    # The strip's own stylesheet may no longer name a connection state. The two
    # rules it used to carry are what disagreed.
    data_css = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "pitwall"
        / "ui"
        / "src"
        / "styles"
        / "data.css"
    ).read_text(encoding="utf-8")
    for stale in (".strip-chip.is-connected", ".strip-chip.is-lost"):
        assert stale not in data_css, f"{stale} is a second owner of a socket colour"

    # And every state the host can report has an entry, so a fourth word cannot
    # arrive and be coloured by a `.get(..., default)` nobody notices.
    host = PitwallHost(_ConnectableClient(_tick(1), connected=True), window_count=1)
    reported = {host.get_connection()["label"]}
    host._client.connected = False
    reported.add(host.get_connection()["label"])
    fresh = PitwallHost(_ConnectableClient(None, connected=False), window_count=1)
    reported.add(fresh.get_connection()["label"])
    assert reported == set(CONNECTION_COLOURS), f"the host reports {reported}"


# --- The connection label, which band 1 renders ------------------------------


class _ConnectableClient(_FakeClient):
    """A fake that can also be up or down, which the base one cannot."""

    def __init__(self, payload=None, connected: bool = True):
        super().__init__(payload)
        self.connected = connected


def test_the_data_window_alone_still_learns_that_the_producer_died():
    """The memory behind "Disconnected" belongs to the host, not to a window.

    Inferring it from the last label the AGENTS view was served would mean
    that, with only the DATA window open - the case band 1 exists for -
    a producer that had been up for an hour and then died reads
    "Connecting..." forever, which is a lie about which direction the
    session is going in.
    """
    client = _ConnectableClient(_tick(3), connected=True)
    host = PitwallHost(client, window_count=1)

    assert host.get_connection()["label"] == "Connected"

    client.connected = False

    assert host.get_connection()["label"] == "Disconnected", (
        "the socket has been up once, so this is a loss and not a first attempt"
    )


def test_before_the_socket_has_ever_been_up_the_word_is_connecting():
    """Retrying is not the same as having been dropped."""
    host = PitwallHost(_ConnectableClient(None, connected=False), window_count=1)

    assert host.get_connection()["label"] == "Connecting..."


def test_both_windows_read_the_same_label_from_the_same_memory():
    """One socket, one answer.

    The AGENTS header and the DATA strip are on screen together, so two
    labels disagreeing about whether the arcade is alive is the visible form
    of the twin defect this repo keeps paying for.
    """
    client = _ConnectableClient(_tick(1), connected=True)
    host = PitwallHost(client, window_count=2)
    host.get_agents_view(-1)

    client.connected = False

    assert host.get_connection()["label"] == "Disconnected"
    assert host.get_agents_view(1, "Connected")["header"]["connection"] == "Disconnected"


def test_a_second_agents_consumer_also_learns_the_producer_died():
    """The one consumer above cannot see #950, and that is why it existed.

    There are TWO consumers of this view in a shipped run: the AGENTS webview
    and `/agents.html` on the loopback server `__main__` starts unconditionally.
    Remembering the transition in ONE host field means whichever polled
    first consumes it and the second is answered `None` forever - a green
    "Connected" chip on a race that had stopped, measured over 50 polls.

    Both callers here hold "Connected" and neither has a newer tick, which is
    exactly the state a dead producer leaves. Both must be served.
    """
    client = _ConnectableClient(_tick(1), connected=True)
    host = PitwallHost(client, window_count=2)
    first = host.get_agents_view(-1)
    second = host.get_agents_view(-1)
    held = first["header"]["connection"]
    assert held == second["header"]["connection"] == "Connected"

    client.connected = False

    # The window notices...
    window = host.get_agents_view(first["seq"], held)
    # ...and so does the browser, which polled second and holds the same label.
    browser = host.get_agents_view(second["seq"], held)

    assert window is not None, "the first consumer was not told the producer died"
    assert browser is not None, "the SECOND consumer was not told - #950"
    assert window["header"]["connection"] == "Disconnected"
    assert browser["header"]["connection"] == "Disconnected"


def test_a_caller_that_already_rendered_the_state_is_told_nothing_changed():
    """The other half: `since_connection` must still SUPPRESS a repeat.

    Without this the view would be rebuilt on every poll of a dead race, which
    is the cost the single host field was buying and the reason it existed.
    """
    client = _ConnectableClient(_tick(1), connected=True)
    host = PitwallHost(client, window_count=2)
    view = host.get_agents_view(-1)

    client.connected = False
    host.get_agents_view(view["seq"], "Connected")

    assert host.get_agents_view(view["seq"], "Disconnected") is None
