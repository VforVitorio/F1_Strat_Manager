"""The PITWALL host and its stream client (`src/pitwall/`).

Sprint 2 is the vertical slice, and the delivery plan named three things it
had to get right because they are expensive to change later. Two of them are
here; the third (design-token drift) is `test_pitwall_tokens.py`.

Nothing in this file imports pywebview. Only `src/pitwall/__main__.py` does,
so a machine with no system webview still runs the suite.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from src.pitwall.host import PitwallHost
from src.pitwall.stream_client import ArcadeStreamClient


class _FakeClient:
    """Stands in for the socket, so the host's own logic is what is tested."""

    def __init__(self, payload=None):
        self.latest = payload
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def _tick(seq: int) -> dict:
    return {"seq": seq, "schema_version": 1, "arcade": {"lap": 12}}


# --- Trap 1: the tick is sequenced, never a blind slot -----------------------


def test_a_window_only_gets_a_tick_it_has_not_seen():
    """The whole reason `since_seq` exists.

    Two windows polling one latest-payload slot on independent 10 Hz timers
    were measured reading a different frame on 58 % of polls, with 15
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
