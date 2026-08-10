"""The same two pages, over http on loopback (`src/pitwall/webserver.py`).

Additive to the pywebview windows, which remain the product. The 2026-08-07
decision that PITWALL is a desktop surface killed the FastAPI relay because a
browser cannot open a raw TCP socket - but the host already holds that socket,
so handing the same payload to a browser costs a small server rather than a
backend.

What these pin is the part that is easy to get wrong in a hurry: the payload a
browser gets is the SAME sequenced payload a window gets, a request can never
name a file, and nothing here listens beyond the loopback interface.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from src.pitwall.webserver import BROWSER_HOST, BrowserServer


class _FakeHost:
    """The narrow `TickSource` the server takes, and nothing else.

    Narrow on purpose: the browser path must not be able to reach
    `release_window` or `shutdown`, which belong to the OS windows.
    """

    def __init__(self) -> None:
        self.tick = {"seq": 7, "arcade": {"lap": 23}}
        self.view = {"seq": 7, "header": {"session": "Melbourne · 2025"}}

    def get_tick(self, since_seq: int = -1):
        return None if since_seq == self.tick["seq"] else self.tick

    def get_agents_view(self, since_seq: int = -1):
        return None if since_seq == self.view["seq"] else self.view


@pytest.fixture
def served(tmp_path: Path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "data.html").write_text("<html>data</html>", encoding="utf-8")
    (dist / "agents.html").write_text("<html>agents</html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    server = BrowserServer(dist, _FakeHost())
    url = server.start()
    assert url, "the server must come up on a real bundle"
    yield url.rstrip("/")
    server.stop()


def _get(url: str) -> tuple[int, str, str]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.headers.get("Content-Type", ""), response.read().decode()


def test_the_bundle_is_served_including_the_root(served: str):
    for route, expected in (("/data.html", "data"), ("/agents.html", "agents"), ("/", "data")):
        status, content_type, body = _get(served + route)
        assert status == 200 and expected in body, f"{route} served {body!r}"
        assert content_type == "text/html", f"{route} served as {content_type}"
    assert _get(served + "/assets/app.js")[1] == "text/javascript"


def test_a_browser_gets_the_same_sequenced_payload_a_window_gets(served: str):
    """The sequencing is the whole reason two consumers do not disagree.

    Gate A measured two pollers against a blind latest-payload slot reading a
    different frame on 58 % of polls. Adding a transport that ignored `since`
    would put that back for the browser.
    """
    status, content_type, body = _get(served + "/api/tick?since=-1")
    tick = json.loads(body)
    assert status == 200 and content_type == "application/json"
    assert tick["seq"] == 7 and tick["arcade"]["lap"] == 23

    assert json.loads(_get(served + "/api/tick?since=7")[2]) is None, "a seen tick must be null"
    assert json.loads(_get(served + "/api/agents?since=-1")[2])["header"]["session"].startswith(
        "Melbourne"
    )


def test_a_junk_since_is_treated_as_having_seen_nothing(served: str):
    """A URL typed by hand is not a reason to 500 - it is a reason to answer."""
    assert json.loads(_get(served + "/api/tick?since=nonsense")[2])["seq"] == 7
    assert json.loads(_get(served + "/api/tick")[2])["seq"] == 7


def test_a_request_can_never_name_a_file(served: str):
    """The bundle is read into memory at startup and served from a dict.

    `serve-dist.mjs` learned this twice - a URL can carry `../`, which CodeQL
    called a path injection, and the `stat`-per-entry version that replaced it
    was a filesystem race. Reading the tree once removes the question rather
    than guarding it, so these are 404s and not near-misses.
    """
    for attempt in ("/../pyproject.toml", "/..%2fpyproject.toml", "/assets/../../secret"):
        with pytest.raises(urllib.error.HTTPError) as caught:
            _get(served + attempt)
        assert caught.value.code == 404, f"{attempt} returned {caught.value.code}"


def test_it_listens_on_loopback_only(served: str):
    """The broadcast carries a whole race's live state and there is no auth.

    Asserted on the URL the server reports, which is the address it bound.
    """
    assert served.startswith(f"http://{BROWSER_HOST}:"), served
    assert BROWSER_HOST == "127.0.0.1"


def test_no_bundle_means_no_server_rather_than_a_crash(tmp_path: Path):
    """`f1-arcade` spawns PITWALL; a missing build must not take the arcade
    down with it. `__main__` already refuses to open windows in that case."""
    assert BrowserServer(tmp_path / "nothing", _FakeHost()).start() is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert BrowserServer(empty, _FakeHost()).start() is None
