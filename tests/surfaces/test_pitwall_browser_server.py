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
from unittest import mock

import pytest

from src.pitwall.config import WINDOWS, window_arguments, window_target
from src.pitwall.webserver import BROWSER_HOST, BrowserServer


class _FakeHost:
    """The narrow `TickSource` the server takes, and nothing else.

    Narrow on purpose: the browser path must not be able to reach
    `release_window` or `shutdown`, which belong to the OS windows.
    """

    def __init__(self) -> None:
        self.tick = {"seq": 7, "arcade": {"lap": 23}}
        # The real view's header ALWAYS carries the connection label - it is
        # what the caller hands back as `since_connection` - so a stub without
        # it could not exercise the route's second parameter at all.
        self.view = {"seq": 7, "header": {"session": "Melbourne · 2025", "connection": "Connected"}}

    def get_tick(self, since_seq: int = -1):
        return None if since_seq == self.tick["seq"] else self.tick

    def get_agents_view(self, since_seq: int = -1, since_connection: str | None = None):
        # Both halves of what the caller holds, as the real host takes them.
        # The route passes `connection` through, so a stub that ignored it
        # would let the query string rot without anything noticing (#950).
        if since_connection is not None and since_connection != self.view["header"]["connection"]:
            return self.view
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
    different frame on 58% of polls. Adding a transport that ignored `since`
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


def test_the_agents_route_carries_the_connection_the_caller_holds(served: str):
    """The browser is the SECOND consumer, so the query string is load-bearing.

    `/api/agents` takes two things the caller holds, not one: the sequence and
    the connection label it last rendered. The second is what lets a browser on
    `/agents.html` learn the producer died after the window beside it already
    has - with the state kept host-side instead, whichever polled first consumed
    the transition and the other kept a green chip on a dead race forever (#950).

    Dropping `&connection=` from the route would leave the parameter always
    `None`, which reads as "I hold nothing" and is silently generous rather than
    visibly broken. This is what fails then.
    """
    seen = json.loads(_get(served + "/api/agents?since=-1")[2])
    seq = seen["seq"]
    held = seen["header"]["connection"]

    # Same sequence, same label: nothing to say.
    assert json.loads(_get(served + f"/api/agents?since={seq}&connection={held}")[2]) is None

    # Same sequence, a label this caller has NOT rendered: it must be told.
    assert (
        json.loads(_get(served + f"/api/agents?since={seq}&connection=Disconnected")[2]) is not None
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


# --- Where the OS windows point (#995) ---------------------------------------


def test_a_window_points_at_the_server_that_is_running_and_not_at_a_file(served: str):
    """Every window's URL is a route THIS server answers, not a filesystem path.

    **The defect this closes was pywebview's, and it was invisible from here.**
    Given a file path, pywebview 6.2.1 serves the window through its own bottle
    server rooted at `os.path.commonpath` of the window URLs, and with two windows
    it gets that wrong racily: the same build produced a 404 for
    `…/6754/dist/data.html` on one run and for `…/52646/data.html` on another, each
    of them bottle's own page. Reproduced against the installed bottle - rooted at
    `ui/dist`, `data.html` is 200 and `dist/data.html` is 404 "File does not exist.";
    rooted at `ui` it is the other way round.

    So the assertion is not "the URL contains http". It is that the exact URL a
    window is handed is **fetchable from the server the host actually started**,
    which is the only property that makes the window and a browser tab the same
    surface. `window_target` is pure so this needs no system webview.
    """
    for spec in WINDOWS:
        target = window_target(spec, served)
        assert target.startswith(served), (
            f"{spec.key} does not point at the running server: {target}"
        )
        assert "file:" not in target and "\\" not in target, target
        # The route ANSWERS. A window handed an unfetchable URL is the whole bug,
        # and it is the only assertion here that could have caught it.
        status, content_type, body = _get(target)
        assert status == 200, f"{spec.key} -> {status} at {target}"
        assert content_type.startswith("text/html"), f"{spec.key} -> {content_type}"
        assert spec.key in body, f"{spec.key} served the wrong document: {body!r}"
        assert spec.entry in target


def test_a_window_falls_back_to_the_file_when_there_is_no_server():
    """No bundle to serve means no URL, and a window through pywebview's own
    server beats no window at all - which is what `__main__` logs a warning for."""
    for spec in WINDOWS:
        assert window_target(spec, None) == spec.url
        assert spec.url.endswith(spec.entry)


def test_the_window_arguments_carry_the_served_url_and_the_placed_geometry(served: str):
    """The SEAM `__main__` unpacks, so the wiring is guarded and not just the rule.

    **`window_target` being right was never the gap.** Reverting `__main__`'s call
    site to `spec.url` kept 227 surface tests, 176 data-smoke checks and 19
    agents-smoke checks green, because nothing outside the product loads that module
    - so the racy pywebview 404 (#995) could walk back in on a fully green board and
    only opening the OS window would show it. That is the same
    verified-through-the-page-and-not-the-window gap this sprint had to confess to
    once already.

    `main()` now unpacks this dict straight into `create_window`, so asserting its
    `url` key asserts what the window is handed. The geometry is checked in the same
    breath because it comes from the same call and `place`'s clamp is load-bearing.
    """
    screen = (1707, 960)
    for index, spec in enumerate(WINDOWS):
        arguments = window_arguments(spec, index, screen, served)

        assert arguments["url"] == window_target(spec, served)
        assert arguments["url"].startswith(served), arguments["url"]
        status, _content_type, body = _get(arguments["url"])
        assert status == 200 and spec.key in body, f"{spec.key} -> {status}"

        x, y, width, height = spec.place(index, *screen)
        assert (arguments["x"], arguments["y"]) == (x, y)
        assert (arguments["width"], arguments["height"]) == (width, height)
        assert arguments["title"] == spec.title
        # The toolkit's own keyword names, because this dict is splatted into it.
        assert set(arguments) == {"title", "url", "width", "height", "x", "y"}


def test_the_window_arguments_fall_back_to_the_file_with_no_server():
    for index, spec in enumerate(WINDOWS):
        assert window_arguments(spec, index, (1707, 960), None)["url"] == spec.url


def test_a_port_it_cannot_bind_degrades_to_no_server_rather_than_a_crash(tmp_path: Path):
    """The failure that actually happens, and the one the fallback was written for.

    `start()` returning None used to mean "no bundle" only - which `ui_is_built()`
    has already ruled out by the time `__main__` calls it, making the documented
    file-path fallback reachable only by deleting the bundle between two checks.
    A bind failure is the realistic one (a Windows port-exclusion range, a security
    product holding the port) and it propagated out and killed the host before a
    single window opened.

    **The OSError is injected, and the reason is worth stating.** A second server on
    an already-bound port was the obvious probe and it does not work: `HTTPServer`
    sets `allow_reuse_address`, so on Windows the second bind SUCCEEDS and returns a
    live URL - measured, before this test was rewritten. The failures that do happen
    are a reserved port range or a security product, neither of which a test can
    provoke portably, so what is asserted here is the DEGRADATION: whatever the
    socket raises, the caller gets None and the host lives.
    """
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "data.html").write_text("<html>data</html>", encoding="utf-8")
    (dist / "agents.html").write_text("<html>agents</html>", encoding="utf-8")

    def refuse(*_args, **_kwargs):
        raise OSError(10013, "An attempt was made to access a socket in a forbidden way")

    with mock.patch("src.pitwall.webserver.ThreadingHTTPServer", refuse):
        assert BrowserServer(dist, _FakeHost()).start() is None
