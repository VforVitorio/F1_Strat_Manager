"""Work the user waits on must not sit on the request they are waiting for.

Two fixes, one shape. The pit-wall host loaded the race inside `_session_for` on
the first `get_bulk`, a call the window blocks on while it paints nothing
(#1004), and `ensure_data` ran the whole multi-gigabyte download twice, the first
time silently under a "Resolving..." spinner (#168).

Measured here, on a payload shaped the way `app.py:658` publishes it:

    old start(), connect only    first get_bulk   300.1 ms
    new start(), pre-warmed      first get_bulk     0.3 ms

The pre-warm reads the race from `payload["arcade"]`, which is where `get_bulk`
and `get_live_lap` read it from. The first version of this read the top level
instead: against the real producer it would have found nothing, polled for the
full 30 s timeout and warmed nothing at all, while a test with a flattened
fixture stayed green. `test_a_flattened_payload_is_not_the_real_shape` is that
mistake, kept as an assertion.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

from src.pitwall.host import PitwallHost

ROOT = Path(__file__).resolve().parents[2]
CACHE_SOURCE = ROOT / "src" / "f1_strat_manager" / "data_cache.py"

RACE = {
    "year": 2025,
    "location": "Melbourne",
    "lap": 20,
    "total_laps": 57,
    "frame_index": 0,
    "global_t_min": 0.0,
    "drivers": {"NOR": {"laps_completed": 20}},
    "driver_colors": {},
}
TICK = {"seq": 1, "arcade": RACE, "playback": {"speed": 1, "paused": False}}


class FakeClient:
    """Publishes one tick as soon as it is started, like a producer already up."""

    def __init__(self, payload: dict | None = TICK) -> None:
        self.started = False
        self._payload = payload

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    @property
    def latest(self) -> dict | None:
        return self._payload if self.started else None

    @property
    def connected(self) -> bool:
        return self.started

    def snapshot(self):
        return (self._payload if self.started else None), ()


@pytest.fixture
def loaded(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Record every race the host loads, without touching the disk."""
    calls: list[tuple] = []

    class Stub:
        @staticmethod
        def load(root, year, location):
            calls.append((year, location))
            return None

    import src.pitwall.host as host_module

    monkeypatch.setattr(host_module.SessionLaps, "load", Stub.load)
    monkeypatch.setattr(host_module.RadioCorpus, "load", Stub.load)
    return calls


def test_start_warms_the_race_off_the_request_path(loaded: list[tuple]) -> None:
    """The whole point: the load happens before any window asks."""
    host = PitwallHost(client=FakeClient(), window_count=2)
    host.start()
    host._warm_thread.join(10)

    assert loaded == [(2025, "Melbourne"), (2025, "Melbourne")], (
        "the pre-warm did not load the laps and the radio corpus"
    )


def test_the_first_get_bulk_finds_the_cache_warm(loaded: list[tuple]) -> None:
    """After the warm-up, the request does no loading of its own.

    Asserted by counting loads rather than by timing, because a wall-clock
    threshold on a shared runner is a flake waiting to happen. The count is the
    property; the milliseconds are in the module docstring.
    """
    host = PitwallHost(client=FakeClient(), window_count=2)
    host.start()
    host._warm_thread.join(10)
    before = len(loaded)

    host.get_bulk(-1)
    assert len(loaded) == before, "get_bulk loaded the race again"


def test_a_flattened_payload_is_not_the_real_shape(loaded: list[tuple]) -> None:
    """The bug the first version shipped, stated as its own test.

    `app.py:658` nests the race under `arcade`. A pre-warm reading `year` and
    `location` off the top level finds them only in a hand-written fixture, so it
    would warm nothing in production and every test would still pass. Here the
    flattened payload must warm NOTHING, which is what proves the reader is
    looking in the right place rather than in both.
    """
    host = PitwallHost(client=FakeClient(payload={"seq": 1, **RACE}), window_count=2)
    host.start()
    host._warm_thread.join(2)
    assert loaded == [], "the pre-warm read the race off the top level"


def test_the_warm_thread_is_a_daemon(loaded: list[tuple]) -> None:
    """It waits up to 30 s for a race, so it must never hold the process open."""
    host = PitwallHost(client=FakeClient(payload=None), window_count=2)
    host.start()
    assert host._warm_thread.daemon


def test_the_warm_up_gives_up_rather_than_spinning(loaded: list[tuple]) -> None:
    """A producer that never names a race is a real state, not an error.

    The arcade may still be building telemetry, or may never have been started,
    and both windows already have a waiting screen for it.
    """
    host = PitwallHost(client=FakeClient(payload=None), window_count=2)
    started = time.perf_counter()
    host._warm_session(timeout_s=0.2, poll_s=0.02)
    assert 0.15 < time.perf_counter() - started < 3.0
    assert loaded == []


def _ensure_setup_ast() -> ast.FunctionDef:
    tree = ast.parse(CACHE_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "ensure_setup":
            return node
    raise AssertionError("ensure_setup is gone from data_cache.py")


def test_the_first_run_downloads_once() -> None:
    """`ensure_setup` calls `_snapshot_download` exactly once (#168).

    It used to call it twice: once with progress off inside the spinner, which
    downloaded everything silently, and once with progress on, which then found
    every file present and returned at once. A first-time user watched a spinner
    for the whole 7-8 GB and then a progress bar that completed instantly.

    Counted structurally rather than by running a download, for the obvious
    reason. The comment on the old first call claimed it "warms the HTTP client
    and validates credentials", which is a real thing to want and not what a full
    download is; `_resolve_repo` does it in one metadata call, measured at 965 ms
    against the real Hub with no bytes fetched.
    """
    calls = [
        node.lineno
        for node in ast.walk(_ensure_setup_ast())
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_snapshot_download"
    ]
    assert len(calls) == 1, f"ensure_setup downloads {len(calls)} times, at lines {calls}"


def test_the_spinner_waits_on_the_probe_not_on_the_download() -> None:
    """The "Resolving..." spinner has to cover the resolve, not the transfer.

    If the download moves back inside the `console.status` block the progress
    bars are hidden again, which is the visible half of the defect: the bytes
    are the part the user wants to watch.
    """
    inside = [
        getattr(call.func, "id", None)
        for node in ast.walk(_ensure_setup_ast())
        if isinstance(node, ast.With)
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    ]
    assert "_resolve_repo" in inside, "the spinner does not wrap the reachability probe"
    assert "_snapshot_download" not in inside, (
        "the download is back inside the spinner, so its progress bars are hidden"
    )


DOCS_CSS = ROOT / "docs" / "styles" / "docs.css"
ROADMAP = ROOT / "docs" / "pages" / "roadmap.md"

# `--fg-4` is rgba(255,255,255,0.32), which measures 2.78-2.91 against the
# grounds the docs site paints on, against AA's 4.5 for normal text. These two
# are the only uses that are not text a reader is meant to read.
DECORATIVE = {".search-icon", ".breadcrumb-sep"}


def test_only_decoration_uses_the_sub_aa_token() -> None:
    """Ten small labels used a token the site's own comment calls "disabled".

    Measured in the browser at 1440x900, compositing each element's own
    translucent backgrounds down to an opaque ground: `.sidebar-section-title`
    2.78, `.toc-title` 2.78, `.breadcrumb` 2.78, `.search-kbd` 2.87,
    `.page-footer-label` 2.90, `.docs-footer-col-title` 2.83,
    `.docs-footer-legal` 2.83, `.rl-date` 2.91. On `--fg-3` the same eight
    measure 5.53 to 5.70.

    The token stays where it is rather than being raised, because raising it to
    clear 4.5 needs alpha 0.48 and `--fg-3` is 0.52: the four-step ramp would
    collapse into three to keep a role its own comment says is decorative.
    """
    offenders = []
    for path in (DOCS_CSS, ROADMAP):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if "var(--fg-4)" not in line:
                continue
            selector = "?"
            for j in range(i, max(-1, i - 14), -1):
                if "{" in lines[j]:
                    selector = lines[j].split("{")[0].strip()
                    break
            if selector not in DECORATIVE:
                offenders.append(f"{path.name}:{i + 1} {selector}")
    assert not offenders, f"read text back on the sub-AA token: {offenders}"


BESTS_PANEL = ROOT / "src" / "pitwall" / "ui" / "src" / "features" / "data" / "BestsPanel.tsx"


def test_the_fit_callback_does_not_close_over_the_state_it_sets() -> None:
    """#1083, and the regression the first attempt at it shipped.

    The observer was rebuilt on every decision because its effect depended on
    `fit`, a `useCallback` over the fit state, so a resize landing in the gap was
    unobserved. The first fix put `fit` behind a ref and dropped it from the deps.
    That made it worse: `fitRef.current = fit` runs in a PASSIVE effect, and under
    load a ResizeObserver callback is delivered before that effect runs, so the
    observer called the previous render's closure. Its stale `if (fitState.ranked)`
    branch re-latched the floor from the compact card at 60 px and `room 63 >= 60`
    flipped the panel back to ranked in a slot whose real floor card is 115 px.
    Measured: the signature fired on 24 of 24 throttled loads, and `smoke-data`
    failed 7 of 9 unthrottled runs where the pre-fix bundle passed 3 of 3.

    Both defects have one cause, which is why one assertion covers both: `fit`
    read the state it sets. It now reads the rendered rows from the DOM instead,
    so it is identity-stable per card node, and the observer effect can depend on
    it directly without ever being torn down by a decision.

    Still a text assertion, and the reason is unchanged: there is no TSX parser
    here. What changed is WHAT it pins. The previous version pinned
    `fitRef.current()` verbatim, an implementation, and would have blocked this
    repair; these pin the property, that no fit state appears in the callback's
    dependencies and that no ref stands between the observer and the callback.
    """
    source = BESTS_PANEL.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith(("//", "*"))
    )
    assert "fitState" not in code.split("}, [card]);")[0].split("const fit = useCallback")[-1], (
        "the fit callback reads fitState again, so it closes over its own output and a "
        "stale copy can re-latch the floor from the compact card"
    )
    assert "fitRef" not in code, (
        "a ref is back between the observer and the callback; its updater runs in a "
        "passive effect, after ResizeObserver delivery under load"
    )
    assert "new ResizeObserver(fit)" in code, "the observer no longer calls fit directly"
    assert "}, [card, fit, content]);" in code, (
        "the observer effect's dependency array changed; `fit` belongs there and is safe "
        "there only while it stays identity-stable"
    )
