# ADVERSARIAL GATE — the window-transport fix (#995) and the ten exit-gate fixes, one commit

**Date:** 2026-08-18 · **Auditor:** adversarial verification gate (Fable 5)
**Subject:** commit `af92229` on `fix/pitwall-exit-gate-sprint9`
(`fix(pitwall): the windows load over one transport, and what the exit gate found`).
**Bundle:** `npm run build` WAS run by this gate (permitted by its contract) so that the measured
bundle is provably this commit's source — the pre-existing `dist/` was built at 22:26, six minutes
before the commit, which is close but not proof. Every measurement below is against the rebuilt
bundle unless it says otherwise.
**Live rig:** `scripts/dev_pitwall_producer.py` (Melbourne 2025) + `python -m src.pitwall`
(the REAL OS windows) + the same host's loopback server for Playwright probes.
**Contract:** no repository file modified except this report; probes named `_wingate_*`;
inventory at the end.

*(Report is written incrementally; sections fill in as they are executed.)*

---

## Checklist — W (the window fix) and X (the exit-gate fixes)

| # | Claim under attack | Verdict |
|---|---|---|
| W1 | `window_target` correct for both entries, base with/without trailing slash, non-bare entry | **VERIFIED** — executed on all six cases |
| W2 | pywebview starts NO internal bottle server when handed `http://` URLs — proven by observation | **VERIFIED** — 1 listener (the BrowserServer), `webview.http.global_server` is None, live |
| W3 | `window.pywebview` injected on an `http://` load, real js_api calls work, BOTH windows | **VERIFIED** — ~2.5 min of real calls from inside both real windows |
| W4 | Nothing that worked off a file URL broke: storage origin, assets, teardown, one window closing | **VERIFIED** — and the app stores nothing, so the origin change orphans nothing |
| W5 | The file fallback still behaves — and whether it is reachable at all after `ui_is_built()` | **MIXED** — the pure rule works; the live fallback branch is unreachable in practice, and the one realistic server failure CRASHES instead of falling back (finding G3) |
| W6 | Ordering/failure modes: port taken, `start()` raises, thread dies, `browser.stop()` on exits | **VERIFIED with notes** — port 0 cannot be taken; all threads daemon; a mid-run server death does NOT starve the OS windows (js_api transport); one uncovered exit path, theoretical (G3) |
| W7 | Nothing else in the repo opens these windows or assumes a file URL | **VERIFIED** — one spawner (`src/arcade/app.py:484`, `python -m src.pitwall`); no other file-URL consumer; docs clean |
| W8 | The two new tests are real: would they pass on a broken fix? | **MIXED** — the tests kill every broken `window_target`; the WIRING in `__main__` is guarded by nothing (finding G1) |
| X1 | BESTS re-decides on content + card resize; no oscillation; intermediate sizes | **VERIFIED** — 66 fresh mounts over 11 sizes, 0 clipped; 1 transition per direction; stable holds. And the guard IS red on the pre-fix bundle — its own comment says otherwise (finding G2) |
| X2 | `brightness(0.72)` preserves hue relationships on rendered pixels | **VERIFIED** — channel ratios uniform ≈0.72 on sampled pixels; matrix says −28 % on both pairs exactly, vs the old filter's −62 % |
| X3 | AGENTS frozen treatment: real killed producer, no latch, no pre-first-view fire | **VERIFIED** — real windows + real kill; unlatch both windows; `get_agents_view` returns None before any payload |
| X4 | "two laps, 33 at 13/16 and 46 at 3/15" — recomputed | **VERIFIED** — exact |
| X5 | `formatSeconds` em-dash guard changes no call site's rendering for wire values | **VERIFIED** — 3 call sites, all null-guarded, wire non-negative; `-0` takes the old path |
| X6 | "17.48:1" — recomputed | **VERIFIED** — 17.48 on `#181633`; the sibling claim (15.99 on `--qt-elevated #1e1b4b`) also reproduces |
| X7 | Ruler measures a `tbody td`; no oscillation via the body cell | **VERIFIED** — `table-layout: fixed` + one 9 px font make the cell form-independent; tenths at 1485x833, 0 clipped |
| X8 | `widestFine`: tenths kept at 1485x833, coarsens for a 10-minute lap, independent of `coarse` | **VERIFIED** — injected 601.2 s → wide client coarsened, stable over 10 samples; all branches independent of `coarse`, deleted laps counted, empty-bulk falls back |
| X9 | Frozen starved traces caption; `single-driver mode` placeholder precedence | **MIXED** — fresh-on-dead-feed captions correctly; but a frozen single-driver window's delta caption asserts a false cause (finding G4) |
| X10 | Compound sentinel in `_lap_row`; nothing downstream relied on the raw string | **VERIFIED** — zero pixel change on Melbourne (census over every driver's last row); `is_real_compound(None)` is False so the stop count is unchanged |

---

## Evidence log (running)

- **Rebuild**: `npm run build` produced byte-identical asset hashes (`data-WAG6unQu.js`,
  `agents-mo2wvm7E.js`, `qt-base-BYpsrYS6.js`) to the pre-existing `dist/` — the committed source
  IS what was already built. All measurements are against this bundle.
- **Suites**: `tests/surfaces/` **227 passed** · `smoke-data.mjs` **176 checks OK** ·
  `smoke-agents.mjs` **19 checks OK** — all three match the commit's claims.
- **W1 (pure)**: `window_target` returns `http://127.0.0.1:56787/<entry>` for BOTH entries under a
  base with AND without the trailing slash; a nested entry (`sub/page.html`) joins correctly; a base
  with a path (`http://h:1/app/`) joins correctly; `None` AND `""` both fall back to the file path.
  Executed, `_wingate_census.py`. **VERIFIED.**
- **W2 (observed, not read)**: ran the exact `__main__` wiring (real host, real `BrowserServer`,
  real `window_target`, real `create_window` x2) in `_wingate_windows.py`. With both OS windows
  OPEN and rendering: the process owned exactly **ONE** listening socket — `127.0.0.1:53647`, the
  BrowserServer — and `webview.http.global_server` was `None`. The 26 WebView2 child processes
  owned no listeners. **pywebview starts NO internal bottle server when handed `http://` URLs: the
  fix removed the CAUSE, not the symptom. VERIFIED.**
- **W3 (from inside the real windows)**: both windows answered `evaluate_js` polls over ~2.5 min:
  `location.href` on the loopback (`/data.html`, `/agents.html`), `typeof window.pywebview` =
  `"object"`, and REAL js_api calls through the bridge — `get_tick(-1)` (seq advancing 6562→7238),
  `get_bulk(-1)` (rev advancing 3→20, `available: true`), `get_connection()` (`"Connected"`),
  `get_agents_view(-1, null)` (header carried). DATA rendered 6 cards, AGENTS 14. A desktop
  screenshot confirms the AGENTS window fully rendered (PIT NOW, scenario scores, charts, reasoning).
  **VERIFIED.**
- **W4 (partial, live)**: `localStorage` set/get works over the http origin in both windows (the app
  itself uses NO localStorage/sessionStorage/file:// anywhere — grep over `ui/src` is empty, so the
  origin change orphans nothing). Built HTML uses relative `./assets/…` paths — served. Destroying
  the DATA window mid-run left the AGENTS window answering polls; destroying both returned
  `webview.start()`, ran `browser.stop()` + `host.shutdown()` (probe printed CLEAN EXIT) and the
  loopback port was released (0 listeners after exit). **VERIFIED** on those axes.
- **X3 (REAL windows, REAL kill)**: killed the producer process that owned `127.0.0.1:9998` while
  both OS windows were open. DATA: `frozenClass true`, `DATA FROZEN` chip, status
  `DATA FROZEN · last tick lap 37`. AGENTS: `frozenClass true`, chips strip read
  `Disconnected | DATA FROZEN | — | L 37/57` — the playback chip is the dash, not `2.00x · PLAYING` —
  status bar `DATA FROZEN · last tick L 37/57`, non-transient (stable across 5 polls, 20 s).
  Pre-first-view: `get_agents_view` returns `None` until the client holds a payload
  (`host.py:174-176`), and `frozen` requires `view !== null` — cannot fire before the first view.
  Unlatch test pending (below).
- **X4**: recomputed through the decode path (`neutralised_label(_none_if_nan(TrackStatus))` over
  non-generated rows, exactly what `_lap_row` publishes): mixed laps = `[(33, 16 rows, 13 marked),
  (46, 15 rows, 3 marked)]`. The new sentence is exact. **VERIFIED.**
- **X6**: white on `--qt-panel #181633` = **17.48:1** (recomputed, WCAG formula). **VERIFIED.**
- **X10**: Melbourne 2025 after `repair_tyre_stints` holds NO sentinel compound anywhere — real rows
  {HARD, INTERMEDIATE, MEDIUM}, all 6 generated rows {INTERMEDIATE, MEDIUM}. For every driver whose
  LAST row is generated (ALO, BOR, DOO, HAD, LAW, SAI) the old and new `tyreCell` render
  byte-identically (`I 33`, `I 2`, `I 1`, `M 14`…) — zero pixel change on the curated race, exactly
  as the comment claims. `_tyre_stops` semantics unchanged: `is_real_compound(None)` is `False`
  (`tyre_stint_repair.py:124`), so a nulled compound still cannot invent a stop. **VERIFIED.**
- **X5 (static)**: the only call sites are `BestsPanel.tsx:254`, `TimingTower.tsx:254`,
  `racePace.ts:180` — all three guard `null` first and feed non-negative seconds off the wire
  (`allow_nan=False` upstream). `0` and `-0` take the unchanged path (`-0 < 0` is false in JS).
  No call site's rendering changes for any value the wire can carry. **VERIFIED** (static; the 176
  smoke checks exercise all three surfaces on served values).

## Evidence log — the live rig and the Playwright phase

- **The real windows, the real transport** (`_wingate_windows.py`, the exact `__main__` wiring):
  both OS windows opened on `http://127.0.0.1:53647/{data,agents}.html`, rendered fully (desktop
  screenshot read and checked: PIT NOW card, scenario scores, both charts, reasoning tabs), and
  answered 38 polls of `evaluate_js` including real bridge calls. Seq advanced 6562 → 7238 across
  the session; bulk rev 3 → 20.
- **X3 on the REAL windows against a REAL kill**: killed the producer process owning port 9998.
  Next poll: DATA `frozenClass true` + `DATA FROZEN` chip + `DATA FROZEN · last tick lap 37`;
  AGENTS chips `Disconnected | DATA FROZEN | — | L 37/57` (the playback chip IS the dash), status
  `DATA FROZEN · last tick L 37/57`, stable and non-transient across 5 further polls.
- **X3 unlatch, both windows** (loopback pages, freeze simulated with the consistent pair a dead
  producer serves — tick→null + connection→"Disconnected" — then removed): DATA reverted to
  `lap 28 · live`; AGENTS reverted to `Connected | 2.00× · PLAYING | L 28/57`, frozen chip gone.
  Cannot latch.
- **X1 fresh mounts** (`_wingate_a.mjs`, live loopback, real payload, 6 mounts/size): 1485x833
  ranked, 1265x593 leaders, and **1265x650 / 1350x660 / 1350x673 / 1265x620 / 1300x640 / 1330x655 /
  1350x646 leaders, 1400x700 / 1450x780 ranked — all 66 mounts 0 px hidden, THEORETICAL visible,
  single form per size**. The author tried five sizes; six more inside and around the band behave.
- **X1 oscillation**: a 833→593→833 sweep in 8 px steps produced exactly one transition each way at
  the same boundary (681↔673 at 1350 wide); holds at 685/681/677 px were a single value over 10
  samples each. The compact→ranked re-latch path converges (the latch is only refreshed while
  ranked, and a wrong optimistic flip is corrected on the very next `fit`).
- **X1 guard discrimination — executed, twice**: built the PRE-FIX `BestsPanel.tsx` (from `239babd`)
  into an isolated bundle (junctioned node_modules, zero repo modification) and ran the TRACKED
  `smoke-data.mjs` against it: **`smoke-data FAILED (2)` — `bests fits at 1265x650 … (ranked, 18 px
  hidden, THEORETICAL 3 px over)` and `1350x660 … (ranked, 8 px hidden)` — identically on both
  runs.** Against the committed bundle: 176 OK. The guard is genuinely red on the un-fixed latch
  and green on the fix — see finding G2 for what the guard's own comment claims instead.
- **X2 pixels** (`_wingate_c.mjs` + `_wingate_pixels.py`): same five cell coordinates sampled live
  and frozen at 1485x833. Channel ratios per patch cluster at ≈0.72 (pit cell: 0.718/0.716/0.721);
  channel ORDER (the hue) preserved on every patch. Matrix arithmetic on the same live RGBs:
  the new `brightness(0.72)` costs **28 % of the pair distance on both pairs exactly** (t1/t3
  112.4→81.0, best/deleted 60.6→43.7) where the old `saturate(0.45) brightness(0.82)` cost 62 %.
  Screenshot distances (65 %/60 % kept) differ from the matrix only by glyph anti-aliasing.
- **X7/X8** (`_wingate_b.mjs`): real data at 1485x833 → fine (`1:29.3`, cell 38 px, 0 clipped);
  1265x593 → coarse, 0 clipped. `/api/bulk` intercepted to plant one 601.2 s lap → the wide client
  **coarsened** (`1:29`, 0 clipped) and held that answer over 10 samples. `table-layout: fixed;
  width: 100%` makes cell width form-independent and one `font-size: 9px` covers both forms, so
  measuring a body cell cannot self-oscillate.
- **X9** (`_wingate_c.mjs`): a page opened fresh onto a dead-after-one-tick feed shows the tower
  fully populated (20 rows) and captions the starved plot ("no telemetry since the feed stopped");
  the three telemetry charts had ≥2 samples from the single tick's span and drew traces, so nothing
  renders as silent bare axes. Precedence: see finding G4.

---

## Findings

### G1 · P2 — the wiring that fixes #995 is guarded by nothing: reverting `__main__.py` to `spec.url` keeps every suite green

**Where:** `src/pitwall/__main__.py:100` (`window_target(spec, url)` inside `create_window`) ·
`tests/surfaces/test_pitwall_browser_server.py:154-199` (the two new tests).

**What breaks:** the two new tests pin the RULE (`window_target` as a pure function) and pin it
well — a broken join, a file path, a wrong route all fail. But nothing anywhere asserts that
`__main__` **calls** it. I reverted the call in a scratch copy of the module logic and ran the
whole battery mentally and then actually: `pytest tests/surfaces/` (227), `smoke-data` (176),
`smoke-agents` (19) touch `__main__.py` in **zero** places — the smokes serve `dist/` through their
own server and the tests exercise `BrowserServer` + `window_target` directly. A future refactor of
`main()` that hands `spec.url` back to `create_window` reintroduces the racy 404 with a fully green
board — and this exact class ("verified through the loopback PAGE, not the OS window") is what the
commit's own message confesses this sprint had been doing.

**Executed evidence:** `grep -rn "window_target" src tests` → the only non-test caller is
`__main__.py`; the only executable that loads `__main__.py` at all is the product itself.
My live probe verified today's wiring by observation (`TARGET data: 'http://127.0.0.1:53647/data.html'`),
which is exactly the kind of check that does not survive into CI.

**Prescription:** extract the per-window argument assembly into a pure function beside
`window_target` (e.g. `window_arguments(spec, index, screen_size, base) -> dict`) that `main()`
unpacks into `create_window`, and let a test assert its `url` key is `window_target(spec, base)`
for both windows. One seam, no pywebview import needed.

### G2 · P3 — the X1 guard's own comment claims the guard cannot catch the defect it demonstrably catches; the comment and the commit message contradict each other

**Where:** `src/pitwall/ui/scripts/smoke-data.mjs:1911-1918` — "*This guard … is NOT the
discriminator for the race that produced the defect. Driven against the un-fixed latch it still
passes: … the stub transport has the panel already compact by the time the first measurement lands,
so the mount never commits to `ranked` against the empty height*".

**Refuted by execution, twice.** Against the pre-fix `BestsPanel` (from `239babd`, rebuilt in
isolation), the committed smoke **fails**: `ranked, 18 px hidden` at 1265x650 and `ranked, 8 px
hidden` at 1350x660 — the panel very much DOES commit to `ranked` against the empty height under
the stub transport, and the guard catches it. Identical result on a second run. The commit message
says the opposite of the comment and agrees with my measurement ("The X1 guard is red against the
un-fixed latch and green with it, four consecutive runs each way").

**Why it matters:** a maintainer who reintroduces the latch and sees this guard go red will read
the comment, conclude the guard is flaky-by-design ("it is NOT the discriminator… it still
passes"), and be invited to ignore or delete the one assertion that protects the sprint's P1. A
false claim in a comment is this repo's most-documented bug class; this one disarms a working gun.
(Probable cause: the paragraph describes an earlier revision of the block — it narrates the
`page.route` version's failure and was not re-measured after the `__holdBulk` stub-withholding
was added. That is a hypothesis; the refutation is not.)

**Prescription:** replace the paragraph with the measured truth: the guard is red against the
un-fixed latch at 1265x650 and 1350x660 (and may pass at 1350x673, where the race is
probabilistic), and keep the honest clause that the LIVE six-mounts-per-size run is the stronger
evidence.

### G3 · P3 — the file-path fallback is unreachable on every realistic path, and the one realistic server failure crashes PITWALL instead of falling back

**Where:** `src/pitwall/__main__.py:36-38` (`ui_is_built()` gate), `:49-56`
(`url = browser.start()` with no try), `src/pitwall/webserver.py:214-231` (`start`).

**What the code actually does:** `start()` returns `None` only when `dist/` is not a directory or
holds no `data.html` — but `ui_is_built()` has already verified both files exist, three lines
earlier. So the `else: logger.warning(… fall back to file paths …)` branch and the fallback arm of
`window_target` are reachable only through a delete-between-two-checks race. Meanwhile the failure
that CAN happen — `ThreadingHTTPServer` raising on bind (Windows port-exclusion ranges, a
security product), or `_read_bundle` raising on an unreadable file — propagates out of `start()`
uncaught and kills `main()` before any window opens, with `browser.stop()`/`host.shutdown()` never
reached (all threads are daemons, so the process still exits; nothing hangs). The commit message's
"The file path stays as the fallback for when the bundle cannot be served at all" is therefore
true of the pure function and not of the program: when the bundle cannot be served, PITWALL
crashes; the fallback fires only when the bundle vanished mid-startup.

**Executed evidence:** `test_no_bundle_means_no_server_rather_than_a_crash` passes (the None
contract holds at unit level); code-path analysis above; `ui_is_built()` and `start()` check the
same two files.

**Prescription:** wrap `browser.start()` in `try/except OSError` returning `None`, which makes the
documented fallback actually cover the documented case. Three lines, and the warning branch stops
being dead code.

### G4 · P3 — a frozen single-driver window's delta caption asserts a false cause

**Where:** `src/pitwall/ui/src/features/data/OwnCarTraces.tsx:153`
(`placeholder={starved(frame.delta) ?? (rivalCode ? null : "single-driver mode")}`).

**What a strategist sees:** in true single-driver mode (`driver_rival` null, no rival telemetry),
`deltaSeries` is `[]` by construction — the delta plot is empty because there is NO RIVAL, and the
caption says so. When the feed then dies, `starved(frame.delta)` takes precedence and the caption
flips to **"no telemetry since the feed stopped"** while the three charts beside it display full
traces of exactly the telemetry the sentence denies. The caption's cause is false; the pre-freeze
caption was the true one.

**Executed evidence** (`_wingate_d.mjs`, rival code AND rival telemetry stripped from the tick,
9 s live, then the consistent dead-producer pair):
`LIVE: captions=["single-driver mode"], charts=3` →
`FROZEN: frozenClass=true, captions=["no telemetry since the feed stopped"]`.

**Prescription:** give the session property precedence on that one chart:
`rivalCode ? starved(frame.delta) : "single-driver mode"`. The starved caption stays right for the
fresh-open-on-dead-feed case (rival present, buffers empty), which my X9 run confirmed separately.

### G5 · P3 — `DataWindow.tsx` still narrates the retired treatment: "Desaturated and mildly dimmed"

**Where:** `src/pitwall/ui/src/features/data/DataWindow.tsx:92` — the comment directly above
`className={frozen ? "data-main is-frozen" : "data-main"}`.

This commit removed `saturate()` from `.data-main.is-frozen` (X2) and rewrote the CSS comment
("DIMMED, never desaturated…") and the AGENTS twin ("Dimmed, not desaturated…") — and left the
DATA window's own inline comment claiming the class desaturates. The twin that never got the fix,
in the fix for a twin that never got a fix. Comment-only, no pixel wrong.

**Prescription:** s/Desaturated and mildly dimmed/Dimmed, never desaturated/ — and the rest of the
sentence still holds.

---

# Fix list, ordered by value over risk

1. **G1 — put the window-URL wiring behind a test** (extract `window_arguments`, assert its `url`
   for both windows equals `window_target(spec, base)`). This is the user's "it must not fail
   again" translated into CI; everything else about #995 is already sound.
2. **G2 — rewrite the smoke-guard comment to the measured truth** (the guard IS red against the
   un-fixed latch at two of its three sizes). Zero risk; it re-arms trust in a working guard.
3. **G4 — flip the delta chart's placeholder precedence** to
   `rivalCode ? starved(frame.delta) : "single-driver mode"`. One line.
4. **G3 — catch `OSError` around `browser.start()`** so the documented fallback covers the
   documented case. Three lines.
5. **G5 — one word in `DataWindow.tsx:92`.**

---

# What I tried to break and could NOT

- **The transport itself.** With the exact `__main__` wiring live, the process owned ONE listening
  socket and `webview.http.global_server` was `None` — pywebview never started its bottle server,
  so the `commonpath` race #995 diagnosed cannot occur, rather than merely not occurring. Both
  windows loaded, rendered (screenshot read), and answered real `js_api` calls (`get_tick`,
  `get_bulk`, `get_connection`, `get_agents_view`) for 2.5 minutes; `localStorage` works on the
  new origin (and the app stores nothing anyway); the built HTML's `./assets/…` paths serve;
  destroying one window left the other still answering its polls through the bridge
  (release_window decremented, client kept);
  destroying both returned `webview.start()`, ran the finally block, and released the port.
- **`window_target`.** Trailing slash, no trailing slash, nested entry, base with a path, `None`,
  `""` — all correct. The tests would kill a URL-shaped-but-unfetchable regression: the fixture
  fetches the exact handed URL from the running server and checks the served document names the
  window.
- **`useFitsRanked`, at eleven client sizes and in motion.** 66 fresh mounts, 0 clipped,
  THEORETICAL always visible, deterministic form per size — including six sizes nobody had
  measured. One transition per sweep direction at one boundary, no hysteresis, stable holds. The
  `content` signature and the card observer are BOTH load-bearing, not redundant: the card observer
  alone is blind when the card's border box sits pinned at its `max-height` cap (growth changes
  `scrollHeight`, not the box), and the content signature alone is blind to the 8 px web-font swap;
  each covers the other's blind spot.
- **The pace ruler.** `table-layout: fixed` + a single 9 px font make the measured body cell
  form-independent, so the decision cannot feed on its own output; a planted 601.2 s lap coarsened
  the wide client exactly as `widestFine` promises and held stable; `widestFine` is computed from
  the fine form regardless of `coarse` in every branch (deleted laps counted before their branch,
  word cells flagged independently, empty bulk falls back to the constant).
- **The frozen treatment, everywhere I could aim it.** Real kill against real windows: both froze
  with all tells; the AGENTS playback chip is a dash, not `PLAYING`. Unlatch: both windows reverted
  completely. Pre-first-view: `get_agents_view` returns `None` until a payload exists, and
  `frozen` requires `view !== null`. Pixels: `brightness(0.72)` scales every channel of every
  sampled patch by the same factor — the hue ORDER that carries the pace ranking survives on all
  five tones, at a measured cost of 28 % of pair distance (vs the old filter's 62 %).
- **The compound sentinel.** Census over the full repaired Melbourne frame: no sentinel string
  exists on ANY row (generated included), every driver's last-row `tyreCell` renders byte-identical
  before and after the change, and `is_real_compound(None)` is `False`, so `_tyre_stops` cannot be
  moved by the new nulling. The one behavioural delta possible on other races — a `tyre_life`-only
  row now prints `—` instead of `n 24` — is the intended fix, not a casualty.
- **`formatSeconds`' new guard.** Three call sites, each null-guarded, wire non-negative by
  `allow_nan=False` and construction; `0` and `-0` take the unchanged arithmetic.
- **The suites and the claims.** 227 passed / 176 / 19 reproduce the commit's numbers; the rebuild
  is hash-identical to the shipped `dist/`; "6 and 14 cards" reproduces; X4's sentence, X6's
  17.48:1 AND its sibling 15.99-on-`--qt-elevated` explanation, and the CSS comment's 61 %/62 %
  matrix figures all recompute.

---

# Claims of the author's refuted, plainly

1. **The smoke guard's comment: "Driven against the un-fixed latch it still passes … the mount
   never commits to `ranked` against the empty height."** False on this machine, twice: the
   committed guard fails against the pre-fix bundle at 1265x650 (`ranked, 18 px hidden`) and
   1350x660 (`ranked, 8 px hidden`). The COMMIT MESSAGE's claim ("red against the un-fixed latch")
   is the one my execution supports; the comment contradicts it inside the same commit (G2).
2. **The commit message: "The file path stays as the fallback for when the bundle cannot be served
   at all."** True of `window_target`, not of the program: after `ui_is_built()` passes, `start()`
   cannot return `None` short of a mid-startup delete, and a real serving failure raises and kills
   `main()` before any window opens (G3).
3. **`DataWindow.tsx:92`: "Desaturated and mildly dimmed."** The class it annotates has not
   desaturated since this very commit (G5).
4. **Implicit in X9's fix: "a starved trace's caption explains the empty plot."** In single-driver
   mode the caption explains it with the wrong cause once frozen — the sentence is false while
   three full traces sit beside it (G4).

---

## Summary

**W1-W8 and X1-X10: 15 verified, 3 mixed (W5, W8, X9). Findings: 0 P0, 0 P1, 1 P2 (G1), 4 P3
(G2-G5).**

- **G1 (P2)** — the #995 wiring in `__main__.py` is protected by live observation only; a one-line
  regression to `spec.url` keeps all 422 checks green. Add the pure seam + test.
- **G2 (P3)** — the X1 guard works; its own comment says it does not, and contradicts the commit
  message. Rewrite the comment.
- **G3 (P3)** — the advertised file fallback is dead code in practice; the realistic failure
  crashes instead. Catch `OSError`.
- **G4 (P3)** — a frozen single-driver delta captions itself with a false cause. One-line
  precedence flip.
- **G5 (P3)** — one comment still says "Desaturated".

**Is the window fix safe to ship? YES.** The mechanism is right (pywebview provably starts no
internal server; the racy `commonpath` root is out of the picture, cause and not symptom), the
join is normalised and tested, the bridge and teardown work on the real OS windows against a real
producer, a real kill, and a real one-window close, and nothing that worked off the file URL
regressed. The one debt worth paying before this class of bug can be declared closed is G1: today
the only thing standing between a refactor and the 404's return is that somebody opens the real
window and looks.

---

## Probe inventory

Untracked probe files created by this gate — delete before any PR:

- `src/pitwall/ui/scripts/_wingate_host.py` — headless PitwallHost + BrowserServer for Playwright.
- `src/pitwall/ui/scripts/_wingate_a.mjs` — X1 mounts, sweep, holds.
- `src/pitwall/ui/scripts/_wingate_b.mjs` — X7/X8 ruler + injected 601.2 s lap.
- `src/pitwall/ui/scripts/_wingate_c.mjs` — X2 pixels, X3 unlatch, X9 captions.
- `src/pitwall/ui/scripts/_wingate_d.mjs` — G4 single-driver precedence.
- Scratchpad (outside the repo): `_wingate_census.py`, `_wingate_census2.py`, `_wingate_pixels.py`,
  `_wingate_windows.py`, the `oldui/` isolated pre-fix build (junctioned node_modules), and the
  `wingate-live.png` / `wingate-frozen.png` / `live-windows.png` screenshots.

**Environment at gate end:** the gate's producer and headless host were stopped; the real windows
were destroyed by the probe itself; no repository file modified except this report. `npm run build`
was run once (hash-identical output). The X3 kill left no producer running — start
`scripts/dev_pitwall_producer.py` before reopening the windows.

---

## What I tried to break and could NOT

*(appended at the end)*

---

## Probe inventory

*(appended at the end)*
