# ADVERSARIAL DESIGN GATE — PITWALL DATA (telemetry) window

**Date:** 2026-08-18 · **Auditor:** adversarial design gate (Fable), sprint 9 pre-work
**Repo state:** `dev` @ `d4ef7cf` (Merge PR #977)
**Bundle measured:** the one the live loopback server snapshotted at startup —
`dist/assets/data-DWGt9GtQ.js` (24.6K) + `data-DAzFNt-6.css` (9.6K) + `qt-base-86dNR9Zk.css`.
No rebuild was performed at any point; the served bundle is the measured bundle.
**Session measured:** real Melbourne 2025, real producer (`scripts/dev_pitwall_producer.py`),
mid-race (lap ~24 at start of audit), loopback `http://127.0.0.1:62712/data.html`.
**Inputs given:** the 10 `features/data/*.tsx` files + `traceBuffer.ts`, the 13 `lib/*.ts` files,
`styles/data.css` (1037 lines) + `qt-base.css`, the backend wiring
(`session_data.py`, `host.py`, `webserver.py`, `radio_feed.py`, `config.py`,
`arcade/app.py`, `arcade/stream.py`), the live `/api/tick|bulk|live|connection` endpoints,
9 real-window screenshots in `~/.claude/plans/pitwall-sprint9/shots/`, and the author's own
`MEASURED-BASELINE.md` — which I was instructed to treat as claims to attack, not as ground truth.
**Probe:** `src/pitwall/ui/scripts/_gate_probe.mjs` (my own untracked copy, extended from the
author's `_probe_s9.mjs`; left in place, delete before PR).

**What this gate judges:** does a race strategist TRUST this window at a glance and can they
READ it — density, hierarchy, legibility at real sizes, colour semantics, first-glance eye path,
dangerous-state vs benign-state distinguishability, wasted area. NOT fidelity to any prior window.

---

## Checklist — claims and surfaces set out to attack (written before any finding)

**Baseline claims to verify/refute (MEASURED-BASELINE.md):**
- [x] B1. Geometry table at 1485x833 and 1265x593 (left-column 600-in-510, pace cell 27.75px, 411/520 clipped).
- [x] B2. BESTS loses rows 2-3 + THEORETICAL footer at 1265x593, silently.
- [x] B3. Pace card 60% empty at lap 24; scroll pin dead until ~lap 55. Attack the CAUSE, not just the number.
- [x] B4. SC-lap ranking: 213/776 ranked rows on SC laps; `LapRow.track_status` read by nothing.
- [x] B5. Driver colour contrast: VER/LAW 1.88:1 on panel; three copies of the colour render site.
- [x] B6. Dead-producer state "still reads as live".
- [x] B7. `.bests-row` overflows its 146px box by 7px.

**Surfaces to attack beyond the baseline:**
- [x] S1. StatusStrip — hierarchy, what a state change looks like mid-glance, PROVISIONAL/track-status semantics.
- [x] S2. TimingTower — column semantics, gap column, tyre/stint encoding, leader row, retired/generated cars, interval readability.
- [x] S3. BestsPanel — does it earn 153px of the left column; sentinel handling.
- [x] S4. OwnCarTraces 2x2 + TraceChart — axis honesty, units, y-domain policy, rival overlay semantics.
- [x] S5. TrackRing — dot pileup, team-colour ambiguity, SC state, what it earns vs its area.
- [x] S6. RadioFeed — timestamps, ordering, truncation, empty state.
- [x] S7. RacePaceGrid — colour scale semantics, IN PIT/OUT sentinels, SC laps, row order, newest-lap ergonomics.
- [x] S8. RaceTraceChart — reference-line semantics, label collisions, y-axis direction, team-mate ambiguity.
- [x] S9. Footer status bar + tab strip — do they say anything a strategist uses.
- [x] S10. Cross-cutting: unknown-rendered-as-confident, sentinel collisions, twin computations, dead/phantom wire fields, comments whose headline is false.
- [x] S11. States: waiting / dead / SC lap 1 / empty trace — is a dangerous state distinguishable at a glance.
- [x] S12. The tab strip itself: what is hidden behind an unselected tab while it matters (radio during SC, pace during traces).

**Method:** every finding below carries executed evidence (a measured box, a payload field, a
contrast ratio, a crop, a line of served JSON). Findings are appended in the order confirmed,
then severity-ranked in the fix list.

---

## Findings

### D1 · P1 — The STOPS column says every car has made 3 stops when the true count is 0, and the tower prints the contradiction in the adjacent cell

- **Where:** `src/pitwall/session_data.py:447` (`"stops": sum(1 for row in revealed if row["pit_in"])`), rendered at `src/pitwall/ui/src/features/data/TimingTower.tsx:155`.
- **What the strategist loses:** the stop count is the number a rival-strategy read starts from ("they have used their stop / they still owe one"). This window reports **3 stops for all 17 runners from lap 5 onwards** on the only race on disk, when the real tyre-stop count at that point is **zero**. Any undercut/overcut reasoning seeded from this column is wrong all race.
- **Executed evidence:**
  - `laps.parquet` Melbourne 2025: `PitInTime` is set for **17 cars on each of laps 2, 3 and 4** — the safety car led the field through the pit lane three times while the lap-1 wrecks were cleared. NOR's rows: laps 2-4 carry `PitInTime` while `Compound` stays INTERMEDIATE and `TyreLife` counts 2 → 3 → 4 uninterrupted; FastF1 nevertheless opens a new `Stint` each pass (4 distinct stints by lap 24, zero tyre changes).
  - Live probe at lap 31, real bundle: `col-stops` = `"3"` for all 17 running rows while `col-tyre` reads `"I 30"` — a thirty-lap-old set sitting beside a claim of three stops, in adjacent cells of the same row.
  - The pace grid (D7) paints the same three laps as full-field `IN PIT` rows plus a full-field `OUT` row, in DANGER red.
- **The bug-class instance:** `session_data.py:420` — *"`stops` counts in-laps rather than `max(stint) - 1`. The two agree on every driver of a healthy race"* — the parenthetical is TRUE (both count 3 here) and the headline is FALSE: on this race **neither** counts stops. A true clause inside a false headline, in a docstring, on the exact race the module ships against.
- **Prescription:** count a stop as a **tyre-set transition**, which is already computable from fields on the wire: consecutive `LapRow`s where `compound` changes or `tyre_life` resets downward (`laps[i+1].tyre_life < laps[i].tyre_life`). Do it producer-side in `_driver_view` (one reduction, masked rows only) and correct the docstring; the wire shape does not change. No new palette constant, no new field.

### D2 · P1 — SAFETY CAR is an 18-px chip; the window's only reaction to the race's most decision-dense state is ~0.4 % of its pixels

- **Where:** `StatusStrip.tsx:37-40` + `.strip-chip` (`data.css`).
- **What the strategist loses:** the SC window is where pit-now calls are made in seconds. In this window the ENTIRE difference between green running and safety car is one outline chip swapping text (`GREEN`, measured 54.3x18 px, for `SAFETY CAR`, ~86x18 in the capture) in a 1465x28 strip. Every panel below keeps its composition, colours and rhythm. Glanced at from the arcade window beside it, the two states are the same picture. The dedicated shot (`state-safetycar-lap1.png`) needs a deliberate look at the top-left corner to notice the state at all.
- **Executed evidence:** chip boxes from the live probe (`GREEN` 54.3x18, `Connected` 76.9x18); `state-safetycar-lap1.png` vs `pace-1485x833.png` — no other element differs in kind. The wire already carries `track_status_label` + `track_status_color` on every tick.
- **Prescription (traces to fields on the wire, respects the settled palette):** keep the outline chip for GREEN, but render any non-green label as a **filled** chip — `track_status_color` background, `--qt-bg` text — and let the status strip's card border take the same colour. Both colours are already the palette's own, delivered by the producer; no new constant. A salience change measured in hundreds of pixels, not a redesign, and it degrades honestly: `NO STATUS` stays the dim unknown chip.

### D3 · P1 — A dead producer leaves a window full of confident, frozen numbers; `PLAYBACK 2x` keeps claiming motion

- **Where:** `DataWindow.tsx:70-71` (`live` latches true forever after the first tick, so `StatusStrip` keeps the last tick), `StatusStrip.tsx:127-130` (`playbackLabel` renders the LAST tick's `2x`), plus every panel holding its final values.
- **What the strategist loses:** the ability to distinguish "the race is like this" from "my feed died 40 seconds ago". Tower, BESTS, traces, ring, radio and the session clock all hold their last state; the lap counter still says `L 28/57`; the track chip still says `GREEN`; `PLAYBACK 2x` still asserts the replay is advancing. The only tells are a 76x18 red chip and a status bar that quietly went blank (`useStatusText`'s 1.5 s timeout).
- **Executed evidence:** `state-dead.png` — fully-populated window, `Disconnected` chip, empty footer. The probe's dead-state route reproduces it deterministically.
- **Prescription:** the connection channel already answers `"Disconnected"` (`useConnection`, 1 Hz). On that value: render `PLAYBACK —`, drop `.data-main` to a visibly stale treatment (reduced opacity / grayscale filter on the two columns), and put a non-transient `DATA FROZEN · last tick L28` in the status bar (`useStatusText` already supports non-transient text). No wire change; the state is already known client-side.

### D4 · P0 — At the 1080p-laptop client size the pace grid truncates the final glyph of ~96 % of its populated cells, silently

- **Where:** `.pace-table` (`data.css`: `font-size: 9px`, `table-layout: fixed`, cell `overflow: hidden`, padding `0 1px`) at the 1265x593 client; scrollbars globally hidden (`qt-base.css`).
- **What the strategist loses:** the number itself. `1:59.4` renders `1:59.`, `2:24.1` renders `2:24.` — the tenths digit, the exact resolution the panel's own docstring says it exists to carry ("tenths is also the resolution the grid needs"), is deleted from nearly every cell, and `IN PIT` renders `IN PI`, running into the neighbour as `IN PIIN PI`. Nothing announces the cut; a trailing dot is the only tell.
- **Executed evidence:** live probe at 1265x593, lap 31: **495 of 514 populated cells clip** (`scrollWidth 35 > clientWidth 28`); samples `1:59.4 / cw 28 / sw 35`. Screenshot `pace-1265x593.png` shows the run-together `IN PIIN PI` rows. (Baseline claimed 411/520 at lap 24 — same defect, smaller lap population; both executed.)
- **Prescription:** the cell needs ~7 px it cannot get at 583 px over 20 columns, so change the CONTENT, not the box: below a measured column-width threshold render the width-aware form `m:ss` (a 4-glyph label measures ~22 px at 9 px mono) and say so once in `.pace-subtitle` ("times to the second at this width"). The tone already carries the ranking, which is the panel's stated design; the value shown is then a true value, just coarser. Client-only; no wire change. A degraded-but-honest cell beats a silently wrong one.

### D5 · P1 — At the same client size the BESTS panel silently loses everything below its first row, THEORETICAL included

- **Where:** `.left-column` (`grid-template-rows: auto minmax(0, 1fr)`) — the tower is fixed at 437 px, so at a 510 px column the bests card's slot is 63 px against 151 px of content; scrollbars hidden, no overflow affordance.
- **What the strategist loses:** ranks 2-3 of all four sections (the "who is close to the best" half of the panel) and the THEORETICAL footer — with nothing on screen saying the panel continues. A reader at this size believes BESTS is a one-row panel.
- **Executed evidence:** live probe at 1265x593: bests card bottom 650 vs left-column clip edge 560 — **90 px of the card cut**; `visibleCount: 0` of 12 rows fit whole (row 1's text is readable in `pace-1265x593.png`; its box bottom lands past the edge), `theoreticalVisible: false`.
- **Prescription:** the panel already owns its row count (`RANKED = 3`, `BestsPanel.tsx:29`). Make the count answer the measured space: at a short client render 1 ranked row per section and keep THEORETICAL — the same honest degradation the panel's own docstring argues for ("the fourth-fastest S2 is not a number anybody reads off a live wall"; at this height, neither is the third). Client-only.

### D6 · P1 — On 213 of 776 ranked cells the pace colour encodes the safety-car queue, not pace — and the field that could say so (`LapRow.track_status`) is on the wire, read by nothing

- **Where:** `lib/racePace.ts` (`rankedByLap`/`tone` rank every non-pit racing lap), `lib/bridge.ts:111` (`track_status` typed per lap row, zero consumers — the only track-status read in the window is the ARCADE-level chip at `StatusStrip.tsx:38`).
- **What the strategist loses:** on the 22 SC laps (1-7, 33-41, 46-51 — 27.4 % of everything the grid ranks) a green cell means "happened to be at the compressing end of the accordion", not "quick". Reading tyre life or driver form off those columns — during exactly the laps stop decisions get made — is reading noise.
- **Executed evidence (past the baseline's counts, to the mechanism):** recomputed on the parquet: 776 ranked rows, 213 on SC laps, SC lap times 86.4-148.2 s (median 131.6) vs green median 91.9. Then, per SC lap, Spearman correlation between lap-time rank and running position: **|rho| >= 0.75 on 11 of 17 measurable SC laps**, flipping sign with the accordion phase (lap 1: +1.00; lap 7: -1.00; laps 39-41: -0.88 to -0.98; laps 49-51: -0.79 to -0.96). The thirds are queue order wearing pace colours, in both directions.
- **Prescription:** mark the lap from the dead field: a glyph or amber lap number in `.pace-lapcol` for any lap where the majority of rows carry a `4` digit (SC), and an ECharts `markArea` over the same lap ranges on the race trace's x-axis (which also explains that chart's laps-5-8 V shape on screen). The label converts a lie into a caption; client-only; the field already crosses the bridge on every row.

### D7 · P2 — "OUT" means retired in the tower and out-lap in the pace grid — the exact word-collision the tower's own docstring says it exists to avoid

- **Where:** `lib/racePace.ts:246` (`if (row.pit_out) return { text: "OUT", tone: "out" }`) vs `TimingTower.tsx:227-233` (`lastCell`: retired → `OUT`, out-lap → `PIT EXIT`; its docstring: *"an out-lap says PIT EXIT rather than borrowing the same word for a car that is very much still racing"*) and the ring legend's `○ out` (retired).
- **What the strategist loses:** on `pace-1485x833.png`, BEA's lap-1 cell reads `OUT` in DANGER red (an out-lap — he is racing) while three rows down the tower prints `OUT` for SAI/DOO/HAD (crashed). Same window, same four letters, opposite meanings. Lap 5 shows a full-field red `OUT` row (the SC pit-lane pass), which in tower vocabulary is 17 simultaneous retirements.
- **Executed evidence:** live probe: 19 `OUT` cells and 51 `IN PIT` cells in the grid at lap 31, while the tower's `col-last` simultaneously holds three `OUT`s meaning retired. The tower fixed this collision explicitly; the grid is the copy that never got the fix — the repo's dominant defect class, inside one window.
- **Also here:** the pit tone is `#ef4444` — palette DANGER, the same red as the `Disconnected` chip — for a routine event, and it measures **4.25:1 on the banded columns** at 9 px (below the 4.5:1 floor for this text size). Three all-red full-field rows (laps 2-4) read as a crisis.
- **Prescription:** rename the grid's out-lap cell to the tower's own `PIT EXIT` (`P.EXIT` if width demands) — one string in `racePace.ts`, and the smoke that pins cell text moves with it. If red is kept, keep it for the in-lap only and give the out-lap the WARNING amber already in this stylesheet, so only the lap you cannot unwind is red.

### D8 · P2 — The radio feed drops the two wire fields that rank its lines, so SAFETY CAR THROUGH THE PIT LANE renders identically to stewards boilerplate

- **Where:** `lib/bridge.ts:164-165` (`category`, `flag` — typed, populated, zero consumers), `RadioFeed.tsx` (renders `kind`, `lap`, `driver`, `text` only).
- **What the strategist loses:** the feed's fold is ten rows (measured), and on the live session eight of the visible RCM rows are the near-identical *"FIA STEWARDS: INCIDENT INVOLVING CAR N (XXX) NO FURTHER ACTION"* clamped to two lines — while `SAFETY CAR THROUGH THE PIT LANE` (category `SafetyCar`) and `DOUBLE YELLOW IN TRACK SECTOR 20` (flag `DOUBLE YELLOW`) wear exactly the same dress. The one panel that could shout the race's state whispers it in a crowd of no-further-action notices.
- **Executed evidence:** live `/api/bulk` at lap ~31: 39 RCM events carrying `category` in {Other: 23, Flag: 13, SafetyCar: 2, Drs: 1} and `flag` in {CLEAR: 8, BLUE: 4, DOUBLE YELLOW: 1} — all discarded at render. `state-safetycar.png` shows the visible fold: 9 of 11 rows are stewards/incident boilerplate.
- **Dead-wire census while here:** `LapRow.position`, `LapRow.stint`, `LapRow.pb` and `DriverLaps.theoretical` also cross the bridge and are read by nothing in this window (grep over `features/data` + `lib`). `pb` and per-driver `theoretical` are documented as deliberately recomputed/unused; `position` and `stint` are just freight.
- **Prescription:** use the two fields that already arrive: a compact category chip on RCM rows (`SC`, `FLAG`, `DRS` — text chip in existing tones: WARNING for SafetyCar/Flag, fg-3 for Other), and collapse CONSECUTIVE identical-text RCM rows into one row with a `x4` count (the four identical BLUE FLAG lines the component's own comment describes). Both are client-only.

### D9 · P2 — Radio history is unreachable: 36 of 46 events sit below the fold with `overflow: hidden`, and no user input can scroll to them

- **Where:** `.radio-list` / `.radio-feed` (`data.css`: both `overflow: hidden`).
- **What the strategist loses:** re-reading anything older than ~10 events — "what exactly did race control say two laps ago?" has no answer on this surface. This is materially different from the window's hidden-scrollbar rule: `qt-base.css` hides the CHROME and keeps bodies `overflow: auto` ("a wheel, a trackpad, a touch drag and the keyboard all still reach the content"); the radio list is the one panel where the overflow is `hidden`, so the content is genuinely gone, not just unannounced. The header count (46) says how many exist, not that 36 are unreachable.
- **Executed evidence:** live probe: `listOverflowY: "hidden"`, `listCH 344` vs `listSH 1379`, `visibleRows: 10` of 46; setting `scrollTop = 200` programmatically STICKS (the content is there and scrollable by JS — only the user has no path to it).
- **Prescription:** `overflow-y: auto` on `.radio-list` — one declaration. Scrollbars are already invisible globally, so the pixels do not change; the wheel starts working. Newest-first ordering already puts the fold where it should be.

### D10 · P2 — The rival's broadcast-tier dashes paint OVER the own car's pit-wall-grade trace on all four charts — the opposite z-order to the one the race trace deliberately builds

- **Where:** `TraceChart.tsx:100-119` — series array is `[main, rival]`, so ECharts paints the rival LAST (on top). Contrast `lib/raceTrace.ts:269-270`: *"our own car moved LAST so it draws on top of the nineteen it has to be picked out from"* — the same rule, applied in one panel and inverted in the other.
- **What the strategist loses:** whenever the two cars run comparable numbers — which is precisely when the comparison matters — the own car's line is under the rival's dashes. In `traces-full-1485x833.png` the Speed and Throttle plots read as a single amber dashed line with slivers of blue; the solid line the panel exists to show is the one you cannot see.
- **Executed evidence:** the series order at `TraceChart.tsx:100-119` (main first, rival second; ECharts z-order is declaration order), plus the capture above.
- **Prescription:** declare the rival series first and the main series second (keep the markLine on whichever series is first — it hangs off index 0 by design, so move the marks or hang them off the rival series). Two-line reorder, no palette or wire change, and it re-aligns the two panels' shared rule.

### D11 · P2 — At 1485x833 the pace card is half empty and the newest lap — the only row a live wall reads — walks down the screen for 30+ laps

- **Where:** `.pace-scroll` (no flex-grow: the scroller is content-sized inside a 720 px card), `RacePaceGrid.tsx:73-77` (the `scrollTop = scrollHeight` pin).
- **What the strategist loses:** a stable place to look. The newest row starts at the top of an otherwise-empty card and migrates 12 px per lap (probe: row at y 471 on lap 30 of an 833 window; 348 px of the card empty — 434 px / 60 % at the baseline's lap 24). The pin the code carries is a no-op until the table outgrows the card (probe: `scrollable: false`, `scrollTop: 0` at both client sizes), which at 1485x833 happens around lap 55 of 57 — the pin activates for the last three laps of the race. Meanwhile the radio feed one tab over solved the identical problem by anchoring the newest entry to a fixed edge; the grid is the twin that kept the archive convention.
- **Executed evidence:** probe at 1485x833, lap 30: `cardH 720, tableH 372, emptyCardPx 348, newestRowY 471, scrollable false`. Cause verified in CSS: `.pace-scroll { min-height: 0; overflow-y: auto; }` with default `flex: 0 1 auto` — it never grows, and the pin has nothing to pin until content exceeds the card.
- **Prescription:** anchor the table to the scroller's BOTTOM edge (e.g. `.pace-scroll { display: flex; flex-direction: column; } .pace-table { margin-top: auto; }`): the newest lap then sits at a fixed y from lap 1 to lap 57, the empty space (honest — the race has not happened yet) moves above the history where the eye does not start, and the existing pin takes over seamlessly once the table fills. Add a subtle current-row emphasis (`tbody tr:last-child th` in fg-1) so "now" is findable in one saccade. CSS-only.

### D12 · P2 — Six of twenty driver codes are illegible where identity is the panel's whole answer: VER/LAW at 1.88:1, ALO/STR at 2.55:1, HAM/LEC at 3.71:1

- **Where:** three render sites (baseline confirmed): `TimingTower.tsx:266-269` (`driverColour`, 11 px bold code), `RaceTraceChart.tsx:52-59` + `endLabel` at 9 px (its own docstring: *"the end label is the only thing that tells two team-mates apart"*), `TrackRing.tsx:62-64`.
- **What the strategist loses:** the tower's DRV column — the row key of the window's primary panel — is near-invisible for VER and LAW (`racetrace-1485x833.png`: VER's end label at 9 px in rgb(6,0,239) on `#181633` has to be found, not read). On the race trace the failing label is the ONLY identification of the line, per the component's own comment.
- **Executed evidence:** recomputed WCAG ratios (independent implementation, agrees with the baseline to the second decimal): VER/LAW 1.88 (panel) / 1.72 (elevated); ALO/STR 2.55 / 2.33; HAM/LEC 3.71 / 3.39. Four of twenty fail even the 3.0:1 large-text floor; text sizes here are 9-11 px, so 4.5:1 applies.
- **Prescription (keeps the settled team-colour rule — colour stays, it stops being the TEXT fill):** identity by colour is already impossible (two cars per colour), so the colour does not need to carry the glyphs. Tower: a 3-px team-colour swatch bar before the code, code itself in `--qt-fg-1` (both already exist; no new constant). Race trace: `endLabel.color` = AXIS_TEXT (`#d1d5db`, 11.9:1) for every code — the label sits ON its line's end, so adjacency keeps the mapping; own car stays bold. Ring: no change (dots are fills, not text). The tokens test moves with the sites it pins.

### D13 · P3 — The minute-boundary rounding defect `paceLabel` fixed and documented is alive in BOTH of its sibling formatters, which are also duplicates of each other

- **Where:** `TimingTower.tsx:252-257` (`formatLapTime`) and `BestsPanel.tsx:96-101` (`formatTime`) — two byte-identical functions in two files; compare `lib/racePace.ts:139-144` (`paceLabel`), whose docstring names the defect: *"Splitting first and rounding the remainder renders a non-time in the 50 ms under every minute boundary: 119.96 s came out as 1:60.0"* — and fixes it by rounding FIRST.
- **Executed evidence:** evaluated the shipped arithmetic in node: `formatLapTime(59.9996)` → `"60.000"` (not `1:00.000`), `formatLapTime(119.9996)` → `"1:60.000"`. At three decimals the window is 0.5 ms per minute boundary — no lap on this disk lands in it; a season of sector times (three per lap per car) eventually will.
- **What the strategist loses:** little, rarely — this is filed for the CLASS: a documented fix whose two twins in the same window never got it, plus a duplicated helper that contradicts the window's own one-copy doctrine (`gapCell`, `driverStatus`, `sessionBests`, `stableColumns` all centralise for exactly this reason).
- **Prescription:** one `formatSeconds(seconds, decimals)` in `lib/` with `paceLabel`'s round-first arithmetic; the three call sites import it.

### D14 · P3 — The tower is tyre-blind for the whole of lap 1 — the one lap where "who started on what" is the question — while the tick carries every car's fitted compound

- **Where:** `TimingTower.tsx:245-249` (`tyreCell` reads the last COMPLETED bulk row; null until lap 1 completes) vs the tick's `drivers[code].compound` (int) + `tyre_life`, which are on every tick from frame 0.
- **What the strategist loses:** on a mixed-conditions start (this race: wet, everyone on inters, three cars dead by lap 2) the TYRE column is twenty dashes exactly when a slick-vs-inter split would be the first thing a wall checks. Evidence: `state-safetycar-lap1.png` / `state-emptytrace-lap1.png` — TYRE, ST, LAST, STOPS all dashes for all 20 rows.
- **The honest constraint:** the tick's `compound` is an integer whose letter table lives in `src/arcade/palette.py` (`COMPOUND_COLORS` keys 0-4); decoding it client-side would fork that constant across languages — the exact defect `OwnCarTraces` refuses for `drs`. So this is **priced as a producer change**: publish a decoded compound letter (or name) per driver on the tick, the same treatment `track_status_label` already gets, then let `tyreCell` fall back to it while the bulk has no row.
- **Prescription:** producer: `drivers[code].compound_name` beside the int (one lookup it already owns); client: fallback in `tyreCell`. Until then the dash is at least honest.

### D15 · P3 — "NO POSITION: HAD" is a permanent caption for a car that crashed on lap 1, which spends the ring's only telemetry alarm on a non-event

- **Where:** `TrackRing.tsx:83-95` — the blind list collects every car with `rel_dist === null`, regardless of status; HAD has no telemetry rows at all on this race, so the line renders from lap 1 to lap 57 (visible in every capture, all states).
- **What the strategist loses:** the blind line exists to flag a RUNNING car the telemetry lost — a real alarm. Permanently lit by a retired car, it becomes furniture; the day a live car goes blind, the alarm it raises has already been trained away (the same reads-as-nothing failure the PROVISIONAL docstring names: "a permanent PROVISIONAL says nothing").
- **Executed evidence:** every one of the nine captures shows `NO POSITION: HAD`; HAD's status is `out` from lap 1 (tower row 20, `driverStatus` = out).
- **Prescription:** filter the blind list to cars whose `driverStatus(car) !== "out"` — retired cars are already accounted for by the legend's hollow-dot state and the tower's OUT. Three lines in `TrackRing.tsx`.

### D16 · P3 — The window's controls are 15-22 px hit targets, and the race trace's only control is the smallest thing on it

- **Where:** `.ref` (`data.css`: 9 px font, `padding: 1px 6px`) and `.tab` (10 px font, `padding: 3px 10px`).
- **Executed evidence:** probe boxes — refs `LEADER` 50.2x15, `FIELD` 44.2x15, `NOR` 32.1x15; tabs 64-91 x 22.
- **What the strategist loses:** time and misclicks under pressure; 15 px is half the ~28-32 px a comfortable mouse target wants, and these three buttons change what the whole panel MEANS (the zero line).
- **Prescription:** `padding: 4px 8px` on `.ref`, `5px 12px` on `.tab` — pure whitespace, no reflow risk beyond the header rows they sit in (both headers have slack at both client sizes; the ref strip is `margin-left: auto` in a 28-px header).

### D17 · P3 — One amber, four meanings — and the stylesheet's own comment claims the opposite

- **Where:** `#f59e0b` is simultaneously: BROADCAST tier (rival trace + rival chip + `.radio-tier`), the PROVISIONAL warning chip, the tower's "slower than own best" sector tone, and the pace grid's slowest third (`is-t3`). `data.css`'s `.radio-tier` comment asserts: *"the same WARNING the rival chip uses, so one colour means one tier across the whole window."*
- **The bug-class instance:** the clause is true (the tier tag does share WARNING with the rival chip) and the headline is false (one colour does NOT mean one tier across the whole window — on the very tab that comment styles, amber is also every t3 pace cell and every yellow sector one column left). A true clause inside a false headline, in a stylesheet comment.
- **What the strategist loses:** modest in practice — each surface disambiguates locally — but the window has spent its entire WARNING budget four ways, which is why D2 (safety car) had nothing loud left to reach for.
- **Prescription:** fix the comment now (cheap, stops the claim propagating); when D2 lands, let non-green track status own the FILLED-amber treatment so weight, not hue, separates "state of the race" from "annotation".

---

## Fix list, ordered by value over risk

1. **D9** — `overflow-y: auto` on `.radio-list`. One declaration, zero visual change, restores the whole radio history.
2. **D10** — swap the two series in `TraceChart.tsx` so the own car paints on top. Two lines.
3. **D7** — rename the pace grid's out-lap `OUT` → `PIT EXIT`; move the red to in-laps only. One string + one CSS tone.
4. **D2** — filled chip + strip border colour for non-green track status. CSS + ~5 lines; the fields already ride every tick.
5. **D6** — SC lap marker in the pace lap column (+ `markArea` on the race trace). Reads the dead `track_status` field; converts 213 lying cells into captioned ones.
6. **D3** — dead-producer treatment: `PLAYBACK —`, stale filter on `.data-main`, non-transient `DATA FROZEN` status line. Client-only.
7. **D11** — bottom-anchor the pace table (`margin-top: auto`) + last-row emphasis. CSS-only; fixes the walking row and the dead pin together.
8. **D1** — stops = tyre-set transitions in `session_data._driver_view` + docstring correction. Producer-side, wire shape unchanged; the one fix here that touches Python.
9. **D4** — width-aware `m:ss` form for the pace label at narrow columns + header note. Client-only; needs the smoke's cell-text pins updated.
10. **D5** — height-aware `RANKED` (3 → 1) + keep THEORETICAL at short clients. Client-only.
11. **D12** — tower swatch-bar + fg-1 code; race-trace end labels in AXIS_TEXT. Touches the tokens test deliberately.
12. **D8** — RCM category chips + consecutive-duplicate collapse in `RadioFeed`. Client-only.
13. **D15** — filter retired cars out of the ring's blind list. Three lines.
14. **D16** — padding on `.ref` / `.tab`. CSS-only.
15. **D13** — one shared `formatSeconds` with round-first arithmetic. Refactor + micro-fix.
16. **D14** — producer publishes `compound_name` per driver; tower falls back on lap 1. Producer change, priced separately.
17. **D17** — correct the `.radio-tier` comment. One comment.

---

## Where I disagree with MEASURED-BASELINE.md

- **The BESTS clip at 1265x593 is 90 px, not 80.** The baseline's own geometry table implies it (left-column content 600 in a 510 row) and its bullet says "80 px clipped"; measured live: card bottom 650 vs column clip edge 560. The table and the bullet disagree inside the baseline, and the table is right. Its statement of the LOSS also flatters slightly: not only rows 2-3 and the footer go — row 1's own box already crosses the clip edge (`visibleCount: 0` whole rows), though its glyphs remain readable in the capture.
- **".bests-row overflows its own 146 px box by 7 px" is true of 2 rows of 12, not of the row class.** Only the LAP section's ranked rows 2-3 — the ones carrying a delta AND a compound letter — overflow (probe: exactly 2, `sw 153` vs `cw 146`, both sizes). Stated as a property of `.bests-row`, it would send a fix at all twelve rows when the offender is the LAP section's column budget.
- **The 411-of-520 clip count was already stale when I measured (495 of 514 at lap 31)** — not wrong, but it is a lap-dependent count presented as a property of the layout. The stable statement is: every populated cell whose text is the 6-glyph form clips by ~7 px at this client size, i.e. ~96 % and rising with the race. Same defect, sturdier phrasing.
- **The pace-pin claim ("dead until about lap 55") checks out**, including the cause — I verified the scroller is content-sized (`flex: 0 1 auto`, probe `scrollable: false, scrollTop: 0` at both sizes) rather than merely observing the emptiness. No refutation; the baseline's stated cause is the right one.
- **The colour-contrast table replicates exactly** under an independent implementation (all six figures to the second decimal). Its framing "this is a RENDER-SITE problem" is right and D12 builds on it.
- **"213 of the 776 ranked rows sit on a SC lap" replicates exactly** — and understates the harm: the baseline says the rows are ranked while neutralised; the Spearman measurement (D6) shows the resulting colours are not merely diluted pace but a different variable (queue position) in pace's clothes.

---

## What I tried to break and could NOT

- **`gapCell`'s four-branch contract.** Probed the tower live across leader/lapped/retired rows: `LEADER`, `+N LAPS`, seconds at two decimals, `OUT` for SAI/DOO/HAD, `—` on lap 1 (no classification). The order of decision holds; no clamped inversion found; the lap-quantised (L) claim is carried once in the header as designed.
- **`sessionBests` / tower purple consistency.** The purple in the tower's sector cells, the BESTS panel's leaders and the pace grid's single `is-best` cell (probe: exactly 1 in 600) all come from the one module; I could not make two panels disagree about which lap was quickest. The tie-handling difference (a to-the-millisecond sector tie would paint two purples in the tower while BESTS lists one) is real but requires an exact 3-decimal tie between team rivals — I could not exhibit it on this race and file it as a note, not a finding.
- **The reveal mask.** No future data leaks: BESTS at lap 1 is honestly empty ("waiting for a sector nobody has set yet"), the race trace refuses to draw without a common lap and SAYS so (`state-emptytrace-lap1.png`), the radio count grows with the reveal. A rewind was not testable against the live producer without disturbing it (the producer is shared with the author's session); the eviction paths are covered by the existing smoke and were not re-proven here.
- **`stableColumns`.** The car-number order is total and stable live (probe headers in strict number order); the documented Infinity-for-unknown rule holds in code; could not produce the old cyclic-comparator failure.
- **The waiting state.** Honest: `NO STATUS` chip, dashes, an instruction that names the real command, nothing invented (`state-waiting.png`).
- **The status strip's unknowns.** `NO STATUS` renders dim, never green — the "absence must not borrow a meaning" rule holds everywhere I looked (gap `—`, tyre `—`, ring blind list rather than a dot at fraction 0).
- **The RCM one-lap lag** (a lap-33 SC message appears when the leader completes lap 33) is real but DOCUMENTED as blocked on the UTC↔SessionTime bridge (#931/#842), and `radio_feed.py` states the coarse-reveal rule on screen. Not filed: the blocker is named and owned elsewhere.
- **The traces' empty-at-lap-start behaviour** (rival trace opens each lap empty for the gap between the cars) is inherited, documented in `traceBuffer.ts`, and correct given the distance-keyed store; the alternative (mixing laps) measurably spikes the delta 4-6 s. Not filed.
- **The delta chart's locked [-3, 3] range** clips the rival line entirely when the gap exceeds 3 s (it happens around stops). It is the Qt panel's own locked-axis design, the docstring concedes it, and the settled rules forbid autorange churn; the SC markArea of D6 does not help here. Noted, not filed: at 10 Hz a locked axis is the right call and the tower's GAP column carries the number the chart momentarily cannot.

---

## Summary

| Severity | Count | Ids |
|---|---|---|
| P0 | 1 | D4 |
| P1 | 5 | D1 D2 D3 D5 D6 |
| P2 | 6 | D7 D8 D9 D10 D11 D12 |
| P3 | 5 | D13 D14 D15 D16 D17 |

Bundle measured: `data-DWGt9GtQ.js` / `data-DAzFNt-6.css` on the live loopback server, Melbourne 2025, laps 24-32 during the audit. Probe: `src/pitwall/ui/scripts/_gate_probe.mjs` (mine, untracked — delete before the PR).
