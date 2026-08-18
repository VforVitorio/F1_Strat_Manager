# ADVERSARIAL EXIT GATE — PITWALL Sprint 9 (DATA window)

**Date:** 2026-08-18 · **Auditor:** adversarial exit gate (Fable 5)
**Tree measured:** branch `dev` @ `239babd` (Merge PR #994). Sprint bounded by `d07a10a` (#989, stops)
through `239babd`; diff base for the UI work `3492934..dev` (20 files, +1999/−162).
**Live host:** the handed-over host at `127.0.0.1:57594` had exited by the time I reached it, so I
started my own headless host + loopback server against the SAME already-running producer
(`PitwallHost` + `BrowserServer(ui_dist(), host)`, no pywebview) at **http://127.0.0.1:58476/** —
serving the same `dist/` the sprint built, snapshotted at that host's startup. **No rebuild was
performed at any point; the served bundle is the measured bundle.** The producer was the real
`scripts/dev_pitwall_producer.py` broadcasting Melbourne 2025 from frame 60000 (mid-race, laps 23-51
observed across the session).
**Session data:** `data/raw/2025/Melbourne` parquet set (the same the producer serves).
**Ran:** `pytest tests/surfaces/ -q` · `node scripts/smoke-data.mjs` · `node scripts/smoke-agents.mjs`
· six purpose-built Playwright probes, `src/pitwall/ui/scripts/_exit_probe{,2,3,4,5,6}.mjs`
(they exist, untracked; inventory at the end of this report)
· read-only Python censuses against the parquet.

**Contract:** no repository file modified except this report. No `git` state change, no build.
Earlier gates' findings (GATE_PITWALL_DATA_DESIGN.md, GATE_PITWALL_DATA_ITERATION_1.md) are NOT
re-reported; this gate hunts what SURVIVED and what the sprint's own fixes BROKE.

---

## Checklist — the lettered claims

| # | Claim | Verdict |
|---|-------|---------|
| A | `_tyre_stops`: 36 vs true 31, residual = {ALB,BEA,LAW,OCO,STR} = #988 artefact | **VERIFIED** — 36 / 82 / 31 and the residual five all reproduce |
| B | `fineFormFits` cannot oscillate across a continuous resize | **VERIFIED** — one boundary each way, stable when held |
| C | `useFitsRanked` cannot oscillate; BESTS fits at both settled sizes + between + after resize | **MIXED** — no oscillation, fits both settled sizes, **clips in between (X1)** |
| D | `neutralisedLaps` & pace rail agree with parquet, both directions; two panels cannot disagree | **VERIFIED** — set-equal both ways; the two panels share one reducer |
| E | Band `from − 0.5` outside the axis (lap 0.5) cannot break the chart | **VERIFIED** — and `to + 0.5` overruns too; both clipped, no error |
| F | TraceChart series-order swap broke nothing index-keyed | **VERIFIED** — nothing index-keyed anywhere |
| G | `formatSeconds` byte-equivalent to the three old formatters except the two boundary bugs | **VERIFIED** — 0 diffs over 2,080 served values; both bugs fixed |
| H | Frozen treatment cannot latch after producer returns; `frozen` false before first tick | **VERIFIED** — against a real kill AND a real restart |
| I | Swatch costs the tower no width, both sizes, colourless car | **VERIFIED** — 4 px, both sizes, colourless car included |
| J | `track_status.py` move broke no importer | **VERIFIED** — every importer, 225 tests + 192 smoke checks |
| K | `session_data` → `src.arcade.track_status` drags no pyglet into host/tests | **VERIFIED BY IMPORT** — no arcade/pyglet in `sys.modules` |
| L | `neutralised` on the wire: `allow_nan=False`, bulk revisions, hand-written fixtures all fine | **VERIFIED** — serialisation, revisions, fixtures |

---

## Findings


## A — `_tyre_stops`: VERIFIED (arithmetic, population, and residual all reproduce)

Executed: `SessionLaps.load(data, 2025, "Melbourne")` + full-reveal `masked_view`, plus an
independent transition census on the repaired frame.

- Served total **36**; old in-lap rule **82**; compound-change-only census **31** — all three match
  the claim exactly.
- Residual five = ALB L2→3, STR L2→3, LAW L3→4, BEA L4→5, OCO L4→5, all `AGE n→1` with compound
  unchanged (INTERMEDIATE), all on safety-car pit-lane transit laps. Matches #988's list by name.
- **Artefact, not real stops, proven two ways:** (1) pit-lane transit durations for those passes are
  13.0–22.3 s across ALL 17 runners (queue speed; STR's real lap-33 stop reads 18.8 s in #988's own
  measurement — durations do not separate, but positions do); (2) **none of the five changed
  position** across laps 1–7 (ALB P7, STR P11, LAW P16, BEA P17, OCO P15, constant). A real stop
  while the field queues through the pit lane loses places to every car behind; none were lost.
- True 31 confirmed independently: every real Melbourne stop was a compound change
  (I→slick→I); the 31 compound-change transitions are the 31 stops. The rule's deliberate blind spot
  (same-compound refit with a HIGHER used age) has no instance in this race.
- Edge attacks that did NOT break it: ALO retires with a generated final row (lap 33) → 0 stops
  beside a 32-lap tyre ✓; all 6 `FastF1Generated` rows sit at sequence ENDS or are the only row
  (SAI/HAD/DOO lap 1), so drop-before-pairing has no mid-sequence case to corrupt; a final-lap
  pit-in pairs with no successor and counts nothing; `is_real_compound` rejects
  `""/nan/none/unknown` and Melbourne's only unknown compounds are on generated rows, already
  dropped.
- Nuance, not a finding: for ALB/STR/BEA/OCO the age restart PERSISTS (ALB reads `I 31` at lap 33
  where truth is 33), so the TYRE column under-reads age for those cars all race. That is #988's
  data defect, already filed with the correct five-car list, and cannot be fixed in `_tyre_stops`
  without inventing stops.


## B — `fineFormFits` cannot oscillate: VERIFIED

Executed with `scripts/_exit_probe.mjs` on the live bundle: a continuous width sweep 1485 -> 1265 -> 1485
in 10 px steps, then a hold at the boundary with 10 samples 150 ms apart.

- Trail: fine down to 1445, coarse from 1435 down; on the way back coarse to 1435, fine from 1445.
  Same boundary in both directions, so no hysteresis and no flip-flop.
- Held at 1434 / 1435 / 1436 px: the state is a single value at each (`[true]`), never alternating.
- The mechanism is sound for the reason claimed: the ruler measures the FINE form regardless of what
  the cell currently renders, so the decision does not depend on its own output.

## C — `useFitsRanked` cannot oscillate: VERIFIED. BESTS fits at both settled sizes: VERIFIED.
## C — but it does NOT fit at intermediate client heights → **REFUTED for the third size** (finding X1)

- Continuous resize 1485x833 -> 1265x593 -> back: one transition each way at the SAME boundary
  (ranked at 1359x696, leaders at 1344x679, identically on the return). No oscillation.
- Both settled sizes are deterministic over 6 fresh mounts each: 1485x833 always ranked and unclipped,
  1265x593 always leaders and unclipped.
- **A fresh mount at an intermediate height silently clips.** See X1.

## D — `neutralisedLaps` and the pace rail agree with the parquet, both directions: VERIFIED

Measured on the live page against `/api/bulk` fetched inside the same document:
- Bulk rows say neutralised: laps 1-7, 33, 34, 35, 36 (11 laps at that reveal, all `SAFETY CAR`).
- Rendered rails: laps 1-7, 33, 34, 35, 36 — set-equal, and every `title` is `SAFETY CAR`.
  No railed lap the bulk does not mark, no marked lap without a rail.
- The two panels cannot disagree: both call `neutralisedLaps(bulk)` from `lib/neutralised.ts`, and the
  trace's bands on the same document resolved to the same laps (1-7 and 33-36 as two ranges).

## E — the band's `from − 0.5` cannot break the chart: VERIFIED (and `to + 0.5` overruns too, harmlessly)

- Live at a 36-lap reveal: axis extent `[1, 36]`, band 1 is `0.5 → 7.5` and band 2 `32.5 → 36.5`. So BOTH
  ends fall outside the locked axis, not just the low one — the claim under-states its own case.
- The axis is unaffected — `min`/`max` are locked to the data, extent stayed `[1, 36]` — and the canvas
  renders with **zero page errors** across every probe run.
- Sampled the canvas' own pixels: inside the band the pixel carries the amber (`[242,153,13]` before
  alpha), 8 px and 40 px past the axis end it is `[0,0,0]`/transparent, and likewise before the axis
  start. ECharts clips the `markArea` to the grid, so the overrun paints nothing outside the plot.

## F — the `TraceChart` series-order swap broke nothing index-keyed: VERIFIED

Every consumer, checked:
- Repo-wide grep for `series[` in `src/pitwall/ui/scripts`, `src/pitwall/ui/src` and `tests/surfaces`:
  **no index-keyed series access anywhere.**
- `smoke-data.mjs` reads series BY NAME (`seriesByName`, `.find((s) => s.name === …)`) and asserts the
  order as a string (`"rival>main"` on all four charts) — an intentional order assertion, not an
  accidental index dependency. `settle.mjs` only collects `__pitwallChart` handles.
  `shot-data.mjs` touches no series. `test_pitwall_tokens.py` parses source text, not indices.
- `useEChart` is order-agnostic (`setOption` with `notMerge: true`).
- The two reference marks (`yAxis: 0`, the cursor) hang off the FIRST series, which the swap changed from
  main to rival; both are `silent` and geometric, and the rival series is declared on every chart even
  when it holds no data, so they still render — confirmed on the live capture, where the delta chart's
  zero baseline and the cursor are both present.

## G — `formatSeconds` is byte-equivalent except the two boundary bugs: VERIFIED

Harness ran the retired implementations (lifted verbatim from `f580922`) against the new one over
**2,080 distinct values pulled from the live `/api/bulk`** — every `lap_time`, `s1`, `s2`, `s3`,
per-driver best and theoretical the window renders:

    diffs on the served distribution: tower 0 · paceFine 0 · paceCoarse 0

Boundaries, old -> new: `59.9996` old `60.000` / new `1:00.000` (the sub-minute bug, fixed);
`119.9996` old `1:60.000` / new `2:00.000` (the documented bug, fixed); `60`, `60.0004`, `0.5`,
`0.0004`, `9.9996`, `599.95`, `600`, `600.04`, `149.413`, `89.96` all identical between old and new.
`600 -> "10:00.0"` reproduces the docstring's seven-glyph case exactly.

Degenerate inputs are NOT equivalent and are ugly in the new one (`NaN -> "NaN:000NaN"`,
`-5 -> "-1:55.000"`, `-0.04 -> "-1:59.960"` at 3 dp but `"0:00.0"` at 1 dp). No caller can reach them —
every call site is guarded by an explicit `null` check and the wire emits `None`, never NaN
(`allow_nan=False`) — so this is recorded as X4 (P3), not as a live defect.

## H — the frozen treatment cannot latch, and cannot precede the first tick: VERIFIED

Driven on the live page by intercepting `/api/tick` and `/api/connection` to kill the producer and then
BRING IT BACK:

| state | frozen chip | `.data-main.is-frozen` | PLAYBACK | status bar | connection |
|---|---|---|---|---|---|
| live | no | no | `2x` | `lap 37 · live` | Connected |
| dead | **yes** | **yes** | `—` | `DATA FROZEN · last tick lap 37` | Disconnected |
| revived | no | no | `2x` | `lap 37 · live` | Connected |

The treatment came off completely on reconnect (all five tells reverted; 20 tower rows throughout).
With the tick held at `null` from the start and the connection reporting `Disconnected`, the window shows
the WAITING state, no frozen chip, and `Waiting for arcade stream…` — `frozen` is false before the first
tick, as `tick !== null` requires.

## I — the swatch costs the tower no width: VERIFIED, colourless car included

- 1485x833: 20 rows, 20 swatches, all 20 with an inline `rgb()`; the swatch's right edge is at **4 px**
  inside an 8 px cell padding; `tower-table` and `.tower` clip 0 px; 0 of 20 `td.col-drv` clipped.
- 1265x593: identical numbers (maxRight 4, 0 clipped, 0 table clip).
- A car the wire has no colour for: intercepted the tick and deleted `driver_colors.RUS`. The swatch fell
  back to `var(--qt-border)` → painted `rgb(45,45,58)`, still 4 px wide, still 0 clipped. The bright-white
  fallback the comment says it avoids is genuinely avoided.

## J — the `track_status.py` move broke no importer: VERIFIED

Every caller in the repo, tests included:
- `src/arcade/app.py:56-63` imports `track_status_label` from `src.arcade.overlays` — still resolves via
  the `# noqa: F401` re-export at `src/arcade/overlays.py:49-52`.
- `src/arcade/overlays.py:685` uses `track_status_banner` — same re-export.
- `tests/surfaces/test_arcade_wire_contract.py:841,862` import from `src.arcade.overlays` — pass.
- `src/pitwall/session_data.py:46` imports from the new module directly.
- `src/arcade/__init__.py` re-exports nothing; `overlays.py` declares no `__all__` at all, so no
  star-import surface was narrowed.
- `tests/surfaces/` **225 passed**, `node scripts/smoke-data.mjs` **173 checks OK**,
  `smoke-agents.mjs` **19 checks OK**.

## K — `session_data` drags no GUI library in: VERIFIED BY IMPORT

    $ .venv/Scripts/python.exe -c "import src.pitwall.session_data; …"
    GUI modules loaded: NONE
    heavy modules loaded: NONE

`src.arcade.track_status` imports only `src.arcade.config` (json / logging / os / pathlib / typing plus
`src.f1_strat_manager.data_cache`) — no `arcade`, no `pyglet`, no fastf1 / torch / lightgbm.

## L — the `neutralised` field breaks nothing on the wire: VERIFIED

- Serialisation: the value is `str | None` by construction (`neutralised_label` returns a label from a
  fixed set or `None`), so it cannot be a NaN and `allow_nan=False` is untouched. The live `/api/bulk`
  serves it on every row.
- Bulk revisions: the signature is `(year, location, reveal_map)` (`src/pitwall/host.py:216`) and is
  unchanged by a new row field, so the field cannot make the revision move or stall. Live `rev` advanced
  normally.
- Fixtures: every hand-written lap row in `smoke-data.mjs` carries `neutralised` (3 of 3 object literals
  containing `pit_in` include it — lines 885, 1386, 1451). The Python fixtures build rows through
  `_lap_row`, so they inherit it.

---

# Findings

## X1 · P1 — BESTS silently clips at every intermediate client height, THEORETICAL included; the fix removed the defect at the two measured sizes and left the band between them

**Where:** `src/pitwall/ui/src/features/data/BestsPanel.tsx:60-82` (`useFitsRanked`) ·
`src/pitwall/ui/src/styles/data.css:376-390` (`.bests { max-height: 100%; overflow-y: auto }`).

**What a strategist loses:** the THEORETICAL lap — by the panel's own docstring "the one value a wall
reads off this panel that no other panel carries" — plus the rank-3 rows, cut off with **no tell of any
kind**: the card is capped by `max-height: 100%`, the overflow becomes a scroll inside the card, and
scrollbars are hidden globally. This is the same pixel-level failure as design-gate finding D5, at a
different client height.

**Executed evidence** (`scripts/_exit_probe3.mjs`, `_exit_probe4.mjs`, real bundle, real payload;
six fresh page loads per size, 3 s settle, no request interception):

| client | outcome over 6 fresh mounts | hidden | THEORETICAL |
|---|---|---|---|
| 1485x833 (settled) | `ranked` x6 | 0 px | visible |
| 1265x593 (settled) | `leaders` x6 | 0 px | visible |
| **1265x650** | `leaders` x2, **`ranked`+33 px clipped x4** | 33 px | **cut, 22 px below the card** |
| **1350x660** | `leaders` x3, **`ranked`+23 px clipped x3** | 23 px | **cut, 12 px below** |
| **1350x673** | `leaders` x3, **`ranked`+10 px clipped x3** | 10 px | just fits; a section row is cut |

A screenshot of the clipped state at 1265x650 shows the four sections with three rows each and **no
THEORETICAL line on the card at all**.

**Cause, verified separately from the effect.** `useFitsRanked` latches the ranked panel's natural
height the first time `fit()` runs, and `fit()` runs only on mount and from a `ResizeObserver` on the
**column**. The card mounts as soon as the first TICK lands, but its content depends on the **BULK**,
which arrives on its own poll:

- With `/api/bulk` held at `null` (the panel's empty state), the ranked card measures **114 px** — and
  the room at 1265x650 is **120 px**, so `room < needed` is false and the panel commits to `ranked`.
- The populated ranked card is **151 px**. Nothing re-measures it: `.left-column` is a grid whose row
  heights do not change when the bests data arrives, so the observer never fires. Forcing a 1 px
  viewport *width* change did not re-decide it either (measured: still `ranked`, still 33 px hidden).
- Holding the bulk at `null` for 4 s and then releasing it reproduces the clip **deterministically**
  (`B.delayedBulk 1265x650`: `ranked`, 33 px hidden, THEORETICAL 22 px below the card). That is the
  race the six-mount table samples.

The two settled sizes cannot show it, which is why the sprint's own measurement and the smoke's guard
both pass: at 1265x593 the room (63 px) is below even the EMPTY panel's height, so it degrades whatever
happens; at 1485x833 the room (303 px) is above the populated height. **The defect lives exactly between
the two numbers anybody measured.** The band is `room ∈ [~115, ~151)`, and the client height is a
continuous function of the screen (`WindowSpec.place` clamps to `screen_height − 90`), so ordinary
machines land in it — a 1366x768 laptop at 100 % scaling gives roughly 1350x641-ish, and 1600x900 or
1920x1200 at 150 % land inside the band.

**Prescription:** re-run `fit()` when the CONTENT changes, not only when the column does — e.g. add the
bests' own identity to the effect (`bulk?.rev`, or the count of populated entries) so the latched
`rankedHeight` is the populated height. `RacePaceGrid`'s sibling `fit` already does the analogous thing
(`useEffect(…, [fit, grid.columns.length])`), which is why claim B has no equivalent race — the
asymmetry between the two is the whole bug. Then extend the smoke's `bests.hidden === 0` assertion,
which is correctly written as an EFFECT, to a **third client inside the band** (1350x660 is a
one-line addition and fails today).

## X2 · P2 — The frozen treatment desaturates the one channel that carries the pace ranking, and the comment promises the opposite

**Where:** `src/pitwall/ui/src/styles/data.css:124-126` (`.data-main.is-frozen { filter:
saturate(0.45) brightness(0.82) }`), interacting with `lib/racePace.ts`'s tone scale and the
neutralised rail this same sprint added.

**What a strategist loses:** on the frozen board — the state in which a wall re-reads the last known
picture — the RACE PACE grid's colour code loses most of its separation. `racePace.ts` states that
"the tone still carries the ranking, which is where this panel puts the ordering anyway", and the CSS
comment promises the treatment "has to say 'this is history' while leaving every number readable".
The numbers stay readable; the ranking does not.

**Executed evidence:** sampled the same cell coordinates in two real screenshots of the same page,
live and frozen (`exit-pace-live.png` / `exit-pace-frozen.png`, 1485x833), 7x3-pixel patch average:

| pair | live RGB distance | frozen | loss |
|---|---|---|---|
| quickest third / slowest third (`t1`/`t3`) | 113.0 | 38.3 | **−66 %** |
| `t1` / out-lap | 104.2 | 35.1 | −66 % |
| session best (purple) / deleted | 39.3 | 14.3 | **−64 %** |
| `t1` / session best | 106.0 | 53.2 | −50 % |
| in-lap / out-lap | 53.8 | 26.4 | −51 % |
| neutralised rail (amber) | — | shifts 27.8 | rail dims with everything else |

Band 1 and the status bar are deliberately exempt from the filter and do carry the explanation, so this
is a cost, not a lie — but it is an unstated cost on the panel whose colour IS its content, and it was
introduced by #982 over markers #980 had just added.

**Prescription:** exempt the tone-bearing surfaces from the filter (scope it to the tower and band 4,
or apply it to text colour rather than the whole board), or drop `saturate()` and carry "this is
history" with `brightness()` plus the chip alone — the chip, the `PLAYBACK —`, the neutral track chip
and the status line are already four independent tells. If the filter stays as-is, the CSS comment
should say which channel it spends.

## X3 · P2 — The AGENTS window is the twin that never got #982: a dead producer still reads `PIT NOW · Confidence: 71% · 2.00× · PLAYING`

**Where:** `src/pitwall/ui/src/features/agents/AgentsWindow.tsx` (no frozen state at all) ·
`src/pitwall/agents_view/panels.py:146-158` (`build_status_bar`, `transient: True`) — against
`features/data/DataWindow.tsx:76` (`const frozen = …`).

**What a strategist loses:** the DATA window's own justification for #982 is *"a dead producer left a
full board of confident values, the lap counter still saying `L 28/57`, the track chip still asserting
GREEN and `PLAYBACK 2x` still claiming the replay was advancing, with a 77 x 18 chip and a blank status
bar as the only tells"*. That sentence is a verbatim description of the AGENTS window today, and its
frozen content is a **recommendation** rather than a readout.

**Executed evidence — the real path, not a mock.** I killed the producer process that owned
`127.0.0.1:9998` (PID 27508; confirmed `port 9998: NO LISTENER`), left the host untouched, and loaded
both pages. `/api/connection` returned `"Disconnected"`.

| | DATA (fixed) | AGENTS (twin) |
|---|---|---|
| frozen chip | `DATA FROZEN`, filled DANGER | none |
| board treatment | desaturated + dimmed | none (`filter` count: 0, no dimmed container) |
| playback field | `PLAYBACK —` | **`2.00× · PLAYING`** |
| lap counter | `L 44/57` under the frozen chip | `L 44/57`, unmarked |
| decision surface | n/a | **`PIT NOW`, `Confidence: 71%`, `Pace: PUSH`, `Risk: AGGRESSIVE`, `Pit: L24 · Next: HARD · UCUT: RUS`** at full strength |
| connection chip | red `Disconnected` | red `Disconnected` |
| status bar | `DATA FROZEN · last tick lap 44` (non-transient) | **blank** (transient, cleared after 1.5 s) |

The screenshot shows a full-size scarlet `PIT NOW` with a 71 % confidence bar and a reasoning panel
narrating in the present tense, beside a 77-px `Disconnected` chip.

**A correction to my own first measurement, stated plainly:** my initial probe faked the dead feed by
returning `null` from `/api/agents`, which simulates a dead HOST, not a dead producer, and it made the
connection chip appear to freeze at `Connected`. That was wrong — the host re-serves the view on a
connection change (`host.py:151` explains exactly this), and against a genuinely killed producer the
chip does go red. The finding survives the correction; the chip does not.

**Prescription:** the same `frozen` derivation the DATA window now has (`connection === "Disconnected"
&& view.seq !== null`) plus the two things that carry it — a `DATA FROZEN`-equivalent chip and a
non-transient status line — and `playback` must stop asserting `PLAYING`. Because the payload is built
host-side, the honest place for the playback string is `build_header`. This belongs on #976 or its own
issue; it is not sprint 9's scope, but it IS the asymmetry sprint 9 created.

## X4 · P2 — The mixed-lap sentence in `neutralised.ts` is measured on the raw digit strings, not on the label the code uses; and the clause that justifies the whole rule is false

**Where:** `src/pitwall/ui/src/lib/neutralised.ts:22-30` (the `neutralisedLaps` docstring).

**The claim:** *"On the real race the difference is three laps — 33, 34 and 47 carry mixed statuses —
and on those the SC digit is on the majority of rows anyway."*

**Measured on Melbourne 2025 through the code's own decode path** (`neutralised_label` over
non-generated rows, which is exactly what `neutralisedLaps` walks):

    mixed laps: [(33, 16 rows, 13 marked), (46, 15 rows, 3 marked)]

- **Two laps, not three**, and they are **33 and 46**, not 33/34/47.
- On lap 46 the SC digit is on **3 of 15 rows (20 %)** — so "on those the SC digit is on the majority of
  rows anyway" is **false on the worse of the two cases**, and it is the sentence that justifies the
  "ANY row, not a majority of them" rule.
- Laps 34 and 47 are not mixed at all under the decoded rule: every row decodes to `SAFETY CAR`
  (16/16 and 14/14).

**Cause, verified.** Measuring mixedness on the **raw `TrackStatus` string** instead of the decoded
label yields `[32, 33, 34, 45, 46, 47]`, because `'124'`, `'24'` and `'4'` are three different strings
that all decode to `SAFETY CAR`. The quoted trio is a subset of that list. So the figure describes a
population the sentence is not about — the same class the sprint already caught twice in its own work,
and the same class as the retired "STR alone" figure in `_tyre_stops`.

**The design is right and the pixels are right** — the conservative direction is more necessary than the
comment claims, not less. The danger is precisely that a future reader, told the majority always agrees,
"simplifies" the rule to a majority vote and silently drops lap 46's rail while three cars queue through
the pit lane.

**Prescription:** replace the sentence with the measured one — two mixed laps, 33 (13/16) and 46 (3/15),
and say that lap 46 is the case that makes ANY-row the right rule rather than an incidental one.

## X5 · P3 — `formatSeconds` renders garbage for NaN and negatives, and it is now the single arithmetic for three surfaces

**Where:** `src/pitwall/ui/src/lib/format.ts:37-46`.

`formatSeconds(NaN, 3)` -> `"NaN:000NaN"`; `formatSeconds(-5, 3)` -> `"-1:55.000"`;
`formatSeconds(-0.04, 3)` -> `"-1:59.960"` while `formatSeconds(-0.04, 1, true)` -> `"0:00.0"`.
Unreachable today — every one of the four call sites guards on `null` first and the wire cannot emit NaN
(`allow_nan=False`) — but consolidating three formatters into one makes it the function the NEXT surface
calls, and a negative delta is a plausible future argument. **Prescription:** return `"—"` for a
non-finite or negative input, or document in the docstring that the contract is a finite non-negative
number and let the caller guard.

## X6 · P3 — "15.8:1" is 17.48:1

**Where:** `src/pitwall/ui/src/features/data/TimingTower.tsx:133`.

The tower has no row striping (`grep` over `data.css`: the only `nth-child(even)` rule belongs to
`.pace-table`), so `td.col-drv` sits on `--qt-panel` `#181633`, and `--qt-fg-1` is `#ffffff`:
**17.48:1**, not 15.8:1. (15.8 is close to white-on-`--qt-elevated`, 16.0, which is the pace grid's
banding, not the tower's ground.) Conservative direction — the fix delivers more than it claims — but
every other contrast figure in the same comment reproduced exactly (VER/LAW **1.88**, ALO/STR **2.55**,
HAM/LEC **3.71**, six below 4.5, four below 3.0, ten distinct colours over twenty cars), which is what
makes the one loose number worth correcting.

## X7 · P3 — `fineFormFits` measures the BOLD header cell to decide the regular-weight body cells

**Where:** `src/pitwall/ui/src/features/data/RacePaceGrid.tsx:145` passes
`thead th + th`; `fineFormFits` then reads `getComputedStyle(cell)`.

The whole point of the helper is that it measures "a cell's OWN computed font" rather than a tuned
constant. The cell it measures is a header (`font-weight: 700`); the cells that must fit the label are
body cells (`400`). Measured live: `700 9px` and `400 9px` both give `0:00.0` a width of **33.23 px**, so
there is no difference on this machine and no defect today; with a fallback whose bold is wider the
guard coarsens slightly EARLY, which is the safe direction. **Prescription:** measure a
`tbody td` (falling back to the header before the first reveal) so the measured font is the rendered
one, and say in the comment that the header is a proxy if it stays.

## X8 · P3 — the fine-form ruler cannot see the one label that would overflow

**Where:** `RacePaceGrid.tsx:100-104` (`WIDEST_FINE_LABEL = "0:00.0"`).

`paceLabel`'s own docstring says the fine form "runs to seven characters from ten minutes upward
(`600 -> "10:00.0"`)" — and I reproduced that exactly in the G harness. The ruler tests six glyphs, so
the fallback that exists to prevent silent truncation is blind to the only case that truncates at the
wide client. The constant's comment scopes itself honestly ("under ten minutes"), so this is a gap, not
a false claim; no downloadable race reaches it (Melbourne's slowest ranked lap is 149.4 s), and a
red-flagged race would. **Prescription:** measure `"10:00.0"`, or measure the widest label the grid
actually built this render.

## X9 · P3 — a window that OPENS onto a dead feed shows four empty telemetry plots with no caption

**Where:** `features/data/OwnCarTraces.tsx` / `TraceChart.tsx`, reachable only via the state #982 made
legible.

With the producer killed and the page loaded fresh, the tower, bests, ring and radio feed are fully
populated from the host's last payload, while band 4's four plots render axes, a zero baseline and a
cursor with **no traces and no explanation** — the buffers accumulate per tick and only one tick was
ever served. The window elsewhere always says why a panel is empty (`data-waiting`,
`trace-band-empty`, `unavailable`, the `placeholder` prop). `DATA FROZEN` does explain the board as a
whole, which is why this is P3 rather than higher. **Prescription:** when `frozen` and a trace has
fewer than two samples, show the existing `placeholder` with "no telemetry since the feed stopped".

## X10 · P3 — the STOPS column applies the compound-sentinel rule; the TYRE cell beside it does not

**Where:** `src/pitwall/session_data.py:259` uses `is_real_compound`, while
`features/data/TimingTower.tsx:252-256` (`tyreCell`) tests only `if (!last?.compound)` and then prints
`compound[0]`, and `BestsPanel`'s `entry.compound[0]` does the same.

`_COMPOUND_SENTINELS` is `{"", "nan", "none", "unknown"}` and is not speculative — it exists because
those strings have been seen. A stringified sentinel that survives `_none_if_nan` (which only catches
float NaN) would print **`n`** or **`u`** in the TYRE column as if it were a compound letter. **I could
not demonstrate a wrong pixel:** Melbourne 2025's `Compound` holds only `INTERMEDIATE`, `HARD`,
`MEDIUM`, so no instance exists on the one race a curated install carries. Recorded because the module
that made the rule public says a stringified missing compound "must never read as evidence" — and the
consumer next to the one that got the rule did not get it. **Prescription:** filter the compound through
the same rule in `_lap_row` (publish `None` for a sentinel), so no TypeScript consumer needs the rule.

---

# Fix list, ordered by value over risk

1. **X1 — re-measure the ranked BESTS height when its CONTENT changes** (add `bulk?.rev` or the
   populated-entry count to `useFitsRanked`'s effect), and extend the smoke's existing
   `bests.hidden === 0` assertion to **one client inside the band (1350x660)**. Two small edits; the
   guard is already written as an effect, so it will go red against today's code. Highest value: it is
   the sprint's own P0 class surviving at ordinary screen sizes.
2. **X4 — correct the mixed-lap sentence in `neutralised.ts`** to the measured pair (33 at 13/16, 46 at
   3/15) and make lap 46 the reason the ANY-row rule is right. Zero risk, and it removes the invitation
   to replace the rule with a majority vote.
3. **X2 — scope `.data-main.is-frozen` off the tone-bearing panels** (or drop `saturate()` and keep
   `brightness()`), then re-shoot the frozen pace tab and look at it. Small CSS change, needs a visual
   check rather than a test.
4. **X10 — publish `None` for a sentinel compound in `_lap_row`**, so no TypeScript consumer needs the
   rule the stop count already applies. Three lines, removes a latent class rather than a live bug.
5. **X3 — give the AGENTS window the frozen treatment**, including a `playback` that stops asserting
   `PLAYING` (host-side, in `build_header`). Bigger, and it belongs to the AGENTS window's own issue —
   but it is the asymmetry this sprint created and the frozen content there is a decision.
6. **X6 / X7 / X8 / X5 / X9** — the numeric correction, the body-cell measurement, the seven-glyph
   ruler, a non-finite guard on `formatSeconds`, and a frozen placeholder for band 4. All small; none
   changes a pixel today.

---

# What I tried to break and could NOT

- **`_tyre_stops` (claim A), attacked on arithmetic AND population.** 36 served, 82 in-laps, 31
  compound-change transitions — all three reproduce. The residual five are exactly ALB, BEA, LAW, OCO,
  STR, all `AGE n→1` with the compound unchanged on a safety-car transit lap, and **none of the five
  changed position across laps 1-7**, which is the independent evidence that no work was done. ALO, who
  retired on his starting set with a generated final row, reads 0 stops beside `I 32`. A final-lap
  pit-in pairs with no successor and counts nothing. All six `FastF1Generated` rows sit at sequence ends
  or are a car's only row, so dropping them before pairing cannot hide a change; structurally, a
  generated row between two real ones would still let the change count. No compound sentinel exists in
  this race and `is_real_compound` rejects all four anyway. I could not make the rule invent a stop.
- **Oscillation, in both new fit hooks.** A continuous 22-step resize in each direction, plus a
  10-sample hold at the boundary, produced one transition at one boundary for each hook and no
  alternation anywhere. The pace grid's ruler measures the fine form regardless of what is rendered;
  the bests panel latches rather than flips. (The bests panel's defect is the latch, not a flip.)
- **The band's out-of-axis edges.** Both ends fall outside the locked axis at a live reveal
  (`0.5 → 7.5` and `32.5 → 36.5` against an extent of `[1, 36]`); the extent did not move, no error was
  raised in any of ~20 page loads, and canvas pixel sampling shows nothing painted beyond the plot area.
- **Anything index-keyed on the series order.** No `series[` index access exists in the scripts, the
  app or the tests; the smoke reads by name and asserts the order as a string. The reference marks that
  hang off the first series still render after the swap.
- **`formatSeconds` on the served distribution.** 2,080 distinct values off the live bulk, zero
  divergence from the three retired formatters, and both documented boundary bugs demonstrably fixed.
- **The frozen state, against a REAL killed producer and a REAL restart.** DATA showed all five tells;
  when I restarted the producer the host reconnected on its own and `/api/connection` returned
  `"Connected"` with no intervention — the treatment cannot latch. It also cannot precede the first
  tick. And the `filter` did NOT break the pace grid's sticky header (`position: sticky`, top unchanged
  at 249 px) or any ECharts canvas.
- **The driver swatch.** Zero table width at both clients (right edge at 4 px inside 8 px of padding),
  zero clipped cells, and a car whose colour I deleted from the wire fell back to `--qt-border`, not to
  a bright white bar.
- **The `track_status.py` move.** Every importer in the repo still resolves; `tests/surfaces` 225
  passed, `smoke-data` 173 checks, `smoke-agents` 19 checks; and `import src.pitwall.session_data`
  loads **no** `arcade`/`pyglet` module, proven by inspecting `sys.modules` rather than by reading.
- **The `neutralised` field on the wire.** Cannot be NaN by construction, cannot move the bulk
  revision (the signature does not include row fields), and all three hand-written row fixtures carry
  it.
- **The reveal, forwards and backwards.** Over all 58 reveal values, the stop total and the neutralised
  mark set are monotonic in the reveal and the recomputation is deterministic — so a rewind strictly
  un-reveals both. No mark or stop appears and later vanishes.
- **The compound-sentinel rule, on the stops path.** One definition, one implementation; no consumer of
  the PITWALL/repair path re-encodes it. (A blanket "no second copy anywhere in Python" stood here for
  one revision of this report and was WRONG: a late background sweep found
  `src/agents/race_state_builder.py:75` carrying `_MISSING_COMPOUND_MARKERS = {"", "nan", "none"}` — a
  sibling set that PRE-DATES this sprint (present at `3492934`), lives in the agents subsystem, and is
  deliberate on both ends: that module folds missing compounds into its own `UNKNOWN_COMPOUND` output
  marker, which is exactly why its set omits `"unknown"` while `_COMPOUND_SENTINELS` includes it. The
  two compose rather than conflict — an `"UNKNOWN"` emitted there IS caught by `is_real_compound` here —
  but they are two hand-kept lists of the same feed strings, unguarded against drifting apart, and
  `is_real_compound`'s own docstring implies no sibling exists. Pre-existing and out of this sprint's
  scope, so recorded as a note, not a finding.)
- **Every other number the sprint's new comments assert.** All reproduced exactly on the served payload:
  22 of 57 laps neutralised; ranges 1-7, 33-41, 46-51; **213 of 776** ranked cells (27.4 %); neutralised
  lap times 86.4-148.2 s, median **131.6** against a green median of **91.9**; median delta **+13.03 %**
  and **80.7 %** past +10 %; the alpha crossover at **0.1034** with 0.12 → `rgb(51,38,46)` and 0.08 →
  `rgb(42,33,48)` over `--qt-panel` `#181633`; `AXIS_TEXT` at **11.86:1**; the six driver colours below
  AA (VER/LAW 1.88, ALO/STR 2.55, HAM/LEC 3.71) and the four below 3.0; ten distinct colours over twenty
  cars; and both control strips computing to **26 px** at both clients with JetBrains Mono absent.

---

# Claims of the author's that I refuted, plainly

1. **"BESTS degrades to a one-line leaders form at a short client" is only true at the two client sizes
   anybody measured.** Between them there is a band of ordinary screen heights where the panel commits
   to the ranked form and silently loses up to 33 px including THEORETICAL — the same defect D5
   described, at a different height, 3-4 times out of 6 fresh page loads (X1). The commit message's
   framing, and the smoke's guard, are both true and both blind to it.
2. **"On the real race the difference is three laps — 33, 34 and 47 carry mixed statuses — and on those
   the SC digit is on the majority of rows anyway."** All three parts are wrong: two laps, 33 and 46,
   and on lap 46 the digit is on 3 of 15 rows. The figure was measured on the raw `TrackStatus` strings
   rather than the decoded label the code uses (X4).
3. **"The code is `--qt-fg-1` now, 15.8:1."** It is 17.48:1 on the ground the tower actually paints
   (X6).
4. **"The treatment has to say 'this is history' while leaving every number readable."** True of the
   numbers, false of the channel the RACE PACE panel puts its ordering in: quickest-third against
   slowest-third loses 66 % of its separation and purple-against-deleted 64 % (X2).
5. **A framing, not an error:** the `markArea` note worries that "band 1 asks for lap 0.5" — at a live
   reveal the HIGH end is outside the axis too (`36.5` against a max of `36`). Both are clipped and
   harmless; the comment describes half of what its own arithmetic does.

**And a second claim of my own, caught by my own late-running sweep:** an earlier revision of this
report asserted the compound-sentinel rule had "no second copy anywhere in Python". False —
`src/agents/race_state_builder.py:75` holds a pre-existing, deliberate sibling set (details in the
tried-to-break section above). The name-based grep I first ran (`is_real_compound|_COMPOUND_SENTINELS`)
could only find copies that share a NAME, which is precisely how a semantic twin hides; the sweep that
found it searched for the sentinel STRINGS instead.

**And one claim of MY OWN that I had to refute:** my first AGENTS measurement reported the connection
chip frozen at `Connected` under a dead feed. That came from faking the death by returning `null` from
`/api/agents`, which is a dead HOST, not a dead producer. Against a genuinely killed producer the chip
does go red. I re-ran it the honest way — killing the process that owned port 9998 — and X3 stands on
that run, with the chip explicitly excluded from the finding.

---

# Summary

**12 lettered claims: 11 VERIFIED, 1 MIXED (C).** Ten findings: **1 P1, 3 P2, 6 P3.** No P0.

The sprint's measurement discipline is high — every numeric claim in `track_status.py`, `racePace.ts`,
`chart.ts` and `TimingTower.tsx` reproduced exactly on the served payload, including the ones that
needed a census rather than arithmetic. Four of the eight changes I could not dent at all
(`formatSeconds`, the series swap, the module move, the swatch).

What survived and what the fixes broke:

- **X1 (P1)** — the BESTS degradation removes the silent clip at the two measured client sizes and
  leaves it in the band between them, 3-4 times out of 6 fresh page loads, THEORETICAL cut. The cause is
  a height latched before the bulk arrives, with no re-measure trigger.
- **X2 (P2)** — #982's `saturate(0.45)` costs the RACE PACE grid up to 66 % of the separation between its
  tone classes, on the board a strategist studies precisely because it is frozen — an interaction between
  two of this sprint's own changes.
- **X3 (P2)** — the AGENTS window never got #982; a genuinely killed producer leaves `PIT NOW ·
  Confidence: 71% · 2.00× · PLAYING` at full strength, with the exact pair of tells #982 declared
  insufficient.
- **X4 (P2)** — the mixed-lap sentence that justifies `neutralisedLaps`' ANY-row rule is measured on raw
  digit strings, and its load-bearing clause is false on the case that matters.

---

## Probe files (untracked, mine; delete before any PR)

`src/pitwall/ui/scripts/_exit_probe.mjs` (B, C, D, E, F, I, H) · `_exit_probe2.mjs` (the BESTS latch,
the colourless car, the band's canvas clip) · `_exit_probe3.mjs` (the latch's CAUSE, a fleet-height
sweep) · `_exit_probe4.mjs` (six fresh mounts per size) · `_exit_probe5.mjs` (control targets, the
frozen pace grid, the first — mistaken — AGENTS run) · `_exit_probe6.mjs` (both windows against a
genuinely dead producer). Python censuses and the `formatSeconds` differ harness ran from the
scratchpad and are not in the repo.

**Environment at gate end:** the producer was killed for X3 and a fresh one restarted
(`scripts/dev_pitwall_producer.py 3600`, broadcasting until its window expires); the gate's own headless
host at `127.0.0.1:58476` reconnected to it on its own (`/api/connection` read `"Connected"`, which is
also the executed proof of H's reconnect path) and was then torn down when the gate finished — neither
it nor the handed-over `57594` host is running now. To re-open the window: start the producer if it has
expired, then `python -m src.pitwall` (or a headless `PitwallHost` + `BrowserServer`). No repository
file was modified except this report; the six `_exit_probe*.mjs` files are untracked.
