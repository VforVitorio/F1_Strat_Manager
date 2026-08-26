# PITWALL · DATA - the traces, the slack, and the radio: the space spec

**Status: SPEC, awaiting sign-off on the questions in section 8. Nothing here is built.**

Sprint 9's elevate pass shipped (#987-#996) and skipped the three items the agreed layout drawing
had explicitly assigned to it. Two of them are visible in the shipped window: an empty gap at the
bottom left of the second telemetry tab, and four charts still sized too large. The radio feed
reads as cramped alongside them. This document is the implementation spec for those three items
plus the radio squeeze, checked line by line against the drawing agreed on 2026-08-13.

## 0. Provenance - what this spec is built from

**Read (binding, in this order):**

- `memory/project_pitwall_data_layout.md` - the DATA layout drawn in ASCII and agreed
  2026-08-13, with the measured height budget. The ground truth this spec is checked against.
- `memory/reference_pitwall_real_monitors.md` - the evidence base from six photographs
  plus seven sources (2026-08-07).
- `memory/project_pitwall_window.md` - the agreed two-window shape, including the band-4 line
  quoted in section 1.
- `memory/project_pitwall_sprint5/6/7/8/9.md`, `~/.claude/plans/pitwall-sprint9/MEASURED-BASELINE.md`
  and `~/.claude/plans/pitwall-sprint7/TODO.md` (the measured deferral list).
- `documents/research/PITWALL_REALISM_AND_TELEMETRY_SURFACE.md` (section 2.1-2.2 tiers, section 3.1, section 3.5),
  `PITWALL_V2_ARCHITECTURE.md`, `PITWALL_DELIVERY_PLAN.md` (the band-4 scope note at lines
  ~195-240 and the sprint-4 status line at ~393).
- Shipped code: `src/pitwall/ui/src/features/data/OwnCarTraces.tsx`, `TraceChart.tsx`,
  `traceBuffer.ts`, `BestsPanel.tsx`, `RadioFeed.tsx`, `TrackRing.tsx`, `DataWindow.tsx`,
  `lib/chart.ts`, `lib/bridge.ts`, `styles/data.css`; producer: `src/arcade/app.py`
  (`_frame_to_telemetry`, line 162), `src/arcade/stream.py` (the additive-key contract),
  `src/arcade/track.py:40` (`_DRS_ACTIVE`), `src/arcade/palette.py`; tests:
  `tests/surfaces/test_arcade_telemetry_span.py:190` (the frozen sample key set),
  `test_arcade_wire_contract.py:261-276` (the type map), `test_pitwall_tokens.py`.
- Issues built on, not re-specced: **#986** (D8 radio chips + duplicate-RCM collapse, D14
  `compound_name` producer precedent), #936, #931.

**Measured:**

- Live geometry at both clients is INHERITED from the sprint-9 probe baseline
  (`MEASURED-BASELINE.md`, measured on the real loopback server against the real Melbourne 2025
  payload) plus fresh live numbers measured on the shipped TRACES tab (left column
  750 with 150 empty; `.traces-grid` 533 x 666, cells 262 x 328; `.radio-feed` 260 x 404, 42
  events ~10 visible; ring SVG 200). Every one of those numbers was re-derived here from the
  shipped CSS and matches to within 4 px; the derivations are shown inline in section 4.
- **Executed fresh for this spec** on `data/cache/arcade/Melbourne_2025_race.pkl` (the real
  session the producer serves): `circuit_length_m = 5219.98`; 154,173 frames x 20 drivers; NOR's
  gear distribution spans **1-8, all eight values present** (gear 1: 1,309 frames); NOR's raw
  `drs` codes are dominated by 0/1/8 with the open set `{10, 12, 14}` totalling **555 frames,
  0.4% of the race** - Melbourne 2025 was safety-car-heavy and DRS was rarely enabled, so the
  DRS lane on the only race on disk is flat almost everywhere. That is the data, not a bug, and
  Section 5.2 designs for it.
- The PITWALL window itself was NOT reopened for this spec; no pixel below claims otherwise.

**Searched:** MoTeC i2 worksheet conventions, Cosworth Pi
Toolbox, AiM Race Studio, The Field's F1 chart documentation, MultiViewer, and the sim-racing
engineering guides. Findings with URLs in section 3.

---

## 1. What the agreed drawing says - the quoted inventory

The band-4 contract, from `project_pitwall_window.md` (the agreed shape):

> **own car**: speed/throttle/brake/gear/DRS stacked on ONE x axis (distance within the lap) with
> a shared vertical cursor, pinned rival overlaid and labelled broadcast tier. **The ring** in the
> corner.

The sprint-9 assignment, from `project_pitwall_data_layout.md`, table *"What sprint 5 did NOT
build, and where each piece goes"*:

> | the traces stacked on ONE x axis with a shared cursor (today they are the Qt 2x2, which the
>   research calls the wrong shape) | 9, the elevate pass |
> | gear and DRS on the traces | 9, and needs a decoded `drs_open` from the producer |
> | the left column's slack under the bests card | 9 |

and its closing sentence on the traces:

> **The traces are not oversized; the right column is doing one job with room for two.** In
> sprint 6 it gains a tab strip and band 3 shares it (the traces and the ring hide on that tab);
> in sprint 9 the 2x2 becomes the stacked form.

The sentence "the research calls the wrong shape" resolves to
`reference_pitwall_real_monitors.md`, the section "The own-car screen", verbatim:

> Stacked traces sharing ONE x axis with a shared vertical cursor: speed / throttle / brake /
> RPM / gear / DRS. [...] **the stacking and the shared cursor are the transferable part**, and
> they are why a 2x2 grid is the wrong shape: with five channels on one axis a single cursor
> gives you all five values at the same point of track.

The drawing's own boxes (agreed 2026-08-13, at the 1485 x 833 client):

| element | drawn size (W x H) |
|---|---|
| band 1 | full width x 29 |
| TIMING TOWER | 620 x 439 |
| BESTS | **620 x 302** |
| OWN-CAR TRACES | 565 x 751 (2x2 cells 277 x 290 drawn as the sprint-5 interim) |
| RING column | 260 wide |
| RADIO / RCM | **"417 x 260"** (see section 2.1 - the pair is internally inconsistent) |
| status bar | full width x 23 |

## 2. Deviation table - agreed vs shipped, both numbers

Measured shipped values from the sprint-9 baseline + the live report; drawing values from section 1.

| # | element | agreed (drawing) | shipped (measured) | delta | verdict |
|---|---|---|---|---|---|
| 1 | left column width | 620 | 630 | +10 | **Deliberate, documented in the drawing memory itself**: the tower's natural width is 597 px and 620 landed one pixel clear of compression, so sprint 6 re-measured to 630. Not a defect. |
| 2 | tower | 620 x 439 | 630 x 437 | -2 h | Rounding on the real border box. Not a defect. |
| 3 | **BESTS** | 620 x **302** | 630 x **153** | **-149 h** | ⭐ Assigned to sprint 9, never built. The drawing already expected a taller bests panel. Section 6. |
| 4 | right column | 835 x 751 | 825 x 750 | -10 w | Consequence of #1. Not a defect. |
| 5 | tab strip | anticipated ("the tab strip is born here") but drawn at zero height | 26 px + 6 gap = **32 px** | -32 from every tab panel | Evolution the drawing priced at 0; every column-height number below is 718, not 751, because of it. |
| 6 | traces card | 565 x 751 | 555 x 718 | -10 w, -33 h | Consequence of #1 + #5. |
| 7 | **traces FORM** | "stacked on ONE x axis with a shared cursor" | **the Qt 2x2**: four separate plots, cells 262 x 328, four x-axis bands, four title rows, one shared cursor VALUE drawn four times | - | ⭐ The headline deviation. Section 5. |
| 8 | **gear + DRS** | on the stack ("speed/throttle/brake/gear/DRS") | absent, deliberately - `OwnCarTraces.tsx`'s docstring refuses to fork `{10,12,14}` into TypeScript | -2 channels | ⭐ Waiting on the producer's `drs_open`. Section 5.2. |
| 9 | radio / RCM feed | "417 x 260" | 260 x 404 | see section 2.1 | Section 7. |
| 10 | tabs | 2 anticipated (TRACES / RACE PACE) | 3 (+ RACE TRACE) | +1 | Sprint-6 decision, measured (band 3 is two panels). Not a defect. |

**One docs inconsistency found on the way**: `PITWALL_DELIVERY_PLAN.md:393` says sprint 4 shipped
*"Own-car traces stacked on one x axis with a shared vertical cursor"*. What shipped is the 2x2
with a shared cursor **value** repeated on four independent charts; the four x axes are locked to
the same range but are four axes. The plan's line over-claims and should be corrected in the PR
that builds section 5 (the drawing memory states it correctly: "today they are the Qt 2x2").

### 2.1 The radio's "417 x 260" - the pair is transposed in the drawing

The brief takes the W x H reading (417 wide, 260 tall) and calls the shipped 260 x 404 "157 px
narrower and 144 px taller than agreed". Both numbers are real, but the W x H reading breaks the
drawing's own geometry, and the transposed reading fits it exactly:

- Every other box in the drawing is quoted W x H, and the drawing places RADIO / RCM **inside the
  260-px side column, under the ring** ("The ~417 x 260 px under the ring is the home"). A
  417-wide box cannot sit in a 260-wide column; the drawing shows no restructure that would make
  room for one, and 417 is not derivable from any horizontal split of the 835-px right side
  (835 - 260 - 10 = 565, not 417).
- Read transposed - **260 wide x 417 tall** - the number falls straight out of the drawing's own
  vertical arithmetic: side column 751, minus a ~324-px ring card, minus the 10-px gap = 417.
- Shipped is 260 x 404: the tab strip took 32 px of the column (deviation #5) and the ring card
  settled at ~304, leaving 404. **Against the transposed reading the shipped feed is 13 px short,
  not 157 px narrow.**

Therefore the radio's real deviation is small and structural (the tab strip), and the cramped feel is a
DENSITY problem rather than a geometry one: 42 events with ~10 visible behind a fold,
every long RCM row spending two clamped lines, four identical BLUE FLAG rows spending four slots.
Section 7 fixes density, and section 8 Q4 offers the width restructure if the literal 417 was meant.

---

## 3. Research - how real telemetry clients draw a stacked trace

What was found, with URLs, and what it implies. Where the convention disagrees with the drawing,
it says so plainly.

1. **MoTeC i2** - the reference analysis package. Its classic compare worksheet stacks
   **SPEED, RPM, GEAR, BRAKE, THROTTLE from the top**, against distance, each channel in its own
   lane of a single time/distance graph with one cursor; the channel tag carries the value at the
   cursor. Sources:
   [Trinacria - Beginner's Guide to Telemetry Analysis (MoTeC)](https://trinacriasimracing.wordpress.com/beginners-guide-to-telemetry-analysis-motec/),
   [Coach Dave Academy - MoTeC in ACC](https://coachdaveacademy.com/tutorials/how-to-use-motec-data-in-assetto-corsa-competizione/),
   [SimRacerCentral - MoTeC practical guide](https://simracercentral.com/motec-sim-racing-telemetry-guide/).
   Implication: **speed on top**; one axis; per-lane value readout at the cursor.
2. **The Field (F1 charts documentation)** - already this project's timing-screen source. Speed
   trace is the primary channel ("the vertical axis is car speed in km/h" against "track distance
   in metres from the start/finish line"); **gear is drawn as "a step chart"**; throttle and brake
   are the 0-100 pair; the **time delta "plots the cumulative time gap between two drivers at
   every track distance point"**; and the panels are synchronised by a shared crosshair -
   verbatim: *"All charts share a synchronised crosshair - hover over one and a cursor appears on
   all others at the same track distance."*
   [thefieldf1.com/charts](https://www.thefieldf1.com/charts).
   Implication: gear = step form; delta = cumulative vs distance (which `deltaSeries` already is);
   the cursor spans all channels at one x.
3. **DRS as a binary band** - fan and analysis clients draw DRS as a band/strip marking where the
   system is open, not as a value curve:
   [F1 Live Pulse - telemetry](https://www.f1livepulse.com/en/telemetry/),
   [TracingInsights - telemetry reference](https://mintlify.wiki/TracingInsights/2026/reference/telemetry).
   Implication: the DRS lane is the thinnest lane on the stack, a two-level step.
4. **Reading order in race engineering** - the beginner-engineering guides converge on
   *delta -> speed -> inputs*: the delta locates the corner that moved lap time, speed explains
   the behaviour, inputs explain the speed.
   [MySimRig - telemetry for beginners](https://mysimrig.nl/en/blog/simracing/sim-racing-telemetry-for-beginners/),
   [RacingMojo - finding speed with telemetry](https://www.racingmojo.com/blog/finding-more-speed-using-telemetry-data/),
   [Podium Prophets - reading F1 telemetry](https://podiumprophets.com/blog/reading-f1-telemetry-beginners-guide).
   Implication: Δ TIME earns a tall lane adjacent to speed; whether it sits above or below speed
   is a coin the convention does not call (section 8 Q2).
5. **MultiViewer** (the live F1 fan client the project already cites) overlays
   **speed / throttle / brake / gear / RPM / DRS** as one onboard telemetry block -
   [multiviewer.app](https://multiviewer.app/). Implication: the six-channel set in section 5.1 is the
   standard live set minus RPM (not on the wire) plus Δ (this window's two-car product).
6. **Cosworth Pi Toolbox / AiM Race Studio** - the user guides confirm the same waveform-strip
   model (channels as horizontal strips over one distance axis, cursor readout per strip) but the
   public docs carry no layout numbers worth citing beyond that:
   [Pi Toolbox user guides](https://www.cosworth.com/user-guides-and-manuals/pi-toolbox/),
   [AiM RaceStudio 3 Analysis manual](https://www.aim-sportline.com/docs/racestudio3/manual/html/analysis.html).
   AiM's channel tags carry the value/average readout
   ([Rennlist thread on RS2 features](https://rennlist.com/forums/data-acquisition-and-analysis-for-racing-and-de/991767-new-aim-race-studio-2-features.html)).

**Where the convention disagrees with the drawing:**

- Convention stacks **RPM**; the wire does not carry it (broadcast car data does - realism doc
  Section 2.1 row 3 - but `FrameData` does not resample it). Not added; it would be a producer + loader
  change for a channel the strategist does not act on.
- Convention has **no Δ TIME lane** in the single-car worksheet - the delta is an overlay-mode
  channel. This window is permanently two-car (main + broadcast rival), and the Qt original's
  headline chart is the delta, so it stays: this is a pit wall, not a driver-coaching sheet.
- The Field's shared cursor is **hover-driven** (analysis); this window's is **the car's live
  position** (a race in progress). The hover/inspection cursor is deliberately deferred (section 8:
  a live wall is not moused - and at 10 Hz `notMerge: true` would fight the axisPointer state).
- Convention keeps a y-axis per lane. This design keeps its own (locked ranges, bounds unlabelled),
  which is already the shipped `valueAxis` behaviour.

---

## 4. The budget - the arithmetic everything below must sum to

Derived from the shipped CSS and confirmed against the sprint-9 baseline measurements.

```
1485 x 833 client                          1265 x 593 client (1080p laptop at 150 %)
833 - 23 status bar - 20 body padding      593 - 23 - 20                       = 550
                              = 790        550 - 29 band1 - 10 gap             = 511  (measured 510)
790 - 29 band1 - 10 gap       = 751        columns: left 630 | right 605
columns: left 630 | right 825 (measured 750 tall)

LEFT  = tower 437 + 10 gap + BESTS slot 303   LEFT  = tower 437 + 10 + BESTS slot 63
RIGHT = tabs 26 + 6 gap + tab panel 718       RIGHT = tabs 26 + 6 + tab panel 478

TRACES tab panel = traces card 555 + 10 + side column 260
traces card interior = 718 - 2 border - 20 padding - 18 header - 8 gap = 670 (grid measured 666)
side column = ring ~304 + 10 gap + radio 404
```

At 1265 the traces card interior is `478 - 48 = 430`, and its width `605 - 260 - 10 - 22 = 313`.

---

## 5. SPEC A - the stacked traces (deviation #7)

### 5.1 The shape

**One ECharts instance, six horizontal lanes over ONE distance axis, one unbroken cursor.**
Replaces the four independent `TraceChart` instances and the `.traces-grid` 2x2.

Lane order, top to bottom (convention: speed first, per section 3.1; the Δ-first alternative is section 8 Q2):

| lane | channel | y range (locked) | own-car stroke | rival stroke | height 1485 | height 1265 |
|---|---|---|---|---|---|---|
| 1 | **SPEED** km/h | `[0, 360]` (`SPEED_Y`, unchanged) | solid 2 px `INFO #3b82f6` | dashed 2 px `WARNING #f59e0b` | **145** | **88** |
| 2 | **Δ TIME** s (rival − main) | `[-3, 3]` (`DELTA_Y`, unchanged) | zero baseline, solid 2 px `INFO` | delta trace, dashed 2 px `WARNING` | **145** | **88** |
| 3 | **THROTTLE** % | `[-5, 105]` (`THROTTLE_Y`, its own constant) | solid 2 px `SUCCESS #10b981` | dashed 2 px `WARNING` | **96** | **58** |
| 4 | **BRAKE** % | `[-5, 105]` (`BRAKE_Y`, its own constant - never merged with throttle's; that coincidence-coupling defect is already paid for) | solid 2 px `DANGER #ef4444` | dashed 2 px `WARNING` | **96** | **58** |
| 5 | **GEAR** (step) | `[0, 9]` (NEW `GEAR_Y`) | solid 2 px `ACCENT #a78bfa`, `step: "end"` | dashed 2 px `WARNING`, step | **82** | **50** |
| 6 | **DRS** (binary step) | `[-0.2, 1.2]` (NEW `DRS_Y`) | solid 2 px `INFO #3b82f6`, step | dashed 2 px `WARNING`, step | **38** | **24** |
| | inter-lane gaps, 5 x 6 px | | | | 30 | 30 |
| | shared x-axis band (ticks + "Distance (m)") | | | | 34 | 34 |
| | **total** | | | | **666** ✓ | **430** ✓ |

666 is exactly the box `.traces-grid` occupies today at 1485; 430 is its box at 1265. **The card
does not change size** (section 8 Q1 records the alternative that shrinks it).

Checks against reality (executed on the real session pickle for this spec): gear spans 1-8 with
all eight values present, so `[0, 9]` frames every step with half a step of air; the raw `drs`
open set `{10, 12, 14}` occurs (555 frames) so the DRS lane is real, and it is rare on Melbourne
2025 (0.4% of frames) so the lane is flat most laps - the readout (section 5.4) is what tells a reader
"closed" from "broken".

### 5.2 Every axis rule carried over, one rule new

- **X**: `[0, circuit_length_m]` with the `MIN_CREDIBLE_CIRCUIT_M` guard and `FALLBACK_X_MAX`,
  unchanged. **Only lane 6 renders tick labels and the axis name** ("Distance (m)", `1k`-form
  labels, `nameGap 20`); lanes 1-5 hide `axisLabel` and `axisTick`. This is where the height
  comes from: the 2x2 paints four 36-px axis bands, the stack paints one 34-px band.
- **Y**: locked per lane, bounds unlabelled, interior ticks at 10 px - the shipped `valueAxis`
  behaviour, applied per grid. GEAR labels integers only; DRS hides y labels entirely (a binary
  lane's label row and readout carry the meaning).
- **All six grids share `left: 44, right: 12`** so the lanes align and the cursor is one straight
  line. Lane plot width: `533 - 56 = 477 px` at 1485 (today: 206 per cell), `313 - 56 = 257 px`
  at 1265 (today: ~152 per cell, "the plot is 152 px, the axis about 120").

### 5.3 How the two cars read in one lane

Unchanged semantics, restated for the stacked form:

- Own car: solid, in the metric's own colour (the per-metric colour is the Qt heritage and each
  is a pinned slot in `test_pitwall_tokens.py`; GEAR adds one new site for `ACCENT`, DRS one new
  site reusing `INFO`).
- Rival: **dashed, WARNING, on every lane** - dashed = broadcast tier, the taken stroke. The
  rival's gear and `drs_open` ride the same span, so the two new lanes get the rival for free.
  The `BROADCAST` tag stays on the header chip (one card, one header, six lanes - saying it six
  times would be the wrongness the per-cell legends already died for).
- Δ lane: the main car IS the solid zero baseline; the delta trace is dashed WARNING because it
  is rival-derived. `deltaSeries`, `lerpSorted`, the same-lap-only rival rule and the
  eviction-before-append rule in `TraceAccumulator` are untouched. The locked `[-3, 3]` clip
  around pit stops stays (settled: no autorange churn at 10 Hz; the tower's GAP column carries
  the number meanwhile - already noted in #986).

### 5.4 The shared cursor, and the readout the research says it owes

- **The cursor is the car's live lap position**: `drivers[driver_main].rel_dist * xMax`, exactly
  today's source (the DRIVERS block, never the span tail). Rendered as **ONE absolutely
  positioned overlay div** - 1 px, solid, `CURSOR_LINE #9ca3af` (`TEXT_TERTIARY`, the taken
  stroke) - spanning from lane 1's top to lane 6's bottom at
  `x = 44 + (cursorX / xMax) * laneWidth`. One div replaces six per-grid markLines, is unbroken
  across the gaps, and cannot shimmer (the solid-not-dashed lesson is inherited). The transform
  is linear against a locked axis, so the duplication risk is two constants (44, 12) already
  owned by the option builder.
- **Per-lane readout** (MoTeC/AiM convention, sections 3.1 and 3.6): each lane carries a 12-px label row
  inside its own height - left: the channel name + unit in the lane's colour
  (`SPEED km/h · Δ TIME s · THROTTLE % · BRAKE % · GEAR · DRS`); right: the current value from
  the newest main-span sample, 10 px mono (`287 · +0.42 · 100 · 0 · 7 · OPEN/CLOSED`). The
  ECharts grid for each lane starts 12 px down so data never collides with the row. HTML
  overlays, not ECharts graphics, so `notMerge: true` cannot restart them.
- No hover/inspection cursor in v1 (section 3, last bullet).

### 5.5 Placeholders and the frozen board

- Single-driver mode: the Δ lane renders its in-lane caption; the other five lanes render the
  main car alone. (Today the caption replaces a whole 262 x 328 cell.)
- Starved-when-frozen: ONE caption over the whole stack region when
  `frozen && main.xs.length < 2` - the four-copies-of-one-sentence form dies with the 2x2. The
  cause-precision lesson (#993's "a true state with a false cause") carries over verbatim: the
  caption only claims starvation when the MAIN trace is starved.

### 5.6 What the stacked form returns - the before/after arithmetic (1485)

```
BEFORE (2x2, measured):  grid 533 x 666, four cells 262 x 328
  per cell:  title row 18 + gap 4 + plot 306;  plot = 8 top + DRAWING 262 + 36 axis band
  drawing area:   4 channels x (206 w x 262 h)
  chrome, counted in the HEIGHT dimension (one column of the 2x2):
      2 title rows 44 + 2 axis bands 72 + 2 grid tops 16 + 1 grid gap 10 = 142 of the 666
  height spent on data: 524 of 666

AFTER (stack, same 533 x 666 box):
  drawing area:   6 channels totalling 602 px tall x 477 w   chrome: 1 axis band 34 + 5 gaps 30 = 64
  height spent on data: 602 of 666
```

- **Height returned to the column: 0 - by design.** The drawing draws the traces at full column
  height and says "the traces are not oversized". What the stack returns is INTERNAL:
  **78 px of chrome converted to data** (142 -> 64), **two new channels** (gear 82 + DRS 38 =
  120 px, paid for by that chrome plus the four original lanes each giving up height), and
  **2.3x the x-resolution per channel** (206 -> 477 px; 25.3 -> 10.9 m per pixel). At the 1265
  client - where this sprint's P0 lived - the width gain is 152 -> 257 px per lane.
- The honest cost, stated: the four original channels each lose drawing height (262 -> 145 for
  speed/Δ, 262 -> 96 for throttle/brake). That is the stacked-lane convention (section 3: lanes of
  ~90-150 px are the norm), and it is the direct answer to "los 4 graficos siguen estando muy
  grandes": the same pixels now carry six channels one cursor can cut through at a single point
  of track, instead of four charts a reader has to cross-reference by eye.
- Four ECharts instances become one: one canvas, one `setOption` per tick at 10 Hz instead of
  four.

### 5.7 Implementation shape (files)

| file | change |
|---|---|
| `features/data/TraceStack.tsx` | NEW - one `useEChart` instance; `laneLayout(stackHeightPx)` computes each grid's `{top, height}` from the section 5.1 weights (weights `3 / 3 / 2 / 2 / 1.7 / 0.8`, axis band 34, gaps 6); builds 6 grids / 6 x-axes / 6 y-axes / 12 series; the cursor overlay div; the 6 label+readout rows. |
| `features/data/OwnCarTraces.tsx` | keeps the accumulator, header, frozen/starved logic; renders `<TraceStack>` instead of four `<TraceChart>`. Its "gear and DRS are deliberately absent" docstring block is DELETED by the PR that lands section 5.2's producer change - the refusal it encodes is honoured, not overruled: the constant is decoded producer-side. |
| `features/data/TraceChart.tsx` | RETIRED (its only consumer is the 2x2). |
| `features/data/traceBuffer.ts` | `TraceRow` gains `gear: number; drsOpen: boolean`; `store()` copies them; `channel()` gains the two keys. Idempotence, eviction and the same-lap rival rule untouched. |
| `lib/bridge.ts` | `TelemetrySample` gains `drs_open: boolean`. |
| `styles/data.css` | `.traces-grid` and `.trace-cell` rules retired; `.trace-stack { position: relative; flex: 1 1 auto; min-height: 0 }`, label-row and cursor-overlay classes. No new hex: every colour is an existing `chart.ts` export or the lane colours pinned in the TSX. |
| `tests/surfaces/test_pitwall_tokens.py` | +1 site for `ACCENT` (gear), +1 for `INFO` (DRS); the retired per-cell sites removed. |
| headless smoke | assert 6 grids, ONE labelled x-axis, lane y-extents equal the locked constants, and the cursor div's `left` equals the ECharts `convertToPixel` of the same x (the effect, not the mechanism). |

## 5.2-bis SPEC B - gear and DRS, and the producer edit priced (deviation #8)

**Lane shapes** are in the section 5.1 table: gear is a step function (`step: "end"` - the value holds
until the next sample; The Field: "a step chart"), range `[0, 9]` framing the measured 1-8;
DRS is a two-level step in the thinnest lane, range `[-0.2, 1.2]`, no y labels.

**The producer change, exactly:**

The wire already carries `gear` (int, unread by any consumer today) and raw `drs` (the FastF1
code). The open set `{10, 12, 14}` lives ONLY at `src/arcade/track.py:40` (`_DRS_ACTIVE`), and
`OwnCarTraces.tsx` refuses to fork it into TypeScript - correctly. So the producer decodes, the
same treatment `track_status_label` and #986's proposed `compound_name` already get:

1. `src/arcade/config.py` - the constant moves to the shared-constants home:
   `DRS_OPEN_CODES: Final[frozenset[int]] = frozenset({10, 12, 14})` (with track.py:35-39's
   comment about code 10 travelling with it).
2. `src/arcade/track.py:40` - `_DRS_ACTIVE` is replaced by the import; `track.py:232`'s
   `np.isin(...)` consumes the same name. One constant, one home, two consumers.
3. `src/arcade/app.py:162` (`_frame_to_telemetry`) - the packed dict gains one key:
   `"drs_open": int(frame.drs) in DRS_OPEN_CODES`. Both spans (main and rival) flow through this
   one function, so the rival lane is fed by the same edit. The dev producer
   (`scripts/dev_pitwall_producer.py`) drives `F1ArcadeView`'s real snapshot path, so it inherits
   the field with zero edits.
4. **No `STREAM_SCHEMA_VERSION` bump** - `stream.py`'s own contract: "Adding a key an old
   consumer can ignore does not bump it."
5. Tests that MUST move in the same PR (they freeze the shape):
   `tests/surfaces/test_arcade_telemetry_span.py:190` asserts the exact key set;
   `test_arcade_wire_contract.py:261-276` pins the per-key type map (add `"drs_open": "bool"`).
   Plus one positive case: a frame with `drs=14` packs `drs_open=True`, `drs=8` packs `False`
   (8 is "eligible, not open" and is the third-commonest code on the real session - 566 frames).
6. Payload price: ~16 bytes per sample, ~2-3 samples per driver per tick at 1x, two spans ->
   **~80 bytes on a 17,278-byte tick, under 0.5%**. Nothing.

**The data honesty note**: on the only race on disk DRS is open 0.4% of frames (measured for
this spec), so the lane will read CLOSED for whole laps. The label row's live readout
(`DRS CLOSED` / `DRS OPEN`) is what makes a flat lane legible as a true state - the same
dead-feed-must-look-dead doctrine, one lane down.

---

## 6. SPEC C - the left column's 150 px (deviation #3)

The slot under the tower is a property of the tower: 437 fixed + 10 gap leaves **303 px at 1485**
(the drawing's 302) and **63 px at 1265**. Shipped BESTS is 153 -> **150 px of nothing at the tall
client, zero slack at the narrow one**. That asymmetry rules the options:

| option | verdict |
|---|---|
| **(a) BESTS ranks deeper when the room exists** - RANKED becomes a depth derived from the slot | **RECOMMENDED.** The RaceX evidence is ranked lists ("four ranked lists (S1, S2, S3, Lap) plus Theoretical"); the shipped `RANKED = 3` was argued against the space then available, not against the drawing's 302. Elastic, so it is a no-op at 1265 where the compact ladder already owns the answer. |
| (b) give the tower breathing (taller rows) | REJECTED: 20-px rows are a measured legibility floor for 150% scaling, and +7 px of air per row buys no information. |
| (c) move a panel in (e.g. the radio) | REJECTED: contradicts the agreed home ("under the ring"), and anything moved in must degrade to zero at 1265, where the slot is 63 px - only elastic content survives both clients. |

**The depth rule.** One derived value replaces the constant:

```
depth = clamp( 3 + floor((room - H3 - FONT_GUARD) / ROW), 3, 10 )
  room       = the slot the tower leaves (the existing hook already measures it)
  H3         = the ranked card's height at depth 3, latched populated (the existing X1-hardened latch)
  ROW        = 17  (the pinned .bests-row line-height)
  FONT_GUARD = 8   (the measured `--qt-mono` swap growth from the sprint-9 exit gate, so a late
                    font swap cannot turn a boundary fit into a clip)
```

- At 1485: `3 + floor((303 - 153 - 8)/17) = 3 + 8 = 11 -> capped at 10`. Card = 153 + 7 x 17 =
  **272 px in the 303 slot, 31 px of honest air**. **What fills the drawing's 302: ranks 1-10 in
  all four sections** - P10 is the last point-scoring position, which is where a ranked list on
  a wall stops meaning anything.
- At 1265: room 63 < H3 -> the shipped compact-leaders ladder, unchanged.
- No oscillation by construction: depth is a function of `room` (a property of the tower and
  column, which do not move when this card renders more) - the same argument that made
  `useFitsRanked` stable, extended one step.
- `BestsPanel.tsx`'s "top three is a structural limit" docstring is rewritten to "three is the
  FLOOR, ten the cap, the room decides" - the sentence it replaces was true of a 63-303 slot
  assumed fixed, and this spec unfixes it.
- The subtitle should say what depth is showing (`session · top 10`) so a reader at a different
  client is never comparing two silently different panels.

The alternative fill - a second block of **speed bests** (V1 / V2 / VFL / VST, RaceX's own
`Bests - Time | Bests - Speed` tab strip; all four fields already ride the bulk in
`DriverLaps.best`) - is real, wire-ready and NOT recommended as the default: it is new
information architecture where option (a) is one derived constant. It is section 8 Q3.

---

## 7. SPEC D - the radio feed (deviation #9)

**Decision: the fix is the fold and the density, not the width.** The 260-px home under the ring
stands (it is the drawing's internally consistent reading, section 2.1). Three moves, two of them already
owned by **#986** - build on it, do not re-spec it:

1. **#986 D8 as written**: category chips from the wire's `category`/`flag` (populated, read by
   nothing today - 39 RCM events on Melbourne: Other 23, Flag 13, SafetyCar 2, Drs 1) and
   **collapsing consecutive identical RCM rows into one with a count** (`BLUE FLAG ... x4`).
   On the measured feed this alone raises distinct-information-per-fold visibly: 42 events,
   ~10 visible, with duplicate runs spending 3-4 slots each.
2. **The fold announces itself.** The header count says how many EXIST (42); it does not say the
   panel is showing ~10. Add the shipped convention other panels use: the count becomes
   `10 / 42` (visible/total) or the list's last visible row fades under a 12-px
   `+ 32 older ·  scroll` line. Scrollbars stay hidden; the AFFORDANCE is what was missing.
3. **No geometry change.** The 13-px shortfall against the transposed drawing (404 vs 417) is the
   tab strip's structural cost and is not worth a move. If the literal 417-wide
   reading, that is a restructure of the whole tab (section 8 Q4, priced there) - not a patch to this
   panel.

Interaction with sections 5 and 6: none at the wide client (different columns). The one geometry lever that
WOULD feed the radio - shrinking the stack and giving the freed band to a full-width feed - is
Option B in section 8 Q1, priced and not recommended.

---

## 8. The drawings - both clients, every height, sums shown

Drawn the way the ground-truth memory draws it, for side-by-side comparison. TRACES tab, spec
applied (Option A, recommendation):

### 1485 x 833

```
+==================================================================================================+
| BANDA 1   L 24/57  |  GREEN  |  01:12:44.318  |  > 4x  |  Connected                        29 px |
+=========================================+========================================================+
| TIMING TOWER                  630 x 437 | [ TRACES ][ RACE PACE ][ RACE TRACE ]           26 px  |
| P  #  DRV GAP   INT  S1..  (20 rows)    +====================================+===================+
|                                         | OWN-CAR STACK          555 x 718   | RING    260 x 304 |
+=========================================+ | SPEED km/h              287 | 145|  (200 px SVG)     |
| BESTS                         630 x 272 | | ~~~~~~~~~~~~~~~~~~~~~~~~~|~~~~   | RUN  FIN  OUT     |
|  S1        S2        S3        LAP      | | Δ TIME s              +0.42 | 145+-------------------+
|  1 NOR ..  1 VER ..  1 NOR ..  1 NOR .. | | ------------0---------~~~|~~~   | RADIO / RCM       |
|  2 ..      2 ..      2 ..      2 ..     | | THROTTLE %              100 |  96|       260 x 404   |
|  .. ranks 1-10, all four sections ..    | | ~~~~~~~~~~~~~~~~~~~~~~~~~|~~~~   | 10/42 · chips     |
| 10 ..     10 ..     10 ..     10 ..     | | BRAKE %                   0 |  96| L24 RCM  BLUE x4  |
|  THEORETICAL  1:24.883   (NOR)          | | ~~~~~~~~~~~~~~~~~~~~~~~~~|~~~~   | L24 VER  BROADCAST|
|  (31 px air in the 303 slot)            | | GEAR                      7 |  82| L23 NOR  ...      |
|                                         | | _|~|__|~~|_|~~~~|______|~|__     | ...               |
|                                         | | DRS                  CLOSED |  38| + 32 older·scroll |
|                                         | | __________________________|__    |                   |
|                                         | |  0    1k    2k  Distance(m) |  34|                   |
+=========================================+====================================+===================+
|  lap 24 - live                                                                             23 px |
+==================================================================================================+
   |<---------------- 630 ---------------->|<---------------------- 825 ---------------------->|

   29 + 10 + 751 = 790 ✓        left: 437 + 10 + 303 slot (272 card + 31 air) = 750 ✓
   right: 26 + 6 + 718 = 750 ✓  stack: 145+145+96+96+82+38 + 5x6 + 34 = 666 = the grid box ✓
   the | in each lane = the ONE cursor div, spanning lane 1 top to lane 6 bottom
```

### 1265 x 593

```
+===================================================================================+
| BANDA 1   L 24/57 | GREEN | 01:12:44 | 4x | Connected                       29 px |
+===================================+===============================================+
| TIMING TOWER          630 x 437   | [ TRACES ][ RACE PACE ][ RACE TRACE ]  26 px  |
| (20 rows, never squeezed)         +=========================+=====================+
|                                   | OWN-CAR STACK 335 x 478 | RING     260 x 304  |
|                                   | | SPEED km/h  287 |  88 |  (200 px SVG)       |
+===================================+ | ~~~~~~~~~~~~|~~~~     +---------------------+
| BESTS (compact leaders) 630 x 62  | | Δ TIME s  +0.42 |  88 | RADIO / RCM         |
| S1 NOR .. S2 VER .. LAP NOR ..    | | ------0-----|~~~~     |        260 x 164    |
| THEO 1:24.883                     | | THROTTLE %  100 |  58 | 4/42 · chips        |
| (the shipped ladder, unchanged:   | | ~~~~~~~~~~~~|~~~~     | L24 RCM BLUE x4     |
|  63 px slot, depth formula        | | BRAKE %       0 |  58 | ...                 |
|  yields the compact form)         | | ~~~~~~~~~~~~|~~~~     | + 38 older·scroll   |
|                                   | | GEAR          7 |  50 |                     |
|                                   | | _|~|__|~~|__|~~__     |                     |
|                                   | | DRS      CLOSED |  24 |                     |
|                                   | | _____________|___     |                     |
|                                   | |  0   2k  4k Dist | 34 |                     |
+===================================+=========================+=====================+
|  lap 24 - live                                                              23 px |
+===================================================================================+
   |<------------- 630 ------------->|<------------------ 605 ------------------->|

   29 + 10 + 511 = 550 ✓   left: 437 + 10 + 63 = 510 ✓   right: 26 + 6 + 478 = 510 ✓
   stack: 88+88+58+58+50+24 + 30 + 34 = 430 = the card interior ✓   lane width 257 px
   side column: 478 - 304 ring - 10 = 164 for the radio (the fold affordance carries it)
```

---

## 9. Element-by-element wire trace

Constraint 7's table: every element reads a field already on the wire, or is priced.

| element | source | producer change |
|---|---|---|
| SPEED / THROTTLE / BRAKE lanes | `telemetry.drivers[code][]` `.speed/.throttle/.brake` (was `.main[]`/`.rival[]` before schema v2, #1048) | none |
| Δ TIME lane | computed from `.t` via `deltaSeries` (unchanged) | none |
| **GEAR lane** | `telemetry.drivers[code][].gear` - on the wire since #841, read by nothing today | **none** |
| **DRS lane** | NEW `telemetry.drivers[code][].drs_open` | **Section 5.2-bis, priced: 3 source files + 2 test files + bridge.ts, ~80 bytes/tick, no schema bump** |
| cursor + readouts | `drivers[driver_main].rel_dist` x `circuit_length_m`; newest main-span sample | none |
| lane colours | `palette.ACCENT` (gear, new site), `palette.INFO` (DRS, new site); rest existing | none (token-test counts move) |
| BESTS depth | `bulk` (unchanged data); depth is client geometry | none |
| radio chips / collapse | `RadioEvent.category` / `.flag` - populated, unread (#986 D8) | none |
| radio fold affordance | client-side count of rendered vs total | none |

## 10. Build order - what is independent, what each step returns

| step | what | depends on | worth |
|---|---|---|---|
| **1** | Producer `drs_open` (section 5.2-bis): config.py + track.py + app.py + the two wire tests + bridge.ts | nothing | unblocks the DRS lane; deletes the last cross-language-constant refusal on this window |
| **2** | `TraceStack` with the four existing channels (section 5.1-5.7 minus the two new lanes): one instance, one axis, one cursor, readouts; retires `TraceChart`; fixes `PITWALL_DELIVERY_PLAN.md:393`'s over-claim in the same PR | nothing | 78 px of chrome -> data; x-resolution 206 -> 477 px/channel (152 -> 257 at 1265); 4 instances -> 1; **the drawing's headline item** |
| **3** | GEAR + DRS lanes (section 5.2-bis client side) | 1 (DRS), 2 (both) | +2 channels for the 120 px the chrome paid; completes the agreed five-channel line |
| **4** | BESTS adaptive depth (section 6) | nothing | fills 119 of the 150 empty px with ranks 4-10; 31 px stays as air; no-op at 1265 |
| **5** | Radio density: #986 D8 + the fold affordance (section 7) | nothing (already tracked in #986) | no geometry; the "apretujado" fix |

Steps **1, 2, 4, 5 are mutually independent** and can land in any order or in parallel; only
step 3 waits (on 1 and 2). Suggested sequence for review sanity: 1 -> 2 -> 3 (one story: the
stack), with 4 and 5 interleaved anywhere.

## 11. What this spec could not settle

**Q1 - Does the stack keep the full card, or shrink to feed the radio?**
The drawing says full height ("the traces are not oversized") and this spec follows it
(**Option A, recommended**: stack fills 666/430, radio stays 260-wide, density-fixed). Read
literally, though, "charts too large" plus "radio cramped" supports **Option B**: render the 1265
lane heights at BOTH clients (stack region 430), which frees `718 - 478 - 10 = 230 px` at 1485
for a full-width `825 x 230` radio band under the traces+ring row - single-line rows, no clamp,
~11 unwrapped events visible. Cost: the wide client shows the small stack (halved lane heights
for no information gain), the TRACES tab gets a second structure that exists only above ~700 px
of client height, and the drawing's shape is abandoned. **Recommendation: A. Build A; if the
radio still feels squeezed after #986 + the fold affordance, B is a layout-only follow-up.**

**Q2 - Lane order: SPEED first (convention) or Δ TIME first (workflow)?**
MoTeC and The Field put speed on top; the engineering reading order starts at the delta. This
spec ships SPEED first. The flip is one array literal, and if the window is read delta-first,
say so and it moves.

**Q3 - What fills the BESTS slot: depth (recommended) or a Bests-Speed block?**
Section 6 recommends ranks 1-10 (RaceX's ranked lists; one derived constant). The alternative -
V1/V2/VFL/VST sections, RaceX's own `Bests - Speed` tab, all four fields already on the bulk -
adds a different KIND of information for ~74 px + new `sessionBests` fields. They are not
exclusive (depth to ~6 + the speed block also sums inside 303), but that combination should be
chosen by the person who reads the wall, not by this spec.

**Q4 - The radio's 417.**
Section 2.1's arithmetic says the drawing's "417 x 260" is transposed (260 wide x 417 tall) and the
shipped panel is 13 px off, not 157. This spec proceeds on that reading. If a literal
417-wide feed, that is Option B's band (Q1) or a tab-level restructure - name it and it gets
specced; nothing in steps 1-5 blocks it.

**Q5, settled here and recorded for visibility:** no hover/inspection cursor in v1 (live wall,
10 Hz `notMerge` fights axisPointer state); RPM stays off (not on the wire; a strategist does
not act on it); the ring does not shrink to feed the radio (its lap-1 clumping is already
borderline at 200 px).

---

## Summary

- **Deviations found (agreed drawing vs shipped): 10 catalogued; 3 are real unbuilt work**
  (BESTS at 153 vs 302 · the 2x2 vs the stacked form · gear+DRS absent), **1 is a transposed
  pair in the drawing itself** (the radio's "417 x 260" - shipped is 13 px off the internally
  consistent reading, not 157), the rest are documented consequences of the 630-px column and
  the tab strip. Plus one docs over-claim (`PITWALL_DELIVERY_PLAN.md:393` says "stacked" shipped).
- **Height returned to the column by the stacked form: 0, by design** - the drawing draws the
  traces at full column height. The return is internal: 78 px of axis/title chrome converted to
  data, two new channels (gear 82 px + DRS 38 px) riding on it, x-resolution per channel 2.3x
  (206 -> 477 px at 1485; 152 -> 257 at 1265), four chart instances collapsed to one. The left
  column's 150 px is filled by BESTS ranking to depth 10 (272 of 303, 31 px of air); the radio
  keeps its agreed 260-px home and gets #986's chips + duplicate collapse + a visible-fold count.
- **Producer change: one, priced**: `drs_open` decoded in `_frame_to_telemetry`
  (`src/arcade/app.py:162`) from a `DRS_OPEN_CODES` constant promoted to `src/arcade/config.py`,
  ~80 bytes/tick, no schema bump, two wire tests move with it.
- **Decisions still open: Q1** (full-height stack per the drawing, recommended, vs
  compact stack + a full-width 825 x 230 radio band at the wide client) **and Q3** (BESTS fills
  its 302 with depth-10 ranks - recommended - vs a RaceX-style Bests-Speed block). Q2 (lane
  order) and Q4 (the 417) are one-line follow-ups either way.
