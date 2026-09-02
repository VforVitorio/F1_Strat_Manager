# Design gate: PITWALL responsive + motion

Gate run 2026-08-24 against `dev` @ `52f1f214`, on the live loopback host
(`http://127.0.0.1:52528`), Melbourne 2025 producer. This file is appended as findings settle;
an abrupt end means the gate died mid-run, not that the remaining questions passed.

Verdicts are separated from prescriptions throughout. Where a change's blast radius could not be
priced, that is said instead of prescribing.

## Question index

1. BESTS middle form: REFUTED as needed; a depth-1 ranked floor closes the real band.
2. The 12 px pace row: CONFIRMED unsupported; its payoff exists on one screen class only.
3. The real client set: CONFIRMED replaced by the place() table; only layout adaptation reaches the failing screens.
4. AGENTS: clip CONFIRMED worth fixing (it is the charts' x axis); void REFUTED as the BESTS defect.
5. Reflow needed in two places: the sub-1360 right column and the AGENTS chart minimum at laptop heights.
6. Motion items: 1, 3, 6 survive; 2 blocked by radio keying; 4 desktop-only as scoped; 5 collides with unmount-on-switch; 7 see Q7.
7. Hover: the 10 Hz cost premise REFUTED by measurement; the real hazards are row identity, not paint.
8. Frozen filter: only user-driven items fire under it; 28 percent contrast loss; the tooltip portal escapes it correctly.
9. The sprint spends itself on the horizontal defect, CONFIRMED, and its first fix is small.
10. 1470 px: the number is not a layout constant; it is the header note's width. Not to be accepted.
11. The 145 px gap: REFUTED as a sprint defect; unreachable at any default placement.

## Rig used by this gate

Live loopback pages, fresh browser context per size, `document.fonts.ready` plus 900 ms settle,
matching the report's rig. The producer advances during a run, so lap-dependent text differs
between probe rows; where that matters it is named. Fleet client heights are derived from
`config.py:106-107` (`min(950, screen_h - 90)` outer) minus the 37.3 px title bar measured in
`project_pitwall_data_layout.md`, the same machine-measured constant the report's width table
leans on (its 14 px frame delta). Both constants are one machine's chrome applied fleet-wide;
the direction of every conclusion survives that approximation, exact pixel values do not.

The derived fleet clients, which the rest of this report uses:

| screen | DATA client | AGENTS client |
|---|---|---|
| 1920x1080 | 1486 x 913 | 1486 x 913 |
| 1707x960 | 1486 x 833 | 1486 x 833 |
| 1536x864 | 1486 x 737 | 1482 x 737 |
| 1440x900 | 1426 x 773 | 1386 x 773 |
| 1366x768 | 1352 x 641 | 1312 x 641 |
| 1280x720 | 1266 x 593 | 1226 x 593 |
| 16:10 laptop class (1280x800, 1920x1200 at 150 %) | 1266-1352 x 673 | 1226-1312 x 673 |

## Attacks on MEASUREMENTS.md

### A1 (P1). The 1470 px floor is not a constant of the layout; it is the header note's text width, and it moves per lap

The report's section 2 says the right edge "sits at 1465.2 px whatever the viewport" and calls the
layout rigid. Measured live at 1266x593, the binding element is `.traces-lap`: with the note
`NO TRACK IN COMMON WITH PIA YET` up, band 4's first grid track refuses to shrink below 545.4 px
and the document overflows 199 px; with `Δ FROM 5026 m` up it refuses below 380 px and overflows
34 px; with the label hidden by a style injection, the track collapses to the viewport-driven
336 px and the overflow is exactly 0. The mechanism is `data.css:581`
(`grid-template-columns: 1fr 260px`, whose `1fr` carries an `auto` minimum) meeting the
`white-space: nowrap` on `.traces-lap` (`data.css:626`): a nowrap line's min-content is its full
text, and nothing between it and the grid track has `min-width: 0`, so one line of race-state
prose sets the whole window's minimum width. The notes compound
(`OwnCarTraces.tsx:211-217`: blind list, delta anchor, and one of three no-delta reasons can all
render at once), so the worst-case floor is not 1470, it is unbounded in the length of the note.

Consequence for the sprint: "the layout is rigid below 1470 and does not reflow" prescribes a
layout rebuild. The measured truth prescribes something much smaller first: the layout IS
substantially fluid (the only fixed columns are the 630 px tower and the 260 px ring,
`data.css:196` and `data.css:581`), and the floor is a min-content leak. An injected
`minmax(0, 1fr)` on band 4 plus `min-width: 0` on `.traces` was tried live at 1266x593: overflow
went to 0, the ECharts stack resized to 314 px through its own observer (`chart.ts:94`) without
error, and the header fell back to its already-designed ellipsis. What that one-line change does
NOT buy is a GOOD window at 1266, only an honest one; six lanes at 336 px were never designed or
audited, and that blast radius is not priced here.

### A2 (P1). Sections 2 and 3 are one defect, and section 3 understates what #1068 did

The report presents the 1470 floor (section 2) and the `.traces-lap` truncation (section 3) as
independent findings, and frames #1068's cost as "converted a wrap into a silent horizontal
truncation" of the header. Before commit `9c573847` the label had no `nowrap`
(verified against `9c573847^:data.css:607`), so its min-content was one word and the window had
no note-driven floor; after it, the same commit created the entire section 2 overflow. The
truncation the report describes is the SMALL symptom; the large one is that the nowrap turned a
2 px-per-lane wrap into up to 200 px of off-screen ring and radio on laptop clients. This is the
repo's own "a fix that moves a defect rather than removing it" class, which the brief flags as a
possibility; it is confirmed, and the move was larger than the report says.

### A3 (P2). The 51 px AGENTS clip belongs to the chart cards, not the text cards the report names

Re-measured at 1265x593: the two 51 px overflows are on the PACE and TIRE `.agent-card-body`
elements; SITUATION overflows 10 px and PIT 8 px. The report's sentence attaches the 51 to the
cards whose visible losses it then describes (SITUATION's gap line, PIT's `UCUT 63% -> RUS`),
which are the 10 and 8 px clips. What the 51 px actually hides is both charts' entire x axis,
read off `shots/agents-1265x593.png`: neither chart shows its Lap axis at that client while both
show it whole at 1486x833. The conclusion "51 px of card body is lost at the small client"
survives; the diagnosis of WHERE changes what a fix would target.

Also a characterisation error in the word "clips": `.agent-card-body` is `overflow: auto`
(`agents.css:390`), so the content is wheel-reachable, invisible only because `qt-base.css:72-77`
hides every scrollbar. Unreadable without an affordance, but not lost. The distinction is this
window's own established vocabulary (the pace grid's edge fade exists for exactly this state).

### A4 (P2). Both AGENTS probe sizes miss the product's actual worst client, and one of them is not a client at all

`place()` gives AGENTS 1226 x 593 on a 1280x720 screen (width minus the 40 px stagger, height
identical to DATA's). The report measured 1265x593 (a DATA width) and 1226x630 (630 is the outer
height minus the OS allowance, with the 37 px title bar not yet subtracted, so it is not a client
height the product produces). Re-measured at the true 1226x593: the clips are 51/51/10/8, the
same as at 1265x593, so the report's "14 px at 1226x630" understates the 720p laptop by 37 px of
height, and the conclusion happens to survive only because width barely matters here. Wrong
population, right verdict, by luck.

### A5 (P2). The report never crosses its own height band with the heights the product opens at

Section 2 crosses `place()` with WIDTH and stops. Crossing it with HEIGHT (table above) changes
how several headline numbers read:

- The BESTS dead band's worst waste, 81 px at h=673, is realized by 16:10 laptops
  (1280x800-class, including 1920x1200 at 150 %). The six screens in the report's own table
  realize at most ~49 px (measured live: gap 49 at the 1366x768 client 1352x641) and 1 px at
  720p.
- The "second hole", 65 px at h=1000 and 145 px at h=1080, is unreachable at ANY default
  placement: `WindowSpec` caps the outer height at 950 (`config.py:107,123`), the tallest client
  the product can open is ~913, and `place()` never grows a window (`config.py:103-104`).
  Measured at 1486x913: depth 15, gap 14. Even the depth-17 data saturation is unreachable at
  defaults. The second hole exists only after a manual resize.
- RACE PACE's "overflows at every client the product opens at" is one screen short of true: at
  the 1920x1080 client (h=913) the measured overflow is 0. It is true of every laptop client.

### A6 (P3). Provenance and small wording

- The recorded probe (`probes/measure_band.mjs`) cannot produce section 4: it hardcodes
  `data.html`, waits on `.bests` and clicks DATA's tabs. The AGENTS numbers came from an
  unrecorded rig. They reproduce (51/14 at the report's sizes, void 221.8 px at 1486x833), so
  this is a provenance gap, not a data defect.
- "Every row above is byte-identical at 1265, 1350, 1485 and 1600 px wide" is true of the BESTS
  columns and false of the full probe rows (the clipped-element census reads 30 at narrow widths
  against 25-26 at wide, which is the horizontal overflow itself). The width-independence claim
  survives for the table it is about.
- The width-sweep floor of section 2 was measured with the longest single note up. With shorter
  notes the same sweep would find a lower floor; the fleet-loss column ("DATA loses 39/113/199")
  is therefore the worst steady state, not the constant state. The 1440x900 client measured 0
  overflow in this gate's own run purely because the note happened to be `Δ FROM 1 m` at that
  instant.

## Answers

### Q1. A middle form for BESTS, or fewer forms

Facts first. The card's height is linear in depth: 99 + 18k px (measured 153 at depth 3, 207 at
6, 243 at 8, 297 at 11, 369 at 15, 405 at 17; row height 18, `band-data.log`). So a depth-1
ranked card is 117 px and depth-2 is 135. The compact card is 62 px, and its own comment says it
is one wrap from folding THEO under an unannounced fold (`BestsPanel.tsx:244-252`), so the
compact form is already at ITS floor. Crossing with the fleet: the room under the tower is
roughly gap + 62; the 16:10 laptop client (h=673, room 143) fits a depth-2 card with 8 px to
spare, and the 1366x768 client (h=641, room ~111) fits nothing between 62 and 117, so no
lowering of the ranked floor reaches it.

Verdict: REFUTED that a new middle FORM is needed for the band the sprint was scoped around.
Lowering `RANKED_FLOOR` from 3 to 1 (`BestsPanel.tsx:53`) closes the 16:10 dead band with the
existing ranked form, at 117-135 px against 143 of room. The genuinely unserved band is room
63-116, which only the 1366x768 client occupies, for ~49 px of waste; a third form to serve one
screen's 49 px is more form than defect, and the docstring's own argument (a rank-1 list is still
the four best holders plus THEO) covers the product meaning. The "tower takes the slack" option
is dead on arrival: the tower renders 20 rows always as a deliberate decision
(`project_pitwall_data_layout.md`, "20 rows always, retirements included"). The
"column stops being content-sized" option re-opens sprint 9's measured lesson that anchoring
moves the void inside the card. One caution the sprint must carry: at depth 1 the subtitle still
has to say the depth, and the fit guard has to be driven red at h=641 and h=673 specifically,
per the scope's own trap list.

### Q2. The 12 px pace row height

CONFIRMED that the constraint is real and unsupported, and the question is narrower than the
brief phrases it. `--pace-row-h: 12px` with `padding: 0 1px` is at `data.css:1300,1309-1310`.
The literature claim (Material 52, condensed 40, trading terminals 32-36, 12 px only as a font
minimum) is quoted from the sprint's research file and was not re-derived by this gate.

What this gate adds is who the fit is FOR. Zero-scroll needs client h >= 900 (`band-data.log`:
paceOver 0 from 900 up, 22 at 833). Per the fleet table, h >= 900 exists only on the
1920x1080-class client (913). Every laptop the product opens on scrolls the grid TODAY, with the
edge fade and wheel already shipped for it (sprint 9). So "raising the row height breaks the
57-row whole-race fit" is true on exactly one screen class, and the fit is knife-edge even
there: at h=913 the scroller leaves room for 13.0 px rows, at h=900 for 12.8, so 12 is the only
integer that fits both, and the 1707x960 reference machine already misses the fit by 22 px.

Verdict: the whole-race fit is a desktop-only property that most of the fleet has already lost,
so it should not be allowed to veto every responsive decision on the laptop clients where it
does not exist. Whether the desktop keeps it is a product call for Víctor, not a gate finding;
the honest framing for that call is "913-tall desktops keep zero-scroll at 12 px, or every
client gets a taller row and the desktop joins the scroll it already has everywhere else". The
fit is not the sprint's hard constraint; it is one screen class's feature.

### Q3. The real client set

CONFIRMED that the two designed-for sizes were guesses and the `place()` table replaces them;
the widths in the report's section 2 reproduce exactly from `config.py:106-110` plus the 14 px
frame, and the heights this gate derives are in the rig table above. Of the brief's four
options, only "make the layout adapt" reaches the failing screens. Changing the default sizes
cannot help: the spec already asks for 1500x950 and `place()` already clamps to the screen, so
on a 1366 or 1280 laptop there are no pixels a different default could claim. A floor with OS
clamping is the pre-`place()` world the module's own docstring documents as windows being CLIPPED
with the status bar never visible (`config.py:76-80`); that is a regression, not an option. The
one size-side decision worth making deliberately is the 40 px stagger, which taxes AGENTS' width
on every screen narrower than 1540 (`config.py:108-109`); AGENTS absorbs it today, and whether
it should keep absorbing it is a design decision this sprint can take with the layout in front
of it.

### Q4. AGENTS: the 51 px clip and the 220 px void

First half, CONFIRMED worth fixing, with the corrected diagnosis from A3: at the 720p laptop
client both chart cards hide 51 px of `overflow: auto` body, which is precisely both charts'
x axis, and the two text cards cut a line mid-glyph (SITUATION 10 px, PIT 8 px, the cut running
through `UCUT 63% -> RUS`, which is the window's next action). Severity P2: reachable by wheel,
but nothing says so, and a mid-glyph cut reads as a rendering fault on the surface whose job is
to be believed. The chart clip traces to the 140 px chart minimum
(`agents.css:424`, the Qt `setMinimumHeight(140)` parity) meeting a row that h=593 cannot fund.

Second half, REFUTED as the same defect as the BESTS hole. The BESTS hole was withheld DATA:
seventeen rank holders existed while ten were shown, and raising the cap closed it. The void
under PIT is exhausted data: SITUATION and PIT are showing everything they have (at 593 they
clip, so content exceeds room; at 833 the same content ends and 221.8 px of column remain,
301.8 at the 913 desktop client, measured this gate). The BESTS remedy has no analogue here
because there is nothing more to show. It is the same VISUAL shape Víctor has called out twice,
so it will read as the same hole to the person who matters; but a fix would have to invent
content or stretch cards, and sprint 9 already measured card-stretching as moving the void
inside the card, which read worse. Severity P3, and the honest options are narrow. Blast radius
of any "give the side column less width/height" rebalance is not priced here.

### Q5. Anything else needing reflow rather than resize

Two places have hit a content minimum, which is the only test that separates reflow from resize:

1. Sub-1470 DATA widths, AFTER the A1 leak is fixed: the traces and the fixed 260 px ring-radio
   column compete for width the same way band 3 and the ring competed for the right column. At
   1266 the traces get 336 px beside a 260 px side column; whether six lanes at 336 px are
   usable was never measured, and if they are not, the side column is the reflow candidate, with
   the window's own worlds-take-turns tab precedent (`data.css:1105-1110`) as the established
   pattern. This gate does not prescribe which; it records that below about 1360 px of client
   the question exists and resizing alone cannot answer it.
2. AGENTS at laptop heights: the 140 px chart minimum against the row h=593 funds is a reflow
   question (which card, if any, gives up its slot at small heights), not a resize question,
   because the minimum is Qt-parity by intent.

Nothing else measured needs it: the tower is fixed by decision, BESTS already reflows by form,
the radio feed scrolls by design, and AGENTS' band absorbs width in PLAN down to 1226 with zero
overflow (measured at every fleet client, `docOverX` 0 in all eight runs).

### Q6. Which of the seven motion items survive the new layout

| # | item | verdict |
|---|---|---|
| 1 | SC chip one-shot pulse | SURVIVES. The chip lives in band 1, outside `.data-main`, so it also escapes the frozen filter (`DataWindow.tsx:132,138`). No layout dependency. |
| 2 | new radio row fade+slide | BLOCKED BY CODE, P1. `RadioFeed.tsx:247` keys rows by index into a newest-first list (`RadioFeed.tsx:137`); a new event shifts every index, every key changes, and React remounts the whole list, so a mount animation fires on ~69 rows at once, not one. The key's comment records why the index is there (legitimate duplicate rows), so the fix is not free; until keys are stable, item 2 cannot ship as designed. This is also the item most plausibly solved BY layout: rows land in a stable slot at the top, and the question of whether an entrance cue is still needed should be re-asked after the keying decision, not before. |
| 3 | pace lap tint decay | SURVIVES, with one constraint from the wire: reveals arrive every ~4.5 s at 1x but the product plays at up to 4x, so a decay longer than ~1 s stacks tints at max speed. The tint must also respect the follow-tail: if the reader has scrolled away, the tint decays off-screen, which is correct behaviour and needs no code. |
| 4 | BESTS flash on a new best | SURVIVES ON DESKTOPS ONLY as scoped. On the two laptop clients the panel is compact and has no ranked rows to flash; if Q1's depth-1 floor lands, the flash regains a home at 16:10 clients but still has none at 1366x768 or 720p. The sprint has to either give the compact form its own flash target or accept that the item is desktop-only. Rows are keyed by driver code (`BestsPanel.tsx:364`), so identity survives re-ranking and the flash can target the changed row. |
| 5 | tab crossfade | COLLIDES WITH THE IMPLEMENTATION. Tabs are conditional renders (`DataWindow.tsx:175-193`); the outgoing world unmounts on switch, so a true crossfade needs both worlds mounted for the fade, which means a second live ECharts instance during every switch. A one-sided fade-in of the incoming world needs no dual mount. Cost of dual-mount not priced here. |
| 6 | button press state | SURVIVES. Tabs, the trace buttons and the tower's clickable rows all keep DOM identity (measured: 20 tower rows, zero remounts over 4 s). |
| 7 | hover on RCM/radio rows | see Q7. |

### Q7. Hover against the 10 Hz repaint

Measured on the live page at 1485x833 over 4 s of streaming: 240 rAF frames, p50 and p95 both
16.7 ms, max 16.8, zero frames over 25 ms, zero long tasks. The 10 Hz repaint leaves the frame
budget effectively empty on this machine, so the COST premise of the question does not survive
measurement: a background-swap hover on a row is noise against that budget. Caveats named
rather than hidden: headless Chromium on the dev machine, under safety car (low churn), and a
green-flag restart with position swaps was not exercised.

Two real hazards remain, and neither is paint cost:

1. Row identity. Radio rows are remounted wholesale on every new event (Q6 item 2), so a hover
   TRANSITION restarts mid-hover when the list moves; a hover style with no transition has no
   restart artifact. Tower rows keep identity but reorder with positions, so a stationary cursor
   inherits a different driver when rows shuffle under it; that is semantics, not performance,
   and it argues for hover styling that reads as "this row" rather than "this driver".
2. The premise "nothing hovers today on either window" is false, and the existing counterexample
   is the best evidence available: AGENTS' tooltip contract already opens on hover, today, via
   mouseenter and a `document.body` portal (`Tooltip.tsx:105-121`, `agents.css:436-445`), and it
   coexists with the same 10 Hz stream without complaint. The claim that is true is "zero CSS
   :hover rules and zero transitions exist" (re-verified this gate: 0 across all four
   stylesheets, and 0 `prefers-reduced-motion`, so the guard the constraints demand does not
   exist yet either).

### Q8. The frozen-board treatment

Three concrete interactions, none previously written down:

1. Which items can even fire while frozen. `frozen` requires a dead feed
   (`DataWindow.tsx:117`), and items 1-4 are data-driven, so they cannot fire under the filter.
   Only the user-driven items reach it: tab switch, button press, hover. The AGENTS twin is
   `agents.css:103`.
2. What the filter does to them. `brightness(0.72)` scales every channel, so a hover tint or
   crossfade tuned on the live board arrives at 72 % of its contrast on the frozen one; sprint
   9's own figure is that this filter costs 28 % of tone-pair distance. A subtle press state
   tuned at full brightness may drop below perception. Worth one look on the real frozen board
   before shipping, not worth engineering around in advance.
3. What escapes it. The AGENTS tooltip portals into `document.body`, OUTSIDE the filtered
   subtree, so it renders at full brightness over a dimmed board and its `position: fixed`
   geometry is untouched by the filter's containing-block rule. That is correct behaviour and
   worth keeping deliberate: any future motion element that portals out of `.data-main` or
   `.agents-body` will read at full strength while frozen, and anything `position: fixed` left
   INSIDE them would be re-anchored by the filter. Today nothing inside is fixed
   (`data.css` has one `position: sticky`, the pace header, which is unaffected).

### Q9. Which defect the sprint spends itself on

The horizontal one, CONFIRMED, and by more than the report argues. Three of six fleet screens
open DATA with working surfaces off-screen and nothing saying so: the ring cut in half and every
radio body unreachable at 720p (199 px, `shots/data-1265x593.png`), messages cut mid-word at
1366x768 (113 px), up to 39 px at 1440x900 when the long note is up. That is a P0 shape: silent
loss of live content on common hardware. The vertical BESTS band, re-priced against the fleet in
A5, is at most 81 px of WASTE on 16:10 laptops and ~49 px at 1366x768; nothing becomes
unreadable, P2.

They do not really compete. The horizontal defect's first fix is small (A1: the min-content
leak), and Q1's answer is a one-constant change plus its red-driven guards. What actually
competes for the sprint is the DECISION under Q5.1, whether sub-1360 clients get a reflowed
right column or an honestly-degraded narrow traces panel, and the motion half. The order that
wastes nothing: settle A1, then Q1, then motion against the settled layout, which is the
sprint's own stated reason for auditing both halves together.

### Q10. Is 1470 px an acceptable minimum

The question dissolves under A1: 1470 is not a property of the layout, it is the longest
header note's width plus the fixed columns, and the floor moves per lap (measured 1465.4 with
the long note, ~1380 with a mid note, 1266 fits with none). So no, 1470 must not be ACCEPTED,
because accepting it means accepting that a line of prose decides whether the ring is on
screen. After the leak is closed, the real floor is the fixed skeleton: 10 + 630 + 10 + traces
+ 10 + 260 + 10. What gives first, in order: the header note already gives (its own ellipsis,
designed in #1068); the traces then absorb everything down to 336 px at 1266, which is
unaudited territory; the tower's 630 and the ring column's 260 are decisions
(`data.css:196,581`) and do not give without reopening them. Whether the product SUPPORTS
sub-1360 clients well, rather than merely without overflow, is the Q5.1 decision; this gate
prices the overflow removal (small, measured live) and does not price the narrow-traces design.

### Q11. The 145 px gap at height 1080

REFUTED as a sprint defect, on population grounds (A5): no default placement can produce a
client taller than ~913, because `WindowSpec` caps at 950 outer and `place()` never grows a
window (`config.py:103-107,123`). At the tallest real client the measured gap is 14 px with
depth 15, and the depth-17 saturation that creates the growing gap is unreachable. The 145 px
exists only after a manual resize past the default, where the panel has genuinely run out of
drivers holding a lap time (17 of 20 at Melbourne), and a gap after data exhaustion is the
"there is no more" state the cap redesign in #1003 deliberately chose. It becomes a sprint
question only if the sprint raises the default height, and then it should be re-measured, not
assumed. Severity P3, no action inside this sprint's scope.

## Prescriptions, separate from the verdicts

Ordered by value against cost. Blast radii are named where they are not priced.

1. Close the A1 min-content leak: `minmax(0, 1fr)` on `.band4`'s first track plus `min-width: 0`
   on `.traces` (`data.css:581,588`). Verified live to remove all document overflow at every
   fleet width with charts resizing correctly. Guard it red-first at 1266x593 with a long note
   forced up, since the defect is content-dependent and a short-note run passes without the fix.
2. Lower `RANKED_FLOOR` to 1 (`BestsPanel.tsx:53`), subtitle still naming the depth, fit guard
   driven red at h=641 and h=673. Closes the 16:10 dead band with no new form.
3. Decide the sub-1360 right column (reflow the side column by tab, or accept narrow traces)
   only after 1 lands and the narrow window can be looked at; the decision needs eyes on a real
   336 px trace stack, which has never existed until now. Unpriced here.
4. AGENTS laptop clients: give the two chart bodies the pace grid's existing edge-fade
   affordance, or spend the reflow decision from Q5.2. The mid-glyph text cuts (10/8 px) are
   the cheaper half and a line-height-aware clip would remove the "broken rendering" read.
   Chart-minimum changes are Qt-parity territory, unpriced here.
5. Motion, after the layout settles: item 2 is blocked until the radio keying decision is made;
   item 5 should be specified as fade-in-only unless someone prices dual-mounted charts; item 4
   needs a compact-form answer or an explicit desktop-only scope; items 1, 3, 6, 7 can proceed,
   every one behind `prefers-reduced-motion`, which currently appears nowhere in the four
   stylesheets.
6. Record somewhere that the sprint memory's BESTS table (41 px compact, 102 px waste) is stale:
   the compact card has been 62 px since the leaders form gained its second wrapped line, and
   the code comment at `BestsPanel.tsx:244-252` already documents the 62. The measurement
   report's re-derivation is right and the memory's numbers should not be quoted again.

## What was tried and could not be broken

- The `place()` arithmetic and the report's fleet width table: recomputed from
  `config.py:106-110`, every value matches given the 14 px frame delta.
- The band table itself: spot-checked live at 1350x673 (compact 62, gap 81), 1485x833 (ranked
  depth 11, card 297, gap 6), 1266x593 (gap 1, overflow 199). All reproduce, and the BESTS
  height formula 99 + 18k fits every logged row.
- The claim that AGENTS has no horizontal defect: pushed at all six fleet widths plus both
  report sizes, `docOverX` 0 in every run, down to 1226.
- The claim that the RACE TRACE tab clips nothing: the probe log's third census column is 0 at
  all 48 sizes.
- The 220 px void under PIT: reproduces at 221.8 px at 1486x833 and grows to 301.8 at the real
  desktop client, exactly as the report's "roughly 220" says.
- The AGENTS tooltip under the frozen filter: the portal lands in `document.body`, so neither
  the containing-block trap nor the dimming reaches it. An attempt to find a `position: fixed`
  element inside the filtered subtrees found none.
- The 10 Hz stream as a performance threat to hover: 240 frames, zero over 25 ms, zero long
  tasks, and both row families keep DOM identity between events. The stream is innocent on this
  machine; the hazards found are semantic (Q7), not paint.
- The zero-motion census in PLAN.md: re-run independently, 0 transitions, 0 keyframes,
  0 animations, 0 `prefers-reduced-motion` across `data.css`, `agents.css`, `qt-base.css`,
  `tokens.css`.

Not exercised, said plainly: the real OS windows (this gate measured the loopback pages, which
are the same transport since #996 but not the same chrome), a green-flag restart's tower churn
under a hovering cursor, the pre-#1068 bundle (the wrap-era behaviour is derived from CSS
semantics and the commit diff, not run), and the literature figures behind Q2, which are quoted
from the sprint's research file unverified.

## Post-close observation

`git status` shows an untracked `src/pitwall/ui/_probe_band.mjs` inside the repo: the measurement
session ran its probe from the ui directory (the recorded copy in `probes/` still carries the
`node _probe_band.mjs` usage line) and left the working copy behind. Not touched by this gate,
which modified no repository file; whoever owns the sprint should delete or adopt it before the
first commit, or it will ride into the branch. The `src/telemetry` submodule entry also predates
this gate.
