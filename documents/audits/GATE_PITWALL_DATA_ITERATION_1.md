# DESIGN ITERATION GATE 1 — PITWALL DATA (telemetry) window

**Date:** 2026-08-18 · **Auditor:** design-iteration gate (second pass; first pass = `GATE_PITWALL_DATA_DESIGN.md`, findings D1-D17, all reported FIXED on this branch)
**Branch:** `fix/pitwall-bests-degrade-honestly`
**Bundle measured:** the one the live loopback server snapshotted at startup —
`dist/assets/data-p-ZIuEWG.js` + `data-jT2Ts2gb.css` + `qt-base-86dNR9Zk.css` + `qt-base-DsVXPXSy.js`.
No rebuild performed; the served bundle is the measured bundle.
**Session measured:** real Melbourne 2025, real producer (`scripts/dev_pitwall_producer.py`),
mid-race (lap 27/57 at gate start, `track_status: 1`, playback 2x), loopback
`http://127.0.0.1:53843/data.html`.
**Inputs:** the 10 `features/data/*.tsx` files, the 15 `lib/*.ts` files (incl. the new
`neutralised.ts`), `styles/data.css` + `qt-base.css`, `session_data.py`, the new
`src/arcade/track_status.py`, live endpoints, and 16 before/after captures in
`~/.claude/plans/pitwall-sprint9/shots/`.
**Probe:** `src/pitwall/ui/scripts/_iter_probe.mjs` (my own copy, extended from the author's
`_probe_s9.mjs`; untracked, delete before PR).

**What this gate judges:** whether the seven fixes made the window BETTER — including whether any
fix overshot, introduced a new imbalance, or is now the loudest wrong thing on screen. Not a
correctness audit.

---

## Checklist (written before any verdict)

- [x] V1. D4/D11 pace grid: `m:ss` degrade + bottom-anchored card — better/worse/mixed.
- [x] V2. D5 BESTS one-line degrade — deliberate density change or broken-looking?
- [x] V3. D6/D2/D17 safety car: rail + shaded band + filled chip — and the band's 0.12 alpha.
- [x] V4. D7 `P.EXIT` out-lap relabel.
- [x] V5. D10 own-trace z-order over rival dashes.
- [x] V6. D9/D15/D16 radio scroll, blind-list retirement filter, 26px targets.
- [x] V7. D1 STOPS = tyre-set transitions.
- [x] A1. Author suspicion: SC band too heavy at 0.12 over a third of the plot — measure.
- [x] A2. Bottom-anchored card early-race: small panel + tall air column.
- [x] A3. Narrow-client `m:ss`: is the grid still worth its area without tenths?
- [x] A4. Rail vs `is-t3` amber text distinguishability at 9 px.
- [x] B1. Four telemetry charts' x-axis labels at 1265x593 (untouched).
- [x] B2. Track ring nose-to-tail under SC (untouched).
- [x] B3. Planned #982 dead-producer treatment — right shape?
- [x] B4. What is now the loudest wrong thing on the window.

Findings and verdicts appended below as they are confirmed.

---

## Part 1 — Verdict per change

### V1 · D4/D11 — the pace grid's `m:ss` degrade and the bottom-anchored card — **BETTER**, with one caveat on the captures

**The narrow client is transformed.** Before (`pace-1265x593.png`): `1:59.1:57.2:24.` and
`IN PIIN PIIN PI` — truncated glyph soup that actively misread. After (`after-pace-1265x593.png`
and live): clean `1:31` cells, `PIT`, `EXIT`, and the subtitle says out loud
*"times to the second at this width"*. Measured on the SERVED bundle at 1265x593
(`_iter_probe.mjs`): first data column 27 px, coarse form active, the only non-numeric cell
texts on the whole grid are `EXIT` and `PIT` — zero truncation. The honest-subtitle move
(`RacePaceGrid.tsx:162`) is exactly right: the degrade is announced, not silent.

**Is the grid still worth its area without tenths (author question A3)? Yes.** The panel's
subtitle states its own contract — "colour ranks each lap against itself" — and the tone channel
carries the ranking untouched; `m:ss` only loses intra-second discrimination between cells the
colour already ranks. The alternative `ss.d` was rightly rejected (`racePace.ts:155-163`): a
2:19 SC lap rendering `19.7` beside a green `59.4` misreads magnitude, and a cell that can be
misread is worse than one that is openly coarser.

**Caveat — two of the after-captures predate the D7 fix.** `after-pace-1485x833.png` and
`after-pace-1265x593.png` render the out-lap as `OUT` (crop verified pixel-level), while
`sc-rail-pace.png` and the served bundle render `P.EXIT`/`EXIT` (live probe: cell texts are
`EXIT`/`PIT` only). Anyone judging D7 from those two files would call it unfixed. Refresh them
before they mislead a third pass.

### V1b · The bottom-anchored card early-race (author suspicion A2) — **BETTER, keep it, do not touch it**

At lap 7 (`sc-rail-pace.png`) the card is a ~140 px strip at the bottom of a ~615 px void, and
at zero laps it is a header strip alone. That looks bottom-heavy — and it is still the right
answer, for the reason the fix names (`RacePaceGrid.tsx:133-144` + `data.css:955-963`): the
newest lap is the row every decision is about, and it now sits at a FIXED height from lap 1 to
lap 57 instead of migrating ~12 px per lap for 30+ laps. A fixed eye line for the whole race is
worth fifteen laps of empty column, and the void is honest — the grid literally has nothing to
show there yet. Every alternative costs more: void inside a full-height card was tried and
correctly reverted (426 px of dead space INSIDE a bordered panel reads as a rendering bug);
newest-lap-first row order would pin the row at the top but make the grid's vertical time axis
run backwards against the race trace's horizontal one; a top-aligned card with a mid-race
switch to anchored reintroduces the migration exactly once, at the worst moment. Accept the
early-race air.

One stale capture here too: `after-pace-earlyrace.png` shows the header at the TOP and the code
row at the BOTTOM of a full-height card — that is the reverted table-anchor build, not what
ships (`.pace` is content-sized, `align-self: end`, so the header rides with the card at the
bottom). Refresh it.

### V2 · D5 — BESTS one-line degrade — **BETTER**, and it is honestly two lines, not one

`bests-final-1265x593.png` + live probe at 1265x593: the card renders the `leaders` form,
62 px tall in the 63 px slot, `BESTS leaders · S1 VER 31.103 · S2 PIA 18.319 · S3 NOR 37.812 ·
LAP NOR 1:27.695` with `THEO 1:27.234` right-aligned below. It reads as a deliberate density
change, not a broken panel — three things sell it: the word `leaders` where the subtitle was
(the header says what got dropped), the unchanged typography (same field labels, same purple
weight), and `is-theoretical { margin-left: auto }` (`data.css:383-385`), which makes the
wrapped THEO read as a footer rather than as overflow. Compare the BEFORE at this client: rows
2-3 and THEORETICAL silently amputated mid-panel. No contest.

**But the code's own account of it is wrong at this exact client.** `BestsPanel.tsx:92-96`
says the compact form "puts the title on the same line as the values... One line has 24 px of
air", and `data.css:359-363` pins `line-height: 17px` against a one-pixel overhang. Measured:
`.bests-leaders` is **40 px tall — it wraps to two lines at 1265** (THEO top 531 vs first
leader top 509), and the card is 62 in 63 — the 24 px of air is spent. It fits, and it looks
fine, but the margin the comment claims does not exist, and the card is `overflow-y: auto`
(`data.css:303`) with globally hidden scrollbars — one more wrap (a slightly narrower window,
a wider font fallback) and a third line slides under the fold in silence, which is D5's
original failure mode one wrap away. See finding N3.

### V3 · D6/D2/D17 — the safety-car treatment — **MIXED: right design, two wrong numbers**

**The rail (D6): right.** A 2 px amber border on the lap-number gutter
(`data.css:1069-1071`), keyed in the subtitle (`pace-legend-rail`, only when a marked lap is on
the panel — `RacePaceGrid.tsx:165-170`), labelled on hover. On `sc-rail-pace.png` contiguous
neutralised laps merge into one continuous rail — clean, and clearly a MARK rather than a value.
**Author question A4 (rail vs `is-t3` amber text at 9 px): distinguishable in practice.** Same
hue, different channel — the rail is a positional line in the gutter, `is-t3` is glyphs inside
a cell; on the real capture nothing invites confusion, and the stylesheet's argument
(`data.css:1065-1068`) matches what the pixels show. Live probe: 8 rail rows at lap 33, legend
present.

**The filled chip (D2): right.** Before (`state-dead.png` header): GREEN and SAFETY CAR were
both 18 px outline chips — the race's most decision-dense state differed from its calmest by
nothing but the word. After (`sc-rail-pace.png`, `sc-filled-chip.png`, live): the chip fills
with the wire's own colour (rgb(255,140,0), decoded from `palette.py` by the producer —
`StatusStrip.tsx:90-98`), dark text, AA-checked per label (`data.css:97-107`). Weight now
separates race state from annotation, which also quietly resolves most of D17: the comment at
`data.css:872-880` now states the five ambers honestly instead of claiming one. Beside
PROVISIONAL (outline amber) the filled chip clearly dominates — the hierarchy reads.

**The band (D6 on the race trace): right idea, wrong alpha AND a live-edge defect.** Detail in
findings N1/N2 and the disagreement section: 0.12 composites to rgb(51,38,45) over the
rgb(24,22,51) panel — the red channel more than doubles and OVERTAKES blue, so the surface
flips from cool navy to warm brown and reads as a solid block (the author is right); and a
band whose revealed range is so far a single lap (live SC, lap 33) has **zero width** — it
paints nothing while its label floats over the driver endLabels at the plot's right edge
(executed evidence: live capture at lap 35, second SC).

### V4 · D7 — `P.EXIT` — **BETTER, done, nothing to add**

Wide: `P.EXIT` in WARNING amber (`racePace.ts:299`), coarse: `EXIT`; the in-lap keeps DANGER
red. The tower's `OUT` = retirement vocabulary is no longer contradicted three rows away — and
the comment documents the collision it kills. Verified on the served bundle (probe cell texts)
and on `sc-rail-pace.png` (lap-5 full-field `P.EXIT` row under the SC pit-lane pass — the exact
row that used to read as seventeen simultaneous retirements).

### V5 · D10 — own trace painted last — **BETTER, done**

`TraceChart.tsx` series order is now rival (dashed, first) then main (solid, last). Before
(`traces-full-1485x833.png`): Speed read as one amber dashed line with slivers of blue under
it. After (`iter1-traces.png`): the solid pit-wall-grade line defines every shape and the
broadcast dashes peek from behind — the z-order now matches the tier hierarchy the dash
semantics claim, and matches the race trace's own convention.

### V6 · D9/D15/D16 — radio scroll, blind list, hit targets — **BETTER, all three verified live**

- Radio (`data.css:830-834`): `overflow-y: auto`; probe: scrollHeight 1553 vs clientHeight 120,
  scrollable. 36-events-below-an-unscrollable-fold is dead.
- Blind list: live at lap 35 with FOUR `OUT` rows in the tower (ALO, SAI, DOO, HAD) the ring
  shows **no** `NO POSITION:` caption — the telemetry alarm is no longer spent on retirements.
  (At lap 1, `sc-filled-chip.png` still shows `NO POSITION: HAD` — correct, HAD is not yet
  marked out on the wire at that instant; honest, not a regression.)
- Targets: probe at both clients — TRACES/RACE PACE/RACE TRACE tabs and the race trace's
  LEADER/FIELD/NOR strip all exactly 26 px tall. D16's 15-22 px targets are gone.

### V7 · D1 — STOPS counts tyre-set transitions — **BETTER, and Melbourne is the proof**

`session_data.py:215-229` (`_tyre_stops`): counted off the tyre-set evidence, not pit
entries, with the docstring naming why Melbourne is the strongest case — the SC led the field
through the pit lane on laps 2, 3 AND 4 with no tyre work. Before: every classified car showed
STOPS 3 by lap 24 next to a TYRE cell reading `I 25` — the tower printed its own contradiction.
After (lap 23 capture + live): 0 for the field, 1 for the cars that actually changed (ALB, STR,
LAW, OCO, BEA), consistent with their TYRE ages. The column now answers the question a
strategist asks of it.

---

## Part 2 — New findings

### N1 · P2 — The neutralised band at 0.12 flips the plot surface from cool to warm and reads as a solid block; 0.08 keeps it a tint

**Where:** `src/pitwall/ui/src/lib/chart.ts:55` (`NEUTRALISED_BAND = "rgba(245, 158, 11, 0.12)"`),
consumed at `RaceTraceChart.tsx:152-168`.

**Executed evidence.** Pixel-sampled `iter1-racetrace.png` (5 points inside the band, 5 outside):
band composite **rgb(51,38,45)** over panel **rgb(24,22,51)**. The red channel goes 24 → 51
(2.1x) and OVERTAKES blue (51 → 45): the surface's hue flips from navy to warm brown-pink, so a
third of the plot reads as a different MATERIAL, not a tinted region of the same one. Rendered
the alpha ladder over the real panel colour with the real WARNING amber: the red-over-blue
crossover sits at alpha ≈ 0.105. Below it the band stays a cool surface with a warm cast; at
0.12 it is past the flip. By the flag the effect covers ~34% of the axis permanently (bands
1-7, 33-41, 46-51 — `neutralised.ts:46-47`).

**What a strategist loses:** the twenty lines are the data and the band is an annotation, but at
0.12 the band has more visual mass than any line inside it — the eye parses the brown block
first on every glance at the panel.

**Prescription:** `rgba(245, 158, 11, 0.08)` — composite rgb(42,33,48), blue still dominant,
clearly visible as a band (verified on a rendered swatch over the real panel colour), and the
`SAFETY CAR` label plus the pace grid's keyed rail carry the semantics, so the fill does not
need to shout. 0.07 also works; do not go to 0.05 (barely distinguishable). One constant, no
layout risk. This respects settled rule 2 — the treatment stays a translucent amber fill.

### N2 · P1 — At 1265x593 the four telemetry x-axes render as an unbroken digit string: five labels of ~30 px on a ~120 px axis

**Where:** `src/pitwall/ui/src/lib/chart.ts:129-155` (`valueAxis`, `axisLabel` fontSize 10, no
formatter), used by all four charts in `features/data/TraceChart.tsx` with x locked to
`[0, circuit_length]`.

**Executed evidence.** Live probe at 1265x593, TRACES tab (the DEFAULT tab): each `.trace-plot`
is **152 px wide**; after grid margins the axis span is ~120 px. Five labels `1,000`…`5,000`
at 10 px mono are ~30 px each — 150 px of glyphs on 120 px of axis. `bests-final-1265x593.png`
shows the result on all four charts: `1,0002,0003,0004,0005,000`, one unreadable run. The wide
client (203 px plots, `traces-full-1485x833.png`) is tight but separated.

**What a strategist loses:** the distance scale on every telemetry chart at the laptop client —
and worse, the axis looks like a rendering fault, which taxes trust in the three panels above
it that are correct. This is the loudest broken-looking thing at 1265 and it is on the tab the
window opens on.

**Prescription:** a formatter on the x `valueAxis` used by TraceChart:
`axisLabel.formatter: (v) => \`${v / 1000}k\`` → `1k 2k 3k 4k 5k`, ~12 px per label at both
clients, self-consistent with the axis name "Distance (m)" becoming "Distance" or "Distance
(km)" if preferred. No measurement machinery needed; it also reads better at 1485. (An
`interval` that drops to three labels is the fallback if the k-form is unwanted.)

### N3 · P3 — The BESTS compact form is two lines, not the one line its comment claims, and its safety margin is spent

**Where:** `BestsPanel.tsx:92-96` (comment: "puts the title on the same line as the values…
One line has 24 px of air"), `data.css:353-363` (`flex-wrap: wrap`, `line-height: 17px`),
`data.css:303` (`overflow-y: auto` on the card, scrollbars hidden globally).

**Executed evidence.** Live probe at 1265x593: `.bests-leaders` height **40 px** — two flex
lines (first leader top 509, THEO top 531); card 62 px in a 63 px slot. It FITS, and the
right-aligned THEO happens to read as a deliberate footer — but the 24 px of air the comment
budgets does not exist, and the failure mode behind D5 (content sliding under an unscrollable,
unannounced fold) is one wrap away: any client a few px narrower than 1265, or a wider
fallback font, wraps a second entry and pushes THEO below the fold in silence.

**What a strategist loses today:** nothing — both settled client sizes render correctly.
This is a margin-and-honesty finding, not a visible defect.

**Prescription:** (a) correct the comment — it currently documents a mechanism the panel does
not have, which is the defect class `feedback_errors_gates_caught_in_me` names; (b) cheap
guard: at compact, if `.bests-leaders` scrollHeight exceeds the room, drop the `.bests-value`
spans and keep `S1 VER · S2 PIA · S3 NOR · LAP NOR · THEO 1:27.234` — codes are the panel's
glance answer and the third degrade step continues the existing ladder (ranked → leaders →
codes). (b) can wait; (a) should not.

### N4 · P2 — A LIVE safety car paints a zero-width band: the range from==to renders nothing while its label floats over the driver codes

**Where:** `RaceTraceChart.tsx:163-166` (`data: neutral.map((band) => [{ name, xAxis:
band.from }, { xAxis: band.to }])`), ranges from `neutralised.ts:49-58`.

**Executed evidence.** Live capture at lap 35 (second SC, only lap 33 revealed as neutralised
so far, `from == to == 33`): the 1-7 band is shaded, but at lap 33 there is NO shading — a
zero-width markArea — while its `SAFETY CAR` label renders at the plot's right edge on top of
the NOR/PIA endLabels (`~/.claude/plans/pitwall-sprint9/shots/iter1-live-trace-2bands.png`). The moment the
band matters most — an SC out RIGHT NOW — is the moment it is invisible and its label is noise.
A second, quieter cost of the same encoding: a lap is a POINT on this axis, so even a wide band
leaves its boundary laps' data points ON the band edge rather than inside it.

**What a strategist loses:** the annotation exactly during the live neutralisation it exists
for; plus a label collision in the only identification zone the chart has.

**Prescription:** pad the range to the lap's cell — `{ xAxis: band.from - 0.5 }, { xAxis:
band.to + 0.5 }`. A one-lap band becomes one lap wide, boundary laps sit inside their band, and
nothing else changes. Additionally set the markArea `label.position` to `"insideTopLeft"` so a
band ending at the plot's right edge keeps its label at the band's left, away from the
endLabels. Two lines, no new constants, respects settled rule 2.

### N5 · P3 — Two columns with zero revealed laps (HAD, DOO) occupy grid width for the whole race — and dropping them would NOT buy the tenths back

**Where:** `racePace.ts` `stableColumns` (grid columns are the full field), visible in every
pace capture: HAD and DOO are empty for all 55 rows (`after-pace-1485x833.png`); SAI keeps
early-lap rows so his column earns its place.

**What a strategist loses:** ~2 columns × 27.75-38.75 px of the panel's scarcest axis, spent on
cars that never set a revealed lap. Honest arithmetic before anyone oversells the fix: at 1265,
dropping both gives 18 columns of ~30.8 px — still under the ~35 px the fine (`0:00.0`) form
needs, so the coarse form STAYS; the gain is only decluttering. Rule if taken: drop a column
only at zero revealed laps (a car that retires mid-race must keep its column and its history).
Low value, low risk; fine to leave.

### N6 · P3 — The safety car wears two different ambers: the chip fills with the wire's rgb(255,140,0), the band/rail/legend use WARNING #f59e0b

**Where:** `StatusStrip.tsx:97-98` (background from `track_status_color`) vs `chart.ts:55` and
`data.css:1070/1078` (WARNING). Both trace to `palette.py` names, so settled rule 7 is
respected, and at a glance the hues are indistinguishable — this costs nothing today. It is
noted so the pair cannot drift apart silently: if the producer's SC state colour ever changes,
the chip moves and the band does not, and the window would then say "safety car" in two
colours. A one-line comment at `NEUTRALISED_BAND` naming the pairing is enough.

---

## Part 3 — What to change next, in order

1. **N1 — band alpha 0.12 → 0.08** (`chart.ts:55`). One constant; the biggest visual win at
   1485 per character of diff.
2. **N2 — x-axis `formatter: v => v/1000 + "k"`** on the four telemetry charts. Kills the
   digit soup at 1265; improves 1485 too.
3. **N4 — pad neutralised bands ±0.5 lap and move the markArea label to `insideTopLeft`**
   (`RaceTraceChart.tsx:156-166`). Makes the live-SC band exist.
4. **N3a — fix the false "one line / 24 px of air" comment** (`BestsPanel.tsx:92-96`).
5. **Refresh the three stale captures** (`after-pace-1485x833.png`, `after-pace-1265x593.png`,
   `after-pace-earlyrace.png`) so the next pass doesn't re-litigate D7/D11 from pre-fix pixels.
6. **#982 dead-producer treatment** — the planned shape is right; see Part 4a notes.
7. (Optional) N3b codes-only third degrade; N5 zero-revealed-lap column drop; N6 pairing
   comment.

All of 1-5 are safe under the no-rebuild constraint of this gate — they are prescriptions for
the NEXT build, none was applied.

---

## Part 4 — Judgments on the untouched surfaces

**4a · The planned #982 dead treatment (`PLAYBACK —` + stale filter on both columns +
non-transient `DATA FROZEN · last tick L28`) — right shape.** `state-dead.png` shows why all
three legs are needed: today a dead producer leaves GREEN chip + `PLAYBACK 2x` + a full board
of confident numbers, and the only tell is one small `Disconnected` chip. Four additions to the
plan, all in its spirit: (1) the bottom status line reads `lap 28 · live` — it must flip with
the same signal, or the window contradicts its own banner; (2) the track-status chip must go
hollow/neutral when frozen — a dead feed cannot assert GREEN (the track may be red-flagged
for all it knows), and a frozen FILLED SAFETY CAR chip would be worse; (3) the stale filter
should be desaturation (~0.5) plus a mild dim, never blur or opacity below ~0.55 — the last
known state is still operationally useful and must stay readable; (4) `DATA FROZEN · last tick
L28` belongs in the header strip where the chips already live, so the reader who only checks
the top-left gets it.

**4b · The track ring under a live SC (`state-safetycar-lap1.png`) — leave it.** Nineteen dots
collapsing into a nose-to-tail caterpillar IS the safety-car message; the gestalt is the
information, and identity was never the ring's job (settled rule 1 — ten colours, twenty cars;
only NOR/PIA are labelled and they stay on top). Any de-overlap (jitter, radial fanning) would
draw a field spread the race does not have — worse than the overlap it fixes.

**4c · What is now the loudest WRONG thing on the window.** At 1485: the band's brown mass
(N1). At 1265: the axis digit soup (N2). Both are in the next-list. After those two land, the
loudest remaining wrong thing is not from this iteration at all — it is D12, the six
sub-3:1-contrast driver codes (VER/LAW at 1.88:1), already tracked in #976; worth pulling
forward, because identity is the tower's whole answer and every fix in this pass sharpened the
panels AROUND it.

---

## Part 5 — What I tried to break and could not

- **The coarse/fine flip-flop.** `fineFormFits` (`RacePaceGrid.tsx:66-76`) always measures the
  FINE form in the cell's computed font regardless of what is rendered — the classic
  measure-what-you-just-rendered oscillation is structurally absent. Code-read plus live
  verification at both clients (fine at 1485, coarse at 1265); I did not drive a continuous
  resize, so "no oscillation" is asserted for the two settled sizes, not for every width.
- **The BESTS ranked/compact hysteresis.** The ranked height is latched once
  (`BestsPanel.tsx:55-79`) and compact never re-measures itself — the 63<153→compact,
  63>=54→ranked loop the comment describes cannot occur. Verified the compact form live at
  1265 and the ranked form at 1485.
- **Rail vs `is-t3` text confusion (author question A4).** Same hue, different channel; on the
  real captures at 9 px I could not construct a glance that misreads one for the other — the
  rail is a continuous gutter line, amber text is glyphs inside ranked cells.
- **STOPS vs TYRE contradiction.** Searched every capture and the live tower for a car whose
  STOPS disagrees with its TYRE age under the new definition; none found (the before-capture
  shows the old contradiction on every classified row).
- **The visible-range header (#949 fix).** Live at 1265: header `LAPS 1-33 of 57` with exactly
  laps 1-33 rendered and all rows inside the scroller; at the 56-lap capture the header
  windows to `19-55` and the newest row is bold and last. Could not make it lie.
- **The filled chip's AA claim.** Re-derived SAFETY CAR: rgb(255,140,0) text `--qt-bg` #121127
  → 7.92:1 against the claimed 7.93 (`data.css:101-103`). The one label I checked matches; the
  other three were not re-derived.
- **The neutralised twin.** The rail and the trace band read the same `neutralisedLaps` module
  (`neutralised.ts:30`, `RaceTraceChart.tsx:92-95`, `RacePaceGrid` via `racePaceGrid`), so the
  two panels cannot disagree about which laps were neutralised — the twin defect this repo pays
  for most often is structurally closed here.

---

## Part 6 — Where I disagree with the author, plainly

- **The band: you are right, and the number is 0.08.** The capture is not misleading you — the
  measured composite flips warm at 0.12 (red overtakes blue at alpha ≈ 0.105). But alpha is
  only half the finding: the LIVE band is zero-width (N4), and if you only dim the fill the
  live-SC case gets worse, not better. Land N1 and N4 together.
- **The bottom-anchored card: your worry is misplaced — keep it exactly as is.** The early-race
  void is the honest cost of a fixed newest-lap eye line, and every alternative (including the
  one you already tried and reverted) costs more. Do not spend another commit on it.
- **BESTS "one line": it is two lines at 1265 and that is fine — but the comment says one, and
  a comment that mis-describes its own mechanism is this repo's named defect class. Fix the
  comment, not the panel.**

---

## Summary

- V1 (D4/D11 pace degrade + anchor): **BETTER** · V1b (anchor early-race): **keep, don't touch**
- V2 (D5 BESTS): **BETTER** (honestly two lines; comment wrong)
- V3 (D6/D2/D17 SC): **MIXED** — rail right, chip right, band right idea / wrong alpha + zero-width live band
- V4 (D7 P.EXIT): **BETTER** · V5 (D10 z-order): **BETTER** · V6 (D9/D15/D16): **BETTER** · V7 (D1 STOPS): **BETTER**

**P1:** N2 (telemetry x-axis digit soup at 1265, default tab).
**P2:** N1 (band alpha flips the surface warm; 0.08), N4 (zero-width live SC band + label collision).
**P3:** N3 (BESTS two-line reality vs one-line comment; margin spent), N5 (two zero-lap columns), N6 (two ambers for one state).

**Counts:** P0 0 · P1 1 · P2 2 · P3 3. Probe left at `src/pitwall/ui/scripts/_iter_probe.mjs`
(untracked; delete before PR), alongside the author's `_probe_s9.mjs`.
