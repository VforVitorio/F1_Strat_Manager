# PITWALL: the strategy surfaces, in web technology

Two windows, **PITWALL · DATA** and **PITWALL · AGENTS**, rendering React
against the arcade's live broadcast. They replaced the two PySide6 windows the
[arcade dashboard (legacy)](#/arcade-dashboard) page describes. Both stacks ran
side by side through the migration so every ported panel could be compared
against the window it replaced while that window still existed; the Qt pair
was later retired, and the comparison baseline survives as committed
screenshots under `documents/dev_docs/migration/pitwall/`.

## Desktop surface, web technology

The distinction matters and it was a deliberate decision.

What renders inside those windows is React, ECharts and CSS in a real Chromium
engine. What surrounds them is a native OS window, because the host process is
Python and [pywebview](https://pywebview.flowrl.com/) embeds the browser engine
in one.

The alternative (a page in a browser, served by a relay) was designed first
and rejected for a concrete reason: **a browser cannot open a raw TCP socket**,
so a page would have needed a FastAPI WebSocket relay in front of the arcade's
broadcast, which makes the backend a runtime dependency of a surface that
otherwise has none. A Python host opens the socket itself.

That said, the same pages are **also served over loopback**. The host already
holds the socket and the payload, so handing that payload to a browser costs a
small HTTP server rather than a relay:

```
INFO __main__: Also serving the same windows at
      http://127.0.0.1:62042/data.html and http://127.0.0.1:62042/agents.html
```

The URL is printed on startup. It binds the loopback interface only (the
broadcast carries a whole race's live state and there is no authentication),
and it exists for devtools, a second screen, and anything that is not this
desktop.

## Launching

Nothing extra to run. `f1-arcade --strategy` (or `python -m src.arcade.main …
--strategy`) spawns it.

The first launch of any given race builds its replay telemetry, which takes
minutes, so the menu reports the stages while a worker thread does the work.
`f1-prefetch --year 2025` runs that same preparation for a whole season ahead
of time, and skips the rounds already cached. That skip tests for the file, not
for its version, so after a release that changes the replay format the stale
rounds read as cached; `--force` rebuilds them.

To develop against an already-running arcade:

```bash
python -m src.pitwall
```

The UI is a Vite project under `src/pitwall/ui/`. It ships as built static
files, so a source checkout needs `npm install && npm run build` in that
directory once before the windows have anything to show, and the entry point
says so rather than opening blank.

## The two windows

### PITWALL · DATA

A status strip across the top, then two columns: the **all-cars world** on the
left and the **own-car world** on the right. That is the zoning a real pit wall
uses (the two live on physically different surfaces there), and it is also the
only arrangement that fits, because four stacked bands need 908 px of a window
that has 790.

**Left: the timing tower**, twenty rows of
`P · # · DRV · GAP · INT · S1 · S2 · S3 · LAST · ST · TYRE · STOPS`, with the
sector colour code every timing screen uses: purple for fastest of the session,
green for a driver's own best, amber for slower than his own. Under it, the
**bests**: S1, S2, S3 and Lap ranked across the field with their percentage off
the leader, and the theoretical lap those three sectors recombine into.

**Right: a tab strip over three panels**, because the column has 825 px of
width to give and sharing it costs one of them too much: with the track ring
still mounted, the pace grid's columns narrow enough that 1,101 of 1,140
cells clip. The panels take turns instead.

**TRACES**, the tab open by default, is the own car's lap as four locked-axis
traces against distance: Δ Time, Speed, Brake, Throttle, ported field by field
from the Qt telemetry panel, plus a shared vertical cursor marking where the
car is on the lap. Beside it, a schematic **track ring** placing the whole
field by lap fraction, and the **radio feed**: race control messages and
driver radio in one list, newest line on top, the header naming the total so
the cut at the bottom is never silent. The feed carries the whole field
rather than only the pinned car, a rival's line marked `BROADCAST`, the same
tag the rival's traces carry. A real team receives the public broadcast radio
feed too, so showing it is fidelity rather than a privilege the window grants
itself.

**RACE PACE** is a grid, one column per driver and one row per lap: how quick
each lap was. **RACE TRACE** is a chart of the same laps read the other way,
where everyone is relative to a reference over the whole race: one cut down
it at a given lap gives every gap in the field at that moment, a question
the grid cannot answer.

Splitting the two across tabs rather than stacking them repeats the column's
own arithmetic: a grid squeezed to half this height stops showing enough laps
to read as a history, and a trace squeezed the same way, to roughly 300 px,
stops resolving the gaps it exists to show.

Shipping both, with the grid one click from the default tab, follows what the
pit-wall research behind this window found and what it could not settle.
Across seven sources and six photographs of a real SBG/Catapult RaceX client,
not one race-trace line chart appears on a wall, while the literature names
the gapper plot the central strategy tool, and the pace grid takes up more
screen area than anything else in the photographs. The photographs cover the
in-session view, so they cannot rule out a trace living on an analysis tab;
what they establish is which of the two a wall keeps in front of it during a
race. RACE TRACE stays because the question it
answers has no other tab that can answer it.

Four details that are not cosmetic:

- **The sector columns are the lap in progress.** They blank at the line and
  fill as the car crosses each sector, because `laps.parquet` records the
  instant of every crossing. A sector faster than the session's best paints
  purple immediately and joins the bests ranking only when the lap completes,
  which is what a broadcast does.
- **The GAP and INT columns are quantised to the line**, and the header says so
  with an `(L)`. They are the difference of two crossings, taken from the
  official timing table rather than from the replay's own interpolation, which
  means they can differ from the arcade's leaderboard beside them by a few
  hundredths, and PITWALL is the one that matches official timing.
- The rival's traces carry a **BROADCAST** tag. Rival car data is real and
  public, but it is the coarse low-rate channel every team sees, not
  pit-wall-grade telemetry. The Qt window rendered it unlabelled.
- The ring is **schematic, not the circuit**. No track geometry crosses the
  wire, and `rel_dist` is a fraction of the car's *own* lap, so a dot sits a
  median 1.3° and up to 24° (on a pit lap) from its true circuit position. It
  answers *where is everyone*. The pyglet window next to it answers *where
  exactly*.

One consequence of taking the order from the replay and the seconds from the
timing table: they disagree by one place at about 0.7% of line crossings, both
times within a tenth of a second of each other. When they do, the interval cell
shows a dash rather than a negative number. It is two measurements of the same
moment, not an error in either.

### PITWALL · AGENTS

Four strata, top to bottom: header, the decision band, the agent grid, status
bar. The band answers one question per module across the line the eye lands on
anyway, left to right in the order a reader asks them, which is what the
orchestrator is doing, why, on what evidence, and what happens next.

**The Qt lineage ends at the layout, and only at the layout.** The host formats
nothing on the React side: every headline, body line and colour is produced by
`src/pitwall/agent_formatters.py`, the Qt window's own formatting code, and
`agents_view/builder.py` states that as its first invariant. Chart colours come
from `src/arcade/palette`, which the replay uses too. The Qt window itself is
gone, so the rule now buys a single source for the strings rather than agreement
between two surfaces. What the port inherited and then dropped is the geometry: a header
strip over a 540 / 740 horizontal split, decision in the left column and a 3x2
card grid in the right. That split made the decision a peer of the agent grid,
two territories with no reading order, and put the most important content on the
window in the same column as a reasoning panel measured at 1.9% ink.

The grid places its cards by name rather than in reading order, because three of
the six want a shape reading order does not give them:

```
pace   tire   side
radio  exit   rag
```

**PIT EXIT is the one output with no Qt counterpart.** It answers the question a
pit wall asks before every stop: box on this lap, and what position does the car
rejoin in, with which cars either side. It reads `P1 → P3` under an *if we box
now* header, the car ahead and the car behind named with the gap to each. The
number is the one the projection layer is graded on, 86.1% within one position
over 552 real green-flag stops of 2025, because the card computes at the same
two-lap horizon the ground truth measures rather than at the five-lap one the
strategy scoring uses.

The card is a hypothetical and says so in its own header, which is what keeps it
readable on a STAY_OUT lap. Suppressing it there was designed and rejected: it
would idle the card on exactly the laps somebody is asking whether to box, which
is when the readout earns its space. It carries no identity colour either. The
live call wears ACCENT one card up, and a branch that is not happening must not.

RADIO gave up the second of its two columns to make room, so the bottom row runs
three cards wide. That is not cosmetic. At one column a radio transcript no
longer fits on one line and the body wraps rather than clips, so the height comes
out of the charts above it, which is why `agent_formatters.BODY_LINE_LIMIT`
budgets the lines.

Before the first view arrives the window renders what the Qt window shows at
startup rather than a spinner. The scenario scores are the one field that no
longer follows Qt: they read `--` where it painted `0%`, because before the first
tick nothing has been simulated and 0% is a measurement.

## How it is wired

```
arcade process                    pitwall process
  pyglet replay                     ArcadeStreamClient  (ONE socket)
  TelemetryStreamServer  ──TCP──▶     └─ latest payload slot
  127.0.0.1:9998                          │
                                          ├─ window: DATA    ┐ js_api
                                          ├─ window: AGENTS  ┘ get_tick(since_seq)
                                          └─ loopback HTTP    /api/tick
```

**One client, however many consumers.** Both windows and any browser tab read
through the same `get_tick(since_seq)`, and the sequence is what makes them
agree: against a blind latest-payload slot, two pollers on independent 10 Hz
timers were measured reading a different frame on 58% of polls.

**Closing one window does not blind the other.** The client belongs to the
host, not to a window; a window closing only decrements a count, and the last
one out tears the client down.

## Related reading

- [Arcade dashboard (legacy)](#/arcade-dashboard), the Qt windows PITWALL replaced, and
  the wire protocol both stacks read.
- [Arcade quick start](#/arcade-quick-start), running the whole thing.
- [Roadmap](#/roadmap), the v2.6.0 milestone this belongs to.
