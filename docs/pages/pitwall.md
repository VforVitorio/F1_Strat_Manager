# PITWALL: the strategy surfaces, in web technology

Two windows — **PITWALL · DATA** and **PITWALL · AGENTS** — rendering React
against the arcade's live broadcast. They replaced the two PySide6 windows the
[arcade dashboard (legacy)](#/arcade-dashboard) page describes. Both stacks ran
side by side through the migration so every ported panel could be compared
against the window it replaced while that window still existed; sprint 7
retired the Qt pair, and the comparison baseline survives as committed
screenshots under `documents/dev_docs/migration/pitwall/`.

## Desktop surface, web technology

The distinction matters and it was a deliberate decision.

What renders inside those windows is React, ECharts and CSS in a real Chromium
engine. What surrounds them is a native OS window, because the host process is
Python and [pywebview](https://pywebview.flowrl.com/) embeds the browser engine
in one.

The alternative — a page in a browser, served by a relay — was designed first
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

The URL is printed on startup. It binds the loopback interface only — the
broadcast carries a whole race's live state and there is no authentication —
and it exists for devtools, a second screen, and anything that is not this
desktop.

## Launching

Nothing extra to run. `f1-arcade --strategy` (or `python -m src.arcade.main …
--strategy`) spawns it. To develop against an already-running arcade:

```bash
python -m src.pitwall
```

The UI is a Vite project under `src/pitwall/ui/`. It ships as built static
files, so a source checkout needs `npm install && npm run build` in that
directory once before the windows have anything to show — the entry point says
so rather than opening blank.

## The two windows

### PITWALL · DATA

A status strip across the top, then two columns: the **all-cars world** on the
left and the **own-car world** on the right. That is the zoning a real pit wall
uses — the two live on physically different surfaces there — and it is also the
only arrangement that fits, because four stacked bands need 908 px of a window
that has 790.

**Left: the timing tower**, twenty rows of
`P · # · DRV · GAP · INT · S1 · S2 · S3 · LAST · ST · TYRE · STOPS`, with the
sector colour code every timing screen uses — purple for fastest of the session,
green for a driver's own best, amber for slower than his own. Under it, the
**bests**: S1, S2, S3 and Lap ranked across the field with their percentage off
the leader, and the theoretical lap those three sectors recombine into.

**Right: the own car's lap** as four locked-axis traces against distance — Δ
Time, Speed, Brake, Throttle — ported field by field from the Qt telemetry
panel, plus a shared vertical cursor marking where the car is on the lap, and a
schematic **track ring** placing the whole field by lap fraction.

Four details that are not cosmetic:

- **The sector columns are the lap in progress.** They blank at the line and
  fill as the car crosses each sector, because `laps.parquet` records the
  instant of every crossing. A sector faster than the session's best paints
  purple immediately and joins the bests ranking only when the lap completes,
  which is what a broadcast does.
- **The GAP and INT columns are quantised to the line**, and the header says so
  with an `(L)`. They are the difference of two crossings, taken from the
  official timing table rather than from the replay's own interpolation — which
  means they can differ from the arcade's leaderboard beside them by a few
  hundredths, and PITWALL is the one that matches official timing.
- The rival's traces carry a **BROADCAST** tag. Rival car data is real and
  public, but it is the coarse low-rate channel every team sees, not
  pit-wall-grade telemetry. The Qt window rendered it unlabelled.
- The ring is **schematic, not the circuit**. No track geometry crosses the
  wire, and `rel_dist` is a fraction of the car's *own* lap, so a dot sits a
  median 1.3° and up to 24° (on a pit lap) from its true circuit position. It
  answers *where is everyone* — the pyglet window next to it answers *where
  exactly*.

One consequence of taking the order from the replay and the seconds from the
timing table: they disagree by one place at about 0.7 % of line crossings, both
times within a tenth of a second of each other. When they do, the interval cell
shows a dash rather than a negative number. It is two measurements of the same
moment, not an error in either.

### PITWALL · AGENTS

A 1:1 port of the Qt strategy window: header, the orchestrator card, scenario
bars, the six-tab reasoning panel, and the 3x2 grid of sub-agent cards with
their embedded charts.

It is 1:1 **by construction** rather than by inspection. The host calls the Qt
window's own formatters and hands the React side an already-formatted view, so
the two surfaces cannot describe the same lap differently.

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
timers were measured reading a different frame on 58 % of polls.

**Closing one window does not blind the other.** The client belongs to the
host, not to a window; a window closing only decrements a count, and the last
one out tears the client down.

## Related reading

- [Arcade dashboard (legacy)](#/arcade-dashboard), the Qt windows PITWALL replaced, and
  the wire protocol both stacks read.
- [Arcade quick start](#/arcade-quick-start), running the whole thing.
- [Roadmap](#/roadmap), the v2.6.0 milestone this belongs to.
