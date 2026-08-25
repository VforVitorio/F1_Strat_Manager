# Sprint 4 — where every piece of data comes from

One page, written 2026-08-09, so nobody building bands 1-2 has to rediscover a producer or
re-invent one that already exists. Every row was verified by execution on the real Melbourne 2025
session, not by reading code. Where a number is quoted, it was measured.

**Read the dead ends section too.** Half the cost of this sprint is picking the wrong source, and
three of the wrong sources look exactly like the right ones.

---

## 1. What the wire carries (TCP, ~10 Hz, `src/arcade/app.py::_build_arcade_snapshot`)

Everything here is published by the producer so no consumer re-derives it. Types are in
`src/pitwall/ui/src/lib/bridge.ts`.

| Field | Where it comes from | What it means, and the catch |
|---|---|---|
| `race_order` | `LeaderboardPanel._rank_drivers` — **the arcade panel's own code**, so the wire and the panel cannot drift apart | Best first. **Meaningless until every car has completed a lap**: on frame 0 the field is ordered by millimetres (measured: 6 mm), and through lap 1 each fraction is normalised by that car's OWN first-lap length, reading a P7 start as P2. Render the tower provisional until `laps_completed >= 1` |
| `drivers.<code>.laps_completed` | the crossing map in `gaps.py` | **The reveal carrier**: reveal lap *L* iff `L <= laps_completed`. Monotone forward (swept 20 × 154,173 frames, no counter-example) but **not frame-exact**: 76 of 921 crossings (8.3%) open before the parquet's `Time`, worst 0.463 s. A **rewind un-reveals** — a cache keyed on "seen once" leaks the whole future after one seek to the end |
| `drivers.<code>.progress` | `gaps.progress` | Laps + fraction, the ordering coordinate. **`null` = the telemetry never placed the car** (#886). Never `0.0` for that case any more |
| `drivers.<code>.has_finished` | `gaps.has_finished` ← FastF1's official `Status` (#879) | Chequered flag vs retirement. **OUT is the pair `!active && !has_finished`** — `active` alone reads the winner as OUT. Silent path: a driver missing from the official table falls back to the derived rule with **no** warning |
| `track_status` | `SessionData.track_status_by_lap`, the same source the arcade's own pill reads | **This is the band-1 status strip**: FastF1 TrackStatus digits → SC / VSC / yellow / red. `""` when the loader has no entry, rendered as clear |
| `driver_colors` | `_color_for` ← `src/arcade/palette.py` | RGB per driver. Published precisely so the tower does not hardcode a sixth copy of the palette — five copies have already been found in this repo |
| `drivers.<code>.active`, `.lap`, `.dist`, `.rel_dist`, `.has_position` | the resampled frames | `lap` is an interpolation of a step function; `dist` is race-cumulative and per-car. **Neither is a race-progress axis** — that is what `progress` and `laps_completed` are for |

Cost of the whole addition: **+1.3 KB/tick**, producer compute p95 **126 µs** — 0.1% of the tick.

## 2. What the BULK reader carries (`laps.parquet`, read directly, not over the wire)

| Need | Source | Catch |
|---|---|---|
| Lap times, sectors, compounds, stints | `laps.parquet` for the race, `(927, 35)` on the race checked | Everything the tower and the bests need is here |
| The gap column | the same parquet, **lap-quantised**, labelled at-the-line on screen | A precise-looking wrong number on a fidelity surface is the defect class this project exists to avoid |
| Personal bests | recompute from the **revealed** subset | `IsPersonalBest` is a running flag and safe under masking (18-24 per driver), but the two sequences are not identical |

## 3. Race control messages and radio — they EXIST and are already rendered

Corrected on 2026-08-09. An earlier note in the delivery plan said race-control messages had no
producer. **That was true of one dead field and false of the project.**

| Piece | Producer | Already rendered in |
|---|---|---|
| Race control messages | `src/nlp/radio_runner.py`, from `rcm.parquet` → `RaceState.rcm_events` | The **Radio card** body and tooltip — `agent_formatters.py:280-325`. Ported 1:1 into PITWALL's AGENTS window |
| Radio transcriptions | the same runner → `RaceState.radio_msgs` / `radio_events` | Same card, `<b>Radio</b>` section |
| The radio agent's verdict | N29 → `RadioOutput.alerts` | Same card's headline and status glyph |

Executed, with the real formatter:

```
CARD BODY: ('no alerts', …, [('1 radios · 2 rcm', …),
            ('RCM L23 PENALTY: CAR 44 (HAM) 5 SECOND TIME PENALTY - TRACK LIMITS', …),
            ('NOR INFO: "box this lap the rear is gone"', …)], 'OK')

TOOLTIP:
  <b>RCM</b>
  L23 YELLOW: YELLOW FLAG IN TRACK SECTOR 7
  L23 PENALTY: CAR 44 (HAM) 5 SECOND TIME PENALTY - TRACK LIMITS
  <b>Radio</b>
  NOR INFO: "box this lap the rear is gone"
```

A real Melbourne run reports `radio src corpus/14r·90rcm` — **90 race-control messages** in that
race alone.

**So the sprint-4 question is not "can we have them" but "where do they belong":** the flag STATE
is the status strip (`track_status`, already on the wire); the message TEXT already has a home in
the AGENTS window. If the DATA window wants its own ticker, it needs `rcm_events` put on the wire
— which is one additive field from a producer that exists, not a pipeline.

## 4. Dead ends — sources that look right and are not

| Looks like the source | Why it is not |
|---|---|
| `SessionData.events` | **Always empty. Nothing populates it.** This is the field that made the plan claim race control had no producer. The RCM live in `rcm_events`, from `radio_runner` |
| `get_rival_states` | A **simulation-layer** method. PITWALL cannot reach it |
| `overlays._gap_value` | **Deleted by #844.** It was the hardcoded-55.56-m/s gap |
| `FrameData.dist` for ordering | Race-cumulative and per-car: puts the wrong car in the lead on **37%** of sampled frames |
| `FrameData.lap` for the reveal | A rounded interpolation of a step function: non-monotone on 101 of 2.49 M frames, and it never opens a finisher's final lap |
| `FrameData.rel_dist` for progress | FastF1 leaves it NaN for a whole driver, and the resampler clamps it past a car's last sample — it drew a crashed car as the leader for 68 s |
| `active` alone for OUT | Reads the winner as OUT (#855) |
| A hardcoded TypeScript palette | `driver_colors` is on the wire for exactly this reason |

## 5. Cache and environment facts a sprint-4 session will hit

- `CACHE_VERSION` is **v12**. First launch per GP rebuilds (< 3 min, no re-download).
- `data/` is a **curated** download: only 2025/Melbourne has raw laps. Anything scoring across
  races needs `data_cache.ensure_race()`.
- The orchestrator imports on a clean install since #883; `tests/agents` runs **236 tests** where
  it used to skip two thirds of them.
- The submodule's lite CI installs neither pandas nor fastmcp, so its strategy tests **skip
  entirely**. A green CI there is not coverage.

---

Related: `PITWALL_DELIVERY_PLAN.md` §5 (the sprint's rules) · `src/arcade/gaps.py` (the coordinate
and every candidate refuted on the way) · `~/.claude/plans/pitwall-sprint3/pre-sprint4-gate.md`
(the exit gate that produced most of this page).
