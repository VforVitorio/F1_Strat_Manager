# GATE A — PITWALL v2 architecture: DATA PLANE and RUNTIME CORRECTNESS

> **Gate**: adversarial, read-only. **Date**: 2026-08-07.
> **Target**: `documents/research/PITWALL_V2_ARCHITECTURE.md` (design, no code written yet).
> **Lens**: the DATA PLANE and RUNTIME CORRECTNESS of the design. Packaging, dependency
> inventory, issue reconciliation and blast radius belong to GATE B and are deliberately
> NOT covered here.
> **Constraints honoured**: no repository file modified except this report; no product code
> written; no LLM API call, no `f1-sim` / `f1-strat` / `f1-eval`, zero spend. All evidence is
> from reading code and running local python/grep against on-disk parquet.
> **Success condition**: find what is broken. A clean claim below is only credible because the
> closing section names what I tried to break and could not.

---

## Checklist (claims under attack)

| # | Claim | Verdict | Findings |
|---|---|---|---|
| A | Own-car trace panel can be fed from BULK per lap (~2,000 samples at 25 Hz), killing decimation/accumulation/rewind for that panel | **REFUTED** — no artefact on disk can serve a lap slice; the 25 Hz is interpolated; ~2,000 is the floor | D-01 (P0), D-02 (P1) |
| B | After 3.3 the DATA window accumulates nothing: every panel = f(bulk, current lap, frame_index) | **PARTLY REFUTED** — 2 of 6 panels fail (ring, live gap); and `frame_index` addresses nothing in BULK | D-06 (P0), D-07, D-08 (P1) |
| C | The race-pace grid is a progressive reveal masked by the current lap | **PREMISE HOLDS, RULE REFUTED** — "the current lap" is not a scalar; leaks 1-2 laps on 96% of instants | D-09 (P1) |
| D | The `js_api` pull model at 10 Hz across two windows is cheap and loses nothing | **UPHELD on loss, REFUTED on sync** — drops nothing the design needs; measured 58% inter-window divergence | D-10 (P1), D-11, D-12 (P2) |
| E | One TCP client instead of two fixes P3 finding A7 by construction | **REFUTED in both halves** — today's clients are byte-identical; the replacement is measurably worse | D-10 (P1) |
| F | The wire does not carry position, gap, interval or rival lap times | **ENUMERATION RIGHT, CONCLUSION WRONG** — 2 of the 4 are derivable; 2 unnamed fields are the real gap | D-04 (P1), D-14 (P2) |
| G | Today: pause works by accident (`app.py:402`), rewind is broken, decimation arithmetic 25 Hz vs 10 Hz | **VERIFIED IN FULL** — every citation and both percentages reproduce exactly | D-13 (P1) |

Additional findings outside the seven claims: **D-03** (P1, the bootstrap resolves the session from the wrong name field), **D-05** (P1, the only existing gap computation is a hardcoded 200 km/h divisor over a double-counted distance — a live bug not in the A1-A19 register), **D-15** (P2, "reuse the loader Arcade uses" is ambiguous; two-process download race).

Findings are appended below as they are confirmed. Written incrementally on purpose.

---

## Findings

### D-01 (P0) — Claim A is REFUTED at its foundation: there is no artefact on disk from which the PITWALL process can read a 25 Hz lap trace

**Claim under attack** (design §3.3): *"Fed from bulk instead, the panel requests `get_lap_trace(driver, lap)` … and receives the full 25 Hz trace for that lap"*, sourced per §3.2 from *"the same on-disk artefacts the arcade process reads"*, with risk 3 mitigated by *"PITWALL should only ever read the small lap table plus per-lap slices, never the whole session."*

**What is actually on disk.** The per-race raw tree holds four parquets and a metadata file. There is **no telemetry artefact**:

```
$ ls -la data/raw/2025/Melbourne/
intervals.parquet  356.4K
laps.parquet        97.7K
metadata.json         761B
pitstops.parquet    26.0K
weather.parquet      8.1K
```

`laps.parquet` is (927, 35) — a lap table, one row per driver-lap. There is no X/Y/Speed/Throttle/Brake channel anywhere under `data/raw/`.

**Where the 25 Hz trace actually lives.** Exactly one place: `SessionData.frames_by_driver`, built by `SessionLoader._resample_driver` (`src/arcade/data.py:389-422`) and persisted as **one monolithic pickle**:

- `src/arcade/data.py:334-335` — `pickle.dump(sd, f, protocol=pickle.HIGHEST_PROTOCOL)`
- `src/arcade/data.py:345-347` — `_cache_path` → `data/cache/arcade/<gp>_<year>_race.pkl`

A pickle has no random access. `pickle.load` (`data.py:264`) deserialises the **entire** `SessionData` — 20 drivers × `total_frames` `FrameData` objects — to reach one lap of one driver. **Risk 3's mitigation is not implementable against the artefact that holds the traces.** "Read lap slices from disk, never the whole session" describes an access pattern the storage format does not offer.

**The only other producer is FastF1 itself.** A repo-wide search for a lap-trace reader finds `get_telemetry()` at exactly three live call sites, all of them requiring a fully loaded FastF1 session:

```
src/arcade/data.py:182                                    (the arcade loader, cold path)
src/arcade/data.py:456, :480                              (reference-lap geometry)
src/telemetry/backend/services/telemetry_service.py:427   (the backend's per-lap endpoint)
```

`telemetry_service.py:407` reaches it through `get_loaded_session(...)`, i.e. `session.load(telemetry=True)` — the same path `SessionLoader.load` documents as the *"cold path <3 min"* (`data.py:250`). There is no cheap third option.

**Concrete failing scenario.** User launches `f1-arcade` with strategy on. Arcade unpickles `Melbourne_2025_race.pkl` (P2 finding F-05: ~8.0 s AoS unpickle) and holds `frames_by_driver` in RAM. PITWALL spawns as a subprocess, receives the first tick, and calls `session_data.lap_trace(2025, "Melbourne", "NOR", 1)`. To answer it, PITWALL must either (a) `pickle.load` the same file — paying the 8 s a second time and holding a second full copy of the race in a second process, which is precisely the double-cost risk 3 says to avoid and which *cannot* be avoided by "only reading slices"; or (b) `session.load(telemetry=True)` a second FastF1 session; or (c) consume an artefact that does not exist yet.

**Severity.** P0 for the design: §3.3 is called *"the single most important design decision in this document"* and *"the consequence that removes three problems at once"*, and its data source is unbudgeted work. §5 (`session_data.py` contract) and risk 3 both describe a slice-capable store that no part of the repo provides.

**Fix direction** (see fix list): decide the trace source explicitly. Either PITWALL reads the arcade pickle once and accepts the second full load (then say so, and drop risk 3's mitigation as unachievable), or `SessionLoader` gains a **per-lap sidecar** written at session-load time (`data/cache/arcade/<gp>_<year>_traces.parquet`, partitioned by driver+lap) so a slice really is a slice. The second is real work and belongs in the plan.

---

### D-02 (P1) — the "25 Hz" in claim A is INTERPOLATED, not measured, and the two candidate sources produce different series

Two distinct sub-claims fail here, and they fail in opposite directions.

**(a) 25 Hz is synthetic.** `SessionLoader._resample_driver` (`data.py:396-400`) builds every channel with `np.interp(timeline, t, data[k])` onto `timeline = np.arange(0.0, global_t_max - global_t_min, DT)` (`data.py:376`) where `DT = 1.0 / FPS` and `FPS = 25` (`src/arcade/config.py:22-23`). The 25 Hz grid is a rendering convenience, not a sample rate.

**(b) The FastF1 route does not give 25 Hz at all.** FastF1 does not resample by default:

```
$ grep -n "TELEMETRY_FREQUENCY" .venv/lib/site-packages/fastf1/core.py
122:    TELEMETRY_FREQUENCY = 'original'
123:    """Defines the frequency used when resampling the telemetry data. Either
124:    the string 'original' or an integer to specify a frequency in Hz."""
```

`core.py:447-460`: `if frequency == 'original'` → no resampling; `get_telemetry` merges pos_data into car_data on the union of their native timestamps (`core.py:2859-2860`). So `lap.get_telemetry()` returns the merged native series, **not** the 25 Hz grid the circuit window is animating.

**Consequence.** The design's own §3.3 rationale is that the trace panel and the circuit window stop disagreeing. If PITWALL sources the trace from FastF1 while arcade animates the interpolated grid, the cursor placed at `frame_index` sits on a *different* series from the one under it. If PITWALL sources it from the pickle, it matches — but that is D-01's monolith. **The design never says which source it means, and the two are not interchangeable.**

**(c) "roughly 2,000 samples" is the FLOOR, not the typical value.** Measured on the only race present on disk:

```
$ python -c "... pd.read_parquet('data/raw/2025/Melbourne/laps.parquet') ... LapTime*25 ..."
Melbourne 2025 laps with a LapTime: 858
lap seconds  min/median/mean/max: 82.2 / 93.5 / 103.6 / 149.4
25 Hz samples min/median/mean/max: 2054 / 2337 / 2591 / 3735
pct of laps within 1500-2500 samples: 68.1%
pct >= 2500 samples: 31.9%
```

The minimum lap of the race is 2,054 samples. The median is 2,337 (+17%), the maximum 3,735 (+87%). Serialised as compact JSON at 8 fields/sample that is 191 KB / 224 KB / 358 KB per fetch. Caveat, stated because this repo has a written lesson about it: Melbourne 2025 is a **wet, SC-heavy** race and is the only one in `data/raw/`, so it is the slow end of the distribution. A dry 80-95 s lap lands at 2,000-2,375. The order of magnitude in the design is right; the specific number is the floor of the wettest race available, and a "rewind + pin a rival" interaction fetches two of them.

---

### D-03 (P1) — the bootstrap resolves the session from `gp_name`, the one of the two name fields the codebase explicitly declares untrustworthy for this exact purpose

**Claim under attack** (design §3.2): *"Bootstrap: the first tick carries `gp_name`, `year`, `driver_main` and `driver_rival`, which is enough to resolve the session and load its lap table."*

`_build_arcade_snapshot` puts the **display label** on the wire:

- `src/arcade/app.py:481` — `"gp_name": self._session.gp_name`

`SessionData` carries a second field, `location`, added specifically because the two diverge, and its own comment says so:

- `src/arcade/data.py:86-91` — *"Kept separate from `gp_name` which is the arcade-facing display label so the header can still read 'Australia' while the strategy pipeline loads from `data/raw/2025/Suzuka/` — the two diverge whenever the hardcoded `GP_NAMES` table drifts from the active season calendar."*

And the arcade's own on-disk resolver prefers `location` for precisely the job PITWALL is being handed:

- `src/arcade/app.py:346-359` — `_resolve_gp_name()`: *"Prefers the FastF1 Location (`Suzuka`, `Melbourne`, …) because that is what the `data/raw/<year>/` folders use."* → `if self._session.location: return self._session.location`.

**`location` is not on the wire.** The design asks PITWALL to do disk resolution from the field the arcade refuses to do disk resolution from.

**When it actually fires.** With the canonical calendar JSON present, `get_gp_names(year)` returns Locations, so the two agree:

```
$ python -c "from src.arcade.config import get_gp_names, GP_NAMES; ..."
2025 rounds: 24
  1 -> 'Melbourne'   3 -> 'Suzuka'   11 -> 'Spielberg'
hardcoded fallback GP_NAMES[3] = Australia
years in tire_compounds_by_race.json: ['2023', '2024', '2025', ...]
```

They diverge on the two documented fallback paths in `config.py:274-300`: (1) `data/tire_compounds_by_race.json` missing or unparseable → `GP_NAMES` (country labels: round 3 = `"Australia"`, folder is `Melbourne`); (2) **any year outside 2023-2025** → same fallback. `f1-arcade --year 2022 --round 3` is accepted by `main.py:49` and takes that branch. The failing scenario is therefore a `gp_name="Australia"` on the wire and a `FileNotFoundError` (or, worse, a silent empty lap table) in PITWALL for `data/raw/2022/Australia/`.

**A third naming convention exists.** `GP_TO_LOCATION` (`config.py:303-328`) maps to **underscored** names (`Marina_Bay`, `Mexico_City`, `São_Paulo`, `Las_Vegas`, `Yas_Island`), while the canonical calendar JSON uses **spaces** for the same five races:

```
$ python -c "json.load(open('data/tire_compounds_by_race.json'))['2025'] keys with a space"
['Marina Bay', 'Mexico City', 'São Paulo', 'Las Vegas', 'Yas Island']
keys containing underscore: []
```

So a single race name has up to three spellings in this repo, and the design's bootstrap picks a string off the wire and hands it to a path join. **Fix: put `location` on the wire** (one line in `_build_arcade_snapshot`) and resolve from it, or better, put the already-resolved value of `_resolve_gp_name()` there so exactly one resolver exists.

---

### D-04 (P1) — the wire drops `active`, so the DATA window renders retired cars as still circulating

**Claim under attack** (design risk 5): *"Position, gap, interval and rival lap times are NOT in the broadcast (`app.py:455-461` carries `lap, dist, speed, compound, tyre_life` only)."* The field enumeration is correct. The list of what is **missing and matters** is not: it omits the one field whose absence is a correctness bug rather than a gap in coverage.

`FrameData` carries an `active` flag whose entire reason for existing is DNFs — the module docstring names it as one of three deliberate fixes over the reference implementation:

- `src/arcade/data.py:5-7` — *"and an `active` flag that stops DNF'd drivers from sitting as ghosts at their crash position."*
- `src/arcade/data.py:403` — `active = ti <= t_max_local`

The circuit window honours it (`app.py:680-681`, `app.py:694-695`: `if not f.active: return`). **The broadcast deliberately strips it**, and says so:

- `src/arcade/app.py:447-449` — *"we drop fields the dashboard does not use (rel_dist, throttle, brake, active flag) to keep the broadcast JSON small."*

That was sound while the only consumer was a Qt strategy/telemetry pair that never listed the field. The PITWALL DATA window's band 2 **is** a timing table over all 20 cars and band 4 has a ring showing "all cars' positions around the current lap". Both iterate `arcade.drivers`, and on the wire a retired car is indistinguishable from a running one.

**Concrete failing scenario, measured.** Melbourne 2025 had six cars fail to reach the end:

```
$ python -c "pd.read_parquet('.../laps.parquet').groupby('Driver')['LapNumber'].max()"
SAI 1.0   DOO 1.0   HAD 1.0   ALO 33.0   BOR 46.0   LAW 47.0   (14 others 57.0)
```

`np.interp` clamps past the driver's last sample, so SAI's `dist`, `speed`, `lap`, `compound` and `tyre_life` are **frozen at their lap-1 values for the remaining 56 laps** and are broadcast every 100 ms exactly like a running car's. The DATA window's timing table would show SAI classified P-something on lap 1 for the whole race, and the ring would draw a stationary dot at the Turn-1 crash site. This is the repo's documented "dead entities treated as live" class, and the fix is one field on the wire.

---

### D-05 (P1) — the only gap computation that exists today is a hardcoded 200 km/h divisor, and its distance term double-counts a lap per lap of difference

Relevant because the DATA window's timing table needs gaps and the design's stated posture is to reuse what the arcade already does.

`src/arcade/overlays.py:326-336`:

```python
    @staticmethod
    def _gap_value(sign, other, self_entry) -> str:
        other_code, other_prog = other
        _, self_prog = self_entry
        dist = abs(other_prog - self_prog)
        time_s = dist / 55.56 if dist > 0 else 0.0
        return f"{other_code} {sign}{time_s:.2f}s"
```

`55.56` m/s is 200 km/h, hardcoded, with no name and no comment. Measured against the real lap-average speed of the one race on disk (Albert Park, 5,278 m):

```
Melbourne 2025 real lap-average speed (m/s): min 35.3  median 56.4  max 64.2
arcade constant 55.56 m/s -> error vs median lap:  -1.6%
                          -> error vs fastest lap: -13.5%
                          -> error vs slowest (SC) lap: +57.3%
```

It happens to land near the median of a **wet** race. On the fastest lap it under-reads the gap by 13.5%; under Safety Car it over-reads by 57%. On a fast dry circuit (Monza, ~5,793 m / ~80 s ≈ 72 m/s) it would under-read by ~23%.

**And the distance it divides is wrong.** `_rank_drivers` (`overlays.py:502-511`) computes `progress = (lap - 1) * track_len + dist`, but `dist` is **already race-cumulative** — `FrameData`'s own docstring (`data.py:54-56`) says *"`dist` is race-cumulative metres"*, built by the accumulator at `data.py:210-211`. So `progress` adds a completed-laps term to a value that already contains it. For two cars on the same lap the extra term cancels and the gap is right; for a **lapped** car it is inflated by exactly one `track_len` per lap of difference. On Melbourne that is 5,278 m ÷ 55.56 = **95 s of phantom gap per lap down**.

The ranking survives (the error is monotone in true progress, so the *order* is preserved), which is why nobody has noticed — a true sentence ("the leaderboard order is right") sitting on top of a false one ("therefore the gaps are right"). The P3 audit register did not catch either half; it is not in A1-A19.

**Design consequence.** A timing table is a gap-first surface. If PITWALL reuses this, it ships a fabricated number on the surface whose whole thesis is data fidelity — the same category as P3 finding A2 (fabricated weather), which the repo treated as a P1.

---

### D-06 (P0) — the wire has NO time anchor, so nothing on it can be joined to any lap-table artefact by time; the arcade computes the anchor and throws it away

This is the finding that makes several of the others structural rather than cosmetic.

**The tick's `t` carries zero information beyond `frame_index`.** `_resample_driver` assigns `t=float(ti)` for `ti` iterating `timeline` (`data.py:402, 406`), and `timeline = np.arange(0.0, global_t_max - global_t_min, DT)` (`data.py:376`). So `frames[i].t == i * 0.04` for **every driver**, and `_build_arcade_snapshot`'s `"t": main_frame.t` (`app.py:484`) plus `_frame_to_telemetry`'s `"t"` (`app.py:89`) are both restatements of `frame_index`.

**The offset that would make it a session time is discarded.** `global_t_min` is computed at `data.py:374` (`min(r["t_min"] for r in results)`) and used only to shift the per-driver arrays (`data.py:384`). It is **not** a field of `SessionData` — the dataclass (`data.py:75-124`) has 17 fields and none of them is a time origin. `session.t0_date` (the UTC anchor FastF1 offers) is never touched anywhere in `src/arcade/`.

**What that costs, concretely.**

| BULK artefact | Its time key | Joinable to the frame clock? |
|---|---|---|
| `laps.parquet` `Time`, `LapStartTime`, `Sector*SessionTime` | session-elapsed `timedelta` | **No** — needs `global_t_min` |
| `laps.parquet` `LapStartDate` | UTC | **No**, and it is empty anyway: `0 / 927` non-null (measured) |
| `intervals.parquet` `date` | UTC wall clock | **No** — needs `global_t_min` *and* `t0_date` |
| `weather.parquet` `Time` | session-elapsed | **No** — same as laps |

Measured:

```
$ python -c "... laps.parquet ..."
laps.parquet columns that are wall-clock: ['LapStartDate']
LapStartDate non-null: 0 / 927

$ python -c "... intervals.parquet ..."
intervals rows 18965  drivers 19
date span 2025-03-16 03:07:28.621+00:00 -> 2025-03-16 06:01:09.841+00:00
median seconds between samples per driver: 4.3
```

So the **only** join key the wire actually offers PITWALL is the integer `lap`. Every "function of (bulk data, current lap, frame_index)" in design §3.4 is really a function of *(bulk data, current lap)* — `frame_index` cannot address anything in BULK. That is why D-07 (the ring), D-08 (live gaps) and D-04 (`active`) all fail together: they are the three panels that need sub-lap addressing.

**`intervals.parquet` is dead weight today.** It is the one artefact with genuine sub-lap gap resolution (~4.3 s/sample) and nothing in `src/simulation/` or `src/f1_strat_manager/` reads it — `RaceReplayEngine` loads `laps.parquet` and `weather.parquet` only (`replay_engine.py:73-84`). The design does not mention it either.

**Fix (cheap, one line each, unlocks three findings).** Put `global_t_min` on the wire in `_build_arcade_snapshot`, and store it on `SessionData` (a `CACHE_VERSION` bump, which §B.1 of the P3 plan already schedules). Add `t0_date` too if the intervals join is ever wanted. Without at least the first, §3.3's "the tick supplies `frame_index`, which the panel uses to place the cursor" cannot be implemented against any BULK series that is not itself indexed by frame.

---

### D-07 (P1) — Claim B: the ring CANNOT be computed from the wire. It needs `rel_dist`, which the broadcast deliberately removes, and `dist mod circuit_length` is not a substitute

**Claim under attack** (design §3.4): *"The DATA window, after 3.3, accumulates nothing. Every one of its panels is a function of (bulk data, current lap, frame_index)."* §4 lists `TrackRing.tsx` in band 4.

A ring is angular position around the current lap — `rel_dist ∈ [0,1)` per car, at the instant. The wire drops it, by name:

- `src/arcade/app.py:447-449` — *"we drop fields the dashboard does not use (**rel_dist**, throttle, brake, active flag)"*
- `src/arcade/app.py:455-461` — the 20-car block: `lap, dist, speed, compound, tyre_life`

`dist` is the race-cumulative accumulator (`data.py:54-56`, built at `data.py:210-211`), so the obvious substitute is `dist % circuit_length_m`. **The repo already rejected that substitution, in this file, for this reason** — `_frame_to_telemetry`'s docstring:

- `src/arcade/app.py:68-73` — *"Uses `frame.rel_dist * circuit_length` as the broadcast `dist` because `frame.dist` is the race-cumulative accumulator and would push the X axis to tens of kilometres as the race progresses."*

And the modulo is not just inconvenient, it is wrong: `total_dist_so_far += float(d_lap[-1])` accumulates **each lap as actually driven** (a pit in-lap and out-lap follow a different route and a different length), while `circuit_length_m` is a single constant taken from the **fastest lap's** `add_distance()` (`data.py:634-637`). The residual `dist − N·circuit_length_m` therefore drifts monotonically across a race, and the drift is largest for exactly the cars a strategy surface cares about — the ones that pitted.

**The asymmetry is the tell.** The wire already carries per-lap distance, correctly derived from `rel_dist`, for **two** cars (`telemetry.main`, `telemetry.rival`, `app.py:473-479`) and race-cumulative distance for **twenty**. The ring needs the per-lap form for twenty. **Fix: add `rel_dist` to the 20-car block** (5 bytes/car/tick) or move `TrackRing` off the wire entirely.

---

### D-08 (P1) — Claim B: a gap/interval is a function of the INSTANT, and the design's own risk 5 asks the right question. The answer is that the DATA window can only show a value that FREEZES for a whole lap

Risk 5 says: *"A gate should check that assumption holds for gaps specifically, because a gap is a function of the current instant, not of the lap."* Checked. It does not hold, and here is the precise shape of the failure.

**The good news first, because it changes the fix.** The repo already has a correct, lap-indexed rival view that gives the DATA window almost everything risk 5 lists as missing:

- `src/simulation/race_state_manager.py:470-538` — `get_rival_states(lap_number)` returns, per rival: `position`, `lap_time_s`, `compound`, `tyre_life`, `stint`, `speed_st`, **`gap_to_leader_s`**, **`interval_to_driver_s`**, `is_pitting`.

Its docstring (`:484-489`) even documents the sentinel discipline this repo has a scar about: an unknown position stays `None` and the `99` placeholder never leaves the sort key. That is the loader the design's risk 1 should be pointing at, and it does not name it.

**The bad news.** `interval_to_driver_s` is `rival_cum − driver_cum` where `*_cum` is `session_time_s`, the driver's session-elapsed time **at the end of that lap** (`race_state_manager.py:48-63, 494-520`). It is a **line-crossing delta**, one value per lap. Rendered in a timing table it is constant for the whole lap and steps at the line — which is not what a timing screen looks like and, worse, is not what the number means: the interval to the car ahead at the moment they crossed is not the interval now.

**The sub-lap source exists and cannot be reached.** `intervals.parquet` has ~4.3 s resolution (measured, D-06) — real live intervals. Its key is UTC. Per D-06 the wire has no wall-clock and no session-time anchor, and `laps.parquet.LapStartDate` is `0/927` non-null. **There is no join.**

**Verdict on claim B: refuted for two of the six panels.** The timing table's gap column and the ring are the counter-examples the brief asked for. The rest of claim B survives (see the closing section). The honest restatement is: *"every DATA panel is a function of (bulk data, current lap) — `frame_index` addresses nothing in BULK, and the two panels that genuinely need the instant must either accept a lap-quantised value or get a new field on the wire."*

---

### D-09 (P1) — Claim C: "the current lap" is not a scalar. At 96% of instants the running field spans 2-3 different laps, and the tick carries only the MAIN driver's

**Claim under attack** (design §3.2): *"In a replay the entire race is known before lap 1. The race-pace grid, the bests panel and the timing table's history are therefore a progressive reveal masked by the current lap."*

The artefact is fine: `laps.parquet`, (927, 35), all 20 drivers × all laps, carrying `LapTime`, `Sector1/2/3Time`, `SpeedI1/I2/FL/ST`, `Compound`, `TyreLife`, `Stint`, `Position`, `TrackStatus`. A drivers × laps grid and a gapper plot are both directly computable from it. Claim C's *premise* holds.

**The masking rule does not.** "The current lap" in the design is a single number, and the tick supplies exactly one:

- `src/arcade/app.py:483` — `"lap": main_frame.lap if main_frame else 1` — **the main driver's lap**.

Measured on Melbourne 2025, at each of NOR's 56 lap-crossing instants:

```
ALL cars:      lap spread median 28, max 56 | distinct lap numbers median 4, max 6
               field NOT all on one lap: 56/56 = 100%
RUNNING cars:  lap spread median  1, max  2 | distinct lap numbers median 2, max 3
               running field NOT on one lap: 55/57 = 96%
```

Masking every driver's row at the main driver's lap `M` is wrong in both directions **at the same time**:

- a car **ahead** (on lap `M+1`) has completed lap `M`; the mask hides it → the grid lags reality by a lap for the leaders, on 96% of instants;
- a car **behind or lapped** (on lap `M−1`) has **not** completed lap `M−1`; the mask reveals it → a **look-ahead leak** of 1-2 laps for running cars, and up to 56 laps of "should read OUT, will read pending" for the three cars that retired on lap 1 (SAI, DOO, HAD).

**The correct rule, stated so it can be implemented.** Per driver, not global; strict, not inclusive:

> reveal driver *d*'s lap *L* **iff** `L < wire.arcade.drivers[d].lap` — and render *d* as retired, not pending, when `d` is inactive (D-04).

`wire.arcade.drivers[d].lap` is already on the wire for all 20 cars (`app.py:456`), so the fix costs nothing but writing the rule down. `L < c(d)` rather than `L <= c(d)` matters: a lap only has a `LapTime` once it is crossed, so `<=` publishes the time of a lap still in progress — the same look-ahead the mask exists to prevent.

**Where the grid is indexed by lap and the clock by frame.** There is no mismatch to fix *inside* a lap, because per D-06 `frame_index` cannot address anything in the lap table anyway. The reveal is quantised to the lap by construction; the design should say so rather than imply frame-level granularity.

---

### D-10 (P1) — Claim E is REFUTED in both halves: today's two clients do NOT drift, and the replacement introduces real, measured divergence

**Claim under attack** (design §2): *"One TCP client, not two. This fixes P3 finding A7 by construction: today two independent `TelemetryStreamClient` sockets in the same Qt process receive, decode and JSON-parse every broadcast twice **and drift out of sync**."*

**A7 says what the design says it says.** Verified: `AUDIT_P3_ARCADE.md:101` — *"Two independent `TelemetryStreamClient` sockets in the same Qt process: every broadcast is received, decoded and JSON-parsed twice, and the two windows drift out of sync (baseline M5)."* Evidence `window.py:205-209`, `telemetry_window.py:53-57`. Both confirmed at `window.py:209` and `telemetry_window.py:57` (`self._client.start()` on a `TelemetryStreamClient` each).

**Half one — the duplicate parse — is TRUE.** Two `QThread`s, two sockets, two `json.loads` per broadcast (`stream_client.py:120`). One client genuinely halves that.

**Half two — "drift out of sync" — is FALSE, by execution.** `TelemetryStreamServer.broadcast` encodes **once** (`stream.py:93`) and `sendall`s the identical bytes to every client (`stream.py:101-105`). I drove the real server with two raw sockets:

```
$ python scratchpad/twoclients.py        # real src.arcade.stream.TelemetryStreamServer, 200 broadcasts
client A received: 200 client B received: 200
identical sequences: True
A first/last: 0 199
B first/last: 0 199
```

Byte-identical, complete, in order, zero drops. The two Qt windows consume the same payload sequence and both dispatch onto the same GUI thread; the worst case is a transient one-payload skew in event-loop ordering, not drift. **A7's third clause is an overstatement that this design inherited and promoted into a justification.**

**Half three — the replacement is strictly WORSE at the thing it claims to fix.** The proposed topology is one slot overwritten by the TCP thread and two windows polling `pywebview.api.get_tick()` on independent timers (§3.5). Two free-running timers at the same nominal rate beat against each other. Simulated at the design's own 10 Hz on both sides, half-period offset:

```
$ python scratchpad/pullslot.py
samples per window: 55
windows read a DIFFERENT frame on 32/55 polls = 58%
window A duplicate reads (same frame twice): 15/54
window A skipped frames (gap > 1):           15/54
A[:12] = [0, 0, 2, 2, 4, 4, 6, 7, 7, 9, 10, 10]
B[:12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

58% of polls, the two windows are looking at different frames. A JS `setInterval` is no better behaved than a Python `sleep` loop (browser timer clamping and rAF jitter make it worse). The skew is bounded at ~1 tick / 100 ms, which for a read-only follow-the-replay surface is defensible — **but it must be argued, not claimed as a fix.** Say instead: *"one client removes the duplicate decode (A7's real half); the two windows remain up to one tick apart and always were."* If sync actually matters, stamp the slot with a sequence number and let each window render the max it has seen, or have one poller push to both.

**Did anything depend on two independent clients?** Checked, and no. `dashboard/__main__.py:6-8` claims the independence is a feature — *"Both subscribe to the same arcade TCP stream independently, so the user can drag them across monitors and close one without affecting the other"* — and `TelemetryWindow.closeEvent` (`telemetry_window.py:78-82`) stops only its own client. Under one shared client owned by the PITWALL host, closing one window must **not** stop the client. That is a real teardown requirement the design does not state, and it is the one place where "one client" is a regression risk rather than a win.

---

### D-11 (P2) — the `truncate(frameIndex)` guard is in the wrong UNIT for its only subscribers, and applying it turns a benign bug into permanent data loss

**Claim under attack** (design §3.4): *"One module, `lib/frameClock.ts`, holds the last seen `frame_index`. When an arriving tick carries a LOWER index, it emits a `truncate(frameIndex)` event. Panels that accumulate subscribe and drop everything after that point."*

**Unit mismatch.** After §3.3 the design names exactly two accumulators: the AGENTS window's `PaceChart` and `TireChart`. Their Qt originals are **lap-keyed dicts**, not frame-keyed:

- `src/arcade/dashboard/window.py:179-180` — `self._pace_history: dict[int, ...]` / `self._tire_history: dict[int, ...]`, keyed by `lap_number` (`window.py:249, 251`).

A `truncate(frameIndex)` cannot be applied to a lap-keyed store. The subscriber would have to convert, and per D-06 the wire offers no frame→lap mapping other than `arcade.lap` itself — in which case the guard should emit `truncate(lap)` and the module is misnamed and mis-specified.

**Applying it destroys data that never comes back.** Two mechanisms, both verified:

1. **The producer never rewinds.** `SimConnector._drive_pipeline` iterates `engine.replay()` once, forward (`strategy.py:376`). `_wait_for_arcade(lap_num)` (`strategy.py:241-247`) only *blocks* until arcade catches up; `_should_skip_stale` (`strategy.py:277-286`) only *skips* when arcade is ahead. No path re-emits a past lap.
2. **The tail cannot rebuild predictions.** `snapshot_dict` strips `per_agent` from `history_tail` (`strategy.py:178-181`), and the re-seed only restores actuals — `_seed_history_from_tail` sets `actual`, `tyre_life`, `compound`, `lap_time_s` and nothing else (`window.py:244-256`).

So: rewind at lap 50 → truncate wipes laps 11-50 → play forward → the connector has already passed those laps → `pred` / `ci_p10` / `ci_p90` for laps 11-50 are **gone for the rest of the session**. Today, with no truncation, the charts merely show the future — visually wrong, losslessly recoverable by playing forward. **The guard makes it worse.** Given Víctor's stated intent (§3.4: rewind is *"so you do not miss something"*, not a study tool), the right call is arguably to truncate **only** what is genuinely re-derivable — i.e. nothing in the AGENTS charts — and leave them alone.

**Also unguarded: the EQUAL index.** `frameClock` is specified to fire only on a *lower* index. Duplicate ticks at the same index are routine, not exceptional:

```
$ python -c "60 Hz on_update / STREAM_BROADCAST_EVERY_N_FRAMES=6 vs FPS=25 * speed"
speed 0.25x: clock 6.2 frames/s vs 10.00 broadcast/s -> 1.6x DUPLICATE broadcasts of the same frame
speed  0.5x: 20% of frames never leave arcade
paused:      infinite duplicates of one frame
```

Plus the poller's own duplicates (D-10: 15 of 54 reads). The Qt side is immune by accident — both accumulators are keyed dicts, so a duplicate overwrites (`telemetry_panel._append` at `:279-291` keys on `int(dist)`; `_pace_history` keys on lap). **Any JS port that appends to an array instead of writing to a keyed map re-introduces the bug the Qt version avoided without knowing it.** Write the keyed-store requirement into §6's TypeScript directives.

---

### D-12 (P2) — the 30-entry history tail is a 30-**lap** window and the AGENTS charts can never be reconstructed beyond it

`STREAM_HISTORY_TAIL = 30` (`config.py:182`) and `snapshot_dict` takes `self.history[-history_tail:]` (`strategy.py:179`), one entry per lap. A Melbourne race is 57 laps, Monza 53, Spa 44.

Because the AGENTS window is the design's only accumulator and §3.4 gives it no re-seed beyond the tail, a window that starts late or is closed and reopened — which two independent pywebview windows invite, and which `dashboard/__main__.py:6-8` advertises as a feature today — recovers at most the last 30 laps of actuals and **zero** laps of per-agent predictions (D-11 mechanism 2). This is the same limitation the Qt code already documents at `window.py:246-249` (*"an accepted limitation of the mid-stream reconnect path"*), so it is inherited rather than introduced — but the design promotes reconnect from an edge case to a normal interaction (a closable window per surface) without re-examining it.

---

### D-13 (P1) — Claim G verified in full, with the arithmetic reproduced

All three of the design's statements about today's behaviour hold.

**(a) Pause works by accident.** `on_update` gates the clock advance on `if not self._is_paused:` (`app.py:393-395`) but calls `self._broadcast_if_due()` **unconditionally** at `app.py:402`, outside that block. A paused arcade keeps broadcasting the same `frame_idx` at 10 Hz forever, and the follower freezes because the content stops changing, not because anything told it to. Line number in the design (`app.py:402`) is exact.

**(b) Rewind is broken.** Both citations check out:
- `src/arcade/dashboard/telemetry_panel.py:262` — `self._append(self._main_buffer, main)`; `_append` (`:279-291`) writes into a **distance-keyed** bucket (`key = int(float(dist))`). Nothing deletes keys ahead of the cursor. Precision worth adding: the panel *does* clear on a lap change (`:257-260`), so rewinding **across** a lap boundary self-heals; rewinding **within** a lap leaves every future-distance sample of that lap in the bucket until the car drives over that distance again.
- `MainWindow._pace_history` / `_tire_history` are lap-keyed dicts (`window.py:179-180`) and the only eviction, `_trim_history(keep=40)` (`window.py:291-300`), drops the **oldest** laps. The future is never deleted. Confirmed.

**(c) The decimation arithmetic is exactly right.** `main.py:33` constructs `arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, WINDOW_TITLE, resizable=True)` with **no `update_rate`**, and the installed default is 1/60:

```
$ python -c "import arcade, inspect; print(arcade.VERSION, inspect.signature(arcade.Window.__init__).parameters['update_rate'])"
arcade 3.3.3
update_rate default: update_rate: 'float' = 0.016666666666666666
```

`_broadcast_if_due` counts **`on_update` calls**, not replay frames (`app.py:426-428`), so 60 Hz ÷ `STREAM_BROADCAST_EVERY_N_FRAMES = 6` = 10.0 Hz, while the clock advances `delta_time * FPS * playback_speed` with `FPS = 25` (`app.py:394`, `config.py:22`). Full table:

```
speed 0.25x:   6.2 replay frames/s vs 10.00 broadcast/s -> 1.6x DUPLICATE broadcasts of the same frame
speed  0.5x:  12.5 -> 20% of frames never leave arcade
speed  1.0x:  25.0 -> 60% of frames never leave arcade      <- design says 60%. correct.
speed  2.0x:  50.0 -> 80%
speed  4.0x: 100.0 -> 90%
speed  8.0x: 200.0 -> 95%                                    <- design says 95%. correct.
paused:        0.0 -> infinite duplicates of one frame
```

**One thing the design's framing misses**: the loss is not monotone in speed. Below ~0.4x the broadcast **over**-samples and the same frame goes out repeatedly. So the channel is lossy at five of the six playback speeds and duplicative at the sixth, and the design's §3.3 argument ("degrades as you speed up, exactly backwards from what a user expects") is only the top half of the problem.

---

### D-14 (P2) — Claim F: the field enumeration is right, but two of the four "missing" fields are already derivable from the wire, and the design does not say so

**Claim under attack** (design risk 5): *"Position, gap, interval and rival lap times are NOT in the broadcast (`app.py:455-461` carries `lap, dist, speed, compound, tyre_life` only). They exist in `lap_state` and in the parquet."*

The literal enumeration at `app.py:455-461` is correct. The conclusion drawn from it is too pessimistic in one place and too optimistic in another.

**Where each field can legitimately come from:**

| DATA-window field | Legitimate source | Notes |
|---|---|---|
| position (live order) | **the wire**, `sort by drivers[*].dist desc` | `dist` is race-cumulative (`data.py:54-56`), so a plain descending sort is the correct live order including lapped cars. No `(lap-1)*track_len` term — that is D-05's double count. |
| position (classified, end of lap) | BULK `laps.parquet.Position` | 921/927 non-null on Melbourne. |
| gap / interval, lap-quantised | BULK `get_rival_states(lap)` → `gap_to_leader_s`, `interval_to_driver_s` | Correct, sentinel-safe, already written (`race_state_manager.py:470-538`). Freezes for a whole lap — D-08. |
| gap / interval, live | **nowhere reachable** | `intervals.parquet` has it at 4.3 s resolution; no join key exists — D-06. |
| rival lap times | BULK `laps.parquet.LapTime` | Fine. |
| **`active` / retired** | **nowhere** | Not on the wire (D-04), and not derivable from BULK either: a retired car's wire `lap` freezes at a value `<= ` its last lap, which is indistinguishable from a running car sitting on that lap. The only stateless route needs a time anchor (D-06); the only other route is to *accumulate* "has this driver's lap stopped advancing", which contradicts claim B outright. |
| **`rel_dist` (ring angle)** | **nowhere** | D-07. |

**So the corrected version of risk 5 is:** four fields named, two of them (position, rival lap times) already available; two fields *not* named (`active`, `rel_dist`) that are unavailable and load-bearing; and one field (live gap) that is unavailable for a structural reason the risk does not identify.

---

### D-15 (P2) — risk 1 checked: "reuse the loader Arcade uses" is not a well-formed instruction, because Arcade uses three, on two different roots, and the design names the wrong reuse rule

Risk 1 asks the gate to *"verify that the chosen reader is the same one Arcade uses"*. Arcade uses three readers for three different artefacts:

| Artefact | Arcade's reader | Root it resolves against |
|---|---|---|
| 25 Hz frames | `SessionLoader.load` → FastF1 → pickle (`data.py:256-343`) | `get_data_root()` (`config.py:159-160`) |
| featured laps | `SimConnector._load_laps_df` → `augment_featured_laps` (`strategy.py:559-575`) | **`REPO_ROOT`** (`strategy.py:571`) |
| raw race dir | `SimConnector._resolve_race_dir` → `RaceReplayEngine` → `RaceStateManager` (`strategy.py:577-597`, `replay_engine.py:73-84`) | **`REPO_ROOT`** (`strategy.py:589`) |

Three consequences the design should absorb:

1. **The rule risk 1 cites is the wrong rule.** *"Every consumer calls `augment_featured_laps`, never the parquet directly"* governs `data/processed/laps_featured_<year>.parquet`, and the docstring at `strategy.py:560-570` explains exactly why (the missing `Time_s`, the Lusail 0.49 s vs 3.29 s gap). The DATA window's lap table is **`data/raw/<year>/<gp>/laps.parquet`**, a different artefact with a different loader. Pointing `session_data.py` at `augment_featured_laps` would be reuse of the wrong thing; the right reuse target is `RaceStateManager` (which already owns `session_time_s`, `get_rival_states`, and the `None`-not-sentinel discipline).

2. **Two roots are already in play, and PITWALL would be the third process choosing between them.** `config.py:158` defines `REPO_ROOT` from `__file__`; `config.py:159-160` route the caches through `get_data_root()`, which honours `$F1_STRAT_DATA_ROOT` (docker-compose sets it to `/app/data`). With the override set, the arcade's strategy layer and the arcade's session cache **already** read from different trees (P3 A19 / P2 F-11). A PITWALL process that correctly uses `get_data_root()` will silently disagree with the arcade process that spawned it about which race it is showing.

3. **Name resolution is a ritual, not a lookup, and it is about to be duplicated.** `_resolve_race_dir` (`strategy.py:577-597`) tries `GP_TO_LOCATION[gp]`, then the same with spaces replaced by underscores, then returns the miss — three attempts, because the repo has three spellings per race (D-03), and a fourth in `src/f1_strat_manager/gp_slugs.py`. If `session_data.py` writes its own resolver from the wire's `gp_name`, that is the fourth copy of a ritual that already needed three tries. **Reuse `_resolve_race_dir`, or better, put the already-resolved path/location on the wire.**

**And a genuine two-process hazard the design creates.** `ensure_race` (`data_cache.py:431-452`) guards with `if race_dir.exists() and any(race_dir.iterdir()): return race_dir` and otherwise calls `_snapshot_download` (`data_cache.py:276-306`), which is `huggingface_hub.snapshot_download(local_dir=...)`. That guard is a **check-then-act on a partially-populated directory**: a race folder holding 1 of its 5 parquets passes `any(...iterdir())`. Today only one process ever calls it. The moment PITWALL also resolves race data on disk, a cold race gives you two concurrent `snapshot_download` calls into the same `local_dir`, and the loser can observe a half-written folder as "complete". **Mitigation: PITWALL must never trigger a download — set `F1_STRAT_OFFLINE=1` in its environment (the `ensure_race` early-return at `:447-448` already supports it) and treat a missing race as an error the arcade should have prevented.**

---

## Fix list (ordered by value, then by risk)

1. **Decide and write down the trace source for `get_lap_trace` (D-01, D-02).** The design's central decision has no data behind it. Three options, in increasing cost: (a) PITWALL unpickles the arcade cache once and holds it — cheapest, but delete risk 3's mitigation because it is unachievable; (b) `SessionLoader` writes a per-lap sidecar (`data/cache/arcade/<gp>_<year>_traces.parquet`, partitioned driver+lap) at load time, so a slice really is a slice — this is the honest answer and it is real work, and it rides the `CACHE_VERSION` bump P3 §B.1 already schedules; (c) FastF1 per lap — rejected, it is a second full session load and a different sample rate. **Do this before anything else; §3.3 is load-bearing for §3.4, §5 and the whole "accumulates nothing" argument.** *Value: highest. Risk: none, it is a decision.*

2. **Widen the wire by four scalars (D-03, D-04, D-06, D-07).** In `_build_arcade_snapshot`: add `location` (or the output of `_resolve_gp_name()`), add `global_t_min`, add `active` and `rel_dist` to the per-driver block. Roughly 12 bytes/car/tick plus two constants. This single change unlocks correct session resolution, correct DNF rendering, the ring, and any BULK join by time. *Value: very high. Risk: low — additive fields, existing consumers ignore them; needs `global_t_min` stored on `SessionData`, hence one `CACHE_VERSION` bump.*

3. **Write the masking rule into §3.2 as per-driver and strict (D-09).** `reveal driver d's lap L iff L < wire.arcade.drivers[d].lap`, and render inactive drivers as retired rather than pending. Costs nothing, prevents a look-ahead leak on 96% of instants. *Value: high. Risk: none.*

4. **Correct claim E's wording and state the teardown requirement (D-10).** Keep one client — the duplicate-decode saving is real. Drop "fixes the drift by construction": the two windows do not drift today and will be up to one tick apart afterwards. If sync matters, stamp the slot with a monotonic sequence number. And state explicitly that closing one window must not stop the shared client, since the current two-client design advertises independent close as a feature. *Value: high (it is a false claim in a design doc). Risk: none.*

5. **Re-specify `frameClock` in laps, not frames, and reconsider truncating the AGENTS charts at all (D-11).** The only accumulators are lap-keyed; a frame-indexed truncate cannot address them, and truncating destroys `per_agent` predictions that no channel can rebuild. Also add "accumulating panels use keyed maps, never arrays" to §6's TypeScript directives, so duplicate ticks (routine at ≤0.5x and permanent while paused) stay idempotent the way the Qt version is by accident. *Value: high. Risk: low.*

6. **Decide what the timing table's gap column actually shows, and say so on screen (D-05, D-08).** Either lap-quantised `interval_to_driver_s` from `get_rival_states` — correct, labelled as at-the-line — or nothing. Do **not** port `overlays._gap_value`: its 55.56 m/s divisor is a fabricated constant (+57% under SC on the one race measured) and its distance term double-counts a lap per lap of difference. *Value: high — it is the P3 A2 "fabricated data on a fidelity surface" class. Risk: low.*

7. **File the arcade leaderboard gap bug as its own issue (D-05).** It is live today, it is not in the A1-A19 register, and per the repo's own rule an important bug gets an issue before its fix. Scope: `overlays.py:502-511` double count + `:326-336` hardcoded divisor. *Value: medium. Risk: none.*

8. **Point risk 1 at `RaceStateManager`, not `augment_featured_laps`, and forbid a fourth name resolver (D-15).** Add to §5's `session_data.py` contract: resolve the race directory through `_resolve_race_dir` (or the wire), resolve the root through `get_data_root()` only, and run with `F1_STRAT_OFFLINE=1` so PITWALL can never race the arcade into a `snapshot_download`. *Value: medium. Risk: low.*

9. **Note in §3.3 that the channel is lossy at five speeds and duplicative at the sixth (D-13c).** The current framing ("degrades as you speed up") is half the picture and hides the paused/slow-motion duplicate case that any array-appending panel would trip over. *Value: low. Risk: none.*

10. **Record that the DATA window's reveal is lap-quantised (D-06, D-09).** `frame_index` addresses nothing in BULK. §3.4's phrase "a function of (bulk data, current lap, frame_index)" should read "(bulk data, per-driver current lap)" for every panel except the own-car traces, where `frame_index` is a cursor within an already-fetched array. *Value: low. Risk: none.*

---

## WHAT I TRIED TO BREAK AND COULD NOT

Stated explicitly so the rest of the report is worth something.

- **"A latest-slot pull model drops a strategy decision."** It does not, and I checked three ways. Decisions are **levels, not events**: `StrategyState.latest` persists until the next lap overwrites it (`strategy.py:145-183`), so any number of dropped ticks still leaves the current decision readable. The producer cannot outrun the poller: `_wait_for_arcade` (`strategy.py:241-247`) blocks each lap until the arcade reaches it, and the shortest lap at the fastest playback speed (8x, ~82 s lap) is ~10 s — about 100 poll intervals. And `history_tail` re-seeds continuously via `setdefault` (`window.py:244-256`), so even a late connect recovers the last 30 laps of actuals. **The pull model's dropped ticks cost nothing the design depends on.** Its problem is the sync *claim* (D-10), not lost data.
- **"`error` is a transient the slot can miss."** It is not. `state.error` is set at `strategy.py:338, 347, 353, 408, 458, 507` and cleared at `:441, 475, 533, 549` — but every set/clear pair straddles a phase that lasts seconds (model warmup, radio-corpus load) or is terminal. Nothing flips inside one 100 ms poll interval.
- **"`IsPersonalBest` is a session-final flag, so a progressive reveal of the bests panel leaks the answer from lap 1."** This was my strongest expected hit on claim C and it is **wrong**. Measured: it is a *running* flag — NOR carries it on 21 separate laps (`[6,7,8,9,10,12,13,14,15,16,23,24,25,26,28,29,30,31,33,42,43]`), against a running-cummin of `[1,8,9,...,43]`. Every driver has 18-24 flagged laps. It is safe under masking. (The two sequences are not identical, so a bests panel should still recompute from the revealed subset rather than trust the column, but there is no look-ahead leak.)
- **"The lap table is missing laps, so a lap-indexed grid has holes."** Not on the one race available: `drivers with a missing lap number in laps.parquet: []`. Every driver's `LapNumber` sequence is contiguous from 1 to their last.
- **"The arcade's lap channel is interpolated, so `get_lap_trace(driver, lap)` can be asked for a lap that does not exist."** The mechanism is real and I reproduced it by executing the exact code path (`np.interp` at `data.py:400` + `max(1, int(round(...)))` at `data.py:414`): with lap 11's telemetry dropped by the `get_telemetry()` catch at `data.py:181-186`, the resampler labels **2,000 frames (80 s) as lap 11** — a lap with no telemetry at all. **But I could not show it fires on real data**, because proving the drop requires loading a FastF1 session (network + spend-adjacent, out of bounds here) and the local lap table shows no gaps. Recorded as a latent P2 for whoever implements `get_lap_trace`: key the fetch on the BULK lap table's `LapNumber` and handle "the arcade says lap N, BULK has no lap N" as a real branch.
- **"Two Qt clients drift out of sync."** Executed against the real server: byte-identical, complete, in-order for both. This is the finding that refutes half of claim E (D-10) — I went looking for the drift to confirm A7 and found it is not there.
- **"The design's decimation numbers are wrong."** They are exactly right (60% at 1x, 95% at 8x), and so are all three of claim G's file:line citations (`app.py:402`, `telemetry_panel.py:262`, `main.py:33`). Reproduced in full at D-13.
- **"`get_rival_states` has a sentinel collision."** It does not. `position` stays `None` when absent and the `99` placeholder is confined to the sort key with a docstring explaining why (`race_state_manager.py:484-489, 526, 538`). This is the repo's own #428 lesson correctly applied, and it is the right thing for `session_data.py` to reuse.
- **"`snapshot_dict` recomputes `asdict` over the 30-entry tail on every broadcast, so the pull model would make it worse."** It does (`strategy.py:178-181`, P3 A8) — but the design does not change the broadcast cadence, so this is inherited, not introduced, and P3 D.3 already owns it. Not counted as a finding here.

---

## Scope note

Not covered here, by instruction: packaging, `pywebview`/`webkit2gtk` dependency inventory, `PySide6`/`pyqtgraph` removal, issue reconciliation (§9), published-prose drift (risks 7 and 8), token/palette duplication (§7 test 1), and blast radius on `pyproject.toml` or CI. Those are GATE B's.

Two items inside my lens that I deliberately did not chase, because they are unmeasurable without spend or network: the actual `js_api` round-trip cost (risk 4 — unmeasured by the design and unmeasurable by me; the *design* critique in D-10 stands on its own), and whether the `data.py:181-186` telemetry-drop branch fires on any real 2025 race (see above).


