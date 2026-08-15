# PITWALL delivery plan (v2.6.0)

**One sprint per Claude session.** Each sprint closes with every PR merged to `dev` with green CI,
memory updated, and a handoff prompt written for the next session. Nothing here is code; this is
the sequencing contract.

Reads with: `documents/research/PITWALL_V2_ARCHITECTURE.md` (the design, including the sections the
gates refuted), `documents/audits/GATE_PITWALL_ARCH_A.md` (data plane, 15 findings),
`documents/audits/GATE_PITWALL_ARCH_B.md` (repo fit, 8 findings, plus two inventories that are
deliverables in their own right).

---

## 0. How to run a sprint

Same shape every time, because the value is in the repetition:

1. **Open a clean session.** Paste the handoff prompt the previous sprint produced.
2. **Read the pointers it names**, not the whole memory tree.
3. **One branch per issue**, `feat/`, `fix/` or `docs/`. PR to `dev`, single concern, `Closes #N`.
4. **Merge with `gh pr merge <N> --merge --body ""`** and `--delete-branch`. The empty body is not
   optional: `gh` puts the PR title in the merge-commit body and release-please parses it, which is
   how the CHANGELOG got duplicated twice before.
5. **CI green on every PR**, not just at the end of the sprint.
6. **NOTHING reaches `main` until PITWALL is finished** (Víctor, 2026-08-07). Every sprint lands on
   `dev` and stops there; the single `dev -> main` promotion happens once, after sprint 7's exit
   gate. Consequence, accepted deliberately: because issues stay open until the work is on `main`,
   **#841-#844 and the rewritten #281/#284/#285 all stay open for the whole programme.** The open
   list will not shrink for seven sprints. That is the rule working, not a backlog leak.
7. **Close the sprint**: update `MEMORY.md` plus the sprint's topic file, then write the next
   handoff prompt (task, compact, memory pointers).

Commands handed over are PowerShell, one command per physical line, starting with an absolute
`cd` and then the branch checkout. Víctor runs every commit, push and merge himself.

---

## 1. The sprint table

| # | Sprint | Issues | Gate at the end |
|---|---|---|---|
| **1** | **The wire** | #841, #842, #843, #844 | none (verifiable by running) |
| 2 | PITWALL skeleton: the vertical slice | new | adversarial: does the chain really work end to end |
| 3 | AGENTS window, 1:1 | rewritten #285 | adversarial: is it ACTUALLY 1:1, field by field |
| 4 | **DATA band 4: own-car traces + the ring** (REORDERED 2026-08-09, see §5) | rewritten #284 (c) | none |
| 5 | DATA bands 1-2: status, timing table, bests | rewritten #284 (a) | none |
| 6 | DATA band 3: race pace grid + race trace | rewritten #284 (b) | adversarial: tier discipline and fidelity claims |
| 7 | Retire Qt, package, fix the prose | rewritten #285, new | **exit gate, then the ONE `dev -> main`** |
| **8** | **AGENTS: the elevate pass** | new | Fable as senior dashboard designer, P0-P3 with file:line |
| **9** | **DATA: the elevate pass** | new | same |

### Sprints 8 and 9: where the 1:1 constraint is LIFTED

Added 2026-08-08 by Víctor, deliberately at the END. Sprints 2-7 deliver a faithful port; these
two are the second half of "1:1 first, improve later", one per window, and neither starts until
the window it covers is on `dev` rendering real data.

Same shape both times: a **Fable agent framed as a senior dashboard-analytics designer**
(`~/.claude/projects/.../feedback_tab_migration_flow`, step 7) is handed (1) the before-and-after
screenshots in the states that matter, (2) the files that render the window, (3) the backend
wiring — `host.py`, `stream_client.py` and the golden payload, so no proposal can ask for a field
the tick does not carry — and (4) `tokens.css`, so proposals speak the design system. It returns
prioritised P0-P3 findings with `file:line` and a concrete fix, covering **layout** (hierarchy,
density, what earns the top-left) and **surgical** changes (a colour semantic, a truncation, a
chart baseline, an undesigned state).

**Where the skills come from, decided by Víctor 2026-08-10: the catalogue, not improvisation.**
Before either elevate sprint runs its audit, install from `npx ui-skills` and from `nutlope/hallmark`
per `~/.claude/FRONTEND_TOOLKIT.md`, which is the standing directive for all frontend work in every
project and names the flow: `hallmark` **Audit** for the anti-AI-look dimension, `baseline-ui` for
the deslop pass, `pbakaus/audit` for a11y and anti-patterns, `fixing-motion-performance` for
anything that moves, `optimize` for load. Route with `npx ui-skills start` and pick by topic →
stack → specificity rather than loading everything. The Fable agent is then handed the installed
skills alongside the screenshots, so its findings speak the same vocabulary the fixes will use.

**Sprint 8 (AGENTS)** also pays the debt sprint 3 takes on knowingly: the two tooltips still emit
Qt's restricted rich-text dialect because the Python formatters are reused as-is, and whether the
formatters get a TypeScript reimplementation at all is this sprint's call.

**Sprint 9 (DATA)** starts differently — its bands were designed fresh, not ported, so the
question is less "did we lose something" than "does a strategist trust this at a glance".

**Retiring Qt in sprint 7 does not destroy the baseline**: the captures are committed under
`documents/dev_docs/migration/pitwall/`, which is why they live in the repo and not in a session
scratchpad.

Deferred beyond this plan, unchanged: #282 (observability contract, independent and also feeds the
Rival Agent), #286 (rival intent, gated on the Rival Agent), #287 (the parity gate).

---

## 2. Sprint 1 — the wire

**Why it is first and separate.** It is the producer, not PITWALL. All of it is verifiable **today**
against the existing PySide6 telemetry window, which simply receives better data. Landing it first
means PITWALL starts against a wire that already works instead of moving two things at once. It
also fixes a live user-visible bug on the way.

**Order.** #842 and #841 both edit `_build_arcade_snapshot`, so they are sequential; #842 first
because it carries the `CACHE_VERSION` bump and a stale cache should be invalidated once, not
twice. #843 lands on top. **#844 touches `src/arcade/overlays.py` only and is independent**, so it
can be worked in parallel or slotted anywhere.

| PR | Issue | Touches | Notes |
|---|---|---|---|
| 1 | #842 | `data.py`, `config.py`, `app.py` | store `global_t_min` on `SessionData`, bump `CACHE_VERSION` from `v7`, add `active` + `rel_dist` per car, add `global_t_min` + `location` once |
| 2 | #841 | `app.py` | span instead of sample; explicit branches for pause and rewind |
| 3 | #843 | `app.py`, `stream.py`, `tests/` | `schema_version`, `seq`, golden-payload test |
| 4 | #844 | `overlays.py`, `tests/` | kill the 55.56 divisor and the per-lap double count |

**Exit criteria.** Four PRs on `dev`, CI green on each. `f1-arcade --strategy` runs and the Qt
telemetry window shows visibly denser traces at 4x and 8x than it does today. A stale cache
regenerates rather than being mis-read. **The sprint stops on `dev`. #841-#844 stay OPEN** until the
single promotion after sprint 7.

**Trap.** The `CACHE_VERSION` bump invalidates every cached session pickle, so the first run of each
GP after PR 1 pays a full reload. Say so in the PR body; do not let it be mistaken for a regression.

---

## 3. Sprint 2 — the vertical slice

The thinnest thing that proves the whole chain: `f1-arcade` spawns `python -m src.pitwall`, two
pywebview windows open, and both render the live lap number and playback state from the real
broadcast.

**What lands:** the `src/pitwall/` package (`__main__`, `config`, `host`, `stream_client`), the Vite
project with two entry points, `bridge.ts`, `frameClock`, the `tokens.css` copy, and the spawn +
teardown wiring in `app.py` alongside the existing Qt spawn (both run during sprints 2-6; the Qt one
dies in sprint 7).

**Three things this sprint must get right, because they are expensive to change later:**

1. **`get_tick(since_seq)`, never a blind slot.** Gate A measured two independent 10 Hz pollers
   against one slot: the windows read a different frame on 58% of polls, with 15 duplicate reads
   and 15 skips out of 54. The sequence from #843 is what removes both.
2. **Closing one window must NOT stop the shared TCP client.** Today's two-client design advertises
   independent close as a feature (`dashboard/__main__.py:6-8`, `telemetry_window.py:78-82`). One
   shared client is the single place that regresses, and only if nobody writes this down.
3. **The token drift test covers every copy, not just the new pair.** Gate B measured that the drift
   A16 warned about **has already happened**: the Python palette in `config.py` / `theme.py` and the
   webapp's `tokens.css` disagree on every semantic colour. A test guarding only pitwall-vs-webapp
   would leave the broken pair uncovered, which is this repo's most-repeated defect committed inside
   the fix for it.

**Exit criteria.** Both windows open from `f1-arcade`, show live data, survive closing one, and the
drift test is red-then-green on a deliberate token change.

---

## 4. Sprint 3 — the AGENTS window, 1:1

**The checklist already exists.** Gate B produced the field-by-field inventory of
`src/arcade/dashboard/window.py:141-207` and its widgets. Use that as the acceptance list; do not
port from memory.

Layout is frozen: HeaderBar, horizontal split at 540/740, left column
(OrchestratorCard / ScenarioBars / ReasoningTabs), right column 3x2 AgentCards (Pace+PaceChart,
Tire+TireChart, Situation, Pit, Radio, RAG), StatusBar.

**The five things Gate B found that are NOT pure dict-in/string-out**, and therefore need a decision
rather than a port: the `QSyntaxHighlighter` doing regex highlighting in the reasoning tabs (no
TypeScript home is planned for it), two tooltip builders constrained to Qt rich text,
`classify_action` living outside `agent_formatters.py`, and a second parallel formatting layer
(`_LINE_BUILDERS`) the architecture document never named.

**Two accumulators survive here** and they are lap-keyed, not frame-keyed: PaceChart and TireChart
own their series because `history_tail` strips `per_agent`. Gate A finding D-11: a frame-indexed
truncate cannot address a lap-keyed map, and truncating destroys `per_agent` predictions that no
channel can rebuild. **Re-specify `frameClock` in laps here**, and use keyed maps rather than arrays
so a duplicate tick is idempotent.

One bug fixes itself: Qt gives ReasoningTabs about 268 px, so the decision-memory counterweight
sentence falls below the fold. HTML has no fixed heights.

**Gate at the end:** an adversarial pass whose only question is *what is different from the Qt
window*, with the two rendered side by side.

---

## 5. Sprints 4 to 6 — the DATA window

> **⇄ ORDER CHANGED 2026-08-09 (Víctor).** Band 4 goes FIRST, then bands 1-2, then band 3. The
> sections below keep their original band numbering; only the sprint each lands in moved.
>
> **The sprint labels below were renumbered on 2026-08-11 to match** — they had been left at the
> pre-reorder values, so the same file said *"Sprint 4, bands 1-2"* two screens under a note saying
> the BULK reader arrives with *"the sprint-5 selector"*. **The band is the stable identifier and
> the sprint number is not**; cite bands when the two could be confused.
>
> **Why.** Band 4 is the one band with an EXISTING ORIGINAL to port: the Qt telemetry window
> (`src/arcade/dashboard/telemetry_window.py` + `telemetry_panel.py`, whose docstring carries the
> 2x2 layout — Delta Time / Speed over Brake / Throttle, locked axes, a per-trace legend). That is
> exactly the setup that made sprint 3 work: a reference to compare against field by field, so
> fidelity is checkable instead of being a matter of taste. Bands 1-3 have no original at all —
> they are new design, and the 1:1 discipline cannot apply to them.
>
> **Its dependencies are already met.** Band 4 was scheduled last because the traces needed the
> telemetry span (#841) and the ring needed `rel_dist` (#842). Both landed in sprint 1. Nothing
> blocks it today; it was last only because of the order this plan happened to be written in.
>
> **What the reorder costs.** Bands are panels inside one window, so "each sprint ends with
> something on screen" survives — band 1 being the frame the others sit in is a container
> question, not a dependency.
>
> ✅ **ASKED AND ANSWERED, 2026-08-10: the wire carries everything band 4 needs, with zero
> blocking producer changes** — the opposite of band 1's answer. A Fable design gate measured it
> on the real Melbourne 2025 session (154,173 frames × 20 drivers) by driving
> `_build_arcade_snapshot`'s own functions; the report and its field-by-field implementation
> contract are at `~/.claude/plans/pitwall-sprint4/wire-band4-design.md`. Not luck: #841's span,
> #842's `rel_dist`/`active` and the #857 batch had already covered it.
>
> **Two scope lines the gate forced, because the prose above promises more than sprint 4 can do:**
>
> - **"Pinned rival" in sprint 4 means `driver_rival`, nothing else.** The wire carries telemetry
>   spans for exactly two cars and BOTH are chosen in the arcade process; `host.get_lap_trace`
>   and `session_data.py` do not exist (the architecture doc claimed otherwise and has been
>   corrected). Arbitrary pinning arrives with the sprint-5 selector and the BULK reader. The Qt
>   original renders exactly main + `driver_rival` too, so 1:1 is unaffected.
> - **The ring is schematic, and it has to be.** `ref_lap_xy`, `circuit_rotation_deg`,
>   `ref_lap_drs` and per-frame `x`/`y` all exist in the loader and none of them crosses the wire.
>   `rel_dist` is a fraction of the car's OWN lap, so a dot sits a median 1.3° and up to 24° (a
>   pit lap) from its true circuit position. An outline ring is a new host capability, not a tweak.
>
> **And one thing band 4 will NOT chart: gear and DRS.** They are on the wire and they carry real
> values, but the Qt original charts neither, `§3.5` of the realism doc describes the *elevated*
> surface rather than this port, and `drs` is the raw FastF1 code whose open set `{10, 12, 14}`
> lives only in `src/arcade/track.py:40` — a TypeScript copy would be a cross-language twin of the
> exact kind `driver_colors` is on the wire to prevent. If they are wanted, the producer publishes
> a decoded `drs_open` first, in sprint 9.


Built band by band so each sprint ends with something on screen.

**Sprint 5, bands 1-2.** Status strip, timing table, bests. Introduces the BULK reader over
`laps.parquet`, which is (927, 35) on the race checked and carries everything the tower and the
bests need.

> ✅ **THE BULK READER IS BUILT** (2026-08-11, ahead of the sprint): `src/pitwall/session_data.py`
> + `PitwallHost.get_bulk` + `/api/bulk` + `bridge.ts::getBulk`. **Band 1 needs nothing from it** —
> its four items are all on the tick already; only the connection label is missing and that is
> five lines when the strip is drawn.
>
> A Fable design gate measured the contract first, on the real race and the real
> `_gap_label` chain. **Do not re-derive it**: `~/.claude/plans/pitwall-sprint5/bulk-reader-design.md`.
> Its load-bearing answers: the HOST masks (not the client) · resolve on `location`, never
> `gp_name` (2025's folder is `Miami_Gardens`, its calendar key is `Miami`) · `FastF1Generated`
> rows are rendered but counted in NOTHING (their `Time` sorts before the field: a naive ranking
> puts the lap-1 crashers P1-P3 for 172 s) · the gap SECONDS come from the parquet, which is the
> official clock, while ORDER, status and laps-down keep coming from the wire · `_gap_label`'s
> four branches must be ported as ONE helper or the 22-minute-stale interval returns.

Two rules that must be written into the code, not assumed:

- **The reveal is per driver and strict**: reveal driver *d*'s lap *L* iff
  `L <= wire.drivers[d].laps_completed`. **This used to be written against `lap`**, which is a
  rounded interpolation of a step function: measured non-monotone on 101 frames of 2.49 M, so it
  flickers a lap open a tick early at the line, and it never opens a finisher's final lap.
  `laps_completed` is read off the crossing map and published per driver since #857.
  Gate A measured that at **96% of instants the running field spans 2 or 3 different laps**, and the
  tick carries only the main driver's lap. Masking everyone at the main driver's lap lags the
  leaders by a lap and leaks 1-2 laps of look-ahead for cars behind, simultaneously.
- **`race_order` is meaningless until every car has completed a lap.** On frame 0 the field is
  ordered by millimetres of accumulated distance (measured: HUL "leads" by 6 mm), and through
  lap 1 each car's fraction is normalised by its OWN first-lap length, which biases the back of
  the grid (measured: a car starting P7 reads P2). `gaps.py` says so - "excluding the opening
  lap, where no classification exists yet" - and the wire publishes it anyway. **Band 1 must
  render the tower as provisional until `laps_completed >= 1` for every driver**, exactly as a
  broadcast timing tower does.
  ⚠️ **"Every driver" had to become "every driver still IN the race", and the real race is what
  said so.** Read literally the rule never switches off on Melbourne 2025: SAI, DOO and HAD
  crashed on lap 1, so their `laps_completed` stays 0 for the whole afternoon. Measured against
  the live wire at lap 23 - three of twenty drivers under one lap, and all three OUT. A chip that
  is permanently lit marks nothing. The test is over the cars that can still contribute a
  classification, which a stopped one never will (#922).
- **A rewind UN-reveals.** `laps_completed` falls when the clock goes back, so a lap that was
  open must close again; a reveal cache keyed only on "seen once" leaks the whole future after
  one seek to the end.
- **The bests panel RECOMPUTES from the revealed subset; it does not trust `IsPersonalBest`.**
  **This rule used to sit in the sprint-6 paragraph below**, which owns band 3 — but the bests
  panel is band 2, so sprint 5 either applies it or ships trusting the flag and sprint 6
  "corrects" a panel it does not build. The column is safe under masking (Gate A: a running flag,
  18-24 flagged laps per driver, not a session-final one) but the two sequences are **not**
  identical: measured on Melbourne 2025, they differ on **47 lap-flags across all 20 drivers**,
  in both directions, concentrated on the wet-start laps {1, 5, 6, 7}. They converge only at the
  final frame — the last flagged lap is the session best for all 20 — and mid-race is the only
  state a masked panel ever renders. Recompute over `lap_time is not None and not deleted and not
  generated`: a deleted time does not count, and a generated row has no time at all. The column
  also holds a literal `None` alongside True/False, so it never crosses the bridge as a third
  state.
- **The gap column is lap-quantised and says so on screen.** Take it from the BULK reader over `laps.parquet`, labelled
  as at-the-line. **Two API names this line used to give are dead ends**: `get_rival_states` is a
  simulation-layer method PITWALL cannot reach, and `overlays._gap_value` no longer exists (#844
  removed it). The lap-quantised intent survives both.
  A precise-looking wrong number on a fidelity surface is the P3 A2 defect class.

> ✅ **SPRINT 5 SHIPPED 2026-08-13** (#922, #924, #926, #928, #930), and three things about it are
> worth carrying rather than rediscovering.
>
> **The four bands are two COLUMNS, not four rows.** The window's real client area is 1485 x 833
> logical (measured DPI-aware on the open window, not the 1500 x 950 `WindowSpec` asks for), which
> leaves 790 px for bands. Band 1 (29) + a full 20-row tower (439) + band 4's floor (420) + gaps is
> **908**, over by 118 with band 3 still at zero, on the largest screen in the fleet. So: band 1
> full width on top, then a 630 px left column (tower over bests) and a right column holding band
> 4. That is also the zoning a real wall uses - the all-cars world and the own-car world sit on
> physically different surfaces. Budget: `~/.claude/plans/pitwall-sprint5/band-height-budget.md`.
>
> **The SECTOR columns are the lap IN PROGRESS, and that is a second reveal coordinate.**
> `laps.parquet` carries `Sector{1,2,3}SessionTime`, so a sector is revealed at the instant it was
> crossed and the columns blank at the line and fill as the car goes round. It does not weaken the
> rule above: `L <= laps_completed` is the rule for lap ROWS, which only exist once the lap is
> over. It is a separate reader because a sector opens somewhere in the field every **2.22 s**, and
> a clock-driven mask over the whole bulk would re-send up to 342 KB at that cadence (#930).
>
> **The wire's order and the parquet's clock disagree at about 0.7 % of crossings, by one place.**
> Seen live on the tower: P11 read `+45.83s` above P12's `+42.79s`, and P12's INT rendered a dash
> because the sign guard refused a negative interval. The tower renders the WIRE order because
> only the wire can order mid-lap; the seconds come from the parquet because it is the official
> clock. Both are right, they are just different measurements. **Do not file it as a bug.**

> ✅ **SPRINT 6 SHIPPED 2026-08-15** (#939 the radio/RCM feed, #941 band 3, #942 the exit gate's
> findings). Four things about it are worth carrying rather than rediscovering.
>
> **The radio data is on the wire for ONE LAP, not for the race.** `strategy.per_agent.radio` does
> carry the transcripts, but `StrategyState.snapshot_dict` strips `per_agent` from every entry of
> `history_tail`. A chronological FEED built from the wire has to accumulate client-side, which
> keeps events across a rewind, starts empty on a mid-race attach, holes at 8x and dies without
> `--strategy`. It is read from disk and masked by the same reveal instead - the shape
> `session_data.py` already argued for - and rides IN the bulk payload, because it is a pure
> function of that channel's own signature. Measured: +9 % on a payload of 337 KB at full reveal.
>
> **Band 3 kept the real client's orientation, and the measurement is why.** Transposed gives cells
> 13.5 px wide against 19-22 px of text; with the ring still mounted the grid gets 555 px and 1,101
> of 1,140 cells clip. The lap time is `m:ss.d`, because the seconds form clips 205 of 1,140 on the
> real payload the moment a cell carries a pixel of padding. The ring and the radio feed hide on
> that tab.
>
> **The heat colour ranks each lap AGAINST ITSELF.** On the real payload the median lap is +13.79 %
> off the session best and 82.4 % of the race sits past +10 %, because Melbourne 2025 was wet and
> ran safety cars - any fixed percentage band paints four fifths of the grid one colour.
>
> **The screenshot found two defects that "0 cells clipped" could not:** at 10 px nothing clips and
> the grid is still unreadable because adjacent cells touch, and the columns re-sorted themselves
> every time two cars swapped position. WHICH drivers comes from `race_order`; the ORDER is by car
> number.
>
> Design record: `~/.claude/plans/pitwall-sprint6/`. **Note for whoever reads that folder:** the
> band-3 design gate died twice mid-run and its deliverables 3-7 were completed by the orchestrator,
> marked as such inside the report rather than passed off as an independent verdict.

**Gate at the end of sprint 6 — RUN.** Eleven findings, one HIGH: a deleted lap was painted the
FASTEST tone, because the ranking excludes deleted times and `indexOf`'s `-1` shared a branch with
"top third". On the real race the slowest car on the lap wore the green the legend means as
quickest. Two of its findings against the fixes were artefacts of it reading the working tree during
a red-check window - **never run a mutation while a gate reads the tree.** Report:
`~/.claude/plans/pitwall-sprint6/correctness-gate.md`.

**What sprint 5 finished, and the one piece of its own scope it did not.** Bands 1 and 2 are on
`dev`: the status strip, the twenty-row tower with the sector colour code, the bests panel, and a
live sector reveal the band list never asked for (#930, because the columns were showing the
previous lap for the whole of the next one).

**The band-2 spec's SELECTOR is not built** - *"pinning a row overlays its traces in band 4"* -
and the reason is a decision rather than an omission. The tick carries a telemetry span for **2 of
20** drivers, and which car is the rival is the arcade's choice. Measured on the real socket: the
whole tick is 17,278 bytes and the telemetry block 1,161 of them, so all twenty carrying a span is
**1.6x the tick, 271 KB/s at 10 Hz** - affordable. What is NOT affordable is the alternative: a
control channel would end PITWALL's read-only-follower posture, which section 3.4 of the realism
doc parks as a v2 question. Víctor's call (2026-08-14): take the option that breaks nothing, so
the producer publishes all twenty spans - **and it lands with #199 phase D.3, never before it**,
because `snapshot_dict` still re-runs a recursive `asdict` behind a blocking `sendall` ten times a
second and 1.6x makes that worse inside the render loop. Filed as **#936**.

There is no route through disk: **there is no telemetry on disk** (gate A). The 25 Hz frames live
in one monolithic pickle, which is why #841 put the span on the wire at all.

**Sprint 5's own deferred item, filed rather than improvised:** #931, sub-lap gaps from
`intervals.parquet`, whose per-driver sample cadence is a measured **4.27 s median**. It is blocked
on a time anchor (its `date` is UTC; every clock here is SessionTime, so it needs FastF1's
`t0_date` - the same shape as #842) and on the decision that it would put a THIRD clock on one
column, covering 19 drivers of 20.

**Sprint 4, band 4 — SHIPPED 2026-08-10** (#897, #898). Own-car traces stacked on one x axis with
a shared vertical cursor, pinned rival overlaid and labelled broadcast tier, and the ring.
**Depended on sprint 1**: the traces need the span (#841) and the ring needs `rel_dist` (#842).
Retired cars must render as retired, not as pending, which needs `active` (#842).

✅ **#857 is DONE** (the prerequisite below is cleared). The wire now publishes `race_order`,
and `laps_completed` / `progress` / `has_finished` per driver, plus `track_status`, all from the
producer's own `_rank_drivers` so the wire and the arcade panel cannot drift apart. The original
statement of the problem follows, for the record.

⛔ **Bands 1-2 had a prerequisite: #857.** The wire publishes only `lap`, `dist` and `rel_dist` per
driver — the two coordinates #844 spent a sprint refuting, plus a fraction. It publishes no race
order, no `progress` and no interval, and a consumer that attaches mid-race cannot rebuild the
crossing map from a 10 Hz snapshot stream. Decide what the timing band actually needs and publish
it from the producer, which already owns `self._gaps`; otherwise the DATA window re-derives the
order from the refuted coordinates and this repo's dominant defect lands one more time. It must
come after #855, which changed what `progress` returns for a finisher.

**Gate at the end of sprint 6:** adversarial, on tier discipline and on every claim the surface
makes about what it is showing.

---

## 6. Sprint 7 — retire Qt, package, fix the prose

⛔ **Two modules under `src/arcade/dashboard/` MOVE, they do not get deleted.** Sprint 3 made
PITWALL render the AGENTS window by calling the Qt window's own formatters, which is what makes
the port 1:1 by construction rather than by inspection. `agent_formatters.py` and the reasoning
line builders are therefore live product code with a consumer that outlives Qt; deleting the
package wholesale takes the AGENTS window's entire content layer with it. Move them to
`src/pitwall/` and keep going. `src/arcade/palette.py` already lives outside the package for this
reason and needs nothing.

**Deleting the rest of `src/arcade/dashboard/` is bigger than it looks.** Gate B built the reference graph; the
one that would not have been found by a doc sweep is
`tests/agents/test_overtake_domain.py:231`, which imports `format_situation` from the dashboard
package for a **domain** test. Also: `tests/surfaces/test_arcade_dashboard_imports.py` (13 modules),
`src/arcade/app.py:332`, `src/arcade/stream.py:6` (docstring), `docs/pages/arcade-quick-start.md:34`,
two drawio diagrams, and `documents/audits/AUDIT_P2_LOADING.md:71`.

**Packaging has zero precedent** (Gate B, finding H). CI has never run a Node build step, package
data covers only yaml/yml/json, and the repo already has a scar from that same mechanism leaking
`node_modules` into a wheel. Treat "the Vite build ships inside the wheel" as real work with its own
PR and its own verification: build a wheel, install it in a clean venv, run it.

**Also in this sprint, because leaving it is how documentation starts teaching bugs:**

- `ROADMAP.md` v2.6.0 and `docs/pages/roadmap.md:464` say the surfaces move to a "web-native view".
  Web technology yes, web app no.
- `src/arcade/__init__.py:4` claims Arcade renders the SSE simulation stream. It has not for a long
  time.
- `documents/dev_docs/diagrams/arcade_3window_architecture.drawio` describes three windows that stop
  existing, and its name is part of the claim.
- PySide6 and pyqtgraph leave `pyproject.toml`. Gate B correction: **there is no `arcade` extra**;
  they are unconditional core dependencies today.

**The exit gate, and then the single `dev -> main` of the whole programme.** Adversarial, per
`~/.claude/ADVERSARIAL_AUDIT.md`. Never close a piece of work this size on "all the sub-issues
merged" — and here that temptation is at its strongest, because seven sprints of commits promote at
once. Only after that promotion do #841-#844, #281, #284 and #285 close.

Note that this promotion is also what un-gates release PR #712 (`chore(main): release 2.6.0`), which
has been held open waiting for PITWALL to exist. Do not merge it before this point.

---

## 7. Patterns that apply throughout

Full directives in `PITWALL_V2_ARCHITECTURE.md` section 6. The five that will actually be violated:

1. **Chart data goes into ECharts imperatively through refs, never through per-frame React state.**
   This is P3 finding A6 (six cards, six syntax-highlighted text areas and a full chart rebuild ten
   times a second for content that changes once a lap) applied before it happens instead of after.
2. **Animate the entrance, never the update.** `useFirstPaintAnimation` in the webapp already
   encodes the contract. Port it, do not reinvent it.
3. **Accumulating panels use keyed maps, never arrays**, so a duplicate tick is idempotent. Duplicate
   ticks are routine below 1x and permanent while paused.
4. **`None` means unknown data; exceptions mean a failed operation.** Never a sentinel a search could
   also find. The repo's scar: a NaN `Position` defaulted to `0`, so the leader looked for the car
   ahead at position 0 and found the one that had just crashed.
5. **One reader, one resolver.** Gate A finding D-15: "reuse the loader Arcade uses" is not a
   well-formed instruction because Arcade uses three, on two different roots. Resolve the race
   directory through the existing resolver, the root through `get_data_root()` only, and run with
   `F1_STRAT_OFFLINE=1` so PITWALL can never race the arcade into a download.

## 8. Testing

Per sprint, not at the end. The Qt surface's coverage today is one import-smoke file (P3 A18); the
bar is low and easy to clear.

- **Sprint 1**: sample continuity across a speed change and across a seek; the golden payload; the
  lapped-car gap case (#844's cause 2 is largest exactly there).
- **Sprint 2**: token drift across ALL copies; every `js_api` method returns JSON-serialisable data
  (a numpy scalar or a `Timestamp` fails silently across the bridge, which is the worst kind);
  `frameClock` truncation.
- **Sprint 3**: the formatter layer, which is dict-in/string-out and was never tested.
- **Sprints 4-6**: the masking rule, per driver and strict, including a lapped car and a lap-1
  retirement.
- **Sprint 7**: a wheel built, installed clean, and run.

Assert the **effect**, never the constant. A test that asserts `CFG.x == 0.2335` passed for a year
over a threshold that could never fire.

## 9. Carried forward, unresolved

- **Race control messages: RESOLVED 2026-08-09 — they already exist, and an earlier note here was
  wrong.** That note said they had no producer, on the strength of `SessionData.events` being
  empty. `SessionData.events` is a dead field; the messages come from `src/nlp/radio_runner.py`
  (out of `rcm.parquet`) as `RaceState.rcm_events`, and the **Radio card already renders them**
  in its body and tooltip — so PITWALL's AGENTS window shows them today, alongside the radio
  transcriptions and N29's verdict. A real Melbourne run reports 90 of them.
  So the sprint-4 question is only WHERE they belong: the flag STATE is the status strip
  (`track_status`, already on the wire); the message TEXT already has a home. If the DATA window
  wants its own ticker, that is one additive wire field from an existing producer, not a pipeline.
  Full source map: `PITWALL_SPRINT4_SOURCES.md`.
- **A driver whose telemetry drops mid-race under-reveals** (OBS-4). An officially-Finished car
  with a dropout gets its flag crossing at the dropout frame, so `laps_completed` stops there and
  the reveal withholds that driver's later parquet laps. Data-honest - the replay has no telemetry
  to place them - and unreachable on Melbourne, where all twenty run to the line. Written down so
  it is not filed as a reveal bug.
- **A latent P2 for whoever writes any trace reader**: the arcade's `lap` channel is interpolated,
  so it can label ~2,000 frames as a lap that has no telemetry behind it. Gate A reproduced the
  mechanism by executing the code path but could not show it firing on real data.
- **#199 Phase D.3 gains weight.** `snapshot_dict` re-runs a recursive `asdict` over the 30-entry
  tail ten times a second with a blocking `sendall`, so one stalled subscriber can hitch the pyglet
  frame loop. Inherited, not introduced, but the wire is now load-bearing.
- **#283 must not be closed wholesale.** Its relay premise is void, but its gap-provider bullet
  (`intervals.parquet`) is exactly what the timing table's live gaps need once `global_t_min` ships.
