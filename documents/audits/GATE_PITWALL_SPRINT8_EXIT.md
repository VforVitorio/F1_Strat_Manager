# EXIT GATE — PITWALL sprint 8 (the AGENTS elevate pass)

**Date:** 2026-08-17 · **Branch:** `sprint8-integration` vs `origin/dev` (10 commits ahead, 17 files, +673/−144)
**Role:** adversarial exit gate. Success = finding what is STILL broken. No repo file modified.

## Checklist

- [x] Read the full diff `origin/dev...sprint8-integration` (1312 lines) + the dev-side sprint-8 PRs (#969, #971)
- [x] Claim A — veto highlight: enacted-not-scored, ties, case/format of `enacted_action`, 4 scenarios → **F1, F2**
- [x] Claim B — 6% floor: rendered width measured 23 px of a 383 px track → HOLDS
- [x] Claim C — `readable_on` / `legible_fill`: 32³ sweep converges, never touches a passing colour, SOFT deepens to #d52f31 (still red, 4.91:1); token-test counts → **F6** (two new boot copies unguarded)
- [x] Claim D — rendered at 1485×833 (s1-s4) and 1200×700 (s1, s4): no phantom boxes, nothing clipped → HOLDS
- [x] Claim E — degenerates fine (single/identical/empty/None; no inversion; `lapAxis(null)` falls back) but the +22 s in-lap case FAILS → **F3**; and the whole fix is guarded by NO test → **F7**
- [x] Claim F — tooltip path clean (typed segments, `null` sentinel, 4-chunk cap + footer real, both call sites), but the body lines beside it still push unescaped wire text through `dangerouslySetInnerHTML` and the new comment claims otherwise → **F5**
- [x] Claim G — happy path verified RENDERED for the first time (x4-chip.png); float/string `lap_number` → **F4**; duplicate laps OK; future laps OK
- [x] Claim H — verdict below
- [x] Bug-class hunt: twins (F5, F10), false headlines (F3 docstring, F5 comment), missing guard (F7), sentinel (none new), unmigrated stub (#974 already filed by the team — verified, not re-reported), stale references (F9)
- [x] `npm run build` green · smoke-agents 19 OK · smoke-data 147 OK · pytest tests/surfaces 218 passed
- [x] Comments/docstrings vs behaviour → F3, F5, F9
- [x] Design gate's unfixed P2/P3s recorded? → **F12: they are not, and it matters**
- [x] "What I tried to break and could not"

---

## Progress notes

- Sprint-8 scope spans BOTH `origin/dev` (PRs #969 scenario-truth, #971 contrast — claims A/B/C code)
  and the 10 branch commits (claims D-G + tooltips + chip). Audited at the branch checkout, which
  contains both.
- Executed so far: `uv run pytest tests/surfaces/ -q` → **218 passed**; `npm run build` → green
  (agents bundle 9.56 kB); `node scripts/smoke-agents.mjs` → **19 checks OK**;
  `node scripts/smoke-data.mjs` → **147 checks OK**; `gate_exit_attacks.py` (scratchpad) — the
  lettered edge cases against the real producer functions.

## Findings (appended as confirmed)

### F1 · P1 — Claim A fails its first attack case: a legal enacted action outside the four scenario keys re-crowns the vetoed winner, and `is_enacted` lies on the wire
- **Where:** `src/pitwall/agents_view/decision.py:252` (`highlighted = enacted if enacted in raw else winner`), against `src/agents/strategy_orchestrator.py:273` (`_ACTION_VALUES = Literal["STAY_OUT", "PIT_NOW", "UNDERCUT", "OVERCUT", "ALERT"]`).
- **Executed evidence:** `build_scenarios({"PIT_NOW": 0.71, "STAY_OUT": 0.29}, "ALERT")` →
  `PIT_NOW: is_enacted=True, note='', bar=#a78bfa fill=100.0`. ALERT is a LEGAL orchestrator
  action (the fifth member of the Literal), and when it is published the panel crowns PIT_NOW
  with the full enacted regalia and no `VETOED`, while the badge one card up says ALERT.
- **What a strategist experiences:** the exact #962 misread, reincarnated for the fifth action —
  the "why" panel says the system is executing PIT while the badge says ALERT. And `is_enacted:
  true` is now FALSE DATA on the wire for any consumer.
- **Fix:** when `enacted` is truthy but not in `raw`, highlight NOTHING (`highlighted = None`) and
  keep the vetoed logic keyed off the winner only if a veto actually happened; or add the enacted
  action as a rowless chip. The one-line version: `highlighted = enacted if enacted in raw else
  (None if enacted else winner)` — plus mark the winner `VETOED` in that branch too, since it was
  equally overruled.

### F2 · P2 — A tie at the top fabricates a `VETOED` mark and renders BOTH best candidates at the 6 % floor
- **Where:** `decision.py:247` (`max(raw, key=raw.get)` — first-max tiebreak), `decision.py:256-257`
  (`span = (hi - lo) or 1.0` → everything scales to 0 → floor).
- **Executed evidence:** `build_scenarios({"STAY_OUT": 0.50, "PIT_NOW": 0.50}, "PIT_NOW")` →
  `STAY_OUT: winner=True, note='VETOED', fill=6.0` · `PIT_NOW: enacted=True, fill=6.0`.
- **What a strategist experiences:** two candidates tied at the TOP both draw near-empty bars (the
  floor is doing duty as the whole bar), and the runner-up wears `VETOED` although no guardrail
  fired — the orchestrator merely broke a tie. A fabricated rule-violation flag on a routine lap.
- **Fix:** treat `hi == lo` as "no discrimination": fill 1.0 for all present rows (they are all
  the max), and set `vetoed_key` only when `raw[winner] > raw[highlighted]` strictly.

### F3 · P1 — Claim E fails its named attack case: one in-lap and the tyre y-axis is bounded to the SMOOTHED series, the raw point clips out, and the flatness the fix exists to cure comes back for the rest of the race
- **Where:** `src/pitwall/agents_view/charts.py:211` (`"y_range": _y_range(trend or flat)` — `or
  flat` is dead code: `trend` is empty iff `flat` is), `charts.py:246-266` (docstring: "Bound the
  lap-time axis to the laps actually plotted" — it bounds to the 3-lap rolling MEAN of them).
- **Executed evidence:** stint 14-22 at ~81 s + in-lap 103.4 s (inside the 30-200 s sanity window)
  through the real `build_tire_series` → `y_range=[78.53, 94.9]` while the plotted max is
  **103.4** — the in-lap point is OUTSIDE the axis and ECharts clips it (line-series default
  `clip: true`). The trend meanwhile ends `[22, 88.72], [23, 92.4]` — INSIDE the axis.
- **What a strategist experiences:** (1) at the decisive moment — the stop — the white trend line
  spikes ~11 s with the data point that explains it invisible; (2) from the first stop onward the
  in-lap stays in history, the span is ~16 s, and a real 0.05 s/lap degradation is back to ~4 px
  of a 150 px plot — the exact hairline the sprint's own commit message says it fixed, for every
  stint after the first.
- **Fix:** the design gate's own prescription — bound to the median of plotted values ±2.5 s
  clamped to the observed extent per visible window, or exclude points > (median + N s) from the
  bound and DROP them from the plot explicitly (an in-lap is not tyre-degradation data), so the
  axis and the plotted set agree. And make the docstring say which series it bounds.

### F4 · P3 — The `was <call>` chip trusts `lap_number` asymmetrically: a float/string current lap makes the chip quote the CURRENT call as the previous one
- **Where:** `decision.py:133-136` — tail rows must be `isinstance(int)` but `latest["lap_number"]`
  is only USED for filtering when it happens to be int; otherwise the filter silently disappears.
- **Executed evidence:** `build_orchestrator({"action": "PIT_NOW", "lap_number": 23.0,
  "plan_changed": True}, tail)` → `changed='was PIT NOW (0.71) · L23'` — the chip says the call
  moved FROM the call it moved TO, on the same lap. Same with `"23"` (string). A float in the
  TAIL rows instead suppresses the chip entirely (inconsistent: lies one way, goes silent the other).
- **Reachability:** low on today's wire (`lap_number=race_state.lap` is int end-to-end), so P3 —
  but the guard's shape is exactly the half-migrated-pair class this repo pays for.
- **Fix:** normalise both sides through one `_as_lap(value) -> int | None` and bail out (`""`)
  when the CURRENT lap is unknown, rather than comparing nothing.
- **Also (same function):** a previous row with `confidence: None` renders `was STAY OUT (0.00) ·
  L22` — a fabricated 0.00, the same invented-metric shape as the design gate's S6 (executed:
  probe G5). Omit the parenthesis when confidence is absent.

### F5 · P2 — Claim F's twin: the card BODY still pushes unescaped wire text through `dangerouslySetInnerHTML`, and the sprint's new comment claims it is covered
- **Where:** `src/pitwall/ui/src/features/agents/AgentCard.tsx:141,149` (`__html: card.headline` / `__html: line.text`), fed by `src/pitwall/agent_formatters.py:409-420` (`format_radio` body: RCM label + message, driver radio message — interpolated with NO escaping; `import html` was removed from the module this sprint, so nothing in it escapes anymore).
- **Executed evidence:** `git show origin/dev:src/pitwall/agent_formatters.py | grep html.escape` — all 9 sites were inside the two TOOLTIP builders. The body ticker never escaped, and still does not; the tooltip (which DID escape) is the half that got the structured-data fix. The same wire strings (`radio_events[].message`, `rcm_events[].message`) now reach the DOM safely via the tooltip and as raw markup via the body line.
- **What a reader experiences:** any `<` in a radio transcript (Whisper free text) or RCM message is swallowed as a tag in the body ticker — content silently disappears; a crafted payload is DOM injection. Meanwhile the sprint's OWN new docstring (`AgentCard.tsx:8-11`) says the headline and body lines "carry the compound pill and the flag chips - HTML spans built in `src/arcade/palette.py` with every free-text field escaped there" — TRUE of the pills, FALSE as a headline: the radio/RCM messages in those same lines are escaped nowhere. A true clause inside a false headline, in the comment written this sprint.
- **Fix:** `html.escape` the free-text interpolations in `format_radio` (and any sibling formatter interpolating wire text into body lines), or move the body lines to typed segments as the reasoning tabs already did. The pill spans stay as they are.

### F6 · P2 — The token test does not count the sprint's two new boot copies, and its subset assertion structurally cannot notice
- **Where:** `tests/surfaces/test_pitwall_tokens.py:213-230` (`BOOT_SLOTS`) vs `src/pitwall/ui/src/features/agents/AgentsWindow.tsx:79` (`action_text_colour: "#121127"` — a copy of `palette.BG_COLOR`) and `:134,148` (`cursor_colour: "#9ca3af"` — copies of `TEXT_TERTIARY`).
- **Executed evidence:** grep of AgentsWindow.tsx boot hexes against the `BOOT_SLOTS` keys — `action_text_colour` and `cursor_colour` are found by the test's own regex but appear in no slot; `assert set(BOOT_SLOTS) <= set(found)` is subset-only, so an unlisted copy passes silently. Flip `action_text_colour` to `#ffffff` (the exact 2.72:1 failure #971 fixed) and the suite stays green: the producer-side legibility test (`test_pitwall_agents_view.py:131`) checks `build_orchestrator`'s output, not the TSX boot literal.
- **Why it matters here of all files:** this test's own docstring doctrine is "a copy with no guard is how palettes drift", and `lib/chart.ts`'s WHERE-TO-CHANGE block names "AgentsWindow.tsx's boot literals, and the counts in test_pitwall_tokens.py" as one move. The sprint added the copies and did not add the counts.
- **Fix:** two rows in `BOOT_SLOTS` (`action_text_colour: "BG_COLOR"`, `cursor_colour: "TEXT_TERTIARY"`), and consider asserting the found-set equals the slot map so the NEXT new copy fails loudly.

### F7 · P2 — The sprint's headline chart fix has no guard at all: `y_range`, the shared lap axis and the cursor are asserted by nothing
- **Where:** `tests/surfaces/test_pitwall_agents_view.py` — `y_range` appears in NO test (grep: zero hits across tests/); `smoke-agents.mjs` carries the new fields in its fixture but never asserts the rendered axis extent, though the `__pitwallChart` handle exists for exactly that (its own comment: "a check written against the mechanism passes over a broken effect").
- **Executed evidence:** revert-probe by inspection — change `charts.py:211` to `"y_range": None` and every one of the 218 tests and 19+147 smoke checks still passes; the tyre chart silently returns to the 40-140 autorange the sprint exists to cure.
- **Fix:** one producer test (`build_tire_series` on a flat stint → `y_range` spans ≤ ~7 s and contains the data; pace `x_range is tire x_range`; `current_lap` echoes the lap) and one smoke line reading `__pitwallChart.getModel()`'s y extent.

### F8 · P3 — The borrowed lap axis can crop the pace chart's own data
- **Where:** `src/pitwall/agents_view/builder.py:71` — `build_pace_series(self._history.pace, tire["x_range"], ...)` unconditionally; `_x_range` (`charts.py:285`) computes the lower bound from the TYRE chart's plottable laps only.
- **Executed evidence:** pace history laps 10-23 with predictions, tyre rows 10-14 carrying `lap_time_s: None` (an SC/missing-telemetry shape) → `tire x_range = [14.5, 26.0]`, and **5 pace points (laps 10-14) fall left of the locked axis** and are clipped.
- **What a reader experiences:** the pace card silently loses its oldest laps whenever the tyre store lacks a sane lap time the pace store has a prediction for.
- **Fix:** the shared range should be the union of both extents — compute it in the builder from `min(pace laps ∪ tire laps)`, or pass the pace extent into `_x_range`.

### F9 · P3 — Two docstrings name functions this sprint deleted
- **Where:** `src/pitwall/agent_formatters.py:365` ("exposed via ``radio_tooltip_html``") and `:558` ("wires onto the card via ``rag_tooltip_html``") — both renamed to `radio_tooltip`/`rag_tooltip` in this very diff. The `format_radio` docstring also still narrates a "QLabel" (`:363-367`).
- **Fix:** rename in the two docstrings; s/QLabel/card body/ while there.

### F10 · P3 — The current-lap mark is implemented twice
- **Where:** `useEChart.ts::currentLapMark` (used by PaceChart) vs `TireChart.tsx:72-79` (the same mark inlined into its `marks` array). Both currently agree (`== null` guard, solid, width 1) — but they are the twin shape this repo's memory names as its dominant defect: change the cursor in the helper and the tyre chart keeps the old one.
- **Fix:** TireChart builds its marks list as `[...cliff, ...boundaries, ...(currentLapMark(...)?.data ?? [])]` or the helper grows a raw-datum variant; either way one code path.

### F11 · P3 — The `VETOED` note eats the vetoed row's track, so bar widths stop being comparable across rows
- **Where:** `ScenarioBars.tsx:36` (note rendered between bar and score inside the flex row) + `agents.css:385-386` (`.scenario-bar { flex: 1 }`).
- **Executed evidence:** in `s3-guardrail.png` the PIT row's track ends ~45 px short of the STAY row's (measured; STAY track border spans x 77-460). The vetoed winner's "full" bar is therefore drawn on a shorter axis than the rows it is meant to dominate.
- **Fix:** overlay the note on the track's right edge (`position: absolute`) or reserve its width on every row.

### F12 · P2 (process) — The design gate's eleven unfixed findings are recorded nowhere a future sprint will look
- **Executed evidence:** full open-issue listing (50 issues). #962-#968, #960 cover what this sprint fixed; #974 (filed during the sprint) covers the guardrail_reason wire gap. NOTHING records: S7 (dev language: "stub", "no radio/rcm pipeline output", "SC >30%" — still on screen in s4), S8 (PIT active renders as WARNING; "pit 22.40s" vs the pit_delta/pit_duration pair), S9 (radio picks recency over the driver's own PROBLEM), S10 (decision wears a button costume), S11 (ARIA/colour-alone), S12 (four pill dialects + inline hex on the wire), S13 (stub headline over a confidently drawn chart — still visible in s4), S14 (RAG has no reasoning tab), L3 (confidence at 11 px), L5 (540 px pin), L6 (scroll affordance). The gate report lives in `documents/audits/`, but this repo's own doctrine is that the TRACKER is the source of truth, and sprint 9 will be planned from issues.
- **Fix:** one umbrella issue ("AGENTS elevate pass, deferred design findings") linking `GATE_PITWALL_AGENTS_DESIGN.md` with the eleven letters, or individual issues for S8/S9/S13 (the three with misread potential).

---

## Verdicts on the eight claims

| Claim | Verdict |
|---|---|
| **A** | **HOLDS for the guardrail shape it was built for; REFUTED at its edges.** The veto render works (s3: enacted STAY takes the highlight, PIT keeps its grey fill + `VETOED`). But a legal enacted action outside the four keys (ALERT — the fifth member of the orchestrator's own Literal) re-crowns the overruled winner with `is_enacted: true` on the wire (F1, rendered proof x2-alert.png), and an exact tie fabricates a `VETOED` and floors BOTH top candidates to 6 % (F2, x1-tie.png). Case variants: `pit_now`/`PIT_NOW` fine; `"pit now"` (space) silently disables the mechanism — unreachable on today's wire, noted only. |
| **B** | **CONFIRMED.** The floor renders 23 px of ACCENT on a 383 px track at 1485×833 (measured from s3 pixels) — visible, not sub-pixel. Unscored rows draw no track at all and read `--`; the two states are unmistakable. Caveat: F2 makes the same 6 % also the rendering of a TIED TOP score. |
| **C** | **CONFIRMED, with one guard gap.** 32³-colour sweep: `legible_fill` always reaches ≥4.5:1 within its 12 steps, and never alters a colour already passing (0 of 32,768). SOFT deepens #e63232→#d52f31 — hue ratios R−G=166, R−B=164, unambiguously red — at 4.91:1. MEDIUM/HARD/INTER/WET unchanged. But the token test does NOT count every copy: the two new boot literals are unguarded (F6). |
| **D** | **CONFIRMED.** Fresh build + fresh states through the real host: s1-s4 at 1485×833 and s1/s4 at 1200×700 show no phantom `.agent-chart` boxes (smoke asserts count === 2 against the built bundle), text cards content-sized, no clipped content, the scroll-guard smoke's deliberate-overflow probe green. |
| **E** | **REFUTED at exactly the attack the brief named.** Degenerates hold (single point → ±2.5 pad; identical → same; empty → None → autorange; no inversion possible; `lapAxis(null)` falls back). But a +22 s in-lap: `y_range` is computed from the SMOOTHED trend (`trend or flat` — `or flat` is dead code), the raw 103.4 s point plots OUTSIDE [78.53, 94.9] and clips out, the trend spikes to 92.4 inside the axis with its cause invisible, and the ~16 s span re-flattens every later stint's trend to ~4 px — the bug's own geometry, back for the rest of the race (F3, rendered proof x3-inlap.png). Docstring headline ("the laps actually plotted") is false. And nothing tests any of it (F7). |
| **F** | **CONFIRMED for the tooltip path; the twin beside it was left behind.** Structured data end-to-end, `null` (not `""`) from both call sites, React text nodes, 4-chunk cap + `+N more` real, reasoning arrives as typed segments. But the card body lines still interpolate the SAME wire strings unescaped into `dangerouslySetInnerHTML`, and the sprint's new comment claims they are escaped (F5). |
| **G** | **CONFIRMED on the real wire's types, and now verified RENDERED (first time): x4-chip.png shows the chip with amber dot and tertiary text.** Future laps in the tail are filtered; duplicates resolve to the first max. But the type guard is asymmetric: a float/string CURRENT lap silently un-filters the tail and the chip quotes the current call as the previous one (F4); an absent previous confidence renders a fabricated `(0.00)`. Note: none of the five golden fixture states carries a `history_tail`, so no screenshot before this one had ever exercised the chip. |
| **H** | **The floor made B real and H honest, but did not make the bar truthful — and in one corner it now actively lies.** The docstring states plainly that the bar encodes RANK (`decision.py:225-229`) — the right kind of honesty for a deliberate decision. With four candidates the interior fills do carry relative margin (probe: +0.30/+0.71/+0.65/+0.10 → 32.8/100/90.2/6.0) but the scale always stretches to full span: four scores within 0.03 of each other render identically to four spread over 0.61, and the worst-scored draws the same 6 % whether it lost by 0.01 or 0.61. With two candidates the bar remains pure rank. At a tie (F2) the floor renders the TOP scores at 6 % — the one configuration where the old empty bar was arguably less misleading. Verdict: better (absent vs scored is real), not hidden (the docstring names it), one new corner where it is worse. |

---

## What I tried to break and could not

1. **`legible_fill` / `readable_on`** — 32,768-colour sweep: always converges inside 12 steps, never touches a passing colour, every compound pill ≥ 4.87:1, SOFT still reads red. The convergence argument holds: `away` computed once cannot oscillate, because moving the fill toward it only strengthens the chosen text's contrast.
2. **The scenario floor's arithmetic** — `fill_pct` 6.0 minimum for any scored candidate, rounding at 0.1 %, absent rows trackless with `--`; four-scored normalisation correct (probe A3); empty/None scores degrade to four trackless rows with no crown and no VETOED.
3. **`_previous_call` on the wire's real types** — int laps end-to-end (`lap_number=race_state.lap`), future-lap and duplicate-lap tails handled, first lap and no-tail suppress the chip, `plan_changed=False` suppresses it. The docstring's `strategy.py:439-440` citation is accurate (latest set and history appended under one lock).
4. **The tooltip structure test** — pins the exact dict shape, the uncapped message, the `None` sentinel, the 4-chunk cap and footer; both call sites hand `dict | None`; the TSX renders text nodes only; the `TooltipView` typing matches the producer.
5. **Claim D's geometry** — fresh bundle, fresh states through the real `PitwallHost`, shots at two viewports; smoke asserts `.agent-chart` count === 2 against the BUILT bundle plus the deliberate-overflow scroll-guard probe. The one-expression children fix is real (an array of nulls is truthy; a single ternary yields null).
6. **`lapAxis`/`secondsAxis` bound-label suppression** — `valueAxis` computes `locked` from its own spec; the rendered axes show interior ticks only (no 12.5, no 78.53 printed as ticks in s3/x3-inlap).
7. **The builder's restart/rewind logic around the new chart wiring** — the reordered `charts` computation changes no behaviour (same inputs, same tick); accumulate-before-build order preserved.
8. **The dev producer's fixture coherence** — `fixture_call` makes `memory_block` quote the same action/confidence `history` generates (checked lap 22: STAY_OUT 0.62 both ways); `guardrail_reason=None` matches the real path, whose own wire gap is already filed as #974 by the team (verified against `no_llm.py:302`, `engine.py:303`, `strategy.py:796` — deliberately not re-reported here).
9. **Idle-dim contrast scoping** — `.is-idle` now dims border/glyph/chart only; idle text arrives in TEXT_TERTIARY (6.88:1 on panel). The `readable_on` docstring's numbers (SUCCESS 7.29, ACCENT 6.80, WARNING 8.61, DANGER 4.92) reproduce under `contrast_ratio`.
10. **The 218-test suite, both smokes, and the build** — all green on the branch as checked out; the smoke fixture was migrated with the contract (`is_scored`/`is_enacted`/`note`, structured tooltip), not left to default.

**Not executed:** the real pywebview window against the live arcade producer (no race replay was run in this gate — all rendering evidence is the built bundle over states produced by the real `PitwallHost`); the `_previous_call` docstring's "40 lap pairs of a real race" measurement is taken on trust — noting that it implies the chip fires on ~none of that race's laps, i.e. the feature earns its keep on rare laps by design.
