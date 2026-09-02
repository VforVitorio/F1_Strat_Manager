# GATE — PITWALL · AGENTS, the elevate pass (sprint 8)

**Date:** 2026-08-17 · **Gate role:** senior dashboard-analytics design review, pixel-fidelity
constraint LIFTED. Success condition: find what is weak.

**Inputs consumed:** 5 screenshots at 1485×833 (`s0`–`s4`), their view JSONs (`s0` is literally
`null` — the idle view is 100% client defaults), the React feature (`src/pitwall/ui/src/features/agents/`),
`lib/agents.ts`, `styles/agents.css` + `qt-base.css`, the producer (`src/pitwall/agents_view/*`,
`agent_formatters.py`, `reasoning_lines.py`), the golden test, `tokens.css` + `arcade/palette.py`,
and the three installed skills (hallmark Audit, baseline-ui, fixing-accessibility).

## Checklist

- [x] Read the five screenshots
- [x] Read the five state JSONs
- [x] Read the React renderers + CSS (`AgentsWindow`, `AgentCard`, `OrchestratorCard`,
      `ScenarioBars`, `ReasoningTabs`, `PaceChart`, `TireChart`, `useEChart`, `lib/chart.ts`,
      `agents.css`, `qt-base.css`)
- [x] Read the producer (`agents_view/decision|panels|charts|builder|reasoning`,
      `agent_formatters.py`, `reasoning_lines.py`) + `arcade/palette.py` constants
- [x] Verify claim A — executed pixel diff of the STAY vs UCUT rows in s1
- [x] Verify claim B — `AgentsWindow.tsx:95` vs `decision.py:175`, plus a third variant
- [x] Verify claim C — CSS form analysis + affordance inversion
- [x] Verify claim D — s3 JSON: `is_winner` stays on PIT_NOW while the badge says STAY OUT
- [x] Verify claim E — tire trace occupies 4 px of ~150 px plot height (measured)
- [x] Verify claim F — dead-strip measurement per card (44–57 % on the text cards)
- [x] Verify claim G — 22 contrast pairs computed + 2 verified against rendered pixels
- [x] High-res crops: not needed — every string was legible at 1x and the state JSONs carry
      the exact text; measurements were taken from the PNGs directly
- [x] Free-hunt beyond the floor (7 additional findings)
- [x] "What I tried to break and could not"

**Executed evidence, in one place:**
- `scratchpad/gate_measure.py` — 22 WCAG pairs from the declared tokens; per-card ink % and
  dead-strip height on s1; rendered-pixel contrast of the dimmed idle card on s2 (2.38:1) and
  of the guardrail line on s3 (4.64:1); tire-chart data-ink height (4 px).
- A pixel diff of the STAY (+0.29) row against the UCUT (`--`) row in s1: **0 differing pixels
  in the bar track**; the only diffs are the 4-letter label glyphs and the score text.
- Fix-verification pass: `--qt-bg` text on SUCCESS 7.29:1, on ACCENT 6.80:1, on DANGER 4.92:1;
  idle `opacity: 0.75` still fails (4.39:1), 0.85 barely passes (5.31:1).

---

## Verdicts on the eight named claims

| Claim | Verdict |
|---|---|
| **A** runner-up vs absent scenario pixel-identical | **CONFIRMED** — the bar track is pixel-identical (executed diff: 0 px); only the 11 px score text differs. Worse than claimed: with two scored candidates, min-max maps the runner-up to an empty bar *regardless of margin* (+0.70 vs +0.69 renders full vs empty). → S2 |
| **B** idle `0%` vs live `--` | **CONFIRMED, and there are three variants**, not two: pre-first-tick (`  0%`, ACCENT badge), `build_orchestrator(None)` (`--`, tertiary badge), live-absent (`  --`). The units also change (% idle, signed score live). → S6 |
| **C** decision most prominent, but button-shaped | **CONFIRMED with an inversion**: prominence is correct; the form is a pressable primary button, the chips are outline buttons, and the only real buttons on screen (the tabs) look least like buttons. → S10 |
| **D** override and overridden winner at equal weight | **CONFIRMED AND UNDERSTATED** — the overridden winner keeps *more* weight than the enacted call: full ACCENT bar, white `+0.71`, `is_winner: true` on the wire, while STAY draws an empty grey bar. Only an 11 px red line links them. → S1 (the one P0) |
| **E** chart axes scaled to something other than their data | **CONFIRMED three ways**: tire y-axis 40–140 against a trace at 81.2 s (4 px of data ink in a ~150 px plot); the two adjacent charts disagree on x extent (12–24 vs 12.5–35) and neither marks the current lap; the tire x-axis prints its own fractional bound `12.5` as if it were a tick, bypassing `valueAxis`'s own locked-axis rule. → S5 |
| **F** card space allocated by grid position, not content | **CONFIRMED, measured**: SITUATION 45 %, PIT 57 %, RADIO 44 %, RAG 55 % dead strip below last ink; rows are `minmax()` bands fixed per grid row. The reasoning panel is the extreme case: 1.9 % ink, 77 % dead. → L1 |
| **G** contrast failures at rendered size | **MEASURED — five failures**: radio ALERT chip 2.54:1; STAY OUT badge 2.54:1 (fails even the 3.0 large-text bar); idle ACCENT badge 2.72:1; dimmed idle headline 2.38:1 *rendered* (sampled from s2 pixels); dimmed idle card title 3.34:1. Everything else passes — full table in S4. |

---

## LAYOUT findings (ranked)

### L1 · P1 — Vertical space is allocated by grid position; half of four cards and 98 % of the reasoning panel is empty
- **Files:** `src/pitwall/ui/src/styles/agents.css:90-99` (`.agents-right` `grid-template-rows:
  minmax(260px, 420px) minmax(140px, 260px) minmax(140px, 260px)`), `agents.css:83-89`
  (`.agents-left` `auto auto 1fr` — the reasoning tabs absorb everything).
- **What a reader sees:** four cards that are half hole — the eye must scan past ~100 px of
  panel-coloured nothing per card to find the next headline — and, bottom-left, the largest
  bordered surface on the screen (520×411) carrying five short lines of text.
- **Measured (s1, 1485×833):** SITUATION ink 2.9 %, dead strip 89 px (45 %); PIT ink 2.1 %,
  dead 112 px (57 %); RADIO 7.4 %/89 px (44 %); RAG 2.7 %/110 px (55 %); REASONING ink 1.9 %,
  dead 318 px below the last line (77 % of the panel). Meanwhile the two charts — the densest
  surfaces — sit at 319 px in a row that could go to 420.
- **Why it matters:** density is hierarchy on a glance surface. Space that belongs to nothing
  tells the reader nothing is there; here it outweighs the content four cards out of six.
- **Fix (tokens/CSS only):** rows 2 and 3 of `.agents-right` become `minmax(140px, auto)` so a
  three-line card sits at its content height, and row 1 keeps the slack
  (`minmax(260px, 1fr)`), giving the freed ~200 px to the two charts — which S5 needs anyway.
  Left column: `.reasoning` gets `max-height: fit-content` behaviour via
  `grid-template-rows: auto auto minmax(180px, max-content)` on `.agents-left` — the panel ends
  where its text ends and the leftover is honest window background, not bordered emptiness.

### L2 · P1 — "What changed since the last lap" has no first-class answer
- **Files:** `src/pitwall/agents_view/reasoning.py::_orchestrator_body` (memory block only when
  `plan_changed`, only inside the tab body), everything else replaces values in place.
- **What a reader sees:** lap 22 → 23 flips the badge from STAY OUT to PIT NOW and *nothing on
  the screen marks the flip* — the only trace is `--- why this call changed ---` in 11 px
  monospace inside a tab panel, below the fold of attention. Confidence, scenario fills and all
  six cards overwrite silently at 10 Hz.
- **Evidence:** the s1 orchestrator tab carries `lap 22: STAY_OUT (0.58)` — the previous call —
  as the only change record anywhere in five screenshots; no other element differs
  in *kind* between s1 (PIT NOW) and s3 (STAY OUT) beyond the swapped values themselves.
- **The brief's own test** ("can a reader tell in one second what changed") fails: they cannot
  tell that anything changed at all, let alone what.
- **Fix (fields already on the wire):** `latest.plan_changed` and `latest.memory_block` reach
  the builder today. Surface them in the orchestrator card: when `plan_changed`, a one-line
  chip under the badge — `was STAY_OUT (0.58)` in `--qt-fg-3` with a `WARNING` dot — built in
  `build_orchestrator` exactly like `guardrail` is (same mechanism, same renderer line
  `OrchestratorCard.tsx:46`). No new producer field.

### L3 · P2 — The most important number on the screen renders at the size of an axis tick
- **Files:** `agents.css:245-248` (`.orch-conf-label` 11 px), `agents.css:225-238` (badge 26 px).
- **What a reader sees:** "Confidence: 71%" at 11 px `--qt-fg-2` — the number that decides
  whether to trust the 26 px action next to it is the same visual weight as `±0.55s (CI)` and
  one pixel larger than the chart tick labels.
- **Evidence:** s1 vs s3 — 71 % green vs 44 % amber is the entire epistemic difference between
  the two shots, carried by a 1×240 px bar fill and an 11 px label; the action itself got
  20× the ink.
- **Fix:** promote the numeral out of the label: `71%` at ≥20 px `--qt-fg-1` right-aligned over
  the bar, label shrinks to `CONFIDENCE` in the existing 11 px caps style (`.scenarios-title`
  is already exactly this pattern). Colour stays `confidence_colour` — the tri-band already
  exists in `decision.py:58-64`.

### L4 · P2 — The idle state (s0) is what is left when data is absent, not a designed state
- **Files:** `AgentsWindow.tsx:44-133` (`IDLE_VIEW` — client-side defaults; the s0 state JSON is
  literally `null`).
- **What a reader sees:** the most saturated element on the screen is an ACCENT-purple primary
  button announcing `--`; four scenario rows claim a measured `0%`; the TIRE panel draws real
  axes over a 0→1 fake lap range; the PACE panel draws axis names with no scale at all. It
  reads as a broken live view, not as "waiting".
- **hallmark language:** honest-copy gate — `0%` is an invented metric (the system has measured
  nothing); the purple `--` badge is fabricated status.
- **Fix (one file, client-side):** `IDLE_VIEW` adopts the producer's own empty forms —
  `score: "  --"`, badge `TEXT_TERTIARY` like `build_orchestrator(None)` already emits
  (`decision.py:93-106`), and the two charts suppressed behind the existing
  `status_bar.text` ("Waiting for arcade stream…") repeated in-panel at `--qt-fg-3` instead of
  empty axes. This also collapses claim B's three variants to one (see S6).

### L5 · P3 — Left/right weighting: 36 % of the width is pinned to the column with the least ink
- **Files:** `agents.css:64-74` (`grid-template-columns: 540px 1fr` — deliberate, ported).
- **What a reader sees:** at 1485 px the left column (540 px fixed) carries a badge, four bars
  and a mostly-empty log (measured ink 33 % / 30 % / 1.9 %); six agents share the remaining
  945 px in equal boxes regardless of stake — SITUATION (threat, overtake, SC risk: race
  context) gets the same box as RAG (a regulation footnote).
- **Fix (cheap version, no re-architecture):** keep the split; fix the emptiness via L1. The
  full reweighting (e.g. SITUATION promoted into the left column under the decision it feeds)
  is a sprint-9 shape question — flagged, not priced here.

### L6 · P3 — Cards can hide content with no affordance that more exists
- **Files:** `qt-base.css:54-77` — scrollbars hidden globally; its own comment concedes:
  "nothing now hints that a card has more below. That is a design question… it belongs to the
  elevate pass." This is the elevate pass.
- **What a reader sees (when it bites):** a card whose last visible line is not its last line,
  with zero signal — the previously measured overflow was 14 px on four of six cards.
- **Evidence:** in the five given states nothing overflows at 1485×833 (all last-ink rows sit
  inside their cards) — the *mechanism* is armed, not currently firing.
- **Fix:** a bottom fade on scrollable card bodies: `.agent-card-body` gets a
  `mask-image: linear-gradient(...)` applied via a `.has-more` class toggled from a
  `scrollHeight > clientHeight` check in `AgentCard` (3 lines), colours from `--qt-panel`.

---

## SURGICAL findings (ranked)

### S1 · P0 — Guardrail state: the vetoed plan keeps the winner's regalia; the two panels disagree on the call
- **Files:** `src/pitwall/agents_view/decision.py:136-182` (`build_scenarios` knows nothing of
  the guardrail; `is_winner` stays on the MC winner), `decision.py:132` (guardrail is one text
  line), `ScenarioBars.tsx` (renders whatever `is_winner` says), s3 JSON lines 13-14 vs 39-48.
- **What a reader sees (s3):** the badge says **STAY OUT** (green); one card below, the
  scenario panel crowns **PIT** — full ACCENT bar, brightest score `+0.71` — while STAY, the
  action actually taken, draws an *empty grey bar*. The only connective tissue is an 11 px red
  line. A strategist reading the scenario panel (the "why" panel) reads the opposite of the
  call. It is also ambiguous which plan the amber "Confidence: 44%" describes.
- **The brief's P0 bar is met:** a strategist misreads the race — specifically, which strategy
  the system is executing.
- **Fix (fields already on the wire):** `AgentsViewBuilder.build` already holds both `latest`
  (with `action` and `guardrail_reason`) and the scores — pass `action` +
  `bool(guardrail_reason)` into `build_scenarios`. When the enacted action ≠ MC winner: winner
  row keeps its fill but demotes to `TEXT_TERTIARY` with a `VETOED` suffix on the score (or a
  struck bar via a dashed `--qt-border` overlay — the dashed stroke is taken for
  broadcast-tier only in the DATA window, so use strike/hollow, not dash); the enacted row's
  label takes the badge's own `action_colour`. Copy fix in the same pass: guardrail text
  `"minimum stint length not met - 4 laps"` — the hyphen reads as a minus sign next to a
  number; use `·` (`decision.py:132` prefix already exists, the clause is producer free text).

### S2 · P1 — A scored runner-up and an unscored scenario are the same pixels (claim A)
- **Files:** `decision.py:151-162` (min-max; worst scored → `fill 0.0`; absent → `fill 0.0`,
  same `TEXT_SECONDARY` colours), docstring at `decision.py:141-143` promising "not the same".
- **What a reader sees:** STAY (`+0.29`, the runner-up, and in s3 the *enacted call*) and UCUT
  (never scored) render identical empty tracks; the executed pixel diff over the two bar strips
  found **0 differing pixels** — only the label glyphs and the 11 px score text differ.
- **The deeper defect:** with n=2 scored candidates min-max collapses margin entirely — the bar
  encodes *rank* while sitting beside a numeric *score*. +0.70 vs +0.69 would render exactly
  like +0.70 vs absent.
- **Fix:** in `build_scenarios`, floor a present candidate's fill (`fill = max(fill, 0.06)` —
  one line at `decision.py:162`) so "scored" always draws ink, and give absent rows a visibly
  different track: `bar_colour` empty + the row's track border dropped to `--qt-border` at
  reduced opacity, or track hidden entirely with `--` centred. The docstring then becomes true.

### S3 · P1 — Red means six different things, and two of them sit one line apart
- **Files:** `decision.py:34-47` (`PUSH`/`AGGRESSIVE` → DANGER), `decision.py:58-64`
  (confidence <0.33 → DANGER), `src/arcade/strategy.py:669-681` (`classify_action` — PIT NOW →
  DANGER), `panels.py:55-60` (ALERT glyph → DANGER), `panels.py:45-49` (Disconnected → DANGER),
  `agents.css:274-279` (guardrail → literal `#ef4444`).
- **What a reader sees:** in s1, four red elements — action badge, two chips, radio dot — of
  which exactly one is an alarm. In s3, `Risk: AGGRESSIVE` (a chosen setting, red) sits
  directly above `⚠ Guardrail: minimum stint length not met` (a constraint violation, same
  red). The one colour carries: imperative action, posture fact, low confidence, radio alert,
  dead link, rule violation.
- **Why it matters:** the entire point of an alarm colour is pre-attentive triage; six
  semantics deny the reader exactly that under time pressure.
- **Fix (existing tokens only):** reserve `DANGER` for alarm-class facts (ALERT glyph,
  guardrail, Disconnected). Chips render posture as *facts*: text `--qt-fg-2`, the word carries
  the meaning (they are settings, not warnings) — `_PACE_COLOURS`/`_RISK_COLOURS` in
  `decision.py:34-47` collapse to `TEXT_SECONDARY`. The action badge encodes *identity*, not
  severity: PIT NOW → `ACCENT` or `INFO` (`#3b82f6` is currently unused outside the pace
  chart), STAY OUT stays `SUCCESS`-family. One test updates: the palette counts in
  `test_pitwall_tokens.py` (settled rule 5 — counts move, no new hex needed).

### S4 · P1 — Contrast: the most alarming and the most dimmed text on the screen both fail AA (claim G, measured)
- **Files:** `src/arcade/palette.py` `flag_chip_html` (white 10 px/700 on `#9ca3af`),
  `classify_action` + `OrchestratorCard.tsx:17` (white 26 px/800 on badge fills),
  `agents.css:123-125` (`.is-idle { opacity: 0.45 }`).
- **Measured failures** (need 4.5:1, badges 3.0:1 large-text):
  | Element | Ratio | |
  |---|---|---|
  | Radio ALERT chip `UNDERCUT THREAT FROM RUS`, white on `#9ca3af`, 10 px/700 | **2.54:1** | FAIL — the alarm is the least legible text on screen |
  | STAY OUT badge, white on SUCCESS `#10b981`, 26 px/800 | **2.54:1** | FAIL (even at 3.0) — the primary decision, in the guardrail shot |
  | Idle badge `--`, white on ACCENT `#a78bfa` | **2.72:1** | FAIL |
  | Dimmed idle headline (trigger hint), rendered pixels sampled from s2 | **2.38:1** | FAIL — the text explaining *why a card is dark* |
  | Dimmed idle card title | **3.34:1** | FAIL |
  | PIT NOW badge, white on DANGER | 3.76:1 | passes 3.0 (large) — borderline |
  - Everything else passes: fg2/fg3 on panel 11.86/6.88, guardrail red 4.64 (barely), WARNING
    8.14, compound pill 11.16, all four reasoning highlight colours 6.6–11.4.
- **Fixes, verified by computation:** badge text `--qt-bg` (#121127) instead of white —
  SUCCESS 7.29:1, ACCENT 6.80:1, DANGER 4.92:1, all pass (the compound pill already uses
  exactly this dark-on-saturated pattern at 11.16:1 — copy it). Alert chip: same treatment,
  `flag_chip_html` background `WARNING` with `#121127` text = 8.61:1, or DANGER + dark text
  4.92:1. Idle cards: **the opacity dim cannot be tuned into compliance** (0.75 → 4.39:1
  still fails; 0.85 barely passes and no longer reads as dimmed) — drop `.is-idle` opacity for
  text and let the colour system carry it: the producer already sends idle text in
  `TEXT_TERTIARY` (6.88:1); scope the opacity to the glyph and border only.

### S5 · P1 — The tire chart's y-axis makes its own subject invisible; the two charts disagree on x (claim E)
- **Files:** `useEChart.ts:43` (shared `yAxis` autorange, no bounds), `charts.py:220-244`
  (`_x_range` — the producer clamps X against exactly this failure mode, "a bad p90 cannot
  flatten the series to a hairline", and never guards Y), `TireChart.tsx:85` (locked x via
  spread), `PaceChart` (autoranged x).
- **What a reader sees (s1):** TIRE y runs 40–140 s against a trace at 81.2 s — **the data
  occupies 4 px of a ~150 px plot** (measured). The degradation trend the chart exists to show,
  0.031 s/lap, is sub-pixel. Any real in-lap (~+20 s, inside the 30–200 s sanity window) or a
  near-constant early stint reproduces this in production; the ECharts zero-span expansion the
  fixture triggers is just the worst case.
- **Also:** the two adjacent lap-axis charts disagree — PACE autoranges 12–24, TIRE locks
  12.5–35 — so lap 23 ("now") sits at ~95 % of one plot and ~47 % of the other, and *neither
  marks the current lap*. And the TIRE x-axis prints its own bounds (`12.5`, `35`) as if they
  were ticks: `TireChart.tsx:85` spreads `min`/`max` onto an axis `valueAxis` built as
  *autoranged*, bypassing the helper's own locked-axis label suppression
  (`lib/chart.ts::valueAxis` computes `locked` from its spec only). A half-lap tick on an
  integer quantity is an axis implying something untrue.
- **Fix (producer + one-line client):** `build_tire_series` emits `y_range` exactly as it
  already emits `x_range` — e.g. median of plotted green-flag `lap_time_s` ±2.5 s, clamped to
  the observed extent — and `TireChart` passes it as `min`/`max` **through
  `valueAxis({min, max})`** so the bound-label suppression applies. Share `x_range` with
  `PaceChart` (the builder computes both series in the same call, `builder.py:67-74`) and add
  a current-lap `markLine` from `latest.lap_number` (already on the wire) in `CURSOR_LINE`
  (`#9ca3af`) — the token that exists for precisely this in the DATA window.

### S6 · P2 — Three different renderings of "no data yet", one of which is a fabricated `0%` (claim B)
- **Files:** `AgentsWindow.tsx:95` (`score: "  0%"`, badge `#a78bfa`, plan `—` em-dashes) vs
  `decision.py:93-106` (`build_orchestrator(None)`: badge `TEXT_TERTIARY`, plan `--`) vs
  `decision.py:175` (live absent: `"  --"`).
- **What a reader sees:** before the first tick the window asserts all four scenarios score
  `0%` — a measurement that never happened, in a different *unit* (percent) than the live view
  ever uses (signed scores). Reconnect mid-session and the same "nothing" wears three costumes.
- **Fix:** `IDLE_VIEW` mirrors the producer's own empty forms (`"  --"`, tertiary badge,
  `--` plan) — one file, and L4's designed idle state finishes the job.

### S7 · P2 — Developer language reaches the strategist: "stub", "no radio/rcm pipeline output", `SC >30%`
- **Files:** `agent_formatters.py:102/154/216` (`"no prediction — stub"`),
  `agent_formatters.py:353` (`"no radio/rcm pipeline output"`), `:429` and `:563` (trigger
  hints — `"SC >30%, or FIA warning/penalty"` is spec-sheet syntax), `panels.py:156`
  (`"pipeline: {error}"` in the status bar), and the reasoning tabs, whose *only* per-agent
  content is a snake_case debugger dump (`sc_prob_3lap  =   8.0%`, `sc_reactive = False` —
  `reasoning_lines.py`).
- **What a reader sees (s4):** six cards all telling the engineer about our software
  ("stub", "pipeline") instead of about the race.
- **Fix (copy pass in the formatters, no wire change):** `"no reading this lap"`;
  `"radio silent"`; trigger hints in operational voice: `"wakes on tyre cliff, compound
  change, or a problem radio"` / `"wakes on compound change, SC risk above 30%, or an FIA
  warning"`. The reasoning dumps may stay — they are the debug drill-down — but they are
  currently the *only* "why" surface, which is a sprint-9 content question, not a copy fix.

### S8 · P2 — The PIT card renders "agent is active" as a warning, identically to a real warning beside it
- **Files:** `agent_formatters.py:442-448` — active `format_pit` returns headline colour
  `WARNING` and `STATUS_WATCH` *unconditionally*.
- **What a reader sees (s1):** PIT shows an amber ◐ and amber headline because the agent has
  output; TIRE shows the same amber ◐ for a genuine MONITOR warning. Two different facts —
  "conditional agent fired" and "something needs watching" — rendered identically, side by
  side. (The brief's exact category.)
- **Also:** `"pit 22.40s → HARD"` — an unlabeled 22.40 s in a domain that famously carries two
  nearby quantities (total pit delta vs `pit_duration_s`, the physical stop — this repo's own
  memory flags the pair). One word disambiguates.
- **Fix:** active-with-nothing-wrong → `TEXT_PRIMARY` headline, `STATUS_OK`; keep `WATCH` for
  actual pressure (`sc_reactive`, cliff window). Headline `"stop 22.40s → HARD"`.

### S9 · P2 — The radio ticker picks recency over relevance: the driver's own PROBLEM radio is invisible
- **Files:** `agent_formatters.py:397-406` (`radio_events[-1]`), s1 JSON: card shows
  `RUS INFORMATION: "What is Norris doing?…"`; NOR's `PROBLEM: "Rear grip is going away…"`
  exists **only in the hover tooltip**.
- **What a reader sees:** the one radio line on the card is a rival's commentary; their own
  driver's grip complaint — the input that likely drove the PIT call — requires a hover.
- **Fix (fields on the wire):** rank the visible radio by severity then recency — own-driver
  `PROBLEM`/`WARNING` intents first (`analysis.intent` is already read at `:275-283`), fall
  back to `[-1]`. Two lines in `format_radio`.

### S10 · P3 — The decision wears a button costume; the real buttons don't (claim C)
- **Files:** `agents.css:225-238` (`.orch-badge`: 200×70 min, radius 10, filled, centred
  26 px/800 — a primary CTA), `agents.css:262-268` (chips: outline pills — secondary buttons),
  `agents.css:348-364` (the actual buttons, the tabs, are flat text with an underline).
- **What a reader sees:** the two highest-affordance shapes on the screen do nothing; the six
  clickable things look least clickable. In a desktop webview there is no hover to correct the
  impression.
- **Fix:** the badge becomes a banner: full card width, radius 2–4, action colour as a left
  border + low-alpha tint of the same token (Qt's own `elevated` layering pattern), text keeps
  its 26 px. Chips → plain 12 px text with a coloured status dot (the glyph vocabulary the
  cards already use).

### S11 · P3 — Colour-alone state encoding and missing semantics (fixing-accessibility)
- **Files:** `panels.py:55-60` — OK and ALERT share the same glyph `●`, distinguished only by
  green vs red (WCAG 1.4.1; the two states most important to tell apart are the pair
  deuteranopia collapses); `ReasoningTabs.tsx` — `role="tab"` without arrow-key navigation,
  `aria-controls`, or `aria-labelledby` on the panel; both charts are bare `<canvas>` with no
  `role="img"`/`aria-label`; the confidence bar div carries no `role="progressbar"`.
- **Fix:** ALERT gets its own glyph (`▲`), keeping colour; add the three ARIA wires and a
  keyboard handler (the WAI-ARIA tabs pattern is ~10 lines); charts get
  `role="img"` + a one-line label from fields already in view (`aria-label={card.headline}`).

### S12 · P3 — Four pill dialects and inline hex on the wire (hallmark: locked tokens)
- **Files:** header chip (`agents.css:45-51`, radius 10, elevated bg) · orch chip (radius 10,
  outline) · compound pill (`palette.py::compound_pill_html` — radius 7, inline
  `background-color: #e6e6e6`, its own font stack, 10 px/800) · flag chip (radius 6, inline
  `#9ca3af`). The producer ships styling as inline HTML attributes inside JSON strings — hex
  that no stylesheet, token file, or the palette-count test can see as CSS.
- **What a reader sees:** four subtly different pill shapes for four kinds of tag; no shared
  radius, size, or weight.
- **Fix:** one `.pill` class family in `agents.css` consuming `--qt-*`; the producer's span
  builders emit `class="pill pill-compound-medium"` etc. instead of inline styles (the span
  transport itself can stay — sprint 8 already owns "unpicking the rich-text debt" per
  `AgentCard.tsx:12-13`).

### S13 · P2 — The degraded state contradicts itself: "no prediction" above a chart drawing a prediction
- **Files:** s4 JSON — `cards.pace.headline: "no prediction — stub"` while
  `charts.pace.pred` ends at `[23, 81.0]`; the charts render accumulated history
  (`builder.py:125-134`) independent of `per_agent`.
- **What a reader sees (s4):** the PACE card says the agent produced nothing this lap while,
  30 px below in the same card, a dashed predicted line runs confidently through the current
  lap. Nothing marks the chart as history rather than a live claim.
- **Fix:** when the card's status is `IDLE`, the chart dims its predicted series
  (`lineStyle.opacity`) — the card view and chart series are already siblings in the same
  `AgentCard` render, so the flag is one prop; or the producer stamps the last-updated lap on
  the series and the card shows `"history to L23"` in `--qt-fg-3`.

### S14 · P3 — RAG is the only agent with no reasoning tab; its evidence is hover-only
- **Files:** `reasoning.py::TABS` (six tabs = orchestrator + five agents; `rag` absent);
  the retrieved chunks and the question exist only in `rag_tooltip_html`.
- **What a reader sees:** the regulation grounding for a compliance-relevant call is reachable
  only by hovering a card, on a surface designed to be read hands-off.
- **Fix:** a seventh tab from the fields already shipped in the tooltip (`question`, `chunks`),
  or accept and document the asymmetry. Small; priced as a tab-list entry + one
  `LINE_BUILDERS`-style formatter.

---

## What I tried to break and could not

1. **The scenario mathematics.** All-negative score normalisation, the `fill_pct` rounding
   contract, and the None-score drop (`_normalize_scores` — a scored-None candidate is dropped,
   not zeroed) are correct and pinned by `tests/surfaces/test_pitwall_agents_view.py:437-497`.
   The *data* is right everywhere I probed; S1/S2 are purely rendering collapses.
2. **The guardrail line's own contrast.** Exact `#ef4444` sampled from s3 pixels: 4.64:1 on the
   panel — passes AA at 11 px. (Barely; any darker panel tint breaks it. Not a finding today.)
3. **The compound pill.** Dark-on-`#e6c832` at 11.16:1 — the best-contrast element on the
   screen and the pattern S4's badge fix copies.
4. **The chart plumbing under the settled rules.** `animation: false` at 10 Hz, `notMerge`
   reasoning, the callback-ref host survival, and the `__pitwallChart` test handle — sound;
   no motion findings exist and none were invented (rules 3–4 respected).
5. **The tire x-clamp deviation.** `_x_range`'s "+3 laps past max(observed, cliff hi)" checks
   out on the pixels: s1 renders to 35 = cliff hi 32 + 3, against Qt's unreadable +100. The
   deviation is an improvement and correctly flagged in its docstring.
6. **Truncation.** No rendered truncation defect in any of the five states — every radio, RAG
   and headline string fits under the 70-char caps (S9 is a *selection* problem, not
   truncation). The `_truncate` helper's behaviour at the boundary is fine.
7. **The tooltip mechanics.** Portal + `position: fixed`, beside-the-card placement, content
   sizing, and the restored thin scrollbar close all three historical failure modes; the clamp
   maths in `AgentCard.tsx:60-72` is correct for both edges.
8. **The header.** Session/driver/lap/playback/connection render correctly from the wire in
   all five states; the connection tri-state (`panels.py:45-49`) is consistent, and the idle
   "Connecting…" amber choice is documented and right.
9. **Geometry at the real 1485×833.** No horizontal scroll, no clipped cards, every row inside
   its `minmax` band, the 540 px pin behaves as documented. The dead space of L1 is a design
   allocation problem, not an overflow bug.
10. **Reasoning highlighter colours.** All four (`#f472b6`, `#d946ef`, `#facc15`, `#22d3ee`)
    pass AA on the panel at 6.6–11.4:1, and the per-line rule application (the Qt paragraph
    semantics) held up against the multi-line memory block in s1/s3.

## Producer-field pricing note

Every fix above is executable against fields already on the wire. Two proposals touch the
**producer view** (not the tick): S5's `y_range` (computed from `lap_time_s` history the
builder already holds) and S1's enacted-action flag (computed from `latest.action`, already in
`build`'s scope). Zero proposals need a new field from the *arcade producer* or the agents
themselves. The only new-sprint-priced item is S14's seventh tab if the team wants chunk
rendering richer than the tooltip's existing strings.

