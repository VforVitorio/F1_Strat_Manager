# GATE: PITWALL v2 architecture — fit-with-repo audit (Lens B)

**Role:** adversarial gate. Lens = does the design fit the repo as it actually is, and what breaks
when it lands. Data-plane correctness is another gate's lens; not duplicated here except where it
touches repo fit.

**Target:** `documents/research/PITWALL_V2_ARCHITECTURE.md` (340 lines, read in full before starting).

**Status:** COMPLETE.

## Checklist

- [x] A. `src/arcade/dashboard/` deletion — full reference enumeration
- [x] B. PySide6/pyqtgraph removal / pywebview addition — dependency diff
- [x] C. Zero-HTTP-calls claim, both directions
- [x] D. AGENTS window 1:1 port — field inventory + purity check on agent_formatters.py
- [x] E. Design-token copy count — A16 verification + current drift risk
- [x] F. webapp features/strategy reuse claim — honest inventory + name collisions
- [x] G. Issue reconciliation table — #281-#287, #199
- [x] H. Packaging reality — wheel/build backend/CI
- [x] I. Cross-check against CLAUDE.md §11 and CONTRIBUTING.md

---

## Findings log

### A. "Deleting `src/arcade/dashboard/` entirely is safe" — REFUTED as stated

`src/arcade/dashboard/` is 13 `.py` modules (`__init__.py` empty), **3,409 lines**, confirmed by
`wc -l`. The design's own line count ("13 modules") is correct. What is NOT accounted for is the
full reference graph outside the package.

**Complete reference table** (grep for `arcade.dashboard`, `arcade/dashboard`, `PySide6`,
`pyqtgraph`, `QApplication`, `dashboard`, `Qt` across `src/`, `scripts/`, `tests/`, `.github/`,
`docs/`, `documents/`, `README.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `CONTRIBUTING.md`,
`INSTALL.md`, and the `src/telemetry` submodule):

| File | Kind of reference | Breaks how, if untouched |
|---|---|---|
| `src/arcade/app.py:332` | `subprocess.Popen([sys.executable, "-m", "src.arcade.dashboard"], ...)` — the actual spawn | **Loud.** `f1-arcade --strategy` raises `ModuleNotFoundError` inside the child process at runtime the moment the package is deleted and this line is not repointed at `python -m src.pitwall`. Confirmed: `_spawn_dashboard` (line 321) catches only `(OSError, ValueError)` around `Popen` itself — a `ModuleNotFoundError` inside the *spawned* interpreter is NOT one of those, so it surfaces as a separate console window crashing with a traceback while arcade keeps running silently "without it". This is a **silent-from-the-replay's-perspective, loud-from-the-user's-perspective** failure — worth naming because the design's "the spawn mechanism does not change" (section 2) undersells that the target string on that exact line is the one thing that DOES have to change, atomically with the deletion. |
| `src/arcade/app.py:268-278,321-328` (docstrings) | Prose says "PySide6 dashboard subprocess" | Silent (stale prose, no crash) |
| `src/arcade/stream.py:1-14` (module docstring) | Explains why `stream.py` is stdlib-only "so we can launch the dashboard as a subprocess without pulling Qt into the replay window" | Silent. The reasoning becomes half-true (still true that arcade must stay light; the specific "Qt" noun is wrong once pywebview replaces Qt) |
| `src/arcade/README.md:1-7,25,42` | Package overview names "PySide6 dashboard subprocess", links to `docs/pages/arcade-dashboard.md`, layout tree shows `dashboard/` | Silent. Pre-existing bug found in passing: the prose link at line 25 is `docs/pages/arcade-dashboard.md` (correct) but the comment on the tree at line 42 says `see docs/arcade/dashboard.md` (wrong path, predates this design) |
| `tests/surfaces/test_arcade_dashboard_imports.py` (109 lines) | Import-smoke test for all 13 dashboard modules | **Loud in CI**: every import target vanishes, the whole file goes red. Design's §7 test list does not mention deleting or replacing this file — it lists 5 NEW tests but never says "and delete this one." |
| `tests/agents/test_overtake_domain.py:231` | `from src.arcade.dashboard.agent_formatters import format_situation` — a **domain** test (not a dashboard test) imports dashboard formatting code to assert an absent-value renders as "—" not "0%" | **Loud in CI**, and easy to miss because the file lives under `tests/agents/`, not `tests/surfaces/`. This is exactly the kind of hidden cross-package dependency the design's file-layout section does not surface: `agent_formatters.py`'s `format_situation` is depended on from OUTSIDE the dashboard package. Whoever ports `agent_formatters.py` to `AgentCard.tsx`/`format.ts` must either keep a Python-side `format_situation` alive somewhere reachable, or migrate this specific test's assertion into the TS port and delete the Python test. Neither option is in the design. |
| `tests/infra/test_dep_imports.py:337-338` | Comment: `# Arcade (heavy import — PySide6 / pyqtgraph already covered by tests/test_arcade_dashboard_imports.py, do not re-import here)` | Silent (stale comment pointing at a deleted file) |
| `.github/workflows/release-please.yml:62-75` | Wheel smoke-test checks `src/arcade/main.py` exists post-install; does NOT check `dashboard/` | **No break** — this job is already agnostic to the dashboard package, confirming Claim H's packaging path is separate. Included here to show the CI surface was checked, not skipped. |
| `README.md:26-32,67,68,102-106,121` | Hero demo caption + GIF/MP4 (`docs/assets/demo/arcade-demo.mp4`/`.gif`) call it "2D race replay, strategy dashboard and live telemetry"; feature table row 67 says "Three-window 2D race replay + PySide6 strategy dashboard + live telemetry grid"; layout tree line 121 says "PySide6 strategy dashboard" | Silent, but this is the **published, public-facing README** — the design's own risk list (§8.7) names only `ROADMAP.md` and `docs/pages/roadmap.md`. The README's stale claim is more visible than either, and the demo video itself needs re-recording, which the design never mentions as a cost. |
| `ARCHITECTURE.md:9-10,39-50` | "Arcade three-window topology" section, names `MainWindow`/`TelemetryWindow`/shared `QApplication`, links to `docs/pages/arcade-dashboard.md` | Silent, live architecture doc |
| `CONTRIBUTING.md:43` | Entry-points table row: `` f1-arcade --strategy `` → "2D replay + PySide6 dashboard + telemetry" | Silent, contributor-facing setup doc — not in Claim A's original grep sweep, surfaced instead while cross-checking Claim I; folded in here since it belongs in the same reference list. |
| `ROADMAP.md:647` | "Next core releases" table, v2.6.0 row: "The strategy and telemetry surfaces move to a web-native view" (same stale sentence the design's own §8.7 flags for `docs/pages/roadmap.md`, but the **root** `ROADMAP.md` carries it too and is not named) | Silent |
| `docs/pages/roadmap.md:468` | Same sentence design's §8.7 already flags — **line number correction**: design cites `roadmap.md:464`; the actual sentence is at **line 468** (464 is the `<span class="rl-version">v2.6.0</span>` line, 4 lines earlier). Minor precision miss in the design's own citation, but it means anyone jumping to 464 lands on the wrong line. | Silent |
| `docs/pages/arcade-dashboard.md` (154 lines) | Entire page is a "Developer-level reference for the PySide6 dashboard" — process topology diagram, package layout per-module | Silent. This is not a stale line, it is a **whole page that becomes actively wrong** the day the package is deleted, and it is not in the design's file list at all. |
| `docs/pages/arcade-quick-start.md` (15 hits) | End-user quick start, describes the 3-window launch | Silent, whole-page rewrite |
| `docs/pages/arcade-strategy-pipeline.md` (3 hits) | Shared engine doc, references the dashboard as a consumer | Silent, partial rewrite |
| `docs/pages/multi-agent.md:75-95` | **Mermaid diagram** with a literal `subgraph qt["Dashboard subprocess (single QApplication)"]` node plus prose "Two windows inside one event loop is cheaper... avoids duplicated imports of PySide6 + pyqtgraph" | Silent, but a **rendered diagram** on the public docs site, not just prose — the design's risk list does not mention any Mermaid diagram needing a redraw |
| `docs/pages/{tags,streamlit,changelog,meet-the-author,architecture,getting-started,driver-colors,home}.md` | 1-5 hits each, mostly historical/changelog mentions or cross-links | Mostly silent/low-cost, but adds up: **69 total "dashboard/PySide6/pyqtgraph/Qt" occurrences across 13 files under `docs/pages/` alone**, only 2 of which (`arcade-dashboard.md`, `arcade-quick-start.md`) account for 35 of them. The other 34 are scattered thin references the design does not enumerate. |
| `documents/dev_docs/diagrams/README.md:17,21-22` + `arcade_3window_architecture.drawio`, `subprocess_launch_sequence.drawio`, `tcp_broadcast_dataflow.drawio` | Three `.drawio` diagrams explicitly marked **"current"** as of 2026-07-26, modelling the exact two-Qt-window-in-one-subprocess topology this design replaces | Silent, but these are diagrams the project's own audit process (see the README's self-correction paragraph about over/under-reporting) is set up to catch eventually — **not automatically**, and not named in this design's risk section. |
| `INSTALL.md:21,63,73-77` | "Arcade, 3-window race replay + live dashboard + telemetry" section names PySide6 explicitly twice and states "Docker is NOT recommended for Arcade: pyglet + Qt need a host OpenGL context and a native display" | Silent, but this is exactly where the design's own risk #2 (pywebview needs `webkit2gtk` on Linux) has to land, and today nothing there anticipates it |
| `src/simulation/README.md` | 1 hit (context: describes the pipeline that Arcade, CLI and backend all route through; names Arcade's dashboard as a consumer) | Silent |
| `src/telemetry/` submodule | **Zero real hits.** The initial broad grep for `dashboard` matched 78 files, but every one is the webapp's own unrelated `features/dashboard/` (its post-race telemetry tab) — a name collision, not a reference to `src/arcade/dashboard/`. A second, PySide6/pyqtgraph/QApplication-scoped grep over the submodule returned nothing. | **Confirms** the design's implicit assumption that the submodule is untouched by this deletion — the one part of Claim A that holds cleanly. |

**Severity: P1.** Nothing here is unfixable, and the design's core claim ("safe to delete") is
defensible IF the full list above is executed as one atomic change. But the design's own file
layout (§4) and test plan (§7) name only a fraction of it — the spawn string, the two test files,
the README/ARCHITECTURE/ROADMAP/CONTRIBUTING prose, and above all the **hidden cross-package
import** in `tests/agents/test_overtake_domain.py` are not mentioned anywhere in the source
document. A gate re-run after implementation should specifically re-check that last one, because it
is the one a "grep for dashboard in docs" sweep would not catch (it is a domain test, correctly
located under `tests/agents/`, that happens to reach into UI-formatting code).

---

### B. PySide6/pyqtgraph leave, pywebview enters — mostly confirmed, one framing correction

`pyproject.toml:86-89`:
```
"arcade>=3.0.0",
"scipy>=1.10.0",
"pyside6>=6.5.0",
"pyqtgraph>=0.13.0",
```

**Correction to the audit's own framing**: there is **no `arcade` extra**. These four lines sit
inside the single, unconditional `[project.dependencies]` list (section header comment "ARCADE /
2D REPLAY" at line 84, but it is not an `[project.optional-dependencies]` entry). `uv sync` installs
PySide6 and pyqtgraph for **every** consumer of this repo today — a CLI-only user, a webapp-only
user, a notebook contributor — whether or not they ever run `f1-arcade`. There is only one optional
group in the whole file: `dev` (pytest/ruff/mypy/jupyter, lines 128-139).

Consequence for the design: "leaves pyproject.toml" is accurate (two lines delete), but framing it
as leaving "the arcade extra" is not — it leaves the **core** dependency set, shrinking install size
for literally everyone. Symmetrically, `pywebview` entering will also land in core `dependencies`,
not a new extra, unless the design additionally proposes introducing extras (which section 4/5
never does). That is a reasonable thing to flag back to the design author, not a defect in the
architecture itself.

**Independent verification — no other importer of PySide6/pyqtgraph exists.** Grepped `PySide6`,
`pyqtgraph`, `QApplication`, `QtWidgets`, `QtCore`, `QtGui` across `src/`, `scripts/`, `tests/`, and
the `src/telemetry` submodule. Hits: the 13 `src/arcade/dashboard/*.py` files (real imports),
`src/arcade/app.py` and `src/arcade/stream.py` (docstrings only, no import), `src/simulation/README.md`
and `src/arcade/README.md` (prose only), `scripts/run_webapp.py` (one docstring line, no import),
`tests/surfaces/test_arcade_dashboard_imports.py` and `tests/infra/test_dep_imports.py` (test
code/comment, no product import). **Zero product-code imports of PySide6/pyqtgraph outside
`src/arcade/dashboard/` itself.** The dependency diff is real and clean: delete 2 lines, no orphaned
imports elsewhere break.

**Severity: P3** (framing note only — the technical claim holds).

---

### C. "Arcade makes zero HTTP calls" — CONFIRMED independently, both directions

Forward: grepped `import requests`, `import httpx`, `BACKEND_URL`, `requests\.`, `httpx\.` across
`src/arcade/`. Only hit: `src/arcade/config.py:169-170` declaring `BACKEND_URL` and
`STRATEGY_ENDPOINT` as `Final` constants. A second, repo-wide grep for those two names plus the
three `SSE_*` siblings (`SSE_RECONNECT_DELAY_S`, `SSE_MAX_CONSECUTIVE_FAILURES`,
`SSE_BACKOFF_AFTER_FAILURES_S`) found **no other reference anywhere in `src/`** — not read, not
imported, not used in a conditional. These five constants are dead exactly as the design states
(citing P3 finding A15). Confirmed independently.

Reverse direction — does anything assume the Qt dashboard exists or that `f1-arcade` spawns it,
beyond what Claim A already enumerates: `scripts/f1_cli.py` (the interactive CLI menu) has **zero**
mentions of `arcade`, `dashboard`, or `PySide6` — it shells out to `f1-arcade` as an opaque
subcommand and does not itself assume Qt exists, so the CLI surface is clean. The assumption lives
entirely in docs/tests already listed under Claim A, not in any other runtime code path.

**Severity: informational** — claim holds in both directions, no new defect.

---

### D. AGENTS window 1:1 port — field inventory (deliverable) + purity check

Read `window.py` (346 lines) in full, plus `orchestrator_card.py` (257), `scenario_bars.py` (157),
`reasoning_tabs.py` (293), `agent_card.py` (174), `pace_chart.py` (142), `tire_chart.py` (first 150
of 410), `theme.py` (224), `agent_formatters.py` (582, full read).

**Line-range check**: `window.py:141-207` is `MainWindow.__init__`, and it does cover exactly what
the design says — HeaderBar + `QSplitter` (left: OrchestratorCard, ScenarioBars, ReasoningTabs;
right: 3×2 AgentCard grid) + status bar wiring. **But that range is construction/layout only.** The
actual data-fan-out logic (`_on_data`, lines 215-231) and the two rolling-history reducers
(`_seed_history_from_tail`, `_ingest_latest_history`, `_trim_history`, lines 239-293) sit OUTSIDE
141-207 and are not layout at all — they are the stateful accumulation the design's own §3.4
depends on ("Which panels actually accumulate... the AGENTS window's PaceChart and TireChart").
A port that treats 141-207 as "the checklist" and stops there will silently drop this reducer.

**Field inventory** (the deliverable):

| Widget | Fields rendered | Source in broadcast dict |
|---|---|---|
| **HeaderBar** | session (`gp · year`), driver, lap counter (`L n/total`), connection chip (Disconnected/Connecting.../Connected, 3-way colour), playback chip (`{speed}× · PLAYING\|PAUSED`) | `arcade.{gp_name,year,driver_main,lap,total_laps}`, `strategy.start.{gp,year,driver}`, `playback.{speed,paused}` |
| **OrchestratorCard** | action badge (label + colour via `classify_action`, imported from `src/arcade/strategy.py` — NOT in `agent_formatters.py`), confidence % + gradient bar (3-tier traffic light: ≥0.66 green / ≥0.33 amber / else red), pace_mode chip (4-entry colour map), risk_posture chip (5-entry colour map), plan line (`Pit: L{n} · Next: {pill} · UCUT: {target}`, with a STAY_OUT-specific "stint continues" branch and a generic "Pit plan pending" branch when all three are empty), guardrail line (danger-red, hidden unless `guardrail_reason` set) | `strategy.latest.{action,confidence,pace_mode,risk_posture,pit_lap_target,compound_next,undercut_target,guardrail_reason}` |
| **ScenarioBars** | 4 rows (STAY_OUT/PIT_NOW/UNDERCUT/OVERCUT → labels STAY/PIT/UCUT/OCUT), each: bar fill = **min-max normalised** across the 4 raw scores (`fill = (v - lo) / (hi - lo)`, guarded `span = (hi-lo) or 1.0`), winner = `max(raw, key=raw.get)` highlighted in ACCENT, raw signed score printed to 2dp on the right | `strategy.latest.scenario_scores` (case-insensitive key upper-casing) |
| **ReasoningTabs** | 6 tabs: Orchestrator (reasoning text + `memory_block` appended ONLY when `plan_changed` is true — a conditional the design must reproduce exactly, since unconditional display was "measured as wallpaper" per the code's own comment), Pace/Tire/Situation/Radio/Pit (reasoning text + a SECOND, independent set of "key = value" metric lines from `_LINE_BUILDERS`, one function per agent, that duplicates several fields `agent_formatters.py` already formats differently — e.g. `lap_time_pred` appears in both `format_pace`'s headline math and `_pace_lines`' raw dump). Plus **regex syntax highlighting**: 5 compiled patterns (lap refs, `P\d{2}` quantiles, percentages, signed deltas, action keywords) recolour matched spans inside the text via Qt's `QSyntaxHighlighter`. | `strategy.latest.{reasoning,memory_block,plan_changed,per_agent.*.reasoning}` |
| **AgentCard × 6 (shared shell)** | status glyph (●/◐/●/○ for OK/WATCH/ALERT/IDLE), title, rich-text headline (colour), up to 3 rich-text body lines (colour per line), idle dimming (`opacity: 0.45`), optional embedded chart | `format_*` return tuples |
| — Pace (N25) | headline `Δnext ±X.XXXs (YY.YYs)`, body: `pred`, `vs median`, `±CI half-width`; embedded **PaceChart**: actual line, dashed predicted line, filled P10-P90 band, window ≤40 laps (trimmed), values outside 30-200s dropped as TCN-stub noise | `per_agent.pace.*` + `_pace_history` (window's own accumulator) |
| — Tire (N26) | headline `Cliff ~N laps · L{life}` or `cliff stabilising… · L{life}` (branch at `p50 > 100` or `p50 <= 0`), body: range p10-p90, deg rate + compound pill, warning_level; embedded **TireChart**: one `PlotDataItem` per compound stint (colour break at stint boundary, NOT a continuous line), 3-lap centred rolling-mean dashed overlay, translucent cliff band `[current_lap+p10, current_lap+p90]`, dashed p50 marker | `per_agent.tire.*` + `_tire_history` |
| — Situation (N27) | headline `Threat {LOW\|MEDIUM\|HIGH}`, body: `overtake {pct}` OR `overtake — (out of model range)` when `overtake_prob is None` (a **deliberate `None`-vs-`0` distinction the design's §6 CLEAN_CODE directive already names as a scar to avoid reintroducing**), `safety car {pct}` (amber above 15%), `gap {s} · Δpace {signed s}/lap` | `per_agent.situation.*` |
| — Radio (N29) | headline: flag-chip row (≤3) when `alerts` non-empty, else `"no alerts"` when radio/RCM activity exists with no alerts, else `"quiet"`; body: counts line + last RCM line + last radio line (each truncated to 70 chars); **tooltip**: full-lap transcript as Qt-rich-text HTML (`<b>`/`<br>`/`&nbsp;` only, HTML-escaped free text) | `per_agent.radio.{radio_events,rcm_events,alerts}` |
| — Pit (N28, conditional) | headline `pit {p50}s → {compound}` with `" · SC"` suffix when `sc_reactive`, else idle trigger-hint text when not `active`; body: p05-p95 range, `UCUT {pct} → {target}` or `"no undercut target"` | `per_agent.pit.*`, gated on `"N28" in per_agent.active` |
| — RAG (N30, conditional) | headline `"regulation loaded"` or idle trigger-hint when not `active`; body: 70-char answer snippet, ≤3 deduplicated article refs (`"Art. X.Y, X.Z, ..."`, prefix-stripped); **tooltip**: question + up to 4 chunks (truncated 280 chars each, `+N more` footer) as the same constrained Qt rich-text HTML | `per_agent.{rag,regulation_context}.*` (RAG has a legacy-key fallback the port must keep), gated on `"N30" in per_agent.active` |
| **Status bar** | pipeline error message, or `"lap {n} · streaming"` auto-clearing after 1.5s | `strategy.error`, `arcade.lap` |

**What is NOT a pure dict-in/string-out transform** (the specific ask):

1. **`_ReasoningHighlighter` (`reasoning_tabs.py:58-86`)** — a `QSyntaxHighlighter` subclass that
   recolours regex matches live inside a `QTextDocument`. This is a rendering-engine-coupled
   mechanism, not a formatter. **The design's file tree has no module for it** — `ReasoningTabs.tsx`
   is listed as a single file with no `lib/highlight.ts` or equivalent. Porting it means either (a)
   pre-processing the reasoning string into an HTML string with `<span>` wraps per match (consistent
   with the tooltip convention elsewhere) or (b) a CodeMirror/Monaco-style decoration layer. Neither
   is free, and the design commits to neither.
2. **`radio_tooltip_html` / `rag_tooltip_html` (`agent_formatters.py:273-317,478-528`)** — technically
   pure (deterministic, no I/O), but their OUTPUT is a deliberately Qt-constrained HTML dialect:
   the docstrings say verbatim "Qt's tooltip rich-text subset rejects CSS and most layout tags," so
   only `<b>`/`<br>`/`&nbsp;` are used. A web tooltip has the full CSS/HTML surface available. Copying
   these two functions verbatim is possible but would carry a needless Qt-era constraint into a
   context that does not have it — worth flagging as "technically pure, practically a false-purity
   trap" rather than a clean 1:1 copy target.
3. **`classify_action` (`src/arcade/strategy.py:669-681`, re-exported through `theme.py:22-24`)** —
   the action-badge colour/label mapping `OrchestratorCard` depends on is NOT inside
   `agent_formatters.py` at all; it lives in a file the design's §4 "Unchanged" list does not
   mention (`src/arcade/strategy.py`). It is a simple lookup table (harmless to port), but it has no
   designated home in the new layout — `lib/format.ts` is described only as "lap times, gaps, deltas."
4. **`ScenarioBars.update_from` (`scenario_bars.py:97-142`)** — real, non-decorative algorithm: shift
   the 4 raw (possibly all-negative) scores by their minimum, scale by range, tie-break the winner
   with `max(raw, key=raw.get)`. Pure in the FP sense, but a naive "it's just 4 bars" read under-rates
   it; getting the shift/scale/tie-break wrong changes which scenario reads as the winner.
5. **`PaceChart`/`TireChart` `update_from`** — confirmed non-trivial: lap-time sanity filtering
   (30-200s), per-stint segmentation (`TireChart`, a compound-boundary-aware for-loop building a
   `list[_Stint]`), a 3-lap centred rolling mean, and cliff-band geometry keyed off the *current* lap
   plus p10/p50/p90 offsets. The design's §6 already flags the ECharts-via-refs discipline for this
   class of problem, so this is confirmation rather than a new gap — but it is a bigger algorithmic
   surface (~250 combined lines across the two files) than "chart" suggests.
6. **`_LINE_BUILDERS` in `reasoning_tabs.py:108-180`** — a SECOND, independent formatting layer (5
   functions: `_pace_lines`, `_tire_lines`, `_situation_lines`, `_radio_lines`, `_pit_lines`) that
   renders largely the same underlying fields `agent_formatters.py` already formats, but as raw
   `key = value` diagnostic dumps rather than headline prose. Not mentioned as a distinct porting
   target anywhere in the design's file tree; an implementer who ports only `agent_formatters.py`'s
   functions and calls the AGENTS window "1:1" will have silently dropped every reasoning tab's
   metrics fallback.

**Severity: P1.** The claim that `agent_formatters.py` is pure holds for its core headline/body
formatters (items verified clean: `format_pace`, `format_tire`, `format_situation`, `format_radio`,
`format_pit`, `format_rag`, `_status_colour`, `_signed`, `_truncate`, `_format_article_refs`). The
claim frays at exactly the two places the design's own instruction anticipated (tooltips) plus three
places it did not (the syntax highlighter, the external `classify_action` dependency, the parallel
`_LINE_BUILDERS` layer). None of these are blockers; all of them are checklist items a "1:1, from
memory" port would plausibly miss, which is the failure mode this claim was checked against.

---

### E. Design tokens — A16 verified, and the drift it warns about has ALREADY happened

`documents/audits/AUDIT_P3_ARCADE.md:110`, finding A16, reads exactly as the design quotes it:
"Palette/classification constants triplicated with no drift guard: `config.py` (arcade) vs
`dashboard/theme.py` (Qt) vs `telemetry/frontend/app/styles.py` (Streamlit)... The duplication is
deliberate (process isolation), but nothing detects drift."

**Confirmed: `src/telemetry/frontend/` no longer exists** (zero matches for `*styles.py*` or any
`*frontend*` directory under `src/telemetry/`) — the Streamlit surface member of A16's "three" is
gone, exactly as the audit prompt anticipated.

**But the real count today is still three, not two — the third member is now the webapp**: (1)
`src/arcade/config.py:73-84`, (2) `src/arcade/dashboard/theme.py:27-38` (byte-identical to (1) —
confirmed by reading both), (3) `src/telemetry/webapp/src/styles/tokens.css` (the React SPA's live
design-token source, `--bg-*`/`--purple-*`/`--success`/`--warning`/`--danger` custom properties).

**The drift A16 warned about is not a risk, it already happened, measured:**

| Token | `config.py` / `theme.py` (Python, identical to each other) | `webapp/src/styles/tokens.css` (current) |
|---|---|---|
| Background | `(18,17,39)` → `#121127` | `--bg-1: #0c0d14` / `--bg-2: #111827` (neither matches) |
| Accent / purple | `ACCENT (167,139,250)` → `#a78bfa` | `--purple-600` (**PRIMARY**) `#6c5ce7` — `#a78bfa` is close to the webapp's `--purple-300` **hover** state, not its primary |
| Success | `(16,185,129)` → `#10b981` (emerald) | `--success: #43ff64` (bright green) |
| Warning | `(245,158,11)` → `#f59e0b` (amber) | `--warning: #ffbd33` |
| Danger | `(239,68,68)` → `#ef4444` | `--danger: #ff5733` |

Every single semantic colour differs. `config.py:69-72`'s own comment claims "Mirrors
`src/telemetry/frontend/app/styles.py`... The Streamlit file owns the canonical hexes" — that
comment is **doubly stale**: the file it names no longer exists, and even when it did, the CURRENT
canonical source (the webapp) has since moved to a different, more refined palette (a full 0-900
purple ramp, WCAG-tuned rgba foregrounds) the Python side never picked up. This is the exact,
unmonitored drift A16 predicted, now with executed evidence rather than a hypothetical.

**What the design's plan actually fixes, and what it leaves untouched.** §4/§7 add a FOURTH copy
(`src/pitwall/ui/src/styles/tokens.css`) and a drift test against the webapp's tokens.css — that
closes drift for the *new* pair going forward. It does nothing for the *existing, already-drifted*
pair: `config.py`'s Python RGB palette (which survives this design — it is not part of the
`dashboard/` deletion and is still needed for pyglet track/HUD colours) has zero test coverage
against the webapp, before or after PITWALL ships. `theme.py`'s copy disappears when `dashboard/` is
deleted (Claim A), which trims the count from three to two, but the survivor is the one nobody is
testing.

**Severity: P2.** Not a blocker for the design as scoped (it only promised to guard the pair it
introduces), but the design's framing ("per P3 finding A16") implies A16 is being addressed, when in
fact the drift A16 flagged is confirmed live today and will still be live, unguarded, after PITWALL
ships. A cheap, in-scope fix: extend the same drift test to also assert `config.py`'s tuples match
`tokens.css`'s hexes (a five-line addition to the same test), or file a follow-up issue naming this
specific gap so it does not read as closed.

---

### F. webapp `features/strategy` components — honest per-component inventory

Read all 8 files under `src/telemetry/webapp/src/features/strategy/components/` and 3 chart
components under `src/telemetry/webapp/src/charts/`. The design's own hint (RaceTrace is a
single-driver trace, not a gapper) is **confirmed correct** — and the same collision pattern
recurs in components the design did not name.

| Component | What it actually is | Reusable for AGENTS window (Qt layout preserved)? |
|---|---|---|
| `RaceTrace.tsx` (30.1K) | Single-driver pace-over-laps trajectory with a "spotlight" window overlay; **always plots one driver's whole lap range**. Docstring confirms verbatim. | **Name collision only.** DATA window's `RaceTraceChart.tsx` (§4 file tree) needs a multi-driver GAP/interval plot. Zero code overlap beyond both being ECharts line charts. Exactly the risk the design flagged for itself. |
| `ScenarioScoresChart.tsx` (16.3K, in `charts/`, aka `ScoresPlot`) | P10-P90 error-bar/whisker plot over the same 4 MC candidate scenarios (`scenario_scores`) `ScenarioBars` renders, using a custom ECharts `renderItem` series (no built-in horizontal error bar) | **Strong, real reuse candidate the design does NOT name.** It plots the identical data ScenarioBars needs, and does it with a P10-P90 band the Qt bar chart never had (Qt shows only the point score). The design's file tree lists a plain `ScenarioBars.tsx` as if porting the Qt min-max bar verbatim; this component is arguably the better source to port FROM. |
| `AgentModelChart.tsx` (11.4K, in `charts/`) | Reusable "actual vs model estimate" line chart (solid actual, dashed model line, one point per lap) backing the webapp's Pace/Tyres agent-breakdown tabs | **Strong, real reuse candidate, unnamed by the design.** Structurally close to `PaceChart`'s actual/predicted lines (missing only the CI band `PaceChart` has and the per-stint colour breaks `TireChart` has) — a much closer starting point than reimplementing from the Qt pyqtgraph code from scratch. |
| `Gauge.tsx` (6.4K, in `charts/`) | Radial progress gauge for a bounded 0-1 metric (used for overtake/SC/undercut probability) | Plausible alternative to Situation card's plain percentage text lines — not in the Qt original, so "reuse" here would be a deliberate upgrade, not a port. Worth the design author's judgement call, not a defect either way. |
| `DecisionBanner.tsx` (6.7K) + `DecisionDetails.tsx` (5.6K) + `ContingencyList.tsx` (2.8K) | Together render `StrategyRecommendation`: action + confidence + pace/risk instruction + pit plan + MC-scores chart (banner), then key risks / IF-THEN playbook / reasoning / regulation context (details) | **Closest field-level match to `OrchestratorCard` + `ScenarioBars` combined** — same underlying fields (action, confidence, pace_mode, risk_posture, pit target, contingencies). But built around a one-shot POST `/recommend` response, not a 10 Hz push stream; the composition/field-selection is reusable, the data-binding model is not. |
| `AgentTabs.tsx` (22.1K) | 4 segmented tabs (Pace/Tyres/Situation/Pit, NOT 6 — no Radio, no RAG, no Orchestrator) showing each sub-agent's raw output, each tab **lazy-fetched on demand** via `useAgent` + TanStack Query keyed on `(gp, driver, lap)` | **Conceptual collision, not a code-reuse target**, same pattern as `RaceTrace`. The Qt `ReasoningTabs` is push-fed at ~1 update/lap-change from a live stream and includes a `QSyntaxHighlighter` (Claim D, finding 1) this component has no equivalent of — it renders structured stat rows, not highlighted free text. A port that assumes "AgentTabs already does this" would be wrong on both the tab count and the data-fetch model. |
| `AgentDeliberation.tsx` (4.9K) | A **scripted 4-stage loading narration** (`elapsedMs`-driven fake progress: "Building lap state... -> Running agents... -> Scoring 500 MC... -> Synthesizing...") built specifically because `/recommend` "has no progress channel of its own — it returns exactly once, at the very end" | **Solves a problem PITWALL does not have.** PITWALL's tick stream already IS a real progress channel (frame-by-frame lap decisions arrive live); porting this component would import a workaround for a constraint that does not exist in the new architecture. |
| `LapReadout.tsx` (7.2K) | Dense single-line pit-wall timing readout (lap, gap, compound, weight-and-colour hierarchy instead of boxed stat cards) | Loosely relevant to DATA window's `StatusStrip.tsx` (band 1) as a layout idiom, not as portable code — different data source (`lap_state` REST fetch vs. TICK channel). |
| `ScenarioBar.tsx` (10.6K) | The Strategy tab's *scenario-configuration* form (GP/driver/rival pickers, lap-range slider, Run button) — **not a chart**, a name collision with `ScenarioBars`/`ScenarioScoresChart` in the design's own vocabulary | **Not related to AGENTS window at all.** Flagging because the design's file tree uses the name `ScenarioBars.tsx` for the AGENTS-window port and the webapp has THREE similarly-named things (`ScenarioBar.tsx` the config form, `ScenarioScoresChart.tsx` the MC evidence chart, and the Qt `scenario_bars.py`) — an implementer skimming the webapp tree for "the scenario component" has a 1-in-3 chance of opening the wrong one. |

**Severity: P2.** Claim F is not wrong that there is code to borrow from — it is incomplete in a way
that would waste effort: the design's only named example (RaceTrace) is a correctly-flagged
non-match, while at least two real, well-matched candidates (`ScenarioScoresChart.tsx`,
`AgentModelChart.tsx`) go unnamed. An implementer following the design's prose alone would likely
either reinvent `ScenarioScoresChart`'s error-bar rendering from scratch or copy `RaceTrace`'s
single-driver logic into a multi-driver slot and have to unwind it later.

---

### G. Issue reconciliation table (section 9) — checked against `gh issue view`

Fetched full bodies of #281-#287 and #199 via `gh issue view --json title,state,body`. All 8 are
still `OPEN`.

| Issue | Design's stated fate | Verified? | The contradicting/preserved sentence |
|---|---|---|---|
| **#281** epic | "Body rewritten: the Arcade split, the destination, and the relay sentence are all wrong." | **Correct, and understated by one item.** Beyond the relay sentence, the body's own "Decisions (Víctor, 2026-07-06)" section says verbatim: *"MIGRATE the agent-cards window to the #25 web stack... KILL the Arcade telemetry window (this new dashboard replaces it)."* The new design's decision is the opposite of "kill" — the telemetry window **grows** into PITWALL DATA (§1 table, row 1). The design's own table already captures the kill→grow reversal in its "What changed" section, but the fate row for #281 itself only names "the relay sentence" as wrong; the kill/grow reversal is an equally load-bearing sentence in the same body that also needs to be gone when the body is rewritten. |
| **#282** Phase 0 | "Survives unchanged. It is Topic 1 work, independent of where the UI lives." | **Confirmed.** Body is entirely `RaceStateManager.get_rival_states` field additions (R1-R6), an observed/hidden taxonomy split, and boundary-leak tests — nothing in it assumes a browser, a relay, or any UI technology. Holds. |
| **#283** Phase 1 | "Void. Its entire premise was the browser-tab assumption. Close with the reason recorded." | **Mostly correct, one real gap.** The WS relay, the versioned stream schema and the bulk-prefetch endpoint are indeed all browser-tab-only concerns and correctly void. But the body's third bullet is NOT relay-shaped: *"The **gap provider** (per-lap / sub-lap gap evolution from `intervals.parquet`), shared with Phase 0."* This is a genuine data-plane building block — and it is exactly what the design's OWN §8 risk 5 needs and flags as unverified ("The design assumes they [gap, interval, rival lap times] come from BULK... A gate should check that assumption holds for gaps specifically"). If #283 is closed wholesale without rehoming this specific bullet into #282 or into `session_data.py`'s scope, the one concrete task that would resolve risk 5 disappears with the rest of the void issue. |
| **#284** Phase 2 | "Rewritten as the PITWALL DATA window." | **Confirmed, with one scope note.** Body lists "timing tower, per-driver telemetry traces, tyre/stint/strategy board, gap/interval charts, **track map with positions**, weather, sector deltas, SC/flag status" — all present in some form across DATA's four bands (§4) except the full track map, which is a much smaller ask now that Arcade's pyglet window still exists as a separate, always-present process (`TrackRing.tsx` is described only as "band 4 corner," implying a small ring, not the cartographic map #284 originally scoped for a browser tab that would NOT have had pyglet alongside it). Not a contradiction, just a scope reduction worth naming explicitly rather than leaving implicit. |
| **#285** Phase 3 | "Rewritten: both Qt windows die, the cards move to PITWALL AGENTS 1:1." | **Confirmed, and the same kill/grow sentence recurs here too.** Body: *"KILL the Arcade telemetry window (the Phase-2 dashboard replaces it)... Arcade keeps ONLY the circuit / track-map window (native pyglet). Retire both Qt windows."* Same contradiction as #281: the new design keeps a second subprocess (pywebview, not "circuit-only") hosting BOTH windows. The design's fate is correct ("Rewritten"), but the specific false sentence worth flagging in the rewrite is this "Arcade keeps ONLY the circuit... window" line — under the new design Arcade still spawns exactly one companion process, just as it always has. |
| **#286** Phase 4 | "Survives, still gated on the Rival Agent." | **Confirmed.** Body (rival intent overlay, realism-mode toggle) names no UI technology at all. Holds independent of the desktop-vs-browser decision. |
| **#287** Phase 5 | "Survives, unchanged in spirit: does the circuit view eventually move too." | **Confirmed.** Body asks whether to eventually move the circuit into "Three.js or canvas" inside the web surface. A pywebview host can render Three.js/canvas exactly as well as a browser tab can (it is still a web-rendering surface, just desktop-hosted) — the question #287 poses is unaffected by the desktop-vs-browser decision. Holds. |
| **#199** P3 Arcade epic | "Phase D items D.1/D.2 stay moot. D.3 survives and gains weight. Phases A, B, C unaffected." | **Confirmed.** Body's Phase A/B/C summaries (pipeline duplication, fabricated weather/dead flags/ignored `--provider`, static track re-tessellation) name nothing about the Qt dashboard or transport technology. Phase D's efficiency findings (A7/A8, both cited correctly elsewhere in the design) are the only phase touching the dashboard's existence. Holds. |

**Severity: P2** for the #283 gap-provider bullet (a concrete, real task that risks silently
disappearing inside a "close the whole issue" action) and the recurring "KILL/circuit-only" sentence
in #281/#285 (already implicitly handled by the design's own table, but not explicitly flagged as
the sentence to excise when the body gets rewritten). The rest of section 9 checks out.

---

### H. Packaging reality — "ships inside the wheel" has zero precedent in this repo

**Build backend and package discovery**: `pyproject.toml:1-3` — `setuptools.build_meta`.
`[tool.setuptools.packages.find]:141-148` — `include = ["src*", "scripts*"]`, excludes
`*node_modules*`. `[tool.setuptools.package-data]:185-186` — `"*" = ["*.yaml", "*.yml", "*.json"]`,
**no `.html`, `.js`, `.css`, `.woff2`, or any built-asset extension**. There is no `MANIFEST.in`
anywhere in the repo (confirmed by glob — the only 4 hits are inside `.venv/site-packages/`,
third-party). `[tool.setuptools.exclude-package-data]:188-191` additionally strips `**/*.map`.

**No existing precedent for "a JS build ships inside the Python wheel."** The one other frontend in
this repo, the webapp (`src/telemetry/webapp`, React SPA), is explicitly NOT wheel-shipped — it is
served by `docker compose up` (nginx serving the Vite build) per `scripts/run_webapp.py`'s own
docstring and `ROADMAP.md`'s R3 description. Grepped the whole `.github/workflows/` tree for
`npm ci`, `npm run build`, `npm install`, `actions/setup-node`, `vite build`: the only Node usage
anywhere in CI is `docs.yml`'s `npm install marked ... && node scripts/prerender_docs.mjs` (a
docs-prerendering script, not a build step, and it runs entirely outside `uv build`/the wheel
pipeline). **`ci.yml` has zero Node/npm steps of any kind.** The claim "ships inside the wheel" is
therefore a wholly new packaging capability, not an extension of something already working.

**The existing wheel smoke test proves the gap rather than closes it.** `release-please.yml:52-75`
already runs `uv build`, installs the wheel with `--no-deps`, and asserts every `[project.scripts]`
entry point resolves plus a fixed list of source files exist on disk post-install. It checks
`src/arcade/main.py` — never anything under `src/pitwall/ui/`. This job is exactly where a
"wheel does not contain the built PITWALL UI" regression would be caught, and today it would pass
green on a wheel missing the entire UI, because nothing in it looks for `dist/` output.

**Concretely, what would have to change, itemised:**

1. A CI step (new, or added to `release-please.yml` before `uv build`) that sets up Node and runs
   `npm ci && npm run build` inside `src/pitwall/ui/`, producing `src/pitwall/ui/dist/`.
2. `[tool.setuptools.package-data]` extended with the built-asset extensions actually emitted by
   Vite's default output (`*.html`, `*.js`, `*.css`, plus any font/icon assets used) — the current
   entry only covers `yaml`/`yml`/`json`.
3. A **converse exclusion** for the `ui/` SOURCE tree: the existing repo-wide package-data glob
   (`"*" = ["*.yaml", "*.yml", "*.json"]`) would, unmodified, also sweep `ui/package.json`,
   `ui/tsconfig.json`, and any `.json` fixtures into the wheel as a side effect of matching by
   extension repo-wide — none of that belongs in a shipped artefact. `ui/node_modules/` is already
   safe (existing `exclude = ["*node_modules*"]` at line 148 covers it), and that specific exclusion
   exists **precisely because this exact class of bug already happened once**: its own comment cites
   "#392 D1" — a dev box that ran `npm install` leaking ~600 node_modules files into a wheel via this
   same `"*"` package-data glob. That is a real, already-paid cost in this repo for the SAME
   mechanism the design's plan would re-trigger for `ui/`'s TypeScript source unless explicitly
   scoped to `ui/dist/**` only.
4. The wheel smoke test in `release-please.yml` extended to assert the built UI assets are actually
   present post-install (mirroring how it already asserts `src/arcade/main.py` exists) — otherwise a
   broken packaging step ships silently, exactly the failure mode the existing test exists to catch
   for the Python entry points.

**Named risk, not in the design's own §8 list.** Section 8 enumerates 8 risks (duplication, Linux
webview dependency, double session load, unmeasured pull cadence, wire-vs-bulk gap data, RCM
messages, stale prose, stale docstring). **None of the 8 is "will the wheel actually contain the
built UI."** Given this is the FIRST time this repo would ship a JS build inside a Python artefact,
and the repo has already been bitten once by the adjacent failure mode (#392 D1, node_modules
leaking into a wheel via the same package-data mechanism), this belongs on that list.

**Severity: P1.** Not a design flaw in the architecture itself — `pywebview` loading a `dist/`
directory off disk is a completely standard pattern — but the packaging mechanics to make that
`dist/` directory actually exist inside an installed wheel are unbuilt, untested, and unmentioned as
a risk, in a repo whose own CI has never once run a JS build step.

---

### I. Cross-check against `CLAUDE.md` §11 and `CONTRIBUTING.md`

`CONTRIBUTING.md`'s own entry-points table (line 43) is a second live, contributor-facing file
carrying the same "PySide6 dashboard" claim caught in Claim A — folded into that table above rather
than repeated. Nothing in `CONTRIBUTING.md`'s "What NOT to touch" list (lines 69-81) blocks this
design: `src/arcade/dashboard/` is not untouchable, `src/agents/` internals are correctly left
untouched by the design (it never proposes editing them), and `scripts/run_simulation_cli.py` /
`notebooks/**` / `legacy/**` are irrelevant to this surface. No conflict there.

**Does the design repeat a recorded lesson?**

- **"A second implementation of something that already exists" (the repo's dominant recorded
  defect, §11's own opening framing, e.g. the 2026-07-16 entries about `RaceStateManager` being
  reimplemented three times).** `session_data.py`'s own §8 risk 1 already names this exact concern
  and requires it to reuse `src/f1_strat_manager/data_cache.py` and the existing loaders rather than
  reading parquet directly — the design pre-empts the lesson rather than repeating it. Good.
- **"Documentation left describing the old behaviour" — 2026-07-16, verbatim: "la documentación no
  estaba desactualizada: enseñaba los bugs."** This is the lesson Claim A's reference table is
  built around, and the design's §8 risk 7 only names two of the ~20 files actually affected
  (`ROADMAP.md`, `docs/pages/roadmap.md`). Per that lesson's own closing line — "cuando un fix
  cambia un contrato, la página que lo describe es parte del fix, no un follow-up" — the design's
  own scope is under-drawn here; see Claim A for the full list.
- **"A test that would assert a constant rather than an effect."** Checked all 5 items in §7: items
  2 (frame-clock truncation fires), 3 (bulk reader shape/count), 4 (JSON-serialisable) all assert an
  EFFECT. Item 1 (token drift, equality between two files) and item 5 (golden payload, frozen shape)
  are closer to constant-assertion in form, but both are explicitly used for their intended purpose
  (drift detection and contract-freezing), which is a legitimate use of a snapshot-style assertion,
  not the anti-pattern the lesson warns about (a canned value standing in for a real invariant).
  No violation found.
- **"One member of a pair fixed, its twin not" — the repo's single most-repeated lesson
  (`feedback_the_twin_that_never_got_the_fix`, referenced at the top of the project's own memory
  index).** This is the exact shape of Claim E: `theme.py` and `config.py` were a matched pair,
  drifted from their claimed source silently, and the design's drift test guards only the NEW pair
  (pitwall vs webapp) while leaving the OLD, already-drifted pair (`config.py` vs webapp) exactly as
  exposed as it is today. If this design ships as scoped, it is a textbook instance of the lesson
  the project's own memory flags as its dominant recurring defect, not a new problem this audit
  invented — see Claim E for the executed evidence.

**Severity: P2**, folded into the Claim A/E findings above rather than a new standalone defect —
recorded here because the audit prompt specifically asked for the cross-check.

---

## What I tried to break and could not

- **The core "desktop app opens its own TCP socket" premise.** Verified Arcade genuinely makes zero
  HTTP calls today (Claim C) and that nothing outside the dashboard package assumes a browser
  transport. The pywebview `js_api` pull model (§3.5) is architecturally sound for this topology;
  I could not find a caller-side reason it would not work.
- **Whether the AGENTS window's data model (owns its own rolling history, fed by the tick stream)
  is internally consistent with the DATA window's model (owns nothing, fed by BULK + tick).** Section
  3.3/3.4's reasoning is self-consistent and matches what `history_tail` actually strips in the real
  broadcast payload (`window.py:171-174`'s own comment, confirmed against the code).
- **The submodule boundary.** Tried hard to find any reference from `src/telemetry/` back into
  `src/arcade/dashboard/`, PySide6, or pyqtgraph — genuinely zero, confirmed by two independent
  greps (broad "dashboard" and PySide6/pyqtgraph-scoped). The submodule is cleanly unaffected.
- **The issue reconciliation table's overall shape** (Claim G) — 6 of 8 fates are exactly correct
  with no caveat; the two with caveats (#281/#285's un-flagged kill/grow sentence, #283's
  gap-provider bullet) are refinements, not reversals of the stated fate.
- **A16's own accuracy as a citation** (Claim E) — the audit itself is right about what it found; the
  gap is only in what changed (Streamlit deleted) and what remains unaddressed (`config.py`) since
  it was written.

## Summary

Eight P1/P2-weight findings, zero P0s: the architecture itself (desktop app, own TCP socket, BULK
vs TICK split, js_api pull) is sound and the reconciliation with existing issues is 85% accurate on
first read. Everything found here is a **scope-completeness** problem, not a soundness problem: an
implementer following the document as written would ship something that mostly works, then hit
(in likely order of discovery) the wheel missing its own UI (H), a crashing dashboard-subprocess
spawn on the exact line that was supposed to "not change" (A), a reasoning tab with no highlighting
and half its metrics missing (D), and a design-token drift test that congratulates itself on fixing
A16 while leaving A16's actual live instance untouched (E).
