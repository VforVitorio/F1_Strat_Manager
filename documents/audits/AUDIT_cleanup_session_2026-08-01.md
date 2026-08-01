# Dedicated clean-code / anti-slop cleanup session — 2026-08-01

**Instrument:** foreground session (Fable), not a background agent. Method: `~/.claude/CLEAN_CODE.md`
as the standard, `ponytail` doctrine as the "simplest thing that works" filter, then senior-dev
judgment ("would a senior engineer keep this?"). This session both finds AND fixes the general
AI-slop findings in scope; the single-contract/architecture question is detected and reported here,
fixed in a separate dedicated session (per `project_cleanup_audit_session_plan`).

**Scope, recomputed mechanically** (not the informal 2026-08-01 audit's broader scope):
`git log --name-only aa1d274~1..HEAD` (aa1d274 = PR #730's merge commit), matching `gh pr list
--state merged --limit 30`, i.e. the last 30 merged PRs, #730-#775, all landed 2026-07-29 to
2026-08-01. Non-notebook, non-legacy files touched in that window:

```
scripts/measure_fresh_reference_gate.py, scripts/measure_fresh_reference_gate_2025.py,
scripts/measure_tyre_reference.py, scripts/run_simulation_cli.py,
src/agents/pace_agent.py, src/agents/pit_strategy_agent.py, src/agents/position_projection.py,
src/agents/race_situation_agent.py, src/agents/strategy_orchestrator.py, src/agents/tire_agent.py,
src/agents/tire_parsing.py, src/arcade/strategy.py, src/simulation/race_state_manager.py,
src/strategy/eval/decision_modes.py, plus CLAUDE.md (checked on explicit instruction),
documents/audits/*, documents/eval_reports/*, tests/**, pyproject.toml, uv.lock,
.release-please-manifest.json, CHANGELOG.md
```

**Important scoping consequence:** this window is almost entirely the epic-724 decision-layer
sprint + the tyre fresh-reference gate — a slice that already went through heavy adversarial
scrutiny (FABLE_G1-G3, FABLE_S1-S5, MEASURE_*, AUDIT_A1-A5; see `project_epic724_scorecard`). Most
of the informal 2026-08-01 audit's headline findings (A1 RCM-classifier bench-script fork, A3
`theme.py`'s `_ACTION_STYLE` mirror, M4 bootstrap-preamble duplication across ~15 scripts, B1-B4
legacy fossils) live in files **outside** this window (`bench_nlp_pipeline_cpu.py`, `theme.py`,
`gap_calculation.py`, `tire_predictor.py`, `fastf1_extractor.py`) and are therefore **not touched in
this session** — not because they stopped being valid, but because the user's explicit instruction
was to scope this session to files touched in the last ~30 merged PRs, recomputed mechanically, not
a fixed list. They remain open for a future sweep once a new 30-PR window covers them, or a
deliberately widened scope.

---

## Part 1 — Verification of the 2026-08-01 informal audit against the recomputed scope

| ID | Finding | In this scope? | Status |
|---|---|---|---|
| A1 | RCM classifier 3rd copy in `bench_nlp_pipeline_cpu.py` | No (`scripts/bench_nlp_pipeline_cpu.py` not touched in #730-#775) | Deferred, still valid per informal audit |
| A2 | `_build_race_state` CLI vs Arcade divergence | **Yes** (`run_simulation_cli.py`, `arcade/strategy.py`) | **Confirmed, expanded — see Part 3 (architecture report)** |
| A3 | `theme.py` `_ACTION_STYLE` mirror | No (`src/arcade/dashboard/theme.py` not touched) | Deferred, still valid per informal audit |
| M1 | Dead code: pace_agent constants, 4× `get_*_react_agent`, `simulate_real_time_predictions` | Partially — pace_agent.py constants and all 4 `get_*_react_agent` fns are in scope; `tire_predictor.py` is not | **Fixed here (see Part 4)** for the in-scope pieces |
| M2 | Restated constants: `total_laps=57`, weather defaults, tire stub duplication, compound RGB | **Yes**, all example files are in scope except `theme.py`'s RGB tuples | **Fixed here** for `total_laps` and the tire stub; weather defaults folded into the architecture report (see below) |
| M3 | CLAUDE.md doc drift (`f1-streamlit`, `src/shared/`) | Yes, user named CLAUDE.md explicitly | **Fixed here** |
| M4 | Bootstrap preamble duplication across ~15 scripts | No (only 3 of the ~15 scripts are in this window, and none is the divergence example) | Deferred |
| B1-B4 | Legacy fossils, false docstrings | No (`src/vision/`, `strategy/inference/tire_predictor.py`, `fastf1_extractor.py` not touched) | Deferred |
| B5 | Minor twins (`_bar_style`, `measure_*_gate*.py` near-identical harness, `_load_ner_model`) | `measure_fresh_reference_gate.py` vs `_2025.py` is in scope | Re-verified: still a deliberately-frozen reproducibility-script pair per the original audit's own judgment call — not touched |

---

## Part 2 — New findings from this session's deeper sweep (not in the informal audit)

### N1. `race_situation_agent.py` disagrees with itself on the TrackTemp fallback

`_compute_weather_features` (line 676) reads `session_meta.get('TrackTemp', 35.0)` — but the
**same file's own** `run()` (line 1314) and `run_from_state()` (line 1401) build that very
`session_meta['TrackTemp']` key with a fallback of **38.0** when the source weather is missing. If
`_compute_weather_features` is ever called on a hand-built `session_meta` that omits `TrackTemp`
(e.g. a test fixture), it silently produces a temperature 3°C different from what the rest of the
file considers its own default — a same-file twin-that-never-got-the-fix, not a cross-file one.
**Fixed** (Part 4): align to 38.0, the value the file's own constructors use.

### N2. `tire_agent.py::_run_core` restates its own "conservative stub" twice, byte-for-byte except the message

Lines 1566-1578 and 1609-1621 both construct an identical `TireOutput` (same `deg_rate=0.03`,
`laps_to_cliff_p10/50/90=20.0/30.0/40.0`) with only the `reasoning` string differing. The second
site's own comment says "Fall back to the same conservative stub the wet/intermediate branch above
already uses" — the intent to share was explicit, the code never did. **Fixed** (Part 4): extracted
`_conservative_tire_stub(...)` helper.

### N4. `pace_agent.py`'s entire LangGraph ReAct scaffold is unreachable — flagged, not removed

Beyond the dead `get_pace_react_agent()` free function (M1, fixed below), the whole
subsystem it wrapped is unreachable too: the `get_react_agent()` instance method,
`self._react_agent`, `PACE_TOOLS`, `predict_pace_tool`, `get_session_median_tool`, and
`_PACE_SYSTEM_PROMPT` (`pace_agent.py:752-988`, confirmed via a repo-wide grep for
`.get_react_agent(` — zero callers anywhere, unlike `tire_agent`/`race_situation_agent`/
`pit_strategy_agent`, whose identical-shaped scaffolds ARE called internally from
`_run_core`). `PaceAgent.run()`/`run_from_state()` call the XGBoost model directly and
never touch any of it.

**Not removed.** The section header directly above it reads `# LangGraph tools and ReAct
agent (preserved 100% — no functional changes)` — a signal that a past change
deliberately chose to keep this block untouched, most likely for eventual parity with
the other three agents' ReAct capability. Deleting ~235 lines on the strength of "zero
callers" alone would override that stated intent without knowing the reason behind it.
Flagging for Víctor: either delete it (if the parity plan is abandoned) or wire it into
`run()`/`run_from_state()` (if still planned) — this session does neither.

### N3. Weather-default divergence spans pace/tire/race_situation agents (3 different value sets) — folded into the architecture report

`pace_agent.py` (`run_pace_agent_from_state`) defaults air/track temp to **25.0/35.0** reading the
raw `weather` sub-dict directly; `tire_agent.py` and `race_situation_agent.py` default to
**28.0/38.0** reading their own already-transformed `session_meta['AirTemp'/'TrackTemp']`. This is
the same disease as `_build_race_state`'s weather divergence (Part 3) — multiple independent
"assemble race/session context with a plausible-looking default" implementations that were never
reconciled — so it is reported as architecture evidence, not fixed here. Picking a single canonical
value requires a product decision this session should not make unilaterally.

---

## Part 3 — Architecture question: is there duplicated "backend"/state-assembly logic that should be single-sourced?

**Detected and reported only, per explicit instruction — not fixed in this session.**

### The three confirmed independent implementations

All three build a `src.agents.strategy_orchestrator.RaceState` from the same conceptual
`lap_state` dict shape (`RaceStateManager.get_lap_state()`'s contract, CLAUDE.md §6). This is
confirmed, not just "hinted at via a comment" — a fourth suspect, `simulator.py::_local_build_race_state`,
turned out to be a genuine **thin wrapper** over the backend copy (its own docstring: "Thin wrapper
over the shared `build_race_state` helper"), so it does not count as a fourth implementation.

| Field | CLI `run_simulation_cli.py:1304-1373` | Arcade `src/arcade/strategy.py:599-715` | Backend `src/telemetry/backend/utils/race_state_builder.py:53-120` |
|---|---|---|---|
| `total_laps` | `lap_state["session_meta"]["total_laps"]` — **direct index, raises if absent** | `meta.get("total_laps", 57)` | `meta.get("total_laps", 57)` |
| `compound` | `.get("compound", "UNKNOWN")` | `.get("compound", "MEDIUM")` | `.get("compound", "MEDIUM")` |
| `tyre_life` | `.get("tyre_life", 0)` | `.get("tyre_life", 1)` | `.get("tyre_life", 1)` |
| `air_temp` | `.get("air_temp", 25.0)` | `.get("air_temp", 25.0)` | `.get("air_temp", 25.0)` |
| `track_temp` | `.get("track_temp", 40.0)` | `.get("track_temp", 35.0)` | `.get("track_temp", 35.0)` |
| `position is None` | raises `ValueError` (#628) | raises `ValueError` (#465) | raises `ValueError` (#465) |
| `gap_ahead_s` unknown fallback | `GAP_UNKNOWN_FALLBACK_S` (imported, single-sourced) | `GAP_UNKNOWN_FALLBACK_S` (imported, single-sourced) | `GAP_UNKNOWN_FALLBACK_S` (imported, single-sourced) |
| `radio_msgs`/`rcm_events` | left empty by `_build_race_state` itself, populated **370 lines later in the main loop** (`run_simulation_cli.py:1739-1762`, `.extend()`/`.append()` on the already-built `race_state` object) | populated **inline inside** `_build_race_state` from `RadioPipelineRunner` | accepted as **parameters** of `build_race_state`, `None` → `[]` |

`compound` and `track_temp` are the two fields where CLI disagrees with BOTH other copies
(`"UNKNOWN"` vs `"MEDIUM"`, `40.0` vs `35.0`), confirming the informal audit's claim with exact
line numbers. `total_laps`'s CLI copy is actually the "more correct" one (fails loudly instead of
guessing), which itself is a divergence in *error-handling philosophy*, not just in literal values.

**Correction (caught by this session's own adversarial gate, see the GATE report):** an earlier
draft of this document claimed the CLI's `RaceState` ends up "radio-blind" — that claim was
**refuted**. The CLI is not missing radio/RCM context: `run_simulation_cli.py`'s main loop mutates
`race_state.radio_msgs`/`race_state.rcm_events` in place immediately after `_build_race_state`
returns (lines 1739-1762 — real corpus radios/RCMs via `.extend()`, the SC-tracker's synthetic
re-assertion, and simulated radios via `.append()`), and the arcade docstring's own words ("same as
the CLI") already said as much. The real divergence is **structural, not functional**: Arcade and
the backend build `radio_msgs`/`rcm_events` as part of the single `_build_race_state`/`build_race_state`
call, while the CLI splits that responsibility — the builder returns an empty-list `RaceState` and a
separate, later block in the caller fills it in. Whether that split is intentional (the CLI's
main loop already owns the `sc_tracker`/`RadioPipelineRunner` instances for other reasons) or an
artifact of the CLI predating the other two surfaces is exactly the kind of question the next
dedicated session should resolve — but it is a "where does this responsibility live" question, not
a missing-data one.

### The prior unification attempt, and its documented blocker

The maintainers have **already reasoned about this exact question once**. `arcade/strategy.py:599-606`'s
own docstring:

> "Duplicate of `_local_build_race_state` from simulator.py, small enough to inline so the arcade
> stays independent of `backend.utils.race_state_builder` (which requires a sys.path shim that only
> the FastAPI startup provides)."

And `arcade/strategy.py:641-644`:

> "The telemetry backend still has its own copies and is a submodule, so this is a two-way
> unification, not the three-way one an earlier draft of this comment claimed."

So a **two-way** unification already happened for exactly one field (`GAP_UNKNOWN_FALLBACK_S`,
imported from `src.agents.position_projection` in all three copies — genuinely single-sourced, not
duplicated). A **three-way** unification of the whole function was considered and explicitly
rejected, for a concrete, named reason: `src/telemetry` is a **git submodule** with its own FastAPI
startup path that establishes a `sys.path` shim CLI/Arcade do not have, so importing
`backend.utils.race_state_builder` from `src/arcade` or `scripts/` is not just a style choice, it is
currently **not importable** without that shim.

### What the next dedicated session needs to resolve

1. **The import-boundary blocker is real and needs a design decision**, not a mechanical merge:
   either (a) give CLI/Arcade a way to import the backend module (requires resolving the sys.path
   shim, and accepting a dependency from `src/` onto the `src/telemetry` submodule — a layering
   change), or (b) move the shared logic to a location both sides can import without the submodule
   boundary (e.g. a new `src/shared_race_state.py` that the backend then imports, inverting today's
   direction), or (c) accept 3 copies and enforce parity with a test that diffs their literal
   defaults (cheapest, weakest).
2. **`radio_msgs`/`rcm_events` responsibility placement** — not a data gap (see the correction
   above), but worth deciding: should the CLI's `_build_race_state` build these fields inline like
   Arcade/backend do, or should Arcade/backend instead adopt the CLI's split (build empty, populate
   from the caller)? Either answer removes one more structural difference between the three copies.
3. **`compound`/`track_temp`/`tyre_life`/`total_laps` fallback values** need one canonical answer
   each — this session found the weather-default divergence extends further (`pace_agent.py`'s
   25.0/35.0 is a *fourth* value set for the same physical quantities, Part 2 §N3), so the "pick one
   number" scope is bigger than just the three `_build_race_state` copies.
4. Whichever direction is chosen, it touches the untouchable CLI PMV (`run_simulation_cli.py`) and
   the `src/telemetry` submodule — per the project's issue-first rule for important/risky changes,
   file the issue before writing code, as `project_cleanup_audit_session_plan` already specifies.

---

## Part 4 — Fixes applied in this session

| # | Finding | Fix | Files |
|---|---|---|---|
| 1 | M1: 3 dead constants (`_CLUSTER_PARQUET`, `_LAPS_FEATURED`, `_FEATURE_MANIFEST`) | Deleted — zero references anywhere in the repo, `_load_encoding_maps` builds the same paths inline | `src/agents/pace_agent.py` |
| 2 | M1: 4 dead `get_*_react_agent()` free functions (~90 lines) | Deleted, plus their stale mentions in each file's own "Public API" module docstring | `pace_agent.py`, `tire_agent.py`, `race_situation_agent.py`, `pit_strategy_agent.py` |
| 3 | M2: `total_laps` fallback literal `57` restated 6× across 3 files | Extracted `DEFAULT_TOTAL_LAPS` into a new leaf module `src/agents/_shared_defaults.py` (mirrors the existing `guard_rails.py` leaf-import pattern already used in `pit_strategy_agent.py`), imported everywhere | `pit_strategy_agent.py` (×3), `race_situation_agent.py` (×2), `tire_agent.py` (×1) |
| 4 | N1: `race_situation_agent.py` disagreed with itself on `TrackTemp` fallback (35.0 vs 38.0) | Aligned `_compute_weather_features`'s fallback to 38.0, matching the file's own `run()`/`run_from_state()` | `src/agents/race_situation_agent.py` |
| 5 | N2: `tire_agent.py::_run_core` restated the same "conservative stub" `TireOutput` twice | Extracted `_conservative_stub()` static helper, both call sites now pass only the differing `reason` string | `src/agents/tire_agent.py` |
| 6 | M3: CLAUDE.md documented `f1-streamlit`/Streamlit (retired for `f1-webapp`/React SPA in #551) in 4 places, and described `src/shared/` as the live extraction home instead of the archived one | Updated §1 overview, §2 tech stack table, §3 project tree (added `data_extraction/`, marked `shared/` archived, corrected `telemetry/`'s description), §9 tooling note, §10 skills table | `CLAUDE.md` |

### Findings detected but deliberately NOT fixed (flagged for follow-up, not silently dropped)

| # | Finding | Why not fixed here |
|---|---|---|
| N4 | `pace_agent.py`'s entire LangGraph ReAct scaffold (~235 lines: `get_react_agent`, `_react_agent`, `PACE_TOOLS`, both tools, the system prompt) is unreachable — confirmed via repo-wide grep, zero callers | Its own section header says "preserved 100% — no functional changes", signalling deliberate intent (likely future parity with the other 3 agents). Deleting on "zero callers" alone would override that without knowing the reason. Flagged for Víctor to decide. |
| N3 | Weather-default divergence across `pace_agent.py` (25.0/35.0) vs `tire_agent.py`/`race_situation_agent.py` (28.0/38.0) | Same disease as the architecture question (Part 3) — multiple independent "assemble race context with a plausible default" implementations. Picking one canonical value is a product decision, not a mechanical dedup; folded into the architecture report instead. |
| A2/Part 3 | `_build_race_state` triplicated across CLI/Arcade/telemetry-backend | Explicit user instruction: detect and report only, fix in the dedicated follow-up session. |

### Verification

- `ruff check` on every touched Python file: clean (only `F821` is enforced inside `src/agents/**` per `pyproject.toml`'s deliberate per-file-ignores; nothing else applies there).
- `ruff format --check` reports these files as "needs formatting" — **expected and not a regression**: `[tool.ruff.format]` explicitly excludes `src/agents/**` (deliberate hand-aligned `=` style), so this never runs in CI either.
- AST parse + `importlib.import_module` on all 5 touched agent modules: clean.
- `pytest tests/agents/ tests/audit/`: 129 passed.
- `pytest tests/mc/ tests/simulation/ tests/eval/`: 286 passed.
- Real `f1-sim Budapest NOR McLaren --no-real-radios --no-llm` run: all 70 laps OK, positions P5 → P1,
  actions STAY_OUT·59 / PIT_NOW·5 / UNDERCUT·6, no errors.
- Independent adversarial gate (Fable, separate agent, full details in
  `documents/audits/AUDIT_cleanup_session_2026-08-01_GATE.md`): reproduced the test runs and the
  real `f1-sim` run itself, re-verified every "zero callers" claim, confirmed all dedup sites
  behaviourally identical, and found 1 HIGH (the radio_msgs/rcm_events claim corrected above), 3
  MEDIUM (this branch's non-bisectable commit order, this section's then-unfilled placeholders, a
  stale `docs/pages/agents-api.md` line), 3 LOW (a "2022-2025" vs "2023-2025" dataset-year slip in
  the new constant's comment, a line-number-hardcoded comment, a stale "unchanged" docstring claim
  in `pace_agent.py`). All 7 fixed post-gate; see the branch's commit history for the reordering and
  follow-up commits.

This is where PR #776 stopped (merged to `dev`, CI green). Part 5 below is a second wave on the
same branch pattern, done the same session at Víctor's explicit request to keep going.

---

## Part 5 — Second wave: extending past the 30-PR window

Víctor asked, after #776 merged: what else can this session fix, including outside the 30-PR
window, including `tests/`? This part covers that follow-up sweep.

### 5.1 — `tests/` audit (a dedicated Explore agent, read-only)

Findings, then fixes:

| Severity | Finding | Fix |
|---|---|---|
| Medium | `_canned_outputs()` — a 30-line fixture (4 dataclass builds) byte-identical between `tests/mc/test_mc_state_helpers.py` and `tests/mc/test_strategy_goldens.py`, **and** three more files (`test_mc_is_a_real_decision.py`, `test_projection_golden.py`, `test_tyre_wear_term.py` ×2 call sites) each imported it from whichever of the first two happened to define it — five consumers, two competing definitions (the third consumer, `test_tyre_wear_term.py`, was missed in the first pass and only found by the follow-up adversarial gate, see 5.8). | Extracted to `tests/mc/canned_outputs.py::canned_outputs()`, matching the existing `tests.X.Y import Z` cross-file pattern already used by `tests/engine/ast_helpers.py`. All 5 consumers now import the one definition. |
| Low-Medium | `_HAS_MODELS = (data/models/tire_degradation/routing_config.json).exists()` (+ a `_skip_no_models` marker) restated in 22 test files (21 module-level `skipif` guards + 1 fixture-scope `pytest.skip` in `test_pace_orchestrator_hardening.py`, also missed in the first pass and found by the gate). Two had already drifted to `.is_file()`; one used a `_MODELS_DIR` intermediate variable. 4 further files check a **different** model artifact on purpose (overtake/pit-prediction/lap-time) — correctly left alone, not slop. | Centralised the shared tire-routing-config guard into `tests/conftest.py` (`HAS_TIRE_MODELS`, `skip_no_tire_models`), imported by all 22 files that were checking that exact artifact. Confirmed via `ast.get_docstring` + `ast.parse` on every touched file that no import landed inside a docstring (see the incident below) and `ruff format --check` is clean on all 62 files in `tests/`. |
| Info | `_race_state()` name collision between `test_engine_memory.py`/`test_engine_no_llm.py` | Verified genuinely different helpers — not touched. |
| Info | No dead/tautological tests, no orphaned helper files, every import resolves | Nothing to fix. |

**Self-caught incident during this fix:** a first migration script (a naive "insert after the last
line starting with `import`/`from`" heuristic) inserted the new import **inside a module
docstring** in `tests/mc/test_mc_is_a_real_decision.py`, because a docstring sentence happened to
start with the word "from" ("...and that is provable / from the code alone."). Caught by re-reading
the file before committing, not by tooling. Reverted the 17 files the buggy script had touched via
`git checkout --`, rewrote the migration using `ast.parse` to find the real top-level import block
(skipping the docstring `ast.Expr` node explicitly), reran, and this time verified every file with
both an AST parse and an explicit "does the docstring contain the import line" check before moving
on. Recorded here because it is exactly the kind of self-inflicted slop this whole session exists to
prevent — a mechanical fix is not exempt from causing the disease it is curing.

### 5.2 — A3 fixed for real (previous session had left it deferred, out of scope)

`src/arcade/dashboard/theme.py`'s `_ACTION_STYLE`/`classify_action` were a hand-mirrored copy of
`src/arcade/strategy.py`'s originals — the *exact* pattern the file's own comment named as the #620
bug mechanism, sitting 20 lines from the dict (`_ALERT_SEVERITY`) that #620 already fixed. Also
removed `severity_color()` (dead, 0 callers, restated `classify_alerts`'s mapping).

**A real bug found and fixed in the process, not just a merge:** neither copy of `classify_action`
actually guarded a `None` action — both called `.upper()` on it unguarded as the dict lookup key
and would have crashed identically (verified by calling the pre-fix function with `None` in this
session; it raised `AttributeError`). No live caller currently passes `None`
(`orchestrator_card.py` sanitises with `str(latest.get("action") or "--")` first), but the
type-hinted-`str` signature invites it. Fixed once, in the single now-canonical
`strategy.py::classify_action`, with a guard clause and a comment explaining why it wasn't there
before. `theme.py` now imports the fixed version.

### 5.3 — B2 fixed: false docstring in `fastf1_extractor.py`

Verified against the live notebook JSON (`json.load` + scan every cell's source for the string
"fastf1_extractor"): `notebooks/data_engineering/N01_data_download.ipynb` mentions the file exactly
once, in a **markdown** cell, as a link — `[fastf1_extractor.py](../../src/shared/data_extraction/
fastf1_extractor.py)`. No code cell imports it. The module's own docstring claimed the opposite
("Kept because ... still imports it"). Corrected to state the real reason (the markdown link would
break, not any running code) and confirmed via a repo-wide grep that nothing else references the
module either.

### 5.4 — M4 fixed, right-sized: 8 of 16 scripts had a real latent bug, not just duplication

A dedicated Explore agent read all 16 scripts' bootstrap preambles (excluding the untouchable
`run_simulation_cli.py`). Result: **7 scripts already use a robust `.git`-search-with-fallback**
for the repo root (textually near-identical across all 7); **8 use a fragile fixed-depth
`Path(__file__).resolve().parent.parent` / `.parents[N]`**, which silently resolves to the wrong
directory the moment the script is not exactly N levels below the repo root — the exact scenario a
`uv tool install` layout can produce. This is a genuine latent correctness bug, not merely restated
code.

**What was NOT done, and why:** extracting a single importable `find_repo_root()` helper module was
considered and rejected — it has a bootstrapping problem. A script cannot `import scripts.something`
to learn where the repo root is, because it needs the repo root on `sys.path` *before* any
`scripts.*` import can resolve. The `.git`-search snippet is therefore structurally required to stay
inline in every entry-point script; what can and did get fixed is which VERSION of that snippet each
one uses.

**Fix:** replaced the fragile resolution in all 8 files with the same proven `.git`-search snippet
the other 7 already carry (preserving each file's own variable name — `_REPO_ROOT`, `ROOT`,
`REPO_ROOT` — to keep the diff minimal): `build_radio_dataset.py`, `download_data.py`,
`measure_fresh_reference_gate.py`, `measure_fresh_reference_gate_2025.py`, `measure_mc_tables.py`,
`measure_tyre_reference.py`, `verify_drs_zones.py`, `prompt_ab/_common.py`. Verified: syntax +
ruff clean on all 8; `download_data.py` was run for real (fully executed `ensure_setup()`,
downloaded/verified 144 HF files, completed successfully); `verify_drs_zones.py --help` rendered
correctly; the remaining 6 import cleanly with no import-time error.

### 5.5 — Evaluated, evidence expanded, still NOT fixed (higher-risk than previously scoped)

| # | Finding | What changed from the earlier characterisation |
|---|---|---|
| A1 | `bench_nlp_pipeline_cpu.py`'s `_classify_rcm_event` vs the canonical `src/f1_strat_manager/rcm_events.py::classify_rcm_event` | The prior audit's "~30 min, thin adapter, ~95 lines removed" estimate undersold the real scope. Direct comparison this session: the bench copy branches PER CATEGORY (`SafetyCar`/`Flag`/`Drs`/`CarEvent`/`Other`, each with its own keyword set) and produces event types the canonical classifier does not have at all (`CAR_MECHANICAL`, `TRACK_CONDITION`, `LAPPED_CARS_OVERTAKE`, `PIT_EXIT` vs canonical's `PIT_LANE_CLOSED`/`PIT_LANE_OPEN`). A drop-in swap, the same pattern `eval/nlp.py:590` uses, would silently reclassify a large share of `CarEvent`/`Other` rows to `OTHER` — changing the benchmark's meaning, not just its code path. The right fix is a product decision (extend the canonical classifier to cover the missing branches, or accept the benchmark's numbers move) — not a mechanical adapter. Flagged for the next dedicated pass, with the exact branch-by-branch diff now on record here instead of assumed away. |
| B4 | `_arrow_pick()` in `scripts/cli/pickers.py:59`, the prior audit's "cleanest decomposition candidate" | Re-read in full this session. It is already decomposed into four named closures (`_text`, `_redraw`, `_confirm`, `_jump_to_digit`); the remaining ~120 lines are two full platform-specific raw-terminal key-reading loops (Windows `msvcrt` vs POSIX `termios`+`tty`) that cannot share a body without introducing a key-reading abstraction neither platform actually needs elsewhere, and that abstraction cannot be exercised without a live TTY (untestable in this environment). Judgment: this is inherent complexity from housing two real platform implementations side by side, not disorganisation — matches CLEAN_CODE.md's own safeguard against buying fewer lines with worse architecture. Not touched. |
| B1 | `src/vision/gap_calculation.py` byte-identical to `legacy/app_streamlit_v1/.../gap_calculation.py` | Re-verified byte-identical (`diff -q`, no output). Re-reading the prior audit's own text more carefully: `src/vision/`'s copy is "the only one anything imports" — i.e. it is load-bearing, not the dead half. The prunable copy is inside `legacy/`, which is explicitly frozen for this session (and for the whole project's untouchable-zone rule). Not actionable without either editing `legacy/` or breaking a real import; left as a finding for whoever owns a decision to lift the freeze for this one file. |
| B3 | `src/strategy/inference/tire_predictor.py` fossil | Re-confirmed (fresh grep): its only real-code consumer is `legacy/experta_engine/rules/degradation_rules.py` — frozen. Moving the fossil INTO `legacy/` would itself be an edit to the frozen zone (adding content there), which this session's explicit scope excludes. Not moved. |
| N4 | `pace_agent.py`'s unreachable LangGraph scaffold | Unchanged from the first-wave finding — still flagged, not deleted, for the "preserved 100%" comment reason already on record above. |

### 5.6 — Clean-code check: large functions decomposed into helpers, on the files this session touched

Explicit ask: does CLEAN_CODE.md's "functions >~30 lines should be orchestrators of small named
helpers" rule hold for the files edited in this session? Checked `_build_tools` (208/168/117 lines
across `pit_strategy_agent.py` / `tire_agent.py` / `race_situation_agent.py` — the largest functions
in the three files this session edited most).

**Verdict: already compliant, not a violation.** Read in full: each `_build_tools` is a container
for 2-3 `@lc_tool`-decorated closures (required to close over `self` for the LangChain tool-calling
framework — a module-level function cannot do this), and every closure's actual logic is already
extracted into small, named, independently-callable instance methods (`agent._validate_driver_on_track`,
`agent._get_driver_stint`, `agent._build_stint_tensor`, etc. — confirmed by reading
`tire_agent.py:1172-1230`). The closures themselves are thin: parse tool args, delegate to a helper,
format the LLM-facing response string. Their size comes from each tool's own docstring, which the
project deliberately writes long and complete because it doubles as the tool's contract with the
LLM (CLEAN_CODE.md's docstring section explicitly asks for this). This matches the earlier informal
audit's own "docstring culture mitigates" judgment — independently re-verified here rather than
taken on faith, on the specific functions this session's edits sit next to.

### 5.7 — Second-wave verification

- `ruff check` + `ruff format --check` on every touched file (3 Python files in `src/arcade/` +
  `src/shared/`, 22 in `tests/`, 8 in `scripts/`, plus the 2 new `tests/` modules): clean.
- Full syntax (`ast.parse`) + import smoke test on every touched module: clean.
- `pytest tests/` (the full suite, all tiers): 554 passed, 4 skipped, 2 pre-existing failures
  unrelated to this branch (see the note above).
- `f1-sim` real run re-confirmed after the `theme.py`/`strategy.py` changes (arcade dashboard code
  is not on the CLI's hot path, but `classify_action`'s home module, `strategy.py`, is imported by
  both surfaces).
- Full `pytest tests/` (all tiers, direct `uv run pytest`, not the `rtk` wrapper — see 5.8): 554
  passed, 4 skipped (environment-gated), 2 failed. Both failures
  (`tests/eval/test_ml_recompute_golden.py::test_pace_mae_reproduces_from_featured_laps`,
  `tests/eval/test_registry_golden.py::test_reproduction_matches_overtake_auc_pr`) reproduce
  identically on a clean `dev` checkout with none of this session's changes applied — confirmed by
  checking out `dev`, stashing everything, and re-running just those two tests
  (`KeyError: ['AirTemp', 'TrackTemp', 'Humidity', 'Rainfall'] not in index`, a pre-existing
  data/schema issue in `tests/eval/`, a directory this session never touched). Pre-existing, not a
  regression from this branch.

### 5.8 — A second adversarial gate, on the second wave

Same discipline as Part 4's gate: an independent Fable agent reviewed this wave's 6 commits before
push. Full report: `documents/audits/AUDIT_cleanup_session_2026-08-01_GATE2.md`. It found real
problems — all fixed before push, git history rewritten (`git reset --soft dev` + a clean re-commit
in dependency order) rather than patched on top, since one of the findings was itself about commit
ordering:

| # | Sev | Finding | Fix |
|---|---|---|---|
| 1 | MEDIUM | The canned-outputs commit imported `tests.conftest` one commit before `tests/conftest.py` existed — bisect-hostile, and the split wasn't disclosed in either commit message. | Whole branch rebuilt from `dev` with commits in dependency order: the two new modules (`conftest.py`, `canned_outputs.py`) and every file that needs both land in one self-contained commit; everything downstream follows. |
| 2 | MEDIUM | A **fifth** `_canned_outputs` consumer was missed in the first pass: `tests/mc/test_tyre_wear_term.py` (two call sites) still imported indirectly via `test_strategy_goldens`. Both fresh `# noqa: F401 -- re-exported for X` comments also named the WRONG consumer (both cited files that had already switched to direct imports). | Pointed `test_tyre_wear_term.py` at `tests.mc.canned_outputs` directly; removed both now-unnecessary `noqa` comments (the import is used locally in both files, so `F401` never fired — the annotation was dead weight with an inaccurate justification). |
| 3 | MEDIUM | `theme.py`'s rewritten comment claimed "Both imported from src.arcade.strategy" — only `classify_action` is; the severity dict was never imported into `theme.py` in this wave. A wrong-mechanism comment, the project's own top-listed bug class, introduced by this session. | Rewrote the comment to describe only what the code actually does. |
| 4 | MEDIUM (pre-existing, live) | `theme.py`'s `_FLAG_BG_BY_INTENT` — a third, independent severity-style mapping (for RCM/radio alert chips) — was missing `YELLOW_FLAG_SECTOR`, the exact key #398 already had to add to `_ALERT_SEVERITY` for the identical reason. Traced as a live path: `rcm_events.py` emits the event type → `radio_agent`'s alert generation → `flag_chip_html` misses the key → renders neutral grey instead of amber. | Added the missing key, with a comment naming both the mechanism and the #398 precedent. |
| 5 | LOW (pre-existing, now fully exposed) | Deleting `severity_color` (its last caller) left `classify_alerts` + `_ALERT_SEVERITY` (`strategy.py`) with zero callers anywhere in the repo — an abandoned `agent_alerts`-banner feature, confirmed dead independently of the caller this wave removed. | Deleted both; this wave's own justification for removing `severity_color` had assumed `classify_alerts` was live, which the gate refuted. |
| 6 | LOW | A 22nd copy of the tire-routing-config guard, in `test_pace_orchestrator_hardening.py`'s fixture body (an imperative `pytest.skip`, not a module-level marker) — missed by the original sweep because it doesn't match the `_HAS_MODELS = ...` textual pattern. | Migrated to `HAS_TIRE_MODELS` from `tests.conftest` like the other 21. |
| 7 | LOW | Several shipped counts didn't reproduce: `tests/conftest.py`'s docstring said "26 copies" (actual 21 at the time, 22 after fix 6); this document said "~28 files" and "four consumers". | Corrected in this document (see 5.1) and in `tests/conftest.py`'s own docstring. |

The gate's own verdict, after re-executing everything above: safe to push once these were fixed; ran
its own full-suite `pytest tests/` invocation independently rather than trusting a cached number (its
run was still in flight when the gate's summary was returned — the 554-passed/2-pre-existing-failed
number above is this session's own separately-run, completed invocation, cross-checked against `dev`
as described).
