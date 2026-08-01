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
| `radio_msgs`/`rcm_events` | **not populated** (RaceState field left at its Pydantic default) | populated inline from `RadioPipelineRunner` | accepted as parameters, `None` → `[]` |

`compound` and `track_temp` are the two fields where CLI disagrees with BOTH other copies
(`"UNKNOWN"` vs `"MEDIUM"`, `40.0` vs `35.0`), confirming the informal audit's claim with exact
line numbers. `total_laps`'s CLI copy is actually the "more correct" one (fails loudly instead of
guessing), which itself is a divergence in *error-handling philosophy*, not just in literal values.
`radio_msgs`/`rcm_events` is a **new**, previously-unreported divergence this session found: the CLI
path silently produces a `RaceState` with no radio/RCM context at all, while Arcade and the backend
populate it — meaning the CLI's orchestrator input is missing information that the other two
surfaces carry, not just different numbers for the same fields.

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
2. **`radio_msgs`/`rcm_events` in the CLI path** is a functional gap, not just a style one — worth
   flagging to Víctor as its own question: is the CLI's `RaceState` deliberately radio-blind, or is
   this an oversight from before Radio/RCM wiring existed?
2. **`compound`/`track_temp`/`tyre_life`/`total_laps` fallback values** need one canonical answer
   each — this session found the weather-default divergence extends further (`pace_agent.py`'s
   25.0/35.0 is a *fourth* value set for the same physical quantities, Part 2 §N3), so the "pick one
   number" scope is bigger than just the three `_build_race_state` copies.
3. Whichever direction is chosen, it touches the untouchable CLI PMV (`run_simulation_cli.py`) and
   the `src/telemetry` submodule — per the project's issue-first rule for important/risky changes,
   file the issue before writing code, as `project_cleanup_audit_session_plan` already specifies.

---

## Part 4 — Fixes applied in this session

(Appended incrementally as each fix lands — see commit history on `fix/cleanup-anti-slop-scoped`.)
