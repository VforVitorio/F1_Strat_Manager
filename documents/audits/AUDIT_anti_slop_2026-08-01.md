# Anti-slop / clean-code audit — 2026-08-01

**Instrument:** background `general-purpose` agent on Fable, launched informally during the
tyre fresh-reference gate session (not the dedicated cleanup session — see
[[project_cleanup_audit_session_plan]] for that). One-shot report, no fixes applied.

**Scope:** all of `src/` except the `src/telemetry/` submodule, plus `scripts/`. Excluded by
instruction: `notebooks/`, `legacy/`, deep `tests/` review. ~43,500 Python lines, AST scan
(duplicate functions, identical bodies, unreferenced definitions, nesting, length) + manual
grep-verification of every candidate against the whole repo (including notebooks, docs,
`documents/`, and the telemetry submodule) to rule out false positives.

**Headline:** the codebase is well above average hygiene. Zero `TODO/FIXME/HACK` in live src,
zero speculative abstractions, disciplined error handling, archived zones documented with a
README + status + successor, and several exemplary anti-twin delegators already in place. The
findings below are real but the background noise is low.

---

## 🔴 HIGH IMPACT

### A1. A third, live copy of the RCM classifier — missing fix #305, already diverged
`scripts/bench_nlp_pipeline_cpu.py:421-517` (`_classify_rcm_event`, 97 lines, self-described as
"direct transcription of N24"). `src/f1_strat_manager/rcm_events.py` exists precisely so this
classifier lives in one place (its own docstring: *"WHERE TO CHANGE IF THE FIA MESSAGE FORMAT
CHANGES: Here, and only here"*, written after #632 measured a 28.2% divergence between two
copies). The bench script violates that contract with a full copy, and it has already diverged:
for `cat == "SafetyCar"`, the canonical classifier treats `"IN THIS LAP"` as
`SAFETY_CAR_ENDING` (fix #305); the bench copy still classifies it as
`SAFETY_CAR_IN_PIT_LANE`, the pre-#305 behaviour. The VSC branch differs too.
**Fix:** replace with the same thin adapter pattern `eval/nlp.py:590` already uses over the same
canonical module. ~95 lines removed.

### A2. Confirmed drift between the two live copies of `_build_race_state`
`src/arcade/strategy.py:599-715` (117 lines) vs `scripts/run_simulation_cli.py:1304-1373` (70
lines) — a third copy exists in the telemetry submodule (out of scope here). Both received fix
#750, but their DEFAULTS have already diverged: `track_temp` fallback 40.0 (CLI) vs 35.0
(arcade); `compound` fallback `"UNKNOWN"` vs `"MEDIUM"`; `tyre_life` fallback 0 vs 1;
`total_laps` raises KeyError in one, defaults to 57 in the other. The ~14-line comment
explaining #750 is pasted verbatim in both, so it will drift too.
**Caveat:** the CLI copy is inside the untouchable PMV (`run_simulation_cli.py`) — this needs an
issue to decide where the single source lives (the `strategy_pipeline.py` pattern already shows
how), not an in-place edit.

### A3. `theme.py` deduplicated `_ALERT_SEVERITY` after #620 burned it — but kept `_ACTION_STYLE`'s twin 20 lines away
`src/arcade/dashboard/theme.py:59-84` vs `src/arcade/strategy.py:721-734`. The file's own
comment (lines 70-79) tells the #620 story: a hand-copied `_ALERT_SEVERITY` drifted the moment
#398 added `YELLOW_FLAG_SECTOR` to the original, because a "mirrors X" comment told readers the
two dicts were already checked — and the fix was to import it. `_ACTION_STYLE` +
`classify_action` (lines 59-68, 82-84) are still that exact same hand-mirrored pattern, with the
same "mirrors src/arcade/strategy.py::classify_action" comment the file itself names as the bug
mechanism. `theme.py:22` already imports from `strategy.py`, so the isolation argument is moot.
**Fix:** import instead of mirror. ~26 lines, closes #620 properly.

---

## 🟠 MEDIUM IMPACT

### M1. Verified dead code (zero references across src+scripts+tests+notebooks+docs+documents+telemetry)
| What | Where | Detail |
|---|---|---|
| `severity_color()` | `arcade/dashboard/theme.py:87-92` | 0 callers, and restates the mapping `strategy.py::classify_alerts:760` already has — dead AND a twin. |
| `_CLUSTER_PARQUET`, `_LAPS_FEATURED`, `_FEATURE_MANIFEST` | `agents/pace_agent.py:88-90` | Defined, never read. `_load_encoding_maps` (:206-218) builds the same paths inline, so "fixing" the constant changes nothing. |
| `get_pace_react_agent` (×2), `get_tire_react_agent`, `get_race_situation_react_agent`, `get_pit_strategy_react_agent` | pace/tire/race_situation/pit_strategy agents | ~110 lines total. Docstrings say "created only when N31 or tests actually invoke the agent" — none do (contrast `get_rag_react_agent`, which IS invoked). If kept as an announced public API, fix the false docstring instead. |
| `simulate_real_time_predictions` | `strategy/inference/tire_predictor.py:885` | 139 lines, dead even inside its own fossil file (see B3). |

### M2. Restated constants — the same bug class #765/#772 just fixed in the prompts, still live in code
- `total_laps` fallback `57` hardcoded in 7 live sites (`arcade/strategy.py:703`, `pit_strategy_agent.py:1039,1273,1456`, `race_situation_agent.py:1010,1362`, `tire_agent.py:1481`, +2 in telemetry).
- Weather defaults `25.0`/`35.0` in 5 live sites, with a 6th already diverged to `40.0` (CLI:1371) — proof the bug class is real, not theoretical.
- The tire agent's conservative stub (`deg_rate=0.03, p10=20.0, p50=30.0, p90=40.0`) is duplicated inside the SAME function (`tire_agent.py::_run_core`, lines 1566-1578 and 1609-1621).
- Compound RGB tuples written literally twice in the same file (`theme.py`: `COMPOUND_COLORS` lines 39-45 and `_COMPOUND_COLOUR_BY_LABEL` lines 123-142) instead of deriving the second from the first.
- Already-documented-deliberate, not counted as slop: `FUEL_GAIN_PER_LAP_S` duplication (#446 deferral note), the 30-200s sane-lap-time band in two chart widgets (documented "independently substitutable widgets" rationale — judged weak but is a recorded decision, not an oversight).

### M3. Documentation drift inside CLAUDE.md itself (the source of truth)
- §5/§9 still document `f1-streamlit`, retired in #551 for `f1-webapp`; no streamlit entry point in `pyproject.toml` anymore.
- §3 describes `src/shared/` as the live extraction home, when `src/shared/README.md` says it is archived and the canonical extractors live in `src/data_extraction/` (not listed in §3's tree).

### M4. Duplicated bootstrap preamble across ~15 scripts, 3+ divergent variants
Repo-root discovery + `sys.path.insert` + dotenv + `reconfigure(utf-8)`, repeated in `f1_cli.py`, `run_simulation_cli.py`, `debug_agent.py`, four `bench_*.py`, `build_radio_dataset.py`, `upload_radio_corpus.py`, `verify_drs_zones.py`, three `measure_*.py`, `download_data.py`, `prompt_ab/_common.py`. Root-resolution logic already disagrees between variants (`.git`-search-with-fallback vs `parents[1]` vs `parent.parent`) — a `uv tool install` without `.git` would resolve different roots. `scripts/bench/_common.py` and `scripts/prompt_ab/_common.py` already show the shared-`_common` pattern is idiomatic here; a root-level one is missing. (`run_simulation_cli.py` stays out — PMV.)

---

## 🟡 LOW IMPACT

- **B1.** `src/vision/gap_calculation.py` (803 lines) is a byte-identical copy of `legacy/app_streamlit_v1/.../gap_calculation.py`, the only one anything imports. Deliberate (documented README), but its own justification ("keeps git history self-contained") doesn't survive git history actually being self-contained. Safe prune: 803 lines.
- **B2.** False docstring in `shared/data_extraction/fastf1_extractor.py:6-7`: claims N01 "still imports it" — verified N01 only link-references it in markdown, no code cell imports it.
- **B3.** `strategy/inference/tire_predictor.py` (1,023 lines) is a jupytext N09-era fossil living inside the live inference package next to `engine.py`, documented as "reference only," sole consumer is `legacy/experta_engine/`. Belongs in `legacy/`, not `strategy/inference/`.
- **B4.** Large-function inventory (not a mandate): `run()` 717 lines (PMV, untouchable), `_build_orchestrator_prompt` 256 / `_run_projection_mc` 216 / `_run_mc_simulation` 156 / `_assemble_recommendation` 146 (strategy_orchestrator), `_build_tools` 208/168/117 (pit/tire/situation agents), `_arrow_pick` 122 lines at nesting 6 (`scripts/cli/pickers.py:59` — the cleanest decomposition candidate outside frozen zones).
- **B5.** Minor twins: `_bar_style` (Qt progress gradient) duplicated in `orchestrator_card.py:228` and `scenario_bars.py:145`; `measure_fresh_reference_gate.py` vs `_2025.py` share near-identical `scored_bound`/`load_bundles` (tolerable — deliberately frozen reproducibility scripts); `_load_ner_model` duplication (radio_agent/eval-nlp) already root-caused to open issue #167 — the real fix is #167, not the dedup.

---

## What was checked and NOT flagged (avoid re-auditing this)

- **Error handling:** no bare `except` in live code (the only one is in an archived file), no unexplained `except: pass`, specific types with context comments throughout.
- **Correct anti-twin delegators already in place:** `strategy_orchestrator._scope_laps_to_gp`, `arcade/strategy_pipeline.py` (killed a body-copy #166 proved dangerous), `GAP_UNKNOWN_FALLBACK_S` single-sourced.
- **False positives ruled out after reading:** `_to_seconds` (df-column vs scalar, genuinely different), `_parse_tool_outputs` (different tool formats), `_load_encoding_maps` (different artefacts per model), six `_render_table` variants in eval (different tables), `_run_core`/`run_from_state` scaffolding across four agents (parallel structure, genuinely different domain per agent — abstracting it would touch a near-frozen zone for less than it risks).
- **Archived zones** (`src/shared/`, `src/vision/`, `src/data_extraction/{legacy,fastf1}`, `intervals_extractor`): all README-documented with status + successor, already covered by prior audits (A5, P5) — deliberate, not slop.
- **Speculative abstractions:** zero found. **Naming:** consistently good. **Orphan scripts:** none — bench/measure/debug/verify are deliberate evidence tooling.

---

## Summary table

| Category | Cases | Estimated impact |
|---|---|---|
| Diverged twin with a lost fix (RCM bench, ACTION_STYLE, `_build_race_state` defaults) | 3 clusters | **High** — the project's dominant bug class, already measurably diverged |
| Verified dead code | 9 symbols (~260 lines) | Medium — safe direct prune |
| Restated constants (57 ×7, weather ×6 with 1 diverged, tire stub ×2, RGB ×2) | 4 families, ~17 sites | Medium — same bug class as #765/#772 |
| CLAUDE.md documentation drift | 3 points | Medium — it's the source of truth |
| Duplicated bootstrap preamble across scripts | ~15 files, 3 variants | Medium-low — real divergence risk on installs without `.git` |
| Misplaced fossils / false docstring / byte-identical legacy copy | 3 | Low — documented, prune optional |
| Functions >100 lines / nesting >2 outside frozen zones | ~8 relevant | Low — docstring culture mitigates; `_arrow_pick` best decomposition candidate |
| **Total prunable outside untouchable zones** | | **≈1,300-1,500 lines** (incl. the two fossils) or **≈400 lines** if only live dead/duplicated code is counted |

**Recommended first three moves, by value/risk:** (1) thin adapter over `classify_rcm_event` in
`bench_nlp_pipeline_cpu.py` — ~30 min, kills A1; (2) import `_ACTION_STYLE`/`classify_action` in
`theme.py`, delete `severity_color` — ~15 min, properly closes #620; (3) file an issue to unify
`_build_race_state`'s defaults into one source (touches the PMV and the submodule, so it earns
its own issue before any code, per the project's issue-first rule).
