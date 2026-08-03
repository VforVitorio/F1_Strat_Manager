# GATE — Final correctness gate before push (#784 epic + #788 fix + partial #789)

**Date:** 2026-08-02 · **Branch:** `refactor/single-source-race-state-builder` (parent, uncommitted) · **Submodule:** `7f394a8` (committed inside, pointer bumped in the parent working tree)
**Role:** final adversarial correctness gate. Success = finding what is STILL broken. No repository file modified except this report, written incrementally.

## Checklist (updated as verified)

- [x] A. CLI edit surgical; radio/RCM main-loop block byte-identical — CONFIRMED (executed)
- [x] B. Three surfaces build identical RaceState; shim IS the canonical function object — CONFIRMED (executed + Qatar gate's cross-surface table)
- [x] C. `reading_or_default` correct + exactly equivalent to pace_agent's old inline guard — CONFIRMED (executed, 8-case table)
- [x] D. tire_agent/no_llm compound/tyre_life canonicalisation changes no previously-sane model input — CONFIRMED SAFE (executed, see verdict)
- [x] E. `startswith('C')` guard still routes "UNKNOWN" correctly; nothing assumed "MEDIUM" — CONFIRMED (executed)
- [x] F. No import cycle; leaf-module guarantee holds — CONFIRMED (executed, both import orders)
- [x] G. Every comment/docstring claim in the diff verified against code and data — mostly TRUE; two false-at-ship comments (F1) + one mechanism nit (F5)
- [x] Tests: coverage of the agent-side changes; wrong-reason passes — coverage gap is F2; no wrong-reason pass found
- [x] Submodule commit still correct after parent-side rename; no old-name references — CONFIRMED (grep at 7f394a8)
- [x] git status: nothing rides along that should not — one unrelated PNG (F6)

## Claim-by-claim verdicts (executed evidence)

**A. CONFIRMED.** `git diff scripts/run_simulation_cli.py` = exactly two hunks (`@@ -1287,18 +1287,14 @@`, `@@ -1306,71 +1302,24 @@`). Byte-compare of `git show HEAD:` vs working tree: every line from old L1400 to EOF is identical modulo the 51-line shift — which covers the radio/RCM block (old 1710-1764) byte for byte. `def _build_race_state(` and the call site (`race_state = _build_race_state(lap_state, args.driver, prev_lap_time)`, old :1711 -> new :1660) are character-identical.

**B. CONFIRMED.** Executed: `backend.utils.race_state_builder.build_race_state IS src.agents.race_state_builder.build_race_state` -> True (parent venv, sys.path = root + src/telemetry), with zero langchain/langgraph/torch modules loaded by the shim import. CLI delegates as `build_race_state(lap_state, driver=driver_code)`; Arcade as `build_race_state(lap_state, risk_tolerance=..., radio_msgs=..., rcm_events=...)` — each surface passes only what it owns. The field-by-field cross-surface agreement on real data is the Qatar gate's executed Task-1 table (this session); every DIFFER there has a named owner.

**C. CONFIRMED.** Property test (executed, this venv): old inline `wx.get(k) if wx.get(k) is not None else d` vs `reading_or_default(wx,k,d)` on {absent, present-None, 0.0, False, True, 25.3, NaN, "25.3"} — identical on ALL eight, including the #633-sensitive falsy cases (0.0/False pass through untouched; the helper tests `is None`, not truthiness). Migrating `pace_agent` changed its behaviour NOT AT ALL, absent key included. The three call sites keep their per-agent defaults (pace 25/35/50; tire + race_situation 28/38/50), which #789 deliberately leaves unreconciled.

**D. CONFIRMED SAFE — this was attacked hardest; the verdict is NOT a regression.**
- The old key-absent defaults (`"MEDIUM"`, `1`) were DEAD on every real path: RSM always emits both keys (`race_state_manager.py:352-355`, compound as `str(...)`, tyre_life as int-or-None) and so does the backend producer (`endpoints/strategy.py:489-491`). The live degraded inputs were the STRING `'nan'`/`'None'` and present-`None` — which the old defaults never caught.
- Compound: executed `_compound_name_to_id` on Miami/Lusail/Spa-Francorchamps/unknown-GP 2025: `'nan'`, `'None'`, `''` and `'UNKNOWN'` ALL map to `'C3'`. So on every real lap where the old code fed the model 'nan', the new code produces the SAME compound id. Zero model-input delta.
- Stint window: featured 2025 carries 379 `Compound=='nan'` rows (Miami L4-24, 19 drivers) + 53 `'None'` rows (Spa L30-44) and EVERY one of them also has TyreLife NaN (executed groupby: `tl_ok=0` everywhere). The `TyreLife <= x` clause therefore empties the window in BOTH worlds (executed on pandas 2.3.3: `Series <= None` -> 0 True rows, no exception; `<= 0` -> 0 rows; NaN satisfies neither) -> old = conservative stub, new = conservative stub. **No lap that previously produced a sane TCN prediction produces a different one.**
- Raw 2025: ZERO NaN-compound laps pass the replay guards (Position + LapTime present; executed over all 24 files) — the replay surfaces never even see the changed path.
- "Is 0 in-distribution for the TCN?" — **0 never reaches the TCN.** The scalar tyre_life only (a) bounds the stint window (`TyreLife <= 0` is empty by construction, season min 1.0, so the tool returns "No laps found" -> conservative stub) and (b) lands on `TireOutput.current_tyre_life`, whose only consumers are display (`agent_formatters.py:150`, `reasoning_tabs.py:121`) and the backend response model `current_tyre_life: int` (`endpoints/strategy.py:141`) — where 0 is valid and the old `None` was a latent Pydantic failure. The TCN's features come from the window ROWS (real TyreLife values), never from the scalar.
- The one reachable behavioural change is an improvement: on the 451 TyreLife-NaN 2025 rows via /recommend and MCP, old no-llm CRASHED (LangChain tool `tyre_life: int` + None -> ValidationError, sweep E1) where new returns the stub; old LLM prompted "tyre life None laps" where new says 0 and lands on the same empty-window stub.

**E. CONFIRMED.** `'UNKNOWN'.startswith('C')` is False -> `_compound_name_to_id` -> `'C3'` (executed, incl. an unknown GP; the fallback dict has no UNKNOWN/NAN keys, `.get(..., 'C3')` covers it). Nothing downstream assumed the old strings: `_get_driver_stint`'s own `'MEDIUM'` fallback fires only when `session_meta` lacks `f'{driver}_compound'`, which `run_from_state` always sets; the orchestrator's synthetic lap_state path is idempotent (`normalise_compound('UNKNOWN') == 'UNKNOWN'`, tyre_life 0 stays 0 — executed). Grep for the old private name `_normalise_compound` across parent src/scripts/tests: zero matches.

**F. CONFIRMED.** Fresh-interpreter import of the builder pulls zero langchain/langgraph/torch/xgboost/lightgbm/transformers/whisper modules (executed; also enforced by the new subprocess test). Both import orders work: builder-first (probe 1) and tire_agent-first (probe 3 — `tire_agent` -> `race_state_builder` at module top completes because the builder never imports agent modules at module scope; `strategy_orchestrator` stays a call-time lazy import). No cycle exists at runtime.

**G. Mostly TRUE; the false ones are F1, the nit is F5.** Re-executed independently this gate: "every 2025 laps parquet ships without weather columns" (featured 48-col + all 24 raw: zero weather cols — TRUE); "all 71 race dirs carry a readable weather.parquet with zero NaN AirTemp/TrackTemp rows" (71/71 readable, non-empty, zero NaN — TRUE); dataset medians (TrackTemp median exactly 35.0, mean 35.3; AirTemp median 24.1, mean 23.9 — TRUE); TyreLife==0 zero rows, 2025 min 1.0 — TRUE. The twin narrative (pace was the only guarded copy, with a trap comment) verified against `git show HEAD:src/agents/pace_agent.py` — TRUE. The 2.3-lap cliff figure is consistently quoted from the Qatar gate's executed measurement (10.5 vs 8.2; post-fix 7.9 vs 8.1) — not re-measured here. One nuance, not false: `_shared_defaults.py`'s "it crashed /recommend ... (#788) via race_situation_agent" is the executed truth of THIS branch's intermediate state; on `main` the same conflict crashes one layer earlier (the backend builder consumer #788's own text names). A parenthetical would make it bulletproof.

## Tests and suites (executed in THIS working tree, all agent-side changes included)

- `tests/agents/test_race_state_builder.py` -> **29 passed** (18.6s).
- `tests/agents/ tests/audit/ tests/simulation/` -> **192 passed** (178.3s) — matches the IMPL baseline+29 with zero new failures, now WITH changes 4/5 present (the IMPL number predates them).
- `tests/engine/ tests/mc/ tests/surfaces/ tests/infra/` -> result appended below; note the IMPL log never re-ran engine/mc after `no_llm.py` changed — this gate closes that hole.
- Wrong-reason scan: the builder tests assert EFFECTS on the constructed RaceState (defaults fire on key-absent states, warnings captured via caplog, explicit 0.0 honoured, rival fallback lands on positional values); the literal-via-constant convention is documented and is a wiring assertion by design, not an empty-set pass. No assertion found that passes on a band that never fires.

## Submodule cross-check

- `git show 7f394a8:backend/utils/race_state_builder.py`: 24-line shim, imports ONLY `build_race_state` (public, exists). Executed: shim object IS the canonical function.
- Grep at `7f394a8` for `_normalise_compound` / `normalise_compound` / `UNKNOWN_TYRE_LIFE` / `_targeting_against_rival` / `_compute_gap_ahead`: the only hit is a prose docstring in `simulator.py:350` accurately describing the absorbed copy. The parent-side `normalise_compound` publication postdates the commit and nothing in the submodule references either spelling. The commit remains correct.
- Submodule working tree: only untracked local dirs (`.claude/`, `docs/migration/streamlit-reference/`) — cannot ride the pointer bump.

## What I tried to break and could NOT

1. **`reading_or_default` vs the old pace inline** — 8-case executed property table including 0.0/False/NaN: byte-equivalent behaviour, no #633 conflation, no change on absent keys.
2. **A real lap whose tyre prediction changes under claim D** — hunted across featured+raw 2025: every `'nan'`/`'None'`-compound row also has TyreLife NaN, both worlds' stint windows come out empty, `_compound_name_to_id` maps old and new degraded spellings to the same `'C3'`, and zero NaN-compound laps pass the replay guards. Could not construct a single regression from shipped data.
3. **The leaf/cycle constraint** — fresh-interpreter import, shim import, and the tire_agent-first order: no langchain/torch leak, no cycle.
4. **The PMV byte-identity** — compared the full tail (old L1400-EOF) programmatically, not just hunk headers.
5. **The doc/data claims** — independently re-executed the 71-file weather scan, the medians, the TyreLife facts and the "every 2025 laps parquet" claim; all held.
6. **`Series <= None` semantics** — pandas 2.3.3 returns all-False, no exception (confirms stub-not-crash equivalence for the old tyre_life=None path).
7. **The orchestrator's synthetic lap_state round-trip** — `normalise_compound` is idempotent on "UNKNOWN" and tyre_life 0 survives unchanged; no double-degradation.

## Could not verify here

- `/recommend` over real HTTP with the LLM profile end-to-end (no OpenAI connectivity in this environment — `APIConnectionError` is environmental; the deterministic profile exercising the exact crash-site code is the IMPL log's executed evidence).
- The 2.3-lap cliff figure itself (accepted the Qatar gate's executed measurement; internally consistent across two documents).
- The submodule suite under a submodule-local venv (the IMPL log's known `fastmcp` skip).

## Findings (appended as confirmed)

<!-- appended incrementally -->

### [MEDIUM / docs-in-code] F1 — Two shipped comments claim "the backend still runs its own copy", in the same tree that bumps the pointer to the shim (EXECUTED evidence: git state)

- `scripts/run_simulation_cli.py:1290-1296` (new comment block): "The telemetry backend still runs its own copy until #786 lands its re-export shim, which needs a submodule commit and a pointer bump; until then this is a two-way unification, not a three-way one."
- `src/arcade/strategy.py:~600-612` (new `_build_race_state` docstring): same claim — "the telemetry backend still runs its own copy until #786 lands its re-export shim ... This is a two-way unification today, not yet the three-way one" — inside a paragraph that itself preaches "a comment claiming a closed drift is worse than no comment".

Both statements were true when #785 was implemented and are FALSE in the tree being pushed: submodule commit `7f394a8` (the re-export shim; verified `git show 7f394a8:backend/utils/race_state_builder.py` — 24-line shim importing the canonical function) is committed inside the submodule AND the parent working tree carries the pointer bump `82f6382 -> 7f394a8` (`git diff src/telemetry`). Executed proof that the claim is false in this tree: `backend.utils.race_state_builder.build_race_state IS src.agents.race_state_builder.build_race_state` -> **True**. Whoever reads the merged code will be told the drift is open when it is closed — the exact claim-vs-tree mismatch class this session already corrected twice (the shim-before-it-existed docstring and the weather-path premise). Milder echo: `src/agents/race_state_builder.py:49` "a re-export shim once #784's submodule half lands" (future tense for a landed thing).

It is also an INTERNAL CONTRADICTION in the CLI file itself: ten lines below that comment, the new `_build_race_state` docstring (`run_simulation_cli.py:1308`) says the delegation is "the single canonical mapping shared with the arcade and the telemetry backend" — the same file asserts both that the backend shares the mapping and that it still runs its own copy. The docstring is the true one at ship time.

Fix: rewrite both to present tense before committing (comment-only edits inside already-touched blocks; the CLI one is inside the comment block this branch already replaced, so the PMV surgical constraint is not re-opened).

### [MEDIUM / test coverage] F2 — The agent-side half of the ship (changes 4 and 5) has ZERO test coverage

Verified by grep over `tests/` (executed): no test imports or exercises `reading_or_default`; no test covers `tire_agent.run_from_state` / `no_llm._tire_no_llm`'s new `normalise_compound` + `UNKNOWN_TYRE_LIFE` derivation; no test feeds any agent a present-`None` weather dict. `tests/agents/test_race_state_builder.py` (29 tests, green in 18.6s, executed) covers the BUILDER surface only — including present-None weather — but the #788 crash site was one layer below the builder, in the agents' RAW-lap_state adapters, and that layer's fix ships untested. Given that this exact bug class regressed once already in this same session ("the fix moved the crash one layer down"), the migrated reads are one refactor away from silently reverting: a test that runs `race_situation_agent.run_from_state`-level session_meta building (or just `reading_or_default` + the three call sites' output on a `{'air_temp': None}` dict) would pin it. Cheap to add; nothing pins it today.

### [RESOLVED — not a finding] F3 — The sweep's R1 crash (TyreLife NaN -> `int()` ValueError) survives by DELIBERATE scoping and IS tracked

`src/agents/race_situation_agent.py:914-915` still crashes on the 451 shipped featured-2025 rows with TyreLife NaN (re-measured this gate), reachable via `/recommend` and MCP. I initially drafted this as "untracked" — WRONG: **issue #790 exists** ("fix(agents): int(NaN) on TyreLife crashes N27's overtake tool on 451 real 2025 laps"), filed from the sweep, correctly citing the guarded twin at `pit_strategy_agent.py:983-999`. Survives the push knowingly and with a paper trail. No action needed.

### [LOW] F4 — `pace_agent` left `rainfall` on the two-arg get its siblings just migrated off

`src/agents/pace_agent.py:650` keeps `_rainfall = wx.get('rainfall', 0)` while the same diff migrates `tire_agent` to `float(reading_or_default(wx, 'rainfall', 0.0))`. Latent-only (executed sweep evidence: no producer emits `rainfall` as None — RSM emits `False`, backend emits int), but it is a fresh inconsistent twin of the exact shape this branch exists to kill. One-line alignment.

### [LOW / nit] F5 — A test docstring misstates the CLI's old crash mechanism

`tests/agents/test_race_state_builder.py:182-185`: "All three pre-#784 builders crashed here via float(None)". The old CLI builder had NO `float()` wrap (`air_temp=weather.get("air_temp", 25.0)`, verified in `git show HEAD`); it crashed via Pydantic float validation rejecting None. Arcade/backend did crash via `float(None)`. 2 of 3; the conclusion (all three crashed) is right, the named mechanism is wrong for one — the same "comment naming the wrong mechanism" class recorded in the errors-gates-caught file.

### [LOW / hygiene] F6 — An unrelated PNG is sitting untracked next to the work

`notebooks/strategy/overtake_probability/outputs/n12b_scoreboard.png` (untracked; the IMPL log itself calls it "an unrelated PNG"). A `git add -A` would ship it inside this refactor PR. Exclude it or commit it separately.

### [LOW / docs] F7 — The shipped SWEEP doc still describes the tyre_life/compound twins as UNFIXED, but this branch fixed them after the sweep ran

`documents/audits/SWEEP_present_none_traps.md` F1/F2 and its closing verdict ("tire_agent.run_from_state still reads tyre_life (:1479) with the two-arg get") describe the pre-change-5 tree; the working tree now derives both via `normalise_compound`/`UNKNOWN_TYRE_LIFE` (verified at `tire_agent.py:1485-1487`, `no_llm.py:139-141`). The IMPL log records the closure, but the SWEEP file itself carries no correction note — unlike DESIGN §F11, which got one when it went stale. A reader grepping the audits later will believe :1479 is still broken. One-line addendum in the SWEEP (F1/F2 tyre_life+compound halves fixed on this branch; R1 -> #790 and R4 remain live) closes it.
