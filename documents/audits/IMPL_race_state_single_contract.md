# IMPL LOG — Single canonical RaceState builder (#784, parent-repo half)

**Date:** 2026-08-02 · **Branch:** `refactor/single-source-race-state-builder` (off `dev`)
**Scope:** parent repo only. `src/telemetry/` (submodule) untouched by design — its re-export
shim is the separate next step.

Progress is appended as it happens, per the incremental-persistence rule.

## Plan

1. New leaf module `src/agents/race_state_builder.py` (lazy `RaceState` import).
2. Rewire `src/arcade/strategy.py::_build_race_state` → delegation (radio/RCM sourcing stays).
3. Rewire `scripts/run_simulation_cli.py::_build_race_state` → body-only delegation (PMV surgical).
4. `tests/agents/test_race_state_builder.py`.
5. Verify: new tests · suite deltas vs baseline · ruff · real `f1-sim Budapest NOR McLaren`.

## Log

- **[start]** Branch confirmed `refactor/single-source-race-state-builder`; working tree clean
  except the design doc + an unrelated PNG. Baseline
  `uv run pytest tests/agents/ tests/audit/ tests/simulation/ -q` launched in background BEFORE
  any edit, to have a real delta reference.
- Read and confirmed against the design gate: backend builder at
  `src/telemetry/backend/utils/race_state_builder.py` (`_targeting_against_rival` at :7-50,
  builder at :53-120), arcade at `src/arcade/strategy.py:599-715`, CLI at
  `scripts/run_simulation_cli.py:1304-1373` + post-construction radio block :1710-1764.
  CLI's `_GAP_UNKNOWN_FALLBACK_S` import (:1299-1301) is used ONLY inside the body being
  replaced → it and its now-false comment block (:1290-1298) get replaced by the new import,
  not left orphaned.
- **BASELINE:** `uv run pytest tests/agents/ tests/audit/ tests/simulation/ -q` →
  **163 passed**, 169.89s, exit 0.
- **[1. new module]** `src/agents/race_state_builder.py` written. Constants
  (`UNKNOWN_COMPOUND` + marker set, `UNKNOWN_TYRE_LIFE=0`, `DEFAULT_AIR_TEMP_C=25.0`,
  `DEFAULT_TRACK_TEMP_C=35.0`) each carry their F10 measurement in a comment;
  `total_laps` imports `DEFAULT_TOTAL_LAPS` from `_shared_defaults` (57 never restated).
  `_targeting_against_rival` ported VERBATIM (docstring included). Helpers:
  `_car_ahead`, `_gap_to_car_ahead` (two-zeros rationale + honest-2.0 caveat moved here
  from the arcade comment), `_pace_delta_vs_car_ahead` (#750 wording preserved from the
  CLI comment), `_normalise_compound`, `_weather_reading` (present-but-None as missing,
  same pattern as `pace_agent.py:642-643`), `_resolve_lap`, `_resolve_total_laps` (both
  loud on fallback). `RaceState` imported lazily inside `build_race_state`. The
  frozen-RaceState/CLI-mutation coupling is recorded in the module docstring, plus a
  WHERE-TO-CHANGE consumer list.
- **[2. arcade]** `src/arcade/strategy.py::_build_race_state` rewired: keeps the
  radio/RCM sourcing from `self._radio_runner` + the stateful SC re-injection
  (byte-identical logic), then delegates with
  `risk_tolerance=self._request.risk_tolerance` and the two lists as params. The false
  "duplicate ... so the arcade stays independent of backend.utils.race_state_builder"
  docstring replaced with the accurate delegation one; `prev_lap_time` paragraph kept
  as-is (out of scope). The long gap/pace comments moved to the canonical module, not
  deleted. Lazy imports of `GAP_UNKNOWN_FALLBACK_S`/`RaceState` dropped (no longer used).
  **One deliberate ordering note:** the position fail-loud now fires INSIDE the builder,
  i.e. AFTER the radio/SC-tracker sourcing, whereas the old body raised before it. That
  path is only reachable when `_lap_skip_reason`'s invariant already broke (the method's
  own contract), so one extra `sc_tracker.ingest` on an errored lap is the only delta;
  the audit test covering exactly this path still passes (below).
- **[3. CLI]** `scripts/run_simulation_cli.py`: import block :1290-1301 (comment +
  `_GAP_UNKNOWN_FALLBACK_S`) replaced by an accurate comment + `from
  src.agents.race_state_builder import build_race_state  # noqa: E402`; the body of
  `_build_race_state` replaced by `return build_race_state(lap_state, driver=driver_code)`
  with the docstring updated (delegation + where the reasoning lives + the kept
  `prev_lap_time` note). Signature and call site (:1711) untouched. **Surgical-edit
  proof:** `git diff scripts/run_simulation_cli.py` shows exactly TWO hunks
  (`@@ -1287,18 +1287,12 @@` and `@@ -1306,71 +1300,24 @@`); the radio/RCM main-loop
  block 1710-1764 is byte-identical (no hunk touches it).
- **[4. tests]** `tests/agents/test_race_state_builder.py` — 29 tests, two tiers:
  pure-helper + leaf-guard tests run with no models/submodule; `build_race_state` tests
  carry the sibling `data/models/lap_time` skipif (same convention as
  `test_prev_lap_default_is_single_sourced.py`). The leaf test runs a FRESH subprocess
  (`sys.executable -c`) asserting no `langchain*`/`langgraph*` module is loaded by
  importing the builder. Every literal asserted against the module constant, never a
  restated number.

## Verification evidence (all executed)

1. `uv run pytest tests/agents/test_race_state_builder.py -v` → **29 passed** in 12.01s.
2. `uv run pytest tests/agents/ tests/audit/ tests/simulation/ -q` → **192 passed**
   (baseline 163 + 29 new, **zero new failures**), 176.79s.
3. `uvx ruff check src/agents/race_state_builder.py src/arcade/strategy.py
   scripts/run_simulation_cli.py tests/agents/test_race_state_builder.py` →
   "All checks passed!". (`ruff format` deliberately not run on `src/agents/**` — the
   pyproject excludes it from formatting.)
4. **Real PMV run:** `uv run f1-sim Budapest NOR McLaren --no-real-radios --no-llm` →
   exit 0, "**All 70 lap(s) OK**", positions **P5 → P1 (+4)**, actions
   **STAY_OUT·61 / PIT_NOW·5 / UNDERCUT·4**, final stint HARD/40, 1 compound switch,
   wallclock 39.3s (0.6s/lap), best lap 79.918s. No `[ERROR]` rows.
5. **Arcade (no GUI run, architectural evidence instead):**
   `tests/audit/test_engine_scope_defaults.py::test_build_race_state_fails_loudly_instead_of_defaulting_position`
   constructs a real `SimConnector` and calls the rewired `_build_race_state` end-to-end
   (radio sourcing path with runner=None, SC tracker ingest, delegation, ValueError with
   "position" in the message) — green in run 2. Grep confirms no other caller of the
   arcade method and no test matching the old error-message strings.
6. **Extra safety:** `uv run pytest tests/surfaces/ tests/infra/ -q` → **89 passed,
   5 skipped** (missing optional deps + missing Qatar parquets). The PROCESS exits
   -1073740004 (native crash during interpreter teardown, AFTER the full green summary).
   **Verified pre-existing:** stashing only `run_simulation_cli.py` + `strategy.py` and
   re-running reproduces the identical exit code with identical 89 passed/5 skipped —
   unrelated to this change (test_dep_imports imports torch/whisper/onnx; known
   teardown-crash shape on Windows).

## Not done / open

- `src/telemetry/**` untouched (by design): the backend still runs its own copy until
  the submodule re-export commit + pointer bump land. The twin exists knowingly until
  step 3 of the #784 sequence.
- No commit, no push — working tree left dirty for orchestrator review.
- The submodule showed `modified: src/telemetry (untracked content)` at session START —
  pre-existing, not created here, not touched.

---

# Submodule half (#786) — `src/telemetry` (F1_Telemetry_Manager)

Appended by the #786 implementation agent, 2026-08-02. Parent half above untouched.

## What changed (2 files, +35/−168)

1. **`backend/utils/race_state_builder.py` → re-export shim.** The whole local
   implementation (module `_targeting_against_rival` + `build_race_state`) is replaced by
   `from src.agents.race_state_builder import build_race_state` plus a docstring
   explaining the seam. Before deleting, `_targeting_against_rival` was AST-extracted
   from both files and compared: **source-identical, byte for byte** (`IDENTICAL SOURCE:
   True`), so nothing was silently discarded. No other module imports
   `_targeting_against_rival` (grep across the submodule: only its own definition/use).
2. **`services/simulation/simulator.py`** — deleted `_compute_gap_ahead` (the fourth gap
   copy, old lines 245-279) and its now-dead `GAP_UNKNOWN_FALLBACK_S` import (old line
   30, used nowhere else in the file); `_local_build_race_state` stops passing
   `gap_ahead_s` so the canonical builder derives it (None-means-compute), and its
   docstring now says so instead of claiming it computes the gap from the rivals list.
   `pace_delta_s` stays pinned at `0.0` (the #750 neutral) — the canonical builder could
   now derive the rival-relative delta, but that is a real behavioural change to the
   prompt and deliberately NOT taken under #786.

## The `position is None` reachability question — answered

The design gate's claim holds, and the concern is doubly moot:

- `_local_build_race_state` has **exactly one caller** in the entire submodule:
  `simulator.py:877` (grep over `src/telemetry/**`; no test imports it). That caller is
  guarded at `:870` by `_lap_skip_reason`, which returns
  `"incomplete lap (position is None)"` and `continue`s BEFORE the builder.
- Even on a hypothetical breach of that guard, **the old code raised too**: the pre-#786
  submodule `build_race_state` itself raised `ValueError` on `position is None` (old
  lines 98-103). Old flow: `_compute_gap_ahead` returned a silent `0.0`, which was then
  fed into a builder that raised anyway. So "silent 0.0 → raised exception" was never a
  real delta — the 0.0 never survived to a `RaceState`. The per-lap
  `try/except Exception` at `:875/:926` turns it into an SSE `error` event either way.

## Call-site compatibility (the three untouched sites, argument-by-argument)

- `simulator.py:~395` (`_local_build_race_state`): passes `pace_delta_s=0.0`,
  `risk_tolerance`, `rcm_events` as kwargs → all exist on the canonical signature;
  `gap_ahead_s` intentionally dropped (see above); `radio_msgs` omitted → `[]` both
  before and after.
- `api/v1/endpoints/strategy.py:1349` (`/recommend`): passes
  `gap_ahead_s=request.gap_ahead_s`, `pace_delta_s=request.pace_delta_s` —
  `RecommendRequest` declares both as **non-Optional floats with defaults**
  (`strategy.py:110-111`), so they are never `None` and the canonical builder uses them
  unchanged (client-override semantics identical). `risk_tolerance`, `radio_msgs`,
  `rcm_events`, `rival` all map 1:1.
- `mcp_tools.py:595`: `gap_ahead_s=driver_state.get("gap_ahead_s") or
  GAP_UNKNOWN_FALLBACK_S` (never None), `pace_delta_s=0.0`, `radio_msgs=None` /
  `rcm_events=None` → canonical maps None → `[]`, same as the old `or []`. Maps 1:1.

All three call by keyword only; the canonical signature's extra `driver=None` kwarg is
inert for them. **Zero changes needed — none made.**

## Behavioural deltas the backend inherits by adoption (named, intended per #784)

- `compound`: missing-key default `"MEDIUM"` → `"UNKNOWN"`, **plus** normalisation of
  the live `"nan"`/`""`/`"None"` spellings (the delta that fires on real data).
- `tyre_life` missing-key default `1` → `0`; `lap` resolution gains the driver-dict
  fallback + `int()` cast; `total_laps` literal `57` → `DEFAULT_TOTAL_LAPS` (verified
  `== 57`) + warning; present-but-`None` weather values now fall back instead of
  crashing `float(None)`.
- The `ValueError` message for position-None laps changes text (old: "Cannot build
  RaceState: driver position is unknown…"; canonical: "build_race_state: driver position
  is None…(#628, #465)"). `/recommend` returns it inside the 422 detail, so API clients
  see the new wording. No submodule test asserts on the old string (grepped).

## Verification (executed, real outputs)

1. Shim identity, from parent root with `.venv/Scripts/python.exe` (sys.path = parent
   root + `src/telemetry`): `backend.utils.race_state_builder.build_race_state IS
   src.agents.race_state_builder.build_race_state` → **True**; `langchain` NOT in
   `sys.modules` after the shim import (leaf discipline preserved).
2. Submodule suite: `cd src/telemetry && PYTHONPATH="../..;." ../../.venv/Scripts/python.exe
   -m pytest tests/ -q` → **116 passed, 1 skipped, 1 xfailed** in 9.44s. The skip is
   `test_strategy_audit_fixes.py` at collection: `could not import 'fastmcp'` — the
   parent venv does not carry the submodule-only `fastmcp` dep. **Pre-existing, not
   caused by this change** (the skip guard is the module's own `importorskip`, present
   on `main`). Not re-run under a submodule-local venv — noted as the one thing not
   verified here; the submodule's own CI will run it deps-lite (where it skips on
   pandas by design).
3. End-to-end exercise of the absorbed gap logic through the real
   `_local_build_race_state`: car ahead at −1.8 s → `gap_ahead_s=1.8`; interval `None` →
   `GAP_UNKNOWN_FALLBACK_S=2.0` (the #633 distinction survives); leader → `0.0`;
   `pace_delta_s=0.0` pinned. `hasattr(simulator, '_compute_gap_ahead')` → False.
4. Ruff, submodule has no config of its own (no `[tool.ruff]`, no ruff.toml — CI runs
   `--select=E9,F63,F7,F82`): CI-parity select → "All checks passed!"; full default
   select on both touched files → "All checks passed!".

## Not done / open (submodule half)

- Commit made inside the submodule on `main`; **not pushed**; parent pointer NOT bumped
  (orchestrator's step 3).
- The real-run acceptance item (`/simulate` SSE + `/recommend` on Qatar 2025) belongs to
  #787 per the issue text — not run here.
- Untracked `.claude/` and `docs/migration/streamlit-reference/` in the submodule left
  untouched, as instructed.

---

## CLI real runs on the Qatar 2025 reference case (orchestrator, #787 Task 2)

Race directory `data/raw/2025/Lusail`, driver NOR, McLaren. Note the CLI takes the
LOCATION name (`Lusail`), not the menu label (`Qatar`) — `GP_TO_LOCATION` maps the latter
only inside the Arcade/webapp entry points.

| Run | Command | Result |
|---|---|---|
| Deterministic, corpus disabled | `f1-sim Lusail NOR McLaren --no-real-radios --no-llm` | exit 0 · all 57 laps OK · P3 → P4 · STAY_OUT·55 UNDERCUT·2 · 0 errors · 99.2 s |
| Deterministic, **real corpus** | `f1-sim Lusail NOR McLaren --no-llm` | exit 0 · all 57 laps OK · P3 → P4 · STAY_OUT·52 **PIT_NOW·3** UNDERCUT·2 · radio alerts 20 · **radio src corpus/24r·66rcm** · 253.3 s |

The second run is the one that matters, and the first is why. `--no-real-radios` disables the
OpenF1 corpus, which is exactly where lap 7's real `SAFETY CAR DEPLOYED` message lives — so the
first run exercised the builder but NOT the reference case. Re-running with the corpus enabled
ingested 24 radios and 66 RCM events and moved the decision distribution (three `PIT_NOW` laps
appear that the corpus-disabled run never produced). That difference is the evidence that
`radio_msgs`/`rcm_events` still reach the agents after the builder stopped populating them
itself: the parameter-not-internal decision preserved the path end to end.

Verified as NOT a regression from this change:
- `Tire tool output did not parse for C2 (tyre_life=1)` appears 3 times in 57 laps. Traced to
  `src/agents/tire_agent.py:1618-1620`, the pre-existing #436 conservative-stub fallback for a
  tool-output regex miss. Unrelated to the lap_state -> RaceState mapping.

Found while running, NOT caused by this change and NOT fixed here: `f1-sim` exits **0** after
printing `[FATAL] Race directory not found` for an unknown GP. A CI harness checking the exit
code would score that run as a pass. Worth its own issue; deliberately out of this epic's scope.

Lint over every touched parent file (`ruff check src/agents/race_state_builder.py
src/arcade/strategy.py scripts/run_simulation_cli.py tests/agents/test_race_state_builder.py`):
All checks passed. Test suite `tests/agents tests/audit tests/simulation`: 192 passed.

---

## The fix was incomplete: #788 was NOT closed by the builder alone (orchestrator, 2026-08-02)

The Arcade/backend gate proved over real HTTP that `/recommend` still returned **422 on the 2025
reference lap** after the canonical builder landed. The builder no longer crashed; the crash had
moved one layer DOWN, into the agents that read the RAW `lap_state` instead of the `RaceState`:

- `race_situation_agent.py` `run_from_state` built its `session_meta` with `wx.get('air_temp', 28.0)`,
  so the producer's present-`None` flowed to `_compute_weather_features` -> `float(None)` -> TypeError.
  Line 1416 immediately below it WAS guarded, and `pace_agent.py:641-645` guarded the identical read
  with a comment naming this exact crash. One twin fixed, two not.
- `tire_agent.py` did the same, without crashing: the `None`s reached `_add_weather_cols` and the
  TCN's feature frame.

Fixed by lifting `pace_agent`'s working guard into a shared `reading_or_default` helper in
`_shared_defaults.py` and migrating all three agents onto it, pace included — so there is one
implementation rather than one guarded copy plus a helper for the others. The per-caller `default`
argument is deliberate: pace uses 25/35 and tire/race_situation use 28/38, and reconciling those
numbers is a modelling decision tracked in #789, not something to smuggle in behind a crash fix.

A separate gap the sweep found and this session also closed: `tire_agent.run_from_state` and its
twin `src/strategy/inference/no_llm.py::_tire_no_llm` re-derive `compound`/`tyre_life` from the RAW
`lap_state` with the pre-#784 defaults (`"MEDIUM"`, `1`), bypassing the canonical builder entirely.
Both now use `normalise_compound` and `UNKNOWN_TYRE_LIFE` (the normaliser was made public for this).

### Executed evidence

Deterministic no-llm path on the real backend producer's 2025 lap_state (zero LLM calls):

```
PRODUCER weather: {'air_temp': None, 'track_temp': None, 'track_temp_start': None,
                   'humidity': None, 'rainfall': 0}
build_race_state -> MEDIUM 7 25.0 35.0 6.001
run_lap(profile="no-llm") -> STAY_OUT        # previously raised TypeError here
```

Tyre cliff on the same lap, before vs after:

| lap_state weather | cliff P50 before (gate-measured) | cliff P50 after |
|---|---|---|
| `None` (backend producer, every 2025 lap) | 10.5 | 7.9 |
| Real RSM readings 23.4 / 29.5 | 8.2 | 8.1 |

The silent 2.3-lap optimistic error collapses to 0.2 laps. The residual 0.2 is honest — it is the
distance between the declared default (28/38) and the real readings, i.e. the #782 data gap — and
sits inside the TCN's own MC-Dropout variance.

**Not verified here:** `/recommend` over real HTTP end to end. The LLM path needs an OpenAI
connection that this environment does not currently have (`openai.APIConnectionError` on the
ReAct call, an environment condition, not a code path). The deterministic profile exercises the
exact feature-building code that raised, which is the crash site; the HTTP-level 200 should be
re-confirmed once connectivity is available.
