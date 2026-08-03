# DESIGN GATE — Single canonical `RaceState` builder (options a/b/c)

**Date:** 2026-08-02
**Role:** adversarial design gate, read-only. No implementation code. Success = finding what is
still broken in the proposal, not approving it.
**Scope:** the three (or four) independent builders of `src.agents.strategy_orchestrator.RaceState`
from the `lap_state` contract, and the decision between options (a) shim-down import,
(b) shared module in `src/`, (c) parity test over accepted copies.

## Verification checklist (updated as each item is confirmed/refuted)

- [x] V1. CLI `_build_race_state` — CONFIRMED, all defaults as briefed (F1)
- [x] V2. Arcade `_build_race_state` — CONFIRMED, plus extra divergences the brief missed (F1)
- [x] V3. Backend `build_race_state` — CONFIRMED (F1)
- [x] V4. `GAP_UNKNOWN_FALLBACK_S` shared import — CONFIRMED in all three (F1)
- [x] V5. CLI post-construction radio/rcm mutation + `--radio-every` — CONFIRMED (F3)
- [x] V6. `_local_build_race_state` is a thin wrapper — CONFIRMED (F3)
- [x] V7. `_compute_gap_ahead` is a genuine 4th copy — CONFIRMED (F7)
- [x] V8. `compound` plain str; "UNKNOWN" traced through all consumers, never crashes (F5)
- [x] V9. `total_laps` present from every internal producer; the ONE reachable hole is client JSON at /recommend (F6)
- [x] V10. Shim direction — CONFIRMED: backend reaches UP (bare-metal shim + Docker `../../src` mount); reverse NOT wired; zero parent-side `backend` imports (F8)
- [x] V11. "Independence" is a REAL, load-bearing install contract, not historical caution (F8)
- [x] V12. CLI imports RaceState natively at `scripts/run_simulation_cli.py:158` — CONFIRMED
- [x] V13. `_targeting_against_rival` is a pure function; composes as an optional param (F2)
- [x] V14. TyreLife: measured on the real parquets — 0 NEVER occurs, 1 does (F10)

## Findings

### F1. The three builders and their literal divergences — VERIFIED (V1, V2, V3 ✔)

All three re-read 2026-08-02. The brief's table is accurate, and it is INCOMPLETE — there are
more divergences than the four it lists.

| Field | CLI `scripts/run_simulation_cli.py:1361-1373` | Arcade `src/arcade/strategy.py:699-715` | Backend `src/telemetry/backend/utils/race_state_builder.py:105-120` |
|---|---|---|---|
| `driver` | `driver_code` param (line 1362) | `driver_st.get("driver", "UNK")` (700) | `drv.get("driver", "UNK")` (106) |
| `lap` | `driver_st["lap_number"]` — from DRIVER dict, direct index (1363) | `int(lap_state.get("lap_number", 1) or 1)` — from TOP-LEVEL dict, default 1 (677, 701) | `lap_state.get("lap_number", 1)` — top-level, default 1, no int cast (107) |
| `total_laps` | `lap_state["session_meta"]["total_laps"]` direct index → KeyError if absent (1364) | `meta.get("total_laps", 57)` (703) | `meta.get("total_laps", 57)` (108) |
| `compound` | default `"UNKNOWN"` (1366) | default `"MEDIUM"` (705) | default `"MEDIUM"` (110) |
| `tyre_life` | default `0` (1367) | default `1` (706) | default `1` (111) |
| `air_temp` | `25.0` (1370) | `25.0` (709) | `25.0` (114) |
| `track_temp` | `40.0` (1371) | `35.0` (710) | `35.0` (115) |
| `position is None` | raise ValueError "#628" (1332-1337) | raise ValueError "#465" (632-636) | raise ValueError, no issue number in msg, #465 in comment (98-103) |
| `gap_ahead_s` | no-car-ahead → 0.0; unknown interval → `GAP_UNKNOWN_FALLBACK_S` (1338-1343) | same shape (652-659) | value comes in as PARAMETER `gap_ahead_s: float = GAP_UNKNOWN_FALLBACK_S` (56) — the positional-car-ahead computation lives in the CALLER (`simulator.py::_compute_gap_ahead`) |
| `pace_delta_s` | computed inline vs car ahead, #750 (1355-1359) | computed inline vs car ahead, #750 (671-675) | PARAMETER, default 0.0 (57); rival-targeted recompute if `rival` set (81-87) |
| `radio_msgs`/`rcm_events` | NOT passed to constructor — schema defaults, main loop mutates after (see F3) | built INSIDE from `self._radio_runner`/`self._sc_tracker` (678-697) | optional params, `None → []` (59-60, 117-118) |
| `risk_tolerance` | NOT passed (schema default) | `float(self._request.risk_tolerance)` (714) | param, default 0.5 (58, 119) |
| SC re-injection (`should_inject`) | in the CLI main loop, not the builder | INSIDE the builder (695-697) | in the CALLER (`simulator.py::_rcm_events_for_lap` at ~357) |

**Divergences the brief missed:** (i) the `lap` field has THREE different sources/behaviors —
CLI reads `driver_st["lap_number"]` (fail-loud, from the driver dict), Arcade/backend read
top-level `lap_state["lap_number"]` with default 1. A missing top-level `lap_number` would
silently build "lap 1" state in Arcade/backend and crash the CLI. (ii) `driver` default
"UNK" exists only in Arcade/backend. (iii) `risk_tolerance` plumbing differs in all three.
(iv) The SC-tracker injection sits at a different layer in each surface (builder vs caller vs
main loop). Any canonical builder has to decide (i)-(iv) too, not just the four fields flagged.

**GAP_UNKNOWN_FALLBACK_S (V4 ✔):** all three import it from `src.agents.position_projection` —
CLI at `scripts/run_simulation_cli.py:1299-1301`, Arcade at `src/arcade/strategy.py:615`,
backend at `src/telemetry/backend/utils/race_state_builder.py:3`. So the backend ALREADY
imports up into the parent repo's `src/agents` from this exact module — the (b) mechanism is
proven at the exact file that would consume it.

### F2. `_targeting_against_rival` (#431) — VERIFIED (V13 partially, see F8)

`src/telemetry/backend/utils/race_state_builder.py:7-50`. Pure function of
`(lap_state, rival, fallback_gap_s, fallback_pace_s)` → `(gap, pace)`. It contains ZERO
webapp-specific machinery — no FastAPI, no request objects; "webapp-only" is about who calls
it, not what it needs. It composes cleanly as an optional `rival: Optional[str]` parameter on
a canonical builder (exactly as `build_race_state` already exposes it at line 61).

### F3. radio_msgs / rcm_events placement — VERIFIED (V5, V6 ✔)

- **CLI**: builder returns them as schema defaults (empty lists); the main loop mutates the
  built object at `scripts/run_simulation_cli.py:1744-1762` — `.extend()` of corpus radios
  (1745-1748), SC-tracker re-injection (1756-1758), and the CLI-ONLY synthetic
  `--radio-every` generator `_generate_radio_event`/`_generate_rcm_event` (1726-1737,
  appended 1759-1762). Precedence rule at 1714-1720: corpus suppresses synthetic entirely.
- **Arcade**: INSIDE the builder from instance state (`self._radio_runner` at
  `src/arcade/strategy.py:680-688`, `self._sc_tracker` ingest+inject at 695-697).
- **Backend**: optional keyword params (`race_state_builder.py:59-60`), populated by callers:
  - `simulator.py::_local_build_race_state` (378-406) is CONFIRMED a thin wrapper over
    `build_race_state` — passes `rcm_events` only, `radio_msgs` never (the SSE stream is
    RCM-only by design, `_build_rcm_feed` docstring at 330-335); SC injection lives in its
    caller-side helper `_rcm_events_for_lap` (357-375).
  - `backend/api/v1/endpoints/strategy.py:1349-1357` passes `request.radio_msgs` +
    `rcm_events` (with an RCM-autoload fallback at 1332-1336) + `rival=request.rival`.
  - `backend/mcp_tools.py:595-604` passes `radio_msgs=None, rcm_events=None`.

Three surfaces, three different SOURCES for the same two fields (corpus+synthetic via
mutation / instance state / caller params). The parameter shape is the only one of the three
that all others can be expressed in.

### F4. Backend call sites of `build_race_state` — VERIFIED

Exactly three: `simulator.py:400`, `endpoints/strategy.py:1349`, `mcp_tools.py:595`. All
already use keyword-only params. A relocation of the canonical function changes ONE import
line per site (or zero, if `backend/utils/race_state_builder.py` becomes a re-export shim).

### F5. What `"UNKNOWN"` compound actually does downstream — VERIFIED (V8 ✔)

`RaceState.compound` is a plain `str` (`src/agents/strategy_orchestrator.py:253`); the
`Literal["SOFT","MEDIUM","HARD"]` at :271 constrains only OUTPUT fields (`compound_next`).
Traced every consumer of `race_state.compound`:

1. **Pace agent (N06)** — `strategy_orchestrator.py:1871` → `pace_agent.py:261`
   `self.compound_id.get(compound, 1)`: unknown string silently encodes as id 1 ("most
   common training value" per its docstring at 250-251). No crash, no log.
2. **Tire agent (N26)** — via the synthetic `lap_state["driver"]["compound"]` at
   `strategy_orchestrator.py:2435` → `tire_agent.py:785-811` `_compound_name_to_id`:
   "UNKNOWN" is not in the map → falls back to `'C3'` (:804/:811); then
   `_add_compound_cols` (:667-687) LOGS A WARNING (:681-683) and encodes C3/SOFT defaults.
   No crash. `'C3'` is a real routing_config key, so bundle lookup survives.
3. **Pit agent (N28)** — `pit_strategy_agent.py:394-412` `_compound_to_id("UNKNOWN")` → `''`
   from the JSON map → `_COMPOUND_FALLBACK.get("UNKNOWN", 3)` → 3. Additionally
   `_N15_COMPOUND_ORDER` (:117-120) maps unknowns to `-1`, which IS the notebook's own
   trained `.fillna(-1)` bucket — for N15 specifically, "UNKNOWN" is in-distribution.
4. **RAG question** — `strategy_orchestrator.py:2028` → `:1496-1514`: free text ("changing
   to UNKNOWN compound"); degraded question, no crash.
5. **LLM prompt** — `:1734` renders `Compound: UNKNOWN` — honest to the model.

**Conclusion:** "UNKNOWN" never crashes; every numeric consumer degrades to a mid-range
encoding (roughly what "MEDIUM" would produce anyway), one path warns loudly, and the LLM
sees the truth instead of a fabricated MEDIUM. "MEDIUM" produces nearly the same numbers
while asserting knowledge that does not exist — and MEDIUM is a value the code CAN
legitimately find, which is precisely the project's sentinel-collision doctrine violation.

**However — the default is nearly DEAD on the RSM path** (see F6): `RaceStateManager`
emits `"compound": str(r.get("Compound", ""))` (`race_state_manager.py:352`) — the key is
ALWAYS present. When FastF1 has no compound the value is `""` or `"nan"`, and `.get`'s
default never fires (the very trap in CLAUDE.md §11, 2026-07-16, `Series.get`). So the
real-world "unknown compound" string on the RSM path is `""`/`"nan"`, not any builder
default. The canonical builder should therefore normalise falsy/`"nan"` → the chosen
default, or the chosen literal is a decision about an almost-unreachable branch.

### F6. `session_meta` producers and `total_laps` — VERIFIED (V9 ✔, with one real hole)

Producers found (grep `"session_meta"` across all .py):
- `src/simulation/race_state_manager.py:518` — `"total_laps": self.total_laps`,
  unconditional; `self.total_laps = int(enriched["LapNumber"].max())` at :130. ✔ always present.
- `backend/api/v1/endpoints/strategy.py:592-598` (the /lap-state family) — includes
  `total_laps` (:597, from `int(gp_df["LapNumber"].max())` at :585). ✔
- `backend/api/v1/endpoints/strategy.py:942-948` (the single-agent lap_state producer) —
  includes `total_laps` (:947). ✔
- `strategy_orchestrator.py:2448-2454` (run_lap's internal adapter) — includes
  `total_laps` from `race_state.total_laps`. ✔
- **The hole:** `/recommend` (`endpoints/strategy.py:1349`) builds from
  `request.lap_state` — ARBITRARY CLIENT JSON over HTTP. A client may legally omit
  `session_meta` entirely (`race_state_builder.py:79` guards with `.get("session_meta", {})`).
  This is the one reachable path where the CLI's fail-loud direct index would raise and the
  backend's `.get(..., 57)` actually fires. So: fail-loud is safe for CLI/Arcade (RSM-fed),
  but the canonical builder must keep SOME defined behavior for the HTTP boundary.
- Precedent: `src/agents/_shared_defaults.py:19` already defines `DEFAULT_TOTAL_LAPS = 57`
  with a documented rationale (median/mode of the 71-race 2023-2025 dataset), consolidated
  from six agent-side call sites. A canonical builder that defaults should import THIS, not
  restate 57.

### F7 (early). `simulator.py::_compute_gap_ahead` IS a genuine 4th copy — VERIFIED (V7 ✔)

`src/telemetry/backend/services/simulation/simulator.py:245-279`. Same algorithm as the two
inline copies: `next(r for r in rivals if r.get("position") == our_pos - 1)`, then
`interval None → GAP_UNKNOWN_FALLBACK_S`, `abs(interval)` otherwise, 0.0 when no car ahead.
Differences from CLI/Arcade: it returns 0.0 (instead of raising) on `our_pos is None`
(264-265) — safe only because `_lap_skip_reason` (296-321) filters those laps first, as its
own docstring admits at 257-259. Its docstring at 248-250 states the duplication is
INTENTIONAL ("kept inline here because the two copies are short and intentionally
decoupled") — that rationale is exactly what this design gate is deciding to overturn, so
unification should absorb it (analysis in F9).

### F8. The layering evidence: why the "independence" framing is load-bearing — VERIFIED (V10, V11 ✔)

- **Backend → parent is the ONLY existing direction.** Bare-metal: `mcp_tools.py:33`,
  `simulator.py:45`, `endpoints/strategy.py:33` all `sys.path.insert(0, get_repo_root())`,
  and `backend/core/paths.py:27-41` walks up for a `.git` DIRECTORY (skipping the submodule
  gitlink file — the #27 fix), landing on the parent root. Docker:
  `src/telemetry/docker-compose.yml:16` mounts `../../src:/app/src:ro` and paths.py falls
  back to `/app` (:24, :41) — so `src.agents.*` resolves inside the container too. The
  backend depends on the parent's `src/` tree in BOTH deployment modes, today, by design.
- **Parent → backend does not exist anywhere.** Grep over `scripts/`, `src/agents/`,
  `src/arcade/`, `src/simulation/`, `src/strategy/`, `src/f1_strat_manager/`, `tests/` for
  `from backend.` / `import backend` / `telemetry.backend`: **zero matches**.
- **The install contract makes that a feature, not an accident.** `INSTALL.md:37-59`
  installs CLI and Arcade via `uv tool install git+...` or `uv sync` on a source checkout,
  with NO submodule step; `--recurse-submodules` is documented as required ONLY for the
  webapp flow (`INSTALL.md:89-95`). There is no `src/telemetry/__init__.py` and no
  `src/__init__.py` (checked on disk), so nothing guarantees the backend package chain is
  even importable from the parent side without a NEW mirrored sys.path shim onto
  `src/telemetry` (backend modules import each other as top-level `backend.*`, e.g.
  `endpoints/strategy.py:1323`). The arcade docstring's rejection of that shim
  (`src/arcade/strategy.py:600-606`) reflects this real invariant.
- Parent CI checks out submodules (`.github/workflows/ci.yml:52-54`), so parent-side tests
  CAN see backend files — relevant to option (c) — but a user checkout is not CI.

### F9. `_compute_gap_ahead` absorption — YES, with one behavioral note

Deleting it and letting the canonical builder compute the gap internally is safe IF the
canonical API treats "caller did not supply a gap" as "compute from rivals":
- `simulator.py:877-879` calls `_local_build_race_state` only after `_lap_skip_reason`
  (:870-873) has excluded `position is None` laps (:315-316), so `_compute_gap_ahead`'s
  lenient `return 0.0` on None position (:264-265) is unreachable on the live path — the
  canonical builder's fail-loud ValueError is behavior-identical where it matters.
- Only one caller exists (`simulator.py:402`); no test or other module references it.
- NOT the same thing: `endpoints/strategy.py:472-476` computes a gap from cumulative
  session times over the DataFrame — that is a lap_state PRODUCER computation (different
  inputs, before rivals exist as dicts), a cousin, not a fifth copy. Leave it out of scope,
  but note it in the issue so nobody "unifies" it into the wrong layer later.

### F10. Executed evidence for the literal recommendations

Measured directly on the shipped parquets (2026-08-02):

```
TyreLife  2023 min 2.0 · 2024 min 2.0 · 2025 min 1.0 · rows with TyreLife==0: ZERO in all seasons
Compound  values found: HARD/MEDIUM/SOFT/INTERMEDIATE/WET + the STRINGS 'None' (2023, 2025)
          and 'nan' (2025); 'UNKNOWN': ZERO rows in all three seasons
TrackTemp (2023+2024; the 2025 featured parquet carries no weather cols)
          mean 35.3 · median 35.0 · p10 24.5 · p90 46.5
AirTemp   mean 23.9 · median 24.1
```

Consequences:
- `tyre_life=0` is a value the data can NEVER legitimately contain; `tyre_life=1` collides
  with real fresh-tyre laps (2025 has them). Doctrine picks 0.
- `compound="UNKNOWN"` collides with nothing in this corpus (0 rows; FastF1's API does
  define UNKNOWN as a possible label, but the 2023-2025 data never emits it — flag, not
  blocker). Meanwhile the REAL missing-compound strings that already flow out of
  `race_state_manager.py:352` (`str()` of NaN/None) are `'nan'`/`'None'`/`''` — none of the
  three builders normalises them, so today an incomplete row reaches the models as the
  literal string `"nan"` regardless of which default was chosen.
- `track_temp=35.0` IS the dataset median; the CLI's 40.0 corresponds to nothing measured.
- `air_temp=25.0` ≈ median 24.1; all three already agree. Keep.

### F11. Which defaults are actually live vs nearly dead (the honest stakes)

> **CORRECTION (2026-08-02, from the follow-up gate in `GATE_weather_path_findings.md`).**
> The claim below that the weather defaults are a "genuinely LIVE divergence" because they fire
> "when `weather.parquet` is missing for a race" is **FALSE**. An executed scan found all 71 race
> directories carry a readable `weather.parquet` with zero NaN `AirTemp`/`TrackTemp` rows, so on
> the replay path (CLI and Arcade, both via `RaceReplayEngine`, which passes the frame at
> `replay_engine.py:137`) that branch is unreachable with the shipped data. This is the same
> mistake `CLAUDE.md` §11 records about the Arcade's temperatures on 2026-07-16.
>
> The **decision** (canonical 35.0) still stands, on the measured-median argument alone: 35.0 is
> the dataset median and the CLI's 40.0 corresponds to nothing measured. Only this section's
> justification was wrong.
>
> What the same gate found instead IS live, and is bigger: the backend's `lap_state` producer
> emits `None` per weather key on 2025 laps (those parquets carry no weather columns), which the
> old consumer passed to `float()` — a `TypeError` caught as a 422 on **every** 2025 lap of
> `/recommend`. Filed as #788; fixed by this change's present-but-`None` handling.


`dict.get(key, default)` fires only on a MISSING KEY (CLAUDE.md §11 2026-07-16 lesson).
Checked against every producer:
- `compound` / `tyre_life` / `position` / `driver` / `lap_number`: RSM ALWAYS emits the keys
  (`race_state_manager.py:316-374, 596`), backend producers too (`endpoints/strategy.py:479,
  588, 905-906, 945`). On RSM-fed paths the compound/tyre_life defaults are DEAD; the live
  hazard is present-with-None / string-coerced NaN (F10). The defaults only breathe on
  hand-built lap_states: HTTP clients of `/recommend` and test fixtures.
- `total_laps`: same, except the /recommend client-JSON hole is real (F6).
- **weather is the exception — genuinely LIVE divergence:** `get_weather_state` with no
  `weather_df` returns ONLY `{"track_status": ...}` (`race_state_manager.py:470-472`), so
  when `weather.parquet` is missing for a race, the KEYS are absent and every default
  fires. On that same race the CLI feeds N14 `track_temp=40.0` and Arcade/backend feed
  `35.0` — a real, reachable, surface-dependent model-input divergence today.
- Present-with-None weather values (NaN row → None at :478-479) crash all three builders
  (Pydantic float rejects None; arcade/backend `float(None)` TypeError) — caught by per-lap
  try/except on CLI/arcade, a 422/500 on the backend. The canonical builder should treat
  None-as-missing for weather (an `x if x is not None else DEFAULT` read), a strict
  robustness gain, behavior-identical on every lap that works today.

### F12. Option (a) — resolve the shim downward: REJECTED

Three independent kills, each sufficient:
1. **It breaks a documented install contract.** f1-sim/f1-arcade run today from a checkout
   with `src/telemetry` EMPTY (INSTALL.md:37-59 vs :89-95; zero parent→backend imports,
   F8). Under (a) the PMV would ImportError on every clone made without
   `--recurse-submodules` and every `uv tool install` where the submodule content is
   absent or unpackaged (no `__init__.py` chain, F8).
2. **Layering inversion.** The parent's UNTOUCHABLE PMV would depend on a submodule pointer
   versioned in another repo; a backend refactor could break the CLI, and fixing it would
   require the commit-then-bump dance for what is today a parent-only file.
3. **Import-name schizophrenia.** Backend code is imported as top-level `backend.*` (its
   internal absolute imports demand it). The parent would need a mirrored
   `sys.path.insert(src/telemetry)`; the same file could then be materialised twice under
   two module names in one process — double side effects, the exact hazard class the
   arcade docstring rejected (`src/arcade/strategy.py:600-606`).

### F13. Option (c) — accept the copies + parity test: REJECTED (fallback only)

- A parity test in the parent can only run when the submodule is checked out; on a
  contributor clone without it, it silently skips — and in the SUBMODULE's own repo/CI
  (where backend edits actually happen, per the commit-first rule) the parent-side test
  does not exist at all. The guard is absent precisely where the drift originates.
- Literal parity is the WEAK half of the problem. The drift that actually bit (#750
  pace_delta axis, #465 dead position default, #633 gap zero-conflation) was LOGIC drift;
  a field-by-field literal diff would have caught none of those bugs' next instances. To
  assert logic parity the test must call all three builders — which needs the submodule
  present AND importable, i.e. the same machinery (c) exists to avoid.
- Precedent says convergence is the direction already chosen: the shared
  `GAP_UNKNOWN_FALLBACK_S` import and `_shared_defaults.DEFAULT_TOTAL_LAPS` are both
  fragments of (b) that already shipped.

### F14. Option (b) — canonical module in `src/agents/`: RECOMMENDED, with the exact shape

**Location:** `src/agents/race_state_builder.py` (new module — ADDITIVE, which even the
strict CLAUDE.md §0.2 reading permits). Rationale: the output type `RaceState` is owned by
`src/agents/strategy_orchestrator.py`; the two constants the builder needs already live in
`src/agents` (`position_projection.GAP_UNKNOWN_FALLBACK_S`,
`_shared_defaults.DEFAULT_TOTAL_LAPS`); and the backend's reverse shim + Docker mount
already deliver exactly this path (F8). `src/simulation/` was considered and rejected: it
would create a simulation→agents dependency that does not exist today.
**Leaf-module constraint (hard):** import `RaceState` LAZILY inside the function, exactly
as `backend/utils/race_state_builder.py:75` and the arcade (:616) do —
`strategy_orchestrator` drags LangChain/LangGraph, and the builder must stay importable
without paying that (mirror the discipline documented in `_shared_defaults.py:3-4`).

**Canonical API (design, not code):**
```
build_race_state(
    lap_state, *,
    driver=None,            # CLI's explicit driver_code override; None → driver dict → "UNK"
    gap_ahead_s=None,       # None → compute positional-car-ahead gap (absorbs _compute_gap_ahead)
    pace_delta_s=None,      # None → compute vs car ahead per the #750 contract
    risk_tolerance=0.5,
    radio_msgs=None, rcm_events=None,   # None → []
    rival=None,             # #431 targeting — _targeting_against_rival moves here verbatim
) -> RaceState
```
`None`-means-compute keeps every existing caller byte-compatible: the /recommend endpoint
keeps passing `request.gap_ahead_s` (client-override semantics unchanged), mcp_tools keeps
its `or GAP_UNKNOWN_FALLBACK_S`, the simulator passes nothing and drops
`_compute_gap_ahead`, CLI/Arcade pass nothing and lose their inline copies.

**#431 stays clean (V13):** `_targeting_against_rival` (`race_state_builder.py:7-50`) is a
pure `(lap_state, rival, fallbacks) → (gap, pace)` function with zero webapp imports;
`rival: Optional[str] = None` on the canonical API leaks nothing — CLI/Arcade simply never
pass it (the CLI's `--rival` is display-only today, `run_simulation_cli.py:1694-1700`, and
wiring it later becomes a feature (b) enables for free, not a cost).

**radio_msgs / rcm_events: PARAMETERS, not built internally.** Decision + justification:
- The three surfaces have three different SOURCES (F3): CLI corpus+synthetic with a
  precedence rule, Arcade instance state, backend caller params. Only the parameter shape
  expresses all three without importing any surface's machinery into the shared module.
- The CLI keeps its post-construction `.extend()/.append()` block UNCHANGED (RaceState list
  fields are mutable; `scripts/run_simulation_cli.py:1744-1762` works identically on a
  canonically-built object). The `--radio-every` synthetic generator, the corpus-suppresses-
  synthetic rule and the SC tracker stay in the main loop where they live: zero PMV rewrite.
- Arcade's builder becomes a thin wrapper: compute `radio_msgs`/`rcm_events` from
  `self._radio_runner`/`self._sc_tracker` exactly as today (:678-697), then pass them as
  params. The SC-injection LOGIC stays arcade-side (stateful, per-surface).
- Building them INSIDE the canonical function would require it to hold a
  RadioPipelineRunner and an SC tracker — instance state, Whisper deps, per-surface policy.
  Rejected.

**CLI surgical-edit test (the PMV constraint): PASSES.** The entire CLI change is (1) one
import, (2) replace the BODY of `_build_race_state` (:1338-1373) with a delegation
`return build_race_state(lap_state, driver=driver_code)` keeping the local function and its
call site (:1711) untouched, (3) lines 1710-1764 stay byte-identical. The behavioral deltas
are exactly the approved canonical literals, each validated with a real `f1-sim` run per
the PMV rule — the same class of edit as the already-landed #750 fix to this function.

**Submodule sequence (the commit-first rule):**
1. **Parent PR** — add `src/agents/race_state_builder.py`; switch CLI + Arcade to it; unit
   tests for the canonical builder land in parent `tests/` (they run WITHOUT the
   submodule); real `f1-sim` run. Backend untouched and unbroken (its copy still works).
2. **Submodule commit** (F1_Telemetry_Manager) — `backend/utils/race_state_builder.py`
   becomes a re-export (`from src.agents.race_state_builder import build_race_state`,
   preserving the public name so `simulator.py:400`, `endpoints/strategy.py:1349`,
   `mcp_tools.py:595` need ZERO changes); delete `_compute_gap_ahead` (:245-279) and stop
   passing `gap_ahead_s` from `_local_build_race_state` (:402). Note: the module ALREADY
   imports `src.agents.*` at module top (:3), so the re-export adds no failure mode that
   does not exist today.
3. **Parent PR** — bump the submodule pointer. Until 3 lands the backend runs its old copy:
   the drift window stays open between 1 and 3, so keep all three in one sprint.

**Residual risks accepted (stated, not hidden):**
- Between steps 1 and 3 the twin exists again, briefly and knowingly.
- A future editor may top-level-import RaceState in the canonical module and slow every
  surface's boot — guard with the module docstring + a cheap test asserting the module
  imports without `langchain` appearing in `sys.modules`.
- The canonical literals CHANGE observable CLI behavior on the weather-missing path
  (40.0 → 35.0) — that is the point, but it must be named in the PR body.

## Recommended canonical literals — one decision per row, each overridable

| Field | Recommend | Grounds | Trade-off if overridden |
|---|---|---|---|
| `total_laps` | `session_meta.get("total_laps")`; missing → `DEFAULT_TOTAL_LAPS` (import from `_shared_defaults`, never restate 57) + `logger.warning` | Every internal producer supplies it (F6); the only reachable miss is client JSON at /recommend, and downstream agents ALREADY fall back to the same 57 — a builder stricter than its consumers buys a crash, not correctness. The warning keeps it loud. | Fail-loud (CLI today) is doctrine-purer; costs turning working (if sloppy) API calls into 422s. Defensible if the human prefers it. |
| `compound` | `"UNKNOWN"`, PLUS normalise `""`/`"nan"`/`"None"`/absent → `"UNKNOWN"` | Never crashes (F5: pace encodes id-1 silently, tire warns + C3, N15 maps to its own trained −1 bucket, LLM prompt sees the truth); zero collisions in 71 races (F10); "MEDIUM" is a findable value = the sentinel-collision doctrine violation, and it lies to the prompt. The normalisation is what makes the choice REAL — today `"nan"` flows through regardless (F10). | "MEDIUM" yields near-identical model encodings and never prints an odd word in the UI. If "UNKNOWN" on screen is unacceptable, fix the DISPLAY, not the state. |
| `tyre_life` | `0` | TyreLife==0 occurs ZERO times in 2023-2025 (executed, F10) → non-colliding sentinel; 1 collides with real fresh-tyre laps. Fires only on hand-built states (key missing), so the slightly out-of-distribution input is a corner, not a path. | `1` keeps the model strictly in-distribution at the cost of being indistinguishable from a real fresh tyre — the #428/#465 bug shape. |
| `track_temp` | `35.0`; also treat present-but-None as missing | Measured dataset median is exactly 35.0 (F10); the divergence is LIVE today (fires whenever weather.parquet is absent, F11); 40.0 matches nothing measured. | Any float here is fabricated (schema forces float; `Optional[float]` is the honest long-term fix but a RaceState contract change, same as gap_ahead_s per the arcade's own comment :646-651). Choosing 40.0 needs a reason nobody has produced. |
| `air_temp` | `25.0` (keep) | All three agree; ≈ dataset median 24.1. | — |
| `lap` | top-level `lap_state["lap_number"]`, fallback driver dict, then default 1 + warning | RSM emits both (:596, :319); behavior-identical on all real paths; CLI's driver-dict direct index crashes on a hand-built state the other two accept. | Fail-loud purism, same argument as total_laps. |
| `position` | keep fail-loud ValueError (all three agree); unify the message, cite #628+#465 | Already converged; the guard is load-bearing (#428 shape). | none |

## What I tried to break and could not (option b)

1. **"src/agents is untouchable, so (b) is forbidden."** Fails: the constraint is "additive
   entry points only" (CLAUDE.md §0.2), and a NEW leaf module is the additive case —
   `_shared_defaults.py` is the exact shipped precedent, created for the same
   consolidate-the-restated-constant reason.
2. **"The backend can't import it in Docker."** Fails: `docker-compose.yml:16` mounts
   `../../src:/app/src:ro` and `paths.py` resolves `/app` — the canonical module ships into
   the container by the SAME mechanism `position_projection` already does. The backend
   image copies only `backend/` (Dockerfile:17); `src/` is a mount, so no rebuild semantics
   change.
3. **"It forces a CLI rewrite via the radio/RCM block."** Fails — but ONLY because of the
   parameters-not-internal decision plus RaceState's mutable list fields. If a future
   refactor freezes RaceState, the CLI's post-construction mutation breaks — record this in
   the canonical module's docstring.
4. **"A circular import: agents ↔ builder."** Fails: the builder imports
   `strategy_orchestrator` lazily; nothing in `src/agents` imports the builder; Arcade and
   CLI already import `src.agents.*` today (:615-616, :158).
5. **"The submodule's standalone context breaks on `src.agents`."** Fails to be NEW:
   `backend/utils/race_state_builder.py:3` already imports `src.agents.position_projection`
   at module import time. Whatever handles that today handles the re-export identically —
   a pre-existing property of the submodule, not a cost introduced by (b).
6. **Could NOT fully close:** (i) whether every `uv` version users run fetches submodules
   on `uv tool install git+...` — irrelevant to (b) (nothing parent-side will import
   backend), decisive only against (a), which has two other independent kills; (ii) whether
   any consumer outside this repo pair imports `backend.utils.race_state_builder` (MCP
   clients call tools, not modules — none found); (iii) the hypothetical frozen-RaceState
   future (risk recorded in 3).

## Bottom line

**Option (b).** One canonical `build_race_state` in `src/agents/race_state_builder.py`
(leaf module, lazy RaceState import), radio/rcm/risk/rival as optional parameters,
gap/pace computed internally when not supplied (absorbing `_compute_gap_ahead`),
`_targeting_against_rival` moved verbatim, the backend file kept as a thin re-export so its
three call sites do not change, the CLI edit strictly surgical (delegate the body, main
loop untouched), landed as parent-PR → submodule-commit → pointer-bump in one sprint.
Every literal above is a separate human decision — approve or override per row.
