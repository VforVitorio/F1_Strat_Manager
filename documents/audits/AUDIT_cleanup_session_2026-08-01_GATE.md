# ADVERSARIAL GATE — cleanup session 2026-08-01 (branch `fix/cleanup-anti-slop-scoped`)

**Instrument:** adversarial gate agent (Fable), read-only except this file. Success condition:
find what is STILL broken or overclaimed in the cleanup diff (4 commits on top of `dev`) and the
gitignored CLAUDE.md edit. Findings are appended incrementally as they are confirmed.

**Commits under review:**

```
99f17eb fix(agents): add the leaf module backing DEFAULT_TOTAL_LAPS
08864fb fix(agents): dedupe restated defaults across tire/race_situation/pit_...
1fec855 fix(agents): remove pace_agent's dead ReAct scaffold entry point + ar...
ea280e9 docs(audit): open the dedicated cleanup-session report
```

## Checklist (claims a-k)

- [x] a) 4 `get_*_react_agent` free functions had ZERO callers — VERIFIED
- [x] b) pace_agent's 3 deleted constants dead — VERIFIED
- [x] c) `DEFAULT_TOTAL_LAPS=57` identical at all 6 sites; leaf verified by execution — VERIFIED
- [x] d) TrackTemp 35.0→38.0 — VERIFIED (executed: empty session_meta → 38.0); caveat j-2
- [x] e) `_conservative_stub()` identical values — VERIFIED
- [x] f) deeper pace scaffold unreachable; keeping it was right — VERIFIED
- [x] g) Part 3 table VERIFIED row-by-row; **radio-blind-CLI claim REFUTED (HIGH)**
- [x] h) CLAUDE.md edits — VERIFIED, no overclaim
- [x] i) ruff clean; tests/agents+audit 129 passed; smoke 5 passed/1 pre-existing data skip; mc/simulation/eval below
- [x] j) new slop: j-1 bisect break (MED), j-4 uncommitted report + dangling placeholders (MED), j-7 stale docs page (MED), j-2/j-3/j-6 (LOW)
- [x] k) no residual in-scope literals; no surviving removed symbols — VERIFIED

## Findings

### a) Dead free functions — VERIFIED

Repo-wide ripgrep (src/, scripts/, tests/, notebooks/, documents/, docs/, and the `src/telemetry`
submodule — 27,391 .py files on disk, so the submodule content WAS searched) for
`get_pace_react_agent|get_tire_react_agent|get_race_situation_react_agent|get_pit_strategy_react_agent`:
the ONLY hits are the two audit markdown files describing the deletion. `src/agents/__init__.py`
read in full: `_EXPORTS` (lines 25-51) and the `TYPE_CHECKING` block (79-101) list only the
`run_*`/`run_*_from_state` names plus dataclasses — no react-agent entry ever existed there.
Methodology note: my first grep pass was silently broken (the RTK hook rewrote `rg` to a `grep`
that rejects `--no-ignore`, and my `2>/dev/null` hid the error) — re-executed with a working
ripgrep before trusting the zero.

### b) pace_agent's 3 deleted constants — VERIFIED

`_load_encoding_maps` (pace_agent.py:183-221) builds `feature_manifest_laptime.json`,
`circuit_clustering/circuit_clusters_k4_2025.parquet` and `laps_featured_2025.parquet` inline from
its `processed_dir` argument — byte-identical relative paths to the three deleted constants.
Repo-wide grep for `_CLUSTER_PARQUET|_LAPS_FEATURED|_FEATURE_MANIFEST`: only the two audit docs.
`_PROCESSED` (the module global the constants used) is NOT newly orphaned — still the default for
`processed_dir` at line 148.

### d) TrackTemp 35.0 → 38.0 — VERIFIED with caveats

The changed line is genuinely inside `_compute_weather_features` (race_situation_agent.py:675-693;
the git hunk header naming `_compute_rcm_features` is just git's nearest-context artifact). The
same file's `run()` builds `'TrackTemp': ... else 38.0` at line 1317 (+ `track_temp_start` 38.0 at
1319) and `run_from_state()` uses `wx.get('track_temp', 38.0)` at line 1403 — the claim holds
against current code. `_compute_weather_features` has exactly one caller (line 1009, fed by
run/run_from_state-built session_meta where the key always exists), so production behaviour is
unchanged; only hand-built session_meta changes. No test asserts the old 35.0 fallback: the eight
`35.0` hits in tests/ all SET `track_temp=35.0` explicitly on constructed states (pass-through, not
fallback), and `test_race_situation_hardening.py:164` sets `"TrackTemp": 38.0` explicitly.
**Caveat (LOW, see j-2):** the new comment hardcodes "lines ~1314/~1401" — actual lines today are
1317/1403, and any edit above them rots the reference.

### e) `_conservative_stub()` — VERIFIED

Old inline constructors (diff, both sites): `deg_rate=0.03`, `laps_to_cliff_p10/50/90 =
20.0/30.0/40.0`, `compound`/`current_tyre_life`/`gp_name` from locals, `reasoning` per-site. New
helper reproduces exactly these fields; both call sites pass the same locals and only the `reason`
string differs (site 2 still appends the parse-`reasoning` suffix, as before). No value drift.

### f) pace_agent's deeper ReAct scaffold — VERIFIED (and keeping it was the right call)

`.get_react_agent(` grep: 3 hits, all *self*-calls inside tire/race_situation/pit_strategy
`_run_core` — none for pace. `PACE_TOOLS`: only pace_agent.py itself (798, 985, 988), the frozen
notebook N25, and the audit docs. So the ~235-line scaffold is unreachable from outside, as
claimed. Judgment: leaving it was correct — not because of the "preserved 100%" header alone, but
because the other three siblings' identical-shaped scaffolds ARE live, so deleting pace's would
make the four agents structurally asymmetric while the parity question is still open. Deleting is
a one-commit follow-up once Víctor decides; resurrecting after deletion is also trivial via git.
Either way it's a decision, not a cleanup, and the audit correctly routed it to the owner.

### k) Residual literals — VERIFIED (with one honest-scope note)

Repo-wide grep for `total_laps` near `57`: the only remaining *fallback* literals are
`src/arcade/strategy.py:703` and `src/telemetry/backend/utils/race_state_builder.py:108` — exactly
the two out-of-scope `_build_race_state` copies Part 3 defers to the architecture session, and the
audit's Part 3 table already lists both. Everything else is tests/notebooks/docs passing 57 as an
explicit VALUE, not a fallback. No in-scope site was missed. No removed symbol survives anywhere
in code.

### i) Test suites + lint — VERIFIED (executed by this gate, not read from the report)

- `uv run ruff check` on all 5 touched files: "All checks passed!"
- `uv run pytest tests/agents/ tests/audit/ -q`: **129 passed** in 165s (matches 08864fb's claim
  exactly).
- `uv run pytest tests/mc/ tests/simulation/ tests/eval/ -q`: **286 passed**, 627 warnings, 14:49.
  This is the number the audit report's dangling "see below" never recorded (j-4) — recorded here.
- `uv run pytest tests/infra/test_smoke.py -q`: 5 passed, 1 skipped ("Qatar 2025 parquets not
  available in this environment" — pre-existing data-availability skip, unrelated to this diff).

### g) Architecture report (Part 3) — table VERIFIED, but one headline claim REFUTED

**Table: verified row by row against current code.** CLI (`run_simulation_cli.py:1304-1373`):
`total_laps` direct index (raises), compound `"UNKNOWN"`, tyre_life `0`, air 25.0, track **40.0**,
position-None → ValueError (#628), gap fallback imported `_GAP_UNKNOWN_FALLBACK_S`. Arcade
(`strategy.py:599-715`): `.get("total_laps", 57)`, `"MEDIUM"`, `1`, 25.0, 35.0, ValueError (#465),
imported fallback, radios from `RadioPipelineRunner`. Backend (`race_state_builder.py:53-120`):
identical to Arcade's literals, radios as parameters `None → []`. Both quoted arcade comments exist
verbatim (`strategy.py:600-603` sys.path-shim blocker; `strategy.py:642-644` "two-way unification,
not the three-way one an earlier draft of this comment claimed"). `simulator.py:378-406`
`_local_build_race_state` is a genuine thin wrapper delegating to `build_race_state`. Line ranges
in the audit are exact.

**REFUTED (HIGH, report-accuracy): "the CLI's orchestrator input is missing radio/RCM context".**
The audit's self-described *new, previously-unreported divergence* — "the CLI path silently
produces a `RaceState` with no radio/RCM context at all … the CLI's orchestrator input is missing
information that the other two surfaces carry" — is false. `_build_race_state` indeed leaves the
fields at their Pydantic default, but the CLI main loop then populates the SAME object before
inference: `run_simulation_cli.py:1744-1762` extends `race_state.radio_msgs` with `real_radios`,
`race_state.rcm_events` with `real_rcms`, appends the SC-tracker synthetic event, and appends
simulated radio/rcm — 370 lines below the builder. The arcade's own docstring even says it
populates radios "(same as the CLI)" (`strategy.py:604-606`), which the audit quoted around but
did not reconcile. The divergence that EXISTS is only *where* population happens (inside the
builder vs in the caller); the audit's follow-up question for Víctor ("is the CLI's RaceState
deliberately radio-blind, or an oversight?") rests on a false premise and, uncorrected, would send
the dedicated architecture session hunting a gap that does not exist — or "fixing" it into
double-injection. This is the project's own documented failure shape: a claim written into the
planning artifact without verifying against the consumer (cf. "a false claim inside an ISSUE is
SCOPE").

### h) CLAUDE.md edits — VERIFIED

Live file (gitignored, read directly): `f1-webapp = "scripts.run_webapp:main"` IS in
`pyproject.toml` `[project.scripts]` (line 124, alongside f1-strat/f1-sim/f1-arcade/f1-eval).
`src/shared/README.md` opens "**Status: archived.**" and points at `src/data_extraction/` as
canonical. `scripts/run_webapp.py`'s docstring says "post-race web app (FastAPI backend + React
SPA)" served via docker compose. In CLAUDE.md: `grep -i streamlit` → **zero matches** (all 4
stale references gone), line 55 adds `data_extraction/` as canonical, line 56 marks `shared/`
ARCHIVED, line 99 documents `f1-webapp`, §9/§10 updated. No overclaim found in the CLAUDE.md edit.

### j-0) No newly-orphaned code — VERIFIED

`_get_default_pace_agent`/`_get_default_pit_agent`/`_get_default_situation_agent`/
`_get_default_tire_agent` all retain live callers (each file's `run_*`/`run_*_from_state` free
functions). `_PROCESSED` still used. Nothing the deletions touched became dead. The one sibling
that must NOT be deleted, `get_rag_react_agent`, is untouched and still invoked
(`rag_agent.py:134/195`) — the cleanup did not over-delete into the live twin.

### c) `DEFAULT_TOTAL_LAPS` — VERIFIED

All 6 call sites confirmed in the diff (pit_strategy ×3: 1037/1271/1452; race_situation ×2:
1008/1360; tire ×1: 1478); every one is the same `meta.get('total_laps', <57>)` shape, value and
`.get`-fallback semantics unchanged (race_situation's `int(...)` wrap kept). Leaf-module claim
verified by execution, not reading: `importlib.import_module('src.agents._shared_defaults')` in a
fresh interpreter returns 57 and loads NO heavy modules (torch/xgboost/lightgbm/transformers/
langgraph/pandas all absent from `sys.modules`) — no import cycle, and the lazy `src/agents/
__init__` stays lazy through it.

### j-1) MEDIUM — non-bisectable commit order: 08864fb imports a module that does not exist until 99f17eb

`git ls-tree 08864fb -- src/agents/_shared_defaults.py` → empty, while at that same commit
pit_strategy/race_situation/tire all contain `from src.agents._shared_defaults import
DEFAULT_TOTAL_LAPS` (git grep at 08864fb, 3 import sites). Checking out 08864fb gives
`ModuleNotFoundError` on three agent modules; `git bisect` across this branch will land on a
broken tree. The companion commit even says "Companion to the previous commit" — the order is
simply inverted. Fix before pushing: reorder (add the leaf module first) via rebase, or squash
99f17eb into 08864fb. CI on the PR head won't catch this; only bisect/checkout will.

### j-2) LOW — the new TrackTemp comment hardcodes line numbers

`race_situation_agent.py:678` says "(lines ~1314/~1401)"; actual lines today are 1317/1403 and
any edit above them rots the pointer. The project's own style rule bans internal line references
in comments. Name the functions (`run` / `run_from_state`), not the lines.

### j-3) LOW — the new module's docstring perpetuates the "2022-2025 dataset" falsehood

`_shared_defaults.py` says "57 is the median/mode race length across the 2022-2025 dataset".
Executed check: `data/processed/laps_featured.parquet` holds years **[2023, 2024, 2025]**, 71
races, median **57.0**, mode **57.0** — the NUMBER is right, the dataset naming is wrong, and
CLAUDE.md §1 explicitly warns against exactly this conflation ("'2022–2025' names the
ground-effect regulation ERA, not the data range … there is no 2022 season anywhere in `data/`").
The error was inherited from the old inline comments (race_situation's deleted comment said
"2022-2025 dataset (71 races)") — the consolidation was the one moment to correct it and instead
promoted it into the single-sourced docstring. One-word fix: "2023-2025".

### j-4) MEDIUM — the audit report's Part 4 + Verification section is UNCOMMITTED, and ends with two dangling "see below" placeholders

`git status`: `documents/audits/AUDIT_cleanup_session_2026-08-01.md` is modified in the working
tree (+44 lines vs HEAD) — the committed version (ea280e9) ends at "(Appended incrementally as
each fix lands…)" and contains NEITHER the N4 finding, NOR the Part 4 fix/deferred tables, NOR
the Verification section. Pushing the branch as-is ships a report missing its own results. Worse,
even the working-tree version's Verification section still reads "`pytest tests/mc/
tests/simulation/ tests/eval/`: **see below**" and "Real `f1-sim …` run: **see below**" — and the
file ends there. The 70/70-laps real-run claim exists only in commit 08864fb's message, recorded
nowhere with evidence. Fix: fill both placeholders with actual results and commit the report
update as a fifth commit before pushing.

### j-7) MEDIUM — the docs site still teaches the deleted API (the twin that never got the fix, docs edition)

`docs/pages/agents-api.md:25`: "Every agent except N29 also exposes a `get_*_react_agent()`
factory returning a compiled LangGraph `CompiledGraph`, for callers that want to drive the graph
directly." After this diff that sentence is false for FOUR of the six agents — only N30's
`get_rag_react_agent` survives. A reader following the docs site gets an `AttributeError`. The
exact-name greps (mine included, first pass) miss it because the page names the *pattern*
(`get_*_react_agent()`), not the identifiers — the literal instance of "grep is not an audit".
This is the project's own codified lesson (CLAUDE.md §11, 2026-07-16): "cuando un fix cambia un
contrato, la página que lo describe es parte del fix, no un follow-up". The audit's scope defense
(agents-api.md wasn't touched in the 30-PR window) does not apply: the CLEANUP itself changed the
contract the page describes, so the page became part of THIS fix. Update the sentence (only N30
exposes a factory; the other four keep an internal `get_react_agent()` method — pace's unwired)
in this same PR.

### j-6) LOW — pace_agent's docstring header now makes a false claim this diff itself created

`pace_agent.py:7` still titles its API list "Public API (**unchanged — backward compatible**)"
— but this diff REMOVED a public function (`get_pace_react_agent`) from that very list. The
parenthetical was written for an earlier refactor's promise; after this deletion it asserts the
opposite of what the commit did. The sibling files are consistent (their docstrings just list the
API without the "unchanged" claim). One-line fix: drop the parenthetical.

### j-5) Commit-message honesty — VERIFIED otherwise

1fec855's diff is exactly what it says (39 deletions, pace_agent only; explicitly discloses NOT
removing the deeper scaffold). 08864fb's "four fixes bundled" matches its diff; its "129 passed"
claim reproduced exactly by this gate. ea280e9's message matches its content (though that content
carries the Part-3 radio-blind claim refuted in g). `_shared_defaults.py` is a justified
extraction, not premature abstraction: 6 real call sites across 3 files, mirrors the existing
`guard_rails.py` leaf pattern, and the leaf property was verified by execution (see c).

### Real-run reproduction — VERIFIED

`uv run f1-sim Budapest NOR McLaren --no-real-radios --no-llm`, executed by this gate on the
post-diff tree: exit 0, "**All 70 lap(s) OK**", P5 → P1, actions STAY_OUT·59 / PIT_NOW·5 /
UNDERCUT·6, wallclock 44.0s. Commit 08864fb's 70/70 claim is independently reproduced — it just
needs to be RECORDED in the report's dangling placeholder (j-4).

---

## Severity ranking

| # | Severity | Finding | Where | Fix effort |
|---|---|---|---|---|
| 1 | **HIGH** (report accuracy, not code) | Part 3's "CLI orchestrator input has no radio/RCM context" claim is FALSE — the CLI main loop populates `race_state.radio_msgs`/`rcm_events` at `run_simulation_cli.py:1744-1762`. Uncorrected, it aims the dedicated architecture session at a nonexistent gap (or a double-injection "fix"). | `AUDIT_cleanup_session_2026-08-01.md` Part 3 | Rewrite the paragraph: the divergence is WHERE population happens, not WHETHER. |
| 2 | **MEDIUM** | Non-bisectable history: 08864fb imports `_shared_defaults`, created only in 99f17eb. Checkout/bisect at 08864fb = ModuleNotFoundError in 3 agent modules. | branch history | `git rebase -i` reorder or squash the two commits before pushing. |
| 3 | **MEDIUM** | The audit report's Part 4 + Verification (+ N4) are UNCOMMITTED (+44 lines in working tree), and both "see below" placeholders are still dangling — the mc/sim/eval result and the real-run result are recorded nowhere. | working tree + report | Fill placeholders (286 passed measured by this gate) and commit as a 5th commit. |
| 4 | **MEDIUM** | `docs/pages/agents-api.md:25` still says every agent except N29 exposes `get_*_react_agent()` — false for 4 of 6 after this diff. The project's own §11 lesson: the page describing a changed contract is part of the fix. | docs site | One-sentence rewrite in this PR. |
| 5 | LOW | New docstring in `_shared_defaults.py` perpetuates "2022-2025 dataset" — data is 2023-2025 (verified: 71 races, years [2023,2024,2025]); CLAUDE.md §1 explicitly warns against this exact conflation. The 57 itself is correct (median 57.0, mode 57.0). | `_shared_defaults.py:14` | s/2022-2025/2023-2025/. |
| 6 | LOW | New comment hardcodes "(lines ~1314/~1401)" (actual: 1317/1403) — banned internal line refs, rots on next edit. | `race_situation_agent.py:678` | Name the functions instead. |
| 7 | LOW | `pace_agent.py:7` still claims "Public API (unchanged — backward compatible)" in the very diff that removed a public function from that list. | `pace_agent.py:7` | Drop the parenthetical. |

No HIGH/MEDIUM **code-behaviour** defect was found: every code change in the diff is behaviourally
identical to what it replaced (verified by diff, by execution, and by 420 passing tests).

## What I tried to break and could NOT

- **Hidden callers of the 4 deleted functions**: exact-name grep over the full tree including the
  27k-file telemetry submodule, notebooks (.ipynb JSON), documents/, docs/, plus the lazy-loading
  `__init__.py`'s `_EXPORTS` dict and `TYPE_CHECKING` block, plus a PATTERN-level grep
  (`react_agent`) that caught what exact names couldn't (j-7 was found this way — in docs, not
  code). No code caller exists. Also verified the one live twin (`get_rag_react_agent`) was not
  over-deleted.
- **Value drift in the dedup**: all 6 `DEFAULT_TOTAL_LAPS` sites diff-compared (same `.get`
  semantics, same 57); `_conservative_stub` field-by-field identical at both sites; TrackTemp
  fallback executed with an empty dict → 38.0/28.0/delta 0.0 exactly as claimed.
- **Import-time regression**: fresh-interpreter import of `_shared_defaults` loads zero heavy
  modules — the lazy-package property survives; no cycle.
- **Test suite regressions**: 129 (agents+audit) + 286 (mc+simulation+eval) + 5 (smoke) all pass
  post-diff on this machine; ruff clean on all 5 touched files.
- **The architecture table**: re-derived every row from current code at the cited line ranges —
  all values, both ValueError issue numbers, both quoted arcade comments, and the thin-wrapper
  claim check out. The only thing that broke was the radio-blind interpretation (finding 1).
- **A newly-dead orphan**: every helper the deleted code called (`_get_default_*`, `_PROCESSED`)
  still has live callers.
