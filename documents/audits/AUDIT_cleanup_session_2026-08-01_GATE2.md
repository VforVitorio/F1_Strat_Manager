# Adversarial gate 2 — second-wave cleanup (`fix/cleanup-anti-slop-wave2`)

Date: 2026-08-01 · Gate model: Fable 5 · Branch: `fix/cleanup-anti-slop-wave2` (6 commits ahead of `dev`)
Rules: no repository file modified except this report. Findings appended incrementally as executed.

Commits under review:

| # | SHA | Subject |
|---|---|---|
| 1 | `cd3f63f` | fix(tests): single-source the canned MC/golden sub-agent outputs |
| 2 | `896d773` | fix(tests): single-source the tire-routing-config skip guard |
| 3 | `906fcb1` | fix(arcade): import classify_action instead of mirroring it in theme.py |
| 4 | `7ec75da` | docs(shared): correct fastf1_extractor.py's false import claim |
| 5 | `d732e1d` | fix(scripts): use the robust repo-root search everywhere, not a fixed depth |
| 6 | `c161d7f` | docs(audit): record the second-wave cleanup (post-#776) |

## Checklist

- [x] a) VERIFIED (24 files AST-parsed; caveat: wrong copy-count in conftest docstring; 1 residual fixture-level twin)
- [x] b) VERIFIED inert (file on disk, 591 B; drift direction analysed)
- [x] c) VERIFIED untouched (empty diff, own guards intact)
- [x] d) VERIFIED (pre-fix crash executed ×2; post-fix executed ×6 inputs; caller sanitizes)
- [x] e) VERIFIED WITH CAVEATS (wrong "Both imported" comment; dead `classify_alerts` island; `_FLAG_BG_BY_INTENT` missing YELLOW_FLAG_SECTOR — live pre-existing twin)
- [x] f) VERIFIED (placement/uniqueness table; `_common.py` fallback correct; smoke-tested)
- [x] g) VERIFIED (0 code cells, 1 markdown link, executed json scan; INFO: README attribution imprecise)
- [x] h) VERIFIED WITH CAVEAT (fingerprint ×1; FIFTH consumer `test_tyre_wear_term.py` still on the old indirect path; both fresh noqa comments name wrong consumers)
- [x] i) ruff check clean, ruff format clean (64 files); pytest: see below
- [x] j) VERIFIED WITH CAVEATS (commit 1 leaves a broken intermediate tree — imports `tests.conftest` one commit before it exists; "two"→four miscount in commit 2's body)

## Preliminary note — file-count reconciliation

Commit 2's stat shows only 17 test files + `conftest.py`, while the audit doc claims "all 21 files".
Resolved before treating it as a finding: the 4 `tests/mc/` consumer files
(`test_mc_is_a_real_decision.py`, `test_mc_state_helpers.py`, `test_projection_golden.py`,
`test_strategy_goldens.py`) received the guard migration folded into commit 1 (`cd3f63f`) because
they were already being edited there. 17 + 4 = 21. A live grep for
`skip_no_tire_models|HAS_TIRE_MODELS` returns exactly those 21 test files + `tests/conftest.py`.
Not a defect, but the audit doc does not say the migration is split across two commits (see j).

---

## Findings

### a) Docstring-corruption recovery — VERIFIED (with one documentation caveat)

Executed: `ast.parse` + `ast.get_docstring` + leftover-definition regexes over **all 24 files**
(21 migrated test files, `tests/conftest.py`, `tests/mc/canned_outputs.py`, and the 4 canned-outputs
consumers, which overlap with the 21). Full list, each parsed OK:

`tests/agents/`: test_agents.py, test_n15_envelope.py, test_orchestrator_prompt.py,
test_prompt_constants_match_tables.py, test_tire_cumulative_deg.py ·
`tests/audit/`: test_engine_scope_defaults.py, test_tire_agent_hardening.py,
test_tire_mc_determinism.py · `tests/engine/`: test_engine.py, test_engine_memory.py,
test_engine_no_llm.py, test_engine_threads_every_argument.py, test_memory_scope_is_deliberate.py ·
`tests/mc/`: test_mc_is_a_real_decision.py, test_mc_state_helpers.py, test_projection_golden.py,
test_sc_regulatory_rails.py, test_strategy_goldens.py, test_tyre_wear_term.py,
test_undercut_targets_are_on_track.py, canned_outputs.py · `tests/simulation/`:
test_simulation.py · plus `tests/conftest.py`.

- No module docstring contains `tests.conftest` or `canned_outputs` as an import. 8 docstrings
  contain the WORD "import" — all verified to be pre-existing prose ("importing the engine pulls…"),
  zero import STATEMENTS inside any docstring (regex `^(from \S+ import |import \S+$)` over each
  docstring: 0 hits across all 24).
- No leftover `_HAS_MODELS = (...)` inline definition, no `_MODELS_DIR`, no
  `_skip_no_models = pytest.mark...` definition anywhere in `tests/` (live grep). The surviving
  `_skip_no_models` / `_HAS_MODELS` occurrences are the intentional import ALIASES
  (`from tests.conftest import skip_no_tire_models as _skip_no_models`), which also keep the old
  docstring prose accurate (e.g. `test_engine_no_llm.py:12` still names `_skip_no_models` — the
  alias still exists at line 25, so the prose is not stale).
- The corruption victim itself, `tests/mc/test_mc_is_a_real_decision.py`: docstring is 772 chars of
  clean prose; the new import sits at line 30, outside the docstring.

**Caveat (LOW, doc accuracy):** `tests/conftest.py:8` claims "26 near-identical copies of the
tire-degradation-routing-config check alone", and the audit doc says "~28 test files". Executed
count on `dev`: exactly **21** module-level `_HAS_MODELS = ...routing_config.json` guard copies
(+ 4 different-artifact guards = 25 total `_HAS_MODELS` definitions, + 1 fixture-style runtime
check, see below). Neither 26 nor 28 is reproducible. A shipped docstring carrying a wrong count is
the project's own "comment naming the wrong mechanism" class — should say 21 (or 22).

**Residual twin (LOW):** `tests/audit/test_pace_orchestrator_hardening.py:247` still hand-checks
the same `routing_config.json` path inside its `clamp_fn` fixture (imperative `pytest.skip`, not a
`skipif` marker). Functionally harmless and a different mechanism (fixture-scope lazy import), but
it is a 22nd copy of the exact path expression the migration exists to single-source; it could
import `HAS_TIRE_MODELS` from `tests.conftest`. The audit doc does not mention it.

### b) `.exists()` vs `.is_file()` — VERIFIED inert

- Executed on this machine: `data/models/tire_degradation/routing_config.json` → `exists()=True`,
  `is_file()=True`, `is_dir()=False`, size 591 B. A regular file, so the two predicates agree.
- The two drifters confirmed from the `dev` diff: `test_prompt_constants_match_tables.py:40` and
  `test_tyre_wear_term.py:44` both used `.is_file()`; both now use the shared `.exists()` guard.
- Divergence analysis: they differ only if the path exists as a **directory**. In that pathological
  case the old `.is_file()` files would SKIP while the new guard says "models present" and the test
  fails loudly at model-load. On CI (path absent) both are False → skip, identical. Failing loudly
  on a corrupted layout is the better behaviour anyway. Inert in every real state; the change is
  defensible even in the unreal one.
- `test_agents.py`'s `_MODELS_DIR` intermediate was fully removed with no dangling reference
  (grep: 0 hits outside the deleted line); `pytest` import still used (line 28 marker, parametrize).

### c) 4 different-artifact-gated files — VERIFIED untouched

`git diff dev..HEAD` over `tests/eval/test_hygiene_golden.py`, `tests/eval/test_registry_golden.py`,
`tests/audit/test_pit_agent_hardening.py`, `tests/agents/test_prev_lap_default_is_single_sourced.py`
is **empty**, and each still carries its own independent guard on HEAD (verified by live grep:
overtake `model_config.json` ×2, pit_prediction `model_config.json`, lap_time `.is_dir()`). Exactly
the 4 files the audit doc names, none accidentally collapsed into the tire-only guard.

### d) `classify_action` None-fix — VERIFIED (executed)

- (i) Both pre-fix bodies reconstructed from the `dev` diff and EXECUTED with `None`:
  `strategy.py` old (`_ACTION_STYLE.get(action.upper(), (ACCENT, action.upper() or "--"))`) and
  `theme.py` old (`_ACTION_STYLE.get(action.upper(), (ACCENT, (action or "--").upper()))`) both
  raise `AttributeError: 'NoneType' object has no attribute 'upper'`. The audit doc's "crashed
  identically" claim holds.
- (ii) Post-fix `src.arcade.strategy.classify_action` executed with 6 inputs:
  `None → ((167,139,250), '--')` · `'' → ((167,139,250), '--')` · `'PIT_NOW' → ((239,68,68),
  'PIT NOW')` · `'pit_now' → same` (case-folding preserved) · `'banana' → ((167,139,250),
  'BANANA')` · `'STAY_OUT' → ((16,185,129), 'STAY OUT')`. Every non-None result is identical to
  what both old bodies produced (checked by construction against both old fallback expressions —
  they only ever disagreed on nothing; `''` gave `(ACCENT, '--')` in both).
- (iii) `orchestrator_card.py:155` sanitises with `str(latest.get("action") or "--")` before the
  line-165 call, and imports `classify_action` via theme's re-export (`theme.py:23` noqa
  comment matches reality). The fix hardens a latent path, it does not paper over a live crash.

### e) theme.py orphan check — VERIFIED WITH CAVEATS (one wrong comment, one dead island, one live twin left behind)

- `_ALERT_SEVERITY` is no longer imported by theme.py (old import line deleted); grep confirms no
  use inside theme.py. ✓
- `TEXT_TERTIARY` is NOT orphaned: still used at `theme.py:179` (`flag_chip_html` fallback) and
  `:219` (status-bar stylesheet), plus re-exported to `orchestrator_card.py`. ✓
- `severity_color`: zero callers repo-wide (only mentions are in the two audit .md files). ✓

**CAVEAT 1 (MEDIUM, wrong comment in freshly-written code):** the rewritten block
`theme.py:61-72` says "Action classification + severity — **Both imported** from
src.arcade.strategy (not duplicated) … Importing both closes it." False: the import block
(`theme.py:22-24`) imports ONLY `classify_action`. The severity dict is not imported anywhere in
theme.py anymore (its only consumer, `severity_color`, was deleted in the same commit). The comment
narrates a mechanism that does not exist — the project's own top-listed gate-caught bug class ("a
comment naming the wrong MECHANISM is worse than none"), introduced by this wave.

**CAVEAT 2 (LOW, dead island created):** deleting `severity_color` removed the LAST external
consumer of `_ALERT_SEVERITY`. Executed grep (branch AND `dev`): `classify_alerts`
(`strategy.py:763`) has **zero call sites anywhere in the repo** — it was already dead on `dev`,
and `_ALERT_SEVERITY` now feeds only that dead function. The audit doc justifies the deletion as
"restated `classify_alerts`'s mapping" as if `classify_alerts` were the live canonical; it is not
live. The wave's cleanup is still correct, but strategy.py now carries a dead
`classify_alerts` + `_ALERT_SEVERITY` island that the next cleanup should delete or wire up.

**CAVEAT 3 (MEDIUM, pre-existing twin the wave walked past):** `theme.py:164-173`
`_FLAG_BG_BY_INTENT` is a THIRD hand-maintained copy of the flag-severity semantics — and it lacks
the `"YELLOW_FLAG_SECTOR"` key, the exact key whose absence WAS bug #398 in the severity dict
(strategy.py's comment: "the form DOUBLE YELLOW resolves to"). The path is live and reachable:
`rcm_events.py:144` emits `YELLOW_FLAG_SECTOR` → it is in `radio_agent.py`'s `_SAFETY_FLAGS`
(alert-generating) → `agent_formatters.py:351` calls `flag_chip_html(intent)` → lookup misses →
chip renders neutral grey (`TEXT_TERTIARY`). A sector/double yellow renders as an unstyled grey
chip in the dashboard alerts card while the severity dict renders it amber — the same visual bug
shape as #398, alive in the same file this commit edited, ~100 lines below a comment narrating that
exact drift mechanism. Pre-existing on `dev` (not a wave regression), but this gate exists to find
what the fixes walked past.

### f) 8-script repo-root fix — VERIFIED

Executed a placement/uniqueness scan over all 8 files. Per file: `.git`-search snippet line,
occurrence count, `sys.path.insert` line, first `src.*`/`scripts.*` import line:

| File | snippet | count | sys.path | first project import | order |
|---|---|---|---|---|---|
| build_radio_dataset.py | 105 | 1 | 109 | 111 | OK |
| download_data.py | 33 | 1 | 36 | 38 | OK |
| measure_fresh_reference_gate.py | 42 | 1 | 45 | 47 | OK |
| measure_fresh_reference_gate_2025.py | 34 | 1 | 37 | 39 | OK |
| measure_mc_tables.py | 75 | 1 | 78 | 80 | OK |
| measure_tyre_reference.py | 50 | 1 | 53 | 55 | OK |
| prompt_ab/_common.py | 28 | 1 | 32 | 58 | OK |
| verify_drs_zones.py | 52 | 1 | 56 | none | OK |

- No duplicated snippet; no orphaned `_SCRIPT_DIR`/`ROOT`/`_REPO_ROOT` (each `_SCRIPT_DIR` feeds
  the snippet; each root var feeds `sys.path` and, in `measure_mc_tables.py`/`verify_drs_zones.py`,
  later path construction — `verify_drs_zones.py:227` uses `_REPO_ROOT` for the fastf1 cache dir,
  so the fix is load-bearing there beyond `sys.path`). `ruff check scripts/` clean (would flag
  unused).
- `prompt_ab/_common.py` fallback: `_SCRIPT_DIR = <root>/scripts/prompt_ab` →
  `.parent.parent = <root>`. Matches the old `parents[2]` of the FILE path
  (`[prompt_ab, scripts, root][2]`). Correct for its nesting depth.
- Executed smoke: `uv run python scripts/verify_drs_zones.py --help` renders the argparse help.
- Minor observation (INFO): `verify_drs_zones.py` imports no project module at all, so its
  `sys.path.insert` block is inert scaffolding (pre-existing; the root var itself is still needed
  for the cache path).

### g) `fastf1_extractor.py` docstring — VERIFIED (one attribution imprecision)

Independently executed `json.load` over `notebooks/data_engineering/N01_data_download.ipynb`:
**0 code cells** reference `fastf1_extractor`; exactly **1 markdown cell** (cell 0) carries the
link `- Legacy extraction: [fastf1_extractor.py](../../src/shared/data_extraction/fastf1_extractor.py)`.
Repo-wide grep: no other reference outside `src/shared/README.md` and audit docs; **nothing in the
whole repo imports `src.shared`** (grep for `from src.shared|import src.shared`: only README
mentions). The corrected docstring is factually right.

**INFO caveat:** the new docstring says "Kept per `src/shared/README.md`'s stated reason" — the
README's actual stated reason is that deletion "would break the historical jupytext exports under
`src/strategy/` and `src/vision/`", which for THIS file is itself inaccurate (nothing under either
references it; only the N01 markdown link does). The docstring's own facts are correct; the
attribution slightly launders a stale README claim.

### h) References to deleted things — VERIFIED WITH CAVEAT (a fifth canned-outputs consumer was missed)

- `severity_color`: 0 code references repo-wide (only the two audit .md narratives). ✓
- Old inline `_HAS_MODELS`/`_skip_no_models` definitions: 0 left (see a). ✓
- Canned-outputs body fingerprint `stop_duration_p05=2.2`: exactly **1** occurrence,
  `tests/mc/canned_outputs.py:43`. ✓ No duplicate body anywhere.

**CAVEAT (MEDIUM, the wave's own bug class inside its own fix):** the audit doc enumerates "four
call sites"; there are **five**. `tests/mc/test_tyre_wear_term.py:317` and `:426` still import the
fixture through the OLD indirect path — `from .test_strategy_goldens import _canned_outputs` —
not from `tests.mc.canned_outputs`. It works (the chain resolves to the single body), but the
canned_outputs docstring's claim that the extraction closed the "two different import paths"
problem is not fully true: one indirect path survives. Worse, the two fresh `noqa` comments name
the WRONG consumers:
  - `test_mc_state_helpers.py:26` — `# noqa: F401 -- re-exported for test_mc_is_a_real_decision.py`:
    false; that file now imports directly from `tests.mc.canned_outputs` (line 437). Nothing
    re-imports from `test_mc_state_helpers` anymore; the noqa is also unnecessary (the alias is
    used in-file at 218/245, so F401 cannot fire).
  - `test_strategy_goldens.py:30` — `# noqa: F401 -- re-exported for test_projection_golden.py`:
    false; `test_projection_golden.py:47` imports directly. The ACTUAL surviving dependent is
    `test_tyre_wear_term.py` (317/426), which the comment does not name.
This is precisely [[feedback_verify_the_enumeration_too]] + the wrong-mechanism-comment class: a
reader following either comment will "clean up" the re-export and break `test_tyre_wear_term.py`,
or keep a re-export nobody uses. Fix: point `test_tyre_wear_term.py`'s two imports at
`tests.mc.canned_outputs`, then delete both stale noqa comments.

### j) Commit hygiene — VERIFIED WITH CAVEATS (one broken intermediate tree, two wrong counts)

Read all six full messages against their diffs:

- `7ec75da`, `d732e1d`, `c161d7f`: honest and accurate. `d732e1d`'s verification claims
  (ruff clean, `--help` renders) independently re-executed and confirmed here.
- `906fcb1`: accurate on the None-crash and the dedup; minor: frames `classify_alerts` as the live
  canonical of the severity mapping when it has 0 callers (see e, caveat 2).
- `cd3f63f` (**MEDIUM — broken intermediate commit**): its diff ALSO performs the
  `HAS_TIRE_MODELS`/`skip_no_tire_models` guard migration in the 4 mc files — importing
  `from tests.conftest import ...` — but `tests/conftest.py` **does not exist at that commit**
  (executed: `git show cd3f63f:tests/conftest.py` → "exists on disk, but not in cd3f63f"; the file
  is born in `896d773`, the NEXT commit). At `cd3f63f` the entire `tests/mc/` golden tier fails at
  collection with ImportError. Head of branch is fine, CI (which runs on the head) will be green,
  but `git bisect` across this branch breaks, and the commit message never mentions the guard
  migration it smuggles in. The dependency order between commits 1 and 2 is inverted; the guard
  migration of those 4 files belonged in (or after) the conftest commit.
- `896d773`: discloses the split but with the wrong count — "the **two** already touched for the
  canned_outputs dedup are in the previous commit"; it was **four**
  (state_helpers, strategy_goldens, mc_is_a_real_decision, projection_golden — all verified in
  `cd3f63f`'s diff). Also says "~26 test files" restated the guard; actual total on `dev` is 25
  `_HAS_MODELS` definitions (21 tire + 4 other-artifact). Tilde'd, tolerable — but
  `tests/conftest.py:8`'s "26 copies of the tire check ALONE" (actual: 21) is not (see a).

Structural check that could have made the explicit-conftest-import pattern subtly wrong and did
not: `tests/` and every subdir have `__init__.py`, so pytest imports the conftest plugin as
`tests.conftest` — the explicit `from tests.conftest import ...` binds the SAME module instance,
no double evaluation of `HAS_TIRE_MODELS`. Verified via Glob + pyproject `testpaths`.

### Audit-doc numeric accuracy (cross-cutting, LOW)

The Part-5 narrative carries three unreproducible counts alongside the two already noted
(conftest's "26 copies", commit 2's "two → four"):

- §5.7: "13 Python files in `src/arcade/`, `src/shared/`" — actual second-wave `src/` diff is
  **3** files (`theme.py`, `strategy.py`, `fastf1_extractor.py`; executed
  `git diff dev..HEAD --name-only -- src/`).
- §5.1: "~28 test files" restating the guard — actual `_HAS_MODELS` definitions on `dev`: **25**.
- §5.1's "four consumers" of `_canned_outputs` — actual: **five** (see h).

Verified-consistent counts, for contrast: "21 files" migrated ✓, "4 files different-artifact" ✓,
"62 files in tests/" ruff-format-clean ✓ (re-executed: "62 files already formatted"), full branch
scope = 23 tests + 8 scripts + 3 src + 1 audit doc = 35 files ✓ (matches `git diff --stat`), and
no untouchable zone (`run_simulation_cli.py`, `src/agents/`, `notebooks/`, `legacy/`) appears in
the diff. The project's own standard ("a false claim in an issue is scope") applies: these are
narrative-only errors, but three of the five sit in SHIPPED artifacts (a docstring, a noqa
comment, a commit body), not just the audit doc.

### i) Test suite + lint — executed

- `uv run ruff check tests/ src/arcade/ src/shared/ scripts/` → **All checks passed!**
- `uv run ruff format --check tests/ src/arcade/dashboard/theme.py src/arcade/strategy.py` →
  **64 files already formatted** (tests/ alone: 62).
- `uv run pytest tests/ -q`: this gate's own invocation was still running when its findings were
  handed off. Filled in afterward by the main session, from its own separate, completed
  invocation: **554 passed, 4 skipped, 2 failed**. Both failures
  (`tests/eval/test_ml_recompute_golden.py::test_pace_mae_reproduces_from_featured_laps`,
  `tests/eval/test_registry_golden.py::test_reproduction_matches_overtake_auc_pr`) reproduce
  identically on a clean `dev` checkout with none of this branch's commits applied — confirmed by
  checking out `dev`, stashing all changes, and re-running just those two tests. Pre-existing, not a
  regression from this wave. `tests/eval/` is a directory this wave never touched.

---

## Severity ranking (real problems only)

| # | Sev | Finding | Where |
|---|---|---|---|
| 1 | MEDIUM | Commit `cd3f63f` leaves a broken intermediate tree: 4 files import `tests.conftest` one commit before the file exists — bisect-hostile, and the commit message hides the guard migration it carries | commits 1↔2 |
| 2 | MEDIUM | Fifth `_canned_outputs` consumer missed (`test_tyre_wear_term.py:317,426` still imports via `test_strategy_goldens`), and BOTH fresh `noqa: F401 -- re-exported for X` comments name consumers that switched to direct imports; following either comment breaks or preserves the wrong thing | tests/mc |
| 3 | MEDIUM | `theme.py:61-72` comment claims "Both imported from src.arcade.strategy" — only `classify_action` is; the severity dict is no longer imported there at all. Wrong-mechanism comment in freshly written code | src/arcade/dashboard/theme.py |
| 4 | MEDIUM (pre-existing) | `_FLAG_BG_BY_INTENT` lacks `YELLOW_FLAG_SECTOR` — the exact #398 key — and the path is live (`rcm_events.py:144` → `_SAFETY_FLAGS` → `agent_formatters.py:351` → grey chip). Same file the wave edited | src/arcade/dashboard/theme.py:164 |
| 5 | LOW (pre-existing, now fully orphaned) | `classify_alerts` + `_ALERT_SEVERITY` are a dead island (0 callers repo-wide, on `dev` too); the wave's own rationale treats `classify_alerts` as live | src/arcade/strategy.py:747-771 |
| 6 | LOW | Shipped wrong counts: conftest docstring "26 copies" (actual 21), commit 2 "two" (actual four), audit doc "13 src files" (actual 3), "~28 files" (actual 25), "four consumers" (actual five) | conftest.py:8, commit bodies, audit doc |
| 7 | LOW | Residual 22nd copy of the routing-config path check in `test_pace_orchestrator_hardening.py:247` (fixture-level `pytest.skip`) | tests/audit |
| 8 | INFO | `fastf1_extractor.py` docstring attributes to `src/shared/README.md` a reason the README states differently (jupytext exports that don't reference this file) | src/shared |
| 9 | INFO | `verify_drs_zones.py`'s `sys.path.insert` is inert (no project imports); root var still needed for the cache path | scripts |

None of 1-9 blocks the branch at HEAD: the working tree is coherent, ruff/format clean, behaviour
preserved everywhere I could execute it.

## What I tried to break and could NOT

- **Docstring corruption recovery**: parsed all 24 touched files with `ast.parse`; hunted for
  import STATEMENTS inside every module docstring (regex over `ast.get_docstring` output) — zero.
  The 8 docstrings containing the word "import" are all pre-existing prose.
- **Leftover twin definitions**: regex-swept all touched files for inline `_HAS_MODELS = (`,
  `_MODELS_DIR`, `_skip_no_models = pytest.mark` — zero survivors; live grep over `tests/` confirms
  the only guard definition is `tests/conftest.py`.
- **Behaviour drift in `classify_action`**: reconstructed BOTH pre-fix bodies from the dev diff and
  executed them alongside the post-fix function on None/empty/known/lowercase/unknown inputs —
  identical results everywhere the old code didn't crash; the crash is gone.
- **`.exists()` vs `.is_file()` divergence**: checked the artifact on disk (regular file) and
  reasoned both directions of the only divergent state (path-as-directory) — no real-world state
  where the migration changes a skip decision.
- **Wrong-order or duplicated bootstrap snippets**: scripted scan of all 8 scripts — every snippet
  unique and ahead of both `sys.path.insert` and the first project import;
  `prompt_ab/_common.py`'s fallback resolves to the repo root at its actual depth.
- **Accidental scope creep**: full-branch `git diff --name-only` — exactly 35 files, no
  untouchable zone touched (`run_simulation_cli.py`, `src/agents/`, `notebooks/`, `legacy/` clean).
- **Double-evaluation of the shared guard**: `tests/` is a real package (`__init__.py`
  everywhere), so pytest's conftest plugin and the explicit `tests.conftest` import are the same
  module object.
- **The `fastf1_extractor` claim**: independent `json.load` scan of every notebook cell — the
  corrected docstring survives; zero code-cell references anywhere in the repo.
- **Ruff/format regressions**: re-executed on all touched trees — clean.

