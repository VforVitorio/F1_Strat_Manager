# Contributing

Short guide for anyone cloning the TFG to experiment, fix a bug, or
propose a change.

## Development setup

```bash
git clone https://github.com/VforVitorio/F1_Strat_Manager.git
cd F1_Strat_Manager
git submodule update --init --recursive     # src/telemetry/ is a submodule
uv sync                                      # installs every dependency
cp .env.example .env                         # add OPENAI_API_KEY here
```

Run once to pre-populate the data cache on first launch:

```bash
python -c "from f1_strat_manager.data_cache import ensure_setup; ensure_setup(show_progress=True)"
```

Three entry points after install (`pyproject.toml::[project.scripts]`):

| Command | What it runs |
|---|---|
| `f1-sim` | CLI strategy simulation with Rich live panel |
| `f1-arcade --strategy` | 2D replay + PySide6 dashboard + telemetry |
| `f1-streamlit` | Post-race analysis + chat UI |

## Code style

- **Classes for stateful logic, pure functions for stateless helpers.**
  One responsibility per helper; if a function passes 50 lines or mixes
  concerns, split it.
- **English only** in source, docstrings, comments, and commit
  messages.
- **Prose docstrings** explaining WHY + WHAT and what each field
  enables for downstream code. No code examples inline.
- **No floating logic at module level** — only imports, setup,
  constants. Anything else belongs inside a function or class.
- **Type hints everywhere** on public function signatures; annotate
  variables only when the type is non-obvious.
- **Comments only when the WHY is non-obvious** — hidden constraints,
  subtle invariants, workarounds. Do not narrate what well-named code
  already says.

The conventions are enforced informally by review, not by a hard
linter pipeline. `ruff` and `mypy` are configured in `pyproject.toml`
with exclusions for `legacy/`, `notebooks/`, and the submodule paths;
they exist to catch regressions, not to style-police.

## What NOT to touch

Some code carries hard rules set by the TFG author:

- **`scripts/run_simulation_cli.py`** — the TFG's PMV (first working
  CLI). Duplicate before modifying; do not refactor in-place.
- **`src/agents/` internals** — stable contract for the CLI + Streamlit
  + Arcade paths. Additive entry points are welcome (see the verbose
  variant in `src/arcade/strategy_pipeline.py`), but do not refactor
  existing ones.
- **`notebooks/**`** and **`legacy/**`** — exploration / historical
  archive, different conventions.

## Platform safeguards (Windows)

A few load-bearing patches live near the top of the CLI to keep
`f1-sim` / `f1-strat` usable on Windows hosts. **Do not remove them
without testing on Windows first** — they paper over real issues that
only surface there:

- **`threading.excepthook` filter** in
  [`scripts/run_simulation_cli.py`](scripts/run_simulation_cli.py).
  Whisper / torch / triton fall back to subprocess JIT or ffmpeg
  decoding paths whose stderr is **cp1252** on the Windows console
  host. Python's `subprocess._readerthread` decodes that as UTF-8 and
  crashes mid-byte (`UnicodeDecodeError: byte 0x82`). The parent loop
  is unaffected, but the traceback floods the Rich live panel. The
  hook swallows **only** `UnicodeDecodeError` whose stack passes
  through `_readerthread`; every other thread exception goes to the
  default hook unchanged.

- **`KeyboardInterrupt` wrapper** on the `main()` of both
  `run_simulation_cli.py` and `scripts/f1_cli.py`. Exits with status
  130 and prints a single italic *Interrupted.* line so Ctrl+C in a
  Rich Live render does not leak a stack trace through the panel
  borders.

- **`soundfile` decode path** in
  [`src/nlp/radio_runner.py`](src/nlp/radio_runner.py)
  `WhisperTranscriber.transcribe`. We avoid `librosa.load` because on
  Windows it can fall back to the `audioread` backend, which spawns
  ffmpeg with the same cp1252 / utf-8 reader-thread issue. Decoding
  the OpenF1 MP3 corpus through libsndfile + `librosa.resample`
  bypasses that fallback entirely.

Linux / WSL hosts emit UTF-8 from the same subprocesses, so none of
these safeguards trigger there. They are no-ops on POSIX.

## Pull request checklist

- [ ] Branch off `dev` (`main` is release-only).
- [ ] `pytest tests/ -x` green.
- [ ] If you touched `src/telemetry/*`, commit inside the submodule and
      bump the submodule pointer in the parent repo.
- [ ] `ROADMAP.md` and the relevant `docs/` file updated when behaviour
      changes.
- [ ] One logical change per commit; imperative subject line; no
      co-authored trailers unless you actually pair-programmed.
- [ ] If you added a new sub-agent output, update
      `docs/agents-api-reference.md`.

## CI pipeline

Three parallel jobs run on every push and PR (`.github/workflows/ci.yml`):

| Job | Installs | Runs |
|---|---|---|
| `lint` | nothing — ruff via `uvx` | `ruff check .` + `ruff format --check .` |
| `typecheck` | `uv sync --extra dev` (mypy + project deps, no voice extras) | `mypy src/rag/` |
| `test` | `uv sync --all-extras` (full ML/voice/arcade stack) | `pytest -v` |

All jobs share uv's wheel cache (`enable-cache: true`, keyed off
`uv.lock`), so the cache only invalidates when the resolved graph
actually changes — cosmetic edits to `pyproject.toml` (re-ordering,
tool config, comments) reuse the wheel store. Sync calls use
`--frozen` to skip resolution and install the locked versions
directly. The `typecheck` job additionally caches `.mypy_cache/` so
incremental runs only re-check files whose hash has changed.

**On `uv.lock`:** the lockfile IS committed. Bumping a dependency
manually means running `uv lock` locally (or letting `uv add` /
Dependabot do it for you) and committing the updated lockfile
alongside the `pyproject.toml` change. CI runs `uv sync --frozen`,
which fails when the lockfile and pyproject disagree — that is the
intended early-warning when someone forgets to re-lock. Subsequent runs drop "Install dependencies" from ~60s to
<10s.

A `concurrency:` block at the top of the workflow cancels in-flight runs
when a newer commit lands on the same ref. This was added after
release-please's back-to-back PR refreshes piled 4+ runs into the queue
at once and stranded them for 20+ minutes.

### PR labels

Every pull request gets auto-tagged with one or more `area:` labels by
`.github/workflows/labeler.yml` (path-based via `actions/labeler@v5`).
Dependabot PRs additionally get their label set by `dependabot.yml` so
the tag is in place the moment the PR opens — no need to wait for the
labeler workflow to run.

| Label | Triggered by | Quick risk read |
|---|---|---|
| `area: codebase` | `src/`, `scripts/`, `notebooks/` | Default to careful review — touches product code |
| `area: deps` | `pyproject.toml`, `uv.lock`, pip Dependabot | Low-medium; `test_dep_imports.py` is the safety net |
| `area: ci-cd` | `.github/workflows/`, `.github/dependabot.yml`, `.github/labeler.yml`, GitHub Actions Dependabot | Low impact on product; max damage is breaking the pipeline |
| `area: docs` | `docs/`, root `.md` files | Merge and forget |
| `area: tests` | `tests/` | Low impact on shipping code; useful diff signal |

PRs that combine labels (e.g. `area: codebase` + `area: deps`) get more
attention than single-area PRs, since they couple a code change with a
dependency move. The labeler workflow re-evaluates on each push, so
labels stay in sync with the actual file diff over the life of the PR.

To add a new label, create it via `gh label create "area: <name>" --color <hex>`
first, then add a matching block to `.github/labeler.yml`. Labels do
not auto-create — the workflow silently skips entries pointing to
non-existent labels.

### Dependency-bump safety net

`.github/dependabot.yml` opens weekly PRs whenever an upstream library
publishes a release that does not fit the upper bound declared in
`pyproject.toml`. Two layers guard the project against silently
accepting a bump that breaks something:

1. **Major bumps blocked on core ML libs** — `numpy`, `pandas`,
   `scikit-learn`, `lightgbm`, `xgboost`. Dependabot will still propose
   minor and patch bumps; majors must be reviewed and applied manually.
2. **`tests/test_dep_imports.py`** runs on every CI invocation (including
   Dependabot PRs). It is organised in three tiers:
   - **Tier 1** — exercises the actual API surface of the most critical
     dependencies (fit/predict on a tiny matrix, parquet round-trips,
     tokeniser encode/decode). Catches breaking changes that a plain
     `import` would not see.
   - **Tier 2** — parametrised plain-import smoke for every other
     declared dependency. Catches binary / DLL / wheel breakage at
     install time.
   - **Tier 3** — pins specific API shapes that have bitten the project
     before (`huggingface_hub.snapshot_download` kwargs, `langchain_core`
     import paths, `np.bool_` alias). Grows whenever a new upstream
     incident reveals a fragile call site.

## Issue templates

File an issue via the GitHub UI. Three templates are available under
[.github/ISSUE_TEMPLATE/](.github/ISSUE_TEMPLATE): bug report, feature
request, data issue.

## Related reading

- [`README.md`](README.md) — project overview.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — one-page topology.
- [`INSTALL.md`](INSTALL.md) — deep-dive install per surface.
- [`ROADMAP.md`](ROADMAP.md) — release plan and completed phases.

## Commit message convention

This repository is now versioned automatically by
[release-please](https://github.com/googleapis/release-please). The bot
watches every push to `main`, reads the commit subjects, and opens a
`chore: release X.Y.Z` pull request whenever it detects bumpable commits.

Commit subjects must follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

| Prefix          | Effect on the next release           |
|-----------------|--------------------------------------|
| `feat:`         | minor bump (`X.Y.Z` -> `X.Y+1.0`)    |
| `fix:`          | patch bump (`X.Y.Z` -> `X.Y.Z+1`)    |
| `feat!:` / `fix!:` or a `BREAKING CHANGE:` body line | major bump (`X.Y.Z` -> `X+1.0.0`) |
| `chore:`, `ci:`, `docs:`, `refactor:`, `test:`, `style:`, `build:` | no bump, still listed under the right CHANGELOG section if not hidden |
| `bench:`, `eval:`, `perf:` | no bump, surfaced in the CHANGELOG under their own section |
| `lint:`         | no bump, hidden from the CHANGELOG    |

Scopes are optional but encouraged for clarity, e.g. `feat(orchestrator): ...`,
`fix(rag): ...`, `bench(whisper): ...`. Subjects should stay under 72
characters; body text is free-form and goes after a blank line.

Squash-merging on GitHub edits the squash commit subject — make sure
that subject still follows the convention, otherwise release-please
will miss the bump.

See [the release-please config](release-please-config.json) for the
exact section mapping used by this repo.
