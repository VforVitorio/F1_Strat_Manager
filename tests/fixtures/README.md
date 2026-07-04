# Test fixtures (versioned on purpose)

Tiny, deterministic inputs for the test suite, committed to git as an **explicit exception**
to the "data and models live on Hugging Face, not in git" rule (see the carve-out in the repo
`.gitignore`). This is what lets the engine and contract test tiers run without the 7-8 GB HF
assets that CI runners lack.

Keep everything here small (target **< 150 KB per file**). Large assets stay on Hugging Face.

Intended contents (added as the test tiers land, see `documents/audits/AUDIT_TESTING_QA.md`):

- `mini_race.parquet` — a < 150 KB slice of one race for engine and data-path tests
- `*.sse` — recorded chat SSE transcripts, shared by the Python parser and the SPA's TS parser
- canned `lap_state` JSON and agent outputs for deterministic engine golden tests (MC seed 42)
- a RAG mini-collection for retrieval tests
