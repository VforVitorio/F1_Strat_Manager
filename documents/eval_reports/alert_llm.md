# alert_llm

- harness `b727baa` · schema v1 · generated 2026-07-11T17:47:16+00:00
- era 2022-2025 · dataset radios_raw.csv unlabeled subset · seed deterministic · llm openai/gpt-4.1-mini
- artifacts: none

> PROXY METRIC - LLM-judged, not ground truth. Needs a human-review pass before any paper claim. The paper-grade alert precision is the gold-based one in the nlp report.

| metric | value |
|---|---|
| sample size (unlabeled) | 80 |
| predicted alerts | 16 |
| judged genuine | 0 |
| proxy precision | 0.0000 |

PROXY: 0/16 predicted alerts judged genuine by gpt-4.1-mini; LLM labels are not ground truth - needs human review before any paper claim

**Finding**: the unlabeled radios are dominated by garbled Whisper transcripts (the subset excluded from hand-labeling), so this proxy reflects transcript noise, not alert precision - a low value here is expected and NOT informative. Use the gold-based alert precision (0.9185, nlp report) for the paper; a meaningful LLM-judge would need a curated unlabeled set.
