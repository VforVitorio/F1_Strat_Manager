# nlp

- harness `5f1f29d` · schema v1 · generated 2026-07-11T15:38:36+00:00
- era 2022-2025 · dataset radio_labeled_data.csv (fixed set) · seed deterministic · llm none
- artifacts: roberta_sentiment=`d7d0ada739c6`

| stage | metric | value | reference | status | detail |
|---|---|---|---|---|---|
| sentiment | accuracy | 0.9415 | 0.84 | delta | n=530; full labeled set (train+test); optimistic vs the held-out 0.84 (split not pinned on disk) |
| sentiment | macro_f1 | 0.9153 | 0.75 | delta | n=530; 3-class |
| intent | accuracy | - | - | blocked | setfit does not import under transformers 5.3.0 (#303); stage cannot run |
| ner | entity_f1 | - | - | pending | GLiNER/BERT reconstruction not wired this phase |
| rcm | accuracy | - | - | pending | RCM classifier reconstruction not wired this phase |
| alert_precision | precision | - | - | pending | no labeled alert ground-truth on disk (the MoE-routing metric #304 targets; data-collection task) |
