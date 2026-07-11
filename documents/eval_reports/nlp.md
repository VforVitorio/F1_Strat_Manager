# nlp

- harness `1c8ff7c` · schema v1 · generated 2026-07-11T16:13:28+00:00
- era 2022-2025 · dataset radio_labeled_data.csv + intent_labeled_data.csv (fixed sets) · seed deterministic · llm none
- artifacts: roberta_sentiment=`d7d0ada739c6`, intent_head=`7c995ba21bc0`

| stage | metric | value | reference | status | detail |
|---|---|---|---|---|---|
| ner | dead_classes | 4.0000 | 0 | flagged | 4 entity types with frozen-eval B-F1 < 0.15 (situation, incident, strategy instruction, track condition); suppressed as untrustworthy, re-measured in #304 |
| sentiment | accuracy | 0.9415 | 0.84 | delta | n=530; full labeled set (train+test); optimistic vs the held-out 0.84 (split not pinned on disk) |
| sentiment | macro_f1 | 0.9153 | 0.75 | delta | n=530; 3-class |
| intent | weighted_f1 | 0.8881 | 0.5934 | delta | n=529; vs deployed test weighted-F1 (full-set optimistic) |
| intent | accuracy | 0.8885 | - | reproduced | n=529; full labeled set (train+test); optimistic vs the published 0.5934 test weighted-F1 |
| intent | macro_f1 | 0.8922 | - | reproduced | n=529; 5-class, no anchor |
| intent | predict_proba_order | 1.0000 | 1 | reproduced | head.classes_=[0, 1, 2, 3, 4] map to intent_names via intent_mapping; 5/5 columns aligned (no swap) |
| ner | entity_f1 | - | - | pending | BERT-bio entity-level F1 reproduction not wired this phase (#304) |
| rcm | accuracy | - | - | pending | RCM parser reproduction not wired this phase (#304) |
| alert_precision | precision | - | - | pending | no labeled alert ground-truth on disk (the MoE-routing metric #304 targets; data task) |
