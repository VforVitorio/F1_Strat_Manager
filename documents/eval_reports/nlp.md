# nlp

- harness `6bc5413` · schema v1 · generated 2026-07-11T16:31:09+00:00
- era 2022-2025 · dataset radio + intent labeled CSVs, entity annotations, 2025 RCM corpus · seed deterministic · llm none
- artifacts: roberta_sentiment=`d7d0ada739c6`, intent_head=`7c995ba21bc0`, ner_bert_bio=`9e60de9bf538`

| stage | metric | value | reference | status | detail |
|---|---|---|---|---|---|
| ner | dead_classes | 4.0000 | 0 | flagged | 4 entity types with frozen-eval B-F1 < 0.15 (situation, incident, strategy instruction, track condition); suppressed as untrustworthy, re-measured in #304 |
| sentiment | accuracy | 0.9415 | 0.84 | delta | n=530; full labeled set (train+test); optimistic vs the held-out 0.84 (split not pinned on disk) |
| sentiment | macro_f1 | 0.9153 | 0.75 | delta | n=530; 3-class |
| intent | weighted_f1 | 0.8881 | 0.5934 | delta | n=529; vs deployed test weighted-F1 (full-set optimistic) |
| intent | accuracy | 0.8885 | - | reproduced | n=529; full labeled set (train+test); optimistic vs the published 0.5934 test weighted-F1 |
| intent | macro_f1 | 0.8922 | - | reproduced | n=529; 5-class, no anchor |
| intent | predict_proba_order | 1.0000 | 1 | reproduced | head.classes_=[0, 1, 2, 3, 4] map to intent_names via intent_mapping; 5/5 columns aligned (no swap) |
| ner | entity_f1 | 0.4248 | 0.4151 | reproduced | n=529; micro entity-F1 (P 0.304/R 0.704); full-set optimistic; 4 dead classes flagged separately |
| rcm | coverage | 0.9868 | - | reproduced | n=1515 across 24 2025 races; per-category [Drs 1.000, Flag 1.000, Other 0.971, SafetyCar 1.000]; vs config Flag 1.0/Other 0.928/Drs 1.0/SafetyCar 1.0 (N23 sample differs) |
| alert_precision | precision | - | - | pending | no labeled alert ground-truth on disk (the MoE-routing metric #304 targets; data task) |
