# Program completeness review (closing red-team pass)

Adversarial completeness check of the whole Fable program: 16 code audits, 10 research/design docs,
and `IMPLEMENTATION_ROADMAP.md`. Scope: what the program got wrong, missed, or contradicts. Not a
redesign. Where a claim was cheap to verify against the code, it was verified (noted below).

Date: 2026-07-07.

---

## 1. Cross-doc contradictions and inconsistencies

- **The roadmap is stale on the same day it was authored.** Its header says "15 Fable code audits
  and 8 research/design docs"; the program is 16 audits and 10 designs. Concretely, the roadmap
  contains ZERO references to the RAG audit (`AUDIT_RAG_LAYER.md`, epic #318, sub #319-323): not in
  the sprints, not in the linchpin table, not in the P0 register. Yet RAG-01 (season scoping) is the
  query-time complement of 2026-reg F-10, and the RAG eval is supposed to fold into the shared
  `src/strategy/eval/` package that Sprint 3 builds. Same absence for `ECOSYSTEM_DATA_CONTRACTS.md`
  and `AGENT_ORCHESTRATION_FLOW.md` (both landed 2026-07-07): the orchestration doc's "additive v2
  StateGraph on the P2b shared engine + golden parity gate" directly reshapes Sprint 2's biggest
  item and is not sequenced anywhere.
- **Goldens have three proposed owners and no arbiter.** Testing #182 (engine goldens + fixtures),
  ML-eval #206 (`tests/eval/` goldens under the metrics registry), and AGENT_ORCHESTRATION_FLOW
  (golden parity gate for the v2 graph) each propose a golden framework. The roadmap's fix-once list
  names the engine, the eval harness, the timeout, the metric divergences and #166, but NOT goldens.
  Without a named owner, Sprints 2-3 will build two or three overlapping golden suites.
- **The P1 Backend audit is invisible to the program's own index.** Its doc lives in the submodule
  (`src/telemetry/docs/audits/AUDIT_P1_BACKEND.md`) while the other 15 live in the parent
  `documents/audits/`; the roadmap never cites its epic (#53) or findings, even though Sprint 5's
  frontend migration codes against exactly that API surface. Anyone reading `documents/audits/`
  sees 15 of 16 audits and no pointer to the missing one.
- **Copy-count drift on the orchestrator wiring.** P3 Arcade calls `arcade/strategy_pipeline.py`
  the "4th body-copy" of `run_strategy_orchestrator_from_state`; AGENT_ORCHESTRATION_FLOW says
  "dedup the triplicated wiring". 3 vs 4 is not pedantry: the engine extraction's parity scope
  (which call sites must produce identical output) depends on the real count.
- **Pit-wall verdict vs the committed Rival Agent design.** The pit-wall doc's Topic-1 verdict
  orders a patch to `RIVAL_AGENT_DESIGN.md` section 4 (R1-R6 observability refinements; rival
  throttle/brake reclassified as observed-broadcast, excluded from Rival v1 features). The
  committed design contains neither: no R1-R6, no observed-broadcast labeling, zero mentions of
  throttle/brake. Its observability tiers are frozen against the older, stingier contract that
  #282 supersedes. The patch is recorded only in session memory, which future implementers of the
  TFM will not read.
- **Agent count.** The Docs audit rules the canonical count is six sub-agents plus N31 (docs were
  counting circuit clustering as an agent), yet the project's operating docs still say "all 7
  agents" (CLAUDE.md §5) and roadmap prose follows suit. Settle the number once before #213/#214
  rewrite the site, or the rewrite will republish the drift.

## 2. Coverage gaps

- **The voice stack has no owner.** Whisper STT + Edge-TTS is a shipping surface (memory: e2e still
  pending) but no audit covers its correctness, latency, or failure modes. Security touches its
  endpoint exposure and LLM-cost notes blocking `requests` freezing SSE+voice; nobody audited the
  pipeline itself. It is the only live code path with zero findings.
- **Legal/ToS posture of data already published.** Radiogate ruled on FUTURE audio redistribution
  and explicitly said "revisit the 529 already-public MP3s before scaling under the f1stratlab
  brand", but no issue tracks that revisit, and nothing at all audits the existing
  `VforVitorio/f1-strategy-dataset` (telemetry-derived data + those MP3s) before the AEPIA/paper
  spotlight raises its visibility. Low probability, high embarrassment.
- **No propagation path if the paper gate moves the numbers.** Docs #213 reconciles repo docs TO
  the thesis finals; ML-eval #207 may overturn a thesis final (leakage). The two assume opposite
  directions of truth, and nothing owns the one-pass update of thesis errata + IEEE draft + docs +
  README should E-02 verification actually find contamination.
- **No capacity model.** The roadmap's sprints are unbounded "milestone-sized batches" for one
  developer against a fixed 31-jul clock; Sprint 4 alone stacks P0s from five epics (P5, NLP,
  Packaging, Security, DevEx). "Re-sequence freely" is the plan's only resource model.
- **Submodule CI baseline.** PK-03 covers the parent's missing scanner stack; nothing states the
  submodule's CI/security baseline, and Sprint 5 (migration) lands mostly in the submodule.
- Otherwise coverage is genuinely strong: every core subsystem (engine, agents, NLP, RAG, data,
  CLI, Arcade, backend, packaging, security, docs, DevEx, cost, testing, 2026-reg) has a dedicated
  doc, and the untouchable-file rule is applied consistently across all of them.

## 3. Mis-prioritizations

- **The paper track is serialized behind the riskiest refactor, contradicting the roadmap's own
  dependency graph.** #207 (threshold provenance + leakage), #213 (metric reconcile) and NR-08 need
  at most the eval harness, NOT the shared engine, yet Sprint 3 is ordered after Sprint 2 (engine
  extraction). The graph in roadmap §4 shows `#206 -> #207` with no engine edge. One developer, ~24
  days to the AEPIA close: start the paper-gate items now, in parallel with or before the engine.
- **NR-08 is underpriced.** A possible live swap of SetFit intent confidences would corrupt MoE
  routing behavior AND the paper's intent numbers. A 5-minute pickle check that gates Sprint 3
  should be the first action of the implementation phase, not an "opportunistic" quick win.
- **PK-01 is overranked at P0-register #1 given the clock.** The wheel bug is real and verified,
  but nobody installs the published wheel today; E-02 blocks the only hard external deadline.
  E-02 belongs at #1, PK-01 at #2.
- **ML-eval E-01 (#210) is called P0 and scheduled as backlog.** The register lists "validation is
  only 3 case studies" as P0 item 8; the sprint plan defers #210 past Sprint 5. A paper whose
  validation section rests on 3 case studies cannot be strengthened after freeze. Either scope the
  paper's validation claims to what 3 case studies support (a writing decision, cheap, do it now)
  or pull a minimal multi-race replay into Sprint 3. Leaving a "P0" in the backlog with no explicit
  decision is the register's one real rank-vs-schedule mismatch.
- Security #224 in Sprint 4 (before any deploy, before the next tool) is correctly placed. No
  complaint on the migration-last-after-fixes ordering either; it matches the directive.

## 4. Shaky or unverified claims (check before building on them)

Verified during this review (safe to build on): the ML-eval claim that the MC draws pace samples it
never uses (`strategy_orchestrator.py:685`, literally annotated `# noqa: F841` in the code) and the
existence of the NLP dead-branch check (`:524` gates on `alert_intents & {"PENALTY", "WARNING"}`).
PK-01 was verified empirically by its own audit (0 vs 737 submodule files in the wheel).

Still resting on assumptions:

- **NR-08 (highest risk).** Whether SetFit `predict_proba` column order actually mismatches the
  hardcoded intent tuple is unverified. The 5-minute check decides between "live correctness bug"
  and "false alarm"; everything written about it so far is conditional.
- **NR-04's producer half.** The dead-branch line exists, but the claim that NO producer ever emits
  `intent` (alerts carry `event_type` only) needs the producer-side grep before anyone rewires or
  deletes the branch.
- **P5's "~6 GPs/season silently lose radio + compound labels"** is an estimate. Enumerate the
  actual missed-GP list before designing the canonical-race-identity mapping around it; the design
  should be driven by the real miss set, not the count.
- **The six-vs-seven agent count** should be settled against `src/agents/` before the doc rewrite,
  not asserted from the Docs audit alone.
- **LLM-cost's "stalled LM Studio pins a lap ~30 min"**: 600 s timeout x 2 retries is 20 minutes;
  the 30 does not obviously follow. Harmless for the fix (add the timeout regardless) but do not
  republish the 30-minute figure in docs.
- **AGENT_ORCHESTRATION_FLOW's parity gate has an unstated dependency on #166.** Keeping the frozen
  pipeline as "the control arm" requires deterministic no-LLM runs; `--no-llm` has been broken
  since 2026-05-09. The parity gate cannot exist until Sprint 2's #236 lands, which the doc never
  says.

## 5. The single biggest risk

**The AEPIA/paper deadline colliding with the shared-engine linchpin.** Nearly everything (Arcade
#200, P4 #236 and with it #166, the ML-eval regression bed, the orchestration v2 graph, the
pit-wall parity gate) hangs off one large additive refactor executed by one person, and the only
externally-clocked deliverable (the paper's numbers) is scheduled behind that refactor. If the
engine extraction slips or destabilizes mid-July, the fix train AND the paper gate stall at the
same time, and there is no slack before 31-jul. The mitigation is cheap and already implied by the
roadmap's own graph: decouple the paper-gate trio (#207, #213, NR-08) plus an explicit E-01 scope
decision from the engine work and run them first.

---

## Verdict

The program is sound to execute, but not exactly as written. The audits themselves are high quality
(the line-level claims spot-checked here were all accurate) and coverage is near complete. What
needs a reconciliation pass, about half a day, before Sprint 2 starts: fold the RAG audit (#318)
and the two 2026-07-07 design docs into the roadmap; name a single owner for goldens; reorder the
paper-critical track ahead of (or parallel to) the engine extraction and make the E-01 scope call
explicitly; patch or annotate `RIVAL_AGENT_DESIGN.md` with the pit-wall verdict; and run the two
five-minute checks (NR-08 pickle order, NR-04 producer grep) so Sprint planning rests on verified
facts. None of this is a redesign; all of it is bookkeeping the program skipped in its final
48 hours because the last three docs landed after the roadmap was written.
