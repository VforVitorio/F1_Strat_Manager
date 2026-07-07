# F1 StratLab - Cross-audit implementation roadmap

This document sequences the findings from the **16 Fable code audits** and the **10 research/design docs** into an executable sprint plan. It is the "what to do with all those audits" map: it resolves cross-audit dependencies, groups the fix-once-unblocks-many work, and applies Víctor's priority directive. It does not restate each finding (those live in the audit docs under `documents/audits/` + `documents/research/`, and are digested in the Claude-side memory `project_audit_findings_digest`). Every item points to its GitHub epic / sub-issue.

Authored 2026-07-07 (Claude, no Fable). Sprint 1 (Foundations) is already done.

---

## 0. Priority directive (governs the whole sequence)

**MIGRATION (frontend #25 + pit-wall #281) + AJUSTES/MEJORAS (audit fixes) come BEFORE any NEW features/repos/agents** (radiogate, gridmind, box-bot, Rival Agent build, real-time consumer, pitlab). Migration and fixes first; new features last.

One deadline overrides pure ordering: the **IEEE paper + AEPIA award (cierre 31-jul-2026)** need the eval/metrics work done early, so the paper-critical track (ML-eval Ph1-2 + Docs metric reconcile) runs near the front even though it is "validation," not "migration."

---

## 0b. Reconciliation update (2026-07-07, from `PROGRAM_COMPLETENESS_REVIEW.md`)

A closing red-team pass found this roadmap was written before the last three docs landed. Deltas that govern the sprints below:

- **Count**: the program is **16 code audits + 10 designs** (not 15 + 8). Landed after this roadmap: the **RAG audit** (epic #318, sub #319-323), **ECOSYSTEM_DATA_CONTRACTS**, **AGENT_ORCHESTRATION_FLOW**. The **P1 Backend** audit doc lives in the submodule (`src/telemetry/docs/audits/AUDIT_P1_BACKEND.md`), not `documents/audits/`.
- **Paper track decoupled (deadline fix)**: ML-eval #207 (leakage) + Docs #213 (metric reconcile) + NR-08 need only the **eval harness**, NOT the shared engine. Run the Sprint-3 paper track **in parallel with / not blocked by** Sprint 2 - AEPIA closes 31-jul and the shared-engine refactor is the riskiest single item. NR-08 is a **do-first** 5-minute check, not "opportunistic". ML-eval E-01 needs an explicit scope call (minimal multi-race protocol now for the paper vs full protocol later as the TFM baseline).
- **Goldens have ONE owner**: **Testing #182** owns the golden fixtures; ML-eval #206 and the agent-flow v2 parity gate **consume** them, they do not create parallel goldens.
- **Shared engine = the v2 graph**: Sprint 2's shared-engine extraction follows the `AGENT_ORCHESTRATION_FLOW.md` additive `StateGraph` design (multi-agent architecture kept intact), coding against the fixed `ECOSYSTEM_DATA_CONTRACTS.md` schemas.
- **RAG #318 into the sprints**: RAG Phase 3 (eval) folds into the Sprint-3 shared eval harness; RAG-01/RAG-02 (season scoping + grounding) into Sprint 4 (data integrity).
- **Two coverage gaps to own** (not yet audited): the **voice stack** has no audit owner; and the **legal/ToS posture of the already-published HF dataset** (incl. the 529 radio MP3s radiogate said to revisit) needs a call before scaling.
- **Two 5-minute checks first** (verify-before-implement): NR-08 (SetFit `predict_proba` column order) and the P2b torch-thread-safety test that gates the parallel-fan-out latency win.

---

## 1. Cross-cutting linchpins (fix once, unblocks many)

These are the highest-leverage items because several audits depend on them. Do them first.

| Linchpin | What | Unblocks | Home |
|---|---|---|---|
| **Shared inference engine** | Extract the additive `src/strategy/inference/engine.py` (P2b F10 design) so CLI, Arcade and backend share ONE strategy code path | P3 Arcade #200 (kills the 4th body-copy), P4 CLI #235/#236 (duplicate delegates to it), ML-eval regression bed, `--no-llm` #166 | P2b #169 |
| **`--no-llm` #166** | The 3-tuple/2-tuple unpack crash (broken every lap since 2026-05-09) | P4, DevEx, ML-eval, LLM-cost all reference it; blocks a zero-cost install verify | P4 #236 (on the shared engine) |
| **Shared eval harness** | One `src/strategy/eval/` package (metrics registry + goldens) | ML-eval (#206-210) AND NLP (#304) both build on it | ML-eval #206 |
| **Test fixtures / goldens** | FakeOpenAI stub (#181) + engine goldens + fixtures (#182) | The shared-engine extraction + the eval harness + the #180 spy | Testing #179 |
| **LM Studio / provider timeout** | No request timeout anywhere (agents 7 sites + backend `DEFAULT_TIMEOUT=None`) | Same one fix closes Security S-5 (#226) and LLM-cost L-1 (#263) | LLM-cost #263 + Security #226 |
| **Metric divergences** | Pace 0.392 -> 0.4104, sentiment 87.5% -> 0.84 | Surface in BOTH Docs (#213) and ML-eval (#207); fix once, propagate | Docs #213 |

---

## 2. The real P0 register (deduped across all audits)

The must-fix-soon list, in rough impact order:

1. **PK-01 (Packaging #289)** - the released wheel omits the `src/telemetry` submodule, so `f1-streamlit` is dead on any published install. Blocks shipping.
2. **ML-eval E-02 (#207)** - test-set contamination of the paper's headline numbers cannot be ruled out. **Blocks the IEEE paper freeze** (AEPIA deadline).
3. **NR-02 (NLP #305)** - `sc_currently_active` is stateless, so the Safety-Car override drops mid-SC-stint. Real strategy-quality bug under SC.
4. **Security S-1/S-2 (#224)** - no auth + open `/mcp` + prompt-injection reaches tool execution. Must be built BEFORE any deploy and BEFORE adding any write/export tool.
5. **P5 F-01/F-02 (#243/#244)** - no canonical race identity (~6 GPs/season silently run with no radio + no compound labels) + no data validation.
6. **NR-01 (NLP #304)** - no reproducible NLP eval; alert precision that MoE routing depends on never computed.
7. **#166 (P4 #236)** - `--no-llm` crashes every lap.
8. **ML-eval E-01 (#210)** - validation is only 3 case studies; no systematic multi-race protocol (also the Rival Agent TFM baseline).

Quick wins (hours, do opportunistically): **PK-09 #296** (gate the network test -> un-red `dev` CI), **NR-08 #303** (5-min SetFit `predict_proba` column-order pickle check, possible live confidence swap), **Docs #212** (3 broken copy-paste commands).

---

## 3. Sprint sequence

Sprint 1 (Foundations) is done. Sprints below build on it. Sprints 2-3 are largely parallelizable if there is more than one pair of hands; otherwise run in order.

### Sprint 2 - Foundation + quick P0s (unblocks the most)
Goal: the shared engine + green CI + fixtures, so everything downstream can proceed.
- **Testing #181** (FakeOpenAI stub) + **#182** (engine goldens + fixtures) - the current next step; also unblocks the #180 spy.
- **P2b #169** - extract the additive shared inference engine `src/strategy/inference/engine.py`.
- **P4 #236** - wire the CLI duplicate to the shared engine -> **closes #166** (`--no-llm`), kills the double per-lap inference.
- Quick wins: **PK-09 #296** (un-red CI), **NR-08 #303** (pickle check), **Docs #212** (broken commands), **LLM-cost #263 + Security #226** (the provider timeout, one fix two homes).

### Sprint 3 - Paper-critical validation (AEPIA / IEEE deadline track)
Goal: clear or correct the paper's headline numbers before freeze. Deadline-gated (31-jul).
- **ML-eval #206** (metrics registry, on the shared eval harness) + **#207** (threshold provenance + leakage verification = the E-02 paper blocker).
- **Docs #213** (reconcile published metrics to thesis/IEEE finals - fixes the same divergences).
- **ML-eval #208** (orchestrator MC + routing eval, deterministic/no-LLM) if time allows.
- **NLP #304** (NLP eval harness) rides the same `src/strategy/eval/` package - fold in here since it shares infra.

### Sprint 4 - Data integrity + shipping reliability + security boundary
Goal: the system runs correctly on real data, ships correctly, and is safe to expose.
- **P5 #243** (canonical race identity) + **#244** (data validation) - the data P0s.
- **NLP #305** (RCM correctness sprint - the stateful `RaceControlStateTracker` fixing NR-02 SC-drop, parser superset, dead PENALTY routing branch).
- **Packaging #289** (fix the release wheel PK-01) + **#290** (sync-uv-lock) + **#291** (parent security-scanner stack).
- **Security #224** (gate the surface: auth + `/mcp` + tool allowlist) - before any deploy.
- **DevEx #252** (unbreak the 3 quickstarts) + **#253** (dependency hygiene).

### Sprint 5+ - MIGRATION (the priority-directive product goal)
Goal: the Streamlit -> web migration and the pit-wall, reusing one React stack.
- **Pit-wall #282** (observability contract - also feeds the Rival Agent) + **#283** (FastAPI WS relay data plane).
- **Frontend migration #25** (submodule `F1_Telemetry_Manager`, its own 5 sprints S0-S5).
- **Pit-wall #284** (read-only dashboard in the SPA) + **#285** (migrate agent cards to web + retire the Qt windows -> executes the P3 #199 re-scope, drops #203 D.1/D.2).
- **Arcade P3 #200/#201/#202** (the shared-engine decoupling + real weather/flags/provider fixes) - #200 rides Sprint 2's shared engine.

### Backlog (audit polish, after the above)
- NLP #306 (robustness) / #307 (model-refresh bars). LLM-cost #262/#264/#265/#266. P2 loading #175-177. P2b remaining. Packaging #292-295/#297. DevEx #254-257. ML-eval #209/#210. Pit-wall #286/#287.

### NEW FEATURES - LAST (per the directive)
Only after migration + fixes:
- **2026-reg retraining #189** - very future, hard-gated on the measurement layer (ML-eval + the eval harness must exist first). This is the biggest future block.
- **Ecosystem repos** (all designed, docs in `documents/research/`): pitlab (Studio), radiogate (corpus + picaresca), gridmind (LoRA), real-time OpenF1 consumer, box-bot (multi-platform bot). Dependency order among them: real-time consumer -> (gridmind ~parallel) -> box-bot; radiogate's auto-labeler can use gridmind; pitlab wraps the training extracted by 2026-reg Phase 0.
- **Rival Agent (the TFM)** - forward design done (`RIVAL_AGENT_DESIGN.md`); build when the master starts. Its Phase-0 data need (per-lap gap from `intervals.parquet`) is surfaced by both P5 and the pit-wall observability contract (#282), so it is partly pre-built by the migration track.

---

## 4. Dependency graph (the short version)

```
Testing #181/#182  ─┐
                    ├─> P2b shared engine #169 ─┬─> P4 #236 (closes #166)
                    │                           ├─> Arcade #200 (kills 4th copy)
                    │                           └─> ML-eval regression bed
ML-eval eval harness #206 ─┬─> ML-eval #207 (paper gate, E-02)
                           └─> NLP #304 (shares src/strategy/eval/)
Pit-wall observability #282 ─┬─> Pit-wall dashboard #284/#285
                             └─> Rival Agent (TFM) data need
2026-reg #189 ──requires──> ML-eval measurement layer (eval harness + calibration)
real-time consumer ──feeds──> box-bot ; gridmind ──phrases──> box-bot
```

Fix-once items (do NOT duplicate the work across the audits that mention them): the shared engine, the eval harness, the provider timeout, the metric divergences, #166.

---

## 5. How to use this

- Each item is an existing GitHub sub-issue; a sprint = a milestone-sized batch of them, landed one PR at a time (issue-first, single-concern PRs, `Closes #N`), per the repo's flow (feature branch -> PR -> `test`/`dev` -> `main`).
- Re-sequence freely, but respect the linchpins (Section 1) and the priority directive (Section 0). The only hard external clock is the AEPIA/paper deadline on the Sprint-3 track.
- Full per-finding detail: the audit docs in `documents/audits/` and the designs in `documents/research/`.
