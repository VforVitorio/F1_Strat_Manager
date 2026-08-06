# GATE — PR5 overtake domain (adversarial audit)

- **Date:** 2026-08-05
- **Branches:** parent `fix/overtake-domain-gate` · submodule `src/telemetry` `fix/overtake-domain-nullable`
- **Auditor role:** adversarial gate. Success = finding what is STILL broken. No repo file modified except this report.
- **Prior context (not re-reported):** GATE_DATA_WIRING.md (F13 + N27 notes), GATE_801_ARTEFACTS.md §6, PR3_GP_KEYSPACE_SWEEP.md, PR4_PACE_INPUTS.md.

## Checklist (claims to verify or refute)

- [x] A. 2.5 s training bound — VERIFIED (model + calibrator + split, executed on artefact + `.nb_py`).
- [x] B. 43.1% — re-derived EXACTLY (20,449 / 8,816 / 43.1% on the served frame); `rival_ahead` pure position lookup confirmed at both sites.
- [x] C. `overtake_prob` never reaches MC / decision layer — VERIFIED by exhaustive sweep.
- [x] D. Neutralisation override ordering — VERIFIED; None cannot leak past it.
- [x] E. Band direction — structurally down-only; QUANTIFIED: ≤57 pairs lose a MEDIUM, 0 lose a HIGH (the PR does not carry the number).
- [x] F. **REFUTED IN PART** — arithmetic matches, SERIES does not: 29.4% of in-domain pairs get a different rolling window than N12's rule; 81/38 pairs cross the MEDIUM/HIGH band between rules.
- [x] G. -1 → LightGBM missing — VERIFIED executed (≡ NaN, ≠ cluster 0, no later fill).
- [x] H. Parser — one real defect: cross-call field tearing (executed); "legit 0.000" attack killed by the calibrator floor (0.0018).
- [x] I. Consumers — none crash; `reasoning_tabs.py` unlisted-but-safe; the real miss is the `/situation` route's unscoped frame the gate now rides on (MEDIUM).
- [x] Bug-class hunt — stale twin docstring (`no_llm.py:156`), dead `pd.notna` guard over a NaN→0.0 sentinel collapse, residual 0.0 sentinels on gap/pace, unscoped-route sibling in the same file.
- [x] Fixes audited as new code — `_pair_rolling_features` (the F finding), `_fmt_prob` (clean), `_lap_count` (pre-existing, correct), `__post_init__` (correct; NaN input would band LOW without crashing).

## Findings

<!-- appended as confirmed -->

### Claim A — VERIFIED (executed)

- `.nb_py/N11_overtake_eda.py:233-235`: `gap = abs(row_x["Time_s"] - row_y["Time_s"]); if gap > 2.5: continue` — the builder drops every pair beyond 2.5 s before labelling. Note the builder's gap is `abs(...)`; inference clamps `max(0.0, ...)` (equal for the normal positive case).
- Artefact executed: `overtake_pairs_2023_2025.parquet` = **28,494 rows, max gap exactly 2.5, 0 rows beyond** (2023: 9,230 · 2024: 9,047 · 2025: 10,217).
- **Calibrator**: `.nb_py/N12_overtake_model.py:655-657` — Platt `LogisticRegression` "fitted on val 2024", i.e. the 9,047 2024 rows of the same ≤2.5 s frame. **N12 split**: `temporal_split` (train 2023+2024 / test 2025) over `load_dataset(overtake_pairs_2023_2025.parquet)` — same frame. The 2.5 s bound therefore applies to the model, the calibrator AND the split. Claim A holds in full.

### Claim B — VERIFIED (re-derived independently)

Executed on the SERVED frame (`augment_featured_laps(laps_featured_2025, 2025)`, 22,760 rows, 24 GPs, 0 NaN in Time_s/LapTime_s), N11's pairing rule with no gap filter:
- **20,449 position-adjacent pairs; 8,816 outside 2.5 s = 43.1%** — exactly the implementer's numbers.
- Full-distribution median 2.06 s (matches their "median gap 2.06"). NOTE (LOW): their table line reads as if 2.06/9.11 describe the *outside* subset; the outside subset's own median is **5.16 s**, p90 **15.13 s**. The numbers they printed are of the full adjacent-pair distribution. Cosmetic, not load-bearing.
- Runtime pairing rule verified in code: `run_from_state` (race_situation_agent.py:1595-1600) and `no_llm.py:172-176` both derive `rival_ahead` from `position == driver_pos - 1` with **no gap filter** — same population as the measurement. Claim B holds.

### Claim C — VERIFIED (exhaustive sweep)

`overtake_prob` grep across parent + submodule + tests + docs (full inventory executed): consumers are the N31 prompt (`strategy_orchestrator.py:1632-1633`), threat_level (`__post_init__`), CLI `_add_situation_row`, arcade `format_situation` + `reasoning_tabs._situation_lines` (`_pct(None)` → "—", pre-existing None-safe helper), arcade `strategy.py:736` `_dump_dataclass` (asdict → JSON null), backend `/situation` `_to_dict` (asdict), backend MCP `predict_situation` (`json.dumps`, null passes), webapp `strategy.ts`/`AgentTabs`/`SituationResultView`, chat `toolResultParsing.numOrUndef` (null → undefined → gauge skipped in `MetricsResult.tsx:64`). `_run_mc_simulation` consumes only `sc_prob_3lap` / `vsc_active` / `sc_currently_active` (strategy_orchestrator.py:1408, 1417); backend simulator stores `_sit_out` but nothing reads it. **No numeric path to the MC or decision layer.**

### Claim D — VERIFIED (code path)

`_run_core` (race_situation_agent.py:1731-1734): `raw_overtake_prob = None if parsed["overtake_prob"] is None else round(parsed["overtake_prob"], 3)`, then `effective_overtake_prob = 0.0 if is_neutralized else raw_overtake_prob` — the neutralisation branch selects the literal 0.0 regardless of None; a None can only pass on a green lap. The override note renders via `_fmt_prob` (line 1744), so no TypeError. Ordering: parse → override → construct. Holds.

### Claim E — quantified (executed, the report does NOT carry this number)

Scored all 8,816 out-of-domain 2025 pairs with the real model+calibrator (N12-style featureization; rolling over the unfiltered adjacent series — an approximation of the old serving path, see caveat):
- calibrated p: min 0.002 · median 0.003 · **max 0.554**
- **≥ high_overtake (0.65): 0 pairs** — the old path could never have produced a HIGH from an out-of-domain overtake (calibrator ceiling 0.7659 > 0.65, but the extrapolations top out at 0.554).
- **≥ medium_overtake (0.40): 57 pairs (0.65% of out-of-domain)** — on those laps the old path could report MEDIUM where the new path reports LOW (unless the SC term independently raises it; medium_sc fires on 13.6% of laps). Direction of every change: DOWN. The report's "an unknown cannot RAISE the band" is true; what it does not say is that ~57 real 2025 laps LOSE a MEDIUM they used to get. That is the intended design (those MEDIUMs were extrapolations), but it is a real behavioural delta the PR text does not quantify.
- In-domain reference: 8.10% ≥ 0.40, 3.34% ≥ 0.65.
- Calibrator floor at raw=0: **0.001831 → prints as "0.002"** — a legitimate `P(overtake) = 0.000` string is unreachable, which kills one H attack (see below).

### F — REFUTED IN PART (executed, measured on the served 2025 distribution)

**Claim: "`pace_delta_rolling3` now reproduces N12's rule." It reproduces N12's *arithmetic* (min_periods=1, 2-lap windows, trend sign — verified identical on a clean continuous battle), but NOT N12's *series*. N12's rolling/diff run over the pair's LABELLED series — only laps that survived N11's builder (position-adjacent AND gap ≤ 2.5 AND non-NaN). `_pair_rolling_features` (race_situation_agent.py:455-482) runs over ALL laps where both drivers merely have rows.**

- Synthetic proof (executed): pair freshly caught up, gaps 4.0/3.0/2.0 s over laps 8/9/10. Helper: `rolling3=-1.500, gap_trend=-1.000`. N12's rule (only lap 10 survives the builder → first row of the series): `rolling3=-2.000, gap_trend=0.000`. The divergence hits exactly the just-entered-the-domain battles — the laps the new domain gate makes freshly interesting.
- Measured over all 11,633 in-domain 2025 adjacent pairs (battle series reconstructed per N11's rule, helper series per the code's rule, both re-scored through the real model+calibrator):
  - rolling3 window content differs on **3,425 pairs (29.44%)**; gap_trend previous-lap differs on **2,109 (18.13%)**.
  - |Δ rolling3| mean 0.113 s, p95 0.683 s, max 12.97 s; |Δ gap_trend| mean 0.323 s, p95 1.464 s, max 47.05 s.
  - Calibrated |Δp|: mean 0.0062, p95 0.0294, **max 0.480**; >0.05 on 409 pairs (3.5%).
  - **81 pairs cross the 0.40 MEDIUM band and 38 cross the 0.65 HIGH band between the two rules** — larger than the 57-pair effect the PR treats as the headline of claim E.
- A second series defect folded in: after a position swap-back, the helper includes shared laps where x was AHEAD of y — `_pair_gap_seconds` clamps those to 0.0 (race_situation_agent.py:489) and they enter gap_trend; in N12's frame those laps do not exist for the (x,y) direction at all. N11's gap is also `abs(...)` where the helper clamps `max(0.0, ...)` — same number only when x is genuinely behind.
- Also confirmed: a NaT LapTime on a shared window lap poisons rolling3 to NaN (no guard at :465-467) where N11's builder dropped such laps — **latent**: LapTime_s NaN = 0 in all three featured frames and the FastF1 path uses `pick_accurate()`.
- Severity: **HIGH** as a refutation of the PR's stated claim; MEDIUM in mean served effect (0.0062). The fix is a large improvement over the array-position bug it replaced, but "reproduces N12's rule" is false for 3 in 10 scored pairs, and this is the same train/serve-skew class the fix itself was correcting.
- Corollary: `tests/agents/test_overtake_domain.py:217-258` pins the HELPER's shared-lap series rule (its scenario asserts a trend diffed against a shared non-battle lap), so the test suite now protects the divergent rule; fixing the series rule will rightly require changing that test.

### Claim G — VERIFIED (executed)

Booster `pandas_categorical` = `[['C1'..'C6'], ['C1'..'C6'], [0, 1, 2, 3]]`. Casting `circuit_cluster=-1` with `pd.Categorical(..., categories=[0,1,2,3])` → `isna()=True`, and the prediction is **identical to an explicit NaN** (raw 0.3076 / cal 0.0180 for the probe row) and **different from cluster 0** (raw 0.4402 / cal 0.0472) — so -1 takes LightGBM's native missing path, not the old silent "cluster 0" path. `feat_df` goes straight from the cast to `predict_proba` (race_situation_agent.py:1339-1363) — no fillna in between, and the domain gate returns before the model for OOD pairs. Both call sites (`run` :1531 and `run_from_state` :1618-1624) carry the fix — no unfixed twin.

### Claim H — parser: one real defect found (executed)

- The marker string cannot match the overtake pattern (digits required after `= `) — verified: `[marker only]` → `overtake_prob=None`, gap/pace still parsed from the marker's own `gap=`/`pace_delta=` fields. Tool-never-ran and REFUSED paths also stay None.
- `in (0.0, None)` preserves the old `== 0.0` semantics for the three 0.0-default keys (float `==` inside `in`).
- **FINDING (MEDIUM-LOW) — cross-call field tearing.** Executed: messages `[SC, marker(gap=9.11, pace=0.300), scored(0.412, gap=1.20, pace=-0.150)]` parse to `{overtake_prob: 0.412, gap_ahead_s: 9.11, pace_delta_s: 0.3}` — the probability from the SECOND tool call paired with the gap and pace of the FIRST (declined) call. The old parser locked all three fields to the first call that carried numbers; the None default re-opens `overtake_prob` (and only it) for later messages, so a declined-then-scored sequence emits a `RaceSituationOutput` whose `overtake_prob` and `gap_ahead_s` describe DIFFERENT car pairs ("overtake 41% · gap 9.1s" on a dashboard). Reachable only in LLM mode when the ReAct agent calls `predict_overtake_tool` more than once (retry with another rival after a decline — plausible); the no-llm path makes exactly one call. `race_situation_agent.py:961`.
- The "legitimate 0.000 overwritten by a second message" attack is REAL in code (executed: 0.000-then-0.412 → 0.412) but unreachable in practice: the calibrator floor is 0.001831, printed "0.002" — the tool can never print `P(overtake) = 0.000`. Pre-existing semantics for the other keys.

### Bug-class findings

- **(LOW, wrong-mechanism docstring — the unfixed twin of the parser docstring)** `src/strategy/inference/no_llm.py:156`: "predict_overtake_tool only when a rival ahead is derivable (**parser defaults overtake fields to 0.0 otherwise**)". False on this branch: the parser now defaults `overtake_prob` to None, and every P1/no-rival lap in no-llm mode now emits None (rendered "overtake —"), not 0.0. The module whose behaviour changed most (it drives `f1-sim --no-llm` and the backend simulator) still documents the old sentinel. Exactly the "comment naming the wrong mechanism" class.
- **(LOW, guard that asserts nothing)** The new domain gate's `pd.notna(gap)` clause (race_situation_agent.py:1350) is dead code: `_build_overtake_features:1096` runs `gap_ahead_s = max(0.0, gap_ahead_s)` first, and `max(0.0, float('nan'))` returns **0.0** (executed) — a NaN gap is converted to a fabricated ZERO gap (in-domain, `drs_window=1` on a green lap) before the gate can see it. `(Timedelta - NaT).total_seconds()` → nan (executed), so the path exists in code. Unreachable on today's artefacts (LapTime_s NaN = 0 in all three featured frames — executed; FastF1 path uses `pick_accurate()`), but the guard documents a protection it does not provide, and the NaN→0.0 collapse is a sentinel collision waiting for the first artefact with a NaT lap.
- **(INFO)** `src/arcade/dashboard/reasoning_tabs.py:132` is a consumer the PR's table does not list; it is None-safe by accident (`_pct(None)` → "—", a pre-existing helper). It renders a bare "—" without the "(out of model range)" context its sibling `agent_formatters.format_situation` now gives. Cosmetic inconsistency between the two arcade surfaces.
- **(INFO)** The implementer's own admission checked: the `SituationResult` "would 500 like #788" claim was indeed false and is corrected in both the PR doc and the code comment (`backend/api/v1/endpoints/strategy.py:150-166` states it is documentation-only; verified — no route uses it as `response_model`, all seven declare `StrategyResponse`).
- **(LOW, residual instance of the class the PR eliminates)** `gap_ahead_s` and `pace_delta_s` keep their 0.0 parse defaults (race_situation_agent.py:944-946). On a lap with no rival ahead (P1, or roster gap), the output carries `overtake —` next to a fabricated `gap 0.0s · Δpace +0.00s/lap` on the arcade wall (`format_situation` renders both unconditionally, agent_formatters.py:211-224) — 0.0 is a findable real gap, so "never measured" and "genuinely nose-to-tail" are the same pixel. Deliberate scope choice per the PR's own docstring ("the other three default to 0.0, unchanged"), but it is the same sentinel pattern one field over.

### MEDIUM — the new domain gate runs on the WRONG RACE for every webapp Model Lab / strategy-tab call (pre-existing scoping defect the gate now rides on)

- `POST /situation` (`backend/api/v1/endpoints/strategy.py:1141-1155`) hands `run_race_situation_agent_from_state` the FULL season frame from `require_laps_df(year)` (`backend/utils/laps_cache.py:44-55`, all 24 races). `run_from_state` does not scope by GP (`race_situation_agent.py:1602`: `self.laps_df = _ensure_timedelta_laps(laps_df)`), and `predict_overtake_tool` looks rows up by `(Driver, LapNumber)` only (:1312-1317).
- Executed: on the augmented 2025 season frame, `(VER, lap 20)` matches **21 rows across 21 GPs and `iloc[0]` picks Austin** — frame order, not the user's GP. Same for NOR. So a Model Lab "Situation" run for Qatar computes the gap — and therefore the new UNKNOWN/scored decision, plus every other N12/N14 feature — from **Austin's** telemetry. The webapp surfaces this PR carefully made None-aware (`SituationResultView`, `AgentTabs`) render an honest-looking "No prediction / n/a (beyond 2.5s)" that was measured on another circuit.
- Root cause predates this branch (the scoping lesson landed at `run_lap` and `/recommend`; the per-agent POST routes never got it), and the same file CONTAINS the fix pattern with an explanatory comment for the tyre-eval route (`strategy.py:1019-1026` scopes `gp_df` citing `_scope_laps_to_gp` #429/#480) — the classic one-sibling-fixed-in-the-same-file. The CLI/arcade/no-llm paths are scoped upstream and unaffected. Not re-reported from prior context: none of GATE_DATA_WIRING F13, GATE_801 §6, PR3, PR4 covers route-level row scoping.

---

## Findings ranked

| # | Sev | Finding | Where |
|---|-----|---------|-------|
| 1 | HIGH (claim) / MEDIUM (mean effect) | "`pace_delta_rolling3` now reproduces N12's rule" is false for the SERIES: helper rolls over shared laps, N12 over labelled battle laps. 29.44% of in-domain 2025 pairs get a different window; 18.13% a different gap_trend base; calibrated Δp max 0.480; 81 pairs cross the 0.40 band, 38 the 0.65 band. Hits hardest exactly on freshly-caught-up battles. | `src/agents/race_situation_agent.py:455-482` |
| 2 | MEDIUM | The new domain gate (and every N27 feature) is computed from the WRONG RACE on the webapp per-agent routes: `/situation` passes the unscoped 24-race frame and `(Driver, LapNumber)` lookups resolve to the first GP in frame order (executed: always Austin). Pre-existing defect; this branch's advertised webapp behaviour rides on it, and the scoped sibling with the explanatory comment sits in the same file. | `src/telemetry/backend/api/v1/endpoints/strategy.py:1141-1155` + `race_situation_agent.py:1602` |
| 3 | MEDIUM-LOW | Parser cross-call field tearing: a declined first overtake call locks gap/pace to pair 1 while the None default lets a second call's probability through — output mixes two car pairs. LLM-mode only (needs 2 tool calls). | `src/agents/race_situation_agent.py:961` |
| 4 | LOW | Stale twin docstring: `no_llm.py` still says "parser defaults overtake fields to 0.0 otherwise" — the exact mechanism this branch changed, in the module that drives `f1-sim --no-llm` and the backend simulator. | `src/strategy/inference/no_llm.py:156` |
| 5 | LOW | Dead guard + sentinel collapse: `pd.notna(gap)` in the gate can never see NaN because `max(0.0, nan)` → 0.0 upstream turns an unknown gap into a fabricated zero (in-domain, DRS open). Latent today (0 NaN lap times in all featured frames; FastF1 path picks accurate). | `race_situation_agent.py:1096, 1350` |
| 6 | LOW | Residual sentinel: `gap_ahead_s` / `pace_delta_s` keep 0.0 defaults, so "never measured" renders as `gap 0.0s` beside the honest `overtake —`. Deliberate scope choice, same class one field over. | `race_situation_agent.py:944-946` |
| 7 | LOW | PR table's "median gap 2.06 s, p90 9.11 s" describes the FULL adjacent-pair distribution, not the outside subset (whose median is 5.16 s, p90 15.13 s). | `documents/audits/PR5_OVERTAKE_DOMAIN.md` |
| 8 | INFO | `reasoning_tabs._situation_lines` is an unlisted consumer, safe by accident (`_pct(None)` → "—"), but renders no "(out of model range)" context, unlike its arcade sibling. | `src/arcade/dashboard/reasoning_tabs.py:132` |

## Fix list (by value, then risk)

1. **Scope the per-agent webapp routes to the request's GP** before calling `run_*_from_state` (`/situation`, `/pace`, `/pit`, `/tire`, `/radio` in `backend/api/v1/endpoints/strategy.py`) — reuse the `gp_df` pattern already present at :1019-1026 with the same guard rules as `_scope_laps_to_gp`. Fixes finding 2 and makes every Model Lab number mean what the page says.
2. **Restrict `_pair_rolling_features` to the pair's BATTLE series**: keep only shared laps that were position-adjacent with gap ≤ 2.5 (and drop NaT laps, mirroring N11's dropna). Fixes finding 1; update `test_the_rolling_window_pairs_by_lap_number_not_by_array_position` to the battle-series expectation, and re-run the Lusail parity check.
3. **Close the parser tearing window**: parse gap/pace only from the message whose overtake field was accepted (or lock all three fields together per message). Fixes finding 3.
4. **Update `no_llm.py:156`** to say the parser now yields None when the tool declines or is not called. One line. Fixes finding 4.
5. **Move the NaN honesty up**: replace `max(0.0, gap_ahead_s)` with a NaN-preserving clamp so the gate's `pd.notna` branch actually protects (declining or NaN-scoring instead of fabricating a 0.0 gap with DRS open). Fixes finding 5.
6. Optional: give `reasoning_tabs` the same "(out of model range)" wording as `format_situation`; correct the PR doc's median/p90 sentence.

## What I tried to break and could NOT

- **The parse of the marker string**: the overtake regex cannot match "UNKNOWN" (digits required); gap/pace still parse from the marker; REFUSED and never-called paths stay None. Executed.
- **A legitimate `P(overtake) = 0.000` being overwritten**: unreachable — the Platt floor is 0.001831, printed "0.002". Executed.
- **The neutralisation override with a declined model**: the ternary picks the literal 0.0 regardless of None; `_fmt_prob` renders the note; ordering parse → override → construct is airtight. Code-verified at :1731-1757.
- **`__post_init__` raising the band on an unknown, or suppressing SC terms**: structurally impossible (a term removed from an OR chain); executed over the branch's own tests (13/13 pass) and the SC-term cases; also fed it NaN — bands LOW, no crash.
- **A consumer crashing on None**: swept every consumer in parent + submodule + tests (CLI row, arcade formatter + reasoning tabs + TCP stream via asdict/JSON null, backend `_to_dict`, MCP `json.dumps`, webapp strategy.ts/AgentTabs/SituationResultView/chat `numOrUndef`, MC layer, no-llm engine). None does arithmetic/format on it unguarded. The MC provably never reads it (`_run_mc_simulation` consumes only `sc_prob_3lap`/`vsc_active`/`sc_currently_active`).
- **The -1 sentinel resolving to a real cluster**: cast → NaN ≡ explicit NaN, ≠ cluster 0, no downstream fill. Executed against the real booster.
- **The helper's min_periods / 2-lap-window / trend-sign arithmetic**: matches N12 exactly on a clean continuous battle (executed parity check F3), and the helper's fallback path (`no shared laps`) is unreachable from the tool (current lap always shared) but returns N12's min_periods=1 semantics anyway.
- **The 43.1% measurement**: re-derived independently on the served frame — same numbers to the digit (20,449 / 8,816 / 43.1%).
- **ruff on the changed files**: green. **Webapp `tsc --noEmit`**: 0 errors. **Branch tests**: 13/13. Executed.
