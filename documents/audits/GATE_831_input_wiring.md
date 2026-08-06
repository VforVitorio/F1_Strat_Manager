# GATE 831 — input-wiring hygiene (PR #831, branch `chore/input-wiring-hygiene`)

**Role:** adversarial gate over CLAIMS about model inputs. No repository file modified except
this report. No OpenAI credit spent (no `--provider openai`, no `profile="rich"`, no
`alert-llm`; only `profile="no-llm"` / free offline paths / local parquet reads).

**Under attack:** `git diff dev...HEAD` — `src/agents/pace_agent.py`,
`src/agents/race_situation_agent.py`, `src/simulation/race_state_manager.py`.

**Prior art the PR argues against:** `GATE_801_ARTEFACTS.md` §9 ranked this bundle LAST,
"zero measured impact". `GATE_DATA_WIRING.md` F2/F3/F5 are the original diagnoses.

---

## Checklist

- [x] **A.** `DriverNumber` was 0 on 100% of replay laps; now the real number; `None` reaches
      the model as NaN, not coerced to 0; no twin builder still passing 0.
- [x] **B.** `FreshTyre` is the set-was-new flag, constant across a stint — verified against
      the DATA, and the disagreement share quantified per race.
- [x] **C.** `tyre_life <= 1` is the right stint-opener test; scrubbed sets starting at
      `tyre_life >= 2`; None survives to the model as NaN.
- [x] **D.** `circuit_cluster` defaulted to 0 on EVERY path (backend / arcade / MCP / tests),
      the SC vector really consumes it, NaN is safe for the booster, and the parquet swap
      (24/24 vs 17/24) reproduces — plus no other consumer of `circuit_cluster_map` depended
      on the pooled values.
- [x] **E.** The measured prediction effect (the gap the PR leaves): per-race lap counts where
      `sc_prob_3lap` / `overtake_prob` / `predicted_lap_time_s` / the action change.
- [x] **F.** The exclusion list: gap-sign hardening, `pace_delta_rolling3` pairing, the two
      docstrings, W-F12, the Vegas note, and the "N15 `TyreLife` already fixed" claim.

---

## Verdicts

**Bottom line: the PR is right about two of its four items and wrong about the only one that moves a prediction.** W-F2/W-F3/W-F5 change model INPUTS on 100%/79.8%/0.06% of laps and change model OUTPUTS on 1 lap in 45 — GATE_801 §9's "zero measured impact" label was correct for them, and the PR body never reports an output number. The `circuit_cluster` change, presented as the one that "is not hygiene", rests on a measurement against the wrong artefact and **regresses** N27 from 24/24 to 17/24 agreement with what N14 actually trained on.

---

### A. `DriverNumber` — claim MOSTLY UPHELD, two unfixed twins, one wrong-mechanism docstring created

**Upheld.** `RaceStateManager.get_driver_state` (`race_state_manager.py:354-417`) emitted no
`driver_number` key on `dev`; `pace_agent.run_from_state` did `d.get("driver_number") or 0`.
Every replay lap (CLI, Arcade, `f1-eval decision-modes`, `/simulate`) served `DriverNumber=0`.
The new emission (`race_state_manager.py:446`) is on the right method, and
`run_from_state` to `run()` to `_build_feature_row` (`pace_agent.py:635`) writes it into the
row that becomes the model input. **Executed:** across all 71 raw races (77,720 rows)
`DriverNumber` is `dtype=object` (strings `'1'`, `'10'`, ...), **0 NaN, 0 non-integer-parseable**,
so `int(r["DriverNumber"])` is safe on the shipped data and never raises.

**`None` genuinely reaches the model as NaN.** `_build_feature_row:665`
`df.apply(pd.to_numeric, errors="coerce")` converts it. `_bootstrap_ci`
(`pace_agent.py:715-763`) does NOT touch it: `noise_cols` is
`{Prev_LapTime, Prev_SpeedST, mean_sector_speed, AirTemp, TrackTemp, TyreLife}`, so a NaN
`DriverNumber` / `Prev_TyreLife` passes through `base.copy()` unperturbed. No coercion to 0
anywhere downstream. `DriverNumber` is also deliberately outside the envelope
(`pace_agent.py:117-121`), so nothing clips it.

**F-A1 (MEDIUM) — the twins that still pass the literal 0.** Two builders of the same
`lap_state["driver"]` dict were not touched:

- `src/strategy/inference/engine.py:364` — `_build_default_lap_state`, `"driver_number": 0`
- `src/agents/strategy_orchestrator.py:2444` — the inline `lap_state=None` fallback, same line

Both feed `run_from_state`, where the new `.get` **cannot rescue them**: the key is present
holding `0`, so `d.get("driver_number")` returns `0`, not `None`. The PR's own justification
("a genuinely missing number must stay None and reach the model as NaN") is defeated exactly
where it is needed. Reachability: every production surface passes a real `lap_state`
(`run_simulation_cli.py:1743`, `arcade/strategy_pipeline.py:47`,
`backend/services/simulation/simulator.py:402,860`, `eval/decision_modes.py:354`,
`mcp_tools.py:620`, `endpoints/strategy.py:1413`), so today these fire only from
`tests/engine/test_engine_memory.py` and `scripts/prompt_ab/gen_inputs.py`. Latent, not dead.

**F-A2 (MEDIUM) — the backend producers still coerce a missing car number to 0.**
`endpoints/strategy.py:518` and `:947` both do
`"driver_number": int(_safe(r.get("DriverNumber", 0)))`, and `_safe` (`strategy.py:373-385`) is
documented as "NaN to 0 ... only for fields where 0 is a legitimate stand-in". For
`DriverNumber` it is not, which is precisely this PR's argument. The sibling `_safe_none`
already exists for exactly this case and is used two lines away for `Position` / `tyre_life`.
Not reachable on today's data (featured_2025 `DriverNumber` is `int32`, range 1-87, 0 NaN), so
LOW impact, but it is the same defect the PR is fixing, in the producer the PR did not open.

**F-A3 (MEDIUM) — a wrong-mechanism docstring, in the Args block whose type annotation this PR
changed.** `pace_agent.py:830`: `driver_number: Car number used to look up TeamID encoding.`
That is false. `_build_feature_row:614` calls `self._encode_categorical(compound, team, gp_name)`
and `TeamID` is derived from `team`; `driver_number` is written straight into the row as a raw
feature (`:635`) and is used for nothing else. The PR edits this parameter's annotation
(`driver_number: int` to `Optional[int]`) at `:797`, adds a `Prev_TyreLife` note four lines
below, and rewrites the `FreshTyre` docstring 300 lines above *specifically because naming the
wrong mechanism "is how a proxy survives review"* — and leaves this one. Same defect class,
same file, same docstring, same PR.

**F-A4 (LOW) — the same Args block now mis-describes `tyre_life`.** `pace_agent.py:833`:
`tyre_life: Laps on current tyre set; drives FreshTyre flag.` After this PR `tyre_life` drives
`FreshTyre` only on the fallback path; on every `run_from_state` call the flag comes from
`fresh_tyre`. The PR created this staleness and did not update the line.

**F-A5 (LOW) — the flat public entry point cannot carry the fix.** `run_pace_agent`
(`pace_agent.py:1074-1130`) has no `fresh_tyre` parameter, so `_run_always_on_agents`
(`strategy_orchestrator.py:1878`) — the `run_strategy_orchestrator` flat path, an **exported**
symbol (`src/agents/__init__.py:99`) documented as THE usage example
(`strategy_orchestrator.py:17`, `docs/pages/agents-api.md:19`, `src/agents/README.md:19`) —
keeps the outlap proxy *and* `prev_tyre_life = race_state.tyre_life - 1`
(`strategy_orchestrator.py:1899`, not even clamped at 0). Confirmed no production caller
(grep: notebooks / docs / tests only), consistent with `SWEEP_present_none_traps.md:144`.

---

### B. `FreshTyre` — claim FULLY UPHELD by the data, and understated

Measured on `data/processed/laps_featured_2025.parquet` (22,760 rows), not on the docstring.

- **Constant across a stint: yes, 1134 / 1134.** Grouping by N04's own
  `['Year','GP_Name','DriverNumber','Stint']`, `FreshTyre.nunique()` is 1 in **every** stint.
  It is the set-was-new flag, exactly as claimed.
- **Disagreement with `int(tyre_life <= 1)`: 17,809 / 22,309 = 79.83% of laps.** Cross-tab
  (rows = old proxy, cols = trained flag): `(0,0)=4,487  (0,1)=17,808  (1,0)=1  (1,1)=13`.
- The trained flag is TRUE on **79.9%** of laps; the old proxy was TRUE on **0.06%** (14 laps in
  the whole season). The served feature was therefore not "a different flag" — it was
  **effectively the constant 0** where training saw 1 four laps in five.
- Per race, worst to best: Lusail 97.68%, Las Vegas 97.11%, Shanghai 95.36%,
  Spa-Francorchamps 94.73%, Silverstone 93.25% ... Melbourne 30.30%. Every race above 30%.
  "Most racing laps" is if anything conservative.

**F-B1 (LOW) — the fallback is unreachable where it is described, and silently wrong where it
is not.** `race_state_manager.py:405` emits `bool(r.get("FreshTyre", False))`, which is never
`None`, so on the `from_state` path `fresh_tyre is not None` is always true and the
`int(tyre_life <= 1)` fallback is dead. On a frame with no `FreshTyre` column the same
expression yields `False` (not `None`), so the agent serves a hard `0` and the fallback still
never fires — the two-arg `.get` default hides the absence instead of surfacing it. Same
`Series.get` / present-but-absent family the repo has logged repeatedly.

---

### C. `Prev_TyreLife` — claim is a TRUE STATEMENT INSIDE A FALSE HEADLINE

`tyre_life <= 1` is **not** the stint-opener test. Measured on `laps_featured_2025.parquet`:

| quantity | value |
|---|---|
| stint openers (first row per `Year/GP_Name/DriverNumber/Stint`) | **1,153** |
| openers with `TyreLife <= 1` (what the new gate catches) | **14 (1.2%)** |
| openers with `TyreLife >= 2` (scrubbed / used sets — the gate MISSES) | **1,114 (96.6%)** |
| rows where trained `Prev_TyreLife` is NaN but the new rule still sends a number | **1,114 = 4.89% of all rows** |
| rows where the new rule sends `None` and training had a number | **0** |
| old rule `max(0, TL-1)` agrees with the trained column | **93.26%** |
| new rule agrees with the trained column | **93.33%** |

The fix is directionally safe (it never invents a `None`) and moves agreement with the trained
quantity by **0.07 percentage points**. `TyreLife` at a stint opener is overwhelmingly `2`
(656 of 1,153) — exactly the "set fitted mid-race that starts at 2 or higher" case the gate was
asked about, and it is the *majority*, not the exception.

**F-C1 (MEDIUM) — the guard is about (nearly) the empty set on the training distribution.** On
the delta model's actual training pool (2023 + 2024, rows with `Prev_LapTime` known, n = 41,821):
`TyreLife <= 1` occurs **0 times** (`TyreLife` min = 3.0) and 0 of 2,343 stint openers have
`TyreLife <= 1`. The trigger condition the PR added never occurs in training and occurs on
0.06% of served 2025 laps.

**F-C2 (MEDIUM) — "NaN is a direction XGBoost learned" is asserted, not measured, and the
measurement contradicts it.** On the same training pool `Prev_TyreLife` is NaN on **34 rows =
0.08%**. A default split direction fitted from 34 of 41,821 rows is not a learned direction in
any useful sense. The claim appears twice (the `_previous_tyre_life` docstring and the
`run_from_state` comment) and is the whole justification for preferring `None` over a number.
It should say what it is: the honest encoding of an absence, whose behaviour under this booster
is essentially untested.

**Correct, and worth recording:** the docstring's "below the trained minimum of 2.0" checks out
— `Prev_TyreLife` min on the trained rows is exactly 2.0. (It is 1.0 on the served 2025 frame,
so the sentence is true of training and slightly false of serving.)

---

### D. `circuit_cluster` — THE HEADLINE IS FALSE, AND THE PARQUET SWAP IS A REGRESSION

This is the finding that matters. Three separate refutations.

#### D-1 (HIGH) — `session_meta` DOES carry `circuit_cluster`, so `get(..., 0)` never fired

The PR comment (`race_situation_agent.py:1288-1291`) states: *"`session_meta` carries `gp_name`
and never `circuit_cluster` on the replay path (measured: its keys are driver / gp_name / team /
total_laps / year), so `get(..., 0)` fired on 100% of laps"*.

Those ARE the keys of the **RaceStateManager's** `lap_state["session_meta"]`. That is not the
dict `_build_sc_features` receives.

`_build_sc_features` has exactly **one** call site: `race_situation_agent.py:1493`, inside
`predict_sc_tool`, and it passes **`agent.session_meta`** — the agent's own internal meta. There
are exactly two assignments of `self.session_meta` (`:1622` FastF1 path, `:1718`
`run_from_state` / replay path) and **both** set
`"circuit_cluster": self.cfg.cluster_for(gp_name, _UNKNOWN_CIRCUIT_CLUSTER)` (`:1633`, `:1729`).
So on `dev` the key was present and resolved on every lap, `get(..., 0)` never fired, and the
`-1` there (`_UNKNOWN_CIRCUIT_CLUSTER`, `:119`) is already the non-colliding sentinel the PR
argues for — with a six-line comment saying so.

The evidence compares the wrong pair of things. This is the repo's own 2026-08-03 lesson
verbatim.

#### D-2 (HIGH) — the 24/24-vs-17/24 measurement reproduces, but against the WRONG artefact

Reproduced exactly (over `laps_featured_2025.Cluster` vs both maps): pooled 17/24,
`circuit_clusters_k4_2025` 24/24; the seven disagreements are Barcelona 3 to 1, Budapest 3 to 1,
Melbourne 2 to 0, Shanghai 0 to 1, Spielberg 3 to 1, Sao Paulo 2 to 0, Zandvoort 2 to 0.

But `laps_featured_2025.Cluster` is **N06's** training column, which is why `pace_agent.py:349`
loads the `_2025` file. **N14 has nothing to do with it.**

`.nb_py/N13_sc_eda.py:88-91` — `load_circuit_clusters` reads **`circuit_clusters_k4.parquet`**
(the pooled file) and attaches it by fuzzy `Location` match (`:94-99`, `:114`). That is what
lands in `sc_labeled_2023_2025.parquet`, which is the file `.nb_py/N14_sc_model.py:77` trains on
and `:180` lists `circuit_cluster` among its features.

**Executed against the trained artefact itself**
(`data/processed/sc_labeled/sc_labeled_2023_2025.parquet`, 3,275 rows, 58 race-events, 2023-2025):

```
TRAINED circuit_cluster agrees with circuit_clusters_k4.parquet       58 / 58
TRAINED circuit_cluster agrees with circuit_clusters_k4_2025.parquet  43 / 58
```

Per race, including the 2025 rows: Melbourne trained **2** (k4_2025 says 0), Zandvoort **2** (0),
Budapest **3** (1), Barcelona **3** (1), Spielberg **3** (1), Sao Paulo **2** (0),
Shanghai **0** (1).

#### D-3 (HIGH) — net effect: the PR turns a 24/24-correct feature into a 17/24-correct one

Because D-1 shows the pre-PR served value was `cluster_for(gp_name)` **over the pooled map**,
and D-2 shows the pooled map is exactly what N14 trained on, the value served on `dev` matched
the trained value on **every** 2025 race. After this PR the SC model is handed the 2025 refit's
label on 7 of 24 races, a categorical level the model associates with a different kind of
circuit. Budapest, one of the two thesis windows, moves 3 to 1.

The sibling agent already encodes the correct principle and the PR contradicts it without citing
it: `tire_agent.py:343-355` deliberately loads the **pooled** file with the mirror-image
measurement — *"it agrees with `laps_tiredeg` on 24 of 24 GPs, where `circuit_clusters_k4_2025`
agrees on only 17"*. Each agent must load the clustering **its own model trained against**:
`pace_agent` to `_2025`, `tire_agent` to pooled, `race_situation_agent` to pooled. The PR moved
the third one onto the first one's file.

**F-D4 (MEDIUM) — a second, quieter consequence of the same swap.** `predict_overtake_tool`
(`race_situation_agent.py:1419-1432`) reads `agent.session_meta["circuit_cluster"]` and casts it
through the N11 booster's `pandas_categorical` levels. The PR did **not** touch that line, but by
changing which parquet `CFG.circuit_cluster_map` loads it changed that value too, on the same 7
races, for a model (N11) whose own cluster provenance the PR never checked. An untouched line
whose input silently moved is the least reviewable kind of change.

**F-D5 (LOW) — the `-1` to NaN change loses a documented convention.** `:1633` / `:1729` carry a
six-line comment explaining that `-1` is *N11's* unknown-circuit code
(`.nb_py/N11_overtake_eda.py:210-212`) and that the Categorical cast turns it into LightGBM's
native missing. `_build_sc_features` now emits `float("nan")` instead, so the two halves of the
same config disagree about how "unknown circuit" is spelled. Unreachable today (all 24 races
resolve through `resolve_gp_key`), so LOW, but it is a new inconsistency, not a removed one.

**On NaN safety (asked, answered):** unreachable today, therefore untested by this change.
`pd.DataFrame([feat])[sc_features]` would give `circuit_cluster` dtype `float64`; LightGBM
handles missing natively for a numeric feature. Whether the SC booster declares
`circuit_cluster` categorical (as the N11 booster does) is not established by this PR, and that
is the case where a `float64` NaN column behaves differently. Nothing in the PR exercises it.

---

### E. The measured effect — the gap filled, and it inverts the PR's own summary

Harness: `profile="no-llm"` (deterministic, zero LLM clients, zero API spend), real 2025 laps
through `RaceReplayEngine` -> `RaceStateManager` -> `run_lap`, `return_agent_outputs=True`.
"OLD" is `dev` emulated in-process by rewriting exactly the four values the PR changed back to
`dev`'s expressions (`DriverNumber -> 0`, `FreshTyre -> int(TyreLife<=1)`,
`Prev_TyreLife -> max(0, TyreLife-1)`, `circuit_cluster -> int(session_meta.get(..., 0))` over
the pooled map). No `dev` checkout, no rebuild. 45 laps x 2 runs across three races, chosen so
one is a **control** where the cluster does not move.

**The emulation is not a no-op** (executed, so the Monza zero is evidence rather than a broken
patch): `rsm.get_driver_state` at Budapest returns `driver_number=16`, `fresh_tyre=True` on laps
34 / 40 / 41 / 45 with `tyre_life` 15 / 21 / 1 / 5 — the old proxy would have said `1` only on
lap 41. The PR's "driver_number=16 and fresh_tyre=True on every lap" verification reproduces.
And `rsm.get_lap_state(40)["session_meta"]` keys are exactly
`['driver', 'gp_name', 'team', 'total_laps', 'year']` — the PR's key list is TRUE of *that*
dict, which is D-1's whole point.

| race (driver) | cluster trained -> served | laps | pace moved >1 ms | `sc_prob_3lap` moved | `overtake_prob` moved | action changed |
|---|---|---|---|---|---|---|
| **Monza** (NOR) — CONTROL | 1 -> 1 (no change) | 15 | **0** | **0** | **0** | 0 |
| **Budapest** (LEC) | 3 -> 1 | 15 | 1 | **15 (100%)** | 0 | 0 |
| **Zandvoort** (NOR) | 2 -> 0 | 15 | 1 | **12 (80%)** | 2 | 0 |

**Read the control first.** Monza changes `DriverNumber` 0 -> 4 (NOR) and `FreshTyre` 0 -> 1 on
essentially every lap of the window, and **not one prediction moves, at any precision**. That is
the honest verdict on W-F2 and W-F3: the inputs were wrong on 100% / 79.8% of laps and the served
model does not read them (gain 0.03% / 0.00%, per `GATE_DATA_WIRING`). `GATE_801` §9's "zero
measured impact" label was **correct** for those two items. The PR's headline — *"two of its four
items turn out to fire on 100% of laps"* — is true about the INPUT and silently reads as being
about the OUTPUT; it never reports an output number, which is exactly why the label survived
being contradicted.

**W-F5 moved exactly one lap in 45, and it is the one lap the gate fires.** Budapest lap 41
(`TyreLife=1`, the stint 3 opener) `-0.0430 s`; Zandvoort lap 24 (`TyreLife=1`) `-0.0440 s`.
Consistent with C: 0.06% of served laps.

**Everything else that moved is the cluster swap, and it moves the SC probability by 40-70%
relative, in the direction away from the trained value.**

Budapest (trained cluster 3, now served 1), selected laps:

```
lap 43   0.0100 -> 0.0170   (+70%)
lap 45   0.0220 -> 0.0370   (+68%)
lap 47   0.0130 -> 0.0210   (+62%)
lap 37   0.0100 -> 0.0150   (+50%)
```

Zandvoort (trained cluster 2, now served 0), and here it crosses the published bands.
Executed with `threat_level` read off the dataclass, `CFG.medium_sc=0.0432`, `CFG.high_sc=0.0864`:

```
lap  sc_OLD   threat_OLD   sc_NEW   threat_NEW   BAND CHANGED
24   0.1760   HIGH         0.0650   MEDIUM       YES
25   0.1650   HIGH         0.0600   MEDIUM       YES
26   0.1650   HIGH         0.0600   MEDIUM       YES
27   0.0570   MEDIUM       0.0530   MEDIUM
28   0.0220   LOW          0.0200   LOW
29   0.0350   LOW          0.0320   LOW
30   0.0300   LOW          0.0290   LOW
31   0.1120   HIGH         0.0380   LOW          YES   <- two bands
```

**4 of 8 laps change `threat_level`, one of them by two bands.** `threat_level` is a
`RaceSituationOutput` field that reaches the N31 prompt and the arcade / dashboard surfaces. In
this window it moves from HIGH to MEDIUM/LOW on a track whose trained cluster the model was
never asked about.

The N30 routing threshold (`CFG.sc_prob_threshold = 0.30`,
`strategy_orchestrator.py:143,603`) is not crossed anywhere in the sample, and **no action
changed on any of the 45 laps**. So the blast radius today is `sc_prob_3lap`, `threat_level`, and
`overtake_prob` (2 laps, via the untouched `predict_overtake_tool`, F-D4 confirmed live) — not
the final recommendation, in this sample.

**Is "no action changed" reassuring? No, and it should not be reported as such.** Three reasons.
(1) The sample is 45 laps of three races; `decision_modes` runs windows over 24 races and the MC
consumes `sc_prob_3lap` directly. (2) The direction is wrong: this is not noise around a correct
value, it is a systematic move away from the value N14 was fitted with, on 7 of 24 races. (3)
`threat_level` already changed, and it is a published output; "the argmax did not flip in my
sample" is a weaker claim than "nothing changed", and the PR makes neither.

---

### F. The exclusion list — one claim FALSE, two STALE, three fine

| item | PR says | verdict |
|---|---|---|
| gap-sign hardening | not done, moves nothing | **CORRECT.** `GATE_DATA_WIRING:250-257`: 0 of 25,215 adjacent 2025 pairs invert, so `max(0,.)` == `abs(.)` on the replay geometry. Still unfixed at `race_situation_agent.py:919-923`. |
| `pace_delta_rolling3` positional pairing | not done, moves nothing | **STALE and WRONG on both halves.** It was fixed in PR 5, twice: `race_situation_agent.py:450-490` now reconstructs N11's membership rule (`_battle_series`), with a docstring narrating both attempts. And it **did** move predictions: `PR5_OVERTAKE_DOMAIN.md:97-118` measures 10.78% of adjacent pairs with different windows, then 29.44% of in-domain pairs after the first fix, calibrated abs-delta max **0.480**, 81 pairs crossing MEDIUM and 38 crossing HIGH. |
| docstring `tire_agent.py:1023` | not done | **CORRECT.** Still says `stint_laps: FastF1 laps ...` for a frame that is the augmented featured frame. Documentation only. |
| docstring `_add_prev_cols` | not done | **STALE.** Already fixed: `tire_agent.py:652-667` now states NaN is kept, cites `N10:176,181`, names the old wrong mechanism and its 0.198 s effect. Harmless direction (claims undone what is done), but it is a stale citation in a PR body arguing from citations. |
| W-F12 (tiredeg msp broadcast) | not done, gate says document | **CORRECT** — `GATE_801` §9 says document, don't change. |
| B1 Vegas dataset-card note | not done | **CORRECT** — documentation only. |
| **N15 `TyreLife` "already fixed in an earlier PR"** | already fixed | **FALSE.** See below. |

**F-F1 (MEDIUM) — the "already fixed" is not fixed; a different defect was.**
What an earlier PR fixed is the **crash**: `row.get('TyreLife', 1)` on a Series returns the
stored NaN, `int(nan)` raises inside the ReAct loop on 451 rows (1.98%) of the 2025 featured
parquet. `_tyre_life_in` (`pit_strategy_agent.py:843-870`) now guards that, with a full
rationale. Good fix, wrong item.

The **wiring** item is the value, and `GATE_DATA_WIRING`'s fix list says it in so many words:
*"align `_tyre_life_in` missing -> 0 with N15"* (`:395`). N15 encodes a missing tyre age as **0**:

```
.nb_py/N15_pit_duration.py:268
out["tyre_life_in"] = out["TyreLife"].clip(upper=50).fillna(0).astype(float)
```

`_tyre_life_in` returns **1**, on the argument that *"a missing tyre age means fresh is the only
defensible read"*. That is a strategy opinion overriding a trained encoding, and it picks the
**colliding** value: `TyreLife == 0` occurs zero times in 2023/2024/2025 (season minimums
2.0/2.0/1.0 — the constant `UNKNOWN_TYRE_LIFE = 0` in `race_state_builder.py` exists for exactly
that reason), so 0 is N15's clean missing code and 1 is a real fresh-set age the model also sees
for real. Sentinel-collides-with-a-legitimate-value, in the file whose docstring argues against
sentinels colliding with legitimate values. Fires on 451 rows (1.98%).

The PR body's parenthetical — *"(`pit_strategy_agent.py` carries the guard and its rationale)"* —
is what makes this survivable: the guard IS there and IS rationalised, so a reviewer who follows
the pointer finds a convincing docstring and stops. Executed-evidence-shaped prose about the
wrong thing.

**F-F2 (LOW) — the PR's own verification bullet is misleading about the case that matters.**
*"`_previous_tyre_life`: opener -> `None`, 2 -> 1, 7 -> 6."* Presented as three cases, but `2` IS
the modal stint opener (656 of 1,153) and it returns `1`, not `None`. The bullet reads as
"openers return None"; measured, 1.2% of openers do.

**F-F3 (LOW) — `303 passed` is TRUE and proves nothing about the change.** Reproduced:
`uv run pytest tests/agents tests/simulation tests/inference -q` -> **303 passed, 1 warning in
62.33 s**. Not one of them asserts the served `circuit_cluster` against the artefact N14 trained
on. The mirror-image guard already exists one directory over —
`tests/agents/test_tire_serving_frame.py:94-122`,
`test_the_agent_reads_the_cluster_family_the_tcn_trained_on`, whose docstring is literally
*"Pooled, not 2025. The two disagree, and the disagreement is the bug"* and which additionally
asserts the two maps still differ so the loader cannot be repointed unnoticed. N27 shipped the
opposite repointing with no equivalent test, and the suite could not see it.

---

## Findings, ranked

| # | sev | finding | file:line |
|---|---|---|---|
| **D-3** | **HIGH** | The cluster parquet swap **regresses** N27: the served value matched N14's trained value on 24/24 of 2025 before, on 17/24 after. Measured: `sc_prob_3lap` moves on 100% of Budapest laps (+40..70% relative) and 80% of Zandvoort laps (-60%), and `threat_level` changes band on 4 of 8 Zandvoort laps, one by two bands. | `race_situation_agent.py:263-268` |
| **D-2** | **HIGH** | The 24/24-vs-17/24 table is measured against `laps_featured_2025.Cluster` (N06's column) and reported as *"the Cluster column N14's training rows actually carry"*. Against the real trained artefact `sc_labeled_2023_2025.parquet`: pooled **58/58**, `_2025` **43/58**. The claim is in the PR body, the commit message, and a source comment. | `race_situation_agent.py:257-262`, commit `58be267` |
| **D-1** | **HIGH** | *"`session_meta` ... never `circuit_cluster` ... `get(..., 0)` fired on 100% of laps"* is false. `_build_sc_features` has one caller (`:1493`) and it passes `agent.session_meta`, which both constructors (`:1633`, `:1729`) populate with a resolved cluster and a `-1` unknown code. The default never fired. | `race_situation_agent.py:1288-1301` |
| **F-F1** | **MEDIUM** | "N15 `TyreLife` already fixed" — the crash was fixed, the value was not, and it went the opposite way to the gate's instruction. N15 trains `fillna(0)`; serving sends `1`. 451 rows (1.98%). | `pit_strategy_agent.py:858-861` vs `.nb_py/N15_pit_duration.py:268` |
| **F-A1** | **MEDIUM** | Twin builders still pass the literal `driver_number: 0`, which the new `.get` cannot rescue (present key). | `engine.py:364`, `strategy_orchestrator.py:2444` |
| **F-A3** | **MEDIUM** | Wrong-mechanism docstring left in the Args block the PR edited: `driver_number: Car number used to look up TeamID encoding` — TeamID comes from `team`. | `pace_agent.py:830` |
| **F-C1** | **MEDIUM** | `tyre_life <= 1` catches 14 of 1,153 stint openers (1.2%); 96.6% start at `TyreLife >= 2`. Agreement with the trained column moves 93.26% -> 93.33%. The trigger occurs 0 times in the training pool. | `pace_agent.py:246-256` |
| **F-C2** | **MEDIUM** | *"NaN is a direction XGBoost learned"* — the training pool has 34 NaN `Prev_TyreLife` rows in 41,821 (0.08%). Asserted, not measured. | `pace_agent.py:246-256`, `:1019-1029` |
| **F-A2** | **MEDIUM** | Backend `/lap-state` producers still coerce a missing `DriverNumber` to 0 via `_safe`, the sentinel this PR argues against; `_safe_none` is used two lines away. | `endpoints/strategy.py:518`, `:947` |
| **F-D4** | **MEDIUM** | The swap silently changed `predict_overtake_tool`'s cluster too (untouched line, moved input). Observed on 2 Zandvoort laps. | `race_situation_agent.py:1419-1432` |
| **F-F2** | LOW | Verification bullet implies openers return `None`; the modal opener (`TyreLife=2`) returns `1`. | PR body |
| **F-F3** | LOW | `303 passed` is true and blind to the change; the mirror-image guard already exists for `tire_agent`. | `tests/agents/test_tire_serving_frame.py:94-122` |
| **F-A4** | LOW | `tyre_life: ... drives FreshTyre flag` is now false on the primary path; the PR created the staleness. | `pace_agent.py:833` |
| **F-A5** | LOW | `run_pace_agent` has no `fresh_tyre`, so the exported flat path keeps the proxy and `tyre_life - 1` unclamped. No production caller. | `pace_agent.py:1074`, `strategy_orchestrator.py:1878,1899` |
| **F-B1** | LOW | `bool(r.get("FreshTyre", False))` can never be `None`, so the fallback is dead where described and silently serves `0` on a frame lacking the column. | `race_state_manager.py:405`, `pace_agent.py:571` |
| **F-D5** | LOW | Unknown-cluster is now spelled `NaN` in `_build_sc_features` and `-1` in the two `session_meta` builders, which carry a comment explaining why `-1`. | `race_situation_agent.py:1301` vs `:1633,1729` |
| **F-F-stale** | LOW | Exclusion list names `pace_delta_rolling3` and `_add_prev_cols` as outstanding; both are already fixed, and the first moved predictions materially when it was. | PR body |

## Fix list, ordered

1. **Revert `race_situation_agent.py:263-268` to `circuit_clusters_k4.parquet`.** It is what N13
   built `sc_labeled_2023_2025.parquet` with (`.nb_py/N13_sc_eda.py:90`) and what N14 trained on
   (58/58, executed). This is the only change in the PR that moves a prediction, and it moves it
   the wrong way.
2. **Add the mirror of `test_the_agent_reads_the_cluster_family_the_tcn_trained_on` for N27**,
   asserting the served map against `sc_labeled_2023_2025.parquet` *and* that the two maps still
   differ. Copy the shape from `tests/agents/test_tire_serving_frame.py:94-122`.
3. **Rewrite the `circuit_cluster` comment** (`:1288-1301`) to say what is true: `session_meta`
   *does* carry a resolved cluster on both paths; the change under discussion (if any survives)
   is re-resolution and the unknown code, not a fired default. Keep `-1`, not `NaN`, so the three
   sites agree; or change all three together and say so.
4. **Decide `_tyre_life_in` on the trained convention, not on strategy** — return `0` to match
   `N15:268`, or write down why the serving convention deliberately differs. Either way, stop
   describing it as "already fixed".
5. **Fix the two `driver_number: 0` twins** (`engine.py:364`, `strategy_orchestrator.py:2444`) to
   `None`, so the PR's own None-not-zero rule holds on the fallback path.
6. **Repoint `endpoints/strategy.py:518,947` at `_safe_none`.**
7. **Correct `pace_agent.py:830` and `:833`** — `driver_number` is a raw feature, not a TeamID
   key; `tyre_life` no longer drives `FreshTyre` on the `from_state` path.
8. **Restate W-F5's scope honestly in the docstring**: the gate is `TyreLife <= 1`, which is 1.2%
   of stint openers; the remaining 96.6% still receive a number where training had NaN. Say
   whether that is deliberate (it can be — `TyreLife-1` is the right answer whenever consecutive
   laps survived) rather than leaving "stint opener" to imply the class.
9. **Refresh the exclusion list**: drop `pace_delta_rolling3` and `_add_prev_cols` (done), and
   delete "none of those moves a prediction" — for `pace_delta_rolling3` it moved a calibrated
   probability by up to 0.480.
10. **Report an output number, not an input rate.** The control race is one extra run and it is
    what turns "fires on 100% of laps" into a defensible sentence.

## What I tried to break and could NOT

- **The `DriverNumber` emission itself.** Swept all 71 raw race parquets (77,720 rows):
  `DriverNumber` is `dtype=object` everywhere, **0 NaN, 0 values that `int()` would reject**, so
  `int(r["DriverNumber"])` cannot raise on shipped data. `pd.notna` on a `str` behaves. The
  featured parquets are `int32`, range 1-87, no zeros.
- **Downstream coercion of the new `None`.** `_build_feature_row`'s
  `to_numeric(errors="coerce")` gives NaN; `_bootstrap_ci`'s `noise_cols` excludes both
  `DriverNumber` and `Prev_TyreLife`, so the perturbation path leaves the NaN intact rather than
  multiplying it into a `sigma`; the envelope excludes `DriverNumber` by design
  (`pace_agent.py:117-121`), so nothing clips it. No path turns it back into 0.
- **The `FreshTyre` semantics claim.** I expected the "constant across a stint" story to be
  approximately true. It is exactly true: 1,134 of 1,134 stints have a single value, on N04's own
  grouping keys. The 79.83% disagreement figure is real and the PR understates it — the old proxy
  was TRUE on 14 laps in a 22,760-lap season.
- **The `_previous_tyre_life` direction.** I looked for rows where the new rule sends `None` and
  training had a number: **0 rows**. It is incomplete, not harmful.
- **The 24/24-vs-17/24 arithmetic.** Recomputed independently against
  `laps_featured_2025.Cluster`: pooled 17/24, `_2025` 24/24, same seven races. The number is
  right; only the artefact it is being compared to is wrong.
- **`driver_number=16` / `fresh_tyre=True` for Leclerc at Budapest, and the `session_meta` key
  list.** Both reproduce exactly as the PR states.
- **`303 passed`.** Reproduced, 62.33 s, zero failures.
- **A crash or a 422 from the new `NaN` cluster.** Could not reach it: all 24 of 2025's races
  resolve through `resolve_gp_key` against the `_2025` map, so `_cluster is None` never happens
  today. The NaN branch is untested rather than broken.
- **An action flip.** Across 45 lap-pairs in three races, `recommendation.action` was identical
  on every single lap. The damage this PR does today stops at `sc_prob_3lap` /
  `threat_level` / `overtake_prob`.
- **`ruff format --check .`** — the follow-up commit `88bd948` does what it says; reproduced at `155 files already formatted`. `src/agents/**` is genuinely outside `[tool.ruff.format] exclude` (pyproject.toml:218), so leaving race_situation_agent unformatted is correct, not an oversight.
