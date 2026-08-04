# GATE — Data wiring: training-time producers vs inference-time construction

**Date:** 2026-08-04 · **Ref:** `main`/`dev` @ `73788e0b` · **Gate role:** adversarial, read-only (this file is the only write).

**Question:** feature by feature, for every model the system runs on a 2025 race
(`RaceReplayEngine.replay()` → `RaceStateManager.get_lap_state()` → `run_from_state()`),
is the value inference constructs the SAME QUANTITY the producer notebook computed at
training time? Same quantity, same units, same rule — not "plausible", not "metric reproduces".

**Producers read:** `.nb_py/N01, N04, N06, N09/N10, N12, N14, N15, N16` (plain-`.py` notebook exports).
**Consumers read:** `src/agents/pace_agent.py`, `tire_agent.py`, `race_situation_agent.py`,
`pit_strategy_agent.py`, `race_state_builder.py`, `src/simulation/race_state_manager.py`.

Known-fixed priors (NOT re-reported unless the fix itself broke something):
`mean_sector_speed` speed-trap swap (#797), `LapsSincePitStop`←TyreLife (#800),
weather columns absent from published artefact (#782), season-broadcast `mean_sector_speed` (#784 era).

Findings are appended AS CONFIRMED, each with producer `file:line`, inference `file:line`,
and a divergence measured on real 2025 laps.

---

## Audit checklist (updated as worked)

- [x] N06 lap time — 25 features (`pace_agent.py`) — F1-F7, verified list §"could NOT"
- [x] N26 tire degradation TCN (`tire_agent.py`) — F8-F12, 42 columns diffed vs artefact
- [x] N27 overtake + SC (`race_situation_agent.py`) — F13 + notes (SC feature internals vs
      N13/N14 producers only spot-checked: weather sourcing, cluster default, thresholds
      already covered by the #450 audit — residual risk noted, not exhausted)
- [x] N28 pit: N15 duration + N16 undercut (`pit_strategy_agent.py`) — F14 + verified notes
- [x] Cross-cutting: keyspace family (F6/F14), manifest-vs-training (F1/F8), artefact
      disagreements (F9, undercut_clean dual key), weather join (F7)

---

## Findings

### F1 — N06 `CompoundID`: trained on N01's 1-based codes, served the manifest's 0-based codes — off by one on EVERY lap [code-level CONFIRMED; impact measured below]

- **Producer:** the served delta model trains on `df23_d[FEATURES_DELTA]` (`.nb_py/N06_laptime_model.py:1380`), and `FEATURES_DELTA` contains `CompoundID` inherited from `FEATURES_PROD` (`N06:1245-1259`). `CompoundID` in the featured parquet is N01's mapping — `.nb_py/N01_data_download.py:232` ("1=Soft, 2=Medium, 3=Hard, 4=Inter, 5=Wet"), applied at `N01:247`. **Measured in the artefact** (`laps_featured_2025.parquet`): SOFT→1, MEDIUM→2, HARD→3, INTERMEDIATE→4.
- **The false headline:** N06's `encode_features` (`N06:128-133`) maps the `Compound` STRING column through the manifest's 0-based map (SOFT:0…WET:4) — but writes it into a column named `Compound`, which is **not in `features_in`**. The re-encoding is orphaned; the model consumed the untouched 1-based `CompoundID` column. The manifest's `categorical_encoding.Compound` block (`data/processed/feature_manifest_laptime.json:66-73`) is true about what `encode_features` did and false about what the model ate.
- **Inference:** `pace_agent.py:318-319` loads `manifest["categorical_encoding"]["Compound"]` (0-based) as `self.compound_id`; `_encode_categorical` (`pace_agent.py:459`) feeds `self.compound_id.get(compound, 1)`. So a MEDIUM lap is served CompoundID=1 — which the trained model learned as SOFT. Every compound is shifted one class down, on 100% of laps, on every path (replay, CLI, backend — they all go through `_build_feature_row`).
- **Failing scenario:** any 2025 lap on HARDs: training saw 3, inference feeds 2 (=trained MEDIUM). The model's compound-dependent degradation signal is systematically read one compound softer.

### F2 — N06 `DriverNumber`: lap_state has no such key, so inference feeds the constant 0 where training saw car numbers 1–81 [code-level CONFIRMED; impact measured below]

- **Producer:** `DriverNumber` is a real trained feature (`xgb_laptime_delta_feature_names.json:2`), int car numbers (featured 2025: 1, 4, 5, 6, 7, 10, 12, 14, …; 0 never occurs).
- **Inference:** `RaceStateManager.get_driver_state` (`race_state_manager.py:354-417`) emits NO `driver_number` key — the dict has `driver` (the three-letter code) but no number. `pace_agent.run_from_state` (`pace_agent.py:909`) does `d.get("driver_number") or 0` → **0 on every replay lap**, a constant outside the trained vocabulary. Bug class: "feature hardcoded to a constant where training saw a distribution" + "sentinel that is a findable value" (0 sorts below every real car number, so every DriverNumber split sends the row down its left branch).
- **Failing scenario:** every lap of every replay. The envelope cannot catch it — identifiers were deliberately excluded from bounds (`pace_agent.py:117-119`).

### F3 — N06 `FreshTyre`: trained = FastF1's set-was-new flag (constant across the whole stint); inference = outlap flag `int(tyre_life <= 1)` [code-level CONFIRMED; share + impact measured below]

- **Producer:** N06 consumes the parquet's `FreshTyre` cast to int (`N06:132`); that column is FastF1's flag for "the fitted set was NEW", which stays True for **every lap of a fresh-set stint**.
- **Inference:** `_compute_derived` (`pace_agent.py:526`) builds `FreshTyre = int(tyre_life <= 1)` — a first-lap-of-stint flag. The docstring (`pace_agent.py:476-477`, "binary flag for the first lap on a new tyre set — captures the outlap pace loss") names the wrong mechanism: that is not what the trained column measures. The RSM even emits the real thing (`fresh_tyre`, `race_state_manager.py:405`) and `run_from_state` ignores it — dropped input.
- **Divergence shape:** agree on outlaps and on all laps of used-set stints; disagree on every lap ≥ 2 of a fresh-set stint (training 1, inference 0) — the majority of racing laps.

### F4 — N06 `Prev_SpeedST`: trained = PREVIOUS lap's speed trap (stint-grouped shift); inference = THIS lap's trap [code-level CONFIRMED; magnitude + impact measured below]

- **Producer:** `N04_feature_engineering.py:384-392` — `Prev_SpeedST = grp['SpeedST'].shift(1)` grouped by `['Year','GP_Name','DriverNumber','Stint']`.
- **Inference:** `run_from_state` (`pace_agent.py:902,956`) feeds `d.get("speed_st")` — and `get_driver_state` emits `speed_st` from **the current lap's row** (`race_state_manager.py:410`). This is the #435 `prev_lap_time` bug's un-fixed twin: `prev_lap_time` was rewired to the real previous lap (parquet `Prev_LapTime` / reconstruction), but `prev_speed_st` still receives the current lap's reading on every call. The featured parquet even carries `Prev_SpeedST`; the raw replay parquet carries `SpeedST` per lap from which the true rule is computable — neither is used.
- Fallback `or 300.0` is additionally a findable real value (trained range 156–362 km/h), masking missing-trap laps as plausible readings.

### F5 — N06 `Prev_TyreLife`: trained = stint-shift over SURVIVING laps (NaN on stint openers); inference = `max(0, tyre_life - 1)` always [code-level CONFIRMED; share measured below]

- **Producer:** `N04:390` `Prev_TyreLife = grp['TyreLife'].shift(1)` — stint-grouped, over the featured frame that `filter_baseline_laps` has already thinned, so the "previous" lap can be ≥2 laps back; stint openers are NaN (and the delta model drops them only via `LapTime_Delta` NaN, `N06:1377-1378`).
- **Inference:** `pace_agent.py:955` `prev_tyre_life=max(0, (d.get("tyre_life") or 1) - 1)` — always exactly current−1, and **0 on outlaps** where training had NaN (0 is below the trained min 2.0; the envelope will warn, but the model still reads 0 as a value, not as missing).

### F6 — `Cluster` + `delta_vs_median`: the gp_name keyspace fix (#797) was applied to `_resolve_mean_sector_speed` only — its twins `_encode_categorical` and `_session_median` still take the raw metadata name [MEASURED: 2025 Miami misses]

- **Producer:** `Cluster` comes from `circuit_clusters_k4_2025.parquet` keyed by parquet slug (`'Miami'`, `'Las Vegas'`, …).
- **Inference:** `_encode_categorical` (`pace_agent.py:461`) does `self.circuit_cluster.get(gp_name, 1)` with `gp_name` straight from `metadata.json` via `session_meta` — no `_normalise_gp_key`. **Measured over all 24 races of `data/raw/2025/`:** every metadata `gp_name` resolves except **`'Miami Gardens'`**, which is `'Miami'` in both the cluster parquet and `laps_featured_2025`. Every lap of the 2025 Miami replay is served the DEFAULT cluster 1 instead of Miami's real cluster, and `_session_median` (`pace_agent.py:734-738`) finds no rows → `delta_vs_median=NaN` for the whole race.
- Same bug class as the lesson of 2026-07-16 (one copy fixed, its twin not): `_resolve_mean_sector_speed` got the four-keyspace chain and a test; the two other consumers of `gp_name` in the same class did not.

---

## Measured impact on the served N06 delta model (real 2025 laps)

Harness: `laps_featured_2025.parquet`, 21,247 anchored laps (`Prev_LapTime` known — the laps
where the model has a real anchor; the other 1,513 are dominated by the documented 90.0
placeholder). Base = trained-truth values from the artefact; variant = the value inference
constructs from the same row; weather held constant across base/variant except in F7.
Model: `xgb_laptime_delta_final.json`. Deltas are on the model's output (s), which moves the
absolute prediction 1:1 (`pred = prev + delta`).

Served-model feature importances (gain): TrackTemp 20.9%, LapsSincePitStop 16.2%, AirTemp
12.5%, Prev_LapTime 11.9%, Prev_DegradationRate 7.5% … Prev_SpeedST 2.9% … CompoundID 0.55%,
DriverNumber 0.03%, **FreshTyre 0.00%**.

| Finding | value differs on | preds moved >1ms | mean\|Δ\| | p95\|Δ\| | max\|Δ\| |
|---|---|---|---|---|---|
| F1 CompoundID (manifest 0-based) | 100.0% | 3.7% | 0.0055s | ~0 | 0.221s |
| F2 DriverNumber=0 | 100% | **0.0%** | 0.0000s | 0 | 0.000s |
| F3 FreshTyre outlap-flag | 79.7% | **0.0%** | 0.0000s | 0 | 0.000s |
| F4 Prev_SpeedST=current trap | 88% (mean 4.6 km/h, p95 19.0) | 8.2% | 0.0083s | 0.0125s | 2.438s |
| F5 Prev_TyreLife=TL−1 | 1.8% of known rows (+66 NaN→0) | 0.2% | 0.0003s | 0 | 0.262s |
| F6 Cluster Miami→default 1 | 0% — **Miami's real cluster IS 1** | 0.0% | 0 | 0 | 0 |
| F1–F5 combined | — | 11.4% | 0.0136s | 0.0719s | 2.567s |

Honest downgrades from the code-level read: **F2 and F3 move nothing** — the served delta
model has 0.03% / 0.00% importance on DriverNumber / FreshTyre, so the wrong quantity is fed
but never read. F6's cluster default coincides with Miami's real cluster (1), so only
`delta_vs_median` (NaN all race) actually degrades today; the Cluster defect is latent — it
bites the day any keyspace-missing race has cluster ≠ 1. **F4 is the largest single-feature
mover** and F1 is systematic but small. Severity ranks by these numbers, not by how wrong
the wiring looks.

### F7 — HIGH. The weather block: training = N04's `merge_asof(nearest)` on session `Time`; serving replay = proportional row-index lookup. Largest measured mover in N06.

- **Producer:** N04 Step 5, reproduced verbatim by `src/f1_strat_manager/weather_restore.py:57-94`
  (`merge_asof(direction="nearest")` joining each driver-lap's session `Time` to the nearest
  weather sample; docstring records the committed 22,760/22,760 reproduction against N04's own
  combined-parquet output).
- **Inference (replay path):** `RaceStateManager.get_weather_state`
  (`race_state_manager.py:515-518`): `idx = int((lap-1)/(total-1) * (len(weather_df)-1))` —
  a lap-number-proportional index into the weather frame, ignoring session time entirely.
  Every weather consumer on the replay path eats this: N06 (AirTemp/TrackTemp/Humidity/
  Rainfall = 39.7% of the served model's gain), N14's SC features, N26, N28.
- **Measured on all 24 races of 2025, 26,692 driver-laps:**
  - TrackTemp differs (>0.05°C) on **92.6%** of laps — mean |Δ| 1.11°C, p95 3.40, max 11.80°C.
  - AirTemp differs on 84.5% (mean 0.38°C), Humidity on 76.2% (mean 2.27%, max 17).
  - **Rainfall flag flips on 4.79% of laps (1,279 laps)** — a dry lap served as wet or wet as dry.
  - Worst races (mean |TrackTemp Δ|): Shanghai 2.38°C, Las Vegas 2.20, Spa 2.15, Miami 2.07.
- **Prediction impact (weather block swap, same 21,247-lap frame): 26.8% of laps move >1ms,
  mean |Δ| 0.0367s, p95 0.1189s, max 8.280s** — roughly 3× the combined effect of F1–F5.
- **Failing scenario:** any race where track temperature drifts monotonically (evening races:
  Las Vegas, Lusail) — the proportional index reads the session's time axis as if laps were
  uniform, so every SC-lengthened lap shifts every later lap's weather sample; the 8.28s tail
  laps are rain-flag flips.
- **Blast-radius caveat, measured:** on the CLI PMV path the TIRE model does NOT eat this —
  its laps_df is the augmented featured frame, whose per-lap weather is restored by the
  N04-faithful `weather_restore.weather_for_race`. F7 hits the consumers that read
  `lap_state["weather"]`: N06 pace, N27's SC/overtake features, and any weather default path.

---

## N26 — tire degradation TCN (`src/agents/tire_agent.py`)

Harness: for **335 real 2025 stints across 8 races** (Melbourne, Shanghai, Barcelona,
Silverstone, Monza, Lusail, Austin, Sakhir), build the agent's own 42-feature frame via
`TireAgent._build_stint_features` on the exact serving path (race-scoped augmented featured
frame, session_meta as `run_from_state` builds it), and diff EVERY column against
`laps_tiredeg.parquet` — the artefact N09/N10 actually trained on (N10 `.nb_py:63`) — for the
same `(GP, Driver, Stint, LapNumber)` rows. Then run each compound's TCN bundle on both frames.

**Aggregate divergence of the agent's frame vs the trained truth, TCN output (cumulative
fuel-adjusted degradation, the quantity the ~2s cliff thresholds are compared against):
mean |Δ| = 0.424s, p50 = 0.302s, p95 = 1.226s, max = 4.99s.** Attribution by single-group
swap (base = trained-truth frame):

| cause | TCN mean\|Δ\| | p95 | max |
|---|---|---|---|
| F8 deg pair shift bug | 0.185s | 0.510s | 4.609s |
| F9 lap_time_vs_cluster_mean artefact split | 0.208s | 0.811s | 2.709s |
| F10 Prev_*/delta fillna block | 0.198s | 0.546s | 1.934s |
| F12 mean_sector_speed convention | 0.021s | 0.079s | 0.267s |
| F11 LapsSincePitStop=TyreLife | 0.005s | 0.032s | 0.174s |
| ALL combined | 0.424s | 1.226s | 4.989s |

### F8 — HIGH. `DegradationRate`/`DegAcceleration`: the agent lags them by one lap; training never did. The "leakage fix matching N10 training" comment matches nothing.

- **Producer:** N04 `add_degradation_rate_features` (`.nb_py/N04_feature_engineering.py:481-504`):
  `DegradationRate[i]` = polyfit slope over laps `[i-2..i]` — **includes lap i, no shift**;
  `DegAcceleration[i] = deg[i] - deg[i-1]`, also unshifted. N09 consumed the stored columns
  as-is: `PRODUCTION_FEATURES = BASE_FEATURES` (`.nb_py/N09_tiredeg_tcn.py:219-220`) — the
  `LAPTIME_SHORTCUTS` set only defines the (unused-in-production) PURE variant. No `shift`
  exists anywhere in N09/N10's feature construction. The manifest's "use their lagged values"
  note (`tiredeg_feature_manifest.json:66-67`) was ADVICE the notebooks never took.
- **Inference:** `tire_agent.py:663-664` — `.shift(1).fillna(0)` on both, with a comment
  ("Both are shifted by 1 lap (leakage fix matching N10 training)") that names a mechanism
  absent from training. The agent follows the manifest's advice instead of the model's
  actual diet: every window position reads the slope of `[i-3..i-1]` where training saw
  `[i-2..i]` — a one-lap-stale degradation signal at every timestep.
- **Measured:** value-level mean |Δ| 0.28 s/lap (DegradationRate) / 0.40 (DegAcceleration)
  vs the artefact; TCN impact above. Failing scenario: the cliff onset lap — exactly when
  the slope spikes — is seen one lap late, on a model whose whole job is cliff timing.

### F9 — HIGH. `lap_time_vs_cluster_mean`: two artefacts of the same thing disagree (mean 5.75s), and serving feeds the one the model was NOT trained on.

- **Measured artefact-vs-artefact** (22,760 joined 2025 rows): `laps_featured_2025.parquet`
  vs `laps_tiredeg.parquet` differ on **100% of rows, mean |Δ| 5.75s, p95 11.78s** for
  `lap_time_vs_cluster_mean`, and on **30.3% of rows for `Cluster` itself** — the two
  artefact families used different clusterings (`circuit_clusters_k4` vs `k4_2025`), so
  N07 recomputed the cluster-mean delta against different cluster assignments.
- **Inference:** the TCN trained on the tiredeg version. `_add_session_cols`
  (`tire_agent.py:768`) keeps the FEATURED column when present — the serving frame always
  carries it — so 100% of serving rows get the other family's quantity. Meanwhile
  `_build_stint_features` (`tire_agent.py:954`) stomps `Cluster` from the k4 map, which
  MATCHES tiredeg — so the served frame mixes families: tiredeg's Cluster with featured's
  cluster-mean delta. TCN impact: mean 0.208s, p95 0.811s.

### F10 — MEDIUM. `_add_prev_cols` fills EVERY missing predecessor with the CURRENT lap's value; training's "no reading" was a scaled zero. The docstring says "first lap of a stint" — wrong mechanism.

- **Producer:** N04's `Prev_*` are stint-grouped shifts, NaN where no predecessor survives
  (`N04:384-392`); N09/N10 scale with `fillna(0)` (`N10:176,181`), so the trained "missing"
  signal is a raw zero.
- **Inference:** `tire_agent.py:619` — `df[src_col].shift(1).fillna(df[src_col])` fills EVERY
  NaN in the shifted series (stint openers AND every lap whose predecessor reading is NaN)
  with the current lap's value. Measured NaN-pattern divergence per column: Prev_SpeedI1 22.0%,
  Prev_SpeedST 16.7%, LapTime_Trend 13.1%, Prev_LapTime/Prev_TyreLife/deltas 6.1% of rows.
  Where both sides have values they agree exactly (mean |Δ| = 0) — the defect is purely the
  missing-value convention. TCN impact: mean 0.198s, p95 0.546s.

### F11 — LOW (measured), one-line fix. `LapsSincePitStop = TyreLife` — the #800 bug's un-fixed twin, stomping a trained column the frame already carries.

- **Producer:** N01's pit counter (`N01:262-282`), present in both featured and tiredeg
  artefacts (they agree 100%).
- **Inference:** `tire_agent.py:593` overwrites it unconditionally with `TyreLife`, even
  though the serving frame carries the real column two keys away. Measured on the 335-stint
  frame: 15.8% of rows differ, mean 2.48 laps (used-set stints, where the tyre's age ≠ laps
  since the stop). The pace agent got the RSM `laps_since_pit` fix (#800); this alias
  survived. TCN impact is small (0.005s mean) because the TCN barely reads it — but the fix
  is deleting one line, since the guarded pattern used for FuelLoad already covers it.

### F12 — LOW. `mean_sector_speed`: the tiredeg artefact broadcasts ONE per-GP value across seasons (95.5% of GPs identical 2023 vs 2025); serving feeds the per-year 2025 measurement — season-correct, training-inconsistent.

- The inverse of #797: for N06 the per-year value is right because N06 trained per-year rows
  from per-year artefacts; the TCN trained on tiredeg's broadcast convention, so the "right"
  2025 value differs from the trained quantity on 96.1% of rows (mean 4.31 km/h). Impact
  0.021s mean — record, don't panic.

### N26 notes (no measured impact, keep honest)

- `_build_stint_tensor`'s docstring line (`tire_agent.py:1023-1024`) still says short stints
  are "left-padded by repeating the first row" — the body zero-pads (correctly, per N09).
  Wrong-mechanism docstring, one line.
- N09 trains with input = laps 1..L−1 and target at L (`N09 Step 2`); the agent's window
  includes the current lap, so its "current degradation" reading is, by the training
  contract, the model's estimate for the NEXT step. Consistent construction, shifted
  interpretation — worth one sentence in the tool's docstring, not a code change.
- Cluster keyspace: the tire agent's k4 map contains BOTH 'Miami' and 'Miami Gardens'
  (both cluster 1), so the F6 keyspace miss does not reproduce here.

---

## N27 — overtake + SC (`src/agents/race_situation_agent.py`)

### F13 — MEDIUM. No domain gate on the overtake tool: N11 trained ONLY on pairs within 2.5s; 41.9% of real adjacent pairs are farther than that, and the tool scores them anyway.

- **Producer:** N11's pair builder drops every pair with `gap > 2.5` before labeling
  (`.nb_py/N11_overtake_eda.py:233-235`) — the model has never seen a labeled example
  beyond 2.5s.
- **Inference:** `predict_overtake_tool` (`race_situation_agent.py:1140-1204`) has range
  and roster guards but NO gap guard; `_build_overtake_features` happily builds a 9s-gap
  row and LightGBM extrapolates.
- **Measured on all 24 races of 2025 (25,215 position-adjacent pairs, N11's own pairing
  rule):** 41.9% of pairs have gap > 2.5s (p50 1.98s, p90 9.27s). Four in ten "score my
  car vs the car ahead" calls on the replay path are answered from outside the trained
  domain, unlabeled by any envelope (#710 family, N27 edition).

### N27 notes

- `gap_ahead_s`: N11 uses `abs(Time_x − Time_y)` (`N11:233`), inference uses
  `max(0.0, t_x − t_y)` (`race_situation_agent.py:919-923`). **Measured: 0 of 25,215
  adjacent pairs invert (t_x < t_y)** — end-of-lap position order guarantees the sign, so
  the difference is unreachable for adjacent pairs. Only an LLM tool call naming an
  inverted (non-adjacent) pair would hit it, and it would then read gap=0.0 + DRS open.
  Cheap hardening, not a live bug.
- `circuit_cluster` unknown default is 0 (`race_situation_agent.py:1449`), a REAL cluster;
  N11's unknown was −1 (`N11:212`). Only fires on a keyspace miss — the k4 map carries
  both Miami spellings, so no 2025 race hits it today. Latent sentinel.
- `pace_delta_rolling3` (`race_situation_agent.py:957-961`) pairs the two drivers'
  last-3-lap arrays positionally (`[:n_shared]`), not by LapNumber; N12 rolled over the
  battle-pair series (`N12:141-146`). Diverges only when one driver misses a lap in the
  window. Not measured — flagged as a rule difference with a bounded trigger.
- Weather for the SC features comes from `lap_state["weather"]` → F7's proportional-lookup
  misalignment applies to `track_temp`/`air_temp`/`humidity` here too (TrackTemp differs on
  92.6% of laps, mean 1.11°C). `track_temp_start` itself is correctly the session's first
  sample (`race_state_manager.py:538-542`).

---

## N28 — pit duration (N15) + undercut (N16) (`src/agents/pit_strategy_agent.py`)

### F14 — HIGH (for the 2025 Miami replay), keyspace family. `tire_compounds_by_race.json` is keyed 'Miami'; the replay's gp_name is 'Miami Gardens' — compound resolution falls back TWO steps hard on every lap of that race, and it routes the tire agent's BUNDLE choice.

- **Measured:** `tc['2025']['Miami']` = {SOFT: C5, MEDIUM: C4, HARD: C3}. The fallback used
  on a keyspace miss is {SOFT: C3, MEDIUM: C2, HARD: C1}
  (`tire_agent.py:812`, `pit_strategy_agent.py` `_COMPOUND_FALLBACK`).
- **Consumers hit on a 2025 Miami replay** (`gp_name='Miami Gardens'` from metadata.json):
  - `tire_agent._compound_name_to_id` (`tire_agent.py:819-821`): SOFT stints route to the
    **C3 TCN bundle instead of C5** — wrong model, wrong window, wrong scaler — and
    `AbsoluteCompoundID`/`CompoundHardness` encode 3/4 where training for that stint's
    physics would say 5/2. The whole race, every stint.
  - `pit_strategy_agent._compound_to_id` (`pit_strategy_agent.py:425-428`): N16's
    `compound_x_id`/`compound_y_id` come out 2 low (`compound_delta` survives because both
    shift equally).
  - Same family as F6 (pace `_encode_categorical`/`_session_median`). The fix is one
    normalisation (`_normalise_gp_key`) applied at every gp_name-keyed lookup, plus the
    enumeration test `test_pace_circuit_speed.py` already models.
- **Bonus artefact finding:** `undercut_clean.parquet` itself contains BOTH `'Miami'` (12
  stops) and `'Miami Gardens'` (3 stops) as distinct `circuit_key` values — the dual
  keyspace leaked into the training artefact, splitting one circuit's statistics.

### N28 notes (verified, mostly clean)

- N15 encoders verified against `.nb_py/N15_pit_duration.py`: compound order {SOFT:0…WET:4}
  + unknown −1 ✓ (`N15:253,265` vs `pit_strategy_agent.py:133-136`), tyre-life clip at 50 ✓
  (`N15` "cell 11" vs `_MAX_TRAINED_TYRE_LIFE`), feature list ✓ (`N15:312`).
- ONE convention drift: training filled missing TyreLife with **0** (`N15` engineer_features:
  `.clip(upper=50).fillna(0)`); `_tyre_life_in` returns **1** for missing
  (`pit_strategy_agent.py:853-855`) with a docstring arguing "fresh". Trained missing = 0,
  served missing = 1. Reachable on 1.98% of 2025 rows (the 451 NaN-TyreLife rows). LOW.
- N16 `pos_gap` order verified: `pos_X_before − pos_Y_before` (`N16:654`) = the agent's
  construction (#444 fix holds). `Lap_gap` = canonical 1-lap offset ✓ documented. `pit_delta_X`
  is a deliberate distribution→constant substitution (`traversal + 4.5`) where training saw
  per-stop measured values (mean 24.7s, per-circuit std up to 23.6s at Monaco) — documented
  design (#444), recorded here so nobody rediscovers it as a bug.

---

## Cross-cutting

- **The augment guard's claim is false for the CURRENT local artefacts**
  (`laps_augment.py:203-205`: "2023 and 2024 take this exit by construction" — measured:
  `laps_featured_2023/2024.parquet` carry NO weather columns locally, so `wants_weather=True`
  and the restore path runs for the training seasons too). Because `weather_restore`
  reproduces N04's join, the VALUES are right; the docstring's mechanism is wrong. This is
  #801's territory (featured-artefact regeneration) — noted, not re-litigated.
- **The manifest as a liar's mirror:** `feature_manifest_laptime.json` documents an encoding
  (0-based Compound) the served model never ate (F1), and `tiredeg_feature_manifest.json`
  advises a lag the training never applied (F8). Both times the inference code trusted the
  documentation over the training data. The gate rule this suggests: **wire inference to
  what the notebook DID, not to what the manifest SAYS, and verify by diffing against the
  artefact.**

---

## What I tried to break and could NOT

Verified same-quantity, same-rule, with executed evidence — do not re-audit without cause:

1. **`FuelLoad` (N06):** serving `(total−lap)/total` vs N01's `round(1 − lap/max_lap, 4)` —
   max |Δ| 0.000049 across four 2025 races (rounding only). Same `max_lap` source
   (raw-frame max = `RaceStateManager.total_laps`, `race_state_manager.py:130`).
2. **`laps_since_pit` (N06):** RSM rule vs N01's trained column — 0 mismatches on 3,795
   driver-laps across Melbourne/Lusail/Monza/Silverstone 2025. The #800 fix holds on the
   pace path (the tire path is F11).
3. **`FuelEffect` (N06 + N26):** `(TyreLife − stint_baseline) * 0.055` with baseline =
   stint min ✓ — 0.0% divergence on 5,804 stint-lap rows vs the trained artefact.
4. **N26 frame vs trained artefact, clean columns:** TyreLife, Position, FuelLoad, LapTime_s,
   Sector1-3_s, Speed traps, TeamID, CompoundID, AbsoluteCompoundID, CompoundHardness,
   Cluster (k4-stomp matches tiredeg), AirTemp/TrackTemp/Humidity/Rainfall (augmented
   featured path), `lap_time_pct_of_race_fastest`, `laps_remaining`, `track_status_clean`,
   Year — 0.0% divergence over 5,804 rows / 335 stints.
5. **N26 zero-pad + truncate-from-start + scaler-`fillna(0)`:** verbatim N09
   (`tire_agent.py:1048-1067` vs `N09 _pad_or_truncate`, `N10:133-140,176-181`).
6. **Overtake gap sign (N27):** 0 of 25,215 adjacent 2025 pairs invert, so `max(0,·)` ≡
   `abs(·)` on the replay's pair geometry. `Time` restoration via `_ensure_timedelta_laps`
   feeds the N11-trained gap rule (#447 fix holds).
7. **N15 encoders (N28):** compound order, unknown −1, tyre-life clip 50, feature order —
   all match the notebook.
8. **GP scoping:** `_scope_laps_to_gp` runs before every agent call on the RSM path
   (`strategy_orchestrator.py:2471`), so the season-frame leak (#465 family) is closed on
   the paths audited here.
9. **Pace `Stint`/`TyreLife`/`Position`/`prev_lap_time` wiring:** parquet-sourced, rule-true
   (`prev_lap_time` reconstruction reproduces N04's filtered stint-shift; #728 fix holds).

---

## Fix list, ordered by measured impact

1. **F7 — replace `get_weather_state`'s proportional row-index lookup with the N04 rule**
   (merge_asof nearest on session `Time`, already implemented in
   `weather_restore.weather_for_race`): removes the largest measured N06 mover (26.8% of
   laps, mean 0.037s, max 8.28s, 1,279 flipped rain flags) and cleans N27's SC weather
   features on the same path.
2. **F8 — delete the `.shift(1)` on `DegradationRate`/`DegAcceleration` in
   `tire_agent.py:663-664`** (training is unshifted): mean 0.185s / p95 0.51s of TCN output
   on every stint, concentrated exactly at cliff onset.
3. **F9 — serve `lap_time_vs_cluster_mean` from the quantity the TCN trained on** (recompute
   with tiredeg's cluster family, or regenerate the artefacts to agree): mean 0.208s / p95
   0.81s of TCN output. Requires deciding which artefact family is canonical (#801 adjacent).
4. **F10 — stop filling missing predecessors with the current lap in `_add_prev_cols`**
   (`tire_agent.py:619`): let NaN flow to the scaler's `fillna(0)` as trained. Mean 0.198s /
   p95 0.55s of TCN output. Also fix the "first lap of a stint" docstring (wrong mechanism).
5. **F14/F6 — one `_normalise_gp_key` at every gp_name-keyed lookup** (pace
   `_encode_categorical` + `_session_median`, tire `_compound_name_to_id`, pit
   `_compound_to_id`), with an all-71-races enumeration test like
   `test_pace_circuit_speed.py`: un-breaks the entire 2025 Miami replay (wrong TCN bundle,
   compound codes 2 off) and the latent Cluster default.
6. **F1 — feed N06 the parquet's 1-based CompoundID** (or retrain with the manifest
   encoding): 100% of laps off by one class; measured 3.7% of predictions move, max 0.221s.
   Fix the manifest's `categorical_encoding` block in the same change.
7. **F13 — gate `predict_overtake_tool` at the trained 2.5s domain** (refuse or label as
   extrapolation): 41.9% of adjacent-pair calls are out-of-domain today.
8. **F4 — pass the parquet's `Prev_SpeedST` (or the RSM's previous-lap trap) instead of the
   current lap's `speed_st`** (`pace_agent.py:956`): 8.2% of predictions move, max 2.44s.
9. **F11 — delete `tire_agent.py:593`** (`LapsSincePitStop = TyreLife`) and guard like
   FuelLoad: measured impact small (0.005s) but it is a one-line fix for a wrong quantity on
   15.8% of rows.
10. **F2/F3/F5/F12 + notes — bundle as a hygiene PR:** emit `driver_number` in
    `get_driver_state` and stop defaulting to 0; read the RSM's `fresh_tyre` instead of the
    outlap proxy; stint-opener `Prev_TyreLife` → NaN not 0; document the tiredeg
    `mean_sector_speed` broadcast convention; fix the two wrong-mechanism docstrings
    (`tire_agent.py:1023`, `_add_prev_cols`); align `_tyre_life_in` missing→0 with N15.
    All measured ≤0.021s or zero — correctness debt, not fires.

