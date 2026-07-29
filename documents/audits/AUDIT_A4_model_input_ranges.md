# AUDIT A4 — Model input ranges: trained vs served

Adversarial gate. Read-only on the repo (this file is the one exception). No LLM/API
calls. Every number below is executed against real files in this checkout, not
estimated — the exact commands are inlined so any number can be reproduced.

Date: 2026-07-29. Scope requested: N15 (pit duration), N12 (overtake), N14 (safety
car), N06 (lap time) get a trained-range + served-range comparison; N26 (tire TCN)
gets a qualitative sequence-validity writeup instead of a per-feature range.

Status: COMPLETE. All five models in scope (N06, N12, N14, N15, N26) have a measured
section below, closed by a ranked findings list and a "what I tried and could not
break" section.

---

## Method note (read before the per-model sections)

For each tabular model the "trained range" is computed from the **exact feature
construction the training notebook performed**, applied to the **exact rows the
notebook selected** (train years, target-scope filters), read from the parquet(s)
the notebook itself loaded. Where the notebook's raw training frame is not
materialized on disk (N15 — see its section), this is stated explicitly rather than
approximated silently.

The "served range" drives real 2025 laps through the SAME feature-builder code the
agents call in production (`RaceSituationAgent._build_overtake_features` /
`_build_sc_features`, `PaceAgent._build_feature_row` via `run_from_state`,
`PitStrategyAgent._build_pit_duration_features`) using `src.simulation.replay_engine.
RaceReplayEngine` fed from `data/raw/2025/<GP>` — the same real telemetry the CLI/
arcade/backend replay. No feature value below is hand-computed; every one is what
the model actually received when the agent code ran.

One methodology correction made mid-audit, recorded because it is itself a finding
about how easy this is to get wrong: an early pass fed `RaceSituationAgent`'s feature
builders the RAW `data/raw/<year>/<gp>/laps.parquet` DataFrame directly. That is wrong
— every real no-llm caller (`src/strategy/eval/decision_modes.py`,
`engine.run_lap(profile="no-llm")`, and therefore the CLI/arcade/backend) passes the
**featured** parquet (`laps_featured_<year>.parquet` via `augment_featured_laps`,
scoped to the GP) as `laps_df`; the raw `data/raw/` tree is only used to build the
per-lap `RaceReplayEngine`/`RaceStateManager` snapshot. Feeding the raw frame silently
produced a different `lap_pct` range (a hand-built `session_meta` without `total_laps`
defaulted to 57, so Monaco's 78 laps read as 137% of the race — not a real bug, an
artefact of the wrong harness). All N12/N14 numbers below are from the corrected
harness: `engine.run_lap(race_state, laps_2025, lap_state, profile="no-llm")` with
`laps_2025 = augment_featured_laps(pd.read_parquet(".../laps_featured_2025.parquet"), 2025)`,
exactly as this audit's brief specified, with `RaceSituationAgent._build_sc_features`
/ `_build_overtake_features` monkeypatched to record (not alter) every fed row.

---

## N06 — pace / lap-time (XGBoost delta model)

**Trained range.** `notebooks/strategy/lap_time_prediction/N06_laptime_model.ipynb`
Step 9.3 ("Full retrain on 2023+2024 → test on 2025") is the production model —
confirmed against `data/models/lap_time/xgb_laptime_delta_feature_names.json` (the
25-feature list actually loaded at inference). Training frame = `laps_featured_2023.
parquet` + `laps_featured_2024.parquet` concatenated (22,106 + 23,256 = 45,362 rows,
which matches `feature_manifest_laptime.json`'s `n_laps_train`/`n_laps_val` exactly —
cross-check passed), with cell 46's `add_lag_deg_features` (a `groupby(['GP_Name',
'Year','DriverNumber','Stint']).shift(1)`) reproduced verbatim to build
`Prev_DegradationRate`/`Prev_CumulativeDeg`/`Prev_DegAcceleration`, and `FreshTyre`
cast to int per cell 8. Test year is 2025 (out-of-distribution by design — the
notebook's own Step 8/9 protocol, not a bug; see the `Year` row below).

**Served range.** `PaceAgent._build_feature_row` monkeypatched to record every row it
builds, driven via `run_pace_agent_from_state`/`PaceAgent.run_from_state` (the exact
function `src/strategy/inference/no_llm.py` calls) over 5 real 2025 races (Lusail,
Monza, Silverstone, Monaco, Spa-Francorchamps), 284 laps, one call per lap for our
driver (NOR/LEC/HAM/VER/PIA).

```
feature                  trained_min trained_max  served_min  served_max n_uniq escapes  pct
DriverNumber                   1.000      81.000       0.000       0.000      1     284  100.0%
LapNumber                      2.000      78.000       1.000      78.000     78       5    1.8%
Stint                          1.000       8.000       1.000       3.000      3       0    0.0%
TyreLife                       2.000      78.000       1.000      49.000     49      11    3.9%
FreshTyre                      0.000       1.000       0.000       1.000      2       0    0.0%
Position                       1.000      20.000       1.000       8.000      8       0    0.0%
CompoundID                     0.000       5.000       0.000       3.000      4       0    0.0%
TeamID                         0.000      10.000       1.000       4.000      3       0    0.0%
LapsSincePitStop               2.000      77.000       1.000      49.000     49      11    3.9%
FuelLoad                       0.000       0.974       0.000       0.987    250       5    1.8%
Year                        2023.000    2024.000    2025.000    2025.000      1     284  100.0%  (deliberate test-year OOD, see note)
FuelEffect                     0.000       4.125       0.000       2.640     49       0    0.0%
Prev_LapTime                  67.719     148.991      90.000      90.000      1       0    0.0%  (constant — see HIGH finding)
Prev_TyreLife                  2.000      77.000       0.000      48.000     49      22    7.7%
Prev_SpeedST                 156.000     362.000      77.000     352.000     88       4    1.4%
AirTemp                       14.500      33.800      14.600      27.400     76       0    0.0%
TrackTemp                     16.700      50.700      20.000      45.900    128       0    0.0%
Humidity                      18.000      92.000      40.000      91.000     43       0    0.0%
Rainfall                       0.000       1.000       0.000       1.000      2       0    0.0%
laps_remaining                 0.000      76.000       0.000      77.000     78       1    0.4%
Cluster                        0.000       3.000       0.000       3.000      3       0    0.0%
mean_sector_speed            196.629     314.971      77.000     352.000     88      73   25.7%  (100% substituted — see HIGH finding)
Prev_DegradationRate          -2.000       2.000       0.000       0.000      1       0    0.0%  (100% constant — see HIGH finding)
Prev_CumulativeDeg           -65.346      60.713       0.000       0.000      1       0    0.0%  (100% constant — see HIGH finding)
Prev_DegAcceleration          -2.000       2.000       0.000       0.000      1       0    0.0%  (100% constant — see HIGH finding)
```

### Finding N06-1 (HIGH) — `Prev_LapTime` is a hardcoded 90.0s constant on every real-replay call, not the true previous lap time

`Prev_LapTime` is the feature the absolute-time model's own feature-importance
analysis called out at 51.4% of gain (N06 cell 43), and the delta model still reads
it directly: `PaceAgent._predict` returns `prev + delta` — the predicted absolute lap
time is `Prev_LapTime` plus the model's delta. `pace_agent.py`'s
`run_from_state` reads it as `d.get('prev_lap_time') or 90.0`, with an inline comment
citing fix **#435** and claiming *"RaceStateManager.get_driver_state now emits the
real 'prev_lap_time' sourced from the parquet's Prev_LapTime column."*

That claim does not hold for the data the production entrypoints actually feed it.
`RaceStateManager.get_lap_state` reads `r.get("Prev_LapTime")` from whatever `laps_df`
it was constructed with (`race_state_manager.py:258`), and `RaceReplayEngine.__init__`
— the class every real surface uses (`src/arcade/strategy.py`,
`src/telemetry/backend/services/simulation/simulator.py`,
`scripts/run_simulation_cli.py:1408`, this project's own `decision_modes.py` eval
harness) — constructs it from `race_dir/laps.parquet`, i.e. the **raw** FastF1 export
in `data/raw/<year>/<gp>/`. Verified directly: `data/raw/2025/Lusail/laps.parquet` has
no `Prev_LapTime` column at all (checked its 34 columns). `Prev_LapTime` only exists
in the **featured** parquet (N04's derived column), which `RaceStateManager` never
sees — it is fed the raw frame by every real caller. So `r.get("Prev_LapTime")`
returns `None` on literally every lap of every race replayed this way, which is what
the measurement shows: `prev_lap_time` was `None` on all 162 sampled `lap_state`
dicts checked directly (`car.get('prev_lap_time')` printed `None` on laps 1, 2, 3, 4,
5, 20 of Lusail — including well into the race, where a real previous lap obviously
exists), and the served `Prev_LapTime` feature is the constant `90.0` on **100% of
284 calls across 5 races** (n_unique=1).

**Measured downstream damage** (executed, not estimated): comparing
`PaceOutput.lap_time_pred` against the real `lap_time_s` for the same lap, over full
green-flag replays:

```
race        n   mean_abs_err   mean_signed_err
Lusail     57   7.72 s         +0.44 s
Monza      53   6.97 s         +6.49 s
Monaco     78   13.83 s        +12.47 s
```

N06's own reported test MAE (2025, properly fed) is **0.392 s** (N06 cell 52). The
production replay path is 18-35x worse than the model's real accuracy, and the error
is circuit-dependent and directional exactly as the mechanism predicts: Monaco's real
laps run ~72-76 s, so anchoring every prediction to a 90.0 s constant plus a small
delta produces a large, consistently POSITIVE bias (+12.47 s); Lusail's laps run
close to 90 s by coincidence, so the same bug is nearly invisible there (+0.44 s) even
though `Prev_LapTime` is just as wrong. **The bug is silent specifically at the one
circuit (Lusail) this repo's own examples and this audit's brief happen to use most
often** — a `delta_vs_prev` reading looks perfectly reasonable there because `delta`
is computed self-consistently against the same wrong 90.0 s constant on both sides
(`delta_vs_prev = lap_time_pred - prev_lap_time = (90+δ) - 90 = δ`), so only the
**absolute** `lap_time_pred` (and `delta_vs_median`, which compares against the real
session median) carries the damage.

This is not an out-of-envelope value in the min/max sense — 90.0 s sits inside the
trained [67.7, 149.0] band — which is exactly why `OperatingEnvelope` as designed
(bounds-only) would not catch it. It is a **dead, constant feature standing in for a
per-lap value with real 67.7-149.0 s spread**, silently, on every replay-driven call
in this codebase.

### Finding N06-2 (HIGH) — `mean_sector_speed` is substituted with `Prev_SpeedST` on every call, a different physical quantity

Trained: `mean_sector_speed` is a **per-circuit constant** (24 unique values across
2023+2024, one per GP, std=0.0 within a circuit — verified directly), ranging
196.6-315.0 km/h. `PaceAgent._compute_derived`'s own docstring says it should fall
back to `prev_speed_st` only "when circuit_features are unavailable for the current
GP" — but `PaceAgent.run_from_state`'s docstring already half-admits the real
behaviour: *"Fields absent from the RSM schema (prev_deg_rate, prev_cum_deg,
prev_deg_accel, mean_sector_speed) default to 0.0/None since the replay engine does
not compute degradation history."* `run_from_state` never passes `mean_sector_speed`
at all, so it is **always** `None`, and `_compute_derived` **always** falls back to
`prev_speed_st` — a driver's own single-lap speed-trap reading (a very different,
much more volatile quantity: 77-352 km/h across the 284 sampled calls, n_unique=88,
vs. the trained feature's 24-value, low-variance circuit constant). 25.7% of served
values (73/284) land numerically outside the trained [196.6, 315.0] band; the other
74.3% are "in range" only by coincidence (many speed-trap readings happen to fall in
the same 196-315 km/h window most circuits' mean sector speed occupies), while still
being the wrong quantity on 100% of calls.

### Finding N06-3 (MEDIUM) — `DriverNumber` served as 0 on 100% of calls, below the trained floor of 1

`run_from_state`: `driver_number = d.get('driver_number') or 0`. The `driver` dict
`RaceStateManager.get_lap_state` builds has no `driver_number` key at all (verified:
absent from the 24-key dict printed for every lap sampled), so this is `0` on every
call — 100% escape from the trained `[1, 81]` floor. Likely low practical impact
(XGBoost's split structure on a near-identity-like integer ID feature is unlikely to
carry much signal), which is why this is MEDIUM not HIGH, but it is the same class of
bug as N06-1/N06-2: a value the model never saw at training time, served unconditionally.

### Finding N06-4 (LOW, already partially known) — the three lag-1 degradation features are a dead, constant 0.0

`run_from_state` passes `prev_deg_rate=0.0, prev_cum_deg=0.0, prev_deg_accel=0.0`
unconditionally (its own docstring says so). Trained ranges are wide
(`Prev_DegradationRate` ∈ [-2, 2], `Prev_CumulativeDeg` ∈ [-65.3, 60.7],
`Prev_DegAcceleration` ∈ [-2, 2]) with real per-lap variation; served is always
exactly 0.0, so **0% min/max escape but 100% signal loss** — measured percentile of
0.0 within the training distribution is 42.6% / 44.7% / 46.8% respectively (roughly
central, not an extreme value), so this defect is quieter than N06-1/2 in min/max
terms but identical in kind: a feature the model was trained to read per-lap is
silently replaced with one constant on every real call. Rated LOW here specifically
*because* 0.0 sits near the distribution's centre (unlike N06-1's 90.0s, which is
directionally biased per circuit) — but it is real, 100%-of-calls signal loss on
three separate features and should not be read as "harmless".

### Finding N06-5 (LOW) — early-stint coverage gap: TyreLife/LapsSincePitStop/Prev_TyreLife/LapNumber

3.9-7.7% escape, all in the same direction: the training frame's minimum observed
`TyreLife`/`LapsSincePitStop`/`Prev_TyreLife` is 2 (lap 1 of a race, where
`Prev_LapTime` is NaN by construction, appears to have been dropped somewhere in the
N04→N06 pipeline), while real replay legitimately reaches 0-1 on the first lap of a
stint. This is a genuine coverage gap (the model has never seen a laps-old-tyre value
of 0 or 1) but small in magnitude and not a code defect — noted for completeness.

---

## N12 — overtake probability (LightGBM + Platt calibration)

**Trained range.** `data/models/overtake_probability/model_config.json`: `train_seasons:
[2023, 2024]`, `test_season: 2025`, `n_train: 18277`. Verified directly: filtering
`overtake_pairs_2023_2025.parquet` to `Year in [2023, 2024]` gives exactly 18,277 rows
(cross-check passed). 15 features; the 4 derived ones (`gap_pace_product`,
`drs_ready_gap`, `gap_trend`, `pace_delta_rolling3`) were reconstructed from the
stored base columns using the exact formulas `model_config.json`'s
`derived_features` dict and `_build_overtake_features` both state.

**Served range.** `RaceSituationAgent._build_overtake_features` monkeypatched,
driven via `engine.run_lap(profile="no-llm")` (the corrected v3 harness, see the
methodology note above) over the same 5 races, 155 calls (a call only happens when a
rival is directly ahead by `position - 1`, which is why n < 284).

```
feature                        tr_min     tr_max     sv_min     sv_max n_uniq   esc    pct
gap_ahead_s                     0.002      2.500      0.270     21.769    155    84  54.2%
pace_delta_s                  -35.793     19.584     -3.095      2.454    145     0   0.0%
tyre_life_x                     2.000     77.000      2.000     32.000     31     0   0.0%
tyre_life_y                     2.000     77.000      2.000     40.000     37     0   0.0%
tyre_life_diff                -56.000     47.000    -33.000     11.000     12     0   0.0%
speed_trap_delta             -105.000    120.000    -31.000     69.000     34     0   0.0%
LapNumber                       4.000     77.000      2.000     57.000     56     4   2.6%
drs_window                      0.000      1.000      0.000      1.000      2     0   0.0%
gap_pace_product              -30.102     14.071    -33.571     20.299    155     4   2.6%
drs_ready_gap                   0.000      0.999      0.000      0.913     21     0   0.0%
gap_trend                      -2.403      2.358   -220.962    198.026    145    12   7.7%
pace_delta_rolling3           -31.283      9.262     -2.289      2.024    153     0   0.0%
categorical: compound_x/compound_y ∈ {C1..C5} (train); circuit_cluster ∈ {0,1,2,3} (train)
```

### Finding N12-1 (HIGH) — the no-llm path calls the overtake model on gaps up to 21.8 s, 8.7x its trained ceiling, on 54% of real calls

`RaceSituationAgent._build_overtake_features`'s training data (`overtake_pairs_2023_
2025.parquet`) tops out at `gap_ahead_s = 2.5 s` by construction — that is the scope
N12 was fit on. The rule that is supposed to enforce that scope at serve time lives
**only in the LLM system prompt** (`_RACE_SITUATION_SYSTEM_PROMPT`: *"If the gap to
the car ahead is less than 2.5 seconds, call predict_overtake_tool... If gap ahead >
2.5s, skip overtake tool and assume P(overtake) = 0.0"*) — advisory text an LLM reads,
not a check `predict_overtake_tool` itself enforces (its body checks lap range and
driver liveness, never `gap_ahead_s`). `src/strategy/inference/no_llm.py`'s
`_situation_no_llm` — the deterministic path every `profile="no-llm"` caller uses,
including this project's own `decision_modes.py` evaluation tier — calls
`predict_overtake_tool` for **any** rival directly ahead (`position - 1`), with no
gap check at all:

```python
if rival:
    calls.append((tools["predict_overtake_tool"], {"driver_x": driver, "driver_y": rival, ...}))
```

Measured: 84 of 155 real no-llm overtake calls (54.2%) across 5 races carried
`gap_ahead_s` above 2.5 s, up to **21.77 s** — 8.7x the model's trained ceiling, a
gap at which "is there an overtaking opportunity" is not a meaningful question. The
model still returns a calibrated probability with the same confidence as an in-range
call. This is exactly the class of defect `OperatingEnvelope` exists to label (and it
is straightforward to wire: `_N15_TYRE_LIFE_ENVELOPE` in `pit_strategy_agent.py` is
the template), but nothing currently checks `gap_ahead_s` against N12's trained scope
on the no-llm path.

### Finding N12-2 (MEDIUM) — `gap_trend` swings ±200 s because the featured parquet the no-llm path is fed has lap gaps

`gap_trend` is defined as `gap_ahead_s[this lap] - gap_ahead_s[previous lap in
laps_recent]` and trained on values in `[-2.4, +2.4]` (a one-lap gap change is
physically bounded — cars do not close or open 200 s of gap in one lap). Served
values reach **-220.96 / +198.03**, escaping the trained band 7.7% of the time. Root
cause: `laps_recent` is sliced from the **featured** parquet
(`laps_featured_2025.parquet`, which is what `engine.run_lap` actually threads into
`_build_overtake_features` via `self.laps_df` on the no-llm path — see the
methodology note), and the featured parquet is not a complete per-lap grid: N04's
`IsAccurate` gate drops SC/VSC/pit/inaccurate laps, so a driver pair's "previous lap
inside the window" can genuinely be several real laps back even though
`_build_overtake_features` computes the trend as if it were one lap earlier. The
formula is not wrong in isolation; it silently assumes a complete lap sequence that
the frame it is actually fed does not provide in production.

---

## N14 — safety car probability within 3 laps (LightGBM + Platt calibration)

**Trained range.** `data/models/safety_car_probability/feature_list_v1.json`:
`train_years: [2023, 2024]`, `test_year: 2025`. `sc_labeled_2023_2025.parquet`
filtered to `year in [2023, 2024]` gives 2,280 of 3,275 rows. 30 of 32 features map
directly to stored columns; `lap_time_mean_z`/`lap_time_std_z`/`lap_time_min_z` were
reconstructed using the **non-causal, whole-race** z-score
(`race_situation_agent._compute_laptime_features`'s own docstring states plainly:
*"N14 was trained on z-scores computed non-causally over the WHOLE race... That is a
genuine train/serve skew (#450)"* — this is the same mismatch already tracked in
project memory as `project_n14_causal_zscores_open_decision`, reproduced here rather
than re-litigated). `anomaly_and_yellow` could not be reconstructed from the labeled
parquet alone (it needs per-driver lap-time-vs-rolling-median comparisons the
aggregated table does not retain) — reported as not established rather than guessed.
`lap1_chaos` reconstructed as `is_lap1 * |n_drivers_delta|`, which came out
identically 0 in the labeled training rows (plausible: `n_drivers_delta` on lap 1 has
no lap 0 to diff against and the source likely backfills it to 0), so its trained
range is a degenerate `[0, 0]` — noted, not asserted as fully verified.

**Served range.** `RaceSituationAgent._build_sc_features` monkeypatched, driven via
the same corrected v3 harness, 284 calls (every lap, since `predict_sc_tool` runs
unconditionally in `_situation_no_llm`).

```
feature                        tr_min     tr_max     sv_min     sv_max n_uniq   esc    pct
lap_time_mean_z                -1.973      7.697     -2.592      1.771    251    12   4.3%
lap_time_std_z                 -1.253      6.154     -2.567      4.382    251    23   8.2%
lap_time_min_z                 -3.505      7.916     -3.129      1.456    252     0   0.0%
lap_time_cv                     0.000      0.228      0.000      0.038    257     0   0.0%
lap_time_trend_5                0.142      1.543      0.890      1.007    212     0   0.0%
n_drivers                       7.000     20.000      0.000     20.000     15    37  13.0%
n_drivers_delta               -11.000      0.000    -20.000     14.000     19    79  27.8%
tyre_life_mean                  1.000     66.400      2.000     36.714    238     0   0.0%
tyre_life_max                   1.000     78.000      2.000     71.000     70     0   0.0%
tyre_age_high_risk_count        0.000     13.000      0.000      7.000      8     0   0.0%
active_pitstop_count            0.000     18.000      0.000      0.000      1     0   0.0%
outlap_drivers                  0.000     20.000      0.000     18.000     10     0   0.0%
track_status_enc                0.000      2.000      0.000      1.000      2     0   0.0%
status_changed                  0.000      1.000      0.000      1.000      2     0   0.0%
status_change_direction        -1.000      1.000     -1.000      1.000      3     0   0.0%
yellow_escalation_count         0.000      2.000      0.000      1.000      2     0   0.0%
laps_since_last_yellow          0.000     10.000      0.000     10.000     11     0   0.0%
had_incident_msg                0.000      1.000      0.000      0.000      1     0   0.0%
incident_escalation             0.000      1.000      0.000      0.000      1     0   0.0%
yellow_sectors_this_lap         0.000     10.000      0.000      0.000      1     0   0.0%
yellow_sectors_prev3            0.000     13.000      0.000      0.000      1     0   0.0%
rcm_incident_count_prev3        0.000     18.000      0.000      0.000      1     0   0.0%
track_temp                     17.724     49.311     20.000     45.900    128     0   0.0%
air_temp                       15.771     33.102     14.600     27.400     76     7   2.5%
humidity                       21.497     74.650     40.000     91.000     43    82  28.9%
track_temp_delta              -16.800      5.200     -6.000      5.400    102     1   0.4%
circuit_cluster                 0.000      3.000      0.000      3.000      3     0   0.0%
circuit_sc_rate                 0.000      0.250      0.150      0.150      1     0   0.0%
lap_pct                         0.013      1.000      0.013      1.000    250     0   0.0%
is_lap1                         0.000      1.000      0.000      1.000      2     0   0.0%
lap1_chaos                      0.000      0.000      0.000      0.000      1     0   0.0%
```

### Finding N14-1 (HIGH) — `n_drivers_delta` reaches +14 in production; the model was trained on a domain where a POSITIVE delta cannot happen at all

Trained `n_drivers_delta` ∈ **[-11, 0]** — every single one of 2,280 training rows has
`n_drivers_delta <= 0`, which makes physical sense: cars retire during a race, they do
not un-retire, so the driver count can only fall or hold lap over lap. Served values
range **[-20, +14]**, escaping the trained band on **27.8%** of calls (79/284), and a
positive value is not a rare edge case at the boundary — it is a value the training
distribution says is impossible. Root cause is the same lap-gap issue as N12-2: the
no-llm path's `_build_sc_features` is fed the featured parquet (not a complete raw
per-lap grid), so `_compute_driver_tyre_features(cur, prev)` counts distinct drivers
present at `LapNumber == lap_number` vs `LapNumber == lap_number - 1` in a frame where
either side can be sparsely populated (N04's accuracy gate drops rows unevenly across
drivers), producing swings that look like drivers "appearing" lap over lap. This also
explains `n_drivers` itself reaching **0** (13.0% escape, trained floor 7) — an empty
`cur` slice, not zero cars actually racing.

### Finding N14-2 (MEDIUM) — `humidity` reaches 91%, above the 74.65% trained ceiling, on 28.9% of calls

This one measures real weather diversity rather than a code defect: `humidity` is read
straight from `lap_state['weather']['humidity']`, sourced from each race's real
`weather.parquet`. 2023-2024's SC-labeled sample (2,280 rows) never recorded humidity
above 74.65%; the 5 sampled 2025 races include conditions up to 91%. Whether this
reflects a genuinely narrower training-era weather sample or a fixable data-coverage
gap is not established here — flagged as a real, measured escape and left at that.

### Finding N14-3 (MEDIUM, cites a known/tracked mismatch) — the z-score features are trained non-causally, served causally

`lap_time_mean_z`/`lap_time_std_z` escape their trained band 4.3%/8.2% of the time.
The mechanism is already documented in this repo (`project_n14_causal_zscores_open_
decision`, and restated in `_compute_laptime_features`'s own docstring): training
z-scored each lap against the **whole race's** mean/std (including future laps
relative to the labeled row), while `_zscore` at serve time is deliberately **causal**
(only laps up to the current one). Early in a race the causal sample is small, so its
std is unstable and the resulting z-score runs hotter than the non-causal training
z-score would for the same underlying lap-time pattern — which is exactly the
direction and rough magnitude of the escapes measured here. This audit does not
re-open that decision (project memory already records it as deliberately not
retrained); it is reproduced here because it is a real, measured contributor to N14's
served-range escapes, not because it is new.

### Note — `track_status_enc` did NOT escape in this sample (unlike an earlier, incorrect pass)

An earlier version of this measurement (before the featured-parquet correction, see
the methodology note) showed `track_status_enc` escaping to 9.9% with a served max of
5 (a red-flag code the 2023-2024 SC-labeled rows never reached). The corrected v3
harness shows 0% escape (served max 1) over this specific 5-race, 284-lap sample —
none of the 5 races sampled here happened to run under a full SC/VSC/red flag on our
driver's laps. This is recorded so the number is not silently dropped rather than
corrected: it is plausible this escape re-appears with a different race selection
(Qatar/Spain 2025 both had real SC/VSC periods per project memory), and a wider sweep
than this audit's 5-race sample would be needed to confirm either way.

Note: N12/N14 are not fed an explicit `Year` feature (unlike N06), so there is no
per-feature row for it above — the train(2023-2024)/serve(2025) split itself is the
same designed generalisation protocol described just below for N06.

---

---

## N15 — pit stop physical duration (HistGBT quantile P05/P50/P95)

**Trained scope, established with confidence:** `data/models/pit_prediction/model_config.
json` + `N15_pit_duration.ipynb`: `train_years: [2023, 2024]`, `test_year: 2025`, 9
features (`team, year, tyre_life_in, lap_number, compound_id, compound_change, under_sc,
tight_pit_box, team_year_median`). Target scope is deliberately narrow —
`physical_stop_est ∈ [2.0, 4.5] s` ("normal" wheel changes only; penalties/jack
failures/unsafe releases are explicitly out of scope, cell 16). `tyre_life_in` is
`TyreLife.clip(upper=50).fillna(0)` (cell 11) — a **hard clip**, so its trained upper
bound is exactly 50 by construction, and this is the one N15 feature already guarded at
serve time: `pit_strategy_agent.py`'s `_N15_TYRE_LIFE_ENVELOPE` + `_tyre_life_in()`
clips and **logs a warning** on every out-of-envelope call (`#710`). Notebook cell 14's
own printed stats give the trained categorical rates directly: `compound_change` 31.3%,
`under_sc` 20.6%, `tight_pit_box` 11.2%, over 2,205 base stops (725 in the 2023-2024
train split).

**What could NOT be established with confidence, and is reported as such rather than
guessed:** the exact per-feature numeric distribution (min/p1/p99/max) of `tyre_life_in`
and `lap_number` in the real training frame. The notebook downloads pit data live via
`fastf1.get_event_schedule`/session loads into a local FastF1 cache and never persists
`df_raw`/`clean`/`train` to `data/processed/pit_labeled/` (that directory exists,
`mkdir`'d by the notebook, and is empty in this checkout). Reconstructing the same
in-scope stops (`TyreLife_out <= 5`, `physical_stop_est ∈ [2.0, 4.5] s`, matching
`_extract_physical_stops`'s own logic) from `data/raw/2023` + `data/raw/2024` in this
checkout yields only **18 in-scope stops**, against the notebook's own reported 725 —
`data/raw/` here holds a partial subset of the 2023-2024 calendar, not the full set the
notebook trained on. Presenting an 18-row sample as "the trained range" would
understate it; it is not reported as such. (For reference only, not as a claim: that
18-row proxy spans `tyre_life_in` 1-50, `lap_number` 1-63 — directionally plausible,
not authoritative.)

**Served range** (this part IS fully executed): `PitStrategyAgent._build_pit_duration_
features` called directly (N28 is LLM-backed and does not run under `no-llm`, per this
audit's brief) with `self.laps_df`/`self.session_meta` populated exactly as
`PitStrategyAgent.run_from_state` does (featured parquet scoped to the GP, real
`total_laps`) — 284 calls, one per real lap, across the same 5 races.

### Finding N15-1 (HIGH) — `team_year_median` is the flat 2.8s global fallback on 100% of served calls, even though the team's own real prior-year data is loaded in memory

The notebook's own procedure text is explicit: *"Add `team_year_median` as numeric
team-quality prior (**test uses each team's 2024 median**)"* (cell 16), and
`add_team_year_median`'s `get_med(team, year)` falls back from an exact `(team, year)`
match to **that team's own median across whatever years it has**, and only then to the
global median — i.e. the notebook's design already anticipated that 2025 (the serving
year) would never have training data for that exact year, and built a same-team,
prior-year fallback for it.

The shipped inference code does not reproduce that fallback chain. `PitAgentCFG.
_load_team_year_medians()` aggregates real `(team, year)` medians from
`data/raw/<year>/<gp>/pitstops.parquet` across 2023-2025 (16 entries total, verified by
loading `PitStrategyAgent().cfg.team_year_median` directly) — real, meaningfully
different values per team and year:

```
('Ferrari', 2023)          2.619        ('McLaren', 2023)           2.260
('Ferrari', 2024)          4.370        ('McLaren', 2024)           4.289
('Red Bull Racing', 2023)  3.219        ('Williams', 2024)          2.049
('Red Bull Racing', 2024)  2.654        ('Williams', 2025)          4.163
fallback constant:         2.800
```

But `team_year_median_for(team, year)` (`pit_strategy_agent.py:308-321`) is
`self.team_year_median.get((team, year), self.team_year_median_fallback)` — an **exact**
`(team, year)` lookup with no same-team fallback to a prior year. When serving 2025 (the
only year this system runs live inference on), that exact-year entry exists for only 2
of the ~12 teams on the 2025 grid (`Haas F1 Team`, `Williams`) — and even for
`Williams`, the code reads the `(Williams, 2025)` entry (4.163, itself derived from only
a handful of in-scope 2025 stops so far) rather than the notebook-intended, richer
`(Williams, 2024)` prior. For every other team — including McLaren, Ferrari, and Red
Bull Racing, three of the grid's most active teams, all present in this audit's 5-race
sample — `team_year_median_for` falls straight past their own loaded, real 2023/2024
data to the flat **2.8 s** constant.

Measured: across the 284 served `_build_pit_duration_features` calls in this audit
(McLaren, Ferrari, Red Bull Racing across 5 races), `team_year_median` is **2.8000000...
on 100% of calls** (`std ≈ 8.9e-16`, i.e. exactly constant to floating-point noise) —
verified directly from the captured feature frame. This is the same defect #450 was
opened to fix (a frozen constant standing in for a per-team prior on every call), now
reproduced by a different mechanism: the fix computes real per-team data but the lookup
that reads it cannot reach 2025, the only year that is ever actually served.

### Finding N15-2 (LOW) — `tyre_life_in` is the one N15 feature that is already correctly guarded

Included for balance: `tyre_life_in`'s hard [0, 50] training clip is mirrored at serve
time by `_tyre_life_in()` + `_N15_TYRE_LIFE_ENVELOPE`, which clips to the same ceiling
and **logs a warning** identifying the violation before the clip erases the evidence
(`pit_strategy_agent.py:817-825`). Served values in this sample ranged 2-48 (0 escapes
by construction — clipping cannot produce an out-of-bound value). This is the
`OperatingEnvelope` pattern working exactly as designed, and the only N15/N12/N14/N06
feature in this whole audit with that protection.

---

---

## N26 — tire degradation TCN (sequence model, treated qualitatively per the brief)

A per-feature min/max table does not apply to a sequence model; the question is
whether the SEQUENCE served at inference matches what training produced, not whether
one scalar sits in a band.

**What defines a valid input sequence (established with confidence, from
`data/processed/tiredeg_sequence_config.json` + `tiredeg_feature_manifest.json` +
`tire_agent.py`):** one 42-feature timestep per lap of a **single (driver, stint)**,
window length fixed per compound bundle (`C1`=25, `C2`=31, `C3`=30, `C4`=26, `C5`=22,
`global`=28; `C6` has too few training stints (`n_stints: 4`) and falls back to the
global bundle by design). Stints shorter than the window are **left-zero-padded**;
stints longer are **truncated from the start**, keeping the most recent laps (the
config's own note: truncation does not blind the model to degradation, because
`TyreLife` still reflects real tyre age at every kept position). Features are scaled
with a `StandardScaler` fitted on 2023-2024 data, with NaN filled to 0.0 **before**
scaling (`scaler.transform(df[features].fillna(0))`, matching N09's `apply_scaler`
exactly).

**What the notebook produced at training time:** the same left-zero-pad / truncate-
from-start / per-stint grouping described above (N08/N09/N10), `masking: true` in the
config.

**What inference actually builds today:** `TireAgent._build_stint_tensor` (`tire_
agent.py:837-892`) reproduces the left-zero-pad exactly — `pad = np.zeros((window -
len(scaled), scaled.shape[1])); seq = np.vstack([pad, scaled])` — with a comment
explicitly citing the OLD, now-fixed behaviour: *"This used to tile the stint's first
lap instead... measured: the tile-pad put 87% of predictions outside the training
target's 5-95% band (mean -29.97 s of cumulative degradation against a band of
[-5.80, +2.46]), and flipped `warning_level` on 10.5% of laps."* This is the exact
defect this audit's brief cites as the motivating case (`envelope.py`'s docstring,
project memory's "N26 did that in roughly 87% of calls for two years") — **verified
here to already be fixed in the current checkout**, not still open. The comment also
notes the pad branch is not an edge case: "Windows are 28-31 laps and the median
stint is 21-23, so the pad branch is the COMMON path... measured at Barcelona 2024,
97% of calls take it" — meaning most real calls ARE padded, so getting the padding
right matters on nearly every call, not a rare tail case.

### Observation N26-A (not rated — architectural note, not fully verified against the training notebook) — the model's `mask` parameter is accepted but never used

`TireDegTCN.forward(self, x, mask: Optional[torch.Tensor] = None)` never references
`mask` in its body (`tire_agent.py:169-173`) — it is inert. `_build_stint_tensor`'s
caller (`predict_tire_deg_tool`, `estimate_laps_to_cliff_tool`) calls `model(tensor)`
with no mask argument, so `mask=None` on every real call. Whether this is benign
depends on whether N09/N10 used the sequence config's `masking: true` only to mask
padded positions OUT of the **training loss** (in which case a trained model never
needs a mask at inference, since it always reads the final — always real, never
padded — timestep via `x.transpose(1,2)[:, -1, :]`) or whether the architecture was
meant to consume a mask directly. This audit did not open N09/N10 far enough to
settle which; recorded honestly as unresolved rather than guessed.

### Observation N26-B (not rated — noted, not diagnosed) — parse-miss warnings concentrated on `tyre_life=1` laps, observed live during this audit's own drive

Driving the no-llm harness for the N12/N14 measurement above (which also runs N26 via
`_tire_no_llm`) logged repeated `"Tire tool output did not parse for {compound}
(tyre_life=1) — using conservative defaults instead of a 0.0 cliff"` warnings (12
occurrences across the 5-race, 284-lap sample, all but one at `tyre_life<=4`). This is
an existing, deliberately guarded fallback (#436: a parse miss returns a conservative
stub rather than a dangerous fabricated 0.0-laps-to-cliff, and logs why) — not a new
defect — but the concentration on very fresh tyres is a real pattern this audit
observed and did not chase further; worth a follow-up to confirm whether a 1-lap-old
stint produces a degenerate sequence some downstream regex/parse step chokes on.

---

### Note — N06's `Year` = 2025 on 100% of calls is NOT a bug

`Year` reads 2025 on every served call against a trained domain of {2023, 2024}. This
is the model's own designed generalisation test (N06 Step 8/9: train on 2023+2024,
evaluate on 2025) — the production system runs the model on exactly the year its own
notebook validated it against. Listed here so the escape percentage is not misread as
a defect; it is the one row in this table that is *supposed* to read 100%.

---

## Ranked findings

**HIGH**

1. **N06-1** — `Prev_LapTime` is a hardcoded 90.0 s constant on 100% of real replay-
   driven calls (raw `data/raw/` laps parquet has no `Prev_LapTime` column, so
   `RaceStateManager` always emits `None`). Measured downstream damage: lap-time
   prediction mean absolute error 6.97-13.83 s across real races, against the model's
   own reported test MAE of 0.392 s (18-35x worse). Circuit-dependent and directional
   (worst at Monaco, +12.47 s mean signed error; smallest at Lusail purely by
   coincidence, because Lusail's real lap times happen to sit near 90 s).
2. **N06-2** — `mean_sector_speed` (a per-circuit constant at training time) is
   substituted with `Prev_SpeedST` (a volatile per-lap reading) on 100% of calls;
   25.7% of served values additionally fall numerically outside the trained band.
3. **N12-1** — the no-llm overtake path calls N12 on gaps up to 21.8 s, 8.7x its
   2.5 s trained ceiling, on 54.2% of real calls — the scope check exists only as
   advisory LLM-prompt text, never enforced in code on the deterministic path.
4. **N14-1** — `n_drivers_delta` reaches +14 in production against a trained domain
   where it is provably `<= 0` (drivers cannot un-retire); `n_drivers` reaches 0
   against a trained floor of 7. Root cause: the featured parquet fed to the no-llm
   agents has per-lap gaps (N04's accuracy filtering), which N14's driver-count
   features were never designed to tolerate.
5. **N15-1** — `team_year_median` is the flat 2.8 s global fallback on 100% of served
   calls for McLaren/Ferrari/Red Bull Racing (three of the grid's most active teams),
   even though their real, materially different 2023/2024 medians (2.05-4.37 s) are
   already loaded in memory — the shipped exact-`(team, year)` lookup cannot reach
   2025 (the only year ever served), unlike the notebook's own designed same-team
   prior-year fallback.

**MEDIUM**

6. **N06-3** — `DriverNumber` served as 0 on 100% of calls, below the trained floor
   of 1 (the field is simply absent from the RSM driver dict).
7. **N12-2** — `gap_trend` swings ±200 s (trained band ±2.4 s) on 7.7% of calls, same
   root cause as N14-1 — a "one lap back" computation over a frame with lap gaps.
8. **N14-2** — `humidity` reaches 91% against a 74.65% trained ceiling on 28.9% of
   calls; likely genuine weather diversity between the sampled 2025 races and the
   2023-2024 SC-labeled training sample rather than a code defect — not fully
   diagnosed either way.
9. **N14-3** — the SC z-score features are trained non-causally (whole-race mean/std)
   and served causally (only-past mean/std); already tracked in project memory as a
   deliberate, not-yet-retrained mismatch, quantified here as a measured contributor
   to `lap_time_mean_z`/`lap_time_std_z` escaping their trained band 4.3%/8.2% of the
   time.

**LOW**

10. **N06-4** — `Prev_DegradationRate`/`Prev_CumulativeDeg`/`Prev_DegAcceleration` are
    hardcoded 0.0 on 100% of calls (0% min/max escape, since 0.0 sits near the
    trained distribution's centre — percentile 42.6-46.8% — which is why this reads
    quieter than N06-1/2 despite being the same class of defect: three features with
    real per-lap variation silently replaced with one constant, everywhere).
11. **N06-5** — early-stint coverage gap on `TyreLife`/`LapsSincePitStop`/
    `Prev_TyreLife`/`LapNumber` (1.8-7.7% escape); a real gap in what the training
    frame observed, not a code defect.
12. **N15-2** (positive finding, not a bug) — `tyre_life_in` is the one feature in
    this whole audit that is already correctly guarded end to end: hard-clipped to
    its trained [0, 50] ceiling with a logged warning on violation
    (`OperatingEnvelope` + `_tyre_life_in`).
13. **N26-A/B** — not rated. An architectural observation (the TCN's `mask` parameter
    is accepted but never referenced in `forward()`, and never populated by any
    caller) and a live-observed pattern (parse-miss fallback warnings concentrated on
    `tyre_life=1` laps) — both recorded as unresolved rather than diagnosed.

---

## What I tried to break, and could not

- **Every weather feature for N06** (`AirTemp`, `TrackTemp`, `Humidity`, `Rainfall`)
  — 0% escape across 284 real calls. Well covered by training.
- **Every categorical/domain feature checked against its trained domain** —
  `CompoundID`, `TeamID`, `Cluster`, `circuit_cluster`, `compound_x`/`compound_y`,
  `drs_window`, `FreshTyre`, `Rainfall`, `is_lap1`, `status_changed` — all stayed
  inside their trained categorical set in every sample drawn. No sentinel-collision
  pattern found in any of the four tabular models' numeric ID/flag features (the
  `Position`-defaults-to-0 collision documented in project memory for other call
  sites does not reappear in `PaceAgent`/`RaceSituationAgent`/`PitStrategyAgent`'s
  feature builders specifically — each reads `Position` as a plain, unguarded
  optional rather than defaulting it to a searchable value).
- **N26's historical padding bug** — went looking for the exact defect this audit's
  own brief describes (`envelope.py`'s motivating case, ~87% of calls fed a tiled-
  first-lap sequence) and could not reproduce it: `_build_stint_tensor` already
  implements the correct left-zero-pad, with an inline comment stating the fix and
  citing the same 87%/-29.97s/10.5% numbers project memory records. Confirmed fixed,
  not still open.
- **Crash-testing the whole harness** — 1,007 total feature-builder calls executed
  across this audit (284 N06 + 155 N12 + 284 N14 + 284 N15) produced **zero
  exceptions**. Every guard this audit exercised (`_live_drivers` liveness checks,
  missing-rival handling, wet-compound skip in `_build_undercut_features`'s sibling
  path, the N26 parse-miss fallback) held under real, unfiltered 2025 race data —
  the codebase does not crash on out-of-envelope input, which is a different property
  from answering correctly, and the two should not be conflated. The defects found
  here are all **silent wrong answers**, never a crash.
- **N15's `tyre_life_in` clip** — tried to find a value above 50 in the served range;
  impossible by construction (the clip is unconditional), which is exactly the point
  of Finding N15-2.

**What I did not verify, and why (scope limits, stated rather than silently
skipped):** this audit sampled 5 races (Lusail, Monza, Silverstone, Monaco,
Spa-Francorchamps) and one front-running driver per race (P1-P8 in this sample);
`Position` was never observed above 8, so a genuinely mid-pack or tail-of-the-field
serving pattern is untested here. `track_status_enc` reaching a red-flag/VSC code
(measured in an earlier, since-corrected pass of this same harness, before the
featured-vs-raw laps_df bug was fixed) did not reappear in the final 5-race sample —
plausible but unconfirmed that a race with a real SC/VSC/red-flag period (Qatar or
Spain 2025, per project memory) would reproduce it. N28's undercut model (N16) and
N30 (RAG) were out of this audit's requested scope and were not measured. The exact
numeric trained range of N15's `tyre_life_in`/`lap_number` distributions could not be
established with confidence from this checkout (the notebook's raw training frame is
never persisted to disk, and `data/raw/` here holds a small subset of the 2023-2024
calendar) — reported as not established rather than approximated as if it were.
