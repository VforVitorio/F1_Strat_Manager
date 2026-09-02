# Sprint 3 pre-measurement — the tyre channel is disconnected at a different seam than planned

**Date:** 2026-07-29 · **Issue:** #727 · **Status:** the sprint's premise is REFUTED; the diagnosis survives

Sprint 3 was specified as *"consume `TireOutput.deg_rate` as a per-lap cost"*. Before writing it, the
field was measured on the same 110 laps the epic uses (Lusail/NOR + Monza/LEC, `profile="no-llm"`,
zero LLM calls via `_NullReActRunner`). **The field named in the plan cannot carry the channel.**

The diagnosis behind the sprint is unaffected: the tyre channel really is disconnected, the
consequence really is that a worn tyre scores like a fresh one, and a properly connected channel
really would move the decision. Only the *seam* was wrong.

---

## 1. `TireOutput.deg_rate` is not the quantity a scorer needs

Measured over 110 laps:

| statistic | value |
|---|---|
| median | **+0.0058 s/lap** |
| mean | −0.0006 s/lap |
| negative | 43 of 110 laps |
| exactly `0.0` (also the parse-miss sentinel) | 12 of 110 |
| above `FRESH_GAIN = 0.25` | 6 of 110 |
| correlation with tyre life | **+0.115** |

Median by tyre-life band — **non-monotonic**, so it does not separate a worn tyre from a fresh one:

```
 0-5 laps   0.0000
 5-10      -0.0245
10-15      -0.0175
15-20      +0.0605
20+        +0.0478
```

Over the 5-lap window the median contributes **0.03 s = 0.02 positions**. Wiring it in would connect
a noise channel and let us report the tyre channel as "connected" while nothing changed.

**Why it looks like a degradation rate and is not one:** `predict_tire_deg_tool` reads
`feat_df['DegradationRate'].iloc[-1]` — the last row of a raw lap-to-lap derivative. Fuel burn-off
makes lap times *fall* through a stint at roughly the rate tyre wear makes them rise, so the raw
derivative sits on zero. Every other tyre quantity in the system is computed on the **fuel-adjusted**
series; this one is not.

## 2. The signal exists, is computed on every lap, and is thrown away

`predict_tire_deg_tool` (`src/agents/tire_agent.py:1054`) runs the TCN and produces `pred`, then
prints it:

```python
pred = model(tensor).item()
...
f"Cumulative degradation: {pred:.3f} s | Degradation rate: {deg_rate:.4f} s/lap"
```

`_parse_tool_outputs` (`:658-668`) has regexes for `Degradation rate`, `P10`, `P50`, `P90` — and
**none for `Cumulative degradation`**. `TireOutput` has **no field** for it. So the TCN's actual
prediction — the model N07-N10 exists to produce — reaches nothing: not the Monte Carlo, not the
orchestrator prompt, not the UI.

Measured, same 110 laps, captured by spying on the tool string:

| statistic | `cum_deg` | `deg_rate` |
|---|---|---|
| correlation with tyre life | **+0.369** | +0.133 |
| median, 0-5 laps | −0.308 | 0.000 |
| median, 20+ laps | **+0.103** | +0.048 |
| swing across a stint | **0.411 s/lap** | ~0.07 s/lap |

**0.411 s/lap across a stint is 2.06 s over the window = 1.37 positions** — larger than the flat
`FRESH_GAIN` term it would refine (0.83 positions). A correctly connected tyre channel moves the
decision materially, which is exactly why connecting the wrong one is worth avoiding.

## 3. The complication that stops this being a drop-in

`N04_feature_engineering.ipynb` defines the target the TCN predicts:

```python
baseline_tyrelife = group["TyreLife"].min()
baseline_laptime = group.loc[group["TyreLife"] == baseline_tyrelife, "LapTime_s"].mean()
FuelAdjustedDegAbsolute = adjusted - baseline_laptime
```

The baseline is **the freshest lap of that stint** — so the semantics are right ("seconds per lap
slower than this tyre was when new", fuel-corrected). But that lap is an out-lap or a standing-start
lap, and it is slow. The reference is therefore biased slow, and the measured level comes out
**negative on most laps** (median −0.161; −0.308 in the 0-5 band, where construction says it should
be ~0).

**The slope is trustworthy; the intercept is not.** Charging the level directly would pay STAY_OUT a
*negative* cost on most laps — rewarding staying out, the opposite of the fix.

## 4. Where that leaves the sprint

**Unambiguous, no decision needed:** carry `pred` onto `TireOutput` (new field + regex) and into the
orchestrator prompt. A model output that is computed and discarded is a defect independent of how the
scorer eventually uses it, and it unblocks every option below. Give it a `None` default, **never
`0.0`** — `deg_rate` already demonstrates the collision, with 12 of 110 laps carrying a parse-miss
that is indistinguishable from a real zero.

**Needs a decision, because the options differ materially:**

| option | what it costs | what it needs |
|---|---|---|
| **A. Ship the plan as written** (`deg_rate`) | ~0.02 positions | nothing — but it connects noise and lets a false claim into the record |
| **B. Consume `pred` relative to a fresh reference** | ~1.37 positions | a defensible fresh reference, i.e. a calibration step, not a code change |
| **C. Grade the cliff term instead of adding a rate** | unmeasured | uses only already-calibrated quantities (`laps_to_cliff_*`), but the ramp shape is an invented opinion |

Option B is the one that matches the diagnosis. Its blocker is real: the per-stint baseline
contamination is a modelling question, and inventing a correction constant would be exactly the
"rails encode opinions" failure this project has already paid for once.

## 5. What was tried and did not hold

- **That `FRESH_GAIN` and `deg_rate` double-count.** They do not, because `deg_rate` is not the same
  quantity — the double-count trap in the plan applies to option B, not to the plan's own option A.
- **That the raw parquet would answer the semantics question.** `FuelAdjustedDegAbsolute` and
  `CumulativeDeg` are the same column by different names (identical median and mean), which was worth
  confirming before treating them as two channels.
- **That the probe was cheap because of the profile name.** It was cheap because `no_llm.py` injects
  `_NullReActRunner`; `_run_always_on_agents_from_state` itself has no profile switch and calls the
  tyre agent unconditionally. Checked rather than assumed, because the provider is `openai`.
