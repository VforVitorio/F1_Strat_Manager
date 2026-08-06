# PR 5 — the overtake model has a domain, and it now says so

**Date:** 2026-08-05 · Extends `GATE_DATA_WIRING.md` F13 and its N27 notes, and
`GATE_801_ARTEFACTS.md` §6. Those are dated findings and stay as written.

## The defect

N11's pair builder **drops every pair more than 2.5 s apart before labelling** — "not an
active battle" (`.nb_py/N11_overtake_eda.py:233-235`). The model has no labelled example
beyond it. `predict_overtake_tool` had a lap-range guard and a roster guard but **no gap
guard**, so `_build_overtake_features` built a 9-second-gap row and LightGBM extrapolated.

### Re-measured rather than quoted

| | pairs | outside the domain |
|---|---|---|
| this sweep, 2025, N11's own pairing rule | 20,449 (median gap 2.06 s, p90 9.11 s) | **8,816 (43.1%)** — of THOSE, median 5.16 s, p90 15.13 s, max 91.1 s |
| the earlier gate | 25,215 | 10,565 (41.9%) |

The denominators differ — a differently filtered starting frame — and the magnitude does
not. Four in ten "can I pass the car ahead?" questions were answered from outside the
training set. The labelled artefact settles the bound directly: `overtake_pairs_2023_2025`
holds 28,494 rows, **max gap exactly 2.500, zero rows beyond**.

### Why this one is worse than a noisy number

`overtake_prob` does **not** reach the Monte Carlo. Traced: it reaches `threat_level`, the
**N31 prompt**, and the dashboards. An invented probability there is not averaged away, it
is *argued from* by the model writing the recommendation. And `overtake_prob` is **not** one
of the 14 fields of the frozen `StrategyRecommendation` contract, so widening it breaks no
frozen surface.

## The treatment, and why not the other two

Víctor's call, taken before any code was written: **`None` plus an explicit label**.

- **Label only** (the #710 envelope pattern) was the precedent, but #710 labels a *feature*
  fed to a regressor. This is a *probability* fed to a prompt: labelling it and passing the
  number anyway leaves the LLM the extrapolation to reason from.
- **Clamp to 0.0** was rejected for the reason two other fixes in this batch were made: a
  car 9 s ahead is not one you *cannot* pass, it is one the model has never been shown. And
  0.0 already means something else here — see below.

### `None` is not `0.0`, and that distinction is the design

Under a neutralisation, `overtake_prob = 0.0` is asserted by the **regulation** (Art. 55.8
bans overtaking; 56.6 under a VSC), and `_run_core` sets exactly that. Leaving the
no-answer case on 0.0 made "the rules forbid it" and "the model has no idea" the same
number to every consumer downstream.

So the neutralisation override still applies **whether or not the model answered** —
regulation beats absence — while an unscored pair on a green lap keeps its `None`.

## What changed

| site | before | after |
|---|---|---|
| `predict_overtake_tool` | scored any gap | returns `P(overtake) = UNKNOWN (gap X.XXs beyond N11's trained 2.5s domain)`; `gap` and `pace_delta` stay, they are measurements |
| `_parse_tool_outputs` | defaulted `overtake_prob` to **0.0** | defaults to `None` — the trap: with no number in the string, the old default would have quietly produced a real zero |
| `RaceSituationOutput.overtake_prob` | `float` | `Optional[float]` |
| `threat_level` | `None` would raise on `>=` | an unknown cannot RAISE the band, and does not suppress the SC terms |
| N31 prompt | `overtake={p:.2f}` | `overtake=unknown (cars farther apart than the model's trained range)` |
| `run_simulation_cli._add_situation_row` | `float(getattr(..., 0.0))` → **TypeError** | renders `overtake —` |
| arcade `format_situation` | `or 0.0` → "overtake 0%" | `overtake — (out of model range)` |
| backend `SituationResult` | `overtake_prob: float` | `Optional[float]` — documentation only, see below |
| webapp `SituationResult` | `number` | `number \| null` |
| webapp `AgentTabs` | `?? 0` → a dial reading 0% | an `n/a` placeholder |
| webapp `SituationResultView` | unguarded → JS coerces null to a confident "0%" | a "No prediction" panel |
| 4 doc surfaces | declared a plain float | declare the nullability |

`getattr(obj, name, default)` does **not** substitute for a present-but-None attribute. That
is the third form of this trap the project has now hit, after `dict.get` and `Series.get`,
and it is what made 35 of 57 Lusail laps error on the first real run of this branch.

### Two contracts, and one claim of mine that was wrong

They are different things and this document originally blurred them:

- **`StrategyRecommendation`, the frozen 14-field contract** (N31's output). Verified by
  reading `model_fields`: `action, reasoning, confidence, pit_lap_target, compound_next,
  undercut_target, pace_mode, target_lap_time_s, risk_posture, contingencies, key_risks,
  expected_stint_end, scenario_scores, regulation_context`. **`overtake_prob` is not among
  them**, so nothing frozen moved.
- **`SituationResult`, the backend model.** It declares the field, and it is **not wired as
  a `response_model` anywhere** — all seven routes in that file declare `StrategyResponse`,
  whose `result` is `Dict[str, Any]`.

The comment first written on that change said the old `float` would turn every out-of-domain
lap into a 500, "the same way #788 did". **That was false**: nothing validates against this
class, so it could not 500. The change is still right — it is the shape the webapp's
TypeScript mirrors by hand, and a declaration that lies about nullability is how the first
route to adopt it for real inherits the 500 — but the stated mechanism was not the
mechanism, which is the defect class this repo already has a name for.

## Two siblings fixed alongside, both measured first

**`pace_delta_rolling3` paired the two cars by array position.** The earlier gate flagged
this as "a rule difference with a bounded trigger. Not measured." It is measured now:
**10.78% of adjacent pairs (2,158 of 20,012)** have windows holding different laps, because
the featured frame drops pit, out and safety-car laps per driver.

And the rule itself was not "pair by LapNumber": N12 takes
`groupby(PAIR_KEYS).pace_delta_s.rolling(3, min_periods=1).mean()` over **the pair's own
series** (`.nb_py/N12_overtake_model.py:141-146`), with `gap_trend` as `.diff()` of the
pair's gap series. Inference computed a different quantity. `gap_trend` shared the same
window and is corrected with it rather than left as the next instance of the twin.

### The first version of that fix was itself skewed, and the gate refuted it

Pairing by LapNumber corrected the arithmetic and left a second, subtler train/serve skew —
**the same class the fix was correcting.** N12's rows are N11's LABELLED pairs, and N11 only
emits a row when the cars are position-adjacent **and** within 2.5 s. A lap where the pair
existed but sat 4 s apart, or where they had swapped order, is simply not in the series N12
rolled over. The first version rolled over every lap where both cars merely had a row.

Measured by the gate over the 11,633 in-domain 2025 pairs, both rules re-scored through the
real model and calibrator: the window content differed on **29.44%**, the `gap_trend` base on
**18.13%**, calibrated |Δp| max **0.480**, and **81 pairs crossed the MEDIUM band, 38 the
HIGH band** — larger than the 57-pair effect this report had treated as its headline. It bit
hardest on battles that had just closed up, which is exactly what the domain gate makes
interesting.

`_battle_series` now reconstructs N11's membership test term for term. The gap helper also
switched from the caller's `max(0.0, ...)` clamp to N11's `abs(...)`: with adjacency asserted
separately the two are equal, and the clamp was turning a swapped-order lap into a fabricated
zero-second gap.

**The unknown-circuit sentinel was `0`, a real cluster.** N11's is `-1`. The booster's
trained levels are `[0,1,2,3]`, so `-1` becomes the missing value LightGBM handles natively.
Latent — every race resolves today — but it was a live collision waiting for a keyspace miss.

**A prompt line that gave the LLM the wrong reason.** `_run_core` told the model "No car is
within overtaking range (gap > 2.5s)" whenever `rival_ahead` was None. That value comes from
a **position lookup with no gap filter at all**: it is None when the driver is leading, when
the car ahead is missing from the timing feed, or when our own position is unknown. Never
because of a gap. Corrected to say what is actually true.

## Verification

- `tests/agents/test_overtake_domain.py`, 13 tests. The domain bound is **re-derived from
  the labelled artefact**, not asserted against the constant. One test pins that the gate is
  not vacuous (it fires on four pairs in ten); another pins that an unknown does not
  *suppress* the safety-car path, which a lazy "return LOW when missing" fix would break.
- `ruff check` + `ruff format --check` green; `mypy src/rag/` green; webapp `tsc --noEmit`
  exit 0, 0 errors.
- Real `f1-sim` on Lusail: **all 57 laps OK**, `STAY_OUT·53 PIT_NOW·1 UNDERCUT·3` —
  identical to the `dev` baseline. The gate corrects the inputs without moving the decisions
  on that race.
