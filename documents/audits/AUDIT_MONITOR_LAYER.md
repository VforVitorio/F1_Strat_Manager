# Audit: the MONITOR layer, and why it should not be built

**Date:** 2026-07-26 · **Verdict: do not build it.** The system already does what it was meant to add, under a different name.

Three independent audits, all offline (no LLM calls), all report-only. Raw reports at `~/.claude/plans/monitor-audit/`.

---

## The proposal

Stop emitting a full strategy decision every lap; speak only when there is something to say. The stated problem: if the tool says PIT_NOW or STAY_OUT with equal conviction 55 times, the user stops reading, so the noise erodes the calls that matter.

The product argument was right. The proposed mechanism was aimed at the wrong layer.

---

## A: does the problem exist? Measured over 415 real laps

7 driver-races, 5 circuits, 2 seasons, deterministic profile.

| quantity | value |
|---|---|
| action changes | **2 in 415 laps (0.49 %)** |
| races with zero changes | **6 of 7** |
| laps emitting STAY_OUT | **414 of 415** |
| the one meaningful change | Lusail 2025 lap 44, **exactly Norris's real stop lap** |
| recommendations changed by injecting the real Safety Car | **0 in 230 laps** |

**The deterministic layer a MONITOR would gate is already silent.** A threshold above it would gate something that does not speak.

The caveat that matters: the `rich` profile is the default and **11 of its 14 output fields are LLM-authored**. Any noise a user perceives is generated ABOVE the Monte Carlo, in the synthesis. A gate on the MC layer would not touch it.

---

## B: where does the threshold go? The proposed rule cannot discriminate

Measured: the margin rule scores **AUC 0.5707** against a 0.5 coin flip, never exceeds lift 1.39 at any threshold, and prefers STAY_OUT on **89 % of laps where a driver really stopped**. The best alternative found (a fact-based rule on tyre life, a rival in the pit lane, or neutralisation) silences 64.3 % of laps but **misses 620 of the 1810 real stops, 34.3 %**.

### The framing correction, which matters more than the numbers

The audit derived a seconds break-even: a stop costs about 22.8 s and recovers at most 0.4735 s/lap (0.25 fresh tyre + 0.2235 measured clean air), so it never pays back inside a 5-lap window. Recomputed independently, that break-even is **~48 laps**.

**That analysis measures a currency the layer stopped using.** `position_projection.payoff` states it outright:

> Positions are the currency, which is the point of the redesign.

The Monte Carlo does not ask *"does this stop pay back in seconds"*. It asks *"where do I rejoin, in track position, over the next few laps"*, and that question **is** answerable in a short window. The 86.5 % validation over 1810 real stops uses a **2-lap** horizon, not 48.

So `STAY_OUT` winning on 89 % of real stop laps is **not a failure**. Within five laps a stop genuinely does cost places, roughly 20 s of deficit at a measured 2.226 s per position. The real driver stopped anyway because they were optimising a longer horizon than the one the layer models. Two different questions, and the layer answers its own correctly.

**Do not lengthen the window.** The short horizon is the design, the lap-by-lap character of the product depends on it, and a 22 or 48-lap window would be most of a race at many circuits.

---

## C: what would break? Less than feared, and one ugly thing

**17 consumers traced. Only 4 need the contract to change:** the CLI, the arcade, and both backend `/simulate` profiles. `/recommend`, the MCP tool and the webapp are structurally outside the blast radius because they bypass `run_lap` and call the orchestrator directly.

**The silence decision could be centralised** in `src/strategy/inference/engine.py::run_lap`, which is already the sole funnel for the three live per-lap surfaces. That matters given this repo's dominant defect: one decision in one place, not four waiting to drift.

**The ugly thing**, verified verbatim in both files:

```python
action = str(getattr(rec, "action", "ERROR"))
```

`src/arcade/strategy.py:807` and the submodule's `simulator.py:523`. A deliberately silent lap would render as a **red ERROR badge** while the dashboard status bar simultaneously shows `lap N · streaming`. Silence and failure are today indistinguishable. Not a live bug while nothing is deliberately silent, but the first thing to fix if this is ever revisited.

---

## The decision

**Do not build the MONITOR layer.**

`STAY_OUT` already IS monitoring, under a different name. 414 of 415 laps emitting it is not noise, it is the layer correctly reporting that nothing should change in the next few laps. Wrapping that in a second mechanism adds a suppression rule, four consumer contracts and a silence-versus-failure ambiguity, in order to hide output that is already both correct and quiet.

**What to do instead, and it is much smaller:** change how the **LLM synthesis** frames a sustained run of `STAY_OUT`. The audit showed the noise is generated there, not in the Monte Carlo, and 11 of 14 fields are written fresh every lap regardless of whether anything changed. The prompt can say what a long STAY_OUT run means, that the system is watching rather than deciding, and reserve the full decision-brief register for the laps where the call actually changes.

That is a prompt change on one surface, not a layer across four.

---

## Bugs this audit surfaced, which was not its job

- **#645** — an exact Monte Carlo tie is resolved by dict insertion order. Six near-tie laps were found in 415, **all six on decision laps, three of them exact 0.000 ties on real pit stops**. The margin is smallest exactly where the decision is hardest, which also means any epsilon-threshold rule would mute the stop calls first.
- The duplicated `"ERROR"` default above, which is the fifteenth instance of this repo's twin-drift pattern.

## What was NOT verified

The `rich` LLM profile, by mandate: these audits were offline. Since that is the default profile and it authors 11 of 14 fields, **the layer where the remaining work actually lives is the one none of the three measured.** Any follow-up on the synthesis framing needs its own measurement.
