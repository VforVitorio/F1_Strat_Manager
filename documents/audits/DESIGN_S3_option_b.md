# Sprint 3 option B — consuming the tyre prediction without inventing a constant

**Date:** 2026-07-29 · **Issue:** #727 · **Status:** design settled, pending empirical validation of the reference

Companion to `MEASURE_S3_tyre_channel.md`, which refuted the sprint's original premise. That document
established what to consume; this one establishes **how**, and records two designs that were worked
out and rejected before the third.

---

## The problem in one line

The scorer needs *seconds per lap that staying out costs versus having a fresh set*. The model
supplies `cumulative_deg_s` — seconds per lap slower than **this stint's baseline lap**. Those are
not the same zero.

N04 fixes the baseline:

```python
baseline_tyrelife = group['TyreLife'].min()
baseline_laptime  = group.loc[group['TyreLife'] == baseline_tyrelife, 'LapTime_s'].mean()
FuelAdjustedDegAbsolute = adjusted - baseline_laptime
```

The stint's freshest lap is an out-lap or a standing-start lap: cold tyres, slow. So the reference is
biased slow, and the level comes out **negative on most laps** — median −0.161 over 110 measured
laps, and −0.308 in the 0-5 tyre-life band where construction says it should be ≈0.

**The slope is trustworthy; the intercept is not.** Charging the level directly would pay STAY_OUT a
negative cost on most laps, which is worse than the flat constant it replaces.

## Rejected: a pooled per-compound floor measured from the data column

Measured on 2023-2024 (45,327 laps, training seasons only, deliberately excluding the 2025 test
season for the reason `src/strategy/eval/hygiene.py` already documents):

| tyre life | SOFT | MEDIUM | HARD |
|---|---|---|---|
| 3 | −0.212 | −0.129 | 0.000 |
| 10 | +0.191 | +0.101 | +0.207 |
| 15 | +0.496 | +0.486 | +0.472 |
| 20 | +0.634 | +0.839 | +0.676 |

Clean, monotonic, physically sensible, and the floor of each curve is an obvious fresh reference
(SOFT −0.212 at life 3, MEDIUM −0.141 at life 4, HARD 0.000 at life 2).

**Rejected anyway, and this is the important part.** Those numbers come from the **data column**. The
code would subtract them from the **model's prediction**, and the two are not on the same scale: over
the same laps the model reports +0.103 at 20+ tyre life where the data says +0.68 to +0.84. Mixing
them is precisely the defect `CLAUDE.md` §11 records for 2026-07-27 — *"un umbral tuneado y una banda
de alerta no viven en la misma escala; confundirlos convierte una señal en una constante"*. That one
cost a whole signal (0/1420 and 0/8171 firings). Paying for the same mistake twice would be
indefensible.

## Rejected: a per-compound floor measured from the model's own predictions

Correct on the scale question, and it fits the repo's existing measured-table pattern
(`scripts/measure_mc_tables.py` → `data/mc_measured_v1.json`, with a test asserting the committed file
matches a fresh run).

Rejected on cost and on fit: that script is deliberately raw-parquet pandas and would have to grow a
TCN-loading dependency, and a **pooled cross-season constant is a worse reference than one this car
already provides** — see below.

## Chosen: the model against itself, same stint

Ask the TCN for its prediction at the current tyre life **and** at a fresh tyre life on the same
stint, and take the difference:

```
deg_cost_s_per_lap = pred(stint, tyre_life=now) - pred(stint, tyre_life≈3)
```

Why this is the right shape:

- **The baseline cancels exactly.** Both terms carry the same `baseline_laptime`, so the subtraction
  removes it algebraically rather than approximately. No calibration constant exists to be wrong.
- **Same scale by construction.** Both sides are model output, so the scale question cannot arise.
- **A better reference than any pooled table.** It is *this car's own fresh pace, on this set, at this
  circuit, in these conditions* — not a cross-season average over every driver and track.
- **No new artefact.** No JSON, no script, no committed table to regenerate and drift.
- **Cheap.** One extra deterministic forward pass (`model.eval()`, no MC sampling) per lap.

### The one real risk, and why it is acceptable

At tyre life ≈3 the stint tensor is mostly padding. That was a genuine defect once — inference padded
by replicating the first lap while N09 trained with `np.zeros` (`CLAUDE.md` §11) — and it is fixed, so
inference now pads the way training did. Every stint in training also contains its own laps 1-3, so
the model has seen that regime rather than being extrapolated into it.

**This must be validated before the code lands, not asserted.** The check: over the same 110 laps,
`pred(now) − pred(3)` should be (a) non-negative on the large majority and (b) rising with tyre life.
If it is not, the padding concern is real and the rejected model-scale table becomes the fallback.

## What happens to FRESH_GAIN

`FRESH_GAIN = 0.25  # s/lap advantage of fresh vs degraded tyre` is **the same quantity**, hardcoded.
So this is a replacement, not an addition, and the double-count trap the plan warned about is avoided
by that fact rather than by a correction term.

The arithmetic, worked out rather than assumed:

- today, no stop: `worn·cliff_loss`
- today, stop at offset *k*: `pit_loss + worn·cliff_loss − (racing−k)·0.25 − …`

Charging `deg_cost` on the laps run on the old set and dropping the post-stop fresh credit shifts
**every plan by the same `racing·deg_cost`** when `deg_cost = 0.25`. A uniform shift across all four
candidates cancels in every comparison, so **the no-signal fallback path is argmax-identical to
today** — which is exactly the property that makes the change safe to ship and testable against the
existing goldens.

`FRESH_GAIN` therefore survives as the fallback when `cumulative_deg_s` is `None`, in the same spirit
as the module's other `DEFAULT_*` constants: not an invented number, but the value the system held
before the measurement existed.

## Both scorers, and why that is not optional

`simulate_lap_window` is not a golden-pinned relic. Three shipping builders hardcode `"rivals": []` —
`engine.py:449`, `strategy_orchestrator.py:2356`, and **`src/telemetry/backend/api/v1/endpoints/
strategy.py:882`** — and `_has_usable_gaps` also demotes to the legacy path when every gap is NaN. So
**the backend endpoint runs the legacy scorer in production.**

Connecting only the projection branch would fix arcade and the CLI and silently leave the backend on
the old behaviour: the twin defect this repo keeps paying for. Both move, and the legacy golden is
re-frozen deliberately rather than treated as breakage.

## Out of scope, stated so it is not mistaken for an oversight

The 5-lap window still prices 100% of a stop's cost against ~5 laps of its benefit. That is the third
cause from the epic and it is #729's subject, to be decided with a measurement rather than folded in
here.
