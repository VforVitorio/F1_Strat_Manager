# prompt_ab — measuring a Layer 3 prompt change

A change to `_build_orchestrator_prompt` moves 12 of the 14 `StrategyRecommendation`
fields, and no test asserts on any of them. This harness is how you find out whether a
prompt edit did anything, and it exists because the first attempt to answer that
question needed six goes to get right.

## The one thing you must not skip

**Run the unchanged prompt TWICE and diff those two runs first.** That is the noise
floor, and on this stack it is enormous: `OrchestratorCFG.temperature` is requested and
discarded by the client for the gpt-5.x family, so Layer 3 samples at the provider
default. Two identical passes over 41 laps disagreed on `confidence` in 36 laps, on
`pit_lap_target` in 23, and produced opposite actions on the one lap where the call
actually changed.

Without the floor, every number the harness prints looks like a result.

## The four stages

| stage | what it does | API calls |
|---|---|---|
| `gen_inputs` | drive real laps on the `no-llm` profile, pickle the per-lap orchestrator inputs | **0** |
| `run_pass` | sweep the cached race under one prompt variant | 1 per lap |
| `run_repeats` | one lap, N times, both variants — the experiment with statistical power | 2N |
| `analyse` | diff two passes against a floor, plus the within-pass statistics | **0** |

## A full run

```bash
# 1. cache the inputs (free, ~3 min, needs data/ and the model weights)
python -m scripts.prompt_ab.gen_inputs --gp Lusail --driver NOR --team McLaren \
    --year 2025 --laps 5-45 --out data/eval/prompt_ab/lusail_nor.pkl

# 2. two runs of the unchanged prompt (the floor) and one of the variant
python -m scripts.prompt_ab.run_pass --inputs data/eval/prompt_ab/lusail_nor.pkl \
    --variant none   --out data/eval/prompt_ab/pass_a.json
python -m scripts.prompt_ab.run_pass --inputs data/eval/prompt_ab/lusail_nor.pkl \
    --variant none   --out data/eval/prompt_ab/pass_a2.json
python -m scripts.prompt_ab.run_pass --inputs data/eval/prompt_ab/lusail_nor.pkl \
    --variant memory --out data/eval/prompt_ab/pass_memory.json

# 3. read it
python -m scripts.prompt_ab.analyse --floor-a data/eval/prompt_ab/pass_a.json \
    --floor-b data/eval/prompt_ab/pass_a2.json --other data/eval/prompt_ab/pass_memory.json

# 4. the decision lap, with repeats
python -m scripts.prompt_ab.run_repeats --inputs data/eval/prompt_ab/lusail_nor.pkl \
    --history data/eval/prompt_ab/pass_a.json --lap 44 --repeats 10 \
    --out data/eval/prompt_ab/transition.json
```

Three `run_pass` invocations can run in parallel; each is network-bound. Roughly 25
minutes and ~120 calls for the three together.

## How to read it

- **Per-field diff**: only believe a field whose `signal` clears `noise` by a wide
  margin. Deltas of a few laps are nothing.
- **Within-pass statistics**: these carry no cross-pass sampling noise and are where the
  memory effect actually appeared — distinct contingency triggers over a race (80 without
  memory, 6 with) and total `pit_lap_target` movement (311 laps against 214).
- **Repeats**: report the count, the variant split and a Fisher exact test. n=10 cannot
  separate a 6/10 from a 4/10.
- **Read the no-memory arm before the result.** If it is already at 0/n or n/n, the lap has
  no room to move and the comparison measures nothing. That is what happened to the lap-44
  anchoring experiment on `--model gpt-4.1-mini`: both arms stayed out 10 of 10, which is a
  degenerate experiment and not evidence that memory is harmless.

`--model` overrides `CFG.model_name` for one run. It exists because the shipped model
discards `temperature`, so the only way to ask "does this survive without the sampler" is to
measure a model that keeps it. Cross-model results are directional, never absolute.

## Deviations you must state with any result

- `no-llm` never runs the conditional agents, so `pit_out` is `None` and
  `regulation_context` empty. Production populates them **only** when N28/N30 are routed:
  2 of 41 green laps at Lusail 2025, and **every** lap of a `--safety-car` run. Those
  prompts are not production-shaped, and an absolute rate measured on them is not a
  product finding.
- One circuit, one driver, one race per cache.

## Adversarial runs

`--safety-car` on `gen_inputs` injects a Safety Car through an RCM message on every lap,
the way `RaceControlStateTracker` does on the live surfaces. N27 responds correctly
(`overtake_prob` 0.00 per Art. 55.8) and the deterministic Monte Carlo flips to `PIT_NOW`,
which gives you a lap where changing the call is the right answer — the case a memory
block is most likely to break.

## Where the outputs go

`data/eval/prompt_ab/`, which has its own `.gitignore` rule. Note that **`data/eval/` is
otherwise tracked** — it holds the published measurement tables — so do not assume anything
you write under `data/` is ignored. The reports that cite these numbers live in
`documents/audits/`.
