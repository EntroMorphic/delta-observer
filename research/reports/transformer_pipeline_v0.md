# Pipeline staleness on a standard model: results

*2026-10-30. Exploratory. The protocol and its amendment
([`transformer_pipeline_protocol.md`](transformer_pipeline_protocol.md))
were committed before the runs they cover.*

**Setup.**
- **Model and data:** a 0.82M-parameter character GPT (4 blocks, width
  128) on Tiny Shakespeare.
- **Pipeline:** 4 stages, PipeDream-style pipeline staleness, 2,000 Adam
  steps.
- **Seeds:** 700–702.
- **Scope:** the grid was stopped at 41 of 114 runs, at the user's request,
  for time. Only D = 24 and D = 48 are complete, so P3 (the floor falls with
  the learning rate) is **untestable**.

## Verdicts

| Prediction | Verdict |
|---|---|
| P1: harm is directional (twin ≤ ⅓ of stale excess) | **held** at both delays (about 1/12) |
| P2: no method in use recovers half the gap at the largest harmful D | **held** at D = 48 (best: Basis Rotation, 46%) |
| P3: the floor falls with the learning rate | untestable (grid stopped) |
| M1: the real error's curvature is at least 3× its twin's | **held** (950–2,000×) |
| M2: the real error points against the gradient | **held** (cosine −0.10 to −0.43) |

## Validation loss

Nats per character; mean of 3 seeds, standard deviations 0.003–0.047.
Excess is the seed-paired gap to synchronous Adam at the same learning
rate.

| lr 1e-3 | D = 24 | D = 48 |
|---|---|---|
| Synchronous Adam | 1.745 | 1.745 |
| Stale | 2.443 (+0.70) | 2.490 (+0.74) |
| **Permutation twin (bound)** | **1.804 (+0.06)** | **1.806 (+0.06)** |
| PipeMare learning-rate rescheduling | 2.379 (+0.63) | 2.289 (+0.54) |
| Weight prediction (SpecTrain-style) | 2.565 (+0.82) | 2.696 (+0.95) |
| Basis Rotation | 2.034 (+0.39 over its own 1.642) | 2.176 (+0.53 over its own) |

| lr 3e-3 | D = 24 | D = 48 |
|---|---|---|
| Synchronous Adam | 1.618 | 1.618 |
| Stale | 2.522 (+0.90) | 2.553 (+0.94) |
| Permutation twin | 1.652 (+0.03, 2 seeds) | 1.663 (+0.05) |

**Recovery of the gap between stale and the bound:**

| Method | D = 24 | D = 48 |
|---|---|---|
| Basis Rotation | 64% (about 48% against its own synchronous run) | 46% |
| PipeMare learning-rate rescheduling | 10% | 29% |
| Weight prediction | −19% | −30% |

## Mechanism

Stale Adam at D = 24, seed 700. The probe compares the real error with its
twin on one fixed training batch:

| Step | Curvature, real/twin | cos(real, g) | cos(twin, g) | \|e\| / \|g\| |
|---|---|---|---|---|
| 250 | 955× | −0.43 | −0.001 | 0.82 |
| 500 | 2,011× | −0.24 | −0.000 | 0.54 |
| 1000 | 1,243× | −0.18 | +0.000 | 0.37 |
| 2000 | 1,510× | −0.10 | −0.001 | 0.46 |

## What this means

- **The small-network result transfers, and gets stronger.** On a
  standard transformer with real text, almost all of the pipeline-staleness
  harm is directional. With its direction scrambled, the same error costs
  about 0.06 nats per character. As is, it costs about 0.7. The real error
  sees about 1,000× the curvature of its twin, against 9–16× on the small
  networks. It also points against the gradient throughout training.
- **The methods in use leave most of the headroom.** At D = 48 the best one
  tested (Basis Rotation) recovers under half the gap. PipeMare
  rescheduling recovers 10–29%. Extrapolating the weights from the last
  update makes things worse. The likely reason is that Adam's step
  direction is a poor predictor of where the weights will be 24–48 steps
  later. That is an untested guess.
- **The bound is close to synchronous training here.** A
  direction-correcting method could in principle get within about 0.06
  nats of synchronous training even at 48-step delay.

## Caveats

- One small model, one dataset, 2,000 steps and 3 seeds.
- Only large delays were completed. D = 24–48 stands for deep or
  heavily asynchronous pipelines. Plain 4-stage 1F1B (D = 6) was not run.
- PipeMare and SpecTrain are our own simplified reimplementations, and
  neither was tuned.
- The learning rate of 1e-3 was not optimal even synchronously: 3e-3
  reached 1.618 against 1.745.
- The mechanism probe uses a single seed and a single batch.

Data:
- `research/results/transformer_pipeline_{v0,eval_v0,probe_v0}.json`.

Harnesses:
- `transformer_pipeline.py`;
- `transformer_pipeline_partial.py`;
- `transformer_pipeline_eval.py`;
- `transformer_pipeline_probe.py`.
