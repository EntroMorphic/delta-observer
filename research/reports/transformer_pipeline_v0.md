# Pipeline staleness on a standard model: results

*2026-10-30. Exploratory. The protocol and its amendments
([`transformer_pipeline_protocol.md`](transformer_pipeline_protocol.md))
were committed before the runs they cover.*

**Setup.**
- **Model and data:** a 0.82M-parameter character GPT (4 blocks, width
  128) on Tiny Shakespeare.
- **Pipeline:** 4 stages, PipeDream-style pipeline staleness, 2,000 Adam
  steps. D = 6 is plain 4-stage 1F1B.
- **Seeds:** 700–702. All 114 runs are complete.
- **How it was run:** D = 24 and D = 48 came from the original harness,
  which was stopped at 41 of 114 runs, at the user's request, for time.
  The rest came from a vectorized engine (amendment 2) that trains all
  runs at a given delay as one batched program on the GPU. It matches the
  original harness parameter for parameter over 12 steps, to within thread
  noise for Basis Rotation. The two sets of rows are pooled.

## Verdicts

| Prediction | Verdict |
|---|---|
| P1: harm is directional (twin ≤ ⅓ of stale excess) | **held** at every delay, D = 3 to 48 (ratio 0.08–0.27) |
| P2: no method in use recovers half the gap at the largest harmful D | **held** at D = 48 (best: Basis Rotation, 46%) |
| P3: the floor falls with the learning rate | **untestable**: both floors are 0, because even D = 3 is harmful |
| M1: the real error's curvature is at least 3× its twin's | **held** (950–2,000×) |
| M2: the real error points against the gradient | **held** (cosine −0.10 to −0.43) |

## Validation loss

Nats per character; mean of 3 seeds. Excess is the seed-paired gap to
synchronous Adam at the same learning rate (1.745 at lr 1e-3, 1.618 at lr
3e-3). The harm threshold is +0.05.

| lr 1e-3 | D = 3 | D = 6 | D = 12 | D = 24 | D = 48 |
|---|---|---|---|---|---|
| Stale | +0.11 | +0.30 | +0.61 | +0.70 | +0.74 |
| **Permutation twin (bound)** | **+0.03** | **+0.05** | **+0.06** | **+0.06** | **+0.06** |
| PipeMare learning-rate rescheduling | +0.09 | +0.18 | +0.36 | +0.63 | +0.54 |
| Weight prediction (SpecTrain-style) | +0.11 | +0.40 | +0.62 | +0.82 | +0.95 |
| Basis Rotation, against its own synchronous run (1.642) | +0.03 | +0.08 | +0.19 | +0.39 | +0.53 |

| lr 3e-3 | D = 3 | D = 6 | D = 12 | D = 24 | D = 48 |
|---|---|---|---|---|---|
| Stale | +0.24 | +0.72 | +0.83 | +0.90 | +0.94 |
| Permutation twin | +0.01 | +0.02 | +0.03 | +0.03 | +0.04 |

Stale seeds are tight: at lr 1e-3, D = 3 gives +0.104, +0.105, +0.127.

**Recovery of the gap between stale and the bound (lr 1e-3):**

| Method | D = 3 | D = 6 | D = 12 | D = 24 | D = 48 |
|---|---|---|---|---|---|
| Basis Rotation, against synchronous Adam | 229% | 126% | 94% | 64% | 46% |
| Basis Rotation, against its own synchronous run | 102% | 85% | 75% | 48% | 31% |
| PipeMare learning-rate rescheduling | 25% | 45% | 44% | 10% | 29% |
| Weight prediction | 6% | −42% | −2% | −19% | −30% |

Basis Rotation's first row is over 100% at small D because the optimizer
is better than Adam here even without staleness (1.642 against 1.745). Its
second row is the fair measure of how much staleness it removes.

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
  harm is directional at every delay tested. With its direction scrambled,
  the same error costs 0.03–0.06 nats per character. As is, it costs 0.11
  at D = 3 and about 0.7 at D = 24–48. The real error sees about 1,000× the
  curvature of its twin, against 9–16× on the small networks. It also
  points against the gradient throughout training.
- **There is no harmless delay for plain stale Adam here.** Even 3 steps
  costs 0.11 nats at lr 1e-3. Plain 1F1B (D = 6) costs 0.30. At lr 3e-3,
  D = 3 already erases the higher learning rate's synchronous advantage:
  stale reaches 1.857 at both learning rates.
- **The higher learning rate is hurt more at every delay** (+0.24 against
  +0.11 at D = 3; +0.72 against +0.30 at D = 6). That is the direction P3
  predicted, but P3 was defined on floors, and with both floors at 0 it
  can't be scored.
- **The methods in use leave most of the headroom at large delay.**
  Basis Rotation removes nearly all of the staleness harm at D = 3 and most
  of it at D = 6–12, but under half at D = 24–48. PipeMare rescheduling
  recovers 10–45%. Extrapolating the weights from the last update is
  neutral at best and usually worse. The likely reason is that Adam's step
  direction is a poor predictor of where the weights will be several steps
  later. That is an untested guess.
- **The bound is close to synchronous training here.** A
  direction-correcting method could in principle get within about 0.06
  nats of synchronous training even at 48-step delay.

## Caveats

- One small model, one dataset, 2,000 steps and 3 seeds.
- PipeMare and SpecTrain are our own simplified reimplementations, and
  neither was tuned.
- The learning rate of 1e-3 was not optimal even synchronously: 3e-3
  reached 1.618 against 1.745.
- The floor grid starts at D = 3. A floor between 1 and 2 would need
  another run.
- The mechanism probe uses a single seed, a single batch and a single
  delay (D = 24).

Data:
- `research/results/transformer_pipeline_{v0,eval_v0,probe_v0}.json`;
- `research/results/transformer_pipeline_vec/*.json` (the vectorized
  chunks).

Harnesses:
- `transformer_pipeline.py`;
- `transformer_pipeline_partial.py`;
- `transformer_pipeline_vec.py`;
- `transformer_pipeline_eval.py`;
- `transformer_pipeline_probe.py`.
