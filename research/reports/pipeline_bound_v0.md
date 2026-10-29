# Pipeline staleness: Basis Rotation against the permutation bound (exploratory)

**Status: exploratory.**
- **Harness:** `research/experiments/pipeline_bound.py`, with tests.
- **Raw results:** `research/results/pipeline_bound_v0.json`.
- **Data:** seeds 400–404, one arm per table, so **5 runs per cell**.
- **Training:** Adam's learning rates, 5-epoch warmup, 120-epoch cap.

## Why this test

[`basis_rotation_bound_v0.md`](basis_rotation_bound_v0.md) used a cross-stage
staleness model, where each stage sees itself live and the others stale. A
per-matrix rotation cannot target that. This test uses the delay model Basis
Rotation is designed for, the PipeMare abstraction that its analysis also
uses:
- stage k's gradient is computed at the *whole model's* weights from
  τ_k = s·(K−1−k) steps ago, its own weights included;
- the first stage is the most delayed and the last is fresh;
- the maximum delay is D = 5s.

**Permuted:** the fresh gradient plus the real pipeline error, with each
tensor's entries scrambled by a fixed permutation. It is the bound for any
correction of the error's direction.

## Results

Each cell gives fits out of 5 and the median epochs to fit.

### Adder chain (width 4), lr 1/300

| Optimizer | Condition | Fresh | D=5 | D=10 | D=20 | D=40 |
|---|---|---|---|---|---|---|
| Adam | stale | 4, 11 | 5, 16 | 1, 41 | 0 | 0 |
| Adam | permuted | | 4, 12 | 4, 11 | 4, 12 | 5, 11 |
| Basis Rotation | stale | 5, 7 | 4, 9 | **5, 19** | 0 | 0 |
| Basis Rotation | permuted | | 4, 7 | 5, 9 | 4, 9 | 5, 8 |

### Plain 64-wide pipeline, lr 1/300

| Optimizer | Condition | Fresh | D=5 | D=10 | D=20 | D=40 |
|---|---|---|---|---|---|---|
| Adam | stale | 5, 25 | **0** | 0 | 0 | 0 |
| Adam | permuted | | 5, 28 | 5, 25 | 5, 28 | 5, 24 |
| Basis Rotation | stale | 5, 8 | **5, 24** | **5, 48** | 1, 104 | 0 |
| Basis Rotation | permuted | | 5, 29 | 5, 30 | 5, 31 | 5, 28 |

### lr 1/100

On both arms, every stale run fails from D=5, for Adam and for Basis
Rotation. Every permuted run fits 4–5/5 at roughly fresh speed, up to D=40.

## Findings

1. **Under pipeline staleness, nearly all of the harm is directional.**
   - With the error's entries scrambled, training fits at fresh speed up to
     D=40, for both optimizers, both arms and both learning rates.
   - With the real error, Adam fails from D=10 on the chain and from D=5 on
     the 64-wide pipeline.
   - So the permutation bound sits at 8× or more beyond Adam's floor. That
     gap is larger than in the cross-stage model, where it was 2–8×.
2. **In its design regime, Basis Rotation closes part of the gap.** Taking
   the floor as the largest delay with at least 4/5 fits:

   | Arm | Adam floor | Basis Rotation floor | Bound | Share of the log-delay gap closed |
   |---|---|---|---|---|
   | 64-wide pipeline | below 5 (0/5 at D=5) | **10** (5/5) | ≥ 40 | at least 1 doubling of at least 3, ≈ ⅓ |
   | Adder chain | 5 | 10 | ≥ 40 | 1 of at least 3, ≈ ⅓ |

   - This is the first method we have tested that recovers any directional
     gap. In the cross-stage model, DC-ASGD, exact first-order Taylor and
     Basis Rotation recovered none.
   - Even here it leaves most of the gap: at least 4× in delay between Basis
     Rotation's floor and the bound.
3. **At the faster learning rate, neither optimizer survives any pipeline
   delay.** The permuted runs still do. This fits the margin picture: the
   faster rate leaves no room for the directional error.

## What this means

- **The permutation bound separates two questions.**
  - *How much of staleness harm is directional?* Under pipeline delay,
    nearly all of it.
  - *How much does a method recover?* Basis Rotation recovers about a third,
    in log-delay, of what is available. First-order compensation recovers
    none.
- **Two results from the earlier test now read differently.** Basis Rotation
  is helpful in the regime it targets. It failed to recover direction in the
  cross-stage test because that test was outside its design.
- **Headroom remains.** No tested method approaches the bound, so a method
  that neutralized the misaligned error fully would be worth at least 4×
  more tolerable pipeline delay here.

## Caveats

- 5 runs per cell, tiny networks, exploratory, and a reimplementation with
  two unstated details chosen by us.
- The staleness model has no weight stashing: forward and backward use the
  same snapshot per stage.
- The bound is not attainable by any real method as such. Permuting needs the
  fresh gradient. It is a measuring stick, not a method.

## Reproduce

```bash
python -m unittest research.tests.test_pipeline_bound
python -m research.experiments.pipeline_bound
```
