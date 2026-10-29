# Stability margin, prior art and the permutation bound (exploratory)

**Status: exploratory.** This follows the Lincoln Manifold synthesis in
`journal/findings_synth.md` (gitignored).

| Part | Harness | Raw results |
|---|---|---|
| F1 | `research/experiments/sharpness_floor.py` | `research/results/sharpness_floor_{sharpness,floors,predict}_v0.json` |
| F1b, post hoc | `research/experiments/block_stability.py` | `research/results/block_stability_v0.json` |
| Compensation | `research/experiments/compensation_bound.py` | `research/results/compensation_bound_v0.json` |

All harnesses have tests.

## The hypothesis

A delay d is tolerable along a direction of curvature λ only if lr·λ stays
below the delayed-feedback boundary of the linear recurrence:
- without momentum, 1.0, 0.35 and 0.095 at d = 1, 4 and 16;
- with momentum 0.9, about 10× stricter.

Synchronous training sits at the edge of stability, lr·λ ≈ 2. So it should
have no delay margin, and the floor should be predictable from sharpness.

## F3. Prior art

- **The margin theory is published.** Giladi, Nacson, Hoffer and Soudry,
  ICLR 2020, "At Stability's Edge", analyze asynchronous SGD through
  dynamical stability.
  - The learning rate must shrink inversely with the delay to keep the same
    minima accessible.
  - Momentum should be turned off or modified.
- **The eigenbasis mechanism is close to published.** Jung, Shin and Lee,
  ICML 2026 (arXiv 2602.03515), "Basis Rotation":
  - Delayed-gradient damage comes from misalignment between the Hessian
    eigenbasis and the coordinate axes, which makes coordinate-wise adaptive
    optimizers oscillate.
  - Their fix rotates the optimizer into the estimated Hessian eigenbasis.
  - They report 76.8–81.7% fewer iterations than the best asynchronous
    baseline at 1–3B parameters.
  - This covers most of the subspace-damping idea (F2), so F2 is not run as
    a new method.
- **Not found:**
  - the permuted-error ("permutation twin") control;
  - its use as a ceiling for compensation methods;
  - an analysis of delay *interacting with* edge-of-stability
    self-stabilization.

## F1. Sharpness predicts the floor: killed

**Pre-registered mapping**, committed at `22ef6f7` before any sharpness was
measured:

| Optimizer | a | Momentum in the model |
|---|---|---|
| SGD(β) | lr·λ_max(H) | β |
| Adam | lr(1−β₁)·λ_max(P^½ H P^½) | 0.9 |
| Muon | ‖Δθ‖/‖g‖·λ_max(H) | 0 |

The predicted floor is the largest grid delay whose boundary exceeds the
peak value of a along fresh trajectories (seeds 500–504, adder chain width 4).

**Result: 0 of 8 settings within 2×.** The model predicts floor 0
everywhere, against measured floors of 1–8. The cells are:

| Setting | Peak a (median) | Predicted floor | Measured floor |
|---|---|---|---|
| SGD 0.1, β 0.9 | 0.79 | 0 | 4 |
| SGD 0.3, β 0.9 | 1.94 | 0 | 1 |
| SGD 1.0, β 0 | 3.49 | 0 | 8 |
| SGD 3.0, β 0 | 9.0 | 0 | 0 (fresh fits only 1/10) |
| Muon 0.01 | 11.7 | 0 | 8 |
| Muon 0.03 | 14.6 | 0 | 4 |
| Adam 1/300 | 34 | 0 | 4 |
| Adam 1/100 | 218 | 0 | 4 |

The Adam values are dominated by near-dead coordinates, where √v ≈ 0 inflates
the preconditioner.

**The momentum check points the predicted way, but weakly.**

| Setting | Stale floor | Permuted floor |
|---|---|---|
| SGD, momentum 0.9, lr 0.1 | 4 | 8 |
| SGD, no momentum, lr 1.0 (the same effective step) | **8** | 16 |

Removing momentum doubles the floor; the linear model predicted about 10×.

## F1b. A block model that delays only cross-stage curvature: also killed

This part is post hoc. Our staleness model shows each stage its own
parameters live, and the curvature study had found 74–88% of curvature
within modules. So the model:
- delays only the cross-stage blocks;
- works in the top-20 eigensubspace, with H split into a block-diagonal part
  A_in and a cross-stage part A_cross;
- tests the spectral radius of x_{t+1} = x_t + β(x_t − x_{t−1}) −
  η(A_in x_t + A_cross x_{t−d});
- takes the minimum floor over epochs 2–16, on seeds 500–502.

**Result:** it predicts floor 0 in all 15 runs, against measured floors of
1–8. The cross-stage share of the curvature is 0.15–0.25.

| Setting | Median η·λ_max |
|---|---|
| SGD 0.1, β 0.9 | 0.44 |
| SGD 0.3, β 0.9 | 1.37 |
| SGD 1.0, β 0 | **1.95** |
| Muon 0.01 | 10 |
| Muon 0.03 | 14 |

**Why both models fail.** Momentum-free SGD sits at η·λ ≈ 1.95, essentially
on the synchronous boundary of 2. Linear theory calls these runs marginal
*with no delay at all*, yet they train, held by nonlinear self-stabilization
at the edge of stability (Damian et al.).
- **No linear stability model can set the floor in this regime.** The floor
  is set by how much delay that nonlinear mechanism absorbs.
- **Muon's mapping is not meaningful.** Its effective η·λ of 10–14 shows that
  fixed-norm steps are not a linear step size.

**What survives is qualitative:**
- harm lives in sharp directions;
- direction matters;
- the floor moves with learning rate × delay;
- momentum costs tolerance (2× here).

## F1c. A gradient-based predictor: also killed

This is a follow-up. The question: can a signal measured directly from
gradients predict the floor, since it already includes the nonlinear
dynamics?

- **The signal:** the descent ratio r(d) = g_stale(d)·g / ‖g‖². It is the
  share of the true gradient that the d-step-stale gradient (all-interface
  staleness) still points along.
- **Where it is measured:** on a 1024-example probe batch, every epoch from 2
  to 16 of a fresh trajectory, on seeds 500–502, for the same eight settings.
- **The rule,** committed at `a64c2f1` before running:
  - the predicted floor is the largest delay at which r > 0 at every
    checkpoint;
  - r > 0.5 is the secondary.
- **Harness and raw results:** `research/experiments/gradient_floor.py` and
  `research/results/gradient_floor_v0.json`.

**Result.**
- **Primary: 1 of 8 within 2×,** and that one is SGD 0.1, predicted 8–16
  against a measured 4.
- **The rule is far too optimistic.** It predicts 16–64 where the measured
  floors are 0–8.
- **It does not even order the settings.** SGD 0.3 (measured floor 1) gets
  32–64.
- **Secondary: 3 of 8,** but individual seeds disagree by up to 16×; for
  example 2, 4 and 32 for SGD 1.0 without momentum.

**Why it fails.** The stale gradient still points downhill at every probe,
long after training under that delay would fail. Failure is a *closed-loop*
instability: the stale trajectory feeds its own error back and the
oscillation grows. A probe of the fresh trajectory is open-loop. It measures
one step's misdirection, not the growth of a feedback loop. With linear
theory also failing (F1, F1b), the evidence so far says the floor has to be
measured with short closed-loop stale runs. It cannot be read off a fresh
run.

## F1d. A closed-loop stress test: also killed, from the other side

This is a follow-up to F1c, which suggested that the floor must be measured
in closed loop.

- **The rule,** committed at `4e090fb` before running:
  - train fresh (seeds 500–502, same eight settings) to the first epoch with
    at least 25% accuracy, the start of the learning transition;
  - from there, for each grid delay, branch a 256-step all-interface stale
    run and a fresh run on identical batches;
  - a delay passes if the stale run's final full-batch loss is at most 2×
    the fresh run's, with 1.25× and 4× as secondaries;
  - the predicted floor is the largest delay up to which every delay passes.
- **Harness and raw results:** `research/experiments/stress_floor.py` and
  `research/results/stress_floor_v0.json`.

**Result.**

| Threshold | Within 2× | Rank correlation with the measured floor |
|---|---|---|
| 2× (primary) | **0 of 8** | 0.18 |
| 1.25× | 0 of 8 | 0.30 |
| 4× | 2 of 8 | 0.46 |

- **It is consistently pessimistic.** It predicts 0–2 where the measured
  floors are 1–8.
- **Why:** a short stale burst at the transition costs more than 2× in loss
  even at delays training survives. A stale run falls behind at the
  transition, then catches up and fits. The burst measures *slowdown at the
  most sensitive moment*; the floor measures *eventual failure*.

**The two gradient-era predictors bracket the floor.**

| Predictor | Kind | Direction of error | Typical prediction |
|---|---|---|---|
| Descent ratio (F1c) | open loop | too optimistic | 16–64 |
| Stress burst (F1d) | short closed loop | too pessimistic | 0–2 |

The measured floor, 1–8, lies between them. A calibrated threshold could
probably map one onto the floor. That was then tested properly: a threshold
fitted on these 8 settings and frozen, then scored on 8 new learning rates
([`calibrated_floor_v0.md`](calibrated_floor_v0.md)). It passed its frozen
rule only just (3/8 within 2×, Spearman 0.52). A free rule that scales the
nearest known floor by lr_cal / lr did better on every measure (6/8, 0.83).

**Status.** Four predictors failed under rules fixed in advance: linear,
block, descent ratio and stress burst. A fifth, the calibrated burst, held on
paper but loses to learning-rate scaling. In this regime the floor is known
only by running to it, or by interpolating from known floors with
learning-rate scaling.

## The permutation bound applied to compensation

The permuted error removes all directional harm and keeps the error's
magnitude, entries and timing. So the floor and fit rate of permuted
training are a ceiling for any method that corrects the error's direction.

**Methods tested:**
- **DC-ASGD, outer-product form.** It adds λ·g_m(g_o·Δθ_o) to each stage's
  stale gradient, with λ = 1 (not tuned). The usual *diagonal* DC-ASGD has no
  cross-stage terms, and our staleness is entirely cross-stage, so the
  diagonal form cannot correct it at all.
- **Exact first-order Taylor.** It adds H_{m,o}Δθ_o by an exact
  Hessian-vector product. This is the best any first-order Taylor method can
  do. A harness test confirms it removes 84% of the error for a controlled
  movement of 1e-3. At a movement of 1e-2 it removes only about 24%, because
  ReLU kinks dominate there.

Seeds 400–404, adder chain widths 1 and 4, so 10 runs per cell. Each cell
gives fits out of 10 and the mean epochs to fit, counting failures as 121.

| Setting | Method | d=4 | d=8 | d=16 | d=32 |
|---|---|---|---|---|---|
| Adam 1/300 | stale | 9, 25 | 7, 48 | 10, 24 | 9, 75 |
| | DC-ASGD outer | 9, 25 | 9, 26 | 7, 58 | 0 |
| | exact Taylor | 5, 67 | 4, 78 | 0 | 0 |
| | **permuted (bound)** | 9, 28 | 9, 25 | 10, 15 | 9, 32 |
| SGD 0.1 | stale | 8, 42 | 4, 88 | 0 | 0 |
| | DC-ASGD outer | 8, 43 | 5, 75 | 0 | 0 |
| | exact Taylor | 6, 57 | 2, 100 | 0 | 0 |
| | **permuted (bound)** | 9, 30 | 9, 30 | 4, 78 | 2, 100 |

- **Neither first-order compensation closes any of the gap to the bound.**
  - Exact Taylor makes training worse at every delay; under Adam it fails at
    d=16, where plain stale training fits 10/10.
  - DC-ASGD outer helps only at Adam d=8 and is worse at d≥16.
- **Why this is plausible.** The real error lives in sharp directions where
  the parameter trajectory oscillates, and real steps are outside the linear
  regime because of ReLU kinks. A linear correction of an oscillating,
  kinked error overshoots.
- **What the bound adds.** Without it, "DC-ASGD does about as well as nothing"
  reads as "staleness can't be helped". With it: a 2–8× improvement is
  available, and first-order corrections capture none of it.

## Overall

1. The margin theory exists (Giladi et al., 2020), and in this regime it
   fails quantitatively, because training runs at the edge of stability.
   Delay interacting with edge-of-stability self-stabilization is the open
   theoretical question.
2. The eigenbasis mechanism and its fix are published (Basis Rotation, 2026).
   Our curvature and alignment measurements corroborate it from a different
   direction.
3. **The permutation bound is the one new tool.** It quantifies how much
   directional correction could help: 2–8× in floor, and fresh-like speed
   inside the floor. It shows that first-order compensation captures none of
   that here. The natural next test is Basis Rotation against the bound.

Everything is on toy networks and exploratory.

## Reproduce

```bash
python -m unittest research.tests.test_sharpness_floor research.tests.test_block_stability research.tests.test_compensation_bound
python -m research.experiments.sharpness_floor sharpness
python -m research.experiments.sharpness_floor floors
python -m research.experiments.sharpness_floor predict
python -m research.experiments.block_stability
python -m research.experiments.compensation_bound
```
