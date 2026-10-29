# Basis Rotation against the permutation bound (exploratory)

**Status: exploratory.**
- **Reimplementation:** `research/optimizers/basis_rotation.py`, of Jung, Shin
  and Lee, ICML 2026, arXiv 2602.03515.
- **Harness:** `research/experiments/basis_rotation_bound.py`.
- **Tests:** `research/tests/test_basis_rotation.py`. With the basis held at
  identity, Basis Rotation reproduces Adam exactly, and the bases stay
  orthonormal.
- **Raw results:** `research/results/basis_rotation_bound_v0.json`. Adam's
  rows come from `staleness_floor_surface_v0.json`: the same seeds (400–404),
  arms (adder chain, widths 1 and 4) and harness. That gives 10 runs per cell.

## Method as implemented

Per weight matrix:
- **Basis:** Kronecker-factored second moments, L = EMA(G Gᵀ) and R =
  EMA(Gᵀ G) with β₂ = 0.999. They are refreshed every 10 steps, and U and V
  are updated by one power step and a QR decomposition. This is the
  bilateral, second-moment default.
- **Optimizer:** Adam with β₁ = 0.9 and β₂ = 0.999 runs in the rotated basis,
  and the update is rotated back.
- **Biases:** Adam.

**Choices the paper does not state:**
- U and V start from an exact eigendecomposition, as in SOAP.
- The rotated second moment is not reset when the basis changes.

**Learning rates:** 1/300 (primary) and 1/100, the same as Adam's surface.

## Results at lr 1/300

Median epochs to fit, among runs that fit, with the number of fits out of 10
in brackets:

| Optimizer | Fresh | d=4 stale / permuted | d=8 | d=16 | d=32 | d=64 fits, stale / permuted |
|---|---|---|---|---|---|---|
| Adam | 12 (9) | 15 / 12 | 17 (7) / 12 (9) | 23 / 10 | 67 (9) / 10 (9) | 0 / 5 |
| Basis Rotation | **6 (10)** | 8 / 7 | 10 (9) / 7 (10) | 17 / 9 | 49 (9) / 8 (5) | 2 / 5 |

**Directional penalty within each optimizer** (median stale ÷ median
permuted):

| Optimizer | d=8 | d=16 | d=32 |
|---|---|---|---|
| Adam | 1.4 | 2.3 | 6.7 |
| Basis Rotation | 1.4 | 1.9 | 6.1 |

**Paired test within Basis Rotation** (permuted faster than stale):

| Delay | Permuted faster | Permuted slower | p (one-sided) |
|---|---|---|---|
| 4 | 8 | 2 | 0.014 |
| 8 | 8 | 0 | 0.004 |
| 16 | 8 | 2 | 0.053 |
| 32 | 5 | 4 | 0.82 |

At d=32 only 5/10 permuted runs fit, against 9/10 stale.

At lr 1/100 the pattern is similar. Basis Rotation is better than Adam at d=8
(8/10 fits against 4/10). Both fail at d≥16 under stale training. The
permuted condition fits fastest at d=4–8 (5–6 epochs).

## Findings

1. **Basis Rotation beats Adam under staleness in absolute terms.**
   - It fits fresh about 2× faster, at 6 epochs against 12.
   - It takes 25–40% fewer epochs at every delay: 17 against 23 at d=16,
     and 49 against 67 at d=32.
   - It fits more often at d=8 and d=64.

   That is consistent with the paper's direction.
2. **Against the permutation bound, it does not remove the directional
   harm.**
   - Its stale/permuted penalty matches Adam's at every delay: 1.4 against
     1.4, 1.9 against 2.3, and 6.1 against 6.7.
   - Within Basis Rotation, the permuted error still beats the real one at
     d=4–16 (8 of 10 pairs).
   - So its gain here comes from being a faster optimizer, not from
     neutralizing the misaligned part of the staleness error.
3. **Its own bound weakens at long delay.** At d=32, Basis Rotation with the
   permuted error fits only 5/10, against 9/10 for Adam with the permuted
   error. Adam in a rotated basis seems to amplify scrambled error, much as
   the AdaMuon-style hybrid did.

## Why this may not test Basis Rotation's design regime

Basis Rotation rotates each weight matrix's own basis, and its analysis
assumes a block-diagonal Hessian, with delay on a stage's *own* parameters.
That is the pipeline case, where stage k's gradient is K−k steps old.

Our staleness model is different. Each stage sees its own parameters live and
the other stages stale, so the whole error is cross-stage:
H_{m,o}Δθ_o. A per-matrix rotation cannot target cross-block coupling.

So this result says Basis Rotation helps by speed but not by direction **in a
cross-stage staleness model**. It does not test the within-stage delay it was
designed for. A fair test would add a pipeline-style staleness model to the
harness: each stage's gradient computed at its own older weights, with
weight stashing. Then the same permutation bound could be applied there.

## Reproduce

```bash
python -m unittest research.tests.test_basis_rotation
python -m research.experiments.basis_rotation_bound
```
