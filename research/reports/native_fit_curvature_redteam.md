# Red-team, arithmetic fit ladder, and non-quantized curvature diagnostic

## Demonstrated red-team fixes

- Scoring formerly constructed a dictionary before checking row count, allowing
  an appended duplicate ID to be silently collapsed. Require exactly 100 rows
  and 100 unique registered task IDs; regression test added. Existing completed
  evaluations contain the expected 100 rows, so their scores are unchanged.
- Trainer now verifies both input and supervised token totals against manifest.
- Resume now verifies every historical row index and step against the fixed
  permutation, including later epochs. Previously it trusted history implicitly.
- Added explicit epoch/rank budgets, per-pass immutable checkpoints, and recorded
  order policy; repeated exposure never silently becomes unique-data coverage.
- Remaining limitations: one seed, exposed diagnostic tasks, FP16 generation
  versus FP32 training, no isolated FFN-target control, and no broad retention
  evaluation. This is not certification that the full codebase is pristine.

## Lincoln Manifold loop-back

Applied RAW → NODES → REFLECT → SYNTHESIZE using anjaustin/lmm commit
`55719c0a2308d1a3f29ca656ed7b6287be33bca2`.
Artifacts: `research/lfm/manifold/native_fit_{raw,nodes,reflect,synth}.md`.

The main challenged assumption was that one pass establishes a model floor.
The fixed cut held arithmetic data, within-task order, targets, optimizer,
alpha/rank ratio, seed, and four-pass budget constant while varying rank.

## Exact compound training fit

| Passes | Rank 8, alpha 16 | Rank 32, alpha 64 |
|---|---:|---:|
| 1 | 77/128 | 58/128 |
| 2 | 99/128 | 70/128 |
| 3 | 118/128 | 115/128 |
| 4 | **125/128** | 116/128 |

Both arms see the same 512 examples per pass; four passes total 2,048 updates,
80,896 input-token exposures and 15,360 supervised-token exposures. These are
repetitions, not new examples. Rank changes trainable parameter count and
initialization shape; one seed cannot establish universal rank superiority.

Rank 8 trains 835,584 parameters and rank 32 trains 3,342,336. Measured loop
times are 269.571578 and 369.337654 seconds, including checkpoint writes and
excluding loading/validation. Each run's four epoch exposure sequences match
exactly. Rank 8's first-pass adapter replicates the earlier one-pass specialist
bit-exactly (maximum tensor difference 0.0), validating continuity of this
training-duration comparison. Frozen base hashes match before/after both runs.

The observed one-pass underfit is largely reduced by additional exposure at
rank 8. Larger rank does not help in this measured schedule. Neither reaches
perfect local fit, and no irreducible capacity floor is established.

## Fixed endpoint transfer

On the same exposed v2 100-task diagnostic test:

| Exact success | Rank 8, four passes | Rank 32, four passes |
|---|---:|---:|
| Arithmetic | 8/20 | 9/20 |
| JSON records | 0/20 | 18/20 |
| Sorting | 0/20 | 0/20 |
| Ordering | 0/20 | 1/20 |
| Spanish number words | 0/20 | 0/20 |
| Total | 8/100 | 28/100 |

Task-only training is not a general model replacement. Near-perfect arithmetic
training fit does not transfer to unseen wording. The next causal question is
supervision/representation transfer, not simply more epochs until the training
score reaches 100%. No intermediate test checkpoint selection occurred.

## Hessian-vector diagnostic without quantization

The user's curvature idea was tested narrowly on rank-8/four-pass checkpoint:
six fixed first compound-training rows; native FP32 weights, eager attention;
exact autograd Hessian directional derivatives of CE with respect to LoRA B
tensors with A fixed. Three unit directions are current B for all adapters,
attention adapters, and conv adapters. This is neither a full Hessian nor an
effective-native-weight quantization-sensitivity map.

Along the combined B direction, measured directional curvature ranges from
0.000263982 to 0.027182553 on the six losses. Directional gradient has both
signs across examples. Symmetric ±0.01 perturbations provide separate measured
loss changes; all parameters were restored bit-exactly.

Finite-difference second derivatives disagree noticeably at the smallest losses,
where FP32 subtraction cancellation and finite-step effects can matter. Do not
use these values as a validated quadratic predictor or a full-model sensitivity
ranking. The quadratic unit test checks the mathematical implementation, not
the quality of a model-wide curvature approximation.

Curvature depends on loss dataset, direction, parameterization, and operating
point. LoRA-factor curvature can change under equivalent factor rescaling and
cannot be interpreted directly as native-weight rounding tolerance. A true
quantization-sensitivity study would need fixed hypothetical perturbation
directions in effective weight space, held-out losses, and measured actual
perturbation effects—without deploying quantized weights.

No inference speedup, greater intelligence, safe pruning decision, or capability
gain follows from constructing the map alone. Potential applications are safer
update allocation or later structural optimization, each requiring a controlled
capability/retention and runtime comparison.

## Artifacts

- `research/results/native_fit_ladder_rank{8,32}_step{512,1024,1536,2048}.json`
- `research/results/native_fit_ladder_rank{8,32}_eval/baseline.json`
- `research/results/native_curvature_diagnostic_v1/curvature.json`
- `research/results/native_curvature_diagnostic_v1/provenance_audit.json`
- `research/results/native_fit_ladder_summary.json`

100/100 remains unmet. Do not claim a pristine system or silently replace native
LLM reasoning with an answer-key router or calculator to satisfy that target.
