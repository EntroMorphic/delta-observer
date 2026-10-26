# Observer.D Initial AdamW/Muon Mechanism Test

**Experiment:** DOR-001 Phase 4 initial screen  
**Task:** Four-bit addition, 384/64/64 train/validation/held-out split  
**Seeds:** 0--4  
**Status:** Preliminary; optimizer learning rates are not tuned

## Question

Does aligning a detached student's hidden representation to a label-free
summary of two source-learning trajectories improve exact behavioral accuracy?
Does the result differ between AdamW and a Muon hybrid?

The trajectory-distillation method is designated **Observer.D**. Its signature
encoder uses a native C/ARM NEON inference path in the results below, with a
PyTorch reference pass retained for numerical-parity testing.

## Result

Mean final exact held-out accuracy from the initial fixed-learning-rate screen:

| Student objective | AdamW | Muon hybrid |
|---|---:|---:|
| Direct task supervision | 0.853 | 0.969 |
| Task + Observer.D guidance | 0.869 | 0.978 |
| Output distillation | 0.809 | 0.997 |
| Output + Observer.D guidance | 0.850 | 0.988 |

Paired mean observer effects:

| Comparison | AdamW | Muon hybrid |
|---|---:|---:|
| Task + Observer.D minus task | +0.016 | +0.009 |
| Observer.D distillation minus distillation | +0.041 | -0.009 |

## Interpretation

This run does **not** establish a trajectory advantage. Observer guidance
stabilized one poor AdamW seed, but reduced several stronger AdamW runs. With
Muon, the unguided distillation baseline was already nearly saturated and the
observer term slightly reduced mean accuracy. Five seeds and a 64-example test
set are insufficient to distinguish these small effects reliably.

At the selected learning rates, Muon was substantially more stable than AdamW.
That is a hyperparameter-dependent observation, not evidence that Muon is
intrinsically superior. AdamW's seed-3 result was an outlier, and both
optimizers require validation-only learning-rate and weight-decay sweeps before
an optimizer conclusion.

Mean CPU wall time per seed was approximately 0.76 seconds for both source
models and 3.46 seconds for the observer, versus 0.21--0.34 seconds for one
student arm. Muon student training was roughly 60% slower than AdamW in this
small implementation. Consequently, no end-to-end efficiency advantage exists
in this initial test; observer cost would need to be amortized across many
students or offset by a much larger reduction in student training.

## C/NEON backend

Observer.D's trained signature encoder was executed over 4,224 trajectory rows
by the native C backend in every final seed. The compiled ARM64 library reports
`c-neon`, and disassembly confirms vector `fmla.4s` instructions. Numerical
agreement with the equivalent PyTorch encoder was strong:

- Maximum absolute error across seeds: **4.3 × 10⁻⁶**
- Mean native C/NEON encoding time: **32.2 ms**
- Mean PyTorch encoding time: **1.12 ms**

The initial single-threaded NEON implementation is therefore about 29 times
slower than PyTorch's optimized batched kernels for this matrix shape. Native C
correctness is established, but native speed superiority is falsified for this
implementation. Observer training remains in PyTorch and dominates either
encoding path.

## Falsification status

- **Detached behavioral transfer:** Pass for output distillation; students run
  without sources or observer at evaluation.
- **Label-free observer:** Pass narrowly; the observer reconstructs source
  states/deltas and never receives ground truth. Its sources were label-trained.
- **Unique trajectory value:** Not demonstrated.
- **Optimizer superiority:** Not tested adequately; tuning is outstanding.
- **Generalization:** Not established. Random held-out four-bit combinations
  test interpolation, not variable-width algorithmic transfer.

The machine-readable result, including every seed and validation learning
curve, is in `research/results/optimizer_comparison.json`.
