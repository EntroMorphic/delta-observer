# Interface confirmation: protocol

Frozen before any confirmatory run. This protocol tests the two candidates
that survived the prior-art check ([`prior_art_check_v0.md`](prior_art_check_v0.md)).
Both were found in exploration
([`interface_sweep_v0.md`](interface_sweep_v0.md),
[`contract_experiments_v0.md`](contract_experiments_v0.md)) on seeds 100–104
and 200–209. Those seeds are not reused here.

Harness: `research/experiments/interface_confirm.py`. Tests:
`research/tests/test_interface_confirm.py`.

## Shared setup

- **Training:** batch 128; linear warmup over 5 epochs, then a constant
  learning rate.
- **Fit:** 100% exact accuracy on all cases. Runs stop at fit or after 200
  epochs.
- **Time to fit:** the epoch of fit. A run that never fits is censored at
  201.
- **Staleness:** every stage computes its gradient against all other stages
  delayed by d optimizer steps, from step 0 (the exploration model).
- **Confirmatory seeds:** 0–9.

## H1: locality and width

**Tasks:**
- *Adder:* the 6-bit ripple-carry chain, all 8,192 cases, hidden widths
  (8, 32, 8, 16, 32, 8).
- *Automaton:* a random 6-state machine over 2-bit symbols, all 4,096
  sequences, with the task seed equal to the run seed. Hidden widths are 32.

**Arms:** all trained with Adam at lr 1/300, at delays 0, 4 and 16.

| Arm | Adder | Automaton |
|---|---|---|
| Chain, narrow-but-sufficient message | width 4 | width 16 |
| Chain, wide message | width 64 | width 64 |
| Pipeline with a local output head per stage, 64 wide | ✓ | ✓ |
| Plain pipeline, 64 wide | ✓ | ✓ |

**Primary hypotheses**, tested at d=16 and paired by seed:

- **H1a, locality.** The wide chain fits sooner than the plain pipeline of the
  same width.
- **H1b, width.** The narrow-but-sufficient chain fits sooner than the wide
  chain.

**Test:** a one-sided Wilcoxon signed-rank test on the paired times to fit,
dropping ties. A hypothesis is **supported on a task** if p < 0.05 and the
median paired log ratio is below 0. It is **confirmed** if supported on both
tasks.

**Secondary analyses**, reported without a verdict:
- the same tests at d=4;
- wide chain against the pipeline with local heads;
- pipeline with local heads against the plain pipeline.

## H2: directional staleness harm

**Task:** the adder. **Arms:** chain widths 1 and 4.

**Conditions**, each paired by (arm, seed), so 20 units per optimizer:
- fresh;
- stale at d=16;
- **permuted** at d=16: fresh gradients plus the true staleness error, with
  each tensor's entries scrambled by a fixed random permutation. This keeps
  the error's norm, its entries and its correlation over time, and destroys
  only its alignment with the network.

**Optimizers:**

| Optimizer | Learning rate | Notes |
|---|---|---|
| Adam | 1/300 | The exploration setting |
| SGD, momentum 0.9 | 0.3 | Selected in calibration, below |
| Muon | 0.03 | Muon for every weight matrix, no weight decay; Adam at 1/300 for biases. Selected in calibration |

**Primary hypothesis H2.** Under a given optimizer, the permuted condition
fits sooner than the stale one. That would mean staleness harm comes from the
error's direction rather than its size or timing.

- Same test and support rule as H1, over the 20 pairs.
- H2 is **general** if supported for at least 2 of the 3 optimizers.
- If it is supported for Adam only, it is reported as Adam-specific.

**Secondary analyses:** stale against fresh, and permuted against fresh, per
optimizer.

## Calibration (done before freezing; seeds 300–302 only)

Fresh runs only, on the H2 arms. The selection rule was fixed before looking:
take the most fits out of 6, and break ties by the lower median fit epoch. An
optimizer that fit fewer than 4 of 6 at every learning rate would have been
dropped from H2.

| Optimizer | Learning rate | Fits out of 6 | Median fit epoch |
|---|---|---|---|
| SGD | 0.03 | 6/6 | 42.5 |
| SGD | 0.1 | 6/6 | 16.5 |
| SGD | **0.3** | **6/6** | **8** |
| SGD | 1.0 | 5/6 | 6 |
| Muon | 0.003 | 6/6 | 13 |
| Muon | 0.01 | 6/6 | 7.5 |
| Muon | **0.03** | **6/6** | **4.5** |
| Muon | 0.1 | 2/6 | 3 |

Raw results: `research/results/interface_confirm_calibration.json`. No stale
or permuted runs were made in calibration.

## Kill criteria

- **H1a not supported on either task:** the locality finding is an
  exploration artifact.
- **H1b not supported on either task:** width does not matter beyond
  capacity.
- **H2 supported for no optimizer:** the directional finding is killed. If it
  is supported for Adam only, then it is an optimizer quirk and the paper's
  angle is the optimizer dependence, not a general property of staleness.

## Run

```bash
python -m unittest research.tests.test_interface_confirm
python -m research.experiments.interface_confirm run   # 420 runs
```
