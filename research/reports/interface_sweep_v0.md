# Interface design sweep v0 (exploratory)

**Status: exploratory. Not pre-registered.** Seeds 100–104 only, which are
never to be reused for confirmation. Plan: the Lincoln Manifold synthesis in
`journal/possibilities_synth.md` (gitignored). Harness:
`research/experiments/interface_sweep.py`; tests:
`research/tests/test_interface_sweep.py`. Raw results:
`research/results/interface_sweep_v0.json` and
`research/results/interface_sweep_v0_followup.json`.

## Question

The two Second Star Slip tests tried to *schedule* communication: predict which
interface must be kept fresh. Both were killed. This sweep asks a design
question instead: does what crosses a module boundary decide how much
staleness the system can absorb?

## Setup

- **Task and data:** 6-bit addition, all 8,192 cases.
- **Optimizer:** Adam, peak lr 1/300, 5-epoch linear warmup, then constant.
- **Batches:** batch size 128; each seed has a fixed batch order.
- **Fit:** 100% exact accuracy. Training stops at fit or after 200 epochs.
- **Staleness model:** every stage computes its gradient against *all other*
  stages delayed by d optimizer steps, from step 0. This is the
  parameter-view model used in v1 and v2, applied to every boundary at once.

### Arms

| Arm | Structure | Message |
|---|---|---|
| A | Adder chain (v2 widths 8, 32, 8, 16, 32, 8): each stage gets its own bits and emits its own sum bit | Continuous, width 1 |
| B | Adder chain | Continuous, width 4 and width 16 |
| G | Adder chain | Continuous, width 64 (follow-up) |
| C | Adder chain | Width 1, rounded, straight-through |
| D | Adder chain | Width 4, sign, straight-through |
| E | Adder chain | Width 1; each stage trains on *linearly extrapolated* forward and backward messages, from its views at d and 2d |
| F | Monolithic 64-wide MLP cut into 6 stages; only the last stage emits outputs | 64-wide hidden state |
| H | Like F, but each stage emits one output through its own head (follow-up) | 64-wide hidden state |

### Changes from the plan, made before seeing any full-sweep results

- **Delay grid lowered** from 0–1024 to 0, 1, 4, 16, 64. In a one-seed smoke
  run, delay 64 stopped every arm.
- **C and D ran fresh only.** They did not learn fresh in the smoke run, so
  they were reported rather than tuned.
- **Added a 99%-accuracy readout.**

The follow-up arms G and H and the batch-order control were added *after*
seeing the main sweep.

## Results: fits out of 5 seeds, median epochs to fit

| Arm | d=0 | d=1 | d=4 | d=16 | d=64 |
|---|---|---|---|---|---|
| A cont w1 | 4/5, 12 | 5/5, 13 | 5/5, 14 | **5/5, 22** | 0/5 |
| B cont w4 | 5/5, 9 | 5/5, 11 | 5/5, 11 | **5/5, 23** | 1/5, 167 |
| B cont w16 | 3/5, 18 | 4/5, 12 | 4/5, 22 | 3/5, 40 | 0/5 |
| G cont w64 | 4/5, 46 | 3/5, 19 | 2/5, 31 | 0/5 | – |
| E extrapolated w1 | 4/5, 12 | 4/5, 12 | 5/5, 13 | 4/5, 44 | 0/5 |
| C rounded w1 | 0/5 | – | – | – | – |
| D sign w4 | 0/5 | – | – | – | – |
| H pipeline + local heads | 5/5, 11 | 5/5, 21 | 5/5, 82 | 0/5 | – |
| F pipeline | 5/5, 24 | 5/5, 47 | **0/5** | 0/5 | 0/5 |

## Findings

1. **Narrow interfaces tolerate staleness; wide ones do not.** The adder chain
   is the same in every row; only the message width changes. All 5 seeds fit
   at d=16 with widths 1 and 4, 3 seeds with width 16, and none with width 64.
   At d=4 the counts are 5, 5, 4 and 2. That is a factor of at least 4 in
   tolerated delay between width 4 and width 64.
   - The width-64 chain is also harder to train fresh (4/5 seeds, median 46
     epochs), so some of the gap is trainability.
2. **Local structure matters too.** These three arms share one 64-wide
   message:

   | Arm | Local inputs | Local outputs | Largest delay at which all 5 seeds fit |
   |---|---|---|---|
   | Plain pipeline (F) | no | no | d=1 |
   | Pipeline with local heads (H) | no | yes | d=4, at 7× slower |
   | Wide adder chain (G) | yes | yes | no all-5 level (best is 4/5, fresh) |

   The narrow, local chain fits at d=16 at about 2× slower. Both factors
   contribute, and with this many confounds the split between them is not
   settled.
3. **Staleness rescues stuck seeds, and batch-order luck does not explain it.**
   I ran three fresh batch orders per seed.
   - A, seed 100 needs 115–130 epochs under every order, but fits in 14 at
     d=4 and in 24 at d=16.
   - A, seed 104 fails under 2 of 3 orders and fits in 49 at d=16.
   - B width 16, seed 102 needs 109 epochs, 137 epochs, or does not fit
     fresh, but fits in 12 at d=1 and in 14 at d=4.

   These are 4 seed/arm cases out of 10 that stall fresh. There is no
   control for "any perturbation would do" yet.
4. **Stuck runs have a half-formed interface.** I fitted each message to the
   true carry by linear regression and took R². Carries become faithful
   (R² → 0.95–1.0) between epochs 4 and 7, which is the learning transition
   where v2's signal inverted. In both stalled width-1 seeds, one interface
   stays at R² 0.75–0.78 through epoch 10, while the others reach 0.95–1.0.
5. **Message extrapolation hurts.** Arm E is no better than plain staleness at
   d ≤ 4, and worse at d=16: median 44 epochs against 22, with one seed
   failing.
6. **Straight-through discrete messages do not learn at all** in this setup:
   0/5 seeds for both rounding and sign. No tuning was attempted.
7. **Small staleness wins are real but small.** The v2 retest data has 15 of
   750 windows with lower loss under all three batch orders, against about 1
   expected by chance. The loss falls by 0.1–8%, mostly late in training.
   - In v1 at k=100, a second repeat is negative 89% of the time when the
     first is, against a 10% base rate.

## Bets, scored

| Bet | Odds | Result |
|---|---|---|
| Discrete messages have a staleness budget at least 4× the continuous one | 40% | **Lost**: discrete messages do not learn |
| Wider messages *lower* the budget | 55% | **Won for widths 16 and 64**; width 4 matches width 1 |
| Extrapolation beats plain staleness by at least 3× | 30% | **Lost**: it is worse |
| The conventional pipeline has the smallest budget | 60% | **Won**: it fails at d=4 |
| Negative harm is sign-consistent more often than chance | 25% | **Won, but small** |

## What this does and does not show

- The surviving effect is narrow, local interfaces against wide ones, at a
  factor of 4 or more. It was found on a task whose correct interface is
  known to be 1 bit. That is the unfair-race risk named in the plan, so this
  is not yet a principle.
- The rescue effect (finding 3) is the newest surprise. It has two seed/arm
  cases with a strong effect and no perturbation control.

## Next, if pursued

1. **Confirmatory protocol, frozen before running:** seeds 0–9; widths 1, 4,
   16 and 64 on the adder chain, plus F and H; delays 0, 4 and 16.
   - Primary outcome: the number of seeds that fit at d=16, with width 4
     against width 64.
   - Include an unknown-interface task, for example a modular network on a
     composite function, where narrow and wide are both legitimate designs.
2. **Rescue control:** on the seeds that stall fresh, compare staleness at d=4
   with fresh training plus matched isotropic gradient noise and with a
   one-time weight perturbation. This tests whether desynchronisation does
   something noise does not, such as breaking a half-formed contract.

## Reproduce

```bash
python -m unittest research.tests.test_interface_sweep
python -m research.experiments.interface_sweep            # ~30 min, 9 workers
python -m research.experiments.interface_sweep --arms G_cont_w64 H_pipeline_local_w64 --delays 0 1 4 16 \
    --order-control A_cont_w1 B_cont_w16 F_pipeline_w64 --output research/results/interface_sweep_v0_followup.json
```
