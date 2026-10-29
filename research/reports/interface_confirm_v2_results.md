# Interface confirmation v2: results

Protocol: [`interface_confirm_v2_protocol.md`](interface_confirm_v2_protocol.md),
frozen at `9e5ed19`. The power-check rule was committed earlier, at
`f8e5a38`. Raw results: `research/results/interface_confirm_v2.json`.
Confirmatory seeds 10–19, 240 runs.

## Verdicts, by the pre-registered rules

| Hypothesis | Verdict |
|---|---|
| **H2.** Staleness harm is directional: the permuted error fits sooner than the real error | **General.** Supported for Adam and SGD, not for Muon (2 of 3 optimizers) |
| **H1a.** Locality: the wide chain fits sooner than the same-width plain pipeline at d=4 | **Not confirmed.** Supported on the automaton only |

## H2: directional harm

20 (arm, seed) pairs per optimizer, each at its power-checked delay. Each cell
gives fits out of 20 and the median epochs to fit.

| Optimizer, delay | Fresh | Stale | Permuted | Permuted vs stale |
|---|---|---|---|---|
| Adam, d=16 | 17/20, 11 | 18/20, 23.5 | 19/20, **10** | 17 wins, 3 losses, p = 0.018 |
| SGD, d=2 | 19/20, 8 | 12/20, 12 | 19/20, **8** | 19 wins, 1 loss, p = 7e-5 |
| Muon, d=4 | 20/20, 4 | 18/20, 9 | 15/20, 6 | 14 wins, 5 losses, p = 0.23 |

- **Under Adam and SGD, scrambling the staleness error's direction removes
  essentially all of its cost.** Permuted runs do not differ from fresh runs:
  p = 0.26 for Adam and 0.41 for SGD. Stale runs are clearly slower, at about
  2× for Adam and 1.5× for SGD.
  - The scrambled error keeps the real error's norm, its entries and its
    correlation over time. So, at survivable delays, the harm of staleness
    under these optimizers comes from **the direction of the error relative
    to the network**, not from its size or its timing.
- **Under Muon the permuted error helps less and still hurts.** The direction
  is the same (14 wins to 5), but it is not significant. Permuted runs are
  also slower than fresh ones (16 losses to 2). Muon orthogonalizes each
  update matrix, and that plausibly changes how a misaligned error enters
  the step. This is a hypothesis, not tested here.
- **v1 and v2 together:** under Adam, the directional effect has now held at
  d=16 on seeds 0–9 (20 wins, 0 losses) and on seeds 10–19 (17 wins, 3
  losses).

## H1a: locality at d=4

| Task | Wide chain | Pipeline + local heads | Plain pipeline |
|---|---|---|---|
| Adder | 4/10, 26 | **10/10, 72.5** | 0/10 |
| Automaton | **9/10, 43** | 5/10, 130 | 0/10 |

**Wide chain against plain pipeline:**
- Adder: 4 wins, 0 losses, 6 ties, p = 0.0625. Not supported.
- Automaton: 9 wins, 0 losses, p = 0.002. Supported.

**Pre-registered secondaries, reported without a verdict:**

| Comparison | Adder | Automaton |
|---|---|---|
| Local heads against plain pipeline | 10/0 wins, p = 0.001 | 5/0 wins, p = 0.03 |
| Wide chain against local heads | the chain loses, 3/7 | the chain wins, 9/0 |

**How to read this.** The adder failure is partly power again. In
calibration the wide chain fit 4/5 at d=4, but on the test seeds it fit only
4/10. Five calibration seeds were too few to set the threshold reliably.

The more interesting pattern is in the secondaries:
- **Local outputs** help on both tasks. Adding a local head to each pipeline
  stage beats the plain pipeline 10–0 and 5–0.
- **Local inputs** (the chain) help on the automaton but not on the adder.
  There, a wide chain loses to a pipeline that has local outputs only.

So "locality" splits into two components. Local supervision is the one that
held up on both tasks. That is a reading of the secondary analyses, not a
confirmed claim.

## What is now established, and how strongly

1. **Confirmed (v1):** excess interface width costs staleness tolerance, at
   fixed locality, on two tasks.
2. **Confirmed (v2):** at delays an optimizer can survive, staleness harm under
   Adam and SGD comes from the error's direction. The same error with its
   entries scrambled costs nothing measurable. Muon is the exception: the
   direction is the same, but not significant.
3. **Not confirmed:** locality as a single factor. Local outputs look robust;
   local inputs depend on the task.

Item 2 is the candidate the prior-art check flagged as plausibly open. It now
has a pre-registered confirmation, on toy networks only. Before it could be a
claim beyond toys, it would need:

- a standard architecture, for example a residual MLP or a small transformer
  pipeline;
- the same permuted-error control;
- a mechanism, such as the error's overlap with the top Hessian eigenvectors,
  measured directly.

## Reproduce

```bash
python -m unittest research.tests.test_interface_confirm_v2
python -m research.experiments.interface_confirm_v2 calibrate
python -m research.experiments.interface_confirm_v2 run
```
