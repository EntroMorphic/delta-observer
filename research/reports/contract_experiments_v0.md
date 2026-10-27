# Contract experiments v0 (exploratory)

**Status: exploratory, not pre-registered.**
- Seeds 200–209 are new to all prior work. The rescue control reuses the
  stalled v0 cases (seeds 100–104).
- Plan: the Lincoln Manifold synthesis in `journal/paradigm_synth.md`
  (gitignored).
- Harnesses: `research/experiments/contract_experiments.py` and
  `research/experiments/contract_lifecycle.py`, with tests in
  `research/tests/test_contract_*.py`.
- Raw results: `research/results/contract_{rescue,automaton,lifecycle,discretize}_v0.json`.

## The idea under test

The interface sweep ([`interface_sweep_v0.md`](interface_sweep_v0.md)) and a
re-read of the v2 data suggested a lifecycle for each module boundary.

- **The lifecycle:** an interface starts meaningless, is negotiated during the
  learning transition, then settles.
- **The claim:** asynchrony is ruinous during negotiation and nearly free once
  an interface settles.

Training recipe, as in the sweep:
- Adam, lr 1/300, 5-epoch warmup, then constant.
- Batch 128.
- Fit means 100% exact accuracy; at most 200 epochs.

## Scorecard

| # | Claim | Bet | Result |
|---|---|---|---|
| 1 | Staleness tolerance switches on after the transition | 85% | **Held**. After 10 synchronous epochs, 256-step staleness on every interface is free. From step 0, 64 steps stopped every seed. |
| 2 | A settlement signal that needs no knowledge of the true carry rises *before* harm collapses | 45% | **Killed**. It lags by a median 0.75 epochs at width 1 and shows no consistent order at width 4. |
| 3 | Per-interface decoupling beats the best global switch | 25% | **Not run**. Its gate was claim 2. |
| 4 | Staleness rescues stalled runs better than matched noise | 35% | **Held, weakly**. Staleness is fast in 9/9 rescue runs, matched noise in 4/9. |
| 5 | Snapping a settled interface to two levels keeps the fit and makes it staleness-proof | 50% | **Killed**. Snapping freezes wrong contracts, and it lowers tolerance at d=1024. |
| 6 | Staleness lowers the effective dimension of wide interfaces, on a task with no designed latent | 20% | **Killed**. The participation ratio rises slightly at d=4 and d=16. |
| — | Narrow interfaces are better (from the sweep), on an unknown-interface task | — | **Not replicated**. Width has a sweet spot; only *locality* replicated. |

## X5. An interface the student is not told about

**Task.** Each of 6 stages reads a 2-bit symbol. A random 6-state machine
moves, and each stage emits a random 1-bit label of the state. All 4,096
sequences are used, with all hidden widths set to 32. The adder is the
2-state special case.

Fits out of 5 seeds (200–204), median epochs:

| Arm | d=0 | d=4 | d=16 | d=64 |
|---|---|---|---|---|
| Chain, width 1 | 1/5, 57 | 0/5 | 1/5, 130 | 0/5 |
| Chain, width 2 | 5/5, 42 | 4/5, 64 | 3/5, 97 | 0/5 |
| Chain, width 4 | 5/5, 29 | 5/5, 46 | 4/5, 88 | 0/5 |
| Chain, width 16 | 5/5, 20 | 5/5, 31 | **5/5, 86** | 0/5 |
| Chain, width 64 | 5/5, 20 | 5/5, 38 | 3/5, 160 | 0/5 |
| Pipeline + local heads | 5/5, 36 | 3/5, 135 | 0/5 | 0/5 |
| Plain pipeline | 4/5, 67 | 1/5, 114 | **0/5** | 0/5 |

- **Locality replicates.** Every chain wide enough to learn the task fits at
  d=16 on at least 3/5 seeds. Both pipelines get 0/5.
- **"Narrower is better" was an adder artifact.** The adder's true state is
  1 bit. Here width 1 cannot learn the task, and width 16 tolerates the most
  staleness. Excess width (64) costs some tolerance: 3/5 seeds at median 160
  epochs, against 5/5 at 86.
- **No modularity pressure from staleness.** The mean participation ratio of
  each boundary rises slightly with delay at d=4 and d=16:

  | Width | d=0 | d=4 | d=16 |
  |---|---|---|---|
  | 16 | 2.0–2.8 | 1.8–3.6 | 1.7–3.7 |
  | 64 | 1.9–3.3 | 2.3–3.7 | 1.8–3.6 |

  At d=64 it falls, but those runs had collapsed.

## X3. Rescue control

Stalled seed/arm cases from the sweep, under three batch orders. The noise arm
uses fresh gradients plus isotropic noise, matched per step and per tensor to
the norm of the error that staleness would cause.

| Case | Fresh | Stale | Matched noise |
|---|---|---|---|
| Width 1, seed 100, d=4 | 117, 115, 130 | **14, 12, 11** | 58, 13, 12 |
| Width 1, seed 104, d=16 | none, 111, none | **49, 29, 33** | 135, 46, 153 |
| Width 16, seed 102, d=4 | none, 137, 109 | **14, 15, 14** | 31, 41, 16 |
| Width 16, seed 104, d=16 | none ×3 | 130, none, none | none, 18, none |
| Healthy width-1 seeds 101–103, d=4 | 9–14 | 11–16 | 9–55 |

Staleness rescues more reliably than matched noise, but noise often rescues
too. The noise here is independent across steps, while staleness error is
correlated in time. A fairer control would use temporally correlated noise,
so "structured desynchronisation beats noise" is not established.

## X1. Lifecycle along a fresh trajectory

Adder chain, widths 1 and 4, seeds 200–209. Every 16 steps for 16 epochs,
for each interface, I recorded:

- **Harm:** log loss after one epoch with only this interface 64 steps stale,
  against a fresh branch with the same batch order.
- **Coupling:** the mean of |g·Δm| over a probe set, where Δm is the message
  drift over 64 steps and g is the downstream loss gradient. This is the
  first-order coupling term, and it needs no knowledge of the true carry.

Results:

- **The harm lifecycle holds at width 1.** Median harm peaks around epochs 3–7
  at about 0.01–0.05 (log10) and stays about 0.001 afterwards.
- **It does not hold cleanly at width 4.** Harm flares again late on some
  seeds, up to 0.23 at epoch 11 on seed 200.
- **The coupling signal lags.** I define settlement time as the point where a
  quantity falls below 25% of its peak.

  | Width | Harm settles minus coupling settles | Coupling settles first |
  |---|---|---|
  | 1 | median −0.75 epochs, IQR −1 to 0 | 18% of interfaces |
  | 4 | median 0, IQR −0.75 to +4.75 | 49% of interfaces |

- **Interfaces settle at different times.** The median spread within a run is
  3.75 epochs at width 1 and 7.5 at width 4. In principle, that leaves room
  for per-interface control.
- **The coupling signal ranks interfaces at a given moment better than v2's
  signals did.** Mean ρ is 0.43 at width 1 and 0.40 at width 4, against v2's
  0.2–0.3. The design differs (a one-epoch window at delay 64), so this is not
  a like-for-like comparison.

### A reasoning error in the plan

The plan said a lagging signal would make the detector useless. For
*decoupling*, a signal that fires late is safe; it only spends some extra
communication. The real defect is firing *early*, which happens 18–49% of
the time. I kept the plan's gate and did not run X2, but the gate was badly
chosen.

## X4. Decouple, and optionally snap, after settlement

Adder chain, width 1, seeds 200–209. Training is synchronous for 10 epochs.
Then every interface switches to delay d. In the snapped arms, each message is
also snapped to two levels: the cluster centres from a 1-D 2-means on the
message values. That uses no labels, and no gradient crosses the boundary.

| Mode | d=0 | d=256 | d=1024 |
|---|---|---|---|
| Continuous | 7/10, 11 | **8/10, 12** | 6/10, 24.5 |
| Snapped | 4/10, 11 | 4/10, 11 | 2/10, 39 |

- **Once settled, 256-step staleness on every interface is free.** Compare
  from step 0: 64 steps stopped every v0 seed, and in the sweep, d=16 doubled
  the time to fit.
  - The seeds that were not yet settled behave differently. At the switch,
    accuracy was 0.66–0.83, and these four went two ways: two fit later (at
    epochs 30 and 38) and two collapsed.
- **d=1024 from epoch 10 includes a shock.** 1024 steps is 16 epochs, so the
  stale views reach back to initialization for the first 384 steps.
  Continuous still fits on 6/10 seeds.
- **Snapping only preserves contracts that are already correct** (the 4 seeds
  at 100% when snapped). It freezes every other contract, and it lowers
  tolerance to the d=1024 shock.
- **A bug, found and fixed before reporting.** The first version rounded at a
  fixed 0.5, which destroyed every run. A learned carry need not be encoded
  as 0/1. The results above use the label-free two-level snap, and the
  continuous arms reproduced exactly.

## What survives

1. **Locality** decides how much asynchrony a modular chain absorbs: stages
   with their own inputs and outputs. This holds on two tasks. Message width
   matters only through capacity: too narrow cannot learn, and excess width
   costs some tolerance.
2. **Asynchrony tolerance is a phase of training.** Settled contracts tolerate
   at least 4× more delay than unsettled ones. That gap is the difference
   between "16 steps doubles the time" and "256 steps is free".
   - The global form of this, "sync early, async later", is known (post-local
     SGD, DiLoCo-style warm starts).
   - The open question is whether a per-interface switch beats the best
     global one. Interfaces do settle 4–8 epochs apart. No early-firing
     detector exists yet.
3. **Staleness rescues stalled runs** more reliably than white noise of matched
   norm. Temporally correlated noise has not been tested.

## Reproduce

```bash
python -m unittest research.tests.test_contract_experiments research.tests.test_contract_lifecycle
python -m research.experiments.contract_experiments rescue
python -m research.experiments.contract_experiments automaton
python -m research.experiments.contract_experiments discretize
python -m research.experiments.contract_lifecycle
```
