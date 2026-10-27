# Contract experiments v0 (exploratory)

**Status: exploratory, not pre-registered.**
- Seeds 200–209 are new to all prior work. The rescue control reuses the
  stalled v0 cases (seeds 100–104).
- Plan: the Lincoln Manifold synthesis in `journal/paradigm_synth.md`
  (gitignored).
- Harnesses: `research/experiments/contract_experiments.py` and
  `research/experiments/contract_lifecycle.py`, with tests in
  `research/tests/test_contract_*.py`.
- X2 harness: `research/experiments/contract_switch.py`.
- Raw results: `research/results/contract_{rescue,automaton,lifecycle,discretize,switch}_v0.json`
  and `research/results/contract_rescue_v0_permuted.json`.

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
| 3 | Per-interface decoupling beats the best global switch | 25% | **Killed** (X2, run in a second round). Triggers fire too early on some interfaces and lose fit rate. |
| 4 | Staleness rescues stalled runs because it desynchronises *structure* | 35% | **Killed** by the permutation control. The same error with its entries scrambled rescues *better*. |
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

In the first round, staleness rescued more reliably than white noise of matched
norm. White noise is independent across steps, though, while staleness error
is correlated in time. So I added a sharper control in a second round: the
true staleness error with each tensor's entries scrambled by a fixed random
permutation. That keeps its norm, its entries and its correlation over time,
and destroys only its alignment with the network.

| Case | Stale | White noise | Permuted stale error |
|---|---|---|---|
| Width 1, seed 100, d=4 | 14, 12, 11 | 58, 13, 12 | **7, 7, 7** |
| Width 1, seed 104, d=16 | 49, 29, 33 | 135, 46, 153 | **13, 10, 21** |
| Width 16, seed 102, d=4 | 14, 15, 14 | 31, 41, 16 | 9, 25, 17 |
| Width 16, seed 104, d=16 | 130, none, none | none, 18, none | none, 15, none |
| Healthy width-1 seeds 101–103, d=4 | 11–16 | 9–55 | **8–13** |

- **The permuted error rescues better than staleness itself.** It also
  speeds up healthy seeds, fitting in 7–8 epochs against 9–14 fresh.
- **Alignment with the network costs something.** The helpful ingredient is a
  perturbation of the right size that varies slowly over time.
- **This looks like an optimizer effect, not an interface effect.** It is
  probably about how Adam responds to slowly varying gradient noise.

This is worth its own look, with prior art first: anticorrelated noise
injection (Orvieto et al., 2022), and work on how noise correlated in time
interacts with Adam's moment estimates.

## Held-out test: does the perturbation help learning, or only fitting?

Each seed (200–209) trains on a random 25% of the cases and tests exact-match
accuracy on the other 75%. Training stops at 100% train accuracy or after 200
epochs.

Conditions:
- **Fresh.**
- **Permuted staleness error (d=4).**
- **White noise** matched to the staleness error (d=4).
- **Synthetic correlated noise:** AR(1) with a correlation time of 4 steps, at
  a per-tensor norm of 0.3× the gradient norm. There is no staleness in this
  arm. The 0.3 comes from a measurement: after warmup, the staleness error is
  0.3–0.5× the gradient per tensor.

| Task / arm | Condition | Train fits | Median fit epoch | Median test accuracy | Test wins / losses against fresh |
|---|---|---|---|---|---|
| Adder, chain w1 | Fresh | 8/10 | 34.5 | 1.000 | – |
| | Permuted | **10/10** | 36 | 1.000 | 2 / 0 |
| | White | 9/10 | 34 | 1.000 | 1 / 0 |
| | Correlated | 9/10 | 37 | 1.000 | 1 / 1 |
| Adder, chain w4 | Fresh | 9/10 | 40 | 1.000 | – |
| | Correlated | **10/10** | 36 | 1.000 | 1 / 0 |
| Automaton, chain w16 | Fresh | 9/10 | 88 | 0.993 | – |
| | Permuted | 10/10 | 97 | 0.994 | 2 / 1 |
| | White | 8/10 | 127 | 0.992 | 1 / 4 |
| | Correlated | 10/10 | 82 | 0.996 | 2 / 0 |
| Adder, plain pipeline | Fresh | 10/10 | 81.5 | **0.991** | – |
| | Permuted | 10/10 | 71 | 0.986 | 1 / 6 |
| | White | 7/10 | 162 | 0.986 | 1 / 6 |
| | Correlated | 10/10 | 105 | 0.971 | **0 / 9** |

"Wins" and "losses" count seeds whose test accuracy differs from fresh by more
than 0.005. The omitted w4 rows (permuted and white) match fresh to within one
seed.

- **The modular chains generalize almost perfectly under every condition.**
  Each stage sees only its local inputs, so the architecture does the
  generalizing. The perturbations make the train fit slightly more reliable
  (10/10 against 8–9/10) and do nothing measurable to test accuracy.
- **The full-data speed-up does not replicate.** On a 25% split, median fit
  times are about the same with and without the perturbations.
- **The perturbations hurt generalization where it is non-trivial.** In the
  monolithic pipeline, all three make test accuracy worse. Synthetic
  correlated noise loses on 9 of 9 seeds that differ, by a median of 2.1
  points.
  - The permuted error fits faster (71 against 81.5 epochs) but generalizes
    worse.

**Verdict:** the perturbation helps *fitting*, a little and inconsistently. It
does not help *learning*, and where learning is hard it hurts. What
generalizes here is the architecture: stages with local inputs and outputs
that match the task's structure.

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
the time. I kept the plan's gate and did not run X2 in the first round. The gate was
badly chosen, so X2 was run in a second round, below.

## X2. Cut the wire, per interface against global

Adder chain, widths 1 and 4, seeds 200–209.
- **Mechanics:** training starts synchronous. A decoupled interface goes to
  delay 256 and stays there. Stages see each other live within a run of
  coupled interfaces, and 256 steps old across any decoupled one.
- **Policies:**
  - Always synchronous.
  - Global switch after E epochs, for E in 2, 4, 6, 8, 10, 12 and 16.
  - Per-interface trigger: decouple interface *i* once its coupling signal
    has been below *f* × its running peak for *k* consecutive checks (every 16
    steps), with *f* ∈ {0.1, 0.25} and *k* ∈ {2, 4, 8}.
- **Selection rule, fixed before looking:** on tune seeds 200–202, keep the
  policies whose fit count is at least the synchronous one and whose median
  time to fit is within 1.2× of it. Among those, choose the lowest
  synchronous cost, then report on seeds 203–209.

| Width | Policy | Fits (10 seeds) | Median fit | Decoupling epochs (median, earliest) |
|---|---|---|---|---|
| 1 | Synchronous | 7/10 | 11 | – |
| 1 | Global at 10 (selected) | **8/10** | 11 | 10, 10 |
| 1 | Best trigger (*f* 0.1, *k* 8) | 6/10 | 10.5 | 9.75, **4.75** |
| 4 | Synchronous | 9/10 | 12 | – |
| 4 | Global at 8 | 7/10 | 11 | 8, 8 |
| 4 | Best trigger (*f* 0.1, *k* 8) | 5/10 | 11 | 9.25, **4.5** |

- **No trigger passed the selection rule** on either width.
- **No global switch passed it at width 4**, where synchronous training fit
  3/3 tune seeds.
- **At width 1 the selected global switch generalised.** On test seeds it fit
  5/7, against 4/7 for synchronous training.
- **Why the triggers lose:** each has a median decoupling time close to the
  right epoch, but fires as early as epochs 3.5–4.75 on some interfaces. That
  is during negotiation, and each such early cut can cost a seed. Per-interface
  timing does not pay for that risk here.
- **A caveat on the toy.** Contracts settle only a few epochs before the
  run fits, so even a perfect switch saves little communication. The idea
  matters only where most training happens after settlement, as in long
  training runs. That regime is the territory of global methods like
  DiLoCo, which this result does not beat.

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
2. **Asynchrony tolerance is a phase of training** (the global form). Settled contracts tolerate
   at least 4× more delay than unsettled ones. That gap is the difference
   between "16 steps doubles the time" and "256 steps is free".
   - The global form of this, "sync early, async later", is known (post-local
     SGD, DiLoCo-style warm starts).
   - A per-interface switch did *not* beat the best global one (X2). This
     was the part that would have been new.
3. **Slowly varying, misaligned gradient perturbations rescue stalled Adam
   runs** on full-data fitting (the permutation control). On a held-out split
   they make fitting slightly more reliable, but they do not improve test
   accuracy, and in the monolithic network they reduce it. This is a fitting
   trick, not a learning one.

## Reproduce

```bash
python -m unittest research.tests.test_contract_experiments research.tests.test_contract_lifecycle research.tests.test_contract_switch
python -m research.experiments.contract_experiments rescue
python -m research.experiments.contract_experiments automaton
python -m research.experiments.contract_experiments discretize
python -m research.experiments.contract_experiments generalize
python -m research.experiments.contract_lifecycle
python -m research.experiments.contract_switch
python -m research.experiments.contract_experiments rescue --kinds permuted \
    --output research/results/contract_rescue_v0_permuted.json
```
