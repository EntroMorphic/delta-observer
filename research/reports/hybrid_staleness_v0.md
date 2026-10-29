# Muon/Adam hybrids under staleness (exploratory)

**Status: exploratory, not pre-registered.**
- Seeds 400–404, on the adder chain at widths 1 and 4, so 10 runs per cell.
- Staleness applies to every interface, from step 0. Fit means 100% accuracy
  within 120 epochs.
- Harnesses: `research/optimizers/hybrid_muon.py`,
  `research/experiments/hybrid_staleness.py` and
  `research/experiments/impulse_probe.py`, with tests.
- Raw results: `research/results/hybrid_staleness_{calibrate,surface,probe}_v0.json`
  and `research/results/impulse_probe_v0.json`.
- The parents come from
  [`staleness_floor_v0.md`](staleness_floor_v0.md): the same seeds, arms and
  harness.

## Why a hybrid

In the floor surface, Muon (lr 0.01) has the lowest mean time to fit up to
d=8, and Adam (1/300) is best from d=16. The update-space probe showed that
Adam's second-moment denominator shrinks the real, curvature-aligned
staleness error about 4.5× relative to a scrambled twin. So the idea is to
give Muon Adam's damping.

**Prior art.**
- AdaMuon (Si et al., arXiv 2507.11005) applies an element-wise second moment
  *after* orthogonalization.
- NorMuon (arXiv 2510.05491) normalizes per neuron after orthogonalization.
- Neither has been studied for staleness.
- Zmushko et al. (arXiv 2606.30634) find Muon robust to a one-step delay and
  credit its momentum.

**The two hybrids.** Both treat every weight matrix this way and send biases
to Adam:
- **Pre-conditioned Muon:** divide the Nesterov momentum by Adam's √v, then
  orthogonalize.
- **AdaMuon-style:** orthogonalize, then divide by a second moment of the
  orthogonalized update, then match Muon's RMS. This is a simplified AdaMuon.

**Calibration.** Fresh runs on seeds 300–302 with the earlier rule picked lr
0.03 for both, the same as Muon's calibrated rate. lr 0.01 was added as a
secondary to match Muon's best stale setting. That choice was informed by the
parent surface, so the 0.01 rows are post hoc.

## Results

Each cell gives fits out of 10 and the mean epochs to fit, counting failures
as 121. The floor is the largest delay up to which every cell has at least
8/10 fits.

| Setting | Fresh | d=1 | d=2 | d=4 | d=8 | d=16 | d=32 | Floor |
|---|---|---|---|---|---|---|---|---|
| Adam 1/300 | 9, 23 | 9, 23 | 9, 27 | 9, 25 | 7, 48 | 10, **24** | **9, 75** | 4* |
| Muon 0.01 | 10, **7** | 10, 13 | 9, 19 | 9, 20 | 9, 23 | 7, 56 | 0 | 8 |
| Muon 0.03 | 8, 28 | 9, 18 | 8, 29 | 8, 34 | 2, 99 | 0 | 0 | 4 |
| Pre-conditioned Muon 0.01 | 10, 12 | 9, 19 | 9, 20 | 10, **9** | 10, **13** | 8, 46 | 3, 113 | 16 |
| Pre-conditioned Muon 0.03 | 9, 21 | 9, 18 | 10, 8 | 9, 20 | 2, 101 | 0 | 0 | 4 |
| AdaMuon-style 0.01 | 10, 9 | 10, **9** | 10, **11** | 10, 15 | 10, 14 | **10, 30** | 5, 106 | **16** |
| AdaMuon-style 0.03 | 9, 21 | 8, 28 | 10, 6 | 10, 8 | 6, 58 | 0 | 0 | 4 |

\* Adam's dip to 7/10 at d=8 sets its floor at 4. Its real cliff lies
between 32 and 64.

- **At lr 0.01, both hybrids double Muon's floor, from 8 to 16.**
  - AdaMuon-style fits 10/10 in every cell from fresh to d=16.
  - At d=16 it needs a mean 30 epochs, against 24 for Adam and 56 for Muon.
  - It is the fastest or near-fastest setting at every delay up to 16.
- **Adam still owns d=32,** with 9/10 fits against 5/10 for the best hybrid.
- **The hybrid closes most of the crossover but not all of it.** It extends
  Muon-like speed from d≤8 to d≤16 and leaves the longest delays to Adam.
- **At the calibrated lr 0.03, the hybrids gain nothing.** Every Muon-family
  setting has floor 4 and fails from d=16.

## Why lr 0.03 fails: an oscillatory regime

The impulse probe injects a single staleness error once, along a fresh
trajectory after the learning transition. It then follows three branches on
identical batches for 64 steps: no injection, real error, and permuted error.

| Setting | Step-to-step swing of the fresh loss (median) |
|---|---|
| Adam 1/300 | 0.6% |
| Muon 0.01 | 1.4% |
| SGD 0.3 | 1.6% |
| **Muon 0.03** | **7.1%** |
| **Pre-conditioned Muon 0.03** | **15%** |

- **At lr 0.03 the Muon family oscillates from step to step with no
  perturbation at all.** That is an edge-of-stability-like regime.
  - An injected error mostly shifts the oscillation's phase. Muon 0.03's
    real-error gap alternates sign: +5%, −4%, +3%, −2.5%.
  - For pre-conditioned Muon, a scrambled kick *lowers* the loss on every
    measured injection at steps 1 and 2.
- **This qualifies earlier one-step numbers.** Muon's "+14% per step" in
  [`staleness_floor_v0.md`](staleness_floor_v0.md), and the hybrid update
  probe at 0.03 (`hybrid_staleness_probe_v0.json`), both measure one-step
  loss inside an oscillation. They mix phase with harm and should not be read
  as damage.
- **In the non-oscillatory settings,** Adam, SGD 0.3 and Muon 0.01, one real
  error does small immediate damage that fades within about 16 steps. One
  permuted error does almost none. Staleness hurts because the error is
  *sustained*: every step it points the same way.
- **The Muon exception, re-read.** The directional effect and the hybrids'
  gains both appear at lr 0.01 and vanish at 0.03. The exception was a regime
  effect.

## The general lesson

Calibrating on synchronous speed chose lr 0.03 for every Muon-family
optimizer. That rate is in the oscillatory regime, where staleness tolerance
collapses. **Tuning for fresh-training speed selects learning rates that are
fragile under staleness.** This is consistent with the floor scaling with
learning rate × delay. In practice, an asynchronous run should be tuned under
staleness, not before it.

## Caveats

- These are toy networks with 10 runs per cell.
- The 0.01 hybrid rows were chosen with knowledge of the parent surface.
- The simplified AdaMuon is not the published algorithm. It lacks the
  sign-stabilized variant, and its RMS matching is per matrix.
- A confirmatory test would need:
  - fresh seeds;
  - learning rates chosen by a rule fixed in advance, for example the
    fastest rate whose fresh loss swing stays below a set threshold;
  - the primary comparison set as AdaMuon-style against Muon and Adam at d=16.

## Reproduce

```bash
python -m unittest research.tests.test_hybrid_staleness research.tests.test_impulse_probe
python -m research.experiments.hybrid_staleness calibrate
python -m research.experiments.hybrid_staleness surface
python -m research.experiments.hybrid_staleness probe
python -m research.experiments.impulse_probe
```
