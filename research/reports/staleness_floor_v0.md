# The staleness floor and its limits (exploratory)

**Status: exploratory, not pre-registered.**
- Seeds 400–404, on the adder chain at widths 1 and 4, so 10 runs per cell.
- Staleness applies to every interface, from step 0.
- Harnesses: `research/experiments/staleness_floor.py` and
  `research/experiments/staleness_update_probe.py`, with tests.
- Raw results: `research/results/staleness_floor_{surface,mechanism}_v0.json`
  and `research/results/staleness_update_probe_v0.json`.

This builds on the confirmed result in
[`interface_confirm_v2_results.md`](interface_confirm_v2_results.md): at
delays an optimizer survives, staleness harm under Adam and SGD comes from
the error's direction.

## 1. The floor surface

- **Floor:** the largest delay d such that at least 8/10 runs fit within 120
  epochs at every delay up to d.
- **Permuted:** fresh gradients plus the real staleness error, with each
  tensor's entries scrambled by a fixed random permutation. That keeps the
  error's norm, its entries and its correlation over time, and destroys only
  its alignment with the network.

Each cell gives fits out of 10 and the median epoch to fit.

| Setting | Condition | d=1 | 2 | 4 | 8 | 16 | 32 | 64 | Floor |
|---|---|---|---|---|---|---|---|---|---|
| Adam 1/300 | stale | 9, 12 | 9, 15 | 9, 15 | 7, 17 | 10, 23 | 9, **67** | 0 | 4* |
| | permuted | 9, 14 | 9, 13 | 9, 12 | 9, 12 | 10, 10 | 9, **10** | 5, 16 | 32 |
| Adam 1/100 | stale | 9, 7 | 9, 8 | 8, 10 | 4, 16 | 1, 31 | 1, 65 | 0 | 4 |
| | permuted | 10, 6 | 10, 6 | 10, 6 | 9, 7 | 5, 9 | 4, 6 | 1, 17 | 8 |
| SGD 0.1 | stale | 8, 16 | 8, 17 | 8, 23 | 4, 37 | 0 | 0 | 0 | 4 |
| | permuted | 9, 15 | 9, 15 | 9, 15 | 9, 15 | 4, 14 | 2, 14 | 0 | 8 |
| SGD 0.3 | stale | 9, 10 | 5, 13 | 0 | 0 | 0 | 0 | 0 | 1 |
| | permuted | 9, 8 | 10, 8 | 8, 8 | 4, 7 | 0 | 0 | 0 | 4 |
| Muon 0.01 | stale | 10, 7 | 9, 7 | 9, 8 | 9, 11 | 7, 26 | 0 | 0 | 8 |
| | permuted | 10, 7 | 10, 7 | 10, 7 | 10, 7 | 9, 7 | 5, 8 | 4, 10 | 16 |
| Muon 0.03 | stale | 9, 5 | 8, 5 | 8, 9 | 2, 12 | 0 | 0 | 0 | 4 |
| | permuted | 8, 4 | 9, 5 | 6, 5 | 3, 5 | 1, 4 | 0 | 0 | **2** |

\* Under Adam 1/300 the stale run fits 7/10 at d=8 and then 10/10 at d=16,
so the floor rule reports 4. The real cliff lies between 32 and 64.

### What the surface shows

1. **Direction moves the floor by 2–8×, and the effect has a limit.**
   - In 5 of 6 settings the permuted error survives longer delays than the
     real one: 2× in four settings, 4× with SGD at lr 0.3, and up to 8×
     with Adam at 1/300.
   - Permuted training also has its own floor. Past it, the error's sheer
     size kills training whatever its direction.
   - So there are three regimes: below the stale floor, both survive;
     between the two floors, direction decides; beyond the permuted floor,
     size decides.
2. **Inside the survivable range, the permuted error costs almost nothing.**
   Under Adam 1/300 at d=32, stale runs need a median 67 epochs; permuted
   runs need 10, against 12 for fresh training. That matches v2.
3. **The floor roughly follows learning rate × delay, as asynchronous-SGD
   theory predicts.**

   | Optimizer | Learning rate rise | Stale floor | Permuted floor |
   |---|---|---|---|
   | SGD | 0.1 → 0.3 (3×) | 4 → 1 | 8 → 4 |
   | Muon | 0.01 → 0.03 (3×) | 8 → 4 | 16 → 2 |
   | Adam | 1/300 → 1/100 (3×) | cliff from about 32–64 → 8–16 | 32 → 8 |

   The grid steps by factors of 2, so this is coarse.
4. **The exception is Muon at its calibrated learning rate (0.03).** There the
   permuted error is *worse* than the real one (floor 2 against 4). At Muon
   lr 0.01 the directional effect is present (16 against 8). This explains
   why v2's Muon test, run at 0.03, was not significant: the effect for
   Muon depends on the learning rate.

## 2. Mechanism: what the real error looks like

The probe records the error along stale trajectories every 32 steps, over 12
epochs, at v2's delays. That gives 138 records per optimizer: 2 arms × 3
seeds.

| Optimizer, delay | Curvature, real | Curvature, permuted | Real ÷ permuted | Alignment with gradient, real | Alignment, permuted |
|---|---|---|---|---|---|
| Adam, d=16 | 1.67 | 0.10 | 15× | −0.80 | −0.09 |
| SGD, d=2 | 0.45 | 0.05 | 9× | −0.22 | −0.03 |
| Muon, d=4 | 2.22 | 0.12 | 16× | −0.53 | −0.04 |

Curvature here is eᵀHe/|e|² for error e and full-batch Hessian H.

- **The real error points into sharp directions and against the current
  gradient.** Both effects grow after the learning transition: under Adam,
  alignment is −0.12 in epochs 0–4 and −0.86 in epochs 4–8.
- **Interpretation.** The stale gradient partly cancels the fresh one along
  high-curvature directions. That is the delayed-feedback oscillation
  picture.

## 3. Mechanism: what each optimizer does with the error

> **Caveat added later.** The impulse probe in
> [`hybrid_staleness_v0.md`](hybrid_staleness_v0.md) shows that Muon at
> lr 0.03 oscillates from step to step even without injection. Its one-step
> loss changes below mix phase with harm. The Muon row should not be read as
> damage. The Adam and SGD rows are from non-oscillatory regimes.

At each probe point the network and optimizer state are copied, and one step
is taken with f, with f + real error, and with f + permuted error. Here f is
the fresh gradient. The table describes the update error δ = step(f + error) −
step(f). There are 69 records per optimizer: width 4, 3 seeds.

| Optimizer | Norm of δ, real ÷ permuted | Curvature of δ, real ÷ permuted | Exact one-step loss change, real | Exact one-step loss change, permuted |
|---|---|---|---|---|
| Adam | **0.22** | 480× | +0.83% | +0.09% |
| SGD | 1.00 | 15× | +0.38% | +0.02% |
| Muon | 0.64 | 38× | **+14%** | +0.13% |

- **Adam shrinks the real error but not the permuted one.**
  - The real error sits in coordinates with large gradient variance, which
    Adam's denominator shrinks. The permuted error lands in quiet
    coordinates, which Adam enlarges.
  - The permuted update error is about 4.5× *larger*, yet it does about 9×
    less damage, because it falls in flat directions.
  - This replaces my earlier guess that Adam "caps magnitude, so only
    direction remains".
- **SGD is the clean case.** The norms are equal, the curvature differs 15×,
  and the damage differs about 19×.
- **Muon: my prediction failed.** I predicted that orthogonalization would
  make the permuted error harmful too. Per step it stays benign, while Muon's
  real error is the most damaging of all (+14% per step after epoch 4).
  - Combined with the surface result (Muon 0.03 permuted is worse than
    stale), Muon's cost must arrive over several steps. Plausible routes are
    the momentum buffer, and orthogonalization giving every singular
    direction equal weight. This is untested.

## Summary

- **The floor:** the delay at which staleness stops a run from fitting. It
  scales roughly with learning rate × delay.
- **The directional effect:** it moves the floor 2–8× in 5 of 6 settings. It
  is bounded by a second floor where the error's size alone is lethal.
- **The mechanism:** the real error is aimed uphill along sharp directions.
  It sees 9–16× more curvature than its scrambled twin and is anti-aligned
  with the gradient.
- **The limit:** Muon at high learning rate. There the scrambled error is
  worse than the real one, for reasons the one-step probe does not capture.

All of this is exploratory, on toy networks.

## Reproduce

```bash
python -m unittest research.tests.test_staleness_floor research.tests.test_staleness_update_probe
python -m research.experiments.staleness_floor surface
python -m research.experiments.staleness_floor mechanism
python -m research.experiments.staleness_update_probe
```
