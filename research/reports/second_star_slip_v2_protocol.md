# The Second Star Slip: v2 protocol

Frozen before any confirmatory run. v1
([`second_star_slip_results.md`](second_star_slip_results.md)) was killed. Its
failure anatomy showed four defects in the experiment:

- a harm target confounded by training progress;
- a target that did not predict long-run cost;
- an interface ranking too noisy to measure at short delays;
- a scalar signal that dropped the movement half of the mechanism.

v2 fixes each and asks one question: **at a given moment, does the cross-module
gradient error rank which interface is costliest to leave stale?**

## Testbed

A 6-bit ripple-carry chain adder (`ChainAdder`): six full-adder modules with
hidden widths (8, 32, 8, 16, 32, 8), five carry interfaces. Every a + b +
carry-in (8,192 cases); MSE; Adam, peak learning rate 1/300, 5-epoch warmup,
cosine decay over 60 epochs; batch 128 (64 steps per epoch).

## Calibration (done before freezing, seeds 100–102 only, never confirmatory)

- The first learning rate tried (1/137, the paper's) left calibration seed 100
  stuck at 75% accuracy. 1/300 fit all three calibration seeds (100% by epochs
  8–13). A width sweep in the same session was ineffective (a default argument
  was bound at definition time), so the widths above are the only widths tried.
- Target repeatability (below) on seeds 100–102, onsets 2–10, without
  computing any predictor. Within-moment test-retest Spearman: delay 16 /
  window 1 epoch 0.83, 16/3 0.74, 64/1 0.86, 64/3 0.91, 256/1 0.97, 256/3 0.95.
  Median |target| for 3-epoch windows: 0.04–0.19, versus 0.002–0.08 for
  1-epoch windows. Raw: `research/results/second_star_slip_v2_pilot.json`.
- Harness tests showed that one real optimizer step leaves the linear regime:
  the exact cross-module gradient error is up to ~26× the Hessian prediction,
  which the test attributes to ReLU kinks. The primary signal is therefore
  the secant (exact) form below; the Hessian form is secondary.

## Staleness

Interface `j` stale with delay `d` steps: modules on each side compute their
gradients with the other side's parameters from `d` steps earlier (forward
carries and backward gradients both delayed). Other interfaces stay fresh.

## Units and target

Seeds 0–9; onset epochs 2, 4, 6, 8, 10 (the learning phase); delays 16, 64 and
256 steps; window 3 epochs (192 steps).

From the fresh state at the onset, for each interface and each of 3 batch
orders shared across interfaces: run the window fresh and run it with
interface `j` stale. Target for one order:
`log10 L_stale − log10 L_fresh`, full-batch loss at the end of the window.
Unit target: the mean over the 3 orders. All comparisons are **within a
moment** (same seed and onset, five interfaces), so the shared training
progress cancels.

## Predictors (information available at the onset)

| Predictor | Definition |
|---|---|
| **secant_error (primary)** | ‖g_A(θ_A, θ_B(t−d)) − g_A(θ)‖ and ‖g_B(θ_A(t−d), θ_B) − g_B(θ)‖ combined (root sum of squares), full-batch. A, B = modules on either side of `j` |
| secant_error_adam | Same, each component scaled by Adam's current 1 / (√v̂ + ε) |
| hessian_error | ‖H_AB Δθ_B‖ and ‖H_BA Δθ_A‖ combined, Δθ = θ(t) − θ(t−d); exact HVPs |
| hessian_error_adam | Same, Adam-scaled |
| message_drift | RMS change of carry `j` between θ(t−d) and θ(t), full batch |
| message_size | RMS of ∂L/∂c_j |
| movement_near | ‖Δθ‖ over modules `j` and `j+1` |
| static_prior | Each interface's mean target on the other nine seeds (leave-one-seed-out) |

## Statistic

For each delay: Spearman correlation between predictor and unit target across
the 5 interfaces within each (seed, onset), averaged over the 50 groups.
Uncertainty: 2,000 bootstrap resamples of seeds (all of a seed's groups
together), 90% intervals.

## Decision rules

For each delay:

1. **Reliability gate.** The confirmatory target's within-moment test-retest
   Spearman (mean over the 3 pairs of batch orders) must be ≥ 0.7. If not,
   that delay is *unmeasurable* and is not scored.
2. **Support** at that delay requires the 90% interval of mean ρ(secant_error)
   to lie above 0, and the 90% interval of ρ(secant_error) − ρ(competitor) to
   lie above 0 for each of message_drift, movement_near and static_prior.

**The Second Star Slip's v2 signal is supported if it has support at ≥ 2 of
the 3 delays. Otherwise it is killed.** Other predictors are reported, not
tested.

## Predictions (recorded before running)

1. The reliability gate passes at all three delays.
2. ρ(secant_error) > 0 with its interval above 0 at all three delays.
3. The secant error beats message drift and movement at delays 16 and 64. At
   256, drift is close (as in v1's exploratory result), and I expect no clear
   win there.
4. Against the static prior I am least sure. The carry chain is learned
   progressively from low bits upward, so which interface matters should move
   over time, which favours a dynamic signal. Overall prediction:
   **supported at 16 and 64, not at 256: supported overall.** I would put this
   near even odds.
5. The Hessian form ranks worse than the secant form at every delay
   (kinks dominate real steps).

## Interpretation limits

One synthetic task with scalar interfaces, small widths, ten seeds, and a
learning phase of about ten epochs. The predictors use exact full-batch
gradients; a deployed signal would be stochastic. A pass supports the signal
as worth a scheduling or correction experiment (v3), not as a training
method.
