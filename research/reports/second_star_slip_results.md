# The Second Star Slip: v1 results

Protocol: [`second_star_slip_protocol.md`](second_star_slip_protocol.md),
frozen at `94aa513` before any run. Harness:
`research/experiments/second_star_slip.py` (`da474a0`); tests:
`research/tests/test_second_star_slip.py`. Raw results:
`research/results/second_star_slip_v1.json` (`789cc34`).

## Verdict

**Killed in this setting under the pre-registered criterion.** Cross-module
curvature (κ_cut) ranks the harm of a stale interface no better than the
global top Hessian eigenvalue at any of the three delays. It clearly beats
gradient norm, the momentum-change stand-in, and both message signals.

The exploratory analysis below suggests why. Most of the harm depends on
*when* staleness starts, and any curvature measure tracks that. The part only
an interface signal can address, *which* interface, is small in this toy, and
κ helps there only for very short delays.

## Predictions, scored

| # | Prediction | Result |
|---|---|---|
| 1a | Delay 1 is harmless (final accuracy 100% in ≥ 4/5 seeds, every configuration) | **Held**: 5/5 in every configuration |
| 1b | Delay 100 on all interfaces from epoch 0 prevents a full fit in ≥ 3/5 seeds | **Failed**: 5/5 fit, but first full fit at epochs 127–137 against 14–21 fresh |
| 1c | Onset at epoch 60 is harmless for every delay | **Not held**: final accuracy stays 100%, but delay 10 on all interfaces from epoch 60 knocks all 5 seeds off 100% accuracy until epochs 81–93. The protocol did not define "harmless", so I do not count this one as held |
| 2 | κ_cut beats λ_max, gradient norm and momentum change (kill criterion) | **Failed**: beats gradient norm and momentum change at every delay; ties λ_max at every delay |
| 3 | κ_cut does *not* beat message drift at delays 10 and 100 | **Failed**, in the other direction: κ_cut beats drift and message size at every delay |

Both of my Part 2 predictions were wrong. I expected the interface signal to
beat the global ones and lose to the cheap local ones. It did the reverse.

## Part 1 — price of staleness

Every one of the 245 runs ends at 100% accuracy, so in this task staleness
costs time and stability, not the final answer. Epoch of the first lasting
100% accuracy (fresh runs: 14–21):

| Stale interface | Delay | Onset 0 | Onset 5 | Onset 20 | Onset 60 |
|---|---|---|---|---|---|
| 0 | 10 | 22–59 | 15–52 | 16–84 | 14–81 |
| 0 | 100 | 34–151 | 78–138 | 14–116 | 14–21 |
| 2 | 10 | 18–30 | 19–33 | 39–62 | 14–108 |
| 2 | 100 | 102–163 | 57–135 | 14–88 | 14–21 |
| all | 1 | 15–37 | 16–23 | 14–26 | 14–71 |
| all | 10 | 34–50 | 29–57 | 37–74 | 81–93 |
| all | 100 | 127–137 | 93–129 | 28–108 | 14–154 |

(Ranges over seeds 0–4. Interface 1 resembles interface 0; full table in the
raw results.) Late onsets lose an already-perfect fit and then regain it. Loss
spikes (epochs where loss rises more than 10× above its running minimum) are
most frequent with late onsets, partly because late-training loss is tiny.

## Part 2 — does curvature predict harm?

Spearman correlation with harm over 105 units per delay (5 seeds × 7 onsets ×
3 interfaces):

| Delay | κ_cut | κ_adj | λ_max | Gradient norm | Momentum change | Message size | Message drift |
|---|---|---|---|---|---|---|---|
| 1 | **0.66** | 0.62 | 0.65 | −0.52 | 0.22 | −0.33 | −0.27 |
| 10 | **0.57** | 0.57 | 0.57 | −0.34 | 0.09 | −0.04 | −0.05 |
| 100 | 0.64 | 0.62 | **0.71** | −0.42 | 0.10 | −0.18 | 0.13 |

90% bootstrap intervals (resampling seeds) for ρ(κ_cut) − ρ(λ_max):
[−0.06, 0.10], [−0.08, 0.14] and [−0.17, 0.04]. All include zero. Against
every other competitor, the interval is above zero at every delay.

## Exploratory: when versus which (not pre-registered)

Harm is almost zero before epoch 10, peaks between epochs 15 and 30, and
falls by epoch 60. The (seed, onset) group alone explains 80%, 55% and 88% of
the variance in harm at delays 1, 10 and 100. λ_max is the same for all three
interfaces at a given moment, so its correlation comes entirely from timing.
κ_cut's pooled correlation is mostly timing too.

The question the Second Star Slip actually poses is which interface to protect
at a given moment. Ranking the three interfaces within each (seed, onset),
for onsets from epoch 10 on (25 groups per delay), mean Spearman ± standard
error:

| Delay | κ_cut | κ_adj | Message size | Message drift | Momentum change |
|---|---|---|---|---|---|
| 1 | **+0.52 ± 0.12** | +0.40 ± 0.12 | +0.22 ± 0.11 | +0.06 ± 0.11 | +0.00 ± 0.12 |
| 10 | +0.14 ± 0.13 | +0.20 ± 0.12 | **+0.30 ± 0.14** | +0.28 ± 0.14 | −0.20 ± 0.14 |
| 100 | −0.20 ± 0.15 | −0.06 ± 0.14 | +0.28 ± 0.13 | **+0.42 ± 0.15** | −0.16 ± 0.12 |

This is the pattern the first-order argument in the protocol predicts: harm is
roughly curvature times how far the other side has moved. For short delays,
little moves, so curvature decides which interface hurts. For long delays,
movement dominates, so drift decides. My Part 2 prediction 3 was wrong on the
pooled test but describes the within-moment ranking.

With three interfaces per group and 25 groups, these are weak results, and
they were chosen after seeing the data. They motivate a test; they are not one.

## What this means for the idea

- In its v1 form, the Second Star Slip is killed: curvature *between* modules
  adds nothing over curvature *size* for predicting when staleness hurts.
- A narrower version survives as a hypothesis: *within* a moment, cross-module
  curvature ranks interfaces for short delays, and message drift ranks them for
  long ones. A combined signal (coupling × recent movement across the cut) is
  the natural candidate, and it would need its own frozen protocol.
- This toy has three nearly identical scalar interfaces (mean harm by
  interface differs by at most 1.4×), so it has little "which interface"
  signal to find. A fair test needs interfaces that differ, such as unequal
  widths, depths or learning speeds.

## Failure anatomy (exploratory, not pre-registered)

Harness: `research/experiments/second_star_slip_diagnostics.py`; raw results:
`research/results/second_star_slip_diagnostics_v1.json`. It rebuilds every v1
onset state (the fresh run is deterministic and matches v1 exactly) and
measures the mechanism linking staleness to harm. Five findings, in order of
how much they explain.

### 1. The harm measure mostly tracked training progress

Harm was a log ratio of losses one epoch later. The fresh run's own loss at
the onset predicts it about as well as or better than any candidate
(ρ = −0.70, −0.52, −0.64 at delays 1, 10, 100). The same absolute slip counts
for more as the loss shrinks: mean absolute harm peaks at epoch 10, log harm
at epochs 20–60. Coupling and the top eigenvalue both rise as the model
learns, so both inherit this correlation. That is why they tied: the pooled
test asked which signal tracks progress, and nearly any curvature signal
does.

### 2. One-epoch harm does not predict the long-run cost

Matched against Part 1 runs with the same seed, interface, delay and onset
(135 pairs), one-epoch harm does not predict how much later the model first
reaches lasting 100% accuracy (ρ = −0.03 overall; −0.25, 0.52 and −0.37 at
each delay). It correlates moderately with loss spikes (ρ = 0.51), which are
the same kind of relative loss jump. The v1 target measured local damage
that training mostly recovers from.

### 3. The interface ranking is mostly noise at short delays

Replaying each window with two other batch orders (same state, same
staleness), the within-moment ranking of the three interfaces repeats with
mean Spearman 0.40 ± 0.09 (delay 1), 0.48 ± 0.10 (delay 10) and 0.77 ± 0.05
(delay 100). At delays 1 and 10 no predictor could rank interfaces much better
than chance, because the target does not repeat. Coupling's +0.52 at delay 1
falls to +0.32 ± 0.12 against harm averaged over the three batch orders. Only
delay 100 has a "which interface" signal worth measuring.

### 4. The mechanism is right; the v1 signal discarded half of it

The gradient error that staleness causes is, to first order,
`e = −H_AB Δθ_B`: cross-module curvature applied to how far the other side has
moved. Measured directly against the exact error, this linearization has
median cosine 0.99, 0.96 and 0.74 at delays 1, 10 and 100, with magnitude
ratio 0.95–1.02. So the Second Star Slip's physical picture holds.

The v1 signal κ_cut is a ratio of *norms*. It ignores the movement vector
`Δθ_B`, and within a moment it ranks the actual cross-module error poorly
(ρ = 0.28, 0.28, 0.18). At delay 100, against the averaged harm, the
vector-form error ranks interfaces at +0.38 ± 0.12 (linearized) and
+0.40 ± 0.14 (exact). Message drift ranks them at +0.46 ± 0.14 and κ_cut at
−0.12 ± 0.13.

Pooled, the relative gradient error predicts harm better than the top
eigenvalue at delays 10 and 100 (ρ = 0.67 and 0.83, against 0.57 and 0.71).
But it still shares finding 1's confound, since its denominator shrinks with
the gradient.

### 5. Not explained by Adam or instability

Adam-preconditioned coupling predicts worse than raw coupling (pooled
ρ = 0.45–0.52). Adam's preconditioned sharpness, lr × λ_max(PH), stays at
2–15, below the roughly 38 at which Adam with β₁ = 0.9 reaches its edge of
stability (Cohen et al., 2022). So the windows are not instability events.

### What this means

The v1 failure is mostly a failure of the experiment, not a refutation of the
mechanism. The target was confounded by progress, did not predict long-run
cost, and was too noisy to rank interfaces at short delays. In a testbed
whose three interfaces are nearly identical, the signal was the scalar half of
a vector quantity.

The vector form `H_AB Δθ_B` has a known relative: DC-ASGD (Zheng et al.,
ICML 2017) *corrects* stale gradients in data-parallel training with a
first-order Taylor term `H (w_t − w_τ)`, using a cheap Hessian approximation.
The candidate gap therefore narrows again. What remains open is using the
*cross-module* part of that term to decide, interface by interface, when to
communicate, or to correct a stale interface instead of communicating.

A v2 should:

1. Use a target that predicts long-run cost: matched-budget time-to-fit or
   loss at a fixed horizon, normalized for progress.
2. Measure the target's test-retest reliability first, and score predictors
   only where it is at least 0.7.
3. Use the vector signal `‖H_AB Δθ_B‖`. It costs one Hessian-vector product
   per interface on the recent movement, with message drift as the cheap
   rival to beat.
4. Use interfaces that differ in width, depth or learning speed, so "which
   interface" has an answer.
5. Include the corrective alternative: DC-ASGD-style compensation across
   interfaces, compared with scheduling.

## Reproduce

```bash
python -m unittest research.tests.test_second_star_slip
python -m research.experiments.second_star_slip --workers 6   # ~15 min on 6 cores
python -m research.experiments.second_star_slip_diagnostics     # ~10 min
```
