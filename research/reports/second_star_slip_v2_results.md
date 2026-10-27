# The Second Star Slip: v2 results

Protocol: [`second_star_slip_v2_protocol.md`](second_star_slip_v2_protocol.md),
frozen at `6238498` with harness, tests and calibration pilot. Raw results:
`research/results/second_star_slip_v2.json` (`5649a6f`). Confirmatory seeds
0–9; calibration seeds 100–102 were not reused.

## Verdict

**Killed, repeatably.** The target passed the reliability gate at every delay,
so this is a measured answer, not noise. The cross-module gradient error ranks
which interface is costliest to leave stale above chance at every delay
(ρ ≈ 0.17–0.20, intervals above 0). It does not beat message drift or the
static prior at any delay, so it has support at 0 of 3 delays.

| Delay | Reliability | secant_error ρ [90% CI] | Beats drift | Beats movement | Beats static prior | Support |
|---|---|---|---|---|---|---|
| 16 | 0.73 | +0.19 [+0.04, +0.32] | no | **yes** | no | no |
| 64 | 0.80 | +0.20 [+0.13, +0.27] | no | **yes** | no | no |
| 256 | 0.91 | +0.17 [+0.09, +0.25] | no | no (borderline) | no | no |

## Predictions, scored

| # | Prediction | Result |
|---|---|---|
| 1 | Reliability gate passes at all delays | **Held** (0.73, 0.80, 0.91) |
| 2 | ρ(secant_error) above 0 at all delays | **Held**, but small |
| 3 | Beats drift and movement at delays 16 and 64 | **Failed**: beats movement, not drift |
| 4 | Supported overall (16 and 64) | **Failed**: support at 0 of 3 delays |
| 5 | The Hessian form ranks worse than the secant form | **Failed**: the Hessian form ranks as well or better (0.27, 0.22, 0.25) |

## All predictors (within-moment mean ρ, 50 groups per delay)

| Predictor | Delay 16 | Delay 64 | Delay 256 |
|---|---|---|---|
| secant_error (primary) | +0.19 | +0.20 | +0.17 |
| secant_error_adam | +0.22 | +0.28 | +0.29 |
| hessian_error | +0.27 | +0.22 | +0.25 |
| hessian_error_adam | +0.27 | +0.30 | +0.28 |
| message_drift | +0.13 | +0.22 | +0.13 |
| message_size | +0.11 | +0.21 | **+0.34** |
| movement_near | −0.01 | +0.05 | +0.04 |
| static_prior | **+0.28** | **+0.31** | +0.14 |

No predictor exceeds ρ = 0.34, although the target repeats at 0.73–0.91.
Most of what decides which interface is costly is not captured by any of
these signals.

## What the details say (exploratory, not pre-registered)

- **Adam scaling matters.** Scaling the error by Adam's per-parameter step size
  improves the secant form at every delay. The improvement's 90% interval
  excludes zero at delays 64 and 256. The harm comes from the *update* error,
  which Adam rescales, not from the gradient error itself. A v1 diagnostic
  hinted at this; here it is clear.
- **Kinks add noise, not signal.** The exact (secant) error includes
  activation flips that the Hessian misses, and those were up to 26× larger
  in a harness test. Yet the Hessian form ranks harm at least as well. The
  flips change the gradient but mostly do not translate into lasting cost
  over a 3-epoch window.
- **The learning transition inverts the signal.** At onset epoch 4, where
  accuracy jumps (roughly 10% to 55% on calibration seeds) and the warmup is
  ending, the Adam-scaled error ranks interfaces *negatively* (−0.32 and
  −0.13 at delays 16 and 64). At the other onsets it is +0.19 to +0.44. The
  static prior also fails at epoch 4 at delay 256 (−0.49). Which interface is
  costly during the transition is not what its current error predicts. This
  was found after looking and is a hypothesis.
- **Static structure explains little.** Interface identity accounts for
  16–31% of the within-moment target variance. The static prior is the best
  single signal at short delays, but most variance is dynamic and unexplained.
- **Combining signals helps only a little.** Rank-averaging the static prior
  with an Adam-scaled error reaches ρ ≈ 0.32–0.36, and three dynamic signals
  combined reach 0.36 at delay 256. These were chosen after looking.

## What this means for the idea

Two pre-registered tests, one with a repeatable target, both kill the claim
that cross-module curvature or gradient error is *the* signal for which
interface to keep fresh. What survives is narrower and weaker:

- The mechanism is real. Cross-module error ranks interfaces above chance at
  every delay, and is best when scaled by Adam.
- But the effect is modest (ρ ≈ 0.2–0.3), no better than cheap alternatives
  (the size of the backward message across the interface, or a prior from
  other runs), and it inverts during the learning transition.

An honest summary for anyone building on this: in a heterogeneous modular
chain trained with Adam, the per-interface cost of staleness is repeatable but
mostly not predicted by local curvature, gradient error, message drift or
static structure.

## Reproduce

```bash
python -m unittest research.tests.test_second_star_slip_v2
python -m research.experiments.second_star_slip_v2 --pilot --onsets 2 4 6 8 10 --delays 16 64 256 --windows 1 3
python -m research.experiments.second_star_slip_v2 --onsets 2 4 6 8 10 --delays 16 64 256 --windows 3   # ~20 min, 8 cores
```
