# Calibrated floor predictor: protocol

*Fixed 2026-10-29, before any held-out run. Exploratory.*

The uncalibrated stress burst
([`stability_margin_v0.md`](stability_margin_v0.md)) was too pessimistic at a
2× loss-ratio threshold. This tests whether a threshold **fitted on the eight
existing settings and then frozen** predicts the floor on settings it has
never seen.

## Calibration (done, on existing data only)

`python -m research.experiments.calibrated_floor calibrate`. Candidates R ∈
{2, 4, 8, 16, 32, 64, 128, 256, 1000}. Choose the most settings within 2×,
then the lowest mean |log2((p+0.5)/(m+0.5))|, then the smallest R.

| R | within 2× | Spearman |
|---|---|---|
| 8 | 3/8 | 0.57 |
| 16 | 2/8 | 0.62 |
| **32 (frozen)** | **5/8** | **0.78** |
| 64 | 5/8 | 0.78 |
| 128 | 4/8 | 0.68 |

**In-sample this is weak:** the constant guess "floor = 4" (the median
calibration floor) gets 6/8 within 2×.

## Held-out test

- **Settings, new learning rates only:** Adam 1/200, 1/50; SGD 0.05, 0.2;
  Muon 0.005, 0.02; momentum-free SGD 0.5, 2.0.
- **Measured floor:** seeds 610–614, both H2 arms, stale condition, 120
  epochs. The floor is the largest grid delay up to which at least 8 of 10
  runs fit, the same rule as before.
- **Predictor:** the stress burst unchanged (seeds 600–602, arm B_cont_w4,
  start at 25% accuracy, 256 steps); the median of the three per-seed floors
  at R = 32.
- **Baselines:**
  - constant 4;
  - learning-rate scaling: the nearest same-optimizer calibration setting
    with a nonzero floor, times lr_cal / lr, snapped to the grid.

## Rule

The calibrated predictor **holds** only if both of these are true on the 8
held-out settings:
1. it has strictly more settings within 2× than the constant baseline;
2. its Spearman correlation with the measured floor is at least 0.5.

Otherwise it is **killed**. The comparison against learning-rate scaling is
reported, but it is secondary. A predictor that loses to it has no practical
use, even if it holds.
