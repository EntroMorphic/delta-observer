# Calibrated floor predictor: results

*2026-10-29. Exploratory. The protocol ([`calibrated_floor_protocol.md`](calibrated_floor_protocol.md))
was committed in 961f716 before any held-out run.*

**Verdict under the frozen rule: held, but only just. In practice it's
useless: a learning-rate-scaling rule that costs nothing beats it on every
measure.**

## Result

The stress-burst threshold (R = 32) was fitted on eight settings and frozen,
then tested on eight new learning rates with fresh seeds:

| Setting | Measured floor | Calibrated burst | Constant 4 | lr scaling |
|---|---|---|---|---|
| Adam 1/200 | 1 | 64 | 4 | 2 |
| Adam 1/50 | 0 | 2 | 4 | 2 |
| SGD 0.05 | 16 | 64 | 4 | 8 |
| SGD 0.2 | 1 | 1 | 4 | 2 |
| Muon 0.005 | 32 | 16 | 4 | 16 |
| Muon 0.02 | 4 | 4 | 4 | 8 |
| SGD (no momentum) 0.5 | 16 | 64 | 4 | 16 |
| SGD (no momentum) 2.0 | 0 | 8 | 4 | 4 |

| Predictor | Within 2× | Spearman | Mean log2 error |
|---|---|---|---|
| Calibrated burst | 3/8 | 0.52 | 2.09 |
| Constant 4 | 1/8 | — | 2.01 |
| **lr scaling (free)** | **6/8** | **0.83** | **1.23** |

The rule needed more settings within 2× than the constant (3 > 1) and a
Spearman correlation of at least 0.5 (0.52). Both conditions were met, so
it holds.

## Why it holds only on paper

- **The constant baseline turned out weak.** The held-out learning rates
  span a wider floor range (0–32) than the calibration set (0–8), so a
  constant guess of 4 misses most of them. Beating it is a low bar. I set
  this rule myself, and I should have made the free learning-rate-scaling
  rule the primary baseline.
- **lr scaling wins everywhere.** It takes the nearest known floor for the
  same optimizer and multiplies by lr_cal / lr. It gets twice as many
  settings within 2×, ranks the settings better and roughly halves the
  error. It needs no runs at all on the new setting.
- **The burst is unstable from seed to seed.** On the same setting, one seed's
  loss ratio can reach 1,000–10,000 while another stays below 20. Examples:
  Adam 1/50 and 1/200, and SGD 0.2. The median of three seeds does not tame
  this. The burst is measuring which side of a short-term instability a
  particular seed lands on, not the floor.

## Side observations

- **Within each optimizer, the floor falls as the learning rate rises,**
  across a range more than 3× wider than before (for example, Muon's floor is
  32, 8, 4 and 4 at lr 0.005, 0.01, 0.02 and 0.03). This supports the
  "floor ∝ 1/lr" guidance. The one exception is Adam 1/200. Its floor of 1
  is a counting artifact: 7 of 10 runs fit at delay 2, and 8 of 10 fit at
  every delay up to 16. With one more success it would be 16, which matches
  the trend.
- **The ≥8-of-10 floor rule is itself noisy by about one run.** This sets a
  limit on how well any predictor can score.

## What this means

The question was whether the floor can be predicted without running to it.
The cheap, honest answer is to interpolate from known floors using
learning-rate scaling. Five sensitivity probes have now been tried: linear
sharpness theory, the block model, the descent ratio, the uncalibrated burst
and the calibrated burst. None beats that baseline. Any future predictor
should be scored against learning-rate scaling, not against a constant.

Data:
- `research/results/calibrated_floor_{calibrate,stress,floors,evaluate}_v0.json`;
- harness: `research/experiments/calibrated_floor.py`.
