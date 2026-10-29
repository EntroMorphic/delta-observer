# Interface confirmation: results

Protocol: [`interface_confirm_protocol.md`](interface_confirm_protocol.md),
frozen at `dca1394` with its harness, tests and calibration. Raw results:
`research/results/interface_confirm_v1.json`. Confirmatory seeds 0–9, 420
runs.

## Verdicts, by the pre-registered rules

| Hypothesis | Verdict |
|---|---|
| **H1a.** Locality: the wide chain fits sooner than the same-width plain pipeline at d=16 | **Not supported on either task.** The kill criterion is formally met, but the test had almost no power (see below). |
| **H1b.** Width: the narrow-but-sufficient chain fits sooner than the wide chain at d=16 | **Confirmed on both tasks** |
| **H2.** Directional harm: the permuted error fits sooner than the real staleness error | **Supported for Adam only.** The rule classifies this as Adam-specific. For SGD and Muon the test was untestable (floor effect, below), not refuted. |

## H1: fits out of 10, median epochs to fit

| Task | Arm | d=0 | d=4 | d=16 |
|---|---|---|---|---|
| Adder | Chain, width 4 | 10/10, 11 | 10/10, 13 | **9/10, 23** |
| Adder | Chain, width 64 | 9/10, 22 | 7/10, 32 | 1/10, 115 |
| Adder | Pipeline + local heads, 64 | 10/10, 11 | 10/10, 63 | 0/10 |
| Adder | Plain pipeline, 64 | 10/10, 22 | **0/10** | 0/10 |
| Automaton | Chain, width 16 | 10/10, 19 | 10/10, 32 | **9/10, 122** |
| Automaton | Chain, width 64 | 10/10, 22 | 9/10, 44 | 3/10, 156 |
| Automaton | Pipeline + local heads, 64 | 10/10, 40 | 5/10, 131 | 0/10 |
| Automaton | Plain pipeline, 64 | 7/10, 89 | 1/10, 169 | 0/10 |

### H1b, primary at d=16

| Task | Narrow chain fits | Wide chain fits | Wins / losses | p (one-sided) |
|---|---|---|---|---|
| Adder | 9/10 | 1/10 | 9 / 0 | 0.002 |
| Automaton | 9/10 | 3/10 | 8 / 1 | 0.004 |

**Confirmed.** At fixed locality, excess message width costs staleness
tolerance.

### H1a, primary at d=16

- **Adder:** the wide chain fit 1/10 and the pipeline 0/10, giving 1 informative
  pair; p = 0.5.
- **Automaton:** the wide chain fit 3/10 and the pipeline 0/10, giving 3
  informative pairs; p = 0.125.

At d=16 the wide chain fails almost as often as the pipeline, so nearly every
pair is a tie and is dropped. **This was a protocol design error.**
Exploration had already shown the width-64 chain failing at d=16 (0/5 on the
adder). The primary locality test should have used d=4, where the wide chain
still trains.

### H1a at d=4 (pre-registered secondary)

| Task | Wide chain fits | Pipeline fits | Wins / losses | p (one-sided) |
|---|---|---|---|---|
| Adder | 7/10 | 0/10 | 7 / 0 | 0.008 |
| Automaton | 9/10 | 1/10 | 9 / 0 | 0.002 |

The local heads also help on their own at d=4:
- adder: 10/10 against 0/10 for the pipeline, p = 0.001;
- automaton: 5/10 against 1/10, p = 0.03.

**How to read this:** the locality effect is strong at d=4. By the rule
written down in advance, it is not confirmed. A d=4 primary would need a new
frozen protocol and fresh seeds before it counts.

## H2: directional harm, 20 (arm, seed) pairs per optimizer, d=16

| Optimizer | Fresh | Stale | Permuted | Permuted vs stale |
|---|---|---|---|---|
| Adam, lr 1/300 | 18/20, 12 | 17/20, 21 | **20/20, 12.5** | 20 wins, 0 losses, p < 1e-5 |
| SGD with momentum, lr 0.3 | 18/20, 8.5 | 0/20 | 1/20 | 1 informative pair, p = 0.5 |
| Muon, lr 0.03 | 19/20, 5 | 0/20 | 1/20 | 1 informative pair, p = 0.5 |

- **Under Adam the effect is large and clean.** Scrambling the staleness
  error's direction removes essentially all of its cost: permuted runs are
  about as fast as fresh ones, while stale runs take about 1.8× as long.
- **Under SGD and Muon, staleness at d=16 is lethal in both forms.** An error
  of that size kills training whatever its direction. Direction cannot matter
  when magnitude already decides the outcome.
- **The adaptive optimizer is the most staleness-tolerant here.**
  - Adam fits 17/20 at d=16; SGD and Muon fit 0/20.
  - That is the reverse of the ordering Zmushko et al. (2026) report for a
    one-step delay at scale, where AdamW degrades and Muon is robust. The
    setting is very different: a 16-step delay on every boundary, tiny
    networks, and calibrated constant learning rates. So this is a contrast,
    not a contradiction.

**How to read this:** the directional finding is real for Adam. For SGD and
Muon it is **undetermined, not refuted**: with stale 0/20 and permuted 1/20,
only 1 pair is informative, so p = 0.5 was the best possible outcome. Testing
those optimizers needs a delay at which stale runs still fit. One explanation
fits the Adam data:
- Adam normalizes each coordinate, which caps the error's per-coordinate
  size. What remains to do harm is then the error's alignment.
- SGD and Muon pass the error's full magnitude into the update, and that
  magnitude alone is fatal.

That explanation is a hypothesis; this test does not establish it.

## What this leaves

1. **Confirmed:** excess interface width costs staleness tolerance, at fixed
   locality, on two tasks. This is the narrowest of the candidates, and it
   held.
2. **Strong but unconfirmed:** locality (stages with their own inputs and
   outputs) tolerates more staleness than a same-width pipeline. The effect
   is large at d=4, but the pre-registered primary at d=16 was underpowered
   by my design.
3. **Shown for Adam only:** staleness harm under Adam comes from the error's
   direction. Under SGD and Muon its size is lethal at d=16, so direction could
   not be tested there. A lower delay is needed before calling it
   Adam-specific.

## The nature of the two non-confirmations

Both are **floor effects**. The code and its tests behaved correctly, and
the test design is what failed.

- **The scoring rule:** a run that never fits is censored at 201 epochs, and
  a seed where both arms are censored is a tie. The Wilcoxon test drops ties.
- **The resulting limit:** with n informative pairs, the smallest one-sided
  p is 1/2ⁿ. Passing at p < 0.05 needs at least 5 informative pairs, all
  pointing the same way. The rule's median-log-ratio condition also fails
  when most pairs are tied at 0.
- **H1a:** 1 and 3 informative pairs, every one favouring the chain. The test
  could not pass at any effect size.
- **H2 under SGD and Muon:** 1 informative pair each, so the same applies.
- **Cause:** I chose d=16 without checking that the arms in each comparison
  could still train at that delay. Exploration had already shown the
  width-64 chain failing at d=16.
- **Fix for any rerun:** pick each primary delay on calibration seeds, so
  that the stronger arm in each comparison fits on most seeds. This is a
  power check made before freezing, without looking at the comparison
  itself.

## Reproduce

```bash
python -m unittest research.tests.test_interface_confirm
python -m research.experiments.interface_confirm calibrate
python -m research.experiments.interface_confirm run
```
