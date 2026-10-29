# Interface confirmation v2: protocol

Frozen before any confirmatory run. v1
([`interface_confirm_results.md`](interface_confirm_results.md)) confirmed H1b
(width). It could not test H1a (locality), or H2 (direction) under SGD and
Muon: at d=16, both arms were censored on most seeds, so the paired test had
1–3 informative pairs and could not reach p < 0.05. v2 reruns only those
tests, at delays chosen by a power check.

- **Harness:** `research/experiments/interface_confirm_v2.py`. It reuses the
  v1 training, staleness, optimizers and `paired_test`.
- **Tests:** `research/tests/test_interface_confirm_v2.py`.
- **Confirmatory seeds:** 10–19, new to all prior runs.

## The power check

The rule was committed in `f8e5a38` before calibration ran. It was applied on
seeds 310–314 to one arm alone, so the comparison stayed unseen.

- **H1a:** the largest delay in {2, 4, 8} at which the wide chain (the
  expected stronger arm) fits at least 4/5 on *both* tasks.
- **H2, for each optimizer:** the largest delay in {1, 2, 4, 8, 16} at which
  the *stale* condition (the expected weaker arm) fits at least 8/10, over
  chain widths 1 and 4 × 5 seeds.

### Calibration results

Raw: `research/results/interface_confirm_v2_calibration.json`.

**H1a, wide chain fits out of 5:**

| Task | d=2 | d=4 | d=8 |
|---|---|---|---|
| Adder | 4 | **4** | 1 |
| Automaton | 5 | **4** | 3 |

**H2, stale fits out of 10:**

| Optimizer | d=1 | d=2 | d=4 | d=8 | d=16 |
|---|---|---|---|---|---|
| Adam | 9 | 9 | 10 | 9 | **9** |
| SGD | 8 | **8** | 0 | 0 | 0 |
| Muon | 10 | 10 | **8** | 2 | 0 |

**Chosen delays:**

| Test | Delay |
|---|---|
| H1a | **4** |
| H2, Adam | **16**, the same as v1 |
| H2, SGD | **2** |
| H2, Muon | **4** |

## H1a: locality

**Arms:** Adam at lr 1/300, at d=4, on both tasks:
- wide chain, width 64;
- pipeline with local heads, 64 wide;
- plain pipeline, 64 wide.

**Primary.** The wide chain fits sooner than the plain pipeline, paired by
seed. Test: one-sided Wilcoxon signed-rank on time to fit, censored at 201,
dropping ties.
- **Supported on a task** if p < 0.05 and the median paired log ratio is
  below 0.
- **Confirmed** if supported on both tasks.

**Secondary,** reported without a verdict: wide chain against local heads,
and local heads against the plain pipeline.

## H2: directional harm

**Setup:** adder chain, widths 1 and 4. Conditions are fresh, stale at the
optimizer's chosen delay, and permuted at that delay. Units are (arm, seed),
20 per optimizer. Learning rates are those from v1: Adam 1/300, SGD
(momentum 0.9) 0.3, Muon 0.03.

**Primary.** Permuted fits sooner than stale, with the same test and support
rule as H1a.
- **General** if supported for at least 2 of the 3 optimizers.
- **Adam-specific** if supported for Adam only.

**Secondary:** stale against fresh, and permuted against fresh.

## Caveat fixed in advance

The delays differ by optimizer, so H2 compares direction against size *at a
delay where that optimizer still trains*, not at matched delays. The
question it answers: once staleness is survivable, is its harm directional?

## Run

```bash
python -m unittest research.tests.test_interface_confirm_v2
python -m research.experiments.interface_confirm_v2 run   # 240 runs
```
