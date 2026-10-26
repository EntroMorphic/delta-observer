# Observer.D Controlled Experiments

This directory extends, but does not modify, the original Delta Observer
experiment. The new trajectory-distillation method is named **Observer.D**.
The initial benchmark asks whether a label-free representation of
two source learning trajectories improves a detached student's behavioral
learning, and whether that result depends on AdamW versus a Muon hybrid.

## Mathematical comparison

For student logits `s(x)`, source ensemble probability `p_T(x)`, student
representation `q(x)`, and observer trajectory signature `z(x)`, the benchmark
uses three matched mechanisms:

| Mechanism | Objective |
|---|---|
| `task` | `BCE(s(x), y)` |
| `task_observer` | `BCE(s(x), y) + λ MSE(q(x), z(x))` |
| `distill` | `BCE(s(x), p_T(x))` |
| `observer_distill` | `BCE(s(x), p_T(x)) + λ MSE(q(x), z(x))` |

Observer.D receives paired source states, state deltas, and a
normalized checkpoint identifier. It is trained only to reconstruct states and
deltas. It never receives task labels. The student receives original inputs
only and is evaluated after the sources and observer are detached.

The default Observer.D signature encoder runs in native C using explicit ARM
NEON vector intrinsics. Training currently remains in PyTorch; results report
training and native encoding costs separately and verify native/PyTorch
numerical parity. Use `--observer-backend pytorch` for the reference path.

The ctypes bridge automatically builds
`research/observers/observer_d_neon.c` with the local C compiler. On ARM64 it
reports the `c-neon` backend; other architectures use the same C API with a
scalar dot-product fallback. Generated shared libraries are ignored by Git.

## Architecture definition

[`reports/observer_d_foundations.md`](reports/observer_d_foundations.md) defines
Observer.D as the executable third model produced during observation—not as a
teacher for a subsequently initialized student. The existing optimizer
comparison is retained as a historical control while the v0 discovery and
deployment paths are implemented as one model.

`observer_distill` versus `distill` is the primary label-free mechanism
comparison. `task_observer` versus `task` measures observer guidance with
direct supervision.
AdamW versus Muon within each row is the optimizer comparison. Source and
observer costs are reported separately.

## Run

```bash
python -m unittest discover -s research/tests -t .
python -m research.experiments.optimizer_comparison --seeds 0 1 2 3 4
```

For a quick smoke test:

```bash
python -m research.experiments.optimizer_comparison \
  --seeds 0 --source-epochs 5 --observer-epochs 5 --student-epochs 5
```

Results are written to `research/results/optimizer_comparison.json`. They
include per-epoch validation learning curves, one-shot final held-out metrics,
split hashes, source/observer
overhead, optimizer state size, approximate student FLOPs, and per-seed final
metrics.

## Interpretation limits

- This first benchmark tests trajectory-assisted distillation, not a new
  parameter-update rule.
- The observer is label-free, but its sources learned from labels.
- Random held-out combinations test interpolation, not algorithmic
  generalization. Variable-width arithmetic remains necessary.
- AdamW and Muon learning rates require a tuning sweep before making optimizer
  efficiency claims.
