# Research

This directory extends, but does not modify, the original Delta Observer
experiment.

## Highlights (2026-10-29)

1. **Staleness in modular training.** Start with
   [`reports/staleness_summary.md`](reports/staleness_summary.md).
   - **New tool, the permutation twin and bound.** Scramble a training
     error's entries within each tensor to test whether its *direction*
     matters, and use the result as a ceiling for delay compensation.
   - **Confirmed:** staleness harm is directional under Adam and SGD, and
     excess interface width costs staleness tolerance.
   - **Exploratory:** under pipeline staleness almost all harm is
     directional. Basis Rotation recovers about a third of the available
     gap; first-order compensation recovers none.
2. **The paper's transient clustering is an observer artifact.** The rise
   needs carry supervision of the observer, and the fall needs its
   bit-position head. The source networks' own activations show no transient
   carry-count clustering. See
   [`reports/scaffolding_ablation_results.md`](reports/scaffolding_ablation_results.md).
3. **Optimizer maths and exact Hessians of the paper's models.** All four
   pre-registered predictions held, including edge of stability within 0.4%.
   See [`reports/curvature_results.md`](reports/curvature_results.md).

## Observer.D

**Observer.D** is this directory's trajectory-distillation method. Its
initial benchmark asks whether a label-free representation of
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

## Transient-clustering ablation

[`reports/scaffolding_ablation_results.md`](reports/scaffolding_ablation_results.md)
tests which factors produce the original paper's transient-clustering result
([protocol](reports/scaffolding_ablation_protocol.md)). Run in the paper
environment (root `requirements.txt`); see the report for commands.

## Optimizer maths and curvature dynamics

[`reports/curvature_results.md`](reports/curvature_results.md) checks
optimizer identities (AdamW, Muon, Newton, Gauss–Newton) and measures exact
Hessians of the paper's source models over training
([protocol](reports/curvature_protocol.md)). Run it in the paper environment.

## Staleness in modular training

Summary and reading order:
[`reports/staleness_summary.md`](reports/staleness_summary.md). The reports,
in the order the work happened:

| Stage | Report | Status |
|---|---|---|
| Gap note | [`second_star_slip.md`](reports/second_star_slip.md) | Candidate gap, with kill criteria |
| Curvature scheduling v1 | [`second_star_slip_results.md`](reports/second_star_slip_results.md) ([protocol](reports/second_star_slip_protocol.md)) | Killed |
| Curvature scheduling v2 | [`second_star_slip_v2_results.md`](reports/second_star_slip_v2_results.md) ([protocol](reports/second_star_slip_v2_protocol.md)) | Killed, repeatably |
| Interface width and locality | [`interface_sweep_v0.md`](reports/interface_sweep_v0.md) | Exploratory |
| Interface lifecycle, rescue, unknown interface | [`contract_experiments_v0.md`](reports/contract_experiments_v0.md) | Exploratory; several claims killed |
| Prior-art check | [`prior_art_check_v0.md`](reports/prior_art_check_v0.md) | — |
| Confirmation v1 | [`interface_confirm_results.md`](reports/interface_confirm_results.md) ([protocol](reports/interface_confirm_protocol.md)) | Width **confirmed**; other tests underpowered |
| Confirmation v2 | [`interface_confirm_v2_results.md`](reports/interface_confirm_v2_results.md) ([protocol](reports/interface_confirm_v2_protocol.md)) | Directional harm **confirmed** (Adam, SGD) |
| Floor map and mechanism | [`staleness_floor_v0.md`](reports/staleness_floor_v0.md) | Exploratory |
| Muon/Adam hybrids | [`hybrid_staleness_v0.md`](reports/hybrid_staleness_v0.md) | Exploratory |
| Stability theory and compensation | [`stability_margin_v0.md`](reports/stability_margin_v0.md) | Theory killed quantitatively; bound introduced |
| Basis Rotation, cross-stage | [`basis_rotation_bound_v0.md`](reports/basis_rotation_bound_v0.md) | Exploratory; outside its design regime |
| Basis Rotation, pipeline | [`pipeline_bound_v0.md`](reports/pipeline_bound_v0.md) | Exploratory; headline figure |

Harnesses are in `research/experiments/`, tests in `research/tests/`, and raw
results in `research/results/`. The optimizers are in `research/optimizers/`:
Muon, two Muon/Adam hybrids and a Basis Rotation reimplementation. Every
experiment runs in the paper environment (root `requirements.txt`) from the
repository root, for example:

```bash
python -m unittest research.tests.test_pipeline_bound
python -m research.experiments.pipeline_bound
python -m research.experiments.plot_pipeline_bound
```

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
