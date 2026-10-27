# The Second Star Slip

A candidate gap, documented on 2026-10-27. **Status: v1 killed under its
pre-registered criterion; a narrower within-moment version is open** (see
[`second_star_slip_results.md`](second_star_slip_results.md)). Not shown to be
novel. This file records what the gap is, how it differs from the
nearest prior work, what evidence prompted it, and how to kill it.

## The idea

Train a network made of modules. Measure, during training, how much loss
curvature crosses each interface between two modules: the off-diagonal block
`H_ij` of the Hessian, not the curvature inside a module and not its overall
size. Use that measurement to decide, interface by interface and over time,
whether two modules must exchange fresh signals every step, can run on stale
signals, or can be frozen.

In one line: **cross-module curvature, measured online, prices how stale each
interface may be.**

## Where the boundary lies

| Nearest work | What it does | What it leaves open |
|---|---|---|
| Asynchronous coordinate descent theory (Liu & Wright, JMLR 2015) | Tolerable delay depends on cross-coordinate interaction (`L_res / L_max`) | A static, global constant from the problem class, for analysis; not measured per interface during nonconvex training, not used to schedule |
| CurvaDion (Kumar, Jin & Quesnelle, 2025) | Synchronizes data-parallel workers more in high-curvature regions, using a per-layer momentum-change proxy | Uses curvature *size*, not curvature *between* modules; decides when replicas sync, not which interfaces need fresh signals |
| Local SGD with adaptive synchronization; LocalNewton | Communication frequency chosen from loss or theory (negative curvature bounds) | Data parallelism over whole-model replicas, not model parallelism across modules |
| Synthetic gradients (Jaderberg et al., 2017); asynchronous pipeline staleness via Hessian eigenbasis (2026) | Modules train on predicted or stale interface signals | Staleness is fixed or mitigated, not scheduled from measured coupling |
| Fisher-drift freezing (2026); FreezeOut, Egeria, SVCCA freeze training | Freeze a layer when its own statistics stop changing | Within-layer signals; no use of coupling to neighbours |
| Block-diagonal curvature methods (K-FAC, block Hessian-free, Botev et al. 2017; Dangel & Hennig) | Drop cross-block curvature for tractability | Assume coupling is negligible instead of measuring it |
| Critical learning periods (Achille, Rovere & Soatto, 2019); Hessian structure (Dong et al., 2025) | Show cross-layer coupling rises early then declines; block structure has architectural and training causes | Descriptive; not turned into a training rule |
| Block-hollow functional Hessian (Singh, Bachmann & Hofmann, 2021; Singh, Hofmann & Schölkopf, 2023) | For piecewise-linear networks the residual curvature has zero within-layer blocks | Supplies the mechanism (below); no scheduling use |

The candidate gap is the intersection the table leaves empty: per-interface,
time-varying, measured coupling as a control signal in model-parallel or
modular training.

## Why it surfaced here

From `curvature_results.md` and its exploratory block-structure addendum
(3 seeds, a 4-bit adder; not pre-registered):

- In the compositional model, cross-module curvature can only pass through the
  carry wire. At initialization 99% of curvature is within modules.
- Coupling rises while the carry interface is learned (epochs 10–20, within-module
  share falling to 0.74–0.85), then partly recedes. After training, keeping only
  within-module curvature recovers 67%, 94% and 98% of the top eigenvalue in
  seeds 0, 2 and 1, ordered the same way as how tightly each seed fit.
- In ReLU networks all residual (error-weighted) curvature, including its
  negative directions, lies *between* layers. The cross-block measurement
  therefore captures exactly the part block-diagonal methods discard.

So the coupling is (a) localized to named interfaces, (b) time-varying and
largest when it matters, and (c) the home of the curvature that local methods
ignore. Those three facts are what a schedule would exploit.

## The signal

For a partition of parameters into modules, define interface coupling

`κ_ij(t) = ‖H_ij‖_F / sqrt(‖H_ii‖_F · ‖H_jj‖_F)`

No full Hessian is needed. With a Rademacher probe `v` supported on block `j`,
`E‖(H v)_i‖² = ‖H_ij‖_F²`, so one Hessian-vector product per probe and per block
`j` estimates every `κ_ij` at once (about 2.7 gradients each, measured A2).
A Gauss-Newton-vector product gives the positive part alone, so the residual
share of each interface's coupling is also available.

## Proposed rule (to be tested, not adopted)

- `κ_ij > τ_high`: modules `i` and `j` exchange fresh interface signals every step.
- `κ_ij < τ_low`: the interface may run up to `k` steps stale.
- Module `i` with small gradient and all its `κ_i·` below `τ_low`: freeze.
- Hysteresis between `τ_low` and `τ_high`; thresholds set relative to the
  run's own early-training range, not tuned per task.

## Falsification plan

Each step gets a frozen protocol before it runs.

1. **Price of staleness.** Compositional adder, seeds 0–4: run each carry
   interface `k` steps stale (k = 0, 1, 10, 100) from the start, and from
   epochs 5, 20 and 60. Measure final accuracy, epochs to fit and loss spikes.
2. **Does κ predict the damage?** Correlate `κ_ij(t)` with the harm caused by
   staleness started at `t`. Compare against curvature size (`λ_max`), a
   momentum-change proxy (CurvaDion-style) and gradient norm. **Kill
   criterion:** if κ does not predict harm better than the best of those, the
   idea adds nothing.
3. **Schedule at matched budget.** Adaptive κ rule versus fixed staleness and
   versus a curvature-size trigger, all at the same number of fresh interface
   exchanges. **Kill criterion:** no improvement in accuracy or time-to-fit at
   equal communication.
4. **Leave the toy.** Wider interfaces (an 8- or 16-bit adder chain; a
   two-stage pipeline split of an MLP or small transformer). Nothing above
   counts as general until it holds here.

## Novelty checks still owed

Found since v1: DC-ASGD (Zheng et al., ICML 2017,
[arXiv 1609.08326](https://arxiv.org/abs/1609.08326)) corrects stale gradients
with the first-order term `H (w_t − w_τ)` in data-parallel training. The
v1 diagnostics show that staleness error across a module interface is exactly
this term restricted to the cross-module block, so the mechanism is known;
only its per-interface use for scheduling or correction remains a candidate.

The search behind this note was about a dozen queries, several answered from
abstracts. Before any novelty claim:

- Read CurvaDion and the selective-synchronization paper (Tyagi & Swany, 2023)
  in full; check whether either measures cross-layer terms.
- Read Liu & Wright's delay bound and later asynchronous block-coordinate work;
  an adaptive, measured version may already exist there.
- Search domain decomposition and Schwarz-preconditioner literature applied to
  neural network training; interface (Schur-complement) coupling deciding
  communication is standard in numerical PDEs and may already have been
  carried over.
- Check asynchronous pipeline-parallel work (PipeDream and successors) for
  staleness schedules driven by curvature.

## Sources

- Liu & Wright, [An Asynchronous Parallel Stochastic Coordinate Descent Algorithm](https://jmlr.org/papers/v16/liu15a.html), JMLR 2015
- Kumar, Jin & Quesnelle, [CurvaDion](https://arxiv.org/abs/2512.13728), 2025
- Tyagi & Swany, [Accelerating Distributed ML Training via Selective Synchronization](https://arxiv.org/abs/2307.07950), CLUSTER 2023
- Haddadpour et al., [Local SGD with Periodic Averaging](https://arxiv.org/abs/1910.13598), 2019
- [LocalNewton](https://arxiv.org/pdf/2105.07320v1.pdf), 2021
- Jaderberg et al., [Decoupled Neural Interfaces using Synthetic Gradients](https://arxiv.org/abs/1608.05343), ICML 2017
- [Mitigating Staleness in Asynchronous Pipeline Parallelism via Basis Rotation](https://arxiv.org/html/2602.03515v1), 2026
- [Fisher-Guided Progressive Parameter Selection for Adaptive Fine-Tuning](https://arxiv.org/pdf/2606.10196), 2026
- [FreezeOut](https://paperswithcode.com/paper/freezeout-accelerate-training-by), 2017
- Botev, Ritter & Barber, [Practical Gauss-Newton Optimisation for Deep Learning](https://proceedings.mlr.press/v70/botev17a.html), ICML 2017
- Dangel & Hennig, [Modular Block-diagonal Curvature Approximations](https://arxiv.org/abs/1902.01813), 2019
- [Block-diagonal Hessian-free Optimization](https://arxiv.org/abs/1712.07296), 2017
- Achille, Rovere & Soatto, [Critical Learning Periods in Deep Neural Networks](https://arxiv.org/abs/1711.08856), ICLR 2019
- Dong et al., [Towards Quantifying the Hessian Structure of Neural Networks](https://arxiv.org/abs/2505.02809), NeurIPS 2025
- Singh, Bachmann & Hofmann, [Analytic Insights into Structure and Rank of Neural Network Hessian Maps](https://arxiv.org/abs/2106.16225), NeurIPS 2021
- Singh, Hofmann & Schölkopf, [The Hessian perspective into the Nature of Convolutional Neural Networks](https://arxiv.org/abs/2305.09088), ICML 2023
