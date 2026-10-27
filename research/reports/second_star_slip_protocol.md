# The Second Star Slip: measurement protocol (v1)

Frozen before any run. Tests steps 1 and 2 of the falsification plan in
[`second_star_slip.md`](second_star_slip.md): what interface staleness costs,
and whether cross-module curvature predicts that cost better than simpler
signals. Exploratory in scope (one toy task); confirmatory in form (predictions
and kill criteria fixed here).

## Model and training

The paper's compositional adder (`PaperComp`): four full-adder modules, module
`j` passing its carry to module `j+1`. Interfaces `j = 0, 1, 2`. Training as in
`research/experiments/curvature_dynamics.train_sources` (Adam, α = 1/137, warmup
and cosine schedule, MSE, batch 32, per-epoch `randperm`, 200 epochs = 3,200
steps), with the same RNG order, so the fresh run reproduces that harness's
compositional trajectory exactly (unit test).

## Staleness

Each interface `j` has a delay `k_j` in optimizer steps. The delay between
modules `m` and `n` is the sum of `k_j` over the interfaces between them. At
step `t`, module `m`'s gradient comes from a forward pass on the current batch
in which `m` uses its current parameters and every other module `n` uses its
parameters from step `t − delay(m, n)` (held fixed). This models every message
across an interface, forward carries and backward gradients, arriving `k_j`
steps late. All `k_j = 0` is ordinary joint training (unit test: identical
parameters to the fresh run). One Adam optimizer over all parameters, as in the
fresh run; only the gradients change.

## Part 1 — price of staleness (descriptive)

Seeds 0–4. One interface stale at a time (`j = 0, 1, 2`) and all three at once;
`k ∈ {1, 10, 100}`; staleness switched on at epoch 0, 5, 20 or 60 and kept to
epoch 200. Recorded: final full-batch loss and exact-match accuracy, first epoch
at 100% accuracy, and the number of epochs where loss rises by more than 10×
over its running minimum. 5 × 4 × 3 × 4 = 240 runs, plus 5 fresh runs.

## Part 2 — does cross-module curvature predict the harm?

**Units.** Seeds 0–4 × onset epochs {2, 5, 10, 15, 20, 30, 60} × interfaces
{0, 1, 2} × `k ∈ {1, 10, 100}`.

**Harm.** From the fresh trajectory at the onset step `t₀`, run 16 steps
(one epoch, same batches) with interface `j` at delay `k` and all others fresh;
the stale views before `t₀` come from the fresh trajectory's history. Harm =
log₁₀ L_stale − log₁₀ L_fresh, full-batch loss after those 16 steps, where
L_fresh is the fresh run at `t₀ + 16`.

**Predictors, all computed at `t₀` from information available then:**

| Predictor | Definition |
|---|---|
| **κ_cut (primary)** | `‖H_AB‖_F / sqrt(‖H_AA‖_F ‖H_BB‖_F)`, exact full-batch Hessian, A = modules ≤ j, B = modules > j |
| κ_adj (secondary) | Same, A = module j, B = module j+1 only |
| λ_max | Top Hessian eigenvalue (curvature size, global) |
| Gradient norm | Full-batch gradient norm (global) |
| Momentum change | Relative change of Adam's first moment over the previous 16 steps, modules j and j+1: `‖m_t − m_{t−16}‖ / ‖m_t‖`. Our stand-in for CurvaDion's signal, not its exact definition |
| Message size | RMS over the batch of ∂L/∂c_j, the backward message across interface j |
| Message drift | RMS change of the carry c_j over the previous `k` steps of the fresh run (how fast the forward message has been changing), computable at `t₀` |

**Statistic.** Spearman rank correlation between each predictor and harm,
computed separately for each `k` across the 105 units (5 seeds × 7 onsets × 3
interfaces). Uncertainty: 2,000 bootstrap resamples over seeds (resampling
whole seeds), 90% interval of the difference `ρ(κ_cut) − ρ(competitor)`.

## Decision rules

- **Kill criterion (from the gap note):** the Second Star Slip's signal is
  *supported* only if `ρ(κ_cut)` exceeds `ρ` of each of λ_max, gradient norm
  and momentum change, with the 90% interval of the difference above 0, for at
  least 2 of the 3 values of `k`. Otherwise it is *killed* in this setting.
- **Stricter test:** the same comparison against message size and message
  drift. These are cheap, interface-local signals; if either matches κ_cut, the
  curvature measurement adds nothing an engineer would pay for.
- Part 1 has no pass/fail rule; it prices staleness for Part 3 (not in this
  protocol).

## Predictions (recorded before running)

1. Part 1: delay 1 is harmless everywhere (final accuracy 100% in ≥ 4/5 seeds
   for every configuration). Delay 100 from epoch 0 on all interfaces prevents
   a full fit in ≥ 3/5 seeds. Onset at epoch 60 is harmless for every `k`.
2. Part 2, kill criterion: **κ_cut passes** against λ_max, gradient norm and
   momentum change (the interface signal beats global signals).
3. Part 2, stricter test: **κ_cut does not beat message drift** for `k = 10` and
   `k = 100`. First-order reasoning: the harm of a stale message is roughly
   curvature × how far the other side has moved, and drift measures the second
   factor directly. I expect the curvature measurement alone to lose to it.
   Uncertain.

## Interpretation limits

One toy task with scalar interfaces and five seeds. A one-epoch harm window
measures local damage, not long-run outcome. Exact Hessians stand in for the
probe-based estimator described in the gap note; nothing here tests the
estimator's noise. A pass is evidence for a signal worth testing at scale, not
for a training method.
