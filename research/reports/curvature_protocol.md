# Optimizer mathematics and curvature-dynamics protocol

Frozen before any run. Part A checks mathematical claims made in conversation
on 2026-10-26 about AdamW, Muon and second-order methods. Part B measures exact
Hessians of the paper's source models over training, asking whether curvature
shows the transient that the observer latent shows (see
`scaffolding_ablation_results.md`).

## Part A — identities (unit tests, float64 unless stated)

| Claim | Test |
|---|---|
| A1. Pearlmutter's trick gives the exact Hessian–vector product | double-backprop `Hv` equals full Hessian × `v` |
| A2. An HVP costs about two backward passes | timing ratio HVP / gradient on the paper monolithic model (reported, not a pass/fail test) |
| A3. Newton's method is attracted to saddles | on `x² − y²` one Newton step lands exactly on the saddle; with an indefinite Hessian `−H⁻¹g` can be an ascent direction |
| A4. The Gauss–Newton matrix is positive semi-definite | for MSE, `H = JᵀJ/n + Σ rᵢ∇²fᵢ / n`; `JᵀJ/n` has no negative eigenvalue while `H` can |
| A5. Adam without moving averages is sign descent | `torch.optim.Adam` with betas (0, 0) steps by `−lr·sign(g)` |
| A6. `sign(g)` is steepest descent under ℓ∞ | `gᵀsign(g) = ‖g‖₁ ≥ gᵀd` for all `‖d‖∞ ≤ 1` |
| A7. `UVᵀ` is steepest descent under the spectral norm | `⟨G, UVᵀ⟩ = ‖G‖_* ≥ ⟨G, D⟩` for all `‖D‖₂ ≤ 1` |
| A8. Newton–Schulz approximates `UVᵀ` | singular values of `zeroth_power_newton_schulz(G)` |
| A9. Muon is Shampoo without accumulation | `(GGᵀ)^{-1/4} G (GᵀG)^{-1/4} = UVᵀ` for full-rank `G` |
| A10. AdamW decay is decoupled and differs from Adam + L2 | `torch.optim.AdamW` matches `w ← w − η(m̂/(√v̂+ε) + λw)`; Adam with `weight_decay` gives a different step |

**Prediction for A8 (recorded to test my own wording):** I described Muon as
"setting all singular values to 1". The quintic coefficients in
`research/optimizers/muon.py` do not converge to 1; I predict the outputs'
singular values spread over roughly [0.5, 1.5] after 5 steps, while the
classic cubic iteration converges to 1. All other claims are predicted to
hold to float64 tolerance.

## Part B — curvature dynamics of the paper's source models

**Training.** Monolithic and compositional source models trained together
exactly as in `models/delta_observer.py` (Adam, α = 1/137 with warmup and
cosine decay, MSE, batch 32, per-epoch `randperm`, 200 epochs) but without
the observer, which only changes the random stream. Seeds 0–4.

**Measurements.** At epochs 0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 60, 100, 150
and 200, the exact full-batch Hessian of each source model's MSE loss
(float64, chunked forward-over-reverse HVPs, symmetrized), its full
eigendecomposition and the full-batch gradient. Recorded: top and bottom
eigenvalues, negative-eigenvalue count and mass, the fraction of the
gradient's squared norm in the top-k eigenspace (k = 1, 5, 10, 20), overlap
of top-5 eigenspaces between consecutive checkpoints, and whether the
Newton direction `−H⁻¹g` (eigenvalues with |λ| < 1e-8 dropped) is a descent
direction.

**B-EoS.** Separately, full-batch gradient descent on the monolithic model,
seeds 0–2. With λ₀ the initial top Hessian eigenvalue of seed 0, use step
sizes with 2/η = 4λ₀ and 2/η = 8λ₀, for 20,000 steps, measuring the top
eigenvalue (Lanczos on HVPs) every 100 steps.

### Predictions (recorded before running)

1. **Transient:** the monolithic top eigenvalue does *not* rise to a maximum
   at epoch ≤ 40 and then fall below half that maximum by epoch 200, in
   ≥ 4/5 seeds. (Consistent with addenda A and B, where the sources' own
   activations showed no transient clustering.)
2. **Indefiniteness:** the Hessian has negative eigenvalues at every
   checkpoint after epoch 0, in all seeds.
3. **Gradient subspace** (Gur-Ari et al., 2018): after epoch 5, at least half
   the gradient's squared norm lies in the top-10 eigenspace at most
   checkpoints. Uncertain: that result was for classification with
   cross-entropy and SGD, not MSE with Adam.
4. **Edge of stability** (Cohen et al., 2021): under full-batch GD the top
   eigenvalue rises and then stays within ±20% of 2/η (median over the last
   quarter of steps) for both step sizes in ≥ 2/3 seeds, and the loss keeps
   decreasing over the long run but not monotonically. Uncertain: this
   requires progressive sharpening by 4–8× on a tiny model.

## Interpretation limits

One toy task. The paper's models train with Adam, whose stability is governed
by preconditioned rather than raw sharpness, so EoS is tested separately with
plain gradient descent. Full Hessians are feasible only because the models are
small (9,285 and 1,480 parameters); nothing here shows the measurements
transfer to large models.
