# Online Delta Observer Design

**Date:** 2026-01-23
**Problem:** Current Delta Observer trains post-hoc on frozen activations. Original intent was to observe training as it occurs.

---

## The Core Issue

The current implementation:
```
Train Mono (200 epochs) → Train Comp (200 epochs) → Extract activations → Train Delta Observer
```

The original intent:
```
Train Mono + Comp + Delta Observer simultaneously, with Delta Observer watching both
```

---

## Why This Matters for the PCA Baseline

PCA on final activations achieves R²=0.9482 ≈ Delta Observer R²=0.9505.

But PCA **cannot** capture:
1. **Temporal dynamics** - How representations change over training
2. **Learning trajectories** - The path from random init to final state
3. **Convergence patterns** - When/how semantic structure crystallizes
4. **Cross-model synchronization** - Do both models learn carry_count at similar rates?

An online observer could learn from these dynamics, giving it information PCA cannot access.

---

## Proposed Implementation: Approach 1 - Concurrent Training

### Architecture
```
                    ┌─────────────────┐
                    │  Shared Input   │
                    │   (4-bit add)   │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
    ┌─────────────────┐            ┌─────────────────┐
    │   Monolithic    │            │  Compositional  │
    │     Model       │            │     Model       │
    └────────┬────────┘            └────────┬────────┘
             │                              │
             │  activations (each batch)    │
             └──────────────┬───────────────┘
                            ▼
                  ┌─────────────────┐
                  │ Delta Observer  │
                  │   (online)      │
                  └─────────────────┘
```

### Training Loop (Pseudocode)
```python
for epoch in range(200):
    for batch in dataloader:
        # 1. Forward pass through both task models
        mono_out, mono_act = mono_model.forward_with_activations(batch)
        comp_out, comp_act = comp_model.forward_with_activations(batch)

        # 2. Task losses (both models learn their task)
        mono_loss = task_loss(mono_out, targets)
        comp_loss = task_loss(comp_out, targets)

        # 3. Delta Observer forward pass
        delta_out = delta_observer(mono_act.detach(), comp_act.detach())
        # Note: detach() so observer doesn't affect task learning

        # 4. Observer losses
        obs_loss = reconstruction_loss + contrastive_loss + classification_loss

        # 5. Update all three models
        mono_optimizer.step()  # Task gradient
        comp_optimizer.step()  # Task gradient
        delta_optimizer.step() # Observer gradient
```

### Key Design Decisions

1. **Detach activations** - Observer learns from activations but doesn't backprop into task models
2. **Same batch** - Observer sees corresponding activations for same inputs
3. **Simultaneous updates** - All three models train together

---

## Proposed Implementation: Approach 2 - Trajectory Learning

### Idea
Save activation snapshots at multiple training stages, then train Delta Observer on the full trajectory.

```python
# During task model training
checkpoints = []
for epoch in [0, 10, 20, 50, 100, 150, 200]:
    mono_acts = extract_all_activations(mono_model)
    comp_acts = extract_all_activations(comp_model)
    checkpoints.append((epoch, mono_acts, comp_acts))

# Delta Observer training
# Input: sequence of (mono_act, comp_act) pairs over time
# Architecture: Transformer or LSTM over time dimension
```

### Architecture for Trajectory Learning
```
Time steps: [t0, t10, t20, t50, t100, t150, t200]

For each input x:
  mono_trajectory = [mono_act(x, t) for t in time_steps]  # (7, 64)
  comp_trajectory = [comp_act(x, t) for t in time_steps]  # (7, 32)

  # Encode trajectories
  mono_encoded = trajectory_encoder(mono_trajectory)  # (7, 32)
  comp_encoded = trajectory_encoder(comp_trajectory)  # (7, 32)

  # Cross-attention between trajectories
  aligned = cross_attention(mono_encoded, comp_encoded)

  # Pool over time for final latent
  latent = temporal_pool(aligned)  # (16,)
```

---

## Proposed Implementation: Approach 3 - Gradient-Aware Observer

### Idea
Observer sees not just activations, but also gradients (how activations are changing).

```python
# After backward pass but before optimizer step
mono_act_grad = mono_model.hidden.weight.grad
comp_act_grad = comp_model.bit0_adder.hidden.weight.grad

# Delta Observer input includes gradient information
delta_input = concat(mono_act, comp_act, mono_act_grad, comp_act_grad)
```

This captures the *direction* of learning, not just current state.

---

## Evaluation: How to Test if Online Beats PCA

### Hypothesis
If online observation captures meaningful temporal structure, then:
1. Online Delta Observer R² > PCA R² (currently ~equal at 0.95)
2. OR: Online observer captures structure PCA misses entirely

### Test Protocol
1. Train with Approach 1 (concurrent training)
2. Extract latent representations
3. Compare:
   - Linear probe R² for carry_count
   - Silhouette scores
   - **New:** Temporal consistency - do nearby training epochs map to nearby latents?
   - **New:** Cross-model alignment over time

### New Metrics for Temporal Structure
```python
# Metric: Representation Drift Correlation
# Do mono and comp representations drift in similar directions?
for t in range(epochs):
    mono_drift = mono_act[t+1] - mono_act[t]
    comp_drift = comp_act[t+1] - comp_act[t]
    correlation = pearsonr(mono_drift, comp_drift)

# If correlated, both models are learning similar transformations
# This is information PCA cannot capture
```

---

## Implementation Priority

| Approach | Complexity | Potential Impact | Recommendation |
|----------|------------|------------------|----------------|
| 1: Concurrent | Medium | High | **Start here** |
| 2: Trajectory | High | Medium | Second priority |
| 3: Gradient-aware | Medium | Unknown | Experimental |

---

## Next Steps

1. Modify `train_4bit_monolithic.py` and `train_4bit_compositional.py` to expose activations
2. Create `train_online_delta_observer.py` that trains all three concurrently
3. Run falsification tests on the online version
4. Compare R² against PCA baseline

---

## Expected Outcome

If the original intuition is correct:
- Online Delta Observer should capture temporal structure
- This should manifest as improved R² or novel capabilities PCA lacks
- The "semantic primitive" may be in the learning dynamics, not just final state
