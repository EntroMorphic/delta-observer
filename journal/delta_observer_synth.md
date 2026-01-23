# Synthesis: Online Delta Observer

## Architecture

### Overview

```
Phase 0: Curriculum Test (validates premise)
    ↓
Phase 1: Minimal Online Observer (establishes baseline)
    ↓
Phase 2: Temporal Buffer Observer (if Phase 1 ≈ PCA)
    ↓
Phase 3: Deep Analysis (what did it learn?)
```

---

## Phase 0: Curriculum Test

**Purpose:** Validate that temporal structure exists before building anything.

**Implementation:**

```python
# During normal training of mono/comp models, track:
per_epoch_results = {
    'epoch': [],
    'mono_acc_by_carry': [],  # [acc_0, acc_1, acc_2, acc_3, acc_4]
    'comp_acc_by_carry': [],
}

# After training, analyze:
# 1. Do both models learn 0-carry before 4-carry?
# 2. Do learning curves correlate? (Pearson r between mono and comp curves)
```

**Success criterion:** Correlation > 0.7 between mono and comp learning curves per carry_count.

**If test fails:** Temporal structure may not exist. Proceed with caution or reconsider approach.

**Estimated effort:** Modify existing training scripts to log per-carry accuracy. <1 hour.

---

## Phase 1: Minimal Online Observer

**Purpose:** Establish whether online observation improves over PCA baseline.

### Training Loop

```python
def train_online(mono_model, comp_model, delta_observer, dataloader, epochs=200):
    mono_opt = Adam(mono_model.parameters(), lr=0.001)
    comp_opt = Adam(comp_model.parameters(), lr=0.001)
    obs_opt = Adam(delta_observer.parameters(), lr=0.001)

    for epoch in range(epochs):
        for batch in dataloader:
            inputs, targets = batch

            # --- Task Model Forward ---
            mono_act = mono_model.forward_with_activations(inputs)
            comp_act = comp_model.forward_with_activations(inputs)

            mono_out = mono_model(inputs)
            comp_out = comp_model(inputs)

            # --- Task Losses ---
            mono_loss = F.mse_loss(mono_out, targets)
            comp_loss = F.mse_loss(comp_out, targets)

            # --- Observer Forward (detached activations) ---
            obs_out = delta_observer(mono_act.detach(), comp_act.detach())

            # --- Observer Losses ---
            obs_loss = compute_observer_loss(obs_out, batch)

            # --- Backward (independent) ---
            mono_opt.zero_grad()
            mono_loss.backward()
            mono_opt.step()

            comp_opt.zero_grad()
            comp_loss.backward()
            comp_opt.step()

            obs_opt.zero_grad()
            obs_loss.backward()
            obs_opt.step()
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Detach activations | Yes | Observer observes, doesn't intervene |
| Shared batch | Yes | Same inputs for fair comparison |
| Observer update frequency | Every batch | Maximizes temporal exposure |
| Observer architecture | Same as current | Isolate online vs post-hoc |

### Required Modifications

1. **Monolithic model:** Add `forward_with_activations()` method
2. **Compositional model:** Add `forward_with_activations()` method
3. **New file:** `train_online_delta_observer.py`

### Evaluation

After training:
1. Extract latent representations from observer
2. Compute R² for carry_count (linear probe)
3. Compute Silhouette score
4. Compare to baselines:
   - PCA baseline: R² = 0.9482
   - Post-hoc observer: R² = 0.9505

**Success criterion:** Online R² > 0.96 (meaningfully above PCA)

---

## Phase 2: Temporal Buffer Observer (Conditional)

**Trigger:** Phase 1 R² ≤ PCA baseline

**Purpose:** Explicitly provide temporal information to observer.

### Architecture Change

```python
class TemporalDeltaObserver(nn.Module):
    def __init__(self, buffer_size=10):
        # ... existing encoder setup ...

        # Temporal aggregation
        self.temporal_attention = nn.MultiheadAttention(embed_dim=32, num_heads=4)
        self.buffer_size = buffer_size

    def forward(self, mono_act, comp_act, mono_buffer, comp_buffer):
        """
        mono_act: current activations (batch, 64)
        mono_buffer: past activations (batch, buffer_size, 64)
        """
        # Encode current
        current_enc = self.encode_current(mono_act, comp_act)

        # Encode history
        history_enc = self.encode_history(mono_buffer, comp_buffer)

        # Attend over time
        attended, _ = self.temporal_attention(
            query=current_enc.unsqueeze(0),
            key=history_enc.transpose(0, 1),
            value=history_enc.transpose(0, 1)
        )

        # Combine
        combined = torch.cat([current_enc, attended.squeeze(0)], dim=-1)
        latent = self.shared_encoder(combined)

        return latent
```

### Buffer Management

```python
class ActivationBuffer:
    def __init__(self, size=10):
        self.size = size
        self.mono_buffer = deque(maxlen=size)
        self.comp_buffer = deque(maxlen=size)

    def push(self, mono_act, comp_act):
        self.mono_buffer.append(mono_act.detach().clone())
        self.comp_buffer.append(comp_act.detach().clone())

    def get(self):
        if len(self.mono_buffer) < self.size:
            # Pad with zeros if buffer not full
            ...
        return stack(self.mono_buffer), stack(self.comp_buffer)
```

---

## Phase 3: Deep Analysis

**Purpose:** Understand what the observer learned.

### Test 1: Trajectory Discrimination

```python
# Train two complete runs (different seeds)
run_a = train_online(seed=42)
run_b = train_online(seed=123)

# Take final activations from both
acts_a = extract_activations(run_a)
acts_b = extract_activations(run_b)

# Use observer trained on run_a
latents_a = observer_a.encode(acts_a)
latents_b = observer_a.encode(acts_b)

# Can we tell them apart?
# Train classifier: latent → which_run
# If accuracy > 50%, observer learned run-specific temporal info
```

### Test 2: Cross-Run Generalization

```python
# Train observer on run_a
# Evaluate R² on run_b (never seen during training)
#
# Post-hoc observer: trained on run_a final state, test on run_b final state
# Online observer: trained on run_a trajectory, test on run_b final state
#
# If online generalizes better → learned more fundamental structure
```

### Test 3: Temporal Ablation

```python
# Compare observers trained with different temporal exposure:
# - Every batch (full temporal)
# - Every 10 batches
# - Every epoch
# - First half only
# - Second half only
#
# Does more temporal exposure help? When during training is most valuable?
```

---

## Implementation Spec

### New Files

| File | Purpose |
|------|---------|
| `analysis/curriculum_test.py` | Phase 0: Validate temporal structure |
| `models/train_online_observer.py` | Phase 1: Main training script |
| `models/temporal_observer.py` | Phase 2: Temporal buffer architecture |
| `analysis/trajectory_analysis.py` | Phase 3: Deep analysis |

### Modified Files

| File | Change |
|------|--------|
| `models/train_4bit_monolithic.py` | Add `forward_with_activations()` |
| `models/train_4bit_compositional.py` | Add `forward_with_activations()` |

### Execution Order

```bash
# Phase 0
python analysis/curriculum_test.py

# Phase 1 (if Phase 0 shows correlation)
python models/train_online_observer.py
python analysis/falsification_tests.py  # Modified to use online observer

# Phase 2 (if Phase 1 ≈ PCA)
python models/temporal_observer.py

# Phase 3 (if any phase shows improvement)
python analysis/trajectory_analysis.py
```

---

## Success Criteria

### Primary

- [ ] Phase 0: Curriculum correlation > 0.7
- [ ] Phase 1: Online R² > 0.96 (beats PCA by meaningful margin)

### Secondary

- [ ] Cross-run generalization: Online > Post-hoc
- [ ] Trajectory discrimination: Accuracy > 60%

### Stretch

- [ ] Identify *what* temporal info the observer uses
- [ ] Visualize learning trajectory in observer's latent space

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Temporal info doesn't exist | Phase 0 tests this cheaply first |
| 3x compute cost | Observer can update every N batches |
| Complexity explosion | Start minimal, add features only if needed |
| No improvement over PCA | This is a valid finding; document and publish |

---

## The Clean Cut

The wood will cut itself if:

1. We validate temporal structure exists (Phase 0)
2. We implement minimal online observation (Phase 1)
3. We measure against clear baselines (PCA, post-hoc)
4. We only add complexity (Phase 2) if minimal version falls short
5. We accept null result as information if nothing helps

The answer to "does online beat PCA?" is valuable regardless of which way it goes.
