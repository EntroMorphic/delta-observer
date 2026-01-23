# Nodes of Interest: Online Delta Observer

## Node 1: The Fossil vs Living Problem
The current Delta Observer studies "fossils" (frozen trained models) rather than "living creatures" (models during training). This fundamentally limits what information it can access.

**Why it matters:** This is the core problem. The original intent was live observation. Post-hoc analysis was a deviation.

---

## Node 2: PCA Equivalence as Information Bound
PCA on final activations matches Delta Observer performance (R²≈0.95). Both see the same information (final state), so they're bounded by the same ceiling.

**Why it matters:** If online observation accesses *more* information than final state, it could exceed this ceiling. If it doesn't, the online version will also match PCA.

---

## Node 3: Temporal Information Inventory
What temporal information exists that PCA cannot access?
- Trajectory of representation change
- Feature emergence timing
- Learning curve shapes
- Cross-model synchronization
- Gradient directions
- Transient representations

**Why it matters:** This is the enumeration of potential advantage. If none of these contain signal, online observation won't help.

---

## Node 4: Transient Representations
Intermediate training states might contain meaningful structure that gets optimized away. The final model passes *through* the semantic primitive; it doesn't *contain* it.

**Why it matters:** This is the strongest argument for online observation. If true, post-hoc analysis is fundamentally limited.

**Tension with Node 2:** If transients matter, why does PCA on final state still achieve R²=0.95? Is 0.95 not the ceiling?

---

## Node 5: Learning Curriculum Hypothesis
Both models might learn in the same order: easy cases (0 carries) before hard cases (4 carries). This shared curriculum would be a temporal signature.

**Why it matters:** If detectable, this proves temporal structure exists. PCA cannot detect learning order.

**How to test:** Track per-carry-count accuracy during training for both models. Do the curves correlate?

---

## Node 6: What IS the Semantic Primitive?
Is `carry_count` the primitive, or a proxy? Carries represent computational difficulty. The primitive might be "how does each model represent difficulty?"

**Why it matters:** If we're measuring the wrong thing, we'll draw wrong conclusions.

**Tension with Node 4:** If carry_count is just a proxy, why does it achieve R²=0.95? Proxies usually show lower signal.

---

## Node 7: Temporal Aggregation Problem
How should the observer consume temporal information?
- Option A: See current batch only (implicit temporal via weight updates)
- Option B: Window of recent activations (explicit short-term)
- Option C: Full trajectory (explicit long-term)
- Option D: Learned attention over time steps

**Why it matters:** This is a key design decision. Wrong choice could waste the temporal information.

---

## Node 8: Force vs Discover
Should we architecturally force the observer to use temporal info, or let it discover if temporal info is useful?

**Arguments for force:** If temporal info is subtle, the model might ignore it in favor of easier final-state features.

**Arguments for discover:** If we force something that isn't there, we add noise.

---

## Node 9: Minimal Viable Online Observer
What's the smallest change that enables online observation?

Current: `Train A → Train B → Extract → Train Observer`
Minimal: `For each batch: update A, update B, update Observer(A.act, B.act)`

**Why it matters:** Complexity has costs. Start simple.

---

## Node 10: Synchronization Problem
What if mono and comp train at different speeds? Mono might converge at epoch 50, comp at epoch 150. The observer sees "done" vs "still learning."

**Why it matters:** Async training could confuse the observer.

**Potential solution:** Normalize by training progress, not epoch number.

---

## Node 11: Trajectory as Latent Space
Radical idea: The semantic primitive isn't a point in latent space, but a *path*. The observer outputs a trajectory, not a vector.

**Why it matters:** This reframes the entire problem.

**Tension:** Much more complex. Probably premature.

---

## Node 12: Gradient Information
The observer could see not just activations, but gradients - the *direction* of learning, not just current state.

**Why it matters:** Gradients are the most direct signal of what the model is learning.

**Tension with Node 9:** This is not minimal. Adds complexity.

---

## Node 13: Success Metric
How do we know if online observation is "better"?
- R² > PCA baseline (0.9482)
- Captures something PCA cannot (new metric)
- Lower Silhouette (even less clustering, same accessibility)
- Generalizes to unseen training runs

**Why it matters:** Without a clear success metric, we can't evaluate.

---

## Node 14: The Detach Question
Should observer gradients flow back into task models?

`delta_observer(mono_act.detach(), comp_act.detach())` → Observer learns from, but doesn't influence, task learning.

`delta_observer(mono_act, comp_act)` → Observer can shape what the task models learn.

**Why it matters:** Huge architectural choice.

**First instinct:** Detach. Keep task learning pure. Observer should observe, not intervene.

---

## Node 15: Compute Cost
Training three models simultaneously is expensive. 3x forward passes, potentially 3x backward passes per batch.

**Why it matters:** If the experiment is too expensive to iterate, we'll make fewer attempts and learn slower.

**Mitigation:** Observer could update less frequently (every 10 batches).
