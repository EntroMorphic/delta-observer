# Reflections: Online Delta Observer

## Core Tension: Node 2 vs Node 4

The central puzzle emerges from two apparently contradictory observations:

**Node 2:** PCA on final state achieves R²=0.9482, nearly matching Delta Observer (0.9505). This suggests final state contains almost all the information.

**Node 4:** If transient representations matter, final state should be information-poor. But it isn't.

**Resolution:** These aren't contradictory. The question isn't "does final state have information?" (it does) but "does temporal information have *additional* information beyond final state?"

The R²≈0.95 might be a ceiling for what *any* method can extract from activations→carry_count. Or it might be that both PCA and post-hoc Delta Observer are hitting the same (suboptimal) ceiling because they're both limited to final state.

**Key insight:** We need a metric that tests *temporal-specific* information, not just overall R².

---

## The Curriculum Hypothesis (Node 5) is Testable

Before building anything, we can test whether temporal structure exists:

1. Train mono and comp models as normal
2. At each epoch, measure per-carry-count accuracy
3. Plot learning curves: accuracy(carry_count, epoch)
4. Check: do both models learn 0-carry cases before 4-carry cases?
5. Check: do their learning curves correlate?

If yes → temporal structure exists, online observation might capture it.
If no → temporal structure might be noise, proceed with caution.

**This test is cheap. Do it first.**

---

## The Minimal Change (Node 9) is the Right Start

Nodes 7, 8, 11, 12 all propose increasingly complex temporal handling. But Node 9 suggests starting simple:

**Minimal online observer:** Train alongside task models, see current-batch activations only.

This is wise because:
1. If minimal version beats PCA, we've proven online helps
2. If minimal version equals PCA, we can add complexity
3. If minimal version is worse than PCA, something is wrong with our assumptions

**Implementation insight:** The minimal observer has no explicit temporal modeling. Its only "temporal" aspect is that its weights evolve while watching evolving task models. The observer's learned weights implicitly encode what it learned from the trajectory.

This is subtle but important: even without explicit temporal input, the online observer's final weights are shaped by the full trajectory it witnessed. Post-hoc observer's weights are shaped only by final-state data.

---

## The Detach Question (Node 14) Has a Clear Answer

Should observer gradients affect task models?

**No. Detach.**

Reasoning:
1. We want to study how models learn naturally, not influence them
2. Observer-influenced learning would be a different experiment
3. If observer shapes task learning, we can't separate "what did models learn" from "what did observer make them learn"
4. Scientific cleanliness: observer is measurement apparatus, not intervention

This also simplifies implementation: observer has its own optimizer, independent of task optimizers.

---

## What "Better" Means (Node 13)

How do we know online observation succeeds?

**Primary metric:** R² on held-out test set > PCA baseline (0.9482)

But R² might be saturated. Secondary metrics:

**Temporal-specific test:** Take two training runs with different random seeds. Same final accuracy, different training trajectories. Can the observer distinguish which run produced which activation?

If yes → it learned trajectory-specific information.
If no → it only learned final-state information.

**Cross-run generalization:** Train observer on run A, test on run B. If it generalizes, it learned something fundamental about learning, not memorized run A's specifics.

---

## The Synchronization Problem (Node 10) Might Be a Feature

What if mono converges faster than comp?

Initially I thought: "This confuses the observer."

But actually: **Convergence timing is information.**

If mono learns carry_count=0 cases at epoch 20, and comp learns them at epoch 40, that's a signal. The observer could learn that comp representations at epoch 40 correspond to mono representations at epoch 20.

**Don't normalize away the async. Let the observer see it.**

---

## Transient Representations (Node 4) Require Active Looking

The observer won't automatically capture transients. If a representation exists at epoch 30 but disappears by epoch 100, and we only analyze the final observer, we've lost it.

**Implication:** We might need to analyze the observer at multiple checkpoints during training, not just at the end.

Or: Add explicit temporal buffer. Save activations from past N epochs. Feed to observer as sequence.

**Decision:** Start without temporal buffer. If R² doesn't improve, add buffer in v2.

---

## What IS the Semantic Primitive? (Node 6)

I've been assuming carry_count is the target. But the paper's claim is about accessibility without clustering. Carry_count is just one test.

**Reframe:** The semantic primitive is "the information that both models must represent to solve the task."

Both models must:
- Know the input bits
- Handle carry propagation
- Produce correct output bits

The *way* they represent this differs (mono: one network, comp: four sub-networks). The semantic primitive is what's *common* despite the architectural difference.

Carry_count is a good proxy because it measures the "hardest" aspect: carry propagation. If this is linearly accessible, simpler aspects probably are too.

**Decision:** Keep carry_count as primary metric. But also test: input_sum, output_value, max_bit_position. See which are accessible.

---

## The Laundry Method Applied

From LMM.md: "Partition first. Search within. The delta is where mistakes hide."

Partitioning the problem:

**Bucket 1: Does temporal information exist?** (Test with curriculum analysis)
**Bucket 2: Can minimal observer access it?** (Test with R² comparison)
**Bucket 3: Does explicit temporal modeling help?** (Test with buffer vs no-buffer)
**Bucket 4: What is the right success metric?** (Test with cross-run generalization)

**The delta:** The boundary between "observer learned from trajectory" and "observer only learned final state." This is where the claim lives or dies.

---

## What Would This Look Like If It Were Easy?

If easy: Train all three models. Extract latents. R² > PCA. Done.

The complexity comes from:
1. Not knowing if temporal info exists
2. Not knowing how to represent temporal info
3. Not knowing how to measure success

**Make it easy:** Test temporal existence first. Start minimal. Measure simply.

---

## Final Understanding

The problem has four layers:

**Layer 1 (prerequisite):** Does temporal structure exist in training? (Test: curriculum analysis)

**Layer 2 (implementation):** Can we build an observer that trains online? (Yes, straightforward)

**Layer 3 (evaluation):** Does online observer beat post-hoc? (Test: R² comparison)

**Layer 4 (insight):** What did the observer learn that PCA couldn't? (Test: cross-run generalization, trajectory discrimination)

**Order of operations:**
1. Validate prerequisite (cheap curriculum test)
2. Build minimal online observer
3. Compare to PCA
4. If better: analyze what it learned
5. If equal: add temporal buffer, repeat
6. If worse: re-examine assumptions

---

## Remaining Questions

1. Is R² of 0.95 a true ceiling, or an artifact of final-state limitation?
2. Will the curriculum test show correlated learning curves?
3. What's the simplest temporal buffer that adds value?
4. Can we visualize "what the observer learned from trajectory"?
