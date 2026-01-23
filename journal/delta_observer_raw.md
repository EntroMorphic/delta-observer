# Raw Thoughts: Online Delta Observer

## Stream of Consciousness

So the whole point is that the Delta Observer should watch training as it happens, not analyze frozen corpses of trained models. The current implementation is like an archaeologist studying fossils instead of a naturalist watching living creatures.

What does "watching training" even mean? The models update their weights every batch. Activations change. Gradients flow. There's temporal structure - early training looks different from late training. A representation that doesn't exist at epoch 10 might crystallize by epoch 50.

The PCA baseline result bugs me. PCA on final activations matches Delta Observer performance. But PCA is static - it can only see the final state. If the Delta Observer also only sees the final state (which it does currently), then of course PCA can match it. They're both working with the same information.

But what information would an online observer have that PCA can't access?

- The trajectory of representation change
- When features emerge (timing)
- How fast different features stabilize
- Correlation between mono and comp learning curves
- Whether they learn the same things at the same time
- Gradient directions - where is learning heading?
- Transient representations that exist during training but disappear

Wait. Transient representations. That's interesting. What if there are meaningful intermediate states that get optimized away? The final model might not contain the semantic primitive - it might have *passed through* it during training.

How do you watch two models train at once? They're separate. Different architectures. Different number of parameters. But they're solving the same task. There must be something common in how they approach the problem, even if the final solutions differ.

The contrastive loss in the current setup tries to pull together representations of the same input. But that's spatial alignment (same input = close embedding). What about temporal alignment? Same input at different training stages - how does its representation evolve?

Actually... what IS the semantic primitive we're looking for? The paper says "carry_count" is linearly accessible. But is carry_count the semantic primitive, or is it a proxy for something deeper?

In binary addition, carries are the "hard" part. They propagate. A carry at bit 0 affects bit 1, which might generate another carry for bit 2. The number of carries is a measure of computational complexity for that particular addition.

Both models have to learn to handle carries. The monolithic model does it implicitly (all at once). The compositional model does it explicitly (bit by bit with carry signals). The "semantic primitive" might be: how does each model represent the difficulty of the computation?

Ugh, but that's still a final-state question. What would the temporal version be?

Temporal version: How does each model *learn* to handle carries? Do they learn easy cases first (0 carries) and hard cases last (4 carries)? Do they learn in the same order?

If they do learn in the same order, that's a shared learning curriculum. The online observer could detect this. PCA cannot.

What scares me about this problem:
- It might not work. The temporal structure might be noise.
- It's expensive. Training three models together is 3x the compute.
- It's complex. Synchronizing three optimizers, managing three loss functions.
- The "right" way to aggregate temporal information isn't obvious. LSTM? Transformer? Simple averaging?

What's my gut reaction to the implementation?
- Concurrent training seems right (all three models train together)
- But I'm not sure how the observer should "consume" the temporal information
- Maybe start simple: observer sees activations every batch, learns to map current state
- Then add complexity: observer also sees previous states, learns from trajectories

First instinct: Just make the observer train alongside the task models. Each batch, it sees current activations. Over time, it will have seen the full trajectory. Its own learned representation will implicitly encode what it learned from watching.

What's wrong with that instinct?
- The observer doesn't explicitly model time
- It might just learn to map the final state like before
- We need to force it to use temporal information

How to force temporal use?
- Give it past states as input (window of previous activations)
- Add a temporal prediction loss (predict next activation?)
- Make the latent space explicitly trajectory-aware

Actually, here's a radical idea: what if the Delta Observer doesn't output a single latent vector, but a trajectory in latent space? The "semantic primitive" isn't a point, it's a path.

That's probably too complicated for now. Start simple.

## Questions Arising

1. What information does online observation provide that post-hoc cannot access?
2. How should the observer aggregate temporal information?
3. Should we force the observer to use temporal info, or let it discover if it's useful?
4. What is the actual semantic primitive - carry_count, or something deeper?
5. Do both models learn in the same order (easy→hard)? Is that detectable?
6. How do we know if the online version is "better" than post-hoc?
7. Is the trajectory more informative than the final state?
8. What's the minimal change to enable online observation?
9. How do we handle different training speeds (if mono converges faster than comp)?
10. Should the observer also see gradients, not just activations?

## First Instincts

1. Start with concurrent training - all three models train together
2. Observer sees activations each batch, no explicit temporal modeling yet
3. Add a "checkpoint buffer" - save activations every N epochs
4. After basic version works, add temporal window to observer input
5. Measure: does online R² > PCA R²? Does it capture something PCA misses?
6. The semantic primitive might be in learning dynamics, not final state
7. Simplest temporal metric: correlation of learning curves between models
