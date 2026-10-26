# Exposure-matched specialist ablation

Frozen before specialist training. Diagnostic comparison, not a fresh benchmark.

Question: do other-task updates account for the bundled v2 model's arithmetic
fit failures and ordering transfer failures?

Two adapters start independently from untouched pinned LFM using exactly the v2
LoRA architecture, initialization seed, AdamW settings, clipping, and precision.

- Arithmetic specialist: v2's 512 arithmetic examples, exactly one exposure each.
- Ordering specialist: v2's 320 ordering examples, exactly one exposure each.

Preserve the exact within-task row sequence from the completed v2 history. The
builder verifies source hashes and a complete permutation of source exposures.
No new labels, repetition, learning-rate changes, or rank changes.

Remove all other-task updates. Total steps, optimizer moment histories, and
compute differ intentionally: this ablation tests removing other-task training,
not a compute-matched measure of gradient interference. A specialist improvement
supports harmful interaction but cannot by itself distinguish gradient conflict,
optimizer-state effects, or total-training-time effects.

Measurements at the one fixed endpoint:

1. Exact in-sample fit using the same generation settings as the v2 fit audit.
2. Full 100-task exposed v2 diagnostic suite, showing all categories/retention.
3. Per-category and per-mode comparison to v2, task-matched gains/regressions.
4. Data budgets, measured runtime, frozen-weight identity, checkpoint provenance.

Primary diagnostic tests: arithmetic compound training fit greater than 81/128;
ordering new-wording accuracy greater than 10/20. No checkpoint selection or
general-purpose deployment approval follows from these results. If fit does not
improve, do not declare interference the sole cause. Do not claim a capacity
floor without subsequent isolated fit experiments.
