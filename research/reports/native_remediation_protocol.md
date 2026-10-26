# Coverage and serialization remediation pilot

Frozen before building or training the remediation adapter. Goal: improve exact
success toward 100/100 while measuring—not assuming—content retention.

## Fixed intervention

Start from untouched pinned LFM, not an inspected adapter. Keep rank-8/alpha-16
attention/conv LoRA, frozen base, FP32 AdamW lr 1e-4, decay .01, clip 1, batch 1,
seed 0, and one fixed epoch: **640 updates**. No test-driven checkpoint choice.

Use 128 deterministic examples per family: JSON copying, signed sorting,
multiply/subtract arithmetic, transitive ordering, and Spanish number words.
JSON copying must contain 16 different values with eight exposures each and
compact/spaced/pretty bare JSON. All targets are algorithmically specified;
no model-generated answers, truncation, or label leakage through prompt masks.
Record exact budgets after tokenization, before any training.

Freeze a new 100-prompt suite (20 per family) and its hash before training.
Check exact and word-5-gram overlap against training; also exclude inspected
development and former 100-task prompts from training. Shared task families and
Spanish mappings are intentional: this is prompt generalization, not novel
knowledge or contamination-free reasoning evaluation.

## Fixed comparisons and scores

Evaluate untouched, old repeated adapter, and new remediation adapter on the
new test. Evaluate the remediation adapter on the exposed old 100 tasks for
diagnosis only. Use unchanged generation settings and strict/content/format
validators. Report every category, paired exact and content losses, runtime,
and actual budgets. Existing adapters have smaller budgets; do not attribute
differences solely to coverage, serialization, or any observer mechanism.

Preliminary improvement requires new-test strict success greater than both
untouched and old repeated, with no lost baseline confirmed content successes.
If that retention gate fails, report failure despite an aggregate gain.
100/100 is the target, never an assumed score or reason to loosen validators.

This is a one-seed bundled remediation pilot. Passing does not establish a
causal component advantage, general LLM capability, or observer-specific value.
