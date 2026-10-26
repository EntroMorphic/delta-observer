# Native-model 100-task evaluation protocol

Frozen before any model generation on this task set. This is a post-training
holdout evaluation of already-existing checkpoints, not a benchmark created
before the research program or a sealed confirmatory study.

## Fixed conditions

- 100 new prompts, 20 per family: JSON records, signed sorting, arithmetic,
  transitive ordering, and Spanish number-word translation.
- Compare untouched pinned LFM, completed repeated-data LoRA, and completed
  unique-data LoRA. No training, selection, or intermediate-checkpoint evaluation.
- Freeze task file SHA-256 and check normalized exact/word-5-gram overlap against
  both training arms, prior development v2, and historical sentinels.
- Same greedy generation, 192-token budget, native chat formatting, FP16 base
  inference and FP32 LoRA factors, as in development evaluation.

## Scores

Primary: exact end-to-end task success, one point per task, reported out of 100
and separately out of 20 per category. No category weighting changes.

Content: exact successes plus the already-defined narrow parsers for whole JSON
inside optional fences and unambiguous boxed integers. Tasks without a separate
parser use the strict result as confirmed content success; failures remain
unresolved for semantic content, not automatically content-incorrect. Report
confirmed successes, parser-determined failures, and unresolved counts with a
fixed denominator of 100. Do not divide by only parseable answers to inflate
accuracy. These parsers are not independent human judgment.

Format: validate JSON syntax without requiring correct values; integers must
match the whole-response signed-decimal form; event IDs must be exactly one of
the three identifiers in the question; Spanish responses must be a single
lowercase alphabetic word (including accents). Format success does not imply
content correctness.

Paired gains/regressions: compare every adapter to untouched task IDs; report
new exact successes, lost exact successes, and net change per category. Any
category regression must remain visible regardless of aggregate improvement.
Record hashes, raw token IDs/text, EOS/truncation, and measured runtime.

## Interpretation

This answers performance on these 100 tasks, not a claim of 100% general
capability. Task families are correlated and training has one seed; no
independent-sample significance or broad observer benefit will be claimed.
Open-ended reasoning, factual fidelity, abstention, code execution, and wider
multilingual performance are outside this machine-scored suite.
