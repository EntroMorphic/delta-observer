# Native LoRA P0 engineering and exploratory pilot

## Scope and lock

Freeze before native adaptation beyond the one-example smoke test. This is an
exploratory pipeline/compliance pilot, not the broad conversational-diversity
study or a confirmatory capability result. No final test set is claimed sealed.

## Model and update

- Native `LiquidAI/LFM2.5-350M`, revision
  `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`.
- Freeze every original parameter; LoRA rank 8, alpha 16 on attention q/k/v/out
  and short-convolution in/out projections. No FFN, embedding, or head updates.
- FP32 on the measured local device; AdamW lr `1e-4`, decay `0.01`, clipping 1.
- Batch size 1; one epoch of 128 updates, seed 0, fixed deterministic row order.
- Full responses plus EOS; prompt labels ignored; no truncation or padding.
- Initialization and budget identical in both arms. Original model is retained.

## Data contract

Use `build_native_pilot_data` generated deterministic inventory-extraction
targets, not self-generated teacher answers. This tests lexical/numeric variant
coverage within one template, not broad semantic diversity.

| Arm | Exposures | Distinct prompts | Input tokens | Supervised tokens |
|---|---:|---:|---:|---:|
| Unique | 128 | 128 | 9,728 | 1,408 |
| Repeated | 128 | 16 | 9,728 | 1,408 |

Every corresponding row has identical input and supervised lengths. Both use
the same shuffle seed. Verify manifest/file hashes and re-tokenize every row
before training. Existing development v2 and historical sentinel registries
are excluded by normalized equality and word-5-gram Jaccard >= 0.8. This lexical
screen is not semantic or pretraining decontamination.

## Engineering gates before pilot

1. Zero-init native logits exactly unchanged.
2. Finite forward/backward and nonzero adapter update.
3. Frozen-weight SHA-256 unchanged and no frozen gradients.
4. Adapter save/reload logits exactly equal.
5. Four updates uninterrupted versus two-plus-two resumed: identical adapters
   and row order. Engineering-only checkpoints are discarded from pilot starts.

## Evaluation and advancement

Evaluate the unchanged model and both adapters on the fixed 44-task development
v2 suite, same greedy settings and FP16 evaluation dtype as the baseline.
No intermediate checkpoint selection. Report strict and separately parsed
content results; never count pending manual reviews as successes or failures.

Preliminary positive signal requires at least 4/14 strict format-following
passes, zero loss of any baseline automatic success (strict or separately
parsed content), and no nonfinite training or changed frozen weights. Report
both arms whether they pass or fail. Do not choose a model using the manual
tasks without independent scoring. Runtime and memory are measured, not assumed
equal from step counts.

Even a passing result authorizes only a larger preregistered study. Multi-seed
replication, a genuinely held-out final evaluation, blind review, and isolated
code execution remain required before a broad LLM-improvement claim.
