# Semantic-transfer measurement: ordinary CE, controlled exposure budgets

Frozen before data generation/training. This tests the next LMM candidate
separately from worked replay or a retention objective.

Fresh pinned native LFM per arm; rank-8/alpha-16 attention/conv LoRA; FP32,
seed 0, AdamW lr 1e-4/decay .01/clip 1, token-weighted gradient accumulation.
128 primary operand triples; four examples per fact/update, repeated across four passes.
Each optimizer update has four examples and eight genuine supervised positions.
Fixed endpoint: 512 updates, 2,048 sequence exposures, 4,096 supervised tokens.

- `repeat`: four copies of the same subtract-question wording.
- `paraphrase`: four meaning-preserving subtract-question wordings.
- `contrast`: two subtract and two add wordings on identical numeric tokens.

All use ordinary supervised CE, no novel contrastive loss. Contrast changes
operation coverage as well as wording count; this does not isolate a benefit
of a contrastive objective. Input lengths/compute differ and must be reported.
Corresponding numerical fact exposure and update/token budgets remain fixed.

Freeze 100 tasks before training: 20 seen facts/unseen template, 20 unseen facts/
seen template, 20 unseen facts/unseen template, 20 add counterfactuals paired to
the preceding subtract cases, and 20 retention tasks. New-fact arithmetic
splits are disjoint from training triples. Paired test cases intentionally share
facts. Verify the exact template rendering and normalized prompt exclusions;
report lexical, semantic, and upstream-contamination limits.

Evaluate untouched and all three fixed endpoints with identical greedy settings.
Measure strict and parsed-content counts, per-split gains/losses, both-correct
paraphrase pairs, and both-correct add/subtract pairs. The primary transfer cell
is unseen-fact/unseen-template subtraction; addition is reported separately,
not allowed to mask subtraction regressions. All scoring rules stay fixed.

A candidate advances only if it beats repeat on the primary transfer cell,
preserves every repeat-arm strict subtraction success on the new-fact splits,
and loses no untouched confirmed content or exact successes in the full test.
Report failed gates; do not scale or select an intermediate checkpoint to hide
them. One seed/five synthetic task families do not establish broad intelligence
or observer advantage. 100/100 is a target, not an assumption.
