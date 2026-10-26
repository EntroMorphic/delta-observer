# SYNTHESIZE: arithmetic fit/capacity ladder

LMM commit: 55719c0a2308d1a3f29ca656ed7b6287be33bca2.
Frozen before training this ladder.

## Fixed experiment

Use the verified 512-row arithmetic-specialist dataset, same within-task source
order repeated each epoch, untouched pinned base, seed 0, same attention/conv
LoRA targets, FP32 AdamW lr 1e-4/decay .01/clip 1, batch 1.

Two fresh runs: rank 8/alpha 16 and rank 32/alpha 64, scale alpha/rank fixed at 2.
Four passes each (2,048 updates), immutable snapshots at 512/1,024/1,536/2,048.
No test-based checkpoint selection or changes to data. Different rank changes
parameter count and random initialization shape; report this limitation.

Measure final-only compound fit at each boundary (128 rows). At fixed endpoints
measure full primitive and ordering fit as applicable and all 100 exposed v2
diagnostic tasks. Keep actual input/supervised exposures and frozen hashes.

## Interpretation gates

- Fit improvement: compound correct count greater than prior specialist 77/128.
- Complete local fit: 128/128, reported as in-sample only.
- Compare rank arms at equal passes; compare duration within rank.
- No general deployment approval from task-only adapters.
- If transfer remains weak despite fit improvement, loop back to supervision
  format/composition rather than silently add more rank or test-driven epochs.
- Do not assume 100/100 or equate local fit with general capability.
