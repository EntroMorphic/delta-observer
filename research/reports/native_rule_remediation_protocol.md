# Rule-remediation v2 protocol

Frozen before data generation/training. Controlled probes found remediation-v1
multiplication/subtraction each 20/20 but composition 1/20, and ordering 37/40
on training wording versus 20/40 on chronology wording. These localize tested
failures, not a unique internal circuit or irreducible capacity floor.

One fresh-base run is authorized: same pinned native model, rank-8/alpha-16
attention/conv LoRA, FP32 AdamW lr 1e-4/decay .01/clip 1, batch 1, seed 0.
One epoch of **1,216 updates**, no intermediate checkpoint selection.

Data: 128 each JSON/sorting/translation retention examples; 128 each isolated
multiplication, isolated subtraction, final-only compound tasks, and worked
compound tasks; 320 ordering examples arranged in exact earliest/latest pairs
on identical premises, four wording forms, and shuffled identifier roles.
Keep EOS and full-answer masks; reject truncation and evaluation overlaps.
Freeze a new 100-task test before training and record actual token budgets.

Compare untouched, completed v1 remediation, and v2 on the same new tasks.
Also audit the exposed v1 test for category/content regressions. Preliminary
advance requires exact success higher than both comparators, arithmetic >0/20,
ordering >10/20, and no lost baseline-confirmed content successes on the new
test. Report every category and regressions even if aggregate score improves.

This changes data mix and budget; it cannot establish isolated component or
observer-specific benefit. New prompts share task families and Spanish facts
with training. Aiming at 100/100 does not authorize scoring changes or repeated
test-guided checkpoint choice.
