# Retention-first answer-format comparison

Frozen before data generation/training. Implements the first comparison from
`native_transfer_lmm_solutions.md`, not a claim of a proven retention mechanism.

## Arms

Fresh untouched pinned LFM in each arm; identical rank-8/alpha-16 native
attention/conv LoRA, FP32, seed 0, AdamW lr 1e-4/decay .01/clip 1.

128 updates, exactly 48 genuine supervised positions per update (6,144 per arm).
Token-weighted gradient accumulation across unpadded examples; clip once per
update. No fake padding labels, truncation, test-driven checkpoint choice, or
different learning rates. Every update includes its same registered main
final-only example. Remaining positions are filled as follows:

1. `final_only`: other final-only examples from the primary 128 fact triples.
2. `worked_replay`: one algorithmically verified worked response for the main
   fact, then other primary final-only examples to meet the token budget.
3. `extra_final`: additional final-only examples from 128 disjoint extra triples.

Exact token packing uses distinct auxiliary prompt rows within an update.
Across updates examples can recur; count all exposures and distinct prompts.
The worked traces are prescribed correct arithmetic, not recorded native
responses or a KL preservation loss. This tests worked-example retention
supervision, not literal distillation of a native reasoning trajectory.

Input tokens, sequence counts, and compute differ because worked responses
contain more supervised tokens per sequence. Measure all differences; equal
supervised tokens and optimizer steps are not equal compute or primary-fact
exposure. Extra-final controls some added-supervision effects, not every coverage
or response-length confound. No isolated causal attribution beyond this design.

## Split contract

Freeze 100 prompts before training: 20 seen-primary-facts/unseen-template,
20 unseen-facts/seen-template, 20 unseen-facts/unseen-template, and 40 retention
tasks (10 each JSON, sorting, ordering, Spanish). Primary and extra fact triples
are disjoint, and both disjoint from unseen triples. Training templates 0/1;
template 2 withheld. Seen-fact split intentionally reuses facts, not prompts.
Audit normalized prompt overlap against all current/past registered suites.

Same generation settings as prior baseline: greedy, 192 tokens, FP16 inference,
FP32 adapter factors. Native strict answer compliance and parsed content remain
separate. Score all splits, paired gains/losses versus untouched model, and
worked-replay versus both controls. No answer extraction is used to inflate
native exact score. No substantive training on the final test.

## Gate

Worked replay is a preliminary positive result only if unseen-fact/unseen-template
strict success exceeds both controls, all 60 arithmetic parsed-content successes
are no lower than untouched, and no baseline-confirmed content or exact success
is lost anywhere in the registered 100 tasks. Report failure if any clause
fails. This strict retention gate may be unmet; do not change it after outputs.

No new full-capability, speed, or observer-advantage claim from one seed or a
templated suite. 100/100 remains a target, not an assumed outcome.
