# SYNTHESIZE: candidate solutions and the next discriminating measurement

LMM source: anjaustin/lmm commit
`55719c0a2308d1a3f29ca656ed7b6287be33bca2`.
Status: execution guide frozen before the fact/wording diagnostic.

## Step 0 — remove the transfer confound, no training

Take the first 20 compound-final arithmetic training rows and all 20 arithmetic
rows from the exposed v2 diagnostic test. Parse and verify their operand triples
against stored labels. Create the complete crossing:

| Facts | Wording |
|---|---|
| Training triples | Original training template |
| Training triples | Original diagnostic template |
| Diagnostic triples | Original training template |
| Diagnostic triples | Original diagnostic template |

Evaluate untouched native LFM and both fixed four-pass arithmetic adapters
(rank 8 and rank 32). Same greedy decoding/precision/token budget; no supplied
intermediate answers, calculator use at inference, weight changes, or retraining.
Keep per-example responses/IDs and operands; score exact and narrow parsed
integer success separately. Record within-wording paired changes across facts
and within-fact paired changes across wording. State the actual overlap of
diagnostic triples with all training triples. These are exposed diagnostic
families, not a new generalization benchmark or independent statistical sample.

## Candidate solution A — semantic-orbit training (primary)

If fixed-fact wording changes cause failures, build groups for each operation
graph (e.g. `(a*b)-c`) containing varied vocabulary, clause order, and expression
renderings with the same answer. Also build minimal pairs with identical numeric
tokens but changed operation/order (`a*b+c`, `(a-c)*b`, earliest/latest).

Use ordinary supervised CE first; do not add an unvalidated contrastive loss.
Audit answer diversity, both instruction directions, operand roles, and complete
group splits. Freeze separate held-out facts, templates, and their intersection.
Match baseline/control token exposures and steps through precomputed bins.
Compare repeated examples, new-number examples, paraphrase-orbit examples, and
orbit-plus-minimal-pair examples. Do not call more unique strings more semantics.

Primary result: greater strict success on locked unseen template/fact splits
without loss of baseline-confirmed content in any registered retention category.
Neither training fit nor an exposed 100-task total can override that gate.

## Candidate solution B — retention-constrained integration

Ordering specialization has a measured local benefit. Test replay of native
correct answers during task updates, with the same target-task exposures and
an explicit extra-data/update control. First measure paired gradients on target
and retention rows; then measure actual update effects rather than assume raw
gradient cosine predicts Adam/clipping outcomes.

Only after a supervised replay baseline is established, compare a constrained
update or calibrated preservation loss. Any adapter routing/merging must be
evaluated as an explicit multi-component system, not silently presented as a
single improved native model. Acceptance requires specialist transfer benefit
surviving integration and zero registered content regressions.

## Candidate solution C — validated effective-weight sensitivity (secondary)

Define hypothetical perturbation directions in effective native weight space,
not only LoRA factors. Keep inference weights unquantized. On fixed target and
retention batches measure gradient/Hessian directional response and actual
symmetric perturbation losses at preregistered magnitudes. Evaluate whether
the map predicts damage on separate retained cases. Fail this branch if the
prediction is unreliable; finite-difference disagreement is not an improvement.

Only a validated map may inform update allocation, adapter placement, or later
selective computation. Capability gains and wall-clock speed gains have separate
baseline comparisons. A map alone provides neither. No full Hessian or native
quantization map is claimed from the current six-case LoRA diagnostic.

## Routing and loop-back rules

- Training-wording accuracy high, new-wording low on identical facts → prioritize A.
- Both wordings weak on new facts, fit high on seen facts → prioritize fact/operation
  coverage controls within A; do not attribute everything to paraphrasing.
- Task gain disappears when retained tasks train → B, keeping optimizer-state
  effects explicit.
- C cannot predict held-out damage → no curvature-directed optimization yet.
- If both fact and wording changes interact, retain all four diagnostic cells;
  do not choose whichever cell supports the preferred hypothesis.
- No new substantive training is launched during this LMM pass. Freeze the
  selected training comparison separately after the diagnostic resolves scope.
