# Lincoln Manifold: solutions for transfer without destroying existing skills

## Deployment

Applied the complete RAW → NODES → REFLECT → SYNTHESIZE cycle using
[`anjaustin/lmm@55719c0a`](https://github.com/anjaustin/lmm/tree/55719c0a2308d1a3f29ca656ed7b6287be33bca2).
Phase artifacts are `research/lfm/manifold/native_transfer_{raw,nodes,reflect,synth}.md`.
The synthesis was frozen before its first diagnostic. This decision memo records
the resulting evidence and explicitly revises priorities; it is not presented
as a preregistration written before those results.

## New measurement: the transfer comparison had a confound

Earlier comparisons changed both wording and operand triples. The new diagnostic
crosses the first 20 training compound triples and the 20 exposed test triples
with the exact training and diagnostic templates. No training, provided answer,
provided intermediate, calculator at inference, or checkpoint selection occurs.
Untouched LFM and the two fixed four-pass arithmetic adapters use the same
greedy/precision/192-token settings. Exact and narrowly parsed content scores
remain distinct.

| Facts / wording | Untouched strict | Untouched parsed content | Rank-8 strict/content | Rank-32 strict/content |
|---|---:|---:|---:|---:|
| Training / training | 0/20 | 20/20 | 20/20 | 19/20 |
| Training / diagnostic | 0/20 | 19/20 | 18/20 | 18/20 |
| Diagnostic / training | 0/20 | 20/20 | 12/20 | 10/20 |
| Diagnostic / diagnostic | 0/20 | 20/20 | 8/20 | 9/20 |

One diagnostic triple `(24,7,5)` was already in the training compounds. Excluding
it yields 19 cases: untouched parsed content 19/19 under both wordings; rank 8
11/19 under training wording and 7/19 under diagnostic wording; rank 32 9/19
and 8/19. Report the overlap rather than claim all numerical examples are new.

The untouched model explains correct arithmetic but violates integer-only
format. The adapters improve exact formatting while losing correctness on
different numerical cases. The first-20 training subset is fixed, not randomly
representative; these diagnostic counts establish local behavior, not population
effect sizes or a universal rule.

## Core conclusion

We should preserve demonstrated native arithmetic behavior while changing its
answer contract—not keep treating poor short-answer transfer as missing capacity.
New wording still matters, but cannot account for all observed errors. No
single internal neural cause is established, and 100/100 remains unmet.

## Solution 1 — retention-first answer-format adaptation (highest priority)

**Candidate, not yet demonstrated.** Compare the following on fixed, separately
split facts and templates:

1. Existing final-only supervised adaptation.
2. The same updates with replay of native, verified-correct worked responses
   on a separate retention set.
3. A matched extra-supervision control, so any replay gain is not simply more
   target tokens or more steps.

Construct labels algorithmically and verify every replay trace against the task;
do not assume model-generated explanations are true. Interleave final-only and
worked formats with explicit instructions. Measure both formats on the same
fact/wording crossings. Keep exactly registered update/token budgets and report
any unmatched compute. Start with ordinary CE before adding KL or Hessian-guided
constraints; isolate one new mechanism at a time.

**Acceptance:** higher native exact success than the locked standard-adaptation
control on unseen fact/template intersections, with no lost baseline-confirmed
content in any registered retention category. Training fit is not the gate.
If final-only compliance cannot improve without damaging retained content,
reject this design rather than relabel regressions as formatting noise.

## Solution 2 — semantic orbits plus counterfactual instruction pairs

**Candidate, not yet demonstrated.** Group multiple surface forms of the same
operation graph and include minimal pairs whose numeric tokens are identical
but whose correct operation changes. Examples: multiply-then-subtract versus
multiply-then-add; earliest versus latest; changing operand roles while keeping
the identifier vocabulary fixed.

Use group-aware splits: held-out facts, held-out templates, and the intersection.
Ensure no complete semantic fact graph leaks across a split labeled fact-held-out.
Audit value/answer/direction coverage, not just unique prompt count. Compare
equal-budget numeric diversity, paraphrase diversity, and paraphrase-plus-contrast
arms. The current diagnostic already exposes number and wording sensitivities;
it does not prove this training scheme will fix them.

**Acceptance:** both invariance to meaning-preserving changes and sensitivity to
meaning-changing changes improve, while Solution 1's retention gate holds.
A model that ignores instructions can be invariant and still wrong.

## Solution 3 — explicit answer-contract layer (separate systems track)

The untouched model already gives parsed-correct answers on the selected
diagnostic arithmetic cases. A narrow extractor can return the model's own
unambiguous boxed final integer, abstaining on ambiguity. This supplies no
answer key or arithmetic solver at inference and does not require destructive
weight updates.

**What is already measured:** all 20 selected diagnostic answers are parsed
correctly by the existing registered boxed-integer parser. Excluding the
training-overlap triple leaves 19/19. This observation can support a small
system prototype but is not a newly validated extraction policy.

**Limits:** count it as model-plus-parser performance, never native exact
compliance or increased model intelligence. The parser cannot repair confidently
wrong boxed answers, and earlier wordings produced such errors. Validate
abstention/ambiguity, adversarial contradictory outputs, and unseen expressions.
Measure prefill/decode latency; longer explanations may be slower. Do not claim
100/100 by silently substituting system scoring for the native-model metric.

## Solution 4 — sensitivity-guided retention (conditional research branch)

Curvature could identify update directions that help target tasks while harming
retention. Before using it, validate effective-native-weight directional
sensitivities on separate target/retention batches against actual perturbation
effects. Factor-space Hessians depend on parameterization; the existing six-row
LoRA diagnostic is not that map. No quantization is required to study sensitivity.

**Acceptance:** prediction of held-out damage must work before applying the map
to update allocation. Then compare to ordinary replay at measured budgets.
Speedups require actual selective computation, pruning, or kernels and separate
runtime evidence; curvature alone supplies neither speed nor intelligence.

## Execution order and loop-back

1. Completed: fact/wording crossing, exact source-label verification, overlap
   audit, and all raw outputs archived.
2. Next: freeze the Solution 1 comparison with grouped fact/template splits and
   explicit retention controls; integrate Solution 2 only as a separate arm.
3. Keep Solution 3 separately labeled if pursued; it cannot satisfy the native
   model's 100/100 target by redefinition.
4. Leave Solution 4 gated on sensitivity validation; do not deploy it based on
   the current small curvature diagnostic.
5. If the control already lacks arithmetic correctness on chosen templates,
   revise development coverage before locking the final test. Never redesign
   a final test after candidate outputs are seen.

No substantive training was launched during this LMM pass. There are actionable
candidate solutions and a stronger diagnosis—not a proven general solution.

## Evidence

- `research/results/native_fact_wording_audit_v1/probes.json`
- `research/results/native_fact_wording_audit_v1/crossing.json`
- `research/results/native_fact_wording_audit_v1/overlap_excluded_summary.json`
- All four `native_transfer_*` manifold phase files.
