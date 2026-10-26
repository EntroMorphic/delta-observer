# Observer-guided improvement of pretrained LLMs

## Decision

The active objective is now to improve an already functioning pretrained LLM
at a controlled training budget. Standalone Observer.D compression and the
native-graph architecture reset are paused. Existing artifacts and negative
results remain intact; the former 10× compression objective is historical,
not a requirement of this improvement study.

No improvement to a standard pretrained LLM has yet been demonstrated.

Current decision: retention-transfer v2 failed all-seed approval and revealed
native arithmetic content 39/40 versus 5–10/40 after adaptation. The candidate
is rejected. See `native_arithmetic_access_findings.md` for the cumulative
finding and `native_arithmetic_access_protocol.md` for the next inference-only
test of recoverable output access versus prompt-conditioned computation damage.
No further training is authorized by a favorable formatting score alone.

Inference-only access controls are now complete: reasoning requests yield
ordinary seed content 15/40, 5/40, 34/40 vs original 6/40, 7/40, 10/40, with
paired regressions in every seed; untouched reasoning is 40/40. Numeric-only
decoding reduces untouched content to 0/40. All-adapters-off reproduces native
tokens exactly. See `native_arithmetic_access_results.md`. This demonstrates
partial conditional recovery, not an all-seed remedy or universal circuit cause.

Same-history causal forks are complete. Native nonnumeric reasoning prefixes
recover adapted content to 38/40, 36/40, 39/40 without supplying calculated
values, but one seed-2 example regresses from 71 to 512. Adapted boxed-answer
prefixes also damage native continuations with adapters off. At identical empty
histories, active adapters change next-token top-1 on all 40 cases per seed.
See `native_same_history_results.md`: early history is a causal mediator, not
the only mechanism or a deployment-ready retention/formatting remedy.

## Transferable evidence and limits

- Broader unique dialogue improved the custom model's instruction CE and
  teacher agreement relative to repeated exposure. That comparison matched
  sequence exposures, not exact supervised-token counts or measured compute;
  the new study must control those explicitly.
- Component-factor ranks can conceal low-rank reconstructed operators.
  Geometry audits must measure the deployed matrix products.
- Teacher support requirements depend on histories: top-64 retained about
  98.7% mass on static histories but 81.6% on student histories. This establishes
  a distribution difference, not that sparse tails caused incoherence.
- Correction gradients can be stronger and poorly aligned with static learning.
  Initial norm calibration does not guarantee the same ratio throughout training
  or after Adam normalization and clipping.
- Higher rank, larger prompt JS, fewer exact cycles, and lower CE did not
  establish coherent capability. They are diagnostics, not success criteria.

## First experiment: diversity under controlled adaptation

Use the unmodified, pinned `LiquidAI/LFM2.5-350M` checkpoint as the initial
candidate, subject to a measured baseline confirming useful task performance.
Keep its native graph; do not initialize from any Observer.D checkpoint.

Compare:

1. Untouched pretrained baseline.
2. Standard instruction adaptation using repeated prompts.
3. Identical adaptation using broader unique prompts.

Fix the adaptation method, trainable parameters, optimizer, learning-rate
schedule, precision, context length, initialization, update count, batch shape,
and supervised-token budget before training. Use matched response-length
strata and verified masks. Report actual token exposures, padding, wall time,
peak memory, and optimizer steps; equal steps are not equal measured compute.
Do not truncate EOS handling or create invalid histories to force matching.

The primary question is whether the unique arm improves held-out task success
over standard adaptation while preserving the starting model's capabilities.
Self-generated targets from the same model are a possible training signal,
not evidence that the targets improve on that model.

## Evaluation before intervention

Create a versioned development set and a separately sealed final test set,
excluded from training by normalized-text hashes and near-duplicate checks.
The former eight prompts have been repeatedly inspected and used to steer
research; retain them as historical diagnostics, not a fresh confirmatory test.

Measure per-category task success for:

- factual explanation and reasoning;
- exact JSON/format following;
- deduction and arithmetic;
- summarization fidelity;
- executable code tests;
- uncertainty/abstention;
- multilingual instructions.

Use deterministic validators where appropriate and blinded, rubric-based
assessment elsewhere. Freeze task definitions, sample counts, scoring,
aggregate weights, non-inferiority margins, seeds, uncertainty reporting, and
endpoint/checkpoint selection before running the comparison. Measure baseline
floor/ceiling effects before finalizing the tasks, using development data only.

Report capability separately from CE, teacher agreement, prompt JS, geometry,
and loop diagnostics. Do not claim success from a single favorable seed or
aggregate that hides a degraded category. A pilot is exploratory; confirmatory
evaluation requires locked choices and multiple training seeds.

## Subsequent mechanism tests

Only after the first controlled study:

- Test adaptive teacher-support selection against fixed support at measured
  extraction/training budgets.
- Test gradient-interference-aware correction against both ordinary adaptation
  and a matched extra-supervision control; track gradient direction and actual
  optimizer updates over time.
- Use operator-level conditioning audits to localize failures in a healthy
  pretrained model before proposing residual scales or architectural changes.

Each intervention requires its own preregistered comparison. Do not combine
unproven mechanisms and attribute an aggregate gain to the observer.

## Immediate next action

The unmodified baseline and development validators have now been measured and
red-teamed; see `pretrained_evaluation_red_team.md`. Duplicate BOS was corrected,
and a 44-task development expansion separates strict compliance from parsed
content success. Nine manual tasks remain pending and code is not executed.

The narrow exploratory native adaptation protocol was frozen in
`native_lora_pilot_protocol.md` and both matched arms have now completed. See
`native_lora_pilot_results.md`: JSON-record compliance improved with no loss of
baseline automatic successes; no diversity or observer-specific advantage was
demonstrated. This does not satisfy the broader study's confirmation requirements.

Next, freeze a larger multi-seed study with a held-out final evaluation and
broader training families. Architecture extraction remains paused.

The subsequent 100-task post-training holdout is complete: untouched exact
success 2/100; both adapters 19/100. Both lose three previously correct content
answers despite improved format compliance. See `native_holdout100_results.md`.
The next study must address content retention, not assume the development
pilot's zero-regression result generalizes. This inspected holdout cannot be
reused as an untouched final test after informing new interventions.

The subsequent atomic audit identifies a single-value training-data selection
defect and serialization-dependent inherited copy failures. See
`native_atomic_bedrock.md`. Builder v2 enforces eight-value coverage; it has not
been trained. Next comparisons must separate coverage from serialization and
use new test prompts rather than rebrand the inspected 100 tasks as unseen.

Remediation v1 has now completed under `native_remediation_protocol.md`:
53/100 on a newly frozen prompt-disjoint test, versus 11/100 untouched and
31/100 old adapter on that same test. JSON copy errors are repaired on the old
diagnostic set, but four arithmetic content regressions remain. See
`native_remediation_v1_results.md`; this is not 100/100 or full retention.

Rule-level probes and v2 remediation now separate primitive arithmetic from
composition and ordering training fit from transfer. V2 scores 36/100 on its
new test versus v1's 22/100 there, but fails content-retention and ordering
gates. See `native_rule_remediation_results.md`. Further work must isolate fit,
transfer, and interference rather than scale the failed bundled intervention.

Exposure-sequence-matched specialist ablations are complete. Removing other
tasks improves ordering transfer from 10/20 to 18/20 but reduces compound
arithmetic training fit from 81/128 to 77/128. Harmful interaction is supported
for ordering; it is not a sufficient explanation for arithmetic. Specialists
are not general replacements. See `native_specialist_results.md` for budgets,
controls, caveats, and the next isolated-fit/capacity experiment.

The fixed rank/duration ladder is complete: rank-8 compound training fit rises
77→99→118→125/128 over four passes; rank 32 ends at 116/128. Endpoint arithmetic
transfer is only 8/20 and 9/20 respectively on the exposed diagnostic test.
An initial exact directional-Hessian diagnostic was measured without
quantization; it is not a full sensitivity map, speedup, or capability result.
See `native_fit_curvature_redteam.md` and the four native-fit LMM artifacts.
The next bottleneck is fit-to-transfer, not an established rank/capacity floor.

A fresh full LMM pass and crossed fact/wording diagnostic now show that the
untouched model preserves correct explanatory arithmetic on 19/19 non-overlap
diagnostic triples, while rank-8 final-answer adaptation reaches only 7/19 on
that wording. Retention-first formatting, semantic-orbit/contrast supervision,
an explicitly separate answer-contract systems layer, and a gated curvature
study are candidate solutions. See `native_transfer_lmm_solutions.md`. None is
yet established as a general training fix; no new training occurred in this pass.

The subsequent retention-format experiment has completed under
`native_retention_format_protocol.md`. Worked supervision fails: 14/100 strict,
zero unseen-fact/unseen-template successes, and 17 lost baseline-confirmed
content successes. Equal updates and supervised tokens did not mean equal
input tokens/compute. See `native_retention_format_results.md`. No replacement
is approved; semantic-orbit training remains untested as an isolated arm.

Semantic-transfer measurement is now complete: same-test native exact scores
repeat 32/100, paraphrase 38/100, and add/subtract contrast 41/100. Unseen-fact/
unseen-template subtraction rises 5→7/20; fully correct operation pairs rise
0→6/20 with addition examples. Both candidates still fail paired retention
gates. See `native_semantic_transfer_results.md`; the positive local signal
does not authorize a general replacement or a 100/100 claim.

Atomic tracing now separates operation sensitivity, numeric candidate selection,
and attention-update-mediated schema-binding damage. Equal-length one-token
controls support the operation result; EOS-delay controls reject premature
stopping as the cause of probed JSON failures. Attention knockouts repair
measured retention but lose arithmetic gains. See `native_semantic_atomic_gap.md`.
The next question is direction-level separation, not simply disabling attention
or claiming a single proven floor. No training occurred during these audits.

Actual-update causal forks and a fixed projected-training run now establish
locally separable schema damage. Same-norm projection against one exposed
native key preserves registered arithmetic transfer counts while repairing
the two JSON regressions; one seen-fact arithmetic and two ordering regressions
remain. See `native_update_conflict_results.md`. This demonstrates avoidable
retention damage permitted by target-only updates, not an ultimate universal
cause or a general native-capability retention guarantee.

Retention transfer v2 completed its preregistered three seeds, ordinary/
four-example gradient-span projection/native replay/angle-matched random arms,
128 fixed updates, fresh locked 100-task evaluation after all endpoints finished.
See `native_retention_transfer_protocol.md` and
`native_retention_transfer_results.md`. One native-wrong protected ordering
continuation is retained without filtering; the experiment tests native behavior
preservation, not oracle correction. All three projected candidates fail the
fresh-retention gate; two also lose ordinary arithmetic exact successes.
Most importantly, native arithmetic confirmed content is 39/40 versus only
5–10/40 across every adapted arm despite higher native exact totals. The fresh
suite is now exposed diagnostic. No replacement is approved. Numerical behavior
retention/output-path changes must be isolated before another broad intervention.
