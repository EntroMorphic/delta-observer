# Remediation v1: 53/100 on the new frozen test, with retention limits

## Outcome

The coverage/serialization fixes are implemented and one fixed 640-update
native-LoRA remediation run has completed. It scores **53/100**, not 100/100,
on the new prompt-disjoint test frozen before training. Same-test untouched and
old repeated-adapter scores are 11/100 and 31/100.

| New-test exact task success | Untouched | Old adapter | Remediation |
|---|---:|---:|---:|
| JSON records | 0/20 | 20/20 | **20/20** |
| Signed sorting | 0/20 | 0/20 | **7/20** |
| Arithmetic | 0/20 | 0/20 | **0/20** |
| Transitive ordering | 10/20 | 10/20 | **10/20** |
| Spanish number words | 1/20 | 1/20 | **16/20** |
| Total | 11/100 | 31/100 | **53/100** |
| Format compliance | 58/100 | 77/100 | **92/100** |

The remediation passes its frozen preliminary new-test gate: it exceeds both
comparators and loses none of the untouched model's confirmed content
successes on this particular test. It does not meet a general zero-regression
requirement across all inspected data; see the diagnostic regressions below.

## What was changed and verified

- Started from untouched pinned LFM, not an inspected old adapter.
- All original parameters frozen; 835,584 attention/conv LoRA parameters trained.
- 128 examples per family, 640 total; 4,206 supervised and 28,041 input tokens.
- 16 copied values, eight exposures each; no single-value length-bucket collapse.
- Compact/spaced/pretty JSON counts: 43/43/42.
- Correct full-response/EOS labels, prompt masking, no duplicate BOS or truncation.
- New 100-prompt test, manifest, and hashes frozen before training; no intermediate
  test selection. Exact and word-5-gram training/evaluation overlap checks passed.
- Frozen base SHA-256 unchanged before/after training.
- Measured training-loop time including checkpoints: 85.463663 seconds, excluding
  initial loading/data validation.

Protocol: `native_remediation_protocol.md`. The old adapter has a smaller
training budget, so differences cannot be attributed solely to value coverage,
serialization diversity, or an observer-specific method. This is a remediation
bundle, not an isolated causal component experiment.

## Old 100-task set: diagnostic only

Exact success rises from untouched 2/100 and old-adapter 19/100 to remediation
36/100. JSON is now 20/20, including the former `fir`, `willow`, and `poplar`
regressions. This set informed the intervention and cannot establish new
generalization.

Confirmed content successes increase from 35 to 52, but four former arithmetic
successes regress:

- `arithmetic-05`: returns boxed 18 after adding shipped bolts rather than
  subtracting them.
- `arithmetic-08`: returns boxed -111.
- `arithmetic-10`: returns boxed -163.
- `arithmetic-12`: returns boxed 2 after unrelated subtraction.

These are wrong answers, not merely stricter formatting failures. Aggregate
improvement must not hide them. The remediation is therefore not approved as a
universally retained-capability replacement for the original model.

## Remaining measured failure modes

On the new test the ordering model always chooses the latest event, including
all ten questions asking for the earliest; this explains its exact 10/20
ceiling on that fixed balanced set. It is a behavioral shortcut, not a proven
capacity ceiling. Arithmetic outputs are integer-formatted but all 20 are wrong.
Sorting and four translation cases also remain incorrect. Formatting is now
much less of a bottleneck than content/rule selection.

The new test shares task families and Spanish mappings with training. It tests
new prompt forms, not novel Spanish knowledge. These one-seed results do not
establish broad reasoning improvement, independent statistical significance,
or observer-specific benefit. If used to guide another round, this test becomes
development data and another frozen evaluation is needed.

## Artifacts

- `research/results/native_remediation_v1_freeze/`: pre-training tasks/hashes/manifest
- `research/results/native_remediation_v1_training.json`
- `research/results/native_remediation_v1_comparison.json`
- `research/results/native_remediation_v1_old_diagnostic_comparison.json`
- Raw token IDs/responses in `research/results/native_remediation_v1_{baseline,old_adapter,corrected,old_diagnostics}/baseline.json`

## Next remediation gate

Keep the successful coverage and mask audits. Localize arithmetic operator
selection and earliest/latest instruction sensitivity, then preregister controls
that distinguish learned rules from identifier/position shortcuts. Include the
four arithmetic regressions in retention diagnostics and use a new frozen test.
Do not scale or claim 100/100 from the present result.
