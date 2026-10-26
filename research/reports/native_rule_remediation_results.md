# Rule-level causal probes and v2 remediation: progress with failed retention

## Findings established by controlled probes

Compared untouched and v1 remediation on the same arithmetic numbers and event
triples. Strict success (content extraction agreed with strict counts here):

| Arithmetic intervention | Untouched /20 | V1 /20 |
|---|---:|---:|
| Multiplication alone | 18 | 20 |
| Subtraction of supplied correct product | 19 | 20 |
| Product supplied in simple subtraction prose | 11 | 20 |
| Combined expression `(a*b)-c` | 0 | 1 |
| Correct product supplied inside compound prose | 0 | 1 |

The selected model can perform both primitives on these inputs but usually
fails to compose/select them in the compound context. Supplying the product
does not repair the compound-context failure. These are behavioral causal
controls, not proof of a unique internal arithmetic circuit defect.

| Ordering wording, paired earliest/latest | Untouched /40 | V1 /40 |
|---|---:|---:|
| Chronology/follows | 20 | 20 |
| Training-style before/after | 14 | 37 |
| Explicit first-to-last list | 21 | 21 |
| Chronology with permuted identifier roles | 20 | 20 |

Thus identifier availability is not the sole limitation; changing wording while
keeping the rule and roles fixed changes performance. V1's gain does not
transfer reliably to equivalent forms.

## Intervention and measured budget

Protocol: `native_rule_remediation_protocol.md`, frozen before training. V2
starts from untouched pinned LFM and adds isolated operations, worked compound
answers, and exact earliest/latest pairs with shuffled identifier roles/four
wording forms, while retaining JSON/sorting/translation data.

One epoch: 1,216 updates, 53,245 input tokens, 8,671 supervised tokens.
835,584 native attention/conv LoRA parameters; original-weight SHA-256 unchanged.
Measured training-loop time including checkpoint writes: 160.682441 seconds,
excluding initial data validation/model load. No intermediate test selection.

## New frozen test: use same-test comparisons only

| Exact success | Untouched | V1 remediation | V2 remediation |
|---|---:|---:|---:|
| JSON records | 0/20 | 18/20 | 17/20 |
| Signed sorting | 0/20 | 0/20 | 0/20 |
| Arithmetic | 0/20 | 3/20 | 6/20 |
| Transitive ordering | 0/20 | 0/20 | 10/20 |
| Spanish number words | 0/20 | 1/20 | 3/20 |
| Total | 0/100 | 22/100 | **36/100** |

Do not compare this 36 with v1's 53 as though task sets were unchanged. The
new prompt wording materially changes baseline and adapter performance.

V2 loses 14 untouched-model confirmed content successes on this test, all
arithmetic, despite gaining exact compliance elsewhere. It also fails the
preregistered ordering threshold of strictly greater than 10/20. Therefore
the v2 remediation **fails its advancement gates**.

## Training fit separates underfitting from transfer

At the final v2 checkpoint, exact free-generation fit on training examples:

- Multiplication: 128/128.
- Subtraction: 128/128.
- Compound final-only arithmetic: 81/128.
- Ordering: 80/80 in each of four wordings, 320/320 total.

Compound arithmetic still fails on 47 supervised training examples. Its
remaining problem is not exclusively unseen-prompt generalization. Ordering
fits all observed paired training questions but fails half of new questions;
its remaining problem includes transfer to unseen phrasing/identifiers.
These results do not identify whether optimizer, adapter placement/rank,
representation, or data quantity is the limiting cause.

## Exposed v1 test: targeted fix and cross-task regression

V1 test exact success changes 53→56/100. Ordering improves 10→20/20, but sorting
falls 7→0/20. JSON remains 20/20, arithmetic 0/20, and translations 16/20.
The targeted ordering intervention works on this now-exposed wording, but
not without interference. This diagnostic aggregate cannot override the new
test's failed retention gate.

## What is and is not remediated

Remediated: broken single-value coverage, missing paired earliest/latest
supervision, and lack of isolated operation examples. Some diagnostic ordering
failures are eliminated. Still open: reliable composition, transfer, and
cross-task retention. Neither 100/100 nor a universal capacity floor is shown.

Next experiments must isolate learning fit/capacity from interference, rather
than bundling more tasks and declaring success from an aggregate. Preserve
all failed controls. Both new test sets are now exposed and require replacement
for future confirmatory evaluation.

## Artifacts

- `research/results/native_rule_audit_v1/rules.json`
- `research/results/native_rule_v2_training_fit.json`
- `research/results/native_rule_v2_freeze/`
- `research/results/native_rule_v2_training.json`
- `research/results/native_rule_v2_comparison.json` (`repeated` key denotes the
  v1 comparator checkpoint; its provenance is retained in raw evaluation)
- `research/results/native_rule_v2_{baseline,previous,corrected,v1_diagnostics}/baseline.json`
