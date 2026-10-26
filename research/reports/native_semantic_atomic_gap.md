# Atomic anatomy of the current semantic-transfer gap

## Scope and discipline

No training occurred. Audited the untouched model and the completed repeat,
paraphrase, and contrast adapters on exposed diagnostic tasks. Measurements:

1. Every serialized training label/exposure and arithmetic answer coverage.
2. First-answer-token logits/probabilities/ranks/entropy/support on 20 paired
   subtraction/addition questions per model.
3. Final prompt-token residual before operator, after operator, and after FFN
   at all 16 layers; diagnostic frozen-head probes at every captured stage.
4. Both-direction donor-state swaps at all 16 layers on the original 20 pairs,
   including self-patch and final-layer donor controls.
5. Equal-length, one-token `loses`/`gains` control on the same 20 triples, plus
   selected-layer state swaps.
6. Native JSON continuation teacher forcing on all five retention records,
   first divergence, target/EOS probabilities, and delayed-EOS generation.
7. Same-prefix baseline-state patches and all/single-layer/operator-group LoRA
   knockouts on three schema cases.
8. Four full-100-task operator-group knockout evaluations.

This is bounded, reproducible instrumentation, not measurement of every possible
weight/state or proof of an irreducible floor. Observational readouts and causal
interventions are explicitly distinguished.

## 1. Answer coverage is a contributor, not a complete explanation

Repeat and paraphrase training each expose 94 distinct numeric answer strings;
contrast exposes 143. Every answer is one vocabulary token followed by EOS:
**exactly 50% of supervised positions are EOS**. This count alone does not prove
EOS causes a failure.

For new facts under new wording:

| Training | Answer absent in labels | Answer present in labels |
|---|---:|---:|
| Repeat | 1/10 correct | 4/10 correct |
| Paraphrase | 2/10 correct | 5/10 correct |
| Contrast | 1/5 correct | 6/15 correct |

There are correct outputs for unseen answer labels and failures for covered
labels. Coverage correlates with performance in these small sets but neither
fully determines it nor establishes causality. Contrast also changes numerical
answer coverage, so do not attribute all gains to instruction pairing alone.

## 2. Many wrong outputs ignore the operation change

Identical operand triples; original prompts change `ships out` to `receives`:

| Model | Same first answer for add/subtract | Both correct first answers |
|---|---:|---:|
| Untouched | 15/20 | 0/20 |
| Repeat | 16/20 | 0/20 |
| Paraphrase | 18/20 | 0/20 |
| Contrast | **6/20** | **6/20** |

These first-token decisions agree with all 40 stored original generation first
tokens in every model. The original prompts are 50 versus 49 tokens, so that
measurement confounds operation meaning with length/position.

### Equal-length control removes that particular confound

Replace the changed phrase by `loses` versus `gains`. Each resulting pair has
identical length and differs at exactly one token, verified for every triple:

| Model | Same first answer | Both correct global top-1 | Both correct if only the two true answers competed |
|---|---:|---:|---:|
| Untouched | 5/20 | 0/20 | 5/20 |
| Repeat | 10/20 | 0/20 | 7/20 |
| Paraphrase | 14/20 | 0/20 | 3/20 |
| Contrast | **2/20** | **7/20** | **11/20** |

The equal-length result supports operation sensitivity after contrast training;
it is not just an original sequence-length effect. It still does not guarantee
correct numeric selection: contrast has eleven pairs with correct pairwise
answer ordering but only seven pairs with correct vocabulary-wide top-1.
On at least those four additional pairs, a third candidate wins somewhere.
This is a distinct numerical-selection failure beyond add/subtract preference.

The two-choice column is a diagnostic of logits, **not constrained decoding**
or a system supplied the answer choices at inference. No such decoder is scored
as native task success.

## 3. The operation signal reaches the output, but weakly in failed models

Original pairs have identical final-token states through the first two local
layers; differences first appear after global attention at layer index 2.
This is expected from local receptive fields and is not evidence those local
layers are defective.

Mean subtraction-versus-addition correct-answer logit selectivity at the final
output is 0.174219 untouched, 1.211426 repeat, 0.733594 paraphrase, and 4.132031
contrast. With equal-length controls the corresponding means are 0.994336,
2.646045, 2.076758, and 6.918359. Positive means do not imply every pair is
correct or even that its correct answer is the global maximum.

For original contrast pairs, last-token residual separation grows from mean
norm 0.02125 after layer-8 operator to 0.20571 after layer-13 operator and
0.63928 after layer-15 FFN. Late FFNs amplify separation in this native model;
the earlier custom-model diagnosis of shared FFN common-mode collapse must
not be transferred here. Native FFN weights are frozen, but their inputs have
changed via adapters.

Intermediate frozen-head probes change strongly late in the graph. These are
logit-lens diagnostics, not the actual final decision from running a shortened
model. They cannot alone identify a unique semantic circuit.

## 4. State swaps establish a causal mediator, not a unique root circuit

Swap only the final prompt-token residual after a layer from add input into the
subtract run, keeping all other recipient positions unchanged. Original-pair
contrast swaps at layer 12 make all 20 recipient top-1 predictions match their
add donors, versus six matches before intervention. On equal-length controls,
layer-10 swaps match 17/20 donor predictions, versus two before intervention;
18 recipient predictions change. A changed prediction need not match its donor.

Self-patch maximum logit error is 0.0 on every original pair/model. Final-layer
cross-patches match donor top-1 predictions, as expected from final normalization
and output projection. These validate mechanics, not a causal localization
claim by themselves.

The late final-position residual is sufficient to carry the measured answer
change under these interventions. It contains all represented context features,
not only operation identity; a donor can convey a computed answer or unrelated
features. Early failure to flip does not prove the early layer lacks relevant
information elsewhere in the sequence. No unique neuron/head is identified.

## 5. Retention damage begins at schema binding, not EOS

The baseline emits the correct key/value schema for all five JSON records,
although fenced Markdown violates native bare-JSON instructions. At the first
shared continuation position where `label` should be emitted, adapted models
instead copy the value as the key:

```text
Expected structure: {"label":"pendant","enabled":true}
Adapted structure:  {"pendant":true}
```

For paraphrase at this fixed prefix:

- `record-0`: correct key probability 0.019276; EOS probability 1.40e-8;
  `label` loses by 3.921875 logits.
- `record-1`: correct key probability 0.159620; EOS probability 2.76e-8;
  `label` loses by 1.656250 logits.

The correct key remains the second-ranked token. Delaying EOS through eight
generated tokens changes none of these malformed schema outputs. Thus early
termination is not the operative cause at these measured divergences, despite
the training set's 50% EOS-label share. Mere EOS-frequency counts would have
suggested the wrong remedy.

## 6. Attention adapters mediate both the gain and the damage

On the three probed records, disabling all adapters restores exact baseline
logits (maximum error 0.0). Disabling attention adapters restores all three
JSON contents for paraphrase and contrast; disabling conv adapters retains the
two malformed schema cases. Same-prefix native residual patches repair key
selection at late layers, but the sufficient layer differs by record/model.

Single-layer knockouts do not uniformly repair the two principal cases. A full
generation can change its prefix even when the fixed-reference key rank does
not change, so neither readout substitutes for the other.

Full-100-task knockout results:

| Model/intervention | Exact /100 | New facts + new wording /20 | Lost untouched content successes |
|---|---:|---:|---:|
| Paraphrase full | 38 | 7 | 2 |
| Paraphrase attention off | 20 | 3 | **0** |
| Paraphrase conv off | 23 | 5 | 2 |
| Contrast full | 41 | 7 | 4 |
| Contrast attention off | 14 | 0 | **0** |
| Contrast conv off | 22 | 3 | 2 |

Removing the attention-update group repairs measured retention but sacrifices
much of the learned arithmetic. This establishes a group-level tradeoff under
these interventions. It **does not prove** that useful and harmful updates
occupy the same parameter directions, that every attention module is necessary,
or that safe projected/constrained updates are impossible. Zero scale still
executes adapter arithmetic; no speedup is implied.

## What the evidence supports now

The gap has at least three measured components:

1. Weak operation sensitivity in repeat/paraphrase models, improved by minimal
   operation pairs under a one-token/equal-length control.
2. Wrong numeric candidate selection even when the add/subtract ordering is
   correct; label exposure explains only part of observed behavior.
3. Cross-task schema-binding regressions mediated by attention adapter updates,
   not EOS stopping, despite their coincident arithmetic benefit.

No global architectural/optimizer/capacity floor or single lowest-level cause
is proven. The next discriminating experiment is **direction-level separation**
of useful arithmetic and harmful retained-schema updates, with measured actual
update effects—not another whole-group knockout or an unvalidated curvature
heuristic. A separate numerical-generalization control is still required;
fixing instruction sensitivity alone cannot yield 100/100.

## Artifacts and reproduction

- `research/results/native_semantic_atomics_v1/atomics.json`
- `research/results/native_semantic_controls_v1/controls.json`
- `research/results/native_schema_interference_v1/schema.json`
- `research/results/native_semantic_{paraphrase,contrast}_{attention,conv}_off/baseline.json`
- `research/results/native_semantic_atomic_summary_v1.json`

Executables: `audit_semantic_atomics`, `audit_semantic_controls`,
`audit_native_schema_interference`, `summarize_semantic_atomics`, and the explicit
`evaluate_pretrained_development --disable-adapter-group` intervention.
No new training or native-capability improvement is claimed from the audits.
