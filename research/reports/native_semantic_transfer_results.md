# Semantic-transfer candidate: partial signal, failed retention gates

## Fixed comparison completed

Protocol: `native_semantic_transfer_protocol.md`, frozen before building data or
training. All arms start from untouched pinned native LFM. Only the previously
specified attention/conv LoRA parameters train (rank 8, alpha 16). Ordinary CE,
not a new contrastive objective, is used in all arms.

Each arm receives the same 128 numerical triples, four examples per update,
four passes, 512 updates, 2,048 sequence exposures, and 4,096 supervised tokens.
Final outputs and EOS are supervised without padding/truncation. The test was
hashed before training; no intermediate checkpoint selection occurred.

- Repeat: four copies of template 0, subtraction only.
- Paraphrase: four subtraction templates, same answer for each triple.
- Contrast: two subtraction and two addition examples on identical numerical
  tokens, using two templates. This changes operation coverage and reduces
  subtraction exposures; it does not isolate a contrastive-loss mechanism.

## Exact results on the same frozen 100 prompts

| Split / metric | Untouched | Repeat | Paraphrase | Contrast |
|---|---:|---:|---:|---:|
| Seen facts / new template | 0/20 | 15/20 | 17/20 | 11/20 |
| New facts / seen template | 0/20 | 10/20 | 11/20 | 10/20 |
| New facts / new template | 0/20 | 5/20 | 7/20 | 7/20 |
| Counterfactual addition | 0/20 | 0/20 | 1/20 | 10/20 |
| Retention tasks | 2/20 | 2/20 | 2/20 | 3/20 |
| Total native exact | 2/100 | 32/100 | 38/100 | 41/100 |
| Both wordings correct on a new triple | 0/20 | 3/20 | 7/20 | 6/20 |
| Both add/subtract correct on a new triple | 0/20 | 0/20 | 0/20 | 6/20 |

The test has 80 arithmetic tasks and 20 retention tasks; this aggregate is not
comparable to earlier 100-task totals with different categories/weights.
Semantic pairs intentionally reuse facts inside the test, so these are not
100 independent capability observations. No significance or multi-seed
generalization claim is made from these counts.

In this fixed run, paraphrase supervision improves the primary subtraction
transfer cell from 5 to 7. Addition examples yield six fully correct add/subtract
pairs where repeat and paraphrase yield none. This is limited positive evidence
for the candidate direction, not a retained-capability solution.

## Paired regressions determine the gate

| Registered loss check | Paraphrase | Contrast |
|---|---:|---:|
| Repeat-arm new-fact subtraction successes lost | 2 | 3 |
| Untouched confirmed content successes lost | 2 | 4 |
| Untouched exact successes lost | 0 | 1 |
| Advance | **No** | **No** |

Paraphrase loses `new_facts_new_template-00` and `-18` relative to repeat and
loses baseline JSON content in `record-0` and `record-1`. Contrast loses
`new_facts_seen_template-03`, `-06`, and `-13` relative to repeat, plus baseline
content in `record-0`, `record-1`, `order-1`, and `order-3`.

All three adaptations retain gains/losses individually in the comparison
artifact. Greater aggregate arithmetic success cannot hide these regressions.
The frozen content parser can leave wrong plain integers unresolved; confirmed
content successes and paired losses remain explicit and no native success is
awarded by extraction or an external solver.

## Measured budgets, compute differences, and integrity

| Measurement | Repeat | Paraphrase | Contrast |
|---|---:|---:|---:|
| Supervised-token exposures | 4,096 | 4,096 | 4,096 |
| Input-token exposures | 90,112 | 78,848 | 87,552 |
| Distinct prompts | 128 | 512 | 512 |
| Subtraction sequence exposures | 2,048 | 2,048 | 1,024 |
| Addition sequence exposures | 0 | 0 | 1,024 |
| Measured training-loop seconds | 136.475730 | 139.050486 | 159.303090 |

Only 835,584 adapter parameters trained. Frozen original-weight hashes match
before and after in all runs. Equal target tokens, sequences, and updates do
not imply equal input compute. Times include checkpoint writes but exclude
loading/initial verification and are not intrinsic kernel-speed comparisons.

## Evaluation integrity and scope

- Primary/held-out triples have zero overlap. Seen-fact/new-template cases
  intentionally reuse supervised facts, not prompts.
- Template 4 is withheld from all training. New-fact template-0/template-4 and
  subtract/add pairs deliberately use the same test facts.
- Exact normalized prompt exclusions include currently registered old suites;
  they are not semantic or upstream-pretraining decontamination.
- Scoring verifies frozen task and training-manifest hashes, model/revision/
  precision/generation settings, adapter arm/provenance, exact IDs and prompts.
- All four fixed endpoints use identical greedy inference and a 192-token cap.
- The untouched baseline scores zero on these arithmetic prompts. This weak
  baseline is reported rather than redesigning the frozen test after results.
  It limits claims of preserving native arithmetic on this particular suite.

## Decision and next scope

Neither candidate passes; do not scale or approve it as a native model
replacement. These measurements support a modest transfer/sensitivity signal,
while confirming that retention remains unsolved. This experiment deliberately
did not combine a new retention mechanism with semantic supervision. The next
separate comparison would need actual native-behavior preservation on verified
retention cases, not simply more worked examples or another exposed score.

No 100/100, broad intelligence improvement, or observer-specific advantage is
established. This test is now exposed and cannot remain a final test for
interventions informed by its outputs.

## Evidence

- `research/results/native_semantic_transfer_v1_freeze/`
- `research/results/native_semantic_transfer_v1_comparison.json`
- `research/results/native_semantic_transfer_v1_{repeat,paraphrase,contrast}_training.json`
- `research/results/native_semantic_transfer_v1_{baseline,repeat,paraphrase,contrast}/baseline.json`
- Executables: `build_semantic_transfer`, `train_retention_format`,
  `evaluate_pretrained_development`, `score_semantic_transfer`.
