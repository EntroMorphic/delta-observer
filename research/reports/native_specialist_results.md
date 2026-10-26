# Specialist ablation: ordering interference and persistent arithmetic underfit

## Fixed comparison

Protocol: `native_specialist_protocol.md`, frozen before specialist training.
Both specialist adapters started from untouched pinned LFM with the same native
LoRA rank/targets, seed, initialization, AdamW settings, precision, and clipping
as bundled v2. Each received exactly the corresponding v2 task examples once,
in the exact within-task sequence reconstructed from v2's completed history.
Source dataset/history hashes and complete source-row exposure were verified.

Other-task updates were removed, not replaced by no-op optimizer steps. Total
updates and optimizer moment histories therefore differ. This intervention
tests removing other-task training; it does not isolate gradient conflict from
optimizer-state and total-training effects.

## Training-fit result

| In-sample strict generation | Bundled v2 | Arithmetic-only | Ordering-only |
|---|---:|---:|---:|
| Multiplication | 128/128 | 128/128 | not trained |
| Subtraction | 128/128 | 128/128 | not trained |
| Compound arithmetic, final-only | 81/128 | **77/128** | not trained |
| Ordering, all four wordings | 320/320 | not trained | **320/320** |

Removing other tasks does **not** repair arithmetic training fit. In this
ablation the arithmetic specialist fails 51 supervised compound examples,
despite fitting both primitive-operation sets perfectly. Interference with
ordering/JSON/sorting/translation is not a sufficient explanation for the
remaining compound failures. This is not proof of an irreducible capacity
floor: optimizer settings, number of exposures, adapter placement/rank, and
compound supervision remain unisolated alternatives.

## Same exposed v2 diagnostic test

| Exact success | Bundled v2 | Arithmetic-only | Ordering-only |
|---|---:|---:|---:|
| JSON records | 17/20 | 0/20 | 0/20 |
| Signed sorting | 0/20 | 0/20 | 0/20 |
| Arithmetic | 6/20 | **5/20** | 0/20 |
| Transitive ordering | 10/20 | 0/20 | **18/20** |
| Spanish number words | 3/20 | 0/20 | 0/20 |
| Total | 36/100 | 5/100 | 18/100 |

Ordering specialist gains eight exact ordering successes and loses none of
bundled v2's ten ordering successes. It substantially improves transfer to the
new wording after removal of other-task updates. Arithmetic specialist gains
two arithmetic answers but loses three, for net -1 in that category.

Specialists never receive the removed task labels; zeros elsewhere are not
evidence they destroyed all original content knowledge. They also lack the
bundled model's compliance gains. Both lose many bundled exact successes and
are **not approved as general replacements**. No new aggregate improvement or
observer-specific benefit is claimed.

This 100-task test is already exposed development/diagnostic data. Neither the
specialist comparisons nor any resulting selection constitute fresh holdout
evidence. No test-driven intermediate checkpoint was evaluated.

## Measured budgets and integrity

| Measurement | Arithmetic-only | Ordering-only |
|---|---:|---:|
| Updates | 512 | 320 |
| Input tokens | 20,224 | 14,240 |
| Supervised tokens | 3,840 | 1,280 |
| Training-loop seconds | 65.619274 | 41.472053 |

Loop times include checkpoint writes and exclude loading/validation. Original
parameter SHA-256 before and after is unchanged in both runs:
`89baecd1ef14b5e4e9ffee7b86224faa0b5a1a7ca1b7a88ec2fa952ad828b60e`.
Only the 835,584 adapter parameters trained. Test generation uses the same
settings/precision as the bundled evaluation. Training fit uses the same
64-token budget as the earlier fit audit.

## Decisions

- Ordering diagnostic criterion (>10/20): **passes**, 18/20. Harmful interaction
  with removed training is supported, not uniquely identified as gradient
  conflict. Retention-aware integration remains open.
- Arithmetic criterion (compound fit >81/128): **fails**, 77/128. Do not keep
  assuming cross-task interference is its root cause.
- General deployment: **not approved**.

Next justified arithmetic experiment is a predeclared isolated fit ladder or
adapter-capacity control on the same fixed training examples, with separate
diagnostic evaluation. Next ordering experiment must preserve the specialist's
gain while explicitly controlling interactions with retained tasks. Do not
merge specialists blindly or describe their task-specific scores as 100/100.

## Artifacts

- `research/results/native_specialist_v1_comparison.json`
- `research/results/native_specialist_v1_{arithmetic,ordering}_fit.json`
- `research/results/native_specialist_v1_{arithmetic,ordering}_training.json`
- `research/results/native_specialist_v1_{arithmetic,ordering}_manifest.json`
- `research/results/native_specialist_v1_{arithmetic,ordering}_eval/baseline.json`

Reproduction uses `build_native_specialists`, `train_native_lora`,
`audit_native_training_fit`, and `evaluate_pretrained_development`. The
specialist manifests preserve the explicit source exposure order; a training
order hash is recorded in each checkpoint configuration.
