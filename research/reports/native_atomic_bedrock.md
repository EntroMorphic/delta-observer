# Native-model atomics: measured mechanisms and remaining unknowns

## What was measured

No training occurred during this investigation. Measurements comprise:

- Canonical-answer teacher forcing on all 100 tasks for untouched, repeated,
  and unique checkpoints: per-token NLL, target probability/rank, correct-versus-
  competing logit margin, entropy, and top-64 mass.
- Adapter/base activation magnitudes at all adapted operators on six record
  cases.
- Twenty adapter interventions per trained checkpoint on those six cases:
  full, all-off, attention-off, conv-off, and each of 16 layers off separately.
- Pristine FP32 weight reload and generation on 11 selected cases per model.
- Five syntax-prefix interventions across all 20 records and three models.
- Four copy/schema counterfactuals across all 20 records and three models.
- Exact inspection of every training response and a repaired data-builder run.

These are finite measurements, not exhaustive instrumentation of every possible
state or a proof of an irreducible capability floor.

## Bedrock 1: the original data did not identify a copying rule

Both training arms contain **128/128 item values equal to `clock`**. The unique
arm has 128 distinct prompts and 79 distinct quantities; the repeated arm has
16 distinct prompts and 16 quantities. Both have item-label entropy exactly
zero. Constantly emitting `clock` and correctly copying the item both fit every
observed item label. Those data cannot distinguish the two rules.

The cause is a demonstrated builder defect: sorted length buckets were filled
until 128 examples were obtained. The first eligible bucket supplied only
`clock`. Exact length matching was achieved at the expense of value coverage.
Earlier claims of 128 unique prompts remain literally correct but do not imply
128 distinct semantic tasks or varied item copying.

This does not prove that the model forgot how to copy, or that more varied data
will fix it. It proves the completed experiment supplied no varied-item
supervision and cannot answer the intended item-diversity question.

## Bedrock 2: compact serialization exposes inherited errors

Force a syntax prefix without supplying any answer value, holding model and
question fixed:

| Forced prefix | Untouched content /20 | Repeated /20 | Unique /20 |
|---|---:|---:|---:|
| `{"label":"` | 17 | 17 | 17 |
| `{"label": "` | 19 | 19 | 19 |
| Pretty bare JSON | 19 | 19 | 19 |
| Pretty fenced JSON | 19 | 0 | 1 |
| Compact fenced JSON | 15 | 17 | 17 |

The untouched model on compact bare JSON makes exactly the same three value
errors as the ordinary adapted outputs: `fir→f`, `willow→label`, and
`poplar→label`. A legal space after the colon repairs `fir` and `poplar` in
all three checkpoints, but not `willow`.

Thus the observed content regression is not evidence that adaptation destroyed
all knowledge of these values. Adaptation changes the generated serialization
path; the base already has errors on that path. Forcing previously native
pretty fences after adaptation frequently produces an extra closing brace,
showing that the learned change also affects structural continuation.

These forced-prefix counts are causal diagnostics, not normal-generation
benchmark scores or a validated deployment intervention. The intervention
supplies the opening key and syntax, but no correct field value.

## Bedrock 3: binding, not an absolute representation limit

Twenty-value controls:

| Intervention | Untouched correct /20 | Repeated /20 | Unique /20 |
|---|---:|---:|---:|
| Plain word echo | 19 | 19 | 19 |
| Quoted value in question + spaced prefix | 20 | 20 | 20 |
| Rename field `label` to `item` + spaced prefix | 19 | 20 | 20 |
| Rename field `label` to `name` + spaced prefix | 20 | 20 | 20 |

Plain echo's remaining failure is `beech→bear`; it is not one of the three
original regressions. Quoting the requested value or changing its key repairs
the residual `willow` failure on these controls. The value can be produced
correctly by the same checkpoint under a changed context.

There is **no demonstrated capacity floor on these 20 records**. There is a
measured dependence on prompt binding and serialization. The counterfactuals
localize behavioral sensitivity; they do not by themselves identify a unique
internal circuit implementing the mistake.

## Bedrock 4: errors are confident, not sparse-tail or selected precision failures

Repeated-adapter decisions on the compact canonical reference path:

| Case | Correct next piece | Actual preferred piece | Correct probability | Correct logit margin |
|---|---|---|---:|---:|
| `fir` | `ir` after `f` | closing quote/comma | 0.105633 | -2.109375 |
| `willow` | `will` | `label` | 0.007093 | -4.703125 |
| `poplar` | `pop` | `label` | 0.070867 | -2.312500 |

Top-64 mass at these decisions is 0.997618, 0.999363, and 0.999659. Missing
sparse support is not the bottleneck at these decisions. Pristine FP32 reload
preserves all three wrong adapted values. This excludes FP16 as a necessary
cause of those three errors, not as a cause of every difference elsewhere.

## Bedrock 5: no single-layer knockout repair

All-adapter-off logits match untouched logits exactly: maximum absolute error
0.0 on every selected reference. This validates the intervention control.

Disabling any single adapted layer, or all attention/conv adapters separately,
does not repair the three records in ordinary generation. Some interventions
change target logit margins substantially without crossing the greedy decision
boundary. This is a distributed effect under these interventions; it does not
justify naming one operator as the sole cause. The largest selected prompt-
boundary update/base norm ratio is about 0.0351, so small operator updates can
coexist with large output-path changes. The full exact values are retained.

## Teacher forcing: distinguish serialization from content

| Canonical reference CE | Untouched | Repeated | Unique |
|---|---:|---:|---:|
| JSON records | 0.947158 | 0.067794 | 0.067582 |
| Signed sorting | 1.763017 | 0.899396 | 0.878934 |
| Arithmetic | 7.821754 | 7.368887 | 7.367652 |
| Spanish number words | 1.834093 | 2.414510 | 2.449830 |
| Transitive ordering | 2.041322 | 1.811935 | 1.841199 |

JSON reference-token agreement reaches 237/241 in both adapters despite three
wrong complete records. One canonical token mismatch is `elm` versus `el`,
where a different subsequent tokenization still produces the correct string.
Reference-token agreement is therefore not interchangeable with semantic
correctness. Bare-answer arithmetic CE is high largely at response start,
where the model prefers explanatory prose; this is not a pure arithmetic-skill
measurement. Spanish reference likelihood worsens despite unchanged strict
success counts, exposing degradation invisible to aggregate task successes.

## Repair completed, not yet trained

Builder v2 now requires 16 exposures of each of eight item values in both arms,
with two groups per item and exact paired lengths. Missing coverage is a hard
error. Tests reproduce the shortest-bucket failure mode and verify balancing.

Measured v2 budgets: 128 exposures, 9,904 input tokens, and 1,520 supervised
tokens per arm. Values are lantern, mug, notebook, ribbon, basket, kettle,
scarf, and clock, each appearing exactly 16 times. The 100-task set is included
in the exclusion registry. Old datasets/checkpoints/results remain untouched.
No v2 training has been launched; its increased budget must not be compared
with v1 as if tokens were held fixed.

## Next justified test

Separate, matched controls for (a) item coverage and (b) serialization diversity,
using builder-v2 integrity checks, not another arbitrary loss. Measure copied
values separately from syntax, and include fields/values not seen in training.
The current 100-task set is now diagnostic/development data; a new frozen test
is required before claiming generalization from interventions it informed.

No evidence here establishes an optimizer, architecture, data-scale, or
irreducible model-capacity floor for broader reasoning. Sorting, arithmetic,
and translation still require their own causal probes.

## Artifacts

- `research/results/native_atomic_audit_v1/atomics.json`
- `research/results/native_prefix_audit_v1/prefixes.json`
- `research/results/native_copy_binding_audit_v1/copy_binding.json`
- `research/results/native_atomic_bedrock_summary.json`
- `research/results/native_lfm_pilot_data_v2_manifest.json`
