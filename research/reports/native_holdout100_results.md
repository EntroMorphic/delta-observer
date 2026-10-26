# Native LFM: measured 100-task holdout

## Result

| Metric, denominator 100 | Untouched | Repeated LoRA | Unique LoRA |
|---|---:|---:|---:|
| Exact task successes | 2 | **19** | **19** |
| Format-compliant outputs | 20 | **57** | **57** |
| Confirmed content successes | **35** | **35** | 34 |
| Content-parser failures | 27 | 27 | 28 |
| Content unresolved | 38 | 38 | 38 |
| Token-budget truncations | 0 | 0 | 0 |

The measured exact scores are **2/100, 19/100, and 19/100**, not 100/100.
Confirmed content is a conservative parser-based count, not full semantic
accuracy: unresolved cases have not been independently judged. Never normalize
content successes over only the parseable responses.

## Per-family exact success

| Family | Untouched | Repeated | Unique |
|---|---:|---:|---:|
| JSON records | 0/20 | **17/20** | **17/20** |
| Signed sorting | 0/20 | 0/20 | 0/20 |
| Arithmetic | 0/20 | 0/20 | 0/20 |
| Transitive ordering | 0/20 | 0/20 | 0/20 |
| Spanish number words | 2/20 | 2/20 | 2/20 |

The entire strict improvement is in JSON records. Arithmetic often has correct
boxed content but violates integer-only instructions. Sorting often becomes
syntactically valid JSON after adaptation but remains numerically wrong.

## Paired retention: aggregate counts hide regressions

Both adapters gain 17 exact successes and lose zero exact baseline successes.
However, both lose three baseline content successes:

- `record-07`: `fir` becomes `f`.
- `record-08`: `willow` becomes `label`.
- `record-15`: `poplar` becomes `label`.

Repeated adaptation gains three other content successes, masking those losses
in its unchanged aggregate 35 count. Unique adaptation gains two, giving 34.
Neither adapter satisfies a zero-content-regression requirement on this set.
There is no demonstrated unique-data advantage.

## Controls and scope

Protocol: `native_holdout100_protocol.md`, frozen before any generation on this
set. All three completed evaluations use the same 100 task IDs, prompts, task
hash, base revision, precision, greedy generation configuration, and 192-token
budget. Checkpoint hashes are retained in adapter evaluation provenance.

The 100 prompts are distinct from registered training and prior development
prompts under normalized equality and the registered word-5-gram screen.
No training or checkpoint selection occurred during evaluation. The task file
hash is `db4c9543facb990be6644d74eade05d0d84ede837d22ddfdcef73b4fc547255d`.

Summed generation-call times were 20.922528 s (untouched), 19.440721 s (repeated),
and 19.938787 s (unique). These exclude model loading and scoring and differ in
generated lengths; they are not evidence of intrinsic speedup.

This suite was constructed after training and uses five correlated task
families. It is a post-training holdout, not an independently sealed benchmark
or a broad capability score. It does not measure factual explanation,
summarization fidelity, open-ended reasoning, uncertainty, or executable code.
One training seed and template-based data do not support significance or an
observer-specific improvement claim.

## Artifacts

- Freeze and overlap audit: `research/results/native_holdout100_frozen/`
- Raw generations: `research/results/native_holdout100_{baseline,repeated,unique}/baseline.json`
- Detailed scores/gains/regressions: `research/results/native_holdout100_comparison_detailed.json`

## Decision

Native LoRA improves strict compliance on unseen JSON-record prompts, but does
not preserve all content successes or repair sorting/reasoning. Before scaling,
the next intervention must target these measured content regressions and be
tested against the existing standard-LoRA baseline. This holdout is now exposed;
if it guides another intervention, it becomes development data and a new final
holdout is required. No further training is launched by this evaluation.
