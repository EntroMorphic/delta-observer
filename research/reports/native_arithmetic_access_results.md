# Arithmetic access v1: partial recoverability, no reliable remedy

## Outcome

The inference-only controls support **conditional recoverability of some
adapted arithmetic**, not universal erasure of native numerical capability.
They do not establish a single ultimate cause, nor restore reliable exact
answering. The adaptation candidates remain rejected.

This is the preregistered post-opening diagnostic in
`native_arithmetic_access_protocol.md`, not a fresh confirmation. No additional
training occurred. Original retention-transfer evaluations and scores remain
unchanged. All 40 exposed arithmetic cases and all three ordinary seeds are
reported; no best seed or favorable subset was selected.

## Measured results

Confirmed-content counts use the original exact/boxed-integer parser. The
fixed-prefix column is explicitly **continuation-only correctness**, not full
assistant native exact or parsed-content compliance. Its full responses start
with `Final answer: ` and score 0/40 under both frozen original contracts.

| Model | Original content /40 | Numeric-only decoder content /40 | Fixed-prefix continuation-only exact /40 | Reasoning-request content /40 | All-adapters-off original content /40 |
|---|---:|---:|---:|---:|---:|
| Untouched | 39 | 0 | 1 | **40** | not applicable |
| Ordinary seed 0 | 6 | 6 | 14 | 15 | **39** |
| Ordinary seed 1 | 7 | 7 | 12 | 5 | **39** |
| Ordinary seed 2 | 10 | 10 | 15 | 34 | **39** |

Original and numeric-only native exact scores equal the adapter content counts
above, but untouched original/numeric exact are both zero. Reasoning-request
responses are not bare integers and score zero under the original exact-format
contract. Their output instruction has deliberately changed; this is not a
claim that they obey the original bare-answer request.

Some non-successes under the reasoning request are malformed boxed outputs or
otherwise unresolved by the frozen parser. The table counts confirmed content,
not a new semantic review of every non-success. No parser was broadened to make
recovery look better. No run hits the 192-token budget.

### Paired recovery and new regressions

| Intervention | Seed | Confirmed-content gains vs own original | Confirmed-content losses vs own original |
|---|---:|---:|---:|
| Numeric-only | 0 | 0 | 0 |
| Numeric-only | 1 | 0 | 0 |
| Numeric-only | 2 | 0 | 0 |
| Fixed prefix, **conditional continuation metric only** | 0 | 10 | 2 |
| Fixed prefix, **conditional continuation metric only** | 1 | 8 | 3 |
| Fixed prefix, **conditional continuation metric only** | 2 | 8 | 3 |
| Reasoning request | 0 | 15 | 6 |
| Reasoning request | 1 | 5 | 7 |
| Reasoning request | 2 | 28 | 4 |

Every seed incurs regressions under both alternative prompting controls. The
reasoning request loses **all** original confirmed successes in seeds 0 and 1,
despite recovering different examples. Seed 2's large gain is not an all-seed
guarantee. All individual gain/loss IDs are in `summary.json` and
`native_arithmetic_access_v1_artifact_audit.json`, not just matched counts.

Untouched reasoning recovers its sole unconfirmed original case
`fresh-subtract-00` without losing any of the 39 original successes. Untouched
numeric-only loses all 39. Untouched fixed-prefix conditional correctness
retains one original success and loses 38. These contrasting outputs occur
without any parameter updates.

## Concrete same-fact example

`fresh-add-00`: 18 boxes, 6 tokens each, then gains 4. Correct answer: **112**.

| Model/control | Actual output (or faithful excerpt) |
|---|---|
| Untouched original | `18 × 6 = 108; 108 + 4 = 112`, followed by boxed 112 |
| Untouched numeric-only | `18` |
| Untouched fixed prefix | `Final answer: 20` |
| Untouched reasoning request | Computes 108, adds 4, boxes 112 |
| Ordinary, all three seeds, original | `108` |
| Ordinary, all three seeds, numeric-only | `108` |
| Ordinary, all three seeds, fixed prefix | `Final answer: 58` |
| Ordinary seeds 0 and 2, reasoning request | Computes 108, adds 4, boxes 112 |
| Ordinary seed 1, reasoning request | Malformed `\boxed220}` |

No correct answer or arithmetic work was supplied as input to any control.
The alternative prompts contain only fixed words, not task-specific hints.

## Engineering and negative controls

- Original generations match all **160/160** stored original model/case token
  sequences, not merely their score counts.
- Scale-zero all-adapters-off generations match **120/120** stored untouched
  model/case token sequences. Adapter scales are restored in `finally`.
- Numeric support is independently reconstructed from the whole vocabulary:
  **1,000** canonical single-token integers (0–999). It is not a per-question
  choice set. All expected answers are representable. The decoder selects the
  highest-logit allowed token, with no expected-answer input, then emits EOS.
- The numeric decoder's failure in the untouched model is often selecting the
  beginning of a worked answer (e.g. 18), not evidence all short decoders fail.
- An independent artifact audit reconstructs all **760 case/condition** inputs,
  continuations, full responses, scores, manifests and source hashes. It verifies
  **280 token-parity** cases. This is a separate code audit, not independent
  blinded model evaluation.
- Checkpoint hashes, source evaluations, protocol and inference implementation
  remain unchanged. Correct-answer IDs are used for readout/scoring, not masks,
  prefixes or generation. This audit performs no new optimization.

## Inference work and timing

All unrestricted conditions have the same maximum token budget, not the same
actual output length or compute. Each measured case includes a separate uncached
first-token readout forward pass in addition to generation. Numeric decoding
instead uses that forward pass plus fixed EOS emission; the two-token output
does not imply that EOS was sampled or independently recomputed.

| Model | Original generated tokens | Numeric-only tokens | Prefix generated tokens (excludes supplied prefix) | Reasoning-request tokens |
|---|---:|---:|---:|---:|
| Untouched | 1,522 | 80 | 80 | 1,839 |
| Seed 0 | 80 | 80 | 80 | 925 |
| Seed 1 | 80 | 80 | 80 | 463 |
| Seed 2 | 80 | 80 | 80 | 1,751 |

Recorded aggregate inference seconds, including the extra readout pass:

| Model | Original | Numeric-only | Prefix | Reasoning request |
|---|---:|---:|---:|---:|
| Untouched | 15.041638 | 0.915440 | 2.638231 | 18.522849 |
| Seed 0 | 2.431552 | 0.938920 | 2.718156 | 11.004381 |
| Seed 1 | 2.427385 | 0.937617 | 2.646735 | 6.739612 |
| Seed 2 | 2.433363 | 0.941997 | 2.654600 | 18.673065 |

These are batch-one diagnostic measurements, not equal-compute comparisons or
speedups at preserved accuracy. All-adapters-off still executes adapter
arithmetic at zero scale; no deployment efficiency is implied.

## Bounded causal conclusion

1. Turning off the adapters exactly restores native behavior. The recorded
   changes are mediated by active adapter computation, not changed base weights
   or a different original evaluation pipeline.
2. Changing the output/request context without updating parameters recovers
   many lost examples, especially seed 2. An explanation asserting complete
   destruction of arithmetic capability in every adapted model is inconsistent
   with these observed recoveries.
3. Format restriction alone can destroy observed numerical accuracy in this
   untouched model. It does not isolate the original training mechanism: numeric
   masking, assistant prefilling and changed reasoning instructions are distinct
   interventions, not identical executions differing only in final formatting.
4. No inference control provides broad, regression-free recovery. The seed-1
   deterioration and all paired regressions remain part of the result.

Not established: a unique internal numerical circuit, whether serial reasoning
tokens are necessary, whether a forced intermediate state would repair the
adapted graph, whether computation vs retrieval explains every correct answer,
or whether training on reasoning text would improve transfer. Existing failed
worked-replay training remains evidence against simply assuming that remedy.

The next discriminating step is a **same-history adapter-on/off causal fork**:
measure the two models' decisions under identical, non-answer-bearing generated
prefixes, with self-controls and source-prefix provenance. Avoid native worked
prefixes containing the correct arithmetic result, which would leak an answer.
That diagnostic has subsequently completed and is documented separately in
`native_same_history_results.md`; the access-control results above are unchanged.

## Artifacts

- `research/results/native_arithmetic_access_v1/access.json`: complete run.
- Same directory: frozen manifest, per-model raw outputs and `summary.json`.
- `research/results/native_arithmetic_access_v1_artifact_audit.json`: independent
  score/input/token reconstruction and conditional-prefix paired examples.
- Executors: `audit_native_arithmetic_access.py`,
  `audit_arithmetic_access_artifacts.py`; tests: `test_native_arithmetic_access.py`.
