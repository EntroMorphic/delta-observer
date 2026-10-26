# Pretrained evaluation red-team and corrected development baseline

## Scope

Reviewed the new pretrained development runner, validators, task suite, and
recent paired-training code. This is a targeted review, not certification of
the entire research tree. No training was launched during this review.

## Demonstrated findings

### Critical: duplicate BOS at chat-template tokenization

The template emits BOS, and tokenizing it with default `add_special_tokens`
prepends another. Direct inspection of the pinned tokenizer gives old token
prefix `[1, 1, 6, ...]` versus corrected `[1, 6, ...]`.

Fixed the active evaluation runner to use `add_special_tokens=False`; added a
regression test. Preserved the initial result rather than overwriting it.
The original 14-task run is not a valid native-format baseline. A corrected
rerun is stored separately and still scores 0/5 strict automatic checks.

Earlier instruction extraction and prompt audits contain the same pattern.
Their numbers remain observations under that preprocessing, not clean evidence
about native-chat model capability. Earlier causal/generalization claims need
revalidation before reuse. This review does not establish how much duplicate
BOS contributed to any earlier compressed-model failure.

### High: result clobbering and incomplete-run ambiguity

The runner previously overwrote `baseline.json` silently. It now reserves the
path with exclusive creation before model load and writes subsequent results
via atomic replacement. Failed initialization leaves an explicitly incomplete
record. Existing results are never overwritten by a fresh invocation.

### Medium: nonstandard JSON accepted as valid

Python's JSON parser accepts duplicate object keys (last value wins), NaN, and
Infinity. Strict task scoring now rejects these and checks exact JSON types.
Tests cover duplicate keys, booleans versus integers, floating versus integer
values, Markdown fences, extra keys, and wrong content.

### Medium: suite and generation provenance

Added suite validation for duplicate IDs/prompts, empty suites, missing fields,
and unknown validators. Require an immutable model revision and a positive
generation budget. Save explicit generation settings and Transformers version;
record truncation independently of EOS. Resident weight bytes are explicitly
not runtime peak memory. Runtime peak remains unmeasured, not estimated.

### Measurement: strict compliance conflated with content

Kept strict scores intact and added narrow, separately labeled content parsers:
whole-response JSON with optional Markdown fences, and unambiguous boxed integer
answers. These are not semantic judges. Incidental numbers and conflicting
boxed answers remain unscored rather than being mined for a favorable answer.

## Legacy paired-training risk: quarantined, not certified

The paired trainer relies on equal shard sizes and matching seeded shuffles;
it does not enforce prompt identity itself. Its resume path skips primary
batches without advancing the paired iterator. It also does not record a full
paired-data provenance/budget in the final result. Earlier completed paired
pilots did not resume, and their external prompt-identity checks are recorded,
but this interface must not be reused for the new pretrained study until those
contracts are tested and enforced. The new native baseline does not use it.

## Next progression completed: expanded development baseline

Built a reproducible 44-task suite, retaining the original 14 tasks and adding
30 templated variants. Evaluated the untouched pinned model with corrected chat
tokenization. Artifacts:

- `research/lfm/pretrained_development_tasks_v2.json`
- `research/results/pretrained_lfm_development_corrected/baseline.json`
- `research/results/pretrained_lfm_development_v2/baseline.json`

Strict checks: **2/35**. Separately parsed content checks: **14/21**; these use
a different subset and cannot be compared as interchangeable denominators.
Fourteen responses pass a content parser while failing strict compliance.
Nine manual tasks remain unscored. One envelope response reaches the token
budget without EOS. No aggregate capability score is reported.

Template variants are correlated. Neither 44 tasks nor 35 automatic checks
constitutes an independent statistical sample suitable for a general capability
claim. Open-ended review is not blinded until an independent review workflow
exists. Generated code has not been executed or called verified-correct.

## Next gate

Define a native-model adaptation comparison that preserves content success
while measuring strict instruction compliance. Freeze train/development/final
exclusions, equal token budgets, native update method, seed schedule, and
per-category retention thresholds before training. Keep code execution and
blinded open-ended scoring blocked until their evaluation infrastructure exists.
