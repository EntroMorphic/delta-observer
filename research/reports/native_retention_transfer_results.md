# Retention transfer v2 — failed fresh-transfer gates

This experiment is preregistered in `native_retention_transfer_protocol.md`.
**The projected candidate fails the preregistered fresh-transfer gates in all
three seeds.** All twelve fixed training endpoints completed before evaluation
began; all thirteen evaluations completed before results were opened. No model
is approved as a replacement. The newly opened set is now diagnostic for future
changes, not an untouched test for another intervention.

The most consequential finding is not the small projection/control difference:
untouched arithmetic confirmed content is **39/40**, but every adaptation arm
falls to **5–10/40**. Native exact scores improve because responses change from
worked explanations to bare integers, yet those integers are mostly wrong.
This is a large measured capability regression, not a content-parser improvement.
No scorer was changed.

## Frozen design

- Three seeds (0, 1, 2), four arms, 128 fixed updates per endpoint.
- Ordinary arithmetic updates; four-example gradient-span projection; native
  retention replay (coefficient 0.1); angle/norm-matched random-direction control.
- All arms process identical arithmetic/protection rows and model passes.
  Runtime is measured, not assumed equal. The projection removes the full
  protected-gradient span, including helpful directions; it differs from the
  earlier one-sided one-key diagnostic.
- 100 fresh tasks: 40 paired arithmetic, 20 JSON, 20 ordering, ten sorting, ten
  translations. All endpoints finish before baseline or candidate evaluation.
- One native-wrong ordering reference is deliberately kept without filtering.
  Two native JSON references have correct content but fenced formatting.
  Protecting native continuations is not correctness training.

## Completed data audit

`research/results/native_retention_transfer_v2_data_audit.json` independently
verifies all target/protection tokenizations and masks, the exact 128-group
source pass, arithmetic labels, no exact normalized prompt overlap against
seven historical repository suites, and no supervised arithmetic-fact overlap.
This is an exact-overlap audit, not a semantic-near-duplicate guarantee.

Frozen manifest SHA-256:
`03463ac1f2741e3a725ced3d669da81334849ecc562103f66d7a8e808c6b0887`.
Repository copies: `research/results/native_retention_transfer_v2_freeze/`.

Runtime artifacts live under the approved external artifact root in
`native-retention-transfer-v2/`. Executables:
`research/lfm/native_retention_transfer.py` and
`research/lfm/audit_retention_transfer.py`.

## Preregistered approval criteria

For every seed, zero losses of native confirmed retention successes **and**
zero losses of same-seed ordinary arithmetic exact successes. Projection-
specific evidence also requires strictly fewer retention losses than ordinary
and random controls in every seed. Ties are inconclusive. Report replay and
all paired examples regardless of these gates. Existing narrow content parsers
remain unchanged. No best-seed selection or pooled-seed significance claims.

## Completed training audit

All three seeds have distinct initial adapters; the four arms match initial
adapters within each seed, all forward/backward example budgets match, and
all frozen base hashes match before/after and across arms. Checkpoint tensors
are finite and match training metadata. No zero-direction projection occurred.
Full trajectories: `research/results/native_retention_transfer_v2_training/`.
Independent audits: `native_retention_transfer_v2_training_audit.json` and
`native_retention_transfer_v2_checkpoint_audit.json` under `research/results/`.

Protected CE increases slightly on many projected steps despite first-order
orthogonality. Across the three seeds, the four initial CEs are
`[0.043155, 0.045244, 0.037627, 0.017806]`; projected endpoints are:

| Seed | JSON 0 | JSON 1 | Ordering 0 (native wrong) | Ordering 1 |
|---|---:|---:|---:|---:|
| 0 | 0.043909 | 0.046018 | 0.040263 | 0.019329 |
| 1 | 0.043722 | 0.045735 | 0.039525 | 0.018886 |
| 2 | 0.043876 | 0.045876 | 0.040305 | 0.019417 |

Maximum absolute realized protected-gradient dot product is below 8e-10 in
all projected runs; ideal double-precision dots are below 1.3e-18. This is
consistent with first-order protection, not exact finite-step loss preservation.
Do not attribute all residual drift to a unique curvature mechanism without a
separate test. The held-out gate still decides transfer.

Measured training loop seconds (excludes initialization/frozen hashing and
final serialization, includes all gradient/projection/loss-measurement passes):

| Arm | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Ordinary | 104.621 | 115.310 | 117.355 |
| Projected | 106.257 | 127.126 | 110.935 |
| Replay | 112.738 | 119.764 | 111.781 |
| Random | 114.157 | 120.407 | 109.426 |

These are measured equal-model-pass controls, not equal runtime or an inference
speed comparison. No held-out performance follows from training loss alone.

## Fresh held-out results

All counts below use the frozen original validators. The untouched baseline's
39 arithmetic successes are unambiguous boxed final answers, not exact
instruction compliance. Each adapted arithmetic response is a bare integer;
the wrong integers remain `content_unresolved` under the frozen narrow parser,
not confirmed successes. The post-opening integer taxonomy below describes
these failures without changing the scores.

| Arm | Seed | Native exact /100 | Confirmed content /100 | Format /100 | Arithmetic content /40 | JSON content /20 | Ordering exact /20 | Spanish exact /10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Untouched | — | 7 | 66 | 28 | **39** | **20** | 5 | 2 |
| Ordinary | 0 | 18 | 35 | 69 | 6 | 17 | 7 | 5 |
| Projected | 0 | 17 | 34 | 69 | 5 | 17 | 7 | 5 |
| Replay | 0 | 19 | 37 | 72 | 6 | 18 | 9 | 4 |
| Random | 0 | 18 | 35 | 69 | 6 | 17 | 7 | 5 |
| Ordinary | 1 | 19 | 36 | 69 | 7 | 17 | 7 | 5 |
| Projected | 1 | 19 | 37 | 69 | 7 | 18 | 7 | 5 |
| Replay | 1 | 21 | 40 | 72 | 7 | 19 | 9 | 5 |
| Random | 1 | 19 | 36 | 69 | 7 | 17 | 7 | 5 |
| Ordinary | 2 | 21 | 39 | 71 | 10 | 18 | 7 | 4 |
| Projected | 2 | 19 | 37 | 71 | 8 | 18 | 7 | 4 |
| Replay | 2 | 22 | 40 | 72 | 10 | 18 | 8 | 4 |
| Random | 2 | 22 | 40 | 71 | 10 | 18 | 7 | 5 |

Signed sorting remains 0/10 strict and confirmed content in every model. JSON
strict remains 0/20 in every model: correct contents still have fences. Both
operations are correct for 0/20 pairs in the untouched model **under native exact
scoring**, versus 1/20 in every adapter. Under confirmed-content scoring,
untouched has **19/20** both-operation pairs correct versus **1/20** in every
adapter. The exact-format result must not be confused with numerical accuracy.

### Gates and individual losses

| Seed | Projected native retention-content losses | Arithmetic exact losses vs same-seed ordinary | Retention gate | Arithmetic gate | Direction-specific gate |
|---|---|---|---|---|---|
| 0 | `fresh-record-02`, `fresh-record-06`, `fresh-record-11` | `fresh-subtract-07` | fail | fail | fail |
| 1 | `fresh-record-06`, `fresh-record-11` | none | fail | pass | pass |
| 2 | `fresh-record-06`, `fresh-record-11`, `fresh-translation-07` | `fresh-subtract-08`, `fresh-subtract-15` | fail | fail | fail |

Projection never gains an arithmetic exact success relative to ordinary. Its
only confirmed-content gain vs ordinary is `fresh-record-02` in seed 1.
Same-seed random and ordinary predictions have identical arithmetic scores in
all three seeds; random additionally repairs `fresh-translation-07` in seed 2.
Projection-specific evidence is thus not reproducible across seeds.

Replay has better JSON/ordering counts than projection in several cells, but
still loses native content and leaves the arithmetic collapse unrepaired. It
is not approved either. No winner is selected by aggregate count or best seed.

The gates intentionally protect held-out **non-arithmetic** native successes
and compare arithmetic against ordinary. They do not themselves demand native
arithmetic-content preservation. That preregistered limitation is material:
do not hide the much larger arithmetic regressions behind these narrow gates.
Full paired native-content losses, all splits and categories are preserved in
`native_retention_transfer_v2_comparison.json`.

## Verified paired examples

1. `fresh-add-00`: 18 boxes × 6 tokens, then gains 4; expected **112**.
   Untouched writes `18 × 6 = 108; 108 + 4 = 112` and boxes 112. Ordinary,
   projected and replay in the inspected examples instead emit **108**.
2. `fresh-subtract-01`: 16 × 4, then loses 7; expected **57**.
   Untouched computes and boxes 57; the same adapted examples emit **64**.
3. `fresh-subtract-07`, seed 0: expected **68**. Ordinary emits 68;
   projected emits **69**. Equal arithmetic totals would not excuse this loss.
4. `fresh-record-06`: baseline correctly emits fenced
   `{"label":"otter","enabled":true}`. Ordinary/projected seed 0 emit
   `{"otter":true}` instead; replay seed 1 repairs it.
5. `fresh-record-11`: baseline correctly emits fenced
   `{"title":"trowel","active":false}`. The inspected adapted examples
   emit `{"trowel":false}`. Protection of `label`/`enabled` does not establish
   preservation of a new schema.

These are stored raw generations, not manually rescored outcomes.

## Post-opening arithmetic error taxonomy (diagnostic only)

Whole-response integer classification: unchanged product means `a*b` instead
of `a*b±c`; opposite operation means the other true operation result. This
classification uses task facts for **analysis**, never for inference, never to
change the frozen content parser. No evaluation was truncated.

| Arm | Seed | Correct /40 | Unchanged product | Opposite operation | Other wrong integer | Identical add/subtract full responses /20 |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 0 | 6 | 18 | 3 | 13 | 15 |
| Projected | 0 | 5 | 20 | 3 | 12 | 14 |
| Replay | 0 | 6 | 18 | 3 | 13 | 15 |
| Random | 0 | 6 | 18 | 3 | 13 | 15 |
| Ordinary | 1 | 7 | 15 | 3 | 15 | 15 |
| Projected | 1 | 7 | 15 | 3 | 15 | 15 |
| Replay | 1 | 7 | 15 | 3 | 15 | 15 |
| Random | 1 | 7 | 15 | 3 | 15 | 15 |
| Ordinary | 2 | 10 | 10 | 4 | 16 | 12 |
| Projected | 2 | 8 | 11 | 3 | 18 | 13 |
| Replay | 2 | 10 | 10 | 4 | 16 | 12 |
| Random | 2 | 10 | 10 | 4 | 16 | 12 |

This directly measures many lost second-operation outputs. It does not prove
the model internally computes multiplication correctly and then stops a
particular circuit, nor that extra output reasoning tokens are necessary.
Output-path suppression, shortcut learning and numerical retrieval remain
candidate mechanisms. Previous failed worked-replay experiments also preclude
declaring “add reasoning traces” a demonstrated remedy.

## Interpretation and next discriminating question

- The local v1 separability result remains valid on its measured cases, but
  does **not** extend to reliable fresh retention in this v2 test.
- Four-example gradient-span constraints preserve those example losses closely,
  but do not cover all native schema/value bindings or arithmetic behaviors.
- This test changes protection objective (full native continuations vs one key),
  geometry (full span vs one-sided), duration (128 vs 512), and evaluation. It
  rejects this registered multi-example candidate, not every possible retention
  constraint. No isolated cause of the difference from v1 is established.
- Native numerical capability is demonstrably present on these fresh prompts;
  adaptation destroys much of it while making answers shorter/more compliant.
  Missing pretrained capacity is not an adequate explanation of this result.
- Schema retention alone was too narrow. Before another broad training change,
  isolate whether the same native arithmetic result survives under an
  independently imposed output-format intervention, and trace the onset of
  product-only responses with same-state causal controls. That would distinguish
  lost computation from lost access to an existing computation. This is a
  proposed next experiment, not a measured cause or an approved fix.

## Artifacts and audit trail

- `research/results/native_retention_transfer_v2_comparison.json`
- `research/results/native_retention_transfer_v2_eval/`: all 13 raw evaluations.
- `research/results/native_retention_transfer_v2_training/`: all 12 trajectories.
- `research/results/native_retention_transfer_v2_output_audit.json`: independent
  frozen-score reproduction plus every arithmetic example's response taxonomy.
- Freeze, implementation hashes and independent data/training/checkpoint audits
  listed above. Adapter checkpoints remain in the approved external artifact root.

Executables: `native_retention_transfer {prepare,train,evaluate,summarize}`,
`audit_retention_transfer`, `audit_retention_transfer_outputs`.
No additional training or alternate decoding was used to produce this
comparison. Subsequent inference-only controls on these now-exposed arithmetic
cases are reported separately in `native_arithmetic_access_results.md`; they
do not change any frozen score or rehabilitate this rejected candidate.
