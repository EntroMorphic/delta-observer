# Retention transfer v2 — preregistration

Fixed before data construction or model outputs. This experiment follows the
local result in `native_update_conflict_results.md`; no universal cause is assumed.

## Design

- Pinned native LFM2.5-350M, standard rank-8/alpha-16 adapters, FP32 AdamW
  lr 1e-4, decay .01, clip 1. Embeddings, head, FFNs and base stay frozen.
- Seeds 0, 1, 2. Exactly 128 updates: the first complete pass of the existing
  contrast arm (512 arithmetic sequences, 1,024 supervised target tokens).
  This is deliberately shorter than the earlier 512-update experiment. No
  duration, coefficient, checkpoint, seed, or example selection after results.
- Four fixed protection prompts: two JSON schemas and two ordering problems.
  Protection targets are the untouched model's complete greedy continuations,
  excluding terminal EOS and adding one EOS through `supervised_chat`. No
  correctness filtering. All continuations, scores, tokens and lengths saved.
- Four arms: ordinary target-only AdamW; projection; retention replay with
  target CE + 0.1 * mean protected CE; random-direction control.
- Every arm computes every protected example's gradient before every target
  update, even when unused. All arms also execute the same projection and
  random-control construction. This matches examples and model forward/backward
  counts, not wall-clock time or effective optimization objectives. Measure
  input/supervised token processing, passes, runtime and frozen hashes.
- Projection: remove the actual post-AdamW displacement's entire component in
  the span of the four protected gradients, then restore displacement norm.
  This protects each example to first order, unlike a mean-gradient constraint
  that can conceal cancellation. It also removes helpful components: an explicit
  conservative change from the earlier one-sided one-example method. No finite-
  step guarantee. If the remaining direction is zero, take zero and report it.
- Random control: independent seeded random direction orthogonal to raw delta,
  mixed with raw delta to match the projection's angle and norm at that arm's
  current state. This tests direction specificity, not just tiny perturbations.
  Adam moments always follow that arm's objective, not projected deltas.

## Fresh locked evaluation

100 tasks: 40 arithmetic (20 new triples, each addition/subtraction under new
wording), 20 JSON (ten same-key/different-value, ten new-key schemas), 20 ordering
(ten familiar structure/new identifiers, ten different wording/new identifiers),
ten signed sorting, ten Spanish number words. Arithmetic triples exclude both
original supervised and diagnostic triples. Exact normalized prompt overlap
against all repository task-suite JSON files and protected/source prompts is
rejected. This is new machine-generated testing, not independent blinded review.

Build and hash evaluation and training/protection manifests before training.
Generate baseline and candidate evaluations only after **all twelve training
endpoints** finish. Neither trainer nor retention-reference generation loads
evaluation cases. No inspection of fresh test responses until all evaluations
finish. Test generators are visible; there is no sealed independent evaluator.
Once opened this set becomes diagnostic for future changes.

Unchanged validators: native exact, optional whole-response JSON-fence parser,
and existing narrow arithmetic content parser. Wrong bare integers remain
unresolved content, never successes. No calculator at inference or answer-choice
decoder. Evaluation greedy, FP16 base/FP32 adapters on accelerator, 192 tokens.

## Fixed gates and reporting

For **every** seed the projected arm must (a) lose zero untouched-model confirmed
retention successes, and (b) preserve every target-only arithmetic exact success
(not just its count). Both gates required; report failures individually.
Projection-specific evidence additionally requires strictly fewer held-out
retention losses than **both** ordinary and random controls for every seed;
ties or zero-loss controls are inconclusive, not positive evidence. Replay is
an explicit extra-retention-objective comparator, not an inferior straw man.

Report all four arms per seed, strict/format/parsed content per category and
split, paired gains/losses vs native and same-seed ordinary, operation pairs,
protected finite-step losses, all endpoint artifacts, runtimes and compute
counts. No pooling seeds as independent test samples; no best-seed selection.
Fresh facts/templates are limited transfer evidence, not broad capability or
100/100 certification. Numerical generalization remains separately unresolved.
