# REFLECT: conclusions, challenged assumptions, and resolved tensions

## Core insight

The next leverage is to train and test **equivalence and contrast of meaning**,
not merely accumulate examples, epochs, or adapter rank.

This is a candidate design principle supported by observed sensitivities, not
a proven training improvement. Its causal prerequisites must be tested.

## Why-chain checks

**Why is arithmetic failing?** Compounds fail while selected primitives work.
**Why suspect a rule rather than missing facts?** More exposures fit the same
compounds, yet performance changes under a different task rendering.
**Why not conclude wording alone causes it?** New renderings also use different
numbers. Cross both factors before attributing the gap.

**Why is ordering failing after mixed training?** The specialist transfers
better than the bundle on the same diagnostic inputs.
**Why not merge the specialist?** A naive sum of adapters has not been measured
and may change both task behaviors.
**Why not assume Hessian guidance prevents this?** The measured derivatives
are six in-sample factor directions, not retained-task curvature in native
effective weight space, and have not predicted update effects on capability.

## Resolved tensions

1. Fit versus transfer: keep high-fit checkpoints as instruments. Use a 2×2
   facts/wording diagnostic before choosing another training intervention.
2. Invariance versus sensitivity: paraphrases must preserve answers, but
   minimal instruction changes must change answers. A paraphrase-only loss can
   teach ignoring instructions; audit both properties with paired prompts.
3. Specialization versus retention: test task-focused updates against an
   identical update schedule with retained-task replay; report optimizer and
   compute differences rather than call every change gradient interference.
4. 100/100 versus validity: lock an untouched final template/fact split before
   training. Do not select or revise it after outputs are observed.

## Assumptions rejected or still open

- Rejected as a default explanation: more rank or input-interface geometry alone
  solves the observed failures.
- Unsupported: current LoRA Hessian measurements imply an intelligence/speed map.
- Open: benefit of FFN adapters versus attention/conv adapters at matched
  trainable-parameter and update budgets.
- Open: whether missing worked-step supervision, operation binding, numerical
  generalization, or their interaction dominates unseen arithmetic cases.
- Open: whether native prompt canonicalization is acceptable as an explicit
  system intervention; it must not be disguised as native-model improvement.

## Post-diagnostic revision — evidence changes the priority

The registered 2×2 crossing is complete. On diagnostic facts, the untouched
model gives parsed-correct explanations on 20/20 under either wording; the
rank-8 adapter gives exact integer answers on only 12/20 under training wording
and 8/20 under diagnostic wording. Removing one exact operand triple already
present in training leaves untouched 19/19, rank 8 at 11/19 and 7/19.

Training-fact wording changes cost only two selected rank-8 successes (20→18),
while changing facts under training wording costs eight (20→12). These fixed
sets are different in more than training membership, so counts are not an
independent causal estimate of memorization. They do falsify the convenient
claim that phrasing alone explains all the transfer gap.

New leading hypothesis: final-only adaptation has not preserved the base
model's demonstrated arithmetic behavior on different numerical examples.
Explanatory answers and integer-only answers differ in computation history;
this is not proof that visible explanations alone cause correctness.

Priority revision: before semantic-orbit scaling, establish retention-first
answer-formatting controls. Keep native exact compliance separate from a
clearly labeled system that extracts the model's own boxed final answer. Such
extraction is not an arithmetic solver or a native-intelligence improvement.
