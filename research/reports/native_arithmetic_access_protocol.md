# Arithmetic access diagnostic v1 — frozen before new inference

## Inputs and endpoints

Use the exposed 40 arithmetic cases from retention-transfer v2. Compare the
pinned untouched model and the three ordinary 128-update adapter seeds. These
are the ordinary comparator endpoints, not best-performing selections. No
training, model selection, or changes to original scores.

## Fixed inference interventions

1. **Original:** rerun original greedy 192-token generation and require exact
   token agreement with stored native/ordinary results. Also record original
   first-token logits for offline correct-answer probability/rank diagnostics.
2. **Numeric-only decoder:** at the first assistant position choose the highest
   logit vocabulary token that decodes to a canonical whole decimal integer
   matching ASCII `-?[0-9]+` and re-encodes to that same single token. Then emit
   EOS. Support is built from the entire vocabulary without task facts or
   correct answers. All expected answers must be representable; otherwise abort
   the registered numeric-control comparison, without filtering examples.
   This constrains both format and output length, not computation alone. It can
   choose a token that would start a worked response rather than a final answer.
3. **Fixed answer prefix:** append exactly `Final answer: ` to the assistant
   start, then normal greedy generation for 192 tokens. No numbers supplied.
   Report full assistant response native exact/content using unchanged validators.
   Additionally report continuation-only correctness as a labeled conditional
   diagnostic, never as native exact compliance after silently stripping a prefix.
4. **Reasoning request:** append this fixed instruction to the user prompt:
   `For this diagnostic, replace the answer-format instruction above: show the
   arithmetic steps and finish with a single \boxed{integer}.` No new facts or
   targets. Greedy 192-token generation; report content and format separately.
5. **All adapters off:** scale-zero original-prompt generation for each ordinary
   seed, requiring exact token parity with stored untouched generation. Restore
   scales with `finally`. This is a causal parameter intervention/engineering
   control, not a proposed inference improvement or speed optimization.

All runs: same pinned tokenizer, native FP16 base and FP32 LoRA on accelerator
(FP32 CPU fallback), batch one, greedy, 192-token budget for unrestricted runs.
No calculators, answer-dependent logits masks, or native-generated reasoning
prefixes. Correct-answer IDs are used only for readout/scoring after selection.
Expected answers never enter input IDs, generation processors, or prefix text.

## Measurements and interpretation

Save full prompts and inputs, generated token IDs, assistant prefills, full and
continuation outputs/scores, EOS/budget status, runtime, hashes, numeric support,
first-token correct probabilities/ranks, and paired gains/losses by case/seed.
Hash protocol, tasks, source evaluations and checkpoints before inference.

Recovery means a previously lost **confirmed-content** success becomes correct
under that intervention. Report every regression as well. For reasoning/prefix
controls token budgets match, actual generation lengths/compute do not. Numeric
decoding has two emitted tokens by construction; never imply equal inference
compute or a guarantee over multi-token integers/general tasks.

A native numeric-only failure with native worked success shows a limitation of
that short decoder, not proof that all short answering must fail. Recovery of
adapted content under reasoning/prefix changes supports available conditional
capability in the adapted parameters, not unique attribution to the mechanism
that originally caused damage. No recovery does not prove irreversible loss.
Adapter-off parity establishes adaptation-mediated differences, not a unique
head/neuron, and does not imply useful adapters should simply be discarded.
Report per-seed outcomes; no pooled-seed significance or 100/100 claims.
