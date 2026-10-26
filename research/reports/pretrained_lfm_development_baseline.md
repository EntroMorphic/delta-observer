# Untouched LFM development baseline

**Correction:** the initial run used duplicate BOS tokens and is not a valid
native-chat baseline. Artifacts are retained. See
`pretrained_evaluation_red_team.md` for the corrected rerun and expanded suite;
both use `add_special_tokens=False` after chat rendering. The corrected
14-task rerun still scores 0/5 strict checks, but some generated answers differ.

## Scope

The unmodified `LiquidAI/LFM2.5-350M` checkpoint at
`9e6c6ccf47cd318696e137d381a7ded8fe4df09f` was evaluated on 14 new development
tasks across seven categories. It contains 354,483,968 parameters. No training,
compression, or architecture substitution occurred.

Artifacts: `research/results/pretrained_lfm_development/baseline.json` contains
the task-file hash, token IDs, responses, EOS status, measured generation time,
model revision, and per-task validation. The suite is development-only and has
not been certified uncontaminated relative to upstream pretraining or the old
UltraChat corpora. It must not become a sealed confirmatory test.

## Automatic checks

Five strict automatic checks scored 0/5:

- Sorted JSON: output used Markdown, lost the negative sign, and placed 4 last.
- JSON record: object content was correct but Markdown violated the format.
- Pencils: arithmetic correctly reached 33, but explanation violated the
  integer-only instruction.
- Ordered boxes: returned `blue` rather than `violet`.
- Spanish translation: returned `cat` rather than `gato`.

This measures exact task compliance, not zero general capability. Future
evaluation should separately score content correctness and format compliance,
with both scoring rules frozen before adaptation.

## Open-ended inspection, not blinded scores

Nine responses remain pending rubric-based review. Non-blinded development
inspection shows coherent short explanations, plausible summaries, an incorrect
clamp implementation (omits the in-range return), and a plausible even-counting
function. Code was not executed. The Moon explanation confuses the role of
Earth's solar orbit; the envelope response is verbose and reaches the token
limit rather than directly answering the uncertainty question.

Do not treat these observations as independently judged benchmark scores.
The baseline is functioning but fragile; the first improvement study should
track both correcting these weaknesses and retaining already-correct behavior.

## Next evaluation work, before training

1. Add separate content and strict-format validators, without changing the
   already-recorded strict results.
2. Build larger development tasks with enough successes and failures to avoid
   floor/ceiling artifacts and support per-category comparisons.
3. Establish an isolated executable-code runner and blinded open-ended review.
4. Freeze a separate final test set, exclusion policy, budget matching, seeds,
   success/non-inferiority thresholds, and checkpoint-selection rules.

No adaptation gain or generalization claim is supported by this baseline.
