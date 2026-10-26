# Native LFM LoRA: engineering gates closed and first pilot measured

**Atomic-audit correction:** the original length-bucket builder selected
`clock` as the item value in every response in both arms. Distinct prompt and
quantity counts are not varied-item supervision. Results below are preserved,
but do not test broad or item-value diversity. Builder v2 repairs coverage and
changes the token budget; the reproduction commands below now build v2, not
the original v1 dataset. See `native_atomic_bedrock.md` for causal diagnostics.

## Outcome

Native-model adaptation is operational. Both 128-update exploratory arms
improved JSON-record format compliance without losing any baseline automatic
success on development v2. There is **no demonstrated unique-data advantage**,
observer-specific advantage, reasoning gain, or confirmatory generalization
result.

## Engineering evidence

- Untouched base: LFM2.5-350M, pinned commit, 354,483,968 frozen parameters.
- Adapters: 835,584 trainable parameters in native attention/conv projections.
- Smoke test: exact zero-init logit parity, finite gradients, nonzero update,
  no frozen gradients, unchanged frozen-weight SHA-256, exact reload parity.
- Four uninterrupted updates versus two-plus-two resumed: maximum adapter
  difference 0.0 and identical loss/row histories. No stochastic dropout is used.
- Response masks include the entire answer and EOS; no duplicate BOS, padding,
  or sequence truncation. Rows are re-tokenized and hashes verified before use.
- Engineering smoke/resume adapters were not used to initialize either pilot.

Evidence: `research/results/native_lfm_lora_smoke.json` and
`research/results/native_lfm_resume_audit.json`.

## Matched pilot

Protocol: `research/reports/native_lora_pilot_protocol.md`.

Both arms use 128 exposures, 9,728 input tokens, 1,408 supervised tokens, and
identical per-row lengths, optimizer settings, initialization seed, and update
count. Unique uses 128 distinct inventory prompts; repeated uses 16 repeated
eight times. Labels are deterministic JSON extraction, not model-generated.
This is variant coverage within one task family, not broad dialogue diversity.

| Development metric | Untouched | Repeated | Unique |
|---|---:|---:|---:|
| Strict automatic success | 2/35 | **10/35** | 9/35 |
| Strict format category | 0/14 | **7/14** | **7/14** |
| JSON-record subset | 0/7 | **7/7** | **7/7** |
| Separate content-parser success | 14/21 | 14/21 | 14/21 |
| Previously successful automatic checks lost | — | 0 | 0 |

All seven record prompts now produce correct bare JSON instead of fenced JSON.
Sorting remains wrong. Nine manual tasks remain unscored; code is not executed.
The strict and content columns use different subsets, not interchangeable
denominators. Both arms pass the frozen exploratory positive-signal gates.

Training-loop wall times were 19.812914 s (repeated) and 20.502821 s (unique).
These include per-step checkpoint writes, but exclude initial model loading and
data validation; they are not end-to-end experiment times. Measured peak
process RSS was 2,529,198,080 and 2,529,689,600 bytes respectively. Process RSS
does not measure peak Metal allocation or total unified-memory use.

Full measured comparison: `research/results/native_lfm_pilot_comparison.json`.
Per-task outputs and generation settings are retained in
`research/results/native_lfm_pilot_{repeated,unique}_eval/baseline.json`.

## Reproduction

Use a fresh output directory for each invocation; existing results are refused.

```bash
.venv-lfm/bin/python -m research.lfm.native_lora_smoke --output PATH/smoke
.venv-lfm/bin/python -m research.lfm.build_native_pilot_data --output PATH/data
.venv-lfm/bin/python -m research.lfm.train_native_lora \
  --data PATH/data --arm repeated --max-steps 128 --output PATH/repeated
.venv-lfm/bin/python -m research.lfm.train_native_lora \
  --data PATH/data --arm unique --max-steps 128 --output PATH/unique
.venv-lfm/bin/python -m research.lfm.evaluate_pretrained_development \
  --tasks research/lfm/pretrained_development_tasks_v2.json \
  --adapter PATH/repeated/checkpoint.pt --output PATH/repeated-eval
```

The environment versions are recorded in the smoke artifact. The native trainer
does not use the legacy paired Observer.D trainer.

## Remaining gates for a larger study

Training feasibility, deterministic resume, masking, budget matching, data
hashing, and native adapter evaluation are now demonstrated. Before a broader
claim, lock a genuinely held-out final set and independently scored evaluation,
run multiple seeds, broaden data beyond one template, and predeclare per-category
retention margins. Blind review and isolated code execution remain blockers for
claims in their categories. No compressed deployment or observer advantage is
claimed from this standard LoRA result.
