# SYNTHESIZE: Observer.D Pre-Collapse Recovery Program

**Method:** Lincoln Manifold Method, pinned to
[`anjaustin/lmm@55719c0a`](https://github.com/anjaustin/lmm/tree/55719c0a2308d1a3f29ca656ed7b6287be33bca2)  
**Phase:** 4 — SYNTHESIZE  
**Status:** Executable experiment specification; R3 remains paused

## Objective

Determine whether Observer.D can preserve input geometry and conditional prompt
signal simultaneously at at least 10× parameter compression, before spending
additional extraction compute.

The experiment begins from the R2 output-distilled checkpoint, where input rank
is 30.4405. It does not begin from the rank-7.33 best-CE checkpoint.

## Architecture boundary

Use the measured rank-96 untied output head:

- Input embedding: `65,536 × 384`
- Output projection: `384 → 96`
- Output embedding: `65,536 × 96`
- Total parameters: 34,301,568
- Teacher parameters: 354,483,968
- Compression: 10.3343×
- Teacher unavailable at deployment; source-free execution unchanged

Do not add an output bias in this pass. It has already failed as an isolated
mechanism and would add a confound.

## Stage 1 — Isolated 512-update arms

All arms use the same R2 checkpoint, SVD initialization procedure, seed, ordered
raw/instruction data, batch size, AdamW settings, sparse KL, and exactly 512
updates. The tied arm is retained as the matched deterioration control.

| Arm | Output head | Attention residual | FFN residual | Purpose |
|---|---|---:|---:|---|
| A | Tied | 1.0 | 1.0 | Matched collapse baseline |
| B | Rank-96 untied | 1.0 | 1.0 | Geometry prevention control |
| C | Rank-96 untied | 1.0 | 0.75 | Test common-mode attenuation |
| D | Rank-96 untied | 1.5 | 1.0 | Test conditional-signal amplification |

Residual scales are fixed for this stage. Learnable gates would let CE silently
return to the measured common-mode optimum and would weaken causal attribution.

## Instrumentation schedule

Measure every arm at steps `0, 64, 128, 256, 512`. Do not infer intermediate
behavior from endpoints.

At every boundary record:

1. Raw and conversational CE, sparse KL, and teacher top-1 agreement.
2. Input-table and output-table effective rank, PC1 variance, norm/exposure
   correlation, and resident bytes.
3. Adam first moment, second moment, implied update geometry, and collapse
   alignment separately for input and output parameters.
4. Prompt-state pairwise cosine, mean-direction energy fraction, and effective
   rank at every recurrent substage.
5. Pairwise first-token JS, unique top-1 count, teacher/student JS, entropy, and
   margins for all eight registered prompts.
6. Greedy 64-token generation with exact repetition and cycle onset.

At step 512 additionally record synchronized forward throughput, autoregressive
tokens per second, peak RSS, parameter bytes, and zero/nonzero KV-cache bytes.

## Stage 1 advancement gates

An experimental arm advances only if all hard gates pass:

- [ ] Parameter compression remains at least 10×.
- [ ] Input effective rank is at least 28.9185, exactly 95% of R2 rank 30.4405.
- [ ] Raw CE penalty versus tied arm is no more than 0.15.
- [ ] Conversational CE penalty versus tied arm is no more than 0.15.
- [ ] Prompt mean-direction energy is lower than rank-96 arm B.
- [ ] Pairwise prompt-output JS is higher than arm B.
- [ ] No registered prompt has an earlier three-repeat cycle than arm B.
- [ ] No corpus hash, mask, BOS, or frozen-prompt exclusion check fails.

The following are diagnostic wins, not advancement substitutes:

- Higher rank without prompt improvement
- Higher prompt JS with earlier degeneration
- Lower CE with worse geometry
- Longer non-repeating but still prompt-inappropriate prose

## Stage 2 — Combined arm only if justified

If both C and D pass Stage 1, create one combined arm E with rank-96 output,
attention scale 1.5, and FFN scale 0.75. If only one passes, advance only that
mechanism. If neither passes, return to REFLECT instead of inventing arm E.

Run the selected arm and matched B control through the full 4,608-update mixed
schedule. Measure at steps `1,024`, `2,048`, `3,072`, and `4,608` using the same
instrument set.

## Stage 2 behavioral gate

Before any R3 extraction resumes:

- [ ] At least 1/8 registered prompts is coherent under blinded review.
- [ ] The frozen reflective-recursion sentinel is not used for model selection.
- [ ] No category has immediate period-1 collapse before step 8.
- [ ] Prompt-output JS does not fall below its step-512 value.
- [ ] Input rank remains at least 28.9185.
- [ ] Raw normalized CE retention remains at least 80%.
- [ ] Compression remains at least 10×.

Passing 1/8 is an architecture-viability gate, not the final 90% behavioral
retention target. Failure remains explicit and triggers reflection.

## Stage 3 — Recursive manifold triggers

Run a new LMM cycle on the indicated subproblem if:

- **Geometry passes, conditioning fails:** recurse on recurrent common-mode
  control and conditional-contrast objectives.
- **Conditioning passes, generation fails:** recurse on decoding-transition and
  autoregressive-attractor dynamics.
- **Behavior improves, coverage remains catastrophic:** recurse on balanced
  vocabulary support and teacher-tail representation.
- **Every arm fails geometry or CE gates:** recurse on the rank-96 factorization
  and optimizer assignment; do not resume R3.
- **One arm passes Stage 2:** then and only then synthesize an R3 extraction and
  training plan.

## Key decisions and rationale

1. **Start from R2:** prevention requires remaining geometry.
2. **Untie at rank 96:** it is the only measured head that stops input-rank loss
   while preserving the strict compression requirement.
3. **Keep sparse KL:** it mitigates long-run rank collapse and broadens token
   support.
4. **Test attention and FFN separately:** both affect prompt contrast, but their
   causal roles differ.
5. **Use fixed scales first:** this prevents CE from hiding the intervention.
6. **Gate on generation:** prompt JS and CE have both produced false optimism.
7. **Keep R3 paused:** more updates currently deepen known failure modes.

## Deliverables

- [ ] Configurable attention and FFN residual scales in `ObserverDLMConfig`.
- [ ] Step-indexed checkpoints and atomic audit snapshots.
- [ ] One machine-readable comparison JSON for arms A–D.
- [ ] Raw continuation records for all eight prompts at step 512.
- [ ] Updated permanent report containing positive and negative results.
- [ ] A loop-back decision recorded against the rules above.

## SYNTHESIZE exit check

- Another person can execute the experiment without clarification: yes.
- Decisions trace to measured nodes and reflections: yes.
- Major tensions have explicit gates: yes.
- Plan is narrower than the RAW solution space: yes.
- R3 authorization remains conditional: yes.
