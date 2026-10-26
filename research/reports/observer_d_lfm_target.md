# Observer.D Full-Model Target: LFM2.5-350M

> **Preprocessing caveat from subsequent red-team:** instruction extraction and
> prompt-audit paths render BOS in the chat template and then tokenize with
> special-token insertion enabled, producing duplicate BOS with the pinned
> tokenizer. Historical numbers below are retained, but native-chat capability
> and causal claims require revalidation. This does not establish that duplicate
> BOS explains the compressed model's failures. See
> `pretrained_evaluation_red_team.md` for the demonstrated defect and scope.

**Target:** `LiquidAI/LFM2.5-350M`  
**Mode:** Full-language compression  
**Status:** 10.518× factorized six-cell substrate leads measured frontier; instruction-coherence gate still failed

## Target facts

- 350 million parameters
- 16 layers: 10 gated short-convolution blocks and 6 grouped-query attention blocks
- Hidden width 1,024
- 65,536-token vocabulary with tied input/output embeddings
- 32,768-token supported context in the model card; the checkpoint config
  advertises 128,000 positions, so 32,768 remains the conservative limit until
  independently validated
- 28-trillion-token reported training budget
- Text-only, instruction-tuned, multilingual
- LFM Open License v1.0

## Honest scope

The original 28T-token training trajectory is unavailable. The first experiment
therefore observes LFM2.5's **computational trajectory** across layers and
generated tokens, not its original pretraining trajectory. A later LoRA run can
produce a controlled learning trajectory.

This distinction is mandatory:

- Layer/token states test computational-method extraction.
- Fine-tuning checkpoints test learning-method extraction.
- Output logits alone test conventional distillation.

## Observer.D-LM v0 target

The first deployable target is approximately 30--50 million parameters. The
65,536-token tied embedding will consume 25.2M parameters at width 384, setting
a hard practical floor for a conventional dense vocabulary interface.

The candidate deployment graph uses:

1. Tied token embeddings and output projection.
2. A shared method cell applied recurrently across depth.
3. Bounded convolutional state for local sequence processing.
4. Sparse grouped-query attention for retrieval.
5. A method bottleneck aligned to projected LFM internal trajectories.

Weight sharing across repeated method cells is the principal architectural
compression mechanism. It must be compared with a parameter-matched ordinary
student to determine whether observation, rather than sharing alone, provides
the advantage.

## Training signals

For token position \(i\):

\[
L = L_{\mathrm{CE}}(q_i, x_{i+1})
+ \lambda_k L_{\mathrm{top-k-KL}}(q_i,p_i^{LFM})
+ \lambda_m L_{\mathrm{method}}(z_i,m_i^{LFM})
+ \lambda_d L_{\mathrm{delta}}.
\]

Teacher targets store top-k indices/logits and captured probability mass rather
than dense vocabulary logits. Internal states are projected from 1,024 to a
fixed low dimension online, with projection provenance recorded.

## Gates

1. **Load and observe:** obtain reproducible hidden states from every LFM layer.
2. **Source-free execution:** Observer.D-LM generates using only its retained graph.
3. **Compression:** retained parameters and measured memory are below LFM2.5.
4. **Behavior:** improve over a capacity-matched direct model and output-only distillation.
5. **Observation value:** layer/delta guidance beats shuffled and final-state controls.
6. **Generalization:** retain improvement across held-out corpora and instruction benchmarks.

No claim of full-model transfer is warranted before Gate 4. No claim about the
original learning method is warranted before a controlled fine-tuning trajectory.

## Primary objective: retention at compression

Observer.D-LM is not required to exceed LFM2.5's intelligence or coherence. The
primary engineering result is useful behavioral retention at **at least 10×
parameter compression**. The current substrate satisfies the size side at
12.67×; it has not yet satisfied the fidelity side.

The definitive evaluation will report:

- Exact teacher and Observer.D held-out cross-entropy
- Student agreement with the teacher's top-1 token
- Sparse teacher-distribution KL
- Perplexity and benchmark-score retention ratios
- Instruction and generation coherence under blinded comparison
- Weight bytes, peak runtime memory, KV-cache bytes, and tokens per second

Cross-entropy retention is normalized against the uniform-vocabulary baseline:

\[
R_{CE}=\frac{\log |V|-CE_{Observer.D}}{\log |V|-CE_{teacher}}.
\]

This measures how much of the teacher's improvement over an uninformed model is
retained and avoids presenting raw CE ratios as capability percentages.

Thresholds must be fixed before the large run. A practical initial target is
at least 90% retention on a preregistered aggregate behavior score at 10× or
greater weight compression, with no catastrophic failure category. Results
will also be published continuously rather than reduced to that one threshold.

### Scaling ladder

| Stage | Training tokens | Purpose |
|---|---:|---|
| R0 | 4,096 | Pipeline validation |
| R1 | 131,072 | Verify stable learning curves and controls |
| R2 | 1,048,576 | Initial retention estimate |
| R3 | 8,388,608 | Architecture viability decision |
| R4 | 67,108,864+ | Serious full-language compression run |

Every stage uses a separately extracted validation corpus. Advancement depends
on continued teacher-gap closure; runs stop rather than burn compute if the
curve clearly plateaus below a useful retention level.

### Frozen qualitative sentinel

The following prompt was registered before R2 evaluation and must not be used
for training or model selection:

> Hypothetically, might reflective recursion be a function of cognition?

Teacher and Observer.D continuations use identical chat formatting, greedy
decoding, token budget, and EOS handling. Raw token IDs and decoded text are
retained for every evaluated checkpoint.

Before conversational-pilot evaluation, seven additional held-out prompts were
registered for factual explanation, exact formatting, deduction, summarization,
code generation, uncertainty calibration, and multilingual response. Corpus
extraction is repeated against that expanded registry so all eight prompts are
provably excluded. They broaden diagnosis but do not retroactively change the
R1/R2 evaluation.

## First probe result

The native checkpoint loaded successfully on Apple MPS using Python 3.12,
PyTorch 2.14.1, and Transformers 5.20.2.

| Measurement | Result |
|---|---:|
| Checkpoint parameters | 354,483,968 |
| Captured hidden-state boundaries | 17 |
| Hidden-state width | 1,024 |
| Probe tokens | 16 |
| Forward time | 0.731 seconds |
| Top-64 next-token probability mass | 0.9799 |
| Projected trajectory width | 128 |

All 10 convolution and 6 attention layer boundaries were observed. The result
files contain projected states and top-k logits rather than dense activations or
full-vocabulary distributions.

A subsequent batched extraction pilot streamed FineWeb-Edu and captured 8,192
tokens in 6.58 seconds (**1,245 tokens/s**) on MPS. It produced two 32-sequence
shards of 18.9 MB each, including top-64 logits and 128-dimensional projections
of all hidden-state boundaries. Both shards and the projection matrix have
independent SHA-256 provenance hashes.

## Initial deployment substrate

`research/lfm/observer_d_lm.py` implements a causal, source-free candidate with
a local/global method cell shared across eight depth repetitions.

| Measurement | Result |
|---|---:|
| Observer.D-LM parameters | 27,973,248 |
| Fraction of LFM2.5 parameters | 7.89% |
| Nominal parameter compression | 12.67× |

The vocabulary embedding/output weights are tied, and positional encodings are
fixed rather than adding a context-length-dependent parameter table. Tests
verify causal isolation, weight tying, output shape, and actual reuse of one
method cell. This is a deployment substrate awaiting training, not yet a
transferred model.

## Implemented training controls

The corpus and training harness now supports three capacity-matched arms:

1. Direct next-token training.
2. Direct training plus sparse output distillation.
3. Direct training, output distillation, and Observer.D state/delta alignment.

Sparse distillation preserves the teacher's probability outside the recorded
top-k as an explicit tail bucket rather than incorrectly renormalizing the saved
tokens. The final corpus shard is excluded from optimization and evaluated only
after training. A separate held-out corpus remains required for definitive runs.

## Invalidated first pilot

The first matched training pilot successfully exercised every code path but was
invalidated after exact teacher CE exposed a packing error: independent chunks
did not begin with LFM2.5's BOS token. This gave the recurrent/convolutional
teacher an invalid initial condition and produced EOS-heavy predictions.

The invalid metrics remain in `research/results/lfm_pilot_summary.json` as a
falsification record and must not be used as evidence. Corpus packing now adds
BOS to every independent sequence. A corrected smoke shard reduced measured
teacher CE from 14.07 to 6.20 and raised teacher top-1 next-token accuracy from
1.6% to 14.1%. The complete matched pilot must be rerun before interpretation.

## BOS-corrected pilot

The corrected matched run used 4,096 training and 4,096 validation tokens:

| Arm | Validation CE | Teacher top-1 agreement |
|---|---:|---:|
| LFM2.5 teacher | 5.0256 | 100% |
| Direct | 8.6597 | 4.69% |
| Output distillation | 8.2740 | 4.69% |
| State/delta trajectory | 8.2741 | 4.69% |

At **12.67× parameter compression**, output distillation improved early
validation CE by 0.3857. Trajectory guidance had no behavioral effect at its
initial weight. The compressed model is nowhere near teacher fidelity after
only 4,096 training tokens; this run validates measurement and optimization,
not capability retention.

## R1 — 131,072 training tokens

R1 used a separately extracted 16,384-token validation corpus:

| Arm | Validation CE | Normalized CE retention | Teacher top-1 agreement |
|---|---:|---:|---:|
| LFM2.5 teacher | 4.9552 | 100% | 100% |
| Direct | 7.6079 | 56.8% | 10.6% |
| Output distillation | **7.1672** | **63.94%** | **15.8%** |
| State/delta trajectory | 7.1683 | 63.93% | 15.7% |

The retention curve improved substantially from R0 and output distillation
clearly helped. The initial trajectory objective did not improve behavior and
was 0.0011 CE worse than output-only distillation. R1 therefore passes the
scaling gate, but does not support a trajectory-advantage claim.

### R1 frozen control result

On the preregistered reflective-recursion prompt, LFM2.5 produced a coherent
continuation discussing self-awareness, memory, reasoning, and recursive
self-evaluation. Observer.D R1 collapsed into a repeating `- 1, 1, 1, ...`
sequence.

R1 therefore **fails the qualitative coherence gate**. Its 63.94% normalized CE
retention must not be interpreted as 63.94% instruction-capability retention.
The exact prompt, raw token IDs, outputs, and timing are preserved in
`research/results/lfm_r1_control_prompt.json`.

## R2 — 1,048,576 training tokens

R2 used a fresh, non-overlapping 65,536-token validation corpus:

| Arm | Validation CE | Normalized CE retention | Teacher top-1 agreement |
|---|---:|---:|---:|
| LFM2.5 teacher | 5.0811 | 100% | 100% |
| Direct | 6.7701 | 71.89% | 16.23% |
| Output distillation | 6.2627 | 80.34% | **22.73%** |
| State/delta trajectory | **6.2602** | **80.38%** | 22.54% |

The trajectory arm improved CE over output-only distillation by only 0.0025
while reducing teacher top-1 agreement by 0.19 percentage points. This
single-seed difference is not credible evidence that trajectory guidance helps;
output-only distillation remains the primary baseline.

### R2 frozen control result

Both distilled checkpoints still failed the preregistered reflective-recursion
prompt. Output distillation repeated `The result of the color of the color ...`;
the trajectory arm repeated `The color of the color ...`. The teacher remained
coherent under the identical decoding protocol.

R2 therefore reaches approximately 80.38% **token-level normalized CE
retention**, not 80.38% instruction-capability retention. FineWeb-Edu scaling
alone has not retained the instruction-tuned behavior. The next experiment adds
teacher-generated conversational trajectories while keeping this prompt frozen
and excluded from training and model selection.

## Conversational pilot

The pilot extracted 128 UltraChat `train_sft` user turns and generated fresh,
greedy LFM2.5 responses under the model's ChatML template. Only assistant
continuations were supervised. All eight registered prompts were excluded. An
independent 32-sequence `test_sft` extraction supplied 3,102 held-out assistant
tokens.

| Checkpoint | Instruction CE retention | Instruction top-1 agreement | Raw CE retention | Coherent prompts |
|---|---:|---:|---:|---:|
| R2 before instruction training | 47.31% | 16.63% | 80.34% | not run |
| One instruction epoch | 53.89% | 23.40% | 80.02% | 0/8 |
| Four instruction epochs | **55.47%** | **27.21%** | 76.32% | 0/8 |

The masked instruction objective clearly improves teacher-forced prediction,
but all eight free-generation diagnostics remain incoherent. Four epochs add
lexical variety while causing substantial raw-language regression; this is not
a behavioral success. The result falsifies the hypothesis that a small
instruction-only fine-tune is sufficient. An interleaved raw/instruction pilot
is the final prerequisite before committing compute to R3.

## Mixed raw/conversational pilot

The mixed pilot resumed from R2 and interleaved 8,192 raw sequences with eight
passes over 128 conversational sequences. Instruction batches were 11.11% of
updates. This improved both held-out domains:

| Metric | R2 | Mixed pilot |
|---|---:|---:|
| Raw normalized CE retention | 80.34% | **84.92%** |
| Raw teacher top-1 agreement | 22.73% | **25.80%** |
| Instruction normalized CE retention | 47.31% | **58.75%** |
| Instruction teacher top-1 agreement | 16.63% | **28.11%** |
| Coherent registered prompts | not run | **0/8** |

Interleaving prevents the severe raw-language forgetting of instruction-only
training and continues closing both teacher-forced gaps. Nevertheless, every
free-generation diagnostic remains incoherent and instruction-insensitive. The
frozen reflective-recursion output begins `The most important important ...`
and drifts into unrelated World Heritage text.

This falsifies the stronger hypothesis that schedule mixing alone is enough.
R3 should not merely repeat these 128 conversations. The next scaling step must
increase unique conversational diversity while retaining the mixed schedule.

## Matched conversational-diversity control

The repetition schedule was rerun with the same R2 initialization, raw corpus,
optimizer, learning rate, 4,608 updates, and 11.11% instruction-batch fraction.
The sole intended change was replacing eight repetitions of 128 conversations
with 1,024 unique teacher-generated conversations.

| Metric | Repeated 128 | Unique 1,024 | Delta |
|---|---:|---:|---:|
| Raw normalized CE retention | 84.92% | **85.43%** | +0.52 pp |
| Raw teacher top-1 agreement | 25.80% | **25.90%** | +0.10 pp |
| Instruction normalized CE retention | 58.75% | **59.71%** | +0.96 pp |
| Instruction teacher top-1 agreement | 28.11% | **28.79%** | +0.68 pp |
| Coherent registered prompts | 0/8 | 0/8 | none |

Unique diversity has a measurable positive effect on teacher-forced metrics,
but it does not cross the free-generation coherence gate. The frozen prompt
regresses to an `important important ...` loop, and all seven added diagnostics
remain unrelated or repetitive. This rules out conversational repetition and
small-scale diversity as sufficient fixes. Before brute-force R3, the next
control directly compresses the teacher's learned token geometry into the tied
Observer.D embedding table.

## Teacher-derived token geometry

LFM's 1,024-wide tied token vectors were projected into 384 dimensions,
orthogonal-Procrustes aligned to the trained R2 Observer.D embedding basis, and
blended into the detached tied table. The exported graph remains exactly
27,973,248 parameters and needs no teacher at inference.

A 50% blend was too disruptive, reducing raw retention to 65.76% before
adaptation. A conservative 10% blend improved raw retention immediately from
80.34% to 81.28% and instruction retention from 47.31% to 47.62% without any
optimization. After the same 4,608-update mixed-diversity schedule:

| Metric | No transfer | 10% teacher geometry | Delta |
|---|---:|---:|---:|
| Raw normalized CE retention | 85.43% | **85.77%** | +0.34 pp |
| Raw teacher top-1 agreement | 25.90% | **25.95%** | +0.05 pp |
| Instruction normalized CE retention | 59.71% | **59.94%** | +0.23 pp |
| Instruction teacher top-1 agreement | 28.79% | **29.05%** | +0.26 pp |
| Coherent registered prompts | 0/8 | 0/8 | none |

This is a small but consistent observer-derived initialization gain. It is
retained for R3, while the failed 50% transfer and unchanged coherence failure
remain explicit negative controls.

## Atomic floor audit

R3 extraction was stopped non-destructively after 18,432 of 57,344 planned
extension sequences. The completed compact shards remain diagnostic data, but
the directory has no completion manifest and is not a completed R3 corpus.
Scaling is paused while the structural floor is measured.

The atomic audits use the best 10%-teacher-geometry mixed checkpoint. The raw
sample contains exactly 8,192 supervised tokens; the independent conversational
sample contains exactly 3,102 supervised assistant tokens. Validation shard
hashes match their manifests.

### Exact deployment accounting

| Measurement | Value |
|---|---:|
| Teacher parameters | 354,483,968 |
| Observer.D parameters | 27,973,248 |
| Parameter compression | 12.67× |
| Resident parameter bytes (FP32) | 111,892,992 |
| Tied embedding parameters | 25,165,824 (89.96%) |
| Tied embedding bytes | 100,663,296 |
| Persistent KV-cache bytes | 0 |
| Measured process maximum RSS | 791,445,504 bytes |
| Native forward throughput, batch 2 × 128 | 14,147 tokens/s |

The throughput number is synchronized MPS forward throughput, not
autoregressive generation throughput.

### Token-level floor

| Measurement | Raw | Conversational |
|---|---:|---:|
| Observer.D CE | 5.85055 | 4.71079 |
| Teacher CE | 4.91902 | 0.44685 |
| Student minus teacher CE | 0.93153 | 4.26394 |
| Student probability on teacher top-1 | 0.11261 | measured in audit file |
| Teacher top-1 probability | 0.53900 | measured in audit file |

The conversational result is the clearest measured behavioral floor: masked
adaptation improves aggregate instruction CE, but the student remains 4.26394
nats behind the teacher on the teacher's own held-out responses and is 0/8 on
coherent free generation.

Training-label exposure explains a large part of the raw tail:

| Measured label exposure | Raw validation CE |
|---|---:|
| 0 | 12.78 |
| 1--3 | 11.00 |
| 4--15 | 9.75 |
| 16--63 | 8.18 |
| 64+ | 4.38 |

Exactly 32,302 of 65,536 vocabulary tokens have zero label exposure in the
measured training corpora. This is a real coverage floor, but it is not by
itself the cause of greedy degeneration: selected generation tokens had at
least 26 measured exposures in the English diagnostics and were generally
high-frequency tokens.

### Learned tied-table collapse

| Checkpoint | Embedding effective rank | First-PC variance |
|---|---:|---:|
| Untrained Observer.D | 382.87 / 384 | 0.30% |
| R2 | 30.44 / 384 | 52.39% |
| Mixed, repeated conversations | 7.97 / 384 | 73.88% |
| Mixed, 1,024 unique conversations | 7.78 / 384 | 74.27% |
| Mixed + 10% teacher geometry | 7.33 / 384 | 75.08% |
| LFM2.5 teacher | 654.78 / 1,024 | 5.57% |

The untrained table is full-rank, so this is learned rather than an
initialization constraint. Additional optimization improves teacher-forced CE
while reducing embedding rank. The dominant token score has Pearson correlation
`0.857531` with log label exposure; embedding norm has correlation `-0.835630`.

The tied gradient paths are asymmetric on the measured raw batch. The input
path updates exactly the 157 token rows present in the batch. The dense output
path updates all 65,536 rows. Median row-gradient norm is 0.07702 on input rows
for the input-only path, while the output-only path gives non-input rows a
median of 0.00001116 and a maximum of 0.18621. Tying therefore applies a dense
frequency-shaped output signal to the same vectors used to encode inputs.

### PC1 is mostly a surrogate output bias

On raw validation, final normalized hidden states have a consistently positive
projection on embedding PC1: mean 3.21659, standard deviation 0.71542. PC1's
contribution to the true-label logit is 4.69763 on average. Regressing token PC1
score on centered log exposure gives slope 0.48861.

| Head intervention | Raw CE | Conversational CE |
|---|---:|---:|
| Unmodified | **5.85055** | **4.71079** |
| Remove 100% PC1 | 6.91333 | 5.74867 |
| Add `0.5 × log-frequency` only | 6.45601 | 5.23244 |
| Remove PC1 + add `0.5 × log-frequency` | 5.93634 | 4.81901 |

Neither naive intervention helps by itself. Replacing PC1 with an explicit
frequency bias recovers most, but not all, of the deleted signal. The measured
interpretation is that the dominant tied-table direction serves largely, but
not exclusively, as an output-frequency bias. The missing residual prevents a
claim that frequency alone explains PC1.

### Recurrent states and depth floor

The recurrent states retain much more dimensional structure than the tied
table. Across the same 8,192 raw tokens, centered state effective rank falls
from 100.33 after depth 1 to 71.15 after depth 8. First-component variance rises
only from 9.14% to 10.89%, far below the embedding table's 75.08%.

Late depth is nevertheless saturating. Consecutive-state cosine is 0.99079 at
depth 5, 0.99589 at depth 7, and 0.99258 at depth 8. Exact CE by available depth
is:

`[13.4735, 7.5947, 6.5837, 6.1865, 6.0807, 5.9227, 5.9010, 5.8505]`.

The large gains occur before depth 6; the final two repetitions improve CE by
only 0.0712 total. Recurrent-state collapse is therefore secondary to the
vocabulary-interface collapse, although late repetitions also have diminishing
effect.

### Exact degeneration onset

Every generation step now records full-vocabulary entropy, selected-token
probability, top-1/top-2 margin, measured training exposure, and exposure rank.
A cycle is preregistered in the audit code as the first suffix containing three
exact repeats of a period from 1 through 16.

| Prompt | First detected three-repeat cycle |
|---|---:|
| Reflective recursion | step 4, token `important` |
| Seasons | none within 64 tokens |
| JSON formatting | none within 64 tokens |
| Deduction | step 4, token `important` |
| Summarization | step 51, period 4 |
| Python | step 5, token `The` |
| Uncertainty | none within 64 tokens |
| French | step 44, period 1 |

The prompts without a detected three-repeat cycle are still incoherent and
contain earlier repeated n-grams. For example, seasons repeats a prior trigram
by step 11 and JSON formatting by step 14. This is not low-confidence random
drift: individual selected-token probabilities reach 0.9582 and top-1/top-2
logit margins reach 4.9856 while producing an incorrect loop.

### Current falsifiable diagnosis

1. The primary measured floor is the tied vocabulary interface: 89.96% of all
   parameters collapse to effective rank 7.33 while recurrent states remain at
   rank 71.15 after the final depth.
2. Dense output gradients and highly imbalanced label exposure make the shared
   table encode output frequency in its dominant direction. That direction is
   useful to CE, so deleting or centering it after training is not a fix.
3. Vocabulary coverage is independently inadequate: 49.29% of token rows have
   no measured label exposure and unseen-label CE is 12.78.
4. Late recurrent repetitions saturate, but they are not the narrowest measured
   bottleneck.
5. Aggregate CE gains are compatible with worsening embedding geometry and
   catastrophic autoregressive behavior. They are not instruction-capability
   retention.

The next controlled changes are therefore an explicit learned output bias and
an untied low-rank output head that preserves at least 10× total compression.
They must be trained in matched small controls before any R3 extraction resumes.

### Matched 512-update architecture controls

The two proposed changes were tested immediately, along with an unchanged tied
control. All three start from the same best checkpoint and use the same seed,
ordered data, AdamW settings, batch size 2, learning rate `1e-4`, and exactly
512 updates. This is a mechanism diagnostic, not an R3 run.

| Head | Parameters | Compression | Raw CE | Raw agreement | Instruction CE | Instruction agreement |
|---|---:|---:|---:|---:|---:|---:|
| Tied | 27,973,248 | 12.672× | **6.01657** | **25.833%** | **4.82535** | 27.950% |
| Tied + bias | 28,038,784 | 12.643× | 6.01697 | 25.826% | 4.82561 | 27.950% |
| Untied rank-96 | 34,301,568 | 10.334× | 6.15644 | 24.345% | 5.08967 | 25.596% |

The explicit-bias control is an exact early negative: it provides no measurable
benefit over tied continuation. Its learned values have Pearson correlation
0.60353 with log exposure after 512 steps, but leaving the dense tied path in
place means it does not prevent further collapse.

The rank-96 head is initialized with the exact truncated SVD of the tied table.
It retains 96.3836% of squared weight energy with relative Frobenius error
0.18981. Before optimization its measured raw CE is 6.16454 and conversational
CE is 5.31809. After 512 steps those become 6.15644 and 5.08967: adaptation
helps, especially conversationally, but remains worse than the tied control.

The geometry control is causal evidence about continued collapse:

| Head after 512 updates | Input-table effective rank | Input PC1 variance |
|---|---:|---:|
| Tied | 6.4738 | 77.019% |
| Tied + bias | 6.4725 | 77.022% |
| Untied rank-96 | **7.3513** | **75.040%** |

The starting input rank was 7.3334. Tied continuation reduces it to 6.47;
removing the dense output gradient leaves it at 7.35. The separate rank-96
output embedding reaches effective rank 40.30. This directly identifies the
dense tied output path as the cause of *continued* input-geometry collapse.

None of the three controls is coherent on any registered prompt. Rank-96 often
delays exact short-cycle onset and produces longer phrases, but remains generic,
repetitive, and prompt-inappropriate. It is therefore a useful prevention
mechanism applied too late, not a behavioral success. Any next matched control
must introduce decoupling before the table reaches rank 7; merely adapting a
post-collapse checkpoint does not recover lost token geometry.

## Causal atomic decomposition

### Exact corpus concentration

Across the measured raw and conversational training corpora there are 1,125,012
supervised labels and 1,179,648 input-token slots.

| Distribution | Nonzero tokens | Gini | Effective support | Top-100 share | Top-1,000 share |
|---|---:|---:|---:|---:|---:|
| Inputs | 33,235 | 0.91948 | 2,001.2 | 42.67% | 64.09% |
| Labels | 33,234 | 0.91664 | 2,170.5 | 41.03% | 62.90% |
| Teacher top-64 entries | 55,466 | 0.86907 | 7,436.6 | 19.30% | 45.90% |

Teacher top-64 supervision reaches 22,242 tokens that never occur as labels,
so sparse distillation materially broadens coverage. Exactly 10,070 tokens
occur neither as labels nor anywhere in teacher top-64. The tail bucket still
supplies a dense aggregate gradient, but not token-specific teacher structure.

The conversational corpus supervises 76,436 of 131,072 token slots (58.32%).
Its generated responses match the teacher top-1 token on 99.878% of supervised
positions, versus 27.29% on the large raw extension. It is consequently a much
sharper but more concentrated conditional target.

### CE, not sparse KL, drives the long-run collapse

| Training stage | Direct rank | Distillation rank | Trajectory rank |
|---|---:|---:|---:|
| R1, 131,072 tokens | **204.12** | 198.07 | 197.90 |
| R2, 1,048,576 tokens | 17.27 | **30.44** | **30.58** |

At R1, KL slightly accelerates early concentration. By R2, however, direct CE
has collapsed far more severely. R2 direct PC1 holds 61.19% of variance versus
52.39% under distillation. Sparse teacher distributions therefore mitigate the
long-run collapse by supplying broader structure; they do not cause it.
Trajectory and output-only geometry remain effectively identical.

On one fixed raw batch, CE and KL tied gradients have cosine 0.534 at R2 and
0.523 at the best checkpoint. Both output paths update all 65,536 rows, but the
instantaneous gradient has only about 0.02% cosine with the already-formed
collapse component and locally points toward slightly reducing it. Collapse is
therefore a cumulative data/optimizer equilibrium, not an effect attributable
to one batch or KL alone. The exact gradient categories and row distributions
are retained in `lfm_loss_gradient_audit.json`.

### Prompt information collapses into a common mode

The eight registered prompts were measured at the first assistant-token
position before autoregressive errors can accumulate.

| Model | Mean pairwise hidden cosine | Mean output JS | Unique top-1 tokens |
|---|---:|---:|---:|
| LFM teacher | 0.29282 | 0.59753 | 6 |
| Observer.D R2 | 0.98815 | 0.00572 | 1 |
| Best mixed Observer.D | 0.92900 | 0.03326 | 2 |
| Post-collapse rank-96 control | 0.93554 | 0.02148 | 1 |

Jensen-Shannon divergence is bounded above by `ln(2) = 0.6931`; the teacher's
prompts therefore induce almost maximally different distributions for many
pairs. R2 treats every prompt almost identically. Instruction adaptation
recovers measurable conditioning, but remains an order of magnitude short.

Relative to an empty-user ChatML prompt, R2's output-distribution JS ranges only
from 0.0017 to 0.0066. The teacher ranges from 0.6356 to 0.6922. The best mixed
model improves to 0.0665--0.2003, confirming that some prompt information is
learned but remains dominated by a common component.

English lexical coverage is not the general cause. Every token in the frozen
reflective prompt appears in training labels; its rarest content tokens have
4--8 exposures. Seven of fourteen French content tokens are unseen, explaining
an additional multilingual floor, but the fully covered English prompts fail
in the same way.

### Attention carries signal; shared FFNs dominate it

Exact attention weights show that the best model's last-token queries assign
48.32%, 50.49%, 54.92%, and 57.80% mean mass to measured user-content spans at
global depths 2, 4, 6, and 8. Attention is not merely reading the common ChatML
suffix.

Every global-attention operation makes prompt states more distinct. Every
subsequent shared FFN adds a larger common direction. For example:

- Best depth 4 cosine: 0.91934 before attention, 0.88427 after attention,
  0.90931 after FFN.
- Best depth 6 cosine: 0.93118 before attention, 0.90361 after attention,
  0.92108 after FFN.
- R2 depth 3 cosine: 0.92859 after local mixing, 0.96437 after FFN.

Absolute pairwise distance is not erased: in the best model it rises from 8.91
at embedding to 31.60 after depth 8. The problem is relative domination. The
mean-direction energy fraction reaches 93.35%, so the output head sees prompt
differences as perturbations around a large shared attractor.

Trajectory supervision has a small internal effect: at R2 it changes final
prompt cosine from 0.98787 to 0.98416 and prompt-state effective rank from 4.49
to 5.18. This remains too small to affect behavior and does not change the tied
embedding collapse.

### Operator-scale falsification

Inference-only scaling confirms the roles without claiming a deployment fix:

| Attention scale | FFN scale | Raw CE | Instruction CE | Prompt JS |
|---:|---:|---:|---:|---:|
| 1.0 | 1.0 | **5.85055** | **4.71079** | 0.03326 |
| 0.0 | 1.0 | 6.22543 | 5.34761 | 0.00100 |
| 1.5 | 1.0 | 5.89472 | 4.85165 | **0.08421** |
| 1.0 | 0.75 | 5.88442 | 4.78112 | 0.06713 |
| 1.0 | 1.25 | 5.87823 | 4.75907 | 0.01679 |

Removing attention eliminates nearly all conditional variation. Increasing
attention or reducing FFN strength restores variation with a modest
teacher-forced penalty. Nevertheless, all generated outputs remain incoherent;
amplification often changes the attractor to repeated `*` tokens. This proves
that recurrent common-mode domination and tied-head collapse are serial
failures. Post-hoc correction of either one alone is insufficient.

### Early output-bias control

A learned bias was also trained from initialization for the complete 512-step
R1 schedule:

| Objective | No-bias rank / CE | Bias rank / CE |
|---|---:|---:|
| Direct | 204.12 / 7.60787 | 205.16 / 7.60430 |
| Distill | 198.07 / 7.16721 | 199.09 / 7.17436 |

The direct bias already correlates 0.89978 with log exposure, but preserves only
about one effective dimension and does not consistently improve CE. Missing
bias is secondary rather than a sufficient structural explanation.

### Early decoupling control

The rank-96 output head was finally introduced at R2, before mixed instruction
adaptation and before rank fell from 30 to 7. Both arms then received the same
512 ordered mixed updates.

| R2 + 512 updates | Input rank | Raw CE | Instruction CE | Instruction agreement |
|---|---:|---:|---:|---:|
| Tied | 21.23 | **6.21175** | **5.64312** | **20.374%** |
| Untied rank-96 | **30.52** | 6.30252 | 5.69847 | 19.536% |

Early decoupling preserves all measured input rank while tied continuation loses
9.21 effective dimensions. The cost is 0.0908 raw CE and 0.0554 conversational
CE after 512 updates. Prompt-state cosine is almost identical (0.97347 tied,
0.97383 rank-96), so the head change does not block recurrent instruction
learning. Neither arm is coherent at this point; rank-96 remains generic and
repetitive despite fewer exact short cycles.

### Revised measured diagnosis

There are four distinct floors:

1. **Output geometry:** tied CE converts the input table into a frequency-coded
   output interface. Sparse KL mitigates but cannot stop it. Early untying stops
   the measured rank loss at a modest CE cost.
2. **Conditional signal-to-common-mode ratio:** attention reads user content and
   creates distinct states, but the repeatedly shared FFN grows a dominant
   prompt-independent direction. Aggregate CE rewards this common predictor.
3. **Coverage:** label effective support is only 2,170.5 tokens; 32,302 labels
   are unseen and 10,070 tokens lack even top-64 teacher-specific supervision.
4. **Autoregressive stability:** once a generic high-frequency token wins, the
   generated history reinforces the same attractor. Higher prompt JS alone does
   not break this loop.

The measurements support addressing output decoupling and recurrent
common-mode control together, from a pre-collapse checkpoint. They do not
support more R3 data, an output bias alone, PC deletion, frequency-bias
injection, post-collapse untying, or attention scaling alone.

### Exact AdamW-state mechanism

The R2 resume checkpoint preserves the complete AdamW state at step 4,000.
Reconstructing its bias-corrected adaptive update field isolates how tiny dense
output gradients become a table-wide low-rank displacement.

| Row category | First-moment mean row norm | Implied adaptive-delta mean row norm |
|---|---:|---:|
| Zero label exposure | 0.000001260 | 0.0002572 |
| Teacher top-64 but no label | 0.000001826 | 0.0002818 |
| Absent from teacher top-64 | 0.000000103 | 0.0002069 |
| 1--3 label exposures | 0.00001849 | 0.0002075 |
| 64+ label exposures | 0.0016679 | 0.0008997 |

Before normalization, frequent-row first moments are 1,324.1× larger than
zero-label rows. After Adam normalization, their implied update norms are only
3.4985× larger. The ratio of those two measured ratios is 378.47×.

The square-root second-moment matrix has effective rank 2.02 and places 91.66%
of centered variance in its first component. The bias-corrected adaptive update
has effective rank 55.15 and positive 0.04284 cosine with the existing collapse
component. Including decoupled weight decay gives cosine 0.04093 and a measured
first-order collapse-energy change of +2.13599 for the stored update field.

Weight decay is not the cause: its Frobenius norm is 0.000986 versus 0.202662
for the adaptive update, and its cosine with the collapse component is
`-0.3951`, opposing collapse. Dense softmax/tail gradients provide the signal;
Adam's coordinate normalization makes extremely small, coherent gradients on
unseen rows comparable in update magnitude to observed rows. This explains why
the current-batch raw gradient can locally oppose collapse while the cumulative
optimizer state continues to reinforce it.

The optimizer measurement strengthens the case for early untying. It also
requires a future matched control in which dense output parameters do not share
Adam-normalized updates with input embeddings. A generic optimizer swap is not
yet supported: its learning rate and parameter assignment would need a matched
preregistered control rather than an unmatched SGD or Muon run.

## Lincoln Manifold deployment

The measured problem was processed through the Lincoln Manifold Method pinned
to upstream commit `55719c0a2308d1a3f29ca656ed7b6287be33bca2`. The phases are
kept separate so observations, reflection, and execution decisions remain
auditable:

1. [`manifold/observer_d_raw.md`](../lfm/manifold/observer_d_raw.md) — unfiltered
   understanding, risks, naive approaches, and open questions.
2. [`manifold/observer_d_nodes.md`](../lfm/manifold/observer_d_nodes.md) — twelve
   measured nodes and five explicit tensions/dependencies.
3. [`manifold/observer_d_reflect.md`](../lfm/manifold/observer_d_reflect.md) —
   three-whys causal reflection, challenged assumptions, and resolved tensions.
4. [`manifold/observer_d_synth.md`](../lfm/manifold/observer_d_synth.md) — the
   executable pre-collapse recovery program and loop-back gates.

The synthesis authorizes no R3 work. Its clean cut is a four-arm, 512-update
pre-collapse experiment starting from R2: tied baseline, rank-96 untying,
rank-96 plus FFN residual 0.75, and rank-96 plus attention residual 1.5. A
combined arm is forbidden unless the isolated operator arm or arms pass the
preregistered geometry, CE, prompt-contrast, and cycle-onset gates.

## Lincoln Manifold execution results

### Stage 1: residual-pathway isolation

All four arms started from the same R2 checkpoint and received 512 ordered
updates. Rank-96 plus FFN residual scale `0.75` was the only arm to pass every
preregistered gate. Its input rank was 30.5161, raw CE was 6.30088,
instruction CE was 5.69868, prompt JS increased relative to rank-96, and the
only detected three-repeat cycle moved from French generation step 22 to 33.
Attention scale `1.5` increased prompt contrast more strongly but introduced
new cycles at steps 12--34 and was rejected. The exact four-arm record is
`research/results/lfm_manifold_stage1_results.json`.

### Stage 2: full matched schedule

The surviving FFN arm and rank-96 control each received the full 4,096-update
schedule. FFN `0.75` improved all measured aggregate diagnostics: raw CE
6.05560, held-out instruction CE 4.73403, instruction agreement 28.046%, prompt
JS 0.05632, and lower final common-mode energy. Input rank remained 31.38.
Nevertheless, all eight frozen generations were incoherent and several entered
lexical cycles. The operator intervention is real but insufficient.

### Conditional-contrast negative controls

At the first supervised assistant position, one measured batch had teacher
pairwise JS 0.50456 and student JS 0.00317. Scalar JS matching at a measured
10% gradient calibration raised frozen-prompt JS from 0.01060 to 0.15266 and
reduced common-mode energy from 97.42% to 82.51%. It did not preserve semantic
identity: EOS and loop shortcuts satisfied the directionless objective.

Semantic assignment contrast replaced scalar distance with correct
student-to-teacher pairing. Its gradient norm was 17.4108 versus 16.1044 for
CE plus KL, giving measured 10% and 100% weights of 0.09250 and 0.92497. The
100% arm reduced common mode to 96.20% and raised prompt JS to 0.0157 without
the EOS shortcut, but remained incoherent. Both objectives are retained as
negative controls, not promoted as solutions.

### Strong input geometry and depth-adapter negative controls

Decoupled teacher input geometry retained effective rank 86.58 at a 50% blend
and 243.56 at a 100% blend after 512 updates. The 50% arm held raw and
instruction CE essentially unchanged; neither arm improved coherent behavior.
Strong token geometry is therefore not sufficient.

Eight independently trainable rank-128 depth adapters used 35,091,072
parameters, exactly 10.102× compression. They produced negligible changes in
raw CE, instruction CE, agreement, prompt JS, rank, and generation. Dense
depth adapters are also insufficient.

### Factorized vocabulary interfaces and six method cells

The dense 65,536-by-384 input table consumed 25.2M parameters while expressing
only about 30 effective dimensions. The current substrate replaces both
vocabulary interfaces with independent rank-128 factors and spends the freed
budget on six independently trainable method cells across eight depths. Its
exact deployment size is 33,702,784 parameters, or **10.5179× compression**.
The truncated-SVD initialization retains 93.087% of the R2 tied-weight squared
energy in each interface.

After 512 updates this substrate improved instruction CE and agreement and had
no detected short cycles. After the complete schedule it established the
current quantitative frontier:

| Metric | Rank-96 + FFN 0.75 | Factor-128 + six cells |
|---|---:|---:|
| Parameters | 34,301,568 | **33,702,784** |
| Compression | 10.334× | **10.518×** |
| Raw CE | 6.05560 | **6.02584** |
| Held-out instruction CE | 4.73403 | **4.64806** |
| Instruction top-1 agreement | 28.046% | **28.562%** |
| Frozen-prompt JS | 0.05632 | **0.06453** |
| Stored input-table effective rank | 31.38 | **113.47** |

The teacher's instruction CE is 0.44685, leaving a measured 4.20121-nat gap.
Free generation remains 0/8 coherent and exhibits `Java`, `universe`, generic
template, and numbering loops. This is not a capability success. It shows that
factorization and independent method cells improve the substrate without
closing the behavioral gap.

Only 76,436 supervised conversational tokens from 1,024 prompts trained the
reported arms. Architecture search has therefore reached diminishing returns.
The next authorized cut is a larger frozen-prompt-excluded, explicitly unique
UltraChat corpus on the six-cell substrate, with immutable source revisions,
per-prompt hashes, checkpointed atomic measurements, and no R3 raw extension
unless conditional supervision first produces coherent behavior.

### Measurement correction: factor rank is not deployed-interface rank

The earlier 113.47 value measures the stored 65,536-by-128 input factor alone.
It does **not** measure the reconstructed 65,536-by-384 embedding consumed by
the recurrent graph. Post-protocol auditing now reconstructs both matrix
products before computing covariance geometry. At the unique-supervision
endpoints, reconstructed input rank is only 10.38--10.43 and reconstructed
output rank is 1.83--2.35. The factor-table number remains exact but must not be
presented as deployed interface geometry.

A centered-SVD control falsifies centering as the remedy. The source R2 table
has centered effective rank 30.4413, but its best centered rank-128
approximation has effective rank 7.8555 because the discarded tail carries
23.314% of centered energy and a disproportionate share of spectral entropy.
The uncentered initializer produced the same 7.855 deployed rank. Truncation,
not failure to subtract the mean, creates the step-zero rank reduction.

## Unique versus repeated conversational supervision

The comparison protocol was frozen in
`research/lfm/manifold/observer_d_unique_supervision_protocol.md`. Both arms
started from the same factor-128/six-cell initialization and saw the same 7,168
raw sequences plus 8,192 conversational sequence exposures. The control
repeated the original 1,024 prompts eight times; the experimental arm used
8,192 normalized-unique prompts.

The unique corpus contains 622,330 supervised tokens. All 128 shard hashes
match, all 8,192 prompt hashes are distinct, and none intersects the eight
frozen prompts. Teacher CE is 0.450059 and saved top-64 mass is 0.986815.

At the fixed 7,680-update endpoint:

| Metric | 1,024 prompts repeated 8× | 8,192 unique prompts |
|---|---:|---:|
| Raw CE | 6.16086 | **5.99976** |
| Instruction CE | 4.52291 | **3.76685** |
| Instruction top-1 agreement | 32.205% | **37.041%** |
| Reconstructed input rank | 10.3766 | **10.4294** |
| Reconstructed output rank | **2.3511** | 1.8342 |
| Frozen-prompt JS | **0.35648** | 0.24266 |
| Final common-mode energy | **59.07%** | 81.70% |
| Coherent frozen generations | 0/8 | 0/8 |

Unique supervision closes 0.75606 nats of instruction CE and simultaneously
improves raw CE by 0.16110 nats relative to repeated exposure. This proves that
prompt diversity, rather than conversational token count alone, materially
improves teacher-forced behavior. It does not close the capability gap:
generation remains incoherent in every category, prompt separation worsens,
and exact short cycles reappear. The unique arm therefore passes all automatic
numeric gates but fails the preregistered coherence gate and does not advance.
R3 remains paused.

### Output-interface freeze falsification

Component geometry localized the trained output-product collapse primarily to
the vocabulary table: its effective rank falls from 111.16 at initialization
to 32.94 by update 960 and 9.00 by update 7,680, while projection rank remains
64.73--74.97. Three matched 960-update controls froze the table, projection,
or both:

| Arm | Instruction CE | Agreement | Output-product rank | Prompt JS | Common mode |
|---|---:|---:|---:|---:|---:|
| Train both | **4.78186** | **28.917%** | 2.4265 | **0.02622** | **95.29%** |
| Freeze table | 5.14738 | 27.434% | **8.4915** | 0.01292 | 96.50% |
| Freeze projection | 4.80966 | 28.885% | 2.3408 | 0.02278 | 95.71% |
| Freeze both | 5.19019 | 27.079% | 7.8546 | 0.01087 | 96.95% |

All four arms remain 0/8 coherent. Preserving output rank therefore does not
force useful conditional behavior; it makes the teacher-forced objective worse
and reduces prompt separation. Output collapse is a consequence of the current
objective and factor dynamics, but is not independently sufficient to explain
generation failure.

The unique-data trajectory instead exposes an off-policy mismatch. From update
960 through 7,680, instruction CE falls monotonically from 4.78186 to 3.76685
and agreement rises from 28.917% to 37.041%, yet self-generated histories
remain incoherent. Static teacher-forced data never trains recovery after the
student's first autoregressive error. The next justified experiment is
student-rollout/teacher-correction distillation on non-frozen prompts.

### Student-history correction negative control

A matched 512-prompt pilot compared static teacher-generated histories with
teacher correction on Observer.D's own greedy histories. Both corpora used the
same prompt hashes, excluded all prior 8,192 training prompts and all frozen
prompts, and were mask-matched to exactly 38,711 supervised positions. Both
arms started from the unique-supervision endpoint and received 256 updates.

The distribution shift is directly measured. Teacher top-64 mass is 0.986652
on static histories but only 0.815994 on malformed student histories. Teacher
top-1 CE on those student histories is 1.12862, versus ordinary teacher CE
0.44952 on static histories.

| Metric | Static histories | Student-history correction |
|---|---:|---:|
| Raw CE | **6.08897** | 6.23328 |
| Instruction CE | **3.82302** | 4.11762 |
| Instruction agreement | **37.524%** | 34.365% |
| Prompt JS | **0.25398** | 0.18470 |
| Final common-mode energy | **79.15%** | 84.20% |
| Coherent frozen generations | 0/8 | 0/8 |

Top-64 student-history correction fails every behavioral gate. It is rejected,
not scaled. The failure does not falsify exposure shift: it shows that the
existing sparse target loses directional information off-policy, where 18.40%
of teacher mass is represented by a single tail bucket. Any further corrective
pilot must first increase measured teacher-distribution support.

A no-training top-512 extraction retained only 90.61% mass on the matched mask
and failed its preregistered 95% gate. An exact support sweep over all 38,711
positions measured 81.60%, 84.83%, 87.82%, 90.61%, 93.16%, 95.42%, and 97.31%
at k = 64, 128, 256, 512, 1,024, 2,048, and 4,096 respectively. Top-2,048 is
the smallest measured support authorized for one further correction pilot.

Top-2,048 extraction matched every prompt, input token, and mask and retained
95.4212% teacher mass. After the fixed 256 updates it improved over top-64
correction (raw CE 6.23328→6.17810; instruction CE 4.11762→4.09801; exact
cycles 4→1), but remained 0/8 coherent and still underperformed static
supervision (instruction CE 3.82302). It therefore fails the coherence gate and
does not advance as a correction-only method. The cycle reduction is retained
as evidence that high-support off-policy targets contain a useful stability
signal; any mixture with static supervision must be gradient-calibrated rather
than assigned an arbitrary weight.

The measured static and correction gradient norms were 5.92892 and 14.67171
with cosine 0.11104. A paired objective therefore used weight 0.0404106, exactly
10% of static gradient norm. It preserved static performance (raw CE 6.08602
versus 6.08897; instruction CE 3.81747 versus 3.82302) but left exact cycle
count at 4 and coherence at 0/8. Gradient-calibrated pairing is stable but has
no credible behavioral effect and is rejected.

### Response-position localization and start-loss negative control

Independent instruction validation has a U-shaped error profile. At response
offset 0, student CE is 4.20743 and agreement is 28.125%; offsets 8--15 improve
to CE 3.19023 and agreement 42.578%; offsets 64--127 degrade to CE 4.13459 and
agreement 32.543%. The model therefore fails both initial response-mode
selection and long-horizon continuation.

The first-eight-token gradient norm was 18.66209 versus 5.92892 for the full
static objective, with cosine 0.42328. A measured 10%-norm auxiliary weight of
0.0317699 was tested for 256 updates. It preserved aggregate CE but worsened
first-token CE from 4.38417 to 4.40086, left cycle count at 4, and remained 0/8
coherent. Explicit response-start upweighting is rejected.
