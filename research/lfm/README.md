# Observer.D × LFM2.5-350M

Target: [`LiquidAI/LFM2.5-350M`](https://huggingface.co/LiquidAI/LFM2.5-350M)

This is the full-language compression track. It is intentionally separate from
the four-bit mechanism test and uses a Python 3.12 environment because LFM2.5
requires Transformers 5.x.

## Environment

```bash
/opt/homebrew/bin/python3.12 -m venv .venv-lfm
.venv-lfm/bin/pip install -r research/lfm/requirements.txt
```

## Instrumentation probe

```bash
.venv-lfm/bin/python -m research.lfm.probe_lfm \
  --prompt "Extract the name and date from: Ada arrived on 4 July 2026."
```

The first run downloads the model from Hugging Face. The probe records layer
types, hidden-state shapes, layer-to-layer deltas, projected internal states,
and top-k next-token targets without saving full 65,536-way logits.

`observer_d_lm.py` defines the initial source-free deployment substrate: a tied
vocabulary interface and one local/global method cell recurrently shared across
depth. It is a controlled candidate to receive the observed method, not yet a
claim that the architecture was discovered from LFM.

## Corpus extraction

```bash
.venv-lfm/bin/python -m research.lfm.extract_lfm_corpus \
  --sequences 1024 --sequence-length 128 --shard-size 64
```

Extract non-overlapping validation data by advancing the same deterministic
packed stream:

```bash
.venv-lfm/bin/python -m research.lfm.extract_lfm_corpus \
  --skip-sequences 1024 --sequences 128 --sequence-length 128 \
  --output research/results/lfm_validation
```

The extractor streams and packs documents, then writes compact shards with
token labels, top-64 teacher logits, full-distribution normalization constants,
exact teacher logits for target tokens, and 128-dimensional projections of all
17 hidden-state boundaries. Every shard
and projection matrix is hashed in `manifest.json`.

After the trajectory objective has been ruled out for a scaling stage, pass
`--no-trajectories` to omit projected hidden states while retaining labels,
top-k logits, the probability-tail normalizer, and teacher CE. These compact
shards support `direct` and `distill`, but intentionally reject `trajectory`.

`distillation.py` implements a sparse top-k KL objective with an explicit tail
probability bucket, plus state-and-delta trajectory alignment between LFM's 17
observed boundaries and Observer.D-LM's repeated method states.

## Training arms

```bash
for mechanism in direct distill trajectory; do
  .venv-lfm/bin/python -m research.lfm.train_observer_d_lm \
    --data research/results/lfm_corpus --mechanism "$mechanism"
done
```

Pass `--validation-data research/results/lfm_validation` to evaluate against a
separately extracted corpus instead of withholding the final training shard.
Long runs write `observer-d-lm-resume.pt` periodically; pass that file through
`--resume` to restore model, alignment, optimizer, epoch, batch, and history.
Incremental corpus stages can be combined without copying shards by passing
one or more directories through `--additional-data`.

## Conversational trajectories

FineWeb-only R1 and R2 checkpoints failed the frozen instruction sentinel. The
instruction extractor takes user turns from an independent instruction corpus,
renders them with LFM's ChatML template, greedily generates teacher responses,
and captures logits plus all 17 projected hidden-state boundaries:

```bash
.venv-lfm/bin/python -m research.lfm.extract_lfm_instructions \
  --split train_sft --sequences 1024 --batch-size 8 \
  --output research/results/lfm_instruction_train
```

Only generated assistant tokens receive a loss mask; user text, template tokens,
and right padding are context only. Normalized source prompts are hashed for
provenance, and all prompts in `control_prompts.json` are excluded. Use a
non-overlapping split or prompt offset for instruction validation.
Generation batches use left-padded prompts with an attention mask; captured
training sequences are repacked without left padding before trajectory capture.

Initialize an instruction pilot from an R2 checkpoint without restoring its
optimizer or training cursor:

```bash
.venv-lfm/bin/python -m research.lfm.train_observer_d_lm \
  --data research/results/lfm_instruction_train \
  --validation-data research/results/lfm_instruction_validation \
  --mechanism distill --initialize-from path/to/observer-d-lm-distill.pt
```

`--resume` instead restores an interrupted run exactly. It cannot be combined
with `--initialize-from`.

## Teacher-derived token geometry

To test whether random initialization of the 25M-parameter tied embedding is a
sample-efficiency bottleneck, project LFM's 1,024-wide token vectors into 384
dimensions, align them to a trained Observer.D basis with orthogonal
Procrustes, and blend them into the detached checkpoint:

```bash
.venv-lfm/bin/python -m research.lfm.initialize_teacher_embedding \
  --base-checkpoint path/to/observer-d-lm-distill.pt --blend 0.5 \
  --output path/to/observer-d-teacher-embedding.pt
```

The teacher is needed only during this export step. The resulting checkpoint
contains the same 27.97M-parameter Observer.D graph and remains source-free at
inference. Projection and alignment provenance are saved beside it.

## Frozen qualitative control

The prompt registry includes:

> Hypothetically, might reflective recursion be a function of cognition?

Run a deterministic teacher/Observer.D comparison after training:

```bash
.venv-lfm/bin/python -m research.lfm.compare_control_prompt \
  --checkpoint path/to/observer-d-lm-distill.pt
```

The evaluator uses the same tokenizer and chat template, greedy decoding, and a
fixed token budget. It saves raw continuation token IDs as well as text. This is
a qualitative sentinel, not a substitute for held-out quantitative evaluation.
Pass `--all-prompts` to run the full preregistered diagnostic suite with one
teacher/model load. Distinct-bigram and distinct-trigram ratios are reported as
degeneration indicators, not as semantic quality scores.

The three arms share the exact deployment architecture and data exposure:

- `direct`: next-token labels only
- `distill`: labels plus sparse teacher behavior
- `trajectory`: labels, teacher behavior, layer states, and layer deltas

By default, the final corpus shard is excluded from optimization and evaluated
once after training. Larger runs should extract a separately versioned held-out
corpus rather than relying only on this shard-level split.

## Licensing

LFM2.5 uses the LFM Open License v1.0. Redistribution requires retaining the
license and attribution notices and marking modified files. Commercial use by
entities at or above the license's US $10 million annual-revenue threshold is
not granted without a separate agreement. This repository does not vendor or
redistribute Liquid AI weights.
