# Pipeline staleness on a standard model: protocol

*Fixed 2026-10-29, before any run on the protocol seeds. Exploratory.*

**Goal.** Take the small-network results to a standard setting and compare
them against methods used in practice. Harness:
`research/experiments/transformer_pipeline.py`.

## Setting

- **Model:** a character-level GPT on Tiny Shakespeare (SHA-256 checked;
  90/10 train/val split). Pre-LN, 4 blocks, width 128, 4 heads, context 64,
  0.82M parameters.
- **Pipeline:** 4 stages: embeddings + block 0, block 1, block 2, and
  block 3 + head.
- **Training:**
  - Adam, betas (0.9, 0.99); 100-step warmup, then cosine decay to 10%;
    gradient clipping at 1.0;
  - batch 32 × 64 characters, 2,000 steps;
  - learning rate 1e-3 (the usual small-character-model setting, not tuned
    here); 3e-3 secondary.
- **Staleness:** PipeDream with weight stashing and vertical sync, as in
  `pipeline_bound.py`. Stage k's gradient is taken at the whole model's
  weights from τ_k = (D/3)(3 − k) steps ago. D ∈ {3, 6, 12, 24, 48}.
  D = 6 is plain 4-stage asynchronous 1F1B; larger D stands for deeper
  pipelines.
- **Seeds:** 700–702. The only other run was a 400-step synchronous sanity
  check of Basis Rotation on seed 999.

**Arms at lr 1e-3:**
- fresh (synchronous Adam);
- stale;
- permuted (the permutation bound);
- PipeMare learning-rate rescheduling (T1, our reimplementation): stage k's
  lr is divided by τ_k^p, with p annealed from 1 to 0 over the first 500
  steps;
- weight prediction (SpecTrain-style): the gradient is taken at the
  snapshot plus τ_k × the update that followed it;
- Basis Rotation (with its own synchronous reference).

**At lr 3e-3:** fresh, stale and permuted only.

## Measures

- **Final validation loss** (nats per character) on 64 fixed validation
  batches.
- **Excess:** the seed-paired mean of (loss − fresh Adam loss) at the same
  learning rate.
- **Harmful delay:** a D where stale excess is at least 0.05.
- **Recovery** of method m: (stale − m) / (stale − permuted), from mean
  losses.
- **Floor:** the largest D up to which every grid delay is non-harmful
  (0 if D = 3 is harmful).

## Predictions and rules

- **P1, harm is directional.** At every harmful D at lr 1e-3, the permuted
  excess is at most ⅓ of the stale excess. It needs at least one harmful
  D; if there is none, P1 is untestable. Killed if it fails at any harmful
  D.
- **P2, the methods in use leave headroom.** At the largest harmful D at lr
  1e-3, each of PipeMare-lr, weight prediction and Basis Rotation recovers
  less than 0.5. Killed if any method recovers 0.5 or more.
  - Basis Rotation is a different optimizer, so its excess over its own
    synchronous reference is also reported.
- **P3, the floor falls with the learning rate.** The floor at 3e-3 is
  strictly below the floor at 1e-3. It's untestable if both floors are 0 or
  both are 48. Killed otherwise if this fails.
