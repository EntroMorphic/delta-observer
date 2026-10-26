# Student-history correction protocol

This protocol was frozen before on-policy extraction completed.

## Causal question

Does teacher supervision on Observer.D's own malformed autoregressive histories
improve free generation beyond the same amount of static teacher-history
supervision?

## Fixed conditions

- Start checkpoint: the 7,680-update unique-supervision endpoint.
- Architecture: factor-128 input/output, six method cells, FFN scale 0.75.
- Compression: 10.517943×.
- Prompts: the same 512 normalized-unique UltraChat prompts in both arms.
- Exclusions: all 8,192 prior unique-training prompt hashes and all eight frozen
  diagnostic prompts.
- Teacher and dataset revisions remain commit-pinned.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `0.01`.
- Batch size: 2; updates: 256; seed: 0.
- Objective: masked CE plus sparse top-64-and-tail KL.
- Supervised positions: 38,711 in each arm. Student-history masks are truncated
  row-for-row to the corresponding static teacher-response length, removing the
  otherwise observed 49,722-versus-38,711 token-count confound.

## Matched arms

1. **Static control:** teacher-generated response histories and ordinary next
   teacher-token labels.
2. **Student-history correction:** Observer.D greedily generates each history;
   at every generated-prefix position, the teacher supplies its top-1 label and
   sparse distribution. Student tokens remain in the input history, so the loss
   teaches recovery rather than imitation of the student's erroneous token.

Prompt-hash files must be identical between arms. Every shard hash and student
checkpoint hash must verify before training.

## Fixed endpoint measurements

- raw and independent instruction CE/KL/agreement;
- reconstructed input/output geometry;
- frozen-prompt JS and final common-mode energy;
- exact cycle onset and all eight retained generations;
- 10× compression.

The correction arm advances only if raw CE is no more than 0.15 nats worse than
the static control, instruction CE is no worse, and coherent frozen-suite count
strictly increases without a new catastrophic category. The frozen suite does
not select an intermediate checkpoint.
