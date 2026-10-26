# Unique conversational supervision protocol

This protocol was frozen before the 8,192-prompt extraction completed.

## Question

Does broader unique conditional supervision improve the factorized six-cell
Observer.D substrate beyond an equal number of repeated conversational
exposures?

## Fixed substrate

- Initialization: `lfm-r2-factor128-cells6.pt`
- Input factor rank: 128
- Output factor rank: 128
- Independently trainable method cells: 6
- FFN residual scale: 0.75
- Parameters: 33,702,784
- Compression: 10.517943× against 354,483,968 teacher parameters
- Objective: masked CE plus sparse top-64-and-tail KL
- Optimizer: AdamW, learning rate `1e-4`, weight decay `0.01`
- Batch size: 2; seed: 0; one epoch

## Matched arms

Both arms contain the same 7,168 R2 raw sequences and exactly 8,192
conversational sequence exposures.

1. **Repeated control:** the existing 1,024-prompt corpus appears eight times.
2. **Unique arm:** one frozen-prompt-excluded corpus of 8,192 normalized-unique
   prompts.

The unique corpus uses teacher revision
`9e6c6ccf47cd318696e137d381a7ded8fe4df09f` and UltraChat revision
`8049631c405ae6576f93f445c6b8166f76f5505a`. Its manifest must report 8,192
accepted unique prompts, expose all per-prompt SHA-256 hashes, and report all
duplicate, frozen, missing-text, and over-length rejections.

## Measurements

Immutable snapshots are written every 960 updates and at the 7,680-update
endpoint. Each snapshot is measured on the unchanged raw validation corpus,
the independent instruction validation corpus, and the fixed eight-prompt
diagnostic suite. Measurements are:

- raw and instruction CE, sparse KL, and teacher top-1 agreement;
- input- and output-factor effective rank;
- final prompt common-mode energy and prompt-distribution JS;
- exact three-repeat cycle onset and retained raw token IDs/text;
- parameter count and compression.

Intermediate frozen-prompt generations are descriptive only. They cannot be
used to choose a checkpoint; endpoint comparison is primary.

## Advancement gates

The unique arm advances only if all are true at the fixed endpoint:

1. compression remains at least 10×;
2. raw CE is no more than 0.15 nats worse than the repeated control;
3. instruction CE is lower than the repeated control;
4. instruction teacher-top-1 agreement is higher than the repeated control;
5. it produces at least one coherent frozen-suite response and introduces no
   catastrophic category failure relative to the repeated control.

Prompt JS, common-mode energy, rank, and cycle onset diagnose the mechanism but
cannot override a failed coherence gate. R3 raw extraction remains paused.
