# Support-selected student-history correction protocol

Top-512 extraction failed the preregistered 95% support gate and was not
trained. A separate exact teacher pass over all 38,711 mask-matched off-policy
positions measured retained mass at fixed support sizes:

| k | Teacher mass |
|---:|---:|
| 64 | 0.815993 |
| 128 | 0.848270 |
| 256 | 0.878206 |
| 512 | 0.906071 |
| 1,024 | 0.931605 |
| 2,048 | 0.954212 |
| 4,096 | 0.973146 |

Top-2,048 is therefore the smallest measured support satisfying the fixed 95%
requirement. One top-2,048 correction arm is authorized. It must use input IDs,
prompt hashes, masks, initialization, optimizer, and 256-update schedule
identical to the rejected top-64 correction arm. It advances only if raw CE is
within 0.15 nats of top-64 correction, instruction CE is lower, and coherent
frozen-suite count strictly increases without a new catastrophic category.
Expanded targets are extraction/training artifacts and do not alter the
33,702,784-parameter, 10.517943× deployment graph.
