# High-support student-history correction protocol

This protocol was frozen before top-512 extraction completed.

## Question

Did top-64 correction fail because its single tail bucket discards too much
teacher direction on off-policy student histories?

## Fixed comparison

- Control: completed top-64 student-history correction arm.
- Experimental arm: identical student checkpoint, 512 prompt hashes, greedy
  histories, static-length masks, initialization, optimizer, seed, batch size,
  and 256 updates; only teacher support changes from top-64 to top-512.
- Both arms contain exactly 38,711 supervised positions.
- Deployment architecture remains unchanged at 33,702,784 parameters and
  10.517943× compression; expanded targets are training artifacts only.

Before training, every prompt hash and input token ID must match the top-64
corpus. The top-512 manifest must report its measured retained probability mass.

## Gates

Top-512 correction advances only if it retains at least 95% mean teacher mass,
raw CE is no more than 0.15 nats worse than top-64 correction, instruction CE
is lower, and coherent frozen-suite count strictly increases without a new
catastrophic category. Prompt JS, common mode, geometry, and exact cycle onset
remain diagnostic and cannot override coherence.
