# Gradient-calibrated paired correction protocol

This protocol was frozen before paired training.

Across the first eight same-prompt batches at the unique-supervision endpoint,
the aggregate static gradient norm is 5.928922, top-2,048 student-history
correction norm is 14.671710, and cosine is 0.111039. The measured correction
weight for a 10% gradient-norm contribution is therefore `0.0404105732`.

One arm is authorized. It uses the 512 static histories as its primary batches
and the row-matched top-2,048 student histories as a paired auxiliary loss.
Every condition matches the completed static control except that each batch adds
`0.0404105732 * (correction CE + correction KL)`. It starts from the same
unique-supervision endpoint and runs 256 updates.

The arm advances only if raw and instruction CE are each no more than 0.05 nats
worse than static control, exact cycle count is lower than static control, and
coherent frozen-suite count strictly increases without a new catastrophic
category. Compression must remain at least 10×.
