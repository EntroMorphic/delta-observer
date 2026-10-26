# Response-start supervision protocol

This protocol was frozen before training.

Independent instruction validation localizes a large gap at the first response
token: student CE 4.20743, teacher CE 0.54495, and agreement 28.125%. Student
performance is best at offsets 8--15 (CE 3.19023; agreement 42.578%) before
degrading again at long range.

Across eight static batches, the ordinary objective gradient norm is 5.928922,
the first-eight-position objective norm is 18.662094, and cosine is 0.423276.
The measured weight for a 10% auxiliary contribution is `0.0317698642`.

One 256-update arm starts from the unique-supervision endpoint and exactly
matches the completed static control, adding only
`0.0317698642 * (first-eight CE + first-eight KL)`.

It advances only if first-token CE falls, aggregate instruction and raw CE are
each no more than 0.05 nats worse than static control, coherent frozen-suite
count strictly increases, no new catastrophic category appears, and deployment
compression remains at least 10×.
