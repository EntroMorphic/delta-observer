# One-retention-direction full-trajectory diagnostic

Frozen after the actual-update fork experiment and before this training run.
This is a mechanism experiment, not an untouched-test training comparison.

Use the exact completed contrast data, same pinned base, rank/seed, target
gradient accumulation, optimizer, and 512-update budget. At each step measure
the exposed native-correct record-0 schema-key gradient. Compute the actual
AdamW displacement, project away a positive retention-gradient component,
and rescale to the raw displacement norm. Keep optimizer moments from the target
gradients; project only realized parameters. No line search or step-size tuning.

This intentionally uses record-0 as a protection/calibration example. It is not
held out. All original 100 tasks are already exposed diagnostic data. Protect
no other schema examples and do not use their losses to select directions.

Primary mechanism gates at the fixed endpoint:

1. Protected key CE no greater than its initial value plus 0.05, and record-0
   parsed content correct.
2. Unseen-fact/unseen-template subtraction at least the unprojected contrast's
   7/20; addition at least 10/20; both-operation pairs at least 6/20.
3. Report all unprotected content/strict losses even if (1)/(2) pass.

Projection's nonpositive dot product is only first order. Measure actual key CE
before/after every step and full generation at the fixed endpoint. Match target
updates/tokens, but report extra retention gradient/data access and runtime;
this is not compute-matched to the old model. No broad retention, 100/100, or
ultimate cause of numerical reasoning follows from one protected key.
