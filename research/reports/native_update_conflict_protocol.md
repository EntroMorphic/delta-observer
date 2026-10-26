# Actual-update causal fork: diagnostic only

Frozen before replay. Reproduce the completed contrast arm exactly from the
untouched pinned FP32 model, identical rank/seed, data, accumulation, clipping,
AdamW settings, and ordered 512 updates. At updates 1,16,64,128,256,512 fork
counterfactual parameter displacements from the same pre-update state.

Measure a retained-schema CE gradient on the native-correct key `label` for
record-0 at the fixed shared native prefix. Compute the actual raw AdamW delta,
including clipping/decay/moment effects. Remove only its positive component
along that retention-loss gradient. Compare zero, raw, projected, same-norm
projected, and raw/same-norm projected at displacement scale 16. Scale 16 is
an extrapolation, not an actual optimizer step or a claimed local guarantee.

For each fork measure actual target-group CE, protected key CE, two unprotected
key CEs, realized rounded parameter displacement/dot product, and five fixed
generated cases (three schemas and one arithmetic add/subtract pair). Restore
the raw post-update tensors exactly before continuing the raw trajectory.
Verify the final replay adapter matches the recorded contrast adapter exactly.
If replication fails, retain the diagnostic but do not claim it reproduces the
original trajectory. No new candidate adapter is selected/deployed.

Interpretation: matching norms tests direction rather than just smaller steps.
A nonpositive gradient dot product is first-order geometry, not a guarantee of
finite-step retention. Actual loss/generation must be measured. A one-key
constraint cannot establish general retention. The experiment tests whether
update-direction conflict is present and locally separable, not an ultimate
universal cause of arithmetic, schema binding, and all other model failures.
