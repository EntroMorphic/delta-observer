# Causal update test: separable schema damage, not a universal root cause

## Finding

For the measured schema regressions, the arithmetic-only optimization permits
parameter-update components that raise retained-schema loss. Those components
are **not necessary for the measured arithmetic transfer counts**: a same-norm
direction constraint prevents the schema damage without losing the registered
new-fact subtraction/addition counts.

This is a demonstrated local causal mechanism and an avoidable training failure,
not an ultimate explanation of numeric computation, every retention category,
or LLM reasoning. A one-key calibration example and an exposed test cannot
establish general retention. The model is not approved as a general replacement.

## Experiment 1: exact AdamW replay and same-state causal forks

Protocol: `native_update_conflict_protocol.md`, frozen before replay. Reproduced
the completed 512-update contrast training trajectory, including clipping,
AdamW moments/decay, accumulation, data order, rank, seed, and FP32 base.
Final adapter maximum difference from recorded contrast checkpoint: **0.0**.
Every fork restored the raw post-update parameters exactly before replay continued.

At six fixed boundaries compute the retained key `label` CE gradient for one
native-correct record at a fixed shared prefix. Let `d` be the actual optimizer
parameter displacement and `g` this retention gradient. If `g·d > 0`, remove
the positive component along `g`, then rescale the result to the original `d`
norm. This uses actual optimizer displacements, not an assertion about raw
gradient cosine or an assumed Adam response.

### Update 16, actual optimizer-step magnitude

| Measured quantity | Pre-update | Raw AdamW | Same-norm projected |
|---|---:|---:|---:|
| Arithmetic group CE | 3.061291 | 2.858141 | 2.858150 |
| Protected schema-key CE | 0.563816 | 0.591734 | 0.563749 |

Only 0.000201853 of a 0.029465725 raw update norm is removed before
renormalization: **0.685044% of the update norm**. The projected step has the
same total norm. Hence this result is not explained by simply shrinking the
whole update. The direct finite-step loss measurement confirms retention damage
is avoided while the target improvement is essentially retained on this group.

Update 64 similarly changes protected CE 1.853350→1.885215 raw, versus 1.853151
projected, with target CE 1.695607 raw versus 1.695711 projected. The key is
already damaged before update 64; a local projection is prevention of additional
loss, not a guarantee of repairing prior accumulated damage.

Positive retention-gradient conflicts occur at three of the six registered
boundaries, not all updates. Some raw steps already help the key and are left
unchanged. The scale-16 forks are explicitly extrapolations, not real optimizer
steps; they are retained in the raw artifact but are not needed for this finding.

## Experiment 2: apply the constraint through the full trajectory

Protocol: `native_projected_update_protocol.md`, frozen before this run.
Same contrast target data, 512 updates, 2,048 target examples, and 4,096 target
supervised tokens. At every step constrain the actual displacement against
**record-0 only**. Preserve the raw displacement norm whenever the projected
direction is nonzero. Adam moments follow actual target gradients; only realized
parameter deltas are projected. No line search, extra retention loss term, or
test-guided endpoint selection.

The one retention example is an explicit calibration/training signal, not a
holdout. Four other JSON records are unprotected by the algorithm but already
exposed as diagnostics. Extra gradient/data access and runtime are not matched
to the old contrast arm.

### Fixed endpoint

| Measurement | Unprojected contrast | Projected |
|---|---:|---:|
| New-fact/new-template subtraction | 7/20 | **7/20** |
| Counterfactual addition | 10/20 | **10/20** |
| Both add/subtract correct | 6/20 | **6/20** |
| New-fact/seen-template subtraction | 10/20 | **10/20** |
| Seen-fact/new-template subtraction | 11/20 | 10/20 |
| JSON parsed content | 3/5 | **5/5** |
| Protected key CE, FP32 at fixed prefix | 3.130009 | **0.014028** |
| Total native strict | 41/100 | 40/100 |
| Total confirmed content | 44/100 | **45/100** |

The JSON repair is specifically the two damaged records `record-0` and
`record-1`; the other three remain correct. Four records were unprotected, but
only one of those four was wrong in the unprojected model. Do not describe this
as five repaired regressions.

All registered arithmetic primary successes remain the same on the new-fact
splits and add/subtract pairs; the sole strict loss relative to unprojected
contrast is `seen_facts_new_template-09`. Two untouched-model ordering
successes (`order-1`, `order-3`) remain lost. Thus the local mechanism gates pass,
but broader retention and 100/100 do not.

The constraint activates at 314/512 updates. Among active steps the removed
norm fraction averages 0.424586%, ranging from 0.001412% to 1.551778% before
renormalization. These are factor-coordinate measurements, not invariant native
weight-space angles. The protected CE starts at 0.321858 and ends at 0.014028;
actual CE before and after every update is recorded, not inferred from dot
products. The maximum post-update protected CE is 0.305816 in this run.

Only the 835,584 adapters train. Frozen base hashes before/after match. Measured
projected training-loop time is 256.709070 seconds, versus 159.303090 seconds
for the recorded raw contrast loop. Extra retention backward passes and CPU
projection/transfer incur overhead; this is not a speed improvement.

## Causal interpretation and limits

Supported:

1. Some actual arithmetic AdamW steps improve target loss while harming a
   retained schema decision.
2. Their harmful retention component can be removed locally without giving up
   that step's arithmetic improvement or merely shrinking the whole step.
3. Applying that constraint prevents the observed JSON damage over the full
   training trajectory while preserving registered new-fact operation counts.

Not established:

- That every failure is gradient conflict, or that the arithmetic errors are
  caused by the same direction as schema damage.
- That Euclidean LoRA-factor projection is an optimal or parameterization-
  invariant preservation method.
- That a one-example constraint preserves broad native capabilities or will
  generalize to independently held-out schemas/models/seeds.
- That the same result holds at equal total compute or without retention-data
  access. No random-direction or multi-example control has yet been run.
- That a Hessian map or quantization sensitivity caused this result; neither
  is used by the intervention.

The deepest demonstrated design failure here is **an unconstrained target-only
update objective that permits avoidable retained-behavior damage**, not proven
inseparability or insufficient adapter capacity. The missing guarantee is
retention across relevant behaviors, not merely a lower task-training loss.
Numerical generalization and operation selection remain separate unresolved
mechanisms. No single ultimate universal cause is claimed.

## Artifacts

- `research/results/native_update_conflict_v1/updates.json`
- `research/results/native_update_conflict_v1/summary.json`
- `research/results/native_projected_update_v1_training.json`
- `research/results/native_projected_update_v1_eval/baseline.json`
- `research/results/native_projected_update_v1_comparison.json`

Reproduction: `audit_native_update_conflict`, `train_native_projected_update`,
`evaluate_pretrained_development`, and `score_native_projected_update`.
