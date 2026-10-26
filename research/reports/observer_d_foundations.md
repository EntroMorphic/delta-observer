# Observer.D Foundations

**Status:** Architecture definition, revision 0.1  
**Purpose:** Define the smallest falsifiable form of Observer.D before further optimization or scaling.

## 1. Central proposition

Two systems, A and B, realize the same objective through different internal organizations. Their architecture-specific states differ, but their successful learning trajectories may contain a shared computational invariant.

Observer.D attempts to identify that invariant and make it executable:

\[
m_i = T_\phi\left(\{h^A_{i,t},h^B_{i,t},\Delta h^A_{i,t},\Delta h^B_{i,t},t\}_{t=0}^{K}\right)
\]

Here, \(m_i\) is the method representation for input \(x_i\), and \(T_\phi\) is a chronology-sensitive method extractor.

The principal claim is not that \(m_i\) is easy to probe. The claim is that the observer can become an executable third model:

\[
\hat y_i = D_\psi(E_\omega(x_i)).
\]

The deployment encoder \(E_\omega\) learns to enter the method space discovered by the observer:

\[
E_\omega(x_i) \approx \operatorname{stopgrad}(m_i).
\]

At inference, A, B, and all trajectory inputs are absent. The retained \(E_\omega\), method-space transform, and \(D_\psi\) are **Observer.D**. They are not a separately initialized student.

## 2. Architectural boundary

### Discovery-time components

```text
A trajectory ──> architecture adapter A ──┐
                                          ├─> temporal method extractor ─> m
B trajectory ──> architecture adapter B ──┘                         │
                                                                    ├─> behavior decoder
original input ─> deployment encoder ───────────────────────────────┘
```

### Deployment-time components

```text
original input ─> deployment encoder ─> method space ─> behavior decoder ─> output
```

The deployment graph must have no references to source models, source activations, trajectory checkpoints, or discovery-only adapters.

## 3. Observer.D v0

The minimum implementation contains:

1. **Architecture adapters** for the differently shaped A and B states.
2. **Delta features** representing state changes between adjacent checkpoints.
3. **A chronological aggregator**, initially a small GRU. Mean pooling is a static-set baseline, not the primary method extractor.
4. **A method bottleneck** with an explicit dimension and parameter budget.
5. **A deployment encoder** mapping original inputs into method space.
6. **A complete behavioral decoder** producing every task output.

The first implementation should remain deliberately small. Capacity is a control variable, not a route to obtaining a positive result.

## 4. Training objectives

### 4.1 Method behavior

Train the discovered method representation to express the complete behavior:

\[
L_{\mathrm{behavior}} = \operatorname{BCEWithLogits}(D_\psi(m_i), y_i).
\]

An alternate label-free condition replaces \(y_i\) with the consensus output of A and B. These conditions must be reported separately.

### 4.2 Deployment alignment

The raw-input branch receives method supervision, not an independent task loss:

\[
L_{\mathrm{align}} =
\left\|\operatorname{norm}(E_\omega(x_i))-
\operatorname{stopgrad}(\operatorname{norm}(m_i))\right\|_2^2.
\]

Stopping the gradient prevents the discovery representation from moving merely to accommodate the deployment encoder.

### 4.3 Representation integrity

Use reconstruction, variance, and covariance controls only as needed to avoid collapse:

\[
L_{\mathrm{observer}} = L_{\mathrm{behavior}}
+ \lambda_a L_{\mathrm{align}}
+ \lambda_r L_{\mathrm{reconstruction}}
+ \lambda_v L_{\mathrm{variance}}.
\]

Improved reconstruction is not itself a success metric.

### 4.4 Deployment calibration

The behavioral decoder is trained on method representations. A small final calibration phase may expose it to \(E_\omega(x)\), but its cost and supervision must be reported. The primary result should also be shown without calibration.

## 5. What makes this different from a student

A conventional student is independently trained to imitate outputs or hidden features. In Observer.D:

- The deployment path is part of the observer from initialization.
- The behavioral decoder is the observer's decoder.
- The deployed representation is explicitly tied to the observer's discovered cross-architecture method space.
- No fourth behavioral model is created after observation.

This structural definition does not establish novelty by itself. The matched controls below determine whether the distinction has empirical value.

## 6. Fundamental controls

| ID | Condition | Question |
|---|---|---|
| F0 | Capacity-matched direct model | Is the deployment architecture simply good? |
| F1 | Output-distilled model | Is teacher output information sufficient? |
| F2 | Final source states only | Are trajectories unnecessary? |
| F3 | Chronological trajectories | Does the proposed signal work? |
| F4 | Temporally shuffled trajectories | Does order matter? |
| F5 | Example-mismatched trajectories | Is the signal input-specific? |
| F6 | One source only | Does cross-model comparison matter? |
| F7 | Same-architecture sources | Does architectural diversity matter? |
| F8 | Untrained sources | Is learning structure required? |

Every condition must use the same deployment architecture and, where possible, the same update count and sample exposure.

## 7. First acceptance gates

### Gate A — Executability

After physically removing A and B, Observer.D produces the complete output from the original input. Probe scores do not satisfy this gate.

### Gate B — Method-space necessity

F3 must outperform F1 and F2, while F4 or F5 degrades. Otherwise the result is consistent with ordinary distillation, extra supervision, or capacity.

### Gate C — Deployment efficiency

Report deployed parameters, FLOPs, latency, and memory independently from discovery cost. Observer.D is deployment-efficient only if its retained graph improves an accuracy/cost frontier over F0 and F1.

### Gate D — Generalization

The result must survive held-out combinations and then variable-width or sequential tasks. Exhaustive four-bit accuracy is a mechanism check, not final evidence of a learned algorithm.

## 8. Two distinct research claims

Keep these claims separate:

1. **Observer.D model synthesis:** observation produces an efficient executable third model.
2. **Observer.D learning-rule synthesis:** observation identifies a reusable state-transition or parameter-update rule.

The model-synthesis claim comes first. A learned latent transition

\[
m_{t+1}=F_\theta(m_t, e_t)
\]

becomes relevant only after the static deployed model passes Gates A--D.

## 9. Implementation order

1. Preserve the existing Delta Observer and initial student benchmark as historical controls.
2. Implement chronology-preserving trajectory records keyed by input and seed.
3. Implement Observer.D v0 as one model with discovery and deployment paths.
4. Verify physical source detachment in a unit test.
5. Run F0--F5 on four-bit addition.
6. Profile the retained graph in PyTorch and C/NEON.
7. Advance to variable-width arithmetic only if the mechanism survives the fundamental controls.

Optimization follows mechanism validation, not the reverse.
