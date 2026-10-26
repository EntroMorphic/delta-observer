# NODES: Observer.D Behavioral-Retention Floor

**Method:** Lincoln Manifold Method, pinned to
[`anjaustin/lmm@55719c0a`](https://github.com/anjaustin/lmm/tree/55719c0a2308d1a3f29ca656ed7b6287be33bca2)  
**Phase:** 2 — NODES  
**Input:** `observer_d_raw.md` and measured atomic audits

## Node 1: Deployment constraints are already hard boundaries

Observer.D must remain the deployed third model, source-free, and at least 10×
smaller than the 354,483,968-parameter teacher. The strict ceiling is 35,448,396
parameters. The rank-96 untied head totals 34,301,568 and fits; a dense untied
384-wide head does not.

**Why it matters:** Architecture options outside the ceiling are not candidates.

## Node 2: The tied table is the narrowest measured representation

The table contains 89.96% of parameters yet reaches effective rank 7.33. Final
recurrent states retain rank 71.15 on raw validation.

**Dependency:** Output-interface repair must precede claims that width 384 is the
fundamental floor.

## Node 3: AdamW turns dense tail pressure into table-wide motion

At R2 step 4,000, Adam normalization compresses a measured 1,324.1× difference
in first moments to 3.4985× in implied update norms. The stored update field
reinforces the collapse component; weight decay opposes it.

**Why it matters:** The cause is not merely “frequency imbalance.” It is the
interaction of tied roles, dense output gradients, concentrated data, and
coordinate-adaptive optimization.

## Node 4: Sparse KL is protective but incomplete

At R2, direct CE reaches rank 17.27 while distillation reaches 30.44. Teacher
top-64 targets cover 55,466 tokens versus 33,234 label tokens.

**Tension with Node 3:** The dense objective both broadens useful support and
damages the tied input geometry. Removing distillation would worsen the floor.

## Node 5: Prompt information exists inside the recurrent graph

Attention assigns 48.32%–57.80% mean mass to user-content spans in the best
model and increases prompt-state separation at every global step.

**Why it matters:** The attention mechanism is weak relative to the teacher, but
it is not blind and should not be replaced without first protecting its signal.

## Node 6: Shared FFNs create common-mode domination

FFNs increase absolute prompt distance and an even larger common direction.
Final mean-direction energy is 93.35%. Increasing FFN scale reduces prompt JS;
reducing it increases prompt JS.

**Boundary delta:** “Erased information” and “submerged information” demand
different fixes. The measurement supports the latter.

## Node 7: CE and conditional fidelity currently disagree

The baseline operator scales minimize CE at prompt JS 0.0333. Attention 1.5×
raises JS to 0.0842 for only +0.0442 raw CE, but generation remains incoherent.

**Tension:** Optimizing average next-token fit rewards the common predictor;
optimizing prompt distinction alone can select a different bad attractor.

## Node 8: Instruction adaptation helps recurrence while hurting the tied head

From R2 to the best mixed model, prompt cosine improves from 0.988 to 0.929 and
output JS from 0.0057 to 0.0333, while embedding rank falls from 30.44 to 7.33.

**Decision point:** The two learning effects must be separated before further
instruction adaptation.

## Node 9: Early decoupling is geometrically successful, behaviorally incomplete

After 512 matched mixed updates from R2, tied rank falls to 21.23 while rank-96
untied remains 30.52. Rank-96 pays +0.0908 raw CE and +0.0554 instruction CE;
both remain incoherent.

**Why it matters:** This is a viable prevention mechanism, not a completed fix.

## Node 10: Coverage is a separate floor

Label effective support is 2,170.5; 32,302 tokens are unseen as labels; 10,070
tokens are absent even from teacher top-64. French has seven unseen content
tokens out of fourteen, while English failures persist despite coverage.

**Dependency:** Coverage work is necessary for multilingual retention but cannot
explain or repair the general English common mode.

## Node 11: Autoregressive failure is a serial downstream process

First-token prompt distributions are nearly identical, and selected generic
tokens feed back into the context. Post-hoc operator changes alter the attractor
without restoring coherence.

**Why it matters:** Teacher-forced and first-token metrics are necessary but not
sufficient gates.

## Node 12: Experiment interpretability is now the scarce resource

Many plausible mechanisms exist: untying, residual scaling, contrastive losses,
frequency balancing, and optimizer partitioning. Combining them immediately
would obscure which boundary moved.

**Tension:** Small isolated arms preserve causality but may not cross the
behavioral gate; combined arms may work but be scientifically uninterpretable.

## Explicit tensions and dependencies

1. **CE vs geometry:** tied training wins short-run CE while destroying input
   rank; early untying preserves rank with a modest CE cost.
2. **Prompt contrast vs stability:** stronger attention/weaker FFN increases
   prompt variation but can intensify another output attractor.
3. **Broad teacher support vs dense damage:** KL protects coverage and rank, but
   dense output updates cannot remain attached to input embeddings.
4. **Isolation vs interaction:** serial failures require a combined remedy, but
   each component must first pass an isolated matched test.
5. **Pilot duration vs irreversible collapse:** short runs may not show
   coherence; long tied runs consume the geometry needed to test prevention.

## NODES exit check

- Twelve discrete nodes: yes.
- At least two tensions/dependencies explicit: yes.
- Observations separated from final plan: yes.
