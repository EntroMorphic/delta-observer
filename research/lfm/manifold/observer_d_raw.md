# RAW: Observer.D Behavioral-Retention Floor

**Method:** Lincoln Manifold Method, pinned to
[`anjaustin/lmm@55719c0a`](https://github.com/anjaustin/lmm/tree/55719c0a2308d1a3f29ca656ed7b6287be33bca2)  
**Phase:** 1 — RAW  
**Rule:** This file records the unfiltered problem view. It is not the plan.

## Stream of consciousness

Observer.D is small enough, detached, executable, and measurably learning, but
it is not behaving like the teacher. The tempting reaction is to scale data,
add a clever regularizer, or call 85.77% normalized raw CE retention a partial
success. All three reactions are dangerous. Free generation is 0/8 coherent,
and the measurements now show that aggregate CE can improve while the model's
conditional behavior becomes structurally worse.

The current substrate compresses LFM2.5 by 12.67×, but 89.96% of its parameters
live in one tied input/output table. That table falls from effective rank 382.87
at initialization to 30.44 at R2 and 7.33 after mixed adaptation. This looks
like a capacity failure until the recurrent states are measured: final states
retain rank 71.15 over raw tokens. It is therefore an interface failure before
it is a global parameter-count failure.

The tied table is being asked to do incompatible jobs under sparse evidence. It
must encode 65,536 input identities while simultaneously acting as a dense
softmax classifier. Only 33,234 tokens occur as labels, and label effective
support is 2,170.5. Sparse KL broadens teacher-specific coverage to 55,466
tokens, but AdamW still applies table-wide coordinate-normalized updates. The
stored R2 state shows the mechanism directly: a 1,324.1× frequent/unseen
first-moment ratio becomes only 3.4985× in implied update magnitude. Tiny dense
tail gradients on unseen rows become real, coherent movement in the same table
used for input meaning.

Yet untying alone is not enough. The best model's attention places substantial
mass on user content and makes prompt states more distinct. The shared FFN then
adds an even larger common component. Prompt distances increase, but the common
direction grows faster; final mean-direction energy is 93.35%. The teacher maps
the eight prompts to mean pairwise hidden cosine 0.2928 and output JS 0.5975.
Observer.D maps them to cosine 0.9290 and JS 0.0333. This is not “no prompt
signal.” It is prompt signal submerged beneath a universal response prior.

Autoregression supplies the final trap. Once `important`, `The`, `best
possible`, or `*` wins, the generated history reinforces the same basin.
Increasing attention after training changes the winning basin but does not
create coherent behavior. Reducing the FFN after training does the same. A
post-collapse rank-96 head preserves no geometry that has already been lost.

My naive approach would be to combine rank-96 untying, FFN scaling, an output
bias, balanced sampling, and a new optimizer in one run. That would be another
uninterpretable experiment. The opposite naive approach is to keep measuring
indefinitely and never cut. The useful boundary is to intervene at R2, where
input rank is still 30.44, and partition the serial failures with matched arms.

The uncomfortable possibility is that width 384 and one recurrent cell simply
cannot retain the teacher's conditional manifold. We have not established that.
Early rank-96 untying preserved rank 30.52 through 512 mixed updates, and
instruction conditioning improved similarly to tied training. The architecture
still has an untested path. But preserving rank is not the objective; coherent
behavior is. Any new mechanism must improve prompt separation and generation
without hiding behind CE.

## Fears and risks

- A larger output head could consume the 10× compression margin without fixing
  recurrence.
- Common-mode control could improve prompt JS while merely switching attractors,
  as inference-time scaling already did.
- A broad optimizer comparison could be invalid because unmatched learning
  rates or parameter assignments dominate the result.
- Continuing from the rank-7 checkpoint could make every prevention mechanism
  look ineffective.
- A 512-step pilot could preserve geometry yet be too short to show coherence;
  a long run could destroy geometry before that becomes visible.
- Adding many losses at once would prevent causal attribution.

## Open questions

1. Can early output decoupling and recurrent common-mode control coexist without
   exceeding 35,448,396 parameters, the strict 10× ceiling?
2. Does fixed FFN attenuation during training preserve attention-created prompt
   contrast, or does optimization recreate the common mode elsewhere?
3. Is attention amplification or FFN attenuation the cleaner intervention under
   matched CE, rank, and generation measurements?
4. Can any 512-step arm produce a coherent registered continuation, rather than
   only higher first-token JS?
5. If geometry and conditioning improve but generation does not, what exact
   transition creates the first attractor during decoding?
6. Should AdamW remain on the separate output head, or does its low-rank update
   field collapse that head too even when inputs are protected?
7. What evidence would falsify the current 384-wide shared-cell substrate rather
   than merely one training recipe?

## First instincts, explicitly uncommitted

- Start from R2, never from the post-collapse best checkpoint.
- Preserve the measured rank-96 head because it stays above 10× compression.
- Test FFN and attention residual scales separately before combining them.
- Instrument every checkpoint boundary; do not infer trajectories from endpoints.
- Require a generation gate in addition to CE and geometry gates.

## RAW exit check

- More than 200 words: yes.
- Uncertainties and conclusions both present: yes.
- At least three open questions: yes.
- One risk and one naive approach: yes.
