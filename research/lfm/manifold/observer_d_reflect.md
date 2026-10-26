# REFLECT: Observer.D Behavioral-Retention Floor

**Method:** Lincoln Manifold Method, pinned to
[`anjaustin/lmm@55719c0a`](https://github.com/anjaustin/lmm/tree/55719c0a2308d1a3f29ca656ed7b6287be33bca2)  
**Phase:** 3 — REFLECT  
**Inputs:** `observer_d_raw.md`, `observer_d_nodes.md`, and atomic audits

## Core insight

**Observer.D is not failing from one insufficient capacity; its conditional
signal is being compressed twice—first by Adam-normalized tied output updates
that corrupt input geometry, then by a shared FFN common mode that overwhelms
attention-created prompt differences—and autoregression amplifies the result.**

This explains why every single post-hoc intervention fails. Repairing the head
after rank reaches 7 cannot restore input geometry. Amplifying attention while
the head remains collapsed only selects another attractor. Adding a bias leaves
the tied Adam pathway intact. More data gives the same mechanisms more updates.

## Three-whys reflection

### Why does the table collapse?

1. Because it receives dense, frequency-shaped output gradients while also
   serving as the input representation.
2. Why do tiny unseen-row gradients matter? Adam's second-moment normalization
   makes their update magnitudes comparable to observed rows.
3. Why does this become low-rank? The data and softmax pressure share a dominant
   frequency/common direction, and the same coherent update is repeated over
   thousands of steps.

### Why are outputs prompt-insensitive?

1. Because final prompt states are nearly parallel and output distributions have
   very low pairwise JS.
2. Why are states parallel if attention reads content? Shared FFNs add a common
   component faster than attention adds conditional contrast.
3. Why does training permit this? Average CE rewards a strong universal token
   prior even when conditional distinctions are too weak for free generation.

### Why do loops appear?

1. Generic high-frequency tokens win narrow first-token margins repeatedly.
2. Their generated occurrence becomes new context aligned with the same common
   state and frequency-coded head.
3. Greedy decoding then closes the feedback loop before weaker prompt-specific
   directions can recover control.

## Resolved tensions

### CE versus geometry

**Resolution:** Treat CE as a bounded cost, not the sole objective. Preserve
input geometry first with rank-96 untying, then require CE penalties to remain
within preregistered limits. A lower CE that destroys rank and generation is not
a winning arm.

### Prompt contrast versus autoregressive stability

**Resolution:** Measure both. Operator changes advance only if they improve
prompt contrast *and* do not move repetition/cycle onset earlier. First-token JS
alone is not a success criterion.

### Isolated tests versus a serial remedy

**Resolution:** Use a staged manifold. First isolate FFN attenuation and
attention amplification on the already-supported early-untied substrate. Only
combine mechanisms that pass isolated gates. This preserves attribution without
pretending serial failures can be solved independently forever.

### Data breadth versus output damage

**Resolution:** Keep sparse KL because it protects rank and covers 22,242 tokens
beyond labels. Move its dense output effects to the untied output head instead
of weakening or deleting the teacher distribution.

### AdamW versus Muon/SGD

**Resolution:** Do not run an unmatched optimizer contest. Untying already
removes dense output Adam updates from input embeddings. Measure the separate
output head's optimizer state first. Only then preregister a parameter-matched,
update-norm-matched optimizer control. Existing AdamW/Muon results remain
preserved but do not answer this LM-specific question.

## Challenged assumptions

1. **“The model is simply too small.”** Not established. Recurrent state rank is
   much larger than table rank, and early untying preserves geometry above 10×.
2. **“Attention ignores the prompt.”** False. Attention reads user spans and
   increases separation; the relative signal is subsequently dominated.
3. **“Distillation causes dense collapse.”** False as a complete explanation.
   R2 direct CE collapses harder than KL.
4. **“More unique instructions will eventually fix coherence.”** Unsupported.
   Diversity improves conditioning while continued tying destroys the interface.
5. **“A higher prompt JS means better generation.”** False. Post-hoc scaling
   raises JS and still produces severe loops.
6. **“An explicit output bias will free the embedding.”** Falsified at R1 and
   after collapse; the dense tied optimizer pathway remains.
7. **“The best-CE checkpoint is the best starting point.”** False for prevention.
   Its input geometry has already fallen to rank 7.33.

## The grain of the problem

The natural split is temporal and structural:

1. **Before collapse:** protect input geometry by untying at R2.
2. **During conditional adaptation:** prevent the FFN common mode from outrunning
   attention-created contrast.
3. **At decoding:** verify that improvements survive feedback, rather than
   merely changing the first-token distribution.
4. **Only after these pass:** broaden vocabulary coverage and resume scaling.

This ordering is simpler than a broad search because each phase protects the
precondition needed by the next. Geometry lost in phase 1 cannot be recreated by
phase 2; conditional contrast lost in phase 2 cannot be repaired by decoding.

## Remaining questions

- Does fixed FFN residual scale 0.75 remain beneficial when trained from R2 with
  an untied head, or does the model route common mode through another operator?
- Does attention scale 1.5 preserve more conditional contrast than FFN 0.75 at
  the same CE cost?
- Does the separate rank-96 output table itself develop a harmful Adam
  second-moment rank collapse?
- What is the first training step at which prompt JS and generation-cycle onset
  diverge?

## Loop-back decision

The synthesis can remain simple and causally staged, so no loop-back is required
before writing it. A loop back to REFLECT is mandatory if an arm improves
teacher-forced metrics and geometry but still leaves all eight generation
categories catastrophic.

## REFLECT exit check

- Core insight stated in one sentence: yes.
- At least two tensions resolved: yes.
- Hidden assumptions challenged: seven.
- Remaining uncertainty explicit: yes.
