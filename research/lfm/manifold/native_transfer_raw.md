# RAW: transfer with retention, not another training-score victory

Method source: anjaustin/lmm, pinned commit
`55719c0a2308d1a3f29ca656ed7b6287be33bca2` (`LMM.md`).
This file records problem framing and uncertainty, not experimental conclusions.

The actual goal is useful improvement of a functioning native LLM. A particular
100-task score is an operational measurement, not the definition of intelligence.
Our strongest local fit is 125/128 compound training questions, but only 8/20
on a different wording and different numerical cases. More adapter rank does
not improve fit in the tested schedule. This weakens the claim that parameter
count is the next lever, but does not exclude target placement or optimization.

Ordering is a different problem: all supervised pairs fit, and removing other
tasks improves exposed transfer to 18/20. Treating arithmetic and ordering as
one bottleneck has repeatedly produced bundles that improve one category while
regressing another. Formatting also changes the path through a preexisting
copying weakness. Our data builder once reduced all copied values to clock.
Unique prompt strings are clearly not enough to certify semantic coverage.

The negative controls are useful. Primitive arithmetic can work. Canonical
training fit can improve with repetitions. We have not shown invariant
operation selection under changes to surface form. We also have not cleanly
crossed facts and wording, so the transfer gap could include numerical-example
generalization as well as phrasing. The next step should expose that boundary
before another long training run.

Potential remedies include training meaning-preserving paraphrase families,
training meaning-changing minimal pairs, and protecting retained examples
during updates. Each could still overfit or interfere. Curvature might help
locate sensitive updates, but six LoRA-factor directional derivatives are not
a native-weight sensitivity map. No speed advantage has been measured.

Open questions:

1. How much error remains when numbers are held fixed and only wording changes?
2. How much remains when wording is held fixed and only numbers change?
3. Do operation words alter behavior when numerical tokens are identical?
4. Can a task-specific gain survive integration with retained task updates?
5. Which proposed intervention improves locked unseen templates without a
   content-retention loss, rather than just increasing an exposed score?
