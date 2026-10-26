# REFLECT: distinguish a budget floor from model capacity

Why compound failure? In-sample generation fails, not just unseen prompts.
Why call it capacity? We cannot yet: only one pass/rank has been measured.
Why not scale everything? It would conflate fit duration, rank, and coverage.

Challenge the earlier assumption that one epoch is enough. Challenge the idea
that all failures have the same cause: ordering-only already improves transfer,
while arithmetic-only does not. Challenge a future 100/100 declaration based on
the inspected test. A finite suite can be optimized without general improvement.

Clean next cut: fixed arithmetic rows/order, two ranks, fixed exposure ladder.
Evaluate fit at preregistered boundaries; test only the fixed endpoints. If both
reach high fit but transfer fails, stop treating rank as the remaining issue.
