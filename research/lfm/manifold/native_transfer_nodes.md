# NODES: partition the problem before selecting interventions

1. **Objective:** native-model capability gain with per-category retention;
   speed and compression are separate outcomes.
2. **Fit:** rank-8 arithmetic reaches 125/128 after four passes, not 128/128.
3. **Transfer:** endpoint arithmetic is 8/20 on exposed new prompts, but both
   wording and facts differ. The comparison is not a pure wording intervention.
4. **Capacity:** rank 32 fits 116/128 with 9/20 transfer at the same exposure
   count. Larger rank is not the demonstrated solution in this schedule.
5. **Instruction sensitivity:** earliest/latest training pairs fit perfectly;
   unseen forms still fail. Ordering-only improves 10/20 to 18/20.
6. **Interference:** bundled ordering loses to specialist, but specialist
   removal also changes optimizer history. Exact gradient cause remains open.
7. **Path sensitivity:** compact versus spaced JSON changes copy errors even
   in the untouched model; syntax and semantic success must be measured apart.
8. **Coverage:** copied-value and semantic-operator coverage require explicit
   auditing; counts of unique prompt strings conceal shortcuts.
9. **Curvature:** factor-space directional sensitivity is measured locally,
   not a full native-weight Hessian or a speed/capability result.
10. **Evidence:** repeated inspected test suites are diagnostics; task/wording
    families must be split before intervention for future confirmation.

Tensions and dependencies:

- Nodes 2/3: fit versus transfer → fact/wording crossing is prerequisite to a
  causal account of the generalization gap.
- Nodes 5/6: specialization versus retention → integration needs a matched
  replay/update control, not blind adapter merging.
- Nodes 7/8: exact compliance versus content → the response path can change
  which latent weakness becomes observable.
- Nodes 1/9: interesting curvature versus useful improvement → sensitivity
  prediction must be validated before it determines optimization or pruning.
