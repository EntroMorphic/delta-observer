# Same-history forks: early output history mediates much of the observed gap

## Outcome and scope

**Nonnumeric native prefixes recover much of the adapted arithmetic content
without disabling adapters or supplying arithmetic results.** Under the fixed
reasoning-request prompt, ordinary seeds rise from **15→38**, **5→36**, and
**34→39/40** confirmed content. This is stronger evidence for conditional access
to useful arithmetic than the earlier prompt-only control.

It is not complete recovery, exact-answer compliance, a fresh confirmation,
or a unique ultimate cause. Seed 2 loses a previously correct example. The
candidate remains rejected. No training occurred.

The experiment follows `native_same_history_protocol.md`, frozen before new
inference, on all 40 exposed arithmetic cases and ordinary seeds 0, 1, 2.
Original scores and prior diagnostic artifacts are unchanged.

## What was held identical

For each prefix, adapter-on and scale-zero adapter-off recipients receive the
**same full input token IDs**, including user prompt and supplied assistant
prefix. Both recompute the graph from scratch; no donor cache is transferred.
This is a same-**history** parameter intervention, not a same-hidden-state swap.

Donor prefixes are taken from already recorded native/ordinary generations.
They are capped at 16 tokens and end before the first numeric character or
English cardinal-number word; partial cardinal tokens are excluded too.
No expected answers, worked values or calculated hints choose or enter a prefix.
They can still contain routing cues such as `initial number of tokens` or
`\boxed`; absence of numerals does not make a prefix semantically inert.

Original and reasoning-request user prompts are both tested. The assistant's
192-token total budget includes supplied prefixes. Identical histories within
a seed are executed once and explicitly aliased; duplicates are not independent
observations. First-token readouts use an additional uncached forward.

## Confirmed-content outcomes, full assistant response

All counts are out of 40. Prefix + continuation is scored using unchanged
original exact/boxed-content validators. No prefix is silently stripped to
claim native compliance.

### Original user prompt

| Seed | Empty prefix off / on | Native prefix off / on | Adapted prefix off / on |
|---|---:|---:|---:|
| 0 | 39 / 6 | 39 / **14** | 39 / 6 |
| 1 | 39 / 7 | 39 / **15** | 39 / 7 |
| 2 | 39 / 10 | 39 / **18** | 39 / 10 |

Only eight native original prefixes are nonempty: seven `Each box contains `
and one `The collection initially had \(`. These eight cases recover under
adapter-on in every seed, with no content loss relative to own empty-prefix
original. All adapted original prefixes are empty because their generations
start with numerical answers; their rows are alias controls, not new evidence.

Original adapter-on **native exact** remains 6, 7, 10/40 for every prefix
condition. The eight recovered cases are worked answers, not improvements in
the requested bare-integer contract. Adapter-off native exact remains zero.

### Fixed reasoning-request user prompt

| Seed | Empty prefix off / on | Native prefix off / on | Adapted prefix off / on |
|---|---:|---:|---:|
| 0 | 40 / 15 | 40 / **38** | **15 / 15** |
| 1 | 40 / 5 | 40 / **36** | **5 / 5** |
| 2 | 40 / 34 | 40 / **39** | **34 / 34** |

Native reasoning prefixes are nonempty in 36/40 cases and comprise only:

- `The collection initially has \(` — 15 cases;
- `The initial number of tokens is \(` — 11;
- `The collection originally has \(` — 8;
- `The collection initially had \(` — 2.

Four native reasoning generations start with numbers and therefore supply no
prefix. No output meets the original native bare-integer exact contract in any
reasoning-mode condition. None hits the output budget.

### Paired native-prefix gains/losses vs own empty prefix, adapters on

| Prompt mode | Seed | Gains | Losses |
|---|---:|---:|---:|
| Original | 0 | 8 | 0 |
| Original | 1 | 8 | 0 |
| Original | 2 | 8 | 0 |
| Reasoning request | 0 | 23 | 0 |
| Reasoning request | 1 | 31 | 0 |
| Reasoning request | 2 | 6 | **1** |

The seed-2 loss is `fresh-add-01`. The six seed-2 gains are
`fresh-subtract-10`, `fresh-add-10`, `fresh-subtract-11`, `fresh-add-11`,
`fresh-subtract-15`, and `fresh-add-15`. Every seed's complete paired IDs are
in `summary.json`; no loss is hidden by matched totals.

## Reverse intervention: an adapted opening can also damage the native path

With adapters **off**, native reasoning accuracy is 40/40 from an empty or
native prefix but only 15/40, 5/40, 34/40 from same-seed adapted prefixes.
Thus some native failures can be induced by early generated history alone,
without activating any learned adapter parameters during the continuation.

Adapted reasoning prefixes split into two broad classes:

| Seed | Opening `\boxed` / `\boxed{` | Natural-language working preamble |
|---|---:|---:|
| 0 | 25 | 15 |
| 1 | 35 | 5 |
| 2 | 6 | 34 |

All native adapter-off failures after these adapted prefixes are **parseable
wrong boxed answers**, not merely malformed outputs rejected by the parser.
All adapter-off successes correspond to the working-preamble class. This
correspondence is measured on these donor histories, not proof that the same
opening causes the same behavior on every task or model.

For `fresh-add-00` (18×6+4=112), seed 1 starts its adapted reasoning output with
`\boxed`. Supplying that nonnumeric prefix to the native adapter-off model
yields **`\boxed{20}`**; the same native model with an empty history or native
working preamble computes 108+4 and boxes **112**. The prefix changes available
generation behavior without transferring a numerical answer.

## Same-history parameter effects remain measurable

Output history is important, but it is not the **only** measured mediator.
With identical empty original histories, adapter-on and adapter-off next-token
top-1 differ in **40/40 cases in every seed**. Mean full-distribution JS is
0.683895, 0.686918, 0.686946 nats (maximum possible ln 2 ≈ 0.693147).
These changes occur before any differing assistant token is generated.

Under reasoning-request native prefixes, on/off next-token top-1 agrees in
36/40 cases for every seed; mean JS falls to 0.027865, 0.029249, 0.022664 nats.
Nevertheless final continuations are token-identical in only 5/40 cases per
seed. Matching the next token is not a guarantee of matching later computation
or the final answer. JS values are recorded paired readouts; their full source
distribution vectors were not persisted for independent recomputation.

### Residual genuine arithmetic error

Seed 2, `fresh-add-01`, receives native prefix
`The initial number of tokens is \(`. Both recipients have identical input
tokens. Adapter-off computes 16×4=64, then 64+7=**71**. Adapter-on instead
writes that the collection gains 7×64=448 and concludes **512**. This is a
parseable wrong answer, not a scorer ambiguity or an injected wrong value.
Its own empty-prefix adapted reasoning response had correctly answered 71.

Remaining native-prefix reasoning non-successes in seeds 0 and 1 are confined
to cases with **empty** prefixes and malformed unconfirmed boxes:

- Seed 0: `fresh-subtract-05`, `fresh-add-05`.
- Seed 1: those two plus `fresh-subtract-08`, `fresh-subtract-09`.

No broad semantic review or parser expansion was performed on these unresolved
outputs. The results do not justify reporting them as correct numeric answers.

## Controls, coverage and runtime

- **1,440** case/condition records, **720** paired identical histories.
- **984** unique model executions; **456** explicit alias records.
- **480/480** empty-prefix engineering controls reproduce prior generations.
- **960/960** self-replay checks agree with stored donor remainder token IDs,
  including nonempty donor-recipient re-prefill; zero self-control mismatches.
  Counts include registered alias rows, not 960 independent executions.
- Independent artifact reconstruction verifies prefix provenance, numeric and
  control-token exclusions, all complete input histories, output budgets,
  aliases, full response scores, summary reproduction and immutable source/
  checkpoint/protocol/implementation hashes.
- Measured unique execution runtime: **337.720711 seconds**; generated tokens:
  **29,718**. Includes extra full-history readouts and generation, excludes
  initialization, tokenization, pairwise CPU distribution analysis and file I/O.
- No training, hidden calculator, answer-dependent mask, target-valued prefill,
  fresh held-out evaluation, deployment improvement or inference speedup claim.

## Bounded causal interpretation and next step

Supported: early, nonnumeric assistant history is a substantial, intervenable
mediator of these exposed arithmetic outcomes. Native-style working openings
allow the adapted model to perform many computations that its usual greedy
path misses. Conversely, adapted boxed-answer openings can make the otherwise
successful native continuation fail. This is inconsistent with describing all
adapted errors as total erasure of arithmetic capability.

Also supported: active adapters still change predictions at identical histories,
and some genuine computation/operation-binding errors persist after a working
opening. History routing is not the unique cause of every failure. Identical
tokens do not establish identical intermediate representations or distinguish
changed computation from changed readout.

The intervention uses task-specific **donor-selected wording**, even though no
numbers are supplied. It has not established that one fixed preamble transfers
to fresh facts/templates, preserves native capabilities, or meets bare-answer
instructions. It also spends more serial generation work. No candidate is
approved, and no seed is presented as a broad solution.

Next discriminating step: test a **single preregistered nonnumeric opening**,
independent of every task/donor, then localize the remaining same-history
adapter effects with controlled residual-state interventions. A new evaluation
must be locked before any resulting training or confirmatory improvement claim.
Those are proposed follow-ups, not results of this experiment.

## Artifacts

- `research/results/native_same_history_v1/forks.json`: raw cases, prefixes,
  execution aliases, first-token paired readouts and full summary.
- Same directory: frozen manifest, per-seed records, `summary.json`.
- `research/results/native_same_history_v1_artifact_audit.json`: independent
  provenance/scoring/input reconstruction, prefix counts and replay controls.
- Executors: `audit_native_same_history.py`, `audit_same_history_artifacts.py`;
  tests: `test_native_same_history.py`.
