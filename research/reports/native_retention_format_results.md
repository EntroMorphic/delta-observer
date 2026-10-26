# Retention-first worked-supervision comparison: rejected

## Decision

The worked-replay candidate fails its preregistered acceptance gate. It does
not improve unseen-fact/unseen-template transfer over either control and loses
baseline-confirmed content. No model from this comparison is approved as a
retained-capability replacement. 100/100 remains unmet.

## Fixed evaluation

Protocol: `native_retention_format_protocol.md`, frozen before building data.
The 100-task suite was hashed before training; no intermediate test selection
occurred. Sixty arithmetic cases cover three registered splits, twenty each;
forty retention tasks cover JSON, sorting, ordering, and Spanish, ten each.

| Exact task success | Untouched | Final-only | Worked replay | Extra-final |
|---|---:|---:|---:|---:|
| Seen facts / new template | 0/20 | 1/20 | 1/20 | 2/20 |
| New facts / seen template | 0/20 | 6/20 | 6/20 | 6/20 |
| New facts / new template | 0/20 | 4/20 | 0/20 | 1/20 |
| Retention tasks | 6/40 | 4/40 | 7/40 | 3/40 |
| Total | 6/100 | 15/100 | 14/100 | 12/100 |

The baseline's low unseen-template arithmetic performance is reported, not
used to redesign the frozen test. Previously observed good native explanatory
arithmetic does not generalize to every prompt family, as this suite confirms.

## Content retention, separately measured

| Confirmed parsed-content successes | Untouched | Final-only | Worked replay | Extra-final |
|---|---:|---:|---:|---:|
| All 60 arithmetic tasks | 11 | 11 | 7 | 9 |
| Retention tasks | 16 | 14 | 8 | 13 |
| Total | 27 | 25 | 15 | 22 |
| Lost baseline-confirmed successes, paired | — | 10 | 17 | 13 |

Worked replay preserves the six baseline exact successes and gains one new
retention exact success, yet loses seventeen baseline content successes: eight
arithmetic and nine JSON records. Some records become malformed fenced JSON.
This is why exact compliance or aggregate retention count cannot replace paired
content tracking. The frozen parser conservatively leaves nonmatching plain
integer answers unresolved rather than classifying semantic error; this does
not turn any into confirmed successes or override failed gates. Scoring rules
were not changed after observing outputs.

## What the experiment actually tested

- Fresh pinned native LFM per arm; rank 8 / alpha 16 attention/conv LoRA.
- Original base weights frozen and verified unchanged in every run.
- 128 AdamW updates; 48 real supervised positions per update; 6,144 per arm.
- Token-weighted accumulation across unpadded microbatches; clipping once/update.
- Same main final-answer sequence registered at each update.
- Worked replay adds an algorithmically correct worked response for that fact,
  then primary final-answer fillers; it is not literal replay of native model
  responses, not a KL preservation objective, and not a native learning trace.
- Final-only fillers come from primary facts; extra-final fillers from a disjoint
  extra fact pool. No artificial supervised padding or truncated EOS.

| Budget / measurement | Final-only | Worked replay | Extra-final |
|---|---:|---:|---:|
| Updates | 128 | 128 | 128 |
| Supervised tokens | 6,144 | 6,144 | 6,144 |
| Input tokens | 122,880 | 69,504 | 122,880 |
| Sequence exposures | 3,072 | 1,664 | 3,072 |
| Distinct prompts | 128 | 256 | 256 |
| Worked-sequence exposures | 0 | 128 | 0 |
| Measured training-loop seconds | 218.432277 | 127.361664 | 224.353948 |

Times include checkpoint writes, exclude initial loading/validation, and are
not a general inference-speed result. These are equal-supervised-token and
equal-update comparisons, not equal input/compute or fact-exposure comparisons.
Long worked responses change the mix of factual examples seen; these confounds
remain explicit. One seed and synthetic templates cannot establish a universal
replay failure or a general capability conclusion.

## Split integrity

128 primary and 128 extra operand triples are disjoint; all forty unseen triples
are disjoint from both. The seen-fact split deliberately reuses primary facts,
not training prompts. Training templates 0/1 and evaluation-only template 2
were separated before training. Normalized exact-prompt checks include all
currently registered test/development suites. This is not semantic or upstream
pretraining decontamination.

## Red-team / engineering checks

Tests cover exact token packing without repeated auxiliary rows within one
update, gradient equivalence for supervised-token weighting, fact split
disjointness, and invalid group budgets. The trainer re-tokenizes every row,
verifies complete labels/EOS, data hashes and all manifest budgets, and rejects
nonfinite loss/gradients. The scorer verifies frozen task/manifest hashes,
matching generation settings, exact task IDs/prompts, and each adapter's arm
and manifest provenance before computing results. This does not certify the
entire pipeline as pristine.

## Next justified direction

Do not scale this failed worked-trace bundle or claim it preserved native
reasoning. The next retention mechanism, if pursued, must directly constrain
actual native behavior on a representative retention set, compared to a
standard supervised baseline at a separately locked budget. Before new
training, audit native correctness across development template families;
previous success on one family is not a universal native arithmetic baseline.
Semantic-orbit/contrast training remains untested as a separate intervention.
The current test is now exposed and cannot remain a final test for a subsequent
intervention it informs.

## Evidence and reproduction

- Data/splits/manifest hashes: `research/results/native_retention_format_v1_freeze/`
- Scores and failed gates: `research/results/native_retention_format_v1_comparison.json`
- Training reports: `research/results/native_retention_format_v1_{final_only,worked_replay,extra_final}_training.json`
- Raw responses: `research/results/native_retention_format_v1_{baseline,final_only,worked_replay,extra_final}/baseline.json`
- Executables: `build_retention_format`, `train_retention_format`,
  `evaluate_pretrained_development`, `score_retention_format`.
