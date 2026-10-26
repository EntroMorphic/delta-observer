# Transient-clustering factor ablation protocol

Frozen before any sweep run. This is an exploratory one-factor-at-a-time
ablation, not a sealed confirmatory study. It is not blind: both endpoints were
already observed during the 2026-10-26 repository audit (paper scripts,
unseeded rerun: silhouette 0.283 → −0.044; notebook 99 variant, seeds 42/0/1:
peak 0.11–0.15, final 0.08–0.11, online observer 0.7–2.1% below PCA).

## Question

The paper reports *transient clustering*: the silhouette of the online
observer's latent space, labelled by carry count, rises during training and
then dissolves (0.33 → −0.02). The paper's scripts (`models/delta_observer.py`)
show it; the simplified notebook pipeline (`notebooks/99_full_reproduction.ipynb`)
does not. Which of the differences between the two pipelines controls the
effect?

## Factors

| Factor | Paper value (P) | Notebook value (N) |
|---|---|---|
| `data` | 512 cases with carry-in bit | 256 cases, no carry-in |
| `loss` | MSE on linear outputs | BCE with logits |
| `schedule` | Adam, peak α = 1/137, 10-epoch warmup, per-step cosine to α/1000 | Adam, constant 0.01 |
| `mono` | 3 hidden layers; observer sees layer 1 | 2 hidden layers; observer sees layer 2 |
| `comp` | per-bit full adders (carry from output); observer sees each module's layer 1 | per-bit modules, learned carry heads, shared output layer; observer sees each module's last layer |
| `observer` | recon + bit-position CE + 0.1·carry; cosine LR | recon + 0.1·carry; constant LR |
| `batch` | 32 | 64 |

`snapshot` (observer in train mode, so dropout is active, vs eval mode) is a
measurement factor. Snapshots run under an isolated RNG, so both modes are
recorded from the same trained models in every run.

## Design

- Baselines P and N, plus each factor flipped alone in both directions
  (P with one factor at its N value; N with one factor at its P value):
  16 configurations.
- Seeds 0–4 for every configuration (80 runs), 200 epochs, every-epoch
  snapshots, CPU, one thread per process.
- Shared harness (`research/experiments/scaffolding_ablation.py`). Batches use
  per-epoch `randperm` for every configuration. The harness's P configuration,
  run in replication mode, must reproduce `models/delta_observer.py`
  snapshots exactly (unit test) before any sweep result is read.

## Metrics

Per epoch, on the observer latent for all cases, labelled by carry count:
silhouette and linear-regression R², in both snapshot modes.

- `peak_sil`: maximum silhouette; `peak_epoch`: its epoch.
- `final_sil`: mean silhouette over the last 10 epochs.
- **Transient clustering present** iff `peak_sil ≥ 0.10` and
  `final_sil < 0.5 · peak_sil`.
- `online_r2`: final eval-mode latent R²; `pca_r2`: R² of a 16-component PCA
  of the same run's final concatenated source activations; `online_minus_pca`.
- Final source accuracies (all five output bits correct).

The paper's PCA baseline used activations from separately trained legacy
models (`data/delta_observer_dataset.npz`). This ablation uses the same run's
models instead. The difference is reported, not corrected after the fact.

## Decision rules

- A factor *controls* the effect in one direction if flipping it alone
  changes the presence classification in ≥ 4 of 5 seeds relative to that
  baseline's majority classification.
- If no single flip changes the classification, the effect is attributed to
  an interaction, and no single factor is named.
- Seeds where a source model ends below 100% accuracy are reported and
  flagged; they are not excluded.

## Prediction (recorded before running)

`schedule` and `loss` matter most; `data` least.

## Interpretation limits

One toy task, 5 seeds, one latent dimensionality. Silhouette with five
unequal-sized carry classes is a coarse clustering measure. A null result
for a factor means "not detected at this sample size", not "irrelevant".
Nothing here tests whether transient clustering occurs in other tasks or
architectures.
