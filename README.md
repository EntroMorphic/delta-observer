# Delta Observer

**Aaron (Tripp) Josserand-Austin** | EntroMorphic Research Team

📄 **Paper:** [Delta Observer (PDF)](paper/260123.delta_observer.pdf)
🔗 **Website:** [entromorphic.com](https://entromorphic.com)
📧 **Contact:** tripp@entromorphic.com

---

<p align="center">
  <img src="figures/delta_observer_training_hires.gif" alt="Delta Observer Training - Watch the Scaffolding Rise and Fall" width="100%">
</p>

<p align="center">
  <em>The Delta Observer's latent space during training, one frame per epoch. The rise and fall of clustering shown here is produced by the observer's supervised objective, not by the networks it watches (see <a href="#whats-new-october-2026">What's new</a>).</em>
</p>

---

## What's new (October 2026)

**1. A correction: transient clustering is an observer artifact.** A
pre-registered ablation
([results](research/reports/scaffolding_ablation_results.md)) finds the
following.
- **The pattern is real in the observer's latent space.** It appears on 5/5
  seeds, with peak silhouette 0.29 falling to −0.02 by the end.
- **But the observer creates it.** The rise needs the observer's carry
  supervision; the fall needs its bit-position head, a `sum(sum bits) mod 4`
  label the paper does not mention.
- **The source networks' own activations show no transient carry-count
  clustering** (silhouette about 0.03, 0/5 seeds), though carry count is
  linearly readable from them.
- **The online observer's advantage over PCA depends on that supervision.**
  It is +2.5 points against PCA of the same run, and negative on every seed
  without supervision.

So the "scaffolding rises and falls" reading below is no longer supported for
the *networks*. The original numbers are kept, with notes.

**2. New research: staleness in modular and pipeline training.** Summary:
[`research/reports/staleness_summary.md`](research/reports/staleness_summary.md).

<p align="center">
  <img src="research/reports/figures/pipeline_bound.png" alt="Pipeline staleness: almost all harm is directional" width="90%">
</p>

- **The permutation twin and bound (new tool).** Scramble a training error's
  entries within each tensor. The twin keeps the error's size, values and
  timing, and destroys only its direction. If the twin is harmless, the harm
  was directional, and training with the twin is a ceiling for any method
  that corrects direction.
- **Staleness harm is directional** (confirmed for Adam and SGD, on fresh
  seeds with frozen protocols). The real staleness error slows training about
  1.5–2×; its twin costs nothing measurable. The real error points into
  sharp directions, seeing 9–16× more curvature, and against the gradient.
- **Under pipeline staleness, almost all harm is directional** (exploratory).
  Adam fails beyond 5–10 steps of delay, while the twin trains to 40.
  - Basis Rotation (ICML 2026) recovers about a third of that gap.
  - First-order compensation (DC-ASGD, exact Taylor) recovers none.
- **Excess interface width costs staleness tolerance** (confirmed on two
  tasks).
- **Tune asynchronous runs under staleness** (exploratory). Tuning for
  synchronous speed picks edge-of-stability learning rates where staleness
  tolerance collapses.

All of this is on small networks. Nothing has been tested yet on a standard
large architecture.

**3. Optimizer maths and exact Hessians** of the paper's models: all four
pre-registered predictions held, including edge of stability within 0.4%
([results](research/reports/curvature_results.md)).

---

## Overview

This repository contains code, pre-computed data, and figures for our paper "Delta Observer: Learning Continuous Semantic Manifolds Between Neural Network Representations."

**Key Finding:** Semantic information in neural networks can be **linearly accessible** (R²=0.9879) without exhibiting **geometric clustering** (Silhouette=-0.02).

**Revised (October 2026):** the paper also reports that **clustering is transient**, existing during training and dissolving by the end. That pattern is real in the observer's latent space but is produced by the observer's supervised objective; it does not appear in the networks' own activations. See [What's new](#whats-new-october-2026).

**Method:** We train two architectures (monolithic and compositional) to solve 4-bit binary addition while a Delta Observer watches both models learn concurrently, discovering shared semantic structure through online observation.

### Observer.D experimental method

**Observer.D** extends Delta Observer from representation analysis to detached
behavioral transfer. It distills paired source states, activation deltas, and
training chronology into a student that runs without either source model or the
observer at inference. The controlled research harness compares direct
training, output distillation, and Observer.D guidance under AdamW and hybrid
Muon. Observer.D's signature encoder includes a native C/ARM NEON backend with
numerical-parity checks against PyTorch.

See [`research/README.md`](research/README.md) for objectives, controls, and
reproduction commands. Current results are preliminary and do not yet establish
a trajectory advantage.

**Current LLM research direction:** use observer-derived diagnostics to improve
functioning pretrained LLMs under controlled adaptation budgets. Standalone
compression experiments are paused. A native LoRA development pilot improved
JSON-record compliance; broader capability and observer-specific advantages
remain unproven. See the [active study plan](research/reports/pretrained_llm_improvement_plan.md)
and [measured pilot](research/reports/native_lora_pilot_results.md).

---

## 🚀 Run in Google Colab

**No installation required!** Open and run notebooks directly in your browser:

| Notebook | Description | Colab |
|----------|-------------|-------|
| **00_quickstart_demo** | Quick demo with pre-computed results | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EntroMorphic/delta-observer/blob/main/notebooks/00_quickstart_demo.ipynb) |
| **01_training_models** | Train source models from scratch (simplified) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EntroMorphic/delta-observer/blob/main/notebooks/01_training_models.ipynb) |
| **02_delta_observer_training** | Train Delta Observer | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EntroMorphic/delta-observer/blob/main/notebooks/02_delta_observer_training.ipynb) |
| **03_analysis_visualization** | Geometric analysis & paper figures | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EntroMorphic/delta-observer/blob/main/notebooks/03_analysis_visualization.ipynb) |
| **99_full_reproduction** | End-to-end online pipeline (simplified) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EntroMorphic/delta-observer/blob/main/notebooks/99_full_reproduction.ipynb) |

**📖 [Colab Setup Guide](notebooks/COLAB_SETUP.md)** - Detailed instructions for running notebooks in Google Colab

---

## Transient Clustering in the Observer Latent (revised)

> **Correction (October 2026).** The table below is reproducible, but it
> measures the *observer's* latent space. A pre-registered ablation
> ([results](research/reports/scaffolding_ablation_results.md)) shows:
> - the rise needs the observer's carry supervision;
> - the fall needs its bit-position head;
> - the source networks' own activations show no transient carry-count
>   clustering.
>
> The "Insight" paragraph is the original interpretation and is no longer
> supported.

The paper's analysis found that geometric clustering in the observer latent is **transient**:

| Training Phase | R² | Silhouette | Interpretation |
|----------------|-----|-----------|----------------|
| Early (epoch 0) | 0.38 | -0.02 | Random initialization |
| Learning (epoch 20) | 0.94 | **0.33** | Clustering emerges |
| Final (epoch 200) | 0.99 | -0.02 | Clustering dissolves |

**Original insight (no longer supported; see the correction above):** Clustering is scaffolding, not structure. Networks build geometric organization to *learn* semantic concepts, then discard that organization once the concepts are encoded in the weights. Post-hoc analysis only sees the final state and concludes "no clusters." But the clusters existed—they were temporary. The semantic primitive isn't in the final representation; it's in the learning trajectory.

---

## Online vs Post-hoc Observation

The Delta Observer was designed to watch training as it occurs. This matters:

| Method | R² | Silhouette | What it sees |
|--------|-----|-----------|--------------|
| **Online Observer** | **0.9879** | -0.02 | Full training trajectory |
| Post-hoc Observer | 0.9505 | 0.03 | Final state only |
| PCA Baseline | 0.9482 | 0.05 | Final state only |

Online observation beats PCA by **4%** because it captures temporal information unavailable to static analysis.

> **Note (October 2026).** The PCA baseline above uses separately trained legacy models. Against PCA of the same run's activations the advantage is +2.5 points, and it depends on the observer's carry supervision: without it, the observer trails PCA by 1.5–4.8 points on every seed ([results](research/reports/scaffolding_ablation_results.md)).

---

## Quick Start

### Online Delta Observer (Recommended)

Train all three models concurrently—the observer watches training as it happens:

```bash
python models/delta_observer.py
```

This single command:
1. Trains monolithic model on 4-bit addition
2. Trains compositional model on 4-bit addition
3. Trains Delta Observer while watching both
4. Saves latent representations and trajectory data
5. Compares to PCA baseline

### Analyze Results

```bash
python analysis/trajectory_analysis.py
python analysis/compare_methods.py
```

Scripts can be run from any directory. Data goes to `data/`, figures to `figures/`, model weights to `models/`,
and JSON logs/results to `logs/` (git-ignored).

---

## Repository Structure

```
delta-observer/
├── models/
│   ├── delta_observer.py            # PRIMARY: Online Delta Observer training
│   ├── delta_observer_hires.py      # Online observer, every-epoch snapshots (for movies)
│   ├── delta_observer_posthoc.py    # [Legacy] Post-hoc observer (baseline)
│   ├── train_4bit_monolithic.py     # [Legacy] Standalone monolithic training
│   └── train_4bit_compositional.py  # [Legacy] Standalone compositional training
├── analysis/
│   ├── trajectory_analysis.py       # Training trajectory analysis
│   ├── compare_methods.py           # Online vs post-hoc vs PCA comparison
│   ├── generate_online_figures.py   # Paper figures from online observer data
│   ├── create_training_movie*.py    # Training movies (MP4 requires ffmpeg)
│   ├── curriculum_test.py           # Temporal structure validation
│   ├── extract_activations.py       # [Legacy] Per-layer activation extraction
│   ├── geometric_analysis.py        # [Legacy] Geometric visualizations
│   ├── prepare_delta_dataset.py     # [Legacy] Post-hoc data extraction
│   └── analyze_delta_latent.py      # [Legacy] Post-hoc latent space analysis
├── data/
│   ├── online_observer_latents.npz  # Online observer outputs
│   ├── online_observer_trajectory.npz # Latent snapshots during training
│   └── delta_latent_umap.npz        # [Legacy] Post-hoc latents
├── notebooks/                       # Jupyter notebooks
├── figures/                         # Generated figures
├── paper/                           # Paper PDF and LaTeX source
└── research/                        # Observer.D, ablations, curvature and staleness research (see research/README.md)
```

---

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy, SciPy, scikit-learn, umap-learn, Matplotlib
- Optional: `ffmpeg` for the MP4 training movies (GIFs need no extra tools)

The Observer.D experiments in `research/` use a separate environment; see
[`research/lfm/requirements.txt`](research/lfm/requirements.txt) (tested with Python 3.12).

### Setup

```bash
git clone https://github.com/entromorphic/delta-observer.git
cd delta-observer
pip install -r requirements.txt
```

---

## Reproducing Results

Training scripts use a fixed seed (42), so reruns are repeatable on the same hardware and library
versions. They won't match the paper's numbers digit for digit: a seeded run gives online R² 0.993,
post-hoc 0.962 and PCA 0.959 (online beats PCA by 3.4%; the paper reports 4.0%).

### Full Pipeline (Recommended)

```bash
# Train online observer (trains all models concurrently)
python models/delta_observer.py

# Analyze trajectory and transient clustering
python analysis/trajectory_analysis.py

# Compare methods
python analysis/compare_methods.py
```

### Legacy Post-hoc Pipeline

For reproducing original paper results (before online observation):

```bash
# Train models separately
python models/train_4bit_monolithic.py
python models/train_4bit_compositional.py

# Geometric analysis of each model's activations
python analysis/extract_activations.py
python analysis/geometric_analysis.py

# Extract activations post-hoc
python analysis/prepare_delta_dataset.py

# Train post-hoc observer
python models/delta_observer_posthoc.py

# Analyze
python analysis/analyze_delta_latent.py
```

---

## Key Results

### Online Observer Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² (Linear Accessibility)** | 0.9879 | Semantic information is highly linearly accessible |
| **Silhouette (Clustering)** | -0.0242 | Points are not geometrically clustered |
| **Epoch Prediction R²** | 0.8523 | Latent space encodes temporal information |

### Method Comparison

| Method | R² | Δ vs PCA |
|--------|-----|---------|
| Online Observer | 0.9879 | **+4.0%** |
| Post-hoc Observer | 0.9505 | +0.2% |
| PCA Baseline | 0.9482 | — |

### Transient Clustering in the Observer Latent

The observer latent's Silhouette score during training. This pattern is produced by the observer's supervised objective, not by the source networks (see [What's new](#whats-new-october-2026)):

```
Epoch  0: -0.02 (no clustering)
Epoch 13:  0.16 (clustering emerging)
Epoch 20:  0.33 (peak clustering)
Epoch 50:  0.00 (clustering dissolving)
Epoch 200: -0.02 (no clustering)
```

**90% of final R² achieved by epoch 13**—the critical learning happens early.

---

## Notebooks

See the **[🚀 Run in Google Colab](#-run-in-google-colab)** section above for one-click notebook access.

| Notebook | Description |
|----------|-------------|
| `00_quickstart_demo.ipynb` | Quick demo with pre-computed results |
| `01_training_models.ipynb` | Train source models from scratch (simplified) |
| `02_delta_observer_training.ipynb` | Train Delta Observer |
| `03_analysis_visualization.ipynb` | Geometric analysis & paper figures |
| `99_full_reproduction.ipynb` | End-to-end online pipeline (simplified) |

Notebooks `00`, `02` and `03` analyze the paper's pre-computed online observer data. Notebooks `01` and `99`
train their own models on a **simplified 256-case setup** (a + b, no carry-in). In that variant the source
models reach 100% accuracy, but transient clustering does not dissolve and the online observer does not beat
PCA. To reproduce the paper's results, use the scripts in [Reproducing Results](#reproducing-results).
Notebook outputs go to `notebooks/outputs/` (git-ignored).

---

## Citation

```bibtex
@misc{josserandaustin2026deltaobserver,
  title={Delta Observer: Learning Continuous Semantic Manifolds Between Neural Network Representations},
  author={Josserand-Austin, Aaron N.},
  year={2026},
  month={January},
  url={https://github.com/EntroMorphic/delta-observer}
}
```

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Contact

**Aaron (Tripp) Josserand-Austin**
- Email: tripp@entromorphic.com
- Website: [entromorphic.com](https://entromorphic.com)

---

**For Science!** 🔬🌊
