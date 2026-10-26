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
  <em>Watch neural networks build scaffolding to learn, then tear it down. Every frame is one epoch of training.</em>
</p>

---

## Overview

This repository contains code, pre-computed data, and figures for our paper "Delta Observer: Learning Continuous Semantic Manifolds Between Neural Network Representations."

**Key Finding:** Semantic information in neural networks can be **linearly accessible** (R²=0.9879) without exhibiting **geometric clustering** (Silhouette=-0.02), and **clustering is transient**—it exists during training but dissolves in the final state.

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

## Key Discovery: Transient Clustering

Our analysis revealed that geometric clustering is not absent—it's **transient**:

| Training Phase | R² | Silhouette | Interpretation |
|----------------|-----|-----------|----------------|
| Early (epoch 0) | 0.38 | -0.02 | Random initialization |
| Learning (epoch 20) | 0.94 | **0.33** | Clustering emerges |
| Final (epoch 200) | 0.99 | -0.02 | Clustering dissolves |

**Insight:** Clustering is scaffolding, not structure. Networks build geometric organization to *learn* semantic concepts, then discard that organization once the concepts are encoded in the weights. Post-hoc analysis only sees the final state and concludes "no clusters." But the clusters existed—they were temporary. The semantic primitive isn't in the final representation; it's in the learning trajectory.

---

## Online vs Post-hoc Observation

The Delta Observer was designed to watch training as it occurs. This matters:

| Method | R² | Silhouette | What it sees |
|--------|-----|-----------|--------------|
| **Online Observer** | **0.9879** | -0.02 | Full training trajectory |
| Post-hoc Observer | 0.9505 | 0.03 | Final state only |
| PCA Baseline | 0.9482 | 0.05 | Final state only |

Online observation beats PCA by **4%** because it captures temporal information unavailable to static analysis.

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
└── research/                        # Observer.D experiments (see research/README.md)
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

### Transient Clustering Discovery

The Silhouette score evolution during training:

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
