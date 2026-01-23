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

This repository contains code, data, and trained models for our paper "Delta Observer: Learning Continuous Semantic Manifolds Between Neural Network Representations."

**Key Finding:** Semantic information in neural networks can be **linearly accessible** (R²=0.9879) without exhibiting **geometric clustering** (Silhouette=-0.02), and **clustering is transient**—it exists during training but dissolves in the final state.

**Method:** We train two architectures (monolithic and compositional) to solve 4-bit binary addition while a Delta Observer watches both models learn concurrently, discovering shared semantic structure through online observation.

---

## 🚀 Run in Google Colab

**No installation required!** Open and run notebooks directly in your browser:

| Notebook | Description | Colab |
|----------|-------------|-------|
| **00_quickstart_demo** | Quick demo with pre-computed results | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EntroMorphic/delta-observer/blob/main/notebooks/00_quickstart_demo.ipynb) |
| **01_training_models** | Train source models from scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EntroMorphic/delta-observer/blob/main/notebooks/01_training_models.ipynb) |
| **02_delta_observer_training** | Train Delta Observer | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EntroMorphic/delta-observer/blob/main/notebooks/02_delta_observer_training.ipynb) |
| **03_analysis_visualization** | Geometric analysis & paper figures | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EntroMorphic/delta-observer/blob/main/notebooks/03_analysis_visualization.ipynb) |
| **99_full_reproduction** | Complete end-to-end reproduction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EntroMorphic/delta-observer/blob/main/notebooks/99_full_reproduction.ipynb) |

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
python analysis/analyze_delta_latent.py
python analysis/trajectory_analysis.py
```

---

## Repository Structure

```
delta-observer/
├── models/
│   ├── delta_observer.py            # PRIMARY: Online Delta Observer training
│   ├── delta_observer_posthoc.py    # Legacy: Post-hoc observer (baseline)
│   ├── train_4bit_monolithic.py     # Standalone monolithic training
│   └── train_4bit_compositional.py  # Standalone compositional training
├── analysis/
│   ├── analyze_delta_latent.py      # Latent space analysis
│   ├── trajectory_analysis.py       # Training trajectory analysis
│   ├── curriculum_test.py           # Temporal structure validation
│   ├── geometric_analysis.py        # Geometric visualizations
│   └── prepare_delta_dataset.py     # [Legacy] Post-hoc data extraction
├── journal/                         # Research journal and methodology
│   ├── findings.md                  # Analysis findings
│   ├── PRC.md                       # Falsification results
│   └── LMM.md                       # Lincoln Manifold Method
├── data/
│   ├── online_observer_latents.npz  # Online observer outputs
│   ├── online_observer_trajectory.npz # Latent snapshots during training
│   └── delta_latent_umap.npz        # [Legacy] Post-hoc latents
├── notebooks/                       # Jupyter notebooks
├── figures/                         # Generated figures
└── paper/                           # Paper PDF
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- scikit-learn
- Matplotlib

### Setup

```bash
git clone https://github.com/entromorphic/delta-observer.git
cd delta-observer
pip install -r requirements.txt
```

---

## Reproducing Results

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
| `01_training_models.ipynb` | Train source models from scratch |
| `02_delta_observer_training.ipynb` | Train Delta Observer |
| `03_analysis_visualization.ipynb` | Geometric analysis & paper figures |
| `99_full_reproduction.ipynb` | Complete end-to-end reproduction |

*All notebooks use the online observer data and include transient clustering analysis.*

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

## Research Journal

The `journal/` directory contains the research process:

- **`findings.md`** - Complete analysis and conclusions
- **`PRC.md`** - Falsification tests and results
- **`LMM.md`** - Lincoln Manifold Method (our exploration methodology)
- **`delta_observer_*.md`** - Raw thinking, nodes, reflections, synthesis

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
