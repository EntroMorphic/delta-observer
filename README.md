# Delta Observer

**Aaron (Tripp) Josserand-Austin** | EntroMorphic Research Team

📄 **Paper:** [OSF MetaArXiv Preprint](https://doi.org/10.17605/OSF.IO/CNJTP)
🔗 **Website:** [entromorphic.com](https://entromorphic.com)
📧 **Contact:** tripp@entromorphic.com

---

## Overview

This repository contains code, data, and trained models for our paper "Delta Observer: Learning Continuous Semantic Manifolds Between Neural Network Representations."

**Key Finding:** Semantic information in neural networks can be **linearly accessible** (R²=0.9505) without exhibiting strong **geometric clustering** (Silhouette=0.0320), challenging the assumption that interpretability requires discrete feature clusters.

**Method:** We train two architectures (monolithic and compositional) to solve 4-bit binary addition, then train a Delta Observer to map between their representation spaces, discovering shared semantic structure.

---

## 🚀 Run in Google Colab

**No installation required!** Open and run notebooks directly in your browser:

| Notebook | Description | Runtime | Colab Link |
|----------|-------------|---------|------------|
| **00_quickstart_demo** | Quick demo with pre-computed results | 5 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EntroMorphic/delta-observer/blob/main/notebooks/00_quickstart_demo.ipynb) |
| **01_training_models** | Train source models from scratch | 15 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EntroMorphic/delta-observer/blob/main/notebooks/01_training_models.ipynb) |
| **02_delta_observer_training** | Train Delta Observer | 20 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EntroMorphic/delta-observer/blob/main/notebooks/02_delta_observer_training.ipynb) |
| **03_analysis_visualization** | Geometric analysis & paper figures | 10 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EntroMorphic/delta-observer/blob/main/notebooks/03_analysis_visualization.ipynb) |
| **99_full_reproduction** | Complete end-to-end reproduction | 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EntroMorphic/delta-observer/blob/main/notebooks/99_full_reproduction.ipynb) |

**📖 [Colab Setup Guide](notebooks/COLAB_SETUP.md)** - Detailed instructions for running notebooks in Google Colab

---

## Repository Structure

```
delta-observer/
├── models/                          # Model architectures and training scripts
│   ├── train_4bit_monolithic.py     # Train monolithic MLP
│   ├── train_4bit_compositional.py  # Train compositional modular network
│   └── delta_observer.py            # Delta Observer architecture
├── analysis/                        # Analysis scripts
│   ├── prepare_delta_dataset.py     # Extract activations and prepare dataset
│   ├── analyze_delta_latent.py      # Analyze Delta Observer latent space
│   └── geometric_analysis.py        # Geometric analysis of representations
├── data/                            # Datasets and activations
│   ├── monolithic_activations.npz
│   ├── compositional_activations.npz
│   ├── delta_observer_dataset.npz
│   └── delta_latent_umap.npz
├── notebooks/                       # Jupyter notebooks for interactive exploration
│   ├── 00_quickstart_demo.ipynb     # Quick demo with pre-computed results (5 min)
│   ├── 01_training_models.ipynb     # Train source models from scratch
│   ├── 02_delta_observer_training.ipynb  # Train Delta Observer
│   ├── 03_analysis_visualization.ipynb   # Geometric analysis & paper figures
│   ├── 99_full_reproduction.ipynb   # Complete end-to-end reproduction (30 min)
│   ├── README.md                    # Notebook documentation
│   └── COLAB_SETUP.md               # Google Colab setup guide
├── figures/                         # Generated figures from paper
│   ├── figure1_model_geometries.png
│   ├── figure2_delta_latent_space.png
│   ├── figure3_accessibility_vs_clustering.png
│   ├── figure4_architecture_diagram.png
│   ├── figure5_training_curves.png
│   └── figure6_perturbation_stability.png
├── paper/                           # Paper PDF
│   └── delta_observer_arxiv.pdf
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── CITATION.bib                     # BibTeX citation
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- scikit-learn
- UMAP-learn
- Jupyter (for interactive notebooks)

### Setup

```bash
# Clone the repository
git clone https://github.com/entromorphic/delta-observer.git
cd delta-observer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Train Models

Train the monolithic and compositional models on 4-bit addition:

```bash
# Train monolithic model
python models/train_4bit_monolithic.py

# Train compositional model
python models/train_4bit_compositional.py
```

### 2. Extract Activations

```bash
python analysis/prepare_delta_dataset.py
```

### 3. Train Delta Observer

```bash
python models/delta_observer.py
```

### 4. Analyze Results

```bash
python analysis/analyze_delta_latent.py
python analysis/geometric_analysis.py
```

---

## Reproducing Paper Results

### Option 1: Python Scripts

```bash
# 1. Train both models
python models/train_4bit_monolithic.py
python models/train_4bit_compositional.py

# 2. Extract activations
python analysis/prepare_delta_dataset.py

# 3. Prepare Delta Observer dataset
python analysis/prepare_delta_dataset.py

# 4. Train Delta Observer
python models/delta_observer.py

# 5. Analyze Delta Observer latent space
python analysis/analyze_delta_latent.py
```

All figures will be saved to `figures/`.

---

## Jupyter Notebooks

We provide interactive Jupyter notebooks for exploration and reproduction:

### Quick Start (5 minutes)

**[00_quickstart_demo.ipynb](notebooks/00_quickstart_demo.ipynb)** - Fast demonstration with pre-computed results

```bash
jupyter notebook notebooks/00_quickstart_demo.ipynb
```

This notebook:
- Loads pre-computed Delta Observer latent space
- Visualizes 2D projections (PCA & UMAP)
- Computes R² (linear accessibility) = 0.9384
- Computes Silhouette (clustering) = 0.0320
- **Demonstrates the accessibility-clustering paradox**

### Step-by-Step Workflow

For a detailed walkthrough of the entire pipeline:

1. **[01_training_models.ipynb](notebooks/01_training_models.ipynb)** - Train source models
   - Generate 4-bit addition dataset
   - Train monolithic MLP
   - Train compositional network
   - Extract activations
   - Compare architectures

2. **[02_delta_observer_training.ipynb](notebooks/02_delta_observer_training.ipynb)** - Train Delta Observer
   - Prepare dataset with semantic labels
   - Define dual-encoder architecture
   - Train with multi-objective loss
   - Extract 16D latent space

3. **[03_analysis_visualization.ipynb](notebooks/03_analysis_visualization.ipynb)** - Analysis & figures
   - PCA variance analysis
   - Linear probe analysis (R²)
   - Clustering analysis (Silhouette)
   - Generate all paper figures
   - Perturbation stability testing

### Complete Reproduction (30 minutes)

**[99_full_reproduction.ipynb](notebooks/99_full_reproduction.ipynb)** - End-to-end reproduction

```bash
jupyter notebook notebooks/99_full_reproduction.ipynb
```

This notebook:
- Trains all models from scratch
- Generates all data files
- Creates all figures
- Validates paper findings
- **Fully self-contained reproduction**

See [notebooks/README.md](notebooks/README.md) for detailed documentation.

---

## Key Results

### Monolithic vs Compositional Representations

| Model | Test Accuracy | Hidden Dimension | Architecture |
|-------|--------------|------------------|--------------|
| Monolithic | 50.78% | 64D | Single MLP |
| Compositional | 38.28% | 4×16D | Modular per-bit |

### Delta Observer Latent Space Analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² (Linear Accessibility)** | 0.9505 | Semantic information is **highly linearly accessible** |
| **Silhouette (Clustering)** | 0.0320 | Points are **not geometrically clustered** |
| **Latent Dimension** | 16D | Compact semantic representation |

**Key Insight:** High R² with low Silhouette demonstrates that semantic information can be linearly accessible without requiring discrete geometric clusters.

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{josserandaustin2026deltaobserver,
  title={The Delta Observer},
  author={Josserand-Austin, Aaron N.},
  year={2026},
  month={January},
  day={22},
  publisher={OSF},
  doi={10.17605/OSF.IO/CNJTP},
  url={https://doi.org/10.17605/OSF.IO/CNJTP}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Aaron (Tripp) Josserand-Austin**
- Email: tripp@entromorphic.com
- Website: [entromorphic.com](https://entromorphic.com)
- GitHub: [@EntroMorphic](https://github.com/EntroMorphic)

---

**For Science!** 🔬🌊
