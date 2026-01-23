# Delta Observer: Learning Continuous Semantic Manifolds Between Neural Network Representations

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
│   └── README.md                    # Notebook documentation
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

Both models achieve 100% accuracy on 4-bit addition (512 possible inputs).

### 2. Extract Activations

Extract hidden layer activations from both models:

```bash
python analysis/prepare_delta_dataset.py
```

This creates `data/delta_observer_dataset.npz` with:
- Monolithic activations (512 × 64D)
- Compositional activations (512 × 64D)
- Carry count labels (0-4)
- Bit position labels (0-3)

### 3. Train Delta Observer

Train the Delta Observer to map between representations:

```bash
python models/delta_observer.py
```

The Delta Observer learns a 16D latent space that:
- Reconstructs both activation spaces
- Predicts semantic properties (carry count, bit position)
- Discovers continuous semantic structure

### 4. Analyze Results

Analyze the Delta Observer's latent space:

```bash
python analysis/analyze_delta_latent.py
```

This generates:
- UMAP visualizations colored by carry count and bit position
- Linear probe analysis (R² scores)
- Clustering analysis (Silhouette scores)
- 6-panel comparison figure

---

## Reproducing Paper Results

To reproduce all results from the paper:

```bash
# 1. Train both models
python models/train_4bit_monolithic.py
python models/train_4bit_compositional.py

# 2. Perform geometric analysis of original models
python analysis/geometric_analysis.py

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

| Model | Parameters | Accuracy | Structure | Silhouette |
|-------|------------|----------|-----------|------------|
| Monolithic | 9,285 | 100% | Magnitude-based | 0.5551 |
| Compositional | 1,480 | 100% | Bit-position-based | 0.8060 |

**Finding:** 6× parameter efficiency with better geometric structure.

### Delta Observer Latent Space

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Linear probe R² (carry) | 0.9505 | Excellent linear accessibility |
| Silhouette score | 0.0320 | Minimal clustering |
| PCA variance (2D) | 93.1% | Low-dimensional structure |

**Finding:** Semantic information is linearly accessible without geometric clustering.

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

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

EntroMorphic Research Team thanks the specific Manus instance for computational resources, web research, coding, and feedback. We thank the specific Claude instance hosted by Anthropic for coding and providing detailed feedback on the Delta Observer methodology during the design phase.

---

## Contact

**Aaron (Tripp) Josserand-Austin**
EntroMorphic Research Team
📧 tripp@entromorphic.com
🔗 [entromorphic.com](https://entromorphic.com)

For questions about the paper or code, please open an issue or contact us directly.
