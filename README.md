# Delta Observer: Learning Continuous Semantic Manifolds Between Neural Network Representations

**Aaron (Tripp) Josserand-Austin** | EntroMorphic Research Team

📄 **Paper:** [ArXiv preprint arXiv:2601.XXXXX](https://arxiv.org/abs/2601.XXXXX)
🔗 **Website:** [entromorphic.com](https://entromorphic.com)
📧 **Contact:** tripp@anjaustin.com

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
│   ├── train_4bit_monolithic.py    # Train monolithic MLP
│   ├── train_4bit_compositional.py # Train compositional modular network
│   └── delta_observer.py           # Delta Observer architecture
├── analysis/                        # Analysis scripts
│   ├── prepare_delta_dataset.py    # Extract activations and prepare dataset
│   ├── analyze_delta_latent.py     # Analyze Delta Observer latent space
│   └── geometric_analysis.py       # Geometric analysis of representations
├── data/                            # Datasets and activations
│   ├── monolithic_activations.npz
│   ├── compositional_activations.npz
│   ├── delta_observer_dataset.npz
│   └── delta_latent_umap.npz
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
@article{josserandaustin2026deltaobserver,
  title={Delta Observer: Learning Continuous Semantic Manifolds Between Neural Network Representations},
  author={Josserand-Austin, Aaron and EntroMorphic Research Team},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

We thank the EntroMorphic team and Manus platform for computational resources and feedback. We thank Claude (Anthropic) for providing detailed feedback on the Delta Observer methodology during the design phase.

---

## Contact

**Aaron (Tripp) Josserand-Austin**
EntroMorphic Research Team
📧 tripp@anjaustin.com
🔗 [entromorphic.com](https://entromorphic.com)

For questions about the paper or code, please open an issue or contact us directly.
