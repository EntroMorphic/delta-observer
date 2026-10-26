# Running the Notebooks in Google Colab

Every notebook runs in a fresh Colab session with no local setup.

## Quick start

1. Click a notebook's **Open in Colab** badge in the [main README](../README.md#-run-in-google-colab).
2. Choose **Runtime → Run all**.
3. If Colab warns that the notebook was not authored by Google, choose **Run anyway**.

## What the notebooks do for you

- **Dependencies.** The first code cell installs anything Colab lacks
  (`umap-learn`, `scikit-learn`, `matplotlib`, `torch`). Packages already
  present are skipped.
- **Data.** Notebooks `00`, `02` and `03` read pre-computed results from
  `data/`. If they are not already inside a checkout, they clone this
  repository into the Colab session automatically.
- **Self-contained notebooks.** `01_training_models` and
  `99_full_reproduction` generate the 4-bit addition dataset and train
  their models in the notebook, so they do not need the cloned data.
- **Paths.** Notebooks look for data at `../data/`, `data/` and
  `delta-observer/data/`, so the same notebook works in Colab, in Jupyter
  launched from `notebooks/`, and in Jupyter launched from the repo root.

## GPU

A GPU is optional. The models are small (512 training examples) and train
on the CPU runtime. Notebooks that train (`01`, `02`, `99`) use CUDA
automatically when you choose **Runtime → Change runtime type → GPU**.

## Saving outputs

Colab sessions are temporary. Figures and files the notebooks write are
lost when the runtime disconnects. To keep them, download them from the
**Files** panel, or mount Google Drive first:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `FileNotFoundError` for a `.npz` file | Re-run the clone cell, or run `!git clone https://github.com/EntroMorphic/delta-observer.git` |
| UMAP import fails | Run `!pip install umap-learn`, then **Runtime → Restart session** |
| Results differ slightly from the paper | Training is stochastic (random initialization and data splits); small numeric differences between runs are expected |

## Running locally instead

```bash
git clone https://github.com/EntroMorphic/delta-observer.git
cd delta-observer
pip install -r requirements.txt jupyter
jupyter notebook notebooks/
```
