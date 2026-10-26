#!/usr/bin/env python3
"""Figure for research/reports/scaffolding_ablation_results.md.

Run from the repository root, in the paper environment:

    python -m research.experiments.plot_scaffolding_ablation
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research/results/scaffolding_ablation_addendum_b.json"
OUTPUT = ROOT / "research/reports/figures/scaffolding_ablation.png"

SURFACE, TEXT, TEXT_2, GRID, REFERENCE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
SERIES = [  # reference palette slots 1-3 (validated all-pairs, light mode)
    ("P", "Paper observer (recon + carry + bit-position)", "#2a78d6"),
    ("P/obs-nobit", "Without bit-position head", "#eb6834"),
    ("P/obs-unsup", "Unsupervised (recon only)", "#1baf7a"),
]
ORDER = ["P", "P/obs-nobit", "P/obs-unsup", "N"]
LABELS = {"P": "Paper\nobserver", "P/obs-nobit": "No bit\nhead", "P/obs-unsup": "Unsuper-\nvised",
          "N": "Notebook\npipeline"}
LABEL_OFFSETS = {"P": 0, "P/obs-nobit": 0, "P/obs-unsup": 6}


def main():
    runs = json.loads(SOURCE.read_text())["runs"]
    by = {}
    for run in runs:
        by.setdefault(run["config_name"], []).append(run)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2,
                         "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    fig.patch.set_facecolor(SURFACE)

    epochs = np.arange(len(by["P"][0]["history"]["sil_train"]))
    for name, label, color in SERIES:
        sil = np.array([r["history"]["sil_train"] for r in by[name]])
        ax.fill_between(epochs, sil.min(0), sil.max(0), color=color, alpha=0.15, linewidth=0)
        ax.plot(epochs, sil.mean(0), color=color, linewidth=2, label=label)
        ax.annotate(label.split(" (")[0], (epochs[-1], sil.mean(0)[-1]), xytext=(6, LABEL_OFFSETS[name]),
                    textcoords="offset points", va="center", fontsize=9, color=TEXT)
    src = np.array([r["history"]["sil_src_pca16"] for r in by["P"]])
    ax.plot(epochs, src.mean(0), color=REFERENCE, linewidth=2, linestyle="--",
            label="Source activations, PCA-16 (no observer)")
    ax.annotate("Sources, no observer", (epochs[-1], src.mean(0)[-1]), xytext=(6, -8),
                textcoords="offset points", va="center", fontsize=9, color=TEXT_2)

    ax.axhline(0, color=GRID, linewidth=1, zorder=0)
    ax.set_xlim(0, epochs[-1])
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Carry-count silhouette")
    ax.set_title("Clustering appears only in a supervised observer's latent", loc="left",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    for i, name in enumerate(ORDER):
        values = [100 * r["summary"]["online_minus_pca"] for r in by[name]]
        ax2.scatter(np.full(len(values), i), values, s=36, color=TEXT_2, edgecolors=SURFACE,
                    linewidths=2, zorder=3)
        ax2.hlines(np.mean(values), i - 0.25, i + 0.25, color=TEXT, linewidth=2, zorder=4)
    ax2.axhline(0, color=GRID, linewidth=1, zorder=0)
    ax2.set_xticks(range(len(ORDER)), [LABELS[n] for n in ORDER], fontsize=9)
    ax2.set_ylabel("Online R² − PCA R² (points)")
    ax2.set_title("Online advantage over same-run PCA", loc="left", fontsize=11, fontweight="bold")

    for a in (ax, ax2):
        a.set_facecolor(SURFACE)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.grid(axis="y", color=GRID, linewidth=0.8)
        a.set_axisbelow(True)

    fig.text(0.01, 0.01, "Seeds 0–4. Left: paper configuration with three observer objectives; mean (line) and "
             "min–max (band) per epoch, train-mode snapshots as in the paper. Right: one dot per seed; bar = mean.",
             fontsize=8, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.04, 0.93, 1))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, facecolor=SURFACE)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
