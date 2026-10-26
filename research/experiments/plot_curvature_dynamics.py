#!/usr/bin/env python3
"""Figure for research/reports/curvature_results.md.

Run from the repository root, in the paper environment:

    python -m research.experiments.plot_curvature_dynamics
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research/results/curvature_dynamics_v1.json"
OUTPUT = ROOT / "research/reports/figures/curvature_dynamics.png"

SURFACE, TEXT, TEXT_2, GRID, REFERENCE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
SERIES = {"mono": ("Monolithic (9,285 params)", "#2a78d6"),  # reference palette slots 1-2
          "comp": ("Compositional (1,480 params)", "#eb6834")}
ETA_COLORS = ("#2a78d6", "#1baf7a")  # slots 1 and 3


def main():
    report = json.loads(SOURCE.read_text())
    by = {}
    for run in report["hessians"]:
        by.setdefault(run["model"], []).append(run)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2,
                         "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.4))
    fig.patch.set_facecolor(SURFACE)

    for kind, (label, color) in SERIES.items():
        epochs = np.array([c["epoch"] for c in by[kind][0]["checkpoints"]])
        top = np.array([[c["lambda_max"] for c in r["checkpoints"]] for r in by[kind]])
        ratio = np.array([[-c["lambda_min"] / c["lambda_max"] for c in r["checkpoints"]] for r in by[kind]])
        for values, ax in ((top, ax1), (ratio, ax2)):
            for row in values:
                ax.plot(epochs, row, color=color, linewidth=1, alpha=0.35)
            ax.plot(epochs, np.median(values, 0), color=color, linewidth=2, marker="o", markersize=4,
                    label=label)
    ax1.set_ylabel("Top Hessian eigenvalue λ_max")
    ax1.set_title("Sharpness under Adam: no transient", loc="left", fontsize=11, fontweight="bold")
    ax2.set_yscale("log")
    ax2.set_ylabel("|λ_min| / λ_max")
    ax2.set_title("Negative curvature fades to 10⁻⁵–10⁻⁸ of λ_max", loc="left", fontsize=11, fontweight="bold")
    for ax in (ax1, ax2):
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(0, 200)
        ax.set_xticks([0, 2, 5, 10, 20, 40, 100, 200], ["0", "2", "5", "10", "20", "40", "100", "200"])
        ax.set_xlabel("Training epoch")
    ax1.legend(frameon=False, fontsize=9, loc="upper left")

    etas = sorted({e["eta"] for e in report["eos"]}, reverse=True)
    for eta, color in zip(etas, ETA_COLORS):
        runs = [e for e in report["eos"] if e["eta"] == eta]
        for run in runs:
            steps = [t["step"] for t in run["trace"]]
            ax3.plot(steps, [t["sharpness"] for t in run["trace"]], color=color, linewidth=1.5, alpha=0.8)
        two_over_eta = runs[0]["two_over_eta"]
        ax3.axhline(two_over_eta, color=REFERENCE, linewidth=1, linestyle="--", zorder=0)
        ax3.annotate(f"2/η = {two_over_eta:.2f}  (η = {eta:.3f})", (20000, two_over_eta), xytext=(-4, 6),
                     textcoords="offset points", ha="right", fontsize=9, color=TEXT)
    ax3.set_xlim(0, 20000)
    ax3.set_ylim(0, 13)
    ax3.set_xlabel("Full-batch gradient-descent step")
    ax3.set_ylabel("Top Hessian eigenvalue (Lanczos)")
    ax3.set_title("Gradient descent: edge of stability", loc="left", fontsize=11, fontweight="bold")

    for a in (ax1, ax2, ax3):
        a.set_facecolor(SURFACE)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.grid(axis="y", color=GRID, linewidth=0.8)
        a.set_axisbelow(True)

    fig.text(0.01, 0.01, "Left, middle: paper source models trained with Adam, seeds 0–4; thin lines = seeds, "
             "markers = median. Exact float64 Hessians. Right: monolithic model, full-batch GD, seeds 0–2 "
             "per step size.", fontsize=8, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, facecolor=SURFACE)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
