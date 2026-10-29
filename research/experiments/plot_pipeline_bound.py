#!/usr/bin/env python3
"""Headline figure for research/reports/pipeline_bound_v0.md.

    python -m research.experiments.plot_pipeline_bound
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DELAYS = [0, 5, 10, 20, 40]
# Okabe-Ito colours (colourblind-safe); identity is also carried by marker and label
SERIES = [("adam", "stale", "Adam, real staleness", "#D55E00", "o", "-"),
          ("basis_rotation", "stale", "Basis Rotation, real staleness", "#0072B2", "s", "-"),
          ("adam", "permuted", "Adam, permuted error (bound)", "#009E73", "^", "--"),
          ("basis_rotation", "permuted", "Basis Rotation, permuted error (bound)", "#56B4E9", "v", "--")]
ARMS = [("B_cont_w4", "Adder chain (width 4)"), ("F_pipeline_w64", "64-wide pipeline")]


def fits(rows, optimizer, arm, condition, delay):
    cond = "fresh" if delay == 0 else condition
    sel = [r for r in rows if r["optimizer"] == optimizer and r["arm"] == arm and abs(r["lr"] - 1 / 300) < 1e-12
           and r["condition"] == cond and r["max_delay"] == delay]
    return sum(r["fit_epoch"] is not None for r in sel)


def main():
    rows = json.load(open(ROOT / "research/results/pipeline_bound_v0.json"))["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, (arm, title) in zip(axes, ARMS):
        for optimizer, condition, label, colour, marker, style in SERIES:
            ys = [fits(rows, optimizer, arm, condition, d) for d in DELAYS]
            ax.plot(range(len(DELAYS)), ys, color=colour, marker=marker, linestyle=style, linewidth=2,
                    markersize=8, label=label)
        ax.set_xticks(range(len(DELAYS)), [str(d) for d in DELAYS])
        ax.set_xlabel("Maximum pipeline delay (optimizer steps)")
        ax.set_title(title)
        ax.set_ylim(-0.3, 5.3)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Runs that fit (of 5)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Pipeline staleness: almost all harm is directional (lr 1/300, seeds 400–404)", fontsize=11)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out = ROOT / "research/reports/figures/pipeline_bound.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
