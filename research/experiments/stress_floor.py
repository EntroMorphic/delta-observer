#!/usr/bin/env python3
"""Closed-loop stress test as a staleness-floor predictor (exploratory).

Rule fixed in advance. Train fresh (seeds 500-502, the eight sharpness_floor
settings, adder chain width 4) to the first epoch whose accuracy is >= 25%:
the start of the learning transition. From that checkpoint, for each grid
delay d, branch a stale run (all-interface staleness, d steps) and a fresh
run for STRESS_STEPS steps on identical batches. A delay passes if the stale
branch's final full-batch loss is at most RATIO x the fresh branch's.
Predicted floor = largest d up to which every delay passes. Primary RATIO 2;
secondary 1.25 and 4.

    python -m research.experiments.stress_floor
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from research.experiments.curvature_dynamics import sha256
from research.experiments.gradient_floor import make
from research.experiments.interface_sweep import BATCH
from research.experiments.sharpness_floor import GRID, SETTINGS, measured_floor

ROOT = Path(__file__).resolve().parents[2]
SEEDS, START_ACCURACY, STRESS_STEPS, MAX_START_EPOCH = [500, 501, 502], 0.25, 256, 60
RATIOS = (2.0, 1.25, 4.0)


def full_loss(trainer):
    with torch.no_grad():
        return float(trainer.full_loss(trainer.live(), trainer.inputs, trainer.outputs))


def branch_loss(trainer, delay, batches):
    order, trainer.order = trainer.order, None
    b = copy.deepcopy(trainer)
    trainer.order = order
    b.delay = delay
    for idx in batches:
        b.step(idx)
    return full_loss(b)


def job(spec):
    optimizer, lr, seed = spec
    torch.set_num_threads(1)
    started = time.time()
    trainer = make(optimizer, lr, seed)
    start_epoch = None
    for epoch in range(1, MAX_START_EPOCH + 1):
        order = torch.randperm(len(trainer.inputs), generator=trainer.order)
        for i in range(0, len(order), BATCH):
            trainer.step(order[i:i + BATCH])
        if trainer.accuracy() >= START_ACCURACY:
            start_epoch = epoch
            break
    row = {"optimizer": optimizer, "lr": lr, "seed": seed, "start_epoch": start_epoch}
    if start_epoch is None:
        row.update(predicted_floor={str(r): 0 for r in RATIOS}, loss_ratio=None)
        return row
    g = torch.Generator().manual_seed(seed * 7 + 1)
    perm = torch.cat([torch.randperm(len(trainer.inputs), generator=g) for _ in range(STRESS_STEPS * BATCH
                                                                                    // len(trainer.inputs) + 1)])
    batches = [perm[i * BATCH:(i + 1) * BATCH] for i in range(STRESS_STEPS)]
    fresh = branch_loss(trainer, 0, batches)
    ratios = {d: branch_loss(trainer, d, batches) / max(fresh, 1e-30) for d in GRID}
    floors = {}
    for r in RATIOS:
        f = 0
        for d in GRID:
            if ratios[d] <= r:
                f = d
            else:
                break
        floors[str(r)] = f
    row.update(fresh_loss=fresh, loss_ratio={str(d): v for d, v in ratios.items()}, predicted_floor=floors,
               seconds=time.time() - started)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    jobs = [(o, lr, s) for o, lr in SETTINGS for s in SEEDS]
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(job, jobs):
            rows.append(row)
            print(row["optimizer"], row["lr"], row["seed"], row["start_epoch"], row["predicted_floor"], flush=True)
    measured = json.load(open(ROOT / "research/results/staleness_floor_surface_v0.json"))["rows"]
    measured += json.load(open(ROOT / "research/results/sharpness_floor_floors_v0.json"))["rows"]
    summary = []
    for o, lr in SETTINGS:
        rs = [r for r in rows if r["optimizer"] == o and abs(r["lr"] - lr) < 1e-12]
        entry = {"optimizer": o, "lr": lr, "measured": measured_floor(measured, o, lr),
                 "start_epochs": [r["start_epoch"] for r in rs]}
        for ratio in RATIOS:
            preds = [r["predicted_floor"][str(ratio)] for r in rs]
            med, m = float(np.median(preds)), entry["measured"]
            entry[f"predicted_{ratio}"] = preds
            entry[f"within_2x_{ratio}"] = bool((m and med and 0.5 <= med / m <= 2) or (m == 0 and med == 0))
        summary.append(entry)
        print(entry)
    scores = {}
    for ratio in RATIOS:
        meas = [e["measured"] for e in summary]
        pred = [float(np.median(e[f"predicted_{ratio}"])) for e in summary]
        scores[str(ratio)] = {"within_2x": sum(e[f"within_2x_{ratio}"] for e in summary),
                              "spearman": float(spearmanr(pred, meas).correlation)}
    print(scores)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"status": "exploratory", "script_sha256": sha256(__file__), "git_commit": commit, "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "settings": {"seeds": SEEDS, "start_accuracy": START_ACCURACY, "stress_steps": STRESS_STEPS,
                           "ratios": RATIOS, "grid": GRID}, "summary": summary, "scores": scores, "rows": rows}
    out = ROOT / "research/results/stress_floor_v0.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
