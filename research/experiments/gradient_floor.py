#!/usr/bin/env python3
"""Can a gradient-based signal predict the staleness floor? (exploratory)

Along fresh trajectories (seeds 500-502, the eight sharpness_floor settings,
adder chain width 4), at every epoch 2-16 and every grid delay d, the descent
ratio r(d) = g_stale(d) . g / |g|^2 on a fixed 1024-example probe batch, where
g_stale(d) is the all-interface stale gradient (the floor surface's
staleness model) computed from the trajectory's own history.

Rule fixed before computing: predicted floor = largest grid delay at which
r(d) > 0 at every checkpoint (primary); threshold 0.5 as a secondary.

    python -m research.experiments.gradient_floor
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from research.experiments.curvature_dynamics import sha256
from research.experiments.interface_confirm import build
from research.experiments.interface_sweep import BATCH
from research.experiments.sharpness_floor import ARM, GRID, SETTINGS, measured_floor

ROOT = Path(__file__).resolve().parents[2]
SEEDS, EPOCHS, PROBE, THRESHOLDS = [500, 501, 502], range(2, 17), 1024, (0.0, 0.5)


def make(optimizer, lr, seed):
    trainer = build("adder", ARM, seed, "stale", max(GRID), "sgd" if optimizer == "sgd0" else optimizer, lr)
    if optimizer == "sgd0":
        for g in trainer.opt.param_groups:
            g["momentum"] = 0.0
    trainer.delay = 0  # train fresh; the history still holds max(GRID) snapshots
    return trainer


def descent_ratios(trainer, x, y):
    flat = lambda gs: torch.cat([g.reshape(-1) for g in gs])  # noqa: E731
    fresh = flat(trainer.gradients(x, y))
    out = {}
    for d in GRID:
        trainer.delay = d
        out[d] = float(flat(trainer.gradients(x, y)) @ fresh / (fresh @ fresh).clamp_min(1e-30))
    trainer.delay = 0
    return out


def job(spec):
    optimizer, lr, seed = spec
    torch.set_num_threads(1)
    trainer = make(optimizer, lr, seed)
    probe = torch.randperm(len(trainer.inputs), generator=torch.Generator().manual_seed(seed))[:PROBE]
    x, y = trainer.inputs[probe], trainer.outputs[probe]
    records = []
    for epoch in range(1, max(EPOCHS) + 1):
        order = torch.randperm(len(trainer.inputs), generator=trainer.order)
        for i in range(0, len(order), BATCH):
            trainer.step(order[i:i + BATCH])
        if epoch in EPOCHS:
            records.append({"epoch": epoch, "accuracy": trainer.accuracy(), "r": descent_ratios(trainer, x, y)})
    floors = {}
    for c in THRESHOLDS:
        ok = [d for d in GRID if all(r["r"][d] > c for r in records)]
        floors[str(c)] = max(ok) if ok else 0
    return {"optimizer": optimizer, "lr": lr, "seed": seed, "records": records, "predicted_floor": floors}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    jobs = [(o, lr, s) for o, lr in SETTINGS for s in SEEDS]
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(job, jobs):
            rows.append(row)
            print(row["optimizer"], row["lr"], row["seed"], row["predicted_floor"], flush=True)
    measured = json.load(open(ROOT / "research/results/staleness_floor_surface_v0.json"))["rows"]
    measured += json.load(open(ROOT / "research/results/sharpness_floor_floors_v0.json"))["rows"]
    summary = []
    for o, lr in SETTINGS:
        rs = [r for r in rows if r["optimizer"] == o and abs(r["lr"] - lr) < 1e-12]
        entry = {"optimizer": o, "lr": lr, "measured": measured_floor(measured, o, lr)}
        for c in THRESHOLDS:
            preds = [r["predicted_floor"][str(c)] for r in rs]
            med = float(np.median(preds))
            m = entry["measured"]
            entry[f"predicted_{c}"] = preds
            entry[f"within_2x_{c}"] = bool(m and med and 0.5 <= med / m <= 2) or (m == 0 and med == 0)
        summary.append(entry)
        print(entry)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"status": "exploratory", "script_sha256": sha256(__file__), "git_commit": commit, "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "settings": {"seeds": SEEDS, "arm": ARM, "epochs": list(EPOCHS), "probe": PROBE,
                           "thresholds": THRESHOLDS, "grid": GRID}, "summary": summary, "rows": rows}
    out = ROOT / "research/results/gradient_floor_v0.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
