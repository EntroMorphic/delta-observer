#!/usr/bin/env python3
"""Basis Rotation against the permutation bound (exploratory).

Fresh, stale and permuted-error training with Basis Rotation (Adam in a
Kronecker-factored Hessian eigenbasis) on the staleness_floor_surface_v0
seeds and arms, compared with plain Adam's rows there. Each optimizer's own
permuted condition is its bound.

    python -m research.experiments.basis_rotation_bound
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
from research.experiments.interface_confirm import H2_ARMS, Warmup, build
from research.experiments.interface_sweep import BATCH
from research.optimizers.basis_rotation import BasisRotationAdam

ROOT = Path(__file__).resolve().parents[2]
SEEDS, DELAYS, MAX_EPOCHS = [400, 401, 402, 403, 404], [4, 8, 16, 32, 64], 120
LRS = [1 / 300, 1 / 100]  # primary: Adam's exploration rate; secondary: Adam's second surface rate


def make(arm, seed, condition, delay, lr):
    trainer = build("adder", arm, seed, condition, delay, "adam", lr)
    trainer.opt = BasisRotationAdam(trainer.net, lr)
    trainer.sched = Warmup(trainer.opt.param_groups, trainer.steps_per_epoch)
    return trainer


def job(spec):
    lr, arm, seed, condition, delay = spec
    torch.set_num_threads(1)
    started = time.time()
    trainer = make(arm, seed, condition, delay, lr)
    fit, acc = None, 0.0
    for epoch in range(1, MAX_EPOCHS + 1):
        order = torch.randperm(len(trainer.inputs), generator=trainer.order)
        for i in range(0, len(order), BATCH):
            trainer.step(order[i:i + BATCH])
        acc = trainer.accuracy()
        if acc == 1.0:
            fit = epoch
            break
    return {"optimizer": "basis_rotation", "lr": lr, "arm": arm, "seed": seed, "condition": condition,
            "delay": delay, "fit_epoch": fit, "final_accuracy": acc, "seconds": time.time() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    jobs = [(lr, a, s, c, d) for lr in LRS for a in H2_ARMS for s in SEEDS for c in ("stale", "permuted")
            for d in DELAYS]
    jobs += [(lr, a, s, "fresh", 0) for lr in LRS for a in H2_ARMS for s in SEEDS]
    jobs.sort(key=lambda j: -j[4])
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(job, jobs):
            rows.append(row)
            print(row, flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"status": "exploratory", "script_sha256": sha256(__file__), "git_commit": commit, "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "settings": {"seeds": SEEDS, "delays": DELAYS, "lrs": LRS, "max_epochs": MAX_EPOCHS,
                           "comparison": "staleness_floor_surface_v0.json (adam rows)"}, "rows": rows}
    out = ROOT / "research/results/basis_rotation_bound_v0.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
