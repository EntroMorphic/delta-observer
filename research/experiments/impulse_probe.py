#!/usr/bin/env python3
"""Impulse response of one staleness error (exploratory; seeds 400-402).

Along a fresh trajectory, at a few points after the learning transition,
branch three copies: no injection, the real delay-d staleness error added to
one step, and that error with each tensor's entries permuted. All continue
fresh on identical batches; record the full-batch loss gap to the
no-injection branch over the next 64 steps.

    python -m research.experiments.impulse_probe
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

from research.experiments.curvature_dynamics import sha256
from research.experiments.hybrid_staleness import build_hybrid
from research.experiments.interface_confirm import build
from research.experiments.interface_sweep import BATCH, Trainer

ROOT = Path(__file__).resolve().parents[2]
SEEDS, ARM = [400, 401, 402], "B_cont_w4"
SETTINGS = [("adam", 1 / 300, 16), ("sgd", 0.3, 2), ("muon", 0.03, 4), ("muon", 0.01, 8), ("muon_pre", 0.03, 4)]
INJECT_EPOCHS, HORIZON = [5, 6, 7, 8], 64
CHECKPOINTS = [1, 2, 4, 8, 16, 32, 64]


def make(optimizer, lr, delay, seed):
    if optimizer == "muon_pre":
        return build_hybrid(ARM, seed, "stale", delay, optimizer, lr)
    return build("adder", ARM, seed, "stale", delay, optimizer, lr)


def full_loss(trainer):
    with torch.no_grad():
        return float(trainer.full_loss(trainer.live(), trainer.inputs, trainer.outputs))


def branch(trainer, first_grads, batches):
    order, trainer.order = trainer.order, None
    b = copy.deepcopy(trainer)
    trainer.order = order
    losses = {}
    for k, idx in enumerate(batches, start=1):
        grads = first_grads if k == 1 else Trainer.gradients(b, b.inputs[idx], b.outputs[idx])
        for p, g in zip(b.net.parameters(), grads):
            p.grad = g
        b.opt.step(); b.sched.step()
        b.history.append(b.snapshot())
        if k in CHECKPOINTS:
            losses[k] = full_loss(b)
    return losses


def job(spec):
    optimizer, lr, delay, seed = spec
    torch.set_num_threads(1)
    trainer = make(optimizer, lr, delay, seed)
    stale_delay, trainer.delay = trainer.delay, 0  # train fresh; history still accumulates
    generator = torch.Generator().manual_seed(9876)
    perms = [torch.randperm(p.numel(), generator=generator) for p in trainer.net.parameters()]
    records = []
    for epoch in range(1, max(INJECT_EPOCHS) + 1):
        order = torch.randperm(len(trainer.inputs), generator=trainer.order)
        batches = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
        if epoch in INJECT_EPOCHS:
            follow = torch.randperm(len(trainer.inputs), generator=torch.Generator().manual_seed(seed * 100 + epoch))
            follow = [follow[i % len(follow):i % len(follow) + BATCH] for i in range(0, HORIZON * BATCH, BATCH)]
            x, y = trainer.inputs[follow[0]], trainer.outputs[follow[0]]
            fresh = Trainer.gradients(trainer, x, y)
            trainer.delay = stale_delay
            stale = Trainer.gradients(trainer, x, y)
            trainer.delay = 0
            error = [s - f for s, f in zip(stale, fresh)]
            permuted = [e.reshape(-1)[p].reshape(e.shape) for e, p in zip(error, perms)]
            base = branch(trainer, fresh, follow)
            real = branch(trainer, [f + e for f, e in zip(fresh, error)], follow)
            perm = branch(trainer, [f + e for f, e in zip(fresh, permuted)], follow)
            records.append({"epoch": epoch, "loss": full_loss(trainer), "base": base,
                            "real_gap": {k: real[k] - base[k] for k in CHECKPOINTS},
                            "permuted_gap": {k: perm[k] - base[k] for k in CHECKPOINTS}})
        for idx in batches:
            trainer.step(idx)
    return {"optimizer": optimizer, "lr": lr, "delay": stale_delay, "seed": seed, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    jobs = [(o, lr, d, s) for o, lr, d in SETTINGS for s in SEEDS]
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(job, jobs):
            rows.append(row)
            print(row["optimizer"], row["lr"], row["seed"], flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"status": "exploratory", "script_sha256": sha256(__file__), "git_commit": commit, "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "settings": {"seeds": SEEDS, "arm": ARM, "settings": SETTINGS, "inject_epochs": INJECT_EPOCHS,
                           "horizon": HORIZON}, "rows": rows}
    out = ROOT / "research/results/impulse_probe_v0.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
