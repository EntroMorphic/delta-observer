#!/usr/bin/env python3
"""Staleness floor and its limits (exploratory; seeds 400-404).

    surface    fit rate vs delay for stale and permuted-error training, per
               optimizer and learning rate. Floor = largest delay with >= 80%
               fits within FLOOR_EPOCHS.
    mechanism  along stale trajectories, how the real and the permuted
               staleness error align with the gradient and with curvature.

    python -m research.experiments.staleness_floor surface
    python -m research.experiments.staleness_floor mechanism
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
from research.experiments.interface_confirm import H2_ARMS, build
from research.experiments.interface_sweep import BATCH

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [400, 401, 402, 403, 404]
FLOOR_EPOCHS = 120
DELAYS = [1, 2, 4, 8, 16, 32, 64]
SETTINGS = [("adam", 1 / 300), ("adam", 1 / 100), ("sgd", 0.1), ("sgd", 0.3), ("muon", 0.01), ("muon", 0.03)]
MECHANISM = [("adam", 1 / 300, 16), ("sgd", 0.3, 2), ("muon", 0.03, 4)]
MECHANISM_EVERY, MECHANISM_EPOCHS = 32, 12


def surface_job(job):
    optimizer, lr, arm, seed, condition, delay = job
    torch.set_num_threads(1)
    started = time.time()
    trainer = build("adder", arm, seed, condition, delay, optimizer, lr)
    fit_epoch, accuracy = None, 0.0
    for epoch in range(1, FLOOR_EPOCHS + 1):
        order = torch.randperm(len(trainer.inputs), generator=trainer.order)
        for i in range(0, len(order), BATCH):
            trainer.step(order[i:i + BATCH])
        accuracy = trainer.accuracy()
        if accuracy == 1.0:
            fit_epoch = epoch
            break
    return {"optimizer": optimizer, "lr": lr, "arm": arm, "seed": seed, "condition": condition, "delay": delay,
            "fit_epoch": fit_epoch, "final_accuracy": accuracy, "seconds": time.time() - started}


def flat(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def mechanism_job(job):
    """Train with real staleness; at intervals compare the real error e with
    its permuted twin on full-batch quantities at the live parameters."""
    optimizer, lr, delay, arm, seed = job
    torch.set_num_threads(1)
    trainer = build("adder", arm, seed, "permuted", delay, optimizer, lr)  # carries the permutations
    perms, params = trainer.perms, list(trainer.net.parameters())
    records, step = [], 0
    for epoch in range(MECHANISM_EPOCHS):
        order = torch.randperm(len(trainer.inputs), generator=trainer.order)
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            x, y = trainer.inputs[idx], trainer.outputs[idx]
            stale = super(type(trainer), trainer).gradients(x, y)  # real staleness, no permutation
            if step % MECHANISM_EVERY == 0 and step >= delay:
                saved, trainer.delay = trainer.delay, 0
                fresh = super(type(trainer), trainer).gradients(x, y)
                trainer.delay = saved
                error = [s - f for s, f in zip(stale, fresh)]
                permuted = [e.reshape(-1)[p].reshape(e.shape) for e, p in zip(error, perms)]
                loss = trainer.full_loss(trainer.live(), trainer.inputs, trainer.outputs)
                grad = torch.autograd.grad(loss, params, create_graph=True)
                g = flat(grad).detach()
                row = {"step": step, "epoch": step / trainer.steps_per_epoch, "accuracy": trainer.accuracy()}
                for name, vec in (("real", error), ("permuted", permuted)):
                    v = flat(vec).detach()
                    hv = flat(torch.autograd.grad(flat(grad) @ v, params, retain_graph=True)).detach()
                    norm2 = float(v @ v)
                    row[name] = {"norm": norm2 ** 0.5,
                                 "cos_with_gradient": float(v @ g / (v.norm() * g.norm()).clamp_min(1e-30)),
                                 "curvature": float(v @ hv) / max(norm2, 1e-30)}
                records.append(row)
            for p, g_ in zip(params, stale):
                p.grad = g_
            trainer.opt.step(); trainer.sched.step()
            trainer.history.append(trainer.snapshot())
            step += 1
    return {"optimizer": optimizer, "lr": lr, "delay": delay, "arm": arm, "seed": seed, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["surface", "mechanism"])
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    if args.mode == "surface":
        jobs = [(o, lr, arm, s, c, d) for o, lr in SETTINGS for arm in H2_ARMS for s in SEEDS
                for c in ("stale", "permuted") for d in DELAYS]
        jobs += [(o, lr, arm, s, "fresh", 0) for o, lr in SETTINGS for arm in H2_ARMS for s in SEEDS]
        jobs.sort(key=lambda j: -j[5])
        fn = surface_job
    else:
        jobs = [(o, lr, d, arm, s) for o, lr, d in MECHANISM for arm in H2_ARMS for s in SEEDS[:3]]
        fn = mechanism_job
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(fn, jobs):
            rows.append(row)
            print({k: v for k, v in row.items() if k != "records"}, flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"status": "exploratory", "mode": args.mode, "script_sha256": sha256(__file__), "git_commit": commit,
              "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "settings": {"seeds": SEEDS, "floor_epochs": FLOOR_EPOCHS, "delays": DELAYS, "settings": SETTINGS,
                           "mechanism": MECHANISM}, "rows": rows}
    out = ROOT / f"research/results/staleness_floor_{args.mode}_v0.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
