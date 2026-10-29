#!/usr/bin/env python3
"""Score delay compensation against the permutation bound (exploratory).

Stage m computes its gradient with every other stage d steps old. Its error
is, to first order, H_{m,o} (theta_o_now - theta_o_old): purely cross-stage.
Compensations added to stage m's stale gradient g_m:

    dc_outer  lambda * g_m * (g_o . dtheta_o)       DC-ASGD with H ~ g g^T
              (the diagonal DC-ASGD form has no cross-stage terms, so it
              cannot correct this staleness at all)
    taylor    H_{m,o} dtheta_o by an exact Hessian-vector product: the best
              any first-order Taylor correction can do

Compared with plain stale training and the permuted error (the bound), on the
staleness_floor_surface_v0 seeds and arms.

    python -m research.experiments.compensation_bound
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
from research.experiments.interface_sweep import BATCH, Trainer, run_from

ROOT = Path(__file__).resolve().parents[2]
SEEDS, DELAYS, MAX_EPOCHS = [400, 401, 402, 403, 404], [4, 8, 16, 32], 120
SETTINGS = [("adam", 1 / 300), ("sgd", 0.1)]
DC_LAMBDA = 1.0  # DC-ASGD's default variance-control constant is O(1); not tuned


class CompensatedTrainer(Trainer):
    method = None

    def gradients(self, x, y):
        if self.delay == 0 or self.method is None:
            return super().gradients(x, y)
        live, old, n = self.live(), self.view(self.delay), self.net.n
        grads = []
        for m in range(n):
            own = list(live[m].values())
            others = [p for s in range(n) if s != m for p in old[s].values()]
            moved = [live[s][k].detach() - old[s][k] for s in range(n) if s != m for k in old[s]]
            if self.method == "taylor":
                others_req = [p.detach().requires_grad_(True) for p in others]
                it = iter(others_req)
                views = [live[s] if s == m else {k: next(it) for k in old[s]} for s in range(n)]
                loss = run_from(self.net, views, x, y, 0, self.net.initial_message(x))
                g_own = torch.autograd.grad(loss, own + others_req, create_graph=True)
                g_m, g_o = g_own[:len(own)], g_own[len(own):]
                coupling = sum((go * dm).sum() for go, dm in zip(g_o, moved))
                correction = torch.autograd.grad(coupling, own, allow_unused=True)
                grads += [g.detach() + (c if c is not None else 0) for g, c in zip(g_m, correction)]
            else:  # dc_outer
                others_req = [p.detach().requires_grad_(True) for p in others]
                it = iter(others_req)
                views = [live[s] if s == m else {k: next(it) for k in old[s]} for s in range(n)]
                loss = run_from(self.net, views, x, y, 0, self.net.initial_message(x))
                g_all = torch.autograd.grad(loss, own + others_req)
                g_m, g_o = g_all[:len(own)], g_all[len(own):]
                scale = DC_LAMBDA * float(sum((go * dm).sum() for go, dm in zip(g_o, moved)))
                grads += [g + scale * g for g in g_m]
        return grads


def make(optimizer, lr, arm, seed, condition, delay):
    if condition in ("stale", "permuted", "fresh"):
        return build("adder", arm, seed, condition, delay, optimizer, lr)
    base = build("adder", arm, seed, "stale", delay, optimizer, lr)
    trainer = CompensatedTrainer.__new__(CompensatedTrainer)
    trainer.__dict__.update(base.__dict__)
    trainer.method = condition
    return trainer


def job(spec):
    optimizer, lr, arm, seed, condition, delay = spec
    torch.set_num_threads(1)
    started = time.time()
    trainer = make(optimizer, lr, arm, seed, condition, delay)
    fit, acc = None, 0.0
    for epoch in range(1, MAX_EPOCHS + 1):
        order = torch.randperm(len(trainer.inputs), generator=trainer.order)
        for i in range(0, len(order), BATCH):
            trainer.step(order[i:i + BATCH])
        acc = trainer.accuracy()
        if not np.isfinite(acc):
            break
        if acc == 1.0:
            fit = epoch
            break
    return {"optimizer": optimizer, "lr": lr, "arm": arm, "seed": seed, "condition": condition, "delay": delay,
            "fit_epoch": fit, "final_accuracy": acc, "seconds": time.time() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    jobs = [(o, lr, a, s, c, d) for o, lr in SETTINGS for a in H2_ARMS for s in SEEDS
            for c in ("dc_outer", "taylor") for d in DELAYS]
    jobs.sort(key=lambda j: (j[4] != "taylor", -j[5]))
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(job, jobs):
            rows.append(row)
            print(row, flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"status": "exploratory", "script_sha256": sha256(__file__), "git_commit": commit, "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "settings": {"seeds": SEEDS, "delays": DELAYS, "settings": SETTINGS, "dc_lambda": DC_LAMBDA,
                           "max_epochs": MAX_EPOCHS, "comparison": "staleness_floor_surface_v0.json"},
              "rows": rows}
    out = ROOT / "research/results/compensation_bound_v0.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
