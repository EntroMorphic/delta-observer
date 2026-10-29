#!/usr/bin/env python3
"""Pipeline staleness: Basis Rotation vs Adam against the permutation bound (exploratory).

Pipeline model (PipeMare; Basis Rotation's analysis): stage k's gradient is
computed at the whole model's weights from tau_k = s (K - 1 - k) steps ago,
its own weights included, so the delay sits inside each stage's matrices.
Max delay D = 5 s. Conditions: fresh, stale (pipeline delay) and permuted
(fresh gradient + the pipeline error with each tensor's entries permuted).

    python -m research.experiments.pipeline_bound
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
from research.experiments.interface_confirm import Warmup, build
from research.experiments.interface_sweep import BATCH, Trainer
from research.optimizers.basis_rotation import BasisRotationAdam

ROOT = Path(__file__).resolve().parents[2]
SEEDS, MAX_DELAYS, MAX_EPOCHS = [400, 401, 402, 403, 404], [5, 10, 20, 40], 120
ARMS, LRS, OPTIMIZERS = ["B_cont_w4", "F_pipeline_w64"], [1 / 300, 1 / 100], ["adam", "basis_rotation"]


class PipelineTrainer(Trainer):
    """Stage k's gradient at the snapshot from tau_k steps ago (all stages)."""

    permute = False

    def stage_delays(self):
        n = self.net.n
        return [self.delay // (n - 1) * (n - 1 - k) for k in range(n)]

    def stale_gradients(self, x, y):
        live, grads = self.live(), []
        for k, tau in enumerate(self.stage_delays()):
            snapshot = self.view(tau) if tau else [{n: p.detach() for n, p in s.items()} for s in live]
            own = {n: p.detach().clone().requires_grad_(True) for n, p in snapshot[k].items()}
            views = [own if s == k else snapshot[s] for s in range(self.net.n)]
            grads += torch.autograd.grad(self.full_loss(views, x, y), list(own.values()))
        return grads

    def gradients(self, x, y):
        if self.delay == 0:
            return super().gradients(x, y)
        stale = self.stale_gradients(x, y)
        if not self.permute:
            return stale
        delay, self.delay = self.delay, 0
        fresh = super().gradients(x, y)
        self.delay = delay
        return [f + (s - f).reshape(-1)[p].reshape(f.shape) for f, s, p in zip(fresh, stale, self.perms)]


def make(optimizer, lr, arm, seed, condition, max_delay):
    base = build("adder", arm, seed, "stale", max_delay if condition != "fresh" else 0, "adam", lr)
    trainer = PipelineTrainer.__new__(PipelineTrainer)
    trainer.__dict__.update(base.__dict__)
    trainer.permute = condition == "permuted"
    generator = torch.Generator().manual_seed(12345)
    trainer.perms = [torch.randperm(p.numel(), generator=generator) for p in trainer.net.parameters()]
    if optimizer == "basis_rotation":
        trainer.opt = BasisRotationAdam(trainer.net, lr)
        trainer.sched = Warmup(trainer.opt.param_groups, trainer.steps_per_epoch)
    return trainer


def job(spec):
    optimizer, lr, arm, seed, condition, max_delay = spec
    torch.set_num_threads(1)
    started = time.time()
    trainer = make(optimizer, lr, arm, seed, condition, max_delay)
    fit, acc = None, 0.0
    for epoch in range(1, MAX_EPOCHS + 1):
        order = torch.randperm(len(trainer.inputs), generator=trainer.order)
        for i in range(0, len(order), BATCH):
            trainer.step(order[i:i + BATCH])
        acc = trainer.accuracy()
        if acc == 1.0:
            fit = epoch
            break
    return {"optimizer": optimizer, "lr": lr, "arm": arm, "seed": seed, "condition": condition,
            "max_delay": max_delay, "fit_epoch": fit, "final_accuracy": acc, "seconds": time.time() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    jobs = [(o, lr, a, s, c, d) for o in OPTIMIZERS for lr in LRS for a in ARMS for s in SEEDS
            for c in ("stale", "permuted") for d in MAX_DELAYS]
    jobs += [(o, lr, a, s, "fresh", 0) for o in OPTIMIZERS for lr in LRS for a in ARMS for s in SEEDS]
    jobs.sort(key=lambda j: -j[5])
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(job, jobs):
            rows.append(row)
            print(row, flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"status": "exploratory", "script_sha256": sha256(__file__), "git_commit": commit, "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "settings": {"seeds": SEEDS, "max_delays": MAX_DELAYS, "arms": ARMS, "lrs": LRS,
                           "optimizers": OPTIMIZERS, "max_epochs": MAX_EPOCHS,
                           "staleness": "stage k at whole-model snapshot tau_k = s (K-1-k)"}, "rows": rows}
    out = ROOT / "research/results/pipeline_bound_v0.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
