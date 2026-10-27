#!/usr/bin/env python3
"""Cut the wire (exploratory X2, journal/paradigm_synth.md).

Training starts synchronous. A decoupled interface switches to delay 256 and
stays decoupled: stages see each other live within a run of coupled
interfaces, and DELAY steps old across any decoupled one.

Policies
    sync              never decouple
    global_E          decouple every interface at epoch E
    trigger_f_k       decouple interface i once its coupling signal (drift x
                      downstream sensitivity, contract_lifecycle.py) has stayed
                      below f x its running peak for k consecutive checks

Cost: synchronous interface-epochs spent before fit (fit = 100% exact).
Tune on seeds 200-202, report 203-209.

    python -m research.experiments.contract_switch
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

from research.experiments.contract_lifecycle import messages_and_sensitivity
from research.experiments.curvature_dynamics import sha256
from research.experiments.interface_sweep import BATCH, MAX_EPOCHS, Trainer

ROOT = Path(__file__).resolve().parents[2]
DELAY, EVERY, LAG, PROBE = 256, 16, 64, 1024
SEEDS, TUNE = list(range(200, 210)), [200, 201, 202]
ARMS = ["A_cont_w1", "B_cont_w4"]
GLOBAL_EPOCHS = [2, 4, 6, 8, 10, 12, 16]
TRIGGERS = [(f, k) for f in (0.1, 0.25) for k in (2, 4, 8)]
POLICIES = ["sync"] + [f"global_{e}" for e in GLOBAL_EPOCHS] + [f"trigger_{f}_{k}" for f, k in TRIGGERS]


def segments(n, decoupled):
    """Runs of stages joined by coupled interfaces (interface i joins i, i+1)."""
    groups, current = [], [0]
    for s in range(1, n):
        if s - 1 in decoupled:
            groups.append(current)
            current = []
        current.append(s)
    return groups + [current]


class SwitchTrainer(Trainer):
    decoupled = frozenset()

    def gradients(self, x, y):
        live, n = self.live(), self.net.n
        if not self.decoupled:
            return torch.autograd.grad(self.full_loss(live, x, y), list(self.net.parameters()))
        old, grads = self.view(self.delay), []
        for group in segments(n, self.decoupled):
            views = [live[s] if s in group else old[s] for s in range(n)]
            grads += torch.autograd.grad(self.full_loss(views, x, y), [p for s in group for p in live[s].values()])
        return grads


def run_job(job):
    arm, seed, policy = job
    torch.set_num_threads(1)
    started = time.time()
    trainer = SwitchTrainer(arm, seed, DELAY)
    n_interfaces = trainer.net.n - 1
    probe = torch.randperm(len(trainer.inputs), generator=torch.Generator().manual_seed(seed))[:PROBE]
    px, py = trainer.inputs[probe], trainer.outputs[probe]
    trigger = None
    if policy.startswith("trigger"):
        _, f, k = policy.split("_")
        trigger = (float(f), int(k))
    switch_epoch = int(policy.split("_")[1]) if policy.startswith("global") else None
    past, peak, below = {}, [0.0] * n_interfaces, [0] * n_interfaces
    decoupled_at, step, curve, fit_epoch, sync_cost = {}, 0, [], None, 0.0
    for epoch in range(1, MAX_EPOCHS + 1):
        if switch_epoch is not None and epoch == switch_epoch + 1:
            for i in range(n_interfaces):
                decoupled_at.setdefault(i, step / trainer.steps_per_epoch)
            trainer.decoupled = frozenset(range(n_interfaces))
        order = torch.randperm(len(trainer.inputs), generator=trainer.order)
        for i in range(0, len(order), BATCH):
            trainer.step(order[i:i + BATCH])
            step += 1
            sync_cost += (n_interfaces - len(trainer.decoupled)) / trainer.steps_per_epoch
            if trigger is None or step % EVERY or len(trainer.decoupled) == n_interfaces:
                continue
            messages, sens = messages_and_sensitivity(trainer, px, py)
            past[step] = messages
            past.pop(step - LAG - EVERY, None)
            if step - LAG not in past:
                continue
            for j, (m, g, b) in enumerate(zip(messages, sens, past[step - LAG])):
                if j in trainer.decoupled:
                    continue
                coupling = float((g * (m - b)).sum(1).abs().mean())
                peak[j] = max(peak[j], coupling)
                below[j] = below[j] + 1 if coupling < trigger[0] * peak[j] else 0
                if below[j] >= trigger[1]:
                    decoupled_at[j] = step / trainer.steps_per_epoch
                    trainer.decoupled = trainer.decoupled | {j}
        curve.append(trainer.accuracy())
        if curve[-1] == 1.0:
            fit_epoch = epoch
            break
    return {"arm": arm, "seed": seed, "policy": policy, "fit_epoch": fit_epoch, "final_accuracy": curve[-1],
            "sync_interface_epochs": sync_cost, "decoupled_at": {str(k): v for k, v in decoupled_at.items()},
            "accuracy_curve": curve, "seconds": time.time() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--arms", nargs="+", default=ARMS)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    jobs = [(a, s, p) for a in args.arms for s in args.seeds for p in POLICIES]
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(run_job, jobs):
            rows.append(row)
            print(f"{row['arm']} seed {row['seed']} {row['policy']:>15}: fit {row['fit_epoch']} "
                  f"sync cost {row['sync_interface_epochs']:.1f} ({row['seconds']:.0f}s)", flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"status": "exploratory", "plan": "journal/paradigm_synth.md (gitignored)",
              "script_sha256": sha256(__file__), "git_commit": commit, "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "settings": {"delay": DELAY, "every": EVERY, "lag": LAG, "probe": PROBE, "seeds": args.seeds,
                           "tune_seeds": TUNE, "arms": args.arms, "policies": POLICIES}, "rows": rows}
    out = ROOT / "research/results/contract_switch_v0.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
