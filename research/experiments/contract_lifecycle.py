#!/usr/bin/env python3
"""Contract lifecycle (exploratory X1, journal/paradigm_synth.md).

Along a fresh adder-chain trajectory, every 16 steps for 16 epochs, per interface:
  coupling  mean |g . dm| over a probe set: the message drift over the last 64
            steps, weighted per example by the downstream loss gradient.
            Needs no knowledge of the true carry.
  drift     mean squared message drift over 64 steps / message variance.
  harm      log10 loss after one epoch with only this interface 64 steps stale,
            relative to a fresh branch with the same batch order.
  r2        true-carry R^2 (for reference only; probe capacity grows with width).

    python -m research.experiments.contract_lifecycle
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
from research.experiments.interface_sweep import BATCH, Trainer

ROOT = Path(__file__).resolve().parents[2]
DELAY, LAG, EVERY, EPOCHS = 64, 64, 16, 16
PROBE = 1024
SEEDS = list(range(200, 210))
ARMS = ["A_cont_w1", "B_cont_w4"]


class OneInterfaceTrainer(Trainer):
    """Fresh training, or, when `interface` is set, the two sides of that one
    interface see each other `delay` steps old (the v2 staleness model)."""

    interface = None

    def gradients(self, x, y):
        live, n = self.live(), self.net.n
        if self.interface is None:
            return torch.autograd.grad(self.full_loss(live, x, y), list(self.net.parameters()))
        old, grads = self.view(self.delay), []
        for side in (range(self.interface + 1), range(self.interface + 1, n)):
            views = [live[s] if s in side else old[s] for s in range(n)]
            grads += torch.autograd.grad(self.full_loss(views, x, y), [p for s in side for p in live[s].values()])
        return grads


def messages_and_sensitivity(trainer, x, y):
    """Message into each stage 1..n-1 and dL/dmessage (total derivative)."""
    net, live = trainer.net, trainer.live()
    message, loss, messages = net.initial_message(x), 0.0, []
    for s in range(net.n):
        if s > 0:
            messages.append(message)
        columns, out, message = net.stage(s, live[s], x, message)
        loss = loss + ((out - y[:, columns]) ** 2).sum()
    grads = torch.autograd.grad(loss, messages)
    return [m.detach() for m in messages], [g.detach() for g in grads]


def full_loss(trainer):
    with torch.no_grad():
        return float(trainer.full_loss(trainer.live(), trainer.inputs, trainer.outputs))


def branch_loss(trainer, interface, generator_seed):
    order, trainer.order = trainer.order, None
    branch = copy.deepcopy(trainer)
    trainer.order = order
    branch.interface = interface
    batches = torch.randperm(len(branch.inputs), generator=torch.Generator().manual_seed(generator_seed))
    for i in range(0, len(batches), BATCH):
        branch.step(batches[i:i + BATCH])
    return full_loss(branch)


def run_job(job):
    arm, seed = job
    torch.set_num_threads(1)
    started = time.time()
    trainer = OneInterfaceTrainer(arm, seed, DELAY)
    probe = torch.randperm(len(trainer.inputs), generator=torch.Generator().manual_seed(seed))[:PROBE]
    px, py = trainer.inputs[probe], trainer.outputs[probe]
    past, records, step = {}, [], 0
    for epoch in range(EPOCHS):
        order = torch.randperm(len(trainer.inputs), generator=trainer.order)
        for i in range(0, len(order), BATCH):
            trainer.step(order[i:i + BATCH])
            step += 1
            if step % EVERY:
                continue
            messages, sens = messages_and_sensitivity(trainer, px, py)
            past[step] = messages
            past.pop(step - LAG - EVERY, None)
            record = {"step": step, "epoch": step / trainer.steps_per_epoch, "accuracy": trainer.accuracy(),
                      "r2": trainer.interface_meaning()}
            if step - LAG in past:
                before = past[step - LAG]
                record["coupling"] = [float((g * (m - b)).sum(1).abs().mean()) for m, g, b in zip(messages, sens, before)]
                record["drift"] = [float(((m - b) ** 2).sum(1).mean() / m.var(0).sum().clamp_min(1e-12))
                                   for m, b in zip(messages, before)]
                if step >= DELAY:
                    generator_seed = 7919 * seed + step
                    fresh = branch_loss(trainer, None, generator_seed)
                    record["harm"] = [float(np.log10(branch_loss(trainer, i, generator_seed) / fresh))
                                      for i in range(trainer.net.n - 1)]
            records.append(record)
    return {"arm": arm, "seed": seed, "records": records, "seconds": time.time() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--arms", nargs="+", default=ARMS)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(run_job, [(a, s) for a in args.arms for s in args.seeds]):
            rows.append(row)
            print(f"{row['arm']} seed {row['seed']}: {row['seconds']:.0f}s", flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"status": "exploratory", "plan": "journal/paradigm_synth.md (gitignored)",
              "script_sha256": sha256(__file__), "git_commit": commit, "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "settings": {"delay": DELAY, "lag": LAG, "every": EVERY, "epochs": EPOCHS, "probe": PROBE,
                           "seeds": args.seeds, "arms": args.arms}, "rows": rows}
    out = ROOT / "research/results/contract_lifecycle_v0.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
