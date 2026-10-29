#!/usr/bin/env python3
"""What each optimizer does with the staleness error (exploratory; seeds 400-402).

Along a stale trajectory, at intervals, copy the network and optimizer state
and take one step three ways: fresh gradient f, f + real error e, and
f + permuted error. For the update error delta = step(f + error) - step(f):
norm, curvature delta'H delta / |delta|^2, cosine with the gradient, and the
exact one-step loss change L(theta + step(f + error)) - L(theta + step(f)).

    python -m research.experiments.staleness_update_probe
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
from research.experiments.interface_confirm import build
from research.experiments.interface_sweep import BATCH
from research.experiments.staleness_floor import MECHANISM, flat

ROOT = Path(__file__).resolve().parents[2]
SEEDS, ARM, EVERY, EPOCHS = [400, 401, 402], "B_cont_w4", 32, 12


def step_delta(trainer, grads):
    """Parameter change one optimizer step would make with these gradients."""
    clone = copy.deepcopy({"net": trainer.net, "opt": trainer.opt})
    params = list(clone["net"].parameters())
    for p, g in zip(params, grads):
        p.grad = g.clone()
    clone["opt"].step()
    return [new.detach() - old.detach() for new, old in zip(params, trainer.net.parameters())]


def loss_after(trainer, delta):
    live = trainer.live()
    shifted = [{k: p.detach() + delta_k for (k, p), delta_k in zip(stage.items(), chunk)}
               for stage, chunk in zip(live, split_like(delta, live))]
    with torch.no_grad():
        return float(trainer.full_loss(shifted, trainer.inputs, trainer.outputs))


def split_like(flat_list, live):
    out, i = [], 0
    for stage in live:
        out.append(flat_list[i:i + len(stage)])
        i += len(stage)
    return out


def probe_job(job):
    optimizer, lr, delay, seed = job
    torch.set_num_threads(1)
    trainer = build("adder", ARM, seed, "permuted", delay, optimizer, lr)
    perms, params = trainer.perms, list(trainer.net.parameters())
    base = super(type(trainer), trainer)
    records, step = [], 0
    for epoch in range(EPOCHS):
        order = torch.randperm(len(trainer.inputs), generator=trainer.order)
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            x, y = trainer.inputs[idx], trainer.outputs[idx]
            stale = base.gradients(x, y)
            if step % EVERY == 0 and step >= delay:
                saved, trainer.delay = trainer.delay, 0
                fresh = base.gradients(x, y)
                trainer.delay = saved
                error = [s - f for s, f in zip(stale, fresh)]
                permuted = [e.reshape(-1)[p].reshape(e.shape) for e, p in zip(error, perms)]
                loss = trainer.full_loss(trainer.live(), trainer.inputs, trainer.outputs)
                grad = torch.autograd.grad(loss, params, create_graph=True)
                g = flat(grad).detach()
                fresh_step = step_delta(trainer, fresh)
                base_loss = loss_after(trainer, fresh_step)
                row = {"step": step, "epoch": step / trainer.steps_per_epoch, "loss": float(loss)}
                for name, err in (("real", error), ("permuted", permuted)):
                    noisy_step = step_delta(trainer, [f + e for f, e in zip(fresh, err)])
                    delta = [a - b for a, b in zip(noisy_step, fresh_step)]
                    d = flat(delta)
                    hd = flat(torch.autograd.grad(flat(grad) @ d, params, retain_graph=True)).detach()
                    n2 = float(d @ d)
                    row[name] = {"norm": n2 ** 0.5, "curvature": float(d @ hd) / max(n2, 1e-30),
                                 "cos_with_gradient": float(d @ g / (d.norm() * g.norm()).clamp_min(1e-30)),
                                 "loss_change": loss_after(trainer, noisy_step) - base_loss}
                records.append(row)
            for p, g_ in zip(params, stale):
                p.grad = g_
            trainer.opt.step(); trainer.sched.step()
            trainer.history.append(trainer.snapshot())
            step += 1
    return {"optimizer": optimizer, "lr": lr, "delay": delay, "arm": ARM, "seed": seed, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    jobs = [(o, lr, d, s) for o, lr, d in MECHANISM for s in SEEDS]
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(probe_job, jobs):
            rows.append(row)
            print(row["optimizer"], row["seed"], len(row["records"]), flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"status": "exploratory", "script_sha256": sha256(__file__), "git_commit": commit, "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "settings": {"seeds": SEEDS, "arm": ARM, "every": EVERY, "epochs": EPOCHS, "mechanism": MECHANISM},
              "rows": rows}
    out = ROOT / "research/results/staleness_update_probe_v0.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
