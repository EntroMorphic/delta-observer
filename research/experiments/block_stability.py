#!/usr/bin/env python3
"""Post-hoc F1b: a stability model that delays only cross-stage curvature.

Our staleness model shows each stage its own parameters live and the other
stages d steps old, so only the cross-stage blocks of the Hessian are delayed.
In the top-k eigensubspace V of H (full batch), with A_in = V'H_bd V (the
block-diagonal part) and A_cross = V'H V - A_in:

    x_{t+1} = x_t + beta (x_t - x_{t-1}) - eta (A_in x_t + A_cross x_{t-d})

Predicted floor at a checkpoint = largest grid delay with spectral radius < 1;
a run's prediction is the minimum over epochs 2-16 of a fresh trajectory.
Mapping as in sharpness_floor.py (SGD: eta = lr; Muon: eta = |dtheta|/|g|,
beta = 0). Adam is not modelled. Seeds 500-502.

    python -m research.experiments.block_stability
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
from research.experiments.interface_sweep import BATCH
from research.experiments.sharpness_floor import ARM, GRID, MOMENTUM, make, measured_floor

ROOT = Path(__file__).resolve().parents[2]
SEEDS, K, SUBSPACE_ITERS, EPOCHS = [500, 501, 502], 20, 12, range(2, 17)
SETTINGS = [("sgd", 0.1), ("sgd", 0.3), ("sgd0", 1.0), ("muon", 0.01), ("muon", 0.03)]


def hvp_factory(trainer):
    params = list(trainer.net.parameters())
    loss = trainer.full_loss(trainer.live(), trainer.inputs, trainer.outputs)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat = torch.cat([g.reshape(-1) for g in grads])
    sizes = [p.numel() for p in params]
    # stage index of every coordinate
    owner = torch.cat([torch.full((p.numel(),), s) for s, stage in enumerate(trainer.net.stages)
                       for p in stage.parameters()])

    def hvp(v):
        return torch.cat([h.reshape(-1) for h in torch.autograd.grad(flat @ v, params, retain_graph=True)])

    def block_hvp(v):
        out = torch.zeros_like(v)
        for s in range(trainer.net.n):
            mask = owner == s
            out[mask] = hvp(v * mask)[mask]
        return out
    return hvp, block_hvp, sum(sizes)


def reduced_matrices(trainer):
    hvp, block_hvp, n = hvp_factory(trainer)
    q = torch.linalg.qr(torch.randn(n, K, generator=torch.Generator().manual_seed(0)))[0]
    for _ in range(SUBSPACE_ITERS):
        q = torch.linalg.qr(torch.stack([hvp(q[:, j]) for j in range(K)], 1))[0]
    hq = torch.stack([hvp(q[:, j]) for j in range(K)], 1)
    bq = torch.stack([block_hvp(q[:, j]) for j in range(K)], 1)
    a_full, a_in = q.T @ hq, q.T @ bq
    a_full, a_in = (a_full + a_full.T) / 2, (a_in + a_in.T) / 2
    return a_full.double().numpy(), a_in.double().numpy()


def radius(a_in, a_cross, eta, beta, d):
    k = a_in.shape[0]
    size = k * (d + 1)
    m = np.zeros((size, size))
    eye = np.eye(k)
    # state [x_t, x_{t-1}, ..., x_{t-d}]
    m[:k, :k] = (1 + beta) * eye - eta * a_in
    m[:k, k:2 * k] += -beta * eye
    m[:k, d * k:(d + 1) * k] += -eta * a_cross
    for i in range(1, d + 1):
        m[i * k:(i + 1) * k, (i - 1) * k:i * k] = eye
    return float(np.max(np.abs(np.linalg.eigvals(m))))


def floor_at(a_full, a_in, eta, beta):
    a_cross, ok = a_full - a_in, []
    for d in GRID:
        if radius(a_in, a_cross, eta, beta, d) < 1:
            ok.append(d)
        else:
            break
    return max(ok) if ok else 0


def job(spec):
    optimizer, lr, seed = spec
    torch.set_num_threads(1)
    trainer = make(optimizer, lr, seed)
    params, beta, records = list(trainer.net.parameters()), MOMENTUM[optimizer], []
    for epoch in range(1, max(EPOCHS) + 1):
        order = torch.randperm(len(trainer.inputs), generator=trainer.order)
        batches = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
        for idx in batches[:-1]:
            trainer.step(idx)
        before = [p.detach().clone() for p in params]
        trainer.step(batches[-1])
        if epoch not in EPOCHS:
            continue
        a_full, a_in = reduced_matrices(trainer)
        if optimizer == "muon":
            loss = trainer.full_loss(trainer.live(), trainer.inputs, trainer.outputs)
            g = torch.cat([x.reshape(-1) for x in torch.autograd.grad(loss, params)])
            moved = torch.cat([(p.detach() - b).reshape(-1) for p, b in zip(params, before)])
            eta = float(moved.norm() / g.norm().clamp_min(1e-30))
        else:
            eta = trainer.opt.param_groups[0]["lr"]
        lam = float(np.max(np.linalg.eigvalsh(a_full)))
        records.append({"epoch": epoch, "eta": eta, "lambda_max": lam,
                        "cross_share": float(np.linalg.norm(a_full - a_in) / np.linalg.norm(a_full)),
                        "floor_block": floor_at(a_full, a_in, eta, beta),
                        "floor_all_delayed": floor_at(a_full, np.zeros_like(a_in), eta, beta)})
    return {"optimizer": optimizer, "lr": lr, "seed": seed, "records": records,
            "predicted_floor": min(r["floor_block"] for r in records),
            "predicted_floor_all_delayed": min(r["floor_all_delayed"] for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    jobs = [(o, lr, s) for o, lr in SETTINGS for s in SEEDS]
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(job, jobs):
            rows.append(row)
            print(row["optimizer"], row["lr"], row["seed"], row["predicted_floor"], flush=True)
    floors = json.load(open(ROOT / "research/results/staleness_floor_surface_v0.json"))["rows"]
    floors += json.load(open(ROOT / "research/results/sharpness_floor_floors_v0.json"))["rows"]
    summary = []
    for o, lr in SETTINGS:
        preds = [r["predicted_floor"] for r in rows if r["optimizer"] == o and r["lr"] == lr]
        summary.append({"optimizer": o, "lr": lr, "predicted": preds, "predicted_median": float(np.median(preds)),
                        "measured": measured_floor(floors, o, lr)})
        print(summary[-1])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"status": "exploratory, post hoc", "script_sha256": sha256(__file__), "git_commit": commit,
              "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "settings": {"seeds": SEEDS, "arm": ARM, "k": K, "epochs": list(EPOCHS)}, "summary": summary,
              "rows": rows}
    out = ROOT / "research/results/block_stability_v0.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
