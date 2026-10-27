#!/usr/bin/env python3
"""Exploratory diagnostics of the Second Star Slip v1 failure (not pre-registered).

Reconstructs every v1 Part 2 onset state (the fresh trajectory is
deterministic) and measures the mechanism linking staleness to harm: the
gradient error staleness causes, its linearization through cross-module
curvature, and both after Adam's preconditioning. Run from the repository root,
in the paper environment:

    python -m research.experiments.second_star_slip_diagnostics
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh

from research.experiments.curvature_dynamics import exact_hessian, flat_loss, sha256
from research.experiments.curvature_structure import block_labels
from research.experiments.second_star_slip import (
    DELAYS, HARM_ONSETS, INTERFACES, MODULES, Trainer, batches, coupling, delay_matrix, mixed_forward, target_ks)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research/results/second_star_slip_v1.json"
OUTPUT = ROOT / "research/results/second_star_slip_diagnostics_v1.json"
WINDOW = 16


def flat(views):
    return torch.cat([p.reshape(-1) for v in views for p in v.values()])


def stale_gradient(trainer, ks):
    """Full-batch gradient each module would receive with interface delays ks."""
    delays, live = delay_matrix(ks), [{n: p.detach().clone().requires_grad_(True) for n, p in v.items()}
                                      for v in trainer.history[-1]]
    parts = []
    for m in range(4):
        views = [live[n] if n == m else trainer.view(delays[m][n])[n] for n in range(4)]
        loss = F.mse_loss(mixed_forward(trainer.model, views, trainer.inputs), trainer.outputs)
        parts += [g.reshape(-1) for g in torch.autograd.grad(loss, list(live[m].values()))]
    return torch.cat(parts).double()


def linearized_error(H, labels, trainer, ks):
    """−Σ_n H_mn (θ_n(t) − θ_n(t − delay(m, n))): first-order gradient error."""
    delays, now = delay_matrix(ks), flat(trainer.history[-1]).double().numpy()
    error = np.zeros_like(now)
    for m in range(4):
        rows = labels == m
        for n in range(4):
            if delays[m][n]:
                cols = labels == n
                moved = now[cols] - flat(trainer.view(delays[m][n])).double().numpy()[cols]
                error[rows] -= H[np.ix_(rows, cols)] @ moved
    return error


def adam_preconditioner(trainer):
    """Diagonal of Adam's current preconditioner 1 / (sqrt(v̂) + ε), flat order."""
    group = trainer.opt.param_groups[0]
    beta2, eps = group["betas"][1], group["eps"]
    parts = []
    for p in trainer.model.parameters():
        state = trainer.opt.state[p]
        v_hat = state["exp_avg_sq"] / (1 - beta2 ** int(state["step"]))
        parts.append((1 / (v_hat.sqrt() + eps)).reshape(-1))
    return torch.cat(parts).double().numpy()


def onset_rows(trainer, onset, seed):
    labels, _ = block_labels(trainer.model)
    loss_fn, w = flat_loss(trainer.model, trainer.inputs, trainer.outputs)
    H, _ = exact_hessian(loss_fn, w)
    H = H.numpy()
    g = torch.func.grad(loss_fn)(w).numpy()
    L0, lr = float(loss_fn(w)), trainer.sched.get_last_lr()[0]
    P = adam_preconditioner(trainer)
    root = np.sqrt(P)
    HP = root[:, None] * H * root[None, :]
    lam_p = float(eigsh(HP, k=1, which="LA", return_eigenvectors=False)[0])
    rows = []
    for j in INTERFACES:
        kappa_p = coupling(HP, labels, list(range(j + 1)), list(range(j + 1, 4)))
        for k in DELAYS:
            ks = target_ks(str(j), k)
            fresh = torch.from_numpy(g)
            e = (stale_gradient(trainer, ks) - fresh).numpy()
            e_lin = linearized_error(H, labels, trainer, ks)
            u = lr * P * e  # per-step update error under Adam's current scaling
            impact = (-WINDOW * float(g @ u) + 0.5 * WINDOW ** 2 * float(u @ H @ u)) / L0
            rows.append({
                "seed": seed, "onset": onset, "interface": j, "k": k, "L0": L0, "lr": lr,
                "grad_error_rel": float(np.linalg.norm(e) / np.linalg.norm(g)),
                "grad_error_abs": float(np.linalg.norm(e)),
                "linearized_error_abs": float(np.linalg.norm(e_lin)),
                "linearization_cosine": float(e @ e_lin / (np.linalg.norm(e) * np.linalg.norm(e_lin) + 1e-300)),
                "update_error": float(np.linalg.norm(u)),
                "update_error_rel": float(np.linalg.norm(u) / np.linalg.norm(lr * P * g)),
                "predicted_impact": impact,
                "kappa_precond": kappa_p,
                "precond_sharpness": lr * lam_p,
            })
    return rows


def window_harm(trainer, ks, order):
    """Harm of one stale window, with its own fresh reference, for a batch order."""
    losses = []
    for delays in ((0, 0, 0), ks):
        branch = copy.deepcopy(trainer)
        for i in range(0, len(order), 32):
            branch.step(order[i:i + 32], delays)
        losses.append(branch.full_loss())
    return float(np.log10(losses[1]) - np.log10(losses[0]))


def reliability_rows(trainer, onset, seed, repeats=2):
    """Re-measure harm with alternative batch orders (test-retest)."""
    generator = torch.Generator().manual_seed(1000 * seed + onset)
    orders = [torch.randperm(len(trainer.inputs), generator=generator) for _ in range(repeats)]
    return [{"seed": seed, "onset": onset, "interface": j, "k": k,
             "harm_repeats": [window_harm(trainer, target_ks(str(j), k), order) for order in orders]}
            for j in INTERFACES for k in DELAYS]


def main():
    source = json.loads(SOURCE.read_text())
    harm = {(u["seed"], u["onset"], u["interface"], u["k"]): u for u in source["part2_units"]}
    rows, retest = [], []
    for seed in source["settings"]["seeds"]:
        torch.set_num_threads(8)
        trainer = Trainer(seed)
        for epoch in range(1, max(HARM_ONSETS) + 2):
            epoch_batches = batches(trainer)
            if epoch - 1 in HARM_ONSETS:
                for row in onset_rows(trainer, epoch - 1, seed):
                    v1 = harm[(row["seed"], row["onset"], row["interface"], row["k"])]
                    rows.append({**row, **{key: v1[key] for key in ("harm", "stale_loss", "fresh_loss",
                                                                     "kappa_cut", "lambda_max", "message_drift")}})
                retest += reliability_rows(trainer, epoch - 1, seed)
                print(f"seed {seed} onset {epoch - 1}", flush=True)
            for idx in epoch_batches:
                trainer.step(idx)
    OUTPUT.write_text(json.dumps({"script_sha256": sha256(__file__), "source": str(SOURCE.relative_to(ROOT)),
                                  "modules": MODULES, "rows": rows, "retest": retest}, indent=1))
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
