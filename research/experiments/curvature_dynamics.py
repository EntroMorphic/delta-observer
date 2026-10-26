#!/usr/bin/env python3
"""Exact-Hessian curvature dynamics of the paper's source models.

Part B of research/reports/curvature_protocol.md. Run from the repository root,
in the paper environment (root requirements.txt):

    python -m research.experiments.curvature_dynamics
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.func import functional_call, grad, vmap

from models.delta_observer import ALPHA, generate_dataset, get_scheduler
from research.experiments.scaffolding_ablation import PaperComp, PaperMono

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "research/reports/curvature_protocol.md"
CHECKPOINTS = (0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 60, 100, 150, 200)
TOP_K = (1, 5, 10, 20)
NEWTON_TOL = 1e-8


# =============================================================================
# Exact Hessian of a model's full-batch MSE loss
# =============================================================================

def flat_loss(model, inputs, targets):
    """Loss as a function of one flat float64 parameter vector."""
    model = copy.deepcopy(model).double()
    names = [n for n, _ in model.named_parameters()]
    shapes = [p.shape for _, p in model.named_parameters()]
    sizes = [p.numel() for _, p in model.named_parameters()]
    x, y = inputs.double(), targets.double()

    def loss(w):
        params = {n: p.reshape(s) for n, p, s in zip(names, torch.split(w, sizes), shapes)}
        return F.mse_loss(functional_call(model, params, (x,)), y)

    w = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    return loss, w


def hvp_fn(loss, w):
    """Exact Hessian-vector product, reverse-over-reverse (Pearlmutter)."""
    return lambda v: grad(lambda p: grad(loss)(p) @ v)(w)


def exact_hessian(loss, w, chunk=128):
    """Full Hessian from chunked, vectorized Hessian-vector products."""
    n = w.numel()
    hvp = hvp_fn(loss, w)
    eye = torch.eye(n, dtype=w.dtype)
    rows = [vmap(hvp)(eye[i:i + chunk]) for i in range(0, n, chunk)]
    H = torch.cat(rows)
    asymmetry = float((H - H.T).abs().max() / H.abs().max())
    return (H + H.T) / 2, asymmetry


def analyze(H, g, previous_top5=None):
    values, vectors = torch.linalg.eigh(H)
    order = torch.argsort(values, descending=True)
    values, vectors = values[order], vectors[:, order]
    coeff = vectors.T @ g
    g_sq = float(g @ g)
    neg = values < 0
    keep = values.abs() > NEWTON_TOL
    newton_dot = float(-(coeff[keep] ** 2 / values[keep]).sum())  # g·d for d = −H⁺g
    step_energy = coeff[keep] ** 2 / values[keep].abs()
    top5 = vectors[:, :5]
    return {
        "top_eigenvalues": values[:10].tolist(),
        "bottom_eigenvalues": values[-10:].tolist(),
        "lambda_max": float(values[0]), "lambda_min": float(values[-1]),
        "negative_count": int(neg.sum()),
        "negative_mass": float(values[neg].abs().sum() / values.abs().sum()),
        "trace": float(values.sum()),
        "grad_norm": g_sq ** 0.5,
        "grad_fraction_topk": {str(k): float((coeff[:k] ** 2).sum() / g_sq) for k in TOP_K},
        "newton_is_descent": newton_dot > 0,
        "newton_negative_curvature_share": float(step_energy[values[keep] < 0].sum() / step_energy.sum()),
        "top5_overlap_with_previous": (None if previous_top5 is None
                                       else float((previous_top5.T @ top5).pow(2).sum() / top5.shape[1])),
    }, top5


# =============================================================================
# Part B: training trajectories and checkpoint analysis
# =============================================================================

def train_sources(seed, epochs=200, batch=32):
    """models/delta_observer.py source training, without the observer."""
    torch.manual_seed(seed)
    inputs, outputs, _, _ = generate_dataset()
    n, n_batches = len(inputs), len(inputs) // batch
    models = {"mono": PaperMono(9), "comp": PaperComp()}
    opts = {k: torch.optim.Adam(m.parameters(), lr=ALPHA) for k, m in models.items()}
    scheds = {k: get_scheduler(o, n_batches * 10, n_batches * epochs, ALPHA, ALPHA / 1000) for k, o in opts.items()}
    states = {(k, 0): copy.deepcopy(m.state_dict()) for k, m in models.items()}
    accuracy = {}
    for epoch in range(1, epochs + 1):
        indices = torch.randperm(n)
        for i in range(0, n, batch):
            idx = indices[i:i + batch]
            for k, m in models.items():
                loss = F.mse_loss(m(inputs[idx]), outputs[idx])
                opts[k].zero_grad(); loss.backward(); opts[k].step(); scheds[k].step()
        if epoch in CHECKPOINTS:
            for k, m in models.items():
                states[(k, epoch)] = copy.deepcopy(m.state_dict())
                with torch.no_grad():
                    accuracy[(k, epoch)] = float((torch.round(m(inputs)) == outputs).all(1).float().mean())
    return states, accuracy


def hessian_job(job):
    seed, kind, states = job
    torch.set_num_threads(1)
    inputs, outputs, _, _ = generate_dataset()
    rows, previous = [], None
    for epoch in CHECKPOINTS:
        model = PaperMono(9) if kind == "mono" else PaperComp()
        model.load_state_dict(states[epoch])
        loss, w = flat_loss(model, inputs, outputs)
        start = time.perf_counter()
        H, asymmetry = exact_hessian(loss, w)
        metrics, previous = analyze(H, grad(loss)(w), previous)
        metrics.update({"epoch": epoch, "loss": float(loss(w)), "hessian_asymmetry": asymmetry,
                        "seconds": time.perf_counter() - start})
        rows.append(metrics)
        del H
    return {"seed": seed, "model": kind, "n_params": int(w.numel()), "checkpoints": rows}


# =============================================================================
# B-EoS: full-batch gradient descent and sharpness
# =============================================================================

def sharpness(loss, w):
    """Largest algebraic Hessian eigenvalue by Lanczos on exact HVPs."""
    hvp = hvp_fn(loss, w)
    n = w.numel()
    op = LinearOperator((n, n), matvec=lambda v: hvp(torch.from_numpy(np.asarray(v).ravel())).numpy(),
                        dtype=np.float64)
    return float(eigsh(op, k=1, which="LA", tol=1e-6, return_eigenvectors=False)[0])


def eos_job(job):
    seed, eta, steps, every = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    inputs, outputs, _, _ = generate_dataset()
    loss, w = flat_loss(PaperMono(9), inputs, outputs)
    w = w.clone()
    trace = []
    for step in range(steps + 1):
        if step % every == 0:
            trace.append({"step": step, "loss": float(loss(w)), "sharpness": sharpness(loss, w)})
        if step < steps:
            w = w - eta * grad(loss)(w)
            if not torch.isfinite(w).all():
                trace.append({"step": step + 1, "loss": float("nan"), "sharpness": float("nan")})
                break
    return {"seed": seed, "eta": eta, "two_over_eta": 2 / eta, "trace": trace}


def timing():
    """A2: cost of one HVP relative to one gradient, paper monolithic model."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    inputs, outputs, _, _ = generate_dataset()
    loss, w = flat_loss(PaperMono(9), inputs, outputs)
    v = torch.randn_like(w)
    p = w.clone().requires_grad_(True)

    def gradient():
        return torch.autograd.grad(loss(p), p)[0]

    def hvp():
        (g,) = torch.autograd.grad(loss(p), p, create_graph=True)
        return torch.autograd.grad(g @ v, p)[0]

    def median_time(fn, reps=200):
        times = []
        for _ in range(reps):
            start = time.perf_counter(); fn(); times.append(time.perf_counter() - start)
        return float(np.median(times))

    for fn in (gradient, hvp):
        median_time(fn, 20)  # warm-up
    t_grad, t_hvp = median_time(gradient), median_time(hvp)
    return {"gradient_seconds": t_grad, "hvp_seconds": t_hvp, "ratio": t_hvp / t_grad}


# =============================================================================

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--eos-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eos-steps", type=int, default=20000)
    parser.add_argument("--eos-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", default="research/results/curvature_dynamics_v1.json")
    args = parser.parse_args()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    print("A2 timing ...", flush=True)
    a2 = timing()
    print(f"  HVP / gradient = {a2['ratio']:.2f}", flush=True)

    torch.manual_seed(args.eos_seeds[0])
    inputs, outputs, _, _ = generate_dataset()
    lambda0 = sharpness(*flat_loss(PaperMono(9), inputs, outputs))
    etas = [2 / (4 * lambda0), 2 / (8 * lambda0)]
    print(f"lambda0 = {lambda0:.4f}; etas = {etas}", flush=True)

    jobs, accuracy = [], {}
    for seed in args.seeds:
        states, acc = train_sources(seed)
        accuracy[seed] = {f"{k}@{e}": v for (k, e), v in acc.items()}
        for kind in ("mono", "comp"):
            jobs.append((seed, kind, {e: states[(kind, e)] for e in CHECKPOINTS}))
    eos_jobs = [(s, eta, args.eos_steps, args.eos_every) for eta in etas for s in args.eos_seeds]

    with get_context("spawn").Pool(args.workers) as pool:
        eos_async = pool.map_async(eos_job, eos_jobs)
        hessians = []
        for result in pool.imap_unordered(hessian_job, jobs):
            hessians.append(result)
            last = result["checkpoints"][-1]
            print(f"[{len(hessians):2d}/{len(jobs)}] seed {result['seed']} {result['model']}: "
                  f"lambda_max@200 {last['lambda_max']:.3f}, neg {last['negative_count']}, "
                  f"{sum(c['seconds'] for c in result['checkpoints']):.0f}s", flush=True)
        eos = eos_async.get()
    hessians.sort(key=lambda r: (r["model"], r["seed"]))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {
        "protocol": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(__file__), "git_commit": commit,
        "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
        "settings": {"seeds": args.seeds, "checkpoints": CHECKPOINTS, "eos_seeds": args.eos_seeds,
                     "eos_steps": args.eos_steps, "eos_every": args.eos_every,
                     "lambda0": lambda0, "etas": etas},
        "a2_timing": a2, "source_accuracy": accuracy, "hessians": hessians, "eos": eos,
    }
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=1))
    print(f"Wrote {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
