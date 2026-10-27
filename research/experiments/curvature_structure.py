#!/usr/bin/env python3
"""Block structure of the source models' exact Hessians (exploratory addendum).

Not pre-registered; see research/reports/curvature_results.md, addendum. Run from
the repository root, in the paper environment:

    python -m research.experiments.curvature_structure
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from torch.func import functional_call, jacrev

from models.delta_observer import generate_dataset
from research.experiments.curvature_dynamics import CHECKPOINTS, exact_hessian, flat_loss, sha256, train_sources
from research.experiments.scaffolding_ablation import PaperComp, PaperMono

ROOT = Path(__file__).resolve().parents[2]
SPLIT_EPOCHS = (0, 2, 20, 200)
TOP = 10


def block_labels(model):
    """Block index per flat parameter: top-level module name (adder or layer)."""
    labels = []
    for name, p in model.named_parameters():
        labels += [name.split(".")[0]] * p.numel()
    names = list(dict.fromkeys(labels))
    return np.array([names.index(n) for n in labels]), names


def within_mass(A, same):
    """Fraction of squared Frobenius norm inside the diagonal blocks."""
    return float((A[same] ** 2).sum() / (A ** 2).sum())


def gauss_newton(model, inputs, w):
    """2/N JᵀJ for the mean-squared-error loss over all N output entries."""
    model = copy.deepcopy(model).double()
    names = [n for n, _ in model.named_parameters()]
    shapes = [p.shape for _, p in model.named_parameters()]
    sizes = [p.numel() for _, p in model.named_parameters()]
    x = inputs.double()

    def outputs(v):
        params = {n: p.reshape(s) for n, p, s in zip(names, torch.split(v, sizes), shapes)}
        return functional_call(model, params, (x,)).reshape(-1)

    J = jacrev(outputs)(w)
    return 2 / J.shape[0] * J.T @ J


def residual_split(H, GN, same):
    """Decompose H = GN + R and summarize where each part's curvature lives."""
    R = H - GN
    values = np.linalg.eigvalsh(R)
    scale = np.abs(values).max()
    return {
        "residual_share": float(np.linalg.norm(R) / np.linalg.norm(H)),
        "within_gauss_newton": within_mass(GN, same),
        "within_residual": within_mass(R, same) if scale > 0 else None,
        "residual_trace_relative": float(R.trace() / np.abs(values).sum()),
        "residual_negative": int((values < -1e-10 * scale).sum()),
        "residual_positive": int((values > 1e-10 * scale).sum()),
    }


def job(args):
    seed, kind, states = args
    torch.set_num_threads(2)
    inputs, outputs, _, _ = generate_dataset()
    rows = []
    for epoch in CHECKPOINTS:
        model = PaperMono(9) if kind == "mono" else PaperComp()
        model.load_state_dict(states[epoch])
        labels, names = block_labels(model)
        same = labels[:, None] == labels[None, :]
        loss, w = flat_loss(model, inputs, outputs)
        H, _ = exact_hessian(loss, w)
        Hn = H.numpy()
        values, vectors = eigsh(Hn, k=TOP, which="LA")
        order = np.argsort(-values)
        values, vectors = values[order], vectors[:, order]
        shares = np.stack([np.bincount(labels, weights=vectors[:, j] ** 2, minlength=len(names))
                           for j in range(TOP)])
        block_top = max(float(np.linalg.eigvalsh(Hn[np.ix_(labels == b, labels == b)])[-1])
                        for b in range(len(names)))
        sizes = np.bincount(labels) / len(labels)
        row = {"epoch": epoch, "loss": float(loss(w)), "blocks": names,
               "within_mass": within_mass(Hn, same), "random_baseline": float((sizes ** 2).sum()),
               "lambda_max": float(values[0]), "block_lambda_max": block_top,
               "top_dominant_share": shares.max(1).tolist(), "top_dominant_block": shares.argmax(1).tolist()}
        if epoch in SPLIT_EPOCHS:
            row["split"] = residual_split(Hn, gauss_newton(model, inputs, w).numpy(), same)
        rows.append(row)
    return {"seed": seed, "model": kind, "checkpoints": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", default="research/results/curvature_structure_v1.json")
    args = parser.parse_args()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    jobs = []
    for seed in args.seeds:
        states, _ = train_sources(seed)
        for kind in ("comp", "mono"):
            jobs.append((seed, kind, {e: states[(kind, e)] for e in CHECKPOINTS}))
    with get_context("spawn").Pool(args.workers) as pool:
        runs = []
        for result in pool.imap_unordered(job, jobs):
            runs.append(result)
            last = result["checkpoints"][-1]
            print(f"seed {result['seed']} {result['model']}: within@200 {last['within_mass']:.3f}, "
                  f"block/full {last['block_lambda_max'] / last['lambda_max']:.3f}", flush=True)
    runs.sort(key=lambda r: (r["model"], r["seed"]))

    report = {"script_sha256": sha256(__file__), "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "settings": {"seeds": args.seeds, "checkpoints": CHECKPOINTS, "split_epochs": SPLIT_EPOCHS},
              "runs": runs}
    destination = ROOT / args.output
    destination.write_text(json.dumps(report, indent=1))
    print(f"Wrote {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
