#!/usr/bin/env python3
"""Mechanism probe for the transformer pipeline (protocol amendment, M1 and M2).

Trains stale Adam (D = 24, seed 700) and, at CHECKPOINTS, compares the real
pipeline error e = g_stale - g_fresh with its permutation twin on one fixed
training batch at the live weights: curvature e^T H e / |e|^2 and cosine
with the fresh gradient.

    python -m research.experiments.transformer_pipeline_probe
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import research.experiments.transformer_pipeline as tp
from research.experiments.curvature_dynamics import sha256

ROOT = Path(__file__).resolve().parents[2]
METHOD, DELAY, SEED, CHECKPOINTS = "stale", 24, 700, (250, 500, 1000, 2000)


def flat(ts):
    return torch.cat([t.reshape(-1) for t in ts])


def curvature(run, x, y, direction):
    params = [p.detach().clone().requires_grad_(True) for p in run.params]
    with sdpa_kernel(SDPBackend.MATH):  # the fused CPU kernel has no double backward
        grads = torch.autograd.grad(run.loss_at(params, x, y), params, create_graph=True)
        hv = torch.autograd.grad(flat(grads) @ direction, params)
    return float(direction @ flat(hv) / (direction @ direction))


def probe(run, x, y):
    stale = flat(run.gradients(x, y))
    fresh = flat(torch.autograd.grad(run.loss_at(run.params, x, y), run.params))
    e = stale - fresh
    pieces, offset = [], 0
    for p, perm in zip(run.params, run.perms):
        n = p.numel()
        pieces.append(e[offset:offset + n][perm])
        offset += n
    twin = torch.cat(pieces)
    cos = lambda a, b: float(a @ b / (a.norm() * b.norm()))  # noqa: E731
    return {"error_norm": float(e.norm()), "gradient_norm": float(fresh.norm()),
            "curvature_real": curvature(run, x, y, e), "curvature_twin": curvature(run, x, y, twin),
            "cos_real": cos(e, fresh), "cos_twin": cos(twin, fresh)}


def main():
    torch.set_num_threads(4)
    run = tp.PipelineRun(METHOD, DELAY, tp.LR, SEED)
    x, y = run.batch(run.train_ids, torch.Generator().manual_seed(4242))
    records = []
    while run.t < max(CHECKPOINTS):
        run.step()
        if run.t in CHECKPOINTS:
            rec = {"step": run.t, "val_loss": run.val_loss(), **probe(run, x, y)}
            rec["curvature_ratio"] = rec["curvature_real"] / rec["curvature_twin"]
            records.append(rec)
            print(rec, flush=True)
    m1 = all(r["curvature_ratio"] >= 3 for r in records)
    m2 = all(r["cos_real"] < 0 for r in records if r["step"] > 250)
    result = {"status": "exploratory", "script_sha256": sha256(__file__), "method": METHOD, "delay": DELAY,
              "seed": SEED, "records": records, "verdicts": {"M1": "held" if m1 else "killed",
                                                              "M2": "held" if m2 else "killed"}}
    print(result["verdicts"])
    out = ROOT / "research/results/transformer_pipeline_probe_v0.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
