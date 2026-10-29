#!/usr/bin/env python3
"""Pipeline staleness on a standard model: a character GPT on Tiny Shakespeare.

Protocol fixed before any run: research/reports/transformer_pipeline_protocol.md.

Model: 4 pre-LN transformer blocks, width 128, 4 heads, context 64 (0.82M
parameters), split into 4 pipeline stages (embeddings + block 0, block 1,
block 2, block 3 + final norm + head). Staleness as in pipeline_bound.py
(PipeDream with weight stashing and vertical sync): stage k's gradient is
taken at the whole model's weights from tau_k = s (K - 1 - k) steps ago,
s = D / 3. D = 6 is 4-stage asynchronous 1F1B.

Arms:
    fresh            synchronous Adam (GPipe-equivalent)
    stale            pipeline-stale Adam
    permuted         fresh gradient + the pipeline error with each tensor's
                     entries permuted (the permutation bound)
    pipemare_lr      stale; stage k's learning rate divided by tau_k ** p,
                     p annealed 1 -> 0 over the first quarter (PipeMare T1)
    weight_predict   stale; the gradient is taken at the snapshot pushed
                     forward by tau_k times that step's update (SpecTrain)
    basis_rotation   stale; Basis Rotation for matrices, Adam otherwise
                     (basis_rotation_fresh: its synchronous reference)

    python -m research.experiments.transformer_pipeline [--workers 9]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
import urllib.request
from collections import deque
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call

from research.experiments.curvature_dynamics import sha256
from research.optimizers.basis_rotation import BasisRotationMatrices

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research/data/tiny_shakespeare.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
WIDTH, HEADS, LAYERS, CONTEXT, BATCH, STAGES = 128, 4, 4, 64, 32, 4
STEPS, WARMUP, EVAL_EVERY, EVAL_BATCHES, CLIP, BETAS = 2000, 100, 250, 64, 1.0, (0.9, 0.99)
SEEDS, DELAYS, LR, SECONDARY_LR = [700, 701, 702], [3, 6, 12, 24, 48], 1e-3, 3e-3
METHODS = ["stale", "permuted", "pipemare_lr", "weight_predict", "basis_rotation"]
HARM = 0.05  # nats/char: excess validation loss that counts as harm


def load_text():
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA)
    raw = DATA.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DATA_SHA256, "unexpected dataset contents"
    text = raw.decode()
    chars = sorted(set(text))
    index = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([index[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(ids))
    return ids[:split], ids[split:], len(chars)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(WIDTH), nn.LayerNorm(WIDTH)
        self.attn = nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(WIDTH, 4 * WIDTH), nn.GELU(), nn.Linear(4 * WIDTH, WIDTH))

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, WIDTH), nn.Parameter(0.02 * torch.randn(CONTEXT, WIDTH))
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.ln, self.head = nn.LayerNorm(WIDTH), nn.Linear(WIDTH, vocab)
        self.register_buffer("mask", torch.triu(torch.ones(CONTEXT, CONTEXT, dtype=torch.bool), 1))

    def forward(self, x):
        h = self.tok(x) + self.pos[:x.shape[1]]
        for b in self.blocks:
            h = b(h, self.mask[:x.shape[1], :x.shape[1]])
        return self.head(self.ln(h))

    def stage_of(self, name):
        if name.startswith("blocks."):
            return min(int(name.split(".")[1]), STAGES - 1)
        return 0 if name in ("pos",) or name.startswith("tok.") else STAGES - 1


class PipelineRun:
    def __init__(self, method, delay, lr, seed):
        torch.manual_seed(seed)
        self.train_ids, self.val_ids, vocab = load_text()
        self.net, self.method, self.lr = GPT(vocab), method, lr
        self.names = [n for n, _ in self.net.named_parameters()]
        self.params = [p for _, p in self.net.named_parameters()]
        self.stage = [self.net.stage_of(n) for n in self.names]
        self.delay = 0 if method in ("fresh", "basis_rotation_fresh") else delay
        self.tau = [self.delay // (STAGES - 1) * (STAGES - 1 - k) for k in range(STAGES)]
        self.batches = torch.Generator().manual_seed(seed)
        g = torch.Generator().manual_seed(12345)
        self.perms = [torch.randperm(p.numel(), generator=g) for p in self.params]
        self.history = deque(maxlen=max(self.tau) + 2)  # (weights, the update that followed them)
        if method.startswith("basis_rotation"):
            mats = [p for p in self.params if p.ndim == 2]
            self.opts = [BasisRotationMatrices(mats, lr=lr, betas=BETAS),
                         torch.optim.Adam([p for p in self.params if p.ndim != 2], lr=lr, betas=BETAS)]
            self.groups = [(g, None) for o in self.opts for g in o.param_groups]
        else:
            by_stage = [[p for p, s in zip(self.params, self.stage) if s == k] for k in range(STAGES)]
            self.opts = [torch.optim.Adam([{"params": ps, "stage": k} for k, ps in enumerate(by_stage)], lr=lr,
                                          betas=BETAS)]
            self.groups = [(g, g["stage"]) for g in self.opts[0].param_groups]
        self.t = 0

    def batch(self, ids, generator):
        starts = torch.randint(0, len(ids) - CONTEXT - 1, (BATCH,), generator=generator)
        x = torch.stack([ids[s:s + CONTEXT] for s in starts])
        return x, torch.stack([ids[s + 1:s + CONTEXT + 1] for s in starts])

    def loss_at(self, weights, x, y):
        logits = functional_call(self.net, dict(zip(self.names, weights)), (x,))
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

    def snapshot(self, tau):
        """Weights from tau steps ago (oldest available early on), and the update that followed them."""
        if tau == 0 or not self.history:
            return [p.detach() for p in self.params], None
        weights, update = self.history[max(0, len(self.history) - tau)]
        return weights, update

    def stage_gradients(self, k, tau, x, y, predict=False):
        weights, update = self.snapshot(tau)
        if predict and update is not None:
            weights = [w + tau * u for w, u in zip(weights, update)]
        own = [w.clone().requires_grad_(True) if s == k else w for w, s in zip(weights, self.stage)]
        grads = torch.autograd.grad(self.loss_at(own, x, y), [w for w, s in zip(own, self.stage) if s == k])
        return grads

    def gradients(self, x, y):
        if self.delay == 0:
            return list(torch.autograd.grad(self.loss_at(self.params, x, y), self.params))
        out = [None] * len(self.params)
        for k, tau in enumerate(self.tau):
            idx = [i for i, s in enumerate(self.stage) if s == k]
            for i, g in zip(idx, self.stage_gradients(k, tau, x, y, self.method == "weight_predict")):
                out[i] = g
        if self.method != "permuted":
            return out
        fresh = torch.autograd.grad(self.loss_at(self.params, x, y), self.params)
        return [f + (s - f).reshape(-1)[p].reshape(f.shape) for f, s, p in zip(fresh, out, self.perms)]

    def set_lr(self):
        t = self.t
        base = self.lr * min(1.0, (t + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / STEPS)))
        power = max(0.0, 1 - t / (STEPS / 4))
        for g, k in self.groups:
            scale = max(self.tau[k], 1) ** -power if self.method == "pipemare_lr" and k is not None else 1.0
            g["lr"] = base * scale

    def step(self):
        x, y = self.batch(self.train_ids, self.batches)
        grads = self.gradients(x, y)
        norm = torch.sqrt(sum((g * g).sum() for g in grads))
        scale = min(1.0, CLIP / (float(norm) + 1e-12))
        before = [p.detach().clone() for p in self.params]
        for p, g in zip(self.params, grads):
            p.grad = g * scale
        self.set_lr()
        for o in self.opts:
            o.step()
        self.history.append((before, [p.detach() - b for p, b in zip(self.params, before)]))
        self.t += 1
        return float(norm)

    @torch.no_grad()
    def val_loss(self):
        g = torch.Generator().manual_seed(0)
        losses = [float(self.loss_at(self.params, *self.batch(self.val_ids, g))) for _ in range(EVAL_BATCHES)]
        return float(np.mean(losses))


def job(spec):
    method, delay, lr, seed = spec
    torch.set_num_threads(1)
    started = time.time()
    run = PipelineRun(method, delay, lr, seed)
    curve = []
    for _ in range(STEPS):
        run.step()
        if run.t % EVAL_EVERY == 0:
            curve.append(run.val_loss())
            if not np.isfinite(curve[-1]):
                break
    return {"method": method, "delay": delay if run.delay else 0, "lr": lr, "seed": seed, "val_curve": curve,
            "final_val_loss": curve[-1], "seconds": time.time() - started}


def jobs():
    out = [(m, d, LR, s) for m in METHODS for d in DELAYS for s in SEEDS]
    out += [(m, 0, LR, s) for m in ("fresh", "basis_rotation_fresh") for s in SEEDS]
    out += [(m, d, SECONDARY_LR, s) for m in ("stale", "permuted") for d in DELAYS for s in SEEDS]
    out += [("fresh", 0, SECONDARY_LR, s) for s in SEEDS]
    return sorted(out, key=lambda j: (j[0] == "fresh", -j[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    load_text()
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(job, jobs()):
            rows.append(row)
            print({k: v for k, v in row.items() if k != "val_curve"}, flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"status": "exploratory", "script_sha256": sha256(__file__), "git_commit": commit, "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "settings": {"seeds": SEEDS, "delays": DELAYS, "lr": LR, "secondary_lr": SECONDARY_LR, "steps": STEPS,
                           "width": WIDTH, "layers": LAYERS, "context": CONTEXT, "batch": BATCH, "stages": STAGES,
                           "data_sha256": DATA_SHA256}, "rows": rows}
    out = ROOT / "research/results/transformer_pipeline_v0.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
