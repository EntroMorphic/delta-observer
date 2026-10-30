#!/usr/bin/env python3
"""Vectorized engine for the transformer pipeline grid (protocol amendment 2).

Same model, initialization, batches, staleness and arms as
transformer_pipeline.py, but every run of a group and every pipeline stage's
snapshot is a slice of one leading "copy" dimension: one batched forward and
backward per step for the whole group, with Adam and Basis Rotation
vectorized across runs. The last stage has tau = 0, so its copy is the fresh
gradient (the permuted twin needs no extra pass).

    python -m research.experiments.transformer_pipeline_vec [--devices mps cpu]
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from multiprocessing import get_context
from pathlib import Path
from queue import Empty

import numpy as np
import torch
import torch.nn.functional as F

import research.experiments.transformer_pipeline as tp
from research.experiments.curvature_dynamics import sha256

ROOT = Path(__file__).resolve().parents[2]
STAGE_METHODS = ("stale", "permuted", "pipemare_lr", "weight_predict", "basis_rotation")
FRESH_METHODS = ("fresh", "basis_rotation_fresh")
EPS, BR_FREQ = 1e-8, 10


def grid_chunks():
    """The remaining cells, as chunks two device workers pull from a queue (largest first)."""
    chunks = {f"lr3e-3_D{d}": [(m, d, tp.SECONDARY_LR, x) for m in ("stale", "permuted") for x in tp.SEEDS]
              for d in (3, 6, 12)}
    for d in (3, 6, 12):
        chunks[f"lr1e-3_D{d}"] = [(m, d, tp.LR, x) for m in STAGE_METHODS for x in tp.SEEDS]
    chunks["lr3e-3_D24_s702"] = [("permuted", 24, tp.SECONDARY_LR, 702)]  # left unfinished by the stopped grid
    return chunks


def linear(h, w, b):
    return torch.baddbmm(b[:, None, :], h, w.transpose(1, 2))


def layer_norm(h, w, b):
    return F.layer_norm(h, (h.shape[-1],)) * w[:, None, :] + b[:, None, :]


def attention(q, k, v):
    """Causal attention; on MPS this beats the fused kernel at these shapes."""
    T = q.shape[-2]
    mask = torch.full((T, T), float("-inf"), device=q.device).triu(1)
    return torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]) + mask, -1) @ v


def block(h, W, p, E, B, T):
    C, H = tp.WIDTH, tp.HEADS
    a = layer_norm(h, W[p + "ln1.weight"], W[p + "ln1.bias"])
    q, k, v = linear(a, W[p + "attn.in_proj_weight"], W[p + "attn.in_proj_bias"]).split(C, dim=-1)
    q, k, v = (z.reshape(E * B, T, H, C // H).transpose(1, 2) for z in (q, k, v))
    o = attention(q, k, v).transpose(1, 2).reshape(E, B * T, C)
    h = h + linear(o, W[p + "attn.out_proj.weight"], W[p + "attn.out_proj.bias"])
    a = layer_norm(h, W[p + "ln2.weight"], W[p + "ln2.bias"])
    return h + linear(F.gelu(linear(a, W[p + "mlp.0.weight"], W[p + "mlp.0.bias"])), W[p + "mlp.2.weight"],
                      W[p + "mlp.2.bias"])


def forward(W, x, vocab, own=None):
    """Per-copy mean cross-entropy for weights W[name] = [E, ...] and tokens x = [E, B, T + 1].

    own: the stage whose weights the caller differentiates. Stages below it run
    without autograd (nothing there needs a gradient); None runs all with it.
    """
    E, B, T1 = x.shape
    T, C = T1 - 1, tp.WIDTH
    inp, tgt = x[..., :-1].reshape(E, B * T), x[..., 1:].reshape(E, B * T)
    last = tp.STAGES - 1
    h = None
    for s in range(tp.STAGES):
        with torch.set_grad_enabled(torch.is_grad_enabled() and (own is None or s >= own)):
            if s == 0:
                h = torch.bmm(F.one_hot(inp, vocab).to(W["tok.weight"].dtype), W["tok.weight"])
                h = (h.view(E, B, T, C) + W["pos"][:, None, :T]).view(E, B * T, C)
            for i in range(tp.LAYERS):
                if min(i, last) == s:
                    h = block(h, W, f"blocks.{i}.", E, B, T)
    logits = linear(layer_norm(h, W["ln.weight"], W["ln.bias"]), W["head.weight"], W["head.bias"])
    return F.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1), reduction="none").view(E, -1).mean(1)


class Group:
    def __init__(self, runs, device="cpu"):
        self.runs, self.device = runs, torch.device(device)
        self.train_ids, self.val_ids, self.vocab = tp.load_text()
        nets = []
        for _, _, _, seed in runs:
            torch.manual_seed(seed)
            nets.append(tp.GPT(self.vocab))
        self.names = [n for n, _ in nets[0].named_parameters()]
        self.stage = {n: nets[0].stage_of(n) for n in self.names}
        self.P = {n: torch.stack([dict(net.named_parameters())[n].detach() for net in nets]).to(self.device)
                  for n in self.names}
        g = torch.Generator().manual_seed(12345)
        self.perm = {n: torch.randperm(self.P[n][0].numel(), generator=g).to(self.device) for n in self.names}
        self.gens = [torch.Generator().manual_seed(seed) for *_, seed in runs]
        R = len(runs)
        self.tau = [[d // (tp.STAGES - 1) * (tp.STAGES - 1 - k) for k in range(tp.STAGES)] if m in STAGE_METHODS
                    else None for m, d, _, _ in runs]
        self.H = max([max(t) for t in self.tau if t] + [0]) + 1
        # Copy classes. Class k (0..3): one copy per staged run at tau_k, differentiated only in stage k's
        # weights. Class None: a tau = 0 copy differentiated in every weight, for fresh-type runs and for the
        # permuted twin, whose fresh gradient it supplies (and its stage-3 gradient, tau_3 = 0).
        last = tp.STAGES - 1
        self.classes = {None: [(r, 0) for r, (m, *_) in enumerate(runs) if self.tau[r] is None or m == "permuted"]}
        for k in range(tp.STAGES):
            self.classes[k] = [(r, taus[k]) for r, taus in enumerate(self.tau)
                               if taus is not None and not (k == last and runs[r][0] == "permuted")]
        self.classes = {k: v for k, v in self.classes.items() if v}
        self.own_index, self.fresh_index = {}, {}
        for n in self.names:
            s = self.stage[n]
            first = [r for r, _ in self.classes.get(s, [])]
            full = [r for r, _ in self.classes.get(None, [])]
            own = [first.index(r) if r in first else len(first) + full.index(r) for r in range(R)]
            fresh = [len(first) + full.index(r) if r in full else own[r] for r in range(R)]
            self.own_index[n] = torch.tensor(own, device=self.device)
            self.fresh_index[n] = torch.tensor(fresh, device=self.device)
        self.permuted = torch.tensor([m == "permuted" for m, *_ in runs], device=self.device)
        self.predict = [m == "weight_predict" for m, *_ in runs]
        self.br = torch.tensor([m.startswith("basis_rotation") for m, *_ in runs], device=self.device)
        self.lr = torch.tensor([lr for _, _, lr, _ in runs], device=self.device)
        self.pipemare = [m == "pipemare_lr" for m, *_ in runs]
        self.buffer = {n: torch.zeros((self.H,) + tuple(p.shape), device=self.device) for n, p in self.P.items()}
        self.m = {n: torch.zeros_like(p) for n, p in self.P.items()}
        self.v = {n: torch.zeros_like(p) for n, p in self.P.items()}
        self.br_runs = torch.nonzero(self.br).flatten()
        self.br_state = {}
        if len(self.br_runs):
            for n in self.names:
                if self.P[n].ndim == 3:
                    rb, (_, a, b) = len(self.br_runs), self.P[n].shape
                    z = lambda *s: torch.zeros(s)  # noqa: E731  (refresh maths runs on the CPU, as in the reference)
                    self.br_state[n] = {"m": torch.zeros(rb, a, b, device=self.device),
                                        "v": torch.zeros(rb, a, b, device=self.device),
                                        "L": z(rb, a, a), "R": z(rb, b, b), "U": z(rb, a, a), "V": z(rb, b, b)}
        self.t = 0

    def draw(self, ids, gens):
        out = []
        for g in gens:
            starts = torch.randint(0, len(ids) - tp.CONTEXT - 1, (tp.BATCH,), generator=g)
            out.append(torch.stack([ids[s:s + tp.CONTEXT + 1] for s in starts]))
        return torch.stack(out)

    def copy_weights(self, copies, leaves):
        t, H = self.t, self.H
        runs = torch.tensor([r for r, _ in copies], device=self.device)
        slots, nxt, scale = [], [], []
        for r, tau in copies:
            j = max(0, t - tau)
            go = self.predict[r] and tau > 0 and t >= 1
            slots.append(j % H)
            nxt.append((j + 1) % H if go else j % H)
            scale.append(float(tau) if go else 0.0)
        slots, nxt = torch.tensor(slots, device=self.device), torch.tensor(nxt, device=self.device)
        scale = torch.tensor(scale, device=self.device)
        W = {}
        for n in self.names:
            w = self.buffer[n][slots, runs]
            if scale.any():
                w = w + scale.view(-1, *([1] * (w.ndim - 1))) * (self.buffer[n][nxt, runs] - w)
            W[n] = w.requires_grad_(True) if n in leaves else w
        return W, runs

    def gradients(self, x):
        for n in self.names:
            self.buffer[n][self.t % self.H] = self.P[n]
        grads = {}
        with torch.enable_grad():
            total, wanted = 0.0, []
            for k, copies in self.classes.items():
                leaves = [n for n in self.names if k is None or self.stage[n] == k]
                W, runs = self.copy_weights(copies, leaves)
                total = total + forward(W, x[runs.cpu()].to(self.device), self.vocab, own=k).sum()
                wanted += [(k, n, W[n]) for n in leaves]
            G = torch.autograd.grad(total, [w for *_, w in wanted])
        for (k, n, _), g in zip(wanted, G):
            grads[(k, n)] = g
        out = {}
        for n in self.names:
            parts = [grads[(c, n)] for c in (self.stage[n], None) if (c, n) in grads]
            g = torch.cat(parts) if len(parts) > 1 else parts[0]
            own, fresh = g[self.own_index[n]], g[self.fresh_index[n]]
            err = (own - fresh).flatten(1)[:, self.perm[n]].view_as(own)
            mask = self.permuted.view(-1, *([1] * (own.ndim - 1)))
            out[n] = torch.where(mask, fresh + err, own)
        return out

    def lr_scales(self):
        t = self.t
        base = min(1.0, (t + 1) / tp.WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / tp.STEPS)))
        power = max(0.0, 1 - t / (tp.STEPS / 4))
        per_stage = torch.ones(len(self.runs), tp.STAGES)
        for r, pm in enumerate(self.pipemare):
            if pm:
                per_stage[r] = torch.tensor([max(tau, 1) ** -power for tau in self.tau[r]])
        return self.lr * base, per_stage.to(self.device)

    @torch.no_grad()
    def step(self):
        x = self.draw(self.train_ids, self.gens)
        g = self.gradients(x)
        norm = torch.sqrt(sum((v.flatten(1) ** 2).sum(1) for v in g.values()))
        clip = torch.clamp(tp.CLIP / (norm + 1e-12), max=1.0)
        lr, per_stage = self.lr_scales()
        b1, b2 = tp.BETAS
        t1 = self.t + 1
        bc1, bc2 = 1 - b1 ** t1, 1 - b2 ** t1
        for n in self.names:
            gn = g[n] * clip.view(-1, *([1] * (g[n].ndim - 1)))
            shape = (-1,) + (1,) * (gn.ndim - 1)
            step_lr = (lr * per_stage[:, self.stage[n]]).view(shape)
            self.m[n].mul_(b1).add_(gn, alpha=1 - b1)
            self.v[n].mul_(b2).addcmul_(gn, gn, value=1 - b2)
            update = -(step_lr / bc1) * self.m[n] / ((self.v[n].sqrt() / math.sqrt(bc2)) + EPS)
            if n in self.br_state:
                br = self.br_update(n, gn[self.br_runs], t1) * lr[self.br_runs].view(shape)
                update[self.br_runs] = br
            self.P[n] += update
        self.t += 1
        return norm

    def br_update(self, n, g, t):
        st, (b1, b2) = self.br_state[n], tp.BETAS
        st["m"].mul_(b1).add_(g, alpha=1 - b1)
        if t % BR_FREQ == 0 or t == 1:
            gc = g.cpu()
            st["L"].mul_(b2).add_(gc @ gc.transpose(1, 2), alpha=1 - b2)
            st["R"].mul_(b2).add_(gc.transpose(1, 2) @ gc, alpha=1 - b2)
            if t == 1:
                st["U"] = torch.stack([torch.linalg.eigh(m)[1].flip(1) for m in st["L"]])
                st["V"] = torch.stack([torch.linalg.eigh(m)[1].flip(1) for m in st["R"]])
            else:
                st["U"] = torch.stack([torch.linalg.qr(a @ u)[0] for a, u in zip(st["L"], st["U"])])
                st["V"] = torch.stack([torch.linalg.qr(a @ v)[0] for a, v in zip(st["R"], st["V"])])
            st["Ud"], st["Vd"] = st["U"].to(self.device), st["V"].to(self.device)
        U, V = st["Ud"], st["Vd"]
        g_rot = U.transpose(1, 2) @ g @ V
        m_rot = U.transpose(1, 2) @ st["m"] @ V
        st["v"].mul_(b2).addcmul_(g_rot, g_rot, value=1 - b2)
        m_hat, v_hat = m_rot / (1 - b1 ** t), st["v"] / (1 - b2 ** t)
        return -(U @ (m_hat / (v_hat.sqrt() + EPS)) @ V.transpose(1, 2))

    @torch.no_grad()
    def val_loss(self, chunk=8):
        g = torch.Generator().manual_seed(0)
        batches = [self.draw(self.val_ids, [g])[0] for _ in range(tp.EVAL_BATCHES)]
        total = torch.zeros(len(self.runs), device=self.device)
        R = len(self.runs)
        for i in range(0, len(batches), chunk):
            xb = torch.cat(batches[i:i + chunk])  # [chunk * B, T + 1]
            x = xb.unsqueeze(0).expand(R, -1, -1).to(self.device)
            total += forward(self.P, x, self.vocab) * len(batches[i:i + chunk])
        return (total / len(batches)).cpu().tolist()


def run_group(runs, device):
    torch.manual_seed(0)
    started = time.time()
    group = Group(runs, device)
    curves = [[] for _ in runs]
    for _ in range(tp.STEPS):
        group.step()
        if group.t % tp.EVAL_EVERY == 0:
            for c, v in zip(curves, group.val_loss()):
                c.append(v)
            print(f"[{device}] step {group.t} {time.time() - started:.0f}s", flush=True)
    seconds = time.time() - started
    return [{"method": m, "delay": d, "lr": lr, "seed": s, "val_curve": c, "final_val_loss": c[-1],
             "engine": "vec", "device": device, "group_seconds": seconds} for (m, d, lr, s), c in zip(runs, curves)]


def write_chunk(name, rows, device, started):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"status": "exploratory", "chunk": name, "script_sha256": sha256(__file__),
              "reference_sha256": sha256(tp.__file__), "git_commit": commit, "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "device": device}, "rows": rows}
    out = ROOT / f"research/results/transformer_pipeline_vec/{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    return out


def worker(device, threads, queue, chunks):
    torch.set_num_threads(threads)
    while True:
        try:
            name = queue.get_nowait()
        except Empty:
            return
        started = time.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"[{device}] start {name}", flush=True)
        out = write_chunk(name, run_group(chunks[name], device), device, started)
        print(f"[{device}] wrote {out.relative_to(ROOT)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devices", nargs="+", default=["mps", "cpu"])
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    chunks = grid_chunks()
    done = {p.stem for p in (ROOT / "research/results/transformer_pipeline_vec").glob("*.json")}
    order = sorted((n for n in chunks if n not in done), key=lambda n: -len(chunks[n]))
    ctx = get_context("spawn")
    queue = ctx.Queue()
    for n in order:
        queue.put(n)
    procs = [ctx.Process(target=worker, args=(d, args.cpu_threads if d == "cpu" else 1, queue, chunks))
             for d in args.devices]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
