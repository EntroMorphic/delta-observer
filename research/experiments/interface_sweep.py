#!/usr/bin/env python3
"""Interface design sweep (exploratory): how much staleness can each interface absorb?

Follows journal/possibilities_synth.md. Every module trains against all other
modules delayed by d optimizer steps, from step 0. Arms change only what crosses
the module boundary. Readout: epochs to fit (100% exact accuracy) and the
staleness budget, the largest delay at which time-to-fit stays within 2x fresh.

Exploratory, not pre-registered: seeds 100-104 only.

    python -m research.experiments.interface_sweep --smoke
    python -m research.experiments.interface_sweep
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections import deque
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from research.experiments.curvature_dynamics import sha256
from research.experiments.second_star_slip_v2 import N_BITS, WIDTHS, adder_dataset

ROOT = Path(__file__).resolve().parents[2]
BATCH, PEAK_LR, WARMUP_EPOCHS, MAX_EPOCHS = 128, 1 / 300, 5, 200
SEEDS = [100, 101, 102, 103, 104]  # exploration seeds; never confirmatory
DELAYS = [0, 1, 4, 16, 64]  # smoke run: 64 on all interfaces already stops every arm
PIPE_WIDTH = 64
# arm -> (stage kind, message width, message quantizer, message extrapolation)
ARMS = {
    "A_cont_w1": ("adder", 1, None, False),
    "B_cont_w4": ("adder", 4, None, False),
    "B_cont_w16": ("adder", 16, None, False),
    "C_round_w1": ("adder", 1, "round", False),
    "D_sign_w4": ("adder", 4, "sign", False),
    "E_extrap_w1": ("adder", 1, None, True),
    "F_pipeline_w64": ("pipeline", PIPE_WIDTH, None, False),
    # follow-ups that separate the F contrast: wide messages with local structure,
    # and a pipeline whose stages each emit one output (local supervision only)
    "G_cont_w64": ("adder", PIPE_WIDTH, None, False),
    "H_pipeline_local_w64": ("pipeline_local", PIPE_WIDTH, None, False),
}


class LocalStage(nn.Module):
    def __init__(self, n_in, width, n_out):
        super().__init__()
        self.trunk, self.head = nn.Sequential(nn.Linear(n_in, width), nn.ReLU()), nn.Linear(width, n_out)

    def forward(self, x):
        h = self.trunk(x)
        return h, self.head(h)


def straight_through(raw, mode):
    """Hard values forward, identity gradient backward."""
    if mode is None:
        return raw
    hard = (raw > 0.5).float() if mode == "round" else torch.where(raw > 0, 1.0, -1.0)
    return raw + (hard - raw).detach()


class StagedNet(nn.Module):
    """A chain of stages that talk only through a message.

    adder: stage s sees bits a_s, b_s and the incoming message; it emits sum bit s
    and, except the last stage, an outgoing message (the last emits the final carry).
    pipeline: a monolithic MLP cut at its hidden layers; only the last stage emits.
    """

    def __init__(self, kind, width, quantizer=None):
        super().__init__()
        self.kind, self.width, self.quantizer, self.n = kind, width, quantizer, len(WIDTHS)
        if kind == "adder":
            self.stages = nn.ModuleList(
                nn.Sequential(nn.Linear(2 + width, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                              nn.Linear(h, 2 if s == self.n - 1 else 1 + width))
                for s, h in enumerate(WIDTHS))
        elif kind == "pipeline_local":
            self.stages = nn.ModuleList(
                LocalStage(2 * N_BITS + 1 if s == 0 else width, width, 2 if s == self.n - 1 else 1)
                for s in range(self.n))
        else:
            first = [nn.Sequential(nn.Linear(2 * N_BITS + 1, width), nn.ReLU())]
            middle = [nn.Sequential(nn.Linear(width, width), nn.ReLU()) for _ in range(self.n - 2)]
            last = [nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, N_BITS + 1))]
            self.stages = nn.ModuleList(first + middle + last)

    def initial_message(self, x):
        if self.kind != "adder":
            return x
        carry = x[:, 2 * self.n:2 * self.n + 1]
        carry = 2 * carry - 1 if self.quantizer == "sign" else carry
        return carry.expand(-1, self.width)

    def stage(self, s, params, x, message):
        """Returns (columns of the output this stage emits, its outputs, outgoing message)."""
        module = self.stages[s]
        if self.kind == "pipeline_local":
            h, out = functional_call(module, params, (message,))
            return slice(s, s + 2 if s == self.n - 1 else s + 1), out, (None if s == self.n - 1 else h)
        if self.kind == "pipeline":
            out = functional_call(module, params, (message,))
            return (slice(0, N_BITS + 1), out, None) if s == self.n - 1 else (None, None, out)
        out = functional_call(module, params, (torch.cat([x[:, s:s + 1], x[:, s + self.n:s + self.n + 1], message], 1),))
        if s == self.n - 1:
            return slice(s, s + 2), out, None
        return slice(s, s + 1), out[:, :1], straight_through(out[:, 1:], self.quantizer)


def sse(net, columns, out, y):
    return 0.0 if columns is None else ((out - y[:, columns]) ** 2).sum() / (len(y) * y.shape[1])


def run_from(net, views, x, y, start, message):
    """Loss from stage `start` onward, given the message into it."""
    loss = 0.0
    for s in range(start, net.n):
        columns, out, message = net.stage(s, views[s], x, message)
        loss = loss + sse(net, columns, out, y)
    return loss


def incoming_messages(net, views, x):
    """Message into every stage under one set of views (no gradient)."""
    with torch.no_grad():
        message, messages = net.initial_message(x), []
        for s in range(net.n):
            messages.append(message)
            _, _, message = net.stage(s, views[s], x, message)
    return messages


def predict(new, old):
    return 2 * new - old


class Trainer:
    def __init__(self, arm, seed, delay, order_offset=0):
        self.arm, (kind, width, quantizer, self.extrapolate) = arm, ARMS[arm]
        self.delay = delay
        torch.manual_seed(seed)
        self.inputs, self.outputs = adder_dataset()
        self.net = StagedNet(kind, width, quantizer)
        self.order = torch.Generator().manual_seed(seed + 10_000 + 1_000 * order_offset)
        self.steps_per_epoch = len(self.inputs) // BATCH
        self.opt = torch.optim.Adam(self.net.parameters(), lr=PEAK_LR)
        warmup = WARMUP_EPOCHS * self.steps_per_epoch
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda t: min(1.0, t / warmup))
        self.history = deque([self.snapshot()], maxlen=2 * delay + 1)

    def snapshot(self):
        return [{k: p.detach().clone() for k, p in st.named_parameters()} for st in self.net.stages]

    def view(self, delay):
        return self.history[max(0, len(self.history) - 1 - delay)]

    def live(self):
        return [dict(st.named_parameters()) for st in self.net.stages]

    def full_loss(self, views, x, y):
        return run_from(self.net, views, x, y, 0, self.net.initial_message(x))

    def gradients(self, x, y):
        """Each stage's gradient when every other stage is `delay` steps old."""
        live = self.live()
        if self.delay == 0:
            loss = self.full_loss(live, x, y)
            return torch.autograd.grad(loss, list(self.net.parameters()))
        old = self.view(self.delay)
        if self.extrapolate:
            return self.extrapolated_gradients(x, y, live, old, self.view(2 * self.delay))
        into = incoming_messages(self.net, old, x)
        grads = []
        for m in range(self.net.n):
            views = [live[s] if s == m else old[s] for s in range(self.net.n)]
            grads += torch.autograd.grad(run_from(self.net, views, x, y, m, into[m]), list(live[m].values()))
        return grads

    def extrapolated_gradients(self, x, y, live, old, older):
        """Each stage extrapolates both messages it would receive: the forward
        message from upstream and the backward message (gradient) from downstream."""
        into_d, into_2d = incoming_messages(self.net, old, x), incoming_messages(self.net, older, x)
        grads = []
        for m in range(self.net.n):
            columns, out, message = self.net.stage(m, live[m], x, predict(into_d[m], into_2d[m]))
            surrogate = sse(self.net, columns, out, y)
            if message is not None:
                backward = []
                for views in (old, older):
                    probe = message.detach().requires_grad_(True)
                    backward.append(torch.autograd.grad(run_from(self.net, views, x, y, m + 1, probe), probe)[0])
                surrogate = surrogate + (predict(*backward) * message).sum()
            grads += torch.autograd.grad(surrogate, list(live[m].values()))
        return grads

    def step(self, idx):
        for p, g in zip(self.net.parameters(), self.gradients(self.inputs[idx], self.outputs[idx])):
            p.grad = g
        self.opt.step(); self.sched.step()
        self.history.append(self.snapshot())

    def accuracy(self):
        with torch.no_grad():
            out = torch.zeros_like(self.outputs)
            message = self.net.initial_message(self.inputs)
            for s in range(self.net.n):
                columns, o, message = self.net.stage(s, self.live()[s], self.inputs, message)
                if columns is not None:
                    out[:, columns] = o
            return float((torch.round(out) == self.outputs).all(1).float().mean())

    def interface_meaning(self):
        """R^2 of the true carry into each adder stage from the message (linear fit)."""
        if self.net.kind != "adder":
            return None
        x, n = self.inputs, self.net.n
        messages = incoming_messages(self.net, [{k: p.detach() for k, p in v.items()} for v in self.live()], x)
        a = sum(x[:, i].long() << i for i in range(n))
        b = sum(x[:, i + n].long() << i for i in range(n))
        cin = x[:, 2 * n].long()
        scores = []
        for s in range(1, n):
            mask = (1 << s) - 1
            truth = (((a & mask) + (b & mask) + cin) >> s).float()
            design = torch.cat([messages[s], torch.ones(len(x), 1)], 1).double()
            coef = torch.linalg.lstsq(design, truth.double().unsqueeze(1)).solution
            residual = truth.double() - (design @ coef).squeeze(1)
            scores.append(float(1 - residual.var() / truth.double().var()))
        return scores


def run_job(job):
    arm, seed, delay, order_offset = job
    torch.set_num_threads(1)
    started = time.time()
    trainer = Trainer(arm, seed, delay, order_offset)
    curve, meaning, fit_epoch, fit99_epoch = [], [], None, None
    for epoch in range(1, MAX_EPOCHS + 1):
        order = torch.randperm(len(trainer.inputs), generator=trainer.order)
        for k, i in enumerate(range(0, len(order), BATCH)):
            trainer.step(order[i:i + BATCH])
            if delay == 0 and epoch <= 10 and (k + 1) % 16 == 0:
                meaning.append({"epoch": epoch - 1 + (k + 1) / trainer.steps_per_epoch,
                                "accuracy": trainer.accuracy(), "r2": trainer.interface_meaning()})
        curve.append(trainer.accuracy())
        if fit99_epoch is None and curve[-1] >= 0.99:
            fit99_epoch = epoch
        if curve[-1] == 1.0:
            fit_epoch = epoch
            break
    return {"arm": arm, "seed": seed, "delay": delay, "order_offset": order_offset, "fit_epoch": fit_epoch, "fit99_epoch": fit99_epoch, "accuracy_curve": curve,
            "final_accuracy": curve[-1], "meaning": meaning or None, "seconds": time.time() - started}


def summarize(rows, key="fit_epoch"):
    table = {}
    for arm in sorted({r["arm"] for r in rows}):
        fits = {}
        for d in DELAYS:
            fits[d] = {r["seed"]: r[key] for r in rows
                       if r["arm"] == arm and r["delay"] == d and r.get("order_offset", 0) == 0}
        budgets = {}
        for seed in fits[0]:
            fresh, budget = fits[0][seed], 0
            if fresh is None:
                budgets[seed] = None
                continue
            for d in DELAYS[1:]:
                f = fits[d].get(seed)
                if f is None or f > 2 * fresh:
                    break
                budget = d
            budgets[seed] = budget
        table[arm] = {"fit_epoch": {str(d): fits[d] for d in DELAYS}, "budget": budgets}
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="one seed, delays 0 and 64")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--fresh-only", nargs="*", default=["C_round_w1", "D_sign_w4"],
                        help="arms that failed to learn fresh in the smoke run; reported, not tuned")
    parser.add_argument("--delays", nargs="+", type=int, default=DELAYS)
    parser.add_argument("--order-control", nargs="*", default=[],
                        help="arms whose fresh runs are repeated under two more batch orders")
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--output", default="research/results/interface_sweep_v0.json")
    args = parser.parse_args()
    seeds, delays = ([100], [0, 64]) if args.smoke else (SEEDS, args.delays)
    jobs = [(a, s, d, 0) for a in args.arms for s in seeds for d in delays if d == 0 or a not in args.fresh_only]
    jobs += [(a, s, 0, k) for a in args.order_control for s in seeds for k in (1, 2)]
    jobs.sort(key=lambda j: -j[2])  # long delays first: they run longest
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows = []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(run_job, jobs):
            rows.append(row)
            print(f"{row['arm']:>15} seed {row['seed']} delay {row['delay']:>4}: fit {row['fit_epoch']} "
                  f"(final {row['final_accuracy']:.3f}, {row['seconds']:.0f}s)", flush=True)
    if args.smoke:
        return
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"status": "exploratory", "plan": "journal/possibilities_synth.md (gitignored)",
              "script_sha256": sha256(__file__), "git_commit": commit, "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "settings": {"order_control": args.order_control, "fresh_only": args.fresh_only, "seeds": seeds, "delays": delays, "arms": {a: ARMS[a] for a in args.arms},
                           "widths": WIDTHS, "batch": BATCH, "peak_lr": PEAK_LR, "warmup_epochs": WARMUP_EPOCHS,
                           "max_epochs": MAX_EPOCHS, "schedule": "linear warmup then constant"},
              "summary": {"fit100": summarize(rows), "fit99": summarize(rows, "fit99_epoch")}, "rows": rows}
    (ROOT / args.output).write_text(json.dumps(report, indent=1))
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
