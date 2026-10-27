#!/usr/bin/env python3
"""Second Star Slip v2: which interface to keep fresh, in a heterogeneous chain adder.

Implements research/reports/second_star_slip_v2_protocol.md. Run from the
repository root, in the paper environment:

    python -m research.experiments.second_star_slip_v2 --pilot      # calibration seed only
    python -m research.experiments.second_star_slip_v2              # confirmatory run
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import subprocess
import time
from collections import deque
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn
from torch.func import functional_call

from research.experiments.curvature_dynamics import sha256

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "research/reports/second_star_slip_v2_protocol.md"
N_BITS = 6
WIDTHS = (8, 32, 8, 16, 32, 8)
BATCH, EPOCHS, PEAK_LR, WARMUP_EPOCHS = 128, 60, 1 / 300, 5
PILOT_SEEDS = [100, 101, 102]  # calibration only; never confirmatory
REPEATS = 3


# =============================================================================
# Task and model
# =============================================================================

def adder_dataset(n_bits=N_BITS):
    """Every a + b + carry-in for n-bit a, b. Inputs: a bits, b bits, carry-in."""
    rows, targets = [], []
    for a in range(2 ** n_bits):
        for b in range(2 ** n_bits):
            for carry in (0, 1):
                total = a + b + carry
                rows.append([(a >> i) & 1 for i in range(n_bits)] + [(b >> i) & 1 for i in range(n_bits)] + [carry])
                targets.append([(total >> i) & 1 for i in range(n_bits + 1)])
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


class FullAdder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(3, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, 2)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class ChainAdder(nn.Module):
    """Ripple-carry chain of full-adder modules; module i passes its carry to i + 1."""

    def __init__(self, widths=WIDTHS):
        super().__init__()
        self.cells = nn.ModuleList(FullAdder(h) for h in widths)

    def forward(self, x, views=None, carry_override=None):
        n = len(self.cells)
        carry, sums, carries = x[:, 2 * n:2 * n + 1], [], []
        for i, cell in enumerate(self.cells):
            bit_input = torch.cat([x[:, i:i + 1], x[:, i + n:i + n + 1], carry], dim=1)
            out = cell(bit_input) if views is None else functional_call(cell, views[i], (bit_input,))
            sums.append(out[:, 0:1])
            carry = out[:, 1:2]
            if carry_override is not None and i < n - 1:
                carry = carry_override[i]
            carries.append(carry)
        return torch.cat(sums + [carry], dim=1), carries[:-1]


def lr_factor(step, steps_per_epoch):
    warmup, total = WARMUP_EPOCHS * steps_per_epoch, EPOCHS * steps_per_epoch
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / (total - warmup)
    return max(1e-3, 0.5 * (1 + math.cos(math.pi * progress)))


# =============================================================================
# Training with one stale interface
# =============================================================================

class Trainer:
    def __init__(self, seed, max_delay=0):
        torch.manual_seed(seed)
        self.inputs, self.outputs = adder_dataset()
        self.model = ChainAdder()
        self.steps_per_epoch = len(self.inputs) // BATCH
        self.opt = torch.optim.Adam(self.model.parameters(), lr=PEAK_LR)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, partial(lr_factor, steps_per_epoch=self.steps_per_epoch))
        self.history = deque([self.snapshot()], maxlen=max_delay + 1)

    def snapshot(self):
        return [{n: p.detach().clone() for n, p in cell.named_parameters()} for cell in self.model.cells]

    def view(self, delay):
        return self.history[max(0, len(self.history) - 1 - delay)]

    def live(self):
        return [dict(cell.named_parameters()) for cell in self.model.cells]

    def step(self, idx, interface=None, delay=0):
        x, y = self.inputs[idx], self.outputs[idx]
        if interface is None or delay == 0:
            loss = F.mse_loss(self.model(x)[0], y)
            self.opt.zero_grad(); loss.backward()
        else:
            live, old = self.live(), self.view(delay)
            n = len(live)
            for side in (range(interface + 1), range(interface + 1, n)):
                views = [live[m] if m in side else old[m] for m in range(n)]
                loss = F.mse_loss(self.model(x, views)[0], y)
                params = [p for m in side for p in live[m].values()]
                for p, g in zip(params, torch.autograd.grad(loss, params)):
                    p.grad = g
        self.opt.step(); self.sched.step()
        self.history.append(self.snapshot())

    def epoch_batches(self, generator=None):
        order = torch.randperm(len(self.inputs), generator=generator)
        return [order[i:i + BATCH] for i in range(0, len(order), BATCH)]

    def full_loss(self):
        with torch.no_grad():
            return float(F.mse_loss(self.model(self.inputs)[0], self.outputs))

    def accuracy(self):
        with torch.no_grad():
            return float((torch.round(self.model(self.inputs)[0]) == self.outputs).all(1).float().mean())


def fresh_trajectory(seed, max_delay):
    """Fresh training; yields (epoch, trainer) at the end of each epoch."""
    trainer = Trainer(seed, max_delay)
    for epoch in range(1, EPOCHS + 1):
        for idx in trainer.epoch_batches():
            trainer.step(idx)
        yield epoch, trainer


# =============================================================================
# Predictors (information available at the onset)
# =============================================================================

def full_batch_hvp(trainer, vector):
    params = list(trainer.model.parameters())
    loss = F.mse_loss(trainer.model(trainer.inputs)[0], trainer.outputs)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    dot = sum((g * v).sum() for g, v in zip(grads, vector))
    return [h.detach() for h in torch.autograd.grad(dot, params)]


def adam_scale(trainer):
    beta2, eps = trainer.opt.param_groups[0]["betas"][1], trainer.opt.param_groups[0]["eps"]
    scale = []
    for p in trainer.model.parameters():
        state = trainer.opt.state[p]
        scale.append(1 / ((state["exp_avg_sq"] / (1 - beta2 ** int(state["step"]))).sqrt() + eps))
    return scale


def side_gradient_change(trainer, interface, old, scale):
    """Secant cross error: how each side's full-batch gradient changes when the
    other side is swapped for its delayed parameters (curvature and kinks)."""
    n, live = len(trainer.model.cells), trainer.live()
    params = list(trainer.model.parameters())
    cell_of = [m for m in range(n) for _ in live[m]]
    fresh = torch.autograd.grad(F.mse_loss(trainer.model(trainer.inputs)[0], trainer.outputs), params)
    error_sq, scaled_sq = 0.0, 0.0
    for side in (range(interface + 1), range(interface + 1, n)):
        views = [live[m] if m in side else old[m] for m in range(n)]
        loss = F.mse_loss(trainer.model(trainer.inputs, views)[0], trainer.outputs)
        stale = torch.autograd.grad(loss, params, allow_unused=True)
        for c, g, f, s in zip(cell_of, stale, fresh, scale):
            if c in side:
                error_sq += float(((g - f) ** 2).sum())
                scaled_sq += float(((s * (g - f)) ** 2).sum())
    return error_sq ** 0.5, scaled_sq ** 0.5


def predictors(trainer, delay):
    """Per-interface predictors at the current step for a given delay."""
    n = len(trainer.model.cells)
    now, old = trainer.snapshot(), trainer.view(delay)
    moved = [[now[m][k] - old[m][k] for k in now[m]] for m in range(n)]
    cell_of = [m for m in range(n) for _ in now[m]]
    scale = adam_scale(trainer)
    with torch.no_grad():
        carries_now = trainer.model(trainer.inputs)[1]
        carries_old = trainer.model(trainer.inputs, old)[1]
    carries_live = [c.detach().requires_grad_(True) for c in carries_now]
    loss = F.mse_loss(trainer.model(trainer.inputs, carry_override=carries_live)[0], trainer.outputs)
    messages = torch.autograd.grad(loss, carries_live)
    rows = []
    for j in range(n - 1):
        side_a = [m <= j for m in cell_of]
        flat_moved = [t for m in range(n) for t in moved[m]]
        error_sq, scaled_sq = 0.0, 0.0
        for source_is_a in (True, False):
            vector = [t if a == source_is_a else torch.zeros_like(t) for t, a in zip(flat_moved, side_a)]
            hv = full_batch_hvp(trainer, vector)
            for h, a, s in zip(hv, side_a, scale):
                if a != source_is_a:  # the other side's gradient error, −H_xy Δθ_y
                    error_sq += float((h ** 2).sum())
                    scaled_sq += float(((s * h) ** 2).sum())
        near = [t for m in (j, j + 1) for t in moved[m]]
        secant, secant_adam = side_gradient_change(trainer, j, old, scale)
        rows.append({
            "interface": j,
            "secant_error": secant,
            "secant_error_adam": secant_adam,
            "hessian_error": error_sq ** 0.5,
            "hessian_error_adam": scaled_sq ** 0.5,
            "message_drift": float((carries_now[j] - carries_old[j]).pow(2).mean().sqrt()),
            "message_size": float(messages[j].pow(2).mean().sqrt()),
            "movement_near": float(torch.sqrt(sum((t ** 2).sum() for t in near))),
        })
    return rows


# =============================================================================
# Target: loss gap after a sustained stale window, averaged over batch orders
# =============================================================================

def window_target(trainer, interface, delay, window_epochs, generator_seed):
    """log10 loss gap at the end of the window, stale vs fresh, same batch order."""
    losses = []
    for stale in (False, True):
        branch, generator = copy.deepcopy(trainer), torch.Generator().manual_seed(generator_seed)
        for _ in range(window_epochs):
            for idx in branch.epoch_batches(generator):
                branch.step(idx, interface if stale else None, delay if stale else 0)
        losses.append(branch.full_loss())
    return float(np.log10(losses[1]) - np.log10(losses[0]))


def unit_job(job):
    """All interfaces, repeats and predictors for one (seed, onset, delay, window)."""
    seed, onset, delay, window, state, pilot = job
    torch.set_num_threads(1)
    trainer = state
    # The pilot calibrates the target only; it never computes predictors.
    rows = ([{"interface": j} for j in range(len(trainer.model.cells) - 1)] if pilot
            else predictors(trainer, delay))
    for row in rows:
        row["targets"] = [window_target(trainer, row["interface"], delay, window, 7919 * seed + 31 * onset + r)
                          for r in range(REPEATS)]
        row.update({"seed": seed, "onset": onset, "delay": delay, "window": window})
    return rows


def onset_states(seed, onsets, max_delay):
    states, accuracy = {}, []
    for epoch, trainer in fresh_trajectory(seed, max_delay):
        accuracy.append(trainer.accuracy())
        if epoch in onsets:
            states[epoch] = copy.deepcopy(trainer)
        if epoch >= max(onsets):
            break
    return states, accuracy


# =============================================================================
# Analysis
# =============================================================================

PRIMARY = "secant_error"
PREDICTORS = ("secant_error", "secant_error_adam", "hessian_error", "hessian_error_adam", "message_drift",
              "message_size", "movement_near")


def within_moment(rows, key, target):
    groups = {}
    for r in rows:
        groups.setdefault((r["seed"], r["onset"]), []).append(r)
    values = []
    for group in groups.values():
        rho = spearmanr([key(r) for r in group], [target(r) for r in group])[0]
        values.append((group[0]["seed"], 0.0 if np.isnan(rho) else rho))
    return values


def reliability(rows):
    """Mean within-moment Spearman between pairs of repeats."""
    pairs = [(a, b) for a in range(REPEATS) for b in range(a + 1, REPEATS)]
    per = [np.mean([v for _, v in within_moment(rows, lambda r, a=a: r["targets"][a], lambda r, b=b: r["targets"][b])])
           for a, b in pairs]
    return float(np.mean(per))


def analyze(rows, resamples=2000, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    mean_target = lambda r: float(np.mean(r["targets"]))
    seeds = sorted({r["seed"] for r in rows})
    # Static prior: each interface's mean target on the other seeds
    prior = {}
    for s in seeds:
        for j in {r["interface"] for r in rows}:
            prior[(s, j)] = float(np.mean([mean_target(r) for r in rows if r["seed"] != s and r["interface"] == j]))
    for r in rows:
        r["static_prior"] = prior[(r["seed"], r["interface"])]
    names = PREDICTORS + ("static_prior",)
    per_seed = {}
    for name in names:
        values = within_moment(rows, lambda r, name=name: r[name], mean_target)
        per_seed[name] = {s: [v for t, v in values if t == s] for s in seeds}
    boot = {name: [] for name in names}
    for _ in range(resamples):
        sample = rng.choice(seeds, len(seeds))
        for name in names:
            boot[name].append(np.mean([v for s in sample for v in per_seed[name][s]]))
    boot = {k: np.array(v) for k, v in boot.items()}
    result = {"reliability": reliability(rows),
              "rho": {name: float(np.mean([v for s in seeds for v in per_seed[name][s]])) for name in names},
              "rho_90ci": {name: [float(x) for x in np.percentile(boot[name], [5, 95])] for name in names},
              "primary_minus_competitor_90ci": {
                  name: [float(x) for x in np.percentile(boot[PRIMARY] - boot[name], [5, 95])]
                  for name in names if name != PRIMARY}}
    gate = result["reliability"] >= 0.7
    ci = result["primary_minus_competitor_90ci"]
    result["gate_passed"] = gate
    result["supported"] = bool(gate and result["rho_90ci"][PRIMARY][0] > 0
                               and all(ci[c][0] > 0 for c in ("message_drift", "movement_near", "static_prior")))
    return result


# =============================================================================

def run(seeds, onsets, delays, windows, workers, pilot=False):
    max_delay = max(delays)
    jobs, accuracy = [], {}
    for seed in seeds:
        states, accuracy[seed] = onset_states(seed, onsets, max_delay)
        jobs += [(seed, onset, d, w, states[onset], pilot) for onset in onsets for d in delays for w in windows]
    with get_context("spawn").Pool(workers) as pool:
        rows = [r for part in pool.imap_unordered(unit_job, jobs) for r in part]
    return rows, accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--onsets", nargs="+", type=int, default=None)
    parser.add_argument("--delays", nargs="+", type=int, default=None)
    parser.add_argument("--windows", nargs="+", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    seeds = PILOT_SEEDS if args.pilot else args.seeds
    rows, accuracy = run(seeds, args.onsets, args.delays, args.windows, args.workers, args.pilot)
    analyses = {}
    for d in args.delays:
        for w in args.windows:
            subset = [r for r in rows if r["delay"] == d and r["window"] == w]
            if args.pilot:
                analyses[f"d{d}_w{w}"] = {"reliability": reliability(subset)}
            else:
                analyses[f"d{d}_w{w}"] = analyze(subset)
    print(json.dumps(analyses, indent=1))
    name = "second_star_slip_v2_pilot.json" if args.pilot else "second_star_slip_v2.json"
    destination = ROOT / (args.output or f"research/results/{name}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"protocol": str(PROTOCOL.relative_to(ROOT)),
              "protocol_sha256": sha256(PROTOCOL) if PROTOCOL.exists() else None,
              "script_sha256": sha256(__file__), "git_commit": commit, "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "scipy": scipy.__version__},
              "settings": {"seeds": seeds, "onsets": args.onsets, "delays": args.delays, "windows": args.windows,
                           "widths": WIDTHS, "batch": BATCH, "epochs": EPOCHS, "repeats": REPEATS},
              "fresh_accuracy": accuracy, "rows": rows, "analysis": analyses}
    destination.write_text(json.dumps(report, indent=1))
    print(f"Wrote {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
