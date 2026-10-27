#!/usr/bin/env python3
"""Interface staleness and cross-module curvature in the compositional adder.

Implements research/reports/second_star_slip_protocol.md. Run from the
repository root, in the paper environment:

    python -m research.experiments.second_star_slip
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import time
from collections import deque
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr
from torch.func import functional_call

from models.delta_observer import ALPHA, generate_dataset, get_scheduler
from research.experiments.curvature_dynamics import exact_hessian, flat_loss, sha256
from research.experiments.curvature_structure import block_labels
from research.experiments.scaffolding_ablation import PaperComp, PaperMono, _carry_in

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "research/reports/second_star_slip_protocol.md"
MODULES = ("bit0", "bit1", "bit2", "bit3")
INTERFACES = (0, 1, 2)
DELAYS = (1, 10, 100)
EPOCHS, BATCH = 200, 32
PRICE_ONSETS = (0, 5, 20, 60)
PRICE_TARGETS = ("0", "1", "2", "all")
HARM_ONSETS = (2, 5, 10, 15, 20, 30, 60)
PREDICTORS = ("kappa_cut", "kappa_adj", "lambda_max", "grad_norm", "momentum_change", "message_size",
              "message_drift")
KILL_COMPETITORS = ("lambda_max", "grad_norm", "momentum_change")
STRICT_COMPETITORS = ("message_size", "message_drift")


# =============================================================================
# Training with stale interfaces
# =============================================================================

def delay_matrix(ks):
    """delay[m][n]: sum of interface delays between modules m and n."""
    return [[sum(ks[min(m, n):max(m, n)]) for n in range(4)] for m in range(4)]


def module_params(model):
    return [{n: p for n, p in getattr(model, name).named_parameters()} for name in MODULES]


def snapshot(model):
    return [{n: p.detach().clone() for n, p in views.items()} for views in module_params(model)]


def mixed_forward(model, views, x):
    """PaperComp forward with module i evaluated at the parameters views[i]."""
    carry, sums = _carry_in(x), []
    for i, name in enumerate(MODULES):
        bit_input = torch.cat([x[:, i:i + 1], x[:, i + 4:i + 5], carry], dim=1)
        out = functional_call(getattr(model, name), views[i], (bit_input,))
        sums.append(out[:, 0:1])
        carry = out[:, 1:2]
    return torch.cat(sums + [carry], dim=1)


def carries(model, views, x):
    """The carry passed across each interface, for parameters views."""
    carry, result = _carry_in(x), []
    for i, name in enumerate(MODULES[:3]):
        bit_input = torch.cat([x[:, i:i + 1], x[:, i + 4:i + 5], carry], dim=1)
        carry = functional_call(getattr(model, name), views[i], (bit_input,))[:, 1:2]
        result.append(carry)
    return result


class Trainer:
    """train_sources' compositional run, with optional per-interface staleness."""

    def __init__(self, seed, epochs=EPOCHS):
        torch.manual_seed(seed)
        self.inputs, self.outputs, _, _ = generate_dataset()
        PaperMono(9)  # consume the RNG exactly as train_sources does
        self.model = PaperComp()
        n_batches = len(self.inputs) // BATCH
        self.opt = torch.optim.Adam(self.model.parameters(), lr=ALPHA)
        self.sched = get_scheduler(self.opt, n_batches * 10, n_batches * epochs, ALPHA, ALPHA / 1000)
        self.history = deque([snapshot(self.model)], maxlen=max(DELAYS) * 3 + 1)
        self.step_count = 0

    def view(self, delay):
        return self.history[max(0, len(self.history) - 1 - delay)]

    def step(self, idx, ks=(0, 0, 0), force_split=False):
        x, y = self.inputs[idx], self.outputs[idx]
        if not any(ks) and not force_split:
            loss = F.mse_loss(self.model(x), y)
            self.opt.zero_grad(); loss.backward()
        else:
            delays, live = delay_matrix(ks), module_params(self.model)
            for m in range(4):
                views = [live[n] if n == m else self.view(delays[m][n])[n] for n in range(4)]
                loss = F.mse_loss(mixed_forward(self.model, views, x), y)
                grads = torch.autograd.grad(loss, list(live[m].values()))
                for p, g in zip(live[m].values(), grads):
                    p.grad = g
        self.opt.step(); self.sched.step()
        self.history.append(snapshot(self.model))
        self.step_count += 1

    def full_loss(self):
        with torch.no_grad():
            return float(F.mse_loss(self.model(self.inputs), self.outputs))

    def accuracy(self):
        with torch.no_grad():
            return float((torch.round(self.model(self.inputs)) == self.outputs).all(1).float().mean())


def batches(seed_trainer):
    indices = torch.randperm(len(seed_trainer.inputs))
    return [indices[i:i + BATCH] for i in range(0, len(indices), BATCH)]


def target_ks(target, k):
    return (k, k, k) if target == "all" else tuple(k if j == int(target) else 0 for j in INTERFACES)


# =============================================================================
# Part 1: price of staleness
# =============================================================================

def price_job(job):
    seed, target, k, onset = job
    torch.set_num_threads(1)
    trainer = Trainer(seed)
    ks = (0, 0, 0) if target is None else target_ks(target, k)
    losses, accs = [], []
    for epoch in range(1, EPOCHS + 1):
        for idx in batches(trainer):
            trainer.step(idx, ks if epoch > onset else (0, 0, 0))
        losses.append(trainer.full_loss()); accs.append(trainer.accuracy())
    running_min, spikes = np.inf, 0
    for loss in losses:
        spikes += loss > 10 * running_min
        running_min = min(running_min, loss)
    first_fit = next((e + 1 for e, a in enumerate(accs) if a == 1.0 and min(accs[e:]) == 1.0), None)
    return {"seed": seed, "target": target, "k": k, "onset": onset, "final_loss": losses[-1],
            "final_accuracy": accs[-1], "first_full_fit_epoch": first_fit, "spike_epochs": int(spikes),
            "loss": losses, "accuracy": accs}


# =============================================================================
# Part 2: does cross-module curvature predict harm?
# =============================================================================

def coupling(H, labels, a, b):
    A, B = np.isin(labels, a), np.isin(labels, b)
    norm = lambda X, Y: np.linalg.norm(H[np.ix_(X, Y)])
    return float(norm(A, B) / np.sqrt(norm(A, A) * norm(B, B)))


def predictors_at(trainer, adam_prev):
    """Every Part 2 predictor at the current step, per interface and delay."""
    model, x, y = trainer.model, trainer.inputs, trainer.outputs
    labels, _ = block_labels(model)
    loss, w = flat_loss(model, x, y)
    H, _ = exact_hessian(loss, w)
    H = H.numpy()
    lam = float(eigsh(H, k=1, which="LA", return_eigenvectors=False)[0])
    params = list(model.parameters())
    grads = torch.autograd.grad(F.mse_loss(model(x), y), params)
    grad_norm = float(torch.sqrt(sum((g ** 2).sum() for g in grads)))

    live = [{n: p.detach().requires_grad_(True) for n, p in v.items()} for v in snapshot(model)]
    carry_now = carries(model, live, x)
    message = torch.autograd.grad(F.mse_loss(mixed_forward_from_carries(model, live, x, carry_now), y), carry_now)

    names = [n for n, _ in model.named_parameters()]
    adam_now = {n: trainer.opt.state[p]["exp_avg"].clone() for n, p in zip(names, params)}
    rows = {}
    for j in INTERFACES:
        near = [n for n in names if n.split(".")[0] in (MODULES[j], MODULES[j + 1])]
        now = torch.cat([adam_now[n].reshape(-1) for n in near])
        prev = torch.cat([adam_prev[n].reshape(-1) for n in near])
        base = {"kappa_cut": coupling(H, labels, list(range(j + 1)), list(range(j + 1, 4))),
                "kappa_adj": coupling(H, labels, [j], [j + 1]),
                "lambda_max": lam, "grad_norm": grad_norm,
                "momentum_change": float((now - prev).norm() / now.norm()),
                "message_size": float(message[j].pow(2).mean().sqrt())}
        for k in DELAYS:
            with torch.no_grad():
                old = carries(model, trainer.view(k), x)[j]
            rows[(j, k)] = {**base, "message_drift": float((carry_now[j].detach() - old).pow(2).mean().sqrt())}
    return rows


def mixed_forward_from_carries(model, views, x, carry_list):
    """Forward in which module i+1 receives the given carry tensors (for ∂L/∂c)."""
    carry, sums = _carry_in(x), []
    for i, name in enumerate(MODULES):
        bit_input = torch.cat([x[:, i:i + 1], x[:, i + 4:i + 5], carry], dim=1)
        out = functional_call(getattr(model, name), views[i], (bit_input,))
        sums.append(out[:, 0:1])
        carry = carry_list[i] if i < 3 else out[:, 1:2]
    return torch.cat(sums + [carry], dim=1)


def harm_job(seed):
    torch.set_num_threads(1)
    trainer = Trainer(seed)
    fresh_loss, units, adam_at = {}, [], {}
    names = [n for n, _ in trainer.model.named_parameters()]
    for epoch in range(1, max(HARM_ONSETS) + 2):
        epoch_batches = batches(trainer)
        if epoch - 1 in HARM_ONSETS:
            onset = epoch - 1
            rows = predictors_at(trainer, adam_at[onset - 1])  # first moment one epoch earlier
            for j in INTERFACES:
                for k in DELAYS:
                    branch = copy.deepcopy(trainer)
                    for idx in epoch_batches:
                        branch.step(idx, target_ks(str(j), k))
                    units.append({"seed": seed, "onset": onset, "interface": j, "k": k,
                                  "stale_loss": branch.full_loss(), **rows[(j, k)]})
        for idx in epoch_batches:
            trainer.step(idx)
        fresh_loss[epoch] = trainer.full_loss()
        adam_at[epoch] = {n: trainer.opt.state[p]["exp_avg"].clone()
                          for n, p in zip(names, trainer.model.parameters())}
    for unit in units:
        unit["fresh_loss"] = fresh_loss[unit["onset"] + 1]
        unit["harm"] = float(np.log10(unit["stale_loss"]) - np.log10(unit["fresh_loss"]))
    return units


# =============================================================================
# Analysis
# =============================================================================

def analyze(units, resamples=2000, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    seeds = sorted({u["seed"] for u in units})
    result = {}
    for k in DELAYS:
        rows = [u for u in units if u["k"] == k]
        rho = lambda rs, p: spearmanr([r[p] for r in rs], [r["harm"] for r in rs])[0]
        by_seed = {s: [r for r in rows if r["seed"] == s] for s in seeds}
        boot = {p: [] for p in PREDICTORS}
        for _ in range(resamples):
            sample = [r for s in rng.choice(seeds, len(seeds)) for r in by_seed[s]]
            for p in PREDICTORS:
                boot[p].append(rho(sample, p))
        boot = {p: np.array(v) for p, v in boot.items()}
        diff = {p: [float(x) for x in np.nanpercentile(boot["kappa_cut"] - boot[p], [5, 95])]
                for p in PREDICTORS if p != "kappa_cut"}
        result[str(k)] = {"n": len(rows), "rho": {p: float(rho(rows, p)) for p in PREDICTORS},
                          "kappa_cut_minus_competitor_90ci": diff}
    beats = lambda comps: sum(all(result[str(k)]["kappa_cut_minus_competitor_90ci"][c][0] > 0 for c in comps)
                              for k in DELAYS)
    result["kill_criterion"] = {"delays_where_kappa_beats_all": beats(KILL_COMPETITORS),
                                "supported": beats(KILL_COMPETITORS) >= 2}
    result["strict_test"] = {c: beats([c]) for c in STRICT_COMPETITORS}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", default="research/results/second_star_slip_v1.json")
    args = parser.parse_args()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    price_jobs = [(s, None, 0, 0) for s in args.seeds]
    price_jobs += [(s, t, k, o) for s in args.seeds for t in PRICE_TARGETS for k in DELAYS for o in PRICE_ONSETS]
    with get_context("spawn").Pool(args.workers) as pool:
        harm_async = pool.map_async(harm_job, args.seeds)
        price = []
        for result in pool.imap_unordered(price_job, price_jobs):
            price.append(result)
            if len(price) % 20 == 0:
                print(f"part 1: {len(price)}/{len(price_jobs)}", flush=True)
        units = [u for seed_units in harm_async.get() for u in seed_units]
    price.sort(key=lambda r: (str(r["target"]), r["k"], r["onset"], r["seed"]))
    analysis = analyze(units)
    print(json.dumps(analysis, indent=1))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"protocol": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": sha256(PROTOCOL),
              "script_sha256": sha256(__file__), "git_commit": commit, "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "scipy": scipy.__version__},
              "settings": {"seeds": args.seeds, "delays": DELAYS, "price_onsets": PRICE_ONSETS,
                           "harm_onsets": HARM_ONSETS},
              "part1_price": price, "part2_units": units, "part2_analysis": analysis}
    destination = ROOT / args.output
    destination.write_text(json.dumps(report, indent=1))
    print(f"Wrote {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
