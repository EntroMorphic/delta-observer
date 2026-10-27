#!/usr/bin/env python3
"""Contract experiments (exploratory), following journal/paradigm_synth.md.

    rescue     X3: on adder seeds that stall fresh, is staleness better than
               isotropic gradient noise matched per step and per tensor to the
               staleness error?
    discretize X4: adder chain, width 1, fresh until epoch 10; then optionally
               snap every message to two levels found by 1-D 2-means
               (no labels; no gradient crosses) and/or switch
               every interface to delay 256 or 1024.
    automaton  X5: an interface the student is not told. Each stage reads a
               2-bit symbol; a random 6-state machine moves; each stage emits a
               random 1-bit label of the state. Width and staleness sweep.

    python -m research.experiments.contract_experiments rescue
    python -m research.experiments.contract_experiments automaton
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
from research.experiments.interface_sweep import BATCH, MAX_EPOCHS, Trainer, incoming_messages

ROOT = Path(__file__).resolve().parents[2]
N_STAGES, N_STATES = 6, 6
AUTOMATON_HIDDEN = (32,) * N_STAGES


def automaton_dataset(task_seed, n_states=N_STATES):
    """Every 6-symbol sequence over 4 symbols, laid out like the adder."""
    rng = np.random.default_rng(task_seed)
    delta = rng.integers(n_states, size=(n_states, 4))
    while True:
        label, final_label = rng.integers(2, size=n_states), rng.integers(2, size=n_states)
        if 0 < label.sum() < n_states and 0 < final_label.sum() < n_states:
            break
    rows, targets = [], []
    for code in range(4 ** N_STAGES):
        symbols = [(code >> (2 * s)) & 3 for s in range(N_STAGES)]
        state, out = 0, []
        for sym in symbols:
            state = delta[state, sym]
            out.append(label[state])
        out.append(final_label[state])
        rows.append([sym & 1 for sym in symbols] + [sym >> 1 for sym in symbols] + [0])
        targets.append(out)
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


def automaton_states(task_seed, inputs, n_states=N_STATES):
    """True machine state entering each stage, for every example."""
    rng = np.random.default_rng(task_seed)
    delta = rng.integers(n_states, size=(n_states, 4))
    symbols = (inputs[:, :N_STAGES] + 2 * inputs[:, N_STAGES:2 * N_STAGES]).long().numpy()
    states = np.zeros((len(inputs), N_STAGES), dtype=int)
    for s in range(1, N_STAGES):
        states[:, s] = delta[states[:, s - 1], symbols[:, s - 1]]
    return states


class NoiseTrainer(Trainer):
    """Fresh gradients plus isotropic noise whose norm, per tensor and per step,
    equals the error that `delay`-step staleness would have caused."""

    def gradients(self, x, y):
        stale = super().gradients(x, y)
        delay, self.delay = self.delay, 0
        fresh = super().gradients(x, y)
        self.delay = delay
        out = []
        for f, s in zip(fresh, stale):
            noise = torch.randn_like(f)
            out.append(f + noise * ((s - f).norm() / noise.norm().clamp_min(1e-12)))
        return out


def participation_ratio(messages):
    """Effective dimension of what crosses each interface: (sum λ)² / sum λ²."""
    ratios = []
    for m in messages[1:]:
        eig = torch.linalg.eigvalsh(torch.cov(m.T.double()).reshape(m.shape[1], m.shape[1])).clamp_min(0)
        ratios.append(float(eig.sum() ** 2 / (eig ** 2).sum().clamp_min(1e-30)))
    return ratios


def state_information(messages, states):
    """Fraction of true states separable by nearest class mean in message space,
    per interface. Width-fair only for comparisons within one width."""
    scores = []
    for s in range(1, N_STAGES):
        m, q = messages[s].double(), torch.tensor(states[:, s])
        classes = q.unique()
        means = torch.stack([m[q == c].mean(0) for c in classes])
        pred = classes[torch.cdist(m, means).argmin(1)]
        scores.append(float((pred == q).double().mean()))
    return scores


DISCRETIZE_EPOCH = 10
MODES = ("continuous", "round")


def two_means(values, iterations=50):
    """1-D 2-means: (threshold, low centre, high centre). Uses no labels."""
    low, high = values.min(), values.max()
    for _ in range(iterations):
        threshold = (low + high) / 2
        low, high = values[values <= threshold].mean(), values[values > threshold].mean()
    return float((low + high) / 2), float(low), float(high)


def two_level_snap(trainer):
    live = [{k: p.detach() for k, p in v.items()} for v in trainer.live()]
    messages = incoming_messages(trainer.net, live, trainer.inputs)
    return [two_means(messages[s + 1][:, 0]) for s in range(trainer.net.n - 1)]


def discretize_job(job):
    _, arm, seed, delay, mode = job
    torch.set_num_threads(1)
    started = time.time()
    trainer = Trainer(arm, seed, delay)
    trainer.delay = 0
    curve, fit_epoch, at_switch = [], None, None
    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == DISCRETIZE_EPOCH + 1:
            at_switch = curve[-1]
            trainer.delay = delay
            if mode == "round":
                trainer.net.snap = two_level_snap(trainer)
        order = torch.randperm(len(trainer.inputs), generator=trainer.order)
        for i in range(0, len(order), BATCH):
            trainer.step(order[i:i + BATCH])
        curve.append(trainer.accuracy())
        if curve[-1] == 1.0 and epoch > DISCRETIZE_EPOCH:
            fit_epoch = epoch
            break
    return {"kind": "discretize", "arm": arm, "seed": seed, "delay": delay, "order_offset": 0, "mode": mode,
            "accuracy_at_switch": at_switch, "fit_epoch": fit_epoch, "final_accuracy": curve[-1],
            "accuracy_curve": curve, "seconds": time.time() - started}


def run_job(job):
    if job[0] == "discretize":
        return discretize_job(job)
    kind, arm, seed, delay, order_offset = job
    torch.set_num_threads(1)
    started = time.time()
    if kind == "automaton":
        trainer = Trainer(arm, seed, delay, order_offset, automaton_dataset(seed), AUTOMATON_HIDDEN)
    elif kind == "noise":
        trainer = NoiseTrainer(arm, seed, delay, order_offset)
    else:
        trainer = Trainer(arm, seed, delay, order_offset)
    curve, fit_epoch = [], None
    for epoch in range(1, MAX_EPOCHS + 1):
        order = torch.randperm(len(trainer.inputs), generator=trainer.order)
        for i in range(0, len(order), BATCH):
            trainer.step(order[i:i + BATCH])
        curve.append(trainer.accuracy())
        if curve[-1] == 1.0:
            fit_epoch = epoch
            break
    row = {"kind": kind, "arm": arm, "seed": seed, "delay": delay, "order_offset": order_offset,
           "fit_epoch": fit_epoch, "final_accuracy": curve[-1], "accuracy_curve": curve,
           "seconds": time.time() - started}
    if kind == "automaton" and trainer.net.kind == "adder":
        live = [{k: p.detach() for k, p in v.items()} for v in trainer.live()]
        messages = incoming_messages(trainer.net, live, trainer.inputs)
        row["participation_ratio"] = participation_ratio(messages)
        row["state_separability"] = state_information(messages, automaton_states(seed, trainer.inputs))
    return row


RESCUE_CASES = [("A_cont_w1", 100, 4), ("A_cont_w1", 104, 16), ("B_cont_w16", 102, 4), ("B_cont_w16", 104, 16)]
HEALTHY_CASES = [("A_cont_w1", 101, 4), ("A_cont_w1", 102, 4), ("A_cont_w1", 103, 4)]
AUTOMATON_ARMS = ["A_cont_w1", "B_cont_w2", "B_cont_w4", "B_cont_w16", "G_cont_w64",
                  "H_pipeline_local_w64", "F_pipeline_w64"]


def jobs_for(experiment, seeds):
    if experiment == "rescue":
        jobs = []
        for arm, seed, delay in RESCUE_CASES + HEALTHY_CASES:
            for k in (0, 1, 2):
                jobs += [("stale", arm, seed, delay, k), ("noise", arm, seed, delay, k)]
        return jobs
    if experiment == "discretize":
        return [("discretize", "A_cont_w1", seed, d, mode) for seed in seeds
                for d in (0, 256, 1024) for mode in MODES]
    return [("automaton", arm, seed, d, 0) for arm in AUTOMATON_ARMS for seed in seeds for d in (0, 4, 16, 64)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["rescue", "automaton", "discretize"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="default 200-204; 200-209 for discretize")
    parser.add_argument("--modes", nargs="+", default=None, help="discretize only")
    parser.add_argument("--output", default=None)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = list(range(200, 210 if args.experiment == "discretize" else 205))
    jobs = sorted(jobs_for(args.experiment, args.seeds), key=lambda j: -j[3])
    if args.modes:
        jobs = [j for j in jobs if j[0] != "discretize" or j[4] in args.modes]
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(run_job, jobs):
            rows.append(row)
            print(f"{row['kind']:>9} {row['arm']:>21} seed {row['seed']} d {row['delay']:>3} order {row['order_offset']}: "
                  f"fit {row['fit_epoch']} (final {row['final_accuracy']:.3f}, {row['seconds']:.0f}s)", flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"status": "exploratory", "plan": "journal/paradigm_synth.md (gitignored)", "experiment": args.experiment,
              "script_sha256": sha256(__file__), "git_commit": commit, "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "settings": {"discretize_epoch": DISCRETIZE_EPOCH, "seeds": args.seeds, "n_states": N_STATES, "automaton_hidden": AUTOMATON_HIDDEN},
              "rows": rows}
    out = ROOT / (args.output or f"research/results/contract_{args.experiment}_v0.json")
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
