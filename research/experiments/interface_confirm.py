#!/usr/bin/env python3
"""Confirmatory test of locality, width and directional staleness harm.

Implements research/reports/interface_confirm_protocol.md. Run from the
repository root, in the paper environment:

    python -m research.experiments.interface_confirm calibrate   # seeds 300-302 only
    python -m research.experiments.interface_confirm run         # seeds 0-9
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
import scipy
import torch
from scipy.stats import wilcoxon

from research.experiments.contract_experiments import AUTOMATON_HIDDEN, PermutedErrorTrainer, automaton_dataset
from research.experiments.curvature_dynamics import sha256
from research.experiments.interface_sweep import BATCH, MAX_EPOCHS, PEAK_LR, WARMUP_EPOCHS, Trainer
from research.experiments.second_star_slip_v2 import WIDTHS, adder_dataset
from research.optimizers import Muon

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "research/reports/interface_confirm_protocol.md"
SEEDS, CALIBRATION_SEEDS = list(range(10)), [300, 301, 302]
CENSOR = MAX_EPOCHS + 1  # time to fit for a run that never fits
CALIBRATION_LRS = {"sgd": [0.03, 0.1, 0.3, 1.0], "muon": [0.003, 0.01, 0.03, 0.1]}
# Filled in from calibration before freezing; Adam is the exploration setting.
LEARNING_RATES = {"adam": PEAK_LR, "sgd": 0.3, "muon": 0.03}

# H1: same-width locality (G vs H vs F) and width at fixed locality.
H1_ARMS = {"adder": ["B_cont_w4", "G_cont_w64", "H_pipeline_local_w64", "F_pipeline_w64"],
           "automaton": ["B_cont_w16", "G_cont_w64", "H_pipeline_local_w64", "F_pipeline_w64"]}
H1_DELAYS = [0, 4, 16]
# H2: real staleness error vs the same error with entries permuted, per optimizer.
H2_ARMS, H2_DELAY, H2_CONDITIONS = ["A_cont_w1", "B_cont_w4"], 16, ["fresh", "stale", "permuted"]


class Warmup:
    """Linear warmup over WARMUP_EPOCHS, then constant, for any optimizer."""

    def __init__(self, groups, steps_per_epoch):
        self.groups, self.base = groups, [g["lr"] for g in groups]
        self.total, self.t = WARMUP_EPOCHS * steps_per_epoch, 0
        self.apply()

    def apply(self):
        for g, base in zip(self.groups, self.base):
            g["lr"] = base * min(1.0, self.t / self.total)

    def step(self):
        self.t += 1
        self.apply()


class MuonWithAdam:
    """Muon (no weight decay) for every weight matrix; Adam for biases."""

    def __init__(self, model, lr):
        matrices = [p for p in model.parameters() if p.ndim == 2]
        vectors = [p for p in model.parameters() if p.ndim != 2]
        self.muon = Muon(matrices, lr=lr, weight_decay=0.0)
        self.adam = torch.optim.Adam(vectors, lr=PEAK_LR)
        self.param_groups = self.muon.param_groups + self.adam.param_groups

    def step(self):
        self.muon.step()
        self.adam.step()


def set_optimizer(trainer, name, lr):
    params = trainer.net.parameters()
    if name == "adam":
        trainer.opt = torch.optim.Adam(params, lr=lr)
    elif name == "sgd":
        trainer.opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    else:
        trainer.opt = MuonWithAdam(trainer.net, lr)
    trainer.sched = Warmup(trainer.opt.param_groups, trainer.steps_per_epoch)
    return trainer


def build(task, arm, seed, condition, delay, optimizer, lr):
    data, hidden = (automaton_dataset(seed), AUTOMATON_HIDDEN) if task == "automaton" else (adder_dataset(), WIDTHS)
    cls = PermutedErrorTrainer if condition == "permuted" else Trainer
    trainer = cls(arm, seed, 0 if condition == "fresh" else delay, 0, data, hidden)
    return set_optimizer(trainer, optimizer, lr)


def run_job(job):
    family, task, arm, seed, condition, delay, optimizer, lr = job
    torch.set_num_threads(1)
    started = time.time()
    trainer = build(task, arm, seed, condition, delay, optimizer, lr)
    curve, fit_epoch = [], None
    for epoch in range(1, MAX_EPOCHS + 1):
        order = torch.randperm(len(trainer.inputs), generator=trainer.order)
        for i in range(0, len(order), BATCH):
            trainer.step(order[i:i + BATCH])
        curve.append(trainer.accuracy())
        if not np.isfinite(curve[-1]):
            break
        if curve[-1] == 1.0:
            fit_epoch = epoch
            break
    return {"family": family, "task": task, "arm": arm, "seed": seed, "condition": condition, "delay": delay,
            "optimizer": optimizer, "lr": lr, "fit_epoch": fit_epoch, "final_accuracy": curve[-1],
            "accuracy_curve": curve, "seconds": time.time() - started}


def time_to_fit(row):
    return row["fit_epoch"] if row["fit_epoch"] is not None else CENSOR


def paired_test(rows, better, worse, key):
    """One-sided Wilcoxon signed-rank: `better` fits sooner than `worse`, paired on key."""
    a = {key(r): time_to_fit(r) for r in rows if better(r)}
    b = {key(r): time_to_fit(r) for r in rows if worse(r)}
    pairs = [(a[k], b[k]) for k in sorted(a) if k in b]
    diffs = np.array([x - y for x, y in pairs], dtype=float)
    fits = (sum(x < CENSOR for x, _ in pairs), sum(y < CENSOR for _, y in pairs))
    nonzero = diffs[diffs != 0]
    p = float(wilcoxon(nonzero, alternative="less").pvalue) if len(nonzero) else 1.0
    log_ratio = float(np.median(np.log([x / y for x, y in pairs]))) if pairs else float("nan")
    return {"n_pairs": len(pairs), "fits_better": fits[0], "fits_worse": fits[1], "p_one_sided": p,
            "median_log_ratio": log_ratio, "wins": int((diffs < 0).sum()), "losses": int((diffs > 0).sum()),
            "supported": bool(p < 0.05 and log_ratio < 0)}


def analyze(rows):
    out = {"H1": {}, "H2": {}}
    h1 = [r for r in rows if r["family"] == "H1"]
    for task, arms in H1_ARMS.items():
        narrow, wide, local, pipe = arms
        for d in (4, 16):
            sel = lambda arm: (lambda r: r["task"] == task and r["arm"] == arm and r["delay"] == d)  # noqa: E731
            seed = lambda r: r["seed"]  # noqa: E731
            out["H1"][f"{task}_d{d}"] = {
                "H1a_chain_vs_pipeline": paired_test(h1, sel(wide), sel(pipe), seed),
                "H1b_narrow_vs_wide": paired_test(h1, sel(narrow), sel(wide), seed),
                "chain_vs_local_heads": paired_test(h1, sel(wide), sel(local), seed),
                "local_heads_vs_pipeline": paired_test(h1, sel(local), sel(pipe), seed),
            }
    h2 = [r for r in rows if r["family"] == "H2"]
    for opt in LEARNING_RATES:
        sel = lambda c: (lambda r: r["optimizer"] == opt and r["condition"] == c)  # noqa: E731
        unit = lambda r: (r["arm"], r["seed"])  # noqa: E731
        out["H2"][opt] = {"H2_permuted_vs_stale": paired_test(h2, sel("permuted"), sel("stale"), unit),
                          "stale_vs_fresh": paired_test(h2, sel("stale"), sel("fresh"), unit),
                          "permuted_vs_fresh": paired_test(h2, sel("permuted"), sel("fresh"), unit)}
    supported = [o for o, v in out["H2"].items() if v["H2_permuted_vs_stale"]["supported"]]
    out["H2_verdict"] = {"supported_optimizers": supported, "general": len(supported) >= 2}
    out["H1_verdict"] = {k: all(out["H1"][f"{t}_d16"][k]["supported"] for t in H1_ARMS)
                         for k in ("H1a_chain_vs_pipeline", "H1b_narrow_vs_wide")}
    return out


def calibration_jobs():
    return [("CAL", "adder", arm, seed, "fresh", 0, opt, lr)
            for opt, lrs in CALIBRATION_LRS.items() for lr in lrs for arm in H2_ARMS for seed in CALIBRATION_SEEDS]


def confirmatory_jobs():
    jobs = [("H1", task, arm, seed, "fresh" if d == 0 else "stale", d, "adam", PEAK_LR)
            for task, arms in H1_ARMS.items() for arm in arms for seed in SEEDS for d in H1_DELAYS]
    jobs += [("H2", "adder", arm, seed, c, H2_DELAY, opt, lr)
             for opt, lr in LEARNING_RATES.items() for arm in H2_ARMS for seed in SEEDS for c in H2_CONDITIONS]
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["calibrate", "run"])
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    if args.mode == "run" and None in LEARNING_RATES.values():
        raise SystemExit("set LEARNING_RATES from calibration and freeze the protocol first")
    jobs = calibration_jobs() if args.mode == "calibrate" else confirmatory_jobs()
    jobs.sort(key=lambda j: (j[4] == "fresh", -j[5]))  # slow (stale) jobs first
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(run_job, jobs):
            rows.append(row)
            print(f"{row['family']} {row['task']:>9} {row['arm']:>21} seed {row['seed']:>3} {row['condition']:>8} "
                  f"d {row['delay']:>2} {row['optimizer']:>4} lr {row['lr']:.4g}: fit {row['fit_epoch']} "
                  f"({row['seconds']:.0f}s)", flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"protocol": str(PROTOCOL.relative_to(ROOT)),
              "protocol_sha256": sha256(PROTOCOL) if PROTOCOL.exists() else None,
              "script_sha256": sha256(__file__), "git_commit": commit, "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "scipy": scipy.__version__},
              "settings": {"mode": args.mode, "learning_rates": LEARNING_RATES, "max_epochs": MAX_EPOCHS,
                           "censor": CENSOR}, "rows": rows}
    if args.mode == "run":
        report["analysis"] = analyze(rows)
        print(json.dumps(report["analysis"], indent=1))
    name = "interface_confirm_calibration.json" if args.mode == "calibrate" else "interface_confirm_v1.json"
    (ROOT / "research/results" / name).write_text(json.dumps(report, indent=1))
    print(f"Wrote research/results/{name}")


if __name__ == "__main__":
    main()
