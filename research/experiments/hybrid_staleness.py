#!/usr/bin/env python3
"""Muon/Adam hybrids under staleness (exploratory).

    calibrate  fresh runs on seeds 300-302; pick each hybrid's lr by the rule
               used before (most fits of 6, ties by lower median fit epoch)
    surface    stale and fresh runs on seeds 400-404, delays 1-32, compared
               with the parents already in staleness_floor_surface_v0.json
    probe      the update-space probe (staleness_update_probe.py) on the hybrids

    python -m research.experiments.hybrid_staleness calibrate|surface|probe
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

import research.experiments.staleness_update_probe as probe
from research.experiments.curvature_dynamics import sha256
from research.experiments.interface_confirm import H2_ARMS, Warmup, build
from research.experiments.interface_sweep import BATCH, PEAK_LR
from research.optimizers.hybrid_muon import AdaMuonStyleMatrices, MatricesPlusAdam, PreconditionedMuonMatrices

ROOT = Path(__file__).resolve().parents[2]
HYBRIDS = {"muon_pre": PreconditionedMuonMatrices, "adamuon_style": AdaMuonStyleMatrices}
CAL_SEEDS, SEEDS = [300, 301, 302], [400, 401, 402, 403, 404]
CAL_LRS = [0.003, 0.01, 0.03]
DELAYS, MAX_EPOCHS = [1, 2, 4, 8, 16, 32], 120
LEARNING_RATES = {"muon_pre": 0.03, "adamuon_style": 0.03}  # chosen by calibration
SECONDARY_LR = 0.01  # matches Muon's best stale setting in the floor surface


def build_hybrid(arm, seed, condition, delay, name, lr):
    trainer = build("adder", arm, seed, condition, delay, "adam", PEAK_LR)
    trainer.opt = MatricesPlusAdam(trainer.net, HYBRIDS[name], lr, PEAK_LR)
    trainer.sched = Warmup(trainer.opt.param_groups, trainer.steps_per_epoch)
    return trainer


def train_job(job):
    family, name, lr, arm, seed, condition, delay = job
    torch.set_num_threads(1)
    started = time.time()
    trainer = build_hybrid(arm, seed, condition, delay, name, lr)
    fit_epoch, accuracy = None, 0.0
    for epoch in range(1, MAX_EPOCHS + 1):
        order = torch.randperm(len(trainer.inputs), generator=trainer.order)
        for i in range(0, len(order), BATCH):
            trainer.step(order[i:i + BATCH])
        accuracy = trainer.accuracy()
        if accuracy == 1.0:
            fit_epoch = epoch
            break
    return {"family": family, "optimizer": name, "lr": lr, "arm": arm, "seed": seed, "condition": condition,
            "delay": delay, "fit_epoch": fit_epoch, "final_accuracy": accuracy, "seconds": time.time() - started}


def probe_job(job):
    name, lr, delay, seed = job
    original = probe.build
    probe.build = lambda task, arm, seed_, condition, delay_, optimizer, lr_: build_hybrid(
        arm, seed_, condition, delay_, name, lr)
    try:
        row = probe.probe_job((name, lr, delay, seed))
    finally:
        probe.build = original
    return row


def choose(rows):
    chosen = {}
    for name in HYBRIDS:
        def score(lr):
            f = [r["fit_epoch"] for r in rows if r["optimizer"] == name and r["lr"] == lr]
            ok = [x for x in f if x is not None]
            return (-len(ok), np.median(ok) if ok else np.inf)
        chosen[name] = min(CAL_LRS, key=score)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["calibrate", "surface", "probe"])
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    if args.mode != "calibrate" and None in LEARNING_RATES.values():
        raise SystemExit("set LEARNING_RATES from calibration first")
    if args.mode == "calibrate":
        jobs = [("CAL", n, lr, a, s, "fresh", 0) for n in HYBRIDS for lr in CAL_LRS for a in H2_ARMS for s in CAL_SEEDS]
        fn = train_job
    elif args.mode == "surface":
        lrs = lambda n: (LEARNING_RATES[n], SECONDARY_LR)  # noqa: E731
        jobs = [("SURF", n, lr, a, s, "stale", d) for n in HYBRIDS for lr in lrs(n) for a in H2_ARMS for s in SEEDS
                for d in DELAYS]
        jobs += [("SURF", n, lr, a, s, "fresh", 0) for n in HYBRIDS for lr in lrs(n) for a in H2_ARMS for s in SEEDS]
        jobs.sort(key=lambda j: -j[6])
        fn = train_job
    else:
        jobs = [(n, LEARNING_RATES[n], d, s) for n in HYBRIDS for d in (4, 16) for s in probe.SEEDS]
        fn = probe_job
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(fn, jobs):
            rows.append(row)
            print({k: v for k, v in row.items() if k != "records"}, flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"status": "exploratory", "mode": args.mode, "script_sha256": sha256(__file__), "git_commit": commit,
              "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "settings": {"learning_rates": LEARNING_RATES, "delays": DELAYS, "max_epochs": MAX_EPOCHS}, "rows": rows}
    if args.mode == "calibrate":
        report["chosen"] = choose(rows)
        print(report["chosen"])
    out = ROOT / f"research/results/hybrid_staleness_{args.mode}_v0.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
