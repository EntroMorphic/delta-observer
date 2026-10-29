#!/usr/bin/env python3
"""Interface confirmation v2: H1a (locality) and H2 (directional harm) at
delays chosen by a power check. Implements
research/reports/interface_confirm_v2_protocol.md.

    python -m research.experiments.interface_confirm_v2 calibrate   # seeds 310-314
    python -m research.experiments.interface_confirm_v2 run         # seeds 10-19
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

from research.experiments.curvature_dynamics import sha256
from research.experiments.interface_confirm import H1_ARMS, H2_ARMS, H2_CONDITIONS, LEARNING_RATES, paired_test, run_job

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "research/reports/interface_confirm_v2_protocol.md"
SEEDS, CALIBRATION_SEEDS = list(range(10, 20)), [310, 311, 312, 313, 314]
H1_CAL_DELAYS, H2_CAL_DELAYS = [2, 4, 8], [1, 2, 4, 8, 16]
H1_MIN_FITS, H2_MIN_FITS = 4, 8  # of 5 per task, of 10 per optimizer
# Filled in from calibration before freezing.
H1_DELAY = 4
H2_DELAYS = {"adam": 16, "sgd": 2, "muon": 4}


def calibration_jobs():
    jobs = [("CAL1", task, "G_cont_w64", seed, "stale", d, "adam", LEARNING_RATES["adam"])
            for task in H1_ARMS for d in H1_CAL_DELAYS for seed in CALIBRATION_SEEDS]
    jobs += [("CAL2", "adder", arm, seed, "stale", d, opt, lr) for opt, lr in LEARNING_RATES.items()
             for d in H2_CAL_DELAYS for arm in H2_ARMS for seed in CALIBRATION_SEEDS]
    return jobs


def choose_delays(rows):
    fits = lambda sel: sum(r["fit_epoch"] is not None for r in rows if sel(r))  # noqa: E731
    h1 = [d for d in H1_CAL_DELAYS
          if all(fits(lambda r: r["family"] == "CAL1" and r["task"] == t and r["delay"] == d) >= H1_MIN_FITS
                 for t in H1_ARMS)]
    h2 = {}
    for opt in LEARNING_RATES:
        ok = [d for d in H2_CAL_DELAYS
              if fits(lambda r: r["family"] == "CAL2" and r["optimizer"] == opt and r["delay"] == d) >= H2_MIN_FITS]
        h2[opt] = max(ok) if ok else None
    return {"H1_delay": max(h1) if h1 else None, "H2_delays": h2}


def confirmatory_jobs():
    _, wide, local, pipe = H1_ARMS["adder"]
    jobs = [("H1", task, arm, seed, "stale", H1_DELAY, "adam", LEARNING_RATES["adam"])
            for task in H1_ARMS for arm in (wide, local, pipe) for seed in SEEDS]
    jobs += [("H2", "adder", arm, seed, c, H2_DELAYS[opt], opt, lr) for opt, lr in LEARNING_RATES.items()
             if H2_DELAYS[opt] is not None for arm in H2_ARMS for seed in SEEDS for c in H2_CONDITIONS]
    return jobs


def analyze(rows):
    _, wide, local, pipe = H1_ARMS["adder"]
    seed, unit = (lambda r: r["seed"]), (lambda r: (r["arm"], r["seed"]))
    sel1 = lambda t, a: (lambda r: r["family"] == "H1" and r["task"] == t and r["arm"] == a)  # noqa: E731
    sel2 = lambda o, c: (lambda r: r["family"] == "H2" and r["optimizer"] == o and r["condition"] == c)  # noqa: E731
    out = {"H1": {t: {"H1a_chain_vs_pipeline": paired_test(rows, sel1(t, wide), sel1(t, pipe), seed),
                      "chain_vs_local_heads": paired_test(rows, sel1(t, wide), sel1(t, local), seed),
                      "local_heads_vs_pipeline": paired_test(rows, sel1(t, local), sel1(t, pipe), seed)}
                  for t in H1_ARMS},
           "H2": {o: {"H2_permuted_vs_stale": paired_test(rows, sel2(o, "permuted"), sel2(o, "stale"), unit),
                      "stale_vs_fresh": paired_test(rows, sel2(o, "stale"), sel2(o, "fresh"), unit),
                      "permuted_vs_fresh": paired_test(rows, sel2(o, "permuted"), sel2(o, "fresh"), unit)}
                  for o in LEARNING_RATES if H2_DELAYS[o] is not None}}
    out["H1a_confirmed"] = all(v["H1a_chain_vs_pipeline"]["supported"] for v in out["H1"].values())
    supported = [o for o, v in out["H2"].items() if v["H2_permuted_vs_stale"]["supported"]]
    out["H2_verdict"] = {"supported_optimizers": supported, "tested_optimizers": list(out["H2"]),
                         "general": len(supported) >= 2}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["calibrate", "run"])
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    if args.mode == "run" and H1_DELAY is None:
        raise SystemExit("set the delays from calibration and freeze the protocol first")
    jobs = calibration_jobs() if args.mode == "calibrate" else confirmatory_jobs()
    jobs.sort(key=lambda j: (j[4] == "fresh", -j[5]))
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(run_job, jobs):
            rows.append(row)
            print(f"{row['family']} {row['task']:>9} {row['arm']:>21} seed {row['seed']:>3} {row['condition']:>8} "
                  f"d {row['delay']:>2} {row['optimizer']:>4}: fit {row['fit_epoch']} ({row['seconds']:.0f}s)",
                  flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"protocol": str(PROTOCOL.relative_to(ROOT)),
              "protocol_sha256": sha256(PROTOCOL) if PROTOCOL.exists() else None,
              "script_sha256": sha256(__file__), "git_commit": commit, "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "scipy": scipy.__version__},
              "settings": {"mode": args.mode, "learning_rates": LEARNING_RATES, "h1_delay": H1_DELAY,
                           "h2_delays": H2_DELAYS}, "rows": rows}
    if args.mode == "calibrate":
        report["chosen"] = choose_delays(rows)
        print(json.dumps(report["chosen"]))
    else:
        report["analysis"] = analyze(rows)
        print(json.dumps(report["analysis"], indent=1))
    name = "interface_confirm_v2_calibration.json" if args.mode == "calibrate" else "interface_confirm_v2.json"
    (ROOT / "research/results" / name).write_text(json.dumps(report, indent=1))
    print(f"Wrote research/results/{name}")


if __name__ == "__main__":
    main()
