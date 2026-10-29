#!/usr/bin/env python3
"""Calibrated stress-burst floor predictor, tested on held-out settings (exploratory).

Rule fixed in advance (research/reports/calibrated_floor_protocol.md).

    calibrate  choose the loss-ratio threshold R on the eight stress_floor_v0
               settings: most within 2x, then lowest mean |log2((p+.5)/(m+.5))|,
               then smallest R. Must reproduce FROZEN_RATIO.
    floors     measure the floor on HELD_OUT (seeds 610-614, both H2 arms,
               stale only; floor = largest grid delay up to which >= 8 of 10 fit)
    stress     stress_floor's burst on HELD_OUT (seeds 600-602)
    evaluate   score the frozen predictor and the two baselines

    python -m research.experiments.calibrated_floor calibrate|floors|stress|evaluate
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
from scipy.stats import spearmanr

import research.experiments.staleness_floor as sf
from research.experiments.curvature_dynamics import sha256
from research.experiments.interface_confirm import H2_ARMS
from research.experiments.sharpness_floor import GRID, floor_job, measured_floor
from research.experiments.stress_floor import job as stress_job

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = (2, 4, 8, 16, 32, 64, 128, 256, 1000)
FROZEN_RATIO, CONSTANT = 32, 4  # CONSTANT: median calibration floor, the best constant guess in-sample
HELD_OUT = [("adam", 1 / 200), ("adam", 1 / 50), ("sgd", 0.05), ("sgd", 0.2),
            ("muon", 0.005), ("muon", 0.02), ("sgd0", 0.5), ("sgd0", 2.0)]
STRESS_SEEDS, FLOOR_SEEDS = [600, 601, 602], [610, 611, 612, 613, 614]
RESULTS = ROOT / "research/results"


def floor_from_ratios(loss_ratio, ratio):
    if loss_ratio is None:
        return 0
    f = 0
    for d in GRID:
        if not loss_ratio[str(d)] <= ratio:  # a NaN (diverged) branch fails
            break
        f = d
    return f


def within_2x(p, m):
    return bool((m and p and 0.5 <= p / m <= 2) or (m == 0 and p == 0))


def log_error(p, m):
    return abs(np.log2((p + 0.5) / (m + 0.5)))


def setting_predictions(rows, settings, ratio):
    out = {}
    for o, lr in settings:
        preds = [floor_from_ratios(r["loss_ratio"], ratio) for r in rows
                 if r["optimizer"] == o and abs(r["lr"] - lr) < 1e-12]
        out[(o, lr)] = float(np.median(preds))
    return out


def score(pred, meas):
    keys = list(meas)
    p, m = [pred[k] for k in keys], [meas[k] for k in keys]
    rho = spearmanr(p, m).correlation if len(set(p)) > 1 and len(set(m)) > 1 else float("nan")
    return {"within_2x": sum(within_2x(a, b) for a, b in zip(p, m)), "spearman": float(rho),
            "mean_log2_error": float(np.mean([log_error(a, b) for a, b in zip(p, m)]))}


def calibrate(rows, meas):
    settings = list(meas)
    table = []
    for r in CANDIDATES:
        s = score(setting_predictions(rows, settings, r), meas)
        table.append({"ratio": r, **s})
    best = sorted(table, key=lambda t: (-t["within_2x"], t["mean_log2_error"], t["ratio"]))[0]
    return best["ratio"], table


def snap(x):
    return 0 if x <= 0 else min(GRID, key=lambda d: abs(np.log2(d) - np.log2(x)))


def lr_scaling(o, lr, cal):
    """Nearest calibration setting of the same optimizer (log lr; floor 0 skipped), floor x lr_cal / lr."""
    same = [(c_lr, m) for (c_o, c_lr), m in cal.items() if c_o == o and m > 0]
    c_lr, m = min(same, key=lambda c: abs(np.log(c[0] / lr)))
    return float(snap(m * c_lr / lr))


def calibration_data():
    stress = json.load(open(RESULTS / "stress_floor_v0.json"))
    meas = {(e["optimizer"], e["lr"]): e["measured"] for e in stress["summary"]}
    return stress["rows"], meas


def evaluate():
    cal_rows, cal_meas = calibration_data()
    ratio, _ = calibrate(cal_rows, cal_meas)
    assert ratio == FROZEN_RATIO, (ratio, FROZEN_RATIO)
    floors = json.load(open(RESULTS / "calibrated_floor_floors_v0.json"))["rows"]
    stress = json.load(open(RESULTS / "calibrated_floor_stress_v0.json"))["rows"]
    meas = {(o, lr): measured_floor(floors, o, lr) for o, lr in HELD_OUT}
    preds = {"calibrated": setting_predictions(stress, HELD_OUT, FROZEN_RATIO),
             "constant": {k: float(CONSTANT) for k in HELD_OUT},
             "lr_scaling": {(o, lr): lr_scaling(o, lr, cal_meas) for o, lr in HELD_OUT}}
    scores = {name: score(p, meas) for name, p in preds.items()}
    c, k = scores["calibrated"], scores["constant"]
    verdict = "held" if c["within_2x"] > k["within_2x"] and c["spearman"] >= 0.5 else "killed"
    per_setting = [{"optimizer": o, "lr": lr, "measured": meas[(o, lr)],
                    **{name: p[(o, lr)] for name, p in preds.items()}} for o, lr in HELD_OUT]
    return {"frozen_ratio": FROZEN_RATIO, "per_setting": per_setting, "scores": scores, "verdict": verdict}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["calibrate", "floors", "stress", "evaluate"])
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    if args.mode in ("calibrate", "evaluate"):
        if args.mode == "calibrate":
            ratio, table = calibrate(*calibration_data())
            result = {"chosen_ratio": ratio, "frozen_ratio": FROZEN_RATIO, "table": table}
        else:
            result = evaluate()
        print(json.dumps(result, indent=1))
        (RESULTS / f"calibrated_floor_{args.mode}_v0.json").write_text(
            json.dumps({"status": "exploratory", "script_sha256": sha256(__file__), **result}, indent=1))
        return
    if args.mode == "floors":
        jobs = [(o, lr, arm, s, "stale", d) for o, lr in HELD_OUT for arm in H2_ARMS for s in FLOOR_SEEDS
                for d in GRID]
        jobs.sort(key=lambda j: -j[5])
        fn = run_floor
    else:
        jobs, fn = [(o, lr, s) for o, lr in HELD_OUT for s in STRESS_SEEDS], stress_job
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(fn, jobs):
            rows.append(row)
            print({k: v for k, v in row.items() if k != "loss_ratio"}, flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"status": "exploratory", "mode": args.mode, "script_sha256": sha256(__file__), "git_commit": commit,
              "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "settings": {"held_out": HELD_OUT, "stress_seeds": STRESS_SEEDS, "floor_seeds": FLOOR_SEEDS,
                           "grid": GRID}, "rows": rows}
    out = RESULTS / f"calibrated_floor_{args.mode}_v0.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


def run_floor(job):
    return floor_job(job) if job[0] == "sgd0" else sf.surface_job(job)


if __name__ == "__main__":
    main()
