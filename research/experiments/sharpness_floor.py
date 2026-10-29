#!/usr/bin/env python3
"""Does sharpness predict the staleness floor? (exploratory F1, journal/findings_synth.md)

    sharpness  fresh runs (seeds 500-504, adder chain width 4): every 32 steps,
               the optimizer's effective curvature a by power iteration
    floors     the two momentum-free SGD settings' floor surface (seeds 400-404,
               same harness as staleness_floor.py) for the momentum check
    predict    combine: predicted floor = largest grid delay whose
               delayed-recurrence boundary exceeds the peak a

Mapping to x_{t+1} - x_t = beta (x_t - x_{t-1}) - a x_{t-d}, fixed in advance:
    SGD(beta)  a = lr * lambda_max(H)
    Adam       a = lr * (1 - beta1) * lambda_max(P^1/2 H P^1/2), P = 1/(sqrt(v_hat)+eps), beta = beta1
    Muon       a = (|dtheta| / |g|) * lambda_max(H), beta = 0 (beta = 0.95 also reported)

    python -m research.experiments.sharpness_floor sharpness|floors|predict
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

import research.experiments.staleness_floor as sf
from research.experiments.curvature_dynamics import sha256
from research.experiments.interface_confirm import build
from research.experiments.interface_sweep import BATCH

ROOT = Path(__file__).resolve().parents[2]
SEEDS, ARM, EVERY, MAX_EPOCHS, POWER_ITERS = [500, 501, 502, 503, 504], "B_cont_w4", 32, 30, 30
SETTINGS = sf.SETTINGS + [("sgd0", 1.0), ("sgd0", 3.0)]
MOMENTUM = {"sgd": 0.9, "sgd0": 0.0, "adam": 0.9, "muon": 0.0}
GRID = sf.DELAYS


def make(optimizer, lr, seed):
    trainer = build("adder", ARM, seed, "fresh", 0, "sgd" if optimizer == "sgd0" else optimizer, lr)
    if optimizer == "sgd0":
        for g in trainer.opt.param_groups:
            g["momentum"] = 0.0
    return trainer


def top_eigenvalue(trainer, precondition=None, iters=POWER_ITERS):
    """Largest eigenvalue of S H S (S = diag(precondition) or identity), full batch."""
    params = list(trainer.net.parameters())
    loss = trainer.full_loss(trainer.live(), trainer.inputs, trainer.outputs)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat = torch.cat([g.reshape(-1) for g in grads])
    s = torch.ones_like(flat) if precondition is None else precondition
    v = torch.randn(flat.shape, generator=torch.Generator().manual_seed(0))
    v, value = v / v.norm(), 0.0
    for _ in range(iters):
        hv = torch.cat([h.reshape(-1) for h in torch.autograd.grad(flat @ (s * v), params, retain_graph=True)])
        w = s * hv
        value = float(v @ w)
        v = w / w.norm().clamp_min(1e-30)
    return value, float(torch.cat([g.reshape(-1) for g in grads]).detach().norm())


def adam_precondition(trainer):
    opt, out = trainer.opt, []
    for p in trainer.net.parameters():
        st = opt.state.get(p, {})
        if "exp_avg_sq" not in st:
            return None
        b2 = opt.param_groups[0]["betas"][1]
        v_hat = st["exp_avg_sq"] / (1 - b2 ** int(st["step"]))
        out.append((1.0 / (v_hat.sqrt() + opt.param_groups[0]["eps"])).sqrt().reshape(-1))
    return torch.cat(out)


def sharpness_job(job):
    optimizer, lr, seed = job
    torch.set_num_threads(1)
    trainer = make(optimizer, lr, seed)
    params, records, step, fit = list(trainer.net.parameters()), [], 0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        order = torch.randperm(len(trainer.inputs), generator=trainer.order)
        for i in range(0, len(order), BATCH):
            before = [p.detach().clone() for p in params]
            trainer.step(order[i:i + BATCH])
            step += 1
            if step % EVERY:
                continue
            lr_now = trainer.opt.param_groups[0]["lr"]
            if optimizer == "adam":
                pre = adam_precondition(trainer)
                lam, gnorm = top_eigenvalue(trainer, pre)
                a = lr_now * (1 - 0.9) * lam
            else:
                lam, gnorm = top_eigenvalue(trainer)
                if optimizer == "muon":
                    moved = float(torch.cat([(p.detach() - b).reshape(-1) for p, b in zip(params, before)]).norm())
                    a = moved / max(gnorm, 1e-30) * lam
                else:
                    a = lr_now * lam
            records.append({"step": step, "epoch": step / trainer.steps_per_epoch, "lambda": lam, "a": a,
                            "lr_now": lr_now, "accuracy": trainer.accuracy()})
        if trainer.accuracy() == 1.0:
            fit = epoch
            break
    return {"optimizer": optimizer, "lr": lr, "seed": seed, "fit_epoch": fit, "records": records}


def boundary(d, beta):
    """Largest a for which the delayed heavy-ball recurrence is stable."""
    def stable(a):
        c = np.zeros(d + 2)
        c[0], c[1] = 1, -(1 + beta)
        c[2] += beta
        c[-1] += a
        return np.max(np.abs(np.roots(c))) < 1
    lo, hi = 1e-8, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if stable(mid) else (lo, mid)
    return lo


def predicted_floor(a_peak, beta):
    ok = [d for d in GRID if a_peak < boundary(d, beta)]
    return max(ok) if ok else 0


def measured_floor(rows, optimizer, lr):
    f = 0
    for d in GRID:
        fits = [r["fit_epoch"] for r in rows if r["optimizer"] == optimizer and abs(r["lr"] - lr) < 1e-12
                and r["condition"] == "stale" and r["delay"] == d]
        if sum(x is not None for x in fits) >= 8:
            f = d
        else:
            break
    return f


def predict():
    sharp = json.load(open(ROOT / "research/results/sharpness_floor_sharpness_v0.json"))["rows"]
    floors = json.load(open(ROOT / "research/results/staleness_floor_surface_v0.json"))["rows"]
    floors += json.load(open(ROOT / "research/results/sharpness_floor_floors_v0.json"))["rows"]
    out = []
    for optimizer, lr in SETTINGS:
        runs = [r for r in sharp if r["optimizer"] == optimizer and abs(r["lr"] - lr) < 1e-12]
        peaks = [max(x["a"] for x in r["records"]) for r in runs if r["records"]]
        a_peak = float(np.median(peaks))
        beta = MOMENTUM[optimizer]
        row = {"optimizer": optimizer, "lr": lr, "a_peak_median": a_peak, "a_peak_range": [min(peaks), max(peaks)],
               "beta": beta, "predicted_floor": predicted_floor(a_peak, beta),
               "measured_floor": measured_floor(floors, optimizer, lr)}
        if optimizer == "muon":
            row["predicted_floor_beta_0.95"] = predicted_floor(a_peak, 0.95)
        p, m = row["predicted_floor"], row["measured_floor"]
        row["within_2x"] = bool(p and m and 0.5 <= p / m <= 2)
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["sharpness", "floors", "predict"])
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    if args.mode == "predict":
        result = predict()
        for r in result:
            print(r)
        out = ROOT / "research/results/sharpness_floor_predict_v0.json"
        out.write_text(json.dumps({"status": "exploratory", "script_sha256": sha256(__file__), "rows": result},
                                  indent=1))
        return
    if args.mode == "sharpness":
        jobs, fn = [(o, lr, s) for o, lr in SETTINGS for s in SEEDS], sharpness_job
    else:
        jobs = [("sgd0", lr, arm, s, c, d) for lr in (1.0, 3.0) for arm in sf.H2_ARMS for s in sf.SEEDS
                for c in ("stale", "permuted") for d in GRID]
        jobs += [("sgd0", lr, arm, s, "fresh", 0) for lr in (1.0, 3.0) for arm in sf.H2_ARMS for s in sf.SEEDS]
        jobs.sort(key=lambda j: -j[5])
        fn = floor_job
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(fn, jobs):
            rows.append(row)
            print({k: v for k, v in row.items() if k != "records"}, flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"status": "exploratory", "mode": args.mode, "script_sha256": sha256(__file__), "git_commit": commit,
              "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "settings": {"seeds": SEEDS, "arm": ARM, "every": EVERY, "settings": SETTINGS, "momentum": MOMENTUM},
              "rows": rows}
    out = ROOT / f"research/results/sharpness_floor_{args.mode}_v0.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


def floor_job(job):
    optimizer, lr, arm, seed, condition, delay = job
    original = sf.build

    def patched(task, arm_, seed_, condition_, delay_, optimizer_, lr_):
        trainer = original(task, arm_, seed_, condition_, delay_, "sgd", lr_)
        for g in trainer.opt.param_groups:
            g["momentum"] = 0.0
        return trainer
    sf.build = patched
    try:
        row = sf.surface_job(("sgd", lr, arm, seed, condition, delay))
    finally:
        sf.build = original
    row["optimizer"] = optimizer
    return row


if __name__ == "__main__":
    main()
