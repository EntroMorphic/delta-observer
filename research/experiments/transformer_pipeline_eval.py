#!/usr/bin/env python3
"""Score transformer_pipeline_v0.json against research/reports/transformer_pipeline_protocol.md.

    python -m research.experiments.transformer_pipeline_eval
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.experiments.curvature_dynamics import sha256
from research.experiments.transformer_pipeline import DELAYS, HARM, LR, SECONDARY_LR

ROOT = Path(__file__).resolve().parents[2]
PRACTICAL = ["pipemare_lr", "weight_predict", "basis_rotation"]


def losses(rows, method, delay, lr):
    return {r["seed"]: r["final_val_loss"] for r in rows
            if r["method"] == method and r["delay"] == delay and abs(r["lr"] - lr) < 1e-12}


def excess(rows, method, delay, lr, reference="fresh"):
    m, f = losses(rows, method, delay, lr), losses(rows, reference, 0, lr)
    seeds = sorted(set(m) & set(f))
    return float(np.mean([m[s] - f[s] for s in seeds])) if seeds else float("nan")


def mean(rows, method, delay, lr):
    return float(np.mean(list(losses(rows, method, delay, lr).values())))


def floor(rows, lr):
    f = 0
    for d in DELAYS:
        if not excess(rows, "stale", d, lr) < HARM:
            break
        f = d
    return f


def evaluate(rows):
    table = {}
    for lr, methods in ((LR, ["stale", "permuted"] + PRACTICAL), (SECONDARY_LR, ["stale", "permuted"])):
        table[str(lr)] = {m: {str(d): excess(rows, m, d, lr) for d in DELAYS} for m in methods}
    table[str(LR)]["basis_rotation_own"] = {str(d): excess(rows, "basis_rotation", d, LR, "basis_rotation_fresh")
                                            for d in DELAYS}
    harmful = [d for d in DELAYS if table[str(LR)]["stale"][str(d)] >= HARM]
    p1 = {str(d): table[str(LR)]["permuted"][str(d)] <= table[str(LR)]["stale"][str(d)] / 3 for d in harmful}
    recovery = {}
    for d in DELAYS:
        s, p = mean(rows, "stale", d, LR), mean(rows, "permuted", d, LR)
        recovery[str(d)] = {m: (s - mean(rows, m, d, LR)) / (s - p) if s != p else float("nan") for m in PRACTICAL}
    top = str(max(harmful)) if harmful else None
    floors = {str(LR): floor(rows, LR), str(SECONDARY_LR): floor(rows, SECONDARY_LR)}
    a, b = floors[str(LR)], floors[str(SECONDARY_LR)]
    p3 = "untestable" if a == b and a in (0, max(DELAYS)) else ("held" if b < a else "killed")
    return {
        "excess": table, "harmful_delays": harmful, "recovery": recovery, "floors": floors,
        "fresh": {str(LR): losses(rows, "fresh", 0, LR), str(SECONDARY_LR): losses(rows, "fresh", 0, SECONDARY_LR),
                  "basis_rotation_fresh": losses(rows, "basis_rotation_fresh", 0, LR)},
        "verdicts": {
            "P1": "untestable" if not harmful else ("held" if all(p1.values()) else "killed"),
            "P1_by_delay": p1,
            "P2": "untestable" if top is None else
                  ("held" if all(v < 0.5 for v in recovery[top].values()) else "killed"),
            "P2_delay": top, "P3": p3}}


def main():
    data = json.load(open(ROOT / "research/results/transformer_pipeline_v0.json"))
    result = evaluate(data["rows"])
    print(json.dumps(result, indent=1))
    out = ROOT / "research/results/transformer_pipeline_eval_v0.json"
    out.write_text(json.dumps({"status": "exploratory", "script_sha256": sha256(__file__),
                               "source_sha256": data["script_sha256"], **result}, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
