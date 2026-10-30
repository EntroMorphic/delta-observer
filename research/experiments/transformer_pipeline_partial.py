#!/usr/bin/env python3
"""Assemble the stopped transformer_pipeline grid (see the protocol's amendment).

Runs the 9 synchronous references with transformer_pipeline.job and combines
them with the 41 rows recovered from the stopped run's log.

    python -m research.experiments.transformer_pipeline_partial logs/transformer_pipeline_partial.log
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from multiprocessing import get_context
from pathlib import Path

import research.experiments.transformer_pipeline as tp
from research.experiments.curvature_dynamics import sha256

ROOT = Path(__file__).resolve().parents[2]
BASELINES = [("fresh", 0, lr, s) for lr in (tp.LR, tp.SECONDARY_LR) for s in tp.SEEDS]
BASELINES += [("basis_rotation_fresh", 0, tp.LR, s) for s in tp.SEEDS]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log")
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    recovered = [ast.literal_eval(line) for line in open(args.log) if line.startswith("{")]
    started, rows = time.strftime("%Y-%m-%dT%H:%M:%S"), []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(tp.job, BASELINES):
            rows.append(row)
            print({k: v for k, v in row.items() if k != "val_curve"}, flush=True)
    report = {"status": "exploratory, partial (stopped at 41 of 114 runs; see protocol amendment)",
              "script_sha256": sha256(tp.__file__), "assembler_sha256": sha256(__file__), "started": started,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "recovered_from_log": len(recovered),
              "rows": recovered + rows}
    out = ROOT / "research/results/transformer_pipeline_v0.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
