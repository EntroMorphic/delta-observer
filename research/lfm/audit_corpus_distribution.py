#!/usr/bin/env python3
"""Measure exact input, label, and teacher-top-k coverage and concentration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from research.lfm.audit_observer_d import quantiles


def gini(values: np.ndarray) -> float:
    sorted_values = np.sort(values.astype(np.float64))
    total = sorted_values.sum()
    if total == 0:
        return 0.0
    indices = np.arange(1, len(sorted_values) + 1, dtype=np.float64)
    return float(
        (2.0 * (indices * sorted_values).sum()) / (len(sorted_values) * total)
        - (len(sorted_values) + 1.0) / len(sorted_values)
    )


def count_metrics(counts: np.ndarray) -> dict:
    total = int(counts.sum())
    probabilities = counts[counts > 0].astype(np.float64) / max(1, total)
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    descending = np.sort(counts)[::-1]
    return {
        "total": total,
        "nonzero_tokens": int((counts > 0).sum()),
        "zero_tokens": int((counts == 0).sum()),
        "count_all_vocabulary": quantiles(counts.astype(np.float64)),
        "count_seen_vocabulary": quantiles(counts[counts > 0].astype(np.float64)),
        "gini": gini(counts),
        "shannon_entropy_nats": entropy,
        "effective_support": float(math.exp(entropy)),
        "top_1_share": float(descending[:1].sum() / max(1, total)),
        "top_10_share": float(descending[:10].sum() / max(1, total)),
        "top_100_share": float(descending[:100].sum() / max(1, total)),
        "top_1000_share": float(descending[:1000].sum() / max(1, total)),
    }


def scan_directory(directory: Path, vocab_size: int) -> dict:
    inputs = np.zeros(vocab_size, dtype=np.int64)
    labels = np.zeros(vocab_size, dtype=np.int64)
    teacher_top = np.zeros(vocab_size, dtype=np.int64)
    sequence_count = 0
    token_slots = 0
    supervised_tokens = 0
    teacher_top1_label_matches = 0
    top_k_mass_sum = 0.0
    top_k_mass_square_sum = 0.0
    top_k_mass_min = float("inf")
    top_k_mass_max = float("-inf")
    for shard in sorted(directory.glob("shard-*.npz")):
        with np.load(shard) as data:
            input_ids = data["input_ids"].astype(np.int64)
            label_ids = data["labels"].astype(np.int64)
            mask = (
                data["loss_mask"].astype(bool)
                if "loss_mask" in data
                else np.ones_like(label_ids, dtype=bool)
            )
            top_indices = data["top_logit_indices"].astype(np.int64)
            selected_labels = label_ids[mask]
            selected_top = top_indices[mask]
            inputs += np.bincount(input_ids.reshape(-1), minlength=vocab_size)
            labels += np.bincount(selected_labels, minlength=vocab_size)
            teacher_top += np.bincount(selected_top.reshape(-1), minlength=vocab_size)
            teacher_top1_label_matches += int((selected_top[:, 0] == selected_labels).sum())
            sequence_count += input_ids.shape[0]
            token_slots += input_ids.size
            supervised_tokens += selected_labels.size
            top_values = data["top_logit_values"].astype(np.float64)[mask]
            logsumexp = data["teacher_logsumexp"].astype(np.float64)[mask]
            mass = np.exp(top_values - logsumexp[:, None]).sum(axis=-1)
            top_k_mass_sum += float(mass.sum())
            top_k_mass_square_sum += float(np.square(mass).sum())
            top_k_mass_min = min(top_k_mass_min, float(mass.min()))
            top_k_mass_max = max(top_k_mass_max, float(mass.max()))
    mean_mass = top_k_mass_sum / supervised_tokens
    return {
        "directory": str(directory),
        "sequences": sequence_count,
        "token_slots": token_slots,
        "supervised_tokens": supervised_tokens,
        "supervised_fraction": supervised_tokens / token_slots,
        "input": count_metrics(inputs),
        "labels": count_metrics(labels),
        "teacher_top_k": count_metrics(teacher_top),
        "coverage_overlap": {
            "input_and_label": int(((inputs > 0) & (labels > 0)).sum()),
            "input_not_label": int(((inputs > 0) & (labels == 0)).sum()),
            "label_not_input": int(((labels > 0) & (inputs == 0)).sum()),
            "teacher_top_k_not_label": int(((teacher_top > 0) & (labels == 0)).sum()),
            "outside_teacher_top_k": int((teacher_top == 0).sum()),
        },
        "teacher_top1_label_match": teacher_top1_label_matches / supervised_tokens,
        "teacher_top_k_probability_mass": {
            "mean": mean_mass,
            "std": math.sqrt(max(0.0, top_k_mass_square_sum / supervised_tokens - mean_mass**2)),
            "min": top_k_mass_min,
            "max": top_k_mass_max,
        },
        "_counts": {"input": inputs, "labels": labels, "teacher_top_k": teacher_top},
    }


def audit(arguments: argparse.Namespace) -> dict:
    records = [
        scan_directory(Path(path), arguments.vocab_size) for path in arguments.data
    ]
    combined_counts = {
        name: sum((record["_counts"][name] for record in records), np.zeros(arguments.vocab_size, dtype=np.int64))
        for name in ("input", "labels", "teacher_top_k")
    }
    for record in records:
        record.pop("_counts")
    inputs = combined_counts["input"]
    labels = combined_counts["labels"]
    teacher_top = combined_counts["teacher_top_k"]
    result = {
        "directories": records,
        "combined": {
            "input": count_metrics(inputs),
            "labels": count_metrics(labels),
            "teacher_top_k": count_metrics(teacher_top),
            "coverage_overlap": {
                "input_and_label": int(((inputs > 0) & (labels > 0)).sum()),
                "input_not_label": int(((inputs > 0) & (labels == 0)).sum()),
                "label_not_input": int(((labels > 0) & (inputs == 0)).sum()),
                "teacher_top_k_not_label": int(((teacher_top > 0) & (labels == 0)).sum()),
                "outside_teacher_top_k": int((teacher_top == 0).sum()),
            },
        },
    }
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--vocab-size", type=int, default=65_536)
    parser.add_argument("--output", default="research/results/lfm_corpus_distribution_audit.json")
    return parser.parse_args()


if __name__ == "__main__":
    result = audit(parse_args())
    print(json.dumps(result["combined"], indent=2))
