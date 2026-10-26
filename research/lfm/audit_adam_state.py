#!/usr/bin/env python3
"""Measure the realized AdamW update field stored in an LM resume checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.lfm.audit_observer_d import exposure_counts, quantiles


def correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.corrcoef(torch.stack((first.float(), second.float())))[0, 1].item())


def matrix_metrics(
    matrix: torch.Tensor,
    log_frequency: torch.Tensor,
    categories: dict[str, torch.Tensor],
    reference_pc: torch.Tensor,
    collapse_component: torch.Tensor,
) -> dict:
    row_norm = matrix.norm(dim=-1)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / centered.shape[0]
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    fractions = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    pc_projection = matrix @ reference_pc
    result = {
        "frobenius_norm": float(matrix.norm().item()),
        "mean_row_vector_norm": float(matrix.mean(dim=0).norm().item()),
        "row_norm": quantiles(row_norm.numpy()),
        "row_norm_log_frequency_pearson": correlation(row_norm, log_frequency),
        "reference_pc_projection": quantiles(pc_projection.numpy()),
        "reference_pc_projection_log_frequency_pearson": correlation(
            pc_projection, log_frequency
        ),
        "effective_rank": float(
            torch.exp(-(fractions * fractions.clamp_min(1e-30).log()).sum()).item()
        ),
        "top_1_variance_fraction": float(fractions[0].item()),
        "collapse_component_cosine": float(
            (
                (matrix * collapse_component).sum()
                / (matrix.norm() * collapse_component.norm()).clamp_min(1e-30)
            ).item()
        ),
        "categories": {},
    }
    for name, mask in categories.items():
        result["categories"][name] = {
            "rows": int(mask.sum().item()),
            "row_norm": quantiles(row_norm[mask].numpy()),
            "reference_pc_projection": quantiles(pc_projection[mask].numpy()),
        }
    return result


def teacher_top_counts(directories: list[Path], vocab_size: int) -> torch.Tensor:
    counts = np.zeros(vocab_size, dtype=np.int64)
    for directory in directories:
        for shard in sorted(directory.glob("shard-*.npz")):
            with np.load(shard) as data:
                mask = (
                    data["loss_mask"].astype(bool)
                    if "loss_mask" in data
                    else np.ones_like(data["labels"], dtype=bool)
                )
                values = data["top_logit_indices"][mask].astype(np.int64)
                counts += np.bincount(values.reshape(-1), minlength=vocab_size)
    return torch.from_numpy(counts)


def audit(arguments: argparse.Namespace) -> dict:
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    optimizer = checkpoint["optimizer"]
    weight = checkpoint["model"]["token_embedding.weight"].float()
    state_key = next(
        key
        for key, state in optimizer["state"].items()
        if tuple(state["exp_avg"].shape) == tuple(weight.shape)
    )
    state = optimizer["state"][state_key]
    group = next(
        group for group in optimizer["param_groups"] if state_key in group["params"]
    )
    step = int(state["step"].item())
    beta1, beta2 = group["betas"]
    first_moment = state["exp_avg"].float()
    second_moment = state["exp_avg_sq"].float()
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    denominator = second_moment.sqrt() / bias_correction2**0.5 + group["eps"]
    normalized_moment = first_moment / denominator
    adaptive_delta = -(group["lr"] / bias_correction1) * normalized_moment
    decay_delta = -group["lr"] * group["weight_decay"] * weight
    total_delta = adaptive_delta + decay_delta

    directories = [Path(path) for path in arguments.training_data]
    label_counter = exposure_counts(directories)
    label_frequency = torch.tensor(
        [label_counter[index] for index in range(weight.shape[0])], dtype=torch.int64
    )
    top_frequency = teacher_top_counts(directories, weight.shape[0])
    log_frequency = torch.log1p(label_frequency.float())
    categories = {
        "zero_label_exposure": label_frequency == 0,
        "nonzero_label_exposure": label_frequency > 0,
        "zero_teacher_top_k_exposure": top_frequency == 0,
        "teacher_top_k_without_label": (top_frequency > 0) & (label_frequency == 0),
        "label_exposure_1_to_3": (label_frequency >= 1) & (label_frequency <= 3),
        "label_exposure_64_plus": label_frequency >= 64,
    }
    centered_weight = weight - weight.mean(dim=0, keepdim=True)
    covariance = centered_weight.T @ centered_weight / centered_weight.shape[0]
    _, eigenvectors = torch.linalg.eigh(covariance)
    first_pc = eigenvectors[:, -1]
    collapse_component = (
        (centered_weight @ first_pc).unsqueeze(1) * first_pc.unsqueeze(0)
    )
    result = {
        "checkpoint": arguments.checkpoint,
        "checkpoint_step": checkpoint["step"],
        "embedding_optimizer_state_key": int(state_key),
        "adam_step": step,
        "hyperparameters": {
            "learning_rate": group["lr"],
            "betas": list(group["betas"]),
            "epsilon": group["eps"],
            "weight_decay": group["weight_decay"],
            "bias_correction1": bias_correction1,
            "bias_correction2": bias_correction2,
        },
        "first_moment": matrix_metrics(
            first_moment, log_frequency, categories, first_pc, collapse_component
        ),
        "second_moment_sqrt": matrix_metrics(
            second_moment.sqrt(), log_frequency, categories, first_pc, collapse_component
        ),
        "normalized_moment": matrix_metrics(
            normalized_moment, log_frequency, categories, first_pc, collapse_component
        ),
        "adaptive_parameter_delta": matrix_metrics(
            adaptive_delta, log_frequency, categories, first_pc, collapse_component
        ),
        "weight_decay_parameter_delta": matrix_metrics(
            decay_delta, log_frequency, categories, first_pc, collapse_component
        ),
        "total_parameter_delta": matrix_metrics(
            total_delta, log_frequency, categories, first_pc, collapse_component
        ),
        "first_order_collapse_energy_change": float(
            (2.0 * (collapse_component * total_delta).sum()).item()
        ),
    }
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-data", nargs="+", required=True)
    parser.add_argument("--output", default="research/results/lfm_adam_state_audit.json")
    return parser.parse_args()


if __name__ == "__main__":
    result = audit(parse_args())
    print(json.dumps({
        "checkpoint_step": result["checkpoint_step"],
        "first_moment": result["first_moment"],
        "normalized_moment": result["normalized_moment"],
        "total_parameter_delta": result["total_parameter_delta"],
        "first_order_collapse_energy_change": result[
            "first_order_collapse_energy_change"
        ],
    }, indent=2))
