#!/usr/bin/env python3
"""Measure how CE and sparse distillation gradients reshape the tied table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from research.lfm.audit_observer_d import exposure_counts, quantiles
from research.lfm.audit_output_head import hidden_states
from research.lfm.distillation import sparse_topk_kl
from research.lfm.observer_d_lm import ObserverDLM, ObserverDLMConfig
from research.lfm.probe_lfm import choose_device
from research.lfm.train_observer_d_lm import TrajectoryShards


def correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.corrcoef(torch.stack((first.float(), second.float())))[0, 1].item())


def gradient_for(
    model: ObserverDLM,
    batch: dict,
    device: str,
    objective: str,
    path: str,
) -> tuple[float, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    detach_input = path == "output_only"
    detach_output = path == "input_only"
    inputs = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    mask = batch["loss_mask"].to(device)
    states = hidden_states(model, inputs, detach_input_weight=detach_input)
    weight = model.output_weight.detach() if detach_output else model.output_weight
    logits = F.linear(states, weight)
    if objective == "ce":
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
        ).reshape_as(labels)
        loss = (losses * mask).sum() / mask.sum()
    elif objective == "kl":
        loss = sparse_topk_kl(
            logits,
            batch["top_indices"].to(device),
            batch["top_values"].to(device),
            batch["logsumexp"].to(device),
            mask=mask,
        )
    else:
        raise ValueError(objective)
    loss.backward()
    gradient = model.token_embedding.weight.grad
    if gradient is None:
        gradient = torch.zeros_like(model.token_embedding.weight)
    result = gradient.detach().float().cpu().clone()
    model.zero_grad(set_to_none=True)
    return float(loss.item()), result


def gradient_metrics(
    gradient: torch.Tensor,
    weight: torch.Tensor,
    first_pc: torch.Tensor,
    collapse_component: torch.Tensor,
    log_frequency: torch.Tensor,
    categories: dict[str, torch.Tensor],
) -> dict:
    row_norm = gradient.norm(dim=-1)
    pc_gradient = gradient @ first_pc
    total_squared = gradient.square().sum()
    collapse_dot = (gradient * collapse_component).sum()
    metrics = {
        "l2_norm": float(total_squared.sqrt().item()),
        "nonzero_rows": int((row_norm > 0).sum().item()),
        "row_norm": quantiles(row_norm.numpy()),
        "row_norm_log_frequency_pearson": correlation(row_norm, log_frequency),
        "pc1_gradient": quantiles(pc_gradient.numpy()),
        "pc1_gradient_log_frequency_pearson": correlation(pc_gradient, log_frequency),
        "pc1_direction_gradient_energy_fraction": float(
            (pc_gradient.square().sum() / total_squared.clamp_min(1e-30)).item()
        ),
        "gradient_collapse_component_cosine": float(
            (
                collapse_dot
                / (total_squared.sqrt() * collapse_component.norm()).clamp_min(1e-30)
            ).item()
        ),
        "first_order_collapse_energy_change_per_unit_learning_rate": float(
            (-2.0 * collapse_dot).item()
        ),
        "categories": {},
    }
    for name, category_mask in categories.items():
        metrics["categories"][name] = {
            "rows": int(category_mask.sum().item()),
            "row_norm": quantiles(row_norm[category_mask].numpy()),
            "pc1_gradient": quantiles(pc_gradient[category_mask].numpy()),
        }
    return metrics


def audit_checkpoint(
    checkpoint_path: Path,
    batch: dict,
    frequency: torch.Tensor,
    device: str,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ObserverDLM(ObserverDLMConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    if model.config.output_head_rank or model.config.output_bias:
        raise ValueError("This audit requires the original tied, bias-free head")
    model.to(device).eval()
    weight = model.token_embedding.weight.detach().float().cpu()
    centered = weight - weight.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / centered.shape[0]
    _, eigenvectors = torch.linalg.eigh(covariance)
    first_pc = eigenvectors[:, -1]
    collapse_component = (centered @ first_pc).unsqueeze(1) * first_pc.unsqueeze(0)
    log_frequency = torch.log1p(frequency.float())

    input_rows = torch.zeros(model.config.vocab_size, dtype=torch.bool)
    input_rows[torch.unique(batch["input_ids"])] = True
    label_rows = torch.zeros_like(input_rows)
    label_rows[torch.unique(batch["labels"][batch["loss_mask"].bool()])] = True
    teacher_top_rows = torch.zeros_like(input_rows)
    teacher_top_rows[torch.unique(batch["top_indices"][batch["loss_mask"].bool()])] = True
    categories = {
        "input_rows": input_rows,
        "label_rows": label_rows,
        "teacher_top_k_rows": teacher_top_rows,
        "outside_teacher_top_k": ~teacher_top_rows,
        "zero_corpus_exposure": frequency == 0,
        "nonzero_corpus_exposure": frequency > 0,
    }

    result = {
        "checkpoint": str(checkpoint_path),
        "batch_sequences": int(batch["input_ids"].shape[0]),
        "batch_supervised_tokens": int(batch["loss_mask"].sum().item()),
        "category_overlaps": {
            "input_and_label_rows": int((input_rows & label_rows).sum().item()),
            "label_and_teacher_top_k_rows": int((label_rows & teacher_top_rows).sum().item()),
            "input_and_teacher_top_k_rows": int((input_rows & teacher_top_rows).sum().item()),
        },
        "objectives": {},
    }
    retained: dict[str, torch.Tensor] = {}
    for objective in ("ce", "kl"):
        result["objectives"][objective] = {}
        for path in ("input_only", "output_only", "tied_both"):
            loss, gradient = gradient_for(model, batch, device, objective, path)
            result["objectives"][objective][path] = {
                "loss": loss,
                **gradient_metrics(
                    gradient,
                    weight,
                    first_pc,
                    collapse_component,
                    log_frequency,
                    categories,
                ),
            }
            if path == "tied_both":
                retained[objective] = gradient
    ce_gradient = retained["ce"]
    kl_gradient = retained["kl"]
    result["combined_tied"] = {
        "ce_kl_gradient_cosine": float(
            F.cosine_similarity(ce_gradient.reshape(1, -1), kl_gradient.reshape(1, -1)).item()
        ),
        **gradient_metrics(
            ce_gradient + kl_gradient,
            weight,
            first_pc,
            collapse_component,
            log_frequency,
            categories,
        ),
    }
    return result


def audit(arguments: argparse.Namespace) -> dict:
    device = choose_device(arguments.device)
    dataset = TrajectoryShards(Path(arguments.validation_data), shuffle=False)
    batch = next(iter(DataLoader(dataset, batch_size=arguments.batch_size, num_workers=0)))
    counts = exposure_counts([Path(path) for path in arguments.training_data])
    frequency = torch.tensor(
        [counts[index] for index in range(65_536)], dtype=torch.int64
    )
    result = {
        "device": device,
        "validation_data": arguments.validation_data,
        "checkpoints": {
            Path(path).stem + f"_{index}": audit_checkpoint(
                Path(path), batch, frequency, device
            )
            for index, path in enumerate(arguments.checkpoint)
        },
    }
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--training-data", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output", default="research/results/lfm_loss_gradient_audit.json")
    return parser.parse_args()


if __name__ == "__main__":
    result = audit(parse_args())
    summary = {}
    for name, checkpoint in result["checkpoints"].items():
        summary[name] = {
            "ce_kl_gradient_cosine": checkpoint["combined_tied"]["ce_kl_gradient_cosine"],
            "combined_collapse_energy_change": checkpoint["combined_tied"][
                "first_order_collapse_energy_change_per_unit_learning_rate"
            ],
        }
    print(json.dumps(summary, indent=2))
