#!/usr/bin/env python3
"""Measure tied-embedding frequency collapse and output-head ablations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from research.lfm.audit_observer_d import exposure_counts, quantiles
from research.lfm.observer_d_lm import ObserverDLM, ObserverDLMConfig
from research.lfm.probe_lfm import choose_device
from research.lfm.train_observer_d_lm import TrajectoryShards


def hidden_states(
    model: ObserverDLM, input_ids: torch.Tensor, detach_input_weight: bool = False
) -> torch.Tensor:
    weight = model.token_embedding.weight.detach() if detach_input_weight else model.token_embedding.weight
    states = F.embedding(input_ids, weight)
    if model.input_projection is not None:
        states = model.input_projection(states)
    states = states * math.sqrt(model.config.width)
    length = input_ids.shape[1]
    states = states + model.sinusoidal_positions(
        length, model.config.width, input_ids.device, states.dtype
    )[None]
    causal_mask = torch.ones(
        length, length, dtype=torch.bool, device=input_ids.device
    ).triu(1)
    for depth in range(model.config.repetitions):
        method_cell = model.method_cell_for_depth(depth)
        states = states + model.depth_embedding[depth]
        states = method_cell.local(states)
        if (depth + 1) % model.config.attention_every == 0:
            states = method_cell.global_attention(states, causal_mask)
        states = method_cell.feed_forward(states)
        states = model.apply_depth_adapter(states, depth)
    return model.final_norm(states)


def masked_ce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).reshape_as(labels)
    return (losses * mask).sum() / mask.sum()


def gradient_rows(
    model: ObserverDLM,
    batch: dict,
    device: str,
    detach_input: bool,
    detach_output: bool,
) -> dict:
    model.zero_grad(set_to_none=True)
    inputs = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    mask = batch["loss_mask"].to(device)
    states = hidden_states(model, inputs, detach_input_weight=detach_input)
    output_weight = (
        model.output_weight.detach() if detach_output else model.output_weight
    )
    loss = masked_ce(F.linear(states, output_weight), labels, mask)
    loss.backward()
    gradient = model.token_embedding.weight.grad
    if gradient is None:
        row_norm = torch.zeros(model.config.vocab_size)
    else:
        row_norm = gradient.detach().float().norm(dim=-1).cpu()
    input_tokens = torch.unique(inputs.cpu())
    input_mask = torch.zeros(model.config.vocab_size, dtype=torch.bool)
    input_mask[input_tokens] = True
    result = {
        "loss": float(loss.item()),
        "nonzero_rows": int((row_norm > 0).sum().item()),
        "input_token_rows": int(input_mask.sum().item()),
        "input_rows": quantiles(row_norm[input_mask].numpy()),
        "non_input_rows": quantiles(row_norm[~input_mask].numpy()),
    }
    model.zero_grad(set_to_none=True)
    return result


def audit(arguments: argparse.Namespace) -> dict:
    device = choose_device(arguments.device)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    model = ObserverDLM(ObserverDLMConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    directory = Path(arguments.validation_data)
    dataset = TrajectoryShards(directory, shuffle=False)
    loader = DataLoader(dataset, batch_size=arguments.batch_size, num_workers=0)
    counts = exposure_counts([Path(path) for path in arguments.training_data])
    frequency = torch.tensor(
        [counts[index] for index in range(model.config.vocab_size)], dtype=torch.float32
    )
    log_frequency = torch.log1p(frequency)
    log_frequency -= log_frequency.mean()

    weight_cpu = model.output_weight.detach().float().cpu()
    centered = weight_cpu - weight_cpu.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / centered.shape[0]
    _, eigenvectors = torch.linalg.eigh(covariance)
    first_pc = eigenvectors[:, -1]
    first_component = (centered @ first_pc).unsqueeze(1) @ first_pc.unsqueeze(0)
    output_weights = {
        str(alpha): (weight_cpu - alpha * first_component).to(device)
        for alpha in arguments.pc_removal
    }
    biases = {
        str(coefficient): (coefficient * log_frequency).to(device)
        for coefficient in arguments.frequency_bias
    }
    totals = {
        "baseline": 0.0,
        **{f"pc_remove_{key}": 0.0 for key in output_weights},
        **{f"frequency_bias_{key}": 0.0 for key in biases},
        **{
            f"pc_remove_{pc_key}_frequency_bias_{bias_key}": 0.0
            for pc_key in output_weights
            for bias_key in biases
        },
    }
    hidden_pc_values = []
    label_pc_logit_values = []
    supervised = 0
    first_batch = None
    sequences = 0
    with torch.inference_mode():
        for batch in loader:
            if sequences >= arguments.sequences:
                break
            take = min(arguments.sequences - sequences, batch["input_ids"].shape[0])
            batch = {key: value[:take] for key, value in batch.items()}
            if first_batch is None:
                first_batch = {key: value.clone() for key, value in batch.items()}
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["loss_mask"].to(device)
            states = hidden_states(model, inputs)
            baseline_logits = F.linear(states, model.output_weight)
            count = int(mask.sum().item())
            hidden_pc = states.float() @ first_pc.to(device)
            label_pc_score = (centered @ first_pc).to(device)[labels]
            hidden_pc_values.extend(hidden_pc[mask.bool()].cpu().tolist())
            label_pc_logit_values.extend(
                (hidden_pc * label_pc_score)[mask.bool()].cpu().tolist()
            )
            totals["baseline"] += float(masked_ce(baseline_logits, labels, mask).item()) * count
            for key, weight in output_weights.items():
                totals[f"pc_remove_{key}"] += float(
                    masked_ce(F.linear(states, weight), labels, mask).item()
                ) * count
            for key, bias in biases.items():
                totals[f"frequency_bias_{key}"] += float(
                    masked_ce(baseline_logits + bias, labels, mask).item()
                ) * count
            for pc_key, weight in output_weights.items():
                removed_logits = F.linear(states, weight)
                for bias_key, bias in biases.items():
                    totals[f"pc_remove_{pc_key}_frequency_bias_{bias_key}"] += float(
                        masked_ce(removed_logits + bias, labels, mask).item()
                    ) * count
            supervised += count
            sequences += take

    result = {
        "checkpoint": str(Path(arguments.checkpoint)),
        "sequences": sequences,
        "supervised_tokens": supervised,
        "cross_entropy_ablations": {
            key: value / supervised for key, value in totals.items()
        },
        "pc1_as_output_bias": {
            "hidden_projection": quantiles(np.asarray(hidden_pc_values)),
            "true_label_logit_contribution": quantiles(
                np.asarray(label_pc_logit_values)
            ),
            "pc1_score_log_frequency_regression_slope": float(
                (
                    ((log_frequency - log_frequency.mean()) * (centered @ first_pc)).sum()
                    / (log_frequency - log_frequency.mean()).square().sum().clamp_min(1e-30)
                ).item()
            ),
        },
        "frequency": {
            "total_exposures": int(frequency.sum().item()),
            "seen_tokens": int((frequency > 0).sum().item()),
            "never_seen_tokens": int((frequency == 0).sum().item()),
            "pc1_log_frequency_pearson": float(torch.corrcoef(torch.stack((
                centered @ first_pc, log_frequency
            )))[0, 1].item()),
            "norm_log_frequency_pearson": float(torch.corrcoef(torch.stack((
                weight_cpu.norm(dim=-1), log_frequency
            )))[0, 1].item()),
        },
        "gradient_decomposition": {
            "input_path_only": gradient_rows(
                model, first_batch, device, detach_input=False, detach_output=True
            ),
            "output_path_only": gradient_rows(
                model, first_batch, device, detach_input=True, detach_output=False
            ),
            "tied_both_paths": gradient_rows(
                model, first_batch, device, detach_input=False, detach_output=False
            ),
        },
    }
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--training-data", nargs="+", required=True)
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--pc-removal", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument(
        "--frequency-bias",
        type=float,
        nargs="+",
        default=[-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output", default="research/results/lfm_output_head_audit.json")
    return parser.parse_args()


if __name__ == "__main__":
    result = audit(parse_args())
    print(json.dumps(result, indent=2))
