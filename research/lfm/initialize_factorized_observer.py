#!/usr/bin/env python3
"""Factorize the vocabulary interface and replicate the learned method cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.lfm.observer_d_lm import ObserverDLM, ObserverDLMConfig


def initialize(arguments: argparse.Namespace) -> dict:
    source = torch.load(arguments.base_checkpoint, map_location="cpu", weights_only=False)
    source_config = ObserverDLMConfig(**source["config"])
    if source_config.input_embedding_rank or source_config.output_head_rank:
        raise ValueError("Base checkpoint must use the original dense tied interface")
    config = ObserverDLMConfig(**{
        **source_config.__dict__,
        "input_embedding_rank": arguments.input_embedding_rank,
        "output_head_rank": arguments.output_head_rank,
        "method_cell_count": arguments.method_cell_count,
        "attention_residual_scale": arguments.attention_residual_scale,
        "ffn_residual_scale": arguments.ffn_residual_scale,
    })
    model = ObserverDLM(config)
    with torch.no_grad():
        model.depth_embedding.copy_(source["model"]["depth_embedding"])
        model.final_norm.weight.copy_(source["model"]["final_norm.weight"])
    method_state = {
        key.removeprefix("method_cell."): value
        for key, value in source["model"].items()
        if key.startswith("method_cell.")
    }
    model.method_cell.load_state_dict(method_state)
    for method_cell in model.additional_method_cells:
        method_cell.load_state_dict(method_state)

    source_weight = source["model"]["token_embedding.weight"].float()
    left, singular_values, right = torch.linalg.svd(source_weight, full_matrices=False)

    def factors(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        root = singular_values[:rank].sqrt()
        return left[:, :rank] * root, root[:, None] * right[:rank]

    input_table, input_projection_transposed = factors(arguments.input_embedding_rank)
    output_table, output_projection = factors(arguments.output_head_rank)
    with torch.no_grad():
        model.token_embedding.weight.copy_(input_table)
        model.input_projection.weight.copy_(input_projection_transposed.T)
        model.output_embedding.weight.copy_(output_table)
        model.output_projection.weight.copy_(output_projection)
    reconstructed_input = model.token_embedding.weight @ model.input_projection.weight.T
    reconstructed_output = model.output_embedding.weight @ model.output_projection.weight
    result = {
        "method": "truncated-SVD input/output factorization plus replicated method cells",
        "base_checkpoint": str(Path(arguments.base_checkpoint)),
        "input_embedding_rank": arguments.input_embedding_rank,
        "output_head_rank": arguments.output_head_rank,
        "method_cell_count": arguments.method_cell_count,
        "attention_residual_scale": arguments.attention_residual_scale,
        "ffn_residual_scale": arguments.ffn_residual_scale,
        "parameter_count": model.parameter_count(),
        "parameter_compression": arguments.teacher_parameters / model.parameter_count(),
        "source_free_after_export": True,
        "input_weight_squared_energy_retained": float(
            (singular_values[: arguments.input_embedding_rank].square().sum()
             / singular_values.square().sum()).item()
        ),
        "input_weight_relative_frobenius_error": float(
            ((source_weight - reconstructed_input).norm() / source_weight.norm()).item()
        ),
        "output_weight_squared_energy_retained": float(
            (singular_values[: arguments.output_head_rank].square().sum()
             / singular_values.square().sum()).item()
        ),
        "output_weight_relative_frobenius_error": float(
            ((source_weight - reconstructed_output).norm() / source_weight.norm()).item()
        ),
    }
    if result["parameter_compression"] < 10.0:
        raise ValueError("Configuration violates the 10x compression requirement")
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": config.__dict__,
        "factorized_initialization": result,
    }, destination)
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--input-embedding-rank", type=int, default=128)
    parser.add_argument("--output-head-rank", type=int, default=128)
    parser.add_argument("--method-cell-count", type=int, default=6)
    parser.add_argument("--attention-residual-scale", type=float, default=1.0)
    parser.add_argument("--ffn-residual-scale", type=float, default=0.75)
    parser.add_argument("--teacher-parameters", type=int, default=354_483_968)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(initialize(parse_args()), indent=2))
