#!/usr/bin/env python3
"""Measure CE and prompt separation under exact attention/FFN residual scaling."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from research.lfm.audit_observer_d import quantiles
from research.lfm.observer_d_lm import ObserverDLM, ObserverDLMConfig
from research.lfm.probe_lfm import choose_device
from research.lfm.train_observer_d_lm import TrajectoryShards


def scaled_forward(
    model: ObserverDLM,
    input_ids: torch.Tensor,
    attention_scale: float,
    ffn_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    length = input_ids.shape[1]
    states = model.embed_tokens(input_ids) * math.sqrt(model.config.width)
    states = states + model.sinusoidal_positions(
        length, model.config.width, input_ids.device, states.dtype
    )[None]
    causal_mask = torch.ones(length, length, dtype=torch.bool, device=input_ids.device).triu(1)
    for depth in range(model.config.repetitions):
        method_cell = model.method_cell_for_depth(depth)
        states = states + model.depth_embedding[depth]
        states = method_cell.local(states)
        if (depth + 1) % model.config.attention_every == 0:
            attended = method_cell.global_attention(states, causal_mask)
            states = states + attention_scale * (attended - states)
        ffn = method_cell.feed_forward(states)
        states = states + ffn_scale * (ffn - states)
        states = model.apply_depth_adapter(states, depth)
    normalized = model.final_norm(states)
    return model.output_logits(normalized), normalized


def pairwise_prompt_metrics(states: torch.Tensor, probabilities: torch.Tensor) -> dict:
    cosines = []
    distances = []
    divergences = []
    for first, second in itertools.combinations(range(states.shape[0]), 2):
        cosines.append(float(F.cosine_similarity(states[first][None], states[second][None]).item()))
        distances.append(float((states[first] - states[second]).norm().item()))
        midpoint = 0.5 * (probabilities[first] + probabilities[second])
        divergences.append(float((0.5 * (
            torch.special.xlogy(
                probabilities[first], probabilities[first] / midpoint.clamp_min(1e-30)
            ).sum()
            + torch.special.xlogy(
                probabilities[second], probabilities[second] / midpoint.clamp_min(1e-30)
            ).sum()
        )).item()))
    return {
        "hidden_cosine": quantiles(np.asarray(cosines)),
        "hidden_distance": quantiles(np.asarray(distances)),
        "distribution_jensen_shannon": quantiles(np.asarray(divergences)),
        "unique_top1_tokens": int(probabilities.argmax(dim=-1).unique().numel()),
    }


def validation_ce(
    model: ObserverDLM,
    directory: Path,
    sequences: int,
    batch_size: int,
    attention_scale: float,
    ffn_scale: float,
    device: str,
) -> dict:
    loader = DataLoader(
        TrajectoryShards(directory, shuffle=False), batch_size=batch_size, num_workers=0
    )
    loss_sum = 0.0
    supervised = 0
    audited = 0
    with torch.inference_mode():
        for batch in loader:
            if audited >= sequences:
                break
            take = min(sequences - audited, batch["input_ids"].shape[0])
            inputs = batch["input_ids"][:take].to(device)
            labels = batch["labels"][:take].to(device)
            mask = batch["loss_mask"][:take].to(device)
            logits, _ = scaled_forward(
                model, inputs, attention_scale, ffn_scale
            )
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
            ).reshape_as(labels)
            loss_sum += float((losses * mask).sum().item())
            supervised += int(mask.sum().item())
            audited += take
    return {
        "sequences": audited,
        "supervised_tokens": supervised,
        "ce": loss_sum / supervised,
    }


def prompt_metrics(
    model: ObserverDLM,
    prompts: list[dict],
    tokenizer,
    attention_scale: float,
    ffn_scale: float,
    device: str,
) -> dict:
    states = []
    probabilities = []
    records = []
    with torch.inference_mode():
        for prompt in prompts:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt["text"]}],
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = tokenizer(rendered, return_tensors="pt")["input_ids"].to(device)
            logits, hidden = scaled_forward(
                model, inputs, attention_scale, ffn_scale
            )
            final_logits = logits[0, -1].float()
            probability = F.softmax(final_logits, dim=-1)
            token_id = int(probability.argmax().item())
            states.append(hidden[0, -1].float().cpu())
            probabilities.append(probability.cpu())
            records.append({
                "prompt_id": prompt["id"],
                "top1_token_id": token_id,
                "top1_token": tokenizer.decode([token_id]),
                "top1_probability": float(probability[token_id].item()),
                "entropy": float(
                    (-(probability * probability.clamp_min(1e-30).log()).sum()).item()
                ),
            })
    return {
        **pairwise_prompt_metrics(torch.stack(states), torch.stack(probabilities)),
        "records": records,
    }


def audit(arguments: argparse.Namespace) -> dict:
    device = choose_device(arguments.device)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    model = ObserverDLM(ObserverDLMConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(arguments.teacher)
    prompts = json.loads(Path(arguments.prompts).read_text())["prompts"]
    settings = [
        (1.0, 1.0),
        (1.0, 0.0), (1.0, 0.25), (1.0, 0.5), (1.0, 0.75), (1.0, 1.25),
        (0.0, 1.0), (0.5, 1.0), (1.5, 1.0), (2.0, 1.0),
        (1.5, 0.5), (2.0, 0.5),
    ]
    records = []
    for attention_scale, ffn_scale in settings:
        records.append({
            "attention_scale": attention_scale,
            "ffn_scale": ffn_scale,
            "raw": validation_ce(
                model,
                Path(arguments.raw_validation),
                arguments.raw_sequences,
                arguments.batch_size,
                attention_scale,
                ffn_scale,
                device,
            ),
            "instruction": validation_ce(
                model,
                Path(arguments.instruction_validation),
                arguments.instruction_sequences,
                arguments.batch_size,
                attention_scale,
                ffn_scale,
                device,
            ),
            "prompts": prompt_metrics(
                model,
                prompts,
                tokenizer,
                attention_scale,
                ffn_scale,
                device,
            ),
        })
    result = {"checkpoint": arguments.checkpoint, "settings": records}
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw-validation", required=True)
    parser.add_argument("--instruction-validation", required=True)
    parser.add_argument("--teacher", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--prompts", default="research/lfm/control_prompts.json")
    parser.add_argument("--raw-sequences", type=int, default=64)
    parser.add_argument("--instruction-sequences", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output", default="research/results/lfm_operator_scaling_audit.json")
    return parser.parse_args()


if __name__ == "__main__":
    result = audit(parse_args())
    print(json.dumps([
        {
            "attention_scale": row["attention_scale"],
            "ffn_scale": row["ffn_scale"],
            "raw_ce": row["raw"]["ce"],
            "instruction_ce": row["instruction"]["ce"],
            "prompt_js": row["prompts"]["distribution_jensen_shannon"]["mean"],
        }
        for row in result["settings"]
    ], indent=2))
