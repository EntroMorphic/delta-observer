#!/usr/bin/env python3
"""Measure recurrent-state geometry and exact greedy degeneration onset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from research.lfm.audit_observer_d import exposure_counts
from research.lfm.observer_d_lm import ObserverDLM, ObserverDLMConfig
from research.lfm.probe_lfm import choose_device
from research.lfm.train_observer_d_lm import TrajectoryShards


def spectrum_metrics(sum_vector: torch.Tensor, gram: torch.Tensor, count: int) -> dict:
    """Return measured centered and uncentered second-moment geometry."""
    mean = sum_vector / count
    second_moment = gram / count
    covariance = second_moment - torch.outer(mean, mean)
    centered = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    uncentered = torch.linalg.eigvalsh(second_moment).clamp_min(0).flip(0)

    def summarize(eigenvalues: torch.Tensor) -> dict:
        fractions = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
        return {
            "effective_rank": float(
                torch.exp(-(fractions * fractions.clamp_min(1e-30).log()).sum()).item()
            ),
            "stable_rank": float(
                (eigenvalues.sum() / eigenvalues[0].clamp_min(1e-30)).item()
            ),
            "top_1_fraction": float(fractions[0].item()),
            "top_16_fraction": float(fractions[:16].sum().item()),
            "top_64_fraction": float(fractions[:64].sum().item()),
            "eigenvalues": eigenvalues.tolist(),
        }

    mean_square = mean.square().sum()
    total_mean_square = second_moment.diag().sum()
    return {
        "samples": count,
        "mean_vector_norm": float(mean.norm().item()),
        "root_mean_square_state_norm": float(total_mean_square.sqrt().item()),
        "mean_direction_energy_fraction": float(
            (mean_square / total_mean_square.clamp_min(1e-30)).item()
        ),
        "centered": summarize(centered),
        "uncentered": summarize(uncentered),
    }


def state_geometry(
    model: ObserverDLM,
    validation_data: Path,
    sequences: int,
    batch_size: int,
    device: str,
) -> dict:
    dataset = TrajectoryShards(validation_data, shuffle=False)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    width = model.config.width
    sums = [torch.zeros(width, dtype=torch.float64) for _ in range(model.config.repetitions)]
    grams = [
        torch.zeros(width, width, dtype=torch.float64)
        for _ in range(model.config.repetitions)
    ]
    transition_cosine_sum = torch.zeros(model.config.repetitions, dtype=torch.float64)
    transition_relative_norm_sum = torch.zeros(model.config.repetitions, dtype=torch.float64)
    transition_count = 0
    sample_count = 0
    audited_sequences = 0
    with torch.inference_mode():
        for batch in loader:
            if audited_sequences >= sequences:
                break
            take = min(sequences - audited_sequences, batch["input_ids"].shape[0])
            input_ids = batch["input_ids"][:take].to(device)
            embedded = model.embed_tokens(input_ids) * math.sqrt(model.config.width)
            embedded = embedded + model.sinusoidal_positions(
                input_ids.shape[1], model.config.width, input_ids.device, embedded.dtype
            )[None]
            output = model(input_ids, return_method_states=True)
            previous = embedded
            for depth, state in enumerate(output["method_states"]):
                flat = state.detach().float().cpu().reshape(-1, width).double()
                sums[depth] += flat.sum(dim=0)
                grams[depth] += flat.T @ flat
                previous_flat = previous.detach().float().reshape(-1, width)
                current_flat = state.detach().float().reshape(-1, width)
                transition_cosine_sum[depth] += float(
                    F.cosine_similarity(previous_flat, current_flat, dim=-1).sum().item()
                )
                transition_relative_norm_sum[depth] += float(
                    (
                        (current_flat - previous_flat).norm(dim=-1)
                        / previous_flat.norm(dim=-1).clamp_min(1e-30)
                    ).sum().item()
                )
                previous = state
            batch_samples = input_ids.numel()
            sample_count += batch_samples
            transition_count += batch_samples
            audited_sequences += take
    return {
        "sequences": audited_sequences,
        "tokens_per_depth": sample_count,
        "depths": [
            {
                "depth": depth + 1,
                **spectrum_metrics(sums[depth], grams[depth], sample_count),
                "previous_state_cosine": float(
                    transition_cosine_sum[depth].item() / transition_count
                ),
                "relative_update_norm": float(
                    transition_relative_norm_sum[depth].item() / transition_count
                ),
            }
            for depth in range(model.config.repetitions)
        ],
    }


def first_repeated_ngram(tokens: list[int], width: int) -> dict | None:
    seen: dict[tuple[int, ...], int] = {}
    for end in range(width, len(tokens) + 1):
        ngram = tuple(tokens[end - width : end])
        if ngram in seen:
            return {
                "step": end - 1,
                "first_start": seen[ngram],
                "repeat_start": end - width,
                "width": width,
                "token_ids": list(ngram),
            }
        seen[ngram] = end - width
    return None


def first_periodic_suffix(tokens: list[int], maximum_period: int = 16) -> dict | None:
    """Find the first step ending in three exact repeats of a period."""
    for end in range(3, len(tokens) + 1):
        for period in range(1, min(maximum_period, end // 3) + 1):
            unit = tokens[end - period : end]
            if tokens[end - 3 * period : end] == unit * 3:
                return {
                    "step": end - 1,
                    "start": end - 3 * period,
                    "period": period,
                    "repetitions": 3,
                    "unit_token_ids": unit,
                }
    return None


def frequency_rank(frequency: torch.Tensor, token_id: int) -> int:
    return 1 + int((frequency > frequency[token_id]).sum().item())


def generation_audit(
    model: ObserverDLM,
    tokenizer: AutoTokenizer,
    prompt_path: Path,
    frequency: torch.Tensor,
    max_new_tokens: int,
    device: str,
) -> dict:
    registry = json.loads(prompt_path.read_text())
    results = []
    for prompt in registry["prompts"]:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["text"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        generated = tokenizer(rendered, return_tensors="pt")["input_ids"].to(device)
        prompt_tokens = generated.shape[1]
        steps = []
        continuation: list[int] = []
        with torch.inference_mode():
            for step in range(max_new_tokens):
                logits = model(generated)["logits"][0, -1].float()
                log_probabilities = F.log_softmax(logits, dim=-1)
                probabilities = log_probabilities.exp()
                top_values, top_indices = torch.topk(logits, 2)
                token_id = int(top_indices[0].item())
                continuation.append(token_id)
                exposure = int(frequency[token_id].item())
                steps.append({
                    "step": step,
                    "token_id": token_id,
                    "token": tokenizer.convert_ids_to_tokens(token_id),
                    "decoded": tokenizer.decode([token_id]),
                    "probability": float(probabilities[token_id].item()),
                    "entropy": float((-(probabilities * log_probabilities).sum()).item()),
                    "top1_top2_logit_margin": float((top_values[0] - top_values[1]).item()),
                    "training_exposure": exposure,
                    "training_exposure_rank": frequency_rank(frequency, token_id),
                    "training_exposure_percentile": float(
                        (frequency <= frequency[token_id]).float().mean().item()
                    ),
                })
                generated = torch.cat(
                    (generated, torch.tensor([[token_id]], device=device)), dim=1
                )
                if token_id == tokenizer.eos_token_id:
                    break
        results.append({
            "prompt": prompt,
            "rendered_prompt": rendered,
            "prompt_tokens": prompt_tokens,
            "continuation_token_ids": continuation,
            "continuation": tokenizer.decode(continuation, skip_special_tokens=True),
            "steps": steps,
            "first_repeated_token": first_repeated_ngram(continuation, 1),
            "first_repeated_bigram": first_repeated_ngram(continuation, 2),
            "first_repeated_trigram": first_repeated_ngram(continuation, 3),
            "first_three_repeat_cycle": first_periodic_suffix(continuation),
        })
    return {
        "suite_version": registry.get("version"),
        "greedy": True,
        "max_new_tokens": max_new_tokens,
        "cycle_definition": "first generated step whose suffix is three exact repeats of a period 1 through 16",
        "results": results,
    }


def audit(arguments: argparse.Namespace) -> dict:
    device = choose_device(arguments.device)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    model = ObserverDLM(ObserverDLMConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(arguments.teacher)
    counts = exposure_counts([Path(path) for path in arguments.training_data])
    frequency = torch.tensor(
        [counts[index] for index in range(model.config.vocab_size)], dtype=torch.int64
    )
    result = {
        "checkpoint": str(Path(arguments.checkpoint)),
        "device": device,
        "state_geometry": state_geometry(
            model,
            Path(arguments.validation_data),
            arguments.sequences,
            arguments.batch_size,
            device,
        ),
        "generation": generation_audit(
            model,
            tokenizer,
            Path(arguments.prompts),
            frequency,
            arguments.max_new_tokens,
            device,
        ),
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
    parser.add_argument("--teacher", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--prompts", default="research/lfm/control_prompts.json")
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output", default="research/results/lfm_state_generation_audit.json")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = audit(arguments)
    summary = {
        "output": str(Path(arguments.output)),
        "state_effective_rank": [
            depth["centered"]["effective_rank"]
            for depth in result["state_geometry"]["depths"]
        ],
        "cycle_onsets": {
            row["prompt"]["id"]: row["first_three_repeat_cycle"]
            for row in result["generation"]["results"]
        },
    }
    print(json.dumps(summary, indent=2))
