#!/usr/bin/env python3
"""Measure how much prompt identity survives to the first response distribution."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.lfm.audit_observer_d import quantiles
from research.lfm.observer_d_lm import ObserverDLM, ObserverDLMConfig
from research.lfm.probe_lfm import choose_device


def distribution_metrics(logits: torch.Tensor) -> dict:
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    probabilities = log_probabilities.exp()
    top_values, top_indices = torch.topk(probabilities, 10)
    return {
        "entropy": float((-(probabilities * log_probabilities).sum()).item()),
        "top_ids": top_indices.cpu().tolist(),
        "top_probabilities": top_values.cpu().tolist(),
    }


def jensen_shannon(first: torch.Tensor, second: torch.Tensor) -> float:
    midpoint = 0.5 * (first + second)
    value = 0.5 * (
        torch.special.xlogy(first, first / midpoint.clamp_min(1e-30)).sum()
        + torch.special.xlogy(second, second / midpoint.clamp_min(1e-30)).sum()
    )
    return float(value.item())


def rank_metrics(states: torch.Tensor) -> dict:
    centered = states.float() - states.float().mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    eigenvalues = singular_values.square()
    fractions = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    return {
        "matrix_shape": list(states.shape),
        "algebraic_rank": int(torch.linalg.matrix_rank(centered).item()),
        "effective_rank": float(
            torch.exp(-(fractions * fractions.clamp_min(1e-30).log()).sum()).item()
        ),
        "top_1_variance_fraction": float(fractions[0].item()),
        "singular_values": singular_values.cpu().tolist(),
    }


def aggregate(records: list[dict], states: torch.Tensor, probabilities: torch.Tensor) -> dict:
    hidden_cosines = []
    distribution_js = []
    top1_matches = 0
    for first, second in itertools.combinations(range(len(records)), 2):
        hidden_cosines.append(float(F.cosine_similarity(
            states[first].unsqueeze(0), states[second].unsqueeze(0)
        ).item()))
        distribution_js.append(jensen_shannon(probabilities[first], probabilities[second]))
        top1_matches += int(records[first]["top_ids"][0] == records[second]["top_ids"][0])
    return {
        "prompt_state_geometry": rank_metrics(states),
        "pairwise_hidden_cosine": quantiles(np.asarray(hidden_cosines)),
        "pairwise_distribution_jensen_shannon": quantiles(np.asarray(distribution_js)),
        "pairwise_top1_matches": top1_matches,
        "pairwise_comparisons": len(hidden_cosines),
        "unique_top1_tokens": len({record["top_ids"][0] for record in records}),
    }


def state_aggregate(states: torch.Tensor) -> dict:
    cosines = [
        float(F.cosine_similarity(states[first].unsqueeze(0), states[second].unsqueeze(0)).item())
        for first, second in itertools.combinations(range(states.shape[0]), 2)
    ]
    return {
        "geometry": rank_metrics(states),
        "pairwise_cosine": quantiles(np.asarray(cosines)),
    }


def observer_measure(
    model: ObserverDLM,
    rendered_prompts: list[str],
    empty_rendered: str,
    tokenizer,
    device: str,
) -> dict:
    records = []
    states = []
    probabilities = []
    depth_states: list[list[torch.Tensor]] = [
        [] for _ in range(model.config.repetitions)
    ]
    empty_ids = tokenizer(empty_rendered, return_tensors="pt")["input_ids"].to(device)
    with torch.inference_mode():
        empty_output = model(empty_ids, return_method_states=True)
        empty_state = model.final_norm(empty_output["method_states"][-1])[:, -1].float()
        empty_logits = model.output_logits(empty_state)[0]
        empty_probability = F.softmax(empty_logits, dim=-1)
        for rendered in rendered_prompts:
            input_ids = tokenizer(rendered, return_tensors="pt")["input_ids"].to(device)
            output = model(input_ids, return_method_states=True)
            state = model.final_norm(output["method_states"][-1])[:, -1].float()[0]
            logits = model.output_logits(state)
            probability = F.softmax(logits.float(), dim=-1)
            record = {
                "prompt_tokens": int(input_ids.shape[1]),
                **distribution_metrics(logits),
                "hidden_norm": float(state.norm().item()),
                "hidden_cosine_to_empty_prompt": float(
                    F.cosine_similarity(state.unsqueeze(0), empty_state).item()
                ),
                "distribution_js_to_empty_prompt": jensen_shannon(
                    probability, empty_probability
                ),
            }
            records.append(record)
            states.append(state.cpu())
            probabilities.append(probability.cpu())
            for depth, depth_state in enumerate(output["method_states"]):
                depth_states[depth].append(depth_state[0, -1].float().cpu())
    stacked_states = torch.stack(states)
    stacked_probabilities = torch.stack(probabilities)
    return {
        "records": records,
        "aggregate": aggregate(records, stacked_states, stacked_probabilities),
        "probabilities": stacked_probabilities,
        "depth_prompt_separation": [
            {"depth": depth + 1, **state_aggregate(torch.stack(values))}
            for depth, values in enumerate(depth_states)
        ],
    }


def teacher_measure(
    teacher,
    rendered_prompts: list[str],
    empty_rendered: str,
    tokenizer,
    device: str,
) -> dict:
    records = []
    states = []
    probabilities = []
    depth_states = None
    empty_ids = tokenizer(empty_rendered, return_tensors="pt")["input_ids"].to(device)
    with torch.inference_mode():
        empty_output = teacher(empty_ids, output_hidden_states=True)
        empty_state = empty_output.hidden_states[-1][0, -1].float()
        empty_probability = F.softmax(empty_output.logits[0, -1].float(), dim=-1)
        for rendered in rendered_prompts:
            input_ids = tokenizer(rendered, return_tensors="pt")["input_ids"].to(device)
            output = teacher(input_ids, output_hidden_states=True)
            state = output.hidden_states[-1][0, -1].float()
            logits = output.logits[0, -1].float()
            probability = F.softmax(logits, dim=-1)
            record = {
                "prompt_tokens": int(input_ids.shape[1]),
                **distribution_metrics(logits),
                "hidden_norm": float(state.norm().item()),
                "hidden_cosine_to_empty_prompt": float(
                    F.cosine_similarity(state.unsqueeze(0), empty_state.unsqueeze(0)).item()
                ),
                "distribution_js_to_empty_prompt": jensen_shannon(
                    probability, empty_probability
                ),
            }
            records.append(record)
            states.append(state.cpu())
            probabilities.append(probability.cpu())
            if depth_states is None:
                depth_states = [[] for _ in output.hidden_states]
            for depth, depth_state in enumerate(output.hidden_states):
                depth_states[depth].append(depth_state[0, -1].float().cpu())
    stacked_states = torch.stack(states)
    stacked_probabilities = torch.stack(probabilities)
    return {
        "records": records,
        "aggregate": aggregate(records, stacked_states, stacked_probabilities),
        "probabilities": stacked_probabilities,
        "depth_prompt_separation": [
            {"depth": depth, **state_aggregate(torch.stack(values))}
            for depth, values in enumerate(depth_states)
        ],
    }


def audit(arguments: argparse.Namespace) -> dict:
    device = choose_device(arguments.device)
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(arguments.teacher)
    registry = json.loads(Path(arguments.prompts).read_text())
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["text"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in registry["prompts"]
    ]
    empty_rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        add_generation_prompt=True,
        tokenize=False,
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        arguments.teacher, dtype=dtype
    ).to(device).eval()
    teacher_result = teacher_measure(
        teacher, rendered, empty_rendered, tokenizer, device
    )
    teacher_probabilities = teacher_result.pop("probabilities")
    result = {
        "prompts": registry["prompts"],
        "teacher": teacher_result,
        "observers": {},
    }
    del teacher
    if device == "mps":
        torch.mps.empty_cache()
    for checkpoint_path in arguments.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = ObserverDLM(ObserverDLMConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        measured = observer_measure(
            model, rendered, empty_rendered, tokenizer, device
        )
        student_probabilities = measured.pop("probabilities")
        measured["teacher_student_first_token_js"] = quantiles(np.asarray([
            jensen_shannon(teacher_probabilities[index], student_probabilities[index])
            for index in range(len(rendered))
        ]))
        result["observers"][str(Path(checkpoint_path))] = measured
        del model
        if device == "mps":
            torch.mps.empty_cache()
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--teacher", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--prompts", default="research/lfm/control_prompts.json")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output", default="research/results/lfm_prompt_information_audit.json")
    return parser.parse_args()


if __name__ == "__main__":
    result = audit(parse_args())
    summary = {"teacher": result["teacher"]["aggregate"]}
    summary.update({
        path: measured["aggregate"]
        for path, measured in result["observers"].items()
    })
    print(json.dumps(summary, indent=2))
