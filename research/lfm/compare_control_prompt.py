#!/usr/bin/env python3
"""Compare LFM2.5 and a detached Observer.D-LM on frozen control prompts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.lfm.observer_d_lm import ObserverDLM, ObserverDLMConfig
from research.lfm.probe_lfm import choose_device


def load_prompt(path: Path, prompt_id: str) -> dict:
    registry = json.loads(path.read_text())
    for prompt in registry["prompts"]:
        if prompt["id"] == prompt_id:
            return prompt
    raise KeyError(f"Unknown control prompt: {prompt_id}")


def continuation_metrics(token_ids: list[int]) -> dict:
    """Report simple degeneration indicators without pretending to score quality."""
    bigrams = list(zip(token_ids, token_ids[1:]))
    trigrams = list(zip(token_ids, token_ids[1:], token_ids[2:]))
    return {
        "tokens": len(token_ids),
        "distinct_bigram_ratio": len(set(bigrams)) / max(1, len(bigrams)),
        "distinct_trigram_ratio": len(set(trigrams)) / max(1, len(trigrams)),
    }


def observer_generate(
    model: ObserverDLM,
    input_ids: torch.Tensor,
    eos_token_id: int,
    max_new_tokens: int,
) -> tuple[torch.Tensor, float]:
    generated = input_ids
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model(generated)["logits"][:, -1]
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=-1)
            if int(next_token.item()) == eos_token_id:
                break
    if generated.device.type == "mps":
        torch.mps.synchronize()
    return generated, time.perf_counter() - started


def compare(arguments: argparse.Namespace) -> dict:
    device = choose_device(arguments.device)
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    registry = json.loads(Path(arguments.prompts).read_text())
    prompt_records = registry["prompts"] if arguments.all_prompts else [
        load_prompt(Path(arguments.prompts), arguments.prompt_id)
    ]
    tokenizer = AutoTokenizer.from_pretrained(arguments.teacher, revision=arguments.revision)
    teacher = AutoModelForCausalLM.from_pretrained(
        arguments.teacher, revision=arguments.revision, dtype=dtype
    ).to(device).eval()
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    config = ObserverDLMConfig(**checkpoint["config"])
    observer = ObserverDLM(config)
    observer.load_state_dict(checkpoint["model"])
    observer.to(device).eval()

    results = []
    for prompt_record in prompt_records:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_record["text"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids = tokenizer(rendered, return_tensors="pt")["input_ids"]
        if input_ids.shape[1] + arguments.max_new_tokens > config.max_context:
            raise ValueError(f"Prompt {prompt_record['id']} and generation exceed context")
        teacher_started = time.perf_counter()
        with torch.inference_mode():
            teacher_tokens = teacher.generate(
                input_ids.to(device),
                do_sample=False,
                max_new_tokens=arguments.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        if device == "mps":
            torch.mps.synchronize()
        teacher_seconds = time.perf_counter() - teacher_started
        teacher_tokens = teacher_tokens.cpu()
        observer_tokens, observer_seconds = observer_generate(
            observer,
            input_ids.to(device),
            tokenizer.eos_token_id,
            arguments.max_new_tokens,
        )
        observer_tokens = observer_tokens.cpu()
        prompt_length = input_ids.shape[1]
        teacher_ids = teacher_tokens[0, prompt_length:].tolist()
        observer_ids = observer_tokens[0, prompt_length:].tolist()
        results.append({
            "prompt": prompt_record,
            "rendered_prompt": rendered,
            "decoding": {
                "method": "greedy",
                "max_new_tokens": arguments.max_new_tokens,
                "eos_token_id": tokenizer.eos_token_id,
            },
            "teacher": {
                "model": arguments.teacher,
                "seconds": teacher_seconds,
                "token_ids": teacher_ids,
                "text": tokenizer.decode(teacher_ids, skip_special_tokens=True),
                "degeneration_metrics": continuation_metrics(teacher_ids),
            },
            "observer_d": {
                "checkpoint": str(Path(arguments.checkpoint)),
                "seconds": observer_seconds,
                "token_ids": observer_ids,
                "text": tokenizer.decode(observer_ids, skip_special_tokens=True),
                "degeneration_metrics": continuation_metrics(observer_ids),
            },
        })
    result = {
        "suite_version": registry.get("version"),
        "results": results,
    } if arguments.all_prompts else results[0]
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prompts", default="research/lfm/control_prompts.json")
    parser.add_argument("--prompt-id", default="reflective-recursion-cognition")
    parser.add_argument("--all-prompts", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output", default="research/results/lfm_control_prompt.json")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(compare(parse_args()), indent=2))
