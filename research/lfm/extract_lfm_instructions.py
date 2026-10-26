#!/usr/bin/env python3
"""Generate masked conversational behavior and trajectories from LFM2.5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.lfm.extract_lfm_corpus import hash_array, save_shard
from research.lfm.probe_lfm import choose_device, random_projection


@dataclass(frozen=True)
class InstructionExtractionConfig:
    model: str
    revision: str
    dataset: str
    dataset_revision: str
    dataset_config: str | None
    split: str
    messages_field: str
    skip_prompts: int
    sequences: int
    sequence_length: int
    max_prompt_tokens: int
    batch_size: int
    shard_size: int
    top_k: int
    projection_dim: int
    projection_seed: int
    capture_trajectories: bool
    log_every: int
    device: str
    frozen_prompts: str
    exclude_prompt_hashes: str | None
    output: str


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def extract_user_text(record: dict, messages_field: str) -> str | None:
    """Read the first user turn from common instruction-dataset schemas."""
    messages = record.get(messages_field)
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role", message.get("from"))
            if role in ("user", "human"):
                content = message.get("content", message.get("value"))
                if isinstance(content, str) and content.strip():
                    return content.strip()
    for field in ("prompt", "instruction", "question"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def prompt_stream(
    dataset,
    messages_field: str,
    excluded: set[str],
    stats: dict[str, int] | None = None,
):
    stats = stats if stats is not None else {}
    seen: set[str] = set()
    iterator = iter(dataset)
    try:
        for record in iterator:
            text = extract_user_text(record, messages_field)
            if text is None:
                stats["rejected_missing_user_text"] = (
                    stats.get("rejected_missing_user_text", 0) + 1
                )
                continue
            normalized = normalize_text(text)
            if normalized in excluded:
                stats["rejected_frozen_prompt"] = (
                    stats.get("rejected_frozen_prompt", 0) + 1
                )
                continue
            if normalized in seen:
                stats["rejected_duplicate_prompt"] = (
                    stats.get("rejected_duplicate_prompt", 0) + 1
                )
                continue
            seen.add(normalized)
            yield text
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def frozen_prompt_set(path: Path) -> set[str]:
    registry = json.loads(path.read_text())
    return {normalize_text(prompt["text"]) for prompt in registry["prompts"]}


def extract(config: InstructionExtractionConfig) -> dict:
    destination = Path(config.output)
    destination.mkdir(parents=True, exist_ok=True)
    device = choose_device(config.device)
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token")
    model = AutoModelForCausalLM.from_pretrained(
        config.model, revision=config.revision, dtype=dtype
    ).to(device).eval()
    dataset = load_dataset(
        config.dataset,
        config.dataset_config,
        split=config.split,
        streaming=True,
        revision=config.dataset_revision,
    )

    projection_cpu = random_projection(
        model.config.hidden_size, config.projection_dim, config.projection_seed
    )
    projection_array = projection_cpu.numpy().astype(np.float32)
    np.save(destination / "projection.npy", projection_array)
    projection = projection_cpu.to(device=device, dtype=dtype)

    frozen_path = Path(config.frozen_prompts)
    frozen_bytes = frozen_path.read_bytes()
    frozen_registry = json.loads(frozen_bytes)
    excluded = {normalize_text(prompt["text"]) for prompt in frozen_registry["prompts"]}
    prompt_stats: dict[str, int] = {
        "rejected_missing_user_text": 0,
        "rejected_frozen_prompt": 0,
        "rejected_duplicate_prompt": 0,
    }
    prompts = prompt_stream(dataset, config.messages_field, excluded, prompt_stats)
    prior_hashes = set()
    if config.exclude_prompt_hashes:
        prior_hashes = set(Path(config.exclude_prompt_hashes).read_text().splitlines())
    for _ in range(config.skip_prompts):
        next(prompts)

    rows: list[dict] = []
    shards: list[dict] = []
    accepted_hashes: list[str] = []
    rejected_too_long = 0
    rejected_prior_prompt = 0
    extracted = 0
    shard_index = 0
    supervised_tokens = 0
    teacher_nll_sum = 0.0
    teacher_top_k_mass_sum = 0.0
    started = time.perf_counter()

    while extracted < config.sequences:
        batch_target = min(config.batch_size, config.sequences - extracted)
        batch_texts: list[str] = []
        batch_prompt_ids: list[list[int]] = []
        while len(batch_texts) < batch_target:
            user_text = next(prompts)
            prompt_hash = hashlib.sha256(normalize_text(user_text).encode()).hexdigest()
            if prompt_hash in prior_hashes:
                rejected_prior_prompt += 1
                continue
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_ids = tokenizer(rendered, add_special_tokens=True)["input_ids"]
            if len(prompt_ids) > config.max_prompt_tokens or len(prompt_ids) >= config.sequence_length:
                rejected_too_long += 1
                continue
            batch_texts.append(user_text)
            batch_prompt_ids.append(prompt_ids)

        padded_width = max(map(len, batch_prompt_ids))
        padded_prompts = []
        attention_masks = []
        for prompt_ids in batch_prompt_ids:
            padding = padded_width - len(prompt_ids)
            padded_prompts.append([tokenizer.eos_token_id] * padding + prompt_ids)
            attention_masks.append([0] * padding + [1] * len(prompt_ids))
        prompt_tensor = torch.tensor(padded_prompts, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)
        response_budget = config.sequence_length + 1 - padded_width
        with torch.inference_mode():
            generated = model.generate(
                prompt_tensor,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=response_budget,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        packed_rows = []
        actual_lengths = []
        for index, prompt_ids in enumerate(batch_prompt_ids):
            response = generated[index, padded_width:].tolist()
            if tokenizer.eos_token_id in response:
                response = response[: response.index(tokenizer.eos_token_id) + 1]
            token_ids = (prompt_ids + response)[: config.sequence_length + 1]
            actual_lengths.append(len(token_ids))
            token_ids.extend(
                [tokenizer.eos_token_id] * (config.sequence_length + 1 - len(token_ids))
            )
            packed_rows.append(token_ids)
        tokens = torch.tensor(packed_rows, dtype=torch.long, device=device)
        input_ids, labels = tokens[:, :-1], tokens[:, 1:]
        loss_mask = torch.zeros_like(labels, dtype=torch.bool)
        for index, (prompt_ids, actual_length) in enumerate(
            zip(batch_prompt_ids, actual_lengths)
        ):
            loss_mask[index, len(prompt_ids) - 1 : actual_length - 1] = True

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=config.capture_trajectories,
                use_cache=False,
            )
            logits = outputs.logits.float()
            top_values, top_indices = torch.topk(logits, config.top_k, dim=-1)
            logsumexp = torch.logsumexp(logits, dim=-1)
            teacher_label_logits = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            projected = torch.stack(
                [hidden.to(dtype) @ projection for hidden in outputs.hidden_states], dim=1
            ) if config.capture_trajectories else None

        mask_float = loss_mask.float()
        count = int(loss_mask.sum().item())
        teacher_nll_sum += float(
            ((logsumexp - teacher_label_logits) * mask_float).sum().item()
        )
        teacher_top_k_mass_sum += float(
            (torch.exp(top_values - logsumexp.unsqueeze(-1)).sum(dim=-1) * mask_float)
            .sum()
            .item()
        )
        supervised_tokens += count
        for index, user_text in enumerate(batch_texts):
            rows.append({
                "input_ids": input_ids[index].cpu().numpy(),
                "labels": labels[index].cpu().numpy(),
                "top_indices": top_indices[index].cpu().numpy(),
                "top_values": top_values[index].cpu().numpy(),
                "logsumexp": logsumexp[index].cpu().numpy(),
                "teacher_label_logits": teacher_label_logits[index].cpu().numpy(),
                "loss_mask": loss_mask[index].cpu().numpy(),
            })
            if projected is not None:
                rows[-1]["projected"] = projected[index].cpu().numpy()
            accepted_hashes.append(
                hashlib.sha256(normalize_text(user_text).encode()).hexdigest()
            )
        extracted += batch_target
        if extracted == config.sequences or extracted % config.log_every == 0:
            print(f"Extracted {extracted}/{config.sequences}", flush=True)
        if len(rows) >= config.shard_size or extracted == config.sequences:
            shards.append(save_shard(destination, shard_index, rows))
            rows.clear()
            shard_index += 1

    prompts.close()
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    prompt_hash_bytes = ("\n".join(accepted_hashes) + "\n").encode()
    (destination / "prompt_hashes.txt").write_bytes(prompt_hash_bytes)
    prompt_hash = hashlib.sha256(prompt_hash_bytes).hexdigest()
    manifest = {
        "format": "observer-d-lfm-instruction-trajectory-v1",
        "config": asdict(config),
        "resolved_device": device,
        "dtype": str(dtype),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "layer_types": list(model.config.layer_types),
        "hidden_state_count": model.config.num_hidden_layers + 1,
        "projection_shape": list(projection_array.shape),
        "projection_sha256": hash_array(projection_array),
        "prompt_hashes_sha256": prompt_hash,
        "prompt_hashes_path": "prompt_hashes.txt",
        "accepted_unique_prompts": len(accepted_hashes),
        **prompt_stats,
        "frozen_prompts_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "frozen_prompts_version": frozen_registry.get("version"),
        "excluded_frozen_prompt_count": len(excluded),
        "rejected_too_long": rejected_too_long,
        "rejected_prior_prompt": rejected_prior_prompt,
        "elapsed_seconds": elapsed,
        "tokens": extracted * config.sequence_length,
        "supervised_tokens": supervised_tokens,
        "tokens_per_second": extracted * config.sequence_length / elapsed,
        "teacher_cross_entropy": teacher_nll_sum / supervised_tokens,
        "teacher_top_k_probability_mass": teacher_top_k_mass_sum / supervised_tokens,
        "shards": shards,
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> InstructionExtractionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument(
        "--dataset-revision", default="8049631c405ae6576f93f445c6b8166f76f5505a"
    )
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--split", default="train_sft")
    parser.add_argument("--messages-field", default="messages")
    parser.add_argument("--skip-prompts", type=int, default=0)
    parser.add_argument("--sequences", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--no-trajectories", dest="capture_trajectories", action="store_false")
    parser.set_defaults(capture_trajectories=True)
    parser.add_argument("--log-every", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--frozen-prompts", default="research/lfm/control_prompts.json")
    parser.add_argument("--exclude-prompt-hashes")
    parser.add_argument("--output", default="research/results/lfm_instruction_corpus")
    values = vars(parser.parse_args())
    if values["dataset_config"].lower() in ("none", "null", ""):
        values["dataset_config"] = None
    return InstructionExtractionConfig(**values)


def main() -> None:
    extract(parse_args())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
