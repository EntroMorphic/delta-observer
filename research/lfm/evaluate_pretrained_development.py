#!/usr/bin/env python3
"""Measure an untouched native LLM on a development-only capability suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.lfm.probe_lfm import choose_device


def strict_json(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Non-JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant)


def validate_suite(suite: dict) -> None:
    if suite.get("role") not in ("development_only_not_confirmatory", "frozen_post_training_holdout"):
        raise ValueError("Unsupported evaluation role; sealed final testing is not implemented")
    if not isinstance(suite.get("tasks"), list) or not suite["tasks"]:
        raise ValueError("Suite must contain tasks")
    ids, prompts = set(), set()
    for task in suite["tasks"]:
        for key in ("id", "category", "prompt"):
            if not isinstance(task.get(key), str) or not task[key].strip():
                raise ValueError(f"Missing or invalid task {key}")
        normalized = " ".join(task["prompt"].casefold().split())
        if task["id"] in ids or normalized in prompts:
            raise ValueError("Duplicate task ID or normalized prompt")
        ids.add(task["id"])
        prompts.add(normalized)
        if task.get("validator") not in ("manual", "text_exact", "json_exact"):
            raise ValueError("Unknown validator")
        if task["validator"] == "manual":
            if not task.get("rubric"):
                raise ValueError("Manual task requires rubric")
        elif "expected" not in task:
            raise ValueError("Machine-scored task requires expected answer")
        if task.get("content_validator") not in (None, "json_unfenced", "boxed_integer"):
            raise ValueError("Unknown content validator")


def content_validation(task: dict, response: str) -> dict:
    """Narrow parsers, not an unrestricted semantic judge. Never mine arbitrary numbers."""
    kind = task.get("content_validator")
    if kind == "json_unfenced":
        match = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\s*```\s*", response, re.DOTALL)
        parsed = match.group(1) if match else response
        return {**validate(task, parsed), "method": "whole_response_json_optional_fence"}
    if kind == "boxed_integer":
        boxes = re.findall(r"\\boxed\{(-?\d+)\}", response)
        if boxes and len(set(boxes)) == 1:
            return {"status": "scored", "pass": boxes[0] == task["expected"],
                    "method": "unambiguous_boxed_integer_not_semantic_review"}
        if response.strip() == task["expected"]:
            return {"status": "scored", "pass": True, "method": "exact_integer"}
        return {"status": "pending_review", "pass": None,
                "method": "no_unambiguous_answer_extracted"}
    return {"status": "not_separately_scored", "pass": None}


def validate(task: dict, response: str) -> dict:
    """Strict machine scores or explicit unscored review; never execute code."""
    kind = task["validator"]
    if kind == "manual":
        return {"status": "pending_blinded_review", "pass": None,
                "rubric": task["rubric"]}
    if kind == "text_exact":
        return {"status": "scored", "pass": response.strip() == task["expected"]}
    if kind == "json_exact":
        try:
            value = strict_json(response)
        except (ValueError, TypeError):
            return {"status": "scored", "pass": False, "reason": "invalid_json"}
        # Python considers True == 1 and 7.0 == 7; enforce JSON types as well.
        def equal(actual, expected):
            if type(actual) is not type(expected):
                return False
            if isinstance(expected, dict):
                return actual.keys() == expected.keys() and all(
                    equal(actual[key], expected[key]) for key in expected
                )
            if isinstance(expected, list):
                return len(actual) == len(expected) and all(
                    equal(a, b) for a, b in zip(actual, expected)
                )
            return actual == expected
        return {"status": "scored", "pass": equal(value, task["expected"])}
    raise ValueError(f"Unknown validator: {kind}")


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def chat_inputs(tokenizer, prompt: str):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False,
    )
    return tokenizer(rendered, add_special_tokens=False, return_tensors="pt")


def run(args: argparse.Namespace) -> dict:
    destination = Path(args.output)
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("Use an immutable 40-character model commit")
    tasks_path = Path(args.tasks)
    task_bytes = tasks_path.read_bytes()
    suite = strict_json(task_bytes)
    validate_suite(suite)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "baseline.json"
    # Reserve the result path before loading the model; never clobber an old run.
    with output_path.open("x") as output:
        json.dump({"complete": False, "status": "initializing"}, output)
    device = choose_device(args.device)
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=dtype
    ).to(device).eval()
    adapter_path = getattr(args, "adapter", None)
    disabled_group = getattr(args, "disable_adapter_group", None)
    if disabled_group and not adapter_path:
        raise ValueError("An adapter-group ablation requires an adapter checkpoint")
    adapter_provenance = None
    if adapter_path:
        from research.lfm.native_lora import install_lora, load_adapter
        adapter_bytes = Path(adapter_path).read_bytes()
        checkpoint = torch.load(adapter_path, weights_only=True, map_location="cpu")
        config = checkpoint["config"]
        if config["model"] != args.model or config["revision"] != args.revision:
            raise ValueError("Adapter and base model revisions differ")
        targets = install_lora(model, config["rank"], config["alpha"])
        if targets != config["targets"]:
            raise ValueError("Adapter target modules differ")
        load_adapter(model, checkpoint["adapter"])
        adapter_provenance = {"path": adapter_path,
             "sha256": hashlib.sha256(adapter_bytes).hexdigest(), "config": config}
        if disabled_group:
            from research.lfm.native_lora import LoRALinear
            marker = '.self_attn.' if disabled_group == 'attention' else '.conv.'
            disabled = []
            for name, module in model.named_modules():
                if isinstance(module, LoRALinear) and marker in name:
                    module.scale = 0.
                    disabled.append(name)
            if not disabled:
                raise ValueError("No matching adapter modules for ablation")
            adapter_provenance['inference_ablation'] = {
                'group': disabled_group, 'scale': 0., 'modules': disabled,
                'note': 'Scale-zero diagnostic; adapter arithmetic still executes, no speedup implied'}
    from transformers import GenerationConfig
    generation_config = GenerationConfig(
        do_sample=False, num_beams=1, repetition_penalty=1.0,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id, use_cache=True,
    )
    results = []
    metadata = {
        "runner_version": 2,
        "role": suite["role"], "model": args.model, "revision": args.revision,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "resident_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "tasks_path": str(tasks_path),
        "tasks_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "device": device, "dtype": str(dtype), "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "performance_scope": "batch-one native Transformers generation; includes full-context prefill; not a kernel benchmark",
        "generation_config": generation_config.to_dict(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": __import__("transformers").__version__},
        "peak_runtime_memory_bytes": None,
        "memory_note": "Resident parameter bytes only; runtime peak not measured",
        "training_performed": bool(adapter_path),
        "adapter": adapter_provenance,
    }
    for task in suite["tasks"]:
        # Chat templates already emit BOS/control tokens. Do not add them twice.
        inputs = chat_inputs(tokenizer, task["prompt"]).to(device)
        synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, generation_config=generation_config,
            )
        synchronize(device)
        seconds = time.perf_counter() - started
        ids = generated[0, inputs["input_ids"].shape[1]:].tolist()
        response = tokenizer.decode(ids, skip_special_tokens=True)
        record = {
            "id": task["id"], "category": task["category"],
            "prompt": task["prompt"],
            "normalized_prompt_sha256": hashlib.sha256(
                " ".join(task["prompt"].casefold().split()).encode()
            ).hexdigest(),
            "input_token_ids": inputs["input_ids"][0].tolist(),
            "generated_token_ids": ids, "response": response,
            "elapsed_seconds": seconds, "generated_tokens": len(ids),
            "tokens_per_second": len(ids) / seconds,
            "ended_with_eos": bool(ids and ids[-1] == tokenizer.eos_token_id),
            "validation": validate(task, response),
            "content_validation": content_validation(task, response),
            "hit_token_limit_without_eos": len(ids) >= args.max_new_tokens and not (
                ids and ids[-1] == tokenizer.eos_token_id
            ),
        }
        results.append(record)
        categories = defaultdict(lambda: {"total": 0, "scored": 0, "passed": 0, "pending": 0})
        for row in results:
            counts = categories[row["category"]]
            counts["total"] += 1
            passed = row["validation"]["pass"]
            counts["pending"] += int(passed is None)
            counts["scored"] += int(passed is not None)
            counts["passed"] += int(passed is True)
        payload = {**metadata, "results": results, "categories": dict(categories),
                   "complete": len(results) == len(suite["tasks"]),
                   "aggregate_capability_score": None}
        temporary = destination / "baseline.pending.json"
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        temporary.replace(output_path)
        print(json.dumps({"id": task["id"], "response": response,
                          "validation": record["validation"]}, ensure_ascii=False), flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--revision", default="9e6c6ccf47cd318696e137d381a7ded8fe4df09f")
    parser.add_argument("--tasks", default="research/lfm/pretrained_development_tasks.json")
    parser.add_argument("--adapter", help="Native LoRA checkpoint to evaluate")
    parser.add_argument("--disable-adapter-group", choices=("attention", "conv"),
                        help="Explicit inference-only zero-scale ablation")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output", default="research/results/pretrained_lfm_development")
    run(parser.parse_args())
