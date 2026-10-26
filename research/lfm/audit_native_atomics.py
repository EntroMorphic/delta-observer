#!/usr/bin/env python3
"""Native-model token/activation audit and exact inference-time LoRA ablations."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from research.lfm.evaluate_pretrained_development import chat_inputs, content_validation, validate
from research.lfm.native_lora import LoRALinear, install_lora, load_adapter, supervised_chat
from research.lfm.native_lora_smoke import MODEL, REVISION
from research.lfm.probe_lfm import choose_device


def answer(task):
    expected = task["expected"]
    return json.dumps(expected, separators=(",", ":"), ensure_ascii=False) if isinstance(expected, (dict, list)) else expected


def first_difference(first, second):
    for index, (a, b) in enumerate(zip(first, second)):
        if a != b:
            return index
    return min(len(first), len(second)) if len(first) != len(second) else None


def token_metrics(logits, labels, tokenizer):
    logits = logits.float()
    log_probs = logits.log_softmax(-1)
    probs = log_probs.exp()
    best = logits.argmax(-1)
    target_logits = logits.gather(-1, labels[:, None]).squeeze(-1)
    target_lp = log_probs.gather(-1, labels[:, None]).squeeze(-1)
    alternatives = logits.clone()
    alternatives.scatter_(-1, labels[:, None], float("-inf"))
    margin = target_logits - alternatives.max(-1).values
    ranks = 1 + (logits > target_logits[:, None]).sum(-1)
    support = probs.topk(64, dim=-1).values.sum(-1)
    entropy = -(probs * log_probs).sum(-1)
    rows = []
    for offset in range(len(labels)):
        target, predicted = int(labels[offset]), int(best[offset])
        rows.append({"offset": offset, "target_id": target,
            "target_piece": tokenizer.decode([target]), "predicted_id": predicted,
            "predicted_piece": tokenizer.decode([predicted]),
            "target_nll": float(-target_lp[offset]), "target_probability": float(target_lp[offset].exp()),
            "target_rank": int(ranks[offset]), "target_vs_best_other_logit_margin": float(margin[offset]),
            "entropy_nats": float(entropy[offset]), "top64_mass": float(support[offset])})
    return rows


def reference_pass(model, tokenizer, task, device):
    row = supervised_chat(tokenizer, task["prompt"], answer(task), 256)
    ids = torch.tensor([row["input_ids"]], device=device)
    start = row["prompt_tokens"] - 1
    with torch.inference_mode():
        logits = model(ids[:, :-1], use_cache=False).logits[0, start:]
    labels = ids[0, row["prompt_tokens"]:]
    return row, logits, token_metrics(logits, labels, tokenizer)


def generate(model, tokenizer, task, device):
    inputs = chat_inputs(tokenizer, task["prompt"]).to(device)
    with torch.inference_mode():
        ids = model.generate(**inputs, generation_config=GenerationConfig(
            do_sample=False, num_beams=1, repetition_penalty=1., max_new_tokens=192,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        ))[0, inputs["input_ids"].shape[1]:].tolist()
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return {"generated_token_ids": ids, "response": text,
            "strict": validate(task, text)["pass"],
            "content_parser": content_validation(task, text)["pass"]}


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    suite_path = Path("research/results/native_holdout100_frozen/tasks.json")
    tasks = json.loads(suite_path.read_text())["tasks"]
    selected_ids = {"record-00", "record-07", "record-08", "record-15", "record-17", "record-19"}
    selected = [task for task in tasks if task["id"] in selected_ids]
    precision_ids = selected_ids | {"sort-00", "arithmetic-03", "order-00", "translation-00", "translation-04"}
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float16).to(device).eval()
    result = {"model": MODEL, "revision": REVISION,
        "tasks_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        "scope": "canonical-answer teacher forcing plus selected inference ablations; no training",
        "canonical_answer_caveat": "Canonical bare JSON and whole reference tokenization are diagnostic paths, not all acceptable answers or free-generation behavior",
        "device": device, "selected_case_ids": sorted(selected_ids), "arms": {}}
    base_logits = {}
    started = time.perf_counter()
    for arm in ("baseline", "repeated", "unique"):
        if arm != "baseline":
            checkpoint_path = Path(args.checkpoint_root) / f"native-lfm-pilot-{arm}/checkpoint.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if arm == "repeated":
                install_lora(model, checkpoint["config"]["rank"], checkpoint["config"]["alpha"])
            load_adapter(model, checkpoint["adapter"])
        records = []
        for task in tasks:
            row, logits, metrics = reference_pass(model, tokenizer, task, device)
            if arm == "baseline" and task["id"] in selected_ids:
                base_logits[task["id"]] = logits.cpu()
            records.append({"id": task["id"], "category": task["category"],
                "canonical_response": answer(task), "input_ids": row["input_ids"],
                "response_start": row["prompt_tokens"], "tokens": metrics})
        result["arms"][arm] = {"teacher_forced": records}
        if arm != "baseline":
            result["arms"][arm]["checkpoint_sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            modules = {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}
            activation = {}
            for task in selected:
                measurements = {}
                hooks = []
                for name, module in modules.items():
                    def hook(mod, inputs, outputs, key=name):
                        x = inputs[0]
                        delta = ((x.float() @ mod.lora_a.T) @ mod.lora_b.T * mod.scale).to(outputs.dtype)
                        base = mod.base(x)
                        positions = [len(chat_inputs(tokenizer, task["prompt"])["input_ids"][0]) - 1]
                        measurements[key] = {"prompt_boundary_delta_norm": float(delta[0, positions[0]].float().norm()),
                            "prompt_boundary_base_norm": float(base[0, positions[0]].float().norm()),
                            "all_position_delta_rms": float(delta.float().square().mean().sqrt()),
                            "all_position_base_rms": float(base.float().square().mean().sqrt())}
                    hooks.append(module.register_forward_hook(hook))
                try:
                    reference_pass(model, tokenizer, task, device)
                finally:
                    for handle in hooks:
                        handle.remove()
                activation[task["id"]] = measurements
            result["arms"][arm]["activation_update_magnitudes"] = activation
            interventions = {"full": [], "all_adapters_off": list(modules),
                "attention_off": [name for name in modules if ".self_attn." in name],
                "conv_off": [name for name in modules if ".conv." in name]}
            for depth in range(16):
                interventions[f"layer_{depth:02d}_off"] = [name for name in modules if f".layers.{depth}." in name]
            ablations = []
            for label, disabled in interventions.items():
                scales = {name: mod.scale for name, mod in modules.items()}
                for name in disabled:
                    modules[name].scale = 0.
                try:
                    cases = []
                    for task in selected:
                        row, logits, metrics = reference_pass(model, tokenizer, task, device)
                        record = {"id": task["id"], "reference_tokens": metrics,
                            **generate(model, tokenizer, task, device)}
                        if label == "all_adapters_off":
                            record["baseline_logit_max_error"] = float((logits.cpu() - base_logits[task["id"]]).abs().max())
                        canonical = row["input_ids"][row["prompt_tokens"]:]
                        record["first_canonical_token_difference"] = first_difference(canonical, record["generated_token_ids"])
                        cases.append(record)
                    ablations.append({"intervention": label, "disabled_modules": disabled, "cases": cases})
                    print(arm, label, [(case["id"], case["strict"]) for case in cases], flush=True)
                finally:
                    for name, scale in scales.items():
                        modules[name].scale = scale
            result["arms"][arm]["ablations"] = ablations
        (output / "atomics.json").write_text(json.dumps(result, indent=2))
    # Reload original weights at FP32: casting quantized FP16 weights to FP32
    # would not test the weight-precision alternative.
    model.cpu()
    del model
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32).to(device).eval()
    for arm in ("baseline", "repeated", "unique"):
        if arm != "baseline":
            checkpoint_path = Path(args.checkpoint_root) / f"native-lfm-pilot-{arm}/checkpoint.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if arm == "repeated":
                install_lora(model, checkpoint["config"]["rank"], checkpoint["config"]["alpha"])
            load_adapter(model, checkpoint["adapter"])
        result["arms"][arm]["pristine_fp32_cases"] = [
            {"id": task["id"], **generate(model, tokenizer, task, device)}
            for task in tasks if task["id"] in precision_ids]
    result["elapsed_seconds"] = time.perf_counter() - started
    result["complete"] = True
    (output / "atomics.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    run(parser.parse_args())
