#!/usr/bin/env python3
"""Causal syntax-prefix interventions without supplying any correct field value."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from research.lfm.evaluate_pretrained_development import content_validation, validate
from research.lfm.native_lora import install_lora, load_adapter
from research.lfm.native_lora_smoke import MODEL, REVISION
from research.lfm.probe_lfm import choose_device


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    tasks = json.loads(Path("research/results/native_holdout100_frozen/tasks.json").read_text())["tasks"]
    tasks = [task for task in tasks if task["category"] == "json_record"]
    prefixes = {"bare_compact": '{"label":"', "bare_spaced": '{"label": "',
        "bare_pretty": '{\n  "label": "', "fenced_pretty": '```json\n{\n  "label": "',
        "fenced_compact": '```json\n{"label":"'}
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float16).to(device).eval()
    result = {"scope": "forced syntax prefix causal diagnostic; not ordinary task success or deployment fix",
        "prefixes_supply_no_field_values": True, "arms": {}}
    config = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.,
        max_new_tokens=128, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id, use_cache=True)
    for arm in ("baseline", "repeated", "unique"):
        if arm != "baseline":
            path = Path(args.checkpoint_root) / f"native-lfm-pilot-{arm}/checkpoint.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if arm == "repeated":
                install_lora(model, checkpoint["config"]["rank"], checkpoint["config"]["alpha"])
            load_adapter(model, checkpoint["adapter"])
        result["arms"][arm] = {}
        for name, prefix in prefixes.items():
            rows = []
            for task in tasks:
                rendered = tokenizer.apply_chat_template([{"role": "user", "content": task["prompt"]}],
                    add_generation_prompt=True, tokenize=False)
                prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
                prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                inputs = torch.tensor([prompt_ids + prefix_ids], device=device)
                with torch.inference_mode():
                    generated = model.generate(inputs, generation_config=config)[0, inputs.shape[1]:].tolist()
                # Decode the forced-prefix tokens together with completion tokens.
                response = tokenizer.decode(prefix_ids + generated, skip_special_tokens=True)
                rows.append({"id": task["id"], "response": response,
                    "forced_prefix_ids": prefix_ids, "completion_ids": generated,
                    "strict": validate(task, response)["pass"],
                    "content": content_validation(task, response)["pass"]})
            result["arms"][arm][name] = rows
            print(arm, name, 'content',sum(row['content'] is True for row in rows),
                  'strict',sum(row['strict'] is True for row in rows), flush=True)
            (output / "prefixes.json").write_text(json.dumps(result, indent=2))
    result["complete"] = True
    (output / "prefixes.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    run(parser.parse_args())
