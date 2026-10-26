#!/usr/bin/env python3
"""Counterfactual copying and schema binding controls on twenty record values."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from research.lfm.evaluate_pretrained_development import strict_json
from research.lfm.native_lora import install_lora, load_adapter
from research.lfm.native_lora_smoke import MODEL, REVISION
from research.lfm.probe_lfm import choose_device


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    tasks = json.loads(Path("research/results/native_holdout100_frozen/tasks.json").read_text())["tasks"]
    records = [task for task in tasks if task["category"] == "json_record"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    device = choose_device(args.device)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float16).to(device).eval()
    result = {"scope": "diagnostic counterfactuals, not new held-out benchmark", "arms": {}}
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
        cases = []
        for task in records:
            value, flag = task["expected"]["label"], task["expected"]["enabled"]
            modes = {
                "plain_echo": (f"Copy this word exactly, with no other text: {value}", "", None),
                "quoted_value": (task["prompt"].replace(f"text {value};", f'text "{value}";'), '{"label": "', 'label'),
                "renamed_item": (task["prompt"].replace('label', 'item'), '{"item": "', 'item'),
                "renamed_name": (task["prompt"].replace('label', 'name'), '{"name": "', 'name'),
            }
            for mode, (prompt, prefix, key) in modes.items():
                rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
                ids = tokenizer.encode(rendered, add_special_tokens=False)
                prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                inputs = torch.tensor([ids + prefix_ids], device=device)
                with torch.inference_mode():
                    generated = model.generate(inputs, generation_config=config)[0, inputs.shape[1]:].tolist()
                response = tokenizer.decode(prefix_ids + generated, skip_special_tokens=True)
                if mode == "plain_echo":
                    correct = response.strip() == value
                else:
                    try:
                        parsed = strict_json(response)
                        correct = (type(parsed) is dict and set(parsed) == {key, 'enabled'}
                            and parsed[key] == value and type(parsed['enabled']) is bool
                            and parsed['enabled'] == flag)
                    except ValueError:
                        correct = False
                cases.append({"id": task['id'], "mode": mode, "prompt": prompt,
                    "forced_prefix": prefix, "response": response, "correct": correct,
                    "completion_ids": generated})
        result['arms'][arm] = cases
        print(arm, {mode:sum(c['correct'] for c in cases if c['mode']==mode)
                    for mode in ('plain_echo','quoted_value','renamed_item','renamed_name')},flush=True)
        (output/'copy_binding.json').write_text(json.dumps(result,indent=2))
    result['complete'] = True
    (output/'copy_binding.json').write_text(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--checkpoint-root',required=True)
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    run(parser.parse_args())
