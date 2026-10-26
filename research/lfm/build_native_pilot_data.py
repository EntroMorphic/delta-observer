#!/usr/bin/env python3
"""Build deterministic, length-matched synthetic instruction adaptation arms."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from research.lfm.native_lora import supervised_chat
from research.lfm.native_lora_smoke import MODEL, REVISION


def normalized(text):
    return " ".join(text.casefold().split())


def ngrams(text):
    words = normalized(text).split()
    return {tuple(words[i:i + 5]) for i in range(max(1, len(words) - 4))}


ITEMS = ("lantern", "mug", "notebook", "ribbon", "basket", "kettle", "scarf", "clock")


def select_balanced_pairs(buckets, items=ITEMS):
    """Require two length-matched groups per item; never fill from one bucket."""
    unique, repeated = [], []
    for item in items:
        selected = []
        for key in sorted(buckets):
            if key[0] != item:
                continue
            bucket = buckets[key]
            for start in range(0, len(bucket) - 7, 8):
                selected.append(bucket[start:start + 8])
                if len(selected) == 2:
                    break
            if len(selected) == 2:
                break
        if len(selected) != 2:
            raise ValueError(f"Insufficient length-matched coverage for {item}")
        for group in selected:
            unique.extend(group)
            repeated.extend([group[0]] * 8)
    if len({normalized(row['prompt']) for row in unique}) != 16 * len(items):
        raise ValueError("Unique arm contains duplicate prompts")
    return unique, repeated


def build(output: Path, exclusions: list[Path]):
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    excluded, exclusion_files = [], []
    for path in exclusions:
        content = path.read_bytes()
        suite = json.loads(content)
        excluded.extend(task.get("prompt", task.get("text"))
                        for task in suite.get("tasks", suite.get("prompts", [])))
        exclusion_files.append({"path": str(path), "sha256": hashlib.sha256(content).hexdigest()})
    exact = {normalized(text) for text in excluded}
    near = [ngrams(text) for text in excluded]
    buckets = defaultdict(list)
    # Human-specified deterministic labels, not self-generated model targets.
    for index in range(4096):
        item = ITEMS[index % len(ITEMS)]
        quantity = 10 + index % 79
        serial = 1000 + index
        prompt = (f"Inventory note {serial}: the item is {item}; its quantity is {quantity}. "
                  "Extract the note as a JSON object with exactly the keys item and quantity. "
                  "Use a string for item and an integer for quantity. Output the object alone, without code fences.")
        grams = ngrams(prompt)
        if normalized(prompt) in exact or any(
            len(grams & other) / max(1, len(grams | other)) >= .8 for other in near
        ):
            continue
        response = json.dumps({"item": item, "quantity": quantity}, separators=(",", ":"))
        encoded = supervised_chat(tokenizer, prompt, response, 128)
        key = (item, len(encoded["input_ids"]), encoded["supervised_tokens"])
        buckets[key].append({"prompt": prompt, "response": response, **encoded})
    unique, repeated = select_balanced_pairs(buckets)
    arms = {}
    for name, rows in (("unique", unique), ("repeated", repeated)):
        data = (json.dumps(rows, indent=2) + "\n").encode()
        path = output / f"{name}.json"
        path.write_bytes(data)
        arms[name] = {
            "path": path.name, "sha256": hashlib.sha256(data).hexdigest(),
            "examples": len(rows), "unique_prompts": len({row["prompt"] for row in rows}),
            "supervised_tokens": sum(row["supervised_tokens"] for row in rows),
            "input_tokens": sum(len(row["input_ids"]) for row in rows),
            "item_exposure_counts": dict(Counter(json.loads(row['response'])['item'] for row in rows)),
            "distinct_item_values": len({json.loads(row['response'])['item'] for row in rows}),
            "prompt_hashes": [hashlib.sha256(normalized(row["prompt"]).encode()).hexdigest() for row in rows],
        }
    assert all((len(a["input_ids"]), a["supervised_tokens"]) ==
               (len(b["input_ids"]), b["supervised_tokens"])
               for a, b in zip(unique, repeated))
    manifest = {"builder_version": 2, "model": MODEL, "revision": REVISION,
        "scope": "exploratory synthetic inventory extraction; not broad dialogue diversity",
        "arms": arms, "exclusions": exclusion_files,
        "near_duplicate_rule": "normalized word 5-gram Jaccard >= 0.8",
        "near_duplicate_limit": "lexical screen only, not proof of semantic disjointness or pretraining decontamination",
        "per_position_token_lengths_match": True, "truncation": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: {key: value for key, value in v.items() if key != "prompt_hashes"}
                      for k, v in arms.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude", nargs="+", default=[
        "research/lfm/pretrained_development_tasks_v2.json", "research/lfm/control_prompts.json"])
    args = parser.parse_args()
    build(Path(args.output), [Path(path) for path in args.exclude])
