#!/usr/bin/env python3
"""Freeze a post-training, machine-checkable 100-task holdout, with overlap audit."""

import argparse
import hashlib
import json
from pathlib import Path

from research.lfm.build_native_pilot_data import ngrams, normalized
from research.lfm.evaluate_pretrained_development import validate_suite


def build_tasks():
    tasks = []
    for i in range(20):
        label = ["cedar", "birch", "maple", "oak", "pine", "elm", "ash", "fir",
                 "willow", "spruce", "beech", "alder", "hazel", "yew", "larch",
                 "poplar", "cypress", "hemlock", "juniper", "holly"][i]
        expected = {"label": label, "enabled": i % 2 == 0}
        tasks.append({"id": f"record-{i:02d}", "category": "json_record",
            "prompt": f"Encode this setting as JSON: label is the text {label}; enabled is the boolean {'true' if expected['enabled'] else 'false'}. Include these two fields and nothing else. Your entire reply must be the bare JSON object, not a code block.",
            "validator": "json_exact", "content_validator": "json_unfenced",
            "expected": expected})
        values = [-(31 + i), 53 - i, i + 2, -2 * i - 1]
        tasks.append({"id": f"sort-{i:02d}", "category": "signed_sorting",
            "prompt": f"Place the numbers {values[1]}, {values[0]}, {values[2]}, {values[3]} in increasing numerical order. Respond with a bare JSON array of integers. Include neither commentary nor code fences.",
            "validator": "json_exact", "content_validator": "json_unfenced",
            "expected": sorted(values)})
        a, b, c = 11 + i, 4 + i % 7, 13 + i
        tasks.append({"id": f"arithmetic-{i:02d}", "category": "arithmetic",
            "prompt": f"A warehouse receives {a} cartons containing {b} bolts apiece, then ships {c} bolts. Calculate the remaining bolt count. Supply just its decimal integer, without any explanation.",
            "validator": "text_exact", "content_validator": "boxed_integer",
            "expected": str(a * b - c)})
        # Rotate the requested end of a transitive ordering to avoid constant-position labels.
        a, b, c = f"R{i+21}", f"S{i+43}", f"T{i+65}"
        requested = "earliest" if i % 2 else "latest"
        tasks.append({"id": f"order-{i:02d}", "category": "transitive_ordering",
            "prompt": f"Event {b} happened after {a} but before {c}. Which of those three happened {requested}? Reply solely with its event identifier.",
            "validator": "text_exact", "expected": a if i % 2 else c})
        english = ["zero", "one", "two", "three", "four", "five", "six", "seven",
                   "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
                   "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"][i]
        spanish = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete",
                   "ocho", "nueve", "diez", "once", "doce", "trece", "catorce",
                   "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve"][i]
        tasks.append({"id": f"translation-{i:02d}", "category": "spanish_number_words",
            "prompt": f"Give the Spanish dictionary form of the English number word '{english}'. Write that word alone in lowercase, with correct accents and no punctuation.",
            "validator": "text_exact", "expected": spanish})
    return {"version": 1, "role": "frozen_post_training_holdout",
        "scope": "100 new prompts in five correlated task families; not a general capability benchmark",
        "tasks": tasks}


def freeze(output: Path, data: Path):
    output.mkdir(parents=True, exist_ok=False)
    suite = build_tasks()
    validate_suite(suite)
    prior_paths = [Path("research/lfm/pretrained_development_tasks_v2.json"),
                   Path("research/lfm/control_prompts.json"),
                   data / "unique.json", data / "repeated.json"]
    prior_prompts, sources = [], []
    for path in prior_paths:
        raw = path.read_bytes()
        value = json.loads(raw)
        rows = value if isinstance(value, list) else value.get("tasks", value.get("prompts", []))
        prior_prompts.extend(row.get("prompt", row.get("text")) for row in rows)
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    exact = {normalized(prompt) for prompt in prior_prompts}
    prior_grams = [ngrams(prompt) for prompt in prior_prompts]
    overlaps = []
    maximum = 0.
    for task in suite["tasks"]:
        grams = ngrams(task["prompt"])
        similarity = max(len(grams & g) / max(1, len(grams | g)) for g in prior_grams)
        maximum = max(maximum, similarity)
        if normalized(task["prompt"]) in exact or similarity >= .8:
            overlaps.append(task["id"])
    if overlaps:
        raise ValueError(f"Holdout overlaps registered prompts: {overlaps}")
    raw = (json.dumps(suite, indent=2, ensure_ascii=False) + "\n").encode()
    (output / "tasks.json").write_bytes(raw)
    audit = {"tasks": 100, "categories": 5, "tasks_per_category": 20,
        "tasks_sha256": hashlib.sha256(raw).hexdigest(), "prior_sources": sources,
        "normalized_exact_or_5gram_jaccard_08_overlaps": overlaps,
        "max_prior_5gram_jaccard": maximum,
        "caveat": "Lexical overlap audit does not prove semantic independence or upstream pretraining decontamination"}
    (output / "freeze.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-data", required=True)
    args = parser.parse_args()
    freeze(Path(args.output), Path(args.training_data))
