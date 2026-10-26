#!/usr/bin/env python3
"""Build transparent templated development tasks, never a sealed benchmark."""

import argparse
import json
from pathlib import Path

from research.lfm.evaluate_pretrained_development import validate_suite


def build() -> dict:
    source = json.loads(Path("research/lfm/pretrained_development_tasks.json").read_text())
    tasks = source["tasks"]
    for task in tasks:
        if task["validator"] == "json_exact":
            task["content_validator"] = "json_unfenced"
        if task["id"] == "arithmetic-boxes":
            task["content_validator"] = "boxed_integer"
    for index in range(6):
        values = [13 + index, -8 - index, 2 * index, 30 - index]
        tasks.append({
            "id": f"sort-{index}", "category": "format_following",
            "prompt": f"Sort these integers in ascending order: {', '.join(map(str, values))}. Return only a JSON array, with no Markdown or explanation.",
            "validator": "json_exact", "content_validator": "json_unfenced",
            "expected": sorted(values),
        })
        city = ["Lima", "Kyoto", "Accra", "Riga", "Bern", "Perth"][index]
        tasks.append({
            "id": f"record-{index}", "category": "format_following",
            "prompt": f"Return only a JSON object with exactly two keys: city with string value {city}, and count with integer value {index + 2}. No Markdown or other text.",
            "validator": "json_exact", "content_validator": "json_unfenced",
            "expected": {"city": city, "count": index + 2},
        })
        boxes, pencils, removed = index + 3, index + 4, index + 2
        tasks.append({
            "id": f"arithmetic-{index}", "category": "deduction_arithmetic",
            "prompt": f"There are {boxes} boxes with {pencils} pencils in each. If {removed} pencils are removed, how many remain? Return only the integer.",
            "validator": "text_exact", "content_validator": "boxed_integer",
            "expected": str(boxes * pencils - removed),
        })
        colors = [("red", "green", "white"), ("gold", "black", "pink"),
                  ("brown", "orange", "gray"), ("cyan", "beige", "blue"),
                  ("yellow", "purple", "silver"), ("maroon", "navy", "teal")][index]
        a, b, c = colors
        tasks.append({
            "id": f"deduction-{index}", "category": "deduction_arithmetic",
            "prompt": f"The {a} box is heavier than the {b} box. The {c} box is heavier than the {a} box. Which box is heaviest? Return only its color in lowercase.",
            "validator": "text_exact", "expected": c,
        })
        english, spanish = [("dog", "perro"), ("house", "casa"), ("book", "libro"),
                            ("milk", "leche"), ("bread", "pan"), ("sun", "sol")][index]
        tasks.append({
            "id": f"spanish-{index}", "category": "multilingual_instructions",
            "prompt": f"Translate the English word '{english}' into Spanish. Return only the lowercase singular word.",
            "validator": "text_exact", "expected": spanish,
        })
    suite = {"version": 2, "role": "development_only_not_confirmatory",
             "design_note": "44 tasks; templated families are correlated, not 44 independent capability measurements. Nine manual tasks still await review. No confirmatory aggregate.",
             "tasks": tasks}
    validate_suite(suite)
    return suite


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.output).open("x") as output:
        json.dump(build(), output, indent=2, ensure_ascii=False)
